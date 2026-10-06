r"""The SSD RESTORE path: read a spilled span back and extend the tier's own load.

`disk_restore.py` is the half of the disk tier that reads. The spill half writes a
file and journals its identity; this module turns that journal into tokens a request
is served without recomputing, by EXTENDING the fork's own host-RAM load - so no new
transfer machinery exists and every refcount stays the fork's. These tests are ordered
by what breaks if the module gets it wrong, because the failure modes here are not
crashes: they are a request served from the wrong KV.

1. SILENT STALE KV IS THE FIRST HAZARD. A restore allocates CPU slots and tells the
   scheduler those tokens are already computed. If nobody fills those slots, the
   fork's own `start_load_kv` DMAs whatever was left in them. So nothing is offered
   unless a worker has PROVEN its half is fitted (`worker_proof`); a read that has not
   settled is PRUNED from the step rather than awaited; and any entry whose bytes
   never arrived has its GPU block reported as a load error, so the scheduler
   recomputes instead of trusting it.
2. AN INDEX THAT LIES IS WORSE THAN NO INDEX. `load_paths` is one entry per load
   entry, always, in the fork's order, "" where nothing is on disk; the worker
   re-derives the path from the identity rather than trusting the string, and a
   disagreement is counted and the entry dropped.
3. THE MERGE MUST LOOK EXACTLY LIKE THE FORK'S OWN TUPLE, because
   `update_state_after_alloc` consumes LEADING entries per group and frees the rest.
   Our slots go at the END; a RAM hit whose per-group length disagrees with its own
   token count is REFUSED outright.
4. THE PHANTOM REPORT. A load event that was never submitted must not stay in
   `_pending_load_event_indices`: the completion watermark is monotone, so a later
   completion would report it as "loaded". Removal and reporting happen in one call,
   and `_phantom_guard` re-checks and counts any trip.
5. NOTHING FIRES UNARMED, and the fork's return value passes through unchanged on
   every refusal path - an unarmed restore is indistinguishable from no plugin.
6. EVERY FAILURE IS LOUD. Boot line, witness line, counters, rank file: a restore
   that never happened has to be provable, not inferred.

Stdlib plus the plugin; `tmp_path` for every file; real `ctypes` memory for the CPU
slots, because the copy under test IS a `memmove`; the reader thread is `disk_read`'s
own and settles on a BOUNDED poll, never an unbounded wait.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import sys
import threading
import time
import types

import fakes_debug
import pytest

from kvtier_dev_plugin import (
    backport,
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_restore,
    disk_spill,
    disk_store,
    engine_side,
    guard_mode,
    snapshot,
    worker_diag,
)

BLOCK = 3200                     # tokens in one scheduler block (the live box)
HASH_BLOCK = 128                 # tokens per engine block hash
STEP = BLOCK // HASH_BLOCK       # hashes inside one scheduler block
GROUP_SIZES = (BLOCK, BLOCK, 1600, 128, BLOCK)
CACHEABLE = (0, 1, 4)            # prefix-cacheable; 2 is small-blocked, 3 is scratch
WIDTH = 4096                     # bytes in one rank's CPU slot (small on purpose)
SLOTS = 8                        # slots in the fake CPU region tensor
DIGEST = "0a1b2c3d4e5f"           # THIS rank's geometry digest (12 hex)
PLACEHOLDER = disk_restore.IDENTITY_DIGEST_PLACEHOLDER


# --- the engine's own key machinery, reimplemented honestly ------------------

def pack(digest, group_id):
    """`kv_cache_utils.py:83`: the digest, then a 4-byte big-endian group id."""
    return bytes(digest) + int(group_id).to_bytes(4, "big")


def resolve(block_hashes, hash_block_size, block_size):
    """One hash per whole `block_size` block: the LAST one inside each."""
    step = int(block_size) // int(hash_block_size)
    if step <= 0:
        raise ValueError("block_size must be a multiple of hash_block_size")
    return [block_hashes[start + step - 1]
            for start in range(0, len(block_hashes) - step + 1, step)]


def raw_digest(seed: int) -> bytes:
    """A 32-byte block hash, the shape a `BlockHash` has."""
    return bytes([(seed * 37 + i) % 251 for i in range(32)])


def view_digest(seed: int) -> bytes:
    """The digest the engine's block-granular view puts at index `seed - 1`."""
    return raw_digest(seed * STEP - 1)


def key_hex(seed: int, group: int) -> str:
    """The identity a spilled file for block `seed`, group `group`, is named for."""
    return pack(view_digest(seed), group).hex()


# --- fakes carrying the fork's attribute names -------------------------------

class Slot:
    """A `KVCacheBlock`: the pool hands these out and only `block_id` matters here.

    `is_null` exists because the fork's own `_free_pending_cpu_hit` filters on it, and
    a fake without the attribute would make the pin-release path look different from
    what it is.
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.is_null = False


class CpuPool:
    """`cpu_block_pool`, with the real pool's one sharp edge reproduced.

    `get_new_blocks` RAISES when the count exceeds the free list
    (`core/block_pool.py:664-676`). The offer path pre-checks the free count only
    because of that, so a permissive fake would be asserting nothing.
    """

    def __init__(self, num_blocks=1024, free_ids=None):
        self.blocks = [Slot(i) for i in range(num_blocks)]
        self._free = [self.blocks[i] for i in (free_ids or range(num_blocks))]
        self.allocated = []
        self.freed = []
        # `BlockPool.null_block` (`block_pool.py:190-194`): a placeholder the fork
        # marks `is_null` and NEVER frees or allocates, used for the skipped entries
        # of an align-mode Mamba hit. An align-mode disk restore appends exactly those
        # objects to a group's list, because `update_state_after_alloc` skips entries
        # whose `is_null` is set while KEEPING THE INDEX - which is what lands the one
        # real state block on the run's LAST GPU block. It is numbered past the
        # allocatable range so a test can tell a null from an allocation, and a pool
        # without one cannot serve a boundary group at all.
        self.null_block = Slot(num_blocks)
        self.null_block.is_null = True
        self.blocks.append(self.null_block)

    def get_num_free_blocks(self):
        return len(self._free)

    def get_new_blocks(self, count):
        if count > len(self._free):
            raise ValueError(f"Cannot get {count} blocks from the queue with only "
                             f"{len(self._free)} free blocks.")
        taken, self._free = self._free[:count], self._free[count:]
        self.allocated.extend(taken)
        return taken

    def free_blocks(self, blocks):
        self.freed.append([int(block.block_id) for block in blocks])
        self._free = list(blocks) + self._free


class Manager:
    """`SimpleCPUOffloadScheduler`'s fields the restore path reads, by name."""

    def __init__(self, *, block_size=BLOCK, group_sizes=GROUP_SIZES,
                 cacheable=CACHEABLE, pool=None, stable=True, reqs_to_load=None,
                 no_coordinator=False):
        self.block_size = block_size
        self.hash_block_size = HASH_BLOCK
        self.group_block_sizes = list(group_sizes)
        self.prefix_cacheable_group_ids = list(cacheable)
        self.cpu_block_pool = pool if pool is not None else CpuPool()
        self.cpu_coordinator = None if no_coordinator else fakes_debug.coordinator([
            fakes_debug.manager_item(group,
                                     fakes_debug.attention_spec(
                                         block_size=group_sizes[group]),
                                     stable=stable)
            for group in range(len(group_sizes))])
        self._pending_cpu_hits = {}
        self._reqs_to_load = dict(reqs_to_load or {})

    # The fork's own three entry points, answering the way the fork does when it
    # finds nothing of its own: the wrapper is tested against THIS, not a mock.
    def get_num_new_matched_tokens(self, request, num_computed_tokens):  # noqa: ARG002,E501
        return 0, False

    def build_connector_meta(self, scheduler_output):  # noqa: ARG002
        return "METADATA"

    def update_connector_output(self, connector_output):  # noqa: ARG002
        return None

    def _free_pending_cpu_hit(self, pending):
        """`manager.py:1255-1264`, the fork's ONLY release of a lookup pin.

        The real method frees every non-null block in the tuple's per-group lists
        through `cpu_block_pool.free_blocks`. It has to be answered here because a
        flagged request that hands the disk the authority calls it, and a fake that
        simply lacked the method would make the disk-authority path look like a
        refusal forever - hiding the exact leak (a pinned CPU block per flagged
        request, for the life of the boot) the call exists to prevent.
        """
        blocks = [block for group in pending[0] for block in group
                  if not getattr(block, "is_null", False)]
        if blocks:
            self.cpu_block_pool.free_blocks(blocks)


class Req:
    """`Request`: the fields the offer path reads, and nothing else.

    Deliberately NO `kv_transfer_params` attribute: since 0.2.31 there is no client
    marker, and a fake that cannot supply one is the proof the offer path never reads
    it - a leftover `request.kv_transfer_params` would raise AttributeError here rather
    than pass silently.
    """

    def __init__(self, request_id="req-1", blocks=4, seed=1, num_tokens=None,
                 skip_reading_prefix_cache=False):
        self.request_id = request_id
        # The digest at the END of each STEP-run is what the engine's view keeps for
        # that scheduler block, so block ordinal b (0-based) carries view_digest(b+1)
        # and  names exactly those files.
        self.block_hashes = [raw_digest((i // STEP + 1) * STEP - 1)
                             for i in range(blocks * STEP)]
        self.num_tokens = (num_tokens if num_tokens is not None
                           else blocks * BLOCK + 1200)
        self.skip_reading_prefix_cache = skip_reading_prefix_cache


class Meta:
    """`SimpleCPUOffloadMetadata`, field-for-field on the load side."""

    def __init__(self, load_event=-1, cpu=None, gpu=None, reqs=None, paths=None):
        self.load_event = load_event
        self.load_cpu_blocks = list(cpu or [])
        self.load_gpu_blocks = list(gpu or [])
        self.load_paths = list(paths if paths is not None
                               else [""] * len(cpu or []))
        self.load_event_to_reqs = dict(reqs or {})
        self.store_event = -1
        self.store_gpu_blocks = []
        self.store_cpu_blocks = []
        self.store_keys = []
        self.need_flush = False


class Region:
    """One region tensor of `cpu_kv_caches`, backed by REAL memory.

    The copy under test is a `ctypes.memmove` into `data_ptr()`, so this cannot be a
    dict of bytearrays: it has to answer `data_ptr`/`stride`/`element_size`/`shape`/
    `nbytes` the way a torch tensor does and hand out a real address. The buffer lives
    on the instance so the memory outlives the copy.
    """

    def __init__(self, slots=SLOTS, width=WIDTH):
        self.buffer = (ctypes.c_char * (slots * width))()
        self._slots = slots
        self._width = width

    def data_ptr(self):
        return ctypes.addressof(self.buffer)

    def stride(self, axis):
        return self._width if axis == 0 else 1

    def element_size(self):
        return 1

    @property
    def shape(self):
        return (self._slots,)

    @property
    def nbytes(self):
        return self._slots * self._width

    def slot_bytes(self, slot):
        return ctypes.string_at(self.data_ptr() + slot * self._width, self._width)


class Worker:
    """`SimpleCPUOffloadWorker`, the fields the restore path touches."""

    def __init__(self, regions=None):
        self.cpu_kv_caches = {"g0": Region()} if regions is None else regions
        self._connector_metadata = None
        self._pending_load_event_indices = set()
        self._pending_store_event_indices = set()
        self.start_load_calls = 0

    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata

    def start_load_kv(self):
        self.start_load_calls += 1

    def get_finished(self, finished_req_ids):  # noqa: ARG002
        return set(), None


# --- fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh():
    """A fresh boot per test: no ledgers, no counters, no installed layer."""
    disk_restore.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    disk_read.reset_state_for_tests()
    yield
    disk_restore.uninstall()
    disk_restore.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    disk_read.reset_state_for_tests()


@pytest.fixture(autouse=True)
def allow_tmp_root(monkeypatch):
    """`tmp_path` lives under /tmp, which a real spill root must refuse.

    The rule belongs to `disk_layout` and is pinned in its own suite; here it would
    only prevent a test from naming a file at all.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())


@pytest.fixture
def armed(monkeypatch, tmp_path):
    """The gate open AND this boot told to offer: five switches, not four.

    The four fork switches are replaced with the value under test rather than driven
    through their own files, because arm-file and root-validation behaviour belongs
    to `disk_spill`/`config` and is pinned there. What this suite pins is the GATE'S
    OWN composition: all of them, or nothing.

    The fifth is this module's own switch, and a fresh boot leaves it DOWN: arming the
    spill tier is not consent to put a disk offer in front of a request. Every test
    that uses this fixture is a test ABOUT offering, so it says so here explicitly -
    `test_the_default_boot_offers_nothing_and_touches_no_index` pins the other side.
    """
    root = tmp_path / "spill"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(disk_spill, "geometry_digest", lambda: DIGEST)
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": True, "count": 1,
                                             "ranks": ["PP0_TP0"], "reason": None,
                                             "ranks_armed": 1, "ranks_seen": 1,
                                             "reads_applied": 0,
                                             "reason_kind": None})
    monkeypatch.setattr(disk_store, "counters", lambda: {
        "geometry_digest": DIGEST, "fingerprint": {"total_bytes_per_block": WIDTH},
        "rank": "PP0_TP0"})
    utils = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    utils.make_block_hash_with_group_id = pack
    utils.resolve_block_hashes = resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, utils)
    # THE SWITCHES, raised on purpose: see the docstring. `armed` raises DISK
    # AUTHORITY, which is the posture the suite was built around - the one where the
    # disk, not the request's host-RAM hit, decides the span. `armed_gap_fill` lowers it
    # and raises the cheaper switch instead; `switches_down` leaves both at the boot
    # default and is what pins that a fresh boot does nothing.
    disk_restore.set_disk_authority(True)
    return str(root)


@pytest.fixture
def armed_gap_fill(armed):
    """The gate open, and ONLY the gap-fill switch up: RAM first, the disk extends.

    This is the posture that keeps the PRE-0.2.22 meaning of a host-RAM hit alive. It is
    what `scope=all` plus an unflagged request used to pin, so every test about the
    merged tuple's RAM-leading shape - the fork's own blocks first, our slots after -
    runs here.
    """
    disk_restore.set_disk_authority(False)
    disk_restore.set_offer_switch(True)
    return armed


@pytest.fixture
def switches_down(armed):
    """The gate open and BOTH switches down: exactly what a fresh boot is.

    The offer path must return the fork's own answer and touch nothing - no journal
    refresh, no index query, no allocation, no CPU slot - which is the property the
    2026-10-05 incident (123,474 un-requested 160,000-token offers) made worth pinning
    harder than any positive test.
    """
    disk_restore.set_disk_authority(False)
    disk_restore.set_offer_switch(False)
    return armed


def fill(root, blocks=4, groups=CACHEABLE, seeds=None, write=True):
    """Journal (and optionally write) one file per (block, group) in the run."""
    seeds = list(seeds if seeds is not None else range(1, blocks + 1))
    for seed in seeds:
        for group in groups:
            made = disk_layout.build_slot_path(root, key_hex(seed, group), DIGEST)
            assert made["ok"], made["reason"]
            if write:
                os.makedirs(os.path.dirname(made["path"]), exist_ok=True)
                with open(made["path"], "wb") as handle:
                    handle.write(bytes([(seed * 11 + group) % 251] * WIDTH))
            appended = disk_index.journal_append(disk_index.SPILL_LINE,
                                                 key_hex(seed, group), WIDTH)
            assert appended["ok"], appended["reason"]
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    return seeds


def offer(manager, request, computed=0, ram=(0, False)):
    """One lookup through the module's real body."""
    return disk_restore.offer(manager, request, computed, ram)


def settle(batch_id, limit=400):
    """Wait for a batch to settle. BOUNDED: a stuck read is a failure, not a hang."""
    for _ in range(limit):
        status = disk_read.batch_status(batch_id)
        if not status.get("known") or (status.get("pending") == 0
                                       and status.get("running", 0) == 0):
            return status
        time.sleep(0.005)
    raise AssertionError(f"batch {batch_id!r} never settled: "
                         f"{disk_read.batch_status(batch_id)}")


def merged_cpu(manager, request_id="req-1"):
    """The CPU ids the merged tuple handed the fork, in the fork's group order."""
    return [int(block.block_id)
            for group in manager._pending_cpu_hits[request_id][0] for block in group]


def disk_meta(root, seeds=(1,), groups=CACHEABLE, ram=1, event=11,
              reqs=("req-1",), gap=False):
    """A metadata whose disk entries carry the scheduler's sentinel-digest paths.

    CPU ids are small because they ARE slot indices in the fake region tensor; GPU ids
    are large and distinct, so a misalignment between the two lists is visible rather
    than plausible.
    """
    cpu, gpu, paths = [], [], []
    for group in groups:
        for seed in seeds:
            made = disk_layout.build_slot_path(root, key_hex(seed, group),
                                               PLACEHOLDER)
            assert made["ok"], made["reason"]
            cpu.append(len(cpu))
            gpu.append(60_000 + len(gpu))
            paths.append(made["path"])
    if gap:
        cpu.insert(1, 400)
        gpu.insert(1, 45_000)
        paths.insert(1, "")
    for index in range(ram):
        cpu.append(410 + index)
        gpu.append(70_000 + index)
        paths.append("")
    return Meta(load_event=event, cpu=cpu, gpu=gpu, paths=paths,
                reqs={event: list(reqs)})


def vanish(root, seeds=(1,), groups=CACHEABLE):
    """Delete the files the WORKER would open (named by its own real digest).

    The scheduler publishes the digest sentinel, so deleting the published path would
    delete nothing: the file that has to disappear is the one built from the rank's
    actual fingerprint width.
    """
    for seed in seeds:
        for group in groups:
            made = disk_layout.build_slot_path(root, key_hex(seed, group), DIGEST)
            assert made["ok"], made["reason"]
            if os.path.exists(made["path"]):
                os.remove(made["path"])


def bind(worker, metadata):
    """The fork's bind, then our hook, exactly as the wrapper composes them."""
    worker.bind_connector_metadata(metadata)
    return disk_restore.worker_bind(worker, metadata)


# --- a: the gate, and the proof that a worker is listening -------------------

def test_gate_refuses_a_boot_with_no_root():
    """No `VLLM_E2_TIER_DISK_ROOT`, no restore: there is no store to read from."""
    verdict = disk_restore.gate()
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "not_configured"
    assert config.ENV_DISK_ROOT in verdict["reason"]


def test_gate_opens_only_when_every_switch_is_up(armed, monkeypatch):
    """Root AND the spill arm AND the tier arm AND a proven worker. Four, not three.

    Each is then pulled alone, and the gate must close naming THAT switch: a gate that
    closes for the wrong reason sends an operator to the wrong file at 3 a.m.
    """
    assert disk_restore.gate()["ok"] is True
    monkeypatch.setattr(disk_spill, "disk_root", lambda: None)
    assert disk_restore.gate()["reason_kind"] == "not_configured"
    monkeypatch.setattr(disk_spill, "disk_root", lambda: armed)
    monkeypatch.setattr(disk_spill, "is_armed", lambda: False)
    assert disk_restore.gate()["reason_kind"] == "disarmed"
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    assert disk_restore.gate()["reason_kind"] == "tier_disarmed"
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": False, "count": 0, "ranks": [],
                                             "reason": "no worker proved itself"})
    assert disk_restore.gate()["reason_kind"] == "no_worker_proof"


def test_worker_half_gate_needs_no_tier_arm(armed, monkeypatch):
    """A worker has no scheduler, so its gate is root + spill arm. Nothing else.

    The tier arm is a scheduler-side switch; requiring it here would make a worker
    refuse to read files its own scheduler has every right to offer.
    """
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    assert disk_restore.gate()["reason_kind"] == "tier_disarmed"
    assert disk_restore._may_restore_worker()["ok"] is True


def test_nothing_fires_when_the_tier_arm_is_down(armed, monkeypatch):
    """Pull the tier arm and no lookup can offer, allocate, or name a file.

    UPDATED FOR 0.2.21, and this is the one existing assertion the change made wrong.
    BEFORE: `sum(refused.values()) == 0` - a gate-closed marked request left no counter
    at all, only `last_reason`. That was the ambiguity that cost the 2026-10-05
    investigation: an empty `refused` was read as "nothing refused us" when the gate had
    refused the request. AFTER: the decline is counted under its own name, and the
    nothing-fired claim is made STRONGER rather than weaker - no offer, no allocation,
    no pending hit, the fork's own tuple back, and exactly one refusal, named.
    """
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    manager = Manager(reqs_to_load={"req-1": object()})
    fill(armed, blocks=2)
    assert offer(manager, Req(blocks=2)) == (0, False)
    assert manager.cpu_block_pool.allocated == []
    assert manager._pending_cpu_hits == {}
    counters = disk_restore.counters()
    assert counters["offers"] == 0
    moved = {name: count for name, count in counters["refused"].items() if count}
    assert moved == {"tier_disarmed": 1}, \
        "the gate's decline is counted, by the name the gate itself gave"


def test_worker_proof_reads_fitted_from_the_rank_file_and_arm_from_the_arm_file(
        tmp_path, monkeypatch):
    """The proof is TWO SOURCES (0.2.28): per-rank fit facts from the rank file,
    the tier ARM from the arm file - and each is proof only of what it uniquely
    knows.

    2026-10-05 proved `installed` alone is not enough: the scheduler offered
    160,000 tokens 123,474 times to workers that would not bind. So 0.2.26 read
    the rank file's `armed` too - and 2026-10-20 proved THAT was the wrong door
    for THIS fact: the arm is a rank-agnostic boot fact EngineCore owns and
    publishes as `disk-spill-arm.json`, while a rank file is rewritten on cadence
    with a signature that never hashed the armed verdict, so an arm with no spill
    activity stayed invisible and `workers_not_servicing` locked out workers that
    would serve (reproduced live: 0/4 armed for 120 s, then 4/4 seconds after a
    spill burst). Now the proof asks the arm file (via `disk_spill._read_arm_file`,
    boot fence intact) whether the TIER is armed, and the rank files only whether
    THIS rank is fitted and can name a root. A rank file's lagged `armed: false`
    under a live arm file no longer vetoes - it is counted as `stale_arm_claims`.

    Covers, in order: no arm file (refused by the arm-file kind, whatever the rank
    files claim), fitted-but-not-fitted (the real AND veto), the stale-armed pass
    that was the lockout, `not_configured` as the per-rank env fact it is, the
    freshness and unreadability edges, and the expiry the arm file owns.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setattr(config, "diagnostics_dir", lambda: directory)
    assert disk_restore.worker_proof(force=True)["ok"] is False

    def write(section, ts=None, rank="PP0_TP0"):
        document = {"version": 1, "epoch": "aabbccdd",
                    "ts": round(time.time(), 6) if ts is None else ts,
                    "role": "worker", "rank": rank, "arena": {}, "counters": {},
                    "writer": {"epoch_scope": "process"}}
        if section is not None:
            document["disk_restore"] = section
        (directory / f"rank-{rank}.json").write_text(json.dumps(document))

    def arm(latched=True):
        # A LATCH (0.2.35): the file says armed and nothing in it expires, so the
        # helper writes what the reader now reads and no field it ignores.
        (directory / disk_spill.ARM_FILE).write_text(json.dumps(
            {"armed": True, "latched": latched}))

    write(None)
    assert disk_restore.worker_proof(force=True)["ok"] is False
    write({"installed": False, "armed": True, "worker_fitted": True})
    assert disk_restore.worker_proof(force=True)["ok"] is False

    # CASE 1 - every rank fitted and claiming armed, NO arm file: refused, and
    # named as the ARM FILE's refusal. The arm file outranks every rank file: no
    # file from the last armed window may arm a rebooted or expired boot.
    write({"installed": True, "armed": True, "worker_fitted": True})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "tier_arm_disarmed"
    assert "arm file" in verdict["reason"]

    # CASE 2 - the real per-rank veto: installed but NOT FITTED in this worker.
    arm()
    write({"installed": True, "armed": False, "worker_fitted": False,
           "service_reason_kind": "worker_half_not_fitted"})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False, "a rank that cannot bind cannot vouch for a load"
    assert verdict["reason_kind"] == "scheduler_armed_workers_disarmed"
    assert verdict["ranks_armed"] == 0 and verdict["ranks_seen"] == 1
    assert "ranks_armed=0/1" in verdict["reason"]
    assert "not fitted in this worker" in verdict["reason"]

    # CASE 3 - TONIGHT'S FIX: installed and fitted, with the rank file's OWN
    # armed verdict false (written before the arm). The arm file is armed; the
    # proof PASSES and counts the stale claim instead of locking the tier out.
    write({"installed": True, "armed": False,
           "reason": "the disk tier is disarmed", "worker_fitted": True})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True, verdict["reason"]
    assert verdict["ranks_armed"] == 1
    assert verdict["stale_arm_claims"] == 1, (
        "the stale armed:false is counted and published, not trusted")

    # A file from a plugin that predates `worker_fitted` (absent field) is NOT a
    # veto: an absent fact is not evidence of a refusal - the old fail-closed on
    # `armed` remains in reporting only.
    write({"installed": True, "armed": "true"})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True, "the string 'true' is not armed, and not a veto"
    assert verdict["stale_arm_claims"] == 1

    # CASE 4 - installed AND armed AND fitted: the full proof, with the rank's
    # own applied-read count and no stale claims.
    write({"installed": True, "armed": True, "worker_fitted": True,
           "reads_applied": 3})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True and verdict["ranks"] == ["PP0_TP0"]
    assert verdict["ranks_armed"] == 1 and verdict["reads_applied"] == 3, \
        "the worker's own applied-read count is the scheduler's only evidence"
    assert verdict["stale_arm_claims"] == 0

    # CASE 5 - `not_configured` is a PER-RANK env fact the arm file cannot see:
    # this worker has no disk root, so it cannot name a path. A real veto.
    write({"installed": True, "armed": False, "worker_fitted": True,
           "service_reason_kind": "not_configured"})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert "no disk root" in verdict["reason"]

    # Freshness and unreadable files, unchanged in meaning.
    write({"installed": True, "armed": True, "worker_fitted": True},
          ts=time.time() - 3600.0)
    assert disk_restore.worker_proof(force=True)["ok"] is False, \
        "a heartbeat from a worker that may be gone proves nothing"
    write({"installed": True, "armed": True, "worker_fitted": True})
    (directory / "rank-huge.json").write_text(
        "x" * (disk_restore.MAX_RANK_FILE_BYTES + 10))
    (directory / "rank-broken.json").write_text("{ not json")
    assert disk_restore.worker_proof(force=True)["ok"] is True, \
        "one unreadable file is not evidence against the readable one"

    # AN EXPIRY IN THE FILE IS INERT (0.2.35). A document carrying an `expires_ts` in
    # the PAST - one written by an older build, or by anything that still thinks the arm
    # is a timer - arms the proof like any other, because the reader honours no clock.
    # The negative this test used to assert ("the proof refuses through the expiry") is
    # the exact behaviour removed: a serving tier going disarmed at an arbitrary minute.
    (directory / disk_spill.ARM_FILE).write_text(json.dumps(
        {"armed": True, "expires_ts": time.time() - 60}))
    assert disk_restore.worker_proof(force=True)["ok"] is True

    # What DOES close the proof is the file's absence or another boot's identity, and
    # the absence is the ordinary one: no file, no disk tier for this worker.
    (directory / disk_spill.ARM_FILE).unlink()
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "tier_arm_disarmed"
    assert "no arm file" in verdict["reason"]


def test_one_unrefused_rank_vetoes_the_whole_scheduler(tmp_path, monkeypatch):
    """THE INCIDENT, as a test: 1 rank serving, 1 rank that cannot bind, 0 offers.

    The load event spans every rank, so a rank that passes disk entries through "as
    though it had none" leaves blocks the fork already committed to - which the
    engine reports as invalid ids, truncates to 0 tokens and reschedules. A proof
    that accepted "some of my ranks serve" would have authorised every one of the
    123,474 offers. The bar is AND over the live ranks, and the refusal names the
    split so a reader sees a partial fleet instead of inferring one.

    0.2.28 changed WHICH field spells the refusal: a rank that cannot serve is one
    whose WORKER HALF IS NOT FITTED (a fact the rank file uniquely knows), not one
    whose lagged `armed` mirror says no - the tier arm moved to its authoritative
    copy, the arm file. Both refusals below read through the same
    `scheduler_armed_workers_disarmed` kind, so the gate's incident defence is
    byte-identical.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setattr(config, "diagnostics_dir", lambda: directory)
    (directory / disk_spill.ARM_FILE).write_text(json.dumps(
        {"armed": True, "expires_ts": time.time() + 600}))

    def write(rank, section):
        document = {"version": 1, "epoch": "aabbccdd",
                    "ts": round(time.time(), 6), "role": "worker", "rank": rank,
                    "arena": {}, "counters": {},
                    "writer": {"epoch_scope": "process"},
                    "disk_restore": section}
        (directory / f"rank-{rank}.json").write_text(json.dumps(document))

    write("PP0_TP0", {"installed": True, "armed": True, "worker_fitted": True,
                      "reads_applied": 7})
    write("PP1_TP0", {"installed": True, "armed": True, "worker_fitted": False,
                      "service_reason_kind": "worker_half_not_fitted"})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False, "a partial fleet is a refusal, not a partial offer"
    assert verdict["reason_kind"] == "scheduler_armed_workers_disarmed"
    assert verdict["ranks_armed"] == 1 and verdict["ranks_seen"] == 2
    assert "ranks_armed=1/2" in verdict["reason"]
    # The armed rank's reads are still reported - the veto is about offering, not
    # about pretending the other rank is doing nothing.
    assert verdict["reads_applied"] == 7

    # Pull the veto and the same two files prove the fleet.
    write("PP1_TP0", {"installed": True, "armed": True, "worker_fitted": True,
                      "reads_applied": 2})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True and verdict["ranks_armed"] == 2
    assert verdict["reads_applied"] == 9


def test_worker_proof_is_cached_and_never_read_when_closed(tmp_path, monkeypatch):
    """One listdir per second, and none at all while the gate is already closed.

    The gate is consulted on every lookup and a lookup happens for every waiting
    request: an uncached listdir plus eight file reads per lookup would put filesystem
    latency in the scheduler's hot path for a fact that changes once per boot. A
    disabled tier must not pay it at all.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setattr(config, "diagnostics_dir", lambda: directory)
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(directory))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: False)
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    probes = []

    def probe():
        probes.append(1)
        return {"ok": True, "count": 1, "ranks": ["r"], "reason": None}

    monkeypatch.setattr(disk_restore, "_read_worker_proof", probe)
    disk_restore.set_offer_switch(True)
    assert disk_restore.gate()["ok"] is False
    assert probes == [], "a closed gate must not read the diagnostics directory"
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    disk_restore.gate()
    disk_restore.gate()
    disk_restore.gate()
    assert len(probes) == 1


# --- b: the identity, which file holds the next span -------------------------

def test_offer_arithmetic_is_a_floor_in_whole_blocks():
    """3,199 matched tokens offers NOTHING; 3,200 offers 3,200; 6,500 offers 3,200.

    Measured on the live box: the scheduler block-aligns external tokens
    (`scheduler.py:974-977`) and DISCARDS a sub-block offer while the request sits in
    WAITING_FOR_REMOTE_KVS for a load that will never come - 142 of 158 measured
    sub-block offers were thrown away. Offering a remainder is worse than silence.
    """
    assert disk_restore.whole_block_tokens(3199, BLOCK)["tokens"] == 0
    assert disk_restore.whole_block_tokens(3200, BLOCK)["tokens"] == 3200
    assert disk_restore.whole_block_tokens(6500, BLOCK)["tokens"] == 6400
    assert disk_restore.whole_block_tokens(6500, BLOCK)["discarded_tokens"] == 100
    assert disk_restore.whole_block_tokens(100, 0)["ok"] is False
    assert disk_restore.whole_block_tokens(-1, BLOCK)["ok"] is False
    assert disk_restore.whole_block_tokens(100, True)["ok"] is False


def test_eligible_groups_are_cacheable_whole_block_and_stable():
    """Prefix-cacheable AND block_size == the scheduler's AND positionally stable.

    A scratch group's CPU hit is dropped before pairing (`manager.py:440-446`), so a
    slot offered there is never paired and never freed; a group with a smaller block
    would land our slots at the wrong GPU offsets while the arithmetic still looked
    whole; and a group whose blocks are not positionally stable has no stable file
    name to read back, so restoring it would paste bytes computed for other positions.
    """
    verdict = disk_restore._eligible_groups(Manager(), BLOCK)
    assert verdict["ok"] is True
    assert verdict["groups"] == list(CACHEABLE)
    assert verdict["num_groups"] == len(GROUP_SIZES)

    refused = disk_restore._eligible_groups(Manager(stable=False), BLOCK)
    assert refused["ok"] is False and refused["reason_kind"] == "no_eligible_group"

    small = disk_restore._eligible_groups(Manager(group_sizes=(100,) * 6), BLOCK)
    assert small["ok"] is False and small["group_block_sizes"] == [100] * 6

    assert disk_restore._eligible_groups(Manager(cacheable=[]), BLOCK)["ok"] is False
    no_coord = disk_restore._eligible_groups(Manager(no_coordinator=True), BLOCK)
    assert no_coord["ok"] is False, "an unverifiable group is not eligible"


def test_keys_are_built_by_the_engines_own_two_helpers(armed, monkeypatch):
    """The group-qualified key comes from `resolve_block_hashes` plus the packer.

    A plugin that re-implemented either would eventually disagree with the engine
    about which bytes belong to which group, and that disagreement is a file read
    under the wrong name - so the module resolves both from `sys.modules` and refuses
    when they are absent. Here they are replaced with spies, pinning that the offer
    asks the ENGINE rather than itself, and at the block size rather than the hash
    size.
    """
    seen = {"resolve": [], "pack": []}

    def spy_resolve(block_hashes, hash_block_size, block_size):
        seen["resolve"].append((hash_block_size, block_size))
        return resolve(block_hashes, hash_block_size, block_size)

    def spy_pack(digest, group_id):
        seen["pack"].append(group_id)
        return pack(digest, group_id)

    utils = sys.modules[disk_restore.KV_UTILS_MODULE]
    monkeypatch.setattr(utils, "resolve_block_hashes", spy_resolve)
    monkeypatch.setattr(utils, "make_block_hash_with_group_id", spy_pack)
    fill(armed, blocks=2)
    assert offer(Manager(), Req(blocks=2)) == (2 * BLOCK, True)
    assert seen["resolve"] == [(HASH_BLOCK, BLOCK)] * len(CACHEABLE)
    assert sorted(seen["pack"]) == sorted(list(CACHEABLE) * 2)


def test_keys_refuse_without_the_helper_module(armed, monkeypatch):
    """No `kv_cache_utils` in this process: refuse, never guess the packing."""
    monkeypatch.delitem(sys.modules, disk_restore.KV_UTILS_MODULE)
    fill(armed, blocks=2)
    assert offer(Manager(), Req(blocks=2)) == (0, False)
    assert disk_restore.counters()["refused"]["no_key_helpers"] == 1


def test_the_run_is_contiguous_a_gap_ends_it(armed):
    """A hole is not skipped over.

    A restore past a gap would place block N+1's bytes at block N's GPU offset and
    serve the request a KV shifted by a block - wrong bytes, no error. Recomputing
    from the gap is the only sound response, so the offer stops short.
    """
    fill(armed, blocks=4)
    assert offer(Manager(), Req(blocks=4)) == (4 * BLOCK, True)
    for group in CACHEABLE:
        removed = disk_index.journal_append(disk_index.REMOVE_LINE,
                                            key_hex(3, group), WIDTH)
        assert removed["ok"], removed["reason"]
    disk_index.refresh(force=True)
    # A FRESH request id: the same id asked again at the same computed tokens is a
    # reschedule after a failed load and is refused by the loop fuse, which is what
    # `test_a_failed_load_is_never_offered_again_on_the_same_request` pins.
    assert offer(Manager(), Req(request_id="req-2", blocks=4)) == (2 * BLOCK, True)


def test_run_refuses_a_hash_that_is_not_bytes(armed):
    """The group suffix can only be packed onto raw digest bytes: refuse the request."""
    fill(armed, blocks=1)
    request = Req(blocks=1)
    request.block_hashes = ["not-bytes"] * STEP
    assert offer(Manager(), request) == (0, False)
    assert disk_restore.counters()["refused"]["no_block_hashes"] == 1


def test_run_refuses_a_request_carrying_no_hashes(armed):
    fill(armed, blocks=1)
    request = Req(blocks=1)
    del request.block_hashes
    assert offer(Manager(), request) == (0, False)
    assert disk_restore.counters()["refused"]["no_block_hashes"] == 1


# --- c: the offer, and the merge into the fork's tuple -----------------------

def test_offer_extends_the_forks_hit_and_parks_nothing(armed):
    """The whole point: (ram, async) becomes (ram + disk, True), and only that.

    The scheduler then adds the external tokens to `num_computed_tokens`
    (`scheduler.py:1030-1032`), allocates GPU blocks and calls
    `update_state_after_alloc` - all fork code, unmodified.
    """
    fill(armed, blocks=3)
    manager = Manager()
    assert offer(manager, Req(blocks=3)) == (3 * BLOCK, True)
    counters = disk_restore.counters()
    assert counters["offers"] == 1
    assert counters["offered_tokens"] == 3 * BLOCK
    assert counters["slots_allocated"] == 3 * len(CACHEABLE)
    assert len(manager.cpu_block_pool.allocated) == 3 * len(CACHEABLE)


def test_merged_tuple_is_the_forks_shape_with_our_slots_last(armed_gap_fill):
    """(per-group lists, hit_length, num_computed), ours APPENDED after the RAM hit.

    `update_state_after_alloc` takes `cdiv(num_external, g_block_size)` LEADING
    entries per group, pairs them at `num_computed // g_block_size + i`, and frees
    the remainder. So RAM must stay leading and our slots must follow; the third
    field is the fork's local prefix and must survive untouched, because the pairing
    offsets are computed from it.

    THIS IS THE GAP-FILL PATH (`armed_gap_fill`), and it is the ONLY path that merges a
    RAM hit with a disk span: under disk authority the RAM hit is not consulted and the
    merge never happens. `test_disk_restore_authority.py` pins what authority does
    instead.
    """
    fill(armed_gap_fill, seeds=[3, 4])
    manager = Manager()
    ram = [[Slot(900 + group * 10 + i) for i in range(2)]
           for group in range(len(GROUP_SIZES))]
    manager._pending_cpu_hits["req-1"] = (ram, 2 * BLOCK, 0)
    request = Req(request_id="req-1", blocks=4)
    assert offer(manager, request, 0, (2 * BLOCK, False)) == (4 * BLOCK, True)
    lists, hit_length, num_computed = manager._pending_cpu_hits["req-1"]
    assert num_computed == 0, "the fork's local prefix must survive untouched"
    assert hit_length == 4 * BLOCK
    assert len(lists) == len(GROUP_SIZES)
    allocated = {block.block_id for block in manager.cpu_block_pool.allocated}
    assert len(allocated) == 2 * len(CACHEABLE)
    for group in range(len(GROUP_SIZES)):
        assert lists[group][:2] == ram[group], "the RAM hit stays leading"
        if group in CACHEABLE:
            assert len(lists[group]) == 4, "two RAM blocks + two disk blocks"
            mine = {block.block_id for block in lists[group][2:]}
            assert mine <= allocated, "our slots are the freshly allocated ones"
        else:
            assert len(lists[group]) == 2, "an ineligible group gains no slots"


def test_merge_refuses_a_ram_hit_that_does_not_add_up(armed_gap_fill):
    """A RAM tuple whose per-group length disagrees with its token count: refuse all.

    The merge rests on "leading entries are the RAM hit, one per whole group block".
    If that is false, our slots land at GPU offsets belonging to someone else's
    tokens - a wrong-bytes restore, the one corruption this module must never cause.
    A refusal costs a recompute. Under gap fill (`armed_gap_fill`) a malformed hit is a
    refusal; under disk authority the hit is irrelevant rather than a reason to refuse.
    """
    fill(armed_gap_fill, seeds=[3, 4])
    manager = Manager()
    manager._pending_cpu_hits["req-1"] = ([[Slot(700)] for _ in GROUP_SIZES],
                                          2 * BLOCK, 0)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (2 * BLOCK, False), \
        "the fork's own answer, untouched"
    assert disk_restore.counters()["refused"]["ram_shape"] == 1
    assert manager.cpu_block_pool.freed, "the allocation was given back"
    assert manager.cpu_block_pool.get_num_free_blocks() == 1024


def test_merge_refuses_a_tuple_the_fork_left_behind(armed_gap_fill):
    """Tokens claimed with no `_pending_cpu_hits` tuple: there is nothing to merge.

    Under gap fill (`armed_gap_fill`) this is fatal; under disk authority there is
    nothing to merge EITHER, and that is fine rather than fatal - see
    `test_a_request_with_no_fork_tuple_costs_nothing_because_ram_is_not_consulted`.
    """
    fill(armed_gap_fill, seeds=[3, 4])
    assert offer(Manager(), Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (2 * BLOCK, False)
    assert disk_restore.counters()["refused"]["ram_shape"] == 1


def test_a_sub_block_match_offers_nothing_through_the_real_path(armed):
    """The clamp is the scheduler's own remaining budget, and the floor is ours.

    One disk block is available but the request is 2,000 tokens long, so at most
    1,999 tokens may be restored - less than one 3,200-token scheduler block. A naive
    offer would promise a partial block the scheduler truncates, parking the request
    for a load it throws away. The floor must win.
    """
    fill(armed, blocks=1)
    request = Req(blocks=1, num_tokens=2000)
    assert offer(Manager(), request, 0, (0, False)) == (0, False)
    assert disk_restore.counters()["refused"]["disk_run_below_one_block"] == 1


def test_the_offer_refuses_a_hit_that_is_not_block_aligned(armed_gap_fill):
    """A hash-granular RAM hit interleaves with our blocks inside a group block.

    UNMARKED ONLY, and that is the 0.2.22 change the operator asked for: this refusal
    used to be the reason a disk restore could not be tested on any prompt that was
    not a whole number of blocks. A flagged request takes this branch never, because
    the RAM hit it carries is not the thing that decides its span.
    """
    fill(armed_gap_fill, blocks=2)
    assert offer(Manager(), Req(blocks=2), 0,
                 (100, False)) == (100, False)
    assert disk_restore.counters()["refused"]["ram_hit_not_block_aligned"] == 1


def test_the_offer_refuses_an_unaligned_local_prefix(armed):
    fill(armed, blocks=2)
    assert offer(Manager(), Req(blocks=2), 100, (0, False)) == (0, False)
    assert disk_restore.counters()["refused"]["unaligned_local_prefix"] == 1


def test_the_offer_refuses_a_request_that_opted_out(armed):
    """`skip_reading_prefix_cache` means the fork returned nothing ON PURPOSE."""
    fill(armed, blocks=2)
    request = Req(blocks=2, skip_reading_prefix_cache=True)
    assert offer(Manager(), request) == (0, False)
    assert disk_restore.counters()["refused"]["skip_read_requested"] == 1


def test_the_offer_refuses_when_no_token_is_left_to_sample(armed):
    """The fork's rule: at least one token must recompute, or the sampler
    starves."""
    fill(armed, blocks=2)
    request = Req(blocks=2, num_tokens=BLOCK)
    assert offer(Manager(), request, BLOCK, (BLOCK, False)) == (BLOCK, False)
    assert disk_restore.counters()["refused"]["no_tokens_left"] == 1


def test_a_refused_offer_gives_every_slot_back(armed, monkeypatch):
    """No slot leak, and freed tail-first.

    Freeing a chain root-first lets the root be evicted before the blocks pinned
    behind it, which is why the fork's own free path reverses; the undo here matches
    it, and the free count has to come back exactly.
    """
    fill(armed, blocks=2)
    manager = Manager()
    monkeypatch.setattr(disk_restore, "_merge_into_pending",
                        lambda *a, **k: disk_restore._fail("injected",
                                                           reason_kind="ram_shape"))
    before = manager.cpu_block_pool.get_num_free_blocks()
    assert offer(manager, Req(blocks=2)) == (0, False)
    assert manager.cpu_block_pool.get_num_free_blocks() == before
    assert manager.cpu_block_pool.freed == [
        [block.block_id for block in reversed(manager.cpu_block_pool.allocated)]]
    assert disk_restore.counters()["refused"]["ram_shape"] == 1


def test_the_offer_never_touches_the_cpu_reserve(armed):
    """A pool with no room above the reserve offers nothing at all.

    The spill path needs CPU slots too. A restore that drained the pool would trade a
    recompute for a store that can never happen, which is a strictly worse box.
    """
    thin = CpuPool(num_blocks=disk_restore.CPU_SLOT_RESERVE + 2)
    fill(armed, blocks=4)
    assert offer(Manager(pool=thin), Req(blocks=4)) == (0, False)
    assert thin.allocated == []
    assert disk_restore.counters()["refused"]["no_cpu_slots"] == 1


def test_the_offer_shrinks_to_whole_spans_only(armed):
    """Five free slots for three groups buys ONE span, never a partial one.

    A span is one slot per eligible group, and a half-span would be a load entry with
    no bytes behind it. So the clamp is in SPANS, and the resolved key list is cut to
    match the slots that exist.
    """
    pool = CpuPool(num_blocks=disk_restore.CPU_SLOT_RESERVE + 5)
    fill(armed, blocks=4)
    manager = Manager(pool=pool)
    assert offer(manager, Req(blocks=4)) == (BLOCK, True)
    assert len(manager.cpu_block_pool.allocated) == len(CACHEABLE)
    record = disk_restore._loads["req-1"]
    assert record["tokens"] == BLOCK
    assert len(record["identities"]) == len(CACHEABLE)


def test_the_offer_refuses_a_pool_that_raises(armed):
    """`get_new_blocks` raises past the free list: a counted refusal, never a raise."""
    pool = CpuPool()

    def refuse(count):
        raise ValueError("no blocks for you")

    pool.get_new_blocks = refuse
    fill(armed, blocks=2)
    assert offer(Manager(pool=pool), Req(blocks=2)) == (0, False)
    assert disk_restore.counters()["refused"]["alloc_refused"] == 1


def test_a_hostile_manager_cannot_raise_into_the_scheduler(armed):
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError(f"no {name} for you")

    fill(armed, blocks=1)
    assert disk_restore.offer(Exploding(), Req(blocks=1), 0,
                              (0, False)) == (0, False)
    assert disk_restore.counters()["errors"] >= 1


def test_a_hostile_request_cannot_raise_into_the_scheduler(armed):
    class Exploding:
        request_id = "req-x"
        num_tokens = 10_000

        @property
        def block_hashes(self):
            raise RuntimeError("no hashes")

    fill(armed, blocks=1)
    assert disk_restore.offer(Manager(), Exploding(), 0,
                              (0, False)) == (0, False)


def test_the_offer_ledger_is_bounded(armed, monkeypatch):
    """Two concurrent disk loads is the cap here, and a third is refused not queued."""
    monkeypatch.setattr(disk_restore, "MAX_TRACKED_LOADS", 2)
    fill(armed, blocks=1)
    for index in range(3):
        offer(Manager(), Req(request_id=f"req-{index}", blocks=1))
    assert len(disk_restore._loads) <= 2
    assert disk_restore.counters()["refused"]["too_many_loads"] == 1


# --- d: the scheduler's other door: name the files ---------------------------

def test_load_paths_are_one_entry_per_load_entry(armed):
    """THE LENGTH RULE. `load_paths` is index-aligned with `load_cpu_blocks` or the
    worker reads a file into the wrong slot.

    Every entry is present, empty where nothing is on disk. The worker decides "is
    this entry a disk one" from the entry's CONTENT, never from the list's length, so
    a short list would silently misalign everything after the gap - the exact bug the
    fork's own `store_keys` comment warns about.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    cpu = merged_cpu(manager)
    assert len(cpu) == len(CACHEABLE)
    metadata = Meta(load_event=7, cpu=[cpu[0], 400] + cpu[1:],
                    gpu=[10_000, 45_000] + [10_100 + i for i
                                            in range(len(cpu) - 1)],
                    reqs={7: ["req-1"]})
    report = disk_restore.build_load_paths(manager, metadata)
    assert report["ok"] is True
    assert len(metadata.load_paths) == len(metadata.load_cpu_blocks)
    named = [path for path in metadata.load_paths if path]
    assert len(named) == len(CACHEABLE)
    assert all(path.endswith(f"-{PLACEHOLDER}.bin") for path in named)
    assert metadata.load_paths[1] == "", "the RAM-only entry stays empty"
    assert disk_restore.counters()["path_entries"] == len(CACHEABLE)


def test_load_paths_are_published_even_for_an_empty_step(armed):
    """An empty step still gets an (empty) list: the worker's test is content-based."""
    metadata = Meta()
    report = disk_restore.build_load_paths(Manager(), metadata)
    assert report["ok"] is True
    assert metadata.load_paths == []


def test_a_slot_the_fork_no_longer_owns_gets_no_path(armed):
    """THE CORRUPTION GUARD, tested as a guarantee rather than a hope.

    The fork frees a pending CPU hit when a request is aborted or preempted, and the
    pool hands those ids to the next request. If the slot->path map outlived that,
    the next owner's slot could be named a file for someone else's tokens - a
    stale-by-one-step KV write. So the sweep runs at the TOP of `build_load_paths`,
    before anything is emitted.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = list(disk_restore._slot_paths)
    assert slots
    manager._reqs_to_load.pop("req-1")
    metadata = Meta(load_event=3, cpu=slots, gpu=[50_000 + i for i in slots])
    disk_restore.build_load_paths(manager, metadata)
    assert metadata.load_paths == [""] * len(slots)
    assert disk_restore.counters()["stale_slots_dropped"] >= 1
    assert disk_restore._slot_paths.get(slots[0]) is None


def test_a_load_older_than_its_ledger_gets_no_path(armed, monkeypatch):
    """The scheduler-side ledger expires too, so a lost report cannot linger."""
    monkeypatch.setattr(disk_restore, "RESTORE_DEADLINE_SECONDS", 0.0)
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = list(disk_restore._slot_paths)
    metadata = Meta(load_event=3, cpu=slots, gpu=[50_000 + i for i in slots])
    disk_restore.build_load_paths(manager, metadata)
    assert metadata.load_paths == [""] * len(slots)
    assert disk_restore.counters()["watchdog_fires"] >= 1


def test_paths_are_blanked_when_the_step_is_too_big(armed):
    """A step over the cap attributes NOTHING: choosing a subset would be guessing."""
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = list(disk_restore._slot_paths)
    extra = [9_000 + i for i in range(disk_restore.MAX_PATHS_PER_STEP)]
    metadata = Meta(load_event=2, cpu=slots + extra,
                    gpu=[100 + i for i in range(len(slots) + len(extra))])
    disk_restore.build_load_paths(manager, metadata)
    assert metadata.load_paths == [""] * len(metadata.load_cpu_blocks)


def test_build_load_paths_never_raises_on_a_step_that_refuses_attributes():
    """A step that will not take the field is a counted error, never a raised one.

    The scheduler's own metadata object refusing an attribute is a fork-shape
    change, and the restore's answer to every fork-shape change is the same: count
    it, emit no paths, and let the step proceed exactly as it would have with no
    plugin installed.
    """
    class Frozen:
        load_cpu_blocks = [1, 2]
        load_gpu_blocks = [3, 4]
        load_event = 5
        load_event_to_reqs = {}

        def __setattr__(self, name, value):
            raise RuntimeError("read-only step")

    frozen = Frozen()
    report = disk_restore.build_load_paths(Manager(), frozen)
    assert report.get("emitted") is not True
    assert disk_restore.counters()["errors"] >= 1


def test_adoption_is_measured_from_emission(armed):
    """"Adopted" means our CPU ids reached a worker's `load_cpu_blocks`.

    On the live box the adoption floor makes most offers vanish between
    `get_num_new_matched_tokens` and `update_state_after_alloc`; without this counter
    there is no way to tell "never offered" from "offered and discarded".
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    assert disk_restore.counters()["adopted"] == 0
    slots = merged_cpu(manager)
    metadata = Meta(load_event=4, cpu=slots, gpu=[70_000 + i for i in slots],
                    reqs={4: ["req-1"]})
    disk_restore.build_load_paths(manager, metadata)
    counters = disk_restore.counters()
    assert counters["adopted"] == 1
    assert counters["adopted_tokens"] == BLOCK
    assert counters["paths_emitted_steps"] == 1


def test_an_unadopted_completion_releases_the_ledger_and_restores_nothing(armed):
    """The counter the witness needs is not "the worker said finished".

    A request whose slots never reached a step restored nothing, whatever the report
    says. The ledger still has to let it go.
    """
    fill(armed, blocks=1)
    output = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"req-1"},
                             finished_sending=set())
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    report = disk_restore.note_reported(manager, output)
    assert report["restorations"] == 0
    assert report["released"] == 1, "the ledger still lets it go"


def test_a_reported_load_with_no_applied_read_is_no_read_not_a_restoration(armed):
    """THE LIE, pinned: a finished report with no disk read behind it.

    The fork reports a load finished for its OWN host-RAM blocks too, so a request
    whose disk slots nobody filled produces a report indistinguishable from success.
    That is how a live boot counted 123,161 `restorations` with `reads_applied=0` in
    the same log line. `no_read` is what it is, and it is never a restoration.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=4, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={4: ["req-1"]}))
    reported = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"req-1"},
                               finished_sending=set())
    report = disk_restore.note_reported(manager, reported)
    assert report["restorations"] == 0, "a finished report with no read is not a " \
                                        "restoration"
    assert report["no_read"] == 1
    assert disk_restore.counters()["no_read"] == 1
    assert disk_restore.counters()["restorations"] == 0
    assert disk_restore._loads == {} and disk_restore._slot_paths == {}


def test_a_reported_load_with_an_applied_read_is_a_restoration(armed):
    """The restoration the witness is allowed to count: adopted, reported, READ.

    The scheduler never reads a file itself, so `reads_applied` here stands for the
    worker's attestation (its own counter in a worker process, its rank file in a
    scheduler process); `applied_reads_evidence()` names which source answered.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=5, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={5: ["req-1"]}))
    done = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"req-1"},
                           finished_sending=set())
    report = disk_restore.note_reported(manager, done)
    assert report["restorations"] == 1
    assert disk_restore._loads == {}
    assert disk_restore._slot_paths == {}, "a finished load frees its mappings"


def test_a_failed_load_is_never_offered_again_on_the_same_request(armed):
    """THE LOOP TEST: offer, fail, asked again on the same id => no second offer.

    The engine's recovery from a KV load failure resets the request and reschedules
    it, which asks `get_num_new_matched_tokens` again; the index has not changed, so
    without a fuse the answer is the same 160,000 tokens and the same failure. On the
    live box that loop ran 123,474 times in 3.9 minutes and starved the harness into
    a watchdog reset.

    Driven through the real path: offer, then the fork's failure report, then the
    same request id coming back at the SAME computed tokens (what a reset looks
    like). The second answer must be the fork's own value, the suppression counter
    must move, and no new slots may be allocated.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    request = Req(request_id="req-1", blocks=2)
    assert offer(manager, request) == (2 * BLOCK, True)
    allocated = disk_restore.counters()["slots_allocated"]
    assert allocated > 0, "the first offer must really have allocated"

    # The fork reports the load as FAILED for this request id, then reschedules it.
    failed = fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                             finished_sending=set(), failed_recving={"req-1"},
                             invalid_block_ids=set())
    report = disk_restore.note_reported(manager, failed)
    assert report["failures"] == 1
    assert disk_restore.counters()["suppressions"] == 1

    # The same request, back at the door at the same place. The fork's answer stands.
    again = offer(Manager(reqs_to_load={"req-1": object()}), request)
    assert again == (0, False), "a failed load re-offered is the loop itself"
    counters = disk_restore.counters()
    assert counters["refused"]["suppressed_after_failure"] == 1
    assert counters["suppressed_hits"] == 1
    assert counters["slots_allocated"] == allocated, \
        "a suppressed request must not allocate a single slot"
    assert counters["offers"] == 1, "exactly one offer happened, for two askings"


def test_a_failed_span_is_refused_for_every_request(armed):
    """The span half of the fuse: a DIFFERENT request wanting the same span is refused.

    The harness that caused the incident replayed the same prompts, so the failing
    span was re-requested by fresh request ids over and over. Fusing only the id
    would have let every one of those through.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)
    failed = fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                             finished_sending=set(), failed_recving={"req-1"},
                             invalid_block_ids=set())
    disk_restore.note_reported(manager, failed)

    counters = disk_restore.counters()
    before = counters["refused"]["suppressed_after_failure"]
    other = offer(Manager(reqs_to_load={"req-9": object()}),
                  Req(request_id="req-9", blocks=2))
    assert other == (0, False)
    assert disk_restore.counters()["suppressed_spans"] == 1
    assert disk_restore.counters()["refused"]["suppressed_after_failure"] == before + 1


def test_invalid_block_ids_without_a_request_name_still_burn_the_load(armed):
    """The block-id-only failure shape is not allowed to keep offering.

    `get_block_ids_with_load_errors` is OURS (the base connector returns an empty
    set), so invalid ids on a step that carried our slots are ours even when
    `failed_recving` stays empty. Over-broad on purpose: a wrongly-fused request loses
    one restore, an unfused failure feeds the loop.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=3, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={3: ["req-1"]}))
    out = fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                          finished_sending=set(), failed_recving=set(),
                          invalid_block_ids={700 + n for n in range(len(slots))})
    report = disk_restore.note_reported(manager, out)
    assert report.get("failures") == 1, report
    assert disk_restore.counters()["suppressions"] == 1
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=1)) == (0, False)


def test_completion_of_an_unknown_request_is_ignored(armed):
    output = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"ghost"},
                             finished_sending=set())
    report = disk_restore.note_reported(Manager(), output)
    assert report["released"] == 0 and report["restorations"] == 0


def test_completion_never_raises():
    class Exploding:
        @property
        def finished_recving(self):
            raise RuntimeError("no output for you")

    assert disk_restore.note_reported(Manager(), Exploding())["ok"] is False


# --- e: the worker's doors ---------------------------------------------------

def test_bind_queues_a_read_per_disk_entry_without_waiting(armed):
    """The engine thread pays a `put_nowait` per entry. No file I/O on its thread.

    Reads are queued at bind and applied one forward pass later at `start_load_kv`;
    whatever has not settled by then is dropped. That is what keeps NVMe out of the
    critical path, and it is why a slow disk can only ever cost a recompute.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    report = bind(worker, metadata)
    assert report["ok"] is True
    assert report["queued"] == len(CACHEABLE)
    counters = disk_restore.counters()
    assert counters["binds"] == 1
    assert counters["disk_binds"] == 1
    assert counters["reads_started"] == len(CACHEABLE)
    assert settle(disk_restore._batch_id(11))["settled"] is True


def test_bind_without_disk_entries_is_free(armed):
    worker = Worker()
    report = bind(worker, Meta(load_event=1, cpu=[1], gpu=[2]))
    assert report["ok"] is True and report["queued"] == 0
    assert disk_restore.counters()["disk_binds"] == 0


def test_bind_refuses_a_misaligned_step(armed):
    """`load_paths` shorter or longer than `load_cpu_blocks`: nothing is attributed."""
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,))
    metadata.load_paths = metadata.load_paths[:-2]
    report = bind(worker, metadata)
    assert report["ok"] is False
    assert report["reason_kind"] == "path_count_mismatch"
    assert disk_restore.counters()["reads_started"] == 0


def test_bind_refuses_a_step_without_a_load_event(armed):
    fill(armed, blocks=1)
    metadata = disk_meta(armed, seeds=(1,))
    metadata.load_event = -1
    assert bind(Worker(), metadata)["reason_kind"] == "bad_batch"


def test_worker_rebuilds_the_name_with_its_own_digest(armed):
    """The scheduler sends an identity; the worker supplies the rank's digest.

    The digest covers THIS rank's slot width (11,468,800 B on PP0, 12,902,400 B on
    PP1, plus the rank), which the scheduler process cannot know. A sentinel digest is
    the normal case; a REAL, DIFFERENT digest means two sites computed the same field
    and disagreed, and neither is trusted.
    """
    ours = disk_layout.build_slot_path(armed, key_hex(1, CACHEABLE[0]),
                                       PLACEHOLDER)["path"]
    resolved = disk_restore._resolve_worker_path(ours, DIGEST)
    assert resolved["ok"] is True
    assert resolved["path"].endswith(f"-{DIGEST}.bin")
    assert resolved["key16"] == key_hex(1, CACHEABLE[0])[:16]
    assert resolved["group_id"] == CACHEABLE[0]

    other = disk_layout.build_slot_path(armed, key_hex(1, CACHEABLE[0]),
                                        "ffee00112233")["path"]
    disagree = disk_restore._resolve_worker_path(other, DIGEST)
    assert disagree["ok"] is False
    assert disagree["reason_kind"] == "digest_disagreement"

    partial = os.path.join(os.path.dirname(ours),
                           os.path.basename(ours) + ".part")
    assert disk_restore._resolve_worker_path(partial,
                                             DIGEST)["reason_kind"] == "bad_path"
    assert disk_restore._resolve_worker_path("/home/elsewhere/x.bin",
                                             DIGEST)["reason_kind"] == "bad_path"
    assert disk_restore._resolve_worker_path("", DIGEST)["ok"] is False


def test_bind_refuses_without_a_fingerprint_width(armed, monkeypatch):
    """No `total_bytes_per_block` means no expected size, and a sizeless read reads a
    file of unknown meaning."""
    fill(armed, blocks=1)
    monkeypatch.setattr(disk_store, "counters",
                        lambda: {"geometry_digest": DIGEST, "fingerprint": {}})
    monkeypatch.setattr(disk_store, "adopt_fingerprint",
                        lambda worker: {"adopted": False, "reason": "no tensors"})
    report = bind(Worker(), disk_meta(armed, seeds=(1,)))
    assert report["ok"] is False
    assert report["reason_kind"] == "fingerprint_incomplete"
    assert disk_restore.counters()["reads_started"] == 0


def test_before_load_moves_bytes_into_the_cpu_slot(armed):
    """The whole path, for real: file -> bounce buffer -> memmove -> CPU slot.

    The bytes are checked at BOTH ends because the ordering IS the contract: the
    file's content is `(seed * 11 + group) % 251` repeated, so a slot holding any
    other byte was named by the wrong identity - the silent-corruption case, failing
    here loudly instead of in production quietly.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    assert bind(worker, metadata)["ok"] is True
    settle(disk_restore._batch_id(11))
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True
    assert report["applied"] == len(CACHEABLE) and report["dropped"] == 0
    counters = disk_restore.counters()
    assert counters["reads_applied"] == len(CACHEABLE)
    assert counters["bytes_applied"] == WIDTH * len(CACHEABLE)
    region = worker.cpu_kv_caches["g0"]
    for index, group in enumerate(CACHEABLE):
        assert region.slot_bytes(index) == bytes([(1 * 11 + group) % 251] * WIDTH)
    assert metadata.load_cpu_blocks == list(range(len(CACHEABLE))), \
        "a step that copied everything keeps every entry"
    assert disk_restore._wbatches == {}, \
        "the batch is closed, so a second pass cannot re-apply these bytes"


def test_the_span_shape_comes_from_the_real_bind_path(armed):
    """END TO END, the field `span_shape` depends on: a job bound through the real
    `_worker_bind` carries its GROUP ID, so the span walk sees one span of three
    groups and not three spans of one.

    Why this is a separate test from the pure-function ones: those hand-build a batch
    with a `group` key, which would keep passing if `_worker_bind` stopped recording
    the group at all - and a batch whose every entry looked like group 0 would make
    every span a single entry, so `spans_partial` and `spans_hole_in_the_middle`
    would count entries and every number they produce would be a lie about the
    ordering question.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    assert bind(worker, metadata)["ok"] is True
    settle(disk_restore._batch_id(11))
    assert disk_restore.worker_before_load(worker)["ok"] is True
    shape = disk_restore.counters()["last_span_shape"]
    assert shape["entries"] == len(CACHEABLE)
    assert shape["spans"] == 1, shape          # 3 groups, 1 block: one span
    assert shape["spans_whole"] == 1 and shape["spans_partial"] == 0
    assert shape["applied_is_prefix"] is True
    assert disk_restore.counters()["spans_hole_in_the_middle"] == 0


def test_unsettled_bytes_are_pruned_never_awaited(armed, monkeypatch):
    """Not settled at `start_load_kv` = pruned, load-errored, DMA'd NEVER.

    The tempting alternative is a bounded wait on the engine thread, which would put
    NVMe latency in the loop every forward pass runs through. Instead the entry
    leaves the step's load lists (so the fork's `launch_copy` cannot reach the stale
    slot) and its GPU block is reported as a load error, which makes the scheduler
    recompute those tokens.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    bind(worker, metadata)
    cpu, gpu = list(metadata.load_cpu_blocks), list(metadata.load_gpu_blocks)
    monkeypatch.setattr(disk_read, "batch_buffer", lambda *a, **k: None)
    monkeypatch.setattr(disk_read, "batch_status",
                        lambda batch_id: {"known": True, "pending": 3, "running": 0,
                                          "settled": False, "incomplete": False,
                                          "deadline_expired": False})
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True
    assert report["dropped"] == len(CACHEABLE)
    assert report["submitted"] is True, "the step's RAM entries still proceed"
    assert metadata.load_cpu_blocks == cpu[len(CACHEABLE):]
    assert metadata.load_gpu_blocks == gpu[len(CACHEABLE):]
    assert len(metadata.load_cpu_blocks) == len(metadata.load_gpu_blocks), \
        "pruning must not separate a CPU id from its GPU id"
    assert disk_restore.counters()["dma_unsettled_entries"] == len(CACHEABLE)
    assert disk_restore.take_invalid_block_ids() == set(gpu[:len(CACHEABLE)])


def test_a_dropped_step_reports_rather_than_going_silent(armed):
    """The "never report" bug, pinned.

    There is no engine-side timeout for WAITING_FOR_REMOTE_KVS, so a load that is
    simply not reported parks the request for the life of the boot. Removal from the
    pending set and the failure report therefore happen in ONE call.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    worker._pending_load_event_indices.add(11)
    vanish(root=armed, seeds=(1,))
    bind(worker, metadata)
    settle(disk_restore._batch_id(11))
    disk_restore.worker_before_load(worker)
    assert disk_restore.counters()["phantom_guard_trips"] == 0
    sending, recving = disk_restore.worker_get_finished(worker, (set(), None))
    assert sending == set() and recving == {"req-1"}
    assert disk_restore.counters()["finished_released"] >= 1
    assert disk_restore.worker_get_finished(worker, (set(), None)) == (set(), None), \
        "the fork's value passes through untouched once the releases are out"


def test_a_submitted_event_is_left_pending_by_design(armed):
    """The mirror test: a SUBMITTED event stays, or loads would never finish."""
    worker = Worker()
    metadata = Meta(load_event=22, cpu=[1], gpu=[2], paths=[""], reqs={22: ["r"]})
    worker._pending_load_event_indices.add(22)
    assert disk_restore._phantom_guard(worker, metadata,
                                       submitted=True)["ok"] is True
    assert worker._pending_load_event_indices == {22}
    assert disk_restore.counters()["phantom_guard_trips"] == 0


def test_a_misaligned_step_is_abandoned_entirely(armed):
    """Lengths disagree at `start_load_kv`: the whole step recomputes.

    With the pairing unknown there is no safe selective action - pruning "the disk
    ones" would be a guess about which GPU block each file belongs to. Recompute is
    always correct, so every GPU block of the step is reported as a load error.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    worker._pending_load_event_indices.add(11)
    bind(worker, metadata)
    metadata.load_gpu_blocks = metadata.load_gpu_blocks[:-1]
    gpu = list(metadata.load_gpu_blocks)
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True
    assert metadata.load_cpu_blocks == [] and metadata.load_paths == []
    assert disk_restore.take_invalid_block_ids() == set(gpu)
    assert disk_restore.counters()["path_length_violations"] == 1
    assert 11 not in worker._pending_load_event_indices


def test_the_region_plan_must_total_the_file_exactly(armed):
    """A slot smaller than the file: refuse, and apply NOTHING.

    Half a slot IS the corruption: the KV for those tokens would be right for its
    first kilobytes and stale for the rest, and nothing downstream could tell. The
    spill side refuses the same width mismatch for the same reason.
    """
    good = Worker(regions={"g0": Region(width=WIDTH)})
    small = Worker(regions={"g0": Region(width=WIDTH // 2)})
    plan = disk_restore._plan_regions(good, 0, WIDTH)
    assert plan["ok"] is True and plan["total"] == WIDTH
    refused = disk_restore._plan_regions(small, 0, WIDTH)
    assert refused["ok"] is False and refused["reason_kind"] == "slot_plan_failed"
    assert disk_restore._plan_regions(Worker(regions={}), 0, WIDTH)["ok"] is False
    assert disk_restore._plan_regions(good, 99, WIDTH)["ok"] is False
    buffer = (ctypes.c_char * WIDTH)()
    ctypes.memset(buffer, 0, WIDTH)
    outcome = disk_restore._apply_buffer(small, {"cpu_block_id": 0}, buffer, WIDTH)
    assert outcome["ok"] is False
    assert ctypes.string_at(buffer, 4) == b"\x00\x00\x00\x00", "nothing was written"


def test_the_invalid_block_report_drains(armed):
    """Drained in one read, because a stale id would truncate an UNRELATED request.

    The engine calls `get_block_ids_with_load_errors` every step and
    `_handle_invalid_blocks` maps an id to whichever request holds it NOW, so
    reporting an id twice is reporting it to the wrong request the second time.
    """
    disk_restore._remember_invalid(501)
    disk_restore._remember_invalid(502)

    def original(self):
        return {400}

    wrapped = disk_restore._wrapped_load_errors(original)
    assert wrapped(object()) == {400, 501, 502}
    assert wrapped(object()) == {400}
    assert disk_restore.counters()["invalid_reports"] == 2

    def explodes(self):
        raise RuntimeError("the fork is having a day")

    disk_restore._remember_invalid(503)
    loud = disk_restore._wrapped_load_errors(explodes)
    assert loud(object()) == {503}, "our report still goes out"


def test_the_watchdog_leaves_a_live_load_alone(armed):
    fill(armed, blocks=1)
    worker = Worker()
    bind(worker, disk_meta(armed, seeds=(1,)))
    assert disk_restore.sweep_watchdog(worker)["fired"] == 0
    assert disk_restore._wloads
    disk_read.cancel_batch(disk_restore._batch_id(11), "test teardown")


def test_the_worker_tables_are_bounded(armed, monkeypatch):
    """Caps hold under a flood: the oldest goes, so a leak cannot grow with uptime.

    Requests do get aborted and preempted, and a dict in a worker that lives for a
    week is a leak with a slow fuse.
    """
    monkeypatch.setattr(disk_restore, "MAX_TRACKED_LOADS", 2)
    monkeypatch.setattr(disk_read, "MAX_BATCHES", 2)
    worker = Worker()
    for event in range(11, 17):
        metadata = disk_meta(armed, seeds=(1,))
        metadata.load_event = event
        metadata.load_event_to_reqs = {event: [f"req-{event}"]}
        bind(worker, metadata)
    assert len(disk_restore._wloads) <= 4
    assert len(disk_restore._wbatches) <= 3
    for batch_id in list(disk_restore._wbatches):
        disk_read.cancel_batch(batch_id, "test teardown")



def test_install_declines_when_nothing_is_imported():
    report = disk_restore.install()
    assert report["installed"] is False
    assert "imported" in report["reason"]
    assert disk_restore.counters()["installed"] is False


def test_make_wrapper_only_knows_its_three_methods():
    assert disk_restore.make_wrapper("get_num_new_matched_tokens",
                                     lambda self, r, n: (0, False)) is not None
    assert disk_restore.make_wrapper("request_finished", lambda self: None) is None


# --- g: observability, because a silent restore is one that never ran --------

def test_boot_line_names_the_verdict_and_the_counters():
    line = disk_restore.boot_line()
    assert line.startswith("DISK[restore] ")
    for field in ("installed=", "class=", "armed=", "root=", "index=", "digest=",
                  "offers=", "reads_applied=", "ranks_armed=", "deadline=",
                  "disk_authority=", "serving_as=", "offers_switch=",
                  "offer_block_cap=", "offer_block_cap_tokens=",
                  "refused="):
        assert field in line, field
    # THE PER-OFFER BOUND, named in the unit that is actually a bound. `offer_cap=` is
    # GONE with the fixed 32,000-token cap it advertised (0.2.32): leaving a field whose
    # name means "the most one offer may carry in tokens" reporting a number that
    # constrains nothing is worse than printing nothing, so the line carries the block
    # walk instead - blocks, and the tokens those blocks are worth.
    for retired in ("offer_cap=", "offer_cap_src="):
        assert retired not in line, retired
    assert f"offer_block_cap={disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER}" in line
    # A fresh boot says so, in the field an operator reads to answer "is this boot
    # serving everyone from SSD?"
    assert "disk_authority=OFF" in line
    assert "serving_as=off" in line
    assert len(line) <= disk_restore.MAX_BOOT_LINE_CHARS * 4


def test_boot_line_logs_once_per_verdict(caplog):
    with caplog.at_level(logging.WARNING):
        disk_restore.emit_boot_line()
        disk_restore.emit_boot_line()
        disk_restore.emit_boot_line()
    lines = [rec.getMessage() for rec in caplog.records
             if "DISK[restore]" in rec.getMessage()]
    assert len(lines) == 1


def test_boot_line_survives_a_state_it_cannot_read(monkeypatch):
    """The fallback line: an unreadable posture still prints, and says so."""
    def explode():
        raise RuntimeError("nope")

    monkeypatch.setattr(disk_restore, "gate", explode)
    line = disk_restore.boot_line()
    assert line.startswith("DISK[restore] ")
    assert "unreadable=RuntimeError" in line


def test_witness_is_refuted_before_anything_is_offered():
    verdict = disk_restore.witness_verdict()
    assert verdict["verdict"] == "REFUTED"
    assert verdict["reason"].startswith("never_offered")


def test_witness_names_where_the_chain_broke(armed):
    """REFUTED is not one answer.

    Offers-without-adoption and adoption-without-completion are different bugs in
    different processes; the reason field is what tells an operator which file to
    open.
    """
    fill(armed, blocks=1)
    offer(Manager(), Req(request_id="req-a", blocks=1))
    assert disk_restore.witness_verdict()["reason"].startswith(
        "offers_not_adopted")

    manager = Manager(reqs_to_load={"req-b": object()})
    offer(manager, Req(request_id="req-b", blocks=1))
    slots = merged_cpu(manager, "req-b")
    disk_restore.build_load_paths(manager, Meta(load_event=1, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={1: ["req-b"]}))
    verdict = disk_restore.witness_verdict()
    assert verdict["adopted"] == 1
    assert verdict["reason"].startswith("adopted_never_reported")


def test_a_reported_restoration_without_a_read_refutes_the_witness(armed):
    """The incident's log line, reproduced: a boot that CONFIRMED itself while
    restoring nothing. `restorations` now stays 0 without a read, so the witness says
    REFUTED and names why.

    This is the single highest-value assertion in the suite: a witness that can be
    satisfied by a finished report is not evidence of anything.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=1, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={1: ["req-1"]}))
    output = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"req-1"},
                             finished_sending=set())
    assert disk_restore.note_reported(manager, output)["restorations"] == 0
    verdict = disk_restore.witness_verdict()
    assert verdict["verdict"] == "REFUTED"
    assert "no_disk_read_behind_the_report" in verdict["reason"]
    assert "verdict=REFUTED" in disk_restore.witness_line()
    assert "reads_applied=0" in disk_restore.witness_line()


def test_witness_is_confirmed_by_a_reported_restoration_and_a_read(armed):
    """CONFIRMED needs a load that reached a worker, was reported finished, AND had a
    disk read applied somewhere in the boot.

    That single greppable line is the answer to "does the restore serve requests, or
    did it never run" - reachable only by the real chain, not by an offer or a byte
    count. The second condition is what was missing on 2026-10-05, where the same line
    printed `verdict=CONFIRMED reads_applied=0` 8,244 times in a boot that had
    restored nothing.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=1, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={1: ["req-1"]}))
    output = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"req-1"},
                             finished_sending=set())
    assert disk_restore.note_reported(manager, output)["restorations"] == 1
    verdict = disk_restore.witness_verdict()
    assert verdict["verdict"] == "CONFIRMED" and verdict["reason"] == "-"
    assert verdict["reads_applied"] >= 1
    assert "verdict=CONFIRMED" in disk_restore.witness_line()
    assert "DISK[restore-witness]" in disk_restore.witness_line()


def test_a_worker_attested_read_counts_even_with_the_scheduler_at_zero(armed):
    """The scheduler's own read counter is 0 BY CONSTRUCTION - it never reads a file.

    So the worker's rank file is the only evidence that can cross the process
    boundary, and a witness that demanded the scheduler's local counter would say
    REFUTED on a healthy boot forever. `applied_reads_evidence()` names which source
    answered so the two kinds of zero stay distinguishable.
    """
    assert disk_restore.applied_reads_evidence()["reads_applied"] == 0
    disk_restore._state["ranks_reads_applied"] = 5
    evidence = disk_restore.applied_reads_evidence()
    assert evidence["reads_applied"] == 5
    assert evidence["source"] == "worker rank file"


def test_the_witness_is_not_re_emitted_while_the_verdict_is_unchanged(armed, caplog):
    """THE FLOOD FIX: the dedupe key is (verdict, reason), not the whole line.

    Keyed on the line, every step re-issued a "changed" witness because
    `offered_tokens` moves constantly - 1,048 lines a second on the incident boot,
    and the log volume was the second outage. Counters may move all they like; one
    verdict gets one line.
    """
    fill(armed, blocks=4)
    with caplog.at_level(logging.WARNING, logger="kvtier_dev_plugin.disk_restore"):
        for ordinal in range(20):
            # Distinct ids: the loop fuse would (correctly) refuse a repeat of one.
            offer(Manager(), Req(request_id=f"flood-{ordinal}", blocks=4))
        lines = [record.getMessage() for record in caplog.records
                 if "restore-witness" in record.getMessage()]
    assert len(lines) == 1, \
        f"20 lookups emitted {len(lines)} witness lines; the verdict never changed"
    assert disk_restore.counters()["witness_emissions"] == 1
    assert disk_restore.counters()["witness_suppressed"] >= 18, \
        "the suppressions have to be counted, or a quiet log looks like a dead module"


def test_the_witness_line_is_logged_once_per_verdict(armed, caplog):
    """One line per VERDICT, not one per step and not one per counter movement."""
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=1, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={1: ["req-1"]}))
    output = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"req-1"},
                             finished_sending=set())
    with caplog.at_level(logging.WARNING):
        disk_restore.note_reported(manager, output)
        # A second report on an empty ledger: the counters move, the verdict does not.
        disk_restore.note_reported(Manager(), output)
        disk_restore.emit_witness()
        disk_restore.emit_witness()
    lines = [rec.getMessage() for rec in caplog.records
             if "verdict=CONFIRMED" in rec.getMessage()]
    assert len(lines) == 1


def test_witness_line_never_raises(monkeypatch):
    def explode():
        raise RuntimeError("nope")

    monkeypatch.setattr(disk_restore, "witness_verdict", explode)
    assert "unreadable=RuntimeError" in disk_restore.witness_line()


def test_worker_report_is_an_allow_listed_rank_file_section(armed):
    """The rank file is a guest: one section, under the cap, allow-listed name."""
    assert "disk_restore" in worker_diag.CONTRIBUTION_SECTIONS
    report = disk_restore.worker_report()
    assert report["section"] == "disk_restore"
    assert len(json.dumps(report)) <= worker_diag.MAX_CONTRIBUTION_BYTES
    first = disk_restore.register_contribution()
    assert first.get("registered") is True or first.get("ok") is True
    again = disk_restore.register_contribution()
    assert again.get("registered") is True or again.get("ok") is True


def test_the_worker_report_publishes_the_per_rank_facts_and_a_change_signal(
        armed):
    """0.2.28: the rank file carries the facts the scheduler's proof reads PER RANK
    (`worker_fitted`, `service_reason_kind`) - and registers a change SIGNAL so an
    arm transition rewrites the file on the next step instead of waiting for the
    spill cadence that never comes on an idle box.

    The proof stopped ASKING the rank files about the tier arm (that moved to the
    arm file, its authoritative copy - see the drain suite); what it still reads
    here is exactly what these two fields say, and what the signal keeps fresh.
    """
    report = disk_restore.worker_report()
    assert isinstance(report["worker_fitted"], bool)
    assert report["service_reason_kind"] is None or isinstance(
        report["service_reason_kind"], str)
    assert disk_restore.register_contribution()["registered"] is True
    signature = worker_diag._signature(object(), 0)
    rows = dict(signature[-1])
    assert "disk_restore" in rows, (
        "the restore contribution registers a change signal, so an arm transition "
        "alone forces a rank-file rewrite (defence in depth for the idle-box hole "
        "the arm-file read closes properly)")


def test_section_reports_the_whole_state(armed):
    view = disk_restore.section()
    for key in ("gate", "deadline_seconds", "limits", "read_latency_seconds",
                "witness", "witness_line", "index", "read", "worker_proof",
                "offer_refusal_reasons", "worker_failure_reasons", "note"):
        assert key in view, key
    assert view["deadline_seconds"] == disk_restore.RESTORE_DEADLINE_SECONDS
    assert view["limits"]["identity_digest_placeholder"] == PLACEHOLDER
    for name in view["offer_refusal_reasons"]:
        assert name in disk_restore.OFFER_REFUSALS


def test_read_latency_reports_from_the_readers_own_ledger():
    average, maximum = disk_restore.read_latency()
    assert average == 0.0 and maximum == 0.0


# --- h: the wiring in engine_side --------------------------------------------

def test_the_worker_hook_installs_the_restore_half(monkeypatch):
    """`_on_worker_report` drives it, with the other worker facilities.

    Four wrappers of `get_finished` in one process: whichever lands last composes
    outermost, so the ORDER is a fact about behaviour and not cosmetic. The restore
    hook goes last because it is the one that must union its own `finished_recving`
    releases into the value the layers beneath it produced.
    """
    calls = []

    def fake_ensure(connector=None):
        calls.append("disk_restore")
        return {"installed": True, "class": "X", "reason": "hooked"}

    monkeypatch.setattr(disk_restore, "ensure_installed", fake_ensure)
    for name in ("worker_diag", "blockdump", "disk_store"):
        module = sys.modules["kvtier_dev_plugin." + name]
        monkeypatch.setattr(module, "ensure_installed",
                            lambda connector=None: {"installed": True})
    engine_side._on_worker_report(object())
    assert calls == ["disk_restore"]


def test_the_plugin_load_install_cannot_be_the_scheduler_fit(monkeypatch,
                                                             caplog):
    """WHAT THE OLD ASSERTION ACTUALLY PROVED - and what this one pins instead.

    This test used to assert that `register()` LOGS about the restore layer, and
    nothing else: an INFO line (which the engine's log drops), about the one install
    call that CANNOT succeed in the EngineCore. `register()` runs at
    `core.py:214-217`; the tier connector is built at `core.py:260`, and the
    connector is what imports the manager - so the manager class does not exist yet
    when plugin load calls `disk_restore.install()`. That ordering is not the bug.
    The bug was that nothing else in this process ever called it again, while four
    WORKER ranks (whose handshake DOES drive a retry) printed `installed=YES` about
    a class they never execute. A tier that could spill but never restore read as
    healthy in every field this suite checked.

    So the assertion is inverted to the claim that matters in this role, at this
    moment: the offering half is NOT fitted, the refusal is COUNTED, and it is
    announced at WARNING - the level the box's log actually keeps.
    """
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setattr(engine_side, "attach_methods",
                        lambda cls: {"attached": True, "reason": "ok"})
    monkeypatch.setattr(engine_side, "_engine_core_class", lambda: object)
    # Boot lines are INFO only on the first install in a process; a full-suite run
    # has already installed, so make this registration the first.
    monkeypatch.setitem(engine_side._INSTALLS, "count", 1)
    # THE ROLE: an EngineCore at plugin-load time. Neither tier class is imported
    # yet - which is exactly why this install can only be a no-op.
    saved = sys.modules.pop(disk_restore.SCHED_MODULE, None)
    saved_worker = sys.modules.pop(disk_restore.WORKER_MODULE, None)
    try:
        with caplog.at_level(logging.INFO, logger="kvtier_dev_plugin.disk_restore"):
            engine_side.register()
        counters = disk_restore.counters()
        assert counters["sched_fitted"] is False
        assert counters["installed"] is False
        # DEFERRED, not refused: no tier class is imported at this moment, which is
        # the expected posture and must not read as a refusal on a healthy boot.
        assert counters["install_deferrals"] >= 1
        assert counters["install_refusals"] == 0
        loud = "\n".join(rec.getMessage() for rec in caplog.records
                         if rec.levelno >= logging.WARNING)
        # The install line is a DEFERRAL here (nothing to hook yet), so it belongs to
        # INFO, not to the WARNING-level noise the boot line still makes. A deferral
        # printed at WARNING is what made every healthy boot look broken.
        deferred = [rec.getMessage() for rec in caplog.records
                    if "restore-install" in rec.getMessage()]
        assert deferred, "the deferral has to be said"
        assert all("deferred=YES" in text for text in deferred), deferred
        assert not any("refused=YES" in text for text in deferred), deferred
        assert "DISK[restore] installed=NO" in loud
    finally:
        for name, previous in ((disk_restore.SCHED_MODULE, saved),
                               (disk_restore.WORKER_MODULE, saved_worker)):
            if previous is not None:
                sys.modules[name] = previous
            else:
                sys.modules.pop(name, None)


def test_no_env_var_arms_the_restore_path():
    """The gate reads the spill tier's root and arm and the tier's arm. Nothing new.

    A fourth switch would be a fourth way to have a half-configured box, and the
    restore path has no business being armed separately from the tier that spills.
    """
    import inspect

    source = inspect.getsource(disk_restore)
    assert "os.environ" not in source and "getenv" not in source


# --- j: the one hazard this feature cannot have: a wait without a deadline ----

def test_closing_a_batch_never_deadlocks_on_the_buffer_pool(armed):
    """REGRESSION: a self-deadlock that hung a unit test for six minutes.

    `disk_read._state_lock` is a plain `threading.Lock`, and `finish_batch`,
    `cancel_batch` and `_settle` all hold it while returning a bounce buffer to the
    pool. When that return took the lock a second time, the calling thread blocked on
    a lock IT ALREADY OWNED - no error, no traceback, no timeout, and a test that
    simply never ended. On the engine's `start_load_kv` path that is a permanently
    wedged scheduler, which is why the fix is a production guarantee rather than a
    test detail: the locked call sites now use `_give_buffer_locked`.

    Asserted with a watchdog thread instead of inline, because an inline call that
    regresses would HANG this test rather than fail it, and a hung test tells nobody
    anything. The bound is 10 s for work measured in microseconds.
    """
    import threading

    root = armed
    batch = disk_read.begin_batch("deadlock-regression", 1)
    assert batch["ok"] is True
    batch_id = batch["batch"]
    fill(root, seeds=[1])
    for index, group in enumerate(CACHEABLE):
        made = disk_layout.build_slot_path(root, key_hex(1, group), DIGEST)
        assert made["ok"] is True
        assert disk_read.request_read(batch_id, index, made["path"],
                                      WIDTH)["ok"] is True
    assert settle(batch_id)["settled"] is True

    outcome = {}

    def closer():
        outcome["report"] = disk_read.finish_batch(batch_id)

    thread = threading.Thread(target=closer, daemon=True)
    thread.start()
    thread.join(timeout=10.0)
    assert not thread.is_alive(), (
        "finish_batch blocked on a lock it already held: the `_give_buffer` "
        "self-deadlock is back"
    )
    assert outcome["report"]["closed"] is True


# --- i: the FORK'S OWN worker rules, so the release can be proven ------------

# Everything above drives `disk_restore`'s functions directly. That is the right
# shape for the arithmetic and the wrong shape for the one question that gates
# deployment: does a request ever end up parked in `WAITING_FOR_REMOTE_KVS`?
# `scheduler.py:3068-3084` releases that state only when the worker reports the id
# in `finished_recving`, and there is NO engine timeout for it - so the proof has to
# run through the installed wrappers, on a worker that submits and mines the way the
# fork does, and it has to name the block ids the engine reads back.
#
# The two fork rules this double transcribes (`worker.py:330-352`, quoted in
# `disk_restore`'s module docstring):
#   * `start_load_kv` launches the copy and records `load_event` in
#     `_pending_load_event_indices` ONLY when there is something to copy - so a step
#     this module prunes to nothing submits nothing and its index must never sit in
#     that set (that is the phantom);
#   * `get_finished` reports an index once the copy stream's high-water mark has
#     passed it, and the watermark only moves FORWARD - which is why a stale index
#     is reported by a LATER completion, and why a fake without a watermark could
#     never test the hazard.
# `get_block_ids_with_load_errors` is inherited from the connector base and lives in
# a `vllm.`-named module, because `disk_restore._install_load_errors` refuses to
# overwrite a report whose home is not vLLM's own tree.

FORK_BASE_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.base"

FORK_SCHED_SOURCE = '''
class SimpleCPUOffloadScheduler:
    """The three tier entry points, with the fork's own answers."""

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def build_connector_meta(self, scheduler_output=None):
        return "METADATA"

    def update_connector_output(self, connector_output):
        return None
'''

FORK_WORKER_SOURCE = '''
class SimpleCPUOffloadWorker:
    def __init__(self):
        self.cpu_kv_caches = {}
        self._connector_metadata = None
        self._pending_load_event_indices = set()
        self._load_hwm = -1
        self.completed = set()
        self.submitted = []
        self.copied = []

    def complete(self, event):
        """The copy stream reports this event's CUDA event as done."""
        self.completed.add(int(event))

    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata

    def start_load_kv(self):
        cpu = list(getattr(self._connector_metadata, "load_cpu_blocks", None)
                   or [])
        if not cpu:
            return                      # nothing to copy: NO submission
        event = int(getattr(self._connector_metadata, "load_event", -1))
        self._pending_load_event_indices.add(event)
        self.submitted.append(event)
        self.copied.extend(cpu)

    def get_finished(self, finished_req_ids):
        for index in self.completed:
            self._load_hwm = max(self._load_hwm, index)
        recving = set()
        for index in list(self._pending_load_event_indices):
            if index <= self._load_hwm:
                self._pending_load_event_indices.discard(index)
                table = (getattr(self._connector_metadata,
                                 "load_event_to_reqs", {}) or {})
                recving |= set(table.get(index, ()))
        return set(), (recving or None)
'''

FORK_BASE_SOURCE = '''
class KVConnectorBase_V1:
    def get_block_ids_with_load_errors(self):
        return set()
'''

FORK_CONNECTOR_SOURCE = '''
class SimpleCPUOffloadConnector(KVConnectorBase_V1):
    pass
'''

FORK_MODULES = (disk_restore.SCHED_MODULE, disk_restore.WORKER_MODULE,
                disk_restore.CONNECTOR_MODULE, FORK_BASE_MODULE)


def register_fork(name: str, source: str, **pre):
    """Exec `source` into a module NAMED `name`, registered in `sys.modules`.

    The `fork_class` posture (see `fakes_debug`): the plugin resolves the fork's
    names only against a module it can find already imported, and refuses anything
    whose `__module__` is not the fork's. A class defined in this file would carry
    `test_disk_restore` and be (correctly) refused - which is precisely the
    silent-failure mode this fixture exists to avoid: a wrapper installed on a name
    the fork does not have installs cleanly and never fires.
    """
    module = types.ModuleType(name)
    module.__dict__.update(pre)
    sys.modules[name] = module
    try:
        exec(source, module.__dict__)  # noqa: S102 - our own source, never input
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def fork():
    """The fork's scheduler, worker and connector classes, importably registered.

    `sys.modules` is restored here rather than through `monkeypatch`, because
    monkeypatch's undo runs AFTER this fixture's teardown: a `delitem` it owns
    re-registers the module behind our back and leaks a wrapped fork class into the
    next test, where a later `_layers_of()` sees a layer it was never supposed to
    find. (That exact leak made an unrelated `test_auto_boot.py` test fail, and it
    cost an hour to name.)
    """
    saved = {name: sys.modules.get(name) for name in FORK_MODULES}
    base = register_fork(FORK_BASE_MODULE, FORK_BASE_SOURCE)
    sched, _ = fakes_debug.fork_class(disk_restore.SCHED_MODULE,
                                      disk_restore.SCHED_CLASS,
                                      FORK_SCHED_SOURCE, register=True)
    worker, _ = fakes_debug.fork_class(disk_restore.WORKER_MODULE,
                                       disk_restore.WORKER_CLASS,
                                       FORK_WORKER_SOURCE, register=True)
    connector = register_fork(
        disk_restore.CONNECTOR_MODULE, FORK_CONNECTOR_SOURCE,
        KVConnectorBase_V1=base.KVConnectorBase_V1).SimpleCPUOffloadConnector
    try:
        yield {"sched": sched, "worker": worker, "connector": connector}
    finally:
        disk_restore.uninstall()
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class Engine:
    """The two objects a model runner touches, and the CPU region they share."""

    def __init__(self, worker, connector):
        self.worker = worker
        self.connector = connector

    @property
    def region(self):
        return self.worker.cpu_kv_caches["g0"]

    def bind(self, metadata):
        """The fork's bind, then our hook - exactly as the installed wrappers run."""
        self.worker.bind_connector_metadata(metadata)

    def step(self, wait_for_reads=True):
        """One forward pass: load prep, the load-error drain, then the mining.

        `wait_for_reads` gives the reads the one-forward-pass window the design
        assumes (a ~11 MB NVMe read lands well inside a step). Without it every
        entry would answer `not_settled`, which is a real mode but not the one
        being measured: it would report five different hardware failures as one.
        """
        if wait_for_reads:
            for batch_id in list(disk_restore._wbatches):
                settle(batch_id)
        self.worker.start_load_kv()
        errors = self.connector.get_block_ids_with_load_errors()
        for event in list(self.worker.submitted):
            self.worker.complete(event)
        sending, recving = self.worker.get_finished(set())
        return {"errors": errors, "sending": sending,
                "recving": recving or set(), "submitted": list(
                    self.worker.submitted)}


@pytest.fixture
def engine(fork, armed):
    """Both halves installed, a worker with a real region tensor, one boot."""
    report = disk_restore.install()
    assert report["installed"] is True, report
    worker = fork["worker"]()
    worker.cpu_kv_caches = {"g0": Region()}
    return Engine(worker, fork["connector"]())


def disk_only_meta(root, event, seeds=(1,), groups=CACHEABLE, ram=0,
                   digest=DIGEST):
    """A step whose load is entirely disk-sourced, with its own request id.

    Disk-only is the shape that matters: with a host-RAM hit alongside, the fork
    submits the RAM copy and reports the load itself, and a release from us would be
    a DOUBLE report. Only when nothing survives the pruning does the engine have
    nobody to report to - which is where a missing report becomes a parked request.
    """
    cpu, gpu, paths = [], [], []
    for group in groups:
        for seed in seeds:
            made = disk_layout.build_slot_path(root, key_hex(seed, group), digest)
            assert made["ok"], made["reason"]
            cpu.append(len(cpu))
            gpu.append(60_000 + len(cpu))
            paths.append(made["path"])
    for index in range(ram):
        cpu.append(410 + index)
        gpu.append(70_000 + index)
        paths.append("")
    return Meta(load_event=event, cpu=cpu, gpu=gpu, paths=paths,
                reqs={event: [f"req-{event}"]})


def expired_batches(event, seconds=1.0):
    """Backdate a live load so the watchdog treats it as expired. BOUNDED: the test
    never sleeps for a deadline, it moves the deadline."""
    batch = disk_restore._wbatches.get(disk_restore._batch_id(event))
    if batch is not None:
        batch["deadline_mono"] = time.monotonic() - seconds
    for record in disk_restore._wloads.values():
        record["deadline_mono"] = time.monotonic() - seconds


# --- j: BLOCKER 1 - every failure mode reaches a report ----------------------

def failure_mode(mode, engine, armed, monkeypatch, event=11):
    """Put one named failure in front of an otherwise-healthy restore.

    Each mode is induced where the real fault happens (the filesystem, the rank's
    own digest, the bounded queue, the thread, the clock, the process boundary),
    never by substituting the code under test.
    """
    fill(armed, blocks=1)
    metadata = disk_only_meta(armed, event,
                              digest=("ffee00112233" if mode == "digest"
                                      else DIGEST))
    if mode in {"vanished", "size_mismatch", "digest"}:
        if mode == "vanished":
            # Deleted AFTER the journal published the identity and BEFORE the worker
            # opened it: exactly the LRU-race the index has to survive. It is deleted
            # by the identity the WORKER will name (its own digest), not the
            # sentinel-digest path the scheduler sent - the published path is not a
            # file this boot could ever open.
            vanish(armed, seeds=(1,))
        if mode == "size_mismatch":
            for path in metadata.load_paths:
                if path:
                    with open(path, "ab") as handle:
                        handle.write(b"x")     # another geometry's slot width
        engine.bind(metadata)
        return metadata
    if mode == "short_read":
        real_open = open

        class Shortened:
            """A handle that answers `readinto` with fewer bytes than it promised."""

            def __init__(self, handle):
                self._handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self._handle.close()
                return False

            def readinto(self, view):
                with self._handle:
                    return self._handle.readinto(view[:1])

        monkeypatch.setattr(
            disk_read, "open",
            lambda path, *a, **k: Shortened(real_open(path, *a, **k)),
            raising=False)
        engine.bind(metadata)
        return metadata
    if mode == "queue":
        # The documented drop path: a bounded queue answers `put_nowait` with
        # `queue.Full` and the read is DROPPED, not queued onto an unbounded pile.
        stub = queue.Queue(maxsize=1)
        stub.put_nowait("the only slot, held by someone else")
        monkeypatch.setattr(disk_read, "_get_queue", lambda: stub)
        engine.bind(metadata)
        return metadata
    if mode == "thread_dead":
        monkeypatch.setattr(disk_read, "_ensure_thread", lambda: False)
        engine.bind(metadata)
        return metadata
    if mode == "worker_restart":
        engine.bind(metadata)
        # A worker process that came back mid-load: the metadata survived (the
        # scheduler sent it) and the batch tables did not.
        disk_restore._wbatches.clear()
        disk_restore._wloads.clear()
        return metadata
    if mode == "batch_deadline":
        engine.bind(metadata)
        settle(disk_restore._batch_id(event))
        # The READER's own batch deadline (`disk_read.LOAD_DEADLINE_SECONDS`), not
        # the watchdog's: bytes that arrive after it are not trusted even when they
        # arrived intact. Moving a clock is not substituting the code under test.
        disk_read._batches[disk_restore._batch_id(event)]["deadline_mono"] = \
            time.monotonic() - 1
        return metadata
    engine.bind(metadata)
    return metadata


FAILURE_MODES = ("vanished", "size_mismatch", "short_read", "digest", "queue",
                 "thread_dead", "worker_restart", "batch_deadline")

# One named bucket per mode, so the counter says WHICH hardware answer we got
# (`file_missing` = the LRU outran us; `size_mismatch` = a foreign geometry;
# `short_read` = a disk that answers short). `disk_read` classifies them; carrying
# the name through the restore layer is what makes the recomputes explainable.
FAILURE_BUCKETS = {"vanished": "file_missing", "size_mismatch": "size_mismatch",
                   "short_read": "short_read", "digest": "digest_disagreement",
                   "queue": "queue_full", "thread_dead": "thread_dead",
                   "worker_restart": "not_tracked",
                   "batch_deadline": "deadline_expired"}


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_every_failure_mode_reports_the_request(engine, armed, monkeypatch, mode):
    """BLOCKER 1, per mode: the GPU blocks AND the request are reported, in one step.

    Asserted on the engine's own two reads, not on a return code:
      * `get_block_ids_with_load_errors()` names every GPU block the load owned
        (that is what `scheduler.py:2008-2015` truncates and recomputes under
        `kv_load_failure_policy=recompute`);
      * `finished_recving` names the request (that is what
        `scheduler.py:3068-3084` releases `WAITING_FOR_REMOTE_KVS` on);
      * `copied` is EMPTY: the fork's DMA never saw a slot whose bytes were not
        there, which is the difference between a recompute and a wrong answer.
    """
    event = 11
    failure_mode(mode, engine, armed, monkeypatch, event)
    result = engine.step()
    counters = disk_restore.counters()
    assert result["recving"] == {f"req-{event}"}, \
        f"{mode}: the request was never released: {result}"
    assert result["errors"] == {60001, 60002, 60003}, \
        f"{mode}: the GPU blocks were not reported: {result}"
    assert engine.worker.copied == [], f"{mode}: a failed load was still DMA'd"
    assert counters["reads_failed"][FAILURE_BUCKETS[mode]] == len(CACHEABLE) or \
        counters["reads_failed"][FAILURE_BUCKETS[mode]] >= 1, \
        f"{mode}: counted under the wrong name: {counters['reads_failed']}"
    assert counters["phantom_guard_trips"] == 0
    assert 11 not in engine.worker._pending_load_event_indices, \
        f"{mode}: a never-submitted index was left pending (phantom)"


def test_a_healthy_restore_is_reported_by_the_FORK_not_by_us(engine, armed):
    """The control that gives the matrix its meaning.

    With good bytes the fork submits the event, the watermark passes it, and
    `finished_recving` arrives from the FORK with NO load errors and the CPU slots
    actually holding the file's bytes. If this test passes for the wrong reason
    (e.g. our own release firing anyway) the failure matrix above would be proving
    nothing about releases at all.
    """
    event = 11
    fill(armed, blocks=1)
    metadata = disk_only_meta(armed, event)
    engine.bind(metadata)
    result = engine.step()
    assert result["recving"] == {f"req-{event}"}
    assert result["errors"] == set(), "a good restore reported load errors"
    assert result["submitted"] == [event], "the fork never submitted the copy"
    assert disk_restore.counters()["reads_applied"] == len(CACHEABLE)
    for index, group in enumerate(CACHEABLE):
        assert engine.region.slot_bytes(index) == bytes([(1 * 11 + group) % 251]
                                                        * WIDTH)
    assert disk_restore.counters()["finished_released"] == 0, \
        "we released a load the fork had already reported"


def test_the_watchdog_does_not_re_report_a_settled_load(engine, armed):
    """REGRESSION: the watchdog used to fire 45 s LATER and release it AGAIN.

    A record left in `_wloads` after `start_load_kv` settled the load expired on its
    own deadline and pushed `finished_recving` for a request the engine had moved on
    from - a release nobody asked for, a `finished_released` counter that lies, and
    for a load still in flight a claim that bytes had loaded before the copy
    carrying them was finished. The record is retired the moment the step's outcome
    is final; the watchdog keeps the case it was built for, a load whose step never
    happened (`test_the_watchdog_releases_a_load_nobody_reported`).
    """
    fill(armed, blocks=1)
    engine.bind(disk_only_meta(armed, 11))
    engine.step()
    assert disk_restore._wloads == {}, "the settled load still owns a watchdog record"
    before = disk_restore.counters()["worker_watchdog_fires"]
    expired_batches(11)
    engine.worker.get_finished(set())
    assert disk_restore.counters()["worker_watchdog_fires"] == before
    assert disk_restore.counters()["finished_released"] == 0


def test_a_finished_batch_never_returns_a_buffer_twice(armed, monkeypatch):
    """REGRESSION, and the worst bug found on this path: two owners, one buffer.

    `finish_batch` returned the buffers of EVERY job it closed - including the ones
    the reader thread had not finished with. The thread returns its own buffer when
    it finds the batch gone, so the buffer went back to the pool TWICE, the pool
    handed the same memory to two later reads, and one request's file bytes would be
    memmoved into another request's CPU KV slot. Reachable through the ordinary
    path (`start_load_kv` while a read is in flight is what
    `dma_unsettled_entries` counts), and it is silent: no exception, no counter.

    Detected by counting the pool's objects against the pool's identities.
    """
    gate = threading.Event()
    real_read = disk_read._read_file
    monkeypatch.setattr(disk_read, "_read_file",
                        lambda path, buffer, size: (gate.wait(5.0),
                                                    real_read(path, buffer, size))[1])
    fill(armed, blocks=1)
    metadata = disk_only_meta(armed, 11)
    assert disk_read.begin_batch("race", len(CACHEABLE))["ok"] is True
    for index, path in enumerate(metadata.load_paths):
        assert disk_read.request_read("race", index, path, WIDTH)["ok"] is True
    settled_running = False
    for _ in range(400):
        if disk_read.batch_status("race")["running"] >= 1:
            settled_running = True
            break
        time.sleep(0.005)
    assert settled_running, "the reader thread never started a read"
    assert disk_read.finish_batch("race")["closed"] is True
    gate.set()
    for _ in range(400):
        if disk_read.counters()["reads_completed"]:
            break
        time.sleep(0.005)
    time.sleep(0.05)
    pooled = [id(buffer) for buffer, _size in disk_read._buffers]
    assert len(pooled) == len(set(pooled)), \
        f"the buffer pool holds the SAME buffer twice: {pooled}"


# --- k: BLOCKER 2 - a cancelled load is never reported as "loaded" -----------

def test_a_stale_index_would_be_reported_by_the_fork(fork):
    """THE HAZARD EXISTS, and this fake can see it: a control, not a tautology.

    An index left in `_pending_load_event_indices` after its load was dropped is
    reported as "loaded" by a LATER event's completion, because the watermark is
    monotone. A test of the guard is worthless unless something proves the phantom
    is real - so here the bug is introduced BY HAND and the fork's own mining is
    shown swallowing it. `disk_restore`'s job is to make this unreachable, and
    `test_a_never_submitted_index_is_removed_before_the_watermark_moves` is the
    half that says it does.
    """
    worker = fork["worker"]()
    worker.bind_connector_metadata(Meta(load_event=11, cpu=[0], gpu=[7],
                                        paths=[""], reqs={11: ["req-11"]}))
    worker._pending_load_event_indices.add(11)     # the bug, on purpose
    worker.complete(22)                            # a LATER event completed
    _sending, recving = worker.get_finished(set())
    assert recving == {"req-11"}, \
        "the fake no longer models the phantom, so the guard tests prove nothing"


def test_a_never_submitted_index_is_removed_before_the_watermark_moves(engine,
                                                                       armed):
    """The invariant, through the installed wrappers: nothing is mined that was
    never submitted.

    The prune that empties a step's load lists also stops the fork from submitting,
    so the index never enters `_pending_load_event_indices`; the guard re-checks the
    pair anyway and would COUNT a trip. Because the guard runs inside
    `start_load_kv` and the watermark only moves when the fork submits, every step
    that could advance the watermark has already run the guard.
    """
    fill(armed, blocks=1)
    metadata = disk_only_meta(armed, 11)
    vanish(armed, seeds=(1,))
    engine.bind(metadata)
    result = engine.step()
    assert engine.worker._pending_load_event_indices == set()
    assert engine.worker.submitted == []
    assert result["recving"] == {"req-11"} and result["errors"]
    assert disk_restore.counters()["phantom_guard_trips"] == 0


def test_phantom_guard_counts_and_closes_a_trip(armed):
    """The guard's own trip path: remove the index, release the requests, COUNT it.

    A trip means some path skipped a submission without releasing it. It must stay
    0 in production, and it must never be silent: the counter is the only evidence
    that the invariant was ever at risk.
    """
    worker = Worker()
    metadata = Meta(load_event=21, cpu=[], gpu=[], paths=[""],
                    reqs={21: ["late"]})
    worker._pending_load_event_indices.add(21)     # the bug, on purpose
    verdict = disk_restore._phantom_guard(worker, metadata, submitted=False)
    assert verdict["ok"] is False and verdict.get("tripped") is True
    assert worker._pending_load_event_indices == set(), "the trip stayed open"
    assert disk_restore.counters()["phantom_guard_trips"] == 1
    _sending, recving = disk_restore.worker_get_finished(worker, (set(), None))
    assert recving == {"late"}, "closing the trip parked the request"


def test_a_late_completion_writes_nothing_after_the_load_was_dropped(engine, armed,
                                                                     monkeypatch):
    """Cancel a load while its read is in flight, then let the reader finish.

    The bounce buffer is the whole reason this is safe: the destination is a
    scheduler-owned CPU slot, and the only writer is the engine thread at submit
    time. So a read that arrives after its load was dropped has nowhere to go - its
    bytes are thrown away, the slot keeps whatever stale block it held (which is why
    its GPU id was reported), and no later step reports the request a second time.
    """
    gate = threading.Event()
    real_read = disk_read._read_file
    monkeypatch.setattr(disk_read, "_read_file",
                        lambda path, buffer, size: (gate.wait(5.0),
                                                    real_read(path, buffer, size))[1])
    fill(armed, blocks=1)
    engine.bind(disk_only_meta(armed, 11))
    result = engine.step(wait_for_reads=False)     # nothing settled: all pruned
    untouched = [engine.region.slot_bytes(i) for i in range(len(CACHEABLE))]
    assert all(byte == 0 for slot in untouched for byte in slot), \
        "a pruned slot was written"
    assert result["recving"] == {"req-11"} and result["errors"]
    gate.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if disk_read.counters()["reads_completed"]:
            break
        time.sleep(0.005)
    after = [engine.region.slot_bytes(i) for i in range(len(CACHEABLE))]
    assert after == untouched, "a late completion wrote into a CPU KV slot"
    later = engine.step(wait_for_reads=False)
    assert later["recving"] == set(), "the dropped load was reported a second time"


def test_a_watchdog_expiry_releases_exactly_once(engine, armed):
    """A load expired by the watchdog is reported ONCE, and its blocks are reported.

    The engine has no timeout on `WAITING_FOR_REMOTE_KVS`, so the watchdog is the
    last resort; reporting twice is as bad as not reporting (the second report is
    attributed to whoever holds those blocks by then). The cross-step pairing of the
    release with its block report is a KNOWN GAP recorded above
    `disk_restore._defer_finish`; this test pins what the code guarantees today and
    is the test the pairing fix has to extend.
    """
    fill(armed, blocks=1)
    engine.bind(disk_only_meta(armed, 11))
    expired_batches(11)
    fired = disk_restore.sweep_watchdog(engine.worker)["fired"]
    blocks = disk_restore.take_invalid_block_ids()
    assert fired == 1
    assert blocks == {60001, 60002, 60003}
    assert disk_restore.counters()["worker_watchdog_fires"] == 1
    assert disk_restore.counters()["finished_released"] == 1
    settle(disk_restore._batch_id(11))
    assert engine.worker.start_load_kv() is None
    assert engine.worker.submitted == []
    _sending, recving = engine.worker.get_finished(set())
    assert recving == {"req-11"}, "the expired request was never released"
    assert disk_restore.counters()["finished_released"] == 1, \
        "the same load was released twice"


def test_our_own_failure_fails_closed_and_never_dma_unverified_bytes(engine, armed,
                                                                     monkeypatch):
    """If OUR prep raises, the disk entries must still never reach the DMA.

    The tempting reading of "our hook never raises" is that the except branch is
    dead code and can return a failure dict and move on. It cannot: the fork's
    `start_load_kv` runs immediately after the wrapper returns, and the entries we
    failed to classify are still in `load_cpu_blocks` - so the fork DMAs whatever
    the CPU slots happened to contain and reports the load as FINISHED. A probe with
    `batch_status` made to raise found exactly that (`copied == [0, 1, 2]`, the event
    submitted, no load error). The hook now abandons the step's disk work on any
    failure of its own: prune, report every GPU block, release the request.
    """
    fill(armed, blocks=1)
    metadata = disk_only_meta(armed, 11)
    engine.bind(metadata)
    settle(disk_restore._batch_id(11))

    def explode(batch_id):
        raise RuntimeError("the ledger is having a day")

    monkeypatch.setattr(disk_read, "batch_status", explode)
    assert engine.worker.start_load_kv() is None   # the engine sees no exception
    assert metadata.load_cpu_blocks == [], "unverified entries were left to DMA"
    assert engine.connector.get_block_ids_with_load_errors() == {60001, 60002,
                                                                 60003}
    _sending, recving = engine.worker.get_finished(set())
    assert recving == {"req-11"}
    assert engine.worker.copied == []
    assert disk_restore.counters()["reads_failed"]["abandoned"] == len(CACHEABLE)


# --- l: the install/composition group (restored, with the reasons inline) ----

def test_install_wraps_both_halves_and_is_idempotent(fork):
    """RESTORED. It went red on `reinstalls == 1` after a second `install()`.

    Wrong expectation, not a changed signature: `install()` short-circuits on
    `_state["installed"]` and answers "already installed", so a second call cannot
    count a re-install at all. `reinstalls` counts the OTHER reuse path - a fresh
    boot onto a class that still carries our layer - and that is the path worth
    pinning, because stacking a second restore layer on the first would DOUBLE every
    offer. Both are asserted below.
    """
    sched, worker_cls = fork["sched"], fork["worker"]
    first = disk_restore.install()
    assert first["installed"] is True
    assert "SimpleCPUOffloadScheduler" in first["class"]
    assert "SimpleCPUOffloadWorker" in first["class"]
    for name in disk_restore.SCHED_METHODS:
        assert getattr(getattr(sched, name), disk_restore.OWNER_ATTR) \
            == disk_restore.OWNER_TAG, name
    for name in disk_restore.WORKER_METHODS:
        assert getattr(getattr(worker_cls, name), disk_restore.OWNER_ATTR) \
            == disk_restore.OWNER_TAG, name
    connector_report = disk_restore.counters()["load_error_installs"]
    assert connector_report == 1

    second = disk_restore.install()          # the short-circuit path
    assert second["installed"] is True and "already" in second["reason"]
    assert disk_restore.counters()["installs"] == 1
    assert disk_restore.counters()["reinstalls"] == 0

    disk_restore.reset_state_for_tests()     # a fresh boot, the layer still fitted
    third = disk_restore.install()
    assert third["installed"] is True
    assert disk_restore.counters()["reinstalls"] == 1, \
        "a class that already carries our layer must be REUSED, not re-wrapped"
    assert disk_restore.counters()["load_error_installs"] == 1, \
        "the load-error report was fitted twice"
    # exactly one restore layer deep, both before and after
    from kvtier_dev_plugin import backport

    _layers, bottom = backport._layers_of(sched.get_num_new_matched_tokens)
    assert getattr(bottom, disk_restore.OWNER_ATTR, None) is None, \
        "the bottom is no longer the fork's own body: the layer was stacked"


def test_installed_scheduler_keeps_the_forks_metadata(fork, armed):
    """RESTORED. Red only because the fixture it needed (`fork`) had gone.

    Through the REAL wrappers, with the module's `Manager` as `self` (which is what
    the scheduler passes on the box): the object the fork returned comes back
    IDENTICAL. The wrapper mutates it - it fills `load_paths` - but never substitutes
    it, because the scheduler hands the same object to the worker and a substituted
    one would silently drop every field the fork put on it.
    """
    disk_restore.install()
    sched = fork["sched"]
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    request = Req(request_id="req-1", blocks=1)
    assert sched.get_num_new_matched_tokens(manager, request, 0) == (BLOCK, True)
    assert sched.build_connector_meta(manager, "output") == "METADATA"
    assert sched.update_connector_output(manager, object()) is None


def test_installed_worker_prunes_before_the_fork_copies(fork, armed):
    """RESTORED. It went red with `FileNotFoundError`: it deleted the WRONG FILE.

    The path the scheduler publishes carries the digest SENTINEL
    (`IDENTITY_DIGEST_PLACEHOLDER`) because the scheduler cannot know a rank's slot
    width; the file that exists - and the one that has to vanish - is the one named
    by the WORKER's own digest. `vanish()` exists for exactly that, and a test that
    removes the published path deletes a file that was never there.

    The behaviour it pins is the ordering: our pass runs BEFORE the fork's
    `start_load_kv`, so what the fork's copy sees is either a slot with bytes in it
    or an entry that is no longer in the lists. After the fork, the ordering is worth
    nothing and a stale slot has already been DMA'd.
    """
    disk_restore.install()
    fill(armed, blocks=1)
    vanish(armed, seeds=(1,))
    worker = fork["worker"]()
    worker.cpu_kv_caches = {"g0": Region()}
    metadata = disk_only_meta(armed, 11)
    worker.bind_connector_metadata(metadata)
    settle(disk_restore._batch_id(11))
    worker.start_load_kv()
    assert worker.copied == [], "the fork copied something from a wholly-failed step"
    assert worker.submitted == [], "a step with nothing to copy reported a submission"
    assert worker._pending_load_event_indices == set()


def test_install_refuses_a_foreign_owner_marker(fork, monkeypatch):
    """RESTORED. Red only because the fixture it needed (`fork`) had gone.

    A wrapper carrying ANOTHER package's marker is refused, not stacked on: we cannot
    reason about a layer whose semantics we have not read, and composing over it
    silently changes what that third party's callers get.
    """
    def foreign(self, request, num_computed_tokens):
        return 0, False

    foreign.__name__ = disk_restore.METHOD_MATCHED
    foreign.__module__ = fork["sched"].__module__
    setattr(foreign, disk_restore.OWNER_ATTR, "someone-else/1")
    monkeypatch.setattr(fork["sched"], disk_restore.METHOD_MATCHED, foreign)
    report = disk_restore.install()
    assert report["installed"] is False
    assert "foreign owner marker" in report["reason"]


def test_the_load_error_report_refuses_a_third_party_body(fork, armed, monkeypatch):
    """RESTORED, and it found a real hole: the refusal was reported, then erased.

    It went red on `installed is False`. That expectation was wrong for this process
    shape - this fixture gives ONE process BOTH halves, so the scheduler half fits
    and `install()` honestly reports itself installed for the half it fitted (the
    half-fitted tier is inert: `gate()` demands a worker's proof, and a refused
    worker cannot write one). What WAS wrong is the code: the success `reason`
    OVERWROTE the refusal and `_state["skip_reason"]` was cleared, so `boot_line()`
    - the operator's only line - lost the fact that the worker half was refused.
    Now the refusal rides in the same sentence and stays in `skip_reason`.
    """
    def theirs(self):
        return set()

    theirs.__module__ = "someone_elses_plugin"
    monkeypatch.setattr(fork["connector"], disk_restore.METHOD_LOAD_ERRORS, theirs)
    report = disk_restore.install()
    assert "SimpleCPUOffloadWorker" not in (report["class"] or "")
    assert "load-error report could not be fitted" in report["reason"], \
        "the refusal vanished from the install report"
    assert "load-error report could not be fitted" in \
        disk_restore.counters()["skip_reason"]
    assert "load-error report could not be fitted" in disk_restore.boot_line()
    assert disk_restore.METHOD_LOAD_ERRORS not in fork["connector"].__dict__ \
        or getattr(fork["connector"], disk_restore.METHOD_LOAD_ERRORS) is theirs, \
        "a third party's load-error report was overwritten anyway"


def test_a_worker_process_refuses_to_install_without_the_report(fork, monkeypatch):
    """The same refusal, in the process shape the box actually runs.

    A worker process has the worker and the connector and never the manager, so the
    refusal is FATAL there and `installed` must be False: a worker that cannot say
    "this read failed" is the one configuration that serves a request on KV that was
    never restored.
    """
    def theirs(self):
        return set()

    theirs.__module__ = "someone_elses_plugin"
    monkeypatch.setattr(fork["connector"], disk_restore.METHOD_LOAD_ERRORS, theirs)
    del sys.modules[disk_restore.SCHED_MODULE]
    report = disk_restore.install()
    assert report["installed"] is False
    assert "load-error report could not be fitted" in report["reason"]


def test_install_fits_under_a_guard_layer_and_guard_stays_outermost(fork):
    """RESTORED, and it found a real bug in `backport._layer_kinds()`.

    It went red on `"restore" in kinds`. The layer walk could not report a restore
    layer because `_layer_kinds()` never listed its marker - although the same
    function's own docstring said it returned one, `_rewrap()` could already rebuild
    one, and `disk_restore._install_sched_method` already refused to stack one. Both
    branches were unreachable, and the consequences were not cosmetic: the in-tree
    stand-down fingerprinted OUR wrapper and reported the FORK as drifted, and a
    later `backport.install()` saw `layers == []` and replaced the class attribute
    with its own copy - deleting the restore layer while this module still reported
    `installed=YES`. Guard outermost, restore at the bottom, and the fork's body
    still the thing that gets fingerprinted.
    """
    sched = fork["sched"]
    for name in disk_restore.SCHED_METHODS:
        setattr(sched, name, guard_mode._make_wrapper(name, getattr(sched, name)))
    report = disk_restore.install()
    assert report["installed"] is True
    assert "under guard" in disk_restore.counters()["nested"][0]
    layers, bottom = backport._layers_of(sched.get_num_new_matched_tokens)
    kinds = [kind for kind, _wrapper in layers]
    assert kinds[0] == "guard", f"the guard is no longer outermost: {kinds}"
    assert disk_restore.LAYER_KIND in kinds, f"the walk cannot see us: {kinds}"
    assert getattr(bottom, disk_restore.OWNER_ATTR, None) is None, \
        "the body underneath our layers is not the fork's own function"
    assert sched.build_connector_meta(object(), "output") == "METADATA"


def test_uninstall_puts_the_fork_back(fork):
    """RESTORED. Red only because the fixture it needed (`fork`) had gone.

    Every wrapped method is put back, including the connector's inherited load-error
    report - and `hasattr` is the test, because a wrapper left on a class attribute
    outlives the module that installed it (the classes are process globals).
    """
    sched, worker_cls = fork["sched"], fork["worker"]
    disk_restore.install()
    report = disk_restore.uninstall()
    assert report["uninstalled"] is True
    assert len(report["restored"]) == (len(disk_restore.SCHED_METHODS)
                                       + len(disk_restore.WORKER_METHODS) + 1)
    for name in disk_restore.SCHED_METHODS:
        assert not hasattr(getattr(sched, name), disk_restore.OWNER_ATTR), name
    for name in disk_restore.WORKER_METHODS:
        assert not hasattr(getattr(worker_cls, name),
                           disk_restore.OWNER_ATTR), name
    assert disk_restore.counters()["installed"] is False


def test_install_tolerates_a_process_with_only_one_half(fork):
    """RESTORED. Red only because the fixture it needed (`fork`) had gone.

    An EngineCore has the manager and never imports the worker; a worker is the
    other way round. Requiring both would mean the layer could never fit anywhere.
    The cross-process risk that tolerance opens - an offer with nobody to fill it -
    is closed by `worker_proof`, and the load-error report is NOT fitted here, which
    is what the last assertion checks.
    """
    del sys.modules[disk_restore.WORKER_MODULE]
    report = disk_restore.install()
    assert report["installed"] is True
    assert "SimpleCPUOffloadWorker" not in report["class"]
    assert disk_restore.METHOD_LOAD_ERRORS not in fork["connector"].__dict__


def test_unattributable_disk_entries_are_never_dmaed(armed):
    """RESTORED. It went red on `released == 1`, and the test was the thing that
    was wrong.

    It stood up a step with three disk entries AND two host-RAM entries, pruned the
    three disk ones, and then expected us to release the request. With RAM entries
    surviving, the fork DOES submit the event and reports the load itself - a second
    report from us would be a double release of the same request. `released == 0` is
    correct here and the release path is pinned per mode in section j (where
    nothing survives the pruning and we are the only reporter). The invariant that
    does hold in both shapes is the one asserted below: unattributable bytes never
    reach the DMA, and their GPU blocks are reported.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=2)
    cpu, gpu = list(metadata.load_cpu_blocks), list(metadata.load_gpu_blocks)
    worker._connector_metadata = metadata
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True and report["dropped"] == len(CACHEABLE)
    assert metadata.load_cpu_blocks == cpu[len(CACHEABLE):]
    assert metadata.load_gpu_blocks == gpu[len(CACHEABLE):]
    assert report["released"] == 0 and report["submitted"] is True
    assert disk_restore.counters()["reads_failed"]["not_tracked"] == len(CACHEABLE)
    assert disk_restore.take_invalid_block_ids() == set(gpu[:len(CACHEABLE)])


def test_the_watchdog_releases_a_load_nobody_reported(armed):
    """RESTORED. It went red on a DOUBLE DRAIN of `take_invalid_block_ids()`.

    The sweep itself fired on its first call (`fired == 1`); the test then read the
    invalid-block set a second time and compared the empty result with three. That
    drain-onces property is deliberate and pinned by
    `test_the_invalid_block_report_drains`: an id reported twice is attributed to
    whichever request holds that block by then, so the read IS the report. The
    earlier verdict "the worker-side sweep never fired" came from this assertion,
    not from the sweep.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    worker._pending_load_event_indices.add(11)
    bind(worker, metadata)
    assert disk_restore._wloads and disk_restore._wbatches
    expired_batches(11)
    assert disk_restore.sweep_watchdog(worker)["fired"] == 1
    assert disk_restore.counters()["worker_watchdog_fires"] == 1
    assert worker._pending_load_event_indices == set()
    blocks = disk_restore.take_invalid_block_ids()
    assert len(blocks) == len(CACHEABLE), blocks
    assert disk_restore.take_invalid_block_ids() == set(), "the report drained twice"
    _sending, recving = disk_restore.worker_get_finished(worker, (set(), None))
    assert recving == {"req-1"}


def test_a_hostile_worker_never_raises(armed):
    """RESTORED. It went red because the fake was hostile about the WRONG FIELD.

    It raised from `_connector_metadata` and then expected `worker_bind` to fail -
    but bind reads `load_paths`, `load_cpu_blocks`, `load_event`, and the identity
    seam; a missing `load_paths` is an ordinary step with no disk work and bind
    correctly answers `ok=True, queued=0`. Expecting a failure there would have
    pinned a bug, not caught one. The hostile attributes are now the ones each path
    actually reads, which is what makes the containment real.
    """
    class Exploding:
        @property
        def _connector_metadata(self):
            raise RuntimeError("no metadata for you")

    class ExplodingStep:
        @property
        def load_paths(self):
            raise RuntimeError("no paths for you")

    class ExplodingIndex:
        @property
        def _pending_load_event_indices(self):
            raise RuntimeError("no indices for you")

    assert disk_restore.worker_before_load(Exploding())["ok"] is False
    assert disk_restore.worker_bind(Exploding(), ExplodingStep())["ok"] is False
    assert disk_restore.worker_get_finished(ExplodingIndex(), (set(), None)) \
        == (set(), None)
    assert disk_restore.worker_before_load(ExplodingIndex())["ok"] is True
    assert disk_restore.counters()["worker_errors"] >= 2


def test_every_counter_has_a_unit():
    """RESTORED, and it caught a real (if small) gap: `witness_emissions` had no
    unit, so the payload carried a bare number under a `units` map whose whole
    purpose is that no number in it is ever bare.

    It walks BOTH views: `counters()` (the module's own numbers) and
    `scheduler_section()` (the one-call payload), because the section publishes keys of
    its own - `offer_block_cap_blocks`, `offer_block_cap_tokens`, `ranks_armed`,
    `ranks_seen`, the three `suppressed_*` - and a
    number that reached the payload through a pass-through is exactly as bare as one
    that came from `_state`.

    The applied-read evidence is pinned here by name (`reads_applied`,
    `disk_reads_applied`, `ranks_reads_applied`, `loads_released`, and the three
    `witness_*`), because those keys are the reason
    the payload exists for an experiment: `restorations` reported CONFIRMED through the
    2026-10-05 livelock while `reads_applied` sat at 0, and the direct counter was not
    in the payload at all. A pass-through that silently loses its unit is the first
    step back to that.
    """
    APPLIED_READ_KEYS = ("reads_applied", "disk_reads_applied",
                         "ranks_reads_applied", "loads_released",
                         "authority_lookups", "witness_verdict",
                         "witness_reason", "witness_kind")
    counters = disk_restore.counters()
    units = disk_restore.counters_view(dict(counters))["units"]
    for name, value in counters.items():
        if isinstance(value, int) and not isinstance(value, bool):
            assert name in units, name
    for name in APPLIED_READ_KEYS:
        assert name in counters, name
        assert name in units, f"{name} reaches counters() with no unit"

    view = disk_restore.scheduler_section()
    section_units = view["units"]
    for name, value in view.items():
        if isinstance(value, int) and not isinstance(value, bool):
            assert name in section_units, name
        if name == "index":
            for inner, field in value.items():
                if isinstance(field, int) and not isinstance(field, bool):
                    assert f"index.{inner}" in section_units, inner
    for name in APPLIED_READ_KEYS:
        assert name in view, f"{name} is in counters() but not in the payload"
        assert name in section_units, f"{name} reaches the payload with no unit"


# --- n: THE SCHEDULER ROLE, the only role that can start a restore -----------
#
# Everything above this line installs into a process that has BOTH tier classes,
# because that is the shape a worker has after the connector imports its modules -
# and a worker cannot make an offer. Which is how the live box came to print four
# `DISK[restore] installed=YES class=SimpleCPUOffloadScheduler+SimpleCPUOffloadWorker`
# lines, none of them from the process that reads the prefix: the suite had never
# asked what an EngineCore does. These tests install into a SCHEDULER-role fake -
# the manager class present, the worker class absent - and ask that question.

@pytest.fixture
def scheduler_role(monkeypatch):
    """A process with the SCHEDULER class and no worker class: an EngineCore.

    The class is the fork-shaped one `test_backport` builds from the fork's REAL
    method text (`fixtures_pr56810`, fingerprint-verified against the fork at HEAD
    10b8e07), registered in `sys.modules` under the fork's manager module name so
    `_class_of` resolves it exactly as the EngineCore does. The worker module is
    removed, because an EngineCore never imports it - and the state that
    `boot_from_connector` writes (an armed tier, an armed guard) is saved and
    restored, so booting here cannot leak a live tier into the next test.
    """
    import test_backport as TB

    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    module = types.ModuleType(disk_restore.SCHED_MODULE)
    module.SimpleCPUOffloadScheduler = cls
    saved = {name: sys.modules.get(name) for name in FORK_MODULES}
    sys.modules[disk_restore.SCHED_MODULE] = module
    sys.modules.pop(disk_restore.WORKER_MODULE, None)
    saved_state = {"backport": dict(backport._state),
                   "guard": dict(guard_mode._state)}
    backport.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    try:
        yield {"cls": cls, "manager": manager, "helpers": TB}
    finally:
        disk_restore.uninstall()
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name, state in (("backport", backport), ("guard", guard_mode)):
            state._state.clear()
            state._state.update(saved_state[name])


def sched_layer_kinds(cls, name: str) -> list:
    """The layer kinds on the method THE SCHEDULER resolves, outermost first."""
    current = getattr(cls, name, None)
    if current is None:
        return []
    layers, _bottom = backport._layers_of(backport._function_of(current))
    return [kind for kind, _wrapper in layers]


def test_the_connector_boot_fits_the_offering_half_in_the_scheduler_role(
        scheduler_role, caplog):
    """THE REGRESSION. A scheduler-role boot must end with the offer wrapped.

    Before the fix, `disk_restore.ensure_installed` was reachable ONLY from
    `engine_side._on_worker_report`, and that hook fires inside the WORKER-side
    `get_handshake_metadata` producer (`cpu_capacity.py`) - so in an EngineCore the
    layer was fitted exactly never, and no offer was ever possible. Four ranks said
    YES; the scheduler said nothing; a 9,216-token prefix recomputed in 2.72 s and
    2.85 s instead of reading ~10 ms per 3,200-token span off NVMe.
    """
    cls, manager, TB = (scheduler_role["cls"], scheduler_role["manager"],
                        scheduler_role["helpers"])
    assert disk_restore.role_here() == disk_restore.ROLE_SCHEDULER
    report = backport.install(cls, TB.our_shape_config(), manager)
    with caplog.at_level(logging.WARNING):
        boot = backport.boot_from_connector(cls, report, manager)

    # THE OFFER IS WRAPPED HERE, and the guard is still outermost.
    kinds = sched_layer_kinds(cls, disk_restore.METHOD_MATCHED)
    assert "restore" in kinds, kinds
    assert kinds[0] == "guard", kinds
    assert disk_restore.sched_fitted_here() is True
    assert disk_restore.fitted_methods_here() == list(disk_restore.SCHED_METHODS)

    # THE BOOT SAYS SO, in its line and in its report, because a verdict that only
    # lives in a dict is a verdict nobody reads at 3 a.m.
    assert boot["restore"]["fitted"] is True
    assert boot["restore"]["methods"] == list(disk_restore.SCHED_METHODS)
    assert "restore=YES" in boot["line"]
    assert "restore_gate=" in boot["line"]
    assert not [item for item in boot["refusals"] if item.startswith("restore:")]

    # THE LINE, at the level the engine's log keeps, naming the ROLE it belongs to.
    lines = [rec.getMessage() for rec in caplog.records
             if "DISK[restore]" in rec.getMessage()
             and rec.levelno >= logging.WARNING]
    assert any("installed=YES" in line and "role=scheduler" in line
               for line in lines), lines


def test_the_restore_wrapper_is_on_the_method_the_live_scheduler_calls(
        scheduler_role, armed):
    """Identity proof: the INSTANCE resolves the wrapper, not just the class dict.

    A wrapper on a method the interpreter never reaches is the whole failure mode,
    so this asserts the object the scheduler call actually binds. Then it drives the
    call for real, with the gate open (the `armed` fixture) and an unusable request:
    the fork's answer must come back byte-identical AND the refusal must be counted,
    which is what makes "live but refusing" distinguishable from "absent".
    """
    cls, manager, TB = (scheduler_role["cls"], scheduler_role["manager"],
                        scheduler_role["helpers"])
    report = backport.install(cls, TB.our_shape_config(), manager)
    backport.boot_from_connector(cls, report, manager)

    attr = cls.__dict__[disk_restore.METHOD_MATCHED]
    assert manager.get_num_new_matched_tokens.__func__ is attr
    assert "restore" in sched_layer_kinds(cls, disk_restore.METHOD_MATCHED)

    assert disk_restore.gate()["ok"] is True
    # THE SWITCHES COME DOWN, and the fork's answer comes back unchanged with NO work
    # done: no journal refresh, no index query, no allocation. `object()` is not even a
    # Request - the only way this can be safe is for the gate to close before the offer
    # path reads anything. The wrapper is provably on the call path (it returned the
    # fork's 4096 through the fitted method) without the restore path having to touch a
    # file. Since 0.2.31 a closed gate is a counted REFUSAL naming the two ops that
    # open it; there is no "nobody asked" category any more, because nothing a client
    # sends can ask.
    disk_restore.set_disk_authority(False)
    disk_restore.set_offer_switch(False)
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    offers_before = disk_restore.counters()["offers"]
    assert manager.get_num_new_matched_tokens(object(), 0) == (4096, True)
    counters = disk_restore.counters()
    assert counters["refusals_total"] == 1, \
        "a closed gate is refused in the open, never silently"
    assert counters["refused"]["offers_disabled"] == 1
    assert counters["offers"] == offers_before

    # Under disk AUTHORITY, a manager the path cannot read is refused IN THE OPEN,
    # counted by name. Same wrapper, other side of the switch.
    class Hostile:
        block_size = None
        hash_block_size = None
        block_hashes = []
        num_tokens = 9999
        request_id = "req-hostile"

    before = disk_restore.counters()["refusals_total"]
    assert manager.get_num_new_matched_tokens(Hostile(), 0) == (4096, True)
    counters = disk_restore.counters()
    assert counters["refusals_total"] == before + 1
    assert counters["last_refusal"]
    assert counters["offers"] == 0


def test_an_install_that_finds_no_target_method_is_counted_and_logged(caplog):
    """A fork rename must be a NO with a counter and a line - never a YES.

    The fake is built by `fakes_debug.fork_class` from source that renames the ONE
    method the offer rides, using the fork's real names for the other two: exactly
    the shape a fork update has. `install()` used to return `installed: False` with
    the reason in a dict and nothing else; if that ever degrades back to a silent
    success, this is the test that fails.
    """
    renamed = '''
class SimpleCPUOffloadScheduler:
    """The fork, one rename away from the shape this module expects."""

    def lookup_offload_hit(self, request, num_computed_tokens):
        return 0, False

    def build_connector_meta(self, scheduler_output=None):
        return "METADATA"

    def update_connector_output(self, connector_output):
        return None
'''
    saved = sys.modules.get(disk_restore.SCHED_MODULE)
    fakes_debug.fork_class(disk_restore.SCHED_MODULE,
                           disk_restore.SCHED_CLASS, renamed, register=True)
    try:
        with caplog.at_level(logging.WARNING):
            report = disk_restore.install()
        assert report["installed"] is False, report
        counters = disk_restore.counters()
        assert counters["install_refusals"] >= 1
        # A REFUSAL, not a deferral: the class is right here and the method the
        # offer rides is not on it. That is the case that must never be silent.
        assert counters["install_deferrals"] == 0
        assert f"{disk_restore.SCHED_CLASS}.{disk_restore.METHOD_MATCHED}" \
            in counters["missing_methods"]
        loud = "\n".join(rec.getMessage() for rec in caplog.records
                         if rec.levelno >= logging.WARNING)
        assert "DISK[restore-install]" in loud
        assert disk_restore.METHOD_MATCHED in loud
        assert disk_restore.sched_fitted_here() is False
    finally:
        if saved is not None:
            sys.modules[disk_restore.SCHED_MODULE] = saved
        else:
            sys.modules.pop(disk_restore.SCHED_MODULE, None)


class _NoPolicyHolder:
    def ensure_loaded(self):
        return {"ok": False, "error": "no policy in this test", "traceback": None}

    @property
    def decide(self):
        return None


def test_the_snapshot_payload_carries_the_scheduler_restore_section():
    """"Why is there no restore?" must be answerable from ONE payload.

    Before this, a recursive walk of `/snapshot` and `/disk` found ZERO keys matching
    `restore`: `section()` existed but nothing surfaced it from the scheduler
    process, and the worker-only contribution seam was the only publisher. The
    diagnosis took four rounds of log archaeology instead of one HTTP call.
    """
    engine = types.SimpleNamespace(scheduler=None)
    payload = snapshot.snapshot_for_engine(engine, _NoPolicyHolder())
    view = payload["disk_restore"]
    assert payload["disk_restore_installed"] is view["layer_installed_here"]
    for key in ("layer_installed_here", "role", "class_here", "methods_expected",
                "methods_here", "methods_missing", "install_refusals",
                "install_deferrals", "install_refusal_line", "gate", "offers",
                "tokens_offered", "blocks_offered", "refused", "refusal_reasons",
                "index", "ranks_proven", "digests_expected", "why_no_restore",
                "units", "boot_line",
                # THE POSTURE, AS FIELDS (the two boot switches, the one-word verdict
                # they collapse to, their lifetimes, the armed-rank split and the
                # fuse): the section used to answer "what happened" and never "what
                # this boot is allowed to do", which was only printed in `boot_line`.
                # `scope` and `marker_key` are GONE with the marker they described.
                "disk_authority", "serving_as", "per_boot", "survives_restart",
                "disk_authority_changes", "offer_switch_changes",
                "offers_enabled", "offer_block_cap_blocks", "offer_block_cap_tokens",
                "offer_block_cap_block_size",
                "offer_block_cap_block_size_source", "offer_block_capped",
                "ranks_armed",
                "ranks_seen", "suppressed_hits", "suppressed_spans",
                "suppressed_requests", "suppressions"):
        assert key in view, key
    for retired in ("scope", "marker_key", "flagged", "skipped_no_marker",
                    "requests_seen",
                    # THE RETIRED TOKEN CAP (0.2.32). `offer_cap` advertised the fixed
                    # 32,000-token per-offer ceiling this release deletes; a payload
                    # field with that name and no bound behind it is a number a
                    # dashboard will keep plotting after it stops meaning anything.
                    "offer_cap", "offer_cap_source", "offer_cap_default",
                    "offer_cap_configured"):
        assert retired not in view, retired
    # EVERY OFFER REFUSAL, counted AND spelled out: the closed vocabulary is the
    # whole answer, and a name without a sentence is a name nobody can act on.
    assert set(view["refused"]) == set(disk_restore.OFFER_REFUSALS)
    for name in disk_restore.OFFER_REFUSALS:
        assert isinstance(view["refused"][name], int)
        assert len(view["refusal_reasons"][name]) > 20, name
        assert "undefined reason" not in view["refusal_reasons"][name], name
    # EVERY FIELD WITH A NUMBER OWNS A UNIT (the repo convention, and the only way
    # `blocks_offered` cannot be read as bytes, seconds or requests).
    units = view["units"]
    for key in ("offers", "tokens_offered", "blocks_offered", "ranks_proven",
                "digests_expected", "install_refusals", "install_deferrals",
                "index.entries", "index.bytes_indexed", "proof_reads",
                "deadline_seconds"):
        assert key in units and units[key], key
    # The index is reported WITH ITS SOURCE, because an empty index and an index
    # read from the wrong directory look identical without it.
    assert "entries" in view["index"] and "source" in view["index"]
    # And the one field that answers the question outright.
    assert isinstance(view["why_no_restore"], str) and view["why_no_restore"]
    assert "not fitted" in view["why_no_restore"]


def test_the_why_no_restore_walks_the_chain_in_order(monkeypatch, armed):
    """The verdict names the FIRST missing link, not the last one to be checked.

    Fitted + gate open + an empty index is "nothing is known to be on disk", not
    "no worker proof" (which is also true) - the order is the diagnosis.
    """
    saved = sys.modules.get(disk_restore.SCHED_MODULE)
    fakes_debug.fork_class(disk_restore.SCHED_MODULE, disk_restore.SCHED_CLASS,
                           FORK_SCHED_SOURCE, register=True)
    try:
        assert disk_restore.install()["installed"] is True
        assert disk_restore.gate()["ok"] is True
        view = disk_restore.scheduler_section()
        assert view["layer_installed_here"] is True
        assert "index holds no identities" in view["why_no_restore"]
        fill(disk_spill.disk_root(), blocks=2)
        view = disk_restore.scheduler_section()
        assert view["index"]["entries"] == 6      # 2 blocks x 3 cacheable groups
        assert view["ranks_proven"] == 1          # the `armed` fixture proves them
        assert "no offer has been made" in view["why_no_restore"]
        assert view["index"]["source"]
    finally:
        if saved is not None:
            sys.modules[disk_restore.SCHED_MODULE] = saved
        else:
            sys.modules.pop(disk_restore.SCHED_MODULE, None)


def test_the_why_no_restore_names_a_journal_that_could_not_be_written(monkeypatch,
                                                                       armed):
    """An empty index now says WHICH of its two causes it is.

    "entries: 0" was the whole 2026-10-05 story, and it explained nothing. Two
    causes produce it and both are now in the first sentence an operator reads: the
    journal directory cannot be written at all (so nothing this boot spills can
    ever be indexed), or lines could not be recorded (so files exist that the index
    cannot name). A counter three payloads away is not a diagnosis.
    """
    saved = sys.modules.get(disk_restore.SCHED_MODULE)
    fakes_debug.fork_class(disk_restore.SCHED_MODULE, disk_restore.SCHED_CLASS,
                           FORK_SCHED_SOURCE, register=True)
    try:
        assert disk_restore.install()["installed"] is True
        assert disk_restore.gate()["ok"] is True
        disk_index.reset_state_for_tests()

        # (a) the directory itself cannot be written: named as such, errno and all.
        blocker = os.path.join(armed, "blocker")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("not a directory")
        monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR,
                           os.path.join(blocker, "journal"))
        assert disk_index.journal_append("+", "a3f0b7c1d4e5f607", 10,
                                        3)["ok"] is False
        view = disk_restore.scheduler_section()
        assert view["index"]["journal_directory_writable"] is False
        assert view["index"]["journal_write_errors"] == 1
        why = view["why_no_restore"]
        assert "JOURNAL DIRECTORY CANNOT BE WRITTEN" in why, why
        assert "NotADirectoryError" in why, why

        # (b) the directory is fine and a LINE failed: the count and its reason.
        disk_index.reset_state_for_tests()
        journal = os.path.join(armed, "journal")
        os.makedirs(os.path.join(journal,
                                 f"{disk_index.JOURNAL_PREFIX}PP0_TP0"
                                 f"{disk_index.JOURNAL_SUFFIX}"), exist_ok=True)
        monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, journal)
        assert disk_index.journal_append("+", "a3f0b7c1d4e5f607", 10,
                                        3)["ok"] is False
        view = disk_restore.scheduler_section()
        assert view["index"]["journal_directory_writable"] is True
        assert view["index"]["journal_write_errors"] == 1
        why = view["why_no_restore"]
        assert "journal line(s) could not be recorded" in why, why
        assert "IsADirectoryError" in why, why
    finally:
        if saved is not None:
            sys.modules[disk_restore.SCHED_MODULE] = saved
        else:
            sys.modules.pop(disk_restore.SCHED_MODULE, None)


def test_the_disk_verb_answers_about_restore_too():
    """`/disk` is where the operator asks about the SSD tier - both directions."""
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    assert "restore" in answer
    assert "layer_installed_here" in answer["restore"]
    assert "why_no_restore" in answer["restore"]


def test_the_boot_line_names_the_role_and_the_methods():
    """A line printed by four ranks and one scheduler must say which printed it."""
    line = disk_restore.boot_line()
    assert "role=" in line and "methods=" in line
    labels = [part.split("=")[0] for part in line.split(" ") if "=" in part]
    assert len(labels) == len(set(labels)), labels


def test_adopt_if_missed_is_the_scheduler_door_and_arms_nothing(monkeypatch):
    """The backstop door: fitting is not arming, and a no-class process is cheap.

    With nothing imported it must return without calling `install()` at all (the
    snapshot verb calls this every few seconds), and it must never touch the gate.
    """
    saved = sys.modules.pop(disk_restore.SCHED_MODULE, None)
    saved_worker = sys.modules.pop(disk_restore.WORKER_MODULE, None)
    calls = []
    monkeypatch.setattr(disk_restore, "install",
                        lambda refit=False: calls.append(refit) or {})
    try:
        report = disk_restore.adopt_if_missed()
        assert report["installed"] is False
        assert calls == []
        assert disk_restore.gate()["ok"] is False
    finally:
        if saved is not None:
            sys.modules[disk_restore.SCHED_MODULE] = saved
        if saved_worker is not None:
            sys.modules[disk_restore.WORKER_MODULE] = saved_worker


def test_the_fitted_wrapper_actually_offers_in_the_scheduler_role(armed):
    """END TO END: a scheduler-role process, wrapped, OFFERS disk tokens.

    The two tests above prove the layer is fitted and reachable; this one proves
    what it is for. A class built from the fork's own three entry points (answering
    the way the fork does when host RAM finds nothing), installed in a process that
    has the manager class and no worker class, then driven the way the scheduler
    drives it: the answer comes back RAISED by the spilled span - the behaviour the
    live box never produced, because in the one process where this could happen
    nothing was ever fitted.
    """
    saved = sys.modules.get(disk_restore.SCHED_MODULE)
    cls, _module = fakes_debug.fork_class(disk_restore.SCHED_MODULE,
                                          disk_restore.SCHED_CLASS,
                                          FORK_SCHED_SOURCE, register=True)
    try:
        assert disk_restore.install()["installed"] is True
        assert disk_restore.role_here() == disk_restore.ROLE_SCHEDULER

        class Live(cls, Manager):
            """The fork's class, an instance carrying the manager's real fields."""

        fill(armed, blocks=3)
        live = Live()
        assert live.get_num_new_matched_tokens(Req(blocks=3), 0) == (3 * BLOCK, True)
        counters = disk_restore.counters()
        assert counters["offers"] == 1
        assert counters["offered_tokens"] == 3 * BLOCK
        assert counters["blocks_offered"] == 3
        # ...and the section reports it, so the same fact is one HTTP call away.
        view = disk_restore.scheduler_section()
        assert view["layer_installed_here"] is True
        assert view["role"] == disk_restore.ROLE_SCHEDULER
        assert view["why_no_restore"].startswith("restore is live")
    finally:
        if saved is not None:
            sys.modules[disk_restore.SCHED_MODULE] = saved
        else:
            sys.modules.pop(disk_restore.SCHED_MODULE, None)
