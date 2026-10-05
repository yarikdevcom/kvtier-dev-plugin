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
import sys
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
    """A `KVCacheBlock`: the pool hands these out and only `block_id` matters here."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0


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


class Req:
    """`Request`: the fields the offer path reads, and nothing else."""

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
    """The gate open: root set, spill armed, tier armed, a worker proven.

    The four switches are replaced with the value under test rather than driven
    through their own files, because arm-file and root-validation behaviour belongs
    to `disk_spill`/`config` and is pinned there. What this suite pins is the GATE'S
    OWN composition: all four, or nothing.
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
                                             "ranks": ["PP0_TP0"], "reason": None})
    monkeypatch.setattr(disk_store, "counters", lambda: {
        "geometry_digest": DIGEST, "fingerprint": {"total_bytes_per_block": WIDTH},
        "rank": "PP0_TP0"})
    utils = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    utils.make_block_hash_with_group_id = pack
    utils.resolve_block_hashes = resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, utils)
    return str(root)


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


def merged_cpu(manager):
    """The CPU ids the merged tuple handed the fork, in the fork's group order."""
    return [int(block.block_id)
            for group in manager._pending_cpu_hits["req-1"][0] for block in group]


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
    """Pull the tier arm and no lookup can offer, allocate, or name a file."""
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    manager = Manager(reqs_to_load={"req-1": object()})
    fill(armed, blocks=2)
    assert offer(manager, Req(blocks=2)) == (0, False)
    assert manager.cpu_block_pool.allocated == []
    assert manager._pending_cpu_hits == {}
    counters = disk_restore.counters()
    assert counters["offers"] == 0
    assert sum(counters["refused"].values()) == 0


def test_worker_proof_needs_a_fresh_rank_file_saying_installed(tmp_path,
                                                               monkeypatch):
    """The rank file IS the proof: fresh, and `disk_restore.installed: true`.

    Without it the scheduler would offer CPU slots no worker fills, and the fork's
    `start_load_kv` would DMA whatever those slots happened to contain. That is the
    one silent failure in this design, so it costs a positive proof, not a hope.
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

    write(None)
    assert disk_restore.worker_proof(force=True)["ok"] is False
    write({"installed": False})
    assert disk_restore.worker_proof(force=True)["ok"] is False
    write({"installed": True})
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True and verdict["ranks"] == ["PP0_TP0"]
    write({"installed": True}, ts=time.time() - 3600.0)
    assert disk_restore.worker_proof(force=True)["ok"] is False, \
        "a heartbeat from a worker that may be gone proves nothing"
    write({"installed": True})
    (directory / "rank-huge.json").write_text(
        "x" * (disk_restore.MAX_RANK_FILE_BYTES + 10))
    (directory / "rank-broken.json").write_text("{ not json")
    assert disk_restore.worker_proof(force=True)["ok"] is True, \
        "one unreadable file is not evidence against the readable one"


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
    assert offer(Manager(), Req(blocks=4)) == (2 * BLOCK, True)


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


def test_merged_tuple_is_the_forks_shape_with_our_slots_last(armed):
    """(per-group lists, hit_length, num_computed), ours APPENDED after the RAM hit.

    `update_state_after_alloc` takes `cdiv(num_external, g_block_size)` LEADING
    entries per group, pairs them at `num_computed // g_block_size + i`, and frees
    the remainder. So RAM must stay leading and our slots must follow; the third
    field is the fork's local prefix and must survive untouched, because the pairing
    offsets are computed from it.
    """
    fill(armed, seeds=[3, 4])
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


def test_merge_refuses_a_ram_hit_that_does_not_add_up(armed):
    """A RAM tuple whose per-group length disagrees with its token count: refuse all.

    The merge rests on "leading entries are the RAM hit, one per whole group block".
    If that is false, our slots land at GPU offsets belonging to someone else's
    tokens - a wrong-bytes restore, the one corruption this module must never cause.
    A refusal costs a recompute.
    """
    fill(armed, seeds=[3, 4])
    manager = Manager()
    manager._pending_cpu_hits["req-1"] = ([[Slot(700)] for _ in GROUP_SIZES],
                                          2 * BLOCK, 0)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (2 * BLOCK, False), \
        "the fork's own answer, untouched"
    assert disk_restore.counters()["refused"]["ram_shape"] == 1
    assert manager.cpu_block_pool.freed, "the allocation was given back"
    assert manager.cpu_block_pool.get_num_free_blocks() == 1024


def test_merge_refuses_a_tuple_the_fork_left_behind(armed):
    """Tokens claimed with no `_pending_cpu_hits` tuple: there is nothing to merge."""
    fill(armed, seeds=[3, 4])
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


def test_the_offer_refuses_a_hit_that_is_not_block_aligned(armed):
    """A hash-granular RAM hit interleaves with our blocks inside a group block."""
    fill(armed, blocks=2)
    assert offer(Manager(), Req(blocks=2), 0, (100, False)) == (100, False)
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


def test_completion_counts_a_restoration_only_when_adopted(armed):
    """The counter the witness needs: a disk load the worker REPORTED finished.

    That is the only chain of facts meaning "these tokens were restored instead of
    recomputed": offers, reads and bytes are each true in boots where no request was
    ever served from disk.
    """
    fill(armed, blocks=1)
    output = fakes_debug.Obj(name="KVConnectorOutput", finished_recving={"req-1"},
                             finished_sending=set())
    first = Manager(reqs_to_load={"req-1": object()})
    offer(first, Req(request_id="req-1", blocks=1))
    unadopted = disk_restore.note_reported(first, output)
    assert unadopted["restorations"] == 0
    assert unadopted["released"] == 1, "the ledger still lets it go"

    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=4, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={4: ["req-1"]}))
    report = disk_restore.note_reported(manager, output)
    assert report["restorations"] == 1
    assert disk_restore._loads == {}
    assert disk_restore._slot_paths == {}, "a finished load frees its mappings"


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
                  "offers=", "reads_applied=", "ranks_proven=", "deadline=",
                  "refused="):
        assert field in line, field
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
    offer(Manager(), Req(blocks=1))
    assert disk_restore.witness_verdict()["reason"].startswith(
        "offers_not_adopted")

    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    slots = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=1, cpu=slots,
                                                gpu=list(range(len(slots))),
                                                reqs={1: ["req-1"]}))
    verdict = disk_restore.witness_verdict()
    assert verdict["adopted"] == 1
    assert verdict["reason"].startswith("adopted_never_reported")


def test_witness_is_confirmed_by_a_reported_restoration(armed):
    """CONFIRMED needs a load that reached a worker AND was reported finished.

    That single greppable line is the answer to "does the restore serve requests, or
    did it never run" - and it can only be reached by the real chain, not by an offer
    or a byte count.
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
    assert disk_restore.note_reported(manager, output)["restorations"] == 1
    verdict = disk_restore.witness_verdict()
    assert verdict["verdict"] == "CONFIRMED" and verdict["reason"] == "-"
    assert "verdict=CONFIRMED" in disk_restore.witness_line()
    assert "DISK[restore-witness]" in disk_restore.witness_line()


def test_the_witness_line_is_logged_once_per_verdict(armed, caplog):
    fill(armed, blocks=1)
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
        disk_restore.note_reported(Manager(), output)
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


def test_register_logs_the_restore_hook(monkeypatch, caplog):
    """The boot says whether the restore layer is fitted, in the engine's own log."""
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setattr(engine_side, "attach_methods",
                        lambda cls: {"attached": True, "reason": "ok"})
    monkeypatch.setattr(engine_side, "_engine_core_class", lambda: object)
    # Boot lines are INFO only on the first install in a process; a full-suite run
    # has already installed, so make this registration the first.
    monkeypatch.setitem(engine_side._INSTALLS, "count", 1)
    with caplog.at_level(logging.DEBUG):
        engine_side.register()
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "SSD restore hook" in text
    assert "restore-witness" in text


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
