r"""The shadow verifier: does the GPU still hold what the arena captured?

`disk_shadow` reads a completed store's GPU bytes and compares them against the blob
the spill path captured for the same block, so this file is ordered by what each
failure would cost on the live box:

1. OFF IS OFF. The default posture reads no device memory at all, and "off" is proved
   by a copy seam that COUNTS every attempt rather than by a counter agreeing with
   itself. A feature that is merely quiet is not a feature that is off.
2. THE COMPARISON MEANS SOMETHING. An equal block is counted equal; a disagreement is
   counted once and CLASSIFIED, and the classification is the whole point: a GPU block
   the engine rewrote between the two reads is `mismatch_gpu_changed_on_recheck`, a
   block that held still and still differs is `mismatch_stable`. Collapsing those into
   one "corruption" counter would make the number an operator acts on the number he
   cannot act on.
3. IT COSTS NOTHING IT DID NOT SAY IT WOULD. The copy runs on the engine's step thread,
   so the per-bind-window cap, the interval throttle and the sampler are each proved to
   do their job - and the interval throttle is proved to COUNT rather than to sleep.
4. IT CHANGES NOTHING. The spill lands the same bytes with the shadow on as with it
   off, no file appears that was not there before, and the bytes it reads are
   unmodified. Every fault - no GPU caches, a device error, a width disagreement, a
   refused spill - is a counter, never an exception on the step thread and never a
   spill that did not happen.
5. THE VIEW NEVER CHANGES SHAPE. With a readable GPU and without one, the published key
   set is identical, and the section stays inside the byte cap the contribution seam
   enforces - a section over the cap is REFUSED, which would make the witness vanish
   exactly when the boot went wrong.

WHAT IS FAKE, STATED WHERE IT CANNOT BE MISSED. This environment has no CUDA device, so
the device->host copy is the ONE expression replaced in these tests
(`_copy_slot_to_host`), and the GPU side is real ctypes memory shaped like the fork's
region tensor. Everything else - the sorted-name order rule, the width check, the
ctypes read of the copied bytes, the sampling, the throttles, the staleness
classification - is the shipped code running. A test that asserted a property of the
copy itself would be asserting a property of the fake, so none does.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time

import fakes_debug as fake
import pytest
from test_disk_spill import SLOT_BYTES, FakeRegion
from test_disk_store import (
    FORK_WORKER_SOURCE,
    LONG_KEY,
    OTHER_KEY,
    SEED_MODULE_SOURCE,
    THIRD_KEY,
    fork_worker_instance,
    metadata,
    metadata_shape,
    register,
    store_files,
)

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    config,
    disk_layout,
    disk_shadow,
    disk_spill,
    disk_store,
    endpoint,
    worker_diag,
)

GPU_BLOCKS = 8


# --- the GPU side of the round trip ------------------------------------------
class FakeGpuRegion:
    """A region tensor on the device, as far as a test can have one.

    Real ctypes memory (so the bytes that get compared are bytes, not a mock's return
    value), the fork's shape, and the four attributes the reader uses. `device` says
    `cuda:0` because that is what a KV region reports, and it is the field
    `_host_landed` reads - the fake has to be able to answer "did this copy arrive in
    host memory?" honestly, because the guard that reads it is what stands between a
    diagnostic and a segmentation fault.
    """

    def __init__(self, blocks, bytes_per_block, *, itemsize=1, device="cuda:0") -> None:
        self.shape = (blocks, bytes_per_block // itemsize)
        self.device = device
        self.nbytes = blocks * bytes_per_block
        self.blocks = blocks
        self.bytes_per_block = bytes_per_block
        self._itemsize = itemsize
        self._buffer = ctypes.create_string_buffer(blocks * bytes_per_block)
        self.set_block_everywhere(b"")

    def set_block_everywhere(self, filler: bytes) -> None:
        base = ctypes.addressof(self._buffer)
        for index in range(self.blocks):
            pattern = self._pattern(index, filler)
            ctypes.memmove(base + index * self.bytes_per_block, pattern,
                           self.bytes_per_block)

    def _pattern(self, index, filler) -> bytes:
        if not filler:
            # Distinct per block and per region, so reading the WRONG block (or the
            # wrong region, or the wrong order) cannot accidentally match.
            return bytes(((index * 11 + offset * 3) % 251)
                         for offset in range(self.bytes_per_block))
        return (bytes(filler) * (self.bytes_per_block // len(filler) + 1))[:
            self.bytes_per_block]

    def set_block(self, index, raw: bytes) -> None:
        ctypes.memmove(ctypes.addressof(self._buffer) + index * self.bytes_per_block,
                       raw[:self.bytes_per_block], self.bytes_per_block)

    def data_ptr(self) -> int:
        return ctypes.addressof(self._buffer)

    def stride(self, axis) -> int:
        return self.bytes_per_block // self._itemsize if axis == 0 else self._itemsize

    def element_size(self) -> int:
        return self._itemsize

    def block_bytes(self, index) -> bytes:
        return ctypes.string_at(self.data_ptr() + index * self.bytes_per_block,
                                self.bytes_per_block)


class HostSlice:
    """What `narrow(0, i, 1).contiguous().cpu()` answers: one block, on the host.

    `reads` counts every address formed through this object. The
    "a copy that never landed is refused before any address is read" test hands out a
    slice whose `data_ptr()` raises, and proves the guard fires by proving that the
    raise was never reached - a SIGSEGV cannot be caught, so the ONLY honest proof is
    that nothing asked for the address.
    """

    def __init__(self, region, index, *, device="cpu", poison=False) -> None:
        self.region = region
        self.index = index
        self.device = device
        self.poison = poison
        self.shape = (1, region.shape[1])
        self.nbytes = region.bytes_per_block
        self.reads = 0

    def data_ptr(self) -> int:
        self.reads += 1
        if self.poison:
            raise AssertionError("a device address was handed to a host read")
        return self.region.data_ptr() + self.index * self.region.bytes_per_block

    def stride(self, axis) -> int:
        return self.region.stride(0) if axis == 0 else self.region.element_size()

    def element_size(self) -> int:
        return self.region.element_size()


class CopySeam:
    """The stand-in for the only expression that touches a GPU tensor.

    It records every call as `(region name, block)` so the ORDER the regions are read
    in is an observable, and it can be told to fail, to hand back a slice that never
    reached the host, or to hand back memory whose contents change on a given call -
    which is how "the engine reused this block under us" is made to happen on cue.
    """

    def __init__(self, gpu_caches) -> None:
        self.gpu_caches = gpu_caches
        self.calls: list = []
        self.fail_with = None
        self.device = "cpu"
        self.poison = False
        self.churn_on_call = None
        self.churn_to = b"\xAA"
        self.slices: list = []

    def __call__(self, tensor, index):
        name = next((key for key, value in self.gpu_caches.items() if value is tensor),
                    "?")
        self.calls.append((name, int(index)))
        if self.fail_with is not None:
            raise self.fail_with
        if self.churn_on_call is not None and len(self.calls) == self.churn_on_call:
            tensor.set_block(int(index), self.churn_to)
        host = HostSlice(tensor, int(index), device=self.device, poison=self.poison)
        self.slices.append(host)
        return host


# --- the stage: an armed store, a fork-shaped worker, a patched engine ---------
@pytest.fixture
def tier(tmp_path, monkeypatch, reset_dev_state):
    """A configured, ARMED store under this test's directory, with a rank.

    Every file that needs one defines its own `tier` (`test_disk_store.py`,
    `test_disk_boot_default.py`, `test_worker_diag.py`), so this one does too. It sets
    the two env vars that name a LOCATION and a SIZE, arms the gate, and pins the rank;
    the tmpfs refusal is relaxed because pytest's `tmp_path` lives under /tmp.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    root = tmp_path / "kv-store"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(64 * SLOT_BYTES))
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "diag"))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    (tmp_path / "diag").mkdir()
    disk_spill.arm()
    return root


@pytest.fixture
def worker_class():
    """The fork's worker class, installed under the fork's module name, and undone."""
    cls, _module = fake.fork_class(disk_store.TARGET_MODULE,
                                   disk_store.TARGET_CLASS,
                                   FORK_WORKER_SOURCE, register=True)
    try:
        yield cls
    finally:
        disk_store.uninstall()
        sys.modules.pop(disk_store.TARGET_MODULE, None)


# --- driving the real path ---------------------------------------------------
@pytest.fixture(autouse=True)
def patched_engine():
    """The engine the shadow can work on: metadata whose `store_keys` exist.

    Autouse because every comparison needs the patched shape, and the one test that
    wants the unpatched one enters `metadata_shape(False)` inside the fixture's
    context, which is the only way to be sure the next test does not inherit it.
    """
    with metadata_shape(True):
        yield


@pytest.fixture(autouse=True)
def none_hash_seed():
    """The fork's `get_none_hash_seed()`, so the fingerprint builder can finish."""
    saved = sys.modules.get(disk_store.SEED_MODULE)
    module = register(disk_store.SEED_MODULE, SEED_MODULE_SOURCE)
    try:
        yield module
    finally:
        if saved is None:
            sys.modules.pop(disk_store.SEED_MODULE, None)
        else:
            sys.modules[disk_store.SEED_MODULE] = saved


@pytest.fixture
def relaxed(monkeypatch):
    """The shipped sample rate and interval relaxed, WITHOUT arming anything.

    A test of the 1-in-32 sampler wants the shipped value; a test of the staleness
    classifier wants to see TWO comparisons inside one 0.5 s interval, which is not
    something a wall clock can be asked to do. So the limits are separate from the
    enablement: `relaxed` moves the limits and leaves the shadow OFF, and `shadow` is
    `relaxed` plus the environment variable. A test that wants to prove "off is off"
    takes `relaxed` and still sees nothing.
    """
    monkeypatch.setattr(disk_shadow, "SAMPLE_EVERY_N_STORES", 1)
    monkeypatch.setattr(disk_shadow, "MIN_INTERVAL_SECONDS", 0.0)
    return disk_shadow


@pytest.fixture
def shadow(relaxed, monkeypatch):
    """The shadow ON, with the limits relaxed."""
    monkeypatch.setenv(config.ENV_DISK_SHADOW, "1")
    return relaxed


@pytest.fixture
def stage(tier, worker_class, monkeypatch):
    """An armed store, a worker with an arena AND GPU regions, and a counting seam.

    The seam is installed in every test here (shadow ON or OFF): "off means no device
    read" is only proved by a counter that stays at zero, and the counter has to be
    watching.
    """
    assert disk_store.install()["installed"] is True
    arena = {"g0": FakeRegion(GPU_BLOCKS, SLOT_BYTES)}
    gpu = {"g0": FakeGpuRegion(GPU_BLOCKS, SLOT_BYTES)}
    worker = fork_worker_instance(worker_class, arena)
    worker.gpu_kv_caches = gpu
    seam = CopySeam(gpu)
    monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", seam)
    return {"worker": worker, "arena": arena, "gpu": gpu, "seam": seam,
            "root": tier}


def bind_and_complete(stage, *, slots, gpu_blocks, keys, event):
    """One step: the fork's own two methods, with the shadow's own witness inside."""
    step = metadata(store_cpu_blocks=list(slots), store_gpu_blocks=list(gpu_blocks),
                    store_keys=list(keys), store_event=event)
    stage["worker"].bind_connector_metadata(step)
    stage["worker"].delivered = list(stage["worker"].delivered) + [event]
    stage["worker"].get_finished(set())
    return step


def match_the_arena(stage, slot: int, gpu_block: int) -> None:
    """Make the GPU's block and the arena's slot hold the same bytes."""
    stage["gpu"]["g0"].set_block(gpu_block, stage["arena"]["g0"].slot_bytes(slot))


def counters() -> dict:
    return disk_shadow.counters()


# --- 1. OFF IS OFF ------------------------------------------------------------
def test_the_default_reads_no_device_memory_at_all(stage):
    """Requirement 1: default OFF, proved by the copy seam's own call list.

    `counters()` agreeing with itself would prove nothing - the counters are the
    feature's own claim. What is asserted is that the ONE function that touches a GPU
    tensor was never called, through a full bind/complete/spill cycle, and that the
    published view says off and names who decided it (nobody: the default).
    """
    assert disk_shadow.effective_enabled() is False
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert stage["seam"].calls == []
    assert disk_spill.drain()["drained"] is True
    section = disk_shadow.section()
    assert section["enabled"] is False
    assert section["enable_source"].startswith("default (off)")
    assert section["latch"] is False
    assert counters()["stores_seen"] == 0
    assert counters()["gpu_read_ok"] == 0
    # And the tier did its real job while the shadow stayed out of the way.
    assert len(store_files(stage["root"])) == 1


def test_the_environment_arms_it_and_a_typo_never_does(stage, monkeypatch):
    """`VLLM_E2_TIER_DISK_SHADOW` is a grammar, not a presence test.

    `1` arms, `0` refuses, and a value in neither vocabulary (`flase`) refuses - the
    direction that does NOT start reading device memory - while still naming the bad
    value in the view. An operator who typed the flag wrong must be told, and must not
    be told by a device read he did not ask for.
    """
    monkeypatch.setenv(config.ENV_DISK_SHADOW, "flase")
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert stage["seam"].calls == []
    section = disk_shadow.section()
    assert section["enabled"] is False and section["env_recognised"] is False
    assert section["env_raw"] == "flase"
    assert any("unrecognised" in line for line in section["reasons"]), (
        section["reasons"])
    assert disk_spill.drain()["drained"] is True
    assert len(store_files(stage["root"])) == 1


def test_the_environment_is_read_once_and_the_view_says_so(stage, relaxed,
                                                           monkeypatch):
    """The cadence is a decision, published rather than hidden.

    A live boot's workers inherit an environment that does not change, so the value is
    resolved at first use. This pins the consequence an operator would otherwise be
    surprised by - exporting the variable after the process resolved it does nothing,
    and the view says so in `enable_source` instead of leaving him to guess - and the
    lever that DOES work live: the POST latch, in both directions.
    """
    assert disk_shadow.enable_state()["env_enabled"] is False
    monkeypatch.setenv(config.ENV_DISK_SHADOW, "1")
    assert disk_shadow.enable_state()["env_enabled"] is False
    assert "read once" in disk_shadow.enable_state()["source"]
    assert disk_shadow.effective_enabled() is False
    assert stage["seam"].calls == []

    armed = disk_shadow.set_enabled(True)
    assert armed["latch"] is True and armed["enabled"] is True
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert stage["seam"].calls == [("g0", 5)]
    assert disk_shadow.counters()["equal"] == 1
    assert disk_shadow.set_enabled(False)["enabled"] is False


# --- 2. THE COMPARISON --------------------------------------------------------
def test_a_block_that_matches_is_counted_equal_and_nothing_else_moves(shadow, stage):
    """Requirement 3: the happy path, counted in exactly one place.

    The bytes compared are the GPU's own (read out of real memory by the same
    `ctypes.string_at` the spill path uses) against the arena blob the spill request
    captured, and the ONLY counter that moves is equal/checked.
    """
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    seen = counters()
    assert seen["checked"] == 1 and seen["equal"] == 1
    assert seen["mismatch"] == 0 and seen["mismatch_stable"] == 0
    assert seen["mismatch_gpu_changed_on_recheck"] == 0
    assert seen["gpu_read_ok"] == 1 and seen["errors"] == 0
    assert stage["seam"].calls == [("g0", 5)], stage["seam"].calls
    assert disk_shadow.section()["last_mismatch"] is None


def test_a_disagreement_is_counted_once_and_classified(shadow, stage):
    """A GPU that HELD STILL and still differs is the layout candidate.

    The re-read is the discriminator: the same bytes twice means the block did not
    move under us, so what is left is a genuine question about the spill. `offset`
    says where they first differ, which is what turns "the tier is wrong" into "the
    first region's width is wrong".
    """
    # The GPU's block 6 vs the arena's slot 5: different bytes, and stable.
    bind_and_complete(stage, slots=[5], gpu_blocks=[6], keys=[LONG_KEY], event=3)
    seen = counters()
    assert seen["mismatch"] == 1 and seen["checked"] == 1 and seen["equal"] == 0
    assert seen["rechecks"] == 1
    assert seen["mismatch_stable"] == 1
    assert seen["mismatch_gpu_changed_on_recheck"] == 0
    assert seen["mismatch_unrechecked"] == 0
    last = disk_shadow.section()["last_mismatch"]
    assert last["category"] == "mismatch_stable"
    assert last["first_difference"] == 0            # different from the first byte
    assert last["scanned_bytes"] > 0
    # The GPU was read TWICE - the original and the recheck - and nothing else moved.
    assert stage["seam"].calls == [("g0", 6), ("g0", 6)]
    assert seen["errors"] == 0


def test_a_gpu_rewritten_under_us_is_staleness_not_a_fault(shadow, stage):
    """Requirement 4, the case a single corruption counter would get WRONG.

    A completed store's GPU block is the engine's to reuse. Here the block changes
    BETWEEN the two reads - which is exactly what a reuse looks like from the outside -
    and the answer must be `mismatch_gpu_changed_on_recheck`, not a number an operator
    would escalate. If this classification ever collapses into `mismatch_stable`, the
    witness stops being able to tell a race from a bug.
    """
    stage["seam"].churn_on_call = 2                 # the re-read sees new bytes
    bind_and_complete(stage, slots=[5], gpu_blocks=[6], keys=[LONG_KEY], event=3)
    seen = counters()
    assert seen["mismatch"] == 1
    assert seen["mismatch_gpu_changed_on_recheck"] == 1
    assert seen["mismatch_stable"] == 0
    assert disk_shadow.section()["last_mismatch"]["gpu_changed_on_recheck"] is True


def test_a_mismatched_block_that_is_bound_again_is_counted_as_reused(shadow, stage):
    """The secondary reuse witness, with a name that says what it proves.

    A block that mismatched and is named by a LATER bind has demonstrably been handed
    back out. It does not prove the rewrite happened before the read - only that the
    block was in play - which is why it is a separate counter and not the verdict.
    """
    bind_and_complete(stage, slots=[5], gpu_blocks=[6], keys=[LONG_KEY], event=3)
    assert counters()["mismatch_rebound_later"] == 0
    assert counters()["mismatch"] == 1
    # A LATER bind names the block again. It matches this time, so the only new fact is
    # the reuse itself - which is what makes this a witness and not a second mismatch.
    match_the_arena(stage, 4, 6)
    bind_and_complete(stage, slots=[4], gpu_blocks=[6], keys=[OTHER_KEY], event=4)
    assert counters()["mismatch_rebound_later"] == 1
    assert counters()["mismatch"] == 1
    # One witness per mismatch: the id leaves the ledger the moment it is witnessed, so
    # naming the same block again cannot invent a second piece of evidence.
    match_the_arena(stage, 3, 6)
    bind_and_complete(stage, slots=[3], gpu_blocks=[6], keys=[THIRD_KEY], event=5)
    assert counters()["mismatch_rebound_later"] == 1


def test_the_regions_are_read_in_sorted_name_order_not_insertion_order(shadow, stage):
    """THE ORDER RULE, asserted explicitly.

    Five regions inserted in an order no reader would guess, and the copy seam is asked
    for them in `sorted(name)` order - the same rule `read_slot_bytes` sums in. Two of
    the names are the pair that makes the trap real: `layers.10` sorts BEFORE
    `layers.2`, so a reader that summed in insertion order (`DiskBackend
    ._tensor_names`, the 0.2.8 bug) produces a different slot from one that sorted, and
    no single-region boot can ever show it. Both halves are asserted: the order the
    regions are READ in, and the order the module PUBLISHES in the rank file, which is
    the only answer a reader has to "which order did you sum in?".
    """
    stage["gpu"].clear()
    inserted = ("zeta", "model.layers.10.self_attn.k", "alpha",
                "model.layers.2.self_attn.k", "mid")
    for name in inserted:
        stage["gpu"][name] = FakeGpuRegion(GPU_BLOCKS, SLOT_BYTES // 3)
    # THE TRAP, stated so a rename cannot quietly defuse it: these two orders differ.
    assert list(inserted) != sorted(inserted)
    assert "model.layers.10.self_attn.k" < "model.layers.2.self_attn.k"
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert [name for name, _ in stage["seam"].calls] == sorted(inserted), \
        stage["seam"].calls
    # THE PUBLISHED LIST IS THE SAME ORDER, so a reader can verify the rule from the
    # rank file alone and never has to trust this file.
    regions = disk_shadow.section()["regions"]
    assert regions["names"] == sorted(inserted)
    assert regions["order_rule"] == "sorted(name)" and regions["count"] == 5
    # A width disagreement is the honest answer here: the GPU's five regions do not
    # total the arena slot's single width. It is counted as a LAYOUT answer, and never
    # as a mismatch or as staleness.
    seen = counters()
    assert seen["width_mismatch"] == 1
    assert seen["mismatch"] == 0 and seen["mismatch_stable"] == 0
    assert seen["checked"] == 0


# --- 3. IT CHANGES NOTHING ----------------------------------------------------
def test_the_spill_lands_the_same_bytes_with_the_shadow_on_and_off(stage,
                                                                   relaxed):
    """Requirement 2: run the same step with the witness on and off and diff the store.

    Not "the counters look sane": the ENTIRE store tree, byte for byte, and the spill
    path's own counters. A shadow that changed a spill's timing, ordering, width check
    or queue accounting in any way shows up here as a different file or a different
    number.

    The limits are relaxed but the ENV IS NOT SET, because the first half has to be a
    genuinely off boot rather than an armed one with a counter poked.
    """
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert disk_spill.drain()["drained"] is True
    off_files = dict(store_files(stage["root"]))
    off_counters = dict(disk_spill.counters()["refused"])
    off_queued = disk_store.counters()["spill_queued"]
    assert stage["seam"].calls == []

    disk_shadow.set_enabled(True)
    match_the_arena(stage, 2, 2)
    bind_and_complete(stage, slots=[2], gpu_blocks=[2], keys=[OTHER_KEY], event=4)
    assert disk_spill.drain()["drained"] is True
    assert counters()["gpu_read_ok"] == 1
    assert counters()["equal"] == 1

    on_files = dict(store_files(stage["root"]))
    assert len(on_files) == len(off_files) + 1
    # Every file the off run wrote is byte-identical in the on run.
    assert all(on_files[name] == payload for name, payload in off_files.items())
    assert disk_spill.counters()["refused"] == off_counters
    assert disk_store.counters()["spill_queued"] == off_queued + 1


def test_it_writes_no_file_and_leaves_the_bytes_it_reads_untouched(shadow, stage):
    """Requirement 5, both halves.

    NOTHING IS WRITTEN: the whole temp tree (store, diagnostics, everything pytest made
    for this test) is fingerprinted before and after, including file CONTENTS.
    NOTHING IS MODIFIED: the arena's slot and the GPU's block are compared byte for
    byte afterwards. A comparison that wrote back, truncated the engine's log, or
    advanced a cursor would be caught by the first half; a comparison that scribbled on
    a KV buffer by the second.
    """
    def fingerprint() -> dict:
        root = stage["root"].parent
        found = {}
        for base, _dirs, names in os.walk(str(root)):
            for name in names:
                full = os.path.join(base, name)
                with open(full, "rb") as handle:
                    found[os.path.relpath(full, str(root))] = handle.read()
        return found

    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert disk_spill.drain()["drained"] is True
    arena_before = stage["arena"]["g0"].slot_bytes(5)
    gpu_before = stage["gpu"]["g0"].block_bytes(5)
    before = fingerprint()

    # A second, deliberately MISMATCHING comparison: the failure path gets the same
    # "writes nothing" guarantee as the happy path, and it is the failure path an
    # operator is reading the rank file about.
    bind_and_complete(stage, slots=[4], gpu_blocks=[6], keys=[OTHER_KEY], event=4)
    assert disk_spill.drain()["drained"] is True
    after = fingerprint()

    assert counters()["mismatch"] == 1
    store = str(stage["root"].name)
    diag = "diag"
    added = [name for name in after if name not in before]
    changed = [name for name in before if after[name] != before[name]]
    # EVERY new file is the spill path's own, under the store root; NOTHING new is
    # anywhere else. Every changed file is the spill's journal, whose writer is
    # `disk_spill` - the shadow has no writer, and this is the assertion that says so.
    assert added and all(name.startswith(store + os.sep) for name in added), added
    assert all("kvslot-" in name for name in added), added
    assert all(name.startswith(diag + os.sep) for name in changed), changed
    assert stage["arena"]["g0"].slot_bytes(5) == arena_before
    assert stage["gpu"]["g0"].block_bytes(5) == gpu_before


# --- 4. THE COST IS BOUNDED ---------------------------------------------------
def test_the_step_thread_never_copies_more_than_the_step_budget(shadow, stage,
                                                                monkeypatch):
    """Requirement 7: the cap THROTTLES, it does not queue, defer, or grow.

    One bind window, five sampled stores, a budget of two. Two device copies happen and
    the other three are a counted number - which is what makes the shadow's step cost a
    constant instead of a function of the store rate. A cap that deferred the rest to
    the next step would put the same cost one step later and hide it from this
    counter's name.
    """
    monkeypatch.setattr(disk_shadow, "MAX_BLOCKS_PER_STEP", 2)
    for index in range(5):
        match_the_arena(stage, index, index)
    # ONE bind (one step marker) carrying five keys: the budget is per bind window.
    bind_and_complete(stage, slots=[0, 1, 2, 3, 4], gpu_blocks=[0, 1, 2, 3, 4],
                      keys=[LONG_KEY, OTHER_KEY, THIRD_KEY,
                            LONG_KEY[:-8] + "00000004", LONG_KEY[:-8] + "00000005"],
                      event=7)
    seen = counters()
    assert len(stage["seam"].calls) == 2, stage["seam"].calls
    assert seen["throttled_step"] == 3
    assert seen["stores_seen"] == 5 and seen["sampled_out"] == 0


def test_the_interval_throttle_counts_instead_of_sleeping(shadow, stage,
                                                          monkeypatch):
    """The throttle that protects the step thread must never BLOCK it.

    `MIN_INTERVAL_SECONDS` is a refusal, not a wait: with a 30-minute interval in force
    the second store's comparison is counted and the call returns immediately. A
    throttle that slept would convert a diagnostic into a step-time bug, and the sleep
    would be invisible in every counter here.
    """
    monkeypatch.setattr(disk_shadow, "MIN_INTERVAL_SECONDS", 1800.0)
    match_the_arena(stage, 5, 5)
    started = time.monotonic()
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    bind_and_complete(stage, slots=[4], gpu_blocks=[4], keys=[OTHER_KEY], event=4)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, elapsed            # refused on the spot, never slept on
    seen = counters()
    assert seen["throttled_interval"] == 1
    assert len(stage["seam"].calls) == 1     # only the first store got its copy
    assert seen["checked"] == 1


def test_the_sampler_chooses_one_store_in_n_deterministically(shadow, stage,
                                                              monkeypatch):
    """The sampling is a counter, not a die roll: exactly one store in N is compared.

    `sample_rate` is published so a rank file can be read without knowing this file
    exists, and the arithmetic `stores_seen = sampled_out + everything else` holds -
    the denominator counts every store the module was asked about, or the published
    rate would be unreadable.
    """
    monkeypatch.setattr(disk_shadow, "SAMPLE_EVERY_N_STORES", 2)
    # Isolated from the step budget: with the shipped cap of 2 this would throttle
    # before it finished sampling, which is the OTHER test's claim.
    monkeypatch.setattr(disk_shadow, "MAX_BLOCKS_PER_STEP", 8)
    for index in range(6):
        match_the_arena(stage, index, index)
    bind_and_complete(stage, slots=[0, 1, 2, 3, 4, 5],
                      gpu_blocks=[0, 1, 2, 3, 4, 5],
                      keys=[LONG_KEY, OTHER_KEY, THIRD_KEY,
                            LONG_KEY[:-8] + "00000004",
                            LONG_KEY[:-8] + "00000005",
                            LONG_KEY[:-8] + "00000006"], event=11)
    seen = counters()
    assert seen["stores_seen"] == 6
    assert seen["sampled_out"] == 3
    assert seen["checked"] == 3
    assert disk_shadow.section()["sample_rate"] == pytest.approx(0.5)
    assert len(stage["seam"].calls) == 3


# --- 5. EVERY FAULT IS A COUNTER ----------------------------------------------
def test_a_worker_with_no_gpu_caches_degrades_to_a_counter(shadow, stage):
    """Requirement 8's GPU-missing case: the arena still spills, the witness counts.

    A worker whose `gpu_kv_caches` is not there (or is empty, or is not a mapping) is
    the normal state for a process that never got its tensors. The spill is unaffected
    and the answer is one named counter.
    """
    del stage["worker"].gpu_kv_caches
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    seen = counters()
    assert seen["no_gpu_caches"] == 1
    assert seen["checked"] == 0 and seen["mismatch"] == 0 and seen["errors"] == 0
    assert disk_spill.drain()["drained"] is True
    assert len(store_files(stage["root"])) == 1
    assert any("no gpu_kv_caches" in line for line in disk_shadow.section()["reasons"])


def test_a_device_fault_during_the_copy_is_counted_and_never_raised(shadow, stage):
    """No CUDA driver, an illegal block, a refused copy: one counter each.

    The copy seam raises the way a device does. The step thread must not see it - a
    raise out of `get_finished` is a crashed engine - and the spill, which already
    decided to write, must still write.
    """
    match_the_arena(stage, 5, 5)
    stage["seam"].fail_with = RuntimeError("CUDA driver not initialized")
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    seen = counters()
    assert seen["gpu_read_failed"] == 1
    assert seen["checked"] == 0 and seen["mismatch"] == 0 and seen["errors"] == 0
    assert disk_spill.drain()["drained"] is True
    assert len(store_files(stage["root"])) == 1
    reasons = disk_shadow.section()["reasons"]
    assert any("CUDA driver not initialized" in line for line in reasons), reasons


def test_a_copy_that_never_landed_is_refused_before_any_address_is_read(shadow,
                                                                        stage):
    """The guard against a SEGFAULT, proved by the absence of an address.

    `ctypes.string_at` on a device pointer is not a Python exception - it is a dead
    worker - so the copy's result is checked against its OWN `device` field before a
    single byte is read. The proof is that the poisoned slice's `data_ptr()` was never
    called: no exception was raised, because nothing asked for an address to read.
    """
    stage["seam"].device = "cuda:0"
    stage["seam"].poison = True
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    seen = counters()
    assert seen["gpu_read_failed"] == 1 and seen["gpu_read_ok"] == 0
    assert seen["checked"] == 0 and seen["errors"] == 0
    assert [slab.reads for slab in stage["seam"].slices] == [0]
    assert any("did not land on the host" in line
               for line in disk_shadow.section()["reasons"])


def test_a_refused_spill_has_no_blob_and_is_never_a_mismatch(shadow, stage,
                                                             monkeypatch):
    """The spill said no, so nothing can be compared: a third answer, not a verdict.

    A full queue or a closed rate window leaves the plan's `blob` None. Counting that
    as a mismatch would make a busy boot look corrupt; counting it as an equal would
    make a broken boot look healthy. It gets its own name.
    """
    match_the_arena(stage, 5, 5)
    # THE QUEUE IS FULL: a real production refusal, arriving after `disk_shadow` has
    # decided to compare. (Disarming is NOT this test: `disk_store`'s completion hook
    # never reaches `request_spill` at all, so there is no plan to leave unfulfilled.)
    monkeypatch.setattr(disk_spill, "MAX_QUEUED_BYTES", 0)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    seen = counters()
    assert seen["no_blob"] == 1
    assert seen["checked"] == 0 and seen["mismatch"] == 0 and seen["gpu_read_ok"] == 0
    assert len(store_files(stage["root"])) == 0


def test_a_bind_that_names_no_gpu_blocks_compares_nothing(shadow, stage):
    """An engine that does not name its source blocks: a counter and a reason, once.

    `store_gpu_blocks` is the 0.2.35 field. When it is absent, short of the CPU list,
    or carries Nones, there is no GPU block to read - the normal state on an older
    engine - and the reason says so in the rank file instead of leaving an operator to
    wonder why a booted witness compared nothing.
    """
    bind_and_complete(stage, slots=[5], gpu_blocks=[None], keys=[LONG_KEY], event=3)
    bind_and_complete(stage, slots=[4], gpu_blocks=[], keys=[OTHER_KEY], event=4)
    seen = counters()
    assert seen["bind_missing_gpu_ids"] == 2
    assert seen["no_gpu_block"] >= 1
    assert stage["seam"].calls == []
    assert seen["errors"] == 0
    assert any("GPU block" in line for line in disk_shadow.section()["reasons"])


# --- 6. THE VIEW NEVER CHANGES SHAPE ------------------------------------------
def test_the_published_key_set_is_identical_with_and_without_a_readable_gpu(
        shadow, stage):
    """Requirement 8: a reader diffing two rank files must tell "moved" from "gone".

    The degraded view is built by the same function as the healthy one, so this is not
    a copy of a key list - it is the same call, twice, under two different worlds: a
    worker that can be read from, and one whose GPU side is gone.
    """
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    with_gpu = disk_shadow.section()
    assert with_gpu["regions"]["source"] == "gpu_kv_caches"
    assert with_gpu["regions"]["count"] == 1

    del stage["worker"].gpu_kv_caches
    bind_and_complete(stage, slots=[4], gpu_blocks=[4], keys=[OTHER_KEY], event=4)
    without_gpu = disk_shadow.section()
    assert without_gpu["regions"]["source"].startswith("cpu_kv_caches")

    assert set(with_gpu) == set(without_gpu)
    # And the same is true of the contribution the rank file actually carries.
    assert set(disk_shadow.worker_report()) == set(with_gpu)
    # The regions block keeps its own keys when there is no worker at all (the
    # rank-file writer has none in hand).
    disk_shadow._state["regions"] = None
    assert set(disk_shadow.section()["regions"]) == {
        "order_rule", "count", "names", "truncated", "multi_region",
        "total_bytes_per_block", "widths", "source"}


def test_the_section_stays_inside_the_seam_byte_cap(shadow, stage):
    """A section over `MAX_CONTRIBUTION_BYTES` is REFUSED, not truncated.

    Which means an unbounded payload would DELETE the whole `disk_shadow` section from
    the file an operator is reading - at the moment the boot started misbehaving. This
    measures with THE WRITER'S OWN ENCODING (`worker_diag._dump`), not a prettier one,
    because that is the number the cap is applied to.
    """
    for index in range(disk_shadow.MAX_REASON_SAMPLES * 3):
        disk_shadow._reason(f"tag{index}", "x" * 4000)
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    blob = worker_diag._dump(disk_shadow.section())
    assert len(blob) <= disk_shadow.MAX_SECTION_BYTES, len(blob)
    assert len(blob) < worker_diag.MAX_CONTRIBUTION_BYTES, len(blob)
    section = disk_shadow.section()
    assert len(section["reasons"]) <= disk_shadow.MAX_REASON_SAMPLES
    # A reason is one clipped string, "tag: detail" - the tag is not counted against the
    # clip, which is why the allowance is the clip plus the tag's own room.
    assert all(len(line) <= disk_shadow.MAX_REASON_CHARS + 40
               for line in section["reasons"]), [len(x) for x in section["reasons"]]
    assert section["trimmed"] == []          # this much text alone still fits


def test_the_worst_case_section_loses_its_illustrations_not_its_numbers(shadow,
                                                                        stage,
                                                                        monkeypatch,
                                                                        caplog):
    """The failure this module must never have: a fat section is REFUSED, and with it
    the witness disappears on the boot that most needed it.

    Filled the way a bad boot fills it - every reason category at its cap, the fattest
    region set a real boot can name, every busy counter at six digits, a recorded
    mismatch - and asserted with the writer's own encoder: it fits its budget, the
    COUNTERS are intact, `trimmed` says what was shed, and the KEY SET IS THE SAME at
    every budget, including the shape of `last_mismatch`. A trim step that invented a
    key instead of shedding one would break the last of those and is the bug this
    section of the test exists to catch.
    """
    # A fat reason table, the fattest region set a real boot can name, and a mismatch.
    for index in range(disk_shadow.MAX_REASON_SAMPLES * 3):
        disk_shadow._reason(f"a-fault-category-number-{index}", "x" * 4000)
    # A busy boot: the counters that grow with load at six digits each. Deliberately
    # NOT the mismatch family - this section is also the one that asserts a recorded
    # mismatch, and `mismatch == 1` is only readable against real comparisons.
    for key in ("binds", "gpu_ids_retained", "stores_seen", "sampled_out",
                "throttled_step", "throttled_interval", "no_gpu_block", "no_blob",
                "gpu_read_ok", "retained_dropped", "errors"):
        disk_shadow._count(key, 123_456)
    names = [
        f"model.layers.{index:02d}.self_attn.attn_output_gate"
        for index in range(8)
    ]
    # FRESH dicts for all three views at once - the arena the spill reads, the GPU the
    # seam reads, and the seam's own name lookup, which resolves a region by IDENTITY.
    # The stage's single 4096-byte `g0` is not the eight-region boot this section
    # describes, and a half-swapped set of dicts would compare one boot's arena against
    # another boot's GPU.
    arena = {name: FakeRegion(GPU_BLOCKS, 128) for name in names}
    gpu = {name: FakeGpuRegion(GPU_BLOCKS, 128) for name in names}
    stage["worker"].cpu_kv_caches = arena
    stage["worker"].gpu_kv_caches = gpu
    stage["arena"] = arena
    stage["gpu"] = gpu
    stage["seam"].gpu_caches = gpu
    # A REAL mismatch, not a refused read: `poison` is the fake that refuses to hand a
    # device address to a host read (the SIGSEGV guard, proven in its own test), and a
    # refused read lands in a read-failure category and is NEVER counted as a mismatch.
    # One bit per region: every region contributes to the slot, so the comparison is
    # the eight-region path, not a one-region one.
    for name in names:
        raw = bytearray(arena[name].slot_bytes(5))
        raw[0] ^= 0xFF
        gpu[name].set_block(5, bytes(raw))
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)

    fat = disk_shadow.section()
    raw = worker_diag._dump(fat)
    assert len(raw) <= disk_shadow.MAX_SECTION_BYTES, len(raw)
    assert disk_shadow.MAX_SECTION_BYTES < worker_diag.MAX_CONTRIBUTION_BYTES
    assert fat["trimmed"], fat                # it shed something
    assert fat["section_trimmed"] >= 1
    # NUMBERS SURVIVE: the counters are the reason the section exists.
    for key in disk_shadow._COUNTER_KEYS:
        assert isinstance(fat[key], int), key
    assert fat["mismatch"] == 1 and fat["mismatch_stable"] == 1
    # WHAT A READER ACTS ON IS THE OFFSET, not the bytes, and the module publishes it
    # as `first_difference` (`_classify` writes that key and `_fit` never touches the
    # block) - the same key the thin section's own classification test reads.
    assert fat["last_mismatch"]["first_difference"] == 0
    assert fat["last_mismatch"]["scanned_bytes"] > 0
    # Eight regions is exactly the published cap, so nothing was hidden: a trimmed
    # section says `truncated`, an untrimmed one does not, and the two must not be
    # confusable.
    assert fat["regions"]["count"] == 8 and fat["regions"]["truncated"] is False

    # THE UNTRIMMED REFERENCE SHAPE, from the same state with no budget in force.
    monkeypatch.setattr(disk_shadow, "MAX_SECTION_BYTES", 10 ** 9)
    full = disk_shadow.section()
    assert full["trimmed"] == []
    assert set(fat) == set(full)
    assert set(fat["last_mismatch"]) == set(full["last_mismatch"])

    # THE PRESSURE CASE: a budget below what the fixed keys need, so EVERY declared
    # shed is taken. The section still arrives with the same top-level keys, the same
    # `last_mismatch` keys and the counters as ints - it sheds its illustrations, never
    # its shape and never its numbers - and it says so in the log, because at this
    # point the rank writer refuses the payload and an absent key is silent.
    monkeypatch.setattr(disk_shadow, "MAX_SECTION_BYTES", 512)
    with caplog.at_level(logging.WARNING, logger="disk_shadow"):
        pinched = disk_shadow.section()
    assert set(pinched) == set(full)
    assert set(pinched["last_mismatch"]) == set(full["last_mismatch"])
    for key in disk_shadow._COUNTER_KEYS:
        assert isinstance(pinched[key], int), key
    assert pinched["reasons"] == []
    assert pinched["trimmed"].count("reasons_dropped") == 1
    assert f"regions:{disk_shadow.TRIM_REGIONS_PUBLISHED}" in pinched["trimmed"]
    assert pinched["regions"]["truncated"] is True
    assert len(pinched["regions"]["names"]) == disk_shadow.TRIM_REGIONS_PUBLISHED
    assert pinched["section_trimmed"] >= 2      # two distinct shapes were counted
    assert any("STILL over its byte budget" in record.getMessage()
               for record in caplog.records), [record.getMessage()
                                               for record in caplog.records]
    # THE KEY SET IS THE SAME ONE A THIN SECTION PUBLISHES - trimming never removes a
    # top-level key, which is what keeps "moved vs gone" readable across two files. The
    # reset is safe: the autouse fixture in `conftest.py` restores this module's state
    # at teardown, so a test cannot leak it into the next one.
    disk_shadow.reset_state_for_tests()
    assert set(fat) == set(disk_shadow.section())



def test_the_multi_region_boot_says_so_once_in_the_log(shadow, stage, caplog):
    """A MULTI-REGION boot is news at BOOT, not news at the first mismatch.

    The 0.2.8 order-rule bug needed more than one region to exist, so a boot that has
    them is the boot the shadow exists for - and the notice is one line, not one line
    per step. The published region block carries the count, the ORDER RULE and the
    width, which is everything needed to answer "which order did you sum in?" without
    reading this file.
    """
    stage["gpu"].clear()
    stage["arena"].clear()
    for name in ("a", "b", "c"):
        stage["gpu"][name] = FakeGpuRegion(GPU_BLOCKS, SLOT_BYTES // 3)
        stage["arena"][name] = FakeRegion(GPU_BLOCKS, SLOT_BYTES // 3)
    with caplog.at_level(logging.WARNING, logger="disk_shadow"):
        for event, key in ((3, LONG_KEY), (4, OTHER_KEY)):
            bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[key],
                              event=event)
    lines = [record.getMessage() for record in caplog.records
             if "MULTI-REGION" in record.getMessage()]
    assert len(lines) == 1, lines
    regions = disk_shadow.section()["regions"]
    assert regions["multi_region"] is True and regions["count"] == 3
    assert regions["order_rule"] == "sorted(name)"
    assert regions["total_bytes_per_block"] == 3 * (SLOT_BYTES // 3)


def test_the_retention_holds_ints_and_drops_what_it_cannot_hold(shadow, stage):
    """The bind witness keeps ints, in a bounded FIFO, and says when it drops one.

    The bind wrapper in `disk_store` deliberately retains neither tensors nor the
    metadata object; this witness is added to the same path, so the rule is asserted
    against the actual retention contents (every key and value an int) and the bound is
    asserted by overfilling it. An eviction costs one comparison, never a wrong answer.
    """
    for event in range(1, disk_shadow.MAX_RETAINED_EVENTS + 5):
        bind_and_complete(stage, slots=[1], gpu_blocks=[1],
                          keys=[LONG_KEY[:-8] + f"0000{event:04d}"], event=event)
        # the spill half forgets its retention; this test is about OURS
        disk_store._retained.clear()
    retained = disk_shadow.view()["retained"]
    assert len(retained) <= disk_shadow.MAX_RETAINED_EVENTS
    assert counters()["retained_dropped"] >= 4
    for event, record in retained.items():
        assert isinstance(event, int)
        for cpu, gpu in record["slots"].items():
            assert isinstance(cpu, int) and isinstance(gpu, int)


# --- 7. THE DOOR AND THE FILE -------------------------------------------------
def test_the_op_is_in_every_table_that_describes_the_surface():
    """`shadow` is an op in all three tables, and POST-only.

    The published ops list and the wire schema's enum are checked against `DISK_OPS`
    rather than against a literal, because the drift this catches is a new op that
    works and is not advertised - or is advertised and cannot be reached. It is listed
    as MUTATING: it writes nothing, but it changes what the process DOES, and a browser
    that can issue a GET must not be able to start a device read.
    """
    assert "shadow" in disk_store.DISK_OPS
    assert "shadow" in disk_store.MUTATING_DISK_OPS
    assert "shadow" in disk_store._DISK_OPS
    spec = endpoint.ARG_SPECS[config.ROUTE_DISK]
    assert set(spec["op"]["values"]) == set(disk_store.DISK_OPS)
    assert spec["enabled"]["kind"] == "bool"
    refused = disk_store.disk_for_engine(None, {"op": "shadow",
                                                "safe_method": True})
    assert refused["ok"] is False and "POST" in refused["reason"]


def test_the_op_reads_by_default_and_arms_only_the_process_answering(shadow, stage,
                                                                    monkeypatch):
    """The read/apply rule, and the scope an operator would otherwise assume.

    `{}` is a question and a present `enabled` is an instruction; a non-boolean leaves
    the switch untouched. And because this module writes no control file, a POST that
    reaches the EngineCore latches the EngineCore - the answer says so in `scope`
    instead of letting him believe he armed the worker.
    """
    monkeypatch.delenv(config.ENV_DISK_SHADOW)
    read = disk_store.disk_for_engine(None, {"op": "shadow"})
    assert read["ok"] is True and read["shadow"]["enabled"] is False
    assert "changed" not in read["shadow"]

    bad = disk_store.disk_for_engine(None, {"op": "shadow", "enabled": "yes"})
    assert bad["ok"] is False and bad["shadow"]["latch"] is False

    armed = disk_store.disk_for_engine(None, {"op": "shadow", "enabled": True})
    assert armed["ok"] is True and armed["shadow"]["enabled"] is True
    assert config.ENV_DISK_SHADOW in armed["shadow"]["scope"]
    assert "worker" in armed["note"]
    # The latch really is the switch: the witness is live in this process now.
    assert disk_shadow.effective_enabled() is True

    off = disk_store.disk_for_engine(None, {"op": "shadow", "enabled": False})
    assert off["shadow"]["enabled"] is False


def test_the_rank_file_carries_the_shadow_verdict_through_the_only_writer(
        tier, worker_class, monkeypatch):
    """End to end: a worker hook, a comparison, and `diagnostics-rank.json`.

    `worker_diag` is the only writer of the file, so the shadow's contribution arrives
    THROUGH it - and the required fields the proxy host's reader needs are asserted
    intact, because a contribution that costs the file a required field would make
    every rank file on the boot `malformed`.
    """
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "0")
    monkeypatch.setenv(config.ENV_DISK_SHADOW, "1")
    monkeypatch.setattr(disk_shadow, "SAMPLE_EVERY_N_STORES", 1)
    monkeypatch.setattr(disk_shadow, "MIN_INTERVAL_SECONDS", 0.0)
    assert worker_diag.install(worker_class)["installed"] is True
    assert disk_store.install()["installed"] is True
    arena = {"g0": FakeRegion(GPU_BLOCKS, SLOT_BYTES)}
    gpu = {"g0": FakeGpuRegion(GPU_BLOCKS, SLOT_BYTES)}
    worker = fork_worker_instance(worker_class, arena)
    worker.gpu_kv_caches = gpu
    seam = CopySeam(gpu)
    monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", seam)
    gpu["g0"].set_block(5, arena["g0"].slot_bytes(5))
    # TWO completed stores, because `worker_diag` writes the file from its own
    # `get_finished` layer, which sits UNDER this one: the document a reader holds
    # describes the boot as of the PREVIOUS completion. That one-behind property is
    # `worker_diag`'s and `test_disk_store.py` pins it for the `disk` section; here it
    # is why the second cycle exists.
    gpu["g0"].set_block(4, arena["g0"].slot_bytes(4))
    for event, key, slot in ((3, LONG_KEY, 5), (4, OTHER_KEY, 4)):
        step = metadata(store_cpu_blocks=[slot], store_gpu_blocks=[slot],
                        store_keys=[key], store_event=event)
        worker.bind_connector_metadata(step)
        worker.delivered = list(worker.delivered) + [event]
        worker.get_finished(set())

    document = json.loads((tier.parent / "diag" / "rank-PP0_TP0.json").read_text())
    assert document["version"] == 1 and document["role"] == "worker"
    assert document["rank"] == "PP0_TP0"
    assert document["arena"]["bytes_total"] > 0        # the reader's fields survive
    assert "disk_shadow" in document["writer"]["contributions"]["written"]
    section = document["disk_shadow"]
    assert section["enabled"] is True
    assert section["equal"] >= 1 and section["mismatch"] == 0
    assert section["regions"]["order_rule"] == "sorted(name)"
    assert section["writer"] == disk_shadow.OWNER_TAG
    assert "QUESTION, not a verdict" in section["note"]


def test_the_copy_seam_narrows_before_it_copies():
    """The real seam, unmonkeypatched, against an object shaped like a tensor.

    The one thing that must be true of the shipped copy is the ORDER of the two
    operations: narrow FIRST, then copy. Copying a region to the host and then
    indexing it would move the whole KV cache across the link to read one block -
    which is not a slow version of the right answer, it is a different program.
    """
    log = []

    class RecordingTensor:
        shape = (GPU_BLOCKS, 4)

        def narrow(self, axis, start, length):
            log.append(("narrow", axis, start, length))
            assert axis == 0 and length == 1, "the copy must take ONE block"
            return RecordingTensor()

        def contiguous(self):
            log.append(("contiguous",))
            return RecordingTensor()

        def cpu(self):
            log.append(("cpu",))
            return "host-block"

    assert disk_shadow._copy_slot_to_host(RecordingTensor(), 7) == "host-block"
    assert log == [("narrow", 0, 7, 1), ("contiguous",), ("cpu",)], log


def test_the_import_time_state_is_inert_and_the_limits_are_the_shipped_ones():
    """The shipped numbers, pinned where a change has to be deliberate.

    These are the limits the safety story is written against: two slot widths of
    device->host traffic per bind window, half a second between episodes, one re-read
    per episode, one store in 32. A change to any of them is a change to what the
    shadow costs a live boot, so it has to break this test and be argued about.
    """
    assert disk_shadow.MAX_BLOCKS_PER_STEP == 2
    assert disk_shadow.MIN_INTERVAL_SECONDS == 0.5
    assert disk_shadow.MAX_RECHECKS_PER_STEP == 1
    assert disk_shadow.SAMPLE_EVERY_N_STORES == 32
    assert disk_shadow.MAX_RETAINED_EVENTS == 16
    assert disk_shadow.MAX_REASON_SAMPLES == 8
    # Nothing at import: no thread, no directory, no latch, no counters moved.
    assert disk_shadow.counters() == dict.fromkeys(disk_shadow._COUNTER_KEYS, 0)
    assert disk_shadow.enable_state()["enabled"] is False
    assert disk_shadow.enable_state()["latch_set"] is False
    assert disk_shadow._state["retained"] == {}
    assert "disk_shadow" in worker_diag.CONTRIBUTION_SECTIONS
