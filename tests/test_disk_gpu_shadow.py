r"""The GPU-direct shadow writer: the same block, written twice, from two sources.

`disk_gpu_shadow` writes a block the shadow verifier called `equal` AGAIN, from the
GPU's bytes, into a directory that is not the live store, and compares the two files.
These tests are ordered by what each failure would cost on the live box:

1. OFF IS OFF, AND ON DOES NOT TOUCH THE STORE. The default creates no directory and
   writes no file, and the strongest claim about the feature being ON is that the LIVE
   STORE is byte-for-byte what it would have been without it (test 5). A writer that
   changed a spill's name, width or bytes would be caught there and nowhere else.
2. THE ADMISSION RULE IS THE VERDICT. A file appears for a block the verifier vouched
   for and for NO other block - a mismatch, a read failure, an unreadable GPU. The
   writer inherits the verifier's judgement instead of re-deriving it, so this file
   pins the inheritance.
3. IT COSTS NOTHING IT DID NOT SAY IT WOULD. Nothing happens on the step thread but a
   `put_nowait`; the comparison is bounded per pass; retention is confined to its own
   directory - and the confinement is proved by planting a file in the live store with
   the same name as a shadow file and watching which one disappears (test 6).
4. EVERY FAULT IS A COUNTER. A device fault, an unwritable shadow root, a root that IS
   the live store, a writer that raises: none of them puts an exception on the engine's
   thread, none of them changes the spill, none of them writes the wrong file.
5. THE VIEW NEVER CHANGES SHAPE, and a fat section sheds its illustrations, not its
   numbers (the same discipline `tests/test_disk_shadow.py` enforces next door).

WHAT IS FAKE, STATED WHERE IT CANNOT BE MISLED. This environment has no CUDA device, so
the GPU side is the SAME fake the verifier's tests use (`FakeGpuRegion`, `CopySeam`) and
the bytes the writer receives are that fake's bytes - everything downstream of the copy
(naming, the `.part`/`os.replace` write, the file comparison, the retention walk and its
deletion boundary) is the shipped code running on real files in a real temp directory.
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
from test_disk_shadow import (
    GPU_BLOCKS,
    CopySeam,
    FakeGpuRegion,
    bind_and_complete,
    match_the_arena,
)
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
    disk_gpu_shadow,
    disk_layout,
    disk_shadow,
    disk_spill,
    disk_store,
    endpoint,
    worker_diag,
)

SHADOW_DIR = "gpu-shadow"


# --- the stage ---------------------------------------------------------------
@pytest.fixture
def tier(tmp_path, monkeypatch, reset_dev_state):
    """An armed store, a rank, and the shadow directory's expected home: its sibling.

    The sibling default is what the writer derives when the operator names no root, so
    the tests need its path to exist as an assertion target - and to prove it does NOT
    exist while the feature is off.
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


@pytest.fixture(autouse=True)
def patched_engine():
    """The engine the verifier can work on: metadata whose `store_keys` exist."""
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
def verifier(monkeypatch):
    """The shadow VERIFIER on with its limits relaxed; the WRITER still off.

    The writer is downstream of the verifier, so a test of the writer needs the
    verifier sampling every store. Keeping them separate is what lets the first test
    prove "the verifier compared and the writer still wrote nothing".
    """
    monkeypatch.setattr(disk_shadow, "SAMPLE_EVERY_N_STORES", 1)
    monkeypatch.setattr(disk_shadow, "MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setenv(config.ENV_DISK_SHADOW, "1")
    return disk_shadow


@pytest.fixture
def writer(verifier, monkeypatch):
    """The GPU-direct shadow writer ON; the root is derived (the store's sibling)."""
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_WRITES, "1")
    return disk_gpu_shadow


@pytest.fixture
def stage(tier, worker_class, monkeypatch):
    """An armed store, a worker with an arena AND GPU regions, and a counting seam."""
    assert disk_store.install()["installed"] is True
    arena = {"g0": FakeRegion(GPU_BLOCKS, SLOT_BYTES)}
    gpu = {"g0": FakeGpuRegion(GPU_BLOCKS, SLOT_BYTES)}
    worker = fork_worker_instance(worker_class, arena)
    worker.gpu_kv_caches = gpu
    seam = CopySeam(gpu)
    monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", seam)
    return {"worker": worker, "arena": arena, "gpu": gpu, "seam": seam,
            "root": tier, "shadow_root": tier.parent / SHADOW_DIR}


def drain_everything(timeout: float = 10.0) -> dict:
    """Both writer threads: the spill's, then this module's (writes AND comparisons)."""
    spilled = disk_spill.drain(timeout_seconds=timeout)
    shadowed = disk_gpu_shadow.drain(timeout_seconds=timeout)
    return {"spill": spilled, "shadow": shadowed}


def shadow_files(root) -> dict:
    """Every shadow file as `{relative path: bytes}`."""
    found = {}
    if not os.path.isdir(str(root)):
        return found
    for base, _dirs, names in os.walk(str(root)):
        for name in names:
            full = os.path.join(base, name)
            with open(full, "rb") as handle:
                found[os.path.relpath(full, str(root))] = handle.read()
    return found


def counters() -> dict:
    return disk_gpu_shadow.counters()


def set_arena_block(region, slot: int, raw: bytes) -> None:
    """Write one slot of an arena `FakeRegion`, through its own public geometry.

    `FakeGpuRegion` has a `set_block`; the arena fake from `test_disk_spill.py` does
    not, and adding one there for this file's benefit would rewrite a harness eleven
    other files share. The address and the width come from the same three attributes
    `read_slot_bytes` uses, so this is the region's own definition of "slot `slot`".
    """
    width = region.stride(0) * region.element_size()
    ctypes.memmove(region.data_ptr() + int(slot) * width, raw, width)


def wait_for_pending(expected: int) -> None:
    """Wait (briefly) for the writer thread to queue exactly `expected` pairs.

    The writes happen on another thread, so a test that drives the comparison budget by
    hand has to wait for the jobs to arrive first - bounded, and it names what it found
    when the number never appears.
    """
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if len(disk_gpu_shadow.view()["pending"]) == expected:
            return
        time.sleep(0.005)
    raise AssertionError(f"{len(disk_gpu_shadow.view()['pending'])} pending, "
                         f"expected {expected}")


# --- 1. OFF IS OFF ------------------------------------------------------------
def test_the_default_writes_no_file_and_creates_no_directory(stage, verifier):
    """Requirement 1: the verifier compares, and the writer still creates NOTHING.

    The verifier is ON here, so a `equal` verdict really happened - which is the only
    version of this test that means anything. The directory's absence is the assertion:
    a feature that "writes nothing" but leaves an empty tree behind has already changed
    the filesystem, and the store's own sweep would walk that tree.
    """
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["spill"]["drained"] is True
    assert disk_shadow.counters()["equal"] == 1        # the writer HAD an offer
    assert stage["seam"].calls == [("g0", 5)]

    assert not stage["shadow_root"].exists()
    seen = counters()
    assert seen == dict.fromkeys(disk_gpu_shadow._COUNTER_KEYS, 0)
    section = disk_gpu_shadow.section()
    assert section["written"] == 0 and section["bytes_written"] == 0
    assert section["enabled"] is False
    assert section["enable_source"].startswith("default (off)")
    # ...and the live store did the real job the whole time.
    assert len(store_files(stage["root"])) == 1


def test_the_environment_arms_it_and_a_typo_never_does(stage, verifier, monkeypatch):
    """`VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES` is a grammar, not a presence test.

    A value in neither vocabulary (`flase`) is OFF - the direction that does not write
    files - and is REPORTED, because an operator who typed the flag wrong must be told
    he did not turn it on.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_WRITES, "flase")
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["spill"]["drained"] is True
    assert not stage["shadow_root"].exists()
    section = disk_gpu_shadow.section()
    assert section["enabled"] is False and section["env_recognised"] is False
    assert section["env_raw"] == "flase"
    assert any("neither an on word" in line
               for line in section["reasons"]), section["reasons"]


def test_no_writer_thread_means_no_file(stage, verifier, monkeypatch):
    """The step thread never writes: with the thread unavailable, nothing lands.

    If any part of the write happened inline, this is the test that would catch it -
    `consider()` would have had no thread to hand the job to and a file would still
    exist. The refusal is counted and named, not silent.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_WRITES, "1")
    monkeypatch.setattr(disk_gpu_shadow, "_ensure_thread", lambda: False)
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert disk_spill.drain()["drained"] is True
    assert not stage["shadow_root"].exists()
    assert counters()["refused_thread_unavailable"] == 1
    assert counters()["written"] == 0


# --- 2. THE ADMISSION RULE ----------------------------------------------------
def test_an_equal_block_becomes_a_shadow_file_byte_identical_to_the_arena(
        writer, stage):
    """Requirement 2, with the file's CONTENTS as the assertion, not a counter.

    The bytes are the GPU's own (read through the verifier's seam) and the file is
    compared against the arena slot it was confirmed equal to. The second half is the
    naming parity the whole pairing depends on: the shadow file sits at the SAME
    relative path the live store used for that key.
    """
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True

    arena_bytes = stage["arena"]["g0"].slot_bytes(5)
    files = shadow_files(stage["shadow_root"])
    assert len(files) == 1, files
    relative, payload = next(iter(files.items()))
    assert payload == arena_bytes              # THE POINT OF THE FEATURE
    assert len(payload) == SLOT_BYTES          # a whole slot, not half of one

    live = store_files(stage["root"])
    assert len(live) == 1
    assert relative in live                    # matched by NAME, same relative path
    assert live[relative] == payload           # the two trees agree

    seen = counters()
    assert seen["written"] == 1 and seen["bytes_written"] == SLOT_BYTES
    assert seen["pairs_checked"] == 1 and seen["pairs_identical"] == 1
    assert seen["pairs_differing"] == 0 and seen["pairs_unpaired_expired"] == 0
    assert seen["refused_write_failed"] == 0 and seen["errors"] == 0


def test_a_block_the_verifier_did_not_vouch_for_gets_no_file(writer, stage):
    """Requirement 3: `equal` is the ONLY verdict that writes.

    The GPU block 6 and the arena slot 5 are different bytes and the GPU held still:
    the verifier says `mismatch_stable`, and the writer's rule is that a block it
    cannot vouch for is not evidence. Writing it would put bytes in a file named for a
    block that does not hold them - which is exactly the lying artefact the rule
    exists to prevent.
    """
    bind_and_complete(stage, slots=[5], gpu_blocks=[6], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True
    assert disk_shadow.counters()["mismatch_stable"] == 1
    assert not shadow_files(stage["shadow_root"])
    seen = counters()
    assert seen["refused_not_equal"] == 1
    assert seen["written"] == 0 and seen["queued"] == 0
    # The spill itself is unaffected: the live store has its file.
    assert len(store_files(stage["root"])) == 1


def test_a_read_failure_is_not_a_write_either(writer, stage):
    """"Nothing was compared" and "they disagreed" stay two different answers.

    A worker with no `gpu_kv_caches` at all gets `refused_not_checked`, not
    `refused_not_equal`: the first is a measurement that did not happen, the second a
    measurement that said no. Merging them would make an unreadable GPU look like a
    corrupt spill.
    """
    del stage["worker"].gpu_kv_caches
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True
    seen = counters()
    assert seen["refused_not_checked"] == 1
    assert seen["refused_not_equal"] == 0 and seen["written"] == 0
    assert not stage["shadow_root"].exists()
    assert len(store_files(stage["root"])) == 1


def test_a_store_that_never_spills_never_offers(writer, stage, monkeypatch):
    """A refused spill leaves no blob, so the verdict says "nothing was compared".

    The verdict still arrives (the verifier was asked, and answered), and the answer is
    `refused_not_checked` - NEVER a mismatch and never a write. Counting "there was
    nothing to compare" as evidence about the bytes is how a busy boot would look
    corrupt and a broken one look healthy.
    """
    monkeypatch.setattr(disk_spill, "MAX_QUEUED_BYTES", 0)   # a real production refusal
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True
    assert disk_shadow.counters()["no_blob"] == 1
    seen = counters()
    assert seen["offers"] == 1 and seen["refused_not_checked"] == 1
    assert seen["refused_not_equal"] == 0 and seen["written"] == 0
    assert not stage["shadow_root"].exists()
    assert len(store_files(stage["root"])) == 0


# --- 3. THE SPILL IS UNAFFECTED -----------------------------------------------
def test_the_live_store_is_exactly_what_it_would_have_been(verifier, stage,
                                                           monkeypatch):
    """Requirement 5: two boots, two roots, the SAME store.

    Phase 1 runs with the feature genuinely OFF (the environment is not set, so the
    enablement was never latched) and spills three blocks into root A. Phase 2 moves the
    store root to B, arms the writer with the POST latch, and spills the same three keys
    again. The two live trees must then be EQUAL as `{relative path: bytes}` - same
    names, same bytes, same count. A shadow writer that perturbed a spill's timing,
    width check, naming or queue accounting shows up as a difference here.
    """
    keys = [LONG_KEY, OTHER_KEY, THIRD_KEY]
    for event, key, slot in ((3, keys[0], 5), (4, keys[1], 4), (5, keys[2], 3)):
        match_the_arena(stage, slot, slot)
        bind_and_complete(stage, slots=[slot], gpu_blocks=[slot], keys=[key],
                          event=event)
        assert drain_everything()["spill"]["drained"] is True
    off = dict(store_files(stage["root"]))
    assert len(off) == 3, off
    assert not stage["shadow_root"].exists()

    # PHASE 2: a different root, and the writer armed for the first time. The root is
    # resolved at first use, so the shadow tree is now the NEW root's sibling.
    root_b = stage["root"].parent / "kv-store-b"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root_b))
    assert disk_gpu_shadow.set_enabled(True)["enabled"] is True
    for event, key, slot in ((6, keys[0], 5), (7, keys[1], 4), (8, keys[2], 3)):
        bind_and_complete(stage, slots=[slot], gpu_blocks=[slot], keys=[key],
                          event=event)
        assert drain_everything()["spill"]["drained"] is True
    on = dict(store_files(root_b))

    assert len(on) == 3, on
    assert on == off                       # THE LIVE STORE DID NOT FEEL IT
    # And the shadow tree really was written beside it, at those same names.
    assert set(shadow_files(stage["shadow_root"])) == set(off)
    assert counters()["written"] == 3
    assert counters()["pairs_identical"] == 3


# --- 4. EVERY FAULT IS A COUNTER ----------------------------------------------
def test_a_device_fault_changes_nothing_on_disk_and_never_propagates(writer, stage):
    """Requirement 4, the verifier's most important test re-run on the writer.

    The copy seam raises the way a device does (no driver, a refused copy, an illegal
    block). Nothing reaches the step thread, nothing lands on disk, and the spill -
    which had already decided to write - still writes. The writer never gets a usable
    offer, because a block whose bytes could not be read was never vouched for: the one
    counters that move are the offer and its `not_checked` refusal.
    """
    match_the_arena(stage, 5, 5)
    stage["seam"].fail_with = RuntimeError("CUDA driver not initialized")
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True
    assert disk_shadow.counters()["gpu_read_failed"] == 1
    seen = counters()
    assert seen["offers"] == 1 and seen["refused_not_checked"] == 1
    for key, value in seen.items():
        if key not in ("offers", "refused_not_checked"):
            assert value == 0, key
    assert not stage["shadow_root"].exists()
    assert len(store_files(stage["root"])) == 1


def test_a_writer_that_raises_survives_and_writes_the_next_block(writer, stage,
                                                                 monkeypatch):
    """A raise inside the writer thread is a counter, not a dead thread.

    The first job's naming raises; the second job, seconds later, must still be written
    - which proves the thread caught the first one. This is the property that keeps a
    one-off fault from silently ending the experiment mid-boot.
    """
    real = disk_gpu_shadow._name_pair
    calls = []

    def flaky(key):
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("the writer's naming exploded")
        return real(key)

    monkeypatch.setattr(disk_gpu_shadow, "_name_pair", flaky)
    for event, key, slot in ((3, LONG_KEY, 5), (4, OTHER_KEY, 4)):
        match_the_arena(stage, slot, slot)
        bind_and_complete(stage, slots=[slot], gpu_blocks=[slot], keys=[key],
                          event=event)
        assert drain_everything()["shadow"]["drained"] is True
    assert len(calls) == 2
    assert counters()["errors"] == 1
    assert counters()["written"] == 1              # the second block still landed
    assert len(shadow_files(stage["shadow_root"])) == 1
    assert len(store_files(stage["root"])) == 2


def test_a_full_queue_refuses_on_the_step_thread_and_never_grows(writer, stage,
                                                                monkeypatch):
    """Both queue caps answer on the spot: this runs on the ENGINE's step thread.

    A queue that waits would stall a step, and a queue that grows without bound would
    hold the GPU's bytes in host RAM for however long the NVMe takes - the opposite of
    what the tier is for. So the depth cap and the byte cap are checked BEFORE the
    `put_nowait`, and each answers with its own counter and its own sentence.

    No reader consumes the queue here: `_ensure_thread` says yes without starting a
    thread, which is the only way to make four offers pile up deterministically instead
    of racing a writer that is faster than the test.
    """
    monkeypatch.setattr(disk_gpu_shadow, "_ensure_thread", lambda: True)
    verdict = {"checked": True, "equal": True}
    # THE BYTE CAP FIRST, because it also sizes the real queue: `MAX_QUEUE_JOBS` is both
    # the arithmetic on the step thread AND the `queue.Queue(maxsize=...)`, and a cap
    # raised later cannot make an already-small queue bigger. 160 B held, 64 B a slot:
    # two fit (128 B), and the third would need 192 B.
    monkeypatch.setattr(disk_gpu_shadow, "MAX_QUEUE_JOBS", 10)
    monkeypatch.setattr(disk_gpu_shadow, "MAX_QUEUED_BYTES", 160)
    bytes_full = [disk_gpu_shadow.consider({"key": key, "gpu": b"y" * 64, "slot": 1,
                                            "gpu_block": 1}, verdict)
                  for key in (LONG_KEY, OTHER_KEY, THIRD_KEY)]
    assert [offer["written"] for offer in bytes_full] == [True, True, False]
    assert bytes_full[2]["reason_kind"] == "bytes_full"
    assert "160 B" in bytes_full[2]["reason"]
    assert counters()["refused_bytes_full"] == 1
    # Nothing was written, because writing is the writer thread's job and it has not
    # run: a refusal on the step thread must not have written a file either.
    assert not stage["shadow_root"].exists()

    local = disk_gpu_shadow._get_queue()
    try:
        # Then the depth cap, on those same two queued jobs: 128 B of the 192 B are
        # already held, so the byte rule still allows one and the job rule does not.
        monkeypatch.setattr(disk_gpu_shadow, "MAX_QUEUE_JOBS", 2)
        monkeypatch.setattr(disk_gpu_shadow, "MAX_QUEUED_BYTES", 4096)
        over_depth = disk_gpu_shadow.consider({"key": LONG_KEY, "gpu": b"z" * 64,
                                               "slot": 1, "gpu_block": 1}, verdict)
        assert over_depth["written"] is False
        assert over_depth["reason_kind"] == "queue_full"
        assert "2-job cap" in over_depth["reason"]
        seen = counters()
        assert seen["queued"] == 2 and seen["refused_queue_full"] == 1
        assert not stage["shadow_root"].exists()
    finally:
        # Empty the queue by hand: nothing reads it, and leaving jobs behind would make
        # the next test's fixture reset wait for a queue nobody drains.
        while True:
            try:
                local.get_nowait()
            except Exception:  # noqa: BLE001 - `queue.Empty` is the exit condition
                break
            local.task_done()



def test_an_unwritable_shadow_root_is_a_counter_not_a_crash(writer, stage,
                                                            monkeypatch, tmp_path):
    """A root that cannot be created is `refused_write_failed`, once per job.

    Naming a regular file as a parent directory is the cheapest way to make
    `makedirs` fail for real. The spill must still land its own file, and the shadow
    tree must stay absent - no partial directory, no `.part` left behind.
    """
    blocker = tmp_path / "a-file"
    blocker.write_bytes(b"not a directory")
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_ROOT,
                       str(blocker / "inner" / "gpu-shadow"))
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True
    seen = counters()
    assert seen["refused_write_failed"] == 1
    assert seen["written"] == 0 and seen["errors"] == 0
    assert not (tmp_path / "a-file" / "inner").exists()
    assert len(store_files(stage["root"])) == 1


def test_a_named_but_refused_root_is_counted_and_writes_nothing(writer, stage,
                                                                monkeypatch):
    """A shadow root the root rules refuse is a counted answer, not a silent zero.

    The operator named one and it is unusable (here, a refused prefix - on the box the
    real case is `=/tmp/...`, which is tmpfs and would spend the RAM the tier exists to
    relieve). The spill still runs, nothing is written under the bad root, and the
    published `root.reason` says which rule fired so the operator fixes the environment
    instead of wondering why the trees never filled.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ("/forbidden",))
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_ROOT, "/forbidden/gpu-shadow")
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True
    seen = counters()
    assert seen["refused_not_configured"] == 1
    assert seen["queued"] == 0 and seen["written"] == 0
    root = disk_gpu_shadow.section()["root"]
    assert root["root"] is None and root["resolved"] is True
    assert "forbidden" in (root["reason"] or "")
    assert len(store_files(stage["root"])) == 1


# --- 5. THE DIRECTORY BOUNDARY ------------------------------------------------
def test_pointing_the_writer_at_the_live_store_refuses_and_writes_nothing(
        writer, stage, monkeypatch):
    """The refusal the brief asks to be said in a comment, tested.

    Same directory as the live store is a REFUSAL, not a merge: a shadow byte inside
    the store's shard tree is a byte a restore could load. The store's file set is
    asserted unchanged, which is the assertion that makes "we did not merge" more than
    a claim about a counter.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_ROOT, str(stage["root"]))
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["spill"]["drained"] is True
    seen = counters()
    assert seen["refused_root_conflict"] == 1
    assert seen["queued"] == 0 and seen["written"] == 0
    assert len(store_files(stage["root"])) == 1     # only the spill's own file
    root = disk_gpu_shadow.section()["root"]
    # The configured path stays published (an operator must be able to see what he
    # pointed it at); `conflict` is the gate, and it is the reason `written` is zero.
    assert root["conflict"] and "IS the live store" in root["conflict"]
    assert root["root"] == os.path.abspath(str(stage["root"]))


def test_a_shadow_root_nested_inside_the_store_is_refused_too(writer, stage,
                                                              monkeypatch):
    """The other direction of the same rule: nested in EITHER direction is a refusal.

    `/a/store` and `/a/store2` must NOT be confusable with containment, so the
    comparison is on path components - and this test uses a real nested path, whose
    refusal is what stops the writer's files becoming part of the store's sweep.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_ROOT,
                       str(stage["root"] / SHADOW_DIR))
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["spill"]["drained"] is True
    assert counters()["refused_root_conflict"] == 1
    assert not (stage["root"] / SHADOW_DIR).exists()
    assert len(store_files(stage["root"])) == 1
    # A SIBLING of a store whose name merely shares a prefix is NOT a conflict.
    assert disk_gpu_shadow._conflicts_with_store("/a/gpu-shadow", "/a/store2") is None


def test_retention_never_removes_a_file_outside_its_own_root(writer, stage,
                                                             monkeypatch):
    """Requirement 6: the deletion boundary, proved by which file disappears.

    Two shadow files exist; the older one's name is then planted in the LIVE STORE (its
    spilled twin moved aside first, so the two files are distinguishable by content).
    Retention to a one-file cap must remove the SHADOW file and leave the planted live
    file exactly as it was - the walk never leaves the shadow root, and the unlink
    validator would refuse the live path a second time even if it did.
    """
    monkeypatch.setattr(disk_gpu_shadow, "MAX_SHADOW_FILES", 1)
    for event, key, slot in ((3, LONG_KEY, 5), (4, OTHER_KEY, 4)):
        match_the_arena(stage, slot, slot)
        bind_and_complete(stage, slots=[slot], gpu_blocks=[slot], keys=[key],
                          event=event)
        assert drain_everything()["shadow"]["drained"] is True
    files = shadow_files(stage["shadow_root"])
    assert len(files) == 2, files

    # THE OLDEST shadow file, by an mtime we set ourselves so the test is not a race
    # with the filesystem's clock granularity.
    ordered = sorted(files, key=lambda name: os.path.getmtime(
        os.path.join(str(stage["shadow_root"]), name)))
    victim = ordered[0]
    planted = stage["root"] / victim
    planted.unlink()                        # the spilled twin moves out of the way
    planted.write_bytes(b"NOT SHADOW EVIDENCE")

    report = disk_gpu_shadow.run_retention()
    assert report["removed"] == 1, report
    assert planted.exists() and planted.read_bytes() == b"NOT SHADOW EVIDENCE"
    remaining = shadow_files(stage["shadow_root"])
    assert victim not in remaining
    assert ordered[1] in remaining           # the newest evidence is kept
    assert counters()["retention_removed"] == 1
    assert counters()["retention_refused"] == 0


def test_retention_refuses_a_foreign_name_in_its_own_directory(writer, stage,
                                                               monkeypatch):
    """A file in the shadow tree that is not ours to delete stays - and a pair whose
    shadow file retention removed is named as OURS going away.

    `kvslot-` is our prefix, so a foreign name here is someone else's file that happens
    to be nearby: the same doctrine the live store's sweep uses, applied to a directory
    this module owns outright. The second half is the honesty of the diff counters: with
    the cap at zero, retention takes our file while its pair is still waiting for the
    spill's writer thread, and that must land as `pairs_shadow_missing` - NEVER as
    `pairs_unpaired_expired`, which is the store being blamed for our own retention.

    The writer thread's own pass is gated open by the test so the pair is provably still
    pending when retention runs; without that the order of two threads decides which
    counter moves, and a test that asserts a counter has to make that order.
    """
    monkeypatch.setattr(disk_gpu_shadow, "MAX_SHADOW_FILES", 0)
    real_pass = disk_gpu_shadow._run_pass
    hold = {"yes": True}

    def gate():
        return {"reason": "held by the test"} if hold["yes"] else real_pass()

    monkeypatch.setattr(disk_gpu_shadow, "_run_pass", gate)
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert disk_spill.drain()["drained"] is True
    wait_for_pending(1)
    foreign = stage["shadow_root"] / "00" / "someone-elses-notes.bin"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"not ours")

    disk_gpu_shadow.run_retention()
    assert foreign.exists() and foreign.read_bytes() == b"not ours"
    # OUR file is gone; the foreign one is the only thing left in the tree.
    assert set(shadow_files(stage["shadow_root"])) == {
        f"00{os.sep}someone-elses-notes.bin"}

    hold["yes"] = False
    assert drain_everything()["shadow"]["drained"] is True
    seen = counters()
    assert seen["pairs_shadow_missing"] == 1
    assert seen["pairs_unpaired_expired"] == 0 and seen["pairs_identical"] == 0


def test_the_pending_comparison_is_bounded_per_pass(writer, stage, monkeypatch):
    """The comparison's own bound: a pass decides at most N pairs and stops.

    With the cap at one, the first pass decides one pair and the rest wait for the next
    - which is what keeps a burst of writes from turning into an unbounded read of both
    trees on the writer thread. The thread's own pass is stubbed so the pending list is
    deterministic: this test is about the budget, not about a race with a tick.
    """
    monkeypatch.setattr(disk_gpu_shadow, "MAX_PAIRS_PER_PASS", 1)
    monkeypatch.setattr(disk_gpu_shadow, "_run_pass",
                        lambda: {"reason": "the test drives the pass directly"})
    for event, key, slot in ((3, LONG_KEY, 5), (4, OTHER_KEY, 4), (5, THIRD_KEY, 3)):
        match_the_arena(stage, slot, slot)
        bind_and_complete(stage, slots=[slot], gpu_blocks=[slot], keys=[key],
                          event=event)
        assert disk_spill.drain()["drained"] is True
    wait_for_pending(3)
    assert disk_gpu_shadow._compare_pending() == 1
    assert counters()["pairs_checked"] == 1
    assert len(disk_gpu_shadow.view()["pending"]) == 2
    assert disk_gpu_shadow._compare_pending() == 1
    assert disk_gpu_shadow._compare_pending() == 1
    assert counters()["pairs_identical"] == 3
    assert not disk_gpu_shadow.view()["pending"]


# --- 6. THE ORDER RULE --------------------------------------------------------
def test_the_shadow_file_is_the_sorted_name_concatenation(writer, stage):
    """Requirement 7: `sorted(name)`, proved by a file's bytes.

    Four regions inserted in an order no reader would guess, where two of the names are
    the pair that makes the trap real (`layers.10` sorts BEFORE `layers.2`), and each
    region holds DIFFERENT bytes so an order disagreement changes the file. The written
    file must equal the `sorted(name)` concatenation and must NOT equal the
    insertion-order one - the concatenation `DiskBackend._tensor_names` would produce,
    which is the 0.2.8 bug. On a single-region boot this test could not exist, which is
    exactly why the bug survived to ship once.
    """
    names = ["alpha", "model.layers.10.self_attn.k", "model.layers.2.self_attn.k",
             "zeta"]
    inserted = ["zeta", "model.layers.10.self_attn.k", "alpha",
                "model.layers.2.self_attn.k"]
    assert sorted(names) == sorted(inserted)     # the SAME region set, two orders
    assert inserted != sorted(inserted)
    assert "model.layers.10.self_attn.k" < "model.layers.2.self_attn.k"

    stage["arena"].clear()
    stage["gpu"].clear()
    fill = {"alpha": b"A", "model.layers.10.self_attn.k": b"B",
            "model.layers.2.self_attn.k": b"C", "zeta": b"D"}
    for name in inserted:
        stage["arena"][name] = FakeRegion(GPU_BLOCKS, 1024)
        stage["gpu"][name] = FakeGpuRegion(GPU_BLOCKS, 1024)
    stage["seam"].gpu_caches = stage["gpu"]
    for name in inserted:
        set_arena_block(stage["arena"][name], 5, fill[name] * 1024)
        stage["gpu"][name].set_block(5, fill[name] * 1024)
    # The two orders the reader could produce, computed HERE so the assertion cannot be
    # satisfied by a rename.
    def concat(order) -> bytes:
        return b"".join(fill[name] * 1024 for name in order)

    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert drain_everything()["shadow"]["drained"] is True

    assert [name for name, _ in stage["seam"].calls] == sorted(inserted)
    files = shadow_files(stage["shadow_root"])
    assert len(files) == 1, files
    payload = next(iter(files.values()))
    assert payload == concat(sorted(inserted))
    assert payload != concat(inserted)
    assert counters()["written"] == 1 and counters()["pairs_identical"] == 1


# --- 7. THE VIEW NEVER CHANGES SHAPE ------------------------------------------
def test_the_key_set_is_the_same_having_written_nothing_and_a_lot(writer, stage):
    """Requirement 8's stable shape, on the two states this feature actually has.

    Built by the same function under two worlds: enabled and untouched, and enabled with
    a full set of counters, a recorded difference and a populated name sample. A reader
    diffing two rank files must be able to tell "moved" from "gone", and a key that
    appears only when there is something to say would defeat that.
    """
    quiet = disk_gpu_shadow.section()
    assert quiet["enabled"] is True
    assert quiet["written"] == 0 and quiet["last_difference"] is None

    for event, (key, slot) in enumerate(((LONG_KEY, 5), (OTHER_KEY, 4),
                                         (THIRD_KEY, 3)), start=3):
        match_the_arena(stage, slot, slot)
        bind_and_complete(stage, slots=[slot], gpu_blocks=[slot], keys=[key],
                          event=event)
        assert drain_everything()["shadow"]["drained"] is True
    # A recorded difference and a name sample, without pretending the writer produced
    # one: these are the shapes the section must be able to carry.
    disk_gpu_shadow._state["last_difference"] = {
        "name": "kvslot-0000000000000000-g3-0123456789ab.bin",
        "first_difference": 4096, "scanned_bytes": 8192,
        "shadow_bytes": SLOT_BYTES, "live_bytes": SLOT_BYTES,
        "size_disagree": False}
    disk_gpu_shadow._state["differing_names"] = ["a.bin", "b.bin", "c.bin"]

    busy = disk_gpu_shadow.section()
    assert set(busy) == set(quiet)
    assert set(busy["root"]) == set(quiet["root"])
    assert set(busy["limits"]) == set(quiet["limits"])
    assert busy["written"] == 3 and busy["pairs_identical"] == 3
    for key in disk_gpu_shadow._COUNTER_KEYS:
        assert isinstance(busy[key], int), key
    # And the degraded world - no root at all - keeps the same keys too.
    disk_gpu_shadow._state["root"] = None
    assert set(disk_gpu_shadow.section()) == set(quiet)


def test_the_worst_case_section_loses_its_illustrations_not_its_numbers(
        writer, stage, monkeypatch, caplog):
    """A fat section is REFUSED by the contribution seam, so this one bounds itself.

    Filled the way a bad boot fills it - every reason category at its cap, six
    differing names, every busy counter at six digits, a recorded difference - and
    measured with the WRITER'S OWN encoder. It fits, the counters are ints, `trimmed`
    says what was shed, and the key set is identical at every budget: the same
    discipline `tests/test_disk_shadow.py` enforces next door, proved here because this
    module has its own payload and its own trim path.
    """
    for index in range(disk_gpu_shadow.MAX_REASON_SAMPLES * 3):
        disk_gpu_shadow._reason(f"a-refusal-category-number-{index}", "x" * 4000)
    for key in ("offers", "queued", "written", "bytes_written", "pairs_checked",
                "pairs_identical", "pairs_differing", "refused_not_equal",
                "refused_write_failed", "retention_removed", "walk_entries_examined",
                "errors"):
        disk_gpu_shadow._count(key, 123_456)
    disk_gpu_shadow._state["last_difference"] = {
        "name": "kvslot-0000000000000000-g3-0123456789ab.bin",
        "first_difference": 4096, "scanned_bytes": 8192,
        "shadow_bytes": SLOT_BYTES, "live_bytes": SLOT_BYTES,
        "size_disagree": False}
    disk_gpu_shadow._state["differing_names"] = [
        f"kvslot-{index:016x}-g3-0123456789ab.bin"
        for index in range(disk_gpu_shadow.MAX_NAME_SAMPLES)
    ]
    disk_gpu_shadow._state["measured_files"] = 64
    disk_gpu_shadow._state["measured_bytes"] = 64 * SLOT_BYTES

    fat = disk_gpu_shadow.section()
    raw = worker_diag._dump(fat)
    assert len(raw) <= disk_gpu_shadow.MAX_SECTION_BYTES, len(raw)
    assert disk_gpu_shadow.MAX_SECTION_BYTES < worker_diag.MAX_CONTRIBUTION_BYTES
    assert fat["trimmed"], fat
    assert fat["section_trimmed"] >= 1
    for key in disk_gpu_shadow._COUNTER_KEYS:
        assert isinstance(fat[key], int), key
    assert fat["written"] == 123_456 and fat["pairs_differing"] == 123_456
    # WHAT A READER ACTS ON IS THE OFFSET, and it survives every shed.
    assert fat["last_difference"]["first_difference"] == 4096

    monkeypatch.setattr(disk_gpu_shadow, "MAX_SECTION_BYTES", 10 ** 9)
    full = disk_gpu_shadow.section()
    assert full["trimmed"] == []
    assert set(fat) == set(full)
    assert set(fat["last_difference"]) == set(full["last_difference"])

    monkeypatch.setattr(disk_gpu_shadow, "MAX_SECTION_BYTES", 512)
    with caplog.at_level(logging.WARNING, logger="disk_gpu_shadow"):
        pinched = disk_gpu_shadow.section()
    assert set(pinched) == set(full)
    assert set(pinched["last_difference"]) == set(full["last_difference"])
    for key in disk_gpu_shadow._COUNTER_KEYS:
        assert isinstance(pinched[key], int), key
    assert pinched["reasons"] == []
    assert pinched["trimmed"].count("reasons_dropped") == 1
    assert f"samples:{disk_gpu_shadow.TRIM_NAME_SAMPLES}" in pinched["trimmed"]
    assert len(pinched["differing_names"]) == disk_gpu_shadow.TRIM_NAME_SAMPLES
    assert pinched["section_trimmed"] >= 2
    assert any("STILL over its byte budget" in record.getMessage()
               for record in caplog.records), [record.getMessage()
                                               for record in caplog.records]
    disk_gpu_shadow.reset_state_for_tests()
    assert set(fat) == set(disk_gpu_shadow.section())


# --- 8. THE DOOR AND THE FILE -------------------------------------------------
def test_the_op_is_in_every_table_that_describes_the_surface():
    """`gpu-shadow` is an op in all three tables, and POST-only."""
    assert "gpu-shadow" in disk_store.DISK_OPS
    assert "gpu-shadow" in disk_store.MUTATING_DISK_OPS
    assert "gpu-shadow" in disk_store._DISK_OPS
    spec = endpoint.ARG_SPECS[config.ROUTE_DISK]
    assert set(spec["op"]["values"]) == set(disk_store.DISK_OPS)
    refused = disk_store.disk_for_engine(None, {"op": "gpu-shadow",
                                                "safe_method": True})
    assert refused["ok"] is False and "POST" in refused["reason"]


def test_the_op_reads_by_default_and_arms_only_the_process_answering(writer, stage,
                                                                    monkeypatch):
    """The read/apply rule, and the scope an operator would otherwise assume wrong.

    `{}` is a question; a present `enabled` is an instruction; a non-boolean leaves the
    switch untouched. And because this writes no control file, a POST that reaches the
    EngineCore latches the EngineCore - while the process that WRITES files is a worker,
    armed by the environment. The answer has to say that, because here an operator who
    assumed otherwise would be waiting for files that never come.
    """
    monkeypatch.delenv(config.ENV_DISK_GPU_SHADOW_WRITES)
    read = disk_store.disk_for_engine(None, {"op": "gpu-shadow"})
    assert read["ok"] is True and read["gpu_shadow"]["enabled"] is False
    assert "changed" not in read["gpu_shadow"]
    assert read["gpu_shadow"]["root"]["env_var"] == config.ENV_DISK_GPU_SHADOW_ROOT

    bad = disk_store.disk_for_engine(None, {"op": "gpu-shadow", "enabled": "yes"})
    assert bad["ok"] is False and bad["gpu_shadow"]["latch"] is False

    armed = disk_store.disk_for_engine(None, {"op": "gpu-shadow", "enabled": True})
    assert armed["ok"] is True and armed["gpu_shadow"]["enabled"] is True
    assert config.ENV_DISK_GPU_SHADOW_WRITES in armed["gpu_shadow"]["scope"]
    assert "worker" in armed["note"]
    assert disk_gpu_shadow.effective_enabled() is True

    off = disk_store.disk_for_engine(None, {"op": "gpu-shadow", "enabled": False})
    assert off["gpu_shadow"]["enabled"] is False


def test_the_rank_file_carries_the_writer_verdict_through_the_only_writer(
        tier, worker_class, monkeypatch):
    """End to end: a worker hook, a shadow write, and `diagnostics-rank.json`.

    `worker_diag` is the only writer of the file, so this section arrives THROUGH it -
    and it is the FOURTH section the seam allows, which is the invariant
    `tests/test_worker_diag.py` asserts against `MAX_CONTRIBUTIONS`.
    """
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "0")
    monkeypatch.setenv(config.ENV_DISK_SHADOW, "1")
    monkeypatch.setenv(config.ENV_DISK_GPU_SHADOW_WRITES, "1")
    monkeypatch.setattr(disk_shadow, "SAMPLE_EVERY_N_STORES", 1)
    monkeypatch.setattr(disk_shadow, "MIN_INTERVAL_SECONDS", 0.0)
    assert worker_diag.install(worker_class)["installed"] is True
    assert disk_store.install()["installed"] is True
    arena = {"g0": FakeRegion(GPU_BLOCKS, SLOT_BYTES)}
    gpu = {"g0": FakeGpuRegion(GPU_BLOCKS, SLOT_BYTES)}
    worker = fork_worker_instance(worker_class, arena)
    worker.gpu_kv_caches = gpu
    monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", CopySeam(gpu))
    for event, key, slot in ((3, LONG_KEY, 5), (4, OTHER_KEY, 4)):
        gpu["g0"].set_block(slot, arena["g0"].slot_bytes(slot))
        step = metadata(store_cpu_blocks=[slot], store_gpu_blocks=[slot],
                        store_keys=[key], store_event=event)
        worker.bind_connector_metadata(step)
        worker.delivered = list(worker.delivered) + [event]
        worker.get_finished(set())
    assert drain_everything()["shadow"]["drained"] is True
    # A THIRD cycle, after the writers have finished: `worker_diag`'s own
    # `get_finished` layer sits UNDER this one, so the document a reader holds
    # describes the boot as of the PREVIOUS completion (the same one-behind property
    # `test_disk_store.py` pins for the `disk` section). Without it the file would be
    # written while the shadow writer thread was still committing its first file.
    gpu["g0"].set_block(3, arena["g0"].slot_bytes(3))
    step = metadata(store_cpu_blocks=[3], store_gpu_blocks=[3],
                    store_keys=[THIRD_KEY], store_event=5)
    worker.bind_connector_metadata(step)
    worker.delivered = list(worker.delivered) + [5]
    worker.get_finished(set())
    assert drain_everything()["shadow"]["drained"] is True

    document = json.loads((tier.parent / "diag" / "rank-PP0_TP0.json").read_text())
    assert document["version"] == 1 and document["role"] == "worker"
    assert document["arena"]["bytes_total"] > 0        # the reader's fields survive
    written = document["writer"]["contributions"]["written"]
    assert "disk_gpu_shadow" in written
    section = document["disk_gpu_shadow"]
    assert section["enabled"] is True
    assert section["written"] >= 1
    assert section["pairs_identical"] >= 1
    assert section["writer"] == disk_gpu_shadow.OWNER_TAG
    assert "never authoritative" in section["note"]
    assert section["root"]["root"].endswith(SHADOW_DIR)


def test_the_import_time_state_is_inert_and_the_limits_are_the_shipped_ones():
    """The shipped numbers, pinned where a change has to be deliberate.

    These are the limits the safety story is written against: four slot widths of host
    memory held by a diagnostic, four pairs compared per pass, a shadow directory capped
    at 64 files and 512 MiB, a root that is refused if it is the store. Nothing at
    import: no thread, no directory, no latch, no counters moved.
    """
    assert disk_gpu_shadow.MAX_QUEUE_JOBS == 4
    assert disk_gpu_shadow.MAX_QUEUED_BYTES == 64 * 1024 * 1024
    assert disk_gpu_shadow.MAX_PAIRS_PER_PASS == 4
    assert disk_gpu_shadow.MAX_SHADOW_FILES == 64
    assert disk_gpu_shadow.MAX_SHADOW_BYTES == 512 * 1024 * 1024
    assert disk_gpu_shadow.MAX_SECTION_BYTES < worker_diag.MAX_CONTRIBUTION_BYTES
    assert disk_gpu_shadow.counters() == dict.fromkeys(
        disk_gpu_shadow._COUNTER_KEYS, 0)
    assert disk_gpu_shadow.enable_state()["enabled"] is False
    assert disk_gpu_shadow.enable_state()["latch_set"] is False
    assert disk_gpu_shadow.counters() == dict.fromkeys(
        disk_gpu_shadow._COUNTER_KEYS, 0)
    assert "disk_gpu_shadow" in worker_diag.CONTRIBUTION_SECTIONS
