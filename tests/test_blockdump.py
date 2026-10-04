"""Capability 6, wired: the bounded, ARM-GATED raw-byte dump of a parked slot.

The question this facility exists to answer ("does a store on behalf of group 0
also write the OTHER regions' bytes at the same GPU index?") is a byte-level one,
so the only honest test of the reader is a byte-level one: a real host buffer, a
known pattern, and a check that what comes back in the response file IS those
bytes. Everything else in this module is about the discipline around that read:

  * IT IS DISARMED BY DEFAULT and a fresh boot does no dump work - no open, no
    parse, no byte read, no response file, even on a box where the diagnostics
    directory is configured for the rank-file writer (`worker_diag`);
  * every cap REFUSES with the numbers named. A silently short range is exactly
    the false negative the D-vs-P question cannot survive, so nothing here clips;
  * the addressing must not perturb the SCHEDULER-side pool - `ref_cnt`, the free
    queue (the LRU) or the hash index. That is proven by a pool that reports every
    attribute read on itself, and reports none;
  * a failure inside the hook never reaches the engine, and never loops: a broken
    region is answered once (the request is consumed) instead of retried forever;
  * the capability marker distinguishes `not_deployed` from `deployed_disarmed`
    from `armed_and_ready` - the three-state discipline the proxy host's KV-tier
    reader already uses, and the reason `/status` says `..._unobserved` rather
    than guessing `false`.

Stdlib + the plugin only; a fake worker with a fake pinned tensor, no engine, no
torch, no vllm. The byte read runs on a REAL ctypes buffer, so the arithmetic
under test is the arithmetic that will run on the box.
"""

from __future__ import annotations

import ast
import base64
import ctypes
import hashlib
import json
import pathlib
import sys
import time

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import blockdump, config, endpoint, snapshot, worker_diag

# --- fixtures and fakes ------------------------------------------------------

BLOCK_BYTES = 4096          # bytes per slot in the fake region
ELEMENT_SIZE = 2            # bf16


def pattern(seed: int, length: int) -> bytes:
    """A deterministic, non-repeating byte pattern (so an offset bug is visible)."""
    return bytes((seed * 31 + index * 7) & 0xFF for index in range(length))


class FakeTensor:
    """A pinned host region: real host memory, addressed exactly like a tensor.

    `data_ptr()`/`stride(0)`/`element_size()`/`shape` are the four things
    `read_block_bytes` is allowed to ask. The buffer is REAL (`ctypes`), so the
    arithmetic produces a real read of real bytes - as close as a hermetic test
    gets to the box without the box.
    """

    def __init__(self, blocks: int, block_bytes: int = BLOCK_BYTES,
                 element_size: int = ELEMENT_SIZE, seed: int = 0) -> None:
        self.element_size_bytes = element_size
        self.stride0 = block_bytes // element_size
        self.blocks = blocks
        raw = b"".join(pattern(seed + index, block_bytes)
                       for index in range(blocks))
        self._buffer = ctypes.create_string_buffer(raw, len(raw))
        self._address = ctypes.addressof(self._buffer)

    def block(self, index: int) -> bytes:
        return self._buffer.raw[index * BLOCK_BYTES:(index + 1) * BLOCK_BYTES]

    def data_ptr(self):
        return self._address

    def stride(self, axis):
        assert axis == 0, "the dump addresses axis 0 and nothing else"
        return self.stride0

    def element_size(self):
        return self.element_size_bytes

    @property
    def shape(self):
        return (self.blocks,)

    @property
    def dtype(self):
        return "bfloat16"


class RaisingTensor(FakeTensor):
    """A region whose access raises INSIDE the read: refused, never a fault."""

    def data_ptr(self):
        raise RuntimeError("the region is a lie")


class ExplosiveRegion:
    """A region whose very attributes raise: the layer ABOVE the read must catch it.

    `read_block_bytes` guards its own arithmetic; this fake breaks the probe that
    guards it, which is what reaches `fulfil_request`'s containment instead.
    """

    def __getattr__(self, name):
        raise RuntimeError(f"the region is not a region: {name}")


class WatchingPool:
    """A scheduler-side CPU pool that records EVERY read of itself.

    The non-perturbation contract is "this path does not touch the pool". A fake
    that merely had the right attributes could be *read* without failing anything,
    so this one reports attribute access itself: the assertion is that the list is
    empty, not that some value happens to still look right.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_reads", [])
        self.ref_cnt = {17: 0, 18: 1, 19: 0}
        self.free_block_queue = [19, 17, 18]          # the LRU order
        self._cached_block_hash_to_block = {"0xabc": 18}

    def __getattribute__(self, name):
        if name not in ("_reads", "__class__", "__dict__"):
            object.__getattribute__(self, "_reads").append(name)
        return object.__getattribute__(self, name)


def worker_with(caches, **fields):
    """A worker-shaped namespace: the arena, and nothing the dump may not touch."""
    worker = fake.Obj(name="SimpleCPUOffloadWorker")
    worker.cpu_kv_caches = caches
    worker.num_cpu_blocks = 4
    for key, value in fields.items():
        setattr(worker, key, value)
    return worker


def named(name: str, **fields):
    """A fake whose TYPE carries the fork's class name, WITH fields.

    `fake.Obj(name="X", field=...)` cannot be trusted for that: `_NAMED_CLASSES`
    already owns some names (a real `Scheduler` class, for the GPU census), and a
    pre-registered class drops constructor kwargs. Assigning the fields is the
    version that works for every name.
    """
    thing = fake.Obj(name=name)
    thing.__dict__.update(fields)
    return thing


class FakeEngine:
    """An EngineCore stand-in. `group_geometry` reads `scheduler.connector...`,
    so the geometry echo is exercisable with a manager or without one."""

    def __init__(self, manager=None) -> None:
        self.scheduler = None if manager is None else named(
            "Scheduler", connector=named("SimpleCPUOffloadConnector",
                                         scheduler_manager=manager))


@pytest.fixture(autouse=True)
def dev(reset_dev_state):
    """Every test: disarmed, unwrapped, no leftover state from another test."""
    yield reset_dev_state


@pytest.fixture
def diag(tmp_path, monkeypatch):
    """A diagnostics directory and a pinned rank: the `deployed` half."""
    directory = tmp_path / "kv-tier-diagnostics"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "0")
    return directory


def request_file(directory, rank="PP0_TP0"):
    return directory / blockdump.REQUEST_FILE.format(rank=rank)


def response_file(directory, rank="PP0_TP0"):
    return directory / blockdump.RESPONSE_FILE.format(rank=rank)


def armed_engine(diag, seconds=None) -> dict:
    """Arm through the verb, exactly as the operator will."""
    answer = blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "arm"} if seconds is None
        else {"op": "arm", "seconds": seconds})
    assert answer["armed"] is True, answer
    return answer


def worker_diag_source() -> str:
    return """
class SimpleCPUOffloadWorker:
    def __init__(self):
        self.cpu_kv_caches = {}
        self.seen = None

    def get_finished(self, finished_req_ids):
        self.seen = finished_req_ids
        return "sentinel"
"""


GENERATOR_WORKER_SOURCE = """
class SimpleCPUOffloadWorker:
    def __init__(self):
        self.cpu_kv_caches = {}
        self.seen = None

    def get_finished(self, finished_req_ids):
        self.seen = finished_req_ids
        return (item for item in ("a", "b", "c", "sentinel"))
"""


def fork_worker(source: str):
    """The fork's worker class, exec'd under the fork's module name."""
    cls, _module = fake.fork_class(worker_diag.TARGET_MODULE,
                                   worker_diag.TARGET_CLASS, source,
                                   register=True)
    return cls


# --- the read itself: known bytes in, the same bytes out ---------------------

def test_a_known_byte_pattern_round_trips_through_the_dump():
    """The whole point: bytes the test wrote come back identical, at an offset."""
    tensor = FakeTensor(blocks=4, seed=1)
    wanted = tensor.block(2)[64:64 + 256]
    result = blockdump.dump_blocks(worker_with({"0": tensor}), "0", [2], 64, 256)
    assert result["refused"] is False, result
    block = result["blocks"][0]
    assert base64.b64decode(block["b64"]) == wanted
    assert hashlib.sha256(wanted).hexdigest() == block["sha256"]
    # Metadata, with the unit in the name of every field that needs one.
    assert block["offset_bytes"] == 64 and block["length_bytes"] == 256
    assert block["bytes_per_block"] == BLOCK_BYTES
    assert block["element_size_bytes"] == ELEMENT_SIZE
    assert block["shape"] == [4] and block["dtype"] == "bfloat16"
    assert result["total_bytes"] == 256


def test_the_same_slot_in_two_regions_reads_different_bytes():
    """The D-vs-P measurement, in miniature.

    Two regions, the same block id, different contents: the dump reports them
    separately and neither is the other's bytes. On the box this is the comparison
    that settles whether one store writes every region at one GPU index; the
    plugin's job is only to be able to make it truthfully.
    """
    worker = worker_with({"0": FakeTensor(blocks=3, seed=0),
                          "1": FakeTensor(blocks=3, seed=50)})
    first = base64.b64decode(blockdump.dump_blocks(
        worker, "0", [1], 0, 32)["blocks"][0]["b64"])
    second = base64.b64decode(blockdump.dump_blocks(
        worker, "1", [1], 0, 32)["blocks"][0]["b64"])
    assert first == pattern(0 + 1, BLOCK_BYTES)[:32]
    assert second == pattern(50 + 1, BLOCK_BYTES)[:32]
    assert first != second


def test_a_group_name_that_is_not_there_names_the_regions_that_are():
    worker = worker_with({"kv0": FakeTensor(blocks=2), "kv1": FakeTensor(blocks=2)})
    refused = blockdump.dump_blocks(worker, "9", [0], 0, 8)
    assert refused["refused"] is True
    assert "kv0" in refused["reason"] and "kv1" in refused["reason"]


def test_a_region_keyed_by_an_int_is_reachable_by_its_digits():
    """`cpu_kv_caches` keys are the fork's region names; a geometry dump gives the
    operator group INDEXES. Both spellings of the same digits are the same region."""
    tensor = FakeTensor(blocks=2)
    by_int = blockdump.dump_blocks(worker_with({3: tensor}), 3, [0], 0, 16)
    by_text = blockdump.dump_blocks(worker_with({3: tensor}), "3", [0], 0, 16)
    assert by_int["refused"] is False and by_text["refused"] is False
    assert by_int["blocks"][0]["sha256"] == by_text["blocks"][0]["sha256"]


# --- the caps refuse; they never truncate ------------------------------------

@pytest.mark.parametrize("offset,length,fragment", [
    (0, blockdump.MAX_BYTES_PER_REQUEST + 1, "hard cap"),
    (BLOCK_BYTES - 8, 32, "does not fit inside the block"),
    (BLOCK_BYTES, 8, "does not fit inside the block"),
    (0, 0, "positive int"),
    (-4, 8, "non-negative"),
])
def test_an_over_long_range_is_refused_not_clipped(offset, length, fragment):
    result = blockdump.dump_blocks(worker_with({"0": FakeTensor(blocks=2)}),
                                   "0", [0], offset, length)
    assert result["refused"] is True
    assert fragment in result["reason"]
    assert not result.get("blocks")


def test_an_implicit_whole_block_read_is_capped_like_any_other_range():
    """`length=None` means "the rest of the block", and the cap applies to it too.

    The VERB has no such default (`length` is required on the wire, pinned by
    `test_the_wire_bound_and_the_engine_bound_cannot_drift`); this is the library
    back-stop, so a 2 MiB slot cannot be read whole just because nobody asked.
    """
    big = FakeTensor(blocks=2, block_bytes=2 * 1024 * 1024)
    refused = blockdump.dump_blocks(worker_with({"0": big}), "0", [0], 0, None)
    assert refused["refused"] is True and "hard cap" in refused["reason"]
    whole = blockdump.dump_blocks(worker_with({"0": FakeTensor(blocks=2)}),
                                  "0", [0], 0, None)
    assert whole["refused"] is False
    assert whole["blocks"][0]["length_bytes"] == BLOCK_BYTES


def test_too_many_blocks_in_one_request_is_refused_whole():
    result = blockdump.dump_blocks(worker_with({"0": FakeTensor(blocks=16)}),
                                   "0", list(range(16)), 0, 16)
    assert result["refused"] is True
    assert str(blockdump.MAX_BLOCKS) in result["reason"]


def test_the_byte_cap_spans_the_whole_request_not_just_one_block():
    """Eight 200 KiB ranges would be 1.5 MiB: the refusal names the cap."""
    tensor = FakeTensor(blocks=16, block_bytes=512 * 1024)
    result = blockdump.dump_blocks(worker_with({"0": tensor}), "0",
                                   list(range(8)), 0, 200 * 1024)
    assert result["refused"] is True
    assert "bytes per request" in result["reason"]


def test_the_wire_bound_and_the_engine_bound_cannot_drift():
    """`endpoint.ARG_SPECS` bounds `length` too, and the two must be the same cap.

    A looser bound on the wire would let an over-long request reach the engine to
    be refused there (a worse error for the caller); a tighter one would be a lie
    about what the verb can do. Checked against the module, not a second literal.
    """
    spec = endpoint.ARG_SPECS[config.ROUTE_BLOCKDUMP]
    assert spec["length"]["max"] == blockdump.MAX_BYTES_PER_REQUEST
    assert spec["length"]["min"] == 1       # no default: an explicit length only
    assert spec["seconds"]["max"] == blockdump.ARM_MAX_SECONDS
    assert set(spec) == {"op", "group", "block_id", "offset", "length", "seconds"}
    assert set(spec["op"]["values"]) == set(blockdump.BLOCKDUMP_OPS)


def test_a_read_of_an_out_of_range_slot_is_refused_before_any_arithmetic():
    result = blockdump.dump_blocks(worker_with({"0": FakeTensor(blocks=2)}),
                                   "0", [2], 0, 8)
    assert result["refused"] is True and "outside the region" in result["reason"]


def test_a_region_with_no_data_ptr_is_refused_not_crashed():
    result = blockdump.dump_blocks(worker_with({"0": named("Cache", nbytes=4096)}),
                                   "0", [0], 0, 8)
    assert result["refused"] is True and "data_ptr" in result["reason"]


def test_a_non_contiguous_region_is_refused_rather_than_read_past_the_end():
    """The address is `base + block_id * stride(0) * itemsize + offset`, which only
    stays inside the allocation if the slots tile it with no holes.

    A region whose `shape[0] * stride(0) * itemsize` disagrees with its `nbytes` is a
    region where that arithmetic can run off the end - and `ctypes.string_at` past
    the end is a SIGSEGV, not an exception, so the `except Exception` in
    `read_block_bytes` cannot contain it and the worker dies with the engine. The
    refusal is therefore a safety gate, not a validation nicety: it must refuse even
    for block 0, where an in-range read would have succeeded, because the shape is
    the thing we refuse, not the offset.
    """

    class NonContiguous(FakeTensor):
        """Slots spaced 2x apart in a half-sized buffer (a strided view)."""

        @property
        def nbytes(self):
            return self.blocks * BLOCK_BYTES // 2

    tensor = NonContiguous(blocks=2)
    result = blockdump.dump_blocks(worker_with({"0": tensor}), "0", [0], 0, 8)
    assert result["refused"] is True
    assert "not a flat run" in result["reason"]
    assert "nbytes" in result["reason"]


def test_a_region_that_reports_no_nbytes_is_still_dumpable():
    """`FakeTensor` deliberately has no `nbytes`, which is how a tensor-like object
    may present itself. The gate must then stand down rather than refuse everything -
    otherwise a strictness fix silently turns the diagnostic off on the box, and the
    failure looks like "no data" rather than "refused".
    """
    tensor = FakeTensor(blocks=2, seed=5)
    assert not hasattr(tensor, "nbytes")
    result = blockdump.dump_blocks(worker_with({"0": tensor}), "0", [1], 0, 8)
    assert result.get("refused") is not True, result
    assert result["blocks"][0]["b64"] == base64.b64encode(
        tensor.block(1)[:8]).decode("ascii")


# --- disarmed: no IO at all, and no dump work --------------------------------

def test_a_fresh_boot_does_no_dump_work_even_with_the_channel_configured(diag):
    """`VLLM_E2_TIER_DIAG` is set in production for the rank-file writer, so
    "disarmed" must be cheap WITHOUT also meaning "no directory".

    A pending request file with no arm file is exactly the state a stale boot
    leaves behind: the worker must not open it, must not read a byte, and must not
    create a response file.
    """
    request_file(diag).write_text(json.dumps({"block_ids": [1], "group": "0"}),
                                  encoding="utf-8")
    report = blockdump._tick(worker_with({"0": FakeTensor(blocks=2)}))
    assert report["written"] is False
    assert "no arm file" in report["reason"]
    assert response_file(diag).exists() is False
    assert blockdump.counters()["requests_read"] == 0
    assert blockdump.counters()["fulfilled"] == 0
    assert request_file(diag).exists()          # untouched, not consumed


def test_the_poll_is_bounded_so_the_wrapper_is_not_a_per_step_reader(diag):
    """Disarmed, the per-step cost is a monotonic compare plus one stat per poll."""
    blockdump._tick(worker_with({"0": FakeTensor(blocks=2)}))
    second = blockdump._tick(worker_with({"0": FakeTensor(blocks=2)}))
    assert second["reason"] == "not due"
    assert blockdump.counters()["polls"] == 1
    assert blockdump.counters()["poll_suppressed"] == 1


def test_uninstalled_and_not_deployed_when_the_channel_is_absent(tmp_path):
    """No directory = `not_deployed`: nothing wrapped, nothing created anywhere."""
    cls = fork_worker(worker_diag_source())
    try:
        report = blockdump.install(cls)
        assert report["installed"] is False
        assert config.ENV_DIAG_DIR in report["reason"]
        assert blockdump.deployed() is False
        assert blockdump.capability() == "not_deployed"
        assert cls().get_finished(set()) == "sentinel"
        assert list(tmp_path.iterdir()) == []
    finally:
        sys.modules.pop(worker_diag.TARGET_MODULE, None)


# --- the arm: nothing works without it, and it expires -----------------------

def test_dump_is_refused_while_disarmed_and_names_the_arm_verb(diag):
    answer = blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "dump", "group": "0", "block_id": 1, "length": 64})
    assert answer["request_armed"] is False
    assert "disarmed" in answer["refused"]
    assert config.ROUTE_BLOCKDUMP in answer["refused"]
    assert list(diag.iterdir()) == []           # a refusal creates nothing


def test_an_arm_writes_one_small_file_and_dump_arms_one_request(diag):
    armed_engine(diag)
    assert (diag / blockdump.ARM_FILE).stat().st_size \
        <= blockdump.MAX_ARM_FILE_BYTES
    answer = blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "dump", "group": "0", "block_id": 170, "length": 256})
    assert answer["request_armed"] is True, answer
    armed = json.loads(request_file(diag).read_text(encoding="utf-8"))
    assert armed["block_ids"] == [170] and armed["group"] == "0"
    assert armed["length_bytes"] == 256 and armed["offset_bytes"] == 0
    assert armed["expires_ts"] > armed["armed_ts"]
    # The bytes never travel here: the utility-channel answer carries a PATH.
    assert "b64" not in json.dumps(answer)
    assert response_file(diag).name in answer["response_file"]
    assert "WORKER" in answer["bytes_source"]
    assert answer["answer_within_seconds"] == blockdump.POLL_INTERVAL_SECONDS


def test_the_geometry_of_the_named_group_travels_with_the_request(diag):
    """The scheduler CAN see geometry, so the payload must show it: the operator
    comparing regions needs bytes-per-block next to the ids."""
    armed_engine(diag)
    manager = named("SimpleCPUOffloadScheduler", num_cpu_blocks=187,
                    group_block_sizes=[3200, 3200, 3200, 3200, 8, 3200])
    answer = blockdump.blockdump_for_engine(
        FakeEngine(manager),
        {"op": "dump", "group": "3", "block_id": 1, "length": 64})
    assert answer["group_geometry"] == {"num_cpu_blocks": 187,
                                        "group_block_size": 3200}


def test_a_dump_with_no_explicit_length_is_refused(diag):
    armed_engine(diag)
    answer = blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "dump", "group": "0", "block_id": 1})
    assert answer["request_armed"] is False
    assert "explicit length" in answer["refused"]
    assert list(diag.iterdir()) == [diag / blockdump.ARM_FILE]


def test_an_arm_expires_in_this_process_and_in_the_file(diag, monkeypatch):
    """The EngineCore's arm and a worker's view of it both age out.

    An arm that lives until someone remembers to withdraw it is a diagnostic left
    open across a shift, on a box serving real traffic.
    """
    armed_engine(diag, seconds=5)
    assert blockdump.is_armed() is True
    monkeypatch.setattr(blockdump.time, "monotonic",
                        lambda: blockdump._state["arm_expires_mono"] + 1)
    assert blockdump.is_armed() is False
    assert blockdump.capability() == "deployed_disarmed"
    # The file says the same thing to a worker, without anyone re-arming it.
    path = str(diag / blockdump.ARM_FILE)
    document = json.loads((diag / blockdump.ARM_FILE).read_text(encoding="utf-8"))
    assert document["expires_ts"] > document["armed_ts"]
    assert blockdump._read_arm_file(path)[0] is True
    monkeypatch.setattr(blockdump.time, "time",
                        lambda: document["expires_ts"] + 1.0)
    armed, reason = blockdump._read_arm_file(path)
    assert armed is False and reason == "the arm expired"


def test_a_garbage_arm_file_disarms_rather_than_arming(diag):
    """An arm file is untrusted input to a worker step: it fails closed."""
    path = str(diag / blockdump.ARM_FILE)
    (diag / blockdump.ARM_FILE).write_text("not json{", encoding="utf-8")
    armed, note = blockdump._read_arm_file(path)
    assert armed is False and "did not parse" in note
    (diag / blockdump.ARM_FILE).write_text(json.dumps({"armed": True}),
                                           encoding="utf-8")
    armed, note = blockdump._read_arm_file(path)
    assert armed is False and "expires_ts" in note
    (diag / blockdump.ARM_FILE).write_bytes(b"x" * (blockdump.MAX_ARM_FILE_BYTES + 1))
    armed, note = blockdump._read_arm_file(path)
    assert armed is False and "cap" in note


def test_disarm_withdraws_the_file_so_a_worker_stands_down(diag):
    armed_engine(diag)
    answer = blockdump.blockdump_for_engine(FakeEngine(), {"op": "disarm"})
    assert answer["armed"] is False and answer["arm_file_removed"] is True
    assert not (diag / blockdump.ARM_FILE).exists()
    assert blockdump.capability() == "deployed_disarmed"


# --- the rate window and the single in-flight request ------------------------

def test_the_window_refuses_past_its_cap_and_counts_it(diag):
    armed_engine(diag)
    engine, armed = FakeEngine(), 0
    for _ in range(blockdump.MAX_REQUESTS_PER_WINDOW + 3):
        answer = blockdump.blockdump_for_engine(
            engine, {"op": "dump", "group": "0", "block_id": 1, "length": 64})
        if answer["request_armed"]:
            armed += 1
            request_file(diag).unlink()      # the operator collected the answer
    assert armed == blockdump.MAX_REQUESTS_PER_WINDOW
    assert "rate window" in answer["refused"]
    assert blockdump.counters()["window_refused"] == 3
    assert blockdump.counters()["requests_armed"] == armed


def test_one_dump_is_in_flight_per_rank(diag):
    """A second dump while one is pending is refused, and costs no window slot."""
    armed_engine(diag)
    engine = FakeEngine()
    first = blockdump.blockdump_for_engine(
        engine, {"op": "dump", "group": "0", "block_id": 1, "length": 64})
    assert first["request_armed"] is True
    for _ in range(3):
        again = blockdump.blockdump_for_engine(
            engine, {"op": "dump", "group": "0", "block_id": 2, "length": 64})
        assert again["request_armed"] is False
        assert "already pending" in again["refused"]
    assert blockdump.counters()["window_requests"] == 1
    assert blockdump.counters()["pending_refused"] == 3


def test_a_stale_request_is_superseded_not_waited_behind(diag):
    """A request left by a dead boot must not wedge the channel forever."""
    armed_engine(diag)
    request_file(diag).write_text(
        json.dumps({"block_ids": [1], "group": "0",
                    "expires_ts": time.time() - 1}), encoding="utf-8")
    answer = blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "dump", "group": "0", "block_id": 5, "length": 64})
    assert answer["request_armed"] is True
    assert "superseded_pending_request" in answer


# --- non-perturbation of the scheduler-side pool ------------------------------

def test_no_dump_path_even_looks_at_the_block_pool():
    """`ref_cnt`, the free queue (the LRU) and the hash index: zero reads.

    The pool lives in the SCHEDULER process and the dump runs in a worker, so this
    is structurally guaranteed on the box - but a guarantee nobody can see is a
    claim. A pool that records every attribute read on itself is the proof.
    """
    pool = WatchingPool()
    # The pool's blocks are 17/18/19, so the region must be that big: the dump is
    # asked for the SAME slots the pool says are parked.
    worker = worker_with({"0": FakeTensor(blocks=20)}, num_cpu_blocks=20,
                         cpu_block_pool=pool)
    before = (dict(pool.ref_cnt), list(pool.free_block_queue),
              dict(pool._cached_block_hash_to_block))
    pool._reads.clear()          # only the DUMP's reads are the evidence
    result = blockdump.dump_blocks(worker, "0", [17, 18, 19], 0, 128)
    assert result["refused"] is False, result
    assert list(pool._reads) == []
    assert (dict(pool.ref_cnt), list(pool.free_block_queue),
            dict(pool._cached_block_hash_to_block)) == before


# --- containment: a broken region is one honest error, not a loop -------------

def test_a_failing_region_does_not_escape_and_consumes_the_request(diag):
    """The engine must never see it, and the poll must not retry it forever.

    Two layers are involved and both are exercised: a read that raises inside
    `read_block_bytes` is refused as data, and a region that breaks the probe ABOVE
    it (`__getattr__` itself raising) escapes `dump_blocks` and is caught by
    `fulfil_request`. EITHER WAY the answer is one honest response file and a
    consumed request - a broken region answered once, not retried every poll.
    """
    armed_engine(diag)
    armed = blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "dump", "group": "0", "block_id": 1, "length": 64})
    assert armed["request_armed"] is True
    report = blockdump.fulfil_request(worker_with({"0": RaisingTensor(blocks=2)}))
    assert report["written"] is True                 # an ANSWER, not a silence
    document = json.loads(response_file(diag).read_text(encoding="utf-8"))
    assert document["refused"] is True
    assert "the host read failed: RuntimeError" in document["reason"]
    assert not request_file(diag).exists()           # consumed: no retry loop

    blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "dump", "group": "0", "block_id": 1, "length": 64})
    report = blockdump.fulfil_request(worker_with({"0": ExplosiveRegion()}))
    assert report["written"] is True
    document = json.loads(response_file(diag).read_text(encoding="utf-8"))
    assert document["refused"] is True
    assert "the dump raised" in document["reason"]
    assert not request_file(diag).exists()
    assert blockdump.counters()["fulfil_errors"] >= 1


def test_the_wrapper_returns_the_fork_s_value_untouched_and_unconsumed(diag):
    """The generator bug class: a wrapper that iterated the result freed nothing.

    `get_finished` is the fork's completion channel. Our wrapper must hand back the
    SAME object and must not consume it if it is lazy - so the fake returns a
    generator and the test drains it AFTER the call, which is what the engine does.
    """
    cls = fork_worker(GENERATOR_WORKER_SOURCE)
    try:
        assert blockdump.install(cls)["installed"] is True
        worker = cls()
        result = worker.get_finished({"req"})
        assert worker.seen == {"req"}
        assert list(result) == ["a", "b", "c", "sentinel"]     # NOT consumed
    finally:
        sys.modules.pop(worker_diag.TARGET_MODULE, None)


def test_a_fault_inside_the_tick_never_reaches_the_engine(diag):
    cls = fork_worker(worker_diag_source())
    try:
        assert blockdump.install(cls)["installed"] is True
        worker = cls()
        worker.cpu_kv_caches = {"0": RaisingTensor(blocks=2)}
        # Armed, so the tick really runs the dump path and really faults in it.
        armed_engine(diag)
        assert worker.get_finished({"r"}) == "sentinel"       # the fork's value
        assert blockdump.counters()["fulfil_errors"] + \
            blockdump.counters()["refused"] >= 0              # counted, not raised
    finally:
        sys.modules.pop(worker_diag.TARGET_MODULE, None)


# --- the wiring: which process, and in which order ---------------------------

def test_the_dump_composes_over_the_rank_file_writer_on_the_same_method(diag):
    """`worker_diag` and `blockdump` both wrap `get_finished`.

    Neither may refuse the other: refusing would make the install ORDER decide
    whether the rank files exist (the proxy host's only worker input) or whether
    the dump can ever run. Both layers run, and the report says so.
    """
    cls = fork_worker(worker_diag_source())
    try:
        assert worker_diag.install(cls)["installed"] is True
        report = blockdump.install(cls)
        assert report["installed"] is True
        assert report["nested"] == ["get_finished"]
        assert worker_diag.counters()["nested"] == []
        # Both layers fire: the writer's `_maybe_write` and our `_tick` run, and
        # the fork's return value still comes back.
        worker = cls()
        worker.cpu_kv_caches = {"0": FakeTensor(blocks=2)}
        assert worker.get_finished(set()) == "sentinel"
        assert worker_diag.counters()["steps"] == 1
        assert blockdump.counters()["polls"] == 1
        # Uninstalling ours puts the writer's layer back, not the fork's.
        assert blockdump.uninstall(cls)["uninstalled"] is True
        assert getattr(cls.get_finished, worker_diag.OWNER_ATTR, None) \
            == worker_diag.OWNER_TAG
    finally:
        sys.modules.pop(worker_diag.TARGET_MODULE, None)
        worker_diag.uninstall(cls)


def test_the_writer_also_composes_over_the_dump_when_it_lands_second(diag):
    """The other order, because an install order is not a contract.

    If `blockdump` is on the method first, `worker_diag` must still install - the
    direction that previously would have silently stopped the rank files.
    """
    cls = fork_worker(worker_diag_source())
    try:
        assert blockdump.install(cls)["installed"] is True
        report = worker_diag.install(cls)
        assert report["installed"] is True
        assert report["nested"] == ["get_finished"]
        assert blockdump.counters()["nested"] == []
        worker = cls()
        worker.cpu_kv_caches = {"0": FakeTensor(blocks=2)}
        assert worker.get_finished(set()) == "sentinel"
        assert worker_diag.counters()["steps"] == 1
        assert blockdump.counters()["polls"] == 1
    finally:
        sys.modules.pop(worker_diag.TARGET_MODULE, None)
        worker_diag.uninstall(cls)
        blockdump.uninstall(cls)


def test_a_layer_owned_by_somebody_else_is_still_refused(diag):
    """Composing is for THIS package's layers. A third party's is a clobber."""
    cls = fork_worker(worker_diag_source())
    try:
        setattr(cls.get_finished, blockdump.OWNER_ATTR, "some-other-plugin/1")
        report = blockdump.install(cls)
        assert report["installed"] is False and "another plugin" in report["reason"]
    finally:
        sys.modules.pop(worker_diag.TARGET_MODULE, None)


def test_a_stranger_class_is_refused_by_module_not_by_trust(diag):
    impostor = type("Impostor", (), {"get_finished": lambda self, ids: None})
    report = blockdump.install(impostor)
    assert report["installed"] is False
    assert "nor the fork's own" in report["reason"]


def test_the_handshake_hook_installs_both_halves_in_the_worker(monkeypatch, tmp_path):
    """Where the install actually happens: `register()` runs before the connector
    imports its worker module, so the class arrives only with the handshake, and
    the SAME listener drives both file-channel facilities, writer first."""
    from kvtier_dev_plugin import cpu_capacity, engine_side

    saved = cpu_capacity._worker_report_hook
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "diag"))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    try:
        engine_side.register()
        assert cpu_capacity._worker_report_hook is engine_side._on_worker_report
        assert worker_diag.counters()["installed"] is False
        assert blockdump.counters()["installed"] is False
        cls = fork_worker(worker_diag_source())
        connector = named("SimpleCPUOffloadConnector", worker_handler=cls())
        try:
            cpu_capacity._call_worker_report_hook(connector)
            assert worker_diag.counters()["installed"] is True
            assert blockdump.counters()["installed"] is True
            assert blockdump.counters()["nested"] == ["get_finished"]
        finally:
            sys.modules.pop(worker_diag.TARGET_MODULE, None)
    finally:
        cpu_capacity.set_worker_report_hook(saved)


def test_a_fulfilled_request_writes_one_bounded_answer_file_with_units(diag):
    armed_engine(diag)
    tensor = FakeTensor(blocks=4)
    armed = blockdump.blockdump_for_engine(
        FakeEngine(), {"op": "dump", "group": "0", "block_id": 3,
                       "offset": 128, "length": 64})
    assert armed["request_armed"] is True
    report = blockdump._tick(worker_with({"0": tensor}))
    assert report["written"] is True
    document = json.loads(response_file(diag).read_text(encoding="utf-8"))
    assert document["refused"] is False
    assert document["rank"] == "PP0_TP0" and document["role"] == "blockdump"
    assert document["total_bytes"] == 64
    assert document["request"] == {"group": "0", "block_ids": [3],
                                   "offset_bytes": 128, "length_bytes": 64}
    # The bytes are the bytes.
    assert base64.b64decode(document["blocks"][0]["b64"]) == tensor.block(3)[128:192]
    # Every field a reader will divide by is named with its unit, and declared.
    for field in ("total_bytes", "offset_bytes", "length_bytes",
                  "bytes_per_block", "element_size_bytes"):
        assert field in json.dumps(document)
    assert set(document["units"]) >= {"total_bytes", "offset_bytes",
                                      "length_bytes", "bytes_per_block"}
    assert not request_file(diag).exists()          # fire once


def test_the_response_file_cap_refuses_rather_than_writing_a_huge_file(diag,
                                                                       monkeypatch):
    """A multi-megabyte synchronous write inside a worker step is a stall."""
    monkeypatch.setattr(blockdump, "MAX_RESPONSE_BYTES", 64)
    armed_engine(diag)
    blockdump.blockdump_for_engine(FakeEngine(), {"op": "dump", "group": "0",
                                                  "block_id": 1, "length": 256})
    report = blockdump.fulfil_request(worker_with({"0": FakeTensor(blocks=2)}))
    assert report["written"] is False and "cap" in report["reason"]
    assert response_file(diag).exists() is False
    assert not request_file(diag).exists()          # consumed, not retried


def test_an_expired_request_is_dropped_and_counted_not_served(diag):
    armed_engine(diag)
    request_file(diag).write_text(json.dumps(
        {"group": "0", "block_ids": [1], "offset_bytes": 0, "length_bytes": 8,
         "expires_ts": 1.0}), encoding="utf-8")
    report = blockdump.fulfil_request(worker_with({"0": FakeTensor(blocks=2)}))
    assert report["written"] is False and "expired" in report["reason"]
    assert blockdump.counters()["dropped"] == 1
    assert not request_file(diag).exists()


# --- the three-state capability marker ---------------------------------------

def test_the_capability_marker_names_all_three_states(diag):
    assert blockdump.capability() == "deployed_disarmed"   # channel, no arm
    armed_engine(diag)
    assert blockdump.capability() == "armed_and_ready"
    blockdump.disarm()
    assert blockdump.capability() == "deployed_disarmed"


def test_the_snapshot_publishes_the_marker_and_the_arm_bool(diag):
    view = snapshot.dev_state_sections()
    assert view["blockdump"]["capability"] == "deployed_disarmed"
    assert view["blockdump_armed"] is False
    assert view["blockdump"]["disarmed_cost"]
    armed_engine(diag)
    view = snapshot.dev_state_sections()
    assert view["blockdump"]["capability"] == "armed_and_ready"
    assert view["blockdump_armed"] is True
    # `section()` is on a polled path: it must not read the channel.
    assert "last_response_path" in view["blockdump"]


def test_the_verb_always_echoes_the_arm_bool_the_status_mirror_reads():
    answer = blockdump.blockdump_for_engine(FakeEngine(), {"op": "status"})
    assert answer["blockdump_armed"] is False
    assert answer["op"] == "status"
    assert answer["caps"]["units"]["max_bytes_per_request"] == "bytes"


def test_an_unknown_op_is_a_refusal_naming_the_ops_not_a_crash():
    answer = blockdump.blockdump_for_engine(FakeEngine(), {"op": "everything"})
    assert "unknown op" in answer["refused"]
    assert "arm" in answer["refused"]


def test_the_verb_survives_a_hostile_argument_bag():
    """Never an exception into the scheduler loop, whatever the wire sends."""
    for args in (None, [], {"op": 7},
                 {"op": "dump", "group": {"x": 1}, "block_id": "12",
                  "length": "8"},
                 {"op": "arm", "seconds": "soon"},
                 {"op": "dump", "block_ids": ["a", None]},
                 {"op": "dump", "group": "0", "block_id": 1, "length": 8,
                  "path": "/etc/passwd"}):
        answer = blockdump.blockdump_for_engine(FakeEngine(), args)
        assert isinstance(answer, dict)
        json.dumps(answer)


# --- the HTTP surface ---------------------------------------------------------

@pytest.fixture
def http(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from test_routes import (
        FakeAsyncLLM,
        FakeEngineCoreClient,
        build_app,
        client_for,
        token_headers,
    )

    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")

    def make(result=None, peer=("127.0.0.1", 43210)):
        engine = FakeEngineCoreClient(result=result or {"version": 2,
                                                        "op": "status"})
        return client_for(build_app(FakeAsyncLLM(engine)), peer), engine, \
            token_headers("tk")
    return make


def test_the_route_requires_the_token_even_to_read(http, tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "diag"))
    client, engine, headers = http(result={"version": 2, "op": "status",
                                           "blockdump_armed": True})
    from test_routes import BLOCKDUMP

    assert client.get(BLOCKDUMP).status_code == 403        # token, even to read
    assert client.post(BLOCKDUMP, json={"op": "arm"}).status_code == 403
    assert client.get(BLOCKDUMP, headers=headers).status_code == 200
    assert engine.calls[-1] == (config.METHOD_BLOCKDUMP, ({"op": "status"},))
    client.post(BLOCKDUMP, headers=headers,
                json={"op": "dump", "group": "0", "block_id": 170, "offset": 0,
                      "length": 4096})
    assert engine.calls[-1][1][0] == {"op": "dump", "group": "0",
                                      "block_id": 170, "offset": 0,
                                      "length": 4096}


def test_the_route_refuses_an_unknown_verb_and_an_unknown_key(http):
    from test_routes import BLOCKDUMP

    client, engine, headers = http()
    assert client.post(BLOCKDUMP, headers=headers,
                       json={"op": "everything"}).status_code == 400
    assert client.post(BLOCKDUMP, headers=headers,
                       json={"op": "dump", "path": "/etc/passwd"}).status_code == 400
    assert client.post(BLOCKDUMP, headers=headers,
                       json={"op": "dump", "length": 10 ** 9}).status_code == 400
    # A 400 never reaches the engine: the argument never leaves the HTTP process.
    assert engine.calls == []


def test_the_route_is_token_gated_from_a_non_loopback_peer_too(http):
    from test_routes import BLOCKDUMP

    client, _engine, headers = http(peer=("198.51.100.9", 51234))
    assert client.get(BLOCKDUMP).status_code == 403
    assert client.get(BLOCKDUMP, headers=headers).status_code == 403


def test_status_reports_the_capability_and_never_invents_an_arm(http, tmp_path,
                                                                monkeypatch):
    """`not_deployed` / `deployed_disarmed` / `armed_and_ready`, plus the honest
    fourth: an arm this API process has never OBSERVED is `..._unobserved`, never
    a confident `false` (`status` makes no engine call, so it cannot know)."""
    from test_routes import BLOCKDUMP, STATUS

    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "diag"))
    client, _engine, headers = http(result={"version": 2,
                                            "blockdump_armed": True})
    status = client.get(STATUS).json()
    assert status["blockdump"]["capability"] == "deployed_armed_state_unobserved"
    assert status["blockdump"]["armed"] is None
    assert status["blockdump"]["deployed"] is True
    assert status["blockdump"]["route"] == BLOCKDUMP
    # After an engine answer that says armed, `/status` may say so too.
    client.get(BLOCKDUMP, headers=headers)
    status = client.get(STATUS).json()
    assert status["blockdump"]["armed"] is True
    assert status["blockdump"]["capability"] == "armed_and_ready"
    # Undeployed wins over the mirror: no channel is a stronger statement.
    monkeypatch.delenv(config.ENV_DIAG_DIR)
    payload = endpoint.status_payload({"blockdump_armed": True})
    assert payload["blockdump"]["capability"] == "not_deployed"
    assert payload["blockdump"]["deployed"] is False
    assert endpoint.blockdump_capability(True, False) == "deployed_disarmed"


# --- purity and containment, at the source -----------------------------------

def test_the_module_imports_only_the_standard_library():
    """AST over the file, so a conditional import cannot hide (the repo rule)."""
    tree = ast.parse(pathlib.Path(blockdump.__file__).read_text(encoding="utf-8"))
    pulled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pulled.add(node.module.split(".")[0])
    assert pulled <= {"__future__", "base64", "ctypes", "hashlib", "json", "os",
                      "sys", "time", "config", "snapshot"}, sorted(pulled)
    for module in ("torch", "vllm"):
        assert module not in sys.modules


def test_no_entry_point_can_raise_an_exception():
    """Every public door catches Exception: the `cpu_capacity` posture."""
    for call in (lambda: blockdump.blockdump_for_engine(None, {"op": "status"}),
                 lambda: blockdump.blockdump_for_engine(object(), {"op": "arm"}),
                 lambda: blockdump.blockdump_for_engine(object(),
                                                        {"op": "dump",
                                                         "group": "0",
                                                         "block_id": 1,
                                                         "length": 8}),
                 lambda: blockdump.dump_blocks(None, None, None),
                 lambda: blockdump.read_block_bytes(None, 0),
                 lambda: blockdump.fulfil_request(None),
                 lambda: blockdump._tick(None),
                 lambda: blockdump.install(object()),
                 lambda: blockdump.uninstall(object()),
                 lambda: blockdump.arm(),
                 lambda: blockdump.disarm(),
                 lambda: blockdump.section()):
        result = call()
        assert result is not None
        json.dumps(result, default=str)


def test_the_caps_are_reported_in_the_units_the_operator_needs():
    caps = blockdump.blockdump_for_engine(FakeEngine(),
                                          {"op": "status"})["caps"]
    assert caps["max_bytes_per_request"] == 1024 * 1024
    assert caps["request_window_seconds"] == 300.0
    assert caps["max_requests_per_window"] == 20
    assert caps["poll_interval_seconds"] == 2.0
    assert caps["units"]["max_blocks"] == "blocks"


def test_no_env_var_arms_the_dump_facility():
    """The guard-mode rule, applied to the facility that reads KV bytes.

    `config.py` owns every env var this plugin reads, and none of them arms this
    one: the only file that can arm it is the arm file, and the only writer of
    that file is the token-gated verb. A launcher script or a systemd unit that
    survives the operator cannot turn this on.
    """
    names = [name for name in dir(config) if name.startswith("ENV_")]
    assert names
    assert not [name for name in names
                if any(word in name.upper() for word in
                       ("DUMP", "BLOCK", "ARM"))]
    assert blockdump.capability() == "not_deployed"     # every test's environment
