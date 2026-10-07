r"""The GPU-DIRECT WRITE: put restored bytes into the window and read them back.

`disk_gpu_write` is the one dev feature in this package that WRITES into live KV memory,
so this file is ordered by what a wrong answer here would cost, and every test says
which part of the ordering argument it is holding up:

1. OFF IS OFF, AND THE SEAMS SAY SO. The default posture issues no copy, allocates no
   pinned memory, starts no thread and moves no counter - proved with seams that RAISE
   if they are touched, not with a counter agreeing with itself. One environment
   variable is the whole difference between that and every comparison below.
2. THE WRITE IS THE POINT, AND THE COMPARISON IS THE PROOF. An armed write puts real
   bytes into a GPU block and the readback says so. The comparison's other side is the
   SPILL FILE and never the CPU slot the copy read from: comparing the readback with its
   own source would say `equal` forever, which is a mirror and not a witness. A test
   moves the FILE and watches the verdict move, with the copy untouched.
3. THE ORDERING IS AN OBSERVABLE, NOT A COMMENT. A fake stream journal records the order
   of every submission, so "our copy, our readbacks, then the fork's copy" is an
   assertion about a list. And the counters that carry the claim are a PAIR: run the
   fork's copy first and `readback_enqueued_before_original` stops while
   `event_already_submitted` starts, and nothing is written.
4. THE NEGATIVE CONTROLS CATCH OUR OWN MISTAKES. Shift ONLY the copy's destination to a
   neighbouring block and the witness must report `readback_mismatch_stable`, the single
   counter that means OUR ADDRESSING IS WRONG. Cut the copy's EXTENT in half and it is
   the same counter with a different `shape`. Let another stream rewrite the window
   between the two readbacks and it must be `readback_mismatch_gpu_changed_on_recheck`
   instead - the expected race, from which no action follows.
5. THE PINNED ROWS ARE RELEASED BY THE LOAD EVENT AND BY NOTHING ELSE. A row whose DMA
   may still be in flight is never handed out again: a slow watermark provably delays
   the release, provably refuses the next write for want of a buffer, and provably LEAKS
   rather than recycles the rows of an entry that ages out. Nothing is ever allocated on
   the step thread, which is the asymmetry that makes `no_pinned_buffer` mean anything.
6. THE REFUSALS REFUSE. A width that does not total, a missing or out-of-range GPU
   block, an entry whose read never settled, no stream, no thread, an empty pool, a copy
   that raises, a file that is not window-wide: each is a name and each leaves the entry
   ENTIRELY to the fork's copy. Nothing partial is ever written. The tail is never
   compared, and a mismatch just outside `W_g` is not a mismatch.
7. THE ORDER RULE IS THE ONE THE FILE IS BUILT ON. `sorted(name)`, and GPU region `name`
   paired with CPU region `name`. The two fake cache dicts are built with DIFFERENT
   insertion orders on purpose, so a positional pairing - what the fork's own
   `build_params` does - scrambles the window and this test fails.
8. IT COSTS NOTHING IT DID NOT SAY, AND SAYS EVERYTHING IT COSTS. The section has a
   stable key set and fits its byte budget even loaded with geometry, verdicts and fat
   reasons; the seam's cap had to move 5 -> 6 for this section to exist at all, and a
   test writes a rank file carrying all SIX and checks the required fields survive under
   the file cap. The verb's two halves and its POST-only rule are pinned, and
   `/snapshot` publishes the same `section()`.
9. THE WRITER STARTS ITSELF, AND SAYS WHETHER IT IS ALIVE (0.2.44). 0.2.43 could not
   write a single byte in the field: rows are allocated only on the module's thread, the
   thread was started only AFTER `_take_rows` had returned rows, and a cold pool refused
   the write BEFORE that line - so no thread, no rows, no thread. Every test here missed
   it because the `armed` fixture supplies both missing pieces (a `_ensure_thread` that
   says yes without starting anything, and a pool filled by hand), which is the lesson
   this section of the file exists to teach: a test that provides what production does
   not is worse than no test. So ONE test drives the production entry point cold - real
   thread, real sample rate, nothing pre-armed - and asserts a write actually happens
   with every pinned allocation on the module's thread and none on the step thread's.
   Beside it: an armed registration starts the thread and allocates nothing, the cold
   refusal and a drained pool are counted and named apart (`pool_never_filled`), and
   `liveness` answers "is this witness alive?" with the same key set in all four states.

WHAT IS FAKE, STATED WHERE IT CANNOT BE MISSED. There is no CUDA device here. The three
expressions that touch one are the module's own seams (`_copy_host_to_device`,
`_copy_device_to_host`, `_allocate_row`) and the tests back them with REAL ctypes
memory, so the bytes compared are bytes and not a mock's return value; the fake
`new_empty` answers with a pinned-shaped host row that the sanctioned host reader can
read. The stream is a fake context manager that journals `__enter__`/`__exit__`, which
is the shape `_stream_context` accepts when a process has not imported torch; a separate
test pins the REAL resolution path against a stub `torch` in `sys.modules`, because that
lookup is the only way this package reaches a heavy module without importing it. Two
tests drive the REAL restore path (`disk_restore.worker_bind` then
`disk_restore.worker_before_load`) so the hook site, the field names and the order
against the fork's copy are pinned by the product and not by a fake batch. Where a test
has to build a state the product would not produce by itself (a slot whose tail
disagrees with the file's, to exercise the tail refusal) it says so in the test.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import types

import pytest
from test_disk_restore import DIGEST, WIDTH, Meta, Worker, key_hex, pack, resolve
from test_disk_restore_verify import (
    GROUP_WIDTHS,
    CacheConfig,
    batch_of,
    job,
    write_window,
)

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    backport,
    config,
    disk_gpu_write,
    disk_index,
    disk_layout,
    disk_restore,
    disk_spill,
    disk_store,
    endpoint,
    snapshot,
    worker_diag,
)

BLOCKS = 8                    # blocks in the fake GPU region
GPU_BLOCK = 5
CPU_BLOCK = 3
KEY16 = "a36ec46fe316bd0d"
NARROW_GROUP = 3              # the 1024 B group inside a 4096 B window


# --- the fakes that can be copied into ----------------------------------------
class Journal:
    """The order of every submission, as an observable.

    `current` is the stream that is current right now, so a submission recorded while no
    stream has been entered shows up as exactly that - which is the failure the whole
    ordering argument exists to exclude, and the reason the stream is part of every
    record rather than a detail a reader has to trust.
    """

    def __init__(self) -> None:
        self.entries: list = []
        self.current = None

    def add(self, *fields) -> None:
        self.entries.append((str(fields[0]), *fields[1:], self.current))

    @property
    def kinds(self):
        return [entry[0] for entry in self.entries]

    def of(self, kind):
        return [entry for entry in self.entries if entry[0] == kind]


class FakeStream:
    """`worker.load_stream`, as far as a test can have one.

    It IS a context manager, which is the shape `disk_gpu_write._stream_context` accepts
    with no torch in the process at all - the honest way to test "the copy went on the
    tier's own stream" without a device.
    """

    def __init__(self, journal, name="load") -> None:
        self.journal = journal
        self.name = name
        self.entered = 0

    def __enter__(self):
        self.journal.add("enter", self.name)
        self.journal.current = self.name
        self.entered += 1
        return self

    def __exit__(self, *_exc) -> None:
        self.journal.add("exit", self.name)
        self.journal.current = None


class MemRow:
    """One row of one region: what `narrow(0, block, 1)` answers.

    `copy_` is the direction-deciding half of the host<->device copy. The two `device`
    fields say which way the bytes go, and equal byte counts are demanded exactly as
    `cuda_mem_ops.build_params` demands them of the fork's own copy. A copy between two
    same-kind memories raises: a fake that accepted it could "pass" a test in which no
    byte ever crossed the boundary this module is all about.
    """

    def __init__(self, region, block: int) -> None:
        self.region = region
        self.block = int(block)

    @property
    def device(self):
        return self.region.device

    def data_ptr(self) -> int:
        return self.region.data_ptr() + self.block * self.region.bytes_per_row

    @property
    def nbytes(self) -> int:
        return self.region.bytes_per_row

    def copy_(self, src, non_blocking=False):  # noqa: ARG002 - the fake IS the DMA
        if int(src.nbytes) != self.nbytes:
            raise ValueError(f"copy_ of {src.nbytes} B into {self.nbytes} B")
        src_is_host = "cpu" in str(src.device).lower()
        dst_is_host = "cpu" in str(self.device).lower()
        if src_is_host == dst_is_host:
            raise ValueError("a host<->device copy needs one host and one device side")
        # The device side is whichever of the two is not host; memmove is the DMA.
        device, host = (self, src) if not dst_is_host else (src, self)
        ctypes.memmove(device.data_ptr(), host.data_ptr(), self.nbytes)
        self.region.journal.add("h2d" if src_is_host else "d2h", self.region.name,
                                self.block, id(self))
        return self


class PinnedRow:
    """A pinned host row: the shape `region.new_empty(..., pin_memory=True)` answers.

    `device` says `cpu` for a reason bigger than tidiness - `disk_shadow._host_landed`
    reads it, and it is the one guard standing between this module's host read and a
    SIGSEGV that no `except Exception` survives. The row is ONE BLOCK of one region,
    which is why the pool keys by name and width: a row sized for a 2048 B region must
    never be handed to a 4096 B one. `copy_` is the readback half, so it records the row
    it read rather than itself - which is what lets a test see that the two readbacks
    used two different rows.
    """

    def __init__(self, elems: int, itemsize=1, journal=None, name="row") -> None:
        self.elems = int(elems)
        self._itemsize = int(itemsize)
        self.buffer = ctypes.create_string_buffer(self.elems * self._itemsize)
        self.device = "cpu"
        self.shape = (1, self.elems)
        self.nbytes = self.elems * self._itemsize
        self.journal = journal
        self.name = name

    def data_ptr(self) -> int:
        return ctypes.addressof(self.buffer)

    def stride(self, axis) -> int:
        return self.elems if axis == 0 else 1

    def element_size(self) -> int:
        return self._itemsize

    def copy_(self, src, non_blocking=False):  # noqa: ARG002 - the fake IS the DMA
        if int(src.nbytes) != self.nbytes:
            raise ValueError(f"readback of {src.nbytes} B into {self.nbytes} B")
        if "cpu" in str(src.device).lower():
            raise ValueError("a readback must read from a device row")
        ctypes.memmove(self.data_ptr(), src.data_ptr(), self.nbytes)
        (src.region.journal or self.journal).add("d2h", src.region.name, src.block,
                                                 id(self))
        return self


class FakeRegion:
    """One region tensor - GPU or arena - with the fork's shape and real memory.

    `device` is the only difference between the two sides, and `narrow`/`copy_` are the
    only methods this module's seams use. `bytes_per_row` is `stride(0) *
    element_size()`, the same product `build_params` and
    `disk_spill.measured_slot_bytes` compute. `new_empty` counts, so "the step thread
    never allocates" is checkable.
    """

    def __init__(self, name, blocks, bytes_per_row, *, device="cuda:0", journal=None,
                 itemsize=1) -> None:
        self.name = name
        self.blocks = int(blocks)
        self.bytes_per_row = int(bytes_per_row)
        self._itemsize = int(itemsize)
        self.device = device
        self.journal = journal or Journal()
        self.buffer = ctypes.create_string_buffer(self.blocks * self.bytes_per_row)
        self.allocated = 0
        self.set_everywhere(b"")

    # --- the shape a tensor answers ---
    @property
    def shape(self):
        return (self.blocks, self.bytes_per_row // self._itemsize)

    @property
    def nbytes(self):
        return self.blocks * self.bytes_per_row

    def data_ptr(self) -> int:
        return ctypes.addressof(self.buffer)

    def stride(self, axis) -> int:
        return self.bytes_per_row // self._itemsize if axis == 0 else self._itemsize

    def element_size(self) -> int:
        return self._itemsize

    def narrow(self, axis, start, count) -> MemRow:  # noqa: ARG002 - one block
        return MemRow(self, int(start))

    def new_empty(self, size, *, device="cpu", pin_memory=False, dtype=None):
        assert "cpu" in str(device) and pin_memory, \
            "the module must ask for a PINNED host row or its readback is synchronous"
        self.allocated += 1
        return PinnedRow(int(size[1]), self._itemsize, self.journal, self.name)

    # --- the memory ---
    def _pattern(self, index, filler) -> bytes:
        if not filler:
            # Distinct per block AND per region name, so reading the wrong block, the
            # wrong region, or the regions in the wrong order cannot match by accident.
            salt = sum(bytearray(self.name.encode()))
            return bytes(((index * 11 + offset * 3 + salt) % 251)
                         for offset in range(self.bytes_per_row))
        repeat = bytes(filler) * (self.bytes_per_row // len(filler) + 1)
        return repeat[:self.bytes_per_row]

    def set_everywhere(self, filler: bytes) -> None:
        base = self.data_ptr()
        for index in range(self.blocks):
            ctypes.memmove(base + index * self.bytes_per_row,
                           self._pattern(index, filler), self.bytes_per_row)

    def set_block(self, index, raw: bytes) -> None:
        ctypes.memmove(self.data_ptr() + int(index) * self.bytes_per_row,
                       raw[:self.bytes_per_row], self.bytes_per_row)

    def block_bytes(self, index) -> bytes:
        return ctypes.string_at(self.data_ptr() + int(index) * self.bytes_per_row,
                                self.bytes_per_row)


class WriteWorker(Worker):
    """`SimpleCPUOffloadWorker` plus the fields ONLY a GPU-direct write touches.

    It IS `test_disk_restore.Worker`, so the CPU side, the bind and the apply point are
    the same fake that suite pins. Three things are added because only this module uses
    them: `gpu_kv_caches`/`load_stream` (the destination and the stream), `_load_hwm`
    (the completion witness) and `_load_events` (the ORDERING witness - the fork appends
    `(event_idx, event)` to it AFTER `copy_blocks`, so its contents are what prove our
    submissions went first).

    `load_hwm` DEFAULTS TO A HIGH VALUE, and that is a statement about the tests, not
    the product: the watermark is pinned by its own two tests, and every other test here
    would otherwise be asserting the gate instead of the classification. A test that
    wants the wait says `load_hwm=-1`.
    """

    def __init__(self, *, gpu=None, cpu=None, widths=GROUP_WIDTHS, load_hwm=1000,
                 journal=None) -> None:
        self.journal = journal or Journal()
        gpu = {"g0": FakeRegion("g0", BLOCKS, WIDTH, journal=self.journal)} \
            if gpu is None else gpu
        cpu = {"g0": FakeRegion("g0", 6, WIDTH, device="cpu",
                                journal=self.journal)} if cpu is None else cpu
        super().__init__(dict(cpu))
        self.gpu_kv_caches = dict(gpu)
        self.kv_cache_config = CacheConfig(widths)
        self.load_stream = FakeStream(self.journal)
        self._load_hwm = load_hwm
        self._load_events: list = []

    @property
    def gpu(self):
        return self.gpu_kv_caches["g0"]

    @property
    def arena(self):
        return self.cpu_kv_caches["g0"]

    def start_load_kv(self):
        """THE FORK'S OWN BODY, as far as the ordering is concerned.

        The real one calls `backend.launch_copy`, which only queues; the copy thread
        then does `copy_blocks` and appends `(event_idx, event)`. The observable the
        module reads is that list, so the fake appends here, and `fork-copy` in the
        journal is the marker every ordering assertion measures itself against.
        """
        super().start_load_kv()
        event = getattr(self._connector_metadata, "load_event", None) \
            if self._connector_metadata is not None else None
        if event is not None and int(event) >= 0:
            self._load_events.append((int(event), object()))
        self.journal.add("fork-copy", event)

    def complete(self, event) -> None:
        """The copy stream reports this load event's CUDA event as done."""
        self._load_hwm = max(self._load_hwm, int(event))


# --- the stage -----------------------------------------------------------------
def counters():
    return disk_gpu_write.counters()


def armed_batch(*, gpu_block=GPU_BLOCK, cpu_block=CPU_BLOCK, width=WIDTH, path=None,
                group=0, event=11) -> dict:
    """One bound batch with one entry, exactly the shape `_worker_bind` stores."""
    entry = job(path or "/tmp/kvslot-unused", group=group, gpu_block=gpu_block,
                cpu_block=cpu_block, key16=KEY16, width=width)
    return batch_of([entry], event=event)


def one_write(worker, **kwargs) -> dict:
    """One armed apply-point call, with the interval floor lifted.

    The floor is pinned by its OWN test; every other test that reaches a second write
    would otherwise be asserting the throttle instead of the write. Touching the private
    clock is the honest way to say "the floor is not what this test is about".
    """
    disk_gpu_write._state["last_write_mono"] = None
    return disk_gpu_write.observe_apply(worker, armed_batch(**kwargs), {})


def fill_pool(worker) -> int:
    """Fill the pinned pool the way this module's OWN thread does.

    Calling the production function is the point: the step thread must never allocate,
    so a test that "allocated" rows some other way would not be testing the one
    asymmetry that makes `no_pinned_buffer` mean anything.
    """
    return disk_gpu_write._refill_pools(worker)


def one_pass(worker) -> dict:
    disk_gpu_write._state["last_pass_mono"] = None
    return disk_gpu_write.run_pass(worker)


def seed_window(worker, path, *, block=CPU_BLOCK, name="g0") -> None:
    """Put the file's bytes in the CPU slot, as `_apply_buffer` would have.

    The CPU slot is the SOURCE of the copy, and filling it here keeps the test honest
    about which bytes the module is responsible for: the memmove out of the read pool's
    buffer is `disk_restore`'s job, pinned in its own suite and driven for real by the
    two tests at the end of this file.
    """
    with open(path, "rb") as handle:
        worker.cpu_kv_caches[name].set_block(block, handle.read(WIDTH))


@pytest.fixture
def journal():
    return Journal()


@pytest.fixture
def worker(journal):
    """A worker with a GPU region, an arena region, five groups and a fake stream."""
    return WriteWorker(journal=journal)


@pytest.fixture
def untouched(monkeypatch):
    """Every device-touching seam made LOUD. "Off" is proved against these.

    A test that asserted "the counters are zero" would be satisfied by a module that
    counted nothing; these raise, so a write attempt in a test that forbids one fails
    with the seam's own name in the traceback.
    """
    def boom(*args, **kwargs):
        raise AssertionError("a device seam was touched while the writer must be off")

    monkeypatch.setattr(disk_gpu_write, "_copy_host_to_device", boom)
    monkeypatch.setattr(disk_gpu_write, "_copy_device_to_host", boom)
    monkeypatch.setattr(disk_gpu_write, "_allocate_row", boom)
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: False)
    return True


@pytest.fixture
def armed(monkeypatch):
    """The environment's own yes. DEFAULT OFF is the posture; this is the opt-in.

    The env var rather than `set_enabled`, because the env path is what arms a WORKER -
    the process that applies a restore and the only process that can write a window. The
    thread is replaced so the tests drive `run_pass` themselves instead of racing a
    daemon.

    READ THIS BEFORE USING IT AS A TEMPLATE. This fixture hands the module the two
    things production never provides - a `_ensure_thread` that says yes WITHOUT starting
    a thread, and (with `fill_pool`) rows nobody had to allocate - and that is exactly
    how 0.2.43 shipped a writer that could not write: 33 green tests, every one of them
    reaching a write through help the field does not give. Determinism is the reason it
    stays, but the cold path is pinned ONLY by `armed_cold` and
    `test_a_cold_writer_starts_its_own_thread_and_then_actually_writes`; a new test that
    needs to know whether the module can start itself may not borrow this shortcut.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    disk_gpu_write._state["env_resolved"] = False
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)
    # THE SAMPLE LIFTED, and its own test below puts it back. `SAMPLE_EVERY_N_APPLIED`
    # is 64 in the product, so a test that called the apply point twice would have its
    # second write `sampled_out` and would be asserting the sampler instead of its
    # subject.
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 1)
    return True


@pytest.fixture
def copy_seams(monkeypatch):
    """The two copy expressions, wrapped so a test can shift, halve or break them.

    Wrapping rather than replacing: the real fake copy still runs, so a test that shifts
    the DESTINATION is a test in which bytes really move somewhere, and the WITNESS has
    to be what catches it rather than the fake refusing to cooperate.
    """
    real_h2d = disk_gpu_write._copy_host_to_device
    real_d2h = disk_gpu_write._copy_device_to_host
    holder = {"shift": 0, "extent": None, "fail_h2d": None, "fail_d2h": None,
              "churn_on": None, "churn_to": b"\xAA" * WIDTH, "h2d": [], "d2h": []}

    def h2d(dst_region, dst_block, src_region, src_block):
        if holder["fail_h2d"] is not None:
            raise holder["fail_h2d"]
        if holder["extent"] is not None:
            # A stand-in for a wrong EXTENT: move only the leading N bytes.
            holder["h2d"].append((dst_region.name, int(dst_block), int(src_block)))
            width = int(holder["extent"])
            ctypes.memmove(dst_region.data_ptr() + int(dst_block) *
                           dst_region.bytes_per_row,
                           src_region.data_ptr() + int(src_block) *
                           src_region.bytes_per_row, width)
            dst_region.journal.add("h2d", dst_region.name, int(dst_block), None)
            return None
        # THE BLOCK THE BYTES ACTUALLY WENT TO, which is what a test needs to see: the
        # shift is applied here, so `h2d` is a record of where the copy landed rather
        # than of what the module asked for - and the difference is the whole negative
        # control.
        target = int(dst_block) + holder["shift"]
        holder["h2d"].append((dst_region.name, target, int(src_block)))
        return real_h2d(dst_region, target, src_region, src_block)

    def d2h(row, src_region, src_block):
        holder["d2h"].append((src_region.name, int(src_block)))
        if holder["fail_d2h"] is not None:
            raise holder["fail_d2h"]
        if holder["churn_on"] is not None and len(holder["d2h"]) == holder["churn_on"]:
            # Another stream rewrites the window. A direct write here is exactly what an
            # unordered stream does to a stream-ordered readback.
            src_region.set_block(int(src_block), holder["churn_to"])
        return real_d2h(row, src_region, src_block)

    monkeypatch.setattr(disk_gpu_write, "_copy_host_to_device", h2d)
    monkeypatch.setattr(disk_gpu_write, "_copy_device_to_host", d2h)
    return holder


# --- 1: default off -----------------------------------------------------------
def test_the_default_boot_writes_nothing_and_touches_no_device(worker, untouched):
    """The same apply-point call, unarmed: no copy, no allocation, no thread, no
    counter.

    The mirror of every comparison below, run against a worker with real regions, so
    "off" cannot be an artifact of a worker that had nothing to write.
    """
    assert disk_gpu_write.effective_enabled() is False
    assert fill_pool(worker) == 0          # the allocator refuses, and nothing retries
    assert worker.gpu.allocated == 0
    outcome = disk_gpu_write.observe_apply(worker, armed_batch(), {})
    assert outcome["seen"] == 0 and outcome["written"] == 0
    assert outcome["reason"] == "disabled"
    assert counters() == {key: 0 for key in disk_gpu_write._COUNTER_KEYS}
    assert worker.journal.entries == []
    assert disk_gpu_write.view()["in_flight"] == {}


def test_an_unrecognised_env_value_leaves_the_writer_off_and_says_so(
        monkeypatch, worker, untouched):
    """The direction a typo moves in is the direction that writes nothing.

    `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE=maybe` is not "on", and it is not silently off
    either: `env_recognised` says so, so an operator who believes he armed the witness
    can read the rank file and find out instead of being told a lie by a zero.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "maybe")
    disk_gpu_write._state["env_resolved"] = False
    assert disk_gpu_write.effective_enabled() is False
    state = disk_gpu_write.enable_state()
    assert state["env_recognised"] is False and state["env_raw"] == "maybe"
    refused = disk_gpu_write.observe_apply(worker, armed_batch(), {})
    assert refused["reason"] == "disabled"
    assert worker.journal.entries == []


# --- 2: the write and its witness ---------------------------------------------
def test_an_armed_write_puts_the_file_s_bytes_in_the_window_and_says_equal(
        armed, worker, tmp_path):
    """THE ROUND TRIP, with the module's own copies and the module's own comparison.

    `equal` here means: our copy really moved the arena's bytes into GPU block 5, the
    post-copy readback really read them back out of the fake device's real memory, and
    they match the SPILL FILE. A write that landed in the wrong place, or a comparison
    that read its own source, fails this test.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    seed_window(worker, path)
    assert fill_pool(worker) == 2          # one pre row and one post row
    assert one_write(worker, path=path)["written"] == 1
    assert worker.gpu.block_bytes(GPU_BLOCK)[:64] == b"\x5a" * 64, \
        "the copy did not put the file's bytes in the window"
    assert counters()["writes"] == 1 and counters()["bytes_written"] == WIDTH
    assert counters()["regions_written"] == 1
    assert counters()["readbacks_enqueued"] == 2
    worker.complete(11)
    assert one_pass(worker)["results"][0]["equal"] is True
    assert counters()["readback_equal"] == 1 and counters()["readback_mismatch"] == 0
    # THE ROWS ARE BACK, and only after the watermark passed.
    assert disk_gpu_write.view()["in_flight"] == {}
    assert len(disk_gpu_write._state["pool"][("g0", WIDTH)]) == 2


def test_the_file_is_the_oracle_and_not_the_cpu_slot_the_copy_read(armed, worker,
                                                                   tmp_path):
    """Moving the FILE moves the verdict, with the copy untouched.

    This is the test that separates a witness from a mirror: comparing the readback
    against the CPU slot it was copied from says `equal` for every write this module can
    ever perform, and would never see a wrong address. The file is the input the verdict
    answers to - so replacing it, with the GPU window and the slot exactly as our copy
    left them, changes the answer.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    assert one_write(worker, path=path)["written"] == 1
    window = worker.gpu.block_bytes(GPU_BLOCK)
    assert window == worker.arena.block_bytes(CPU_BLOCK)     # the copy landed
    write_window(path, b"\x11" * 64)                        # THE ORACLE MOVES
    worker.complete(11)
    verdict = one_pass(worker)["results"][0]
    assert verdict["equal"] is False
    assert verdict["first_difference"] == 0
    assert counters()["readback_mismatch"] == 1
    # The module cannot tell "the file is not what the window holds" from "somebody else
    # wrote the window" - both read the same way - so it reports the expected-race
    # category rather than inventing a third story. The sibling verifier, which reads
    # the window AFTER the fork's copy, is the one that owns a file-level finding.
    assert verdict["category"] == "readback_mismatch_gpu_changed_on_recheck"


# --- 3: the ordering, as an observable ----------------------------------------
def test_the_journal_reads_our_copy_then_our_readbacks_then_the_fork(armed, worker,
                                                                    tmp_path):
    """THE ORDERING ARGUMENT, as a list.

    `pre-readback -> our copy -> post-readback -> the fork's copy`, every submission
    recorded while `load_stream` was current. The fork's own submission is in the same
    list, appended by the fake's `start_load_kv` exactly where the real one queues the
    copy, which is what makes this an ordering test and not a story about order.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x77" * 32)
    seed_window(worker, path)
    fill_pool(worker)
    assert one_write(worker, path=path)["written"] == 1
    worker.bind_connector_metadata(Meta(load_event=11, cpu=[CPU_BLOCK], gpu=[GPU_BLOCK],
                                        reqs={11: ["req-1"]}))
    worker.start_load_kv()
    journal = worker.journal
    assert journal.kinds == ["enter", "d2h", "h2d", "d2h", "exit", "fork-copy"], \
        journal.kinds
    for entry in journal.of("h2d") + journal.of("d2h"):
        assert entry[-1] == "load", f"{entry} was not submitted on load_stream"
    assert len({entry[3] for entry in journal.of("d2h")}) == 2, \
        "the two readbacks wrote into the same row"
    assert {entry[2] for entry in journal.of("d2h")} == {GPU_BLOCK}, \
        "a readback read a block the entry did not name"
    assert counters()["readback_enqueued_before_original"] == 1
    assert counters()["readback_enqueued_after_original"] == 0


def test_a_hook_moved_after_the_original_stops_moving_the_order_counter(armed, worker,
                                                                       tmp_path):
    """THE COUNTERS THAT MAKE THE ORDERING CLAIM CHECKABLE, proved by breaking the
    claim.

    Once the fork has submitted the copy for this load event, a pre-copy readback would
    read the FORK's bytes and our copy would be a racing overwrite. The witness is the
    fork's own `_load_events`, and the counters are a pair: the ordering counter stops
    moving, `event_already_submitted` and `readback_enqueued_after_original` start, and
    NOTHING is written. A worker with no ledger at all is its own named refusal rather
    than a guess.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x77" * 32)
    seed_window(worker, path)
    fill_pool(worker)
    worker.bind_connector_metadata(Meta(load_event=11, cpu=[CPU_BLOCK], gpu=[GPU_BLOCK],
                                        reqs={11: ["req-1"]}))
    worker.start_load_kv()                     # THE FORK FIRST: the ordering is gone
    before = counters()["readback_enqueued_before_original"]
    outcome = one_write(worker, path=path)
    assert outcome["written"] == 0 and outcome["reason"] == "event_already_submitted"
    assert counters()["readback_enqueued_before_original"] == before
    assert counters()["event_already_submitted"] == 1
    assert counters()["readback_enqueued_after_original"] == 1
    assert counters()["writes"] == 0
    assert worker.journal.of("h2d") == []
    del worker._load_events
    assert one_write(worker, path=path)["reason"] == "no_event_ledger"
    assert counters()["no_event_ledger"] == 1


# --- 4: the pinned rows' lifetime ---------------------------------------------
def test_a_slow_load_event_provably_delays_the_row_release_and_the_next_write(
        armed, worker, tmp_path):
    """THE PINNED-BUFFER LIFETIME, as an assertion about a pool.

    A row handed to a `copy_(non_blocking=True)` that has not completed is still being
    written. With the watermark at a fresh worker's -1 the rows MUST stay held: the pass
    refuses to read them, the pool is provably empty, and the next write is refused for
    want of a buffer rather than issued unwitnessed. Completing the event is the ONLY
    thing that frees them.
    """
    worker._load_hwm = -1
    path = write_window(str(tmp_path / "kvslot-window"), b"\x99" * 16)
    seed_window(worker, path)
    assert fill_pool(worker) == 2
    assert one_write(worker, path=path)["written"] == 1
    assert len(disk_gpu_write.view()["in_flight"]) == 1, "no rows are being held"
    assert disk_gpu_write._state["pool"][("g0", WIDTH)] == [], \
        "the pool still has rows: the record did not take both"
    verdict = one_pass(worker)
    assert verdict["results"] == [] and verdict["waiting"] == 1
    assert len(disk_gpu_write.view()["in_flight"]) == 1, \
        "a pass released rows whose load event had not passed"
    assert one_write(worker, path=path)["written"] == 0
    assert counters()["no_pinned_buffer"] == 1
    assert counters()["writes"] == 1
    worker.complete(11)                        # only the fork's own event lets go
    assert one_pass(worker)["results"][0]["equal"] is True
    assert len(disk_gpu_write._state["pool"][("g0", WIDTH)]) == 2
    assert one_write(worker, path=path)["written"] == 1
    assert counters()["writes"] == 2


def test_an_entry_that_ages_out_leaks_its_rows_instead_of_recycling_them(armed, worker,
                                                                        tmp_path,
                                                                        monkeypatch):
    """THE LEAK DIRECTION: a row whose DMA state is unknowable is never handed out
    again.

    An entry whose load event never passes may still have a copy in flight writing one
    of its rows, so the rows are dropped WITHOUT going back to the pool and
    `buffers_leaked` says why. Reusing them would be the silent-corruption shape this
    module exists not to have. The byte accounting DOES come back down - a leak costs
    memory once and must not creep upward until a long boot could no longer allocate -
    so the pool refills with fresh rows on the module's thread, and the property a
    reader can rely on is that a leaked row is never a witness twice. In-process, with
    no thread to refill, the next write is refused with `no_pinned_buffer`, which is the
    honest direction for a witness that cannot afford one.
    """
    worker._load_hwm = -1
    path = write_window(str(tmp_path / "kvslot-window"), b"\x99" * 16)
    seed_window(worker, path)
    fill_pool(worker)
    one_write(worker, path=path)
    monkeypatch.setattr(disk_gpu_write, "JOB_MAX_AGE_SECONDS", -1.0)
    assert one_pass(worker)["results"][0]["reason"] == "dropped_waiting_dma"
    assert counters()["buffers_leaked"] == 2
    assert disk_gpu_write.view()["in_flight"] == {}
    assert disk_gpu_write._state["pool"][("g0", WIDTH)] == []
    assert disk_gpu_write._state["pinned_bytes"] == 0, \
        "a leak kept the pool's byte accounting climbing, so a long boot would " \
        "end up starved by its own leaks"
    assert one_write(worker, path=path)["reason"] == "no_pinned_buffer"


def test_the_step_thread_never_allocates_pinned_memory(armed, worker, tmp_path):
    """THE STEP THREAD'S COST, measured rather than claimed.

    `new_empty` counts on the fake region, so this is a fact about calls and not about a
    comment. The first armed call after a boot refuses with `no_pinned_buffer` because
    the pool is empty - and `worker.gpu.allocated` proves the apply point allocated
    NOTHING to work around that. Only the module's own refill (what its daemon thread
    calls every tick) page-locks memory, and only then does a write happen.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x44" * 8)
    seed_window(worker, path)
    assert one_write(worker, path=path)["reason"] == "no_pinned_buffer"
    assert counters()["no_pinned_buffer"] == 1
    assert counters()["writes"] == 0
    assert worker.gpu.allocated == 0, "the apply point allocated pinned memory"
    assert fill_pool(worker) == 2
    assert worker.gpu.allocated == 2
    assert one_write(worker, path=path)["written"] == 1
    assert worker.gpu.allocated == 2, "the step thread allocated again"


def test_the_stream_context_resolves_through_sys_modules_and_never_imports(
        armed, worker, tmp_path, monkeypatch):
    """THE TORCH LOOKUP, pinned against a stub. `disk_shadow` refused to import torch;
    this is the third possibility - asking the process what it already has.

    A stream that is not itself a context manager must be wrapped by
    `torch.cuda.stream(...)`, which is how torch 2.10 spells it
    (`cuda/__init__.py:651-719` - `Stream` gained no `__enter__`). The lookup goes
    through `sys.modules`, so nothing is imported and a process that never imported
    torch gets `None` and a named refusal.
    """
    entered = []

    class BareStream:                       # a REAL `torch.cuda.Stream`: no __enter__
        pass

    class Ctx:
        def __enter__(self):
            entered.append("ctx")

        def __exit__(self, *_exc):
            return None

    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(stream=lambda stream: Ctx())
    monkeypatch.setitem(sys.modules, "torch", stub)
    context = disk_gpu_write._stream_context(BareStream())
    assert entered == []
    with context:
        assert entered == ["ctx"]
    # And with no torch in the process at all, the answer is None -> a named refusal.
    monkeypatch.setattr(disk_gpu_write, "_torch_module", lambda: None)
    assert disk_gpu_write._stream_context(BareStream()) is None
    path = write_window(str(tmp_path / "kvslot-window"), b"\x44" * 8)
    seed_window(worker, path)
    fill_pool(worker)
    worker.load_stream = BareStream()
    assert one_write(worker, path=path)["reason"] == "stream_unavailable"
    assert counters()["stream_unavailable"] == 1
    assert counters()["writes"] == 0


# --- 5: the negative controls -------------------------------------------------
def test_writing_a_neighbouring_block_is_reported_as_our_own_addressing(
        armed, worker, tmp_path, copy_seams):
    """THE NEGATIVE CONTROL, and the single counter it exists to move.

    Only the COPY's destination is shifted, by the test, one block along; the readbacks
    still read the block the entry named. So the window is exactly what it was before we
    touched it, which is the definition of "our copy went somewhere else" -
    `readback_mismatch_stable`, THE counter for our own addressing. A witness that
    reported this as an expected race, or as equal, could not see the one bug it was
    written for.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    seed_window(worker, path)
    copy_seams["shift"] = 1
    fill_pool(worker)
    assert one_write(worker, path=path)["written"] == 1
    assert copy_seams["h2d"] == [("g0", GPU_BLOCK + 1, CPU_BLOCK)]
    assert worker.gpu.block_bytes(GPU_BLOCK)[:64] != b"\x5a" * 64
    worker.complete(11)
    verdict = one_pass(worker)["results"][0]
    assert verdict["category"] == "readback_mismatch_stable"
    # `shape` and the before-state live in the SAMPLE, not in the pass result: the
    # result says which way the verdict went, the sample carries the evidence behind it.
    sample = disk_gpu_write.view()["last_verdict"]
    assert sample["shape"] == "window-unchanged"
    assert sample["window_matched_pre_copy"] is True
    assert counters()["readback_mismatch_stable"] == 1
    assert counters()["readback_mismatch_gpu_changed_on_recheck"] == 0
    assert counters()["readback_equal"] == 0


def test_a_partial_write_is_ours_too_and_says_so_by_its_shape(armed, worker, tmp_path,
                                                             copy_seams):
    """A copy that covers HALF the window is a width bug in OUR copy, not a rival
    writer.

    The window ends up "the file's bytes up to k, the old bytes from k on". Without the
    before-state that is indistinguishable from another stream; with it, the shape is
    decisive - and it is counted with the other findings about our own copy, with
    `shape` keeping the two distinguishable in the sample.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    seed_window(worker, path)
    copy_seams["extent"] = WIDTH // 2
    fill_pool(worker)
    assert one_write(worker, path=path)["written"] == 1
    window = worker.gpu.block_bytes(GPU_BLOCK)
    assert window[:WIDTH // 2] == worker.arena.block_bytes(CPU_BLOCK)[:WIDTH // 2]
    assert window[WIDTH // 2:] != worker.arena.block_bytes(CPU_BLOCK)[WIDTH // 2:]
    worker.complete(11)
    verdict = one_pass(worker)["results"][0]
    assert verdict["category"] == "readback_mismatch_stable"
    assert disk_gpu_write.view()["last_verdict"]["shape"] == "partial-span"
    assert counters()["readback_mismatch_gpu_changed_on_recheck"] == 0


def test_a_window_rewritten_by_another_stream_is_the_expected_race(
        armed, worker, tmp_path, copy_seams):
    """`readback_mismatch_gpu_changed_on_recheck`, and NOT the stable one.

    Another stream rewrites the window between our copy and the post-copy readback. The
    before-state rules out "nothing happened" and the file rules out "ours landed", so
    the only reading left is a block somebody else owns now - an expected race, from
    which no action follows. If this test moved the OTHER counter, the module would be
    reporting the engine's normal traffic as a bug in itself, forever.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    seed_window(worker, path)
    copy_seams["churn_on"] = 2       # the second d2h is the post-copy one
    fill_pool(worker)
    assert one_write(worker, path=path)["written"] == 1
    worker.complete(11)
    verdict = one_pass(worker)["results"][0]
    assert verdict["category"] == "readback_mismatch_gpu_changed_on_recheck"
    assert disk_gpu_write.view()["last_verdict"]["shape"] is None
    assert counters()["readback_mismatch_gpu_changed_on_recheck"] == 1
    assert counters()["readback_mismatch_stable"] == 0


# --- 6: the refusals refuse ---------------------------------------------------
@pytest.mark.parametrize("field,value,kind", [
    ("bytes", WIDTH + 1, "width_mismatch"),
    ("gpu_block_id", None, "no_gpu_block"),
    ("gpu_block_id", 999, "no_gpu_block"),
    ("cpu_block_id", None, "no_cpu_block"),
])
def test_a_plan_that_cannot_be_right_writes_nothing(armed, worker, tmp_path, field,
                                                   value, kind):
    """A refusal is a NAME, and NOTHING PARTIAL IS EVER WRITTEN.

    The width rule is the fork's own per-region assertion (`cuda_mem_ops.build_params`)
    plus `_plan_regions`' total: a plan whose regions do not total the entry's own byte
    count describes a different engine, and writing half a window is the corruption this
    function exists to make impossible. The block checks are the same idea one level
    down - a block outside the region's `shape[0]` is an address that could leave the
    allocation, and the `copy_` that would follow it is a segfault rather than an
    exception.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 8)
    seed_window(worker, path)
    fill_pool(worker)
    entry = job(path, group=0, gpu_block=GPU_BLOCK, cpu_block=CPU_BLOCK, key16=KEY16,
                width=WIDTH)
    entry[field] = value
    disk_gpu_write._state["last_write_mono"] = None
    outcome = disk_gpu_write.observe_apply(worker, batch_of([entry], event=11), {})
    assert outcome["written"] == 0 and outcome["reason"] == kind
    assert counters()[kind] == 1
    assert counters()["writes"] == 0
    assert worker.journal.of("h2d") == []
    assert worker.journal.of("d2h") == []
    # The pool is untouched: a refusal took no rows, so it has none to leak.
    assert len(disk_gpu_write._state["pool"][("g0", WIDTH)]) == 2


def test_an_entry_whose_read_never_settled_is_counted_and_never_written(armed, worker,
                                                                       tmp_path):
    """`read_not_settled`, with the NAME of the reason kept and its family respected.

    An entry the apply point dropped because its bytes never arrived is not a candidate
    for a write. Counting only the settle family (`not_settled`, `incomplete`,
    `deadline_expired`) is what keeps this counter from meaning "a read failed" - a
    short read is a different diagnosis and must not move it.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 8)
    seed_window(worker, path)
    fill_pool(worker)
    batch = armed_batch(path=path)
    dropped = disk_gpu_write.observe_apply(worker, batch, {0: "not_settled"})
    assert dropped["written"] == 0
    assert counters()["read_not_settled"] == 1
    assert counters()["unsettled_entries"] == 1
    assert counters()["writes"] == 0
    disk_gpu_write.observe_apply(worker, batch, {0: "short_read"})
    assert counters()["read_not_settled"] == 1, \
        "a short read is not a settle problem and must not be counted as one"


def test_the_sample_rate_and_the_step_cap_decide_which_applies_get_written(
        armed, worker, tmp_path, monkeypatch):
    """THE DENOMINATOR AND THE CEILING, both moved in one test.

    `applies_seen` is counted BEFORE the sample decision and every geometry test, so the
    sample's denominator is the restore traffic rather than the traffic that survived -
    "how much of my restore did the writer look at?" has to be a division an operator
    can actually perform. And `max_writes_per_step` is the step's ceiling: two eligible
    entries in one step is ONE write and a named `throttled_step`, because this module's
    whole cost model is that the step thread enqueues a bounded number of copies.
    """
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 2)
    path = write_window(str(tmp_path / "kvslot-window"), b"\x63" * 8)
    seed_window(worker, path)
    fill_pool(worker)
    first = disk_gpu_write.observe_apply(worker, armed_batch(path=path), {})
    assert first["written"] == 1 and first["seen"] == 1
    second = disk_gpu_write.observe_apply(worker, armed_batch(path=path), {})
    assert second["written"] == 0 and counters()["sampled_out"] == 1
    assert counters()["applies_seen"] == 2
    two = batch_of([job(path, group=0, gpu_block=GPU_BLOCK, cpu_block=CPU_BLOCK,
                        key16=KEY16, width=WIDTH),
                    job(path, group=0, gpu_block=GPU_BLOCK + 1, cpu_block=CPU_BLOCK,
                        key16=KEY16, width=WIDTH)], event=12)
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 1)
    # THE ROWS BACK BEFORE THE THIRD STEP. The two writes above hold both pinned rows of
    # this region, and a row is released only when its load event's watermark passes -
    # so without this pass the third step would be refused for want of a buffer and this
    # test would be asserting the pool instead of the two limits it is about.
    worker.complete(11)
    one_pass(worker)
    disk_gpu_write._state["last_write_mono"] = None
    third = disk_gpu_write.observe_apply(worker, two, {})
    assert third["seen"] == 2 and third["written"] == 1
    assert counters()["throttled_step"] == 1
    assert counters()["no_pinned_buffer"] == 0


def test_no_stream_and_no_thread_are_both_refusals_that_write_nothing(
        armed, worker, tmp_path, monkeypatch):
    """Two refusals whose consequence is identical: the fork's copy runs alone.

    Without `load_stream` there is no way to order our copy before the fork's, and a
    copy on the WRONG stream would break the ordering argument rather than weaken it -
    so this is a refusal and not a fallback to the default stream. Without a thread the
    rows could never be released and the write would be unwitnessed, so that is a
    refusal too, taken BEFORE anything is enqueued - which is why the pool still holds
    both rows.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 8)
    seed_window(worker, path)
    fill_pool(worker)
    worker.load_stream = None
    assert one_write(worker, path=path)["reason"] == "stream_unavailable"
    assert counters()["stream_unavailable"] == 1
    worker.load_stream = FakeStream(worker.journal)
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: False)
    assert one_write(worker, path=path)["reason"] == "refused_thread_unavailable"
    assert counters()["refused_thread_unavailable"] == 1
    assert len(disk_gpu_write._state["pool"][("g0", WIDTH)]) == 2
    assert worker.journal.of("h2d") == []


def test_a_copy_that_raises_is_a_refusal_and_leaks_rather_than_reuses(
        armed, worker, tmp_path, copy_seams):
    """A half-enqueued record is the dangerous one, and the module treats it that way.

    The rows go to `buffers_leaked` and NOT back to the pool: the stream may already
    hold a copy that writes one of them after it has been handed to the next entry. And
    the apply-point call still returns normally, because a diagnostic that can break a
    restore is an outage rather than a diagnostic.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 8)
    seed_window(worker, path)
    fill_pool(worker)
    copy_seams["fail_h2d"] = RuntimeError("the device said no")
    outcome = one_write(worker, path=path)
    assert outcome["written"] == 0 and outcome["reason"] == "write_failed"
    assert counters()["write_failed"] == 1
    assert counters()["buffers_leaked"] == 2
    assert disk_gpu_write._state["pool"][("g0", WIDTH)] == []
    assert disk_gpu_write.view()["in_flight"] == {}


def test_a_file_that_is_not_window_wide_refuses_the_comparison(armed, worker, tmp_path):
    """The file side keeps the sibling verifier's assertions, un-edited.

    A spill file is exactly one slot wide, so a short one is a geometry or a
    rank-crossing answer and comparing it would MANUFACTURE a mismatch. The module
    reuses `disk_restore_verify._read_file_head` rather than re-deriving the rule, which
    is why the counter's name is already the sibling's - and the WRITE still happened,
    because this is a refusal to compare and not a refusal to write.
    """
    path = str(tmp_path / "kvslot-short")
    write_window(path, b"\x5a" * 8, size=WIDTH - 1)
    worker.arena.set_block(CPU_BLOCK, b"\x01" * WIDTH)
    fill_pool(worker)
    assert one_write(worker, path=path)["written"] == 1
    worker.complete(11)
    verdict = one_pass(worker)["results"][0]
    assert verdict["reason"] == "file_not_window_wide"
    assert counters()["file_not_window_wide"] == 1
    assert counters()["checked"] == 0 and counters()["readback_equal"] == 0


@pytest.mark.parametrize("offset,expect_equal", [(5, False), (1024, True)])
def test_only_the_leading_w_g_bytes_are_compared(armed, worker, tmp_path, offset,
                                                expect_equal):
    """The tail belongs to other groups and is stale by construction - and the module
    refuses to look at it, in BOTH directions.

    Group 3 owns 1024 B of a 4096 B window. A difference at offset 5 is INSIDE the
    group's own bytes and must be a mismatch with its offset reported; a difference at
    1024 is OUTSIDE them and must NOT be one. If the tail were compared, every narrow
    group would report a mismatch forever and the one real finding would drown in it.
    The state is built by hand here (the slot's bytes are made to disagree with the
    file's at one offset) because what is under test is the WIDTH of the comparison, not
    the plausibility of the state.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    with open(path, "rb") as handle:
        raw = bytearray(handle.read(WIDTH))
    slot = bytearray(raw)
    slot[offset] = (slot[offset] + 1) % 251       # one byte, one place, on one side
    worker.arena.set_block(CPU_BLOCK, bytes(slot))
    fill_pool(worker)
    assert one_write(worker, path=path, group=NARROW_GROUP)["written"] == 1
    worker.complete(11)
    verdict = one_pass(worker)["results"][0]
    assert verdict["equal"] is expect_equal, verdict
    sample = disk_gpu_write.view()["last_verdict"]
    assert sample["compared_bytes"] == GROUP_WIDTHS[NARROW_GROUP]
    assert sample["tail_bytes_excluded"] == WIDTH - GROUP_WIDTHS[NARROW_GROUP]
    if expect_equal:
        assert verdict["bytes"] == GROUP_WIDTHS[NARROW_GROUP]
        assert counters()["readback_equal"] == 1
    else:
        assert verdict["first_difference"] == offset
        assert counters()["readback_mismatch"] == 1


# --- 7: the order rule --------------------------------------------------------
def test_the_regions_are_written_by_name_and_in_sorted_order(armed, journal, tmp_path):
    """`sorted(name)`, and GPU region `name` paired with CPU region `name` - THE ORDER
    THE FILE IS BUILT IN (`disk_spill.read_slot_bytes`, `disk_restore._plan_regions`).

    THE TWO DICTS ARE BUILT IN DIFFERENT INSERTION ORDERS ON PURPOSE. Pairing them
    positionally - what the fork's own `build_params` does, correctly, because it is
    handed the same dict twice - would here put `layers.2`'s bytes into `layers.10`'s
    GPU row, and the readback (a name-ordered concatenation through the sanctioned
    reader) would disagree with the file. That is the only way to pin an order rule: a
    test whose two dicts agree on insertion order cannot tell the two pairings apart.
    """
    widths = {"layers.10": 2048, "layers.2": 2048}
    gpu = {name: FakeRegion(name, BLOCKS, widths[name], journal=journal)
           for name in ("layers.2", "layers.10")}          # insertion order: 2, 10
    cpu = {name: FakeRegion(name, 6, widths[name], device="cpu", journal=journal)
           for name in ("layers.10", "layers.2")}          # insertion order: 10, 2
    worker = WriteWorker(journal=journal, gpu=gpu, cpu=cpu)
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 32)
    with open(path, "rb") as handle:
        raw = handle.read(WIDTH)
    # The CPU slots hold the file in ITS order: `layers.10` sorts first, so its row is
    # the file's first 2048 bytes. A positional pairing would read them the other way
    # round.
    cpu["layers.10"].set_block(CPU_BLOCK, raw[:2048])
    cpu["layers.2"].set_block(CPU_BLOCK, raw[2048:])
    assert fill_pool(worker) == 4
    assert one_write(worker, path=path)["written"] == 1
    assert [entry[1] for entry in journal.of("h2d")] == ["layers.10", "layers.2"], \
        "the writes did not run in sorted-name order"
    for name in widths:
        assert gpu[name].block_bytes(GPU_BLOCK) == cpu[name].block_bytes(CPU_BLOCK), \
            f"region {name} received another region's bytes"
    assert counters()["regions_written"] == 2
    assert counters()["readbacks_enqueued"] == 4
    worker.complete(11)
    assert one_pass(worker)["results"][0]["equal"] is True


def test_a_missing_cpu_region_of_the_same_name_is_refused(armed, journal, tmp_path):
    """No positional fallback, ever.

    A GPU region with no CPU region of the same name could only be paired by position,
    and one region's bytes in another's window is exactly the silent corruption the name
    rule exists to prevent - so it is a name, and nothing is written.
    """
    gpu = {"g0": FakeRegion("g0", BLOCKS, WIDTH, journal=journal),
           "g1": FakeRegion("g1", BLOCKS, WIDTH, journal=journal)}
    cpu = {"g0": FakeRegion("g0", 6, WIDTH, device="cpu", journal=journal)}
    worker = WriteWorker(journal=journal, gpu=gpu, cpu=cpu)
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 8)
    cpu["g0"].set_block(CPU_BLOCK, b"\x01" * WIDTH)
    fill_pool(worker)
    assert one_write(worker, path=path)["reason"] == "region_pair_missing"
    assert counters()["region_pair_missing"] == 1
    assert journal.of("h2d") == []


# --- 8: the view, the seam's cap, and the verb --------------------------------
def test_the_section_key_set_never_changes_and_fits_the_contribution_cap():
    """The degraded view and the fat view have the SAME keys, and both fit.

    `worker_diag` REFUSES an over-cap section rather than truncating it, so a section
    100 bytes too fat is a section that is ABSENT - on the boot with the most to report.
    The fat case is built with live geometry, a verdict, samples and full-length
    reasons, which is the payload the trim ladder has to survive; it must still shed
    something rather than overflow.
    """
    fresh = disk_gpu_write.section()
    # `liveness` is 0.2.44's, and it is IN THIS LIST ON PURPOSE: the 0.2.43 writer could
    # not write at all and its section had no field that said so. The list is the gate
    # that makes adding a published key a decision rather than an accident - it is not a
    # claim that the key set of 0.2.43 was the right one.
    assert set(fresh) == set(disk_gpu_write._COUNTER_KEYS) | {
        "reasons", "last_verdict", "mismatch_samples", "enabled", "enable_source",
        "env_var", "env_raw", "env_recognised", "latch", "sample_every_n_applied",
        "max_writes_per_step", "min_interval_seconds", "max_checks_per_pass",
        "job_max_age_seconds", "max_in_flight", "geometry", "pool", "thread_alive",
        "liveness", "order_rule", "meaning_rule", "section", "writer", "note",
        "trimmed"}
    disk_gpu_write._state["group_bytes"] = list(GROUP_WIDTHS)
    disk_gpu_write._state["window_bytes"] = WIDTH
    disk_gpu_write._state["last_verdict"] = {
        "category": "readback_mismatch_stable", "shape": "window-unchanged",
        "gpu_block": GPU_BLOCK, "group": 0, "event": 11, "key16": KEY16,
        "bytes": WIDTH, "compared_bytes": GROUP_WIDTHS[0], "tail_bytes_excluded": 0,
        "first_difference": 0, "scanned_bytes": 262144, "window_matched_pre_copy": True}
    for index in range(disk_gpu_write.MAX_REASON_SAMPLES + 4):
        disk_gpu_write._reason(f"seed{index}", "a condition described at length " * 6)
    fat = disk_gpu_write.section()
    assert set(fat) == set(fresh), "the key set moved when there was a lot to report"
    for payload in (fresh, fat):
        size = len(json.dumps(payload, separators=(",", ":"), default=str).encode())
        assert size <= disk_gpu_write.MAX_SECTION_BYTES, \
            f"{size} B against a {disk_gpu_write.MAX_SECTION_BYTES} B budget"
        assert size <= worker_diag.MAX_CONTRIBUTION_BYTES, \
            "the seam's own per-section cap is the one that actually refuses"
    assert fat["section_trimmed"] >= 1, "the fat payload was supposed to shed something"
    assert fat["trimmed"], "it must say WHAT it shed"


def test_the_sixth_contribution_is_registered_and_the_file_survives_six():
    """THE SEAM GREW TO SIX, AND THE FILE SURVIVES IT - the assertion that says the
    raise was necessary rather than decorative.

    `MAX_CONTRIBUTIONS` was 5 with an allow-list of exactly five names, and the seam
    REFUSES an unregistered section, so without the raise `disk_gpu_write` would have
    been silently MISSING from the rank file - the worst possible place for a diagnostic
    about live KV writes to be absent, because absence reads as "nothing to report". Six
    sections can ask for 24 KiB of a 32 KiB ceiling, so this writes a document carrying
    all SIX and checks the reader's required fields survive and the file stays under the
    cap.
    """
    assert len(worker_diag.CONTRIBUTION_SECTIONS) == worker_diag.MAX_CONTRIBUTIONS == 6
    assert "disk_gpu_write" in worker_diag.CONTRIBUTION_SECTIONS
    registered = disk_gpu_write.register_contribution()
    assert registered["registered"] is True, registered
    assert worker_diag.contributions()["registered"] == ["disk_gpu_write"]
    for name in ("disk", "disk_restore", "disk_shadow", "disk_gpu_shadow",
                 "disk_restore_verify"):
        made = worker_diag.register_contribution(
            name, lambda: {"note": "a section standing in for its owner"},
            owner=f"test/{name}")
        assert made["registered"] is True, made
    document = worker_diag.build_document(None, "PP0_TP0", "env")
    written = document["writer"]["contributions"]["written"]
    for name in worker_diag.CONTRIBUTION_SECTIONS:
        assert name in written, f"{name} is absent from a full house of six"
    for key in ("version", "epoch", "ts", "role", "rank", "arena", "counters"):
        assert key in document, f"the proxy host's reader requires {key!r}"
    assert document["rank"] == "PP0_TP0"
    assert "disk_gpu_write" in document
    assert len(json.dumps(document).encode()) <= worker_diag.MAX_FILE_BYTES


def test_the_verb_reads_and_latches_and_refuses_a_non_boolean():
    """`POST /plugins/kvtier/disk {"op":"gpu-write"}`: both halves, and the bad value.

    A key that is PRESENT is an instruction, a key that is absent is a question, and a
    value that is not a real JSON boolean is refused WITHOUT touching the latch - so an
    operator who types `"enabled":"yes"` gets a refusal he can read instead of a switch
    he believes he flipped. The answer carries `scope`, and for this op it matters more
    than for any other: the write happens at the RESTORE APPLY POINT, which is code only
    a WORKER runs, so a POST that lands on the EngineCore latches a process that can
    never write a window.
    """
    read = disk_store.disk_for_engine(args={"op": "gpu-write"})
    assert read["ok"] is True and read["gpu_write"]["enabled"] is False
    assert read["gpu_write"]["env_var"] == config.ENV_DISK_GPU_DIRECT_WRITE
    assert config.ENV_DISK_GPU_DIRECT_WRITE in read["gpu_write"]["scope"]
    assert "WORKER" in read["gpu_write"]["scope"]
    armed_answer = disk_store.disk_for_engine(args={"op": "gpu-write", "enabled": True})
    assert armed_answer["gpu_write"]["enabled"] is True
    assert config.ENV_DISK_GPU_DIRECT_WRITE in armed_answer["note"]
    assert disk_gpu_write.effective_enabled() is True
    bad = disk_store.disk_for_engine(args={"op": "gpu-write", "enabled": "yes"})
    assert bad["ok"] is False and "JSON boolean" in bad["reason"]
    assert bad["gpu_write"]["enabled"] is True        # refused, not guessed
    disk_store.disk_for_engine(args={"op": "gpu-write", "enabled": False})
    assert disk_gpu_write.effective_enabled() is False


def test_the_op_is_mutating_listed_post_only_and_in_the_endpoint_enum():
    """It writes no file, but it changes what the process DOES to live KV: POST-only.

    `op` is a query parameter on this route, so a browser's `<img>` preload could arm a
    device-WRITING path from a GET if this listing slipped - a stronger case than the
    device-READING ops. The endpoint's own enum is asserted against
    `disk_store.DISK_OPS` because the two lists drifting is how a verb becomes
    unreachable.
    """
    assert "gpu-write" in disk_store.DISK_OPS
    assert "gpu-write" in disk_store.MUTATING_DISK_OPS
    refused = disk_store.disk_for_engine(args={"op": "gpu-write", "safe_method": True})
    assert refused["ok"] is False and "POST" in refused["reason"]
    values = endpoint.ARG_SPECS[config.ROUTE_DISK]["op"]["values"]
    assert set(values) == set(disk_store.DISK_OPS)


def test_the_status_the_verb_the_rank_file_and_the_snapshot_report_one_number():
    """One builder, four doors: a number that differs by door is unusable.

    `status()` builds from `section()`, the contribution calls `worker_report()` which
    IS `section()`, the verb reads `status()`, and `/snapshot` publishes `section()` too
    - so this asserts the identity rather than trusting the comment that claims it.
    """
    with disk_gpu_write._lock:
        disk_gpu_write._state["counters"]["writes"] = 9
    verb = disk_store.disk_for_engine(args={"op": "gpu-write"})["gpu_write"]
    contribution = disk_gpu_write.worker_report()
    published = snapshot.dev_state_sections()
    assert verb["writes"] == 9 and contribution["writes"] == 9
    assert published["disk_gpu_write"]["writes"] == 9
    assert published["disk_gpu_write_enabled"] is False
    for key in disk_gpu_write._COUNTER_KEYS:
        assert contribution[key] == verb[key], key


# --- 9: the hook site, driven through the real restore path -------------------
@pytest.fixture
def restore_stage(monkeypatch, tmp_path):
    """The gate open, a rank identity pinned, the key machinery present.

    The same switches `test_disk_restore_verify.restore_stage` raises - it is a fixture
    in that file and pytest does not share fixtures across modules, so it is rebuilt
    here rather than forked into a helper that would make two gate definitions. The
    point of having it at all is these last two tests: they drive the REAL `worker_bind`
    and `worker_before_load`, so the hook SITE, the field names the writer reads and the
    ORDER against the fork's copy are pinned by the product rather than by a fake batch.
    """
    root = tmp_path / "spill"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
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
                                             "reads_applied": 0, "reason_kind": None})
    monkeypatch.setattr(disk_store, "counters", lambda: {
        "geometry_digest": DIGEST, "fingerprint": {"total_bytes_per_block": WIDTH},
        "rank": "PP0_TP0"})
    module = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    module.make_block_hash_with_group_id = pack
    module.resolve_block_hashes = resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, module)
    disk_restore.set_disk_authority(True)
    return root


def spill_file(root, seed, group) -> str:
    """One window-wide file at the path the WORKER will re-derive for this key."""
    made = disk_layout.build_slot_path(str(root), key_hex(seed, group), DIGEST)
    assert made["ok"], made["reason"]
    os.makedirs(os.path.dirname(made["path"]), exist_ok=True)
    payload = bytes((seed * 11 + group + offset) % 251 for offset in range(WIDTH))
    with open(made["path"], "wb") as handle:
        handle.write(payload[:WIDTH].ljust(WIDTH, b"\x00"))
    appended = disk_index.journal_append(disk_index.SPILL_LINE,
                                         key_hex(seed, group), WIDTH)
    assert appended["ok"], appended["reason"]
    disk_index.refresh(force=True)
    return made["path"]


def test_the_real_apply_point_writes_the_applied_entry_and_is_ordered_first(
        restore_stage, monkeypatch, worker, tmp_path):
    """bind -> settle -> the wrapper's call -> the fork's copy, with the writer armed.

    Everything the fake-batch tests could not pin by themselves, pinned by the product:
    the hook site sees EXACTLY the entry whose bytes `_apply_buffer` memmoved into the
    CPU slot (so the CPU slot this module copies FROM holds the FILE's bytes, not bytes
    a test chose), the field names are the real `_worker_bind` names, and our three
    submissions are enqueued BEFORE `start_load_kv` runs the fork's body. The verdict
    then says the GPU window holds the file's leading W_g - with the file as the oracle
    and a slot the test never filled.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    disk_gpu_write._state["env_resolved"] = False
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 1)
    path = spill_file(restore_stage, 1, 0)
    meta = Meta(load_event=11, cpu=[0], gpu=[0], paths=[path], reqs={11: ["req-1"]})
    applied_before = int(disk_restore._state["reads_applied"])
    disk_restore.worker_bind(worker, meta)
    worker.bind_connector_metadata(meta)
    assert fill_pool(worker) == 2
    disk_restore.worker_before_load(worker)
    with open(path, "rb") as handle:
        raw = handle.read(WIDTH)
    # THE DELTA, not the absolute: `reads_applied` is process state a previous test has
    # already moved, and a test asserting an absolute number asserts the file's order.
    assert disk_restore._state["reads_applied"] == applied_before + 1
    assert counters()["writes"] == 1, disk_gpu_write.section()["reasons"]
    assert worker.arena.block_bytes(0) == raw, \
        "the slot the copy read from is not the file's bytes"
    assert worker.journal.kinds == ["enter", "d2h", "h2d", "d2h", "exit"], \
        worker.journal.kinds
    assert counters()["readback_enqueued_before_original"] == 1
    held = disk_gpu_write.view()["in_flight"]
    assert len(held) == 1
    record = list(held.values())[0]
    assert record["path"] == path and record["group"] == 0
    assert record["gpu_block"] == 0 and record["cpu_block"] == 0
    assert record["bytes"] == WIDTH and record["event"] == 11
    worker.start_load_kv()                     # THE FORK, after us, on the same stream
    assert worker.journal.kinds[-1] == "fork-copy"
    worker.complete(11)
    assert one_pass(worker)["results"][0]["equal"] is True
    assert worker.gpu.block_bytes(0)[:GROUP_WIDTHS[0]] == raw[:GROUP_WIDTHS[0]]


def test_the_real_apply_point_with_the_writer_off_writes_nothing(restore_stage,
                                                                monkeypatch, worker,
                                                                tmp_path):
    """The same restore with the flag unset: ONE APPLIED READ, ZERO writer state.

    The mirror of the test above on the same path, so "default off" is proved against
    the real apply point rather than against a fake batch. The restore REALLY happened
    (one more applied read), so the zeros below mean "the writer was not there" and not
    "there was nothing to write" - and the seams RAISE, so a single copy, allocation or
    thread start would fail this test with the seam's own name in the traceback.
    """
    def boom(*args, **kwargs):
        raise AssertionError("the disabled writer touched a seam or started a thread")

    monkeypatch.setattr(disk_gpu_write, "_copy_host_to_device", boom)
    monkeypatch.setattr(disk_gpu_write, "_copy_device_to_host", boom)
    monkeypatch.setattr(disk_gpu_write, "_allocate_row", boom)
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", boom)
    path = spill_file(restore_stage, 1, 0)
    meta = Meta(load_event=11, cpu=[0], gpu=[0], paths=[path], reqs={11: ["req-1"]})
    applied_before = int(disk_restore._state["reads_applied"])
    disk_restore.worker_bind(worker, meta)
    worker.bind_connector_metadata(meta)
    assert disk_restore.worker_before_load(worker)["ok"] is True
    assert disk_restore._state["reads_applied"] == applied_before + 1
    assert counters() == {key: 0 for key in disk_gpu_write._COUNTER_KEYS}
    assert worker.journal.entries == []
    assert disk_gpu_write.view()["in_flight"] == {}
    assert disk_gpu_write._state["pool"] == {}


# --- 10: THE COLD START, i.e. the writer starting ITSELF ----------------------
@pytest.fixture
def armed_cold(monkeypatch):
    """Armed by the ENVIRONMENT and by nothing else: no patched `_ensure_thread`.

    THE COLD-BOOT FIXTURE, and it exists because of the bug it regression-tests. Every
    other test here reaches a write through the `armed` fixture, which replaces
    `_ensure_thread` with `lambda: True` and then calls `fill_pool` by hand - it hands
    over BOTH pieces production never provides. A suite built that way is green about a
    module that cannot write a single byte in the field, which is worse than no suite:
    it turns a total failure into a passing build. This fixture arms the module and
    touches nothing else: the thread, if one is ever started, is the module's own real
    thread, and the pinned rows, if any are ever allocated, are allocated by the
    module's own refill.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    disk_gpu_write._state["env_resolved"] = False
    return True


def test_a_cold_writer_starts_its_own_thread_and_then_actually_writes(
        armed_cold, worker, tmp_path, monkeypatch):
    """THE FIRST WRITE MUST BE REACHABLE FROM THE PRODUCTION ENTRY POINT ALONE.

    THE BUG THIS IS THE REGRESSION TEST FOR (0.2.44): rows are allocated only inside a
    refill, the refill runs only on this module's thread, and the thread was started
    only AFTER `_take_rows` had already succeeded - so a cold pool refused the write,
    the refusal returned before the thread start, and nothing could ever start the
    thread.
    No rows, no thread, no rows: the first write was unreachable, therefore EVERY write
    was. In production that read `applies_seen=64 batches_seen=1 writes=0 checked=0
    passes=0 pool_refills=0 buffers_allocated=0 no_pinned_buffer=1 errors=0`, with
    `thread_started=0` the field that actually proves the thread never ran.

    Nothing is pre-armed here: the sample rate and the interval floor are the PRODUCT
    values (64 and 0.5 s), so this test drives 65+ applies exactly as a worker does and
    asserts the four counters that mean the writer is alive, plus the one that means the
    allocation stayed OFF the step thread.
    """
    alloc_threads: list = []
    real_alloc = disk_gpu_write._allocate_row

    def observe_alloc(region, elems):
        """THE REAL ALLOCATION, with the thread it ran on recorded.

        Wrapping the seam rather than replacing it keeps the production call chain
        (`_refill_pools` -> `_allocate_row` -> the fake region's `new_empty`) and makes
        "no allocation on the step thread" an assertion about a list of thread names
        instead of about a comment.
        """
        alloc_threads.append(threading.current_thread().name)
        return real_alloc(region, elems)

    monkeypatch.setattr(disk_gpu_write, "_allocate_row", observe_alloc)
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    seed_window(worker, path)

    # THE FIRST ARMED APPLY, cold: the pool has never been filled, so this refuses - and
    # the refusal must at least START the thread (the fix's safety net) rather than read
    # like a hardware shortage.
    first = disk_gpu_write.observe_apply(worker, armed_batch(path=path), {})
    assert first["written"] == 0
    assert first["reason"] == "no_pinned_buffer"
    assert worker.gpu.allocated == 0, \
        "the apply point allocated pinned memory to work around an empty pool"
    assert counters()["thread_started"] == 1, \
        "the cold refusal did not start the module's own thread: the first write is " \
        "unreachable, which is the 0.2.43 bug"

    deadline = time.monotonic() + 6.0
    seen = {}
    while time.monotonic() < deadline:
        disk_gpu_write.observe_apply(worker, armed_batch(path=path), {})
        seen = counters()
        if (seen["writes"] and seen["pool_refills"] and seen["passes"]
                and seen["readback_enqueued_before_original"]):
            break
        time.sleep(0.005)

    assert seen["writes"] >= 1, \
        f"the writer never wrote from the production entry point alone: {seen}"
    assert seen["pool_refills"] >= 1, "the module's thread never refilled its pool"
    assert seen["passes"] >= 1, "the module's thread never ran a pass over its records"
    assert seen["readback_enqueued_before_original"] >= 1, \
        "a write happened without its ordering witness"
    assert seen["buffers_allocated"] >= 2
    # THE STEP THREAD STILL NEVER ALLOCATES: the fix must not "solve" the cold pool by
    # page-locking memory inside `start_load_kv`, which is a `cudaHostAlloc` on the step
    # thread and exactly the stall this module exists not to cause.
    assert alloc_threads, "the pool filled without a single allocation"
    assert "MainThread" not in alloc_threads, \
        f"a pinned row was allocated on the step thread: {set(alloc_threads)}"
    assert set(alloc_threads) == {"kvtier-disk-gpu-write"}, set(alloc_threads)


def test_registering_the_contribution_while_armed_starts_the_writer_and_nothing_else(
        worker, monkeypatch):
    """FIX (b): the thread exists BEFORE the module ever asks for a row.

    The cold start had two ends and this one is not the apply point: an armed worker
    should have its writer thread - and therefore its first chance at a warm pool - from
    the moment the plugin is installed, not from the moment a restore is refused. What
    this test ALSO pins is the half that makes starting at install legal at all: a
    thread born before any worker exists allocates NOTHING, reads no geometry, refills
    nothing
    and counts nothing, because the refill walks `_resolve_worker()` and only the apply
    point ever registers a worker.
    """
    # OFF FIRST, because "arming starts a thread" only means something against "off
    # starts nothing": a thread in a process that was told to do nothing is a thread the
    # operator never asked for.
    assert disk_gpu_write.register_contribution()["registered"] is True
    assert counters()["thread_started"] == 0
    assert disk_gpu_write.section()["thread_alive"] is False
    assert worker.gpu.allocated == 0

    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    disk_gpu_write._state["env_resolved"] = False
    assert disk_gpu_write.register_contribution()["registered"] is True
    assert counters()["thread_started"] == 1, \
        "an armed registration started no writer: that is the 0.2.43 cold start"
    assert disk_gpu_write.section()["thread_alive"] is True
    # ... and the thread that just started has done NOTHING heavy: no worker yet, so no
    # row, no refill, no CUDA, no counter beyond `thread_started` itself.
    assert counters()["buffers_allocated"] == 0
    assert counters()["pool_refills"] == 0
    assert worker.gpu.allocated == 0
    assert disk_gpu_write.pool_view()["pinned_bytes"] == 0
    assert disk_gpu_write.section()["liveness"]["state"] == (
        disk_gpu_write.LIVENESS_NO_APPLY)


def test_a_cold_pool_and_a_drained_pool_are_two_different_readings(armed, worker,
                                                                  tmp_path):
    """THE TWO FACTS THAT USED TO SHARE THE NAME `no_pinned_buffer`.

    "The witness thread has not filled its pool yet" and "the pool has rows and none is
    free" look identical in a rank file and mean opposite things to a reader: the first
    resolves itself on the thread's next pass, the second means rows are held by records
    whose load event has not passed, or have leaked, which never resolves. They are now
    counted apart (`pool_never_filled` rides along with the cold case and NEVER with the
    drained one), named apart in the refusal sentence, and said apart in the returned
    `cause`. The third member of the family - the in-flight cap - was already separable
    through `in_flight_dropped`.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 8)
    seed_window(worker, path)
    worker._load_hwm = -1            # rows never come back: the drained case below

    # THE COLD ONE: no row has ever been allocated in this process.
    disk_gpu_write._state["last_write_mono"] = None
    cold = disk_gpu_write.observe_apply(worker, armed_batch(path=path), {})
    assert cold["reason"] == "no_pinned_buffer"
    assert cold["cause"] == "pool_never_filled"
    assert counters()["no_pinned_buffer"] == 1
    assert counters()["pool_never_filled"] == 1
    assert any(row.startswith("no_pinned_buffer_cold_pool:")
               for row in disk_gpu_write.section()["reasons"]), \
        disk_gpu_write.section()["reasons"]

    # THE DRAINED ONE: the pool filled, and a record holds every row it has.
    assert fill_pool(worker) == 2
    assert one_write(worker, path=path)["written"] == 1
    drained = one_write(worker, path=path)
    assert drained["reason"] == "no_pinned_buffer"
    assert drained["cause"] == "pool_drained"
    assert counters()["no_pinned_buffer"] == 2
    assert counters()["pool_never_filled"] == 1, \
        "a drained pool counted as a cold one: they are different operational facts"
    assert not any(row.startswith("no_pinned_buffer_cold_pool:")
                   and "all held by records" in row
                   for row in disk_gpu_write.section()["reasons"])


def test_the_section_says_out_loud_which_state_the_writer_is_in(worker, tmp_path,
                                                                monkeypatch):
    """`liveness`: the section must answer "is this witness actually alive?".

    0.2.43's rank file said `writes=0` next to `applies_seen=64` and left the reading to
    whoever found it. Four states, one field, and THE SAME KEY SET IN ALL FOUR - a
    reader who diffs two rank files must be able to tell "the writer died" from "this
    boot has
    no GPU" by reading a value, not by noticing that a key is missing.
    """
    rows = [disk_gpu_write.section()["liveness"]]
    assert rows[0]["state"] == disk_gpu_write.LIVENESS_OFF, rows[0]

    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    disk_gpu_write._state["env_resolved"] = False
    rows.append(disk_gpu_write.section()["liveness"])
    assert rows[1]["state"] == disk_gpu_write.LIVENESS_NO_APPLY, rows[1]

    # ARMED, AN APPLY POINT, AND NOT ONE WRITE: the state production was in.
    disk_gpu_write.observe_apply(worker, armed_batch(path="/tmp/kvslot-never"), {})
    dead = disk_gpu_write.section()["liveness"]
    rows.append(dead)
    assert dead["state"] == disk_gpu_write.LIVENESS_NEVER_WROTE, dead
    assert dead["cause"] == "pool_never_filled", dead
    for fragment in ("NEVER WROTE", "1 applied entr", "thread started",
                     "pool refills", "free rows", "cause: pool_never_filled"):
        assert fragment in dead["verdict"], dead["verdict"]

    # AND THE ALIVE ONE, for the same key set.
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 8)
    seed_window(worker, path)
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 1)
    assert fill_pool(worker) == 2
    assert one_write(worker, path=path)["written"] == 1
    rows.append(disk_gpu_write.section()["liveness"])
    assert rows[-1]["state"] == disk_gpu_write.LIVENESS_WRITING, rows[-1]
    for row in rows:
        assert set(row) == {"state", "verdict", "thread_running", "free_rows",
                            "cause"}, row
