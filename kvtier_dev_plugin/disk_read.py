"""Read spilled KV bytes off the SSD, into a BOUNCE BUFFER, on its own thread.

THE ONE DECISION THAT SHAPES THIS WHOLE MODULE
----------------------------------------------
A restore ends as bytes inside a pinned CPU KV slot, and that slot belongs to the
SCHEDULER's `cpu_block_pool`: the scheduler hands the slot out, touches it for the
duration of the load, and frees it when the worker reports the load finished -
after which another request may allocate it and a store may overwrite it. So the
question "when may a byte be written into that slot?" is a correctness question,
and the answer chosen here is:

    ONLY ON THE ENGINE THREAD, at the moment the load is submitted.

The obvious alternative - have the reader thread `readinto` the destination slot
span directly, which is one copy cheaper - is what makes the failure paths
unsound. A read that is already in flight cannot be cancelled: if the load is
cancelled or expires while the kernel is halfway through filling that memory,
the scheduler is entitled to free the slot (the load was reported failed), hand it
to a new store, and then OUR late bytes land inside someone else's cached block,
which is then published under the NEW block's hash. That is silent KV corruption,
not a slow request, and no counter would ever show it. With a bounce buffer, a
cancelled read is a buffer nobody ever applies: the destination was never
touched, so "we stopped this load" and "the slot is clean" are the same fact.

The price is one extra memcpy per restored slot (the file's bytes into the slot),
paid on the engine thread at submit time by the caller, and one bounded pool of
buffers (`MAX_BUFFERS_BYTES`). Against a restore that saves a multi-hundred-ms
recompute, one memcpy of 11.4 MB is ~1 ms; against silent corruption, it is free.

WHAT THIS MODULE DOES NOT DECIDE
--------------------------------
Not who asked, not which GPU or CPU slots are involved, not when the bytes get
applied, not what the scheduler is told about a failure. Those are
`disk_restore`'s. This module answers exactly two questions - "can you read this
file for me" and "is that batch finished yet" - and both answers are bounded,
classified dicts that never raise, because `request_read` is called from the
worker's bind and `read_ready` from its per-step `get_finished`.

Stdlib only. One thread, one bounded queue, one bounded buffer pool, all created
lazily on the first accepted read (`disk_spill`'s discipline: nothing at import).
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time

from . import disk_layout, snapshot

# --- the caps, and the unit of every one of them ----------------------------

MAX_QUEUE_DEPTH = 64                  # JOBS queued or in flight
MAX_BATCHES = 32                      # OPEN batches tracked at once
MAX_JOBS_PER_BATCH = 4_096            # JOBS one batch may hold (sanity ceiling)
MAX_JOBS_REPORT = 16                  # FAILURES one status report may name
MAX_BYTES_PER_READ = 64 * 1024 * 1024  # BYTES in ONE file (one slot's bytes)
MAX_BUFFERS_BYTES = 96 * 1024 * 1024  # BYTES held by the bounce-buffer pool
MAX_STRING_CHARS = 160                # CHARS of a path/reason kept in state
READ_POLL_SECONDS = 0.5               # SECONDS the thread waits on an empty queue

# The default patience for one batch: comfortably beyond the worst-case bounded
# read (a real batch is a few dozen slots of 11.4 MB, well under a second on a
# healthy SSD) and far below the point where a parked request looks to an
# operator like an engine hang. Expiring a batch costs a recompute; it never
# releases a destination that is still being written (see `cancel_batch`).
LOAD_DEADLINE_SECONDS = 30.0

# Why a read did not happen, or why a file's bytes are not usable. A closed
# vocabulary, one counter each: the witness and the boot line print these names.
REFUSALS = ("not_configured", "disarmed", "bad_path", "bad_key", "bad_size",
            "bad_batch", "bad_job", "batch_cap", "queue_full", "buffer_cap",
            "thread_dead", "overlaps", "bad_call")
READ_FAILURES = ("file_missing", "size_mismatch", "short_read", "read_error",
                 "not_ours")

_state: dict = {
    "requests": 0,
    "accepted": 0,
    "batches_opened": 0,
    "batches_closed": 0,
    "batches_cancelled": 0,
    "reads_completed": 0,
    "bytes_read": 0,
    "read_seconds": 0.0,
    "max_read_seconds": 0.0,
    "queue_depth": 0,
    "queue_depth_peak": 0,
    "buffers_held_bytes": 0,
    "buffers_peak_bytes": 0,
    "refused": {kind: 0 for kind in REFUSALS},
    "failed": {kind: 0 for kind in READ_FAILURES},
    "errors": 0,
    "thread_started": 0,
    "thread_alive": False,
    "last_reason": None,
    "last_error": None,
    "last_path": None,
}

# Open batches: batch id -> {"jobs": {job_id: job}, "opened_mono": float,
# "deadline_mono": float, "expected": int}. Guarded by `_state_lock`: the engine
# thread writes it at bind and reads it in get_finished, and the reader thread
# marks its own jobs done. Every critical section is a dict update.
_batches: dict = {}
_buffers: list = []          # free (ctypes array, size) pairs, pooled
_state_lock = threading.Lock()
_queue = None
_reader_thread = None
_thread_lock = threading.Lock()

# Job states. `queued` has never touched a buffer's contents, `running` is inside
# the kernel read, `done` holds good bytes, `failed` holds none. A cancelled
# queued job can be released immediately; a cancelled RUNNING job keeps its
# buffer until the thread finishes with it, and its bytes are then dropped
# (never applied) - that asymmetry is the whole reason for the bounce buffer.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


def reset_state_for_tests() -> None:
    """Back to a fresh boot: no batches, no buffers, no counters.

    Does NOT stop the thread (a test that wants that calls `stop_reader`): the
    thread reads the queues it was handed, so an empty module state is all it
    takes for the next test to see a clean ledger.
    """
    global _batches, _buffers
    with _state_lock:
        _batches = {}
        _buffers = []
        _state.update({
            "requests": 0, "accepted": 0, "batches_opened": 0,
            "batches_closed": 0, "batches_cancelled": 0, "reads_completed": 0,
            "bytes_read": 0, "read_seconds": 0.0, "max_read_seconds": 0.0,
            "queue_depth": 0, "queue_depth_peak": 0, "buffers_held_bytes": 0,
            "buffers_peak_bytes": 0,
            "refused": {kind: 0 for kind in REFUSALS},
            "failed": {kind: 0 for kind in READ_FAILURES},
            "errors": 0, "last_reason": None, "last_error": None,
            "last_path": None,
        })


def counters() -> dict:
    """A copy for a payload. Never the batch table, only its size."""
    with _state_lock:
        out = dict(_state)
        out["batches_open"] = len(_batches)
        out["queued"] = int(_state["queue_depth"])
    return out


def _note(reason) -> None:
    _state["last_reason"] = str(reason)[:MAX_STRING_CHARS]


def _fail(reason, **extra) -> dict:
    out = {"ok": False, "reason": str(reason)[:MAX_STRING_CHARS]}
    out.update(extra)
    return out


def _ok(**fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    return out


def _refused(kind: str, reason: str, **extra) -> dict:
    """Count a refusal by its closed-vocabulary name and return it."""
    name = kind if kind in _state["refused"] else "bad_call"
    _state["refused"][name] += 1
    _note(reason)
    out = {"ok": False, "reason_kind": name,
           "reason": str(reason)[:MAX_STRING_CHARS]}
    out.update(extra)
    return out


def _failed(kind: str, reason: str, **extra) -> dict:
    name = kind if kind in _state["failed"] else "read_error"
    _state["failed"][name] += 1
    _note(reason)
    out = {"ok": False, "reason_kind": name,
           "reason": str(reason)[:MAX_STRING_CHARS]}
    out.update(extra)
    return out


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- the gate -----------------------------------------------------------------

def gate(*, armed: bool | None = None,
         configured: bool | None = None) -> dict:
    """May this process read a spilled file at all? One place, every caller.

    `armed`/`configured` are injectable so the CALLER's view of the disk tier can
    be checked without this module importing `disk_spill` (which owns both facts
    and would make this a second reader of the arm file). `disk_restore` passes
    the two verdicts it already asked for; a bare call reads the environment only,
    which is the cheaper and more conservative answer: no env root means no read,
    in any process, whatever any file says.
    """
    from . import config, disk_spill  # noqa: PLC0415 - the owner of both facts
    if configured is None:
        configured = disk_spill.disk_root() is not None
    if not configured:
        return _refused("not_configured",
                        f"{config.ENV_DISK_ROOT} is not set: there is no store "
                        "directory a spilled file could be read from")
    if armed is None:
        armed = disk_spill.is_armed()
    if not armed:
        return _refused("disarmed", "the disk tier is disarmed: a restore is "
                                    "spill-side work and obeys the same arm")
    return _ok()


# --- the buffer pool ----------------------------------------------------------

def _take_buffer(size: int):
    """A writable `size`-byte buffer from the pool, or None if the pool is full.

    Buffers are `ctypes` char arrays rather than `bytearray`s so both halves of
    the job work on the same object without a copy: `readinto(memoryview(arr))`
    fills it, and `ctypes.memmove(dst, arr, n)` empties it into a slot. The pool
    is bounded BY BYTES, because the unit that matters on a box already carrying
    a 120 GB pinned KV tier is bytes, not objects.
    """
    with _state_lock:
        for index, (buffer, capacity) in enumerate(_buffers):
            if capacity == size:
                _buffers.pop(index)
                return buffer
        held = int(_state["buffers_held_bytes"])
        if held + size > MAX_BUFFERS_BYTES or size > MAX_BYTES_PER_READ:
            return None
        _state["buffers_held_bytes"] = held + size
        _state["buffers_peak_bytes"] = max(int(_state["buffers_peak_bytes"]),
                                           held + size)
    try:
        return (ctypes.c_char * size)()
    except Exception as exc:  # noqa: BLE001 - no memory is a refusal, not a fault
        with _state_lock:
            _state["buffers_held_bytes"] = max(0, int(_state["buffers_held_bytes"])
                                               - size)
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return None


def _give_buffer(buffer, size: int) -> None:
    """Return a buffer to the pool. The pool shrinks rather than growing forever."""
    with _state_lock:
        if len(_buffers) >= MAX_BATCHES or size > MAX_BYTES_PER_READ:
            _state["buffers_held_bytes"] = max(0, int(_state["buffers_held_bytes"])
                                               - size)
            return
        _buffers.append((buffer, int(size)))


def buffers_held_bytes() -> int:
    with _state_lock:
        return int(_state["buffers_held_bytes"])


# --- the queue and its thread -------------------------------------------------

_STOP = object()


def _get_queue():
    global _queue
    with _thread_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        return _queue


def _ensure_thread() -> bool:
    """Start the module's single reader thread, once, lazily. Never at import."""
    global _reader_thread
    with _thread_lock:
        alive = _reader_thread is not None and _reader_thread.is_alive()
        _state["thread_alive"] = bool(alive)
        if alive:
            return True
        try:
            thread = threading.Thread(target=_reader_loop,
                                      name="kvtier-disk-read", daemon=True)
            thread.start()
        except Exception as exc:  # noqa: BLE001 - no thread is a refusal
            snapshot._error(exc, "disk_read thread start")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            _state["thread_alive"] = False
            return False
        _reader_thread = thread
        _state["thread_started"] += 1
        _state["thread_alive"] = True
        return True


def thread_alive() -> bool:
    thread = _reader_thread
    return bool(thread is not None and thread.is_alive())


def _reader_loop() -> None:
    """The drain loop. ONE thread, and the only place a file is read.

    A job carries a path, an expected size and a buffer it already owns; the loop
    does not need the engine's state to do its work, which is what makes it safe
    to leave running while the caller moves on.
    """
    local = _get_queue()
    while True:
        try:
            job = local.get(timeout=READ_POLL_SECONDS)
        except queue.Empty:
            continue
        if job is _STOP:
            local.task_done()
            break
        try:
            _handle_read(job)
        except Exception as exc:  # noqa: BLE001 - a job never kills the thread
            _state["errors"] += 1
            snapshot._error(exc, "disk_read job")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            _settle(job, _failed("read_error",
                                 f"the read raised {type(exc).__name__}"))
        finally:
            _state["queue_depth"] = max(0, int(_state["queue_depth"]) - 1)
            local.task_done()


def _handle_read(job) -> None:
    path = job.get("path")
    size = int(job["expect_bytes"])
    buffer = job.get("buffer")
    if buffer is None:                 # cancelled before it ever started
        return
    with _state_lock:
        held = _batches.get(job["batch"])
        entry = held["jobs"].get(job["job_id"]) if held else None
        if entry is None or entry["state"] == CANCELLED:
            job["state"] = CANCELLED
        else:
            entry["state"] = RUNNING
    if job.get("state") == CANCELLED:
        _give_buffer(buffer, size)
        return
    started = time.monotonic()
    outcome = _read_file(path, buffer, size)
    elapsed = round(time.monotonic() - started, 6)
    _state["read_seconds"] = round(float(_state["read_seconds"]) + elapsed, 6)
    _state["max_read_seconds"] = round(max(float(_state["max_read_seconds"]),
                                           elapsed), 6)
    if outcome["ok"]:
        _state["reads_completed"] += 1
        _state["bytes_read"] = int(_state["bytes_read"]) + size
        _state["last_path"] = str(path)[:MAX_STRING_CHARS]
    _settle(job, outcome)


def _read_file(path, buffer, size: int) -> dict:
    """Read exactly `size` bytes of `path` into `buffer`. Never raises.

    The size must match EXACTLY, before a single byte is copied: a file shorter
    than the fingerprint's slot width was written by a different geometry or died
    mid-write, and a file longer than it is not the file this name describes.
    Either way the answer is "these bytes are not this block", which the caller
    turns into a reported load failure - and a partial apply is the one thing
    that would turn a missing file into a wrong answer.
    """
    parsed = disk_layout.parse_path(path)
    if not parsed["ok"]:
        return _failed("read_error", f"the path is not a spill file this plugin "
                                     f"could have written: {parsed['reason']}")
    if parsed["partial"]:
        return _failed("not_ours", "the name is an in-flight .part file: its "
                                   "bytes are not a committed block")
    try:
        actual = os.path.getsize(path)
    except FileNotFoundError:
        return _failed("file_missing", f"no file is named "
                                       f"{os.path.basename(str(path))[:60]}: the "
                                       "index said it existed and the filesystem "
                                       "says otherwise (it was evicted, or the "
                                       "journal lied)")
    except OSError as exc:
        return _failed("read_error", f"the file could not be stat'ed: "
                                     f"{type(exc).__name__}")
    if actual != size:
        return _failed("size_mismatch", f"the file is {actual} bytes but this "
                                        f"boot's slot is exactly {size} bytes: "
                                        "not these bytes, never applied")
    try:
        with open(path, "rb") as handle:
            view = memoryview(buffer)
            got = handle.readinto(view)
    except FileNotFoundError:
        return _failed("file_missing", "the file vanished between the size check "
                                       "and the read")
    except OSError as exc:
        return _failed("read_error", f"the read failed: {type(exc).__name__}")
    if got != size:
        return _failed("short_read", f"the read returned {got} of {size} bytes")
    return _ok(bytes=size, buffer=buffer)


def _settle(job, outcome: dict) -> None:
    """Publish a job's outcome into its batch. Locked, and last word wins."""
    with _state_lock:
        batch = _batches.get(job["batch"])
        entry = batch["jobs"].get(job["job_id"]) if batch else None
        if entry is None:
            # The batch was finished or cancelled out from under the read. The
            # buffer goes back to the pool and the bytes are dropped: this is the
            # case the bounce buffer exists for.
            buffer = job.get("buffer")
            if buffer is not None:
                _give_buffer(buffer, int(job["expect_bytes"]))
            return
        if entry["state"] == CANCELLED:
            buffer = job.get("buffer")
            if buffer is not None:
                _give_buffer(buffer, int(job["expect_bytes"]))
            return
        if outcome.get("ok"):
            entry["state"] = DONE
            entry["bytes"] = int(outcome["bytes"])
        else:
            entry["state"] = FAILED
            entry["reason_kind"] = str(outcome.get("reason_kind", "read_error"))
            entry["reason"] = str(outcome.get("reason"))[:MAX_STRING_CHARS]
            buffer = job.get("buffer")
            if buffer is not None:
                _give_buffer(buffer, int(job["expect_bytes"]))
            job["buffer"] = None


# --- the batch API the restore layer calls -----------------------------------

def begin_batch(batch_id, job_count: int,
                deadline_seconds: float | None = None) -> dict:
    """Open a batch of `job_count` reads that must ALL settle before use.

    A batch is the unit the engine cares about: one `load_event`, one set of
    slots, one `finished_recving`. `job_count` is declared up front so a batch
    that never receives all its jobs can be reported INCOMPLETE rather than wait
    for jobs nobody queued - a missing job is a bug in the caller, and an
    incomplete batch that silently "succeeded" is how stale bytes get DMA'd.
    """
    if _is_plain_int(batch_id) is False and not isinstance(batch_id, str):
        return _refused("bad_batch", f"a batch id must be an int or a str, got "
                                     f"{type(batch_id).__name__}")
    if not _is_plain_int(job_count) or job_count < 0:
        return _refused("bad_batch", f"job_count must be a non-negative int, got "
                                     f"{job_count!r}")
    if job_count > MAX_JOBS_PER_BATCH:
        return _refused("batch_cap", f"the batch wants {job_count} jobs; the cap "
                                     f"is {MAX_JOBS_PER_BATCH}")
    now = time.monotonic()
    with _state_lock:
        if batch_id in _batches:
            return _refused("overlaps", f"batch {batch_id!r} is already open: a "
                                        "second load_event cannot reuse one id")
        if len(_batches) >= MAX_BATCHES:
            return _refused("batch_cap", f"{len(_batches)} batches are already "
                                         f"open; the cap is {MAX_BATCHES}, so "
                                         "this load is refused rather than "
                                         "tracked past its bound")
        _batches[batch_id] = {
            "jobs": {}, "expected": int(job_count),
            "opened_mono": round(now, 6),
            "deadline_mono": round(now + float(deadline_seconds
                                               or LOAD_DEADLINE_SECONDS), 6),
        }
        _state["batches_opened"] += 1
    if not _ensure_thread():
        cancel_batch(batch_id, "the reader thread could not be started")
        return _refused("thread_dead", "the reader thread could not be started")
    return _ok(batch=batch_id, expected=int(job_count))


def request_read(batch_id, job_id, path, expect_bytes: int) -> dict:
    """Queue one file read. Cheap, bounded, never blocks, never raises.

    The engine thread pays: a dict lookup, a bound check, one buffer take and one
    `put_nowait`. Everything that can fail on the way to a file - the stat, the
    open, the read - is the thread's problem and comes back as a classified
    failure on the batch's ledger.
    """
    _state["requests"] += 1
    try:
        with _state_lock:
            batch = _batches.get(batch_id)
            if batch is None:
                return _refused("bad_batch", f"batch {batch_id!r} is not open, so "
                                             "this read has nowhere to report to")
            if job_id in batch["jobs"]:
                return _refused("bad_job", f"job {job_id!r} is already in batch "
                                           f"{batch_id!r}: two slots cannot share "
                                           "one identity in one load")
            if len(batch["jobs"]) >= MAX_JOBS_PER_BATCH:
                return _refused("batch_cap", f"batch {batch_id!r} already holds "
                                             f"{MAX_JOBS_PER_BATCH} jobs")
        if not _is_plain_int(job_id) or job_id < 0:
            return _refused("bad_job", f"a job id must be a non-negative int, got "
                                       f"{job_id!r}")
        if not _is_plain_int(expect_bytes) or not 0 < expect_bytes \
                <= MAX_BYTES_PER_READ:
            return _refused("bad_size", f"expect_bytes must be an int in "
                                        f"(0, {MAX_BYTES_PER_READ}], got "
                                        f"{expect_bytes!r} (unit: bytes)")
        parsed = disk_layout.parse_path(path)
        if not parsed["ok"]:
            return _refused("bad_path", f"the path is not a spill file: "
                                        f"{parsed['reason']}")
        if parsed["partial"]:
            return _refused("bad_path", "an in-flight .part name is not a "
                                        "committed file")
        buffer = _take_buffer(int(expect_bytes))
        if buffer is None:
            return _refused("buffer_cap", f"the bounce-buffer pool cannot hold "
                                          f"another {expect_bytes}-byte buffer "
                                          f"(cap {MAX_BUFFERS_BYTES} bytes): this "
                                          "load is refused, not queued onto an "
                                          "unbounded pile")
        job = {"batch": batch_id, "job_id": int(job_id), "path": str(path),
               "expect_bytes": int(expect_bytes), "buffer": buffer,
               "state": QUEUED, "queued_ts": round(time.time(), 6)}
        pending = _get_queue()
        try:
            pending.put_nowait(job)
        except queue.Full:
            _give_buffer(buffer, int(expect_bytes))
            return _refused("queue_full", f"the read queue is at its "
                                          f"{MAX_QUEUE_DEPTH}-job cap: the read "
                                          "was dropped, not queued")
        with _state_lock:
            batch = _batches.get(batch_id)
            if batch is None:
                # Raced with a cancel/finish: the job will settle into nothing and
                # `_settle` returns the buffer. Say so rather than pretend.
                return _refused("bad_batch", f"batch {batch_id!r} closed while "
                                             "this read was being queued")
            batch["jobs"][int(job_id)] = {"state": QUEUED, "bytes": 0,
                                          "reason_kind": None, "reason": None,
                                          "job": job}
            _state["accepted"] += 1
            depth = int(_state["queue_depth"]) + 1
            _state["queue_depth"] = depth
            _state["queue_depth_peak"] = max(int(_state["queue_depth_peak"]),
                                             depth)
        return _ok(batch=batch_id, job=int(job_id), queued=True,
                   bytes=int(expect_bytes))
    except Exception as exc:  # noqa: BLE001 - a read request never breaks a step
        _state["errors"] += 1
        snapshot._error(exc, "disk_read request")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return _refused("bad_batch", f"the read request raised "
                                     f"{type(exc).__name__}")


def batch_status(batch_id) -> dict:
    """Where one batch stands. Reads no files, takes no locks longer than a dict.

    `settled` means every expected job has an answer and none is in flight, which
    is the ONLY state a caller may act on: `pending > 0` means wait, and
    `incomplete` (fewer jobs accepted than declared, none running) is a caller bug
    reported as a failure rather than a success.
    """
    now = time.monotonic()
    with _state_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return _fail(f"batch {batch_id!r} is not tracked", known=False,
                         settled=False, pending=0, done=0, failed=0,
                         running=0, queued=0, incomplete=False,
                         deadline_expired=False, bytes_read=0,
                         elapsed_seconds=0.0,
                         failures=[])
        states = [entry["state"] for entry in batch["jobs"].values()]
        done = states.count(DONE)
        failed = states.count(FAILED)
        running = states.count(RUNNING)
        queued = states.count(QUEUED)
        cancelled = states.count(CANCELLED)
        accepted = len(states)
        expected = int(batch["expected"])
        bytes_read = sum(int(entry["bytes"]) for entry in batch["jobs"].values()
                         if entry["state"] == DONE)
        failures = [{"job": int(job_id),
                     "reason_kind": str(entry["reason_kind"] or "read_error"),
                     "reason": str(entry["reason"] or "")[:MAX_STRING_CHARS]}
                    for job_id, entry in sorted(batch["jobs"].items())
                    if entry["state"] == FAILED][:MAX_JOBS_REPORT]
        failures.sort(key=lambda item: item["job"])
        pending = running + queued
        # Two different "the wait is over" states, deliberately named apart:
        # `settled` (every expected job answered) is the only one a caller may
        # apply bytes from; `incomplete` (fewer jobs than declared, nothing in
        # flight) is a caller bug and must be reported as a LOAD FAILURE, never as
        # a success - a batch that quietly had fewer reads than slots is exactly
        # how a stale slot would reach the DMA.
        settled = pending == 0 and accepted >= expected
        incomplete = pending == 0 and accepted < expected
        return {"known": True, "settled": bool(settled), "pending": pending,
               "done": done, "failed": failed, "running": running,
               "queued": queued, "cancelled": cancelled,
               "incomplete": bool(incomplete),
               "finished_waiting": bool(pending == 0),
               "accepted": accepted, "expected": expected,
               "bytes_read": int(bytes_read),
               "deadline_expired": bool(now >= float(batch["deadline_mono"])),
               "seconds_left": round(max(0.0, float(batch["deadline_mono"]) - now),
                                     3),
               "elapsed_seconds": round(now - float(batch["opened_mono"]), 3),
               "failures": failures}



def batch_buffer(batch_id, job_id):
    """The buffer of one DONE job, or None. The bytes are not copied out.

    The caller memmoves from it into its destination and then calls
    `finish_batch`, which is what returns the buffer to the pool. Holding a
    buffer is what pins it: two callers cannot both be handed one.
    """
    with _state_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return None
        entry = batch["jobs"].get(job_id)
        if entry is None or entry["state"] != DONE:
            return None
        return entry.get("job", {}).get("buffer")


def batch_bytes(batch_id, job_id) -> int:
    with _state_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return 0
        entry = batch["jobs"].get(job_id)
        return 0 if entry is None else int(entry["bytes"])


def finish_batch(batch_id) -> dict:
    """Close a batch and return every buffer it held. Idempotent."""
    with _state_lock:
        batch = _batches.pop(batch_id, None)
        if batch is None:
            return _ok(closed=False, reason="the batch was already closed")
        released = 0
        for entry in batch["jobs"].values():
            buffer = entry.get("job", {}).get("buffer")
            if buffer is None:
                continue
            released += 1
            _give_buffer(buffer, int(entry["job"]["expect_bytes"]))
        _state["batches_closed"] += 1
        return _ok(closed=True, buffers=released, reason=None)


def cancel_batch(batch_id, reason: str) -> dict:
    """Abandon a batch. Queued reads never start; running ones are dropped.

    A RUNNING job's buffer stays held until the thread returns it (it is inside a
    kernel read and cannot be stopped), and its bytes are then thrown away by
    `_settle` because the batch is gone. That is the promise this module exists to
    keep: a cancelled restore never writes a destination.
    """
    with _state_lock:
        batch = _batches.pop(batch_id, None)
        if batch is None:
            return _ok(cancelled=False, running=0, reason="already closed")
        running = 0
        for entry in batch["jobs"].values():
            if entry["state"] in (QUEUED, DONE, FAILED):
                entry["state"] = CANCELLED
                buffer = entry.get("job", {}).get("buffer")
                if buffer is not None:
                    entry["job"]["buffer"] = None
                    _give_buffer(buffer, int(entry["job"]["expect_bytes"]))
            elif entry["state"] == RUNNING:
                running += 1
        _state["batches_cancelled"] += 1
        _note(f"batch {batch_id!r} cancelled: {reason}")
        return _ok(cancelled=True, running=running,
                   reason=str(reason)[:MAX_STRING_CHARS])


def open_batches() -> list:
    """The batch ids currently tracked, for a watchdog sweep. Bounded."""
    with _state_lock:
        return sorted(_batches, key=str)[:MAX_BATCHES]


def drain(timeout_seconds: float = 5.0) -> dict:
    """Wait for the queue to empty. A test and shutdown helper, not an engine path."""
    local = _queue
    if local is None:
        return {"drained": True, "reason": "no queue was ever created", "queued": 0}
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while local.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    return {"drained": local.unfinished_tasks == 0,
            "queued": int(local.unfinished_tasks),
            "reason": (None if local.unfinished_tasks == 0
                       else "the queue was still working at the timeout")}


def stop_reader(timeout_seconds: float = 2.0) -> dict:
    """Ask the thread to exit. Nothing relies on this at shutdown today."""
    global _reader_thread
    local = _queue
    if local is None:
        return {"stopped": True, "reason": "no thread was ever started"}
    try:
        local.put_nowait(_STOP)
    except queue.Full:
        return {"stopped": False, "reason": "the queue is full; try again"}
    thread = _reader_thread
    if thread is not None and thread.is_alive():
        thread.join(max(0.0, float(timeout_seconds)))
    alive = bool(thread is not None and thread.is_alive())
    _state["thread_alive"] = alive
    return {"stopped": not alive,
            "reason": None if not alive
                      else "the thread did not exit inside the timeout"}


# --- the view -----------------------------------------------------------------


def section() -> dict:
    """This module's state for a snapshot or a verb. Reads nothing from disk."""
    with _state_lock:
        snapshot_state = dict(_state)
        snapshot_state["batches_open"] = len(_batches)
    snapshot_state["thread_alive"] = thread_alive()
    snapshot_state["refused"] = dict(snapshot_state["refused"])
    snapshot_state["failed"] = dict(snapshot_state["failed"])
    snapshot_state["caps"] = {
        "max_queue_depth": MAX_QUEUE_DEPTH, "max_batches": MAX_BATCHES,
        "max_jobs_per_batch": MAX_JOBS_PER_BATCH,
        "max_bytes_per_read": MAX_BYTES_PER_READ,
        "max_buffers_bytes": MAX_BUFFERS_BYTES,
        "load_deadline_seconds": LOAD_DEADLINE_SECONDS}
    snapshot_state["units"] = {
        "requests": "read requests", "accepted": "reads queued",
        "reads_completed": "reads finished with good bytes",
        "bytes_read": "bytes", "read_seconds": "seconds",
        "max_read_seconds": "seconds in the slowest single read",
        "queue_depth": "jobs", "queue_depth_peak": "jobs",
        "buffers_held_bytes": "bytes held by bounce buffers",
        "buffers_peak_bytes": "bytes", "batches_open": "batches",
        "batches_opened": "batches", "batches_closed": "batches",
        "batches_cancelled": "batches", "errors": "raised exceptions"}
    snapshot_state["note"] = (
        "reads land in a BOUNCE BUFFER, never in a KV slot: the caller memmoves "
        "them on the engine thread when it submits the DMA, which is what makes "
        "'we cancelled this load' and 'the destination was never touched' the "
        "same fact. A cancelled RUNNING read still costs its buffer until the "
        "kernel returns, and its bytes are dropped.")
    return snapshot_state
