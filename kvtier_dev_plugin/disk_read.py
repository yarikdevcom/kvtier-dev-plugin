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

N READER THREADS (0.2.24), AND THE ONE THING THAT BOUNDS THEM
-------------------------------------------------------------
The loop used to run on exactly one thread. The box measures that thread: 1.32 GB/s
for one `dd iflag=direct` reader against 5.14 GB/s for four on the same spill array,
and the live engine's own `reads_started=36 / read_seconds_max=0.0138` (11,468,800 B
per unit / 13.8 ms ~ 0.83 GB/s) agrees. So the read path is parallelisable, and now
it is: `VLLM_E2_TIER_DISK_READ_THREADS` names how many threads drain the one queue,
and the default is **1** - the same thread, with the same name, doing the same thing
in the same order.

What bounds in-flight work is NOT the thread count, it is the bounce-buffer pool:
a job carries a buffer taken at issue time and only returns it at the apply point, so
the pool must hold a whole offer before the first byte is read. `MAX_BUFFERS_BYTES`
(96 MiB default, `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` to raise it) is therefore the
constraint a thread count runs into, and `threads x pool` is the operator's knob
pair. The published witnesses say which one starved a boot:
`refused.buffer_cap` (no buffer at issue), `reader_starved_polls` (a thread that had
nothing to read), and `parallel_efficiency` (summed reader seconds over wall seconds
- ~1.0 with one thread, ~N when N threads really did read at once).

The discipline that made the single thread safe is unchanged and is the reason N of
them are safe: `_state_lock` is a plain Lock held ONLY for dict updates (a few
hundred nanoseconds), NEVER across a queue wait, a `sleep` or a file read, so an
engine step that wants `batch_status` cannot end up queued behind a kernel read. A
second rule joins it: a reader thread claims its job's buffer inside that lock and
is then the buffer's only owner, so cancel/finish and the reader can never both hand
the same buffer back to the pool (that was reachable with one thread and is more
reachable with four; see `_handle_read`).

Stdlib only. Bounded in every direction: one bounded queue, a bounded pool, a bounded
thread count, a bounded arrival-order sample per batch. Threads are created lazily on
the first accepted read (`disk_spill`'s discipline: nothing at import).
"""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import threading
import time
from collections import OrderedDict

from . import disk_index, disk_layout, snapshot

# --- the caps, and the unit of every one of them ----------------------------

# JOBS queued or in flight. 0.2.35 raises this from 64, and the NEW number is derived
# rather than picked: `disk_restore` caps one offer at `MAX_SLOTS_PER_OFFER` = 256 CPU
# slots, and an offer allocates exactly ONE read job per slot it allocates (one spilled
# file fills one destination block), so 256 is the largest job count ANY single offer
# can ever hand to `request_read()`. A queue at that depth can never be the bound that
# holes an offer while the CPU-slot cap still stands - the offer is cut to 256 slots
# before the queue is asked to hold 256 jobs. Below it, the surplus is refused
# `queue_full`, which is a member of `disk_restore.LOAD_CAPACITY_FAILURES`, which means
# the block arrives as NOTHING - and coverage is usable only up to the FIRST hole, so a
# 132-job offer into a 64-job queue is not "a smaller restore", it is a restore that
# stops at job 64 and recomputes everything behind it. That is how a 62-block run became
# a 15-block restore and cost ~19,200 tokens on 2026-10-08.
#
# WHY 256 AND NOT MORE: it is the offer's own job ceiling, so extra slots could only be
# filled by a SECOND concurrent offer - and concurrent offers are the open reservation
# arbiter (disk_restore TODO 14), not something a queue depth fixes. Going above the
# CPU-slot cap would trade a bound the offer path prices for a deeper pile of pinned
# buffers, and buy nothing.
#
# THE MEMORY CONSEQUENCE, because a queued job is not free: `request_read()` takes the
# bounce buffer BEFORE it queues (see `_take_buffer` above `put_nowait` there), so a
# QUEUED job HOLDS its buffer. That makes this constant an upper bound on queued host
# RAM - but NOT a second budget, because every buffer comes out of the pool and
# `_take_buffer` refuses at the pool's byte ceiling first. At a 1 GiB pool the queue can
# never hold more than the pool's own 1 GiB no matter how deep it is; the queue is the
# JOB-COUNT bound and the pool is the BYTE bound. What depth does cost directly is the
# descriptor and its batch-ledger entry, ~1.2 KB per job, so 256 slots are ~0.3 MB per
# rank process - noise beside the pool. `disk_restore._offer` prices this bound next to
# the pool budget and names it `read_queue` in `bounded_by`.
MAX_QUEUE_DEPTH = 256                 # JOBS queued or in flight
MAX_BATCHES = 32                      # OPEN batches tracked at once
MAX_JOBS_PER_BATCH = 4_096            # JOBS one batch may hold (sanity ceiling)
MAX_JOBS_REPORT = 16                  # FAILURES one status report may name
MAX_JOB_STATES_REPORT = 1024          # JOBS one state probe may name
MAX_BYTES_PER_READ = 64 * 1024 * 1024  # BYTES in ONE file (one slot's bytes)
MAX_BUFFERS_BYTES = 96 * 1024 * 1024  # BYTES held by the bounce-buffer pool
MAX_STRING_CHARS = 160                # CHARS of a path/reason kept in state
READ_POLL_SECONDS = 0.5               # SECONDS the thread waits on an empty queue

DEFAULT_READ_THREADS = 1              # THREADS draining the queue; 1 = pre-0.2.24
MAX_READ_THREADS = 16                 # a ceiling on that request, not a suggestion:
                                      # the box has 48 cores and four ranks, and a
                                      # reader that is not reading costs nothing, so
                                      # the cap only has to stop a typo
MAX_COMPLETION_ORDER = 512            # JOB IDS one batch may keep in completion order
MAX_THREAD_REPORT = MAX_READ_THREADS  # SLOTS one report may name
MAX_POOLED_BUFFERS = 256              # POOLED BUFFERS the free list may hold: an
                                      # OBJECT bound beside the byte bound, because a
                                      # raised byte ceiling would otherwise be capped
                                      # by a 32-entry list and quietly stop pooling.
                                      # Unreachable at the default pool, which hits
                                      # its byte bound at 8 buffers of 11.4 MB.
MAX_BUFFERS_BYTES_CEILING = 8 * 1024 * 1024 * 1024  # the LATEST word on the pool:
                                      # raising the pool is the operator's call, but
                                      # an unbounded ceiling would make the pool's
                                      # whole byte bound a suggestion, so a typo in a
                                      # systemd drop-in is CLAMPED and says so
                                      # (`pool_clamped`), never honoured

# The default patience for one batch: comfortably beyond the worst-case bounded
# read (a real batch is a few dozen slots of 11.4 MB, well under a second on a
# healthy SSD) and far below the point where a parked request looks to an
# operator like an engine hang. Expiring a batch costs a recompute; it never
# releases a destination that is still being written (see `cancel_batch`).
LOAD_DEADLINE_SECONDS = 30.0

# --- recency: a read that lands MOVES THE EVICTION CLOCK (0.2.29) ------------
#
# The capacity ceiling ranks spill files by `st_mtime` (see
# `disk_layout.lru_eviction_plan`), which until now moved on WRITE ONLY. The
# consequence, measured against what the tier is for: a span restored a hundred
# times was evicted by a span written an hour later and never read, because the
# eviction clock could not see a read at all. A restore is a TOUCH, and the
# operator's policy for the tier is "most read + most written, i.e. most touched,
# has priority" - so a successful read now bumps the file's mtime and the single
# existing clock keeps working. No new durable state, no replay after a restart:
# the promotion IS the mtime, and it survives a reboot exactly as well as the
# file does.
#
# `TOUCH_MIN_INTERVAL_SECONDS` is what makes that affordable. A `utime` is a
# metadata write to the same device the restore just read from, and a hot span is
# restored many times a minute, so an unthrottled promotion would add one
# filesystem transaction per read to a path that today does none. One promotion
# per file per interval, measured on the MONOTONIC clock (a wall-clock jump must
# not be able to freeze or flood the promotions).
#
# `MAX_TOUCH_TRACKED` bounds the throttle map itself, LRU-evicting its own
# entries. It is a THROTTLE, not a whitelist: dropping a path's entry costs the
# next read of that file one extra `utime`, and nothing else - so the map cannot
# become a second, larger, unbounded structure pretending to be a bound.
TOUCH_MIN_INTERVAL_SECONDS = 60.0
MAX_TOUCH_TRACKED = 4096

# The four verdicts of one promotion attempt, in the module's closed-vocabulary
# idiom. `skipped` is "this is not a file we may touch" (a bad name, a `.part`),
# never "something went wrong".
TOUCH_PROMOTED = "promoted"
TOUCH_THROTTLED = "throttled"
TOUCH_SKIPPED = "skipped"
TOUCH_FAILED = "failed"

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
    # THE CONCURRENCY LEDGER (0.2.24). `thread_alive` above keeps its old meaning
    # (may a read happen at all?) and is now "at least one reader is alive"; the
    # numbers below are the ones that answer "did parallelism help, and if not, why".
    # The CONFIGURED shape (threads requested/effective/clamped, pool bytes) is NOT
    # ledger state - it is derived from `_limits` in `counters()`, so a payload can
    # never report a thread count that the module does not currently believe.
    "threads_alive": 0,              # a COUNT, not a bool
    "thread_spawn_failures": 0,      # Thread.start() raised, counted per attempt
    "reader_starved_polls": 0,       # a reader waited on an EMPTY queue: the queue,
                                     # not the disk, was the limit that moment
    "units_in_flight": 0,            # jobs inside the kernel read, right now
    "units_in_flight_peak": 0,       # ...the boot's high-water: the proof N readers
                                     # did read at once, or the proof they never did
    "first_read_mono": None,         # the window aggregate throughput is measured
    "last_read_mono": None,          #   over; both monotonic, both boot-wide
    "completions_recorded": 0,       # completions that got a sequence number
    "completions_out_of_order": 0,   # ...completed BEFORE a lower-numbered job did
    # THE RECENCY PROMOTION (0.2.29). `recency_promotions` is the number that proves
    # the LRU clock is being moved by reads; `recency_promotions_throttled` is the
    # number that proves the throttle is doing its job (a high ratio is the expected
    # shape for a hot span - it means one metadata write per interval, not one per
    # read). `recency_promotion_failures` is `os.utime` raising: a read that landed
    # is still a good read when the bump fails, so this counter costs capacity
    # knowledge and nothing else.
    "recency_promotions": 0,
    "recency_promotions_throttled": 0,
    "recency_promotion_failures": 0,
    "last_reason": None,
    "last_error": None,
    "last_path": None,
}

# Open batches: batch id -> {"jobs": {job_id: job}, "opened_mono": float,
# "deadline_mono": float, "expected": int, "completion_order": [job_id...],
# "out_of_order": int}. Guarded by `_state_lock`: the engine thread writes it at
# bind and reads it in get_finished, and a reader thread marks its own jobs done.
# Every critical section is a dict update.
_batches: dict = {}
_buffers: list = []          # free (ctypes array, size) pairs, pooled
_state_lock = threading.Lock()
# The recency throttle: path -> monotonic seconds of its LAST promotion attempt,
# oldest-first so the map can evict its own LRU entry. Its OWN lock, deliberately:
# a promotion takes a filesystem metadata write, and `_state_lock` is the lock the
# engine's own `batch_status`/`get_finished` path waits on - holding it across a
# syscall is the one thing this module's design forbids. Order of acquisition is
# one-way (`_touch_lock` is never held while taking `_state_lock`), so the two
# cannot deadlock.
_touch_map: "OrderedDict[str, float]" = OrderedDict()
_touch_lock = threading.Lock()
_queue = None
_reader_thread = None        # slot 0, kept as the name-of-record for the ledger
_threads: list = []          # slot -> Thread | None; bounded by MAX_READ_THREADS
_slots: list = []            # slot -> per-thread witness dict, same bound
_limits: dict = {}           # the resolved pool/thread limits in force
_limits_resolved = False
_thread_lock = threading.Lock()
_notices: set = set()        # once-per-boot log lines, the repo's own idiom
_logger = logging.getLogger("kvtier_dev_plugin.disk_read")

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
    """Back to a fresh boot: no batches, no buffers, no counters, no thread set.

    DROPS THE QUEUE AND THE THREAD REFERENCES, which is what keeps the per-thread
    witnesses honest: a reader thread is bound to the queue object it was handed, so
    a thread left over from an earlier test drains THAT queue and nothing this test
    ever queues. Keeping the reference (the pre-0.2.24 behaviour) meant a test that
    asked for four readers left four live readers attached to the queue the next test
    inherited, and `threads_alive`/per-thread counters then described a boot that no
    longer existed. `disk_spill.reset_state_for_tests()` has always done exactly this
    (`_queue = None`, `_writer_thread = None`) for the same reason; nothing is joined
    or cancelled, so a test that wants a real shutdown still calls `stop_reader`.

    The limits are marked unresolved so the next `begin_batch` re-reads them from the
    environment the next test actually set.
    """
    global _batches, _buffers, _queue, _reader_thread, _threads, _slots
    global _limits, _limits_resolved
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
            "thread_started": 0, "thread_alive": False, "threads_alive": 0,
            "thread_spawn_failures": 0, "reader_starved_polls": 0,
            "units_in_flight": 0, "units_in_flight_peak": 0,
            "first_read_mono": None, "last_read_mono": None,
            "completions_recorded": 0, "completions_out_of_order": 0,
            "recency_promotions": 0, "recency_promotions_throttled": 0,
            "recency_promotion_failures": 0,
        })
        _limits = _default_limits()
        _limits_resolved = False
    # The throttle map belongs to a boot too: a leftover entry would let the next
    # test's first read of the same path be throttled by a promotion this test never
    # made. Cleared OUTSIDE `_state_lock` - the map has its own lock and the two are
    # never held together.
    with _touch_lock:
        _touch_map.clear()
    with _thread_lock:
        _queue = None
        _reader_thread = None
        _threads = []
        _slots = []
        _grow_witnesses()


def _default_limits() -> dict:
    """Today's behaviour, as data: one reader, the pool it has always had."""
    return {"threads": DEFAULT_READ_THREADS,
            "threads_requested": None, "threads_clamped": False,
            "buffer_bytes": MAX_BUFFERS_BYTES,
            "buffer_bytes_requested": None, "buffer_bytes_clamped": False,
            "problems": []}


def _resolve_limits() -> dict:
    """The pool size and thread count in force, from `config` (the only env reader).

    WHY A CACHE AND NOT A READ: `request_read`/`_take_buffer` run once per unit and
    must not touch the environment 4,096 times a step, so the values are resolved
    when a batch opens (the one moment a thread could need starting or respawning)
    and read cheaply from the cache everywhere else. Refreshing at `begin_batch`
    rather than once per boot also means a test's monkeypatch takes effect on the
    next bind, and on a real boot the engine's environment cannot change under us
    without a restart.

    The cache is REPLACED, never mutated, so a reader thread computing its pool limit
    concurrently sees either the old dict or the new one and never a half-written
    one. `config.disk_read_env()` returns None for "the operator said nothing", and
    None keeps the constant - so an unset environment is byte-identical to 0.2.23 by
    construction rather than by the two numbers being typed the same twice.
    """
    from . import config  # noqa: PLC0415 - config owns the env, this module must not
    global _limits, _limits_resolved
    env = config.disk_read_env()
    limits = _default_limits()
    limits["problems"] = list(env.get("problems") or [])
    wanted = env.get("threads")
    if wanted is not None:
        limits["threads_requested"] = int(wanted)
        limits["threads"] = max(1, min(MAX_READ_THREADS, int(wanted)))
        limits["threads_clamped"] = int(wanted) > MAX_READ_THREADS
    pooled = env.get("buffer_bytes")
    if pooled is not None:
        limits["buffer_bytes_requested"] = int(pooled)
        limits["buffer_bytes"] = max(1, min(MAX_BUFFERS_BYTES_CEILING, int(pooled)))
        limits["buffer_bytes_clamped"] = int(pooled) > MAX_BUFFERS_BYTES_CEILING
    _limits = limits
    _limits_resolved = True
    _announce_limits(limits)
    return limits


def _limits_now() -> dict:
    """The limits in force, resolving them on first use. Never raises.

    THE `global` LINE IS NOT DECORATION. Assigning `_limits` in the except branch
    without declaring it global makes it a LOCAL name for the whole function, so
    both `return _limits` lines raise UnboundLocalError on a boot where reading the
    environment failed - the `global` shadowing class that has crashed this codebase
    twice. Keep the declaration.
    """
    global _limits, _limits_resolved
    if not _limits_resolved:
        try:
            return _resolve_limits()
        except Exception as exc:  # noqa: BLE001 - a limit is never worth a fault
            _state["errors"] += 1
            snapshot._error(exc, "disk_read limits")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            _limits = _default_limits()
            _limits_resolved = True
        return _limits
    return _limits


def _notice_once(key: str, message: str) -> bool:
    """Log at most once per key per process: the repo's own idiom (`disk_index`,
    `disk_restore`, `miss_attribution`). A line on the engine path is printed once
    per episode, because one line per step is how a diagnostic stops being read.
    """
    try:
        if key in _notices:
            return False
        if len(_notices) >= 64:
            _notices.clear()
        _notices.add(key)
        _logger.warning("kvtier-dev-plugin: %s", str(message)[:640])
        return True
    except Exception:  # noqa: BLE001 - a log line is never worth a fault
        return False


def _announce_limits(limits: dict) -> None:
    """Say out loud, once, when this boot reads spilled bytes differently to today.

    The DEFAULT resolution prints nothing: a boot that changes nothing must not add
    a line to the engine log either.
    """
    for problem in limits["problems"]:
        _notice_once(f"env:{problem}", f"DISK[read] IGNORED: {problem}")
    if limits["threads"] == DEFAULT_READ_THREADS \
            and limits["buffer_bytes"] == MAX_BUFFERS_BYTES:
        return
    _notice_once(
        "limits", f"DISK[read] reader threads={limits['threads']} "
                  f"(requested={limits['threads_requested']}) "
                  f"buffer_pool={limits['buffer_bytes']} bytes "
                  f"(default={MAX_BUFFERS_BYTES}) "
                  f"clamped={int(limits['threads_clamped'])}/"
                  f"{int(limits['buffer_bytes_clamped'])}. The pool is PER PROCESS "
                  "and every engine rank is a process: the boot pays "
                  "pool bytes x rank count in resident memory.")


def _touch_map_entries() -> int:
    """How many paths the throttle currently remembers. Locked, cheap, never raises.

    Read OUTSIDE `_state_lock` by both payload builders: the map has its own lock and
    the two must never be held together.
    """
    try:
        with _touch_lock:
            return len(_touch_map)
    except Exception:  # noqa: BLE001 - a witness is never worth a fault
        return 0


def counters() -> dict:
    """A copy for a payload. Never the batch table, only its size."""
    with _state_lock:
        out = dict(_state)
        out["batches_open"] = len(_batches)
        out["queued"] = int(_state["queue_depth"])
    out["recency_promotion_map_entries"] = _touch_map_entries()
    out["recency_promotion_interval_seconds"] = TOUCH_MIN_INTERVAL_SECONDS
    out["recency_promotion_map_cap"] = MAX_TOUCH_TRACKED
    limits = _limits_now()
    out["threads_requested"] = limits["threads_requested"]
    out["threads_configured"] = int(limits["threads"])
    out["threads_clamped"] = bool(limits["threads_clamped"])
    out["threads_alive"] = thread_alive_count()
    out["thread_alive"] = thread_alive()
    out["env_problems"] = list(limits["problems"])
    out["pool_limit_bytes"] = int(limits["buffer_bytes"])
    out["pool_default_bytes"] = MAX_BUFFERS_BYTES
    out["pool_clamped"] = bool(limits["buffer_bytes_clamped"])
    held = int(out["buffers_held_bytes"])
    out["pool_headroom_bytes"] = max(0, int(limits["buffer_bytes"]) - held)
    out["pool_peak_bytes"] = int(out["buffers_peak_bytes"])
    out["threads_detail"] = thread_report()
    out["throughput"] = throughput()
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

    The byte ceiling is read BEFORE the lock is taken: it comes from `_limits`, and
    keeping every lock acquisition in this module to dict work only is what lets an
    engine step call this without queuing behind a reader thread. This is also the
    point where a pool that is too small for the offer shows up, as `buffer_cap` -
    the witness that says "threads could not have helped, there was no buffer".
    """
    limit = int(_limits_now()["buffer_bytes"])
    with _state_lock:
        for index, (buffer, capacity) in enumerate(_buffers):
            if capacity == size:
                _buffers.pop(index)
                return buffer
        held = int(_state["buffers_held_bytes"])
        if held + size > limit or size > MAX_BYTES_PER_READ:
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


def _give_buffer_locked(buffer, size: int) -> None:
    """Return a buffer to the pool. THE CALLER MUST HOLD `_state_lock`.

    Two forms exist because `_state_lock` is a plain `threading.Lock` and NOT an
    `RLock`: a caller that already holds it and calls a lock-taking helper blocks
    forever on a lock it owns - a self-deadlock on the calling thread, with no error
    and no traceback. That was real: `finish_batch` -> `_give_buffer` on the engine's
    `start_load_kv` path, and `_settle` -> `_give_buffer` on the reader thread's, so
    the module could never hand back a single buffer and any caller that closed a
    batch hung. A unit test found it by hanging. Every internal call site holds the
    lock, so this unlocked form is the default and the wrapper below exists only for
    the two call sites that provably run outside the lock (`_handle_read`'s
    already-cancelled path and `request_read`'s queue-full path).
    """
    if len(_buffers) >= MAX_POOLED_BUFFERS or size > MAX_BYTES_PER_READ:
        _state["buffers_held_bytes"] = max(0, int(_state["buffers_held_bytes"])
                                           - size)
        return
    _buffers.append((buffer, int(size)))


def _give_buffer(buffer, size: int) -> None:
    """The LOCK-TAKING form, for a caller that does NOT hold `_state_lock`."""
    with _state_lock:
        _give_buffer_locked(buffer, size)


def pool_state() -> dict:
    """The pool's budget in bytes: the ceiling, what is committed, and the headroom.

    WHY A SEPARATE READER EXISTS. `counters()` is the payload view and builds the
    whole concurrency, throughput and per-thread table (~2 KB of dict, ~40 lock-free
    reads) - far too much to call once per offered read, and it is not the shape the
    arithmetic wants either. This reads the three numbers the arithmetic IS: the
    ceiling from `_limits_now()` (taken BEFORE the lock, the `_take_buffer` idiom, so
    a caller never queues behind a reader thread for a budget read) and the committed
    bytes under the lock, which is held work only.

    `committed_bytes` is the number that makes a `buffer_cap` refusal readable: a
    pool that refused at 128,217,728 of 134,217,728 bytes committed says "the pool is
    too small for an offer this big", and one that refused at 11,468,800 says
    "something is holding buffers it is not returning". Those are different operator
    actions and the refusal cannot be triaged without the number.
    """
    limit = int(_limits_now()["buffer_bytes"])
    try:
        with _state_lock:
            held = int(_state["buffers_held_bytes"])
    except Exception as exc:  # noqa: BLE001 - a budget read never breaks a step
        snapshot._error(exc, "disk_read pool_state")
        held = 0
    return {"budget_bytes": limit,
            "committed_bytes": held,
            "headroom_bytes": max(0, limit - held)}


def afford_reads(bytes_per_job) -> dict:
    """How many `bytes_per_job`-byte reads THIS process's pool can carry at once.

    THE ARITHMETIC THE OFFER PATH ASKS FOR, owned here because this module owns the
    pool. A read takes a buffer at issue and returns it at the apply point (one
    forward pass later), so an offer whose jobs are all issued before any of them is
    applied needs the pool to hold the WHOLE offer at once - see the `pool.note` in
    `counters()`. `jobs_at_empty_pool` is therefore the largest job count ONE offer
    can ever be serviced with, and `jobs_now` is what still fits at this instant.

    A non-positive or unreadable size is `ok: false` with a reason: a zero would be a
    number a caller multiplies, and "the pool can carry infinite jobs" is the one
    answer this function must never give.
    """
    size = bytes_per_job
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return {"ok": False,
                "reason": f"bytes_per_job must be a positive int (unit: bytes per "
                          f"read), got {size!r}",
                "jobs_at_empty_pool": 0, "jobs_now": 0}
    state = pool_state()
    budget = int(state["budget_bytes"])
    committed = int(state["committed_bytes"])
    return {"ok": True,
            "bytes_per_job": size,
            "pool_budget_bytes": budget,
            "pool_committed_bytes": committed,
            "pool_headroom_bytes": int(state["headroom_bytes"]),
            # THE HARD CEILING: the whole pool divided by one job. An offer larger
            # than this can be refused by a pool no other reader is touching.
            "jobs_at_empty_pool": budget // size,
            # ...and what fits right now, which is never larger and is the number a
            # refusal at this instant actually hit.
            "jobs_now": max(0, budget - committed) // size,
            "note": ("a bounce buffer is taken at issue and returned at the apply "
                     "point, so one offer needs the pool to hold the whole offer; "
                     "the pool is per PROCESS, so N ranks cost N times these bytes "
                     "of host RAM")}


# --- the queue and its thread -------------------------------------------------

_STOP = object()


def _get_queue():
    global _queue
    with _thread_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        return _queue


def _thread_name(slot: int) -> str:
    """Slot 0 keeps the pre-0.2.24 name exactly, so `ps`/py-spy traces of a default
    boot are unchanged; the extra readers announce themselves as the extras."""
    return "kvtier-disk-read" if slot == 0 else f"kvtier-disk-read-{slot}"


def _grow_witnesses() -> None:
    """Pre-grow the thread/slot tables to the ceiling ONCE, so nothing else grows
    them. THE CALLER MUST HOLD `_thread_lock`.

    Pre-growing rather than growing on demand is what keeps a reader thread's
    bookkeeping lock-free: `_witness`/`_slot_alive` are then pure indexed reads, and
    N threads appending to a shared list (which is what on-demand growth would be,
    and `_state_lock` is not an RLock so it cannot be borrowed for it) is a race that
    mislabels the very witnesses this release exists to publish. Sixteen small dicts
    is the whole cost.
    """
    while len(_threads) < MAX_READ_THREADS:
        _threads.append(None)
    while len(_slots) < MAX_READ_THREADS:
        _slots.append({"index": len(_slots), "name": _thread_name(len(_slots)),
                       "jobs_read": 0, "bytes_read": 0, "read_seconds": 0.0,
                       "in_flight": 0, "peak_in_flight": 0, "starved_polls": 0,
                       "errors": 0, "spawned": 0})


def _witness(slot: int) -> dict | None:
    """THE CALLING THREAD's record, or None when the module does not publish it.

    The identity test is the point: `_threads[slot] is me` is false for a thread left
    over from before a `reset_state_for_tests()` (which drops the thread table, so
    such a thread keeps draining the queue it was handed and is no one's witness), and
    false for a thread this module never started. Those must not write into a ledger
    they are not part of, or a payload reports a boot that does not exist. Pure reads:
    no lock, no growth, no allocation - safe to call from a reader thread at any time
    and the reason `_state_lock` is never held for witness bookkeeping.
    """
    me = threading.current_thread()
    if not 0 <= slot < len(_threads) or _threads[slot] is not me:
        return None
    return _slots[slot] if 0 <= slot < len(_slots) else None


def _slot_alive(slot: int) -> bool:
    thread = _threads[slot] if 0 <= slot < len(_threads) else None
    return bool(thread is not None and thread.is_alive())


def _spawn_reader(slot: int) -> bool:
    """Start the reader for one slot. Returns False when the OS refused the thread.

    EVERY FAILURE PATH IS A REFUSAL, NOT AN EXCEPTION: `begin_batch` refuses the load
    when nothing is left to read with, which is what keeps `reads_started` from ever
    being nonzero while no thread exists to run a read.

    THE SLOT IS PUBLISHED BEFORE `start()`, and the order is not cosmetic. A reader
    thread's per-thread counters are written only when `_witness(slot)` recognises the
    caller as `_threads[slot]`; publishing after `start()` leaves a window in which a
    thread that gets to its first job immediately is nobody's witness, and the read it
    just did is missing from `threads[]` - a ledger that under-reports the very
    parallelism it exists to show. (A flaky run of
    `test_four_threads_read_four_units_at_once` found it: three of four reads
    attributed, four threads alive.) A `Thread` object exists before it runs, so
    publishing it early is always safe; `is_alive()` is False until it starts, so
    nothing can mistake an unstarted thread for a working one.
    """
    global _reader_thread
    try:
        thread = threading.Thread(target=_reader_loop, args=(slot,),
                                  name=_thread_name(slot), daemon=True)
        if slot < len(_threads):
            _threads[slot] = thread
        if slot == 0:
            _reader_thread = thread
        if slot < len(_slots):
            _slots[slot]["spawned"] += 1
        thread.start()
    except Exception as exc:  # noqa: BLE001 - no thread is a refusal
        snapshot._error(exc, "disk_read thread start")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        _state["thread_spawn_failures"] += 1
        if slot < len(_threads):
            _threads[slot] = None      # the slot is empty again, so it respawns
        if slot == 0:
            _reader_thread = None
        if slot < len(_slots):
            # `spawned` means ACTUALLY STARTED, so an attempt the OS refused is taken
            # back out; the attempt itself is counted by `thread_spawn_failures`.
            _slots[slot]["spawned"] = max(0, int(_slots[slot]["spawned"]) - 1)
        return False
    _state["thread_started"] += 1
    return True


def _ensure_thread() -> bool:
    """Ensure the configured number of reader threads is alive. Never at import.

    THE CONTRACT THIS FUNCTION OWNS (unchanged since 0.2.23, and it is a correctness
    one): False means "there is no reader", and `begin_batch` turns that into a
    `thread_dead` refusal so a batch can never be opened while nothing could read it.
    With N slots the honest reading of that is "not one reader is alive": three dead
    slots and one live one is a degraded boot, publishable and worth reading, but not
    a boot where a read is impossible, so it still returns True.

    A dead slot is RESPAWNED here (the single-thread release did the same thing with
    its one slot), so a thread lost to a `MemoryError` at some point in the boot's
    life comes back on the next bind instead of silently reducing the reader count
    forever. `threads_alive` is refreshed on every call, so a payload's count is
    never older than the last batch that opened.
    """
    global _reader_thread, _queue
    with _thread_lock:
        _grow_witnesses()
        # THE QUEUE IS CREATED HERE, not by the threads. Every reader is handed the
        # queue object it drains, and a thread that had to create it itself would
        # leave `_queue` None for as long as it took the first one to be scheduled -
        # which made `stop_reader()` answer "no thread was ever started" about three
        # live threads and `drain()` report a queue that existed. Built inline rather
        # than through `_get_queue()` because that helper takes `_thread_lock`, which
        # this function already holds, and this lock is not an RLock.
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        try:
            # Re-read the environment on every bind: a batch opening is the one
            # moment a thread could need starting or respawning, and it happens once
            # per step (not once per unit), so the refresh costs two `os.environ.get`
            # calls. Per-unit paths (`_take_buffer`, which runs once per unit) read
            # the CACHE instead, so a step never touches the environment 4,096 times.
            limits = _resolve_limits()
        except Exception as exc:  # noqa: BLE001 - a limit is never worth a fault
            snapshot._error(exc, "disk_read limits")
            _state["errors"] += 1
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            limits = _limits if _limits_resolved else _default_limits()
        wanted = max(1, min(MAX_READ_THREADS, int(limits["threads"])))
        alive = 0
        for slot in range(wanted):
            if not _slot_alive(slot):
                if _spawn_reader(slot):
                    alive += 1
            else:
                alive += 1
        _state["threads_alive"] = int(alive)
        _state["thread_alive"] = bool(alive)
        if alive == 0:
            _state["last_reason"] = ("no reader thread could be started: the load is "
                                     "refused rather than tracked unread")
        return bool(alive)


def thread_alive() -> bool:
    """May a read happen at all? True when AT LEAST ONE reader is alive."""
    for thread in list(_threads):
        if thread is not None and thread.is_alive():
            return True
    thread = _reader_thread
    return bool(thread is not None and thread.is_alive())


def thread_alive_count() -> int:
    return sum(1 for thread in list(_threads)
               if thread is not None and thread.is_alive())


def thread_report() -> list:
    """Per-thread witnesses, bounded by MAX_THREAD_REPORT. Reads no files.

    ONLY THE SLOTS THAT ARE IN PLAY are listed: every slot below the configured count
    (so a slot that died while the boot asked for four is still visible next to the
    `threads_alive` that says one is missing), plus any slot that is alive or has ever
    counted something (so a reader that outlived a shrinking configuration cannot
    vanish with its bytes). The slot tables are pre-grown to the ceiling, so without
    this filter a one-thread boot would print sixteen rows of nothing.

    THE QUESTION EACH FIELD ANSWERS, because the point of these is to be readable
    from across a room without a shell on the box:
      * `alive` vs `threads_configured` - did the threads the operator asked for
        survive the boot?
      * `jobs_read` / `bytes_read` per thread - is the queue actually spreading work,
        or is one thread doing everything while the rest idle (an I/O scheduler or
        cgroup answer, not a code answer)?
      * `in_flight` / `peak_in_flight` - are units being read SIMULTANEOUSLY? With
        one thread `peak_in_flight` can only ever be 1, so this is the field that
        distinguishes a parallel boot from a parallel-looking one.
      * `read_seconds` per thread against its own `jobs_read` - the per-READER rate.
        If aggregate bytes/s rose while THIS stayed put, the disk kept up and the
        parallelism was real; if it fell to about 1/N of the single-reader figure,
        the array was always the limit and the threads only share it.
      * `starved_polls` - a reader that waited out a whole poll on an empty queue.
        Many of those with threads configured > 1 means the reads were never issued
        (or the buffers were), i.e. starved on the queue/pool, not on the disk.
    """
    configured = int(_limits_now()["threads"])
    out: list = []
    for index, record in enumerate(list(_slots)[:MAX_THREAD_REPORT]):
        alive = _slot_alive(index)
        counted = any(int(record.get(key) or 0) for key in
                      ("jobs_read", "bytes_read", "starved_polls", "errors",
                       "spawned", "peak_in_flight"))
        if index >= configured and not alive and not counted:
            continue
        out.append({"index": int(record.get("index", index)),
                    "name": str(record.get("name"))[:MAX_STRING_CHARS],
                    "alive": alive,
                    "jobs_read": int(record.get("jobs_read") or 0),
                    "bytes_read": int(record.get("bytes_read") or 0),
                    "read_seconds": round(float(record.get("read_seconds") or 0.0), 6),
                    "in_flight": int(record.get("in_flight") or 0),
                    "peak_in_flight": int(record.get("peak_in_flight") or 0),
                    "starved_polls": int(record.get("starved_polls") or 0),
                    "errors": int(record.get("errors") or 0),
                    "spawned": int(record.get("spawned") or 0)})
    return out


def throughput() -> dict:
    """Aggregate read throughput over the boot, and the one number that says whether
    the threads ran together.

    TWO RATES, deliberately, because they answer different questions and a reader who
    is given only one will pick the wrong one:
      * `bytes_per_reader_second` = bytes / SUMMED reader seconds. A device figure:
        how fast one reader goes. If this collapses as threads rise, the array is
        saturated and more threads cannot help.
      * `bytes_per_second_wall` = bytes / WALL seconds between the first and last
        completed read. The figure the operator is buying: wall time is what sits
        inside the engine step.
      * `parallel_efficiency` = summed reader seconds / those wall seconds. ~1.0 for
        one reader, ~N for N readers that were all busy at once, and ~1.0 with N
        configured is the signature of "threads configured, parallelism absent" -
        which after this change is a number, not a guess.
    """
    with _state_lock:
        total = int(_state["bytes_read"])
        seconds = float(_state["read_seconds"])
        first = _state["first_read_mono"]
        last = _state["last_read_mono"]
    wall = (round(float(last) - float(first), 6)
            if first is not None and last is not None else 0.0)
    return {"bytes_read": total,
            "reader_seconds": round(seconds, 6),
            "wall_seconds": wall,
            "bytes_per_reader_second": (round(total / seconds, 1) if seconds > 0
                                        else None),
            "bytes_per_second_wall": (round(total / wall, 1) if wall > 0 else None),
            "parallel_efficiency": (round(seconds / wall, 3) if wall > 0 else None),
            "note": ("bytes_per_reader_second is the DEVICE rate and should stay put "
                     "as threads rise; bytes_per_second_wall is the WALL rate the "
                     "engine step pays and should rise with them. "
                     "parallel_efficiency is reader seconds over wall seconds: ~1.0 "
                     "with one reader, ~N when N readers truly read together.")}


def _reader_loop(slot: int) -> None:
    """The drain loop. N of these exist and they are the only place a file is read.

    A job carries a path, an expected size and a buffer it already owns; the loop does
    not need the engine's state to do its work, which is what makes it safe to leave
    running while the caller moves on. Nothing here holds `_state_lock` across a queue
    wait, a sleep or a read - the reason the engine's step thread can always get its
    `batch_status` in a few hundred nanoseconds, with one reader or sixteen. The queue
    is shared and FIFO, so DISPATCH order is submission order and never departs from
    it: a job cannot be dequeued while any earlier-submitted job is still queued. The
    only freedom is in completion - a job can finish ahead of the earlier jobs that
    were already inside the kernel when it started, which is at most (threads - 1) of
    them. That is the exact bound on how out-of-order this module can make anything.
    """
    local = _get_queue()
    while True:
        try:
            job = local.get(timeout=READ_POLL_SECONDS)
        except queue.Empty:
            # Nothing to read: the queue (or the pool that feeds it) was the limit,
            # not the disk. Counted so "4 threads, no speedup, why?" has an answer.
            record = _witness(slot)
            if record is None:
                # THE SELF-EXIT. `_witness` answers only for the thread the module
                # currently publishes in this slot, so an unpublished reader - one
                # left behind by `reset_state_for_tests()`, which drops the thread
                # table - recognises that it belongs to a boot that no longer exists
                # and retires on its next empty poll instead of daemon-tugging a dead
                # queue forever. It is NOT an exit for a healthy reader: a published
                # thread always has its record, whatever the configured count says,
                # and a thread with work to do never reaches this branch at all. The
                # cost is one identity comparison per 0.5 s poll, and the gain is
                # that no leftover thread can write into a later boot's counters.
                return
            with _state_lock:
                _state["reader_starved_polls"] += 1
                record["starved_polls"] += 1
            continue
        if job is _STOP:
            local.task_done()
            break
        if not isinstance(job, dict):    # a poison item: counted, never fatal
            # `request_read` only ever puts dicts here, so a non-dict came from
            # somewhere else (a test's stub queue, a future producer, a corruption).
            # Dropping it and KEEPING THE THREAD is the whole point: a reader that
            # dies on one bad item stops serving every later one, and the only clue
            # would be `thread_dead` on the next bind.
            with _state_lock:
                _state["errors"] += 1
                _state["last_error"] = "foreign queue item dropped"[:MAX_STRING_CHARS]
            _notice_once("poison", "DISK[read] DROPPED: the read queue handed the "
                                   "reader an item that is not a job; it was dropped "
                                   "and the reader stayed alive")
            local.task_done()
            continue
        try:
            _handle_read(job, slot)
        except Exception as exc:  # noqa: BLE001 - a job never kills the thread
            record = _witness(slot)
            with _state_lock:
                _state["errors"] += 1
                if record is not None:
                    record["errors"] += 1
            snapshot._error(exc, "disk_read job")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            _settle(job, _failed("read_error",
                                 f"the read raised {type(exc).__name__}"))
        finally:
            # Under `_state_lock`, because with N readers this line has N authors and
            # a read-modify-write of a shared counter is exactly the thing that must
            # not be best-effort. `max(0, ...)` stays: a depth that could go negative
            # would mean an author this module does not know about.
            with _state_lock:
                _state["queue_depth"] = max(0, int(_state["queue_depth"]) - 1)
            local.task_done()


def _handle_read(job, slot: int = 0) -> None:
    """Read one job's file into the buffer THIS thread now owns.

    THE BUFFER IS CLAIMED INSIDE THE LOCK, and that is the whole reason this function
    looks fussy. `cancel_batch`/`finish_batch` return a QUEUED job's buffer to the
    pool and null the reference; if this thread had grabbed the reference BEFORE the
    lock (which is what 0.2.23 did) and the cancel won, both of them would hand the
    SAME buffer back - the pool then holds one buffer twice, hands it to two later
    reads, and one request's file bytes are memmoved into another request's CPU KV
    slot. Silent corruption, and more reachable with four readers in flight than with
    one. Claiming here means: after this point the buffer has exactly one owner, this
    thread, and it returns it exactly once (through `_settle`).
    """
    path = job.get("path")
    size = int(job["expect_bytes"])
    record = _witness(slot)
    with _state_lock:
        held = _batches.get(job["batch"])
        entry = held["jobs"].get(job["job_id"]) if held else None
        if entry is None or entry["state"] == CANCELLED:
            # The batch is gone or this job was cancelled. Whoever did it already
            # returned the buffer under this same lock, so there is nothing here to
            # give back - and giving it back would be the double return above.
            job["state"] = CANCELLED
            job["buffer"] = None
            return
        entry["state"] = RUNNING
        buffer = job.get("buffer")
        if record is not None:
            record["in_flight"] = int(record["in_flight"]) + 1
            record["peak_in_flight"] = max(int(record["peak_in_flight"]),
                                           int(record["in_flight"]))
        _state["units_in_flight"] = int(_state["units_in_flight"]) + 1
        _state["units_in_flight_peak"] = max(int(_state["units_in_flight_peak"]),
                                             int(_state["units_in_flight"]))
    if buffer is None:                 # cancelled before it ever started
        _finish_in_flight(slot)
        return
    started = time.monotonic()
    outcome = _read_file(path, buffer, size)
    elapsed = round(time.monotonic() - started, 6)
    finished = time.monotonic()
    with _state_lock:
        _state["read_seconds"] = round(float(_state["read_seconds"]) + elapsed, 6)
        _state["max_read_seconds"] = round(max(float(_state["max_read_seconds"]),
                                               elapsed), 6)
        _state["last_read_mono"] = finished
        if _state["first_read_mono"] is None:
            _state["first_read_mono"] = started
        if outcome["ok"]:
            _state["reads_completed"] += 1
            _state["bytes_read"] = int(_state["bytes_read"]) + size
            _state["last_path"] = str(path)[:MAX_STRING_CHARS]
        if record is not None:
            record["jobs_read"] = int(record["jobs_read"]) + 1
            record["read_seconds"] = round(float(record["read_seconds"]) + elapsed, 6)
            if outcome["ok"]:
                record["bytes_read"] = int(record["bytes_read"]) + size
    _finish_in_flight(slot)
    _settle(job, outcome)
    if outcome["ok"]:
        # THE RECENCY PROMOTION (0.2.29). HERE, and for one reason: this is the
        # single place a reader knows a file produced usable bytes, and it runs on a
        # reader thread - `_reader_loop` -> `_handle_read` -> here - so the metadata
        # write can never land on `get_finished`, on the scheduler step, or on any
        # engine thread. It sits AFTER `_settle` so a slow `utime` cannot delay the
        # job's completion (the apply point sees the bytes the moment they are read)
        # and after `_finish_in_flight` so it does not inflate `units_in_flight`,
        # which measures kernel reads. A failed bump is counted inside
        # `promote_recency` and changes nothing about this job: the restore already
        # succeeded. A FAILED outcome (`file_missing`, `size_mismatch`, `short_read`,
        # `not_ours`, `read_error`) deliberately does not reach this line: keeping a
        # wrong-sized or unreadable file alive is worse than evicting it.
        promote_recency(path)


def _finish_in_flight(slot: int) -> None:
    """Retire this thread's in-flight unit. Locked, bounded, never raises."""
    try:
        record = _witness(slot)
        with _state_lock:
            _state["units_in_flight"] = max(0, int(_state["units_in_flight"]) - 1)
            if record is not None:
                record["in_flight"] = max(0, int(record["in_flight"]) - 1)
    except Exception as exc:  # noqa: BLE001 - a witness never breaks a read
        snapshot._error(exc, "disk_read in-flight bookkeeping")


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


def _touch_clock() -> float:
    """The throttle's clock: `time.monotonic()`, behind a NAME a test can rebind.

    Two clocks, deliberately, and mixing them up is a silent inversion of the whole
    feature. THIS one is monotonic (it answers "how long since the last promotion?",
    and a wall-clock adjustment must never be able to freeze the promotions for an
    hour or release one per read). The timestamp WRITTEN to the file is wall-clock
    `now`, by `os.utime`'s own current-time form: the eviction comparator compares
    these mtimes against the mtimes of files written by a spill at a real date, so a
    monotonic value written into `st_mtime` would land every promoted file in 1970 and
    evict the hottest spans FIRST. Keep the two apart.

    Tests rebind this function rather than sleeping: a suite that waits out a 60 s
    window to prove a 60 s window is a suite that never runs.
    """
    return time.monotonic()


def promote_recency(path, now: float | None = None) -> str:
    """Move one spilled file to the FRONT of the eviction queue. Never raises.

    THE CONTRACT, in the direction that matters: this is called for a read that
    ALREADY produced usable bytes, and it is the only place the tier admits that a
    file was just used. Eviction ranks by `st_mtime`, so bumping it IS the
    promotion - no new durable state, nothing to replay after a restart, and no new
    input to `disk_layout.lru_eviction_plan`.

    THREE RULES, each with a failure it exists to prevent:

    1. THROTTLED: at most one `utime` per file per `TOUCH_MIN_INTERVAL_SECONDS`, on
       the monotonic clock. A hot span is restored many times a minute and an
       unthrottled bump adds a metadata transaction to the device on the same path
       that just paid a 11.4 MB read - the promotion would compete with the reads it
       is meant to serve.
    2. BOUNDED: the map remembers `MAX_TOUCH_TRACKED` paths and evicts its own oldest
       entry. Forgetting a path costs one extra `utime` on its next read, so the
       bound degrades to slightly more metadata writes and never to a wrong answer.
    3. FAIL-CLOSED ON THE NAME: a path that does not parse as one of our committed
       spill files, or that parses as an in-flight `.part`, is not touched at all.
       The CALLER owns the byte-level rule (only an `ok` read promotes), so a
       `size_mismatch`, `short_read`, `file_missing`, `not_ours` or `read_error`
       never reaches here; this check is the second lock on the door, because moving
       the clock of a wrong-sized or unreadable file keeps dead weight alive.

    The map is updated BEFORE the syscall on purpose, and the failure counter beside
    it: a path whose `utime` raises (evicted in the gap between the read and the
    bump, most likely) is then retried at most once per interval instead of once per
    read, which is the whole point of the throttle. A failed bump costs capacity
    knowledge and never a read - the bytes are already applied by the time this runs.

    Returns a verdict NAME (`promoted` | `throttled` | `skipped` | `failed`) so a
    caller can count it; nothing here raises into a reader thread.
    """
    text = None
    try:
        if not isinstance(path, (str, os.PathLike)):
            return TOUCH_SKIPPED
        parsed = disk_layout.parse_path(path)
        if not parsed["ok"] or parsed["partial"]:
            return TOUCH_SKIPPED
        text = os.fspath(path)
        moment = _touch_clock() if now is None else float(now)
        with _touch_lock:
            previous = _touch_map.get(text)
            if (previous is not None
                    and moment - float(previous)
                    < TOUCH_MIN_INTERVAL_SECONDS):
                # A hit inside the window. `move_to_end` keeps a hot file hot in the
                # map: the bound then evicts the paths nobody read, which is exactly
                # the population the throttle exists to serve.
                _touch_map.move_to_end(text)
                verdict = TOUCH_THROTTLED
            else:
                _touch_map[text] = moment
                _touch_map.move_to_end(text)
                while len(_touch_map) > MAX_TOUCH_TRACKED:
                    _touch_map.popitem(last=False)
                verdict = TOUCH_PROMOTED
        if verdict != TOUCH_PROMOTED:
            with _state_lock:
                _state["recency_promotions_throttled"] += 1
            # THE INDEX GETS THE TOUCH TOO (0.2.45), on a THROTTLED hit as much as on a
            # promoted one: both mean "this file produced bytes just now", and the
            # throttle exists to spare the DEVICE a metadata transaction,
            # not to make the
            # in-memory LRU clock lie. Without this the file's `st_mtime`
            # moves while the
            # index that maps to it stays where it was inserted, and the tier ends up
            # evicting the MAP of its hottest spans while keeping the spans - the one
            # inversion that makes an index worse than no index.
            disk_index.note_restored_path(text)
            return verdict
        # `os.utime(path, None)` is the API's own "both timestamps to the current
        # time", in the clock that OWNS st_mtime (wall). An explicit `ns` built from
        # the monotonic throttle clock would date every promoted file to 1970 - see
        # `_touch_clock`. (`os.utimes` is the name the task named; it is NOT in this
        # interpreter - Python removed the alias - so `os.utime` is what works, and it
        # is the same syscall with the same current-time form.)
        os.utime(text, None, follow_symlinks=False)
        with _state_lock:
            _state["recency_promotions"] += 1
        # The promoted path gets the same index touch (see the throttled branch).
        disk_index.note_restored_path(text)
        return verdict
    except Exception as exc:  # noqa: BLE001 - a bump is never worth losing a read
        with _state_lock:
            _state["recency_promotion_failures"] += 1
            _state["last_error"] = (
                f"recency promotion: {type(exc).__name__}")[:MAX_STRING_CHARS]
        _notice_once("recency-failure",
                     f"DISK[read] a recency promotion could not be written for "
                     f"{os.path.basename(str(text or path))[:60]}: "
                     f"{type(exc).__name__}. The read itself landed; eviction will "
                     f"see this file's old mtime.")
        return TOUCH_FAILED


def _note_completion_locked(batch, job_id: int) -> None:
    """Record WHEN this job finished relative to its siblings. Caller holds the lock.

    WHY A READER HAS TO CARE ABOUT ORDER AT ALL. A restore is only useful when a span
    is complete across its groups, and the submission order IS span-major,
    group-minor (`disk_restore._merge_pending_hit` walks `for span: for group:`).
    With one reader, completions therefore arrived in that order too, so an apply
    point that caught a batch mid-flight saw a PREFIX of spans done - a hole at the
    END. With N readers the completions interleave, and the same mid-flight batch can
    show span 3 complete while span 1 is still queued: a hole in the MIDDLE. This
    module does not decide what a partial restore is worth (another layer owns that,
    and must keep owning it); it publishes the order so that decision is made against
    a measurement instead of an assumption. See README, *Reading the spill in
    parallel: the ordering question*.

    The FIFO queue bounds the disorder, and the bound is exact rather than
    approximate: dispatch order IS submission order (no job dequeues while an earlier
    one is still queued), so a job can only complete ahead of the earlier jobs already
    in flight when it started - at most (threads - 1) of them. The scatter is a window
    of N, never the whole batch. That bound is pinned by
    `tests/test_disk_read_threads.py`.

    Bounded: the order list stops at `MAX_COMPLETION_ORDER` job ids per batch and
    says so via `completion_order_truncated`; the inversion counter keeps counting
    even after the list is full, because the count is the witness and the list is the
    illustration.
    """
    order = batch.setdefault("completion_order", [])
    highest = int(batch.get("highest_completed", -1))
    if int(job_id) < highest:
        batch["out_of_order"] = int(batch.get("out_of_order", 0)) + 1
        _state["completions_out_of_order"] += 1
    else:
        batch["highest_completed"] = int(job_id)
    _state["completions_recorded"] += 1
    if len(order) < MAX_COMPLETION_ORDER:
        order.append(int(job_id))
    batch["completion_truncated"] = len(order) >= MAX_COMPLETION_ORDER


def _settle(job, outcome: dict) -> None:
    """Publish a job's outcome into its batch. Locked, and last word wins."""
    if not isinstance(job, dict):
        # A poison item has no batch to report to. Returning here is what keeps
        # `_reader_loop`'s promise that a job never kills the thread: `_settle` is
        # called from that loop's OWN error handler, so raising here is what killed a
        # reader for good on a garbage queue item (and a dead reader is a boot that
        # reads nothing, with `thread_dead` the only clue).
        return
    with _state_lock:
        batch = _batches.get(job["batch"])
        entry = batch["jobs"].get(job["job_id"]) if batch else None
        if entry is None:
            # The batch was finished or cancelled out from under the read. The
            # buffer goes back to the pool and the bytes are dropped: this is the
            # case the bounce buffer exists for.
            buffer = job.get("buffer")
            if buffer is not None:
                _give_buffer_locked(buffer, int(job["expect_bytes"]))
            return
        if entry["state"] == CANCELLED:
            buffer = job.get("buffer")
            if buffer is not None:
                _give_buffer_locked(buffer, int(job["expect_bytes"]))
            return
        if outcome.get("ok"):
            entry["state"] = DONE
            entry["bytes"] = int(outcome["bytes"])
            _note_completion_locked(batch, int(job["job_id"]))
        else:
            entry["state"] = FAILED
            entry["reason_kind"] = str(outcome.get("reason_kind", "read_error"))
            entry["reason"] = str(outcome.get("reason"))[:MAX_STRING_CHARS]
            _note_completion_locked(batch, int(job["job_id"]))
            buffer = job.get("buffer")
            if buffer is not None:
                _give_buffer_locked(buffer, int(job["expect_bytes"]))
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
            # THE ARRIVAL ORDER (0.2.24): the job ids in the order they finished,
            # plus the count of finishes that landed BEFORE a lower-numbered job.
            # Empty until a read settles, bounded by MAX_COMPLETION_ORDER, and read
            # by `batch_status` - see `_note_completion_locked` for why a reader
            # module cares about order at all.
            "completion_order": [], "out_of_order": 0,
            "highest_completed": -1, "completion_truncated": False,
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
            # THE FOUR NUMBERS OF A `buffer_cap`, carried out of the refusal rather
            # than left in a counter. `pool_budget_bytes`/`pool_committed_bytes` say
            # whether the pool is too small for the offer or is being held by
            # somebody else, `bytes_needed` is the unit this store writes, and
            # `jobs_that_fit` is the largest offer THIS pool can ever service - which
            # is the figure the scheduler's offer cap is supposed to have used, so a
            # refusal that reaches the log names the ceiling it exceeded instead of
            # the fact that it exceeded one.
            pool = pool_state()
            return _refused("buffer_cap",
                            f"the bounce-buffer pool cannot hold another "
                            f"{expect_bytes}-byte buffer "
                            f"({int(pool['committed_bytes'])} of "
                            f"{int(pool['budget_bytes'])} bytes committed, cap "
                            f"{int(_limits_now()['buffer_bytes'])}): this load is "
                            "refused, not queued onto an unbounded pile",
                            pool_budget_bytes=int(pool["budget_bytes"]),
                            pool_committed_bytes=int(pool["committed_bytes"]),
                            bytes_needed=int(expect_bytes),
                            jobs_that_fit=max(0, int(pool["headroom_bytes"])
                                              // int(expect_bytes)))
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
    reported as a failure rather than a success. NOTHING HERE DEPENDS ON THE ORDER
    jobs finished in: `accepted` is fixed at submit, `settled` is a count, and
    `running` may now be as large as the reader count (it could only ever be 0 or 1
    with one thread, which is why a boot's `running` histogram is itself a witness
    that the threads are real).
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
                         completion_order=[], completion_order_truncated=False,
                         out_of_order_completions=0,
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
        # THE ARRIVAL ORDER, published rather than assumed away. `completion_order` is
        # the job ids in the order they ANSWERED (span-major, group-minor is the
        # SUBMISSION order, so this list equals `[0,1,2,...]` exactly when the reads
        # behaved like the single-thread release); `out_of_order_completions` counts
        # the answers that arrived before a lower-numbered one did. A restore that
        # needs "is this span complete?" reads the two together - see
        # `disk_restore.span_shape`, which turns them into whole-span counts.
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
               "completion_order": [int(item) for item
                                    in batch.get("completion_order") or []],
               "completion_order_truncated": bool(
                   batch.get("completion_truncated")),
               "out_of_order_completions": int(batch.get("out_of_order") or 0),
               "failures": failures}



def job_states(batch_id, limit: int = MAX_JOB_STATES_REPORT) -> dict:
    """The state of every job in one batch, for a reader that must say WHICH ONE.

    `batch_status` answers the aggregate question (`queued: 3, running: 1`) and that
    is enough to decide whether to apply bytes. It cannot answer the per-entry
    question the restore layer has to answer when it drops one entry: did THIS read
    never start (`queued` - the job sat in the queue and the engine got here first)
    or was it inside the kernel when the engine arrived (`running` - the read was
    executing and still lost)? Those are different bugs on different boxes - one is
    thread scheduling, the other is a slow device - and flattening them into
    `not_settled` made the counter that is supposed to explain the race explain
    nothing. READ-ONLY: this takes the lock, reads `state`, and gives back nothing
    but names.

    `truncated` is published rather than assumed away: a probe that silently names
    fewer jobs than the batch holds would let a caller attribute a state it never
    read, and a caller that attributes a state it never read is the reason this
    module counts things instead of guessing them.
    """
    with _state_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return {"known": False, "states": {}, "count": 0, "truncated": False}
        count = len(batch["jobs"])
        kept = sorted(batch["jobs"].items())[:max(1, int(limit))]
        return {"known": True,
                "states": {int(job_id): str(entry["state"])
                           for job_id, entry in kept},
                "count": count,
                "truncated": count > len(kept)}


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


def finish_batch(batch_id) -> dict:
    """Close a batch and return the buffers nobody is writing into any more.

    EXACTLY-ONE-OWNER IS THE RULE, and the owner of a QUEUED or RUNNING job's buffer
    is the READER THREAD, not this function. Returning one of those to the pool is a
    double return: the pool then holds the SAME buffer twice, hands it to two later
    reads, and one request's file bytes are memmoved into another request's CPU KV
    slot - the silent-corruption shape this module exists to make impossible. It was
    reachable through the ordinary restore path (`start_load_kv` while a read is
    still in flight, which is exactly what `dma_unsettled_entries` counts), and a
    probe that closes a batch with a RUNNING job found the pool holding one buffer
    twice. So the same discipline `cancel_batch` already kept: recycle QUEUED (the
    thread skips a bufferless job), DONE and FAILED buffers, null them so no path
    returns them twice, and leave RUNNING alone - `_settle` brings those back when
    the thread finds its batch gone.
    """
    with _state_lock:
        batch = _batches.pop(batch_id, None)
        if batch is None:
            return _ok(closed=False, reason="the batch was already closed")
        released = 0
        for entry in batch["jobs"].values():
            if entry["state"] == RUNNING:
                continue
            job = entry.get("job") or {}
            buffer = job.get("buffer")
            if buffer is None:
                continue
            job["buffer"] = None
            released += 1
            _give_buffer_locked(buffer, int(job["expect_bytes"]))
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
                    _give_buffer_locked(buffer, int(entry["job"]["expect_bytes"]))
            elif entry["state"] == RUNNING:
                running += 1
        _state["batches_cancelled"] += 1
        _note(f"batch {batch_id!r} cancelled: {reason}")
        return _ok(cancelled=True, running=running,
                   reason=str(reason)[:MAX_STRING_CHARS])


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
    """Ask EVERY reader to exit. Nothing relies on this at shutdown today.

    N sentinels, one per live thread, because a single `_STOP` frees exactly one
    waiter and leaves the others parked on the queue forever. The count asked for is
    the number of threads currently alive, so a boot with one thread sends one packet
    and a boot that never started a thread sends none. `threads_stopped` is published
    so a caller can see a partial stop instead of inferring one.
    """
    global _reader_thread
    local = _queue
    live = [(index, thread) for index, thread in enumerate(list(_threads))
            if thread is not None and thread.is_alive()]
    if not live and _reader_thread is not None and _reader_thread.is_alive():
        live = [(0, _reader_thread)]
    if local is None:
        return {"stopped": True, "reason": "no thread was ever started",
                "threads_stopped": 0, "threads_alive": 0}
    for _index, _thread in live:
        try:
            local.put_nowait(_STOP)
        except queue.Full:
            return {"stopped": False, "reason": "the queue is full; try again",
                    "threads_stopped": 0,
                    "threads_alive": len(live)}
    for _index, thread in live:
        thread.join(max(0.0, float(timeout_seconds)))
    still = [(index, thread) for index, thread in live if thread.is_alive()]
    for index, _thread in still:
        if index < len(_threads):
            _threads[index] = None      # a dead slot is respawned by `_ensure_thread`
    if not still:
        _reader_thread = None
    alive = len(still)
    _state["threads_alive"] = int(alive)
    _state["thread_alive"] = bool(alive)
    return {"stopped": alive == 0,
            "reason": (None if alive == 0
                       else "some threads did not exit inside the timeout"),
            "threads_stopped": len(live) - alive, "threads_alive": alive,
            "threads_still_alive": [index for index, _thread in still]}


# --- the view -----------------------------------------------------------------


def section() -> dict:
    """This module's state for a snapshot or a verb. Reads nothing from disk."""
    with _state_lock:
        snapshot_state = dict(_state)
        snapshot_state["batches_open"] = len(_batches)
    snapshot_state["recency_promotion_map_entries"] = _touch_map_entries()
    snapshot_state["recency_promotion_interval_seconds"] = TOUCH_MIN_INTERVAL_SECONDS
    snapshot_state["recency_promotion_map_cap"] = MAX_TOUCH_TRACKED
    limits = _limits_now()
    snapshot_state["thread_alive"] = thread_alive()
    snapshot_state["threads_alive"] = thread_alive_count()
    snapshot_state["threads_requested"] = limits["threads_requested"]
    snapshot_state["threads_configured"] = int(limits["threads"])
    snapshot_state["threads_default"] = DEFAULT_READ_THREADS
    snapshot_state["threads_clamped"] = bool(limits["threads_clamped"])
    snapshot_state["env_problems"] = list(limits["problems"])
    snapshot_state["threads"] = thread_report()
    snapshot_state["throughput"] = throughput()
    snapshot_state["pool"] = {
        "limit_bytes": int(limits["buffer_bytes"]),
        "default_bytes": MAX_BUFFERS_BYTES,
        "requested_bytes": limits["buffer_bytes_requested"],
        "clamped": bool(limits["buffer_bytes_clamped"]),
        "held_bytes": int(snapshot_state["buffers_held_bytes"]),
        "peak_bytes": int(snapshot_state["buffers_peak_bytes"]),
        "headroom_bytes": max(0, int(limits["buffer_bytes"])
                              - int(snapshot_state["buffers_held_bytes"])),
        # THE NUMBER THAT ANSWERS "would more threads have helped?": a pool that
        # holds fewer units than one offer needs means the threads were never the
        # binding constraint. Measured against the unit this store writes, 11,468,800 B
        # for a 3,200-token group and 12,902,400 B otherwise.
        "units_of_11468800": int(limits["buffer_bytes"]) // 11_468_800,
        "units_of_12902400": int(limits["buffer_bytes"]) // 12_902_400,
        # THE SAME ARITHMETIC AGAINST THE UNIT THIS BOOT ACTUALLY READS, so the answer
        # to "how many units can be in flight, and therefore how many threads are worth
        # starting" is a field rather than a calculation someone has to redo (and get
        # wrong: the live boot reads 11,468,800 B units and a wide rank reads
        # 12,902,400 B, and the two give different answers). None until a read lands,
        # because an average of nothing is not a unit size.
        "measured_unit_bytes": (int(snapshot_state["bytes_read"])
                                / int(snapshot_state["reads_completed"])
                                if int(snapshot_state["reads_completed"]) else None),
        "units_of_measured_unit": (int(limits["buffer_bytes"])
                                   // (int(snapshot_state["bytes_read"])
                                       // int(snapshot_state["reads_completed"]))
                                   if int(snapshot_state["reads_completed"]) else None),
        "refused_for_no_buffer": int(
            (snapshot_state.get("refused") or {}).get("buffer_cap") or 0),
        "note": ("the pool bounds IN-FLIGHT BYTES per process regardless of the "
                 "thread count, and a buffer is taken at issue and returned at the "
                 "apply point, so one offer needs the pool to hold the whole offer. "
                 "units_of_measured_unit is also the MAXIMUM USEFUL thread count: a "
                 "thread beyond that number has no buffer to read into and only adds "
                 "reader_starved_polls. Increasing the pool costs host RAM equal to "
                 "the increase times the rank count, because every rank is its own "
                 "process.")}
    snapshot_state["concurrency"] = {
        "threads_configured": int(limits["threads"]),
        "threads_alive": int(snapshot_state["threads_alive"]),
        "units_in_flight": int(snapshot_state["units_in_flight"]),
        "units_in_flight_peak": int(snapshot_state["units_in_flight_peak"]),
        "reader_starved_polls": int(snapshot_state["reader_starved_polls"]),
        "queue_depth": int(snapshot_state["queue_depth"]),
        "queue_depth_peak": int(snapshot_state["queue_depth_peak"]),
        "completions_recorded": int(snapshot_state["completions_recorded"]),
        "completions_out_of_order": int(snapshot_state["completions_out_of_order"]),
        # THE POOL'S VERDICT ON THE THREAD COUNT, as a number: threads configured
        # above this figure are dead weight on this boot's unit size, and the two
        # counters beside it (`reader_starved_polls`, `refused.buffer_cap`) say which
        # side of the pool the starvation showed up on.
        "threads_useful_max": (int(limits["buffer_bytes"])
                               // (int(snapshot_state["bytes_read"])
                                   // int(snapshot_state["reads_completed"]))
                               if int(snapshot_state["reads_completed"]) else None),
        # THE THREE-WAY ANSWER TO "why did parallelism not help", as data:
        # buffers (pool refused at issue) vs queue (readers idled) vs disk (both
        # low, per-reader rate fell as threads rose).
        "starved_on": ("buffers" if int(
            (snapshot_state.get("refused") or {}).get("buffer_cap") or 0) > 0
            else "queue" if int(snapshot_state["reader_starved_polls"]) > 0
            else "disk-or-threads-are-idle"),
    }
    snapshot_state["refused"] = dict(snapshot_state["refused"])
    snapshot_state["failed"] = dict(snapshot_state["failed"])
    snapshot_state["caps"] = {
        "max_queue_depth": MAX_QUEUE_DEPTH, "max_batches": MAX_BATCHES,
        "max_jobs_per_batch": MAX_JOBS_PER_BATCH,
        "max_bytes_per_read": MAX_BYTES_PER_READ,
        # `max_buffers_bytes` stays the EFFECTIVE pool (it equals the constant unless
        # an operator raised it), and the constant itself is published beside it so a
        # reader can tell a default boot from a tuned one without knowing the default.
        "max_buffers_bytes": int(limits["buffer_bytes"]),
        "max_buffers_bytes_default": MAX_BUFFERS_BYTES,
        "max_buffers_bytes_ceiling": MAX_BUFFERS_BYTES_CEILING,
        "max_pooled_buffers": MAX_POOLED_BUFFERS,
        "read_threads": int(limits["threads"]),
        "read_threads_default": DEFAULT_READ_THREADS,
        "read_threads_ceiling": MAX_READ_THREADS,
        "max_completion_order": MAX_COMPLETION_ORDER,
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
        "batches_cancelled": "batches", "errors": "raised exceptions",
        "threads_alive": "reader threads currently alive",
        "thread_started": "reader threads this process has started",
        "thread_spawn_failures": "Thread.start() refusals",
        "reader_starved_polls": "polls that waited on an empty queue",
        "units_in_flight": "reads inside the kernel right now",
        "units_in_flight_peak": "the most reads ever seen inside the kernel at once",
        "completions_recorded": "job answers given a completion sequence number",
        "completions_out_of_order": "answers that landed before a lower-numbered "
                                    "job's answer did"}
    snapshot_state["note"] = (
        "reads land in a BOUNCE BUFFER, never in a KV slot: the caller memmoves "
        "them on the engine thread when it submits the DMA, which is what makes "
        "'we cancelled this load' and 'the destination was never touched' the "
        "same fact. A cancelled RUNNING read still costs its buffer until the "
        "kernel returns, and its bytes are dropped. Since 0.2.24 the queue can be "
        f"drained by up to {int(limits['threads'])} reader thread(s); the queue is "
        "FIFO, so dispatch order is submission order and a job can only complete "
        "ahead of the (threads - 1) earlier jobs that were already in flight when it "
        "started. Completion ORDER is "
        "published per batch (`completion_order`, `out_of_order_completions`) "
        "because a mid-flight batch used to look like a prefix of spans and can now "
        "look like a hole in the middle.")
    return snapshot_state
