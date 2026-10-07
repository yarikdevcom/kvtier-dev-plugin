r"""Write restored KV bytes straight into the GPU window, at the apply point.

THE IDEA, IN ONE SENTENCE: the bytes a disk restore wants in a GPU block are already
sitting in a pinned CPU slot when `start_load_kv`'s wrapper runs, so the plugin can put
its OWN host->device copy on the tier's load stream there - strictly before the fork's
copy is even submitted - and then read the window back to itself and compare that with
the file. The write is a second opinion about the destination, not a replacement for
the fork's copy: the fork's arena->GPU copy still runs afterwards on the same stream and
therefore has the LAST word on every byte.

WHY THE ORDERING IS STRUCTURAL AND NOT A HOPE
---------------------------------------------
The apply point is `disk_restore._worker_before_load`, and the wrapper that reaches it
is `_wrapped_start_load` (`disk_restore.py:8177-8181`):

    def start_load_kv(self, *args, **kwargs):
        worker_before_load(self)        # <- this module's hook runs inside here
        return original(self, *args, **kwargs)

Everything this module enqueues happens on the engine's step thread, synchronously,
BEFORE `original(self, ...)` is entered. `original` is what calls
`backend.launch_copy(...)`, which only PUTS AN ITEM ON A QUEUE
(`copy_backend.py:71-91`); a separate copy thread later dequeues it and does
`copy_blocks(...)` then `event.record(stream)` then `events_list.append(...)`
(`copy_backend.py:124-146`). So the submission order on `load_stream` is fixed by
program order on the step thread:

    our copy  ->  our readback (twice)  ->  the fork's copy  ->  its event

Two consequences, and the module depends on both:

1. THE FORK'S EVENT COVERS OUR WORK. `event.record(stream)` records everything already
   enqueued on that stream, so `finished_recving` cannot be reported until our copy and
   our readbacks have completed. That is the completion witness for our buffers, and it
   is the pool-release discipline this module was asked to respect rather than a
   private event of its own.
2. NOBODY CAN WRITE THE DESTINATION BETWEEN OUR COPY AND OUR READBACK ON THIS STREAM.
   The readback is enqueued after the copy on the same stream, so it observes the window
   as our copy left it - and the fork's copy has not been enqueued yet, so what the
   readback sees is OURS and not the fork's. That is what makes the comparison a witness
   instead of a tautology.

WHAT THIS CANNOT CLAIM, AND SAYS SO: work on ANOTHER stream (the compute stream writing
live KV, or a block handed out again) is unordered with respect to our window. So the
second witness is a readback of the window taken BEFORE our copy, and the first is one
taken AFTER it - a before/after pair, both enqueued inside the wrapper:

    pre-readback -> our copy -> post-readback -> the fork's copy

That pair decides the question that actually has to be decided:

* `post == file` - our write landed where we addressed it. (`readback_equal`)
* `post == pre` - the window is EXACTLY what it was before we touched it, so our copy
  went somewhere else. This is `readback_mismatch_stable`: THE ONE COUNTER THAT MEANS
  OUR ADDRESSING IS WRONG, and the only one an operator acts on.
* `post != pre` and `post != file`, and the window is not "the file's bytes up to some
  offset and the old bytes from there on" - then the window changed, and not to what we
  wrote. Somebody else wrote it: `readback_mismatch_gpu_changed_on_recheck`, an expected
  race (the compute stream, or a block handed out again) that no action follows from.
* `post != file`, but the window IS "the file's bytes up to offset k and the old bytes
  from k on" - our copy changed the window and changed it PART of the way, which is a
  width bug in OUR copy. It is counted with the stable mismatches, because it is the
  same finding (our copy, not the world), and the sample's `shape` field says which.

Two readbacks AFTER the copy - the obvious design, and this module's first draft -
cannot make that last distinction: a third writer that lands before both of them leaves
the two agreeing, and the module would blame its own addressing for somebody else's
write. A late re-read (after the load event) is worse than useless, because the fork's
own copy lands
the expected bytes into the same window in the meantime - every re-read would find the
file and certify a write that had missed. Both witnesses have to be inside the window of
competence, and one of them has to be before the write.

WHY THE WRITE IS SAFE EVEN THOUGH IT TOUCHES LIVE KV
----------------------------------------------------
`start_load_kv` runs before the step's forward pass reads the window, and a GPU block
cannot be freed and re-handed-out before `finished_recving` reports this load complete -
which, by (1), cannot happen before our copy finished. So an in-flight copy of ours can
never overwrite another request's KV. That is the same lifetime the fork's own copy
relies on, borrowed rather than invented.

THE SOURCE OF THE BYTES (AND A DISAGREEMENT WITH THE BRIEF)
-----------------------------------------------------------
The brief described "a pinned bounce buffer owned by the read pool". The read pool's
buffers are NOT pinned and NOT tensors: `disk_read._take_buffer` hands out a plain
`ctypes.c_char * size` array (pageable heap), and a `Tensor.copy_` cannot read one. What
IS pinned, and IS a tensor, at the apply point is the ARENA SLOT: `_apply_buffer` has
just memmoved the file's bytes into `worker.cpu_kv_caches[name][cpu_block_id]`, and the
arena was built with `pin_tensor` (`worker.py:232`). So the source here is the CPU slot
- the identical bytes, from the identical tensor, that the fork's own copy is about to
read. That is a strictly better source than the bounce buffer (same content, genuinely
pinned, and already protected by the tier's own refcounts), and it is what this module
uses. The pinned-buffer-lifetime rule is then honoured for the buffer this module DOES
own: its readback destination rows, which are never returned to the pool until the load
event has passed.

PINNED-BUFFER LIFETIME, AND WHY AN AGE-OUT LEAKS INSTEAD OF RECYCLING
---------------------------------------------------------------------
A row handed to a `copy_(non_blocking=True)` that has not completed is still being
written by the DMA. Reusing it then gives the next entry garbage bytes and NO error -
the one silent-corruption shape this module could create by itself. So a row is
released only when `worker._load_hwm >= event` (the fork's own watermark, the same
witness `disk_restore_verify` uses). If an entry ages out without its event ever
passing, its rows are LEAKED (`buffers_leaked`), not recycled: the pool then drains and
the module stops writing (`no_pinned_buffer` forever after). For a diagnostic that
writes into live KV, permanently-off is the correct failure mode; reusing a row whose
DMA may still be in flight is not.

WHY THERE IS NO `import torch` HERE
-----------------------------------
`tests/test_no_heavy_imports_in_the_read_side` walks every module in this package and
refuses an import of torch/vllm/cuda/numpy/triton. Its stated purpose is that the read
side never PULLS torch into a process that has not already imported it (the EngineCore,
the API server, and this repo's own tooling import these modules on paths that must stay
light). This module needs three torch facilities and obtains all three WITHOUT importing
torch: the stream context manager (`torch.cuda.stream`) comes from `sys.modules`, the
same house idiom `backport.py` uses for the fork's own modules - a dict lookup that
pulls in NOTHING and returns `None` in a torch-free process, where this module refuses
with `stream_unavailable`. A pinned host row is allocated with a TENSOR METHOD,
`region.new_empty((1, elems), device="cpu", pin_memory=True)` - `pin_memory` is part of
`new_empty`'s signature (`torch/_refs/__init__.py:4984`), so no torch symbol is needed
for it. And the copies are `Tensor.narrow(...).copy_(..., non_blocking=True)`, which is
pure method calls. `disk_shadow` refused `importlib.import_module("torch")` for exactly
this reason and its reasoning stands; what this module does instead is the third
possibility, the one that cannot load anything: ask the process what it already has.

EVERY PATH FAILS OPEN. The step-thread entry point is wrapped so that it cannot raise
into a restore, the thread's passes are wrapped so that nothing kills the thread, and
every refusal is a NAME. A refusal means the fork's copy runs alone, exactly as it does
when `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE` is unset - which is the default, and the
posture this whole module is built under.

THE COLD START: THE THREAD MUST EXIST BEFORE THE MODULE ASKS FOR A ROW (0.2.44)
-------------------------------------------------------------------------------
A write needs pinned readback rows. Rows are allocated only by `_refill_pools`, which
runs only on this module's thread. In 0.2.43 that thread was started from ONE place -
inside `_write_one`, on the line AFTER `_take_rows` had already returned rows - so on a
cold boot the empty pool refused the write, the refusal returned, and the start-up call
was never reached. No thread, so no rows; no rows, so no thread: THE FIRST WRITE WAS
UNREACHABLE, AND SO WAS EVERY LATER ONE. Four hundred armed applies in-process reproduce
the live rank file exactly (`writes=0 passes=0 pool_refills=0 buffers_allocated=0
no_pinned_buffer=7 errors=0`) with `thread_started=0`, which is the field that
proves the thread never ran rather than merely found nothing.

THE FIX IS BOTH ENDS OF THAT LOOP. `register_contribution()` starts the thread when the
module is armed, so it exists before the first restore, and `_write_one` asks for it at
the TOP of the attempt, before it takes a row, so even a module that was never
registered starts it on its first refusal. Neither start allocates: the refill walks
`_resolve_worker()`, and the worker weakref is only ever set by the apply point, so a
thread born at install finds no worker, allocates nothing and counts nothing until the
first restore registers one. The thread still never runs on the step thread's behalf -
the cold pool is REFUSED, and the first write waits for the thread's next pass.

WHAT AN OPERATOR READS. `liveness.state` in the rank section says which state this is
(`off` / `armed, no apply point seen in this process` / `armed but never wrote` /
`writing`) and `liveness.verdict` spells the dead one out with the numbers in it. The
refusal name also stopped being one word for two facts: `no_pinned_buffer` now has a
companion counter for its cause, so `pool_never_filled` (the refill has never produced a
row - warm up and try the next apply) is no longer spelled the same way as a pool that
filled and then drained (rows held by records whose load event has not passed, or
leaked), and the in-flight cap was already separable through `in_flight_dropped`.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import threading
import time
import weakref

from . import (
    config,
    disk_restore_verify,
    disk_shadow,
    disk_spill,
    snapshot,
    worker_diag,
)

logger = logging.getLogger("kvtier_dev_plugin.disk_gpu_write")

MODULE = "disk_gpu_write"
OWNER_TAG = "kvtier-dev-plugin/disk-gpu-write-1"

# --- THE BOUNDS ----------------------------------------------------------------
# One applied entry in this many gets written and read back. The population is the
# entries the apply point APPLIED (not offered, not bound), so it is the same
# denominator `disk_restore_verify` uses and the two witnesses can be read side by
# side. 64 rather than verify's 32 because this one WRITES: at 64, a 10,000-restore
# boot writes ~156 windows, which catches a systematic one-slot shift inside the first
# few hundred restores while keeping the writes rare enough that a wrong addressing
# guess cannot flood the window with them.
SAMPLE_EVERY_N_APPLIED = 64

# The step thread's bound, per apply-point call. One write per step is the whole
# posture: this module's copy is on the SAME stream the fork's copy is about to use, so
# a wide batch must not put forty extra submissions ahead of it.
MAX_WRITES_PER_STEP = 1

# The write floor. Retention is per step; writing is not. A batch of forty-five
# restored entries must cost ONE write, not forty-five sampled ones, and the floor
# counts rather than sleeping.
MIN_INTERVAL_SECONDS = 0.5

# The thread's tick and its per-pass bound. At these two numbers the ceiling is eight
# window-wide host reads and eight bounded file reads per second, on a box that is
# already serving traffic.
PASS_TICK_SECONDS = 0.25
MAX_CHECKS_PER_PASS = 2

# How long a record may wait for its load event. Past this the DMA is not coming (the
# fork's own restore deadline is `disk_restore.RESTORE_DEADLINE_SECONDS`, well under
# it), and see the module docstring for why the wait's rows are then LEAKED.
JOB_MAX_AGE_SECONDS = 30.0

# Records in flight. Bounded by the pinned pool as much as by this: with
# MAX_PINNED_ROWS_PER_REGION rows per region and one pre-copy and one post-copy row per
# record, the pool itself admits about one record at a time, and this cap is the belt.
MAX_IN_FLIGHT = 4

# The pinned readback pool, PER REGION NAME AND WIDTH. Two rows is the minimum that lets
# one record read its window BEFORE the copy and again AFTER it, and it is the number
# that prices the pinned host memory this feature costs: at the live box's 11,468,800 B
# window, two rows is ~21.9 MiB of page-locked memory, allocated ONLY while the flag is
# on and ONLY on the daemon thread. Raising it lets more records overlap and costs that
# much more per step of the raise.
MAX_PINNED_ROWS_PER_REGION = 2
MAX_READBACK_LANES = ("pre", "post")   # the before/after pair; the ORDER is the design
MAX_PINNED_ROW_BYTES = 64 * 1024 * 1024     # one row's absurd-geometry refusal
MAX_PINNED_BYTES_TOTAL = 96 * 1024 * 1024   # the whole pool's ceiling

# The apply point's scan bound, on the population rather than the batch (the same belt
# `disk_restore_verify` runs): a future, larger batch must not put an unbounded loop on
# the step thread.
MAX_ENTRIES_SCANNED = 64
MAX_REGIONS = 16                            # mirrors disk_spill.MAX_REGIONS
MAX_LEDGER_SCAN = 64                        # the `_load_events` witness's bound

MAX_REASON_SAMPLES = 6
MAX_REASON_CHARS = 120
MAX_DIFF_SCAN_BYTES = 262_144               # chunked memcmp, `disk_shadow`'s rule

MAX_MISMATCH_SAMPLES = 1
TRIM_REASON_CHARS = 42
# The section's own budget. THE ARITHMETIC OF THE SIXTH SECTION, stated where it cannot
# be missed: `worker_diag.MAX_CONTRIBUTION_BYTES` is 4096 and
# `MAX_CONTRIBUTIONS` was 5 with an allow-list of exactly five names, so a sixth
# section had to RAISE the cap (to 6) rather than be silently refused. Six sections can
# then ask for 6 x 4096 = 24 KiB of `worker_diag.MAX_FILE_BYTES` = 32 KiB, which still
# leaves 8 KiB for the file's own required fields, and contributions remain the first
# thing `_clip_document` sheds. This number keeps ~690 B of margin under the seam's own
# per-section ceiling, which the trim ladder protects when fat reasons arrive.
MAX_SECTION_BYTES = 3400

# The failure names that mean "the read had not settled when the apply point ran". They
# arrive as `failed[job_id]` values from `disk_restore._worker_before_load`, and they
# are the reason `read_not_settled` is countable at all: an entry whose bytes never
# existed is not a candidate for a write, and the name says why.
UNSETTLED_KINDS = ("not_settled", "not_tracked", "incomplete", "deadline_expired")

# THE COUNTER KEYS. A stable set, always all present, for the reason every sibling
# module gives: an operator diffing two rank files must be able to tell "this never
# happened" from "this build does not report it".
_COUNTER_KEYS = (
    # --- the apply point (the step thread): decisions, ints, and nothing else
    "batches_seen",
    "applies_seen",             # the sample's denominator, counted before every gate
    "unsettled_entries",
    "read_not_settled",
    "sampled_out",
    "throttled_step",
    "throttled_interval",
    # --- the ordering witness, the two counters that make the claim checkable
    "readback_enqueued_before_original",
    "readback_enqueued_after_original",
    "event_already_submitted",
    "no_event_ledger",
    # --- the writes
    "writes",
    "bytes_written",
    "regions_written",
    "readbacks_enqueued",
    # --- the refusals, each of which writes NOTHING
    "width_mismatch",
    "row_too_wide",
    "no_gpu_caches",
    "no_arena_caches",
    "no_gpu_block",
    "no_cpu_block",
    "no_group_width",
    "geometry_unavailable",
    "region_pair_missing",
    "too_many_regions",
    "no_pinned_buffer",
    # THE CAUSE OF A `no_pinned_buffer`, counted alongside it and never instead of it.
    # Three refusals used to share the one name (see the module docstring). This one
    # separates "no pinned row has EVER been allocated on this module's thread" - the
    # cold start, which resolves itself on the thread's next pass - from a pool that
    # filled and then drained, which does not. The third, the in-flight cap, already has
    # its own companion in `in_flight_dropped`.
    "pool_never_filled",
    "stream_unavailable",
    "refused_thread_unavailable",
    "write_failed",
    # --- the thread
    "thread_started",
    "passes",
    "worker_gone",
    "dropped_waiting_dma",
    "no_load_hwm",
    "in_flight_dropped",
    # --- the file side (the same four assertions the sibling verifier makes)
    "file_missing",
    "file_not_window_wide",
    "group_wider_than_file",
    "file_short",
    # --- the readback side and the verdicts
    "readback_unreadable",
    "readback_failed",
    "checked",
    "readback_equal",
    "readback_mismatch",
    "readback_mismatch_stable",
    "readback_mismatch_gpu_changed_on_recheck",
    "readback_mismatch_unrechecked",
    "bytes_read_back",
    # --- the pinned pool's own life
    "pool_refills",
    "buffers_allocated",
    "buffers_leaked",
    "readback_dropped",
    # --- the module
    "errors",
    "section_trimmed",
)

# The kinds the FILE side and the READBACK side report, so `section()` and a refusal
# cannot drift apart about what a named failure means.
_FILE_KINDS = ("file_missing", "file_not_window_wide", "group_wider_than_file",
               "file_short")

ORDER_RULE = ("the copy and both readbacks are enqueued inside start_load_kv's wrapper,"
              " before original(); the fork's copy is submitted after, on the same "
              "stream, and keeps the last word")
MEANING_RULE = ("readback_mismatch_stable = OUR ADDRESSING is the candidate; "
                "..._gpu_changed_on_recheck = another stream wrote the window "
                "(an expected race); never act on the second one")


def _fresh_state() -> dict:
    """A brand-new `_state`, so the reset list and the declaration cannot drift."""
    return {
        "latch": False,          # the POST op's decision; the environment is the floor
        "latch_set": False,
        "env_resolved": False,   # the environment is read ONCE, see `_resolve_env`
        "env_enabled": False,
        "env_raw": None,
        "env_recognised": True,
        "seq": 0,                # record ids
        "in_flight": {},         # {id: record}: rows held, awaiting the load watermark
        "pool": {},              # {(name, elems): [free pinned rows]}
        "pinned_bytes": 0,
        "pinned_peak": 0,
        "sample_cursor": 0,
        "last_write_mono": None,
        "last_pass_mono": None,
        "checks_used": 0,
        "worker": None,          # a weakref, see `_resolve_worker`
        "group_bytes": None,     # the per-group widths the comparison used
        "group_bytes_note": None,
        "window_bytes": None,
        "window_note": None,
        "last_verdict": None,
        "mismatch_samples": [],
        "notes": {}, "reasons": [], "reason_seen": set(), "last_trim": None,
        "counters": {key: 0 for key in _COUNTER_KEYS},
    }


_state: dict = _fresh_state()
_lock = threading.Lock()

# The thread and its stop flag are MODULE GLOBALS, not `_state` entries: `_state` is
# what `conftest.py` snapshots and restores around every test, and a thread reference
# restored into the next test is a thread the next test cannot stop.
_thread = None
_thread_lock = threading.Lock()
_stop = None


# --- THE HOUSEKEEPING ----------------------------------------------------------
def _count(name: str, amount: int = 1) -> None:
    # `.get(name, 0)` rather than a subscript: the pass runs on this module's thread
    # while `conftest.py` may be clearing `_state` underneath it, and a counter that
    # cannot be written must never be the thing that raises.
    _state["counters"][name] = int(_state["counters"].get(name, 0)) + int(amount)


def _note_once(key: str, message: str, level: str = "warning") -> bool:
    """One log line per episode, never one per pass - the `disk_shadow` idiom."""
    if _state["notes"].get(key):
        return False
    _state["notes"][key] = True
    logger.log(getattr(logging, level.upper()), "kvtier-dev-plugin: %s", message)
    return True


def _reason(tag: str, detail: str) -> None:
    """Record WHY something was refused, once per distinct tag, and log it once.

    A persistent condition (a worker with no `_load_events`, a pool that filled with
    leaked rows) must be legible in the rank file as a SENTENCE. The integers say how
    often; only the text says what, and for a module whose normal answer is "refused"
    the text IS the product.
    """
    if tag in _state["reason_seen"]:
        return
    _state["reason_seen"].add(tag)
    text = str(detail)
    if len(text) > MAX_REASON_CHARS:
        text = text[:MAX_REASON_CHARS - 3] + "..."
    _state["reasons"].append(f"{tag}: {text}")
    del _state["reasons"][MAX_REASON_SAMPLES:]
    _note_once(f"reason:{tag}", f"[disk_gpu_write] {tag}: {text}")


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --- ENABLEMENT ----------------------------------------------------------------
def _resolve_env() -> dict:
    """Read `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE` ONCE per process, and say so.

    The grammar is `config.disk_gpu_direct_write_enable()` - the same DEFAULT-OFF shape
    as `disk_shadow_enable()`, with the direction a typo moves in being the one that
    writes nothing. The read is cached because a live boot's workers inherit an
    environment that does not change, and re-reading it per apply point would put an
    `os.environ` lookup on the engine's step thread for no information.
    """
    if not _state["env_resolved"]:
        read = config.disk_gpu_direct_write_enable()
        _state["env_raw"] = read["raw"]
        _state["env_enabled"] = bool(read["enabled"])
        _state["env_recognised"] = bool(read["recognised"])
        _state["env_resolved"] = True
        if read["enabled"]:
            _note_once("env-on",
                       f"[disk_gpu_write] ENABLED by "
                       f"{config.ENV_DISK_GPU_DIRECT_WRITE}={read['raw']!r}: one "
                       f"applied restore entry in {SAMPLE_EVERY_N_APPLIED} has its "
                       "window written host->device on the tier's own load stream and "
                       "read back twice, before the fork's copy is submitted; at most "
                       f"{MAX_WRITES_PER_STEP} per step and "
                       f"{MAX_CHECKS_PER_PASS} comparisons per pass",
                       level="info")
        elif read["raw"] is not None and not read["recognised"]:
            _reason("env_value_unrecognised",
                    f"{config.ENV_DISK_GPU_DIRECT_WRITE}={read['raw']!r} is neither an "
                    "on word nor an off word; the direct write is OFF, the direction a "
                    "typo never overrides")
    return {"enabled": bool(_state["env_enabled"]),
            "source": ("environment (read once at first use)"
                       if _state["env_raw"] is not None else
                       "default (off); the environment is read once at first use"),
            "raw": _state["env_raw"],
            "recognised": bool(_state["env_recognised"]),
            "env_var": config.ENV_DISK_GPU_DIRECT_WRITE}


def enable_state() -> dict:
    """One stable-key view of the enablement, including WHO decided it."""
    env = _resolve_env()
    return {"enabled": bool(env["enabled"] or _state["latch"]),
            "env_enabled": bool(env["enabled"]),
            "latch": bool(_state["latch"]),
            "latch_set": bool(_state["latch_set"]),
            "source": ("post op" if _state["latch"] else env["source"]),
            "env_var": config.ENV_DISK_GPU_DIRECT_WRITE, "env_raw": env["raw"],
            "env_recognised": bool(env["recognised"])}


def effective_enabled() -> bool:
    """A write happens only if the environment says yes or the POST latch does."""
    if _state["latch"]:
        return True
    return bool(_resolve_env()["enabled"])


def set_enabled(on: bool) -> dict:
    """The POST op's decision: IN-PROCESS ONLY, and said out loud in the answer.

    No control file is written - this module opens no file handle for writing anywhere -
    so a POST that arrives at the EngineCore latches the EngineCore, which is a process
    that runs no worker and can therefore never write a KV window. That is the same
    scope limit `disk_restore_verify.set_enabled` carries, and it is stated in `scope`
    rather than left for an operator to infer from a counter that never moves. The
    worker that applies a restore is armed with
    `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE=1` in the engine's environment.
    """
    with _lock:
        _state["latch"] = bool(on)
        _state["latch_set"] = True
        answer = enable_state()
    answer["scope"] = ("this process only, and a worker is the ONLY process that can "
                       "write a KV window, so arming the EngineCore writes nothing; "
                       f"the worker is armed with "
                       f"{config.ENV_DISK_GPU_DIRECT_WRITE}=1")
    if on:
        _note_once("latch-on",
                   "[disk_gpu_write] enabled in this process by the POST op "
                   "(scope: the answering process; only a WORKER applies a restore, "
                   "so arm the worker with the environment variable)",
                   level="info")
    return answer


# --- THE THREE FACILITIES THAT NEED TORCH, NONE OF WHICH IMPORT IT -------------
# See the module docstring: `sys.modules` is a lookup, not an import, and it pulls in
# nothing in a process that never imported torch - where the answer is None and every
# path below becomes a named refusal.
def _torch_module():
    """The `torch` module if THIS PROCESS already has it, else None. Never imports."""
    module = sys.modules.get("torch")
    if module is None:
        return None
    return module if getattr(module, "cuda", None) is not None else None


def _stream_context(stream):
    """THE SEAM for "make this stream current". Returns a context manager or None.

    `torch.cuda.stream(s)` is the sanctioned mechanism in torch 2.10 (`Stream` itself is
    not a context manager in that version - `torch/cuda/__init__.py:651-719`), so
    this is
    the one expression that resolves it. A test replaces this function; production runs
    this one, and a test in the suite pins THIS function against a stub `torch` so the
    resolution is exercised and not just the replacement.
    """
    if stream is None:
        return None
    if hasattr(stream, "__enter__") and hasattr(stream, "__exit__"):
        # A fake that IS a context manager (and a future torch that makes Stream one).
        return stream
    module = _torch_module()
    if module is None:
        return None
    factory = getattr(module.cuda, "stream", None)
    if not callable(factory):
        return None
    return factory(stream)


def _allocate_row(region, elems: int):
    """THE SEAM for "a pinned host row the shape of one region row". Returns a tensor.

    `Tensor.new_empty(..., device="cpu", pin_memory=True)` is a METHOD, so this costs no
    torch symbol and no `cudaHostAlloc` through ctypes. It is also the only allocation
    this module makes, and it runs on the DAEMON THREAD ONLY: page-locking memory is
    slow and serialises the device, and the step thread must never pay for it (which is
    what `no_pinned_buffer` means: the pool has not been filled yet, or is empty). The
    cold half of that sentence is its own counter, `pool_never_filled`, so "the witness
    thread has not warmed up" and "the device has run out of rows" are not the same
    reading of one number.
    """
    return region.new_empty((1, int(elems)), device="cpu", pin_memory=True)


def _copy_host_to_device(dst_region, dst_block: int, src_region,
                         src_block: int) -> None:
    """THE ONLY EXPRESSION IN THIS MODULE THAT WRITES INTO LIVE KV MEMORY.

    One region, one block, host->device, non-blocking, on whatever stream is current.
    It mirrors the fork's own per-region pairing (`cuda_mem_ops.build_params` pairs
    region i of `cpu_kv_caches` with region i of `gpu_kv_caches` and asserts their row
    widths agree); this module pairs them BY NAME, which is the same thing while both
    dicts share an insertion order and a well-defined thing when they do not.
    """
    dst_region.narrow(0, int(dst_block), 1).copy_(
        src_region.narrow(0, int(src_block), 1), non_blocking=True)


def _copy_device_to_host(row, src_region, src_block: int) -> None:
    """One region, one block, device->host, non-blocking, on the current stream.

    The destination is one of this module's pinned rows, so the copy is asynchronous by
    construction; a pageable destination would have made it synchronous ON THE STEP
    THREAD, which is the single most expensive mistake available here.
    """
    row.copy_(src_region.narrow(0, int(src_block), 1), non_blocking=True)


# --- THE GEOMETRY AND THE PLAN (step thread: attribute walks and ints) ---------
def _refresh_geometry(worker) -> bool:
    """The widths the comparison needs, read on the step thread, cached. THREE READS,
    all attribute walks, no CUDA call and no file: the per-group widths (the sibling
    verifier's own `_group_widths`, so `W_g` here and `W_g` in the rank file are the
    same
    expression), and this rank's slot width (`disk_spill.measured_slot_bytes`, the same
    measurement a fingerprint's `total_bytes_per_block` is built from).
    """
    sizes, note = disk_restore_verify._group_widths(worker)
    arena = disk_spill.measured_slot_bytes(getattr(worker, "cpu_kv_caches", None))
    with _lock:
        _state["group_bytes"] = list(sizes) if sizes else None
        _state["group_bytes_note"] = note
        _state["window_bytes"] = (int(arena["bytes_per_file"]) if arena.get("ok")
                                  else None)
        _state["window_note"] = (None if arena.get("ok")
                                 else str(arena.get("reason")))
        have = bool(sizes)
        if not have and not _state["counters"].get("geometry_unavailable"):
            _count("geometry_unavailable")
    if not have:
        _reason("group_widths_unavailable",
                note or "the per-group widths are unreadable")
    return have


def _group_width(group) -> int | None:
    with _lock:
        sizes = _state.get("group_bytes")
    if not isinstance(sizes, list) or group is None \
            or not (0 <= int(group) < len(sizes)):
        return None
    return _int_or_none(sizes[int(group)])


def _load_event_state(worker, event) -> str:
    """THE ORDERING WITNESS: has the fork submitted the copy for this load event yet?

    `copy_backend._copy_loop` appends `(event_idx, event)` to `worker._load_events` only
    AFTER `copy_blocks(...)` and `event.record(stream)` (`copy_backend.py:138-146`), and
    the queue item that triggers it is put by `launch_copy` INSIDE the fork's
    `start_load_kv`. So "this event index is not in that list" is a fact about the world
    that can only be true while we are before the fork's copy - which is the premise of
    the whole ordering argument, checked rather than asserted.

    IT IS A PREMISE WITNESS WITH ONE QUEUE-HOP OF SLACK: if a future refactor moved this
    hook after `original()` but the copy thread had not yet drained the queue, the event
    would still be absent. That is exactly why the counters are a PAIR: a moved hook
    makes `event_already_submitted` and `readback_enqueued_after_original` start
    moving, so the regression is loud instead of silent.
    """
    if event is None or int(event) < 0:
        return "no_event"
    ledger = getattr(worker, "_load_events", None)
    if ledger is None:
        return "no_ledger"
    try:
        entries = list(ledger)[:MAX_LEDGER_SCAN]
    except Exception:  # noqa: BLE001 - an unreadable ledger is not evidence of anything
        return "no_ledger"
    for entry in entries:
        try:
            if int(entry[0]) == int(event):
                return "submitted"
        except (TypeError, ValueError, KeyError, IndexError):
            return "no_ledger"
    return "unsubmitted"


def _plan(worker, job: dict, width: int) -> dict:
    """The per-region pairing for one entry, or a named refusal.

    TWO RULES, both copied from code that is already load-bearing:

    1. `sorted(name)`, the order `disk_spill.read_slot_bytes` documents as the FILE's
       byte order and the order `_plan_regions` memmoves the file into the CPU slot in.
       Pairing GPU region `name` with CPU region `name` (rather than positionally) is
       what makes a region's GPU row receive THAT region's bytes even when a dict's
       insertion order is not its sorted order - the failure a test pins.
    2. EVERY GPU ROW WIDTH MUST EQUAL ITS CPU ROW WIDTH, and the widths must total the
       entry's own `bytes` EXACTLY. That is `build_params`' assertion
       (`cuda_mem_ops.py:206-207`) and `_plan_regions`' total check, and a plan that
       totals anything else would write half a window and call it a restore.
    """
    gpu_caches = getattr(worker, "gpu_kv_caches", None)
    if not isinstance(gpu_caches, dict) or not gpu_caches:
        return {"ok": False, "kind": "no_gpu_caches",
                "reason": "the worker exposes no gpu_kv_caches"}
    cpu_caches = getattr(worker, "cpu_kv_caches", None)
    if not isinstance(cpu_caches, dict) or not cpu_caches:
        return {"ok": False, "kind": "no_arena_caches",
                "reason": "the worker exposes no cpu_kv_caches to copy FROM"}
    names = sorted(str(name) for name in gpu_caches)
    if len(names) > MAX_REGIONS:
        return {"ok": False, "kind": "too_many_regions",
                "reason": f"{len(names)} GPU regions exceed this module's "
                          f"{MAX_REGIONS} cap"}
    gpu_block = _int_or_none(job.get("gpu_block_id"))
    if gpu_block is None or gpu_block < 0:
        return {"ok": False, "kind": "no_gpu_block",
                "reason": "the restore entry names no usable gpu_block_id"}
    cpu_block = _int_or_none(job.get("cpu_block_id"))
    if cpu_block is None or cpu_block < 0:
        return {"ok": False, "kind": "no_cpu_block",
                "reason": "the restore entry names no usable cpu_block_id"}
    regions: list = []
    total = 0
    for name in names:
        gpu_region = gpu_caches.get(name)
        cpu_region = cpu_caches.get(name)
        if cpu_region is None:
            return {"ok": False, "kind": "region_pair_missing",
                    "reason": f"GPU region {name!r} has no CPU region of the same "
                              "name, and pairing them positionally is how one "
                              "region's bytes land in another's window"}
        try:
            row_elems = int(gpu_region.stride(0))
            itemsize = int(gpu_region.element_size())
            cpu_elems = int(cpu_region.stride(0))
            cpu_itemsize = int(cpu_region.element_size())
            gpu_blocks = int(tuple(getattr(gpu_region, "shape", ()) or ())[0])
        except Exception as exc:  # noqa: BLE001 - a hostile tensor is a refusal
            snapshot._error(exc, "disk_gpu_write region geometry")
            return {"ok": False, "kind": "width_mismatch",
                    "reason": f"region {name!r}: reading its geometry raised "
                              f"{type(exc).__name__}"}
        bytes_per_row = row_elems * itemsize
        if row_elems <= 0 or itemsize <= 0 or cpu_elems <= 0 or cpu_itemsize <= 0 \
                or bytes_per_row != cpu_elems * cpu_itemsize:
            return {"ok": False, "kind": "width_mismatch",
                    "reason": f"region {name!r}: its GPU row is {bytes_per_row} B "
                              f"({row_elems}x{itemsize}) and its CPU row is "
                              f"{cpu_elems * cpu_itemsize} B; the fork's own copy "
                              "asserts these agree, and a copy between two widths is "
                              "not a witness of anything"}
        if bytes_per_row > MAX_PINNED_ROW_BYTES:
            return {"ok": False, "kind": "row_too_wide",
                    "reason": f"region {name!r}'s row is {bytes_per_row} B, past the "
                              f"{MAX_PINNED_ROW_BYTES} B a pinned readback row may be"}
        if not 0 <= gpu_block < gpu_blocks:
            return {"ok": False, "kind": "no_gpu_block",
                    "reason": f"gpu_block {gpu_block} is outside region {name!r}'s "
                              f"{gpu_blocks} blocks"}
        regions.append({"name": name, "gpu": gpu_region, "cpu": cpu_region,
                        "elems": row_elems, "bytes": bytes_per_row})
        total += bytes_per_row
    if total != int(width):
        return {"ok": False, "kind": "width_mismatch",
                "reason": f"the regions total {total} B but this entry's file is "
                          f"{width} B: not these bytes, never written"}
    return {"ok": True, "regions": regions, "total": total,
            "gpu_block": gpu_block, "cpu_block": cpu_block}


# --- THE PINNED POOL (allocated ONLY on the daemon thread) ---------------------
def _pool_key(name: str, elems: int) -> tuple:
    return (str(name), int(elems))


def _refill_pools(worker) -> int:
    """Top every pool up, on THIS thread. Returns the number of rows allocated.

    The step thread never allocates: `pin_memory=True` means a `cudaHostAlloc`, which is
    slow and serialises the device, and the whole cost model of this module is that the
    step thread pays a dict lookup for a buffer. The price of that choice is that the
    first apply point after a boot finds an empty pool and refuses - counted, published
    and correct, and since 0.2.44 named as the COLD case by `pool_never_filled` rather
    than left to read like a shortage. What must never happen is the reverse trade: this
    function runs ONLY on this module's thread, and the thread must therefore exist
    before any write is attempted, which is why `_ensure_thread` is called at the top of
    `_write_one` and at registration (`register_contribution`) rather than after a
    successful `_take_rows` - the 0.2.43 ordering that made the first write unreachable.
    A worker that is not registered yet (`_resolve_worker()` answering None, which is
    every thread born at install) allocates NOTHING and returns 0.
    """
    allocated = 0
    if worker is None:
        return 0
    caches = getattr(worker, "gpu_kv_caches", None)
    if not isinstance(caches, dict):
        return 0
    for name in sorted(str(key) for key in caches):
        region = caches.get(name)
        try:
            elems = int(region.stride(0))
            itemsize = int(region.element_size())
        except Exception as exc:  # noqa: BLE001 - an unreadable region is not poolable
            # Counted once per episode, not once per tick: this loop runs on the
            # module's thread every `PASS_TICK_SECONDS`, and a worker whose regions
            # cannot be inspected is a fact worth one line and not a wall of them.
            _reason(f"pool_region_unreadable:{type(exc).__name__}",
                    "a GPU region's stride/element_size could not be read, so no "
                    "pinned row can be allocated for it and its writes are refused")
            continue
        if elems <= 0 or elems * itemsize > MAX_PINNED_ROW_BYTES:
            continue
        while True:
            with _lock:
                free = _state["pool"].setdefault(_pool_key(name, elems), [])
                have_free = len(free)
                bytes_held = int(_state["pinned_bytes"])
                if have_free >= MAX_PINNED_ROWS_PER_REGION \
                        or bytes_held + elems * itemsize > MAX_PINNED_BYTES_TOTAL:
                    break
            try:
                row = _allocate_row(region, elems)
            except Exception as exc:  # noqa: BLE001 - no pinned memory is a refusal
                snapshot._error(exc, "disk_gpu_write pinned row allocation")
                _reason("pinned_allocation_failed",
                        f"allocating a pinned {elems * itemsize} B readback row for "
                        f"region {name!r} raised {type(exc).__name__}: the pool stays "
                        "empty and every write is refused with no_pinned_buffer")
                return allocated
            nbytes = int(getattr(row, "nbytes", elems * itemsize))
            with _lock:
                _state["pool"][_pool_key(name, elems)].append(row)
                _state["pinned_bytes"] += nbytes
                if _state["pinned_bytes"] > _state["pinned_peak"]:
                    _state["pinned_peak"] = int(_state["pinned_bytes"])
                _count("buffers_allocated")
            allocated += 1
    if allocated:
        with _lock:
            _count("pool_refills")
    return allocated


def _take_rows(plan: dict) -> dict | None:
    """Two pinned rows per region for one record, or None (and nothing is taken).

    Both rows are needed BEFORE anything is enqueued: a readback with no destination
    would have to be skipped, and a write whose witness silently did not happen is the
    decorative-diagnostic failure this package refuses. So an incomplete take gives back
    whatever was collected and the entry is refused whole.
    """
    taken: dict = {}
    with _lock:
        for region in plan["regions"]:
            for lane in MAX_READBACK_LANES:
                free = _state["pool"].get(
                    _pool_key(region["name"], region["elems"])) or []
                if not free:
                    # Give back what we already took, then refuse the entry whole.
                    for held in taken.values():
                        key = _pool_key(held[0], held[1])
                        _state["pool"].setdefault(key, []).append(held[2])
                    return None
                row = free.pop()
                taken[f"{region['name']}|{lane}"] = (region["name"], region["elems"],
                                                     row)
    return taken


def _lane_rows(taken: dict, lane: str) -> dict:
    """One readback lane as a `read_slot_bytes`-shaped map: {region name: row}."""
    rows: dict = {}
    for key, value in taken.items():
        name, _elems, row = value
        if key.endswith("|" + lane):
            rows[name] = row
    return rows


def _release_rows(taken: dict, kind: str | None = None) -> None:
    """Give rows back to the pool. ONLY AFTER THE LOAD EVENT HAS PASSED.

    `kind` names why they are NOT going back: a row whose DMA may still be in flight is
    LEAKED rather than reused, and `buffers_leaked` is the counter an operator reads.
    The row leaves the byte accounting - a leak costs memory ONCE and must not creep
    upward until a long boot could no longer afford to allocate - but it never reappears
    in the free list, and THAT is the safety property: a row whose DMA state is
    unknowable is never a witness twice. The pool refills with FRESH rows on this
    module's thread; if the device stops handing out pinned memory, every write is
    refused with `no_pinned_buffer` rather than performed unwitnessed.
    """
    leaked = kind is not None
    with _lock:
        for name, elems, row in taken.values():
            if leaked:
                _state["pinned_bytes"] = max(0, int(_state["pinned_bytes"]) - int(
                    getattr(row, "nbytes", 0)))
                _count("buffers_leaked")
                continue
            _state["pool"].setdefault(_pool_key(name, elems), []).append(row)
    if leaked:
        _reason(kind, "readback rows were dropped without their load event completing, "
                      "so they were leaked rather than handed to a later entry whose "
                      "DMA could overwrite one already in flight")


# --- THE THREAD ----------------------------------------------------------------
def _ensure_thread() -> bool:
    """Start the module's single thread, once, lazily. Never at import.

    TWO CALLERS, AND THE ORDER IS THE WHOLE POINT (0.2.44): `register_contribution()`
    asks at install, so an armed process has a writer thread before its first restore,
    and `_write_one` asks at the TOP of a write attempt, before it touches the pool, as
    the safety net for a module that was never registered. It must NOT be asked after a
    successful `_take_rows`, which is where 0.2.43 put it and what made the first write
    unreachable: the rows only exist once this thread has run a refill, and the refusal
    that the missing rows cause returned before this line could be reached.

    One thread, bounded, respawned only if the previous one died. Starting it allocates
    nothing: the thread waits `PASS_TICK_SECONDS` first, and its refill walks
    `_resolve_worker()`, which answers None until the apply point registers a worker.
    """
    global _thread, _stop
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return True
        try:
            stop = threading.Event()
            thread = threading.Thread(target=_loop, args=(stop,),
                                      name="kvtier-disk-gpu-write", daemon=True)
            thread.start()
        except Exception:  # noqa: BLE001 - no thread is a refusal, not a fault
            return False
        _stop = stop
        _thread = thread
        with _lock:
            _count("thread_started")
        return True


def _contained(action):
    """Run one pass so that NOTHING it does can kill the thread.

    `conftest.py` clears this module's `_state` IN PLACE while a pass from the previous
    test may still be running, and a thread that walks a half-cleared dictionary raises
    a `KeyError` that has nothing to do with the product. A loop that dies there stops
    verifying silently, the pool stops draining and every later write is refused with
    `no_pinned_buffer` - which reads as "the witness found nothing".
    """
    try:
        return action()
    except Exception as exc:  # noqa: BLE001 - a pass never kills this module's thread
        with _lock:
            _count("errors")
        _reason("pass", f"the pass raised {type(exc).__name__}")
        return None


def _loop(stop) -> None:
    """The tick: wait `PASS_TICK_SECONDS`, refill the pools, then ONE bounded pass."""
    while not stop.wait(PASS_TICK_SECONDS):
        _contained(lambda: _refill_pools(_resolve_worker()))
        _contained(run_pass)


def _resolve_worker(worker=None):
    """The worker to allocate rows and read the watermark from, or None. A weakref."""
    if worker is not None:
        return worker
    with _lock:
        ref = _state["worker"]
    if ref is None:
        return None
    try:
        return ref()
    except Exception:  # noqa: BLE001 - a dead or broken ref is simply no worker
        return None


def _first_difference(left: bytes, right: bytes) -> tuple:
    """`(offset, scanned_bytes)` - `disk_shadow`'s chunked memcmp, one Python loop."""
    limit = min(len(left), len(right), MAX_DIFF_SCAN_BYTES)
    for start in range(0, limit, 4096):
        stop = min(start + 4096, limit)
        if left[start:stop] == right[start:stop]:
            continue
        for offset in range(start, stop):
            if left[offset] != right[offset]:
                return offset, stop
        return None, stop
    if len(left) != len(right):
        return min(len(left), len(right)), limit
    return None, limit


def _read_rows(rows: dict, width: int) -> dict:
    """A readback map's bytes, through the SANCTIONED host reader.

    `disk_spill.read_slot_bytes` is the same function that defines the file's byte
    order,
    so the concatenation order this comparison uses is not a second opinion about the
    order - it is the order. `disk_shadow._host_landed` runs first because what follows
    forms addresses: a copy that did not reach host memory would otherwise be a
    SIGSEGV, which no `except Exception` survives.
    """
    if not rows:
        return {"ok": False, "kind": "readback_unreadable",
                "reason": "the record holds no readback rows"}
    for name, row in rows.items():
        if not disk_shadow._host_landed(row):
            return {"ok": False, "kind": "readback_unreadable",
                    "reason": f"row {name!r} reports device "
                              f"{getattr(row, 'device', None)!r}: reading a device "
                              "pointer as a host address is a segfault, not an "
                              "exception"}
    read = disk_spill.read_slot_bytes(rows, 0, width)
    if not read.get("ok"):
        return {"ok": False, "kind": "readback_unreadable",
                "reason": str(read.get("reason"))}
    return {"ok": True, "bytes": bytes(read["bytes"])}


def _check_record(worker, record: dict) -> dict:
    """One record, on this module's thread. Never raises."""
    verdict = _contained(lambda: _check_record_inner(worker, record))
    if verdict is None:
        # `_contained` already counted `errors` and named the raise. The rows stay held
        # by the record, which is still in flight, so the NEXT pass releases them once
        # the watermark passes - and if this keeps happening the pool drains and the
        # module stops writing, which is the honest direction for a witness that cannot
        # tell the reader what went wrong.
        return {"checked": False, "reason": "error"}
    return verdict


def _check_record_inner(worker, record: dict) -> dict:
    event = record.get("event")
    hwm = _int_or_none(getattr(worker, "_load_hwm", None))
    if hwm is None:
        with _lock:
            _count("no_load_hwm")
        _reason("no_load_hwm",
                "this worker exposes no `_load_hwm`, so nothing here can say whether "
                "the copy that covers our own has completed; records are dropped "
                "unread and their rows leaked rather than compared against a DMA")
        _release_rows(record["rows"], kind="rows_leaked_no_hwm")
        return {"checked": False, "reason": "no_load_hwm"}
    if event is None or hwm < int(event):
        if (time.monotonic() - float(record.get("mono") or 0.0)) > JOB_MAX_AGE_SECONDS:
            with _lock:
                _count("dropped_waiting_dma")
            # LEAKED, not recycled: the event never passed, so the DMA may still be
            # in flight and a later entry would be handed a row being written.
            _release_rows(record["rows"], kind="rows_leaked_waiting_dma")
            return {"checked": False, "reason": "dropped_waiting_dma"}
        return {"checked": False, "reason": "waiting_for_load_event"}
    # THE COMPLETION WITNESS, and it is the FORK'S: `event.record(load_stream)` is
    # recorded after our copy and our readbacks were enqueued on that stream, so a
    # passed watermark means our readbacks landed. No event of our own, no new CUDA
    # primitive, and the release of these rows follows the same rule the tier's own
    # buffer release follows.
    width = int(record["bytes"])
    group_bytes = int(record["group_bytes"])
    file_side = disk_restore_verify._read_file_head(record["path"], width, group_bytes)
    if not file_side["ok"]:
        kind = file_side["kind"] if file_side["kind"] in _COUNTER_KEYS else \
            "file_missing"
        with _lock:
            _count(kind)
        _reason(kind, str(file_side.get("reason")))
        _release_rows(record["rows"])
        return {"checked": False, "reason": kind, "detail": file_side.get("reason")}
    pre_side = _read_rows(_lane_rows(record["rows"], "pre"), width)
    post_side = _read_rows(_lane_rows(record["rows"], "post"), width)
    if not post_side["ok"]:
        with _lock:
            _count("readback_unreadable")
        _reason("readback_unreadable", str(post_side.get("reason")))
        _release_rows(record["rows"])
        return {"checked": False, "reason": "readback_unreadable",
                "detail": post_side.get("reason")}
    with _lock:
        _count("checked")
        _count("bytes_read_back", width * (1 + (1 if pre_side.get("ok") else 0)))
    # ONLY THE LEADING W_g OF EACH SIDE, for the reason `disk_restore_verify` gives at
    # length: the rest of the window belongs to other groups and is stale by
    # construction, so comparing it would report a mismatch on every narrow group
    # forever and drown the one real finding.
    expected = bytes(file_side["head"])[:group_bytes]
    post = bytes(post_side["bytes"])[:group_bytes]
    if post == expected:
        with _lock:
            _count("readback_equal")
            _state["last_verdict"] = {
                "equal": True, "gpu_block": int(record["gpu_block"]),
                "group": record["group"], "event": record["event"],
                "key16": record["key16"], "compared_bytes": group_bytes,
                "tail_bytes_excluded": width - group_bytes,
                "pre_readback": bool(pre_side.get("ok")),
            }
        _release_rows(record["rows"])
        return {"checked": True, "equal": True, "bytes": group_bytes}
    if not pre_side["ok"]:
        # No before-state, so no classification: the disagreement is real, but this
        # module cannot say whether another stream moved the window or its own copy
        # missed, and pretending otherwise is how a race becomes a false bug report.
        with _lock:
            _count("readback_mismatch")
            _count("readback_mismatch_unrechecked")
        _reason("readback_unreadable",
                f"a mismatch arrived with an unreadable pre-copy readback "
                f"({pre_side.get('reason')}), so it is reported unrechecked")
        _release_rows(record["rows"])
        return {"checked": True, "equal": False,
                "category": "readback_mismatch_unrechecked"}
    pre = bytes(pre_side["bytes"])[:group_bytes]
    offset, scanned = _first_difference(post, expected)
    shape = None
    if post == pre:
        # THE WINDOW IS EXACTLY WHAT IT WAS BEFORE WE TOUCHED IT, so our copy changed
        # nothing here and went somewhere else. THE ONE COUNTER THAT MEANS OUR
        # ADDRESSING IS WRONG, and the only one an operator acts on. It stays a
        # HYPOTHESIS rather than a verdict for one stated reason: a third writer that
        # wrote precisely the pre-existing bytes again would look identical, and no
        # witness inside our window of competence can rule that out.
        category = "readback_mismatch_stable"
        shape = "window-unchanged"
    elif offset is not None and post[offset:] == pre[offset:]:
        # "THE FILE'S BYTES UP TO OFFSET k, THE OLD BYTES FROM k ON": our copy DID
        # change this window and changed only its leading part, which is a width or
        # extent bug in OUR copy rather than a writer from elsewhere. Counted with the
        # stable mismatches because it is the same finding - our copy, not the world -
        # and the `shape` field is what keeps the two distinguishable in the sample.
        category = "readback_mismatch_stable"
        shape = "partial-span"
    else:
        # THE WINDOW CHANGED, AND NOT TO WHAT WE WROTE. Both readbacks are on our
        # stream, so the only way the after-state is neither ours nor the old bytes is
        # that work on ANOTHER stream wrote the window - the compute stream, or a block
        # handed out again. An expected race, and the reason this module does not call
        # every disagreement a bug.
        category = "readback_mismatch_gpu_changed_on_recheck"
    sample = {"category": category, "shape": shape,
              "gpu_block": int(record["gpu_block"]),
              "group": record["group"], "event": record["event"],
              "key16": record["key16"], "bytes": width,
              "compared_bytes": group_bytes,
              "tail_bytes_excluded": width - group_bytes,
              "first_difference": offset, "scanned_bytes": int(scanned),
              "window_matched_pre_copy": post == pre}
    with _lock:
        # THE MEANING, IN COUNTERS. `readback_mismatch` is the family and the category
        # counter is the one an operator acts on: `..._stable` names OUR ADDRESSING as
        # the candidate, `..._gpu_changed_on_recheck` names another stream and asks for
        # nothing. Without these the module would publish a classification it never
        # counted, and a reader watching the rank file between two writes would see the
        # sample move while every number stayed at zero.
        _count("readback_mismatch")
        _count(category)
        _state["mismatch_samples"].append(sample)
        del _state["mismatch_samples"][:max(
            0, len(_state["mismatch_samples"]) - MAX_MISMATCH_SAMPLES)]
        _state["last_verdict"] = sample
    _release_rows(record["rows"])
    return {"checked": True, "equal": False, "category": category,
            "first_difference": offset, "scanned_bytes": scanned}


def run_pass(worker=None) -> dict:
    """At most `MAX_CHECKS_PER_PASS` records. One pass; never raises.

    The thread's body, and the function a test drives directly (the sibling modules'
    idiom: monkeypatch `_ensure_thread` and call this).
    """
    try:
        if not effective_enabled():
            return {"ran": False, "reason": "disabled", "checked": 0}
        now = time.monotonic()
        with _lock:
            last = _state["last_pass_mono"]
            if last is not None and (now - float(last)) < MIN_INTERVAL_SECONDS:
                _count("throttled_interval")
                return {"ran": False, "reason": "throttled_interval", "checked": 0}
            _state["last_pass_mono"] = now
            _state["checks_used"] = 0
            batch = [record for record in list(_state["in_flight"].values())
                     [:MAX_CHECKS_PER_PASS]]
            if not batch:
                return {"ran": True, "reason": "nothing in flight", "checked": 0}
            _count("passes")
        target = _resolve_worker(worker)
        if target is None:
            # Every in-flight record goes with it, and its rows are LEAKED: they were
            # taken against a worker whose stream we can no longer ask about, and a row
            # whose DMA state is unknowable must never be handed to anyone again.
            with _lock:
                _count("worker_gone")
                held = list(_state["in_flight"].values())
                _state["in_flight"] = {}
            for record in held:
                _contained(lambda r=record: _release_rows(r["rows"],
                                                          kind="rows_leaked_worker_gone"))
            _reason("worker_gone",
                    f"the worker this module wrote through is gone, so {len(held)} "
                    "records were dropped unread and their pinned rows leaked")
            return {"ran": True, "reason": "worker_gone", "checked": 0,
                    "dropped": len(held)}
        results = []
        waiting = 0
        for record in batch:
            verdict = _check_record(target, record)
            if verdict.get("reason") == "waiting_for_load_event":
                waiting += 1
                continue
            with _lock:
                _state["in_flight"].pop(record["id"], None)
            results.append({"id": record["id"], **verdict})
        return {"ran": True, "reason": None,
                "checked": sum(1 for item in results if item.get("checked")),
                "waiting": waiting, "results": results}
    except Exception as exc:  # noqa: BLE001 - a pass never escapes its thread
        with _lock:
            _count("errors")
        _reason("run_pass", f"the pass raised {type(exc).__name__}")
        return {"ran": False, "reason": f"error_{type(exc).__name__}", "checked": 0}


# --- THE STEP THREAD'S ONLY CALL ----------------------------------------------
def observe_apply(worker, batch, failed) -> dict:
    """The apply point's GPU-direct write. THE ONLY ENTRY POINT ON AN ENGINE THREAD.

    Called by `disk_restore._worker_before_load` beside `disk_restore_verify`, where
    `failed` is final and the entries in `batch["jobs"]` absent from it are exactly the
    ones whose bytes were memmoved into a CPU slot this step. It is inside the
    `start_load_kv` wrapper's call to `worker_before_load(self)`, which the wrapper
    makes on the line BEFORE `return original(self, ...)`, so anything this function
    enqueues on `worker.load_stream` is enqueued before the fork's copy is submitted.

    IT CANNOT CHANGE THE STEP. It reads, counts, plans, takes a buffer and enqueues
    copies; it never prunes, reorders, renames, re-files or awaits anything, and every
    path is inside a `try` that turns a raise into a counter. The fork's copy still runs
    afterwards on the same stream and has the last word on every byte, which is what
    makes an addressing mistake here a WRONG WITNESS rather than a wrong answer.
    """
    try:
        if not effective_enabled():
            return {"seen": 0, "written": 0, "reason": "disabled"}
        jobs = (batch or {}).get("jobs") or {}
        if not jobs:
            return {"seen": 0, "written": 0, "reason": "no entries"}
        event = _int_or_none((batch or {}).get("event"))
        failed_ids = set(failed or {})
        # THE UNSETTLED ONES, counted before anything else: an entry whose read had not
        # settled has no bytes to write, and the name of the reason (`not_settled`,
        # `incomplete`, `deadline_expired`) is the difference between "the disk was
        # slow"
        # and "the batch was short". Counting these is the only way `read_not_settled`
        # means something other than "we were never asked".
        unsettled = 0
        for job_id, kind in (failed or {}).items():
            if str(kind) in UNSETTLED_KINDS and job_id in jobs:
                unsettled += 1
        survivors = [(job_id, job) for job_id, job in jobs.items()
                     if job_id not in failed_ids][:MAX_ENTRIES_SCANNED]
        with _lock:
            _state["worker"] = weakref.ref(worker) if worker is not None else None
            _count("batches_seen")
            if unsettled:
                _count("unsettled_entries", unsettled)
                _count("read_not_settled", unsettled)
        if not survivors:
            return {"seen": 0, "written": 0, "reason": "nothing applied"}
        if not _refresh_geometry(worker):
            return {"seen": len(survivors), "written": 0,
                    "reason": "group widths unreadable"}
        ledger = _load_event_state(worker, event)
        if ledger == "no_ledger":
            with _lock:
                _count("no_event_ledger", len(survivors))
            _reason("no_event_ledger",
                    "this worker exposes no `_load_events`, so the ordering premise "
                    "(the fork has not submitted this load's copy yet) cannot be "
                    "checked and nothing is written")
            return {"seen": len(survivors), "written": 0, "reason": "no_event_ledger"}
        if ledger != "unsubmitted":
            # THE ORDERING CLAIM HAS FAILED. Either the hook moved after the original,
            # or this event's copy is already submitted - and writing now would put our
            # copy AFTER the fork's, which turns our readback into a witness of the
            # fork's bytes and our write into a racing overwrite. So: refuse, loudly.
            with _lock:
                _count("event_already_submitted", len(survivors))
                _count("readback_enqueued_after_original", len(survivors))
            _reason("event_already_submitted",
                    f"load event {event} is already recorded on the worker's load "
                    "stream, so this hook is no longer running before the fork's "
                    "copy; nothing is written")
            return {"seen": len(survivors), "written": 0,
                    "reason": "event_already_submitted"}
        written = 0
        seen = 0
        refusal = None
        refusal_cause = None
        for _job_id, job in survivors:
            if not isinstance(job, dict):
                continue
            seen += 1
            # Counted BEFORE the sample decision and every geometry test:
            # `applies_seen` is the denominator of the sample rate, exactly as in the
            # sibling verifier, so "how much of my restore traffic did the writer look
            # at?" is a division an operator can actually perform.
            with _lock:
                _count("applies_seen")
                cursor = int(_state["sample_cursor"])
                _state["sample_cursor"] = cursor + 1
                selected = (cursor % max(1, SAMPLE_EVERY_N_APPLIED)) == 0
            if not selected:
                with _lock:
                    _count("sampled_out")
                continue
            if written >= MAX_WRITES_PER_STEP:
                with _lock:
                    _count("throttled_step")
                continue
            with _lock:
                last = _state["last_write_mono"]
                if last is not None and (time.monotonic() - float(last)) \
                        < MIN_INTERVAL_SECONDS:
                    _count("throttled_interval")
                    continue
            outcome = _write_one(worker, job, event)
            if outcome.get("written"):
                written += 1
                with _lock:
                    _state["last_write_mono"] = time.monotonic()
            elif refusal is None:
                # THE NAME OF THE FIRST REFUSAL. The counters are the published record
                # and stay the thing a reader aggregates, but a caller that asked "what
                # happened to THIS restore step?" should not have to diff counters to
                # find out - and a throttle is deliberately not a refusal, so the sample
                # and the step cap leave this None rather than naming something nothing
                # rejected. `cause` travels with it for the one name that carries more
                # than one fact (`no_pinned_buffer`, whose cold and drained readings are
                # different operational stories); every other refusal's name IS its
                # cause and leaves this None.
                refusal = outcome.get("reason")
                refusal_cause = outcome.get("cause")
        return {"seen": seen, "written": written,
                "reason": (None if written else refusal),
                "cause": refusal_cause}
    except Exception as exc:  # noqa: BLE001 - never into the engine's step path
        with _lock:
            _count("errors")
        _reason("observe_apply", f"the apply-point writer raised {type(exc).__name__}")
        return {"seen": 0, "written": 0, "reason": f"error_{type(exc).__name__}"}


def _refuse(kind: str, detail: str, seen: int = 1, cause: str | None = None) -> dict:
    with _lock:
        _count(kind, seen)
    _reason(kind, detail)
    out = {"written": False, "reason": kind}
    if cause:
        out["cause"] = cause
    return out


# THE TWO SENTENCES FOR THE ONE NAME THAT USED TO CARRY TWO FACTS. `no_pinned_buffer`
# was both "the witness thread has not filled its pool yet" and "the pool is out of
# rows", and an operator reading a rank file cannot act on those the same way: the
# first fixes itself on the thread's next pass, the second means rows are held by
# records whose load event has not passed (or leaked, which is permanent). They are now
# counted apart (`pool_never_filled` rides along with the cold case) and said apart.
COLD_POOL_REASON = (
    "no pinned readback row exists because NONE HAS EVER BEEN ALLOCATED: this module's "
    "thread was started by this call (or at install) and fills the pool on its own "
    "first pass, so a later apply will be written; if `pool_refills` stays 0 while "
    "`thread_alive` is true, pinned allocation itself is failing - see "
    "`pinned_allocation_failed` in reasons")
DRAINED_POOL_REASON = (
    "no pinned readback row is FREE for this region: rows were allocated and are all "
    "held by records whose load event has not passed, or leaked, so this write is "
    "refused rather than performed unwitnessed")


def _pool_cold() -> bool:
    """True when no pinned row has EVER been allocated in this process since boot.

    That is the cold start, and it is a fact about the WRITER rather than about the
    device: once a single refill has run, `buffers_allocated` stops being zero for the
    rest of the boot (rows leak and are replaced by fresh ones, they are never
    un-counted), so from then on a refusal means the pool is genuinely drained.
    """
    with _lock:
        return int(_state["counters"]["buffers_allocated"]) == 0


def _refuse_pinned() -> dict:
    """The pool refusal: one NAME, the cause counted beside it and said in its place."""
    cold = _pool_cold()
    with _lock:
        _count("no_pinned_buffer")
        if cold:
            _count("pool_never_filled")
    _reason("no_pinned_buffer_cold_pool" if cold else "no_pinned_buffer",
            COLD_POOL_REASON if cold else DRAINED_POOL_REASON)
    return {"written": False, "reason": "no_pinned_buffer",
            "cause": ("pool_never_filled" if cold else "pool_drained")}


def _write_one(worker, job: dict, event) -> dict:
    """One entry: plan, take rows, enqueue copy + two readbacks, hold the rows.

    NOTHING HERE WAITS. The copies are `non_blocking=True` on the tier's load stream,
    the rows came from a pool that was filled on another thread, and the only CUDA-ish
    call is making that stream current for the duration of four method calls.
    """
    width = _int_or_none(job.get("bytes"))
    if not width or width <= 0:
        return _refuse("width_mismatch",
                       "the restore entry declares no `bytes`, so there is no window "
                       "width to plan against")
    group = _int_or_none(job.get("group"))
    group_bytes = _group_width(group)
    if not group_bytes or group_bytes <= 0:
        return _refuse("no_group_width",
                       f"group {group} has no readable `page_size_bytes`, so W_g is "
                       "unknown and the comparison would have to guess where the "
                       "group's bytes stop")
    if group_bytes > width:
        return _refuse("group_wider_than_file",
                       f"group {group}'s own width {group_bytes} B exceeds this rank's "
                       f"window {width} B: there is no correct number of bytes to "
                       "compare, so nothing is written")
    plan = _plan(worker, job, width)
    if not plan["ok"]:
        return _refuse(plan["kind"] if plan["kind"] in _COUNTER_KEYS
                       else "width_mismatch", str(plan["reason"]))
    stream = getattr(worker, "load_stream", None)
    context = _stream_context(stream) if stream is not None else None
    if context is None:
        return _refuse("stream_unavailable",
                       "this worker exposes no usable `load_stream` (or this process "
                       "has not imported torch), so there is no stream to order our "
                       "copy before the fork's on - and writing on the wrong stream "
                       "would break the ordering argument, not just weaken it")
    # THE THREAD, BEFORE THE POOL. The rows this write needs are allocated only BY this
    # thread, so asking for it AFTER `_take_rows` (where 0.2.43 asked) made the first
    # refusal the thing that prevented the only thing that could have answered it: cold
    # pool -> refusal -> return -> no thread -> cold pool. The first write was
    # unreachable and so was every later one. Asking here means the very first armed
    # apply at least STARTS the writer; the refusal is still a refusal, because an
    # allocation on this thread is a `cudaHostAlloc` inside `start_load_kv`.
    if not _ensure_thread():
        return _refuse("refused_thread_unavailable",
                       "this module's thread could not be started, so no write is "
                       "attempted: an unwitnessed write is worse than no write, and "
                       "the pinned rows this write needs are allocated on that thread")
    with _lock:
        if len(_state["in_flight"]) >= MAX_IN_FLIGHT:
            _count("in_flight_dropped")
            full = True
        else:
            full = False
    if full:
        return _refuse("no_pinned_buffer",
                       f"{MAX_IN_FLIGHT} records are already in flight waiting for "
                       "their load events; the write is skipped rather than queued",
                       cause="in_flight_cap")
    taken = _take_rows(plan)
    if taken is None:
        return _refuse_pinned()
    names = [region["name"] for region in plan["regions"]]
    try:
        with context:
            # THE PRE-COPY READBACK, enqueued FIRST, so on this stream it reads the
            # window as it was BEFORE our copy. It is what turns "the file's bytes are
            # not in the window" into a decidable question: without a before-state, a
            # window that another stream rewrote is indistinguishable from a copy of
            # ours that addressed the wrong block.
            rows_pre = _lane_rows(taken, "pre")
            for region in plan["regions"]:
                _copy_device_to_host(rows_pre[region["name"]], region["gpu"],
                                     plan["gpu_block"])
            # THE COPIES, in sorted-name order, one per region. By name, never by
            # position: see `_plan`.
            for region in plan["regions"]:
                _copy_host_to_device(region["gpu"], plan["gpu_block"], region["cpu"],
                                     plan["cpu_block"])
            # THE POST-COPY READBACK, after every copy and still inside the same stream
            # context, so it observes the window as OUR copy left it. The fork's copy
            # has not been enqueued yet, and that is what makes the comparison a witness
            # instead of a tautology.
            rows_post = _lane_rows(taken, "post")
            for region in plan["regions"]:
                _copy_device_to_host(rows_post[region["name"]], region["gpu"],
                                     plan["gpu_block"])
    except Exception as exc:  # noqa: BLE001 - a failed copy is a refusal, not an outage
        snapshot._error(exc, "disk_gpu_write enqueue")
        # LEAK, do not recycle: the stream may already hold a copy that will write one
        # of these rows after we hand it out again.
        _release_rows(taken, kind="rows_leaked_write_failed")
        return _refuse("write_failed",
                       f"enqueuing the copy raised {type(exc).__name__}; the rows it "
                       "may already have been handed are leaked rather than reused")
    with _lock:
        _state["seq"] = int(_state["seq"]) + 1
        record = {"id": int(_state["seq"]), "event": event, "mono": time.monotonic(),
                  "path": str(job.get("path") or ""),
                  "key16": (str(job.get("key16"))[:32] if job.get("key16") else None),
                  "group": group, "group_bytes": int(group_bytes),
                  "gpu_block": int(plan["gpu_block"]),
                  "cpu_block": int(plan["cpu_block"]),
                  "bytes": int(width), "names": list(names), "rows": taken}
        _state["in_flight"][record["id"]] = record
        _count("writes")
        _count("bytes_written", int(width))
        _count("regions_written", len(names))
        _count("readbacks_enqueued", 2 * len(names))
        # THE ORDERING WITNESS, in the other direction: this counter can only move
        # because the ledger check above passed, i.e. because the fork had not yet
        # submitted this load's copy. If the hook is ever moved after `original()`,
        # `event_already_submitted` starts moving and THIS one stops.
        _count("readback_enqueued_before_original")
    return {"written": True, "bytes": int(width), "regions": len(names)}


# --- THE VIEW ------------------------------------------------------------------
def counters() -> dict:
    with _lock:
        return dict(_state["counters"])


def _dump_size(payload) -> int:
    """The byte count `worker_diag` will measure, in ITS encoding."""
    return len(json.dumps(payload, separators=(",", ":"),
                          default=str).encode("utf-8"))


def _counted_trim(out: dict) -> dict:
    """Count a trim ONCE PER SHAPE OF TRIM, and write the count into the payload."""
    signature = tuple(out["trimmed"])
    with _lock:
        if signature != _state["last_trim"]:
            _state["last_trim"] = signature
            _count("section_trimmed")
        out["section_trimmed"] = int(_state["counters"]["section_trimmed"])
    return out


def _fit(out: dict) -> dict:
    """Never hand the writer a section it would REFUSE - and say what was shed.

    `worker_diag` refuses an over-cap section rather than truncating it, so an over-fat
    payload's failure mode is that the ENTIRE `disk_gpu_write` key vanishes from the
    rank file on the boot with the most to report. The ladder sheds in the order a
    reader acts on: the reason TEXT first (the leading tag survives), then the mismatch
    ILLUSTRATION (the counters above it never move), then the PROSE, then the reasons.
    No counter is ever shed - they are the load-bearing numbers.

    `trimmed` is ALWAYS PRESENT, empty when nothing was shed, for the reason
    `disk_restore_verify._fit` gives at length: a section that gains or loses a
    top-level key exactly when there is most to report is a section a reader cannot diff
    between two writes. A clipped value is not a missing key.
    """
    trimmed = out["trimmed"] = []
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return out
    for item in out.get("reasons") or []:
        if _dump_size(out) <= MAX_SECTION_BYTES:
            break
        if isinstance(item, str) and len(item) > TRIM_REASON_CHARS:
            index = out["reasons"].index(item)
            out["reasons"][index] = item[:TRIM_REASON_CHARS]
            trimmed.append(f"reasons:{TRIM_REASON_CHARS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    if out.get("mismatch_samples"):
        out["mismatch_samples"] = []
        trimmed.append("mismatch_samples")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    for key in ("note", "order_rule", "meaning_rule"):
        if isinstance(out.get(key), str) and len(out[key]) > TRIM_REASON_CHARS:
            out[key] = out[key][:TRIM_REASON_CHARS]
    trimmed.append(f"prose:{TRIM_REASON_CHARS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    out["reasons"] = []
    out["last_verdict"] = None
    trimmed.append("reasons_dropped")
    out = _counted_trim(out)
    if _dump_size(out) > MAX_SECTION_BYTES:
        _note_once("section-over-budget",
                   f"[disk_gpu_write] section STILL over its byte budget after every "
                   f"trim step ({_dump_size(out)} B over {MAX_SECTION_BYTES} B): the "
                   "rank writer will refuse it, so `disk_gpu_write` will be absent "
                   "from diagnostics-rank.json", level="error")
    return out


def pool_view() -> dict:
    """The pool's live shape, so `no_pinned_buffer` is never a mystery."""
    with _lock:
        free = {f"{name}|{elems}B-ELEMS": len(rows) for (name, elems), rows
                in _state["pool"].items()}
        return {"free_rows": free,
                "rows_per_region": int(MAX_PINNED_ROWS_PER_REGION),
                "pinned_bytes": int(_state["pinned_bytes"]),
                "pinned_peak_bytes": int(_state["pinned_peak"]),
                "in_flight": len(_state["in_flight"])}


# --- IS THIS WITNESS ACTUALLY ALIVE? -------------------------------------------
# The four states a rank file can be in, named so that "is the writer running?" is a
# lookup and not a reading of five counters. 0.2.43 shipped a writer that could not
# write and a section that said nothing about it: `writes=0` against `applies_seen=64`
# was true, legible only to someone who already knew that `passes`, `pool_refills` and
# `buffers_allocated` all being zero means the thread never ran - and that
# `thread_started=0` was the proof. These are those counters, in one sentence, computed
# rather than remembered, so they cannot disagree with the numbers beside them.
LIVENESS_OFF = "off"
LIVENESS_NO_APPLY = "armed, no apply point seen in this process"
LIVENESS_NEVER_WROTE = "armed but never wrote"
LIVENESS_WRITING = "writing"


def liveness_view() -> dict:
    """`is this witness alive?`, answered from the counters that are already published.

    A derived view, never a stored one: it reads the same counters the section
    publishes, so there is no second state to keep honest and nothing for a reset to
    forget. The key set is fixed in every state (a reader must be able to diff two rank
    files), and `cause` is non-null only in the dead state it explains.
    """
    with _lock:
        counts = dict(_state["counters"])
        free_rows = sum(len(rows) for rows in _state["pool"].values())
        in_flight = len(_state["in_flight"])
    running = bool(_thread is not None and _thread.is_alive())
    writes = int(counts["writes"])
    applies = int(counts["applies_seen"])
    cause = None
    if not effective_enabled():
        state = LIVENESS_OFF
        verdict = ("not enabled: no thread starts, nothing is page-locked, no copy is "
                   "issued and the fork's copy runs alone")
    elif writes:
        state = LIVENESS_WRITING
        verdict = (f"writing: {writes} write(s) over {applies} applied entr(ies), "
                   f"{counts['checked']} read back and compared, {free_rows} pinned "
                   f"row(s) free")
    elif not applies:
        state = LIVENESS_NO_APPLY
        verdict = ("armed, but this process has run no restore apply point, so it can "
                   "never write: only a WORKER applies a restore (thread started "
                   f"{counts['thread_started']}x, running: "
                   f"{'yes' if running else 'NO'})")
    else:
        state = LIVENESS_NEVER_WROTE
        if counts["pool_never_filled"]:
            cause = "pool_never_filled"
        elif counts["in_flight_dropped"]:
            cause = "in_flight_cap"
        elif counts["no_pinned_buffer"]:
            cause = "pool_drained"
        elif counts["sampled_out"] >= applies:
            cause = "never_sampled_in"
        else:
            cause = "see_reasons"
        verdict = (f"armed but NEVER WROTE: {applies} applied entr(ies) seen, "
                   f"{counts['sampled_out']} sampled out, "
                   f"{counts['no_pinned_buffer']} refused for want of a pinned row "
                   f"({counts['pool_never_filled']} of them with a pool that had never "
                   f"been filled), thread started {counts['thread_started']}x and "
                   f"running: {'yes' if running else 'NO'}, rows allocated "
                   f"{counts['buffers_allocated']}, pool refills "
                   f"{counts['pool_refills']}, passes {counts['passes']}, free rows "
                   f"{free_rows}, in flight {in_flight}; cause: {cause}")
    return {"state": state, "verdict": verdict, "thread_running": running,
            "free_rows": free_rows, "cause": cause}


def geometry_view() -> dict:
    with _lock:
        return {"group_bytes": copy.deepcopy(_state["group_bytes"]),
                "group_bytes_note": _state["group_bytes_note"],
                "window_bytes": _state["window_bytes"],
                "window_note": _state["window_note"],
                "window_rule": "sum(region.stride(0) * element_size)"}


def section() -> dict:
    """The `disk_gpu_write` block: EVERY KEY IS ALWAYS PRESENT.

    The counters, the enablement, the limits in force, the geometry the comparison used,
    the pool's live shape and the last verdicts. The degraded view is built by this same
    function, which is what makes "the key set does not change when there is no GPU" a
    structural property rather than a copy.
    """
    with _lock:
        out = {key: int(_state["counters"][key]) for key in _COUNTER_KEYS}
        out["reasons"] = copy.deepcopy(_state["reasons"])
        out["last_verdict"] = (copy.deepcopy(_state["last_verdict"])
                               if _state["last_verdict"] else None)
        out["mismatch_samples"] = copy.deepcopy(_state["mismatch_samples"])
    enable = enable_state()
    out["enabled"] = enable["enabled"]
    out["enable_source"] = enable["source"]
    out["env_var"] = config.ENV_DISK_GPU_DIRECT_WRITE
    out["env_raw"] = enable["env_raw"]
    out["env_recognised"] = enable["env_recognised"]
    out["latch"] = enable["latch"]
    out["sample_every_n_applied"] = int(SAMPLE_EVERY_N_APPLIED)
    out["max_writes_per_step"] = int(MAX_WRITES_PER_STEP)
    out["min_interval_seconds"] = float(MIN_INTERVAL_SECONDS)
    out["max_checks_per_pass"] = int(MAX_CHECKS_PER_PASS)
    out["job_max_age_seconds"] = float(JOB_MAX_AGE_SECONDS)
    out["max_in_flight"] = int(MAX_IN_FLIGHT)
    out["geometry"] = geometry_view()
    out["pool"] = pool_view()
    out["thread_alive"] = bool(_thread is not None and _thread.is_alive())
    # THE ONE-LINE ANSWER TO "IS THE WRITER RUNNING?", beside the counters it sums.
    out["liveness"] = liveness_view()
    out["order_rule"] = ORDER_RULE
    out["meaning_rule"] = MEANING_RULE
    out["section"] = MODULE
    out["writer"] = OWNER_TAG
    out["note"] = ("a second writer for the destination, not a replacement: the "
                   "fork's own copy runs after ours on the same stream and wins")
    return _fit(out)


def worker_report() -> dict:
    """The rank-JSON contribution. Builds every key, always; owns no file."""
    return section()


def _contribution_signal() -> tuple:
    """The rank writer's "did anything change?" value for this section. Cheap: ONE
    counter snapshot plus the two live facts the counters do not carry.

    `thread_started` and `pool_refills` are in here because they are exactly the facts
    that move with NOTHING else moving, which is what a signal is for: an armed boot
    that has written nothing yet goes from "there is no writer thread" to "there is
    one, and its pool is warm" without any other counter stirring, and a reader should
    not have to wait for restore traffic to see the witness arrive.
    """
    counts = counters()
    return (int(bool(enable_state()["enabled"])),
            int(bool(pool_view()["pinned_bytes"])),
            int(bool(counts["writes"])),
            int(bool(counts["thread_started"])),
            int(bool(counts["pool_refills"])))


def register_contribution() -> dict:
    """Offer this module's verdict to the rank-file writer. Idempotent.

    Registered by `disk_restore.install()` beside its own `disk_restore` and
    `disk_restore_verify` sections, because the facts it reports are that apply point's.
    THE CAP HAD TO MOVE FOR THIS: `worker_diag.MAX_CONTRIBUTIONS` was 5 with an
    allow-list of exactly five names, and the seam REFUSES an unregistered section, so a
    sixth section without the raise would have been silently absent from the rank file -
    which is the failure the two assertions in `test_worker_diag.py` and
    `test_disk_restore_verify.py` exist to catch. The arithmetic of six sections
    (24 KiB of a 32 KiB file, contributions shed first) is in `MAX_SECTION_BYTES`'
    comment above.

    IT ALSO STARTS THE WRITER, WHEN THIS PROCESS IS ARMED - that is the cold-start fix,
    and it belongs HERE because this is the only call this module has before its first
    restore. It is safe at engine start precisely because a thread on its own does
    nothing: `_ensure_thread` creates one daemon thread, the loop waits
    `PASS_TICK_SECONDS` before it acts, its refill asks `_resolve_worker()` - a weakref
    only the APPLY POINT ever sets, so it answers None here and the refill reads no
    geometry, allocates nothing, touches no CUDA and counts nothing - and its pass
    returns on the enablement test or on an empty in-flight map. A process that can
    never write (an EngineCore, which also registers) therefore runs a no-op tick and
    says so in `liveness` as `armed, no apply point seen in this process`. With the
    module OFF, which is the default, no thread starts and no counter moves.
    """
    answer = worker_diag.register_contribution(
        MODULE, worker_report,
        # OWNER_TAG's claim, kept true: this reports a STATE, never an age. What moves
        # with no counter moving is the ENABLEMENT, the POOL and the WRITER THREAD - any
        # of the three can change between two writes of the file with nothing written in
        # between, and a reader must be able to see that without diffing the file.
        owner=f"{OWNER_TAG} state, never a window",
        signal=_contribution_signal)
    if effective_enabled():
        _ensure_thread()
    return answer


def status() -> dict:
    """The verb's view: the enablement, the limits, the pool, the geometry, the
    counters and the last verdicts.

    `section()` builds it, so the verb, `worker_report()` and the rank file cannot
    disagree. In the EngineCore - the process that answers `/disk {"op":"gpu-write"}` -
    this is the only door there is, and it will honestly report zero writes, because a
    scheduler process runs no worker and no apply point.
    """
    out = {"ok": True, "reason": None, "module": MODULE}
    # THE SCOPE, SAID OUT LOUD. This is the one op whose zero needs the sentence most:
    # the write happens at the RESTORE APPLY POINT, which is code only a WORKER runs, so
    # the EngineCore that answers this verb will report `writes: 0` forever for that
    # reason alone. Without the field an operator reads the zero as an all-clear.
    out["scope"] = ("the write happens at the restore apply point, which only a WORKER "
                    "runs; an EngineCore reports zero writes because it has no apply "
                    "point, and the worker is armed with "
                    f"{config.ENV_DISK_GPU_DIRECT_WRITE}=1")
    out.update(section())
    return out


def stop_thread(timeout_seconds: float = 2.0) -> dict:
    """Ask this module's thread to stop, and say whether it did."""
    with _thread_lock:
        thread = _thread
        stop = _stop
    if stop is not None:
        stop.set()
    if thread is None:
        return {"stopped": True, "reason": "no thread was ever started"}
    alive = thread.is_alive()
    if alive:
        thread.join(timeout=max(0.0, float(timeout_seconds)))
        alive = thread.is_alive()
    return {"stopped": not alive,
            "reason": (None if not alive else
                       f"the thread was still alive after {timeout_seconds}s "
                       "(it is a daemon, so it dies with the process)")}


def reset_state_for_tests() -> None:
    """Back to a fresh boot. The thread is STOPPED, not just un-counted.

    A daemon thread left allocating against a worker from the previous test keeps
    mutating the NEXT test's `_state` through rows it filled, which turns a green suite
    into a property of file order (the sibling modules' resets say the same thing about
    their threads for the same reason).
    """
    global _thread, _stop
    stopped = stop_thread(timeout_seconds=2.0)
    if not stopped["stopped"]:
        logger.warning("kvtier-dev-plugin: [disk_gpu_write] reset found a thread that "
                       "would not stop: %s", stopped["reason"])
    _thread = None
    _stop = None
    # IN PLACE, not a rebind: `conftest.py`'s isolation fixture snapshots this object
    # before the reset and restores it afterwards.
    with _lock:
        _state.clear()
        _state.update(_fresh_state())


def view() -> dict:
    """The internal state, deep-copied: for tests and for a human reading a live
    process. The rows themselves are not published - only their shape, under `pool`.
    """
    with _lock:
        return copy.deepcopy({
            "in_flight": {key: {field: value for field, value in record.items()
                                if field != "rows"}
                          for key, record in _state["in_flight"].items()},
            "in_flight_rows": {key: sorted(str(name) for name in record["rows"])
                               for key, record in _state["in_flight"].items()},
            "pinned_bytes": int(_state["pinned_bytes"]),
            "pinned_peak": int(_state["pinned_peak"]),
            "group_bytes": copy.deepcopy(_state["group_bytes"]),
            "window_bytes": _state["window_bytes"],
            "last_verdict": copy.deepcopy(_state["last_verdict"]),
            "mismatch_samples": copy.deepcopy(_state["mismatch_samples"]),
            "reasons": list(_state["reasons"]),
            "latch": bool(_state["latch"]),
        })
