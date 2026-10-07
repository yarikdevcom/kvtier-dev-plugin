"""GPU-direct SHADOW WRITES: a second copy of a spilled block, written from the GPU.

WHAT THIS IS. The route to dropping the host-RAM tier and going GPU <-> NVMe directly
needs a GPU-direct WRITE and a GPU-direct READ, and neither exists. Step 1
(`disk_shadow.py`, 0.2.36) settled the prerequisite - the arena's bytes and the GPU's
bytes are the same bytes, proven offline against the fork's own layout code and then
in production by reading a completed store's GPU block and comparing it with the blob
the spill path captured. THIS IS STEP 2: the write path, built as a SHADOW. For a
block the verifier has just vouched for, this module writes the GPU's bytes to a file
under its OWN root, named exactly the way the live store names that block, and then
compares the two files. Two trees, one name, real traffic, thousands of blocks - the
diff IS the gate.

WHAT IT IS NOT. Not authoritative, ever. The live store's file is the only one any
read path loads; nothing in a spill, a restore, an eviction or an offer decision reads
a byte this module wrote. It never opens the live store for writing, never creates a
file inside it, and refuses outright if it is pointed at it. Default OFF, fail-open,
and its output is disposable evidence with a retention cap of its own.

WHERE THE BYTES COME FROM - AND WHY THERE IS NO SECOND GPU READ. This module never
touches a GPU tensor. The bytes it writes are the bytes `disk_shadow` already copied
to the host for its comparison, and it writes them for a block ONLY when that
comparison said `equal`. So: the device->host seam stays exactly one function in one
module (the one that is proved against a real copy in `test_disk_shadow.py`), the
staleness judgement is inherited rather than re-derived (a block the engine rewrote
under us is never written, because it never got an `equal`), and no code path here can
hand a device address to `ctypes.string_at`. That is the whole reason this module is
downstream of the verifier instead of beside it.

WHAT THE FILE DIFF THEREFORE PROVES, SAID EXACTLY. It does not re-answer "do the GPU
and the arena agree?" - the verifier answers that per block, live, in its own
counters. Given only `equal` blocks are written, the two files' bytes agree by
construction AT CAPTURE TIME, so a diff that disagrees is a fault in the thing this
module adds: the FILE PATH. Specifically it proves (a) naming parity - that
`disk_layout.build_slot_path` under a different root produces the same relative name
the live store wrote, which is the identity a restore looks up; (b) width - that what
landed is a whole slot and not half of one; (c) write discipline - `.part`, verified
size, `os.replace`, no truncation, no interleave; and (d) that across a store
turnover the GPU-direct tree and the arena tree stay byte-for-byte the same files. A
pair that differs, or a shadow file with no live partner, is that kind of fault, and
the counters name which.

THE ORDER RULE IS INHERITED, NOT RE-IMPLEMENTED. A slot's bytes are the sum over the
region tensors in `sorted(name)` order - never dict-insertion order, which is what the
engine's own disk backend walks and what 0.2.8 shipped as a bug - and that sum happens
in ONE place: `disk_shadow._read_gpu_slot()` hands the copied one-block regions to
`disk_spill.read_slot_bytes`, the spill path's own reader, at index 0. This module
never walks `gpu_kv_caches`, never orders anything, and never measures a region. The
same reader's slot-width check is also what refuses the other trap: the KV cache
groups OVERLAY one backing allocation (the slot width is the MAX over groups, not the
sum), so a design that walked group specs would try to write a 43 MB file into an
11.5 MB name and this is the check that says no.

NEVER ON THE STEP THREAD. The only function of this module the engine's thread calls
is `consider()`: two dict lookups, a width check, a cap check and a `put_nowait`. It
does no I/O of any kind - not one `stat`, not one `makedirs`. Every byte of file work
(naming, writing, comparing, pruning) runs on this module's daemon writer thread, the
same shape as `disk_spill`'s, and the comparison is bounded twice over:
`MAX_PAIRS_PER_PASS` pairs per pass and `MAX_PAIR_SCAN_BYTES` per pair, never more
than the slot's own width. The pass also runs on the thread's idle tick, so pairing
completes after the traffic stops.

THE DIRECTORY, AND WHAT HAPPENS IF SOMEONE POINTS IT AT THE LIVE STORE. Files land in
`VLLM_E2_TIER_DISK_GPU_SHADOW_ROOT` if that is set, otherwise in `<live store
root>/../gpu-shadow` - a SIBLING of the store, never inside it, because the two trees
have to be diffable with one command and a subtree of the store would show up as
evidence of itself. THE REFUSAL: before a write, this module compares the resolved
shadow root against the live store's root and refuses (counted as
`refused_root_conflict`) if they are the SAME directory or NESTED in either
direction - not "merge", not "write anyway under a prefix", because a shadow byte
inside the live store's shard tree is a byte a restore could load, and that is the one
outcome this whole module exists to make impossible. The refusal is a counter plus one
log line, and it fires per job, so it cannot be talked past by changing the
environment mid-boot (the root is resolved once, like the enablement).

RETENTION. These files are evidence, not data: `MAX_SHADOW_FILES` and
`MAX_SHADOW_BYTES` bound the directory, and a pass prunes oldest-first (by
`st_mtime`, name as tie-break) back under both. It deletes through
`disk_layout.execute_unlink_plan`, which re-validates every path at the moment of
deletion - a regular file, not a link, resolving DIRECTLY inside one directory of this
root, with a name that parses as ours - so the boundary is that proven function's
property and not a promise in this file. Nothing outside the root is ever a candidate:
the walk itself is confined to it.

THE VIEW. `section()` publishes a stable key set in every state - disabled, no root,
a root that conflicts, a writer that has never written, a writer that has written
thousands - because a reader diffing two rank files must be able to tell "moved" from
"gone". It keeps the `disk_shadow` trim discipline (`_fit`): the section bounds itself
to `MAX_SECTION_BYTES`, tighter than the contribution seam's cap which REFUSES an
over-weight section outright, and it sheds its illustrations first and its numbers
never. It also never walks the filesystem: the directory figures are the last
retention pass's measurements, published as measurements with a timestamp, because
`section()` runs on the step thread inside the rank-file writer.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import threading
import time

from . import config, disk_layout, disk_spill, snapshot

MODULE = "disk_gpu_shadow"
OWNER_TAG = "kvtier-dev-plugin/disk-gpu-shadow-1"
logger = logging.getLogger(MODULE)

# --- THE LIMITS, EACH WITH WHAT IT PROTECTS ----------------------------------
#
# Jobs in flight. A job holds ONE SLOT of GPU bytes (11,468,800 B on the live PP0), so
# this is a host-memory bound as much as a queue bound: 4 jobs is ~44 MB held by a
# diagnostic. It is also the back-pressure that makes "the shadow wrote nothing" a
# possible answer on a boot whose disk is slow, instead of an unbounded buffer.
MAX_QUEUE_JOBS = 4
# The same bound in bytes, checked BEFORE the job is accepted. `disk_spill`'s
# `MAX_QUEUED_BYTES` is the precedent for why the byte bound is the one that matters:
# a job count alone lets one wide slot through and then drops nothing while RAM grows.
MAX_QUEUED_BYTES = 64 * 1024 * 1024
#
# How long the queue may be held before it is dropped for being stale. A shadow file
# for a block the engine spilled 10 minutes ago is not evidence about anything.
JOB_MAX_AGE_SECONDS = 60.0
#
# THE COMPARISON'S TWO BOUNDS. Pairs per pass is the CPU/IO bound (a pass runs on the
# writer thread and must not starve writes); the scan bound is the per-pair bound and
# is deliberately ABOVE the live slot widths, so the real per-pair limit is the slot
# width itself - comparing more bytes than a slot is not a thing, and saying so here
# is cheaper than discovering it in a profile.
MAX_PAIRS_PER_PASS = 4
MAX_PAIR_SCAN_BYTES = 32 * 1024 * 1024
VERIFY_CHUNK_BYTES = 1024 * 1024     # a chunked `bytes` compare is memcmp, in C
#
# A shadow file whose live partner has not appeared within this window is reported as
# `pairs_unpaired_expired` and dropped. The spill's own writer thread is a different
# thread, so "not yet" is normal for milliseconds; "still not after two minutes" means
# the live file was refused, evicted, or named differently - which is a real answer.
PAIR_MAX_AGE_SECONDS = 120.0
#
# The pending-comparison list's own cap. Overflow sheds the OLDEST entry with a
# counter: the newest pairs are the ones describing traffic that is still warm.
MAX_PENDING_PAIRS = 64
#
# How often the writer thread may run a compare/retention pass. The thread's tick is
# 0.5 s, so this is the "don't spin on a full pending list" floor, not a scheduler.
PASS_INTERVAL_SECONDS = 0.25
#
# THE RETENTION CAPS. Count AND bytes, because a slot is 11.4 MB and a count cap alone
# would let a wide rank fill a device with evidence. 64 files x 11.4 MB is ~730 MB at
# the live width; the byte cap is what actually bounds it.
MAX_SHADOW_FILES = 64
MAX_SHADOW_BYTES = 512 * 1024 * 1024
RETENTION_INTERVAL_SECONDS = 5.0
#
# The walk's own bounds. 256 shard directories is the whole shard alphabet this naming
# scheme can produce (`SHARD_CHARS` hex), so the directory cap is exact rather than
# arbitrary; the file cap bounds the stat work, and a truncated walk reports itself.
MAX_WALK_DIRS = 512
MAX_WALK_FILES = 4096
MAX_UNLINKS_PER_PASS = 256
#
# Failure reasons an operator can read: capped, oldest kept, so a persistent refusal
# stays visible instead of being pushed out by a flood of identical ones.
MAX_REASON_SAMPLES = 8
MAX_REASON_CHARS = 120
#
# How many differing file names the section publishes. Names are long
# (`kvslot-<key16>-g<group>-<geodigest12>.bin`) and a reader needs a sample to go look
# at, not a list that eats the section's byte budget.
MAX_NAME_SAMPLES = 6
#
# THE SECTION'S OWN BYTE BUDGET, tighter than the seam's on purpose, for the reason
# `disk_shadow` documents at length: `worker_diag.MAX_CONTRIBUTION_BYTES` REFUSES an
# over-cap section instead of truncating it, so a fat payload deletes the whole section
# on the boot someone is reading the file about. This module sheds its illustrations in
# a declared order and says what it shed in `trimmed`.
MAX_SECTION_BYTES = 3072
TRIM_REASON_CHARS = 48
TRIM_NAME_SAMPLES = 2

# THE STABLE KEY SET. `section()` builds every one of these on every call, in every
# state, and the degraded view is built by the same code as the healthy one - that is
# what makes "the key set does not change when nothing was written" a structural
# property rather than a copy. The counters are NEVER a trim target.
_COUNTER_KEYS = (
    # the hand-off from the verifier
    "offers",
    "refused_not_equal",
    "refused_not_checked",
    "refused_no_bytes",
    "refused_slot_width",
    "refused_no_key",
    # the door: is there somewhere to write that is not the store?
    "refused_not_configured",
    "refused_root_conflict",
    "refused_root_invalid",
    # the name: two naming sites must agree, or nothing is written
    "refused_naming",
    "refused_name_mismatch",
    # the queue
    "queued",
    "refused_queue_full",
    "refused_bytes_full",
    "refused_stale_job",
    "refused_thread_unavailable",
    # the write
    "written",
    "duplicates",
    "bytes_written",
    "refused_write_failed",
    "committed_size_mismatch",
    # the diff
    "pairs_requeued",
    "pairs_checked",
    "pairs_identical",
    "pairs_differing",
    "pairs_size_disagree",
    "pairs_unpaired_expired",
    "pairs_shadow_missing",
    "pairs_dropped_overflow",
    # the retention
    "retention_passes",
    "retention_removed",
    "retention_refused",
    "retention_errors",
    "walk_entries_examined",
    "walk_truncated",
    # the section
    "section_trimmed",
    "errors",
)

_state: dict = {
    "latch": False,           # the POST op's decision; the environment is the floor
    "latch_set": False,       # has the POST said anything at all in this process
    "env_resolved": False,    # the environment is read ONCE, see `_resolve_env`
    "env_enabled": False,
    "env_raw": None,
    "env_recognised": True,
    "root_resolved": False,   # the same cadence for the root: see `_resolve_root`
    "root": None,
    "root_source": None,
    "root_reason": None,
    "live_root": None,
    "notes": {},
    "reasons": [],
    "reason_seen": set(),
    "last_trim": None,
    # the queue's live shape (the step thread's own accounting, never a walk)
    "queue_depth": 0,
    "queue_bytes": 0,
    "pending": [],            # [{name, shadow_path, live_path, width, since}]
    # Directory figures. The RUNNING estimates are the step thread's, and the WALKED
    # ones are the last retention pass's measurement of the real tree: `section()`
    # publishes the latter as a measurement with a timestamp and never walks, because
    # `section()` runs on the step thread inside the rank-file writer.
    "estimated_files": 0,
    "estimated_bytes": 0,
    "measured_files": None,
    "measured_bytes": None,
    "measured_age_seconds": None,
    "last_pass_mono": None,
    "last_retention_mono": None,
    "last_difference": None,
    "differing_names": [],
    "counters": {key: 0 for key in _COUNTER_KEYS},
}
_lock = threading.Lock()
_pass_lock = threading.Lock()


# --- THE BOOKKEEPING PRIMITIVES (the `disk_shadow` idioms, deliberately the
# same) ----------------------------------------------------------------------
def _note_once(key: str, message: str, level: str = "warning") -> bool:
    """One log line per episode, never one per tick.

    `setdefault` rather than a subscript: the test harness resets `_state` in place
    while a writer thread from the previous test may still be inside a pass, and a
    diagnostic must not become the thing that raises. See `_contained`.
    """
    notes = _state.setdefault("notes", {})
    if notes.get(key):
        return False
    notes[key] = True
    logger.log(getattr(logging, level.upper()), "kvtier-dev-plugin: %s", message)
    return True


def _count(name: str, amount: int = 1) -> None:
    """Increment one counter, tolerating a state that was reset underneath us.

    Counting is never the point of a line of this module, so a missing table is a no-op
    rather than a `KeyError` raised on a writer thread.
    """
    counters = _state.get("counters")
    if not isinstance(counters, dict):
        return
    counters[name] = int(counters.get(name, 0)) + int(amount)


def _reason(tag: str, detail: str) -> None:
    """Record WHY something was refused, once per distinct tag, and log it once.

    CLIPPED, because the section is a guest in someone else's file: an unbounded
    exception string from a driver would delete the whole section (`MAX_REASON_CHARS`,
    and `_fit` below).
    """
    seen = _state.setdefault("reason_seen", set())
    if tag in seen:
        return
    seen.add(tag)
    text = str(detail)
    if len(text) > MAX_REASON_CHARS:
        text = text[:MAX_REASON_CHARS - 3] + "..."
    reasons = _state.setdefault("reasons", [])
    reasons.append(f"{tag}: {text}")
    del reasons[MAX_REASON_SAMPLES:]
    _note_once(f"reason:{tag}", f"[disk_gpu_shadow] {tag}: {text}")


# --- ENABLEMENT ---------------------------------------------------------------
def _resolve_env() -> dict:
    """Read `VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES` ONCE per process, and say so.

    Same grammar as `config.disk_shadow_enable()` (unset = OFF, an on word = ON, a
    value in neither vocabulary = OFF *and* reported) and the same cadence argument: a
    live boot's workers inherit an environment that does not change, so re-reading it
    per store would put an `os.environ` lookup on the engine's step thread for no
    information. `source` names the cadence instead of hiding it.
    """
    if not _state["env_resolved"]:
        read = config.disk_gpu_shadow_writes()
        _state["env_raw"] = read["raw"]
        _state["env_enabled"] = bool(read["enabled"])
        _state["env_recognised"] = bool(read["recognised"])
        _state["env_resolved"] = True
        if read["enabled"]:
            _note_once("env-on",
                       f"[disk_gpu_shadow] ENABLED by "
                       f"{config.ENV_DISK_GPU_SHADOW_WRITES}={read['raw']!r}: a block "
                       "the shadow verifier called `equal` is written again, from the "
                       "GPU's bytes, under a SHADOW root that is never the live store",
                       level="info")
        elif read["raw"] is not None and not read["recognised"]:
            _reason("env_value_unrecognised",
                    f"{config.ENV_DISK_GPU_SHADOW_WRITES}={read['raw']!r} is neither "
                    f"an on word nor an off word; the GPU-direct shadow writer is OFF, "
                    f"which is the direction a typo never overrides")
    return {"enabled": bool(_state["env_enabled"]),
            "source": ("environment (read once at first use)"
                       if _state["env_raw"] is not None else
                       "default (off); the environment is read once at first use"),
            "raw": _state["env_raw"],
            "recognised": bool(_state["env_recognised"]),
            "env_var": config.ENV_DISK_GPU_SHADOW_WRITES}


def effective_enabled() -> bool:
    """Shadow writes happen only if the environment says yes or the POST latch does."""
    if _state["latch"]:
        return True
    return bool(_resolve_env()["enabled"])


def enable_state() -> dict:
    """One stable-key view of the enablement, including WHO decided it."""
    env = _resolve_env()
    return {"enabled": bool(env["enabled"] or _state["latch"]),
            "env_enabled": bool(env["enabled"]),
            "latch": bool(_state["latch"]),
            "latch_set": bool(_state["latch_set"]),
            "source": ("post op" if _state["latch"] else env["source"]),
            "env_var": config.ENV_DISK_GPU_SHADOW_WRITES,
            "env_raw": env["raw"],
            "env_recognised": bool(env["recognised"])}


def set_enabled(on: bool) -> dict:
    """The POST op's decision: IN-PROCESS ONLY, and said out loud in the answer.

    No control file is written - the same scope `disk_shadow.set_enabled` publishes,
    for the same reason. What makes the scope worth stating twice here is that this
    feature WRITES FILES: a POST that reaches the EngineCore latches the EngineCore,
    and the process that can actually write a shadow file is the WORKER that spills,
    which is armed with the environment variable. `scope` says that instead of letting
    an operator believe he turned on a writer that is still off.
    """
    with _lock:
        _state["latch"] = bool(on)
        _state["latch_set"] = True
        answer = enable_state()
    answer["scope"] = ("this process only; the worker that spills and writes is armed "
                       f"with {config.ENV_DISK_GPU_SHADOW_WRITES}=1 in the engine's "
                       "environment")
    if on:
        _note_once("latch-on",
                   "[disk_gpu_shadow] enabled in this process by the POST op "
                   "(scope: the answering process, which is usually NOT the worker)")
    return answer


# --- THE ROOT: WHERE THE EVIDENCE GOES, AND THE REFUSAL -----------------------
def _resolve_root() -> dict:
    """Resolve the shadow root ONCE, and refuse the live store by NAME.

    Order of decision, first match wins:
      1. `VLLM_E2_TIER_DISK_GPU_SHADOW_ROOT` - what the operator named.
      2. `<live store root>/../gpu-shadow` - a SIBLING of the store. Sibling and not
         child, because the point of the feature is one `cmp` over two trees, and a
         shadow directory inside the store would be walked by the store's own sweep and
         show up as evidence of itself.
      3. NEITHER - and then the answer is "not resolved yet", not "refused". A rank-file
         poll can reach `root_state()` in a process whose store root has not been
         adopted yet, and a cache that latched "no root" there would leave the writer
         unable to write for the rest of the boot. Resolution is therefore only CACHED
         once there is something to derive from (the operator named a root, or the live
         store has one); until then it stays open and costs one env lookup.

    THE LIVE-STORE REFUSAL IS THE POINT OF THIS FUNCTION. Same directory, or nested in
    either direction (component-boundary compared, so `/a/store2` is not "inside"
    `/a/store`), is a REFUSAL recorded per job: merging shadow files into the live
    store's shard tree would put bytes here that a restore could load, which is the one
    outcome this module exists to make impossible.
    """
    if _state["root_resolved"]:
        return {"root": _state["root"], "source": _state["root_source"],
                "reason": _state["root_reason"], "resolved": True}
    named = config.disk_gpu_shadow_root()["raw"]
    live = disk_spill.configured().get("root")
    _state["live_root"] = live
    if named is not None:
        candidate, source = named, f"environment ({config.ENV_DISK_GPU_SHADOW_ROOT})"
    elif live:
        candidate = os.path.join(os.path.dirname(os.path.abspath(str(live))),
                                 "gpu-shadow")
        source = "derived (the live store root's sibling)"
    else:
        # Nothing to write under and nothing to derive from: NOT a cached refusal.
        return {"root": None,
                "source": "not resolved (nothing named a root yet)",
                "reason": (f"neither {config.ENV_DISK_GPU_SHADOW_ROOT} nor "
                           f"{config.ENV_DISK_ROOT} names a store, so there is no "
                           "shadow root to write under"),
                "resolved": False}
    _state["root_resolved"] = True
    checked = disk_layout.validate_root(candidate)
    if not checked["ok"]:
        _state["root"] = None
        _state["root_source"] = source
        _state["root_reason"] = (f"{candidate!r} is not a root this plugin will write "
                                 f"to: {checked['reason']}")
    else:
        _state["root"] = checked["absolute"]
        _state["root_source"] = source
        _state["root_reason"] = None
        _note_once("root",
                   f"[disk_gpu_shadow] shadow files land in {checked['absolute']} "
                   f"({source}); the live store is "
                   f"{os.path.abspath(str(live)) if live else 'not configured'} and "
                   "is NEVER written by this module", level="info")
    if _state["root"] is None and effective_enabled():
        # Said out loud only when someone actually turned this on: a boot with no disk
        # tier at all must not get a warning line from a facility it never armed.
        _reason("root_unavailable", _state["root_reason"])
    return {"root": _state["root"], "source": _state["root_source"],
            "reason": _state["root_reason"], "resolved": True}


def _conflicts_with_store(root: str, live_root: str | None) -> str | None:
    """Why this shadow root IS the live store, or None. Component boundaries only.

    Checked per job rather than once, because the live root is the ONE piece of this
    module's geography that a boot can change under it (`VLLM_E2_TIER_DISK_ROOT` is
    read per call by `disk_spill.configured`), and a feature that refuses to write
    into the store must still refuse after that happens.
    """
    if not live_root:
        return None
    shadow = os.path.abspath(str(root))
    store = os.path.abspath(str(live_root))
    if shadow == store:
        return "the shadow root IS the live store's root"
    for inside, outside, label in ((shadow, store, "the shadow root is INSIDE the "
                                                  "live store"),
                                   (store, shadow, "the live store is INSIDE the "
                                                   "shadow root")):
        if inside == outside or inside.startswith(outside + os.sep):
            return f"{label} ({shadow} vs {store}): a shadow byte inside the store's " \
                   "shard tree is a byte a restore could load, so nothing is written " \
                   "under either name"
    return None


def root_state() -> dict:
    """A stable-key view of where files go and why they might not go there."""
    resolved = _resolve_root()
    conflict = _conflicts_with_store(resolved["root"], _state["live_root"]) \
        if resolved["root"] else None
    return {"root": resolved["root"], "source": resolved["source"],
            "reason": resolved["reason"], "resolved": bool(resolved["resolved"]),
            "live_root": _state["live_root"], "conflict": conflict,
            "env_var": config.ENV_DISK_GPU_SHADOW_ROOT,
            "env_raw": config.disk_gpu_shadow_root()["raw"],
            "filename_scheme": "kvslot-<key16>-g<group>-<geodigest12>.bin, the "
                               "live store's own scheme under the shadow root"}


# --- THE HAND-OFF: CALLED FROM THE STEP THREAD, DOES NOTHING ------------------
def consider(plan, verdict) -> dict:
    """Offer a compared block for a shadow write. Cheap, bounded, never raises.

    Called by `disk_store.on_store_completed` with the verifier's own plan and the
    verdict `disk_shadow.settle` just produced - the same two objects, so this module
    cannot disagree with the witness about what it saw. THE RULE IT ENFORCES: a block
    the verifier could not vouch for is NOT written. `equal` is the only verdict that
    gets a file, because the GPU block was freed the moment the store completed and
    every other verdict means "these bytes were not confirmed to still be this block's"
    - which is exactly the case where writing them would produce evidence that lies.

    The bytes handed to the writer are `plan["gpu"]`: the host copy
    `disk_shadow._read_gpu_slot` already made through the ONE device->host seam. No
    second copy, no second read, no device address in this file.
    """
    try:
        if not effective_enabled():
            return {"written": False, "reason_kind": "disabled"}
        with _lock:
            # A verdict of None is "not sampled" (the verifier's own 1-in-32, step cap
            # and interval throttle all decided before this), which is the normal state
            # and is not a refusal worth a counter.
            if not isinstance(verdict, dict):
                return {"written": False, "reason_kind": "not_offered"}
            _count("offers")
            key = (plan or {}).get("key") if isinstance(plan, dict) else None
            if not key:
                _count("refused_no_key")
                _reason("no_key", "a compared store carried no block key, so there is "
                                  "no name to write the shadow bytes under")
                return {"written": False, "reason_kind": "no_key"}
            if verdict.get("checked") is not True:
                # "There was nothing to compare" (no blob, a device fault, a width
                # disagreement) is a DIFFERENT fact from "they disagreed", and the
                # counters keep them apart for the same reason `disk_shadow` does.
                _count("refused_not_checked")
                return {"written": False, "reason_kind": "not_checked"}
            if verdict.get("equal") is not True:
                _count("refused_not_equal")
                return {"written": False, "reason_kind": "not_equal",
                        "category": verdict.get("category")}
            blob = plan.get("gpu") if isinstance(plan, dict) else None
            if not isinstance(blob, (bytes, bytearray, memoryview)) or not blob:
                _count("refused_no_bytes")
                _reason("no_gpu_bytes",
                        "the verifier said `equal` but handed no copied bytes: the "
                        "hand-off between the two modules is broken, and nothing is "
                        "written until it is")
                return {"written": False, "reason_kind": "no_bytes"}
            data = bytes(blob)
            root = _resolve_root()
            if not root["root"]:
                _count("refused_not_configured")
                return {"written": False, "reason_kind": "not_configured"}
            conflict = _conflicts_with_store(root["root"], _state["live_root"])
            if conflict:
                _count("refused_root_conflict")
                _reason("root_conflict", conflict)
                return {"written": False, "reason_kind": "root_conflict"}
            width = len(data)
            if int(_state["queue_bytes"]) + width > MAX_QUEUED_BYTES:
                _count("refused_bytes_full")
                return {"written": False, "reason_kind": "bytes_full",
                        "reason": f"holding one more slot ({width} B) would exceed "
                                  f"the {MAX_QUEUED_BYTES} B this writer may hold"}
            if int(_state["queue_depth"]) >= MAX_QUEUE_JOBS:
                _count("refused_queue_full")
                return {"written": False, "reason_kind": "queue_full",
                        "reason": f"the queue is at its {MAX_QUEUE_JOBS}-job cap"}
        if not _ensure_thread():
            with _lock:
                _count("refused_thread_unavailable")
            return {"written": False, "reason_kind": "thread_unavailable"}
        job = {"key": key, "blob": data, "bytes": width,
               "slot": (plan or {}).get("slot"),
               "gpu_block": (plan or {}).get("gpu_block"),
               "queued_ts": time.monotonic()}
        if not _put(job):
            return {"written": False, "reason_kind": "queue_full"}
        return {"written": True, "queued": True, "bytes": width,
                "reason": "the file is written on this module's writer thread"}
    except Exception as exc:  # noqa: BLE001 - never into the engine's step path
        with _lock:
            _count("errors")
        snapshot._error(exc, "disk_gpu_shadow consider")
        _reason("consider", f"the hand-off raised {type(exc).__name__}")
        return {"written": False, "reason_kind": f"error_{type(exc).__name__}"}


# --- THE QUEUE AND ITS THREAD -------------------------------------------------
_queue = None
_writer_thread = None
_thread_lock = threading.Lock()


def _get_queue():
    """The one bounded queue. Created on first accepted job, never at import."""
    global _queue
    with _thread_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_QUEUE_JOBS)
        return _queue


def _ensure_thread() -> bool:
    """Start the module's single writer thread, once, lazily. Never at import."""
    global _writer_thread
    with _thread_lock:
        if _writer_thread is not None and _writer_thread.is_alive():
            return True
        try:
            thread = threading.Thread(target=_writer_loop,
                                      name="kvtier-disk-gpu-shadow", daemon=True)
            thread.start()
        except Exception as exc:  # noqa: BLE001 - no thread is a refusal, not a fault
            snapshot._error(exc, "disk_gpu_shadow thread start")
            return False
        _writer_thread = thread
        return True


_STOP = object()


def _contained(action) -> dict | None:
    """Run one writer-thread action so that NOTHING it does can kill the thread.

    This is not defensive programming for its own sake: `conftest.py`'s per-test reset
    clears this module's `_state` IN PLACE while a writer thread from the previous test
    may still be inside a pass, and a thread that reads a key out of a half-cleared
    dictionary raises a `KeyError` that has nothing to do with the product. A loop with
    an unguarded call after its `finally` dies there silently, the queue stops draining,
    and every later `drain()` times out - which is exactly how this module's own tests
    failed when two files ran together. So: every pass is run through this, counted, and
    logged once by name.
    """
    try:
        return action()
    except Exception as exc:  # noqa: BLE001 - a pass never kills the writer thread
        with _lock:
            _count("errors")
        snapshot._error(exc, "disk_gpu_shadow pass")
        _reason("writer_pass", f"the compare/retention pass raised "
                               f"{type(exc).__name__}")
        return None


def _put(job) -> bool:
    """One `put_nowait`, with the accounting that makes the caps real.

    `.get(name, 0)` rather than a subscript in the accounting: this runs on the ENGINE's
    step thread, and a diagnostic that cannot be written (a `_state` reset underneath by
    the test harness) must never be what raises there. `consider` guards it anyway.
    """
    local = _get_queue()
    try:
        local.put_nowait(job)
    except queue.Full:
        with _lock:
            _count("refused_queue_full")
        return False
    with _lock:
        _state["queue_depth"] = int(_state.get("queue_depth", 0)) + 1
        _state["queue_bytes"] = int(_state.get("queue_bytes", 0)) + int(job["bytes"])
        _count("queued")
    return True


def _writer_loop() -> None:
    """The drain loop. ONE thread, and the only place this module writes a file.

    The tick is `queue.get(timeout=...)`, so the same loop that drains jobs also runs
    the bounded compare/retention pass while the queue is empty - which is what lets a
    pair complete after the traffic stops instead of never.
    """
    local = _get_queue()
    while True:
        try:
            job = local.get(timeout=0.5)
        except queue.Empty:
            _contained(_run_pass)
            continue
        if job is _STOP:
            local.task_done()
            break
        try:
            _handle_job(job)
        except Exception as exc:  # noqa: BLE001 - a job never kills the thread
            with _lock:
                _count("errors")
            snapshot._error(exc, "disk_gpu_shadow job")
            _reason("writer_job", f"the writer raised {type(exc).__name__}")
        finally:
            # `task_done()` FIRST, before any of this module's own accounting: the
            # queue's completion count is what `drain()` waits on, so a diagnostic that
            # cannot be written (a `_state` reset underneath this thread by the test
            # harness) must never strand it - a stranded count is a `drain()` that times
            # out for ten seconds and a writer that looks wedged when it is not.
            local.task_done()
            with _lock:
                _state["queue_depth"] = max(0, int(_state.get("queue_depth", 0)) - 1)
                _state["queue_bytes"] = max(0, int(_state.get("queue_bytes", 0))
                                            - int(job.get("bytes") or 0))
        # THE PASS IS GUARDED TOO, and it sits AFTER `task_done()` on purpose: a
        # comparison or a walk must never leave the queue's accounting one job short,
        # and it must never be the thing that stops the thread. See `_contained`.
        _contained(_run_pass)


def _name_pair(key) -> dict:
    """The shadow path and the live path for one key, or a named refusal.

    TWO CALLS TO THE SAME NAMING CODE, and that is the check. `plan_slot_naming` is
    what the live store writes under (root, shard, filename, digest);
    `build_slot_path` under the shadow root must produce the SAME `<shard>/<filename>`
    or the pair cannot be matched by name - which is the whole premise of the feature,
    so it is verified rather than assumed. Two sites that compute one field
    independently is how files become un-findable (`disk_spill.plan_slot_naming`'s own
    words), so if they ever disagree nothing is written and the counter says so.
    """
    named = disk_spill.plan_slot_naming(key)
    if not named["ok"]:
        return {"ok": False, "kind": "naming",
                "reason": "the live store's own naming refused this key: "
                          f"{named['reason']}"}
    shadowed = disk_layout.build_slot_path(_state["root"], key,
                                           named["geometry_digest"])
    if not shadowed["ok"]:
        return {"ok": False, "kind": "naming",
                "reason": "the shadow root refused the same name: "
                          f"{shadowed['reason']}"}
    relative = os.path.join(named["shard"], named["filename"])
    if os.path.relpath(shadowed["path"], _state["root"]) != relative:
        return {"ok": False, "kind": "name_mismatch",
                "reason": f"the live name ({relative}) and the shadow name "
                          f"({os.path.relpath(shadowed['path'], _state['root'])}) are "
                          "not the same file: two naming sites disagree, so pairing "
                          "them by name would be a guess"}
    width = int(named["bytes_per_file"])
    return {"ok": True, "shadow_path": shadowed["path"],
            "live_path": named["path"], "name": named["filename"], "width": width}


def _handle_job(job) -> None:
    """One queued block, on the writer thread. The whole write, then the pair waits."""
    key = job.get("key")
    blob = job.get("blob")
    with _lock:
        if (time.monotonic() - float(job.get("queued_ts") or 0.0)) \
                > JOB_MAX_AGE_SECONDS:
            _count("refused_stale_job")
            return
    named = _name_pair(key)
    if not named["ok"]:
        with _lock:
            _count(f"refused_{named['kind']}")
        _reason(named["kind"], named["reason"])
        return
    if named["width"] != len(blob):
        # The verifier compared a blob of THIS width, so a disagreement means the
        # fingerprint moved under us between the comparison and the write. Writing
        # would put a slot-shaped file under a name whose width is a different number,
        # which is a mislabeled file - the exact thing the width check exists to stop.
        with _lock:
            _count("refused_slot_width")
        _reason("slot_width", f"the copied bytes are {len(blob)} B but this rank's "
                              f"slot is {named['width']} B: the fingerprint changed "
                              "after the comparison, so these bytes are not what the "
                              "verdict described")
        return
    written = disk_layout.write_committed_file(named["shadow_path"], blob, sync=False)
    if not written["ok"]:
        with _lock:
            _count("refused_write_failed")
        _reason("write_failed", f"{named['name']}: {written['reason']}")
        return
    with _lock:
        if written.get("duplicate"):
            _count("duplicates")
        else:
            _count("written")
            _count("bytes_written", int(written.get("write_bytes") or 0))
            _state["estimated_files"] = int(_state["estimated_files"]) + 1
            _state["estimated_bytes"] = int(_state["estimated_bytes"]) \
                + int(written.get("write_bytes") or 0)
    # THE VERIFIED SIZE, after the rename: the write path's own accounting says how
    # many bytes it committed, and this asks the FILE. A disagreement is a witness, not
    # a reason to stop: the pair goes on to the comparator, which will report the size
    # difference as a differing pair.
    size = _file_size(named["shadow_path"])
    if size != len(blob):
        with _lock:
            _count("committed_size_mismatch")
        _reason("committed_size_mismatch",
                f"{named['name']}: the committed shadow file is {size} B, not "
                f"{len(blob)} B")
    _append_pending(named)


def _file_size(path) -> int | None:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return None


def _append_pending(named: dict) -> None:
    """Queue the pair for comparison, on the writer thread, shedding the oldest."""
    with _lock:
        entry = {"name": named["name"], "shadow_path": named["shadow_path"],
                 "live_path": named["live_path"], "width": int(named["width"]),
                 "since": time.monotonic()}
        pending = _state["pending"]
        pending.append(entry)
        while len(pending) > MAX_PENDING_PAIRS:
            dropped = pending.pop(0)
            _count("pairs_dropped_overflow")
            _note_once("pending-overflow",
                       f"[disk_gpu_shadow] the pending-comparison list is at its "
                       f"{MAX_PENDING_PAIRS}-pair cap; the OLDEST pair "
                       f"({dropped['name']}) was dropped, so the newest pairs are "
                       "still compared")


# --- THE COMPARISON (writer thread; bounded twice over) -----------------------
def _run_pass() -> dict:
    """One bounded compare + retention pass. NEVER called from the engine's thread.

    The lock is not paranoia: `run_retention()` is also a test/shutdown helper, and a
    pass running twice at once would double-count its own unlinks.
    """
    with _pass_lock:
        now = time.monotonic()
        last = _state["last_pass_mono"]
        if last is not None and (now - last) < PASS_INTERVAL_SECONDS:
            return {"reason": "not yet"}
        _state["last_pass_mono"] = now
        compared = _compare_pending()
        pruned = {"removed": 0, "reason": "not due"}
        if _retention_due():
            pruned = run_retention()
        return {"compared": compared, "removed": pruned.get("removed", 0),
                "reason": pruned.get("reason")}


def _retention_due() -> bool:
    """Is the directory over a cap? The caps use RUNNING figures on purpose.

    `estimated_files`/`estimated_bytes` are incremented per committed write, so the
    "am I over cap?" question costs nothing and needs no walk; the walk CORRECTS them
    (a file deleted by hand, or by another boot, is reflected at the next pass). A cap
    that is only discovered by walking is a cap that is exceeded until someone looks -
    and conversely, a pass that walked every few seconds forever would put periodic
    disk I/O on a live boot to maintain a diagnostic that is under its cap. So: walk
    when the running figures say we are over, or when we have written something and
    never measured. Nothing else.
    """
    over_estimate = (int(_state["estimated_files"]) > MAX_SHADOW_FILES
                     or int(_state["estimated_bytes"]) > MAX_SHADOW_BYTES)
    over_measured = (_state["measured_files"] is not None
                     and (int(_state["measured_files"]) > MAX_SHADOW_FILES
                          or int(_state["measured_bytes"] or 0) > MAX_SHADOW_BYTES))
    never_measured = (_state["measured_files"] is None
                      and int(_state["estimated_files"]) > 0)
    if not (over_estimate or over_measured):
        return bool(never_measured)
    # THE ANTI-THRASH FLOOR. A pass that is over cap and cannot get under it (a walk
    # that truncated, or a deletion the validator refused) must not turn into a walk on
    # every 0.25 s pass: `RETENTION_INTERVAL_SECONDS` is the minimum gap between two
    # walks that were both asked for by a cap.
    last = _state["last_retention_mono"]
    return bool(last is None
                or (time.monotonic() - last) >= RETENTION_INTERVAL_SECONDS)


def _compare_pending() -> int:
    """Compare at most `MAX_PAIRS_PER_PASS` pairs. Returns how many were decided."""
    with _lock:
        batch = _state["pending"][:MAX_PAIRS_PER_PASS]
        del _state["pending"][:len(batch)]
    decided = 0
    for entry in batch:
        verdict = _compare_pair(entry)
        if verdict == "waiting":
            with _lock:
                _count("pairs_requeued")
                if (time.monotonic() - float(entry["since"])) \
                        > PAIR_MAX_AGE_SECONDS:
                    _count("pairs_unpaired_expired")
                else:
                    _state["pending"].append(entry)
            continue
        decided += 1
    return decided


def _compare_files(left: str, right: str, limit: int) -> dict:
    """Where two files first differ, bounded by `limit` bytes. Never raises.

    Chunked so the compare is `memcmp` in C and the only Python byte loop is inside
    the ONE chunk that differs - the same shape as `disk_shadow._first_difference`,
    for the same reason: an offset is the answer a reader acts on, and reading both
    whole files into RAM on a writer thread is not the price of an offset.

    WHICH SIDE IS MISSING matters, so it is named: a missing LIVE file is "not yet, or
    evicted" and belongs in the retry window, while a missing SHADOW file is our own
    evidence going away (retention shed it, or someone cleared the directory) and
    retrying it for `PAIR_MAX_AGE_SECONDS` would fill the pending list with pairs that
    can never be decided - and count them as the live store's absence, which is a lie.
    """
    size_left, size_right = _file_size(left), _file_size(right)
    if size_left is None:
        return {"paired": False, "missing": "both" if size_right is None else "shadow"}
    if size_right is None:
        return {"paired": False, "missing": "live"}
    scan = max(0, min(int(limit), size_left, size_right, MAX_PAIR_SCAN_BYTES))
    offset, scanned = None, 0
    try:
        with open(left, "rb") as handle_left, open(right, "rb") as handle_right:
            while scanned < scan:
                want = min(VERIFY_CHUNK_BYTES, scan - scanned)
                chunk_left = handle_left.read(want)
                chunk_right = handle_right.read(want)
                if chunk_left != chunk_right:
                    for position in range(min(len(chunk_left), len(chunk_right))):
                        if chunk_left[position] != chunk_right[position]:
                            offset = scanned + position
                            break
                    scanned += len(chunk_left)
                    break
                scanned += len(chunk_left)
                if not chunk_left or not chunk_right:
                    break
    except OSError as exc:
        return {"paired": False, "reason": f"a file could not be read "
                                           f"({type(exc).__name__})"}
    size_disagree = size_left != size_right
    if offset is None and size_disagree:
        # Identical for every byte they both have: the first difference is where the
        # shorter one ends. `pairs_size_disagree` names it separately because a
        # truncated file and a corrupted byte are different faults.
        offset = min(size_left, size_right)
    return {"paired": True, "identical": offset is None,
            "first_difference": offset, "scanned_bytes": scanned,
            "size_disagree": size_disagree,
            "shadow_bytes": size_left, "live_bytes": size_right}


def _compare_pair(entry: dict) -> str:
    """Decide one pair, report that the live file has not appeared, or drop it."""
    result = _compare_files(entry["shadow_path"], entry["live_path"],
                            entry["width"])
    with _lock:
        missing = result.get("missing")
        if missing in ("shadow", "both"):
            # OUR file is gone: retention shed it (it is disposable evidence, and the
            # cap can outrun the comparison), or the directory was cleared by hand.
            # This is not the store's absence and must never be counted as such, and
            # re-queueing it would hold a slot that can never be decided.
            _count("pairs_shadow_missing")
            return "dropped"
        if not result["paired"] or missing == "live":
            # The live file is missing (or unreadable): normal for milliseconds, since
            # the spill's writer thread is a different thread with its own queue.
            # `since` decides "not yet" from "never" in `_compare_pending`.
            return "waiting"
        _count("pairs_checked")
        if result["identical"]:
            _count("pairs_identical")
        else:
            _count("pairs_differing")
            if result["size_disagree"]:
                _count("pairs_size_disagree")
            _state["last_difference"] = {
                "name": entry["name"], "first_difference": result["first_difference"],
                "scanned_bytes": result["scanned_bytes"],
                "shadow_bytes": result["shadow_bytes"],
                "live_bytes": result["live_bytes"],
                "size_disagree": bool(result["size_disagree"])}
            if entry["name"] not in _state["differing_names"]:
                _state["differing_names"].append(entry["name"])
                del _state["differing_names"][MAX_NAME_SAMPLES:]
            _note_once("differing",
                       f"[disk_gpu_shadow] a shadow file and the live store's file "
                       f"with the SAME NAME differ ({entry['name']}, first differing "
                       f"byte at {result['first_difference']}) - that is a fault in "
                       "the GPU-direct write path or a live file that changed after "
                       "the spill, not a layout question", level="error")
        return "decided"


# --- RETENTION: THIS DIRECTORY ONLY -------------------------------------------
def _walk_shadow_tree() -> tuple:
    """`([(path, name, size, mtime)], examined, truncated)` for THIS root and nothing

    else. The walk is confined by construction: it lists the root, then the
    directories directly under it, and every entry it returns was reached from a
    `realpath` that starts with the root's own `realpath`. A name that does not parse
    as one of ours is examined and SKIPPED (it is not ours to delete, and it is not
    counted as ours to keep).
    """
    root = _state["root"]
    entries: list = []
    examined = 0
    truncated = False
    if not root or not os.path.isdir(root):
        return entries, examined, truncated
    real_root = os.path.realpath(root)
    directories = [root]
    try:
        with os.scandir(root) as walker:
            for found in walker:
                if len(directories) >= MAX_WALK_DIRS:
                    truncated = True
                    break
                try:
                    if found.is_dir(follow_symlinks=False):
                        directories.append(found.path)
                except OSError:
                    truncated = True
    except OSError as exc:
        _reason("walk_failed", f"the shadow root could not be listed "
                               f"({type(exc).__name__})")
        return entries, examined, True
    for directory in directories:
        if os.path.realpath(directory) != real_root \
                and not os.path.realpath(directory).startswith(real_root + os.sep):
            continue
        try:
            with os.scandir(directory) as walker:
                for found in walker:
                    examined += 1
                    if len(entries) >= MAX_WALK_FILES:
                        truncated = True
                        break
                    try:
                        if not found.is_file(follow_symlinks=False):
                            continue
                        parsed = disk_layout.parse_path(found.name)
                        if not parsed["ok"]:
                            continue
                        info = found.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    entries.append((found.path, found.name, int(info.st_size),
                                    float(info.st_mtime)))
        except OSError as exc:
            _reason("walk_failed", f"a shadow shard directory could not be listed "
                                   f"({type(exc).__name__})")
            truncated = True
        if truncated:
            break
    return entries, examined, truncated


def run_retention() -> dict:
    """Prune the shadow directory back under its caps. Its OWN directory only.

    THE BOUNDARY IS `disk_layout.execute_unlink_plan`'s, not a promise here: it
    re-validates every path at the instant of deletion - a regular file, not a symlink,
    resolving DIRECTLY inside the one directory it is handed, with a name that parses
    as ours - and it refuses a directory that is itself a symlink. So the plan this
    function builds is checked again by the code that unlinks, and a file in the live
    store is not a candidate for two independent reasons: it is never walked, and it
    would never resolve inside a shadow directory.

    Shed order is oldest `st_mtime` first, name as the tie-break, which for disposable
    evidence is the right rule (the newest files describe the traffic under test).
    """
    report = {"removed": 0, "bytes_freed": 0, "refused": 0, "errors": 0,
              "reason": None, "truncated": False}
    root = _resolve_root()["root"]
    if not root:
        report["reason"] = _state["root_reason"] or "no shadow root"
        return report
    entries, examined, truncated = _walk_shadow_tree()
    with _lock:
        _count("walk_entries_examined", examined)
        if truncated:
            _count("walk_truncated")
        _state["measured_files"] = len(entries)
        _state["measured_bytes"] = sum(int(size) for _, _, size, _ in entries)
        _state["estimated_files"] = int(_state["measured_files"])
        _state["estimated_bytes"] = int(_state["measured_bytes"])
        _state["measured_age_seconds"] = 0.0
        _state["last_retention_mono"] = time.monotonic()
        _count("retention_passes")
    total = sum(int(size) for _, _, size, _ in entries)
    if len(entries) <= MAX_SHADOW_FILES and total <= MAX_SHADOW_BYTES:
        report["reason"] = "under both caps"
        return report
    plan = sorted(entries, key=lambda item: (item[3], item[1]))
    wanted: list = []
    keep_files, keep_bytes = len(entries), total
    for path, _name, size, _mtime in plan:
        if keep_files <= MAX_SHADOW_FILES and keep_bytes <= MAX_SHADOW_BYTES:
            break
        wanted.append(path)
        keep_files -= 1
        keep_bytes -= int(size)
    report["reason"] = f"over cap ({len(entries)} files, {total} B); " \
                       f"{len(wanted)} oldest are past the cap"
    by_directory: dict = {}
    for path in wanted[:MAX_UNLINKS_PER_PASS]:
        by_directory.setdefault(os.path.dirname(path), []).append(path)
    for directory, paths in sorted(by_directory.items()):
        verdict = disk_layout.execute_unlink_plan(
            directory, paths, dry_run=False, geometry_digest=None,
            for_capacity=False, max_unlinks=MAX_UNLINKS_PER_PASS)
        counts = (verdict or {}).get("counts") or {}
        with _lock:
            _count("retention_removed", int(counts.get("unlinked") or 0))
            _count("retention_refused", int(counts.get("refused") or 0))
            _count("retention_errors", int(counts.get("errors") or 0))
        report["removed"] += int(counts.get("unlinked") or 0)
        report["refused"] += int(counts.get("refused") or 0)
        report["errors"] += int(counts.get("errors") or 0)
        if verdict.get("ok") and counts.get("planned_bytes"):
            report["bytes_freed"] += int(counts["planned_bytes"])
    with _lock:
        _state["measured_files"] = max(0, int(_state["measured_files"] or 0)
                                       - report["removed"])
        _state["measured_bytes"] = max(0, int(_state["measured_bytes"] or 0)
                                       - report["bytes_freed"])
        _state["estimated_files"] = int(_state["measured_files"])
        _state["estimated_bytes"] = int(_state["measured_bytes"])
    if report["refused"]:
        _reason("retention_refused",
                f"{report['refused']} planned deletions were refused by the unlink "
                "validator (not ours to delete, or not a plain file inside this root)")
    if report["removed"]:
        _note_once("retention",
                   f"[disk_gpu_shadow] retention removed {report['removed']} shadow "
                   f"file(s) ({report['bytes_freed']} B) - this module's own "
                   "directory only", level="info")
    return report


# --- THE VIEW -----------------------------------------------------------------
def counters() -> dict:
    with _lock:
        return dict(_state["counters"])


def _dump_size(payload) -> int:
    """The byte count `worker_diag` will measure, with ITS OWN encoding (`separ=
    (",", ":")`, `default=str`) - the number the cap is applied to."""
    return len(json.dumps(payload, separators=(",", ":"),
                          default=str).encode("utf-8"))


def _fit(out: dict) -> dict:
    """Never hand the writer a section it would REFUSE - and say what was shed.

    THE SAME DISCIPLINE AS `disk_shadow._fit`, deliberately not the same function: that
    one trims ITS payload's shape (`regions`, its own `last_mismatch`) and counts into
    ITS OWN counters, so sharing it would couple two unrelated payloads and write one
    module's counter from the other's trim path. What is shared is the ORDER of the
    sheds and the promise: every top-level key survives every step, the COUNTERS are
    never shed, and `trimmed` says what is missing now.

      1. the reason TEXT to 48 chars (the tag at the front survives, and the tag is the
         part that names the category);
      2. the differing-name SAMPLES to two (the `pairs_differing` counter survives -
         the sample is the illustration, the number is the fact);
      3. the reason list itself, emptied.

    `last_difference` is never a trim target and never gains a key: like
    `disk_shadow.last_mismatch` it carries only numbers, and a trim step that shed it
    would make a trimmed section a different SHAPE from an untrimmed one.
    """
    trimmed = out["trimmed"] = []
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return out
    out["reasons"] = [str(line)[:TRIM_REASON_CHARS] for line in out["reasons"]]
    trimmed.append(f"reason_chars:{TRIM_REASON_CHARS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    names = list(out["differing_names"])[:TRIM_NAME_SAMPLES]
    out["differing_names"] = names
    trimmed.append(f"samples:{TRIM_NAME_SAMPLES}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    out["reasons"] = []
    trimmed.append("reasons_dropped")
    out = _counted_trim(out)
    if _dump_size(out) > MAX_SECTION_BYTES:
        # Every shed has been taken and it is still over budget, so the writer will
        # refuse the section and this key simply will not be in the rank file. Nothing
        # is left to shed without shedding a NUMBER, so the log says what the file
        # cannot: the silent absence this function exists to prevent.
        _note_once("section-over-budget",
                   f"[disk_gpu_shadow] section STILL over its byte budget after every "
                   f"trim step ({_dump_size(out)} B over {MAX_SECTION_BYTES} B): the "
                   "rank writer will refuse it, so `disk_gpu_shadow` will be absent "
                   "from diagnostics-rank.json", level="error")
    return out


def _counted_trim(out: dict) -> dict:
    """Count a trim ONCE PER SHAPE OF TRIM, and write the count back into the payload.

    `section()` is also what a polling dashboard calls, so an un-deduplicated counter
    would tick with the polling. What is worth counting is a CHANGE in what is shed.
    """
    signature = tuple(out["trimmed"])
    with _lock:
        if signature != _state["last_trim"]:
            _state["last_trim"] = signature
            _count("section_trimmed")
        out["section_trimmed"] = int(_state["counters"]["section_trimmed"])
    return out


def section() -> dict:
    """The `disk_gpu_shadow` block: EVERY KEY IS ALWAYS PRESENT.

    Counters, the enablement and who decided it, the root and the refusal, the
    comparison's verdicts, the retention's measurements, and the limits in force. The
    degraded view is built by this same function, which is what keeps the key set
    stable in every state - including the one where the feature was never enabled.
    """
    with _lock:
        out = {key: int(_state["counters"][key]) for key in _COUNTER_KEYS}
        # DEEP copy: `_fit` shortens the reason text in place, and a shallow list here
        # would trim the module's OWN stored rows.
        out["reasons"] = copy.deepcopy(_state["reasons"])
        out["differing_names"] = list(_state["differing_names"])
        out["last_difference"] = (copy.deepcopy(_state["last_difference"])
                                  if _state["last_difference"] else None)
        out["queue_depth_now"] = int(_state["queue_depth"])
        out["queue_bytes_now"] = int(_state["queue_bytes"])
        out["pending_pairs_now"] = len(_state["pending"])
        # Never a walk: `section()` runs on the step thread inside the rank writer, so
        # these are the LAST PASS's measurements, labelled as measurements.
        out["shadow_files_measured"] = _state["measured_files"]
        out["shadow_bytes_measured"] = _state["measured_bytes"]
        out["shadow_measured_age_seconds"] = _state["measured_age_seconds"]
        out["shadow_files_estimate"] = int(_state["estimated_files"])
        out["shadow_bytes_estimate"] = int(_state["estimated_bytes"])
    enable = enable_state()
    out["enabled"] = enable["enabled"]
    out["enable_source"] = enable["source"]
    out["env_var"] = config.ENV_DISK_GPU_SHADOW_WRITES
    out["env_raw"] = enable["env_raw"]
    out["env_recognised"] = enable["env_recognised"]
    out["latch"] = enable["latch"]
    out["root"] = root_state()
    out["limits"] = {"max_queue_jobs": int(MAX_QUEUE_JOBS),
                     "max_queued_bytes": int(MAX_QUEUED_BYTES),
                     "max_pairs_per_pass": int(MAX_PAIRS_PER_PASS),
                     "max_pair_scan_bytes": int(MAX_PAIR_SCAN_BYTES),
                     "pair_max_age_seconds": float(PAIR_MAX_AGE_SECONDS),
                     "max_shadow_files": int(MAX_SHADOW_FILES),
                     "max_shadow_bytes": int(MAX_SHADOW_BYTES),
                     "job_max_age_seconds": float(JOB_MAX_AGE_SECONDS)}
    out["section"] = MODULE
    out["writer"] = OWNER_TAG
    out["note"] = ("never authoritative: these files are a second copy of a block the "
                   "shadow verifier already called `equal`, written under a root that "
                   "is not the live store, so the live store stays the only thing a "
                   "restore loads; a differing pair is a fault in this write path")
    return _fit(out)


def worker_report() -> dict:
    """The rank-JSON contribution. Builds every key, always; owns no file."""
    return section()


def register_contribution() -> dict:
    """Offer this module's verdict to the rank-file writer. Idempotent.

    Registered by `disk_store.install()` beside the `disk` and `disk_shadow` sections:
    an EngineCore or an API-server process registers neither, so its rank file carries
    no `disk_gpu_shadow` key at all instead of an empty object claiming a facility that
    process does not have.
    """
    from . import worker_diag  # noqa: PLC0415 - one writer, and no import cycle
    return worker_diag.register_contribution(
        MODULE, worker_report,
        # OWNER_TAG's own claim, kept true: this reports a STATE, never an age. What
        # genuinely moves with no counter moving is the ENABLEMENT and the ROOT - the
        # POST latch can arm the writer between two writes of the file, and the
        # root-conflict refusal can appear without a single counter changing.
        owner=f"{OWNER_TAG} state, never a window",
        signal=lambda: (int(enable_state()["enabled"]),
                        int(bool(_state["env_enabled"])),
                        str(root_state()["root"]),
                        str(root_state()["conflict"])))


def status() -> dict:
    """The verb's view: enablement, root, limits, counters, the diff's verdict."""
    out = {"ok": True, "reason": None, "module": MODULE}
    out.update(section())
    return out


def drain(timeout_seconds: float = 10.0) -> dict:
    """Wait for the queue, the pending pairs AND any pass in flight. A test/shutdown
    helper, not an engine path.

    Bounded, and it reports `drained: False` rather than waiting forever: a wedged
    writer must stay visible as a wedged writer. It waits for the pairs too, because
    for this module "the work is done" means the COMPARISON ran, not the write.

    AND IT WAITS FOR AN IN-FLIGHT PASS, which is the subtle part: `_compare_pending()`
    takes a batch OUT of the pending list BEFORE it compares, so "the list is empty" is
    true for the whole duration of a comparison. A helper that stopped at the list would
    let a caller read the counters before the pass that decides them has counted - and
    that is a test that passes four times out of five, which is worse than one that
    always fails. Taking the pass lock is safe because a pass is bounded: it compares at
    most `MAX_PAIRS_PER_PASS` files up to one slot wide, and walks at most its own caps.
    """
    local = _queue
    if local is None:
        return {"drained": True, "queued": 0, "pending_pairs": 0,
                "reason": "no queue was ever created"}
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while local.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    while (_state.get("pending") or []) and time.monotonic() < deadline:
        time.sleep(0.005)
    # TAKING THE PASS LOCK is what proves no comparison is in flight: acquiring it means
    # any pass that had popped a batch has finished counting what it decided.
    pass_settled = _pass_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
    if pass_settled:
        _pass_lock.release()
    drained = (local.unfinished_tasks == 0
               and not (_state.get("pending") or []) and pass_settled)
    return {"drained": drained,
            "queued": int(local.unfinished_tasks),
            "pending_pairs": len(_state.get("pending") or []),
            "reason": (None if drained else
                       ("a comparison pass was still running at the timeout"
                        if not pass_settled else
                        "the writer was still working at the timeout"))}


def stop_writer(timeout_seconds: float = 2.0) -> dict:
    """Ask the thread to exit. Used by the test reset, and by nothing in a boot."""
    global _writer_thread
    local = _queue
    if local is None:
        return {"stopped": True, "reason": "no thread was ever started"}
    try:
        local.put_nowait(_STOP)
    except queue.Full:
        return {"stopped": False, "reason": "the queue is full; try again"}
    thread = _writer_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=max(0.0, float(timeout_seconds)))
        if thread.is_alive():
            return {"stopped": False, "reason": "the thread did not stop in time"}
    _writer_thread = None
    return {"stopped": True, "reason": None}


def reset_state_for_tests() -> None:
    """Back to a fresh boot. The thread is STOPPED, not just un-counted.

    `disk_spill`'s reset says why and this copies it: a daemon thread left draining a
    queue the next test cannot see keeps mutating the NEXT test's `_state` through jobs
    it queued, which turns a green suite into a property of file order.
    """
    global _queue, _writer_thread
    try:
        # UP TO THREE ASKS. A full queue answers `the queue is full; try again`, and
        # giving up there would leave a DAEMON thread alive on the old queue, draining
        # jobs into the NEXT test's `_state` - which is precisely the failure this
        # function exists to prevent. The thread is working through those jobs, so wait
        # for the queue and ask again; three rounds is bounded at six seconds.
        for _ in range(3):
            stop = stop_writer(timeout_seconds=2.0)
            if stop["stopped"]:
                break
            drain(timeout_seconds=2.0)
    except Exception as exc:  # noqa: BLE001 - a reset must never raise into a fixture
        snapshot._error(exc, "disk_gpu_shadow reset")
    _queue = None
    _writer_thread = None
    # IN PLACE, not a rebind: `conftest.py`'s isolation fixture snapshots this object
    # before the reset and restores it afterwards, and a rebound dict would leave the
    # fixture restoring a dictionary nobody reads.
    with _lock:
        _state.clear()
        _state.update(_fresh_state())


def _fresh_state() -> dict:
    """A brand-new `_state`, so the reset list and the declaration cannot drift."""
    return {
        "latch": False, "latch_set": False,
        "env_resolved": False, "env_enabled": False, "env_raw": None,
        "env_recognised": True,
        "root_resolved": False, "root": None, "root_source": None,
        "root_reason": None, "live_root": None,
        "notes": {}, "reasons": [], "reason_seen": set(), "last_trim": None,
        "queue_depth": 0, "queue_bytes": 0, "pending": [],
        "estimated_files": 0, "estimated_bytes": 0,
        "measured_files": None, "measured_bytes": None,
        "measured_age_seconds": None,
        "last_pass_mono": None, "last_retention_mono": None,
        "last_difference": None, "differing_names": [],
        "counters": {key: 0 for key in _COUNTER_KEYS},
    }


def view() -> dict:
    """The internal state, deep-copied, for tests and for a human reading a live
    process: a caller cannot mutate the pending list it is looking at."""
    with _lock:
        return copy.deepcopy({"pending": list(_state["pending"]),
                              "reasons": list(_state["reasons"]),
                              "root": _state["root"],
                              "root_source": _state["root_source"],
                              "live_root": _state["live_root"],
                              "differing_names": list(_state["differing_names"]),
                              "last_difference": _state["last_difference"]})
