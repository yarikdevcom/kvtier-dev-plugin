"""Stage 2 of the SSD tier: spill ONE KV slot's bytes to ONE file. Write side only.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
The RAM tier parks evicted GPU KV into pinned host-RAM slots that a worker reads
out of `cpu_kv_caches`. This module takes a slot's bytes out of those SAME tensors
and writes them to disk under a name that IS their identity
(`disk_layout.build_slot_path`:
`<root>/<shard>/kvslot-<key16>-g<group>-<geodigest12>.bin`).

IT ALSO KEEPS THE RESTORE INDEX FROM BEING STARVED. Every file this module commits
is journalled (`disk_index.record_spill`, called AFTER the `os.replace`, so a torn
write is never recorded) and every file its ceiling plan deletes is journalled too
(`record_removal`). Both happen on the writer thread and both are strictly
subordinate to the write: the SPILL is the product and the journal is an index of
it, so a journal that cannot be written costs a missed restore and never a failed
spill. `journal_failures` counts those, and the spill still reports success.

AND IT COUNTS GROUP COLLISIONS, because only a number settles how often two
KV-cache groups hash the same token chain. Two groups over one chain share a
`key16` and differ only in the group id - the discriminator the first version of
this name truncated away, which made them name ONE file with different bytes. When
this module is asked to spill a `key16` it has already spilled under a DIFFERENT
group id, it counts a `group_collision`, records both ids and (bounded) whether the
sibling file is actually on disk, and spills BOTH anyway: the names are distinct
now, so the event is a measurement and not a fault. It is readable in the snapshot's
`disk_spill`, in the `/plugins/kvtier/disk` `op:status` payload, and in a rank
file's `disk` section - no restart, no engine change, and nothing armed to see the
rate.

It is the WRITE side of a 5-stage design and THIS module has no read path, no
restore, no lookup and no re-publication of hashes - deliberately, because a hit is
resolved by a lookup in the scheduler's `cached_block_hash_to_block`, and the only
code that publishes into it is scheduler-side store-completion (the fork calls
`_process_store_completion` at `manager.py:1028-1032` on the deployed in-tree-fixed
build, at `manager.py:988-992` at the fork HEAD 10b8e07 this suite's fixture is
generated from; the method itself starts at `manager.py:1009` there and its
`cpu_block_pool._insert_block_hash` calls are at `manager.py:1026-1043`).

WHAT CHANGED IN 0.2.14: a lookup does not have to hit for a restore to happen.
`disk_restore` reads these files back WITHOUT touching that hash index, by extending
the fork's own `get_num_new_matched_tokens` answer with spans whose bytes it puts
into CPU slots the fork then DMAs - so no engine change was needed, and a restored
block helps exactly the request that paid for it instead of becoming matchable by
everyone. "Restore" is therefore live (see `disk_restore.py` and the `restore`
section of `/plugins/kvtier/disk`), and a file written here remains verifiable
(`verify_slot`) and disposable (`Deleting a file is always safe`).

THE HARD PART: THE WORKER HAS THE BYTES, THE SCHEDULER HAS THE HASH
-------------------------------------------------------------------
A content-addressed name needs both. Investigated in the order the brief demands:

1. DOES THE METADATA THAT ALREADY REACHES THE WORKER CARRY A KEY? No.
   `SimpleCPUOffloadMetadata` (`vllm/v1/simple_kv_offload/metadata.py:16-38`) is
   `load_event`, `load_gpu_blocks`, `load_cpu_blocks`, `load_event_to_reqs`,
   `store_event`, `store_gpu_blocks`, `store_cpu_blocks`, `need_flush` - the store
   half is exactly the `(store_gpu_blocks, store_cpu_blocks)` index pair plus an
   event id. `build_connector_meta` (`manager.py:494-556`) builds it from
   `prepare_store_specs`, and in LAZY mode that returns `(gpu_ids, cpu_ids, [],
   None)` (`manager.py:626`) - no hashes at all. No hash, no group id, no per-slot
   key crosses to the worker anywhere.

2. CAN THE SCHEDULER BUILD THE NAME AND THREAD IT THROUGH THAT SAME OBJECT? It
   HAS what it needs, but not a place to put it. At meta-build time the scheduler
   holds each stored block's hash (`manager.py:606` reads `node.block_hash` in lazy
   mode; `_classify_store_candidate` at `manager.py:940-954` refuses an unhashed
   block in eager mode; `make_block_hash_with_group_id` is imported at
   `manager.py:34`), so it could compute `<key16>` per slot. But the metadata
   object has no field for it, and the object is not passed by reference: the
   worker's copy is rebuilt from the wire type
   (`core/sched/output.py:279` declares `kv_connector_metadata`; the decoder is
   `msgspec.convert(result, result_type, ...)` at `v1/serial_utils.py:387`), so a
   field that is not DECLARED on the fork's dataclass is not part of the contract.
   Setting an attribute on the instance from a wrapper is a hack whose behaviour
   depends on which fields msgspec keeps - exactly the "invented channel" the
   brief forbids.

3. SO: THE CORE IS BUILT AGAINST AN INJECTED RESOLVER. (And read THE DIGEST below:
   the digest half of this analysis was WRONG as first written, and the wiring that
   replaced it has each worker compute its own. The wrong reasoning is kept because
   it is the reasoning someone will be tempted by again.) `request_spill(worker,
   slot, key=..., resolver=...)` takes the key from the caller, or from a
   `slot -> key16` callable handed in, and refuses with a counter when neither
   answers. Everything except the wiring is implemented and tested. THE SINGLE
   ENGINE CHANGE THAT UNBLOCKS THE WIRING: add fields to `SimpleCPUOffloadMetadata`
   (metadata.py) - `store_keys: list[str]`, the hex group-qualified block hash per
   stored slot, index-aligned with `store_cpu_blocks` (a slot the engine could not
   hash gets `""`, never a dropped entry); `store_fingerprint: str`, an OPTIONAL
   cross-check digest sent once per store batch (`""` means nothing, and it is
   `""` today); and `load_paths: list[str]` for the restore milestone - then
   populate them in `build_connector_meta` (manager.py), which already holds those
   hashes. THE WIRING FOR IT IS `disk_store.py`: it wraps the worker's
   `bind_connector_metadata`/`get_finished`, reads those fields off the metadata the
   worker already binds, truncates each hash to the layout's 16-hex key, and calls
   `request_spill(worker, slot, key=...)` - the NAME coming from the fingerprint
   this worker built and adopted for itself, not from the broadcast. On an
   unpatched engine the hook refuses and counts (`no_key_for_slot`) and writes
   nothing, which is the point: the same build is correct on both engines.
THE DIGEST: ONE COMPUTING SITE PER FIELD, NOT ONE PER BOOT. The first version of
this contract said "never compute the digest worker-side; the scheduler's
`store_fingerprint` is the only source of truth". That is IMPOSSIBLE, and shipping
it would have caused the exact failure it was written to prevent:
`FINGERPRINT_FIELDS` include `total_bytes_per_block` (the live box measures
11,468,800 B on PP0 and 12,902,400 B on PP1) and `rank` (`PP<n>_TP<m>`), while
there is ONE `SimpleCPUOffloadScheduler` broadcasting ONE `store_fingerprint` to
four ranks. No scheduler-computed digest can equal four different rank digests, so
every rank would write files the others could never resolve. The rule that replaces
it is narrower and actually satisfiable: each FIELD has exactly one computing site.
The rank-local fields (the measured slot width, this rank's id) are the worker's;
the rank-invariant ones are read from the config objects the worker already holds;
and every site calls the same `disk_layout.fingerprint_digest`. The metadata's
`store_fingerprint` survives as an optional CROSS-CHECK only - a non-empty one that
disagrees with this rank's own digest is a counted refusal, and the empty string
the unpatched and freshly-patched engine both send means nothing at all.

SAFETY ENVELOPE (the same posture as `blockdump.py`, which this mirrors)
------------------------------------------------------------------------
  * OFF BY DEFAULT, twice over: `VLLM_E2_TIER_DISK_ROOT` must name a real
    filesystem root (`/tmp`, `/dev/shm` and `/run` are refused by
    `disk_layout.validate_root` - a tmpfs "disk tier" spends the RAM the tier
    exists to relieve) AND the facility must be ARMED. There is no env var that
    arms it: an env var may only ever turn a dev facility OFF, which is the
    `guard_mode` rule, and the same rule `blockdump` follows.
  * NO HOOK IS INSTALLED IN THIS MODULE. It wraps no fork method and touches no
    fork object, so every entry point here returns the fork's value untouched by
    construction, not by promise. The engine hook lives in `disk_store.py`, which
    composes over the existing worker wrappers and keeps this module's shape.
  * NEVER RAISES INTO THE ENGINE. Every public entry point catches `Exception`
    (never `BaseException`) and counts. A disk tier that can crash inference is
    worse than no disk tier.
  * NEVER BLOCKS THE ENGINE THREAD. `request_spill` validates cheaply, then puts
    on a BOUNDED queue with `put_nowait` and returns. Full means a DROP and a
    counter, never growth. The bytes are read and written on this module's own
    single thread, started lazily on the first accepted job - never at import, so
    a fresh boot starts no thread and writes no byte.
  * BOUNDS EVERYWHERE. A region tensor is refused unless `shape[0] * stride(0) *
    element_size() == nbytes` (the same guard `blockdump.read_block_bytes` runs,
    and for the same reason: `ctypes.string_at` past the end of an allocation is a
    SIGSEGV that `except Exception` cannot catch, and it takes serving down). The
    slot index is bounded by the region's own block count, the file's bytes by the
    fingerprint's `total_bytes_per_block`, the queue by `MAX_QUEUE_DEPTH`, the
    write rate by `MAX_FILES_PER_WINDOW`, a sweep/walk by its caps.
  * THE IDENTITY IS NEVER OPTIONAL. A spill is named by a fingerprint whose fields
    all validate (this module's `configure`, fed worker-side by `disk_store` out of
    its own measurements), and no door here accepts a digest from elsewhere. A
    fingerprint whose region bytes do not add up to `total_bytes_per_block` is
    refused: that mismatch means the fingerprint does not describe these bytes,
    which is the one case a content-addressed name cannot survive - and on this box
    it is also the check that stops one PP stage spilling another's bytes.

WHAT IS CAPPED, AND IN WHAT UNIT
--------------------------------
  * `MAX_QUEUE_DEPTH` (jobs)             - the drop line, bounded at the object.
  * `MAX_FILES_PER_WINDOW` / `SPILL_WINDOW_SECONDS` (files / seconds) - the write
    rate, so a spill storm cannot become a disk stall for real traffic.
  * bytes per file = the fingerprint's `total_bytes_per_block`, hard-refused above
    it, and bounded by `disk_layout.MAX_BLOCK_BYTES` (1 GiB).
  * `MIN_FREE_BYTES` (bytes)             - the `statvfs` floor: refuse rather than
    fill the filesystem, reserving the slot we are about to write.
  * the capacity ceiling (bytes, configured) - LRU by `st_mtime`; `st_atime` is
    NOT usable because `/home` is mounted `relatime` here.

Stdlib only (`ctypes`, `os`, `queue`, `threading`, `time`, `hashlib`, `json`). No
torch and no vllm: the tensors are addressed through the attributes the fork's
object already exposes, read BY NAME, the `worker_diag`/`blockdump` posture. The
on-disk contract - names, the digest, the commit protocol, the eviction order, the
space gate - lives in `disk_layout.py` and is not re-implemented here.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import queue
import threading
import time

from . import blockdump, config, disk_index, disk_layout, snapshot

# --- the caps, and the unit of every one of them ----------------------------

MAX_QUEUE_DEPTH = 256                # JOBS in flight; past it, a job is DROPPED
SPILL_WINDOW_SECONDS = 60.0          # SECONDS in the write-rate window
MAX_FILES_PER_WINDOW = 200           # FILES written per window
MIN_FREE_BYTES = 1024 * 1024 * 1024  # BYTES the filesystem must keep free
MAX_REGIONS = 64                     # REGION TENSORS making up one slot
MAX_STRING_CHARS = 160
MAX_SHARD_DIRS = 288                 # shard directories a capacity walk will open
MAX_WALK_FILES = 20_000              # FILES a capacity walk will stat
EVICT_TARGET_FRACTION = 0.9          # FRACTION of the ceiling eviction aims under

# The group-collision witness: which group ids this process has seen spill each
# `key16`. Three caps, because this is a witness and not a database. The map is
# what makes a collision observable in O(1) per spill with no directory scan on the
# write path; each cap bounds what the witness itself costs.
MAX_TRACKED_DIGESTS = 4_096          # key16s remembered (oldest dropped first)
MAX_GROUPS_PER_DIGEST = 8            # group ids remembered for ONE key16
MAX_COLLISION_SAMPLES = 8            # collision EVENTS kept for a payload
MAX_COLLISION_SCAN = 4_096           # directory entries a sibling probe will read

# The arm. In-process here (a worker's truth is the arm FILE, the `blockdump`
# one-way-per-process problem again), with a bounded, stat-first read so an armed
# worker's gate costs nothing.
ARM_DEFAULT_SECONDS = 1800.0
ARM_MIN_SECONDS = 5.0
ARM_MAX_SECONDS = 7200.0
ARM_FILE = "disk-spill-arm.json"
MAX_ARM_FILE_BYTES = 4096
ARM_FILE_POLL_SECONDS = 1.0          # SECONDS a cached file verdict is trusted

WRITER_TAG = "kvtier-dev-plugin/disk-spill-1"
PROTOCOL_VERSION = 1

# Refusal reasons are a closed vocabulary: a counter per reason, so "nothing
# happened" is always attributable rather than inferred.
REFUSAL_REASONS = ("not_configured", "disarmed", "no_key_for_slot",
                   "fingerprint_missing", "fingerprint_invalid", "bad_digest",
                   "no_region_tensors", "region_not_contiguous",
                   "slot_out_of_range", "region_width_mismatch",
                   "read_failed", "no_free_space", "over_rate_window",
                   "queue_full", "write_failed", "bad_key", "bad_call")

_state: dict = {
    # configuration and the arm
    "configured": False,
    "root": None,
    "ceiling_bytes": None,
    "armed": False,
    "arm_expires_mono": None,
    "arm_expires_wall": None,
    "arm_seconds": None,
    "arm_file_verdict": None,
    "arm_file_checked_mono": None,
    # the fingerprint this process writes under (boot-constant)
    "fingerprint": None,
    "geometry_digest": None,
    # the queue and its thread
    "thread_started": 0,
    "thread_alive": False,
    "queue_depth": 0,
    "queue_depth_peak": 0,
    "window_start_mono": None,
    "window_files": 0,
    # counters
    "requests": 0,
    "accepted": 0,
    "spilled": 0,
    "duplicates": 0,
    "bytes_written": 0,
    "write_seconds": 0.0,
    "read_seconds": 0.0,
    "drops": 0,
    "refused": {reason: 0 for reason in REFUSAL_REASONS},
    "evictions": 0,
    "evicted_bytes": 0,
    "estimated_bytes": 0,
    "walk_truncated": False,
    "errors": 0,
    # the group-collision witness (see MAX_TRACKED_DIGESTS) and the journal
    "group_collisions": 0,
    "group_collision_samples": [],
    "digest_groups": {},
    "digest_groups_dropped": 0,
    "digest_group_overflow": 0,
    "sibling_probe_failures": 0,
    "journal_records": 0,
    "journal_failures": 0,
    "journal_removals": 0,
    "last_path": None,
    "last_reason": None,
    "last_error": None,
}


def reset_state_for_tests() -> None:
    """Back to a fresh boot: nothing configured, disarmed, nothing counted.

    The queue and the thread are stopped and dropped, not just un-counted: a test
    that spilled must not leave a daemon thread draining a queue the next test
    cannot see, because that thread would keep mutating the NEW test's `_state`
    through jobs it queued. The stop is sent BEFORE the references are dropped for
    exactly that reason, and it is best-effort: a test that never spilled never
    started a thread and has nothing to join.
    """
    global _queue, _writer_thread
    try:
        stop_writer(timeout_seconds=2.0)
    except Exception as exc:  # noqa: BLE001 - a reset must never raise into a fixture
        # Recorded rather than swallowed: a writer that will not stop is a fact a
        # test reader can see, not a silence.
        snapshot._error(exc, "disk_spill reset")
    _queue = None
    _writer_thread = None
    _state.update({
        "configured": False, "root": None, "ceiling_bytes": None,
        "armed": False, "arm_expires_mono": None, "arm_expires_wall": None,
        "arm_seconds": None, "arm_file_verdict": None,
        "arm_file_checked_mono": None,
        "fingerprint": None, "geometry_digest": None,
        "thread_started": 0, "thread_alive": False,
        "queue_depth": 0, "queue_depth_peak": 0,
        "window_start_mono": None, "window_files": 0,
        "requests": 0, "accepted": 0, "spilled": 0, "duplicates": 0,
        "bytes_written": 0, "write_seconds": 0.0, "read_seconds": 0.0,
        "drops": 0, "refused": {reason: 0 for reason in REFUSAL_REASONS},
        "evictions": 0, "evicted_bytes": 0, "estimated_bytes": 0,
        "walk_truncated": False, "errors": 0,
        "group_collisions": 0, "group_collision_samples": [],
        "digest_groups": {}, "digest_groups_dropped": 0,
        "digest_group_overflow": 0, "sibling_probe_failures": 0,
        "journal_records": 0, "journal_failures": 0, "journal_removals": 0,
        "last_path": None, "last_reason": None, "last_error": None,
    })


def counters() -> dict:
    """A copy for a payload. The collision WITNESS is published as a COUNT.

    `digest_groups` is the map itself, bounded at `MAX_TRACKED_DIGESTS` key16s -
    several KiB of text when full, which no polled payload should carry. Its count
    is the fact a reader wants; the capped `group_collision_samples` are the detail
    that is worth publishing.
    """
    out = dict(_state)
    out["digest_groups"] = int(len(_state["digest_groups"]))
    out["group_collision_samples"] = [dict(row)
                                      for row in _state["group_collision_samples"]]
    return out


def _fail(reason, **extra) -> dict:
    """The same failure shape as `disk_layout`: a reason, and no fake numbers.

    Local rather than imported from `disk_layout` because these payloads are this
    module's contract (they carry `reason_kind`, which the layout layer has no
    use for), and a shared helper that two modules bend in different directions is
    worse than ten duplicated lines.
    """
    out = {"ok": False, "reason": str(reason)}
    out.update(extra)
    return out


def _ok(units=None, **fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    out["units"] = dict(units or {})
    return out


# --- configuration: the two independent switches ----------------------------

def disk_root():
    """The configured store root, or None. Read per call, never at import.

    Unset is the DEFAULT and means the tier cannot exist here: no directory is
    created, no thread is started, no byte is written. The tmpfs roots are refused
    by `disk_layout.validate_root`, which is also where a relative path stops
    being an accident waiting for a different cwd.
    """
    raw = os.environ.get(config.ENV_DISK_ROOT, "").strip()
    if not raw:
        return None
    checked = disk_layout.validate_root(raw)
    if not checked["ok"]:
        return None
    return checked["absolute"]


def capacity_ceiling_bytes():
    """The configured byte ceiling, or None for "no ceiling".

    A garbage value falls back to NO ceiling rather than to 0: a 0 ceiling would
    mean "evict everything you see", which is a destructive reading of a typo.
    """
    raw = os.environ.get(config.ENV_DISK_BYTES, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def configured() -> dict:
    """What the environment says, refreshed. Never creates anything."""
    root = disk_root()
    ceiling = capacity_ceiling_bytes()
    _state["configured"] = root is not None
    _state["root"] = root
    _state["ceiling_bytes"] = ceiling
    return {"configured": root is not None, "root": root,
            "ceiling_bytes": ceiling,
            "env_root": config.ENV_DISK_ROOT, "env_ceiling": config.ENV_DISK_BYTES,
            "units": disk_layout._units("ceiling_bytes")}


def capability() -> str:
    """The three-state marker a reader distinguishes: never a bare bool.

    `not_configured`     - no root: nothing this module does can ever become work.
    `configured_disarmed`- the root exists, no arm: still zero disk work.
    `armed_and_ready`    - armed and inside its lifetime: accepted jobs spill.
    """
    if not configured()["configured"]:
        return "not_configured"
    return "armed_and_ready" if is_armed() else "configured_disarmed"


def _arm_path():
    directory = config.diagnostics_dir()
    if directory is None:
        return None
    return os.path.join(str(directory), ARM_FILE)


def is_armed() -> bool:
    """This process's arm: in-process, expired lazily, then the arm FILE.

    The expiry is checked on EVERY read (the `intentional_stop_active()` lesson -
    a lapsed arm must never answer `True` because nobody swept). The file is the
    `blockdump` one-way-per-process problem again: the verb that arms runs in the
    EngineCore, and the spill happens in a worker, so the arm crosses as one small
    bounded file. Its verdict is cached for `ARM_FILE_POLL_SECONDS` because this is
    consulted per spill request, not per diagnostic poll.
    """
    deadline = _state["arm_expires_mono"]
    if _state["armed"] and isinstance(deadline, (int, float)) \
            and time.monotonic() >= deadline:
        _state["armed"] = False
        _state["last_reason"] = "the arm expired"
    if _state["armed"]:
        return True
    now = time.monotonic()
    checked = _state["arm_file_checked_mono"]
    if isinstance(checked, (int, float)) and (now - checked) < ARM_FILE_POLL_SECONDS:
        return bool(_state["arm_file_verdict"])
    _state["arm_file_checked_mono"] = now
    verdict, reason = _read_arm_file(_arm_path())
    _state["arm_file_verdict"] = verdict
    if not verdict:
        _state["last_reason"] = reason
    return bool(verdict)


def _read_arm_file(path):
    """(armed, reason) from the arm file. Fail CLOSED: any doubt is disarmed."""
    if path is None:
        return False, (f"{config.ENV_DIAG_DIR} is not set, so no arm file can "
                       "reach a worker")
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return False, "no arm file: disarmed"
    except OSError as exc:  # noqa: BLE001 - unreadable is disarmed, never a fault
        return False, f"the arm file could not be stat'ed: {type(exc).__name__}"
    if size > MAX_ARM_FILE_BYTES:
        return False, (f"the arm file is {size} bytes (cap "
                       f"{MAX_ARM_FILE_BYTES}); refusing to trust it")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_ARM_FILE_BYTES + 1)
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a broken arm file is a disarm
        return False, f"the arm file did not parse: {type(exc).__name__}"
    if not isinstance(document, dict) or document.get("armed") is not True:
        return False, "the arm file does not say armed"
    expires = document.get("expires_ts")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        return False, "the arm file has no numeric expires_ts"
    if time.time() >= float(expires):
        return False, "the arm expired"
    return True, "armed by file"


def arm(seconds=None) -> dict:
    """Arm the spill path here, and (when a diagnostics dir exists) by file.

    Refused when nothing is configured: an armed tier with no root is a knob that
    lies, so the arm requires the root, exactly as `blockdump.arm` requires a
    channel. There is NO env var that arms this - the same rule as guard mode and
    the byte dump: a launcher script must not be able to turn a facility that
    writes real KV bytes to disk on.
    """
    wanted = ARM_DEFAULT_SECONDS
    if seconds is not None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return {"armed": False, "reason": "seconds must be a number"}
        wanted = max(ARM_MIN_SECONDS, min(ARM_MAX_SECONDS, float(seconds)))
    state = configured()
    if not state["configured"]:
        _state["refused"]["not_configured"] += 1
        return {"armed": False, "capability": "not_configured",
                "reason": f"{config.ENV_DISK_ROOT} is not set (or names a refused "
                          "path), so there is nowhere to write and arming would "
                          "only be a lie"}
    now = time.time()
    _state["armed"] = True
    _state["arm_expires_mono"] = time.monotonic() + wanted
    _state["arm_expires_wall"] = now + wanted
    _state["arm_seconds"] = wanted
    _state["arm_file_checked_mono"] = None
    written = _write_arm_file(now, wanted)
    return {"armed": True, "seconds": wanted, "capability": "armed_and_ready",
            "arm_file": written.get("path"),
            "arm_file_written": bool(written.get("written")),
            "arm_file_reason": written.get("reason"),
            "reason": None}


def disarm() -> dict:
    was = bool(_state["armed"])
    _state["armed"] = False
    _state["arm_expires_mono"] = None
    _state["arm_expires_wall"] = None
    _state["arm_seconds"] = None
    _state["arm_file_verdict"] = None
    _state["arm_file_checked_mono"] = None
    removed = _remove_arm_file()
    return {"armed": False, "was_armed": was, "capability": capability(),
            "arm_file_removed": removed,
            "reason": "spills are refused again; a queued job still lands, so "
                      "drain before trusting a zero"}


def arm_state() -> dict:
    armed = is_armed()
    remaining = None
    deadline = _state["arm_expires_wall"]
    if armed and isinstance(deadline, (int, float)):
        remaining = round(max(0.0, deadline - time.time()), 3)
    return {"armed": armed, "capability": capability(),
            "seconds_left": remaining,
            "expires_at_wallclock": (round(float(deadline), 6)
                                     if isinstance(deadline, (int, float))
                                     and armed else None),
            "default_seconds": ARM_DEFAULT_SECONDS,
            "max_seconds": ARM_MAX_SECONDS,
            "units": disk_layout._units("seconds_left", "default_seconds",
                                        "max_seconds", "expires_at_wallclock"),
            "note": ("an arm expires on its own; no env var arms this facility, "
                     "and a fresh boot does zero disk work")}


def _write_arm_file(now: float, seconds: float) -> dict:
    path = _arm_path()
    if path is None:
        return {"written": False, "reason": "no diagnostics directory"}
    document = {"version": PROTOCOL_VERSION, "writer": WRITER_TAG,
                "armed": True, "armed_ts": round(now, 6),
                "expires_ts": round(now + seconds, 6),
                "seconds": round(seconds, 3), "root": _state["root"],
                "epoch": str(snapshot.epoch()), "pid": os.getpid()}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        raw = json.dumps(document, separators=(",", ":")).encode("utf-8")
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
        return {"written": True, "path": path, "bytes": len(raw)}
    except Exception as exc:  # noqa: BLE001 - an unpublished arm is data
        snapshot._error(exc, "disk_spill arm file")
        return {"written": False, "path": path,
                "reason": f"{type(exc).__name__}: {exc}"[:MAX_STRING_CHARS]}


def _remove_arm_file() -> bool:
    path = _arm_path()
    if path is None:
        return False
    try:
        os.remove(path)
    except FileNotFoundError:
        return True
    except OSError as exc:  # noqa: BLE001 - a file we cannot remove is reported
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return False
    return True


# --- the fingerprint this process writes under ------------------------------

def configure(fingerprint) -> dict:
    """Validate and adopt the fingerprint that names every file we write.

    Boot-constant by nature (model, dtype, group layout, rank, world sizes), so it
    is set once by the wiring - and refused here, loudly, rather than discovered by
    a half-written store later. A None fingerprint CLEARS it, and with it the
    digest: without a digest this module cannot name a file at all, which is the
    same "refuse rather than guess" rule `disk_layout.validate_fingerprint` applies
    to the fields themselves.
    """
    if fingerprint is None:
        _state["fingerprint"] = None
        _state["geometry_digest"] = None
        return {"configured": False, "reason": "the fingerprint was cleared"}
    digested = disk_layout.fingerprint_digest(fingerprint)
    if not digested["ok"]:
        # Fail CLOSED: a rejected fingerprint leaves no digest behind, so a caller
        # that ignores this report cannot name a file with the PREVIOUS boot's
        # identity. Half a fingerprint is worse than none.
        _state["fingerprint"] = None
        _state["geometry_digest"] = None
        _state["refused"]["fingerprint_invalid"] += 1
        _state["last_reason"] = str(digested["reason"])[:MAX_STRING_CHARS]
        return {"configured": False, "reason": str(digested["reason"])}
    _state["fingerprint"] = digested["fingerprint"]
    _state["geometry_digest"] = digested["geometry_digest"]
    return {"configured": True,
            "geometry_digest": digested["geometry_digest"],
            "total_bytes_per_block": digested["total_bytes_per_block"],
            "num_groups": digested["num_groups"],
            "fingerprint": disk_layout.fingerprint_section(
                digested["fingerprint"]),
            "reason": None}


def geometry_digest():
    return _state["geometry_digest"]


# --- the reader: one slot's bytes, from the worker's region tensors ---------

def _region_span(tensor, slot: int, *, max_bytes: int) -> dict:
    """(base address, bytes) of slot `slot` in one region, or a refusal.

    THE CONTIGUITY GUARD IS NOT AN OPTIMISATION. The address formed below is
    `data_ptr() + slot * stride(0) * element_size()`, which assumes the region's
    slots tile the allocation with no holes and nothing between the last slot and
    the end of the buffer. That is exactly what a contiguous tensor guarantees -
    and exactly what `blockdump.read_block_bytes` checks, for the reason stated
    there: `ctypes.string_at` past the end of the allocation is a SIGSEGV, not a
    Python exception, and no `except Exception` can save a worker process. So the
    invariant is checked against the tensor's OWN byte count rather than trusted.
    """
    for name in ("data_ptr", "stride", "element_size"):
        if not callable(getattr(tensor, name, None)):
            return _fail(f"the region has no {name}() to read a slot out of")
    shape = tuple(getattr(tensor, "shape", ()) or ())
    if len(shape) < 1:
        return _fail("the region is 0-dimensional")
    blocks = int(shape[0])
    if not 0 <= slot < blocks:
        return _fail(f"slot {slot} is outside the region's {blocks} blocks")
    stride0 = tensor.stride(0)
    itemsize = tensor.element_size()
    if not isinstance(stride0, int) or not isinstance(itemsize, int) \
            or isinstance(stride0, bool) or isinstance(itemsize, bool) \
            or stride0 <= 0 or itemsize <= 0:
        return _fail("stride(0) or element_size() is not a positive int")
    bytes_per_block = int(stride0) * int(itemsize)
    if bytes_per_block > max_bytes:
        return _fail(f"this region alone is {bytes_per_block} bytes, past the "
                     f"slot's {max_bytes}-byte budget")
    nbytes = getattr(tensor, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(nbytes, bool) \
            and nbytes != blocks * bytes_per_block:
        return _fail(f"the region is not a flat run of {blocks} slots of "
                     f"{bytes_per_block} B (shape implies "
                     f"{blocks * bytes_per_block} B, nbytes is {nbytes} B); "
                     "refusing to form an address that could leave the "
                     "allocation")
    base = tensor.data_ptr()
    if not isinstance(base, int) or isinstance(base, bool) or base <= 0:
        return _fail("data_ptr() is not a usable address")
    return _ok(units=disk_layout._units("bytes_per_file"),
               base=base + slot * bytes_per_block, bytes_per_file=bytes_per_block,
               blocks=blocks)


def read_slot_bytes(caches, slot: int, total_bytes_per_block: int) -> dict:
    """One slot's bytes across every region of THIS rank, concatenated by name.

    `caches` is the worker's `cpu_kv_caches` (`worker.py:226-235`) - on the live
    box exactly ONE tensor per rank, deduped by `data_ptr` in
    `worker.py:unique_gpu_caches`, but the loop is written for N so a future fork
    shape does not silently spill half a slot. ORDER is `sorted(name)`, documented
    by the fingerprint's width check below: the same set of regions in the same
    boot always concatenates the same way, which is what makes the file's bytes a
    function of the slot and nothing else.

    The total must equal `total_bytes_per_block` EXACTLY. A slot that is smaller
    than the fingerprint claims is a fingerprint describing a different engine -
    writing it under this name would be a mislabeled file, and the mismatch is
    exactly the PP0/PP1 confusion the width field exists to catch.
    """
    if not isinstance(caches, dict) or not caches:
        return _fail("cpu_kv_caches is not a dict of region tensors (the worker "
                     "has not registered its KV caches)")
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        return _fail(f"slot must be a non-negative int, got {slot!r} (unit: slots)")
    if not isinstance(total_bytes_per_block, int) or isinstance(
            total_bytes_per_block, bool) or total_bytes_per_block <= 0:
        return _fail("total_bytes_per_block must be a positive int (unit: bytes)")
    names = sorted(str(name) for name in caches)
    if len(names) > MAX_REGIONS:
        return _fail(f"the region set has {len(names)} tensors; the cap is "
                     f"{MAX_REGIONS}")
    chunks: list[bytes] = []
    total = 0
    widths: dict = {}
    for name in names:
        try:
            span = _region_span(caches[name], slot,
                                max_bytes=total_bytes_per_block)
        except Exception as exc:  # noqa: BLE001 - a hostile tensor is a refusal
            return _fail(f"region {name!r}: inspecting it raised "
                         f"{type(exc).__name__}", region=name,
                         reason_kind="read_failed")
        if not span["ok"]:
            return _fail(f"region {name!r}: {span['reason']}",
                         region=name,
                         reason_kind=("region_not_contiguous"
                                      if "not a flat run" in str(span["reason"])
                                      else "slot_out_of_range"))
        widths[name] = span["bytes_per_file"]
        try:
            raw = ctypes.string_at(span["base"], span["bytes_per_file"])
        except Exception as exc:  # noqa: BLE001 - an unreadable region is data
            return _fail(f"region {name!r}: the host read failed: "
                         f"{type(exc).__name__}", region=name,
                         reason_kind="read_failed")
        if len(raw) != span["bytes_per_file"]:
            return _fail(f"region {name!r}: the host read returned "
                         f"{len(raw)} of {span['bytes_per_file']} bytes",
                         region=name, reason_kind="read_failed")
        chunks.append(raw)
        total += len(raw)
    if total != total_bytes_per_block:
        return _fail(f"the slot's regions total {total} bytes but the fingerprint "
                     f"declares {total_bytes_per_block}: these bytes are not what "
                     "this fingerprint describes",
                     reason_kind="region_width_mismatch",
                     widths={name: widths[name] for name in names})
    return _ok(units=disk_layout._units("bytes_per_file"),
               bytes=b"".join(chunks), bytes_per_file=total,
               regions=names, widths=widths,
               sha256=hashlib.sha256(b"".join(chunks)).hexdigest())


def sample_slot_bytes(worker, group, block_id: int, length_bytes: int) -> dict:
    """A bounded sample of one region's slot, through BLOCKDUMP'S OWN reader.

    Deliberately not a second implementation: `blockdump.read_block_bytes` is the
    path that was proven against real bytes on the live box, so a "did the spill
    and the dump agree?" question is answered by the two readers over the same
    tensor rather than by an argument. The 1 MiB per-request cap there is why this
    exists for samples while `read_slot_bytes` (whole slots) is here.
    """
    caches = getattr(worker, "cpu_kv_caches", None)
    if not isinstance(caches, dict) or not caches:
        return _fail("cpu_kv_caches is not present yet")
    found = blockdump._lookup_region(caches, group)
    if found is None:
        return _fail(f"group {group!r} is not a cpu_kv_caches region; available: "
                     f"{sorted(str(name) for name in caches)[:32]}")
    read = blockdump.read_block_bytes(found[1], block_id, 0, length_bytes)
    if not read.get("ok"):
        return _fail(str(read.get("reason")))
    return _ok(units=disk_layout._units("bytes_per_file", "length_bytes"),
               bytes=read["bytes"], sha256=read["sha256"],
               bytes_per_file=read["bytes_per_block"],
               length_bytes=read["length_bytes"], region=str(found[0]))


# --- the synchronous core: plan, read, commit, shed -------------------------

def measured_slot_bytes(caches) -> dict:
    """The width of one slot as THIS rank's tensors declare it: the sum of every
    region's `stride(0) * element_size()`.

    THIS is what a fingerprint's `total_bytes_per_block` is built FROM
    (`disk_store.build_worker_fingerprint`): the slot width is rank-local - the
    live box measures 11,468,800 B on PP0 and 12,902,400 B on PP1 - so the one
    fingerprint field that differs per rank is read per rank and never accepted
    from a broadcast. A fingerprint whose declared width disagrees with these
    tensors is refused downstream by `read_slot_bytes`, which is the check that
    stops a PP0 fingerprint from spilling PP1's bytes.
    """
    if not isinstance(caches, dict) or not caches:
        return _fail("cpu_kv_caches is not a dict of region tensors")
    names = sorted(str(name) for name in caches)
    if len(names) > MAX_REGIONS:
        return _fail(f"the region set has {len(names)} tensors; the cap is "
                     f"{MAX_REGIONS}")
    total = 0
    widths: dict = {}
    for name in names:
        tensor = caches[name]
        try:
            shape = tuple(getattr(tensor, "shape", ()) or ())
            if len(shape) < 1:
                return _fail(f"region {name!r} is 0-dimensional")
            stride0 = tensor.stride(0)
            itemsize = tensor.element_size()
        except Exception as exc:  # noqa: BLE001 - a hostile tensor is a refusal
            return _fail(f"region {name!r}: inspecting it raised "
                         f"{type(exc).__name__}")
        if isinstance(stride0, bool) or isinstance(itemsize, bool) \
                or not isinstance(stride0, int) or not isinstance(itemsize, int) \
                or stride0 <= 0 or itemsize <= 0:
            return _fail(f"region {name!r}: stride(0) or element_size() is not a "
                         "positive int")
        width = stride0 * itemsize
        if width > disk_layout.MAX_BLOCK_BYTES:
            return _fail(f"region {name!r} is {width} bytes per slot, past the "
                         f"{disk_layout.MAX_BLOCK_BYTES}-byte cap")
        widths[name] = width
        total += width
    if total <= 0 or total > disk_layout.MAX_BLOCK_BYTES:
        return _fail(f"the slot totals {total} bytes, outside (0, "
                     f"{disk_layout.MAX_BLOCK_BYTES}]")
    return _ok(units=disk_layout._units("bytes_per_file"),
               bytes_per_file=total, widths=widths, regions=names)


def plan_slot_write(caches, slot, key, fingerprint=None) -> dict:
    """Everything that can be refused, checked BEFORE any byte is written.

    Returns the validated plan (path, part path, expected bytes, digest) or a
    refusal naming the check that fired. No I/O, no allocation of the slot's
    bytes: the queue's job is a plan, so a job that cannot succeed is dropped
    before the writer thread touches the disk.

    ONE NAMING MODE, deliberately: the digest comes from the fingerprint THIS
    process was configured with, and a missing or invalid one is a refusal
    (`fingerprint_missing` / `fingerprint_invalid`). There is no "hand me a digest"
    door, because the digest is a function of rank-local facts - the slot width and
    the rank are this process's own readings (`disk_store` builds the fingerprint
    from the tensor stride and `worker_diag.rank_id()`), and accepting a digest
    from anywhere else would put two computing sites on the SAME field, which is
    precisely how files become un-findable.

    `key` must be the engine's FULL group-qualified block hash: `build_slot_path`
    splits it into the digest and the `g<group>` name component, so a key that does
    not carry the group suffix is a refusal (`bad_key`) rather than the ambiguous
    pre-fix name. That is the identity bug this closes, enforced at the door that
    writes.
    """
    adopted = fingerprint if fingerprint is not None else _state["fingerprint"]
    if adopted is None:
        return _fail("no fingerprint is configured: a file whose identity is "
                     "unknown must not be written",
                     reason_kind="fingerprint_missing")
    configured()
    digested = disk_layout.fingerprint_digest(adopted)
    if not digested["ok"]:
        return _fail(str(digested["reason"]), reason_kind="fingerprint_invalid")
    named = disk_layout.build_slot_path(_state["root"] or "", key,
                                        digested["geometry_digest"])
    if not named["ok"]:
        return _fail(str(named["reason"]), reason_kind="bad_key")
    read = read_slot_bytes(caches, slot, digested["total_bytes_per_block"])
    if not read["ok"]:
        return _fail(str(read["reason"]),
                     reason_kind=read.get("reason_kind", "read_failed"))
    return _ok(units=disk_layout._units("bytes_per_file", "ceiling_bytes",
                             "slot", "block_id", "group_id"),
               path=named["path"], part_path=named["part_path"],
               directory=named["directory"], filename=named["filename"],
               key16=named["key16"], group_id=named["group_id"],
               group_field=named["group_field"],
               shard=named["shard"],
               geometry_digest=named["geometry_digest"],
               bytes_per_file=read["bytes_per_file"], sha256=read["sha256"],
               slot=slot, blob=read["bytes"])


def _space_target(path: str) -> str:
    """The nearest EXISTING ancestor of `path`, for the free-space read.

    A first-ever spill has no store directory yet, and `statvfs` on a missing path
    reports an honest "unknown" - which, read as a refusal, would mean the tier
    could never create its own root. Walking up is not a guess: the filesystem
    under the nearest ancestor IS the filesystem the child will be created on.
    """
    current = path
    for _ in range(8):
        if os.path.isdir(current):
            return current
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return current
        current = parent
    return current


def _note_digest_group(key16, group_id, filename, shard) -> dict | None:
    """Remember which group spilled a `key16`, and count it when two disagree.

    Returns a bounded collision record when this spill's group id differs from a
    group id already recorded for the same `key16` - two KV-cache groups whose
    prefix hash landed on the same digest - and None otherwise. Called on the spill
    thread only, and does no I/O unless a collision actually happened.

    WHY A MEMORY MAP AND NOT A DIRECTORY SCAN EVERY SPILL: the rate is what we are
    measuring, so the check must cost nothing on the 99.99% of spills with no
    collision. The map is per-process and per-boot, which is exactly the window the
    operator reads (arm the tier today, read the counter tomorrow); a sibling file
    from a previous boot is not silently ignored either, because the collision
    probe looks at the shard directory when the map fires and records what is really
    on disk.

    The spill is NEVER refused by this: both files are legitimately distinct now,
    and refusing one would trade a measured collision for an unmeasured one.
    """
    groups = _state["digest_groups"].get(key16)
    if groups is None:
        while len(_state["digest_groups"]) >= MAX_TRACKED_DIGESTS:
            oldest = next(iter(_state["digest_groups"]))
            _state["digest_groups"].pop(oldest, None)
            _state["digest_groups_dropped"] += 1
        _state["digest_groups"][key16] = {int(group_id): 1}
        return None
    if group_id in groups:
        groups[int(group_id)] += 1
        return None
    # Deterministic and honest: the highest group id ALREADY recorded for this
    # digest. Computed before the new id is added, or the collision would be
    # reported as a collision between the new group and itself.
    prior = max(groups)
    if len(groups) >= MAX_GROUPS_PER_DIGEST:
        # A key16 with more distinct groups than the witness holds is itself the
        # finding; say so with a counter rather than grow the witness without bound.
        _state["digest_group_overflow"] += 1
    else:
        groups[int(group_id)] = 1
    _state["group_collisions"] += 1
    record = {"key16": key16,
              "group_id_a": int(prior), "group_id_b": int(group_id),
              "filename": str(filename)[:disk_layout.MAX_NAME_CHARS],
              "shard": str(shard)[:8],
              "sibling_on_disk": _sibling_on_disk(shard, key16, filename),
              "groups_seen": sorted(groups),
              "ts": round(time.time(), 6),
              "units": disk_layout._units("group_id_a", "group_id_b")}
    _state["group_collision_samples"] = (list(_state["group_collision_samples"])
                                         + [record])[-MAX_COLLISION_SAMPLES:]
    _state["last_reason"] = (f"group collision on {key16}: spilled for group "
                             f"{group_id} after group {prior} used the same digest")
    return record


def _sibling_on_disk(shard, key16, filename):
    """Is a DIFFERENT group's file for this `key16` actually in the shard? Runs only
    on a counted collision, so the cost is bounded by the collision rate.

    `None` means "could not look", never "is not there": an unreadable directory is
    a different fact from an absent file, and the counter must not invent the
    comfortable one.
    """
    root = _state["root"]
    if not root or not shard:
        return None
    directory = os.path.join(str(root), str(shard))
    prefix = f"{disk_layout.SLOT_FILE_PREFIX}{key16}-"
    try:
        seen = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                if seen >= MAX_COLLISION_SCAN:
                    return None
                seen += 1
                name = entry.name
                if (name.startswith(prefix) and name != str(filename)
                        and name.endswith(disk_layout.SUFFIX_DATA)):
                    return True
        return False
    except OSError as exc:
        _state["sibling_probe_failures"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return None


def _journal_write(path, size_bytes, *, removal: bool = False) -> dict:
    """Journal one committed file or one deleted file. The spill does not care.

    THE ONE RULE: this never changes the caller's verdict. A journal failure is
    counted (`journal_failures`) and reported in `last_reason`, and the bytes stay
    on disk either way - the spill is the product and this is an index of it, so the
    worst outcome here is a file the scheduler does not know about, which costs a
    recompute and nothing else.
    """
    try:
        if removal:
            recorded = disk_index.record_removal(path, size_bytes)
        else:
            recorded = disk_index.record_spill(path, size_bytes)
        if recorded.get("ok"):
            if removal:
                _state["journal_removals"] += 1
            else:
                _state["journal_records"] += 1
        else:
            _state["journal_failures"] += 1
            _state["last_reason"] = ("the spill journal did not record "
                                     f"{os.path.basename(str(path))}: "
                                     f"{str(recorded.get('reason'))[:96]}")
        return recorded
    except Exception as exc:  # noqa: BLE001 - an index never breaks a write
        _state["journal_failures"] += 1
        _state["errors"] += 1
        snapshot._error(exc, "disk_spill journal")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return {"ok": False, "reason": f"journaling raised {type(exc).__name__}"}


def spill_slot(caches, slot, key, fingerprint=None) -> dict:
    """Read one slot and commit it. The whole write path, synchronously.

    Used by the writer thread and callable directly (a test, or a future
    synchronous caller). It NEVER raises: every failure is a classified dict, and
    `final_exists: False` on a failed write is the property `disk_layout`
    guarantees and this function preserves.

    Two things happen AFTER a successful commit and neither can affect it: the group
    collision witness is updated (`group_collisions`, see
    `_note_digest_group`) and the file is journalled (`disk_index.record_spill`,
    AFTER `write_committed_file` did its `os.replace`, so a write that never landed
    is never recorded as having landed). A duplicate is not journalled: the file was
    already there, which means it was already recorded - possibly by a boot whose
    journal this one continues.
    """
    started = time.monotonic()
    plan = plan_slot_write(caches, slot, key, fingerprint)
    if not plan["ok"]:
        return {"ok": False, "reason": plan["reason"],
                "reason_kind": plan.get("reason_kind", "read_failed")}
    space = disk_layout.space_check(_space_target(plan["path"]),
                                    watermark_bytes=MIN_FREE_BYTES,
                                    extra_bytes=plan["bytes_per_file"])
    if not space["ok"] or space["below_watermark"]:
        return {"ok": False, "final_exists": False,
                "reason": (f"the filesystem is not usable for a "
                           f"{plan['bytes_per_file']}-byte write: "
                           f"{space['reason']}"),
                "reason_kind": "no_free_space"}
    written = disk_layout.write_committed_file(plan["path"], plan["blob"],
                                               sync=True)
    elapsed = round(time.monotonic() - started, 6)
    if not written["ok"]:
        return {"ok": False, "final_exists": bool(written.get("final_exists")),
                "reason": written["reason"], "reason_kind": "write_failed",
                "path": plan["path"]}
    collision = _note_digest_group(plan["key16"], plan["group_id"],
                                   plan["filename"], plan["shard"])
    journalled = None
    if not written["duplicate"]:
        journalled = _journal_write(plan["path"], int(written["write_bytes"]))
    return {"ok": True, "duplicate": bool(written["duplicate"]),
            "path": plan["path"], "key16": plan["key16"],
            "group_id": plan["group_id"], "group_field": plan["group_field"],
            "shard": plan["shard"], "filename": plan["filename"],
            "bytes_written": int(written["write_bytes"]),
            "bytes_per_file": plan["bytes_per_file"], "sha256": plan["sha256"],
            "slot": slot, "elapsed_seconds": elapsed,
            "group_collision": collision,
            "journalled": bool(journalled and journalled.get("ok")),
            "journal_reason": (journalled or {}).get("reason"),
            "directory_synced": written.get("directory_synced"),
            "reason": written.get("reason"),
            "units": disk_layout._units("bytes_written", "bytes_per_file",
                                        "group_id")}


# --- the queue and its thread ------------------------------------------------

_queue = None
_writer_thread = None
_thread_lock = threading.Lock()


def _get_queue():
    """The one bounded queue. Created on first accepted job, never at import."""
    global _queue
    with _thread_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        return _queue


def _ensure_thread():
    """Start the module's single writer thread, once, lazily. Never at import."""
    global _writer_thread
    with _thread_lock:
        alive = _writer_thread is not None and _writer_thread.is_alive()
        _state["thread_alive"] = bool(alive)
        if alive:
            return True
        try:
            thread = threading.Thread(target=_writer_loop,
                                      name="kvtier-disk-spill", daemon=True)
            thread.start()
        except Exception as exc:  # noqa: BLE001 - no thread is a refusal, not a fault
            _state["refused"]["write_failed"] += 1
            snapshot._error(exc, "disk_spill thread start")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            return False
        _writer_thread = thread
        _state["thread_started"] += 1
        _state["thread_alive"] = True
        return True


_STOP = object()


def _writer_loop() -> None:
    """The drain loop. ONE thread, and the only place a byte is written.

    A job is a plan already validated by `spill_slot`'s own re-check; the loop
    never trusts the caller's `caches` to still be the same object (the worker may
    have re-registered its caches), which is why a job carries the worker's cache
    MAPPING rather than a copy of its bytes and the read happens here.
    """
    local = _get_queue()
    while True:
        try:
            job = local.get(timeout=0.5)
        except queue.Empty:
            continue
        if job is _STOP:
            local.task_done()
            break
        try:
            _handle_job(job)
        except Exception as exc:  # noqa: BLE001 - a job never kills the thread
            _state["errors"] += 1
            snapshot._error(exc, "disk_spill job")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        finally:
            _state["queue_depth"] = max(0, int(_state["queue_depth"]) - 1)
            local.task_done()


def _handle_job(job) -> None:
    caches = job.get("caches")
    started = time.monotonic()
    result = spill_slot(caches, job.get("slot"), job.get("key"),
                        job.get("fingerprint"))
    _state["read_seconds"] = round(float(_state["read_seconds"])
                                   + (time.monotonic() - started), 6)
    if not result.get("ok"):
        kind = str(result.get("reason_kind", "write_failed"))
        if kind in _state["refused"]:
            _state["refused"][kind] += 1
        else:
            _state["refused"]["write_failed"] += 1
        _state["last_reason"] = str(result.get("reason"))[:MAX_STRING_CHARS]
        return
    if result.get("duplicate"):
        _state["duplicates"] += 1
        _state["last_reason"] = "a committed file with this identity was already " \
                                "there"
        return
    _state["spilled"] += 1
    _state["bytes_written"] = int(_state["bytes_written"]) + int(
        result["bytes_written"])
    _state["write_seconds"] = round(float(_state["write_seconds"])
                                    + float(result["elapsed_seconds"]), 6)
    _state["estimated_bytes"] = int(_state["estimated_bytes"]) + int(
        result["bytes_written"])
    _state["last_path"] = result["path"]
    _enforce_ceiling()


def _enforce_ceiling() -> None:
    """Shed files, oldest mtime first, when the estimate crosses the ceiling.

    The estimate is per-process and monotonic-increasing until a walk replaces it
    with a measurement, so the walk only happens at the moment capacity actually
    matters. `disk_layout.lru_eviction_plan` owns the ORDER (foreign digests, then
    oldest mtime); executing it here is the only destructive action in this module,
    and deleting a file is always safe: the cost is a recompute, never a wrong
    answer - there is no read path that could serve a file we deleted.
    """
    ceiling = _state["ceiling_bytes"]
    if not ceiling or int(_state["estimated_bytes"]) < int(ceiling):
        return
    root = _state["root"]
    if not root:
        return
    walk = _walk_store(root)
    if not walk["ok"]:
        _state["last_reason"] = str(walk["reason"])[:MAX_STRING_CHARS]
        return
    plan = disk_layout.lru_eviction_plan(
        walk["entries"], geometry_digest=_state["geometry_digest"],
        ceiling_bytes=int(ceiling), used_bytes=walk["used_bytes"],
        target_fraction=EVICT_TARGET_FRACTION)
    _state["estimated_bytes"] = int(walk["used_bytes"])
    _state["walk_truncated"] = bool(walk["truncated"])
    if not plan["ok"] or not plan["paths"]:
        return
    by_directory: dict = {}
    for path in plan["paths"]:
        by_directory.setdefault(os.path.dirname(path), []).append(path)
    freed = 0
    for directory, paths in by_directory.items():
        done = disk_layout.execute_unlink_plan(
            directory, paths, dry_run=False,
            geometry_digest=_state["geometry_digest"], for_capacity=True)
        if done["ok"]:
            freed += int(done["counts"].get("planned_bytes") or 0)
            # EVERY file this plan actually deleted is journalled, so the
            # scheduler's index stops offering it within one refresh. Without this
            # half the journal the index would keep offering files whose bytes the
            # ceiling already reclaimed - the one direction of index error that
            # costs a reported load failure instead of a plain recompute.
            for action in done.get("actions") or []:
                if isinstance(action, dict) and action.get("action") == "unlinked":
                    _journal_write(action.get("path"),
                                   int(action.get("size_bytes") or 0),
                                   removal=True)
    _state["evictions"] = int(_state["evictions"]) + int(plan["planned"])
    _state["evicted_bytes"] = int(_state["evicted_bytes"]) + freed
    _state["estimated_bytes"] = max(0, int(walk["used_bytes"]) - freed)


def _walk_store(root, *, max_files=None, max_shard_dirs=None) -> dict:
    """Measure the store, bounded. Every shard directory, capped files, no links.

    Two caps rather than one: shard DIRECTORIES (288 - the space is 256, so the cap
    only fires on garbage) and FILES statted. A truncated walk reports `truncated`
    so the ceiling logic knows its measurement is a floor, and a symlink inside
    the store is skipped rather than followed.

    `max_files` / `max_shard_dirs` TIGHTEN the caps for a caller that must be cheap
    (the polled snapshot section stats a handful of files; the `/disk` verb's `ls`
    uses the full caps). They may only lower them: a caller cannot widen a walk
    this module decided was the safe size.
    """
    entries: list[dict] = []
    used = 0
    truncated = False
    cap_files = MAX_WALK_FILES if max_files is None else min(MAX_WALK_FILES,
                                                             int(max_files))
    cap_dirs = MAX_SHARD_DIRS if max_shard_dirs is None \
        else min(MAX_SHARD_DIRS, int(max_shard_dirs))
    try:
        top = list(os.scandir(root))
    except OSError as exc:
        return _fail(f"the store could not be listed: {type(exc).__name__}",
                     entries=[], used_bytes=0, truncated=False)
    shard_dirs = [entry for entry in top if entry.is_dir(follow_symlinks=False)]
    if len(shard_dirs) > cap_dirs:
        truncated = True
    counted = 0
    for shard in shard_dirs[:cap_dirs]:
        try:
            with os.scandir(shard.path) as children:
                for child in children:
                    if counted >= cap_files:
                        truncated = True
                        break
                    if child.is_symlink() or not child.is_file(follow_symlinks=False):
                        continue
                    try:
                        info = child.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    parsed = disk_layout.parse_path(child.name)
                    stale_shape = parsed["classification"] == (
                        disk_layout.NAME_STALE_KEY_SHAPE)
                    if parsed["partial"] or (
                            not parsed["ok"] and not stale_shape):
                        # Only ONE non-current shape is worth measuring and
                        # shedding: the pre-group name, which is ours, occupies
                        # real bytes, and is unlinkable. Anything else that does
                        # not parse (malformed, not ours) is left alone exactly as
                        # `parse_path` says - but it must not be invisible either,
                        # or the ceiling logic under-counts a store it cannot read.
                        continue
                    counted += 1
                    used += int(info.st_size)
                    entries.append({"path": os.path.join(shard.path, child.name),
                                    "filename": child.name,
                                    "size_bytes": int(info.st_size),
                                    "mtime_ns": int(info.st_mtime_ns)})
        except OSError:
            continue
        if truncated:
            break
    return _ok(units=disk_layout._units("used_bytes", "files"),
               entries=entries, used_bytes=used, files=counted,
               truncated=truncated, reason=None)


def request_spill(caches=None, slot=None, key=None, *, worker=None,
                  fingerprint=None, resolver=None) -> dict:
    """THE entry point the store-completion hook calls. Cheap, bounded, never blocks.

    `key` is the 16-hex-char, group-qualified block hash for this slot, or None;
    when it is None the module's injected `resolver(slot)` is asked. A slot the
    resolver cannot name is REFUSED AND COUNTED (`no_key_for_slot`) - that is the
    shape of the wiring gap, made visible instead of guessed at (see the
    docstring's option 3). `caches` may be the worker's mapping directly or a
    worker object, whose `cpu_kv_caches` is read by NAME (the `worker_diag`
    posture).

    The expensive half (reading 11.4 MB and writing it) is handed to the thread, so
    the engine thread pays a dict lookup, an arm check and one `put_nowait`. A full
    queue is a DROP with a counter - the queue never grows, because an unbounded
    queue on a box that is already slow is a memory leak with a nicer name.
    """
    _state["requests"] += 1
    try:
        state = configured()
        if not state["configured"]:
            _state["refused"]["not_configured"] += 1
            return _refused("not_configured",
                            f"{config.ENV_DISK_ROOT} is not set or names a refused "
                            "path: no disk work is possible")
        if not is_armed():
            _state["refused"]["disarmed"] += 1
            return _refused("disarmed",
                            "the spill path is disarmed (arm expires by itself)")
        adopted = fingerprint if fingerprint is not None else _state["fingerprint"]
        if adopted is None:
            _state["refused"]["fingerprint_missing"] += 1
            return _refused("fingerprint_missing",
                            "no fingerprint is configured for this boot: nothing "
                            "here names a file without an identity to name it under")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            _state["refused"]["bad_call"] += 1
            return _refused("bad_call", f"slot must be a non-negative int, got "
                                        f"{slot!r}")
        resolved = key if key is not None else None
        if resolved is None and callable(resolver):
            try:
                resolved = resolver(slot)
            except Exception as exc:  # noqa: BLE001 - a resolver's fault is a refusal
                _state["refused"]["no_key_for_slot"] += 1
                snapshot._error(exc, "disk_spill resolver")
                return _refused("no_key_for_slot",
                                f"the resolver raised {type(exc).__name__}")
        if resolved is None:
            _state["refused"]["no_key_for_slot"] += 1
            return _refused("no_key_for_slot",
                            "nothing named this slot's block hash: the engine does "
                            "not carry a key to the worker (see the module "
                            "docstring's option 3)")
        region_map = caches
        if region_map is None and worker is not None:
            region_map = getattr(worker, "cpu_kv_caches", None)
        if not isinstance(region_map, dict) or not region_map:
            _state["refused"]["no_region_tensors"] += 1
            return _refused("no_region_tensors",
                            "no cpu_kv_caches mapping was handed in")
        if not _window_open(time.monotonic()):
            _state["refused"]["over_rate_window"] += 1
            return _refused("over_rate_window",
                            f"the write-rate window allows {MAX_FILES_PER_WINDOW} "
                            f"files per {SPILL_WINDOW_SECONDS}s")
        job = {"caches": region_map, "slot": int(slot), "key": resolved,
               "fingerprint": adopted,
               "queued_ts": round(time.time(), 6)}
        pending = _get_queue()
        if not _ensure_thread():
            _state["refused"]["write_failed"] += 1
            return _refused("write_failed", "the writer thread could not start")
        try:
            pending.put_nowait(job)
        except queue.Full:
            _state["drops"] = int(_state["drops"]) + 1
            _state["refused"]["queue_full"] += 1
            return _refused("queue_full",
                            f"the queue is at its {MAX_QUEUE_DEPTH}-job cap: the "
                            "job was dropped, not queued")
        _state["accepted"] += 1
        depth = int(_state["queue_depth"]) + 1
        _state["queue_depth"] = depth
        _state["queue_depth_peak"] = max(int(_state["queue_depth_peak"]), depth)
        return {"ok": True, "queued": True, "slot": int(slot),
                "path": None, "reason": None,
                "note": "the bytes are read and written on the spill thread"}
    except Exception as exc:  # noqa: BLE001 - nothing here may reach the engine
        _state["errors"] += 1
        snapshot._error(exc, "disk_spill request")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return _refused("bad_call", f"the spill request raised "
                                    f"{type(exc).__name__}")


def _refused(kind: str, reason: str) -> dict:
    _state["last_reason"] = str(reason)[:MAX_STRING_CHARS]
    return {"ok": False, "queued": False, "reason_kind": kind, "reason": reason,
            "dropped": kind == "queue_full"}


def _window_open(now: float) -> bool:
    """The write-rate window: files per rolling window, restart-empty when it lapses."""
    start = _state.get("window_start_mono")
    if not isinstance(start, (int, float)) \
            or (now - start) >= SPILL_WINDOW_SECONDS:
        _state["window_start_mono"] = now
        _state["window_files"] = 0
    if int(_state.get("window_files") or 0) >= MAX_FILES_PER_WINDOW:
        return False
    _state["window_files"] = int(_state.get("window_files") or 0) + 1
    return True


def drain(timeout_seconds: float = 5.0) -> dict:
    """Wait for the queue to empty. A test and shutdown helper, not an engine path.

    Bounded: it reports `drained: False` rather than waiting forever, because a
    wedged writer must stay visible as a wedged writer.
    """
    local = _queue
    if local is None:
        return {"drained": True, "reason": "no queue was ever created",
                "queued": 0}
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while local.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    return {"drained": local.unfinished_tasks == 0,
            "queued": int(local.unfinished_tasks),
            "reason": None if local.unfinished_tasks == 0
                      else "the queue was still working at the timeout"}


def stop_writer(timeout_seconds: float = 2.0) -> dict:
    """Ask the thread to exit. Nothing here is relied on at shutdown today."""
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
        thread.join(max(0.0, float(timeout_seconds)))
    alive = bool(thread is not None and thread.is_alive())
    _state["thread_alive"] = alive
    return {"stopped": not alive, "reason": None if not alive
            else "the thread did not exit inside the timeout"}


def verify_slot(key, fingerprint=None) -> dict:
    """Read back what we wrote, and check its length. The ONLY read path here.

    Stage 2 is allowed exactly this much reading: enough to prove a spill landed
    with the bytes we handed it. There is deliberately no "find a slot to load"
    helper, because there is nothing that could act on a hit. `key` is the FULL
    group-qualified block hash - the same identity `plan_slot_write` named the file
    with, so a verification cannot succeed against another group's bytes.
    """
    try:
        adopted = fingerprint if fingerprint is not None else _state["fingerprint"]
        if adopted is None:
            return _fail("no fingerprint is configured")
        configured()
        digested = disk_layout.fingerprint_digest(adopted)
        if not digested["ok"]:
            return _fail(str(digested["reason"]))
        named = disk_layout.build_slot_path(_state["root"] or "", key,
                                            digested["geometry_digest"])
        if not named["ok"]:
            return _fail(str(named["reason"]))
        try:
            size = os.path.getsize(named["path"])
        except FileNotFoundError:
            return _fail("the file is not there", exists=False,
                         path=named["path"], bytes_read=0)
        cap = int(digested["total_bytes_per_block"])
        if size > cap:
            return _fail(f"the file is {size} bytes, past the fingerprint's "
                         f"{cap}-byte slot width", exists=True,
                         path=named["path"], bytes_read=0)
        with open(named["path"], "rb") as handle:
            blob = handle.read(cap)
        return _ok(units=disk_layout._units("read_bytes", "bytes_on_disk"),
                   exists=True, path=named["path"], read_bytes=len(blob),
                   sha256=hashlib.sha256(blob).hexdigest(),
                   truncated=size > len(blob), bytes_on_disk=int(size))
    except Exception as exc:  # noqa: BLE001 - a verification is never a fault
        snapshot._error(exc, "disk_spill verify")
        return _fail(f"verification failed: {type(exc).__name__}")


# --- the status view ---------------------------------------------------------

def section() -> dict:
    """The snapshot view. STABLE IN SHAPE, and it reads NO files.

    `snapshot.dev_state_sections()` runs on every polled status, so this reports
    only this process's own state and the configuration - never a directory walk,
    never a statvfs, never the store's size. The counters are the point: spilled,
    refused PER REASON, bytes, seconds, drops, evictions.
    """
    state = configured()
    return {
        "capability": capability(),
        "configured": bool(state["configured"]),
        "root": state["root"],
        "ceiling_bytes": state["ceiling_bytes"],
        "armed": bool(is_armed()),
        "arm_state": arm_state(),
        "fingerprint": disk_layout.fingerprint_section(_state["fingerprint"]),
        "geometry_digest": _state["geometry_digest"],
        "requests": int(_state["requests"]),
        "accepted": int(_state["accepted"]),
        "spilled": int(_state["spilled"]),
        "duplicates": int(_state["duplicates"]),
        "bytes_written": int(_state["bytes_written"]),
        "write_seconds": round(float(_state["write_seconds"]), 6),
        "read_seconds": round(float(_state["read_seconds"]), 6),
        "drops": int(_state["drops"]),
        "refused": dict(_state["refused"],
                        **{"units": disk_layout._units(*REFUSAL_REASONS)}),
        "evictions": int(_state["evictions"]),
        "evicted_bytes": int(_state["evicted_bytes"]),
        "estimated_bytes": int(_state["estimated_bytes"]),
        "walk_truncated": bool(_state["walk_truncated"]),
        "errors": int(_state["errors"]),
        # The collision witness and the journal, next to the counters they qualify:
        # `group_collisions` is the rate the tier was armed to measure, and
        # `journal_failures` is why the restore index may lag a store that is fine.
        "group_collisions": int(_state["group_collisions"]),
        "group_collision_samples": [dict(row) for row
                                    in _state["group_collision_samples"]],
        "digest_groups_tracked": int(len(_state["digest_groups"])),
        "digest_groups_cap": MAX_TRACKED_DIGESTS,
        "digest_groups_dropped": int(_state["digest_groups_dropped"]),
        "digest_group_overflow": int(_state["digest_group_overflow"]),
        "sibling_probe_failures": int(_state["sibling_probe_failures"]),
        "journal_records": int(_state["journal_records"]),
        "journal_removals": int(_state["journal_removals"]),
        "journal_failures": int(_state["journal_failures"]),
        "journal_directory": disk_index.journal_directory(),
        "thread_started": int(_state["thread_started"]),
        "thread_alive": bool(_state["thread_alive"]),
        "queue_depth": int(_state["queue_depth"]),
        "queue_depth_peak": int(_state["queue_depth_peak"]),
        "queue_max": MAX_QUEUE_DEPTH,
        "window_files": int(_state.get("window_files") or 0),
        "caps": {"max_queue_depth": MAX_QUEUE_DEPTH,
                 "spill_window_seconds": SPILL_WINDOW_SECONDS,
                 "max_files_per_window": MAX_FILES_PER_WINDOW,
                 "min_free_bytes": MIN_FREE_BYTES,
                 "max_bytes_per_file": (
                     _state["fingerprint"]["total_bytes_per_block"]
                     if isinstance(_state["fingerprint"], dict) else None),
                 "evict_target_fraction": EVICT_TARGET_FRACTION,
                 "units": disk_layout._units(
                     "max_queue_depth", "spill_window_seconds",
                     "max_files_per_window", "min_free_bytes",
                     "max_bytes_per_file", "evict_target_fraction")},
        "naming": {"template": disk_layout.FILENAME_TEMPLATE,
                   "root": state["root"],
                   "shard": f"<key16[0:{disk_layout.SHARD_CHARS}]>",
                   "key": ("the FULL group-qualified block hash: <key16> is its "
                           "first 16 hex chars and g<group> is the 4-byte suffix "
                           "the engine appended, split by "
                           "disk_layout.split_block_key - a key too short to carry "
                           "the suffix is refused, never truncated into an "
                           "ambiguous name")},
        "inert_when_off": ("no thread, no directory created, no byte written: the "
                           "queue is not even constructed until a job is accepted"),
        "wiring": ("the store-completion hook lives in `disk_store.py`: it wraps "
                   "the worker's bind_connector_metadata/get_finished, reads the "
                   "per-slot keys off the metadata, builds THIS rank's fingerprint "
                   "from its own measured slot width and rank id (configure()), and "
                   "calls request_spill. On an UNPATCHED engine (no store_keys on "
                   "SimpleCPUOffloadMetadata, metadata.py:16-38) that hook refuses "
                   "and counts, and this module is never reached."),
        "read_path": ("none by design: a hit is resolved in the scheduler's "
                      "cached_block_hash_to_block, and only scheduler-side "
                      "store-completion publishes into it (manager.py:1017-1044)"),
        "last_path": _clip(_state["last_path"]),
        "last_reason": _clip(_state["last_reason"]),
        "last_error": _clip(_state["last_error"]),
        "units": disk_layout._units("bytes_written", "evicted_bytes",
                                    "estimated_bytes", "write_seconds",
                                    "read_seconds", "ceiling_bytes", "requests",
                                    "accepted", "spilled", "duplicates",
                                    "drops", "evictions", "errors", "queue_depth",
                                    "queue_depth_peak", "queue_max", "window_files",
                                    "thread_started", "thread_alive",
                                    "group_collisions", "digest_groups_tracked",
                                    "digest_groups_cap",
                                    "digest_groups_dropped",
                                    "digest_group_overflow",
                                    "sibling_probe_failures", "journal_records",
                                    "journal_removals", "journal_failures"),
    }


def _clip(text, chars: int = MAX_STRING_CHARS):
    if isinstance(text, str) and len(text) > chars:
        return text[:chars] + "..."
    return text
