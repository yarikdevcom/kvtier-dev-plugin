"""The SSD tier's engine wiring: when a store COMPLETES in a worker, spill it.

This is the layer that turns `disk_spill.request_spill` from a ready-but-unwired
engine into a spill that happens. It owns three things and nothing else: the
worker-side store-completion hook, the token-gated `/plugins/kvtier/disk` verb,
and the snapshot section that says whether THIS boot can spill at all.

THE ENGINE IT TOUCHES (live tree, read, never edited)
-----------------------------------------------------
`vllm/v1/simple_kv_offload/` - that directory, not
`distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/`, which does not
exist in this fork. The metadata type is `.../metadata.py`
(`SimpleCPUOffloadMetadata`), the worker handler is `.../worker.py`
(`SimpleCPUOffloadWorker`), and the engine patch that adds the fields is authored
as TWO directories with a mandatory apply order (`kvtier-disk-restore-meta`
declares the dataclass fields, `kvtier-disk-restore` fills them; the reverse order
is a `TypeError` on the first store). Nothing here imports vllm: every name is read
off an already-imported module or off the object the fork handed us, the
`worker_diag` posture.

WHERE THE WORKER LEARNS THAT A STORE COMPLETED (read out of the live fork)
-------------------------------------------------------------------------
The fork tells a worker nothing about a store except through CUDA events:

  * `bind_connector_metadata` (`worker.py:246-252`) keeps the step's metadata on
    the handler (`self._connector_metadata`) and adds `metadata.store_event` to
    `_pending_store_event_indices`. THIS is the only moment the per-slot lists
    (`store_cpu_blocks`, and on a patched engine `store_keys`) are in the worker's
    hands.
  * `wait_for_save` (`worker.py:276-303`) submits the copy tagged with that same
    `store_event` index.
  * `get_finished` (`worker.py:305-336`) polls the store stream; for every pending
    index at or below the new high-water mark it discards it from
    `_pending_store_event_indices` and records `self._completed_store_events[j] = 1`
    (`worker.py:330-334`). `build_connector_worker_meta` then drains that dict into
    the scheduler (`worker.py:338-346`).

So THE COMPLETION FACT APPEARS INSIDE `get_finished`, as the difference the fork
makes to `_completed_store_events`, while the key/slot pairing needed to name a
file existed only back at `bind_connector_metadata`. That is why this module wraps
BOTH methods and keeps its own bounded record between them:

  * the bind wrapper copies `(store_cpu_blocks, store_keys)` out of the metadata for
    the step's `store_event` - tuples of ints and hex strings, never a tensor and
    never the metadata object itself;
  * the `get_finished` wrapper diffs `_completed_store_events` around the fork's own
    call and, for each newly-completed event, hands `(slot, key16)` pairs to
    `disk_spill.request_spill`.

Composition, not replacement. Both methods already carry wrappers from this package
(`worker_diag` writes the per-rank file from `get_finished` and counts submissions
from the bind; `blockdump` answers byte dumps from `get_finished`). An install that
refused a sibling layer would make the INSTALL ORDER decide whether the rank files
exist - the `worker_diag`/`blockdump` lesson, and the reason
`engine_side._on_worker_report` installs this one LAST so it composes on top. The
fork's return value is handed back UNTOUCHED AND UNCONSUMED.

THE DIGEST: ONE COMPUTING SITE PER FIELD, NOT ONE PER BOOT
---------------------------------------------------------
An earlier draft of this contract said "never compute the digest worker-side; the
scheduler's `store_fingerprint` is the only source". That is IMPOSSIBLE, and
shipping it would have produced the exact failure it was meant to prevent.
`disk_layout.FINGERPRINT_FIELDS` include `total_bytes_per_block` (the live box
measures 11,468,800 B on PP0 and 12,902,400 B on PP1) and `rank` (`PP<n>_TP<m>`),
while there is ONE `SimpleCPUOffloadScheduler` broadcasting ONE `store_fingerprint`
to four ranks. No scheduler-computed digest can equal four different rank digests:
every rank would write under a name the others could never resolve, which is
permanently un-findable data - the one failure mode the naming scheme exists to
avoid.

What replaced it is narrower and satisfiable: **each field has exactly one computing
site.** `build_worker_fingerprint` assembles the fingerprint in the worker, and

  * the RANK-LOCAL fields are this process's own readings: `total_bytes_per_block`
    from the region tensors' strides (`disk_spill.measured_slot_bytes`), `rank` from
    `worker_diag.rank_id()` (the same place `blockdump` resolves a rank), `dtype`
    from the tensors that hold the bytes, and the group layout from
    `kv_cache_config.kv_cache_groups`;
  * the rank-INVARIANT fields are read by name off the objects the worker already
    holds (`vllm_config.model_config`, `.cache_config`, `.parallel_config`) and from
    `get_none_hash_seed()` on the already-imported `vllm.v1.core.kv_cache_utils`;
  * every site - this module, the tests, any future scheduler-side reader - calls
    the same `disk_layout.compute_geometry_digest`. One digest function, one field
    list, and nothing guessed: a field this worker cannot read is a counted
    `fingerprint_incomplete` refusal, never a made-up default.

`store_fingerprint` survives as an optional CROSS-CHECK. It arrives `""` today. An
empty value means NOTHING (it must never be read as "no fingerprint"); a non-empty
one that disagrees with this rank's own digest is a counted `fingerprint_mismatch`
refusal, because then two sites did compute the same field and something is wrong.

The property that had to survive the change is the one that ever mattered: **a
configuration change produces a different filename**, so stale bytes are
un-findable rather than mis-read. That holds worker-side exactly as before, and the
PP isolation is now carried by the width and rank fields themselves: a PP1 worker
cannot form a PP0 file's name because `total_bytes_per_block` and `rank` differ
(`tests/test_disk_store.py` pins both directions).

AN UNPATCHED ENGINE IS THE NORMAL CASE, AND IT IS VISIBLE
---------------------------------------------------------
On today's box the metadata carries none of the new fields. Every entry point here
then refuses with a reason and a counter and writes ZERO bytes:

    no_key_for_slot        `store_keys` absent, or the whole list empty
    key_count_mismatch     len(store_keys) != len(store_cpu_blocks)  (whole step)
    bad_key_text           one entry is not a usable hex hash (that ONE slot only)
    fingerprint_incomplete a fingerprint field could not be read in this worker
    fingerprint_mismatch   a non-empty store_fingerprint disagreed with this rank

and `engine_fields()` in every payload answers `present` / `absent` for this boot by
asking the already-imported metadata CLASS whether it declares `store_keys`. That
verdict - not a log grep - is how the operator knows whether the engine patch is
live. (`unknown` is the third answer, for a process that has not imported the
module, e.g. an API server. It is never collapsed into `absent`.)

SAFETY ENVELOPE (the `blockdump` posture, because this writes where that only reads)
------------------------------------------------------------------------------------
  * OFF BY DEFAULT, twice over: `VLLM_E2_TIER_DISK_ROOT` must name a real root AND
    `disk_spill` must be ARMED (no env var arms it; an arm expires). Unarmed, the
    bind wrapper's whole cost is two `os.environ.get` calls and the `get_finished`
    wrapper adds one more: no stat, no open, no copy, no queue, no thread.
  * NEVER RAISES INTO THE ENGINE. Every wrapper calls the fork's method FIRST and
    then runs its own work inside `except Exception` (never `BaseException`: a
    `KeyboardInterrupt` or a `GeneratorExit` must still escape). Every public
    function returns a dict.
  * NEVER BLOCKS AN ENGINE THREAD. Nothing on the per-step path writes a file or
    reads a tensor. The only I/O a step may do is the spill-one relay's bounded
    control-file read, at most once per `POLL_INTERVAL_SECONDS`, and only while the
    whole facility is armed with a diagnostics directory configured. Bytes reach
    disk only on `disk_spill`'s own writer thread.
  * BOUNDED MEMORY. Retained steps: `MAX_RETAINED_EVENTS` events of
    `MAX_SLOTS_PER_EVENT` slots. De-duplication memory: `MAX_HANDLED_EVENTS`.
    Per-step hand-off: `MAX_SLOTS_PER_TICK` spill requests; the rest are counted,
    never queued. Fingerprint rebuilds: at most one per `FINGERPRINT_RETRY_SECONDS`
    while one is failing, so a worker whose config is unreadable does not re-walk
    fifteen attribute paths every step.
  * CAPS ON THE VERB. `ls` walks at most `disk_spill.MAX_SHARD_DIRS`/
    `MAX_WALK_FILES` and returns at most `limit` (cap `MAX_LS_FILES`) entries
    newest-first, and a payload is trimmed to `MAX_RESPONSE_BYTES` with
    `truncated: true` rather than allowed to grow. `ls` is read-only: the only
    destructive action in this design is `disk_spill`'s own LRU eviction.
  * PP STAGES CANNOT ADDRESS EACH OTHER'S FILES. The digest carries this rank's id
    and its own measured slot width, so a PP0 name cannot be produced by a PP1
    worker; and `disk_spill.read_slot_bytes` refuses a slot whose regions do not
    total the width the fingerprint declared, so even a hand-copied name would be
    refused at the read rather than spill the wrong bytes.

WHAT THIS DOES NOT CLAIM
------------------------
Restore. There is no read path (`disk_spill`'s docstring says why: only
scheduler-side store-completion publishes into `cached_block_hash_to_block`). One
consequence deserves naming while the write path runs alone: the bytes are read on
`disk_spill`'s writer thread, so a slot reused by a later store between completion
and the write would be captured as its NEW contents under the OLD name. With no read
path that file cannot be served, so it is a wasted write, not a wrong answer - the
same argument that makes deleting a file safe. When restore lands, the spill must
become a copy-at-completion (or prove the RAM tier cannot recycle that slot that
fast); `disk_spill.spill_slot`'s synchronous core is the seam that change is made
at.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sys
import time

from . import config, disk_layout, disk_spill, snapshot

# --- the fork's names, resolved only against an already-imported module ------

TARGET_MODULE = "vllm.v1.simple_kv_offload.worker"
TARGET_CLASS = "SimpleCPUOffloadWorker"
METADATA_MODULE = "vllm.v1.simple_kv_offload.metadata"
METADATA_CLASS = "SimpleCPUOffloadMetadata"
SEED_MODULE = "vllm.v1.core.kv_cache_utils"      # get_none_hash_seed(), :147
PARALLEL_STATE_MODULE = "vllm.distributed.parallel_state"

METHOD_BIND_METADATA = "bind_connector_metadata"
METHOD_GET_FINISHED = "get_finished"
WRAPPED_METHODS = (METHOD_BIND_METADATA, METHOD_GET_FINISHED)

# A layer of THIS package on the same method (`worker_diag`, `blockdump`) is
# composed with, never refused; anything else is refused outright.
SIBLING_PACKAGE = (__package__ or "kvtier_dev_plugin") + "."

OWNER_ATTR = "__kvtier_dev_diskstore_owner__"
OWNER_TAG = "kvtier-dev-plugin/disk-store-2"
ORIGINAL_ATTR = "__kvtier_dev_diskstore_original__"

# --- the wire contract, as names --------------------------------------------

FIELD_STORE_KEYS = "store_keys"
FIELD_STORE_FINGERPRINT = "store_fingerprint"
FIELD_LOAD_PATHS = "load_paths"
INVALID_JOB_ID = -1                 # metadata.py:12
# A key the engine could not fill is the empty string, never a dropped entry
# (contract), so a placeholder costs ONE slot and not the whole step.
KEY_PLACEHOLDER = ""

# --- the caps, and the unit of every one of them ----------------------------

MAX_RETAINED_EVENTS = 32            # STEPS remembered between bind and completion
MAX_SLOTS_PER_EVENT = 4096          # SLOTS copied out of one step's metadata
MAX_COMPLETED_SCAN = 512            # ENTRIES of `_completed_store_events` we diff
MAX_SLOTS_PER_TICK = 64             # spill requests handed over per step
MAX_HANDLED_EVENTS = 256            # EVENTS remembered as already spilled
FINGERPRINT_RETRY_SECONDS = 5.0     # SECONDS between rebuilds while one is failing
MAX_MISSING_REPORTED = 16           # unreadable fields named in one report
MAX_SOURCES_PER_FIELD = 6           # candidate read paths tried for one field

POLL_INTERVAL_SECONDS = 2.0         # SECONDS between spill-one relay polls
REQUEST_TTL_SECONDS = 120.0         # SECONDS a spill-one request stays live
PROTOCOL_VERSION = 2                # 1: the one-digest-per-boot contract (impossible)
ROLE = "disk-spill-one"
WRITER_TAG = "kvtier-dev-plugin/disk-store-2"

PENDING_FILE = "disk-spill-one.json"
RECEIPT_FILE = "disk-spill-one-receipt-{rank}.json"
RECEIPT_PREFIX = "disk-spill-one-receipt-"
FILE_SUFFIX = ".json"
MAX_PENDING_FILE_BYTES = 2048
MAX_RECEIPT_FILE_BYTES = 1024
MAX_RANK_FILES = 32                 # receipt files one status read will open
MAX_SERVED_RECORDS = 64             # per-process memory of answered requests

MAX_LS_FILES = 200                  # ENTRIES in an `ls` response
MAX_LS_LIMIT_DEFAULT = 50
MAX_FILENAME_DISPLAY = 200          # CHARS of a path shown in a response
MAX_RESPONSE_BYTES = 256 * 1024     # BYTES: any verb payload's hard ceiling
MAX_STRING_CHARS = 160

DISK_OPS = ("status", "arm", "disarm", "spill-one", "ls")
MUTATING_DISK_OPS = ("arm", "disarm", "spill-one")

# This module's own refusal vocabulary. `no_key_for_slot` and `bad_digest` share a
# WORD with `disk_spill.REFUSAL_REASONS` because they are the same operational
# statement made one level higher. `delegated_refusals` is a MIRROR (the same
# refusals `disk_spill.refused` counts), never to be summed with them.
REFUSAL_REASONS = ("no_key_for_slot", "key_count_mismatch", "bad_key_text",
                   "no_store_slots", "over_slot_cap",
                   "unretained_event", "fingerprint_incomplete",
                   "fingerprint_mismatch", "bad_digest", "bad_call", "not_armed",
                   "not_configured", "no_channel", "expired", "over_tick_cap")

_state: dict = {
    # the hook
    "installed": False,
    "installs": 0,
    "reinstalls": 0,
    "uninstalls": 0,
    "class": None,
    "skip_reason": None,
    "nested": [],
    "binds": 0,
    "steps_idle": 0,
    "completions": 0,
    "events_retained": 0,
    "retained_dropped": 0,
    "retained_now": 0,
    "slots_offered": 0,
    "spill_queued": 0,
    "delegated_refusals": 0,
    "slots_capped": 0,
    "scan_skipped": 0,
    "refused": {reason: 0 for reason in REFUSAL_REASONS},
    "fallbacks": 0,
    "errors": 0,
    # the rank-local fingerprint this process names files under
    "fingerprint": None,
    "geometry_digest": None,
    "fingerprint_sources": {},
    "fingerprint_missing": [],
    "fingerprint_attempts": 0,
    "fingerprint_attempted_mono": None,
    "fingerprint_normalized_none": [],
    # the engine-field probe (worker: the last metadata bound; any process: the
    # metadata CLASS as it was imported here)
    "keys_observed_on_a_step": None,
    "fingerprint_observed_on_a_step": None,
    "load_paths_observed_on_a_step": None,
    # the spill-one relay
    "polls": 0,
    "poll_suppressed": 0,
    "last_poll_mono": None,
    "pending_stat": None,
    "pending_reads": 0,
    "pending_malformed": 0,
    "adopted": 0,
    "served": {},
    "handled": {},
    "requests_written": 0,
    "fulfilled": 0,
    "fulfil_errors": 0,
    "receipt_errors": 0,
    "pending_seq": None,
    "last_rank": None,
    "last_reason": None,
    "last_error": None,
    "last_path": None,
}


def counters() -> dict:
    """A copy for a status payload. Nothing big, nothing secret, no file reads."""
    return dict(_state)


def reset_state_for_tests() -> None:
    """Back to a fresh boot. Counters only: this module starts no thread and owns no
    queue (the writer thread belongs to `disk_spill`, whose own reset stops it)."""
    _state.update({
        "installed": False, "installs": 0, "reinstalls": 0, "uninstalls": 0,
        "class": None, "skip_reason": None, "nested": [],
        "binds": 0, "steps_idle": 0, "completions": 0,
        "events_retained": 0, "retained_dropped": 0, "retained_now": 0,
        "slots_offered": 0, "spill_queued": 0, "delegated_refusals": 0,
        "slots_capped": 0, "scan_skipped": 0,
        "refused": {reason: 0 for reason in REFUSAL_REASONS},
        "fallbacks": 0, "errors": 0,
        "fingerprint": None, "geometry_digest": None,
        "fingerprint_sources": {}, "fingerprint_missing": [],
        "fingerprint_attempts": 0, "fingerprint_attempted_mono": None,
        "fingerprint_normalized_none": [],
        "keys_observed_on_a_step": None,
        "fingerprint_observed_on_a_step": None,
        "load_paths_observed_on_a_step": None,
        "polls": 0, "poll_suppressed": 0, "last_poll_mono": None,
        "pending_stat": None, "pending_reads": 0, "pending_malformed": 0,
        "adopted": 0, "requests_written": 0, "fulfilled": 0,
        "fulfil_errors": 0, "receipt_errors": 0, "pending_seq": None,
        "last_rank": None, "last_reason": None, "last_error": None,
        "last_path": None,
    })
    _state["served"] = {}
    _state["handled"] = {}
    _retained.clear()


# The one piece of between-call memory that is not a counter: the (slot, key16)
# pairs kept between `bind_connector_metadata` and store completion. Module-level
# because it is worker state that outlives one call, BOUNDED by `_remember_event`
# because a map that grows with load is an outage with a timer. The event-dedup set
# lives in `_state["handled"]` (same shape, same cap idiom, visible in `counters()`).
_retained: dict = {}


# --- small helpers ------------------------------------------------------------

def _fail(reason, **extra) -> dict:
    out = {"ok": False, "reason": str(reason)}
    out.update(extra)
    return out


def _ok(**fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    return out


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _clip(text, chars: int = MAX_STRING_CHARS):
    if isinstance(text, str) and len(text) > chars:
        return text[:chars] + "..."
    return text


def _count(reason: str, detail: str | None = None) -> None:
    """One counted refusal, and the last one remembered. Never raises."""
    key = reason if reason in _state["refused"] else "bad_call"
    _state["refused"][key] = int(_state["refused"][key]) + 1
    if detail is not None:
        _state["last_reason"] = _clip(detail)


def _remember(key: str, value) -> None:
    """Replace a bounded dict-valued piece of state rather than mutating it.

    The `blockdump._remember_record` idiom: a test fixture that saved a shallow copy
    of `_state` gets its OWN dict back, so a leak cannot cross a test.
    """
    current = dict(_state.get(key) or {})
    current[str(value)] = round(time.time(), 6)
    cap = MAX_HANDLED_EVENTS if key == "handled" else MAX_SERVED_RECORDS
    while len(current) > cap:
        current.pop(next(iter(current)))
    _state[key] = current


def _may_touch_disk() -> bool:
    """The two-switch gate, asked on the per-step path. Never creates anything.

    Root set (an `os.environ.get`) and the arm (`disk_spill.is_armed()`, whose file
    verdict is cached for `disk_spill.ARM_FILE_POLL_SECONDS`). Unarmed this is two
    cheap lookups and at most one `stat` per second. A caller that had work to do
    counts its own refusal; the gate itself stays silent.
    """
    if disk_spill.disk_root() is None:
        return False
    return bool(disk_spill.is_armed())


# --- the engine-field probe ---------------------------------------------------

def _declares(cls, name: str) -> bool:
    """Does this metadata TYPE carry `name`? Three honest routes, one answer.

    A dataclass with a mutable default (`field(default_factory=list)`) has NO class
    attribute for it, so `hasattr` alone would call a patched engine unpatched. The
    annotation and the dataclass field list both see it; `hasattr` catches a plain
    attribute or a test double that only assigns in `__init__`.
    """
    if cls is None:
        return False
    try:
        if dataclasses.is_dataclass(cls) and any(
                field.name == name for field in dataclasses.fields(cls)):
            return True
    except Exception as exc:  # noqa: BLE001 - a broken type is an unknown, not a fault
        snapshot._error(exc, "disk_store field probe")
    annotations = getattr(cls, "__annotations__", None)
    if isinstance(annotations, dict) and name in annotations:
        return True
    return hasattr(cls, name)


def metadata_class():
    """`SimpleCPUOffloadMetadata` if this process imported it, else None."""
    module = sys.modules.get(METADATA_MODULE)
    if module is None:
        return None
    return getattr(module, METADATA_CLASS, None)


def engine_fields() -> dict:
    """Is THIS BOOT's metadata carrying the new fields? `present|absent|unknown`.

    THE QUESTION IS `store_keys`, and only `store_keys`: it is the field the spill
    cannot work without. `store_fingerprint` is a cross-check the patched engine
    sends as `""`, so its presence changes nothing about whether a spill is
    possible, and `load_paths` belongs to restore. All three are still reported,
    because "the patch is installed but half of it is missing" is exactly the state
    an operator needs to see.

    Two kinds of evidence, and the payload says which answered:
      * the TYPE, in this process: the already-imported metadata class declares
        `store_keys`. That is the "is the engine patch live" question, answerable
        before a single store happens - the whole point of reading it from a
        snapshot.
      * an INSTANCE, worker-side only: the last metadata object this process bound
        actually had the attribute (`observed_on_a_step`). Runtime proof, and the
        one an EngineCore process can never produce.
    `unknown` means this process has not imported the metadata module; it is never
    collapsed into `absent`, the same discipline as the byte dump's capability.
    """
    cls = metadata_class()
    if cls is None:
        verdict = "unknown"
        evidence = (f"{METADATA_MODULE} is not imported in this process, so the "
                    f"{METADATA_CLASS} shape is unknowable here")
        class_probe: object = "not imported"
    else:
        keys = _declares(cls, FIELD_STORE_KEYS)
        fingerprint = _declares(cls, FIELD_STORE_FINGERPRINT)
        paths = _declares(cls, FIELD_LOAD_PATHS)
        class_probe = {"store_keys": keys, "store_fingerprint": fingerprint,
                       "load_paths": paths}
        verdict = "present" if keys else "absent"
        evidence = (f"{cls.__name__} declares store_keys={keys}, "
                    f"store_fingerprint={fingerprint}, load_paths={paths}")
    observed = _state["keys_observed_on_a_step"]
    if observed is True:
        verdict = "present"
        evidence = "a metadata object this process bound carried store_keys"
    return {"verdict": verdict,
            "store_keys": verdict == "present",
            "evidence": _clip(evidence),
            "observed_on_a_step": observed,
            "fingerprint_observed_on_a_step":
                _state["fingerprint_observed_on_a_step"],
            "class_probe": class_probe,
            "note": ("`absent` means this boot's SimpleCPUOffloadMetadata has no "
                     "store_keys: the spill path refuses and writes nothing, and "
                     "that is the expected state until the engine patch is live. "
                     "`unknown` means this process cannot see the type at all. "
                     "store_fingerprint is a cross-check, not a requirement: the "
                     'engine sends "" and that means nothing.')}


def observe_metadata_fields(metadata) -> dict:
    """Record what THIS metadata object carries. Contained; reads attributes only."""
    keys = hasattr(metadata, FIELD_STORE_KEYS)
    fingerprint = hasattr(metadata, FIELD_STORE_FINGERPRINT)
    paths = hasattr(metadata, FIELD_LOAD_PATHS)
    _state["keys_observed_on_a_step"] = bool(keys)
    _state["fingerprint_observed_on_a_step"] = bool(fingerprint)
    _state["load_paths_observed_on_a_step"] = bool(paths)
    return {"ok": True, "reason": None, "store_keys": bool(keys),
            "store_fingerprint": bool(fingerprint), "load_paths": bool(paths)}


# --- the rank-local fingerprint: what THIS worker names files under ----------

# Each rank-invariant field, and where a worker may read it from, in order. These
# are the FORK'S names, verified against the live tree (config/model.py:127/204,
# config/cache.py:73/91/111/132, config/parallel.py:126/338/357,
# v1/kv_cache_interface.py:1392-1410), and a name that does not answer is a REPORTED
# missing field - never a default. Two paths for a field is a fallback for a fork
# that renamed one; more than a few would be a guess with a table on it.
CONFIG_SOURCES = {
    "model_id": (("vllm_config", "model_config", "model"),
                 ("vllm_config", "model_config", "served_model_name")),
    "model_revision": (("vllm_config", "model_config", "revision"),),
    "cache_dtype": (("vllm_config", "cache_config", "cache_dtype"),),
    "prefix_caching_hash_algo": (
        ("vllm_config", "cache_config", "prefix_caching_hash_algo"),),
    "scheduler_block_size": (("vllm_config", "cache_config", "block_size"),),
    "hash_block_size": (("vllm_config", "cache_config", "prefix_match_unit"),
                        ("vllm_config", "cache_config", "block_size")),
    "world_size": (("vllm_config", "parallel_config", "world_size"),
                   ("__world_group__",)),
    "dcp_world_size": (
        ("vllm_config", "parallel_config", "decode_context_parallel_size"),
        ("__dcp_group__",)),
    "cp_world_size": (
        ("vllm_config", "parallel_config", "prefill_context_parallel_size"),
        ("__pcp_group__",)),
}

# The rank-local fields: computed here, because these are exactly the ones a single
# scheduler cannot broadcast to four ranks.
LOCAL_FIELDS = ("total_bytes_per_block", "rank", "dtype", "group_block_sizes",
                "prefix_cacheable_group_ids", "none_hash_seed")

# A fork-legal `None` that has one stable rendering. `model_revision=None` is the
# normal state of a model checked out by revision name rather than hash, and a
# fingerprint that refused it would make the tier unusable on every ordinary boot;
# a refusal would also not be SAFER, since the field's purpose (a weight change
# under the same name must not revalidate old files) survives any stable text.
NONE_REPLACEMENTS = {"model_revision": "none"}


def _read_path(handler, path) -> tuple:
    """(value, source) for one dotted candidate, or (None, what was tried)."""
    current = handler
    walked = []
    for part in path:
        if part == "__world_group__":
            return _group_world_size("get_world_group")
        if part == "__dcp_group__":
            return _group_world_size("get_dcp_group")
        if part == "__pcp_group__":
            return _group_world_size("get_pcp_group")
        if current is None:
            return None, f"{'.'.join(walked)} is absent"
        current = getattr(current, part, None)
        walked.append(part)
        if current is None:
            return None, f"{'.'.join(walked)} is absent"
    return current, ".".join(walked)


def _group_world_size(getter: str) -> tuple:
    """A parallel group's world size, from the ALREADY-IMPORTED parallel_state.

    The rule `worker_diag.rank_id()` applies to `rank_in_group`: a worker has
    distributed state, and this module never imports the module itself. A scheduler
    or API process raises inside those getters (no `init_process_group`), which is
    the honest "not readable here", not an error.
    """
    module = sys.modules.get(PARALLEL_STATE_MODULE)
    if module is None:
        return None, f"{PARALLEL_STATE_MODULE} is not imported here"
    function = getattr(module, getter, None)
    if not callable(function):
        return None, f"{PARALLEL_STATE_MODULE}.{getter} is missing"
    try:
        group = function()
    except Exception as exc:  # noqa: BLE001 - no distributed state is a fact, not a fault
        return None, f"{getter}() refused: {type(exc).__name__}"
    world = _int(getattr(group, "world_size", None))
    if world is None or world < 1:
        return None, f"{getter}().world_size is not a positive int"
    return world, f"parallel_state.{getter}().world_size"


def _group_fields(handler) -> tuple:
    """(group_block_sizes, prefix_cacheable_group_ids, source, refusal).

    From `handler.kv_cache_config.kv_cache_groups` (`worker.py:41` keeps it and
    `register_kv_caches` reads it at `worker.py:101-106`), index-ORDERED: the
    fingerprint's group list is positional identity, so an unordered set here would
    fold two different layouts into one digest.
    """
    kv_config = getattr(handler, "kv_cache_config", None)
    groups = getattr(kv_config, "kv_cache_groups", None)
    if not groups:
        return None, None, None, ("kv_cache_config.kv_cache_groups is absent or "
                                  "empty")
    sizes = []
    cacheable = []
    for index, group in enumerate(list(groups)):
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            return None, None, None, f"group {index} has no kv_cache_spec"
        size = _int(getattr(spec, "block_size", None))
        if size is None or size < 1:
            return None, None, None, (f"group {index} kv_cache_spec.block_size is "
                                      "not a positive int")
        sizes.append(size)
        cacheable_attr = getattr(spec, "prefix_cacheable", None)
        if not isinstance(cacheable_attr, bool):
            return None, None, None, (f"group {index} kv_cache_spec."
                                      "prefix_cacheable is not a bool")
        if cacheable_attr:
            cacheable.append(index)
    if len(sizes) > disk_layout.MAX_NUM_GROUPS:
        return None, None, None, (f"{len(sizes)} groups exceeds the cap "
                                  f"{disk_layout.MAX_NUM_GROUPS}")
    return sizes, cacheable, "kv_cache_config.kv_cache_groups[*].kv_cache_spec", None


def build_worker_fingerprint(handler) -> dict:
    """The fingerprint THIS worker's bytes belong to, or the fields it could not read.

    Never a partial answer and never a default: a missing field is reported by name
    with the paths tried, and `disk_layout.validate_fingerprint` is the final judge
    (it refuses missing AND unknown fields, so this table cannot drift wider than
    the identity the file names are built from). `disk_spill.configure` is the only
    thing that turns a result here into a naming authority.
    """
    caches = getattr(handler, "cpu_kv_caches", None)
    fingerprint: dict = {}
    sources: dict = {}
    missing: list = []
    normalized_none: list = []

    # 1. the slot width: measured from these tensors, never declared elsewhere.
    measured = disk_spill.measured_slot_bytes(caches)
    if not measured["ok"]:
        missing.append({"field": "total_bytes_per_block",
                        "tried": ["cpu_kv_caches regions: "
                                  + str(measured["reason"])]})
    else:
        fingerprint["total_bytes_per_block"] = measured["bytes_per_file"]
        sources["total_bytes_per_block"] = (
            "measured: sum(region.stride(0) * region.element_size()) over "
            f"{len(measured['regions'])} region(s)")

    # 2. this rank's id, from the process that has distributed state. The one rank
    # rule lives in worker_diag, so this calls it rather than re-deriving PP_TP.
    from . import worker_diag  # noqa: PLC0415 - the one rank rule, not two
    rank, rank_source = worker_diag.rank_id()
    if rank is None:
        missing.append({"field": "rank", "tried": [str(rank_source)]})
    else:
        fingerprint["rank"] = rank
        sources["rank"] = f"worker_diag.rank_id() ({rank_source})"

    # 3. the dtype of the bytes themselves; the config's claim is the fallback.
    dtype = None
    dtype_source = None
    if isinstance(caches, dict) and caches:
        first = caches[sorted(str(name) for name in caches)[0]]
        claimed = getattr(first, "dtype", None)
        if claimed is not None:
            dtype = str(claimed)
            dtype_source = "cpu_kv_caches region .dtype (the bytes' own dtype)"
    if dtype is None:
        value, source = _read_path(handler, ("vllm_config", "model_config",
                                             "dtype"))
        if value is not None:
            dtype = str(value)
            dtype_source = f"{source} (the config's claim, not the tensor's)"
    if dtype:
        fingerprint["dtype"] = dtype
        sources["dtype"] = dtype_source
    else:
        missing.append({"field": "dtype",
                        "tried": ["cpu_kv_caches region .dtype",
                                  "vllm_config.model_config.dtype"]})

    # 4. the group layout.
    sizes, cacheable, group_source, group_refusal = _group_fields(handler)
    if group_refusal:
        missing.append({"field": "group_block_sizes", "tried": [group_refusal]})
        missing.append({"field": "prefix_cacheable_group_ids",
                        "tried": [group_refusal]})
    else:
        fingerprint["group_block_sizes"] = sizes
        fingerprint["prefix_cacheable_group_ids"] = cacheable
        sources["group_block_sizes"] = group_source
        sources["prefix_cacheable_group_ids"] = group_source

    # 5. the NONE_HASH seed, from the fork's own accessor. It is NOT a constant:
    # vllm/v1/core/kv_cache_utils.py:147 returns the PYTHONHASHSEED-derived value
    # for a non-cryptographic hasher and DEFAULT_NONE_HASH_SEED otherwise, so a
    # digest that ignored it would call two boots of one config identical. It is
    # published in the fingerprint block (the same view `disk_spill.section()`
    # publishes) because a file name cannot be reconstructed without it; the surface
    # that carries it is dev state behind the dev token.
    seed_module = sys.modules.get(SEED_MODULE)
    getter = getattr(seed_module, "get_none_hash_seed", None) if seed_module else None
    seed_reason = None
    seed = None
    if callable(getter):
        try:
            seed = getter()
        except Exception as exc:  # noqa: BLE001 - an unreadable seed is a refusal
            seed_reason = f"get_none_hash_seed() raised {type(exc).__name__}"
        if seed is not None:
            fingerprint["none_hash_seed"] = seed
            sources["none_hash_seed"] = f"{SEED_MODULE}.get_none_hash_seed()"
    if seed is None:
        missing.append({"field": "none_hash_seed",
                        "tried": [seed_reason or f"{SEED_MODULE} is not imported "
                                  "here, or has no get_none_hash_seed()"]})

    # 6. the rank-invariant config fields, by the table above.
    for field, candidates in CONFIG_SOURCES.items():
        if len(candidates) > MAX_SOURCES_PER_FIELD:
            missing.append({"field": field,
                            "tried": [f"{len(candidates)} candidate paths; the cap "
                                      f"is {MAX_SOURCES_PER_FIELD}"]})
            continue
        for path in candidates:
            value, source = _read_path(handler, path)
            if value is None and field in NONE_REPLACEMENTS:
                fingerprint[field] = NONE_REPLACEMENTS[field]
                sources[field] = (f"{source}; the fork's value is None, rendered "
                                  "as the stable text 'none'")
                normalized_none.append(field)
                break
            if value is None:
                continue
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                fingerprint[field] = value
                sources[field] = source
                break
            fingerprint[field] = str(value)
            sources[field] = f"{source} (stringified {type(value).__name__})"
            break
        else:
            missing.append({"field": field,
                            "tried": [f"{'.'.join(path)}: unreadable"
                                      for path in candidates]})

    missing = missing[:MAX_MISSING_REPORTED]
    if missing:
        return _fail("this worker cannot read every fingerprint field, so it "
                     "cannot name a file: "
                     + ", ".join(str(item["field"]) for item in missing),
                     missing_fields=missing,
                     fingerprint_partial=fingerprint,
                     note=("the digest is built from these fields; guessing one "
                           "would name a file whose identity is a lie, so the "
                           "spill is refused and counted instead"))
    checked = disk_layout.validate_fingerprint(fingerprint)
    if not checked["ok"]:
        return _fail(f"the fingerprint this worker assembled is not a valid "
                     f"identity: {checked['reason']}",
                     missing_fields=[{"field": "validation",
                                      "tried": [str(checked["reason"])]}],
                     fingerprint_partial=fingerprint)
    digest = disk_layout.compute_geometry_digest(checked["fingerprint"])
    return _ok(fingerprint=checked["fingerprint"], geometry_digest=digest,
               sources=sources, normalized_none=normalized_none,
               total_bytes_per_block=checked["total_bytes_per_block"],
               rank=checked["fingerprint"]["rank"])


def fingerprint_report() -> dict:
    """The fingerprint's STATE, for a verb or a snapshot. Publishes no seed."""
    configured = _state["fingerprint"]
    return {"configured": configured is not None,
            "geometry_digest": _state["geometry_digest"],
            "rank": (configured or {}).get("rank"),
            "total_bytes_per_block": (configured or {}).get(
                "total_bytes_per_block"),
            "dtype": (configured or {}).get("dtype"),
            "sources": dict(_state["fingerprint_sources"]),
            # Which fields are THIS rank's own, published so a reader of a digest can
            # see why four ranks have four digests for one configuration - the exact
            # misunderstanding the first version of this contract fell into.
            "rank_local_fields": list(LOCAL_FIELDS),
            "missing_fields": list(_state["fingerprint_missing"]),
            "normalized_none": list(_state["fingerprint_normalized_none"]),
            "attempts": int(_state["fingerprint_attempts"]),
            "fingerprint": disk_layout.fingerprint_section(configured)
            if isinstance(configured, dict) else None,
            "note": ("the worker builds this itself: the slot width it measures and "
                     "the rank it resolves are the fields a single scheduler cannot "
                     "broadcast to four ranks. The seed appears here in its canonical "
                     "rendering - the SAME view `disk_spill.section()` publishes, "
                     "because a file name cannot be reconstructed without it and two "
                     "views of one identity is how a naming argument starts. The "
                     "mitigation is the surface: this is dev state, behind the dev "
                     "token, loopback-only by default - and with a non-cryptographic "
                     "hasher the seed is per-process random, not a shared secret.")}


def adopt_fingerprint(handler, force: bool = False) -> dict:
    """Build this rank's fingerprint once per boot and hand it to `disk_spill`.

    `disk_spill.configure` is the only naming authority, so a spill is only ever
    possible after this has succeeded here. Retries are rate-limited while failing
    (`FINGERPRINT_RETRY_SECONDS`), because the alternative is re-walking fifteen
    attribute paths on every bound step of a worker whose config is unreadable - and
    the answer to an unreadable config is a counted refusal, not a busy loop.
    """
    now = time.monotonic()
    if _state["fingerprint"] is not None and not force:
        return _ok(adopted=True, cached=True,
                   geometry_digest=_state["geometry_digest"],
                   reason="this rank's fingerprint is already the naming authority")
    last = _state["fingerprint_attempted_mono"]
    if (not force and isinstance(last, (int, float))
            and (now - last) < FINGERPRINT_RETRY_SECONDS):
        return _fail("the fingerprint build is rate-limited after its last failure",
                     retry_in_seconds=round(FINGERPRINT_RETRY_SECONDS - (now - last),
                                            3),
                     reason_kind="fingerprint_incomplete",
                     missing_fields=list(_state["fingerprint_missing"]))
    _state["fingerprint_attempted_mono"] = now
    _state["fingerprint_attempts"] = int(_state["fingerprint_attempts"]) + 1
    built = build_worker_fingerprint(handler)
    if not built["ok"]:
        _state["fingerprint_missing"] = list(built["missing_fields"])
        _count("fingerprint_incomplete", built["reason"])
        return _fail(built["reason"], reason_kind="fingerprint_incomplete",
                     missing_fields=list(built["missing_fields"]))
    adopted = disk_spill.configure(built["fingerprint"])
    if not adopted.get("configured"):
        _state["fingerprint_missing"] = [{"field": "disk_spill.configure",
                                          "tried": [str(adopted.get("reason"))]}]
        _count("fingerprint_incomplete", str(adopted.get("reason")))
        return _fail(f"disk_spill refused this fingerprint: "
                     f"{adopted.get('reason')}",
                     reason_kind="fingerprint_incomplete")
    _state["fingerprint"] = built["fingerprint"]
    _state["geometry_digest"] = built["geometry_digest"]
    _state["fingerprint_sources"] = built["sources"]
    _state["fingerprint_normalized_none"] = built["normalized_none"]
    _state["fingerprint_missing"] = []
    return _ok(adopted=True, cached=False,
               geometry_digest=built["geometry_digest"],
               rank=built["rank"],
               total_bytes_per_block=built["total_bytes_per_block"],
               reason="this rank's fingerprint is now the naming authority")


# --- the resolver: metadata -> (slots, key16s, cross-check digest) -----------

def key16(value) -> tuple:
    """The file-name key for one engine block hash: `(key16 | None, why)`.

    `BlockHashWithGroupId` is BYTES - a digest plus a 4-byte big-endian group id
    (`kv_cache_utils.py:86-92`) - and the engine sends its hex, which is LONGER than
    the 16 hex chars a name carries. TAKING THE FIRST 16 is the layout layer's own
    definition of the key (`disk_layout.validate_key16`: "the first 16 hex chars of
    the engine's block hash"), so this is the documented truncation, not a guess:
    same engine, same hash, same 16 chars, always.

    An entry the engine could not fill arrives as the empty string, which means THAT
    SLOT has no identity: it is refused and counted on its own, and the rest of the
    step still spills. That is the difference between a placeholder (the contract's
    shape) and a peer bug (a length mismatch), which refuses the whole step.
    """
    if not isinstance(value, str):
        return None, f"a key must be a hex string, got {type(value).__name__}"
    text = value.strip().lower()
    if not text:
        return None, "the engine had no block hash for this slot"
    if len(text) < disk_layout.KEY_DIGEST_CHARS:
        return None, (f"the key is {len(text)} hex chars; a name needs "
                      f"{disk_layout.KEY_DIGEST_CHARS}")
    if any(char not in disk_layout.HEX_DIGITS for char in text):
        return None, "the key is not hexadecimal"
    return text[:disk_layout.KEY_DIGEST_CHARS], None


def resolve_store_keys(metadata) -> dict:
    """The per-slot keys for one metadata object, or a classified refusal.

    Pure: no I/O, no allocation beyond tuples, no engine import. THE ORDER IS
    PRESERVED AND THE LENGTH IS ASSERTED, because index i of `store_keys` names
    index i of `store_cpu_blocks` and nothing else can re-establish that pairing
    once the two lists are in different hands.

    Returns `{ok, slots, keys, placeholders, count, expected_digest}` - `keys` being
    16-hex strings with `None` for a slot the engine could not name, and
    `expected_digest` the cross-check value or None - or `{ok: False, reason_kind,
    reason}` with the numbers named.
    """
    if metadata is None:
        _count("bad_call", "no metadata object was handed in")
        return _fail("no connector metadata is bound in this worker",
                     reason_kind="bad_call")
    observe_metadata_fields(metadata)
    slots_raw = getattr(metadata, "store_cpu_blocks", None)
    keys_raw = getattr(metadata, FIELD_STORE_KEYS, None)
    if slots_raw is None:
        _count("bad_call", "the metadata has no store_cpu_blocks")
        return _fail("the metadata has no store_cpu_blocks to name",
                     reason_kind="bad_call")
    if not isinstance(slots_raw, (list, tuple)):
        _count("bad_call", f"store_cpu_blocks is {type(slots_raw).__name__}")
        return _fail(f"store_cpu_blocks is {type(slots_raw).__name__}, not a list "
                     "of slot ids", reason_kind="bad_call")
    count = len(slots_raw)
    if count == 0:
        # Not an error: a step with nothing to store. Counted, because "the tier
        # parked nothing" must be attributable rather than inferred.
        _count("no_store_slots", "this step stored no blocks, so there is nothing "
                                 "to spill")
        return _fail("this step stored no blocks", reason_kind="no_store_slots",
                     count=0)
    if keys_raw is None:
        _count("no_key_for_slot",
               f"store_keys is absent from this boot's metadata: the engine does "
               f"not carry a block hash to the worker ({count} slots unnamed)")
        return _fail(f"store_keys is absent from this boot's metadata: {count} "
                     f"stored slots have no key, so nothing can be named (the "
                     f"engine patch adds the field; see disk_store's docstring)",
                     reason_kind="no_key_for_slot", count=count, keys=0)
    if not isinstance(keys_raw, (list, tuple)):
        _count("key_count_mismatch",
               f"store_keys is {type(keys_raw).__name__}, not a list")
        return _fail(f"store_keys is {type(keys_raw).__name__}, not a list of hex "
                     "block hashes", reason_kind="key_count_mismatch",
                     count=count, keys=None)
    key_count = len(keys_raw)
    if key_count == 0:
        _count("no_key_for_slot",
               f"store_keys is empty for {count} stored slots: the engine says it "
               "has no keys for this step")
        return _fail(f"store_keys is empty while store_cpu_blocks holds {count} "
                     "slots: nothing can be named",
                     reason_kind="no_key_for_slot", count=count, keys=0)
    if key_count != count:
        # A PEER BUG, and the one place a "be liberal" reading would corrupt the
        # store: pairing 3 keys with 5 slots by truncation writes someone else's
        # bytes under the wrong name. Refuse the whole step, name both numbers.
        _count("key_count_mismatch",
               f"store_keys has {key_count} entries for {count} stored slots")
        return _fail(f"store_keys has {key_count} entries but store_cpu_blocks has "
                     f"{count}: the pairing cannot be recovered, so this step is "
                     f"refused whole rather than truncated",
                     reason_kind="key_count_mismatch", count=count, keys=key_count)
    if count > MAX_SLOTS_PER_EVENT:
        _count("over_slot_cap",
               f"one step stored {count} blocks; the cap is {MAX_SLOTS_PER_EVENT}")
        return _fail(f"this step stored {count} blocks; the cap is "
                     f"{MAX_SLOTS_PER_EVENT}, so the step is refused rather than "
                     f"partially remembered", reason_kind="over_slot_cap",
                     count=count)
    keys = []
    placeholders = 0
    for value in keys_raw:
        resolved, why = key16(value)
        keys.append(resolved)
        if resolved is None:
            placeholders += 1
            if value != KEY_PLACEHOLDER:
                _count("bad_key_text",
                       f"a store_keys entry is not a usable block hash: {why}")
    # THE CROSS-CHECK, not the naming source. Empty (the normal case, and what a
    # half-applied patch sends) means NOTHING; a non-empty one must be a digest and
    # must equal this rank's own, or two sites computed the same field.
    expected = None
    raw_expected = getattr(metadata, FIELD_STORE_FINGERPRINT, None)
    if isinstance(raw_expected, str) and raw_expected.strip():
        checked = disk_layout.validate_geometry_digest(raw_expected)
        if not checked["ok"]:
            _count("bad_digest", str(checked["reason"]))
            return _fail(f"store_fingerprint is not a usable digest: "
                         f"{checked['reason']}", reason_kind="bad_digest",
                         count=count)
        expected = checked["value"]
    return _ok(slots=tuple(slots_raw), keys=tuple(keys),
               placeholders=placeholders, count=count, expected_digest=expected)


# --- the between-call memory --------------------------------------------------

def _remember_event(event, pairs, digest, count) -> dict:
    """Keep one step's slot/key pairs until its store event completes. Bounded."""
    existing = int(len(_retained))
    if event in _retained:
        existing -= 1                     # a replacement is not growth
    _retained[event] = {"pairs": pairs, "digest": digest, "count": count,
                        "ts": round(time.time(), 6)}
    _state["events_retained"] = int(_state["events_retained"]) + 1
    while len(_retained) > MAX_RETAINED_EVENTS:
        oldest = next(iter(_retained))
        dropped = _retained.pop(oldest)
        _state["retained_dropped"] = int(_state["retained_dropped"]) + 1
        _state["last_reason"] = (
            f"event {oldest} ({dropped['count']} slots) was evicted from retention "
            f"before its store completed: those spills are lost, and a lost spill "
            f"is only ever a missed file")
    _state["retained_now"] = int(len(_retained))
    return _ok(retained=existing + 1, event=event, count=count, digest=digest,
               reason=None)


def retained_view() -> dict:
    """The retention map's SHAPE, never its keys: bounded and hash-free."""
    return {"events": int(len(_retained)),
            "capacity": MAX_RETAINED_EVENTS,
            "slots": int(sum(int(record.get("count") or 0)
                             for record in _retained.values())),
            "slot_capacity_per_event": MAX_SLOTS_PER_EVENT,
            "dropped": int(_state["retained_dropped"]),
            "note": ("the (slot, key16) pairs kept between bind_connector_metadata "
                     "and store completion. Only the shape is reported: the keys are "
                     "block hashes, and a snapshot is not a place to publish them.")}


# --- the per-step path: bind, then completion --------------------------------

def observe_bind(handler, metadata) -> dict:
    """The bind entry: probe the fields, then remember the step's slot/key pairs.

    Called from the wrapper with the metadata the fork just bound, AFTER the fork's
    own body ran. Never raises; every failure is a counted dict.
    """
    _state["binds"] = int(_state["binds"]) + 1
    try:
        observe_metadata_fields(metadata)
        if not _may_touch_disk():
            return _ok(retained=False,
                       reason="the tier is not configured and armed, so nothing is "
                              "remembered and no byte will be written")
        event = _int(getattr(metadata, "store_event", None))
        if event is None or event <= INVALID_JOB_ID:
            # The NORMAL idle step: the fork only allocates a store event when it has
            # blocks to park, so a step without one is not a refusal and must not
            # inflate a refusal counter on a healthy box. It is still counted, under
            # its own name, because "the tier saw 40,000 steps and stored nothing" is
            # a fact an operator is entitled to see.
            _state["steps_idle"] = int(_state["steps_idle"]) + 1
            return _ok(retained=False,
                       reason="this step has no store event, so it has nothing to "
                              "spill (the fork allocates one only when it parks "
                              "blocks)")
        resolved = resolve_store_keys(metadata)
        if not resolved["ok"]:
            return resolved
        if resolved["placeholders"] == resolved["count"]:
            _count("no_key_for_slot",
                   f"every one of the {resolved['count']} stored slots has an empty "
                   f"block hash")
            return _fail(f"every stored slot has an empty block hash "
                         f"({resolved['count']} of {resolved['count']}): nothing can "
                         f"be named", reason_kind="no_key_for_slot",
                         count=resolved["count"])
        # This rank's own digest, built once per boot from its own measurements.
        own = adopt_fingerprint(handler)
        if not own["ok"]:
            return _fail(own["reason"], reason_kind="fingerprint_incomplete",
                         missing_fields=own.get("missing_fields"))
        digest = own["geometry_digest"]
        expected = resolved["expected_digest"]
        if expected is not None and expected != digest:
            _count("fingerprint_mismatch",
                   f"store_fingerprint says {expected} but this rank's own digest "
                   f"is {digest}")
            return _fail(f"store_fingerprint says {expected} while THIS rank's "
                         f"digest - built from its own slot width and rank id - is "
                         f"{digest}: two sites computed the same field and "
                         f"disagree, so nothing is spilled under either name",
                         reason_kind="fingerprint_mismatch",
                         scheduler_digest=expected, worker_digest=digest)
        # strict=True is a SECOND assertion of the pairing, not the first:
        # resolve_store_keys already refused unequal lengths, and if that ever
        # regressed, silently zipping to the shorter list is exactly the
        # wrong-bytes-under-the-right-name failure this module exists to prevent.
        pairs = tuple(zip(resolved["slots"], resolved["keys"], strict=True))
        named = tuple(pair for pair in pairs if pair[1] is not None)
        return _remember_event(event, named, digest, resolved["count"])
    except Exception as exc:  # noqa: BLE001 - a diagnostic never breaks a bind
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        snapshot._error(exc, "disk_store bind observation")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return _fail(f"the bind observation raised {type(exc).__name__}")


def completed_snapshot(handler):
    """The keys of `_completed_store_events`, or None when we must not bother.

    Taken BEFORE the fork's `get_finished` runs: that call is what ADDS the newly
    completed indices, so the only way to see the difference is to know the before.
    None (not an empty set) means "no diff this step", which is the state whenever
    the tier cannot do work, nothing is retained, or the dict is implausibly large -
    the last case is counted rather than copied.
    """
    if not _may_touch_disk():
        return None
    if not _retained:
        return None
    completed = getattr(handler, "_completed_store_events", None)
    if not isinstance(completed, dict):
        return None
    if len(completed) > MAX_COMPLETED_SCAN:
        _state["scan_skipped"] = int(_state["scan_skipped"]) + 1
        return None
    return frozenset(completed.keys())


def on_store_completed(handler, before) -> dict:
    """Spill the slots each newly-completed store event wrote. Never raises.

    `before` is the pre-call key set from `completed_snapshot`. An event is spilled
    ONCE (`_state["handled"]`), its retention record is then dropped, and at most
    `MAX_SLOTS_PER_TICK` spill requests leave this step: `disk_spill`'s queue and
    write window are the real bounds, and this is the bound that keeps the engine
    thread's work bounded whatever they are set to.
    """
    if before is None:
        return _ok(spilled=0, reason="no completion diff was taken")
    report = {"ok": True, "reason": None, "events": [], "spilled": 0, "queued": 0}
    try:
        completed = getattr(handler, "_completed_store_events", None)
        if not isinstance(completed, dict):
            return report
        handled = dict(_state.get("handled") or {})
        new_events = [index for index in completed
                      if index not in before and str(index) not in handled]
        handed = 0
        for event in sorted(new_events, key=lambda value: str(value)):
            record = _retained.pop(event, None)
            _state["completions"] = int(_state["completions"]) + 1
            # `_remember` REPLACES the dict (the `blockdump._remember_record`
            # idiom), so the local `handled` snapshot above goes stale by design: an
            # event appears once per call, and the set exists to stop a repeat ACROSS
            # calls - the case where the fork has not yet drained
            # `_completed_store_events` into its worker metadata.
            _remember("handled", event)
            handled[str(event)] = round(time.time(), 6)
            _state["retained_now"] = int(len(_retained))
            if record is None:
                _count("unretained_event",
                       f"store event {event} completed with no retained slot/key "
                       f"record (retention held {MAX_RETAINED_EVENTS} events, or "
                       f"the tier was unarmed when it was bound)")
                report["events"].append({"event": event, "spilled": 0,
                                         "reason_kind": "unretained_event"})
                continue
            spilled_here = 0
            for slot, key in record["pairs"]:
                if handed >= MAX_SLOTS_PER_TICK:
                    remaining = len(record["pairs"]) - spilled_here
                    _state["slots_capped"] = int(_state["slots_capped"]) + remaining
                    _count("over_tick_cap",
                           f"{remaining} slots of event {event} were not offered "
                           f"this step (cap {MAX_SLOTS_PER_TICK} per step)")
                    break
                _state["slots_offered"] = int(_state["slots_offered"]) + 1
                handed += 1
                spilled_here += 1
                answer = disk_spill.request_spill(worker=handler, slot=slot, key=key)
                if answer.get("ok"):
                    _state["spill_queued"] = int(_state["spill_queued"]) + 1
                    report["queued"] = int(report["queued"]) + 1
                else:
                    _state["delegated_refusals"] = \
                        int(_state["delegated_refusals"]) + 1
                    _state["last_reason"] = _clip(answer.get("reason"))
            report["spilled"] = int(report["spilled"]) + spilled_here
            report["events"].append({"event": event, "spilled": spilled_here,
                                     "slots": int(record["count"]),
                                     "digest": record["digest"]})
        return report
    except Exception as exc:  # noqa: BLE001 - a spill never breaks a step
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        snapshot._error(exc, "disk_store completion hook")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return _fail(f"the completion hook raised {type(exc).__name__}")


# --- the spill-one relay: one EngineCore ask, answered by every rank ----------

def _pending_path(directory=None):
    directory = config.diagnostics_dir() if directory is None else directory
    if directory is None:
        return None
    return os.path.join(str(directory), PENDING_FILE)


def _next_seq() -> int:
    last = _state.get("pending_seq")
    seq = int(last) + 1 if isinstance(last, int) and not isinstance(last, bool) \
        else 1
    _state["pending_seq"] = seq
    return seq


def _record_key(record) -> str:
    """A worker's de-duplication key for one ask. `epoch` alone cannot work, because
    a worker can never learn the EngineCore's boot id (the `worker_diag` epoch
    problem, in the other direction)."""
    return (f"{record.get('epoch')}:{record.get('seq')}"
            f":{record.get('requested_ts')}")


def _read_pending_record(path) -> tuple:
    """(record | None, reason). One bounded, validated read of an untrusted file."""
    if path is None:
        return None, f"{config.ENV_DIAG_DIR} is not set: no channel to a worker"
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return None, "no spill-one request"
    except OSError as exc:  # noqa: BLE001 - unreadable is reported, never a fault
        return None, f"the request could not be stat'ed: {type(exc).__name__}"
    if size > MAX_PENDING_FILE_BYTES:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, (f"the request is {size} bytes (cap "
                      f"{MAX_PENDING_FILE_BYTES}); refusing to trust it")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_PENDING_FILE_BYTES + 1)
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a broken record is data, not a fault
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, f"the request did not parse: {type(exc).__name__}"
    _state["pending_reads"] = int(_state["pending_reads"]) + 1
    if not isinstance(document, dict):
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, "the request is not an object"
    if document.get("version") != PROTOCOL_VERSION:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, (f"the request is protocol version "
                      f"{document.get('version')!r}, this build speaks "
                      f"{PROTOCOL_VERSION}")
    if _int(document.get("seq")) is None or document["seq"] <= 0:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, "the request has no positive integer seq"
    for field in ("requested_ts", "expires_ts"):
        value = document.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
            return None, f"the request has no numeric {field}"
    if _int(document.get("slot")) is None:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, "the request names no slot"
    if not disk_layout.validate_key16(document.get("key"))["ok"]:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, "the request's key is not 16 hex characters"
    # The fingerprint is an OPTIONAL cross-check here too: empty is normal, and a
    # value that is not a digest is a lie about the naming identity.
    fingerprint = document.get("fingerprint")
    if fingerprint not in (None, "") and not \
            disk_layout.validate_geometry_digest(fingerprint)["ok"]:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, ("the request's fingerprint is neither empty nor 12 hex "
                      "characters")
    return document, None


def _write_receipt(rank, record, *, refused, reason, queued, path) -> dict:
    """One rank's verdict on the spill-one ask. Small, atomic, never raises."""
    report = {"written": False, "path": None, "reason": None}
    directory = config.diagnostics_dir()
    if directory is None or rank is None:
        report["reason"] = "no diagnostics directory or rank to write as"
        return report
    document = {"version": PROTOCOL_VERSION, "role": ROLE, "rank": rank,
                "writer": WRITER_TAG, "epoch": str(snapshot.epoch()),
                "ts": round(time.time(), 6),
                "request_seq": _int(record.get("seq")),
                "slot": _int(record.get("slot")),
                "key": _clip(record.get("key"), 64),
                "fingerprint": _clip(record.get("fingerprint"), 32),
                "refused": bool(refused), "reason": _clip(reason),
                "queued": bool(queued), "expected_path": _clip(path),
                "scope": ("THIS rank's answer only. The ask is rank-agnostic, so "
                          "every rank that polls it spills its OWN slot with that "
                          "index under ITS OWN digest and writes a receipt of its "
                          "own; a rank missing here did not poll in time, which is "
                          "not the same as it having no data.")}
    path_out = os.path.join(str(directory), RECEIPT_FILE.format(rank=rank))
    try:
        raw = _dump(document)
        if len(raw) > MAX_RECEIPT_FILE_BYTES:
            report["reason"] = (f"the receipt is {len(raw)} bytes (cap "
                                f"{MAX_RECEIPT_FILE_BYTES})")
            return report
        tmp = f"{path_out}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path_out)
        report.update({"written": True, "path": path_out})
        return report
    except Exception as exc:  # noqa: BLE001 - a verdict file never breaks a step
        _state["receipt_errors"] = int(_state["receipt_errors"]) + 1
        snapshot._error(exc, "disk_store receipt")
        report["reason"] = f"{type(exc).__name__}"
        return report


def fulfil_spill_one(handler, record=None) -> dict:
    """Answer one spill-one ask in THIS worker: name it, queue it, receipt it.

    The bytes are NOT written here: `disk_spill.request_spill` is a `put_nowait` and
    the writer thread does the read and the write, because a synchronous 11.4 MB
    read-plus-fsync on the worker's step thread is exactly the engine stall this
    module's rules forbid. The receipt therefore says `queued`, names THIS rank's
    digest and the file it will land in, and the operator confirms it with `op: ls`.
    """
    directory = config.diagnostics_dir()
    if directory is None:
        # A worker with no channel cannot be asked anything, and that is a counted
        # fact rather than a silent None: `no_channel` in a refusal table is how an
        # operator learns the rank never saw the ask, as opposed to refusing it.
        _count("no_channel", f"{config.ENV_DIAG_DIR} is unset, so this worker has "
                             f"no channel to read a spill-one request from")
        return {"written": False, "reason": f"{config.ENV_DIAG_DIR} is unset"}
    from . import worker_diag  # noqa: PLC0415 - one rank rule, not two
    rank, _source = worker_diag.rank_id()
    if record is None:
        return {"written": False, "reason": "no request record"}
    key = _record_key(record)
    if key in (_state.get("served") or {}):
        return {"written": True, "reason": "already served by this rank",
                "refused": False, "rank": rank}
    _state["fulfilled"] = int(_state["fulfilled"]) + 1
    refused = False
    reason = None
    queued = False
    expected = None
    try:
        if not _may_touch_disk():
            refused, reason = True, ("the disk tier is not configured and armed in "
                                     "this worker, so writing a slot is not "
                                     "permitted by the tier's own gate")
            _count("not_armed" if disk_spill.disk_root() is not None
                   else "not_configured", reason)
        else:
            own = adopt_fingerprint(handler)
            if not own["ok"]:
                refused, reason = True, own["reason"]
            elif (record.get("fingerprint") not in (None, "")
                    and record["fingerprint"] != own["geometry_digest"]):
                # THE CROSS-CHECK, and it is answered AFTER the fingerprint was
                # adopted so the message can name this rank's real digest: a refusal
                # that reads `own digest is None` on a rank that has one sends the
                # operator to the wrong machine. Adopting first also means a rank
                # that cannot name a file never queues a job for one.
                refused, reason = True, (
                    f"the request carries fingerprint {record['fingerprint']} but "
                    f"THIS rank's own digest is {own['geometry_digest']}: its "
                    f"identity was computed by two sites and disagrees, so nothing "
                    f"is written under either name")
                _count("fingerprint_mismatch", reason)
            else:
                answer = disk_spill.request_spill(worker=handler,
                                                  slot=record["slot"],
                                                  key=record["key"])
                queued = bool(answer.get("ok"))
                refused = not queued
                reason = answer.get("reason")
                # The name the writer thread WILL use. Computed here from the
                # fingerprint this rank adopted - `request_spill`'s queued answer
                # carries no path on purpose (the queue can drop the job, so a path
                # in it would be a promise), which is exactly why the receipt says
                # `queued` and points at `op: ls` for proof.
                named = disk_layout.build_slot_path(
                    disk_spill.disk_root(), record["key"], own["geometry_digest"])
                expected = named.get("path") if named.get("ok") else None
                if queued:
                    _state["last_path"] = _clip(expected)
                else:
                    _state["delegated_refusals"] = \
                        int(_state["delegated_refusals"]) + 1
    except Exception as exc:  # noqa: BLE001 - a spill-one ask never breaks a step
        _state["fulfil_errors"] = int(_state["fulfil_errors"]) + 1
        snapshot._error(exc, "disk_store spill-one fulfilment")
        refused = True
        reason = f"fulfilment raised {type(exc).__name__}"
    _remember("served", key)
    _state["last_rank"] = rank
    _state["last_reason"] = _clip(reason)
    written = _write_receipt(rank, record, refused=refused, reason=reason,
                             queued=queued, path=expected)
    return {"written": bool(written["written"]), "refused": refused, "rank": rank,
            "queued": queued, "reason": reason, "expected_path": expected,
            "receipt": written["path"], "receipt_reason": written["reason"]}


def poll_spill_one(handler) -> dict:
    """The engine-thread half of spill-one: at most one bounded control read, then
    one rank's receipt. Bounded by time, the arm, and a configured directory."""
    if not config.diagnostics_dir() or not _may_touch_disk():
        _state["poll_suppressed"] = int(_state["poll_suppressed"]) + 1
        return _ok(polled=False, reason="no diagnostics dir, or the tier is not "
                                        "configured and armed")
    now = time.monotonic()
    last = _state.get("last_poll_mono")
    if isinstance(last, (int, float)) and (now - last) < POLL_INTERVAL_SECONDS:
        return _ok(polled=False, reason="polled recently")
    _state["last_poll_mono"] = now
    _state["polls"] = int(_state["polls"]) + 1
    outcome = {"polled": True, "adopted": False, "reason": None}
    try:
        path = _pending_path()
        record, reason = _read_pending_record(path)
        outcome["reason"] = reason
        if record is not None:
            expires = float(record["expires_ts"])
            if time.time() >= expires:
                outcome["reason"] = "the request had expired"
                _state["pending_stat"] = "expired"
                _count("expired", f"seq {record['seq']} expired before this rank "
                                  f"answered it")
            else:
                key = _record_key(record)
                if key in (_state.get("served") or {}):
                    outcome["reason"] = "already served by this rank"
                    _state["pending_stat"] = "served"
                else:
                    answered = fulfil_spill_one(handler, record)
                    outcome["adopted"] = True
                    outcome["answer"] = answered
                    _state["adopted"] = int(_state["adopted"]) + 1
                    _state["pending_stat"] = ("refused" if answered["refused"]
                                              else "queued")
    except Exception as exc:  # noqa: BLE001 - a control poll never breaks a step
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        snapshot._error(exc, "disk_store spill-one poll")
        outcome["reason"] = f"poll raised {type(exc).__name__}"
    return outcome


def _op_spill_one(body: dict) -> dict:
    """Write one pending request and read the receipts that arrive in the window."""
    directory = config.diagnostics_dir()
    if directory is None:
        return {"ok": False, "reason": f"{config.ENV_DIAG_DIR} is not set, so there "
                                       f"is no channel to a worker; point it at the "
                                       f"directory the engine's rank files are in"}
    checked_key = disk_layout.validate_key16(body.get("key"))
    if not checked_key["ok"]:
        return {"ok": False, "reason": f"the key is not a usable block hash: "
                                       f"{checked_key['reason']}"}
    if not disk_spill.is_armed():
        # THE ARM IS THE ASK'S PRECONDITION, on both sides of the process boundary.
        # This runs in the EngineCore and `is_armed()` consults the arm FILE that a
        # worker also reads, so the check is honest cross-process and not a read of
        # another process's memory: writing a request that every rank must refuse
        # would leave a file a worker has to chew through for nothing.
        return {"ok": False,
                "reason": "the spill tier is not armed in this process (and the arm "
                          "file a worker reads is absent or expired): arm it with "
                          "op:arm first, because a request every rank must refuse "
                          "is not a diagnostic, it is litter"}
    slot = _int(body.get("slot"))
    if slot is None or slot < 0:
        return {"ok": False, "reason": "slot must be a non-negative CPU slot index"}
    # The fingerprint is OPTIONAL and a cross-check only: the empty string is the
    # normal value (the engine sends "") and means "name it with whatever each rank
    # computes for itself", which is exactly the corrected contract.
    fingerprint = body.get("fingerprint", "")
    fingerprint = "" if fingerprint is None else str(fingerprint).strip()
    if fingerprint:
        checked_digest = disk_layout.validate_geometry_digest(fingerprint)
        if not checked_digest["ok"]:
            return {"ok": False,
                    "reason": f"the optional fingerprint is not 12 hex characters: "
                              f"{checked_digest['reason']}"}
        fingerprint = checked_digest["value"]
    try:
        wait_s = float(body.get("wait_s") or 3.0)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "wait_s must be a number"}
    wait_s = max(0.0, min(30.0, wait_s))
    seq = _next_seq()
    now = time.time()
    record = {"version": PROTOCOL_VERSION, "role": ROLE, "op": "spill-one",
              "seq": seq, "epoch": str(snapshot.epoch()),
              "requested_ts": round(now, 6),
              "expires_ts": round(now + REQUEST_TTL_SECONDS, 6),
              "slot": slot, "key": checked_key["value"], "fingerprint": fingerprint,
              "writer": WRITER_TAG,
              "scope": ("rank-agnostic: every worker that polls this answers it for "
                        "ITS OWN slot with that index, under ITS OWN digest, and "
                        "writes a receipt of its own. A rank missing from the "
                        "receipts did not poll in time, which is NOT the same as it "
                        "having no data."),
              "note": ("the spill is QUEUED on each rank's disk_spill writer thread, "
                       "not completed synchronously: confirm with op:ls")}
    path = _pending_path(directory)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        raw = _dump(record)
        if len(raw) > MAX_PENDING_FILE_BYTES:
            return {"ok": False, "reason": f"the request is {len(raw)} bytes (cap "
                                           f"{MAX_PENDING_FILE_BYTES})"}
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - reported, never raised at a caller
        snapshot._error(exc, "disk_store spill-one request")
        for candidate in (tmp, path):
            # The half-written request must not survive to be adopted by a worker:
            # a request the writer already gave up on, answered, is worse than none.
            with contextlib.suppress(OSError):
                os.unlink(candidate)
        return {"ok": False, "reason": f"the request could not be written: "
                                       f"{type(exc).__name__}"}
    _state["requests_written"] = int(_state["requests_written"]) + 1
    expected, expected_source = _expected_ranks()
    deadline = time.monotonic() + wait_s
    receipts = _read_receipts(directory, seq)
    while time.monotonic() < deadline:
        receipts = _read_receipts(directory, seq)
        # With no handshake recorded, completeness is NOT claimed: the window is
        # waited out in full rather than breaking at the first receipt (or, worse,
        # at an empty expected set, which would mean "answered" before anyone spoke).
        if expected and len(receipts) >= len(expected):
            break
        time.sleep(0.05)
    return {"ok": True, "reason": f"the request was written and the receipts read "
                                  f"for {round(wait_s, 1)}s; a spill is QUEUED on "
                                  f"each rank's writer thread, so confirm the file "
                                  f"with op:ls",
            "seq": seq, "path": path,
            "expected_ranks": expected, "expected_ranks_source": expected_source,
            "ranks_answered": sorted(receipts),
            "completeness": ("unknown" if not expected
                             else ("complete" if len(receipts) >= len(expected)
                                   else "partial")),
            "receipts": receipts}


def _expected_ranks() -> tuple:
    """(rank ids | None, how that is known). THE ONE RULE LIVES IN `blockdump`.

    The EngineCore is the process that must say who did not answer, and it is also
    the process that cannot resolve a rank itself (no `init_process_group`) - the
    rank SET is only knowable from the capacity handshake. `blockdump` already owns
    that reading, so this calls it instead of writing a second rule that can drift:
    two answers for "who should have answered" is how a completeness claim becomes
    a coin flip.
    """
    from . import blockdump  # noqa: PLC0415 - the handshake's rank set, one rule
    try:
        return blockdump._expected_ranks()
    except Exception as exc:  # noqa: BLE001 - a wait budget is never a verb failure
        return None, f"the rank set was unreadable: {type(exc).__name__}"


def _read_receipt_file(path):
    try:
        if os.path.getsize(path) > MAX_RECEIPT_FILE_BYTES:
            return None
        with open(path, "rb") as handle:
            return json.loads(handle.read(MAX_RECEIPT_FILE_BYTES + 1).decode("utf-8"))
    except Exception:  # noqa: BLE001 - a bad receipt is data, not a fault
        return None


def _read_receipts(directory, seq: int) -> dict:
    """The ranks that answered THIS seq, keyed by rank. An unreadable directory is
    NO receipts and a remembered reason, never a fake `error` entry: this dict's
    LENGTH is compared against the expected rank set, so a phantom key could turn an
    I/O failure into a "complete" claim. The caller reports the wait window; the
    reason stays in `_state["last_reason"]` for `op:status` to surface.
    """
    out = {}
    try:
        names = sorted(os.listdir(str(directory)))
    except OSError as exc:
        _state["last_reason"] = _clip(
            f"the receipts could not be listed: {type(exc).__name__}")
        return {}
    for name in names[:MAX_RANK_FILES]:
        if not (name.startswith(RECEIPT_PREFIX) and name.endswith(FILE_SUFFIX)):
            continue
        document = _read_receipt_file(os.path.join(str(directory), name))
        if not document or document.get("request_seq") != seq:
            continue
        rank = document.get("rank")
        if not isinstance(rank, str) or not rank:
            _state["last_reason"] = _clip(
                f"a receipt ({name}) named no rank, so it is not counted as an "
                f"answer")
            continue
        out[rank] = {
            "refused": bool(document.get("refused")),
            "queued": bool(document.get("queued")),
            "reason": document.get("reason"),
            "expected_path": document.get("expected_path"),
            "digest": document.get("fingerprint")}
    return out


# --- the verb ------------------------------------------------------------------

def _dump(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def _trim(response: dict, budget: int | None = None) -> dict:
    """Guarantee a response cannot grow past its budget.

    The listing is the only unbounded part of any answer, so it is what gets eaten,
    from the tail, and the payload SAYS it was trimmed: a truncated `ls` that looks
    complete is worse than a short one, because the operator draws conclusions about
    what the store holds from the rows that are missing.

    The budget is read at CALL time, not bound as a default: a constant captured in
    the signature cannot be moved by anything, including the test that exists to
    prove the ceiling holds.
    """
    budget = MAX_RESPONSE_BYTES if budget is None else budget
    raw = _dump(response)
    if len(raw) <= budget:
        return response
    listing = response.get("files")
    while listing and len(_dump(response)) > budget:
        listing.pop()
        response["truncated"] = True
        response["count"] = len(listing)
        response["note"] = (f"the response exceeded {budget} bytes and was trimmed "
                            f"from {len(raw)} bytes")
        # The op's own sentence may claim "3 of 3 file(s)"; after a trim it would be
        # a completeness claim about rows that are no longer in the payload, so the
        # sentence is marked rather than left to mislead.
        if isinstance(response.get("reason"), str):
            response["reason"] = response["reason"] + " (TRIMMED)"
    if len(_dump(response)) > budget:
        response = {"ok": False, "reason": "the response would exceed "
                                           f"{budget} bytes"}
    return response


def _bounded_int(value, default, cap, floor=1) -> int:
    try:
        number = int(value) if value is not None else default
    except (TypeError, ValueError):
        number = default
    return max(floor, min(cap, number))


def _validate_shard(shard) -> tuple:
    """`(shard | None, why)`: the two-hex-char directory prefix, or a refusal.

    The LENGTH and the ALPHABET come from `disk_layout` rather than being spelled out
    here, because a shard is a piece of the file name and a second definition of one
    is how a listing ends up looking in a directory no writer writes to.
    """
    if not isinstance(shard, str):
        return None, f"a shard must be text, got {type(shard).__name__}"
    text = shard.strip().lower()
    if len(text) != disk_layout.SHARD_CHARS:
        return None, (f"a shard is {disk_layout.SHARD_CHARS} hex chars; "
                      f"{shard!r} is {len(text)}")
    if any(char not in disk_layout.HEX_DIGITS for char in text):
        return None, "a shard is hexadecimal (it is the head of a block hash)"
    return text, None


def _op_ls(body: dict) -> dict:
    """The listing. Read-only, bounded in files, dirs, entries, and bytes.

    The shard filter is applied to the walk's entries rather than pushed into
    `_walk_store`: the walk has no shard argument (the capacity check always wants the
    whole store) and inventing one would mean editing the module that owns the caps
    for a convenience of this verb. The caps bound the walk either way.

    A store directory that does not exist is an EMPTY listing, not a failure: the
    tier creates nothing until its first write (that is what "off means off" costs the
    filesystem), so a fresh boot - or a booted tier that has never seen pressure - has
    no directory, and `ls` saying so is a normal answer, not an error one.
    """
    shard = body.get("shard")
    if shard not in (None, ""):
        checked_shard, why = _validate_shard(shard)
        if checked_shard is None:
            return {"ok": False, "reason": f"the shard is not usable: {why}"}
        shard = checked_shard
    limit = _bounded_int(body.get("limit"), MAX_LS_LIMIT_DEFAULT, MAX_LS_FILES)
    root = disk_spill.disk_root()
    if root is None:
        return {"ok": True, "reason": f"{config.ENV_DISK_ROOT} is not set, so there "
                                      f"is nothing to list", "count": 0,
                "files": [], "root": None, "read_only": True}
    if not os.path.isdir(root):
        return {"ok": True,
                "reason": (f"no store exists at {root} yet: nothing has been spilled "
                           f"by this tier, which creates no directory until its "
                           f"first write"),
                "count": 0, "files": [], "root": root, "shard": shard or "all",
                "read_only": True,
                "ceiling_bytes": disk_spill.capacity_ceiling_bytes(),
                "units": disk_layout._units("ceiling_bytes")}
    walk = disk_spill._walk_store(root, max_files=disk_spill.MAX_WALK_FILES,
                                  max_shard_dirs=disk_spill.MAX_SHARD_DIRS)
    if not walk["ok"]:
        return {"ok": False, "root": root,
                "reason": f"the store could not be measured: {walk['reason']}",
                "read_only": True}
    entries = list(walk["entries"])
    if shard not in (None, ""):
        entries = [row for row in entries
                   if os.path.basename(os.path.dirname(str(row["path"])))
                   == str(shard).lower()]
    entries.sort(key=lambda row: (-int(row.get("mtime_ns") or 0),
                                  str(row.get("filename"))))
    files = [{"name": str(row["filename"])[-MAX_FILENAME_DISPLAY:],
              "bytes": int(row["size_bytes"]),
              "mtime_ns": int(row["mtime_ns"])} for row in entries[:limit]]
    return {"ok": True,
            "reason": (f"{len(files)} of {len(entries)} file(s); the store holds "
                       f"{int(walk['used_bytes'])} bytes measured over "
                       f"{int(walk['files'])} file(s)"),
            "count": len(files), "files": files, "root": root,
            "shard": shard or "all", "walked": int(walk["files"]),
            "walk_used_bytes": int(walk["used_bytes"]),
            "ceiling_bytes": disk_spill.capacity_ceiling_bytes(),
            "walk_capped": bool(walk["truncated"]), "limit": limit,
            "truncated": len(files) < len(entries) or bool(walk["truncated"]),
            "read_only": True,
            "units": disk_layout._units("walk_used_bytes", "ceiling_bytes",
                                        "mtime_ns")}


def _op_status(_body: dict) -> dict:
    from . import worker_diag  # noqa: PLC0415 - one rank rule, not two
    rank, source = worker_diag.rank_id()
    state = disk_spill.counters()
    pending_path = _pending_path()
    _record, pending_reason = _read_pending_record(pending_path)
    return {"ok": True, "reason": None, "op": "status", "rank": rank,
            "rank_source": source, "armed": disk_spill.is_armed(),
            "capability": disk_spill.capability(),
            "root": disk_spill.disk_root(),
            "ceiling_bytes": disk_spill.capacity_ceiling_bytes(),
            "arm_state": disk_spill.arm_state(),
            "geometry_digest": disk_spill.geometry_digest(),
            "engine_fields": engine_fields(),
            "worker_fingerprint": fingerprint_report(),
            # disk_spill's own counters, verbatim: this verb and `snapshot` must
            # report the SAME numbers for the same mechanism, and a renamed or
            # re-derived counter is how two views start disagreeing in production.
            "spill": {key: state[key] for key in
                      ("requests", "accepted", "spilled", "duplicates",
                       "bytes_written", "drops", "evictions", "evicted_bytes",
                       "estimated_bytes", "errors", "thread_started",
                       "thread_alive", "queue_depth", "queue_depth_peak",
                       "window_files")},
            "spill_refused": dict(state["refused"]),
            "queue_max": disk_spill.MAX_QUEUE_DEPTH,
            "spill_one": {"pending": pending_path, "pending_state": pending_reason,
                          "requests_written": int(_state["requests_written"]),
                          "poll_interval_seconds": POLL_INTERVAL_SECONDS,
                          "ttl_seconds": REQUEST_TTL_SECONDS},
            "hook": hook_section(include_fingerprint=False),
            "retained": retained_view()}


def _op_arm(body: dict) -> dict:
    """`disk_spill.arm`, with the body UNPACKED.

    The verb's table calls every op with the whole body, so an op whose engine
    function takes a scalar has to be adapted here rather than handed the dict: a
    body passed where `seconds` belongs fails closed (`seconds must be a number`),
    which is safe but leaves an operator arming a tier with a message about types.
    """
    return disk_spill.arm(body.get("seconds"))


def _op_disarm(_body: dict) -> dict:
    return disk_spill.disarm()


_DISK_OPS = {"status": _op_status, "arm": _op_arm, "disarm": _op_disarm,
             "spill-one": _op_spill_one, "ls": _op_ls}


def disk_for_engine(engine=None, args: dict | None = None) -> dict:  # noqa: ARG001 - the factory binds it as a method; see the docstring
    """The whole `/plugins/kvtier/disk` verb. Token-gated at `endpoint.py`; never
    raises. `op` defaults to `status`; a wrong name is refused, never guessed.

    THE OP IS MUTATING, so a caller that reached this function through a safe method
    is refused before any op runs: a browser's `<img>`/`<script>` preload issues a
    GET with no custom header and no CORS preflight, and the `op` is a QUERY
    PARAMETER, so the URL that spills a slot is as short as the URL that reads
    status. `endpoint.route_disk` therefore answers a safe method with 405.

    `engine` is the bound EngineCore, taken and unused: the sibling `blockdump`
    handler's signature is the convention the factory binds against, and this tier
    needs nothing from the engine - the CPU slots it spills live in the WORKER, and
    the only engine-side state it touches is `disk_spill`'s own gate and the request
    file a worker polls.
    """
    body = args if isinstance(args, dict) else {}
    op = str(body.get("op") or "status").strip().lower()
    if op not in DISK_OPS:
        return {"ok": False,
                "reason": f"unknown op {op!r}; the ops are {', '.join(DISK_OPS)}",
                "ops": list(DISK_OPS)}
    if op in MUTATING_DISK_OPS and body.get("safe_method"):
        return {"ok": False,
                "reason": f"op '{op}' mutates and will not run from a method a "
                          f"browser may issue without a preflight; send it as POST",
                "op": op}
    try:
        response = _DISK_OPS[op](body)
    except Exception as exc:  # noqa: BLE001 - the verb is contained by contract
        snapshot._error(exc, "disk_store verb")
        _state["errors"] = int(_state["errors"]) + 1
        response = {"ok": False, "op": op,
                    "reason": f"the op raised {type(exc).__name__}: {exc}"}
    if not isinstance(response, dict):
        response = {"ok": False, "reason": "an op returned a non-object result",
                    "op": op}
    response.setdefault("op", op)
    response["engine_fields"] = engine_fields()
    if "armed" not in response:
        response["armed"] = disk_spill.is_armed()
    # Trim LAST, after the fields this wrapper adds. A budget enforced before the
    # shared fields were appended would be a guarantee the caller never receives:
    # `engine_fields` alone is a few hundred bytes, so an op that obediently trimmed
    # its own listing could still ship over the ceiling.
    return _trim(response)


# --- install / uninstall: the engine hook ------------------------------------

def _layer_is_ours(method) -> bool:
    return getattr(method, OWNER_ATTR, None) == OWNER_TAG


def _layer_is_sibling(method) -> bool:
    origin = getattr(method, "__module__", "") or ""
    return origin.startswith(SIBLING_PACKAGE)


def _wrapped_bind(original):
    def bind_connector_metadata(self, metadata):
        original(self, metadata)             # THE FORK FIRST; its result is unused
        observe_bind(self, metadata)         # contained: never raises
    bind_connector_metadata.__name__ = METHOD_BIND_METADATA
    return bind_connector_metadata


def _wrapped_get_finished(original):
    def get_finished(self, *args, **kwargs):
        before = completed_snapshot(self)    # BEFORE the fork: it does the completing
        result = original(self, *args, **kwargs)
        on_store_completed(self, before)
        poll_spill_one(self)
        return result                        # UNTOUCHED AND UNCONSUMED
    get_finished.__name__ = METHOD_GET_FINISHED
    return get_finished


_WRAPPERS = {METHOD_BIND_METADATA: _wrapped_bind,
             METHOD_GET_FINISHED: _wrapped_get_finished}

# "This method was already ours." A sentinel rather than None so `install()` can
# count ONE reinstall per install call: a reader asking "did the reload re-attach or
# stack a second layer?" wants 1, not 2 (the number of methods this hook wraps).
_REUSED = object()


def _install_method(cls, name: str) -> str:
    """Wrap one method. Returns a reason, None on success, or `_REUSED` when the
    class already carries THIS owner's layer."""
    current = cls.__dict__.get(name)
    if current is None:
        return (f"{cls.__name__}.{name} is not in the class dict, so the fork "
                f"changed shape")
    if _layer_is_ours(current):
        return _REUSED            # counted once per install(), not per method
    if not callable(current):
        return f"{cls.__name__}.{name} is not callable"
    if _layer_is_sibling(current):
        _state["nested"].append(f"{name}({current.__name__})")
    elif getattr(current, OWNER_ATTR, None) is not None:
        return (f"{cls.__name__}.{name} already carries a wrapper from another "
                f"owner: refusing rather than stacking on an unknown layer")
    wrapper = _WRAPPERS[name](current)
    setattr(wrapper, OWNER_ATTR, OWNER_TAG)
    # The original goes on the WRAPPER, not on the class: TWO methods are wrapped,
    # and a single class-level slot would mean the second install overwrote the
    # first one's undo handle - uninstall would then restore `get_finished`'s body
    # under the name `bind_connector_metadata`. An attribute that survives install
    # and is read exactly once, at undo.
    setattr(wrapper, ORIGINAL_ATTR, current)
    wrapper.__module__ = __name__
    setattr(cls, name, wrapper)
    return None


def install() -> dict:
    """Wrap the two methods on the live class. Idempotent; never raises.

    No env-var gate, unlike `worker_diag`/`blockdump`: the per-step cost when the
    tier is unarmed is a couple of `os.environ.get` calls, and the gate that matters
    (`_may_touch_disk`) is the one that can be changed at runtime by the verb. The
    two-switch rule still holds - no root, no arm, no spill.
    """
    if _state["installed"]:
        return {"installed": True, "reason": "already installed",
                "class": _state["class"]}
    report = {"installed": False, "reason": None, "class": None}
    try:
        module = sys.modules.get(TARGET_MODULE)
        if module is None:
            report["reason"] = (f"{TARGET_MODULE} is not imported in this process, "
                                f"so there is no worker class to hook")
            _state["skip_reason"] = report["reason"]
            return report
        cls = getattr(module, TARGET_CLASS, None)
        if not isinstance(cls, type):
            report["reason"] = f"{TARGET_MODULE}.{TARGET_CLASS} is not a class"
            _state["skip_reason"] = report["reason"]
            return report
        reused = False
        for name in WRAPPED_METHODS:
            refusal = _install_method(cls, name)
            if refusal is _REUSED:
                reused = True
                continue
            if refusal:
                report["reason"] = refusal
                _state["skip_reason"] = refusal
                return report
        if reused:
            # A reload that found its own layer: counted as a REINSTALL and not also
            # as an install, so `installs` reads "boots that attached a layer" and
            # `reinstalls` reads "installs that found one already there". Two names
            # for one event would let a reader add them and invent a second wrapper.
            _state["reinstalls"] = int(_state["reinstalls"]) + 1
        else:
            _state["installs"] = int(_state["installs"]) + 1
        _state["installed"] = True
        _state["class"] = cls.__name__
        _state["skip_reason"] = None
        report.update({"installed": True, "class": cls.__name__,
                       "reason": (f"{cls.__name__}.{METHOD_BIND_METADATA} and "
                                  f"{cls.__name__}.{METHOD_GET_FINISHED} now spill "
                                  f"on store completion; the tier stays inert until "
                                  f"a root is set AND the arm is raised")})
        return report
    except Exception as exc:  # noqa: BLE001 - installation never breaks a boot
        snapshot._error(exc, "disk_store install")
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        _state["errors"] = int(_state["errors"]) + 1
        report["reason"] = f"install raised {type(exc).__name__}"
        _state["skip_reason"] = report["reason"]
        return report


def ensure_installed(connector=None) -> dict:  # noqa: ARG001 - the handshake's object
    """The worker-handshake entry point: install, ignoring the connector it is handed.

    `engine_side._on_worker_report` calls every listener as `hook(connector)`, so this
    wrapper exists for that signature - and the argument is genuinely unused, because
    this module resolves the worker CLASS BY NAME from the already-imported module
    rather than reaching into an instance it was handed. Accepting the object and
    ignoring it is the honest version of the contract; the siblings (`worker_diag`,
    `blockdump`) take the same optional argument for the same reason.
    """
    return install()


def boot_note() -> str:
    """One line for the boot log: what this hook can do on THIS boot, in words.

    A log line is the only place the two halves of the posture meet - "the hook is on
    the class" and "this boot's engine cannot name a file" - and an operator reading a
    boot log should not have to call a route to learn that the tier is inert, or why.
    """
    counters = _state
    fields = engine_fields()["verdict"]
    if not counters["installed"]:
        posture = ("NOT installed ("
                   f"{counters['skip_reason'] or 'never attempted here'})")
    else:
        posture = (f"installed on {counters['class']}, wrapped over "
                   f"{len(counters['nested'])} sibling layer(s)")
    gate = "armed" if disk_spill.is_armed() else "disarmed"
    if disk_spill.disk_root() is None:
        gate += ", NO root configured"
    return (f"{posture}; engine_fields={fields}, tier {gate}. Inert until a root is "
            f"set AND the arm is raised, and inert on an engine whose metadata "
            f"carries no store_keys: a spill needs a key to name its file.")


def uninstall() -> dict:
    """Put the fork's methods back exactly. Test teardown and a clean undo."""
    report = {"uninstalled": False, "reason": None}
    try:
        module = sys.modules.get(TARGET_MODULE)
        cls = getattr(module, TARGET_CLASS, None) if module else None
        if not isinstance(cls, type):
            report["reason"] = "the worker class is not imported here"
            return report
        restored = 0
        for name in WRAPPED_METHODS:
            current = cls.__dict__.get(name)
            if current is None or not _layer_is_ours(current):
                continue
            original = getattr(current, ORIGINAL_ATTR, None)
            if callable(original):
                setattr(cls, name, original)
                restored += 1
        _state["uninstalls"] = int(_state["uninstalls"]) + 1
        _state["installed"] = False
        report.update({"uninstalled": True,
                       "reason": f"{restored} of {len(WRAPPED_METHODS)} methods "
                                 f"restored to the fork's own"})
        return report
    except Exception as exc:  # noqa: BLE001 - a teardown never raises
        snapshot._error(exc, "disk_store uninstall")
        report["reason"] = f"uninstall raised {type(exc).__name__}"
        return report


def hook_section(include_fingerprint: bool = True) -> dict:
    """The hook's STATE. A layer a snapshot reports, so it must never raise or lie.

    `include_fingerprint=False` drops the nested fingerprint block for a caller that
    publishes the report at its own top level (`op:status` does): one payload, one
    copy of one identity. A duplicated fingerprint is not harmless - the day the two
    copies disagree, a reader has to decide which one named the file.
    """
    return {"role": ROLE, "writer": WRITER_TAG,
            "protocol_version": PROTOCOL_VERSION,
            "installed": bool(_state["installed"]),
            "installs": int(_state["installs"]),
            "reinstalls": int(_state["reinstalls"]),
            "uninstalls": int(_state["uninstalls"]),
            "class": _state["class"], "skip_reason": _state["skip_reason"],
            "wraps": list(_state["nested"]),
            "methods": list(WRAPPED_METHODS),
            "binds": int(_state["binds"]),
            "steps_idle": int(_state["steps_idle"]),
            "completions": int(_state["completions"]),
            "slots_offered": int(_state["slots_offered"]),
            "spill_queued": int(_state["spill_queued"]),
            "delegated_refusals": int(_state["delegated_refusals"]),
            "slots_capped": int(_state["slots_capped"]),
            "scan_skipped": int(_state["scan_skipped"]),
            "refused": dict(_state["refused"]),
            "refusal_reasons": list(REFUSAL_REASONS),
            "fallbacks": int(_state["fallbacks"]), "errors": int(_state["errors"]),
            "worker_fingerprint": fingerprint_report()
            if include_fingerprint else "see the top level",
            "engine_fields": engine_fields(),
            "retained": retained_view(),
            "polls": int(_state["polls"]),
            "poll_suppressed": int(_state["poll_suppressed"]),
            "pending_reads": int(_state["pending_reads"]),
            "pending_malformed": int(_state["pending_malformed"]),
            "receipt_errors": int(_state["receipt_errors"]),
            "requests_written": int(_state["requests_written"]),
            "fulfilled": int(_state["fulfilled"]),
            "last_rank": _state["last_rank"], "last_reason": _state["last_reason"],
            "last_path": _state["last_path"], "last_error": _state["last_error"],
            "caps": {"retained_events": MAX_RETAINED_EVENTS,
                     "slots_per_event": MAX_SLOTS_PER_EVENT,
                     "slots_per_tick": MAX_SLOTS_PER_TICK,
                     "handled_events": MAX_HANDLED_EVENTS,
                     "fingerprint_retry_seconds": FINGERPRINT_RETRY_SECONDS,
                     "ls_files": MAX_LS_FILES,
                     "poll_interval_seconds": POLL_INTERVAL_SECONDS}}


def section() -> dict:
    """The snapshot section: the wiring's state, and whether THIS BOOT can name a
    file. Reads no files and walks no directory, like `disk_spill.section()`.

    Deliberately does NOT restate the spill mechanism's counters as its own: `spill`
    is a SUBSET of `disk_spill.counters()` and `refused` is this hook's own refusals.
    The two fail differently - "no root configured" is disk_spill's, "the engine
    never sent a key" is this module's - and a reader that conflates them will chase
    the wrong one.
    """
    state = disk_spill.counters()
    armed = disk_spill.is_armed()
    root = disk_spill.disk_root()
    fields = engine_fields()
    return {"role": ROLE, "writer": WRITER_TAG,
            "protocol_version": PROTOCOL_VERSION,
            "installed": bool(_state["installed"]),
            "class": _state["class"], "skip_reason": _state["skip_reason"],
            "wraps": list(_state["nested"]),
            "methods": list(WRAPPED_METHODS),
            "armed": armed, "configured": bool(state["configured"]),
            "capability": disk_spill.capability(),
            "root": root, "engine_fields": fields,
            "worker_fingerprint": fingerprint_report(),
            "binds": int(_state["binds"]),
            "steps_idle": int(_state["steps_idle"]),
            "completions": int(_state["completions"]),
            "slots_offered": int(_state["slots_offered"]),
            "spill_queued": int(_state["spill_queued"]),
            "delegated_refusals": int(_state["delegated_refusals"]),
            "slots_capped": int(_state["slots_capped"]),
            "refused": dict(_state["refused"]),
            "refusal_reasons": list(REFUSAL_REASONS),
            "fallbacks": int(_state["fallbacks"]),
            "spill": {key: state[key] for key in
                      ("spilled", "accepted", "requests", "drops", "bytes_written",
                       "evictions", "errors")},
            "retained": retained_view(),
            "spill_one": {"requests_written": int(_state["requests_written"]),
                          "fulfilled": int(_state["fulfilled"]),
                          "poll_interval_seconds": POLL_INTERVAL_SECONDS,
                          "ttl_seconds": REQUEST_TTL_SECONDS},
            "wiring": ("THE ENGINE CALLS THIS. `bind_connector_metadata` remembers "
                       "each step's (slot, key16) pairs and `get_finished` spills "
                       "them when the store event completes; both compose with "
                       "worker_diag and blockdump and hand the fork's return value "
                       "back untouched. Names come from the fingerprint THIS worker "
                       "builds for itself (its own measured slot width, its own "
                       "rank), because one scheduler digest cannot be four ranks' "
                       "digests; store_fingerprint is a cross-check only, and "
                       "disagreeing with it is a counted refusal. Bytes are read on "
                       "disk_spill's writer thread, never on an engine thread. "
                       "Inert until a root is set AND the arm is raised; "
                       f"engine_fields={fields['verdict']} on this boot."),
            "note": "the SSD tier's wiring: what it can name, and why it cannot yet."}
