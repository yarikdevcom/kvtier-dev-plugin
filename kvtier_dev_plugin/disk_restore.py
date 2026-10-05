"""Restore spilled KV bytes off the SSD tier, through the fork's OWN load path.

WHAT THIS IS
------------
`disk_spill` writes one file per (group-qualified block key, rank) and
`disk_index` tells the SCHEDULER process which of those files exist. This module
closes the loop: when the host-RAM tier comes up short on a prefix, it offers the
SSD's continuation as external tokens, allocates destination CPU slots, has
`disk_read` fetch the bytes on its own thread, and lets the fork's existing
CPU->GPU DMA carry them to the GPU. Nothing here invents a transfer: the fork's
`update_state_after_alloc`, `build_connector_meta`, `start_load_kv` and
`get_finished` do the bookkeeping, the pinning and the copy, because a plugin that
duplicates that bookkeeping is a plugin that leaks refs behind the engine's back.

IT DOES NOT PUBLISH ANYTHING. A restored block is NOT inserted into
`cached_block_hash_to_block` / `_insert_block_hash`, so no later request can find
it; a later miss simply re-reads the file (at the measured 11.5 GB/s that is ~10 ms
per 3,200-token span against ~3.2 s to recompute those tokens). The scope decision
matters: mutating the scheduler's hash index is the one action in this feature
that could corrupt OTHER requests' lookups, and it buys nothing correctness-wise.
Cross-boot adoption is likewise absent - `disk_index` indexes what THIS boot's
ranks journalled, so a fresh boot restores nothing until it has spilled.

THE THREE ENGINE FACTS THIS IS BUILT ON (all read from the deployed fork)
------------------------------------------------------------------------
1. THE IDENTITY IS DERIVABLE, EXACTLY. The store path spilled
   `gpu_block_pool.blocks[i].block_hash.hex()`, and `block_pool.cache_block_hashes`
   builds that value as `make_block_hash_with_group_id(block_hashes[i], group)`
   where `block_hashes = resolve_block_hashes(request.block_hashes,
   hash_block_size, block_size)` (`core/block_pool.py:269-271, 283, 287`;
   `core/kv_cache_utils.py:74` packs the 4-byte big-endian group suffix, and
   `:2859` is the view that re-indexes the request's hash list at a group's own
   block size). So the key for group `g` at absolute block index `i` is
   re-computed here by CALLING THOSE TWO ENGINE FUNCTIONS - never by re-implementing
   the packing or the view, which is the same rule `disk_layout.split_block_key`
   keeps on the unpacking side. If either function is not importable, this module
   offers nothing.
2. THE MERGED TUPLE IS THE ONLY SOUND WAY IN. `get_num_new_matched_tokens`
   (`manager.py:323-379`) pins its CPU hit and stashes
   `_pending_cpu_hits[req_id] = (cpu_hit_blocks, hit_length, num_computed_tokens)`;
   `update_state_after_alloc` (`manager.py:381-492`) pops it, takes
   `cdiv(num_external_tokens, group_block_size)` LEADING blocks per prefix-cacheable
   group, pairs them with GPU blocks at `num_computed_tokens // group_block_size + i`,
   touches both sides and creates the `LoadRequestState`. Appending our freshly
   allocated destination slots to the END of each group's list and raising the offer
   therefore makes the FORK do the touch, the free of the untaken remainder
   (`_free_pending_cpu_hit`, `manager.py:1307`) and the state bookkeeping - which is
   why this module wraps nothing on that method and owns no refs of its own.
3. SILENCE IS DEATH. `WAITING_FOR_REMOTE_KVS` is released only when the worker
   reports the request id in `finished_recving` (`scheduler.py:3068-3084`) and the
   fork has NO engine timeout for that state: a load that never reports is
   re-prepended forever. So every path out of a disk load ends in a report. The
   failure report is the pair the engine already supports -
   `get_block_ids_with_load_errors()` (`kv_connector/v1/base.py:405-423`, whose
   docstring requires the request to be reported finished anyway) plus
   `finished_recving`, which `scheduler.py:2008-2015` turns into "truncate to the
   last valid prefix and recompute" under the box's
   `kv_load_failure_policy=recompute`. A missing file is therefore a recompute, and
   a recompute is exactly what would have happened without this module.

THE ADOPTION FLOOR IS 3,200 TOKENS, AND IT IS NOT NEGOTIABLE
------------------------------------------------------------
Measured on live traffic: 142 of 158 offers of 40-3,080 tokens were DISCARDED by
the scheduler (it adopts external tokens only in whole scheduler blocks), and the
single offer at or above 3,200 (19,200) was adopted complete. An offer below one
scheduler block is therefore worse than no offer: it parks a request in
`WAITING_FOR_REMOTE_KVS` for a load the scheduler will throw away. Hence
`whole_block_tokens()` - `floor(matched / block_size) * block_size` - and the rule
that an offer of 0 is not an offer.

THE PHANTOM-REPORT HAZARD, AND WHY IT IS CLOSED
-----------------------------------------------
The worker reports "loaded" from `_pending_load_event_indices` ∩ the
`_poll_stream_events` high-water mark (`worker.py:330-352`), and the watermark is
MONOTONE over event indices. So a load event that is *dropped* (never submitted to
the DMA) but left in `_pending_load_event_indices` becomes reportable as soon as a
LATER event completes - the engine would then believe bytes it never received are
valid KV. That is the phantom, and the invariant this module maintains is:

    every index still in `_pending_load_event_indices` had `launch_copy` called
    with it.

The only place this module can skip a submission is `start_load_kv`, when nothing
survives the disk-entry pruning; `release_event()` there discards the index from
that set in the SAME call and reports the req ids itself, and
`_phantom_guard()` re-checks it after the fact and counts a trip if the invariant
is ever violated (a trip means "we nearly believed garbage", and it is the one
counter that must stay 0).

NEVER BLOCK THE ENGINE THREAD, AND NEVER WAIT FOR A READ
---------------------------------------------------------
Reads are submitted at `bind_connector_metadata` and applied at `start_load_kv`,
which is one forward pass later: a read that has NOT settled by then is treated as
a failure (its entries are pruned from the DMA, so no stale bytes are copied, and
the request recomputes). A bounded wait on the engine thread was rejected - this is
a best-effort optimization, and making the serving loop wait for NVMe converts a
restore miss into latency for every other request. `dma_unsettled_entries` is the
counter that says whether that race dominates on a given box.

THE GATE IS TWO SWITCHES, AND NEITHER IS AN ENV VAR FOR RESTORE
---------------------------------------------------------------
A restore needs `VLLM_E2_TIER_DISK_ROOT` (configured) AND `disk_spill`'s arm file
AND the tier's own arm. A fresh boot with the root set but unarmed emits no offer,
allocates no CPU slot, reads no file and installs no behaviour change. Arming the
spill tier (`POST /plugins/kvtier/disk {"op":"arm"}`, which expires) is what arms
the restore, deliberately: the two halves must never be separable in operation,
because restoring bytes that the same switch stopped being written is nonsense.

Stdlib only (`ctypes`, `os`, `sys`, `time`). Every public door is total: it catches
`Exception` (never `BaseException`, so `GeneratorExit` and `CancelledError`
propagate), counts, and returns a classified dict or the value it was handed.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time

from . import (
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_spill,
    disk_store,
    snapshot,
)

_logger = logging.getLogger("kvtier_dev_plugin.disk_restore")

# --- install targets ---------------------------------------------------------

SCHED_MODULE = "vllm.v1.simple_kv_offload.manager"
SCHED_CLASS = "SimpleCPUOffloadScheduler"
WORKER_MODULE = "vllm.v1.simple_kv_offload.worker"
WORKER_CLASS = "SimpleCPUOffloadWorker"
CONNECTOR_MODULE = ("vllm.distributed.kv_transfer.kv_connector.v1."
                    "simple_cpu_offload_connector")
CONNECTOR_CLASS = "SimpleCPUOffloadConnector"
KV_UTILS_MODULE = "vllm.v1.core.kv_cache_utils"

METHOD_MATCHED = "get_num_new_matched_tokens"
METHOD_BUILD_META = "build_connector_meta"
METHOD_UPDATE_OUTPUT = "update_connector_output"
METHOD_BIND = "bind_connector_metadata"
METHOD_START_LOAD = "start_load_kv"
METHOD_GET_FINISHED = "get_finished"
METHOD_LOAD_ERRORS = "get_block_ids_with_load_errors"

SCHED_METHODS = (METHOD_MATCHED, METHOD_BUILD_META, METHOD_UPDATE_OUTPUT)
WORKER_METHODS = (METHOD_BIND, METHOD_START_LOAD, METHOD_GET_FINISHED)

OWNER_ATTR = "__kvtier_dev_diskrestore_owner__"
OWNER_TAG = "kvtier-dev-plugin/disk-restore-1"
ORIGINAL_ATTR = "__kvtier_dev_diskrestore_original__"
LAYER_KIND = "restore"          # the name `backport._layer_kinds()` knows us by
SIBLING_PACKAGE = (__package__ or "kvtier_dev_plugin") + "."

FIELD_LOAD_PATHS = "load_paths"
FIELD_LOAD_CPU = "load_cpu_blocks"
FIELD_LOAD_GPU = "load_gpu_blocks"
FIELD_LOAD_EVENT = "load_event"
FIELD_LOAD_EVENT_TO_REQS = "load_event_to_reqs"

# The scheduler's path entry is an IDENTITY carried in path shape: it knows the
# key and the group, and it knows NOTHING about the rank that will read the file
# (the geometry digest covers the slot width, which is 11,468,800 B on PP0 and
# 12,902,400 B on PP1, plus the rank). So the digest half of the name it sends is
# this sentinel, and the worker REPLACES it with its own digest - see
# `_resolve_worker_path`, where a REAL digest that disagrees is a counted failure
# rather than a silent choice. It is not a file name this boot could ever open:
# `disk_layout.compute_geometry_digest` cannot produce twelve zeros, so a path that
# still carries it is, by construction, unreadable.
IDENTITY_DIGEST_PLACEHOLDER = "0" * disk_layout.GEOMETRY_DIGEST_CHARS

# --- the proof that the WORKER half of this path is live ---------------------
# THE ONE ASYMMETRY THIS MODULE CANNOT SEE FOR ITSELF. The scheduler and the workers
# are different processes, and an offer is only SAFE if a worker will fill the CPU
# slots the offer allocates. If the scheduler half installed and the worker half did
# not (a fork rename of one worker method, the plugin loaded in one process only), the
# fork's own `start_load_kv` would DMA slots nobody had filled and the request would be
# served from STALE KV - the one failure mode of this design that is silent, so it may
# not be left to luck.
#
# The proof is the rank file the worker ALREADY writes: `worker_diag` publishes this
# module's `disk_restore` section in `rank-*.json`, so a file that is FRESH and says
# `installed: YES` is a worker attesting that its layer is fitted. It adds NO new
# precondition (that directory is already the only way this process can know what
# exists on disk) and it is deliberately not a handshake: no socket, no new file name,
# no engine channel.
RANK_FILE_PREFIX = "rank-"
RANK_FILE_SUFFIX = ".json"
MAX_RANK_FILES_PROBED = 8
# The reader's own ceiling (`llm-api/kv_tier_diagnostics.py`): a file over it is
# unreadable THERE too, so reading more here would buy nothing.
MAX_RANK_FILE_BYTES = 256 * 1024
# A rank file is a heartbeat, not a lease: `worker_diag` rewrites it as the worker
# steps, so this is generous on purpose (an idle engine writes rarely) while still
# refusing to believe a worker that has been gone for an hour.
RANK_FILE_FRESH_SECONDS = 900.0
PROOF_CACHE_SECONDS = 1.0

# --- the bounds, and the unit of every one of them ---------------------------

RESTORE_DEADLINE_SECONDS = 45.0
"""SECONDS an in-flight disk load may stay unreported before the watchdog reports
it as a failure. Arithmetic: the measured cost of one 3,200-token span is ~10 ms
(4 slots x 11,468,800 B = 45.9 MB at the 11.5 GB/s this store measured), and an
offer is capped at `MAX_RESTORE_BLOCKS_PER_OFFER` spans, so the largest legitimate
read is 64 x 45.9 MB = 2.94 GB = ~255 ms of streaming - and 45 s is a ~175x margin
on that. It is ALSO `disk_read.LOAD_DEADLINE_SECONDS` (30 s, the reader's own
promise) + 50%: the batch deadline must always fire first, so the watchdog only
ever sees the case the batch deadline could not reach (a batch that settled into
nothing, or a step that never reached `start_load_kv`). 45 s stays far below the
point at which an operator calls a parked request a hang, and a stall it fires on
would otherwise be permanent, because the fork has no timeout for
WAITING_FOR_REMOTE_KVS."""

MAX_RESTORE_BLOCKS_PER_OFFER = 64    # SCHEDULER BLOCKS offered per request
MAX_SLOTS_PER_OFFER = 256            # CPU slots allocated per request
MAX_TRACKED_LOADS = 64               # REQUESTS with an outstanding disk load
MAX_TRACKED_SLOTS = 4096             # cpu-slot -> identity entries
MAX_INVALID_BLOCKS = 4096            # GPU block ids awaiting one report
MAX_REFUSAL_REASONS = 24             # distinct refusal reasons remembered
MAX_STRING_CHARS = 160
MAX_BOOT_LINE_CHARS = 160
CPU_SLOT_RESERVE = 64                # CPU slots the offer must leave free
MAX_PATHS_PER_STEP = 512             # load_paths entries one step may carry
MIN_WITNESS_LOG_SECONDS = 30.0       # SECONDS between witness re-emissions

OFFER_REFUSALS = ("not_configured", "disarmed", "tier_disarmed", "bad_call",
                  "bad_result", "unaligned_local_prefix", "skip_read_requested",
                  "no_tokens_left", "no_block_hashes", "no_key_helpers",
                  "no_eligible_group", "ram_shape", "no_disk_run",
                  "disk_run_below_one_block", "no_cpu_slots", "alloc_refused",
                  "no_channel", "path_build_failed", "index_refresh_failed",
                  "too_many_loads", "ram_hit_not_block_aligned",
                  "no_worker_proof")
WORKER_FAILURES = ("path_count_mismatch", "bad_path", "identity_disagreement",
                   "digest_disagreement", "fingerprint_incomplete", "no_channel",
                   "read_failed", "file_missing", "size_mismatch", "short_read",
                   "read_error", "not_ours", "not_settled", "slot_plan_failed",
                   "copy_failed", "batch_cap", "queue_full", "buffer_cap",
                   "bad_batch", "bad_job", "bad_size", "incomplete",
                   "deadline_expired", "thread_dead", "not_tracked", "abandoned")

_state: dict = {
    # install / posture
    "installed": False, "class": None, "installs": 0, "reinstalls": 0,
    "uninstalls": 0, "skip_reason": None, "nested": [], "errors": 0,
    "fallbacks": 0, "boot_line": None,
    # scheduler-side ledger
    "lookups": 0, "offers": 0, "offered_tokens": 0, "adopted": 0,
    "adopted_tokens": 0, "restorations": 0, "refused": {r: 0
                                                        for r in OFFER_REFUSALS},
    "refusals_total": 0, "last_refusal": None, "last_reason": None,
    "paths_emitted_steps": 0, "path_entries": 0, "path_entries_empty": 0,
    "path_length_violations": 0, "slots_allocated": 0, "stale_slots_dropped": 0,
    "watchdog_fires": 0, "index_refreshes": 0, "index_refresh_errors": 0,
    "loads_released": 0, "ranks_proven": 0, "proof_reads": 0,
    # worker-side ledger
    "binds": 0, "disk_binds": 0, "reads_started": 0, "reads_applied": 0,
    "bytes_applied": 0, "dma_unsettled_entries": 0, "reads_failed": {r: 0
                                                                     for r
                                                                     in
                                                                     WORKER_FAILURES},
    "invalid_reports": 0, "finished_released": 0, "worker_watchdog_fires": 0,
    "phantom_guard_trips": 0, "region_plan_errors": 0, "worker_errors": 0,
    "load_error_installs": 0,
    "witness_verdict": None, "witness_reason": None, "witness_emissions": 0,
}

# Scheduler side: request id -> load record. Bounded by MAX_TRACKED_LOADS.
_loads: dict = {}
# Scheduler side: cpu block id -> (request id, path). Bounded by MAX_TRACKED_SLOTS.
_slot_paths: dict = {}

# Worker side: request id -> in-flight record (the watchdog's table), and the
# batches those records share. Both bounded; `_invalid` is drained once per step.
_wloads: dict = {}
_wbatches: dict = {}
_invalid: dict = {}
_logged_lines: list = []
_witness_state: dict = {"line": None, "emitted_mono": None}


def reset_state_for_tests() -> None:
    """Back to a fresh boot: no ledgers, no counters, no installed layer."""
    global _loads, _slot_paths, _wloads, _wbatches, _invalid, _logged_lines
    _loads = {}
    _slot_paths = {}
    _wloads = {}
    _wbatches = {}
    _invalid = {}
    _logged_lines = []
    _witness_state.update({"line": None, "emitted_mono": None})
    _proof_cache.update({"mono": None, "result": None})
    _deferred_finished.clear()
    for key, value in {
            "installed": False, "class": None, "installs": 0, "reinstalls": 0,
            "uninstalls": 0, "skip_reason": None, "nested": [], "errors": 0,
            "fallbacks": 0, "boot_line": None, "lookups": 0, "offers": 0,
            "offered_tokens": 0, "adopted": 0, "adopted_tokens": 0,
            "restorations": 0, "refusals_total": 0, "last_refusal": None,
            "last_reason": None, "paths_emitted_steps": 0, "path_entries": 0,
            "path_entries_empty": 0, "path_length_violations": 0,
            "slots_allocated": 0, "stale_slots_dropped": 0, "watchdog_fires": 0,
            "index_refreshes": 0, "index_refresh_errors": 0, "loads_released": 0,
            "ranks_proven": 0, "proof_reads": 0,
            "binds": 0, "disk_binds": 0, "reads_started": 0, "reads_applied": 0,
            "bytes_applied": 0, "dma_unsettled_entries": 0, "invalid_reports": 0,
            "finished_released": 0, "worker_watchdog_fires": 0,
            "phantom_guard_trips": 0, "region_plan_errors": 0,
            "worker_errors": 0, "load_error_installs": 0,
            "witness_verdict": None, "witness_reason": None,
            "witness_emissions": 0}.items():
        _state[key] = value
    _state["refused"] = {name: 0 for name in OFFER_REFUSALS}
    _state["reads_failed"] = {name: 0 for name in WORKER_FAILURES}


def counters() -> dict:
    """A copy of the counters for a payload. Never the ledgers themselves."""
    out = dict(_state)
    out["refused"] = dict(_state["refused"])
    out["reads_failed"] = dict(_state["reads_failed"])
    out["tracked_loads"] = len(_loads)
    out["tracked_slots"] = len(_slot_paths)
    out["tracked_worker_loads"] = len(_wloads)
    out["pending_invalid_blocks"] = len(_invalid)
    return counters_view(out)


def counters_view(out: dict) -> dict:
    out["units"] = {
        "lookups": "lookups reaching the restore path",
        "offers": "requests a disk offer was made for",
        "offered_tokens": "tokens offered beyond the host-RAM hit",
        "installs": "times the restore layer was fitted in this process",
        "errors": "internal exceptions caught and counted by this module",
        "fallbacks": "degradations to the fork's own behaviour, each with a reason",
        "reinstalls": "times install() found the layer already fitted",
        "uninstalls": "times the layer was taken back off the fork in this process",
        "load_error_installs": "times the load-error report was fitted on a class",
        "adopted": "requests whose disk slots reached the worker",
        "adopted_tokens": "tokens the scheduler adopted from disk",
        "restorations": "loads with disk bytes reported finished",
        "refusals_total": "refusals", "paths_emitted_steps": "steps",
        "path_entries": "load_paths entries naming a file",
        "path_entries_empty": "load_paths entries naming nothing",
        "path_length_violations": "steps where the length rule was broken",
        "slots_allocated": "CPU slots allocated as disk destinations",
        "stale_slots_dropped": "mappings dropped as their load died",
        "watchdog_fires": "scheduler-side ledger expiries",
        "index_refreshes": "journal refreshes",
        "index_refresh_errors": "raised refreshes",
        "ranks_proven": "rank files attesting a live worker half",
        "proof_reads": "rank files read for that proof",
        "loads_released": "loads removed from the ledger",
        "binds": "worker binds observed", "disk_binds": "binds carrying disk work",
        "reads_started": "file reads queued", "reads_applied": "reads memmoved",
        "bytes_applied": "bytes", "dma_unsettled_entries": "load entries not DMA'd",
        "invalid_reports": "GPU block ids reported as load errors",
        "finished_released": "requests released without a DMA",
        "witness_emissions": "witness lines logged in this boot",
        "worker_watchdog_fires": "worker-side load expiries",
        "phantom_guard_trips": "invariant violations (must stay 0)",
        "region_plan_errors": "slot address plans that failed",
        "worker_errors": "raised worker-path exceptions",
        "tracked_loads": "requests with an outstanding disk load",
        "tracked_slots": "cpu-slot mappings held",
        "tracked_worker_loads": "worker watchdog entries",
        "pending_invalid_blocks": "GPU block ids awaiting their report",
    }
    return out


# --- small helpers ------------------------------------------------------------

def _fail(reason, **extra) -> dict:
    out = {"ok": False, "reason": str(reason)[:MAX_STRING_CHARS]}
    out.update(extra)
    return out


def _ok(**fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    return out


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int(value) -> int | None:
    if _plain_int(value):
        return int(value)
    if isinstance(value, (str, bytes)) and str(value).isdigit():
        return int(value)
    return None


def _clip(text, chars: int = MAX_STRING_CHARS):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= chars else text[:chars] + "..."


def _note(reason) -> None:
    _state["last_reason"] = _clip(reason)


def _refuse(kind: str, reason: str) -> None:
    """Count an offer refusal under its closed-vocabulary name."""
    name = kind if kind in _state["refused"] else "bad_call"
    _state["refused"][name] += 1
    _state["refusals_total"] += 1
    _state["last_refusal"] = _clip(f"{name}: {reason}")
    _note(reason)


def _worker_fail(kind: str, reason: str) -> dict:
    name = kind if kind in _state["reads_failed"] else "read_failed"
    _state["reads_failed"][name] += 1
    _note(reason)
    return _fail(reason, reason_kind=name)


# --- the gate -----------------------------------------------------------------

def gate() -> dict:
    """May THIS process make/serve a disk restore? Configured AND armed AND armed.

    Three switches, no new one: the root (`VLLM_E2_TIER_DISK_ROOT`, which
    `disk_spill` owns and validates), the spill tier's arm file (which expires), and
    the KV tier's own arm. A fresh boot answers `disarmed` and the wrappers cost an
    env read. Reading the arm file is `disk_spill`'s cached verdict, so this is not
    a per-step `open()`.
    """
    root = disk_spill.disk_root()
    if root is None:
        return _fail(f"{config.ENV_DISK_ROOT} is not set: there is no store a "
                     "restored file could be read from",
                     reason_kind="not_configured")
    if not disk_spill.is_armed():
        return _fail("the disk tier is disarmed: the restore path obeys the spill "
                     "tier's arm, which expires on its own",
                     reason_kind="disarmed")
    try:
        from . import backport  # noqa: PLC0415 - lazy: backport imports us lazily
        tier_armed = bool(backport.is_armed())
    except Exception as exc:  # noqa: BLE001 - an unreadable tier is a closed gate
        snapshot._error(exc, "disk_restore tier arm")
        return _fail(f"the tier's arm could not be read: {type(exc).__name__}",
                     reason_kind="tier_disarmed")
    if not tier_armed:
        return _fail("the KV tier itself is disarmed, so the connector schedules "
                     "no load for a disk offer to ride on",
                     reason_kind="tier_disarmed")
    proof = worker_proof()
    if not proof["ok"]:
        return _fail(proof["reason"], reason_kind="no_worker_proof")
    return _ok(root=root, ranks=proof["ranks"])


_proof_cache: dict = {"mono": None, "result": None}


def worker_proof(force: bool = False) -> dict:
    """Has a WORKER process attested that its restore layer is fitted?

    THE SCHEDULER CANNOT SEE THE WORKER. They are separate processes, and an offer is
    only safe if a worker will fill the CPU slots the offer allocates: if the
    scheduler half installed and the worker half did not (a fork rename of one worker
    method, the plugin loaded in one process only), the fork's own `start_load_kv`
    would DMA slots nobody had filled and the request would be served from STALE KV -
    the one failure mode of this design that is silent. So the offer requires proof.

    The proof is the rank file the worker ALREADY writes: `worker_diag` publishes this
    module's `disk_restore` section in `rank-*.json`, so a file that is FRESH and says
    `installed: YES` is a worker attesting that its layer is live. It adds no new
    precondition (that directory is already the only way this process can know what
    exists on disk) and it is not a handshake: no socket, no new file name, no engine
    channel. Cached for `PROOF_CACHE_SECONDS`; bounded reads; never raises.
    """
    now = time.monotonic()
    if not force and _proof_cache["result"] is not None \
            and isinstance(_proof_cache["mono"], (int, float)) \
            and (now - _proof_cache["mono"]) < PROOF_CACHE_SECONDS:
        return _proof_cache["result"]
    result = _read_worker_proof()
    _proof_cache["mono"] = now
    _proof_cache["result"] = result
    _state["ranks_proven"] = int(result["count"])
    return result


def _read_worker_proof() -> dict:
    directory = config.diagnostics_dir()
    if directory is None:
        return {"ok": False, "count": 0, "ranks": [],
                "reason": f"{config.ENV_DIAG_DIR} is not set, so no worker's rank "
                          "file can be read to prove the worker half of this path "
                          "is fitted"}
    try:
        names = sorted(name for name in os.listdir(directory)
                       if name.startswith(RANK_FILE_PREFIX)
                       and name.endswith(RANK_FILE_SUFFIX))
    except OSError as exc:
        return {"ok": False, "count": 0, "ranks": [],
                "reason": f"the diagnostics directory could not be listed: "
                          f"{type(exc).__name__}"}
    if not names:
        return {"ok": False, "count": 0, "ranks": [],
                "reason": "no worker has written a rank file yet, so nothing proves "
                          "the worker half of this path is fitted"}
    proven = []
    for name in names[:MAX_RANK_FILES_PROBED]:
        verdict = _rank_file_proves_restore(directory / name)
        if verdict is not None:
            proven.append(verdict)
    _state["proof_reads"] += min(len(names), MAX_RANK_FILES_PROBED)
    if proven:
        return {"ok": True, "count": len(proven), "ranks": proven[:8],
                "reason": None}
    return {"ok": False, "count": 0, "ranks": [],
            "reason": (f"{min(len(names), MAX_RANK_FILES_PROBED)} rank file(s) were "
                       "read and none reports a fitted disk restore layer "
                       "(disk_restore.installed) - so the worker half of this path "
                       "is not live, and offering disk tokens would DMA slots "
                       "nobody filled")}


def _rank_file_proves_restore(path) -> str | None:
    """One rank file: the rank it attests, or None. Never raises, never over-reads."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read(MAX_RANK_FILE_BYTES + 1)
        if len(raw) > MAX_RANK_FILE_BYTES:
            return None  # unreadable by the reader too, so not evidence of anything
        document = json.loads(raw)
        if not isinstance(document, dict):
            return None
        stamp = document.get("ts")
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
            return None
        if abs(time.time() - float(stamp)) > RANK_FILE_FRESH_SECONDS:
            return None  # a heartbeat from a worker that may be gone
        section = document.get("disk_restore")
        if not isinstance(section, dict) or section.get("installed") is not True:
            return None
        rank = document.get("rank")
        return str(rank)[:32] if rank is not None else str(path.name)[:64]
    except Exception as exc:  # noqa: BLE001 - one bad file is not evidence, that's all
        snapshot._error(exc, "disk_restore rank file")
        return None


def _may_restore_worker() -> dict:
    """The worker's view of the same gate (there is no scheduler in this process)."""
    root = disk_spill.disk_root()
    if root is None:
        return _fail(f"{config.ENV_DISK_ROOT} is not set",
                     reason_kind="not_configured")
    if not disk_spill.is_armed():
        return _fail("the disk tier is disarmed", reason_kind="disarmed")
    return _ok(root=root)


# --- the identity: which file holds the next span? ----------------------------

def whole_block_tokens(matched, block_size) -> dict:
    """`floor(matched / block_size) * block_size`, and why a partial block is 0.

    THE OFFER ARITHMETIC IS A SEPARATE DOOR because it is the rule the live box
    measured: with a 3,200-token scheduler block, 3,199 matched tokens must offer
    NOTHING (a sub-block offer is discarded by the scheduler while parking the
    request), 3,200 offers 3,200, and 6,500 offers 3,200. Exposing the arithmetic
    means a test can pin it without a fake engine, and a reader can find the rule
    by name instead of in the middle of a loop.
    """
    if not _plain_int(block_size) or block_size <= 0:
        return _fail(f"block_size must be a positive int, got {block_size!r} "
                     "(unit: tokens)")
    matched_tokens = _int(matched)
    if matched_tokens is None or matched_tokens < 0:
        return _fail(f"matched must be a non-negative int, got {matched!r} "
                     "(unit: tokens)")
    blocks = matched_tokens // int(block_size)
    return _ok(blocks=blocks,
               tokens=blocks * int(block_size),
               discarded_tokens=matched_tokens - blocks * int(block_size),
               block_size=int(block_size))


def _key_helpers() -> dict:
    """The engine's OWN two packers/view-builders, or a refusal.

    `make_block_hash_with_group_id` (`kv_cache_utils.py:74`) and
    `resolve_block_hashes` (`kv_cache_utils.py:2859`) are resolved from
    `sys.modules` exactly the way `backport._empty_metadata` resolves the metadata
    class: never imported here, and refused when absent. Using them is the whole
    safety argument for deriving keys at all - a plugin that re-implemented either
    would eventually disagree with the engine about which bytes are the group, and
    that disagreement is a file read under the wrong name.
    """
    module = sys.modules.get(KV_UTILS_MODULE)
    if module is None:
        return _fail(f"{KV_UTILS_MODULE} is not imported in this process, so the "
                     "group-qualified key cannot be built",
                     reason_kind="no_key_helpers")
    pack = getattr(module, "make_block_hash_with_group_id", None)
    resolve = getattr(module, "resolve_block_hashes", None)
    if not callable(pack) or not callable(resolve):
        return _fail(f"{KV_UTILS_MODULE} carries no callable "
                     "make_block_hash_with_group_id/resolve_block_hashes: this "
                     "fork renamed the key machinery, and guessing it is how a "
                     "file gets read under the wrong name",
                     reason_kind="no_key_helpers")
    return _ok(pack=pack, resolve=resolve)


def _eligible_groups(manager, block_size: int) -> dict:
    """The groups a disk span can be restored for, and the ones that veto it.

    A group qualifies when it is prefix-cacheable AND its block size IS the
    scheduler block size. Both halves are load-bearing. Prefix-cacheable, because
    `update_state_after_alloc` (`manager.py:440-446`) discards every other group's
    CPU hit before pairing, so a slot offered for a scratch group is a slot the
    fork never pairs and never frees. Equal to the scheduler block size, because the
    merged tuple's entries are consumed in ORDER, one whole group block each: a
    group whose blocks are smaller than the scheduler block would put our slots at
    the wrong GPU offsets while the offer arithmetic stayed a whole number of
    scheduler blocks. That is a geometry this module refuses rather than reasons
    about.
    """
    try:
        sizes = list(getattr(manager, "group_block_sizes", ()) or ())
    except Exception as exc:  # noqa: BLE001 - an unreadable layout is a refusal
        snapshot._error(exc, "disk_restore group_block_sizes")
        return _fail(f"group_block_sizes could not be read: {type(exc).__name__}",
                     reason_kind="bad_call")
    cacheable = getattr(manager, "prefix_cacheable_group_ids", None)
    if not isinstance(cacheable, (list, tuple, set, frozenset)):
        return _fail("prefix_cacheable_group_ids is not a collection, so no group "
                     "can be proven restorable", reason_kind="no_eligible_group")
    try:
        wanted = sorted({int(item) for item in cacheable})
    except Exception as exc:  # noqa: BLE001 - a hostile id list is a refusal
        snapshot._error(exc, "disk_restore prefix_cacheable_group_ids")
        return _fail(f"prefix_cacheable_group_ids is unreadable: "
                     f"{type(exc).__name__}", reason_kind="no_eligible_group")
    groups = [group for group in wanted
              if 0 <= group < len(sizes) and _int(sizes[group]) == block_size
              and _group_positionally_stable(manager, group)]
    if not groups:
        return _fail(f"no prefix-cacheable group has block_size == the scheduler "
                     f"block_size {block_size} (group sizes are {sizes[:8]}, "
                     f"cacheable {wanted[:8]}): the merged CPU-hit tuple cannot be "
                     "extended one whole block at a time for this geometry",
                     reason_kind="no_eligible_group",
                     group_block_sizes=sizes[:8],
                     prefix_cacheable_group_ids=wanted[:8])
    return _ok(groups=groups, num_groups=len(sizes),
               group_block_sizes=[int(sizes[group]) for group in groups])


def _group_positionally_stable(manager, group: int) -> bool:
    """Can this group's blocks be addressed by a POSITIONAL hash at all?

    The fork gates its own spill path on exactly this flag
    (`manager.py:872-873`: `single_type_managers[g].has_positionally_stable_blocks`
    -> `continue`), and the reason carries over untouched: a spilled file is named
    after a hash derived from the request's tokens at a fixed position, so a group
    whose block content is not a function of position (a sliding window that
    rewrites its own blocks, for instance) has no stable name to restore from.
    Reading such a group back would paste bytes computed for different positions
    into the KV, which no recompute policy can undo. Unreadable counts as unstable:
    a restore we cannot justify is a restore we do not do.
    """
    try:
        coordinator = getattr(manager, "cpu_coordinator", None)
        single = getattr(coordinator, "single_type_managers", None)
        entry = single[group] if single is not None else None
        flag = getattr(entry, "has_positionally_stable_blocks", None)
        return flag is True
    except Exception as exc:  # noqa: BLE001 - an unverifiable group is not eligible
        snapshot._error(exc, "disk_restore has_positionally_stable_blocks")
        return False


def _disk_run(request, start_block: int, groups: list, block_size: int,
              hash_block_size: int, helpers: dict) -> dict:
    """The contiguous span of scheduler blocks the SSD tier can supply.

    A block is restorable only when EVERY eligible group has a file for it: the
    fork pairs per group, so one missing group's slot would be a load entry with no
    bytes behind it - a whole-request recompute for a partially-restored span. The
    run therefore stops at the first block any group cannot name.
    """
    block_hashes = getattr(request, "block_hashes", None)
    if block_hashes is None:
        return _fail("the request carries no block_hashes",
                     reason_kind="no_block_hashes")
    try:
        resolved = {}
        for group in groups:
            # THE view the engine itself uses: `resolve_block_hashes(block_hashes,
            # hash_block_size, block_size)` is what `block_pool.cache_block_hashes`
            # (`block_pool.py:269-271`) indexes when it publishes a block's hash, so
            # index `i` of this view is exactly the digest the spilled file for group
            # `g`'s block `i` was named after. `_eligible_groups` has already proved
            # every group here has this block size, so one view serves them all.
            resolved[group] = helpers["resolve"](block_hashes, hash_block_size,
                                                 block_size)
        lengths = {}
        for group in groups:
            lengths[group] = len(resolved[group])
    except Exception as exc:  # noqa: BLE001 - a view we cannot build is a refusal
        snapshot._error(exc, "disk_restore resolve_block_hashes")
        return _fail(f"resolve_block_hashes raised {type(exc).__name__}",
                     reason_kind="no_block_hashes")
    keys: list = []
    run = 0
    while run < MAX_RESTORE_BLOCKS_PER_OFFER:
        index = start_block + run
        block_keys: list = []
        complete = True
        for group in groups:
            if index >= lengths[group]:
                complete = False
                break
            try:
                digest = resolved[group][index]
            except Exception as exc:  # noqa: BLE001 - a hole in the view is a stop
                snapshot._error(exc, "disk_restore hash view")
                complete = False
                break
            if not isinstance(digest, bytes):
                return _fail(f"the engine's block hash at index {index} is "
                             f"{type(digest).__name__}, not bytes: the group "
                             "suffix can only be packed onto raw digests",
                             reason_kind="no_block_hashes")
            try:
                packed = helpers["pack"](digest, group)
                key_hex = bytes(packed).hex()
            except Exception as exc:  # noqa: BLE001 - a packer fault is a refusal
                snapshot._error(exc, "disk_restore make_block_hash_with_group_id")
                return _fail(f"make_block_hash_with_group_id raised "
                             f"{type(exc).__name__}", reason_kind="no_key_helpers")
            if not disk_index.contains(key_hex):
                complete = False
                break
            block_keys.append((group, key_hex))
        if not complete:
            break
        keys.append(block_keys)
        run += 1
    return _ok(blocks=run, keys=keys, view_lengths=lengths)


# --- the scheduler's door 1: offer the SSD's continuation ---------------------

def offer(manager, request, num_computed_tokens, ram_result):
    """Widen the fork's `(tokens, is_async)` with the SSD's contiguous span.

    Called by the wrapper AFTER the fork's own body, with its result. Returns the
    result to hand the scheduler: on EVERY refusal path that is the fork's value,
    byte-identical and unconsumed - which is the property that makes an unarmed or
    confused restore indistinguishable from no plugin at all.

    On success it rewrites `manager._pending_cpu_hits[request_id]` into the MERGED
    tuple (the fork's CPU hit, then our freshly allocated destination slots, per
    group in the fork's own order) and returns the raised offer, so every piece of
    refcounting, pairing and state creation stays the fork's.
    """
    _state["lookups"] += 1
    try:
        return _offer(manager, request, num_computed_tokens, ram_result)
    except Exception as exc:  # noqa: BLE001 - a restore never breaks a schedule
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore offer")
        _refuse("bad_call", f"the offer path raised {type(exc).__name__}")
        return ram_result


def _offer(manager, request, num_computed_tokens, ram_result) -> dict | tuple:
    opened = gate()
    if not opened["ok"]:
        # The unarmed case, and the common one: not a refusal worth a counter,
        # because "the tier is off" is not a fact about this request.
        _note(opened["reason"])
        return ram_result
    if not isinstance(ram_result, tuple) or len(ram_result) < 2:
        _refuse("bad_result", f"the fork returned {type(ram_result).__name__} "
                              "instead of (tokens, is_async)")
        return ram_result
    ram_tokens = ram_result[0]
    if ram_tokens is None:
        ram_tokens = 0
    if not _plain_int(ram_tokens) or ram_tokens < 0:
        _refuse("bad_result", f"the fork reported {ram_tokens!r} matched tokens")
        return ram_result
    block_size = _int(getattr(manager, "block_size", None))
    hash_block_size = _int(getattr(manager, "hash_block_size", None))
    computed = _int(num_computed_tokens)
    if not block_size or block_size <= 0 or not hash_block_size:
        _refuse("bad_call", "the manager exposes no usable block_size/"
                            "hash_block_size")
        return ram_result
    if computed is None or computed < 0 or computed % block_size != 0:
        # The fork's own guard (`manager.py:334-347`) refuses an unaligned local
        # prefix for the same reason: an external suffix cannot start mid-block.
        _refuse("unaligned_local_prefix", f"{computed} local tokens is not a whole "
                                          f"number of {block_size}-token blocks")
        return ram_result
    if ram_tokens % block_size != 0:
        # A fine-grained (hash-aligned) RAM hit would interleave with our blocks
        # inside a group block, so the merged list's ORDER would no longer mean
        # "one whole block per entry". Decline rather than mis-pair.
        _refuse("ram_hit_not_block_aligned", f"the host-RAM hit is {ram_tokens} "
                                             f"tokens, not a multiple of the "
                                             f"{block_size}-token scheduler block")
        return ram_result
    if getattr(request, "skip_reading_prefix_cache", False):
        _refuse("skip_read_requested", "the request asked not to read the prefix "
                                       "cache, so the fork returned nothing and "
                                       "neither may we")
        return ram_result
    num_tokens = _int(getattr(request, "num_tokens", None))
    if num_tokens is None:
        _refuse("bad_call", "the request carries no usable num_tokens")
        return ram_result
    max_hit_len = num_tokens - 1 - computed
    if max_hit_len <= 0:
        # The fork's own rule (`manager.py:353-357`): at least one token must be
        # recomputed, or the sampler has nothing to sample.
        _refuse("no_tokens_left", f"{computed} computed of {num_tokens} tokens "
                                  "leaves nothing to restore")
        return ram_result
    groups = _eligible_groups(manager, block_size)
    if not groups["ok"]:
        _refuse(groups.get("reason_kind") or "no_eligible_group", groups["reason"])
        return ram_result
    helpers = _key_helpers()
    if not helpers["ok"]:
        _refuse("no_key_helpers", helpers["reason"])
        return ram_result
    if len(_loads) >= MAX_TRACKED_LOADS:
        _refuse("too_many_loads", f"{len(_loads)} disk loads are already in flight "
                                  f"(cap {MAX_TRACKED_LOADS})")
        return ram_result
    refreshed = _refresh_index()
    if not refreshed["ok"]:
        _refuse("index_refresh_failed", refreshed["reason"])
        return ram_result
    request_id = str(getattr(request, "request_id", "") or "")
    if not request_id or len(request_id) > 256:
        _refuse("bad_call", "the request id is unusable")
        return ram_result

    start_block = (computed // block_size) + (ram_tokens // block_size)
    run = _disk_run(request, start_block, groups["groups"], block_size,
                    hash_block_size, helpers)
    if not run["ok"]:
        _refuse(run.get("reason_kind") or "no_block_hashes", run["reason"])
        return ram_result
    if run["blocks"] <= 0:
        _refuse("no_disk_run", f"the SSD tier holds no contiguous span starting at "
                               f"block {start_block} for all of groups "
                               f"{groups['groups']}")
        return ram_result
    matched = min(run["blocks"] * block_size, max_hit_len)
    floored = whole_block_tokens(matched, block_size)
    if not floored["ok"]:
        _refuse("bad_call", floored["reason"])
        return ram_result
    disk_tokens = int(floored["tokens"])
    if disk_tokens <= 0:
        _refuse("disk_run_below_one_block", f"{matched} matched tokens is below one "
                                            f"{block_size}-token scheduler block "
                                            f"({floored['discarded_tokens']} "
                                            "tokens discarded): offering it would "
                                            "park the request for a load the "
                                            "scheduler will throw away")
        return ram_result

    blocks = disk_tokens // block_size
    per_block = len(groups["groups"])
    needed = blocks * per_block
    if needed > MAX_SLOTS_PER_OFFER:
        blocks = MAX_SLOTS_PER_OFFER // per_block
        needed = blocks * per_block
        disk_tokens = blocks * block_size
    if disk_tokens <= 0:
        _refuse("no_cpu_slots", f"one span needs {per_block} CPU slots and the cap "
                                f"is {MAX_SLOTS_PER_OFFER}")
        return ram_result
    pool = getattr(manager, "cpu_block_pool", None)
    free = None
    try:
        free = _int(pool.get_num_free_blocks())
    except Exception as exc:  # noqa: BLE001 - an unreadable pool is a refusal
        snapshot._error(exc, "disk_restore get_num_free_blocks")
        _refuse("no_cpu_slots", f"the CPU pool's free count is unreadable: "
                                f"{type(exc).__name__}")
        return ram_result
    if free is None:
        _refuse("no_cpu_slots", "the CPU pool exposes no get_num_free_blocks()")
        return ram_result
    affordable = max(0, (free - CPU_SLOT_RESERVE)) // per_block
    if affordable <= 0:
        _refuse("no_cpu_slots", f"the CPU pool has {free} free blocks and "
                                f"{CPU_SLOT_RESERVE} are reserved, so not even one "
                                "span can be given a destination")
        return ram_result
    if affordable < blocks:
        blocks = affordable
        needed = blocks * per_block
        disk_tokens = blocks * block_size
    # Every clamp above shortens the OFFER, so the resolved key list is cut to
    # match: merging more identities than there are slots would be an alignment
    # failure at the door that allocates, and one fewer would silently restore a
    # shorter span than was offered.
    keys = run["keys"][:blocks]
    allocated = _allocate(pool, needed)
    if not allocated["ok"]:
        _refuse("alloc_refused", allocated["reason"])
        return ram_result
    merged = _merge_into_pending(manager, request_id, ram_tokens, computed,
                                 block_size, groups["groups"],
                                 groups["num_groups"], allocated["blocks"],
                                 keys, opened["root"])
    if not merged["ok"]:
        _release(pool, allocated["blocks"])
        _refuse(merged.get("reason_kind") or "ram_shape", merged["reason"])
        return ram_result
    # THE MERGE IS HANDED TO THE FORK: it pops this tuple in
    # `update_state_after_alloc`, takes the leading blocks per group, touches both
    # sides and frees the untaken remainder. We own no refs of our own.
    try:
        manager._pending_cpu_hits[request_id] = merged["pending"]
    except Exception as exc:  # noqa: BLE001 - a manager we cannot write is a no-op
        _release(pool, allocated["blocks"])
        snapshot._error(exc, "disk_restore pending_cpu_hits")
        _refuse("ram_shape", f"_pending_cpu_hits could not be written: "
                             f"{type(exc).__name__}")
        return ram_result
    _register_load(request_id, merged, disk_tokens)
    _state["offers"] += 1
    _state["offered_tokens"] += disk_tokens
    _state["slots_allocated"] += needed
    _note(f"offered {disk_tokens} disk tokens for {request_id[:48]} in {blocks} "
          f"span(s) across {per_block} group(s)")
    emit_witness()
    return (int(ram_tokens) + disk_tokens, True)


def _refresh_index() -> dict:
    """Let the scheduler learn what the ranks spilled since it last asked.

    Bounded inside `disk_index.refresh` (one stat per journal, a byte cursor, a
    line cap, a 1 s floor) and never raising there; the try/except is for a fork
    that renamed something under it.
    """
    try:
        out = disk_index.refresh()
        _state["index_refreshes"] += 1
        if not out.get("ok", True):
            _state["index_refresh_errors"] += 1
            return _fail(str(out.get("reason")))
        return _ok()
    except Exception as exc:  # noqa: BLE001 - an unreadable index offers nothing
        _state["index_refresh_errors"] += 1
        snapshot._error(exc, "disk_restore refresh")
        return _fail(f"the disk index refresh raised {type(exc).__name__}")


def _allocate(pool, count: int) -> dict:
    """Destination CPU slots, through the pool API the store path uses.

    `get_new_blocks` RAISES when the count exceeds the free list
    (`core/block_pool.py:664-676`), so the free count is checked first and the call
    is still contained. NO touch(): the allocation is itself one reference, which
    the fork's `_free_pending_cpu_hit` releases and its
    `update_state_after_alloc` re-touches for the blocks it takes - adding a ref of
    our own would leak one per slot, forever.
    """
    if count <= 0:
        return _ok(blocks=[])
    try:
        blocks = list(pool.get_new_blocks(count))
    except Exception as exc:  # noqa: BLE001 - the pool says no with an exception
        snapshot._error(exc, "disk_restore get_new_blocks")
        return _fail(f"the CPU pool refused {count} blocks: {type(exc).__name__}")
    if len(blocks) != count:
        _release(pool, blocks)
        return _fail(f"the CPU pool returned {len(blocks)} of {count} blocks")
    for block in blocks:
        if _int(getattr(block, "block_id", None)) is None:
            _release(pool, blocks)
            return _fail("an allocated CPU block carries no usable block_id")
    return _ok(blocks=blocks)


def _release(pool, blocks) -> None:
    """Give slots back when a merge fails AFTER allocation. Tail-first, the order
    `manager._free_pending_cpu_hit` and the coordinator both use (freeing a chain
    root-first lets the root be evicted before the deeper blocks pinned for it)."""
    if pool is None or not blocks:
        return
    try:
        pool.free_blocks(list(reversed(list(blocks))))
    except Exception as exc:  # noqa: BLE001 - a failed undo is counted, not fatal
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore release")


def _identity_path(root: str, key_hex: str) -> dict:
    """The identity in path shape: the scheduler's half of the name.

    `build_slot_path` is the ONE name builder, so the key16/g<group> components are
    exactly the ones `disk_spill` wrote; only the digest half is the sentinel,
    because it is rank-local and the scheduler has no business guessing it.
    """
    named = disk_layout.build_slot_path(root, key_hex,
                                        IDENTITY_DIGEST_PLACEHOLDER)
    if not named["ok"]:
        return _fail(str(named["reason"]))
    return _ok(path=named["path"], key16=named["key16"],
               group_id=named["group_id"])


def _merge_into_pending(manager, request_id, ram_tokens, ram_computed, block_size,
                        groups, num_groups, allocated, keys, root) -> dict:
    """Append our slots to the fork's per-group CPU-hit lists.

    Shape contract with `manager.py:381-492`: element 0 is a sequence with ONE
    entry per group (so the fork's `for g in range(num_groups)` cannot IndexError),
    our slots go at the END of each eligible group's list (the fork consumes
    LEADING entries, so "RAM first, then disk" is the token order it will pair),
    and the tuple becomes `(merged_lists, ram_tokens + disk_tokens,
    num_computed_tokens)` with the third field untouched - the fork needs that
    local prefix to place the external blocks and must not have it re-derived.

    A RAM hit whose per-group length disagrees with its own token count is REFUSED:
    the whole merge rests on "leading entries = the RAM hit, one per whole group
    block", and if that is not true the pairing would put our bytes at GPU offsets
    belonging to someone else's tokens.
    """
    per_block = len(groups)
    ram_blocks = ram_tokens // block_size
    needed = len(keys) * per_block
    if len(allocated) != needed:
        return _fail(f"{len(allocated)} slots were allocated for {needed}",
                     reason_kind="ram_shape")
    pending = None
    try:
        table = getattr(manager, "_pending_cpu_hits", None)
        if table is not None:
            pending = table.get(request_id)
    except Exception as exc:  # noqa: BLE001 - an unreadable table is "no RAM hit"
        snapshot._error(exc, "disk_restore read pending")
    if ram_tokens > 0:
        if not isinstance(pending, tuple) or len(pending) < 3:
            return _fail(f"the fork reported {ram_tokens} host-RAM tokens but left "
                         f"no _pending_cpu_hits tuple for {request_id[:48]}, so "
                         "there is nothing to merge into", reason_kind="ram_shape")
        first = pending[0]
        if not isinstance(first, (list, tuple)):
            return _fail("the fork's pending CPU hit is not a per-group sequence",
                         reason_kind="ram_shape")
        if len(first) != num_groups:
            return _fail(f"the fork's pending CPU hit has {len(first)} group "
                         f"entries, not {num_groups}", reason_kind="ram_shape")
        lists = [list(entry or []) for entry in first]
    else:
        lists = [[] for _ in range(num_groups)]
    for group in groups:
        if group >= len(lists):
            return _fail(f"group {group} is outside the fork's {len(lists)} group "
                         "lists", reason_kind="ram_shape")
        if len(lists[group]) != ram_blocks:
            return _fail(f"group {group}'s host-RAM hit is {len(lists[group])} "
                         f"blocks but the offer says {ram_blocks}, so leading "
                         "entries are not the RAM hit", reason_kind="ram_shape")
    identities: list = []
    index = 0
    for span in keys:
        for group, key_hex in span:
            named = _identity_path(root, key_hex)
            if not named["ok"]:
                return _fail(f"the identity path for {key_hex[:20]} could not be "
                             f"built: {named['reason']}",
                             reason_kind="path_build_failed")
            block = allocated[index]
            index += 1
            lists[group].append(block)
            identities.append({"group": group, "key": key_hex,
                               "cpu_block_id": int(block.block_id),
                               "path": named["path"]})
    if index != needed:
        return _fail(f"{index} identities built for {needed} slots",
                     reason_kind="ram_shape")
    hit_length = ram_tokens + len(keys) * block_size
    return _ok(pending=(tuple(lists), hit_length, ram_computed),
               identities=identities, hit_length=hit_length)


def _register_load(request_id: str, merged: dict, disk_tokens: int) -> None:
    """Remember the load until the worker reports it, or the ledger expires it.

    The mapping `cpu_block_id -> path` is what later fills `load_paths`, and its
    lifetime is deliberately SHORTER than the fork's ownership of those slots:
    `build_load_paths` drops every mapping whose request has left
    `_reqs_to_load` BEFORE it emits anything, so a slot the fork has already freed
    can never be attributed a file in the step it is re-used. That ordering is the
    guard against the one corruption this path could cause - reading a file into a
    slot that has become someone else's host-RAM source block.
    """
    while len(_loads) >= MAX_TRACKED_LOADS:
        _evict_oldest_load()
    _loads[request_id] = {
        "request_id": request_id,
        "tokens": int(disk_tokens),
        "cpu_block_ids": [int(item["cpu_block_id"])
                          for item in merged["identities"]],
        "identities": list(merged["identities"]),
        "offered_mono": round(time.monotonic(), 6),
        "deadline_mono": round(time.monotonic()
                               + RESTORE_DEADLINE_SECONDS, 6),
        "adopted": False,
        "emitted": False,
    }
    for item in merged["identities"]:
        if len(_slot_paths) >= MAX_TRACKED_SLOTS:
            _note(f"the slot ledger is at its {MAX_TRACKED_SLOTS} cap: a span of "
                  "this offer will not be findable (a missed restore, never a "
                  "wrong one)")
            break
        _slot_paths[int(item["cpu_block_id"])] = (request_id, item["path"])


def _evict_oldest_load() -> None:
    """Drop the oldest tracked load (a dropped ledger entry costs a restore)."""
    if not _loads:
        return
    _drop_load(next(iter(_loads)), "the ledger is at its cap")


def _drop_load(request_id: str, why: str) -> None:
    record = _loads.pop(request_id, None)
    if record is None:
        return
    for cpu_block_id in record.get("cpu_block_ids") or ():
        held = _slot_paths.get(int(cpu_block_id))
        if held is not None and held[0] == request_id:
            _slot_paths.pop(int(cpu_block_id), None)
    _state["loads_released"] += 1
    _note(f"{request_id[:48]}: {why}")


def _sweep_loads(manager) -> int:
    """Forget loads the fork no longer owns, or that outlived their deadline.

    Called at the top of `build_load_paths` - i.e. BEFORE any path is emitted - and
    on every completion report. A request that was aborted, preempted-away or
    simply never adopted leaves a ledger entry whose CPU slots the fork has already
    returned to the pool; keeping that mapping is the only way this module could
    ever point a read at a slot it does not own, so the sweep is a correctness
    mechanism and not housekeeping.
    """
    dropped = 0
    now = time.monotonic()
    try:
        live = getattr(manager, "_reqs_to_load", None)
    except Exception as exc:  # noqa: BLE001 - an unreadable table stops the sweep
        snapshot._error(exc, "disk_restore sweep live")
        live = None
    for request_id in list(_loads):
        record = _loads.get(request_id)
        if record is None:
            continue
        if live is not None and request_id not in live:
            _drop_load(request_id, "the fork no longer has a load state for it "
                                   "(aborted, or never adopted)")
            dropped += 1
        elif now >= float(record["deadline_mono"]):
            _state["watchdog_fires"] += 1
            _drop_load(request_id, "its ledger deadline expired without a report")
            dropped += 1
    if dropped:
        _state["stale_slots_dropped"] += dropped
    return dropped


# --- the scheduler's door 2: name the files, index-aligned --------------------

def build_load_paths(manager, metadata) -> dict:
    """Fill `metadata.load_paths` for the fork's own `load_cpu_blocks`.

    THE LENGTH RULE IS THE CONTRACT: one entry per load entry, always, with ""
    where nothing is on disk. A list that merely omitted the un-restorable entries
    would mis-align every index the worker resolves, and a mis-aligned index is a
    file read into the wrong slot - the one failure mode this whole design exists to
    make unreachable. So the length is asserted on the way out and a violation is
    counted and blanked rather than shipped.

    Mutates the metadata in place and returns a report; the WRAPPER returns the
    fork's object untouched either way, so a refusal here is invisible to the fork.
    """
    _sweep_loads(manager)
    try:
        return _build_load_paths(metadata)
    except Exception as exc:  # noqa: BLE001 - a step without paths is a normal step
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore build_load_paths")
        return _fail(f"the path emission raised {type(exc).__name__}")


def _build_load_paths(metadata) -> dict:
    cpu_blocks = getattr(metadata, FIELD_LOAD_CPU, None)
    if not isinstance(cpu_blocks, (list, tuple)):
        return _ok(emitted=False, reason="the step carries no load_cpu_blocks")
    paths = [""] * len(cpu_blocks)
    if not cpu_blocks:
        # Publish the (empty) field anyway: the worker's test must be "does any
        # entry name a file", never "does this attribute exist on this boot".
        _set_load_paths(metadata, paths)
        return _ok(emitted=False, entries=0, reason="the step has no load entries")
    named = 0
    if len(cpu_blocks) > MAX_PATHS_PER_STEP:
        _note(f"a step carried {len(cpu_blocks)} load entries; the cap is "
              f"{MAX_PATHS_PER_STEP}, so nothing was attributed")
    elif _slot_paths:
        for index, value in enumerate(cpu_blocks):
            cpu_block_id = _int(value)
            if cpu_block_id is None:
                continue
            held = _slot_paths.get(cpu_block_id)
            if held is None:
                continue
            paths[index] = held[1]
            named += 1
            record = _loads.get(held[0])
            if record is not None:
                record["emitted"] = True
    if len(paths) != len(cpu_blocks):
        # Unreachable by construction; counted rather than trusted.
        _state["path_length_violations"] += 1
        _set_load_paths(metadata, [""] * len(cpu_blocks))
        return _fail("the length rule was violated, so every entry was blanked")
    _set_load_paths(metadata, paths)
    _state["path_entries"] += named
    _state["path_entries_empty"] += len(paths) - named
    if named:
        _state["paths_emitted_steps"] += 1
        _mark_adopted()
    return _ok(emitted=bool(named), entries=len(paths), named=named)


def _set_load_paths(metadata, paths) -> None:
    try:
        setattr(metadata, FIELD_LOAD_PATHS, paths)
    except Exception as exc:  # noqa: BLE001 - a frozen metadata is a no-op
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore set load_paths")


def _mark_adopted() -> None:
    """Count the offers that reached a worker as ADOPTED.

    "Adopted" here means the scheduler put our CPU slot ids into a step's
    `load_cpu_blocks`, which is the only observable proof that the fork accepted the
    external tokens: otherwise the live box's adoption floor (whole scheduler blocks
    only) makes an offer vanish between `get_num_new_matched_tokens` and
    `update_state_after_alloc` with nothing to measure.
    """
    for record in list(_loads.values()):
        if record["adopted"] or not record["emitted"]:
            continue
        record["adopted"] = True
        _state["adopted"] += 1
        _state["adopted_tokens"] += int(record["tokens"])


# --- the scheduler's door 3: completions, and the ledger's expiry -------------

def note_reported(manager, connector_output) -> dict:
    """A load was reported finished: those disk bytes are now live on the GPU.

    THIS is the counter the witness needs (`restorations`): not "we offered", not
    "we read a file", but a request whose disk-sourced slots were DMA'd and whose
    load the worker reported complete. The counterfactual is inherent - those tokens
    were beyond the GPU hit (the scheduler computed it and came up short) and beyond
    the host-RAM hit (this module only extends a hit that ran out) - so a
    restoration is a recompute that did not happen.
    """
    out = {"released": 0, "restorations": 0}
    try:
        reported = {str(item) for item in
                    (getattr(connector_output, "finished_recving", None) or ())}
        for request_id in reported:
            record = _loads.get(request_id)
            if record is None:
                continue
            if record["adopted"]:
                _state["restorations"] += 1
                out["restorations"] += 1
            _drop_load(request_id, "the worker reported its load finished")
            out["released"] += 1
        _sweep_loads(manager)
        if out["restorations"]:
            emit_witness()
        return out
    except Exception as exc:  # noqa: BLE001 - a completion never breaks a step
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore note_reported")
        return _fail(f"the completion report raised {type(exc).__name__}")


# --- the worker's doors -------------------------------------------------------

def _own_identity(worker) -> dict:
    """THIS rank's digest and slot width - the half of the name only a worker knows.

    `disk_store.adopt_fingerprint` is the ONE builder (it reads the slot width off
    this process's own tensors and the rank id off `worker_diag`), so the restore
    path cannot name a file a spill could never have written. A second digest site
    is exactly what `disk_spill.plan_slot_write`'s docstring refuses to allow.
    """
    try:
        store = disk_store.counters()
        digest = store.get("geometry_digest") or disk_spill.geometry_digest()
        fingerprint = store.get("fingerprint") or {}
        width = _int(fingerprint.get("total_bytes_per_block"))
        if not digest or not width:
            adopted = disk_store.adopt_fingerprint(worker)
            if not adopted.get("adopted"):
                return _fail(str(adopted.get("reason")),
                             reason_kind="fingerprint_incomplete")
            store = disk_store.counters()
            digest = store.get("geometry_digest")
            width = _int((store.get("fingerprint") or {})
                         .get("total_bytes_per_block"))
        if not digest:
            return _fail("the fingerprint was adopted but no digest came out of it",
                         reason_kind="fingerprint_incomplete")
        if not width or width <= 0:
            return _fail("this rank's slot width is unknown, so no read has an "
                         "expected size to enforce",
                         reason_kind="fingerprint_incomplete")
        return _ok(digest=str(digest), bytes=int(width))
    except Exception as exc:  # noqa: BLE001 - no identity, no read
        snapshot._error(exc, "disk_restore own identity")
        return _fail(f"the fingerprint build raised {type(exc).__name__}",
                     reason_kind="fingerprint_incomplete")


def _resolve_worker_path(scheduler_path, digest: str) -> dict:
    """Re-derive THIS rank's path from the identity the scheduler sent.

    The scheduler is authoritative about the IDENTITY (key16 + group: it read the
    engine's own block hashes) and the worker is authoritative about the GEOMETRY
    (the digest, which covers this rank's slot width). `_identity_path` sent the
    digest half as the sentinel, so the normal case is "replace the sentinel". A
    path carrying a REAL digest is the cross-check: if it is not ours, two sites
    computed the same field and disagree, which is a counted failure and never a
    silent choice of either one.

    The name is rebuildable from key16 + group alone because
    `build_slot_path`/`split_block_key` truncate the digest to KEY_DIGEST_CHARS on
    the way in: `key16 + 8 hex of group` re-splits to the same key16 and the same
    group, hence exactly the file name the spill wrote.
    """
    parsed = disk_layout.parse_path(scheduler_path)
    if not parsed["ok"] or not parsed.get("ours"):
        return _fail(f"the scheduler sent a name this plugin could not have "
                     f"written: {parsed.get('reason')}",
                     reason_kind="bad_path")
    if parsed.get("partial"):
        return _fail("the scheduler sent an in-flight .part name",
                     reason_kind="bad_path")
    key16 = parsed.get("key16")
    group_id = parsed.get("group_id")
    if not isinstance(key16, str) or not _plain_int(group_id):
        return _fail("the scheduler's name carries no resolvable identity",
                     reason_kind="identity_disagreement")
    sent = parsed.get("geometry_digest")
    if sent not in (None, IDENTITY_DIGEST_PLACEHOLDER) and sent != digest:
        return _fail(f"the scheduler's path says geometry {sent} but this rank's "
                     f"own digest is {digest}: two sites computed the same field "
                     "and disagree, so neither names a file",
                     reason_kind="digest_disagreement")
    root = disk_spill.disk_root()
    if root is None:
        return _fail(f"{config.ENV_DISK_ROOT} is not set in this worker",
                     reason_kind="no_channel")
    rebuilt = disk_layout.build_slot_path(root, key16 + f"{int(group_id):08x}",
                                          digest)
    if not rebuilt["ok"]:
        return _fail(f"this rank could not name the file: {rebuilt['reason']}",
                     reason_kind="bad_path")
    if sent == digest:
        # Not the sentinel: a second digest site exists somewhere. Harmless here,
        # and worth one line, because it is the early sign of exactly the
        # two-sites-one-field bug the sentinel rule prevents.
        _note("the scheduler sent a path already carrying this rank's digest")
    return _ok(path=rebuilt["path"], key16=key16, group_id=int(group_id),
               digest=digest)


def worker_bind(worker, metadata) -> dict:
    """Queue one read per disk-sourced load entry. Cheap, bounded, never raises.

    THE ORDER IS THE WHOLE DESIGN: reads are queued HERE (bind) and applied at
    `start_load_kv`, one forward pass later, so the engine thread pays a
    `put_nowait` per entry and never waits for NVMe. An entry whose read has not
    settled by then is dropped and recomputed - see the module docstring.
    """
    _state["binds"] += 1
    try:
        return _worker_bind(worker, metadata)
    except Exception as exc:  # noqa: BLE001 - a bind never breaks a forward pass
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore worker bind")
        return _fail(f"the bind raised {type(exc).__name__}")


def _worker_bind(worker, metadata) -> dict:
    paths = _metadata_paths(metadata)
    if not paths or not any(paths):
        return _ok(queued=0, reason="no load entry on this step names a file")
    _state["disk_binds"] += 1
    gate = _may_restore_worker()
    if not gate["ok"]:
        return _worker_fail(gate.get("reason_kind") or "disarmed", gate["reason"])
    cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
    gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    if event is None or event < 0:
        return _worker_fail("bad_batch", "the step has disk work but no load event")
    if len(paths) != len(cpu_blocks) or len(gpu_blocks) != len(cpu_blocks):
        # The alignment IS the contract; a mismatch cannot be repaired by guessing
        # which list is wrong. Every disk entry on this step becomes untracked (and
        # is therefore pruned at `start_load_kv` and never DMA'd) while the fork's
        # RAM-only entries proceed untouched.
        return _worker_fail("path_count_mismatch",
                            f"load_paths has {len(paths)} entries, "
                            f"{FIELD_LOAD_CPU} has {len(cpu_blocks)} and "
                            f"{FIELD_LOAD_GPU} has {len(gpu_blocks)}: the index "
                            "alignment is gone, so nothing on this step can be "
                            "attributed")
    identity = _own_identity(worker)
    if not identity["ok"]:
        return _worker_fail(identity.get("reason_kind") or "fingerprint_incomplete",
                            identity["reason"])
    digest = identity["digest"]
    expect_bytes = identity["bytes"]
    indices = [index for index, path in enumerate(paths) if path]
    batch_id = _batch_id(event)
    opened = disk_read.begin_batch(batch_id, len(indices),
                                   deadline_seconds=disk_read.LOAD_DEADLINE_SECONDS)
    if not opened.get("ok"):
        kind = str(opened.get("reason_kind") or "")
        return _worker_fail(kind if kind in _state["reads_failed"] else "bad_batch",
                            str(opened.get("reason")))
    jobs: dict = {}
    failures: dict = {}
    for job_id in indices:
        resolved = _resolve_worker_path(paths[job_id], digest)
        if not resolved["ok"]:
            kind = str(resolved.get("reason_kind") or "bad_path")
            failures[job_id] = (kind if kind in _state["reads_failed"]
                                else "bad_path")
            _state["reads_failed"][failures[job_id]] += 1
            continue
        asked = disk_read.request_read(batch_id, job_id, resolved["path"],
                                       expect_bytes)
        if not asked.get("ok"):
            kind = str(asked.get("reason_kind") or "read_failed")
            failures[job_id] = (kind if kind in _state["reads_failed"]
                                else "read_failed")
            _state["reads_failed"][failures[job_id]] += 1
            continue
        jobs[job_id] = {"path": resolved["path"], "key16": resolved["key16"],
                        "group": resolved["group_id"],
                        "cpu_block_id": _int(cpu_blocks[job_id]),
                        "gpu_block_id": _int(gpu_blocks[job_id]),
                        "bytes": expect_bytes}
        _state["reads_started"] += 1
    deadline = round(time.monotonic() + RESTORE_DEADLINE_SECONDS, 6)
    _wbatches[batch_id] = {"batch_id": batch_id, "jobs": jobs,
                           "failures": failures, "event": event,
                           "req_ids": _event_req_ids(metadata, event),
                           "deadline_mono": deadline}
    while len(_wbatches) > disk_read.MAX_BATCHES:
        _evict_oldest_batch()
    for request_id in _wbatches[batch_id]["req_ids"]:
        while len(_wloads) >= MAX_TRACKED_LOADS:
            _evict_oldest_worker_load()
        _wloads[request_id] = {"batch": batch_id, "deadline_mono": deadline,
                               "released": False}
    _note(f"queued {len(jobs)} disk read(s) for load event {event} "
          f"({len(failures)} immediate failure(s))")
    return _ok(queued=len(jobs), failed=len(failures), batch=batch_id)


def _batch_id(event) -> str:
    return f"kvtier-restore-{int(event)}"


def _metadata_paths(metadata) -> list:
    raw = getattr(metadata, FIELD_LOAD_PATHS, None)
    if raw is None or isinstance(raw, (str, bytes)):
        return []
    try:
        return [item if isinstance(item, str) else "" for item in raw]
    except Exception as exc:  # noqa: BLE001 - an unreadable field is no field
        snapshot._error(exc, "disk_restore read load_paths")
        return []


def _event_req_ids(metadata, event: int) -> list:
    try:
        table = getattr(metadata, FIELD_LOAD_EVENT_TO_REQS, None) or {}
        raw = list(table.get(event) or [])
        return [str(item)[:256] for item in raw][:MAX_TRACKED_LOADS]
    except Exception as exc:  # noqa: BLE001 - an unreadable map costs a report only
        snapshot._error(exc, "disk_restore read load_event_to_reqs")
        return []


def _plan_regions(worker, cpu_block_id: int, total_bytes: int) -> dict:
    """(address, bytes) per region of one CPU slot, in the SPILL's own order.

    Mirrors `disk_spill.read_slot_bytes` exactly - `sorted(name)`, one
    `_region_span` per region, widths totalling `total_bytes` - because the file's
    byte order IS that iteration order. A plan that totals anything else is a
    fingerprint describing a different engine, and applying half a slot is the
    corruption this function exists to make impossible: it returns a refusal and the
    caller applies NOTHING.
    """
    caches = getattr(worker, "cpu_kv_caches", None)
    if not isinstance(caches, dict) or not caches:
        return _fail("the worker has no cpu_kv_caches to write into",
                     reason_kind="slot_plan_failed")
    names = sorted(str(name) for name in caches)
    if len(names) > disk_spill.MAX_REGIONS:
        return _fail(f"{len(names)} regions exceed the {disk_spill.MAX_REGIONS} "
                     "cap", reason_kind="slot_plan_failed")
    spans: list = []
    total = 0
    for name in names:
        try:
            span = disk_spill._region_span(caches[name], cpu_block_id,
                                           max_bytes=total_bytes)
        except Exception as exc:  # noqa: BLE001 - a hostile tensor is a refusal
            snapshot._error(exc, "disk_restore region span")
            return _fail(f"region {name!r}: {type(exc).__name__}",
                         reason_kind="slot_plan_failed")
        if not span["ok"]:
            return _fail(f"region {name!r}: {span['reason']}",
                         reason_kind="slot_plan_failed")
        spans.append((int(span["base"]), int(span["bytes_per_file"])))
        total += int(span["bytes_per_file"])
    if total != int(total_bytes):
        return _fail(f"the slot's regions total {total} bytes but the file is "
                     f"{total_bytes}: not these bytes, never applied",
                     reason_kind="slot_plan_failed")
    return _ok(spans=spans, total=total)


def _apply_buffer(worker, job: dict, buffer, total_bytes: int) -> dict:
    """One memmove per region, on the engine thread, once the read has settled.

    This is the moment `disk_read`'s docstring reserved for it: the only point at
    which a byte may enter a CPU slot, because it is synchronous, on the thread that
    owns the slot, and reached only with a buffer whose byte count was verified
    against this rank's slot width. A cancelled read's buffer is never reached,
    which is what makes "we stopped this load" and "the slot is clean" one fact.
    """
    plan = _plan_regions(worker, int(job["cpu_block_id"]), total_bytes)
    if not plan["ok"]:
        return plan
    try:
        source = ctypes.addressof(buffer)
        offset = 0
        for base, width in plan["spans"]:
            ctypes.memmove(base, source + offset, width)
            offset += width
        if offset != int(total_bytes):
            return _fail(f"moved {offset} of {total_bytes} bytes into the slot",
                         reason_kind="copy_failed")
    except Exception as exc:  # noqa: BLE001 - a failed copy is a load failure
        snapshot._error(exc, "disk_restore memmove")
        return _fail(f"the copy raised {type(exc).__name__}",
                     reason_kind="copy_failed")
    return _ok(bytes=int(total_bytes))


def abandon_disk_entries(worker, why: str) -> dict:
    """FAIL CLOSED: prune every disk entry of this step and report all of them.

    Called when OUR OWN prep raised. The disk entries in `load_cpu_blocks` were
    allocated for bytes we have not verified, and the fork's `start_load_kv` runs
    right after this hook returns - so leaving the lists alone means the fork DMAs
    whatever those CPU slots happened to contain and reports the load as finished.
    That is the one silent-corruption shape this module can produce by itself, and
    it is exactly what the `except Exception` path used to do (a probe with
    `batch_status` made to raise left `load_cpu_blocks == [0, 1, 2]`, the event
    submitted, and no load error reported). Recompute is always correct, so on any
    failure of our own the whole step's disk work is abandoned and reported.
    """
    metadata = getattr(worker, "_connector_metadata", None)
    if metadata is None:
        return _ok(abandoned=0, reason="no connector metadata on this step")
    paths = _metadata_paths(metadata)
    disk_indices = [index for index, path in enumerate(paths) if path]
    if not disk_indices:
        return _ok(abandoned=0, reason="no disk entries to abandon")
    cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
    gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
    keep = [index for index in range(min(len(cpu_blocks), len(gpu_blocks)))
            if index not in set(disk_indices)]
    for index in disk_indices:
        if index < len(gpu_blocks):
            gpu_id = _int(gpu_blocks[index])
            if gpu_id is not None:
                _remember_invalid(gpu_id)
        _state["reads_failed"]["abandoned"] += 1
    _replace_load_lists(metadata, [cpu_blocks[i] for i in keep],
                        [gpu_blocks[i] for i in keep])
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    batch_id = _batch_id(event) if event is not None else None
    batch = _wbatches.pop(batch_id, None) if batch_id else None
    submitted = bool(list(getattr(metadata, FIELD_LOAD_CPU, None) or []))
    released = 0
    if not submitted:
        released = release_event(worker, metadata, batch, why)
    _phantom_guard(worker, metadata, submitted=submitted)
    for request_id in (batch or {}).get("req_ids") or _event_req_ids(metadata,
                                                                      event or -1):
        _wloads.pop(request_id, None)
    if batch_id is not None:
        disk_read.cancel_batch(batch_id, why)
    _note(f"abandoned {len(disk_indices)} disk entry/entries: {why}")
    return _ok(abandoned=len(disk_indices), released=released)


def worker_before_load(worker) -> dict:
    """Apply settled bytes, prune the rest, and never DMA a stale slot.

    Runs BEFORE the fork's `start_load_kv`, which is the call that submits the
    CPU->GPU copy. Per disk entry, three outcomes: its bytes are in a verified
    bounce buffer and get memmoved into the slot; or it failed for any reason
    (missing, short, size mismatch, digest disagreement, queue drop, never settled,
    expired, unattributable) and its entry is REMOVED from this step's load lists,
    which both stops the stale DMA and lets its GPU block id be reported as a load
    error.

    When pruning leaves nothing to copy, the fork's `start_load_kv` returns without
    submitting an event - and a load event that was never submitted must NOT be left
    in `_pending_load_event_indices`, because the completion watermark is monotone
    over indices and a later completion would make this one reportable as "loaded".
    `release_event` does the removal and the report in the same call, and
    `_phantom_guard` re-checks the invariant afterwards and counts any trip.
    """
    try:
        return _worker_before_load(worker)
    except Exception as exc:  # noqa: BLE001 - a load prep never breaks a forward
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore before_load")
        try:
            abandon_disk_entries(worker, f"the load prep raised "
                                         f"{type(exc).__name__}")
        except Exception as exc2:  # noqa: BLE001 - the fallback cannot be the fault
            snapshot._error(exc2, "disk_restore abandon")
            _state["worker_errors"] += 1
        return _fail(f"the load prep raised {type(exc).__name__}")


def _worker_before_load(worker) -> dict:
    metadata = getattr(worker, "_connector_metadata", None)
    if metadata is None:
        return _ok(applied=0, reason="no connector metadata on this step")
    paths = _metadata_paths(metadata)
    if not paths or not any(paths):
        return _ok(applied=0, reason="no disk entries on this step")
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    batch_id = _batch_id(event) if event is not None else None
    batch = _wbatches.pop(batch_id, None) if batch_id else None
    cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
    gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
    disk_indices = [index for index, path in enumerate(paths) if path]
    if batch is None:
        # Bind registered no batch (disarmed at bind, a bind-time refusal, or a
        # worker-side restart). Every disk entry is then unattributable, and
        # unattributable bytes must never be DMA'd: prune them all.
        batch = {"batch_id": batch_id, "jobs": {}, "failures": {}, "event": event,
                 "req_ids": (_event_req_ids(metadata, event)
                             if event is not None else []),
                 "deadline_mono": time.monotonic()}
        failed = {index: "not_tracked" for index in disk_indices}
        status = {}
    else:
        status = disk_read.batch_status(batch_id)
        failed = dict(batch["failures"])
        # `disk_read` classified the failure on its own ledger (`file_missing`,
        # `size_mismatch`, `short_read`, `read_error`, `not_ours`). Carry the NAME
        # through: "the LRU is evicting faster than we read" (file_missing) and
        # "the file is a different geometry than this boot's" (size_mismatch) and
        # "the disk returned half a slot" (short_read) are three different operator
        # actions, and flattening them into `read_failed` made the counter that is
        # supposed to explain the recomputes explain nothing. The report is capped
        # (`disk_read.MAX_JOBS_REPORT`), so a job past the cap keeps the generic
        # name - the fine grain is best-effort, the failure report is not.
        classified = {}
        for item in status.get("failures") or []:
            try:
                classified[int(item["job"])] = str(item["reason_kind"])
            except (TypeError, ValueError, KeyError):
                continue
        if len(paths) != len(cpu_blocks) or len(gpu_blocks) != len(cpu_blocks):
            # THE STEP IS UNATTRIBUTABLE. `load_paths`, `load_cpu_blocks` and
            # `load_gpu_blocks` are built together by the fork and always share a
            # length, so a disagreement means the metadata is not the object the
            # scheduler built - and with the pairing unknown, we cannot say which GPU
            # block a disk file's bytes belong to, nor even which RAM entries are
            # safe. Pruning selectively would be a guess. So the whole step's load is
            # abandoned and EVERY GPU block it named is reported as a load error:
            # recompute is always correct, and a step that recomputes is a step that
            # serves the right answer slightly later.
            failed = {index: "path_count_mismatch" for index
                      in range(len(cpu_blocks))}
            for value in gpu_blocks:
                gpu_id = _int(value)
                if gpu_id is not None:
                    _remember_invalid(gpu_id)
            _state["path_length_violations"] += 1
        else:
            for job_id, job in batch["jobs"].items():
                if job_id in failed:
                    continue
                if not status.get("known"):
                    failed[job_id] = "not_tracked"
                    continue
                buffer = disk_read.batch_buffer(batch_id, job_id)
                if buffer is None:
                    # Not DONE: still in flight, or failed. Both mean "these bytes
                    # do not exist", and waiting for the difference would put an
                    # NVMe read in the engine loop's critical path.
                    if status.get("pending"):
                        failed[job_id] = "not_settled"
                        _state["dma_unsettled_entries"] += 1
                    else:
                        reason_kind = classified.get(job_id)
                        failed[job_id] = (reason_kind
                                          if reason_kind in _state["reads_failed"]
                                          else "read_failed")
                    continue
                outcome = _apply_buffer(worker, job, buffer, int(job["bytes"]))
                if not outcome["ok"]:
                    kind = outcome.get("reason_kind")
                    failed[job_id] = (kind if kind in _state["reads_failed"]
                                      else "copy_failed")
                    continue
                _state["reads_applied"] += 1
                _state["bytes_applied"] += int(outcome["bytes"])
        if status.get("incomplete"):
            # Fewer reads than declared: a caller bug, and a batch that quietly had
            # fewer reads than slots is exactly how a stale slot reaches the DMA.
            for job_id in batch["jobs"]:
                failed.setdefault(job_id, "incomplete")
        if status.get("deadline_expired"):
            for job_id in batch["jobs"]:
                failed.setdefault(job_id, "deadline_expired")
    drop = sorted({int(job_id) for job_id in failed
                   if 0 <= int(job_id) < len(cpu_blocks)})
    drop_set = set(drop)
    already_counted = set(batch["failures"]) if batch else set()
    for job_id in drop:
        if job_id not in already_counted:
            # Bind-time failures were counted when they happened; counting them again
            # here would make one failed read two, and the counters are the only
            # evidence an operator has about which stage is losing reads.
            kind = failed.get(job_id)
            _state["reads_failed"][kind if kind in _state["reads_failed"]
                                   else "read_failed"] += 1
        if job_id < len(gpu_blocks):
            gpu_id = _int(gpu_blocks[job_id])
            if gpu_id is not None:
                _remember_invalid(gpu_id)
    if drop:
        kept_cpu = []
        kept_gpu = []
        for index in range(min(len(cpu_blocks), len(gpu_blocks))):
            if index in drop_set:
                continue
            kept_cpu.append(cpu_blocks[index])
            kept_gpu.append(gpu_blocks[index])
        _replace_load_lists(metadata, kept_cpu, kept_gpu)
    if batch_id is not None:
        disk_read.finish_batch(batch_id)
    submitted = bool(list(getattr(metadata, FIELD_LOAD_CPU, None) or []))
    released = 0
    if drop and not submitted:
        released = release_event(worker, metadata, batch,
                                 "every load entry on this step came from disk and "
                                 "none had bytes to copy")
    _phantom_guard(worker, metadata, submitted=submitted)
    for request_id in batch["req_ids"]:
        record = _wloads.get(request_id)
        if record is not None and record.get("batch") not in (None, batch_id):
            # A NEWER bind for the same request owns the record now; this step may
            # not retire another step's watchdog coverage.
            continue
        # THE WORKER WATCHDOG HAS NOTHING LEFT TO DO FOR THIS REQUEST, and keeping
        # the record was a real bug, not untidiness. Its one job is a load whose
        # step never reached `start_load_kv`; here the step DID reach it, so this
        # request's outcome is already final: either the fork submitted what
        # survived the pruning and will report the event itself, or
        # `release_event` reported it just now. A record left behind fired ~45 s
        # later and pushed `finished_recving` for a request the engine had long
        # since moved on from - a release nobody asked for, a `finished_released`
        # counter that lies, and (for a load still in flight on a stalled stream)
        # a claim that bytes were loaded before the copy that carries them had
        # finished. That last shape is the phantom this module exists to prevent,
        # arriving from the watchdog instead of from the watermark.
        _wloads.pop(request_id, None)
    applied = len([job_id for job_id in batch["jobs"] if job_id not in failed])
    kinds = sorted({str(failed.get(job_id)) for job_id in drop})[:3]
    _note(f"applied {applied} of {len(batch['jobs'])} disk read(s); dropped "
          f"{len(drop)} for {kinds or '-'}")
    return _ok(applied=applied, dropped=len(drop), released=released,
               submitted=submitted)


def _replace_load_lists(metadata, cpu_blocks, gpu_blocks) -> None:
    """Rewrite a step's load lists with the surviving entries, still aligned."""
    try:
        setattr(metadata, FIELD_LOAD_CPU, list(cpu_blocks))
        setattr(metadata, FIELD_LOAD_GPU, list(gpu_blocks))
        setattr(metadata, FIELD_LOAD_PATHS, [""] * len(cpu_blocks))
    except Exception as exc:  # noqa: BLE001 - an immutable step is a no-op
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore prune")


def _remember_invalid(gpu_block_id: int) -> None:
    if len(_invalid) >= MAX_INVALID_BLOCKS:
        _invalid.pop(next(iter(_invalid)), None)
    _invalid[int(gpu_block_id)] = round(time.monotonic(), 6)


def _phantom_guard(worker, metadata, submitted: bool) -> dict:
    """THE hazard check: an event that was never submitted must not stay pending.

    `worker._pending_load_event_indices` is what `get_finished` mines, and it mines
    it with `index <= watermark` where the watermark is monotone. So leaving a
    never-submitted index in that set is how a LATER completion gets reported as
    "this request's KV arrived" - the engine believing bytes it never received are
    valid KV. Every path that skips a submission calls `release_event` in the same
    breath; this re-checks the pair and COUNTS a trip when the invariant is broken
    (and closes it, because the right response to the hazard is to remove the index,
    not merely to notice it).
    """
    try:
        event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
        pending = getattr(worker, "_pending_load_event_indices", None)
        if event is None or event < 0 or not isinstance(pending, set):
            return _ok(checked=False)
        if submitted or event not in pending:
            return _ok(checked=True)
        _state["phantom_guard_trips"] += 1
        pending.discard(event)
        for request_id in _event_req_ids(metadata, event):
            record = _wloads.get(request_id)
            if record is not None:
                record["released"] = True
            _defer_finish(request_id)
            _state["finished_released"] += 1
        _note(f"PHANTOM GUARD: load event {event} was not submitted but was still "
              "pending; it was removed and its requests released as failures")
        return _fail("the invariant was violated, and closed", tripped=True)
    except Exception as exc:  # noqa: BLE001 - a guard that raises is not a guard
        snapshot._error(exc, "disk_restore phantom guard")
        return _fail(f"the guard raised {type(exc).__name__}")


_deferred_finished: set = set()

# KNOWN GAP, DELIBERATELY NOT CLOSED HERE (the single next action for this path).
# `finished_recving` and `get_block_ids_with_load_errors` are produced by two
# different engine calls in a step, and this module cannot choose their order: the
# model runner reads the load errors at `v1/worker/gpu/kv_connector.py:116` and
# mines `get_finished` separately. A release made INSIDE `get_finished` (this
# watchdog's, which runs there) therefore has its GPU block ids drained by the
# NEXT step's read, so the scheduler sees "the load finished" one step before it
# sees "truncate these blocks and recompute them". Under
# `kv_load_failure_policy=recompute` the truncation does arrive, but one step of
# tokens is generated from KV that was about to be invalidated. The sound fix is a
# per-request pairing: hold a release in `_deferred_finished` until the drain that
# carries ITS OWN block ids has happened, with a bounded wait (never park a
# request waiting for a report). It needs a new piece of state and a deadline, and
# the release-vs-report ordering must be settled against the fork's real step
# order before writing it - the hazard is recorded here, and pinned by
# `test_a_watchdog_expiry_releases_exactly_once`, so the next change has a test to
# extend rather than a mystery to rediscover.


def _defer_finish(request_id: str) -> None:
    if len(_deferred_finished) >= MAX_TRACKED_LOADS:
        _deferred_finished.clear()
    _deferred_finished.add(str(request_id))


def release_event(worker, metadata, batch, why: str) -> int:
    """Abandon a load event, REPORTING it finished instead of going silent.

    Two things happen together and neither is optional: the index leaves
    `_pending_load_event_indices` (the phantom fix) and every request of that event
    is queued for `finished_recving` (the "never report" fix, which with no engine
    timeout for WAITING_FOR_REMOTE_KVS would park the request for the life of the
    boot). The read batch is cancelled so its bounce buffers are dropped - a
    cancelled read never writes a destination, which is `disk_read`'s whole reason
    for existing.
    """
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    req_ids = list((batch or {}).get("req_ids") or [])
    if event is not None and event >= 0:
        try:
            pending = getattr(worker, "_pending_load_event_indices", None)
            if isinstance(pending, set):
                pending.discard(event)
        except Exception as exc:  # noqa: BLE001 - best effort, then report anyway
            snapshot._error(exc, "disk_restore discard pending")
        if not req_ids:
            req_ids = _event_req_ids(metadata, event)
    released = 0
    for request_id in req_ids:
        record = _wloads.get(request_id)
        if record is not None:
            record["released"] = True
        _defer_finish(request_id)
        _state["finished_released"] += 1
        released += 1
    batch_id = (batch or {}).get("batch_id")
    if batch_id is not None:
        disk_read.cancel_batch(batch_id, why)
        _wbatches.pop(batch_id, None)
    # MAKE THE STEP SAFE, not just silent. `release_event` is usually called from
    # `start_load_kv`, which has already pruned the entries it dropped and rewritten
    # the lists; from the watchdog it has not, and the disk entries would still be
    # sitting in `load_cpu_blocks` for the next `start_load_kv` of this metadata to
    # DMA - stale bytes, reported as loaded. So every disk entry this release speaks
    # for leaves the step's lists here, and its GPU block is reported if the caller
    # has not already reported it. This also erases the disk identities (via
    # `_replace_load_lists`), so no later pass can re-release the same load or report
    # the same block ids twice - `_handle_invalid_blocks` attributes an id to
    # whichever request holds that block NOW.
    try:
        paths = _metadata_paths(metadata)
        if any(paths):
            cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
            gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
            owned = set((batch or {}).get("jobs") or {})
            keep: list = []
            for index in range(min(len(cpu_blocks), len(gpu_blocks))):
                if not paths[index]:
                    keep.append(index)
                    continue
                if index not in owned:
                    gpu_id = _int(gpu_blocks[index])
                    if gpu_id is not None:
                        _remember_invalid(gpu_id)
            _replace_load_lists(metadata, [cpu_blocks[i] for i in keep],
                                [gpu_blocks[i] for i in keep])
    except Exception as exc:  # noqa: BLE001 - an immutable step is a no-op
        snapshot._error(exc, "disk_restore release prune")
    _note(f"released load event {event} as a failure: {why}")
    return released


def take_invalid_block_ids() -> set:
    """Drain the GPU block ids whose disk read failed. ONE read, then empty.

    The engine asks every step (`v1/worker/gpu/kv_connector.py:116`) and
    `_handle_invalid_blocks` maps an id to whichever request currently holds it, so
    a stale id would truncate an UNRELATED request later. Draining is therefore part
    of the safety story, not a convenience.
    """
    out = {int(block_id) for block_id in _invalid}
    _invalid.clear()
    if out:
        _state["invalid_reports"] += len(out)
    return out


def worker_get_finished(worker, result) -> tuple:
    """Add this module's releases to the fork's `(sending, finished_recving)`.

    The fork's value is returned UNTOUCHED on the happy path: the only mutation is a
    union when this module has something to report, so an unarmed or idle restore
    gives back the identical object.
    """
    try:
        sweep_watchdog(worker)
        if not _deferred_finished:
            return result
        sending = result[0] if isinstance(result, tuple) and len(result) > 0 else None
        recving = (result[1] if isinstance(result, tuple) and len(result) > 1
                   else None)
        merged = set(recving or ()) | set(_deferred_finished)
        _deferred_finished.clear()
        for request_id in merged:
            _wloads.pop(request_id, None)
        return (sending, merged or None)
    except Exception as exc:  # noqa: BLE001 - a report never breaks a step
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore get_finished")
        _deferred_finished.clear()
        return result


def sweep_watchdog(worker) -> dict:
    """Expire in-flight disk loads. Bounded table, every fire counted, never silent.

    THE LAST RESORT, and the reason a disk load cannot become a permanent stall: the
    batch deadline and the settle check cover the ordinary failures, and this covers
    the case where a step never reached `start_load_kv` (a no-forward step, or an
    exception upstream of us) so nobody ever looked at the batch. On expiry the load
    is reported as a failure for every GPU block it owned and its requests are
    released - a recompute under `kv_load_failure_policy=recompute`, and under no
    policy at all still a request that continues.
    """
    fired = 0
    now = time.monotonic()
    pending = getattr(worker, "_pending_load_event_indices", None)
    for request_id in list(_wloads):
        record = _wloads.get(request_id)
        if record is None:
            continue
        if record.get("released"):
            _wloads.pop(request_id, None)
            continue
        if now < float(record["deadline_mono"]):
            continue
        batch_id = record.get("batch")
        batch = _wbatches.pop(batch_id, None) if batch_id else None
        metadata = getattr(worker, "_connector_metadata", None)
        event = (batch or {}).get("event")
        live_step = (metadata is not None and event is not None
                     and _int(getattr(metadata, FIELD_LOAD_EVENT, None)) == event)
        if batch is not None:
            # Every GPU block this load owned: without the report the scheduler keeps
            # the blocks it allocated for bytes that are never coming. Entries that
            # failed at submit are not in `jobs`; `release_event`'s prune picks those
            # up from the step itself when the step is still live.
            for job in (batch.get("jobs") or {}).values():
                gpu_id = _int((job or {}).get("gpu_block_id"))
                if gpu_id is not None:
                    _remember_invalid(gpu_id)
        if batch is not None and live_step:
            # ONE RELEASE PATH. `release_event` removes the index, reports every
            # request of the event, cancels the read batch AND prunes the step's
            # disk entries - which matters here as much as in `start_load_kv`: the
            # sweep decides the load failed, and a metadata still naming those files
            # would be pruned and released a SECOND time by the next pass over the
            # same step (a second `finished_recving` for one load, and a second
            # report of the same GPU block ids, which the scheduler attributes to
            # whichever request holds those blocks by then).
            release_event(worker, metadata, batch,
                          "the restore watchdog expired the load")
        else:
            if batch is not None:
                disk_read.cancel_batch(batch_id,
                                       "the restore watchdog expired the load")
            _defer_finish(request_id)
            _state["finished_released"] += 1
        if batch is not None and not live_step:
            event = batch.get("event")
            if event is not None and isinstance(pending, set):
                pending.discard(event)
        _state["worker_watchdog_fires"] += 1
        _wloads.pop(request_id, None)
        fired += 1
    if fired:
        _note(f"the restore watchdog released {fired} load(s)")
    return _ok(fired=fired)


def _evict_oldest_batch() -> None:
    oldest = next(iter(_wbatches))
    batch = _wbatches.pop(oldest, None)
    if batch is not None:
        disk_read.cancel_batch(oldest, "the batch table is at its cap")


def _evict_oldest_worker_load() -> None:
    if not _wloads:
        return
    _wloads.pop(next(iter(_wloads)), None)
    _note("the worker ledger is at its cap; the oldest entry was dropped")


# --- the witness --------------------------------------------------------------

def witness_verdict() -> dict:
    """Can this boot PROVE a disk restore served a request? One verdict, one line.

    CONFIRMED requires `restorations >= 1`: a request whose disk-sourced CPU slots
    reached the worker's `load_cpu_blocks` AND whose load was reported
    `finished_recving`. That is the only chain of facts that means "these tokens
    were restored instead of recomputed" - offers, reads and bytes each have ways to
    be true without a single request being served.

    REFUTED names why it never happened, in the order a reader should check: the
    gate never opened, nothing was offered, offers were not adopted (the
    3,200-token floor or a clamp), or an adopted load never reported.
    """
    refused = {name: int(count) for name, count in _state["refused"].items()
               if count}
    verdict = "CONFIRMED" if _state["restorations"] >= 1 else "REFUTED"
    reason = "-"
    if verdict == "REFUTED":
        reason = _refuted_reason(refused)
    _state["witness_verdict"] = verdict
    _state["witness_reason"] = reason
    return {"verdict": verdict, "reason": reason,
            "restorations": int(_state["restorations"]),
            "offers": int(_state["offers"]),
            "offered_tokens": int(_state["offered_tokens"]),
            "adopted": int(_state["adopted"]),
            "adopted_tokens": int(_state["adopted_tokens"]),
            "refused": refused}


def _refuted_reason(refused: dict) -> str:
    """The first link in the chain that did not hold, named."""
    if _state["offers"] == 0:
        if refused:
            worst = max(refused.items(), key=lambda item: item[1])[0]
            return (f"gate_closed_or_refused:{worst} (no offer was ever made; the "
                    "refusals name why)")
        return ("never_offered: the restore path ran and offered nothing, so the "
                "SSD tier never had a contiguous span to offer")
    if _state["adopted"] == 0:
        return (f"offers_not_adopted: {_state['offers']} offer(s) reached the "
                "scheduler and none reached a worker (below the whole-scheduler-"
                "block adoption floor, or clamped to zero)")
    return ("adopted_never_reported: disk slots reached the worker but no load was "
            f"reported finished (watchdog fires: "
            f"{int(_state['worker_watchdog_fires'])}, phantom trips: "
            f"{int(_state['phantom_guard_trips'])})")


def witness_line() -> str:
    r"""THE greppable line. Fixed field order; bounded; never raises.

    `grep 'DISK[restore-witness]'` answers "does the restore serve requests, or did
    it never run" without a verb and without a judgement call.
    """
    try:
        view = witness_verdict()
        parts = ["DISK[restore-witness]",
                 f"verdict={view['verdict']}",
                 f"restorations={view['restorations']}",
                 f"offers={view['offers']}",
                 f"offered_tokens={view['offered_tokens']}",
                 f"adopted={view['adopted']}",
                 f"adopted_tokens={view['adopted_tokens']}",
                 f"reads_applied={int(_state['reads_applied'])}",
                 f"watchdog_fires={int(_state['worker_watchdog_fires'])}",
                 f"phantom_trips={int(_state['phantom_guard_trips'])}"]
        refused = view["refused"]
        head = ",".join(f"{name}:{count}"
                        for name, count in sorted(refused.items())[:4])
        parts.append(f"refused={head or 'none'}")
        if len(refused) > 4:
            parts.append(f"refused_more={len(refused) - 4}")
        parts.append(f"reason={_clip(view['reason'], MAX_BOOT_LINE_CHARS)}")
        line = " ".join(parts)
    except Exception as exc:  # noqa: BLE001 - the witness cannot be allowed to fail
        snapshot._error(exc, "disk_restore witness_line")
        line = ("DISK[restore-witness] verdict=REFUTED "
                f"unreadable={type(exc).__name__}")
    return line[:MAX_BOOT_LINE_CHARS * 4]


def emit_witness(force: bool = False) -> dict:
    """Log the witness when its verdict CHANGED, and never more than once a window.

    Once per distinct verdict is what makes it readable in a boot log; `force` is
    for a caller that wants the line right now. Re-issuing the same line every step
    would be noise, and a witness nobody can grep because it scrolled away is a
    witness that does not exist.
    """
    try:
        line = witness_line()
        now = time.monotonic()
        unchanged = _witness_state["line"] == line
        fired = _witness_state["emitted_mono"]
        if unchanged and not force:
            if isinstance(fired, (int, float)) \
                    and (now - fired) < MIN_WITNESS_LOG_SECONDS:
                return {"logged": False, "reason": "this verdict is already out",
                        "line": line}
            return {"logged": False, "reason": "unchanged verdict", "line": line}
        _witness_state["line"] = line
        _witness_state["emitted_mono"] = now
        _state["witness_emissions"] += 1
        _logger.warning("kvtier-dev-plugin: %s", line)
        return {"logged": True, "reason": None, "line": line}
    except Exception as exc:  # noqa: BLE001 - a log line never breaks a step
        snapshot._error(exc, "disk_restore emit_witness")
        return {"logged": False, "reason": f"logging raised {type(exc).__name__}",
                "line": None}


# --- the boot line and the rank file ------------------------------------------

def boot_line() -> str:
    r"""`DISK[restore] installed=... armed=... ...`, once per distinct verdict.

    The pair to `DISK[spill]`: that line says whether bytes could be WRITTEN, this
    one whether they could be READ BACK. Same posture, same reason - the refusal
    paths are the ones an operator needs, and a plugin that only logs success is a
    plugin whose failure looks exactly like idleness.
    """
    try:
        opened = gate()
        parts = ["DISK[restore]"]
        parts.append("installed={}".format("YES" if _state["installed"] else "NO"))
        parts.append("class={}".format(_state["class"] or "none"))
        parts.append("armed={}".format("YES" if opened["ok"] else "NO"))
        parts.append("root={}".format(_clip(disk_spill.disk_root(), 60) or "-"))
        parts.append("index={}".format(int(disk_index.counters()["entries"])))
        parts.append("digest={}".format(_clip(disk_spill.geometry_digest(), 16)
                                        or "none"))
        parts.append("offers={}".format(int(_state["offers"])))
        parts.append("offered_tokens={}".format(int(_state["offered_tokens"])))
        parts.append("reads_applied={}".format(int(_state["reads_applied"])))
        parts.append("ranks_proven={}".format(int(_state["ranks_proven"])))
        parts.append("deadline={}s".format(RESTORE_DEADLINE_SECONDS))
        refused = {name: int(count) for name, count in _state["refused"].items()
                   if count}
        head = ",".join(f"{name}:{count}"
                        for name, count in sorted(refused.items())[:4])
        parts.append(f"refused={head or 'none'}")
        reason = None if opened["ok"] else opened["reason"]
        if reason is None and _state["skip_reason"]:
            reason = _state["skip_reason"]
        if reason:
            parts.append(f"reason={_clip(reason, MAX_BOOT_LINE_CHARS)}")
        line = " ".join(parts)
    except Exception as exc:  # noqa: BLE001 - the fallback cannot fail
        snapshot._error(exc, "disk_restore boot_line")
        _state["fallbacks"] += 1
        line = (f"DISK[restore] installed={'YES' if _state['installed'] else 'NO'} "
                f"unreadable={type(exc).__name__}")
    return line[:MAX_BOOT_LINE_CHARS * 4]


def emit_boot_line() -> dict:
    """Log `boot_line()` once per distinct verdict per process."""
    try:
        line = boot_line()
        _state["boot_line"] = line
        if line in _logged_lines:
            return {"logged": False, "reason": "this verdict is already logged",
                    "line": line}
        _logged_lines.append(line)
        del _logged_lines[:-4]
        _logger.warning("kvtier-dev-plugin: %s", line)
        return {"logged": True, "reason": None, "line": line}
    except Exception as exc:  # noqa: BLE001 - a log line never breaks an install
        snapshot._error(exc, "disk_restore boot line")
        return {"logged": False, "reason": f"logging raised {type(exc).__name__}",
                "line": None}


def read_latency() -> tuple:
    """(avg, max) seconds per file read, from `disk_read`'s own ledger."""
    try:
        state = disk_read.counters()
        reads = int(state.get("reads_completed") or 0)
        seconds = float(state.get("read_seconds") or 0.0)
        avg = round(seconds / reads, 6) if reads else 0.0
        return avg, round(float(state.get("max_read_seconds") or 0.0), 6)
    except Exception as exc:  # noqa: BLE001 - a missing latency is (0, 0)
        snapshot._error(exc, "disk_restore read latency")
        return 0.0, 0.0


def worker_report() -> dict:
    """This module's worker verdict for the rank file, degraded to fit if needed."""
    from . import worker_diag  # noqa: PLC0415 - the seam owns its own cap
    try:
        report = {"section": "disk_restore", "writer": OWNER_TAG,
                  "installed": bool(_state["installed"]),
                  "class": _clip(_state["class"], 60),
                  "armed": bool(gate()["ok"]),
                  "binds": int(_state["binds"]),
                  "disk_binds": int(_state["disk_binds"]),
                  "reads_started": int(_state["reads_started"]),
                  "reads_applied": int(_state["reads_applied"]),
                  "bytes_applied": int(_state["bytes_applied"]),
                  "dma_unsettled": int(_state["dma_unsettled_entries"]),
                  "failed": {name: int(count) for name, count
                             in _state["reads_failed"].items() if count},
                  "invalid_reports": int(_state["invalid_reports"]),
                  "finished_released": int(_state["finished_released"]),
                  "watchdog_fires": int(_state["worker_watchdog_fires"]),
                  "phantom_trips": int(_state["phantom_guard_trips"]),
                  "deadline_seconds": RESTORE_DEADLINE_SECONDS,
                  "read_seconds_avg": read_latency()[0],
                  "read_seconds_max": read_latency()[1],
                  "reason": _clip(_state["last_reason"], 100),
                  "note": ("restores READ the SSD tier. reads_applied counts slots "
                           "whose bytes reached a CPU slot; dma_unsettled counts "
                           "those that arrived too late and were recomputed. "
                           "phantom_trips must stay 0: a trip means a load event "
                           "that was never submitted was about to be reported as "
                           "loaded.")}
        if len(json.dumps(report, default=str)) > worker_diag.MAX_CONTRIBUTION_BYTES:
            return {"section": "disk_restore", "writer": OWNER_TAG,
                    "truncated": True,
                    "installed": bool(_state["installed"]),
                    "armed": bool(gate()["ok"]),
                    "reads_applied": int(_state["reads_applied"]),
                    "bytes_applied": int(_state["bytes_applied"]),
                    "phantom_trips": int(_state["phantom_guard_trips"])}
        return report
    except Exception as exc:  # noqa: BLE001 - a report is never a step fault
        snapshot._error(exc, "disk_restore worker report")
        return {"section": "disk_restore", "writer": OWNER_TAG,
                "report_error": type(exc).__name__}


def register_contribution() -> dict:
    """Offer the restore verdict to the rank-file writer. Idempotent."""
    from . import worker_diag  # noqa: PLC0415 - one writer, and no import cycle
    return worker_diag.register_contribution("disk_restore", worker_report,
                                             owner=OWNER_TAG)


# --- install / uninstall ------------------------------------------------------

_REUSED = object()

# --- install / uninstall: the wrappers ----------------------------------------

def _layer_is_ours(method) -> bool:
    return getattr(method, OWNER_ATTR, None) == OWNER_TAG


def _layer_is_foreign(method) -> bool:
    """True only for a wrapper carrying ANOTHER PACKAGE's owner marker.

    The marker name is the whole question: our own layers (this module's, `backport`'s)
    carry `OWNER_ATTR` set to this package's tag, so anything that carries it set to
    something else is a layer whose semantics we cannot reason about and must not wrap.
    Layers that carry no marker at all - the fork's own function, or a sibling's
    `functools.wraps` composition - are ordinary callables to compose with, which is
    what lets four wrappers of `get_finished` coexist.
    """
    marker = getattr(method, OWNER_ATTR, None)
    return marker is not None and marker != OWNER_TAG


def _stamp(wrapper, name: str, original=None) -> None:
    setattr(wrapper, OWNER_ATTR, OWNER_TAG)
    if original is not None:
        setattr(wrapper, ORIGINAL_ATTR, original)
    wrapper.__module__ = __name__
    wrapper.__name__ = name


def make_wrapper(name: str, original):
    """One scheduler entry point, with the restore hook AFTER the fork's body.

    Public because `backport._rewrap` rebuilds our layers on top of a new base and
    has to rebuild THIS one exactly as `install` does - the composition rule that
    keeps `guard_mode` outermost, so an exception raised inside this module reaches
    the guard's fault counter instead of becoming a step that silently did nothing.
    Returns None for a method this module has no hook for (a refusal, never a guess).
    """
    if name == METHOD_MATCHED:
        def wrapper(self, request, num_computed_tokens):
            result = original(self, request, num_computed_tokens)
            return offer(self, request, num_computed_tokens, result)
    elif name == METHOD_BUILD_META:
        def wrapper(self, scheduler_output):
            metadata = original(self, scheduler_output)
            build_load_paths(self, metadata)   # contained: never raises
            return metadata                    # THE FORK'S OBJECT, UNTOUCHED
    elif name == METHOD_UPDATE_OUTPUT:
        def wrapper(self, connector_output):
            result = original(self, connector_output)
            note_reported(self, connector_output)
            return result                      # UNTOUCHED (the fork returns None)
    else:
        return None
    _stamp(wrapper, name, original)
    wrapper.__doc__ = (
        "kvtier-dev-plugin SSD restore layer: the tier's own "
        f"{name}, with the disk tier consulted after the fork's body. Refuses and "
        "counts unless a root is configured AND the spill tier is armed AND the "
        "tier is armed; the fork's return value passes through unchanged on every "
        "refusal path.")
    return wrapper


def _wrapped_bind(original):
    def bind_connector_metadata(self, metadata):
        original(self, metadata)        # THE FORK FIRST: it owns the event sets
        worker_bind(self, metadata)     # contained: never raises
    _stamp(bind_connector_metadata, METHOD_BIND, original)
    return bind_connector_metadata


def _wrapped_start_load(original):
    def start_load_kv(self, *args, **kwargs):
        worker_before_load(self)        # apply settled bytes, prune the rest
        return original(self, *args, **kwargs)
    _stamp(start_load_kv, METHOD_START_LOAD, original)
    return start_load_kv


def _wrapped_get_finished(original):
    def get_finished(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return worker_get_finished(self, result)
    _stamp(get_finished, METHOD_GET_FINISHED, original)
    return get_finished


_WORKER_WRAPPERS = {METHOD_BIND: _wrapped_bind,
                    METHOD_START_LOAD: _wrapped_start_load,
                    METHOD_GET_FINISHED: _wrapped_get_finished}


def _wrapped_load_errors(original):
    def get_block_ids_with_load_errors(self, *args, **kwargs):
        """The fork's own answer UNION the disk reads that failed. Drained here."""
        try:
            mine = take_invalid_block_ids()
        except Exception as exc:  # noqa: BLE001 - a report never breaks a step
            snapshot._error(exc, "disk_restore load errors")
            mine = set()
        try:
            theirs = original(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the fork's fault stays the fork's
            snapshot._error(exc, "disk_restore load errors original")
            theirs = set()
        if not mine:
            return theirs
        try:
            return set(theirs or ()) | mine
        except Exception as exc:  # noqa: BLE001 - our set alone is still a report
            snapshot._error(exc, "disk_restore load errors union")
            return set(mine)
    _stamp(get_block_ids_with_load_errors, METHOD_LOAD_ERRORS, original)
    return get_block_ids_with_load_errors


def _install_worker_method(cls, name: str):
    """Wrap one worker method. A sibling layer is composed with, never refused.

    THE ORDERING RULE THAT MAKES FOUR WRAPPERS OF ONE METHOD WORK: this module lands
    OUTERMOST, so `worker_before_load` runs BEFORE the layers under it and
    `worker_get_finished` sees the value they produced. A wrapper carrying a marker of
    OURS is reused, a wrapper carrying ANOTHER PACKAGE's marker is refused outright,
    and anything else (the fork's own function, or a sibling composed with
    `functools.wraps` so `__module__` reads as vLLM's) is composed with.
    """
    current = cls.__dict__.get(name)
    if current is None:
        return (f"{cls.__name__}.{name} is not in the class dict, so the fork "
                f"changed shape")
    if _layer_is_ours(current):
        return _REUSED
    if not callable(current):
        return f"{cls.__name__}.{name} is not callable"
    if _layer_is_foreign(current):
        return (f"{cls.__name__}.{name} carries a wrapper from another owner: "
                "refusing to stack on an unknown layer")
    if getattr(current, "__module__", None) != cls.__module__:
        _state["nested"].append(f"{name}({getattr(current, '__name__', '?')})")
    setattr(cls, name, _WORKER_WRAPPERS[name](current))
    return None


def _install_sched_method(cls, name: str):
    """Wrap one scheduler entry point UNDER our existing layers.

    Uses `backport`'s own layer walk so the guard stays ON TOP: when the class
    attribute is already a guard or kill-switch layer, the restore layer is inserted
    at the BOTTOM (directly over the fork's body) and the layers above are rebuilt.
    That is the only ordering in which `guard_mode`'s fault counter can see an
    exception raised here.
    """
    from . import backport  # noqa: PLC0415 - the layer vocabulary lives there
    current = cls.__dict__.get(name)
    if current is None:
        current = getattr(cls, name, None)
    if current is None:
        return f"{cls.__name__}.{name} does not exist: the fork renamed it"
    if not callable(current):
        return f"{cls.__name__}.{name} is not callable"
    if _layer_is_ours(current):
        return _REUSED
    layers, bottom = backport._layers_of(current)
    if layers:
        kinds = [kind for kind, _ in layers]
        if LAYER_KIND in kinds:
            _state["nested"].append(f"{name}(already {LAYER_KIND} under {kinds})")
            return _REUSED
        if bottom is None or not callable(bottom):
            return (f"the ownership chain under our layers ({kinds}) does not end "
                    f"in a callable body")
        base = make_wrapper(name, bottom)
        if base is None:
            return f"{cls.__name__}.{name} has no restore hook"
        setattr(cls, name, backport._rewrap(name, base, layers))
        _state["nested"].append(f"{name}(under {'+'.join(kinds)})")
        return None
    if getattr(current, OWNER_ATTR, None) is not None \
            and _layer_is_foreign(current):
        return f"{cls.__name__}.{name} carries a foreign owner marker"
    base = make_wrapper(name, current)
    if base is None:
        return f"{cls.__name__}.{name} has no restore hook"
    setattr(cls, name, base)
    return None


def _install_pass(cls, names, installer, report) -> bool:
    """Run one install pass. True to continue, False when a refusal is fatal."""
    global _reused_this_install
    for name in names:
        refusal = installer(cls, name)
        if refusal is _REUSED:
            _reused_this_install = True
            continue
        if refusal:
            report["reason"] = refusal
            _state["skip_reason"] = refusal
            return False
    return True


_reused_this_install = False


def install() -> dict:
    """Fit the restore layer onto whichever halves of the tier this process has.

    Idempotent, never raises, and announces its verdict in a `finally` - the same
    posture as `disk_store.install`, for the same reason: the refusal is the line an
    operator needs, and a quiet install failure looks exactly like a boot with nothing
    to restore.

    THE HALVES INSTALL APART, because the processes do not share modules: an
    EngineCore has the manager (and the connector) but never imports the worker,
    while a worker has the worker and the connector but never the manager. Requiring
    both here would mean the layer could never fit anywhere. The cross-process risk
    that tolerance opens - a scheduler offering slots no worker will fill - is closed
    by `gate()` requiring `worker_proof()`, not by this function.
    """
    if _state["installed"]:
        return {"installed": True, "reason": "already installed",
                "class": _state["class"]}
    report = {"installed": False, "reason": None, "class": None}
    global _reused_this_install
    _reused_this_install = False
    worker_refusal = None
    try:
        sched_cls = _class_of(SCHED_MODULE, SCHED_CLASS)
        worker_cls = _class_of(WORKER_MODULE, WORKER_CLASS)
        if sched_cls is None and worker_cls is None:
            report["reason"] = (
                f"neither {SCHED_MODULE}.{SCHED_CLASS} nor "
                f"{WORKER_MODULE}.{WORKER_CLASS} is imported in this process: "
                "there is nothing to hook here")
            _state["skip_reason"] = report["reason"]
            return report
        halves = []
        if sched_cls is not None:
            if not _install_pass(sched_cls, SCHED_METHODS, _install_sched_method,
                                 report):
                return report
            halves.append(sched_cls.__name__)
        if worker_cls is not None:
            if not _install_pass(worker_cls, WORKER_METHODS, _install_worker_method,
                                 report):
                return report
            error_refusal = _install_load_errors()
            if error_refusal:
                # NOT merely a gap: without the load-error report a failed read has
                # no way to say so, and a request would continue on KV that was never
                # restored. The worker half is refused rather than half-fitted; the
                # scheduler half, if this process also has one, is harmless on its
                # own only because `gate()` demands a worker's proof, and a worker
                # whose layer refused cannot produce that proof.
                _uninstall_worker_half()
                worker_refusal = (
                    f"the load-error report could not be fitted ({error_refusal}): "
                    "a failed read would have no way to say so, so the worker "
                    "half of the restore layer is refused")
                report["reason"] = worker_refusal
                _state["skip_reason"] = worker_refusal
                if not halves:
                    return report
            else:
                halves.append(worker_cls.__name__)
        if _reused_this_install:
            _state["reinstalls"] += 1
        else:
            _state["installs"] += 1
        _state["installed"] = True
        _state["class"] = "+".join(halves)
        # A REFUSED HALF MUST STAY VISIBLE. `skip_reason` is what `boot_line()`
        # prints when the gate is open, so clearing it here used to erase the only
        # record that the worker half was refused - and "the worker half was
        # refused" is the one refusal an operator has to be able to see without a
        # post-mortem, because it means a failed read in this process can never be
        # reported. The scheduler half fitted is still reported as fitted (that is
        # true, and it is inert without a worker's proof); the refusal rides along
        # in the same sentence.
        _state["skip_reason"] = worker_refusal
        register_contribution()
        success = (f"{len(SCHED_METHODS)} scheduler and "
                   f"{len(WORKER_METHODS)} worker methods consult the SSD tier "
                   f"where this process has them ({'+'.join(halves)}); inert "
                   "until a root is set AND the spill arm AND the tier arm are "
                   "raised")
        if worker_refusal:
            success = f"{success}; {worker_refusal}"
        report.update({"installed": True, "class": _state["class"],
                       "reason": success})
        return report
    except Exception as exc:  # noqa: BLE001 - installation never breaks a boot
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore install")
        report["reason"] = f"install raised {type(exc).__name__}"
        _state["skip_reason"] = report["reason"]
        return report
    finally:
        emit_boot_line()


def _class_of(module_name: str, class_name: str):
    """A tier class this process has imported, or None. Never raises."""
    module = sys.modules.get(module_name)
    candidate = getattr(module, class_name, None) if module is not None else None
    return candidate if isinstance(candidate, type) else None


def _install_load_errors():
    """Fit `get_block_ids_with_load_errors` on the connector (worker process).

    The fork's connector does not define it - it inherits the base's `return set()`
    (`kv_connector/v1/base.py:405`), which the model runner calls every step
    (`v1/worker/gpu/kv_connector.py:116`). Installing an override on the FORK'S class
    is legitimate only when the inherited body is vLLM's own, so the home module is
    checked; a third-party report is refused rather than overwritten.
    """
    module = sys.modules.get(CONNECTOR_MODULE)
    cls = getattr(module, CONNECTOR_CLASS, None) if module else None
    if not isinstance(cls, type):
        return (f"{CONNECTOR_MODULE}.{CONNECTOR_CLASS} is not imported here, so "
                "load errors have nowhere to be reported")
    current = cls.__dict__.get(METHOD_LOAD_ERRORS)
    if current is not None and _layer_is_ours(current):
        _state["load_error_installs"] += 1
        return None
    if current is not None and not callable(current):
        return f"{cls.__name__}.{METHOD_LOAD_ERRORS} is not callable"
    inherited = current if current is not None else getattr(cls, METHOD_LOAD_ERRORS,
                                                            None)
    if not callable(inherited):
        return (f"{cls.__name__} exposes no {METHOD_LOAD_ERRORS}, so the engine "
                "would never learn which blocks failed to load")
    home = getattr(inherited, "__module__", "") or ""
    if not home.startswith("vllm."):
        # Checked for an INHERITED body and for an existing override alike: an
        # override that is neither ours nor vLLM's belongs to another package, and
        # wrapping it would silently adopt its facts as ours - or drop them.
        return (f"{METHOD_LOAD_ERRORS} is defined in {home!r}, not in vllm's "
                "own tree: refusing to overwrite a third party's load-error "
                "report")
    setattr(cls, METHOD_LOAD_ERRORS, _wrapped_load_errors(inherited))
    _state["load_error_installs"] += 1
    return None


def ensure_installed(connector=None) -> dict:  # noqa: ARG001 - the handshake's object
    """The worker-handshake entry point: install, ignoring the connector object.

    `engine_side._on_worker_report` calls every listener as `hook(connector)`; this
    module resolves its classes BY NAME from the already-imported modules rather
    than reaching into an instance, so the argument is genuinely unused - the same
    honest signature `disk_store.ensure_installed` has.
    """
    return install()


def _uninstall_worker_half() -> None:
    """Undo the worker methods only - used when the load-error report cannot fit.

    The scheduler half is deliberately NOT undone here: in an EngineCore process the
    two halves are independent, and a half-removed worker layer would leave the class
    in a state no later install could reason about. The scheduler half is made safe by
    the gate, not by teardown.
    """
    cls = _class_of(WORKER_MODULE, WORKER_CLASS)
    if cls is None:
        return
    for name in WORKER_METHODS:
        current = cls.__dict__.get(name)
        if current is None or not _layer_is_ours(current):
            continue
        original = getattr(current, ORIGINAL_ATTR, None)
        if callable(original):
            setattr(cls, name, original)


def uninstall() -> dict:
    """Put the fork's methods back exactly. Test teardown and a clean undo."""
    report = {"uninstalled": False, "restored": [], "reason": None}
    try:
        for module_name, class_name, names in (
                (SCHED_MODULE, SCHED_CLASS, SCHED_METHODS),
                (WORKER_MODULE, WORKER_CLASS, WORKER_METHODS)):
            module = sys.modules.get(module_name)
            cls = getattr(module, class_name, None) if module else None
            if not isinstance(cls, type):
                continue
            for name in names:
                current = cls.__dict__.get(name)
                if current is None or not _layer_is_ours(current):
                    continue
                original = getattr(current, ORIGINAL_ATTR, None)
                if not callable(original):
                    # A restore layer UNDER a rebuilt guard has no restorable
                    # original of its own to hand back: the layer above it does, and
                    # restoring here would delete the guard. Say so, and leave it.
                    report["reason"] = (f"{cls.__name__}.{name} has no restorable "
                                        "original (a layer of ours sits above it)")
                    continue
                setattr(cls, name, original)
                report["restored"].append(f"{cls.__name__}.{name}")
        connector_module = sys.modules.get(CONNECTOR_MODULE)
        connector_cls = (getattr(connector_module, CONNECTOR_CLASS, None)
                         if connector_module else None)
        if isinstance(connector_cls, type):
            current = connector_cls.__dict__.get(METHOD_LOAD_ERRORS)
            if current is not None and _layer_is_ours(current):
                original = getattr(current, ORIGINAL_ATTR, None)
                if callable(original):
                    setattr(connector_cls, METHOD_LOAD_ERRORS, original)
                else:
                    delattr(connector_cls, METHOD_LOAD_ERRORS)
                report["restored"].append(f"connector.{METHOD_LOAD_ERRORS}")
        _state["uninstalls"] += 1
        _state["installed"] = False
        report["uninstalled"] = bool(report["restored"])
        if not report["uninstalled"] and not report["reason"]:
            report["reason"] = "no layer of ours was installed on any class"
        return report
    except Exception as exc:  # noqa: BLE001 - a teardown never raises
        snapshot._error(exc, "disk_restore uninstall")
        report["reason"] = f"uninstall raised {type(exc).__name__}"
        return report


# --- the view ----------------------------------------------------------------

def section() -> dict:
    """This module's state for a payload or a verb. Reads no files, never raises."""
    view = counters()
    view["gate"] = gate()
    view["deadline_seconds"] = RESTORE_DEADLINE_SECONDS
    view["limits"] = {
        "max_restore_blocks_per_offer": MAX_RESTORE_BLOCKS_PER_OFFER,
        "max_slots_per_offer": MAX_SLOTS_PER_OFFER,
        "max_tracked_loads": MAX_TRACKED_LOADS,
        "max_tracked_slots": MAX_TRACKED_SLOTS,
        "max_paths_per_step": MAX_PATHS_PER_STEP,
        "cpu_slot_reserve": CPU_SLOT_RESERVE,
        "identity_digest_placeholder": IDENTITY_DIGEST_PLACEHOLDER,
    }
    average, maximum = read_latency()
    view["read_latency_seconds"] = {"avg": average, "max": maximum}
    view["witness"] = witness_verdict()
    view["witness_line"] = witness_line()
    view["index"] = disk_index.section()
    view["read"] = disk_read.section()
    view["worker_proof"] = worker_proof()
    view["offer_refusal_reasons"] = list(OFFER_REFUSALS)
    view["worker_failure_reasons"] = list(WORKER_FAILURES)
    view["note"] = ("the SSD restore path: it EXTENDS the fork's own load with "
                    "spans read back from disk, publishes nothing into any hash "
                    "index, and reports every failure as a load error so the "
                    "scheduler recomputes instead of trusting bytes it never "
                    "received. Inert until a root is set AND the spill tier's arm "
                    "AND the tier's arm are raised.")
    return view
