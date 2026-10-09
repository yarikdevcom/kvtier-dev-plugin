"""Capability 5: per-rank worker visibility, written where only a worker can see it.

WHY A FILE, AND WHY FROM THE WORKER
-----------------------------------
The UTILITY channel dispatches with `getattr(self, method_name)` on the
**EngineCore** object (vllm/v1/engine/core.py:1669-1671), so nothing in a worker
process is reachable over HTTP: no per-rank arena, no bytes pinned, no copy
counters. A worker therefore writes one small JSON file per rank into the
diagnostics directory the operator names, atomically (`tmp` + `os.replace`), on a
budget. This is the plugin's ONLY writer and the ONLY exception to "no file I/O
on an engine path" - and it runs in a worker, never on the EngineCore loop thread.

The proxy host already has a reader for this directory
(`llm-api/kv_tier_diagnostics.py`, "FILE CONTRACT"); it treats every field as
untrusted input, caps the file size, and classifies anything it cannot read as
`malformed`. That reader is the reason the naming and the schema below are a
contract, and the reason every field here is either a real reading or absent.

FILE NAMING (must match `kv_tier_diagnostics.writer_filename` exactly)
---------------------------------------------------------------------
    <VLLM_E2_TIER_DIAG>/rank-PP<pp>_TP<tp>.json        one file per rank
    <VLLM_E2_TIER_DIAG>/.rank-PP<pp>_TP<tp>.json.<pid>.<seq>.tmp   the tmp half

The rank id is `PP<pp_rank>_TP<tp_rank>` - the reader's `_is_worker_id` accepts
that shape and nothing else, and it is the fork's own handshake key
(`pp_rank = get_pp_group().rank_in_group`, `tp_rank = get_tp_group().rank_in_group`,
gpu_worker.py:885-886). A file whose `rank` field does not match its file name is
classified mislabeled by the reader, so both come from the same string here.

SCHEMA (version 1; the reader's `KV_TIER_SNAPSHOT_VERSION`)
-----------------------------------------------------------
    {"version": 1, "epoch": "<8 hex>", "ts": 1790000000.123,
     "role": "worker", "rank": "PP0_TP0",
     "plugin_version": "0.2.50", "plugin_version_reason": null,
     "plugin_version_scope": "<which process this value describes>",
     "arena": {"bytes_total": int, "bytes_used": int, "num_blocks": int,
               "block_bytes_by_group": [int, ...] | null},
     "counters": {"stored": int, "restored_hits": int,
                  "store_events_pending": int, "load_events_pending": int},
     "writer": {"plugin": "kvtier-dev-plugin/1", "writes": int, ...},
     "error": null}

THREE NAMES THAT LOOK LIKE VERSIONS AND ARE NOT THE SAME FACT:
  * `version` = 1 is the FILE SCHEMA (the reader's `KV_TIER_SNAPSHOT_VERSION`). It
    says "you can parse me", never "this build is 1".
  * `writer.plugin` = "kvtier-dev-plugin/1" is a TAG (a trailing `/1` is part of the
    tag's shape, not a release number), and `writer.heartbeat`/`version` fields in
    contributed sections are those sections' own protocol/schema numbers.
  * `plugin_version` (0.2.50+) is the version of the CODE THIS WORKER IMPORTED,
    read live from `kvtier_dev_plugin.__version__` by the writer process. A worker
    answers no HTTP route, so this file is the only place its loaded version can
    ever be published. It is a writer-owned top-level key, listed in `KEEP_KEYS`,
    so the byte-ceiling triage sheds everything else before it.

    Optionally, a sibling per CONTRIBUTING worker module: today
    `"disk": {...}` from `disk_store`. That is the contribution seam - another
    module hands `worker_diag` a producer and `worker_diag` validates and writes,
    so ONE process still has ONE writer per file. The reader ignores a top-level
    key it does not read (its own `gpu_census` precedent), so an extra sibling
    cannot make a file malformed; the byte ceiling and the allow-list are what
    keep that true, and a contributor that raises or overruns costs its own
    section only.

Which fields the reader REQUIRES, and what this writer can honestly supply:
  `version`, `ts`, `role`, `rank`       always, exactly as above.
  `epoch`     per-process `snapshot.epoch()` UNLESS `VLLM_E2_TIER_DIAG_EPOCH` is
              set. SET THAT ONE: the reader compares epochs across writers and
              declares equality `meaningless` when they disagree, and a worker
              process cannot learn the EngineCore's boot id (the handshake flows
              worker -> EngineCore, never the other way). `writer.epoch_scope`
              says which of the two this file used, so the reader's verdict is
              never a surprise.
  `arena.bytes_total`  the arena's real size: `sum(t.nbytes for t in
              handler.cpu_kv_caches.values())` (worker.py:226-235 allocates one
              CPU tensor per GPU region). Falls back to the configured
              `handler.cpu_capacity_bytes` (worker.py:42) when the caches are not
              up yet, and `writer.bytes_total_source` says which.
  `arena.bytes_used`   REQUIRED by the reader, so a number is always present -
              and it is a PROVEN FLOOR, not the pool's occupancy: the bytes of
              the blocks this rank has bound metadata for right now
              (`len(metadata.store_cpu_blocks) * bytes_per_block` while a store
              is pending, else 0). The CPU pool's real occupancy lives in the
              scheduler process and is not visible here; `snapshot` publishes
              that half. `writer.bytes_used_meaning` states this in the file.
  `arena.num_blocks`   `handler.num_cpu_blocks` - this rank's arena slots
              (worker.py:162), which is exactly what the CPU capacity clamp in
              `cpu_capacity.py` reports through the handshake.
  `arena.block_bytes_by_group`  per KV group, `spec.page_size_bytes` ALONE. On the
              WORKER side of the merged `KVCacheConfig` that number is already the
              group's whole-block total: `MambaSpec.merge()` SUMS `page_size_bytes`
              across the merged layers (`kv_cache_interface.py:1151-1160`) and
              `MambaSpec.page_size_bytes` multiplies by the spec's OWN `num_speces`,
              which the merge has just folded in (`:1135-1143`). Multiplying by
              `len(group.layer_names)` on top counted the layers twice - and it did so
              LOUDLY WRONG in the field: the live PP0 published `[80281600, 58982400,
              58982400, 1638400, 82944, 132710400]`, whose first entry is 7x the entire
              spill file for that rank (11,468,800 B), and the two ranks published
              DIFFERENT values for the same model. NO I/O path reads this field; the
              spill width is `disk_spill.measured_slot_bytes` and the restore width is
              `disk_read.MAX_BYTES_PER_READ`'s unit, both measured. It is published for
              capacity arithmetic, and a capacity number that is 7x too big is worse
              than none - which is why the fix is to stop multiplying rather than to
              document the multiplication. `arena.layer_count_by_group` keeps the layer
              count available separately, and `arena.bytes_per_block` is the rank's
              MEASURED slot width (what a spill file actually is).
              `writer.block_bytes_by_group_meaning` says all of this in the file.
  `arena.free_blocks`  DELIBERATELY ABSENT: the CPU block pool is scheduler-side
              state. The reader notes its absence rather than trusting a guess.
  `counters.stored` / `restored_hits`  blocks this rank handed to its copy
              streams since boot, counted at `bind_connector_metadata`
              (worker.py:246) from the step's own
              `store_cpu_blocks` / `load_gpu_blocks` (metadata.py:27-28).
              SUBMITTED, not confirmed-complete; `writer.completion_note` says
              so, and the pending/hwm fields are next to them.
  `counters.store_events_pending` / `load_events_pending`  lengths of
              `handler._pending_store_event_indices` /
              `_pending_load_event_indices` (worker.py:72-73).
  `counters.dropped_cpu_full`, `counters.boundary_stored`  DELIBERATELY ABSENT:
              scheduler-side counters (`BoundaryStoreStats`, manager.py:125-132).
              The reader notes them and defaults them to 0; a worker inventing
              them would be worse than a note.
  `equality_sample`, `parked`  ABSENT: byte-comparing an arena or naming a parked
              prefix is scheduler/CUDA-side work this plugin never does.
  `error`     null, or the writer's own last failure as a string.

`writer.heartbeat` (0.2.47) is the writer's OWN PULSE, in the one artifact a worker
 definitely produces. A worker answers no HTTP route - `/plugins/kvtier/snapshot` is
 served by the API server, which honestly reports `heartbeats=0 heartbeat_threads=0`
 about a process it has never seen - so the state of the mechanism that keeps THIS file
 fresh is published inside it, with the SAME key set in every state:

     {"state": "beating" | "alive_not_beating" | "started_not_alive"
               | "never_started" | "not_applicable",
      "state_reason": str|null, "alive": bool, "started": bool,
      "beats": int, "beat_writes": int, "inflight_beats": int,
      "interval_seconds": float, "window_seconds": float,
      "beats_in_window": int|null, "last_beat_age_seconds": float|null,
      "last_beat_duration_seconds": float|null, "errors": int,
      "fatal_beats": int, "stalled_beats": int, "jammed_beats": int,
      "floor_skips": int, "respawns": int, "writer_pid": int,
      "last_error": str|null}

 HOW TO READ IT: `beating` in a file whose `ts` is old means the writer stopped AFTER
 writing (the process died, or its supervisor did); `alive_not_beating` means the
 thread is alive and NOT writing - a blocked contributor, and `state_reason` plus
 `last_beat_duration_seconds`/`stalled_beats` say so while the file is still fresh;
 `never_started` names what refused it; `not_applicable` says this process is not a
 worker rank-file writer. `writer_pid` distinguishes two writers of one file name.
 A reader that does not know this key ignores it, exactly as it ignores `gpu_census`.

BUDGET (the whole reason this is allowed on a per-step path)
------------------------------------------------------------
`get_finished` runs once per step (worker.py:305) and is the write trigger:

  * a write happens at most once every `VLLM_E2_TIER_DIAG_INTERVAL` steps
    (default 50) OR when the reported facts CHANGED (a small int signature, so
    the comparison is not a serialisation), and
  * never more than once per `VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS` (default
    1.0 s), so a churning signature cannot turn this into a per-step writer, and
  * the document is capped at `MAX_FILE_BYTES` (well under the reader's
    256 KiB cap), and
  * with `VLLM_E2_TIER_DIAG` unset NOTHING is wrapped and NOTHING is written -
    the default is that this plugin creates no files anywhere.

Stdlib only (`json`, `os`, `sys`, `time`). No torch: the tensors are only ever
touched by reading an existing attribute (`nbytes`) off an object the fork handed
us. Every entry point catches `Exception`, never `BaseException`, and counts
through `snapshot._error`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import weakref

from . import config, loaded_version, snapshot

_logger = logging.getLogger("kvtier_dev_plugin.worker_diag")
# Once-per-episode notices, the `disk_restore.notice_once` idiom. The one thing this
# module must never do is log per step or per beat; the one thing it MUST do is say out
# loud, exactly once, when the mechanism that keeps the proof alive did not start.
_notices: set = set()

TARGET_MODULE = "vllm.v1.simple_kv_offload.worker"
TARGET_CLASS = "SimpleCPUOffloadWorker"
METHOD_GET_FINISHED = "get_finished"
METHOD_BIND_METADATA = "bind_connector_metadata"
WRAPPED_METHODS = (METHOD_BIND_METADATA, METHOD_GET_FINISHED)

OWNER_ATTR = "__kvtier_dev_workerdiag_owner__"
OWNER_TAG = "kvtier-dev-plugin/workerdiag-1"
ORIGINAL_ATTR = "__kvtier_dev_workerdiag_original__"

# A function this package installed itself (another layer on the SAME method, e.g.
# `blockdump`'s wrapper on `get_finished`) is composed with, never refused - the
# `guard_mode.SIBLING_PACKAGE` rule. Refusing it would make the install ORDER a
# silent data-loss switch: whichever facility landed second would report a refusal
# and the rank files this writer produces would simply stop existing, which is the
# one input the proxy host's reader has. A marker or definition owned by ANYONE
# else is still refused outright.
SIBLING_PACKAGE = (__package__ or "kvtier_dev_plugin") + "."

SCHEMA_VERSION = 1          # the reader's KV_TIER_SNAPSHOT_VERSION
# The three version keys, published writer-owned at the top level of the file, and
# exported so a test can name them instead of copying them. `plugin_version` is the
# code THIS worker loaded; `version` above stays the file's schema number.
PLUGIN_VERSION_FIELDS = loaded_version.VERSION_FIELDS
# The keys `_clip_document` may never drop. The first seven are what the proxy
# host's reader REQUIRES (a file missing one is `malformed`); the rest are the
# writer's own optional halves, which the triage sheds in a fixed order. Anything
# NOT listed - a contributed section, or a field a future writer adds - is what
# gets dropped first, so a byte ceiling can never cost a required field. The three
# version keys are listed because "which build wrote this file?" must survive even
# a trimmed file - a rank file that cannot say what wrote it is the exact hole this
# release closed.
KEEP_KEYS = ("version", "epoch", "ts", "role", "rank", "arena", "counters",
             "writer", "error", "parked", "equality_sample",
             "plugin_version", "plugin_version_reason", "plugin_version_scope")
ROLE = "worker"
MAX_FILE_BYTES = 32 * 1024  # the reader's own cap is 256 KiB; we stay far below
MAX_STRING_CHARS = 200
MAX_GROUPS = 32
WRITER_TAG = "kvtier-dev-plugin/1"

# THE RANK FILE'S NAME, PUBLISHED. `disk_restore` reads `rank-*.json` and `disk_spill`
# lists the same files to learn which peers exist; both used to carry their own copy of
# the prefix and suffix, and a writer that renamed its file would have silently disabled
# both readers (a peer map that sees no rank files believes every sibling is gone). This
# is the single place the name is spelled, and the two readers import these.
RANK_FILE_PREFIX = "rank-"
RANK_FILE_SUFFIX = ".json"

# --- the contribution seam: ONE writer, several contributors -----------------
# A worker-side module that has facts for the operator (today `disk_store`: the
# SSD tier's own verdict, `disk_restore`: the SSD tier's READ-BACK verdict, and
# `disk_restore_verify`: the RESTORE direction's byte-for-byte verdict) must
# NOT write a file. A second writer of
# `rank-<rank>.json` would race the atomic `os.replace`, and a second file name
# would be a second contract the proxy host's reader has never heard of. So other
# modules hand this one a producer and stay silent; `worker_diag` validates and
# writes, and it is the only code that opens the file.
#
# THREE LIMITS, because a contribution is a guest in someone else's file:
#   * `CONTRIBUTION_SECTIONS` is a HARD allow-list of the top-level keys a
#     contributor may add. It is the whole shape of the trust: `"disk"`,
#     `"disk_restore"`, `"disk_shadow"` and `"disk_gpu_shadow"` are listed,
#     `"arena"`/`"counters"`/`"version"` are not, so a contributor cannot overwrite a
#     field the reader REQUIRES. An unknown section name is refused at register time,
#     not at write time. `disk_restore_verify` (0.2.39) was the fifth name on that list
#     and the fifth `MAX_CONTRIBUTIONS`: the restore-side verifier has a worker-side
#     verdict (did the bytes that came back from NVMe reach the GPU window?) that no
#     other process can report, and it registers beside `disk_restore` from
#     `disk_restore.install()`. `disk_gpu_write` (0.2.43) is the SIXTH, and it moved the
#     cap for the reason this comment exists: the cap and the allow-list were the same
#     five names, and the seam REFUSES an unregistered section, so a sixth section
#     without the raise would have been SILENTLY ABSENT from the rank file - the one
#     outcome a diagnostic can least afford, on the flag that writes into live KV. Its
#     verdict (did OUR OWN copy land where we addressed it?) is likewise reportable by
#     no other process, and it registers from the same installer.
#     THE CAP MOVING 4 -> 5, AND NOW 5 -> 6, IS THE WHOLE RISK OF THAT CHANGE, and it is
#     a risk to the FILE rather than to the reader: six sections can ask for
#     6 x `MAX_CONTRIBUTION_BYTES` = 24 KiB of a `MAX_FILE_BYTES` = 32 KiB file, so the
#     contributions stay the first thing `_clip_document` sheds and every field the
#     reader REQUIRES survives a full house of them - asserted, not assumed, by
#     `tests/test_disk_restore_verify.py`, which writes a rank file carrying all six
#     sections and checks the required keys survive and the file stays under the cap.
#   * `MAX_CONTRIBUTIONS` bounds how many sections exist at all.
#   * `MAX_CONTRIBUTION_BYTES` bounds ONE section, serialized; a section over it is
#     REFUSED (and counted), never truncated - half a section reads as a fact with
#     something missing.
# Compatibility with the reader (`llm-api/kv_tier_diagnostics.py`) is not a hope,
# it is the file's own design: it reads a fixed set of top-level keys and ignores
# any other (`gpu_census` at :210 and `parked` are the existing optional
# precedents), so an extra sibling key cannot make a file `malformed`. What CAN
# make a file malformed is the 256 KiB read cap, a missing required field, or a
# `rank`/filename mismatch - so the byte ceiling and the required-field rule are
# enforced here, and `epoch`/`version`/`writer` stay exactly as they were:
# a contribution never touches them.
CONTRIBUTION_SECTIONS = ("disk", "disk_restore", "disk_shadow", "disk_gpu_shadow",
                         "disk_restore_verify", "disk_gpu_write")
# SIX. `disk_gpu_shadow` (0.2.37) was the fourth, `disk_restore_verify` (0.2.39) the
# fifth and `disk_gpu_write` (0.2.43) the sixth, and note that they are NOT registered
# by one installer: `disk_store.install()` owns `disk`, `disk_shadow` and
# `disk_gpu_shadow`, while `disk_restore.install()` owns `disk_restore`,
# `disk_restore_verify` and `disk_gpu_write`. So no single process necessarily registers
# all six, and the cap is the size of the whole allow-list rather than of one
# installer's share of it. Cap and tuple are the same number by construction, and
# raising one without the other means a new section is REFUSED at register time and the
# rank file simply lacks it - which is why the two are asserted against each other in
# `tests/test_worker_diag.py`.
MAX_CONTRIBUTIONS = 6
MAX_CONTRIBUTION_BYTES = 4096
MAX_CONTRIBUTION_NOTES = 4

_state: dict = {
    "enabled": False,        # VLLM_E2_TIER_DIAG set at install time
    "installs": 0,
    "reinstalls": 0,
    "uninstalls": 0,
    "skips": 0,
    "skip_reason": None,
    "skip_detail": [],
    "nested": [],
    "class": None,
    "installed": False,
    "dir": None,
    "rank": None,
    "rank_source": None,
    "epoch_scope": None,
    "interval_steps": None,
    "min_interval_seconds": None,
    "steps": 0,
    "steps_since_write": 0,
    "writes": 0,
    "write_errors": 0,
    "budget_skips": 0,
    "store_blocks_submitted": 0,
    "load_blocks_submitted": 0,
    "completion_reported": 0,
    "last_write_mono": None,
    "last_signature": None,
    "last_error": None,
    "last_path": None,
    "bytes_written": 0,
    "fallbacks": 0,
    # THE HEARTBEAT (0.2.45). The rank file is a proof of life and it used to have no
    # pulse of its own: only an engine step rewrote it, so an idle worker's proof aged
    # past `disk_restore.RANK_FILE_FRESH_SECONDS` and the scheduler stopped offering
    # disk restores to a worker that was perfectly alive. These keys are how that is
    # visible instead of inferred.
    "heartbeats": 0,
    "heartbeat_threads": 0,
    "heartbeat_respawns": 0,
    "heartbeat_floor_skips": 0,
    "heartbeat_errors": 0,
    "heartbeat_last_mono": None,
    "heartbeat_thread_alive": False,
    # THE 0.2.47 ADDITIONS: the beat's own health, not just its count.
    # `heartbeat_writes` is the one number that separates a thread that BEATS from a
    # thread that is merely alive; `heartbeat_inflight` is what a supervisor tick would
    # have to queue behind;
    # `heartbeat_stalled`/`heartbeat_jammed` name the state where a contributor is
    # holding a beat hostage; `heartbeat_fatal` counts a beat that raised something an
    # `except Exception` cannot see (a `SystemExit` out of a C extension used to end the
    # proof silently, because the only respawner was an engine step).
    "heartbeat_writes": 0,
    "heartbeat_inflight": 0,
    "heartbeat_stalled": 0,
    "heartbeat_jammed": 0,
    "heartbeat_fatal": 0,
    "heartbeat_last_duration_seconds": None,
    "heartbeat_last_beat_reason": None,
    "heartbeat_start_reason": None,
    "heartbeat_start_failures_logged": 0,
    # WEAK, on purpose: the handler owns the CPU KV arena (gigabytes on this box).
    "handler_ref": None,
}

# The registered producers and the seam's own accounting. Module-level for the
# same reason `_state` is: it is per-process state that outlives one call.
_contributors: dict = {}
_contrib_state: dict = {
    "registered": [],          # section names currently registered, in order
    "refused": [],             # (name, reason) at REGISTER time, bounded
    "errors": {},              # name -> times the producer raised or lied
    "refusals": {},            # name -> times its section was too large
    "notes": [],               # last bounded reasons, for `writer` in the file
    "bytes": 0,                # bytes the last write spent on contributions
    "sections": 0,             # sections in the last document written
    "dropped": 0,              # sections dropped to keep the file under its cap
}


def reset_state_for_tests() -> None:
    # THE HEARTBEAT THREAD IS STOPPED FIRST. A test that only reset the dict would
    # leave a live thread writing rank files into the NEXT test's temporary directory -
    # which is not merely untidy: a file appearing in a directory whose test asserts is
    # empty is a false pass waiting to happen, and this module's whole subject is files
    # appearing.
    stop_heartbeat()
    _state.update({"enabled": False, "installs": 0, "reinstalls": 0,
                   "uninstalls": 0, "skips": 0, "skip_reason": None,
                   "skip_detail": [], "nested": [],
                   "class": None, "installed": False, "dir": None, "rank": None,
                   "rank_source": None, "epoch_scope": None,
                   "interval_steps": None, "steps_since_write": 0, "steps": 0,
                   "writes": 0, "write_errors": 0, "budget_skips": 0,
                   "store_blocks_submitted": 0, "load_blocks_submitted": 0,
                   "completion_reported": 0, "last_write_mono": None,
                   "last_signature": None, "last_error": None,
                   "last_path": None, "bytes_written": 0, "fallbacks": 0,
                   "heartbeats": 0, "heartbeat_threads": 0,
                   "heartbeat_respawns": 0, "heartbeat_floor_skips": 0,
                   "heartbeat_errors": 0, "heartbeat_last_mono": None,
                   "heartbeat_thread_alive": False, "handler_ref": None,
                   "heartbeat_writes": 0, "heartbeat_inflight": 0,
                   "heartbeat_stalled": 0, "heartbeat_jammed": 0,
                   "heartbeat_fatal": 0, "heartbeat_last_duration_seconds": None,
                   "heartbeat_last_beat_reason": None,
                   "heartbeat_start_reason": None,
                   "heartbeat_start_failures_logged": 0})
    _notices.clear()
    # A BEAT THREAD STILL RUNNING IS A FILE STILL APPEARING. The supervisor is stopped
    # and joined by `stop_heartbeat()` above; the short-lived beat threads it launched
    # are drained here, or the next test's "the directory is empty" assertion is a race
    # with a beat that has not reached its write yet.
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        with _state_lock:
            if int(_state["heartbeat_inflight"]) == 0:
                break
        time.sleep(0.005)
    _reset_contributions_for_tests()


def counters() -> dict:
    """A copy for the status/snapshot surface. Nothing big, nothing secret."""
    return dict(_state)


# --- the contribution seam ---------------------------------------------------

def _reset_contributions_for_tests() -> None:
    """No contributors, no history. A leaked contributor makes the next test's
    rank file carry a section it never asked for."""
    _contributors.clear()
    _contrib_state.update({"registered": [], "refused": [], "errors": {},
                           "refusals": {}, "notes": [], "bytes": 0,
                           "sections": 0, "dropped": 0})


def register_contribution(name, produce, owner: str | None = None,
                          signal=None) -> dict:
    """Offer a section for the rank file. Contained; returns a dict; never raises.

    `produce` is a zero-argument callable returning a plain JSON-serialisable
    dict, or a dict to publish verbatim. A callable is the honest form for a
    module with live counters (it is asked at write time, so the file carries the
    numbers of the step it was written on); a dict suits a module with boot-
    constant facts.

    `signal` is an optional ZERO-ARGUMENT, CHEAP, side-effect-free callable
    returning a small hashable value that `_signature` folds into the writer's
    "did anything change?" tuple. It exists for facts that live in the contributor
    and move WITHOUT any of the handler's own numbers moving - the disk restore's
    ARMED verdict is the exact case: an arm transition with no spill activity
    changed no counter, so no rank file was rewritten, so the scheduler read a
    stale `armed: false` for the whole idle spell. A signal that raises is treated
    as `"error"` (it changes the signature exactly once) and a signal is NEVER
    asked more than once per step, so this stays a cheap test, not a second file
    reader.

    Refused - always with a reason, never silently - when the section name is not
    on `CONTRIBUTION_SECTIONS`, when the name is already held by a different
    producer, when there is nothing sensible to produce, or when the seam already
    holds `MAX_CONTRIBUTIONS` sections. Re-registering the SAME producer is
    idempotent, because `install()` runs at plugin load AND again off the
    handshake hook.
    """
    report = {"registered": False, "section": name, "reason": None}
    try:
        if signal is not None and not callable(signal):
            report["reason"] = (f"a contribution signal must be callable or None, "
                                f"got {type(signal).__name__}")
            _note_refusal(str(name), report["reason"])
            return report
        if not isinstance(name, str) or name not in CONTRIBUTION_SECTIONS:
            report["reason"] = (f"{name!r} is not an allow-listed section; the "
                                f"allow-list is {list(CONTRIBUTION_SECTIONS)}")
            _note_refusal(str(name), report["reason"])
            return report
        held = _contributors.get(name)
        if held is not None and held["produce"] is not produce:
            report["reason"] = (f"the {name!r} section already belongs to "
                                f"{held['owner']!r}; one owner per section, so a "
                                f"second module cannot overwrite it")
            _note_refusal(name, report["reason"])
            return report
        if not callable(produce) and not isinstance(produce, dict):
            report["reason"] = (f"a contribution must be a callable or a dict, "
                                f"got {type(produce).__name__}")
            _note_refusal(name, report["reason"])
            return report
        if held is None and len(_contributors) >= MAX_CONTRIBUTIONS:
            report["reason"] = (f"the seam holds {MAX_CONTRIBUTIONS} sections; "
                                f"refusing a {MAX_CONTRIBUTIONS + 1}th")
            _note_refusal(name, report["reason"])
            return report
        _contributors[name] = {"produce": produce,
                               "owner": owner or "unknown",
                               "signal": signal if callable(signal) else None}
        if name not in _contrib_state["registered"]:
            _contrib_state["registered"] = list(
                _contrib_state["registered"]) + [name]
        report["registered"] = True
        report["reason"] = ("re-registered the same producer" if held
                            else "this section is now written by worker_diag")
        return report
    except Exception as exc:  # noqa: BLE001 - a seam refusal never breaks a boot
        snapshot._error(exc, "worker diag register_contribution")
        report["reason"] = f"registering raised {type(exc).__name__}"
        return report


def unregister_contribution(name) -> dict:
    """Take a section back off. Test teardown, and a module standing down."""
    removed = _contributors.pop(str(name), None) is not None
    _contrib_state["registered"] = [row for row in _contrib_state["registered"]
                                    if row != str(name)]
    return {"unregistered": removed, "section": name,
            "reason": None if removed else "no such contribution"}


def contributions() -> dict:
    """The seam's state for a verb or a snapshot: who contributes, what failed."""
    return {"allow_list": list(CONTRIBUTION_SECTIONS),
            "registered": list(_contrib_state["registered"]),
            "owners": {name: row["owner"] for name, row in _contributors.items()},
            "max_sections": MAX_CONTRIBUTIONS,
            "max_section_bytes": MAX_CONTRIBUTION_BYTES,
            "errors": dict(_contrib_state["errors"]),
            "refusals": dict(_contrib_state["refusals"]),
            "register_refused": [dict(row) for row
                                 in _contrib_state["refused"]],
            "bytes": int(_contrib_state["bytes"]),
            "sections": int(_contrib_state["sections"]),
            "dropped": int(_contrib_state["dropped"]),
            "note": ("other worker-side modules contribute sections to THIS file "
                     "through this seam; it stays the only writer of "
                     "rank-<rank>.json, and a contributor that raises, lies or "
                     "overruns its byte cap costs its own section only.")}


def _note_refusal(name: str, reason: str) -> None:
    """Remember a register-time refusal, bounded, so a boot can be read later."""
    rows = list(_contrib_state["refused"])
    rows.append({"section": _clip(str(name), 32), "reason": _clip(reason, 160)})
    _contrib_state["refused"] = rows[-MAX_CONTRIBUTION_NOTES:]


def _count_contribution(name: str, bucket: str, reason: str) -> None:
    counts = dict(_contrib_state[bucket])
    counts[name] = int(counts.get(name, 0)) + 1
    _contrib_state[bucket] = counts
    notes = list(_contrib_state["notes"])
    notes.append(f"{name}: {_clip(reason, 120)}")
    _contrib_state["notes"] = notes[-MAX_CONTRIBUTION_NOTES:]


def _contribution_sections() -> tuple:
    """(sections, notes) for this write. Every failure costs one section only.

    A contributor that is absent is INVISIBLE: no key, no empty object, no
    `{"error": ...}` placeholder. An empty-error object for a module that was never
    relevant would make every rank file on every boot claim a facility it does not
    have, and a reader would have to guess which of the two it is looking at.
    """
    sections: dict = {}
    notes: list = []
    for name in list(_contributors):
        row = _contributors.get(name)
        if row is None:
            continue
        produce = row["produce"]
        try:
            payload = produce() if callable(produce) else produce
            if not isinstance(payload, dict):
                _count_contribution(name, "errors",
                                    f"produced {type(payload).__name__}, not a dict")
                notes.append(f"{name}: not contributed (produced "
                             f"{type(payload).__name__})")
                continue
            if not all(isinstance(key, str) for key in payload):
                _count_contribution(name, "errors",
                                    "carries a non-string key")
                notes.append(f"{name}: not contributed (non-string key)")
                continue
            raw = _dump(payload)
            if len(raw) > MAX_CONTRIBUTION_BYTES:
                _count_contribution(name, "refusals",
                                    f"{len(raw)} bytes, cap "
                                    f"{MAX_CONTRIBUTION_BYTES}")
                notes.append(f"{name}: refused at {len(raw)} bytes (cap "
                             f"{MAX_CONTRIBUTION_BYTES})")
                continue
            sections[name] = payload
        except KeyboardInterrupt:
            # THE ONE CLASS THAT ESCAPES (0.2.47). This code also runs on the STEP
            # path, in the MAIN thread, where a real SIGINT can land inside a
            # contributor, and swallowing the operator's stop command to keep one
            # section is not a trade this package makes. Signals reach only the main
            # thread, so a beat on its own thread hardly ever sees one; if it does,
            # the answer is the same.
            raise
        except BaseException as exc:  # noqa: BLE001 - a contributor is never a step fault
            # BASE, not `Exception` (0.2.47). A contributor is a GUEST in this file, and
            # the contract is that it costs its own section - nothing else. A
            # `SystemExit` or a `GeneratorExit` from under a C extension inside someone
            # else's producer used to propagate out of `build_document`, out of
            # `write_snapshot` (which catches `Exception`), out of the beat, and out of
            # the heartbeat thread - and the only thing that re-armed it was an engine
            # step. One guest raising the wrong class could then close the disk tier
            # for the whole idle spell. A guest gets no such power: it loses its
            # section, the file is still written, and the beat supervisor never
            # notices.
            _count_contribution(name, "errors",
                                f"raised {type(exc).__name__}")
            snapshot._error(exc, f"worker diag contribution ({name})")
            notes.append(f"{name}: not contributed (raised "
                         f"{type(exc).__name__})")
    return sections, notes


def section() -> dict:
    """The snapshot's view of the worker writer, compact and stable in shape.

    In the ENGINE process this is mostly "what is configured" (the writer runs in
    a worker, so `writes` is a worker-side number this process can never see);
    `worker_scope` says exactly that, so a reader does not conclude from a zero
    here that no rank has written a file. The per-rank truth is in the files, and
    the proxy host's reader is the authority on those.
    """
    directory = config.diagnostics_dir()
    return {
        "configured": directory is not None,
        "dir": str(directory) if directory is not None else None,
        "interval_steps": config.worker_write_interval_steps(),
        "min_interval_seconds": config.worker_min_interval_seconds(),
        "enabled": bool(_state["enabled"]),
        "installed": bool(_state["installed"]),
        "class": _state["class"],
        "installs": int(_state["installs"]),
        "reinstalls": int(_state["reinstalls"]),
        "uninstalls": int(_state["uninstalls"]),
        "skips": int(_state["skips"]),
        "skip_reason": _state["skip_reason"],
        "skip_detail": list(_state.get("skip_detail") or []),
        "nested_in": list(_state.get("nested") or []),
        "rank": _state["rank"],
        "rank_source": _state["rank_source"],
        "epoch_scope": _state["epoch_scope"],
        "writes": int(_state["writes"]),
        "write_errors": int(_state["write_errors"]),
        "budget_skips": int(_state["budget_skips"]),
        "bytes_written": int(_state["bytes_written"]),
        "contributions": contributions(),
        "last_path": _clip(_state["last_path"], 200),
        "last_error": _state["last_error"],
        "fallbacks": int(_state["fallbacks"]),
        # THE HEARTBEAT (0.2.45): is this worker keeping its own proof alive, and how
        # close to the gate is it running? `heartbeat_alive` is READ from the thread and
        # `last_heartbeat_age_seconds` is the number that distinguishes a worker beating
        # every minute from one whose thread died an hour ago - the only silent failure
        # a heartbeat thread can have, so it gets an age, a respawn and an error count.
        # `heartbeat_beats_in_window` states the MARGIN, so "is this tight?" is a
        # reading rather than arithmetic an operator has to do under pressure.
        "heartbeat_interval_seconds": float(HEARTBEAT_INTERVAL_SECONDS),
        "heartbeat_floor_seconds": float(HEARTBEAT_MIN_INTERVAL_SECONDS),
        "heartbeat_started": bool(_heartbeat["thread"] is not None
                                  or _state["heartbeat_thread_alive"]),
        # READ, not remembered: the flag in `_state` is only refreshed by readers that
        # ask for it, and reporting a stale False would raise a false alarm about a
        # worker whose thread is fine.
        "heartbeat_alive": _heartbeat_loop_alive(),
        "heartbeats": int(_state["heartbeats"]),
        "heartbeat_threads": int(_state["heartbeat_threads"]),
        "heartbeat_respawns": int(_state["heartbeat_respawns"]),
        "heartbeat_floor_skips": int(_state["heartbeat_floor_skips"]),
        "heartbeat_errors": int(_state["heartbeat_errors"]),
        "last_heartbeat_age_seconds": (
            round(time.monotonic() - float(_state["heartbeat_last_mono"]), 3)
            if isinstance(_state["heartbeat_last_mono"], (int, float)) else None),
        "heartbeat_window_seconds": _heartbeat_window_seconds(),
        "heartbeat_beats_in_window": int(_heartbeat_window_seconds() // max(
            0.001, float(HEARTBEAT_INTERVAL_SECONDS))),
        "schema_version": SCHEMA_VERSION,
        "file_pattern": "rank-PP<pp>_TP<tp>.json",
        "worker_scope": ("this process is not a worker: these are the configured "
                         "values. The per-rank counters live in the files the "
                         "worker writers produce."),
    }


def _clip(value, chars: int = MAX_STRING_CHARS):
    if isinstance(value, str) and len(value) > chars:
        return value[:chars] + "..."
    return value


# --- resolving the fork's names without importing it ------------------------

def worker_class():
    """`SimpleCPUOffloadWorker`, if this process already imported it."""
    module = sys.modules.get(TARGET_MODULE)
    if module is None:
        return None
    return getattr(module, TARGET_CLASS, None)


def _refuse(reason: str) -> None:
    if _state["skip_reason"] == reason:
        return
    _state["skips"] += 1
    _state["skip_reason"] = reason


# --- rank and epoch ---------------------------------------------------------

def _valid_rank(name) -> bool:
    """`PP<digits>_TP<digits>`, exactly what the reader's `_is_worker_id` accepts."""
    if not isinstance(name, str) or len(name) > 32:
        return False
    parts = name.split("_")
    if len(parts) != 2:
        return False
    pp, tp = parts
    return (pp.startswith("PP") and tp.startswith("TP")
            and pp[2:].isdigit() and tp[2:].isdigit()
            and len(pp) > 2 and len(tp) > 2)


def rank_id():
    """This rank's id, or None. The fork's own derivation, or the operator's.

    1. `VLLM_E2_TIER_DIAG_RANK`, if it is a legal id (an operator who knows the
       layout can pin it, which is also the escape hatch when the process has no
       distributed state to ask);
    2. otherwise `PP{get_pp_group().rank_in_group}_TP{get_tp_group().rank_in_group}`
       from the ALREADY-IMPORTED `vllm.distributed.parallel_state` - the same two
       expressions `gpu_worker.py:885-886` uses to key the handshake. No import
       of our own, and a process without distributed state simply gets None.
    """
    forced = config.diagnostics_rank()
    if forced is not None:
        if _valid_rank(forced):
            return forced, "env"
        return None, f"{config.ENV_DIAG_RANK}={forced!r} is not PP<n>_TP<m>"
    module = sys.modules.get("vllm.distributed.parallel_state")
    if module is None:
        return None, "vllm.distributed.parallel_state is not imported here"
    try:
        pp_group = module.get_pp_group()
        tp_group = module.get_tp_group()
        pp = getattr(pp_group, "rank_in_group", None)
        tp = getattr(tp_group, "rank_in_group", None)
    except Exception as exc:  # noqa: BLE001 - no distributed state is a fact
        return None, f"parallel_state refused to answer: {type(exc).__name__}"
    if not isinstance(pp, int) or isinstance(pp, bool) \
            or not isinstance(tp, int) or isinstance(tp, bool):
        return None, "rank_in_group is not an int on both groups"
    candidate = f"PP{pp}_TP{tp}"
    if not _valid_rank(candidate):
        return None, f"derived {candidate!r} is not a legal writer id"
    return candidate, "parallel_state"


# --- the document -----------------------------------------------------------

def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _arena_bytes(handler) -> tuple[int | None, str]:
    """The arena's real size, and which source answered."""
    caches = getattr(handler, "cpu_kv_caches", None)
    if isinstance(caches, dict) and caches:
        total = 0
        for tensor in caches.values():
            nbytes = getattr(tensor, "nbytes", None)
            size = _int(nbytes)
            if size is None:
                return None, "a cpu_kv_caches entry has no usable nbytes"
            total += size
        return total, "cpu_kv_caches.nbytes"
    configured = _int(getattr(handler, "cpu_capacity_bytes", None))
    if configured is not None:
        return configured, "cpu_capacity_bytes (configured, not yet allocated)"
    return None, "no arena to measure"


def _block_bytes_by_group(handler):
    """Per KV group, the group's WHOLE-BLOCK bytes, and its layer count separately.

    `spec.page_size_bytes` ALONE, with the layer count published beside it. The
    multiplication this function used to do (`len(layer_names) * page_size_bytes`) is
    wrong on the worker side of a merged config and was caught by the field, not by a
    test: `MambaSpec.merge()` returns a spec whose `page_size_bytes` is the SUM over the
    merged layers (`vllm/v1/kv_cache_interface.py:1151-1160`, reached from
    `merge_group_spec`'s `typical_spec.merge(duplicate_spec)`), and
    `MambaSpec.page_size_bytes` already multiplies by its own `num_speces`
    (`:1135-1143`). So the group total was counted `len(layer_names)` times over: on
    the live PP0 that made group 0 read 80,281,600 B against a measured spill file of
    11,468,800 B, and because group 5 is an attention spec whose merge takes the MAX
    (`FullAttentionSpec.merge`, `:926-940`) the error differed per group AND per PP
    rank - which is how two ranks came to publish different "per-group" figures.

    Returns `(sizes, layers, note)`. `None` for both lists rather than a partial one
    when any term is missing: the reader notes an absent field instead of folding a
    wrong total into a group sum.
    """
    groups = getattr(getattr(handler, "kv_cache_config", None), "kv_cache_groups",
                     None)
    if not groups:
        return None, None, None
    sizes = []
    layers_seen = []
    for group in list(groups)[:MAX_GROUPS]:
        spec = getattr(group, "kv_cache_spec", None)
        layers = _int(_len_or_none(getattr(group, "layer_names", None)))
        page = _int(getattr(spec, "page_size_bytes", None))
        if layers is None or page is None:
            return None, None, "a group is missing layer_names or page_size_bytes"
        sizes.append(page)
        layers_seen.append(layers)
    if len(list(groups)) > MAX_GROUPS:
        return None, None, f"more than {MAX_GROUPS} groups"
    return sizes, layers_seen, None


def _len_or_none(container):
    if container is None:
        return None
    try:
        return len(container)
    except TypeError:
        return None


def _bytes_per_block(handler, arena_bytes) -> int | None:
    blocks = _int(getattr(handler, "num_cpu_blocks", None))
    if blocks and arena_bytes:
        return arena_bytes // blocks
    return None


def _signature(handler, arena_bytes) -> tuple:
    """The cheap "did anything change?" test. Small ints only, never a serialise.

    The contribution SIGNALS ride along (0.2.28): a fact a contributor owns and
    that moves none of the numbers here (the disk tier's armed verdict is the one
    that earned this) would otherwise wait for the cadence - or, on an idle box,
    for a step that never comes. `unregister_contribution` pops the row, so a
    removed contributor stops contributing to the signature too.
    """
    signals = []
    for name in sorted(_contributors):
        signal = (_contributors.get(name) or {}).get("signal")
        if not callable(signal):
            continue
        try:
            value = signal()
        except Exception as exc:  # noqa: BLE001 - a signal is never worth a fault
            value = f"error:{type(exc).__name__}"
        signals.append((name, value if isinstance(
            value, (bool, int, str, type(None))) else str(value)[:32]))
    return (arena_bytes,
            _int(getattr(handler, "num_cpu_blocks", None)),
            _len_or_none(getattr(handler, "_pending_store_event_indices", None)),
            _len_or_none(getattr(handler, "_pending_load_event_indices", None)),
            _int(getattr(handler, "_store_hwm", None)),
            _int(getattr(handler, "_load_hwm", None)),
            int(_state["store_blocks_submitted"]),
            int(_state["load_blocks_submitted"]),
            tuple(signals))


def build_document(handler, rank, rank_source) -> dict:
    """The file's content. Every number is a real reading or the key is absent.

    The contributed sections are appended LAST, as siblings of `version`/`arena`
    (the reader's own `gpu_census` precedent), and a contributor that produced
    nothing contributes NO key - see `_contribution_sections`.
    """
    arena_bytes, bytes_source = _arena_bytes(handler)
    block_bytes, block_layers, block_bytes_note = _block_bytes_by_group(handler)
    metadata = getattr(handler, "_connector_metadata", None)
    per_block = _bytes_per_block(handler, arena_bytes)
    in_flight_blocks = None
    if metadata is not None and per_block is not None:
        in_flight_blocks = _len_or_none(getattr(metadata, "store_cpu_blocks", None))
    bytes_used = 0
    if in_flight_blocks:
        bytes_used = in_flight_blocks * per_block
    epoch, epoch_scope = config.diagnostics_epoch()
    version_scope = (f"the worker process that wrote this file (rank {rank}, "
                     f"role {ROLE}); a worker answers no HTTP route, so this is "
                     f"the only place its loaded version can appear")
    arena: dict = {"bytes_total": arena_bytes if arena_bytes is not None else 0,
                   "bytes_used": bytes_used,
                   "num_blocks": _int(getattr(handler, "num_cpu_blocks", None))
                   or 0}
    if block_bytes is not None:
        arena["block_bytes_by_group"] = block_bytes
        # The layer count is still the fact a capacity reader wants - it just must not
        # be folded into the byte figure by the publisher that does not know whether the
        # spec behind the byte figure already counted it.
        arena["layer_count_by_group"] = block_layers
    if per_block is not None:
        # THE WIDTH EVERY SPILL FILE ACTUALLY HAS, measured off this rank's own region
        # tensors. `block_bytes_by_group` is a per-GROUP statement read off a spec; this
        # is the per-RANK statement read off the memory the tier copies, which is what
        # `disk_spill.measured_slot_bytes` writes and `disk_read` reads back. Publishing
        # both, named differently, is the point: when they disagree, the disagreement is
        # the bug report (it is how the double-count above was found).
        arena["bytes_per_block"] = int(per_block)
    counters = {"stored": int(_state["store_blocks_submitted"]),
                "restored_hits": int(_state["load_blocks_submitted"]),
                "store_events_pending": _len_or_none(
                    getattr(handler, "_pending_store_event_indices", None)) or 0,
                "load_events_pending": _len_or_none(
                    getattr(handler, "_pending_load_event_indices", None)) or 0}
    sections, contribution_notes = _contribution_sections()
    document = {
        "version": SCHEMA_VERSION,
        # WHICH CODE WROTE THIS FILE, read live in this process (see
        # `loaded_version.py`). Beside `version`, which is the file's SCHEMA and
        # keeps meaning that: a reader must never have to guess which of the two a
        # `version` key is.
        **loaded_version.fields(version_scope),
        "epoch": epoch,
        "ts": round(time.time(), 6),
        "role": ROLE,
        "rank": rank,
        "arena": arena,
        "counters": counters,
        "writer": {
            "plugin": WRITER_TAG,
            "writes": int(_state["writes"]) + 1,
            "steps_since_write": int(_state["steps_since_write"]),
            "interval_steps": _state["interval_steps"],
            "min_interval_seconds": _state["min_interval_seconds"],
            "rank_source": rank_source,
            "epoch_scope": epoch_scope,
            "bytes_total_source": bytes_source,
            "bytes_used_meaning": (
                "a PROVEN FLOOR, not the CPU pool's occupancy: bytes of the "
                f"blocks bound for a store right now "
                f"({in_flight_blocks or 0} blocks x {per_block} B/block). The "
                "pool's real occupancy is scheduler-side and comes from the "
                "plugin's /snapshot."),
            "block_bytes_by_group_meaning": (
                "per KV group: spec.page_size_bytes ALONE - on a merged worker "
                "config that is already the group's whole-block total, because "
                "MambaSpec.merge sums it over the merged layers and its "
                "page_size_bytes multiplies by num_speces, which the merge folded in. "
                "The layer count is published separately as layer_count_by_group and "
                "must NOT be multiplied back in: doing so is what published "
                "80,281,600 B for a group whose spill files are 11,468,800 B. "
                "Cross-check against arena.bytes_per_block, this rank's measured slot "
                "width."),
            "bytes_per_block_meaning": (
                f"this rank's MEASURED slot width in bytes ({per_block} B): the sum of "
                "one row of every region tensor, i.e. exactly the size of one spill "
                "file and exactly the size of one restore. It is a per-RANK number, so "
                "it legitimately differs between PP stages (11,468,800 on PP0, "
                "12,902,400 on PP1 for this model); a per-GROUP figure that differs "
                "between ranks is a bug, not a rank fact."),
            "counter_meaning": ("stored/restored_hits are blocks HANDED TO the "
                                "copy streams since boot (counted at "
                                "bind_connector_metadata), not confirmed "
                                "completions"),
            "completion_reported": int(_state["completion_reported"]),
            "store_hwm": _int(getattr(handler, "_store_hwm", None)),
            "load_hwm": _int(getattr(handler, "_load_hwm", None)),
            "completed_store_events": _len_or_none(
                getattr(handler, "_completed_store_events", None)),
            "notes": [note for note in (block_bytes_note,) if note],
            "contributions": {"registered": list(_contrib_state["registered"]),
                              "written": sorted(sections),
                              "notes": list(contribution_notes)},
            # THE WRITER'S OWN PULSE, IN THE FILE THE SCHEDULER READS (0.2.47). A
            # worker has no HTTP route: `/plugins/kvtier/snapshot` is answered by the
            # API server and reports this module's heartbeat as zeros for a process it
            # has never seen. So the per-process truth about the mechanism that keeps
            # THIS file fresh travels inside the file, where the scheduler, the proxy
            # host's reader and an operator with `cat` can all reach it. Same key set in
            # every state; `state` is one of `worker_diag.HEARTBEAT_STATES`.
            "heartbeat": heartbeat_state(),
        },
        "error": _state["last_error"],
    }
    document.update(sections)
    return document


def _clip_document(document: dict) -> dict:
    """Shrink to under MAX_FILE_BYTES without breaking a required field.

    The order is a triage, not a scramble. A contributed section goes first: it is
    the newest guest in this file and the least load-bearing for the reader, which
    reads a fixed set of keys and ignores an absent one. Then the optional prose
    (`writer`), then the optional group list. The keys the reader REQUIRES -
    `version`/`epoch`/`ts`/`role`/`rank`/`arena`/`counters` - are never touched, so
    a trimmed file is a smaller file, never a `malformed` one.
    """
    raw = _dump(document)
    if len(raw) <= MAX_FILE_BYTES:
        return document
    trimmed = {key: value for key, value in document.items()
               if key in KEEP_KEYS}
    if len(_dump(trimmed)) > MAX_FILE_BYTES:
        trimmed["writer"] = {"plugin": WRITER_TAG, "trimmed": True,
                             "trimmed_reason": f"document exceeded "
                                               f"{MAX_FILE_BYTES} bytes"}
    if len(_dump(trimmed)) > MAX_FILE_BYTES:
        trimmed["arena"] = dict(trimmed.get("arena") or {})
        trimmed["arena"].pop("block_bytes_by_group", None)
    if len(trimmed) != len(document):
        _contrib_state["dropped"] = int(_contrib_state["dropped"]) + 1
    return trimmed


def _dump(document: dict) -> bytes:
    return json.dumps(document, separators=(",", ":"), default=str).encode("utf-8")


def write_snapshot(handler, rank, rank_source) -> dict:
    """Atomically replace this rank's file. Contained; returns a small report."""
    directory = config.diagnostics_dir()
    if directory is None:
        return {"written": False, "reason": f"{config.ENV_DIAG_DIR} unset"}
    report = {"written": False, "path": None, "bytes": 0, "reason": None}
    try:
        document = _clip_document(build_document(handler, rank, rank_source))
        raw = _dump(document)
        if len(raw) > MAX_FILE_BYTES:
            report["reason"] = (f"even the trimmed document is {len(raw)} bytes "
                                f"(cap {MAX_FILE_BYTES}); refusing to write")
            _state["write_errors"] += 1
            return report
        filename = f"rank-{rank}.json"
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(str(directory),
                           f".{filename}.{os.getpid()}.{_state['writes']}.tmp")
        final = os.path.join(str(directory), filename)
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, final)
        _state["writes"] += 1
        _state["bytes_written"] = len(raw)
        _state["last_path"] = final
        _state["last_write_mono"] = time.monotonic()
        # What the contributions cost this file, measured on the document that was
        # actually written: a section refused by the byte ceiling or dropped by the
        # triage must not be reported as shipped.
        shipped = {name: value for name, value in document.items()
                   if name not in KEEP_KEYS}
        _contrib_state["sections"] = len(shipped)
        _contrib_state["bytes"] = sum(len(_dump({name: value}))
                                      for name, value in shipped.items())
        # NOT `last_signature`: only `_maybe_write` may settle it, and only with
        # the signature it computed WITH the arena size. Setting a differently
        # shaped one here made every following step look "changed" and defeated
        # the whole change test.
        report.update({"written": True, "path": final, "bytes": len(raw)})
        return report
    except Exception as exc:  # noqa: BLE001 - a diagnostic file must never be a fault
        _state["write_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}: {exc}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "worker diag write")
        report["reason"] = _state["last_error"]
        return report


def _maybe_write(handler) -> dict:
    """The budgeted trigger. Called from `get_finished`, i.e. once per step."""
    _state["steps"] += 1
    _state["steps_since_write"] += 1
    directory = config.diagnostics_dir()
    if directory is None:
        return {"written": False, "reason": f"{config.ENV_DIAG_DIR} is not set"}
    rank, rank_source = rank_id()
    if rank is None:
        _state["rank_source"] = rank_source
        return {"written": False, "reason": f"no usable rank: {rank_source}"}
    interval = config.worker_write_interval_steps()
    arena_bytes, _source = _arena_bytes(handler)
    signature = _signature(handler, arena_bytes)
    due_by_cadence = _state["steps_since_write"] >= interval
    due_by_change = signature != _state["last_signature"]
    if not (due_by_cadence or due_by_change):
        return {"written": False, "reason": "not due"}
    floor = config.worker_min_interval_seconds()
    last = _state["last_write_mono"]
    if last is not None and (time.monotonic() - last) < floor:
        _state["budget_skips"] += 1
        return {"written": False,
                "reason": f"rate floor {floor}s not elapsed since the last write"}
    report = write_snapshot(handler, rank, rank_source)
    if report.get("written"):
        _state["steps_since_write"] = 0
        _state["rank"] = rank
        _state["rank_source"] = rank_source
        _state["last_signature"] = signature
        _epoch, _scope = config.diagnostics_epoch()
        _state["epoch_scope"] = _scope
    elif _state["rank"] is None:
        _state["rank"] = rank
        _state["rank_source"] = rank_source
    return report


# --- THE HEARTBEAT: an idle worker must LOOK alive, not merely be alive ------
#
# WHY THIS EXISTS AT ALL. `disk_restore` may only offer a disk restore when a WORKER
# process has attested that its restore layer is fitted (they are separate processes,
# and offering without the proof means `start_load_kv` DMAs CPU slots nobody filled -
# the request is served from STALE KV, the one silent failure this design has). The
# attestation is the rank file, and the reader REFUSES a file older than
# `disk_restore.RANK_FILE_FRESH_SECONDS` (900 s) - because a file a dead worker wrote
# must not vouch for it forever. Both halves are right. The WIRING was not:
# `_maybe_write` runs from the wrapped `get_finished`, so the only thing that rewrote
# the file was an ENGINE STEP. A worker with no request writes nothing, its file ages
# past the window, and the scheduler stops believing a worker that is perfectly alive -
# closing the disk path on a quiet box and reopening it only when a burst of steps
# arrives, which is the worst possible moment because the first request after an idle
# spell is the one that wanted the disk. Nothing expired and nothing failed; a heartbeat
# was missing.
#
# WHAT THIS IS NOT. It is not a weaker gate: the window stays 900 s, so a worker that
# has genuinely gone stops being believed on time - that is the gate's whole value, and
# this file does not touch it. A PROOF IS NOT DATA: nothing here keeps a cached byte
# alive, it keeps a live worker legible.
HEARTBEAT_INTERVAL_DEFAULT = 60.0
HEARTBEAT_INTERVAL_SECONDS = HEARTBEAT_INTERVAL_DEFAULT
# THE WRITE FLOOR THE HEARTBEAT HONOURS. `config.worker_min_interval_seconds()` is the
# engine-driven writer's floor and exists because every write is an fsync'd atomic
# replace in a directory the SCHEDULER also reads. A thread on its own timer would
# otherwise be a second writer with no budget at all, so the heartbeat takes its own
# floor - a separate knob rather than a borrowed one, because "how often may a
# background thread touch this directory" and "how often may a step" are different
# questions an operator may want to answer differently.
HEARTBEAT_MIN_INTERVAL_SECONDS = 5.0
HEARTBEAT_THREAD_NAME = "kvtier-worker-diag-heartbeat"
# THE MARGIN THIS DESIGN IS ALLOWED TO LOSE. 60 s inside a 900 s window is fifteen
# beats, so four consecutive dropped ticks (a long GIL hold, a slow device, a
# stop-the-world import) are still safe. Pinned by a test against the UNPATCHED
# constants, because an operator raising the interval to 500 s would technically stay
# inside 900 s and still lose the disk path on the first missed beat.
HEARTBEAT_MIN_BEATS_IN_WINDOW = 4
# HOW LONG ONE BEAT MAY TAKE BEFORE IT IS CALLED STALLED (0.2.47). A beat is not a
# `time.sleep`: it asks the SIX contribution producers for their sections and writes an
# fsync'd atomic replace, and every one of those producers is a guest from another
# module - one of them may be holding a lock that a spill, reader or unlink thread owns
# across a long NVMe pass. Until 0.2.47 the beat ran INLINE in the loop, so a producer
# that BLOCKS does not slow the proof down, it STOPS IT - while `heartbeat_alive` keeps
# answering True, because the thread is alive and parked inside the beat. That is the
# one heartbeat state this design cannot afford: a rank file that ages past the window
# while every readable field says "alive" is indistinguishable from a dead worker, and
# that is precisely the outage 0.2.45 was written to end. So since 0.2.47 the loop
# ISSUES each beat into a short-lived thread and never waits inside the tick: one wedged
# beat costs one tick, the next tick issues another, and the beats that never finish are
# counted (`heartbeat_stalled`, `heartbeat_jammed`) and named in the rank file itself.
HEARTBEAT_BEAT_BUDGET_SECONDS = 20.0
# HOW MANY UNFINISHED BEATS MAY EXIST AT ONCE. Past this the contributor is not slow,
# it is wedged - and launching another thread that will never return is how a
# diagnostic turns a stalled proof into a thread leak. So the launches stop, the counter
# keeps counting, and the rank file says `jammed_beats`.
HEARTBEAT_MAX_INFLIGHT_BEATS = 3
HEARTBEAT_BEAT_THREAD_NAME = "kvtier-worker-diag-beat"

# THE FOUR ANSWERS THE RANK FILE GIVES ABOUT ITS OWN PROCESS, and the key set it gives
# them in. Every state answers with EVERY key (`_heartbeat_state_shell` builds the shape
# and the real reading overwrites it), because a reader that has to guess which fields a
# state omits is a reader that reads nothing. `beating` and `alive_not_beating` are the
# pair that matters most: a file that says `beating` and is nevertheless stale is a
# process that STOPPED writing while its thread was fine, and a file that says
# `alive_not_beating` says so before it goes stale at all.
HEARTBEAT_STATES = ("beating", "alive_not_beating", "started_not_alive",
                    "never_started", "not_applicable")
HEARTBEAT_FILE_KEYS = ("state", "state_reason", "alive", "started", "beats",
                       "beat_writes", "inflight_beats", "interval_seconds",
                       "window_seconds", "beats_in_window",
                       "last_beat_age_seconds", "last_beat_duration_seconds",
                       "errors", "fatal_beats", "stalled_beats", "jammed_beats",
                       "floor_skips", "respawns", "writer_pid", "last_error")

_heartbeat: dict = {"thread": None, "stop": None, "closed": False}
_heartbeat_lock = threading.Lock()
_handler_lock = threading.Lock()
# THE ONE LOCK THE HEARTBEAT MADE NECESSARY. Every counter update the heartbeat thread
# makes goes through it. The step-driven path deliberately does NOT take it: one thread,
# one writer, and a lock per step on the engine's hot path costs real time. The
# heartbeat writes at 60 s and the counters it touches are its own; `counters()` copies
# the dict, which CPython makes atomic for us.
_state_lock = threading.Lock()


def note_handler(handler) -> dict:
    """Remember the worker's handler WEAKLY, so a heartbeat can read it, not own it.

    THE WEAK REFERENCE IS NOT POLITE, IT IS LOAD-BEARING. The handler owns the CPU KV
    arena - gigabytes on this box - and a diagnostic writer that kept it alive by
    holding a reference would make the engine's own memory un-collectable for the life
    of the plugin. So the heartbeat reads the handler when it happens to still be there
    and writes the same document with nothing measured when it is not, because the FACT
    the file carries - "this process is alive" - does not need the handler at all.
    """
    out = {"stored": False, "reason": None}
    try:
        if handler is None:
            out["reason"] = "no handler was offered"
            return out
        with _handler_lock:
            reference = _state["handler_ref"]
            if isinstance(reference, weakref.ref) and reference() is handler:
                out["stored"] = True
                return out
            try:
                _state["handler_ref"] = weakref.ref(handler)
            except TypeError:
                # An object that refuses weakreferences (a slotted class, a C type) is
                # not worth pinning in memory for. The heartbeat writes without it.
                _state["handler_ref"] = None
                out["reason"] = ("this handler cannot be referenced weakly, so the "
                                 "heartbeat writes without reading it")
                return out
        out["stored"] = True
        return out
    except Exception as exc:  # noqa: BLE001 - remembering a handler is never a fault
        out["reason"] = f"{type(exc).__name__}"
        return out


def _resolve_handler():
    """The handler if it is still alive, else None. Never extends its life."""
    with _handler_lock:
        reference = _state["handler_ref"]
    if not isinstance(reference, weakref.ref):
        return None
    try:
        return reference()
    except Exception:  # noqa: BLE001 - a dead reference is simply no handler
        return None


def _heartbeat_window_seconds() -> float:
    """The freshness window this heartbeat must stay inside, read from its OWNER.

    `disk_restore` owns the gate, so the interval is checked against the gate's own
    number rather than a copy of it. Imported lazily because `disk_restore` imports this
    module's contribution seam at load time: a top-level import would make the package
    depend on the order modules happen to be imported in.
    """
    try:
        from . import disk_restore
        return float(disk_restore.RANK_FILE_FRESH_SECONDS)
    except Exception:  # noqa: BLE001 - a missing reader is no reason to stop proving
        return 0.0


def heartbeat_once() -> dict:
    """One beat: rewrite the rank file if the floor allows it.

    IT GOES THROUGH THE SAME WRITE PATH A STEP USES, not a second writer. That is what
    keeps the file's SHAPE identical - one document builder, one atomic replace, one set
    of contributions - so a reader can never tell a heartbeat-written file from a
    step-written one, and a mixed-version fleet's reader sees the keys it expects. A
    separate "heartbeat.json" would have been the obvious implementation and a second
    source of truth about who is alive.
    """
    out = {"written": False, "reason": None, "handler": False}
    directory = config.diagnostics_dir()
    if directory is None:
        out["reason"] = f"{config.ENV_DIAG_DIR} is not set"
        return out
    rank, rank_source = rank_id()
    if rank is None:
        out["reason"] = f"no usable rank: {rank_source}"
        return out
    last = _state["heartbeat_last_mono"]
    floor = max(0.0, float(HEARTBEAT_MIN_INTERVAL_SECONDS))
    if isinstance(last, (int, float)) and (time.monotonic() - float(last)) < floor:
        with _state_lock:
            _state["heartbeat_floor_skips"] = int(_state["heartbeat_floor_skips"]) + 1
        out["reason"] = f"floor {floor}s not elapsed since the last beat"
        return out
    handler = _resolve_handler()
    out["handler"] = handler is not None
    report = write_snapshot(handler, rank, rank_source)
    with _state_lock:
        _state["heartbeat_last_mono"] = time.monotonic()
        if report.get("written"):
            _state["heartbeats"] = int(_state["heartbeats"]) + 1
            # THE NUMBER THE RANK FILE REPORTS AS `beating`: a tick that ACTUALLY WROTE.
            # `heartbeats` alone cannot say that (a refused tick never reaches it),
            # and a
            # beat that ticked but was refused (a floor, a rank that vanished, a write
            # error) must never be misreported to the operator as a live proof.
            _state["heartbeat_writes"] = int(_state["heartbeat_writes"]) + 1
            _state["heartbeat_last_beat_reason"] = None
            out["written"] = True
        else:
            out["reason"] = str(report.get("reason"))[:MAX_STRING_CHARS]
            _state["heartbeat_last_beat_reason"] = out["reason"]
    return out


def _contained_heartbeat() -> dict:
    """One heartbeat that CANNOT kill the thread.

    Copied from `disk_gpu_write._contained`, for the reason that module learned it: a
    contributor's producer, a config read or a filesystem can raise, and a heartbeat
    loop that dies on it stops proving silently - which is indistinguishable from the
    defect this thread exists to fix, just quieter. Every failure is a counter.
    """
    try:
        return heartbeat_once()
    except Exception as exc:  # noqa: BLE001 - the loop outlives any single beat
        with _state_lock:
            _state["heartbeat_errors"] = int(_state["heartbeat_errors"]) + 1
            _state["last_error"] = (
                f"heartbeat: {type(exc).__name__}")[:MAX_STRING_CHARS]
            _state["heartbeat_last_beat_reason"] = (
                f"beat raised {type(exc).__name__}")[:MAX_STRING_CHARS]
        snapshot._error(exc, "worker diag heartbeat")
        return {"written": False, "reason": f"{type(exc).__name__}"}


def _run_beat_guarded() -> None:
    """ONE BEAT, on its own short-lived thread, that cannot end the proof (0.2.47).

    TWO JOBS, and both are the reason the beat no longer runs inline in the loop.

    * `except BaseException`, because `Exception` is not enough. A `SystemExit` or a
      `GeneratorExit` from under a C extension used to end the beat thread - and the
      ONLY thing that re-armed it was `heartbeat_status(force=True)` inside the wrapped
      `get_finished`, which is an ENGINE STEP. An idle worker takes no steps, so a beat
      lost during an idle spell stayed lost for the whole spell and the disk path stayed
      shut until traffic arrived to fix it. That is the 0.2.44 lesson in a third
      outfit: a recovery path driven by the very thing whose absence is the fault. A
      `SystemExit` swallowed in a daemon diagnostic thread costs nothing (the
      interpreter exits when the non-daemon threads finish) and buys the proof a
      supervisor that cannot be silenced by a beat.
    * a DURATION, measured and published. A beat parked on someone's lock cannot be
      interrupted from here, but it can be MEASURED against
      `HEARTBEAT_BEAT_BUDGET_SECONDS`, counted as `heartbeat_stalled`, and named in the
      rank file - and because it holds its own thread, it no longer stands between the
      next tick and its write.
    """
    began = time.monotonic()
    try:
        _contained_heartbeat()
    except KeyboardInterrupt:  # pragma: no cover - not that thread
        raise
    except BaseException as exc:  # noqa: BLE001 - nothing a beat does ends the proof
        with _state_lock:
            _state["heartbeat_fatal"] = int(_state["heartbeat_fatal"]) + 1
            _state["heartbeat_last_beat_reason"] = (
                f"beat died on {type(exc).__name__}")[:MAX_STRING_CHARS]
        snapshot._error(exc, "worker diag beat")
    finally:
        duration = time.monotonic() - began
        with _state_lock:
            _state["heartbeat_inflight"] = max(
                0, int(_state["heartbeat_inflight"]) - 1)
            _state["heartbeat_last_duration_seconds"] = round(duration, 3)
            if duration > float(HEARTBEAT_BEAT_BUDGET_SECONDS):
                _state["heartbeat_stalled"] = int(_state["heartbeat_stalled"]) + 1


def _issue_beat() -> dict:
    """Hand one beat to its own thread. Never waits for it. Never raises.

    THE TICK DOES NOT WRITE ANY MORE, IT DELEGATES - the whole of the 0.2.47 change, and
    the reason a wedged contributor can no longer stop the proof: the supervisor's clock
    is its own `stop.wait()`, so a beat parked on someone else's lock holds up nothing
    except itself. `HEARTBEAT_MAX_INFLIGHT_BEATS` is what keeps that from becoming a
    thread leak: past the bound the supervisor says `jammed`, stops launching, and the
    rank file carries the word instead of a silence.
    """
    out = {"launched": False, "reason": None}
    with _heartbeat_lock:
        if _heartbeat["closed"]:
            out["reason"] = "the heartbeat is stopped in this process"
            return out
    with _state_lock:
        inflight = int(_state["heartbeat_inflight"])
        if inflight >= int(HEARTBEAT_MAX_INFLIGHT_BEATS):
            _state["heartbeat_jammed"] = int(_state["heartbeat_jammed"]) + 1
            out["reason"] = (f"{inflight} beats are already unfinished; refusing "
                             "another")
            return out
        _state["heartbeat_inflight"] = inflight + 1
    try:
        threading.Thread(target=_run_beat_guarded,
                         name=HEARTBEAT_BEAT_THREAD_NAME, daemon=True).start()
    except Exception as exc:  # noqa: BLE001 - no thread is a refusal, not a fault
        with _state_lock:
            _state["heartbeat_inflight"] = max(
                0, int(_state["heartbeat_inflight"]) - 1)
            _state["heartbeat_errors"] = int(_state["heartbeat_errors"]) + 1
            _state["heartbeat_last_beat_reason"] = (
                f"beat thread could not start: {type(exc).__name__}"
            )[:MAX_STRING_CHARS]
        out["reason"] = f"the beat thread could not start: {type(exc).__name__}"
        return out
    out["launched"] = True
    return out


def _heartbeat_loop(stop) -> None:
    """The supervisor's tick: wait, then ISSUE one beat, until asked to stop.

    `stop.wait()` FIRST, never after: a loop that writes before its first wait has
    written a file during `install()`, which is the step-driven path's job and would
    double the writes at boot. The wait is also what makes an idle worker the ONLY
    customer - a busy worker's steps already rewrite the file, and a beat that fires
    anyway costs an fsync to say what the step already said. The write floor is what
    makes that cheap rather than merely rare.

    Since 0.2.47 the tick delegates (`_issue_beat`) and is wrapped against
    `BaseException`. The loop is the only clock an idle worker has, so the rule for it
    is absolute: NOTHING a beat can do may end it, and nothing a beat can do may delay
    it. Both halves are tested, because both were false here.
    """
    while not stop.wait(max(0.001, float(HEARTBEAT_INTERVAL_SECONDS))):
        try:
            _issue_beat()
        except KeyboardInterrupt:  # pragma: no cover - not that thread
            raise
        except BaseException as exc:  # noqa: BLE001 - the clock outlives everything
            with _state_lock:
                _state["heartbeat_fatal"] = int(_state["heartbeat_fatal"]) + 1
            snapshot._error(exc, "worker diag heartbeat tick")


def _heartbeat_loop_alive() -> bool:
    """Is the heartbeat thread actually running? Read, do not assume."""
    with _heartbeat_lock:
        thread = _heartbeat["thread"]
    alive = bool(thread is not None and thread.is_alive())
    with _state_lock:
        _state["heartbeat_thread_alive"] = alive
    return alive


def start_heartbeat() -> dict:
    """Start the heartbeat thread, once, lazily. Never at import.

    STARTED BY `install()`, AND THAT IS THE WHOLE POINT. `disk_gpu_write` shipped this
    bug in 0.2.43: its thread's only starter sat below the code path that needed what
    the thread produces, so the first write was unreachable. The equivalent shape here
    is a heartbeat started by an engine step - an IDLE worker is precisely a worker
    whose starter never runs, which is the defect this block exists to fix. So
    `install()` asks, and a worker that never takes a step still proves it is alive.

    Starting allocates nothing: the thread waits its interval before its first write,
    and with no diagnostics directory there is nothing for it to write, so the disabled
    tier starts no thread at all.

    A REFUSAL IS RECORDED, NOT RETURNED AND FORGOTTEN (0.2.47). Until now the reason
    lived in a dict the caller printed or dropped, which is how "the heartbeat never
    started in this worker" and "the heartbeat is running perfectly" came to look
    identical from outside. `heartbeat_start_reason` is what the rank file's
    `never_started` state names, and `install()` logs it once.
    """
    report = {"started": False, "alive": False, "reason": None}
    directory = config.diagnostics_dir()
    if directory is None:
        report["reason"] = (f"{config.ENV_DIAG_DIR} is not set: an inert tier starts "
                            "no thread and writes no file")
        with _state_lock:
            _state["heartbeat_start_reason"] = report["reason"]
        return report
    with _heartbeat_lock:
        thread = _heartbeat["thread"]
        if thread is not None and thread.is_alive():
            report.update({"started": True, "alive": True})
            return report
        try:
            stop = threading.Event()
            new = threading.Thread(target=_heartbeat_loop, args=(stop,),
                                   name=HEARTBEAT_THREAD_NAME, daemon=True)
            new.start()
        except Exception as exc:  # noqa: BLE001 - no thread is a refusal, not a fault
            report["reason"] = (f"the heartbeat thread could not start: "
                                f"{type(exc).__name__}")
            with _state_lock:
                _state["heartbeat_errors"] = int(_state["heartbeat_errors"]) + 1
                _state["heartbeat_start_reason"] = report["reason"]
            return report
        # THE GATE THE SUPERVISOR LAUNCHES ITS BEATS THROUGH. `stop_heartbeat()` closes
        # it so a beat cannot land after teardown; a start - including a respawn of a
        # thread that died - must open it again, or the re-armed supervisor would tick
        # forever and issue nothing. That is a silent re-arming, which is the exact
        # failure this module exists to remove, so it is a named line rather than a
        # field.
        _heartbeat["stop"] = stop
        _heartbeat["thread"] = new
        reopened = bool(_heartbeat["closed"])
        _heartbeat["closed"] = False
    with _state_lock:
        _state["heartbeat_threads"] = int(_state["heartbeat_threads"]) + 1
        _state["heartbeat_start_reason"] = None
    if reopened:
        _note_once("heartbeat-reopened",
                   "the heartbeat supervisor restarted, so the beat gate opened "
                   "again: beats will be issued from this tick on")
    report.update({"started": True, "alive": True})
    return report


def _note_once(key: str, message: str) -> bool:
    """One log line per episode, never one per poll - the `disk_restore` idiom.

    The rank-file writer runs on the engine's step path and on a 60 s thread; a line it
    emitted per beat would be 1,440 a day per worker. What it must emit is one line per
    EPISODE, which is the only granularity an operator reads under pressure.
    """
    try:
        if key in _notices:
            return False
        if len(_notices) >= 64:
            _notices.clear()
        _notices.add(key)
        _logger.warning("kvtier-dev-plugin: %s", str(message)[:640])
        return True
    except Exception as exc:  # noqa: BLE001 - a log line is never worth a fault
        snapshot._error(exc, "worker diag notice")
        return False


def note_heartbeat_refusal(reason) -> None:
    """SAY OUT LOUD, ONCE, that this process has no heartbeat (0.2.47).

    THE LINE THE OPERATOR GREPS FOR. `install()` refuses nothing quietly, but the
    heartbeat verdict used to ride inside the install report while `_brief_report` kept
    only `YES`/`NO(reason)` - so the engine log's `worker hook install ...` line said
    `ensure_installed=YES` about a worker whose proof mechanism had just failed to
    start.
    One WARNING per process, naming the reason, turns "did it even start?" from an
    unanswerable question into a grep.
    """
    text = _clip(reason, MAX_STRING_CHARS)
    with _state_lock:
        _state["heartbeat_start_failures_logged"] = (
            int(_state["heartbeat_start_failures_logged"]) + 1)
    window = _heartbeat_window_seconds()
    _note_once(f"heartbeat-not-started:{text}",
               f"[WORKERDIAG] the rank-file heartbeat did NOT start: {text}. "
               "Without it an idle worker's rank file ages past "
               f"{window}s and the scheduler stops offering disk restores to a "
               "worker that is perfectly alive")


def stop_heartbeat() -> dict:
    """Ask the thread to stop. Teardown, and the only way a test gets a quiet directory.

    THE TEARDOWN CANNOT RAISE AND MUST NOT SWALLOW LOUDLY EITHER: `contextlib.suppress`
    is this repo's own idiom for "expected to fail, nothing to do" (`disk_store`,
    `miss_attribution`), and it is truer here than `except: pass` - an already-dead
    event or an unjoinable thread are both normal at interpreter shutdown.

    SINCE 0.2.47 THE GATE CLOSES FIRST. The supervisor is only half the mechanism: the
    beats it issued have their own threads, so a teardown that stopped the loop and
    walked away could still have a beat reach its `os.replace` into the NEXT test's
    directory - and this module's whole subject is files appearing where a test asserted
    they would not. So: close the launch gate, stop and join the supervisor, then wait
    briefly for beats already in flight.
    """
    with _heartbeat_lock:
        stop = _heartbeat["stop"]
        thread = _heartbeat["thread"]
        _heartbeat["stop"] = None
        _heartbeat["thread"] = None
        _heartbeat["closed"] = True
    if stop is not None:
        with contextlib.suppress(Exception):
            stop.set()
    if thread is not None:
        with contextlib.suppress(Exception):
            thread.join(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with _state_lock:
            if int(_state["heartbeat_inflight"]) == 0:
                break
        time.sleep(0.005)
    with _state_lock:
        _state["heartbeat_thread_alive"] = False
    return {"stopped": thread is not None}


def _heartbeat_state_shell(state: str, reason) -> dict:
    """THE KEY SET, ONCE. Every heartbeat reading the rank file carries starts here and
    the real numbers overwrite it, so the shape cannot depend on the state - which is
    the property that makes it worth reading from a file at all.
    """
    window = _heartbeat_window_seconds()
    interval = max(0.001, float(HEARTBEAT_INTERVAL_SECONDS))
    return {"state": state,
            "state_reason": _clip(reason, MAX_STRING_CHARS),
            "alive": False,
            "started": False,
            "beats": 0,
            "beat_writes": 0,
            "inflight_beats": 0,
            "interval_seconds": float(HEARTBEAT_INTERVAL_SECONDS),
            "window_seconds": window,
            "beats_in_window": int(window // interval) if window else None,
            "last_beat_age_seconds": None,
            "last_beat_duration_seconds": None,
            "errors": 0,
            "fatal_beats": 0,
            "stalled_beats": 0,
            "jammed_beats": 0,
            "floor_skips": 0,
            "respawns": 0,
            "writer_pid": int(os.getpid()),
            "last_error": _clip(_state.get("last_error"), MAX_STRING_CHARS)}


def heartbeat_state() -> dict:
    """WHAT THIS PROCESS CAN SAY ABOUT ITS OWN PROOF, for the file it writes.

    WHY IT GOES IN THE FILE. The heartbeat's counters live in the worker process, and a
    worker publishes nowhere: the `/plugins/kvtier/snapshot` route is served by the API
    server, which is not a worker and answers `heartbeats=0 heartbeat_threads=0` with
    complete honesty about a process that has never seen the module. That gap - a fix
    with no reader - is why 0.2.45 could ship, fail in production and be
    indistinguishable
    from never having shipped. The rank file is the ONE artifact a worker definitely
    produces and a reader can see per rank, so the worker's own liveness goes in it.

    THE FIVE ANSWERS, and the reading each one is worth:
      * `beating`            - the thread is alive and its last beat WROTE. A file that
                               says this and is still stale means the writer STOPPED
                               after writing: the process died, or its supervisor did.
      * `alive_not_beating`  - the thread is alive and has NOT written. This is the
                               state a blocked contributor produces, and the one the
                               2026-10-21 hunt could not see at all: `state_reason`
                               names why, and `last_beat_duration_seconds` says how
                               long it is taking.
      * `started_not_alive`  - this process created a supervisor and it is gone. Before
                               0.2.47 only an engine step could bring it back - never
                               while the box was idle, which is when it matters.
      * `never_started`      - no supervisor was ever created here, and `state_reason`
                               says what refused it (no diagnostics directory, a thread
                               that would not start, an install that never landed).
      * `not_applicable`     - this process is not a worker rank-file writer at all; it
                               fitted nothing, so it keeps no proof to keep alive.
    """
    out = _heartbeat_state_shell("never_started", None)
    try:
        with _heartbeat_lock:
            thread = _heartbeat["thread"]
        alive = bool(thread is not None and thread.is_alive())
        ever = int(_state["heartbeat_threads"]) > 0
        out["alive"] = alive
        out["started"] = bool(ever)
        out["beats"] = int(_state["heartbeats"])
        out["beat_writes"] = int(_state["heartbeat_writes"])
        out["inflight_beats"] = int(_state["heartbeat_inflight"])
        out["errors"] = int(_state["heartbeat_errors"])
        out["fatal_beats"] = int(_state["heartbeat_fatal"])
        out["stalled_beats"] = int(_state["heartbeat_stalled"])
        out["jammed_beats"] = int(_state["heartbeat_jammed"])
        out["floor_skips"] = int(_state["heartbeat_floor_skips"])
        out["respawns"] = int(_state["heartbeat_respawns"])
        duration = _state["heartbeat_last_duration_seconds"]
        out["last_beat_duration_seconds"] = (round(float(duration), 3)
                                             if isinstance(duration,
                                                         (int, float)) else None)
        last = _state["heartbeat_last_mono"]
        out["last_beat_age_seconds"] = (
            round(time.monotonic() - float(last), 3)
            if isinstance(last, (int, float)) else None)
        if not _state["installed"]:
            out["state"] = "not_applicable"
            out["state_reason"] = (_state["skip_reason"]
                                   or "this process never fitted the worker "
                                      "rank-file writer, so it keeps no proof")
        elif not ever:
            out["state"] = "never_started"
            out["state_reason"] = (_state["heartbeat_start_reason"]
                                   or "install() never asked for a heartbeat thread")
        elif not alive:
            out["state"] = "started_not_alive"
            out["state_reason"] = (
                f"the beat supervisor this process started "
                f"({int(_state['heartbeat_threads'])} created, "
                f"{int(_state['heartbeat_respawns'])} respawned) is gone; an engine "
                "step re-arms it and an idle worker takes no steps")
        elif int(_state["heartbeat_writes"]) == 0:
            out["state"] = "alive_not_beating"
            out["state_reason"] = (_state["heartbeat_last_beat_reason"]
                                   or "the supervisor is alive but no beat has "
                                      "written the file yet")
        else:
            out["state"] = "beating"
            out["state_reason"] = (_state["heartbeat_last_beat_reason"]
                                   or f"{int(_state['heartbeat_writes'])} beat(s) "
                                      "have written this file")
    except Exception as exc:  # noqa: BLE001 - a diagnostic block is never a write fault
        snapshot._error(exc, "worker diag heartbeat state")
        fallback = _heartbeat_state_shell("never_started",
                                          f"the heartbeat state could not be read: "
                                          f"{type(exc).__name__}")
        fallback["state"] = "never_started"
        return fallback
    if out["state"] not in HEARTBEAT_STATES:   # a state no reader has been told about
        out["state"] = "never_started"
    return out


def heartbeat_status(*, force: bool = False) -> dict:
    """THE OPERATOR'S ANSWER TO "IS THIS WORKER'S PROOF BEING KEPT ALIVE?"

    `alive` is READ from the thread, never inferred from a counter, because a thread
    that died in the night is exactly the silent failure a heartbeat thread can produce.
    With `force`, a dead thread is respawned here too - the step-driven path does the
    same, so a worker that resumes stepping re-arms its own heartbeat for the next idle
    spell.
    """
    alive = _heartbeat_loop_alive()
    if not alive and force:
        start_heartbeat()
        alive = _heartbeat_loop_alive()
        if alive:
            with _state_lock:
                _state["heartbeat_respawns"] = int(_state["heartbeat_respawns"]) + 1
    last = _state["heartbeat_last_mono"]
    age = (round(time.monotonic() - float(last), 3)
           if isinstance(last, (int, float)) else None)
    window = _heartbeat_window_seconds()
    return {"started": _heartbeat["thread"] is not None or alive,
            "alive": alive,
            "interval_seconds": float(HEARTBEAT_INTERVAL_SECONDS),
            "window_seconds": window,
            "beats_in_window": (int(window // float(HEARTBEAT_INTERVAL_SECONDS))
                                if window else None),
            "last_heartbeat_age_seconds": age,
            "heartbeats": int(_state["heartbeats"]),
            "errors": int(_state["heartbeat_errors"])}


# --- the counters (the only per-step work) ----------------------------------

def _count_submitted(metadata) -> None:
    """Blocks this rank was just handed work for. Two len()s, no allocation."""
    try:
        stores = _len_or_none(getattr(metadata, "store_cpu_blocks", None))
        loads = _len_or_none(getattr(metadata, "load_gpu_blocks", None))
        if stores:
            _state["store_blocks_submitted"] += stores
        if loads:
            _state["load_blocks_submitted"] += loads
    except Exception as exc:  # noqa: BLE001 - a counter never breaks a step
        _state["fallbacks"] += 1
        snapshot._error(exc, "worker diag counters")


# --- the wrappers -----------------------------------------------------------

def _make_bind_metadata(original):
    def bind_connector_metadata(self, metadata):
        result = original(self, metadata)
        if metadata is not None:
            # Contained twice over: `_count_submitted` is contained inside, and
            # this guard is the backstop that keeps a broken diagnostic from ever
            # becoming a fault in the worker's own call.
            try:
                _count_submitted(metadata)
            except Exception as exc:  # noqa: BLE001 - never into the copy path
                _state["fallbacks"] += 1
                snapshot._error(exc, "worker diag bind wrapper")
        return result
    return bind_connector_metadata


def _make_get_finished(original):
    def get_finished(self, finished_req_ids):
        result = original(self, finished_req_ids)
        # THE HANDLER, NOTED EVERY STEP (0.2.45). A weak reference, taken here because
        # this is the first place a step hands us the live object, and the heartbeat
        # thread needs it to write a document that measures the arena instead of
        # reporting nothing. Cheap enough to do every step: one weakref and a lock,
        # no I/O.
        try:
            note_handler(self)
        except Exception as exc:  # noqa: BLE001 - never into the step
            _state["fallbacks"] += 1
            snapshot._error(exc, "worker diag handler note")
        # The write happens AFTER the fork's own work, so a diagnostic can never
        # reorder or replace what the scheduler is told - and it is contained, so
        # it can never replace it either: the disk may be gone, the step returns.
        try:
            _maybe_write(self)
            # THE SELF-HEALING BEAT (0.2.45). A heartbeat thread that died leaves the
            # worker proving again only while steps arrive - the correct fallback (a
            # busy worker needs no heartbeat), except that nothing would ever re-arm it
            # for the NEXT idle spell. `force=True` respawns a thread this process can
            # see is gone, and the counter says it happened, so "the disk went off
            # again" has a reading in the payload instead of a theory.
            heartbeat_status(force=True)
        except Exception as exc:  # noqa: BLE001 - a file is never a step failure
            _state["fallbacks"] += 1
            snapshot._error(exc, "worker diag write trigger")
        return result
    return get_finished


_FACTORIES = {METHOD_BIND_METADATA: _make_bind_metadata,
              METHOD_GET_FINISHED: _make_get_finished}


def install(cls=None) -> dict:
    """Wrap the two worker methods. Idempotent, never raises, INERT by default.

    With `VLLM_E2_TIER_DIAG` unset this refuses without touching anything, so a
    plugin that is otherwise live still creates no files anywhere.
    """
    report = {"installed": False, "attached": [], "replaced": [], "skipped": [],
              "skip_detail": [], "nested": [], "class": None, "reason": None}

    def refuse(method_name: str, reason: str) -> None:
        """Record a refusal per method, the way `guard_mode.install` does.

        A PARTIAL install is a state an operator has to be able to read: "the rank
        file exists but shows no completion counts" means one of these two names
        was refused, and `skip_detail` says which and why.
        """
        report["skipped"].append(method_name)
        report["skip_detail"].append({"method": method_name,
                                      "reason": reason[:MAX_STRING_CHARS]})
        _refuse(reason)
    try:
        directory = config.diagnostics_dir()
        if directory is None:
            _refuse(f"{config.ENV_DIAG_DIR} is not set: no diagnostics directory, "
                    f"so nothing is written and nothing is wrapped")
            report["reason"] = _state["skip_reason"]
            return report
        if cls is None:
            cls = worker_class()
        if cls is None:
            _refuse(f"{TARGET_MODULE} is not imported in this process "
                    f"({TARGET_CLASS} absent) - a worker that has not built its "
                    f"offload handler yet")
            report["reason"] = _state["skip_reason"]
            return report
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        _state["dir"] = str(directory)
        _state["interval_steps"] = config.worker_write_interval_steps()
        _state["min_interval_seconds"] = config.worker_min_interval_seconds()
        for name in WRAPPED_METHODS:
            original = getattr(cls, name, None)
            if original is None or not callable(original):
                refuse(name, f"{report['class']}.{name} is missing or not callable")
                continue
            owner = getattr(original, OWNER_ATTR, None)
            if owner is not None and owner != OWNER_TAG:
                refuse(name, f"{report['class']}.{name} carries another plugin's "
                             f"owner marker, refusing to overwrite")
                continue
            replaced = owner == OWNER_TAG
            if not replaced:
                home = getattr(original, "__module__", None)
                if home != TARGET_MODULE and not str(home or "").startswith(
                        SIBLING_PACKAGE):
                    refuse(name, f"{report['class']}.{name} is neither ours nor "
                                 f"the fork's own: defined in {home!r}, not "
                                 f"{TARGET_MODULE!r}")
                    continue
                if home != TARGET_MODULE:
                    # A sibling layer of this package on the same method: go ON
                    # TOP OF it so both facilities run, and say so in the report.
                    report["nested"].append(name)
            if replaced:
                original = getattr(original, ORIGINAL_ATTR, original)
                report["replaced"].append(name)
            else:
                report["attached"].append(name)
            wrapped = _FACTORIES[name](original)
            setattr(wrapped, OWNER_ATTR, OWNER_TAG)
            setattr(wrapped, ORIGINAL_ATTR, original)
            wrapped.__name__ = name
            wrapped.__doc__ = (
                "kvtier-dev-plugin: the worker's own "
                f"{name}, plus the budgeted per-rank diagnostics write "
                "(VLLM_E2_TIER_DIAG). Contained; nothing here changes what the "
                "fork returns.")
            setattr(cls, name, wrapped)
        report["installed"] = bool(report["attached"] or report["replaced"])
        if report["attached"]:
            _state["installs"] += 1
            _state["class"] = report["class"]
            _state["installed"] = True
            _state["enabled"] = True
            _state["nested"] = list(report["nested"])
            if not report["skipped"]:
                _state["skip_reason"] = None
            _state["skip_detail"] = list(report["skip_detail"])
        # THE HEARTBEAT STARTS HERE, AT INSTALL, NOT AT THE FIRST STEP (0.2.45).
        # `disk_gpu_write` shipped this mistake in 0.2.43 (its thread's only starter sat
        # below the path that needed what the thread produces, so the first write was
        # unreachable), and the same shape here is the idle-worker defect itself: a
        # heartbeat begun by an engine step can never begin on a worker that is idle -
        # and an idle worker is exactly the one whose proof would otherwise age out.
        # Starting it allocates nothing: the thread waits its interval before its first
        # write, and with no diagnostics directory `start_heartbeat` refuses rather than
        # writing, so an inert tier neither spawns a thread nor creates a file.
        if report["installed"]:
            heartbeat = start_heartbeat()
            report["heartbeat"] = heartbeat
            if not heartbeat.get("started"):
                report["heartbeat_reason"] = heartbeat.get("reason")
                # A SUCCESSFUL INSTALL WITH NO HEARTBEAT IS A HALF AN INSTALL, and until
                # 0.2.47 it was invisible: the install report's `heartbeat` field went
                # into `_brief_report`, which keeps only YES/NO, so the boot line read
                # `ensure_installed=YES` for a worker whose proof mechanism had just
                # failed to start. Say it once, at WARNING, with the reason.
                note_heartbeat_refusal(heartbeat.get("reason"))
        if report["replaced"]:
            _state["reinstalls"] += 1
        if not report["installed"]:
            report["reason"] = _state["skip_reason"]
            _state["skip_detail"] = list(report["skip_detail"])
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        _state["fallbacks"] += 1
        snapshot._error(exc, "worker diag install")
        report["reason"] = f"install raised: {type(exc).__name__}"[:MAX_STRING_CHARS]
        return report


def ensure_installed(connector=None) -> dict:
    """Retry once the worker's offload module exists. Marker-cheap no-ops first.

    `engine_side.register()` runs before the connector imports the worker module,
    so the first attempt is normally refused; the retry is driven by the
    handshake hook in `cpu_capacity` (`engine_side` wires it), which fires exactly
    once this rank has an offload handler - and hands us the connector, whose
    `worker_handler` (`simple_cpu_offload_connector.py:167`) names the exact class
    to wrap. With no connector in hand we fall back to the class named in the
    already-imported module (the `store_hook.manager_class()` idiom).
    """
    if _state["installed"]:
        return {"installed": True, "reason": "already installed"}
    if _state["class"] is not None:
        return {"installed": False,
                "reason": _state["skip_reason"] or "already attempted"}
    handler = getattr(connector, "worker_handler", None) if connector is not None \
        else None
    return install(type(handler) if handler is not None else None)


def uninstall(cls=None) -> dict:
    """Restore the worker's own methods. Never raises."""
    report = {"uninstalled": False, "restored": [], "left": [], "class": None,
              "reason": None}
    try:
        if cls is None:
            cls = worker_class()
        if cls is None:
            report["reason"] = f"{TARGET_CLASS} is not imported here"
            return report
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        for name in WRAPPED_METHODS:
            current = getattr(cls, name, None)
            if getattr(current, OWNER_ATTR, None) != OWNER_TAG:
                report["left"].append(name)
                continue
            setattr(cls, name, getattr(current, ORIGINAL_ATTR))
            report["restored"].append(name)
        report["uninstalled"] = bool(report["restored"])
        if report["restored"]:
            _state["uninstalls"] += 1
            _state["installed"] = False
        return report
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "worker diag uninstall")
        report["reason"] = f"uninstall raised: {type(exc).__name__}"[:MAX_STRING_CHARS]
        return report
