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
     "arena": {"bytes_total": int, "bytes_used": int, "num_blocks": int,
               "block_bytes_by_group": [int, ...] | null},
     "counters": {"stored": int, "restored_hits": int,
                  "store_events_pending": int, "load_events_pending": int},
     "writer": {"plugin": "kvtier-dev-plugin/1", "writes": int, ...},
     "error": null}

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
  `arena.block_bytes_by_group`  per KV group, `len(group.layer_names) *
              spec.page_size_bytes` (kv_cache_interface.py:1400 / :186) - the
              group's bytes per block, which is what the reader's group totals
              want. `writer.block_bytes_by_group_meaning` says how it is derived.
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

import json
import os
import sys
import time

from . import config, snapshot

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
# The keys `_clip_document` may never drop. The first seven are what the proxy
# host's reader REQUIRES (a file missing one is `malformed`); the rest are the
# writer's own optional halves, which the triage sheds in a fixed order. Anything
# NOT listed - a contributed section, or a field a future writer adds - is what
# gets dropped first, so a byte ceiling can never cost a required field.
KEEP_KEYS = ("version", "epoch", "ts", "role", "rank", "arena", "counters",
             "writer", "error", "parked", "equality_sample")
ROLE = "worker"
MAX_FILE_BYTES = 32 * 1024  # the reader's own cap is 256 KiB; we stay far below
MAX_STRING_CHARS = 200
MAX_GROUPS = 32
WRITER_TAG = "kvtier-dev-plugin/1"

# --- the contribution seam: ONE writer, several contributors -----------------
# A worker-side module that has facts for the operator (today `disk_store`: the
# SSD tier's own verdict) must NOT write a file. A second writer of
# `rank-<rank>.json` would race the atomic `os.replace`, and a second file name
# would be a second contract the proxy host's reader has never heard of. So other
# modules hand this one a producer and stay silent; `worker_diag` validates and
# writes, and it is the only code that opens the file.
#
# THREE LIMITS, because a contribution is a guest in someone else's file:
#   * `CONTRIBUTION_SECTIONS` is a HARD allow-list of the top-level keys a
#     contributor may add. It is the whole shape of the trust: `"disk"` is listed,
#     `"arena"`/`"counters"`/`"version"` are not, so a contributor cannot overwrite
#     a field the reader REQUIRES. An unknown section name is refused at register
#     time, not at write time.
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
CONTRIBUTION_SECTIONS = ("disk",)
MAX_CONTRIBUTIONS = 4
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
                   "last_path": None, "bytes_written": 0, "fallbacks": 0})
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


def register_contribution(name, produce, owner: str | None = None) -> dict:
    """Offer a section for the rank file. Contained; returns a dict; never raises.

    `produce` is a zero-argument callable returning a plain JSON-serialisable
    dict, or a dict to publish verbatim. A callable is the honest form for a
    module with live counters (it is asked at write time, so the file carries the
    numbers of the step it was written on); a dict suits a module with boot-
    constant facts.

    Refused - always with a reason, never silently - when the section name is not
    on `CONTRIBUTION_SECTIONS`, when the name is already held by a different
    producer, when there is nothing sensible to produce, or when the seam already
    holds `MAX_CONTRIBUTIONS` sections. Re-registering the SAME producer is
    idempotent, because `install()` runs at plugin load AND again off the
    handshake hook.
    """
    report = {"registered": False, "section": name, "reason": None}
    try:
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
                               "owner": owner or "unknown"}
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
        except Exception as exc:  # noqa: BLE001 - a contributor is never a step fault
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
    """Per KV group, bytes per block: `len(layer_names) * page_size_bytes`.

    A group's layers all share one block table by definition
    (kv_cache_interface.py:1392), so the tier's per-block bytes for that group is
    its layer count times the spec's page size. None (absent) rather than a
    partial list when any term is missing: the reader would rather note a missing
    field than fold a wrong total into a group sum.
    """
    groups = getattr(getattr(handler, "kv_cache_config", None), "kv_cache_groups",
                     None)
    if not groups:
        return None, None
    sizes = []
    for group in list(groups)[:MAX_GROUPS]:
        spec = getattr(group, "kv_cache_spec", None)
        layers = _int(_len_or_none(getattr(group, "layer_names", None)))
        page = _int(getattr(spec, "page_size_bytes", None))
        if layers is None or page is None:
            return None, "a group is missing layer_names or page_size_bytes"
        sizes.append(layers * page)
    if len(list(groups)) > MAX_GROUPS:
        return None, f"more than {MAX_GROUPS} groups"
    return sizes, None


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
    """The cheap "did anything change?" test. Small ints only, never a serialise."""
    return (arena_bytes,
            _int(getattr(handler, "num_cpu_blocks", None)),
            _len_or_none(getattr(handler, "_pending_store_event_indices", None)),
            _len_or_none(getattr(handler, "_pending_load_event_indices", None)),
            _int(getattr(handler, "_store_hwm", None)),
            _int(getattr(handler, "_load_hwm", None)),
            int(_state["store_blocks_submitted"]),
            int(_state["load_blocks_submitted"]))


def build_document(handler, rank, rank_source) -> dict:
    """The file's content. Every number is a real reading or the key is absent.

    The contributed sections are appended LAST, as siblings of `version`/`arena`
    (the reader's own `gpu_census` precedent), and a contributor that produced
    nothing contributes NO key - see `_contribution_sections`.
    """
    arena_bytes, bytes_source = _arena_bytes(handler)
    block_bytes, block_bytes_note = _block_bytes_by_group(handler)
    metadata = getattr(handler, "_connector_metadata", None)
    per_block = _bytes_per_block(handler, arena_bytes)
    in_flight_blocks = None
    if metadata is not None and per_block is not None:
        in_flight_blocks = _len_or_none(getattr(metadata, "store_cpu_blocks", None))
    bytes_used = 0
    if in_flight_blocks:
        bytes_used = in_flight_blocks * per_block
    epoch, epoch_scope = config.diagnostics_epoch()
    arena: dict = {"bytes_total": arena_bytes if arena_bytes is not None else 0,
                   "bytes_used": bytes_used,
                   "num_blocks": _int(getattr(handler, "num_cpu_blocks", None))
                   or 0}
    if block_bytes is not None:
        arena["block_bytes_by_group"] = block_bytes
    counters = {"stored": int(_state["store_blocks_submitted"]),
                "restored_hits": int(_state["load_blocks_submitted"]),
                "store_events_pending": _len_or_none(
                    getattr(handler, "_pending_store_event_indices", None)) or 0,
                "load_events_pending": _len_or_none(
                    getattr(handler, "_pending_load_event_indices", None)) or 0}
    sections, contribution_notes = _contribution_sections()
    document = {
        "version": SCHEMA_VERSION,
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
                "per KV group: len(group.layer_names) * spec.page_size_bytes "
                "(kv_cache_interface.py:1400, :186)"),
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
        # The write happens AFTER the fork's own work, so a diagnostic can never
        # reorder or replace what the scheduler is told - and it is contained, so
        # it can never replace it either: the disk may be gone, the step returns.
        try:
            _maybe_write(self)
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
