"""Read-only view of the scheduler-side KV tier. Stdlib imports only.

This module runs on the thread that serves traffic: an engine-side utility
method is dispatched from the EngineCore busy loop
(vllm/v1/engine/core.py:1669-1671 -> 1712-1728), the same thread that runs
`step_fn()` (core.py:1579). So:

  * imports are `json`, `os`, `time`, `secrets` - nothing else. No torch, no
    CUDA, no vllm, no numpy. `test_no_heavy_imports_in_read_side()` enforces
    it by AST. This mirrors `vllm/v1/simple_kv_offload/manager.py`, which
    contains zero torch references by design; the worker side
    (`simple_kv_offload/worker.py:293-294`) records on the live compute stream
    and is never touched from here.
  * every read is O(1) on the real objects: `len()` of a dict, a maintained
    counter (`BlockPool.get_num_free_blocks()` ->
    `free_block_queue.num_free_blocks`, block_pool.py:861), or a dataclass copy
    (`get_boundary_store_stats()` returns `replace(...)`, manager.py:947-948,
    so it is snapshot-safe by construction).
  * nothing here raises into the caller: `snapshot_for_engine()` is the
    containment boundary, in this fork's idiom
    (vllm/v1/core/sched/e2_alloc_log.py:143,151,201 with the capped `_error()`
    at :92). It catches `Exception`, never `BaseException`: a
    KeyboardInterrupt or SystemExit must not be swallowed by a diagnostic.

Field names below were read out of the fork at HEAD (10b8e07):
  manager.py  : `_lazy_mode`, `_target_free`, `block_size`, `hash_block_size`,
                `fa_block_size`, `fa_gidx`, `group_block_sizes`,
                `num_cpu_blocks`, `cpu_block_pool`, `_gpu_block_pool`,
                `_reqs_to_load`, `_reqs_to_store`, `_pending_cpu_hits`,
                `_load_event_to_reqs`, `_store_event_to_blocks`,
                `_abandoned_store_event_to_blocks`, `_abandoned_reqs_to_load`,
                `_pending_finished_stores`, `_in_flight_store_gpu_blocks`,
                `_store_event_pending_counts`, `_load_event_counter`,
                `_store_event_counter`, `boundary_store_stats`
  manager.py  : `has_pending_stores()` (:1119),
                `get_boundary_store_stats()` (:947),
                `BoundaryStoreStats` fields (:125-132)
  connector   : `scheduler_manager` (simple_cpu_offload_connector.py:129,157)
  block_pool  : `num_gpu_blocks` (:173), `get_num_free_blocks()` (:861)
"""

from __future__ import annotations

import json
import secrets
import time

PROTOCOL_VERSION = 1

# Containment state, in the style of e2_alloc_log._state: count every error
# forever, log at most the first MAX_ERROR_LOGS of them, and never let the
# writer itself become the second failure.
MAX_ERROR_LOGS = 20
_state = {"calls": 0, "errors": 0, "logged": 0, "epoch": None, "logger": None}

# The eight decline reasons the tier counts, manager.py:125-132. Kept as a
# tuple so a fork that renames one shows up as an `unknown_fields` entry here
# rather than as a silently missing number.
BOUNDARY_STAT_FIELDS = (
    "published",
    "stored",
    "dropped_cpu_full",
    "dropped_request_gone",
    "skipped_already_cached",
    "skipped_in_flight",
    "dropped_null_block",
    "dropped_not_hashed",
)


def attach_logger(logger) -> None:
    """Handed a logger by the engine-side installer; None keeps _error quiet."""
    _state["logger"] = logger


def error_counters() -> dict:
    return {"calls": _state["calls"], "errors": _state["errors"],
            "logged": _state["logged"]}


def epoch() -> str:
    """Per-process id: proves two snapshots came from the same engine boot.

    Generated lazily so importing this module costs no syscall and no entropy.
    """
    if _state["epoch"] is None:
        _state["epoch"] = secrets.token_hex(4)
    return _state["epoch"]


def _error(exc: BaseException, where: str) -> None:
    """Count, log up to a cap, and never raise. `exc` is never re-raised."""
    _state["errors"] += 1
    logger = _state["logger"]
    if logger is None or _state["logged"] >= MAX_ERROR_LOGS:
        return
    _state["logged"] += 1
    try:  # noqa: SIM105 - no contextlib import for the last resort
        logger.warning("[KVTIER_DEV] %s failed (%d/%d logged): %s: %s",
                       where, _state["logged"], MAX_ERROR_LOGS,
                       type(exc).__name__, exc)
    except Exception:  # noqa: BLE001, S110
        # Deliberate: the last-resort writer gets no contextlib import, exactly
        # like the nested `except Exception: pass` inside the fork's own
        # e2_alloc_log._error() at :92. A broken log must be silent, never fatal.
        pass


def _int_or_none(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _count(container) -> int | None:
    """len() of a real container, None if the attribute is missing/weird."""
    if container is None:
        return None
    try:
        return len(container)
    except TypeError:
        return None


def _pool_view(pool) -> dict:
    """Total/free for a BlockPool. Both reads are maintained counters."""
    if pool is None:
        return {"bound": False}
    free = None
    try:
        getter = getattr(pool, "get_num_free_blocks", None)
        if callable(getter):
            free = _int_or_none(getter())
    except Exception as exc:  # noqa: BLE001 - a pool that lies is data, not a crash
        _error(exc, "pool.get_num_free_blocks")
    return {"bound": True,
            "total_blocks": _int_or_none(getattr(pool, "num_gpu_blocks", None)),
            "free_blocks": free}


def _copy_boundary_stats(stats) -> dict:
    """Copy the counter dataclass into plain ints (already a copy upstream)."""
    if stats is None:
        return {"available": False}
    try:
        mapping = vars(stats)
    except TypeError:
        return {"available": False,
                "reason": f"not a simple object: {type(stats).__name__}"}
    out: dict = {"available": True}
    for key, value in mapping.items():
        as_int = _int_or_none(value)
        out[key] = value if as_int is not None else repr(value)[:64]
    missing = [f for f in BOUNDARY_STAT_FIELDS if f not in out]
    unknown = [f for f in mapping
               if f not in BOUNDARY_STAT_FIELDS and f != "available"]
    if missing:
        out["missing_fields"] = missing
    if unknown:
        out["unknown_fields"] = unknown
    return out


def tier_view(manager) -> dict:
    """The scheduler-side offload manager, or a reason it is not there."""
    if manager is None:
        return {"present": False,
                "reason": "no scheduler_manager: the engine is running without "
                          "SimpleCPUOffloadConnector, or prefix caching is off"}

    gpu_pool = getattr(manager, "_gpu_block_pool", None)
    pending = None
    try:
        has_pending = getattr(manager, "has_pending_stores", None)
        if callable(has_pending):
            pending = bool(has_pending())
    except Exception as exc:  # noqa: BLE001
        _error(exc, "has_pending_stores")

    try:
        stats = _copy_boundary_stats(manager.get_boundary_store_stats())
    except Exception as exc:  # noqa: BLE001
        _error(exc, "get_boundary_store_stats")
        stats = {"available": False, "reason": "get_boundary_store_stats raised"}

    group_sizes = getattr(manager, "group_block_sizes", None)
    return {
        "present": True,
        "mode": "lazy" if getattr(manager, "_lazy_mode", False) else "eager",
        "target_free_gpu_blocks": _int_or_none(getattr(manager, "_target_free", None)),
        "block_size": _int_or_none(getattr(manager, "block_size", None)),
        "hash_block_size": _int_or_none(getattr(manager, "hash_block_size", None)),
        "fa_block_size": _int_or_none(getattr(manager, "fa_block_size", None)),
        "fa_group_index": _int_or_none(getattr(manager, "fa_gidx", None)),
        "group_block_sizes": (list(group_sizes) if group_sizes is not None else None),
        "num_cpu_blocks": _int_or_none(getattr(manager, "num_cpu_blocks", None)),
        "cpu_pool": _pool_view(getattr(manager, "cpu_block_pool", None)),
        "gpu_pool": _pool_view(gpu_pool),
        "pending_stores": pending,
        "counts": {
            "reqs_to_load": _count(getattr(manager, "_reqs_to_load", None)),
            "reqs_to_store": _count(getattr(manager, "_reqs_to_store", None)),
            "abandoned_reqs_to_load": _count(
                getattr(manager, "_abandoned_reqs_to_load", None)),
            "load_events_to_reqs": _count(
                getattr(manager, "_load_event_to_reqs", None)),
            "store_events": _count(getattr(manager, "_store_event_to_blocks", None)),
            "abandoned_store_events": _count(
                getattr(manager, "_abandoned_store_event_to_blocks", None)),
            "store_events_awaiting_workers": _count(
                getattr(manager, "_store_event_pending_counts", None)),
            "pending_finished_stores": _count(
                getattr(manager, "_pending_finished_stores", None)),
            "in_flight_store_gpu_blocks": _count(
                getattr(manager, "_in_flight_store_gpu_blocks", None)),
            "pending_cpu_hits": _count(getattr(manager, "_pending_cpu_hits", None)),
        },
        "event_counters": {
            "load_event_counter": _int_or_none(
                getattr(manager, "_load_event_counter", None)),
            "store_event_counter": _int_or_none(
                getattr(manager, "_store_event_counter", None)),
        },
        "boundary_store_stats": stats,
    }


def build_view(engine) -> dict:
    """One pass over the engine's scheduler-side state. Reads only."""
    scheduler = getattr(engine, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    manager = getattr(connector, "scheduler_manager", None)
    return {
        "version": PROTOCOL_VERSION,
        "epoch": epoch(),
        "ts": round(time.time(), 6),
        "mono": round(time.monotonic(), 6),
        "engine": {"class": type(engine).__name__},
        "scheduler": {
            "present": scheduler is not None,
            "class": type(scheduler).__name__ if scheduler is not None else None,
            "connector": type(connector).__name__ if connector is not None else None,
            "requests": _count(getattr(scheduler, "requests", None)),
            "running": _count(getattr(scheduler, "running", None)),
            "waiting": _count(getattr(scheduler, "waiting", None)),
        },
        "tier": tier_view(manager),
    }


def policy_section(view, holder) -> dict:
    """Run the operator's policy over the view. Contained, and timed.

    `decide_us` is the evidence that the policy is cheap enough to sit on the
    engine loop thread: if it ever stops being microseconds, it shows up here.
    """
    started = time.perf_counter()
    try:
        report = holder.ensure_loaded()
        if not report.get("ok"):
            return {"loaded": False, "decide_us": None, "result": None,
                    "error": report.get("error"), "traceback": None}
        decide = holder.decide
        if decide is None:
            return {"loaded": True, "decide_us": None, "result": None,
                    "error": "policy module has no decide()", "traceback": None}
        result = decide(view)
        return {"loaded": True,
                "decide_us": int((time.perf_counter() - started) * 1e6),
                "result": result, "error": None, "traceback": None}
    except Exception as exc:  # noqa: BLE001 - BaseException must propagate
        _error(exc, "policy.decide")
        return {"loaded": True,
                "decide_us": int((time.perf_counter() - started) * 1e6),
                "result": None,
                "error": f"{type(exc).__name__}: {exc}"[:300], "traceback": None}


def json_safe(value: dict) -> dict:
    """Force the payload to msgpack/JSON primitives before it leaves the engine.

    The utility result is msgpack-encoded by the EngineCore output thread; a
    value it cannot encode would fail there, not here. A JSON round trip leaves
    only dict/list/str/int/float/bool/None behind. allow_nan=False also turns
    a stray NaN/inf into a contained error instead of JSON the HTTP layer
    cannot emit.
    """
    return json.loads(json.dumps(value, allow_nan=False,
                                 separators=(",", ":")))


def store_hook_section(engine) -> dict:
    """Whether the runtime store-policy hook is live, and what it has done.

    The import is function-level on purpose: `store_hook` imports THIS module
    (for `_error`), so a module-level import here would be a cycle. By the time
    a handler runs, both modules are fully loaded, and the lookup costs one
    `sys.modules` hit per snapshot - not per block, which is where the real
    per-call cost of the hook lives.
    """
    from . import store_hook  # noqa: PLC0415 - see above
    try:
        return store_hook.patch_report(store_hook.manager_of(engine))
    except Exception as exc:  # noqa: BLE001 - a missing report is data, not a 500
        _error(exc, "store hook report")
        return {"patched": False,
                "reason": f"report failed: {type(exc).__name__}"[:200]}


def cpu_capacity_section(engine) -> dict:
    """Whether the CPU block-id ceiling is clamped to what every rank can hold.

    The one section of the view that reports a SAFETY property rather than a
    measurement: `installed` / `reason` (and why not), the per-rank arena sizes
    the handshake carried, the chosen cap, whether the clamp was applied, whether
    the tier is in fail-safe zero mode, and how many allocations the cap has
    refused. Function-level import because `cpu_capacity` imports THIS module.
    """
    from . import cpu_capacity  # noqa: PLC0415 - see above
    manager = None
    try:
        manager = cpu_capacity.manager_of(engine)
    except Exception as exc:  # noqa: BLE001 - a manager we cannot reach is data
        _error(exc, "cpu capacity manager")
    try:
        return cpu_capacity.section(manager)
    except Exception as exc:  # noqa: BLE001 - the shape must survive regardless
        _error(exc, "cpu capacity section")
        return cpu_capacity.empty_section(f"{type(exc).__name__}: {exc}")


def backport_section(engine) -> dict:
    """The PR #56810 backport: is the ring-group fix live, and is the tier armed?

    Two questions in one section because they are read together. (1) Is the
    upstream divisibility fix actually installed - all four methods, or did a
    fingerprint refuse one? (2) Is the tier armed, i.e. is the connector allowed
    to schedule any store or load at all? `armed` is the kill-switch state and it
    belongs in the SAME section as the patch it governs: "patched and disarmed"
    and "not patched" look similar in a payload and mean very different things
    (one serves from GPU KV alone, the other crashes on a hybrid geometry).

    Function-level import because `backport` imports THIS module.
    """
    from . import backport  # noqa: PLC0415 - cycle, see store_hook_section
    manager = None
    try:
        manager = backport.manager_of(engine)
    except Exception as exc:  # noqa: BLE001 - a manager we cannot reach is data
        _error(exc, "backport manager")
    try:
        return backport.section(manager)
    except Exception as exc:  # noqa: BLE001 - the shape must survive regardless
        _error(exc, "backport section")
        return backport.empty_section(f"{type(exc).__name__}: {exc}")


def dev_state_sections() -> dict:
    """The DEVELOPMENT-state sections: guard mode, the trace ring, the worker writer.

    These are the three pieces of the plugin that CHANGE what the engine does, so
    every snapshot publishes them. Guard mode in particular must be impossible to
    miss: `snapshot_for_engine` also copies it to the top level as `guard_mode`,
    because a guarded dev engine mistaken for a production one is the single
    worst way this plugin could hurt.

    Function-level imports again: `guard_mode` records into `trace_ring` and
    `kvconfig` reads `guard_mode`, so a module-level import here would be a cycle.
    """
    from . import guard_mode, step_trace, trace_ring, worker_diag  # noqa: PLC0415
    out: dict = {}
    try:
        out["guard"] = guard_mode.section()
        out["guard_mode"] = guard_mode.is_armed()
    except Exception as exc:  # noqa: BLE001 - a missing section is data, not a 500
        _error(exc, "guard section")
        out["guard"] = {"guard_mode": None, "error": f"{type(exc).__name__}"[:200]}
        out["guard_mode"] = None
    try:
        counters = trace_ring.counters()
        counters["step_observers"] = step_trace.counters()
        out["trace"] = counters
    except Exception as exc:  # noqa: BLE001
        _error(exc, "trace section")
        out["trace"] = {"error": f"{type(exc).__name__}"[:200]}
    try:
        out["worker_diag"] = worker_diag.section()
    except Exception as exc:  # noqa: BLE001
        _error(exc, "worker diag section")
        out["worker_diag"] = {"error": f"{type(exc).__name__}"[:200]}
    return out


def snapshot_for_engine(engine, holder) -> dict:
    """THE engine-side handler body. Never raises an Exception; reads only."""
    _state["calls"] += 1
    started = time.perf_counter()
    try:
        view = build_view(engine)
        view["policy"] = policy_section(view, holder)
        view["store_hook"] = store_hook_section(engine)
        view["cpu_capacity"] = cpu_capacity_section(engine)
        view["pr56810"] = backport_section(engine)
        view.update(dev_state_sections())
        # `dev_state_sections()` already put `guard_mode` at the top level: the
        # field a reader must never have to dig for. `guard` is the detail.
        view["build_us"] = int((time.perf_counter() - started) * 1e6)
        view["self"] = error_counters()
        return json_safe(view)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        _error(exc, "snapshot")
        # Hand-built from formatted strings only: this path cannot raise. The
        # guard question is still the first thing a reader wants, even here.
        return {"version": PROTOCOL_VERSION,
                "epoch": str(epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "guard_mode": _guard_mode_quietly(),
                "counters": {"calls": int(_state["calls"]),
                             "errors": int(_state["errors"]),
                             "logged": int(_state["logged"])}}


def _guard_mode_quietly():
    """guard_mode.is_armed() that cannot fail inside the containment path."""
    try:
        from . import guard_mode  # noqa: PLC0415 - cycle, see dev_state_sections
        return guard_mode.is_armed()
    except Exception:  # noqa: BLE001, S112 - the last resort stays silent
        return None


def reload_policy_on_engine(holder) -> dict:
    """THE engine-side reload handler body. Never raises; reports as data."""
    _state["calls"] += 1
    try:
        report = holder.reload()
        report["self"] = error_counters()
        report["epoch"] = epoch()
        return report
    except Exception as exc:  # noqa: BLE001
        _error(exc, "reload")
        return {"ok": False, "rolled_back": True, "kept": None, "applied": None,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "traceback": None, "self": {"calls": int(_state["calls"]),
                                            "errors": int(_state["errors"]),
                                            "logged": int(_state["logged"])}}
