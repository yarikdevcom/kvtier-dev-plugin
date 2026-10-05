"""Capability 4: GUARD MODE - turn a tier exception into a readable record.

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
A DEVELOPMENT facility. When (and only when) it is explicitly armed, the
tier's scheduler-facing entry points are wrapped so that an `Exception` raised
inside one of them is

  1. captured AS DATA - type, message, a bounded traceback string, and a compact
     context snapshot (the group sizes, the hashes, the free/pinned counts, the
     block ids from the call's own arguments, and the operands of the arithmetic
     the fork's assert uses), pushed into a small capped deque AND into the
     trace ring; and
  2. converted into that method's DECLINED result, so the scheduler keeps
     running instead of dying inside a tier it does not need.

It is OFF at plugin IMPORT and ARMED BY THE BOOT: `backport.boot_from_connector`
installs and arms it at connector creation, immediately before it arms the tier,
so there is no window in which a store or a lookup runs unguarded. Arming it is
not a favour the tier asks for - an unguarded `build_connector_meta` that raises
kills the EngineCore, and the boot refuses to arm the tier unless the guard is
provably a layer of every entry point the class actually has. The state is echoed
as `guard_mode: true|false` in `status` and in EVERY verb payload, because a
guarded dev engine must never be mistaken for a production one. There is no env
var that turns it ON - `VLLM_E2_TIER_DEV_BOOT_INERT` only turns the boot's own
arming OFF - and `POST /plugins/kvtier/guard {"armed": false}` is the operator's
disarm, which also removes the wrappers when nothing else consumes them.

**A guard hides bugs.** A declined store is a park that did not happen; a
declined `get_num_new_matched_tokens` is a prefix hit the tier refused; a
declined `build_connector_meta` is a step that did nothing. None of those are
fatal, and none of them are loud. A decline also leaves the manager's state
partially advanced by the exception, so pins/refs acquired before the raise may
leak until the next `reset`. DISARM BEFORE ANY PRODUCTION BOOT:
`POST /plugins/kvtier/guard {"armed": false}` (and see *Uninstall / rollback* in
the README).

NEVER SWALLOWS A BaseException
------------------------------
Every wrapper catches `except Exception` and nothing wider, so
`KeyboardInterrupt`, `SystemExit` and a generator's `GeneratorExit` propagate
into the engine loop exactly as they would without this plugin - the same rule
`snapshot.snapshot_for_engine()` and the fork's own `e2_alloc_log` follow.
`tests/test_guard_mode.py::test_a_base_exception_always_propagates` pins it.

SOME FAULTS ARE NOT DECINABLE, AND SAY SO
-----------------------------------------
Declining `update_connector_output` would drop a worker's "these stores
finished" report permanently (the metadata object is per-step and not retried),
so the store accounting would desync while looking healthy. For that method -
and for anything the fork adds later that we do not have a decline for - the
wrapper captures the exception as data and **re-raises**: the record you wanted,
with the crash the engine would have had. `_declines` reports which methods have
a known decline and which are capture-and-reraise.

INIT-TIME PATHS ARE NOT WRAPPED AT ALL
--------------------------------------
`bind_gpu_block_pool` (manager.py:315) is deliberately not guarded: its failure
must abort the boot loudly. Masking it would convert a loud boot failure into a
silent tier that asserts later
(`_select_eager_blocks_to_store`, manager.py:810).

TARGET AND HOUSE RULES
----------------------
`vllm/v1/simple_kv_offload/manager.py` / `SimpleCPUOffloadScheduler`, resolved
only through `sys.modules` (same discipline as `store_hook` and
`engine_side._engine_core_class`) and never imported. Stdlib only. The wrapper
is installed while guard mode, the trace ring OR the lookup ledger is armed (one
set of wrappers, three consumers, and `wrapper_customers()` is the question every
removal must ask first), so the default state runs the fork's methods unwrapped.
Everything is `Exception`-contained and counted through `snapshot._error`; the
capture path itself cannot raise.
"""

from __future__ import annotations

import collections
import sys
import time
import traceback

from . import lookup_ledger, snapshot, trace_ring

logger = None  # set by engine_side.attach_logger-style wiring; None stays quiet

TARGET_MODULE = "vllm.v1.simple_kv_offload.manager"
TARGET_CLASS = "SimpleCPUOffloadScheduler"

OWNER_ATTR = "__kvtier_dev_guard_owner__"
OWNER_TAG = "kvtier-dev-plugin/guard-1"
ORIGINAL_ATTR = "__kvtier_dev_guard_original__"

# A wrapper installed by ANOTHER module of this same package (today: `backport`,
# which also wraps `get_num_new_matched_tokens` and `build_connector_meta`) is not
# a foreign owner. Refusing it would leave the two most fault-prone entry points
# unguarded while `arm` still reported "guard mode ON: exceptions are captured and
# declined" - a payload that is worse than a refusal, because it is confidently
# wrong. A sibling wrapper is therefore wrapped ON TOP OF, never replaced, and the
# report says which methods are nested so the composition is visible.
SIBLING_PACKAGE = (__package__ or "kvtier_dev_plugin") + "."

# The payload's wire version, as every other verb module reports. A reader that
# pins `version` must be able to compare it across verbs, so they all start at 1
# and only ever go up.
PROTOCOL_VERSION = 1

MAX_CAPTURES = 8
MAX_TRACEBACK_CHARS = 4000
MAX_SECTION_TRACEBACK_CHARS = 480
MAX_MESSAGE_CHARS = 300
MAX_REASON_CHARS = 200
MAX_IDS = 16
MAX_LIST_ITEMS = 8
MAX_GROUPS = 64
MAX_STRING_CHARS = 96
MAX_CAPTURES_RETURNED = 4

# The tier's scheduler-facing entry points, and what a DECLINE means for each.
# Every one of these is called from the scheduler, not from a worker: see
# `vllm/v1/core/sched/scheduler.py:1228` (update_state_after_alloc), `:1511`
# (build_connector_meta), `:2344` (take_events), and the connector forwarders in
# `simple_cpu_offload_connector.py:242-297`.
#
# A value of None means "no safe decline is known": capture, then re-raise.
# A callable receives the ORIGINAL function (to resolve names from its
# `__globals__`, the `store_hook._admissions` idiom) and returns a fresh value,
# so two declines can never share a mutable object.
def _decline_no_tokens(original):  # noqa: ARG001 - the signature contract
    """(0, False): no external tokens, not async. The scheduler computes the
    request normally (manager.py:320 returns the same shape when it hits
    nothing)."""
    return 0, False


def _decline_none(original):  # noqa: ARG001
    return None


def _decline_no_store_specs(original):  # noqa: ARG001
    """([], [], [], None) - nothing to park this step. Fresh lists per call."""
    return [], [], [], None


def _decline_empty_meta(original):
    """An empty `SimpleCPUOffloadMetadata`: every field defaults to "no work
    this step" (metadata.py:16-38, `load_event`/`store_event` = INVALID_JOB_ID).
    The class is resolved from the ORIGINAL's own globals, so a re-import of the
    fork cannot leave us building a stale object; if the fork renamed it we have
    no decline, and the exception propagates as it would without us."""
    globals_dict = getattr(original, "__globals__", None)
    factory = globals_dict.get("SimpleCPUOffloadMetadata") if isinstance(
        globals_dict, dict) else None
    if not callable(factory):
        return _NO_DECLINE
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - a metadata type we cannot build
        snapshot._error(exc, "guard decline metadata")
        return _NO_DECLINE


def _decline_finished(original):  # noqa: ARG001
    """(False, None) - EXACTLY what manager.py:1152 always returns anyway."""
    return False, None


def _decline_no_events(original):  # noqa: ARG001
    return ()


def _decline_reset_refused(original):  # noqa: ARG001
    """False = "the cache was not reset" (base.py:748-760 documents bool)."""
    return False


# The `op` values the guard verb accepts. `arm`/`disarm` are the only ones that
# change anything, and the HTTP layer requires the token for every request to
# this verb - a route that can change dev state is token-gated in all its shapes.
GUARD_OPS = ("status", "arm", "disarm", "captures")
MUTATING_GUARD_OPS = ("arm", "disarm")


_NO_DECLINE = object()

DECINABLE = {
    "get_num_new_matched_tokens": _decline_no_tokens,
    "update_state_after_alloc": _decline_none,
    "build_connector_meta": _decline_empty_meta,
    "prepare_store_specs": _decline_no_store_specs,
    "request_finished": _decline_finished,
    "request_finished_all_groups": _decline_finished,
    "take_events": _decline_no_events,
    "reset": _decline_reset_refused,
}
# Wrapped, but capture-and-re-raise: a decline would be worse than the fault.
CAPTURE_ONLY = ("update_connector_output",)
WRAPPED_METHODS = tuple(DECINABLE) + CAPTURE_ONLY

_state: dict = {
    "armed": False,
    "arms": 0,
    "disarms": 0,
    "installs": 0,
    "reinstalls": 0,
    "uninstalls": 0,
    "installed": False,
    "class": None,
    "skips": 0,
    "skip_reason": None,
    "skip_detail": [],
    # Which methods the guard currently sits ON TOP OF another of this package's
    # wrappers for (a sibling module, e.g. `backport`). A reader must be able to
    # see the composition long after the arm that created it.
    "nested": [],
    "calls": 0,
    "caught": 0,
    "declined": 0,
    "re_raised": 0,
    "capture_errors": 0,
    "captures_dropped": 0,
    "captures": collections.deque(maxlen=MAX_CAPTURES),
    "last_reason": None,
    "arm_reason": None,
}


def attach_logger(a_logger) -> None:
    global logger
    logger = a_logger


def reset_state_for_tests() -> None:
    """Process-wide state, like `snapshot._state`."""
    _state.update({"armed": False, "arms": 0, "disarms": 0, "installs": 0,
                   "reinstalls": 0, "uninstalls": 0, "installed": False,
                   "class": None, "skips": 0, "skip_reason": None,
                   "skip_detail": [], "calls": 0,
                   "caught": 0, "declined": 0, "re_raised": 0,
                   "capture_errors": 0, "captures_dropped": 0,
                   "last_reason": None, "arm_reason": None})
    _state["captures"] = collections.deque(maxlen=MAX_CAPTURES)


def is_armed() -> bool:
    """THE answer every payload must carry: is this engine in guard mode?"""
    return bool(_state["armed"])


def fault_count() -> int:
    """How many tier exceptions the guard has caught and declined. Never raises.

    The public accessor for the counter the BOOT line prints: "0 faults after
    boot" is the only evidence that an auto-guarded boot is silently declining
    something, so the boot line and the snapshot both have to read it without
    reaching into this module's state.
    """
    try:
        return int(_state.get("caught") or 0)
    except Exception:  # noqa: BLE001 - an unreadable counter is 0, never a raise
        return 0


def wrapper_customers() -> list:
    """Which armed facilities still need the class wrappers. Never raises.

    THE SHARED-PREDICATE RULE. The guard wrappers are ONE set of class attributes
    serving THREE consumers: guard mode itself (capture + decline), the trace ring
    (a record per tier entry point) and the lookup ledger (the offer/adopt pair).
    Every uninstaller must ask this question instead of eyeballing one other
    consumer, or a disarm of one silently blinds the others - `set_guard_mode
    (False)` used to check only `trace_ring`, so a ledger armed on its own lost
    its recording the moment anyone toggled guard mode. Called AFTER the caller
    has cleared its own armed flag, so a consumer that just disarmed is absent.
    """
    out: list = []
    if bool(_state["armed"]):
        out.append("guard mode")
    for name, probe in (("the trace ring", trace_ring.is_armed),
                        ("the lookup ledger", lookup_ledger.is_armed)):
        try:
            if probe():
                out.append(name)
        except Exception:  # noqa: BLE001, S112 - a missing consumer counts as gone
            continue
    return out


def manager_class():
    """The offload scheduler class if this process already imported it."""
    module = sys.modules.get(TARGET_MODULE)
    if module is None:
        return None
    return getattr(module, TARGET_CLASS, None)


def sibling_layer(fn):
    """Which layer of THIS package `fn` is, or None if it is not one of ours.

    THE LAYER-AWARE OWNERSHIP QUESTION. "Is this attribute ours?" cannot be asked
    only about guard mode's own marker, and cannot be answered from `__module__`
    either: `backport` REPLACES a method with a compiled copy that carries the
    FORK's `__module__` (so a copy looks like the fork's own body, and in a test
    fixture built outside the fork's module it looks like a stranger's). The
    markers are the truth, so ask the module that owns the marker vocabulary -
    `backport`, lazily, which is the same direction its own `_layer_kinds()`
    imports this module for. Never raises: an unreadable attribute is "not ours",
    which is the safe answer because it leads to a refusal, not to an overwrite.
    """
    if fn is None:
        return None
    try:
        from . import (
            backport,  # noqa: PLC0415 - lazy: backport imports us lazily
            disk_restore,  # noqa: PLC0415 - lazy, for the same reason
        )
        inner = backport._function_of(fn)
        for attr, tag, kind in ((backport.SWITCH_ATTR, backport.SWITCH_TAG, "switch"),
                                (backport.OWNER_ATTR, backport.OWNER_TAG, "backport"),
                                (disk_restore.OWNER_ATTR, disk_restore.OWNER_TAG,
                                 disk_restore.LAYER_KIND)):
            if getattr(inner, attr, None) == tag:
                return kind
    except Exception as exc:  # noqa: BLE001 - "not ours" is the safe answer
        snapshot._error(exc, "guard sibling layer")
    return None


def manager_of(engine):
    """The live scheduler-side manager, or None. Reads only."""
    scheduler = getattr(engine, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    return getattr(connector, "scheduler_manager", None)


# --- the context snapshot ---------------------------------------------------

def _clip(value, chars: int):
    if isinstance(value, str) and len(value) > chars:
        return value[:chars] + "..."
    return value


def _ints(container, cap: int) -> dict:
    """{"len": n, "head": [...], "dropped": k} for a sized container of ids."""
    try:
        total = len(container)
        head = list(container)[:cap]
    except TypeError:
        return {"kind": type(container).__name__}
    return {"len": total, "head": head, "dropped": max(0, total - cap)}


def _arg_value(arg, depth: int = 0):
    """A bounded, JSON-safe rendering of one call argument. Never raises.

    Duck-typed on the fork's own attribute names; no isinstance on a fork class,
    because this module must not import one. A `Request` arrives as its id and
    its counters (that is what identifies it without copying a token list), a
    `KVCacheBlocks` arrives as per-group counts plus capped id heads (the
    "block ids involved" the operator actually needs), everything else as its
    class and length.
    """
    if arg is None or isinstance(arg, (bool, int, float)):
        return arg
    if isinstance(arg, str):
        return _clip(arg, MAX_STRING_CHARS)
    if isinstance(arg, (bytes, bytearray)):
        return "0x" + bytes(arg)[:16].hex()
    if isinstance(arg, dict):
        return {"kind": "dict", "len": len(arg)}
    if isinstance(arg, (list, tuple, set, frozenset)):
        head = [_arg_value(item, depth + 1) for item in list(arg)[:MAX_LIST_ITEMS]]
        if depth >= 1:
            head = [item if not isinstance(item, (list, tuple))
                    else {"len": len(item)} for item in head]
        return {"kind": type(arg).__name__, "len": len(arg), "head": head,
                "dropped": max(0, len(arg) - MAX_LIST_ITEMS)}
    request_id = getattr(arg, "request_id", None)
    if isinstance(request_id, str):
        hashes = getattr(arg, "block_hashes", None)
        return {"kind": type(arg).__name__, "request_id": _clip(request_id, 64),
                "num_computed_tokens": getattr(arg, "num_computed_tokens", None),
                "num_tokens": getattr(arg, "num_tokens", None),
                "block_hashes": (len(hashes) if hasattr(hashes, "__len__")
                                 else None)}
    get_block_ids = getattr(arg, "get_block_ids", None)
    if callable(get_block_ids):
        try:
            per_group = get_block_ids()
        except Exception as exc:  # noqa: BLE001 - the arg lied, that is data
            snapshot._error(exc, "guard capture get_block_ids")
            return {"kind": type(arg).__name__, "block_ids": "unavailable"}
        return {"kind": type(arg).__name__,
                "block_ids_by_group": [_ints(ids, MAX_IDS) for ids in per_group]}
    try:
        size = len(arg)
    except TypeError:
        size = None
    out = {"kind": type(arg).__name__}
    if size is not None:
        out["len"] = size
    return out


def _geometry_operands(manager) -> dict:
    """The operands of the arithmetic the fork asserts on, read RIGHT NOW.

    `resolve_block_hashes(request.block_hashes, hash_block_size, group_size)`
    asserts `group_size % hash_block_size == 0`
    (vllm/v1/core/kv_cache_utils.py:2888); `group_size` is
    `manager.group_block_sizes[g]` at manager.py:838-844. When the fault IS that
    assert, this is the arithmetic, per group, at the moment it failed.
    """
    from . import kvconfig  # noqa: PLC0415 - guard_mode is imported by kvconfig
    sizes = getattr(manager, "group_block_sizes", None)
    hash_block_size = getattr(manager, "hash_block_size", None)
    per_group = []
    for index, size in enumerate(list(sizes or [])[:MAX_GROUPS]):
        check = kvconfig.divisibility(size, hash_block_size)
        per_group.append({"group": index, "group_block_size": check["group_block_size"],
                          "remainder": check["remainder"],
                          "would_fail": check["would_fail"]})
    failing = [entry["group"] for entry in per_group if entry["would_fail"]]
    return {"hash_block_size": hash_block_size,
            "block_size": getattr(manager, "block_size", None),
            "group_block_sizes": list(sizes or [])[:MAX_GROUPS],
            "per_group": per_group,
            "groups_failing_divisibility": failing}


def manager_context(manager) -> dict:
    """The tier's own sizes and counters, bounded. Reads only, never raises."""
    if manager is None:
        return {"present": False}
    pool = getattr(manager, "cpu_block_pool", None)
    free = None
    getter = getattr(pool, "get_num_free_blocks", None) if pool is not None else None
    if callable(getter):
        try:
            free = getter()
        except Exception as exc:  # noqa: BLE001
            snapshot._error(exc, "guard capture cpu free")
    pending = None
    has_pending = getattr(manager, "has_pending_stores", None)
    if callable(has_pending):
        try:
            pending = bool(has_pending())
        except Exception as exc:  # noqa: BLE001
            snapshot._error(exc, "guard capture has_pending_stores")
    return {
        "present": True,
        "class": type(manager).__name__,
        "mode": "lazy" if getattr(manager, "_lazy_mode", False) else "eager",
        "block_size": getattr(manager, "block_size", None),
        "hash_block_size": getattr(manager, "hash_block_size", None),
        "fa_block_size": getattr(manager, "fa_block_size", None),
        "fa_group_index": getattr(manager, "fa_gidx", None),
        "target_free_gpu_blocks": getattr(manager, "_target_free", None),
        "num_cpu_blocks": getattr(manager, "num_cpu_blocks", None),
        "cpu_free_blocks": free,
        "pending_stores": pending,
        "store_event_counter": getattr(manager, "_store_event_counter", None),
        "load_event_counter": getattr(manager, "_load_event_counter", None),
        "reqs_to_store": _len_or_none(getattr(manager, "_reqs_to_store", None)),
        "reqs_to_load": _len_or_none(getattr(manager, "_reqs_to_load", None)),
        "pending_cpu_hits": _len_or_none(getattr(manager, "_pending_cpu_hits", None)),
        "in_flight_store_blocks": _len_or_none(
            getattr(manager, "_in_flight_store_gpu_blocks", None)),
        "geometry": _geometry_operands(manager),
    }


def _len_or_none(container):
    try:
        return len(container)
    except TypeError:
        return None


def capture(exc: BaseException, method: str, manager, args, kwargs,
            declined) -> dict:
    """Build the record for one fault. Contained twice over: a capture that
    cannot be built still returns a dict naming the failure, because the whole
    point is that the operator learns something."""
    try:
        text = traceback.format_exc()
    except Exception:  # noqa: BLE001 - a broken traceback must not lose the fault
        text = "<traceback unavailable>"
        snapshot._error(exc, "guard capture traceback")
    return {                      # built directly: there is nothing to post-process
        "fault_no": int(_state["caught"]) + 1,
        "mono_ns": time.monotonic_ns(),
        "method": method,
        "exception_type": type(exc).__name__,
        "message": _clip(str(exc), MAX_MESSAGE_CHARS),
        "traceback": _clip(text, MAX_TRACEBACK_CHARS),
        "declined": bool(declined),
        "decline": ("declined" if declined else "re-raised"),
        "manager": manager_context(manager),
        "args": [_arg_value(argument) for argument in (args or ())][:MAX_LIST_ITEMS],
        "kwargs": {name: _arg_value(value)
                   for name, value in list((kwargs or {}).items())[:MAX_LIST_ITEMS]},
        # A placeholder: `_capture` refines it to the sequence number of the ring
        # entry this fault produced, once that entry exists.
        "trace_seq": trace_ring.counters()["seq"],
    }


def _remember(record: dict) -> None:
    captures = _state["captures"]
    if len(captures) == captures.maxlen:
        _state["captures_dropped"] += 1
    captures.append(record)


def _capture(exc: BaseException, method: str, manager, args, kwargs,
             declined) -> None:
    """Capture as data, then record it in the ring. Never raises outward."""
    try:
        record = capture(exc, method, manager, args, kwargs, declined)
        _remember(record)
        _state["last_reason"] = (f"{record['exception_type']}: "
                                 f"{record['message']}"[:MAX_REASON_CHARS])
        trace_ring.record("exception", group=None, n1=None, n2=None,
                          text=f"{method}: {record['exception_type']}")
        # Now that the ring entry EXISTS, point the capture at it: `trace_seq` is
        # "the record this fault produced", so an operator reading the ring and
        # the capture side by side lands on the same slot both ways.
        record["trace_seq"] = trace_ring.counters()["seq"]
        if logger is not None:
            logger.warning("[KVTIER_DEV] guard captured %s in %s -> %s: %s",
                           type(exc).__name__, method,
                           "declined" if declined else "re-raised",
                           record["message"])
    except Exception as exc2:  # noqa: BLE001 - the capture must never be the fault
        _state["capture_errors"] += 1
        snapshot._error(exc2, "guard capture")


# --- what to record on success (the trace ring's per-step facts) ------------

def _record_success(method: str, result, args, manager=None) -> None:
    """A few records per step, from the values the call already returned.

    TWO READERS of the same already-live values, and neither is a reason to fit a
    second wrapper over the same method: the trace ring (what happened, in order)
    and the LOOKUP LEDGER (the `get_num_new_matched_tokens` offer paired with the
    `update_state_after_alloc` adopt, joined per request id). The ledger hook runs
    FIRST and outside the trace `try` below, so a ledger fault cannot be charged to
    the trace path or hide a trace record; it is contained and counted in its own
    module. `manager` is the tier object the call was made on - the ledger only
    ever asks it for `block_size`/`hash_block_size`.
    """
    if method in lookup_ledger.LEDGER_METHODS and lookup_ledger.is_armed():
        lookup_ledger.record_from_call(method, args, result, manager)
    if not trace_ring.is_armed():
        return
    try:
        if method == "build_connector_meta":
            trace_ring.record(
                "step:build_meta",
                n1=getattr(result, "load_event", None),
                n2=getattr(result, "store_event", None),
                text=(f"loads={len(getattr(result, 'load_gpu_blocks', ()) or ())}"
                      f" stores={len(getattr(result, 'store_gpu_blocks', ()) or ())}"))
        elif method == "prepare_store_specs" and isinstance(result, tuple):
            gpu = result[0] if len(result) > 0 else ()
            cpu = result[1] if len(result) > 1 else ()
            trace_ring.record("step:store_specs",
                              n1=len(gpu or ()), n2=len(cpu or ()))
        elif method == "get_num_new_matched_tokens" and isinstance(result, tuple):
            tokens = result[0] if len(result) > 0 else None
            request = args[0] if args else None
            trace_ring.record("tier:load_offer", n1=tokens,
                              n2=1 if (isinstance(result, tuple) and len(result) > 1
                                       and result[1]) else 0,
                              text=_clip(str(getattr(request, "request_id", "")), 64))
        elif method == "update_state_after_alloc":
            request = args[0] if args else None
            trace_ring.record(
                "tier:load_accepted",
                n1=args[2] if len(args) > 2 else None,
                text=_clip(str(getattr(request, "request_id", "")), 64))
        elif method in ("request_finished", "request_finished_all_groups"):
            request = args[0] if args else None
            trace_ring.record(
                "tier:finish",
                text=_clip(str(getattr(request, "request_id", "")), 64))
        elif method == "update_connector_output":
            trace_ring.record("tier:completion_report")
        elif method == "reset":
            trace_ring.record("tier:reset", n1=1 if result else 0)
    except Exception as exc:  # noqa: BLE001 - tracing never reaches the tier
        _state["capture_errors"] += 1
        snapshot._error(exc, "guard trace success")


# --- the wrapper ------------------------------------------------------------

def _make_wrapper(method: str, original):
    """The replacement for one tier entry point.

    Two shapes, so the disarmed-but-tracing case cannot hide anything: with the
    guard DISARMED the original is called OUTSIDE any try/except of ours that
    could swallow, and any capture we take is followed by a re-raise.
    """
    decline = DECINABLE.get(method)

    def _wrapped(self, *args, **kwargs):
        _state["calls"] += 1
        if not _state["armed"]:
            result = original(self, *args, **kwargs)
            _record_success(method, result, args, self)
            return result
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - never BaseException, see above
            _state["caught"] += 1
            safe_decline = None
            if decline is not None:
                try:
                    safe_decline = decline(original)
                except Exception as exc2:  # noqa: BLE001 - a decline we cannot build
                    snapshot._error(exc2, "guard decline builder")
                    safe_decline = _NO_DECLINE
            if decline is None or safe_decline is _NO_DECLINE:
                _capture(exc, method, self, args, kwargs, declined=False)
                _state["re_raised"] += 1
                raise
            _capture(exc, method, self, args, kwargs, declined=True)
            _state["declined"] += 1
            return safe_decline
        _record_success(method, result, args, self)
        return result

    setattr(_wrapped, OWNER_ATTR, OWNER_TAG)
    setattr(_wrapped, ORIGINAL_ATTR, original)
    _wrapped.__name__ = method
    _wrapped.__doc__ = (
        "kvtier-dev-plugin GUARD MODE (development only): the tier's own "
        f"{method}, with an armed-mode exception captured as data and turned "
        "into a declined result. BaseException always propagates. Disarm before "
        "any production boot.")
    return _wrapped


def _refuse(reason: str) -> None:
    """One line per distinct reason (the snapshot polls this path)."""
    if _state["skip_reason"] == reason:
        return
    _state["skips"] += 1
    _state["skip_reason"] = reason
    if logger is not None:
        logger.info("kvtier-dev-plugin: guard-mode wrappers NOT installed: %s",
                    reason)


def install(cls) -> dict:
    """Wrap the tier entry points on `cls`. Idempotent, never raises.

    The same ownership discipline as `store_hook.install`: ours is
    re-installable (re-binding the FORK's function we stored, so a reload loop
    cannot nest wrappers), a foreign owner marker is refused, and an unmarked
    name is only accepted when it is defined in the fork's own module.

    A PARTIAL install still says WHY each name was refused: `skip_detail` pairs the
    method with its reason, so "guard mode is armed but `take_events` is not
    wrapped" is answerable from one payload instead of from a guess - and
    `skip_reason` is only cleared when NOTHING was skipped.

    A function this package installed itself (a sibling wrapper such as the
    PR #56810 backport's) is composed with, not refused: the guard goes on top of
    it and `nested` names those methods. A marker or a definition owned by ANYONE
    else is still refused outright - that is the rule that stops this plugin from
    overwriting a third party's patch.
    """
    report = {"installed": False, "attached": [], "replaced": [], "skipped": [],
              "skip_detail": [], "nested": [], "class": None, "reason": None}

    def refuse(method: str, reason: str) -> None:
        report["skipped"].append(method)
        report["skip_detail"].append({"method": method,
                                      "reason": reason[:MAX_REASON_CHARS]})
        _refuse(reason)

    try:
        if cls is None:
            _refuse(f"no {TARGET_CLASS} is imported in this process "
                    f"({TARGET_MODULE} absent)")
            report["reason"] = _state["skip_reason"]
            return report
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        for method in WRAPPED_METHODS:
            original = getattr(cls, method, None)
            if original is None:
                refuse(method, f"{report['class']} has no {method}: the fork "
                               f"renamed it, refusing to invent one")
                continue
            if not callable(original):
                refuse(method, f"{report['class']}.{method} is not callable "
                               f"({type(original).__name__})")
                continue
            owner = getattr(original, OWNER_ATTR, None)
            if owner is not None and owner != OWNER_TAG:
                refuse(method, f"{report['class']}.{method} carries another "
                               f"plugin's owner marker, refusing to overwrite")
                continue
            replaced = owner == OWNER_TAG
            under = None if replaced else sibling_layer(original)
            if not replaced and under is None:
                home = getattr(original, "__module__", None)
                if home != TARGET_MODULE and not str(home or "").startswith(
                        SIBLING_PACKAGE):
                    refuse(method, f"{report['class']}.{method} is neither ours "
                                   f"nor the fork's own: defined in {home!r}, not "
                                   f"{TARGET_MODULE!r}, refusing to overwrite")
                    continue
                if home != TARGET_MODULE:
                    # Ours, from a sibling module: compose on top of it, and say so.
                    report["nested"].append(method)
            elif under is not None:
                # A layer of THIS package that is not ours - `backport`'s replaced
                # copy, or one of its kill-switch wrappers. Composed with, never
                # refused (that would leave the most fault-prone entry points
                # unguarded behind an "armed" payload) and never replaced. Reported
                # under `nested` like any other composition, because `__module__`
                # cannot be trusted to identify our own copies here: the compiled
                # replacements carry the FORK's module name on purpose.
                report["nested"].append(method)
            if replaced:
                original = getattr(original, ORIGINAL_ATTR, original)
                report["replaced"].append(method)
            else:
                report["attached"].append(method)
            setattr(cls, method, _make_wrapper(method, original))
        report["installed"] = bool(report["attached"] or report["replaced"])
        if report["attached"]:
            _state["installs"] += 1
            _state["installed"] = True
            _state["class"] = report["class"]
            _state["nested"] = list(report["nested"])
            if not report["skipped"]:
                _state["skip_reason"] = None
            _state["skip_detail"] = list(report["skip_detail"])
            if logger is not None:
                logger.warning("kvtier-dev-plugin: GUARD MODE wrappers installed "
                               "on %s (%d methods). This is a DEVELOPMENT mode "
                               "that can mask real bugs: disarm before any "
                               "production boot.", report["class"],
                               len(report["attached"]))
        if report["replaced"]:
            _state["reinstalls"] += 1
            # A re-install recomputes the composition: the methods we already
            # owned keep whatever they were nested under, so merge, do not drop -
            # and keep the fork's own method order, never an arbitrary sort, so a
            # reader sees the same list from the report and from the snapshot.
            combined = set(_state.get("nested") or []) | set(report["nested"])
            _state["nested"] = [method for method in WRAPPED_METHODS
                                if method in combined]
        if not report["installed"]:
            report["reason"] = _state["skip_reason"]
            _state["skip_detail"] = list(report["skip_detail"])
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        snapshot._error(exc, "guard install")
        report["reason"] = f"install raised: {type(exc).__name__}"[:MAX_REASON_CHARS]
        return report


def uninstall(cls) -> dict:
    """Restore the fork's own methods. Only called when nothing needs them.

    Restores the stored ORIGINAL for every method still marked ours; a method we
    do not own is left exactly where it is. Cheap (a handful of setattrs) and
    deliberate: the whole point of installing on demand is that the DEFAULT
    engine runs the fork's unwrapped methods.
    """
    report = {"uninstalled": False, "restored": [], "left": [], "class": None,
              "reason": None}
    try:
        if cls is None:
            report["reason"] = "no class to restore"
            return report
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        for method in WRAPPED_METHODS:
            current = getattr(cls, method, None)
            if getattr(current, OWNER_ATTR, None) != OWNER_TAG:
                report["left"].append(method)
                continue
            setattr(cls, method, getattr(current, ORIGINAL_ATTR))
            report["restored"].append(method)
        report["uninstalled"] = bool(report["restored"])
        if report["restored"]:
            _state["uninstalls"] += 1
            _state["installed"] = False
            _state["nested"] = []
            if logger is not None:
                logger.info("kvtier-dev-plugin: guard-mode wrappers removed from "
                            "%s (%d methods restored)", report["class"],
                            len(report["restored"]))
        return report
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "guard uninstall")
        report["reason"] = f"uninstall raised: {type(exc).__name__}"[:MAX_REASON_CHARS]
        return report


def ensure_installed(engine=None) -> dict:
    """Install from the live manager's class when there is one. Never raises."""
    manager = None
    try:
        manager = manager_of(engine)
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "guard manager_of")
    cls = type(manager) if manager is not None else manager_class()
    return install(cls)


# --- the arming surface -----------------------------------------------------

def set_guard_mode(armed: bool, engine=None) -> dict:
    """Arm or disarm. THE one mutation this module makes, and it is verb-driven.

    Arming installs the wrappers if a tier exists, and says plainly when none
    does - "armed with nothing to guard" is a state an operator must be able to
    read, not one that quietly does nothing.
    """
    wanted = bool(armed)
    report = {"armed": wanted, "changed": wanted != bool(_state["armed"]),
              "install": None, "uninstall": None, "reason": None}
    if wanted:
        _state["armed"] = True
        _state["arms"] += 1
        report["install"] = ensure_installed(engine)
        if report["install"].get("installed"):
            report["reason"] = ("guard mode ON: tier exceptions are captured as "
                                "data and declined. DEVELOPMENT ONLY.")
        else:
            report["reason"] = ("guard mode ON but NOTHING is wrapped yet: no "
                                f"{TARGET_CLASS} exists in this process (the "
                                "engine is running with no connector, or the "
                                "fork renamed the class). Armed state persists; "
                                "the wrappers land as soon as a manager exists.")
    else:
        _state["armed"] = False
        _state["disarms"] += 1
        customers = wrapper_customers()
        if not customers:
            manager = None
            try:
                manager = manager_of(engine)
            except Exception as exc:  # noqa: BLE001
                snapshot._error(exc, "guard disarm manager_of")
            cls = type(manager) if manager is not None else manager_class()
            report["uninstall"] = uninstall(cls)
            report["reason"] = ("guard mode OFF; wrappers removed (nothing else "
                                "consumes them: no trace ring, no lookup ledger)")
        else:
            report["reason"] = ("guard mode OFF; the wrappers STAY because "
                                + " and ".join(customers)
                                + " still need them. Tracing captures faults as "
                                "data and re-raises them: nothing is swallowed.")
    _state["arm_reason"] = report["reason"]
    return report


# --- observability ----------------------------------------------------------

def _capture_preview(capture_record: dict, chars: int) -> dict:
    """A bounded view of one capture, for the per-poll snapshot payload."""
    text = capture_record.get("traceback") or ""
    return {
        "seq": capture_record.get("seq"),
        "trace_seq": capture_record.get("trace_seq"),
        "method": capture_record.get("method"),
        "exception_type": capture_record.get("exception_type"),
        "message": capture_record.get("message"),
        "decline": capture_record.get("decline"),
        "traceback_chars": len(text),
        "traceback_head": text[:chars],
        "manager_present": bool(
            (capture_record.get("manager") or {}).get("present")),
        "groups_failing_divisibility": (
            (capture_record.get("manager") or {}).get("geometry") or {}
        ).get("groups_failing_divisibility"),
        "arg_count": len(capture_record.get("args") or []),
    }


def section(include_captures: int = 0,
            traceback_chars: int = MAX_SECTION_TRACEBACK_CHARS) -> dict:
    """The snapshot section. `include_captures` > 0 returns full captures, capped.

    The per-poll payload deliberately carries PREVIEWS (a bounded traceback head
    per capture): eight full tracebacks in a payload a dashboard polls every few
    seconds is a megabyte of noise. `POST /plugins/kvtier/guard {"op":
    "captures"}` asks for the full text, bounded by `MAX_CAPTURES_RETURNED`.
    """
    captures = list(_state["captures"])
    wanted = include_captures if isinstance(include_captures, int) and not \
        isinstance(include_captures, bool) else 0
    wanted = max(0, min(MAX_CAPTURES_RETURNED, wanted))
    out = {
        "guard_mode": bool(_state["armed"]),
        "development_only": True,
        "warning": ("guard mode can mask real bugs: a declined store/load is "
                    "silent. Disarm before any production boot."),
        "arms": int(_state["arms"]),
        "disarms": int(_state["disarms"]),
        "arm_reason": _state["arm_reason"],
        "installed": bool(_state["installed"]),
        "class": _state["class"],
        "module": TARGET_MODULE,
        "methods": list(WRAPPED_METHODS),
        "declinable": sorted(DECINABLE),
        "capture_and_reraise": list(CAPTURE_ONLY),
        "installs": int(_state["installs"]),
        "reinstalls": int(_state["reinstalls"]),
        "uninstalls": int(_state["uninstalls"]),
        "skips": int(_state["skips"]),
        "skip_reason": _state["skip_reason"],
        "skip_detail": list(_state.get("skip_detail") or []),
        "nested": list(_state.get("nested") or []),
        "calls": int(_state["calls"]),
        "wrapper_customers": wrapper_customers(),
        "caught": int(_state["caught"]),
        "declined": int(_state["declined"]),
        "re_raised": int(_state["re_raised"]),
        "capture_errors": int(_state["capture_errors"]),
        "captures_dropped": int(_state["captures_dropped"]),
        "captures_kept": len(captures),
        "captures_max": MAX_CAPTURES,
        "last_reason": _state["last_reason"],
        "captures_preview": [_capture_preview(record, MAX_SECTION_TRACEBACK_CHARS)
                             for record in captures[-3:]],
        "error": None,
    }
    if wanted:
        out["captures"] = [_capture_preview(record, traceback_chars)
                           for record in captures[-wanted:]]
    return out


def captures(limit: int = MAX_CAPTURES_RETURNED,
             traceback_chars: int = MAX_TRACEBACK_CHARS) -> dict:
    """The full capture text, on explicit request only. Bounded, always."""
    wanted = limit if isinstance(limit, int) and not isinstance(limit, bool) else 1
    wanted = max(1, min(MAX_CAPTURES_RETURNED, wanted))
    chars = (traceback_chars if isinstance(traceback_chars, int)
             and not isinstance(traceback_chars, bool) else MAX_TRACEBACK_CHARS)
    chars = max(0, min(MAX_TRACEBACK_CHARS, chars))
    kept = list(_state["captures"])[-wanted:]
    out = []
    for record in kept:
        entry = dict(record)
        text = entry.get("traceback") or ""
        entry["traceback_chars"] = len(text)
        entry["traceback"] = text[:chars]
        entry["traceback_truncated"] = len(text) > chars
        out.append(entry)
    return {"captures": out, "requested": wanted, "returned": len(out),
            "kept": len(_state["captures"]),
            "dropped": int(_state["captures_dropped"]),
            "guard_mode": bool(_state["armed"])}


def guard_for_engine(engine, args=None) -> dict:
    """THE engine-side handler body for `guard`. Never raises an Exception.

    `op` selects the shape: `status` (the default, a pure read), `captures` (the
    full bounded capture text), `arm`, `disarm`. `{"armed": true|false}` is
    accepted as the ergonomic spelling of arm/disarm, because that is the state an
    operator is actually thinking in.

    Both mutating ops are token-gated by the HTTP layer - loopback included - and
    the resulting state is echoed in `status` and in every verb payload as
    `guard_mode`, because the failure mode this plugin must never produce is a
    guarded dev engine being mistaken for a production one.
    """
    snapshot._state["calls"] += 1
    started = time.perf_counter()
    arguments = args if isinstance(args, dict) else {}
    try:
        op = arguments.get("op")
        if isinstance(op, str):
            op = op.strip().lower()
        elif "armed" in arguments:
            op = "arm" if arguments.get("armed") else "disarm"
        else:
            op = "status"
        if op not in GUARD_OPS:
            payload: dict = {"op": None, "error": (
                f"unknown op {arguments.get('op')!r}; expected one of "
                f"{list(GUARD_OPS)}")}
        elif op == "arm":
            payload = {"op": "arm", **set_guard_mode(True, engine)}
        elif op == "disarm":
            payload = {"op": "disarm", **set_guard_mode(False, engine)}
        elif op == "captures":
            payload = {"op": "captures",
                       **captures(arguments.get("limit"),
                                  arguments.get("traceback_chars"))}
        else:
            payload = {"op": "status", "guard": section()}
        unknown = sorted(key for key in arguments
                         if key not in ("op", "armed", "limit", "traceback_chars"))
        payload["guard_mode"] = is_armed()
        payload["development_only"] = True
        payload["version"] = PROTOCOL_VERSION
        payload["epoch"] = str(snapshot.epoch())
        payload["ts"] = round(time.time(), 6)
        if unknown:
            payload["unknown_args"] = unknown
        payload["build_us"] = int((time.perf_counter() - started) * 1e6)
        payload["self"] = snapshot.error_counters()
        return snapshot.json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        snapshot._error(exc, "guard verb")
        return {"version": PROTOCOL_VERSION, "verb": "guard",
                "epoch": str(snapshot.epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "guard_mode": is_armed(),
                "counters": snapshot.error_counters()}
