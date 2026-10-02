"""The runtime policy hook on the tier's store-decision path. Stdlib only.

WHAT IT PATCHES (read out of the read-only fork at HEAD 10b8e07)
----------------------------------------------------------------
`vllm/v1/simple_kv_offload/manager.py:932`

    def _classify_store_candidate(self, gpu_block: "KVCacheBlock") -> _StoreAdmission:
        if gpu_block.is_null:                  return _StoreAdmission.NULL_BLOCK
        if gpu_block.block_hash is None:       return _StoreAdmission.NOT_HASHED
        if gpu_block.block_id in self._in_flight_store_gpu_blocks:
                                               return _StoreAdmission.IN_FLIGHT
        if self.cpu_block_pool.cached_block_hash_to_block.get_one_block(
                gpu_block.block_hash) is not None:
                                               return _StoreAdmission.ALREADY_CACHED
        return _StoreAdmission.READY

One `gpu_block`, one verdict, no I/O. It is the single gate in front of every
host-RAM store the tier makes in **eager** mode, with exactly two call sites:

  * `manager.py:685` in `_prepare_eager_store_specs` (the boundary hand-off
    path): each non-READY member is counted there under its own
    `BoundaryStoreStats` reason and the block is `continue`d;
  * `manager.py:862` in `_select_eager_blocks_to_store`: any non-READY member
    advances the per-group cursor and `continue`s.

So returning a non-READY member is the ONLY way to veto a store that is
expressible at this hook point, and both callers treat it as "do not store
this block" with no other side effect. That is the veto.

WHAT A VETO COSTS, HONESTLY
---------------------------
`_StoreAdmission` (manager.py:104-109) has five members and NONE of them means
"the policy said no". A veto is therefore reported to the engine as
`ALREADY_CACHED` - the one member whose call-site behaviour is a pure skip and
whose counter (`skipped_already_cached`) is a *skip*, not a data-loss `dropped_*`
- and the real count lives in this module's own `vetoes` counter, which the
snapshot publishes. Do not read `skipped_already_cached` as "the tier is
working".

Two things this hook point CANNOT do, and the design does not pretend to:

  * **the drop-on-full branch** (`manager.py:698-701`,
    `if num_free <= 0: stats.dropped_cpu_full += 1`) is a capacity condition
    the caller evaluates AFTER classification returned READY. There is no
    `_StoreAdmission.CPU_FULL`, so no return value from here can produce it,
    and reaching it would mean reimplementing the body of a 150-line loop.
    A veto is charged to `skipped_already_cached`, never to `dropped_cpu_full`.
  * **lazy mode** (`manager.py:566-617`, `_prepare_lazy_store_specs`) never
    calls `_classify_store_candidate` at all: it walks the GPU free queue and
    calls `cpu_pool.get_new_blocks()` directly. In lazy mode this hook is inert
    and the kill switch does NOT stop stores. `governs_current_mode` in the
    snapshot says so instead of leaving it a mystery.

A veto is also a SKIP, not a DEFERRAL. Both call sites advance past the block
(the boundary hand-off is not retried - the fork says so itself at
manager.py:699-701 - and the eager loop advances `num_stored_blocks[g]`), so a
vetoed block is never offered again for that request. Turning the tier off and
on again does not re-offer the blocks skipped in between; it only resumes
offloading what is still being produced.

HOW IT RUNS
-----------
On the EngineCore busy-loop thread, per candidate block, inside the scheduler's
decision. So: no I/O, no logging, no torch, no allocation beyond one small
dict, and any failure whatsoever falls back to the engine's own verdict.
Errors are counted and reported through `snapshot._error`, which logs at most
`snapshot.MAX_ERROR_LOGS` times in the process.
"""

from __future__ import annotations

import logging
import sys

from . import snapshot

logger = logging.getLogger("kvtier_dev_plugin.store_hook")

# The patch target, named as strings and resolved ONLY against an already
# imported module. See `manager_class()`: never an import of our own.
TARGET_MODULE = "vllm.v1.simple_kv_offload.manager"
TARGET_CLASS = "SimpleCPUOffloadScheduler"
TARGET_METHOD = "_classify_store_candidate"

# The member a veto is reported as, and the member we consult the policy for.
# Both are resolved at CALL time out of the original function's `__globals__`,
# because the call sites compare with `is _StoreAdmission.X` looked up as a
# module global: a stale enum object from an older import of this module would
# match nothing and would silently turn every veto into a store.
VETO_MEMBER = "ALREADY_CACHED"
CONSULT_MEMBER = "READY"

# Owner markers, the same idiom as `engine_side.attach_methods`: a name we own
# is re-installable, a name somebody else owns is refused with a log line.
OWNER_ATTR = "__kvtier_dev_store_owner__"
OWNER_TAG = "kvtier-dev-plugin/store-1"
ORIGINAL_ATTR = "__kvtier_dev_store_original__"

# The verdict contract. Anything not listed is UNRECOGNIZED and means "do what
# the engine would have done" - the conservative direction, and the reason the
# kill switch cannot be defeated by a typo. Kept lowercase: we `.strip().lower()`.
VETO_WORDS = frozenset({"veto", "block", "deny", "no-store"})
ALLOW_WORDS = frozenset({"", "allow", "store", "ok", "pass", "ready"})

# Sentinel results of the policy conversation. Module-level singletons, so the
# wrapper's comparisons are identity checks.
_VETO = "veto"
_ALLOW = "allow"
_UNRECOGNIZED = "unrecognized"

# Counters only - the per-call path logs nothing. `enum_resolved` and
# `store_policy_present` are None until the first candidate reaches that line,
# so the snapshot can tell "not yet exercised" apart from "false".
_state = {
    "installs": 0,
    "reinstalls": 0,
    "skips": 0,  # distinct refusal episodes, not polls
    "skip_reason": None,
    "class": None,
    "calls": 0,
    "policy_calls": 0,
    "vetoes": 0,
    "fallbacks": 0,
    "unrecognized": 0,
    "enum_resolved": None,
    "store_policy_present": None,
    "vetoes_at_last_report": 0,
}

# Identity memo: [the globals dict we resolved from, READY, ALREADY_CACHED].
# Re-resolved the moment the dict object changes, which is the only way a
# re-import of the fork's module can differ from what we cached.
_admission_cache = [None, None, None]


def counters() -> dict:
    """A copy of the counters. Cheap, and safe to hand to the wire."""
    return dict(_state)


def reset_counters_for_tests() -> None:
    """Test hook: counters are process-wide, like `snapshot._state`."""
    _state.update({
        "installs": 0, "reinstalls": 0, "skips": 0, "skip_reason": None,
        "class": None, "calls": 0, "policy_calls": 0, "vetoes": 0,
        "fallbacks": 0, "unrecognized": 0, "enum_resolved": None,
        "store_policy_present": None, "vetoes_at_last_report": 0,
    })
    _admission_cache[:] = [None, None, None]


# --- resolving the fork's names without importing it ------------------------

def _admissions(original):
    """(READY, ALREADY_CACHED) from the original function's own module globals.

    Returns (None, None) if the enum or a member is missing - a fork that
    renamed `_StoreAdmission` makes every veto fall through to the engine's own
    verdict, which is the only safe reading of an unknown return type.
    """
    globals_dict = getattr(original, "__globals__", None)
    if not isinstance(globals_dict, dict):
        _state["enum_resolved"] = False
        return None, None
    if _admission_cache[0] is not globals_dict:
        enum = globals_dict.get("_StoreAdmission")
        ready = getattr(enum, CONSULT_MEMBER, None)
        veto = getattr(enum, VETO_MEMBER, None)
        _admission_cache[:] = [globals_dict, ready, veto]
        _state["enum_resolved"] = ready is not None and veto is not None
    return _admission_cache[1], _admission_cache[2]


def manager_class():
    """The offload scheduler class, if this process already imported it.

    Same discipline as `engine_side._engine_core_class()`: `general_plugins`
    load in worker and arg-parsing processes too, and importing that tree from
    here would be a heavy side effect we do not own. Absent means "nothing to
    patch in this process", which is a fact we report, not an error.
    """
    module = sys.modules.get(TARGET_MODULE)
    if module is None:
        return None
    return getattr(module, TARGET_CLASS, None)


def manager_of(engine):
    """The scheduler-side manager off a live engine, or None. Reads only.

    The same three hops `snapshot.build_view()` walks; duplicated deliberately
    so that installing the hook never depends on the read side's payload.
    """
    scheduler = getattr(engine, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    return getattr(connector, "scheduler_manager", None)


# --- the per-call path ------------------------------------------------------

def _policy_answer(manager, gpu_block, verdict, holder):
    """Ask the policy. Returns a sentinel; raises only into the wrapper's guard."""
    decide = getattr(holder, "decide_store", None) if holder is not None else None
    if decide is None:
        # No `decide_store` in the live policy module. This is the normal state
        # for a deployment that only observes, and it must never be an error.
        _state["store_policy_present"] = False
        return _ALLOW
    _state["store_policy_present"] = True
    _state["policy_calls"] += 1
    answer = decide(build_store_ctx(manager, gpu_block, verdict))
    if isinstance(answer, str):
        text = answer.strip().lower()
        if text in VETO_WORDS:
            return _VETO
        if text in ALLOW_WORDS:
            return _ALLOW
        return _UNRECOGNIZED
    if answer is None:
        return _ALLOW
    return _UNRECOGNIZED


def build_store_ctx(manager, gpu_block, verdict) -> dict:
    """The `decide_store` argument. One small dict, built from maintained state.

    Field derivation (all of it from the real manager/gpu_block attributes):
      gpu_block_id / is_null / has_block_hash  <- KVCacheBlock fields the engine
                                                 itself tests, manager.py:933-936
      engine_admission                         <- the enum member name, i.e. the
                                                 reason the tier WOULD decline
                                                 (always "READY": a block the
                                                 engine already declined is
                                                 never offered to the policy, so
                                                 the eight drop reasons stay
                                                 honest and this stays fast)
      mode / target_free_gpu_blocks            <- _lazy_mode / _target_free
      block_size / hash_block_size / num_cpu_blocks
      cpu_free_blocks                          <- get_num_free_blocks(), an O(1)
                                                 maintained counter
      in_flight_store_blocks                   <- len(_in_flight_store_gpu_blocks)
      manager_class
    Nothing here stringifies a block hash, walks a container, or reads a clock.
    """
    pool = getattr(manager, "cpu_block_pool", None)
    free = None
    if pool is not None:
        getter = getattr(pool, "get_num_free_blocks", None)
        if callable(getter):
            free = getter()
    in_flight = getattr(manager, "_in_flight_store_gpu_blocks", None)
    return {
        "gpu_block_id": getattr(gpu_block, "block_id", None),
        "is_null": bool(getattr(gpu_block, "is_null", False)),
        "has_block_hash": getattr(gpu_block, "block_hash", None) is not None,
        "engine_admission": getattr(verdict, "name", None),
        "mode": "lazy" if getattr(manager, "_lazy_mode", False) else "eager",
        "target_free_gpu_blocks": getattr(manager, "_target_free", None),
        "block_size": getattr(manager, "block_size", None),
        "hash_block_size": getattr(manager, "hash_block_size", None),
        "num_cpu_blocks": getattr(manager, "num_cpu_blocks", None),
        "cpu_free_blocks": free,
        "in_flight_store_blocks": (len(in_flight) if in_flight is not None
                                   else None),
        "manager_class": type(manager).__name__,
    }


def _make_wrapper(original, get_holder):
    """The replacement for `_classify_store_candidate`. Quoted in the README."""

    def _classify_store_candidate(self, gpu_block):
        _state["calls"] += 1
        verdict = original(self, gpu_block)
        try:
            ready, veto = _admissions(original)
            if ready is None or verdict is not ready:
                # Either the fork renamed the enum (so we dare not override
                # anything) or the engine already declined this block.
                return verdict
            answer = _policy_answer(self, gpu_block, verdict, get_holder())
            if answer is _VETO:
                _state["vetoes"] += 1
                return veto
            if answer is _UNRECOGNIZED:
                _state["unrecognized"] += 1
            return verdict
        except Exception as exc:  # noqa: BLE001 - never into the scheduler
            _state["fallbacks"] += 1
            snapshot._error(exc, "store hook")
            return verdict

    setattr(_classify_store_candidate, OWNER_ATTR, OWNER_TAG)
    setattr(_classify_store_candidate, ORIGINAL_ATTR, original)
    _classify_store_candidate.__doc__ = (
        "kvtier-dev-plugin: the engine's own _classify_store_candidate, with a "
        "hot-reloadable policy veto in front of the READY verdict. Contained: "
        "on any problem the engine's own verdict is returned.")
    return _classify_store_candidate


# --- installation -----------------------------------------------------------

def install(cls, get_holder) -> dict:
    """Wrap `cls._classify_store_candidate`. Idempotent, never raises.

    Class-level, not instance-level: the live managers are already constructed,
    and an instance method body resolves its module globals at call time, so a
    class-level wrapper plus `holder.decide_store` (a property) is what makes a
    reloaded policy take effect without a restart.

    `get_holder` is a zero-argument callable, called per candidate: the wrapper
    holds the HOLDER, never a function reference, so a reload is picked up.

    Ownership, the same discipline as `attach_methods` with one difference
    forced by the target. Three cases:
      * marked with our tag  -> a re-install. No second copy; the new wrapper
        re-binds the FORK's function we stored, so a reload loop cannot nest
        wrappers and double-count vetoes.
      * marked with ANY other tag -> refused with a log line, never overwritten.
      * unmarked -> the name exists by definition, so "ours" and "the fork's"
        are told apart by `__module__`: only a function defined in
        `TARGET_MODULE` is accepted as the fork's own. A foreign monkeypatch
        from another module is refused. A third party that patched it in place
        from inside the fork's own module is the one case this cannot see.
    """
    report = {"installed": False, "replaced": False, "class": None,
              "method": TARGET_METHOD, "reason": None}
    try:
        if cls is None:
            return _refuse(report, f"no {TARGET_CLASS} is imported in this "
                                   f"process ({TARGET_MODULE} absent)")
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        original = getattr(cls, TARGET_METHOD, None)
        if original is None:
            return _refuse(report, f"{report['class']} has no {TARGET_METHOD}: "
                                   f"the fork renamed it, refusing to invent one")
        if not callable(original):
            return _refuse(report, f"{report['class']}.{TARGET_METHOD} is not "
                                   f"callable ({type(original).__name__})")
        owner = getattr(original, OWNER_ATTR, None)
        if owner is not None and owner != OWNER_TAG:
            held = getattr(original, "__qualname__", type(original).__name__)
            return _refuse(report, f"{report['class']}.{TARGET_METHOD} carries "
                                   f"another plugin's owner marker ({held!r}), "
                                   f"refusing to overwrite")
        report["replaced"] = owner == OWNER_TAG
        if not report["replaced"]:
            # Nothing marks it, so the only claim we can check is where it came
            # from. `attach_methods` can simply refuse any pre-existing name;
            # here the name exists by definition, so "not ours" and "the fork's
            # own" are told apart by the defining module. A third party that
            # patched it in place from INSIDE this module is the one case this
            # cannot see - the residual limit, stated in the README.
            home = getattr(original, "__module__", None)
            if home != TARGET_MODULE:
                return _refuse(report, f"{report['class']}.{TARGET_METHOD} is "
                                       f"neither ours nor the fork's own: it is "
                                       f"defined in {home!r}, not "
                                       f"{TARGET_MODULE!r}, refusing to "
                                       f"overwrite")
        if report["replaced"]:
            # Re-install over our own wrapper: bind to the ORIGINAL we stored,
            # so a reload loop cannot grow a chain of nested wrappers.
            original = getattr(original, ORIGINAL_ATTR, original)

        setattr(cls, TARGET_METHOD, _make_wrapper(original, get_holder))
        # A re-install replaces the wrapper object rather than skipping: that is
        # what lets a new `get_holder` take effect, and it binds the FORK's
        # function (not the wrapper it replaces), so nothing nests. Cost: one
        # closure per snapshot call, never per block.
        _state["class"] = report["class"]
        _state["skip_reason"] = None
        if report["replaced"]:
            # Nothing changed except the function object; the policy is already
            # live through the holder. Quiet on purpose: the dashboard polls a
            # route that lands here every few seconds.
            _state["reinstalls"] += 1
            logger.debug("kvtier-dev-plugin: store-policy hook already on %s.%s",
                         report["class"], TARGET_METHOD)
        else:
            _state["installs"] += 1
            logger.info("kvtier-dev-plugin: store-policy hook installed on %s.%s "
                        "(#%d)", report["class"], TARGET_METHOD,
                        _state["installs"])
            # A kill-switch policy must be live from the moment the hook lands,
            # not from the moment somebody remembers to reload. Contained:
            # `ensure_loaded` reports, it never raises.
            _cold_load_policy(get_holder)
        report["installed"] = True
        return report
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not break a boot
        _state["fallbacks"] += 1
        snapshot._error(exc, "store hook install")
        report["installed"] = False
        report["reason"] = f"install raised: {type(exc).__name__}: {exc}"[:200]
        return report


def _cold_load_policy(get_holder) -> None:
    """Load the policy once, on the path that just took ownership of stores.

    Without this, a restarted engine would wrap the classifier and then call a
    holder nobody had loaded yet: `decide_store` would be None, every veto
    silently a no-op, until a snapshot or a reload happened to load it. One
    file read plus one small `compile()`, once per install, on the engine loop
    thread - the same cost the reload verb already pays.
    """
    try:
        get_holder().ensure_loaded()
    except Exception as exc:  # noqa: BLE001 - a policy that will not load is data
        _state["fallbacks"] += 1
        snapshot._error(exc, "store hook policy cold load")


def _refuse(report, reason):
    """Record and say why the patch did not happen. Never an exception.

    The dashboard polls a route that lands here every few seconds, so an
    unchanging refusal (no offload manager in this process yet) is counted ONCE
    and then goes quiet: `skips` counts distinct refusal episodes rather than
    polls, and the log gets one line per episode. A successful install clears
    the reason, so a later different refusal is reported again.
    """
    if _state["skip_reason"] == reason:
        logger.debug("kvtier-dev-plugin: store-policy hook still not installed: "
                     "%s", reason)
    else:
        _state["skips"] += 1
        _state["skip_reason"] = reason
        logger.info("kvtier-dev-plugin: store-policy hook NOT installed: %s",
                    reason)
    report["installed"] = False
    report["reason"] = reason
    return report


def ensure_installed(engine, get_holder) -> dict:
    """Install from the LIVE manager's class if there is one. Never raises.

    `register()` usually runs before the connector (and therefore the manager's
    module) exists, so this is the path that actually lands the patch: the
    snapshot and the reload verb both hold a live engine, and the class we take
    from `type(manager)` is correct even if the fork moves the module.
    """
    try:
        manager = manager_of(engine)
    except Exception as exc:  # noqa: BLE001
        _state["fallbacks"] += 1
        snapshot._error(exc, "store hook manager_of")
        manager = None
    cls = type(manager) if manager is not None else manager_class()
    return install(cls, get_holder)


# --- observability ----------------------------------------------------------

def patch_report(manager=None) -> dict:
    """Patch state for the snapshot: is the hook live, and what has it done?

    `live_manager_patched` answers the question a counter cannot: does the
    CLASS OF THE MANAGER THAT IS ACTUALLY RUNNING carry our marker. A subclass
    that overrides the method, or a manager built before the patch on a class
    we never found, both show up here as False while `installs` is non-zero.
    """
    live_patched = None
    live_shadows = None
    mode = None
    if manager is not None:
        live_cls = type(manager)
        attribute = getattr(live_cls, TARGET_METHOD, None)
        live_patched = getattr(attribute, OWNER_ATTR, None) == OWNER_TAG
        live_shadows = TARGET_METHOD in vars(manager)
        mode = "lazy" if getattr(manager, "_lazy_mode", False) else "eager"
    vetoes = int(_state["vetoes"])
    since = vetoes - int(_state["vetoes_at_last_report"])
    _state["vetoes_at_last_report"] = vetoes
    return {
        "patched": bool(_state["installs"] or _state["reinstalls"]),
        "class": _state["class"],
        "module": TARGET_MODULE,
        "method": TARGET_METHOD,
        "installs": int(_state["installs"]),
        "reinstalls": int(_state["reinstalls"]),
        "skips": int(_state["skips"]),
        "skip_reason": _state["skip_reason"],
        "live_manager_patched": live_patched,
        "live_manager_mode": mode,
        "live_instance_shadows_method": live_shadows,
        # False here means the hook is live but INERT for the mode or the fork
        # in hand: no veto can reach the tier. See the module docstring.
        "governs_current_mode": (bool(live_patched) and mode == "eager"
                                 if manager is not None else None),
        "ready_resolved": _state["enum_resolved"],
        "decide_store_present": _state["store_policy_present"],
        "calls": int(_state["calls"]),
        "policy_calls": int(_state["policy_calls"]),
        "vetoes": vetoes,
        "vetoes_since_last_report": int(since),
        "fallbacks": int(_state["fallbacks"]),
        "unrecognized": int(_state["unrecognized"]),
        "veto_member": VETO_MEMBER,
        "veto_charged_to_engine_counter": "skipped_already_cached",
    }
