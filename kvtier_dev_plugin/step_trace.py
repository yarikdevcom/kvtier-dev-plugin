"""Step-level trace observers: the scheduler's own ticks for the trace ring.

WHY THE TIER'S OWN ENTRY POINTS ARE NOT ENOUGH
----------------------------------------------
The engine is currently running with `connector: None`, so the tier entry points
wrapped by `guard_mode.py` do not exist and a ring fed only by them would record
nothing at all in the state this tier is being developed in. The scheduler's
`schedule()` (vllm/v1/core/sched/scheduler.py:595) and
`update_from_output()` (:1975) exist always, and they are the two points that
bound a step. Together they answer "what was the engine doing in the 300 steps
around the decline?" - step numbers, tokens scheduled, requests running - which
is what you need next to a KV-tier trace.

WHAT IT WRAPS, AND ONLY THIS
----------------------------
`type(engine.scheduler).schedule` and `.update_from_output`, on the LIVE
scheduler's class (no vllm import: the object is already in hand). Both wrappers
are pure observers:

  * they call the original first and return EXACTLY its result;
  * they catch nothing - an exception in `schedule()` must reach the engine loop
    exactly as it would without this plugin, so there is no `try` around the call
    at all. (Guard mode is the capability that swallows, and it swallows only
    inside the tier's own entry points.)
  * what they record is written through `trace_ring.record()`, which is contained
    and cannot raise.

COST, HONESTLY
--------------
Installed only while the trace ring is ARMED, and uninstalled when it is
disarmed: in the default state the fork's `schedule` is called directly, with no
extra frame. While armed, one extra Python frame per step plus one tuple into the
ring, and two `len()` calls on maintained containers (`scheduler.running` is a
list, `SchedulerOutput.num_scheduled_tokens` a dict,
`SchedulerOutput.total_num_scheduled_tokens` an int - output.py:240). Nothing
here allocates proportionally to the batch or the sequence length.

Restoration is exact: the fork's function object is stored on the wrapper and put
back by `uninstall()`, and a name that is not ours is never touched (the
`store_hook`/`guard_mode` ownership discipline).
"""

from __future__ import annotations

from . import snapshot, trace_ring

OWNER_ATTR = "__kvtier_dev_steptrace_owner__"
OWNER_TAG = "kvtier-dev-plugin/steptrace-1"
ORIGINAL_ATTR = "__kvtier_dev_steptrace_original__"

METHOD_SCHEDULE = "schedule"
METHOD_UPDATE_FROM_OUTPUT = "update_from_output"
WRAPPED_METHODS = (METHOD_SCHEDULE, METHOD_UPDATE_FROM_OUTPUT)

_state: dict = {"installs": 0, "reinstalls": 0, "uninstalls": 0, "skips": 0,
                "skip_reason": None, "class": None, "installed": False,
                "steps_recorded": 0, "outputs_recorded": 0, "fallbacks": 0}


def reset_state_for_tests() -> None:
    _state.update({"installs": 0, "reinstalls": 0, "uninstalls": 0, "skips": 0,
                   "skip_reason": None, "class": None, "installed": False,
                   "steps_recorded": 0, "outputs_recorded": 0, "fallbacks": 0})


def counters() -> dict:
    return dict(_state)


def _len_or_none(container):
    if container is None:
        return None
    try:
        return len(container)
    except TypeError:
        return None


def _record_step(scheduler, result) -> None:
    """One record per step, from the numbers the step already produced."""
    try:
        _state["steps_recorded"] += 1
        trace_ring.record(
            "sched:step",
            n1=_int(getattr(scheduler, "current_step", None)),
            n2=_int(getattr(result, "total_num_scheduled_tokens", None)),
            text=(f"running={_len_or_none(getattr(scheduler, 'running', None))}"
                  f" waiting={_len_or_none(getattr(scheduler, 'waiting', None))}"
                  f" requests={_len_or_none(getattr(scheduler, 'requests', None))}"))
    except Exception as exc:  # noqa: BLE001 - a trace never breaks a step
        _state["fallbacks"] += 1
        snapshot._error(exc, "step trace record")


def _record_output(scheduler, scheduler_output) -> None:
    try:
        _state["outputs_recorded"] += 1
        trace_ring.record(
            "sched:output",
            n1=_int(getattr(scheduler, "current_step", None)),
            n2=_len_or_none(getattr(scheduler_output, "num_scheduled_tokens", None)),
            text=None)
    except Exception as exc:  # noqa: BLE001
        _state["fallbacks"] += 1
        snapshot._error(exc, "step trace output record")


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _make_schedule(original):
    def schedule(self, *args, **kwargs):
        # No try/except: this is the serving path, and the fork's own failure is
        # the engine's to see. Only the RECORD below is contained.
        result = original(self, *args, **kwargs)
        if trace_ring.is_armed():
            _record_step(self, result)
        return result
    return schedule


def _make_update_from_output(original):
    def update_from_output(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if trace_ring.is_armed():
            _record_output(self, args[0] if args else kwargs.get("scheduler_output"))
        return result
    return update_from_output


_FACTORIES = {METHOD_SCHEDULE: _make_schedule,
              METHOD_UPDATE_FROM_OUTPUT: _make_update_from_output}


def _refuse(reason: str) -> None:
    if _state["skip_reason"] == reason:
        return
    _state["skips"] += 1
    _state["skip_reason"] = reason


def scheduler_class(engine):
    """The live scheduler's class, or None. Never imports vllm."""
    scheduler = getattr(engine, "scheduler", None)
    return type(scheduler) if scheduler is not None else None


def install(engine) -> dict:
    """Wrap the two scheduler methods. Idempotent, never raises."""
    report = {"installed": False, "attached": [], "replaced": [], "skipped": [],
              "class": None, "reason": None}
    try:
        cls = scheduler_class(engine)
        if cls is None:
            _refuse("this engine has no scheduler object to observe")
            report["reason"] = _state["skip_reason"]
            return report
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        for name in WRAPPED_METHODS:
            original = getattr(cls, name, None)
            if original is None or not callable(original):
                report["skipped"].append(name)
                _refuse(f"{report['class']}.{name} is missing or not callable")
                continue
            owner = getattr(original, OWNER_ATTR, None)
            if owner is not None and owner != OWNER_TAG:
                report["skipped"].append(name)
                _refuse(f"{report['class']}.{name} carries another plugin's owner "
                        f"marker, refusing to overwrite")
                continue
            replaced = owner == OWNER_TAG
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
                "kvtier-dev-plugin: the scheduler's own "
                f"{name}, plus one trace-ring record per call while the trace "
                "ring is armed. A pure observer: nothing here catches anything.")
            setattr(cls, name, wrapped)
        report["installed"] = bool(report["attached"] or report["replaced"])
        if report["attached"]:
            _state["installs"] += 1
            _state["class"] = report["class"]
            _state["installed"] = True
            _state["skip_reason"] = None
        if report["replaced"]:
            _state["reinstalls"] += 1
        if not report["installed"]:
            report["reason"] = _state["skip_reason"]
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        snapshot._error(exc, "step trace install")
        report["reason"] = f"install raised: {type(exc).__name__}"[:160]
        return report


def uninstall(engine) -> dict:
    """Put the fork's own scheduler methods back. Never raises."""
    report = {"uninstalled": False, "restored": [], "left": [], "class": None,
              "reason": None}
    try:
        cls = scheduler_class(engine)
        if cls is None:
            report["reason"] = "this engine has no scheduler object"
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
        snapshot._error(exc, "step trace uninstall")
        report["reason"] = f"uninstall raised: {type(exc).__name__}"[:160]
        return report
