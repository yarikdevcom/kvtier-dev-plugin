"""Engine-side half: the `vllm.general_plugins` entry point.

`register()` is called with no arguments, and it is called in EVERY process -
inside `EngineCore.__init__` (vllm/v1/engine/core.py:214-217), from
`vllm/engine/arg_utils.py:837,3009`, and in worker processes - under
"load everything unless VLLM_PLUGINS narrows it" posture. So it is:

  * gated: with VLLM_E2_TIER_DEV unset it returns before touching anything;
  * idempotent: a marker attribute on our functions makes a second attach a
    no-op re-set of an equivalent function, and a name owned by somebody else
    is skipped with a warning instead of clobbered;
  * total: nothing it does can raise into the process that loaded it.

What it installs: SEVEN methods on the `EngineCore` class, plus one wrapper on
`KVConnectorFactory.create_connector` (the connector-creation hook that lands the
PR #56810 install when the tier connector is built). `EngineCoreProc`
(and `DPEngineCoreProc`, the actor classes) inherit them, and the UTILITY
dispatch looks them up with `getattr(self, method_name)`
(vllm/v1/engine/core.py:1669-1671), so a class-level method is exactly what
that lookup needs. Bodies live in `snapshot.py`, `kvconfig.py`, `kvinspect.py`,
`trace_ring.py` and `guard_mode.py`, each contained there. Every one of them
takes ONE optional argument (a plain dict), which is what lets the HTTP layer
forward a JSON body as a single argument - and what keeps a no-argument call
working, exactly as it always did, for callers that predate the arguments.

Enqueueing a UTILITY message is also what wakes an idle engine: the IO thread
puts on the same `input_queue` the busy loop blocks on (core.py:1904 vs
core.py:1562; upstream uses the identical trick for its WAKEUP sentinel at
core.py:1433-1436). That is why these handlers answer while the engine has no
requests in flight - and why they must be O(microseconds) reads.
"""

from __future__ import annotations

import logging
import sys
import threading

from . import (
    backport,
    config,
    cpu_capacity,
    evictions,
    guard_mode,
    kvconfig,
    kvinspect,
    lookup_ledger,
    snapshot,
    store_hook,
    trace_ring,
    worker_diag,
)
from .policy_holder import PolicyHolder

logger = logging.getLogger("kvtier_dev_plugin.engine_side")

# Marker stamped on the functions we attach. A name already in use by code we
# did not install is never overwritten.
OWNER_ATTR = "__kvtier_dev_owner__"
OWNER_TAG = "kvtier-dev-plugin/1"

_INSTALLS = {"count": 0}
_holder: PolicyHolder | None = None
_holder_lock = threading.Lock()


def get_holder() -> PolicyHolder:
    """The process-wide policy holder, built on first use.

    Building it reads no files: the load itself happens inside the contained
    `snapshot.policy_section()` / `snapshot.reload_policy_on_engine()` paths,
    so a broken policy file can never surface as an exception here.
    """
    global _holder
    if _holder is None:
        with _holder_lock:
            if _holder is None:
                _holder = PolicyHolder(str(config.canonical_policy_path()))
    return _holder


def _make_snapshot_method():
    def kvtier_dev_snapshot(self, args=None):  # noqa: ARG001 - bound method
        """Read-only KV-tier snapshot. Runs on the EngineCore loop thread.

        The only mutations on this path are `ensure_installed` adopting the
        store-policy patch and the capacity wrappers exactly once when the
        manager's class shows up after boot (see `_make_reload_method`), plus
        `cpu_capacity.adopt_if_missed`, which settles the CPU block-id ceiling
        once if the init-time handshake never delivered one. Afterwards they are
        marker checks. Without them a restarted engine would run the tier
        unvetted until nobody thought to POST a reload, which is the opposite of
        a kill switch.

        `args` is accepted so this method has the same calling convention as the
        argument-taking verbs; it is not read, and a call with NO arguments (the
        one every live caller makes, and what `GET /snapshot` still sends) is
        unaffected.
        """
        store_hook.ensure_installed(self, get_holder)
        cpu_capacity.ensure_installed()
        cpu_capacity.adopt_if_missed(self)
        backport.ensure_installed(self)
        # Arm the LRU-eviction counter on the live CPU pool once, so the arena's
        # steady state (99.9% full, parked spans recycled to serve new stores) is
        # visible instead of argued about. Fail-closed to a null section, never a
        # zero, if the pool cannot be wrapped. See evictions.py.
        evictions.adopt_if_missed(self)
        return snapshot.snapshot_for_engine(self, get_holder())

    kvtier_dev_snapshot.__doc__ = (
        "kvtier-dev-plugin: read-only scheduler-side KV tier snapshot. "
        "Pure reads, contained, no torch, no writes; adopts the store-policy "
        "patch once if it is not in place yet. Its optional `args` is accepted "
        "and not read, so a no-argument call behaves exactly as it always did.")
    return kvtier_dev_snapshot


def _make_reload_method():
    def kvtier_dev_reload_policy(self, args=None):  # noqa: ARG001 - bound method
        """Reload the policy, and adopt the store-policy patch if needed.

        `register()` runs early in `EngineCore.__init__`, before the connector
        imports `vllm/v1/simple_kv_offload/manager.py`, so this verb is what
        normally lands the hook - which is what makes the documented two-step
        kill switch (edit `policy.py`, POST this) sufficient on its own.
        Idempotent: on an already patched class this re-binds nothing and
        changes nothing but the policy module.

        `args` exists only so this method has the SAME calling convention as the
        argument-taking verbs (`_convert_msgspec_args` at core.py:1735 asserts
        `len(args) <= len(parameters)`, so a uniform arity is what lets the HTTP
        layer pass one dict to any of them). It is NEVER read: the engine reloads
        the one file this package owns, and nothing on the wire can select a
        different one.
        """
        store_hook.ensure_installed(self, get_holder)
        cpu_capacity.ensure_installed()
        cpu_capacity.adopt_if_missed(self)
        backport.ensure_installed(self)
        return snapshot.reload_policy_on_engine(get_holder())

    kvtier_dev_reload_policy.__doc__ = (
        "kvtier-dev-plugin: reload the ONE policy module this plugin owns "
        "(<package dir>/policy.py), and adopt the store-policy patch if the "
        "offload manager exists but is not wrapped yet. Its optional `args` is "
        "accepted and ignored: nothing from the wire can select a different "
        "file.")
    return kvtier_dev_reload_policy


def _make_kvconfig_method():
    def kvtier_dev_kvconfig(self, args=None):  # noqa: ARG001 - bound method
        """The KV-cache geometry: per-group block sizes, specs, and the divisibility
        the tier's first store asserts on. Pure reads, bounded, contained."""
        return kvconfig.kvconfig_for_engine(self, args)

    kvtier_dev_kvconfig.__doc__ = (
        "kvtier-dev-plugin: the live KV-cache geometry (num_blocks, block sizes, "
        "per-group spec class/page size/prefix-cacheability/layers, and every "
        "group's `block_size % hash_block_size == 0` verdict) plus the fork knobs "
        "that interact with it. Pure reads, bounded, no torch, no writes.")
    return kvtier_dev_kvconfig


def _make_inspect_method():
    def kvtier_dev_inspect(self, args=None):  # noqa: ARG001 - bound method
        """"Show me block 412" / "what does this request hold?", answered from the
        engine's own structures. Takes one optional dict of arguments."""
        return kvinspect.inspect_for_engine(self, args)

    kvtier_dev_inspect.__doc__ = (
        "kvtier-dev-plugin: one block (by id and group) or one request (by id), "
        "with per-group block ids and the fork's own 'how many blocks could the "
        "tier park here' expression. Every id list is capped and says how many it "
        "dropped. Pure reads, contained.")
    return kvtier_dev_inspect


def _make_trace_method():
    def kvtier_dev_trace(self, args=None):  # noqa: ARG001 - bound method
        """Read/arm/disarm/clear the bounded event ring. `args.op` selects which.

        Arming INSTALLS the observers (the scheduler-step wrappers and the tier
        entry-point wrappers) and disarming removes them again - that is how the
        disarmed cost is really zero rather than "a cheap branch".
        """
        return trace_ring.trace_for_engine(self, args)

    kvtier_dev_trace.__doc__ = (
        "kvtier-dev-plugin: the bounded scheduler/tier event ring - read, arm, "
        "disarm or clear it. Fixed depth, preallocated, capped answers, and "
        "nothing wrapped at all while it is disarmed.")
    return kvtier_dev_trace


def _make_lookups_method():
    def kvtier_dev_lookups(self, args=None):  # noqa: ARG001 - bound method
        """Read/arm/disarm/clear the bounded LOOKUP LEDGER; `args.op` selects.

        The ledger is what answers "the connector offered N tokens and the
        scheduler adopted 0 - which rule did that?": it records, per request id,
        the local hit the scheduler passed IN, the hit the connector returned,
        and the `num_external_tokens` the scheduler later passed to
        `update_state_after_alloc`. `op: verdict` classifies every joined pair.

        It owns no method of its own on the tier: the recording rides the guard
        wrapper, so `arm` installs those wrappers (composed over the backport's,
        never replacing them) and `disarm` removes them only when no other
        consumer still needs them.
        """
        return lookup_ledger.lookups_for_engine(self, args)

    kvtier_dev_lookups.__doc__ = (
        "kvtier-dev-plugin: the bounded lookup/adopt LEDGER - pair each "
        "get_num_new_matched_tokens() offer with the update_state_after_alloc() "
        "that followed it, read it, or ask for the verdict that classifies the "
        "hits the scheduler did not adopt. Fixed capacity, capped answers, "
        "records nothing while disarmed.")
    return kvtier_dev_lookups


def _make_guard_method():
    def kvtier_dev_guard(self, args=None):  # noqa: ARG001 - bound method
        """Guard mode: arm/disarm it, read its state, or fetch the captured faults.

        A DEVELOPMENT facility, and since the auto boot an ARMED ONE: the boot
        arms it before it arms the tier, so no store or lookup ever runs
        unguarded. That is a deliberate trade - a tier bug becomes a recorded
        DECLINE instead of an engine death - so the fault ring is the evidence a
        boot was clean, and the state is echoed as `guard_mode` in every verb
        payload. `disarm` is the operator's override and removes the wrappers when
        nothing else consumes them.
        """
        return guard_mode.guard_for_engine(self, args)

    kvtier_dev_guard.__doc__ = (
        "kvtier-dev-plugin: GUARD MODE (development only). Arm/disarm the tier "
        "exception capture-and-decline, read its state, or fetch captured faults. "
        "Never swallows a BaseException. Must be disarmed before any production "
        "boot.")
    return kvtier_dev_guard


def _make_tier_method():
    def kvtier_dev_tier(self, args=None):  # noqa: ARG001 - bound method
        """Arm or disarm the KV tier at runtime, without restarting the engine.

        THE TIER IS DISARMED AT PLUGIN LOAD and the boot ARMS IT at connector
        creation - but only on a verified install, because arming before the crash
        fix is in place is the production crash this module exists to prevent, and
        the connector hook is the first instant the manager class exists. So the
        boot patches, guards and arms itself, and `arm`/`disarm` are the operator's
        controls afterwards, not the mechanism that makes a boot work. The opt-out
        (`config.ENV_BOOT_INERT`) restores the old inert boot; the BOOT log line
        and `tier.boot` both say which mode ran and what it decided.

        An ARM is refused - naming `geometry`, `no_manager` or `incomplete_install`
        - unless the fix is fully installed, and a partial install force-disarms
        and logs. `disarm` makes the connector a strict no-op through the manager
        methods `backport` wraps: no prefix hit is reported, the metadata it hands
        the workers is empty, no pending work is claimed, and the finish path
        queues nothing - which is why a disarm gates BOTH routes into the selector
        that contains the upstream assert. It is the reason the tier can be turned
        on at all - a knob that needs a 4-minute reboot is not a kill switch. The
        upstream divisibility fix (PR #56810) is installed on the way through every
        op, so a hybrid geometry never reaches the assert.
        """
        return backport.tier_for_engine(self, args)

    kvtier_dev_tier.__doc__ = (
        "kvtier-dev-plugin: arm/disarm the KV tier at runtime and read its state "
        "(including whether the PR #56810 backport is live). Disarmed, the "
        "connector schedules nothing; the engine keeps serving from GPU KV.")
    return kvtier_dev_tier


_FACTORIES = {
    config.METHOD_SNAPSHOT: _make_snapshot_method,
    config.METHOD_RELOAD_POLICY: _make_reload_method,
    config.METHOD_KVCONFIG: _make_kvconfig_method,
    config.METHOD_INSPECT: _make_inspect_method,
    config.METHOD_TRACE: _make_trace_method,
    config.METHOD_GUARD: _make_guard_method,
    config.METHOD_TIER: _make_tier_method,
    config.METHOD_LOOKUPS: _make_lookups_method,
}


def attach_methods(cls) -> dict:
    """Attach our utility methods to an EngineCore-like class. Never raises.

    Returns a report: {"attached": [...], "replaced": [...], "skipped": [...]}
    """
    report = {"attached": [], "replaced": [], "skipped": []}
    candidates = ((name, factory()) for name, factory in _FACTORIES.items())
    for name, function in candidates:
        existing = getattr(cls, name, None)
        if existing is not None and getattr(existing, OWNER_ATTR, None) != OWNER_TAG:
            logger.warning("kvtier-dev-plugin: refusing to clobber %s.%s "
                           "(owned by %r)", cls.__name__, name,
                           getattr(existing, "__qualname__", type(existing).__name__))
            report["skipped"].append(name)
            continue
        setattr(function, OWNER_ATTR, OWNER_TAG)
        setattr(cls, name, function)
        report["replaced" if existing is not None else "attached"].append(name)
    return report


def _engine_core_class():
    """The EngineCore class, preferring an already-imported module.

    Importing vllm.v1.engine.core from a worker or an arg-parsing process is
    heavy; if it is not loaded yet in this process, importing it here would be
    a side effect we do not need, so we skip and say so.
    """
    module = sys.modules.get("vllm.v1.engine.core")
    if module is None:
        return None
    return getattr(module, "EngineCore", None)


def register() -> None:
    """The `vllm.general_plugins` entry point. Inert unless gated on."""
    try:
        if not config.enabled():
            return
        _INSTALLS["count"] += 1
        snapshot.attach_logger(logger)
        guard_mode.attach_logger(logger)
        # The handshake wrappers come FIRST, and independently of `EngineCore`
        # being imported: `general_plugins` loads in the worker processes too,
        # a worker never imports `vllm.v1.engine.core`, and the worker's half of
        # this pair is the only thing that reports its arena size. A capacity no
        # worker reported is a tier that parks nothing.
        #
        # The worker diagnostics writer is installed in the same breath, and for
        # the same reason - a worker has no utility channel, so a file is the
        # only way to see it. The FIRST attempt usually declines, because the
        # connector has not imported its worker module yet; the retry is driven
        # by the handshake hook below, which fires once this rank really has an
        # offload handler. With VLLM_E2_TIER_DIAG unset both are inert.
        cpu_capacity.set_worker_report_hook(worker_diag.ensure_installed)
        capacity = cpu_capacity.install()
        worker_files = worker_diag.install()
        # THE CONNECTOR-CREATION HOOK, and it comes before anything that could
        # refuse: this is what turns "patched before the first request" from an
        # operator habit into a property of the boot order. `create_connector` is
        # the only thing that imports the connector module, which imports the
        # offload manager at module scope - so the instant it returns is the first
        # instant `SimpleCPUOffloadScheduler` exists, still inside `Scheduler.__init__`
        # (core.py:260 -> scheduler.py:164) and long before any request. Fitted in
        # EVERY process like the capacity wrappers: in a worker the hook fires and
        # finds no `scheduler_manager`, which is the normal `no_manager` answer,
        # not an error, and a failure inside the hook can never stop a connector
        # being built. Contained by its own try/except; never raises.
        connector_hook = backport.install_connector_hook()
        cls = _engine_core_class()
        if cls is None:
            logger.info("kvtier-dev-plugin: vllm.v1.engine.core is not imported "
                        "in this process, nothing to attach (install #%d)",
                        _INSTALLS["count"])
            logger.info("kvtier-dev-plugin: CPU capacity clamp %s", capacity)
            logger.info("kvtier-dev-plugin: worker diagnostics writer %s",
                        worker_files)
            return
        report = attach_methods(cls)
        # Usually a no-op here: at this point in EngineCore.__init__ the
        # connector has not imported the offload manager yet, so the hook lands
        # later, from the reload verb or the first snapshot. When the module IS
        # already imported, this is what lands it immediately.
        hook = store_hook.install(store_hook.manager_class(), get_holder)
        # The PR #56810 backport, same posture as the store hook: usually a
        # no-op here because the connector has not imported the manager yet, and
        # an immediate patch when the module is already in sys.modules. It is
        # installed because without it a hybrid geometry - our five
        # block_size=3200 groups plus the block_size=8 QSA ring - dies on
        # `assert block_size % hash_block_size == 0` inside the store path, and
        # that costs a live-traffic reboot rather than degrading the tier.
        #
        # This refusal at boot is EXPECTED and is no longer a fail-open twice
        # over: at plugin load the tier's state is still `armed: False` from
        # `BOOT_ARMED`, so an engine that has not been patched schedules no tier
        # work, and the hook above lands the install anyway the moment the tier
        # connector is built - before the first request, without anybody curling a
        # verb. That same hook then GUARDS and ARMS (see `boot_from_connector`), so
        # "patched but inert" is no longer where a healthy boot ends. The verbs
        # remain the backstop for a process where the factory is imported after
        # plugin load (`ensure_installed` re-fits the hook), and arming stays
        # gated on the install (`backport.set_armed` refuses an unpatched tier).
        tier_fix = backport.install(backport.manager_class())
        # Unlike the store hook, the capacity wrappers MUST land here: the
        # EngineCore collects the handshake and calls the connector's consumer a
        # few lines after it loads plugins (core.py:288-302), and their targets
        # live on the connector BASE class, which is imported everywhere by then.
        # See cpu_capacity's module docstring.
        level = logger.info if _INSTALLS["count"] == 1 else logger.debug
        level("kvtier-dev-plugin: attached to %s %s (install #%d, policy %s)",
              cls.__name__, report, _INSTALLS["count"],
              config.canonical_policy_path())
        level("kvtier-dev-plugin: store-policy hook %s", hook)
        level("kvtier-dev-plugin: PR #56810 backport %s", tier_fix)
        level("kvtier-dev-plugin: connector-creation hook %s", connector_hook)
        level("kvtier-dev-plugin: CPU capacity clamp %s", capacity)
        level("kvtier-dev-plugin: worker diagnostics writer %s", worker_files)
        # WHAT THE BOOT WILL DO, stated at plugin load, and what it DID stated at
        # connector creation. Guard mode has no env var of its own and never will
        # (arming a bug-masking facility from a launcher script is the thing the
        # guard-mode suite pins); what the operator controls is the ONE opt-out
        # that turns the boot's own arming OFF. Until the tier connector exists
        # nothing is installed and nothing is armed, so this line can only name
        # the posture - the BOOT line from `backport.boot_from_connector` is the
        # verdict, and it is the line to read.
        boot_mode_name, boot_mode_why = backport.boot_mode()
        level("kvtier-dev-plugin: boot posture %s (%s). The verdict line is the "
              "one starting 'BOOT[%s]' below, written when the tier connector is "
              "built; until then nothing is patched, guarded or armed",
              boot_mode_name.upper(), boot_mode_why, boot_mode_name)
        # The fault ring, said out loud at every boot. Under the auto boot a real
        # tier bug becomes a DECLINE, not a crash - which is the whole point, and
        # exactly why "0 faults" is the thing to check rather than "did it boot".
        level("kvtier-dev-plugin: the guard records every declined tier exception "
              "in a bounded fault ring; after a boot with the tier armed, a "
              "non-zero count means the tier is silently NOT doing work it was "
              "asked to do: POST %s {\"op\":\"captures\"}",
              config.ROUTE_PREFIX + config.ROUTE_GUARD)
    except Exception:  # noqa: BLE001 - a diagnostic plugin must not break a boot
        logger.exception("kvtier-dev-plugin: engine-side install failed; "
                         "continuing without it")
