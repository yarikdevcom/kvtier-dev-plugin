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

What it installs: two methods on the `EngineCore` class. `EngineCoreProc`
(and `DPEngineCoreProc`, the actor classes) inherit them, and the UTILITY
dispatch looks them up with `getattr(self, method_name)`
(vllm/v1/engine/core.py:1669-1671), so a class-level method is exactly what
that lookup needs. Both bodies live in `snapshot.py` and are contained there.

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

from . import config, cpu_capacity, snapshot, store_hook
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
    def kvtier_dev_snapshot(self):  # noqa: ARG001 - bound method
        """Read-only KV-tier snapshot. Runs on the EngineCore loop thread.

        The only mutations on this path are `ensure_installed` adopting the
        store-policy patch and the capacity wrappers exactly once when the
        manager's class shows up after boot (see `_make_reload_method`), plus
        `cpu_capacity.adopt_if_missed`, which settles the CPU block-id ceiling
        once if the init-time handshake never delivered one. Afterwards they are
        marker checks. Without them a restarted engine would run the tier
        unvetted until nobody thought to POST a reload, which is the opposite of
        a kill switch.
        """
        store_hook.ensure_installed(self, get_holder)
        cpu_capacity.ensure_installed()
        cpu_capacity.adopt_if_missed(self)
        return snapshot.snapshot_for_engine(self, get_holder())

    kvtier_dev_snapshot.__doc__ = (
        "kvtier-dev-plugin: read-only scheduler-side KV tier snapshot. "
        "Pure reads, contained, no torch, no writes; adopts the store-policy "
        "patch once if it is not in place yet.")
    return kvtier_dev_snapshot


def _make_reload_method():
    def kvtier_dev_reload_policy(self):  # noqa: ARG001 - bound method
        """Reload the policy, and adopt the store-policy patch if needed.

        `register()` runs early in `EngineCore.__init__`, before the connector
        imports `vllm/v1/simple_kv_offload/manager.py`, so this verb is what
        normally lands the hook - which is what makes the documented two-step
        kill switch (edit `policy.py`, POST this) sufficient on its own.
        Idempotent: on an already patched class this re-binds nothing and
        changes nothing but the policy module.
        """
        store_hook.ensure_installed(self, get_holder)
        cpu_capacity.ensure_installed()
        cpu_capacity.adopt_if_missed(self)
        return snapshot.reload_policy_on_engine(get_holder())

    kvtier_dev_reload_policy.__doc__ = (
        "kvtier-dev-plugin: reload the ONE policy module this plugin owns "
        "(<package dir>/policy.py), and adopt the store-policy patch if the "
        "offload manager exists but is not wrapped yet. Takes no arguments, so "
        "nothing from the wire can select a different file.")
    return kvtier_dev_reload_policy


def attach_methods(cls) -> dict:
    """Attach our utility methods to an EngineCore-like class. Never raises.

    Returns a report: {"attached": [...], "replaced": [...], "skipped": [...]}
    """
    report = {"attached": [], "replaced": [], "skipped": []}
    candidates = ((config.METHOD_SNAPSHOT, _make_snapshot_method()),
                  (config.METHOD_RELOAD_POLICY, _make_reload_method()))
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
        # The handshake wrappers come FIRST, and independently of `EngineCore`
        # being imported: `general_plugins` loads in the worker processes too,
        # a worker never imports `vllm.v1.engine.core`, and the worker's half of
        # this pair is the only thing that reports its arena size. A capacity no
        # worker reported is a tier that parks nothing.
        capacity = cpu_capacity.install()
        cls = _engine_core_class()
        if cls is None:
            logger.info("kvtier-dev-plugin: vllm.v1.engine.core is not imported "
                        "in this process, nothing to attach (install #%d)",
                        _INSTALLS["count"])
            logger.info("kvtier-dev-plugin: CPU capacity clamp %s", capacity)
            return
        report = attach_methods(cls)
        # Usually a no-op here: at this point in EngineCore.__init__ the
        # connector has not imported the offload manager yet, so the hook lands
        # later, from the reload verb or the first snapshot. When the module IS
        # already imported, this is what lands it immediately.
        hook = store_hook.install(store_hook.manager_class(), get_holder)
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
        level("kvtier-dev-plugin: CPU capacity clamp %s", capacity)
    except Exception:  # noqa: BLE001 - a diagnostic plugin must not break a boot
        logger.exception("kvtier-dev-plugin: engine-side install failed; "
                         "continuing without it")
