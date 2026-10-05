"""The BOOT: a tier-env engine must come up as a working system, unattended.

The contract this file pins is the one an operator cares about: boot the engine
with the tier env set and it comes up **installed, guarded and armed**, with no
human in the loop - and it comes up that way ONLY if every part verifies, because
an armed tier whose crash fix did not land is the production crash this plugin
exists to prevent.

Six claims, roughly in the order the boot performs them:

1. a clean install arms guard mode and the tier, in that order, at connector
   creation (no unguarded window, no manual `arm`);
2. a partial install, or a live instance the check cannot make safe, refuses to
   arm and names WHICH part failed - in the payload and in the one BOOT line;
3. the guard is fitted and armed BEFORE the tier is armed, and the arm gate asks
   about the attribute the scheduler resolves, not about a report's word for it;
4. `VLLM_E2_TIER_DEV_BOOT_INERT` is the explicit opt-out back to the old inert
   posture, and the status says that is why nothing is armed;
5. a second boot COMPOSES with the first instead of double-wrapping it: one guard
   layer deep no matter how many boots run, over the backport's and switch's
   layers, which survive underneath;
6. a tier fault after the boot is post-boot EVIDENCE, not a crash: the scheduler
   loop sees a decline, the fault ring counts it, and the next boot's line carries
   the count - which is what makes "0 faults at boot" the claim worth reading.

Fixture honesty: the fake fork class is built by `fixtures_pr56810` OUTSIDE the
fork's module, so a method the backport REPLACED carries a test-fixture
`__module__`. `guard_mode.sibling_layer()` is what recognises it as ours anyway -
the markers are the truth and `__module__` lies by design, because
`backport._compile_copy` deliberately restores the FORK's module name on the copy.
Without that, guard mode would refuse the most fault-prone method, the coverage
gate would refuse to arm, and every test here would be proving the wrong thing.
"""
from __future__ import annotations

import types

import fixtures_pr56810 as PR  # noqa: F401 - imported for its module-level setup
import pytest
import test_backport as TB

from kvtier_dev_plugin import backport, config, guard_mode, snapshot


@pytest.fixture(autouse=True)
def boot_state():
    """Save/restore the two pieces of process-wide state the boot writes.

    The boot arms BOTH `backport` and `guard_mode`, so a test here that boots leaks
    an armed tier and an armed guard into whatever file runs next - the failure mode
    the suite already guards against elsewhere. Restored, not cleared: state that
    belonged to someone else before this test must be theirs after it. The boot
    opt-out env var is cleared for every test by conftest's `clean_env`.
    """
    saved_bp = dict(backport._state)
    saved_verdicts = dict(backport._state["method_verdicts"])
    saved_switch = dict(backport._state["switch_verdicts"])
    saved_guard = dict(guard_mode._state)
    backport.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    yield
    backport._state.clear()
    backport._state.update(saved_bp)
    backport._state["method_verdicts"] = saved_verdicts
    backport._state["switch_verdicts"] = saved_switch
    guard_mode._state.clear()
    guard_mode._state.update(saved_guard)


def boot(cls=None, manager=None, report=None):
    """Run the real boot the engine runs: install, then `boot_from_connector`.

    Uses the production-geometry manager by default, so the geometry guard passes
    for the same reason it passes on the box.
    """
    if cls is None:
        cls = TB._six_group_class()
    if manager is None:
        manager = TB._production_manager(cls)
    if report is None:
        report = backport.install(cls, TB.our_shape_config(), manager)
    return report, backport.boot_from_connector(cls, report, manager), cls, manager


def layers_of(cls, method):
    """The layer kinds on the attribute the scheduler resolves, outermost first."""
    current = getattr(cls, method, None)
    if current is None:
        return []
    layers, _bottom = backport._layers_of(backport._function_of(current))
    return [kind for kind, _wrapper in layers]


def fork_function(name, body):
    """A stand-in for a FORK-OWNED method: the module name is what marks it as ours.

    Built by executing real source rather than writing a closure, so the object is
    an ordinary function like the fork's - and `__module__` is stamped to the fork's
    module, which is the only thing that lets `guard_mode.install()` accept it as
    "the fork's own, ours to wrap". Used here to stand in for a fork body that
    RAISES, which is the shape of a real tier bug.
    """
    namespace: dict = {}
    exec(compile(body, "fork_source.py", "exec"), namespace)  # noqa: S102
    fn = namespace[name]
    fn.__module__ = backport.TARGET_MODULE
    return fn


# --------------------------------------------------------------------------
# 1. a clean boot arms everything
# --------------------------------------------------------------------------

def test_a_clean_boot_installs_guards_and_arms_without_any_verb():
    """The whole point: nothing in this test is a verb and the tier still works.

    `create_connector` is the only human-free moment the manager class exists, so
    the boot that runs there is the only thing that can make an unattended tier
    boot correct. If this test ever needs a `set_armed(True)` or a guard POST to
    pass, the auto boot has regressed into "somebody remembers".
    """
    report, boot_report, _cls, _manager = boot()

    assert report["installed"] is True
    # ...the guard: fitted, and armed.
    assert guard_mode.is_armed() is True
    assert guard_mode._state["installs"] == 1
    # ...and the tier, armed by the boot rather than by a curl.
    assert backport.is_armed() is True
    assert boot_report["tier"] is True and boot_report["backport"] is True
    assert boot_report["guard"] is True
    assert boot_report["refusals"] == []
    assert boot_report["mode"] == "auto"
    assert backport.section()["armed"] is True


def test_the_boot_reports_the_manager_it_booted_and_its_fault_count():
    """`boot.manager` / `faults_at_boot` / `faults_now`: post-boot evidence."""
    _report, boot_report, _cls, manager = boot()
    assert boot_report["manager"] == type(manager).__name__
    assert boot_report["faults_at_boot"] == 0

    section = backport.section()
    assert section["boot"]["ran"] is True
    assert section["boot"]["boots"] == 1
    assert section["boot"]["faults_now"] == 0
    assert "faults_now" in section["boot"]["note"]


def test_an_instance_constructed_before_the_patch_is_backfilled_and_armed():
    """The live-object check exists to make a PRE-EXISTING instance safe.

    Upstream's `__init__` addition sets `prefix_cacheable_group_ids`, but OUR
    `__init__` wrapper only fires for instances constructed after the install -
    and production constructs first (`cls.__new__` in the fork's own path, the
    connector built inside `Scheduler.__init__`). So the check back-fills the one
    attribute this patch ADDS from the same source the fork uses. An instance that
    predates the patch is therefore armable, and the boot says it back-filled.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    del manager.prefix_cacheable_group_ids      # what a pre-install instance lacks

    _report, boot_report, _cls, _manager = boot(cls, manager)
    assert boot_report["live_check"]["backfilled"] is True
    assert boot_report["backport"] is True
    assert boot_report["tier"] is True and backport.is_armed() is True


# --------------------------------------------------------------------------
# 2. what refuses to arm, and what it says
# --------------------------------------------------------------------------

def test_a_partial_install_refuses_to_arm_and_names_the_failing_part():
    """A fingerprint drift means the crash fix is NOT in place.

    Arming that tier would schedule tier work into the selector that asserts - the
    exact crash the backport exists to prevent, arriving on the first request of a
    live-traffic boot. So the boot declines, and the payload plus the BOOT line
    must name the part that failed: an operator must not have to guess whether it
    was the backport, the guard, or the geometry.
    """
    cls = TB._six_group_class(drift="_select_eager_blocks_to_store")
    manager = TB._production_manager(cls)
    report, boot_report, _cls, _ = boot(cls, manager)

    assert report["installed"] is False
    assert backport.is_armed() is False, "a partial install must never arm"
    assert boot_report["backport"] is False and boot_report["tier"] is False
    assert any(r.startswith("backport:") and "fingerprint" in r
               for r in boot_report["refusals"]), boot_report["refusals"]
    assert boot_report["guard"] is True, "the guard is still fitted and armed"
    # The kill-switch wrappers stay fitted either way: wrapped + disarmed is what
    # gates BOTH routes to the selector, so a refused boot is a SAFE engine.
    assert all(v["wrapped"] for v in report["switch"].values())
    assert "backport=NO" in boot_report["line"]
    assert "tier=DISARMED" in boot_report["line"]
    assert "refused=" in boot_report["line"]


def test_a_live_instance_that_cannot_be_made_safe_is_not_armed():
    """The live-object check is what makes "the class is patched" mean anything.

    Every required method can carry a marker while the connector instance the
    engine is holding is missing an attribute the patched bodies read - then the
    first request raises `AttributeError` inside the scheduler loop and the payload
    would still say `installed: true`. The boot refuses that, and names the
    attribute, because "install verified" must mean verified ON THE OBJECT.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    del manager.cpu_coordinator             # REQUIRED_LIVE_ATTRIBUTES

    report, boot_report, _cls, _ = boot(cls, manager)
    assert report["live_check"]["ok"] is False
    assert boot_report["backport"] is False, \
        "a class verdict is not an instance verdict"
    assert boot_report["tier"] is False and backport.is_armed() is False
    assert any("cpu_coordinator" in r for r in boot_report["refusals"])


def test_a_method_the_guard_refuses_to_wrap_blocks_the_arm_and_is_named():
    """An entry point the guard cannot own is a hole, not a detail.

    The guard refuses exactly one kind of attribute: one carrying ANOTHER plugin's
    owner marker. If it cannot wrap it, an exception there still kills the
    EngineCore - and arming the tier on top of that is the confidently-wrong state
    (`armed: true`, `guard_mode: true`, unguarded). The gate reads the layer stack
    of the attribute the scheduler resolves, so a foreign layer cannot hide behind
    a report that says the method was skipped.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)

    foreign = fork_function("build_connector_meta",
                            "def build_connector_meta(self, *a, **k):\n"
                            "    return None\n")
    setattr(foreign, guard_mode.OWNER_ATTR, "some_other_plugin")
    cls.build_connector_meta = foreign

    boot_report = backport.boot_from_connector(cls, report, manager)
    assert backport._guard_layer_on(cls, "build_connector_meta") is False
    assert backport.is_armed() is False, "an unguarded entry point means no arm"
    assert any(r.startswith("guard:") and "build_connector_meta" in r
               for r in boot_report["refusals"]), boot_report["refusals"]
    assert "tier=DISARMED" in boot_report["line"]


def test_a_method_the_fork_does_not_define_is_reported_but_does_not_block():
    """`not_covered` is honest reporting, not a veto.

    The guard wraps the nine methods it knows; a fork that inherits or renames one
    leaves the class without it. Treating that as a failure would take the whole
    tier down on a rename, so it is listed under `not_covered` in the payload and
    the BOOT line - the line never claims more coverage than it has, and a rename
    cannot silently narrow what an operator believes is guarded.
    """
    _report, boot_report, cls, _manager = boot()
    assert boot_report["tier"] is True, "not_covered does not block the arm"
    assert boot_report["not_covered"], "this fixture class really is missing some"
    for method in boot_report["not_covered"]:
        assert method in guard_mode.WRAPPED_METHODS
        assert getattr(cls, method, None) is None, \
            "a method the class HAS must not be reported as merely not covered"
    assert "not_covered=" in boot_report["line"]

def test_a_method_inherited_from_outside_the_fork_is_a_limit_not_a_refusal():
    """THE PRODUCTION SHAPE: a real `SimpleCPUOffloadScheduler` has these methods.

    `take_events`, `reset` and `update_connector_output` are inherited from the
    connector BASE class (`vllm.distributed...v1.base`), and `guard_mode.install()`
    will not shadow a method whose home is another module - correctly, since taking
    over another module's method is not this plugin's to do. So on the real engine
    those three can never carry a guard layer, and a gate that read "attribute
    exists, therefore the guard must own it" would refuse to arm on EVERY real boot:
    fail-closed, safe, and a feature that works in a fixture and nowhere else. The
    gate therefore asks who OWNS the attribute. The limit is reported, not hidden.
    """
    cls = TB._six_group_class()
    base_home = "vllm.distributed.kv_transfer.kv_connector.v1.base"

    class Base:  # the connector base, in its own module - not ours, not the fork's
        pass

    Base.__module__ = base_home
    for name in ("take_events", "reset", "update_connector_output"):
        def make(_name):
            def method(self, *args, **kwargs):
                return None
            method.__name__ = _name
            method.__module__ = base_home
            return method
        setattr(Base, name, make(name))

    # A subclass with the fork's module name, exactly as the fork's class looks:
    # the tier entry points in its own `__dict__`, the base's on its ancestors.
    production = types.new_class("SimpleCPUOffloadScheduler", (cls, Base))
    production.__module__ = backport.TARGET_MODULE
    manager = TB._production_manager(cls)
    live = production.__new__(production)
    vars(live).update(vars(manager))

    for name in ("take_events", "reset", "update_connector_output"):
        assert backport._class_owns(production, name) is False, name
        assert backport._class_owns(production, "build_connector_meta") is True

    report, boot_report, _cls, _m = boot(production, live)
    assert report["installed"] is True
    assert boot_report["tier"] is True, "inherited methods must not veto the arm"
    assert backport.is_armed() is True
    assert boot_report["refusals"] == []
    assert set(boot_report["not_covered"]) == {
        "take_events", "reset", "update_connector_output"}
    # And they are in the line, so the limit is visible from the log alone.
    assert "not_covered=" in boot_report["line"]


def test_an_unowned_but_defined_entry_point_still_blocks_the_arm():
    """The other side of that rule: owned AND unguarded is a hole, and stays fatal.

    Relaxing the inherited case must not relax the case that matters. A method this
    class defines itself, which the guard REFUSED because another plugin claims
    ownership of it, is the state the boot must never arm into - otherwise the
    ownership fix would be a way to talk the gate out of its own safety property.
    The foreign marker is what makes this the guard's refusal rather than a hole an
    operator has to find by hand.
    """
    cls = TB._six_group_class()
    foreign = fork_function("reset", "def reset(self, *a, **k):\n    return False\n")
    foreign.__kvtier_dev_guard_owner__ = "some_other_plugin"
    cls.reset = foreign
    assert backport._class_owns(cls, "reset") is True, "defined on the class itself"
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)

    boot_report = backport.boot_from_connector(cls, report, manager)
    assert backport._guard_layer_on(cls, "reset") is False, "the guard refused it"
    assert "reset" not in boot_report["not_covered"], \
        "a method the class defines is not an inherited limit"
    assert any(r.startswith("guard:") and "reset" in r
               for r in boot_report["refusals"]), boot_report["refusals"]
    assert backport.is_armed() is False
    assert "tier=DISARMED" in boot_report["line"]


# --------------------------------------------------------------------------
# 3. guard BEFORE arm
# --------------------------------------------------------------------------

def test_the_guard_is_fitted_and_armed_before_the_tier_is_armed(monkeypatch):
    """Ordering as a safety property, observed rather than asserted about.

    Arming the tier while an entry point is unguarded converts the first real tier
    bug into an EngineCore death, so "guard first" is not a style preference. The
    only honest way to pin an order is to record the calls: a reorder of two lines
    in the boot must fail HERE, not in production.
    """
    seen = []
    real_guard_arm = guard_mode.set_guard_mode
    real_arm = backport.set_armed

    def spy_guard(armed, engine=None):
        seen.append(("guard", armed, backport.is_armed()))
        return real_guard_arm(armed, engine)

    def spy_arm(armed):
        seen.append(("tier", armed, guard_mode.is_armed()))
        return real_arm(armed)

    monkeypatch.setattr(guard_mode, "set_guard_mode", spy_guard)
    monkeypatch.setattr(backport, "set_armed", spy_arm)

    _report, _boot_report, _cls, _manager = boot()

    assert [s[0] for s in seen] == ["guard", "tier"], seen
    # The tier arm was attempted while the guard was ALREADY armed - not merely
    # earlier in a list somebody could reorder tomorrow.
    assert seen[0] == ("guard", True, False)
    assert seen[1] == ("tier", True, True)


def test_the_arm_gate_asks_about_the_resolved_attribute_not_a_report():
    """The gate's question is about the class, so it survives a re-install.

    `guard_mode.install()` reports what it wrapped; the boot's question is whether
    the guard is a layer of the CURRENT attribute wherever it sits. A check written
    against the report would keep passing after another module of this package put
    a layer outside the guard - which is exactly when it would start lying.
    """
    _report, boot_report, cls, _manager = boot()
    assert boot_report["nested"], "a boot that guarded nothing is not a boot"
    for method in guard_mode.WRAPPED_METHODS:
        if method in boot_report["nested"] or method in boot_report["not_covered"]:
            continue
        assert backport._guard_layer_on(cls, method) is True, method


# --------------------------------------------------------------------------
# 4. the explicit opt-out
# --------------------------------------------------------------------------

def test_boot_inert_is_the_explicit_opt_out_and_the_status_says_why(monkeypatch):
    """The escape hatch, and the honesty that has to come with it.

    `VLLM_E2_TIER_DEV_BOOT_INERT` restores the previous posture on purpose (patch,
    then wait for a human). What must NOT come back is ambiguity: an operator
    reading `armed: false` has to be able to tell "I ordered this" from "the boot
    refused", so the mode, the reason, and the refusal all name the env var.
    """
    monkeypatch.setenv(config.ENV_BOOT_INERT, "1")
    report, boot_report, _cls, _manager = boot()

    assert report["installed"] is True, "installing is not arming; it still happens"
    assert backport.is_armed() is False and guard_mode.is_armed() is False
    assert boot_report["mode"] == "inert"
    assert config.ENV_BOOT_INERT in boot_report["mode_why"]
    assert boot_report["tier"] is False and boot_report["guard"] is False
    assert "tier=DISARMED" in boot_report["line"] and "BOOT[inert]" in \
        boot_report["line"]
    assert any(config.ENV_BOOT_INERT in r for r in boot_report["refusals"])
    note = backport.section()["operator_note"]
    assert config.ENV_BOOT_INERT in note and "BOOT" in note


@pytest.mark.parametrize("value,expected", [
    ("1", "inert"), ("true", "inert"), ("YES", "inert"), ("on", "inert"),
    ("banana", "inert"),
    ("0", "auto"), ("false", "auto"), ("no", "auto"), ("off", "auto"),
    ("", "auto"),
])
def test_the_opt_out_reads_the_same_falsy_set_as_every_other_flag(monkeypatch,
                                                                  value,
                                                                  expected):
    """`BOOT_INERT=0` must not mean inert; `BOOT_INERT=banana` must.

    The truth rule is `config.enabled()`'s, shared by every flag in this package:
    the documented falsy words turn a flag off and anything else turns it on. A
    second, stricter rule for this one variable would mean an operator who wrote
    `BOOT_INERT=0` to get the working boot back gets it, and one who typoed gets
    the inert one - which is the same surprise `VLLM_E2_TIER_DEV=banana` already
    gives, applied consistently rather than re-invented here.
    """
    monkeypatch.setenv(config.ENV_BOOT_INERT, value)
    assert backport.boot_mode()[0] == expected
    # And the reason always names the variable, whichever way it resolved.
    assert config.ENV_BOOT_INERT in backport.boot_mode()[1]


# --------------------------------------------------------------------------
# 5. boots compose; they never double-wrap
# --------------------------------------------------------------------------

def test_a_second_boot_composes_and_leaves_exactly_one_guard_layer():
    """Two connector creations in one process, one guard layer.

    The connector-creation hook runs per tier connector and the reload verb re-runs
    the boot path, so a boot that wrapped twice would leave a chain whose exception
    handling runs twice per call - double-counted faults, a decline produced from an
    inner decline, and a `nested` report that no longer describes the class. The
    marker chain is the only evidence of that, so assert the chain.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    first = backport.boot_from_connector(cls, backport.install(
        cls, TB.our_shape_config(), manager), manager)
    assert first["tier"] is True
    before = {m: layers_of(cls, m) for m in guard_mode.WRAPPED_METHODS
              if layers_of(cls, m)}
    assert before, "the first boot must have wrapped something"

    second = backport.boot_from_connector(cls, backport.install(
        cls, TB.our_shape_config(), manager), manager)
    assert second["tier"] is True, "a re-boot stays armed, it does not disarm itself"

    for method, kinds in before.items():
        after = layers_of(cls, method)
        assert after == kinds, f"{method}: {kinds} -> {after}"
        assert after.count("guard") == 1, method
        assert after[0] == "guard", f"{method} must stay outermost: {after}"
    # The backport's and the switch's layers survive UNDERNEATH the guard: guard
    # over switch over backport over the fork. That ordering is the whole
    # layering requirement, and a double-wrap would have broken it here.
    assert "backport" in layers_of(cls, "update_state_after_alloc")
    assert "switch" in layers_of(cls, "get_num_new_matched_tokens")
    assert guard_mode._state["installs"] == 1, "a re-boot is not a second install"
    assert backport.section()["boot"]["boots"] == 2


def test_the_guard_is_nested_over_the_backport_which_module_cannot_see():
    """A layer of THIS package is composed with, never refused as "foreign".

    `backport` REPLACES a method with a compiled copy that deliberately carries the
    FORK's `__module__` (so the fork's own `__globals__` resolve correctly inside
    it), which means `__module__` cannot answer "is this ours?" - in a class built
    outside the fork's module it answers "stranger", and the guard would refuse the
    method most likely to raise. The markers answer, so ask them. This is the one
    place where the fixture's module name and production's differ, and the fix is
    in production's favour, not the test's.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)
    replaced = "update_state_after_alloc"
    assert replaced in backport.REPLACED
    assert report["methods"][replaced]["patched"] is True

    current = backport._function_of(getattr(cls, replaced))
    assert getattr(current, "__module__", None) != backport.TARGET_MODULE, \
        "the trap this test exists for: the copy does not look like ours by module"
    assert guard_mode.sibling_layer(current) == "backport", "but the marker says it is"

    boot_report = backport.boot_from_connector(cls, report, manager)
    assert layers_of(cls, replaced)[:2] == ["guard", "backport"]
    assert replaced in boot_report["nested"], "and the payload says so"
    assert backport.is_armed() is True


# --------------------------------------------------------------------------
# 6. a tier fault after the boot is evidence, not an outage
# --------------------------------------------------------------------------

def test_a_raising_tier_method_becomes_a_record_and_survives_the_next_boot():
    """A tier bug is a DECLINE plus a fault-ring entry - never an engine death.

    Three things have to be true at once, and they are normally traded against each
    other: the scheduler loop sees a decline (not an exception), the fault is
    recoverable AS DATA after the boot (the ring is the evidence a boot was clean,
    and the reason an armed-by-default guard is acceptable), and nothing else is
    corrupted - above all the tier's armed state, which is the operator's kill
    switch and must not be lost to a tier bug.
    """
    cls = TB._six_group_class()
    # The FORK's own body raises, before any of our layers exist: the real bug
    # shape, an assert inside the store path. `request_finished` is guarded but
    # neither replaced nor switch-wrapped, so the guard is its only layer.
    cls.request_finished = fork_function(
        "request_finished",
        "def request_finished(self, *a, **k):\n"
        "    raise RuntimeError('assert block_size % hash_block_size == 0')\n")
    manager = TB._production_manager(cls)

    report, boot_report, _cls, _manager = boot(cls, manager)
    assert boot_report["tier"] is True and backport.is_armed() is True

    declined = cls.request_finished(manager, None)
    assert declined == (False, None), "the fork's own 'nothing finished' answer"
    assert guard_mode.fault_count() == 1
    assert snapshot._state["errors"] == 0, "a captured tier fault is not a plugin bug"

    # The kill switch survives the tier's own failure, and the class is still
    # fully patched, so an operator can still turn the tier off and back on.
    assert backport.set_armed(False)["armed"] is False
    assert backport.set_armed(True)["armed"] is True, "the class is still patchable"

    captures = guard_mode.captures(2)
    assert captures["returned"] == 1 and captures["dropped"] == 0
    assert captures["captures"][0]["method"] == "request_finished"
    assert captures["captures"][0]["exception_type"] == "RuntimeError"
    assert "block_size" in captures["captures"][0]["message"]
    assert captures["captures"][0]["declined"] is True

    # A fault recorded BEFORE a boot is carried into that boot's line, because
    # "faults=0" is the claim the README tells operators to check.
    later = backport.boot_from_connector(cls, backport.install(
        cls, TB.our_shape_config(), manager), manager)
    assert later["faults_at_boot"] == 1
    assert "faults=1" in later["line"]
    assert backport.section()["boot"]["faults_now"] == 1

    # The ring is EVIDENCE, so toggling the guard must not destroy it. Disarming
    # and re-arming is the ordinary thing an operator does while investigating a
    # capture - if it cleared the ring, the act of investigating would erase the
    # evidence, and the "faults=N" claim in the boot line would be unfalsifiable.
    guard_mode.set_guard_mode(False, None)
    assert guard_mode.is_armed() is False
    assert guard_mode.fault_count() == 1
    assert guard_mode.captures(2)["returned"] == 1
    guard_mode.set_guard_mode(True, None)
    assert guard_mode.fault_count() == 1
    assert backport.section()["boot"]["faults_now"] == 1


def test_the_boot_survives_a_class_it_cannot_read_and_still_reports():
    """A boot that cannot complete is a report, not an exception.

    `boot_from_connector` runs inside the connector-creation hook on the engine's
    own construction path, so anything that escapes costs the boot. A class whose
    every attribute read raises is the pathological case; the answers must still be
    a verdict (not installed, not armed), with a reason, counted as a plugin error -
    never a half-booted tier that believes it is fine.
    """
    class Unreadable:
        def __getattr__(self, name):
            raise RuntimeError(f"cannot read {name}")

    report = {"installed": True, "live_check": {"ok": True, "reason": None},
              "reason": None, "switch": {}, "methods": {}}
    boot_report = backport.boot_from_connector(Unreadable(), report, object())

    assert boot_report["guard"] is False, "an unreadable class is not a guarded one"
    assert boot_report["tier"] is False, "and therefore not an armable one"
    assert backport.is_armed() is False
    assert boot_report["refusals"], "and it says why"
    assert boot_report["line"].startswith("BOOT[auto]")
    assert snapshot._state["errors"] >= 1, "the swallowed failure is counted"


# --------------------------------------------------------------------------
# the one line, and the payloads that carry it
# --------------------------------------------------------------------------

def test_the_boot_line_states_every_verdict_and_the_mode_in_one_line():
    """One line, readable from `/tmp/vllm-serve.log` with no curl and no memory.

    The format is the contract: mode, then the three verdicts, then the fault
    count, then the lists. It is what the README tells an operator to grep, so it
    has to survive a refactor - and the `not_covered=` / `refused=` tails exist so
    that a bare `guard=YES` can never imply more coverage than the guard has.
    """
    _report, boot_report, _cls, _manager = boot()
    line = boot_report["line"]
    assert line.startswith("BOOT[auto] backport=YES guard=YES tier=ARMED faults=0")
    assert "guard_over=" in line and "not_covered=" in line
    assert "\n" not in line, "one line, or the grep loses it"
    assert len(line) < 2000, "bounded, or it floods the engine log"

    # Each label appears EXACTLY ONCE. This caught a real bug: the guard tail was
    # emitted as `faults=POST /plugins/kvtier/guard {"op":"captures"}`, so the line
    # carried two `faults=` labels and the second was not a number - a grep for the
    # fault count could read the pointer instead. `faults=0` is the only faults label,
    # and the captures pointer is its own label.
    for label in ("backport=", "guard=", "tier=", "faults=", "guard_over=",
                  "not_covered=", "captures="):
        repeat = line.count(label)
        assert repeat <= 1, f"{label} appears {repeat}x"
    assert line.count("faults=") == 1
    assert "faults=POST" not in line, "the pointer must not masquerade as a count"

    section = backport.section()
    assert section["boot"]["line"] == line
    assert section["boot"]["ran"] is True

    # A refusal appears in the line too - the failure path is greppable as well.
    cls = TB._six_group_class(drift="_select_eager_blocks_to_store")
    _r, refused, _c, _m = boot(cls, TB._production_manager(cls))
    assert "refused=" in refused["line"]


def test_the_refusal_tails_are_bounded_and_say_how_much_was_left_out():
    """A truncated line must never read as a complete one.

    A boot with a dozen refusals cannot be allowed to write a 20 KB log line, and a
    line that quietly drops the eleventh refusal is worse than no line: the operator
    fixes the three they can see and concludes the boot is fine. So items are
    capped, characters are capped, and the count of what was left out is printed.
    """
    boot_report = {"mode": "auto", "backport": False, "guard": False,
                   "tier": False, "faults_at_boot": 0, "nested": [],
                   "refusals": [f"gate {i}: {'x' * 400}"
                                for i in range(backport.MAX_BOOT_ITEMS + 4)],
                   "not_covered": [f"m{i}" for i in range(10)],
                   "guard_armed": False}
    line = backport.boot_line(boot_report)
    assert len(line) < 2000
    assert f"refused_more={4}" in line
    assert f"not_covered_more={10 - backport.MAX_BOOT_ITEMS}" in line
    # Each item is clipped at the cap: the prefix + 400 x's can only appear as at
    # most `MAX_BOOT_ITEM_CHARS` characters worth of that item.
    assert "x" * (backport.MAX_BOOT_ITEM_CHARS - 8) in line, "items are clipped"
    assert "x" * (backport.MAX_BOOT_ITEM_CHARS + 1) not in line
    assert line.count("gate 0:") == 1 and "gate 6:" not in line


def test_a_process_with_no_tier_to_boot_still_writes_the_line():
    """A worker rank and a MultiConnector are normal, but must not be SILENT.

    An operator who greps for `BOOT[` and finds nothing cannot distinguish "the
    plugin decided not to arm" from "the plugin never ran" - the exact confusion the
    one line exists to remove. So the no-tier paths emit the same line shape with
    the reason, and keep `boot.ran` false so the payload cannot claim a boot.
    """
    boot_report = backport.boot_note("tier: not armed, this process builds no "
                                     "scheduler_manager (worker role)")
    assert boot_report["tier"] is False
    assert boot_report["line"].startswith("BOOT[auto] backport=NO guard=NO")
    assert "worker role" in boot_report["line"]
    section = backport.section()
    assert section["boot"]["ran"] is False, "no boot happened, whatever the line says"
    assert section["boot"]["line"] == boot_report["line"]


def test_the_status_payload_names_the_boot_and_the_ring_to_read():
    """Documentation-as-payload: the status text must send you to the fault ring.

    A guard-armed default changes what "the tier is up" means - `armed: true` no
    longer implies the tier is doing work, because an exception is a decline now.
    So the payload an operator (and the proxy host's reader) reads has to point at
    the captures, or the default trades a crash for an invisible failure.
    """
    from kvtier_dev_plugin import endpoint

    payload = endpoint.status_payload()
    tier_note = payload["tier"]["note"]
    assert config.ENV_BOOT_INERT in tier_note, "names the opt-out"
    assert "pr56810.boot" in tier_note, "and where the verdict lives"
    assert "armed: false" in tier_note, "says what a refusal looks like now"
    guard_note = payload["development_facilities"]["guard_mode"]
    assert "captures" in guard_note, "the ring is named where the facilities are"
    assert "BOOT" in guard_note and config.ENV_BOOT_INERT in guard_note
