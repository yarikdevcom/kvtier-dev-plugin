"""GUARD MODE: a development-only facility that turns a tier fault into data.

Armed, the KV-tier entry points are wrapped so that an `Exception` becomes

  * a CAPTURED RECORD (method, exception type, message, bounded traceback, the
    tier's own geometry at that instant, the arguments involved), and
  * a SAFE DECLINE - the value the fork returns when it simply found nothing to
    do - so the step continues instead of the engine dying.

Three properties are non-negotiable, and most of this file is about them:

1. `except Exception`, NEVER `except BaseException`. A KeyboardInterrupt, a
   SystemExit or an asyncio cancellation arriving inside a tier call is the
   engine's signal, not this plugin's; swallowing one would make the process
   unhailable. `test_base_exceptions_always_continue` pins it.
2. OFF by default, with no env var that turns it on. It is armed by a POST that
   needs a token, its state is echoed as `guard_mode` by every verb, and the only
   way to be armed is to have asked.
3. A decline is only invented where the fork has a documented "nothing happened"
   value. `update_connector_output` is wrapped CAPTURE-AND-RE-RAISE because a fake
   value there would corrupt state the scheduler trusts.
"""

from __future__ import annotations

import json
import sys

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import config, guard_mode, snapshot

# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def tier(reset_dev_state):
    """The fork's tier class, with the failure switch, plus its instance."""
    cls, module = fake.fork_class(guard_mode.TARGET_MODULE,
                                  guard_mode.TARGET_CLASS,
                                  fake.GUARD_TIER_SOURCE, register=True)
    manager = fake.tier_manager()
    # Same object, the fork's methods: `__class__` re-binding is legal for two
    # plain-Python classes and keeps ONE object carrying both the geometry fields
    # the capture reads and the entry points the guard wraps.
    manager.__class__ = cls
    manager.calls = []                    # what the fork's own __init__ would set
    yield cls, module, manager
    sys.modules.pop(guard_mode.TARGET_MODULE, None)


def arm(engine=None):
    return guard_mode.set_guard_mode(True, engine)


def fail(module, *names):
    module.FAILING.update(names)


# --- the default: off, and un-configurable ----------------------------------

def test_no_env_var_can_arm_guard_mode_and_at_import_it_is_off(monkeypatch):
    """A facility that can mask bugs must not be arming-able from a launcher script.

    `config.py` owns every env var this plugin reads; none of them is a guard. The
    boot DOES arm guard mode (so an unguarded store or lookup never runs), but there
    is no env var that arms it on a boot that did not arm it: the one variable that
    touches the question, `VLLM_E2_TIER_DEV_BOOT_INERT`, can only turn the boot's
    arming OFF. An operator can therefore stop the boot guarding an engine from a
    launcher script, and cannot start it doing so from one.
    """
    assert guard_mode.is_armed() is False
    assert guard_mode._state["arms"] == 0
    # config.py owns every env var this plugin reads. None of them is a guard: an
    # operator cannot arm the facility from a launcher script or a systemd unit.
    env_names = [name for name in dir(config) if name.startswith("ENV_")]
    assert env_names, "config owns no env vars at all?"
    assert not [name for name in env_names if "GUARD" in name.upper()]
    # And the one variable that does touch guard mode is an opt-OUT, so its very
    # name has to say so: a variable called something neutral that armed a dev
    # facility would be this test's failure wearing a different hat.
    assert "INERT" in config.ENV_BOOT_INERT
    # THE 0.2.66 EXCEPTION, NAMED AND PROVED. `VLLM_E2_TIER_DEV_LOOKUP_LEDGER_BOOT_ARM`
    # contains the word because it arms the per-request lookup RING - a recorder in
    # `lookup_ledger.py` that writes no byte, offers nothing and serves nothing. It is
    # named here instead of the filter being narrowed, because a filter narrowed to let
    # one variable through lets the next one through too. What proves the exception is
    # honest is the line after it: with the variable set to its most permissive value,
    # guard mode is STILL not armed - arming the ledger and arming the guard are
    # different
    # questions, and `tests/test_lookup_ledger_boot_arm.py` pins that the ledger step
    # cannot name a posture, an offer switch or an arm writer at all.
    # The exception is a set of config ATTRIBUTE NAMES, because that is what `dir()`
    # enumerates - the env var string is the same variable's other face.
    ledger_switch = {"ENV_LOOKUP_LEDGER_BOOT_ARM"}
    with_guard_arming = [name for name in env_names
                         if any(w in name.upper() for w in
                                ("ARM", "GUARD", "TIER_ARM"))]
    assert set(with_guard_arming) <= ledger_switch, with_guard_arming
    monkeypatch.setenv(config.ENV_LOOKUP_LEDGER_BOOT_ARM, "1")
    assert config.lookup_ledger_boot_arm()["enabled"] is True
    assert guard_mode.is_armed() is False, "the ledger's gate armed guard mode"


def test_a_wrapper_from_a_sibling_module_is_composed_with_not_refused(tier):
    """`backport` wraps `get_num_new_matched_tokens` too. Refusing it would leave
    the most fault-prone entry point unguarded while `arm` still reported "
    exceptions are captured and declined" - a payload worse than a refusal, because
    it is confidently wrong. So a sibling of THIS package is composed with."""
    cls, module, manager = tier
    fork_function = cls.get_num_new_matched_tokens

    def sibling_wrapper(self, request, num_computed_tokens):
        return fork_function(self, request, num_computed_tokens)

    sibling_wrapper.__name__ = "get_num_new_matched_tokens"
    sibling_wrapper.__module__ = "kvtier_dev_plugin.backport"
    cls.get_num_new_matched_tokens = sibling_wrapper

    report = guard_mode.install(cls)
    assert report["nested"] == ["get_num_new_matched_tokens"]
    assert "get_num_new_matched_tokens" not in report["skipped"]
    # Visible long after the arm that created it: the snapshot says the same thing.
    assert guard_mode.section()["nested"] == ["get_num_new_matched_tokens"]

    # The guard is the OUTER layer: armed, the fault is a record and a decline.
    arm()
    module.FAILING.add("get_num_new_matched_tokens")
    assert manager.get_num_new_matched_tokens(fake.request("r1"), 1024) == (0, False)

    # Removing the guard hands the sibling's wrapper back, byte for byte, and the
    # section stops claiming a composition that is gone.
    guard_mode.uninstall(cls)
    assert cls.__dict__["get_num_new_matched_tokens"] is sibling_wrapper
    assert guard_mode.section()["nested"] == []


def test_a_wrapper_owned_by_someone_else_is_still_refused(tier):
    """The rule above is about THIS package. A third party's patch is never
    touched, marked or not."""
    cls, _module, _manager = tier
    fork_function = cls.take_events

    def stranger(self):
        return fork_function(self)

    stranger.__name__ = "take_events"
    stranger.__module__ = "other_plugin.hooks"
    cls.take_events = stranger

    report = guard_mode.install(cls)
    assert "take_events" in report["skipped"]
    assert report["nested"] == []
    assert "other_plugin.hooks" in report["skip_detail"][0]["reason"]
    assert cls.__dict__["take_events"] is stranger


def test_the_install_report_shape_is_stable_when_nothing_is_nested(tier):
    """`nested` is a key the payload always has: a reader must not have to guess
    whether an absent key meant 'no nesting' or 'older plugin'."""
    cls, _module, _manager = tier
    report = guard_mode.install(cls)
    assert report["nested"] == []
    assert guard_mode.section()["nested"] == []
    assert set(report) >= {"installed", "attached", "replaced", "skipped",
                           "skip_detail", "nested", "class", "reason"}


def test_the_status_of_the_facility_is_a_section_of_every_snapshot(tier):
    cls, _module, manager = tier
    engine = fake.engine(fake.scheduler())
    engine.scheduler.connector = fake.Obj(name="SimpleCPUOffloadConnector",
                                          scheduler_manager=manager)
    payload = snapshot.snapshot_for_engine(engine, None)
    assert payload["guard_mode"] is False
    assert payload["guard"]["guard_mode"] is False
    assert payload["guard"]["development_only"] is True
    assert "mask real bugs" in payload["guard"]["warning"]


# --- installing and removing the wrappers ----------------------------------

def test_arming_installs_the_wrappers_on_the_tier_class(tier):
    cls, _module, manager = tier
    before = {name: getattr(cls, name) for name in guard_mode.WRAPPED_METHODS}
    report = guard_mode.install(cls)
    assert report["installed"] is True
    assert report["reason"] is None
    assert sorted(report["attached"]) == sorted(guard_mode.WRAPPED_METHODS)
    for name in guard_mode.WRAPPED_METHODS:
        wrapped = getattr(cls, name)
        assert getattr(wrapped, guard_mode.OWNER_ATTR) == guard_mode.OWNER_TAG
        assert getattr(wrapped, guard_mode.ORIGINAL_ATTR) is before[name]
        assert "GUARD MODE" in wrapped.__doc__


def test_disarming_puts_the_fork_s_own_functions_back_exactly(tier):
    cls, _module, manager = tier
    before = {name: getattr(cls, name) for name in guard_mode.WRAPPED_METHODS}
    guard_mode.install(cls)
    report = guard_mode.uninstall(cls)
    assert report["uninstalled"] is True
    assert sorted(report["restored"]) == sorted(guard_mode.WRAPPED_METHODS)
    for name in guard_mode.WRAPPED_METHODS:
        assert getattr(cls, name) is before[name]


def test_re_installing_rebinds_the_fork_s_function_and_cannot_nest(tier):
    """A reload loop that installed twice must not wrap twice: the second install
    re-binds the ORIGINAL we stored, so one fault is still ONE capture."""
    cls, module, manager = tier
    guard_mode.install(cls)
    second = guard_mode.install(cls)
    assert second["replaced"] and not second["attached"]
    module.FAILING.add("update_connector_output")
    guard_mode._state["armed"] = True
    guard_mode._state["captures"].clear()
    with pytest.raises(ZeroDivisionError):
        # update_connector_output is capture-and-re-raise, so a fault there must
        # surface exactly once however many wrappers are stacked.
        manager.update_connector_output(None)
    assert guard_mode._state["caught"] == 1
    assert len(guard_mode._state["captures"]) == 1


def test_a_foreign_owner_marker_is_refused_not_overwritten(tier):
    cls, _module, _manager = tier
    setattr(cls.take_events, guard_mode.OWNER_ATTR, "someone-else/1")
    report = guard_mode.install(cls)
    assert "take_events" in report["skipped"]
    # A PARTIAL install still says which name it refused and why: `skip_detail`,
    # because `report["reason"]` is only the "nothing was installed" answer.
    detail = {entry["method"]: entry["reason"] for entry in report["skip_detail"]}
    assert "another plugin" in detail["take_events"]
    assert "take_events" in detail
    # Everything else still lands: one contested name is not a reason to install
    # nothing at all.
    assert "reset" in report["attached"]


def test_a_method_defined_outside_the_fork_module_is_refused():
    """Every name exists and is callable, but each was defined HERE, so the install
    must refuse to wrap it rather than assume it is the fork's.

    This is the check that stops the plugin from overwriting something a third
    party owns, while still accepting a fork that legitimately re-declared a
    method (a fork-defined function is the ONLY unmarked thing it will wrap).
    """
    Impostor = type("Impostor", (), {
        name: (lambda self, *args, **kwargs: None)
        for name in guard_mode.WRAPPED_METHODS})
    report = guard_mode.install(Impostor)
    assert report["installed"] is False
    assert report["attached"] == []
    assert sorted(report["skipped"]) == sorted(guard_mode.WRAPPED_METHODS)
    assert sorted(entry["method"] for entry in report["skip_detail"]) == \
        sorted(guard_mode.WRAPPED_METHODS)
    reason = guard_mode._state["skip_reason"] or ""
    assert "nor the fork's own" in reason
    assert "test_guard_mode" in reason            # it names WHERE it was defined
    assert all("nor the fork's own" in entry["reason"]
               for entry in report["skip_detail"])


def test_a_clean_install_leaves_no_skip_reason(tier):
    cls, _module, _manager = tier
    report = guard_mode.install(cls)
    assert report["skip_detail"] == []
    assert guard_mode._state["skip_reason"] is None
    assert guard_mode.section()["skip_detail"] == []


def test_a_missing_method_is_skipped_and_named_rather_than_invented(tier):
    cls, _module, _manager = tier
    del cls.take_events
    report = guard_mode.install(cls)
    assert "take_events" in report["skipped"]
    assert "renamed" in guard_mode._state["skip_reason"]
    detail = {entry["method"]: entry["reason"] for entry in report["skip_detail"]}
    assert "renamed" in detail["take_events"]
    assert "reset" in report["attached"]


def test_arm_without_a_tier_says_nothing_is_wrapped(tier, reset_dev_state):
    """`armed with nothing to guard` is a state an operator must be able to read."""
    sys.modules.pop(guard_mode.TARGET_MODULE, None)
    report = arm(None)
    assert report["armed"] is True
    assert report["install"]["installed"] is False
    assert "SimpleCPUOffloadScheduler" in report["reason"]
    assert guard_mode.is_armed() is True


def test_arming_from_a_live_manager_finds_the_class_by_itself(tier):
    cls, _module, manager = tier
    engine = fake.engine(fake.scheduler())
    engine.scheduler.connector = fake.Obj(name="SimpleCPUOffloadConnector",
                                          scheduler_manager=manager)
    report = arm(engine)
    assert report["install"]["installed"] is True
    assert report["install"]["class"] == "SimpleCPUOffloadScheduler"


# --- the declines, one per method -------------------------------------------

@pytest.mark.parametrize("method,args,expected", [
    ("get_num_new_matched_tokens", ("request", 128), (0, False)),
    ("update_state_after_alloc", ("request", None, 64), None),
    ("prepare_store_specs", (None,), ([], [], [], None)),
    ("request_finished", ("request", [1, 2]), (False, None)),
    ("request_finished_all_groups", ("request", [1, 2]), (False, None)),
    ("take_events", (), ()),
    ("reset", (), False),
])
def test_an_armed_fault_becomes_the_fork_s_own_nothing_happened_value(
        tier, method, args, expected):
    """Each decline is the value the fork returns when it found nothing, NOT an
    invention: manager.py:320, base.py:748-760, manager.py:1152 and
    metadata.py:16-38 are the sources for the shapes below."""
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, method)
    result = getattr(manager, method)(*args)
    assert result == expected
    assert guard_mode._state["declined"] == 1
    record = guard_mode._state["captures"][-1]
    assert record["method"] == method
    assert record["decline"] == "declined"
    assert record["declined"] is True


def test_a_declined_metadata_is_a_fresh_object_from_the_fork_s_own_globals(tier):
    """`build_connector_meta` declines with an EMPTY SimpleCPUOffloadMetadata
    (every field "no work this step"). It is resolved from the ORIGINAL's
    `__globals__`, so a re-import of the fork cannot leave us building a stale
    type - and two declines must not share one mutable object."""
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "build_connector_meta")
    first = manager.build_connector_meta(None)
    second = manager.build_connector_meta(None)
    assert type(first).__name__ == "SimpleCPUOffloadMetadata"
    assert first is not second
    assert first.load_event == -1 and first.store_event == -1
    assert first.load_gpu_blocks == [] and first.store_gpu_blocks == []


def test_a_decline_we_cannot_build_becomes_a_re_raise(tier):
    """No invented metadata type: if the fork renamed it, the exception continues
    on its way and the operator sees the real failure."""
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "build_connector_meta")
    del module.SimpleCPUOffloadMetadata
    with pytest.raises(ZeroDivisionError):
        manager.build_connector_meta(None)
    record = guard_mode._state["captures"][-1]
    assert record["decline"] == "re-raised"
    assert guard_mode._state["re_raised"] == 1


def test_update_connector_output_is_capture_and_reraise_never_a_decline(tier):
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "update_connector_output")
    with pytest.raises(ZeroDivisionError):
        manager.update_connector_output(None)
    assert "update_connector_output" in guard_mode.CAPTURE_ONLY
    assert guard_mode._state["declined"] == 0
    assert guard_mode._state["re_raised"] == 1


def test_a_healthy_call_is_untouched_by_an_armed_guard(tier):
    """Armed guards must not change ONE return value that did not fault."""
    cls, module, manager = tier
    plain = fake.fork_class(guard_mode.TARGET_MODULE + ".plain",
                            "SimpleCPUOffloadScheduler", fake.TIER_MANAGER_SOURCE)
    reference = plain[0]()
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    assert manager.get_num_new_matched_tokens(None, 0) == \
        reference.get_num_new_matched_tokens(None, 0)
    assert manager.request_finished(None, []) == reference.request_finished(None, [])
    assert manager.reset() is reference.reset()
    assert manager.take_events() == reference.take_events()
    assert guard_mode._state["caught"] == 0


# --- BaseException: never swallowed -----------------------------------------

@pytest.mark.parametrize("exception", [
    KeyboardInterrupt, SystemExit, GeneratorExit, BaseException,
])
def test_base_exceptions_always_continue_and_are_never_captured(tier, exception):
    cls, _module, manager = tier
    install_with(cls, "take_events", exception)
    with pytest.raises(exception):
        manager.take_events()
    assert guard_mode._state["caught"] == 0
    assert len(guard_mode._state["captures"]) == 0
    assert guard_mode._state["calls"] == 1


def install_with(cls, name, exception, *args):
    """Make ONE fork method raise, THEN install, so the wrapper really is the
    thing under test (installing first and replacing after would wrap nothing)."""
    setattr(cls, name, _raising(getattr(cls, name), exception, *args))
    report = guard_mode.install(cls)
    assert report["installed"] is True, report
    guard_mode._state["armed"] = True


def _raising(function, exception, *args):
    """Replace a fork method with one that raises `exception(*args)`.

    `__module__` is carried over, so the replacement is a legitimate wrap target:
    the test is about what the WRAPPER does with the fault, not about which class
    the fault came from.
    """
    def replacement(self, *inner, **kwargs):
        raise exception(*args)
    replacement.__name__ = getattr(function, "__name__", "replacement")
    replacement.__module__ = guard_mode.TARGET_MODULE
    return replacement


def test_an_asyncio_cancellation_is_not_a_tier_fault(tier):
    """`CancelledError` is a BaseException in 3.8+, which is exactly why the
    wrapper catches `Exception` and not the wider class."""
    import asyncio
    cls, _module, manager = tier
    install_with(cls, "reset", asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        manager.reset()
    assert guard_mode._state["caught"] == 0


# --- the captures -----------------------------------------------------------

def test_a_capture_records_the_method_the_fault_and_the_tier_s_geometry(tier):
    cls, module, manager = tier
    manager.group_block_sizes = [1024, 1000, 128]
    manager.hash_block_size = 128
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "get_num_new_matched_tokens")
    manager.get_num_new_matched_tokens(fake.request("cmpl-a"), 4096)
    record = guard_mode._state["captures"][-1]
    assert record["method"] == "get_num_new_matched_tokens"
    assert record["exception_type"] == "ZeroDivisionError"
    assert "divided by zero" in record["message"]
    assert "ZeroDivisionError" in record["traceback"]
    assert record["fault_no"] >= 1
    assert record["manager"]["present"] is True
    assert record["manager"]["num_cpu_blocks"] == 4096
    # The arithmetic the fork asserts on, per group, AT THE MOMENT IT FAILED:
    # group 1 (1000 % 128 = 104) is the crash, and the capture names it.
    assert record["manager"]["geometry"]["groups_failing_divisibility"] == [1]
    assert record["manager"]["geometry"]["per_group"][1]["remainder"] == 104
    # And the arguments, bounded, with the request identified by id + counters.
    request_view = record["args"][0]
    assert request_view["request_id"] == "cmpl-a"
    assert request_view["num_tokens"] == 4096
    assert request_view["block_hashes"] == 32
    assert record["args"][1] == 4096


def test_a_capture_of_a_block_container_names_the_ids_involved(tier):
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    blocks = fake.Obj(name="KVCacheBlocks",
                      get_block_ids=lambda: [list(range(200)), [7]])
    fail(module, "update_state_after_alloc")
    manager.update_state_after_alloc(fake.request("cmpl-a"), blocks, 128)
    view = guard_mode._state["captures"][-1]["args"][1]
    assert view["kind"] == "KVCacheBlocks"
    assert view["block_ids_by_group"][0]["len"] == 200
    assert len(view["block_ids_by_group"][0]["head"]) == guard_mode.MAX_IDS
    assert view["block_ids_by_group"][1]["head"] == [7]


def test_a_message_and_a_traceback_are_bounded(tier):
    cls, _module, manager = tier
    install_with(cls, "reset", RuntimeError, "x" * 100000)
    assert manager.reset() is False                 # reset declines: the fault lands
    record = guard_mode._state["captures"][-1]
    assert len(record["message"]) <= guard_mode.MAX_MESSAGE_CHARS + 3
    assert len(record["traceback"]) <= guard_mode.MAX_TRACEBACK_CHARS + 3


def test_the_capture_ring_is_bounded_and_counts_what_it_dropped(tier):
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "reset")
    # `reset` DECLINES, so every call returns instead of raising: the ring is
    # filled by captures, not by exceptions escaping.
    for _ in range(guard_mode.MAX_CAPTURES + 5):
        assert manager.reset() is False
    assert len(guard_mode._state["captures"]) == guard_mode.MAX_CAPTURES
    assert guard_mode._state["captures_dropped"] == 5
    answer = guard_mode.captures(limit=99)
    assert answer["returned"] == guard_mode.MAX_CAPTURES_RETURNED
    assert answer["kept"] == guard_mode.MAX_CAPTURES
    assert answer["dropped"] == 5


def test_captures_reports_the_full_text_only_when_asked_and_bounded(tier):
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "reset")
    manager.reset()
    answer = guard_mode.captures(limit=1, traceback_chars=40)
    record = answer["captures"][0]
    assert len(record["traceback"]) <= 40
    assert record["traceback_truncated"] is True
    assert record["traceback_chars"] > 40
    # An absurd request is clamped, never honoured.
    assert guard_mode.captures(limit=-3, traceback_chars=10 ** 9)[
        "captures"][0]["traceback_chars"] > 0


def test_a_capture_that_cannot_be_built_is_counted_and_never_adds_a_fault(
        tier, monkeypatch, reset_snapshot_state):
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "reset")

    def explode(*args, **kwargs):
        raise RuntimeError("the capture machinery is broken")

    monkeypatch.setattr(guard_mode, "capture", explode)
    assert manager.reset() is False                 # the decline still happens
    assert guard_mode._state["capture_errors"] >= 1
    assert reset_snapshot_state["errors"] >= 1


def test_a_capture_records_into_the_trace_ring_when_it_is_armed(tier):
    from kvtier_dev_plugin import trace_ring
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    trace_ring.arm(16)
    fail(module, "reset")
    manager.reset()
    kinds = [entry["kind"] for entry in trace_ring.read()["entries"]]
    assert "exception" in kinds
    assert guard_mode._state["captures"][-1]["trace_seq"] == \
        trace_ring.counters()["seq"]


def test_the_disarmed_wrapper_swallows_nothing(tier):
    """The DEFAULT shape: installed (because tracing wants it) but not armed. The
    original runs OUTSIDE any try/except of ours, so a fault reaches the engine and
    only a SUCCESS is recorded."""
    cls, module, manager = tier
    guard_mode.install(cls)
    assert guard_mode.is_armed() is False
    fail(module, "reset")
    with pytest.raises(ZeroDivisionError):
        manager.reset()
    assert guard_mode._state["caught"] == 0
    assert len(guard_mode._state["captures"]) == 0


def test_successes_are_recorded_per_step_only_while_tracing_is_armed(tier):
    from kvtier_dev_plugin import trace_ring
    cls, _module, manager = tier
    guard_mode.install(cls)
    manager.get_num_new_matched_tokens(fake.request("cmpl-a"), 0)
    assert trace_ring.counters()["written"] == 0
    trace_ring.arm(16)
    manager.get_num_new_matched_tokens(fake.request("cmpl-a"), 0)
    entry = trace_ring.read()["entries"][-1]
    assert entry["kind"] == "tier:load_offer"
    assert entry["text"] == "cmpl-a"


# --- arming and disarming through the verb ----------------------------------

def test_the_verb_arms_disarms_and_reports_the_state_every_time(tier):
    cls, _module, manager = tier
    engine = fake.engine(fake.scheduler())
    engine.scheduler.connector = fake.Obj(name="SimpleCPUOffloadConnector",
                                          scheduler_manager=manager)
    armed = guard_mode.guard_for_engine(engine, {"op": "arm"})
    assert armed["op"] == "arm"
    assert armed["guard_mode"] is True
    assert armed["development_only"] is True
    assert armed["install"]["installed"] is True
    assert armed["install"]["class"] == "SimpleCPUOffloadScheduler"
    assert sorted(armed["install"]["attached"]) == sorted(
        guard_mode.WRAPPED_METHODS)

    disarmed = guard_mode.guard_for_engine(engine, {"op": "disarm"})
    assert disarmed["guard_mode"] is False
    assert disarmed["changed"] is True
    assert guard_mode._state["installed"] is False


def test_the_armed_spelling_an_operator_thinks_in(tier):
    cls, _module, manager = tier
    on = guard_mode.guard_for_engine(None, {"armed": True})
    assert on["guard_mode"] is True
    off = guard_mode.guard_for_engine(None, {"armed": False})
    assert off["guard_mode"] is False


def test_an_unknown_op_names_the_valid_ones(tier):
    payload = guard_mode.guard_for_engine(None, {"op": "selfdestruct"})
    assert payload["op"] is None
    assert "arm" in payload["error"]
    assert payload["guard_mode"] is False


def test_the_verb_reports_arguments_it_did_not_understand(tier):
    payload = guard_mode.guard_for_engine(None, {"op": "status",
                                                 "colour": "red"})
    assert payload["unknown_args"] == ["colour"]


def test_an_argument_that_is_not_a_dict_still_answers(tier):
    for argument in (None, "arm", 7, ["armed"]):
        payload = guard_mode.guard_for_engine(None, argument)
        assert payload["op"] == "status"
        assert payload["guard_mode"] is False


def test_a_broken_verb_becomes_an_error_payload_and_never_raises(
        tier, monkeypatch, reset_snapshot_state):
    def explode(*args, **kwargs):
        raise RuntimeError("the guard refused")

    monkeypatch.setattr(guard_mode, "section", explode)
    payload = guard_mode.guard_for_engine(None, {"op": "status"})
    assert payload["error"].startswith("RuntimeError")
    assert payload["guard_mode"] is False
    assert reset_snapshot_state["errors"] >= 1


def test_the_section_shape_is_stable_and_labels_what_it_wraps(tier):
    cls, _module, manager = tier
    section = guard_mode.section()
    assert set(section) >= {"guard_mode", "development_only", "warning", "arms",
                            "disarms", "installed", "class", "module", "methods",
                            "declinable", "capture_and_reraise", "captures_preview",
                            "captures_max", "error"}
    assert section["module"] == "vllm.v1.simple_kv_offload.manager"
    assert sorted(section["declinable"]) == sorted(guard_mode.DECINABLE)
    assert section["capture_and_reraise"] == ["update_connector_output"]
    assert json.dumps(section)


def test_the_snapshot_carries_previews_not_full_tracebacks(tier):
    """A dashboard polls the snapshot every few seconds: eight full tracebacks in
    that payload is a megabyte of noise, so the snapshot gets bounded previews and
    the full text stays behind the explicit `captures` op."""
    cls, module, manager = tier
    guard_mode.install(cls)
    guard_mode._state["armed"] = True
    fail(module, "reset")
    for _ in range(5):
        manager.reset()
    engine = fake.engine(fake.scheduler())
    payload = snapshot.snapshot_for_engine(engine, None)
    previews = payload["guard"]["captures_preview"]
    assert len(previews) <= 3
    for preview in previews:
        assert len(preview["traceback_head"]) <= guard_mode.MAX_SECTION_TRACEBACK_CHARS
        assert preview["traceback_chars"] >= len(preview["traceback_head"])
    assert payload["guard"]["declined"] == 5


def test_the_disarm_keeps_the_wrappers_while_tracing_needs_them(tier):
    from kvtier_dev_plugin import trace_ring
    cls, _module, manager = tier
    engine = fake.engine(fake.scheduler())
    engine.scheduler.connector = fake.Obj(name="SimpleCPUOffloadConnector",
                                          scheduler_manager=manager)
    guard_mode.set_guard_mode(True, engine)
    trace_ring.arm(16)
    report = guard_mode.set_guard_mode(False, engine)
    assert "STAY" in report["reason"]
    assert guard_mode._state["installed"] is True
    # With tracing off too, the same call removes everything.
    trace_ring.disarm()
    guard_mode.set_guard_mode(False, engine)
    assert guard_mode._state["installed"] is False


def test_the_guard_state_is_visible_in_the_other_verbs(tier):
    """A guarded development engine must be distinguishable from a production one
    from ANY verb's payload, not only from the guard's own."""
    from kvtier_dev_plugin import kvconfig, kvinspect, trace_ring
    cls, _module, manager = tier
    guard_mode.guard_for_engine(None, {"armed": True})
    engine, _ = fake.production_geometry()
    assert kvconfig.kvconfig_for_engine(engine)["guard_mode"] is True
    assert kvinspect.inspect_for_engine(engine, {})["guard_mode"] is True
    assert trace_ring.trace_for_engine(engine, {"op": "status"})["guard_mode"] is True
    assert snapshot.snapshot_for_engine(engine, None)["guard_mode"] is True
    guard_mode.guard_for_engine(None, {"armed": False})
    assert kvconfig.kvconfig_for_engine(engine)["guard_mode"] is False
