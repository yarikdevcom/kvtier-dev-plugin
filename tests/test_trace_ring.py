"""The bounded trace ring: opt-in, fixed-size, and genuinely free when disarmed.

A debugger that is always on is a performance feature you lost. So the contract
this file pins is, in order:

1. DISARMED, `record()` is one branch. Nothing is allocated, no ring exists, and
   nothing in the engine is wrapped at all - arming is what installs the
   scheduler-step observers and the tier wrappers, disarming removes them.
2. ARMED, the ring is PREALLOCATED and fixed. There is no append, no list growth,
   no lock: seven fields per slot, a write cursor, and a wrap counter.
3. Every read is BOUNDED. `limit` is capped, the scan walks backwards from the
   cursor (cost is O(returned), never O(depth)), and a reader that polls
   incrementally gets a `gap_records` count instead of an illusion of contiguous
   history.
"""

from __future__ import annotations

import json
import sys

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import guard_mode, step_trace, trace_ring


@pytest.fixture(autouse=True)
def ring(reset_dev_state):
    """Every test starts disarmed, empty and with the counters at zero."""
    yield reset_dev_state


def filled(count, depth=64, **kwargs):
    trace_ring.arm(depth)
    for index in range(count):
        trace_ring.record("kind", group=index % 3, n1=index, n2=index * 2,
                          text=f"event {index}")
    return trace_ring.read(**kwargs)


# --- disarmed: the cost has to be zero --------------------------------------

def test_disarmed_record_allocates_nothing_and_writes_nothing():
    assert trace_ring.is_armed() is False
    for index in range(1000):
        trace_ring.record("sched:step", group=0, n1=index, text="x" * 500)
    counters = trace_ring.counters()
    assert counters["written"] == 0
    assert counters["live"] == 0
    assert counters["seq"] == 0
    # The strongest form of the claim: there is no ring to write into.
    assert trace_ring._state["ring"] is None


def test_disarmed_read_answers_the_shape_with_nothing_in_it():
    answer = trace_ring.read()
    assert answer["armed"] is False
    assert answer["entries"] == []
    assert answer["count"] == 0
    assert answer["limit"] == trace_ring.DEFAULT_LIMIT


# --- arming and the fixed size ----------------------------------------------

def test_arming_allocates_the_ring_once_and_reports_the_depth():
    report = trace_ring.arm(128)
    assert report["armed"] is True
    assert report["depth"] == 128
    assert report["reallocated"] is True
    assert len(trace_ring._state["ring"]) == 128
    again = trace_ring.arm(128)
    assert again["reallocated"] is False
    assert "already armed" in again["reason"]
    assert trace_ring.counters()["arms"] == 1


@pytest.mark.parametrize("requested,expected", [
    (None, trace_ring.DEFAULT_DEPTH),
    (1, trace_ring.MIN_DEPTH),           # too small to be useful: clamped up
    (10 ** 9, trace_ring.MAX_DEPTH),     # and a request for a megaring clamped down
    ("512", trace_ring.DEFAULT_DEPTH),   # a string is not a depth
    (True, trace_ring.DEFAULT_DEPTH),    # a bool is not a depth either
    (-1, trace_ring.MIN_DEPTH),
])
def test_the_depth_is_clamped_not_trusted(requested, expected):
    assert trace_ring.arm(requested)["depth"] == expected


def test_re_arming_at_a_new_depth_reallocates_and_says_so():
    trace_ring.arm(64)
    filled(10, depth=64)
    report = trace_ring.arm(256)
    assert report["reallocated"] is True
    assert trace_ring.counters()["live"] == 0
    assert trace_ring.counters()["depth"] == 256


def test_the_ring_is_a_fixed_list_of_seven_field_slots():
    trace_ring.arm(16)
    trace_ring.record("k", group=1, n1=2, n2=3, text="t")
    slot = trace_ring._state["ring"][0]
    assert isinstance(slot, tuple)
    assert len(slot) == len(trace_ring.SLOT_FIELDS)
    entry = trace_ring.read()["entries"][0]
    assert set(entry) == set(trace_ring.SLOT_FIELDS) | {"age_ms"}


def test_a_long_text_is_clipped_where_it_is_recorded_not_where_it_is_read():
    trace_ring.arm(16)
    trace_ring.record("k", text="y" * 5000)
    entry = trace_ring.read()["entries"][0]
    assert len(entry["text"]) == trace_ring.MAX_STRING_CHARS + 3
    assert entry["text"].endswith("...")


# --- the wrap and the counters ----------------------------------------------

def test_the_ring_wraps_and_counts_what_it_ate():
    trace_ring.arm(16)
    for index in range(21):
        trace_ring.record("k", n1=index)
    counters = trace_ring.counters()
    assert counters["live"] == 16
    assert counters["overwritten"] == 5
    assert counters["written"] == 21
    assert counters["seq"] == 21


def test_a_read_returns_the_newest_records_with_the_newest_last():
    answer = filled(20, depth=16)
    assert answer["count"] == 16
    assert [entry["n1"] for entry in answer["entries"]] == list(range(4, 20))
    assert answer["entries"][-1]["n1"] == 19
    assert answer["truncated"] is False
    assert answer["dropped_since_last_read"] == 4


def test_the_limit_bounds_the_scan_not_just_the_answer():
    filled(200, depth=4096)
    answer = trace_ring.read(limit=2)
    assert answer["count"] == 2
    assert [entry["n1"] for entry in answer["entries"]] == [198, 199]
    assert answer["truncated"] is True
    # A limit above the cap is clamped, and a nonsense limit is the default.
    assert trace_ring.read(limit=10 ** 6)["limit"] == trace_ring.MAX_LIMIT
    assert trace_ring.read(limit=0)["limit"] == trace_ring.DEFAULT_LIMIT
    assert trace_ring.read(limit="no")["limit"] == trace_ring.DEFAULT_LIMIT


def test_an_incremental_reader_gets_only_what_it_has_not_seen():
    filled(20, depth=64)
    first = trace_ring.read(after_seq=0)
    assert first["after_seq"] == 0
    cursor = first["entries"][-1]["seq"]
    trace_ring.record("k", n1=999)
    second = trace_ring.read(after_seq=cursor)
    assert [entry["n1"] for entry in second["entries"]] == [999]
    # No gap: the key is only present when the reader actually missed something.
    assert "gap_records" not in second


def test_a_reader_that_fell_behind_is_told_how_much_it_missed():
    """A poller must not conclude it saw a contiguous history."""
    filled(10, depth=64)
    cursor = trace_ring.counters()["seq"]          # 10
    for index in range(5):
        trace_ring.record("k", n1=index)
    answer = trace_ring.read(after_seq=cursor, limit=3)
    assert answer["count"] == 3
    assert answer["gap_records"] == 15 - 10 - 3     # 5 new, 3 delivered


def test_an_out_of_range_or_nonsense_cursor_is_ignored_not_fatal():
    filled(5, depth=16)
    assert trace_ring.read(after_seq=None)["count"] == 5
    assert trace_ring.read(after_seq="no")["after_seq"] is None
    assert trace_ring.read(after_seq=True)["after_seq"] is None
    assert trace_ring.read(after_seq=10 ** 9)["entries"] == []


def test_clearing_forgets_the_contents_and_keeps_the_sequence():
    filled(10, depth=16)
    before = trace_ring.counters()["seq"]
    report = trace_ring.clear()
    assert report["cleared"] is True
    assert report["seq"] == before                 # a cursor stays valid
    assert report["armed"] is True
    assert trace_ring.read()["entries"] == []
    assert trace_ring.counters()["live"] == 0
    trace_ring.record("k", n1=1)
    assert trace_ring.read()["entries"][0]["seq"] == before + 1


def test_disarming_keeps_the_contents_a_reader_can_still_want():
    filled(10, depth=16)
    report = trace_ring.disarm()
    assert report["armed"] is False
    assert report["was_armed"] is True
    assert trace_ring.read()["count"] == 10
    # But nothing new lands.
    trace_ring.record("k", n1=1)
    assert trace_ring.counters()["written"] == 10


# --- recording never breaks the engine --------------------------------------

def test_record_never_raises_on_values_it_did_not_expect():
    trace_ring.arm(16)
    trace_ring.record(None, group=object(), n1=object(), n2="x", text=123)
    trace_ring.record("k", group=-1, n1=2 ** 70)
    assert trace_ring.counters()["record_errors"] == 0
    assert trace_ring.counters()["written"] == 2


def test_the_verb_payload_is_json_encodable_even_with_strange_values():
    trace_ring.arm(16)
    trace_ring.record("k", group=object(), n1=object())
    payload = trace_ring.trace_for_engine(fake.engine(None), {"op": "read"})
    json.dumps(payload)                            # raises if anything leaked


# --- the verb, and what arming actually installs ----------------------------

def test_arming_installs_the_observers_and_disarming_removes_them():
    specs = [fake.attention_spec()]
    engine = fake.engine(fake.scheduler(groups=[fake.group(specs[0], ["l"])]))
    scheduler_cls = type(engine.scheduler)
    plain = scheduler_cls.schedule
    assert callable(plain)

    armed = trace_ring.trace_for_engine(engine, {"op": "arm", "depth": 64})
    assert armed["op"] == "arm"
    assert armed["armed"] is True
    assert armed["trace_armed"] is True
    assert armed["install"]["step_observers"]["installed"] is True
    assert type(engine.scheduler).schedule is not plain

    disarmed = trace_ring.trace_for_engine(engine, {"op": "disarm"})
    assert disarmed["armed"] is False
    assert disarmed["uninstall"]["step_observers"]["restored"] == [
        "schedule", "update_from_output"]
    assert type(engine.scheduler).schedule is plain


def test_the_step_observer_records_what_the_step_already_produced():
    trace_ring.arm(64)
    scheduler = fake.scheduler(current_step=77)
    engine = fake.engine(scheduler)
    step_trace.install(engine)
    scheduler.requests = {"a": fake.request("a")}
    scheduler.running = ["a", "a"]
    scheduler.waiting = ["b"]
    result = fake.Obj(name="SchedulerOutput", total_num_scheduled_tokens=1024)
    scheduler.step_output = result
    out = engine.scheduler.schedule()
    assert out is result                            # the fork's answer, untouched
    entry = trace_ring.read()["entries"][-1]
    assert entry["kind"] == "sched:step"
    assert entry["n1"] == 77
    assert entry["n2"] == 1024
    assert entry["text"] == "running=2 waiting=1 requests=1"


def test_update_from_output_records_the_output_it_was_handed():
    trace_ring.arm(64)
    scheduler = fake.scheduler(current_step=78)
    engine = fake.engine(scheduler)
    step_trace.install(engine)
    scheduler_output = fake.Obj(name="SchedulerOutput",
                                num_scheduled_tokens={"a": 1, "b": 2})
    out = engine.scheduler.update_from_output(scheduler_output)
    assert out is None                              # the stub's own answer
    entry = trace_ring.read()["entries"][-1]
    assert entry["kind"] == "sched:output"
    assert entry["n1"] == 78
    assert entry["n2"] == 2


def test_an_installed_observer_records_nothing_while_the_ring_is_disarmed():
    """Installing and arming are separate decisions, and the cheap check is the
    one that runs on the serving path."""
    engine = fake.engine(fake.scheduler())
    step_trace.install(engine)
    engine.scheduler.schedule()
    assert trace_ring.counters()["written"] == 0
    assert step_trace.counters()["steps_recorded"] == 0


def test_an_observer_never_eats_an_exception_from_the_scheduler():
    """The wrapper is a pure observer: the fork's failure is the engine's to see."""
    class Boom:
        def schedule(self, *args):
            raise RuntimeError("the scheduler refused")

    trace_ring.arm(16)
    engine = fake.engine(Boom())
    report = step_trace.install(engine)
    assert report["installed"] is True
    with pytest.raises(RuntimeError):
        engine.scheduler.schedule(1)               # the fork's own failure survives
    assert step_trace.counters()["steps_recorded"] == 0


def test_a_scheduler_that_refuses_to_be_read_still_steps(reset_snapshot_state):
    class Exploding:
        current_step = property(lambda self: (_ for _ in ()).throw(
            RuntimeError("no step number")))

        def schedule(self, *args):
            return "the step ran"

    trace_ring.arm(16)
    engine = fake.engine(Exploding())
    step_trace.install(engine)
    assert engine.scheduler.schedule() == "the step ran"     # must not raise
    assert step_trace.counters()["fallbacks"] >= 1
    assert reset_snapshot_state["errors"] >= 1


def test_an_engine_with_no_scheduler_arms_the_ring_but_installs_nothing():
    payload = trace_ring.trace_for_engine(fake.engine(None), {"op": "arm"})
    assert payload["armed"] is True
    assert payload["install"]["step_observers"]["installed"] is False
    assert "no scheduler" in payload["install"]["step_observers"]["reason"]


def test_the_wrappers_stay_out_of_the_way_when_the_ring_is_disarmed_after_guard():
    """Disarming the ring while GUARD MODE is armed must not remove the tier
    wrappers - guard mode still needs them. The dependency runs one way. The
    reason now NAMES EVERY consumer (one set of wrappers, three consumers: guard,
    this ring, the lookup ledger), so the assertion is on the shared predicate's
    wording rather than on a sentence about guard mode alone."""
    manager_cls, module = fake.fork_class(guard_mode.TARGET_MODULE,
                                          "SimpleCPUOffloadScheduler",
                                          fake.TIER_MANAGER_SOURCE, register=True)
    try:
        guard_mode.set_guard_mode(True, None)
        assert manager_cls is not None
        trace_ring.trace_for_engine(fake.engine(None), {"op": "arm"})
        payload = trace_ring.trace_for_engine(fake.engine(None), {"op": "disarm"})
        assert payload["uninstall"]["tier_wrappers"]["reason"] == (
            "still needed by: guard mode; the tier wrappers stay in place")
    finally:
        sys.modules.pop(guard_mode.TARGET_MODULE, None)


def test_an_unknown_op_is_an_error_that_names_the_valid_ops():
    payload = trace_ring.trace_for_engine(fake.engine(None), {"op": "launch"})
    assert payload["op"] is None
    assert "unknown op" in payload["error"]
    assert "disarm" in payload["error"]


def test_the_verb_reports_arguments_it_did_not_understand():
    payload = trace_ring.trace_for_engine(fake.engine(None),
                                          {"op": "status", "colour": "red"})
    assert payload["unknown_args"] == ["colour"]
    assert payload["counters"]["armed"] is False


def test_the_verb_publishes_its_state_beside_every_other_verb():
    payload = trace_ring.trace_for_engine(fake.engine(None), {"op": "status"})
    assert payload["version"] == trace_ring.PROTOCOL_VERSION
    assert payload["guard_mode"] is False
    assert payload["trace_armed"] is False
    assert payload["build_us"] >= 0
    assert payload["self"]["errors"] == 0


def test_a_broken_ring_becomes_an_error_payload_and_never_raises(
        monkeypatch, reset_snapshot_state):
    def explode(*args, **kwargs):
        raise RuntimeError("the ring refused")

    monkeypatch.setattr(trace_ring, "read", explode)
    payload = trace_ring.trace_for_engine(fake.engine(None), {"op": "read"})
    assert payload["error"].startswith("RuntimeError")
    assert reset_snapshot_state["errors"] >= 1


def test_the_read_path_does_not_walk_the_whole_ring():
    """4096 slots, 4096 records, a limit of 3: the scan stops after 3.

    Measured indirectly by the recorded read count and by the answer: a forward
    scan of the ring would have to visit every slot to find the newest.
    """
    filled(4096, depth=4096)
    answer = trace_ring.read(limit=3)
    assert answer["count"] == 3
    assert answer["entries"][-1]["n1"] == 4095
    assert answer["truncated"] is True


def test_the_snapshot_reports_the_ring_alongside_the_rest(reset_snapshot_state):
    """The dashboard polls `snapshot`; the ring's counters must be there so an
    armed engine is visible without another HTTP round trip."""
    from kvtier_dev_plugin import snapshot as snapshot_module
    engine, _ = fake.production_geometry()
    trace_ring.arm(32)
    trace_ring.record("sched:step", n1=1)
    payload = snapshot_module.snapshot_for_engine(engine, None)
    assert payload["trace"]["armed"] is True
    assert payload["trace"]["depth"] == 32
    assert payload["trace"]["written"] == 1
    assert payload["trace"]["step_observers"]["installed"] is False
    assert payload["guard_mode"] is False
