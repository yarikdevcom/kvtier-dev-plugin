"""Test isolation: DEVELOPMENT state must not outlive the test that set it.

Why this file exists
--------------------
A suite that is green in one file order and red in another proves less than it
appears to: after a change, "green" would no longer be evidence that the change
was innocent. This repo had exactly that, and the mechanism was the one this
project has now been bitten by twice in production - **state outliving the thing
that set it** (the `intentional_stop_in_progress` latch that silently disabled
crash detection for 25,945 watchdog cycles; an offer loop fed by per-request
state nobody cleared). Here it was `snapshot._state["errors"]`, a process-wide
ledger that every "must never raise into the scheduler" test legitimately adds
to: a file that never reset it handed its ledger of 13 to the next file, and
`test_auto_boot.py`'s "a captured tier fault is not a plugin bug" assertion -
which demands a ledger of exactly `0` - failed only when
`tests/test_disk_restore.py` happened to run first.

The fix lives in `tests/conftest.py`: `reset_dev_state` is **autouse**. This
file is the guard that keeps it that way.

Why the guard is an ordered PAIR
--------------------------------
The only way to prove a reset runs *unasked* is to dirty the state in one test
and look at it in the next one while asking for nothing. Both halves live in one
file because pytest runs tests in definition order within a file, and a
cross-file ordering assumption is precisely the bug being guarded against. The
module-level flag below makes the pairing loud: if a `-k` selection or a
reordering plugin separates the two, the second one fails and says why instead
of quietly passing on state it never had to inherit.

There are TWO behavioural PAIRS, because there are two things worth leaking: the
counters a neighbour reads as zero, and the ENABLEMENT of the shadow verifier,
which decides whether a later test reads a GPU block at all. The third test
proves the mechanism underneath the fixture itself - that the state snapshot
`reset_dev_state` restores is a DEEP copy, so a nested in-place write cannot
survive the teardown - because that leak is invisible to any behavioural test
and would otherwise be re-introduced by a one-word change to `conftest`.
"""
from __future__ import annotations

import conftest  # the shared fixtures themselves (tests/ is on sys.path)
import pytest

from kvtier_dev_plugin import disk_shadow, disk_spill, guard_mode, snapshot, trace_ring

#: Set by the dirtying test below, read by the guard test that follows it.
_trap_armed = {"dirty_ran": False}
#: The same pairing flag for the second trap pair below (nested writes and the
#: shadow verifier's enablement), so a `-k` selection cannot separate THAT pair too.
_nested_trap_armed = {"dirty_ran": False}


def test_a_test_may_dirty_the_dev_state_and_never_clean_up():
    """THE TRAP. Three pieces of dev-state dirtied, zero cleanup, no fixtures.

    Deliberately the three shapes that leak: a CONTAINED error in the ledger (the
    counter another test reads as zero), the guard's fault ring (the counter the
    BOOT line prints), and an ARMED trace ring (a switch another test reads as
    off). Nothing here returns anything to the modules: this test is the leaker,
    and the next one is the proof that leaking is not survivable.
    """
    snapshot._error(RuntimeError("a contained tier fault, not a plugin bug"),
                    "isolation trap")
    guard_mode._state["caught"] += 1  # a real capture needs a wrapped class;
    # this test is about the LEDGER, and the counter the BOOT line prints is the
    # one an operator reads as "0 faults", so it is the one to dirty.
    trace_ring.arm()
    trace_ring.record("isolation trap")

    assert snapshot._state["errors"] >= 1, "the trap itself must dirty the ledger"
    assert guard_mode.fault_count() >= 1
    assert trace_ring.is_armed() is True
    assert trace_ring.counters()["live"] >= 1
    _trap_armed["dirty_ran"] = True


def test_the_next_test_finds_the_state_clean_without_being_asked():
    """THE GUARD. This test requests NO fixture, so anything it sees was inherited.

    If `reset_dev_state` stops being autouse the ledger arrives as 1, the guard
    ring arrives with a fault in it, and the trace ring arrives ARMED - the exact
    inheritance that made this suite order-dependent once already.
    """
    assert _trap_armed["dirty_ran"], (
        "the isolation pair was separated (a -k selection, or a reordering "
        "plugin): the dirtying test that arms this guard did not run, so this "
        "test would prove nothing")
    assert snapshot._state["errors"] == 0, (
        "dev-state leaked: tests/conftest.py::reset_dev_state must stay autouse")
    assert guard_mode.fault_count() == 0, (
        "the guard's fault ring leaked into the next test")
    assert trace_ring.is_armed() is False, (
        "an armed trace ring leaked into the next test")
    assert trace_ring.counters()["live"] == 0, (
        "trace-ring contents leaked into the next test")


def test_a_test_may_arm_the_shadow_verifier_and_never_clean_up():
    """THE SHADOW TRAP: the shadow's own state, dirtied, with zero cleanup.

    The pair above dirties top-level counters. This one dirties the pieces of
    `disk_shadow` that decide what a LATER test is allowed to do: the enablement latch
    (a travelled latch is a test reading KV bytes off a device it never armed), the
    retained GPU block ids, the reuse-witness map, the comparison counters, and the
    log-once note - a latched `multi-region` note silently swallows the MULTI-REGION
    line the next test asserts, which is exactly how a suite becomes green in one file
    order and red in another. It also writes through a state SNAPSHOT in the shape the
    spill path actually writes (`digest_groups[key][group] = n`, a nested in-place
    write), because that is the shape a `dict(_state)` snapshot cannot undo - see the
    mechanism test below.
    """
    disk_spill._state["digest_groups"]["aabbccddeeff0011"] = {7: 3}
    disk_spill._state["refused"]["no_key_for_slot"] += 1
    disk_shadow._state["latch"] = True
    disk_shadow._state["latch_set"] = True
    disk_shadow._state["notes"]["multi-region"] = True
    disk_shadow._state["retained"][999] = {"slots": {1: 2}, "ts": 0.0}
    disk_shadow._state["mismatch_blocks"][4242] = True
    disk_shadow._state["counters"]["mismatch_stable"] = 7

    assert disk_spill._state["digest_groups"] == {"aabbccddeeff0011": {7: 3}}
    assert disk_shadow.effective_enabled() is True
    assert "multi-region" in disk_shadow._state["notes"]
    assert disk_shadow.counters()["mismatch_stable"] == 7
    _nested_trap_armed["dirty_ran"] = True


def test_the_next_test_finds_the_shadow_clean_without_being_asked():
    """THE SHADOW GUARD. This test requests NO fixture, so anything it sees travelled.

    `digest_groups` is the witness for a cross-group digest collision, so a stale entry
    makes a clean boot report a collision it never had; the shadow's latch is an
    ENABLEMENT. The whole point of the pair is that neither survives: a suite that is
    green only because the previous file happened to reset something is not evidence
    that a change was innocent.
    """
    assert _nested_trap_armed["dirty_ran"], (
        "the shadow isolation pair was separated (a -k selection or a reordering "
        "plugin): the trap that arms this guard did not run")
    assert disk_spill._state["digest_groups"] == {}, (
        "a nested spill map travelled: a clean boot would report a group collision "
        "it never had")
    assert disk_spill.counters()["refused"]["no_key_for_slot"] == 0
    assert disk_shadow.effective_enabled() is False, (
        "the shadow verifier's enablement latch travelled: an unarmed test would have "
        "read device memory it never asked for")
    assert "multi-region" not in disk_shadow._state["notes"], (
        "the notice-once latch leaked: the next MULTI-REGION boot logs nothing")
    section = disk_shadow.section()
    assert section["latch"] is False
    assert disk_shadow.enable_state()["latch_set"] is False
    assert section["retained_events_now"] == 0
    assert section["mismatch_blocks_tracked"] == 0
    assert disk_shadow.counters() == dict.fromkeys(disk_shadow._COUNTER_KEYS, 0)
    assert section["regions"] == {"order_rule": "sorted(name)", "count": 0,
                                  "names": [], "truncated": False,
                                  "multi_region": False,
                                  "total_bytes_per_block": None, "widths": None,
                                  "source": None}


def test_the_state_snapshot_cannot_be_dirty_through_the_original():
    """THE MECHANISM, proved directly because no test BODY can observe it.

    `reset_dev_state` saves each module's state at setup and restores it at teardown,
    and a `dict(_state)` save is SHALLOW: every nested dict in a module's state is the
    SAME OBJECT the test mutates, so the teardown's `update(saved)` puts the dirtied
    object back. Measured across the suite's state dicts, an in-place write to
    `disk_shadow._state["counters"]`, `disk_spill._state["refused"]`,
    `disk_store._state["refused"]` and `disk_index._state["dropped"]` all survive that
    restore - and only the NEXT test's `reset_state_for_tests()` removes it, which is
    why no behavioural pair can catch this: the dirty window is closed before any test
    body runs. So it is latent, and it stays latent only while every module's own reset
    happens to rebuild every nested dict it owns. A reset that forgets one key, or a
    module added later, turns it into a real leak - so the SNAPSHOT is the right place
    to be defenceless-proof, and this test is what keeps `conftest._isolate` a deep
    copy.
    """
    sentinel = "a write straight through the snapshot"
    saved = conftest._isolate(disk_shadow._state)
    disk_shadow._state["counters"]["mismatch_stable"] = sentinel
    assert saved["counters"]["mismatch_stable"] != sentinel, (
        "conftest._isolate has gone shallow: the teardown restore would hand the next "
        "test the dict the previous one mutated in place")
    # And the same for a map whose VALUES are containers, the shape `retained` is.
    saved_retained = conftest._isolate(disk_shadow._state)
    disk_shadow._state["retained"][7] = {"slots": {1: 2}, "ts": 0.0}
    assert 7 not in saved_retained["retained"], (
        "conftest._isolate copied the map but not the objects inside it")
    disk_shadow._state["retained"].pop(7, None)


def test_the_dev_state_reset_is_autouse_not_opt_in():
    """The pair above, but runnable ALONE: inspect the fixture, not a neighbour.

    `reset_dev_state` is deliberately not something a file remembers to ask for -
    that is what armed the trap for every file that forgot. The attribute read is
    pytest's own fixture marker; it is checked for existence FIRST so that a
    pytest that renames its internals fails this test with instructions rather
    than with a spurious AttributeError.
    """
    definition = conftest.reset_dev_state
    marker = (getattr(definition, "_fixture_function_marker", None)
              or getattr(definition, "_pytestfixturefunction", None))
    assert marker is not None, (
        "pytest changed its fixture-marker internals: delete THIS test and rely "
        "on the behavioural pair above - do not weaken the pair")
    assert marker.autouse is True, (
        "tests/conftest.py::reset_dev_state stopped being autouse: an opt-in "
        "reset leaves every file that forgets to request it inheriting state")


@pytest.mark.parametrize("name", [
    "reset_dev_state",          # the dev-state reset itself
    "clean_env",                # the plugin must start inert for every test
])
def test_the_fixtures_that_make_a_test_self_contained_are_autouse(name):
    """Every fixture whose job is to make a test self-contained is autouse.

    A NON-autouse "cleanup" fixture is a request a file can forget, and the
    forgetting is invisible until the file order changes. Same rule for the
    environment: a test that inherits `VLLM_E2_TIER_DEV` from its neighbour is
    testing a plugin posture the author never chose.
    """
    definition = getattr(conftest, name)
    marker = getattr(definition, "_fixture_function_marker", None)
    assert marker is not None, f"conftest.{name} is not a pytest fixture"
    assert marker.autouse is True, f"conftest.{name} must stay autouse"
