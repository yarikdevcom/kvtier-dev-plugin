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
"""
from __future__ import annotations

import conftest  # the shared fixtures themselves (tests/ is on sys.path)
import pytest

from kvtier_dev_plugin import guard_mode, snapshot, trace_ring

#: Set by the dirtying test below, read by the guard test that follows it.
_trap_armed = {"dirty_ran": False}


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
