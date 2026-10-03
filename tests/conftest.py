"""Shared fixtures. The plugin is importable without vllm installed, so nothing
here stubs an engine: where a test needs an engine it builds a plain object with
the real attribute names (see test_snapshot.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import fakes_debug  # noqa: E402 - the shared fake engine structures

from kvtier_dev_plugin import (  # noqa: E402
    config,
    guard_mode,
    lookup_ledger,
    snapshot,
    step_trace,
    trace_ring,
    worker_diag,
)

ENV_VARS = (config.ENV_ENABLE, config.ENV_TOKEN, config.ENV_TIMEOUT,
            config.ENV_ALLOW, config.ENV_DIAG_DIR, config.ENV_DIAG_EPOCH,
            config.ENV_DIAG_INTERVAL, config.ENV_DIAG_MIN_INTERVAL,
            config.ENV_DIAG_RANK, config.ENV_BOOT_INERT)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with the plugin inert and no token."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(config.ENV_ENABLE, "1")


@pytest.fixture
def reset_snapshot_state():
    """Containment counters are process-wide; keep tests from stacking on them."""
    saved = dict(snapshot._state)
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0, "epoch": None})
    yield snapshot._state
    snapshot._state.update(saved)


@pytest.fixture
def reset_dev_state():
    """The pieces of DEVELOPMENT state, disarmed and zeroed, then restored.

    These are the plugin's only engine-changing facilities, and all four keep
    process-wide state (a ring, a deque, a class marker). Without this a test that
    armed guard mode would leave the wrappers installed on whatever class it
    passed and leave the next test believing guard mode is off.
    """
    saved = {"snapshot": dict(snapshot._state),
             "trace": dict(trace_ring._state),
             "step": dict(step_trace._state),
             "guard": dict(guard_mode._state),
             "worker": dict(worker_diag._state),
             "lookups": dict(lookup_ledger._state)}
    trace_ring.reset_state_for_tests()
    step_trace.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    worker_diag.reset_state_for_tests()
    lookup_ledger.reset_state_for_tests()
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0})
    yield {"trace": trace_ring._state, "guard": guard_mode._state,
           "worker": worker_diag._state, "step": step_trace._state,
           "lookups": lookup_ledger._state}
    # Teardown ORDER matters: unwrap first (the fake classes are shared per name,
    # see fakes_debug._named_class), then reset the counters.
    fakes_debug.unwrap_everything()
    for key, state in (("trace", trace_ring._state), ("step", step_trace._state),
                       ("guard", guard_mode._state),
                       ("worker", worker_diag._state),
                       ("lookups", lookup_ledger._state)):
        state.clear()
        state.update(saved[key])
    snapshot._state.update(saved["snapshot"])
