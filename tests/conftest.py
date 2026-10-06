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
    blockdump,
    config,
    disk_index,
    disk_spill,
    disk_store,
    evictions,
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
            config.ENV_DIAG_RANK, config.ENV_BOOT_INERT,
            config.ENV_MISS_LOG, config.ENV_DISK_ROOT, config.ENV_DISK_BYTES,
            # The settle wait's three knobs. Listed so no test can inherit a cap from
            # the environment it runs in: a suite that silently waits inside the apply
            # point is a suite whose timings mean nothing.
            config.ENV_DISK_SETTLE_MS, config.ENV_DISK_SETTLE_READ_BPS,
            config.ENV_DISK_PREFILL_TOKENS_PER_S,
            # The reader-thread count and the bounce-buffer pool. Listed for the same
            # reason, and more sharply: the DEFAULT of both is the claim that this
            # release changes nothing, so a test that inherited `=4` from the shell it
            # ran in would be measuring a boot the plugin does not ship.
            config.ENV_DISK_READ_THREADS, config.ENV_DISK_READ_BUFFER_BYTES,
            # The ceiling pass's time budget and its unlink worker count. Same rule,
            # and a sharper one: a test that inherited a 0.001 s budget would "prove"
            # the deadline fires on a store far too small to need it, and one that
            # inherited EVICT_THREADS=4 would be testing a pool the release does not
            # ship by default.
            config.ENV_DISK_CEILING_BUDGET_SECONDS,
            config.ENV_DISK_EVICT_THREADS)


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


@pytest.fixture(autouse=True)
def reset_dev_state():
    """The pieces of DEVELOPMENT state, disarmed and zeroed, then restored.

    These are the plugin's only engine-changing facilities, and all four keep
    process-wide state (a ring, a deque, a class marker). Without this a test that
    armed guard mode would leave the wrappers installed on whatever class it
    passed and leave the next test believing guard mode is off.

    AUTOUSE, deliberately: an opt-in reset is a reset only the files that remember
    it get. `snapshot._state["errors"]` is a process-wide ledger that every
    "must not raise" test adds to, so a file that never asked for this fixture
    handed its ledger to the next file - which made a green suite a property of
    FILE ORDER (see tests/test_dev_state_isolation.py). Tests may still request it
    by name to read the live state dicts it yields; requesting it does not run it
    twice, because pytest instantiates one fixture per test.
    """
    saved = {"snapshot": dict(snapshot._state),
             "trace": dict(trace_ring._state),
             "step": dict(step_trace._state),
             "guard": dict(guard_mode._state),
             "worker": dict(worker_diag._state),
             "worker_contributions": dict(worker_diag._contrib_state),
             "worker_contributors": dict(worker_diag._contributors),
             "blockdump": dict(blockdump._state),
             "evictions": dict(evictions._state),
             "disk_spill": dict(disk_spill._state),
             "disk_index": dict(disk_index._state),
             "disk_store": dict(disk_store._state),
             "lookups": dict(lookup_ledger._state)}
    trace_ring.reset_state_for_tests()
    step_trace.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    worker_diag.reset_state_for_tests()
    # The raw-byte dump holds an ARM, and an armed dump is exactly the kind of
    # development state this fixture exists to keep from leaking into the next
    # test: without this, one test that armed it would leave every later test (and
    # every later `section()`/`capability()` claim) reporting `armed_and_ready`.
    blockdump.reset_state_for_tests()
    evictions.reset_state_for_tests()
    lookup_ledger.reset_state_for_tests()
    # The SSD spill tier holds an arm, a queue and a THREAD. Dropping the queue
    # reference is what keeps a spilled job from one test landing in the next.
    disk_spill.reset_state_for_tests()
    # The spill journal's index is per-process scheduler state (entries, cursors,
    # sequences, and a `journal_seeded` flag that decides whether the FIRST append
    # reads the file's tail). A test that inherited it would either replay another
    # test's journal or seed its sequence from a file that is not there.
    disk_index.reset_state_for_tests()
    # The store-completion hook wraps the worker class and keeps its own retention
    # map of (slot, key16) pairs; both are engine-visible state that a test must not
    # inherit. Its unwrap handle lives on the wrapper, so `unwrap_everything` is what
    # puts the fork's methods back - the reset only zeroes counters and `_retained`.
    disk_store.reset_state_for_tests()
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0})
    yield {"trace": trace_ring._state, "guard": guard_mode._state,
           "worker": worker_diag._state, "step": step_trace._state,
           "blockdump": blockdump._state,
           "evictions": evictions._state,
           "disk_spill": disk_spill._state,
           "disk_index": disk_index._state,
           "disk_store": disk_store._state,
           "lookups": lookup_ledger._state}
    # Teardown ORDER matters: unwrap first (the fake classes are shared per name,
    # see fakes_debug._named_class), then reset the counters. The eviction wrap
    # is INSTANCE-level on a per-test fake pool (never a shared named class), so
    # unwrap_everything has nothing to remove; resetting _state clears `watched`.
    fakes_debug.unwrap_everything()
    for key, state in (("trace", trace_ring._state), ("step", step_trace._state),
                       ("guard", guard_mode._state),
                       ("worker", worker_diag._state),
                       ("worker_contributions", worker_diag._contrib_state),
                       ("worker_contributors", worker_diag._contributors),
                       ("blockdump", blockdump._state),
                       ("evictions", evictions._state),
                       ("disk_spill", disk_spill._state),
                       ("disk_index", disk_index._state),
                       ("disk_store", disk_store._state),
                       ("lookups", lookup_ledger._state)):
        state.clear()
        state.update(saved[key])
    snapshot._state.update(saved["snapshot"])
