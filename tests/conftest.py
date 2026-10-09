"""Shared fixtures. The plugin is importable without vllm installed, so nothing
here stubs an engine: where a test needs an engine it builds a plain object with
the real attribute names (see test_snapshot.py)."""

from __future__ import annotations

import copy
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
    disk_boot,
    disk_gpu_shadow,
    disk_gpu_write,
    disk_index,
    disk_restore_verify,
    disk_shadow,
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
            # The settle wait's one knob. Listed so no test can inherit a cap from
            # the environment it runs in: a suite that silently waits inside the apply
            # point is a suite whose timings mean nothing. Since 0.2.32 the UNSET value
            # is a 5,000 ms FLAT allowance, so this deletion is what makes an unset
            # environment mean the shipped default rather than whatever a developer's
            # shell happens to export - and a test that wants to observe a WAIT must
            # name the cap it waits under, out loud.
            config.ENV_DISK_SETTLE_MS,
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
            config.ENV_DISK_EVICT_THREADS,
            # THE DISK TIER'S BOOT DEFAULT (0.2.33). Listed for the sharpest reason of
            # any variable here: its DEFAULT IS ON, so a suite that inherited
            # `KVTIER_DISK_BOOT_ENABLE=0` from a developer's shell would be testing the
            # off posture while claiming to test the shipped one - and a suite that
            # inherited nothing would be right. Deleting it makes "unset = ON" the fact
            # every test measures, and the tests that want the other value name it.
            config.ENV_DISK_BOOT_ENABLE,
            # THE SHADOW VERIFIER (0.2.36). Listed because its DEFAULT IS OFF and the
            # OFF state is a claim every test of this release makes: a suite that
            # inherited `VLLM_E2_TIER_DISK_SHADOW=1` from a developer's shell would be
            # running a device read on the step thread inside tests that assert no GPU
            # read is ever attempted. Deleting it makes "unset = off" the fact each
            # test measures, and the tests that want it on name it.
            config.ENV_DISK_SHADOW,
            # THE GPU-DIRECT SHADOW WRITER (0.2.37). Same rule, and a heavier one: this
            # one WRITES FILES. A suite that inherited
            # `VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES=1` from a developer's shell would be
            # writing KV files into a directory the tests never asked for, from tests
            # that assert nothing is ever written. `..._ROOT` is listed for the mirror
            # reason: an inherited root would point a test's shadow files somewhere
            # outside its `tmp_path`, which is the opposite of what a temp-directory
            # test suite means.
            config.ENV_DISK_GPU_SHADOW_WRITES, config.ENV_DISK_GPU_SHADOW_ROOT,
            # THE CPU-SLOT RESERVE (0.2.38). Listed for the same reason as the pool
            # size, and this one bites the ARITHMETIC rather than the bytes: the whole
            # claim of the release is that an unset variable still reserves 64 slots,
            # and a suite that inherited `=0` from a developer's shell would be
            # offering the arena's last slot while asserting the shipped bound. The
            # affordability tests below `VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE` name it.
            config.ENV_DISK_CPU_SLOT_RESERVE,
            # THE RESTORE-SIDE VERIFIER (0.2.39). Listed for `disk_shadow`'s reason and
            # sharper one: this one reads a GPU block AND opens a spill FILE for
            # reading on its own thread, from the apply point. A suite that inherited
            # `VLLM_E2_TIER_DISK_RESTORE_VERIFY=1` from a developer's shell would be
            # running device copies and file opens inside tests that assert the apply
            # point touches neither - and, because the work is on another thread, it
            # would be NON-DETERMINISTIC rather than merely wrong. Deleting it makes
            # "unset = off" the fact every test measures, and tests that want it on name
            # it out loud.
            config.ENV_DISK_RESTORE_VERIFY,
            # THE GPU-DIRECT WRITE (0.2.43). The sharpest case in this list: this one
            # ISSUES A HOST->DEVICE COPY INTO A LIVE KV WINDOW from the apply point, on
            # the step thread. A suite that inherited
            # `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE=1` from a developer's shell would be
            # running device copies inside tests that assert the apply point touches no
            # device - and, because the module also allocates pinned rows on its own
            # thread, the interference would be non-deterministic rather than merely
            # wrong. Deleting it makes "unset = off" the fact every test measures, and
            # tests that want it on name it out loud.
            config.ENV_DISK_GPU_DIRECT_WRITE,
            # THE STEP-2 LOAD-PAIR PRUNE (0.2.49). Its DEFAULT IS OFF, and OFF is
            # a claim every test of this release makes: a suite that inherited
            # `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS=1` from a developer's shell would
            # be REMOVING load pairs inside tests that assert the emitted step is what
            # the scheduler shipped - and a positional-pairing change is exactly the
            # kind of interference no assertion in those files would name. Deleting it
            # makes 'unset = off' the fact every test measures; the suites that want it
            # up raise it themselves and drop the module's cached switch with it.
            config.ENV_DISK_PRUNE_RESTORED_LOADS)


def _isolate(state: dict) -> dict:
    """A snapshot the test cannot dirty through the original it was taken from.

    `dict(_state)` is a SHALLOW copy, so every nested container in a module's state is
    the SAME OBJECT the test mutates: `disk_spill._state["digest_groups"][key][g] += 1`
    writes straight through the snapshot, and the teardown `update(saved)` then hands
    the NEXT test the object the previous one dirtied. Measured, an in-place write
    survives that restore in every state dict that owns a nested dict -
    `disk_shadow._state["counters"]`, `disk_spill._state["refused"]`,
    `disk_store._state["refused"]` and `disk_index._state["dropped"]` - and on
    `tests/test_disk_shadow.py` the spill path's `digest_groups` map grows by one event
    per test and is resurrected after every teardown. Nothing in the suite is
    order-dependent today only because each module's own `reset_state_for_tests()`
    happens to rebuild every nested dict it knows about - so a reset that forgets one
    key (or a module added later, like `disk_shadow` was) is a silent leak, and a leak
    here is invisible until the file order changes. Copying the snapshot deeply makes
    the isolation a property of THIS fixture instead of a property of every module's
    reset list. A value that cannot be copied (a lock, a thread, a module) is one none
    of these states mutates in place, so it is kept by reference rather than replaced
    by something that could never be put back. (`disk_store._logged_lines` is a third
    shape: a module-level list no snapshot covers at all, which is why
    `disk_store.reset_state_for_tests()` clears it itself.)
    """
    out: dict = {}
    for key, value in state.items():
        try:
            out[key] = copy.deepcopy(value)
        except Exception:  # noqa: BLE001 - un-copyable: keep the reference itself
            out[key] = value
    return out


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
    saved = _isolate(snapshot._state)
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
    saved = {"snapshot": _isolate(snapshot._state),
             "trace": _isolate(trace_ring._state),
             "step": _isolate(step_trace._state),
             "guard": _isolate(guard_mode._state),
             "worker": _isolate(worker_diag._state),
             "worker_contributions": _isolate(worker_diag._contrib_state),
             "worker_contributors": _isolate(worker_diag._contributors),
             "blockdump": _isolate(blockdump._state),
             "evictions": _isolate(evictions._state),
             "disk_spill": _isolate(disk_spill._state),
             "disk_index": _isolate(disk_index._state),
             "disk_store": _isolate(disk_store._state),
             # THE DISK TIER'S BOOT DEFAULT (0.2.33): an honoured boot default holds an
             # arm, a serve posture and the operator's disarm latch, all process-wide.
             # Without this a test that ran the real boot with a disk root would hand
             # the next file a latch it never set - or an arm it never armed.
             "disk_boot": _isolate(disk_boot._state),
             # THE SHADOW VERIFIER (0.2.36): an enablement latch, a retention map of GPU
             # block ids and the comparison counters are all process-wide. A test that
             # armed the shadow must not hand it to the next test, which would then
             # observe a device read it never asked for.
             "disk_shadow": _isolate(disk_shadow._state),
             # THE GPU-DIRECT SHADOW WRITER (0.2.37): its latch, its resolved root, the
             # pending-comparison list and its write/compare counters are all
             # process-wide, and unlike the read side it holds a THREAD. A test that
             # armed it must not hand a live writer to the next one.
             "disk_gpu_shadow": _isolate(disk_gpu_shadow._state),
             # THE RESTORE-SIDE VERIFIER (0.2.39): its latch, its retained-entry map
             # (which holds a path and a worker weakref), its geometry cache and its
             # counters are all process-wide, and it owns a THREAD. A test that armed it
             # must not hand a live verifier - or a retained path from a `tmp_path` that
             # is about to be deleted - to the next one.
             "disk_restore_verify": _isolate(disk_restore_verify._state),
             # THE GPU-DIRECT WRITE (0.2.43): its latch, its in-flight records (which
             # hold PINNED TENSORS and a worker weakref), its pinned-row pool, its byte
             # accounting and its counters are all process-wide, and it owns a THREAD. A
             # test that armed it must not hand a live writer - or rows allocated
             # against a fake worker about to be garbage-collected - to the next.
             "disk_gpu_write": _isolate(disk_gpu_write._state),
             "lookups": _isolate(lookup_ledger._state)}
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
    # THE DISK TIER'S BOOT DEFAULT (0.2.33). Its state decides whether the renewal tick
    # is allowed to arm the spill tier on the NEXT test's first offer, and whether an
    # operator's disarm latch is standing - both are boot-scoped facts a test must never
    # inherit from another test. Reset before the yield, like every sibling here.
    disk_boot.reset_state_for_tests()
    # THE SHADOW VERIFIER. Its latch is an enablement, and an enablement a test set is
    # exactly the state this fixture exists to keep from travelling.
    disk_shadow.reset_state_for_tests()
    # THE GPU-DIRECT SHADOW WRITER. Same reason, plus its writer thread: the reset stops
    # the thread and drops the queue, so a shadow file queued by one test cannot be
    # written into a directory the next test is asserting is empty.
    disk_gpu_shadow.reset_state_for_tests()
    # THE RESTORE-SIDE VERIFIER. Same reason, plus its thread: the reset stops the
    # thread and drops the retained entries, so an entry retained against one test's
    # fake worker cannot be verified against the next test's.
    disk_restore_verify.reset_state_for_tests()
    # THE GPU-DIRECT WRITE. Same reason, plus its thread and its pinned pool: the reset
    # stops the thread and forgets every row, so a row allocated against one test's fake
    # worker can never be handed to another test's copy.
    disk_gpu_write.reset_state_for_tests()
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0})
    yield {"trace": trace_ring._state, "guard": guard_mode._state,
           "worker": worker_diag._state, "step": step_trace._state,
           "blockdump": blockdump._state,
           "evictions": evictions._state,
           "disk_spill": disk_spill._state,
           "disk_index": disk_index._state,
           "disk_store": disk_store._state,
           "disk_boot": disk_boot._state,
           "disk_shadow": disk_shadow._state,
           "disk_gpu_shadow": disk_gpu_shadow._state,
           "disk_restore_verify": disk_restore_verify._state,
           "disk_gpu_write": disk_gpu_write._state,
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
                       ("disk_boot", disk_boot._state),
                       ("disk_shadow", disk_shadow._state),
                       ("disk_gpu_shadow", disk_gpu_shadow._state),
                       ("disk_restore_verify", disk_restore_verify._state),
                       ("disk_gpu_write", disk_gpu_write._state),
                       ("lookups", lookup_ledger._state)):
        state.clear()
        state.update(saved[key])
    snapshot._state.update(saved["snapshot"])
