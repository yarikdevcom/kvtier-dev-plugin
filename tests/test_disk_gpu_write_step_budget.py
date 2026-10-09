r"""THE APPLY POINT MUST RETURN (0.2.56): the delivery loop's step-thread budget.

WHY THIS FILE EXISTS. On 2026-10-09 at 14:19 the live box died with

    TimeoutError: RPC call to sample_tokens timed out

from `multiproc_executor.get_response`, sixty seconds after
`No available shared memory broadcast block found in 60 seconds` - a worker that had
stopped servicing the executor's message queue, which the engine core treats as fatal.
The window was the first boot with `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY=1` and
`VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS=1` up, and the restore it died on was real:
`disk_restore.offers=1`, `adopted_tokens=198,400`. Every counter in this module's rank
section read `None` and the prune read `steps=0` with all twelve refusal counters at 0 -
which reads like "the hook never ran", and is instead what a step that never LEFT the
delivery loop looks like, because every delivery counter is counted after the copy loop.

THE MECHANISM, in one paragraph. A delivery copy's source is `worker.cpu_kv_caches`, and
the running fork allocates that arena with `torch.zeros(cpu_shape, dtype=...,
device="cpu")` (`v1/simple_kv_offload/worker.py:232`) - PAGEABLE, not page-locked. A
host->device `copy_(src, non_blocking=True)` out of pageable host memory is NOT
asynchronous: the runtime stages it through its own bounce buffer and returns only once
the bytes have left, so each copy BLOCKS THE ENGINE STEP THREAD. This module already
knows that rule - `_copy_device_to_host` calls a pageable destination "the single most
expensive mistake available here" - and the delivery branch broke it in the other
direction while the mode lifted the sample (1-in-64 to 1), the per-step cap (1 to every
survivor) and the 0.5 s floor together. What remained was `MAX_ENTRIES_SCANNED = 64`: a
bound on ENTRIES, not on bytes and not on seconds. One entry is a whole rank window
(11,468,800 B on that box), so one apply point could put ~735 MB of host-synchronous
PCIe in front of the worker's message loop.

WHY THE SUITE COULD NOT SEE IT, stated plainly because the same gap is the reason to
read this file before adding a delivery test. The fakes' `copy_` is a
`ctypes.memmove`: it
returns in microseconds, so no existing test ever spent measurable TIME in the loop, and
the only bound the loop had was a COUNT, which several tests assert on directly. A
feature whose whole failure mode is "the step thread is inside this loop too long" was
tested by a fake that cannot be slow. The fixture below adds the one property the fake
was missing - the COST of a synchronous copy - and changes no behaviour.

WHAT THIS SUITE PINS:

1. THE APPLY POINT RETURNS. A step of 48 delivered entries, driven on a stand-in step
   thread with a bounded join. Before 0.2.56 the thread is still inside `observe_apply`
   when the bound expires, and the assertion says so with the counters in its hand. The
   failure is bounded: the copies wait on an event the teardown sets, so a failing run
   leaves no sleeper behind. COLD: no stubbed `_ensure_thread`, no `fill_pool`, nothing
   pre-armed - the same discipline `test_disk_gpu_write_cold_delivery.py` exists to
   impose.
2. THE DEFERRAL IS A NAMED THROTTLE, NOT A LOSS. The entries after the stop are absent
   from `written_indices`, so `disk_restore.prune_adopted_restores` cannot remove them
   and the fork copies them; `delivery_deferred_entries`/`_steps` count them, and
   `delivery_attempted` proves the loop ran.
3. THE FIRST ENTRY IS BOUNDED TOO. A single `copy_` cannot be interrupted from this
   thread, so an entry wider than the step's whole byte budget is REFUSED rather than
   copied - with the same safe consequence: un-written means un-covered means the fork
   copies it.
4. THE WITNESS POSTURE IS UNTOUCHED. The budget is delivery-only; the shipped 1-per-step
   cap and 0.5 s floor already bound it, and this file must not change a path that was
   never the problem.
"""

from __future__ import annotations

import threading
import time

import pytest
from test_disk_gpu_write import (
    CPU_BLOCK,
    GPU_BLOCK,
    WIDTH,
    FakeRegion,
    Journal,
    WriteWorker,
    seed_window,
)
from test_disk_gpu_write_delivery import many
from test_disk_restore import Meta
from test_disk_restore_verify import write_window

from kvtier_dev_plugin import (  # noqa: E402 - the module under test
    config,
    disk_gpu_write,
    disk_restore,
)

# THE COST OF ONE SYNCHRONOUS PAGEABLE COPY. Deliberately far larger than a memmove and
# far smaller than the live box's 11.5 MB window: what matters is the RATIO to the
# budget, and the numbers below are set so that the unfixed loop runs ~5x the return
# bound while the fixed one lands inside it at any plausible CI load.
SLOW_COPY_SECONDS = 0.1
ENTRIES = 48
RETURN_BUDGET_SECONDS = 1.5


@pytest.fixture(autouse=True)
def fresh_state():
    """A fresh `disk_restore` and a fresh writer per test; the thread we START,
    we stop.

    The cold-start file's reason, unchanged: a daemon thread left running between tests
    is a prober that moves the next test's counters from another thread.
    """
    disk_restore.reset_state_for_tests()
    disk_gpu_write.reset_state_for_tests()
    yield
    disk_gpu_write.stop_thread()


@pytest.fixture
def journal():
    return Journal()


@pytest.fixture
def worker(journal):
    """Wide enough for ENTRIES distinct destinations (GPU_BLOCK + ENTRIES < 64)."""
    return WriteWorker(journal=journal,
                       gpu={"g0": FakeRegion("g0", 64, WIDTH, journal=journal)})


@pytest.fixture
def cold_delivery(monkeypatch):
    """THE COLD BOOT, and nothing else: two env vars.

    No `monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)` and no
    `fill_pool(worker)`. The thread this test starts, if one starts, is the module's
    real one, and the pool is as empty as a first restore step finds it.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    disk_gpu_write._state["env_resolved"] = False
    disk_gpu_write._state["delivery_resolved"] = False
    return True


@pytest.fixture
def slow_h2d(monkeypatch):
    """ONE host->device COPY COSTS `SLOW_COPY_SECONDS`; nothing else about it changes.

    The seam is the module's own `_copy_host_to_device`, WRAPPED and not replaced: the
    real fake copy still runs, so the bytes really cross the boundary and the journal
    really records them. What the wrapper adds is the property the fake was missing and
    the product has - a pageable-source copy returns only after the bytes leave - which
    is the entire defect. `abort` lets the teardown release an unfixed loop at once, so
    the failing case costs a second and not a CI timeout.
    """
    abort = threading.Event()
    calls: list = []
    real = disk_gpu_write._copy_host_to_device

    def slow(dst_region, dst_block, src_region, src_block):
        calls.append((dst_region.name, int(dst_block)))
        abort.wait(SLOW_COPY_SECONDS)
        return real(dst_region, dst_block, src_region, src_block)

    monkeypatch.setattr(disk_gpu_write, "_copy_host_to_device", slow)
    yield {"calls": calls, "abort": abort}
    abort.set()


def _run_on_step_thread(worker, batch) -> tuple:
    """`observe_apply` on a stand-in step thread, joined with a BOUND.

    Returns `(answer or None, seconds spent, the thread)`. A thread still alive after
    the join is the reproduction; the caller turns that into an assertion with a
    message, because "it hung" and "it was slow" must not be the same sentence.
    """
    box: dict = {}

    def run():
        box["answer"] = disk_gpu_write.observe_apply(worker, batch, {})

    thread = threading.Thread(target=run, name="step-thread-standin", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(RETURN_BUDGET_SECONDS)
    return box.get("answer"), time.monotonic() - started, thread


# --- 1. the apply point returns, and the deferral is what made it return ---------

def test_the_apply_point_returns_within_its_budget_and_defers_the_rest(
        cold_delivery, slow_h2d, worker, tmp_path):
    """THE CRASH, REPRODUCED IN-PROCESS, AND THE FIX MEASURED.

    One step, ENTRIES entries, each with its own destination block, and each copy
    costing what a synchronous pageable copy costs. Before 0.2.56 the loop's only bound
    is `MAX_ENTRIES_SCANNED`, so this thread is still inside `observe_apply` when the
    bound expires. After it, the step delivers what fits the budget and DEFERS the
    rest -
    and a deferred entry is an entry the FORK still copies.
    """
    path = write_window(str(tmp_path / "kvslot-budget"), b"\x5a" * 64)
    seed_window(worker, path)
    batch = many(path, ENTRIES)

    answer, spent, thread = _run_on_step_thread(worker, batch)
    assert not thread.is_alive(), (
        "THE 0.2.55 FAILURE, REPRODUCED: `observe_apply` had still not returned "
        f"{spent:.2f} s after {ENTRIES} entries whose copies each cost "
        f"{SLOW_COPY_SECONDS} s. It is inside its own delivery loop, whose only bound "
        f"is `MAX_ENTRIES_SCANNED = {disk_gpu_write.MAX_ENTRIES_SCANNED}` - a bound on "
        "ENTRIES, not on bytes or seconds, with the sample, the per-step cap and "
        "the interval floor all lifted by delivery mode. Each entry's "
        "`copy_(non_blocking=True)` out of the fork's PAGEABLE `cpu_kv_caches` "
        "(`v1/simple_kv_offload/worker.py:232`, no `pin_memory`) is SYNCHRONOUS on the "
        "engine step thread, so a restore step sits inside `start_load_kv` for "
        "entries x copy cost while the executor counts down to `RPC call to "
        "sample_tokens timed out` - the 2026-10-09 14:19 EngineCore death. The fix is "
        "DELIVERY_STEP_BUDGET_SECONDS / DELIVERY_STEP_BYTE_BUDGET. Counters: "
        f"{disk_gpu_write.counters()}")

    assert answer is not None and answer["delivery_mode"] is True
    # WHAT IT DID MANAGE, in the journal and not only in a counter.
    delivered = answer["written"]
    assert 0 < delivered < ENTRIES, (
        f"expected the budget to stop the step part-way, got {delivered} of {ENTRIES}")
    assert len(slow_h2d["calls"]) == delivered, (
        f"the journal saw {len(slow_h2d['calls'])} copies for {delivered} writes")
    assert len(worker.journal.of("h2d")) == delivered
    assert answer["written_indices"] == set(list(batch["jobs"])[:delivered]), (
        "the covered set must be exactly the delivered prefix: the prune removes an "
        "entry ONLY on membership here, so anything else in it is a byte nobody wrote")
    # THE DEFERRAL, named and counted. A throttle with a name, not a fault: the entries
    # after the stop stay uncovered, so `prune_adopted_restores` cannot remove them and
    # the fork's copy thread delivers them.
    seen = disk_gpu_write.counters()
    assert seen["delivery_deferred_entries"] == ENTRIES - delivered, seen
    assert seen["delivery_deferred_steps"] == 1, seen
    assert seen["delivery_attempted"] == delivered, (
        "every delivered entry is ATTEMPTED before its copy, which is the counter that "
        "distinguishes 'the apply point ran' from 'it never came back': "
        f"{seen}")
    assert seen["delivery_writes"] == delivered, seen
    assert seen["delivery_refused_entry_too_wide"] == 0, seen
    # THE STEP'S OWN WALL CLOCK, published: the positive proof the apply point ran,
    # which the crash window did not have.
    state = disk_gpu_write.delivery_state()
    assert state["step_seconds_last"] is not None
    assert state["step_seconds_last"] >= SLOW_COPY_SECONDS
    assert state["step_seconds_peak"] >= state["step_seconds_last"]
    assert spent < RETURN_BUDGET_SECONDS


def test_a_budget_stop_leaves_the_uncovered_entries_in_the_fork_s_lists(
        cold_delivery, slow_h2d, worker, tmp_path, monkeypatch):
    """THE CONSEQUENCE THAT MAKES THE BUDGET SAFE, checked at the consumer.

    With the Step-2 flag up, `prune_adopted_restores` decides whether the fork's copy
    happens, and it prunes on membership of `written_indices` alone. So the claim "a
    budget stop costs a missed optimization and never a hole" is only true if the
    deferred ids stay OUT of that set and therefore stay in the step's lists. This hands
    the real answer to the real consumer with the real flag up.
    """
    path = write_window(str(tmp_path / "kvslot-budget-fork"), b"\x6b" * 64)
    seed_window(worker, path)
    batch = many(path, ENTRIES)
    monkeypatch.setenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, "1")
    disk_restore._state["prune_switch"] = None
    # THE DESTINATIONS `many()` ACTUALLY BOUND: the prune cross-checks every job id
    # against the step's own lists (`destination_drift` is its name when they differ),
    # so this metadata is the batch, not a decoration.
    meta = Meta(load_event=11, cpu=[CPU_BLOCK] * ENTRIES,
                gpu=[GPU_BLOCK + i for i in range(ENTRIES)],
                paths=[path] * ENTRIES, reqs={11: ["req-1"]})

    answer, spent, thread = _run_on_step_thread(worker, batch)
    assert not thread.is_alive(), (
        f"the apply point never returned ({spent:.2f} s), so a budget stop cannot be "
        "read at its consumer at all")
    delivered = answer["written"]
    assert 0 < delivered < ENTRIES

    report = disk_restore.prune_adopted_restores(
        meta, batch, set(), direct_witnesses=delivered,
        covered=answer["written_indices"])
    assert report["ok"] is True, report
    assert report["pruned"] == delivered, report
    assert report["uncovered_entries"] == ENTRIES - delivered, report
    # THE SURVIVING LISTS, as the answer carries them: the prune does not mutate the
    # metadata in place, `worker_before_load` writes these two lists back into it. What
    # the fork will still DMA is exactly what the budget deferred - not one entry the
    # plugin failed to write is gone from them.
    assert len(report["cpu"]) == ENTRIES - delivered, report
    assert len(report["gpu"]) == ENTRIES - delivered, report
    assert disk_restore.prune_state()["unwitnessed_entries"] == 0, (
        "an entry was pruned that this module had not covered")


# --- 2. the first entry of a step is bounded too --------------------------------

def test_an_entry_wider_than_the_step_budget_is_refused_and_never_copied(
        cold_delivery, slow_h2d, worker, tmp_path, monkeypatch):
    """ONE `copy_` CANNOT BE INTERRUPTED, so the only bound on it is refusal.

    `WIDTH` is the fake's window; the budget is dropped below it to make the guard bind
    at a testable number. The entry is refused BEFORE any copy, counted under its own
    name, and `delivery_attempted` stays 0 - which is the pair an operator reads: `
    attempted` moved but `writes` did not means a copy was started and did not come
    back; neither moved means the module refused to start one.
    """
    monkeypatch.setattr(disk_gpu_write, "DELIVERY_STEP_BYTE_BUDGET", WIDTH // 2)
    path = write_window(str(tmp_path / "kvslot-budget-wide"), b"\x7c" * 64)
    seed_window(worker, path)

    answer = disk_gpu_write.observe_apply(worker, many(path, 1), {})
    assert answer["written"] == 0, answer
    assert answer["reason"] == "delivery_refused_entry_too_wide", answer
    seen = disk_gpu_write.counters()
    assert seen["delivery_refused_entry_too_wide"] == 1, seen
    assert seen["delivery_attempted"] == 0, seen
    assert slow_h2d["calls"] == [], "a refused entry must not reach the copy"
    assert answer["written_indices"] == set()


# --- 3. the witness posture is not this file's business ------------------------

def test_the_witness_posture_is_unchanged_by_the_budget(cold_delivery, slow_h2d,
                                                        worker, tmp_path,
                                                        monkeypatch):
    """NO DELIVERY, NO BUDGET: the shipped path still refuses rather than stalls.

    With the delivery flag down the writer is the 0.2.51 witness: one write per step, a
    0.5 s floor, and - with an empty pool and a cold boot - a refusal that RETURNS. The
    budget must not have entered that path, in either direction: no deferral counted,
    and still exactly one write for a 48-entry step.
    """
    monkeypatch.delenv(config.ENV_DISK_GPU_WRITE_DELIVERY)
    disk_gpu_write._state["delivery_resolved"] = False
    path = write_window(str(tmp_path / "kvslot-budget-witness"), b"\x76" * 64)
    seed_window(worker, path)

    answer, spent, thread = _run_on_step_thread(worker, many(path, ENTRIES))
    assert not thread.is_alive(), f"the witness path hung: {spent:.2f} s"
    assert answer["delivery_mode"] is False
    assert answer["written"] <= disk_gpu_write.MAX_WRITES_PER_STEP, answer
    seen = disk_gpu_write.counters()
    assert seen["delivery_deferred_steps"] == 0, seen
    assert seen["delivery_deferred_entries"] == 0, seen
    assert seen["delivery_attempted"] == 0, seen
    assert seen["throttled_step"] == 0, (
        f"the budget must not touch the witness loop: {seen}")
    # WHAT THE SHIPPED PATH ACTUALLY DID on a cold boot, named so this test cannot
    # silently stop asserting anything: the sampler took 1 entry in 64, the empty pool
    # refused it, and - the point - the call RETURNED with a name.
    assert seen["sampled_out"] >= 1, seen
    assert answer["reason"] == "no_pinned_buffer", answer
    assert seen["pool_never_filled"] == 1, seen
