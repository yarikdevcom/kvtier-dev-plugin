r"""COLD-START DELIVERY (0.2.53): the delivery lane reached from a worker with nothing
pre-armed.

WHY THIS FILE EXISTS. `tests/test_disk_gpu_write_delivery.py` is a good suite, and every
test in it is armed by the `witness` fixture, which does this:

    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)

Every delivery test inherits that line, and one more after it: the end-to-end test
(`test_the_real_apply_point_delivers_every_entry_and_prunes_only_those`) calls
`fill_pool(worker)` before its first apply. Those two facts together mean that suite
proves *a writer with a running thread and a warm pool delivers its step* - which is not
the field condition. On a real boot:

  * the module's thread is started by the FIRST ARMED APPLY (or by
    `register_contribution`), so until one happens there is no thread at all;
  * rows are allocated only inside a refill, and a refill runs only on that thread;
  * a thread born before any worker exists refills NOTHING, because its
    `_resolve_worker()` answers None until an apply registers the worker (pinned by the
    sibling's `test_registering_the_contribution_while_armed_starts_the_writer_and_
    nothing_else`);
  * so the FIRST apply of every boot finds an empty pool, and the worker has just been
    registered by that same apply.

The sibling file has the cold-boot regression for the WITNESS posture
(`test_a_cold_writer_starts_its_own_thread_and_then_actually_writes`) and it is a good
test. Nothing equivalent existed for DELIVERY mode - the posture whose refusals matter
MORE, because the Step-2 prune removes the fork's copy on the strength of this module's
`written_indices`. A writer that could not deliver a single entry from a cold boot would
have been green. Four of this file's five tests FAIL the moment the sibling's
`lambda: True` is put back, which is the measurement of the gap this closes.

WHAT THIS SUITE PINS, in the order a reviewer should read it:

1. THE FIRST DELIVERY OF A BOOT HAPPENS, with the real `_ensure_thread`, no pool fill
   and the product's own throttles: the cold pool costs the WITNESS and never the copy.
   That is delivery mode's whole claim, tested at the one moment it could be false.
2. THE THREAD IS THE MODULE'S OWN and the first refusal starts it - asserted on
   `thread_started` and on the thread NAMES the allocations ran on. A suite that stubs
   the thread cannot make either assertion.
3. THE KILL-SWITCH LANE IS REACHABLE FROM COLD. A readback needs a pinned row; a row
   needs the thread; the thread needs an apply. A cold boot must therefore go on to
   WITNESS a delivery and then LATCH on an injected wrong copy, or the kill switch is a
   feature that exists only in tests.
4. NO FIXTURE PRE-ARMS ANYTHING, machine-checked rather than claimed.

The two environment variables are the only thing the fixture sets: the sampler, the
per-step cap and the interval floor are lifted by the mode itself or the tests fail.
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
    one_pass,
    seed_window,
)
from test_disk_gpu_write_delivery import (
    copy_seams as _sibling_copy_seams,
)
from test_disk_gpu_write_delivery import (
    many,
)
from test_disk_restore_verify import write_window

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    config,
    disk_gpu_write,
    disk_restore,
)

# THE SIBLING'S SEAM-WRAPPER, UNDER THIS FILE'S NAME. It arrives already decorated, so
# this is a plain ALIAS and not a second `@pytest.fixture`; and an assignment, unlike an
# import of that name, does not read to ruff as a parameter shadowing it in every
# signature below.
copy_seams = _sibling_copy_seams


@pytest.fixture(autouse=True)
def fresh_state():
    """A fresh `disk_restore` AND a fresh writer for every test in this file.

    The sibling resets `disk_restore` because `conftest.py` deliberately does not (that
    suite resets it through its own autouse `fresh` fixture). This file adds the writer:
    a cold-start suite that inherited another test's resolved env, started thread or
    pinned rows would not be measuring a cold start at all.

    The teardown stops the thread this file STARTS - the module's real one. A daemon
      left
    running between tests is a prober that moves the next test's counters from another
    thread, which is how a suite starts failing only in a full run.
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
    """A worker whose GPU region is wide enough for a delivered step.

    The sibling's reason, unchanged: a step with several DISTINCT destinations must not
    run off the end of an 8-block fake region, or the test asserts
    `gpu_block_out_of_range` instead of delivery.
    """
    return WriteWorker(journal=journal,
                       gpu={"g0": FakeRegion("g0", 32, WIDTH, journal=journal)})


@pytest.fixture
def cold_delivery(monkeypatch):
    """THE COLD BOOT: two environment variables and NOTHING else.

    Compare it with the sibling's `witness` fixture; the difference is the point:

        monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)

    That line hands a test a writer whose daemon thread exists before a single batch has
    been applied. Production has no such line: the thread is started by the first armed
    apply, so a cold boot has an EMPTY POOL at its first step. Here nothing is stubbed,
    nothing is filled and no throttle is lifted by hand - the thread, if one starts, is
    the module's real thread, and the pinned rows, if any are allocated, are allocated
      by
    the module's own refill on that thread.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    disk_gpu_write._state["env_resolved"] = False
    disk_gpu_write._state["delivery_resolved"] = False
    return True


@pytest.fixture
def prune_on(monkeypatch):
    """THE STEP-2 FLAG RAISED, for the one test that asks the CONSUMER what it thinks.

    `prune_adopted_restores` returns `attempted: False` while its own flag is down, and
    an un-consulted gate says nothing about the kill switch - so a test that reads the
    prune's refusal has to raise the flag first. It is a separate fixture because
    raising it in `cold_delivery` would make four of the five tests about a feature they
    are not testing.
    """
    monkeypatch.setenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, "1")
    disk_restore._state["prune_switch"] = None
    return True


def counters() -> dict:
    return disk_gpu_write.counters()


def wait_for(predicate, *, seconds=8.0, what=""):
    """Poll the COUNTERS until a condition holds or the budget is spent.

    Bounded and reported: a cold start that never converges must fail with the counters
    in the message, because "the thread never filled the pool" and "the test was too
    impatient" are different incidents and only one of them is a bug.
    """
    deadline = time.monotonic() + seconds
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = counters()
        if predicate(seen):
            return seen
        time.sleep(0.005)
    raise AssertionError(f"{what} never happened: {seen}")


# --- 1. the first delivery of a boot happens, with nothing pre-armed ------------

def test_the_first_delivery_of_a_cold_boot_lands_and_starts_its_own_thread(
        cold_delivery, worker, tmp_path):
    """THE GAP THE SIBLING SUITE COULD NOT SEE, closed.

    Nothing is pre-armed: the env is resolved by this call, the thread has never run,
      the
    pool has never been filled, the worker has never been registered. `observe_apply` is
    the ONLY production entry point this test touches.

    The claim delivery mode makes is that the cold pool costs the WITNESS and never the
    COPY, because a delivery's source is the CPU arena slot and not a pinned row. This
      is
    that sentence tested at the one moment in a boot's life when it could be false - and
    against the counter that proves the thread is real, which a stubbed `_ensure_thread`
    can never produce.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x5a" * 64)
    seed_window(worker, path)

    answer = disk_gpu_write.observe_apply(worker, many(path, 4), {})
    seen = counters()
    assert answer["written"] == 4, (
        f"a cold pool refused a DELIVERY on a cold boot: {answer} {seen}")
    assert answer["written_indices"] == {0, 1, 2, 3}
    assert answer["delivery_mode"] is True
    assert seen["no_pinned_buffer"] == 0, "a delivery is not a refusal"
    assert seen["delivered_unwitnessed"] == 4, \
        "a cold boot claimed a WITNESS it had no pinned row to provide"
    assert seen["delivery_writes"] == 4
    # FOUR DISTINCT WINDOWS, so "4" is four copies and not a counter agreeing with
    # itself.
    assert len(worker.journal.of("h2d")) == 4
    # THE THREAD IS REAL. `thread_started` is the field the 0.2.43 incident was read by,
    # and the one a stubbed `_ensure_thread` cannot move: put the sibling's stub back
    # and THIS assertion fails, which is exactly the coverage this file adds.
    assert seen["thread_started"] == 1
    # NOTHING IS IN FLIGHT, which is what lets one apply point deliver a whole step
    # before the thread has ever run a pass.
    assert disk_gpu_write.pool_view()["in_flight"] == 0


def test_a_cold_delivery_costs_the_witness_and_never_the_copy(cold_delivery, worker,
                                                              tmp_path):
    """THE ASYMMETRY THE MODE IS BUILT ON, on the cold path.

    Under the WITNESS posture the same cold apply refuses - `no_pinned_buffer`, the
    production signature of the 0.2.43 bug. Under delivery it must not, and the reason
      is
    a property of the copy rather than a special case: the delivery's source is the
      arena
    slot, so it needs no pinned row at all. `delivered_unwitnessed` is the honest name
      for
    what a cold boot actually pays, and the counter an operator reads to learn the
      thread
    has not caught up yet.
    """
    path = write_window(str(tmp_path / "kvslot-window-2"), b"\x6b" * 64)
    seed_window(worker, path)
    assert disk_gpu_write.delivery_state()["effective"] is True
    # NOTHING PRE-FILLED: no region is even in the pool yet, because a region is learned
    # from a worker and this boot has not applied a batch.
    assert disk_gpu_write.pool_view()["free_rows"] == {}

    answer = disk_gpu_write.observe_apply(worker, many(path, 2), {})
    assert answer["written"] == 2, answer
    assert counters()["no_pinned_buffer"] == 0
    assert counters()["delivered_unwitnessed"] == 2


# --- 2. the cold boot goes on to WITNESS, or the kill switch is a fiction -------

def test_a_cold_boot_goes_on_to_witness_a_delivery_so_the_kill_switch_is_reachable(
        cold_delivery, worker, tmp_path):
    """THE SECOND HALF OF THE COLD-START STORY, and the more important half.

    A kill switch that can only fire once a warm pool exists is a switch that does not
    exist for the first minutes of every boot - and the first minutes are when a bad
      build
    gets its traffic. The chain from a cold boot to a FIREABLE switch is: an apply
      starts
    the thread; the thread refills; a refill allocates rows; a later apply witnesses its
    delivery against a row; a pass judges the readback. Nothing else in the suite walks
    that chain, and it is the chain the sibling's stubbed `_ensure_thread` skips at its
    first link.

    The loop drives the PRODUCT (repeated applies, as a serving worker does), never a
    seam, and the assertions are on the counters that mean the chain completed plus the
    thread names the allocations ran on.
    """
    path = write_window(str(tmp_path / "kvslot-window-3"), b"\x7c" * 64)
    seed_window(worker, path)

    # THE ALLOCATION SEAM, WRAPPED BEFORE ANYTHING RUNS, so "no pinned row is allocated
    # on the step thread" is an assertion about a list of thread names and not about a
    # comment. Wrapped rather than replaced, which keeps the production call chain
    # (`_refill_pools` -> `_allocate_row` -> the fake region's `new_empty`).
    alloc_threads: list = []
    real_alloc = disk_gpu_write._allocate_row

    def observe_alloc(region, elems):
        alloc_threads.append(threading.current_thread().name)
        return real_alloc(region, elems)

    def step() -> dict:
        # THE PRODUCT ENTRY POINT, over and over. The interval floor is 0 in delivery
        # mode, so nothing here waits on a throttle the flag lifted itself.
        return disk_gpu_write.observe_apply(worker, many(path, 1), {})

    disk_gpu_write._allocate_row = observe_alloc
    seen: dict = {}
    try:
        step()
        wait_for(lambda c: c["thread_started"] >= 1,
                 what="the module's own thread never started from a cold apply")
        wait_for(lambda c: c["pool_refills"] >= 1,
                 what="the module's thread never refilled its pool")
        # THE THREAD ALONE CANNOT WITNESS ANYTHING: a record is created by an apply, so
        # a boot that stops applying never gets its witness either. Keep applying until
        # a delivery IS witnessed - the first moment a mismatch could be judged at all.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            step()
            seen = counters()
            if seen["readbacks_enqueued"] >= 1:
                break
            time.sleep(0.005)
    finally:
        disk_gpu_write._allocate_row = real_alloc

    assert seen["readbacks_enqueued"] >= 1, (
        "a cold boot never reached a witnessed delivery, so the kill switch is "
        f"unreachable from the production entry point: {seen}")
    assert seen["buffers_allocated"] >= 2, \
        "a readback was enqueued without the pinned rows it reads into"
    assert seen["delivered_unwitnessed"] >= 1, \
        "the cold deliveries were retroactively renamed: the gap is the finding"
    # THE ALLOCATION STAYED OFF THE STEP THREAD on the path this test drove, not one it
    # started by hand. The fix must not "solve" a cold pool by page-locking memory in
    # `start_load_kv`, which is a `cudaHostAlloc` on the step thread - the stall this
    # module exists not to cause.
    assert alloc_threads, "the pool filled without a single allocation"
    assert set(alloc_threads) == {"kvtier-disk-gpu-write"}, set(alloc_threads)
    # AND THE PASS RUNS ON A COLD BOOT: a witnessed record whose pass never ran is a
    # record that can never be judged, so the switch would still be unreachable.
    result = one_pass(worker)
    assert result["ran"] is True, result
    assert result["checked"] >= 1, f"the witness was never checked: {result}"
    assert counters()["kill_switch_fired"] == 0, \
        "a CORRECT delivery latched the switch: the latch is not evidence-independent"


def test_the_kill_switch_fires_on_a_cold_boot_with_nothing_pre_armed(
        cold_delivery, worker, tmp_path, copy_seams, prune_on):
    """THE LANE END TO END, COLD: an injected wrong copy latches the switch, and the
    prune on the other side of the boundary stops trusting this module.

    This is the sibling's kill-switch test with the two pre-arms REMOVED - no
    stubbed `_ensure_thread`, no `fill_pool` - which makes it the answer to the
    question that actually matters at ship time: *would a bad build be caught on a boot
    that has only just started serving?* The copy seam is the test's own mutation (the
    destination shifted by one block), the same shift the sibling uses and for the same
    reason: a kill switch that never fires is indistinguishable from one that was never
    wired up.
    """
    path = write_window(str(tmp_path / "kvslot-window-4"), b"\x76" * 64)
    seed_window(worker, path)
    copy_seams["shift"] = 1

    def step() -> dict:
        return disk_gpu_write.observe_apply(worker, many(path, 1), {})

    step()
    wait_for(lambda c: c["pool_refills"] >= 1,
             what="the module's thread never refilled its pool")
    # A RECORD EXISTS ONLY WHEN AN APPLY FOUND A FREE ROW, and the row is freed by the
    # pass that judges the record before it - so a cold boot needs a few applies, and
    # the pass between them, before a verdict is possible. Driving `one_pass` here is
    # the module's own production pass (the same call the sibling's tests use); the
    # thread is doing the same thing on its own tick, and whichever gets there first is
    # the same code.
    seen: dict = {}
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        step()
        worker.complete(11)
        one_pass(worker)
        seen = counters()
        if seen["readback_mismatch"] or seen["readback_equal"]:
            break
        time.sleep(0.005)
    assert seen["readbacks_enqueued"] >= 1, (
        f"a cold boot never witnessed a delivery, so the switch could not be "
        f"reached: {seen}")
    assert seen["readback_mismatch"] >= 1, (
        f"a shifted copy produced no mismatch on a cold boot: {seen}")
    wait_for(lambda c: c["readback_mismatch"] >= 1 or c["readback_equal"] >= 1,
             what="a shifted copy produced no verdict at all")

    view = disk_gpu_write.kill_switch_view()
    assert view["fired"] is True, (
        f"a cold boot's shifted copy produced {view['category']!r} instead of a latch")
    assert view["category"] == "readback_mismatch_stable"
    assert view["evidence"]["delivery"] is True, \
        "the write judged was not a delivery write"
    assert counters()["kill_switch_fired"] == 1
    # THE LATCH'S EFFECT ON THE WRITER, on a cold boot: the three throttles come back.
    limits = disk_gpu_write.limits_in_force(8)
    assert limits["delivery"] is False, limits
    assert limits["max_writes_per_step"] == disk_gpu_write.MAX_WRITES_PER_STEP
    # THE CONSUMER SEES IT TOO, in the one process that has both: the prune refuses,
    # under its own name, for the rest of this process - so the fork's copy comes back.
    disk_restore._state["prune_switch"] = None
    refusal = disk_restore.prune_adopted_restores(None, {"jobs": {}}, set())
    assert refusal["ok"] is False
    assert refusal["reason_kind"] == "kill_switch_fired"
    assert disk_restore.prune_state()["kill_switch_fired"] is True


# --- 3. what a cold boot does NOT provide, stated as a machine check ------------

def test_no_fixture_here_provides_a_thread_a_pool_or_a_resolved_env(
        cold_delivery, worker, tmp_path):
    """THE FIXTURES THAT QUIETLY PRE-ARM THE SIBLING SUITE, named and machine-checked.

    Two pieces of production state that suite hands every test:

      * `_ensure_thread`. The `witness` fixture replaces it with `lambda: True`, so in
        every delivery test the daemon thread exists before a single batch has been
        applied. Production starts it from the first armed apply, so a cold boot's first
        step has an empty pool. That line is why
        `test_an_empty_pool_costs_the_readback_and_never_the_delivery` - written as
        a cold-pool test, and a good one - still cannot assert `thread_started`: the
        thread it believes absent is a stub, not a cold start.
      * `fill_pool`. The end-to-end test
        `test_the_real_apply_point_delivers_every_entry_and_prunes_only_those`
        fills the pinned pool through the production `_refill_pools` before its first
        apply, so the step it judges has a WARM pool. On a real boot no refill can have
        happened yet, because the thread's `_resolve_worker()` answers None until that
        first apply REGISTERS the worker. That test is a correct test of a warm step; it
        is not, and does not claim to be, a cold-start test.

    What a cold boot does provide is asserted here: no thread, no rows, nothing resolved
    - and then the first delivery anyway.
    """
    before = counters()
    assert before["thread_started"] == 0
    assert before["pool_refills"] == 0
    assert before["buffers_allocated"] == 0
    assert disk_gpu_write.pool_view()["free_rows"] == {}
    assert disk_gpu_write.pool_view()["in_flight"] == 0
    assert disk_gpu_write.section()["thread_alive"] is False
    # The env is resolved by its first reader, and this is that first read.
    assert disk_gpu_write.delivery_state()["effective"] is True
    assert counters()["delivery_writes"] == 0

    path = write_window(str(tmp_path / "kvslot-window-5"), b"\x21" * 64)
    seed_window(worker, path)
    assert disk_gpu_write.observe_apply(worker, many(path, 3), {})["written"] == 3
    assert counters()["thread_started"] == 1, \
        "the cold apply did not start the thread the pool depends on"
    # THE WORKER IS REGISTERED BY THE APPLY - the fact that makes a pre-filled pool
    # unreachable from a real boot, and the reason `fill_pool` is a pre-arm.
    assert disk_gpu_write._resolve_worker() is worker
    # THE COPIES LANDED IN THE REGION THE FAKE BUILT, at the batch's own destinations:
    # "3 delivered" means three named windows, not a counter agreeing with itself.
    assert {row[2] for row in worker.journal.of("h2d")} == {
        GPU_BLOCK, GPU_BLOCK + 1, GPU_BLOCK + 2}
    assert worker.cpu_kv_caches["g0"].block_bytes(CPU_BLOCK)
