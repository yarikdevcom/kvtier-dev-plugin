r"""THE APPLY POINT MUST RETURN (0.2.57): the delivery loop's ONE wall-clock bound.

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

THE TWO FORK NUMBERS, MEASURED IN THE RUNNING FORK (the checkout at
`/home/user/Source/experiment/vllm-dev`), because 0.2.56's comment had one of them wrong
and a bound is only defensible against a real threshold:

  * `No available shared memory broadcast block found in 60 seconds` is NOT a deadline.
    `VLLM_RINGBUFFER_WARNING_INTERVAL` defaults to 60 (`vllm/envs.py:20`, `:1090`) and
    the reader logs that sentence every 60 s while it waits - and it waits INDEFINITELY
    (`SHM_READER_RECHECK_INTERVAL_MS` = 5,000 ms, `shm_broadcast.py:66`, `:739-746`). It
    is the first SYMPTOM of a stalled step thread.
  * The deadline that ended the process is `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, default
    **300 s** (`vllm/envs.py:242`), passed by `sample_tokens` at
    `multiproc_executor.py:609` and raised as the line above at `:686`. Nothing on this
    boot overrides it. (The "~118 s" in the 0.2.56 comment was the observed gap between
    log lines, not the threshold.)

0.2.57's bound is therefore ONE generous timeout, `DELIVERY_APPLY_TIMEOUT_SECONDS` =
10 s - ~30x inside the only deadline the engine enforces, ~60x above the ~0.15 s a full
735 MB restore measured - overridable without a rebuild by
`VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS`. 0.2.56's 0.25 s and 64 MiB
per-entry budgets are gone: the operator's house rule is one generous timeout rather
than fine-grained caps (the disk read/write path learned that the hard way), and a
0.25 s cap would defer ~57 of a normal 62-block restore, making the prune it gates a
no-op.

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
   thread with a bounded join. Before 0.2.57 (and before 0.2.56, which is the failure
   this reproduces) the thread is still inside `observe_apply` when the bound expires,
   and the assertion says so with the counters in its hand. The failure is bounded: the
   copies wait on an event the teardown sets, so a failing run leaves no sleeper behind,
   and the join itself is bounded, so this file can never hang CI. COLD: no stubbed
   `_ensure_thread`, no `fill_pool`, nothing pre-armed - the same discipline
   `test_disk_gpu_write_cold_delivery.py` exists to impose.
2. THE DEFERRAL IS A NAMED THROTTLE, NOT A LOSS. The entries after the stop are absent
   from `written_indices`, so `disk_restore.prune_adopted_restores` cannot remove them
   and the fork copies them; `delivery_deferred_entries`/`_steps` count them, and
   `delivery_attempted` proves the loop ran.
3. THE BOUND IS ONE TUNABLE NUMBER AND NOTHING ELSE. The shipped default is 10 s, the
   env var overrides it, an unusable value means the default and says so, and the clamp
   is published rather than silently applied. There is NO byte budget and no entry is
   too wide to copy: 0.2.56's `delivery_refused_entry_too_wide` is a REMOVED key, and
   this file pins that a whole-width entry is copied rather than refused, so the key
   cannot come back as a permanent zero.
4. THE WITNESS POSTURE IS UNTOUCHED. The bound is delivery-only; the shipped 1-per-step
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

# THE COST OF ONE SYNCHRONOUS PAGEABLE COPY. Deliberately far larger than a memmove
# and far smaller than the live box's 11.5 MB window: what matters is the RATIO to the
# bound, and the numbers below are set so that the UNBOUNDED loop (0.2.55, whose env var
# this boot cannot even read) runs ~10x the return bound while the bounded one lands
# inside it at any plausible CI load.
SLOW_COPY_SECONDS = 0.1
ENTRIES = 48
# THE BOUND THIS FILE TESTS AGAINST. The shipped 10 s is far too generous to reproduce a
# stall in a test suite, and the point of 0.2.57 is that the number is an `env` override
# rather than a rebuild - so the fixture overrides it the same way an operator would,
# which also means a regression to a hard-coded constant FAILS here instead of passing.
STEP_TIMEOUT_SECONDS = 0.5
# HOW LONG THE APPLY POINT IS ALLOWED TO TAKE. Above the bound plus one copy plus the
# whole unfixed loop's headroom below, and the join is bounded by it, so a failure costs
# seconds rather than a CI timeout.
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
    """THE COLD BOOT, and three env vars.

    No `monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)` and no
    `fill_pool(worker)`. The thread this test starts, if one starts, is the module's
    real one, and the pool is as empty as a first restore step finds it.

    THE TIMEOUT IS SET THE WAY AN OPERATOR SETS IT - through the environment - because a
    test that monkeypatched a module constant would keep passing if the constant stopped
    being readable. 0.5 s here is 1/20th of the shipped 10 s, which is the only way to
    make a copy that never returns observable in a test that also has to finish.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    # THE NAME IS RESOLVED DEFENSIVELY, and that is load-bearing rather than tidy: run
    # this file against a 0.2.55 checkout (no such constant) and a bare
    # `config.ENV_...` raises AttributeError IN THE FIXTURE - an error, and the
    # reproduction message, which is the entire artifact of the first test, never
    # prints. With the literal fallback the old build sets a variable it does not read,
    # stays unbounded, and FAILS with the message that names the mechanism.
    monkeypatch.setenv(
        getattr(config, "ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT",
                "VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS"),
        str(STEP_TIMEOUT_SECONDS))
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

def test_the_apply_point_returns_within_its_timeout_and_defers_the_rest(
        cold_delivery, slow_h2d, worker, tmp_path):
    """THE CRASH, REPRODUCED IN-PROCESS, AND THE FIX MEASURED.

    One step, ENTRIES entries, each with its own destination block, and each copy
    costing what a synchronous pageable copy costs. With no bound in force the loop's
    only limit is `MAX_ENTRIES_SCANNED`, so this thread is still inside `observe_apply`
    when the join bound expires - the 0.2.55 shape, which is what this file must keep
    failing on. With the bound in force the step delivers what fits inside it and DEFERS
    the rest, and a deferred entry is an entry the FORK still copies.
    """
    path = write_window(str(tmp_path / "kvslot-budget"), b"\x5a" * 64)
    seed_window(worker, path)
    batch = many(path, ENTRIES)
    # THE SHIPPED BOUND, READ DEFENSIVELY, because the message below is the artifact of
    # this test: interpolating the attribute directly raises AttributeError while pytest
    # formats the failure, so a run against an unbounded build would report a crash
    # instead of the reproduction. "ABSENT" names exactly the bug the text describes.
    shipped = getattr(disk_gpu_write, "DELIVERY_APPLY_TIMEOUT_SECONDS", None)
    bound_named = (f"{shipped} s shipped" if shipped is not None else
                   "ABSENT - this build has NO bound on the delivery loop at all")
    # THE VARIABLE NAME, likewise: the message must survive a build that has no such
    # constant, or the reproduction is an AttributeError again.
    timeout_var = getattr(config, "ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT",
                          "VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS")

    answer, spent, thread = _run_on_step_thread(worker, batch)
    assert not thread.is_alive(), (
        "THE 0.2.55 FAILURE, REPRODUCED: `observe_apply` had still not returned "
        f"{spent:.2f} s after {ENTRIES} entries whose copies each cost "
        f"{SLOW_COPY_SECONDS} s. It is inside its own delivery loop, whose only bound "
        f"is `MAX_ENTRIES_SCANNED = {disk_gpu_write.MAX_ENTRIES_SCANNED}` - a bound on "
        "ENTRIES, not on seconds, with the sample, the per-step cap and the interval "
        "floor all lifted by delivery mode. Each entry's `copy_(non_blocking=True)`, "
        "made out of the fork's PAGEABLE `cpu_kv_caches` "
        "(`v1/simple_kv_offload/worker.py:232`, no `pin_memory`), is SYNCHRONOUS on "
        "the engine step thread, so a restore step sits inside `start_load_kv` for "
        "entries x copy cost while the executor counts "
        "down to `RPC call to sample_tokens timed out` - the 2026-10-09 14:19 "
        "EngineCore death, whose deadline is `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` = "
        "300 s (`vllm/envs.py:242`). The fix is one generous bound checked before each "
        "entry: `DELIVERY_APPLY_TIMEOUT_SECONDS` = "
        f"{bound_named}, overridden to "
        f"{STEP_TIMEOUT_SECONDS} s by {timeout_var}. If this fails because that "
        "environment variable is not read, the bound has become a constant again. "
        f"Counters: {disk_gpu_write.counters()}")

    assert answer is not None and answer["delivery_mode"] is True
    # WHAT IT DID MANAGE, in the journal and not only in a counter.
    delivered = answer["written"]
    assert 0 < delivered < ENTRIES, (
        f"expected the bound to stop the step part-way, got {delivered} of {ENTRIES}")
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
    # THE STEP'S OWN WALL CLOCK, published: the positive proof the apply point ran,
    # which the crash window did not have. `step_seconds_peak` beside
    # `delivery_mode.timeout_seconds` is how an operator tells whether the bound has
    # EVER bitten, which is why 0.2.57 kept both keys.
    state = disk_gpu_write.delivery_state()
    assert state["step_seconds_last"] is not None
    assert state["step_seconds_last"] >= SLOW_COPY_SECONDS
    assert state["step_seconds_peak"] >= state["step_seconds_last"]
    assert spent < RETURN_BUDGET_SECONDS


def test_a_timeout_stop_leaves_the_uncovered_entries_in_the_fork_s_lists(
        cold_delivery, slow_h2d, worker, tmp_path, monkeypatch):
    """THE CONSEQUENCE THAT MAKES THE BOUND SAFE, checked at the consumer.

    With the Step-2 flag up, `prune_adopted_restores` decides whether the fork's copy
    happens, and it prunes on membership of `written_indices` alone. So the claim "a
    timeout stop costs a missed optimization and never a hole" is only true if the
    deferred ids stay OUT of that set and therefore stay in the step's lists. This hands
    the real answer to the real consumer with the real flag up - the composition 0.2.56
    proved and 0.2.57 keeps unchanged, which is the whole reason replacing two budgets
    with one timeout was safe to do.
    """
    path = write_window(str(tmp_path / "kvslot-bound-fork"), b"\x6b" * 64)
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
        f"the apply point never returned ({spent:.2f} s), so a timeout stop cannot be "
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
    # the fork will still DMA is exactly what the bound deferred - not one entry the
    # plugin failed to write is gone from them.
    assert len(report["cpu"]) == ENTRIES - delivered, report
    assert len(report["gpu"]) == ENTRIES - delivered, report
    assert disk_restore.prune_state()["unwitnessed_entries"] == 0, (
        "an entry was pruned that this module had not covered")


# --- 2. ONE bound, tunable, and no second bound in bytes ----------------------

def test_the_default_bound_is_the_shipped_timeout_and_no_entry_is_too_wide(
        cold_delivery, worker, tmp_path, monkeypatch):
    """THE DEFAULT IS 10 s, IT IS THE ONLY BOUND, AND A FULL-WIDTH ENTRY IS COPIED.

    Two halves of what 0.2.57 REMOVED, asserted together: the shipped value is a
    generous escape hatch (pinned at 10 s so a quiet regression to a 0.25 s cap fails
    here), and the byte budget's refusal is gone - an entry of the fake's full window
    `WIDTH` is copied rather than refused, exactly as it was when the budget existed and
    the test dropped the budget below the width to make it bind.

    THE KEY ASSERTION IS THE NEGATIVE ONE at the end. A removed counter must not come
    back as a permanent zero: `counters()` publishes exactly `_COUNTER_KEYS`, so a key
    that can never move is a reader trained to skim past the zeros - worse than the key
    never existing. That is the same reason the entry is asserted COPIED and not merely
    "not refused under an old name".
    """
    monkeypatch.delenv(config.ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT)
    path = write_window(str(tmp_path / "kvslot-bound-default"), b"\x7c" * 64)
    seed_window(worker, path)

    state = disk_gpu_write.delivery_state()
    assert state["timeout_seconds"] == disk_gpu_write.DELIVERY_APPLY_TIMEOUT_SECONDS
    assert disk_gpu_write.DELIVERY_APPLY_TIMEOUT_SECONDS == 10.0, (
        "the shipped bound is the number the fork's own deadline justifies (300 s "
        "RPC timeout, ~0.15 s measured restore); changing it is a decision, not a "
        "default")
    assert state["timeout_clamped"] is False

    batch = many(path, 1)
    answer = disk_gpu_write.observe_apply(worker, batch, {})
    assert answer["written"] == 1, (
        f"a full-width entry must be COPIED now, not refused: {answer}")
    seen = disk_gpu_write.counters()
    assert seen["delivery_attempted"] == 1, seen
    assert seen["delivery_writes"] == 1, seen
    assert seen["delivery_deferred_entries"] == 0, seen
    assert answer["written_indices"] == set(list(batch["jobs"])[:1]), answer
    assert "delivery_refused_entry_too_wide" not in seen, (
        "0.2.56's byte-budget refusal key is back, and with the byte budget gone it "
        "could only ever sit at zero")


def test_the_bound_is_the_operator_s_number_and_a_typo_means_the_default(
        cold_delivery, monkeypatch):
    """TUNABLE WITHOUT A REBUILD, AND THE GRAMMAR FAILS CLOSED.

    Three readings of one env var, in the same shape every other measured constant in
    this package uses (`ceiling_env`, `disk_read_env`, `settle_env`): a usable value is
    the bound in force, an absurd one is CLAMPED AND SAYS SO rather than being honoured,
    and an unusable one means the shipped default plus a report - never a step with no
    bound at all. Zero is refused by the grammar because a 0 s bound would defer every
    entry of every delivery step, which reads as "configured" while delivering nothing.

    The KEY SET is asserted stable across all of them: a published value that appears
    only when the operator set it is a reader's bug waiting to happen.
    """
    shipped = disk_gpu_write.DELIVERY_APPLY_TIMEOUT_SECONDS
    baseline = set(disk_gpu_write.delivery_state())

    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT, "2.5")
    state = disk_gpu_write.delivery_state()
    assert state["timeout_seconds"] == 2.5, state
    assert state["timeout_clamped"] is False, state

    # AN ABSURD VALUE IS A VISIBLE CLAMP, not a silent honour: past the engine's own
    # warning cadence the number describes nothing real.
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT, "600")
    state = disk_gpu_write.delivery_state()
    assert state["timeout_seconds"] == disk_gpu_write.DELIVERY_APPLY_TIMEOUT_MAX, state
    assert state["timeout_clamped"] is True, state

    # THE GRAMMAR: nothing usable means the shipped default, in every direction.
    for raw in ("", "0", "-3", "abc", "nan", "inf"):
        monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT, raw)
        state = disk_gpu_write.delivery_state()
        assert state["timeout_seconds"] == shipped, (raw, state)
        assert state["timeout_clamped"] is False, (raw, state)

    assert set(disk_gpu_write.delivery_state()) == baseline, (
        "the delivery view's keys moved with the operator's value")


# --- 3. the witness posture is not this file's business ------------------------

def test_the_witness_posture_is_unchanged_by_the_bound(cold_delivery, slow_h2d,
                                                       worker, tmp_path,
                                                       monkeypatch):
    """NO DELIVERY, NO BOUND: the shipped path still refuses rather than stalls.

    With the delivery flag down the writer is the 0.2.51 witness: one write per step, a
    0.5 s floor, and - with an empty pool and a cold boot - a refusal that RETURNS. The
    bound must not have entered that path, in either direction: no deferral counted,
    and still exactly one write for a 48-entry step.
    """
    monkeypatch.delenv(config.ENV_DISK_GPU_WRITE_DELIVERY)
    disk_gpu_write._state["delivery_resolved"] = False
    path = write_window(str(tmp_path / "kvslot-bound-witness"), b"\x76" * 64)
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
        f"the bound must not touch the witness loop: {seen}")
    # WHAT THE SHIPPED PATH ACTUALLY DID on a cold boot, named so this test cannot
    # silently stop asserting anything: the sampler took 1 entry in 64, the empty pool
    # refused it, and - the point - the call RETURNED with a name.
    assert seen["sampled_out"] >= 1, seen
    assert answer["reason"] == "no_pinned_buffer", answer
    assert seen["pool_never_filled"] == 1, seen
