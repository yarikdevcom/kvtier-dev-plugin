r"""THE BEAT'S OWN CLOCK, AND THE SUSPEND THAT USED TO FREEZE IT (0.2.51).

WHAT THIS FILE IS ABOUT
-----------------------
0.2.45 gave an idle worker a heartbeat. 0.2.47 made the heartbeat's beats independent
of the engine's steps and unkillable. Both were real fixes, and on 2026-10-09 the
operator reported that an idle worker still lost its proof: `beats: 0`,
`state: alive_not_beating` seven minutes into a quiet boot, and a `no_worker_proof`
refusal after a 16.8-minute idle gap. The theory offered was "beats are issued only
while the engine takes steps".

THE MECHANISM TURNED OUT TO BE DIFFERENT, and the difference matters, because it
decides what to fix. The supervisor has always had its own clock (`_heartbeat_loop`,
which waits and then issues a beat) and no step has ever issued a beat; the live rank
files on the box say so directly, with the engine stepping about once a second:

    beats      17 -> 18 in 60.0 s          one beat per interval, not per step
    floor_skips  0                         no step ever tried to beat through the floor

What the box was doing instead is SUSPENDING. `CLOCK_MONOTONIC` (which is what
CPython's `Event.wait(timeout)` waits on) does not advance while the box sleeps;
`CLOCK_BOOTTIME` does, and the gate that consumes this file ages it on a WALL clock
(`disk_restore.py:3410`, `age = time.time() - ts`). Measured on the live host on
2026-10-09: `CLOCK_BOOTTIME - CLOCK_MONOTONIC = 19,612 s`, `suspend_stats/success=2`.
So a beat deadline set before a sleep is still unpaid when the box wakes, the beat that
must follow a wake was refused by a write floor measured on the clock the sleep skipped,
and the rank file meanwhile reported `last_beat_age_seconds: 3.0` next to a gate that
read the same file as many minutes stale. The first request after a wake - the one that
wants a disk restore more than any other - is the one the gate closes on.

WHAT THE DOUBLE MODELS, AND WHAT ONLY PRODUCTION CAN SHOW
---------------------------------------------------------
`BoxClock` below moves the three clocks the way the kernel does across a suspend:
BOOTTIME and the wall clock gain the suspended seconds, `monotonic()` does not. That
is not a metaphor for "time passed", it is the exact reading the code under test makes,
and both the floor and the gate's freshness are arithmetic on those readings.

What NO monkeypatch can model is the kernel freezing the thread mid-wait: a patched
`time` module cannot stop `threading.Event.wait()` from expiring on real monotonic
time. So no assertion here claims to reproduce "the supervisor slept through the
suspend" - that consequence is measured in production (the 19,612 s above, and the
`beats: 0` the operator read) and is fixed by construction, by keeping the deadline on
a clock the suspend cannot freeze and polling it in bounded slices. A test that
asserted the freeze by patching a clock the wait never consults would be the reverse of
this repo's standing rule - a test that provides what production does not - so it is
not written here. What IS asserted is everything the in-process code decides from the
clocks: which clock the deadline is on, which clock the floor is on, whether a resume is
detected and beaten for, and whether the gate's input comes back on its own.

THE ZERO-STEP RULE, KEPT. Nothing here pre-arms a thread, pre-fills state, or hands the
heartbeat a rank through `VLLM_E2_TIER_DIAG_RANK`: the rank is DERIVED through a
`vllm.distributed.parallel_state` double, because `rank_source: "parallel_state"` is
what the live rank files say and an env-pinned rank is a convenience production does
not have. Every test asserts `handler.step_calls == 0` where it claims to be about an
idle worker.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import types

import fakes_debug  # noqa: F401 - the shared fake-engine idiom
import pytest

# THE SCAFFOLDING IS BORROWED, NOT COPIED.
# `test_idle_worker_proof_stays_alive_and_legible`
# owns the fork-shaped worker/scheduler/connector modules (a stub defined in THIS file
# would carry the wrong `__module__` and `install()` would correctly refuse it, which is
# the trap that made an early draft of that file assert about a worker that never
# existed). Re-using it also means a change to that boot chain has to break both files.
import test_idle_worker_proof_stays_alive_and_legible as idle

from kvtier_dev_plugin import (
    config,
    cpu_capacity,
    disk_layout,
    disk_restore,
    disk_spill,
    engine_side,
    worker_diag,
)

# THE CADENCE, SCALED. Every interval-shaped number here is the shipped shape at a
# scale a suite can wait for; `test_the_shipped_numbers_still_add_up` pins the shipped
# values against the UNPATCHED constants so shortening these cannot hide a bad one.
INTERVAL_SECONDS = 0.2
RESUME_POLL_SECONDS = 0.05
FRESH_WINDOW_SECONDS = 3.0
# The freshness window is scaled with the interval; the WRITE FLOOR is not scaled at
# all (the headline test restores `HEARTBEAT_MIN_INTERVAL_DEFAULT`), because "the
# post-wake beat was refused by a floor the suspend did not age" is half the defect.
SUSPEND_SECONDS = 20.0
RANK = "PP0_TP0"


class BoxClock:
    """THE BOX'S CLOCKS, with a `suspend()` that moves only the two that move.

    `monotonic()` (CLOCK_MONOTONIC) stands still while the box sleeps;
    `clock_gettime(CLOCK_BOOTTIME)` and `time()` (the wall) do not. Modelled by adding
    the suspended seconds to the two that move, which is the only distinction any
    observer of a real box can make: BOOTTIME gained seconds that MONOTONIC did not.

    It is installed on BOTH `worker_diag.time` and `disk_restore.time`, because that is
    the situation being modelled: one box, one clock, and two processes reading it
    differently. Advancing only the writer's clock would have manufactured a clock skew
    this file is trying to remove.
    """

    def __init__(self, real):
        self._real = real
        self.suspended = 0.0
        self.suspends = 0

    # --- the three clocks the plugin reads -----------------------------------
    def monotonic(self) -> float:
        return self._real.monotonic()

    def time(self) -> float:
        return self._real.time() + self.suspended

    def clock_gettime(self, clock_id) -> float:
        if clock_id == self._real.CLOCK_BOOTTIME:
            return self._real.clock_gettime(clock_id) + self.suspended
        if clock_id == self._real.CLOCK_MONOTONIC:
            return self._real.monotonic()
        return self._real.clock_gettime(clock_id)

    # --- what the box does --------------------------------------------------
    def suspend(self, seconds: float) -> float:
        """Sleep for `seconds`: BOOTTIME and the wall advance, MONOTONIC does not."""
        self.suspended += float(seconds)
        self.suspends += 1
        return self.suspended

    # --- everything else passes through (including `time.sleep`) -------------
    def __getattr__(self, name):
        return getattr(self._real, name)


def install_parallel_state(pp: int = 0, tp: int = 0) -> None:
    """A `parallel_state` double, so the rank is DERIVED the way production derives it.

    `rank_id()`'s production path is
    `PP{get_pp_group().rank_in_group}_TP{get_tp_group().rank_in_group}`; the live rank
    files say `rank_source: "parallel_state"`. Pinning `VLLM_E2_TIER_DIAG_RANK` instead
    would take the env branch and never exercise the branch that a beat's refusal used
    to vanish into.
    """
    module = types.ModuleType("vllm.distributed.parallel_state")

    class _Group:
        def __init__(self, rank):
            self.rank_in_group = rank

    module.get_pp_group = lambda: _Group(pp)
    module.get_tp_group = lambda: _Group(tp)
    sys.modules["vllm.distributed.parallel_state"] = module


def wait_for(predicate, *, seconds: float = 8.0) -> bool:
    """Poll until the predicate holds. Never a fixed sleep (the repo's timing rule)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def rank_files(directory):
    return sorted(name for name in list_dir(directory)
                  if name.startswith(worker_diag.RANK_FILE_PREFIX)
                  and name.endswith(worker_diag.RANK_FILE_SUFFIX))


def list_dir(directory):
    try:
        return os.listdir(str(directory))
    except OSError:
        return []


def rank_document(directory):
    names = rank_files(directory)
    if not names:
        return None
    path = f"{directory}/{names[0]}"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def heartbeat_block(directory) -> dict:
    return (rank_document(directory) or {}).get("writer", {}).get(
        "heartbeat") or {}


def beat_writes(directory) -> int:
    return int(heartbeat_block(directory).get("beat_writes") or 0)


def stamps(directory, *, seconds: float) -> set:
    """The DISTINCT `ts` values the file carried during `seconds` of silence."""
    seen: set = set()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        document = rank_document(directory)
        if document is not None:
            seen.add(document.get("ts"))
        time.sleep(0.02)
    return seen


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A worker booted the way a worker boots, on a box whose clocks can be moved.

    NO ENGINE STEP, NO PRE-ARMED THREAD, NO ENV-PINNED RANK. The only things this
    fixture changes are the numbers a suite cannot wait for (the interval, the resume
    poll, the freshness window) and the three clocks `BoxClock` exists to move.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    root = tmp_path / "tier"
    root.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)     # DERIVED, not given
    monkeypatch.delenv(config.ENV_DIAG_EPOCH, raising=False)
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(8 * 1024 * 1024 * 1024))
    monkeypatch.delenv(config.ENV_DISK_BOOT_ENABLE, raising=False)

    install_parallel_state(0, 0)

    monkeypatch.setattr(worker_diag, "HEARTBEAT_INTERVAL_SECONDS",
                        INTERVAL_SECONDS)
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(worker_diag, "HEARTBEAT_RESUME_POLL_SECONDS",
                        RESUME_POLL_SECONDS)
    monkeypatch.setattr(worker_diag, "HEARTBEAT_BEAT_BUDGET_SECONDS", 0.5,
                        raising=False)
    monkeypatch.setattr(disk_restore, "RANK_FILE_FRESH_SECONDS",
                        FRESH_WINDOW_SECONDS)

    clock = BoxClock(time)
    monkeypatch.setattr(worker_diag, "time", clock)
    monkeypatch.setattr(disk_restore, "time", clock)

    worker_diag.reset_state_for_tests()
    disk_restore.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    try:
        armed = disk_spill.arm()
        assert armed["armed"] is True, armed

        # THE PRODUCTION BOOT: plugin load declines, the handshake lands the install.
        engine_side.register()
        declined = worker_diag.counters()["skip_reason"]
        assert declined and "is not imported in this process" in declined, declined
        worker, _ = fakes_debug.fork_class(
            disk_restore.WORKER_MODULE, disk_restore.WORKER_CLASS,
            idle.FORK_WORKER_SOURCE, register=True)
        handler = worker()
        connector = idle.register_fork(
            disk_restore.CONNECTOR_MODULE, idle.FORK_CONNECTOR_SOURCE,
            KVConnectorBase_V1=idle.register_fork(
                idle.FORK_BASE_MODULE, idle.FORK_BASE_SOURCE
            ).KVConnectorBase_V1).SimpleCPUOffloadConnector()
        connector.worker_handler = handler
        cpu_capacity._call_worker_report_hook(connector)

        yield {"dir": directory, "handler": handler, "clock": clock,
               "connector": connector}
    finally:
        worker_diag.stop_heartbeat()
        worker_diag.reset_state_for_tests()
        disk_restore.reset_state_for_tests()
        disk_spill.reset_state_for_tests()
        sys.modules.pop("vllm.distributed.parallel_state", None)
        for name in (disk_restore.WORKER_MODULE, disk_restore.CONNECTOR_MODULE,
                     idle.FORK_BASE_MODULE, disk_restore.SCHED_MODULE):
            sys.modules.pop(name, None)


# ===========================================================================
# 1. THE OUTAGE, REPRODUCED: the box wakes, and nothing arrives until the beat does
# ===========================================================================

def test_the_gate_reeopens_after_a_suspend_with_no_request_arriving(box,
                                                                   monkeypatch):
    """THE HEADLINE, AND THE TEST THAT FAILS BEFORE THIS FIX.

    A box that has slept longer than the freshness window wakes with a stamp that is
    stale on arrival, because the gate charges the whole sleep to it. Nothing can make
    that stamp young again; what can be fixed is whether the NEXT beat lands at once or
    only after the supervisor's deadline and write floor - both of which, before 0.2.51,
    were measured on the clock the suspend had just skipped. The SHIPPED write floor
    (`HEARTBEAT_MIN_INTERVAL_DEFAULT`, 5 s) is restored here instead of being left
    at the fixture's 0.0, because the floor is half of what this test is about: the
    one beat a sleeping box most needs was refused as "too soon after the last write",
    by a last write that happened before the sleep. A test that zeroes the floor cannot
    see that defect at all - which is why the fixture's 0.0 is overridden, not trusted.

    Pre-fix this fails twice over: the beat is refused by a floor the suspend did not
    age, so `worker_proof()` stays shut for the floor's 5 s of monotonic, and there is
    no resume detection at all, so `resumes` and `beats_after_resume` stay 0. Post-fix
    the resume is detected, the floor - kept on the same clock as the deadline - has
    already expired, and a fresh stamp reopens the gate with no engine step and no
    request.
    """
    directory = box["dir"]
    handler = box["handler"]
    # THE QUIET WINDOW THIS TEST ALWAYS IMPLICITLY REQUIRED, stated instead of hoped for.
    # A beat that is in the air when the box sleeps reaches its write after the wake, and
    # that write is an honest proof (its document is built and stamped post-wake): it can
    # leave the gate OPEN at the wake, invalidating the "the proof was already dead"
    # precondition below, and it can reopen the gate before the resume detector has been
    # polled. At the fixture's 0.2 s interval such a beat exists every 200 ms, and since
    # 0.2.69 this test arrives at the suspend with one already launched - the boot's prime
    # satisfies the wait that used to complete on a beat that had just FINISHED (measured:
    # 0/10 failures at 0.2.68, 2/10 on the 0.2.69 tree with the original setup). So: restart
    # the supervisor behind a long deadline and drain the air. The mechanism under test is
    # untouched - a resume is detected per poll, not per interval - and every assertion
    # below is exactly the one this test has always made. THE FLOOR IS RESTORED AFTER the
    # window is opened, because the shipped 5 s floor would (correctly) refuse this very
    # restart's own first beat, and the window would never open.
    monkeypatch.setattr(worker_diag, "HEARTBEAT_INTERVAL_SECONDS", 30.0)
    worker_diag.stop_heartbeat()
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0), (
        f"a beat thread will not finish: {worker_diag.heartbeat_status()}")
    assert worker_diag.start_heartbeat()["started"] is True
    writes0 = int(worker_diag.counters()["heartbeat_writes"])
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_writes"])
                    >= writes0 + 1), (
        "the heartbeat never wrote at all before the suspend: this test is about a "
        "worker whose proof was already dead, not about a suspend")
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0), (
        f"a beat is still in the air at the suspend: {worker_diag.heartbeat_status()}")
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MIN_INTERVAL_SECONDS",
                        worker_diag.HEARTBEAT_MIN_INTERVAL_DEFAULT)
    proven = disk_restore.worker_proof(force=True)
    assert proven["count"] >= 1, proven
    assert handler.step_calls == 0

    # THE BOX SLEEPS - and it sleeps in the case that matters most, a few moments
    # after a beat wrote. The line below states that state rather than hoping the
    # timing works out: the last beat is BRAND NEW on CLOCK_MONOTONIC and 20 s stale on
    # the clocks the suspend moved. A write floor kept on MONOTONIC therefore refuses
    # the post-wake beat for its whole 5 s (on a real box that slept an hour, "5 s of
    # monotonic" is 5 s of real time after the wake); a floor kept on the cadence clock
    # sees a directory untouched for the whole sleep and lets it write.
    worker_diag._state["heartbeat_last_beat_time"] = time.monotonic()
    # BOOTTIME and the wall gain 20 s; MONOTONIC does not. The freshness window in this
    # fixture is 3 s, so the wake arrives with the proof already stale.
    box["clock"].suspend(SUSPEND_SECONDS)
    assert disk_restore.worker_proof(force=True)["count"] == 0, (
        "the gate is not even closed at the wake, so this test would be measuring "
        "nothing: the suspend must be long enough to matter")

    # ...and NOTHING ELSE HAPPENS. No request, no step, no POST, no pre-armed thread.
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1,
                    seconds=1.5), (
        "the disk proof gate stayed shut after the wake with the supervisor alive: the "
        "first request to arrive would be refused a disk restore with "
        f"`no_worker_proof` - {heartbeat_block(directory)}")

    # THE RESUME IS IN THE ARTIFACT - waited for rather than read once. Even with the air
    # drained above, the write that reopens the gate and the write that carries the
    # attribution are two different threads' work, and a test that reads the file at the
    # instant the gate flips is asking a beat that has not been issued yet about itself
    # (measured on this tree: 2 runs in 10 before the setup was tightened, 0 in 10 at
    # 0.2.68). What stays asserted is unchanged and still specific: the detector beats
    # after the wake, the rank file says so, and no step did any of it. (The production
    # half of keeping that fast is the resume beat's floor exemption in `_issue_beat`,
    # pinned by `test_a_resume_beat_is_not_locked_out_by_a_write_that_raced_the_sleep`
    # below: with a warm floor the attribution is missing for its whole 5 s, which is longer
    # than any reasonable wait and is a defect in the artifact an operator reads.)
    assert wait_for(lambda: int(heartbeat_block(directory).get("resumes") or 0) >= 1), (
        f"the resume never reached the rank file: {heartbeat_block(directory)}")
    block = heartbeat_block(directory)
    assert block["beats_after_resume"] >= 1, block
    assert handler.step_calls == 0, (
        "a step served as the beat's starter: the whole point is that nothing arrived")


def test_a_resume_beat_is_not_locked_out_by_a_write_that_raced_the_sleep(box,
                                                                         monkeypatch):
    """THE WARM FLOOR BELONGS TO THE BEAT THAT RACED THE SLEEP, NOT TO THE WAKE (0.2.69).

    THE STATE, BUILT WITHOUT A RACE. A supervisor is started post-wake, so its first tick
    (due immediately since 0.2.69) writes a document whose stamp is the post-wake clock -
    which is exactly what a beat already in the air when the box slept leaves behind. That
    write is now the floor's reference, and the shipped 5 s floor is warm. The beat whose
    entire job is to say "the box came back" therefore arrives to a floor that a beat from
    the previous sleep spent, and before 0.2.69 it was refused: with this exemption removed
    THIS test fails 12 runs in 12, and the wake test above failed 2 runs in 10 on the
    0.2.69 tree where it had failed 0 in 10 at 0.2.68 - the attribution missing for the
    floor's whole 5 s, reporting `resumes: 0` beside a gate that had already reopened. The
    artifact an operator reads to debug suspend behaviour, saying the resume was never
    beaten when it was.

    SO THE RESUME BEAT IS EXEMPT FROM THE FLOOR, and only from the floor, and only for that
    cause: the ordinary beat asked for immediately afterwards is refused by the same warm
    floor, which is the assertion that keeps this a scoping rather than a bypass.
    """
    directory = box["dir"]
    monkeypatch.setattr(worker_diag, "HEARTBEAT_INTERVAL_SECONDS", 30.0)
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MIN_INTERVAL_SECONDS",
                        worker_diag.HEARTBEAT_MIN_INTERVAL_DEFAULT)
    worker_diag.stop_heartbeat()
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0)
    # THE SLEEP, then a supervisor that starts after it: a new supervisor's clock pair is
    # post-wake, so it detects no resume of its own and cannot beat for one. Every beat
    # from here is one this test asked for.
    box["clock"].suspend(SUSPEND_SECONDS)
    assert worker_diag.start_heartbeat()["started"] is True
    writes0 = int(worker_diag.counters()["heartbeat_writes"])
    # THE WRITE THAT RACED THE SLEEP, and the floor it spends.
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_writes"])
                    >= writes0 + 1), (
        f"the restarted supervisor's own first beat never wrote: "
        f"{worker_diag.heartbeat_status()}")
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0)
    warm = worker_diag.heartbeat_once()
    assert warm["written"] is False and "floor" in (warm["reason"] or ""), (
        f"the floor is not warm, so this test would assert nothing: {warm}")

    # THE RESUME BEAT WRITES ANYWAY. The wait is on the FILE, not on `heartbeat_writes`:
    # a beat is counted before it writes (0.2.51's ordering, so the first beat's own file
    # never reads `beats: 0`), so the counter can run ahead of the document by the length
    # of one write.
    before = worker_diag.counters()
    launched = worker_diag._issue_beat("resume")
    assert launched["launched"] is True, launched
    resume_target = int(before["heartbeat_beats_after_resume"]) + 1
    assert wait_for(lambda: int(heartbeat_block(directory).get("beats_after_resume")
                                or 0) >= resume_target), (
        f"the resume beat was refused by a warm floor: {worker_diag.counters()}")
    assert beat_writes(directory) >= int(before["heartbeat_writes"]) + 1, (
        f"the rank file reports no write for the resume beat: "
        f"{heartbeat_block(directory)}")

    # AND THE EXEMPTION IS SCOPED TO THE CAUSE: the next ordinary beat meets the same warm
    # floor and is refused by it.
    again = worker_diag.heartbeat_once()
    assert again["written"] is False and "floor" in (again["reason"] or ""), (
        f"the floor stopped rate-limiting ordinary beats: {again}")


def test_a_suspend_is_named_in_the_log_once(box, caplog):
    """One line per resume, naming the sleep - never one per poll.

    `asleep_notice_once` is this repo's rule everywhere else, and this is the line that
    turns "the gate was shut after the wake" into a timestamp an operator can put beside
    a request.
    """
    directory = box["dir"]
    assert wait_for(lambda: beat_writes(directory) >= 1)
    caplog.clear()
    box["clock"].suspend(SUSPEND_SECONDS)
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_resumes"]) >= 1)
    lines = [r.getMessage() for r in caplog.records
             if "woke from a" in r.getMessage()]
    assert len(lines) == 1, lines
    assert f"{SUSPEND_SECONDS:.1f}s suspend" in lines[0], lines[0]
    box["clock"].suspend(SUSPEND_SECONDS)
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_resumes"]) >= 2)
    lines = [r.getMessage() for r in caplog.records
             if "woke from a" in r.getMessage()]
    assert len(lines) == 2, f"one line per resume, not one per poll: {lines}"


def test_a_note_that_raises_cannot_end_the_beat_thread(box, monkeypatch, caplog):
    """THE LOOP'S RULE IS ABSOLUTE, AND A LOG LINE IS SUBJECT TO IT.

    "Nothing a beat can do may end this loop" was written about the beat itself, and the
    0.2.51 resume note found the exception to it: the note sat outside the `try`, so a
    note that raised ended the loop on its way past the counter it had just incremented.
    What that leaves behind is the worst possible artifact - `resumes` moved, so the
    state says the resume was seen; the log is empty, so nothing says it;
    `heartbeat_fatal` is 0, so the counter that reports a killed loop reports nothing;
    and the thread is simply gone until the respawn watchdog notices a deadline.

    So the note is guarded like the beat, and this test is the reason: a `_note_once`
    that raises must cost a log line and nothing else. The beats continue afterwards -
    which is the only observable difference between "a note failed" and "the idle path
    is dead", and the difference has to be in the artifact, not in the reading.
    """
    directory = box["dir"]
    assert wait_for(lambda: beat_writes(directory) >= 1)
    before = beat_writes(directory)

    def explode(*args, **kwargs):
        raise RuntimeError("the note exploded")

    monkeypatch.setattr(worker_diag, "_note_once", explode)
    caplog.clear()
    box["clock"].suspend(SUSPEND_SECONDS)
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_resumes"]) >= 1), (
        "the resume counter is incremented before the note, so if even that stopped, "
        "the suspend was not noticed at all and this test measures the wrong thing")

    # THE THREAD IS STILL A BEATING THREAD. Without the guard this never completes: the
    # RuntimeError propagates out of the loop body and the supervisor is gone.
    assert wait_for(lambda: beat_writes(directory) > before, seconds=3.0), (
        "the beat thread died on a raising log line: no further rank file was written "
        f"({heartbeat_block(directory)})")
    assert [r.getMessage() for r in caplog.records
            if "resume note" in r.getMessage()], (
        "the failure was swallowed: a note that raises must at least say so, because "
        "the counter it cost is the only other trace that it happened")


# ===========================================================================
# 2. THE COLD START, WITH NOTHING GIVEN THAT PRODUCTION DOES NOT HAVE
# ===========================================================================

def test_a_beat_lands_with_zero_steps_and_a_rank_nobody_handed_it(box):
    """THE COLD START, HONESTLY: nothing pre-armed, zero steps, rank DERIVED.

    This is the test the brief asks for, and its pre-fix status is stated rather than
    glossed: 0.2.45 and 0.2.47 already made this pass, so it passes here too. What it
    adds is the rank DERIVED through `parallel_state` (the live files'
    `rank_source: "parallel_state"`) instead of an env-pinned one, because the beat's
    refusal path through `rank_id()` is one of the two places a beat used to vanish
    without a trace, and an env-pinned rank never walks it.
    """
    directory = box["dir"]
    handler = box["handler"]
    assert handler.step_calls == 0
    assert wait_for(lambda: rank_document(directory) is not None, seconds=4.0), (
        "no rank file with zero engine steps: the proof would exist only as a side "
        "effect of a step")
    # THE FIRST FILE IS THE PRIME'S (0.2.69), and a prime is not a clock beat. Waiting for
    # `beat_writes >= 1` therefore returns while the supervisor's own first beat is still
    # on its way, and the counter this test exists to read - `beats_from_clock`, the beat
    # the supervisor issued because ITS CLOCK EXPIRED - is honestly 0 in that first file.
    # So wait for the counter itself, which is what the assertion is about.
    assert wait_for(lambda: int(heartbeat_block(directory).get(
        "beats_from_clock") or 0) >= 1, seconds=4.0), (
        "no supervisor-clock beat ever reached the file with zero engine steps: "
        f"{heartbeat_block(directory)}")
    assert wait_for(lambda: beat_writes(directory) >= 1, seconds=4.0)
    document = rank_document(directory)
    assert document["writer"]["rank_source"] == "parallel_state", document["writer"]
    assert document["rank"] == RANK, document["rank"]
    block = heartbeat_block(directory)
    # THE COUNTER THAT ANSWERS "IS THE IDLE PATH ALIVE?": a beat whose cause was the
    # supervisor's own clock expiring, with the engine taking no steps at all.
    assert block["beats_from_clock"] >= 1, block
    assert block["beats_from_step"] == 0, block
    assert block["beats_after_resume"] == 0, block
    # THE PARTITION, with the one term 0.2.69 had to add. Read from a SETTLED artifact: the
    # supervisor is stopped and the air drained first, because `heartbeat_state()` reads
    # each key separately and a beat counted between the `ticks` read and the cause reads
    # makes the sum momentarily outrun the ticks - a torn read, not a broken invariant, and
    # an identity must not be asserted across a torn read.
    #
    # `beats_from_prime` is the boot proof the handshake wrote, and it is the ONLY beat no
    # tick issued: keeping it out of `beats_from_clock`/`beats_after_resume` is what keeps
    # the suspend/resume statistics describing suspends and resumes rather than a boot. The
    # two inequalities are the README's own identity with the prime term added:
    # `beat_writes` may not outrun the beats that were decided, and the beats decided are
    # the ticks plus the prime.
    worker_diag.stop_heartbeat()
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0), (
        f"a beat was still in flight when the identity was read: "
        f"{worker_diag.heartbeat_status()}")
    settled = heartbeat_block(directory)
    assert settled["beats_from_prime"] == 1, settled
    assert (settled["beats_from_clock"] + settled["beats_from_step"]
            + settled["beats_after_resume"] == settled["ticks"]), settled
    assert settled["beats"] <= settled["ticks"] + settled["beats_from_prime"], settled
    assert (settled["beat_writes"]
            <= settled["ticks"] + settled["beats_from_prime"]), settled
    assert settled["cadence_clock"] == worker_diag.HEARTBEAT_CADENCE_CLOCK, settled
    assert handler.step_calls == 0


def test_a_quiet_period_longer_than_the_window_keeps_the_gates_input_fresh(box):
    """SILENCE IS NOT DEATH, AND THE GATE MUST NEVER SEE IT AS DEATH.

    A silence 2.5x the freshness window, zero steps, and the reader asked continuously:
    the gate must never close, and the stamp must never age more than about one
    interval.
    If the beat stops for any reason the window expires inside the test, so this
    assertion can only pass while the idle path is beating - which is the property, not
    an artefact of the window being generous.
    """
    directory = box["dir"]
    handler = box["handler"]
    assert wait_for(lambda: beat_writes(directory) >= 1)
    silence = FRESH_WINDOW_SECONDS * 2.5
    stale = []
    seen = set()
    oldest_age = 0.0
    deadline = time.monotonic() + silence
    while time.monotonic() < deadline:
        verdict = disk_restore.worker_proof(force=True)
        if verdict["count"] < 1:
            stale.append((time.time() - float(rank_document(directory)["ts"]),
                          verdict["reason_kind"]))
        document = rank_document(directory)
        if document is not None:
            seen.add(document["ts"])
            oldest_age = max(oldest_age, box["clock"].time()
                             - float(document["ts"]))
        time.sleep(0.02)
    assert not stale, (
        f"the gate closed {len(stale)} times during {silence:.1f}s of silence with "
        f"zero steps: {stale[:3]}")
    expected = int(silence / INTERVAL_SECONDS) // 2
    assert len(seen) >= expected, (
        f"the file carried {len(seen)} stamps in {silence:.1f}s of silence "
        f"(expected at least {expected} at a {INTERVAL_SECONDS}s interval)")
    assert oldest_age < INTERVAL_SECONDS * 4, (
        f"the stamp aged {oldest_age:.3f}s against a {INTERVAL_SECONDS}s interval")
    assert handler.step_calls == 0


# ===========================================================================
# 3. THE STEP MAY WAKE A BEAT. IT NEVER RUNS ONE.
# ===========================================================================

def test_a_step_wakes_a_beat_early_and_the_beat_never_runs_on_the_step(box,
                                                                       monkeypatch):
    """The desirable half of the design, and the 0.2.47 property kept.

    The interval is long (30 s) so the supervisor's own clock cannot be the cause of any
    beat THIS TEST'S STEP produces; `due` is then pushed into the past, which is exactly
    the state a supervisor is in when a box has just woken under it or the thread has been
    starved. One engine step then asks for a beat, and the beat must land on the beat
    thread - never inline on the thread that serves the engine, which is the defect
    0.2.47 removed and this new wake channel must not reintroduce.

    WHAT 0.2.69 CHANGED HERE is the arithmetic, not the property. This test used to
    restart the supervisor and assert `beat_writes(directory) == 0`, which rested on two
    accidents: that the counter started at zero for the whole process, and that a
    restarted supervisor wrote nothing for a whole interval. Neither is true now - the
    fixture's boot has already primed a proof and beaten at 0.2 s, and the first tick of
    a restarted supervisor is due immediately - so the assertion is stated as the exact
    account it always meant: the restart's own immediate first beat, one write, and then
    silence while the deadline is 30 s away. A loose `<= 1` would let an unattributed
    second write through, which is the thing this test exists to forbid. The count is kept
    in the process counters rather than in the rank file, because the file is written AFTER
    a beat is counted and two beats in flight can publish out of order - a lag an exact
    assertion cannot be built on once boot has two writers by design.
    """
    directory = box["dir"]
    handler = box["handler"]
    monkeypatch.setattr(worker_diag, "HEARTBEAT_INTERVAL_SECONDS", 30.0)
    worker_diag.stop_heartbeat()
    # DRAIN THE AIR BEFORE THE ACCOUNT IS OPENED, and give the old supervisor the length of
    # a poll to leave. `stop_heartbeat()` closes the launch gate and joins the thread, but a
    # loop that had already decided to beat can still be inside `_issue_beat`, and a beat
    # already launched has its own thread and its own write; either one landing inside the
    # window below would charge the restarted supervisor for a beat it never issued.
    time.sleep(0.2)                     # more than one HEARTBEAT_RESUME_POLL_SECONDS
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0), (
        f"a beat thread will not finish: {worker_diag.heartbeat_status()}")
    started = worker_diag.start_heartbeat()
    assert started["started"] is True, started

    # THE ACCOUNT IS KEPT IN THE PROCESS COUNTERS, not in the rank file. The file lags by a
    # write by construction - a beat is counted BEFORE it writes (0.2.51, so the first
    # beat's own file never reads `beats: 0`) and two beats in flight can publish out of
    # order - and 0.2.69 makes concurrent beats at boot the normal case: the handshake
    # prime and the immediately-due first tick are two writers by design. Keeping an exact
    # count off a lagging artifact is what made this assertion un-exact (measured: 1 run in
    # 15 read 3 writes where 1 beat had been issued).
    def counter(key: str) -> int:
        return int(worker_diag.counters()[key])

    writes0 = counter("heartbeat_writes")
    clock0 = counter("heartbeat_beats_clock")
    step0 = counter("heartbeat_beats_step")

    # THE ONE BEAT THE RESTART OWNS: the first tick, due immediately since 0.2.69. It is
    # counted as a CLOCK beat, which is the truth - the supervisor's own deadline issued
    # it - and the assertion after the wait is that nothing else follows.
    assert wait_for(lambda: counter("heartbeat_writes") >= writes0 + 1), (
        f"the restarted supervisor never issued its first beat: "
        f"{worker_diag.heartbeat_status()}")
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0), (
        f"the restart's own beat never finished: {worker_diag.heartbeat_status()}")
    # With the next deadline 30 s away and nothing else to prompt it, an idle worker gets
    # no further beats: not one more write, and none from any other cause.
    time.sleep(0.4)
    assert counter("heartbeat_writes") == writes0 + 1, (
        f"expected exactly one beat after the restart - the first tick the restarted "
        f"supervisor is immediately due ({writes0} + 1) - and found "
        f"{counter('heartbeat_writes')}: something other than the deadline is issuing "
        "beats, and `beats_from_step` below would be meaningless")
    assert counter("heartbeat_beats_clock") == clock0 + 1, (
        f"more than the first tick came from the supervisor's own clock: "
        f"{worker_diag.counters()}")
    assert counter("heartbeat_beats_step") == step0, (
        "a step-caused beat with zero steps taken")

    # Where a resumed or starved supervisor is found: behind its own deadline.
    worker_diag._heartbeat["due"] = 0.0
    seen_on: dict = {}
    real_produce = worker_diag._contributors["disk_restore"]["produce"]

    def watch_the_thread():
        seen_on["thread"] = threading.current_thread().name
        return real_produce()

    worker_diag.unregister_contribution("disk_restore")
    assert worker_diag.register_contribution(
        "disk_restore", watch_the_thread, owner="test")["registered"]

    result = handler.get_finished(set())     # ONE ENGINE STEP
    assert result == (set(), None), result   # the fork's own answer, untouched
    assert handler.step_calls == 1
    assert wait_for(lambda: counter("heartbeat_beats_step") >= step0 + 1), (
        f"a step found the supervisor behind its deadline and no beat landed: "
        f"{worker_diag.heartbeat_status()}")
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0)
    assert counter("heartbeat_writes") >= writes0 + 2, (
        f"the step's beat wrote nothing: {worker_diag.counters()}")
    assert counter("heartbeat_beats_clock") == clock0 + 1, (
        "a beat the step woke was attributed to the supervisor's clock")
    assert counter("heartbeat_beats_step") >= step0 + 1
    assert seen_on.get("thread", "").startswith(
        worker_diag.HEARTBEAT_BEAT_THREAD_NAME), (
        f"the beat's producer ran on {seen_on.get('thread')!r}: a beat executed inline "
        "on the stepping thread can wedge the caller that serves the engine")


def test_waking_a_stopped_heartbeat_is_a_refusal_not_a_fault(box):
    """`wake_heartbeat()` answers a question, it never raises one."""
    worker_diag.stop_heartbeat()
    report = worker_diag.wake_heartbeat()
    assert report["woken"] is False, report
    assert "stopped" in report["reason"], report
    assert worker_diag.heartbeat_is_overdue() is False, (
        "a stopped supervisor has no deadline, and 'overdue' about a supervisor that "
        "is gone would have every step set an Event nobody waits on, forever")


# ===========================================================================
# 4. THE REFUSALS THAT USED TO VANISH
# ===========================================================================

def test_a_beat_that_refuses_to_write_is_counted_and_named(box, monkeypatch):
    """THE OBSERVABILITY HOLE THAT MADE 2026-10-09 UNDECIDABLE, CLOSED.

    `heartbeat_once()` returns without writing in three ways, and before 0.2.51 only the
    floor counted it: "no diagnostics directory" and "no usable rank" returned a reason
    that the beat thread threw away. So a rank file in `alive_not_beating` read
    identically whether the supervisor never ticked, ticked and was refused by the rank
    derivation, or ticked and was refused by the floor - and the operator's question
    ("is the idle path alive at all?") could not be answered from the artifact at all.

    Here the rank derivation fails the way it fails in a worker whose distributed state
    is gone. Pre-fix: `refusals` does not exist and `state_reason` is the generic
    fallback. Post-fix: counted, and named.
    """
    directory = box["dir"]
    assert wait_for(lambda: beat_writes(directory) >= 1)
    monkeypatch.setattr(worker_diag, "rank_id",
                        lambda: (None, "parallel_state refused to answer: ValueError"))
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_refusals"]) >= 1), (
        "the beat refused to write and nothing anywhere counted it")
    block = worker_diag.heartbeat_state()
    assert block["refusals"] >= 1, block
    assert "no usable rank" in block["state_reason"], block
    assert "parallel_state refused" in block["state_reason"], block
    assert block["state"] in worker_diag.HEARTBEAT_STATES, block


def test_a_floor_skip_names_itself_instead_of_leaving_the_state_reason_generic(
        box, monkeypatch):
    """A counted counter with no reason beside it is half a counter.

    `floor_skips` existed in 0.2.47; `state_reason` stayed on its fallback text, so the
    file said "the supervisor is alive but no beat has written the file yet" about a
    supervisor that had beaten, been refused, and known why.
    """
    directory = box["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MIN_INTERVAL_SECONDS", 3600.0)
    worker_diag._heartbeat["due"] = 0.0
    worker_diag.wake_heartbeat()

    def skipped() -> bool:
        return int(worker_diag.counters()["heartbeat_floor_skips"]) >= 1

    assert wait_for(skipped), (
        "the floor never refused anything, so this test measured nothing")
    block = worker_diag.heartbeat_state()
    assert "floor" in block["state_reason"], block
    assert "no beat has written" not in block["state_reason"], block


# ===========================================================================
# 5. THE CLOCK PRIMITIVES AND THE ARITHMETIC, AGAINST THE SHIPPED VALUES
# ===========================================================================

def test_a_suspend_is_detected_from_the_two_clocks_and_nothing_else_is():
    """A suspend is the ONE ordinary event that moves BOOTTIME without MONOTONIC.

    And what must NOT be mistaken for one: both clocks advancing together (a long run),
    a clock that went backwards, a first sample with nothing to compare against. A
    false positive here would mean a beat storm on a box that never slept.
    """
    def pair(cadence, monotonic):
        return (cadence, monotonic)

    # Ordinary running: both clocks gain the same seconds.
    assert worker_diag._suspended_between(pair(100.0, 100.0),
                                          pair(160.0, 160.0)) is None
    # A suspend: BOOTTIME gained 600 s that CLOCK_MONOTONIC did not.
    assert worker_diag._suspended_between(pair(100.0, 100.0),
                                          pair(760.0, 160.0)) == pytest.approx(600.0)
    # Noise, and a backwards or unusable sample: not a suspend.
    assert worker_diag._suspended_between(pair(100.0, 100.0),
                                          pair(161.0, 160.0)) is None
    assert worker_diag._suspended_between(pair(100.0, 100.0),
                                          pair(90.0, 90.0)) is None
    assert worker_diag._suspended_between(pair(None, None),
                                          pair(1.0, 1.0)) is None


def test_the_cadence_clock_is_named_and_the_deadline_uses_it(box):
    """`cadence_clock` tells a reader which clock every age in the block is kept on.

    This is the field that makes the artifact and the gate reconcilable: the gate ages
    this file on a wall clock, so an age kept on CLOCK_MONOTONIC and an age kept on the
    wall differ by exactly the suspended seconds, and only one of them is the number the
    gate acts on. Naming the clock is what lets a reader tell those two readings apart
    instead of arguing about them at 3 a.m.
    """
    directory = box["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    block = heartbeat_block(directory)
    assert block["cadence_clock"] == worker_diag.HEARTBEAT_CADENCE_CLOCK
    assert worker_diag.HEARTBEAT_CADENCE_CLOCK in ("boottime", "monotonic")
    # On the box this ships to, the cadence clock IS boottime: the fix is not inert.
    assert worker_diag.HEARTBEAT_CADENCE_CLOCK == "boottime", (
        "this host has no CLOCK_BOOTTIME, so the suspend-proof deadline is inert - "
        "if this fails on a developer machine that is fine, on the GPU box it is not")
    # And the deadline the step path consults is on that clock.
    assert worker_diag._heartbeat["due"] > 0.0


def test_the_resume_poll_is_shorter_than_the_interval_it_bounds():
    """The bound that makes a resume noticed, pinned against the shipped interval.

    `HEARTBEAT_RESUME_POLL_SECONDS` is what turns "the deadline is on a clock the sleep
    cannot freeze" into "the beat lands within N seconds of the wake". Longer than the
    interval and it would be the thing that delays the beat; longer than the freshness
    window and the fix would be decorative. No fixture: this pins shipped constants, and
    a booted box would hand it a scaled window and make it measure the fixture.
    """
    assert worker_diag.HEARTBEAT_RESUME_POLL_SECONDS > 0.0
    assert worker_diag.HEARTBEAT_RESUME_POLL_SECONDS \
        < worker_diag.HEARTBEAT_INTERVAL_DEFAULT
    assert worker_diag.HEARTBEAT_RESUME_POLL_SECONDS \
        < disk_restore.RANK_FILE_FRESH_SECONDS
    assert worker_diag.HEARTBEAT_SUSPEND_EVIDENCE_SECONDS \
        < worker_diag.HEARTBEAT_RESUME_POLL_SECONDS * 1000, (
        "a suspend threshold above the poll bound could not be reached between polls")


def test_the_shipped_numbers_still_add_up():
    """THE ARITHMETIC THE DESIGN RESTS ON, against the UNPATCHED constants.

    60 s inside a 900 s window is fifteen beats, so four consecutive lost ticks are
    still safe; the resume poll is a fraction of the interval; and the beat budget still
    fits inside the window. This file shortens the cadence to observe a thread, so this
    is where the shipped numbers get pinned instead - which is why it takes NO fixture:
    a booted box would hand it a shortened window and it would pin the fixture.
    """
    interval = worker_diag.HEARTBEAT_INTERVAL_DEFAULT
    window = disk_restore.RANK_FILE_FRESH_SECONDS
    assert window == 900.0
    assert interval == 60.0
    assert interval * (1 + worker_diag.HEARTBEAT_MIN_BEATS_IN_WINDOW) < window
    assert window > worker_diag.HEARTBEAT_BEAT_BUDGET_SECONDS
    assert worker_diag.HEARTBEAT_MAX_INFLIGHT_BEATS >= 1
    # The floor must stay below the interval, or a beat would be refused faster than
    # the supervisor is allowed to issue one and the idle path would throttle itself.
    assert worker_diag.HEARTBEAT_MIN_INTERVAL_DEFAULT == 5.0
    assert interval > worker_diag.HEARTBEAT_MIN_INTERVAL_DEFAULT


# ===========================================================================
# 6. THE KEY SET, WHICH IS A CONTRACT
# ===========================================================================

def test_the_new_keys_are_in_every_state_and_the_block_stays_small(box):
    """0.2.47's contract, extended: 0.2.51 added SEVEN keys and no state may drop one.

    `beating` was already the shape a reader depends on; a degraded state that omits
    `beats_from_clock` is a degraded state that cannot answer the one question the new
    keys were added for.
    """
    directory = box["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    saved = dict(worker_diag._state)
    seen: dict = {}
    try:
        seen["beating"] = worker_diag.heartbeat_state()
        worker_diag.stop_heartbeat()
        seen["started_not_alive"] = worker_diag.heartbeat_state()
        worker_diag._state["heartbeat_threads"] = 0
        worker_diag._state["heartbeat_start_reason"] = "the install never asked"
        seen["never_started"] = worker_diag.heartbeat_state()
        worker_diag._state["installed"] = False
        seen["not_applicable"] = worker_diag.heartbeat_state()
        worker_diag._state["installed"] = True
        worker_diag._state["heartbeat_threads"] = 1
        worker_diag._state["heartbeat_writes"] = 0
        seen["alive_not_beating"] = worker_diag.heartbeat_state()
    finally:
        worker_diag._state.clear()
        worker_diag._state.update(saved)

    assert set(seen) == set(worker_diag.HEARTBEAT_STATES), sorted(seen)
    new = {"cadence_clock", "ticks", "beats_from_clock", "beats_from_step",
           "beats_after_resume", "refusals", "resumes"}
    assert new <= set(worker_diag.HEARTBEAT_FILE_KEYS), (
        sorted(new - set(worker_diag.HEARTBEAT_FILE_KEYS)))
    for name, block in seen.items():
        assert set(block) == set(worker_diag.HEARTBEAT_FILE_KEYS), (
            f"{name} differs from the declared key set by "
            f"{sorted(set(block) ^ set(worker_diag.HEARTBEAT_FILE_KEYS))}")
        for key in new:
            assert block[key] is not None, f"{name} dropped {key}"
    size = len(json.dumps(seen["beating"], separators=(",", ":")).encode("utf-8"))
    assert size < 1024, f"the heartbeat block costs {size} bytes of a 32 KiB file"
    whole = len(json.dumps(rank_document(box["dir"]), separators=(",", ":"),
                           default=str).encode("utf-8"))
    assert whole < worker_diag.MAX_FILE_BYTES, whole
