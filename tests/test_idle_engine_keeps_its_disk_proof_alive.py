r"""AN IDLE ENGINE MUST NOT LOSE THE RIGHT TO SERVE FROM DISK.

THE OPERATOR'S RULE, AND THE ONE IT EXPOSES.

    "remove expiration time on disk cache etc etc, so it stays infinite till lru CACHED"

The store's files obey it. This file is about the OTHER clock in the tier - the one that
expires a PROOF, not data - and the way it was wired made an idle engine stop serving
from disk at all.

THE CHAIN, IN THE ORDER IT BREAKS.

  1. The scheduler may only offer a disk restore when a WORKER has attested that its
     restore layer is fitted. They are separate processes, and offering without that
     proof means `start_load_kv` DMAs CPU slots nobody filled: the request is served
     from STALE KV. That is the one silent failure this design has, so the gate is
     deliberate and it stays.
  2. The attestation is the rank file: `worker_diag` publishes `disk_restore`'s section
     in `rank-<rank>.json`.
  3. `disk_restore._rank_file_proves_restore` refuses a file older than
     `RANK_FILE_FRESH_SECONDS` (900 s). Also deliberate: a file a dead worker wrote
     must not vouch for it forever.
  4. AND HERE IS THE DEFECT. `worker_diag._maybe_write` is called from the wrapped
     `get_finished`, so the ONLY thing that rewrote the rank file was an ENGINE STEP. A
     worker that is idle - no request, no step, nothing to do - writes NOTHING. Its
     file ages, crosses 900 s, and the scheduler stops believing a worker that is
     perfectly alive. The tier closes its disk path on a quiet box, then needs a burst
     of steps to reopen it - which is the worst possible moment, because the first
     request after an idle spell is exactly the one that wanted the disk.

    So the operator saw the disk tier "going off" on a quiet engine. Nothing expired.
    Nothing was wrong. A HEARTBEAT WAS MISSING.

THE FIX, AND THE LINE IT MUST NOT CROSS. `worker_diag` writes the file itself, on its
own bounded daemon thread, on `HEARTBEAT_INTERVAL_SECONDS` (60 s, fifteen beats inside
the window it has to stay inside). The thread idiom is the one
`disk_gpu_write._ensure_thread` uses, INCLUDING ITS LESSON: the starter is `install()`,
not the first step. A heartbeat that needs a step to begin is the defect wearing a
thread - an idle worker is precisely a worker whose step-driven starter never runs.

What it does NOT do: it does not weaken the freshness gate (a worker that is GONE still
stops being believed, on time, which is the gate's whole value), it does not hold the
worker's GPU memory alive (the handler is referenced WEAKLY, and a heartbeat with no
handler still proves liveness), it does not invent a second document (the beat goes
through the SAME write path a step uses, so a reader can never tell the two files
apart), and it starts no thread when the diagnostics directory is unset (an inert tier
creates no files and no threads).

A PROOF IS NOT DATA. Nothing here keeps a cached byte alive; it keeps a LIVE WORKER
legible. The freshness window stays exactly the sharp gate it is.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import weakref

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import (
    config,
    disk_layout,
    disk_restore,
    disk_spill,
    worker_diag,
)

RANK = "PP0_TP0"


@pytest.fixture
def fork_worker():
    """The fork's worker class, exec'd under the fork's OWN module name.

    `install()` refuses to wrap a method whose `__module__` is not the fork's - the
    check that stops this plugin overwriting something a third party owns - so a class
    defined in this file would be (correctly) refused and every assertion below would
    be about a refusing install. The repo's own `fakes_debug.fork_class` is how the
    other `worker_diag` tests solve that, and the same fake worker supplies the arena
    numbers the document measures.
    """
    cls, _module = fake.fork_class(worker_diag.TARGET_MODULE,
                                   worker_diag.TARGET_CLASS,
                                   fake.WORKER_SOURCE, register=True)
    yield cls
    sys.modules.pop(worker_diag.TARGET_MODULE, None)


@pytest.fixture
def diag_dir(tmp_path, monkeypatch):
    """A diagnostics directory, a rank, and NO leftover thread."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANK)
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    monkeypatch.delenv(config.ENV_DISK_BYTES, raising=False)
    # A heartbeat interval short enough to OBSERVE. The production arithmetic (interval
    # inside the window, with margin) is pinned against the unpatched default in its own
    # test below, so shortening this one cannot hide a bad shipped number.
    monkeypatch.setattr(worker_diag, "HEARTBEAT_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0)
    worker_diag.reset_state_for_tests()
    disk_restore.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    yield directory
    worker_diag.stop_heartbeat()
    worker_diag.reset_state_for_tests()
    disk_restore.reset_state_for_tests()
    disk_spill.reset_state_for_tests()


def rank_file(directory, rank: str = RANK) -> str:
    return os.path.join(str(directory),
                        f"{worker_diag.RANK_FILE_PREFIX}{rank}"
                        f"{worker_diag.RANK_FILE_SUFFIX}")


def read_document(directory):
    path = rank_file(directory)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def wait_for(predicate, *, seconds: float = 5.0) -> bool:
    """Poll until the predicate holds or the budget runs out.

    The heartbeat is a real thread, so a test that asserts on its OUTPUT has to wait for
    it. Never a fixed sleep: a fixed sleep is either flaky or slow, and this repo's own
    timing-test rule is that a bound must not depend on machine jitter. The predicate is
    re-checked every 10 ms and the helper answers False rather than hanging.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def fitted_restore() -> dict:
    """A `disk_restore` contribution saying this worker's layer is fitted.

    THE REAL SEAM, not a stub: `worker_diag.register_contribution` is what
    `disk_restore` registers in production, and the gate under test reads exactly this
    section out of the file. So the test proves the gate opens for a heartbeat-written
    file, not for a shape this file invented.
    """
    return {"section": "disk_restore", "installed": True, "armed": True,
            "worker_fitted": True, "service_reason_kind": None,
            "reads_applied": 0, "root": "/tmp/not-used-by-the-gate"}


def register_fitted_worker() -> None:
    worker_diag.register_contribution("disk_restore", fitted_restore)


# ===========================================================================
# 1. THE DEFECT: NO STEPS, NO PROOF, NO DISK
# ===========================================================================

def test_an_idle_worker_keeps_its_own_rank_file_fresh(diag_dir, fork_worker):
    """THE HEADLINE. With zero engine steps, the rank file's stamp must keep moving.

    Before the fix this needed `get_finished` to be called: an idle worker wrote its
    file once and then watched it age out. The assertion is on the FILE, because the
    FILE is the proof - not on a counter that merely claims a thread ran.
    """
    register_fitted_worker()
    installed = worker_diag.install(cls=fork_worker)
    assert installed["installed"] is True, installed

    # THE FIRST PROOF, WITH NO STEP EVER CALLED. `install()` itself still does not
    # write the file (only the write path does); what changed in 0.2.69 is WHEN the
    # supervisor's first beat is due - immediately, not one interval later - because a
    # worker that installs at model-load time and proves nothing for 60 s leaves the
    # scheduler's restore gate shut for the first cold request. So a file appearing
    # here can only have come from the heartbeat, which is the whole defect in one
    # line: before 0.2.45 this directory stayed empty until the engine stepped, and
    # between 0.2.45 and 0.2.69 it stayed empty for the first interval after every
    # boot. (In production the handshake prime in `engine_side._prime_worker_proof`
    # writes it even earlier; this fixture calls `install()` directly, so what this
    # test observes is the supervisor's own first beat.)
    def written() -> bool:
        return read_document(diag_dir) is not None

    assert wait_for(written), (
        "install alone, with zero engine steps, produced no rank file: the proof "
        "existed only as a side effect of a step, which is the defect this file is "
        "for")
    first = read_document(diag_dir)
    stamp = float(first["ts"])

    # NOTHING CALLS THE ENGINE HERE. No get_finished, no step, no request.
    def rewritten() -> bool:
        document = read_document(diag_dir)
        return document is not None and float(document["ts"]) > stamp

    assert wait_for(rewritten), (
        "the rank file was written once and then left to age: an idle worker's proof "
        "passes `disk_restore.RANK_FILE_FRESH_SECONDS` "
        f"({disk_restore.RANK_FILE_FRESH_SECONDS}s) and the scheduler stops offering "
        "disk restores to a worker that is perfectly alive")

    second = read_document(diag_dir)
    assert second["writer"]["plugin"] == worker_diag.WRITER_TAG, \
        "a heartbeat must write the SAME document a step writes, not a second shape"
    assert second["rank"] == RANK
    assert int(worker_diag.counters()["heartbeats"]) >= 1


def test_the_disk_gate_stays_open_across_an_idle_spell(diag_dir, fork_worker,
                                                       monkeypatch):
    """THE OPERATOR-VISIBLE VERSION OF THE SAME BUG: `worker_proof` must not go False.

    The scheduler's question is one count over the rank files. This writes a fitted
    worker's file, makes it as stale as an idle spell makes it, and asks the gate - the
    exact sequence that closed the disk path on the live box after a quiet period.
    """
    register_fitted_worker()
    # THE ARM, the honest way. Since 0.2.28 the tier arm is a BOOT FACT read from the
    # arm file that `disk_spill.arm()` writes - the rank files are no longer asked,
    # precisely so a reboot cannot arm a cold box. So this test arms the real latch and
    # lets the proof read the real file; asserting a rank-file-only proof would test a
    # shape the reader stopped accepting.
    root = os.path.join(str(diag_dir.parent), "tier")
    os.makedirs(root, exist_ok=True)
    monkeypatch.setenv(config.ENV_DISK_ROOT, root)
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(8 * 1024 * 1024 * 1024))
    armed = disk_spill.arm()
    assert armed["armed"] is True, armed
    worker_diag.install(cls=fork_worker)

    def written() -> bool:
        return read_document(diag_dir) is not None

    assert wait_for(written), (
        "the heartbeat wrote no rank file to prove the worker alive")
    proven = disk_restore.worker_proof(force=True)
    assert proven["count"] >= 1, proven

    # Age the FILE the way an idle spell ages it. Stamping the mtime is not enough: the
    # reader uses the document's own `ts`, so both move together.
    path = rank_file(diag_dir)
    old_stamp = float(read_document(diag_dir)["ts"])
    stale = time.time() - (disk_restore.RANK_FILE_FRESH_SECONDS + 60.0)
    document = read_document(diag_dir)
    document["ts"] = round(stale, 6)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document))
    os.utime(path, (stale, stale))
    assert disk_restore.worker_proof(force=True)["count"] == 0, (
        "the gate must still refuse a stale file: this test needs that refusal to be "
        "real")

    # And the heartbeat re-opens it, with NO engine step in sight.
    def proven_again() -> bool:
        return disk_restore.worker_proof(force=True)["count"] >= 1

    assert wait_for(proven_again), (
        "a stale file stayed stale with a live worker behind it: this is the "
        "operator's 'the disk tier went off by itself', and it is a missing heartbeat, "
        "not a fault")
    assert float(read_document(diag_dir)["ts"]) > old_stamp


# ===========================================================================
# 2. STARTED AT INSTALL, NOT AT THE FIRST STEP
# ===========================================================================

def test_the_heartbeat_is_started_at_install_not_by_a_step(diag_dir, fork_worker):
    """THE 0.2.44 LESSON, APPLIED TO THE PROOF.

    A thread started by its own first customer is a thread that never starts when the
    customer needed it to have already run. `disk_gpu_write` shipped exactly that in
    0.2.43: its starter sat below the code path that needed the rows the thread
    produces, so the first write was unreachable. The same shape here is a heartbeat
    begun by a step: an IDLE worker is a worker whose starter never runs, which is this
    file's defect.

    So the starter is `install()`, and this asserts the thread is alive with ZERO calls
    to a wrapped method - and that a second install does not stack a second thread.
    """
    worker_diag.install(cls=fork_worker)
    status = worker_diag.heartbeat_status()
    assert status["started"] is True, status
    assert status["alive"] is True, status
    assert int(worker_diag.counters()["heartbeat_threads"]) == 1
    worker_diag.install(cls=fork_worker)
    assert int(worker_diag.counters()["heartbeat_threads"]) == 1, \
        "one heartbeat thread per process: a reinstall must not stack threads"


def test_the_thread_is_a_daemon_with_a_findable_name(diag_dir, fork_worker):
    """It must never hold an interpreter open, and an operator must find it.

    `threading.enumerate()` is the only way to see a daemon from outside, and "which
    thread is writing my rank file" is a question the operator debugging this WILL ask.
    """
    worker_diag.install(cls=fork_worker)
    threads = [t for t in threading.enumerate()
               if t.name == worker_diag.HEARTBEAT_THREAD_NAME]
    assert len(threads) == 1, [t.name for t in threading.enumerate()]
    assert threads[0].daemon is True, \
        "a non-daemon heartbeat thread would keep a shutting-down worker alive forever"


# ===========================================================================
# 3. NO LEAKS, NO FLOODING, NO THREADS WHEN OFF
# ===========================================================================

def test_the_heartbeat_does_not_keep_the_worker_alive(diag_dir, fork_worker):
    """A proof-of-life writer that pins the engine is a leak with good manners.

    The handler owns the CPU KV arena. Holding it strongly to make the document richer
    would make the engine's own memory un-collectable for the life of the plugin -
    gigabytes on this box. So the reference is weak, and a heartbeat with no handler
    still writes, because the FACT the file carries ("this process is alive") does not
    need the handler at all.
    """
    register_fitted_worker()
    worker_diag.install(cls=fork_worker)
    handler = fork_worker()
    worker_diag.note_handler(handler)
    ref = weakref.ref(handler)
    del handler
    gc.collect()
    assert ref() is None, (
        "worker_diag still holds the offload handler: a rank file is not worth the CPU "
        "KV arena it would keep pinned")

    def written() -> bool:
        return read_document(diag_dir) is not None

    assert wait_for(written)
    stamp = float(read_document(diag_dir)["ts"])

    def rewritten() -> bool:
        document = read_document(diag_dir)
        return document is not None and float(document["ts"]) > stamp

    assert wait_for(rewritten), (
        "a heartbeat that needs a live handler is not a heartbeat")
    document = read_document(diag_dir)
    assert document["rank"] == RANK
    assert document["arena"]["bytes_total"] == 0, \
        "with no handler the file must say it measured nothing, not invent a number"


def test_the_rate_floor_still_bounds_heartbeat_writes(diag_dir, fork_worker,
                                                      monkeypatch):
    """The heartbeat inherits a WRITE floor, so a tick cannot become a flood.

    `config.worker_min_interval_seconds()` exists because every write is an fsync'd
    atomic replace in a directory the SCHEDULER also reads. A thread on its own timer
    would be a second writer with no budget at all, so the heartbeat takes its own
    floor: an operator who set an interval of 10 ms for an experiment must not get a
    hundred fsyncs a second on the directory the offer path polls.
    """
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MIN_INTERVAL_SECONDS", 3600.0)
    worker_diag.install(cls=fork_worker)
    beat = worker_diag.heartbeat_once()
    assert beat["written"] is True, beat
    first = read_document(diag_dir)
    assert first is not None
    for _ in range(5):
        worker_diag.heartbeat_once()
    second = read_document(diag_dir)
    assert float(second["ts"]) == float(first["ts"]), (
        "the floor did not bound the heartbeat: five forced beats wrote five files, "
        "and the floor is the only thing between a write cadence and an fsync storm on "
        "the scheduler's read directory")
    assert int(worker_diag.counters()["heartbeat_floor_skips"]) >= 1


def test_no_diagnostics_directory_means_no_thread(tmp_path, monkeypatch,
                                                  fork_worker):
    """An inert tier creates no files AND no threads.

    `VLLM_E2_TIER_DIAG` unset is the tier's OFF switch, honoured today by refusing to
    wrap anything. A heartbeat started anyway would be a background writer in a process
    whose operator asked for nothing - and it would create the very files the refusal
    exists to prevent.
    """
    directory = tmp_path / "diag-off"
    directory.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANK)
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    worker_diag.reset_state_for_tests()
    report = worker_diag.install(cls=fork_worker)
    assert report["installed"] is False, report
    status = worker_diag.heartbeat_status()
    assert status["started"] is False, status
    assert status["alive"] is False
    assert os.listdir(str(directory)) == [], \
        "an inert install wrote into the diagnostics directory"
    worker_diag.reset_state_for_tests()


# ===========================================================================
# 4. THE ARITHMETIC THE FIX DEPENDS ON, AND THE SIGNAL IT IS READABLE
# ===========================================================================

def test_the_heartbeat_interval_sits_inside_the_window_it_must_stay_inside():
    """THE ONE EQUATION THIS WHOLE FILE IS.

    A proof stays alive only while `interval` fits inside the freshness window WITH
    SPARE BEATS: one lost tick (a long GIL hold, a slow device, a stop-the-world
    import) must not land the file past the gate. FOUR spare beats is the margin, so
    the shipped 60 s inside a 900 s window tolerates four consecutive dropped beats.

    Pinned against `HEARTBEAT_INTERVAL_DEFAULT` - the UNPATCHED default - because this
    file's own fixture shortens the interval to make the thread observable, and a test
    that read the patched attribute would pin nothing at all.
    """
    default_interval = worker_diag.HEARTBEAT_INTERVAL_DEFAULT
    window = disk_restore.RANK_FILE_FRESH_SECONDS
    assert default_interval == 60.0, default_interval
    assert window == 900.0, (
        "`disk_spill.PEER_RANK_FILE_FRESH_SECONDS` reads the SAME files and must move "
        "with this window, never apart from it")
    assert default_interval * (1 + worker_diag.HEARTBEAT_MIN_BEATS_IN_WINDOW) < window
    assert default_interval > worker_diag.HEARTBEAT_MIN_INTERVAL_SECONDS, \
        "a floor at or above the interval would make every other beat a silent no-op"
    counters = worker_diag.counters()
    for key in ("heartbeats", "heartbeat_threads", "heartbeat_floor_skips",
                "heartbeat_errors", "heartbeat_respawns"):
        assert key in counters, key


def test_the_heartbeat_reports_its_own_age(diag_dir, fork_worker):
    """The payload must answer "is this worker's proof being kept alive?" in one read.

    `last_heartbeat_age_seconds` is that answer. A counter that only totals beats cannot
    tell a worker beating every minute from one whose thread died an hour ago - and a
    thread that died in the night is precisely the silent failure a heartbeat thread can
    produce, so it gets an age, a bounded respawn, and an error counter.
    """
    worker_diag.install(cls=fork_worker)
    assert wait_for(lambda: worker_diag.counters()["heartbeats"] >= 1), \
        "no beat landed within the budget, so there is no age to report"
    section = worker_diag.section()
    assert section["heartbeat_interval_seconds"] == 0.05
    assert section["heartbeat_alive"] is True
    assert section["heartbeat_started"] is True
    assert section["last_heartbeat_age_seconds"] is not None
    # THE NUMBER THAT DISTINGUISHES A LIVE BEAT FROM A DEAD THREAD. Bounded by something
    # derived from the fixture's own interval rather than a magic constant, so this
    # cannot fail on machine jitter (the repo's timing-test rule) while still refusing
    # to accept a stale age.
    assert (section["last_heartbeat_age_seconds"]
            < worker_diag.HEARTBEAT_INTERVAL_SECONDS * 20), section
    assert (section["heartbeat_window_seconds"]
            == disk_restore.RANK_FILE_FRESH_SECONDS)
    assert section["heartbeat_beats_in_window"] >= 4, (
        "the payload must state the margin, so 'is this tight?' is a reading and not a "
        "calculation")


def test_a_died_thread_is_respawned_and_said(diag_dir, fork_worker):
    """A heartbeat that dies silently is the same outage as no heartbeat.

    The beat is contained, but a thread can still be lost (a `SystemExit` from a C
    extension, an interpreter quirk). The step-driven path is the respawn point, so a
    worker that resumes stepping re-arms its own heartbeat for the NEXT idle spell - and
    the counter says it happened, which is what turns "the disk went off again" from a
    theory into a reading.
    """
    worker_diag.install(cls=fork_worker)
    assert worker_diag.heartbeat_status()["alive"] is True

    # KILL IT FOR REAL rather than faking the liveness reader: a faked reader would let
    # `start_heartbeat` see the still-living thread and correctly refuse to start a
    # second one, so the test would assert a respawn no implementation could perform.
    # Stopping the thread and forgetting it is the state a dead thread really leaves.
    worker_diag.stop_heartbeat()
    status = worker_diag.heartbeat_status()
    assert status["alive"] is False, status

    # The step-driven path re-arms it. This is what a real `get_finished` call does.
    worker_diag.heartbeat_status(force=True)
    again = worker_diag.heartbeat_status()
    assert again["alive"] is True, again
    assert int(worker_diag.counters()["heartbeat_respawns"]) >= 1
    assert int(worker_diag.counters()["heartbeat_threads"]) == 2, \
        "the respawn is a new thread, and counting it is what makes the death visible"
