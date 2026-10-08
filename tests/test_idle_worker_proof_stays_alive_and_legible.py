r"""AN IDLE WORKER'S PROOF MUST STAY ALIVE *AND* BE LEGIBLE WHILE IT DOES.

WHY THIS FILE EXISTS SEPARATELY FROM `test_idle_engine_keeps_its_disk_proof_alive.py`.
That file proves the mechanism 0.2.45 shipped. It does so by calling `install()` with a
worker class handed to it, with the interval shortened, in a process where the fixture
has already arranged everything. Production says the disk tier went dark anyway - a
16.8-minute idle gap, then `no_worker_proof` on the one request that followed it, on the
box running this very fix - and the plugin could not say WHY, because the heartbeat's
counters live in the worker process and a worker answers no HTTP route: the snapshot
route is served by the API server and honestly reports `heartbeats=0 heartbeat_threads=0
installs=0` about a process it has never seen. A fix nobody can observe is a fix that
can be false forever. So this file refuses all three conveniences:

  * NO monkeypatched starter, no pre-filled state, no pre-started thread. The driver is
    `engine_side.register()` - what vLLM actually calls - followed by the handshake hook
    `cpu_capacity` calls inside the wrapped `get_handshake_metadata`. That is the ONLY
    chain by which a worker gets a rank-file writer, and an assertion about a heartbeat
    that never went through it is an assertion about a thread production cannot make.
  * ZERO ENGINE STEPS. No `get_finished`, no `bind_connector_metadata`, no request. An
    idle worker is the whole subject, and the handler's own step counter is asserted to
    still read zero.
  * THE READER'S CLOCK, not the writer's, is what moves. The gate asks
    `time.time() - document["ts"] > 900`; a suite cannot wait fifteen minutes, so the
    fixtures advance `disk_restore`'s clock while the worker keeps writing with the real
    one. Nothing about the mechanism under test is faked: the thread is real, the
    starter is the production one, the file is the production file.

AND WHAT THE 0.2.47 FIXES HERE ARE. (1) A beat used to run INLINE in the supervisor's
loop, so one contributor blocked on a lock somewhere stopped the proof dead while
`heartbeat_alive` kept answering True - the exact unobservable state the operator was
left in. A beat now gets its own bounded thread, and a wedged one costs a tick, not the
tier. (2) `except Exception` did not cover a `SystemExit` from under a C extension: one
such beat ended the proof, and the ONLY thing that re-armed it was an engine step - a
recovery path driven by the very thing whose absence is the fault, which is 0.2.44's
lesson in a third outfit. (3) The worker's own state is now published IN THE RANK FILE
(all five states, one key set), because that file is the one artifact a worker
writes. (4) The proof gate now logs ONE timestamped line per transition, naming the
reason and the measured age of the freshest rank file, and it distinguishes "every rank
file is stale" from "no rank file carries a `disk_restore` section" - two outages that
used to arrive as one counter.
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

from kvtier_dev_plugin import (
    config,
    cpu_capacity,
    disk_layout,
    disk_restore,
    disk_spill,
    engine_side,
    worker_diag,
)

RANK = "PP0_TP0"

# --- the fork, registered under the fork's OWN module names -------------------
# `install()` refuses a method whose `__module__` is not the fork's, and
# `disk_restore.install()` additionally needs `start_load_kv` in the class dict and the
# connector's inherited load-error report to live in vllm's own tree. A worker stub that
# cannot satisfy those gets a REFUSED install, and a refused install writes no rank file
# - so a suite built on the easy stub asserts about a worker that never existed. (The
# first draft of this file did exactly that: the file it wrote carried no `disk_restore`
# section, which is not evidence to the gate, so not one test in it could have watched
# the gate open. That is the same trap production fell into from the other side.)

FORK_BASE_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.base"

FORK_BASE_SOURCE = '''
class KVConnectorBase_V1:
    def get_block_ids_with_load_errors(self):
        return set()
'''

FORK_CONNECTOR_SOURCE = '''
class SimpleCPUOffloadConnector(KVConnectorBase_V1):
    pass
'''

FORK_SCHED_SOURCE = '''
class SimpleCPUOffloadScheduler:
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def build_connector_meta(self, scheduler_output=None):
        return "METADATA"

    def update_connector_output(self, connector_output):
        return None
'''

FORK_WORKER_SOURCE = '''
class _Cache:
    def __init__(self, nbytes):
        self.nbytes = nbytes


class SimpleCPUOffloadWorker:
    """The worker half of the tier: what `worker_diag` measures, plus the three methods
    `disk_restore` insists on finding in the class dict."""

    def __init__(self):
        self.cpu_kv_caches = {"g0": _Cache(4096)}
        self.cpu_capacity_bytes = 16384
        self.num_cpu_blocks = 4
        self._connector_metadata = None
        self._pending_store_event_indices = []
        self._pending_load_event_indices = set()
        self._completed_store_events = {}
        self._store_hwm = 0
        self._load_hwm = 0
        self.step_calls = 0

    def bind_connector_metadata(self, metadata):
        self.step_calls += 1
        self._connector_metadata = metadata

    def start_load_kv(self):
        self.step_calls += 1

    def get_finished(self, finished_req_ids):
        self.step_calls += 1
        return set(), None
'''

FORK_MODULES = (disk_restore.SCHED_MODULE, disk_restore.WORKER_MODULE,
                disk_restore.CONNECTOR_MODULE, FORK_BASE_MODULE)


def register_fork(name: str, source: str, **pre):
    """Exec `source` into a module NAMED `name`, registered in `sys.modules`."""
    module = types.ModuleType(name)
    module.__dict__.update(pre)
    sys.modules[name] = module
    try:
        exec(source, module.__dict__)  # noqa: S102 - our own source, never input
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def fork():
    """The fork's worker, scheduler and connector - imported only WHEN TOLD TO BE.

    The deferral is the point. `engine_side.register()` runs in a worker BEFORE the
    connector imports its worker module, which is why the plugin-load install declines
    there and the retry rides the handshake. A fixture that imported the fork first
    would make `register()` itself install the writer, and every assertion about "the
    boot declines, then the handshake lands it" would be about a boot order production
    does not have. So `load()` is what stands for the connector's `import`.
    """
    saved = {name: sys.modules.get(name) for name in FORK_MODULES}
    loaded: dict = {}

    def load():
        if loaded:
            return loaded
        base = register_fork(FORK_BASE_MODULE, FORK_BASE_SOURCE)
        register_fork(disk_restore.SCHED_MODULE, FORK_SCHED_SOURCE)
        worker, _ = fakes_debug.fork_class(disk_restore.WORKER_MODULE,
                                           disk_restore.WORKER_CLASS,
                                           FORK_WORKER_SOURCE, register=True)
        connector = register_fork(
            disk_restore.CONNECTOR_MODULE, FORK_CONNECTOR_SOURCE,
            KVConnectorBase_V1=base.KVConnectorBase_V1
        ).SimpleCPUOffloadConnector
        loaded.update({"worker": worker, "connector": connector})
        return loaded

    try:
        yield {"load": load}
    finally:
        worker_diag.uninstall()
        disk_restore.uninstall()
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


# --- a reader's clock that can be moved, so a test can wait 15 minutes --------

class FastClock:
    """A stand-in for the `time` module whose WALL clock runs at an offset.

    Only `disk_restore`'s clock moves, and only its `time()`. The worker keeps writing
    `ts` with the real clock, so advancing the reader's clock by 900 s is exactly what
    fifteen idle minutes do to the gate's arithmetic - and nothing else in either module
    changes. `monotonic()` and everything else pass straight through, because the
    freshness rule is the only wall-clock judgement on this path.
    """

    def __init__(self, real):
        self._real = real
        self.offset = 0.0

    def time(self) -> float:
        return self._real.time() + self.offset

    def monotonic(self) -> float:
        return self._real.monotonic()

    def advance(self, seconds: float) -> float:
        self.offset += float(seconds)
        return self.offset

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def clock(monkeypatch):
    shifted = FastClock(time)
    monkeypatch.setattr(disk_restore, "time", shifted)
    return shifted


def wait_for(predicate, *, seconds: float = 8.0) -> bool:
    """Poll until the predicate holds. Never a fixed sleep (the repo's timing rule)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


# --- the boot, driven the way a worker drives it ------------------------------

def boot_worker(directory, load_fork):
    """THE PRODUCTION BOOT OF A WORKER, in the order the fork performs it.

    1. `engine_side.register()` - what the `vllm.general_plugins` entry point calls in
       EVERY process. In a worker the connector has not imported its worker module yet,
       so the rank-file writer REFUSES here; that refusal is part of the boot, so it is
       asserted rather than arranged away.
    2. `load_fork()` stands for the connector importing `simple_kv_offload/worker.py`,
       which is the first moment the class exists in this process.
    3. The handshake. `cpu_capacity._call_worker_report_hook(connector)` is the call the
       wrapped `get_handshake_metadata` makes - the one moment a worker has an
       offload handler, and therefore the only moment `worker_diag.ensure_installed`
       runs in that process. `register()` itself wired that hook; nothing here does.

    NO ENGINE STEP IS TAKEN. The handler's own `step_calls` counter proves it.
    """
    engine_side.register()
    refused = worker_diag.counters()["skip_reason"]
    assert refused and "is not imported in this process" in refused, (
        f"expected the plugin-load install to decline in a worker, got {refused!r}")
    assert worker_diag.heartbeat_status()["alive"] is False, (
        "a heartbeat started before the worker class exists would write a rank file "
        "claiming a layer that was never fitted - which is the `not_applicable` state")
    assert not rank_files(directory), (
        "a pre-fit rank file would report `not fitted` and VETO every other rank")

    fork = load_fork()
    handler = fork["worker"]()
    connector = fork["connector"]()
    connector.worker_handler = handler
    cpu_capacity._call_worker_report_hook(connector)
    return handler, connector


@pytest.fixture
def booted(tmp_path, monkeypatch, fork, clock):
    """A worker booted to the point where its proof must be alive, tier armed."""
    directory = tmp_path / "diag"
    directory.mkdir()
    root = tmp_path / "tier"
    root.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANK)
    monkeypatch.delenv(config.ENV_DIAG_EPOCH, raising=False)
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(8 * 1024 * 1024 * 1024))
    monkeypatch.delenv(config.ENV_DISK_BOOT_ENABLE, raising=False)
    # The cadence, not the starter. `HEARTBEAT_INTERVAL_SECONDS` is what the thread
    # waits on, so a test that left it at the shipped 60 s would simply be slow;
    # `test_the_shipped_cadence_still_sits_inside_the_window_it_must_stay_inside` pins
    # the shipped number against the UNPATCHED default, so shortening this one cannot
    # hide a bad shipped value.
    monkeypatch.setattr(worker_diag, "HEARTBEAT_INTERVAL_SECONDS", 0.2)
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0)
    # `raising=False` because this file must run - informatively - against the 0.2.46
    # code as well as the fixed one. If the fixture died on a missing constant, every
    # test below would ERROR in setup and a reader could not tell a fix that is absent
    # from a suite that never got to its first assertion.
    monkeypatch.setattr(worker_diag, "HEARTBEAT_BEAT_BUDGET_SECONDS", 0.5,
                        raising=False)
    worker_diag.reset_state_for_tests()
    disk_restore.reset_state_for_tests()
    disk_spill.reset_state_for_tests()

    handler = None
    connector = None
    armed = disk_spill.arm()
    assert armed["armed"] is True, armed
    handler, connector = boot_worker(directory, fork["load"])
    try:
        yield {"dir": directory, "handler": handler, "connector": connector,
               "clock": clock, "fork": fork}
    finally:
        worker_diag.stop_heartbeat()
        worker_diag.reset_state_for_tests()
        disk_restore.reset_state_for_tests()
        disk_spill.reset_state_for_tests()


def rank_files(directory):
    return sorted(name for name in os.listdir(str(directory))
                  if name.startswith(worker_diag.RANK_FILE_PREFIX)
                  and name.endswith(worker_diag.RANK_FILE_SUFFIX))


def rank_document(directory):
    names = rank_files(directory)
    if not names:
        return None
    with open(os.path.join(str(directory), names[0]), "r",
              encoding="utf-8") as handle:
        return json.load(handle)


def rewrite_rank_file(directory, mutate) -> dict:
    """Apply `mutate` to the document and replace the file with the result."""
    document = rank_document(directory)
    assert document is not None, "no rank file to rewrite"
    mutate(document)
    path = os.path.join(str(directory), rank_files(directory)[0])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document))
    return document


def replace_section(name: str, produce) -> None:
    """Swap ONE contributor's producer out and put this test's in.

    The seam is full at six by design (the boot registers all six), so a test that wants
    a hostile contributor must take a slot rather than ask for a seventh - which is the
    same rule production has, and the reason the cap and the allow-list are asserted
    against each other.
    """
    worker_diag.unregister_contribution(name)
    report = worker_diag.register_contribution(name, produce, owner="test")
    assert report["registered"] is True, report


def stamps(directory, *, seconds: float) -> int:
    """How many DISTINCT `ts` values the file carried during `seconds` of quiet."""
    seen = set()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        document = rank_document(directory)
        if document is not None:
            seen.add(document.get("ts"))
        time.sleep(0.02)
    return len(seen)


def proof_gate_lines(caplog) -> list:
    return [record.getMessage() for record in caplog.records
            if "DISK[proof] GATE" in record.getMessage()]


# ===========================================================================
# 1. THE PRODUCTION BOOT: no steps, no conveniences, and a proof that lives
# ===========================================================================

def test_a_worker_booted_through_the_real_entry_point_keeps_its_proof_fresh(
        booted):
    """ZERO ENGINE STEPS, nothing pre-armed, and the rank file's stamp keeps moving.

    The chain is the one a worker walks: plugin load declines, the handshake lands the
    install, `install()` starts the supervisor. Everything before the first assertion is
    `register()` plus the handshake hook; the only thing the fixture shortens is the
    interval the thread waits on.
    """
    handler = booted["handler"]
    directory = booted["dir"]
    assert handler.step_calls == 0, (
        "this test is about an IDLE worker; if a step ran, every assertion below is "
        "about a worker production does not have")

    def written() -> bool:
        return rank_document(directory) is not None

    assert wait_for(written), (
        "the handshake installed the writer and the heartbeat, and still no rank file "
        "appeared: the proof would exist only as a side effect of an engine step, "
        "which is the defect 0.2.45 claimed to end")
    first = float(rank_document(directory)["ts"])

    def rewritten() -> bool:
        document = rank_document(directory)
        return document is not None and float(document["ts"]) > first

    assert wait_for(rewritten), "the file was written once and then left to age"
    assert handler.step_calls == 0, "a beat must not need the engine"
    status = worker_diag.heartbeat_status()
    assert status["alive"] is True and status["started"] is True, status
    # THE FILE MUST BE EVIDENCE, not merely fresh: a fresh file with no `disk_restore`
    # section is exactly as useless to the gate as a stale one.
    assert "disk_restore" in rank_document(directory), (
        f"the file carries {sorted(rank_document(directory))} - the gate reads only "
        "`disk_restore`, so a worker this far from production still closes the tier")


def test_the_install_report_carries_the_heartbeat_verdict_into_the_boot_line(booted):
    """`worker hook install ... YES+heartbeat=YES`, per rank.

    Until 0.2.47 the boot line reported the INSTALL and dropped the HEARTBEAT:
    `_brief_report` kept only YES/NO, so a worker whose proof-of-life thread had just
    failed to start read as a clean install. Three strings now - `YES`,
    `YES+heartbeat=NO(...)`, `NO(...)` - so "did it come up in this worker?" is a grep
    over the boot log and not a theory.
    """
    report = worker_diag.install(cls=type(booted["handler"]))
    assert report["installed"] is True, report
    assert report["heartbeat"]["started"] is True, report
    assert engine_side._brief_report(report) == "YES+heartbeat=YES", report
    refused = dict(report)
    refused["heartbeat"] = {"started": False, "reason": "no threads left"}
    assert engine_side._brief_report(refused) == (
        "YES+heartbeat=NO(no threads left)"), refused
    inert = {"installed": False, "reason": "the worker module is not imported here"}
    assert engine_side._brief_report(inert) == (
        "NO(the worker module is not imported here)")


def test_a_heartbeat_that_refused_to_start_is_said_once_at_warning(
        booted, monkeypatch, caplog):
    """The refusal gets a line, once, naming the reason - never one per poll.

    `start_heartbeat()` had always returned a reason; nobody printed it. With the
    diagnostics directory gone (the tier's own off switch) this is the state an operator
    would otherwise discover only as a stale rank file fifteen minutes later.
    """
    monkeypatch.setattr(worker_diag.config, "diagnostics_dir", lambda: None)
    report = worker_diag.start_heartbeat()
    assert report["started"] is False, report
    assert worker_diag.counters()["heartbeat_start_reason"] == report["reason"]
    worker_diag.note_heartbeat_refusal(report["reason"])
    worker_diag.note_heartbeat_refusal(report["reason"])
    lines = [record.getMessage() for record in caplog.records
             if "heartbeat did NOT start" in record.getMessage()]
    assert len(lines) == 1, lines
    assert "VLLM_E2_TIER_DIAG" in lines[0], lines[0]
    assert int(worker_diag.counters()["heartbeat_start_failures_logged"]) == 2


# ===========================================================================
# 2. THE GATE, ACROSS AN IDLE SPELL, AND THE TWO WAYS IT MUST STILL CLOSE
# ===========================================================================

def test_the_gate_stays_open_through_idle_spells_nearly_the_length_of_the_window(
        booted):
    """THE OPERATOR'S OUTAGE. Two idle spells, each 60 s shorter than the window, with
    a real beat landing in each. The rank file's stamp is the proof; `worker_proof()` is
    the scheduler's actual question. Zero steps throughout.
    """
    directory = booted["dir"]
    clock = booted["clock"]
    assert wait_for(lambda: rank_document(directory) is not None)
    proven = disk_restore.worker_proof(force=True)
    assert proven["count"] >= 1, proven
    assert proven["stale_rank_files"] == 0, proven

    for spell in (1, 2):
        # Move the READER'S clock to just inside the window: the state of a box that has
        # been quiet for almost fifteen minutes. Then let the worker take one real beat.
        clock.advance(disk_restore.RANK_FILE_FRESH_SECONDS - 60.0)
        assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1), (
            f"idle spell {spell}: the gate closed on an idle worker whose heartbeat is "
            "running - the disk tier going off by itself on a quiet box")
        # ...and back, because the spell ended; the next spell starts from now.
        clock.advance(-(disk_restore.RANK_FILE_FRESH_SECONDS - 60.0))
        assert wait_for(lambda: rank_document(directory) is not None)
    assert booted["handler"].step_calls == 0


def test_a_worker_that_is_genuinely_gone_stops_being_believed(booted, caplog):
    """THE GATE'S WHOLE VALUE, KEPT. A dead worker must NOT keep the disk tier open.

    The heartbeat renews the file because the PROCESS is alive; stop the process's beat
    and the file must age out and the gate must shut, with the closure line naming the
    measured age so an operator can see WHEN the proof died.
    """
    clock = booted["clock"]
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1)
    caplog.clear()

    worker_diag.stop_heartbeat()          # the beat supervisor is really gone
    clock.advance(disk_restore.RANK_FILE_FRESH_SECONDS + 60.0)
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["count"] == 0, verdict
    assert verdict["reason_kind"] == "no_worker_proof", verdict
    assert verdict["stale_rank_files"] == 1, verdict
    assert (verdict["youngest_rank_file_age_seconds"]
            > disk_restore.RANK_FILE_FRESH_SECONDS), verdict

    closures = [line for line in proof_gate_lines(caplog) if "GATE CLOSED" in line]
    assert len(closures) == 1, closures
    assert "reason_kind=no_worker_proof" in closures[0]
    assert "stale=1" in closures[0]


def test_a_clock_ahead_rank_file_is_still_refused(booted):
    """A stamp in the future is not freshness. The one-sided rule stays one-sided."""
    directory = booted["dir"]
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1)
    worker_diag.stop_heartbeat()
    rewrite_rank_file(directory, lambda document: document.update(
        {"ts": time.time() + 3600.0}))
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["count"] == 0, verdict
    assert verdict["future_stamped_rank_files"] == 1, verdict


# ===========================================================================
# 3. THE 0.2.47 FIXES: a beat can no longer stop the proof, or end the thread
# ===========================================================================

def test_a_wedged_beat_cannot_stop_the_proof(booted):
    """ONE BLOCKED CONTRIBUTOR MUST NOT CLOSE THE DISK TIER.

    A beat asks six other modules for their sections and writes an fsync'd replace. One
    of them may hold a lock that a spill, reader or unlink thread owns across a long
    NVMe pass - and until 0.2.47 the beat ran INLINE in the supervisor's loop, so a
    producer that blocks does not slow the proof, it STOPS IT, while `heartbeat_alive`
    keeps answering True because the thread is alive and parked. That is the state this
    repo was left in on 2026-10-21: a stale rank file, every readable field saying
    "alive", and no way to tell the two apart.

    The FIRST call wedges and every later call answers at once: the difference between a
    slow device and a deadlock, and the case a supervisor must survive. Pre-fix the loop
    is parked inside the first beat and the file never moves again.
    """
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    state = {"calls": 0}

    def wedges_the_first_time():
        state["calls"] += 1
        if state["calls"] == 1:
            time.sleep(1.5)      # three times the test's beat budget
        return {"installed": True, "files_written": 1}

    replace_section("disk", wedges_the_first_time)
    moved = stamps(directory, seconds=1.4)
    assert state["calls"] >= 2, (
        "no beat after the wedged one was even issued: the supervisor is parked inside "
        "the blocked beat, which is the pre-0.2.47 shape")
    assert moved >= 2, (
        f"the rank file moved {moved} times in 1.4 s while a contributor was wedged: a "
        "beat that blocks inside the supervisor's loop stops the proof dead, and a "
        "stale rank file closes the disk path on a worker that is perfectly alive")
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_stalled"]) >= 1), (
        "a beat that overran its budget must be COUNTED, or the operator sees a stale "
        "file with no explanation of it")
    assert int(worker_diag.counters()["heartbeat_fatal"]) == 0
    assert worker_diag.heartbeat_status()["alive"] is True


def test_a_beat_that_dies_on_a_base_exception_does_not_end_the_proof(booted,
                                                                    monkeypatch):
    """`except Exception` is not enough for the only clock an idle worker has.

    A `SystemExit` (a C extension, a library that decides to exit) used to escape the
    beat, escape `_contained_heartbeat` and escape the loop - and the ONLY respawner was
    `heartbeat_status(force=True)` inside the wrapped `get_finished`, an ENGINE STEP. An
    idle worker takes no steps, so a heartbeat lost during an idle spell stayed lost for
    the whole spell: a recovery path driven by the very thing whose absence is the
    fault, which is 0.2.44's cold-start bug wearing a third outfit.
    """
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    real = worker_diag.heartbeat_once
    armed = {"fatal": True, "raised": False}

    def sometimes(*args, **kwargs):                    # noqa: ARG001 - a stub
        if armed["fatal"]:
            armed["raised"] = True
            raise SystemExit("the extension decided this process should exit")
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_diag, "heartbeat_once", sometimes)
    assert wait_for(lambda: armed["raised"]), "the beat was never attempted"
    # Give the supervisor a few ticks to die in the shape that cannot recover: pre-fix
    # the exception ends the thread and NOTHING in an idle worker starts another.
    assert wait_for(lambda: not worker_diag.heartbeat_status()["alive"],
                    seconds=1.2) is False, (
        "a supervisor that can be killed by one beat is the failure 0.2.47 removes: "
        "the thread is gone, and the only thing that re-arms it is an engine step")
    assert worker_diag.heartbeat_status()["alive"] is True
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_fatal"]) >= 1), (
        "the beat died and nothing counted it")

    armed["fatal"] = False
    moved = stamps(directory, seconds=1.0)
    assert moved >= 2, (
        f"the rank file moved {moved} times after the beat could write again: the "
        "supervisor is not just alive, it must still be beating")


def test_a_fatal_contributor_costs_its_own_section_and_not_the_proof(booted):
    """A contributor is a GUEST. It may cost its own section and nothing else.

    A `BaseException` from someone else's producer used to propagate out of
    `build_document`, out of `write_snapshot` (which catches `Exception`), out of the
    beat and out of the thread. One guest raising the wrong class could shut the disk
    path for an entire idle spell. Now it loses its section, the file is still written,
    and the supervisor never notices.
    """
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)

    def fatal():
        raise SystemExit("the contributor is not having a good day")

    replace_section("disk", fatal)
    moved = stamps(directory, seconds=1.0)
    assert moved >= 2, (
        "a contributor raising SystemExit stopped the rank file: the proof of life is "
        "not that contributor's to cancel")
    errors = worker_diag.contributions()["errors"]
    assert errors.get("disk", 0) >= 1, errors
    written = rank_document(directory)["writer"]["contributions"]["written"]
    assert "disk" not in written, written
    assert worker_diag.heartbeat_status()["alive"] is True


def test_the_in_flight_bound_stops_a_wedged_beat_becoming_a_thread_leak(booted,
                                                                       monkeypatch):
    """`HEARTBEAT_MAX_INFLIGHT_BEATS` is what keeps the fix from being a leak.

    If every beat wedges, launching a fresh thread per tick is how a diagnostic turns a
    stalled proof into thousands of parked threads. Past the bound the supervisor says
    `jammed`, stops launching, and the rank file carries the word.
    """
    monkeypatch.setattr(worker_diag, "HEARTBEAT_MAX_INFLIGHT_BEATS", 2, raising=False)
    release = threading.Event()

    def wedges_forever():
        release.wait(30.0)
        return {"installed": True}

    replace_section("disk", wedges_forever)
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_jammed"]) >= 1,
                    seconds=8.0), (
        "beats kept being issued with no bound: the supervisor must stop launching and "
        "say `jammed` instead of parking threads for the life of the process")
    inflight = int(worker_diag.counters()["heartbeat_inflight"])
    assert inflight <= 2, f"{inflight} beats in flight against a bound of 2"
    release.set()


# ===========================================================================
# 4. OBSERVABILITY: the worker's own state, in the file the worker writes
# ===========================================================================

def test_the_rank_file_says_the_worker_is_beating(booted):
    """`beating`: alive AND its last beat wrote. The state the operator wants."""
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)

    def beating() -> bool:
        document = rank_document(directory)
        block = (document or {}).get("writer", {}).get("heartbeat") or {}
        return block.get("state") == "beating"

    assert wait_for(beating), (
        f"the rank file says {rank_document(directory)['writer'].get('heartbeat')!r}; "
        "a worker has no HTTP route, so if the file does not say it is keeping itself "
        "alive, nothing anywhere does")
    block = rank_document(directory)["writer"]["heartbeat"]
    assert block["alive"] is True and block["started"] is True
    assert block["beat_writes"] >= 1
    assert block["writer_pid"] == os.getpid()
    assert block["window_seconds"] == disk_restore.RANK_FILE_FRESH_SECONDS


def test_the_rank_file_says_when_the_thread_started_and_then_died(booted):
    """`started_not_alive`, with a reason naming what re-arms it."""
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    worker_diag.stop_heartbeat()
    block = worker_diag.heartbeat_state()
    assert block["state"] == "started_not_alive", block
    assert "gone" in block["state_reason"], block
    assert block["started"] is True and block["alive"] is False


def test_the_rank_file_says_when_the_beat_never_ran_and_why():
    """`never_started`, and the reason is the one that refused it.

    The case nobody could see: a process where the install landed and the heartbeat did
    not. Before 0.2.47 the reason was returned inside the install report and dropped by
    the log-line formatter.
    """
    worker_diag.reset_state_for_tests()
    worker_diag._state["installed"] = True
    worker_diag._state["heartbeat_start_reason"] = "no threads left in this process"
    try:
        block = worker_diag.heartbeat_state()
        assert block["state"] == "never_started", block
        assert "no threads left" in block["state_reason"], block
        assert block["started"] is False and block["alive"] is False
    finally:
        worker_diag.reset_state_for_tests()


def test_the_rank_file_says_this_process_is_not_a_worker_writer():
    """`not_applicable`: a process that fitted nothing keeps no proof to keep alive."""
    worker_diag.reset_state_for_tests()
    worker_diag._state["installed"] = False
    worker_diag._state["skip_reason"] = "SimpleCPUOffloadWorker absent here"
    try:
        block = worker_diag.heartbeat_state()
        assert block["state"] == "not_applicable", block
        assert "SimpleCPUOffloadWorker absent here" in block["state_reason"], block
    finally:
        worker_diag.reset_state_for_tests()


def test_the_rank_file_says_alive_and_not_beating(booted, monkeypatch):
    """`alive_not_beating`: THE STATE THAT WAS INVISIBLE, and the one to look for.

    A thread that is alive and not writing is a stalled proof. The file must say so
    while it is still fresh, rather than leaving an operator to infer it from a stale
    file beside a log line that says `alive`.
    """
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    monkeypatch.setitem(worker_diag._state, "heartbeat_writes", 0)
    monkeypatch.setitem(worker_diag._state, "heartbeat_last_beat_reason",
                        "floor 3600s not elapsed since the last beat")
    block = worker_diag.heartbeat_state()
    assert block["state"] == "alive_not_beating", block
    assert "floor" in block["state_reason"], block
    assert block["alive"] is True


def test_the_heartbeat_block_carries_the_same_keys_in_every_state(booted):
    """A reader must be able to read the block WITHOUT knowing which state it is in.

    All five states, one key set, asserted as sets - the same contract the proxy host's
    reader holds every other field to.
    """
    directory = booted["dir"]
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
    for name, block in seen.items():
        assert set(block) == set(worker_diag.HEARTBEAT_FILE_KEYS), (
            f"{name} differs from the declared key set by "
            f"{sorted(set(block) ^ set(worker_diag.HEARTBEAT_FILE_KEYS))}")
        assert block["state"] in worker_diag.HEARTBEAT_STATES, name
    assert seen["beating"]["state"] == "beating"
    assert seen["not_applicable"]["state"] == "not_applicable"


def test_the_heartbeat_block_is_declared_once_and_stays_within_its_budget(booted):
    """The block is small, and its key set is a DECLARATION, not an accident.

    Six contributors can ask for 24 KiB of a 32 KiB file (the seam's own arithmetic), so
    the writer's own pulse must be hundreds of bytes, not thousands - and the file as a
    whole must stay under the cap the proxy host's reader refuses to read past.
    """
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    block = rank_document(directory)["writer"]["heartbeat"]
    assert set(block) == set(worker_diag.HEARTBEAT_FILE_KEYS)
    size = len(json.dumps(block, separators=(",", ":")).encode("utf-8"))
    assert size < 1024, f"the heartbeat block costs {size} bytes of a 32 KiB file"
    whole = len(json.dumps(rank_document(directory),
                           separators=(",", ":"), default=str).encode("utf-8"))
    assert whole < worker_diag.MAX_FILE_BYTES, whole


# ===========================================================================
# 5. THE GATE'S OWN LOG: one line per transition, with the age and the reason
# ===========================================================================

def test_the_boot_says_which_side_the_proof_gate_started_on(booted, caplog):
    """`GATE AT BOOT`, once. The posture that must not be mistaken for a closure.

    The complaint this answers is exact: one `no_worker_proof` in the request log cannot
    be told apart from a gate that has been shut since the boot, because the counter
    says how many and never when. The first read therefore reports a POSTURE and never
    a transition - and it is the posture as measured, including the honest `open=False`
    of a scheduler asked before any worker had taken its first beat.
    """
    directory = booted["dir"]
    disk_restore.worker_proof(force=True)      # a read with no rank file on disk yet
    boot_lines = [line for line in proof_gate_lines(caplog) if "GATE AT BOOT" in line]
    assert len(boot_lines) == 1, proof_gate_lines(caplog)
    assert "open=False" in boot_lines[0] and "rank_files=0" in boot_lines[0], (
        boot_lines[0])
    assert not [line for line in proof_gate_lines(caplog) if "GATE CLOSED" in line], (
        "the first read is a posture, not a closure: an operator must not count it as "
        "an outage that began and ended between two reads")

    assert wait_for(lambda: rank_document(directory) is not None)
    disk_restore.worker_proof(force=True)
    disk_restore.worker_proof(force=True)
    disk_restore.worker_proof(force=True)
    lines = [line for line in proof_gate_lines(caplog) if "GATE AT BOOT" in line]
    assert len(lines) == 1, proof_gate_lines(caplog)
    openings = [line for line in proof_gate_lines(caplog) if "GATE OPEN" in line]
    assert len(openings) == 1, proof_gate_lines(caplog)
    assert "fresh=1" in openings[0], openings[0]


def test_a_gate_closure_logs_exactly_one_line_naming_the_reason_and_the_age(
        booted, caplog):
    """ONE timestamped line for the closure, naming WHY and HOW LONG.

    The reason kind, the per-file counts, and the freshest rank file's age AND wall
    clock - the last being the number the request log could not give: it turns "a
    refusal at 23:51" into "the last proof was written at 23:34".
    """
    clock = booted["clock"]
    assert wait_for(lambda: rank_document(booted["dir"]) is not None)
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1)
    caplog.clear()
    worker_diag.stop_heartbeat()
    clock.advance(disk_restore.RANK_FILE_FRESH_SECONDS + 60.0)

    for _ in range(6):
        disk_restore.worker_proof(force=True)

    lines = [line for line in proof_gate_lines(caplog) if "GATE CLOSED" in line]
    assert len(lines) == 1, f"{len(lines)} closure lines for one closure: {lines}"
    line = lines[0]
    assert "reason_kind=no_worker_proof" in line, line
    assert "stale=1" in line, line
    assert "youngest=" in line and "window=900.0s" in line, line
    assert "last_proof_written=2" in line, line
    assert disk_restore.counters()["proof_closures"] == 1


def test_a_gate_reopening_logs_exactly_one_line(booted, caplog):
    """And one line when it comes back, saying how long it was shut."""
    clock = booted["clock"]
    assert wait_for(lambda: rank_document(booted["dir"]) is not None)
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1)
    worker_diag.stop_heartbeat()
    clock.advance(disk_restore.RANK_FILE_FRESH_SECONDS + 60.0)
    assert disk_restore.worker_proof(force=True)["count"] == 0
    caplog.clear()

    # The spell ends: the reader's clock returns to the real one and the worker's beats
    # are younger than the window again.
    clock.advance(-(disk_restore.RANK_FILE_FRESH_SECONDS + 60.0))
    worker_diag.start_heartbeat()
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["ok"] is True), (
        "the beat never resumed, so there is no reopening to log")
    for _ in range(4):
        disk_restore.worker_proof(force=True)

    lines = [line for line in proof_gate_lines(caplog) if "GATE OPEN" in line]
    assert len(lines) == 1, f"{len(lines)} reopening lines for one reopening: {lines}"
    assert "after" in lines[0] and "closed" in lines[0], lines[0]
    assert disk_restore.counters()["proof_openings"] == 1


def test_the_closure_line_distinguishes_a_stale_file_from_a_sectionless_one(
        booted, caplog):
    """THE TWO OUTAGES THAT USED TO BE ONE COUNTER.

    A fresh rank file with no `disk_restore` section is as useless to the gate as a
    stale one, and means something completely different: a contributor that raised or
    overran its byte cap, or a worker whose restore half never fitted. Those reasons
    live in the worker's `_contrib_state`, in a process with no HTTP route - so from the
    scheduler's side the two conditions were the same number. They are not, and the line
    now says which one happened.
    """
    directory = booted["dir"]
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1)
    caplog.clear()

    worker_diag.stop_heartbeat()

    def strip_the_section(document):
        document.pop("disk_restore", None)

    rewrite_rank_file(directory, strip_the_section)
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["count"] == 0, verdict
    assert verdict["reason_kind"] == "no_worker_proof", verdict
    assert verdict["sectionless_rank_files"] == 1, verdict
    assert verdict["stale_rank_files"] == 0, verdict

    lines = [line for line in proof_gate_lines(caplog) if "GATE CLOSED" in line]
    assert len(lines) == 1, lines
    assert "sectionless=1" in lines[0] and "stale=0" in lines[0], lines[0]


def test_the_gate_log_does_not_flood_while_the_state_holds(booted, caplog):
    """The transitions are edges, not a level: a poller every second costs nothing.

    `worker_proof()` is cached for a second and the dashboard polls far more often than
    that. This is the `asleep_notice_once` / `_settle_refusal_logged` rule: one line per
    episode, or the diagnostic stops being readable.
    """
    clock = booted["clock"]
    assert wait_for(lambda: rank_document(booted["dir"]) is not None)
    assert wait_for(lambda: disk_restore.worker_proof(force=True)["count"] >= 1)
    caplog.clear()
    worker_diag.stop_heartbeat()
    clock.advance(disk_restore.RANK_FILE_FRESH_SECONDS + 60.0)
    for _ in range(50):
        disk_restore.worker_proof(force=True)
        time.sleep(0.01)
    lines = proof_gate_lines(caplog)
    assert len([line for line in lines if "GATE CLOSED" in line]) == 1, lines
    assert len(lines) <= 2, lines


# ===========================================================================
# 6. THE ARITHMETIC THE WHOLE DESIGN RESTS ON, PINNED AGAINST THE SHIPPED VALUE
# ===========================================================================

def test_the_shipped_cadence_still_sits_inside_the_window_it_must_stay_inside():
    """The fixture shortens the interval to make a thread observable; this pins the
    number that actually ships, against the UNPATCHED constants.

    60 s inside a 900 s window is fifteen beats, so four consecutive lost ticks are
    still safe. A supervisor that ISSUES beats instead of running them cannot make the
    cadence tighter, only more reliable - which is why this is still the equation to
    check before believing any of the above.
    """
    interval = worker_diag.HEARTBEAT_INTERVAL_DEFAULT
    window = disk_restore.RANK_FILE_FRESH_SECONDS
    assert window == 900.0
    assert interval == 60.0
    assert interval * (1 + worker_diag.HEARTBEAT_MIN_BEATS_IN_WINDOW) < window
    budget = worker_diag.HEARTBEAT_BEAT_BUDGET_SECONDS
    assert budget < window
    assert worker_diag.HEARTBEAT_MAX_INFLIGHT_BEATS >= 1
    counters = worker_diag.counters()
    for key in ("heartbeat_writes", "heartbeat_inflight", "heartbeat_stalled",
                "heartbeat_jammed", "heartbeat_fatal"):
        assert key in counters, key


def test_the_beat_gate_reopens_when_the_supervisor_is_rearmed(booted):
    """A respawned supervisor must ACTUALLY beat.

    `stop_heartbeat()` closes the gate that `start_heartbeat()` opens. If a respawn
    re-armed the loop without reopening that gate, the worker would tick forever, issue
    nothing, and report `alive` - a silent re-arming, which is the failure this whole
    file exists to remove.
    """
    directory = booted["dir"]
    assert wait_for(lambda: rank_document(directory) is not None)
    worker_diag.stop_heartbeat()
    before = int(worker_diag.counters()["heartbeat_writes"])
    started = worker_diag.start_heartbeat()
    assert started["started"] is True, started
    assert wait_for(lambda: int(worker_diag.counters()["heartbeat_writes"])
                    > before), (
        "the supervisor came back and issued nothing: a re-arm that cannot beat is "
        "indistinguishable from no re-arm")
