r"""A COLD BOOT MUST PROVE ITSELF BEFORE THE FIRST REQUEST, NOT ONE INTERVAL LATER.

THE MEASUREMENT THIS FILE EXISTS FOR. Three boots that had been suspended for longer
than `disk_restore.RANK_FILE_FRESH_SECONDS` (900 s) came up with the scheduler-side
restore gate CLOSED, and the first request behind each got `refused=no_worker_proof`
and paid a **61 s cold prefill on 231,392 tokens**. The lag from "the box is up" to
"the gate proves restorable" measured 14.2 s, 17.6 s and 19.9 s. About 35 of ~38 boots
hide this completely, because a previous boot's rank file is still under 900 s old -
which is exactly how a defect survives in a system whose usual restart is shorter than
its own freshness window.

THE CAUSE, IN ONE LINE. The only writers of a rank proof are `worker_diag`'s step path
and its heartbeat, and on a cold boot neither had produced anything by the time the
first request arrived: `install()` starts the supervisor and that loop's first beat was
scheduled a FULL INTERVAL later, while `install()` itself writes nothing on purpose.
For the first minute of a boot the worker is alive, fitted and armed, and believed by
the gate to be nothing.

WHAT A NAIVE EARLY WRITE WOULD HAVE LIED ABOUT - and what these tests refuse to accept.
Two false statements were available to the obvious fix, and the fix is built to avoid
both, so the assertions below are the load-bearing part of the change:

  1. `install()` runs FIRST inside the worker handshake loop, before
     `disk_restore.install()` raises `worker_fitted` and registers its contribution. A
     proof written there carries no `disk_restore` section, the reader verdicts it
     `no_restore_section`, and the gate answers `no_worker_proof` anyway. An early
     write in the wrong place is a proof that proves nothing - so the prime runs AFTER
     the loop.
  2. The handler is normally captured from `get_finished`, so a beat fired before any
     step would publish `arena.bytes_total: 0`. That is not an omission, it is a FALSE
     NUMBER in the one field a capacity reader divides by - so the handler is noted
     from `connector.worker_handler` (real by handshake time: `register_kv_caches`
     fills `num_cpu_blocks` before the handshake is collected) BEFORE the installs
     start, and the assertions here demand the arena's actual size, not its presence.

WHAT THE FIX IS, AND WHAT IT IS NOT. (a) `engine_side._prime_worker_proof` fires one
contained, floor-exempt `heartbeat_once()` after all four worker installs have run, in
the WORKER process - the EngineCore's gate can only read a proof another process wrote,
so a proof from the scheduler would be exactly the attestation the design refuses to
trust. (b) The supervisor's first tick is due immediately, the belt behind that braces.
Neither one weakens the gate: the 900 s window, the freshness rule and the rule that a
worker which is genuinely gone stops being believed are all asserted CLOSED by
`tests/test_idle_worker_proof_stays_alive_and_legible.py`, and nothing here invents a
second document - the prime goes through the same `write_snapshot` a step uses.
"""

from __future__ import annotations

import os
import threading
import time

import fakes_debug  # noqa: F401 - the shared fake-engine idiom
import pytest
import test_idle_worker_proof_stays_alive_and_legible as idle

from kvtier_dev_plugin import (
    config,
    cpu_capacity,
    disk_layout,
    disk_restore,
    disk_spill,
    engine_side,
    snapshot,
    worker_diag,
)

# THE FORK, THE CLOCK AND THE BOOT ARE THE ONES `test_idle_worker_proof_stays_alive_
# and_legible.py` already owns. Duplicating them here would put two worker fakes in the
# suite and let one of them drift from what `disk_restore.install()` accepts; importing
# them keeps this file about ONE new fact - the moment the first proof appears. The two
# fixtures are re-exported under their own names because pytest injects by NAME.
fork = idle.fork  # noqa: F811 - re-exported as a fixture, not shadowed
clock = idle.clock  # noqa: F811 - re-exported as a fixture, not shadowed

# The fork's arena, from `idle.FORK_WORKER_SOURCE`: `{"g0": _Cache(4096)}`. Demanded
# EXACTLY, because "nonzero" would let a placeholder number through and the whole point
# of the prime is that it measures instead of guessing.
ARENA_BYTES = 4096


@pytest.fixture
def boot_env(tmp_path, monkeypatch, fork, clock):
    """A worker's environment, tier armed, on the SHIPPED heartbeat cadence.

    THE CADENCE IS THE FIXTURE'S WHOLE POINT. `test_idle_worker_proof_stays_alive_
    and_legible.py` shortens `HEARTBEAT_INTERVAL_SECONDS` to 0.2 s so it can watch a
    thread tick; here the interval stays at the shipped 60 s, because a shortened one
    would let the supervisor's own beat satisfy every assertion below and the test
    would be about a thread instead of about the prime. Nothing else about the boot is
    faked: the starter is `engine_side.register()`, the installer is the handshake
    hook, and the file is the production file.

    The boot itself is NOT performed here. Each test patches what it needs first and
    then calls `idle.boot_worker`, so a test can arrange a seam that fails DURING the
    boot rather than around it.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    root = tmp_path / "tier"
    root.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, idle.RANK)
    monkeypatch.delenv(config.ENV_DIAG_EPOCH, raising=False)
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(8 * 1024 * 1024 * 1024))
    monkeypatch.delenv(config.ENV_DISK_BOOT_ENABLE, raising=False)
    worker_diag.reset_state_for_tests()
    disk_restore.reset_state_for_tests()
    disk_spill.reset_state_for_tests()

    armed = disk_spill.arm()
    assert armed["armed"] is True, armed
    try:
        yield {"dir": directory, "fork": fork, "clock": clock}
    finally:
        worker_diag.stop_heartbeat()
        worker_diag.reset_state_for_tests()
        disk_restore.reset_state_for_tests()
        disk_spill.reset_state_for_tests()


# ===========================================================================
# 1. THE OUTAGE: THE GATE MUST BE OPEN BEFORE THE FIRST REQUEST, NOT AFTER IT
# ===========================================================================

def test_a_boot_with_zero_steps_proves_restorable_the_moment_it_fits(boot_env):
    """THE 61-SECOND PREFILL, AS A TEST. No step, no request, no wait: the handshake
    alone must leave a proof the scheduler can act on.

    AGAINST THE OLD CODE this is the failure the operator measured: the worker boots,
    every install says YES, and `worker_proof` answers `reason_kind="no_worker_proof"`
    because the only thing that could have written a rank file was a beat due 60 s
    later, or a step an idle box never takes.
    """
    directory = boot_env["dir"]
    handler, _connector = idle.boot_worker(directory, boot_env["fork"]["load"])

    assert handler.step_calls == 0, (
        "this test is about a boot that took no step; a step would write the rank "
        "file through the step path and the assertion below would prove nothing")
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True, (
        f"a fitted, armed worker with zero steps did not prove restorable at the "
        f"handshake: {verdict['reason_kind']} / {verdict['reason']} - this is the "
        "cold-boot gate that cost 61 s of cold prefill")
    assert verdict["count"] >= 1, verdict
    assert verdict["stale_rank_files"] == 0, verdict

    # ONE file, from this boot, and nothing else in the directory pretending to be it.
    names = idle.rank_files(directory)
    assert names == [f"rank-{idle.RANK}.json"], names
    document = idle.rank_document(directory)
    assert document is not None
    assert time.time() - float(document["ts"]) < 5.0, (
        "the proof the gate accepted is not this boot's", document["ts"])


def test_the_primed_proof_measures_the_arena_and_carries_the_restore_section(boot_env):
    """THE ANTI-PLACEHOLDER TEST: the first proof must be a MEASUREMENT, not a shape.

    Two lies are possible for a proof written at boot, and this asserts against both,
    because each one alone would have shipped the outage wearing a different message:

      * NO `disk_restore` SECTION. A proof written before `disk_restore.install()` has
        registered its contribution is fresh and useless - the reader verdicts it
        `no_restore_section` and the gate still says `no_worker_proof`. This is why the
        prime runs after the install loop, and a reordering that moves it earlier fails
        HERE rather than in production.
      * `arena.bytes_total: 0`. A beat whose handler was never noted writes a zero,
        which is a false statement about the arena rather than a missing one. Demanding
        the fork's actual 4096 - not merely "nonzero" - is what stops the fix from
        shipping the placeholder it exists to prevent.
    """
    directory = boot_env["dir"]
    handler, _connector = idle.boot_worker(directory, boot_env["fork"]["load"])
    assert handler.step_calls == 0

    document = idle.rank_document(directory)
    assert document is not None, "the handshake produced no rank file at all"

    section = document.get("disk_restore")
    assert isinstance(section, dict), (
        f"a primed proof with no `disk_restore` section is verdicted "
        f"`no_restore_section` and still reads as no_worker_proof: {document}")
    assert section["installed"] is True, section
    assert section["worker_fitted"] is True, section

    arena = document.get("arena")
    assert isinstance(arena, dict), document
    assert arena.get("bytes_total") == ARENA_BYTES, (
        f"the primed proof published {arena.get('bytes_total')!r} for an arena of "
        f"{ARENA_BYTES} bytes: a beat with no handler yet writes a zero, and a zero "
        "here is a false number, not an empty field", arena)
    assert arena.get("bytes_per_block"), (
        "the width every spill file has is measured off this rank's own tensors; its "
        "absence says nothing was measured", arena)

    # THE PRIME IS A BEAT, SO IT CARRIES THE BEAT'S OWN STATE: a reader must be able to
    # tell "the writer is alive and has written" from "a file exists".
    block = document.get("writer", {}).get("heartbeat")
    assert isinstance(block, dict), document.get("writer")
    assert block["state"] == "beating", block
    assert block["beat_writes"] >= 1, block


# ===========================================================================
# 2. BEST-EFFORT MEANS BEST-EFFORT: A FAILED PRIME COSTS NOTHING BUT THE PRIME
# ===========================================================================

def test_a_prime_that_raises_never_breaks_the_worker_and_later_beats_still_run(
        boot_env, monkeypatch, caplog):
    """The prime is an OPTIMISATION of the boot, never a dependency of it.

    THE HANDSHAKE MUST SURVIVE IT. `_prime_worker_proof` runs inside the worker's
    handshake hook; a prime that raised into that hook would cost the worker its
    offload tier (and the engine its worker), which is a worse outage than the one
    being fixed. So the prime is contained, the failure is counted and logged, and the
    boot proceeds.

    AND THE BACKSTOP MUST STILL WORK. The loop's own first tick is due immediately, so
    a boot whose prime failed still produces a rank file a moment later - the belt
    behind the braces. This asserts the file ARRIVES after the failure, which is the
    difference between "contained" and "silently gave up".
    """
    directory = boot_env["dir"]
    real_beat = worker_diag.heartbeat_once
    calls = {"n": 0}

    def exploding_beat(force: bool = False, primed: bool = False) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the prime exploded")
        return real_beat(force=force, primed=primed)

    monkeypatch.setattr(worker_diag, "heartbeat_once", exploding_beat)
    errors_before = int(snapshot._state["errors"])
    caplog.clear()

    handler, _connector = idle.boot_worker(directory, boot_env["fork"]["load"])
    assert handler.step_calls == 0
    assert calls["n"] >= 1, (
        "the prime never reached `heartbeat_once`, so this test asserted about a boot "
        "that had no prime to fail")

    # ONE failure, COUNTED AND NAMED, never re-raised into the handshake.
    assert int(snapshot._state["errors"]) == errors_before + 1, (
        "a prime that raised must cost exactly one counted containment")
    assert [line for line in (r.getMessage() for r in caplog.records)
            if "worker proof prime" in line], (
        "the failure was swallowed: a prime that failed has to say so, because the "
        "missing rank file is the only other symptom and it names nothing")

    # THE BOOT IS OTHERWISE INTACT: installed, fitted, heartbeat running.
    counters = worker_diag.counters()
    assert counters["installed"] is True, counters
    assert worker_diag.heartbeat_status()["alive"] is True, (
        "a failed prime must not have cost the supervisor its thread")

    # AND THE LOOP'S LATER BEATS WRITE ANYWAY - the backstop, as an artifact on disk.
    assert idle.wait_for(lambda: idle.rank_document(directory) is not None), (
        "the prime failed and nothing ever wrote the rank file: the immediate first "
        "tick is the belt behind the prime, and it did not hold")
    assert idle.wait_for(lambda: int(worker_diag.counters()["heartbeat_writes"])
                         >= 1)
    document = idle.rank_document(directory)
    assert isinstance(document.get("disk_restore"), dict), document
    assert document["arena"]["bytes_total"] == ARENA_BYTES, document["arena"]


def test_a_boot_with_no_diagnostics_directory_still_primes_nothing(
        boot_env, monkeypatch, caplog):
    """AN INERT TIER CREATES NO FILES AND NO THREADS - the prime changes none of that.

    The prime calls `heartbeat_once` once the writer is installed, so the question this
    pins is what happens when `VLLM_E2_TIER_DIAG` is unset: the writer never installs,
    so the prime must DECLINE before the beat, create nothing, start no thread and log
    no prime line. A "best-effort" write that materialised a diagnostics directory on
    a boot the operator declared inert would be a new side effect on every engine in
    the fleet.

    BOOTED BY HAND, not by `idle.boot_worker`: that helper asserts the plugin-load
    refusal names the WORKER MODULE, and with no diagnostics directory the refusal
    names the directory instead - a true statement about a different refusal, and the
    reason to drive the three production calls here directly.
    """
    directory = boot_env["dir"]
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=True)
    caplog.clear()

    engine_side.register()
    loaded = boot_env["fork"]["load"]()
    handler = loaded["worker"]()
    connector = loaded["connector"]()
    connector.worker_handler = handler
    cpu_capacity._call_worker_report_hook(connector)   # must not raise

    # No RANK FILE. The directory may legitimately hold the spill tier's own arm file,
    # written by the fixture's `disk_spill.arm()`; what the prime must not add is a
    # `rank-*.json`, and that is the artifact a reader (and the gate) acts on.
    assert idle.rank_files(directory) == [], os.listdir(str(directory))
    counters = worker_diag.counters()
    assert counters["installed"] is False, counters
    assert int(counters["heartbeat_writes"]) == 0, counters
    assert worker_diag.heartbeat_status()["alive"] is False, (
        "no diagnostics directory means no supervisor at all")
    assert config.ENV_DIAG_DIR in str(counters["skip_reason"]), counters
    assert [line for line in (r.getMessage() for r in caplog.records)
            if "worker proof primed" in line] == [], (
        "an inert boot logged a prime it did not perform")


# ===========================================================================
# 3. THE ONE CLAIM THE OPERATOR WILL CHECK ON THE BOX
# ===========================================================================

def test_a_boot_that_never_sees_a_request_ends_up_with_a_gate_open_proof(boot_env):
    """ZERO REQUESTS, ZERO STEPS, AND THE PROOF IS STILL ON DISK WHEN THE BOOT ENDS.

    THE PROPERTY, STATED THE WAY IT WILL BE VERIFIED: boot a worker, ask it nothing,
    and the gate's own answer must be `open` with this boot's file as the evidence. The
    request path is never touched here - no `get_finished`, no bind, no offer - so what
    this asserts is that the proof does not depend on traffic existing to justify
    itself.
    """
    directory = boot_env["dir"]
    handler, _connector = idle.boot_worker(directory, boot_env["fork"]["load"])

    # Nothing arrives. No step, no request, no forced read.
    time.sleep(0.05)
    assert handler.step_calls == 0
    counters = worker_diag.counters()
    assert int(counters["steps"]) == 0, counters

    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True, verdict
    assert not verdict.get("reason_kind"), verdict
    assert verdict["ranks"], verdict
    assert verdict["fresh_rank_files"] >= 1, verdict
    assert verdict["sectionless_rank_files"] == 0, (
        "a primed file that carries no `disk_restore` section is counted here, and "
        "the gate would be shut on the next request even though a file exists",
        verdict)


# ===========================================================================
# 4. TWO WRITERS AT BOOT IS THE DESIGN - THE TEMP NAME MUST NOT LET THEM COLLIDE
# ===========================================================================

def test_two_beats_writing_at_once_lose_no_proof(boot_env):
    """THE RANK FILE HAS ONE NAME; ITS TEMP FILE MUST NOT.

    WHY THIS BELONGS TO 0.2.69 AND NOT TO THE PAST. The handshake prime and the
    supervisor's immediately-due first tick write the SAME rank file at the SAME moment -
    that race IS the release, and the prime wins it only because the write floor cannot
    refuse it. But the temp half of the atomic replace used to be named
    `.<rank>.<pid>.<writes>.tmp`, where `writes` counts COMPLETED writes: two beats in
    flight read the same value, write the same path, and the loser's `os.replace` raises
    FileNotFoundError because the winner renamed the file out from under it. Observed on
    this tree as a cold-boot proof that did not land, with the rank file's `state_reason`
    reading `FileNotFoundError: ... .rank-PP0_TP0.json.<pid>.0.tmp` - the outage this file
    exists to end, arriving through a door the fix opened. So: eight concurrent writes, and
    not one of them may be lost, and the directory must hold one rank file and no litter.
    """
    directory = boot_env["dir"]
    handler, _connector = idle.boot_worker(directory, boot_env["fork"]["load"])
    worker_diag.stop_heartbeat()
    assert idle.wait_for(
        lambda: int(worker_diag.counters()["heartbeat_inflight"]) == 0)
    errors_before = int(worker_diag.counters()["write_errors"])
    reports: list = []

    def write_one() -> None:
        reports.append(worker_diag.write_snapshot(handler, idle.RANK, "env"))

    threads = [threading.Thread(target=write_one) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    assert not [t for t in threads if t.is_alive()], "a write thread is wedged"

    counters = worker_diag.counters()
    assert len(reports) == 8, reports
    lost = [report for report in reports if not report.get("written")]
    assert not lost, (
        f"{len(lost)} of 8 concurrent writes were lost: {lost[:2]} | write_errors="
        f"{counters['write_errors']} (was {errors_before}) - two writers of one rank file "
        "that share a temp name silently drop the cold-boot proof")
    assert int(counters["write_errors"]) == errors_before, counters
    assert idle.rank_files(directory) == [f"rank-{idle.RANK}.json"], (
        os.listdir(str(directory)))
    assert [name for name in os.listdir(str(directory))
            if name.endswith(".tmp")] == [], os.listdir(str(directory))
    document = idle.rank_document(directory)
    assert document is not None and document["rank"] == idle.RANK, document
