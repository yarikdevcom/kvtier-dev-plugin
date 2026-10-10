r"""THE SPILL FLUSH: a bounded wait for the asynchronous backlog.

WHY THIS EXISTS AT ALL. Mirror mode writes asynchronously, so a stop that happens the
instant the queue still holds work loses exactly the data the change exists to
protect. The backlog is normally a few hundred milliseconds of NVMe; the failure is
the one where it is not (a ceiling pass in the way, a writer that died, a queue
carrying 256 jobs at 11.4 MB each).

TWO DOORS ON ONE MECHANISM, and they are not interchangeable:

  * `POST /plugins/kvtier/disk {"op":"flush","timeout_seconds":20}` - the PRIMITIVE.
    Useful by hand, testable, and the door a proxy can call when it wants to stop the
    engine politely. It answers for the process that runs it.
  * THE SHUTDOWN DRAIN - the automatic one, installed at connector creation on the
    connector object itself, so it runs in the process that OWNS the queue. This is
    the primary mechanism because it covers every stop path (a human pressing Stop, a
    restart, the proxy, a reboot) rather than the one path the proxy controls.

WHY THE DRAIN IS NOT AN `atexit` HANDLER, which is the finding that decided the
mechanism, read out of the live fork rather than inferred:

  * a worker ends its orderly shutdown in `multiproc_executor.py:_exit_worker_process`
    with `os._exit(exit_code)` - and `os._exit` skips `atexit`, every pending `finally`
    and interpreter finalization. A drain registered with `atexit` in a WORKER process
    would therefore never run: it is dead code on the stop path, silently.
  * the worker's teardown DOES call the connector: `Worker.shutdown()`
    (`v1/worker/gpu_worker.py:1699`) -> `ensure_kv_transfer_shutdown()`
    (`kv_transfer_state.py:97`) -> `_KV_CONNECTOR_AGENT.shutdown()`, and that runs
    BEFORE the `os._exit`. The spill queue lives in the worker, so the connector's own
    shutdown is the seam that exists in the right process at the right moment.

So the drain wraps the connector's `shutdown` (instance-level, idempotent, chained:
the fork's method is called after the drain and its own exceptions still propagate)
and `atexit` stays registered as the SECONDARY seam for the processes that really do
exit normally. `atexit` is not the mechanism, and pretending it was would be the
silent no-op at the worst possible moment.

WHAT THE VERB MUST NEVER DO: block the serving path, arm or disarm anything, move a
posture, or raise. A flush that fails returns a `reason` and the caller's stop or boot
proceeds anyway. The AST pins below hold that line structurally, and the budget pins
hold the "cheap when there is nothing to do" rule the suspend path depends on.
"""

from __future__ import annotations

import ast
import inspect
import logging
import time

import pytest
import test_disk_store as store_fixtures  # the module the fixtures live in
from test_disk_spill import SLOT_BYTES
from test_disk_store import handler, metadata, metadata_shape

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    backport,
    config,
    disk_restore,
    disk_spill,
    disk_store,
)

none_hash_seed = store_fixtures.none_hash_seed  # noqa: F811 - autouse: the fingerprint needs the fork's seed module
no_fake_distributed_state = store_fixtures.no_fake_distributed_state  # noqa: F811 - autouse
tier = store_fixtures.tier  # noqa: F811 - the configured, ARMED store under tmp_path
unpatched_engine = store_fixtures.unpatched_engine  # noqa: F811 - autouse

KEY_A = "a3f0b7c1d4e5f607" + "1" * 48 + "00000003"
KEY_B = "0f1e2d3c4b5a6978" + "2" * 48 + "00000003"

CONTRACT_FIELDS = ("ok", "drained", "queue_depth_before", "queue_depth_after",
                   "bytes_written", "elapsed_seconds", "timed_out", "reason")


def flush(budget=None, **extra) -> dict:
    """The verb, called the way the route calls it (the body IS the args dict)."""
    body = {"op": "flush"}
    if budget is not None:
        body["timeout_seconds"] = budget
    body.update(extra)
    return disk_store.disk_for_engine(None, body)


def fill_queue(jobs: int) -> None:
    """Real jobs, in the real queue, with the writer NOT running.

    The writer is what would drain them, so every timeout test has to keep it down.
    `MAX_QUEUE_DEPTH` is raised rather than the depth lowered so a filler job and a
    real job are the same object shape - a fake whose shape makes the assertion
    unachievable is the trap `tests/test_in_tree_standdown.py:46-49` warns about.
    """
    pending = disk_spill._get_queue()
    for index in range(jobs):
        pending.put_nowait({"blob": b"\0" * SLOT_BYTES, "slot": index,
                            "key": KEY_A, "bytes": SLOT_BYTES})


# --- 1. the contract, field by field ------------------------------------------

def test_the_flush_op_reports_the_contract_fields_flat(tier, monkeypatch):
    """Every contracted field is present, FLAT on the returned object, and true.

    `bytes_written` is what THIS flush saw land, not the boot's total: a caller that
    wants to know whether its own backlog arrived must be able to read that off one
    answer. The jobs here are real (the same shape `request_spill` puts on the queue),
    so the drain is the real writer writing real files.

    THE SLOWED WRITER IS FOR DETERMINISM, NOT FOR SPEED. An unblocked writer can finish
    both files before the flush is even called, and then `queue_depth_before` is 0 and
    the test asserts nothing about a wait - a fixture whose timing makes the assertion
    unachievable is the trap `tests/test_in_tree_standdown.py:46-49` warns about. The
    delay goes AROUND the real `_handle_job`, so the bytes, the files and the counters
    are all the production path; only the schedule changes. And the depth is read as
    `unfinished_tasks`, which counts the job the writer is INSIDE, so the value below is
    2 whether or not the writer had started - the assertion does not depend on winning a
    race, which is the whole reason that currency was chosen over `qsize()`.
    """
    real_handle = disk_spill._handle_job

    def slow_handle(job):
        time.sleep(0.15)
        real_handle(job)

    monkeypatch.setattr(disk_spill, "_handle_job", slow_handle)
    role = handler()
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[6, 7], store_keys=[KEY_A, KEY_B],
                        store_event=5)
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        disk_store.on_store_completed(role, [])
    answer = flush(10.0)
    assert answer["ok"] is True, answer
    for field in CONTRACT_FIELDS:
        assert field in answer, f"{field} is missing from the flush answer"
        assert not isinstance(answer[field], dict), "the fields are FLAT"
    assert answer["drained"] is True
    assert answer["timed_out"] is False
    assert answer["reason"] is None
    assert answer["queue_depth_before"] == 2
    assert answer["queue_depth_after"] == 0
    assert answer["bytes_written"] == 2 * SLOT_BYTES
    assert answer["elapsed_seconds"] >= 0.0
    assert answer["op"] == "flush"
    # The bytes are on disk, which is the only claim `drained` is making.
    import os
    written = [paths for _base, _dirs, names in os.walk(str(tier)) for paths in names]
    assert len(written) == 2


def test_an_empty_queue_returns_in_milliseconds(tier):
    """The suspend path calls this on EVERY sleep, and the common case is nothing to do.

    A flush that slept first and looked afterwards would cost every suspend 100 ms+
    and would teach the operator to stop calling it. The bound here is a pin on the
    ORDER of operations: look, then wait only if the look found something.
    """
    started = time.monotonic()
    answer = flush(20.0)
    elapsed = time.monotonic() - started
    assert answer["drained"] is True
    assert answer["reason"] is None
    assert answer["queue_depth_before"] == 0
    assert elapsed < 0.05, (f"an empty flush took {elapsed:.3f}s: it is not looking "
                        "first")


# --- 2. the named reasons -----------------------------------------------------

def test_a_backlog_that_cannot_drain_times_out_and_names_the_reason(tier, monkeypatch):
    """A writer that is alive and cannot finish inside the budget: `timeout`, honestly.

    `ok` stays True - the CALL was accepted, the operator's contract - while `drained`
    is False and `reason` is the named cause. A proxy that conflated the two would
    suspend on a store that still holds the conversation.

    WHY A REAL WRITER, SLOWED, RATHER THAN NO WRITER AT ALL: a queue with no live writer
    is a DIFFERENT named reason (`writer_down`, asserted below) and is knowable in one
    look, so the flush must not burn its budget on it. The `timeout` case is the
    slow-disk
    one - a thread that exists, is alive, and will not finish in time, which is what a
    wedged NVMe path looks like from here. The delay wraps the real `_handle_job`, so
    the
    jobs and the counters are the production path and only the schedule is staged.
    """
    real_handle = disk_spill._handle_job

    def slow_handle(job):
        time.sleep(1.0)          # longer than the budget below, shorter than the suite
        real_handle(job)

    monkeypatch.setattr(disk_spill, "_handle_job", slow_handle)
    role = handler()
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[6, 7], store_keys=[KEY_A, KEY_B],
                        store_event=5)
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        disk_store.on_store_completed(role, [])
    started = time.monotonic()
    answer = flush(0.3)
    elapsed = time.monotonic() - started
    assert answer["ok"] is True
    assert answer["drained"] is False
    assert answer["timed_out"] is True
    assert answer["reason"] == "timeout"
    assert answer["queue_depth_before"] == 2
    assert answer["queue_depth_after"] == 2, "a 1s job cannot have finished in 0.3s"
    assert answer["bytes_written"] == 0
    assert elapsed < 1.5, f"a 0.3s budget took {elapsed:.3f}s"


def test_a_writer_that_is_not_running_is_reported_as_writer_down(tier, monkeypatch):
    """A dead writer is KNOWABLE in one look, so the flush must not burn its budget.

    The difference matters on the suspend path: `writer_down` in 2 ms tells the proxy
    the backlog cannot be saved and it may proceed, while `timeout` after 20 s tells
    it the same thing late. Jobs in a queue with no live writer are never going to
    land, so waiting for them is a hang with extra steps.
    """
    monkeypatch.setattr(disk_spill, "MAX_QUEUE_DEPTH", 64)
    fill_queue(2)
    assert disk_spill.counters()["thread_alive"] is False
    started = time.monotonic()
    answer = flush(20.0)
    elapsed = time.monotonic() - started
    assert answer["drained"] is False
    assert answer["reason"] == "writer_down"
    assert answer["timed_out"] is False
    assert answer["queue_depth_after"] == 2
    assert elapsed < 1.0, f"the writer-down case waited {elapsed:.3f}s"


def test_a_disabled_flush_says_disabled_and_touches_nothing(tier, monkeypatch):
    """Budget 0 is the operator's off switch, and it is a NAMED reason, not silence."""
    monkeypatch.setenv(config.ENV_DISK_SHUTDOWN_FLUSH_SECONDS, "0")
    monkeypatch.setattr(disk_spill, "MAX_QUEUE_DEPTH", 64)
    fill_queue(2)
    answer = flush()
    assert answer["drained"] is False
    assert answer["reason"] == "disabled"
    assert answer["queue_depth_after"] == 2


# --- 3. the argument is validated ---------------------------------------------

@pytest.mark.parametrize("value", ["soon", None, [5], True, {}])
def test_a_non_numeric_timeout_is_rejected_without_touching_the_queue(
        tier, monkeypatch, value):
    """A malformed budget is refused BEFORE the queue is touched, and says why.

    The route's argument spec rejects this shape too (`{"kind": "int"}`); the verb
    still has to survive it, because the engine method is reachable without the route
    and a silent fallback to "wait 20 seconds" would be a wait nobody asked for.
    """
    monkeypatch.setattr(disk_spill, "MAX_QUEUE_DEPTH", 64)
    fill_queue(2)
    answer = disk_store.disk_for_engine(None, {"op": "flush",
                                               "timeout_seconds": value})
    assert answer["ok"] is False
    assert "timeout_seconds" in answer["reason"]
    assert answer["drained"] is False
    assert answer["queue_depth_before"] == 2
    assert disk_spill.counters()["flush_calls"] == 0


def test_an_absurd_budget_is_clamped_and_says_so(tier):
    """The clamp is a bound on a WAIT a caller asked for, and it is reported.

    A caller that asked for an hour while the box waits on its suspend must not get an
    hour, and must be able to see that it did not get what it asked for.
    """
    answer = flush(disk_spill.FLUSH_BUDGET_SECONDS_MAX * 10)
    assert answer["ok"] is True
    assert answer["budget_seconds"] == disk_spill.FLUSH_BUDGET_SECONDS_MAX
    assert answer["clamped"] is True


# --- 4. the counters live beside the other spill tallies ----------------------

def test_the_flush_counters_live_beside_the_other_spill_tallies(tier):
    """Four counters, in the section that already carries the spill tallies.

    Chosen deliberately: `spill` in `/disk op:status` and in `snapshot.disk_store` is
    the process's OWN spill block (the same block that already says so in its scope
    note), and a flush is a spill-side act of that same process. The rank rollup is
    the WRONG home for these: it sums four workers' files into a fleet total, and a
    flush's elapsed/budget accounting cannot be summed meaningfully - `queue_depth`
    is a gauge for exactly that reason and is labelled as one.
    """
    flush(5.0)
    flush(5.0)
    status = disk_store.disk_for_engine(None, {"op": "status"})
    spill = status["spill"]
    for key in ("flush_calls", "flush_drained", "flush_timed_out",
                "flush_bytes_written"):
        assert key in spill, f"{key} is missing from the verb's spill block"
        assert key in disk_store.section()["spill"], f"{key} missing from /snapshot"
    assert spill["flush_calls"] == 2
    assert spill["flush_drained"] == 2
    assert spill["flush_timed_out"] == 0
    counters = disk_spill.counters()
    assert counters["flush_bytes_written"] == 0, "nothing was queued to write"


# --- 5. the verb is wired into the route, in order ----------------------------

def test_the_verb_advertises_flush_and_the_route_carries_its_argument():
    """`flush` is in the op tuple, the dispatch table, and the route's argument spec.

    The tuple order is load-bearing (an existing test pins the advertised order
    against the dispatch order), and `flush` is NOT in `MUTATING_DISK_OPS`: it waits,
    it does not change state, and a 405 on a GET would be a surprise to the operator
    who only wanted to look.
    """
    assert "flush" in disk_store.DISK_OPS
    assert "flush" in disk_store._DISK_OPS
    assert "flush" not in disk_store.MUTATING_DISK_OPS
    assert list(disk_store._DISK_OPS) == list(disk_store.DISK_OPS)
    from kvtier_dev_plugin import endpoint
    spec = endpoint.ARG_SPECS[config.ROUTE_DISK]
    assert set(spec["op"]["values"]) == set(disk_store.DISK_OPS)
    assert "timeout_seconds" in spec


# --- 6. the shutdown drain: the connector seam --------------------------------

class FakeConnector:
    """The worker-role connector, and a `shutdown` that RECORDS the queue it saw.

    The record is the point: the drain has to run BEFORE the fork's own teardown work,
    because what follows it in the fork is `destroy_model_parallel()` and then
    `os._exit`. A wrapper that ran after would drain nothing.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.depth_at_shutdown = None
        self.role = "worker"

    def shutdown(self) -> None:
        self.calls += 1
        pending = disk_spill._queue
        self.depth_at_shutdown = 0 if pending is None else pending.qsize()


def test_the_shutdown_drain_runs_before_the_connectors_own_shutdown(tier, monkeypatch):
    """The drain empties the queue, then the fork's `shutdown` runs and sees it empty.

    Two assertions, one per half of the contract: the data got to disk, AND the fork's
    method was still called exactly once. A wrapper that drained but swallowed the
    original would pass half of this and break the worker's teardown.
    """
    role = handler()
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[6, 7], store_keys=[KEY_A, KEY_B],
                        store_event=5)
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        disk_store.on_store_completed(role, [])
    connector = FakeConnector()
    report = disk_spill.install_shutdown_flush(connector)
    assert report["installed"] is True
    assert report["already_installed"] is False
    connector.shutdown()
    assert connector.calls == 1, "the connector's own shutdown was not called"
    assert connector.depth_at_shutdown == 0, "the drain ran after the fork's shutdown"
    assert disk_spill.counters()["spilled"] == 2
    # Idempotent: the second install must not wrap twice, or one stop drains twice.
    again = disk_spill.install_shutdown_flush(connector)
    assert again["installed"] is False and again["already_installed"] is True
    before = connector.calls
    connector.shutdown()
    assert connector.calls == before + 1


def test_the_shutdown_drain_never_breaks_the_connectors_shutdown(tier, monkeypatch,
                                                                 caplog):
    """A drain that raises is one log line; the fork's teardown still runs.

    The fork wraps `worker.shutdown()` in `except BaseException` and exits 1, which
    would take the whole worker down over a diagnostic plugin's flush. So the wrapper
    owns its own failure, counts it, and passes the original through untouched - and
    the ORIGINAL's own exception still propagates, because that is the fork's story
    and not ours to swallow.
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("the flush raised")

    monkeypatch.setattr(disk_spill, "flush", explode)
    connector = FakeConnector()
    disk_spill.install_shutdown_flush(connector)
    with caplog.at_level(logging.WARNING):
        connector.shutdown()                    # must not raise
    assert connector.calls == 1
    assert disk_spill.counters()["flush_errors"] == 1

    class Raising(FakeConnector):
        def shutdown(self) -> None:
            super().shutdown()
            raise ValueError("the fork's own teardown failed")

    loud = Raising()
    disk_spill.install_shutdown_flush(loud)
    with pytest.raises(ValueError):
        loud.shutdown()
    assert loud.calls == 1


def test_the_shutdown_drain_is_gated_defaulted_and_clamped(monkeypatch):
    """`VLLM_E2_TIER_DISK_SHUTDOWN_FLUSH_SECONDS`: unset is 15 s, `0` is off, a typo is
    the default with its complaint, and the ceiling is `FLUSH_BUDGET_SECONDS_MAX`.

    Same env grammar as its neighbours: the shipped default is the claim, so the
    variable is deleted for every test by conftest's `clean_env` and a test that wants
    another value names it.
    """
    default = disk_spill.shutdown_flush_verdict()
    assert default["seconds"] == disk_spill.SHUTDOWN_FLUSH_SECONDS_DEFAULT
    assert default["seconds"] == 15.0
    assert default["source"] == "default"

    monkeypatch.setenv(config.ENV_DISK_SHUTDOWN_FLUSH_SECONDS, "0")
    off = disk_spill.shutdown_flush_verdict()
    assert off["seconds"] == 0.0 and off["enabled"] is False
    assert off["source"] == "env" and off["recognised"] is True

    monkeypatch.setenv(config.ENV_DISK_SHUTDOWN_FLUSH_SECONDS, "sometime")
    bad = disk_spill.shutdown_flush_verdict()
    assert bad["seconds"] == disk_spill.SHUTDOWN_FLUSH_SECONDS_DEFAULT
    assert bad["problems"], "a refused value must arrive with its complaint"

    monkeypatch.setenv(config.ENV_DISK_SHUTDOWN_FLUSH_SECONDS, "3600")
    big = disk_spill.shutdown_flush_verdict()
    assert big["seconds"] == disk_spill.FLUSH_BUDGET_SECONDS_MAX
    assert big["clamped"] is True


def test_the_shutdown_flush_installs_at_connector_creation_in_the_worker_role(
        monkeypatch):
    """The connector-creation hook installs the drain where the queue actually lives.

    THE WORKER CASE is the one that matters and it used to be a bare `return`: the
    spill queue is built in the worker process (`request_spill` runs in the worker's
    `get_finished`), and the worker ends its shutdown with `os._exit`, so the
    connector's own `shutdown` is the ONLY seam in that process that runs before the
    process is gone. The scheduler role gets it too, cheaply and harmlessly, so a
    single-process boot is covered by the same line of code.
    """
    connector = FakeConnector()
    monkeypatch.setattr(backport, "_is_tier_connector", lambda _c: True)
    backport._after_connector_created(connector)
    assert getattr(connector, disk_spill.SHUTDOWN_FLUSH_ATTR, None) == \
        disk_spill.SHUTDOWN_FLUSH_ATTR
    verdict = backport.shutdown_flush_state()
    assert verdict["installed"] is True
    assert verdict["budget_seconds"] == 15.0
    assert verdict["seams"] == ["connector.shutdown", "atexit"]
    # A non-tier connector is still nobody's business, and still counted as such.
    other = FakeConnector()
    monkeypatch.setattr(backport, "_is_tier_connector", lambda _c: False)
    backport._after_connector_created(other)
    assert getattr(other, disk_spill.SHUTDOWN_FLUSH_ATTR, None) is None


def test_a_disabled_budget_installs_nothing_and_says_so(tier, monkeypatch):
    """`0` means the stop path runs the fork's code and nothing of ours.

    The rollback has to be structurally visible, not just a budget of zero: a wrapper
    that runs and does nothing is a wrapper that can still fail.
    """
    monkeypatch.setenv(config.ENV_DISK_SHUTDOWN_FLUSH_SECONDS, "0")
    connector = FakeConnector()
    report = disk_spill.install_shutdown_flush(connector)
    assert report["installed"] is False
    assert report["reason_kind"] == "disabled"
    assert getattr(connector, disk_spill.SHUTDOWN_FLUSH_ATTR, None) is None
    connector.shutdown()
    assert connector.calls == 1


def _code_only(func) -> str:
    """A function's source with its docstring gone: the CODE is what is pinned.

    THE ONE LINE THAT MAKES THIS A TEST AND NOT A FORMALITY: `ast.get_docstring` answers
    about the node you hand it, and the node parsed out of `inspect.getsource(func)` is
    a
    Module whose first statement is a FunctionDef - so asking the Module for its
    docstring
    returns None, nothing is stripped, and the assertion silently degrades into "this
    function's English does not contain the word". Ask the FunctionDef instead, and
    assert
    the strip removed something.
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    target = tree.body[0]
    doc = ast.get_docstring(target, clean=False)
    if doc:
        stmt = target.body[0]
        lines = source.splitlines()
        kept = [line for number, line in enumerate(lines, start=1)
                if not (stmt.lineno <= number <= stmt.end_lineno)]
        assert len(kept) < len(lines), "the docstring strip stripped nothing"
        source = "\n".join(kept)
    return source


# --- 7. spill-side only, structurally -----------------------------------------

FORBIDDEN = ("set_disk_authority", "set_offer_switch", "set_armed", "disk_authority",
             "offers_enabled", "serving_as", "serve_as", "arm(", "disarm(",
             "boot_enable", "disk_boot")


@pytest.mark.parametrize("module,func", [
    (disk_spill, "flush"), (disk_spill, "install_shutdown_flush"),
    (disk_store, "_op_flush"),
])
def test_the_flush_names_no_posture_or_arm_writer_in_its_own_source(module, func):
    """STRUCTURE: a flush cannot become a posture change by accident.

    Docstrings stripped, code pinned - the `test_mamba_anchor_patch.py` idiom, for the
    same reason: the claim is about every path, and a behavioural test only sees one.
    """
    source = _code_only(getattr(module, func))
    for word in FORBIDDEN:
        assert word not in source, f"{module.__name__}.{func} names {word!r}"


def test_a_flush_moves_no_posture_no_arm_and_no_serving_state(tier):
    """Behavioural half of the same claim, on the live state of this process."""
    before = {"posture": disk_restore.disk_posture(),
              "spill_armed": disk_spill.is_armed(),
              "ram_armed": backport.is_armed()}
    answer = flush(5.0)
    assert answer["ok"] is True
    assert disk_restore.disk_posture() == before["posture"]
    assert disk_spill.is_armed() == before["spill_armed"]
    assert backport.is_armed() == before["ram_armed"]


def test_the_flush_reports_which_process_answered(tier):
    """The scope is in the payload, because the queue is per-process.

    `/disk` is answered by whichever process the utility call landed on, and on a
    4-rank boot the writer thread runs in the WORKERS - the same trap that made
    `op:status` report `evictions: 0` for a store that had shed 234 GiB. A flush that
    said only `drained: true` would be a fleet claim it cannot make, so it names the
    scope of its own answer and points at the rank rollup for the fleet.
    """
    answer = flush(5.0)
    assert answer["scope"]["role"] == disk_store.ROLE
    assert answer["scope"]["note"], "the scope must be explained, not just labelled"
    assert "spill_rollup" in answer or "ranks" in answer
