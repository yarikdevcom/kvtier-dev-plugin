"""The worker-side writer: one small JSON file per rank, budgeted and atomic.

Why a file at all: `general_plugins` loads inside the worker processes, a worker
never imports `vllm.v1.engine.core`, and the UTILITY channel is EngineCore-only -
so a worker has no request path and no HTTP surface. A file is the only way to see
what a rank's host arena is actually doing, and the proxy host already reads these
files (`kv_tier_diagnostics.py` on the llm-api side).

The contract this file pins is therefore in two halves. The first is plugin
hygiene: inert unless `VLLM_E2_TIER_DIAG` names a directory, nothing wrapped
otherwise, the fork's return values untouched, a write failure recorded instead of
raised, and a write budgeted by BOTH cadence and change. The second is the wire
format, because a file the reader calls `malformed` is worse than no file:

    vllm/v1/simple_kv_offload/worker.py  the two wrapped entry points
    kv_tier_diagnostics.py (proxy host)  version/epoch/ts/role/rank required,
                                         rank must MATCH the file name,
                                         `arena.bytes_total`/`bytes_used` required
                                         non-negative ints, `counters.*` optional
                                         with a note, 256 KiB read cap
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import config, cpu_capacity, snapshot, worker_diag

# --- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def dev(reset_dev_state):
    yield reset_dev_state


@pytest.fixture
def worker_class():
    """The fork's worker class, exec'd under the fork's module name."""
    cls, module = fake.fork_class(worker_diag.TARGET_MODULE,
                                  worker_diag.TARGET_CLASS,
                                  fake.WORKER_SOURCE, register=True)
    yield cls
    sys.modules.pop(worker_diag.TARGET_MODULE, None)


@pytest.fixture
def writes_to(tmp_path, monkeypatch):
    """A diagnostics directory plus the env that turns the writer on."""
    directory = tmp_path / "kv-tier-diagnostics"
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "0")
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "1")
    return directory


def handler(**fields):
    """A worker handler: an arena, a block count, and the fork's own counters."""
    worker = fake.Obj(name="SimpleCPUOffloadWorker")
    worker.__dict__.update(cpu_kv_caches={"g0": fake.Obj(name="Cache", nbytes=4096)},
                           num_cpu_blocks=4,
                           cpu_capacity_bytes=16384,
                           _connector_metadata=None,
                           _pending_store_event_indices=[],
                           _pending_load_event_indices=[],
                           _completed_store_events={},
                           _store_hwm=0, _load_hwm=0,
                           kv_cache_config=None,
                           get_finished_calls=0, bind_calls=0)
    worker.__dict__.update(fields)
    return worker


def metadata(stores=0, loads=0):
    return fake.Obj(name="SimpleCPUOffloadMetadata",
                    store_cpu_blocks=[0] * stores, load_gpu_blocks=[0] * loads,
                    load_event=-1, store_event=-1)


# --- inert unless the directory is named ------------------------------------

def test_unset_means_no_files_and_nothing_wrapped(monkeypatch, tmp_path,
                                                  worker_class):
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    report = worker_diag.install(worker_class)
    assert report["installed"] is False
    assert config.ENV_DIAG_DIR in report["reason"]
    assert list(tmp_path.iterdir()) == []
    # The class is untouched: no owner markers, no wrappers.
    for name in worker_diag.WRAPPED_METHODS:
        assert getattr(getattr(worker_class, name), worker_diag.OWNER_ATTR,
                       None) is None


def test_configured_but_no_worker_class_yet_refuses_and_retries_later(writes_to):
    """`engine_side.register()` runs before the connector imports its worker
    module, so the first attempt is EXPECTED to decline - and must be cheap."""
    sys.modules.pop(worker_diag.TARGET_MODULE, None)
    first = worker_diag.install()
    assert first["installed"] is False
    assert "not imported in this process" in first["reason"]
    again = worker_diag.ensure_installed()
    assert again["installed"] is False
    assert worker_diag.counters()["skips"] == 1       # one log line, not two


def test_a_worker_handler_names_the_class_to_wrap(writes_to, worker_class):
    connector = fake.Obj(name="SimpleCPUOffloadConnector",
                         worker_handler=worker_class())
    report = worker_diag.ensure_installed(connector)
    assert report["installed"] is True
    assert report["class"] == "SimpleCPUOffloadWorker"
    assert sorted(report["attached"]) == sorted(worker_diag.WRAPPED_METHODS)
    # And a third attempt is a no-op that says so.
    assert worker_diag.ensure_installed(connector) == {
        "installed": True, "reason": "already installed"}


def test_an_already_attempted_class_is_not_retried_forever(writes_to):
    """`install()` gave a definite answer about THIS class; the handshake hook
    fires on every rank report, so the retry must not spin."""
    sys.modules.pop(worker_diag.TARGET_MODULE, None)
    worker_diag.install(object())                # any class, refused by __module__
    assert worker_diag.counters()["class"] is None
    report = worker_diag.ensure_installed()
    assert report["installed"] is False
    assert "already attempted" in report["reason"] or report["reason"]


def test_a_foreign_owner_marker_is_refused(writes_to, worker_class):
    setattr(worker_class.get_finished, worker_diag.OWNER_ATTR, "someone-else/1")
    report = worker_diag.install(worker_class)
    assert "get_finished" in report["skipped"]
    detail = {entry["method"]: entry["reason"] for entry in report["skip_detail"]}
    assert "another plugin" in detail["get_finished"]
    assert "bind_connector_metadata" in report["attached"]


def test_a_class_defined_outside_the_fork_module_is_refused(writes_to):
    Impostor = type("Impostor", (), {name: (lambda self, *a, **k: None)
                                     for name in worker_diag.WRAPPED_METHODS})
    report = worker_diag.install(Impostor)
    assert report["installed"] is False
    assert "nor the fork's own" in worker_diag._state["skip_reason"]


def test_uninstall_restores_the_worker_s_own_methods(writes_to, worker_class):
    before = {name: getattr(worker_class, name)
              for name in worker_diag.WRAPPED_METHODS}
    worker_diag.install(worker_class)
    report = worker_diag.uninstall(worker_class)
    assert report["uninstalled"] is True
    for name in worker_diag.WRAPPED_METHODS:
        assert getattr(worker_class, name) is before[name]


# --- the fork's behaviour is untouched --------------------------------------

def installed_worker(worker_class, writes_to, **fields):
    """Install the wrappers, then build one instance with the test's fields."""
    report = worker_diag.install(worker_class)
    assert report["installed"] is True, report
    worker = worker_class()
    worker.__dict__.update(fields)
    return worker


def test_the_wrappers_return_exactly_what_the_fork_returned(worker_class,
                                                            writes_to):
    worker = installed_worker(worker_class, writes_to)
    sentinel = object()
    assert worker.bind_connector_metadata(sentinel) is None
    assert worker._connector_metadata is sentinel         # the fork's own effect
    finished = worker.get_finished({"r1"})
    assert finished == (set(), None)
    assert worker.get_finished_calls == 1
    assert worker.bind_calls == 1


def test_bind_counts_the_blocks_handed_to_this_rank(worker_class, writes_to):
    worker = installed_worker(worker_class, writes_to)
    worker.bind_connector_metadata(metadata(stores=5, loads=3))
    counters = worker_diag.counters()
    assert counters["store_blocks_submitted"] == 5
    assert counters["load_blocks_submitted"] == 3
    # A step with no work adds nothing, and a None metadata is not a fault.
    worker.bind_connector_metadata(metadata())
    worker.bind_connector_metadata(None)
    assert worker_diag.counters()["store_blocks_submitted"] == 5


def test_a_counter_that_cannot_be_read_is_a_fallback_not_a_fault(
        worker_class, writes_to, reset_snapshot_state):
    class Lying:
        @property
        def store_cpu_blocks(self):
            raise RuntimeError("the metadata refused")

        load_gpu_blocks = None

    worker = installed_worker(worker_class, writes_to)
    worker.bind_connector_metadata(Lying())               # must not raise
    assert worker_diag.counters()["fallbacks"] >= 1
    assert reset_snapshot_state["errors"] >= 1


# --- the write: atomic, budgeted, and named the way the reader expects ------

def written_files(directory):
    return sorted(entry.name for entry in os.scandir(directory)) \
        if directory.exists() else []


def test_the_first_step_writes_one_file_named_the_way_the_reader_expects(
        worker_class, writes_to):
    worker = installed_worker(worker_class, writes_to)
    worker.get_finished(set())
    assert written_files(writes_to) == ["rank-PP0_TP0.json"]
    # No temporary file is left behind: the write is tmp-then-rename.
    assert not [name for name in written_files(writes_to) if ".tmp" in name]
    assert worker_diag.counters()["writes"] == 1


def test_the_file_is_valid_json_with_every_field_the_reader_requires(
        worker_class, writes_to):
    worker = installed_worker(worker_class, writes_to)
    worker.bind_connector_metadata(metadata(stores=2, loads=1))
    worker.get_finished(set())
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    # version/epoch/ts/role/rank are REQUIRED by the reader: a file missing one is
    # classified `malformed`, which is worse than a file that does not exist.
    assert document["version"] == worker_diag.SCHEMA_VERSION == 1
    assert isinstance(document["epoch"], str) and 0 < len(document["epoch"]) <= 32
    assert isinstance(document["ts"], float) and abs(document["ts"] - time.time()) \
        < 300
    assert document["role"] == "worker"
    assert document["rank"] == "PP0_TP0"
    assert document["rank"] == "PP0_TP0"          # the reader demands rank == file
    arena = document["arena"]
    for key in ("bytes_total", "bytes_used", "num_blocks"):
        assert isinstance(arena[key], int) and not isinstance(arena[key], bool)
        assert arena[key] >= 0
    counters = document["counters"]
    assert counters["stored"] == 2
    assert counters["restored_hits"] == 1
    assert all(isinstance(counters[key], int) and counters[key] >= 0
               for key in counters)


def test_the_worker_omits_the_two_scheduler_only_counters_by_contract(
        worker_class, writes_to):
    """`dropped_cpu_full` and `boundary_stored` are scheduler-side
    (`BoundaryStoreStats`, manager.py:125-132): the worker would have to invent
    them. The reader notes an absent counter and adds 0, which is the honest
    aggregate; a fabricated zero here would be indistinguishable from a reading."""
    worker = installed_worker(worker_class, writes_to)
    worker.get_finished(set())
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    assert "dropped_cpu_full" not in document["counters"]
    assert "boundary_stored" not in document["counters"]
    assert document["counters"]["store_events_pending"] == 0


def test_a_re_write_replaces_the_file_rather_than_appending(worker_class,
                                                            writes_to):
    worker = installed_worker(worker_class, writes_to)
    worker.get_finished(set())
    first = (writes_to / "rank-PP0_TP0.json").read_bytes()
    worker.num_cpu_blocks = 8
    worker.get_finished(set())
    second = (writes_to / "rank-PP0_TP0.json").read_bytes()
    assert written_files(writes_to) == ["rank-PP0_TP0.json"]
    assert json.loads(second)["arena"]["num_blocks"] == 8
    assert first != second


def test_the_arena_size_comes_from_the_real_tensors_then_the_config(
        worker_class, writes_to):
    worker = installed_worker(worker_class, writes_to)
    worker.get_finished(set())
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    assert document["arena"]["bytes_total"] == 4096          # one 4096-byte cache
    assert document["writer"]["bytes_total_source"] == "cpu_kv_caches.nbytes"
    worker.cpu_kv_caches = {}
    worker.num_cpu_blocks = 9
    worker.get_finished(set())
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    assert document["arena"]["bytes_total"] == 16384         # the configured value
    assert "configured" in document["writer"]["bytes_total_source"]


def test_bytes_used_is_a_proven_floor_from_the_blocks_in_flight(worker_class,
                                                                writes_to):
    """4096 bytes / 4 blocks = 1024 per block, so three blocks on their way to a
    store is a floor of 3072 bytes - stated as a floor, with the arithmetic."""
    worker = installed_worker(worker_class, writes_to)
    worker.bind_connector_metadata(metadata(stores=3))
    worker.get_finished(set())
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    assert document["arena"]["bytes_used"] == 3 * 1024
    assert "PROVEN FLOOR" in document["writer"]["bytes_used_meaning"]


def test_per_group_block_bytes_are_the_fork_s_own_expression(worker_class,
                                                             writes_to):
    specs = [fake.attention_spec(page_size_bytes=16384),
             fake.mamba_spec(page_size_bytes=8192)]
    groups = [fake.group(spec, [f"l{i}" for i in range(4)]) for spec in specs]
    worker = installed_worker(worker_class, writes_to,
                              kv_cache_config=fake.kv_cache_config(groups))
    worker.get_finished(set())
    arena = json.loads((writes_to / "rank-PP0_TP0.json").read_text())["arena"]
    assert arena["block_bytes_by_group"] == [4 * 16384, 4 * 8192]
    assert "page_size_bytes" in \
        json.loads((writes_to / "rank-PP0_TP0.json").read_text())["writer"][
            "block_bytes_by_group_meaning"]


def test_a_group_missing_a_term_omits_the_list_and_says_why(worker_class,
                                                            writes_to):
    groups = [fake.group(fake.attention_spec(), ["l0"])]
    groups[0].kv_cache_spec = fake.Obj(name="NoPageSize")
    worker = installed_worker(worker_class, writes_to,
                              kv_cache_config=fake.kv_cache_config(groups))
    worker.get_finished(set())
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    assert "block_bytes_by_group" not in document["arena"]
    assert document["writer"]["notes"] == [
        "a group is missing layer_names or page_size_bytes"]


# --- the budget -------------------------------------------------------------

def test_the_cadence_and_the_change_test_both_open_the_door(worker_class,
                                                            writes_to,
                                                            monkeypatch):
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "1000")     # never due by cadence
    worker = installed_worker(worker_class, writes_to)
    worker.get_finished(set())
    # The FIRST step always writes: nothing has ever been reported, so "no file"
    # would be indistinguishable from "the tier is not running on this rank".
    assert written_files(writes_to) == ["rank-PP0_TP0.json"]
    writes = worker_diag.counters()["writes"]
    # Unchanged handler, and the cadence is far away: the signature test closes
    # the door, and no write happens.
    worker.get_finished(set())
    worker.get_finished(set())
    assert worker_diag.counters()["writes"] == writes
    # Something changed: the file is rewritten even though the cadence says no.
    worker.num_cpu_blocks = 5
    worker.get_finished(set())
    assert worker_diag.counters()["writes"] == writes + 1
    assert json.loads((writes_to / "rank-PP0_TP0.json").read_text())["arena"][
        "num_blocks"] == 5


def test_the_rate_floor_holds_even_when_everything_else_says_write(
        worker_class, writes_to, monkeypatch):
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "3600")   # one hour floor
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "1")
    worker = installed_worker(worker_class, writes_to)
    worker.get_finished(set())
    assert len(written_files(writes_to)) == 1
    writes = worker_diag.counters()["writes"]
    for blocks in (5, 6, 7):
        worker.num_cpu_blocks = blocks
        worker.get_finished(set())
    assert worker_diag.counters()["writes"] == writes
    assert worker_diag.counters()["budget_skips"] == 3


def test_the_interval_and_floor_are_read_once_and_reported(writes_to, monkeypatch):
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "7")
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "2.5")
    worker_diag.install()
    section = worker_diag.section()
    assert section["interval_steps"] == 7
    assert section["min_interval_seconds"] == 2.5
    assert section["configured"] is True
    assert section["dir"] == str(writes_to)


def test_an_absurd_interval_or_floor_is_clamped_not_trusted(monkeypatch,
                                                            writes_to):
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "10000000")
    assert config.worker_write_interval_steps() == config.MAX_DIAG_INTERVAL_STEPS
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "0")
    assert config.worker_write_interval_steps() == config.MIN_DIAG_INTERVAL_STEPS
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "-4")
    assert config.worker_min_interval_seconds() == config.MIN_DIAG_MIN_INTERVAL_SECONDS
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "1e9")
    assert config.worker_min_interval_seconds() == config.MAX_DIAG_MIN_INTERVAL_SECONDS
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "banana")
    assert config.worker_min_interval_seconds() == \
        config.DEFAULT_DIAG_MIN_INTERVAL_SECONDS


# --- rank and epoch ---------------------------------------------------------

def test_an_operator_pinned_rank_wins_and_must_be_legal(writes_to, monkeypatch,
                                                        worker_class):
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP1_TP3")
    worker_diag.install(worker_class)
    worker = worker_class()
    worker.get_finished(set())
    assert written_files(writes_to) == ["rank-PP1_TP3.json"]
    # The reader refuses a rank that does not match its file name, so an illegal
    # id must never reach the disk at all.
    monkeypatch.setenv(config.ENV_DIAG_RANK, "..")
    worker_diag.reset_state_for_tests()
    worker_diag.install(worker_class)
    worker.get_finished(set())
    assert written_files(writes_to) == ["rank-PP1_TP3.json"]     # no second file
    assert "not PP<n>_TP<m>" in worker_diag._maybe_write(worker)["reason"]


@pytest.mark.parametrize("candidate", ["PP0_TP0", "PP10_TP11", "PP0_TP0_TP0",
                                       "PP_TP", "PPx_TPy", ""])
def test_the_rank_id_the_reader_will_accept(candidate):
    assert worker_diag._valid_rank(candidate) == (
        candidate in ("PP0_TP0", "PP10_TP11"))


def test_with_no_pinned_rank_the_distributed_state_is_asked(writes_to, monkeypatch,
                                                            worker_class):
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)
    module = types.ModuleType("vllm.distributed.parallel_state")
    module.get_pp_group = lambda: fake.Obj(name="Group", rank_in_group=0)
    module.get_tp_group = lambda: fake.Obj(name="Group", rank_in_group=1)
    saved = sys.modules.get("vllm.distributed.parallel_state")
    sys.modules["vllm.distributed.parallel_state"] = module
    try:
        worker_diag.install(worker_class)
        assert worker_diag.rank_id() == ("PP0_TP1", "parallel_state")
        worker = worker_class()
        worker.get_finished(set())
        assert written_files(writes_to) == ["rank-PP0_TP1.json"]
    finally:
        if saved is None:
            sys.modules.pop("vllm.distributed.parallel_state", None)
        else:
            sys.modules["vllm.distributed.parallel_state"] = saved


def test_a_process_with_no_distributed_state_writes_nothing(writes_to, monkeypatch,
                                                            worker_class):
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)
    sys.modules.pop("vllm.distributed.parallel_state", None)
    worker_diag.install(worker_class)
    rank, reason = worker_diag.rank_id()
    assert rank is None
    assert "parallel_state is not imported" in reason
    worker_class().get_finished(set())
    assert written_files(writes_to) == []


def test_a_group_that_refuses_to_answer_is_a_reason_not_a_crash(writes_to,
                                                                monkeypatch,
                                                                worker_class):
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)
    module = types.ModuleType("vllm.distributed.parallel_state")

    def refuse():
        raise RuntimeError("distributed state is not initialised")

    module.get_pp_group = refuse
    module.get_tp_group = refuse
    sys.modules["vllm.distributed.parallel_state"] = module
    try:
        rank, reason = worker_diag.rank_id()
        assert rank is None and "refused to answer" in reason
    finally:
        sys.modules.pop("vllm.distributed.parallel_state", None)


def test_the_epoch_is_pinned_by_the_operator_or_stable_per_process(writes_to,
                                                                   monkeypatch,
                                                                   worker_class):
    monkeypatch.setenv(config.ENV_DIAG_EPOCH, "boot-a1b2c3")
    worker_diag.install(worker_class)
    worker_class().get_finished(set())
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    assert document["epoch"] == "boot-a1b2c3"
    assert document["writer"]["epoch_scope"] == "env"
    monkeypatch.delenv(config.ENV_DIAG_EPOCH)
    first, scope = config.diagnostics_epoch()
    second, _ = config.diagnostics_epoch()
    assert first == second and scope == "process"       # one boot, one epoch
    assert 0 < len(first) <= config.MAX_DIAG_EPOCH_CHARS


# --- failures are data ------------------------------------------------------

def test_a_directory_that_cannot_be_created_is_recorded_not_raised(monkeypatch,
                                                                    tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory")
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(blocked / "sub"))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setenv(config.ENV_DIAG_MIN_INTERVAL, "0")
    worker = handler()
    report = worker_diag.write_snapshot(worker, "PP0_TP0", "env")
    assert report["written"] is False
    assert worker_diag.counters()["write_errors"] == 1
    assert worker_diag.counters()["last_error"]


def test_an_oversized_document_is_trimmed_then_written(writes_to, monkeypatch,
                                                       worker_class):
    monkeypatch.setattr(worker_diag, "MAX_FILE_BYTES", 600)
    worker_diag.install(worker_class)
    report = worker_diag.write_snapshot(handler(), "PP0_TP0", "env")
    assert report["written"] is True
    document = json.loads((writes_to / "rank-PP0_TP0.json").read_text())
    assert document["writer"]["trimmed"] is True
    assert "exceeded" in document["writer"]["trimmed_reason"]
    assert len((writes_to / "rank-PP0_TP0.json").read_bytes()) <= 600


def test_a_document_that_cannot_be_trimmed_enough_is_refused(writes_to, monkeypatch,
                                                             worker_class):
    monkeypatch.setattr(worker_diag, "MAX_FILE_BYTES", 40)
    worker_diag.install(worker_class)
    report = worker_diag.write_snapshot(handler(), "PP0_TP0", "env")
    assert report["written"] is False
    assert "refusing to write" in report["reason"]
    assert written_files(writes_to) == []
    assert worker_diag.counters()["write_errors"] == 1


def test_a_write_never_breaks_the_step_even_when_the_writer_is_broken(
        worker_class, writes_to, monkeypatch, reset_snapshot_state):
    def explode(*args, **kwargs):
        raise RuntimeError("the writer is on fire")

    monkeypatch.setattr(worker_diag, "_maybe_write", explode)
    worker = installed_worker(worker_class, writes_to)
    assert worker.get_finished({"r1"}) == (set(), None)     # the fork's answer


def test_no_directory_configured_means_no_write_reason(writes_to, monkeypatch):
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    report = worker_diag._maybe_write(handler())
    assert report["written"] is False
    assert config.ENV_DIAG_DIR in report["reason"]


# --- the wiring and the visibility ------------------------------------------

def test_the_plugin_registers_the_handshake_retry_hook(monkeypatch, tmp_path):
    """`engine_side.register()` cannot install the worker writer itself (the class
    does not exist yet), so it hands a LISTENER to the capacity handshake, which
    fires exactly once this rank has a handler.

    That listener drives BOTH file-channel facilities - the rank-file writer and
    the raw-byte dump hook - because both wrap the worker's `get_finished`, and it
    drives the writer FIRST because the rank files are the one input the proxy
    host's reader has. So the assertion is no longer "is this function exactly
    `worker_diag.ensure_installed`" (a composite is the right shape now) but "does
    firing the hook install both, with the dump composed ON TOP of the writer".
    """
    from kvtier_dev_plugin import (  # noqa: PLC0415 - the modules under test
        blockdump,
        engine_side,
    )
    saved = cpu_capacity._worker_report_hook
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "diag"))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    try:
        engine_side.register()
        assert cpu_capacity._worker_report_hook is engine_side._on_worker_report
        # Nothing has been wrapped yet: at plugin load the worker class does not
        # exist, which is the whole reason the retry exists.
        assert worker_diag.counters()["installed"] is False
        assert blockdump.counters()["installed"] is False
        # Fire it the way the handshake does, with a connector naming a real
        # worker class: both facilities land, one on top of the other.
        cls, _module = fake.fork_class(worker_diag.TARGET_MODULE,
                                       worker_diag.TARGET_CLASS,
                                       fake.WORKER_SOURCE, register=True)
        connector = fake.Obj(name="Connector", worker_handler=cls())
        try:
            cpu_capacity._call_worker_report_hook(connector)
            assert worker_diag.counters()["installed"] is True
            assert blockdump.counters()["installed"] is True
            assert blockdump.counters()["nested"] == ["get_finished"]
            assert worker_diag.counters()["nested"] == []
        finally:
            sys.modules.pop(worker_diag.TARGET_MODULE, None)
    finally:
        cpu_capacity.set_worker_report_hook(saved)


def test_a_broken_first_listener_does_not_cost_the_second_one(monkeypatch,
                                                               tmp_path):
    """Two facilities share one hook, so the rank files must not depend on the dump
    hook's health. The composite contains each call on its own."""
    from kvtier_dev_plugin import blockdump, engine_side

    def broken(connector):
        raise RuntimeError("the writer is broken")

    seen = []
    monkeypatch.setattr(engine_side, "worker_diag",
                        fake.Obj(name="wd", ensure_installed=broken))
    monkeypatch.setattr(blockdump, "ensure_installed",
                        lambda connector: seen.append(connector))
    connector = fake.Obj(name="Connector")
    engine_side._on_worker_report(connector)     # must not raise
    assert seen == [connector]


def test_a_broken_hook_never_breaks_a_handshake(monkeypatch):
    def hook(connector):
        raise RuntimeError("the listener is broken")

    monkeypatch.setattr(cpu_capacity, "_worker_report_hook", hook)
    # The hook is called from the worker-side wrapper; a listener must never be
    # able to break the handshake it was registered to observe.
    cpu_capacity._call_worker_report_hook(fake.Obj(name="Connector"))
    assert cpu_capacity._state["worker_fallbacks"] >= 1


def test_the_snapshot_publishes_the_worker_configuration_without_reading_files():
    """The ENGINE process cannot see a worker's counters, and must not pretend to:
    `worker_scope` says so in the payload."""
    payload = snapshot.snapshot_for_engine(fake.engine(None), None)
    section = payload["worker_diag"]
    assert section["configured"] is False              # env unset in this test
    assert "not a worker" in section["worker_scope"]
    assert section["schema_version"] == worker_diag.SCHEMA_VERSION
    assert section["file_pattern"] == "rank-PP0_TP0.json".replace("PP0_TP0",
                                                                  "PP<pp>_TP<tp>")


def test_the_status_endpoint_reports_the_directory_it_was_told_to_use(monkeypatch,
                                                                      tmp_path):
    from kvtier_dev_plugin import endpoint
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path))
    monkeypatch.setenv(config.ENV_DIAG_INTERVAL, "17")
    payload = endpoint.status_payload()
    assert payload["worker_diagnostics"]["configured"] is True
    assert payload["worker_diagnostics"]["dir"] == str(tmp_path)
    assert payload["worker_diagnostics"]["interval_steps"] == 17
    assert payload["worker_diagnostics"]["env"] == config.ENV_DIAG_DIR


def test_the_writer_writes_only_into_the_directory_it_was_given(writes_to,
                                                                worker_class,
                                                                monkeypatch,
                                                                tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    worker_diag.install(worker_class)
    worker_class().get_finished(set())
    assert written_files(writes_to) == ["rank-PP0_TP0.json"]
    assert list(other.iterdir()) == []
