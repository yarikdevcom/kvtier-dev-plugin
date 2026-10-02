"""Containment: a diagnostic must never propagate into the scheduler.

The idiom is copied from this fork's own debug hook
(vllm/v1/core/sched/e2_alloc_log.py:143,151,201 with the capped `_error()` at
:92): count every failure, log at most N, and put a nested `except Exception`
around the writer so a broken log cannot become a second failure.

`except Exception`, never `except BaseException`: a KeyboardInterrupt or a
SystemExit from a diagnostic path has to reach the engine, because the engine
is the only thing that can act on it. test_base_exception_is_not_swallowed
pins that.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
from test_snapshot import (  # noqa: F401 - helper classes live in that module
    FakeConnector,
    FakeEngine,
    FakeManager,
    FakeScheduler,
    HostileLogger,
    QuietLogger,
    RaisingEngine,
    make_engine,
)

from kvtier_dev_plugin import config, engine_side, snapshot
from kvtier_dev_plugin.policy_holder import PolicyHolder


class BrokenSchedulerManager(FakeManager):
    def has_pending_stores(self):
        raise RuntimeError("busy counter blew up")

    def get_boundary_store_stats(self):
        raise ValueError("stats blew up")


class StaticHolder:
    """A holder with a policy that does something specific."""

    def __init__(self, decide=None, load_ok=True, load_error=None) -> None:
        self._decide = decide
        self._load_ok = load_ok
        self._load_error = load_error

    def ensure_loaded(self):
        if not self._load_ok:
            return {"ok": False, "error": self._load_error or "load failed",
                    "traceback": "Traceback (most recent call last): ..."}
        return {"ok": True, "error": None, "traceback": None}

    @property
    def decide(self):
        return self._decide


# --- the snapshot boundary --------------------------------------------------

def test_an_engine_that_raises_on_attribute_access_does_not_propagate(
        reset_snapshot_state, enabled):
    snapshot.attach_logger(QuietLogger())
    payload = snapshot.snapshot_for_engine(RaisingEngine(), StaticHolder())
    assert payload["error"].startswith("RuntimeError: scheduler exploded")
    assert payload["version"] == snapshot.PROTOCOL_VERSION
    assert payload["counters"]["errors"] == 1
    snapshot.attach_logger(None)


def test_a_broken_connector_counter_is_data_not_an_exception(reset_snapshot_state):
    logger = QuietLogger()
    snapshot.attach_logger(logger)
    payload = snapshot.snapshot_for_engine(
        make_engine(BrokenSchedulerManager()), StaticHolder())
    assert "error" not in payload                      # fully contained
    stats = payload["tier"]["boundary_store_stats"]
    assert stats["available"] is False and "raised" in stats["reason"]
    assert payload["tier"]["pending_stores"] is None   # the counter gave up
    assert reset_snapshot_state["errors"] == 2         # both counted
    assert len(logger.warnings) == 2
    snapshot.attach_logger(None)


def test_a_raising_policy_is_reported_and_the_rest_of_the_view_survives(
        reset_snapshot_state):
    snapshot.attach_logger(QuietLogger())

    def angry(view):
        raise RuntimeError("policy is angry")

    payload = snapshot.snapshot_for_engine(make_engine(), StaticHolder(angry))
    assert payload["policy"]["error"] == "RuntimeError: policy is angry"
    assert payload["policy"]["result"] is None
    assert payload["tier"]["present"] is True           # tier still readable
    assert reset_snapshot_state["errors"] == 1
    snapshot.attach_logger(None)


def test_a_policy_that_returns_garbage_is_still_contained():
    payload = snapshot.snapshot_for_engine(
        make_engine(), StaticHolder(lambda view: {"sets": {1, 2, 3}}))
    assert "error" in payload and "TypeError" in payload["error"]


def test_base_exception_is_not_swallowed():
    """The spec's rule: catch Exception, never BaseException."""

    def ctrl_c(view):
        raise KeyboardInterrupt

    def goodbye(view):
        raise SystemExit(3)  # noqa: RSE102 - carries an exit code on purpose

    engine = FakeEngine(FakeScheduler(FakeConnector(FakeManager())))
    with pytest.raises(KeyboardInterrupt):
        snapshot.snapshot_for_engine(engine, StaticHolder(ctrl_c))
    with pytest.raises(SystemExit):
        snapshot.snapshot_for_engine(engine, StaticHolder(goodbye))


def test_a_broken_log_writer_cannot_become_the_second_failure(reset_snapshot_state):
    snapshot.attach_logger(HostileLogger())
    payload = snapshot.snapshot_for_engine(RaisingEngine(), StaticHolder())
    assert payload["counters"]["errors"] == 1     # counted...
    assert payload["counters"]["logged"] == 1     # ...and it tried to log once
    snapshot.attach_logger(None)


def test_error_logging_is_capped(reset_snapshot_state):
    logger = QuietLogger()
    snapshot.attach_logger(logger)
    for _ in range(snapshot.MAX_ERROR_LOGS + 25):
        snapshot._error(RuntimeError("again"), "test")
    assert reset_snapshot_state["errors"] == snapshot.MAX_ERROR_LOGS + 25
    assert reset_snapshot_state["logged"] == snapshot.MAX_ERROR_LOGS
    assert len(logger.warnings) == snapshot.MAX_ERROR_LOGS
    snapshot.attach_logger(None)


# --- the reload boundary ----------------------------------------------------

def test_reloading_a_missing_file_reports_and_never_raises(tmp_path):
    holder = PolicyHolder(str(tmp_path / "nope.py"))
    report = snapshot.reload_policy_on_engine(holder)
    assert report["ok"] is False
    assert report["rolled_back"] is False      # nothing was live to roll back to
    assert "FileNotFoundError" in report["error"] or "No such file" in report["error"]
    assert holder.decide is None


def test_reload_keeps_the_previous_policy_on_failure(tmp_path):
    policy = tmp_path / "policy.py"
    policy.write_text("VERSION = 1\n"
                      "def decide(view):\n    return {'v': 1}\n")
    holder = PolicyHolder(str(policy))
    assert holder.ensure_loaded()["ok"] is True
    policy.write_text("VERSION = 2\ndef broken(:\n    pass\n")
    report = snapshot.reload_policy_on_engine(holder)
    assert report["ok"] is False
    assert report["rolled_back"] is True       # v1 is still what runs
    assert report["kept"]["source_sha256_8"] != report.get("applied")
    assert "SyntaxError" in report["traceback"]
    assert holder.decide({}) == {"v": 1}


def test_reload_result_is_json_safe(tmp_path):
    holder = PolicyHolder(str(tmp_path / "absent.py"))
    report = snapshot.reload_policy_on_engine(holder)
    import json

    json.dumps(report)


# --- the installer ----------------------------------------------------------

class Hostile:
    """Stands in for an EngineCore that already owns our method name."""

    kvtier_dev_snapshot = "not a function we installed"


def test_attach_refuses_to_clobber_a_foreign_attribute():
    class Engine:
        pass

    Engine.kvtier_dev_snapshot = Hostile.kvtier_dev_snapshot
    report = engine_side.attach_methods(Engine)
    assert Engine.kvtier_dev_snapshot == "not a function we installed"
    assert report["skipped"] == [config.METHOD_SNAPSHOT]
    # Everything else still lands: one hostile name does not cost the whole
    # surface. Listed from config.METHODS so a new verb cannot slip past here.
    assert report["attached"] == [name for name in config.METHODS
                                  if name != config.METHOD_SNAPSHOT]


def test_attach_is_idempotent_and_reports_a_replacement():
    class Engine:
        pass

    first = engine_side.attach_methods(Engine)
    assert sorted(first["attached"]) == sorted(config.METHODS)
    second = engine_side.attach_methods(Engine)
    assert second["attached"] == []
    assert sorted(second["replaced"]) == sorted(config.METHODS)
    assert getattr(Engine.kvtier_dev_snapshot,
                   engine_side.OWNER_ATTR) == engine_side.OWNER_TAG


def test_every_installed_method_takes_exactly_one_optional_argument():
    """The arity contract the HTTP layer depends on, for EVERY verb.

    `_convert_msgspec_args` asserts `len(args) <= len(parameters)`
    (vllm/v1/engine/core.py:1731-1744), so a verb that accepted only `self` would
    turn a JSON body into an AssertionError inside the engine loop. Uniform
    `(self, args=None)` is also what lets one HTTP body map onto one engine call
    for every verb, and what keeps a NO-argument call (what every GET sends)
    working exactly as it always did.
    """
    class Engine:
        pass

    engine_side.attach_methods(Engine)
    engine = Engine()
    engine.scheduler = None
    for name in config.METHODS:
        function = getattr(Engine, name)
        own = list(inspect.signature(function).parameters)
        assert own == ["self", "args"], (name, own)
        for parameter in inspect.signature(function).parameters.values():
            if parameter.name != "self":
                assert parameter.default is not inspect.Parameter.empty, name
                assert parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        # Both dispatch shapes the fork can produce must be answerable.
        answer = getattr(engine, name)()
        assert isinstance(answer, dict), name
        answer = getattr(engine, name)({})
        assert isinstance(answer, dict), name


def test_installed_methods_are_bound_engine_methods_with_no_arguments():
    class Engine:
        pass

    engine_side.attach_methods(Engine)
    engine = Engine()
    engine.scheduler = None
    # Called exactly the way UTILITY dispatch calls it: getattr(self, name)()
    payload = getattr(engine, config.METHOD_SNAPSHOT)()
    assert payload["tier"]["present"] is False
    report = getattr(engine, config.METHOD_RELOAD_POLICY)()
    assert set(["ok", "error", "traceback"]) <= set(report)


def test_register_is_a_no_op_when_inert():
    module = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        pass

    module.EngineCore = EngineCore

    sys.modules["vllm.v1.engine.core"] = module
    try:
        engine_side.register()
        assert not hasattr(EngineCore, config.METHOD_SNAPSHOT)
    finally:
        del sys.modules["vllm.v1.engine.core"]


def test_register_attaches_when_gated_on(monkeypatch):
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    module = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        pass

    module.EngineCore = EngineCore

    sys.modules["vllm.v1.engine.core"] = module
    try:
        engine_side.register()
        assert hasattr(EngineCore, config.METHOD_SNAPSHOT)
        assert hasattr(EngineCore, config.METHOD_RELOAD_POLICY)
        # And again: the second load (another process, another call) is a no-op.
        engine_side.register()
        assert getattr(EngineCore, config.METHOD_SNAPSHOT).__module__ == \
            "kvtier_dev_plugin.engine_side"
    finally:
        del sys.modules["vllm.v1.engine.core"]


def test_register_cannot_raise_when_vllm_is_absent(monkeypatch):
    """A worker process with no vllm.v1.engine.core imported must survive us."""
    monkeypatch.setenv(config.ENV_ENABLE, "1")

    saved = sys.modules.pop("vllm.v1.engine.core", None)
    try:
        engine_side.register()          # must not raise
    finally:
        if saved is not None:
            sys.modules["vllm.v1.engine.core"] = saved


def test_register_survives_a_module_without_the_class(monkeypatch):
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    module = types.ModuleType("vllm.v1.engine.core")

    sys.modules["vllm.v1.engine.core"] = module
    try:
        engine_side.register()          # logs, does not raise
    finally:
        del sys.modules["vllm.v1.engine.core"]
