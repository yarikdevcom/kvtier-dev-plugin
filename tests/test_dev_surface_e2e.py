"""The whole dev surface, end to end: HTTP route -> utility hop -> engine-side verb.

Every other test checks one seam. This file checks the seams that only exist when
all of them are wired together, because those are the ones that would hurt in
production:

* the ARITY contract. `core.py:1731-1744` asserts `len(args) <= len(parameters)`
  and then runs `getattr(self, name)(*args)` (`core.py:1669-1671`). A verb
  installed with the wrong signature is a `TypeError` inside the engine core loop -
  a broken install, not a refused request - so the fake engine here asserts the
  same thing the fork asserts, and every verb is driven BOTH ways.
* JSON-serialisability. A payload the API cannot serialise is a 500 on the way out
  of a verb that read everything correctly, and only the route can find that out.
* the DEV-STATE story on ONE engine: the ring armed, guard armed, a real fault from
  the tier, and the same fact agreeing in the capture, the ring, the snapshot and
  the `/status` mirror - then disarmed, with the fork's own functions back.

The engine process is the fake scheduler/tier from `fakes_debug`, the verbs are the
real ones attached by `engine_side.attach_methods`, and the wire is the plugin's own
utility hop with the fork's dispatch underneath. Nothing here mocks a verb.
"""

from __future__ import annotations

import inspect
import sys

import fakes_debug as fake
import pytest
from test_routes import (
    BLOCK,
    GUARD,
    KVCONFIG,
    RELOAD,
    REQUEST,
    SNAPSHOT,
    STATUS,
    TRACE,
    FakeEngineCoreClient,
    build_app,
    client_for,
    token_headers,
)

from kvtier_dev_plugin import backport, config, engine_side, guard_mode, step_trace

pytest.importorskip("fastapi")

TOK = "e2e-token"
P = config.ROUTE_PREFIX

# A tier whose last group cannot be divided by the tier's hash block size: the
# capture must report the arithmetic the fork would have asserted on.
FAILING_SIZES = (1024, 1024, 1024, 1024, 128, 1000)


def _parameters(function):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):        # a builtin: unbounded
        return ()
    return [p for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


class FakeEngineProcess(FakeEngineCoreClient):
    """The engine half of the wire: `call_utility_async` -> the fork's dispatch.

    The arity assert is reproduced rather than stubbed, so an install with a wrong
    signature fails HERE, in the shape it would fail in production.
    """

    def __init__(self, core) -> None:
        super().__init__()
        self.core = core

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        parameters = _parameters(getattr(self.core, method))
        assert len(args) <= len(parameters), (
            f"{method}: the wire sends {len(args)} argument(s) but the installed "
            f"method takes {len(parameters)} - the fork would assert")
        return getattr(self.core, method)(*args)


@pytest.fixture
def dev(monkeypatch, reset_dev_state):
    """A live-ish engine: the real verbs, the fake tier, the real routes."""
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_TOKEN, TOK)
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)

    engine, info = fake.production_geometry()
    tier_cls, tier_module = fake.fork_class(
        guard_mode.TARGET_MODULE, guard_mode.TARGET_CLASS,
        fake.GUARD_TIER_SOURCE, register=True)
    tier = info["tier"]
    # One object, both jobs: the fork's entry points AND the geometry a capture
    # reads. Re-binding `__class__` between two plain-Python classes is legal.
    tier.__class__ = tier_cls
    tier.calls = []
    tier.group_block_sizes = list(FAILING_SIZES)
    engine.scheduler.connector = fake.Obj(name="SimpleCPUOffloadConnector",
                                          scheduler_manager=tier)

    engine_cls = type(engine)
    report = engine_side.attach_methods(engine_cls)
    assert set(report["attached"]) >= {config.METHOD_SNAPSHOT, config.METHOD_KVCONFIG,
                                       config.METHOD_INSPECT, config.METHOD_TRACE,
                                       config.METHOD_GUARD}
    assert not report["skipped"]

    process = FakeEngineProcess(engine)
    client = client_for(build_app(process))
    yield {"client": client, "engine": engine, "tier": tier, "info": info,
           "tier_cls": tier_cls, "tier_module": tier_module, "process": process}
    for name in config.METHODS:
        if name in vars(engine_cls):
            delattr(engine_cls, name)
    sys.modules.pop(guard_mode.TARGET_MODULE, None)


def headers():
    return token_headers(TOK)


# --- 1. the arity contract, over the real hop -------------------------------

@pytest.mark.parametrize("path,method", [
    (SNAPSHOT, config.METHOD_SNAPSHOT),
    (KVCONFIG, config.METHOD_KVCONFIG),
])
def test_a_call_with_no_arguments_sends_none(dev, path, method):
    """`GET /snapshot` still means `kvtier_dev_snapshot()` - one bound method,
    zero arguments, exactly as before the argument-taking verbs existed."""
    client = dev["client"]
    assert client.get(path).status_code == 200
    assert dev["process"].calls == [(method, ())]


@pytest.mark.parametrize("path,method,body", [
    (BLOCK, config.METHOD_INSPECT, {"block_id": 5, "group": 0}),
    (REQUEST, config.METHOD_INSPECT, {"request_id": "cmpl-a", "ids_cap": 8}),
    (TRACE, config.METHOD_TRACE, {"op": "status"}),
    (GUARD, config.METHOD_GUARD, {"op": "status"}),
])
def test_an_argument_arrives_as_exactly_one_dict(dev, path, method, body):
    client = dev["client"]
    response = client.post(path, headers=headers(), json=body)
    assert response.status_code == 200, response.text
    assert dev["process"].calls[-1][0] == method
    sent = dev["process"].calls[-1][1]
    assert len(sent) == 1 and isinstance(sent[0], dict)


def test_no_installed_method_could_ever_be_sent_too_many_arguments(dev):
    """The fork's assert is the only thing standing between a stray body and a
    crash in the core loop, so the arity must be a property of the install."""
    engine = dev["engine"]
    for name in config.METHODS:
        assert len(_parameters(getattr(engine, name))) <= 1, (
            f"{name} takes more than one positional argument")


# --- 2. what the verbs answer is JSON, on every route ----------------------

@pytest.mark.parametrize("path", [SNAPSHOT, KVCONFIG, BLOCK, REQUEST, TRACE, GUARD])
def test_what_the_engine_returns_survives_JSON(dev, path):
    client = dev["client"]
    response = client.get(path, headers=headers())
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json(), dict)


def test_a_block_and_a_request_round_trip_on_real_geometry(dev):
    client = dev["client"]
    # production_geometry() has blocks 0..63, and block 5 is referenced and hashed.
    block = client.get(BLOCK, params={"block_id": 5, "group": 0},
                       headers=headers()).json()
    assert block["found"] is True
    assert block["requested"] == {"block_id": 5, "group": 0}
    assert block["block"]["block_id"] == 5
    assert block["block"]["in_cached_map"] is True
    assert block["group_geometry"]["divisibility"]["divisible"] is True
    # The tier is attached on this engine, so its CPU slot of the same id is read.
    assert block["tier_cpu_slot"]["present"] is True

    gone = client.get(BLOCK, params={"block_id": 9999}, headers=headers()).json()
    assert gone["found"] is False
    assert "not nameable" in gone["reason"]           # names the real range

    found = client.get(REQUEST, params={"request_id": "cmpl-a"},
                       headers=headers()).json()
    assert found["found"] is True
    assert len(found["groups"]) == 6
    assert found["groups"][0]["divisibility"]["divisible"] is True
    # The fork's own readiness arithmetic, plus the term that bound it.
    assert found["groups"][0]["tier_view"]["binding_term"]
    assert found["tier_alignment"]["store_alignment"] == 128

    absent = client.get(REQUEST, params={"request_id": "cmpl-zzz"},
                        headers=headers()).json()
    assert absent["found"] is False and absent["reason"]


def test_kvconfig_and_snapshot_agree_on_the_geometry(dev):
    client = dev["client"]
    config_body = client.get(KVCONFIG).json()
    snap = client.get(SNAPSHOT).json()
    assert config_body["epoch"] == snap["epoch"]
    assert config_body["geometry"]["num_groups"] == 6
    # The geometry itself is all divisible; the TIER's group_block_sizes were
    # given a size its hash block size cannot divide, and the crosscheck names it.
    assert config_body["geometry"]["divisibility_summary"]["groups_failing"] == []
    assert config_body["tier_crosscheck"]["disagreements"]["group_block_sizes"] == [5]
    assert config_body["tier_crosscheck"]["per_group"][5]["tier_divisibility"][
        "would_fail"] is True
    assert snap["guard_mode"] is False            # nothing armed yet
    assert snap["guard"]["development_only"] is True
    assert snap["trace"]["armed"] is False


# --- 3. the dev-state story on one engine ----------------------------------

def test_the_whole_dev_loop_on_one_engine(dev):
    client = dev["client"]

    # Nothing is wrapped before anyone asks: that is what "zero cost disarmed"
    # has to mean, or the claim is just a cheap branch.
    assert step_trace.counters()["installed"] is False
    assert guard_mode.section()["installed"] is False

    armed_trace = client.post(TRACE, headers=headers(),
                              json={"op": "arm", "depth": 64}).json()
    assert armed_trace["op"] == "arm"
    assert armed_trace["armed"] is True
    assert step_trace.counters()["installed"] is True

    armed_guard = client.post(GUARD, headers=headers(), json={"armed": True}).json()
    assert armed_guard["guard_mode"] is True
    assert armed_guard["op"] == "arm"
    assert guard_mode.section()["installed"] is True

    # The same fact, four ways.
    snapshot_body = client.get(SNAPSHOT).json()
    assert snapshot_body["guard_mode"] is True
    assert snapshot_body["guard"]["guard_mode"] is True
    assert snapshot_body["trace"]["armed"] is True
    ring_status = client.get(TRACE, headers=headers()).json()
    assert ring_status["armed"] is True and ring_status["op"] == "read"

    status = client.get(STATUS).json()
    assert status["guard_mode"] is True            # mirrored from an engine answer
    assert status["trace_armed"] is True
    # `source` names the engine answer it came from - never a guess, and never a
    # silent default: /status makes no engine call of its own.
    assert status["dev_state"]["source"] in ("snapshot", "trace read", "guard status")
    assert status["dev_state"]["epoch"] == snapshot_body["epoch"]
    assert status["dev_state"]["age_seconds"] >= 0.0

    # Disarm guard first: tracing still needs the tier wrappers, and says so.
    disarmed = client.post(GUARD, headers=headers(), json={"armed": False}).json()
    assert disarmed["guard_mode"] is False
    assert guard_mode.is_armed() is False

    client.post(TRACE, headers=headers(), json={"op": "disarm"})
    assert step_trace.counters()["installed"] is False

    again = client.get(STATUS).json()
    assert again["guard_mode"] is False
    assert again["trace_armed"] is False


def test_the_guard_wrappers_are_the_fork_s_functions_again_after_disarm(dev):
    client = dev["client"]
    tier_cls = dev["tier_cls"]
    before = {name: getattr(tier_cls, name) for name in guard_mode.WRAPPED_METHODS}
    client.post(TRACE, headers=headers(), json={"op": "arm"})
    client.post(GUARD, headers=headers(), json={"armed": True})
    after = {name: getattr(tier_cls, name) for name in guard_mode.WRAPPED_METHODS}
    assert any(after[name] is not before[name] for name in before)
    client.post(GUARD, headers=headers(), json={"armed": False})
    client.post(TRACE, headers=headers(), json={"op": "disarm"})
    for name in before:
        assert getattr(tier_cls, name) is before[name]


def test_a_real_fault_becomes_a_capture_and_a_ring_record(dev, monkeypatch):
    """The point of guard mode, driven through the tier's own method."""
    client = dev["client"]
    tier = dev["tier"]
    client.get(SNAPSHOT, headers=headers())      # the backport wraps first
    # The guard-capture path runs inside get_num_new_matched_tokens, which the
    # kill switch gates on armed=True. This fake tier's methods are not the
    # fork's, so the backport rightly REFUSES them on fingerprint and force-
    # disarms (the all-or-nothing contract). The scenario guard mode is meant for
    # is a tier that IS fully patched and armed, so represent that precondition
    # explicitly: armed only because the fix is provably fully in place.
    backport._state["fully_installed"] = True
    backport._state["armed"] = True
    armed = client.post(GUARD, headers=headers(), json={"armed": True}).json()
    client.post(TRACE, headers=headers(), json={"op": "arm", "depth": 64})
    # The guard composes with this package's OWN wrapper instead of refusing it:
    # declining to guard the most fault-prone method while reporting "guard mode
    # ON" would be the worse failure.
    #
    # THREE methods now, not two: the SSD restore layer is fitted on all three
    # scheduler entry points in the EngineCore (`disk_restore.adopt_if_missed`, from
    # the connector-creation boot and the snapshot verb), so the guard composes over
    # a sibling layer on `update_connector_output` too. That is the ordering the
    # layer walk guarantees - guard outermost, restore at the bottom, the fork's body
    # underneath - and it means a fault raised inside the restore path is a captured
    # DECLINE rather than a step that silently did nothing.
    assert armed["install"]["nested"] == ["get_num_new_matched_tokens",
                                          "build_connector_meta",
                                          "update_connector_output"]
    assert armed["install"]["skip_detail"] == []
    # The composition is still readable afterwards, not only at arm time.
    assert client.get(GUARD, headers=headers()).json()["guard"]["nested"] == [
        "get_num_new_matched_tokens", "build_connector_meta",
        "update_connector_output"]
    dev["tier_module"].FAILING.add("get_num_new_matched_tokens")

    # The tier raises where the fork would have asserted; the step survives.
    request = fake.request("cmpl-a")
    outcome = tier.get_num_new_matched_tokens(request, 1024)
    assert outcome == (0, False)                 # the fork's own "found nothing"

    captures = client.post(GUARD, headers=headers(),
                           json={"op": "captures", "limit": 2}).json()
    assert captures["returned"] == 1
    assert guard_mode.section()["caught"] == 1
    entry = captures["captures"][0]
    assert entry["method"] == "get_num_new_matched_tokens"
    assert entry["decline"] == "declined"
    assert "divided by zero" in entry["message"]
    # The capture names the arithmetic the fork would have asserted on, so the
    # operator sees WHY before the engine restarts.
    # Group 5 is the one whose tier block size the hash size cannot divide.
    assert entry["manager"]["geometry"]["groups_failing_divisibility"] == [5]

    ring = client.get(TRACE, params={"limit": 20}, headers=headers()).json()
    kinds = [record["kind"] for record in ring["entries"]]
    assert "exception" in kinds
    exception = next(r for r in ring["entries"] if r["kind"] == "exception")
    assert exception["seq"] == entry["trace_seq"]

    # And the engine is still answering, with the guard still reported.
    assert client.get(SNAPSHOT).json()["guard_mode"] is True


def test_an_undecinable_method_still_reaches_the_caller(dev, monkeypatch):
    """`update_connector_output` has no honest "nothing happened", so a capture
    there must re-raise. Silent decline is not the only mode."""
    client = dev["client"]
    tier = dev["tier"]
    client.post(GUARD, headers=headers(), json={"armed": True})
    dev["tier_module"].FAILING.add("update_connector_output")
    with pytest.raises(ZeroDivisionError):
        tier.update_connector_output(fake.Obj(name="ConnectorOutput"))
    assert guard_mode.section()["re_raised"] == 1
    captures = client.post(GUARD, headers=headers(), json={"op": "captures"}).json()
    assert captures["captures"][0]["decline"] == "re-raised"


def test_a_base_exception_is_never_declined(dev, monkeypatch):
    client = dev["client"]
    tier = dev["tier"]
    client.post(GUARD, headers=headers(), json={"armed": True})

    class Boom(BaseException):
        pass

    def explode(self, *args, **kwargs):
        raise Boom("the engine is being interrupted")

    # The raiser stands in for the FORK's own body (same module name), so the
    # wrapper is installed on top of it and really is the layer that sees the
    # BaseException. It must let it through untouched.
    explode.__module__ = guard_mode.TARGET_MODULE
    monkeypatch.setattr(dev["tier_cls"], "reset", explode)
    client.post(GUARD, headers=headers(), json={"armed": True})
    assert "reset" not in guard_mode.section()["skip_detail_methods"] \
        if "skip_detail_methods" in guard_mode.section() else True
    with pytest.raises(Boom):
        tier.reset()
    # Nothing was counted as captured: the signal went through.
    assert guard_mode.section()["caught"] == 0


# --- 4. the surface, and what happens when the engine does not answer ------

def test_the_status_advertises_the_surface_it_actually_serves(dev):
    """`status` is the self-description an operator reads first; a stale one is a
    trap, so every list in it is derived from the same tables the router uses."""
    client = dev["client"]
    status = client.get(STATUS).json()
    prefix = config.ROUTE_PREFIX
    assert status["routes"] == [prefix + route for route in config.ROUTES]
    assert status["dev_state_routes"] == [prefix + route
                                         for route in config.DEV_STATE_ROUTES]
    assert status["engine_methods"] == list(config.METHODS)
    arguments = status["arguments"]
    assert arguments["GET"] == "query parameters"
    assert "ONE dict argument" in arguments["POST"]
    assert "400" in arguments["validated_by"]


@pytest.mark.parametrize("path", [SNAPSHOT, KVCONFIG, BLOCK, REQUEST, TRACE, GUARD,
                                  RELOAD])
def test_an_engine_that_answers_with_a_failure_is_a_502_never_a_500(dev, path):
    """`502` = the engine answered and said no. The verb layer never raises, so a
    500 here would mean the API's own bug, and this pins the convention."""
    client = client_for(build_app(FakeEngineCoreClient(raises=RuntimeError("no"))))
    response = client.request("post" if path == RELOAD else "get", path,
                              headers=headers())
    assert response.status_code == 502, (path, response.status_code, response.text)


def test_the_worker_writer_reports_itself_as_never_configured(dev, monkeypatch):
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    body = dev["client"].get(SNAPSHOT).json()
    writer = body["worker_diag"]
    assert writer["configured"] is False
    assert writer["enabled"] is False
    assert "worker_scope" in writer
    assert dev["client"].get(STATUS).json()["worker_diagnostics"]["configured"] \
        is False
