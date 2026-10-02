"""The new verbs over HTTP: how an argument travels, and what may not travel.

The UTILITY channel is a `getattr(self, method_name)(*args)` call
(vllm/v1/engine/core.py:1669-1671), which makes two things a security property
rather than a convenience:

* the METHOD is chosen by the route, never by the wire - otherwise a token holder
  would hold an arbitrary-method-call primitive on the EngineCore;
* the ARGUMENTS are a plain dict of JSON primitives within bounds, validated
  against `endpoint.ARG_SPECS` before the call, so no verb receives a shape its
  own module did not declare.

GET and POST are held to the SAME spec: a bound you can evade by switching
methods is not a bound. `/trace` and `/guard` can change engine dev state, so
they require the token in every shape, loopback included - the one place where a
read is not free.
"""

from __future__ import annotations

import pytest
from test_routes import (
    BLOCK,
    GUARD,
    KVCONFIG,
    PROXY,
    RELOAD,
    REQUEST,
    SNAPSHOT,
    STATUS,
    TIER,
    TRACE,
    FakeAsyncLLM,
    FakeEngineCoreClient,
    build_app,
    client_for,
    token_headers,
)

from kvtier_dev_plugin import config, endpoint
from kvtier_dev_plugin.endpoint import TOKEN_HEADER

pytest.importorskip("fastapi")

# This module is about arguments and routing, not about the plugin being off, so
# every test runs with it on. The inert/peer/token-order tests live in
# test_routes.py and are parametrized over the WHOLE route list there.
@pytest.fixture(autouse=True)
def plugin_on(monkeypatch):
    monkeypatch.setenv(config.ENV_ENABLE, "1")


@pytest.fixture
def engine():
    return FakeEngineCoreClient(result={"version": 1, "verb": "inspect"})


@pytest.fixture
def armed_token(monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")


# --- the argument hop -------------------------------------------------------

def test_a_post_body_becomes_one_dict_argument(engine):
    """Exactly one positional argument, exactly the validated keys: this is the
    whole shape of the hop, and the engine-side arity contract exists for it."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(BLOCK, json={"block_id": 412, "group": 5})
    assert response.status_code == 200
    assert engine.calls == [(config.METHOD_INSPECT, ({"block_id": 412,
                                                      "group": 5},))]


def test_a_query_string_takes_the_same_spec_as_a_body(engine):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.get(BLOCK, params={"block_id": "412", "group": "5"})
    assert response.status_code == 200
    # Parsed to ints: the engine must not have to guess whether "412" is an id.
    assert engine.calls == [(config.METHOD_INSPECT, ({"block_id": 412,
                                                      "group": 5},))]


def test_a_request_id_travels_as_a_string(engine):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    client.get(REQUEST, params={"request_id": "cmpl-a"})
    client.post(REQUEST, json={"request_id": "cmpl-a", "ids_cap": 8})
    assert engine.calls == [(config.METHOD_INSPECT, ({"request_id": "cmpl-a"},)),
                            (config.METHOD_INSPECT,
                             ({"request_id": "cmpl-a", "ids_cap": 8},))]


@pytest.mark.parametrize("path,method", [
    (SNAPSHOT, "get"), (KVCONFIG, "get"),
])
def test_a_call_with_no_arguments_still_sends_no_arguments(engine, path, method):
    """The old contract is untouched: a message with no arguments is `method()`,
    which is what every caller of this plugin has always sent."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.request(method, path).status_code == 200
    expected = {SNAPSHOT: config.METHOD_SNAPSHOT,
                KVCONFIG: config.METHOD_KVCONFIG}[path]
    assert engine.calls == [(expected, ())]


def test_an_empty_body_is_no_arguments_not_an_empty_dict(engine):
    """`{}` on the wire would be a positional `{}` to a method that has never seen
    one; "no keys" and "no arguments" mean the same thing here."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(KVCONFIG).status_code == 200
    assert engine.calls == [(config.METHOD_KVCONFIG, ())]


def test_the_route_picks_the_method_and_the_body_cannot(engine):
    """The one thing a token holder must never control: which EngineCore method
    runs. `{"method": ...}` is an unknown key, and even a known key cannot redirect
    the call."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    refused = client.post(BLOCK, json={"method": "kvtier_dev_reload_policy"})
    assert refused.status_code == 400
    allowed = client.post(BLOCK, json={"block_id": 1, "op": "arm"})
    assert allowed.status_code == 400          # `op` is not a /block key either
    assert engine.calls == []


# --- the bounds -------------------------------------------------------------

@pytest.mark.parametrize("body,fragment", [
    ({"block_id": -1}, "block_id"),
    ({"block_id": 2 ** 40}, "must be <="),
    ({"block_id": "abc"}, "must be an integer"),
    ({"block_id": True}, "not a boolean"),
    ({"block_id": 1.5}, "must be an integer"),
    ({"group": 99}, "group"),
    ({"ids_cap": 0}, "ids_cap"),
    ({"block_id": 1, "colour": "blue"}, "unsupported request keys"),
    ({"request_id": "x" * 200}, "at most 128 characters"),
    ({"request_id": "a\nb"}, "control characters"),
    ({"request_id": 7}, "must be a string"),
])
def test_a_body_outside_the_spec_is_a_400_that_never_reaches_the_engine(
        engine, body, fragment):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(BLOCK if "request_id" not in body else REQUEST,
                           json=body)
    assert response.status_code == 400
    assert fragment in response.json()["detail"]
    assert engine.calls == []


@pytest.mark.parametrize("params,fragment", [
    ({"block_id": "-3"}, "must be >="),
    ({"block_id": ""}, "must not be empty"),
    ({"block_id": "412", "colour": "red"}, "unsupported request keys"),
])
def test_a_query_string_outside_the_spec_is_the_same_400(engine, params, fragment):
    """Symmetry: switching from POST to GET must not widen what is accepted."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.get(BLOCK, params=params)
    assert response.status_code == 400
    assert fragment in response.json()["detail"]
    assert engine.calls == []


@pytest.mark.parametrize("body", ['"a string"', "[1,2]", "42", "null"])
def test_a_body_that_is_not_an_object_is_a_400(engine, body):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(BLOCK, content=body,
                           headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert "JSON object" in response.json()["detail"]
    assert engine.calls == []


def test_a_malformed_body_is_a_400_not_a_500(engine):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(BLOCK, content="{not json",
                       headers={"content-type": "application/json"}
                       ).status_code == 400


def test_a_snapshot_query_parameter_is_refused_so_nothing_silently_ignores_it(
        engine):
    """`?cachebuster=1` would be an argument nobody validated: the strict rule is
    that an unrecognised key is an error, on every route."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.get(SNAPSHOT, params={"_": "1"}).status_code == 400
    assert engine.calls == []


# --- the dev-state routes: token in every shape -----------------------------

def test_the_trace_route_needs_the_token_even_from_loopback(engine, armed_token):
    """The one read that is not free: this route can arm the recorder, so the
    whole route is token-gated rather than the individual op."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.get(TRACE).status_code == 403
    assert TOKEN_HEADER in client.post(TRACE, json={"op": "read"}).json()["detail"]
    assert engine.calls == []
    assert client.get(TRACE, headers=token_headers()).status_code == 200


def test_a_trace_read_defaults_to_the_read_op_without_inventing_one_elsewhere(
        engine, armed_token):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    client.get(TRACE, headers=token_headers())
    assert engine.calls == [(config.METHOD_TRACE, ({"op": "read"},))]
    client.get(TRACE, params={"limit": 5}, headers=token_headers())
    assert engine.calls[-1] == (config.METHOD_TRACE, ({"op": "read", "limit": 5},))


def test_arming_the_recorder_over_http(engine, armed_token):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(TRACE, json={"op": "arm", "depth": 64},
                           headers=token_headers())
    assert response.status_code == 200
    assert engine.calls == [(config.METHOD_TRACE, ({"op": "arm", "depth": 64},))]


def test_the_guard_route_is_gated_the_same_way_and_carries_its_op(engine,
                                                                  armed_token):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(GUARD, json={"armed": True}).status_code == 403
    response = client.post(GUARD, json={"armed": True}, headers=token_headers())
    assert response.status_code == 200
    assert engine.calls == [(config.METHOD_GUARD, ({"armed": True},))]
    client.get(GUARD, headers=token_headers())
    assert engine.calls[-1] == (config.METHOD_GUARD, ({"op": "status"},))


@pytest.mark.parametrize("body,fragment", [
    ({"op": "launch"}, "op must be one of"),
    ({"op": "arm", "depth": 1}, "depth"),
    ({"op": "read", "limit": 10 ** 6}, "limit"),
    ({"op": "read", "after_seq": -1}, "after_seq"),
])
def test_a_trace_argument_outside_its_bounds_is_refused(engine, armed_token, body,
                                                        fragment):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(TRACE, json=body, headers=token_headers())
    assert response.status_code == 400
    assert fragment in response.json()["detail"]
    assert engine.calls == []


@pytest.mark.parametrize("body,fragment", [
    ({"op": "explode"}, "op must be one of"),
    ({"armed": "maybe"}, "must be a boolean"),
    ({"limit": 99}, "limit"),
    ({"traceback_chars": -1}, "traceback_chars"),
])
def test_a_guard_argument_outside_its_bounds_is_refused(engine, armed_token, body,
                                                        fragment):
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(GUARD, json=body, headers=token_headers())
    assert response.status_code == 400
    assert fragment in response.json()["detail"]


def test_a_remote_peer_still_needs_the_token_for_every_new_verb(engine,
                                                                monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    for path, method in ((KVCONFIG, "get"), (BLOCK, "get"), (BLOCK, "post"),
                         (REQUEST, "post"), (TRACE, "get"), (GUARD, "post")):
        assert client.request(method, path).status_code == 403
    assert engine.calls == []


# --- status: what this process has learned ----------------------------------

def test_status_reports_guard_mode_as_unknown_until_an_engine_says_otherwise(
        engine):
    """/status makes NO engine call (it must answer while the engine is wedged), so
    the only honest answer before the first verb payload is `null` - never a
    confident `false` that a guarded engine could hide behind."""
    client = client_for(build_app(FakeAsyncLLM(engine)))
    status = client.get(STATUS).json()
    assert status["guard_mode"] is None
    assert status["trace_armed"] is None
    assert "UNKNOWN" in status["dev_state"]["note"]
    assert "guard_mode=false" in status["dev_state"]["note"]   # the boot default
    assert engine.calls == []


def test_status_repeats_what_the_engine_last_told_this_process(engine):
    engine.result = {"version": 1, "guard_mode": True, "trace_armed": True,
                     "epoch": "boot-77"}
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.get(KVCONFIG).status_code == 200
    status = client.get(STATUS).json()
    assert status["guard_mode"] is True
    assert status["trace_armed"] is True
    assert status["dev_state"]["source"] == "kvconfig read"
    assert status["dev_state"]["epoch"] == "boot-77"
    assert status["dev_state"]["age_seconds"] >= 0


def test_a_verb_that_changes_state_updates_the_mirror_immediately(engine,
                                                                  armed_token):
    """The operator who just armed guard mode must see it on the very next
    `/status`, without the status route having to ask the engine anything."""
    engine.result = {"version": 1, "guard_mode": True, "epoch": "boot-1"}
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(GUARD, json={"armed": True}).status_code == 403
    assert client.get(STATUS).json()["guard_mode"] is None    # a 403 taught it nothing
    response = client.post(GUARD, json={"armed": True}, headers=token_headers())
    assert response.status_code == 200
    status = client.get(STATUS).json()
    assert status["guard_mode"] is True
    assert status["dev_state"]["source"] == "guard verb"
    assert status["dev_state"]["epoch"] == "boot-1"


def test_a_payload_without_the_fields_does_not_invent_state(engine):
    engine.result = {"version": 1}
    client = client_for(build_app(FakeAsyncLLM(engine)))
    client.get(SNAPSHOT)
    assert client.get(STATUS).json()["guard_mode"] is None


def test_status_advertises_the_surface_it_actually_serves(engine):
    status = client_for(build_app(FakeAsyncLLM(engine))).get(STATUS).json()
    prefix = config.ROUTE_PREFIX
    assert status["routes"] == [prefix + route for route in config.ROUTES]
    assert sorted(status["routes"]) == sorted([
        STATUS, SNAPSHOT, KVCONFIG, BLOCK, REQUEST, TRACE, GUARD, TIER, RELOAD])
    assert status["dev_state_routes"] == [prefix + config.ROUTE_TRACE,
                                          prefix + config.ROUTE_GUARD,
                                          prefix + config.ROUTE_TIER]
    assert status["engine_methods"] == list(config.METHODS)
    assert "DEVELOPMENT ONLY" in status["development_facilities"]["guard_mode"]
    assert "BaseException" in status["development_facilities"]["guard_mode"]
    assert "ONE dict argument" in status["arguments"]["POST"]


# --- engine failures stay data ----------------------------------------------

def test_a_new_verb_answers_504_when_the_engine_stops_turning(enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_TIMEOUT, "0.05")
    engine = FakeEngineCoreClient(delay=5)
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(BLOCK, json={"block_id": 1})
    assert response.status_code == 504
    assert "not turning" in response.json()["detail"]


def test_a_new_verb_answers_502_when_the_utility_call_raises(enabled):
    engine = FakeEngineCoreClient(raises=RuntimeError("the engine refused"))
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.get(KVCONFIG)
    assert response.status_code == 502
    assert "kvtier_dev_kvconfig raised RuntimeError" in response.json()["detail"]


def test_an_engine_answer_that_is_not_a_dict_is_wrapped_not_assumed(enabled):
    engine = FakeEngineCoreClient(result=["weird"])
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.get(KVCONFIG).json() == {"result": ["weird"]}


def test_the_validator_is_one_table_for_every_route():
    """`ARG_SPECS` is the whole argument surface; a route missing from it accepts
    nothing, which is the safe direction - but it must be a decision, not an
    accident, so the table names every route the config declares."""
    for route in (config.ROUTE_SNAPSHOT, config.ROUTE_KVCONFIG, config.ROUTE_BLOCK,
                  config.ROUTE_REQUEST, config.ROUTE_TRACE, config.ROUTE_GUARD,
                  config.ROUTE_TIER):
        assert route in endpoint.ARG_SPECS, route
    assert set(endpoint.ARG_SPECS) == {config.ROUTE_SNAPSHOT, config.ROUTE_KVCONFIG,
                                       config.ROUTE_BLOCK, config.ROUTE_REQUEST,
                                       config.ROUTE_TRACE, config.ROUTE_GUARD,
                                       config.ROUTE_TIER}


@pytest.mark.parametrize("route,raw,expected", [
    (config.ROUTE_BLOCK, {"block_id": "412"}, {"block_id": 412}),
    (config.ROUTE_BLOCK, {"block_id": 412}, {"block_id": 412}),
    (config.ROUTE_GUARD, {"armed": "ON"}, {"armed": True}),
    (config.ROUTE_GUARD, {"armed": "off"}, {"armed": False}),
    (config.ROUTE_GUARD, {"armed": False}, {"armed": False}),
    (config.ROUTE_TRACE, {"op": " ARM "}, {"op": "arm"}),
    (config.ROUTE_TRACE, {}, {}),
])
def test_the_validator_normalises_without_inventing_keys(route, raw, expected):
    assert endpoint.validate_arguments(route, raw) == expected


def test_the_validator_never_adds_a_default():
    """Defaults belong to the engine's own handler, so there is ONE place that
    decides what an absent `limit` means."""
    assert endpoint.validate_arguments(config.ROUTE_TRACE, {}) == {}
    assert endpoint.validate_arguments(config.ROUTE_BLOCK, {}) == {}
