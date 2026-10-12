"""Route level: the real FastAPI stack, a fake engine client, no vllm.

THE REACH MODEL, SINCE 0.2.72: `VLLM_E2_TIER_DEV` is the only gate, and it is an
opt-in rather than an access control. With it unset, every route answers 503 to
every peer; with it set, no peer is filtered by address and no header grants
anything. The peer-address check that used to be asserted here (loopback or
`VLLM_E2_TIER_DEV_ALLOW`) is deleted, because a local proxy IS the loopback peer - so
the check fell to the ordinary way this engine is fronted while reporting itself as
working. The tests that used to prove "a stranger is refused" now prove the weaker,
true statement: "the answer does not depend on who you are", driven over the same
peer matrix so the property is measured across the range that used to matter.

Two transports, deliberately.

* TestClient's `client=` argument is what starlette puts into
  `scope["client"]` - the same field uvicorn fills from the accepted socket.
  It is the only way to drive a peer this host cannot source packets from
  (the real proxy at `192.168.2.64`, a TEST-NET address, a spoofed
  `X-Forwarded-For`), so the no-peer-filtering matrix is driven that way.
* The real-uvicorn tests at the bottom (`served_on_real_socket`) accept a real
  TCP connection, which is the only way to prove that nothing about the peer
  uvicorn writes into the scope changes an answer - on loopback AND from one of
  this host's own LAN addresses. A CI box with no such address skips those.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import socket
import threading
import time

import pytest

from kvtier_dev_plugin import config, disk_store, endpoint
from kvtier_dev_plugin.endpoint import TOKEN_HEADER, KVTierDevEndpointPlugin

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")  # noqa: F401 - TestClient's transport

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

STATUS = config.ROUTE_PREFIX + "/status"
SNAPSHOT = config.ROUTE_PREFIX + "/snapshot"
RELOAD = config.ROUTE_PREFIX + "/policy/reload"
KVCONFIG = config.ROUTE_PREFIX + config.ROUTE_KVCONFIG
BLOCK = config.ROUTE_PREFIX + config.ROUTE_BLOCK
REQUEST = config.ROUTE_PREFIX + config.ROUTE_REQUEST
TRACE = config.ROUTE_PREFIX + config.ROUTE_TRACE
GUARD = config.ROUTE_PREFIX + config.ROUTE_GUARD
TIER = config.ROUTE_PREFIX + config.ROUTE_TIER
LOOKUPS = config.ROUTE_PREFIX + config.ROUTE_LOOKUPS
BLOCKDUMP = config.ROUTE_PREFIX + config.ROUTE_BLOCKDUMP
DISK = config.ROUTE_PREFIX + config.ROUTE_DISK

# (path, http method) for EVERY route this plugin serves. The guard tests are
# parametrized over it, so a new route is automatically tested for the peer rule
# and the inert rule - and `test_PLUGIN_ROUTES_is_the_whole_surface` fails if this
# list and the router ever disagree. Since 0.2.68 the token rule is gone: the same
# parametrization proves no route on the surface asks for a token.
PLUGIN_ROUTES = [(STATUS, "get")] + [
    (config.ROUTE_PREFIX + route, verb)
    for route in (config.ROUTE_SNAPSHOT, config.ROUTE_KVCONFIG, config.ROUTE_BLOCK,
                  config.ROUTE_REQUEST, config.ROUTE_TRACE, config.ROUTE_GUARD,
                  config.ROUTE_TIER, config.ROUTE_LOOKUPS, config.ROUTE_BLOCKDUMP,
                  config.ROUTE_DISK)
    for verb in ("get", "post")
] + [(RELOAD, "post")]


class FakeEngineCoreClient:
    """Stands in for the EngineCoreClient that actually owns call_utility_async."""

    def __init__(self, result=None, raises=None, delay=None) -> None:
        self.result = result or {"version": 1, "tier": {"present": True}}
        self.raises = raises
        self.delay = delay
        self.calls: list[tuple[str, tuple]] = []

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if self.delay is not None:
            await asyncio.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        return self.result


class FakeAsyncLLM:
    """Stands in for AsyncLLM: the utility channel lives one level down
    (vllm/v1/engine/async_llm.py:169 -> self.engine_core)."""

    def __init__(self, engine_core) -> None:
        self.engine_core = engine_core


def build_app(engine_client=None):
    app = FastAPI()
    plugin = KVTierDevEndpointPlugin()
    plugin.attach_router(app)
    asyncio.run(plugin.init_state(engine_client, app.state, argparse.Namespace()))
    return app


def served_paths(app) -> list[str]:
    """Every dispatchable path, flattened.

    The fork pins `fastapi >= 0.133.0, < 0.137.0` (requirements/common.txt:15)
    and that route tree flattens an included router into `app.routes`; newer
    FastAPI keeps an `_IncludedRouter` wrapper with the routes nested inside.
    Handle both so this test checks the mount point, not the FastAPI version.
    """
    paths: list[str] = []
    stack = list(app.routes)
    while stack:
        route = stack.pop()
        path = getattr(route, "path", None)
        if path:
            paths.append(path)
        nested = getattr(route, "routes", None)
        if nested is None:
            nested = getattr(getattr(route, "original_router", None), "routes", None)
        if nested:
            stack.extend(nested)
    return sorted(paths)


def client_for(app, peer=("127.0.0.1", 43210)):
    return TestClient(app, client=peer)


def token_headers(token="tk"):
    """Headers carrying the header this plugin USED to require.

    Kept on purpose after 0.2.68 deleted the gate: the tests that still send it
    (several files) now prove the opposite property - a caller who has not heard
    the news, or an operator who has not unset `VLLM_E2_TIER_DEV_TOKEN`, keeps
    working. Nothing in the product reads either the header or the variable.
    """
    return {"x-kvtier-dev-token": token}


# The peers the reach matrix is driven over: loopback in all its spellings, the two
# addresses the production box actually names in its unit file, an address on another
# network entirely, and the shapes the removed guard used to refuse as "unprovable".
# Under the old gate these divided the surface into served and refused; the whole
# point of the new contract is that they do not divide it at all.
REACH_PEERS = [
    ("127.0.0.1", 43210),
    ("127.2.3.4", 1),
    ("::1", 1),
    ("::ffff:127.0.0.1", 1),
    ("10.0.0.7", 1),
    ("192.168.2.42", 8000),
    ("203.0.113.9", 51234),
    ("0.0.0.0", 1),
    ("testclient", 50000),      # what a test harness reports: not an IP
    ("127.0.0.1:1234", 1),      # not a parseable IP either
]


# --- real-socket machinery (see the module docstring) -----------------------

def _free_port(host) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def _non_loopback_ipv4():
    """A real, bindable, non-loopback IPv4 address of THIS host, else None.

    The UDP `connect()` sends nothing: it asks the routing table which source
    address a packet to that destination WOULD use. So this is a local lookup -
    no traffic, no peer required to be up (192.0.2.0/24 is TEST-NET-1).
    """
    candidates: set[str] = set()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 53))
        candidates.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.add(info[4][0])
    except OSError:
        pass
    for address in sorted(candidates):
        if ipaddress.ip_address(address).is_loopback:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bindable:
                bindable.bind((address, 0))
        except OSError:
            continue
        return address
    return None


LAN_PEER = _non_loopback_ipv4()
needs_lan = pytest.mark.skipif(
    LAN_PEER is None,
    reason="no bindable non-loopback IPv4 address on this host")


@contextlib.contextmanager
def served_on_real_socket(app, host):
    """uvicorn in a thread on a real socket; yields the base URL.

    Bound to ONE explicit address (never 0.0.0.0) on an ephemeral port, and
    `lifespan="off"` because these apps have none. The only routes it can ever
    answer are the plugin's three, behind the same guards.
    """
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_port(host)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port,
                                           log_level="critical",
                                           access_log=False, lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15.0
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            pytest.fail(f"uvicorn never started on {host}:{port}")
        time.sleep(0.01)
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15.0)


def real_client(base_url, source=None):
    """An HTTP client, optionally bound to a specific SOURCE address, so the
    peer the server sees is chosen here rather than asserted afterwards."""
    transport = (httpx.HTTPTransport(local_address=source) if source
                 else httpx.HTTPTransport())
    return httpx.Client(transport=transport, base_url=base_url, trust_env=False)


# --- mount point ------------------------------------------------------------

def test_routes_live_under_the_plugin_prefix_and_nowhere_else():
    app = build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    # served_paths() is per ROUTE (path + method), so a path registered for GET
    # and POST appears twice; the mount points are what this test is about.
    served = sorted(set(served_paths(app)))
    assert [path for path in served
            if path.startswith(config.ROUTE_PREFIX)] == sorted(
        [config.ROUTE_PREFIX + route for route in config.ROUTES])
    # A plugin route must not shadow /v1: --api-key's coverage of /v1 is the
    # only reason that prefix is interesting to an attacker (security.md:349).
    assert [path for path in served if path.startswith("/v1")] == []
    # Nothing else was added either: only FastAPI's own documentation routes.
    assert sorted(set(served) - {config.ROUTE_PREFIX + route
                                 for route in config.ROUTES}) == [
        "/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"]
    assert KVTierDevEndpointPlugin.name == config.PLUGIN_NAME
    assert KVTierDevEndpointPlugin.required_tasks is None


def test_the_openapi_surface_matches_the_router():
    """The cheap operator proof (`curl .../openapi.json`) must show exactly the
    routes this plugin claims - that is what README.md tells you to check on the
    box, and the claim comes from config.ROUTES so the two cannot drift."""
    app = build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    assert sorted(app.openapi()["paths"]) == sorted(
        [config.ROUTE_PREFIX + route for route in config.ROUTES])


def test_PLUGIN_ROUTES_is_the_whole_surface_and_cannot_rot():
    """The guard tests are parametrized over PLUGIN_ROUTES, so that list must be
    the router's surface exactly - a route added to the router and forgotten in
    the list would be an UNTESTED route, which is the failure this pins."""
    app = build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    registered = {(path, method)                       # openapi uses lower case
                  for path, item in app.openapi()["paths"].items()
                  for method in item}
    assert set(PLUGIN_ROUTES) == registered
    # And every one of them is under the prefix, never under /v1.
    assert all(path.startswith(config.ROUTE_PREFIX) for path, _ in PLUGIN_ROUTES)


def test_every_route_is_registered_for_the_methods_its_doc_promise_needs():
    """GET where a browser should reach it, POST where a body is required."""
    app = build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    methods = {}
    for path, item in app.openapi()["paths"].items():
        methods[path] = sorted(item)
    prefix = config.ROUTE_PREFIX
    assert methods[f"{prefix}{config.ROUTE_STATUS}"] == ["get"]
    assert methods[f"{prefix}{config.ROUTE_RELOAD}"] == ["post"]
    for route in (config.ROUTE_SNAPSHOT, config.ROUTE_KVCONFIG,
                  config.ROUTE_BLOCK, config.ROUTE_REQUEST):
        assert methods[f"{prefix}{route}"] == ["get", "post"], route
    for route in config.DEV_STATE_ROUTES:
        assert methods[f"{prefix}{route}"] == ["get", "post"], route


def test_the_guard_and_trace_verbs_advertise_themselves_as_development_facilities():
    """The OpenAPI text is the endpoint's own description, and it is the place an
    operator reads before arming something. Guard mode must be labelled there."""
    app = build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    paths = app.openapi()["paths"]
    guard = paths[f"{config.ROUTE_PREFIX}{config.ROUTE_GUARD}"]["post"]
    assert "DEVELOPMENT ONLY" in guard["description"]
    assert "BaseException" in guard["description"]
    trace = paths[f"{config.ROUTE_PREFIX}{config.ROUTE_TRACE}"]["post"]
    assert "arm" in trace["description"]
    # The arity contract - one body, one dict argument, uniform engine arity -
    # is documented where an operator reads it: the module docstring, which is
    # also the file's own security notes.
    doc = endpoint.__doc__ or ""
    assert "ONE positional argument" in doc
    assert "GUARD MODE IS A DEVELOPMENT FACILITY" in doc


# --- the reach model: the switch answers for every peer ---------------------
#
# WHAT THIS SECTION REPLACED. Until 0.2.72 it asserted a split: loopback (or an
# address in `VLLM_E2_TIER_DEV_ALLOW`) was served, everything else got 403 on peer
# grounds. That gate is deleted, and the reason is in the module docstring: a local
# proxy IS the loopback peer, so the check passed for exactly the deployment that is
# exposed. What is asserted here instead is the true and weaker property - the answer
# does not depend on the peer - over the SAME peer matrix, so the range that used to
# matter is still exercised. The peer addresses below are TestClient-constructed on
# purpose: `192.168.2.64` is the production proxy and no test box can legitimately
# source packets from it. The real-socket section covers the peers we CAN source from.

PROXY = ("192.168.2.64", 51000)
WORKSTATION = ("192.168.2.120", 51001)


@pytest.mark.parametrize("peer", REACH_PEERS)
def test_every_peer_is_served_when_the_switch_is_on(enabled, peer):
    """No address is admitted and no address is refused, over the whole matrix -
    including the two shapes the removed guard refused as UNPROVABLE (`testclient`,
    `127.0.0.1:1234`): with no peer question to fail, an unparsable address is just
    another client.
    """
    app = build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    response = client_for(app, peer).get(STATUS)
    assert response.status_code == 200, (peer, response.text)


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
@pytest.mark.parametrize("peer", [("127.0.0.1", 1), PROXY, ("203.0.113.9", 1),
                                  ("testclient", 50000)])
def test_no_peer_is_filtered_on_any_route(enabled, method, path, peer):
    """THE rule, over the whole surface: every route answers every peer.

    This includes every member of `config.DEV_STATE_ROUTES` (trace, guard, tier,
    lookups, blockdump, disk) - the routes that used to demand a token in every shape
    and then a permitted address. They are still listed as dev-state routes (they can
    move engine state); they simply gate on nothing but the switch, which is the
    honest arrangement now that nothing pretends to identify the caller.
    """
    engine = FakeEngineCoreClient(result={"version": 1})
    response = client_for(build_app(FakeAsyncLLM(engine)), peer).request(method,
                                                                        path)
    assert response.status_code == 200, (peer, method, path, response.text)


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_inert_is_503_for_every_peer_on_every_route(method, path, monkeypatch):
    """The other half of one gate: with the switch off, an address buys nothing and
    the dead knobs buy nothing, on every route, for every peer.

    This is the test that used to be two - "an allowlisted peer still learns the
    plugin is inert" and "a disallowed peer is 403 even when inert, so an inert
    plugin is not a way to enumerate configuration". With one gate there is one
    answer, and the 503 is the whole of it: the plugin publishes its own
    configuration to anyone who can reach the port the moment it is enabled, and
    nothing when it is not.
    """
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    monkeypatch.setenv(config.ENV_TOKEN, "tk")          # the dead knob, set
    monkeypatch.setenv(config.ENV_ALLOW_RETIRED, "0.0.0.0/0")   # the other one
    for peer in (("127.0.0.1", 1), PROXY, ("10.9.8.7", 1)):
        response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                              peer).request(method, path,
                                            headers=token_headers("tk"))
        assert response.status_code == 503, (peer, method, path, response.text)
        assert config.ENV_ENABLE in response.json()["detail"]


def test_the_inert_answer_tells_the_operator_what_enabling_opens(monkeypatch):
    """The 503 is read at the moment the surface is OFF, which makes it the right
    place to say what turning it on costs: open to every client that can reach the
    port, not covered by vLLM's `--api-key`, and the protection is network-level and
    the operator's job.
    """
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    detail = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))
                        ).get(STATUS).json()["detail"]
    assert config.ENV_ENABLE in detail and config.ROUTE_PREFIX in detail
    assert "every client" in detail.lower()
    assert "--api-key" in detail
    assert "operator" in detail.lower()


def test_a_remote_peer_gets_the_same_answer_as_loopback(enabled):
    """The inversion of the old "no configuration leak to a stranger" test.

    There is no stranger, so there is no redaction to assert: `/status` answers the
    same payload - `policy_target`, `peer_gate`, `guard`, all of it - to the LAN
    address it used to refuse. What this DOES pin is that the payload states the
    exposure out loud, because that payload is now the only warning a reader gets
    before they start using the surface.
    """
    loopback = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))
                          ).get(STATUS)
    remote = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                        ("10.1.2.3", 1)).get(STATUS)
    assert loopback.status_code == remote.status_code == 200
    body = remote.json()
    assert body["peer_gate"]["peers_filtered_by_address"] is False
    assert body["peer_gate"]["removed"] is True
    assert body["vllm_api_key_covers_these_routes"] is False
    assert body["policy_target"]["path"]


def test_a_forged_forwarded_header_grants_nothing_and_refuses_nothing(enabled):
    """The attack the old guard existed to stop, restated against the new gate.

    `X-Forwarded-For: 127.0.0.1` from a remote peer used to be refused because the
    guard read the socket and not the header. Now there is no peer decision to make,
    so the interesting half is that the header still does NOTHING - it does not open
    a route that would otherwise be closed, and it does not close one that is open.
    Nothing on this surface reads a header: the structural half of that is pinned in
    `tests/test_guards.py` by AST.
    """
    engine = FakeEngineCoreClient(result={"version": 1})
    client = client_for(build_app(FakeAsyncLLM(engine)), ("203.0.113.9", 51234))
    for headers in ({}, {"x-forwarded-for": "127.0.0.1"},
                    {"x-forwarded-for": "127.0.0.1, 127.0.0.1"},
                    {"x-real-ip": "127.0.0.1"},
                    {**token_headers("tk"), "x-forwarded-for": "127.0.0.1"}):
        assert client.get(STATUS, headers=headers).status_code == 200, headers
        assert client.get(SNAPSHOT, headers=headers).status_code == 200, headers
        assert client.post(RELOAD, headers=headers).status_code == 200, headers
    assert [name for name, _ in engine.calls] == [
        config.METHOD_SNAPSHOT, config.METHOD_RELOAD_POLICY] * 5


def test_the_operator_at_the_box_still_needs_no_header_and_no_address(enabled):
    """The `ssh llm curl 127.0.0.1` reflex is unchanged, and a header naming some
    OTHER address does not change it either - there is nothing for it to change."""
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("127.0.0.1", 1))
    assert client.get(STATUS, headers={"x-forwarded-for": "203.0.113.9"}
                      ).status_code == 200


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_the_retired_allowlist_changes_no_answer_on_any_route(enabled, monkeypatch,
                                                              method, path):
    """The variable that used to decide this is now inert on every route.

    Four values of `VLLM_E2_TIER_DEV_ALLOW` - the production pair, `0.0.0.0/0`,
    garbage, and unset - and the answer is the same request the peer would get with
    the variable deleted from its environment. This is the retirement, asserted
    where it would otherwise be assumed: the live unit file still exports the name,
    and every README before 0.2.72 told operators to set it.
    """
    for allow in ("192.168.2.64,192.168.2.120", "0.0.0.0/0", "garbage", None):
        if allow is None:
            monkeypatch.delenv(config.ENV_ALLOW_RETIRED, raising=False)
        else:
            monkeypatch.setenv(config.ENV_ALLOW_RETIRED, allow)
        engine = FakeEngineCoreClient(result={"version": 1})
        # A peer the old parser would have refused, and the proxy it would have
        # admitted: both are served, and the engine hears the same one call.
        for peer in (PROXY, ("10.9.8.7", 1)):
            response = client_for(build_app(FakeAsyncLLM(engine)),
                                  peer).request(method, path)
            assert response.status_code == 200, (allow, peer, method, path,
                                                 response.text)


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_a_leftover_token_header_changes_no_answer_on_any_route(enabled,
                                                                monkeypatch,
                                                                method, path):
    """An operator's old script must not start failing, and must not start working
    either: the header is inert on every route, and an unset token env is inert the
    same way (both are tested by the same pair of requests)."""
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)   # nothing configured...
    client = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())), PROXY)
    plain = client.request(method, path)
    stale = client.request(method, path, headers=token_headers("not-the-token"))
    assert plain.status_code == 200, (method, path, plain.text)
    assert stale.status_code == plain.status_code, (method, path, stale.text)
    if path == STATUS:
        # The only volatile half of this payload is the dev-state mirror's age.
        assert {k: v for k, v in stale.json().items()
                if k != "dev_state"} == {k: v for k, v in plain.json().items()
                                         if k != "dev_state"}
    else:
        assert stale.json() == plain.json()


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_a_stranger_holding_the_dead_token_is_served_like_anyone_else(
        enabled, monkeypatch, method, path):
    """What the removal actually means, stated at the route that used to prove the
    opposite: a caller on a network nobody trusts, holding the old token in its env
    and its header, is served - because nothing identifies anybody any more.

    Until 0.2.72 this request got 403 with a detail naming the peer rule, and that
    403 was the file's most-repeated assertion. It is gone, and the test that
    replaces it must fail the day a peer check comes back: that is the only way this
    contract stays true in the product rather than in the README.
    """
    monkeypatch.setenv(config.ENV_TOKEN, "tk")          # the dead knob, set
    monkeypatch.setenv(config.ENV_ALLOW_RETIRED, "192.168.2.64/32")  # admits nobody
    engine = FakeEngineCoreClient(result={"version": 1})
    response = client_for(build_app(FakeAsyncLLM(engine)),
                          ("10.9.8.7", 1)).request(method, path,
                                                   headers=token_headers("tk"))
    assert response.status_code == 200, (method, path, response.text)
    # `/status` is answered from this process and makes NO engine call by design, so
    # "did the route run?" is only observable on the routes that forward.
    if path != STATUS:
        assert engine.calls, (method, path, "the route answered without the engine")


def test_a_malformed_allow_env_is_not_an_error_because_nothing_parses_it(
        enabled, monkeypatch):
    """Garbage in the retired variable used to be a careful non-event: parsed,
    skipped, warned about, refused nobody. Now it is not even parsed, so the only
    honest assertion is that it cannot be seen at all - no 500, no warning line, no
    change in the answer.

    This replaces `test_malformed_allow_env_is_403_not_500` and
    `test_cidr_entry_admits_inside_and_refuses_outside` and
    `test_unset_allow_refuses_an_allowlisted_looking_peer` and
    `test_a_good_entry_survives_a_bad_neighbour_on_the_route`: four tests that
    measured a parser, for a parser that no longer exists. Keeping any of them would
    have meant keeping the parser.
    """
    for env in ("garbage", "", "   ", "10.0.0.0/99", "192.168.2.999", "localhost",
                "::x", "0.0.0.0/0,192.168.2.64"):
        monkeypatch.setenv(config.ENV_ALLOW_RETIRED, env)
        engine = FakeEngineCoreClient(result={"version": 1})
        client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
        for method, path in (("get", STATUS), ("post", RELOAD)):
            response = client.request(method, path)
            assert response.status_code == 200, (env, method, response.text)


@pytest.mark.parametrize("route", list(config.DEV_STATE_ROUTES))
@pytest.mark.parametrize("peer", [("127.0.0.1", 1), PROXY, ("203.0.113.9", 1)])
def test_the_dev_state_routes_answer_every_peer_with_no_token(enabled, monkeypatch,
                                                              route, peer):
    """Every member of DEV_STATE_ROUTES, GET and POST, answers a peer that sends
    nothing but the URL - and it does so from loopback AND from the far side of the
    address space, which is the price of the same rule.

    Until 0.2.68 these six routes required the token in EVERY shape - loopback
    included, `GET /trace` and `POST /disk {"op":"status"}` alike - which is what
    made the surface unusable from a script or from the proxy. `DEV_STATE_ROUTES`
    is read from `config` here, so a seventh route is covered the day it is added.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    engine = FakeEngineCoreClient(result={"version": 1})
    client = client_for(build_app(FakeAsyncLLM(engine)), peer)
    path = config.ROUTE_PREFIX + route
    assert client.get(path).status_code == 200, (peer, route, "GET")
    assert client.post(path, json={}).status_code == 200, (peer, route, "POST")


def test_the_mutating_verb_needs_no_token_and_no_particular_peer(enabled,
                                                                 monkeypatch):
    """`POST /policy/reload` used to be the ONE mutating verb and the ONE thing the
    token was documented to gate. It needs neither a token nor an address - and what
    keeps it safe is not identity: the engine-side method takes no argument, so the
    wire never selects a file, and the path validation here accepts exactly one."""
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    engine = FakeEngineCoreClient(result={"ok": True})
    for peer in (("127.0.0.1", 1), WORKSTATION, ("203.0.113.9", 1)):
        assert client_for(build_app(FakeAsyncLLM(engine)), peer).post(
            RELOAD).status_code == 200, peer
    assert engine.calls == [(config.METHOD_RELOAD_POLICY, ())] * 3


def test_the_surface_is_reachable_from_another_network_and_says_so(enabled):
    """THE consequence, asserted rather than implied: a peer on a network the box
    does not trust reads the whole `/status` payload, and the payload names that as
    the operator's exposure rather than claiming a gate it does not have.

    This is the test that used to be `test_a_wide_allowlist_reaches_the_surface…`,
    where `0.0.0.0/0` in an env var was the thing that opened the surface. Nothing
    opens it now except enabling it, so the reach is unconditional and the only
    remaining honest artefact is the report.
    """
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(STATUS).json()
    assert body["enabled"] is True
    assert body["peer_gate"]["removed"] is True
    assert "network" in body["peer_gate"]["operators_job"]
    assert body["vllm_api_key_covers_these_routes"] is False


# --- THE SSD TIER'S FILL BLOCK, read from off the box (0.2.30) ---------------
#
# WHY THIS IS A ROUTE TEST AND NOT ONLY A BLOCK TEST. `disk_fill` exists because the
# proxy on `192.168.2.64` is going to poll it every few seconds, and the live box's
# allowlist is now `192.168.2.64,192.168.2.120`. A block that is correct in a unit test
# and unreachable from the one peer that will read it is not a feature. So the whole
# peer matrix runs against THIS block: an allowlisted peer reads it with no header at
# all, a peer off the list does not read it even holding the old token, loopback needs
# nothing, and a spoofed `X-Forwarded-For` buys nothing.

FILL_CEILING = 148 * 1024 ** 3
FILL_USED = 156 * 1024 ** 3     # over the ceiling, as the store was on 2026-10-22


def fill_rank_file(directory, rank="PP0_TP0", *, used=FILL_USED,
                   ceiling=FILL_CEILING, age=0.0, evictions=2106):
    """One rank file, as `worker_diag` writes it, carrying the 0.2.30 fill fields."""
    document = {"version": 1, "epoch": "aabbccdd",
                "ts": round(time.time() - age, 6), "role": "worker", "rank": rank,
                "error": None,
                "disk": {"installed": True, "armed": True,
                         "ceiling_bytes": ceiling,
                         "ceiling_share_bytes": ceiling // 4,
                         "ceiling_divisor": 4, "ceiling_passes_declined": 0,
                         "estimated_bytes": used,
                         "ceiling_measurement_basis": "full_measurement",
                         "files_written": 92, "bytes_written": 92 * 13_142_016,
                         "evictions": evictions, "journal_removals": evictions,
                         "drops": 0, "queue_depth": 0,
                         "writer_thread_alive": True},
                "disk_restore": {"readers": {"recency_promotions": 4100,
                                             "recency_promotions_throttled": 585}}}
    (directory / f"rank-{rank}.json").write_text(json.dumps(document),
                                                 encoding="utf-8")


@pytest.fixture
def fill_store(tmp_path, monkeypatch):
    """A diagnostics directory with rank files in it, and no leaked rollup cache."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    disk_store.reset_rank_rollup_cache_for_tests()
    yield directory
    disk_store.reset_rank_rollup_cache_for_tests()


def test_the_proxy_peer_reads_the_fill_block_with_no_token(enabled, monkeypatch,
                                                           fill_store):
    """THE DEPLOYMENT CASE: the proxy on `192.168.2.64` polls `/status`, nothing is
    configured in `VLLM_E2_TIER_DEV_TOKEN`, and the answer must be the block - not a
    redacted status payload, and not an engine call: `/status` makes none, and the
    fill comes from the rank files the workers write on this host. This is the case
    that made the token worth removing: a poller that has to carry a secret for a
    read it was let through for is a poller that gets dropped.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    response = client.get(STATUS)
    assert response.status_code == 200, response.text
    fill = response.json()["disk_fill"]
    assert fill["available"] is True, fill["reason"]
    assert fill["used_bytes"] == FILL_USED
    assert fill["ceiling_bytes"] == FILL_CEILING
    assert fill["target_bytes"] == int(FILL_CEILING * 0.9)
    assert fill["over_ceiling"] is True and fill["over_target"] is True
    assert fill["basis"] == "full_measurement", fill["basis"]
    assert fill["per_rank"][0]["journal_removals"] == 2106
    assert fill["per_rank"][0]["recency_promotions"] == 4100
    assert fill["per_rank"][0]["file_age_seconds"] < 60
    assert fill["per_process_note"], "the label must travel INSIDE the block"
    # ... and the block did not make `/status` talk to the engine.
    assert engine.calls == []


def test_the_workstation_peer_reads_the_same_block(enabled, monkeypatch, fill_store):
    """Two peers, one dialect: the block is not peer-specific - there is no peer
    question left to make it specific."""
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(STATUS).json()
    assert body["disk_fill"]["used_bytes"] == FILL_USED


def test_the_fill_block_is_served_to_every_peer_when_enabled(enabled, monkeypatch,
                                                             fill_store):
    """The block that exists so an off-box poller can read it is served to every
    peer - the proxy, the workstation, and an address nobody recognised.

    This replaces two tests that asserted the opposite ("not served to a peer off the
    allowlist", "a peer off the allowlist does not get the fill block"). Their real
    point - that the answer is not tailored per peer - survives; only its direction
    flipped, because the thing that used to tailor it is gone. What no peer gets is a
    DIFFERENT payload: every caller reads the same body, which is the property worth
    keeping now that "who" is not a question.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    answers = set()
    for peer in (PROXY, WORKSTATION, ("10.9.8.7", 1)):
        response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                              peer).get(STATUS, headers=token_headers("tk"))
        assert response.status_code == 200, (peer, response.text)
        assert response.json()["disk_fill"]["used_bytes"] == FILL_USED, peer
        answers.add(response.text)
    assert len(answers) == 1, "the payload must not vary by peer"


def test_loopback_reads_the_fill_block_with_no_token_at_all(enabled, monkeypatch,
                                                            fill_store):
    """The operator at the box: `/status` on loopback needs nothing, which is what
    makes a plain curl a usable reflex. The same curl from another host gets the same
    answer - that is the cost side of this test, kept here so the pair is readable."""
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert body["disk_fill"]["used_bytes"] == FILL_USED
    assert body["disk_fill"]["ranks_reported"] == 1


def test_a_spoofed_forwarded_for_unlocks_the_fill_block_for_nobody(enabled,
                                                                   monkeypatch,
                                                                   fill_store):
    """The same attack as the spoof test above, aimed at the new block.

    `X-Forwarded-For: 127.0.0.1` from a remote host used to be the interesting case
    because a guard existed to defeat it. There is no guard to defeat, so what this
    pins is that the header is not a substitute for one: it neither opens the block
    (nothing needs opening) nor closes it, and the answer is byte-identical with and
    without the forgery and with and without the dead token header.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("203.0.113.9", 51234))
    answers = set()
    for response in (
        client.get(STATUS),
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1"}),
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1",
                                   **token_headers("tk")}),
        client.get(STATUS, headers={"x-real-ip": "192.168.2.64",
                                   **token_headers("tk")}),
    ):
        assert response.status_code == 200, response.text
        assert response.json()["disk_fill"]["used_bytes"] == FILL_USED
        answers.add(response.text)
    assert len(answers) == 1, "the fill answer must not depend on any header"


def test_the_snapshot_route_carries_the_fill_block_to_an_off_box_peer(enabled,
                                                                      monkeypatch,
                                                                      fill_store):
    """The engine-side view reaches the proxy too.

    `/snapshot` is the ENGINE's payload (the utility call), and `dev_state_sections()` -
    where `disk_fill` is built - is embedded at its top level. The fake engine here
    returns what the real handler would embed, so this proves the ROUTE carries the
    block to an off-box peer; building it is `test_disk_fill_block.py`.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    engine = FakeEngineCoreClient(result={"version": 1,
                                          "disk_fill": disk_store.disk_fill()})
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    response = client.get(SNAPSHOT)
    assert response.status_code == 200, response.text
    fill = response.json()["disk_fill"]
    assert fill["used_bytes"] == FILL_USED
    # The rank file above claims a 4-way share, which is what a rank on the live
    # 4-rank boot publishes: the verdict reads that, not the number of files on disk.
    assert fill["cross_process_verdict"]["verdict"] == "shared_share", fill
    assert fill["cross_process_verdict"]["n_effective"] == 4, fill
    assert engine.calls == [(config.METHOD_SNAPSHOT, ())]


def test_the_fill_block_is_bounded_over_the_wire(enabled, monkeypatch, fill_store):
    """A dashboard polls this every few seconds, so its SIZE is the contract.

    Twenty-four rank files, one capped entry list, and a response that stays well under
    the verb's own ceiling: rank entries only, never an inventory.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    for pp in range(6):
        for tp in range(4):
            fill_rank_file(fill_store, rank=f"PP{pp}_TP{tp}", age=float(pp))
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      PROXY).get(STATUS).json()
    fill = body["disk_fill"]
    assert fill["ranks_reported"] == 24, fill["ranks_reported"]
    assert len(fill["per_rank"]) == disk_store.DISK_FILL_RANK_CAP
    assert fill["per_rank_truncated"] is True
    assert len(json.dumps(fill)) < 64 * 1024, len(json.dumps(fill))


def test_no_rank_files_reports_a_null_fill_rather_than_an_empty_tier(enabled,
                                                                    monkeypatch):
    """An unmeasured store is `null`, over the wire too.

    `used_bytes: 0` would render as "the tier is empty" on a dashboard at exactly the
    moment the workers have not written a rank file yet - the most misleading possible
    reading of a cold boot.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      PROXY).get(STATUS).json()
    fill = body["disk_fill"]
    assert fill["available"] is False
    assert fill["used_bytes"] is None
    assert fill["basis"] == "no_rank_files", fill["basis"]
    assert fill["used_fraction"] is None


def test_an_inert_plugin_learns_the_fill_block_never_ran(enabled, monkeypatch,
                                                         fill_store):
    """enabled is the FIRST thing the route does, so an inert plugin reads no rank
    files and publishes no block - for any peer, including the one that used to be
    the only one allowed to ask."""
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    fill_rank_file(fill_store)
    for peer in (("127.0.0.1", 1), PROXY, ("10.9.8.7", 1)):
        response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                              peer).get(STATUS)
        assert response.status_code == 503, (peer, response.text)
        assert "disk_fill" not in response.text


# --- what status reports about the new state ----------------------------------

OLD_STATUS_KEYS = {
    "plugin", "enabled", "enable_env", "route_prefix", "engine_methods",
    "token_configured", "token_env", "timeout_seconds", "timeout_env",
    "policy_target",
}


def test_status_keeps_every_existing_key(enabled):
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert OLD_STATUS_KEYS.issubset(body)
    assert body["token_env"] == config.ENV_TOKEN


def test_status_reports_the_retired_allowlist_as_a_dead_knob(enabled, monkeypatch):
    """The three keys that answered "who can reach this?" cannot survive the gate
    they described, and they are handled two different ways - which is the part this
    test exists to pin.

    `allow_env` keeps its LITERAL meaning ("which variable", now the retired name)
    and gains a companion that says whether it does anything, because a variable the
    live unit file still exports must stay reported - that is how a dead knob gets
    found instead of assumed. `allow_networks` and `loopback_only` are GONE: they
    could only ever describe a parser that no longer exists, and answering them
    truthfully (`[]`, `False`) would read as "an empty allowlist" - a lie that looks
    like a fact. The one clear statement lives in `peer_gate`.
    """
    monkeypatch.setenv(config.ENV_ALLOW_RETIRED, "192.168.2.0/24")
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert "allow_networks" not in body
    assert "loopback_only" not in body
    assert body["allow_env"] == config.ENV_ALLOW_RETIRED
    assert body["allow_env_gates"] is False
    assert body["allow_list_configured"] is True      # set, and read for nothing
    # 0.2.68 retired the token, 0.2.72 the address: NO peer - loopback or not - is
    # asked for a token or filtered by where it came from. The keys STAYS with its
    # value flipped rather than being deleted: something reading it must find a
    # truthful `false`, not a KeyError that looks like the payload changed shape.
    assert body["non_loopback_peers_require_token"] is False
    assert body["token_gating"] is False
    assert config.ROUTE_PREFIX in body["guard"]
    assert "the ONLY gate" in body["guard"]
    assert "every client" in body["guard"]


def test_status_says_the_token_no_longer_means_what_it_used_to(enabled, monkeypatch):
    """THE dated, explicit statement this project prefers over a removed key.

    `token_configured`/`token_env` keep their literal meaning ("is the variable
    set", "which variable"), because a key vanishing from `/status` is a silent
    change for anything that reads it. Beside them: `token_gating` is `false`, the
    note names the release that stopped reading the variable, and the note and
    `guard` both say the thing a future reader most needs - there is no gate here but
    the enable switch, and vLLM's `--api-key` does not cover `/plugins/*`.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert body["token_configured"] is False
    assert body["token_env"] == config.ENV_TOKEN
    assert body["token_gating"] is False
    note = body["token_note"]
    assert config.ENV_TOKEN in note and "no longer" in note
    assert "0.2.68" in note
    assert "0.2.72" in note, "the note must not stop at the first retirement"
    assert "token" not in body["guard"].lower() or "no token" in body["guard"].lower()
    assert body["vllm_api_key_covers_these_routes"] is False
    assert "--api-key" in body["guard"]
    # The header name an operator's script still sends is named as the ignored
    # thing it is, rather than left to be guessed at from a 403 that never comes.
    assert body["token_header"] == TOKEN_HEADER


def test_setting_the_token_env_changes_the_report_and_nothing_else(
        enabled, monkeypatch):
    """The report-only property, measured as a DIFF of two payloads.

    The same request with the variable unset, then set: the ONLY key that moves is
    `token_configured`. If a future field starts depending on the token - or the
    note starts lying about which way the variable was set - this fails and names
    the key that moved.
    """
    client = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())))
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    off = client.get(STATUS).json()
    monkeypatch.setenv(config.ENV_TOKEN, "s3cr3t-token")
    on = client.get(STATUS).json()
    moved = {key for key in set(off) | set(on) if off.get(key) != on.get(key)}
    assert moved == {"token_configured"}, sorted(moved)
    assert on["token_configured"] is True and off["token_configured"] is False
    assert on["token_gating"] is False and off["token_gating"] is False


def test_status_reports_the_exposure_instead_of_the_parsed_allowlist(enabled,
                                                                    monkeypatch):
    """An operator must be able to see what is open. What changed in 0.2.72 is that
    the answer stopped being a list: there is no parsed set of networks to show, so
    `/status` says the rule instead - enabled means reachable by whoever can route to
    the port - and reports the retired variable as the dead knob it is.

    The two tests this replaces asserted `allow_networks == [...]` and that a
    malformed entry was "absent". Both were statements about a parser; the parser is
    gone, and a `[]` here would read as "nothing is allowlisted" rather than as
    "nobody is filtered", which is the kind of false-comfort key this project names
    incidents for. So the keys are deleted, and the test asserts their absence.
    """
    monkeypatch.setenv(config.ENV_ALLOW_RETIRED, "192.168.2.64, 192.168.2.0/24, bogus")
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(STATUS).json()
    assert "allow_networks" not in body and "loopback_only" not in body
    assert body["allow_list_configured"] is True, "a set knob must still be visible"
    assert body["allow_env_gates"] is False
    assert body["peer_gate"]["peers_filtered_by_address"] is False
    assert body["non_loopback_peers_require_token"] is False


# --- the one gate runs first, and nothing runs before it ---------------------

def test_inert_answers_503_to_the_peer_that_used_to_be_the_only_allowed_one(monkeypatch):
    """enabled is the first thing a route does, and the only thing: the production
    proxy - the peer the old guard existed to admit - learns the plugin is inert
    rather than getting a 200, and the dead token variable changes nothing."""
    monkeypatch.setenv(config.ENV_ALLOW_RETIRED, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          PROXY).get(STATUS)
    assert response.status_code == 503
    assert config.ENV_ENABLE in response.json()["detail"]


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_an_inert_plugin_answers_the_same_503_to_every_peer(monkeypatch, method,
                                                            path):
    """The inversion of "a disallowed peer learns nothing even when inert".

    That test existed because the peer check ran BEFORE the 503, so an inert plugin
    was not a way to enumerate configuration from an unauthorised host. There is no
    peer check to run first, so the same address that used to learn nothing now gets
    the same 503 as loopback - and that IS the disclosure this release accepts:
    `/status` on an ENABLED plugin publishes this plugin's own configuration to any
    peer that can route here. What is still true, and what this asserts, is that
    nothing leaks before the switch is thrown: while inert, every peer gets the
    refusal and not the payload.
    """
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    monkeypatch.setenv(config.ENV_ALLOW_RETIRED, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    for peer in (("127.0.0.1", 1), PROXY, ("10.9.8.7", 1)):
        response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                              peer).request(method, path, headers=token_headers("tk"))
        assert response.status_code == 503, (peer, method, path, response.text)
        assert config.ENV_ENABLE in response.json()["detail"]
        # The self-description (policy_target, the parsed allowlist that used to be
        # here) is published only by an ENABLED surface, to every peer or to none.
        assert "policy_target" not in response.text


# --- real sockets: the peer uvicorn writes is the peer of the socket ---------

def test_real_socket_loopback_read_needs_no_token(enabled):
    """THE operator workflow, over a real accepted TCP connection: no token, no
    allowlist, `curl 127.0.0.1:<port>/plugins/kvtier/status` -> 200."""
    engine = FakeEngineCoreClient(result={"version": 1})
    with served_on_real_socket(build_app(FakeAsyncLLM(engine)),
                              "127.0.0.1") as base:
        with real_client(base) as client:
            assert client.get(STATUS).status_code == 200
            assert client.get(SNAPSHOT).status_code == 200
    assert engine.calls == [(config.METHOD_SNAPSHOT, ())]


@needs_lan
def test_real_socket_non_loopback_peer_is_served_because_nothing_filters_it(
        enabled, monkeypatch):
    """A genuine non-loopback peer on a genuine socket: served, with no header of any
    kind, and with the dead token variable set in the engine's environment.

    This is the test whose whole existence was the peer gate, and the reason it had to
    be a real socket: what uvicorn writes into `scope["client"]` is the peer of the
    socket, and that is the value the removed check read. Now the interesting half is
    that a peer on the LAN is simply served - and that the payload it reads says so,
    because the only remaining protection is the network the operator drew.
    """
    peer = LAN_PEER
    engine = FakeEngineCoreClient(result={"version": 1})
    monkeypatch.setenv(config.ENV_TOKEN, "tk")      # the dead knob, set on purpose
    with served_on_real_socket(build_app(FakeAsyncLLM(engine)), peer) as base:
        with real_client(base, source=peer) as client:
            # Sanity: this really is a non-loopback peer as far as uvicorn sees - the
            # fact the old guard refused on, asserted here as a fact and not a verdict.
            assert not ipaddress.ip_address(peer).is_loopback
            body = client.get(STATUS).json()
            assert body["peer_gate"]["peers_filtered_by_address"] is False
            assert body["token_gating"] is False
            # A stale header still changes nothing.
            assert client.get(STATUS, headers=token_headers("nope")).status_code == 200
            assert client.get(SNAPSHOT).status_code == 200
            assert client.post(RELOAD).status_code == 200
    assert [name for name, _ in engine.calls] == [config.METHOD_SNAPSHOT,
                                                  config.METHOD_RELOAD_POLICY]


@needs_lan
def test_real_socket_forwarded_for_is_not_a_credential_and_not_a_refusal(
        enabled, monkeypatch):
    """The forgeable-header attack against a real socket, restated for a gate that
    never looks at a header: a real LAN peer that claims to be 127.0.0.1 gets the
    same answer as the same peer that claims nothing, because there is no identity in
    the header for it to steal.

    Before 0.2.72 every one of these was a 403 naming the peer's real address, and
    that was the guard's whole purpose: read the socket, ignore the header. The
    property that survives is the narrower one - the header has no effect at all -
    and it is what a reviewer should re-check if a future change starts parsing one.
    """
    peer = LAN_PEER
    engine = FakeEngineCoreClient(result={"version": 1})
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    with served_on_real_socket(build_app(FakeAsyncLLM(engine)), peer) as base:
        with real_client(base, source=peer) as client:
            plain = client.get(STATUS)
            assert plain.status_code == 200
            for headers in ({"x-forwarded-for": "127.0.0.1"},
                            {"x-forwarded-for": "127.0.0.1",
                             **token_headers("tk")},
                            {"x-real-ip": "127.0.0.1",
                             **token_headers("tk")}):
                forged = client.get(STATUS, headers=headers)
                assert forged.status_code == 200, headers
                assert forged.json()["peer_gate"] == plain.json()["peer_gate"], headers


# --- inert by default -------------------------------------------------------

@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_inert_answers_503_even_on_loopback(method, path, monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")     # the dead knob does not unlock it
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))
                          ).request(method, path)
    assert response.status_code == 503
    assert config.ENV_ENABLE in response.json()["detail"]


# --- snapshot ---------------------------------------------------------------

def test_snapshot_returns_the_engine_payload(enabled):
    payload = {"version": 1, "tier": {"present": True, "num_cpu_blocks": 4096}}
    engine = FakeEngineCoreClient(result=payload)
    response = client_for(build_app(FakeAsyncLLM(engine))).get(SNAPSHOT)
    assert response.status_code == 200
    assert response.json() == payload
    assert engine.calls == [(config.METHOD_SNAPSHOT, ())]


def test_snapshot_answers_504_when_the_engine_does_not_answer(enabled, monkeypatch):
    """No engine-side timeout exists (core_client.py:971), so this is the only
    thing that keeps a wedged EngineCore from hanging the client."""
    monkeypatch.setenv(config.ENV_TIMEOUT, "0.05")
    engine = FakeEngineCoreClient(delay=5.0)
    response = client_for(build_app(FakeAsyncLLM(engine))).get(SNAPSHOT)
    assert response.status_code == 504
    assert "0.05" in response.json()["detail"]


def test_snapshot_answers_502_when_the_utility_call_raises(enabled):
    engine = FakeEngineCoreClient(raises=RuntimeError("engine said no"))
    response = client_for(build_app(FakeAsyncLLM(engine))).get(SNAPSHOT)
    assert response.status_code == 502
    assert "engine said no" in response.json()["detail"]


def test_snapshot_answers_503_without_an_engine_channel(enabled):
    app = build_app(None)                        # e.g. the CPU render server
    response = client_for(app).get(SNAPSHOT)
    assert response.status_code == 503


def test_the_utility_channel_is_found_on_the_engine_client_itself(enabled):
    """Some callers hand us an object that already has call_utility_async."""
    class Direct(FakeEngineCoreClient):
        pass

    direct = Direct(result={"version": 1})
    response = client_for(build_app(direct)).get(SNAPSHOT)   # no .engine_core
    assert response.status_code == 200
    assert direct.calls == [(config.METHOD_SNAPSHOT, ())]


# --- reload -----------------------------------------------------------------

def test_reload_needs_no_token_from_loopback(enabled):
    """The verb the token was ORIGINALLY invented for. It now needs only the peer
    rule, with the variable unset and no header - and reaches the engine."""
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(RELOAD).status_code == 200
    assert engine.calls == [(config.METHOD_RELOAD_POLICY, ())]


def test_reload_ignores_the_token_header_entirely(enabled):
    """No header, a wrong header, an empty header: all three reload the policy.

    A caller that still sends the header must not start failing, and a caller that
    believes the header is what authorises it must not believe it works: there is
    no answer in which the header is the reason this call succeeded.
    """
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    for headers in ({}, token_headers("nope"), token_headers(""),
                    token_headers("s3cr3t-token")):
        assert client.post(RELOAD, headers=headers).status_code == 200, headers
    assert engine.calls == [(config.METHOD_RELOAD_POLICY, ())] * 4


def test_a_mutating_disk_op_on_get_is_still_a_405_without_any_token(
        enabled, monkeypatch):
    """The other half of POST-only discipline, which this release deliberately did
    NOT relax: `MUTATING_DISK_OPS` turns a mutating `op` on a GET into 405 before
    the engine is touched, so a page load, a prefetch or a proxy retry still cannot
    arm the spill tier. It was never a token rule and it is not one now - and with
    the token gone it is the ONLY thing standing between `<img src>` and `op=arm`.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    for op in disk_store.MUTATING_DISK_OPS:
        engine = FakeEngineCoreClient(result={"version": 1})
        response = client_for(build_app(FakeAsyncLLM(engine)),
                              ("127.0.0.1", 1)).get(f"{DISK}?op={op}")
        assert response.status_code == 405, (op, response.text)
        assert "will not run from GET" in response.json()["detail"], op
        assert engine.calls == [], op
    # And the safe read on the same route still answers, token-free - with the
    # `safe_method` marker that tells the engine which method asked.
    engine = FakeEngineCoreClient(result={"version": 1})
    assert client_for(build_app(FakeAsyncLLM(engine))).get(
        DISK).status_code == 200
    assert engine.calls == [(config.METHOD_DISK,
                             ({"op": "status", "safe_method": True},))]


def test_reload_takes_no_arguments_on_the_engine_side(enabled):
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    client.post(RELOAD, json={})
    client.post(RELOAD)
    assert engine.calls == [(config.METHOD_RELOAD_POLICY, ())] * 2


def test_reload_rejects_a_foreign_path_and_never_reaches_the_engine(
        enabled, monkeypatch, tmp_path):
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    for candidate in ("/etc/passwd", "os.path", "../elsewhere/policy.py",
                      str(tmp_path / "policy.py"), "policy.py?",
                      str(config.canonical_policy_path().parent / "evil.py")):
        response = client.post(RELOAD, json={"path": candidate})
        assert response.status_code == 400, (candidate, response.text)
        assert "only loadable policy" in response.json()["detail"]
    assert engine.calls == []


def test_reload_rejects_unknown_keys_and_non_objects(enabled):
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(RELOAD, json={"module": "os"}).status_code == 400
    assert client.post(RELOAD, json=[1, 2]).status_code == 400
    assert client.post(RELOAD, data=b"{not json").status_code == 400
    assert engine.calls == []


def test_reload_accepts_the_canonical_path_and_reports_both_sides(
        enabled, monkeypatch):
    engine = FakeEngineCoreClient(result={"ok": True, "applied": {"v": 2}})
    response = client_for(build_app(FakeAsyncLLM(engine))).post(
        RELOAD, json={"path": str(config.canonical_policy_path())})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == {"ok": True, "applied": {"v": 2}}
    assert body["request"]["path_on_wire"] is True
    assert body["request"]["validated_path"] == str(config.canonical_policy_path())


def test_reload_reports_the_peer_that_asked(enabled, monkeypatch, caplog):
    engine = FakeEngineCoreClient(result={"ok": True})
    with caplog.at_level("INFO"):
        client_for(build_app(FakeAsyncLLM(engine)), ("127.0.0.1", 9999)).post(RELOAD)
    assert any("127.0.0.1" in record.getMessage() and "reload" in record.getMessage()
               for record in caplog.records)


# --- init_state contract ----------------------------------------------------

def test_init_state_stores_the_client_and_survives_a_none_one():
    app = FastAPI()
    plugin = KVTierDevEndpointPlugin()
    plugin.attach_router(app)
    engine_core = FakeEngineCoreClient()
    asyncio.run(plugin.init_state(FakeAsyncLLM(engine_core), app.state,
                                 argparse.Namespace()))
    assert app.state.kvtier_dev["engine_client"] is not None
    asyncio.run(plugin.init_state(None, app.state, argparse.Namespace()))
    assert app.state.kvtier_dev["engine_client"] is None
    # With no engine at all, reads still answer 503 instead of 500.
    assert client_for(app).get(SNAPSHOT).status_code == 503


def test_routes_work_when_init_state_never_ran(enabled):
    """run_batch.py builds a bare State without init_app_state; the fallback to
    app.state.engine_client must hold."""
    app = FastAPI()
    KVTierDevEndpointPlugin().attach_router(app)
    app.state.engine_client = FakeAsyncLLM(FakeEngineCoreClient(
        result={"version": 1}))
    assert client_for(app).get(SNAPSHOT).status_code == 200


# --- the dead variable announces itself (0.2.68) ----------------------------
#
# "No silent dead config": a variable an operator still exports, believing it is
# a security control, must be named out loud at the boot that ignores it. One
# WARNING line, only when it is set - and `/status` carries the same sentence for
# anyone who boots without reading the log.

def _boot_lines(caplog, needle):
    return [record.getMessage() for record in caplog.records
            if needle in record.getMessage()]


def test_a_boot_with_the_dead_token_var_set_warns_about_once(caplog, monkeypatch,
                                                             enabled):
    monkeypatch.setenv(config.ENV_TOKEN, "s3cr3t-token")
    with caplog.at_level("INFO"):
        build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    lines = _boot_lines(caplog, config.ENV_TOKEN)
    assert len(lines) == 1, lines
    line = lines[0]
    assert "no longer" in line.lower() and "0.2.68" in line
    # WHAT REPLACED THE TOKEN is no longer "the peer allowlist" - that gate is gone
    # too, so the line must name the switch that IS the gate and the release that
    # removed the other one, or an operator reads it as "set the allowlist instead".
    assert config.ENV_ENABLE in line, "must name the gate that is left"
    assert "0.2.72" in line, "must not claim the allowlist still stands"
    assert config.ENV_ALLOW_RETIRED not in line, "each dead variable gets its own line"
    assert "s3cr3t-token" not in line       # the VALUE is never echoed, only the name
    assert "token_env" not in line


def test_the_dead_token_boot_line_is_a_warning(caplog, monkeypatch, enabled):
    """Level matters: an INFO line among the dozen boot lines an operator never
    re-reads is how a dead knob stays dead."""
    monkeypatch.setenv(config.ENV_TOKEN, "s3cr3t-token")
    with caplog.at_level("WARNING"):
        build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    lines = [record for record in caplog.records
             if config.ENV_TOKEN in record.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelname == "WARNING"


def test_a_boot_without_the_token_var_says_nothing_about_it(caplog, monkeypatch,
                                                            enabled):
    """The warning is about a variable that EXISTS. Silence when it is unset is
    what keeps this one line instead of noise on every boot."""
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    with caplog.at_level("INFO"):
        build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    assert _boot_lines(caplog, config.ENV_TOKEN) == []
    assert _boot_lines(caplog, "no longer") == []
