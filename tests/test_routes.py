"""Route level: the real FastAPI stack, a fake engine client, no vllm.

Two transports, deliberately.

* TestClient's `client=` argument is what starlette puts into
  `scope["client"]` - the same field uvicorn fills from the accepted socket.
  It is the only way to assert on a peer this host cannot source packets from
  (the real proxy at `192.168.2.64`, the outside half of a CIDR, a spoofed
  `X-Forwarded-For`), so the allowlist matrix is driven that way.
* The real-uvicorn tests at the bottom (`served_on_real_socket`) accept a real
  TCP connection, which is the only way to prove that the peer uvicorn writes
  into the scope is the peer of the socket - and to prove the operator's
  token-free `ssh llm curl 127.0.0.1` read still works over a real socket.
  A non-loopback peer is driven from a client socket bound to one of this
  host's own LAN addresses; a CI box with no such address skips those.
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


def assert_refused_on_peer_grounds(response, peer):
    """THE one shape a refusal of this surface may have since 0.2.68.

    403 whose detail names the PEER rule (loopback, or `VLLM_E2_TIER_DEV_ALLOW`),
    names the peer's own address, does NOT leak the allowlist's contents, and
    mentions no token: the peer gate is the only gate, so a refusal that read like
    a token refusal would send an operator looking for a knob that no longer exists
    (and used to make "is my peer wrong or is my token wrong" unanswerable).
    """
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert "not loopback" in detail, detail
    assert config.ENV_ALLOW in detail and "allowlist" in detail, detail
    assert peer in detail, detail
    assert "token" not in detail.lower(), detail
    return detail


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


# --- loopback ---------------------------------------------------------------

@pytest.mark.parametrize("peer", [
    ("127.0.0.1", 43210),
    ("127.2.3.4", 1),
    ("::1", 1),
    ("::ffff:127.0.0.1", 1),
])
def test_loopback_peers_may_read(enabled, peer):
    app = build_app(FakeAsyncLLM(FakeEngineCoreClient()))
    response = client_for(app, peer).get(STATUS)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("peer", [
    ("10.0.0.7", 1),
    ("192.168.2.42", 8000),
    ("0.0.0.0", 1),
])
def test_remote_peers_are_403_on_peer_grounds(enabled, peer):
    """No token involved and none needed: the peer rule alone answers 403, and the
    detail is the peer's own address against the allowlist."""
    assert_refused_on_peer_grounds(
        client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                   peer).get(STATUS), peer[0])


def test_a_peer_that_is_not_an_ip_is_refused_as_unprovable_not_as_unauthorized(
        enabled):
    """`testclient` is what a test harness reports: not a parseable address, so
    loopback cannot be proved and the refusal says so - and still says nothing
    about a token, which no route here asks for."""
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          ("testclient", 50000)).get(STATUS)
    assert response.status_code == 403
    assert "parseable" in response.json()["detail"]
    assert "token" not in response.text.lower()


def test_spoofed_forwarded_for_from_remote_peer_is_403(enabled):
    """The attack the guard exists for: a LAN client claims to be loopback.

    The last call still carries the (now inert) token header, so this doubles as
    proof that a header cannot help a peer the allowlist refused.
    """
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("203.0.113.9", 51234))
    for response in (
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1"}),
        client.get(SNAPSHOT, headers={"x-forwarded-for": "127.0.0.1, 127.0.0.1"}),
        client.get(SNAPSHOT, headers={"x-real-ip": "127.0.0.1"}),
        client.post(RELOAD, headers={**token_headers("tk"),
                                     "x-forwarded-for": "127.0.0.1"}),
    ):
        assert_refused_on_peer_grounds(response, "203.0.113.9")
    assert engine.calls == []          # the engine was never touched


def test_a_loopback_peer_does_not_need_a_forged_header_to_pass(enabled):
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("127.0.0.1", 1))
    assert client.get(STATUS, headers={"x-forwarded-for": "203.0.113.9"}
                      ).status_code == 200


def test_remote_peer_gets_no_configuration_leak(enabled):
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          ("10.1.2.3", 1)).get(STATUS)
    assert "policy_target" not in response.text
    assert "10.1.2.3" in response.json()["detail"]


# --- the allowlist (VLLM_E2_TIER_DEV_ALLOW) ----------------------------------
#
# Peers here are TestClient-constructed on purpose: `192.168.2.64` is the
# production proxy and no test box can legitimately source packets from it.
# The real-socket section below covers the peers we CAN source from.

PROXY = ("192.168.2.64", 51000)
WORKSTATION = ("192.168.2.120", 51001)


def test_an_allowlisted_peer_may_read_without_a_token(enabled, monkeypatch):
    """The whole point of 0.2.68: an allowlisted peer reads the surface with the
    headers a plain `curl` sends. No `x-kvtier-dev-token` exists any more."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    engine = FakeEngineCoreClient(result={"version": 1})
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    assert client.get(STATUS).status_code == 200
    assert client.get(SNAPSHOT).status_code == 200
    assert engine.calls == [(config.METHOD_SNAPSHOT, ())]


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_an_allowlisted_peer_needs_no_token_on_every_route(enabled, monkeypatch,
                                                           method, path):
    """THE rule of 0.2.68, over the whole surface: peer, then enabled, then the
    route runs - for every route, both methods, no header anywhere.

    This includes every member of `config.DEV_STATE_ROUTES` (trace, guard, tier,
    lookups, blockdump, disk), which is the shape that used to refuse even a
    LOOPBACK reader: those routes are still listed as dev-state routes, they just
    no longer carry a second gate.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    engine = FakeEngineCoreClient(result={"version": 1})
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    response = client.request(method, path)
    assert response.status_code == 200, (method, path, response.text)


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_a_leftover_token_header_changes_no_answer_on_any_route(enabled,
                                                                monkeypatch,
                                                                method, path):
    """An operator's old script must not start failing, and must not start working
    either: the header is inert on every route, and an unset token env is inert the
    same way (both are tested by the same pair of requests)."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
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
def test_a_peer_off_the_allowlist_is_403_on_every_route_on_peer_grounds(
        enabled, monkeypatch, method, path):
    """The gate that REMAINS, asserted alone: a stranger holding the old token (env
    set, header sent) is refused because of its ADDRESS, and the detail proves it.

    Before 0.2.68 this test could not exist in this shape: the same 403 came back
    from two different guards and only their wording told them apart. Now there is
    one guard, so every refusal must read as the peer rule - a token-shaped refusal
    here would mean a gate came back.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")      # the dead knob, set on purpose
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("10.9.8.7", 1))
    response = client.request(method, path, headers=token_headers("tk"))
    assert_refused_on_peer_grounds(response, "10.9.8.7")
    assert engine.calls == []


def test_a_stranger_holding_the_token_is_still_refused(enabled, monkeypatch):
    """The allowlist is the outer gate and the token is not a gate at all: a peer
    that is not on it is 403 on every route and the engine never hears about it."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("10.9.8.7", 1))
    for method, path in (("get", STATUS), ("get", SNAPSHOT), ("post", RELOAD)):
        response = client.request(method, path, headers=token_headers("tk"))
        assert_refused_on_peer_grounds(response, "10.9.8.7")
    assert engine.calls == []


@pytest.mark.parametrize("peer,expected", [
    (("192.168.2.1", 1), 200),      # inside the CIDR
    (("192.168.2.255", 1), 200),    # ... right up to the boundary
    (("192.168.3.1", 1), 403),      # one octet out
    (("192.168.1.255", 1), 403),
])
def test_cidr_entry_admits_inside_and_refuses_outside(enabled, monkeypatch,
                                                      peer, expected):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          peer).get(STATUS)
    assert response.status_code == expected
    if expected == 403:
        assert_refused_on_peer_grounds(response, peer[0])


def test_unset_allow_refuses_an_allowlisted_looking_peer(enabled, monkeypatch):
    """Fail-closed default, restated at the route level: without the env there
    is no allowlist, so the production proxy is refused - with the dead token env
    set and its header sent, which buys it nothing."""
    monkeypatch.delenv(config.ENV_ALLOW, raising=False)
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    assert_refused_on_peer_grounds(client.get(STATUS, headers=token_headers("tk")),
                                   "192.168.2.64")
    assert_refused_on_peer_grounds(client.post(RELOAD, headers=token_headers("tk")),
                                   "192.168.2.64")
    assert engine.calls == []


@pytest.mark.parametrize("env", [
    "garbage", "", "   ", "10.0.0.0/99", "192.168.2.999", "localhost", "::x",
])
def test_malformed_allow_env_is_403_not_500(enabled, monkeypatch, env):
    """Garbage in the env must not break the request path and must not open it.
    `localhost` in particular is never resolved."""
    monkeypatch.setenv(config.ENV_ALLOW, env)
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    for method, path in (("get", STATUS), ("post", RELOAD)):
        response = client.request(method, path)
        assert response.status_code == 403, (env, method, response.text)
        assert_refused_on_peer_grounds(response, "192.168.2.64")
    assert engine.calls == []


def test_a_good_entry_survives_a_bad_neighbour_on_the_route(
        enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "garbage,,192.168.2.64,10.0.0.0/99")
    client = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())), PROXY)
    assert client.get(STATUS).status_code == 200
    assert_refused_on_peer_grounds(
        client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                   WORKSTATION).get(STATUS), "192.168.2.120")


@pytest.mark.parametrize("peer", [("127.0.0.1", 1), ("127.2.3.4", 1),
                                 ("::1", 1), ("::ffff:127.0.0.1", 1)])
def test_loopback_reads_stay_token_free_when_an_allowlist_is_set(
        enabled, monkeypatch, peer):
    """The operator's `ssh llm curl 127.0.0.1` workflow, unchanged by the
    allowlist - and an allowlist naming other hosts must not remove loopback."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          peer).get(STATUS)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("route", list(config.DEV_STATE_ROUTES))
def test_the_dev_state_routes_answer_a_loopback_peer_with_no_token(
        enabled, monkeypatch, route):
    """THE rule this release exists for, on exactly the routes that made testing
    painful: every member of DEV_STATE_ROUTES, GET and POST, answers a loopback
    peer that sends nothing but the URL.

    Until 0.2.68 these six routes required the token in EVERY shape - loopback
    included, `GET /trace` and `POST /disk {"op":"status"}` alike - which is what
    made the surface unusable from a script or from the proxy. `DEV_STATE_ROUTES`
    is read from `config` here, so a seventh route is covered the day it is added.
    """
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    engine = FakeEngineCoreClient(result={"version": 1})
    client = client_for(build_app(FakeAsyncLLM(engine)), ("127.0.0.1", 1))
    path = config.ROUTE_PREFIX + route
    assert client.get(path).status_code == 200, (route, "GET")
    assert client.post(path, json={}).status_code == 200, (route, "POST")


def test_the_mutating_verb_needs_no_token_on_loopback(enabled, monkeypatch):
    """`POST /policy/reload` used to be the ONE mutating verb and the ONE thing the
    token was documented to gate. It now needs only the peer rule - and the engine
    call it drives happens with no header in the request."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(RELOAD).status_code == 200
    assert engine.calls == [(config.METHOD_RELOAD_POLICY, ())]


def test_a_wide_allowlist_reaches_the_surface_without_a_token(enabled, monkeypatch):
    """`0.0.0.0/0` is now the whole gate, and it is the ONLY thing that opens: the
    peer is admitted by its address alone. (Which is exactly why README says keep it
    narrow: these routes are NOT covered by vLLM's `--api-key`.)"""
    monkeypatch.setenv(config.ENV_ALLOW, "0.0.0.0/0")
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    engine = FakeEngineCoreClient(result={"version": 1})
    response = client_for(build_app(FakeAsyncLLM(engine)),
                          WORKSTATION).get(STATUS)
    assert response.status_code == 200


def test_spoofed_forwarded_for_from_a_peer_not_on_the_allowlist_is_403(
        enabled, monkeypatch):
    """Same attack, now against the widened surface, with the dead header in hand:
    a forged header and an old token together still cannot reach one route."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("203.0.113.9", 51234))
    for response in (
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1"}),
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1",
                                   **token_headers("tk")}),
        client.get(SNAPSHOT, headers={"x-real-ip": "127.0.0.1",
                                     **token_headers("tk")}),
        client.post(RELOAD, headers={"x-forwarded-for": "192.168.2.64",
                                    **token_headers("tk")}),
    ):
        assert_refused_on_peer_grounds(response, "203.0.113.9")
    assert engine.calls == []


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
    """THE DEPLOYMENT CASE, with the live box's actual allowlist.

    `192.168.2.64` is the proxy that will poll this; `192.168.2.120` is the workstation.
    Both are on the list, NOTHING is configured in `VLLM_E2_TIER_DEV_TOKEN`, and the
    answer must be the block - not a redacted status payload, and not an engine call:
    `/status` makes none, and the fill comes from the rank files the workers write on
    this host. This is the case that made the token worth removing: a poller that has
    to carry a secret for a read it is allowlisted for is a poller that gets dropped.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
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
    """Both allowlisted peers, one dialect: the block is not peer-specific."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(STATUS).json()
    assert body["disk_fill"]["used_bytes"] == FILL_USED


def test_the_fill_block_is_not_served_to_a_peer_off_the_allowlist(
        enabled, monkeypatch, fill_store):
    """The allowlist is necessary AND sufficient now: no allowlist means no fill,
    and no token is left to make up the difference."""
    monkeypatch.delenv(config.ENV_ALLOW, raising=False)
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    client = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())), PROXY)
    for headers in ({}, token_headers("tk"), {"x-kvtier-dev-token": ""}):
        response = client.get(STATUS, headers=headers)
        assert_refused_on_peer_grounds(response, "192.168.2.64")
        assert "disk_fill" not in response.text, response.text


def test_a_peer_off_the_allowlist_does_not_get_the_fill_block(enabled,
                                                             monkeypatch,
                                                             fill_store):
    """The proxy's address is not on the list here, and its old token is spent: 403,
    no fill, no leak, and the refusal is the peer's own."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          ("10.9.8.7", 1)).get(STATUS, headers=token_headers("tk"))
    assert_refused_on_peer_grounds(response, "10.9.8.7")
    assert "disk_fill" not in response.text


def test_loopback_reads_the_fill_block_with_no_token_at_all(enabled, monkeypatch,
                                                            fill_store):
    """The operator at the box: `/status` on loopback needs nothing even with an
    allowlist set, which is what makes a plain curl a usable reflex."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert body["disk_fill"]["used_bytes"] == FILL_USED
    assert body["disk_fill"]["ranks_reported"] == 1


def test_a_spoofed_forwarded_for_does_not_unlock_the_fill_block(enabled, monkeypatch,
                                                                fill_store):
    """The same attack as the existing spoof test, aimed at the new block.

    `X-Forwarded-For: 127.0.0.1` from a host that is not allowlisted must not turn a 403
    into a read of the tier's fill - and neither does the old token header, which is
    what the second and third requests are for.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    fill_rank_file(fill_store)
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("203.0.113.9", 51234))
    for response in (
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1"}),
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1",
                                   **token_headers("tk")}),
        client.get(STATUS, headers={"x-real-ip": "192.168.2.64",
                                   **token_headers("tk")}),
        client.get(SNAPSHOT, headers={"x-forwarded-for": "192.168.2.64",
                                     **token_headers("tk")}),
    ):
        assert_refused_on_peer_grounds(response, "203.0.113.9")
        assert "disk_fill" not in response.text, response.text
    assert engine.calls == []


def test_the_snapshot_route_carries_the_fill_block_to_the_proxy_peer(enabled,
                                                                    monkeypatch,
                                                                    fill_store):
    """The engine-side view reaches the proxy too.

    `/snapshot` is the ENGINE's payload (the utility call), and `dev_state_sections()` -
    where `disk_fill` is built - is embedded at its top level. The fake engine here
    returns what the real handler would embed, so this proves the ROUTE carries the
    block to an authorised off-box peer; building it is `test_disk_fill_block.py`.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
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
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
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
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
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
    """peer -> enabled: the 503 comes before any fill read."""
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    fill_rank_file(fill_store)
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          PROXY).get(STATUS)
    assert response.status_code == 503
    assert "disk_fill" not in response.text


def test_a_refused_peer_does_not_learn_the_allowlist_contents(enabled,
                                                              monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24,10.7.7.0/24")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")     # the dead knob, set on purpose
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          ("10.1.2.3", 1)).get(STATUS, headers=token_headers("tk"))
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "10.1.2.3" in detail and config.ENV_ALLOW in detail
    assert "192.168.2.0" not in detail and "10.7.7.0" not in detail


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


def test_status_reports_loopback_only_when_nothing_is_allowlisted(enabled):
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert body["allow_networks"] == []
    assert body["loopback_only"] is True
    assert body["allow_env"] == config.ENV_ALLOW
    # 0.2.68: the peer allowlist is the whole gate, so NO peer - loopback or not -
    # is asked for a token. The key STAYS with its value flipped rather than being
    # deleted: something reading it must find a truthful `false`, not a KeyError
    # that looks like the payload changed shape.
    assert body["non_loopback_peers_require_token"] is False
    assert body["token_gating"] is False
    assert config.ENV_ALLOW in body["guard"]
    assert "loopback" in body["guard"]


def test_status_says_the_token_no_longer_means_what_it_used_to(enabled, monkeypatch):
    """THE dated, explicit statement this project prefers over a removed key.

    `token_configured`/`token_env` keep their literal meaning ("is the variable
    set", "which variable"), because a key vanishing from `/status` is a silent
    change for anything that reads it. Beside them: `token_gating` is `false`, the
    note names the release that stopped reading the variable, and the note and
    `guard` both say the thing a future reader most needs - the peer allowlist is
    the ONLY gate here, and vLLM's `--api-key` does not cover `/plugins/*`.
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
    assert config.ENV_ALLOW in note
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


def test_status_reports_the_parsed_allowlist_to_a_peer_it_admits(
        enabled, monkeypatch):
    """An operator must be able to see what is open, including which entries
    the parser dropped."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64, 192.168.2.0/24, bogus")
    monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(STATUS).json()
    assert body["allow_networks"] == ["192.168.2.64/32", "192.168.2.0/24"]
    assert body["loopback_only"] is False
    assert body["non_loopback_peers_require_token"] is False


def test_a_malformed_entry_shows_up_as_absent_to_a_loopback_reader(
        enabled, monkeypatch):
    """`status` is always a readable place to check whether the env parsed the way
    you meant (nothing asks for a token to read it), and `bogus` is simply absent."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,10.0.0.0/99,bogus")
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert body["allow_networks"] == ["192.168.2.64/32"]
    assert body["loopback_only"] is False


# --- guard order -------------------------------------------------------------

def test_inert_still_answers_503_to_an_allowlisted_peer(monkeypatch):
    """peer -> enabled: an admitted peer still learns the plugin is inert rather
    than getting a 200, and the dead token variable changes nothing about that."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          PROXY).get(STATUS)
    assert response.status_code == 503
    assert config.ENV_ENABLE in response.json()["detail"]


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_a_disallowed_peer_learns_nothing_even_when_inert(
        monkeypatch, method, path):
    """The peer check runs BEFORE the 503, so an inert plugin is not a way to
    enumerate configuration from an unauthorised host."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          ("10.9.8.7", 1)).request(method, path,
                                                   headers=token_headers("tk"))
    assert_refused_on_peer_grounds(response, "10.9.8.7")
    # The 503 "inert" answer (and everything the plugin knows about itself) is
    # only ever handed to a peer the guard accepted. `inert:` is the 503's text;
    # ENV_ENABLE alone is useless as a marker because ENV_ALLOW contains it.
    assert "inert" not in response.text
    assert "policy_target" not in response.text


# --- real sockets: the peer uvicorn writes is the peer of the socket ---------

def test_real_socket_loopback_read_needs_no_token(enabled):
    """THE operator workflow, over a real accepted TCP connection: no allowlist,
    no token, `curl 127.0.0.1:<port>/plugins/kvtier/status` -> 200."""
    engine = FakeEngineCoreClient(result={"version": 1})
    with served_on_real_socket(build_app(FakeAsyncLLM(engine)),
                              "127.0.0.1") as base:
        with real_client(base) as client:
            assert client.get(STATUS).status_code == 200
            assert client.get(SNAPSHOT).status_code == 200
    assert engine.calls == [(config.METHOD_SNAPSHOT, ())]


@needs_lan
def test_real_socket_non_loopback_peer_is_refused_then_served(enabled, monkeypatch):
    """A genuine non-loopback peer on a genuine socket: refused while the allowlist
    is unset - holding the old token changes nothing about that - and served the
    moment its address is allowlisted, with no header of any kind.

    The refusal is asserted on PEER grounds by the peer's own address appearing in
    the detail, which is the half only a real socket can prove: what uvicorn wrote
    into `scope["client"]` is what the guard refused.
    """
    peer = LAN_PEER
    engine = FakeEngineCoreClient(result={"version": 1})
    monkeypatch.setenv(config.ENV_TOKEN, "tk")      # the dead knob, set on purpose
    with served_on_real_socket(build_app(FakeAsyncLLM(engine)), peer) as base:
        with real_client(base, source=peer) as client:
            # Sanity: this really is a non-loopback peer as far as uvicorn sees.
            assert not ipaddress.ip_address(peer).is_loopback
            assert_refused_on_peer_grounds(client.get(STATUS), peer)
            assert_refused_on_peer_grounds(
                client.get(STATUS, headers=token_headers("tk")), peer)
            assert engine.calls == []
            monkeypatch.setenv(config.ENV_ALLOW, peer)
            # Allowlisted: the address alone is the whole credential.
            body = client.get(STATUS).json()
            assert body["loopback_only"] is False
            assert body["allow_networks"] == [f"{peer}/32"]
            assert body["token_gating"] is False
            # A stale header still changes nothing.
            assert client.get(STATUS, headers=token_headers("nope")).status_code == 200
            assert client.get(SNAPSHOT).status_code == 200
            assert client.post(RELOAD).status_code == 200
    assert [name for name, _ in engine.calls] == [config.METHOD_SNAPSHOT,
                                                  config.METHOD_RELOAD_POLICY]


@needs_lan
def test_real_socket_forwarded_for_does_not_help_a_real_remote_peer(
        enabled, monkeypatch):
    """The forgeable-header attack against a real socket: a real LAN peer that
    claims to be 127.0.0.1 is refused, with the dead header and without it."""
    peer = LAN_PEER
    engine = FakeEngineCoreClient()
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    with served_on_real_socket(build_app(FakeAsyncLLM(engine)), peer) as base:
        with real_client(base, source=peer) as client:
            for headers in ({"x-forwarded-for": "127.0.0.1"},
                            {"x-forwarded-for": "127.0.0.1",
                             **token_headers("tk")},
                            {"x-real-ip": "127.0.0.1",
                             **token_headers("tk")}):
                assert_refused_on_peer_grounds(client.get(STATUS, headers=headers),
                                               peer)
            assert engine.calls == []


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
    assert config.ENV_ALLOW in line, "must name the gate that replaced it"
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
