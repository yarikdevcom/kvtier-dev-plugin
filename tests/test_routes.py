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
# parametrized over it, so a new route is automatically tested for the peer rule,
# the inert rule and the token rule - and `test_PLUGIN_ROUTES_is_the_whole_surface`
# fails if this list and the router ever disagree.
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
    return {"x-kvtier-dev-token": token}


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
    ("testclient", 50000),      # a harness peer is not a proven loopback peer
])
def test_remote_peers_are_403(enabled, peer):
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          peer).get(STATUS)
    assert response.status_code == 403


def test_spoofed_forwarded_for_from_remote_peer_is_403(enabled):
    """The attack the guard exists for: a LAN client claims to be loopback."""
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("203.0.113.9", 51234))
    for response in (
        client.get(STATUS, headers={"x-forwarded-for": "127.0.0.1"}),
        client.get(SNAPSHOT, headers={"x-forwarded-for": "127.0.0.1, 127.0.0.1"}),
        client.get(SNAPSHOT, headers={"x-real-ip": "127.0.0.1"}),
        client.post(RELOAD, headers={**token_headers("tk"),
                                     "x-forwarded-for": "127.0.0.1"}),
    ):
        assert response.status_code == 403, response.text
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


def test_allowlisted_peer_may_read_with_the_token(enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"version": 1})
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    assert client.get(STATUS, headers=token_headers("tk")).status_code == 200
    assert client.get(SNAPSHOT, headers=token_headers("tk")).status_code == 200
    assert engine.calls == [(config.METHOD_SNAPSHOT, ())]


@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_allowlisted_peer_without_a_token_is_403_everywhere(
        enabled, monkeypatch, method, path):
    """The rule for off-box traffic: the token is required, reads included, and
    the routes that can change dev state require it even from loopback (which
    this peer is not). Nothing is served and the engine is never touched."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    response = client.request(method, path)
    assert response.status_code == 403
    assert TOKEN_HEADER in response.json()["detail"]
    assert engine.calls == []


def test_allowlisted_peer_with_a_wrong_token_is_403(enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    for token in ("nope", "", "t"):
        assert client.get(STATUS, headers=token_headers(token)).status_code == 403
    assert engine.calls == []


def test_a_token_never_bypasses_the_allowlist(enabled, monkeypatch):
    """The allowlist is the outer gate: a valid token from a peer that is not
    on it is 403 on every route, and the engine never hears about it."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), ("10.9.8.7", 1))
    for method, path in (("get", STATUS), ("get", SNAPSHOT), ("post", RELOAD)):
        response = client.request(method, path, headers=token_headers("tk"))
        assert response.status_code == 403, (method, path, response.text)
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
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          peer).get(STATUS, headers=token_headers("tk"))
    assert response.status_code == expected


def test_unset_allow_refuses_an_allowlisted_looking_peer(enabled, monkeypatch):
    """Fail-closed default, restated at the route level: without the env there
    is no allowlist, so the production proxy is refused even with the token."""
    monkeypatch.delenv(config.ENV_ALLOW, raising=False)
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    assert client.get(STATUS, headers=token_headers("tk")).status_code == 403
    assert client.post(RELOAD, headers=token_headers("tk")).status_code == 403
    assert engine.calls == []


@pytest.mark.parametrize("env", [
    "garbage", "", "   ", "10.0.0.0/99", "192.168.2.999", "localhost", "::x",
])
def test_malformed_allow_env_is_403_not_500(enabled, monkeypatch, env):
    """Garbage in the env must not break the request path and must not open it.
    `localhost` in particular is never resolved."""
    monkeypatch.setenv(config.ENV_ALLOW, env)
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    for method, path in (("get", STATUS), ("post", RELOAD)):
        response = client.request(method, path, headers=token_headers("tk"))
        assert response.status_code == 403, (env, method, response.text)
    assert engine.calls == []


def test_a_good_entry_survives_a_bad_neighbour_on_the_route(
        enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "garbage,,192.168.2.64,10.0.0.0/99")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    client = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())), PROXY)
    assert client.get(STATUS, headers=token_headers("tk")).status_code == 200
    assert client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(
        STATUS, headers=token_headers("tk")).status_code == 403


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


def test_the_mutating_verb_still_needs_the_token_on_loopback(
        enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(RELOAD).status_code == 403
    assert engine.calls == []
    assert client.post(RELOAD, headers=token_headers("tk")).status_code == 200


def test_an_allowlisted_peer_is_not_loopback_for_the_token_rule(
        enabled, monkeypatch):
    """Being on the allowlist buys reach, not exemption from the token."""
    monkeypatch.setenv(config.ENV_ALLOW, "0.0.0.0/0")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient()
    response = client_for(build_app(FakeAsyncLLM(engine)),
                          WORKSTATION).get(STATUS)
    assert response.status_code == 403
    assert engine.calls == []


def test_spoofed_forwarded_for_from_a_peer_not_on_the_allowlist_is_403(
        enabled, monkeypatch):
    """Same attack, now against the widened surface, with a valid token in hand."""
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
        assert response.status_code == 403, response.text
    assert engine.calls == []


# --- THE SSD TIER'S FILL BLOCK, read from off the box (0.2.30) ---------------
#
# WHY THIS IS A ROUTE TEST AND NOT ONLY A BLOCK TEST. `disk_fill` exists because the
# proxy on `192.168.2.64` is going to poll it every few seconds, and the live box's
# allowlist is now `192.168.2.64,192.168.2.120`. A block that is correct in a unit test
# and unreachable from the one peer that will read it is not a feature. So the whole
# peer matrix runs against THIS block: allowlisted plus token reads it, allowlisted
# without a token does not, a token does not bypass the allowlist, loopback needs no
# token at all, and a spoofed `X-Forwarded-For` buys nothing.

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


def test_the_proxy_peer_reads_the_fill_block_with_the_token(enabled, monkeypatch,
                                                            fill_store):
    """THE DEPLOYMENT CASE, with the live box's actual allowlist.

    `192.168.2.64` is the proxy that will poll this; `192.168.2.120` is the workstation.
    Both are on the list, the token is set, and the answer must be the block - not a
    redacted status payload, and not an engine call: `/status` makes none, and the fill
    comes from the rank files the workers write on this host.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    response = client.get(STATUS, headers=token_headers("tk"))
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
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(STATUS, headers=token_headers("tk")).json()
    assert body["disk_fill"]["used_bytes"] == FILL_USED


def test_the_fill_block_is_not_served_without_the_token(enabled, monkeypatch,
                                                        fill_store):
    """Allowlisted is necessary, not sufficient: a read of the fill needs the token."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    client = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())), PROXY)
    for headers in ({}, token_headers("wrong"), {"x-kvtier-dev-token": ""}):
        response = client.get(STATUS, headers=headers)
        assert response.status_code == 403, (headers, response.text)
        assert "disk_fill" not in response.text, response.text


def test_a_token_does_not_bypass_the_allowlist_for_the_fill_block(enabled,
                                                                  monkeypatch,
                                                                  fill_store):
    """The proxy's token, from a host that is not on the list: 403, no fill, no leak."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          ("10.9.8.7", 1)).get(STATUS, headers=token_headers("tk"))
    assert response.status_code == 403
    assert "disk_fill" not in response.text


def test_loopback_reads_the_fill_block_with_no_token_at_all(enabled, monkeypatch,
                                                            fill_store):
    """The operator at the box: `/status` on loopback stays token-free even with an
    allowlist set, which is what makes a plain curl a usable reflex."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,192.168.2.120")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert body["disk_fill"]["used_bytes"] == FILL_USED
    assert body["disk_fill"]["ranks_reported"] == 1


def test_a_spoofed_forwarded_for_does_not_unlock_the_fill_block(enabled, monkeypatch,
                                                                fill_store):
    """The same attack as the existing spoof test, aimed at the new block.

    `X-Forwarded-For: 127.0.0.1` from a host that is not allowlisted must not turn a 403
    into a read of the tier's fill, with or without a valid token.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
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
        assert response.status_code == 403, response.text
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
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    engine = FakeEngineCoreClient(result={"version": 1,
                                          "disk_fill": disk_store.disk_fill()})
    client = client_for(build_app(FakeAsyncLLM(engine)), PROXY)
    response = client.get(SNAPSHOT, headers=token_headers("tk"))
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
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    for pp in range(6):
        for tp in range(4):
            fill_rank_file(fill_store, rank=f"PP{pp}_TP{tp}", age=float(pp))
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      PROXY).get(STATUS, headers=token_headers("tk")).json()
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
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      PROXY).get(STATUS, headers=token_headers("tk")).json()
    fill = body["disk_fill"]
    assert fill["available"] is False
    assert fill["used_bytes"] is None
    assert fill["basis"] == "no_rank_files", fill["basis"]
    assert fill["used_fraction"] is None


def test_an_inert_plugin_learns_the_fill_block_never_ran(enabled, monkeypatch,
                                                         fill_store):
    """peer -> enabled -> token: the 503 comes before any fill read."""
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    fill_rank_file(fill_store)
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          PROXY).get(STATUS, headers=token_headers("tk"))
    assert response.status_code == 503
    assert "disk_fill" not in response.text


def test_a_refused_peer_does_not_learn_the_allowlist_contents(enabled,
                                                              monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24,10.7.7.0/24")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
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
    assert body["non_loopback_peers_require_token"] is True
    assert config.ENV_ALLOW in body["guard"]
    assert "loopback" in body["guard"]


def test_status_reports_the_parsed_allowlist_to_a_peer_it_admits(
        enabled, monkeypatch):
    """An operator must be able to see what is open, including which entries
    the parser dropped."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64, 192.168.2.0/24, bogus")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                      WORKSTATION).get(STATUS, headers=token_headers("tk")).json()
    assert body["allow_networks"] == ["192.168.2.64/32", "192.168.2.0/24"]
    assert body["loopback_only"] is False
    assert body["non_loopback_peers_require_token"] is True


def test_a_malformed_entry_shows_up_as_absent_to_a_loopback_reader(
        enabled, monkeypatch):
    """Loopback needs no token, so `status` is always a readable place to check
    whether the env parsed the way you meant: `bogus` is simply absent."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64,10.0.0.0/99,bogus")
    body = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))).get(
        STATUS).json()
    assert body["allow_networks"] == ["192.168.2.64/32"]
    assert body["loopback_only"] is False


# --- guard order -------------------------------------------------------------

def test_inert_still_answers_503_to_an_allowlisted_token_holder(
        monkeypatch):
    """peer -> enabled -> token: an admitted, token-holding peer still learns
    the plugin is inert rather than getting a 200 or a 403."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    response = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient())),
                          PROXY).get(STATUS, headers=token_headers("tk"))
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
    assert response.status_code == 403
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
    """A genuine non-loopback peer on a genuine socket: refused while the
    allowlist is unset (even holding the token), then reachable only with the
    token once it is allowlisted."""
    peer = LAN_PEER
    engine = FakeEngineCoreClient(result={"version": 1})
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    with served_on_real_socket(build_app(FakeAsyncLLM(engine)), peer) as base:
        with real_client(base, source=peer) as client:
            # Sanity: this really is a non-loopback peer as far as uvicorn sees.
            assert not ipaddress.ip_address(peer).is_loopback
            assert client.get(STATUS).status_code == 403
            assert client.get(STATUS, headers=token_headers("tk")).status_code == 403
            assert engine.calls == []
            monkeypatch.setenv(config.ENV_ALLOW, peer)
            # Allowlisted, but a read still needs the token.
            assert client.get(STATUS).status_code == 403
            assert client.get(STATUS, headers=token_headers("nope")).status_code == 403
            assert engine.calls == []
            body = client.get(STATUS, headers=token_headers("tk")).json()
            assert body["loopback_only"] is False
            assert body["allow_networks"] == [f"{peer}/32"]
            assert client.get(SNAPSHOT, headers=token_headers("tk")).status_code == 200
            assert client.post(RELOAD, headers=token_headers("tk")).status_code == 200
    assert [name for name, _ in engine.calls] == [config.METHOD_SNAPSHOT,
                                                  config.METHOD_RELOAD_POLICY]


@needs_lan
def test_real_socket_forwarded_for_does_not_help_a_real_remote_peer(
        enabled, monkeypatch):
    """The forgeable-header attack against a real socket: a real LAN peer that
    claims to be 127.0.0.1 is refused, with the token and without it."""
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
                response = client.get(STATUS, headers=headers)
                assert response.status_code == 403, response.text
                assert peer in response.json()["detail"]
            assert engine.calls == []


# --- inert by default -------------------------------------------------------

@pytest.mark.parametrize("path,method", PLUGIN_ROUTES)
def test_inert_answers_503_even_on_loopback(method, path, monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")     # token does not unlock it
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

def test_reload_is_refused_without_a_token(enabled):
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(RELOAD).status_code == 403
    assert client.post(RELOAD, headers=token_headers("tk")).status_code == 403
    assert engine.calls == []


def test_reload_requires_the_right_token(enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(RELOAD, headers=token_headers("nope")).status_code == 403
    assert client.post(RELOAD, headers=token_headers("")).status_code == 403
    assert engine.calls == []
    response = client.post(RELOAD, headers=token_headers("tk"))
    assert response.status_code == 200
    assert engine.calls == [(config.METHOD_RELOAD_POLICY, ())]


def test_reload_takes_no_arguments_on_the_engine_side(enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    client.post(RELOAD, headers=token_headers("tk"), json={})
    client.post(RELOAD, headers=token_headers("tk"))
    assert engine.calls == [(config.METHOD_RELOAD_POLICY, ())] * 2


def test_reload_rejects_a_foreign_path_and_never_reaches_the_engine(
        enabled, monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    for candidate in ("/etc/passwd", "os.path", "../elsewhere/policy.py",
                      str(tmp_path / "policy.py"), "policy.py?",
                      str(config.canonical_policy_path().parent / "evil.py")):
        response = client.post(RELOAD, headers=token_headers("tk"),
                               json={"path": candidate})
        assert response.status_code == 400, (candidate, response.text)
        assert "only loadable policy" in response.json()["detail"]
    assert engine.calls == []


def test_reload_rejects_unknown_keys_and_non_objects(enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"ok": True})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    assert client.post(RELOAD, headers=token_headers("tk"),
                       json={"module": "os"}).status_code == 400
    assert client.post(RELOAD, headers=token_headers("tk"),
                       json=[1, 2]).status_code == 400
    assert client.post(RELOAD, headers=token_headers("tk"),
                       data=b"{not json").status_code == 400
    assert engine.calls == []


def test_reload_accepts_the_canonical_path_and_reports_both_sides(
        enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"ok": True, "applied": {"v": 2}})
    response = client_for(build_app(FakeAsyncLLM(engine))).post(
        RELOAD, headers=token_headers("tk"),
        json={"path": str(config.canonical_policy_path())})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == {"ok": True, "applied": {"v": 2}}
    assert body["request"]["path_on_wire"] is True
    assert body["request"]["validated_path"] == str(config.canonical_policy_path())


def test_reload_reports_the_peer_that_asked(enabled, monkeypatch, caplog):
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    engine = FakeEngineCoreClient(result={"ok": True})
    with caplog.at_level("INFO"):
        client_for(build_app(FakeAsyncLLM(engine)), ("127.0.0.1", 9999)).post(
            RELOAD, headers=token_headers("tk"))
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
