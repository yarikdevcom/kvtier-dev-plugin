"""HTTP-side half: the `vllm.endpoint_plugins` entry point.

Three routes, all under `/plugins/kvtier` (the prefix is baked into the
APIRouter, not added at mount time, so it cannot be forgotten by whoever
mounts it):

    GET  /plugins/kvtier/status         what is live, which file is the policy
    GET  /plugins/kvtier/snapshot       the scheduler-side tier view, live
    POST /plugins/kvtier/policy/reload  reload <package dir>/policy.py

The route to the engine is the async utility call
(`call_utility_async`, vllm/v1/engine/core_client.py:1199-1214), the same hop
`/is_sleeping` uses to read scheduler state
(vllm/entrypoints/serve/dev/sleep/api_router.py:41-44 ->
vllm/v1/engine/core.py:1035-1037).

SECURITY. The engine binds 0.0.0.0:8000 and `--api-key` covers only the /v1,
/v2, /inference and /cohere prefixes (docs/usage/security.md:349), so a plugin
route is unauthenticated on the LAN unless the plugin says otherwise. This one
does:

  (a) everything is under /plugins/kvtier, never /v1/*;
  (b) EVERY request is refused unless the ASGI peer address is loopback OR is
      contained in VLLM_E2_TIER_DEV_ALLOW (a comma-separated list of addresses
      or CIDRs; UNSET/EMPTY = LOOPBACK ONLY, the fail-closed default). The
      check reads `scope["client"]` and nothing else - never X-Forwarded-For,
      never X-Real-IP. A forged header grants nothing (see
      tests/test_routes.py::test_spoofed_forwarded_for_from_remote_peer_is_403).
      Caveat for the operator: do not put uvicorn's proxy-headers middleware in
      front of this, because that middleware REWRITES scope["client"] from
      X-Forwarded-For and would hand the forgeability back.
  (c) the one mutating verb requires VLLM_E2_TIER_DEV_TOKEN, compared with
      hmac.compare_digest, and is refused when that env is unset; AND, because
      the surface is now reachable off-box, a NON-LOOPBACK peer must present
      that same token on every request, reads included. Loopback needs no token
      for reads, so `ssh llm curl 127.0.0.1` keeps working. The transport is
      plaintext HTTP, so the allowlist is the compensating control for the
      sniffable bearer token: keep it as narrow as a single address.
  (d) the mutating verb takes no path on the engine side at all: the engine
      reloads the single file this package owns, and the HTTP layer separately
      validates any `path` the caller sends so it can only ever be that file.

Timeout: nothing in the engine bounds a utility call (the sync client blocks on
`future.result()`, core_client.py:971, and the async client just awaits its
future), so the HTTP layer imposes VLLM_E2_TIER_DEV_TIMEOUT and answers 504.
Cancelling that await is safe: when the engine eventually answers,
`_process_utility_output` pops the future and swallows the resulting
InvalidStateError (core_client.py:847-865).
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config

logger = logging.getLogger("kvtier_dev_plugin.endpoint")

TOKEN_HEADER = "x-kvtier-dev-token"  # noqa: S105 - a header NAME, not a secret
STATE_KEY = "kvtier_dev"

router = APIRouter(prefix=config.ROUTE_PREFIX, tags=["kvtier-dev"])


class GuardDenied(Exception):
    """A guard refused the request. Carries the HTTP status to answer with."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# --- guards (framework-free: they take a scope / a header mapping) ---------

def loopback_denial(scope) -> str | None:
    """None if the peer is loopback, else the reason it is not.

    Reads ONLY the ASGI scope's client address, which is the peer of the TCP
    connection. Request headers are never consulted here: an attacker who can
    set X-Forwarded-For can set it to 127.0.0.1.
    """
    client = scope.get("client") if isinstance(scope, dict) else None
    if not isinstance(client, (tuple, list)) or not client or client[0] is None:
        # No peer address (a unix socket, or a non-standard ASGI server): we
        # cannot prove loopback, so we do not serve.
        return "ASGI scope carries no client address, cannot prove loopback"
    raw = str(client[0])
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        return f"peer address {raw!r} is not a parseable IP address"
    # A dual-stack listener reports an IPv4 peer as ::ffff:a.b.c.d. Unwrap it
    # so 127.0.0.1 over IPv6 still counts as loopback, independent of the
    # ipaddress version's treatment of mapped addresses.
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    if not address.is_loopback:
        return f"peer {address} is not loopback"
    return None


def require_loopback(scope) -> None:
    denial = loopback_denial(scope)
    if denial is not None:
        raise GuardDenied(403, denial)


def _peer_address_unwrapped(scope):
    """The peer's IP with the IPv4-mapped form unwrapped, or None if absent or
    unparseable. Same unwrap rule as loopback_denial(), so the allowlist and
    the loopback test can never disagree about who the peer is.
    """
    client = scope.get("client") if isinstance(scope, dict) else None
    if not isinstance(client, (tuple, list)) or not client or client[0] is None:
        return None
    try:
        address = ipaddress.ip_address(str(client[0]))
    except ValueError:
        return None
    mapped = getattr(address, "ipv4_mapped", None)
    return address if mapped is None else mapped


def peer_is_loopback(scope) -> bool:
    """True only when loopback_denial() says so - the one definition of
    loopback, so 'needs a token' and 'is loopback' cannot drift apart."""
    return loopback_denial(scope) is None


def peer_denial(scope) -> str | None:
    """None if this peer may reach these routes at all, else the why-not.

    Loopback is accepted unconditionally (the pre-allowlist behaviour). Anyone
    else is accepted only if contained in one of config.allow_networks(); an
    empty allowlist therefore means loopback-only, and a malformed entry admits
    nobody. Headers are never consulted here either: like loopback_denial(),
    this reads the socket peer and nothing a client can type.
    """
    not_loopback = loopback_denial(scope)
    if not_loopback is None:
        return None
    address = _peer_address_unwrapped(scope)
    if address is not None:
        # A v4 address is never `in` a v6 network and vice versa (ipaddress
        # returns False rather than raising), so no version guard is needed.
        for network in config.allow_networks():
            if address in network:
                return None
    # Name the variable, never its contents: a refused peer learns nothing it
    # did not already have (tests/test_routes.py::test_remote_peer_gets_no_*).
    return (f"{not_loopback}, and it is not in the {config.ENV_ALLOW} "
            "allowlist")


def require_peer(scope) -> None:
    denial = peer_denial(scope)
    if denial is not None:
        raise GuardDenied(403, denial)


def require_bearer(headers, verb: str) -> None:
    """The token check, shared by the mutating verb and by any read from a
    non-loopback peer. Fail closed: an unset/blank token refuses."""
    expected = config.mutating_token()
    if not expected:
        raise GuardDenied(403, f"{verb} refused: {config.ENV_TOKEN} is not set")
    presented = ""
    if headers is not None:
        presented = headers.get(TOKEN_HEADER, "") or ""
    if not presented or not hmac.compare_digest(presented, expected):
        raise GuardDenied(403, f"missing or wrong {TOKEN_HEADER}")


def require_mutating_token(headers) -> None:
    require_bearer(headers, "reload")


def require_token_for_remote_read(scope, headers) -> None:
    """Reads are token-free from loopback only.

    Off-box traffic is plaintext HTTP on a shared LAN segment, so once the
    allowlist opened the surface the token had to become a read requirement
    too; loopback stays token-free so the operator's `ssh llm curl 127.0.0.1`
    workflow is unchanged.
    """
    if peer_is_loopback(scope):
        return
    require_bearer(headers, "read from a non-loopback peer")


def _guard_read(request: Request) -> None:
    # Order is the contract: prove the peer, then prove the plugin is even on,
    # then prove the token where the token is required.
    require_peer(request.scope)
    if not config.enabled():
        raise HTTPException(status_code=503,
                            detail=f"inert: {config.ENV_ENABLE} is not enabled")
    require_token_for_remote_read(request.scope, request.headers)


# --- engine hop -------------------------------------------------------------

def engine_utility_caller(request: Request):
    """Resolve something with `call_utility_async`, or None.

    `app.state.engine_client` satisfies the `EngineClient` protocol, which does
    NOT expose call_utility_async; the real object (AsyncLLM,
    vllm/v1/engine/async_llm.py:169) holds the EngineCore client that does.
    Both spots are checked so this does not care which one it is handed.
    """
    client = None
    state_entry = getattr(request.app.state, STATE_KEY, None)
    if isinstance(state_entry, dict):
        client = state_entry.get("engine_client")
    if client is None:
        client = getattr(request.app.state, "engine_client", None)
    for candidate in (client, getattr(client, "engine_core", None)):
        caller = getattr(candidate, "call_utility_async", None)
        if callable(caller):
            return caller
    return None


async def call_engine(request: Request, method: str):
    caller = engine_utility_caller(request)
    if caller is None:
        raise HTTPException(status_code=503,
                            detail="no engine utility channel in this process "
                                   "(engine_client absent or not an MP client)")
    budget = config.timeout_seconds()
    try:
        return await asyncio.wait_for(caller(method), timeout=budget)
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"engine did not answer {method} within {budget}s; the "
                   "engine-side handler is a pure read, so no answer means "
                   "the EngineCore loop is not turning") from None
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - the engine said no; report as data
        raise HTTPException(status_code=502,
                            detail=f"{method} raised "
                                   f"{type(exc).__name__}: {exc}"[:300]) from None


# --- payload helpers --------------------------------------------------------

def status_payload() -> dict:
    try:
        target = str(config.validate_policy_target(config.canonical_policy_path()))
        loadable = True
        refusal = None
    except config.PolicyPathError as exc:
        target = str(config.canonical_policy_path())
        loadable = False
        refusal = str(exc)
    allowed = config.allow_networks()
    return {
        "plugin": config.PLUGIN_NAME,
        "enabled": config.enabled(),
        "enable_env": config.ENV_ENABLE,
        "route_prefix": config.ROUTE_PREFIX,
        "engine_methods": [config.METHOD_SNAPSHOT, config.METHOD_RELOAD_POLICY],
        "token_configured": config.token_configured(),
        "token_env": config.ENV_TOKEN,
        "timeout_seconds": config.timeout_seconds(),
        "timeout_env": config.ENV_TIMEOUT,
        "policy_target": {"path": target, "loadable": loadable, "refusal": refusal},
        # New: what an operator can actually reach from. `allow_networks` is
        # the PARSED list (a dropped entry is absent, which is how a typo shows
        # up here), and `loopback_only` is the whole answer to "is this open?".
        "allow_env": config.ENV_ALLOW,
        "allow_networks": [str(network) for network in allowed],
        "loopback_only": not allowed,
        "non_loopback_peers_require_token": True,
        "guard": "ASGI scope['client'] must be a loopback IP or be contained "
                 "in VLLM_E2_TIER_DEV_ALLOW; forwarded headers are ignored; a "
                 "non-loopback peer must also send the token header",
    }


async def read_json_object(request: Request):
    """{} for an empty body, the object, or None if it is not a JSON object."""
    raw = await request.body()
    if not raw or raw.isspace():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


# --- routes -----------------------------------------------------------------

@router.get("/status")
async def route_status(request: Request):
    """Local facts only - no engine call, so it answers while the engine is idle
    or wedged, which is exactly when you want it."""
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    return JSONResponse(content=status_payload())


@router.get("/snapshot")
async def route_snapshot(request: Request):
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    result = await call_engine(request, config.METHOD_SNAPSHOT)
    return JSONResponse(content=result if isinstance(result, dict)
                        else {"result": result})


@router.post("/policy/reload")
async def route_reload_policy(request: Request):
    # Order matters: prove the peer is allowed (loopback or allowlisted), then
    # prove we are even on, then prove the caller holds the token. Nothing is
    # leaked to a peer that is not allowed.
    try:
        require_peer(request.scope)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    if not config.enabled():
        raise HTTPException(status_code=503,
                            detail=f"inert: {config.ENV_ENABLE} is not enabled")
    try:
        require_mutating_token(request.headers)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None

    body = await read_json_object(request)
    if body is None:
        raise HTTPException(status_code=400,
                            detail="body must be empty or a JSON object")
    unknown = sorted(key for key in body if key != "path")
    if unknown:
        raise HTTPException(status_code=400,
                            detail=f"unsupported request keys: {unknown}")

    supplied = "path" in body
    candidate = body["path"] if supplied else config.canonical_policy_path()
    try:
        validated = str(config.validate_policy_target(candidate))
    except config.PolicyPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The validated value is REPORTED only. The engine-side method takes no
    # argument, so the wire never selects a file: the engine always reloads
    # its own <package dir>/policy.py. This check exists to tell the caller
    # "you asked for something we will never load" instead of silently
    # ignoring it.
    logger.info("[KVTIER_DEV] policy reload requested path=%s from_scope_client=%s",
                validated, request.scope.get("client"))
    result = await call_engine(request, config.METHOD_RELOAD_POLICY)
    return JSONResponse(content={
        "request": {"path_on_wire": supplied, "validated_path": validated},
        "engine": result,
    })


# --- the plugin object ------------------------------------------------------

class KVTierDevEndpointPlugin:
    """`vllm.endpoint_plugins` factory (see
    vllm/plugins/endpoint_plugins/interface.py).

    `required_tasks = None`: this surface has no task requirement, it answers
    503 until VLLM_E2_TIER_DEV is on. `init_state` handles a None
    `engine_client` (the CPU-only render server) by leaving the engine routes
    to answer 503.
    """

    name: str = config.PLUGIN_NAME
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        app.include_router(router)
        logger.info("kvtier-dev-plugin: routes attached under %s "
                    "(inert until %s is set)", config.ROUTE_PREFIX,
                    config.ENV_ENABLE)

    # `args` is part of the EndpointPlugin protocol; this plugin needs
    # nothing from it.
    async def init_state(self, engine_client, state, args) -> None:  # noqa: ARG002
        state.kvtier_dev = {  # noqa: B010 - attribute name is the contract
            "engine_client": engine_client,
            "status": status_payload(),
        }
        if engine_client is None:
            logger.info("kvtier-dev-plugin: no engine client (render server); "
                        "%s/snapshot and %s/policy/reload will answer 503",
                        config.ROUTE_PREFIX, config.ROUTE_PREFIX)
