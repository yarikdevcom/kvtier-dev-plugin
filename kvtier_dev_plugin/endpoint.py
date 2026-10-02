"""HTTP-side half: the `vllm.endpoint_plugins` entry point.

The whole surface, all under `/plugins/kvtier` (the prefix is baked into the
APIRouter, not added at mount time, so it cannot be forgotten by whoever mounts
it):

    GET  /plugins/kvtier/status          what is live; makes NO engine call
    GET  /plugins/kvtier/snapshot        the scheduler-side tier view, live
    POST /plugins/kvtier/snapshot        the same, body forwarded as one dict
    GET|POST /plugins/kvtier/kvconfig    the KV-cache geometry, including every
                                         group's divisibility verdict
    GET|POST /plugins/kvtier/block       one block, by id (and group)
    GET|POST /plugins/kvtier/request     one request by id; no id = the id list
    GET  /plugins/kvtier/trace           the trace ring's newest events
    POST /plugins/kvtier/trace           arm / disarm / clear / read   [TOKEN]
    GET  /plugins/kvtier/guard           guard-mode state                   [read]
    POST /plugins/kvtier/guard           arm / disarm / fetch captures   [TOKEN]
    POST /plugins/kvtier/policy/reload   reload <package dir>/policy.py  [TOKEN]

GET takes its arguments from query parameters, POST from a JSON object body.
Both are validated against the SAME per-route spec (`ARG_SPECS`) before anything
reaches the engine, so a bound cannot be evaded by choosing a method, and an
unknown key is a 400 that never reaches the engine.

HOW AN ARGUMENT REACHES THE ENGINE
----------------------------------
The UTILITY channel passes arguments straight through: `call_utility_async
(method, *args)` (core_client.py:1199-1214) encodes `(client_index, call_id,
method, args)` and the engine calls `getattr(self, method_name)(*args)`
(core.py:1669-1671). So one JSON body becomes ONE positional argument: a plain
 dict of JSON primitives, which msgpack encodes without help.

Every engine method this plugin owns therefore has the signature `(self,
args=None)`, for two reasons. Old callers keep working: a UTILITY message with no
arguments still calls `method()`, which is what every GET sends and what every
live caller has always sent, and `call_engine` passes NO positional argument when
there is no body. And the arity is uniform, which matters because the fork asserts
`len(args) <= len(signature(method).parameters)` while converting msgspec
arguments (core.py:1731-1744) - a method that accepted nothing would turn a body
into an AssertionError inside the engine loop.

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
  (c) the mutating verbs require VLLM_E2_TIER_DEV_TOKEN, compared with
      hmac.compare_digest, and are refused when that env is unset; AND, because
      the surface is now reachable off-box, a NON-LOOPBACK peer must present
      that same token on every request, reads included. Loopback needs no token
      for reads, so `ssh llm curl 127.0.0.1` keeps working. `/trace` and `/guard`
      can change engine dev state, so they require the token in EVERY shape -
      even a loopback `POST /trace {"op": "read"}` needs it, deliberately: a
      route that can change dev state is token-gated as a whole, not per field.
      The transport is plaintext HTTP, so the allowlist is the compensating
      control for the sniffable bearer token: keep it as narrow as a single
      address.
  (d) the mutating verbs cannot name a file, a method or a module. The engine
      reloads the single file this package owns, and the HTTP layer separately
      validates any `path` the caller sends so it can only ever be that file.
      Which engine method a route reaches is fixed per route and never comes from
      the wire: `getattr(self, method_name)` on the EngineCore would otherwise be
      an arbitrary-method-call primitive handed to whoever holds a token.

GUARD MODE IS A DEVELOPMENT FACILITY. `POST /plugins/kvtier/guard {"armed":
true}` makes the tier's entry points turn an exception into a record plus a safe
decline, which is how you develop against a live engine without an outage. It can
also MASK REAL BUGS: a declined store is silent, and a decline can leave tier
state half-advanced. It is off at every boot (there is no env var for it), its
state is echoed as `guard_mode` by `status` and by every verb, and it must be
DISARMED before any production boot.

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
import time

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


def require_dev_state_token(headers) -> None:
    """The token for a route that CAN change dev state, in every shape.

    `/trace` and `/guard` are mutating-capable verbs, so the whole route is
    token-gated rather than the individual `op`: a caller who cannot produce the
    token cannot even ask for a read through the same door that arms guard mode.
    Loopback is not an exemption here - exactly like the reload verb.
    """
    require_bearer(headers, "dev-state verb")


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


async def call_engine(request: Request, method: str, args: dict | None = None):
    """One UTILITY hop. `args` is forwarded as ONE positional argument.

    `args=None` sends the message with NO arguments at all - not an empty dict -
    because that is what every caller of this plugin has always sent and what
    keeps a no-argument engine call working (`method()` versus `method({})`).
    """
    caller = engine_utility_caller(request)
    if caller is None:
        raise HTTPException(status_code=503,
                            detail="no engine utility channel in this process "
                                   "(engine_client absent or not an MP client)")
    budget = config.timeout_seconds()
    try:
        if args is None:
            return await asyncio.wait_for(caller(method), timeout=budget)
        return await asyncio.wait_for(caller(method, args), timeout=budget)
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


# --- argument specs (one contract for GET query strings and POST bodies) -----

MAX_ARG_STRING_CHARS = 128

# Per route: the allowed keys and how each is validated. `kind` is `int`, `str`,
# `bool` or `enum`; `enum` carries `values`. Anything not listed here is a 400,
# so a new key can only exist if it is added to this table. That is the whole
# argument attack surface: JSON primitives, from a fixed key set, within bounds,
# to a fixed engine method.
ARG_SPECS: dict[str, dict] = {
    config.ROUTE_SNAPSHOT: {},
    config.ROUTE_KVCONFIG: {},
    config.ROUTE_BLOCK: {"block_id": {"kind": "int", "min": 0, "max": 2 ** 31 - 1},
                         "group": {"kind": "int", "min": 0, "max": 63},
                         "ids_cap": {"kind": "int", "min": 1, "max": 1024}},
    config.ROUTE_REQUEST: {"request_id": {"kind": "str",
                                          "max_chars": MAX_ARG_STRING_CHARS},
                           "ids_cap": {"kind": "int", "min": 1, "max": 1024}},
    config.ROUTE_TRACE: {"op": {"kind": "enum", "values": ("read", "arm", "disarm",
                                                           "clear", "status")},
                         "depth": {"kind": "int", "min": 16, "max": 4096},
                         "limit": {"kind": "int", "min": 1, "max": 512},
                         "after_seq": {"kind": "int", "min": 0, "max": 2 ** 62}},
    config.ROUTE_GUARD: {"op": {"kind": "enum", "values": ("status", "arm",
                                                           "disarm", "captures")},
                         "armed": {"kind": "bool"},
                         "limit": {"kind": "int", "min": 1, "max": 4},
                         "traceback_chars": {"kind": "int", "min": 0,
                                             "max": 4000}},
    config.ROUTE_TIER: {"op": {"kind": "enum",
                               "values": ("status", "arm", "disarm")}},
}


class _ArgumentError(ValueError):
    """A request argument was refused. Carries the human-readable why."""


def _query_arguments(request: Request) -> dict:
    """Query parameters as a raw dict (repeated keys keep the LAST value)."""
    raw = {}
    try:
        items = request.query_params.multi_items()
    except AttributeError:            # a plain dict-like in a unit test
        items = list((request.query_params or {}).items())
    for key, value in items:
        raw[key] = value
    return raw


def validate_arguments(route: str, raw: dict) -> dict:
    """Turn raw wire values into the dict the engine is allowed to receive.

    Same table for GET (query strings, all strings) and POST (JSON, real types),
    so the bounds are identical whichever method a caller chose. Nothing here
    invents a default: an absent key stays absent, and the engine's own defaults
    stay the engine's business.
    """
    spec = ARG_SPECS.get(route, {})
    unknown = sorted(key for key in raw if key not in spec)
    if unknown:
        raise _ArgumentError(f"unsupported request keys: {unknown}")
    out: dict = {}
    for key, rule in spec.items():
        if key not in raw:
            continue
        value = raw[key]
        kind = rule["kind"]
        if kind == "int":
            if isinstance(value, bool):
                raise _ArgumentError(f"{key} must be an integer, not a boolean")
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    raise _ArgumentError(f"{key} must not be empty")
                try:
                    value = int(text)
                except ValueError:
                    raise _ArgumentError(
                        f"{key} must be an integer, got {text!r}") from None
            if not isinstance(value, int):
                raise _ArgumentError(f"{key} must be an integer, got "
                                     f"{type(value).__name__}")
            low, high = rule.get("min"), rule.get("max")
            if low is not None and value < low:
                raise _ArgumentError(f"{key} must be >= {low}")
            if high is not None and value > high:
                raise _ArgumentError(f"{key} must be <= {high}")
        elif kind == "str":
            if not isinstance(value, str):
                raise _ArgumentError(f"{key} must be a string, got "
                                     f"{type(value).__name__}")
            limit = rule.get("max_chars", MAX_ARG_STRING_CHARS)
            if len(value) > limit:
                raise _ArgumentError(f"{key} must be at most {limit} characters")
            if any(ord(char) < 32 for char in value):
                raise _ArgumentError(f"{key} must not contain control characters")
        elif kind == "bool":
            if isinstance(value, str):
                text = value.strip().lower()
                if text in ("1", "true", "yes", "on"):
                    value = True
                elif text in ("0", "false", "no", "off"):
                    value = False
                else:
                    raise _ArgumentError(f"{key} must be a boolean, got {text!r}")
            if not isinstance(value, bool):
                raise _ArgumentError(f"{key} must be a boolean")
        elif kind == "enum":
            if not isinstance(value, str):
                raise _ArgumentError(f"{key} must be one of "
                                     f"{list(rule['values'])}")
            text = value.strip().lower()
            if text not in rule["values"]:
                raise _ArgumentError(f"{key} must be one of "
                                     f"{list(rule['values'])}, got {text!r}")
            value = text
        out[key] = value
    return out


async def body_arguments(request: Request, route: str) -> dict:
    """The POST body, validated. 400 on anything that is not a valid object."""
    body = await read_json_object(request)
    if body is None:
        raise HTTPException(status_code=400,
                            detail="body must be empty or a JSON object")
    try:
        return validate_arguments(route, body)
    except _ArgumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def query_arguments(request: Request, route: str) -> dict:
    """The query string, validated against the same spec as the body."""
    try:
        return validate_arguments(route, _query_arguments(request))
    except _ArgumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# --- the dev-state mirror ---------------------------------------------------

DEV_STATE_KEYS = ("guard_mode", "trace_armed")
DEV_STATE_UNKNOWN_NOTE = (
    "no engine answer has been seen by this API process yet, so guard mode is "
    "UNKNOWN here, not false. The engine's own answer in any verb payload "
    "(snapshot/kvconfig/block/request/trace/guard) is authoritative; the "
    "default at every engine boot is guard_mode=false.")


def note_dev_state(request: Request, payload, source: str) -> None:
    """Mirror the engine's dev-state claims into app.state, best effort.

    WHY A MIRROR AT ALL: `status` deliberately makes no engine call (it must
    answer while the engine is wedged), yet guard mode has to be prominent in it.
    So `status` reports what this process has LEARNED from the engine, labelled
    with its epoch and age, and says `null` when it has learned nothing - it
    never guesses "false".
    """
    state = getattr(request.app.state, STATE_KEY, None)
    if not isinstance(state, dict) or not isinstance(payload, dict):
        return
    seen = {key: payload[key] for key in DEV_STATE_KEYS
            if key in payload and isinstance(payload[key], bool)}
    if not seen:
        return
    entry = state.setdefault("dev_state", {})
    entry.update(seen)
    entry["source"] = source
    entry["epoch"] = payload.get("epoch") if isinstance(payload.get("epoch"),
                                                           str) else None
    entry["observed_at"] = time.time()


async def engine_call(request: Request, method: str, args: dict | None = None,
                      source: str = "engine read"):
    """`call_engine` plus the dev-state mirror, so no route forgets it.

    An EMPTY argument dict is sent as NO argument at all: `{}` on the wire would
    otherwise be a positional `{}` to a method that has never seen one, and "no
    arguments" and "no keys" mean the same thing here.
    """
    if not args:
        args = None
    result = await call_engine(request, method, args)
    note_dev_state(request, result if isinstance(result, dict) else None, source)
    return result


def as_json(result):
    """The engine's dict, or a wrapper for anything unexpected. Never a 500."""
    return JSONResponse(content=result if isinstance(result, dict)
                        else {"result": result})


def dev_state_view(request: Request) -> dict:
    """The `status` view of guard/trace state, honest about never having asked."""
    state = getattr(request.app.state, STATE_KEY, None)
    entry = state.get("dev_state") if isinstance(state, dict) else None
    if not isinstance(entry, dict) or not entry:
        return {"guard_mode": None, "trace_armed": None, "source": None,
                "epoch": None, "observed_at": None, "age_seconds": None,
                "note": DEV_STATE_UNKNOWN_NOTE}
    observed = entry.get("observed_at")
    age = None
    if isinstance(observed, (int, float)):
        age = round(max(0.0, time.time() - observed), 3)
    return {"guard_mode": entry.get("guard_mode"),
            "trace_armed": entry.get("trace_armed"),
            "source": entry.get("source"), "epoch": entry.get("epoch"),
            "observed_at": observed, "age_seconds": age,
            "note": ("mirrored from the engine's own answers in this API "
                     "process; null means never observed. A plugin restart "
                     "clears this, not the engine's state.")}


# --- payload helpers --------------------------------------------------------

def status_payload(dev_state: dict | None = None) -> dict:
    try:
        target = str(config.validate_policy_target(config.canonical_policy_path()))
        loadable = True
        refusal = None
    except config.PolicyPathError as exc:
        target = str(config.canonical_policy_path())
        loadable = False
        refusal = str(exc)
    allowed = config.allow_networks()
    mirrored = dev_state if isinstance(dev_state, dict) else {
        "guard_mode": None, "trace_armed": None, "note": DEV_STATE_UNKNOWN_NOTE}
    return {
        "plugin": config.PLUGIN_NAME,
        "enabled": config.enabled(),
        "enable_env": config.ENV_ENABLE,
        "route_prefix": config.ROUTE_PREFIX,
        "engine_methods": list(config.METHODS),
        "routes": [config.ROUTE_PREFIX + route for route in config.ROUTES],
        "dev_state_routes": [config.ROUTE_PREFIX + route
                             for route in config.DEV_STATE_ROUTES],
        "arguments": {"GET": "query parameters", "POST": "a JSON object body, "
                        "forwarded to the engine as ONE dict argument",
                      "validated_by": "ARG_SPECS in this module; an unknown key "
                                      "is a 400 and never reaches the engine"},
        "token_configured": config.token_configured(),
        "token_env": config.ENV_TOKEN,
        "timeout_seconds": config.timeout_seconds(),
        "timeout_env": config.ENV_TIMEOUT,
        "policy_target": {"path": target, "loadable": loadable, "refusal": refusal},
        # What an operator can actually reach from. `allow_networks` is the
        # PARSED list (a dropped entry is absent, which is how a typo shows up
        # here), and `loopback_only` is the whole answer to "is this open?".
        "allow_env": config.ENV_ALLOW,
        "allow_networks": [str(network) for network in allowed],
        "loopback_only": not allowed,
        "non_loopback_peers_require_token": True,
        "guard": "ASGI scope['client'] must be a loopback IP or be contained "
                 "in VLLM_E2_TIER_DEV_ALLOW; forwarded headers are ignored; a "
                 "non-loopback peer must also send the token header",
        # GUARD MODE, up top and impossible to miss, and the trace ring beside
        # it. `null` here means THIS API PROCESS HAS NEVER SEEN AN ENGINE ANSWER
        # - never "off": status makes no engine call, so it cannot know, and a
        # confident lie about guard mode is the one thing this field must never
        # do. `source`/`age_seconds` say how current the claim is.
        "guard_mode": mirrored.get("guard_mode"),
        "trace_armed": mirrored.get("trace_armed"),
        "dev_state": mirrored,
        "worker_diagnostics": {
            "env": config.ENV_DIAG_DIR,
            "configured": config.diagnostics_dir() is not None,
            # str(), never the Path: this payload goes through json.dumps, and a
            # Path is not serialisable - a 500 from the one route that must always
            # answer.
            "dir": (str(config.diagnostics_dir())
                    if config.diagnostics_dir() is not None else None),
            "interval_steps": config.worker_write_interval_steps(),
            "interval_env": config.ENV_DIAG_INTERVAL,
            "note": "written by the WORKER processes, one file per rank; this "
                    "process only reports the configuration. The live per-rank "
                    "truth is in those files (the proxy host's reader reads them).",
        },
        "tier": {
            "route": config.ROUTE_PREFIX + config.ROUTE_TIER,
            "note": "the KV tier's runtime kill switch and the status of the "
                    "upstream PR #56810 backport (excluding non-prefix-cacheable "
                    "KV groups from store/load, which a hybrid geometry needs or "
                    "the store path asserts). Disarmed, the connector reports no "
                    "prefix hit, hands the workers empty store/load lists and "
                    "claims no pending work, so the engine serves from GPU KV "
                    "alone - no engine restart. `tier.patched: false` means the "
                    "divisibility fix is NOT live and a hybrid geometry will "
                    "still crash the store path; arming the tier does not change "
                    "that.",
            "ops": ["status", "arm", "disarm"],
        },
        "development_facilities": {
            "guard_mode": "DEVELOPMENT ONLY. Armed, it turns a KV-tier exception "
                          "into a captured record plus a safe decline. Off at "
                          "every boot (no env var exists for it); arm it with "
                          "POST /plugins/kvtier/guard and DISARM it before any "
                          "production boot. It never swallows a BaseException.",
            "trace": "bounded ring, opt-in, nothing wrapped while disarmed",
        },
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

@router.get(config.ROUTE_STATUS)
async def route_status(request: Request):
    """Local facts only - no engine call, so it answers while the engine is idle
    or wedged, which is exactly when you want it. `guard_mode` is therefore the
    last value this process MIRRORED from an engine answer (with its age), or
    null when nothing has been seen - never a guessed `false`.
    """
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    return JSONResponse(content=status_payload(dev_state_view(request)))


@router.get(config.ROUTE_SNAPSHOT)
async def route_snapshot(request: Request):
    """The scheduler-side tier view. Takes no arguments; an unknown query
    parameter is a 400, so the strictness is symmetrical with POST."""
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = query_arguments(request, config.ROUTE_SNAPSHOT)
    return as_json(await engine_call(request, config.METHOD_SNAPSHOT, args,
                                     source="snapshot read"))


@router.post(config.ROUTE_SNAPSHOT)
async def route_snapshot_post(request: Request):
    """The same view. A body is validated (it may only be empty) and forwarded."""
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = await body_arguments(request, config.ROUTE_SNAPSHOT)
    return as_json(await engine_call(request, config.METHOD_SNAPSHOT, args,
                                     source="snapshot read"))


@router.get(config.ROUTE_KVCONFIG)
async def route_kvconfig(request: Request):
    """The KV-cache geometry: block sizes, spec classes, divisibility per group.
    Read-only; no arguments are accepted, on either method."""
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = query_arguments(request, config.ROUTE_KVCONFIG)
    return as_json(await engine_call(request, config.METHOD_KVCONFIG, args,
                                     source="kvconfig read"))


@router.post(config.ROUTE_KVCONFIG)
async def route_kvconfig_post(request: Request):
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = await body_arguments(request, config.ROUTE_KVCONFIG)
    return as_json(await engine_call(request, config.METHOD_KVCONFIG, args,
                                     source="kvconfig read"))


@router.get(config.ROUTE_BLOCK)
async def route_block(request: Request):
    """`GET /plugins/kvtier/block?block_id=412&group=5` - the block-inspector
    shape, straight out of a browser or one curl."""
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = query_arguments(request, config.ROUTE_BLOCK)
    return as_json(await engine_call(request, config.METHOD_INSPECT, args,
                                     source="block inspect"))


@router.post(config.ROUTE_BLOCK)
async def route_block_post(request: Request):
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = await body_arguments(request, config.ROUTE_BLOCK)
    return as_json(await engine_call(request, config.METHOD_INSPECT, args,
                                     source="block inspect"))


@router.get(config.ROUTE_REQUEST)
async def route_request(request: Request):
    """`?request_id=cmpl-...` for one request; no id at all lists the ids the
    engine currently has (capped)."""
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = query_arguments(request, config.ROUTE_REQUEST)
    return as_json(await engine_call(request, config.METHOD_INSPECT, args,
                                     source="request inspect"))


@router.post(config.ROUTE_REQUEST)
async def route_request_post(request: Request):
    try:
        _guard_read(request)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    args = await body_arguments(request, config.ROUTE_REQUEST)
    return as_json(await engine_call(request, config.METHOD_INSPECT, args,
                                     source="request inspect"))


async def _guard_dev_state_route(request: Request) -> None:
    """Peer, then enabled, then ALWAYS the token - see require_dev_state_token."""
    try:
        require_peer(request.scope)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None
    if not config.enabled():
        raise HTTPException(status_code=503,
                            detail=f"inert: {config.ENV_ENABLE} is not enabled")
    try:
        require_dev_state_token(request.headers)
    except GuardDenied as denied:
        raise HTTPException(status_code=denied.status_code,
                            detail=denied.detail) from None


@router.get(config.ROUTE_TRACE)
async def route_trace(request: Request):
    """The trace ring's newest events. Token required even for this read: the
    same door arms the recorder."""
    await _guard_dev_state_route(request)
    raw = query_arguments(request, config.ROUTE_TRACE)
    args = dict(raw)
    args.setdefault("op", "read")
    return as_json(await engine_call(request, config.METHOD_TRACE, args,
                                     source="trace read"))


@router.post(config.ROUTE_TRACE)
async def route_trace_post(request: Request):
    """`{op: arm|disarm|clear|read|status, depth?, limit?, after_seq?}`.

    `arm` INSTALLS the scheduler/tier observers (and can force guard mode on, so
    a wrapped method that faults is recorded rather than raised); `disarm`
    removes every wrapper this plugin owns outside the init-time ones.
    """
    await _guard_dev_state_route(request)
    args = await body_arguments(request, config.ROUTE_TRACE)
    return as_json(await engine_call(request, config.METHOD_TRACE, args,
                                     source="trace verb"))


@router.get(config.ROUTE_GUARD)
async def route_guard(request: Request):
    """Guard-mode state and its error histogram. Token required, as for /trace."""
    await _guard_dev_state_route(request)
    args = dict(query_arguments(request, config.ROUTE_GUARD))
    args.setdefault("op", "status")
    return as_json(await engine_call(request, config.METHOD_GUARD, args,
                                     source="guard status"))


@router.post(config.ROUTE_GUARD)
async def route_guard_post(request: Request):
    """GUARD MODE - DEVELOPMENT ONLY.

    `{op: status|arm|disarm|captures, armed?, limit?, traceback_chars?}`. Armed,
    the KV-tier entry points turn an `Exception` into a captured record and a
    safe decline, which is what makes a live dev engine survivable. It can also
    MASK REAL BUGS (a declined store is silent, and a decline can leave tier
    state half-advanced), so it is OFF at every boot, there is deliberately no
    env var to turn it on, and it must be disarmed before any production boot.
    It NEVER swallows a `BaseException`: cancellation, SystemExit and
    KeyboardInterrupt always continue on their way.
    """
    await _guard_dev_state_route(request)
    args = await body_arguments(request, config.ROUTE_GUARD)
    return as_json(await engine_call(request, config.METHOD_GUARD, args,
                                     source="guard verb"))


@router.get(config.ROUTE_TIER)
async def route_tier(request: Request):
    """KV-tier state: is the PR #56810 fix live, and is the tier armed? Token
    required, because the same door disarms the connector."""
    await _guard_dev_state_route(request)
    args = dict(query_arguments(request, config.ROUTE_TIER))
    args.setdefault("op", "status")
    return as_json(await engine_call(request, config.METHOD_TIER, args,
                                     source="tier status"))


@router.post(config.ROUTE_TIER)
async def route_tier_post(request: Request):
    """`{op: status|arm|disarm}` - the tier's runtime kill switch.

    `disarm` makes the connector a STRICT no-op through the manager methods the
    plugin wraps: it reports no prefix hit, hands the workers empty store/load
    lists, and claims no pending work, so the engine serves from GPU KV alone.
    No restart, no reboot of a 4-minute boot. `arm` puts the scheduling back.

    What it cannot undo is reported honestly in the payload under
    `tier.disarm_limits`: an in-flight copy already handed to the workers
    finishes, blocks already parked in host RAM stay parked (re-arm makes them
    reachable again), and the worker PROCESS is never contacted - its no-op is
    delivered as empty metadata. Nothing here touches the backport itself: if
    `tier.patched` is false, a hybrid geometry will still crash the store path
    and disarming does not prevent that.
    """
    await _guard_dev_state_route(request)
    args = await body_arguments(request, config.ROUTE_TIER)
    return as_json(await engine_call(request, config.METHOD_TIER, args,
                                     source="tier verb"))


@router.post(config.ROUTE_RELOAD)
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
    result = await engine_call(request, config.METHOD_RELOAD_POLICY, None,
                               source="policy reload")
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
            # The dev-state mirror starts EMPTY: guard mode is unknown until an
            # engine answer says so (see DEV_STATE_UNKNOWN_NOTE).
            "dev_state": {},
        }
        if engine_client is None:
            logger.info("kvtier-dev-plugin: no engine client (render server); "
                        "%s/snapshot and %s/policy/reload will answer 503",
                        config.ROUTE_PREFIX, config.ROUTE_PREFIX)
