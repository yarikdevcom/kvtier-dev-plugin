"""HTTP-side half: the `vllm.endpoint_plugins` entry point.

The whole surface, all under `/plugins/kvtier` (the prefix is baked into the
APIRouter, not added at mount time, so it cannot be forgotten by whoever mounts
it):

    GET  /plugins/kvtier/status          what is live, including WHICH CODE THIS
                                         PROCESS loaded (`plugin_version`); makes
                                         NO engine call
    GET  /plugins/kvtier/snapshot        the scheduler-side tier view, live
    POST /plugins/kvtier/snapshot        the same, body forwarded as one dict
    GET|POST /plugins/kvtier/kvconfig    the KV-cache geometry, including every
                                         group's divisibility verdict
    GET|POST /plugins/kvtier/block       one block, by id (and group): the GPU
                                         pool's block AND the tier's CPU-pool slot of
                                         that id, each bounded by its own pool
    GET|POST /plugins/kvtier/request     one request by id; no id = the id list
    GET  /plugins/kvtier/trace           the trace ring's newest events
    POST /plugins/kvtier/trace           arm / disarm / clear / read   [TOKEN]
    GET  /plugins/kvtier/guard           guard-mode state                   [read]
    POST /plugins/kvtier/guard           arm / disarm / fetch captures   [TOKEN]
    GET  /plugins/kvtier/lookups         the lookup/adopt ledger's newest records
    POST /plugins/kvtier/lookups         arm / disarm / clear / verdict    [TOKEN]
    GET  /plugins/kvtier/blockdump       the raw-byte dump's capability state
    POST /plugins/kvtier/blockdump       arm / disarm / dump bounded ranges [TOKEN]
    GET  /plugins/kvtier/disk            the SSD spill tier's state
    POST /plugins/kvtier/disk            arm / disarm / spill-one / ls       [TOKEN]
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
state half-advanced. THE BOOT ARMS IT (before it arms the tier, so nothing runs
unguarded); there is no env var that turns it ON, only
`VLLM_E2_TIER_DEV_BOOT_INERT`, which turns the boot's own arming off. Its state is
echoed as `guard_mode` by `status` and by every verb, the fault ring is the
evidence a boot was clean, and it must be DISARMED before any production boot.

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

from . import config, loaded_version

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
    config.ROUTE_LOOKUPS: {"op": {"kind": "enum", "values": ("read", "arm",
                                                             "disarm", "clear",
                                                             "status", "verdict",
                                                             "requests", "depth",
                                                             "order", "rates")},
                           "cap": {"kind": "int", "min": 16, "max": 4096},
                           "limit": {"kind": "int", "min": 1, "max": 256},
                           # capability 9: filter `requests` by one verdict string.
                           "verdict": {"kind": "str",
                                       "max_chars": MAX_ARG_STRING_CHARS},
                           # opt-in per-group deep probes (one extra read-only
                           # lookup per `sample_every` lookups, hard-capped).
                           "deep": {"kind": "bool"},
                           "sample_every": {"kind": "int", "min": 1,
                                            "max": 100000}},
    # THE RAW-BYTE DUMP. `length` is deliberately bounded by the SAME cap the
    # engine enforces (`blockdump.MAX_BYTES_PER_REQUEST`), pinned equal by
    # test_blockdump_caps.py: a wire bound looser than the engine's would let a
    # caller's over-long request through to be refused there instead of here, and
    # a tighter one would be a lie about what the verb can do. There is no
    # default for `length`: an over-long range must be asked for explicitly, and
    # nothing here can ask for "the rest of the span".
    config.ROUTE_BLOCKDUMP: {"op": {"kind": "enum", "values": ("status", "arm",
                                                               "disarm", "dump")},
                             "group": {"kind": "str",
                                       "max_chars": MAX_ARG_STRING_CHARS},
                             "block_id": {"kind": "int", "min": 0,
                                          "max": 2 ** 31 - 1},
                             "offset": {"kind": "int", "min": 0,
                                        "max": 2 ** 31 - 1},
                             "length": {"kind": "int", "min": 1, "max": 1048576},
                             "seconds": {"kind": "int", "min": 1, "max": 7200}},
    # THE SSD SPILL TIER. Every bound here is pinned equal to the one the engine
    # module enforces (`disk_store.MAX_LS_FILES`, `disk_store.DISK_OPS`), the
    # `ROUTE_BLOCKDUMP` rule: a wire bound looser than the engine's would only move
    # the refusal, and a tighter one would lie about what the verb can do.
    #
    # `seconds` IS THE ONE BOUND NO LONGER PINNED TO AN ENGINE CONSTANT (0.2.35): the
    # spill tier's arm became a LATCH and `disk_spill` deleted `ARM_DEFAULT_SECONDS` /
    # `ARM_MIN_SECONDS` / `ARM_MAX_SECONDS` along with the expiry. The bound stays, for
    # wire compatibility: a caller that already sends `{"op":"arm","seconds":7200}` must
    # not start getting a 400 the day this ships, and a value outside it was refused
    # before and still is. The number simply does nothing - `arm()` reports
    # `seconds_ignored` - and this is the comment that says so rather than a knob.
    # `key` is 64 chars so a caller can send the hash text
    # they have; `disk_layout.validate_key16` still demands exactly 16 hex chars,
    # and `fingerprint` 32 so the 12-hex digest check stays the layout's too.
    config.ROUTE_DISK: {"op": {"kind": "enum",
                               "values": ("status", "arm", "disarm", "spill-one",
                                          "ls", "contains", "identities", "offers",
                                          "authority", "failures", "shadow",
                                          "gpu-shadow", "restore-verify",
                                          "gpu-write", "restore-source",
                                          "adoptions",
                                          # LAST, because `DISK_OPS` order is pinned
                                          # against the dispatch table IN ORDER by a
                                          # test and this enum is derived from the same
                                          # tuple: `flush` waits BOUNDED for this
                                          # process's spill queue (0.2.66).
                                          "flush",
                                          # LAST, for the same reason: the runtime
                                          # reader-thread count (0.2.67), whose own
                                          # argument is the `threads` rule below.
                                          "read-threads")},
                        # `op=flush` (0.2.66): how long the caller is willing to wait
                        # for the spill queue. Bounded at the ceiling the ENGINE
                        # enforces (`disk_spill.FLUSH_BUDGET_SECONDS_MAX`) for the
                        # standing reason - a wire bound looser than the engine's only
                        # moves the refusal, and a tighter one lies about what the verb
                        # can do. A ceiling on a WAIT matters more than one on a
                        # listing: this call runs on a thread that can be holding a
                        # suspend, so a caller that asks for an hour must get 30 seconds
                        # and see that it did.
                        "timeout_seconds": {"kind": "int", "min": 0, "max": 30},
                        # `op=restore-source` (0.2.59): where a restore's destination
                        # rows come from, and how many of them may be in flight. A
                        # three-word enum: two postures (`arena`, `bounded`) and one
                        # DEPRECATED ALIAS (`bounce`, which resolves to `bounded` and
                        # says so in the posture payload's `alias` field). A typo is
                        # still a 400 that never reaches the engine and cannot mutate
                        # the posture, while a caller still using the superseded word
                        # gets a truthful answer rather than a rejection that reads as a
                        # bug. The setter refuses it a second time for a caller that
                        # arrives by some other door.
                        "source": {"kind": "enum",
                                   "values": ("arena", "bounded", "bounce")},
                        "slot": {"kind": "int", "min": 0, "max": 2 ** 31 - 1},
                        "key": {"kind": "str", "max_chars": 64},
                        # `op=identities` (0.2.46): the paged, digest-only enumeration
                        # of what the index believes PER RANK, so it can be diffed
                        # against `op=ls`. `after` is the cursor the previous page
                        # returned, bounded at the width of an identity
                        # (`key16:g<n>`) rather than the generic string width because a
                        # longer cursor is a caller that has not read the verb. `rank`
                        # filters to one rank's claims and `which` narrows the page to
                        # `offerable` or `partial` identities; both are bounded at the
                        # width of a rank label, and the SHAPES stay the engine's rules
                        # (`_op_identities` refuses a cursor with characters an identity
                        # cannot contain, and a `which` outside its three values).
                        "after": {"kind": "str", "max_chars": 64},
                        "rank": {"kind": "str", "max_chars": 32},
                        "which": {"kind": "str", "max_chars": 16},
                        "fingerprint": {"kind": "str", "max_chars": 32},
                        "seconds": {"kind": "int", "min": 1, "max": 7200},
                        # `op=read-threads` (0.2.67): how many spill-reader threads THIS
                        # process should run. The bound is pinned equal to the ENGINE's
                        # (`disk_read.MAX_READ_THREADS`, default 1) for the standing
                        # reason on this route: a wire bound looser than the engine's
                        # only moves the refusal to a place where it reads as a verb
                        # bug, and a tighter one lies about what the verb can do. The
                        # floor is 1 and is not negotiable at either layer: a pool of 0
                        # readers is the `thread_dead` refusal path, so a typo must not
                        # be able to stop every later restore in this process.
                        "threads": {"kind": "int", "min": 1, "max": 16},
                        # `op=read-threads` (0.2.67), the RELAY half. `ranks: false`
                        # asks the local process only. `wait_s` is how long the handler
                        # will sit before answering, and for read-threads THE DEFAULT IS
                        # ZERO (post the ask, return, read the receipts later) because
                        # this handler runs ON THE ENGINECORE LOOP THREAD while the
                        # answer can only arrive on a step that same loop dispatches - a
                        # wait waits on what it is stopping, and every second of it is
                        # serving not happening. The wire bound stays the [0, 30] window
                        # `spill-one` has always offered, because this key is shared by
                        # both ops and narrowing it would remove spill-one's reachable
                        # range; read-threads clamps the number IT honours to the door's
                        # own budget (`_disk_relay._resize_wait_ceiling()`, derived from
                        # `config.timeout_seconds()` minus a margin) and says so in
                        # `relay.wait_cap`, since a wait that outlives
                        # `asyncio.wait_for` at `endpoint.py:315` answers 504 over a
                        # resize every rank has already applied. Reported as
                        # PRE-EXISTING and NOT changed here: `spill-one` still offers
                        # its full 30 s against that same door, on that same thread,
                        # with that same self-deadlock, and its default is 3.0 s of
                        # sleep. `wait_s` also makes `spill-one`'s own engine-side
                        # window reachable from the wire, which it never was: absent
                        # from this spec, and an unknown key is a 400.
                        "ranks": {"kind": "bool"},
                        "wait_s": {"kind": "int", "min": 0, "max": 30},
                        "limit": {"kind": "int", "min": 1, "max": 200},
                        # `op=offers` and `op=authority`: this boot's two disk
                        # switches. A real boolean, validated here so a typo is a 400
                        # rather than a silently ignored field. There is NO `scope`
                        # key any more (0.2.31): the per-request opt-in marker it
                        # governed is gone, so a caller still sending it gets
                        # "unsupported request keys" instead of a posture it did not
                        # ask for. `op=shadow` (0.2.36) uses the same key for the
                        # read-only shadow verifier's in-process latch,
                        # `op=gpu-shadow` (0.2.37) the GPU-direct shadow writer's,
                        # `op=restore-verify` (0.2.39) the RESTORE-side verifier's and
                        # `op=gpu-write` (0.2.43) the GPU-DIRECT WRITE's - the last two
                        # being the pair whose scope note matters most: the apply point
                        # they arm is code only a worker runs.
                        "enabled": {"kind": "bool"},
                        # `ls` has always accepted a shard filter engine-side
                        # (`disk_store._op_ls`); it was unreachable because it was not
                        # in THIS table, so every attempt answered `unsupported request
                        # keys: ['shard']`. A listing of one shard is how an operator
                        # names the four rank files of one parked span without
                        # listing the whole store, which is exactly what `contains`
                        # sends them to do next - so the key is whitelisted at the
                        # width of a wire string and its SHAPE stays the engine's
                        # rule (`_validate_shard` demands exactly SHARD_CHARS hex and
                        # says so; a bound here tighter than that would lie about what
                        # the verb can do).
                        "shard": {"kind": "str",
                                  "max_chars": MAX_ARG_STRING_CHARS}},
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


def _conflict(key, query_value, body_value) -> HTTPException:
    """The 400 for one key that disagrees between the query string and the body."""
    return HTTPException(
        status_code=400,
        detail=f"{key} was given twice and disagrees: the query string says "
               f"{str(query_value)[:60]!r} and the body says "
               f"{str(body_value)[:60]!r}; send it once, or send the same value "
               "in both places")


async def post_arguments(request: Request, route: str) -> dict:
    """A POST's arguments: the JSON body AND the query string, both honoured.

    WHY THIS EXISTS (0.2.21). Every POST handler used to read the BODY ALONE, while
    every GET read the query string - and the routes are documented for both forms, so
    operators wrote things like `POST /plugins/kvtier/disk?op=arm&seconds=3600`. The
    query string was never read, so the request ran `op=status`, and the answer said
    `ok: true` with `"op": "status"` while the arm file kept its old mtime. An
    UNEXECUTED OPERATION REPORTING SUCCESS is the worst shape this plugin can have: a
    `true` that says a live-code path is open when it is closed (and the inverse for
    `disarm`) is worse than an error, because the operator stops looking.

    THE RULE, spelled out because "honour the query string" is not yet a rule:
      * both halves are validated against the SAME spec as GET, so a query key outside
        the table is a 400 here too instead of being ignored; a POST is no longer a
        place where a typo costs nothing;
      * a key present in BOTH must carry the SAME value once validated (`?seconds=3600`
        and `{"seconds": 3600}` agree, because the int rule coerces the query string);
        a disagreement is a 400 naming both values. There is deliberately NO precedence
        rule that silently picks a winner: silent precedence is a smaller version of
        the bug being fixed;
      * the SAFE-method route is untouched. A mutating `op` still cannot run from a
        method a browser may issue without a preflight (`route_disk`'s 405 and the
        engine-side `safe_method` refusal are the two guards), and the token gate is
        method-independent, so a query-string `op` is exactly as authenticated as a
        body `op`.
    """
    body = await body_arguments(request, route)
    query = query_arguments(request, route)
    for key, value in query.items():
        if key not in body:
            body[key] = value
            continue
        if body[key] != value:
            raise _conflict(key, value, body[key])
    return body


# --- the dev-state mirror ---------------------------------------------------

DEV_STATE_KEYS = ("guard_mode", "trace_armed", "lookup_ledger_armed",
                  "blockdump_armed")
DEV_STATE_UNKNOWN_NOTE = (
    "no engine answer has been seen by this API process yet, so guard mode, the "
    "trace ring, the lookup ledger and the raw-byte dump are UNKNOWN here, not "
    "false. The engine's own answer in any verb payload "
    "(snapshot/kvconfig/block/request/trace/guard/lookups/blockdump) is "
    "authoritative; the default at every engine boot is guard_mode=false, "
    "trace_armed=false, lookup_ledger_armed=false, blockdump_armed=false.")


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
        return {"guard_mode": None, "trace_armed": None,
                "lookup_ledger_armed": None, "blockdump_armed": None,
                "source": None,
                "epoch": None, "observed_at": None, "age_seconds": None,
                "note": DEV_STATE_UNKNOWN_NOTE}
    observed = entry.get("observed_at")
    age = None
    if isinstance(observed, (int, float)):
        age = round(max(0.0, time.time() - observed), 3)
    return {"guard_mode": entry.get("guard_mode"),
            "trace_armed": entry.get("trace_armed"),
            "lookup_ledger_armed": entry.get("lookup_ledger_armed"),
            "blockdump_armed": entry.get("blockdump_armed"),
            "source": entry.get("source"), "epoch": entry.get("epoch"),
            "observed_at": observed, "age_seconds": age,
            "note": ("mirrored from the engine's own answers in this API "
                     "process; null means never observed. A plugin restart "
                     "clears this, not the engine's state.")}


# --- payload helpers --------------------------------------------------------

def blockdump_capability(deployed: bool, armed) -> str:
    """The three-state marker, from what THIS process can actually know.

    The same discipline the KV-tier reader on the proxy host uses for the probe
    channel: "not deployed" and "deployed but not armed" and "armed" are three
    different operational statements and must never collapse into one boolean.
    The fourth value is the one this API process can produce and the ENGINE never
    needs: `status` makes no engine call, so an arm it has never observed is
    `..._unobserved`, never `disarmed` (the guard-mode rule).
    """
    if not deployed:
        return "not_deployed"
    if armed is True:
        return "armed_and_ready"
    if armed is False:
        return "deployed_disarmed"
    return "deployed_armed_state_unobserved"


# WHICH PROCESS'S VERSION `/status` CARRIES. The route is answered here, in the
# API-server process, and deliberately makes no engine call - so its
# `plugin_version` is what THIS process imported. `/snapshot`'s is the engine's.
# Comparing the two is how a reader catches "the package was upgraded, the API
# server restarted, the engine did not".
STATUS_VERSION_SCOPE = ("the API-server process that answered this route; it "
                        "makes no engine call, so this is NOT the engine's "
                        "loaded version - read " + config.ROUTE_PREFIX +
                        "/snapshot for that")


def status_payload(dev_state: dict | None = None) -> dict:
    # HOW FULL THE SSD TIER IS, built BEFORE the payload so a failure cannot take the
    # route down with it. THE SAME BLOCK as `snapshot.disk_fill`, deliberately: the
    # operator asked for one answer reachable from either read route, not two dialects
    # of it. `/status` makes NO engine call and never will - this is a rank-file read
    # behind the rollup's one-second cache, which is the same cost `worker_proof`
    # already accepts on a polled path, and it is the only fleet-wide fill number
    # available to a process whose own spill counters are structurally zero.
    try:
        from . import disk_store  # noqa: PLC0415 - the fill block lives with the tier
        disk_fill_block = disk_store.disk_fill()
    except Exception as exc:  # noqa: BLE001 - /status must always answer
        logger.warning("kvtier-dev-plugin: /status disk_fill failed: %s: %s",
                       type(exc).__name__, exc)
        try:  # the block's OWN key set, all null, if the module came in at all
            disk_fill_block = disk_store.disk_fill_unavailable(type(exc).__name__)
        except Exception:  # noqa: BLE001 - the import itself is what failed
            # The only shape that is honest here: there is no skeleton to reuse, so
            # say `not available` and name the reason, and let `available` stay the
            # flag a caller must check before touching any other key.
            disk_fill_block = {"available": False,
                               "reason": f"did not build: {type(exc).__name__}"[:160]}
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
        "guard_mode": None, "trace_armed": None, "lookup_ledger_armed": None,
        "blockdump_armed": None, "note": DEV_STATE_UNKNOWN_NOTE}
    return {
        "plugin": config.PLUGIN_NAME,
        # WHICH CODE IS RUNNING HERE. `plugin` above is a NAME, and no field in
        # this payload carried a version before 0.2.50: an operator could not tell
        # a loaded 0.2.47 from a loaded 0.2.49 from either route. See
        # `loaded_version.py` for why this is `__version__` and never the installed
        # distribution metadata.
        **loaded_version.fields(STATUS_VERSION_SCOPE),
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
        "lookup_ledger_armed": mirrored.get("lookup_ledger_armed"),
        "blockdump_armed": mirrored.get("blockdump_armed"),
        "dev_state": mirrored,
        # THE SSD TIER'S FILL, from the rank files the workers write. See
        # `disk_store.disk_fill` for why the flat per-process counters beside it (the
        # `disk` block below) cannot answer this question at all.
        "disk_fill": disk_fill_block,
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
        # THE RAW-BYTE DUMP's capability, three states, never a bare bool. The
        # engine's own answer (`GET/POST /blockdump`, or `snapshot.blockdump`) is
        # authoritative for `armed`; what is here is the local deployment fact
        # plus the last arm this API process OBSERVED, and `..._unobserved` when
        # it has observed none - the guard-mode rule again.
        "blockdump": {
            "route": config.ROUTE_PREFIX + config.ROUTE_BLOCKDUMP,
            "deployed": config.diagnostics_dir() is not None,
            "armed": mirrored.get("blockdump_armed"),
            "capability": blockdump_capability(
                config.diagnostics_dir() is not None,
                mirrored.get("blockdump_armed")),
            "ops": ["status", "arm", "disarm", "dump"],
            "note": ("a bounded, read-only dump of parked KV bytes: one region, "
                     "one slot, an explicit byte range, base64 in a file a WORKER "
                     "writes (the utility channel never carries bytes). DISARMED "
                     "at every boot and there is no env var that arms it; only "
                     "`POST {\"op\":\"arm\"}` does, and an arm expires on its own. "
                     "Disarmed, a worker's cost is one stat per poll interval. "
                     "`not_deployed` means no VLLM_E2_TIER_DIAG, so no channel."),
        },
        # THE SSD SPILL TIER's route and ops. What is knowable HERE is the surface
        # and this process's own env; the arm, the counters, the store's size and
        # the `engine_fields` verdict all belong to the engine, and come from
        # `GET|POST /disk` or `snapshot.disk_store`. `not_configured` there means
        # "this boot has no disk tier", not "this verb is broken".
        "disk": {
            "route": config.ROUTE_PREFIX + config.ROUTE_DISK,
            # THE OPS TABLE, READ FROM ITS OWNER. This used to be a hand-copied list,
            # and a new op added to `DISK_OPS` silently stayed unadvertised here - the
            # exact drift the status route exists to prevent. Importing the table makes
            # the two views un-mergeable by construction.
            "ops": list(disk_store.DISK_OPS),
            "root_env": config.ENV_DISK_ROOT,
            "ceiling_env": config.ENV_DISK_BYTES,
            # The journal gets its own path because it is the one piece of state
            # that cannot be rebuilt: unset it falls back to VLLM_E2_TIER_DIAG, and
            # an operator who wants the index to survive a REBOOT points it at
            # persistent storage. Only the journal follows it - the arm file stays
            # in the diagnostics directory and dies at boot, on purpose.
            "journal_dir_env": config.ENV_DISK_JOURNAL_DIR,
            "note": ("the SSD spill tier, both halves. WRITE: KV slots evicted "
                     "from the host-RAM tier are spilled to one file each on the "
                     "engine's own disk, named "
                     "kvslot-<key16>-g<group>-<geodigest12>.bin. The group id is "
                     "PART OF THE NAME, because two KV-cache groups can hash the "
                     "same token chain: the older 2-field shape "
                     "kvslot-<key16>-<geodigest12>.bin gave them ONE file for two "
                     "different payloads and is now refused as malformed. The "
                     "digest half is the RANK'S OWN: every worker rebuilds the name "
                     "with its own measured slot width and rank id (the scheduler "
                     "sends a 12-zero placeholder for that field and the worker "
                     "replaces it), because one scheduler cannot broadcast one "
                     "digest that is four ranks' digests - the per-rank "
                     "re-derivation is the safety keystone, not an oversight. "
                     "OFF twice over: "
                     f"{config.ENV_DISK_ROOT} must name a real filesystem "
                     "(tmpfs roots are refused) AND the tier must be armed; no env "
                     "var arms it and the arm is a LATCH: it stays armed until "
                     "disarmed, and disarming stops serving from disk too. "
                     "`spill-one` is relayed to "
                     f"the workers through {config.ENV_DIAG_DIR} because a worker "
                     "has no utility channel, and the write runs on the spill "
                     "thread with NO fsync (buffered writes: the page cache owns "
                     "device ordering, and a journaled file that never reached the "
                     "device is reported as `file_missing` and recomputed). The "
                     f"spill journal lives in {config.ENV_DISK_JOURNAL_DIR} when "
                     f"set, else in {config.ENV_DIAG_DIR}; the journal directory's "
                     "filesystem is reported under `journal.fs`, because a journal "
                     "on tmpfs is a map that dies at reboot while the KV files it "
                     "describes survive. "
                     "THE DIRECTORY IS MADE FOR YOU on the first journal line "
                     "(mode 0750, the mode the store's own shard directories get) "
                     "and if it cannot be made or written the answer says so - "
                     "`journal.writable: false`, `journal.directory_error`, "
                     "`journal.write_errors`, and `journal_dir=unwritable` on the "
                     "boot line: the spill still lands (a logging problem does not "
                     "get to cost the tier) and the index never learns of it, which "
                     "is a reported state rather than a silent one."
                     "READ BACK: a restore path HAS existed since 0.2.14, and every "
                     "answer from this verb carries it under `restore` - the "
                     "SCHEDULER's own verdict (layer fitted here, gate, index "
                     "entries, worker proof, offers per refusal reason) with "
                     "`restore.why_no_restore` naming the first missing link. It "
                     "EXTENDS the fork's host-RAM load with spans read off this "
                     "store, needs the spill arm AND the tier arm, and refuses "
                     "without a rank file proving the worker half. Read "
                     "`engine_fields` from the engine to know whether this boot's "
                     "metadata carries store_keys: without it the worker refuses "
                     "and counts `no_key_for_slot` and writes zero bytes. "
                     "`failures` answers WHY A READ DID NOT HAPPEN: the bounded "
                     "ring of refused/failed loads this process wrote, merged with "
                     "what the live rank files carried (a worker's reasons cannot "
                     "reach EngineCore any other way), each record naming its "
                     "request, span, group, block key and GPU block, the failure "
                     "kind from the worker vocabulary, the block's provenance, and "
                     "for `buffer_cap` the pool budget, the bytes already committed, "
                     "the bytes the read needed and the jobs the offer needed versus "
                     "the jobs that fit. a GET "
                     "will not run a mutating op."),
        },
        "tier": {
            "route": config.ROUTE_PREFIX + config.ROUTE_TIER,
            "note": "the KV tier's runtime kill switch and the status of the "
                    "upstream PR #56810 backport (excluding non-prefix-cacheable "
                    "KV groups from store/load, which a hybrid geometry needs or "
                    "the store path asserts). THE BOOT BRINGS THE TIER UP WORKING: "
                    "when the tier connector is built, the connector-creation hook "
                    "installs the backport, installs the guard, and ARMS the tier - "
                    "but only if all three verify, so a boot that cannot prove the "
                    "fix is in place leaves the tier DISARMED and says which check "
                    "failed. `pr56810.boot` is that verdict (mode, backport, guard, "
                    "tier, refusal reasons, fault count at boot) and its `line` is "
                    "the one line written to the engine log. "
                    f"{config.ENV_BOOT_INERT}=1 is the explicit opt-out back to "
                    "the old inert boot. `armed: false` on a boot that did not opt "
                    "out is a FAULT TO READ, not the expected state - read "
                    "pr56810.operator_note and boot.refusals. Disarmed, the "
                    "connector reports no prefix hit, hands the workers empty "
                    "store/load lists and claims no pending work, so the engine "
                    "serves from GPU KV alone - no engine restart. "
                    "`pr56810.installed: false` means the divisibility fix is NOT "
                    "fully live (see `pr56810.refused_methods`) and a hybrid "
                    "geometry will still crash the store path. The tier can NEVER "
                    "be armed - by the boot or by a verb - unless the fix is fully "
                    "installed and the geometry guard passes: an `arm` is refused, "
                    "not applied, otherwise, and `pr56810.arm_block`/"
                    "`disarm_reason` say why (distinguishing `geometry`, "
                    "`no_manager` - nothing to arm in this process - and "
                    "`incomplete_install`). After a boot, the thing to READ is "
                    "pr56810.boot.line plus guard.captures (a non-zero fault count "
                    "means the tier is declining work, silently). `op: status` "
                    "re-runs the install as a backstop; `arm`/`disarm` remain the "
                    "operator's controls afterwards. Read the state from the "
                    "engine in `snapshot` or the engine's "
                    "`" + config.ROUTE_PREFIX + config.ROUTE_TIER + "` route.",
            "ops": ["status", "arm", "disarm"],
        },
        "development_facilities": {
            "guard_mode": "DEVELOPMENT ONLY. Armed, it turns a KV-tier exception "
                          "into a captured record plus a safe decline. THE BOOT "
                          "ARMS IT, before it arms the tier, so no store or lookup "
                          "ever runs unguarded; no env var turns it ON "
                          f"({config.ENV_BOOT_INERT} only turns the boot's arming "
                          "off) and POST /plugins/kvtier/guard is the operator's "
                          "arm/disarm. BECAUSE IT MASKS REAL BUGS, the fault ring "
                          "is the evidence a boot was clean: read "
                          "POST /plugins/kvtier/guard {\"op\":\"captures\"} after "
                          "every boot and DISARM before any production boot. It "
                          "never swallows a BaseException.",
            "trace": "bounded ring, opt-in, nothing wrapped while disarmed",
            "lookup_ledger": "bounded ring of get_num_new_matched_tokens "
                             "offers joined to their update_state_after_alloc "
                             "adopts, per request id. Opt-in: it records nothing "
                             "until armed (POST /plugins/kvtier/lookups "
                             "{\"op\":\"arm\"}), it changes no scheduling "
                             "decision, and `op: verdict` names the records where "
                             "the scheduler adopted 0 of a positive offer. Its "
                             "preview is in `snapshot.lookup_ledger`.",
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
    args = await post_arguments(request, config.ROUTE_SNAPSHOT)
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
    args = await post_arguments(request, config.ROUTE_KVCONFIG)
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
    args = await post_arguments(request, config.ROUTE_BLOCK)
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
    args = await post_arguments(request, config.ROUTE_REQUEST)
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
    args = await post_arguments(request, config.ROUTE_TRACE)
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
    args = await post_arguments(request, config.ROUTE_GUARD)
    return as_json(await engine_call(request, config.METHOD_GUARD, args,
                                     source="guard verb"))


@router.get(config.ROUTE_TIER)
async def route_tier(request: Request):
    """KV-tier state: is the PR #56810 fix live, and is the tier armed? Token
    required, because the same door arms AND disarms the connector. With no `op`
    this is `op=status`: it installs the fix against the live manager and reports
    `armed: false` - arming is a separate, deliberate act (see the runbook)."""
    await _guard_dev_state_route(request)
    args = dict(query_arguments(request, config.ROUTE_TIER))
    args.setdefault("op", "status")
    return as_json(await engine_call(request, config.METHOD_TIER, args,
                                     source="tier status"))


@router.post(config.ROUTE_TIER)
async def route_tier_post(request: Request):
    """`{op: status|arm|disarm}` - the tier's runtime kill switch.

    THE TIER IS DISARMED AT BOOT BY DESIGN, so a fresh engine parks nothing until
    someone arms it; `arm` is refused unless the PR #56810 fix is fully installed
    and the geometry guard passes (the refusal names its kind: `geometry`,
    `no_manager` or `incomplete_install`). Being disarmed is not the same as being
    unpatched: the connector-creation hook (`backport.install_connector_hook`,
    fitted by `register()`) runs the install when the tier connector is built -
    the first moment the manager class exists, inside `Scheduler.__init__` and
    long before the first request - so the boot sequence is `status` (CONFIRMS it:
    expect `installed: true`, `armed: false`, `patched_before_first_request: true`),
    then `arm` (expect `armed: true`), and only then traffic. The verbs are the
    backstop, not the mechanism.

    `disarm` makes the connector a STRICT no-op through the manager methods the
    plugin wraps: it reports no prefix hit, hands the workers empty store/load
    lists, claims no pending work, and queues nothing on the finish path - which
    is why a disarm now gates BOTH callers of the selector that holds the upstream
    assert (`manager.py:745` and `manager.py:1177`). No restart, no reboot of a
    4-minute boot. `arm` puts the scheduling back.

    What it cannot undo is reported honestly in the payload under
    `tier.disarm_limits`: an in-flight copy already handed to the workers
    finishes, blocks already parked in host RAM stay parked (re-arm makes them
    reachable again), and the worker PROCESS is never contacted - its no-op is
    delivered as empty metadata. The two entries about the switch's own machinery
    now read in its favour: `install()` fits the wrappers FIRST (so a refused or
    half-done install still leaves "wrappers + disarmed"), and the connector's
    FINISH path is `GATED` like the per-step trio. `connector_hook` says whether
    this process ever saw a tier connector at all, which is the only way the
    wrappers can be missing. Nothing here touches the backport itself: if
    `tier.patched` is false the tier is harmless ONLY while it is disarmed, and
    arming such an engine is what `set_armed` refuses.
    """
    await _guard_dev_state_route(request)
    args = await post_arguments(request, config.ROUTE_TIER)
    return as_json(await engine_call(request, config.METHOD_TIER, args,
                                     source="tier verb"))


@router.get(config.ROUTE_LOOKUPS)
async def route_lookups(request: Request):
    """The lookup/adopt ledger's newest records. Token required even for this
    read, as for `/trace`: the same door arms the recorder."""
    await _guard_dev_state_route(request)
    args = dict(query_arguments(request, config.ROUTE_LOOKUPS))
    args.setdefault("op", "read")
    return as_json(await engine_call(request, config.METHOD_LOOKUPS, args,
                                     source="lookup ledger read"))


@router.post(config.ROUTE_LOOKUPS)
async def route_lookups_post(request: Request):
    """The LOOKUP LEDGER - the `get_num_new_matched_tokens` offer paired with the
    `update_state_after_alloc` adopt, per request id.

    `{op: read|arm|disarm|clear|status|verdict, cap?, limit?}`. It exists because
    the live engine logs "update_state_after_alloc called with no external tokens
    but get_num_new_matched_tokens() recorded a pending CPU hit": the connector
    offered a positive number of host-RAM tokens and the scheduler adopted zero,
    and nothing said which rule did it. Each record carries the local hit the
    scheduler passed IN, the hit returned, and the `num_external_tokens` the
    scheduler later passed; `op: verdict` classifies the pairs and reports the
    `dropped_to_zero` ones with `local % block_size`, so the partial-tail rule can
    be confirmed or refuted from data.

    DEVELOPMENT ONLY and it changes no decision: it records what the scheduler
    already did. `arm` installs the tier wrappers it reads through (composed over
    the PR #56810 backport's own layers, never replacing them) and `disarm`
    removes them only when nothing else still needs them. The ring has a hard
    capacity, a monotonic `dropped` count, and records nothing while disarmed.
    """
    await _guard_dev_state_route(request)
    args = await post_arguments(request, config.ROUTE_LOOKUPS)
    return as_json(await engine_call(request, config.METHOD_LOOKUPS, args,
                                     source="lookup ledger verb"))


@router.get(config.ROUTE_BLOCKDUMP)
async def route_blockdump(request: Request):
    """The raw-byte dump's capability state. Token required even for this read,
    as for `/trace` and `/guard`: the same door ARMS the facility."""
    await _guard_dev_state_route(request)
    args = dict(query_arguments(request, config.ROUTE_BLOCKDUMP))
    args.setdefault("op", "status")
    return as_json(await engine_call(request, config.METHOD_BLOCKDUMP, args,
                                     source="blockdump status"))


@router.post(config.ROUTE_BLOCKDUMP)
async def route_blockdump_post(request: Request):
    """THE BOUNDED RAW-KV-BYTE DUMP - read-only, arm-gated, and file-delivered.

    `{op: status|arm|disarm|dump, group?, block_id?, offset?, length?, seconds?}`.
    It exists to answer a byte-level question no counter can (whether a store on
    behalf of one group also writes other regions' bytes at the same GPU index),
    so the payload is bytes - which is exactly why this route is shaped the way it
    is:

      * TOKEN-GATED like `/guard`: the door that can read KV bytes is not a read.
      * DISARMED at every boot, no env var that arms it, and an arm EXPIRES
        (`seconds`, default 1800, max 7200). `op: dump` on a disarmed engine is a
        refusal that names the arm verb, never a silent no-op.
      * `length` is REQUIRED and capped (this module's spec, mirrored by the
        engine's own `MAX_BYTES_PER_REQUEST`): a caller asks for N bytes of one
        region at one slot, and nothing here can ask for a span.
      * a rolling rate window (requests per window), one request in flight per
        rank, a response-file size cap, and a `refused` reason for every one of
        them. Nothing is ever truncated to fit.
      * the BYTES NEVER CROSS THE UTILITY CHANNEL. This half writes one small
        control file; the worker writes the base64 + per-block sha256 to
        `blockdump-<rank>.json` in the diagnostics directory and the payload says
        where. Reading a slot is host-memory arithmetic only: no CUDA, and no
        refcount, free-queue or hash-state change (see `blockdump.py`).
    """
    await _guard_dev_state_route(request)
    args = await post_arguments(request, config.ROUTE_BLOCKDUMP)
    return as_json(await engine_call(request, config.METHOD_BLOCKDUMP, args,
                                     source="blockdump verb"))


@router.get(config.ROUTE_DISK)
async def route_disk(request: Request):
    """The SSD spill tier's state. Token required even for this read, as for
    `/blockdump`: the same door ARMS a facility that writes real KV bytes.

    READ-ONLY BY CONSTRUCTION. The `op` is a QUERY PARAMETER here, so the URL that
    arms the tier is as short as the URL that reads its status - and a browser
    issues a GET with no CORS preflight and no custom header, which means a page
    that can make the client load an image can issue it. A mutating `op` on this
    method is therefore refused with 405 before the engine is touched, and
    `safe_method` is passed along so the refusal holds even if someone calls the
    engine-side verb directly (the `GET /api/config/auto-suspend` lesson: one verb,
    two methods, and the mutating half must not answer on the safe one).
    """
    await _guard_dev_state_route(request)
    args = dict(query_arguments(request, config.ROUTE_DISK))
    from . import disk_store  # noqa: PLC0415 - the op lists live with the verb, and
    # this module otherwise knows only `config` (no engine-side module at import).
    op = str(args.get("op") or "status").strip().lower()
    if op in disk_store.MUTATING_DISK_OPS:
        raise HTTPException(
            status_code=405,
            detail=f"op '{op}' changes state and will not run from GET; send it as "
                   f"POST {config.ROUTE_PREFIX + config.ROUTE_DISK} "
                   f"(this route's GET answers op=status only)")
    args.setdefault("op", "status")
    if op == "contains":
        # The same rule as the POST: "you asked badly" is a 400 here too, never a
        # `present: false` - a GET is the form an operator reaches for when they are
        # already unsure, so it must not answer their malformed key with a "not there".
        problem = disk_store.contains_key_problem(args.get("key"))
        if problem is not None:
            raise HTTPException(status_code=400, detail=f"op=contains: {problem}")
    args["safe_method"] = True
    return as_json(await engine_call(request, config.METHOD_DISK, args,
                                     source="disk spill status"))


@router.post(config.ROUTE_DISK)
async def route_disk_post(request: Request):
    """THE SSD SPILL TIER - write path and restore path, arm-gated, off by default.

    `{op: status|arm|disarm|spill-one|ls|contains|offers|authority|failures,
    identities, read-threads, slot?, key?, fingerprint?, seconds?, limit?, shard?,
    after?, rank?, which?, enabled?, threads?}` - `op=identities` (0.2.46) is the paged,
    digest-only enumeration
    of what the index believes PER RANK (`after` is its cursor, `rank` and `which` the
    filters), the other half of the diff against `op=ls`. KV slots evicted from
    the RAM tier are spilled to
    `<root>/<shard>/kvslot-<key16>-g<group>-<geodigest12>.bin` on the engine's own
    disk, so the RAM tier can shrink - the group id is part of the identity, since
    two groups can hash the same token chain, and the digest half is rebuilt by
    each WORKER from its own slot width and rank id.

    THERE IS A RESTORE PATH (`disk_restore`, since 0.2.14): on a host-RAM shortfall
    the SCHEDULER offers the SSD's continuation as external tokens, extends the
    fork's own load tuple with the CPU slots it allocated, and the worker reads the
    file back into them.

    IT DOES NOT SERVE UNTIL YOU TELL IT TO, AND WHICH SWITCH YOU RAISE IS THE
    POSTURE. Two per-boot switches, both FALSE at every fresh boot, both POST-only:

      * `{"op":"authority","enabled":true}` - THE DISK IS THE AUTHORITY FOR EVERY
        REQUEST, every request of this boot. Each request's host-RAM hit is treated
        as zero tokens for the offer
        arithmetic and released, so the disk index decides the span from the local
        GPU-computed boundary. It opens the offer gate BY ITSELF, so this one POST is
        the whole change: one POST serves every request from SSD, a restart turns it
        off, and no env var, config file, spill arm or recovery path can raise it.
        `{"op":"authority"}` alone reads the posture back.
      * `{"op":"offers","enabled":true}` - the cheaper posture: RAM first, the disk
        EXTENDING a prefix the host-RAM tier left short. `{"op":"offers"}` alone reads
        the posture.

      `restore.serving_as` collapses the two into one verdict (`off` /
      `disk_authority` / `ram_first_gap_fill`), and a fresh boot is
      `serving_as:"off"`. The reason both switches exist and neither is the spill arm:
      the incident of 2026-10-05 was a spill arm raised for a WRITE measurement being
      enough to put 123,474 un-requested 160,000-token offers in front of live traffic.

      NO REQUEST FIELD DECIDES ANY OF THIS (0.2.31). Until now the authority rule was
      carried by a client-supplied marker - the `kvtier_disk_restore` boolean inside the
      engine's own `kv_transfer_params` field - and a `scope` argument chose whether
      that
      marker was required (`optin`) or ignored (`all`). A request
      body must not decide whether its KV comes off NVMe, so the marker, the `scope`
      argument and the `flagged` / `skipped_no_marker` / `marker_key` / `requests_seen`
      fields it fed are RETIRED: sending `scope` is now refused as an unsupported
      request key, and no client can ask for disk authority or decline it.
      `disk_restore` no longer reads `request.kv_transfer_params` at all. Do NOT send
      the old marker - it is ignored, and a request that sends it is served exactly like
      one that does not. It is named here as retired vocabulary an operator may still
      see in an old client, not as an instruction.

      The restore never publishes into `cached_block_hash_to_block`, which is why a
      restored block helps only the request that paid for it. Every answer
      from this verb carries the scheduler's own verdict under `restore`, and
      `restore.why_no_restore` names the first link that is missing (layer not fitted
      here / gate closed / index empty / no worker proof / no offer made).

      WHAT THE SWITCH CANNOT DO. It does not zero `num_computed_tokens`: that number
      is the offset the fork's own DMA writes at, so zeroing it would put disk bytes
      into the request's already-cached, shared GPU prefix blocks. A request
      with a full GPU prefix hit therefore has nothing to restore and recomputes, like
      any other. And a request the disk cannot serve is a silent, cheap
      fallback: the fork's own answer, no allocation left behind, `refused.no_disk_run`
      counted. To tell a forced read from a real one, read
      `restore.last_offer` (`ram_tokens_reported` vs `ram_tokens_used`),
      `restore.local_hit_ignored` and `restore.local_hit_tokens_ignored`.

      **On a model where a disk claim cannot be complete, there is no offer
      to read.** Since 0.2.25 a prefix-cacheable group with no positionally
      stable name (an align-mode Mamba/GDN group: one claimed span costs it
      ONE boundary-state block, and that block has no per-block name to read
      back) makes the whole offer refused as
      `refused.incomplete_group_set`, so `last_offer` stays **None** and
      anything indexing `last_offer["groups"]` must guard for it. Read
      `restore.group_serviceability` instead - which groups are cacheable,
      stable, restorable, blocked, and why - and
      `restore.last_groups_required` (per group: `blocks_needed`,
      `null_blocks`, `restored`, `blocks_filled`). Both are published on the
      refusal door. The fallback stays silent and cheap: the fork's own
      answer.

      * TOKEN-GATED like `/guard` and `/blockdump`: this route can arm a facility
        that writes KV bytes to disk, so it is a dev-state route in every shape.
      * OFF TWICE OVER: `VLLM_E2_TIER_DISK_ROOT` must name a real filesystem (the
        tmpfs roots are refused by the layout layer), AND the tier must be ARMED.
        No env var arms it; `{"op":"arm"}` does, and since 0.2.35 an arm is a LATCH:
        it comes down on `{"op":"disarm"}` and not on a clock. It gates SERVING from
        disk as well as writing to it, so a disarm closes both directions. The
        `seconds` argument is still accepted and does nothing (it is ignored, and
        reported as ignored), because an existing caller must not start failing.
      * `spill-one` names ONE slot (`{slot, key, fingerprint}`) so the path can be
        proved to write real bytes without waiting for eviction pressure. It
        requires the arm, and it is RELAYED: this process has no KV memory, so the
        ask becomes one small file that every worker answers for its own rank
        (`VLLM_E2_TIER_DIAG` is that channel, the same one the rank files use).
        The write happens on the spill thread, never on an engine thread, and the
        receipt reports where the file will be.
      * `fingerprint` on `spill-one` is an OPTIONAL CROSS-CHECK, not the naming
        source, and `""` (what the engine sends today) means nothing. A file name
        comes from the fingerprint each WORKER builds for itself - its own measured
        slot width and its own rank id - because one scheduler broadcasting one
        digest to four ranks cannot produce four ranks' digests: `total_bytes_per_block`
        and `rank` are rank-local. A non-empty `fingerprint` that disagrees with a
        rank's own digest is a counted `fingerprint_mismatch` refusal for that rank,
        never a silent override.
      * `read-threads` (0.2.67) reads or resizes the SPILL-READER THREAD COUNT of
        the process that answers: `{"op":"read-threads"}` alone answers the posture,
        `{"op":"read-threads","threads":6}` moves it, in both directions, in one boot,
        instead of costing an engine restart. It is MUTATING and POST-only for the
        same reason as the four latching ops above - it starts and stops threads. A
        shrink is cooperative and therefore ASYNCHRONOUS: a surplus slot finishes the
        read it is holding and exits at its next poll, so `resize.surplus_draining`
        says what has not happened yet. `threads` outside `[1,
        disk_read.MAX_READ_THREADS]`, or not an integer, is a 400 at the wire and a
        refusal here, and it changes nothing. TWO BOUNDARIES, both in the answer:
        per BOOT (a restart reverts to `VLLM_E2_TIER_DISK_READ_THREADS`, which stays
        the boot-time source of truth) and per PROCESS - and since 0.2.67 the workers
        ARE reached, by a relay of their own (`disk-read-threads.json`, a second channel
        because a resize must never overwrite a pending `spill-one` ask), so
        `relay.ranks_applied` lists the ranks whose OWN counters prove they moved,
        `relay.ranks_pending` the ones that have not answered, and `scope
        .this_process_reads_spill_files` still says whether the process you resized
        directly is one that ever reads a spill file at all. Each rank publishes
        `readers.configured` / `readers.threads_source` / `readers.last_resize` in its
        own rank file.
      * `ls` is read-only and bounded (`limit` <= `disk_store.MAX_LS_FILES`, plus a
        response-byte ceiling that trims and SAYS so); `status` reports this
        process's own counters and the wiring's state, and `ls` is what walks the
        store to measure it.
      * A MUTATING `op` is refused with 405 on GET; the safe method answers
        `op=status` only.
      * ON THIS POST EVERY ARGUMENT MAY ARRIVE IN THE QUERY STRING OR THE BODY, and
        BOTH are read (`post_arguments`). `?op=arm&seconds=3600` with no body arms the
        tier, and so does `?op=arm` beside `{"seconds": 3600}`. A key given in both
        places must AGREE or the answer is a 400 naming the two values: there is no
        silent precedence, because until 0.2.21 this route read the body only and a
        query-string `op` was IGNORED WHILE ANSWERING `ok: true` - an unexecuted
        operation reporting success, on a verb whose whole job is to open and close a
        live path.
      * `contains` answers one identity: is it in the disk index, for WHICH RANKS, and
        whether the index is even able to answer (`present: null` with a `verdict` of
        `index_never_refreshed` / `index_has_no_journal_directory` when it is not).
        It reads the replayed journals in memory and NEVER walks the store - `ls` (with
        `shard`) is the verb that walks. A key that cannot name an identity is a `400`
        here, never a `present: false`.
      * `engine_fields` answers `present|absent|unknown` for this boot: whether
        `SimpleCPUOffloadMetadata` carries `store_keys` at all. On an unpatched
        engine the worker hook refuses and counts `no_key_for_slot` and writes
        nothing - which is the same build being correct on both engines.
    """
    await _guard_dev_state_route(request)
    args = await post_arguments(request, config.ROUTE_DISK)
    from . import disk_store  # noqa: PLC0415 - the key rule lives with the verb
    # A malformed key is a `400`, answered HERE so that "you asked badly" can never
    # reach the caller as a `present: false` from the engine. The ONE definition of a
    # usable key is `disk_store.contains_key_problem`, shared with the verb itself.
    if str(args.get("op") or "").strip().lower() == "contains":
        problem = disk_store.contains_key_problem(args.get("key"))
        if problem is not None:
            raise HTTPException(status_code=400, detail=f"op=contains: {problem}")
    return as_json(await engine_call(request, config.METHOD_DISK, args,
                                     source="disk spill verb"))


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
