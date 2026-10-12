"""The gate, in isolation: no FastAPI transport, no engine, no vllm.

SINCE 0.2.72 THERE IS ONE GATE ON THIS SURFACE AND IT IS THE OPT-IN SWITCH.
`endpoint.guard_route` asks `config.enabled()` and nothing else. This file exists to
pin what that means, in both directions:

  * with `VLLM_E2_TIER_DEV` unset, EVERY route answers 503 to EVERY peer - loopback,
    the production proxy, a machine on the far side of the world, a scope with no
    client address at all, a scope forged with `X-Forwarded-For: 127.0.0.1`. The
    switch, not the caller, decides;
  * with it set, NOTHING filters the peer. No address is admitted and no address is
    refused, and no header grants or withholds anything.

WHAT WAS DELETED, and why both deletions are asserted rather than trusted. Two
families of guards stood here and both are gone from the product, not gated off:
`require_bearer` and its three helpers (0.2.68, a token nobody is asked for any more)
and `loopback_denial`/`require_loopback`/`_peer_address_unwrapped`/`peer_denial`/
`require_peer` plus the `GuardDenied` they raised (0.2.72). A helper nobody calls
misdescribes the gate exactly as surely as a variable nobody reads, so
`test_the_*_guards_are_gone_from_the_product` fails the day either family comes back
as an unreachable function.

The peer check is the one whose REMOVAL is the security fix. It was never a control:
it accepted the loopback peer, and a local proxy IS the loopback peer - so a
fronting proxy (llama-swap's `/upstream/` path, nginx on the host) republished these
routes to its own clients while the check saw a permitted address and reported
success. A gate that cannot fail is worse than no gate, because it manufactures
assurance. What replaces it is honest rather than stronger: the explicit opt-in, the
safe-method rule (`tests/test_post_query_arguments.py`), the argument specs, each
facility's own arm - and network-level protection, which is the operator's job.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from kvtier_dev_plugin import config, endpoint

# Peers that used to be the interesting axis of this file. Kept as the PARAMETER so
# the property that replaced the axis is measured over the same range: the answer
# must not depend on any of them.
EVERY_PEER = [
    ("127.0.0.1", 50000),
    ("127.2.3.4", 1),              # the whole 127/8
    ("::1", 1),
    ("::ffff:127.0.0.1", 1),       # a dual-stack listener's rendering of IPv4
    ("192.168.2.64", 51000),       # the production proxy
    ("192.168.2.120", 51001),      # the development workstation
    ("10.0.0.7", 1),
    ("169.254.1.1", 1),
    ("0.0.0.0", 1),
    ("::", 1),
    ("fe80::1", 1),
    ("203.0.113.9", 51234),        # TEST-NET-3: some other network entirely
    ("testclient", 50000),         # what a test harness reports: not an IP
    ("127.0.0.1:1234", 1),         # not a parseable IP either
    ("", 1),
]

# Scopes that carry no usable peer address. Under the old gate these were refused
# ("we cannot prove loopback, so we do not serve"); under a switch they are
# irrelevant, which is the point of keeping them in the matrix.
UNPROVABLE_SCOPES = [
    {},                                            # no client key at all
    {"client": None},                              # a unix socket
    {"client": ()},                                # empty
    {"client": [None, None]},                      # address unknown
    {"client": "127.0.0.1"},                       # not a tuple
    "not-a-scope",                                 # not a dict
]

# The headers a caller could use to claim an identity. None of them was ever read for
# a decision, and none of them is read now - the difference is that there is no
# longer any decision for them to be absent from.
FORGED_HEADERS = [
    {"x-forwarded-for": "127.0.0.1"},
    {"x-forwarded-for": "127.0.0.1, 127.0.0.1"},
    {"x-real-ip": "127.0.0.1"},
    {"x-kvtier-dev-token": "s3cr3t-token"},
]


class FakeRequest:
    """Just enough ASGI request for `guard_route`: a scope and some headers.

    Deliberately carries BOTH fields a guard could consult, so a reintroduced
    peer-or-header check is exercised by these tests instead of passing them by
    being unreachable. `guard_route` reads neither.
    """

    def __init__(self, client=("127.0.0.1", 43210), headers=None, scope=None
                 ) -> None:
        self.scope = {"type": "http", "client": client,
                      "headers": list((headers or {}).items())
                      } if scope is None else scope
        self.headers = dict(headers or {})


def gate(client=("127.0.0.1", 43210), headers=None):
    """Call the gate, returning the HTTPException it raises, or None when it serves."""
    try:
        endpoint.guard_route(FakeRequest(client, headers))
    except HTTPException as exc:
        return exc
    return None


# --- the peer gate is GONE from the product ---------------------------------

# Every name the peer-address gate was made of. All five are production code that
# 0.2.72 deleted, plus the exception type only it raised; listing them here is what
# stops any of them coming back as a function no route calls.
DEAD_PEER_GATES = ("loopback_denial", "require_loopback", "_peer_address_unwrapped",
                   "peer_denial", "require_peer", "GuardDenied")


@pytest.mark.parametrize("name", DEAD_PEER_GATES)
def test_the_peer_guards_are_gone_from_the_product(name):
    assert not hasattr(endpoint, name), (
        f"endpoint.{name} is reachable again - the peer-address gate was removed in "
        "0.2.72 because it could not fail (a local proxy IS the loopback peer), and a "
        "helper with no caller is a lie about the gate")


@pytest.mark.parametrize("name", ("allow_networks", "ENV_ALLOW"))
def test_the_allowlist_machinery_is_gone_from_config(name):
    """`allow_networks()` is deleted, not stubbed to return nothing.

    A function that returns an empty allowlist would keep the shape of the old gate
    alive for any caller that found it, which is the dead-config lie in its most
    convincing form: the name still reads like a control.
    """
    assert not hasattr(config, name), f"config.{name} is reachable again"


def test_the_token_guards_and_their_imports_are_absent_from_the_package():
    """Source-level, not name-level: the machinery of both removed gates is gone.

    `hmac` was the token comparison; `ipaddress` was the peer parser and the
    CIDR containment test. Either reappearing in `endpoint.py` means something
    decides access from a secret or from an address again, and neither is true of
    this surface any more.
    """
    import inspect  # noqa: PLC0415 - one assertion

    assert not hasattr(endpoint, "hmac")
    assert not hasattr(config, "mutating_token")
    source = inspect.getsource(endpoint)
    assert "hmac" not in source
    assert "mutating_token()" not in source
    assert "ipaddress" not in source, (
        "endpoint.py parses addresses again - the peer gate was removed in 0.2.72")


# --- the gate reads no address and no header (structural) -------------------

def test_the_gate_reads_no_address_and_no_headers_source_level():
    """Structural check: `guard_route()` has no way to reach a header or a peer.

    The same property the old file asserted about `loopback_denial` and
    `peer_denial`, moved onto the one gate that remains - because the invariant did
    not change ("nothing decides from who you claim to be"), only the number of
    places that could violate it did. With the docstring removed, the body must
    contain no header access, no `scope["client"]` read and no address parsing.
    """
    import ast
    import inspect  # noqa: PLC0415
    import textwrap

    source = textwrap.dedent(inspect.getsource(endpoint.guard_route))
    body = ast.parse(source).body[0]
    docstring = ast.get_docstring(body, clean=False) or ""
    assert docstring in source
    code = source.replace(docstring, "")
    for forbidden in ("headers", "forwarded", "real_ip", "X-Real", "client",
                      "scope", "ipaddress", "loopback"):
        assert forbidden.lower() not in code.lower(), forbidden
    assert "enabled" in code  # ... and it does read the one switch that is the gate


def test_no_route_or_helper_in_endpoint_decides_from_a_header():
    """The package-level version, by AST: nothing in `endpoint.py` touches a header.

    Asserted on the syntax tree rather than on the text, because the file's own
    docstrings SAY the words this rule is about - prose naming a forbidden thing is
    how a text grep turns a correct file into a failing one. `TOKEN_HEADER` survives
    as a NAME to be reported (an operator's script still sends it, and `/status` must
    be able to say which header is ignored) and `route_reload_policy` LOGS the
    scope's client address for audit. Neither reads a header, and this is what makes
    that a fact rather than a claim.
    """
    import ast
    import inspect  # noqa: PLC0415

    tree = ast.parse(inspect.getsource(endpoint))
    on_request = {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute)
                  and isinstance(node.value, ast.Name)
                  and node.value.id == "request"}
    assert "headers" not in on_request, sorted(on_request)
    assert not {attr for attr in on_request if "forward" in attr.lower()}, on_request
    # The peer parser's own machinery is gone from the import list, not just unused.
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    assert "ipaddress" not in imported, sorted(imported)
    assert "hmac" not in imported, sorted(imported)


# --- the switch answers for every peer --------------------------------------

@pytest.mark.parametrize("peer", EVERY_PEER)
def test_inert_is_503_for_every_peer(peer, monkeypatch):
    """The whole answer, for every address this file can name: 503, and it names the
    switch. Nothing about the caller changes it."""
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    refused = gate(peer)
    assert refused is not None and refused.status_code == 503, peer
    assert config.ENV_ENABLE in refused.detail


@pytest.mark.parametrize("scope", UNPROVABLE_SCOPES)
def test_inert_is_503_for_a_scope_that_proves_nothing(scope, monkeypatch):
    """A scope with no peer address is no longer a refusal of its own - there is no
    peer question to fail. It gets the same 503 as everyone else when off."""
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    try:
        endpoint.guard_route(FakeRequest(scope=scope))
        pytest.fail("expected the inert 503")
    except HTTPException as exc:
        assert exc.status_code == 503


@pytest.mark.parametrize("peer", EVERY_PEER)
def test_enabled_serves_every_peer(peer, monkeypatch):
    """THE new contract, at the guard level: with the switch on, no address is
    filtered. This is not a wish - it is the exposure the operator opted into, and it
    is asserted here so that a peer check reappearing by accident fails loudly."""
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    assert gate(peer) is None, peer


@pytest.mark.parametrize("peer", EVERY_PEER)
@pytest.mark.parametrize("headers", FORGED_HEADERS)
def test_no_header_grants_anything_and_withholds_anything(peer, headers,
                                                           monkeypatch):
    """Headers are not part of the answer, in either direction.

    Under the old gate the interesting half was "a forged X-Forwarded-For buys a
    stranger nothing". There is no stranger and no insider any more, so the property
    is the flat one: with the switch on, every peer with every one of these headers
    is served exactly as it would be without them; with it off, none of them opens
    the 503.
    """
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    assert gate(peer, headers) is None, (peer, headers)
    monkeypatch.delenv(config.ENV_ENABLE)
    refused = gate(peer, headers)
    assert refused is not None and refused.status_code == 503, (peer, headers)


# --- the exposure is stated where the opt-in is read ------------------------

def test_the_inert_answer_states_what_enabling_opens(monkeypatch):
    """The 503 is the last place an operator is told BEFORE the surface is open.

    Not prose padding: every clause is a fact about the deployment - the engine
    binds 0.0.0.0, `--api-key` does not cover `/plugins/*`, there is no token and no
    address filter, and the protection that limits the damage is network-level and
    not provided here. A 503 that only said "inert" would leave the operator to
    discover the rest from a request that worked.
    """
    monkeypatch.delenv(config.ENV_ENABLE, raising=False)
    detail = gate().detail
    assert config.ENV_ENABLE in detail
    assert config.ROUTE_PREFIX in detail
    assert "every client" in detail.lower()
    assert "--api-key" in detail and "/plugins" in detail
    assert "no token" in detail.lower()
    assert "operator" in detail.lower(), "must name whose job the protection is"
    assert "s3cr3t" not in detail          # nothing here echoes a value
    assert config.ENV_TOKEN not in detail  # and it is not about the dead token


def test_the_status_payload_reports_the_peer_gate_as_removed(monkeypatch):
    """`/status` is the read an operator takes AFTER enabling, and it must carry the
    same sentence the boot logged: the peer gate is gone, what replaced it is an
    opt-in, and the network half is theirs.

    `loopback_only`/`allow_networks` used to answer "who can reach this?" and are
    deleted rather than answered falsely - see the comment at the `peer_gate` block
    in `endpoint.status_payload`.
    """
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    body = endpoint.status_payload()
    assert body["enabled"] is True
    gate_block = body["peer_gate"]
    assert gate_block["removed"] is True
    assert gate_block["peers_filtered_by_address"] is False
    assert gate_block["headers_grant_access"] is False
    assert "opt-in" in gate_block["replaced_by"]
    assert "firewall" in gate_block["operators_job"]
    assert "loopback_only" not in body
    assert "allow_networks" not in body
    # `token_note` cites `exposure` by name, so the key must exist and must carry the
    # three facts a reader needs: who can reach it, that --api-key is not it, and whose
    # job the network half is. A payload pointing at a key it does not publish is the
    # same class of lie as a boot line naming a variable nobody set.
    assert body["exposure"]
    assert "EVERY client" in body["exposure"]
    assert "--api-key" in body["exposure"]
    assert "operator's job" in body["exposure"]
    assert "exposure" in body["token_note"]
    # The safe-method matrix is published from the rule's owner, so it cannot drift.
    from kvtier_dev_plugin import route_policy
    assert body["mutating_ops_refused_on_get"]["disk"] == list(
        route_policy.mutating_ops(config.ROUTE_DISK))
    assert "arm" in body["mutating_ops_refused_on_get"]["guard"]
    assert "dump" in body["mutating_ops_refused_on_get"]["blockdump"]


# --- the two dead variables are reported and read by nothing ----------------

DEAD_TOKEN_GATES = ("require_bearer", "require_mutating_token",
                    "require_dev_state_token", "require_token_for_remote_read",
                    "peer_is_loopback")


@pytest.mark.parametrize("name", DEAD_TOKEN_GATES)
def test_the_token_guards_are_gone_from_the_product(name):
    assert not hasattr(endpoint, name), (
        f"endpoint.{name} is reachable again - the token gate was removed in 0.2.68 "
        "and a helper with no caller is a lie about the gate")


def test_the_token_env_reaches_the_report_and_nothing_else():
    """Where `VLLM_E2_TIER_DEV_TOKEN` may still be touched in the package.

    It is REPORTED (status, and the boot warning) and never READ for access. This
    enumerates the files that mention it, so a new `os.environ.get(ENV_TOKEN)` in
    some future module is a failing test rather than a silent second reader.
    """
    import pathlib  # noqa: PLC0415

    package = pathlib.Path(config.__file__).parent
    readers = sorted(path.name for path in package.glob("*.py")
                     if config.ENV_TOKEN in path.read_text(encoding="utf-8"))
    # config.py DEFINES the name and owns `token_configured()`; endpoint.py is the
    # only consumer, and both of its uses are reporting.
    assert readers == ["config.py", "endpoint.py"]


def test_the_retired_allowlist_reaches_the_report_and_nothing_else():
    """The same enumeration for `VLLM_E2_TIER_DEV_ALLOW` (retired 0.2.72).

    Two files may name it: the one that defines the retired name and owns
    `allow_list_configured()`, and the one that reports it. A third file naming it
    is either a second reader or a doc string that has drifted - and a variable that
    is parsed again somewhere is a gate that came back.
    """
    import pathlib  # noqa: PLC0415

    package = pathlib.Path(config.__file__).parent
    readers = sorted(path.name for path in package.glob("*.py")
                     if config.ENV_ALLOW_RETIRED in path.read_text(encoding="utf-8"))
    assert readers == ["config.py", "endpoint.py"]


@pytest.mark.parametrize("token_env", [None, "s3cr3t-token", "   ", "wrong"])
def test_the_token_env_grants_nothing_and_refuses_nothing(token_env, monkeypatch):
    """The rule of 0.2.68, restated against the gate of 0.2.72: the token variable
    is inert, and so is everything it used to be paired with.

    Same switch, four values of the token variable (absent, set, blank, arbitrary) -
    and no answer moves. There is no peer to be admitted and no header to be
    compared, so the variable can neither open the surface nor close it.
    """
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    if token_env is None:
        monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    else:
        monkeypatch.setenv(config.ENV_TOKEN, token_env)
    assert gate(("192.168.2.64", 1)) is None
    assert gate(("203.0.113.9", 1), {"x-kvtier-dev-token": "s3cr3t-token"}) is None


@pytest.mark.parametrize("allow_env", [None, "0.0.0.0/0", "192.168.2.64", "garbage"])
def test_the_retired_allowlist_admits_nobody_and_blocks_nobody(allow_env,
                                                               monkeypatch):
    """The whole point of retiring the variable: whatever it says, the answer is the
    same, because nothing parses it any more.

    `0.0.0.0/0` used to mean "everyone is admitted" and `garbage` used to mean
    "nobody is". Both now mean "the switch decides", which is the only reading that
    survives a proxy in front of the port.
    """
    if allow_env is None:
        monkeypatch.delenv(config.ENV_ALLOW_RETIRED, raising=False)
    else:
        monkeypatch.setenv(config.ENV_ALLOW_RETIRED, allow_env)
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    assert gate(("192.168.2.64", 1)) is None
    assert gate(("10.9.8.7", 1)) is None
    assert gate(("testclient", 1)) is None
    assert config.allow_list_configured() is bool(allow_env and allow_env.strip())
    monkeypatch.delenv(config.ENV_ENABLE)
    assert gate(("192.168.2.64", 1)).status_code == 503


def test_allow_list_configured_still_answers_the_env_question():
    """The reporting function survives, and must say in its own docstring that it
    gates nothing - the `token_configured()` precedent it was copied from."""
    assert config.allow_list_configured() is False
    assert config.allow_list_configured.__doc__ is not None
    assert "gates nothing" in config.allow_list_configured.__doc__.lower()


def test_token_configured_still_answers_the_env_question():
    assert config.token_configured() is False
    assert config.token_configured.__doc__ is not None
    assert "no longer" in config.token_configured.__doc__.lower()


# --- gating -----------------------------------------------------------------

def test_inert_by_default():
    assert config.enabled() is False


@pytest.mark.parametrize("value", ["", "0", "false", "OFF", "no", "  0 "])
def test_falsy_switch_values_keep_it_inert(value, monkeypatch):
    monkeypatch.setenv(config.ENV_ENABLE, value)
    assert config.enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
def test_other_switch_values_enable(value, monkeypatch):
    monkeypatch.setenv(config.ENV_ENABLE, value)
    assert config.enabled() is True


def test_timeout_defaults_and_clamps(monkeypatch):
    assert config.timeout_seconds() == config.DEFAULT_TIMEOUT_SECONDS
    monkeypatch.setenv(config.ENV_TIMEOUT, "0.001")   # below the floor
    assert config.timeout_seconds() == config.DEFAULT_TIMEOUT_SECONDS
    monkeypatch.setenv(config.ENV_TIMEOUT, "3600")    # above the ceiling
    assert config.timeout_seconds() == config.DEFAULT_TIMEOUT_SECONDS
    monkeypatch.setenv(config.ENV_TIMEOUT, "nonsense")
    assert config.timeout_seconds() == config.DEFAULT_TIMEOUT_SECONDS
    monkeypatch.setenv(config.ENV_TIMEOUT, "1.5")
    assert config.timeout_seconds() == 1.5
