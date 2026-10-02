"""The guards, in isolation: no FastAPI, no engine, no vllm.

`loopback_denial()`, `peer_denial()`, `require_token_for_remote_read()` and
`require_mutating_token()` are deliberately framework free (they take an ASGI
scope / a header mapping), which is what lets the security properties be tested
here directly instead of through a transport.
"""

from __future__ import annotations

import pytest

from kvtier_dev_plugin import config
from kvtier_dev_plugin.endpoint import (
    TOKEN_HEADER,
    GuardDenied,
    loopback_denial,
    peer_denial,
    peer_is_loopback,
    require_loopback,
    require_mutating_token,
    require_peer,
    require_token_for_remote_read,
)


def scope(client):
    return {"type": "http", "client": client, "headers": []}


class FakeHeaders(dict):
    """Starlette's Headers are a case-insensitive mapping; a plain dict keyed
    the way the router lowercases them is enough for these checks."""


# --- loopback ---------------------------------------------------------------

@pytest.mark.parametrize("peer", [
    ("127.0.0.1", 50000),
    ("127.0.0.2", 1),          # the whole 127/8 is loopback
    ("127.255.255.254", 1),
    ("::1", 1),
    ("::ffff:127.0.0.1", 1),   # dual-stack listener reporting an IPv4 peer
])
def test_loopback_peers_are_accepted(peer):
    assert loopback_denial(scope(peer)) is None
    require_loopback(scope(peer))  # must not raise


@pytest.mark.parametrize("peer", [
    ("10.0.0.7", 1),
    ("169.254.1.1", 1),
    ("192.168.2.42", 443),     # the GPU box itself is not loopback
    ("0.0.0.0", 1),
    ("::", 1),
    ("fe80::1", 1),
    ("::ffff:10.0.0.7", 1),
    ("testclient", 50000),     # what a test harness reports: not an IP
    ("127.0.0.1:1234", 1),     # not a parseable IP either
    ("", 1),
])
def test_non_loopback_peers_are_refused(peer):
    denial = loopback_denial(scope(peer))
    assert denial is not None
    with pytest.raises(GuardDenied) as excinfo:
        require_loopback(scope(peer))
    assert excinfo.value.status_code == 403


@pytest.mark.parametrize("bad_scope", [
    {},                                                 # no client key at all
    {"client": None},                                   # unix socket
    {"client": ()},                                     # empty
    {"client": [None, None]},                            # address unknown
    {"client": "127.0.0.1"},                            # not a tuple
    "not-a-scope",                                       # not a dict
])
def test_a_scope_that_cannot_prove_loopback_is_refused(bad_scope):
    # "we cannot prove it" must never be "serve it anyway".
    assert loopback_denial(bad_scope) is not None
    with pytest.raises(GuardDenied):
        require_loopback(bad_scope)


def test_a_forged_forwarded_header_grants_nothing():
    """The guard reads the peer, not the headers.

    A scope with a remote peer and an X-Forwarded-For claiming loopback must
    still be refused, and the denial must not even mention the header: nothing
    in the guard looks at it.
    """
    hostile = {
        "type": "http",
        "client": ("203.0.113.9", 51234),
        "headers": [(b"x-forwarded-for", b"127.0.0.1"),
                    (b"x-real-ip", b"127.0.0.1")],
    }
    denial = loopback_denial(hostile)
    assert denial is not None and "203.0.113.9" in denial
    with pytest.raises(GuardDenied) as excinfo:
        require_loopback(hostile)
    assert excinfo.value.status_code == 403


def test_guard_never_reads_headers_source_level():
    """Structural check: loopback_denial() has no way to reach a header.

    Same property as above, but asserted on the code rather than on one input:
    with the docstring removed, the body must contain no header access at all.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(loopback_denial)))
    body = tree.body[0]
    docstring = ast.get_docstring(body, clean=False) or ""
    source = textwrap.dedent(inspect.getsource(loopback_denial))
    assert docstring in source
    code = source.replace(docstring, "")
    for forbidden in ("headers", "forwarded", "real_ip", "X-Real"):
        assert forbidden.lower() not in code.lower(), forbidden
    assert "client" in code  # ... and it does read the peer


def test_peer_denial_never_reads_headers_source_level():
    """The allowlist half of the gate is header-blind for the same reason."""
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(peer_denial))
    body = ast.parse(source).body[0]
    docstring = ast.get_docstring(body, clean=False) or ""
    assert docstring in source
    code = source.replace(docstring, "")
    for forbidden in ("headers", "forwarded", "real_ip", "X-Real"):
        assert forbidden.lower() not in code.lower(), forbidden
    assert "allow_networks" in code  # ... and it does read the allowlist


# --- the allowlist (VLLM_E2_TIER_DEV_ALLOW) ----------------------------------

def test_unset_allow_is_empty_and_means_loopback_only(monkeypatch):
    """THE regression guard for the old default: no env, no widening."""
    monkeypatch.delenv(config.ENV_ALLOW, raising=False)
    assert config.allow_networks() == ()
    assert peer_denial(scope(("192.168.2.64", 1))) is not None
    assert peer_denial(scope(("127.0.0.1", 1))) is None


def test_bare_address_and_cidr_both_parse_and_normalise(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW,
                       "192.168.2.64, 192.168.2.120/32 , 10.7.0.0/16")
    assert [str(n) for n in config.allow_networks()] == [
        "192.168.2.64/32", "192.168.2.120/32", "10.7.0.0/16"]
    # strict=False: a host-in-network entry is the network it belongs to.
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/24")
    assert [str(n) for n in config.allow_networks()] == ["192.168.2.0/24"]


def test_cache_follows_the_env_and_a_shrink_shrinks_immediately(monkeypatch):
    """The memo is keyed on the raw env string, so an env change IS the
    invalidation - including the narrowing direction, which must never be
    served from a stale wider parse."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    assert len(config.allow_networks()) == 1
    monkeypatch.setenv(config.ENV_ALLOW, "")
    assert config.allow_networks() == ()
    monkeypatch.setenv(config.ENV_ALLOW, "10.0.0.0/8")
    assert [str(n) for n in config.allow_networks()] == ["10.0.0.0/8"]
    monkeypatch.delenv(config.ENV_ALLOW)
    assert config.allow_networks() == ()


@pytest.mark.parametrize("env", [
    "garbage",
    "",
    "   ",
    "10.0.0.0/99",            # prefix length no version has
    "192.168.2.999",          # octet out of range
    "192.168.2.0/24/32",      # double slash
    "garbage,,  ,!!!",
    "localhost",              # a hostname is not an address: never resolved
    "::x",
])
def test_malformed_allow_entries_never_raise_and_never_admit(env, monkeypatch):
    """Garbage in the env opens a door for nobody and cannot break the request
    path: allow_networks() must not raise, and every probe stays refused."""
    monkeypatch.setenv(config.ENV_ALLOW, env)
    networks = config.allow_networks()          # must not raise
    assert networks == ()                       # nothing survived the parse
    for peer in (("10.0.0.5", 1), ("192.168.2.64", 1), ("192.168.2.999", 1),
                 ("::1", 1), ("fe80::2", 1)):
        denial = peer_denial(scope(peer))
        if peer[0] == "::1":                    # loopback is always allowed
            assert denial is None
        else:
            assert denial is not None, peer
            with pytest.raises(GuardDenied):
                require_peer(scope(peer))


def test_a_good_entry_survives_a_bad_neighbour(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "garbage,,192.168.2.64/32,10.0.0.0/99")
    assert [str(n) for n in config.allow_networks()] == ["192.168.2.64/32"]
    assert peer_denial(scope(("192.168.2.64", 1))) is None
    assert peer_denial(scope(("192.168.2.65", 1))) is not None
    assert peer_denial(scope(("10.0.0.5", 1))) is not None


def test_a_malformed_entry_is_warned_about_once_per_env_value(monkeypatch, caplog):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32,bogus-entry")
    with caplog.at_level("WARNING"):
        config.allow_networks()
        config.allow_networks()                 # cached: no second warning
    warnings = [r.getMessage() for r in caplog.records
                if r.levelname == "WARNING" and "bogus-entry" in r.getMessage()]
    assert len(warnings) == 1
    assert config.ENV_ALLOW in warnings[0]


def test_cidr_admits_inside_and_refuses_outside(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    for inside in ("192.168.2.1", "192.168.2.64", "192.168.2.255"):
        assert peer_denial(scope((inside, 1))) is None, inside
        require_peer(scope((inside, 1)))
    for outside in ("192.168.3.1", "192.168.1.254", "192.168.2.64.1"):
        denial = peer_denial(scope((outside, 1)))
        assert denial is not None, outside
        assert outside in denial                # the refusal names the peer
        with pytest.raises(GuardDenied) as excinfo:
            require_peer(scope((outside, 1)))
        assert excinfo.value.status_code == 403


def test_ipv6_allow_entries_work(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "fd00::/64")
    assert peer_denial(scope(("fd00::5", 1))) is None
    assert peer_denial(scope(("fd01::5", 1))) is not None
    # A v4 peer is never `in` a v6 network (and does not raise):
    assert peer_denial(scope(("192.168.2.64", 1))) is not None


def test_mapped_ipv4_peer_matches_the_ipv4_allowlist(monkeypatch):
    """The ::ffff: unwrap is shared, so a dual-stack listener's rendering of an
    allowlisted peer still matches the allowlist."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    assert peer_denial(scope(("::ffff:192.168.2.64", 1))) is None
    assert peer_denial(scope(("::ffff:10.0.0.7", 1))) is not None


def test_loopback_is_admitted_whatever_the_allowlist_says(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "10.0.0.0/8")
    for peer in (("127.0.0.1", 1), ("::1", 1), ("::ffff:127.0.0.1", 1)):
        assert peer_denial(scope(peer)) is None, peer


def test_a_denial_says_why_without_leaking_the_allowlist(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    denial = peer_denial(scope(("10.0.0.7", 1)))
    assert "not loopback" in denial
    assert config.ENV_ALLOW in denial and "allowlist" in denial
    assert "10.0.0.7" in denial
    assert "192.168.2.0" not in denial          # the entries are not leaked


@pytest.mark.parametrize("bad_scope", [
    {},
    {"client": None},
    {"client": [None, None]},
    {"client": "127.0.0.1"},
    "not-a-scope",
])
def test_an_allowlist_cannot_save_a_scope_that_proves_nothing(bad_scope, monkeypatch):
    """`0.0.0.0/0` in the env must not turn "no peer address" into "serve"."""
    monkeypatch.setenv(config.ENV_ALLOW, "0.0.0.0/0")
    assert peer_denial(bad_scope) is not None
    with pytest.raises(GuardDenied):
        require_peer(bad_scope)


def test_a_non_ip_peer_string_is_refused_even_when_allowlisted(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "0.0.0.0/0")
    # Unparseable -> no containment test is possible -> refuse, whatever the env.
    assert peer_denial(scope(("testclient", 1))) is not None
    assert peer_denial(scope(("127.0.0.1:1234", 1))) is not None


# --- the token on reads from a non-loopback peer -----------------------------

def test_remote_reads_need_the_token_but_loopback_reads_do_not(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    monkeypatch.setenv(config.ENV_TOKEN, "s3cr3t-token")
    # Loopback: the operator's `ssh llm curl 127.0.0.1` workflow, no token.
    require_token_for_remote_read(scope(("127.0.0.1", 1)), FakeHeaders({}))
    require_token_for_remote_read(scope(("::1", 1)), FakeHeaders({}))
    # Allowlisted non-loopback peer: the token is required, and works.
    with pytest.raises(GuardDenied) as excinfo:
        require_token_for_remote_read(scope(("192.168.2.64", 1)), FakeHeaders({}))
    assert excinfo.value.status_code == 403
    assert TOKEN_HEADER in excinfo.value.detail
    with pytest.raises(GuardDenied):
        require_token_for_remote_read(scope(("192.168.2.64", 1)),
                                      FakeHeaders({TOKEN_HEADER: "s3cr3t-toke"}))
    require_token_for_remote_read(scope(("192.168.2.64", 1)),
                                  FakeHeaders({TOKEN_HEADER: "s3cr3t-token"}))


def test_remote_read_is_refused_when_no_token_is_configured(monkeypatch):
    """Fail closed: an allowlisted peer with no VLLM_E2_TIER_DEV_TOKEN set gets
    nothing, because there is no way to prove who it is."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    assert config.token_configured() is False
    with pytest.raises(GuardDenied) as excinfo:
        require_token_for_remote_read(scope(("192.168.2.64", 1)),
                                      FakeHeaders({TOKEN_HEADER: "anything"}))
    assert excinfo.value.status_code == 403
    assert config.ENV_TOKEN in excinfo.value.detail


def test_an_allowlisted_peer_is_still_not_loopback(monkeypatch):
    """`peer_is_loopback()` answers the address question, not the access
    question: widening the allowlist must not silently widen token exemption."""
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    assert peer_is_loopback(scope(("192.168.2.64", 1))) is False
    assert peer_is_loopback(scope(("127.0.0.1", 1))) is True
    assert peer_is_loopback(scope(("203.0.113.9", 1))) is False


def test_mutating_token_message_is_unchanged_for_the_reload_verb(monkeypatch):
    """The shared token helper must not rewrite the reload verb's refusal."""
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    with pytest.raises(GuardDenied) as excinfo:
        require_mutating_token(FakeHeaders({}))
    assert excinfo.value.detail == f"missing or wrong {TOKEN_HEADER}"


# --- token ------------------------------------------------------------------

def test_reload_is_refused_when_no_token_is_configured(enabled):
    """Fail closed: an unset VLLM_E2_TIER_DEV_TOKEN means no mutation at all."""
    assert config.token_configured() is False
    with pytest.raises(GuardDenied) as excinfo:
        require_mutating_token(FakeHeaders({TOKEN_HEADER: "anything"}))
    assert excinfo.value.status_code == 403
    assert config.ENV_TOKEN in excinfo.value.detail


def test_token_must_match(enabled, monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "s3cr3t-token")
    with pytest.raises(GuardDenied):
        require_mutating_token(FakeHeaders({}))
    with pytest.raises(GuardDenied):
        require_mutating_token(FakeHeaders({TOKEN_HEADER: ""}))
    with pytest.raises(GuardDenied):
        require_mutating_token(FakeHeaders({TOKEN_HEADER: "s3cr3t-toke"}))
    with pytest.raises(GuardDenied):
        require_mutating_token(FakeHeaders({TOKEN_HEADER: "s3cr3t-token2"}))
    require_mutating_token(FakeHeaders({TOKEN_HEADER: "s3cr3t-token"}))


def test_whitespace_only_token_counts_as_unset(monkeypatch):
    monkeypatch.setenv(config.ENV_TOKEN, "   ")
    assert config.token_configured() is False
    with pytest.raises(GuardDenied):
        require_mutating_token(FakeHeaders({TOKEN_HEADER: "   "}))


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
