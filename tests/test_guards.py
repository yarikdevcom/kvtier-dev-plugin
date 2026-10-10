"""The guards, in isolation: no FastAPI, no engine, no vllm.

`loopback_denial()` and `peer_denial()` are deliberately framework free (they
take an ASGI scope), which is what lets the security properties be tested here
directly instead of through a transport.

SINCE 0.2.68 THE PEER ALLOWLIST IS THE ONLY GATE ON THIS SURFACE. There used to
be a second guard here - `require_bearer()` and the three helpers built on it -
and this file used to test them. They are gone from the product and from this
file; what replaced them is `test_the_token_guards_are_gone_*` below (a deleted
gate must not survive as an unreachable function) and
`test_the_token_env_grants_nothing_and_refuses_nothing` (a variable that is read
by nothing must also be able to HURT nothing).
"""

from __future__ import annotations

import pytest

from kvtier_dev_plugin import config, endpoint
from kvtier_dev_plugin.endpoint import (
    GuardDenied,
    loopback_denial,
    peer_denial,
    require_loopback,
    require_peer,
)


def scope(client):
    return {"type": "http", "client": client, "headers": []}


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


# --- the token is GONE from the gate (0.2.68) --------------------------------
#
# Two directions of the same lie are pinned here. A gate that was removed but is
# still read (`require_bearer()` with no route behind it) and a name that was
# removed but is still reachable (a helper nobody calls). The first is the
# incident this repo is named for; the second is its mirror image, and both are
# checked against the SOURCE of the package rather than against one input, so a
# future edit cannot re-add a reader or a reader-less helper unnoticed.

# Every name the bearer-token gate was made of. All five are production code that
# 0.2.68 deleted; listing them here is what stops any of them coming back as a
# function no route calls.
DEAD_TOKEN_GATES = ("require_bearer", "require_mutating_token",
                    "require_dev_state_token", "require_token_for_remote_read",
                    "peer_is_loopback")


@pytest.mark.parametrize("name", DEAD_TOKEN_GATES)
def test_the_token_guards_are_gone_from_the_product(name):
    assert not hasattr(endpoint, name), (
        f"endpoint.{name} is reachable again - the token gate was removed in 0.2.68 "
        "and a helper with no caller is a lie about the gate")


def test_the_token_helpers_and_their_imports_are_absent_from_the_package():
    """Source-level, not name-level: `hmac` was the token comparison and is gone.

    A module-level `import hmac` in `endpoint.py` would mean something still
    compares a secret, and `config.mutating_token()` was the one reader of the
    token env - both must stay deleted.
    """
    assert not hasattr(endpoint, "hmac")
    assert not hasattr(config, "mutating_token")
    import inspect  # noqa: PLC0415 - one assertion

    source = inspect.getsource(endpoint)
    assert "hmac" not in source
    assert "mutating_token()" not in source


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


@pytest.mark.parametrize("token_env", [None, "s3cr3t-token", "   ", "wrong"])
def test_the_token_env_grants_nothing_and_refuses_nothing(token_env, monkeypatch):
    """THE rule of 0.2.68, at the guard level: the token variable is inert.

    Same peer, same allowlist, four values of VLLM_E2_TIER_DEV_TOKEN (absent,
    correctly set, blank, arbitrary) - and no header is passed at all, because
    the gate cannot ask for one. An allowlisted peer is admitted under every
    value; the variable can neither open the peer gate nor close it.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.64/32")
    if token_env is None:
        monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    else:
        monkeypatch.setenv(config.ENV_TOKEN, token_env)
    peer = scope(("192.168.2.64", 1))
    assert peer_denial(peer) is None
    require_peer(peer)                                # must not raise
    # And loopback is admitted the same way, whatever the variable says.
    require_peer(scope(("127.0.0.1", 1)))


@pytest.mark.parametrize("token_env", [None, "s3cr3t-token", "wrong"])
def test_a_refused_peer_is_refused_on_peer_grounds_alone(token_env, monkeypatch):
    """The refusal a stranger sees names the PEER rule, with no token involved.

    Before 0.2.68 two guards could both say 403 on one request and the messages
    were the only way to tell them apart. There is now one guard, so every refusal
    of a non-allowlisted peer must be the peer message - and must not be able to
    mention a token, because a stranger reading "missing or wrong <header>" would
    go looking for a knob that no longer exists.
    """
    monkeypatch.setenv(config.ENV_ALLOW, "192.168.2.0/24")
    if token_env is None:
        monkeypatch.delenv(config.ENV_TOKEN, raising=False)
    else:
        monkeypatch.setenv(config.ENV_TOKEN, token_env)
    hostile = scope(("203.0.113.9", 51234))
    denial = peer_denial(hostile)
    assert denial is not None
    assert "not loopback" in denial and config.ENV_ALLOW in denial
    assert "203.0.113.9" in denial
    assert "token" not in denial.lower()
    assert config.ENV_TOKEN not in denial
    with pytest.raises(GuardDenied) as excinfo:
        require_peer(hostile)
    assert excinfo.value.status_code == 403
    assert "token" not in excinfo.value.detail.lower()


def test_token_configured_still_answers_the_env_question():
    """`token_configured()` survives as the REPORT's source of truth (say so rather
    than disappear quietly), so it must still read the env honestly - and must say
    in its own docstring that it no longer gates anything.
    """
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
