"""Environment gating and the one path this plugin is ever allowed to load.

Import of this module has no side effects: it reads no environment value at
import time. Every gate is a function called per request / per plugin load, so
the operator can turn the plugin on and off with an env var and a restart of
the *reader* of that env (the engine process), never a re-install.

Env vars (this fork's `VLLM_E2_*` diagnostic family; see vllm/envs.py:392-456
for `VLLM_E2_MEMSTATS`, `VLLM_E2_SCHED_LOG`, `VLLM_E2_GUARD_MODE`, ...):

  VLLM_E2_TIER_DEV          -- master switch. Anything not in
                               {"", "0", "false", "off", "no"} enables it.
                               Unset = the plugin is inert: the general plugin
                               attaches nothing, the endpoint answers 503.
  VLLM_E2_TIER_DEV_TOKEN    -- required bearer token for the ONE mutating verb
                               (policy reload). Unset/empty = reload is refused
                               (fail closed), reads still work.
  VLLM_E2_TIER_DEV_TIMEOUT  -- seconds the HTTP layer waits for the engine
                               utility call before answering 504. Default 5.0,
                               clamped to [0.05, 60.0]; garbage falls back.
  VLLM_E2_TIER_DEV_ALLOW    -- comma-separated list of IP addresses or CIDR
                               networks allowed in ADDITION to loopback, e.g.
                               `192.168.2.64,192.168.2.120` or `192.168.2.0/24`.
                               UNSET OR EMPTY = LOOPBACK ONLY, which is the
                               pre-allowlist behaviour and the safe default.
                               A malformed entry is logged and skipped, never
                               fatal and never an opening. A peer admitted here
                               must still present VLLM_E2_TIER_DEV_TOKEN on
                               every request, reads included (loopback peers do
                               not need it for reads). Traffic is plaintext
                               HTTP, so keep this list as narrow as possible.

There is deliberately NO env var that selects the policy file. The reloadable
policy is `kvtier_dev_plugin/policy.py` next to this file, and nothing else.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path

ENV_ENABLE = "VLLM_E2_TIER_DEV"
ENV_TOKEN = "VLLM_E2_TIER_DEV_TOKEN"  # noqa: S105 - a variable NAME, not a secret
ENV_TIMEOUT = "VLLM_E2_TIER_DEV_TIMEOUT"
ENV_ALLOW = "VLLM_E2_TIER_DEV_ALLOW"  # empty/unset = loopback only

logger = logging.getLogger("kvtier_dev_plugin.config")

# Engine-side method names. `EngineCore` dispatches a UTILITY message with
# `getattr(self, method_name)` (vllm/v1/engine/core.py:1669-1671), so these
# must be unique on the EngineCore class. The `kvtier_dev_` prefix is the only
# name collision protection we have; `grep -rn kvtier vllm/` finds nothing.
METHOD_SNAPSHOT = "kvtier_dev_snapshot"
METHOD_RELOAD_POLICY = "kvtier_dev_reload_policy"

# All HTTP surface lives here, never under /v1 (see docs/usage/security.md:349:
# `--api-key` only protects the /v1, /v2, /inference, /cohere prefixes).
ROUTE_PREFIX = "/plugins/kvtier"

# Name used for logs, for `VLLM_PLUGINS` allowlisting and as the entry-point
# name in both groups. One name allowlists both halves at once.
PLUGIN_NAME = "kvtier_dev"

DEFAULT_TIMEOUT_SECONDS = 5.0
MIN_TIMEOUT_SECONDS = 0.05
MAX_TIMEOUT_SECONDS = 60.0

_PKG_DIR = Path(__file__).resolve().parent
_FALSY = frozenset({"", "0", "false", "off", "no"})

# The single-slot allowlist cache. `_ALLOW_CACHE_RAW` starts as None, which no
# env value can equal, so the first call always parses.
_ALLOW_CACHE_RAW: str | None = None
_ALLOW_CACHE_VALUE: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()


class PolicyPathError(ValueError):
    """A requested policy path was refused. Carries the human-readable why."""


def enabled() -> bool:
    """Master switch. False unless VLLM_E2_TIER_DEV says otherwise."""
    return os.environ.get(ENV_ENABLE, "").strip().lower() not in _FALSY


def token_configured() -> bool:
    return bool(os.environ.get(ENV_TOKEN, "").strip())


def mutating_token() -> str:
    """The token a mutating request must present. "" means "reload refused"."""
    return os.environ.get(ENV_TOKEN, "").strip()


def allow_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Extra peer networks admitted by VLLM_E2_TIER_DEV_ALLOW.

    An EMPTY tuple means "loopback only" - today's behaviour, and the default
    when the variable is unset, empty, or entirely garbage. Nothing here ever
    widens access on a parse failure: an entry that does not parse contributes
    nothing to the result, so a typo in the env opens a door for nobody.

    Both a bare address (`192.168.2.64`, stored as a /32) and a network
    (`192.168.2.0/24`) are accepted, hence `strict=False`. Loopback is NOT
    added here; the guard always accepts loopback and uses this only to widen.

    Cost: this runs per request in the HTTP process, so the parse is memoised
    in a single slot keyed on the raw env string - no lock (a torn read of two
    module globals is impossible in CPython, and the assignment order below
    means a reader that matches on the key always sees the matching value), and
    an env change invalidates it by definition because the key IS the env.
    """
    global _ALLOW_CACHE_RAW, _ALLOW_CACHE_VALUE
    raw = os.environ.get(ENV_ALLOW, "")
    if raw == _ALLOW_CACHE_RAW:
        return _ALLOW_CACHE_VALUE
    parsed: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            parsed.append(ipaddress.ip_network(entry, strict=False))
        except ValueError as exc:
            # Never fatal, never an opening: the entry is simply absent.
            logger.warning("[KVTIER_DEV] ignoring malformed %s entry %r (%s)",
                           ENV_ALLOW, entry, exc)
    # Value before key: a concurrent reader can only ever pair a raw string
    # with the value parsed from that same raw string.
    _ALLOW_CACHE_VALUE = tuple(parsed)
    _ALLOW_CACHE_RAW = raw
    return _ALLOW_CACHE_VALUE


def timeout_seconds() -> float:
    raw = os.environ.get(ENV_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    if not (MIN_TIMEOUT_SECONDS <= value <= MAX_TIMEOUT_SECONDS):
        return DEFAULT_TIMEOUT_SECONDS
    return value


def package_dir() -> Path:
    """The directory this plugin was imported from (resolved, symlinks gone)."""
    return _PKG_DIR


def canonical_policy_path() -> Path:
    """The only file the reload verb can ever load."""
    return _PKG_DIR / "policy.py"


def validate_policy_target(candidate: object) -> Path:
    """Return the resolved Path for `candidate`, or raise PolicyPathError.

    The rule is deliberately narrower than "inside the plugin directory":
    the target must resolve to exactly `<package dir>/policy.py`. That kills
    `../`, symlinks out of the tree, `policy2.py`, module names like
    `os.path`, and absolute paths to anything else - from the wire or from
    anywhere else. Nothing here trusts the caller.
    """
    if not isinstance(candidate, (str, Path)):
        raise PolicyPathError("policy path must be a string, got "
                              f"{type(candidate).__name__}")
    text = str(candidate).strip()
    if not text:
        raise PolicyPathError("policy path is empty")
    if "\x00" in text:
        raise PolicyPathError("policy path contains a NUL byte")

    path = Path(text)
    if not path.is_absolute():
        path = _PKG_DIR / path
    try:
        # strict=False: a missing file must be reported by name below, not as
        # an OSError from resolve(). resolve() follows symlinks, which is what
        # makes the parent/name comparison below a real containment check.
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:  # ELOOP, deep nesting, ...
        raise PolicyPathError(f"policy path does not resolve: {exc}") from exc

    if resolved.parent != _PKG_DIR or resolved.name != "policy.py":
        raise PolicyPathError(
            "refused: the only loadable policy is "
            f"{str(canonical_policy_path())!r}, not {str(resolved)!r}"
        )
    if not resolved.is_file():
        raise PolicyPathError(f"refused: {str(resolved)!r} is not a file")
    return resolved
