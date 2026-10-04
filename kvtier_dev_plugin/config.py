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
  VLLM_E2_TIER_DEV_BOOT_INERT -- THE BOOT OPT-OUT. UNSET (the default) = the
                               boot brings the tier up WORKING: at connector
                               creation the plugin installs PR #56810, arms GUARD
                               MODE, and arms the tier -- and only arms it, if the
                               install VERIFIES (all methods patched, live object
                               safe). Set it to anything not in
                               {"", "0", "false", "off", "no"} to get the old
                               inert boot back: patched at connector creation, no
                               guard, tier left disarmed until a human POSTs
                               {"op":"arm"}. It never ARMS anything by itself:
                               truthy = inert, falsy = auto. `status` and the
                               engine's BOOT log line both say which mode ran.
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

THE WORKER DIAGNOSTICS FAMILY (all optional; see `worker_diag.py`'s docstring for
the file contract the proxy host's reader expects):

  VLLM_E2_TIER_DIAG              where one JSON file per rank is written, on the
                                 ENGINE's host. UNSET = nothing is wrapped and no
                                 file is ever created, which is the default.
  VLLM_E2_TIER_DIAG_EPOCH        the boot id shared by every writer. Set it once
                                 per boot if you want the reader's cross-rank
                                 verdicts; unset, each process stamps its own id
                                 and says `writer.epoch_scope: "process"`.
  VLLM_E2_TIER_DIAG_INTERVAL     write cadence in steps (default 50, clamped).
  VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS
                                 the rate floor under it (default 1.0 s), so a
                                 changing signature cannot make this a per-step
                                 writer.
  VLLM_E2_TIER_DIAG_RANK         pin the rank id (`PP<n>_TP<m>`) instead of
                                 deriving it from the process's distributed state.
  VLLM_E2_TIER_MISS_LOG          a path to append one JSON line per tier lookup
                                 to. Unset = no file, no open(), no I/O of any
                                 kind (see miss_attribution.py).
  VLLM_E2_TIER_DISK_ROOT         the SSD tier's store root, one file per KV
                                 slot. UNSET = the tier cannot exist: no
                                 directory is created, no thread starts, no byte
                                 is written. tmpfs roots are refused.
  VLLM_E2_TIER_DISK_BYTES        the tier's byte ceiling in bytes; above it files
                                 are shed LRU by st_mtime. Unset = no ceiling,
                                 NEVER zero. Neither variable ARMS the tier: the
                                 arm is `disk_spill.arm()`, which expires by
                                 itself, exactly like the raw-byte dump's.
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
# THE BOOT OPT-OUT. Unset = the boot arms (guard + tier) once the install
# verifies; truthy = the old inert boot. The name deliberately says what it
# DISABLES rather than what it enables: no env var of this plugin ARMS guard mode,
# which is the property `test_no_env_var_can_arm_guard_mode_and_at_import_it_is_off`
# pins. The boot arms it; this variable only takes the boot back to inert.
ENV_BOOT_INERT = "VLLM_E2_TIER_DEV_BOOT_INERT"

# --- the worker-side diagnostics writer (`worker_diag.py`) ------------------
# The directory one JSON file per rank is written into, on the ENGINE's host. The
# proxy host has a reader for the same layout (`llm-api/kv_tier_diagnostics.py`);
# the operator ships the files over, or points both sides at a shared path.
# UNSET = the writer installs nothing and creates no files anywhere, which is the
# default even when every other part of the plugin is live.
ENV_DIAG_DIR = "VLLM_E2_TIER_DIAG"
# The boot id shared by every writer. The reader compares epochs across writers
# and declares an equality conclusion `meaningless` when they disagree, and a
# worker process CANNOT learn the EngineCore's boot id (the handshake flows one
# way: worker -> EngineCore). Set this once per boot in the launcher's
# environment so all ranks agree; unset, each process stamps its own id and the
# file says `writer.epoch_scope = "process"`.
ENV_DIAG_EPOCH = "VLLM_E2_TIER_DIAG_EPOCH"
# Write cadence: at most once every N steps or on a change of facts, and never
# more than once per MIN seconds. See `worker_diag`'s BUDGET section.
ENV_DIAG_INTERVAL = "VLLM_E2_TIER_DIAG_INTERVAL"
ENV_DIAG_MIN_INTERVAL = "VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS"
# Pin the rank id (`PP<n>_TP<m>`) instead of deriving it from the process's
# distributed state. Optional; a value in the wrong shape is refused loudly
# rather than written, because the reader classifies a file whose `rank` does not
# match its file name as mislabeled.
ENV_DIAG_RANK = "VLLM_E2_TIER_DIAG_RANK"
# THE RAW-BYTE DUMP (`blockdump.py`) HAS NO ENV VAR OF ITS OWN, and never will.
# Its channel is this directory; its arm is `POST /plugins/kvtier/blockdump
# {"op":"arm"}` and it expires by itself. That is deliberate, the same posture as
# guard mode: a launcher script must not be able to turn a facility that reads
# real KV bytes on. With this directory unset the dump verb reports the honest
# three-state marker as `not_deployed` and no worker hook is installed at all.
# THE SSD TIER (`disk_spill.py`) -----------------------------------------------
# Where the one-file-per-slot store lives, and how big it may get. BOTH are
# required for the tier to exist at all, and neither ARMS it: the arm is the
# in-process/arm-file gate in `disk_spill.arm()` (no env var arms a facility that
# writes real KV bytes - the `guard_mode`/`blockdump` rule, pinned by a test).
# Unset root = no directory is created, no thread starts, no byte is written.
# `/tmp`, `/dev/shm` and `/run` are refused by `disk_layout.validate_root`: on the
# box they are tmpfs, so a "disk" tier there spends the RAM the tier relieves.
ENV_DISK_ROOT = "VLLM_E2_TIER_DISK_ROOT"
# The ceiling in BYTES of stored slot files (a 64 GiB store is ~5,900 slots at
# 11.4 MB). Unset or unparseable = no ceiling (never 0: a 0 ceiling would mean
# "evict everything", which is a destructive reading of a typo).
ENV_DISK_BYTES = "VLLM_E2_TIER_DISK_BYTES"

# The miss-attribution ledger's OPTIONAL file mirror: a path to append one JSON
# line per tier lookup to. Unset = no file, no open(), no I/O of any kind. It is
# env-only on purpose: an HTTP argument naming a path the engine appends to would
# be a write surface this plugin does not have anywhere else. See
# miss_attribution.py (bounded by MAX_MIRROR_BYTES, which it stops at, counted).
ENV_MISS_LOG = "VLLM_E2_TIER_MISS_LOG"

logger = logging.getLogger("kvtier_dev_plugin.config")

# Engine-side method names. `EngineCore` dispatches a UTILITY message with
# `getattr(self, method_name)` (vllm/v1/engine/core.py:1669-1671), so these
# must be unique on the EngineCore class. The `kvtier_dev_` prefix is the only
# name collision protection we have; `grep -rn kvtier vllm/` finds nothing.
METHOD_SNAPSHOT = "kvtier_dev_snapshot"
METHOD_RELOAD_POLICY = "kvtier_dev_reload_policy"
METHOD_KVCONFIG = "kvtier_dev_kvconfig"
METHOD_INSPECT = "kvtier_dev_inspect"
METHOD_TRACE = "kvtier_dev_trace"
METHOD_GUARD = "kvtier_dev_guard"
METHOD_TIER = "kvtier_dev_tier"
METHOD_LOOKUPS = "kvtier_dev_lookups"
# The raw-byte dump verb. Read-only, bounded, and DISARMED until a human arms it
# (there is no env var that arms it): see blockdump.py.
METHOD_BLOCKDUMP = "kvtier_dev_blockdump"

# Every engine method this plugin owns, in attach order. `attach_methods` walks
# this, so a new verb cannot be attached in one place and forgotten elsewhere.
METHODS = (METHOD_SNAPSHOT, METHOD_RELOAD_POLICY, METHOD_KVCONFIG, METHOD_INSPECT,
           METHOD_TRACE, METHOD_GUARD, METHOD_TIER, METHOD_LOOKUPS,
           METHOD_BLOCKDUMP)

# All HTTP surface lives here, never under /v1 (see docs/usage/security.md:349:
# `--api-key` only protects the /v1, /v2, /inference, /cohere prefixes).
ROUTE_PREFIX = "/plugins/kvtier"

# Route names, for status and the tests. The verbs that CAN change dev state
# (`/guard`, `/trace`, `/tier`, `/lookups`) require the token in every shape,
# loopback included.
ROUTE_STATUS = "/status"
ROUTE_SNAPSHOT = "/snapshot"
ROUTE_RELOAD = "/policy/reload"
ROUTE_KVCONFIG = "/kvconfig"
ROUTE_BLOCK = "/block"
ROUTE_REQUEST = "/request"
ROUTE_TRACE = "/trace"
ROUTE_GUARD = "/guard"
ROUTE_TIER = "/tier"
ROUTE_LOOKUPS = "/lookups"
# The bounded raw-KV-byte dump. It can ARM a diagnostic that reads real KV bytes,
# so it is a dev-state route: token required in every shape, loopback included.
ROUTE_BLOCKDUMP = "/blockdump"

# The same routes as one tuple, in the order a reader wants them, so `status`
# and the tests enumerate the surface from ONE place. Suffixes: the router
# carries ROUTE_PREFIX itself.
ROUTES = (ROUTE_STATUS, ROUTE_SNAPSHOT, ROUTE_KVCONFIG, ROUTE_BLOCK, ROUTE_REQUEST,
          ROUTE_TRACE, ROUTE_GUARD, ROUTE_TIER, ROUTE_LOOKUPS, ROUTE_BLOCKDUMP,
          ROUTE_RELOAD)

# The routes that can change engine dev state. Token in every shape, loopback
# included - see endpoint.require_dev_state_token.
DEV_STATE_ROUTES = (ROUTE_TRACE, ROUTE_GUARD, ROUTE_TIER, ROUTE_LOOKUPS,
                    ROUTE_BLOCKDUMP)

# Name used for logs, for `VLLM_PLUGINS` allowlisting and as the entry-point
# name in both groups. One name allowlists both halves at once.
PLUGIN_NAME = "kvtier_dev"

DEFAULT_TIMEOUT_SECONDS = 5.0
MIN_TIMEOUT_SECONDS = 0.05
MAX_TIMEOUT_SECONDS = 60.0

DEFAULT_DIAG_INTERVAL_STEPS = 50
MIN_DIAG_INTERVAL_STEPS = 1
MAX_DIAG_INTERVAL_STEPS = 100_000
DEFAULT_DIAG_MIN_INTERVAL_SECONDS = 1.0
MIN_DIAG_MIN_INTERVAL_SECONDS = 0.0
MAX_DIAG_MIN_INTERVAL_SECONDS = 3600.0
MAX_DIAG_EPOCH_CHARS = 32
MAX_DIAG_RANK_CHARS = 32

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


def boot_inert() -> bool:
    """True = the INERT boot (patch only: no guard, no tier arming).

    The same falsy set as `enabled()`, so `0`/`false`/`off`/`no`/empty all mean
    "not inert" = the auto boot. Read per call, never at import: the mode an
    operator wants changes with a restart of the engine, not a re-install, and
    `status` reports the mode it read at the last boot.
    """
    return os.environ.get(ENV_BOOT_INERT, "").strip().lower() not in _FALSY


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


def diagnostics_dir() -> Path | None:
    """Where the per-rank worker files go. None = the writer must not run.

    Read per write, not at import, so turning it on is an env change in the
    engine's environment (which means a restart) and turning it off takes effect
    on the next write without one.
    """
    raw = os.environ.get(ENV_DIAG_DIR, "").strip()
    if not raw or "\x00" in raw:
        return None
    return Path(raw)


def diagnostics_epoch() -> tuple[str, str]:
    """(epoch, scope). `scope` is "env" or "process", and it ships in the file.

    A per-process epoch is honest but incomparable: the reader would (correctly)
    refuse to compare facts across ranks. An operator who wants the cross-rank
    verdicts sets VLLM_E2_TIER_DIAG_EPOCH once per boot, and every writer uses it
    verbatim (bounded, whitespace-stripped, control characters refused).
    """
    raw = os.environ.get(ENV_DIAG_EPOCH, "").strip()
    if raw and len(raw) <= MAX_DIAG_EPOCH_CHARS and not any(
            ord(char) < 32 for char in raw):
        return raw, "env"
    return snapshot_epoch(), "process"


def snapshot_epoch() -> str:
    """The plugin's per-process epoch. Imported lazily: `snapshot` is the read
    side and this module must stay importable on its own."""
    from . import snapshot  # noqa: PLC0415 - config is imported by snapshot
    return snapshot.epoch()


def worker_write_interval_steps() -> int:
    """The step cadence, clamped. Garbage falls back to the default."""
    raw = os.environ.get(ENV_DIAG_INTERVAL, "").strip()
    if not raw:
        return DEFAULT_DIAG_INTERVAL_STEPS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_DIAG_INTERVAL_STEPS
    return max(MIN_DIAG_INTERVAL_STEPS, min(MAX_DIAG_INTERVAL_STEPS, value))


def worker_min_interval_seconds() -> float:
    """The rate floor, clamped - the same policy as the interval above.

    Out of range is CLAMPED (a negative floor means "no floor", which is a legal
    choice; an absurd one becomes the cap), while unparseable falls back to the
    default. A typo must not silently remove a budget.
    """
    raw = os.environ.get(ENV_DIAG_MIN_INTERVAL, "").strip()
    if not raw:
        return DEFAULT_DIAG_MIN_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_DIAG_MIN_INTERVAL_SECONDS
    if value != value:                      # NaN: not comparable, so not usable
        return DEFAULT_DIAG_MIN_INTERVAL_SECONDS
    return max(MIN_DIAG_MIN_INTERVAL_SECONDS,
               min(MAX_DIAG_MIN_INTERVAL_SECONDS, value))


def diagnostics_rank() -> str | None:
    """An operator-pinned rank id, or None. Shape is validated by the writer."""
    raw = os.environ.get(ENV_DIAG_RANK, "").strip()
    if not raw or len(raw) > MAX_DIAG_RANK_CHARS:
        return None
    return raw


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
