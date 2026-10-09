"""Which CODE THIS PROCESS LOADED - as opposed to which code is INSTALLED.

WHY THIS MODULE IS A MODULE (2026-10-09). An operator spent an evening unable to
answer "which plugin version is running?", because the instrument they trusted
printed the answer of `importlib.metadata.version(...)`. That is a DIFFERENT
FACT: it reads the *installed distribution metadata on disk*, i.e. the version
the NEXT process to import this package will get. A long-running engine that
booted before an upgrade keeps running the code it imported at boot, and no
amount of reading disk will ever show it. Every reader in this package therefore
publishes `kvtier_dev_plugin.__version__` - the one value that IS the code
executing right now - and nothing else. Three consequences, each enforced by a
test in `tests/test_loaded_version.py`:

  * the published value must FOLLOW `__version__` at runtime (a test monkeypatches
    it and asserts the payloads move), so no payload can be carrying a copied
    literal;
  * it must be read live off the package module object, which is why this reads
    `sys.modules[__package__]` instead of `from . import __version__` - the
    latter binds a snapshot at import time and would freeze the old value across
    a monkeypatch exactly like a literal would;
  * a process that cannot know the answer says so in `plugin_version_reason`
    instead of publishing a plausible default.

WHAT THE THREE SURFACES PUBLISH, and whose process each one describes. They are
three DIFFERENT processes and a reader must never read one as another:

    GET  /plugins/kvtier/status    built in the API-server process (no engine call)
    GET  /plugins/kvtier/snapshot  built on the EngineCore loop thread; the API
                                   server only forwards the bytes it was handed
    rank-PP<pp>_TP<tp>.json        built in a WORKER process, which answers no route

Before this release the running version was only INFERABLE from version-gated
keys in the rank files (`disk_gpu_shadow.pairs_cross_execution` exists from 0.2.48 -
that section is a worker's contribution and the snapshot does not carry it - and
`writer.heartbeat` from 0.2.47) - and that fallback stays the ONLY way to date an
engine booted on an older build, so it is documented in README, not retired here.

Stdlib `sys` only, so it is legal on the EngineCore loop thread and on a worker's
step path (see `tests/test_snapshot.py::test_no_heavy_imports_in_the_read_side`).
"""

from __future__ import annotations

import sys

# THE FIELD NAMES. One name for one meaning, in all three surfaces, so a reader
# greps for one string. `version` already means SCHEMA version in the snapshot
# document (`snapshot.PROTOCOL_VERSION`) and in the rank file
# (`worker_diag.SCHEMA_VERSION`) and keeps meaning that: these keys never touch it.
PLUGIN_VERSION_FIELD = "plugin_version"
PLUGIN_VERSION_REASON_FIELD = "plugin_version_reason"
PLUGIN_VERSION_SCOPE_FIELD = "plugin_version_scope"

# The key set, in publication order. Every surface carries all three in EVERY
# state, so "the key is missing" is never a reader's question.
VERSION_FIELDS = (PLUGIN_VERSION_FIELD, PLUGIN_VERSION_REASON_FIELD,
                  PLUGIN_VERSION_SCOPE_FIELD)

# Where the value comes from, stated in the payload itself. `importlib.metadata`
# is named so the next reader does not reach for it.
SOURCE = "kvtier_dev_plugin.__version__ (the code this process imported, not " \
         "importlib.metadata and not pyproject.toml)"

# A version that needs more than this is not a version. Refused, never clipped:
# a clipped version string is a value that points at no build that ever existed.
MAX_VERSION_CHARS = 64

_UNKNOWN_SCOPE = "the process that built this document"


def _package_module():
    """This package's module object, or None if it is not in this process."""
    return sys.modules.get(__package__ or "kvtier_dev_plugin")


def _read() -> tuple:
    """(version, reason): exactly one of the two is ever non-None."""
    package = _package_module()
    if package is None:
        return None, ("this process has no kvtier_dev_plugin package object in "
                      "sys.modules, so nothing was imported to read")
    raw = getattr(package, "__version__", None)
    if raw is None:
        return None, "the package object carries no __version__ attribute"
    if not isinstance(raw, str):
        return None, f"__version__ is {type(raw).__name__}, not a string"
    text = raw.strip()
    if not text:
        return None, "__version__ is blank"
    if len(text) > MAX_VERSION_CHARS:
        return None, (f"__version__ is {len(text)} characters, over the "
                      f"{MAX_VERSION_CHARS}-character cap, so it is not a "
                      f"version string")
    return text, None


def loaded_version():
    """THIS process's loaded version string, or None when it is unknowable."""
    return _read()[0]


def loaded_version_reason() -> str:
    """Why `loaded_version()` is None; the empty string when it is not."""
    return _read()[1] or ""


def fields(scope=None) -> dict:
    """The three version keys. Same key set in every state. Never raises.

    `scope` names WHICH PROCESS'S loaded version this is, in the payload, because
    the three surfaces live in three different processes and the whole defect this
    module closes is a reader substituting one for another. It is a label written
    by the publishing code, never a guess about the runtime.
    """
    version = None
    reason = None
    try:
        version, reason = _read()
    except Exception as exc:  # noqa: BLE001 - a diagnostic is never a crash
        version, reason = None, (f"reading __version__ raised "
                                 f"{type(exc).__name__}")
    label = scope if isinstance(scope, str) and scope.strip() else _UNKNOWN_SCOPE
    return {
        PLUGIN_VERSION_FIELD: version,
        PLUGIN_VERSION_REASON_FIELD: reason if version is None else None,
        PLUGIN_VERSION_SCOPE_FIELD: f"{SOURCE}; reported by {label[:160]}",
    }
