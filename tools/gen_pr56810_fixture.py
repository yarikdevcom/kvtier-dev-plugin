"""Generate tests/fixtures_pr56810.py from the fork, fingerprint-verified.

Run under the vllm venv. The plugin's own suite must stay stdlib-only, so it
cannot import the offload manager; instead it rebuilds a stand-in
`SimpleCPUOffloadScheduler` from the fork's REAL method text. This generator is
what makes that text trustworthy: it asserts each captured source hashes to the
fingerprint `pr56810_sources.py` ships, so a fixture that drifted from the fork
cannot be written at all.
"""

from __future__ import annotations

import hashlib
import inspect
import sys

sys.path.insert(0, "/home/user/Source/experiment/kvtier-dev-plugin")

import vllm.v1.simple_kv_offload.manager as mgr  # noqa: E402

from kvtier_dev_plugin import pr56810_sources  # noqa: E402
from kvtier_dev_plugin.backport import METHODS, _underlying  # noqa: E402

CLS = mgr.SimpleCPUOffloadScheduler
OUT = "/home/user/Source/experiment/kvtier-dev-plugin/tests/fixtures_pr56810.py"
REBUILT_FILE = "<fixtures_pr56810_fork>"

# --- capture the fork's ORIGINAL (unpatched) method text, verified ----------
captured = {}
for name in METHODS:
    fn, is_static = _underlying(CLS, name)
    src = inspect.getsource(fn)
    sha = hashlib.sha256(src.encode()).hexdigest()
    expected = pr56810_sources.SOURCES[name]["orig_sha256"]
    assert sha == expected, (
        f"{name}: captured {sha} != shipped fingerprint {expected}; the fork "
        f"moved and pr56810_sources.py is stale")
    captured[name] = src
    print(f"  verified {name:34s} static={is_static} {sha[:12]}")

# --- emit the fixture -------------------------------------------------------
parts = [
    '"""Fork-authoritative ORIGINAL method sources (generated, do not edit).\n'
    "\n"
    'Captured by tools/gen_pr56810_fixture.py from the fork at HEAD 10b8e07\n'
    "(byte-identical to upstream tag v0.30.0) and asserted, method by method,\n"
    "against the fingerprints in kvtier_dev_plugin/pr56810_sources.py. This is\n"
    "the UNPATCHED text: `build_fork_class()` rebuilds a stand-in\n"
    "`SimpleCPUOffloadScheduler` so the plugin suite can drive the REAL install\n"
    "path (fingerprints match, so install() accepts it) without importing vLLM.\n"
    '"""\n\nfrom __future__ import annotations\n\nimport linecache\n\n',
]

var_of = lambda n: "_ORIG_" + n.strip("_").upper()  # noqa: E731

for name in METHODS:
    parts.append(f"{var_of(name)} = {captured[name]!r}\n\n")

parts.append("ORIGINAL_SOURCES = {\n")
for name in METHODS:
    parts.append(f'    "{name}": {var_of(name)},\n')
parts.append("}\n\n")
parts.append(f"REBUILT_SOURCE_FILE = {REBUILT_FILE!r}\n")

parts.append(r'''

def _indent(src: str) -> str:
    """Method sources arrive already indented to class-body depth (4 spaces)."""
    return src if src.endswith("\n") else src + "\n"


def class_text() -> str:
    """The stand-in class source, assembled from the fork's own text."""
    return "class SimpleCPUOffloadScheduler:\n" + "".join(
        _indent(ORIGINAL_SOURCES[name]) for name in (
            "__init__", "_estimate_lazy_target_blocks",
            "update_state_after_alloc", "_select_eager_blocks_to_store"))


def build_fork_class(ns=None):
    """Rebuild a stand-in `SimpleCPUOffloadScheduler` from the fork's own text.

    `ns` supplies the module-level names the originals close over
    (`resolve_block_hashes`, `_StoreAdmission`, `cdiv`, `logger`, ...). A name an
    original references but `ns` omits raises NameError at CALL time, never at
    exec time, so a test only provides what the method it drives actually uses.

    Returns (cls, module_namespace). The source is registered in `linecache`
    under `REBUILT_SOURCE_FILE` so `inspect.getsource()` - which is what the
    plugin's fingerprint check calls - works on the rebuilt methods.
    """
    src = class_text()
    # linecache is how inspect finds source for exec'd code; without this the
    # fingerprint guard would see "no source" and refuse, for the wrong reason.
    linecache.cache[REBUILT_SOURCE_FILE] = (
        len(src), None, src.splitlines(True), REBUILT_SOURCE_FILE)
    module = {}
    module.update(ns or {})
    module["__name__"] = "fixtures_pr56810_fork"
    module["__file__"] = REBUILT_SOURCE_FILE
    exec(compile(src, REBUILT_SOURCE_FILE, "exec"), module, module)
    cls = module["SimpleCPUOffloadScheduler"]
    cls.__module__ = "fixtures_pr56810_fork"
    return cls, module
''')

text = "".join(parts)
with open(OUT, "w") as handle:
    handle.write(text)
print(f"wrote {OUT} ({len(text)} bytes)")

# --- prove the fixture reproduces the fork's text byte-for-byte -------------
ns: dict = {}
exec(compile(text, OUT, "exec"), ns)
rebuilt, _mod = ns["build_fork_class"]({})
for name in METHODS:
    fn, _ = _underlying(rebuilt, name)
    got = inspect.getsource(fn)
    assert got == captured[name], f"{name}: fixture does not reproduce the fork"
    assert hashlib.sha256(got.encode()).hexdigest() == \
        pr56810_sources.SOURCES[name]["orig_sha256"], f"{name}: hash drifted"
print("round-trip verified: rebuilt class hashes to the fork's fingerprints")
