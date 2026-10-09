"""Generate tests/fixtures_mamba.py from the fork, fingerprint-verified.

Same job as `gen_pr56810_fixture.py`, for the SECOND runtime-patch target:
`MambaManager.reachable_block_mask`. The plugin's own suite is stdlib-only and
cannot import vLLM, so the suite rebuilds a stand-in `MambaManager` from the
fork's REAL method text; this generator is what makes that text trustworthy.

WHY THIS ONE IS EXTRACTED FROM THE FILE RATHER THAN FROM `inspect.getsource()`
------------------------------------------------------------------------------
`gen_pr56810_fixture.py` runs under the vllm venv and hashes
`inspect.getsource(fn)` directly. That needs an importable vllm, and this fixture
was captured from a checked-out TREE (the live-tree reconstruction at
`/home/user/Source/experiment/vllm-live-repro`, branch `live`) rather than from a
running interpreter. The two extractions are the SAME BYTES, and that is not an
assumption: `test_mamba_anchor_patch.py::
test_the_extraction_method_reproduces_the_inspect_fingerprints` re-runs this
file's extraction over `fixtures_pr56810.py`'s four bodies - text that WAS
captured with real `inspect.getsource()` by the venv generator and whose hashes
`pr56810_sources.py` ships - and asserts every one of them hashes to the shipped
fingerprint. Decorators included, comments above a `def` excluded, exactly as
`inspect.getsource` does.

Usage:
    uv run python tools/gen_mamba_fixture.py [git-ref]   # ref default: `live`
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys

REPRO = "/home/user/Source/experiment/vllm-live-repro"
FORK_PATH = "vllm/v1/core/single_type_kv_cache_manager.py"
CLASS = "MambaManager"
METHOD = "reachable_block_mask"
OUT = "/home/user/Source/experiment/kvtier-dev-plugin/tests/fixtures_mamba.py"
REBUILT_FILE = "<fixtures_mamba_fork>"


def fork_text(ref: str) -> str:
    """The file as one blob, straight out of the repro's git object store."""
    out = subprocess.run(["git", "-C", REPRO, "show", f"{ref}:{FORK_PATH}"],
                         capture_output=True, text=True, check=True)
    return out.stdout


def method_source(src: str, cls_name: str, method: str, ref: str) -> str:
    """The method's own text, decorator lines included, exactly as inspect yields.

    `inspect.getsourcelines` walks BACK over the decorator lines and stops there,
    so the span is `min(decorator lineno) .. end_lineno`. A comment sitting above
    the decorator is NOT part of the function's source and stays out.
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for fn in node.body:
                if (isinstance(fn, ast.FunctionDef)
                        and fn.name == method):
                    start = fn.lineno - 1
                    if fn.decorator_list:
                        start = min(d.lineno for d in fn.decorator_list) - 1
                    return "".join(lines[start:fn.end_lineno])
    raise SystemExit(f"{cls_name}.{method} not found in {ref}:{FORK_PATH}")


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "live"
    src = fork_text(ref)
    md5 = hashlib.md5(src.encode()).hexdigest()
    sha = hashlib.sha256(src.encode()).hexdigest()
    body = method_source(src, CLASS, METHOD, ref)
    method_sha = hashlib.sha256(body.encode()).hexdigest()
    decorators = [ast.unparse(d) for d in
                  [n for n in ast.parse("class C:\n" + body).body[0].body
                   if isinstance(n, ast.FunctionDef)][0].decorator_list]

    parts = [
        '"""Fork-authoritative `MambaManager.reachable_block_mask` text '
        '(generated, do not edit).\n\n'
        f'Captured by tools/gen_mamba_fixture.py from the live-tree '
        f'reconstruction\n`{REPRO}`, branch `{ref}` (= upstream v0.30.0 + the '
        '58-patch\nseries + our four kvtier hunks, byte-identical to what the '
        'engine imports).\nThat file\'s md5 is\n'
        f'    {md5}\n'
        'which is ALSO its md5 on branch `built` and in '
        '`/home/user/Source/experiment/vllm-dev`, so the\nfingerprint below is '
        'not a property of one checkout. The method is\n'
        f'    sha256 {method_sha}\n'
        'decorators included, exactly as `inspect.getsource()` yields it '
        '(validated\nagainst the #56810 fingerprints by '
        '`test_the_extraction_method_reproduces_the_\ninspect_fingerprints`).\n\n'
        'This is the UNPATCHED text: `build_fork_mamba_class()` rebuilds a '
        'stand-in\n`MambaManager` so the plugin suite can drive the REAL '
        'install path - the\nfingerprints match, so `mamba_anchor.install()` '
        'accepts it - without importing\nvLLM.\n"""\n\n'
        'from __future__ import annotations\n\nimport linecache\n\n',
        f'FORK_REF = {ref!r}\nFORK_FILE = {FORK_PATH!r}\nFORK_FILE_MD5 = '
        f'{md5!r}\nFORK_FILE_SHA256 = {sha!r}\nCLASS_NAME = {CLASS!r}\n'
        f'METHOD_NAME = {METHOD!r}\nMETHOD_DECORATORS = {tuple(decorators)!r}\n'
        f'METHOD_SHA256 = {method_sha!r}\n\n',
        f'_ORIG_REACHABLE_BLOCK_MASK = {body!r}\n\n',
        'ORIGINAL_SOURCES = {\n    "reachable_block_mask": '
        '_ORIG_REACHABLE_BLOCK_MASK,\n}\n\n',
        f'REBUILT_SOURCE_FILE = {REBUILT_FILE!r}\n\n\n',
        'def class_text() -> str:\n'
        '    """The stand-in class source, assembled from the fork\'s own text."""'
        '\n    return ("class MambaManager(SingleTypeKVCacheManagerStub):\\n"\n'
        '            + "".join(ORIGINAL_SOURCES[name] if name.endswith("\\n")\n'
        '                    else ORIGINAL_SOURCES[name] + "\\n"\n'
        '                    for name in (METHOD_NAME,)))\n\n\n',
        'def build_fork_mamba_class(ns=None):\n'
        '    """Rebuild a stand-in `MambaManager` from the fork\'s own text.\n\n'
        '    `ns` supplies the module-level names the method closes over '
        '(`MambaSpec`,\n'
        '    `Sequence`, `KVCacheSpec`). Like `fixtures_pr56810`, the source is '
        'registered\n'
        '    in `linecache` under `REBUILT_SOURCE_FILE` so `inspect.getsource()` '
        '-\n'
        '    which is what the plugin\'s fingerprint check calls - works on the '
        'rebuilt\n'
        '    method. Returns (cls, module_namespace).\n'
        '    """\n'
        '    src = class_text()\n'
        '    linecache.cache[REBUILT_SOURCE_FILE] = (\n'
        '        len(src), None, src.splitlines(True), REBUILT_SOURCE_FILE)\n'
        '    module = {"SingleTypeKVCacheManagerStub": object}\n'
        '    module.update(ns or {})\n'
        '    module["__name__"] = "fixtures_mamba_fork"\n'
        '    module["__file__"] = REBUILT_SOURCE_FILE\n'
        '    exec(compile(src, REBUILT_SOURCE_FILE, "exec"), module, module)\n'
        '    cls = module["MambaManager"]\n'
        '    cls.__module__ = "fixtures_mamba_fork"\n'
        '    return cls, module\n',
    ]
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    print(f"  ref={ref} file md5={md5} sha256={sha[:12]}")
    print(f"  {CLASS}.{METHOD} decorators={decorators} sha256={method_sha}")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
