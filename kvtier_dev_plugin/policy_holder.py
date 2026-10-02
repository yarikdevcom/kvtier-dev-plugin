"""Holder for the live policy module: late binding plus atomic trial-load swap.

Two properties matter here and both come from a verified constraint:

1. LATE BINDING. Callers use `_POLICY.decide(view)`. `decide` is a *property on
   the holder*, so every call looks the function up on whatever module is live
   right now. A `from ...policy import decide` binding would freeze the old
   function object and a reload would appear to do nothing - the failure mode
   this whole design exists to avoid.

2. ROLLBACK IS STRUCTURAL, NOT A REPAIR. `importlib.reload(M)` mutates `M` in
   place, so a half-broken new version is visible to the world the moment the
   reload raises, and "rolling back" means re-executing something else into the
   same dict. Instead this holder builds a brand new module object from the
   source, and only swaps `_POLICY._module` if that succeeded. A failed reload
   leaves the live module untouched by construction; the traceback comes back
   as data.

Why compile-from-source instead of importlib: importlib's source loader trusts
`(mtime, size)` in `__pycache__`. An edit that keeps the byte length and lands
in the same mtime tick reloads the OLD code, which looks exactly like "my
reload did not work". `compile()` on the bytes we just read cannot be fooled
that way, and a ~100 us compile on the engine loop thread is the price of not
having that bug. The module never enters `sys.modules`, so nothing else in the
process can import a half-loaded copy of it.
"""

from __future__ import annotations

import builtins
import hashlib
import os
import time
import traceback
import types

MAX_TRACEBACK_CHARS = 4000


class PolicyLoadError(Exception):
    """The candidate policy file could not be turned into a usable module."""


def _source_stamp(path: str) -> dict:
    """Cheap identity of the file right now (for `GET .../status` proofs)."""
    stat = os.stat(path)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return {
        "path": path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "source_sha256_8": digest[:8],
    }


def _compile_policy(path: str, module_name: str) -> types.ModuleType:
    """Execute `path` into a fresh module object. Raises on any problem.

    Never touches `sys.modules`, never writes a `.pyc`, never consults one.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    module = types.ModuleType(module_name)
    module.__file__ = path
    # Same thing CPython injects for an exec whose globals lack __builtins__;
    # stated explicitly so the module's globals are fully known to the reader.
    module.__dict__["__builtins__"] = builtins
    code = compile(raw, path, "exec")
    exec(code, module.__dict__)  # noqa: S102 - our own file, path-validated
    decide = getattr(module, "decide", None)
    if not callable(decide):
        raise PolicyLoadError("policy module defines no callable decide()")
    version = getattr(module, "VERSION", None)
    if not isinstance(version, (int, str, type(None))):
        raise PolicyLoadError(f"policy VERSION must be int/str/None, got "
                              f"{type(version).__name__}")
    return module


class PolicyHolder:
    """Holds the live policy module and swaps it only on a fully good load."""

    def __init__(self, path: str, module_name: str = "kvtier_dev_policy") -> None:
        self._path = str(path)
        self._module_name = module_name
        self._module: types.ModuleType | None = None
        self._stamp: dict | None = None
        self._loads = 0
        self._failures = 0
        self._last_error: str | None = None

    # --- late binding -----------------------------------------------------

    @property
    def module(self) -> types.ModuleType | None:
        return self._module

    @property
    def decide(self):
        """The live `decide`, looked up at CALL time. None if not loaded."""
        module = self._module
        if module is None:
            return None
        return getattr(module, "decide", None)

    @property
    def decide_store(self):
        """The live `decide_store`, looked up at CALL time. None if not there.

        Same late-binding rule as `decide`, and the same reason: the store hook
        calls `holder.decide_store(...)` per candidate block, so a reload of
        `policy.py` changes what the tier does with no restart. A frozen
        `from ...policy import decide_store` binding would not.

        None is the NORMAL answer for a policy that only observes (and for a
        deployment whose policy predates this API - a live one does), and the
        store hook reads it as "no opinion": the engine keeps its own verdict.
        Deliberately NOT required by `_compile_policy`: requiring it would make
        every existing policy file fail to reload.
        """
        module = self._module
        if module is None:
            return None
        return getattr(module, "decide_store", None)

    # --- reporting --------------------------------------------------------

    def describe(self) -> dict:
        return {
            "loaded": self._module is not None,
            "loads": self._loads,
            "failures": self._failures,
            "last_error": self._last_error,
            "live": dict(self._stamp) if self._stamp else None,
        }

    # --- loading ----------------------------------------------------------

    def ensure_loaded(self) -> dict:
        """Load once. Contained: returns a report, never raises."""
        if self._module is not None:
            return {"ok": True, "loaded_from_cold": False, "error": None,
                    "traceback": None}
        return self._load_and_swap(cold=True)

    def reload(self) -> dict:
        """Re-read the file. Contained: returns a report, never raises.

        A failure leaves `self._module` exactly as it was, so the engine keeps
        serving with the last good policy and the caller gets the traceback.
        """
        return self._load_and_swap(cold=False)

    def _load_and_swap(self, *, cold: bool) -> dict:
        before = dict(self._stamp) if self._stamp else None
        started = time.perf_counter()
        try:
            stamp = _source_stamp(self._path)
            module = _compile_policy(stamp["path"], self._module_name)
        except Exception as exc:  # noqa: BLE001 - never into the scheduler
            self._failures += 1
            self._last_error = f"{type(exc).__name__}: {exc}"[:300]
            return {
                "ok": False,
                "loaded_from_cold": cold,
                "kept": before,
                "applied": None,
                "rolled_back": self._module is not None,
                "error": self._last_error,
                "traceback": _capped_traceback(),
                "compile_us": int((time.perf_counter() - started) * 1e6),
            }

        # Success: the swap is a single attribute assignment, and the module
        # was not reachable by anyone before this line.
        self._module = module
        self._stamp = stamp
        self._loads += 1
        self._last_error = None
        return {
            "ok": True,
            "loaded_from_cold": cold,
            "kept": before,
            "applied": dict(stamp),
            "rolled_back": False,
            "error": None,
            "traceback": None,
            "compile_us": int((time.perf_counter() - started) * 1e6),
        }


def _capped_traceback() -> str:
    text = traceback.format_exc()
    if len(text) > MAX_TRACEBACK_CHARS:
        return text[:MAX_TRACEBACK_CHARS] + "\n... truncated"
    return text
