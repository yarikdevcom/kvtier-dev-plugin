"""Runtime backport of upstream PR #56810 into `SimpleCPUOffloadScheduler`.

Stdlib only. vLLM is reached through `sys.modules` exactly like
`store_hook.manager_class()` / `engine_side._engine_core_class()`: the class is
ALWAYS looked up, never imported. `general_plugins` loads in worker and
arg-parsing processes too, and importing the offload manager from here would be
a heavy side effect this plugin does not own.

THE BUG (upstream issue #56396, fixed by PR #56810, merged 2026-09-20, NOT in
the v0.30.0 tag this fork is built on)
--------------------------------------------------------------------------
`SimpleCPUOffloadScheduler` assumed every KV cache group's block size is a
multiple of the hash block size. Core does NOT guarantee that for every group:
`resolve_kv_cache_block_sizes` (`kv_cache_utils.py:768-781`) validates
divisibility only over the PREFIX-CACHEABLE groups. A hybrid model with a
non-prefix-cacheable scratch group - our QSA raw-key ring is a
`CircularBufferSpec` of block size 8 beside five `block_size=3200` groups, and
GLM-5.3-Flash's `KpoolTailSpec` tail is the same shape - reaches
`resolve_block_hashes(request.block_hashes, 3200, 8)` and dies on
`assert block_size % hash_block_size == 0` (`kv_cache_utils.py:2888`):

  manager.py:842 `_select_eager_blocks_to_store` -> `resolve_block_hashes(...)`

The ring group reaches that line because the loop's only filter is
`if not group_manager.has_positionally_stable_blocks: continue`
(`manager.py:835`): `MambaManager` returns False there (align mode), but
`CircularBufferManager` does NOT override it, so the ring is not filtered.

WHY THE RING'S BLOCK SIZE IS 8, AND WHY `--prefix-match-unit` IS A DEAD END
---------------------------------------------------------------------------
The 8 is not a tuning knob. It is computed: `qsa_cache.py:900-901` builds the
ring's capacity as `compress_ratio * cdiv(compress_ratio + num_speculative_tokens,
compress_ratio)` and passes it as `CircularBufferSpec(block_size=capacity, ...)`
(`qsa_cache.py:906-907`). With the served `compress_ratio=4` and
`num_speculative_tokens=4` that is `4 * cdiv(4+4, 4) = 8`. The ring's block size
is therefore pinned by the draft depth, not by anything we may choose.
`CircularBufferSpec.prefix_cacheable` is a hard-coded False
(`kv_cache_interface.py:895-897`), which is exactly what makes the group
ineligible - and `CircularBufferManager` inherits
`has_positionally_stable_blocks = True` (it does not override it, unlike
`MambaManager`), which is what makes it reach the assert anyway.

The tempting "just make the sizes divisible" alternative - setting
`--prefix-match-unit` to 4 or 8 so `hash_block_size` divides 3200 and the
assert passes - is a DEAD END, for two independent reasons, and must not be
re-proposed as a fix:

1. A ring block of 8 tokens is NOT HASHABLE at that granularity: the ring's
   content is positionally unstable (a circular buffer reuses slots across
   steps), so hashing it in 8-token units would create prefix-cache entries
   that can be looked up and hit against stale ring content. The group is
   `prefix_cacheable=False` because its content must never be matched; the
   assert is telling us something true.
2. It would multiply the hash work. Prefix hashing every 8 tokens instead of
   every 3200 is ~400x more `BlockHash` objects per request, per group, per
   step - on the request path, for a group whose entries can never be hit.

The correct fix is the one upstream chose: keep the ring out of the
store/load/hash path entirely.

THE FIX (four changes, all in `SimpleCPUOffloadScheduler`)
----------------------------------------------------------
1. `__init__`              record `self.prefix_cacheable_group_ids`.
2. `_estimate_lazy_target_blocks`  iterate `prefix_cacheable_groups`.
3. `update_state_after_alloc`  non-cacheable group -> `cpu_hit_blocks.append([])`.
4. `_select_eager_blocks_to_store`  non-cacheable group -> `continue`, placed
   BEFORE the `len(gpu_block_ids) >= num_free` break so a scratch group can
   never consume CPU blocks or emit `BlockStored`.

WHAT IS REPLACED vs WRAPPED (and why)
-------------------------------------
* 2, 3 and 4 are REPLACED with a byte-for-byte copy of the upstream patched
  function (`pr56810_sources.py`), exec'd in the ORIGINAL function's
  `__globals__` so every free name (`resolve_block_hashes`, `_StoreAdmission`,
  `cdiv`, ...) resolves exactly as the fork's own does. A wrapper CANNOT inject
  a `continue` into the middle of a loop it cannot see, and the task forbids
  improvising an equivalent: the whole point is semantic identity with upstream.
* 1 is WRAPPED, not copied: the upstream change is purely ADDITIVE (one
  attribute assignment) and it is provably identical to setting that attribute
  immediately after the original `__init__` returns, because no line of the
  ORIGINAL body reads `self.prefix_cacheable_group_ids` (the attribute does not
  exist before this patch). Replacing a 119-line constructor to add one line is
  the larger risk, so we take the smaller one. The fingerprint still guards the
  original `__init__`: if a future fork edits it, we refuse, because a changed
  constructor could start reading the attribute mid-body and break the proof.

SOURCE FINGERPRINTS ARE MANDATORY
---------------------------------
For every method touched we recorded at authoring time the SHA-256 of the live
`inspect.getsource()` at fork HEAD 10b8e07. At install time a mismatch means
the function changed underneath us: we REFUSE to patch it, log once, and report
`patched: False` with the reason in `status`/`snapshot`. A stale fingerprint
degrades to "not patched" - never to a best-effort guess. This is the only
thing between us and silently patching a function whose body has moved.

THE GEOMETRY GUARD (never skip a prefix-cacheable group)
--------------------------------------------------------
`manager.py:456-458` silently skips any group with no parked blocks, and the
CPU lookup returns `[]` for a scratch group
(`kv_cache_coordinator.py:979-981`). So a *blanket* "skip whatever trips the
assert" policy is a silent output-corruption machine: drop a genuinely
prefix-CACHEABLE group and its KV is simply missing on resume, with no error.
The filter must therefore be EXACTLY upstream's `g not in
prefix_cacheable_group_ids`. `geometry_guard()` recomputes the set of groups
the patch would skip from a live `KVCacheConfig` and, if any of them reports
`prefix_cacheable=True`, REFUSES to install and names the group.

THE KILL SWITCH (runtime, no restart)
-------------------------------------
`set_armed(False)` makes the tier a strict no-op through three manager
methods, mirroring the connector's own `scheduler_manager is None` code path
(`simple_cpu_offload_connector.py:250,262,296`):

  get_num_new_matched_tokens -> (0, False)        (no prefix match)
  build_connector_meta       -> SimpleCPUOffloadMetadata()  (empty store+load)
  has_pending_stores         -> False

An empty metadata is what makes the WORKER a no-op: `worker.start_load_kv`
copies only `metadata.load_cpu_blocks` and `worker.wait_for_save` only
`metadata.store_gpu_blocks` (`worker.py:266,289`), both empty. The worker lives
in another process the HTTP call never reaches, so the no-op is delivered as
data (empty metadata), not as a second patch. `has_pending_block_frees` is
already always False on this connector (it does not override the base
implementation, `base.py:625`). See `disarm_limits()` for what this cannot undo.

THE ALL-OR-NOTHING INSTALL CONTRACT (never armed unless the fix is complete)
----------------------------------------------------------------------------
The invariant this module exists to guarantee: THE TIER IS NEVER ARMED UNLESS
THE FIX IS PROVABLY, FULLY IN PLACE. It closes the fail-open where the store
selector (the method that CONTAINS the crash) is refused by its fingerprint
while the other methods install: an "any method patched" install would report
`installed: true` and boot armed-and-incomplete, which is exactly the
live-traffic crash this backport is supposed to make impossible.

* "Fully installed" means EVERY method in `METHODS` carries `patched: true`
  (a fresh install whose fingerprint matched, or a re-install verified over our
  own marker). One refusal makes `installed: false`. The per-method verdicts
  stay visible so the operator still sees WHICH one was refused.
* A not-fully-installed install force-DISARMS the tier (`set_armed(False)`) and
  records `disarm_reason: "incomplete_install"`. The kill-switch wrappers install
  independently of the fingerprints, so a disarmed connector is a strict no-op
  and never reaches the crashing selector even unpatched.
* `set_armed(True)` is REFUSED - state unchanged, never raising - while either
  the geometry guard has vetoed the live config (`"geometry"`) or the fix is not
  fully installed (`"incomplete_install"`). The refusal names the refused
  methods and is published in `status`/`snapshot` (`installed`,
  `refused_methods`, `auto_disarmed`, `disarm_reason`, `arm_block`) and in the
  `/tier` verb response, so a dashboard operator reads "not armed, because X"
  rather than a silent no-op. Disarm is never refused: it is the safe direction.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import sys
import textwrap

from . import pr56810_sources, snapshot

logger = logging.getLogger("kvtier_dev_plugin.backport")

TARGET_MODULE = "vllm.v1.simple_kv_offload.manager"
METADATA_MODULE = "vllm.v1.simple_kv_offload.metadata"
TARGET_CLASS = "SimpleCPUOffloadScheduler"

# Ownership markers, the same idiom as store_hook / cpu_capacity, with their
# OWN distinct tags so a store-veto wrapper and a backport replacement can
# never be mistaken for one another.
OWNER_ATTR = "__kvtier_dev_backport_owner__"
OWNER_TAG = "kvtier-dev-plugin/backport-1"
ORIGINAL_ATTR = "__kvtier_dev_backport_original__"
SWITCH_ATTR = "__kvtier_dev_switch_owner__"
SWITCH_TAG = "kvtier-dev-plugin/switch-1"

METHODS = ("__init__", "_estimate_lazy_target_blocks",
           "update_state_after_alloc", "_select_eager_blocks_to_store")
# __init__ is wrapped; the other three are replaced by the upstream copy.
REPLACED = ("_estimate_lazy_target_blocks", "update_state_after_alloc",
            "_select_eager_blocks_to_store")
# Kill-switch wrappers (independent of the backport; gated on the arm flag).
SWITCH_METHODS = ("get_num_new_matched_tokens", "build_connector_meta",
                  "has_pending_stores")

_state = {
    "installs": 0,
    "reinstalls": 0,
    "skips": 0,
    "skip_reason": None,
    "class": None,
    "armed": True,                       # tier is live until someone disarms it
    "disarm_calls": 0,
    "arm_calls": 0,
    "fully_installed": False,            # every method in METHODS patched/verified
    "refused_methods": [],               # METHODS not patched on the last attempt
    "auto_disarmed": False,              # last refusal force-disarmed the tier
    "disarm_reason": None,               # "geometry" | "incomplete_install" | None
    "disarm_detail": None,               # the text behind the disarm
    "arm_refusals": 0,                   # set_armed(True) refused, count
    "last_arm_refusal": None,            # last refused-arm reason
    "method_verdicts": {},               # name -> {"patched","reason","sha_ok"}
    "switch_verdicts": {},               # name -> {"wrapped","reason"}
    "geometry": None,                    # last guard result
    "fallbacks": 0,
    "last_error": None,
}


def reset_state_for_tests() -> None:
    """Process-wide counters, like `snapshot._state` and `store_hook._state`."""
    _state.update({
        "installs": 0, "reinstalls": 0, "skips": 0, "skip_reason": None,
        "class": None, "armed": True, "disarm_calls": 0, "arm_calls": 0,
        "fully_installed": False, "refused_methods": [], "auto_disarmed": False,
        "disarm_reason": None, "disarm_detail": None, "arm_refusals": 0,
        "last_arm_refusal": None,
        "method_verdicts": {}, "switch_verdicts": {}, "geometry": None,
        "fallbacks": 0, "last_error": None,
    })


# --- resolving the fork's names without importing it ------------------------

def manager_class():
    """`SimpleCPUOffloadScheduler` if this process already imported it."""
    module = sys.modules.get(TARGET_MODULE)
    if module is None:
        return None
    return getattr(module, TARGET_CLASS, None)


def _empty_metadata():
    """A fresh empty `SimpleCPUOffloadMetadata`, or None if unresolvable.

    Looked up through `sys.modules` (never imported). The manager module itself
    imports this class at its top, so if we have a manager we can always find
    it; if we cannot, the disarmed `build_connector_meta` reports it rather
    than inventing a payload shape.
    """
    module = sys.modules.get(METADATA_MODULE)
    return getattr(module, "SimpleCPUOffloadMetadata", None) if module else None


# --- fingerprinting ---------------------------------------------------------

def _underlying(cls, name):
    """The function behind a class attribute, unwrapping staticmethod.

    Returns (function_or_None, is_staticmethod). `getattr(cls, name)` on a
    staticmethod already yields the plain function in py3.9+; we read the raw
    class-dict entry so a name only reachable via a base class is reported as
    "not defined on THIS class" rather than silently re-bound onto us.
    """
    raw = cls.__dict__.get(name, None)
    if raw is None:
        raw = getattr(cls, name, None)
    if isinstance(raw, staticmethod):
        return raw.__func__, True
    return raw, False


def fingerprint_of(fn) -> str | None:
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    except Exception as exc:  # noqa: BLE001 - no source is a refusal, not a crash
        _state["fallbacks"] += 1
        snapshot._error(exc, "backport fingerprint")
        return None


def _fingerprint_verdict(name, fn) -> tuple[bool, str | None, str | None]:
    """(source_matches, live_sha, expected_sha). Refuse on any mismatch."""
    expected = pr56810_sources.SOURCES[name]["orig_sha256"]
    live = fingerprint_of(fn)
    if live is None:
        return False, None, expected
    return live == expected, live, expected


# --- the geometry guard -----------------------------------------------------

def geometry_guard(kv_cache_config) -> dict:
    """Refuse to skip a group that is actually prefix-cacheable.

    The patch skips exactly `set(range(num_groups)) - prefix_cacheable_group_ids`.
    That is safe only because every one of those groups reports
    `prefix_cacheable=False`. If a config EVER names a group
    non-prefix-cacheable while its spec still says `prefix_cacheable=True`, the
    same `continue` would silently drop cacheable KV - the corruption the
    blanket-skip trap would walk straight into. Refuse, and name the group.
    """
    out = {"checked": False, "ok": False, "skipped": [], "cacheable_skipped": [],
           "reason": None}
    try:
        if kv_cache_config is None:
            out["reason"] = "no kv_cache_config available to check"
            return out
        groups = list(getattr(kv_cache_config, "kv_cache_groups", ()))
        ids = tuple(getattr(kv_cache_config, "prefix_cacheable_group_ids", ()))
        if not groups:
            out["reason"] = "kv_cache_config has no groups"
            return out
        out["checked"] = True
        out["skipped"] = [g for g in range(len(groups)) if g not in set(ids)]
        for g in out["skipped"]:
            spec = getattr(groups[g], "kv_cache_spec", None)
            if bool(getattr(spec, "prefix_cacheable", False)):
                out["cacheable_skipped"].append(
                    {"group": g, "block_size": getattr(spec, "block_size", None),
                     "layers": list(getattr(groups[g], "layer_names", ()) or ())[:4]})
        if out["cacheable_skipped"]:
            out["reason"] = (
                "refusing to install: the patch would skip group(s) that still "
                f"report prefix_cacheable=True: {out['cacheable_skipped']}")
            return out
        out["ok"] = True
        out["reason"] = None
        return out
    except Exception as exc:  # noqa: BLE001 - a config we cannot read is a refusal
        _state["fallbacks"] += 1
        snapshot._error(exc, "backport geometry guard")
        out["reason"] = f"geometry guard raised {type(exc).__name__}: {exc}"[:200]
        out["ok"] = False
        return out


# --- the __init__ wrapper (additive attribute) ------------------------------

def _make_init_wrapper(original):
    def __init__(self, *args, **kwargs):  # noqa: N807 - it IS __init__
        original(self, *args, **kwargs)
        # Upstream's one added statement, applied after the original body.
        # Provably identical: nothing in the ORIGINAL __init__ reads this
        # attribute (it does not exist until this patch), so setting it after
        # the constructor returns cannot change any earlier decision.
        object.__setattr__(
            self, "prefix_cacheable_group_ids",
            self.cpu_kv_cache_config.prefix_cacheable_group_ids)

    setattr(__init__, OWNER_ATTR, OWNER_TAG)
    setattr(__init__, ORIGINAL_ATTR, original)
    __init__.__doc__ = (
        "kvtier-dev-plugin: the fork's own __init__, then "
        "self.prefix_cacheable_group_ids = "
        "cpu_kv_cache_config.prefix_cacheable_group_ids (PR #56810).")
    return __init__


def _install_copy(cls, name) -> dict:
    """exec the byte-identical upstream body into the ORIGINAL's globals.

    Executing in `original.__globals__` is what makes the copy a faithful one:
    every free name it references (`resolve_block_hashes`, `_StoreAdmission`,
    `cdiv`, `maybe_convert_block_hash`, ...) resolves to exactly the object the
    fork's own function resolves to, in this process, right now.
    """
    report = {"patched": False, "reason": None, "sha_ok": None,
              "live_sha256": None, "expected_sha256": None}
    entry = pr56810_sources.SOURCES[name]
    original = cls.__dict__.get(name)
    shadowed = original is None
    fn, _ = _underlying(cls, name)
    if fn is None:
        report["reason"] = f"{name} does not exist: the fork renamed it"
        return report
    ok, live, expected = _fingerprint_verdict(name, fn)
    report.update(sha_ok=ok, live_sha256=live, expected_sha256=expected)
    if not ok:
        report["reason"] = (
            "source fingerprint mismatch: the fork changed this function since "
            f"10b8e07 (live {str(live)[:12]} != expected {str(expected)[:12]}), "
            "refusing to patch a body we can no longer vouch for")
        return report
    if shadowed:
        # The fingerprint matched an INHERITED function. Patching the subclass
        # with our own copy is still correct (same source, same globals), but
        # it is a shadow, not a replacement, so say so in the reason field.
        report["reason"] = f"installed as a shadow of an inherited {name}"
    body = _compile_copy(name, entry["source"], fn)
    if body is None:
        report["reason"] = "compiled copy is missing its own fingerprint anchor"
        return report
    setattr(cls, name, body)
    report["patched"] = True
    return report


def _function_of(body):
    """The plain function under a class attribute (unwrap `staticmethod`)."""
    return body.__func__ if isinstance(body, staticmethod) else body


def _compile_copy(name, source, original):
    """Build the patched method from its stored source, byte-faithfully.

    `inspect.getsource()` returns the method WITH its four-space class-body
    indent, so it must be dedented before `compile()`; the dedent is the only
    edit, and `test_no_edit_beyond_dedent` proves removing it re-adds the
    original indent exactly. The body is exec'd in the ORIGINAL function's
    `__globals__`, so every free name it looks up
    (`resolve_block_hashes`, `_StoreAdmission`, `cdiv`, ...) is the very object
    the fork's own function resolves to. Identity of the produced function is
    re-checked against the stored fingerprint before anything is installed, so
    an environment where `compile`/`exec` yields a different body cannot ship a
    patch we did not verify.

    `__qualname__` and `__module__` are restored from the original so the
    replaced method reports the same names the fork's did (the store-hook
    ownership checks and every log line read them).
    """
    expected = pr56810_sources.SOURCES[name]["orig_sha256"]
    live = (hashlib.sha256(inspect.getsource(_function_of(original)).encode())
            .hexdigest() if _has_source(original) else None)
    if live is not None and live != expected:
        return None
    namespace: dict = {}
    code = compile(textwrap.dedent(source),
                   f"<kvtier backport PR#56810 {name}>", "exec")
    exec(code, getattr(_function_of(original), "__globals__", {}), namespace)  # noqa: S102
    body = namespace.get(name)
    if body is None:
        return None
    fn = _function_of(body)
    try:
        fn.__qualname__ = _function_of(original).__qualname__
        fn.__module__ = getattr(_function_of(original), "__module__", fn.__module__)
    except Exception as exc:  # noqa: BLE001 - cosmetic metadata, never a blocker
        snapshot._error(exc, "backport qualname")
    _tag(body, OWNER_ATTR, OWNER_TAG, original)
    return body


def _tag(body, attr, value, original):
    """Mark a patched callable, on the function AND on a wrapper descriptor.

    `setattr` on a `staticmethod` object stores the attribute on the wrapper and
    does NOT make it visible on `__func__` (verified on 3.13). Our own readers
    use `cls.__dict__[name]`, which would find it either way, but a reader that
    unwraps first (as the tests and any future audit do) would not. Tagging both
    costs nothing and removes a way to miss that a method is ours.
    """
    inner = _function_of(body)
    for target in ({id(inner): inner, id(body): body}).values():
        try:
            setattr(target, attr, value)
            if attr == OWNER_ATTR and original is not None:
                setattr(target, ORIGINAL_ATTR, original)
        except Exception:  # noqa: BLE001, S110 - a descriptor that refuses attrs
            pass           # is still installed; the marker is belt-and-braces


def _has_source(fn) -> bool:
    try:
        inspect.getsource(fn)
    except Exception:  # noqa: BLE001, S110 - no source -> nothing to re-check
        return False


# --- the kill-switch wrappers -----------------------------------------------

def _make_matched_wrapper(original):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if not _state["armed"]:
            return 0, False
        return original(self, request, num_computed_tokens)

    setattr(get_num_new_matched_tokens, SWITCH_ATTR, SWITCH_TAG)
    setattr(get_num_new_matched_tokens, ORIGINAL_ATTR, original)
    get_num_new_matched_tokens.__doc__ = (
        "kvtier-dev-plugin kill switch: (0, False) while the tier is disarmed, "
        "else the engine's own prefix-match answer.")
    return get_num_new_matched_tokens


def _make_meta_wrapper(original):
    def build_connector_meta(self, scheduler_output):
        if not _state["armed"]:
            cls = _empty_metadata()
            if cls is not None:
                return cls()
            # Cannot build the empty payload (metadata module vanished): fall
            # back to the engine's own behaviour rather than returning None and
            # corrupting the step. Counted so the snapshot is not silently calm.
            _state["fallbacks"] += 1
        return original(self, scheduler_output)

    setattr(build_connector_meta, SWITCH_ATTR, SWITCH_TAG)
    setattr(build_connector_meta, ORIGINAL_ATTR, original)
    build_connector_meta.__doc__ = (
        "kvtier-dev-plugin kill switch: an empty SimpleCPUOffloadMetadata while "
        "the tier is disarmed, so no store/load is issued and the worker does "
        "nothing; else the engine's own metadata.")
    return build_connector_meta


def _make_pending_wrapper(original):
    def has_pending_stores(self):
        if not _state["armed"]:
            return False
        return original(self)

    setattr(has_pending_stores, SWITCH_ATTR, SWITCH_TAG)
    setattr(has_pending_stores, ORIGINAL_ATTR, original)
    has_pending_stores.__doc__ = (
        "kvtier-dev-plugin kill switch: False while the tier is disarmed "
        "(connector.has_pending_push_work delegates here).")
    return has_pending_stores


_SWITCH_FACTORIES = {
    "get_num_new_matched_tokens": _make_matched_wrapper,
    "build_connector_meta": _make_meta_wrapper,
    "has_pending_stores": _make_pending_wrapper,
}


def _install_switch(cls, name) -> dict:
    """Wrap one scheduler-facing method with the arm check. Idempotent.

    No fingerprint: the wrapper only ADDS a gated early-return around the
    engine's own function (it never reimplements it), so a changed body cannot
    be silently mispatched the way a copied loop can. Ownership discipline and
    the original-binding are identical to store_hook.
    """
    report = {"wrapped": False, "reason": None}
    existing = cls.__dict__.get(name)
    if existing is None:
        existing = getattr(cls, name, None)
    if existing is None:
        report["reason"] = f"{name} does not exist: the fork renamed it"
        return report
    if not callable(existing):
        report["reason"] = f"{name} is not callable ({type(existing).__name__})"
        return report
    owner = getattr(existing, SWITCH_ATTR, None)
    if owner is not None and owner != SWITCH_TAG:
        held = getattr(existing, "__qualname__", type(existing).__name__)
        report["reason"] = (f"{name} carries another switch marker ({held!r})")
        return report
    if owner == SWITCH_TAG:
        # Re-bind to the FORK's function, so a re-install cannot nest wrappers.
        existing = getattr(existing, ORIGINAL_ATTR, existing)
    setattr(cls, name, _SWITCH_FACTORIES[name](existing))
    report["wrapped"] = True
    return report


# --- installation -----------------------------------------------------------

def install(cls, kv_cache_config=None) -> dict:
    """Backport PR #56810 onto `cls`, then fit the kill-switch wrappers.

    Never raises. Idempotent and marker-guarded like store_hook. A stale
    fingerprint is a per-method refusal, so one moved function cannot make the
    others silently wrong: the verdict for each is reported independently.
    """
    report = {"installed": False, "class": None, "reason": None,
              "methods": {}, "switch": {}, "geometry": None}
    try:
        if cls is None:
            return _refuse(report, f"no {TARGET_CLASS} is imported in this "
                                   f"process ({TARGET_MODULE} absent)")
        report["class"] = getattr(cls, "__name__", type(cls).__name__)

        guard = geometry_guard(
            kv_cache_config if kv_cache_config is not None
            else _config_of(cls))
        report["geometry"] = guard
        _state["geometry"] = guard
        if kv_cache_config is not None and not guard.get("ok"):
            # The patch would skip a group whose KV is needed on resume: turn
            # the tier OFF rather than leave it armed to drop cacheable KV. The
            # methods are not attempted at all here, so nothing is refused per
            # method - the whole install is refused on geometry.
            _state["fully_installed"] = False
            _state["refused_methods"] = []
            report["refused_methods"] = []
            _force_disarm("geometry", guard.get("reason"))
            report["auto_disarmed"] = True
            report["disarm_reason"] = "geometry"
            return _refuse(report, guard.get("reason"))

        for name in METHODS:
            verdict = _install_one(cls, name)
            report["methods"][name] = verdict
            _state["method_verdicts"][name] = verdict

        for name in SWITCH_METHODS:
            verdict = _install_switch(cls, name)
            report["switch"][name] = verdict
            _state["switch_verdicts"][name] = verdict

        # ALL-OR-NOTHING: `installed` means EVERY method in METHODS is patched
        # (a fingerprint that matched, or a re-install verified over our own
        # marker). One refusal is a failed install, not a partial success - so
        # the method that happens to contain the crash can never be the one left
        # unpatched behind an `installed: true`. The per-method verdicts stay in
        # the report so the operator still sees WHICH one was refused.
        refused = [name for name in METHODS
                   if not report["methods"][name].get("patched")]
        report["refused_methods"] = refused
        fully_installed = not refused
        report["installed"] = fully_installed
        _state["fully_installed"] = fully_installed
        _state["refused_methods"] = refused

        if fully_installed:
            if _state["installs"] == 0:
                _state["installs"] += 1
                logger.warning(
                    "kvtier-dev-plugin: PR #56810 backport INSTALLED on %s "
                    "(all %d methods; geometry ok=%s)", report["class"],
                    len(METHODS), guard.get("ok"))
            else:
                _state["reinstalls"] += 1
            _state["skip_reason"] = None
            _state["auto_disarmed"] = False
            _state["disarm_reason"] = None
            _state["disarm_detail"] = None
            report["auto_disarmed"] = False
            report["disarm_reason"] = None
        else:
            # A partial install is the fail-open this contract exists to kill.
            # Without the full fix the store path asserts, so leave the tier OFF
            # rather than armed-and-incomplete; the kill-switch wrappers (which
            # install independently of the fingerprints) make the disarmed
            # connector a strict no-op, so it never reaches the crashing
            # selector. Distinct `disarm_reason` from the geometry guard above.
            reason = _first_refusal(report) or "all required methods refused"
            report["reason"] = reason
            report["auto_disarmed"] = True
            report["disarm_reason"] = "incomplete_install"
            _force_disarm("incomplete_install", reason)
            return _refuse(report, reason)
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        _state["fallbacks"] += 1
        _state["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
        snapshot._error(exc, "backport install")
        report["installed"] = False
        report["reason"] = f"install raised: {_state['last_error']}"
        return report


def _install_one(cls, name) -> dict:
    existing = cls.__dict__.get(name)
    holder_attr = OWNER_ATTR
    marked = existing is not None and getattr(existing, holder_attr, None) == OWNER_TAG
    if marked:
        # Re-install over our own copy/wrapper: rebind the fork's original, so
        # a reload loop cannot nest copies or double-apply the __init__ set.
        original = getattr(existing, ORIGINAL_ATTR, None)
        if original is None:
            return {"patched": True, "reason": "already installed (no original)"}
        if name == "__init__":
            setattr(cls, name, _make_init_wrapper(original))
            return {"patched": True, "reason": "re-installed over our own wrapper"}
        body = _compile_copy(name, pr56810_sources.SOURCES[name]["source"], original)
        if body is None:
            return {"patched": False,
                    "reason": "re-install refused: the bound original no longer "
                              "matches its own fingerprint"}
        setattr(cls, name, body)
        return {"patched": True, "reason": "re-installed over our own copy"}

    existing = cls.__dict__.get(name)
    if existing is not None:
        foreign = getattr(existing, OWNER_ATTR, None)
        if foreign is not None and foreign != OWNER_TAG:
            held = getattr(existing, "__qualname__", type(existing).__name__)
            return {"patched": False,
                    "reason": f"carries another plugin's backport marker ({held!r})"}

    if name == "__init__":
        fn, _ = _underlying(cls, name)
        if fn is None:
            return {"patched": False,
                    "reason": "__init__ does not exist: the fork renamed it"}
        ok, live, expected = _fingerprint_verdict(name, fn)
        if not ok:
            return {"patched": False, "sha_ok": False, "live_sha256": live,
                    "expected_sha256": expected,
                    "reason": "source fingerprint mismatch on __init__; the "
                              "constructor changed, so wrapping it is no longer "
                              "provably safe"}
        setattr(cls, name, _make_init_wrapper(fn))
        return {"patched": True, "sha_ok": True, "reason": "wrapped (additive)"}
    return _install_copy(cls, name)


def _config_of(_cls):
    """A `kv_cache_config` for the guard, if we can name one without a manager.

    The class itself carries no config; only live managers do. So at
    class-install time there is nothing to guard against and the geometry guard
    reports `checked: False` - the guard is enforced for real in
    `ensure_installed`, which holds a live manager.
    """
    return


def _first_refusal(report):
    for name, v in report["methods"].items():
        if not v.get("patched") and v.get("reason"):
            return f"{name}: {v['reason']}"
    return None


def _refuse(report, reason):
    """One log line per distinct refusal episode (the snapshot polls this)."""
    if _state["skip_reason"] == reason:
        logger.debug("kvtier-dev-plugin: backport still not installed: %s", reason)
    else:
        _state["skips"] += 1
        _state["skip_reason"] = reason
        logger.warning("kvtier-dev-plugin: PR #56810 backport NOT installed: %s",
                       reason)
    report["reason"] = reason
    return report


def already_installed(cls) -> bool:
    """True when every method we own on `cls` still carries our marker.

    A cheap read-only check (seven `getattr`s, no hashing, no `compile`) that
    lets the polling path skip the install entirely. It is deliberately NOT a
    memo of "we installed once": if another plugin, or a module reload, replaced
    one of our methods, the marker disappears and the next call does the work
    again. That is the point - it is a claim about the class as it is NOW.
    """
    if cls is None:
        return False
    for name in METHODS:
        raw = cls.__dict__.get(name)
        if raw is None or getattr(_function_of(raw), OWNER_ATTR, None) != OWNER_TAG:
            return False
    for name in SWITCH_METHODS:
        raw = cls.__dict__.get(name)
        if raw is None or getattr(_function_of(raw), SWITCH_ATTR, None) != SWITCH_TAG:
            return False
    return True


def ensure_installed(engine) -> dict:
    """Install from the LIVE manager's class and guard against its geometry.

    The connector imports the manager long after `register()` runs, so this is
    the path that actually lands the backport - and the only one that can see
    the real `KVCacheConfig`, which is what the geometry guard needs. Never
    raises.

    The cheap exit matters more here than it looks: the snapshot verb runs on
    the EngineCore loop thread and the dashboard polls it every few seconds, and
    a full install re-compiles three upstream functions. Once the class carries
    our markers, `already_installed` keeps that work off the serving loop.
    """
    try:
        manager = manager_of(engine)
    except Exception as exc:  # noqa: BLE001 - a manager we cannot reach is data
        _state["fallbacks"] += 1
        snapshot._error(exc, "backport manager_of")
        manager = None
    config = getattr(manager, "kv_cache_config", None) if manager is not None else None
    cls = type(manager) if manager is not None else manager_class()
    if already_installed(cls):
        # Still re-run the geometry guard: a re-initialised engine can hand us a
        # DIFFERENT config on the same class, and that is exactly the case the
        # guard exists for. It reads a handful of attributes, so it is cheap.
        guard = geometry_guard(config)
        _state["geometry"] = guard
        if config is not None and not guard.get("ok"):
            report = {"installed": False, "class": getattr(cls, "__name__", None),
                      "reason": None, "methods": {}, "switch": {},
                      "geometry": guard}
            out = _refuse(report, guard.get("reason"))
            # The patch is already live on this class and the config has turned
            # out to be one it must not be live for: turn the tier OFF rather
            # than let a live patch skip a group whose KV is needed on resume.
            # `disarm_reason: "geometry"` distinguishes this from an incomplete
            # install; the methods ARE installed, so arming is blocked by the
            # geometry gate in `_arm_block_reason`, not by `fully_installed`.
            _force_disarm("geometry", guard.get("reason"))
            out["auto_disarmed"] = True
            out["disarm_reason"] = "geometry"
            out["refused_methods"] = []
            return out
        # Markers verified on every method: the fix is fully in place on this
        # class. Publish that so `installed` is honest on the polling path too.
        _state["fully_installed"] = True
        _state["refused_methods"] = []
        return {"installed": True, "class": getattr(cls, "__name__", None),
                "reason": "already installed (markers verified, nothing to do)",
                "methods": dict(_state["method_verdicts"]),
                "switch": dict(_state["switch_verdicts"]), "geometry": guard,
                "refused_methods": [], "auto_disarmed": bool(_state["auto_disarmed"]),
                "disarm_reason": _state["disarm_reason"]}
    return install(cls, config)


def manager_of(engine):
    """The scheduler-side manager off a live engine, or None. Reads only."""
    scheduler = getattr(engine, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    return getattr(connector, "scheduler_manager", None)


# --- the kill switch --------------------------------------------------------

def _force_disarm(kind: str, detail) -> None:
    """Turn the tier OFF and remember WHY, so the snapshot names the reason.

    The shared tail of every "the fix is not safe to run" path: the geometry
    guard and an incomplete install both land here, and `kind` is what keeps the
    two refusal reasons distinguishable in the report ("geometry" vs
    "incomplete_install"). Disarm is the safe direction, so `set_armed(False)`
    can never be refused and this cannot fail.
    """
    set_armed(False)
    _state["auto_disarmed"] = True
    _state["disarm_reason"] = kind
    _state["disarm_detail"] = (str(detail) if detail else "")[:200]


def _arm_block_reason() -> dict | None:
    """Why arming the tier is unsafe right now, or None if arming is allowed.

    Two independent gates, checked in this order:

    * the GEOMETRY guard, when it has actually checked a live config and said no
      -- the patch would skip a prefix-cacheable group (silent output loss).
    * an INCOMPLETE install -- not every method in METHODS is patched, so the
      store path would still reach the upstream assert.

    A geometry guard that never checked (a class-install before any config
    exists) is NOT a block: booting patches the class first, and refusing to arm
    there would strand a healthy engine.
    """
    geometry = _state["geometry"]
    if geometry and geometry.get("checked") and not geometry.get("ok"):
        return {"kind": "geometry",
                "reason": "the geometry guard refuses this config, so arming "
                          "would let the patch skip a prefix-cacheable group: "
                          f"{geometry.get('reason')}"}
    if not _state["fully_installed"]:
        refused = list(_state["refused_methods"])
        names = ", ".join(refused) if refused else "(none attempted)"
        return {"kind": "incomplete_install",
                "reason": f"the PR #56810 fix is not fully installed; refused "
                          f"method(s): {names}. The tier stays disarmed until "
                          f"every required method is patched."}
    return None


def set_armed(armed: bool) -> dict:
    """Arm/disarm the tier at runtime. Contained; never raises.

    DISARM is unconditional -- it is the safe direction and must always work.
    ARM is GATED: it is refused (state unchanged, nothing raised) while the
    geometry guard has vetoed the live config or the fix is not fully installed,
    because the invariant is that the tier is never armed unless the divisibility
    fix is provably, fully in place. A refused arm says WHY and names the refused
    methods, and shows up in `status`/`snapshot` and the `/tier` verb so an
    operator reads "not armed, because X" instead of a silent no-op.
    """
    armed = bool(armed)
    if armed:
        block = _arm_block_reason()
        if block is not None:
            _state["arm_refusals"] += 1
            _state["last_arm_refusal"] = block["reason"][:200]
            logger.warning("kvtier-dev-plugin: REFUSED to arm the KV tier: %s",
                           block["reason"])
            return {"armed": bool(_state["armed"]), "changed": False,
                    "refused": True, "kind": block["kind"],
                    "reason": block["reason"],
                    "refused_methods": list(_state["refused_methods"]),
                    "switch_installed": bool(_state["switch_verdicts"])}
    changed = armed != _state["armed"]
    _state["armed"] = armed
    if armed:
        _state["arm_calls"] += 1
    else:
        _state["disarm_calls"] += 1
    if changed:
        logger.warning("kvtier-dev-plugin: KV tier %s",
                       "ARMED (tier live)" if armed else
                       "DISARMED (connector forced to a strict no-op)")
    return {"armed": armed, "changed": changed, "refused": False, "kind": None,
            "reason": None, "refused_methods": [],
            "switch_installed": bool(_state["switch_verdicts"])}


def is_armed() -> bool:
    return bool(_state["armed"])


def disarm_limits() -> dict:
    """What the kill switch canNOT cover. Honest, and shown in the snapshot.

    Everything here is a consequence of the switch operating on the SCHEDULER
    role through the manager, on the step boundary:

    * An in-flight copy already handed to the workers at the instant of disarm
      still runs to completion and is freed normally - the scheduler does not
      talk to workers mid-transfer (connector.py:330-332 says so of reset).
    * Blocks already parked in host RAM stay parked and stay reachable by
      prefix-cache lookups once RE-armed; disarm stops NEW stores/loads, it does
      not wipe the tier. `store_hook`'s veto is likewise a skip, not a deferral.
    * It cannot stop a store already inside `prepare_store_specs` for the
      CURRENT step: `build_connector_meta` is the gate and it is evaluated once
      per step. A misbehaving store in the step already in progress completes.
    * The worker process is never contacted: it is a separate process and the
      HTTP call lands on EngineCore. Worker-side no-op is delivered as empty
      metadata (worker.py:266,289), which is why an unresolvable metadata module
      makes the disarmed wrapper fall back to the engine instead of returning
      None.
    * `has_pending_block_frees` is not wrapped: this connector does not override
      it, so it is already `False` unconditionally (base.py:625).
    """
    return {
        "worker_process_patched": False,
        "worker_noop_via": "empty SimpleCPUOffloadMetadata (worker.py:266,289)",
        "in_flight_transfers_undone": False,
        "parked_blocks_freed": False,
        "has_pending_block_frees": "not overridden by this connector (base "
                                   "returns False)",
    }


# --- the engine-side verb ---------------------------------------------------

TIER_OPS = ("status", "arm", "disarm")


def tier_for_engine(engine, args=None) -> dict:
    """`{op: status|arm|disarm}` for `POST /plugins/kvtier/tier`. Never raises.

    `arm`/`disarm` flip the kill switch; `status` only reads. An unknown or
    missing `op` is a DECLINE with 400 semantics in the payload, never a flip:
    a mistyped body must not silently turn the tier off (or on) on the way to
    doing something else. The endpoint validates `op` against `TIER_OPS` first,
    so this is the second line of defence for a direct engine call.

    The switch is applied to THIS process's manager class (the caller is the
    EngineCore loop thread, which holds the live manager), so the verb is also
    what makes a freshly-adopted class switchable.
    """
    args = args if isinstance(args, dict) else {}
    op = str(args.get("op", "status") or "status")
    out: dict = {"op": op, "applied": False, "reason": None}
    try:
        report = ensure_installed(engine)
        if op not in TIER_OPS:
            out["reason"] = (f"unknown op {op!r}; expected one of "
                             f"{list(TIER_OPS)}. Nothing was changed.")
            out["status_code"] = 400
            out["tier"] = section(manager_of(engine))
            return out
        if op != "status":
            state = set_armed(op == "arm")
            out["state"] = state
            # `applied` is False when an arm was REFUSED: the operator asked and
            # nothing was allowed to change, which is a different answer from a
            # no-op re-arm and must not read as success.
            out["applied"] = not state.get("refused", False)
            if state.get("refused"):
                out["reason"] = state["reason"]
        # The install verdict comes from the authoritative _state so a refused
        # arm is explained by the same fields a snapshot shows.
        out["install"] = {"installed": bool(_state["fully_installed"]),
                          "reason": report.get("reason"),
                          "refused_methods": list(_state["refused_methods"]),
                          "auto_disarmed": bool(_state["auto_disarmed"]),
                          "disarm_reason": _state["disarm_reason"]}
        out["tier"] = section(manager_of(engine))
        out["status_code"] = 200
        return out
    except Exception as exc:  # noqa: BLE001 - never into the scheduler loop
        _state["fallbacks"] += 1
        snapshot._error(exc, "tier verb")
        out["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        out["status_code"] = 500
        return out


# --- observability ----------------------------------------------------------

def _fields(**overrides) -> dict:
    block = _arm_block_reason()
    fields = {
        "patched": bool(_state["installs"] or _state["reinstalls"]),
        "class": _state["class"],
        "module": TARGET_MODULE,
        "pr": "56810",
        "upstream_issue": "56396",
        "fork_head": "10b8e07",
        "upstream_tag": "v0.30.0",
        "installs": int(_state["installs"]),
        "reinstalls": int(_state["reinstalls"]),
        "skips": int(_state["skips"]),
        "skip_reason": _state["skip_reason"],
        "armed": bool(_state["armed"]),
        # All-or-nothing install state: `installed` is true ONLY when every
        # method in METHODS is patched; `refused_methods` names the ones that
        # are not, so a partial install is visible, not just a failed one.
        "installed": bool(_state["fully_installed"]),
        "refused_methods": list(_state["refused_methods"]),
        "auto_disarmed": bool(_state["auto_disarmed"]),
        "disarm_reason": _state["disarm_reason"],
        "disarm_detail": _state["disarm_detail"],
        "arm_refusals": int(_state["arm_refusals"]),
        "last_arm_refusal": _state["last_arm_refusal"],
        # Why an arm would be refused right now (None means arming is allowed).
        # This is the "not armed, because X" an operator reads on a dashboard.
        "arm_block": None if block is None else {"kind": block["kind"],
                                                 "reason": block["reason"]},
        "arm_calls": int(_state["arm_calls"]),
        "disarm_calls": int(_state["disarm_calls"]),
        "methods": {name: dict(v) for name, v in _state["method_verdicts"].items()},
        "replaced": list(REPLACED),
        "wrapped": ["__init__"],
        "switch_methods": list(SWITCH_METHODS),
        "switch": {name: dict(v) for name, v in _state["switch_verdicts"].items()},
        "geometry": _state["geometry"],
        "disarm_limits": disarm_limits(),
        "fallbacks": int(_state["fallbacks"]),
        "last_error": _state["last_error"],
        # Set only by the live-manager view.
        "live_manager_patched": None,
        "live_manager_armed_noop": None,
    }
    fields.update(overrides)
    return fields


def section(manager=None) -> dict:
    """The snapshot section. Key set identical on every path (`_fields`)."""
    if manager is None:
        return _fields()
    live_patched = None
    live_armed = None
    try:
        live_cls = type(manager)
        attr = getattr(live_cls, "_select_eager_blocks_to_store", None)
        live_patched = getattr(attr, OWNER_ATTR, None) == OWNER_TAG
        live_armed = bool(_state["armed"])
    except Exception as exc:  # noqa: BLE001 - a manager that lies is data
        snapshot._error(exc, "backport live view")
    return _fields(live_manager_patched=live_patched,
                   live_manager_armed_noop=live_armed)


def empty_section(reason: str) -> dict:
    """The degraded section. Same key set, by construction."""
    return _fields(error=str(reason)[:200])
