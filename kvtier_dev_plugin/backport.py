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

THE BOOT DEFAULT IS DISARMED AT IMPORT; THE BOOT ITSELF ARMS ONCE IT VERIFIES
--------------------------------------------------------------------------------
At plugin IMPORT the tier does nothing (`BOOT_ARMED = False`). At CONNECTOR
CREATION - the first instant the manager class exists - the boot runs
`boot_from_connector()`: install PR #56810, arm GUARD MODE, and arm the TIER,
but ONLY on a verified install (every method patched, the live object safe, the
geometry guard passed, and the guard actually wrapped the methods that exist).
Anything less leaves the tier disarmed and says why in one BOOT log line.

Why the disarmed-at-import posture must stay: the arm gate is the ONLY thing
standing between a half-patched store path and the production crash, and at
plugin-load time nothing is patched yet - see the fail-open history below.
Why the boot now finishes the job instead of waiting for a human: a restart
nobody follows up on used to leave the box either doing nothing or (before the
connector hook existed) UNPATCHED in front of live traffic. Patched-but-inert is
still available with `VLLM_E2_TIER_DEV_BOOT_INERT=1`.

Why the boot default had to change:

* `vllm` calls `load_general_plugins()` at `core.py:217`, but the Scheduler - and
  with it the connector and the offload manager - is not constructed until
  `core.py:260`. So `register()`'s immediate `install(manager_class())` is
  ALWAYS refused at boot (`cls is None`), and that refusal path returns before
  any disarm - the state used to stay at its initial value, `armed: True`.
* Nothing on the scheduler or store path called `ensure_installed`: its only call
  sites were the two engine verbs (`engine_side.py:99`, `engine_side.py:131`) and
  the `/tier` verb itself (`backport.py:953`).

Booted armed, that combination meant the FIRST request reached the UNPATCHED
store selector and died on the assert - a fail-open whose only defence was an
operator remembering to curl a verb first. `BOOT_ARMED = False` removes the
reliance on remembering in the direction that mattered: a tier boot stopped being
live in front of traffic before it was patched. What it could not do was make the
boot WORK - every tier boot still needed a human to arm it - which is why the boot
now arms it itself at connector creation, on proof (`boot_from_connector`).

That fixes "inert", not "patched", so the second half closed the other gap: the
install is no longer left to verbs. `install_connector_hook()` (fitted by
`register()`) wraps `KVConnectorFactory.create_connector`, and when that returns
a tier connector carrying a `scheduler_manager` it runs the whole install against
the class the connector just imported, with the LIVE `kv_cache_config`. That is
the first moment the manager class exists and it is still inside
`Scheduler.__init__`, long before the first request - so an engine that boots
with the tier configured is patched by the boot itself, and geometry-checked by
the boot too (a class-only install cannot see a config; this one can).

The arm gate therefore seeks proof rather than trusting its caller:
`set_armed(True)` runs `_arm_proof()` (an install onto the class, when a class is
resolvable) before deciding, and `tier_for_engine` calls `ensure_installed`
(the live manager's class, WITH its `KVCacheConfig`, so the geometry guard
actually checks) before every op. A cold-boot sequence is:

  (import) tier disarmed, nothing patched
  (boot) connector creation -> install() -> wrappers fitted + the fix installed,
           guard mode armed, and the tier ARMED if - and only if - the install
           verified; one BOOT line states which of the three landed
  POST /tier {"op":"status"}  -> confirms it against the live manager
  POST /tier {"op":"arm"}     -> refuses unless installed AND the guard passed
  (a refusal names its kind: "geometry" | "no_manager" | "incomplete_install")

AUTO-ARMING IS STRICTER THAN A MANUAL ARM IN ONE RESPECT: it also requires a
LIVE manager (so the live-object check actually ran against the instance the
engine will call) and a guard that wrapped every entry point the class HAS.
`set_armed(True)` from the verb does not require those, because a human reading
the payload is the check. A method the fork simply does not define
(`take_events`) is reported under `boot.not_covered` and does NOT block the
auto-arm - refusing to arm for a method that cannot exist would mean the boot
never works on a fork that trimmed its own interface, which is the opposite of
the point.

WHAT THE KILL SWITCH GATES (and what it still cannot reach)
-----------------------------------------------------------
The kill switch is a set of wrappers this module fits; `armed` is only ever read
INSIDE them. Four manager methods, which together are the ONLY two routes to the
method that contains the assert (`_select_eager_blocks_to_store`, and inside it
`resolve_block_hashes` at `manager.py:842`):

1. ROUTE 1, per step: `get_num_new_matched_tokens`, `build_connector_meta`
   (`manager.py:488`) -> `prepare_store_specs` (`:552`) -> selector at `:745`.
2. ROUTE 2, per finished request: `_queue_finished_eager_store` (`:1169`), the
   only other caller of the selector (`:1177`), reached from
   `request_finished_all_groups` (`:1161`). It is gated too, and it had to be:
   the scheduler calls `update_state_after_alloc` for EVERY scheduled request
   (`scheduler.py:1228`) and that registers `_reqs_to_store` before its
   `num_external_tokens == 0` early return - so a disarmed tier reached the
   selector on every finished request, gated elsewhere or not. The gate is on the
   INNER method because the outer one returns the `(bool, dict|None)` that drives
   delayed freeing, while this one returns `None` either way.

There is no third caller: `_select_eager_blocks_to_store` has exactly two call
sites in the fork (`:745`, `:1177`) and `resolve_block_hashes` is called once
inside the manager (`:842`). Two honest limits remain, both in `disarm_limits()`:
the worker process is never patched (it builds no `scheduler_manager` and runs
none of these methods), and a tier connector nested inside another connector
(e.g. `MultiConnector`) is not the object `create_connector` returns, so its hook
reports `other_connector` instead of installing.

THE KILL SWITCH (runtime, no restart)
-------------------------------------
`set_armed(False)` makes the tier a strict no-op through four manager methods,
the first three mirroring the connector's own `scheduler_manager is None` code
path (`simple_cpu_offload_connector.py:250,262,296`):

  get_num_new_matched_tokens -> (0, False)        (no prefix match)
  build_connector_meta       -> SimpleCPUOffloadMetadata()  (empty store+load)
  has_pending_stores         -> False
  _queue_finished_eager_store-> None              (queue nothing on finish)

An empty metadata is what makes the WORKER a no-op: `worker.start_load_kv`
copies only `metadata.load_cpu_blocks` and `worker.wait_for_save` only
`metadata.store_gpu_blocks` (`worker.py:266,289`), both empty. The worker lives
in another process the HTTP call never reaches, so the no-op is delivered as
data (empty metadata), not as a second patch. `has_pending_block_frees` is
already always False on this connector (it does not override the base
implementation, `base.py:625`). See `disarm_limits()` for what this cannot undo.

THE LIVE OBJECT IS THE AUTHORITY (production CONSTRUCTS FIRST)
--------------------------------------------------------------
Both production incidents (2026-10-03) were ONE mistake: assuming a class-
first, patch-then-construct world, while production constructs first and our
layers arrive afterwards. `SimpleCPUOffloadScheduler` is built INSIDE
`create_connector`, before any hook fits the class, and no second instance is
ever constructed. Two consequences, now both closed:

* The additive `prefix_cacheable_group_ids` came ONLY from the `__init__`
  wrapper, which never runs for an already-built instance, while the patched
  methods reading it went live on the class: the only instance that mattered
  raised `AttributeError: 'SimpleCPUOffloadScheduler' object has no attribute
  'prefix_cacheable_group_ids'` (engine death 12:41, patched selector line 35).
  Fix: the install BACK-FILLS the attribute onto the live instance at install
  time from the same source upstream uses (`cpu_kv_cache_config.
  prefix_cacheable_group_ids`, the fork exposes it as a `cached_property` on
  `KVCacheConfig`, falling back to `kv_cache_config` and REPORTING which one
  was used), keeps the `__init__` wrapper for re-initialised engines, and the
  install verdict ASSERTS the live object is safe (`_live_object_check`):
  every attribute the fingerprint-verified bodies read off `self` must be
  present (and the new one an int-sequence). A live manager we cannot make
  safe is an install failure - `installed: false`, refused ARM with kind
  `incomplete_install`, force-disarmed - so patched code never runs against
  an object it would crash on. A class-only install (no manager visible)
  reports the check as unchecked; the connector-creation hook and every verb
  run it against the real manager.
* A re-install while guard mode is armed fingerprinted OUR OWN guard wrapper
  of `update_state_after_alloc` and reported "the fork changed this function
  since 10b8e07" - blaming the fork for our own layering, refusing the
  install and auto-disarming a healthy tier. Fix: fingerprinting DESCENDS our
  own layers (backport/switch/guard markers + ORIGINAL chains) and hashes the
  body underneath; a refusal now says the body UNDERNEATH our layers changed,
  and the fork-blaming wording is reserved for that real case. Re-installs
  rebuild the copy under the guard wrapper, preserving nesting, and
  `already_installed` sees through the layers so the polling path stays off
  the recompile. `patched` and `installed` in every payload now come from the
  SAME all-or-nothing verdict and can never disagree.

WHAT IT MEANS CONCRETELY: every attribute the patched bodies read off `self`
is set by the fork's own (fingerprint-verified) `__init__` - which already
RAN on a construct-first instance - EXCEPT the one this patch adds.
`_estimate_lazy_target_blocks` is a staticmethod: it reads nothing off `self`
(it reads `kv_cache_config.prefix_cacheable_group_ids` from its ARGUMENT, and
only ever runs inside `__init__`, so a construct-first instance cannot be
exposed to it). The live check therefore enumerates the `self` reads of the
three replaced bodies (`cpu_kv_cache_config`, `block_size`, `hash_block_size`,
`cpu_coordinator`, `cpu_block_pool`, `group_block_sizes`,
`enable_kv_cache_events`, `_lazy_mode`, `_reqs_to_store`, `_pending_cpu_hits`,
`_reqs_to_load`, `_gpu_block_pool`) plus the one added attribute.

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
  records `disarm_reason: "incomplete_install"`. The kill-switch wrappers are
  fitted FIRST, before the guard and before any upstream body is compiled, and
  they need no fingerprint - so a refused backport, a geometry veto or an
  install that raised midway all leave the SAME state: wrappers present +
  disarmed, which now gates BOTH routes to the selector. A refused backport must
  never leave the tier armed-and-unpatched, and never says "armed": it logs
  `backport NOT installed`, arms nothing, and `arm_block.kind` reads
  `incomplete_install`.
* `set_armed(True)` is REFUSED - state unchanged, never raising - while the
  geometry guard has vetoed the live config (`kind: "geometry"`), no
  `SimpleCPUOffloadScheduler` exists in this process to install onto or arm at
  all (`kind: "no_manager"`), or the fix is not fully installed
  (`kind: "incomplete_install"`). The refusal names the refused methods and is
  published in `status`/`snapshot` (`installed`, `refused_methods`,
  `auto_disarmed`, `disarm_reason`, `arm_block`) and in the `/tier` verb
  response, so a dashboard operator reads "not armed, because X" rather than a
  silent no-op. Disarm is never refused: it is the safe direction.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import sys
import textwrap

from . import config, mamba_anchor, pr56810_sources, route_policy, snapshot

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

# THE STAND-DOWN VERDICT. When the engine's OWN source tree already carries the
# upstream #56810 bodies (the tracked, apply/verify/rollback-able patch the
# operator installs with `config/box/kvtier-patch.sh`), the runtime backport
# installs NOTHING and says so with this one token. It exists because patching
# from outside a process we do not control has now cost three production
# incidents; nine lines in the engine's own file ends the class, and this is the
# half that keeps the two mechanisms from stacking on each other.
IN_TREE_FIXED = "already_fixed_in_tree"

METHODS = ("__init__", "_estimate_lazy_target_blocks",
           "update_state_after_alloc", "_select_eager_blocks_to_store")
# __init__ is wrapped; the other three are replaced by the upstream copy.
REPLACED = ("_estimate_lazy_target_blocks", "update_state_after_alloc",
            "_select_eager_blocks_to_store")
# Kill-switch wrappers (independent of the backport; gated on the arm flag).
# The first three are the connector's per-step entry points; the fourth is the
# FINISH path, the only other caller of the crashing selector, and the reason a
# disarmed tier is a gate rather than merely an intent. See "THE KILL SWITCH".
SWITCH_METHODS = ("get_num_new_matched_tokens", "build_connector_meta",
                  "has_pending_stores", "_queue_finished_eager_store")

# THE TIER IS DISARMED AT IMPORT. See "THE BOOT DEFAULT IS DISARMED AT IMPORT" at
# the top of this file: the install cannot land at plugin-load time (the manager
# class does not exist yet), so a boot-armed tier is a boot-UNPATCHED tier facing
# live traffic. The boot ARMS it later, at connector creation, and only on a
# verified install (`boot_from_connector`); this flag is the state BEFORE that
# moment, and `VLLM_E2_TIER_DEV_BOOT_INERT=1` makes it the state FOREVER.
BOOT_ARMED = False

# The two boot postures, named so the log line, the payload and the operator all
# use one vocabulary. `auto` = the boot finishes the job (install, guard, arm);
# `inert` = the pre-autoboot posture: install only, every effect by hand.
BOOT_MODE_AUTO = "auto"
BOOT_MODE_INERT = "inert"

# The BOOT line names at most this many refusal reasons / not-covered methods, so
# one line cannot be long enough to be unreadable; the FULL detail stays in the
# `/tier` and `/status` payloads (`boot.refusals`, `boot.not_covered`).
MAX_BOOT_ITEMS = 6
MAX_BOOT_ITEM_CHARS = 220

# The refusal kinds an arm can get, so `arm_block.kind` has a closed vocabulary.
ARM_BLOCK_KINDS = ("geometry", "no_manager", "incomplete_install")

_state = {
    "installs": 0,
    "reinstalls": 0,
    "skips": 0,
    "skip_reason": None,
    "class": None,
    "armed": BOOT_ARMED,            # inert at IMPORT; the boot arms it on proof
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
    "live_check": None,                  # last live-object verdict
    "fallbacks": 0,
    "last_error": None,
    # The connector-creation hook: the moment the manager class EXISTS, which is
    # the only reason a boot can be patched before its first request.
    "hook_fitted": False,
    "hook_reason": None,
    "hook_fits": 0,
    "hook_calls": 0,
    "hook_tier_connectors": 0,
    "hook_no_manager": 0,
    "hook_other_connector": 0,
    "hook_installs": 0,
    "hook_errors": 0,
    "hook_last": None,
    # THE BOOT VERDICT: what the connector-creation boot actually achieved, and
    # the one line that says so. None = no boot has run in this process yet (no
    # tier connector built), which is NOT the same as "the boot failed".
    "boot": None,
    "boots": 0,
    "boot_line": None,
    # THE STAND-DOWN: the last in-tree-fix classification, and how many times we
    # stood down. `in_tree_notice` is the log-once flag (the snapshot polls this
    # every few seconds, so the stand-down must not log per poll).
    "in_tree": None,
    "in_tree_standdowns": 0,
    "in_tree_notice": False,
}


def reset_state_for_tests() -> None:
    """Process-wide counters, like `snapshot._state` and `store_hook._state`."""
    _state.update({
        "installs": 0, "reinstalls": 0, "skips": 0, "skip_reason": None,
        "class": None, "armed": BOOT_ARMED, "disarm_calls": 0, "arm_calls": 0,
        "fully_installed": False, "refused_methods": [], "auto_disarmed": False,
        "disarm_reason": None, "disarm_detail": None, "arm_refusals": 0,
        "last_arm_refusal": None,
        "method_verdicts": {}, "switch_verdicts": {}, "geometry": None,
        "live_check": None,
        "fallbacks": 0, "last_error": None,
        "hook_fitted": False, "hook_reason": None, "hook_fits": 0,
        "hook_calls": 0, "hook_tier_connectors": 0, "hook_no_manager": 0,
        "hook_other_connector": 0, "hook_installs": 0, "hook_errors": 0,
        "hook_last": None, "boot": None, "boots": 0, "boot_line": None,
        "in_tree": None, "in_tree_standdowns": 0, "in_tree_notice": False,
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


# --- our own layers: the wrappers THIS package fits -------------------------
#
# `install()` composes layers (guard over switch over the fork's body), so a
# class attribute can be one of OUR wrappers by the time a re-install runs.
# Two rules follow, both from the 2026-10-03 construct-first incidents:
# * a fingerprint must be taken on the body UNDERNEATH our layers, never on
#   the layer itself (hashing guard_mode's `_wrapped` and then reporting "the
#   fork changed this function" sent an operator hunting a fork update that
#   does not exist);
# * `already_installed` must see our marker THROUGH the layers, or every poll
#   re-compiles three upstream functions on the serving loop.

def _layer_kinds():
    """(owner_attr, owner_tag, kind, original_attr) for every layer we own.

    The guard's and the restore layer's constants are read from their own
    modules (lazy: same package, no cycle - neither imports backport at import
    time), so the tag values cannot drift from what they actually stamp.

    THE RESTORE LAYER MUST BE LISTED HERE, and its absence was a real bug, not a
    cosmetic omission. A layer that `_layers_of()` does not know is reported as
    the BOTTOM - the fork's own body - so with `disk_restore` fitted on
    `get_num_new_matched_tokens` / `build_connector_meta`:
      * the in-tree stand-down fingerprinted OUR wrapper and reported the fork as
        drifted ("live <sha> is neither the v0.30.0 body nor the #56810 body"),
        which is exactly the misattribution this function's comment above warns
        about, and it auto-disarmed a healthy tier; and
      * `install()` saw `layers == []` and took `_install_copy`, which REPLACES
        the class attribute with backport's own copy - silently deleting the
        restore layer while `disk_restore` still believed it was fitted. An
        offer-less tier that reports `installed=YES` is the silent-failure shape
        this whole file exists to prevent.
    `_rewrap()` could already rebuild a restore layer and
    `disk_restore._install_sched_method` already refused to double-stack one;
    both branches were unreachable until this entry existed.
    """
    from . import (  # noqa: PLC0415 - sibling layers, lazy by idiom
        disk_restore,
        guard_mode,
    )
    return (
        (SWITCH_ATTR, SWITCH_TAG, "switch", ORIGINAL_ATTR),
        (OWNER_ATTR, OWNER_TAG, "backport", ORIGINAL_ATTR),
        (guard_mode.OWNER_ATTR, guard_mode.OWNER_TAG, "guard",
         guard_mode.ORIGINAL_ATTR),
        (disk_restore.OWNER_ATTR, disk_restore.OWNER_TAG,
         disk_restore.LAYER_KIND, disk_restore.ORIGINAL_ATTR),
    )


def _layers_of(fn):
    """Walk one class attribute down through our own layers.

    Returns `(layers, bottom)`: `layers` is `[(kind, body), ...]` outermost
    first with kind in {"switch", "backport", "guard", "restore"}; `bottom` is the first
    body NOT marked as ours - normally the fork's own function, or a foreign
    plugin's body (which is then just a body we must vouch for by
    fingerprint), or None when an ORIGINAL chain is broken. The walk is
    depth-capped: our factories cannot cycle, but a hand-damaged ORIGINAL
    attribute must yield `bottom: None` (a refusal), never a hang inside the
    step loop's thread.
    """
    kinds = _layer_kinds()
    layers: list = []
    current = _function_of(fn) if fn is not None else None
    while current is not None:
        if len(layers) > 16:
            return layers, None  # a cycle is a broken chain, not a bottom
        under = None
        hit = None
        for attr, tag, kind, orig_attr in kinds:
            if getattr(current, attr, None) == tag:
                under = getattr(current, orig_attr, None)
                hit = kind
                break
        if hit is None:
            return layers, current
        layers.append((hit, current))
        current = _function_of(under) if under is not None else None
    return layers, None


def _rewrap(name, base, layers_above):
    """Rebuild our wrapper layers (outermost-first list) on top of `base`.

    A `backport` layer is REPLACED by `base` (the freshly compiled copy IS
    that layer); a `switch` layer is rebuilt by its factory; a `guard` layer
    is rebuilt by `guard_mode._make_wrapper` (same package, the same factory
    its own `install()` composes with - so guard semantics, decline table and
    markers are unchanged under a re-install); a `restore` layer is rebuilt by
    `disk_restore.make_wrapper`, the same factory ITS install uses. The `or body`
    fallback is unreachable by construction (a restore layer exists only on the
    three scheduler methods `make_wrapper` has a hook for) and keeps the chain
    callable if that ever stops being true. Never touches any callable
    that is not ours.
    """
    from . import (
        disk_restore,  # noqa: PLC0415 - see _layer_kinds
        guard_mode,  # noqa: PLC0415 - see _layer_kinds
    )
    body = base
    for kind, _wrapper in reversed(layers_above):
        if kind == "backport":
            continue  # `body` already replaces the previous backport layer
        if kind == "switch":
            body = _SWITCH_FACTORIES[name](body)
        elif kind == "guard":
            body = guard_mode._make_wrapper(name, body)
        elif kind == disk_restore.LAYER_KIND:
            body = disk_restore.make_wrapper(name, body) or body
    return body


# --- the live-object check (attributes the patched bodies read) -------------

_MISSING = object()

# The `self` reads of the three fingerprint-verified replacement bodies,
# enumerated from the stored source (the `__init__` wrapper reads only
# `cpu_kv_cache_config`, already listed). Existence-checked on the live
# instance: on a construct-first instance the fork's own constructor RAN, so
# a missing one means a fork we no longer recognize - and the patched body
# would raise inside the step loop. `_gpu_block_pool` is included by NAME
# only: the fork legitimately keeps it None until bind_gpu_block_pool.
REQUIRED_LIVE_ATTRIBUTES = (
    "cpu_kv_cache_config", "block_size", "hash_block_size", "cpu_coordinator",
    "cpu_block_pool", "group_block_sizes", "enable_kv_cache_events",
    "_lazy_mode", "_reqs_to_store", "_pending_cpu_hits", "_reqs_to_load",
    "_gpu_block_pool",
)


def _is_int_group_ids(value) -> bool:
    """True for a tuple/list/set/frozenset of ints (empty allowed)."""
    if not isinstance(value, (tuple, list, set, frozenset)):
        return False
    return all(isinstance(item, int) and not isinstance(item, bool)
               for item in value)


def _live_object_check(manager) -> dict:
    """Assert the LIVE instance is safe for the patched bodies, and back-fill
    the one attribute this patch ADDS. Contained; never raises.

    Upstream's `__init__` addition sets `self.prefix_cacheable_group_ids =
    self.cpu_kv_cache_config.prefix_cacheable_group_ids`, but the `__init__`
    WRAPPER only fires for instances constructed AFTER install - production
    constructs first. So: back-fill from the same source (the fork exposes
    `prefix_cacheable_group_ids` as a `cached_property` on `KVCacheConfig`,
    which is what `cpu_kv_cache_config` is), fall back to `kv_cache_config`
    and say which was used; then require every other `self` read to be
    present. A manager we cannot make safe is a REFUSAL (the caller
    disarms), because the patched methods would raise AttributeError on the
    only instance that matters. `manager is None` is `checked: False` - a
    class-only install cannot see an instance and says so honestly.
    """
    out = {"checked": False, "ok": False, "manager": None, "source": None,
           "backfilled": False, "missing": [], "reason": None}
    try:
        if manager is None:
            out["reason"] = ("no live manager visible at install time: the "
                             "check ran against the class only, so an "
                             "instance constructed BEFORE the patch was "
                             "fitted was not verified")
            return out
        out["checked"] = True
        out["manager"] = type(manager).__name__
        pcids = getattr(manager, "prefix_cacheable_group_ids", None)
        if pcids is None:
            source = None
            cpu_cfg = getattr(manager, "cpu_kv_cache_config", None)
            gpu_cfg = getattr(manager, "kv_cache_config", None)
            for cfg, name in ((cpu_cfg, "cpu_kv_cache_config"),
                              (gpu_cfg, "kv_cache_config")):
                value = getattr(cfg, "prefix_cacheable_group_ids", None)
                if value is not None:
                    pcids = value
                    source = (f"{name}.prefix_cacheable_group_ids"
                              + ("" if name == "cpu_kv_cache_config" else
                                 " (fallback; cpu_kv_cache_config carries "
                                 "none)"))
                    break
            if pcids is None:
                out["reason"] = ("live manager has no prefix_cacheable_group_ids "
                                 "and no config can supply it (checked "
                                 "cpu_kv_cache_config then kv_cache_config): "
                                 "the patched methods would raise "
                                 "AttributeError on the only instance that "
                                 "matters")
                return out
            if not _is_int_group_ids(pcids):
                out["reason"] = (f"prefix_cacheable_group_ids from "
                                 f"{source} is not a sequence of ints "
                                 f"({type(pcids).__name__})")
                return out
            object.__setattr__(manager, "prefix_cacheable_group_ids",
                               tuple(pcids))
            out["backfilled"] = True
            out["source"] = f"back-filled from {source}"
        else:
            if not _is_int_group_ids(pcids):
                out["reason"] = (f"live prefix_cacheable_group_ids is not a "
                                 f"sequence of ints ({type(pcids).__name__})")
                return out
            out["source"] = "already present on the live instance"
        out["missing"] = [attr for attr in REQUIRED_LIVE_ATTRIBUTES
                          if getattr(manager, attr, _MISSING) is _MISSING]
        if out["missing"]:
            out["reason"] = ("live manager is missing attribute(s) the "
                             f"patched methods read: {out['missing']}")
            return out
        out["ok"] = True
        return out
    except Exception as exc:  # noqa: BLE001 - a manager we cannot read is a refusal
        _state["fallbacks"] += 1
        snapshot._error(exc, "backport live-object check")
        out["reason"] = f"live-object check raised {type(exc).__name__}: {exc}"[:200]
        out["ok"] = False
        return out


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


def _make_finished_store_wrapper(original):
    """Gate ROUTE 2 into the crashing selector: the connector's FINISH path.

    `request_finished_all_groups` (manager.py:1161) calls this and then returns
    `request_finished(...)`'s real `(bool, dict|None)`; THIS method returns
    `None` and is a pure queueing side effect (`_pending_finished_stores.append`
    + pool touches), so declining it is free and it is the correct place to
    gate. Gating the OUTER method instead would lie to the scheduler about
    delayed freeing, which is what it drives.

    Disarmed, nothing is queued, so the step that follows
    (`build_connector_meta`, which drains `_pending_finished_stores`) has
    nothing to store either. `None` is also exactly what the fork returns when
    it queues nothing (`state is None or gpu_pool is None`), so a decline is
    indistinguishable from "nothing to do" - which it is.
    """
    def _queue_finished_eager_store(self, request, block_ids):
        if not _state["armed"]:
            return None
        return original(self, request, block_ids)

    setattr(_queue_finished_eager_store, SWITCH_ATTR, SWITCH_TAG)
    setattr(_queue_finished_eager_store, ORIGINAL_ATTR, original)
    _queue_finished_eager_store.__doc__ = (
        "kvtier-dev-plugin kill switch: queue nothing while the tier is "
        "disarmed, so the finish path never reaches "
        "_select_eager_blocks_to_store; else the engine's own queueing.")
    return _queue_finished_eager_store


_SWITCH_FACTORIES = {
    "get_num_new_matched_tokens": _make_matched_wrapper,
    "build_connector_meta": _make_meta_wrapper,
    "has_pending_stores": _make_pending_wrapper,
    "_queue_finished_eager_store": _make_finished_store_wrapper,
}


def _install_switch(cls, name) -> dict:
    """Wrap one scheduler-facing method with the arm check. Idempotent.

    No fingerprint: the wrapper only ADDS a gated early-return around the
    engine's own function (it never reimplements it), so a changed body cannot
    be silently mispatched the way a copied loop can. Ownership discipline and
    the original-binding are identical to store_hook.

    Guard-mode composition: when the current attribute is one of OUR layers
    (guard armed before this install ran, or a re-install arriving while the
    guard sits on top), the switch goes UNDER it and the guard is rebuilt on
    top - nesting preserved, never replaced. A stack that already contains a
    switch layer is left untouched, so a re-install can neither nest a second
    switch nor strip the guard off it.
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
    layers, bottom = _layers_of(existing)
    if layers:
        kinds = [kind for kind, _ in layers]
        if "switch" in kinds:
            report["wrapped"] = True
            report["reason"] = ("already wrapped (verified our switch layer "
                                f"under kvtier-dev-plugin layers: {kinds})")
            return report
        if "backport" in kinds:
            # METHODS and SWITCH_METHODS are disjoint; a backport layer here
            # means a stack we cannot rebuild faithfully. Say so, do not guess.
            report["reason"] = (f"{name} sits under a backport layer we cannot "
                                "rebuild for a switch method, refusing")
            return report
        if bottom is None or not callable(bottom):
            report["reason"] = (f"the ownership chain under our layers "
                                f"({kinds}) does not end in a callable body")
            return report
        setattr(cls, name, _rewrap(name, _SWITCH_FACTORIES[name](bottom), layers))
        report["wrapped"] = True
        report["reason"] = ("wrapped under kvtier-dev-plugin's own layer(s) "
                            f"({kinds}); composition preserved, the body "
                            "underneath is the engine's own function")
        return report
    setattr(cls, name, _SWITCH_FACTORIES[name](existing))
    report["wrapped"] = True
    return report


# --- installation -----------------------------------------------------------

def in_tree_state(cls) -> dict:
    """Classify each backported method against the engine's OWN source text.

    Per method, exactly one of:
      `in_tree_fixed` - the body IS the upstream #56810 text (the patch has been
                        applied to the engine's own file; stand down),
      `unpatched`     - the body IS the v0.30.0 text we know how to replace,
      `other`         - neither: the fork changed this function (or a third
                        plugin replaced it) - the runtime path already refuses
                        these per method, and this function only names them,
      `unreadable`    - no source to hash (a refusal, never a guess),
      `missing`       - the method is not on this class at all.

    The hash is taken on the body the engine would ACTUALLY RUN, found by
    walking DOWN through our own layers with `_layers_of()` - never on the layer
    itself. That is the whole lesson of the third incident: hashing our own guard
    wrapper and reporting "the fork changed this function" sent an operator
    hunting a fork update that does not exist. Two consequences fall out of that
    choice, and both are load-bearing:
      * a class we patched at RUNTIME classifies as `unpatched` (the bottom is
        still the fork's original), so a stand-down can never be claimed for our
        own copy, and the runtime fallback stays the honest answer; and
      * an in-tree-fixed class stays `in_tree_fixed` even with guard/switch
        layers on top of it, which is what lets a re-poll stand down too.

    Read-only and never raises into the caller: a classification we cannot make
    is reported as such.
    """
    out = {"checked": False, "reason": None, "methods": {},
           "fixed": [], "unpatched": [], "other": [],
           "all_fixed": False, "partial": False}
    if cls is None:
        out["reason"] = (f"no {TARGET_CLASS} is available to inspect")
        return out
    for name in METHODS:
        entry = pr56810_sources.SOURCES[name]
        patched = pr56810_sources.PATCHED_SHA256[name]
        raw = cls.__dict__.get(name)
        fn = _function_of(raw) if raw is not None else None
        layers, bottom = _layers_of(fn) if fn is not None else ([], None)
        info = {"state": None, "reason": None, "live_sha256": None,
                "orig_sha256": entry["orig_sha256"], "patched_sha256": patched,
                "layers": [kind for kind, _ in layers]}
        body = _function_of(bottom) if bottom is not None else None
        if body is None:
            info["state"] = "missing"
            info["reason"] = (f"{name} is not defined on this class" if fn is None
                              else "the ownership chain under our layers is broken")
        else:
            live = fingerprint_of(body)
            info["live_sha256"] = live
            if live is None:
                info["state"] = "unreadable"
                info["reason"] = ("the engine's own body has no readable source")
            elif live == patched:
                info["state"] = IN_TREE_FIXED
            elif live == entry["orig_sha256"]:
                info["state"] = "unpatched"
            else:
                info["state"] = "other"
                info["reason"] = (f"live {live[:12]} is neither the v0.30.0 body "
                                  f"({str(entry['orig_sha256'])[:12]}) nor the "
                                  f"#56810 body ({patched[:12]})")
        out["methods"][name] = info
        if info["state"] == IN_TREE_FIXED:
            out["fixed"].append(name)
        elif info["state"] == "unpatched":
            out["unpatched"].append(name)
        else:
            out["other"].append(name)
    out["checked"] = True
    out["all_fixed"] = len(out["fixed"]) == len(METHODS)
    # `partial` is the state that must never be accepted: some of the fix is in
    # the tree and some is not, so the crash lives on in whichever half did not
    # land. It is reported as a refusal by install(), not patched over.
    out["partial"] = bool(out["fixed"]) and not out["all_fixed"]
    if out["all_fixed"]:
        out["reason"] = IN_TREE_FIXED
    elif out["partial"]:
        behind = out["unpatched"] + out["other"]
        out["reason"] = (
            f"in the tree already: {', '.join(out['fixed'])}; "
            f"still on the v0.30.0 or a fork-changed body: "
            f"{', '.join(behind)}")
    return out


def _in_tree_verdict(name, state) -> dict:
    """The per-method verdict for a body the engine already carries fixed.

    `patched: True` because the requirement the flag reports is satisfied - the
    body that runs is the upstream one - while NOTHING on the class was touched.
    Callers must read `already_fixed_in_tree` alongside it to know we installed
    no code of our own.
    """
    info = state["methods"][name]
    return {"patched": True, "sha_ok": True,
            "already_fixed_in_tree": True,
            "live_sha256": info["live_sha256"],
            "in_tree_sha256": info["patched_sha256"],
            "expected_sha256": info["orig_sha256"],
            "reason": f"{IN_TREE_FIXED}: the engine's own {name} already carries "
                      f"the upstream #{pr56810_sources.UPSTREAM_PR.rsplit('#', 1)[-1]} "
                      "body, so the runtime backport installed nothing"}


def install(cls, kv_cache_config=None, manager=None) -> dict:
    """Fit the kill-switch wrappers, then backport PR #56810 onto `cls`.

    Never raises. Idempotent and marker-guarded like store_hook. A stale
    fingerprint is a per-method refusal, so one moved function cannot make the
    others silently wrong: the verdict for each is reported independently.

    `manager` is the LIVE `SimpleCPUOffloadScheduler` instance this class will
    actually run (the connector-creation hook and `ensure_installed` have it;
    class-only callers do not). When present, the install must leave THAT
    object safe: production constructs first, so the `__init__` wrapper alone
    cannot deliver the additive attribute - `_live_object_check` back-fills it
    and verifies every other `self` read. A live manager we cannot make safe
    fails the whole install (disarmed, `incomplete_install`), because patched
    methods running against an instance missing their input is the exact
    engine death of 2026-10-03 12:41.

    THE ORDER IS THE POINT: the wrappers go on FIRST, before the geometry guard
    and before a single upstream body is compiled. The backport can be refused
    (a fingerprint drift) or can raise midway, and in BOTH cases the process must
    be left in "wrappers present + disarmed" - which, with the finish path gated
    as well as the per-step trio, means NEITHER route to the crashing selector is
    open. Fitting the wrappers last would leave a geometry-vetoed or
    half-installed class with the finish route still wide open.
    """
    report = {"installed": False, "class": None, "reason": None,
              "methods": {}, "switch": {}, "geometry": None,
              "live_check": None}
    try:
        if cls is None:
            # THE BOOT CASE (`register()` runs at core.py:217, the Scheduler - and
            # so this class - arrives at core.py:260). Refusing here does NOT
            # disarm and never did; what makes it safe is that the state it
            # refuses INTO is now `armed: False`, so an uninstalled tier is also
            # an idle one. `arm_block.kind` reports "no_manager" in this state.
            # There is nothing to fit wrappers ONTO here - which is exactly the
            # hole the connector-creation hook closes.
            return _refuse(report, f"no {TARGET_CLASS} is imported in this "
                                   f"process ({TARGET_MODULE} absent)")
        report["class"] = getattr(cls, "__name__", type(cls).__name__)

        # FIRST: the kill switch. It needs no fingerprint (it only adds a gated
        # early-return around the engine's own function) and it is what makes a
        # refused, vetoed or half-installed tier a strict no-op on BOTH routes
        # into the selector.
        for name in SWITCH_METHODS:
            verdict = _install_switch(cls, name)
            report["switch"][name] = verdict
            _state["switch_verdicts"][name] = verdict

        guard = geometry_guard(
            kv_cache_config if kv_cache_config is not None
            else _config_of(cls))
        if (not guard.get("checked") and _state["geometry"]
                and _state["geometry"].get("checked")
                and not _state["geometry"].get("ok")):
            # A check with NO config proves nothing, so it must not overwrite a
            # veto we were already given from a live one: forgetting it here
            # would re-open exactly the fail-open the guard exists to close
            # (patching - and then arming - a config that must not be patched).
            guard = dict(_state["geometry"])
        report["geometry"] = guard
        _state["geometry"] = guard
        if kv_cache_config is not None and not guard.get("ok"):
            # The patch would skip a group whose KV is needed on resume: turn
            # the tier OFF rather than leave it armed to drop cacheable KV. The
            # methods are not attempted at all here, so nothing is refused per
            # method - the whole install is refused on geometry. The kill-switch
            # wrappers are already fitted above, so this class is a no-op with
            # both selector routes gated rather than an unwrapped crash.
            _state["fully_installed"] = False
            _state["refused_methods"] = []
            report["refused_methods"] = []
            _force_disarm("geometry", guard.get("reason"))
            report["auto_disarmed"] = True
            report["disarm_reason"] = "geometry"
            return _refuse(report, guard.get("reason"))

        # THE LIVE OBJECT IS CHECKED BEFORE ANY METHOD BODY IS PATCHED: on a
        # construct-first engine, patching the class while the one live
        # instance cannot carry the new attribute is the 12:41 crash. A failed
        # check therefore refuses the METHODS loop entirely (the kill-switch
        # wrappers above keep both selector routes gated in the meantime).
        # A manager-less install must NOT forget a veto a manager-carrying
        # install already got - the same preservation rule as the geometry
        # guard above (`_arm_proof` installing onto the class cannot make a
        # live instance safe, and did before this rule; a test caught it).
        prior_live = _state["live_check"]
        if (manager is None and prior_live and prior_live.get("checked")
                and not prior_live.get("ok")):
            live = dict(prior_live)
        else:
            live = _live_object_check(manager)
            if manager is not None or prior_live is None:
                _state["live_check"] = live
        report["live_check"] = live
        if manager is None and live.get("checked") and not live.get("ok"):
            # Preserved veto: refuse exactly as the checked install did.
            reason = (f"live-object check failed: {live.get('reason')}")[:400]
            _state["fully_installed"] = False
            _state["refused_methods"] = []
            report["refused_methods"] = []
            report["auto_disarmed"] = True
            report["disarm_reason"] = "incomplete_install"
            _force_disarm("incomplete_install", reason)
            return _refuse(report, reason)
        if manager is not None and not live["ok"]:
            reason = (f"live-object check failed: {live.get('reason')}")[:400]
            _state["fully_installed"] = False
            _state["refused_methods"] = []
            report["refused_methods"] = []
            report["auto_disarmed"] = True
            report["disarm_reason"] = "incomplete_install"
            _force_disarm("incomplete_install", reason)
            return _refuse(report, reason)

        # THE STAND-DOWN, decided BEFORE any body is compiled: if the engine's
        # own file already carries the upstream #56810 text, installing our copy
        # on top would be a second, invisible patch of a file we now track - and
        # the fingerprint guard would (correctly, from its point of view) refuse
        # the fixed body and blame the fork. So we look first, and act on three
        # states:
        #   all four in-tree fixed -> install nothing, report `already_fixed_in_tree`
        #   some fixed, some not   -> REFUSE. A half-applied in-tree patch keeps
        #                             the crash alive in whichever half did not
        #                             land, and quietly patching the rest from a
        #                             plugin is how the fix becomes unauditable.
        #   none fixed             -> exactly the runtime path as before, which
        #                             is the working fallback for a box where the
        #                             patch has not been applied.
        in_tree = in_tree_state(cls)
        report["in_tree"] = in_tree
        _state["in_tree"] = in_tree
        if in_tree.get("partial"):
            # Fail closed and say which methods are which. The kill-switch
            # wrappers fitted above are what keep both routes into the crashing
            # selector gated while the operator fixes the tree.
            reason = (f"the engine is PARTIALLY patched in-tree "
                      f"({in_tree['reason']}); refusing to complete a "
                      "half-applied in-tree #56810 patch from a plugin - apply "
                      "the tracked patch (or roll it back) so the engine's own "
                      "file is one whole state")
            _state["fully_installed"] = False
            _state["refused_methods"] = []
            report["refused_methods"] = []
            report["auto_disarmed"] = True
            report["disarm_reason"] = "incomplete_install"
            _force_disarm("incomplete_install", reason)
            return _refuse(report, reason)
        stand_down = bool(in_tree.get("all_fixed"))
        if stand_down:
            report["already_fixed_in_tree"] = True

        for name in METHODS:
            verdict = (_in_tree_verdict(name, in_tree) if stand_down
                       else _install_one(cls, name))
            report["methods"][name] = verdict
            _state["method_verdicts"][name] = verdict

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
            if stand_down:
                # Nothing was installed, so the install/reinstall counters stay
                # untouched: they count code we fitted, and claiming a fit we did
                # not make would make `INSTALLED` a lie in the log. Logged once per
                # episode - the snapshot polls the install path every few seconds.
                _state["in_tree_standdowns"] += 1
                if not _state["in_tree_notice"]:
                    _state["in_tree_notice"] = True
                    logger.warning(
                        "kvtier-dev-plugin: PR #56810 is ALREADY FIXED IN TREE "
                        "on %s (all %d methods carry the upstream body); runtime "
                        "backport STOOD DOWN, nothing installed",
                        report["class"], len(METHODS))
            elif _state["installs"] == 0:
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

    layers, bottom = _layers_of(_function_of(existing) if existing is not None
                                else None)
    if layers:
        # The current attribute is one of OUR layers (guard mode sits on top
        # of this method, or of our copy of it). The fingerprint belongs to
        # the body UNDERNEATH - never to our own wrapper (that produced the
        # "the fork changed this function since 10b8e07" refusal that blamed
        # the fork for our own composition and auto-disarmed a healthy tier).
        return _install_under_layers(cls, name, layers, bottom)

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


def _install_under_layers(cls, name, layers, bottom) -> dict:
    """Re-fit our copy BENEATH our own layers; fingerprint the body beneath.

    `layers` is outermost-first and non-empty (so is guard mode's wrapper,
    a switch layer, or our previous copy). The fork's body is `bottom`; it
    alone is fingerprinted. On a match, the copy is rebuilt from it and every
    layer is rebuilt on top (nesting preserved, guard semantics unchanged);
    on a mismatch the refusal says what is true - the body UNDERNEATH our
    layers changed - and never blames our own wrapper (or the fork, for our
    wrapper's fingerprint). This is also the path where guard was armed
    BEFORE the backport ever installed: `bottom` is then the pristine fork
    function, and the result is the same composed stack.
    """
    kinds = [kind for kind, _ in layers]
    report = {"patched": False, "reason": None, "sha_ok": None,
              "live_sha256": None, "expected_sha256": None}
    if bottom is None:
        report["reason"] = (f"the ownership chain under our layers ({kinds}) is "
                            "broken (an ORIGINAL attribute is missing)")
        return report
    ok, live, expected = _fingerprint_verdict(name, _function_of(bottom))
    report.update(sha_ok=ok, live_sha256=live, expected_sha256=expected)
    if not ok:
        if live is None:
            report["reason"] = (
                f"{name}: the fork's own body underneath kvtier-dev-plugin's "
                f"layer(s) ({' + '.join(kinds)}) has no readable source; "
                "refusing to patch a body we cannot vouch for")
        else:
            report["reason"] = (
                "source fingerprint mismatch UNDER kvtier-dev-plugin's own "
                f"layer(s) ({' + '.join(kinds)}): the fork's own body "
                "underneath changed since 10b8e07 "
                f"(live {str(live)[:12]} != expected {str(expected)[:12]}); "
                "refusing to patch a body we can no longer vouch for "
                "(the refusal is NOT about our own wrapper)")
        return report
    if name == "__init__":
        base = _make_init_wrapper(_function_of(bottom))
    else:
        base = _compile_copy(name, pr56810_sources.SOURCES[name]["source"],
                             _function_of(bottom))
        if base is None:
            report["reason"] = ("compiled copy is missing its own fingerprint "
                                "anchor")
            return report
    setattr(cls, name, _rewrap(name, base, layers))
    report["patched"] = True
    report["reason"] = (f"installed under kvtier-dev-plugin's own layer(s) "
                        f"({' + '.join(kinds)}); the current implementation "
                        "is our wrapper, and the fork's body underneath was "
                        "fingerprint-verified")
    return report


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
    """True when every method we own still carries OUR LAYER on `cls`.

    A cheap read-only check (a handful of `getattr`s through the ownership
    chains, no hashing, no `compile`) that lets the polling path skip the
    install entirely. It is deliberately NOT a memo of "we installed once": if
    another plugin, or a module reload, replaced one of our methods, the marker
    disappears and the next call does the work again. That is the point - it is
    a claim about the class as it is NOW.

    Layer-aware (2026-10-03): a method whose current attribute is guard mode's
    wrapper OVER our layer still counts - our code is live beneath it, and
    claiming otherwise every few seconds would push three upstream recompiles
    onto the serving loop while a dev guard is armed.
    """
    if cls is None:
        return False
    for name, wanted in ([(n, "backport") for n in METHODS]
                         + [(n, "switch") for n in SWITCH_METHODS]):
        raw = cls.__dict__.get(name)
        if raw is None:
            return False
        layers, _bottom = _layers_of(_function_of(raw))
        if not layers or all(kind != wanted for kind, _ in layers):
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
        if not _state["hook_fitted"]:
            # The connector-creation hook is what makes a boot patch itself. If
            # this process imported the factory only AFTER plugin load, fit it
            # here: once fitted it is a marker check, and re-fitting re-binds the
            # fork's own function, so this can never nest a second hook.
            install_connector_hook()
        # THE SECOND TARGET, retried here. The boot fits it at connector creation;
        # this covers the one case that hook cannot reach - a process that imported
        # `single_type_kv_cache_manager` only after plugin load. `ensure_installed`
        # is the verb path, so a `/tier` contact settles the verdict without a
        # restart. Contained by `mamba_anchor.install()`'s own contract (it never
        # raises), and it is a marker check once the class carries our wrapper.
        mamba_anchor.ensure_installed()
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
        # class. The LIVE OBJECT is re-verified too (cheap attribute reads):
        # markers on the class prove nothing about an instance constructed
        # between two polls by a half-patched boot, and the back-fill is
        # idempotent, so this is also where a late-arriving manager gets its
        # additive attribute.
        live = _live_object_check(manager)
        if manager is not None or _state["live_check"] is None:
            _state["live_check"] = live
        if manager is not None and not live["ok"]:
            report = {"installed": False, "class": getattr(cls, "__name__", None),
                      "reason": None, "methods": {}, "switch": {},
                      "geometry": guard, "live_check": live}
            out = _refuse(report,
                          f"live-object check failed: {live.get('reason')}")
            # Same rule as install(): a patched class over an instance that
            # would raise is the crash, not an install. Refuse, disarm, and
            # the arm gate keeps the patched code unreachable.
            _force_disarm("incomplete_install", report["reason"])
            out["auto_disarmed"] = True
            out["disarm_reason"] = "incomplete_install"
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
                "live_check": live,
                "refused_methods": [], "auto_disarmed": bool(_state["auto_disarmed"]),
                "disarm_reason": _state["disarm_reason"]}
    return install(cls, config, manager)


def manager_of(engine):
    """The scheduler-side manager off a live engine, or None. Reads only."""
    scheduler = getattr(engine, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    return getattr(connector, "scheduler_manager", None)


# --- the connector-creation hook: when the patch ACTUALLY lands --------------
#
# WHY A HOOK AT ALL. `register()` cannot install: it runs at `core.py:217` and
# the manager class is first imported while the Scheduler is being built at
# `core.py:260`, so the boot install attempt is ALWAYS refused on `cls is None`
# and the only later chances were HTTP verbs. That made "patched before the
# first request" a claim about operator habit, not about the boot order.
#
# WHY THIS HOOK IS THE RIGHT MOMENT. The connector module
# (`.../v1/simple_cpu_offload_connector.py:21`) imports the manager at module
# scope, and `KVConnectorFactory.create_connector` is the only thing that
# imports THAT module - in both of its two callers (`scheduler.py:164` for the
# scheduler role, `kv_transfer_state.py:90` for the worker role). So the instant
# `create_connector` returns is the first instant the manager class exists, and
# it is still inside `EngineCore.__init__`/`Scheduler.__init__`: before the first
# step, before the first `bind_gpu_block_pool`, long before any request. The
# scheduler-role call also hands us a manager with a LIVE `kv_cache_config`, so
# the geometry guard gets a real check at boot instead of `checked: False`.
#
# WHY IT CANNOT BE TOO EARLY OR TOO LATE. `load_general_plugins()` (core.py:217)
# precedes every `create_connector` call in the tree (`scheduler.py:164` is
# reached from core.py:260; the worker's pair is `worker_base.py:288` then
# `kv_transfer_state.py:90`), so the hook is fitted before the class it wraps can
# be called. Verified against the fork: importing `vllm.v1.engine.core` - which
# is what makes `register()` able to attach verbs at all - already puts the
# factory module in `sys.modules`, while the scheduler and offload-manager
# modules are NOT there yet.

CONNECTOR_FACTORY_MODULE = "vllm.distributed.kv_transfer.kv_connector.factory"
CONNECTOR_FACTORY_CLASS = "KVConnectorFactory"
CREATE_CONNECTOR = "create_connector"
CONNECTOR_MODULE = ("vllm.distributed.kv_transfer.kv_connector.v1."
                    "simple_cpu_offload_connector")
CONNECTOR_CLASS = "SimpleCPUOffloadConnector"

# Its OWN marker, distinct from backport-1/switch-1/store-1/capacity-1: a hook on
# a factory classmethod is a different kind of patch and must never be mistaken
# for one on the manager.
HOOK_ATTR = "__kvtier_dev_connector_hook__"
HOOK_TAG = "kvtier-dev-plugin/connector-hook-1"
HOOK_ORIGINAL = "__kvtier_dev_connector_hook_original__"


def connector_factory_class():
    """`KVConnectorFactory` if this process already imported it. Never imports."""
    module = sys.modules.get(CONNECTOR_FACTORY_MODULE)
    if module is None:
        return None
    return getattr(module, CONNECTOR_FACTORY_CLASS, None)


# --- THE BOOT: install, guard, arm - in that order --------------------------
#
# WHAT RUNS, AND WHY IT RUNS HERE. `register()` cannot do any of this: it runs at
# `core.py:217`, while the manager class is first imported inside
# `KVConnectorFactory.create_connector`, so the connector-creation hook is the
# earliest instant at which any of it is possible - and it is still inside
# `Scheduler.__init__`, long before the first request. The boot order is:
#
#   install(cls, config, manager)  -> kill-switch wrappers FIRST, then PR #56810,
#                                     all-or-nothing, with the LIVE-object check
#   guard                          -> `guard_mode` armed and its wrappers fitted,
#                                     composed OVER the layers just installed and
#                                     never replacing them, so there is no window
#                                     in which a store or a lookup runs unguarded
#   arm                            -> `set_armed(True)` ONLY if the install
#                                     verified: every method patched, the live
#                                     instance checked and safe, and no entry
#                                     point the class actually HAS left unguarded
#
# ARMING IS GATED ON PROOF, NOT ON OPTIMISM. A verified install is the state a
# manual `POST {"op":"arm"}` already had to prove; the boot seeks the same proof
# through the same `set_armed()` gate, which re-runs `_arm_block_reason()` itself
# - a second request for proof, not a second and weaker rule. It adds ONE
# requirement over a human arm: the LIVE-OBJECT CHECK MUST HAVE RUN AND PASSED. A
# human reading `live_check: unchecked` can decide; a boot cannot, because
# patching a class whose one live instance is missing an attribute the patched
# body reads is the 2026-10-03 12:41 engine death.
#
# WORKING OVER PERFECT, STATED PLAINLY: an entry point the fork does not define
# at all (`take_events` on this fork is one) is reported under `not_covered` and
# does NOT block the auto-arm. Refusing to arm because of a method that cannot
# exist would mean the boot never works on a trimmed fork, which is the opposite
# of the point. It is reported, not hidden.

def boot_mode() -> tuple:
    """("auto" | "inert", why an operator would read it). Read fresh, per boot.

    `VLLM_E2_TIER_DEV_BOOT_INERT` set = the old inert boot (install only, every
    effect by hand). Unset = auto. That variable only ever DISABLES a boot effect:
    nothing in this package reads an env var that WARMS a dev facility up, which
    is the property `test_no_env_var_can_arm_guard_mode_and_at_import_it_is_off`
    pins. An operator can turn the working boot OFF from a launcher script; they
    cannot arm guard mode from one.
    """
    if config.boot_inert():
        return BOOT_MODE_INERT, (config.ENV_BOOT_INERT + " is set: the boot "
                                 "installs and stops there (no guard, no arming). "
                                 "Arm by hand: POST " + tier_route()
                                 + ' {"op":"arm"}')
    return BOOT_MODE_AUTO, (config.ENV_BOOT_INERT + " unset (the default): at "
                            "connector creation the boot installs PR #56810, arms "
                            "the guard, and arms the tier once the install "
                            "verifies")


def _guard_layer_on(cls, method) -> bool:
    """Is the guard a layer of `cls.method`, wherever it sits in the stack?

    Layer-aware on purpose: `guard_mode.install()` composes its wrapper OVER the
    backport's and the switch's, and a later install by another module of this
    package could put a layer outside it. Asking `_layers_of()` for a `guard` kind
    answers "is an exception here captured and declined?" about the attribute the
    scheduler actually resolves, which is the question the auto-arm gate needs.
    """
    try:
        current = getattr(cls, method, None)
        if current is None:
            return False
        layers, _bottom = _layers_of(_function_of(current))
        return any(kind == "guard" for kind, _wrapper in layers)
    except Exception as exc:  # noqa: BLE001 - unreadable counts as "not guarded"
        snapshot._error(exc, "backport guard coverage")
        return False


def _class_owns(cls, method) -> bool:
    """Does THIS class own `method`, or only inherit it from some other module?

    THE QUESTION THAT DECIDES WHETHER THE BOOT EVER ARMS ON A REAL ENGINE.
    `guard_mode.install()` will not shadow a method whose home is neither the fork's
    manager module nor this package - it refuses to take over another module's
    method, and that rule is correct. But `SimpleCPUOffloadScheduler` inherits
    `take_events`, `reset` and `update_connector_output` from the connector BASE
    class (`vllm.distributed.kv_transfer.kv_connector.v1.base`), so on a real engine
    those three can NEVER carry a guard layer. A gate that read "the class has the
    attribute, therefore the guard must own it" would refuse to arm on every real
    boot: fail-closed, safe, and a feature that works in a test fixture and nowhere
    else. So the gate asks who OWNS the attribute: an entry point this class (or the
    fork's own module) defines is a hole if it is unguarded; one it merely inherits
    from a module the guard will not touch is a known, reported limit.

    Fails CLOSED on an unreadable MRO: "cannot tell" reads as "owned", which sends
    the decision to `_guard_layer_on`, where unreadable reads as "not guarded".

    The home is read from the FUNCTION, falling back to the defining class, because
    that is the same question `guard_mode.install()` asks when it decides whether an
    attribute is ours to wrap - the gate and the guard must agree, or the gate
    blocks on a method the guard was never going to touch.
    """
    try:
        for klass in getattr(cls, "__mro__", None) or (cls,):
            found = vars(klass).get(method)
            if found is None:
                continue
            home = getattr(_function_of(found), "__module__", None) \
                or getattr(klass, "__module__", None)
            return home == TARGET_MODULE
        return False
    except Exception as exc:  # noqa: BLE001 - unreadable ownership is a hole
        snapshot._error(exc, "backport method ownership")
        return True


def _guard_coverage(cls) -> dict:
    """What the guard covers on THIS class, and what it does not. Reads only.

    `wrapped`     methods whose live attribute carries a guard layer.
    `not_covered` methods the guard cannot cover on this class: ones the fork
                  renamed or dropped (nothing to guard) AND ones the class only
                  inherits from a module outside the fork's own, which
                  `guard_mode.install()` will not shadow (see `_class_owns`). They
                  do NOT block the auto-arm, but they are reported - the payload
                  must say what is NOT covered rather than imply completeness.
    `refused`     entry points THIS class owns that carry no guard layer (a foreign
                  owner marker, an attribute that is not callable, or an install
                  that declined them). These DO block the auto-arm: running the
                  tier with a live hole in the guard is the exact state the box
                  tool has always refused.
    """
    from . import guard_mode  # noqa: PLC0415 - sibling layer, lazy by idiom
    out = {"wrapped": [], "not_covered": [], "refused": []}
    if cls is None:
        out["not_covered"] = list(guard_mode.WRAPPED_METHODS)
        return out
    for method in guard_mode.WRAPPED_METHODS:
        if getattr(cls, method, None) is None:
            out["not_covered"].append(method)
        elif _guard_layer_on(cls, method):
            out["wrapped"].append(method)
        elif not _class_owns(cls, method):
            out["not_covered"].append(method)
        else:
            out["refused"].append(method)
    return out


def _boot_guard(cls) -> dict:
    """Arm GUARD MODE and fit its wrappers on the live class. Never raises.

    Goes through `set_guard_mode()` - the one mutation that owns the armed flag -
    and only falls back to installing on `cls` directly for the case where the
    module-level class is not resolvable but the connector's manager class is.
    Idempotent either way: a method already carrying a guard layer is re-bound to
    ITS `ORIGINAL`, so a re-run cannot nest a guard inside a guard.
    """
    from . import guard_mode  # noqa: PLC0415 - sibling layer, lazy by idiom
    out = {"armed": False, "installed": False, "nested": [], "reason": None}
    try:
        report = guard_mode.set_guard_mode(True, None)
        install_report = report.get("install") or {}
        if not install_report.get("installed") and cls is not None:
            install_report = guard_mode.install(cls)
        out["armed"] = bool(guard_mode.is_armed())
        out["installed"] = bool(install_report.get("installed"))
        out["nested"] = list(install_report.get("nested") or [])
        out["reason"] = (report.get("reason") if out["installed"] else
                         (install_report.get("reason") or report.get("reason")))
        return out
    except Exception as exc:  # noqa: BLE001 - a guard we cannot fit is a report
        snapshot._error(exc, "backport boot guard")
        out["reason"] = f"{type(exc).__name__}: {exc}"[:MAX_BOOT_ITEM_CHARS]
        return out


def _boot_restore(cls) -> dict:
    """Fit the SCHEDULER half of the SSD restore path, and report what is on.

    A dev-path convenience with the same safety posture as everything else in this
    boot: `disk_restore.adopt_if_missed(cls)` installs nothing unless there IS a
    scheduler class to fit (the one resolved by name, or the one this boot is
    holding), arms nothing (the restore gate is the spill tier's arm plus the
    tier's arm, read at offer time), and cannot raise.

    The report answers the boot-time question - "can this process restore from the
    SSD tier, and if not, which link is missing?" - in the same shape on every
    path, because `pr56810.boot.restore` is what an operator reads without a grep.
    """
    from . import disk_restore  # noqa: PLC0415 - sibling layer, lazy by idiom
    out = {"fitted": False, "class": None,
           "manager_class": getattr(cls, "__name__", None),
           "methods": [],
           "methods_expected": list(disk_restore.SCHED_METHODS),
           "gate": "unknown", "gate_reason": None, "ranks_proven": 0,
           "index_entries": 0, "install_refusals": 0, "install_deferrals": 0,
           "reason": None, "role": None}
    try:
        out["role"] = disk_restore.role_here()
        report = disk_restore.adopt_if_missed(cls)
        view = disk_restore.scheduler_section()
        out["fitted"] = bool(view["layer_installed_here"])
        out["class"] = view["class_here"]
        out["methods"] = list(view["methods_here"])
        out["gate"] = "open" if view["gate"]["ok"] else "closed"
        out["gate_reason"] = view["gate"]["reason"]
        out["ranks_proven"] = int(view["ranks_proven"])
        out["index_entries"] = int(view["index"]["entries"])
        out["install_refusals"] = int(view["install_refusals"])
        out["install_deferrals"] = int(view["install_deferrals"])
        if not out["fitted"]:
            detail = (view["install_refusal_line"] or report.get("reason")
                      or view["gate"]["reason"] or "no reason recorded")
            out["reason"] = (
                f"the scheduler half is not fitted here (role={out['role']}, "
                f"wrapped={view['methods_here'] or 'none'}): {detail}"
            )[:MAX_BOOT_ITEM_CHARS]
        return out
    except Exception as exc:  # noqa: BLE001 - a restore verdict never breaks a boot
        snapshot._error(exc, "backport boot restore")
        out["reason"] = f"the restore fit raised {type(exc).__name__}"[:200]
        return out


def boot_from_connector(cls, install_report, manager,
                        load_failure_policy=None) -> dict:
    """THE BOOT, run by the connector-creation hook. Never raises.

    Three verdicts and one log line: `backport` (the fix fully installed AND the
    live instance safe), `guard` (wrappers on, and armed), `tier` (armed, and only
    armed because the first two verified). It reports the install verdict rather
    than re-deciding it, and the arm goes back through `set_armed()`, which
    re-checks the gate - so a partial install still force-disarms and logs loudly
    exactly as it did before the boot was automated.
    """
    report = install_report if isinstance(install_report, dict) else {}
    boot = {"mode": None, "mode_why": None, "backport": False, "guard": False,
            "guard_armed": False, "tier": False, "faults_at_boot": 0,
            "refusals": [], "not_covered": [], "nested": [], "live_check": None,
            "manager": type(manager).__name__ if manager is not None else None,
            # WHERE the fix comes from: "in_tree" (the engine's own file carries
            # it, the runtime backport stood down) or "runtime" (our copies are
            # live on the class). `backport: True` alone cannot tell those apart,
            # and after this work `in_tree` is the state every box should reach.
            "backport_source": None,
            # THE SSD RESTORE PATH's scheduler half, fitted at this same moment.
            # `None` means this boot never reached the restore step; a dict is the
            # verdict, and `fitted: false` there is a FAULT TO READ, because a boot
            # that has a manager and does not fit the restore layer is a boot whose
            # spilled KV can never be read back. See `_boot_restore`.
            "restore": None,
            # THE SSD TIER'S BOOT DEFAULT (0.2.33): whether
            # `KVTIER_DISK_BOOT_ENABLE` asked for a boot that comes up armed and
            # serving, whether this boot honoured it, and what an operator can do about
            # it. `None` means this boot never reached the step; the verdict itself is
            # `disk_boot`'s own stable-key view. See `disk_boot.py` - including the
            # prohibition it must never cross, which is that this is computed FRESH and
            # never restored from a previous boot's arm file.
            "disk_boot": None,
            # THE SPILL JOURNAL DRAIN (0.2.65): what the boot read out of the
            # journals, how long it took and which bound stopped it - the answer to
            # "will the first request find a converged index, or recompute?". `None`
            # means this boot never reached the step (a boot note for a process with no
            # tier to boot).
            "journal_drain": None,
            # THE LOOKUP LEDGER'S BOOT ARM (0.2.66): whether this boot armed the
            # per-request offer/adopt ring, and if not which of the five verdicts says
            # why. OBSERVABILITY ONLY, and that is the whole design: a refusal here is
            # counted and logged and NEVER joins `refusals`, because a missing
            # diagnostic
            # is not a tier that declined to serve. `None` means the step never ran.
            "lookup_ledger": None,
            # WHO DECIDES THE ENGINE'S KV LOAD-FAILURE POLICY (0.2.73): the verdict of
            # the step that gives `kv_load_failure_policy` the value `recompute` when
            # nothing in the operator's own configuration named it. `None` means the
            # step never ran in this boot; a dict is the verdict, and its `after` is
            # what the engine will ACTUALLY do with a KV load that cannot be served -
            # read back off the config object, never the value this plugin intended to
            # write. Reporting, never a refusal: `fail` here is a posture somebody chose
            # or could not be located, not a tier that declined to serve.
            "load_failure_policy": (load_failure_policy
                                    if isinstance(load_failure_policy, dict)
                                    else None),
            "install_reason": report.get("reason"), "line": None}
    try:
        mode, why = boot_mode()
        boot["mode"], boot["mode_why"] = mode, why
        live = report.get("live_check") or {}
        boot["live_check"] = live

        # "Verified" is the install's own all-or-nothing verdict AND a live-object
        # check that ran on the instance the engine is about to call. No manager
        # means no such check, and a boot cannot make the judgement a human would
        # make from `live_check: unchecked` - so it declines, and says so.
        installed = bool(report.get("installed"))
        live_ok = bool(live.get("ok")) and manager is not None
        boot["backport"] = installed and live_ok
        if installed:
            boot["backport_source"] = (
                "in_tree" if report.get("already_fixed_in_tree") else "runtime")
        if not installed:
            boot["refusals"].append("backport: {}".format(
                report.get("reason") or "the install did not complete"))
        elif not live_ok:
            boot["refusals"].append(
                "backport: the live-object check did not pass: {}".format(
                    live.get("reason") or "no live manager was checked, so an "
                    "instance constructed BEFORE the patch was fitted would be "
                    "unverified"))

        if mode == BOOT_MODE_INERT:
            boot["refusals"].append(
                f"guard: not installed ({config.ENV_BOOT_INERT})")
            boot["refusals"].append(
                f"tier: not armed ({config.ENV_BOOT_INERT})")
        else:
            guard = _boot_guard(cls)
            boot["guard"] = bool(guard["installed"])
            boot["guard_armed"] = bool(guard["armed"])
            boot["nested"] = list(guard.get("nested") or [])
            coverage = _guard_coverage(cls)
            boot["not_covered"] = list(coverage["not_covered"])
            if not boot["guard"]:
                boot["refusals"].append("guard: {}".format(
                    guard.get("reason") or "the wrappers were not installed"))
            if coverage["refused"]:
                boot["refusals"].append("guard: no layer on {}".format(
                    ", ".join(coverage["refused"])))

            if boot["backport"] and boot["guard"] and not coverage["refused"]:
                state = set_armed(True)
                boot["tier"] = bool(state.get("armed")) and not state.get("refused")
                if not boot["tier"]:
                    boot["refusals"].append("tier: {}".format(
                        state.get("reason") or "arming was refused"))
            else:
                blocked = ("the install is not verified" if not boot["backport"]
                           else ("the guard is not installed" if not boot["guard"]
                                 else "the guard left entry points unwrapped"))
                boot["refusals"].append(f"tier: not armed, {blocked}")
            boot["faults_at_boot"] = guard_mode_faults()

        # THE SSD RESTORE PATH, SCHEDULER HALF, fitted HERE - the first instant in
        # the one process that can start a restore at which the manager class exists
        # AND our guard/backport layers are already on its methods, so
        # `disk_restore._install_sched_method` can insert the restore layer at the
        # BOTTOM and keep the guard outermost (its own layer walk, no new rule).
        # Nothing else in the EngineCore process ever retried the install after the
        # plugin-load attempt found the manager un-imported: the worker half's retry
        # driver is the worker's handshake hook, which by construction never runs
        # here. That is why four ranks reported `installed=YES` while this process
        # made no offer at all - see `disk_restore.install`'s docstring.
        restore = _boot_restore(cls)
        boot["restore"] = restore
        if not restore.get("fitted"):
            boot["refusals"].append("restore: {}".format(
                (restore.get("reason") or "the scheduler half of the disk restore "
                 "layer is not fitted on this manager: spilled KV can be written "
                 "but never read back")[:MAX_BOOT_ITEM_CHARS]))

        # THE ARENA THE DISK TIER LIVES IN, asked at the one moment the manager is in
        # hand. An armed spill tier over an arena that cannot give an offer a
        # destination slot moves no byte in either direction, and until now nothing
        # detected that combination: the line printed `disk_tier=ARMED` and the offers
        # died at `no_cpu_slots`. The arm field keeps its meaning (it is a true fact)
        # and the arena's own verdict rides beside it, in the line and in `refusals`.
        arena = _boot_arena(manager)
        boot["arena"] = arena
        if arena.get("state") in ("tiny", "no_pool", "unreadable"):
            boot["refusals"].append("disk: {}".format(
                (arena.get("reason") or "the host-RAM arena is unusable")[
                    :MAX_BOOT_ITEM_CHARS]))

        # THE SSD TIER'S BOOT DEFAULT, last, because it is the only boot step that acts
        # rather than reports: with `KVTIER_DISK_BOOT_ENABLE` at its default it arms the
        # spill path and raises this boot's serve posture so the box comes up
        # `serving_as=disk_authority` with no POST. Every input it needs is a verdict
        # this boot already reached above, and every failure leaves the boot exactly as
        # disarmed as an inert one - see `disk_boot.py` for the gates, the rollback and
        # the prohibition on resurrecting a posture from an earlier boot's arm file.
        disk_verdict = _boot_disk_default(boot)
        boot["disk_boot"] = disk_verdict
        if disk_verdict.get("outcome") == "REFUSED":
            # COUNTED AS A REFUSAL only when the disk tier exists and the default could
            # not be honoured: `NO_STORE` (no root) and `OFF` (the operator's own flag)
            # are normal postures, not failures, and putting them in `refusals` would
            # make every engine without an SSD tier look broken to the reader of this
            # list. The one log line is printed either way.
            boot["refusals"].append("disk_boot: {}".format(
                (disk_verdict.get("reason") or "the boot default was not honoured")[
                    :MAX_BOOT_ITEM_CHARS]))

        # THE SPILL JOURNAL, READ HERE RATHER THAN BY THE FIRST REQUEST. The index
        # learns what is on the NVMe tier by replaying the journals, and until now that
        # replay began on the OFFER path: the boot's bounded drain stopped, and the rest
        # was paid per arriving request at 512 KiB a refresh. Measured on the box: a
        # boot at 23:00:43, a request at 23:02 recomputing 129,285 tokens behind `blind:
        # true`, and a replay that finished at 23:12. This is the one moment in the boot
        # where those seconds cost NOTHING - `create_connector` is called from
        # `Scheduler.__init__` (`scheduler.py:164`), so the drain runs before the first
        # step, before the first `bind_gpu_block_pool` and long before any request; and
        # the connector existing is the only precondition the reader has (it needs the
        # spill tier's journal directory from the environment, not the manager).
        # Contained by `disk_index.drain_journal_at_boot()`'s own contract: a drain that
        # raises leaves the boot exactly as un-drained as it was, counted and logged
        # once. It is never a REFUSAL and never joins `refusals`: a slow index costs a
        # recompute, and a boot that could not read a journal is still a boot.
        boot["journal_drain"] = _boot_journal_drain(boot)

        # THE LOOKUP LEDGER, ARMED HERE RATHER THAN BY A POST THAT ARRIVES AFTER THE
        # REQUEST YOU NEEDED IT FOR. The ring joins each `get_num_new_matched_tokens`
        # offer to the `update_state_after_alloc` adopt that followed it, and it records
        # NOTHING while disarmed - so a boot that needs an answer about a request that
        # already happened has no answer. That is exactly the hole the 16:34 request
        # left
        # after a 15-hour suspend: 0.0% reuse, `reads_applied: 0`, and a ledger that had
        # never been armed, which turned one query into two investigations.
        #
        # IT IS AN OBSERVABILITY STEP AND IS ALLOWED TO FAIL. A boot that could not arm
        # the ring boots, counts the refusal and logs ONE line; the verdict is on the
        # boot
        # LINE as `LOOKUP_LEDGER[...]` because a bare absence cannot be told from a
        # value
        # nobody grepped. It never joins `refusals`: the tier's own verdicts describe
        # what
        # the engine will serve, and a missing recorder changes nothing about that. Its
        # own source is pinned - in `tests/test_lookup_ledger_boot_arm.py`, against the
        # AST - to name no serve posture, no offer switch and no arm writer.
        boot["lookup_ledger"] = _boot_lookup_ledger(boot)

        _state["boots"] += 1
        _state["boot"] = boot
        boot["line"] = boot_line(boot)
        _state["boot_line"] = boot["line"]
        # THE ONE LINE THE OPERATOR READS. WARNING, so it survives a quieter log
        # level, and unconditional, so "did this boot come up working?" is
        # answerable from the log without curling a verb or remembering a runbook.
        logger.warning("kvtier-dev-plugin: %s", boot["line"])
        return boot
    except Exception as exc:  # noqa: BLE001 - a boot report never breaks a boot
        snapshot._error(exc, "backport boot")
        boot["refusals"].append(f"boot: {type(exc).__name__}: {exc}"
                                [:MAX_BOOT_ITEM_CHARS])
        _state["boot"] = boot
        boot["line"] = boot_line(boot)
        _state["boot_line"] = boot["line"]
        logger.warning("kvtier-dev-plugin: %s", boot["line"])
        return boot


def guard_mode_faults() -> int:
    """The guard's fault counter, through its public accessor. Never raises."""
    from . import guard_mode  # noqa: PLC0415 - sibling layer, lazy by idiom
    try:
        return int(guard_mode.fault_count())
    except Exception:  # noqa: BLE001 - an unreadable counter reads as 0
        return 0


def _disk_tier_field() -> str:
    """`ARMED` / `DISARMED`: the DISK SPILL tier's own arm, never the RAM tier's.

    A separate field because the two arms are separate facts with separate lives. The
    RAM/host tier's arm is what `set_armed()` puts up at boot (and what
    `kvtier_gate_remote.sh` arms); the disk spill tier's arm is a FILE in the
    diagnostics directory - tmpfs - so a reboot wipes it while the RAM tier comes back
    armed on its own. On 2026-10-05 a boot printed `BOOT[auto] ... tier=ARMED` next to
    a restore refusal reading `the disk tier is disarmed`, and the two lines were read
    as a contradiction. They are two tiers, and now the line says which is which.

    `disk_spill.is_armed()` is the door (not `counters()["armed"]`) because it checks
    the arm's expiry on every read - a lapsed arm must never answer ARMED. It fails
    closed and never raises; the try/except is only for the import and is reported.
    """
    try:
        from . import disk_spill  # noqa: PLC0415 - sibling layer, lazy by idiom
        return "ARMED" if disk_spill.is_armed() else "DISARMED"
    except Exception as exc:  # noqa: BLE001 - a boot line never fails because of a field
        snapshot._error(exc, "backport boot line disk tier")
        return "UNKNOWN"


def _boot_arena(manager) -> dict:
    """THE HOST-RAM ARENA'S verdict, for the boot report. Never raises.

    WHY IT BELONGS IN THE BOOT LINE. The disk tier owns no KV bytes: spill reads them
    out of the arena's pinned CPU tensors and is triggered by the RAM tier's own store
    completions, and restore allocates its destination slots out of the arena's CPU
    block pool. So an arena that is missing or clamps to a slot or two per rank (a
    mis-set `KV_OFFLOAD_SIZE`, a missing `VLLM_USE_SIMPLE_KV_OFFLOAD`, the fork's own
    `max(1, ...)` clamp) leaves the tier STRUCTURALLY INERT while its arm file still
    says ARMED - and `disk_tier=ARMED` on that boot is, literally, true and useless.

    The verdict keeps the boot line's existing contract intact: `disk_tier=` stays the
    spill arm's fact (unchanged field, unchanged meaning), and the arena gets its own
    adjacent field plus a refusal line naming it. Fails CLOSED on its own errors:
    `unreadable`, never a silent `ok`.
    """
    try:
        from . import disk_restore  # noqa: PLC0415 - sibling layer, lazy by idiom
        return disk_restore.arena_verdict(manager)
    except Exception as exc:  # noqa: BLE001 - a boot line never fails over a field
        snapshot._error(exc, "backport boot arena")
        return {"state": "unreadable", "reason":
                f"reading the arena verdict raised {type(exc).__name__}",
                "cpu_slot_reserve": None, "free_blocks": None,
                "pool_total_blocks": None, "blocks_usable_for_an_offer": None}


def _boot_disk_default(boot: dict) -> dict:
    """THE SSD TIER'S BOOT DEFAULT (0.2.33), asked at the one moment all its inputs
    exist. Never raises.

    WHY HERE AND NOT EARLIER: the decision needs the RAM tier's arm verdict, the
    restore layer's fit verdict and the arena's verdict - and all three are only known
    after the install, the guard and the restore fit have run. It re-measures nothing:
    the dict handed to `disk_boot.boot_enable` is this boot's own report, plus the
    journal directory's writability, which is the one fact the arena verdict does not
    cover and the one that decides whether a spilled byte can ever be named again.

    WHY IT MAY NEVER RAISE: this runs on the engine's connector-construction path.
    A disk-tier default that cannot be computed is a disarmed boot with a log line, and
    never a boot that failed - which is the same fail-safe direction the flag itself
    takes. THE PROHIBITION this path must never cross (see `disk_boot`'s docstring and
    `llm-api/AGENTS.md`, 2026-10-03): it may not read a PREVIOUS boot's arm file or any
    persisted posture and honour it. It computes this boot's posture from the
    environment and the boot's own measurements, and nothing else.
    """
    from . import disk_boot  # noqa: PLC0415 - sibling layer, lazy by idiom
    out = {"outcome": disk_boot.OUTCOME_UNDECIDED,
           "reason": "the disk boot default never ran in this process"}
    try:
        restore = boot.get("restore") if isinstance(boot.get("restore"), dict) else {}
        arena = boot.get("arena") if isinstance(boot.get("arena"), dict) else {}
        journal: dict = {}
        try:
            from . import disk_index  # noqa: PLC0415 - sibling layer, lazy by idiom
            journal = disk_index.journal_fs_state() or {}
        except Exception as exc:  # noqa: BLE001 - an unreadable journal is no verdict
            snapshot._error(exc, "backport boot disk default: journal")
            journal = {}
        pre = {"inert": bool(config.boot_inert()),
               "tier_armed": bool(boot.get("tier")),
               "restore_fitted": bool(restore.get("fitted")),
               "arena_state": str(arena.get("state") or "unseen"),
               "arena_reason": arena.get("reason"),
               # None means "nobody could tell", and that is NOT a refusal; only an
               # observed `False` (EACCES on the directory, or a non-directory on the
               # way to it) withholds the boot default.
               "journal_writable": journal.get("writable"),
               "journal_error": journal.get("directory_error")}
        return disk_boot.boot_enable(pre)
    except Exception as exc:  # noqa: BLE001 - a boot default never costs a boot
        snapshot._error(exc, "backport boot disk default")
        out["reason"] = (f"the disk boot default raised {type(exc).__name__}: "
                         f"{exc}"[:MAX_BOOT_ITEM_CHARS])
        return out


def _disk_boot_field(boot: dict) -> str:
    """The boot line's `disk_boot=` token. Never raises; unknown reads as REFUSED.

    Sits AFTER the bare `tier=` in the field order for the same reason `arena=` does:
    the restart verifier in the llm-api repo takes the last `tier=`-shaped grep match,
    and this field's name does not contain `tier=`, so it is invisible to that grep
    while still reading next to `disk_tier=` for a human scanning the line.
    """
    try:
        from . import disk_boot  # noqa: PLC0415 - sibling layer, lazy by idiom
        verdict = boot.get("disk_boot")
        if not isinstance(verdict, dict):
            return disk_boot.OUTCOME_UNDECIDED
        return disk_boot.field_token(verdict.get("outcome"))
    except Exception:  # noqa: BLE001 - a boot line never fails over a field
        return "REFUSED"


def _arena_field(boot: dict) -> str:
    """The boot line's `arena=` token: OK / TINY / NO_POOL / UNSEEN / UNREADABLE."""
    try:
        verdict = boot.get("arena")
        if not isinstance(verdict, dict):
            return "UNSET"
        state = str(verdict.get("state") or "unseen").upper()
        return {"OK": "OK", "TINY": "TINY", "NO_POOL": "NO_POOL",
                "NO_MANAGER": "UNSEEN", "UNREADABLE": "UNREADABLE"}.get(
                    state, "UNKNOWN")
    except Exception:  # noqa: BLE001 - a boot line never fails over a field
        return "UNKNOWN"


def _journal_dir_field() -> str:
    """`disk` / `tmpfs` / `unknown` / `unset` / `unwritable`: the SPILL JOURNAL's home.

    A separate field from `disk_tier=` because it is a different fact about a
    different thing: the arm says whether this boot writes KV to disk, this says
    whether the INDEX that lets it be read back will still exist after a reboot -
    and whether it can exist at all.

    `unwritable` is not a fifth way of saying `unknown`, and the difference is the
    whole point of having a field: `unknown` means the mount table could not tell
    us (look), `unwritable` means the directory is not there and cannot be made, or
    is there and cannot be written (chmod it, or point
    `VLLM_E2_TIER_DISK_JOURNAL_DIR` somewhere writable) - so every spill this boot
    makes commits a KV file and records nothing, and the restore index will come up
    at `entries: 0` over a store full of bytes. Read through
    `disk_index.journal_fs_label()`, which cannot raise; the try/except here only
    covers the import and is reported rather than swallowing the field.
    """
    try:
        from . import disk_index  # noqa: PLC0415 - sibling layer, lazy by idiom
        return disk_index.journal_fs_label()
    except Exception as exc:  # noqa: BLE001 - a boot line never fails because of a field
        snapshot._error(exc, "boot line journal_dir")
        return "unknown"


def boot_line(boot: dict) -> str:
    """THE boot line: one line, fixed field order, bounded.

    `BOOT[mode] backport=YES|NO guard=YES|NO ram_tier=ARMED|DISARMED
    disk_tier=ARMED|DISARMED|UNKNOWN tier=ARMED|DISARMED faults=N ...`

    Refusals and not-covered methods appear only when non-empty and each is
    capped - the full detail is in the payload - but the line says how many were
    left out, so a truncated line can never read as a complete one.
    """
    parts = ["BOOT[{}]".format(boot.get("mode") or "?")]
    # IN_TREE, not YES, when the engine's own file carries the fix: the fix is
    # equally in effect, but a box that says YES is a box still being patched at
    # runtime from a plugin, and that is the difference an operator acting on
    # this line needs to see.
    if boot.get("backport_source") == "in_tree" and boot.get("backport"):
        parts.append("backport=IN_TREE")
    else:
        parts.append("backport={}".format("YES" if boot.get("backport") else "NO"))
    parts.append("guard={}".format("YES" if boot.get("guard") else "NO"))
    # THE TWO TIERS, NAMED APART, and `tier=` UNCHANGED. Two rules here:
    #
    # 1. `tier=` stays, exactly as it was - same position in the field order, same
    #    ARMED/DISARMED values. It is the field the restart verifier in the llm-api
    #    repo (`server/kvtier_restart.sh`) treats as a PASS criterion, and the README
    #    tells an operator to grep it; renaming or dropping it would fail a good boot.
    # 2. `ram_tier=`/`disk_tier=` are printed BEFORE it, not after, because that
    #    verifier pulls the field with `grep -oE "tier=[A-Za-z_]+" | tail -1` - LAST
    #    match. Every new `*tier=` field therefore has to sit upstream of the bare one
    #    or a `disk_tier=DISARMED` would be read as the tier verdict and a healthy boot
    #    would be reported as broken. (It is also why `ram_tier=` carries the SAME
    #    value as `tier=`: they are the same arm, one named ambiguously and once named
    #    plainly, so a grep that lands on either gets the RAM tier's verdict.)
    parts.append("ram_tier={}".format("ARMED" if boot.get("tier") else "DISARMED"))
    parts.append("disk_tier={}".format(_disk_tier_field()))
    parts.append("tier={}".format("ARMED" if boot.get("tier") else "DISARMED"))
    parts.append("faults={}".format(int(boot.get("faults_at_boot") or 0)))
    # WHERE THE SPILL JOURNAL LIVES, in one token: `disk`, `tmpfs`, `unknown`,
    # `unset` or `unwritable`. A journal is the only map from a KV identity to a
    # spilled file, and on the default configuration it still lands in `/tmp` - so
    # a reboot eats the map while the 107 GB of KV files it describes survive,
    # silently, which is exactly what happened on 2026-10-05. The warning is logged
    # once (`disk_index.journal_fs_state`); this is the field that stays greppable
    # afterwards. `unwritable` is the newer half of the same idea and the one that
    # needs no reboot to hurt: the directory named by
    # `VLLM_E2_TIER_DISK_JOURNAL_DIR` could not be created or written, so this boot
    # spills into a store nobody can read back. INFORMATIONAL EITHER WAY: a tmpfs
    # journal degrades the tier and an unwritable one blinds its index, and neither
    # stops a spill - a tier that refused to write because of its INDEX would be a
    # diagnostic causing the outage it measures.
    parts.append("journal_dir={}".format(_journal_dir_field()))
    # THE HOST-RAM ARENA the disk tier has to move bytes through: `OK`, `TINY`,
    # `NO_POOL`, `UNSEEN`, `UNREADABLE` (or `UNSET` for a line written without a
    # boot verdict, like the no-tier note). Placed AFTER the bare `tier=` on purpose:
    # the restart verifier in the llm-api repo takes the LAST `tier=`-shaped token,
    # and this field's name does not contain `tier=`, so it is invisible to that grep -
    # while `arena=` sitting here still reads next to `disk_tier=` in a line a human
    # scans. An ARMED spill tier over a `TINY`/`NO_POOL` arena is the boot that looks
    # healthy and serves nothing, and the reason is in `boot.arena.reason` and in the
    # refusal tail below.
    parts.append("arena={}".format(_arena_field(boot)))
    # THE SSD TIER'S BOOT DEFAULT (0.2.33): ARMED / OFF / NO_STORE / INERT / REFUSED /
    # UNDECIDED. `ARMED` here and `disk_tier=DISARMED` on the same line is a real and
    # readable contradiction, so this field's job is to say WHICH of the two the reader
    # is looking at: `disk_tier=` is the spill arm's live fact at boot, `disk_boot=` is
    # what the boot default decided about it. A REFUSED/INERT/OFF token next to a
    # `disk_tier=DISARMED` is the boot doing exactly what its flag said.
    parts.append("disk_boot={}".format(_disk_boot_field(boot)))
    # THE SECOND RUNTIME-PATCH TARGET'S VERDICT, in one bracketed token: `patched`,
    # `refused`, `env-refused` or `disabled`. It is a TOKEN and not a `key=value`
    # field on purpose (see `mamba_anchor.boot_token`): the restart verifier in the
    # llm-api repo pulls the PASS criterion with `grep -oE "tier=[A-Za-z_]+" |
    # tail -1`, and a token with no `=` can never become a candidate match. Its job
    # is the one-glance question "did this boot fit the run-end anchor patch, or is
    # it silently absent?" - the same reason `backport=` and `restore=` are on this
    # line, and the reason a bare absence is not acceptable: a target that installs
    # nothing has to SAY so.
    parts.append(mamba_anchor.boot_token())
    # THE SPILL JOURNAL'S VERDICT, in the same bracketed-token shape and for the same
    # reason as `MAMBA_ANCHOR[...]` (see `disk_index.boot_drain_token`): a closed
    # vocabulary, no `=`, so the restart verifier's `grep -oE "tier=[A-Za-z_]+" |
    # tail -1` can never land on it. `JOURNAL_DRAIN[converged]` is the answer to "will
    # the first request find the NVMe tier, or recompute it?" - the question tonight's
    # 129,285-token recompute asked and the boot log could not answer.
    parts.append(_journal_drain_token(boot))
    # THE LOOKUP LEDGER'S BOOT ARM (0.2.66), in the same bracketed-token shape and for
    # the
    # same reason as the two above it: `LOOKUP_LEDGER[armed]` answers "will this boot be
    # able to explain a 0-token offer later?" in one glance, and `[env-refused]` /
    # `[refused]` / `[inert]` / `[not_run]` say which kind of no it is. Observability,
    # so
    # it is deliberately NOT part of `refused=` below: a missing recorder is not a tier
    # that declined to serve, and putting it there would make the one list an operator
    # triages on lie about what actually went wrong.
    parts.append(lookup_ledger_token(boot))
    # WHO DECIDES WHAT A FAILED KV LOAD COSTS THE USER (0.2.73), in the same
    # bracketed-token shape and for the same reason as the three above it: `LOAD_POLICY
    # [recompute:plugin]` answers "if a spill file is unreadable, does my user get an
    # answer or an error, and who decided that?" in one glance, and the WHO half is the
    # point - `fail:operator` means a human chose it, `fail:unresolved` means nobody can
    # prove that and the plugin will not guess, and the two have different remedies. A
    # boot that had silently inherited upstream's `fail` used to be indistinguishable
    # from a boot that was told to use it. Reporting, so it is NOT part of `refused=`.
    parts.append(load_failure_policy_token(boot))
    if boot.get("nested"):
        parts.append("guard_over={}".format(
            ",".join(list(boot["nested"])[:MAX_BOOT_ITEMS])))
    refusals = list(boot.get("refusals") or [])
    if refusals:
        parts.append("refused={}".format(" | ".join(
            item[:MAX_BOOT_ITEM_CHARS] for item in refusals[:MAX_BOOT_ITEMS])))
        if len(refusals) > MAX_BOOT_ITEMS:
            parts.append("refused_more={}".format(len(refusals) - MAX_BOOT_ITEMS))
    not_covered = list(boot.get("not_covered") or [])
    if not_covered:
        parts.append("not_covered={}".format(
            ",".join(not_covered[:MAX_BOOT_ITEMS])))
        if len(not_covered) > MAX_BOOT_ITEMS:
            parts.append("not_covered_more={}".format(
                len(not_covered) - MAX_BOOT_ITEMS))
    restore = boot.get("restore")
    if isinstance(restore, dict):
        # THE SSD RESTORE VERDICT, on the line an operator already reads. Before
        # this, a boot whose scheduler could not restore from NVMe said so nowhere:
        # the four workers' `DISK[restore] installed=YES` lines described a role
        # that cannot make an offer, and this line said nothing about the role that
        # can. `restore=NO` here is that reading, in one field.
        parts.append("restore={}".format("YES" if restore.get("fitted") else "NO"))
        parts.append("restore_gate={}".format(restore.get("gate") or "?"))
        parts.append("restore_ranks={}".format(int(restore.get("ranks_proven")
                                                   or 0)))
    if boot.get("guard_armed"):
        # LABEL DISCIPLINE: this tail used to read `faults=POST ...`, so the one line
        # an operator greps at boot carried `faults=0` AND `faults=POST ...` - two
        # labels with the same name, one of them non-numeric. A reader (or a grep that
        # takes the last match) got a pointer where it expected a count. Every label in
        # the boot line must appear exactly once; the test in test_auto_boot.py holds
        # that line.
        parts.append('captures=POST {} {{"op":"captures"}}'.format(
            config.ROUTE_PREFIX + config.ROUTE_GUARD))
    return " ".join(parts)


def boot_section() -> dict:
    """The BOOT verdict for `status`/`snapshot`/`/tier`. The same keys every path.

    `ran: false` is a DISTINCT answer from "the boot failed": no tier connector
    has been built in this process yet, so nothing has been decided.
    """
    boot = _state.get("boot")
    mode, why = boot_mode()
    if not isinstance(boot, dict):
        return {"ran": False, "mode": mode, "mode_why": why, "boots": 0,
                "backport": False, "guard": False, "guard_armed": False,
                "tier": False, "faults_at_boot": 0, "refusals": [],
                "not_covered": [], "nested": [], "live_check": None,
                "install_reason": None, "manager": None, "restore": None,
                "arena": None, "disk_boot": None, "journal_drain": None,
                "lookup_ledger": None, "load_failure_policy": None,
                "line": _state.get("boot_line"),
                "faults_now": guard_mode_faults(),
                "note": ("no tier connector has been built in this process yet, so "
                         "the boot has decided nothing: the connector-creation "
                         "hook is what runs the install, the guard and the arm"),
                "error": None}
    out = dict(boot)
    out["ran"] = True
    out["boots"] = int(_state.get("boots") or 0)
    out["mode"] = boot.get("mode") or mode
    out["mode_why"] = boot.get("mode_why") or why
    out["faults_now"] = guard_mode_faults()
    out["note"] = ("the connector-creation boot: install, then guard, then arm on "
                   "a verified install. faults_now above faults_at_boot means the "
                   "guard has DECLINED something since boot - a silent miss by "
                   "design; read the captures.")
    out["error"] = None
    return out


def _is_tier_connector(connector) -> bool:
    """Is this object the simple-CPU offload connector (or a subclass)?

    Identity by MRO name+module, never `isinstance` on a fork class: this module
    does not import vllm and must not start. A MultiConnector holding a tier
    connector inside it is NOT matched, and says so as `other_connector` - the
    gap is reported, not hidden.
    """
    try:
        for klass in type(connector).__mro__:
            if (klass.__name__ == CONNECTOR_CLASS
                    and getattr(klass, "__module__", "") == CONNECTOR_MODULE):
                return True
    except Exception as exc:  # noqa: BLE001 - an unreadable object is "not ours"
        snapshot._error(exc, "backport connector identity")
    return False


def _boot_journal_drain(boot: dict) -> dict:
    """READ THE SPILL JOURNAL NOW (0.2.65). The boot's journal step; never raises.

    WHY HERE. The connector exists, which is all the journal reader needs (its directory
    is the spill tier's, resolved from the environment, not the manager), and the caller
    is `KVConnectorFactory.create_connector` - inside `Scheduler.__init__`, so nothing
    is being served yet. Everything the drain builds is in-process memory, and it
    reaches `disk_index` and NOTHING else: it cannot arm, cannot offer, cannot pick a
    serve posture and cannot refuse a spill. The AST of this function is pinned against
    exactly that claim by `tests/test_boot_journal_drain.py`.

    INERT MEANS INERT. `VLLM_E2_TIER_DEV_BOOT_INERT` is the operator's "install and stop
    there", and the drain costs boot seconds, so an inert boot reads no journal byte and
    says so (`stopped_at: "inert"`) rather than quietly doing work the opt-out refused.
    """
    del boot  # this step needs no other boot verdict: it reads no boot state
    from . import disk_index  # noqa: PLC0415 - sibling layer, lazy by idiom
    if boot_mode()[0] == BOOT_MODE_INERT:
        record = disk_index.boot_drain_view()
        record["stopped_at"] = "inert"
        record["note"] = (f"{config.ENV_BOOT_INERT} is set: this boot read no journal "
                          "byte. The first offer pays for the drain, as it did before "
                          "0.2.65")
        return record
    return disk_index.drain_journal_at_boot()


def _journal_drain_token(boot: dict) -> str:
    """The BOOT line's `JOURNAL_DRAIN[...]` token, from this boot's own record.

    Read from the BOOT DICT rather than module state (the `arena=` / `disk_boot=`
    discipline, not `MAMBA_ANCHOR`'s): the line must describe the boot that produced it,
    and a boot that never reached the step renders `JOURNAL_DRAIN[not_run]` instead of
    whatever a previous boot left in the index.
    """
    try:
        from . import disk_index  # noqa: PLC0415 - sibling layer, lazy by idiom
        return disk_index.boot_drain_token(boot.get("journal_drain")
                                           if isinstance(boot, dict) else None)
    except Exception as exc:  # noqa: BLE001 - a token is never worth a lost line
        snapshot._error(exc, "backport journal drain token")
        return "JOURNAL_DRAIN[not_run]"


def _install_shutdown_flush(connector) -> dict:
    """Chain the bounded spill-queue drain onto one connector object. Never raises.

    The work is `disk_spill.install_shutdown_flush`'s (it owns the queue, the budget and
    the fail-open guard); this wrapper exists so the connector hook cannot be broken by
    a
    layer it installed, and so the verdict is counted where the other hook outcomes are.
    A budget of 0 installs NOTHING at all - the rollback has to be structural, not a
    wrapper that runs and does nothing.
    """
    from . import disk_spill  # noqa: PLC0415 - sibling layer, lazy by idiom
    try:
        report = disk_spill.install_shutdown_flush(connector)
    except Exception as exc:  # noqa: BLE001 - a drain install never costs a connector
        snapshot._error(exc, "backport shutdown flush install")
        report = {"installed": False, "reason": f"the install raised "
                                                f"{type(exc).__name__}",
                  "reason_kind": "error"}
    _state["shutdown_flush"] = report
    return report


def shutdown_flush_state() -> dict:
    """WHAT THIS PROCESS INSTALLED for the shutdown drain, and what it would cost.

    Published through the tier's own payload so the answer to "will a stop wait for the
    spill queue, and for how long" is readable from a route rather than from the source.
    Never raises: an unreadable install reports "not installed", which is the honest
    pessimistic reading for a caller deciding whether to wait for this process itself.
    """
    from . import disk_spill  # noqa: PLC0415 - sibling layer, lazy by idiom
    try:
        state = disk_spill.shutdown_flush_state()
        state["install_report"] = _state.get("shutdown_flush")
        return state
    except Exception as exc:  # noqa: BLE001 - a payload section is never a fault
        snapshot._error(exc, "backport shutdown flush state")
        return {"installed": False, "error": type(exc).__name__,
                "note": "the drain's state could not be read"}


def _boot_lookup_ledger(boot: dict) -> dict:
    """ARM THE LOOKUP/ADOPT RING AT BOOT (0.2.66). Never raises, never a refusal.

    All the decision lives in `lookup_ledger.boot_arm()` - the gate, the inert posture,
    the guard-wrapper dependency and the fail-open guard. This function's job is the one
    thing only the boot knows: whether THIS boot put wrappers on the manager, since the
    ring is fed from inside those wrappers and arming it without them would publish
    `armed: true` over a ring that can never hold a record.

    IT CANNOT JOIN `refusals`, and the containment below is written so that it cannot:
    whatever happens here, `boot["tier"]`, `boot["guard"]`, `boot["backport"]`,
    `boot["arena"]`, the disk tier's own boot verdict and the journal drain's record are
    exactly what the same boot decided without this step. A diagnostic that changes the
    boot it observes is worse than no diagnostic, and
    `tests/test_lookup_ledger_boot_arm.py` compares those decisions field by field
    against a baseline boot to prove it.
    """
    from . import lookup_ledger  # noqa: PLC0415 - sibling layer, lazy by idiom
    try:
        return lookup_ledger.boot_arm(
            guard_wrappers_installed=bool(boot.get("guard")),
            inert=boot.get("mode") == BOOT_MODE_INERT)
    except Exception as exc:  # noqa: BLE001 - an observability arm never costs a boot
        snapshot._error(exc, "backport lookup ledger boot arm")
        try:
            verdict = lookup_ledger.VERDICT_REFUSED
            cap = int(lookup_ledger.counters().get("cap") or 0)
        except Exception:  # noqa: BLE001 - the fallback cannot be allowed to fail
            verdict, cap = "refused", 0
        return {"armed": False, "verdict": verdict, "cap": cap, "recognised": True,
                "raw": None, "problems": [], "reason":
                f"the boot step raised {type(exc).__name__}"[:200]}


def lookup_ledger_token(boot: dict) -> str:
    """The BOOT line's `LOOKUP_LEDGER[...]` token, from THIS boot's own record.

    A bracketed token and never `key=value`, for the reason `mamba_anchor.boot_token`
    gives: the restart verifier in the llm-api repo pulls its PASS criterion with
    `grep -oE "tier=[A-Za-z_]+" | tail -1`, and a token containing an `=` would become a
    candidate match for a grep that was never meant to see it.

Read from the BOOT DICT (the `JOURNAL_DRAIN` discipline), so a line describes the boot
that produced it rather than whatever the last boot in this process left in module
state. A boot that never reached the step renders `[not_run]`; a record whose verdict is
outside the closed vocabulary renders `[refused]`, because the honest pessimistic
reading of a verdict this module cannot name is "it did not work".
    """
    try:
        from . import lookup_ledger  # noqa: PLC0415 - sibling layer, lazy by idiom
        record = boot.get("lookup_ledger") if isinstance(boot, dict) else None
        if not isinstance(record, dict):
            return f"LOOKUP_LEDGER[{lookup_ledger.VERDICT_NOT_RUN}]"
        verdict = record.get("verdict")
        if verdict not in lookup_ledger.BOOT_VERDICTS:
            verdict = lookup_ledger.VERDICT_REFUSED
        return f"LOOKUP_LEDGER[{verdict}]"
    except Exception as exc:  # noqa: BLE001 - a token is never worth a lost line
        snapshot._error(exc, "backport lookup ledger token")
        return "LOOKUP_LEDGER[not_run]"


def load_failure_policy_token(boot: dict) -> str:
    """The BOOT line's `LOAD_POLICY[what:who]` token, from THIS boot's own record.

    Bracketed, not `key=value`, and read from the BOOT DICT rather than module state -
    both for the reasons `mamba_anchor.boot_token()` and `lookup_ledger_token()` give:
    the restart verifier in the llm-api repo pulls its PASS criterion with
    `grep -oE "tier=[A-Za-z_]+" | tail -1`, and a line must describe the boot that
    produced it. The token renders the two facts the upstream finding asked for and
    nothing else: the policy the engine will actually run, and who decided it.
    `not_run` is the honest rendering of "this boot never reached the step", and a
    record this module cannot name renders `refused`, never the plugin's own hope.
    """
    try:
        from . import load_failure_policy  # noqa: PLC0415 - sibling layer, lazy
        return load_failure_policy.boot_token(
            boot.get("load_failure_policy") if isinstance(boot, dict) else None)
    except Exception as exc:  # noqa: BLE001 - a token is never worth a lost line
        snapshot._error(exc, "backport load failure policy token")
        return "LOAD_POLICY[not_run]"


def boot_note(reason: str) -> dict:
    """The BOOT line for a process that has NO tier to boot. Same format.

    A worker (no `scheduler_manager`) and a connector nested inside a
    `MultiConnector` are both normal, non-fatal answers - but an operator who
    greps for `BOOT` and finds nothing cannot tell "the plugin decided not to
    arm" from "the plugin never ran". So they report the same line shape, with
    the reason, and the payload's `boot.ran` stays false.
    """
    mode, why = boot_mode()
    boot = {"mode": mode, "mode_why": why, "backport": False, "guard": False,
            "guard_armed": False, "tier": False, "faults_at_boot": 0,
            "refusals": [reason], "not_covered": [], "nested": [],
            "live_check": None, "manager": None, "install_reason": None,
            "restore": None, "arena": None, "disk_boot": None,
            "journal_drain": None, "lookup_ledger": None,
            "load_failure_policy": None, "line": None}
    boot["line"] = boot_line(boot)
    _state["boot_line"] = boot["line"]
    logger.warning("kvtier-dev-plugin: %s", boot["line"])
    return boot


def _after_connector_created(connector, vllm_config=None) -> None:
    """The install moment. Contained: the engine ALWAYS gets its connector.

    Nothing here may raise into `create_connector`, and nothing here may swap the
    object the caller asked for: a diagnostic that can break connector
    construction is a worse outage than an unpatched tier.

    `vllm_config` is the FIRST ARGUMENT the fork's `create_connector` was called with
    (`scheduler.py:140`: `config=self.vllm_config`), passed through untouched. It is
    the only handle on `kv_transfer_config` this hook will ever have, and the
    load-failure-policy step below has to write that field before the caller reads it
    at `scheduler.py:147`. A caller that hands in nothing (an older test, a foreign
    caller) leaves it None and the step reports `no-config` rather than guessing.
    """
    try:
        _state["hook_calls"] += 1
        if not _is_tier_connector(connector):
            # Some other connector (or a MultiConnector wrapping one). Not ours,
            # not an error, and NOT silent: it is counted and published.
            _state["hook_other_connector"] += 1
            _state["hook_last"] = (f"{type(connector).__name__}: not the tier "
                                   f"connector, nothing installed")
            boot_note(f"tier: not armed, {type(connector).__name__} is not the "
                      f"tier connector (a tier nested inside it is the known "
                      f"gap: see disarm_limits)")
            return
        # THE SPILL QUEUE'S SHUTDOWN DRAIN, chained onto THIS connector's own
        # `shutdown`, before anything else in this hook can divert the path. WHY HERE:
        # this is the one moment this plugin is handed the connector OBJECT, and the
        # worker role (the `manager is None` branch below) otherwise returns without
        # this process ever seeing the object again - while the worker is exactly where
        # the spill queue lives, because `request_spill` runs inside the worker's
        # `get_finished`. A drain registered only with `atexit` would never run there: a
        # worker ends its orderly shutdown in
        # `multiproc_executor.py:_exit_worker_process` with `os._exit`, which skips
        # `atexit`, every pending `finally` and interpreter finalization. The
        # connector's `shutdown` is reached BEFORE that, by `Worker.shutdown()` ->
        # `ensure_kv_transfer_shutdown()`, so wrapping the instance is the only seam in
        # the right process at the right moment.  THE SCHEDULER ROLE GETS IT TOO, by the
        # same line and for the same reason: its queue is its own, its budget is the
        # same environment variable, and a single process boot is covered by the same
        # code rather than by a second rule. Contained: an install that raises costs the
        # drain and nothing else, and the engine always gets its connector.
        _install_shutdown_flush(connector)
        manager = getattr(connector, "scheduler_manager", None)
        if manager is None:
            # THE WORKER CASE. `role=WORKER` builds `worker_handler`, never a
            # `scheduler_manager`, and the worker process runs no manager code at
            # all - so there is nothing to patch HERE and that is the NORMAL
            # answer, not a failure. The scheduler role gets its own call in the
            # EngineCore process, which is where the manager actually runs.
            _state["hook_no_manager"] += 1
            _state["hook_last"] = ("role=WORKER: no scheduler_manager in this "
                                   "process (no_manager), nothing to install")
            logger.info("kvtier-dev-plugin: connector hook saw a tier connector "
                        "with no scheduler_manager (worker role) - normal, "
                        "nothing installed")
            boot_note("tier: not armed, this process builds no scheduler_manager "
                      "(worker role: it runs no tier scheduler code)")
            return
        _state["hook_tier_connectors"] += 1
        cls = type(manager)
        # THE ENGINE'S KV LOAD-FAILURE POLICY, decided HERE and BEFORE EVERYTHING ELSE
        # IN THIS BRANCH, because this is the last moment it is still a choice.
        # `Scheduler.__init__` calls `create_connector` at `scheduler.py:140` and reads
        # `kv_transfer_config.kv_load_failure_policy` EIGHT LINES LATER at `:147`,
        # latching `recompute_kv_load_failures` for the life of the process; the
        # upstream default is `fail`, which fails a request outright on a KV load that
        # cannot be served (`scheduler.py:1928`, and `should_fail` in
        # `_handle_invalid_blocks`). With the SSD tier under this connector, a capacity
        # eviction or an unreadable spill file would therefore be a user-visible error
        # instead of a recompute - and until now one launcher variable was all that
        # stood between the two, with nothing in the boot saying which one this was.
        #
        # WHY FIRST, AHEAD OF THE INSTALL: the policy is a fact about the engine's
        # error handling and does not depend on our patch landing, so an install that
        # fails must not also cost the boot the policy. WHY ONLY HERE AND NOT IN
        # `register()`: the worker role returns above, and a worker process never reads
        # the field, so writing it there would be a side effect nothing asked for.
        # WHY IT CANNOT OVERRIDE AN OPERATOR: see `load_failure_policy.py` - the write
        # needs a witness that nobody named the field, and the boot line prints what
        # the engine will do AND who decided it. Contained twice over: the step never
        # raises, and this whole hook is inside a try.
        from . import load_failure_policy  # noqa: PLC0415 - lazy by idiom
        policy_verdict = load_failure_policy.boot_step(vllm_config)
        report = install(cls, getattr(manager, "kv_cache_config", None),
                         manager)
        wrapped = sum(1 for v in report["switch"].values() if v.get("wrapped"))
        _state["hook_installs"] += 1
        _state["hook_last"] = (f"{getattr(cls, '__name__', '?')}: "
                               f"installed={report.get('installed')} "
                               f"switch={wrapped}/{len(SWITCH_METHODS)} "
                               f"reason={report.get('reason') or 'none'}")
        # LOUD, because this is the state an operator must be able to read from
        # the boot log without curling anything: patched before the first request,
        # or refused. These two lines describe the INSTALL verdict only - what
        # happened to the tier is the BOOT line's job, written next.
        if report.get("installed"):
            logger.warning("kvtier-dev-plugin: PR #56810 install run at CONNECTOR "
                           "CREATION (before the first request): %s",
                           _state["hook_last"])
        else:
            logger.warning("kvtier-dev-plugin: connector-creation install was NOT "
                           "completed (%s) - the tier keeps its kill-switch "
                           "wrappers fitted, so BOTH routes to the divisibility "
                           "assert are gated and arming is refused",
                           _state["hook_last"])
        # THE SECOND RUNTIME-PATCH TARGET, fitted at the SAME moment and for the
        # SAME reason: this is the first instant the EngineCore process provably has
        # the tier's manager, and `vllm/v1/engine/core.py:60` has already imported
        # `single_type_kv_cache_manager`, so `MambaManager` is resolvable here
        # without this plugin importing it. It runs BEFORE `boot_from_connector` so
        # the BOOT line can carry `MAMBA_ANCHOR[...]`: an operator must be able to
        # tell "fitted" from "silently absent" in one glance.
        #
        # WHY HERE AND NOT IN `register()`: the mask decides which align-mode state
        # blocks survive into the store, so it only has a purpose on a boot that has
        # a tier at all - and `register()` runs in worker and arg-parsing processes
        # too, where patching a scheduler-side cache class would be a side effect
        # nothing asked for. A boot with no tier connector gets no anchor patch, and
        # says so with `MAMBA_ANCHOR[disabled]`.
        mamba = mamba_anchor.install()
        logger.warning(
            "kvtier-dev-plugin: run-end anchor patch at CONNECTOR CREATION: %s%s",
            mamba.get("verdict"),
            "" if mamba.get("installed")
            else ": %s" % str(mamba.get("reason") or "no reason given"))
        # THE BOOT SEQUENCE: guard wrappers (armed, composed over the layers just
        # fitted), then the tier arm - and the arm only if the install verified.
        # After the install verdict lines above, so the BOOT line is the last
        # thing the boot writes and the one line an operator reads. Contained:
        # the engine still got its connector whatever happens here.
        boot_from_connector(cls, report, manager,
                            load_failure_policy=policy_verdict)
    except Exception as exc:  # noqa: BLE001 - a hook must never break a connector
        _state["hook_errors"] += 1
        _state["hook_last"] = f"hook raised {type(exc).__name__}: {exc}"[:200]
        snapshot._error(exc, "backport connector hook")


def _make_create_connector_hook(original):
    """Pass every argument through untouched, install on the way out."""
    def create_connector(cls, *args, **kwargs):
        # The fork's own call FIRST, and its exceptions propagate verbatim (the
        # HMA/`kv_transfer_config` ValueErrors it raises are the engine's story,
        # not ours to swallow). Only a SUCCESSFUL construction gets inspected.
        connector = original(cls, *args, **kwargs)
        # THE VLLM CONFIG, picked out of the arguments the fork was handed and passed
        # along as a second, purely informational parameter. Both call shapes are read
        # because the factory's signature is positional while `scheduler.py:140` calls
        # it with `config=` as a keyword: a hook that understood only one would answer
        # "no config" about a boot that had one.
        config_arg = kwargs.get("config")
        if config_arg is None and args:
            config_arg = args[0]
        # THE SSM BOOT GUARD: the ONE place this plugin raises into the engine on
        # purpose, and deliberately HERE rather than inside `_after_connector_created`.
        # That hook runs inside its own `try/except Exception` ("nothing here may raise
        # into `create_connector`"), so a refusal raised in there is swallowed and the
        # guard would be theatre - a gate that cannot fail. Out here the ValueError
        # reaches the operator: `Scheduler.__init__` and `EngineCore.__init__` have no
        # try around this call, and the background EngineCore's `run_engine_core`
        # `except Exception` logs `EngineCore failed to start.` and RE-RAISES. Read
        # `ssm_guard.py` for the semantics (truthy = refuse, ABSENT = not SSM, an
        # unreadable flag = warn only) and for why this boot must die instead of
        # degrading: a spilled Mamba boundary state that predates the RecoverSSM commit
        # would be served back as a correct restore, and a wrong KV claim returns
        # plausible output rather than an error.
        #
        # THE PREDICATE IS OUR OWN CONNECTOR, not every connector: with a foreign one
        # the tier never arms (`_is_tier_connector` -> `hook_other_connector`), so it
        # spills nothing and there is no wrong claim to prevent - refusing somebody
        # else's boot would be this plugin over-reaching past its own tier.
        if _is_tier_connector(connector):
            from . import ssm_guard  # noqa: PLC0415 - lazy by idiom
            ssm_guard.boot_check(config_arg)
        _after_connector_created(connector, config_arg)
        return connector

    setattr(create_connector, HOOK_ATTR, HOOK_TAG)
    setattr(create_connector, HOOK_ORIGINAL, original)
    create_connector.__name__ = CREATE_CONNECTOR
    create_connector.__qualname__ = f"{CONNECTOR_FACTORY_CLASS}.{CREATE_CONNECTOR}"
    create_connector.__doc__ = (
        "kvtier-dev-plugin: the fork's own create_connector, unchanged, then the "
        "PR #56810 install against the manager class the connector just imported "
        "- the first moment that class exists. Contained: a hook failure cannot "
        "stop the connector being built. The ONE exception is `ssm_guard` above, "
        "which refuses an SSM-recovery boot by raising on purpose, outside the "
        "contained region - see its module docstring.")
    return create_connector


def install_connector_hook() -> dict:
    """Wrap `KVConnectorFactory.create_connector`. Idempotent, never raises.

    Same ownership discipline as `_install_switch`: ours is re-fitted by
    re-binding the fork's original (so a reload loop cannot nest hooks), a
    foreign marker is refused, and anything that is not the class's OWN
    `classmethod` is refused rather than guessed at - the fork calls it as
    `KVConnectorFactory.create_connector(config=..., role=..., kv_cache_config=...)`
    and a wrapper that got that convention wrong would break every boot that has
    a connector, tier or not.
    """
    report = {"fitted": False, "class": None, "reason": None}
    try:
        factory = connector_factory_class()
        if factory is None:
            # A process that has not imported the factory yet (an arg-parsing or
            # worker process at plugin-load time). Nothing to wrap now;
            # `ensure_installed` re-fits from the first verb contact.
            report["reason"] = (f"{CONNECTOR_FACTORY_MODULE} is not imported in "
                                f"this process, nothing to wrap")
            _note_hook_fit(report)
            return report
        report["class"] = getattr(factory, "__name__", type(factory).__name__)
        raw = factory.__dict__.get(CREATE_CONNECTOR)
        if raw is None:
            report["reason"] = (f"{report['class']}.{CREATE_CONNECTOR} is not "
                                "defined on this class: the fork moved it, "
                                "refusing to re-bind it here")
            _note_hook_fit(report)
            return report
        if not isinstance(raw, classmethod):
            report["reason"] = (f"{report['class']}.{CREATE_CONNECTOR} is a "
                                f"{type(raw).__name__}, not a classmethod: the "
                                "calling convention changed, refusing to guess")
            _note_hook_fit(report)
            return report
        inner = raw.__func__
        owner = getattr(inner, HOOK_ATTR, None)
        if owner is not None and owner != HOOK_TAG:
            held = getattr(inner, "__qualname__", type(inner).__name__)
            report["reason"] = (f"{report['class']}.{CREATE_CONNECTOR} carries "
                                f"another plugin's connector-hook marker "
                                f"({held!r}), refusing to overwrite it")
            _note_hook_fit(report)
            return report
        # Re-fitting our own hook re-binds the FORK's function, so a reload loop
        # is one wrapper deep forever.
        original = getattr(inner, HOOK_ORIGINAL, None) or inner
        setattr(factory, CREATE_CONNECTOR,
                classmethod(_make_create_connector_hook(original)))
        first = not _state["hook_fitted"]
        _state["hook_fitted"] = True
        _state["hook_fits"] += 1
        _state["hook_reason"] = None
        report["fitted"] = True
        if first:
            logger.warning("kvtier-dev-plugin: connector-creation hook fitted on "
                           "%s.%s - the PR #56810 install now lands when the "
                           "tier connector is built, before its first request",
                           report["class"], CREATE_CONNECTOR)
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        _state["hook_errors"] += 1
        report["reason"] = f"install_connector_hook raised: {type(exc).__name__}"[:200]
        snapshot._error(exc, "backport connector hook fit")
        _note_hook_fit(report)
        return report


def _note_hook_fit(report) -> None:
    """One log line per distinct refusal reason (the snapshot polls this path)."""
    reason = report.get("reason")
    if _state["hook_reason"] == reason:
        return
    _state["hook_reason"] = reason
    if reason is not None:
        logger.info("kvtier-dev-plugin: connector-creation hook NOT fitted: %s",
                    reason)


def connector_hook_section() -> dict:
    """What the hook is and did, for `status`/`snapshot`."""
    return {
        "fitted": bool(_state["hook_fitted"]),
        "class": f"{CONNECTOR_FACTORY_MODULE}.{CONNECTOR_FACTORY_CLASS}",
        "method": CREATE_CONNECTOR,
        "reason": _state["hook_reason"],
        "fits": int(_state["hook_fits"]),
        "calls": int(_state["hook_calls"]),
        "tier_connectors": int(_state["hook_tier_connectors"]),
        "worker_role_no_manager": int(_state["hook_no_manager"]),
        "other_connector": int(_state["hook_other_connector"]),
        "installs_run": int(_state["hook_installs"]),
        "errors": int(_state["hook_errors"]),
        "last": _state["hook_last"],
        "lands": ("when create_connector returns a SimpleCPUOffloadConnector "
                  "built with a scheduler_manager (core.py:217 plugin load -> "
                  "scheduler.py:164 connector creation -> core.py:260 scheduler "
                  "-> first request)"),
        "order": ("install() fits the kill-switch wrappers FIRST, then the "
                  "backport, so a refused backport still leaves both routes to "
                  "the selector gated"),
    }



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

    Three independent gates, checked in this order, each with its own `kind` so
    an operator never has to guess which one they hit:

    * the GEOMETRY guard, when it has actually checked a live config and said no
      -- the patch would skip a prefix-cacheable group (silent output loss).
    * NO MANAGER in this process -- there is no class to install onto and no
      connector to arm, so an "arm" would be a flag nobody reads. Reporting it
      as `incomplete_install` would blame a fix that was never refused.
    * an INCOMPLETE install -- not every method in METHODS is patched, so the
      store path would still reach the upstream assert.

    A geometry guard that never checked (a class-install before any config
    exists) is NOT a block: the verbs patch the class first, and refusing to arm
    there would strand a healthy engine. A veto that WAS given is never
    forgotten - see the guard-preservation note in `install()`.
    """
    geometry = _state["geometry"]
    if geometry and geometry.get("checked") and not geometry.get("ok"):
        return {"kind": "geometry",
                "reason": "the geometry guard refuses this config, so arming "
                          "would let the patch skip a prefix-cacheable group: "
                          f"{geometry.get('reason')}"}
    if not _state["fully_installed"]:
        refused = list(_state['refused_methods'])
        attempted = bool(refused) or bool(_state["method_verdicts"])
        live = _state.get("live_check") or {}
        live_failed = bool(live.get("checked")) and not live.get("ok")
        if not attempted and not live_failed and manager_class() is None:
            return {"kind": "no_manager",
                    "reason": f"no {TARGET_CLASS} exists in this process "
                              f"({TARGET_MODULE} is not imported) and nothing has "
                              "been installed yet: there is no tier connector to "
                              "arm. Either this engine booted without the tier, "
                              "or the engine is not built yet. Arming here would "
                              "set a flag nothing reads."}
        names = ", ".join(refused) if refused else "(none attempted)"
        if live_failed:
            # The methods may all be patched on the class and STILL be an
            # incomplete fix: the patch lives on the class AND the instance,
            # and the instance is the one the engine runs. Same kind, because
            # "not fully in place" is exactly what it means.
            names = f"{names}; live-object check: {live.get('reason')}"
        return {"kind": "incomplete_install",
                "reason": f"the PR #56810 fix is not fully installed; refused "
                          f"method(s): {names}. The tier stays disarmed until "
                          f"every required method is patched."}
    return None


def _arm_proof() -> None:
    """Seek the proof an arm requires, instead of trusting the caller to have
    obtained it. Contained; never raises.

    `tier_for_engine` already calls `ensure_installed(engine)` for EVERY op, so
    the operator path has always had its proof (against the LIVE manager, with
    its config) before this runs. This is the backstop for any caller that arms
    without one - a future verb, a direct import, a test - and it can only
    install onto the CLASS, which is why the live-config geometry check still
    belongs to `ensure_installed`.

    Skipped when the guard has already vetoed: install onto a class without a
    config cannot un-veto a config, and trying would only churn the state.
    """
    if _state["fully_installed"]:
        return
    geometry = _state["geometry"]
    if geometry and geometry.get("checked") and not geometry.get("ok"):
        return
    live = _state.get("live_check") or {}
    if live.get("checked") and not live.get("ok"):
        # Same rule as the geometry veto: a class-only install cannot make the
        # live instance safe, so seeking proof here could only paper over a
        # refusal that a manager-carrying install already proved.
        return
    cls = manager_class()
    if cls is None:
        return
    try:
        install(cls)
    except Exception as exc:  # noqa: BLE001 - a proof we cannot get is a refusal
        _state["fallbacks"] += 1
        snapshot._error(exc, "backport arm proof")


def set_armed(armed: bool) -> dict:
    """Arm/disarm the tier at runtime. Contained; never raises.

    DISARM is unconditional -- it is the safe direction and must always work.
    ARM is GATED: it first seeks its own proof (`_arm_proof`) and is then refused
    (state unchanged, nothing raised) while the geometry guard has vetoed the
    live config, no manager exists in this process, or the fix is not fully
    installed, because the invariant is that the tier is never armed unless the
    divisibility fix is provably, fully in place. A refused arm says WHY, names
    its `kind` and the refused methods, and shows up in `status`/`snapshot` and
    the `/tier` verb so an operator reads "not armed, because X" instead of a
    silent no-op.

    The boot default is DISARMED (`BOOT_ARMED`), so this is the only thing that
    makes a tier live - nothing else calls it with True at startup.
    """
    armed = bool(armed)
    if armed:
        _arm_proof()
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

    Everything here follows from the switch operating on the SCHEDULER role,
    through the manager, at the scheduler's own call boundaries (a step for the
    per-step trio, a finished request for the finish path):

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
    * The switch is IMPLEMENTED BY WRAPPERS, so it only exists once an install
      has fitted them. `install()` fits them FIRST (before the geometry guard and
      before the backport), and the connector-creation hook runs that install the
      moment the tier connector is built - so on a boot that reaches
      `create_connector` with a `scheduler_manager`, `armed: false` is a real gate
      from before the first request. What that does NOT cover: a process where the
      tier connector is never built by the factory (a connector nested inside a
      `MultiConnector`, reported as `connector_hook.other_connector`), and a
      worker process, which builds no scheduler-side manager at all and runs none
      of these methods. `connector_hook` in the payload says which happened.
    * The connector's FINISH path is now GATED: `request_finished_all_groups` ->
      `_queue_finished_eager_store` (`manager.py:1166`) was the second caller of
      `_select_eager_blocks_to_store` (`manager.py:1177`) and is wrapped like the
      per-step trio, so disarmed it queues nothing. Note WHY it mattered: the
      scheduler calls `update_state_after_alloc` for every scheduled request
      (`scheduler.py:1228`, unconditionally on `self.connector`), and that method
      registers `_reqs_to_store` BEFORE its `num_external_tokens == 0` early
      return - so while disarmed the finish path DID reach the selector, on every
      finished request, whether or not the tier ever stored anything.
    * Gating the finish path is not the same as cancelling a store the engine has
      already committed to: the queued `_pending_finished_stores` entries that
      exist at the instant of disarm stay queued and are drained by the next
      armed `build_connector_meta`.
    """
    return {
        "worker_process_patched": False,
        "worker_noop_via": "empty SimpleCPUOffloadMetadata (worker.py:266,289)",
        "in_flight_transfers_undone": False,
        "parked_blocks_freed": False,
        "has_pending_block_frees": "not overridden by this connector (base "
                                   "returns False)",
        "enforced_by_wrappers_fitted_at": "install(), which fits the wrappers "
                                          "FIRST, run from the connector-creation "
                                          "hook (backport.connector_hook) as soon "
                                          "as the tier connector is built - and "
                                          "from install()/ensure_installed() on "
                                          "any verb contact. NOT fitted only where "
                                          "the factory never builds a tier "
                                          "connector in this process",
        "finished_request_store_path": "GATED: request_finished_all_groups -> "
                                       "_queue_finished_eager_store "
                                       "(manager.py:1166) -> "
                                       "_select_eager_blocks_to_store "
                                       "(manager.py:1177) is wrapped, so a "
                                       "disarmed tier queues nothing and reaches "
                                       "the selector by NEITHER route. The gate "
                                       "sits on the inner method because the "
                                       "outer one returns the (bool, dict|None) "
                                       "that drives delayed freeing, while this "
                                       "one returns None either way",
        "queued_before_disarm": "a _pending_finished_stores entry already queued "
                                "at the instant of disarm stays queued and is "
                                "drained by the next armed build_connector_meta",
    }


# --- the engine-side verb ---------------------------------------------------

TIER_OPS = ("status", "arm", "disarm")


def tier_route() -> str:
    """The route an operator posts to arm the tier, for notes and next-steps."""
    return config.ROUTE_PREFIX + config.ROUTE_TIER


def _operator_note(armed: bool, installed: bool) -> str:
    """The one-line answer to "the engine booted with tier env set - why is it
    parking nothing?".

    Inert-at-boot is still the posture of a boot that could not verify, and the
    confusion it causes is the price of not crashing on the first request; the
    JSON pays that price here, so an operator reading `pr56810` or a `/tier`
    payload never has to know the design to read the state correctly. With the
    auto boot the DISARMED branch is now the FAILURE case, not the expected one,
    and the note says so and points at the one log line that explains it.
    """
    route = tier_route()
    if armed:
        return ("ARMED: the tier parks and restores KV. In the default boot mode "
                "this is what a verified install produces on its own; with "
                f"{config.ENV_BOOT_INERT} set it is the result of an explicit "
                f'POST {route} {{"op":"arm"}}. Either way the BOOT log line names '
                "the mode that ran.")
    if installed:
        return ("DISARMED although the fix is fully installed, so this is NOT the "
                "expected result of an auto boot: either "
                f"{config.ENV_BOOT_INERT} is set (inert boot, by the operator's "
                "order) or the guard did not wrap what the class has. Read the "
                f"BOOT line (`boot.line`, `boot.refusals`); arming is allowed "
                f"right now: POST {route} {{\"op\":\"arm\"}}.")
    return ("DISARMED and NOT fully installed, so arming is refused until every "
            "required method is patched (see refused_methods / arm_block / "
            "boot.refusals). With the fix incomplete, parking nothing is the "
            "safe state, not a bug - but it is no longer the expected boot "
            "outcome, and the BOOT line says which install failed and why.")


def _next_step(refusal: dict | None) -> str:
    """What to DO about the current state, keyed on the refusal kind."""
    kind = (refusal or {}).get("kind")
    if kind == "geometry":
        return ("do NOT arm this config: the patch would skip a prefix-cacheable "
                "group and lose its KV on resume. Boot without the tier, or fix "
                "the geometry.")
    if kind == "no_manager":
        return ("nothing to arm in this process: the tier connector is not "
                "configured (or the engine is not built yet). Boot it with "
                "KV_OFFLOAD_SIZE / VLLM_USE_SIMPLE_KV_OFFLOAD, then POST "
                + tier_route() + ' {"op":"status"}.')
    if kind == "incomplete_install":
        return ("fix the install first - the refused methods are named in "
                "install.refused_methods (a fingerprint mismatch means the fork "
                "moved that function); then POST " + tier_route()
                + ' {"op":"arm"} again.')
    return ("arming is allowed now: POST " + tier_route() + ' {"op":"arm"}, '
            "confirm armed:true, then send traffic.")



def tier_for_engine(engine, args=None) -> dict:
    """`{op: status|arm|disarm}` for `POST /plugins/kvtier/tier`. Never raises.

    `arm`/`disarm` flip the kill switch; `status` only reads. A MISSING `op` is
    `status` (the friendly read default this verb has always had, and what a bare
    `GET /tier` sends); an `op` that IS named and is not one of `TIER_OPS` is a
    DECLINE with 400 semantics in the payload, never a flip: a mistyped body must not
    silently turn the tier off (or on) on the way to doing something else. The
    endpoint validates `op` against `TIER_OPS` first, so this is the second line of
    defence for a direct engine call.

    A call marked `safe_method` (forwarded from a method a browser may issue without
    a preflight) that asks to arm or disarm is refused outright, the way
    `disk_store.disk_for_engine` refuses: the route answers 405 first, and this is
    what makes that refusal not depend on the route staying correct.

    The switch is applied to THIS process's manager class (the caller is the
    EngineCore loop thread, which holds the live manager), so the verb is also
    what makes a freshly-adopted class switchable.
    """
    args = args if isinstance(args, dict) else {}
    refused = route_policy.safe_method_refusal(config.ROUTE_TIER, args)
    if refused is not None:
        return refused
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
        # Top level, next to `applied`, because this is the payload an operator
        # actually reads after `POST /tier`: "is it armed, is the fix in, and is
        # the answer supposed to be no?".
        out["armed"] = bool(_state["armed"])
        out["boot_default_armed"] = bool(BOOT_ARMED)
        out["boot"] = boot_section()
        out["operator_note"] = out["tier"]["operator_note"]
        refusal = out.get("state") if isinstance(out.get("state"), dict) else None
        out["next_step"] = (_next_step(refusal) if refusal and refusal.get("refused")
                            else _next_step(_arm_block_reason()))
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
    # `patched` and `installed` answer the SAME question and come from the SAME
    # verdict, and always will: the historical `patched` ("an install ran at
    # some point") kept reporting true after a later install was REFUSED - the
    # 2026-10-03 payload said tier.patched: true while installed: false. A
    # counter of past attempts is not a claim about the class; the all-or-
    # nothing verdict is, so both fields read `_state["fully_installed"]`.
    fields = {
        "patched": bool(_state["fully_installed"]),
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
        # THE BOOT POSTURE, stated in the payload: `boot` is the verdict of the
        # connector-creation boot (mode, the three parts, their refusal reasons,
        # the faults the guard had caught at boot and has caught since), and
        # `boot_line` is the exact text the boot logged. `boot.ran: false` means
        # no tier connector has been built in this process yet.
        "boot": boot_section(),
        "boot_line": _state.get("boot_line"),
        # The STATE AT IMPORT, which is not the same claim as the boot verdict:
        # the tier starts disarmed and the boot arms it only on a verified
        # install, so `boot_default_armed` stays false in every mode.
        "boot_default_armed": bool(BOOT_ARMED),
        "boot_is_inert": not BOOT_ARMED,
        "arm_requires": ("the PR #56810 fix fully installed AND the geometry "
                         "guard passing (refusal kinds: "
                         + ", ".join(ARM_BLOCK_KINDS) + ")"),
        "operator_note": _operator_note(bool(_state["armed"]),
                                        bool(_state["fully_installed"])),
        # All-or-nothing install state: `installed` is true ONLY when every
        # method in METHODS is patched; `refused_methods` names the ones that
        # are not, so a partial install is visible, not just a failed one.
        "installed": bool(_state["fully_installed"]),
        "refused_methods": list(_state["refused_methods"]),
        "live_check": (dict(_state["live_check"]) if _state["live_check"]
                       else None),
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
        "boots": int(_state.get("boots") or 0),
        "methods": {name: dict(v) for name, v in _state["method_verdicts"].items()},
        "replaced": list(REPLACED),
        "wrapped": ["__init__"],
        "switch_methods": list(SWITCH_METHODS),
        "switch": {name: dict(v) for name, v in _state["switch_verdicts"].items()},
        # WHERE THE PATCH LANDED: the answer to "was this engine patched before
        # its first request, or only after somebody curled a verb?".
        "connector_hook": connector_hook_section(),
        # THE SECOND RUNTIME-PATCH TARGET, in the same section as the first because
        # an operator reads them together: "is this engine patched?" now has two
        # answers, and `mamba_anchor.installed` is the field a payload reader checks
        # beside the `MAMBA_ANCHOR[...]` token on the boot line.
        "mamba_anchor": mamba_anchor.section(),
        "patched_before_first_request": bool(_state["hook_tier_connectors"]
                                             and _state["hook_installs"]),
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
        raw = live_cls.__dict__.get("_select_eager_blocks_to_store")
        layers, _bottom = _layers_of(_function_of(raw)) if raw is not None \
            else ([], None)
        # Layer-aware: our copy UNDER guard mode's wrapper is still live code.
        live_patched = any(kind == "backport" for kind, _ in layers)
        live_armed = bool(_state["armed"])
    except Exception as exc:  # noqa: BLE001 - a manager that lies is data
        snapshot._error(exc, "backport live view")
    return _fields(live_manager_patched=live_patched,
                   live_manager_armed_noop=live_armed)


def empty_section(reason: str) -> dict:
    """The degraded section. Same key set, by construction."""
    return _fields(error=str(reason)[:200])
