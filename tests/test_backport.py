"""Tests for the PR #56810 runtime backport (`backport.py`).

Two kinds of test live here, and the distinction matters:

* The REAL patched code is driven, not a re-implementation of it. Methods come
  from `fixtures_pr56810.py` (the fork's own text, fingerprint-verified by
  tools/gen_pr56810_fixture.py) and are installed by `backport.install()`. The
  store selector is driven with a poison-pill `resolve_block_hashes` that
  reproduces the production assertion, so "the patch works" is a claim about
  behaviour, not about a string match.
* Everything here is stdlib-only, matching the rest of the suite; the plugin is
  importable with no vllm installed. End-to-end proof against the real engine
  class is the separate 3-way matrix (unmodified/reference/monkeypatch) run
  under the vllm venv.
"""

from __future__ import annotations

import inspect
import linecache
import subprocess
import sys
import types
from pathlib import Path

import fixtures_pr56810 as FX
import pytest

from kvtier_dev_plugin import (
    backport,
    config,
    engine_side,
    guard_mode,
    pr56810_sources,
    snapshot,
)

# --- the geometry that broke production ------------------------------------
HASH_BLOCK_SIZE = 3200          # five groups of 3200 ...
CACHEABLE_GROUPS = 5
RING_GROUP = 5                  # ... plus one QSA ring of 8
RING_BLOCK_SIZE = 8             # NOT divisible by 3200 -> the assert
GROUP_BLOCK_SIZES = [HASH_BLOCK_SIZE] * CACHEABLE_GROUPS + [RING_BLOCK_SIZE]
PCIDS = tuple(range(CACHEABLE_GROUPS))


@pytest.fixture(autouse=True)
def reset_backport_state():
    """`backport._state` is process-wide, like every other piece of plugin state.

    It resets through the MODULE's own `reset_state_for_tests()` rather than a
    hand-copied field list: a hand-copied list silently stops covering new
    fields, and a leaked `fully_installed`/`armed` is exactly the kind of leak
    that makes an arm-refusal test pass for the wrong reason. The boot default
    comes from `backport.BOOT_ARMED`, so a test that wants an ARMED tier has to
    arm it - which is the production story.
    """
    saved = dict(backport._state)
    saved_verdicts = dict(backport._state["method_verdicts"])
    saved_switch = dict(backport._state["switch_verdicts"])
    backport.reset_state_for_tests()
    yield
    backport._state.clear()
    backport._state.update(saved)
    backport._state["method_verdicts"] = saved_verdicts
    backport._state["switch_verdicts"] = saved_switch


# --- fakes ------------------------------------------------------------------

class _Spec:
    def __init__(self, block_size, prefix_cacheable=True):
        self.block_size = block_size
        self.prefix_cacheable = prefix_cacheable


class _Group:
    def __init__(self, block_size, prefix_cacheable=True, name="layer"):
        self.kv_cache_spec = _Spec(block_size, prefix_cacheable)
        self.layer_names = [name]


class _Config:
    """A stand-in `KVCacheConfig` exposing the two properties the patch reads."""

    def __init__(self, groups, prefix_cacheable_ids=None):
        self.kv_cache_groups = list(groups)
        self.kv_cache_specs = {g: grp.kv_cache_spec
                               for g, grp in enumerate(self.kv_cache_groups)}
        self.prefix_cacheable_group_ids = (
            tuple(range(len(groups))) if prefix_cacheable_ids is None
            else tuple(prefix_cacheable_ids))

    @property
    def prefix_cacheable_groups(self):
        return tuple(self.kv_cache_groups[g]
                     for g in self.prefix_cacheable_group_ids)


def our_shape_config():
    """5 x block_size 3200 (cacheable) + 1 x block_size 8 (the ring)."""
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(RING_BLOCK_SIZE, False, "qsa_ring"))
    return _Config(groups, PCIDS)


class _Manager:
    """`CircularBufferManager` does NOT override this, which is the whole bug."""

    def __init__(self, stable=True):
        self.has_positionally_stable_blocks = stable


class _Coordinator:
    def __init__(self, num_groups, partial=False):
        self.enable_partial_hash_hits = partial
        self.single_type_managers = [_Manager(True) for _ in range(num_groups)]


class _Pool:
    def __init__(self, free=1000):
        self._free = free

    def get_num_free_blocks(self):
        return self._free

    def get_new_blocks(self, n):
        """Route 2 allocates its CPU destinations this way (manager.py:1191)."""
        return [_GpuBlock(9000 + i, b"") for i in range(n)]


class _GpuBlock:
    def __init__(self, block_id, block_hash):
        self.block_id = block_id
        self.is_null = False
        self.block_hash = block_hash


class _GpuPool:
    def __init__(self, blocks):
        self.blocks = blocks
        self.touched = []

    def touch(self, blocks):
        """Route 2 pins the GPU blocks it queued (manager.py:1195)."""
        self.touched.extend(blocks)


class _Request:
    def __init__(self, num_blocks_of_3200, request_id="req-0"):
        self.request_id = request_id
        self.num_computed_tokens = HASH_BLOCK_SIZE * num_blocks_of_3200
        self.num_output_placeholders = 0
        self.lora_request = None
        self.all_token_ids = list(range(self.num_computed_tokens + 64))
        self.block_hashes = [f"h{i}".encode() for i in range(512)]


class _State:
    def __init__(self, request, num_groups):
        self.request = request
        self.num_stored_blocks = [0] * num_groups


class _Admission:
    READY = "READY"
    DEFERRED = "DEFERRED"


def resolve_block_hashes_poison(block_hashes, hash_block_size, block_size):
    """The production assertion, verbatim from kv_cache_utils.py:2888.

    A group whose block size is not a multiple of the hash granularity raises
    here. That is the crash: `_select_eager_blocks_to_store` reaches it with
    (hash_block_size=3200, block_size=8) for the ring group.
    """
    if block_size == hash_block_size:
        return list(block_hashes)
    assert block_size % hash_block_size == 0, (
        f"tokens_per_block={block_size} not divisible by "
        f"tokens_per_hash={hash_block_size}")
    step = block_size // hash_block_size
    return [block_hashes[i * step] for i in range(len(block_hashes) // step)]


class _SelectorSelf:
    """Everything `_select_eager_blocks_to_store` touches, and nothing else."""

    def __init__(self, pcids=PCIDS, num_free=1000):
        self.prefix_cacheable_group_ids = pcids
        self.group_block_sizes = list(GROUP_BLOCK_SIZES)
        self.hash_block_size = HASH_BLOCK_SIZE
        self.block_size = HASH_BLOCK_SIZE
        self.cpu_kv_cache_config = _Config(
            [_Group(b) for b in GROUP_BLOCK_SIZES], pcids)
        self.cpu_coordinator = _Coordinator(len(GROUP_BLOCK_SIZES))
        self.cpu_block_pool = _Pool(num_free)
        self.enable_kv_cache_events = False
        n = 4096
        self._gpu_block_pool = _GpuPool(
            [_GpuBlock(i, f"bh{i}".encode()) for i in range(n)])
        self._store_admission = _Admission

    def _classify_store_candidate(self, gpu_block):
        return _Admission.READY

    def _cached_gpu_block(self, hashes, idx, g):
        return None


def selector_namespace():
    """Module globals the original function closes over."""
    return {
        "resolve_block_hashes": resolve_block_hashes_poison,
        "_StoreAdmission": _Admission,
        "BlockStoreMeta": dict,
        "maybe_convert_block_hash": lambda h: h,
        "make_block_hash_with_group_id": lambda h, g: (h, g),
        "generate_block_hash_extra_keys": lambda r, s, e, i: (None, i),
        "StoreRequestState": _State,
        "logger": types.SimpleNamespace(warning=lambda *a, **k: None),
        "cdiv": lambda a, b: -(-a // b),
    }


def marker_on(cls, name):
    """The ownership marker on a class attribute, unwrapping staticmethod."""
    raw = cls.__dict__.get(name)
    if isinstance(raw, staticmethod):
        raw = raw.__func__
    return getattr(raw, backport.OWNER_ATTR, None) if raw is not None else None


def original_under(cls, name):
    raw = cls.__dict__.get(name)
    if isinstance(raw, staticmethod):
        raw = raw.__func__
    return getattr(raw, backport.ORIGINAL_ATTR, None) if raw is not None else None


def ring_and_cacheable_ids():
    """Distinct GPU ids per group; the ring's ids are the ones that must not move."""
    per_group = []
    nxt = 1
    for g in range(len(GROUP_BLOCK_SIZES)):
        count = 4 if g != RING_GROUP else 1
        per_group.append(list(range(nxt, nxt + count)))
        nxt += count
    return tuple(per_group), set(per_group[RING_GROUP])


def drive(selector_cls, self):
    """Call the installed (or original) selector with the 6-group shape."""
    block_ids, ring_ids = ring_and_cacheable_ids()
    state = _State(_Request(2), len(GROUP_BLOCK_SIZES))
    fn = selector_cls.__dict__["_select_eager_blocks_to_store"]
    if isinstance(selector_cls.__dict__.get("_select_eager_blocks_to_store"),
                staticmethod):
        fn = fn.__func__
    return fn(self, state, block_ids), block_ids, ring_ids


# --- 1. the fingerprint guard ----------------------------------------------

def test_install_accepts_the_fork_text_and_patches_every_method():
    cls, _ns = FX.build_fork_class(selector_namespace())
    report = backport.install(cls)
    assert report["installed"] is True, report
    for name in backport.METHODS:
        verdict = report["methods"][name]
        assert verdict["patched"] is True, (name, verdict)
        assert verdict.get("sha_ok") is not False, (name, verdict)
    for name in backport.REPLACED:
        assert marker_on(cls, name) == backport.OWNER_TAG, name


def test_stale_fingerprint_refuses_that_method_and_names_it():
    """A drifted method is refused, not patched, and the reason says fingerprint."""
    ns = selector_namespace()
    cls, _mod = FX.build_fork_class(ns)
    # Simulate a future fork editing ONE method: rewrite its source text so the
    # live hash no longer matches the recorded one.
    drifted = FX.ORIGINAL_SOURCES["update_state_after_alloc"].replace(
        "def update_state_after_alloc(",
        "def update_state_after_alloc(  # fork edit\n    # noqa\n", 1)
    module = types.ModuleType("drifted")
    module.__dict__.update(ns)
    src = ("class SimpleCPUOffloadScheduler:\n"
           + FX.ORIGINAL_SOURCES["__init__"]
           + FX.ORIGINAL_SOURCES["_estimate_lazy_target_blocks"]
           + drifted
           + FX.ORIGINAL_SOURCES["_select_eager_blocks_to_store"])
    name = "<drifted_fork>"
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    exec(compile(src, name, "exec"), module.__dict__,  # noqa: S102
         module.__dict__)
    drifted_cls = module.SimpleCPUOffloadScheduler

    report = backport.install(drifted_cls)
    verdict = report["methods"]["update_state_after_alloc"]
    assert verdict["patched"] is False, verdict
    assert "fingerprint" in verdict["reason"]
    # The refusal is per-method: a moved function cannot poison the others.
    assert report["methods"]["_select_eager_blocks_to_store"]["patched"] is True


def test_unreadable_source_is_a_refusal_not_a_crash(monkeypatch):
    cls, _ = FX.build_fork_class(selector_namespace())
    monkeypatch.setattr(backport, "fingerprint_of", lambda fn: None)
    report = backport.install(cls)
    assert all(v["patched"] is False for v in report["methods"].values())
    assert report["installed"] is False


def test_recorded_fingerprints_are_stable_and_sixty_four_hex():
    for name, entry in pr56810_sources.SOURCES.items():
        sha = entry["orig_sha256"]
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), name
        # The patched text must differ from the original it replaces.
        assert entry["source"] != FX.ORIGINAL_SOURCES[name], name


# --- 2. idempotence ---------------------------------------------------------

def test_install_is_idempotent_and_never_nests_its_own_wrapper():
    """Re-installing must rebind the FORK's original, never wrap our own copy.

    Nesting is the failure this guards: a wrapped wrapper would apply the
    `__init__` attribute assignment twice per construction, and a copy compiled
    from a copy would compute its fingerprint against our own text and then
    believe itself to be upstream's.
    """
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    # Unwrap descriptors on BOTH sides: `original_under` returns the function
    # under a staticmethod, so the recorded baseline must be unwrapped too.
    fork_inits = {name: backport._function_of(cls.__dict__[name])
                  for name in backport.METHODS}

    first = backport.install(cls)
    second = backport.install(cls)
    assert first["installed"] is True and second["installed"] is True
    assert backport._state["reinstalls"] >= 1
    assert (second["methods"]["__init__"]["reason"]
            == "re-installed over our own wrapper")

    for name in backport.METHODS:
        assert marker_on(cls, name) == backport.OWNER_TAG, name
        assert original_under(cls, name) is fork_inits[name], (
            f"{name}: the second install re-bound something other than the "
            f"fork's original -- the wrapper nested")

    # Every method must still behave, after two installs.
    self = _SelectorSelf()
    (selected, _adv, _meta), _ids, ring_ids = drive(cls, self)
    assert not (set(selected) & ring_ids)


def test_install_survives_a_repeated_reload_loop():
    """The plugin reloads per engine launch; ten installs must equal one."""
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    fork_select = backport._function_of(
        cls.__dict__["_select_eager_blocks_to_store"])
    for _ in range(10):
        report = backport.install(cls)
        assert report["installed"] is True
        assert original_under(cls, "_select_eager_blocks_to_store") is fork_select
    self = _SelectorSelf()
    (selected, _adv, _meta), _ids, ring_ids = drive(cls, self)
    assert not (set(selected) & ring_ids)


# --- 3. the kill switch -----------------------------------------------------

class EmptyMetadata:
    """Stand-in `SimpleCPUOffloadMetadata`; the real one is a dataclass."""

    def __init__(self):
        self.store_gpu_blocks = []
        self.store_cpu_blocks = []
        self.load_gpu_blocks = []
        self.load_cpu_blocks = []
        self.store_event = -1
        self.load_event = -1


def _fresh_class_with_switch():
    """A freshly rebuilt class carrying the four switch methods, NOT installed.

    The fixture rebuilds only the four PR #56810 methods, so the four the kill
    switch wraps get stand-in originals here (the fork really does define all
    four - `_queue_finished_eager_store` is `manager.py:1169`). They are
    deliberately loud: each records that it ran, which is how the tests prove a
    disarmed connector never reaches the engine's own code. A NEW class every
    call, so a test can watch the first install happen.
    """
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    calls = {"matched": 0, "meta": 0, "pending": 0, "finished": 0}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        calls["matched"] += 1
        return 4096, True

    def build_connector_meta(self, scheduler_output):
        calls["meta"] += 1
        return "ENGINE METADATA"

    def has_pending_stores(self):
        calls["pending"] += 1
        return True

    def _queue_finished_eager_store(self, request, block_ids):
        # The real one returns None and only ever queues; a decline is the same
        # shape the fork returns when it queues nothing.
        calls["finished"] += 1
        return "QUEUED"

    cls.get_num_new_matched_tokens = get_num_new_matched_tokens
    cls.build_connector_meta = build_connector_meta
    cls.has_pending_stores = has_pending_stores
    cls._queue_finished_eager_store = _queue_finished_eager_store

    module = types.ModuleType(backport.METADATA_MODULE)
    module.SimpleCPUOffloadMetadata = EmptyMetadata
    sys.modules[backport.METADATA_MODULE] = module
    return cls, calls


def _install_with_metadata():
    cls, calls = _fresh_class_with_switch()
    report = backport.install(cls)
    assert report["installed"] is True, report
    for name, verdict in report["switch"].items():
        assert verdict["wrapped"] is True, (name, verdict)
    return cls, EmptyMetadata, calls


def teardown_module():
    sys.modules.pop(backport.METADATA_MODULE, None)


def test_disarmed_reports_no_prefix_hit():
    cls, _meta, calls = _install_with_metadata()
    self = object()
    backport.set_armed(True)      # armed by an explicit act, never at boot

    assert cls.__dict__["get_num_new_matched_tokens"](self, None, 0) == (4096, True)
    assert calls["matched"] == 1

    backport.set_armed(False)
    assert cls.__dict__["get_num_new_matched_tokens"](self, None, 0) == (0, False)
    assert calls["matched"] == 1, "disarmed must not reach the engine's matcher"

    backport.set_armed(True)
    assert cls.__dict__["get_num_new_matched_tokens"](self, None, 0) == (4096, True)
    assert calls["matched"] == 2


def test_disarmed_builds_an_empty_metadata_so_the_worker_does_nothing():
    cls, EmptyMetadata, calls = _install_with_metadata()
    self = object()
    backport.set_armed(True)

    assert cls.__dict__["build_connector_meta"](self, None) == "ENGINE METADATA"
    backport.set_armed(False)
    meta = cls.__dict__["build_connector_meta"](self, None)
    assert isinstance(meta, EmptyMetadata)
    assert calls["meta"] == 1, "disarmed must not ask the engine for metadata"
    # The worker copies ONLY these lists (worker.py:266,289): empty == no-op.
    assert list(meta.store_gpu_blocks) == []
    assert list(meta.load_gpu_blocks) == []
    assert list(meta.store_cpu_blocks) == []
    assert list(meta.load_cpu_blocks) == []


def test_disarmed_claims_no_pending_work():
    cls, _meta, calls = _install_with_metadata()
    self = object()
    backport.set_armed(True)

    assert cls.__dict__["has_pending_stores"](self) is True
    backport.set_armed(False)
    assert cls.__dict__["has_pending_stores"](self) is False
    assert calls["pending"] == 1
    backport.set_armed(True)
    assert cls.__dict__["has_pending_stores"](self) is True
    assert calls["pending"] == 2


def test_switch_wrappers_do_not_nest_across_reloads():
    cls, _meta, calls = _install_with_metadata()
    for _ in range(5):
        report = backport.install(cls)
        assert all(v["wrapped"] for v in report["switch"].values())
    backport.set_armed(True)
    assert cls.__dict__["has_pending_stores"](object()) is True
    assert calls["pending"] == 1, "five installs must still be one wrapper deep"
    backport.set_armed(False)
    assert cls.__dict__["has_pending_stores"](object()) is False
    assert calls["pending"] == 1


def test_kill_switch_reports_its_own_state():
    _install_with_metadata()
    # Installing does not arm: a freshly installed tier is still the boot default
    # until someone asks for it to be live.
    assert backport.is_armed() is backport.BOOT_ARMED is False
    assert backport.set_armed(True)["changed"] is True
    assert backport.set_armed(False)["changed"] is True
    assert backport.set_armed(False)["changed"] is False, "a no-op must not count"
    assert backport.is_armed() is False
    assert backport.section()["armed"] is False


def test_switch_is_independent_of_the_backport_fingerprints(monkeypatch):
    """A stale fingerprint must not BREAK the kill switch, but it must make the
    tier un-armable: the disarmed connector still no-ops (the switch is live),
    yet arming is refused because the fix is not fully in place.

    This is the fail-open closed: a partial install leaves the tier force-
    disarmed, and `set_armed(True)` refuses rather than re-arming an engine whose
    store selector was never patched.
    """
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    _switch_standin_methods(cls)
    monkeypatch.setattr(backport, "fingerprint_of", lambda fn: None)
    report = backport.install(cls)
    assert report["installed"] is False, "fingerprints were poisoned"
    # The refusal force-disarms the tier and records WHY (incomplete, not geometry).
    assert report["auto_disarmed"] is True
    assert report["disarm_reason"] == "incomplete_install"
    assert backport.is_armed() is False
    # The switch wrappers still install (they need no fingerprint): disarmed, the
    # connector is a strict no-op even though the selector was never patched.
    assert all(v["wrapped"] for v in report["switch"].values()), report["switch"]
    assert cls.__dict__["has_pending_stores"](object()) is False
    assert cls.__dict__["get_num_new_matched_tokens"](object(), None, 0) == (0, False)
    # Arming is REFUSED (state unchanged, nothing reached the engine), naming the
    # refused methods rather than silently flipping on a half-patched tier.
    res = backport.set_armed(True)
    assert res["refused"] is True and res["changed"] is False
    assert res["kind"] == "incomplete_install"
    assert "_select_eager_blocks_to_store" in res["reason"]
    assert backport.is_armed() is False
    assert cls.__dict__["get_num_new_matched_tokens"](object(), None, 0) == (0, False)


# --- 3c. the all-or-nothing install contract --------------------------------

def _drifted_fork_class(drift_name):
    """The fork's class with exactly ONE method's source drifted off fingerprint.

    The drifted method is refused; the other three (including the store selector
    that carries the crash) install cleanly. That is the fail-open the contract
    kills: the tier must NOT be armed when the crash method could be the refused
    one, so a refused method anywhere makes the whole install a failure.
    """
    ns = selector_namespace()
    order = ("__init__", "_estimate_lazy_target_blocks",
             "update_state_after_alloc", "_select_eager_blocks_to_store")
    parts = []
    for name in order:
        text = FX.ORIGINAL_SOURCES[name]
        if name == drift_name:
            text = text.replace(f"def {name}(", f"def {name}(  # fork edit\n", 1)
        parts.append(text)
    module = types.ModuleType("drifted_fork")
    module.__dict__.update(ns)
    src = "class SimpleCPUOffloadScheduler:\n" + "".join(parts)
    fname = "<drifted_fork_class>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, "exec"), module.__dict__, module.__dict__)  # noqa: S102
    cls = module.SimpleCPUOffloadScheduler
    cls.__module__ = "drifted_fork"
    return cls


def _switch_standin_methods(cls):
    cls.get_num_new_matched_tokens = lambda self, r, n: (1, True)
    cls.build_connector_meta = lambda self, so: "ENGINE METADATA"
    cls.has_pending_stores = lambda self: True
    cls._queue_finished_eager_store = lambda self, r, b: None
    return cls


def test_partial_install_reports_incomplete_and_disarms():
    """One drifted fingerprint => installed False AND armed False.

    Drift the store selector itself (the method that contains the crash): the
    other three install, yet the tier must come up DISARMED, not armed with the
    crash still live.
    """
    cls = _switch_standin_methods(_drifted_fork_class("_select_eager_blocks_to_store"))
    report = backport.install(cls)
    # The three that matched are patched; the crash method is refused.
    assert report["methods"]["update_state_after_alloc"]["patched"] is True
    assert report["methods"]["_select_eager_blocks_to_store"]["patched"] is False
    # ALL-OR-NOTHING: one refusal is a failed install, not a partial success.
    assert report["installed"] is False
    assert report["refused_methods"] == ["_select_eager_blocks_to_store"]
    # Force-disarmed, and the reason is the incomplete install (not geometry).
    assert report["auto_disarmed"] is True
    assert report["disarm_reason"] == "incomplete_install"
    assert backport.is_armed() is False
    section = backport.section()
    assert section["installed"] is False
    assert section["refused_methods"] == ["_select_eager_blocks_to_store"]
    assert section["arm_block"]["kind"] == "incomplete_install"


def test_runtime_arm_is_refused_while_incomplete_and_names_the_method():
    cls = _switch_standin_methods(_drifted_fork_class("update_state_after_alloc"))
    backport.install(cls)
    assert backport.is_armed() is False
    res = backport.set_armed(True)             # refused: fix is not fully in place
    assert res["refused"] is True
    assert res["armed"] is False and res["changed"] is False
    assert res["kind"] == "incomplete_install"
    assert "update_state_after_alloc" in res["reason"]
    assert backport.is_armed() is False, "a refused arm must not flip the state"
    assert backport._state["arm_refusals"] == 1
    # Never raises, and the disarmed switch still holds the tier as a no-op.
    assert cls.__dict__["has_pending_stores"](object()) is False


def test_arm_succeeds_after_a_full_install():
    """Once every method installs, arming is allowed and reports no refusal."""
    cls, _meta, _calls = _install_with_metadata()
    assert backport.section()["installed"] is True
    assert backport.section()["arm_block"] is None
    # Start disarmed, then arm: fully installed + geometry not vetoing => allowed.
    backport.set_armed(False)
    res = backport.set_armed(True)
    assert res["refused"] is False and res["armed"] is True and res["changed"] is True
    assert backport.is_armed() is True


def test_geometry_refusal_disarms_and_reports_its_own_reason():
    """A geometry refusal disarms too, with a reason DISTINCT from incomplete."""
    cls, _ = FX.build_fork_class(selector_namespace())
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(HASH_BLOCK_SIZE, True, "group_five_is_cacheable"))
    cfg = _Config(groups, PCIDS)

    report = backport.install(cls, kv_cache_config=cfg)
    assert report["installed"] is False
    assert report["auto_disarmed"] is True
    assert report["disarm_reason"] == "geometry"
    assert backport.is_armed() is False
    # A geometry veto blocks arming even though nothing was refused per method.
    res = backport.set_armed(True)
    assert res["refused"] is True and res["kind"] == "geometry"
    assert "prefix_cacheable" in res["reason"]
    assert backport.is_armed() is False


def test_the_two_refusal_reasons_are_distinguishable():
    """`disarm_reason` separates the geometry guard from an incomplete install."""
    incomplete = _switch_standin_methods(
        _drifted_fork_class("_select_eager_blocks_to_store"))
    backport.install(incomplete)
    assert _state_disarm_reason() == "incomplete_install"

    backport.reset_state_for_tests()
    cls, _ = FX.build_fork_class(selector_namespace())
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(HASH_BLOCK_SIZE, True, "group_five_is_cacheable"))
    backport.install(cls, kv_cache_config=_Config(groups, PCIDS))
    assert _state_disarm_reason() == "geometry"


def _state_disarm_reason():
    return backport.section()["disarm_reason"]


def test_disarm_is_never_refused_even_when_incomplete():
    """Disarm is the safe direction: it always applies, even before any install.

    With `BOOT_ARMED = False` this is also the boot state itself: refusing to
    install must not leave anything armed behind, so the tier is already inert
    here and a disarm is a refused-free no-op.
    """
    backport.install(None)                     # nothing installed at all
    assert backport.is_armed() is False        # the boot default, no connector
    res = backport.set_armed(False)
    assert res["refused"] is False and res["changed"] is False
    assert backport.is_armed() is False


def test_tier_verb_reports_a_refused_arm_to_the_operator():
    """POST /tier {op:arm} on an incomplete tier says why, and did not apply."""
    cls = _switch_standin_methods(_drifted_fork_class("update_state_after_alloc"))
    engine = _FakeEngine(_live_manager(cls))
    status = backport.tier_for_engine(engine, {"op": "status"})
    assert status["install"]["installed"] is False
    assert "update_state_after_alloc" in status["install"]["refused_methods"]
    assert status["tier"]["armed"] is False

    arm = backport.tier_for_engine(engine, {"op": "arm"})
    assert arm["applied"] is False, "a refused arm is not applied"
    assert arm["status_code"] == 200           # a refusal is reported, not raised
    assert arm["state"]["refused"] is True
    assert "update_state_after_alloc" in arm["reason"]
    assert arm["tier"]["armed"] is False


# --- 3b. the tier verb (POST /plugins/kvtier/tier) --------------------------

class _FakeEngine:
    """The three hops `manager_of` walks, plus the manager's own attributes."""

    def __init__(self, manager):
        self.scheduler = types.SimpleNamespace(
            connector=types.SimpleNamespace(scheduler_manager=manager))


def _live_manager(cls, config=None):
    """A real instance of the patched class, WITHOUT running its constructor.

    `cls.__new__(cls)` is deliberate: the fixture's `__init__` is the fork's own
    119-line constructor and cannot run without a real vllm config. `__new__`
    gives an instance whose `type()` IS the patched class - which is exactly what
    `ensure_installed` reads - and leaves `__init__` uninvolved.

    The instance carries the constructor's attribute set MINUS
    `prefix_cacheable_group_ids`: that is what an instance constructed before
    the patch actually looks like, and what `backport._live_object_check`
    now asserts and back-fills (see tests/test_live_object_order.py). The
    attribute donor is the same `_SelectorSelf` as `_production_manager`.
    """
    manager = cls.__new__(cls)
    helper = _SelectorSelf()
    for attr in ("group_block_sizes", "hash_block_size", "block_size",
                 "cpu_kv_cache_config", "cpu_coordinator", "cpu_block_pool",
                 "enable_kv_cache_events", "_gpu_block_pool"):
        setattr(manager, attr, getattr(helper, attr))
    manager._lazy_mode = False
    manager._reqs_to_store = {}
    manager._pending_cpu_hits = {}
    manager._reqs_to_load = {}
    manager.kv_cache_config = config if config is not None else our_shape_config()
    return manager


def _engine_with_live_manager():
    cls, _meta, calls = _install_with_metadata()
    return _FakeEngine(_live_manager(cls)), cls, calls


def test_tier_verb_disarms_through_the_engine_and_reports_it():
    engine, _cls, calls = _engine_with_live_manager()
    status = backport.tier_for_engine(engine, {"op": "status"})
    assert status["status_code"] == 200
    # The cold-boot truth: installed by the verb, STILL disarmed, because arming
    # is a separate deliberate act (see the runbook's boot sequence).
    assert status["tier"]["armed"] is False
    assert status["armed"] is False
    assert status["tier"]["patched"] is True

    armed = backport.tier_for_engine(engine, {"op": "arm"})
    assert armed["applied"] is True and armed["tier"]["armed"] is True

    out = backport.tier_for_engine(engine, {"op": "disarm"})
    assert out["applied"] is True and out["status_code"] == 200
    assert out["tier"]["armed"] is False
    assert backport.is_armed() is False

    assert backport.tier_for_engine(engine, {"op": "arm"})["tier"]["armed"] is True


def test_tier_verb_needs_no_arguments_and_defaults_to_status():
    engine, _cls, _calls = _engine_with_live_manager()
    out = backport.tier_for_engine(engine)          # exactly `GET /snapshot`-style
    assert out["status_code"] == 200 and out["applied"] is False
    assert out["op"] == "status"


def test_tier_verb_refuses_an_unknown_op_without_changing_anything():
    """A typo must not flip the tier on the way to doing something else."""
    engine, _cls, _calls = _engine_with_live_manager()
    backport.set_armed(True)
    assert backport.is_armed() is True
    out = backport.tier_for_engine(engine, {"op": "dissarm"})
    assert out["status_code"] == 400
    assert out["applied"] is False
    assert "Nothing was changed" in out["reason"]
    assert backport.is_armed() is True, "a refused op must not disarm the tier"

    backport.set_armed(False)
    backport.tier_for_engine(engine, {"op": "dissarm"})
    assert backport.is_armed() is False, "a refused op must not arm the tier either"


def test_tier_verb_installs_the_backport_on_the_way_through():
    """The verb adopts the live manager's class, so a late import still lands it.

    The class here has NEVER been installed on, which is the real boot order:
    `register()` runs before the connector imports the manager, so this verb is
    what actually patches a live engine.
    """
    cls, _calls = _fresh_class_with_switch()
    backport._state["installs"] = 0
    backport._state["method_verdicts"] = {}
    out = backport.tier_for_engine(_FakeEngine(_live_manager(cls)),
                                   {"op": "status"})
    assert out["install"]["installed"] is True
    assert backport._state["installs"] == 1
    verdict = out["tier"]["methods"]["_select_eager_blocks_to_store"]
    assert verdict["patched"] is True, verdict


def test_ensure_installed_does_not_recompile_on_the_polling_path():
    """The snapshot verb runs on the EngineCore loop thread every few seconds.

    A full install re-compiles three upstream functions; `already_installed`
    must keep that off the serving loop once the markers are in place. This
    asserts the cheap path by making the expensive one explode.
    """
    cls, _calls = _fresh_class_with_switch()
    engine = _FakeEngine(_live_manager(cls))
    assert backport.ensure_installed(engine)["installed"] is True
    assert backport.already_installed(cls) is True

    def explode(*a, **k):
        raise AssertionError("the polling path must not recompile")

    original = backport._compile_copy
    backport._compile_copy = explode
    try:
        out = backport.ensure_installed(engine)
    finally:
        backport._compile_copy = original
    assert out["installed"] is True
    assert "already installed" in out["reason"]


def test_ensure_installed_disarms_when_a_later_config_fails_the_guard():
    """A patch already live on a class, then a config that must not use it.

    The guard cannot un-patch a class instances are already built from, so the
    only safe action left is to stop the tier scheduling anything at all.
    """
    cls, _calls = _fresh_class_with_switch()
    engine = _FakeEngine(_live_manager(cls))
    assert backport.ensure_installed(engine)["installed"] is True
    backport.set_armed(True)      # the scenario is a tier that IS live
    assert backport.is_armed() is True

    # Same class, but now the live config omits a group that IS cacheable.
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(HASH_BLOCK_SIZE, True, "suddenly_cacheable"))
    bad = _Config(groups, PCIDS)
    engine.scheduler.connector.scheduler_manager.kv_cache_config = bad

    out = backport.ensure_installed(engine)
    assert out["installed"] is False
    assert out.get("auto_disarmed") is True
    assert "suddenly_cacheable" in out["reason"]
    assert backport.is_armed() is False, (
        "a config the patch must not run for must leave the tier disarmed")


def test_tier_verb_survives_an_engine_with_no_connector():
    """No manager is a normal state (tier off); the verb must still answer."""
    out = backport.tier_for_engine(_FakeEngine(None), {"op": "status"})
    assert out["status_code"] == 200
    assert out["tier"]["patched"] in (True, False)


def test_tier_verb_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("manager exploded")

    monkeypatch.setattr(backport, "manager_of", boom)
    out = backport.tier_for_engine(object(), {"op": "disarm"})
    assert out["status_code"] == 500
    assert "manager exploded" in out["reason"]
    assert backport.snapshot._state["errors"] > 0


# --- 3d. THE BOOT DEFAULT: the tier is inert until someone arms it ----------
#
# `BOOT_ARMED = False` exists because the install CANNOT land at plugin-load
# time: `load_general_plugins()` runs at `core.py:217` while the Scheduler - and
# with it the connector and the offload manager - arrives at `core.py:260`. A
# tier that booted armed was a tier that booted UNPATCHED and faced the first
# request with the upstream assert still in its store path.

def _registered_manager_class(monkeypatch, cls):
    """Make `manager_class()` resolve, as it does once the connector has
    imported the manager - which is exactly the moment `register()` missed."""
    module = types.ModuleType(backport.TARGET_MODULE)
    module.SimpleCPUOffloadScheduler = cls
    monkeypatch.setitem(sys.modules, backport.TARGET_MODULE, module)
    return cls


def test_a_freshly_imported_module_is_disarmed():
    """The boot default, measured in a process that only IMPORTS the plugin.

    The fixtures reset state, so they can only prove what a reset produces - and
    "the tier does nothing at boot" is a claim about the first instant of a real
    process, before any test helper has touched it.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "from kvtier_dev_plugin import backport; "
         "print(backport.BOOT_ARMED, backport.is_armed(), "
         "backport.section()['armed'])"],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False", "False"], out.stdout


def test_the_boot_refusal_leaves_the_tier_disarmed_and_explains_itself():
    """`register()`'s install attempt, which the boot order always refuses.

    The refusal path has never disarmed, and no longer needs to: what it refuses
    INTO is now the disarmed default, so "not installed" and "idle" are the same
    state at boot instead of a crash waiting for a request.
    """
    report = backport.install(None)
    assert report["installed"] is False
    assert backport.is_armed() is False

    section = backport.section()
    assert section["boot_default_armed"] is False
    assert section["boot_is_inert"] is True
    assert section["armed"] is False and section["installed"] is False
    # Nothing to install onto and nothing attempted => its OWN kind.
    assert section["arm_block"]["kind"] == "no_manager"
    assert "boot" in section["operator_note"].lower()
    for kind in backport.ARM_BLOCK_KINDS:
        assert kind in section["arm_requires"]


def test_arm_from_a_cold_boot_installs_then_succeeds(monkeypatch):
    """The runbook sequence - status, then arm - from a class never installed on.

    `installing is not arming`: the status call lands the fix and leaves the tier
    disarmed; only the arm flips it, and only because it can prove the fix is
    fully in place against the live manager's config.
    """
    cls, calls = _fresh_class_with_switch()
    _registered_manager_class(monkeypatch, cls)
    backport.install(None)                       # the boot-time refusal
    engine = _FakeEngine(_live_manager(cls))

    status = backport.tier_for_engine(engine, {"op": "status"})
    assert status["install"]["installed"] is True, status["install"]
    assert status["armed"] is False, "installing is not arming"
    assert status["tier"]["arm_block"] is None
    assert "arm" in status["next_step"]

    arm = backport.tier_for_engine(engine, {"op": "arm"})
    assert arm["applied"] is True and arm["armed"] is True
    assert arm["state"]["kind"] is None
    assert calls == {"matched": 0, "meta": 0, "pending": 0, "finished": 0}, \
        "arming must not itself schedule any store or load"


def test_set_armed_seeks_its_own_proof(monkeypatch):
    """Arming without a verb first still cannot arm an unpatched tier.

    `set_armed(True)` runs `_arm_proof()` (an install onto the class, the only
    thing an arm can prove without a live manager), so no caller - a future verb,
    a direct import - can skip the proof by forgetting to call `ensure_installed`.
    """
    cls, _calls = _fresh_class_with_switch()
    _registered_manager_class(monkeypatch, cls)
    assert backport._state["fully_installed"] is False

    res = backport.set_armed(True)
    assert res["refused"] is False and res["armed"] is True
    assert backport.section()["installed"] is True


def test_arm_is_refused_from_a_cold_boot_when_the_fix_cannot_install(monkeypatch):
    """No fingerprint match => no arm, named, through the verb."""
    cls, _calls = _fresh_class_with_switch()
    _registered_manager_class(monkeypatch, cls)
    monkeypatch.setattr(backport, "fingerprint_of", lambda fn: None)
    engine = _FakeEngine(_live_manager(cls))

    status = backport.tier_for_engine(engine, {"op": "status"})
    assert status["install"]["installed"] is False
    assert "_select_eager_blocks_to_store" in status["install"]["refused_methods"]

    arm = backport.tier_for_engine(engine, {"op": "arm"})
    assert arm["applied"] is False and arm["armed"] is False
    assert arm["state"]["kind"] == "incomplete_install"
    assert "_select_eager_blocks_to_store" in arm["reason"]
    assert "refused" in arm["next_step"]


def test_arm_is_refused_on_a_geometry_veto_and_names_the_group():
    """A config the patch must not run for is refused with the geometry kind."""
    cls, _calls = _fresh_class_with_switch()
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(HASH_BLOCK_SIZE, True, "group_five_is_cacheable"))
    engine = _FakeEngine(_live_manager(cls, config=_Config(groups, PCIDS)))

    status = backport.tier_for_engine(engine, {"op": "status"})
    assert status["install"]["installed"] is False
    assert status["install"]["disarm_reason"] == "geometry"

    arm = backport.tier_for_engine(engine, {"op": "arm"})
    assert arm["applied"] is False
    assert arm["state"]["kind"] == "geometry"
    assert "prefix_cacheable" in arm["reason"]
    assert "prefix-cacheable" in arm["next_step"]


def test_a_class_install_cannot_forget_a_geometry_veto():
    """The arm-proof shape must not erase a veto a live config already gave.

    `install(cls)` with no config gets `checked: False` from the guard, which
    proves nothing. Overwriting the recorded veto with it would let `_arm_proof`
    arm a config the guard explicitly rejected - the fail-open the guard exists
    to close, re-opened through the arm gate.
    """
    cls, _calls = _fresh_class_with_switch()
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(HASH_BLOCK_SIZE, True, "group_five_is_cacheable"))
    bad = _Config(groups, PCIDS)
    assert backport.install(cls, kv_cache_config=bad)["installed"] is False
    assert backport.section()["arm_block"]["kind"] == "geometry"

    backport.install(cls)                        # kv_cache_config=None
    assert backport.section()["arm_block"]["kind"] == "geometry", \
        "a check with no config must not overwrite a veto with a verdict"
    assert backport.set_armed(True)["kind"] == "geometry"


def test_a_disarmed_tier_does_no_store_or_load_work_without_anyone_disarming():
    """The boot default, proven through the WRAPPERS, not through the flag.

    Nothing here calls `set_armed`: these are the return shapes a disarmed
    connector hands the scheduler (the fork's own "found nothing", an empty
    metadata, no pending work) with the engine's own methods never reached - the
    loud stand-ins count every call, so a wrapper that fell through is visible.
    """
    cls, EmptyMetadata, calls = _install_with_metadata()
    self = object()

    assert cls.__dict__["get_num_new_matched_tokens"](self, None, 0) == (0, False)
    meta = cls.__dict__["build_connector_meta"](self, None)
    assert isinstance(meta, EmptyMetadata)
    assert [list(getattr(meta, name)) for name in
            ("store_gpu_blocks", "store_cpu_blocks",
             "load_gpu_blocks", "load_cpu_blocks")] == [[], [], [], []]
    assert cls.__dict__["has_pending_stores"](self) is False
    assert calls == {"matched": 0, "meta": 0, "pending": 0, "finished": 0}
    # ROUTE 2 is gated too, so the finish path queues nothing and never reaches
    # the selector. And the selector on this class is the PATCHED one, because
    # `_install_with_metadata()` ran the full all-or-nothing install - the state
    # an armed tier is required to be in (`test_the_disarm_gates_both_routes...`
    # proves the gate itself with the UNPATCHED selector in place).
    assert cls.__dict__["_queue_finished_eager_store"](self, None, ()) is None
    assert calls["finished"] == 0, "disarmed must not queue a finished-store"
    selector = cls.__dict__["_select_eager_blocks_to_store"]
    assert getattr(selector, backport.OWNER_ATTR, None) == backport.OWNER_TAG


def test_the_disarm_gates_the_finished_request_store_path_too():
    """The kill switch covers BOTH callers of the crashing selector, by contract.

    `request_finished_all_groups` -> `_queue_finished_eager_store`
    (`manager.py:1166`) -> `_select_eager_blocks_to_store` (`manager.py:1177`) is
    the second route, and it is wrapped exactly like the per-step trio. Behaviour
    is proven in `test_disarmed_never_calls_the_selector...`; this test pins the
    contract so a future edit that drops the method from the switch fails here
    first, and keeps `disarm_limits` honest about what it now covers.
    """
    assert "_queue_finished_eager_store" in backport.SWITCH_METHODS
    assert set(backport._SWITCH_FACTORIES) == set(backport.SWITCH_METHODS)
    limits = backport.disarm_limits()
    assert limits["finished_request_store_path"].startswith("GATED")
    assert "manager.py:1177" in limits["finished_request_store_path"]
    # The gate point is the INNER method: the outer one returns the tuple that
    # drives delayed freeing, so it must never be the gate.
    assert "request_finished_all_groups" not in backport.SWITCH_METHODS
    assert "connector-creation" in limits["enforced_by_wrappers_fitted_at"]


def test_the_payload_says_why_an_idle_tier_is_expected():
    """Inert-at-boot is a design decision; the JSON has to explain it.

    Otherwise the first thing an operator concludes from `armed: false` on a
    tier-env boot is that the tier is broken - the exact confusion this design
    buys with its safety.
    """
    _install_with_metadata()
    installed_note = backport.section()["operator_note"]
    assert "boot default" in installed_note and "arm" in installed_note

    backport.set_armed(True)
    assert backport.section()["operator_note"].startswith("ARMED")

    backport.reset_state_for_tests()
    drifted = _switch_standin_methods(
        _drifted_fork_class("_select_eager_blocks_to_store"))
    backport.install(drifted)
    note = backport.section()["operator_note"]
    assert "NOT fully installed" in note
    assert "expected" in note


def test_the_tier_verb_reports_its_refusal_kind_and_next_step():
    """`applied: false` alone is not an explanation; the payload carries one."""
    cls = _switch_standin_methods(_drifted_fork_class("update_state_after_alloc"))
    engine = _FakeEngine(_live_manager(cls))
    arm = backport.tier_for_engine(engine, {"op": "arm"})
    assert arm["applied"] is False
    assert arm["armed"] is False
    assert arm["boot_default_armed"] is False
    assert arm["state"]["kind"] == "incomplete_install"
    assert "update_state_after_alloc" in arm["reason"]
    assert "arm" in arm["next_step"]
    assert "DISARMED" in arm["operator_note"]


def test_the_route_validates_op_before_the_engine_sees_it():
    from kvtier_dev_plugin import endpoint
    assert endpoint.validate_arguments(config.ROUTE_TIER, {"op": "disarm"}) == \
        {"op": "disarm"}
    with pytest.raises(endpoint._ArgumentError):
        endpoint.validate_arguments(config.ROUTE_TIER, {"op": "explode"})
    with pytest.raises(endpoint._ArgumentError):
        endpoint.validate_arguments(config.ROUTE_TIER, {"armed": True})


# --- 4. the geometry guard (MANDATORY) --------------------------------------

def test_guard_passes_on_our_real_six_group_shape():
    cfg = our_shape_config()
    verdict = backport.geometry_guard(cfg)
    assert verdict["checked"] is True
    assert verdict["ok"] is True, verdict
    assert verdict["skipped"] == [RING_GROUP]
    assert verdict["cacheable_skipped"] == []


def test_guard_refuses_when_a_skipped_group_is_prefix_cacheable():
    """The trap: omitting a CACHEABLE group would silently lose KV on resume."""
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    # Group 5 is omitted from prefix_cacheable_group_ids but its SPEC still says
    # prefix_cacheable=True -- exactly the config the patch must never accept.
    groups.append(_Group(HASH_BLOCK_SIZE, True, "mistakenly_omitted"))
    cfg = _Config(groups, PCIDS)

    verdict = backport.geometry_guard(cfg)
    assert verdict["ok"] is False
    assert verdict["cacheable_skipped"][0]["group"] == RING_GROUP
    assert verdict["cacheable_skipped"][0]["layers"] == ["mistakenly_omitted"]


def test_install_refuses_and_names_the_cacheable_group():
    cls, _ = FX.build_fork_class(selector_namespace())
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(HASH_BLOCK_SIZE, True, "group_five_is_cacheable"))
    cfg = _Config(groups, PCIDS)

    report = backport.install(cls, kv_cache_config=cfg)
    assert report["installed"] is False
    assert "prefix_cacheable=True" in report["reason"]
    assert "group_five_is_cacheable" in report["reason"]
    # Refusal means refusal: nothing may be patched on the way out.
    assert all(v["patched"] is False for v in report["methods"].values())
    assert backport.OWNER_ATTR not in getattr(
        cls.__dict__["_select_eager_blocks_to_store"], "__dict__", {})


def test_guard_with_no_config_is_reported_unchecked_not_ok():
    verdict = backport.geometry_guard(None)
    assert verdict["checked"] is False and verdict["ok"] is False
    assert "no kv_cache_config" in verdict["reason"]


def test_guard_on_a_pure_prefix_cacheable_config_skips_nothing():
    cfg = _Config([_Group(3200, True) for _ in range(3)])
    verdict = backport.geometry_guard(cfg)
    assert verdict["ok"] is True and verdict["skipped"] == []


# --- 5. the patched code actually fixes the geometry ------------------------

def test_unpatched_selector_hits_the_production_assertion():
    """Regression demonstration: the fork's OWN text crashes on our shape."""
    cls, _ = FX.build_fork_class(selector_namespace())
    self = _SelectorSelf()
    with pytest.raises(AssertionError) as exc:
        drive(cls, self)
    assert "not divisible by" in str(exc.value)
    assert "tokens_per_block=8" in str(exc.value)


def test_patched_selector_never_selects_a_ring_block():
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    assert backport.install(cls)["installed"] is True

    self = _SelectorSelf()
    (selected, advanced, block_meta), block_ids, ring_ids = drive(cls, self)

    assert block_ids[RING_GROUP] and ring_ids
    assert not (set(selected) & ring_ids), (
        f"a ring block was selected for offload: {set(selected) & ring_ids}")
    # The cacheable groups still work: their first block is selected.
    for g in range(CACHEABLE_GROUPS):
        assert block_ids[g][0] in selected, g
    assert advanced[RING_GROUP] == 0, "the ring cursor must not advance"
    assert advanced[:CACHEABLE_GROUPS] == [2] * CACHEABLE_GROUPS
    assert block_meta is None  # events off in this fake


def test_patched_selector_survives_an_unfilterable_ring_manager():
    """`CircularBufferManager` reports positionally-stable blocks (inherited True).

    The pre-patch loop had ONLY that filter, so the ring reached
    `resolve_block_hashes`. With the manager reporting stable blocks the patched
    group-id filter is the only thing standing between us and the assert.
    """
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    backport.install(cls)
    self = _SelectorSelf()
    self.cpu_coordinator.single_type_managers[RING_GROUP] = _Manager(True)
    (selected, _adv, _meta), _ids, ring_ids = drive(cls, self)
    assert not (set(selected) & ring_ids)


def test_patched_selector_still_filters_genuinely_unstable_groups():
    """The patch must not have widened what gets stored."""
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    backport.install(cls)
    self = _SelectorSelf()
    self.cpu_coordinator.single_type_managers[0] = _Manager(False)
    (selected, advanced, _meta), block_ids, _ring = drive(cls, self)
    assert block_ids[0][0] not in selected, "an unstable group must stay skipped"
    assert advanced[0] == 0


def test_guard_and_selector_agree_on_the_ring_group():
    """The guard's skip set and the selector's skip set are the same set."""
    cfg = our_shape_config()
    guard = backport.geometry_guard(cfg)
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    backport.install(cls)
    self = _SelectorSelf()
    (_sel, advanced, _m), block_ids, ring_ids = drive(cls, self)
    skipped_by_selector = {g for g in range(len(GROUP_BLOCK_SIZES))
                           if advanced[g] == 0 and g not in PCIDS}
    assert set(guard["skipped"]) == skipped_by_selector == {RING_GROUP}


# --- 6. containment ---------------------------------------------------------

def test_install_never_raises_on_a_class_that_is_not_the_manager():
    class NotTheManager:
        pass

    report = backport.install(NotTheManager)
    assert report["installed"] is False
    assert report["reason"]


def test_install_with_no_class_is_a_clean_refusal():
    report = backport.install(None)
    assert report["installed"] is False
    assert "SimpleCPUOffloadScheduler" in report["reason"]


def test_errors_are_counted_not_raised(monkeypatch, reset_snapshot_state):
    def boom(fn):
        raise RuntimeError("no source for you")

    monkeypatch.setattr(inspect, "getsource", boom)
    cls, _ = FX.build_fork_class(selector_namespace())
    report = backport.install(cls)
    assert report["installed"] is False
    assert snapshot._state["errors"] > 0



# --- 5. THE BOOT INVARIANT ---------------------------------------------------
#
# Two claims, both about the boot rather than about a verb:
#
#   (a) while the tier is DISARMED, `_select_eager_blocks_to_store` is never
#       called - by EITHER of its two callers in the fork - even when the fix is
#       NOT installed and the selector is the unpatched one that contains the
#       production assert;
#   (b) the patch lands when the connector is BUILT, so a tier boot is patched
#       before its first request without anyone curling a verb.
#
# Everything below drives the fork's own method text through the real install
# path. The poison-pill `resolve_block_hashes` reproduces the production assert,
# and the control test proves the six-group shape reaches it when the gate is not
# there - which is what makes (a) a claim about the gate, not about the test.

class TransferMeta:
    """The fork's `TransferMeta(gpu_block_ids, cpu_block_ids, block_meta)`."""

    def __init__(self, gpu_block_ids, cpu_block_ids, block_meta=None):
        self.gpu_block_ids = gpu_block_ids
        self.cpu_block_ids = cpu_block_ids
        self.block_meta = block_meta


def _prepare_store_specs(self, scheduler_output):
    """`manager.py:552`, in the eager mode `lazy_offload=False` gives us: walk the
    store states `update_state_after_alloc` registered, call the selector for
    each. The store-spec arithmetic is not under test; the CALL CHAIN is."""
    selected: list[int] = []
    for state in list(self._reqs_to_store.values()):
        gpu_ids, advanced, _meta = self._select_eager_blocks_to_store(
            state, self._store_block_ids)
        for g, n in enumerate(advanced):
            state.num_stored_blocks[g] += n
        selected.extend(gpu_ids)
    return selected, [], [], None


def _build_connector_meta(self, scheduler_output):
    """`manager.py:488`: the store specs, then the finished-store queue."""
    store_gpu, _cpu, _reqs, _meta = self.prepare_store_specs(scheduler_output)
    return {"store_gpu_blocks": list(store_gpu),
            "finished": list(self._pending_finished_stores)}


def _queue_finished_eager_store(self, request, block_ids):
    """The fork's body (manager.py:1169-1195), structure unedited.

    The selector call is the FIRST thing after the state/pool check, so this is
    the route's real exposure: a registered store state plus a bound GPU pool is
    all it takes.
    """
    state = self._reqs_to_store.get(request.request_id)
    gpu_pool = self._gpu_block_pool
    if state is None or gpu_pool is None:
        return
    gpu_ids, _, block_meta = self._select_eager_blocks_to_store(state, block_ids)
    partial_tail = self._find_fa_partial_tail_source(request, block_ids)
    if (partial_tail is not None
            and partial_tail not in gpu_ids
            and len(gpu_ids) < self.cpu_block_pool.get_num_free_blocks()):
        gpu_ids.append(partial_tail)
        if block_meta is not None:
            block_meta.append({})
    if not gpu_ids:
        return
    cpu_blocks = self.cpu_block_pool.get_new_blocks(len(gpu_ids))
    self._pending_finished_stores.append(
        TransferMeta(gpu_ids, [b.block_id for b in cpu_blocks], block_meta))
    self._in_flight_store_gpu_blocks.update(gpu_ids)
    gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])


def _request_finished(self, request, block_ids):
    """manager.py:1127: always (False, None); GPU blocks are ref-counted."""
    return False, None


def _request_finished_all_groups(self, request, block_ids):
    """manager.py:1161, verbatim in shape: queue, then the REAL tuple."""
    self._queue_finished_eager_store(request, block_ids)
    return self.request_finished(request, block_ids=[])


def _six_group_class(drift=None):
    """Our production geometry (5 x 3200 cacheable + the ring of 8), both routes.

    `drift` names a method to perturb so its fingerprint REFUSES it. Drifting
    `_select_eager_blocks_to_store` reproduces the worst install there is: the
    method that contains the crash is left unpatched, and the kill switch is the
    only thing standing between the finish path and the assert.

    The stand-ins carry the FORK's module name because they stand in for methods
    the fork really defines (`build_connector_meta` is `manager.py:488`, ...): a
    later ownership check by another module of this package - `guard_mode`'s
    "is this the fork's own, or a sibling's?" rule - must see what it would see
    in production, not "test_backport".
    """
    if drift:
        cls = _drifted_fork_class(drift)
    else:
        cls, _ns = FX.build_fork_class(selector_namespace())
    standins = {
        "get_num_new_matched_tokens": lambda self, r, n: (4096, True),
        "has_pending_stores": lambda self: bool(self._pending_finished_stores),
        "_find_fa_partial_tail_source": lambda self, request, block_ids: None,
        "prepare_store_specs": _prepare_store_specs,
        "build_connector_meta": _build_connector_meta,
        "_queue_finished_eager_store": _queue_finished_eager_store,
        "request_finished": _request_finished,
        "request_finished_all_groups": _request_finished_all_groups,
    }
    for name, function in standins.items():
        function.__module__ = backport.TARGET_MODULE
        setattr(cls, name, function)

    module = types.ModuleType(backport.METADATA_MODULE)
    module.SimpleCPUOffloadMetadata = EmptyMetadata
    sys.modules[backport.METADATA_MODULE] = module
    return cls


def _production_manager(cls, request_id="req-1"):
    """A manager instance for this geometry, with both routes' state filled in.

    `cls.__new__(cls)` is the `_live_manager` idiom: the fixture's `__init__` is
    the fork's own 119-line constructor and cannot run without a real config.
    `_reqs_to_store` is pre-populated because `update_state_after_alloc` registers
    it for EVERY scheduled request (`scheduler.py:1228` calls it unconditionally
    and the fork registers before its `num_external_tokens == 0` early return) -
    which is exactly why the finish path is live while the tier is disarmed.
    """
    mgr = cls.__new__(cls)
    helper = _SelectorSelf()
    for attr in ("prefix_cacheable_group_ids", "group_block_sizes",
                 "hash_block_size", "block_size", "cpu_kv_cache_config",
                 "cpu_coordinator", "cpu_block_pool", "enable_kv_cache_events",
                 "_gpu_block_pool"):
        setattr(mgr, attr, getattr(helper, attr))
    mgr._classify_store_candidate = helper._classify_store_candidate
    mgr._cached_gpu_block = helper._cached_gpu_block
    mgr.kv_cache_config = our_shape_config()
    # The remaining constructor-set attributes `update_state_after_alloc`
    # reads: the live-object check requires them because the patched body
    # does (manager.py __init__ sets all three on every real instance).
    mgr._lazy_mode = False
    mgr._pending_cpu_hits = {}
    mgr._reqs_to_load = {}
    state = _State(_Request(2, request_id), len(GROUP_BLOCK_SIZES))
    mgr._reqs_to_store = {request_id: state}
    mgr._store_block_ids, mgr._ring_ids = ring_and_cacheable_ids()
    mgr._pending_finished_stores = []
    mgr._in_flight_store_gpu_blocks = set()
    return mgr


def _spy_the_selector(cls):
    """Count every call to the selector, delegating to whatever it is RIGHT NOW."""
    current = cls.__dict__["_select_eager_blocks_to_store"]
    current = current.__func__ if isinstance(current, staticmethod) else current
    calls: list = []

    def _select_eager_blocks_to_store(self, state, block_ids_by_group):
        calls.append(block_ids_by_group)
        return current(self, state, block_ids_by_group)

    cls._select_eager_blocks_to_store = _select_eager_blocks_to_store
    return calls


def test_disarmed_never_calls_the_selector_on_either_route_unpatched():
    """THE invariant: refused backport, unpatched selector, disarmed -> no call.

    `install()` is the real one, so the wrappers are the real ones; the selector
    is the fork's UNPATCHED text with the production assert inside it. Both
    callers are then driven: `request_finished_all_groups` (route 2, the finish
    path) and `build_connector_meta` (route 1, the per-step path). The spy must
    have seen nothing, and the shapes the scheduler receives must be the
    connector's own "nothing to do".
    """
    cls = _six_group_class(drift="_select_eager_blocks_to_store")
    report = backport.install(cls, our_shape_config())
    assert report["installed"] is False
    assert report["refused_methods"] == ["_select_eager_blocks_to_store"]
    assert report["disarm_reason"] == "incomplete_install"
    # A refused backport still means WRAPPERS PRESENT + DISARMED, all four of them.
    assert all(v["wrapped"] for v in report["switch"].values()), report["switch"]
    assert backport.is_armed() is False

    calls = _spy_the_selector(cls)
    mgr = _production_manager(cls)

    # ROUTE 2 (manager.py:1161 -> :1169 -> :1177). The outer method must still
    # answer with the real tuple: delayed freeing is driven by it.
    assert cls.request_finished_all_groups(mgr, _Request(1, "req-1"),
                                           mgr._store_block_ids) == (False, None)
    # ROUTE 1 (manager.py:488 -> :552 -> :745).
    meta = cls.build_connector_meta(mgr, None)
    assert isinstance(meta, EmptyMetadata)
    assert list(meta.store_gpu_blocks) == []

    assert calls == [], "a disarmed tier must never reach the store selector"
    assert mgr._pending_finished_stores == []
    assert mgr._in_flight_store_gpu_blocks == set()


def test_without_the_gate_this_very_shape_reaches_the_assert():
    """The control that keeps the test above from being vacuous.

    Same drifted class, same manager, NO wrappers fitted at all (a plugin-less
    boot on this geometry): both routes raise the production assertion, so the
    absence of a call in the test above is the GATE's doing.
    """
    cls = _six_group_class(drift="_select_eager_blocks_to_store")
    calls = _spy_the_selector(cls)
    mgr = _production_manager(cls)
    with pytest.raises(AssertionError, match="not divisible by tokens_per_hash"):
        cls.request_finished_all_groups(mgr, _Request(1, "req-1"),
                                        mgr._store_block_ids)
    assert len(calls) == 1
    with pytest.raises(AssertionError, match="not divisible by tokens_per_hash"):
        cls.build_connector_meta(mgr, None)
    assert len(calls) == 2


def test_armed_and_patched_the_selector_is_reachable_on_both_routes():
    """The mirror: same setup, fix installed and tier ARMED -> both routes run.

    And they run the PATCHED selector, which skips the ring group instead of
    asserting on it - so the queue it builds holds cacheable blocks only. One
    fresh manager per route: route 1 advances the store cursor, which would
    otherwise leave route 2 with nothing new to select.
    """
    cls = _six_group_class()
    report = backport.install(cls, our_shape_config())
    assert report["installed"] is True, report
    assert report["geometry"]["checked"] is True and report["geometry"]["ok"] is True
    armed = backport.set_armed(True)
    assert armed["armed"] is True and armed["refused"] is False, armed

    calls = _spy_the_selector(cls)

    first = _production_manager(cls)
    meta = cls.build_connector_meta(first, None)
    assert not isinstance(meta, EmptyMetadata)
    assert meta["store_gpu_blocks"], "armed: route 1 must reach the selector"

    second = _production_manager(cls)
    assert cls.request_finished_all_groups(second, _Request(1, "req-1"),
                                          second._store_block_ids) == (False, None)
    assert len(calls) == 2, "armed: BOTH routes must reach the selector"
    assert second._pending_finished_stores, "armed: the finish path queues"
    # The patched selector skipped the ring: no ring id was queued or stored.
    queued = [gid for t in second._pending_finished_stores
              for gid in t.gpu_block_ids]
    assert not (set(queued) & second._ring_ids), queued
    assert not (set(meta["store_gpu_blocks"]) & first._ring_ids)


# --- 5b. the connector-creation hook: the patch lands at boot ----------------

def _fake_vllm_modules(monkeypatch, cls, with_scheduler=True):
    """A `create_connector` that behaves like the fork's, for a boot test.

    Three names, all resolved through `sys.modules` exactly as the plugin
    resolves them: the factory, the connector the factory builds, and the
    manager the connector's module imports at module scope
    (`simple_cpu_offload_connector.py:21`). `with_scheduler=False` is the worker
    role: `role=WORKER` builds a `worker_handler` and leaves `scheduler_manager`
    None, which is what the hook must tolerate.
    """
    manager_module = types.ModuleType(backport.TARGET_MODULE)
    manager_module.SimpleCPUOffloadScheduler = cls
    monkeypatch.setitem(sys.modules, backport.TARGET_MODULE, manager_module)

    built: list = []

    class SimpleCPUOffloadConnector:
        def __init__(self, config, role, kv_cache_config):
            self.vllm_config = config
            self.role = role
            self.kv_cache_config = kv_cache_config
            self.scheduler_manager = None
            self.worker_handler = None
            if getattr(role, "name", role) == "SCHEDULER":
                if with_scheduler:
                    self.scheduler_manager = _production_manager(cls)
            else:
                self.worker_handler = object()
            built.append(self)

    SimpleCPUOffloadConnector.__module__ = backport.CONNECTOR_MODULE
    connector_module = types.ModuleType(backport.CONNECTOR_MODULE)
    connector_module.SimpleCPUOffloadConnector = SimpleCPUOffloadConnector
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_MODULE, connector_module)

    class KVConnectorFactory:
        @classmethod
        def create_connector(cls, config, role, kv_cache_config):
            return SimpleCPUOffloadConnector(config, role, kv_cache_config)

    factory_module = types.ModuleType(backport.CONNECTOR_FACTORY_MODULE)
    factory_module.KVConnectorFactory = KVConnectorFactory
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_FACTORY_MODULE,
                        factory_module)
    return KVConnectorFactory, SimpleCPUOffloadConnector, built


def test_create_connector_lands_the_patch_before_any_verb(monkeypatch):
    """(b) THE BOOT CLAIM, proven: no HTTP verb appears in this test at all.

    `register()` fits the hook; the fork's own `KVConnectorFactory.create_connector
    (config=..., role=..., kv_cache_config=...)` call - the exact form at
    `scheduler.py:164` and `kv_transfer_state.py:90` - then lands the install on
    the way out. After it returns, the class carries every marker and the
    geometry guard has seen a LIVE config, which a class-only install can never
    do. The tier is still disarmed: patching is not arming.
    """
    cls = _six_group_class()
    factory, connector_cls, built = _fake_vllm_modules(monkeypatch, cls)

    assert backport._state["hook_fitted"] is False
    assert backport.install_connector_hook()["fitted"] is True
    assert backport._state["fully_installed"] is False, "nothing has been built yet"

    role = types.SimpleNamespace(name="SCHEDULER")
    connector = factory.create_connector(config="CFG", role=role,
                                        kv_cache_config="KVC")

    # The engine got exactly the connector it asked for, arguments untouched.
    assert isinstance(connector, connector_cls) and len(built) == 1
    assert connector.vllm_config == "CFG" and connector.role is role
    assert connector.kv_cache_config == "KVC"
    # ...and by the instant create_connector RETURNED, the boot was patched.
    assert backport._state["fully_installed"] is True
    assert backport.already_installed(cls) is True
    assert backport._state["hook_installs"] == 1

    section = backport.section()
    assert section["installed"] is True
    assert section["patched_before_first_request"] is True
    assert section["geometry"]["checked"] is True, "the LIVE config was guarded"
    assert section["geometry"]["ok"] is True
    assert section["armed"] is False, "patching is not arming"
    hook = section["connector_hook"]
    assert hook["fitted"] is True and hook["tier_connectors"] == 1
    assert hook["installs_run"] == 1 and hook["errors"] == 0
    assert all(v["wrapped"] for v in section["switch"].values())
    assert tuple(section["switch_methods"]) == backport.SWITCH_METHODS


def test_the_hook_is_idempotent_and_never_nests(monkeypatch):
    cls = _six_group_class()
    factory, _connector_cls, built = _fake_vllm_modules(monkeypatch, cls)
    for _ in range(3):
        assert backport.install_connector_hook()["fitted"] is True
    assert backport._state["hook_fits"] == 3
    role = types.SimpleNamespace(name="SCHEDULER")
    for _ in range(2):
        factory.create_connector(None, role, None)
    assert len(built) == 2, "every connector the engine asked for was built"
    assert backport._state["hook_tier_connectors"] == 2, \
        "three fits must still be one hook deep"


def test_the_hook_treats_a_worker_connector_as_no_manager(monkeypatch,
                                                          reset_snapshot_state):
    """A worker builds `worker_handler`, never a `scheduler_manager`.

    That is the NORMAL answer in a process that runs no manager code, so it is
    counted as `worker_role_no_manager` and must not be an error, must not patch
    anything, and must still hand the worker its connector.
    """
    cls = _six_group_class()
    factory, connector_cls, _built = _fake_vllm_modules(
        monkeypatch, cls, with_scheduler=False)
    backport.install_connector_hook()
    connector = factory.create_connector(None, types.SimpleNamespace(name="WORKER"),
                                        None)
    assert isinstance(connector, connector_cls)
    assert connector.scheduler_manager is None
    hook = backport.section()["connector_hook"]
    assert hook["worker_role_no_manager"] == 1
    assert hook["tier_connectors"] == 0 and hook["installs_run"] == 0
    assert hook["errors"] == 0 and backport._state["hook_errors"] == 0
    assert snapshot._state["errors"] == 0, "a worker is not an error"
    assert backport._state["fully_installed"] is False


def test_a_hook_failure_cannot_break_connector_construction(monkeypatch,
                                                             reset_snapshot_state):
    cls = _six_group_class()
    factory, connector_cls, _built = _fake_vllm_modules(monkeypatch, cls)
    backport.install_connector_hook()

    def boom(*args, **kwargs):
        raise RuntimeError("install exploded")

    monkeypatch.setattr(backport, "install", boom)
    connector = factory.create_connector(None, types.SimpleNamespace(name="SCHEDULER"),
                                        None)
    assert isinstance(connector, connector_cls), "the engine still gets its tier"
    assert backport._state["hook_errors"] == 1
    assert snapshot._state["errors"] == 1, "contained AND counted"


def test_the_hook_does_not_swallow_the_forks_own_connector_errors(monkeypatch):
    """`create_connector` raises on purpose (a bad kv_transfer_config, HMA off).

    Those ValueErrors are the engine's story. The wrapper calls the fork first
    and catches nothing, so a connector that cannot be built still fails the
    boot exactly as it would without us.
    """
    factory_module = types.ModuleType(backport.CONNECTOR_FACTORY_MODULE)

    class KVConnectorFactory:
        @classmethod
        def create_connector(cls, config, role, kv_cache_config):
            raise ValueError("kv_transfer_config must be set to create a connector")

    factory_module.KVConnectorFactory = KVConnectorFactory
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_FACTORY_MODULE,
                        factory_module)
    assert backport.install_connector_hook()["fitted"] is True
    with pytest.raises(ValueError, match="must be set"):
        KVConnectorFactory.create_connector(None, None, None)
    assert backport._state["hook_errors"] == 0


def test_the_hook_refuses_to_clobber_a_foreign_hook(monkeypatch):
    factory_module = types.ModuleType(backport.CONNECTOR_FACTORY_MODULE)

    class KVConnectorFactory:
        @classmethod
        def create_connector(cls, config, role, kv_cache_config):
            return "connector"

    setattr(KVConnectorFactory.__dict__["create_connector"].__func__,
            backport.HOOK_ATTR, "someone-else/hook-1")
    factory_module.KVConnectorFactory = KVConnectorFactory
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_FACTORY_MODULE,
                        factory_module)

    report = backport.install_connector_hook()
    assert report["fitted"] is False
    assert "another plugin" in report["reason"]
    assert backport._state["hook_fitted"] is False
    assert KVConnectorFactory.create_connector(None, None, None) == "connector"


def test_the_hook_reports_a_factory_that_is_not_imported_yet():
    """An arg-parsing or worker process at plugin-load time: nothing to wrap."""
    saved = sys.modules.pop(backport.CONNECTOR_FACTORY_MODULE, None)
    try:
        report = backport.install_connector_hook()
    finally:
        if saved is not None:
            sys.modules[backport.CONNECTOR_FACTORY_MODULE] = saved
    assert report["fitted"] is False
    assert "not imported" in report["reason"]
    assert backport._state["hook_fitted"] is False
    assert "not imported" in backport.section()["connector_hook"]["reason"]


def test_register_fits_the_hook_at_plugin_load(monkeypatch, enabled):
    """The hook is fitted by `register()`, before the Scheduler exists.

    The precondition the whole design rests on: the process that can see
    `vllm.v1.engine.core` (which is what lets `register()` attach verbs at all)
    already has the connector factory imported - verified against the fork,
    where importing `vllm.v1.engine.core` puts the factory module in
    `sys.modules` while the scheduler and offload-manager modules are absent.
    """
    factory, _connector_cls, _built = _fake_vllm_modules(monkeypatch,
                                                          _six_group_class())
    core_module = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        pass

    core_module.EngineCore = EngineCore
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)

    engine_side.register()                       # exactly what vllm calls at boot
    assert backport._state["hook_fitted"] is True
    assert isinstance(factory.__dict__[backport.CREATE_CONNECTOR], classmethod)
    connector = factory.create_connector(None, types.SimpleNamespace(name="SCHEDULER"),
                                        None)
    assert connector.scheduler_manager is not None
    assert backport._state["hook_installs"] == 1, \
        "the boot patched itself with no verb involved"


# --- 5c. composition with guard mode on the newly gated method ---------------

def test_the_finish_wrapper_survives_guard_mode_wrapping_on_top(reset_dev_state):
    """Guard mode composes ON TOP of the switch; uninstall hands it back exactly.

    `guard_mode.install` treats a sibling package's wrapper as composable (its
    `SIBLING_PACKAGE` rule) rather than as a foreign owner, and `uninstall`
    restores the function it stored - which is our wrapper, by identity. So a
    guard arm/disarm cycle cannot silently remove the kill switch, and while
    guard mode is armed the disarmed tier still declines.
    """
    cls = _six_group_class()
    backport.install(cls, our_shape_config())
    backport.set_armed(False)
    switch_meta = cls.__dict__["build_connector_meta"]
    switch_finish = cls.__dict__["_queue_finished_eager_store"]

    report = guard_mode.install(cls)
    assert report["installed"] is True, report
    # The methods both modules wrap are nested and SAY so; the finish method is
    # not in guard mode's list at all, so our wrapper stays the only layer.
    assert "build_connector_meta" in report["nested"]
    assert "get_num_new_matched_tokens" in report["nested"]
    assert "_queue_finished_eager_store" not in report["attached"]
    assert cls.__dict__["_queue_finished_eager_store"] is switch_finish

    calls = _spy_the_selector(cls)
    mgr = _production_manager(cls)
    assert cls.request_finished_all_groups(mgr, _Request(1, "req-1"),
                                          mgr._store_block_ids) == (False, None)
    assert isinstance(cls.build_connector_meta(mgr, None), EmptyMetadata)
    assert calls == [], "guard mode armed must not re-open a disarmed tier"

    out = guard_mode.uninstall(cls)
    assert "build_connector_meta" in out["restored"]
    assert cls.__dict__["build_connector_meta"] is switch_meta, \
        "uninstall must hand back exactly what it found"
    assert cls.__dict__["_queue_finished_eager_store"] is switch_finish
    assert isinstance(cls.build_connector_meta(mgr, None), EmptyMetadata)
    assert calls == [], "still gated after the guard cycle"


def test_the_switch_fitted_after_guard_mode_composes_and_still_declines(
        reset_dev_state):
    """The other order: guard first, the switch landing UNDER it.

    Since the 2026-10-03 live-object work, `_install_switch` descends its own
    layers: with guard already wrapped, the switch is fitted beneath the guard
    and the guard is rebuilt on top (composition preserved, kill switch still
    the first thing the engine's own code sees). Guard's `uninstall` then
    restores OUR wrapper by identity - a clean unwind to the switch, so the
    gate survives a full guard cycle either way.
    """
    cls = _six_group_class()
    guard_report = guard_mode.install(cls)
    assert "build_connector_meta" in guard_report["attached"]

    backport.install(cls, our_shape_config())
    top = cls.__dict__["build_connector_meta"]
    # Guard OUTSIDE, our switch under it - never the other way round.
    assert getattr(top, guard_mode.OWNER_ATTR, None) == guard_mode.OWNER_TAG
    under = getattr(top, guard_mode.ORIGINAL_ATTR, None)
    assert getattr(under, backport.SWITCH_ATTR, None) == backport.SWITCH_TAG

    calls = _spy_the_selector(cls)
    mgr = _production_manager(cls)
    assert isinstance(cls.build_connector_meta(mgr, None), EmptyMetadata)
    assert calls == []
    assert guard_mode.is_armed() is False

    out = guard_mode.uninstall(cls)
    assert "build_connector_meta" in out["restored"]
    assert cls.__dict__["build_connector_meta"] is under, \
        "unwinding the guard must hand back exactly our switch wrapper"
    assert isinstance(cls.build_connector_meta(mgr, None), EmptyMetadata)


def test_the_wrappers_are_fitted_before_the_backport_is_attempted():
    """THE ORDER, pinned: an install that dies MIDWAY still leaves the gate on.

    The kill switch is fitted FIRST inside `install()` - before the geometry guard
    and before a single upstream body is compiled - so an install that explodes
    part-way through the patch leaves "wrappers present + disarmed", which gates
    BOTH routes to the selector. Fitting the wrappers last would leave this class
    with the finish route wide open and `installed: false`.
    """
    cls = _six_group_class()
    real_install_one = backport._install_one

    def explode_on_the_selector(cls_, name):
        if name == "_estimate_lazy_target_blocks":
            raise RuntimeError("compile exploded halfway through the install")
        return real_install_one(cls_, name)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(backport, "_install_one", explode_on_the_selector)
        report = backport.install(cls, our_shape_config())
        assert report["installed"] is False
        assert "install raised" in report["reason"]

        # Every wrapper was already on the class when the install died.
        for name in backport.SWITCH_METHODS:
            raw = cls.__dict__[name]
            assert getattr(raw, backport.SWITCH_ATTR, None) == backport.SWITCH_TAG, name
        assert backport.is_armed() is False

        calls = _spy_the_selector(cls)
        mgr = _production_manager(cls)
        assert cls.request_finished_all_groups(mgr, _Request(1, "req-1"),
                                              mgr._store_block_ids) == (False, None)
        assert isinstance(cls.build_connector_meta(mgr, None), EmptyMetadata)
        assert calls == [], "a half-installed class must still be gated"
    finally:
        monkeypatch.undo()


def test_a_geometry_veto_still_fits_the_wrappers(reset_dev_state):
    """A vetoed config gets NO patch - and therefore MUST get the gate.

    The guard refuses to patch a config whose skipped group is still
    prefix-cacheable, so the selector on this class stays the crashing one. That
    is survivable only because the wrappers are fitted before the guard runs: a
    vetoed engine is a no-op on both routes, not a crash waiting for a request.
    """
    cls = _six_group_class()
    groups = [_Group(HASH_BLOCK_SIZE, True, f"layer_{g}")
              for g in range(CACHEABLE_GROUPS)]
    groups.append(_Group(HASH_BLOCK_SIZE, True, "group_five_is_cacheable"))
    report = backport.install(cls, _Config(groups, PCIDS))
    assert report["installed"] is False
    assert report["disarm_reason"] == "geometry"
    assert all(v["wrapped"] for v in report["switch"].values()), report["switch"]

    calls = _spy_the_selector(cls)
    mgr = _production_manager(cls)
    assert cls.request_finished_all_groups(mgr, _Request(1, "req-1"),
                                          mgr._store_block_ids) == (False, None)
    assert isinstance(cls.build_connector_meta(mgr, None), EmptyMetadata)
    assert calls == []
    assert backport.set_armed(True)["kind"] == "geometry"
