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
import sys
import types

import fixtures_pr56810 as FX
import pytest

from kvtier_dev_plugin import backport, config, pr56810_sources, snapshot

# --- the geometry that broke production ------------------------------------
HASH_BLOCK_SIZE = 3200          # five groups of 3200 ...
CACHEABLE_GROUPS = 5
RING_GROUP = 5                  # ... plus one QSA ring of 8
RING_BLOCK_SIZE = 8             # NOT divisible by 3200 -> the assert
GROUP_BLOCK_SIZES = [HASH_BLOCK_SIZE] * CACHEABLE_GROUPS + [RING_BLOCK_SIZE]
PCIDS = tuple(range(CACHEABLE_GROUPS))


@pytest.fixture(autouse=True)
def reset_backport_state():
    """`backport._state` is process-wide, like every other piece of plugin state."""
    saved = dict(backport._state)
    saved_verdicts = dict(backport._state["method_verdicts"])
    saved_switch = dict(backport._state["switch_verdicts"])
    backport._state.update({
        "installs": 0, "reinstalls": 0, "skips": 0, "skip_reason": None,
        "class": None, "armed": True, "disarm_calls": 0, "arm_calls": 0,
        "fallbacks": 0, "last_error": None, "geometry": None,
    })
    backport._state["method_verdicts"] = {}
    backport._state["switch_verdicts"] = {}
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


class _GpuBlock:
    def __init__(self, block_id, block_hash):
        self.block_id = block_id
        self.is_null = False
        self.block_hash = block_hash


class _GpuPool:
    def __init__(self, blocks):
        self.blocks = blocks


class _Request:
    def __init__(self, num_blocks_of_3200):
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
    """A freshly rebuilt class carrying the three switch methods, NOT installed.

    The fixture rebuilds only the four PR #56810 methods, so the three the kill
    switch wraps get stand-in originals here. They are deliberately loud: each
    records that it ran, which is how the tests prove a disarmed connector never
    reaches the engine's own code. A NEW class every call, so a test can watch
    the first install happen.
    """
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    calls = {"matched": 0, "meta": 0, "pending": 0}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        calls["matched"] += 1
        return 4096, True

    def build_connector_meta(self, scheduler_output):
        calls["meta"] += 1
        return "ENGINE METADATA"

    def has_pending_stores(self):
        calls["pending"] += 1
        return True

    cls.get_num_new_matched_tokens = get_num_new_matched_tokens
    cls.build_connector_meta = build_connector_meta
    cls.has_pending_stores = has_pending_stores

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
    assert backport.is_armed() is True
    assert backport.set_armed(False)["changed"] is True
    assert backport.set_armed(False)["changed"] is False, "a no-op must not count"
    assert backport.is_armed() is False
    assert backport.section()["armed"] is False


def test_switch_is_independent_of_the_backport_fingerprints(monkeypatch):
    """A stale fingerprint must not also disarm the operator's kill switch."""
    ns = selector_namespace()
    cls, _ = FX.build_fork_class(ns)
    cls.get_num_new_matched_tokens = lambda self, r, n: (1, True)
    cls.build_connector_meta = lambda self, so: "ENGINE METADATA"
    cls.has_pending_stores = lambda self: True
    monkeypatch.setattr(backport, "fingerprint_of", lambda fn: None)
    report = backport.install(cls)
    assert report["installed"] is False, "fingerprints were poisoned"
    assert all(v["wrapped"] for v in report["switch"].values()), report["switch"]
    backport.set_armed(False)
    assert cls.__dict__["has_pending_stores"](object()) is False
    assert cls.__dict__["get_num_new_matched_tokens"](object(), None, 0) == (0, False)
    backport.set_armed(True)
    assert cls.__dict__["get_num_new_matched_tokens"](object(), None, 0) == (1, True)


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
    """
    manager = cls.__new__(cls)
    manager.kv_cache_config = config if config is not None else our_shape_config()
    return manager


def _engine_with_live_manager():
    cls, _meta, calls = _install_with_metadata()
    return _FakeEngine(_live_manager(cls)), cls, calls


def test_tier_verb_disarms_through_the_engine_and_reports_it():
    engine, _cls, calls = _engine_with_live_manager()
    status = backport.tier_for_engine(engine, {"op": "status"})
    assert status["status_code"] == 200
    assert status["tier"]["armed"] is True
    assert status["tier"]["patched"] is True

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
    assert backport.is_armed() is True
    out = backport.tier_for_engine(engine, {"op": "dissarm"})
    assert out["status_code"] == 400
    assert out["applied"] is False
    assert "Nothing was changed" in out["reason"]
    assert backport.is_armed() is True, "a refused op must not disarm the tier"


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

