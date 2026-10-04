"""The GPU-pool census: the split the usage gauge cannot make, and its limits.

The claim under test, from the fork: `BlockPool.get_usage()` is
`1 - free/(total-1)` and `free` is the count of `ref_cnt == 0` blocks, while a block
sitting in the free queue that still carries a `block_hash` is STILL matchable
(`get_cached_block`, block_pool.py:204, checks no refcount) until
`_maybe_evict_cached_block` (:726) discards it. So the gauge's "free" is a MIX of
"matchable but idle" and "genuinely gone", and nothing in the stack separated them.

Everything here runs on stubs built from the fork's attribute names
(`fakes_debug`), because the census is pure attribute reads - which is the same
reason the rest of this suite needs no vllm.

The properties pinned, in order:
  * the four-way split, its identity, and its agreement with the engine's own
    free-queue counter;
  * `cached_idle` being exactly the part of the gauge's free that is still matchable;
  * per-group maps that PARTITION the totals (a block carries one primary hash, so
    exactly one group), and the one thing that is NOT derivable: per-group unhashed
    counts;
  * group labels read off the live config - never hardcoded indices;
  * every failure mode refusing rather than reporting a partial count or a zero;
  * a census failure being unable to escape into the engine, and a
    `BaseException` NOT being swallowed;
  * the walk being O(blocks) and side-effect-free.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import fakes_debug as fd
import pytest

from kvtier_dev_plugin import gpu_census, snapshot

SRC = Path(gpu_census.__file__).resolve()
TREE = ast.parse(SRC.read_text(encoding="utf-8"))

_DEFAULT = object()   # "give me the fixture's cache_config", distinct from "it is gone"


# --- the AST-pinned field contract (the test_monitor_schema.py idiom) --------

def _module_constant(name: str):
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name and node.value:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level literal in {SRC.name}")


def _dict_literals(node):
    if isinstance(node, ast.Dict):
        yield node
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            yield child


def _builder_keys(func_name: str) -> set:
    """Every literal key the `_fields` dict assigns, read off the AST."""
    func = next((n for n in TREE.body
                 if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    assert func is not None, f"{func_name} is missing from {SRC.name}"
    keys: set = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for mapping in _dict_literals(node.value):
            for key in mapping.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


# Fields that carry prose or a constant rather than a measurement, so a refusal
# legitimately leaves them populated.
# Every field that would carry a MEASUREMENT: a refusal must leave all of them null.
COUNT_FIELDS = (
    "walked_blocks", "blocks_total", "walk_us", "referenced", "referenced_hashed",
    "referenced_unhashed", "cached_idle", "free_uncached", "null_blocks",
    "negative_ref_cnt_blocks", "hashed_per_group", "hashed_idle_per_group",
    "hashed_referenced_per_group", "hashed_groups_unattributed", "groups_truncated",
    "unhashed_per_group",
    "free_queue_reported", "free_queue_reconciles", "sum_matches_walk",
    "walked_matches_pool_size", "joint_hashed_blocks_min", "pool_class",
    "pool_enable_caching",
)


def test_the_refusal_fields_are_all_measurements_so_a_refusal_cannot_half_lie():
    """The never-zero rule has teeth only if the refusal covers every count field."""
    assert set(COUNT_FIELDS) <= set(_module_constant("EXPECTED_FIELDS"))
    emitted = _builder_keys("_fields")
    measured = {key for key in emitted
                if not key.endswith(("reason", "meaning", "schema_version",
                                     "pure_read", "walked_bound", "available"))
                and key not in ("error", "group_labels", "retention", "pool_source")}
    escaped = measured - set(COUNT_FIELDS)
    assert escaped == set(), \
        f"a new count field escaped the refusal test: {sorted(escaped)}"


def test_every_emitted_key_is_declared_and_every_declared_key_is_emitted():
    declared = set(_module_constant("EXPECTED_FIELDS"))
    allowed_unemitted = set(_module_constant("ALLOW_UNEMITTED"))
    emitted = _builder_keys("_fields")
    assert emitted - declared == set(), \
        f"emitted but undeclared: {sorted(emitted - declared)}"
    assert declared - emitted - allowed_unemitted == set(), \
        f"declared but never emitted: {sorted(declared - emitted)}"


def test_the_degraded_view_carries_the_healthy_key_set_exactly():
    """A consumer must never branch on structure: absent means null, not missing."""
    healthy = gpu_census.census(_pool(cached_idle=2, free_uncached=1))
    degraded = gpu_census.empty_section("a reason")
    assert set(healthy) == set(degraded) == set(_module_constant("EXPECTED_FIELDS"))
    assert degraded["schema_version"] == _module_constant("SCHEMA_VERSION") == 1


# --- stubs ------------------------------------------------------------------

class Block:
    """`KVCacheBlock`: the three attributes the census reads, nothing else."""

    def __init__(self, block_id, block_hash=None, ref_cnt=0, is_null=False):
        self.block_id = block_id
        self.block_hash = block_hash
        self.ref_cnt = ref_cnt
        self.is_null = is_null


class Pool:
    """`BlockPool`: `blocks`, `num_gpu_blocks`, the maintained free counter."""

    def __init__(self, blocks, free=None, total=None, enable_caching=True):
        self.blocks = list(blocks)
        self.num_gpu_blocks = len(self.blocks) if total is None else total
        self.enable_caching = enable_caching
        self._free = len(self.blocks) if free is None else free
        self.calls = []

    def get_num_free_blocks(self):
        self.calls.append("get_num_free_blocks")
        return self._free


class Engine:
    """`EngineCore` -> `scheduler` -> the three handles the census resolves.

    `fd.Scheduler` is the fixture's class LITERALLY named `Scheduler`, so
    `type(scheduler).__name__` matches the fork's; `fd.Obj(name="Scheduler")` is not
    used here because a registered name takes the registered class's own
    `__init__`, which drops the extra fields.
    """

    def __init__(self, pool=None, groups=None, config=_DEFAULT, tier_pool=None):
        manager = None if pool is None else fd.Obj(name="KVCacheManager",
                                                   block_pool=pool)
        connector = None
        if tier_pool is not None:
            connector = fd.Obj(
                name="SimpleCPUOffloadConnector",
                scheduler_manager=fd.Obj(
                    name="SimpleCPUOffloadScheduler", _gpu_block_pool=tier_pool))
        self.scheduler = fd.Scheduler(
            kv_cache_manager=manager,
            connector=connector,
            kv_cache_config=None if groups is None else fd.kv_cache_config(groups),
            cache_config=fd.cache_config() if config is _DEFAULT else config)


def hash_key(seed=1, group=0):
    """A `BlockHashWithGroupId`: hash bytes then 4 big-endian group bytes."""
    return bytes([seed % 251] * 20) + group.to_bytes(4, "big")


def _pool(*, cached_idle=0, free_uncached=0, referenced=0, referenced_hashed=0,
          null=True, free=None, group_cycle=6):
    """A pool whose composition is the test's argument, in readable units."""
    blocks = [Block(0, is_null=True)] if null else []
    index = 1
    for count in range(cached_idle):
        blocks.append(Block(index, block_hash=hash_key(index, count % group_cycle),
                            ref_cnt=0))
        index += 1
    for _ in range(free_uncached):
        blocks.append(Block(index, ref_cnt=0))
        index += 1
    for _ in range(referenced):
        blocks.append(Block(index, ref_cnt=2))       # being written: not yet hashed
        index += 1
    for count in range(referenced_hashed):
        blocks.append(Block(index, block_hash=hash_key(index, count % group_cycle),
                            ref_cnt=1))
        index += 1
    # The engine's own counter: every ref_cnt == 0 block except the null placeholder.
    return Pool(blocks, free=cached_idle + free_uncached if free is None else free)


# --- the split, and the identity that keeps it honest -----------------------

def test_the_four_buckets_partition_the_pool_and_add_up_to_it():
    section = gpu_census.census(_pool(cached_idle=7, free_uncached=3, referenced=2,
                                      referenced_hashed=4))
    assert section["available"] is True
    assert section["cached_idle"] == 7
    assert section["free_uncached"] == 3
    assert section["referenced"] == 6                 # ref_cnt != 0, hashed or not
    assert section["referenced_hashed"] == 4
    assert section["referenced_unhashed"] == 2
    assert section["null_blocks"] == 1
    assert section["walked_blocks"] == 1 + 7 + 3 + 2 + 4
    assert section["sum_matches_walk"] is True
    assert section["walked_matches_pool_size"] is True


def test_cached_idle_is_the_half_of_the_gauges_free_that_can_still_match():
    """56 blocks: 50 of them sit in the free queue, 40 of those are still matchable.

    Asserted against the gauge's OWN formula (block_pool.py:876-888), so this pins
    that the census is the missing decomposition of the number already on the panel -
    not a new, unrelated quantity.
    """
    pool = _pool(cached_idle=40, free_uncached=10, referenced=5)
    section = gpu_census.census(pool)
    free = pool.get_num_free_blocks()
    gauge_free = 1.0 - free / (pool.num_gpu_blocks - 1)
    assert pool.num_gpu_blocks == 56 and free == 50
    assert section["free_queue_reported"] == 50
    assert section["cached_idle"] + section["free_uncached"] == free
    assert section["free_queue_reconciles"] is True
    # The gauge says the pool is 9.1% free. 4/5 of that "free" is prefix data.
    assert gauge_free == pytest.approx(1 - 50 / 55)   # 9.1% "free"
    assert section["cached_idle"] / free == pytest.approx(0.8)


def test_a_lying_free_counter_is_published_as_a_disagreement_not_hidden():
    pool = _pool(cached_idle=4, free_uncached=1, free=99)
    section = gpu_census.census(pool)
    assert section["free_queue_reported"] == 99
    assert section["free_queue_reconciles"] is False


def test_the_null_block_is_counted_apart_and_never_as_free():
    """Its refcount is deliberately unmaintained (block_pool.py:190-192)."""
    section = gpu_census.census(_pool(cached_idle=2, free_uncached=1))
    assert section["null_blocks"] == 1
    assert section["free_uncached"] == 1        # not 2: the placeholder is excluded
    assert section["free_queue_reconciles"] is True


# --- per group: what a block knows, and what it does not --------------------

def test_per_group_maps_partition_the_totals_and_share_one_key_set():
    section = gpu_census.census(_pool(cached_idle=12, free_uncached=4,
                                      referenced_hashed=5, group_cycle=3))
    hashed = section["hashed_per_group"]
    idle = section["hashed_idle_per_group"]
    referenced = section["hashed_referenced_per_group"]
    assert sorted(hashed) == sorted(idle) == sorted(referenced) == ["0", "1", "2"]
    assert sum(hashed.values()) == section["cached_idle"] + section["referenced_hashed"]
    assert sum(idle.values()) == section["cached_idle"] == 12
    assert sum(referenced.values()) == section["referenced_hashed"] == 5
    assert all(hashed[key] == idle[key] + referenced[key] for key in hashed)
    assert section["per_group_reason"] is None


def test_per_group_attribution_is_the_hash_tail_not_a_block_id_range():
    """Group id = the last 4 big-endian bytes (kv_cache_utils.py:74-93)."""
    blocks = [Block(1, block_hash=hash_key(1, 5), ref_cnt=0),
              Block(2, block_hash=hash_key(2, 5), ref_cnt=0),
              Block(3, block_hash=hash_key(3, 2), ref_cnt=0)]
    section = gpu_census.census(Pool(blocks, free=3, total=3))
    assert section["hashed_idle_per_group"] == {"2": 1, "5": 2}
    assert section["cached_idle"] == 3


def test_per_group_unhashed_counts_are_refused_with_a_reason_and_never_invented():
    """The honest limit of the data structure.

    All groups allocate from ONE shared `BlockPool` (kv_cache_coordinator.py:98) and
    an unhashed block carries no group, so "is the ring group's share 100% unhashed?"
    is unanswerable from here. A null with the reason is the answer; a zero or a
    per-group guess would become the number somebody believes.
    """
    section = gpu_census.census(_pool(cached_idle=3, free_uncached=9))
    assert section["free_uncached"] == 9
    assert section["unhashed_per_group"] is None
    assert "shared BlockPool" in section["unhashed_per_group_reason"]
    assert "popleft_n" in section["unhashed_per_group_reason"]


def test_a_hash_that_decodes_to_no_group_refuses_the_maps_rather_than_summing_short():
    pool = Pool([Block(1, block_hash=b"\x01\x02", ref_cnt=0),        # < 4-byte tail
                 Block(2, block_hash=hash_key(2, 1), ref_cnt=0)], free=2, total=2)
    section = gpu_census.census(pool)
    assert section["cached_idle"] == 2                     # the totals stay true
    assert section["hashed_groups_unattributed"] == 1
    assert section["hashed_per_group"] is None
    assert section["hashed_idle_per_group"] is None
    assert "cannot decode" in section["per_group_reason"]


def test_more_groups_than_the_bound_refuses_the_maps_and_says_so(monkeypatch):
    monkeypatch.setattr(gpu_census, "MAX_GROUPS", 2)
    blocks = [Block(index, block_hash=hash_key(index, index), ref_cnt=0)
              for index in range(5)]
    section = gpu_census.census(Pool(blocks, free=5, total=5))
    assert section["groups_truncated"] is True
    assert section["hashed_per_group"] is None
    assert "subset" in section["per_group_reason"]


# --- labels and retention: derived from the live config, never hardcoded -----

def test_group_labels_come_from_the_spec_class_names_and_name_the_ring_by_mro():
    engine, info = fd.production_geometry()
    section = gpu_census.section(engine)
    labels = section["group_labels"]
    assert section["group_labels_reason"] is None
    # The fixture's indices: a Mamba group, and a QSA ring that is a SUBCLASS of
    # CircularBufferSpec - recognised by the MRO walk, which is how the fork's own
    # is_circular_buffer() decides.
    assert labels[str(info["mamba_index"])]["spec_class"] == "MambaSpec"
    assert labels[str(info["ring_index"])]["spec_class"] == "QsaRingSpec"
    assert labels[str(info["ring_index"])]["is_circular_buffer"] is True
    assert labels[str(info["ring_index"])]["spec"]["prefix_cacheable"] is False
    assert labels["0"]["spec"]["prefix_cacheable"] is True
    assert labels["0"]["block_size"] == labels["0"]["spec"]["block_size"]


def test_the_joint_min_excludes_the_non_cacheable_ring_group():
    engine, info = fd.production_geometry()
    section = gpu_census.section(engine)
    per_group = section["hashed_per_group"]
    ring = str(info["ring_index"])
    cacheable = [value for key, value in per_group.items() if key != ring]
    assert section["joint_hashed_blocks_min"] == min(cacheable)
    assert "UPPER BOUND" in section["joint_hashed_blocks_min_meaning"]
    assert "NOT a hit length" in section["joint_hashed_blocks_min_meaning"]


def test_retention_carries_the_knobs_that_explain_a_sparse_state_group():
    engine, info = fd.production_geometry()
    engine.scheduler.kv_cache_config.prefix_cache_retention_interval = 0
    engine.scheduler.cache_config.mamba_cache_mode = "aligned"
    section = gpu_census.section(engine)
    knobs = section["retention"]
    assert knobs["kv_cache_config.prefix_cache_retention_interval"] == 0
    assert knobs["mamba_cache_mode"] == "aligned"
    assert section["retention_reason"] is None


def test_a_missing_config_labels_the_census_as_unlabelled_not_as_group_zero():
    """A fork that renamed the config must leave the labels ABSENT, not zero."""
    section = gpu_census.section(Engine(pool=_pool(cached_idle=2),
                                        groups=None, config=None))
    assert section["available"] is True
    assert section["group_labels"] is None
    assert "kv_cache_groups is missing" in section["group_labels_reason"]
    assert section["joint_hashed_blocks_min"] is None
    assert section["retention"] is None
    assert section["retention_reason"] == "scheduler exposes no cache_config"


# --- the pool handle --------------------------------------------------------

def test_the_pool_is_reached_through_either_spelling_and_says_which():
    """The GPU pool belongs to the scheduler; the tier only borrows it.

    Both spellings name the SAME instance in the fork (base.py:475-476 binds
    `kv_cache_manager.block_pool`, which is `coordinator.block_pool`, onto the
    connector as `_gpu_block_pool`), so either one must work and the payload must
    name the one it used.
    """
    pool = _pool(cached_idle=2, free_uncached=1)
    tier_only = Engine(pool=None, tier_pool=pool)
    section = gpu_census.section(tier_only)
    assert section["available"] is True
    assert section["pool_source"] == \
        "scheduler.connector.scheduler_manager._gpu_block_pool"
    assert gpu_census.section(Engine(pool=pool))["pool_source"] == \
        "scheduler.kv_cache_manager.block_pool"


def test_no_pool_at_all_is_a_refusal_with_nulls_and_never_a_zero():
    section = gpu_census.section(Engine(pool=None))
    assert section["available"] is False
    assert "no GPU block pool bound" in section["reason"]
    for field in COUNT_FIELDS:
        assert section[field] is None, f"{field} invented {section[field]!r}"


# --- failure containment: refuse, never partially count, never raise ---------

def test_a_pool_without_blocks_refuses_instead_of_guessing_where_the_blocks_are():
    section = gpu_census.census(fd.Obj(name="BlockPool", num_gpu_blocks=10))
    assert section["available"] is False
    assert "no `blocks`" in section["reason"]
    assert section["cached_idle"] is None


def test_a_blocks_attribute_that_is_not_a_list_is_refused_not_iterated():
    section = gpu_census.census(fd.Obj(name="BlockPool", blocks=42, num_gpu_blocks=42))
    assert section["available"] is False
    assert "not a list" in section["reason"]


def test_a_block_the_walk_cannot_classify_refuses_the_whole_census(
        reset_snapshot_state):
    """Counting the blocks it managed to read would publish a prefix as a total."""
    pool = Pool([Block(1, block_hash=hash_key(1, 0), ref_cnt=0),
                 fd.Obj(block_id=2, block_hash=None, ref_cnt="two", is_null=False),
                 Block(3, ref_cnt=0)], free=3, total=3)
    section = gpu_census.census(pool)
    assert section["available"] is False
    assert "ref_cnt is str" in section["reason"]
    for field in ("cached_idle", "free_uncached", "referenced", "walked_blocks"):
        assert section[field] is None, f"a partial walk leaked into {field}"
    assert snapshot.error_counters()["errors"] >= 1


def test_the_walk_bound_refuses_rather_than_truncating(monkeypatch):
    monkeypatch.setattr(gpu_census, "MAX_WALKED_BLOCKS", 4)
    pool = Pool([Block(index, ref_cnt=0) for index in range(10)], free=10, total=10)
    section = gpu_census.census(pool)
    assert section["available"] is False
    assert "above the 4 walk bound" in section["reason"]
    assert section["walked_blocks"] is None


def test_a_wedged_free_counter_costs_the_reconciliation_only():
    pool = _pool(cached_idle=2, free_uncached=1)
    pool.get_num_free_blocks = lambda: (_ for _ in ()).throw(
        RuntimeError("the counter is wedged"))
    section = gpu_census.census(pool)
    assert section["available"] is True     # the walk is the product, not the gauge
    assert section["cached_idle"] == 2
    assert section["free_queue_reported"] is None
    assert section["free_queue_reconciles"] is None


def test_a_raising_census_cannot_escape_snapshot_for_engine(reset_snapshot_state):
    """THE containment proof: a census that explodes must not reach the scheduler.

    Patched at the module boundary `snapshot.gpu_census_section` calls into, so this
    covers the real call site in the snapshot, not a copy of it.
    """
    engine, info = fd.production_geometry()
    original = gpu_census.section

    def explode(_engine):
        raise RuntimeError("the census exploded")

    gpu_census.section = explode
    try:
        payload = snapshot.snapshot_for_engine(engine, _NullHolder())
    finally:
        gpu_census.section = original

    assert payload.get("error") is None, "the snapshot itself degraded"
    census = payload["tier"]["gpu_pool_census"]
    assert census["available"] is False
    assert "RuntimeError" in census["reason"]
    assert census["cached_idle"] is None
    assert payload["self"]["errors"] >= 1


def test_a_hostile_pool_cannot_raise_out_of_the_snapshot(reset_snapshot_state):
    """A property that explodes on the loop thread must degrade the section only."""

    class HostilePool:
        @property
        def blocks(self):
            raise RuntimeError("no blocks for you")

    class HostileManager:
        block_pool = HostilePool()

    class HostileEngine:
        class scheduler:                        # noqa: N801 - attribute-shaped fixture
            kv_cache_manager = HostileManager
            connector = None
            kv_cache_config = None
            cache_config = None

    payload = snapshot.snapshot_for_engine(HostileEngine(), _NullHolder())
    assert payload["tier"]["gpu_pool_census"]["available"] is False
    assert payload.get("error") is None


def test_base_exception_is_never_swallowed_by_the_census():
    """A diagnostic that eats KeyboardInterrupt/SystemExit is a second outage."""

    class InterruptingBlock:
        block_id = 1
        block_hash = None
        is_null = False

        @property
        def ref_cnt(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        gpu_census.census(Pool([InterruptingBlock()], free=1, total=1))


# --- cost and purity --------------------------------------------------------

def test_the_walk_covers_every_block_and_is_cheap_at_the_live_pool_size():
    """317 blocks is the live pool size. The walk is O(blocks); this measures it."""
    blocks = [Block(index, block_hash=hash_key(index, index % 6),
                    ref_cnt=index % 3) for index in range(317)]
    pool = Pool(blocks, free=200, total=317)
    started = time.perf_counter()
    section = gpu_census.census(pool)
    wall_us = (time.perf_counter() - started) * 1e6
    assert section["walked_blocks"] == 317
    # Both the self-reported and the wall-clock figure, so a regression reads as a
    # number rather than as a flaky assertion. The ceilings are deliberately loud.
    assert section["walk_us"] < 20_000
    assert wall_us < 50_000


def test_the_walk_scales_linearly_because_it_reads_each_block_once():
    """4x the blocks must not cost 16x: the walk is one pass, and this proves it."""
    small = Pool([Block(index, ref_cnt=0) for index in range(10_000)],
                 free=10_000, total=10_000)
    large = Pool([Block(index, ref_cnt=0) for index in range(40_000)],
                 free=40_000, total=40_000)
    small_us = min(gpu_census.census(small)["walk_us"] for _ in range(3))
    section = gpu_census.census(large)
    large_us = min(gpu_census.census(large)["walk_us"] for _ in range(3))
    assert section["walked_blocks"] == 40_000
    assert large_us < 16 * small_us, \
        f"4x blocks cost {large_us / max(small_us, 1):.1f}x - the walk is not linear"


def test_the_census_mutates_nothing_and_calls_no_engine_mutator():
    pool = _pool(cached_idle=4, free_uncached=2, referenced_hashed=1)
    before = [(block.block_id, block.ref_cnt, block.block_hash, block.is_null)
              for block in pool.blocks]
    gpu_census.census(pool)
    after = [(block.block_id, block.ref_cnt, block.block_hash, block.is_null)
             for block in pool.blocks]
    assert before == after
    # The ONLY pool method the census touches is the read-only free counter.
    assert pool.calls == ["get_num_free_blocks"]


def test_the_census_section_is_json_safe_like_the_rest_of_the_snapshot():
    engine, info = fd.production_geometry()
    payload = snapshot.json_safe({"tier": {"gpu_pool_census":
                                           gpu_census.section(engine)}})
    census = payload["tier"]["gpu_pool_census"]
    assert census["available"] is True
    assert isinstance(census["cached_idle"], int)


def test_the_census_rides_the_snapshot_and_adds_no_new_engine_path():
    """It is a section of the existing snapshot, so the cadence is the snapshot's."""
    engine, info = fd.production_geometry()
    view = snapshot.build_view(engine)
    census = view["tier"]["gpu_pool_census"]
    # The fixture's composition: blocks 0-39 are hashed AND referenced, 40-63 are
    # free and unhashed - so this boot has nothing idle-but-matchable, and the
    # census says exactly that rather than calling the 24 unhashed ones retained.
    assert census["available"] is True
    assert census["referenced_hashed"] == 40
    assert census["cached_idle"] == 0
    assert census["free_uncached"] == 24
    assert sum(census["hashed_per_group"].values()) == 40


# --- helpers ----------------------------------------------------------------

class _NullHolder:
    def ensure_loaded(self):
        return {"ok": False, "error": "no policy in this test", "traceback": None}

    @property
    def decide(self):
        return None
