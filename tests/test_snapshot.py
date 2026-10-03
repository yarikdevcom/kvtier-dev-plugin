"""The snapshot reader, against a stub engine with the REAL attribute names.

Nothing here imports vllm (it is not installed for these tests) - which is the
point: the read side is pure attribute reads, so a stub is a faithful engine.
The stubs use the names read out of the fork at HEAD, so a rename in the fork
shows up here as a `None` in the snapshot rather than as a crash.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from kvtier_dev_plugin import snapshot

PKG = Path(snapshot.__file__).resolve().parent

BOUNDARY_FIELDS = ("published", "stored", "dropped_cpu_full",
                   "dropped_request_gone", "skipped_already_cached",
                   "skipped_in_flight", "dropped_null_block",
                   "dropped_not_hashed")


@dataclasses.dataclass
class FakeBoundaryStoreStats:
    """Mirrors vllm/v1/simple_kv_offload/manager.py:125-132 exactly."""

    published: int = 0
    stored: int = 0
    dropped_cpu_full: int = 0
    dropped_request_gone: int = 0
    skipped_already_cached: int = 0
    skipped_in_flight: int = 0
    dropped_null_block: int = 0
    dropped_not_hashed: int = 0


class FakePool:
    """Mirrors BlockPool: num_gpu_blocks + an O(1) get_num_free_blocks()."""

    def __init__(self, total: int, free: int) -> None:
        self.num_gpu_blocks = total
        self._free = free

    def get_num_free_blocks(self) -> int:
        return self._free


class ExplodingPool(FakePool):
    def get_num_free_blocks(self) -> int:
        raise RuntimeError("the pool lied")


class FakeManager:
    """Mirrors SimpleCPUOffloadScheduler's scheduler-side state."""

    def __init__(self, *, stats=None, pending=False,
                 cpu_pool=None, gpu_pool=None) -> None:
        self._lazy_mode = True
        self._target_free = 128
        self.block_size = 256
        self.hash_block_size = 64
        self.fa_block_size = 32
        self.fa_gidx = 0
        self.group_block_sizes = [256, 64, 64, 64, 16, 256]
        self.num_cpu_blocks = 4096
        self.cpu_block_pool = cpu_pool or FakePool(4096, 1024)
        self._gpu_block_pool = gpu_pool if gpu_pool is not None else FakePool(2048, 12)
        self._reqs_to_load = {"a": 1, "b": 2}
        self._reqs_to_store = {"a": 1}
        self._abandoned_reqs_to_load = {}
        self._load_event_to_reqs = {7: ["a"]}
        self._store_event_to_blocks = {3: object()}
        self._abandoned_store_event_to_blocks = {}
        self._pending_finished_stores = []
        self._in_flight_store_gpu_blocks = {11, 12, 13}
        self._pending_cpu_hits = {}
        self._store_event_pending_counts = {3: 2}
        self._load_event_counter = 7
        self._store_event_counter = 4
        self.boundary_store_stats = stats or FakeBoundaryStoreStats(
            published=9, stored=5, dropped_cpu_full=2, dropped_request_gone=1,
            skipped_already_cached=1, skipped_in_flight=0,
            dropped_null_block=0, dropped_not_hashed=0)
        self._pending = pending

    def has_pending_stores(self) -> bool:
        return self._pending

    def get_boundary_store_stats(self):
        # The real manager returns a `replace(...)` COPY (manager.py:947-948);
        # the copy below is what makes the snapshot safe to hand to a caller.
        return dataclasses.replace(self.boundary_store_stats)


class FakeConnector:
    def __init__(self, manager) -> None:
        self.scheduler_manager = manager


class FakeScheduler:
    def __init__(self, connector) -> None:
        self.connector = connector
        self.requests = {"a": 1, "b": 2}
        self.running = ["a"]
        self.waiting = ["b"]


class FakeEngine:
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler


def make_engine(manager=FakeManager()):
    return FakeEngine(FakeScheduler(FakeConnector(manager)))


class RaisingEngine:
    @property
    def scheduler(self):  # a fork that blows up on a property must not crash us
        raise RuntimeError("scheduler exploded")


class QuietLogger:
    def __init__(self) -> None:
        self.warnings = []

    def warning(self, *args, **kwargs):
        self.warnings.append(args)


class HostileLogger(QuietLogger):
    def warning(self, *args, **kwargs):
        raise ValueError("the log writer is on fire")


# --- the read side stays light ---------------------------------------------

HEAVY = {"torch", "vllm", "cuda", "numpy", "triton", "numba", "cupy", "psutil"}


@pytest.mark.parametrize("module_file", sorted(p.name for p in PKG.glob("*.py")))
def test_no_heavy_imports_in_the_read_side(module_file):
    """Requirement: the engine-side path never pulls torch/CUDA/vllm in.

    Checked by AST over every import in the file, so it cannot be satisfied by an
    import that only runs conditionally. The file list is the PACKAGE DIRECTORY,
    not a hand-written list: a new verb module is covered by this the moment it
    exists, which is the only way a forgotten `import torch` in a module added
    next month gets caught.
    """
    tree = ast.parse((PKG / module_file).read_text(encoding="utf-8"))
    pulled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pulled.add(node.module.split(".")[0])
    assert not (pulled & HEAVY), (module_file, sorted(pulled & HEAVY))


def test_snapshot_never_names_the_worker_side():
    """The worker handler records on the live compute stream
    (simple_kv_offload/worker.py:293-294) and must stay untouched."""
    source = (PKG / "snapshot.py").read_text(encoding="utf-8")
    assert "worker_handler" not in source
    assert "SimpleCPUOffloadWorker" not in source


def test_building_a_snapshot_pulls_in_nothing_heavy():
    for module in ("torch", "vllm"):
        assert module not in sys.modules
    snapshot.snapshot_for_engine(make_engine(), _NullHolder())
    for module in ("torch", "vllm"):
        assert module not in sys.modules


class _NullHolder:
    """A holder whose policy is absent: the policy section must degrade, not raise."""

    def ensure_loaded(self):
        return {"ok": False, "error": "no policy in this test", "traceback": None}

    @property
    def decide(self):
        return None


# --- content ----------------------------------------------------------------

def test_tier_view_reports_the_real_counters():
    view = snapshot.build_view(make_engine())
    tier = view["tier"]
    assert tier["present"] is True
    assert tier["mode"] == "lazy"
    assert tier["target_free_gpu_blocks"] == 128
    assert tier["block_size"] == 256
    assert tier["hash_block_size"] == 64
    assert tier["fa_block_size"] == 32
    assert tier["fa_group_index"] == 0
    assert tier["group_block_sizes"] == [256, 64, 64, 64, 16, 256]
    assert tier["num_cpu_blocks"] == 4096
    assert tier["cpu_pool"] == {"bound": True, "total_blocks": 4096,
                                "free_blocks": 1024}
    assert tier["gpu_pool"] == {"bound": True, "total_blocks": 2048,
                                "free_blocks": 12}
    assert tier["pending_stores"] is False
    assert tier["counts"] == {
        "reqs_to_load": 2,
        "reqs_to_store": 1,
        "abandoned_reqs_to_load": 0,
        "load_events_to_reqs": 1,
        "store_events": 1,
        "abandoned_store_events": 0,
        "store_events_awaiting_workers": 1,
        "pending_finished_stores": 0,
        "in_flight_store_gpu_blocks": 3,
        "pending_cpu_hits": 0,
    }
    assert tier["event_counters"] == {"load_event_counter": 7,
                                      "store_event_counter": 4}


def test_boundary_stats_expose_all_eight_decline_reasons():
    stats = snapshot.tier_view(FakeManager())["boundary_store_stats"]
    assert stats["available"] is True
    for field in BOUNDARY_FIELDS:
        assert field in stats, field
    assert stats["stored"] == 5 and stats["dropped_cpu_full"] == 2
    assert "missing_fields" not in stats and "unknown_fields" not in stats


def test_boundary_stats_are_a_copy_not_a_window():
    manager = FakeManager()
    stats = snapshot.tier_view(manager)["boundary_store_stats"]
    stats["stored"] = 999999
    stats["not_a_field"] = "written"
    assert manager.boundary_store_stats.stored == 5
    assert not hasattr(manager.boundary_store_stats, "not_a_field")


def test_unknown_and_missing_boundary_fields_are_named_not_hidden():
    @dataclasses.dataclass
    class FutureStats:
        published: int = 1
        brand_new_counter: int = 2

    stats = snapshot.tier_view(_ManagerWith(FutureStats()))["boundary_store_stats"]
    assert stats["missing_fields"] and "stored" in stats["missing_fields"]
    assert stats["unknown_fields"] == ["brand_new_counter"]


def _ManagerWith(stats):
    return FakeManager(stats=stats)


def test_a_renamed_counter_survives_as_a_string_not_a_crash():
    @dataclasses.dataclass
    class OddStats:
        published: str = "twelve"

    stats = snapshot.tier_view(_ManagerWith(OddStats()))["boundary_store_stats"]
    assert stats["published"] == "'twelve'"


def test_missing_connector_is_reported_as_a_reason():
    class NoConnector:
        connector = None

    view = snapshot.build_view(FakeEngine(NoConnector()))
    assert view["tier"]["present"] is False
    assert "no scheduler_manager" in view["tier"]["reason"]
    assert view["scheduler"]["connector"] is None


def test_no_gpu_pool_bound_is_bound_false():
    manager = FakeManager(gpu_pool=None)
    manager._gpu_block_pool = None
    assert snapshot.tier_view(manager)["gpu_pool"] == {"bound": False}


def test_a_lying_pool_is_data_and_an_error_count(reset_snapshot_state):
    snapshot.attach_logger(QuietLogger())
    engine = make_engine(FakeManager(cpu_pool=ExplodingPool(4096, 0)))
    view = snapshot.build_view(engine)
    assert view["tier"]["cpu_pool"]["free_blocks"] is None
    assert view["tier"]["cpu_pool"]["total_blocks"] == 4096
    assert reset_snapshot_state["errors"] == 1
    snapshot.attach_logger(None)


def test_epoch_is_stable_per_process_and_proves_one_boot():
    first = snapshot.build_view(make_engine())["epoch"]
    second = snapshot.build_view(make_engine())["epoch"]
    assert first == second and len(first) == 8


# --- payload safety ---------------------------------------------------------

def test_payload_is_json_and_msgpack_primitive_only():
    engine = make_engine()
    payload = snapshot.json_safe(snapshot.build_view(engine))
    assert json.dumps(payload, allow_nan=False)
    allowed = (dict, list, str, int, float, bool, type(None))
    walk = [payload]
    while walk:
        node = walk.pop()
        assert isinstance(node, allowed), type(node)
        if isinstance(node, dict):
            assert all(isinstance(key, str) for key in node)
            walk.extend(node.values())
        elif isinstance(node, list):
            walk.extend(node)


def test_json_safe_refuses_nan_instead_of_emitting_invalid_json():
    class Holder:
        def ensure_loaded(self):
            return {"ok": True, "error": None, "traceback": None}

        @property
        def decide(self):
            return lambda view: {"bad": float("nan")}

    payload = snapshot.snapshot_for_engine(make_engine(), Holder())
    assert "error" in payload and "ValueError" in payload["error"]


def test_snapshot_reports_how_long_it_took_and_its_own_counters(reset_snapshot_state):
    payload = snapshot.snapshot_for_engine(make_engine(), _NullHolder())
    assert payload["self"] == {"calls": 1, "errors": 0, "logged": 0}
    assert isinstance(payload["build_us"], int)
    # A read-side snapshot must not cost milliseconds. This is a smoke ceiling
    # for a stub, not a performance claim; real numbers come from build_us.
    assert payload["build_us"] < 100_000


# --- the host-RAM parked-KV hash index ---------------------------------------
#
# WHY THIS BLOCK EXISTS. The live snapshot reported
#
#     cpu_pool.total_blocks = 665   cpu_pool.free_blocks = 664
#     boundary_store_stats.stored = 4372
#
# and read those two together they look like a contradiction. They are not: a
# parked CPU block is returned to the ALLOCATOR when its store finishes, and what
# keeps it retrievable is its entry in `cached_block_hash_to_block` - the index the
# fork itself resolves hits through (get_one_block, manager.py:605, :940, :1251).
# Allocator freeness does NOT mean the index is empty. These tests pin that the
# reading is real, bounded, and labelled so it cannot be misread as freeness.

def packed(hash_bytes: bytes, group: int) -> bytes:
    """`make_block_hash_with_group_id` (kv_cache_utils.py:74-82): hash bytes then
    the group id as 4 big-endian unsigned bytes."""
    return hash_bytes + group.to_bytes(4, "big", signed=False)


class FakeCachedBlock:
    """A KVCacheBlock as the index sees it: an id, a hash, and the null flag."""

    def __init__(self, block_id: int, block_hash=None, is_null=False) -> None:
        self.block_id = block_id
        self.block_hash = block_hash
        self.is_null = is_null


class FakeHashIndex:
    """`BlockHashToBlockMap`: __len__ over its keys, private `_cache`, and NO
    public key iterator - which is exactly why the reader falls back to `_cache`
    and says it did."""

    def __init__(self, cache=None) -> None:
        self._cache = dict(cache or {})

    def __len__(self) -> int:
        return len(self._cache)

    def get_one_block(self, key):
        blocks = self._cache.get(key)
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        return blocks


class ExplodingIndex:
    def __len__(self) -> int:
        raise RuntimeError("the index lied")


class IndexPool(FakePool):
    """A CPU pool that carries the parked-KV index, like the fork's."""

    def __init__(self, total, free, index) -> None:
        super().__init__(total, free)
        self.cached_block_hash_to_block = index


def filled(count, groups=6):
    """An index of `count` entries spread over `groups` groups, each with a live
    block behind it, exactly as the fork's map holds them."""
    cache = {}
    for i in range(count):
        key = packed(b"h%05d" % i, i % groups)
        cache[key] = FakeCachedBlock(i, key)
    return FakeHashIndex(cache)


def index_view(index, hash_block_size=64):
    return snapshot._hash_index_view(
        IndexPool(665, 664, index), hash_block_size)


def test_the_index_reports_how_much_parked_kv_is_retrievable_not_how_free_the_pool_is():
    """The live contradiction, reproduced: a pool that reads 664/665 free with
    thousands of retrievable entries."""
    pool = IndexPool(665, 664, filled(4372))
    view = snapshot._hash_index_view(pool, 64)
    assert view["available"] is True
    assert view["entries"] == 4372
    assert view["counted_by"] == "len"
    assert view["tokens_upper_bound"] == 4372 * 64
    assert "UPPER BOUND" in view["tokens_note"]
    # The label is the whole point: it must say what it is NOT.
    assert "free_blocks" in view["meaning"]
    assert "allocator" in view["meaning"]
    # And the ALLOCATOR half is still reported next to it, unchanged: the two
    # numbers mean different things and the payload shows both.
    assert snapshot._pool_view(pool) == {"bound": True, "total_blocks": 665,
                                         "free_blocks": 664}


def test_per_group_counts_come_from_the_packed_group_id_and_say_how_they_were_read():
    index = FakeHashIndex({packed(b"a", 0): FakeCachedBlock(1, packed(b"a", 0)),
                           packed(b"b", 3): FakeCachedBlock(2, packed(b"b", 3)),
                           packed(b"c", 3): FakeCachedBlock(3, packed(b"c", 3)),
                           packed(b"d", 4): FakeCachedBlock(4, packed(b"d", 4))})
    view = index_view(index)
    assert view["per_group"] == {"0": 1, "3": 2, "4": 1}
    assert view["key_handle"] == "_cache"
    assert "private _cache" in view["key_handle_note"], (
        "reaching past the public surface must be admitted in the payload")
    assert view["entries_scanned"] == 4
    assert view["scan_truncated"] is False


def test_len_counts_keys_not_blocks_and_a_multi_block_key_is_visible():
    """`BlockHashToBlockMap` does not de-duplicate (block_pool.py:47-51): one key
    can hold several blocks. `entries` is DISTINCT HIT-ABLE PREFIXES, and the
    sample says when a key is holding more than one block."""
    key = packed(b"dup", 0)
    index = FakeHashIndex({key: {7: FakeCachedBlock(7, key),
                                 8: FakeCachedBlock(8, key)},
                           packed(b"solo", 0): FakeCachedBlock(9, packed(b"solo", 0))})
    view = index_view(index)
    assert view["entries"] == 2
    assert view["sampled_keys_with_multiple_blocks"] == 1


def test_a_stale_entry_would_show_up_as_a_sampled_value_without_a_hash():
    """The tripwire: a live entry's block still carries its hash. An entry whose
    block lost the hash is the stale case the count would otherwise hide."""
    good = packed(b"g", 0)
    index = FakeHashIndex({good: FakeCachedBlock(1, good),
                           packed(b"s", 0): FakeCachedBlock(2, None),
                           packed(b"n", 0): FakeCachedBlock(3, packed(b"n", 0),
                                                            is_null=True)})
    view = index_view(index)
    assert view["sampled_entries"] == 3
    assert view["sampled_values_without_hash"] == 1
    assert view["sampled_null_blocks"] == 1


def test_the_staleness_question_is_answered_with_evidence_not_dodged():
    """`stale_entries` must be a verdict plus the reasoning that a reader can
    check, and it must name what it cannot rule out."""
    view = index_view(FakeHashIndex())
    assert view["stale_entries"] == "not_expected_on_pool_paths"
    assert "get_new_blocks" in view["stale_entries_why"]
    assert "_remove_cached_block_hashes" in view["stale_entries_why"]
    assert "cannot see the copy" in view["stale_entries_not_ruled_out"]
    assert "eviction candidate" in view["retrievable_now_vs_retained"]


def test_a_pool_without_the_index_says_so_and_invents_no_number():
    view = snapshot._hash_index_view(FakePool(665, 664), 64)
    assert view["available"] is False
    assert "entries" not in view
    assert "has no cached_block_hash_to_block" in view["reason"]
    assert snapshot._hash_index_view(None, 64)["available"] is False


def test_an_unsized_index_invents_no_number():
    view = index_view(object())
    assert view["available"] is False
    assert "entries" not in view
    assert "no occupancy number invented" in view["reason"]


def test_an_index_whose_len_raises_is_contained_and_named(reset_snapshot_state):
    """A container whose len() RAISES must not be reported like one that has no
    size handle: those two need different answers from an operator."""
    view = index_view(ExplodingIndex())
    assert view["available"] is False
    assert "entries" not in view
    assert view["counted_by"] == "len"
    assert "__len__ raised RuntimeError" in view["reason"]
    assert snapshot.error_counters()["errors"] == 1


def test_the_key_scan_is_bounded_and_says_when_it_stopped(monkeypatch):
    monkeypatch.setattr(snapshot, "MAX_HASH_INDEX_SCAN", 10)
    view = index_view(filled(25, groups=1))
    assert view["entries"] == 25
    assert view["entries_scanned"] == 10
    assert view["scan_truncated"] is True


def test_the_group_cap_is_honoured(monkeypatch):
    monkeypatch.setattr(snapshot, "MAX_HASH_INDEX_GROUPS", 3)
    view = index_view(filled(6, groups=6))
    assert len(view["per_group"]) == 3
    assert view["groups_truncated"] is True


def test_a_key_too_short_to_carry_a_group_id_is_counted_not_decoded():
    index = FakeHashIndex({b"": FakeCachedBlock(1, b""),
                           b"abcd": FakeCachedBlock(2, b"abcd"),
                           packed(b"e", 1): FakeCachedBlock(3, packed(b"e", 1))})
    view = index_view(index)
    assert view["keys_without_group_id"] == 2
    assert view["per_group"] == {"1": 1}


class ExtraHashSetPool(IndexPool):
    """A CPU pool that also carries `cached_block_hashes_by_block` - the pool's own
    record of the EXTRA hashes filed under a block it already hashed
    (block_pool.py:188, populated at :641 by the secondary inserts at
    manager.py:1039-1044). It is what makes a secondary key positively identifiable
    instead of merely "not the primary"."""

    def __init__(self, total, free, index, extra=None) -> None:
        super().__init__(total, free, index)
        self.cached_block_hashes_by_block = dict(extra or {})


# The live fixture, measured read-only from the running engine on 2026-10-05:
# group 5 is the fine-graining group (fa_group_index=5, fa_block_size=3200,
# hash_block_size=800 -> a primary plus the 800/1600/2400 secondary ticks), the
# others mint one key per span. 701 KEYS, 470 BLOCKS, 665-block arena.
LIVE_SPANS_BY_GROUP = {0: 98, 1: 98, 2: 98, 3: 99, 5: 77}
LIVE_KEYS_PER_SPAN = {0: 1, 1: 1, 2: 1, 3: 1, 5: 4}
LIVE_PER_GROUP = {group: LIVE_SPANS_BY_GROUP[group] * LIVE_KEYS_PER_SPAN[group]
                  for group in LIVE_SPANS_BY_GROUP}
BLOCK_SIZE_TOKENS = 3200


def mint(spans_by_group, keys_per_group, fine_tick_is_own_hash=False):
    """An index and its pool's extra-hash set, minted the way the fork mints them.

    One span == one CPU block per group (`get_new_blocks()` hands out distinct ids,
    manager.py:755-758 / block_pool.py:678). A group that fine-grains adds its
    intra-block hashes as EXTRA KEYS pointing at the SAME block, one of which is
    the block's own `block_hash` and the rest of which land in
    `cached_block_hashes_by_block` - keys that cost no block and no bytes.
    """
    cache, extra, block_id = {}, {}, 0
    for group in sorted(spans_by_group):
        spans = spans_by_group[group]
        ticks = keys_per_group.get(group, 1)
        for span in range(spans):
            seed = b"s%04d" % span
            keys = [packed(seed if tick == 0 else seed + b"t%d" % tick, group)
                    for tick in range(ticks)]
            own = keys[-1] if fine_tick_is_own_hash else keys[0]
            block = FakeCachedBlock(block_id, own)
            cache[keys[0]] = block
            for key in keys[1:]:
                cache[key] = block
            if ticks > 1:
                extra[block_id] = {key for key in keys if key != own}
            block_id += 1
    return FakeHashIndex(cache), extra


def live_shaped_view(hash_block_size=800):
    """`cpu_hash_index` for the live 2026-10-05 geometry: 701 keys, 470 blocks."""
    index, extra = mint(LIVE_SPANS_BY_GROUP, LIVE_KEYS_PER_SPAN)
    return snapshot._hash_index_view(
        ExtraHashSetPool(665, 664, index, extra), hash_block_size)


def test_the_tier_view_carries_the_index_next_to_the_allocator_it_is_not_the_same_as():
    index = FakeHashIndex({packed(b"a", 0): FakeCachedBlock(1, packed(b"a", 0))})
    manager = FakeManager(cpu_pool=IndexPool(665, 664, index))
    tier = snapshot.snapshot_for_engine(make_engine(manager), _NullHolder())["tier"]
    assert tier["cpu_pool"]["free_blocks"] == 664
    assert tier["cpu_hash_index"]["entries"] == 1
    assert tier["cpu_hash_index"]["meaning"]
    assert tier["boundary_store_stats"]["available"] is True


# --- parked BLOCKS per group, and the primary/secondary key split -----------
#
# Why these exist: `per_group` counts KEYS of `BlockHashToBlockMap`, and the groups
# mint keys at different rates per parked span - the fine-graining group also mints
# its intra-block secondary hashes against the SAME cpu_block with num_tokens=None
# (manager.py:1039-1044), which cost dictionary entries and zero bytes. A consumer
# that sums or mins key counts publishes an upper bound and cannot know it. The live
# arithmetic that started this (2026-10-05, idle box):
#
#     per_group  {0:98, 1:98, 2:98, 3:99, 5:308}  -> sum 701 keys / 665 blocks = 1.00
#     the truth  {0:98, 1:98, 2:98, 3:99, 5: 77}  -> sum 470 blocks / 665 = 70.7%
#     joint depth  min(blocks) = 77 spans  -> 77 x 3200 = 246,400 tokens
# and the key-based min over-reports that joint depth by 27% (98 spans, not 77).


def test_parked_blocks_per_group_counts_spans_not_keys_on_the_live_geometry():
    view = live_shaped_view()
    # The key counts, unchanged: additive only, a consumer already reads them.
    assert view["per_group"] == {str(g): LIVE_PER_GROUP[g] for g in LIVE_PER_GROUP}
    assert view["entries"] == 701 and sum(view["per_group"].values()) == 701
    # The new field: distinct blocks per group = spans parked in that group.
    assert view["parked_blocks_per_group"] == {"0": 98, "1": 98, "2": 98,
                                               "3": 99, "5": 77}
    assert sum(view["parked_blocks_per_group"].values()) == 470
    # True occupancy, computable at last, and the bound it replaces.
    assert abs(470 / 665 - 0.707) < 1e-3
    assert min(sum(view["per_group"].values()) / 665, 1.0) == 1.0, (
        "the key-based ratio clamps to 'saturated' - that is the fabrication")
    # Joint coverage a five-group resume can actually use.
    assert min(view["parked_blocks_per_group"].values()) == 77
    assert (min(view["parked_blocks_per_group"].values()) * BLOCK_SIZE_TOKENS
            == 77 * 3200 == 246400)
    assert "SPANS" in view["parked_blocks_meaning"]
    assert "MAX_PARKED_BLOCK_IDS" in view["parked_blocks_source"]


def test_the_key_split_lets_a_consumer_cross_check_the_divisor_not_guess_it():
    view = live_shaped_view()
    assert view["per_group_primary"] == {"0": 98, "1": 98, "2": 98, "3": 99, "5": 77}
    assert view["per_group_secondary"] == {"0": 0, "1": 0, "2": 0, "3": 0, "5": 231}
    for group in LIVE_PER_GROUP:
        keys = view["per_group"][str(group)]
        assert (view["per_group_primary"][str(group)]
                + view["per_group_secondary"][str(group)]) == keys
    # One primary key per block, so the split and the block counts must agree; a
    # False here is the divisor assumption failing, stated as a field.
    assert view["key_split_matches_blocks"] is True
    assert "cached_block_hashes_by_block" in view["per_group_key_split_meaning"]


def test_the_discriminator_is_the_blocks_own_hash_not_which_tick_minted_first():
    """A partial tail can be hashed FIRST, making the 800-tick key the block's own
    hash and the 3200-tick one the extra. The counts must not care which it is."""
    index, extra = mint(LIVE_SPANS_BY_GROUP, LIVE_KEYS_PER_SPAN,
                        fine_tick_is_own_hash=True)
    view = snapshot._hash_index_view(
        ExtraHashSetPool(665, 664, index, extra), 800)
    spans = {str(g): s for g, s in LIVE_SPANS_BY_GROUP.items()}
    assert view["parked_blocks_per_group"] == spans
    assert view["per_group_primary"] == spans
    assert view["key_split_matches_blocks"] is True


def test_a_pool_without_an_extra_hash_set_withholds_the_split_but_not_the_blocks():
    """A fork that renames `cached_block_hashes_by_block` loses the SPLIT, not the
    block counts: 'not the primary' is not the same claim as 'secondary'."""
    index, _extra = mint(LIVE_SPANS_BY_GROUP, LIVE_KEYS_PER_SPAN)
    view = snapshot._hash_index_view(IndexPool(665, 664, index), 800)
    assert view["parked_blocks_per_group"] == {
        str(g): s for g, s in LIVE_SPANS_BY_GROUP.items()}
    assert view["per_group_primary"] is None
    assert view["per_group_secondary"] is None
    assert "cached_block_hashes_by_block" in view["per_group_primary_reason"]
    assert "cached_block_hashes_by_block" in view["per_group_secondary_reason"]


def test_an_entry_the_reader_cannot_classify_omits_the_split_instead_of_zeroing_it():
    """The requirement in its sharp form: one entry that is neither its block's own
    hash nor in that block's extra-hash set means the reader does not fully model
    this fork. It must NOT be filed under secondary, and the split must not be
    published with a plausible-looking 0."""
    index, extra = mint(LIVE_SPANS_BY_GROUP, LIVE_KEYS_PER_SPAN)
    stranger = packed(b"stranger", 5)
    index._cache[stranger] = FakeCachedBlock(99, packed(b"elsewhere", 5))
    view = snapshot._hash_index_view(
        ExtraHashSetPool(665, 664, index, extra), 800)
    assert view["per_group"]["5"] == 309          # keys: still counted
    assert view["parked_blocks_per_group"]["5"] == 78   # blocks: one more span
    assert view["per_group_primary"] is None
    assert view["per_group_secondary"] is None
    assert "neither their block's own hash" in view["per_group_primary_reason"]
    assert "key_split_matches_blocks" not in view


def test_a_truncated_key_scan_refuses_the_block_counts_rather_than_a_prefix():
    """`per_group` keeps its historical partial-scan behaviour; the block view is
    refused, because a min() over a prefix of the index is exactly the silent
    under-count this field exists to remove."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(snapshot, "MAX_HASH_INDEX_SCAN", 10)
        view = index_view(filled(25, groups=1))
    assert view["scan_truncated"] is True
    assert view["per_group"] == {"0": 10}
    assert view["parked_blocks_per_group"] is None
    assert "PREFIX" in view["parked_blocks_per_group_reason"]
    assert view["per_group_primary"] is None
    assert view["per_group_secondary"] is None


def test_the_parked_block_set_bound_is_honoured_explicitly(monkeypatch):
    """The one container the scan grows is capped, and hitting the cap is a refusal
    rather than a quietly truncated count."""
    index, extra = mint({0: 10}, {0: 1})
    with monkeypatch.context() as mp:
        mp.setattr(snapshot, "MAX_PARKED_BLOCK_IDS", 5)
        view = snapshot._hash_index_view(
            ExtraHashSetPool(665, 664, index, extra), 800)
    assert view["per_group"] == {"0": 10}
    assert view["parked_blocks_per_group"] is None
    assert "bound" in view["parked_blocks_per_group_reason"]


def test_an_index_value_that_is_not_a_block_refuses_the_block_view():
    """A value this reader cannot turn into block ids makes the WHOLE block view
    absent: counting the entries it did understand would mislabel a partial as a
    total, which is the failure mode this payload was built to avoid."""
    index, extra = mint({0: 3, 5: 2}, {0: 1, 5: 4})
    index._cache[packed(b"junk", 0)] = "not a block"
    view = snapshot._hash_index_view(ExtraHashSetPool(665, 664, index, extra), 800)
    assert view["available"] is True              # the key counts survive
    assert view["per_group"] == {"0": 4, "5": 8}
    assert view["parked_blocks_per_group"] is None
    assert "neither a KVCacheBlock nor a block dict" in (
        view["parked_blocks_per_group_reason"])


def test_a_key_carrying_several_blocks_is_counted_and_named():
    """The map does not de-duplicate (block_pool.py:47-51), so one key CAN hold
    several blocks - the one case where distinct blocks over-counts spans. It is
    counted, named, and the cross-check says so."""
    key = packed(b"dup", 0)
    index = FakeHashIndex({
        key: {7: FakeCachedBlock(7, key), 8: FakeCachedBlock(8, packed(b"own8", 0))},
        packed(b"solo", 0): FakeCachedBlock(9, packed(b"solo", 0)),
    })
    view = snapshot._hash_index_view(
        ExtraHashSetPool(665, 664, index, {8: {key}}), 800)
    assert view["parked_blocks_per_group"] == {"0": 3}
    assert view["keys_backed_by_multiple_blocks"] == 1
    # Both keys are a primary: `dup` is block 7's own hash, `solo` is block 9's.
    # Two primary keys, three blocks - the over-count the next field names.
    assert view["per_group_primary"] == {"0": 2}
    assert view["per_group_secondary"] == {"0": 0}
    assert view["key_split_matches_blocks"] is False, (
        "3 blocks behind 1 primary key: the one-block-per-span premise is what "
        "broke here, and the field must say so")


def test_an_empty_index_is_a_real_zero_not_a_refusal():
    view = index_view(FakeHashIndex())
    assert view["per_group"] == {}
    assert view["parked_blocks_per_group"] == {}
    assert view["per_group_primary"] == {}
    assert view["per_group_secondary"] == {}
    assert "parked_blocks_per_group_reason" not in view


def test_keys_with_no_parsable_group_id_refuse_the_block_view_too():
    """An index whose keys carry no group id is NOT an empty index, and `{}` would
    read as "nothing parked"."""
    index = FakeHashIndex({b"": FakeCachedBlock(1, b""),
                           b"abcd": FakeCachedBlock(2, b"abcd")})
    view = snapshot._hash_index_view(ExtraHashSetPool(665, 664, index, {}), 800)
    assert view["per_group"] == {}
    assert view["keys_without_group_id"] == 2
    assert view["parked_blocks_per_group"] is None
    assert "no group is claimable" in view["parked_blocks_per_group_reason"]
    assert view["per_group_primary"] is None
