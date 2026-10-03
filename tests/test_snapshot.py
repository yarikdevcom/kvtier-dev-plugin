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


def test_the_tier_view_carries_the_index_next_to_the_allocator_it_is_not_the_same_as():
    index = FakeHashIndex({packed(b"a", 0): FakeCachedBlock(1, packed(b"a", 0))})
    manager = FakeManager(cpu_pool=IndexPool(665, 664, index))
    tier = snapshot.snapshot_for_engine(make_engine(manager), _NullHolder())["tier"]
    assert tier["cpu_pool"]["free_blocks"] == 664
    assert tier["cpu_hash_index"]["entries"] == 1
    assert tier["cpu_hash_index"]["meaning"]
    assert tier["boundary_store_stats"]["available"] is True
