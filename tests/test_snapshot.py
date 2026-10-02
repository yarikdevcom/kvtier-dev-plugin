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


@pytest.mark.parametrize("module_file", [
    "snapshot.py", "policy.py", "policy_holder.py", "config.py",
    "engine_side.py", "endpoint.py", "store_hook.py",
])
def test_no_heavy_imports_in_the_read_side(module_file):
    """Requirement: the engine-side path never pulls torch/CUDA/vllm in.

    Checked by AST over every import in the file, so it cannot be satisfied by
    an import that only runs conditionally.
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
