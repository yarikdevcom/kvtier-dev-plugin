"""The shipped policy must read ONLY keys the snapshot producer writes, and a
missing input must never become a number.

WHY THIS FILE EXISTS. `policy.decide()` - the OBSERVE hook - read
`tier["cpu_free_blocks"]`, a key `snapshot.tier_view()` has never emitted (the
tier view exposes that number as `cpu_pool.free_blocks`; `cpu_free_blocks` is the
*store-context* name from `store_hook.build_store_ctx`). `free = tier.get(...) or
0` turned the phantom into "0 free blocks", so every snapshot answered
`cpu_used_ratio: 1.0` while the live pool reported `free_blocks: 664/665`.

This is a REPORTING bug, not a throughput bug, and the tests say so: `decide()`
only ever populates `policy.result`, and the store gate is the separate
`decide_store()` hook. Two layers, in the order they catch the defect:

  1. a structural (AST) test that fails if any key read off the view is a key the
     producer does not write - it catches this bug from the other direction, and
     it is the durable version of the audit that found it;
  2. behavioural tests: an absent input yields `None` plus a verdict that names
     the missing input - never a fabricated ratio in either direction;
  3. the ratio pinned against a view built by the REAL producer, shaped like the
     live snapshot (665 total / 659 free), reported as PINNED-ness because
     `total - free` counts blocks held by in-flight copies, not parked KV: a
     parked block returns to the free queue and stays retrievable through its
     hash-index entry. `dropped_cpu_full` is the capacity-pressure signal.

Nothing imports vllm or torch: the stubs use the fork's real attribute names, so
`snapshot.tier_view()` runs its real code over them.
"""

from __future__ import annotations

import ast
import inspect
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_snapshot import (  # noqa: E402
    FakeBoundaryStoreStats,
    FakeManager,
    IndexPool,
    filled,
)

from kvtier_dev_plugin import policy, snapshot  # noqa: E402

UNKNOWN = "cpu-pool-pinned-unknown"


# --- 1. the structural contract: read keys must be written keys --------------

# name of the local in `decide` -> the path in the view it is bound to.
READER_PATHS = {
    "view": (),
    "tier": ("tier",),
    "counts": ("tier", "counts"),
    "stats": ("tier", "boundary_store_stats"),
    "pool": ("tier", "cpu_pool"),
    "index": ("tier", "cpu_hash_index"),
}


def _literal_key_readers(tree: ast.Module) -> dict[str, set[str]]:
    """Every `name.get("lit")` / `name["lit"]` for the view-bound locals."""
    found: dict[str, set[str]] = {name: set() for name in READER_PATHS}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "get"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in found and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found[func.value.id].add(node.args[0].value)
        elif isinstance(node, ast.Subscript):
            value = node.value
            if (isinstance(value, ast.Name) and value.id in found
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                found[value.id].add(node.slice.value)
    return found


def _subtree(view: dict, path: tuple[str, ...]):
    node = view
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _live_shaped_view(total: int = 665, free: int = 659,
                      entries: int = 719, stats_obj=None) -> dict:
    """A tier view built by the producer with the live geometry and counters.

    The pool carries the parked-KV hash index, so `cpu_hash_index.entries` and
    `tokens_upper_bound` are emitted for real rather than assumed. The stats are
    deliberately quiet (no drops) so a verdict assertion is about the pool.
    """
    manager = FakeManager(cpu_pool=IndexPool(total, free, filled(entries)),
                          stats=stats_obj or FakeBoundaryStoreStats(
                              published=100, stored=100))
    manager.num_cpu_blocks = total
    manager.block_size = 3200
    manager.hash_block_size = 800
    manager.fa_block_size = 3200
    manager.fa_gidx = 5
    manager.group_block_sizes = [3200, 3200, 3200, 3200, 8, 3200]
    return {"tier": snapshot.tier_view(manager)}


def test_every_key_the_policy_reads_is_a_key_the_snapshot_writes():
    """The defect class, killed structurally: a phantom key fails the suite.

    The view is built by `snapshot.tier_view()` over a stub manager, so this
    asserts against the producer's real output and not against a hand-written
    dict that could invent a field the way the old policy did.
    """
    view = _live_shaped_view()
    tree = ast.parse(inspect.getsource(policy.decide))
    readers = _literal_key_readers(tree)

    # A rename of a local must not make this test silently vacuous.
    assert set(readers) == set(READER_PATHS)
    for name, keys in readers.items():
        assert keys, (f"decide() reads no literal keys off {name!r}: the local "
                      f"was renamed and this contract test has gone blind")

    phantoms = []
    for name, keys in readers.items():
        subtree = _subtree(view, READER_PATHS[name])
        where = ".".join(READER_PATHS[name]) or "<view>"
        for key in keys:
            if not isinstance(subtree, dict) or key not in subtree:
                phantoms.append(f"{name}[{key!r}] (not in the producer's "
                                f"{where})")
    assert not phantoms, \
        "policy.decide reads keys the snapshot producer never writes: " + \
        ", ".join(sorted(phantoms))


def test_the_tier_view_has_no_cpu_free_blocks_key_to_read():
    """Pins the exact phantom: the fix must not "fix" it by emitting the key.

    `cpu_free_blocks` belongs to the store context; if the tier view grew one it
    would be a second, independently-maintained copy of `cpu_pool.free_blocks`
    and the pair would drift.
    """
    tier = snapshot.tier_view(FakeManager())
    assert "cpu_free_blocks" not in tier
    assert tier["cpu_pool"] == {"bound": True, "total_blocks": 4096,
                                "free_blocks": 1024}


# --- 2. a missing input is never a number ------------------------------------

def test_a_missing_free_counter_is_never_fabricated_as_a_full_pool():
    """The regression: the live view with the phantom key removed.

    The shipped bug answered `cpu_used_ratio: 1.0` for exactly this shape, on
    every pass.
    """
    view = {"tier": {"present": True, "num_cpu_blocks": 665,
                     "cpu_pool": {"bound": True, "total_blocks": 665},
                     "counts": {}, "boundary_store_stats": {}}}
    result = policy.decide(view)
    assert result["cpu_pinned_ratio"] is None
    assert result["cpu_pinned_blocks"] is None
    assert result["verdict"] == UNKNOWN
    assert "cpu_pool.free_blocks" in result["missing_inputs"]
    # The phantom key must not be honoured even if a caller supplies it.
    phantom = policy.decide({"tier": {"present": True, "num_cpu_blocks": 665,
                                      "cpu_free_blocks": 0,
                                      "counts": {},
                                      "boundary_store_stats": {}}})
    assert phantom["cpu_pinned_ratio"] is None
    assert phantom["verdict"] == UNKNOWN
    assert "cpu_pool.free_blocks" in phantom["missing_inputs"]


def test_a_missing_total_is_not_fabricated_as_an_empty_pool():
    """The mirror-image default: a missing denominator must not read "0 pinned"."""
    view = {"tier": {"present": True,
                     "cpu_pool": {"bound": True, "free_blocks": 659},
                     "counts": {}, "boundary_store_stats": {}}}
    result = policy.decide(view)
    assert result["cpu_pinned_ratio"] is None
    assert result["verdict"] == UNKNOWN
    assert "cpu_pool.total_blocks" in result["missing_inputs"]


def test_counters_that_are_not_counters_do_not_become_zero():
    """An unbound pool reports `bound: False` and no numbers at all."""
    view = {"tier": {"present": True, "cpu_pool": {"bound": False},
                     "counts": {}, "boundary_store_stats": {}}}
    result = policy.decide(view)
    assert result["cpu_pinned_ratio"] is None
    assert set(result["missing_inputs"]) == {"cpu_pool.total_blocks",
                                             "cpu_pool.free_blocks"}
    assert result["cpu_pool_total_blocks"] is None
    assert result["cpu_pool_free_blocks"] is None
    # No index section either: retrievability is unknown, not zero.
    assert result["parked_entries"] is None


def test_free_above_total_is_reported_not_silently_clamped():
    """Two counters that disagree cannot yield a ratio - say which."""
    view = {"tier": {"present": True,
                     "cpu_pool": {"bound": True, "total_blocks": 665,
                                  "free_blocks": 700},
                     "counts": {}, "boundary_store_stats": {}}}
    result = policy.decide(view)
    assert result["cpu_pinned_ratio"] is None
    assert result["verdict"] == UNKNOWN
    assert any("free_blocks" in why and "total_blocks" in why
               for why in result["missing_inputs"])


def test_a_zero_capacity_pool_is_not_reported_as_empty_and_happy():
    view = {"tier": {"present": True,
                     "cpu_pool": {"bound": True, "total_blocks": 0,
                                  "free_blocks": 0},
                     "counts": {}, "boundary_store_stats": {}}}
    assert policy.decide(view)["cpu_pinned_ratio"] is None
    assert policy.decide(view)["verdict"] == UNKNOWN


def test_a_count_of_zero_is_a_real_answer_not_a_missing_one():
    """A pool with 0 free must still be believed (0 is a value, not absence)."""
    assert policy._count(0) == 0
    for not_a_count in (None, True, False, "5", 3.0, [0]):
        assert policy._count(not_a_count) is None


def test_capacity_pressure_is_reported_even_with_the_pool_counters_absent():
    """`dropped_cpu_full` is the capacity signal, and it outranks an unknown."""
    view = {"tier": {"present": True, "cpu_pool": {"bound": False},
                     "counts": {},
                     "boundary_store_stats": {"dropped_cpu_full": 3,
                                              "stored": 10}}}
    result = policy.decide(view)
    assert result["verdict"] == "cpu-pool-full-dropping-stores"
    assert result["dropped_cpu_full"] == 3


# --- 3. the true value, from the real producer ------------------------------

@pytest.mark.parametrize(("total", "free", "expected"), [
    (665, 659, 0.009),    # the shape the load test contradicted itself about
    (665, 664, 0.0015),   # what /plugins/kvtier/snapshot reported live
    (665, 0, 1.0),        # every block pinned by in-flight copies
    (665, 665, 0.0),      # nothing pinned
])
def test_the_pinned_ratio_matches_the_pool_the_snapshot_reports(total, free,
                                                                expected):
    view = _live_shaped_view(total=total, free=free)
    assert view["tier"]["cpu_pool"] == {"bound": True, "total_blocks": total,
                                        "free_blocks": free}
    result = policy.decide(view)
    assert result["cpu_pinned_ratio"] == expected
    assert result["cpu_pinned_blocks"] == total - free
    assert result["cpu_pool_total_blocks"] == total
    assert result["cpu_pool_free_blocks"] == free
    assert result["missing_inputs"] == []


def test_a_pool_that_is_99_percent_free_is_not_reported_as_full():
    """The one-line version of the bug."""
    result = policy.decide(_live_shaped_view(665, 659))
    assert result["verdict"] == "ok"
    assert result["cpu_pinned_ratio"] < policy.NEARLY_FULL_RATIO


def test_the_word_nearly_full_is_gone_because_it_meant_occupancy_it_cannot_measure():
    """95% PINNED must be named pinned, and no shape may claim "nearly full".

    A parked block returns to the free queue, so a high free count says nothing
    about how much KV the arena holds; the old verdict word invited exactly the
    inference the live snapshot contradicted.
    """
    pinned_high = policy.decide(_live_shaped_view(100, 5))
    assert pinned_high["cpu_pinned_ratio"] == 0.95
    assert pinned_high["verdict"] == "cpu-pool-pinned-high"

    empty_arena_but_full_index = policy.decide(_live_shaped_view(665, 664))
    assert empty_arena_but_full_index["verdict"] == "ok"
    assert empty_arena_but_full_index["parked_entries"] == 719
    # The producer's own arithmetic: one hash block per entry.
    assert (empty_arena_but_full_index["parked_tokens_upper_bound"]
            == 719 * 800)
    assert pinned_high["verdict"] != "cpu-pool-nearly-full"


def test_no_tier_is_not_reported_as_an_unknown_pinned_share():
    """A boot without the connector legitimately has no pool keys."""
    assert policy.decide({"tier": {"present": False}})["verdict"] == "no-tier"


def test_the_verdict_stays_json_and_msgpack_safe_with_a_none_ratio():
    view = _live_shaped_view()
    del view["tier"]["cpu_pool"]["free_blocks"]
    assert snapshot.json_safe({"policy": policy.decide(view)})


def test_the_shipped_policy_is_still_cheap():
    """`decide` runs on the engine loop thread; keep it microseconds."""
    view = _live_shaped_view()
    started = time.perf_counter()
    for _ in range(1000):
        policy.decide(view)
    per_call = (time.perf_counter() - started) / 1000
    assert per_call < 50e-6, f"{per_call * 1e6:.1f} us per decide() call"
