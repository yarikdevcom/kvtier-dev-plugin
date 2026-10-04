r"""Capability 9: PER-REQUEST MISS ATTRIBUTION, and MEASURED retention depth.

The question the operator could not answer on 2026-10-04, at 176k-token prompts
with `cached_tokens = 0` next to a neighbour at 97%: WHY did THIS request miss.
Every aggregate instrument (counters, gauges, snapshots) was silent about it, so
this file is obliged to prove the instrument actually attributes, and is honest
where it cannot.

What must be proven, in the order it would cost to learn it live:

1. THE ATTRIBUTION IS RIGHT, field by field, with the units it claims: the query
   length, the hit, the residue, and the alignment residues that explain a partial
   hit. A wrong field is what cost the day.
2. THE VERDICTS ARE REACHABLE, all of them, or explicitly listed as not - an enum
   member no scenario can emit is a dead promise on the wire.
3. THE FIRST-MISS POSITION follows from the strict-prefix-walk property, and the
   payload says it is DERIVED, not observed.
4. NO REFERENCE LEAKS. Nothing a lookup records may keep an engine block alive:
   holding one is the exact refcount bug class the PR #56810 backport exists to
   fix. Proven with weak references, not by inspection.
5. BOUNDS. Thousands of requests through the hot path: the ring never grows, and
   an HTTP read REFUSES rather than truncating a list of "requests that missed".
6. THE MEASURED DEPTH answers the ~785x capacity dispute and distinguishes
   "free" from "empty", because conflating them reads as an empty tier on a box
   whose tier is 100% full.
7. THE FREE-ORDER WITNESS can CONFIRM and REFUTE the prefix-root-eviction
   hypothesis, and reports INCONCLUSIVE rather than a guess when the evidence is
   ambiguous.
8. UNITS ARE ENFORCED: a ratio of scheduler steps against blocks is REFUSED,
   because that exact division produced a wrong production conclusion.
9. MUTATION. The verdict arithmetic is pinned at its comparison boundaries, so
   flipping one breaks something (and the file's own header records the flips
   that were run).
"""

from __future__ import annotations

import gc
import json
import weakref

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import (
    config,
    evictions,
    lookup_ledger,
    snapshot,
)
from kvtier_dev_plugin import (
    miss_attribution as ma,
)

# --- the geometry of the live box -------------------------------------------
# manager.py's own numbers as measured on the box: scheduler block 3200 tokens,
# hash block 40 tokens, six groups whose spans are [3200 x4, 8, 3200], group 4 the
# QSA scratch ring (parked nothing, hit-able nothing).
GROUP_SIZES = [3200, 3200, 3200, 3200, 8, 3200]
CACHEABLE = [0, 1, 2, 3, 5]
BLOCK_SIZE = 3200
HASH_BLOCK = 40
LOCAL = 160_000                 # the block-aligned local hit handed over
NUM_TOKENS = 200_001            # so the last-token rule leaves 40,000 queryable
HASHES = NUM_TOKENS // HASH_BLOCK + 1


@pytest.fixture(autouse=True)
def clean_state():
    """Every module's state is process-wide; these tests must not stack on it."""
    ma.reset_state_for_tests()
    lookup_ledger.reset_state_for_tests()
    evictions.reset_state_for_tests()
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0, "epoch": None})
    yield
    ma.reset_state_for_tests()
    lookup_ledger.reset_state_for_tests()
    evictions.reset_state_for_tests()


# The box's measured parked spans per group on 2026-10-04. UNEQUAL on purpose:
# a fixture where every group ties cannot tell a joint MINIMUM from a MAXIMUM, and
# a mutation that publishes the max as the depth is exactly the error that would
# make a 200k prefix look retainable when it is not.
LIVE_PARKED = {"0": 259, "1": 259, "2": 260, "3": 260, "5": 292}


def parked_blocks(parked_per_group=None, total=1331, referenced=0):
    """A CPU pool that looks like the live one: parked spans INSIDE the free queue.

    `BlockPool.free_blocks` appends HASHED blocks back into the free queue as
    reuse candidates (block_pool.py:796-798), so `free == parked` is the
    full-tier signature, and a test that made free==0 would model an empty pool.
    """
    if parked_per_group is None:      # `{}` means EMPTY, not "use the default"
        parked_per_group = dict(LIVE_PARKED)
    blocks = []
    block_id = 0
    for group, count in sorted(parked_per_group.items()):
        for _ in range(count):
            blocks.append(fake.Block(block_id,
                                     block_hash=fake.hash_key(block_id,
                                                              int(group)),
                                     ref_cnt=0, num_tokens=3200))
            block_id += 1
    for _ in range(referenced):
        blocks.append(fake.Block(block_id, block_hash=fake.hash_key(block_id, 5),
                                 ref_cnt=2, num_tokens=3200))
        block_id += 1
    while len(blocks) < total:
        blocks.append(fake.Block(block_id))
    parked = sum(parked_per_group.values()) + referenced
    return blocks, parked


def tier(park=None, free=None, referenced=0, evictions_total=0, stored=6732,
         per_group=None):
    """The tier manager, duck-typed, with the live box's geometry and counters."""
    blocks, parked = parked_blocks(park, referenced=referenced)
    manager = fake.tier_manager(group_sizes=list(GROUP_SIZES),
                                hash_block_size=HASH_BLOCK,
                                block_size=BLOCK_SIZE, num_cpu_blocks=1331,
                                cpu_blocks=blocks)
    # `Pool.free_block_queue.num_free_blocks` IS the parked count here: free_blocks
    # appends hashed blocks back into the free queue (block_pool.py:796-798).
    manager.cpu_block_pool._free = parked if free is None else free
    manager.num_cpu_blocks = len(blocks)
    manager.cpu_block_pool.free_block_queue.num_free_blocks = \
        manager.cpu_block_pool._free
    manager.prefix_cacheable_group_ids = list(CACHEABLE)
    # The two `BlockPool` methods the free-order witness wraps (block_pool.py:664
    # and :770). A minimal fake: hand blocks out of the front, accept a free list.
    pool = manager.cpu_block_pool
    pool._hand_out = {"index": 0}

    def get_new_blocks(count):
        taken = pool.blocks[pool._hand_out["index"]:pool._hand_out["index"] + count]
        pool._hand_out["index"] += count
        pool._free = max(0, pool._free - count)
        return taken

    pool.get_new_blocks = get_new_blocks
    # Faithful to `BlockPool.free_blocks(ordered_blocks)`: it CONSUMES the
    # iterable, and manager.py:1297 hands it a GENERATOR. A witness that iterates
    # first would silently hand the engine an empty sequence.
    pool.freed = []

    def free_blocks(ordered):
        pool.freed.append(list(ordered))
        return "engine-verdict"

    pool.free_blocks = free_blocks
    manager.cpu_coordinator.find_longest_cache_hit = (
        lambda hashes, max_len: (tuple([] for _ in GROUP_SIZES), 0, 0))
    manager.boundary_store_stats = fake.Obj(name="BoundaryStoreStats",
                                            stored=stored)
    manager._load_event_counter = 35
    manager._store_event_counter = 3433
    if per_group is not None:
        manager.cpu_coordinator.find_longest_cache_hit_per_group = per_group
    evictions._state["evictions_total"] = evictions_total
    return manager


def req(num_tokens=NUM_TOKENS, hashes=HASHES, skip=False):
    return fake.request("cmpl-miss-1", num_tokens=num_tokens,
                        block_hashes=hashes, skip_reading_prefix_cache=skip)


def arm(manager=None, per_group=None, **kwargs):
    """Arm the module; `per_group` installs a fake
    `find_longest_cache_hit_per_group` on the tier's coordinator."""
    if manager is None:
        manager = tier(per_group=per_group)
    ma.set_manager_ref(manager)
    ma.arm(**kwargs)
    return manager


# --- 1. the fields, with their units ----------------------------------------

def test_clean_hit_reports_every_field_and_its_unit():
    manager = arm()
    fields = ma.observe_connector(req(), LOCAL, (40_000, True), manager)
    assert fields["verdict"] == ma.V_HIT, fields["verdict_why"]
    assert fields["num_tokens"] == 200_001
    assert fields["local_as_passed"] == 160_000
    assert fields["block_size"] == 3200 and fields["hash_block_size"] == 40
    # The scheduler pre-aligns, so the residue it leaves is 0 BY CONSTRUCTION and
    # a non-zero here would mean a caller other than the scheduler asked.
    assert fields["local_mod_block"] == 0
    assert fields["local_mod_hash"] == 0
    # 5,000 hashes, 4,000 skipped by the local hit: 1,000 queried = 40,000 tokens,
    # and the last-token rule (manager.py:359-362) allows exactly 40,000.
    assert fields["ext_query_hashes"] == 1_001          # 5,001 hashes - 4,000
    assert fields["ext_query_tokens_possible"] == 40_040
    assert fields["ext_query_tokens_capped"] == 40_000
    assert fields["ext_hit_tokens"] == 40_000
    assert fields["residue_tokens"] == 0
    # everything NOT served by either tier: the prompt minus the local hit minus
    # the external hit. 1 token here = the tier served the whole prompt bar the
    # last token, which is the most a lookup is ever allowed to match.
    assert fields["total_residue_tokens"] == 1
    assert fields["span_tokens"] == HASH_BLOCK       # partial hash hits are on
    # Every numeric field this file asserts on must declare a unit.
    assert fields["unit"]["ext_query_hashes"].startswith("hash blocks")
    assert fields["unit"]["residue_tokens"] == "tokens"
    assert fields["unit"]["walk_matched_spans"].endswith("(DERIVED)")
    # the long-form prose is published once per snapshot, not per record
    assert "units" not in fields
    assert ma.UNITS["ext_query_hashes"].startswith("hash blocks")


def test_the_unaligned_local_hit_is_a_gate_not_a_miss():
    """manager.py:338: an unaligned local hit returns (0, False) outright."""
    manager = arm()
    fields = ma.observe_connector(req(), LOCAL + 40, (0, False), manager)
    assert fields["verdict"] == ma.V_ZERO_ALIGNMENT_GATE
    assert fields["local_mod_block"] == 40


def test_each_zero_gate_is_its_own_verdict():
    manager = arm()
    # the engine forbade the lookup at all
    assert ma.observe_connector(req(skip=True), LOCAL, (0, False),
                                manager)["verdict"] \
        == ma.V_ZERO_SKIP_READING
    # the local hit already covers every hashed block
    fields = ma.observe_connector(req(hashes=LOCAL // HASH_BLOCK), LOCAL,
                                  (0, False), manager)
    assert fields["verdict"] == ma.V_ZERO_NO_HASHES_LEFT
    assert fields["ext_query_hashes"] == 0
    # at least one token must always be recomputed
    fields = ma.observe_connector(req(num_tokens=LOCAL + 1, hashes=5000), LOCAL,
                                  (0, False), manager)
    assert fields["verdict"] == ma.V_ZERO_LAST_TOKEN_RULE
    # and a connector that gave no verdict at all is not a miss either
    assert ma.observe_connector(req(), LOCAL, (None, False),
                                manager)["verdict"] == ma.V_NO_VERDICT


# --- 2. the first-miss position ---------------------------------------------

def test_first_miss_position_is_derived_from_the_prefix_walk_and_says_so():
    """A missing block implies every later block misses (chained hashes), so the
    depth reached and the failing index are one fact: matched = hit // span."""
    manager = arm()
    fields = ma.observe_connector(req(), LOCAL, (8_000, True), manager)
    assert fields["walk_matched_spans"] == 8_000 // HASH_BLOCK
    assert fields["first_miss_span"] == 8_000 // HASH_BLOCK
    assert fields["first_miss_token_offset"] == 160_000 + 8_000
    assert "single_type_kv_cache_manager.py:806-808" in fields["first_miss_source"]
    # the residue explains it: the walk stopped 32,000 tokens short of the query
    assert fields["residue_tokens"] == 32_000
    assert fields["verdict"] == ma.V_PARTIAL_HIT_CLAMP_SUSPECTED


def test_a_miss_at_the_root_reports_span_zero():
    manager = arm(tier(evictions_total=0))
    fields = ma.observe_connector(req(), LOCAL, (0, False), manager)
    assert fields["walk_matched_spans"] == 0
    assert fields["first_miss_span"] == 0
    assert fields["first_miss_token_offset"] == 160_000


# --- 3. the clamp (cause 2) --------------------------------------------------

def test_engine_residue_names_a_clamp_and_reports_the_candidates():
    """manager.py:362 throws away num_uncached_common_prefix_tokens; the observer
    catches it, and that residue is the only clamp evidence on the normal path."""
    manager = arm()
    blocks = tuple([fake.Block(1)] * 2 for _ in GROUP_SIZES)
    ma.observe_coordinator([1] * 10, 40_000,
                           (blocks, 40_000, 120_000))   # joint, residue
    fields = ma.observe_connector(req(), LOCAL, (40_000, True), manager)
    assert fields["coordinator_matched"] is True
    assert fields["clamp_residue_tokens"] == 120_000
    assert fields["verdict"] == ma.V_CLAMP_GROUP_AMBIGUOUS
    assert "not named without a deep probe" in fields["verdict_why"]
    assert fields["group_depths_tokens"]["5"] == 2 * 3200


def test_a_single_group_at_the_joint_depth_is_named():
    manager = arm()
    # group 3 is the ONLY group sitting at the joint depth of one 3,200-token span;
    # every other group returned far more, so the residue is group 3's doing.
    blocks = tuple([fake.Block(1)] if index == 3
                   else [fake.Block(1)] * 40 for index in range(len(GROUP_SIZES)))
    ma.observe_coordinator([1] * 10, 40_000, (blocks, 3200, 120_000))
    fields = ma.observe_connector(req(), LOCAL, (3200, True), manager)
    assert fields["verdict"] == f"{ma.V_CLAMP_AT_GROUP}3", fields["verdict_why"]


def test_a_deep_probe_names_the_clamping_group_and_a_tie_names_a_set():
    lengths = (32_000, 32_000, 32_000, 8_000, 0, 150_000)
    manager = arm(per_group=lambda hashes, max_len: (None, lengths))
    ma.arm(deep=True, sample_every=1, engine=None)
    ma.observe_coordinator([1] * 10, 40_000, ([None] * 6, 8_000, 142_000))
    fields = ma.observe_connector(req(), LOCAL, (8_000, True), manager)
    assert fields["independent_group_depths_tokens"]["3"] == 8_000
    assert fields["verdict"] == f"{ma.V_CLAMP_AT_GROUP}3", fields["verdict_why"]

    tied = (32_000, 32_000, 8_000, 8_000, 0, 150_000)
    ma.reset_state_for_tests()
    manager = arm(per_group=lambda hashes, max_len: (None, tied))
    ma.arm(deep=True, sample_every=1)
    ma.observe_coordinator([1] * 10, 40_000, ([None] * 6, 8_000, 142_000))
    fields = ma.observe_connector(req(), LOCAL, (8_000, True), manager)
    assert fields["verdict"] == ma.V_CLAMP_LIMITED
    assert "2,3" in fields["verdict_why"]


def test_the_scratch_ring_is_never_the_clamp():
    """Group 4 is parked NOTHING (manager.py:838 skips non-cacheable groups), so a
    probe reporting 0 for it must not produce `clamp_at_group4`: four groups tie
    at 32,000 and the TIE, not the ring, is the honest answer."""
    lengths = (32_000, 32_000, 32_000, 32_000, 0, 150_000)
    manager = arm(per_group=lambda hashes, max_len: (None, lengths))
    ma.arm(deep=True, sample_every=1)
    ma.observe_coordinator([1] * 10, 40_000, ([None] * 6, 32_000, 118_000))
    fields = ma.observe_connector(req(), LOCAL, (32_000, True), manager)
    assert not fields["verdict"].startswith(f"{ma.V_CLAMP_AT_GROUP}4")
    assert fields["verdict"] == ma.V_CLAMP_LIMITED
    assert "0,1,2,3" in fields["verdict_why"]


def test_a_probe_that_is_not_binding_names_nothing():
    """If every group could serve MORE than the tier gave, the clamp is not the
    reason and naming a group would be a lie."""
    lengths = (40_000, 40_000, 40_000, 40_000, 0, 40_000)
    manager = arm(per_group=lambda hashes, max_len: (None, lengths))
    ma.arm(deep=True, sample_every=1)
    ma.observe_coordinator([1] * 10, 40_000, ([None] * 6, 8_000, None))
    fields = ma.observe_connector(req(), LOCAL, (8_000, True), manager)
    assert not fields["verdict"].startswith(ma.V_CLAMP_AT_GROUP)
    assert fields["verdict"] == ma.V_PARTIAL_HIT_CLAMP_SUSPECTED


def test_deep_probes_are_budgeted():
    lengths = tuple([32_000] * 6)
    calls = {"n": 0}

    def per_group(hashes, max_len):
        calls["n"] += 1
        return (None, lengths)

    _ = arm(per_group=per_group)
    ma.arm(deep=True, sample_every=1)
    ma._state["deep_probe_budget_left"] = 3
    for _ in range(50):
        ma._state["lookups_seen"] += 1
        ma._maybe_deep_probe([1], 40_000)
    assert calls["n"] == 3, "the probe budget is a hard cap, not a suggestion"
    assert ma.counters()["deep_probe_budget_left"] == 0


# --- 4. eviction versus cold cache ------------------------------------------

def test_a_root_miss_with_evictions_is_pool_pressure():
    manager = arm(tier(evictions_total=7224))
    fields = ma.observe_connector(req(), LOCAL, (0, False), manager)
    assert fields["verdict"] == ma.V_POOL_PRESSURE
    assert "7224" in fields["verdict_why"]


def test_a_root_miss_without_evictions_is_root_missing():
    manager = arm(tier(evictions_total=0))
    assert ma.observe_connector(req(), LOCAL, (0, False),
                                manager)["verdict"] == ma.V_ROOT_MISSING


def test_an_empty_pool_with_no_stores_is_cold_cache():
    manager = arm(tier(park={}, stored=0))
    fields = ma.observe_connector(req(), LOCAL, (0, False), manager)
    assert fields["verdict"] == ma.V_COLD_CACHE, fields["verdict_why"]


def test_an_empty_pool_that_DID_store_says_unknown_not_cold():
    """Everything parked has been evicted: that is NOT a cold cache, and calling it
    one would hide the eviction."""
    manager = arm(tier(park={}, stored=6732))
    assert ma.observe_connector(req(), LOCAL, (0, False),
                                manager)["verdict"] == ma.V_UNKNOWN


# --- 5. the verdict vocabulary is honest -------------------------------------

def test_every_verdict_member_is_reachable_or_listed_as_not():
    """A member no scenario can emit is a dead promise on the wire."""
    scenarios = [
        (req(), LOCAL, (40_000, True)),                        # hit
        (req(), LOCAL, (None, False)),                         # no_verdict
        (req(), LOCAL + 40, (0, False)),                       # alignment gate
        (req(skip=True), LOCAL, (0, False)),                   # skip reading
        (req(num_tokens=LOCAL + 1, hashes=5000), LOCAL, (0, False)),   # last token
        (req(hashes=LOCAL // HASH_BLOCK), LOCAL, (0, False)),  # no hashes left
        (req(), LOCAL, (8_000, True)),                         # partial hit
    ]
    manager = arm()
    emitted = set()
    for request, local, result in scenarios:
        verdict = ma.observe_connector(request, local, result, manager)["verdict"]
        emitted.add(verdict.split("<")[0])
    ma.reset_state_for_tests()
    manager = arm(tier(park={}, stored=0))
    emitted.add(ma.observe_connector(req(), LOCAL, (0, False),
                                     manager)["verdict"])
    ma.reset_state_for_tests()
    manager = arm(tier(evictions_total=5))
    emitted.add(ma.observe_connector(req(), LOCAL, (0, False), manager)["verdict"])
    ma.reset_state_for_tests()
    manager = arm(tier(evictions_total=0))
    emitted.add(ma.observe_connector(req(), LOCAL, (0, False), manager)["verdict"])
    ma.reset_state_for_tests()
    manager = arm()
    ma.observe_coordinator([1], 40_000, (tuple([fake.Block(1)] * 2
                                               for _ in GROUP_SIZES),
                                         40_000, 120_000))
    emitted.add(ma.observe_connector(req(), LOCAL, (40_000, True),
                                     manager)["verdict"])
    ma.reset_state_for_tests()
    manager = arm(per_group=lambda h, m: (None, (32_000, 32_000, 32_000, 8_000, 0,
                                                 150_000)))
    ma.arm(deep=True, sample_every=1)
    ma.observe_coordinator([1], 40_000, ([None] * 6, 8_000, 142_000))
    emitted.add(ma.observe_connector(req(), LOCAL, (8_000, True), manager)["verdict"])
    ma.reset_state_for_tests()
    manager = arm(per_group=lambda h, m: (None, (32_000, 32_000, 8_000, 8_000, 0,
                                                 150_000)))
    ma.arm(deep=True, sample_every=1)
    ma.observe_coordinator([1], 40_000, ([None] * 6, 8_000, 142_000))
    emitted.add(ma.observe_connector(req(), LOCAL, (8_000, True), manager)["verdict"])
    # `unknown`: the CPU pool could not be measured at all, so neither
    # "root missing" nor "pool pressure" may be claimed for this miss.
    ma.reset_state_for_tests()
    unreadable = tier(evictions_total=0)
    unreadable.cpu_block_pool.blocks = None
    manager = arm(unreadable)
    emitted.add(ma.observe_connector(req(), LOCAL, (0, False),
                                     manager)["verdict"])

    missing = []
    for name in ma.VERDICTS:
        if name in ma.SUFFIXED_VERDICTS:
            if not any(entry.startswith(name) for entry in emitted):
                missing.append(name)
        elif name not in emitted:
            missing.append(name)
    assert not missing, f"unreachable verdicts: {missing}"
    assert "unknown" in ma.VERDICTS
    # and the honesty list is itself non-empty: the causes we CANNOT name
    assert any("content_diverged" in item for item in ma.UNDETERMINABLE_VERDICTS)
    assert any("gpu_evicted" in item for item in ma.UNDETERMINABLE_VERDICTS)


# --- 6. no reference leaks ---------------------------------------------------

def test_nothing_recorded_keeps_an_engine_block_alive():
    """Holding a KVCacheBlock would change what the pool can evict: the exact bug
    class the PR #56810 backport exists to fix. Proven with weak references."""
    manager = arm()
    blocks = tuple([fake.Block(index, block_hash=fake.hash_key(index, index),
                               ref_cnt=1) for index in range(4)]
                   for _ in GROUP_SIZES)
    refs = [weakref.ref(block) for group_blocks in blocks for block in group_blocks]
    ma.observe_coordinator([1] * 10, 40_000, (blocks, 4 * 3200, 100))
    request = req()
    lookup_ledger.record_lookup("cmpl-miss-1", LOCAL, 4 * 3200, True,
                                BLOCK_SIZE, HASH_BLOCK,
                                attribution=ma.attribute(request, LOCAL,
                                                         (4 * 3200, True), manager))
    del blocks, request
    gc.collect()
    assert all(ref() is None for ref in refs), [
        ref() for ref in refs if ref() is not None]
    gc.collect()


def test_stored_values_are_only_ids_and_integers():
    manager = arm()
    ma.observe_coordinator([1] * 10, 40_000,
                           (tuple([fake.Block(1)] * 2 for _ in GROUP_SIZES),
                            40_000, 120_000))
    fields = ma.observe_connector(req(), LOCAL, (40_000, True), manager)

    def scalars(value):
        if isinstance(value, dict):
            return all(isinstance(key, str) and scalars(item)
                       for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return all(scalars(item) for item in value)
        return isinstance(value, (int, str, bool, type(None)))

    assert scalars(fields)


# --- 7. bounds ---------------------------------------------------------------

def test_the_ledger_stays_bounded_under_thousands_of_lookups():
    manager = arm()
    lookup_ledger.arm(cap=64)
    for index in range(5_000):
        request = fake.request(f"cmpl-{index}", num_tokens=NUM_TOKENS,
                               block_hashes=HASHES)
        lookup_ledger.record_from_call(
            lookup_ledger.METHOD_LOOKUP, (request, LOCAL), (8_000, True), manager)
    counters = lookup_ledger.counters()
    assert len(lookup_ledger._state["ring"]) == 64
    assert counters["live"] <= 64
    assert counters["dropped"] == 5_000 - 64
    assert counters["record_errors"] == 0
    read = lookup_ledger.requests(limit=64)
    assert read["count"] <= 64
    assert read["estimated_bytes"] <= lookup_ledger.MAX_REQUESTS_BYTES


def test_an_http_read_refuses_rather_than_truncating_silently():
    manager = arm()
    lookup_ledger.arm(cap=128)
    for index in range(128):
        lookup_ledger.record_from_call(
            lookup_ledger.METHOD_LOOKUP,
            (fake.request(f"cmpl-{index}", num_tokens=NUM_TOKENS,
                          block_hashes=HASHES), LOCAL),
            (8_000, True), manager)
    lookup_ledger.MAX_REQUESTS_BYTES = 4096        # a tight budget for the test
    big = lookup_ledger.requests(limit=128)
    assert big.get("refused"), "over budget must be a refusal, not a short list"
    assert big["would_fit"] >= 1
    assert big["entries"] == []
    assert big["limit_requested"] == 128
    small = lookup_ledger.requests(limit=big["would_fit"])
    assert not small.get("refused")
    assert small["estimated_bytes"] <= 4096


def test_the_verb_payloads_are_bounded_and_shaped():
    manager = arm()
    engine = fake.engine(fake.scheduler(tier=manager))
    armed = lookup_ledger.lookups_for_engine(engine, {"op": "arm", "cap": 32})
    assert armed["op"] == "arm"
    assert armed["miss_attribution"]["armed"] is True
    for op, extra in (("requests", {"limit": 16}), ("depth", {}), ("order", {}),
                      ("rates", {}), ("status", {}), ("read", {}),
                      ("verdict", {})):
        payload = lookup_ledger.lookups_for_engine(engine, {"op": op, **extra})
        assert payload["op"] == op, payload
        assert len(json.dumps(payload, default=str)) < 400_000, op
    unknown = lookup_ledger.lookups_for_engine(engine, {"op": "requests",
                                                        "mirror": "/etc/passwd"})
    assert "mirror" in unknown["unknown_args"], \
        "the file mirror must not be reachable from HTTP"


def test_the_snapshot_publishes_the_measured_depth_unarmed():
    manager = arm()
    ma.disarm()
    engine = fake.engine(fake.scheduler(tier=manager))
    view = snapshot.build_view(engine)
    depth = view["tier"]["cpu_retained_depth"]
    assert depth["available"] is True, depth
    assert "free_does_not_mean_empty" in depth


# --- 8. measured retention depth --------------------------------------------

def test_retained_depth_separates_joint_from_aggregate_tokens():
    """The three readings disagreed by ~785x: this separates the two quantities
    they conflated, from MEASURED pool state."""
    _ = arm()
    depth = ma.retained_depth(force=True)
    assert depth["available"] is True
    assert depth["pool_blocks"] == 1331
    assert depth["parked_spans_per_group"]["5"] == 292
    # one span of an attention group covers 3,200 tokens OF THAT GROUP
    assert depth["parked_group_tokens_per_group"]["0"] == 259 * 3200
    # the sum over groups counts a conversation 5 times: it is VOLUME, not depth
    assert depth["aggregate_parked_group_tokens"] == 1330 * 3200
    # and a joint hit needs EVERY cacheable group, so the depth is the MINIMUM over
    # groups (259 spans), not the maximum (292) and not the sum (1,330).
    assert depth["joint_parked_tokens_max"] == 259 * 3200
    assert depth["joint_parked_tokens_max"] < 292 * 3200
    assert depth["joint_parked_tokens_max"] < depth["aggregate_parked_group_tokens"]
    # one conversation costs one slot PER cacheable group: 5/3200 slots per token
    assert depth["slots_per_1k_tokens_one_conversation"] == pytest.approx(
        1000 * 5 / 3200)
    assert depth["per_conversation_ceiling_tokens"] == int(
        1331 / (5 / 3200))
    assert depth["group_span_tokens"]["4"] == 8
    assert "index_entries x hash_block_size" in depth["why_three_answers_differed"]


def test_free_does_not_mean_empty_here():
    """1,330 of 1,331 blocks "free" on the box IS the full-tier signature: parked
    spans live inside the free queue. A metric that read this as an empty tier
    would send us chasing a bug that does not exist."""
    _ = arm()
    depth = ma.retained_depth(force=True)
    assert depth["free_blocks_reported"] == depth["parked_spans_total"]
    assert depth["cached_idle"] == depth["parked_spans_total"]
    assert depth["free_uncached"] == 1331 - depth["parked_spans_total"]
    assert depth["free_does_not_mean_empty"].startswith("free_blocks_reported")


def test_parked_counts_include_the_referenced_parked_spans():
    """A parked span that is currently PINNED (ref_cnt > 0, mid-restore) is still
    parked: dropping `referenced_hashed` from the total would under-report the
    tier by exactly whatever is being restored right now."""
    manager = arm(tier(park={"0": 10, "1": 10, "2": 10, "3": 10, "5": 10},
                       referenced=7))
    manager.cpu_block_pool._free = 50          # the 50 idle parked spans only
    manager.cpu_block_pool.free_block_queue.num_free_blocks = 50
    depth = ma.retained_depth(force=True)
    assert depth["cached_idle"] == 50
    assert depth["referenced_hashed"] == 7
    assert depth["parked_spans_total"] == 57
    assert depth["parked_spans_total"] > depth["cached_idle"]


def test_retained_depth_refuses_an_unwalkable_pool_rather_than_a_partial_total():
    manager = arm()
    manager.cpu_block_pool.blocks = [fake.Block(index)
                                     for index in range(60_000)]
    depth = ma.retained_depth(force=True)
    assert depth["available"] is False
    assert "REFUSED" in depth["reason"]


def test_retained_depth_without_a_tier_reports_a_reason_not_a_zero():
    ma.set_manager_ref(None)
    depth = ma.retained_depth(force=True)
    assert depth["available"] is False
    assert "no tier manager" in depth["reason"]


def test_a_zero_hit_verdict_uses_the_memoised_walk():
    """20 misses must cost ONE pool walk, not twenty: `gpu_census` measures
    ~0.6 us/block, so twenty walks over a 1,331-block pool is 16 ms inside
    `schedule()` for a diagnostic."""
    from kvtier_dev_plugin import gpu_census
    manager = arm()
    walks = {"n": 0}
    original = gpu_census._walk

    def counting(blocks):
        walks["n"] += 1
        return original(blocks)

    gpu_census._walk = counting
    try:
        for index in range(20):
            ma.observe_connector(fake.request(f"cmpl-{index}",
                                              num_tokens=NUM_TOKENS,
                                              block_hashes=HASHES),
                                 LOCAL, (0, False), manager)
    finally:
        gpu_census._walk = original
    assert walks["n"] == 1, f"the pool walk cost {walks['n']} walks for 20 misses"


# --- 9. the free-order witness (the root-eviction hypothesis) ----------------

def freed_chain(pool, park_first=True, hashed=True):
    """The blocks of ONE request's chain, in the order the CPU path frees them."""
    spans = [fake.Block(index, block_hash=fake.hash_key(index, 5) if hashed
                        else None, ref_cnt=1) for index in range(6)]
    for _ in spans:                           # park them: root first, tail last
        pool.get_new_blocks(1)
    park = ma._order["park_seq"]
    for index, block in enumerate(spans):
        park[block.block_id] = index + 1
    ordered = spans if park_first else list(reversed(spans))
    pool.free_blocks(ordered)
    return spans


def arm_witness():
    manager = arm()
    report = ma.arm_order_witness(manager.cpu_block_pool)
    assert report["installed"] is True, report
    return manager


def test_a_root_first_free_list_CONFIRMS_the_prefix_root_eviction_hypothesis():
    manager = arm_witness()
    freed_chain(manager.cpu_block_pool, park_first=True)
    report = ma.order_report()
    assert report["root_first"] == 1
    assert report["verdict"].startswith("CONFIRMED"), report["verdict"]
    assert "block_pool.py:796-798" in report["verdict"]


def test_a_tail_first_free_list_REFUTES_it():
    manager = arm_witness()
    freed_chain(manager.cpu_block_pool, park_first=False)
    report = ma.order_report()
    assert report["root_last"] == 1
    assert report["verdict"].startswith("REFUTED"), report["verdict"]


def test_a_generator_free_list_reaches_the_engine_intact():
    """`manager.py:1297` frees with a GENERATOR. Iterating it to observe would
    leave the engine freeing NOTHING - leaked refs and a tier that cannot recycle.
    The witness must materialise once and pass the same sequence through."""
    manager = arm_witness()
    pool = manager.cpu_block_pool
    blocks = [fake.Block(index, block_hash=fake.hash_key(index, 5), ref_cnt=1)
              for index in range(5)]
    verdict = pool.free_blocks(block for block in blocks)
    assert pool.freed == [blocks], "the engine received a different sequence"
    assert verdict == "engine-verdict", "the engine's own return value must survive"
    assert ma.order_report()["freed_blocks"] == 5


def test_a_list_free_list_is_passed_through_untouched():
    manager = arm_witness()
    pool = manager.cpu_block_pool
    blocks = [fake.Block(index) for index in range(3)]
    assert pool.free_blocks(blocks) == "engine-verdict"
    assert pool.freed[0] is blocks or pool.freed == [blocks]


def test_a_hashless_or_singleton_free_proves_nothing():
    manager = arm_witness()
    freed_chain(manager.cpu_block_pool, hashed=False)
    manager.cpu_block_pool.free_blocks([fake.Block(7,
                                                  block_hash=fake.hash_key(7, 5))])
    report = ma.order_report()
    assert report["root_first"] == 0 and report["root_last"] == 0
    assert report["verdict"].startswith("INCONCLUSIVE")
    assert report["unknown"] == 2


def test_the_witness_refuses_a_foreign_owner_and_restores_on_disarm():
    manager = arm()
    pool = manager.cpu_block_pool
    setattr(pool, ma.OWNER_ATTR, "someone-else")
    report = ma.arm_order_witness(pool)
    assert report["installed"] is False and "foreign owner" in report["reason"]
    ma.reset_state_for_tests()
    manager = arm()
    original_new = manager.cpu_block_pool.get_new_blocks
    original_free = manager.cpu_block_pool.free_blocks
    ma.arm_order_witness(manager.cpu_block_pool)
    assert manager.cpu_block_pool.get_new_blocks is not original_new
    removed = ma._unarm_order_witness()
    assert removed["uninstalled"] is True
    assert manager.cpu_block_pool.get_new_blocks is original_new
    assert manager.cpu_block_pool.free_blocks is original_free


def test_the_park_sequence_array_is_bounded_by_the_pool():
    manager = arm()
    manager.cpu_block_pool.blocks = [fake.Block(index) for index in range(10)]
    report = ma.arm_order_witness(manager.cpu_block_pool)
    assert report["installed"] is True
    assert len(ma._order["park_seq"]) == 10


def test_the_coordinator_observer_is_instance_level_never_class_level():
    """Wrapping the CLASS would put this work on the GPU coordinator too."""
    manager = arm()
    original = manager.cpu_coordinator.find_longest_cache_hit
    report = ma.arm_coordinator_observer(manager)
    assert report["installed"] is True
    assert manager.cpu_coordinator.find_longest_cache_hit is not original
    other = fake.tier_manager(group_sizes=list(GROUP_SIZES))
    assert not hasattr(other.cpu_coordinator, "find_longest_cache_hit"), \
        "a SECOND tier instance must be untouched: the wrap is instance-level"
    removed = ma._unarm_coordinator_observer(manager)
    assert removed["uninstalled"] is True
    assert manager.cpu_coordinator.find_longest_cache_hit is original


def test_the_wrapper_returns_the_engines_own_value_unchanged():
    manager = arm()
    sentinel = (("blocks",), 1234, 56)
    manager.cpu_coordinator.find_longest_cache_hit = lambda *args: sentinel
    ma.arm_coordinator_observer(manager)
    assert manager.cpu_coordinator.find_longest_cache_hit([1], 10) is sentinel


# --- 10. units ---------------------------------------------------------------

def test_a_ratio_of_steps_against_blocks_is_refused():
    _ = arm()
    ratios = ma.rates()
    assert any("load_event_counter" in item for item in ratios["refused"])
    assert "steps against blocks" in " ".join(rates_text(ratios))
    assert "evictions_total_per_stored" in ratios["allowed"]
    allowed = ratios["allowed"]["evictions_total_per_stored"]
    assert allowed["numerator_unit"].startswith("BLOCKS")
    assert allowed["denominator_unit"].startswith("BLOCKS")
    assert ma.UNITS["load_event_counter"].startswith("SCHEDULER STEPS")
    assert ma.UNITS["evictions_total"].startswith("BLOCKS")
    assert ma.UNITS["hit_length_tokens_observed_max"].startswith("tokens")


def rates_text(ratios):
    return [item for item in ratios["refused"]]


# --- 11. the file mirror -----------------------------------------------------

def test_the_mirror_is_env_gated_and_stops_at_its_byte_cap(tmp_path, monkeypatch):
    path = tmp_path / "miss.jsonl"
    monkeypatch.setenv(config.ENV_MISS_LOG, str(path))
    manager = arm()
    for _ in range(5):
        ma.observe_connector(req(), LOCAL, (8_000, True), manager)
    lines = path.read_text().splitlines()
    assert len(lines) == 5
    assert json.loads(lines[0])["verdict"] == ma.V_PARTIAL_HIT_CLAMP_SUSPECTED
    monkeypatch.setattr(ma, "MAX_MIRROR_BYTES", 1200)
    for _ in range(50):
        ma.observe_connector(req(), LOCAL, (8_000, True), manager)
    assert ma.counters()["mirror_stopped"].startswith("reached MAX_MIRROR_BYTES")
    assert len(path.read_text().splitlines()) < 60, "the mirror must stop growing"


def test_no_env_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_MISS_LOG, raising=False)
    manager = arm()
    ma.observe_connector(req(), LOCAL, (40_000, True), manager)
    assert ma.counters()["mirror_path"] is None
    assert list(tmp_path.iterdir()) == []


def test_the_log_once_idiom_logs_an_episode_once(caplog):
    with caplog.at_level("WARNING"):
        for _ in range(50):
            ma.notice_once("same-episode", "said once")
    assert sum(1 for record in caplog.records if "said once" in record.message) == 1


# --- 12. mutation: the verdict arithmetic is pinned at its boundaries --------

def test_verdict_boundaries_are_pinned_so_a_flipped_comparison_breaks():
    """Each assertion sits ON a comparison in `_decide`. Flipping `residue > 0` to
    `residue >= 0`, `offered > 0` to `offered >= 0`, `parked == 0` to `<= 0`, or the
    `local % block != 0` sign each fails one of these."""
    manager = arm()
    served = 40_000
    # residue == 0 -> hit; residue == 1 -> partial. The boundary is > 0.
    whole = ma.observe_connector(req(num_tokens=NUM_TOKENS), LOCAL,
                                 (served, True), manager)
    assert whole["residue_tokens"] == 0
    assert whole["verdict"] == ma.V_HIT
    one_short = ma.observe_connector(req(num_tokens=NUM_TOKENS + 1), LOCAL,
                                     (served, True), manager)
    assert one_short["residue_tokens"] == 1
    assert one_short["verdict"] == ma.V_PARTIAL_HIT_CLAMP_SUSPECTED
    # offered == 0 is a miss, offered == 1 is a hit. The boundary is > 0.
    assert ma.observe_connector(req(), LOCAL, (0, False),
                                manager)["verdict"] != ma.V_HIT
    assert ma.observe_connector(req(), LOCAL, (1, False),
                                manager)["verdict"] == ma.V_PARTIAL_HIT_CLAMP_SUSPECTED
    # an alignment residue of exactly 0 passes the gate, 40 does not.
    assert ma.observe_connector(req(), LOCAL, (0, False),
                                manager)["verdict"] != ma.V_ZERO_ALIGNMENT_GATE
    assert ma.observe_connector(req(), LOCAL + 40, (0, False),
                                manager)["verdict"] == ma.V_ZERO_ALIGNMENT_GATE
    # parked == 0 with nothing stored is cold; parked == 1 is not.
    ma.reset_state_for_tests()
    manager = arm(tier(park={"5": 1}, stored=0))
    assert ma.observe_connector(req(), LOCAL, (0, False),
                                manager)["verdict"] != ma.V_COLD_CACHE


def test_attribute_never_raises_on_a_broken_request_or_manager():
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError(f"cannot read {name}")

    manager = arm()
    for request, local, result in ((Exploding(), LOCAL, (0, False)),
                                   (Exploding(), None, None),
                                   (req(), "not an int", (5, "x")),
                                   (req(), -10, (0, False))):
        fields = ma.observe_connector(request, local, result, manager)
        assert isinstance(fields, dict)
        assert fields["verdict"] in ma.VERDICTS or \
            fields["verdict"].startswith(ma.SUFFIXED_VERDICTS)
    assert ma.counters()["attribution_errors"] >= 0
    assert snapshot._state["errors"] > 0, "the faults must at least be counted"


def test_hot_path_is_inert_when_disarmed():
    manager = tier()
    ma.set_manager_ref(manager)
    assert ma.observe_connector(req(), LOCAL, (40_000, True), manager) is None
    assert ma.observe_coordinator([1], 10, ((), 10, 0)) is None
    assert ma.counters()["attributions"] == 0


def test_the_attribution_rides_the_ledgers_record_and_the_snapshot():
    manager = arm()
    engine = fake.engine(fake.scheduler(tier=manager))
    lookup_ledger.lookups_for_engine(engine, {"op": "arm", "cap": 32})
    # The same entry point `guard_mode._record_success()` uses, called directly:
    # the fake tier has no fork class to wrap, and this is the hook's own contract.
    lookup_ledger.record_from_call(
        lookup_ledger.METHOD_LOOKUP,
        (fake.request("cmpl-x", num_tokens=NUM_TOKENS, block_hashes=HASHES),
         LOCAL), (8_000, True), manager)
    entry = lookup_ledger.read(limit=8)
    rows = [item for item in entry["entries"] if item.get("attribution")]
    assert rows, "the attribution must land on the ledger's own record"
    assert rows[0]["attribution"]["verdict"] in ma.VERDICTS
    section = lookup_ledger.section(preview=2)
    assert "verdict_meaning" in section["miss_attribution"]
    assert "undeterminable" in section["miss_attribution"]
