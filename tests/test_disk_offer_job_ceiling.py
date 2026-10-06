r"""THE OFFER IS SIZED BY THE SMALLER OF TWO JOB BOUNDS, AND SAYS WHICH (0.2.35).

Every bound the offer path priced before 0.2.35 was either a DESTINATION count (CPU
slots, the block walk) or a BYTE count (the bounce-buffer pool). The reader has a third,
and the offer path did not know it: `disk_read.MAX_QUEUE_DEPTH` jobs sit in one bounded
`queue.Queue`, a job past the cap is refused `queue_full`, and `queue_full` is a member
of `LOAD_CAPACITY_FAILURES` - the bytes are fine and the read was never issued, so the
block ARRIVES AS NOTHING.

WHY THAT IS NOT A COSMETIC MISMATCH, in the sentence that decides everything here: a
missing block is not a smaller restore, it is a HOLE, and coverage is usable only up to
the FIRST hole - everything behind it is recomputed no matter how much valid data sits
behind it. That is how a 62-block run became a 15-block restore and cost ~19,200 tokens.
So an offer sized on the pool alone converts "short but correct" into "long but holed",
which is strictly worse, and on the box that raises its pool to 1 GiB it is the EXPECTED
case: a 1 GiB pool tells the offer "~93 jobs are affordable" while a 64-job queue says
64, and a maximum 128-span offer needs 128 + 4 = 132.

THE THREE NUMBERS, and the single derivation that sizes all of them. One offer allocates
exactly ONE CPU slot per read job (one spilled file fills one destination block), so
`MAX_SLOTS_PER_OFFER` = 256 is the largest job count ANY offer can ever carry:

  * `disk_read.MAX_QUEUE_DEPTH` = 256 - every job of a maximum offer is submittable.
  * `POOL_PRICE_MAX_IDENTITIES` = 256 - the walk names one identity per job, so a
    maximum offer is priced EXACTLY and its sample is never truncated.
  * `MAX_SLOTS_PER_OFFER` = 256 - unchanged, and the reason the other two are 256.

A queue shallower than the offer's own job ceiling HOLES it; an identity walk shorter
than the offer's identity count UNDER-PRICES it and so OVER-offers it. Both are the same
hole class, arriving from opposite sides, which is why they are pinned in one file.

Offline fakes only: no engine, no torch, no CUDA, no network, no live box. The
real-queue submission tests (132 jobs through `request_read`, and the buffer-vs-queue
memory proof) live in `test_disk_read_threads.py`, whose teardown owns reader-thread
hygiene.
"""

from __future__ import annotations

import pytest
import test_disk_gdn_boundary_restore as gdn
import test_disk_restore as base
from test_disk_restore_pool_budget import (
    PP0_SHAPE,
    PP1_SHAPE,
    fill_sized,
    last_pool,
    pool_at_most_jobs,
)

from kvtier_dev_plugin import (
    disk_index,
    disk_read,
    disk_restore,
    disk_spill,
)

# The fixtures live in the sibling modules; importing the names is how this suite reuses
# them, autouse ones included (a fresh boot per test, and a /tmp root that is allowed).
armed = base.armed                      # noqa: F811 - the gate open, the switch up
allow_tmp_root = base.allow_tmp_root    # noqa: F811 - autouse
fresh = base.fresh                      # noqa: F811 - autouse
offer = base.offer
Manager = base.Manager
Req = base.Req
hybrid = gdn.hybrid                     # the live box's align-mode geometry

BLOCK = base.BLOCK
GDN_GROUPS = gdn.GDN_GROUPS             # the four align-mode boundary-state groups
ATTENTION_GROUP = gdn.ATTENTION_GROUP   # the one positional group on the live box

# THE LIVE GEOMETRY'S JOB BILL. One positional group means one read per span,
# and each of the four align-mode groups contributes exactly ONE boundary-state
# read for the whole run, so a 128-span offer costs 132 jobs - 20 more than the
# queue used to hold.
BOUNDARY_JOBS = len(GDN_GROUPS)         # 4
MAXIMUM_OFFER_JOBS = (disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER + BOUNDARY_JOBS)


def _fill_hybrid(armed, blocks, size=PP0_SHAPE):
    """A contiguous `blocks`-span run on the live box's shape, journalled at `size`."""
    fill_sized(armed, blocks=blocks, groups=GDN_GROUPS, size=size)
    fill_sized(armed, blocks=blocks, groups=(ATTENTION_GROUP,), size=size)
    return blocks


# --- 1. the queue is a priced bound, and it names itself -----------------------

def test_when_the_queue_is_the_binding_bound_bounded_by_names_read_queue(armed,
                                                                        monkeypatch):
    """THE POINT OF THE CHANGE: a reader can tell WHICH limit cut the offer.

    A pool that affords 40 jobs and a queue that affords 20. The queue decides, and
    `bounded_by` says `read_queue` - not `pool_budget`, which is what this offer used to
    report, and which points an operator at a bigger pool that would have changed
    nothing. A half-truth in this field is not a missing datum: it is a wrong diagnosis
    with a number attached.

    Both ceilings are published beside the label, because "which bound" and "how far
    below the other one" are the two questions a short restore actually asks.
    """
    blocks = _fill_hybrid(armed, 20)
    pool_at_most_jobs(monkeypatch, 40, PP0_SHAPE)
    monkeypatch.setattr(disk_read, "MAX_QUEUE_DEPTH", 20)
    tokens, flag = offer(hybrid(), Req(blocks=blocks), 0, (0, False))
    assert flag is True, disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["bounded_by"] == ["read_queue"], (
        f"the queue cut this offer, so the queue must be named: {pricing}")
    assert pricing["jobs_the_pool_can_carry"] == 40, pricing
    assert pricing["jobs_the_queue_can_carry"] == 20, pricing
    assert pricing["jobs_affordable"] == 20, pricing
    # 20 slots minus the 4 boundary states leaves 16 spans, at one read per span.
    assert pricing["jobs_offered"] == 20, pricing
    assert pricing["jobs_dropped"] == (blocks + BOUNDARY_JOBS) - 20, pricing
    assert tokens == 16 * BLOCK, tokens
    counters = disk_restore.counters()
    assert counters["offer_queue_capped"] == 1, counters
    assert counters["offer_pool_capped"] == 0, (
        "counting a queue-clamped offer under the pool would put the fix in the wrong "
        "column: this counter is the one that says 'a bigger pool would not have "
        "helped this offer'")


def test_the_pool_keeps_its_own_name_when_the_pool_is_the_binding_bound(armed,
                                                                       monkeypatch):
    """THE OLD LABEL SURVIVES, and is not quietly widened to mean "some job bound".

    40 jobs of pool, the shipped 256-job queue: the pool decides and `pool_budget` is
    still the answer. Half of naming a new bound is not moving an existing one, and the
    published queue figure is what proves the queue was never the problem here.
    """
    blocks = _fill_hybrid(armed, 20)
    pool_at_most_jobs(monkeypatch, 12, PP0_SHAPE)
    tokens, flag = offer(hybrid(), Req(blocks=blocks), 0, (0, False))
    assert flag is True, disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["bounded_by"] == ["pool_budget"], pricing
    assert pricing["jobs_the_queue_can_carry"] == disk_read.MAX_QUEUE_DEPTH, pricing
    assert pricing["jobs_affordable"] == 12, pricing
    assert pricing["jobs_offered"] == 12, pricing
    counters = disk_restore.counters()
    assert counters["offer_pool_capped"] == 1
    assert counters["offer_queue_capped"] == 0


@pytest.mark.parametrize("pool_jobs,queue_depth", [
    (40, 20),      # the queue binds
    (20, 40),      # the pool binds
    (30, 30),      # a tie: the pool is named, having cost the offer nothing
    (60, 12),      # the queue binds hard
    (12, 60),      # the pool binds hard
    (100, 24),     # a 1 GiB pool's shape, with a queue exactly as deep as the run
    (24, 100),
])
def test_the_arithmetic_never_reports_more_jobs_than_the_smaller_bound(armed,
                                                                      monkeypatch,
                                                                      pool_jobs,
                                                                      queue_depth):
    """`jobs_affordable` IS `min(pool, queue)` - as a relation, over a ladder.

    One lucky pair proves nothing, which is the lesson of the deleted 32,000-token cap:
    a number that happened to be safe looked like a bound and was only an accident. So
    the claim is asserted as an inequality at every rung, and the rung also names the
    bound. The offer's own job count is checked against the smaller ceiling, because
    that is the number the bind loop would have exceeded.

    A run of 20 spans on the live geometry wants 20 + 4 = 24 jobs, so a ceiling at or
    above that cuts NOTHING and `bounded_by` stays empty - the honest answer, and the
    one that keeps this ladder from asserting a bound that never bit.
    """
    blocks = _fill_hybrid(armed, 20)
    wanted = blocks + BOUNDARY_JOBS           # 24 jobs
    pool_at_most_jobs(monkeypatch, pool_jobs, PP0_SHAPE)
    monkeypatch.setattr(disk_read, "MAX_QUEUE_DEPTH", queue_depth)
    smaller = min(pool_jobs, queue_depth)
    tokens, flag = offer(hybrid(), Req(blocks=blocks), 0, (0, False))
    pricing = last_pool()
    assert pricing["jobs_affordable"] == smaller, pricing
    if smaller < BOUNDARY_JOBS + 1:
        # Not even one span fits above the boundary states: NO offer, named for the
        # bound that refused, rather than a hole-shaped offer.
        assert (tokens, flag) == (0, False), (tokens, flag)
        refused = disk_restore.counters()["refused"]
        assert refused["no_read_queue" if queue_depth < pool_jobs
                           else "no_pool_budget"] == 1, refused
        return
    assert flag is True, disk_restore._state.get("last_reason")
    offered = pricing.get("jobs_offered", wanted)
    if smaller >= wanted:
        # Nothing needed cutting, so the offer carries its whole 24 jobs and no bound is
        # credited - a bound that never bit must never appear in `bounded_by`.
        assert offered == wanted, pricing
        expected = []
    else:
        # THE INVARIANT: the offer's own job count never exceeds the ceiling the reader
        # can carry, because an excess is not a shorter restore but a hole at the first
        # refused job.
        assert offered <= smaller, pricing
        assert offered > 0, pricing
        expected = (["read_queue"] if queue_depth < pool_jobs
                    else ["pool_budget"])
    assert pricing["bounded_by"] == expected, pricing


def test_a_queue_that_cannot_hold_one_span_refuses_with_its_own_reason(armed,
                                                                      monkeypatch):
    """Fail closed, and name the bound: the same rule as `no_pool_budget`, new bound.

    The queue holds exactly the four boundary-state jobs, so a span cannot be claimed.
    The honest answer is NO OFFER, not a shorter one: a job the queue refuses at bind is
    `queue_full`, and a `queue_full` block is a hole, and a hole at the run's first
    block is a restore of nothing. The refusal kind is new (`no_read_queue`) rather than
    a reuse of `no_pool_budget`, because the fix is a deeper queue and an operator
    reading `no_pool_budget` would go and buy RAM.
    """
    blocks = _fill_hybrid(armed, 20)
    pool_at_most_jobs(monkeypatch, 100, PP0_SHAPE)
    monkeypatch.setattr(disk_read, "MAX_QUEUE_DEPTH", BOUNDARY_JOBS)
    assert offer(hybrid(), Req(blocks=blocks), 0, (0, False)) == (0, False)
    counters = disk_restore.counters()
    assert counters["refused"]["no_read_queue"] == 1, counters["refused"]
    assert "the reader queue holds" in disk_restore._state["last_reason"]
    why = disk_restore.OFFER_REFUSAL_WHY["no_read_queue"]
    assert "HOLE" in why and "deeper reader queue" in why, why
    # The load-failure record is filed under the refusal the bind WOULD have raised, so
    # a queue-limited offer shows up where queue refusals are looked for.
    kinds = [str(record.get("kind")) for record in disk_restore._load_failures]
    assert "queue_full" in kinds, kinds


# --- 2. the identity walk: the same hole class from the price side --------------

def test_a_capped_identity_walk_refuses_to_trust_its_sample(armed, monkeypatch):
    """A PARTIAL SAMPLE'S MAXIMUM IS NOT A MAXIMUM, so a capped walk prices the floor.

    The store holds one shape (11,468,800 B) and the box's widest rank is 12,902,400 B.
    With the walk wide enough to see every identity the offer would read, the price is
    the exact journalled figure. Cap the walk and the sample could just as well have
    landed on the narrow half of a two-shape store - and an UNDER-price makes the offer
    too BIG, so the surplus dies at bind and the restore is holed. A capped walk
    therefore takes the larger of its sample and the box's width floor, and says so.

    THE DIRECTION IS THE WHOLE TEST: the truncated price must never be SMALLER than the
    exact one. Shortening an offer costs a recompute; lengthening it on a guess costs
    the hole.
    """
    blocks = _fill_hybrid(armed, 12)
    monkeypatch.setattr(disk_spill, "slot_width_bytes", lambda: None)
    monkeypatch.setattr(
        disk_restore, "worker_proof",
        lambda force=False: {"ok": True, "count": 1, "ranks": ["PP0_TP0"],
                             "reason": None, "ranks_armed": 1, "ranks_seen": 1,
                             "reads_applied": 0, "reason_kind": None,
                             "bytes_per_file_max": PP1_SHAPE,
                             "bytes_per_file_ranks": 2})
    pool_at_most_jobs(monkeypatch, 100, PP1_SHAPE)

    identities = [[(ATTENTION_GROUP, base.key_hex(seed, ATTENTION_GROUP))]
                  for seed in range(1, blocks + 1)]
    exact = disk_restore._pool_price_bytes(identities, blocks)
    assert exact[0] == PP0_SHAPE, exact
    assert exact[1].startswith("exact:"), exact

    monkeypatch.setattr(disk_restore, "POOL_PRICE_MAX_IDENTITIES", 4)
    capped = disk_restore._pool_price_bytes(identities, blocks)
    assert capped[0] == PP1_SHAPE, (
        f"a truncated walk must price the floor, not its own sample: {capped}")
    assert capped[1].startswith("walk capped and widened"), capped
    assert disk_restore.counters()["offer_price_walk_capped"] == 1

    # And through the offer path, so the counter an operator reads is the one that
    # moved.
    offer(hybrid(), Req(blocks=blocks), 0, (0, False))
    pricing = last_pool()
    assert pricing["bytes_per_job"] == PP1_SHAPE, pricing
    assert pricing["price_source"].startswith("walk capped and widened"), pricing
    assert disk_restore.counters()["offer_price_walk_capped"] >= 2


def test_a_maximum_offer_is_priced_exactly_at_the_shipped_walk_width(armed,
                                                                    monkeypatch):
    """At 256 identities the walk sees a whole maximum offer, so `exact:` is the answer.

    A maximum offer names 132 identities - ONE PER READ JOB, which is exactly why the
    walk cap and the queue depth are the same number: `_disk_run` collects the keys this
    offer will read, so its identity count IS its job count (128 spans + 4 boundary
    states), and 132 fits the 256 cap. That is what the raise buys: no truncation guard
    needed, no conservative widening, and a counter that stays at zero in a healthy boot
    - so the day it is non-zero, a bound moved under the offer path.
    """
    cap = disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER
    _fill_hybrid(armed, cap)
    # A pool that affords the WHOLE maximum offer: 132 jobs of the live shape.
    pool_at_most_jobs(monkeypatch, MAXIMUM_OFFER_JOBS, PP0_SHAPE)
    tokens, flag = offer(hybrid(), Req(blocks=cap), 0, (0, False))
    assert flag is True, disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["price_source"].startswith("exact:"), pricing
    assert "walk capped" not in pricing["price_source"], pricing
    assert disk_restore.counters()["offer_price_walk_capped"] == 0, (
        "a maximum offer must not be priced on a sample")
    # The full 132-job offer survives pricing: neither the pool nor the queue cut it, so
    # no `jobs_offered` clamp figure exists and no bound is credited.
    assert pricing["jobs_the_offer_needed"] == MAXIMUM_OFFER_JOBS, pricing
    assert pricing["jobs_offered"] == MAXIMUM_OFFER_JOBS, (
        f"a maximum offer must arrive at the bind loop whole, or the surplus is "
        f"a hole at the first refused job: {pricing}")
    assert pricing["blocks_offered"] == cap, pricing
    # Only the positional walk is credited here: the pool and the queue both carried the
    # whole offer, which is the whole point of sizing them from the offer's job ceiling.
    assert "read_queue" not in pricing["bounded_by"], pricing
    assert "pool_budget" not in pricing["bounded_by"], pricing
    assert pricing["jobs_affordable"] >= MAXIMUM_OFFER_JOBS, pricing
    assert tokens == cap * BLOCK, tokens


# --- 3. the hole mechanism, named so a future change cannot miss it ------------

def test_an_offer_is_never_sized_above_the_queue_because_a_hole_is_not_smaller(armed,
                                                                              monkeypatch):
    """REGRESSION GUARD FOR THE HOLE MECHANISM. Change one of the three numbers without
    the others and this fails.

    A 62-block run came home as a 15-block restore and cost ~19,200 tokens because the
    blocks that did not make it were not reported as missing - they arrived as NOTHING,
    and coverage is usable only up to the FIRST hole. The offer path is the only place
    that can prevent that shape: by the time `request_read` refuses job N+1, the offer
    has already promised N+1 spans of coverage and the engine has already planned around
    it.

    So the invariant is asserted where the promise is made: NO offer's job count may
    exceed the queue it will be submitted into. The middle of the test pins the chain
    that makes an excess a hole rather than a shorter restore, so a reader who wants to
    relax one of these bounds has to argue with the mechanism, not with a constant.
    """
    blocks = _fill_hybrid(armed, 40)
    pool_at_most_jobs(monkeypatch, 40, PP0_SHAPE)    # the pool affords the whole run
    monkeypatch.setattr(disk_read, "MAX_QUEUE_DEPTH", 20)
    tokens, flag = offer(hybrid(), Req(blocks=blocks), 0, (0, False))
    assert flag is True, disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["jobs_offered"] <= disk_read.MAX_QUEUE_DEPTH, (
        "an offer with more jobs than the queue holds is a HOLE at job "
        f"{disk_read.MAX_QUEUE_DEPTH}: every valid block behind the first refused "
        "one is recomputed anyway, so the extra promise buys nothing and costs the "
        "run its coverage. This is the 62-block-run-became-15-blocks failure.")
    assert pricing["bounded_by"] == ["read_queue"], pricing

    # THE CHAIN, pinned so the reasoning cannot be lost with the numbers:
    #  1. a queue refusal is a CAPACITY failure, not a timing one...
    assert "queue_full" in disk_restore.LOAD_CAPACITY_FAILURES
    assert "queue_full" not in disk_restore.LOAD_TIMING_FAILURES
    #  2. ...which is the class that means "the bytes are fine and the read never
    #     happened", i.e. the block is absent rather than wrong...
    assert disk_restore.failure_kind_class("queue_full") == "capacity"
    #  3. ...and an absent block is why the offer has to be sized to the queue.
    assert "queue_full" in disk_read.REFUSALS


def test_the_three_numbers_are_sized_from_the_offers_own_job_ceiling():
    """`MAX_SLOTS_PER_OFFER` is 256, one read job per slot, and the other two follow.

    THE DERIVATION, so nobody has to guess why three unrelated-looking constants move
    together: one offer allocates one CPU slot per read job (one spilled file fills one
    destination block), so the CPU-slot cap is the offer's job ceiling, and

      * the queue must be at least that deep, or the surplus jobs are refused and the
        restore is HOLED at the first one;
      * the identity walk must be at least that wide, or the price is a partial sample
        whose maximum can under-state the real bill, the offer is too big, and the
        restore is holed from that side instead.

    A single-group model needs the cap + 4 jobs (`MAX_RESTORE_BLOCKS_PER_OFFER` spans
    plus one boundary-state block per align-mode group), which is the arithmetic the
    last assertion pins. If you lower one of these three without the others, this test
    is the failure that explains why they are not independent.
    """
    jobs = disk_restore.MAX_SLOTS_PER_OFFER
    assert jobs <= disk_read.MAX_QUEUE_DEPTH, (
        f"the queue holds {disk_read.MAX_QUEUE_DEPTH} jobs but one offer may allocate "
        f"{jobs} CPU slots, and it allocates one read job per slot: the surplus "
        "arrives as NOTHING and coverage stops at the first hole")
    assert jobs <= disk_restore.POOL_PRICE_MAX_IDENTITIES, (
        f"the price walk samples {disk_restore.POOL_PRICE_MAX_IDENTITIES} identities "
        f"but one offer names up to {jobs}, so its maximum is a partial "
        "sample's maximum and an under-price over-offers - the same hole class from "
        "the price side")
    assert jobs >= MAXIMUM_OFFER_JOBS, (
        f"a maximum single-group offer needs {MAXIMUM_OFFER_JOBS} jobs "
        f"({disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER} spans + {BOUNDARY_JOBS} "
        f"boundary states) and the slot cap is {jobs}, so the slot clamp would cut "
        f"the offer below the block cap the operator just raised")
    # 256 was not picked for headroom; it is the offer's job ceiling, so extra queue
    # slots can only be filled by a SECOND concurrent offer - the open reservation
    # (TODO 14), which a queue depth does not fix.
    assert jobs == 256


# --- 4. the published view ----------------------------------------------------

def test_the_offer_pool_view_publishes_both_job_ceilings_and_their_minimum(armed):
    """A reader can see which bound is operative WITHOUT making an offer.

    One number called "what the offer can carry" is exactly the half-truth this field
    existed to fix: it was the pool's answer, read as the offer's answer, on a box whose
    queue was the real limit. Both ceilings and the minimum are published, each with the
    name of what it measures.
    """
    fill_sized(armed, blocks=2, groups=GDN_GROUPS, size=PP0_SHAPE)
    fill_sized(armed, blocks=2, groups=(ATTENTION_GROUP,), size=PP0_SHAPE)
    offer(Manager(), Req(blocks=2), 0, (0, False))
    view = disk_restore.section()["offer_pool"]
    assert view["jobs_the_pool_can_carry"] == disk_read.pool_state()["budget_bytes"] \
        // view["bytes_per_job"], view
    assert view["jobs_the_queue_can_carry"] == disk_read.MAX_QUEUE_DEPTH, view
    assert view["jobs_affordable"] == min(view["jobs_the_pool_can_carry"],
                                          view["jobs_the_queue_can_carry"]), view
    assert view["identity_walk_cap"] == disk_restore.POOL_PRICE_MAX_IDENTITIES, view
    for key in ("queue_capped_offers", "price_walk_capped_offers"):
        assert key in view, key
    assert "jobs_the_queue_can_carry" in view["note"], view["note"]
    scheduler = disk_restore.scheduler_section()["offer_pool"]
    for key in ("jobs_affordable", "jobs_the_queue_can_carry", "identity_walk_cap"):
        assert scheduler[key] == view[key], (key, scheduler, view)


def test_the_new_counters_are_published_with_a_why():
    """A counter a payload cannot describe is a counter nobody acts on.

    `offer_queue_capped` and `offer_price_walk_capped` both point at a specific fix (a
    deeper queue; a wider identity walk), so the description has to name the fix or the
    number is decoration.
    """
    counters = disk_restore.counters()
    assert counters["offer_queue_capped"] == 0
    assert counters["offer_price_walk_capped"] == 0
    why = disk_restore.counters()["units"]
    assert "MAX_QUEUE_DEPTH" in why["offer_queue_capped"], why["offer_queue_capped"]
    assert "hole" in why["offer_queue_capped"].lower()
    assert "partial sample" in why["offer_price_walk_capped"]
    assert disk_restore.OFFER_REFUSAL_WHY["no_read_queue"]


def test_the_refreshed_index_never_leaks_a_stale_identity_into_the_walk(armed):
    """Fixture honesty check: the walk prices identities the INDEX still holds.

    Not a product claim - a guard on this file's own reasoning. `_pool_price_bytes` asks
    `disk_index.entry_bytes` per identity, so a test that journalled a size and then
    refreshed away from it would be asserting a number no offer path can see. Refreshed
    here, and the sampled size comes back exactly.
    """
    fill_sized(armed, blocks=3, groups=(ATTENTION_GROUP,), size=PP0_SHAPE)
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    price, source = disk_restore._pool_price_bytes(
        [[(ATTENTION_GROUP, base.key_hex(seed, ATTENTION_GROUP))] for seed
         in (1, 2, 3)], 3)
    assert price == PP0_SHAPE, (price, source)
    assert source.startswith("exact:"), source
