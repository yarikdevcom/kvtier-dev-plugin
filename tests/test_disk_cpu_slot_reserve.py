r"""THE CPU-SLOT RESERVE IS A KNOB NOW (0.2.38): `VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE`.

`CPU_SLOT_RESERVE` was 64, hardcoded, and it was chosen for an arena many times bigger
than the one the operator runs now. The engine's KV arena went 64 GiB -> 8 GiB, which
left 187 allocatable CPU slots per rank on the PP0 pair and 166 on the wide PP1 one, so
a 64-slot reservation eats 41% of what is left on the wide rank BEFORE the offer path
sees a slot - and an offer is clamped to `(free - reserve - boundary slots) // slots per
span` spans. On the live geometry (one positional group, four align-mode boundary
states) that is 98 spans of offer at reserve 64 and 146 at reserve 16; 313,600 tokens
versus 467,200 tokens of a prefix that is already on the SSD and unreadable. Lowering
the reserve is therefore the cheapest host-RAM win left on that box, and it needs no
engine change.

WHAT THE RESERVE IS FOR, because a knob without its purpose gets set to 0 by the first
reader in a hurry: an offer allocates its DESTINATION slots out of the arena's CPU block
pool, the same pool the RAM tier parks its own blocks in and the same pool the spill
path reads OUT of. The reserve is the headroom an offer deliberately leaves that
traffic. The test below named `test_zero_is_honoured_...` states the failure mode of 0
rather than pretending it is free.

WHAT THIS SUITE PINS, in order:

1. THE DEFAULT IS 64 AND NOTHING MOVED. An unset environment resolves to 64, the shipped
   refusal for a pool with no room above 64 still fires, and the boot line says the
   number came from the default.
2. THE ARITHMETIC, ON THE LIVE NUMBERS. Reserve 64 at 166 free buys 98 spans; reserve 16
   buys 146; the difference is exactly the 48 slots the change freed, asserted through
   the real offer path at the real 12,902,400-byte width and again as a table over both
   ranks.
3. THE SAFETY CASE. A reserve AT or ABOVE the free count declines every offer
   (`no_cpu_slots`), allocates nothing, oversubscribes nothing and raises nothing. This
   is why there is no ceiling on the variable: honouring an absurd value is the SAFE
   direction, because the reserve can only ever make an offer smaller.
4. THE GRAMMAR. 0 is honoured (a posture, not a dead knob); a non-integer or a negative
   falls back to 64 and produces ONE log line, not one per offer.
5. PUBLICATION. The value in force appears in the offer/restore section, in the section
   that names the bound (`bounded_by`), in the arena verdict, in the boot line with its
   origin, and inside the `no_cpu_slots` refusal reason.
6. NO KEY MOVED. Moving the knob changes values and no key set - the payload's, the
   scheduler section's, the arena verdict's (healthy, tiny, no-manager and the
   exception path alike) and the rank file's.
7. NO HALF-MIGRATION. An AST scan asserts the bare constant is read by exactly one
   function: the resolver. A second reader would be a bound that quietly stops being the
   bound.

Offline fakes only: no engine, no torch, no CUDA, no network, no live box.
"""

from __future__ import annotations

import ast
import inspect
import logging

import pytest
import test_disk_gdn_boundary_restore as gdn
import test_disk_restore as base
from test_disk_restore_pool_budget import (
    PP1_SHAPE,
    fill_sized,
    last_pool,
    pool_at_most_jobs,
)

from kvtier_dev_plugin import (
    backport,
    config,
    disk_restore,
)

# The fixtures live in the sibling modules; importing the names is how this suite reuses
# them, autouse ones included (a fresh boot per test, and a /tmp root that is allowed).
armed = base.armed                      # noqa: F811 - the gate open, the switch up
allow_tmp_root = base.allow_tmp_root    # noqa: F811 - autouse
fresh = base.fresh                      # noqa: F811 - autouse
fill = base.fill
offer = base.offer
Manager = base.Manager
CpuPool = base.CpuPool
Req = base.Req
hybrid = gdn.hybrid                     # the live box's align-mode geometry

BLOCK = base.BLOCK
GDN_GROUPS = gdn.GDN_GROUPS
ATTENTION_GROUP = gdn.ATTENTION_GROUP
RESERVE = disk_restore.CPU_SLOT_RESERVE
ENV = config.ENV_DISK_CPU_SLOT_RESERVE

# THE LIVE BOX, transcribed from the operator's measurement after the arena shrank:
# 166 free slots per rank on the wide PP1 pair, 187 on PP0, one slot per span (one
# positional group) plus four boundary states, and a file of 12,902,400 B on the wide
# rank. `STORE_SPANS` is deliberately BELOW the 128-span walk cap, so the reserve - and
# not the walk - is the bound these tests are measuring.
LIVE_FREE_PP1 = 166
LIVE_FREE_PP0 = 187
BOUNDARY = len(GDN_GROUPS)
STORE_SPANS = 120


def live_manager(free: int = LIVE_FREE_PP1):
    """The live box's scheduler, on a pool holding exactly `free` allocatable slots."""
    return hybrid(pool=CpuPool(num_blocks=free))


def offered_spans(pricing: dict) -> int:
    return int(pricing["spans_the_cpu_slots_can_carry"])


# --- 1. the default is 64, and nothing moved for anyone who set nothing ---------

def test_an_unset_environment_still_reserves_sixty_four_slots(armed, monkeypatch):
    """THE SHIPPED BEHAVIOUR, UNCHANGED: unset = 64, and 64 still refuses what it did.

    This is the test that makes the release a knob and not a behaviour change. The
    grammar says "the operator said nothing" (`slots is None`, no problems), the
    resolver answers with the constant, and the pool-with-no-room-above-64 case that
    `test_disk_restore.py::test_the_offer_never_touches_the_cpu_reserve` pins still gets
    no offer at all. A default that drifted would pass every other test in this file and
    still be a regression, so it is pinned twice: as a number, and as an outcome.
    """
    monkeypatch.delenv(ENV, raising=False)
    grammar = config.cpu_slot_reserve_env()
    assert grammar["slots"] is None and grammar["problems"] == []
    assert grammar["env_var"] == ENV and grammar["raw_value"] is None

    verdict = disk_restore.cpu_slot_reserve_verdict()
    assert disk_restore.cpu_slot_reserve() == RESERVE == 64
    assert verdict["cpu_slot_reserve"] == 64 and verdict["default_slots"] == 64
    assert verdict["source"] == "default" and verdict["raw_value"] is None

    limits = disk_restore.section()["limits"]
    assert limits["cpu_slot_reserve"] == 64
    assert limits["cpu_slot_reserve_default"] == 64
    assert limits["cpu_slot_reserve_source"] == "default"
    assert "cpu_slot_reserve=64(default)" in disk_restore.boot_line()

    thin = CpuPool(num_blocks=RESERVE + 2)
    fill(armed, blocks=4)
    assert offer(Manager(pool=thin), Req(blocks=4)) == (0, False)
    assert thin.allocated == []
    assert disk_restore.counters()["refused"]["no_cpu_slots"] == 1


def test_lowering_the_reserve_opens_the_very_pool_the_default_refused(armed,
                                                                     monkeypatch):
    """THE KNOB DOES SOMETHING, on the smallest case: the same 66-slot pool.

    Same fixture as the test above, one environment variable apart: at reserve 2 the
    arithmetic has room for 4 spans of 3 groups and the offer stands. Without this half
    the pair, "the reserve is configurable" would be a payload field and nothing more.
    """
    monkeypatch.setenv(ENV, "2")
    pool = CpuPool(num_blocks=RESERVE + 2)
    fill(armed, blocks=4)
    assert disk_restore.cpu_slot_reserve() == 2
    assert offer(Manager(pool=pool), Req(blocks=4)) == (4 * BLOCK, True)
    assert disk_restore.counters()["refused"]["no_cpu_slots"] == 0
    assert disk_restore.counters()["offers"] == 1


# --- 2. the arithmetic, on the live box's own numbers --------------------------

def test_a_smaller_reserve_raises_the_affordable_count_by_exactly_the_slots_freed(
        armed, monkeypatch):
    """THE OPERATOR'S NUMBERS, THROUGH THE REAL OFFER PATH: 64 -> 98, 16 -> 146.

    166 free slots, per_span = 1, four boundary states, files priced at the live
    wide-rank 12,902,400 B, and a bounce-buffer pool made deliberately too roomy to be
    the bound - so the ONLY thing clamping this offer is the arena reserve. At reserve
    64 the offer is 98 spans = 313,600 tokens and `bounded_by` says `cpu_slots`; at 16
    the arithmetic affords 146 and the run itself (120 spans) is what stops the offer,
    with no
    `cpu_slots` bound left in it. 146 - 98 is 48, which is 64 - 16: the slots the change
    freed, and not one slot more.

    The published `bytes_per_job` is the real 12,902,400 either way. It arrives from the
    width floor rather than an exact walk at this depth - 120 spans x 5 groups is 600
    identities against a 256-identity price walk - and the floor is the same number,
    because the fallback constant IS the wide shape; the figure, and so the arithmetic,
    does not depend on which of the two produced it.
    """
    fill_sized(armed, blocks=STORE_SPANS, groups=GDN_GROUPS, size=PP1_SHAPE)
    fill_sized(armed, blocks=STORE_SPANS, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)
    pool_at_most_jobs(monkeypatch, 200, PP1_SHAPE)   # the pool must NOT be the bound

    monkeypatch.setenv(ENV, "64")
    wide = live_manager()
    tokens, flag = offer(wide, Req(request_id="req-64", blocks=STORE_SPANS), 0,
                         (0, False))
    assert flag is True, disk_restore._state.get("last_refusal")
    assert tokens == 98 * BLOCK == 313_600
    pricing = last_pool()
    assert pricing["bounded_by"] == ["cpu_slots"], pricing["bounded_by"]
    assert pricing["cpu_slots_free"] == LIVE_FREE_PP1
    assert pricing["cpu_slot_reserve"] == 64
    assert pricing["cpu_slot_reserve_source"] == "env"
    assert pricing["cpu_slot_reserve_env_var"] == ENV
    assert offered_spans(pricing) == 98
    assert pricing["bytes_per_job"] == PP1_SHAPE
    assert wide.cpu_block_pool.get_num_free_blocks() == LIVE_FREE_PP1 - 102, \
        "98 spans of 1 slot plus 4 boundary states is 102 slots, and no more"

    monkeypatch.setenv(ENV, "16")
    same = live_manager()
    tokens, flag = offer(same, Req(request_id="req-16", blocks=STORE_SPANS), 0,
                         (0, False))
    assert flag is True, disk_restore._state.get("last_refusal")
    assert tokens == STORE_SPANS * BLOCK, "the reserve stopped binding, so the run does"
    pricing = last_pool()
    assert pricing["cpu_slot_reserve"] == 16
    assert offered_spans(pricing) == 146
    assert "cpu_slots" not in pricing["bounded_by"], pricing["bounded_by"]
    assert offered_spans(pricing) - 98 == 64 - 16, (
        "the affordable count rose by exactly the slots the reserve gave back")

    # AND THE OFFER IS WHAT IT EVER WAS below the walk cap: the reserve is one bound of
    # several, and 146 spans of arithmetic cannot become 146 spans of offer, because the
    # positional walk stops at `MAX_RESTORE_BLOCKS_PER_OFFER`.
    assert disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER == 128
    assert offered_spans(pricing) > disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER, \
        "this is the rung where the walk, not the reserve, is the operative bound"


@pytest.mark.parametrize("free, reserve, spans", [
    (LIVE_FREE_PP1, 64, 98),      # the live PP1 rank as it runs today
    (LIVE_FREE_PP1, 32, 130),     # already past the walk cap: no more offer to be had
    (LIVE_FREE_PP1, 16, 146),     # the operator's proposal
    (LIVE_FREE_PP1, 0, 162),      # the posture, not a recommendation
    (LIVE_FREE_PP0, 64, 119),     # the narrow rank: 64 costs it less, and still buys
    (LIVE_FREE_PP0, 32, 151),     # ...nothing, for the same walk-cap reason
    (LIVE_FREE_PP0, 16, 167),
])
def test_the_affordance_arithmetic_is_the_live_geometry(free, reserve, spans):
    """THE RULE, published as a function and asserted as a table.

    `spans = (free - reserve - boundary_slots) // per_span`, with the live geometry
    (per_span 1, boundary 4) at both ranks' real free counts. The table is the argument
    for the knob AND the limit on it: below about 34 on the wide rank and 55 on the
    narrow one the affordable count exceeds `MAX_RESTORE_BLOCKS_PER_OFFER` (128), so the
    walk cap decides the offer and a smaller reserve buys nothing but risk. A reader who
    takes "lower the reserve" as an unbounded instruction is reading a number out of a
    comment instead of out of this table.
    """
    got = disk_restore.cpu_slot_affordance(free, per_span=1, boundary_slots=BOUNDARY,
                                           reserve=reserve)
    assert got["spans_affordable"] == spans, got
    assert got["slots_affordable"] == spans + BOUNDARY, (
        "a run costs its spans AND its four boundary states, and the offer's `needed` "
        "is comparable against this number")
    assert got["free_blocks"] == free and got["cpu_slot_reserve"] == reserve
    assert got["per_span"] == 1 and got["boundary_slots"] == BOUNDARY


# --- 3. the safety case: an over-large reserve declines, it never oversubscribes --

@pytest.mark.parametrize("reserve", ["166", "200"])
def test_a_reserve_at_or_above_the_free_count_declines_offers(armed, monkeypatch,
                                                              reserve):
    """THE FAILURE MODE OF AN ABSURD VALUE: no offer, no allocation, no exception.

    There is deliberately no ceiling on the variable, so this is the behaviour that has
    to be true rather than merely likely: a reserve of 166 against 166 free slots (or
    200 against 166) leaves `(free - reserve - 4) // 1` = 0 spans, and the offer dies at
    `no_cpu_slots` exactly as a genuinely tiny arena does. The pool is untouched - not
    one slot allocated, the free count unmoved - because the clamp runs BEFORE any
    allocation, and the path must not raise on its way out: a refusal that throws takes
    the request's RAM hit with it.

    WHY THIS IS THE SAFE DIRECTION, and so why clamping would be wrong: the reserve can
    only ever make an offer SMALLER. Silently clamping 10,000 to 64 would turn "the
    operator asked for a reserve larger than the arena" into "the operator asked for
    64", which is a different machine than the one they asked for and a lie in the
    payload.
    """
    monkeypatch.setenv(ENV, reserve)
    fill_sized(armed, blocks=12, groups=GDN_GROUPS, size=PP1_SHAPE)
    fill_sized(armed, blocks=12, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)
    pool_at_most_jobs(monkeypatch, 64, PP1_SHAPE)
    assert disk_restore.cpu_slot_reserve() == int(reserve)

    pool = CpuPool(num_blocks=LIVE_FREE_PP1)
    before = pool.get_num_free_blocks()
    errors = disk_restore.counters()["errors"]
    fallbacks = disk_restore.counters()["fallbacks"]
    assert offer(hybrid(pool=pool), Req(blocks=12), 0, (0, False)) == (0, False)
    counters = disk_restore.counters()
    assert counters["refused"]["no_cpu_slots"] == 1
    assert counters["offers"] == 0, "a declined offer is not a short offer"
    assert pool.allocated == [], "the clamp runs before the allocation, not after"
    assert pool.get_num_free_blocks() == before
    assert counters["errors"] == errors and counters["fallbacks"] == fallbacks, \
        "a refusal is arithmetic, not an exception swallowed"

    reason = disk_restore._state["last_refusal"]
    assert reason.startswith("no_cpu_slots:"), reason
    assert f"{LIVE_FREE_PP1} free blocks, {reserve} are reserved (env)" in reason, \
        "the refusal names the reserve IN FORCE, and says the environment set it"


# --- 4. the grammar: 0 is a posture, a non-number is a fallback with a log line --

def test_zero_is_honoured_because_it_is_a_posture_and_not_a_dead_knob(armed,
                                                                     monkeypatch):
    """0 IS ACCEPTED, unlike every other integer knob in this package - and this names
    what it costs, because the next reader will set it.

    `ceiling_env` refuses 0 (a ceiling pass with no budget does nothing while reading as
    configured) and `disk_read_env` refuses 0 threads for the same reason. A reserve of
    0 is not that shape: it means "an offer may take the arena's last free slot", which
    is coherent arithmetic and a real posture for a box that exists to restore prefixes.

    WHAT IT COSTS, stated instead of discovered: the RAM tier parks its own blocks in
    this same pool, and the spill path reads OUT of it. A boot at 0 lets a burst of disk
    offers take the arena to zero free slots, so the tier above has nowhere to park and
    the next spill has no destination - the box trades a RAM hit for a recompute in
    order to buy a disk hit. It is also the only setting under which an offer can
    allocate the LAST slot out from under a request that was about to need it.
    Honourable on a box that is disk-restoring everything and RAM-parking nothing; the
    reason the default is 64 and not 0.
    """
    monkeypatch.setenv(ENV, "0")
    grammar = config.cpu_slot_reserve_env()
    assert grammar["slots"] == 0 and grammar["problems"] == []
    verdict = disk_restore.cpu_slot_reserve_verdict()
    assert verdict["cpu_slot_reserve"] == 0 and verdict["source"] == "env"
    assert disk_restore.cpu_slot_reserve() == 0
    assert disk_restore.cpu_slot_affordance(
        LIVE_FREE_PP1, 1, BOUNDARY, 0)["spans_affordable"] == 162
    assert disk_restore.section()["limits"]["cpu_slot_reserve"] == 0
    assert "cpu_slot_reserve=0(env)" in disk_restore.boot_line()

    thin = CpuPool(num_blocks=RESERVE + 2)
    fill(armed, blocks=4)
    assert offer(Manager(pool=thin), Req(blocks=4)) == (4 * BLOCK, True)


@pytest.mark.parametrize("value", ["16.0", "-1", "sixteen", "16 slots", "0x10"])
def test_a_value_that_cannot_be_a_reserve_falls_back_to_64_and_says_so_once(
        armed, monkeypatch, caplog, value):
    """A TYPO IS THE SHIPPED DEFAULT PLUS ONE LOG LINE, not a crashed boot and not one
    line per offer.

    The three cases a payload must keep apart are "unset", "the environment's number
    won" and "the environment said something the grammar refused" - which is why the
    fallback is 64 (the conservative value, the one every release before this used) and
    `source` says `ignored` rather than `default`. The log line goes through the
    module's `notice_once` idiom, so a boot that resolves the reserve a thousand times
    on the request path prints ONE line; a warning per offer on the step path is how a
    diagnostic becomes an outage.

    A NEGATIVE value is in this list and not in the honoured ones because it is not a
    smaller reserve: `free - (-1)` hands the offer MORE slots than the pool has, and the
    fork's own `get_new_blocks` raises when a count exceeds the free list.
    """
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv(ENV, value)
    grammar = config.cpu_slot_reserve_env()
    assert grammar["slots"] is None and len(grammar["problems"]) == 1, grammar
    assert grammar["raw_value"] == value

    for _ in range(5):
        assert disk_restore.cpu_slot_reserve() == RESERVE
    ignored = [record.getMessage() for record in caplog.records
               if "DISK[reserve] IGNORED" in record.getMessage()]
    assert len(ignored) == 1, ignored
    assert ENV in ignored[0] and value in ignored[0], ignored[0]
    assert "64" in ignored[0], "the line must say what stood in, not only what failed"

    verdict = disk_restore.cpu_slot_reserve_verdict()
    assert verdict["source"] == "ignored" and verdict["raw_value"] == value
    assert verdict["env_problems"] == grammar["problems"]
    assert disk_restore.section()["limits"]["cpu_slot_reserve"] == 64
    assert disk_restore.section()["limits"]["cpu_slot_reserve_source"] == "ignored"
    assert disk_restore.section()["offer_pool"]["cpu_slot_reserve_problems"] \
        == grammar["problems"]
    assert "cpu_slot_reserve=64(env-ignored)" in disk_restore.boot_line()
    assert disk_restore.counters()["errors"] == 0, "a refused value is not an error"


def test_a_reserve_below_the_boundary_states_still_offers_nothing(armed, monkeypatch):
    """A SMALL RESERVE IS NOT A FREE OFFER: the boundary states are still spent first.

    On the live geometry a run cannot be claimed without one state slot in each of the
    four align-mode groups, and the arithmetic nets those off BEFORE the reserve: at
    reserve 2 against a 6-slot pool there is room for 0 spans, not 4. Getting this order
    wrong is the difference between a short offer and a run the fork will not accept.
    """
    monkeypatch.setenv(ENV, "2")
    got = disk_restore.cpu_slot_affordance(6, per_span=1, boundary_slots=BOUNDARY,
                                           reserve=2)
    assert got["spans_affordable"] == 0
    fill_sized(armed, blocks=4, groups=GDN_GROUPS, size=PP1_SHAPE)
    fill_sized(armed, blocks=4, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)
    pool = CpuPool(num_blocks=6)
    assert offer(hybrid(pool=pool), Req(blocks=4), 0, (0, False)) == (0, False)
    assert pool.allocated == []
    assert disk_restore.counters()["refused"]["no_cpu_slots"] == 1


# --- 5. publication ------------------------------------------------------------

def test_the_effective_reserve_is_published_wherever_the_bound_is_named(armed,
                                                                        monkeypatch):
    """Every surface that names the bound names the SAME number, and says where it came
    from.

    `section()["limits"]` (the payload's bounds), the offer/restore section beside
    `jobs_the_pool_can_carry`, the same view through `scheduler_section()`, the
    per-offer pricing that `bounded_by` belongs to, the arena verdict that decides
    whether the tier is inert at boot, the `no_cpu_slots` refusal reason and the closed
    vocabulary's
    sentence for it. A bound that is published in one of those and applied from another
    is the failure this release was about, so the assertion is made against one
    environment value (16) everywhere at once.
    """
    monkeypatch.setenv(ENV, "16")
    fill_sized(armed, blocks=4, groups=GDN_GROUPS, size=PP1_SHAPE)
    fill_sized(armed, blocks=4, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)

    section = disk_restore.section()
    assert section["limits"]["cpu_slot_reserve"] == 16
    assert section["limits"]["cpu_slot_reserve_env_var"] == ENV
    assert section["offer_pool"]["cpu_slot_reserve"] == 16
    assert section["offer_pool"]["cpu_slot_reserve_default"] == 64
    assert section["offer_pool"]["cpu_slot_reserve_env_var"] == ENV
    scheduler = disk_restore.scheduler_section()
    assert scheduler["offer_pool"]["cpu_slot_reserve"] == 16
    assert "cpu_slot_reserve" in scheduler["offer_pool"]["note"], \
        "the note that explains the bounds must explain this one too"

    verdict = disk_restore.arena_verdict(Manager(pool=CpuPool(num_blocks=100)))
    assert verdict["cpu_slot_reserve"] == 16
    assert verdict["blocks_usable_for_an_offer"] == 84, (
        "the boot verdict measures the arena against the reserve IN FORCE")
    assert "16-block reserve" in verdict["reason"], verdict["reason"]

    # THE ARENA TOO SMALL TO OFFER, with the variable named in the sentence the boot
    # refusal prints, because "arena is tiny" without the lever that fixes it is a
    # dead end at 3 a.m.
    tiny = disk_restore.arena_verdict(Manager(pool=CpuPool(num_blocks=12)))
    assert tiny["state"] == "tiny"
    assert tiny["blocks_usable_for_an_offer"] == 12 - 16, \
        "the usable figure is the SIGNED difference, so an over-large reserve reads " \
        "as the deficit it is instead of flattening to a meaningless zero"
    assert "VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE" in tiny["reason"], tiny["reason"]
    assert "KV_OFFLOAD_SIZE" in tiny["reason"], \
        "the reserve is one lever; the arena's size is the other and both must be named"

    # THE REFUSAL REASON: the number in force, from the environment, in the reason the
    # operator reads for the lookup that got nothing.
    pool_at_most_jobs(monkeypatch, 64, PP1_SHAPE)
    assert offer(live_manager(free=16), Req(blocks=4), 0, (0, False)) == (0, False)
    reason = disk_restore._state["last_refusal"]
    assert reason.startswith("no_cpu_slots:") and "16 are reserved (env)" in reason
    assert ENV in disk_restore.OFFER_REFUSAL_WHY["no_cpu_slots"]
    assert ENV in disk_restore.scheduler_section()["refusal_reasons"]["no_cpu_slots"]


def test_the_per_offer_pricing_answers_the_question_the_bound_asks(armed, monkeypatch):
    """WHY WAS MY OFFER ONLY 98 JOBS? - answered by the payload, not by this file.

    `bounded_by: ["cpu_slots"]` used to be the end of the story. The pricing now carries
    the reserve, its source, the variable that changes it, what the pool had free, and
    the spans that arithmetic leaves: the terms of `(166 - 64 - 4) // 1` are all in the
    one dict, so the decision to lower the reserve can be made FROM a payload instead of
    from a code read.
    """
    fill_sized(armed, blocks=STORE_SPANS, groups=GDN_GROUPS, size=PP1_SHAPE)
    fill_sized(armed, blocks=STORE_SPANS, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)
    pool_at_most_jobs(monkeypatch, 200, PP1_SHAPE)
    monkeypatch.setenv(ENV, "64")
    assert offer(live_manager(), Req(blocks=STORE_SPANS), 0, (0, False))[1] is True
    pricing = last_pool()
    free, reserve = pricing["cpu_slots_free"], pricing["cpu_slot_reserve"]
    assert offered_spans(pricing) == (free - reserve - BOUNDARY) // 1
    assert (free, reserve) == (LIVE_FREE_PP1, 64)
    assert pricing["cpu_slot_reserve_source"] == "env"
    assert pricing["cpu_slot_reserve_env_var"] == ENV
    # ...and the same figures reach the payload's copy of the view.
    assert disk_restore.scheduler_section()["last_offer_pool"][
        "cpu_slot_reserve"] == 64


# --- 6. no key set moves -------------------------------------------------------

class _RaisingPool:
    """A pool whose free count cannot be read: the arena verdict's exception path."""

    blocks = (1, 2, 3)

    def get_num_free_blocks(self):
        raise RuntimeError("no free count today")


def _key_sets() -> dict:
    section = disk_restore.section()
    scheduler = disk_restore.scheduler_section()
    return {
        "section": set(section),
        "limits": set(section["limits"]),
        "offer_pool": set(section["offer_pool"]),
        "scheduler": set(scheduler),
        "scheduler_offer_pool": set(scheduler["offer_pool"]),
        "arena_no_manager": set(disk_restore.arena_verdict(None)),
        "arena_ok": set(disk_restore.arena_verdict(Manager())),
        "arena_tiny": set(disk_restore.arena_verdict(
            Manager(pool=CpuPool(num_blocks=4)))),
        "arena_unreadable": set(disk_restore.arena_verdict(
            Manager(pool=_RaisingPool()))),
        "rank_file": set(disk_restore.worker_report()),
    }


def test_moving_the_knob_moves_values_and_no_key(armed, monkeypatch):
    """THE KEY SET IS STABLE ACROSS THE KNOB, AND ACROSS THE ARENA'S DEGRADED VIEWS.

    A payload field that appears only when a variable is set is a field a dashboard
    cannot bind to and a diff cannot be trusted on, so the new keys are published
    UNCONDITIONALLY and this test proves it by moving the knob between three states
    (unset, honoured, ignored) and comparing every key set. The arena verdict is checked
    in all four of its states at once - `no_manager`, `ok`, `tiny` and the `unreadable`
    that its own exception path builds - because that view's degraded shape and its
    healthy shape are required to carry the same keys, and the dict the exception builds
    by hand is the one a future edit forgets.

    The rank file's key set is checked too: it lives inside a
    `worker_diag.MAX_CONTRIBUTION_BYTES` budget, which is precisely why this knob's
    value is published in the scheduler's views and NOT added to it.
    """
    monkeypatch.delenv(ENV, raising=False)
    unset = _key_sets()
    assert "cpu_slot_reserve" in unset["limits"]
    assert "cpu_slot_reserve" in unset["offer_pool"]
    assert unset["arena_no_manager"] == unset["arena_ok"] == unset["arena_tiny"] \
        == unset["arena_unreadable"], (
        "the arena's degraded views must read like its healthy one, key for key")
    assert unset["arena_ok"] == set(backport._boot_arena(None)), \
        "the boot verdict's own fallback dict carries the same keys"

    monkeypatch.setenv(ENV, "16")
    assert _key_sets() == unset
    monkeypatch.setenv(ENV, "not-a-number")
    assert _key_sets() == unset
    assert disk_restore.section()["offer_pool"]["cpu_slot_reserve_problems"] != []

    # The VALUES move, which is the other half: a stable key set that publishes a stale
    # number is the bug this release exists to fix.
    assert disk_restore.section()["limits"]["cpu_slot_reserve"] == 64
    monkeypatch.setenv(ENV, "16")
    assert disk_restore.section()["limits"]["cpu_slot_reserve"] == 16
    assert disk_restore.scheduler_section()["offer_pool"]["cpu_slot_reserve"] == 16


# --- 7. the boot line ----------------------------------------------------------

def test_the_boot_line_says_whose_number_the_reserve_is(armed, monkeypatch):
    """ONE LINE, IN THE STYLE OF THE OTHER BOOT LINES, WITH THE ORIGIN.

    `DISK[restore] ... offer_block_cap=128 offer_block_cap_tokens=409600
    cpu_slot_reserve=64(default) ...`: the reserve sits with the other bounds that size
    an offer, because that is where a reader greps for "what limited this boot". The
    suffix is the whole point - `default`, `env`, `env-ignored` - since the number alone
    cannot answer "did my drop-in arrive?", which is the question the line exists for.
    Three states, three different lines, and a line that stays inside the module's own
    length budget so the tail (which carries the reason) is not what gets truncated.
    """
    def line():
        return disk_restore.boot_line()

    monkeypatch.delenv(ENV, raising=False)
    assert "cpu_slot_reserve=64(default)" in line()
    monkeypatch.setenv(ENV, "16")
    assert "cpu_slot_reserve=16(env)" in line()
    assert "cpu_slot_reserve=64" not in line()
    monkeypatch.setenv(ENV, "16.0")
    assert "cpu_slot_reserve=64(env-ignored)" in line()

    monkeypatch.setenv(ENV, "16")
    text = line()
    assert len(text) <= disk_restore.MAX_BOOT_LINE_CHARS * 4, text
    # THE FIELD IS CHEAP ENOUGH TO PAY FOR. The line is truncated at the cap, and the
    # piece at the tail is `reason=` (itself clipped to `MAX_BOOT_LINE_CHARS`) - the one
    # field an operator needs most. So the property is not "the line is short" but "the
    # line still has room for a whole reason after this field was added".
    assert len(text) + disk_restore.MAX_BOOT_LINE_CHARS \
        <= disk_restore.MAX_BOOT_LINE_CHARS * 4, text
    assert text.index("cpu_slot_reserve=") < text.index("offer_block_cap=") \
        + len("offer_block_cap=128 offer_block_cap_tokens=409600") + 60, \
        "the reserve belongs with the other per-offer bounds, not at the tail"


# --- 8. no half-migrated constant ------------------------------------------------

def test_no_code_reads_the_cpu_slot_reserve_constant_directly():
    """AN ENFORCED RULE, NOT A COMMENT: exactly one function reads the constant.

    "A half-migrated constant is worse than an untunable one" - the payload used to
    publish `CPU_SLOT_RESERVE` while the offer path computed with it, and both read as
    "the bound". So the name survives as the DEFAULT, and this scan is what stops it
    becoming a second source of truth: every LOAD of the name must live inside the one
    resolver (`cpu_slot_reserve_verdict`), and there must be exactly one Store (the
    definition). A new reader is a bug this test catches at the line number, which is
    the only honest way to keep a promise like this one over a 9,000-line module.
    """
    tree = ast.parse(inspect.getsource(disk_restore))
    loads, stores, inside = [], [], {}
    for function in [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and node.id == "CPU_SLOT_RESERVE":
                inside.setdefault(function.name, []).append(node.lineno)
                if isinstance(node.ctx, ast.Load):
                    loads.append(id(node))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "CPU_SLOT_RESERVE" \
                and isinstance(node.ctx, ast.Store):
            stores.append(node.lineno)
    assert len(stores) == 1, f"the constant is defined once, at {stores}"
    assert set(inside) == {"cpu_slot_reserve_verdict"}, \
        f"the resolver is the only reader, found {inside}"
    unclaimed = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Name) and node.id == "CPU_SLOT_RESERVE"
                 and isinstance(node.ctx, ast.Load) and id(node) not in loads]
    assert unclaimed == [], f"module-level reads of the default: {unclaimed}"
