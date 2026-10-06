r"""THE PER-OFFER BOUNDS AFTER THE FIXED TOKEN CAP WAS DELETED (0.2.32).

Every release up to 0.2.31 fixed the per-offer ceiling at 32,000 tokens - 10 spans of a
3,200-token scheduler block - and that number was an ARTIFICIAL ceiling on restoring a
prefix from disk: below what the process can carry, below what the bounce-buffer pool
can afford at the operator's live 256 MiB setting, and below the run depth the store
reaches anyway. It is DELETED, along with its clamp and its `offer_clamped` counter.
The deletion was made for a reason, and it knowingly accepted a risk; both are recorded
at `MAX_RESTORE_BLOCKS_PER_OFFER` and at the clamp's former site in `_offer`. This file
is where the consequences are pinned.

WHY THE DELETION IS SAFE, in the one sentence the incident needs: the cap was a proxy
for "the engine must be able to PLACE what we offer", and since 0.2.31 the offer path
computes that directly - `disk_read.afford_reads()` says how many bounce buffers the
pool can hold at once, the offer is cut to it, and `bounded_by` says which bound bit
(`pool_budget` / `cpu_slots` / `block_cap`). So the guard is now the thing itself, and
these tests assert IT rather than a token constant that only looked like a guard.

THE ORDER A REVIEWER SHOULD READ THIS IN:

1. THE PRIMARY PROPERTY, asserted as a relation and not a number: an offer never
   carries more jobs than the pool can hold at once. Asserted over a ladder of pool
   sizes, because "60,800 tokens" would pass while the property was broken - which is
   precisely how the 32,000 constant came to be trusted as a safety net it never was.
2. THE BOUND THAT SURVIVES: `MAX_RESTORE_BLOCKS_PER_OFFER` blocks, where the positional
   walk stops, published in blocks AND in the tokens they are worth with the block
   size's source. `offer_cap` and `offer_clamped` are pinned ABSENT, because a field
   named for a deleted bound is a number a dashboard keeps plotting.
3. THE ACCEPTED RISK, stated as a positive test: with a pool that can afford a
   160,000-token offer, one is now possible - the incident's exact size. Nothing but
   affordability stands between this path and that boot, and the arbiter (TODO 14) that
   would make it safe under concurrency is open.
4. THE SURFACES: boot line, `section()["limits"]`, and `GET /plugins/kvtier/snapshot`.

WHAT THIS SUITE DELIBERATELY DOES NOT CLAIM: that a long prefix is restorable. The
deepest contiguous run the live store held on the boot the operator measured was ~11
spans, because spill follows RAM-tier parking and the CPU arena is ~187 blocks per
rank, so a 34-span prefix is unreachable in one pass at any ceiling; the tests that use
a deeper run say SYNTHETIC FIXTURE in their names.

Offline fakes only: no engine, no torch, no CUDA, no network, no live box.
"""

from __future__ import annotations

import pytest
import test_disk_gdn_boundary_restore as gdn
import test_disk_restore as base
from test_disk_restore_pool_budget import (
    PP0_SHAPE,
    fill_sized,
    last_pool,
    pool_at_most_jobs,
)

from kvtier_dev_plugin import (
    config,
    disk_read,
    disk_restore,
    snapshot,
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
WIDTH = base.WIDTH
GDN_GROUPS = gdn.GDN_GROUPS
ATTENTION_GROUP = gdn.ATTENTION_GROUP

# THE OPERATOR'S PROMPT LENGTHS AND SETTINGS, transcribed: the live prompts that made
# this bound visible are 109,034 and 116,026 tokens, the pool is
# `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` = 268,435,456 (256 MiB), and the deepest
# contiguous run the store held was ~11 spans.
SYNTHETIC_DEEP_SPANS = 34
LIVE_RUN_DEPTH_SPANS = 11
LIVE_POOL_BYTES = 268_435_456
# The size of every one of the incident's 123,474 offers. It is no longer unreachable by
# any cap, which is the point of `test_the_pool_is_now_the_only_guard...` below.
INCIDENT_OFFER_TOKENS = 160_000


# --- 1. THE PRIMARY PROPERTY: an offer never outruns the pool ------------------

@pytest.mark.parametrize("jobs", [4, 5, 6, 9, 11, 23])
def test_the_offer_never_carries_more_jobs_than_the_pool_can_hold(armed, monkeypatch,
                                                                  jobs):
    """REGRESSION GUARD FOR 2026-10-05, stated as the engine-placeable property.

    A 34-span run on the live geometry costs 34 positional reads (group 5, one file per
    span) plus one boundary-state read in each of the four align-mode groups: 38 jobs.
    Whatever the pool affords is then the whole answer: the offer's job count is at most
    `jobs_the_pool_can_carry` and its bytes are inside the budget, which is what "the
    engine could place them" MEANS at the bind point. Every rung of the ladder gets the
    same two assertions, because a single lucky pool size proves nothing - and a token
    count would prove less still, since the deleted constant was exactly a token count
    that made this property look handled.

    A pool that cannot hold the four boundary states PLUS one span gets NO offer at all
    (`no_pool_budget`): zero jobs is trivially placeable, and a shorter offer is what
    used to be refused at bind and fuse the request.
    """
    fill_sized(armed, blocks=SYNTHETIC_DEEP_SPANS, groups=GDN_GROUPS, size=PP0_SHAPE)
    fill_sized(armed, blocks=SYNTHETIC_DEEP_SPANS, groups=(ATTENTION_GROUP,),
               size=PP0_SHAPE)
    budget = pool_at_most_jobs(monkeypatch, jobs, PP0_SHAPE)
    answer = offer(hybrid(), Req(blocks=SYNTHETIC_DEEP_SPANS), 0, (0, False))
    counters = disk_restore.counters()
    if jobs < len(GDN_GROUPS) + 1:
        assert answer == (0, False), answer
        assert counters["offers"] == 0
        assert counters["refused"]["no_pool_budget"] == 1
        return
    tokens, flag = answer
    assert flag is True, disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["jobs_offered"] <= pricing["jobs_the_pool_can_carry"], pricing
    assert pricing["jobs_offered"] == jobs, (
        "the pool is the binding bound, so the offer is exactly what it can carry")
    assert pricing["bytes_offered"] <= budget, (
        "an offer whose bytes exceed the pool IS the incident: refused at bind as "
        "`buffer_cap`, pruned into `invalid_block_ids`, rescheduled as a load failure")
    assert pricing["bounded_by"] == ["pool_budget"], pricing["bounded_by"]
    assert counters["offer_pool_capped"] == 1
    assert counters["offer_block_capped"] == 0, "34 spans never reaches the block walk"
    assert "offer_clamped" not in counters, "the token clamp is deleted, not zeroed"
    assert tokens % BLOCK == 0 and tokens > 0, "whole spans only"


def test_a_tiny_pool_bounds_a_long_prefix_and_no_cap_takes_credit(armed, monkeypatch):
    """THE tiny-pool regression: nine jobs is nine jobs, whatever the prompt wants.

    The primary guard for the 2026-10-05 failure mode, isolated on purpose: a run of 34
    spans (108,800 tokens) costs 38 jobs, the pool affords nine, and the offer is five
    spans with `bounded_by == ["pool_budget"]`. Every other bound in the offer path
    would have allowed more here - the block bound stands at 64 spans, and 34 spans x 1
    positional group plus 4 boundary slots is nowhere near `MAX_SLOTS_PER_OFFER` - which
    is exactly the point: after the deletion, the BYTE bound is the one protecting the
    engine, so the test that matters asserts the byte bound and not a token figure.
    """
    fill_sized(armed, blocks=SYNTHETIC_DEEP_SPANS, groups=GDN_GROUPS, size=PP0_SHAPE)
    fill_sized(armed, blocks=SYNTHETIC_DEEP_SPANS, groups=(ATTENTION_GROUP,),
               size=PP0_SHAPE)
    pool_at_most_jobs(monkeypatch, 9, PP0_SHAPE)
    tokens, flag = offer(hybrid(), Req(blocks=SYNTHETIC_DEEP_SPANS), 0, (0, False))
    assert flag is True, disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert tokens == 5 * BLOCK
    assert pricing["jobs_the_offer_needed"] == (SYNTHETIC_DEEP_SPANS
                                                + len(GDN_GROUPS))
    assert pricing["jobs_the_pool_can_carry"] == 9
    assert pricing["jobs_offered"] == 9
    assert pricing["bounded_by"] == ["pool_budget"], pricing["bounded_by"]
    assert "offer_cap" not in pricing["bounded_by"], "retired with the cap"
    counters = disk_restore.counters()
    assert counters["offer_pool_capped"] == 1
    assert counters["offer_block_capped"] == 0


# --- 2. the accepted risk, stated as a test -----------------------------------

def test_the_pool_is_now_the_only_guard_and_a_160000_token_offer_is_possible(armed,
                                                                            monkeypatch):
    """THE risk the deletion accepted, pinned as a FACT rather than a worry.

    The incident's offers were EXACTLY 160,000 tokens. Nothing in this module refuses
    that size any more - the cap that did is deleted - so a pool that can afford the
    jobs gets an offer larger than the incident's. That is the operator's considered
    trade, recorded at MAX_RESTORE_BLOCKS_PER_OFFER: today's offer rate (4 in 332
    lookups) makes it a non-issue, and the arbiter that would make it safe under
    concurrency (TODO 14: reserve at adoption, release at `finish_batch`) is still open,
    because two simultaneous offers price affordability against the pool's CURRENT free
    bytes and can over-commit in total. This test exists so nobody rediscovers that
    under load.
    """
    blocks = 60                              # 192,000 tokens of contiguous run
    fill_sized(armed, blocks=blocks)         # journalled at WIDTH = 4,096 B
    # A pool wide enough to afford 60 spans x 3 positional groups = 180 jobs.
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, str(200 * WIDTH))
    disk_read.reset_state_for_tests()
    tokens, flag = offer(Manager(), Req(blocks=blocks), 0, (0, False))
    assert flag is True, disk_restore._state.get("last_reason")
    assert tokens == blocks * BLOCK
    assert tokens > INCIDENT_OFFER_TOKENS, (
        "an offer bigger than the incident's is now reachable by design")
    pricing = last_pool()
    assert pricing["bounded_by"] == [], pricing
    assert pricing["block_cap_blocks"] == disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER
    counters = disk_restore.counters()
    assert counters["offers"] == 1
    assert "offer_clamped" not in counters


# --- 3. the bound that survives: blocks, published honestly -------------------

def test_the_block_bound_is_published_in_blocks_and_in_tokens_with_its_source():
    """`offer_block_ceiling()`: the walk's limit, and where the multiplier came from.

    The field this replaces (`offer_cap`) advertised the deleted 32,000-token cap. A
    payload reader needs the bound that DOES bind, and needs to see that the token
    figure is a derived product rather than a configured number - hence the blocks, the
    tokens, the block size, and the block size's SOURCE. The source is not decoration:
    a worker rank has no manager to ask, and a payload that claimed `engine` from a
    fallback constant would be a payload making a claim it cannot support.
    """
    disk_restore.reset_state_for_tests()
    assert (disk_restore.offer_block_ceiling()["block_size_source"] == "fallback")
    seen = disk_restore.offer_block_ceiling(1600)
    assert seen["blocks"] == disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER == 64
    assert seen["block_size"] == 1600 and seen["block_size_source"] == "engine"
    assert seen["tokens"] == 64 * 1600
    disk_restore._state["offer_block_size_seen"] = 3200
    remembered = disk_restore.offer_block_ceiling()
    assert remembered["block_size_source"] == "last-lookup"
    assert remembered["tokens"] == 64 * 3200
    # The bound is BLOCKS: the same 64 whatever the block size, because the walk counts
    # spans and one offer can only name so many spans inside its deadline.
    assert remembered["blocks"] == seen["blocks"]
    # And the payload says plainly that this is not what makes a long prefix restorable.
    assert "binds in practice" in remembered["note"]


def test_a_SYNTHETIC_deep_run_hits_the_block_bound_and_says_block_cap(armed):
    """SYNTHETIC FIXTURE (60+ contiguous spilled spans); the live store held ~11.

    What it pins: when the walk stops at `MAX_RESTORE_BLOCKS_PER_OFFER` while the
    request still has tokens left, `bounded_by` says `block_cap` and
    `offer_block_capped`
    increments - and the retired `offer_cap` name stays out of the record, because a
    block bound reporting itself as a token-cap clamp is how a counter stops being
    evidence.
    """
    blocks = disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER + 8
    fill_sized(armed, blocks=blocks)
    tokens, flag = offer(Manager(), Req(blocks=blocks), 0, (0, False))
    assert flag is True, disk_restore._state.get("last_reason")
    assert tokens == disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER * BLOCK
    assert disk_restore.counters()["offer_block_capped"] == 1
    bounds = last_pool()["bounded_by"]
    assert bounds == ["block_cap"], (
        f"the block bound must never read as the retired `offer_cap`: {bounds!r}")


# --- 4. the surfaces -----------------------------------------------------------

def test_the_boot_line_names_the_block_bound_and_not_the_retired_cap():
    """The line at boot: the bound in blocks, the same bound in tokens, no `offer_cap`.

    `offer_cap=32000` used to be on every restore boot line. After the deletion that
    number would be a fiction, so the name is gone from the line as well as the code -
    and a field an operator greps for in a log that no longer emits it is worse than a
    renamed one, which is why the retired spelling is asserted ABSENT here.
    """
    disk_restore.reset_state_for_tests()
    line = disk_restore.boot_line()
    assert f"offer_block_cap={disk_restore.MAX_RESTORE_BLOCKS_PER_OFFER}" in line, line
    assert "offer_block_cap_tokens=" in line, line
    assert "offer_cap=" not in line, line
    assert len(line) <= disk_restore.MAX_BOOT_LINE_CHARS * 4


def test_the_snapshot_surfaces_the_block_bound_and_retires_the_cap():
    """`GET /plugins/kvtier/snapshot`: the bound as fields, the retired names gone.

    The route reaches this through `snapshot.disk_fill()`'s `disk_restore` section, so
    these are what a dashboard reads. Both halves matter: the new fields must be present
    AND unit-labelled (the repo's rule that no number in that payload is bare), and the
    retired ones must be absent so a dashboard cannot keep plotting a bound that
    constrains nothing.
    """
    engine = type("E", (), {"scheduler": None})()
    payload = snapshot.snapshot_for_engine(
        engine, type("P", (), {"ensure_loaded": lambda self: {
            "ok": False, "error": "no policy in this test", "traceback": None},
            "decide": None})())
    view = payload["disk_restore"]
    ceiling = disk_restore.offer_block_ceiling()
    assert view["offer_block_cap_blocks"] == ceiling["blocks"] == 64
    assert view["offer_block_cap_tokens"] == ceiling["tokens"]
    assert view["offer_block_cap_block_size"] == ceiling["block_size"]
    assert view["offer_block_cap_block_size_source"] == ceiling["block_size_source"]
    assert view["offer_block_capped"] == 0
    for key in ("offer_block_cap_blocks", "offer_block_cap_tokens",
                "offer_block_cap_block_size",
                "offer_block_cap_block_size_source", "offer_block_capped"):
        assert key in view, key
        assert str(view["units"].get(key, "")).strip(), f"{key} has no unit"
    for retired in ("offer_cap", "offer_cap_source", "offer_cap_default",
                    "offer_cap_configured", "offer_cap_notice"):
        assert retired not in view, retired
    # `section()` carries the whole derivation for the verb that asks for limits.
    limits = disk_restore.section()["limits"]
    assert limits["offer_block_ceiling"]["blocks"] == 64
    assert limits["max_restore_blocks_per_offer"] == 64
    counters = disk_restore.counters()
    assert "offer_clamped" not in counters
    assert "offer_clamped" not in counters["units"]


def test_the_deleted_cap_leaves_no_reachable_name_anywhere():
    """The deletion is a deletion: no constant, no counter, no clamp, no field.

    A cheap sweep, and the kind that pays: a half-deleted feature leaves a name behind
    that reads as a live bound (a constant some future reader "restores", a units entry
    for a counter that never increments, a `bounded_by` value nothing can append). Every
    one of those would be believed, because this module's payload is the only thing most
    readers ever see.
    """
    assert not hasattr(disk_restore, "MAX_OFFER_TOKENS_PER_OFFER")
    assert not hasattr(disk_restore, "offer_token_ceiling")
    counters = disk_restore.counters()
    assert "offer_clamped" not in counters
    for name in ("MAX_OFFER_TOKENS_PER_OFFER", "offer_clamped", "offer_cap"):
        assert name not in disk_restore.section()
    # The pool-pricing view's own note names the retired vocabulary nowhere.
    doc = disk_restore._offer_pool_view()["note"]
    assert "offer_cap" not in doc, doc
