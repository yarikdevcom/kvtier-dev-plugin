r"""A disk offer must CLAIM NOTHING when it cannot RESTORE EVERYTHING (0.2.25).

WHY THIS FILE EXISTS, IN ONE SENTENCE FROM THE LIVE BOX. At 00:08 on 2026-10-06 a
flagged request on a 31,826-token prompt got a mechanically perfect disk read on all
four ranks - `reads_started=9 reads_applied=9 failed={}`, 18 ms, settled on the first
settle check - and answered `duct Register Register Register...` at `temperature=0`
where the identical prompt recomputed from scratch answered the planted code word
correctly. Nothing failed. Every read landed. The answer was wrong anyway, and this
file exists because a restore path can be flawless at its own job and still corrupt
the model.

THE MECHANISM, END TO END, WHICH IS WHAT THE GATE ENFORCES:

1. The eligible-group set was a FILTER: prefix-cacheable, block size equal to the
   scheduler's, and `has_positionally_stable_blocks`. On this hybrid model
   (`Qwen3.8-Flash-Next-W4A16-Merlin`: groups 0-3 `MambaSpec`/GDN with `MambaManager`
   in `align` mode, whose flag is False - `single_type_kv_cache_manager.py:1494`;
   group 4 a QSA ring at block_size 8, not cacheable; group 5 full attention) the
   intersection was `{5}` while the RAM tier's own cacheable set is `[0,1,2,3,5]`.
2. The number that set produces is what `get_num_new_matched_tokens` returns
   (`manager.py:372`), and that return advances `num_computed_tokens` for the WHOLE
   model. The engine skipped prefill of the GDN and QSA layers across the claimed
   28,800 tokens.
3. `update_state_after_alloc` builds load entries only for non-null groups and its
   `n_ext_g == 0 -> continue` (`manager.py:445-470`) is a SILENT no-op, so groups 0-3
   and 4 were CLAIMED AND NEVER LOADED - and nothing reported a failure, so nothing
   came back as a load error and nothing recomputed the claimed range. A GDN recurrent
   state is one running value per sequence: the token-hash-keyed spill store cannot
   address it even in principle. Precisely ONE thing per claimed span is needed from
   such a group, because `MambaManager.find_longest_cache_hit`
   (`single_type_kv_cache_manager.py:1533-1610`) answers a hit with
   `[null_block] * (n - 1)` plus ONE real boundary-state block - and the disk tier
   fills none of it.

So positional stability is a NECESSARY condition (a name to read by) that was treated
as a SUFFICIENT one (a complete claim). The rule now is: if any prefix-cacheable group
cannot be restored for those tokens, the disk offers NOTHING, and it says so under its
own countable name (`refused.incomplete_group_set`) with the blocking groups named in
the sentence, the whole arithmetic published as data (`group_serviceability`) and what
a claim would cost each group published as a table (`last_groups_required`).

NOTHING HERE KNOWS A MODEL. Every predicate comes from the managers themselves
(`prefix_cacheable_group_ids`, `group_block_sizes` against the scheduler block size,
`has_positionally_stable_blocks` and `mamba_cache_mode` per group), so the veto is
automatic: a pure-attention model has nothing blocked and is offered at exactly as
it was - `test_a_pure_attention_configuration_still_offers_normally` is the proof of
that, and it is the most important test in this file, because the gate's whole job is
to refuse ONE shape without touching the other.

WHAT CHANGED IN 0.2.27, AND WHAT DID NOT. An align-mode Mamba/GDN group is no longer
a blocker, because the reason 0.2.25 made it one was a mistake about the fork: the
group's recurrent state is NOT unnameable. `MambaManager.find_longest_cache_hit`
answers a hit with `[null_block]*(n-1)` plus ONE real boundary-state block
(`single_type_kv_cache_manager.py:1591-1607`), the align hand-off publishes that
block's own group-qualified hash (`:2051-2080`), and the tier stores it under exactly
that hash (`manager.py:1050-1078`) - so the disk already holds those files, which is
why the live store's shard `a3/` contains `…-g0-…` through `…-g5-…` for one key16.
The gate therefore opens BY CONSTRUCTION on a model whose every cacheable group has
an implementable contract, and `tests/test_disk_gdn_boundary_restore.py` is the new
suite for the second contract. What did NOT change is the invariant the gate exists
for: a group with NO implementable contract - a non-align group that is not
positionally scannable - still refuses, under the same name, with the same sentence,
and `test_the_hybrid_geometry_refuses_when_no_contract_is_implementable` is that
proof. The 00:08 failure mode (claim the tokens, load the attention group, answer
confidently from a stale recurrent state) is unreachable in both directions: the
offer either has bytes for every group or it does not happen.
"""

from __future__ import annotations

import logging

import fakes_debug
import test_disk_restore as base

from kvtier_dev_plugin import disk_restore

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them (pytest collects fixtures from imported names).
armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
switches_down = base.switches_down  # noqa: F811 - `armed` with both boot switches down
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

BLOCK = base.BLOCK
Manager = base.Manager
Req = base.Req
fill = base.fill
offer = base.offer

# THE LIVE MODEL'S GEOMETRY, transcribed from the rank witnesses at 00:08: a 3,200-token
# scheduler block, four GDN groups at the scheduler block size, a QSA ring at 8 tokens
# (which is why the scheduler block is 3,200: it is the LCM), one full-attention group.
# `block_bytes_by_group` was [80281600, 58982400, 58982400, 1638400, 82944, 132710400];
# bytes do not enter the predicate, only which group is cacheable, which is stable and
# which hands off boundary states. (Those published numbers turned out to double-count
# the layer count at the publisher - see `worker_diag` and the byte-figure test in
# `test_disk_gdn_boundary_restore.py`; NO I/O path uses them. Every spill file on the
# box is the rank's measured slot width: 11,468,800 B on PP0, 12,902,400 B on PP1, the
# same size for a -g0- file and a -g5- file.)
# The layer math behind those roles (g0=7, g1=6, g2=6 g3=1 GDN, g4=6 QSA ring, g5=12
# MLA of 32 layers) is why a g5-only "win" was never the whole prompt anyway: it is at
# most the attention share of prefill.
HYBRID_SIZES = (BLOCK, BLOCK, BLOCK, BLOCK, 8, BLOCK)
HYBRID_CACHEABLE = (0, 1, 2, 3, 5)
GDN_GROUPS = (0, 1, 2, 3)


def hybrid(*, stable_by_group=None, sizes=HYBRID_SIZES,
           cacheable=HYBRID_CACHEABLE, mode="align", **kwargs):
    """A manager answering the predicates the live hybrid box answers.

    `stable_by_group` overrides `has_positionally_stable_blocks` per group; the
    default is the live answer - the four GDN groups False (MambaManager in `align`
    mode), the QSA ring and the full-attention group True. `mode` is the other half of
    the live answer and the half the BLOCK CONTRACT is derived from: "align" (what
    `GET /plugins/kvtier/kvconfig` reports for groups 0-3) gives those groups the
    boundary-state contract, while None says "unstable for a reason we cannot
    implement", which is the shape the gate must still refuse.

    Everything else is the sibling suite's own `Manager`, so the geometry, the pool and
    the fork's three entry points are the ones every other restore test already trusts.
    """
    stable = {group: (group not in GDN_GROUPS) for group in range(len(sizes))}
    stable.update(stable_by_group or {})
    manager = Manager(group_sizes=sizes, cacheable=cacheable, **kwargs)
    items = []
    for group in range(len(sizes)):
        if group in GDN_GROUPS and mode is not None:
            items.append(fakes_debug.manager_item(
                group,
                fakes_debug.mamba_spec(block_size=sizes[group],
                                       mamba_cache_mode=mode),
                stable=bool(stable[group]), mamba_cache_mode=mode))
        else:
            items.append(fakes_debug.manager_item(
                group,
                fakes_debug.attention_spec(block_size=sizes[group]),
                stable=bool(stable[group])))
    manager.cpu_coordinator = fakes_debug.coordinator(items)
    return manager


def scheduler_here(monkeypatch):
    """Say this process IS the scheduler half, so `why_no_restore` answers the question.

    `scheduler_section()` walks the chain - layer, gate, index, worker proof, offers -
    and the FIRST link is "is the layer fitted here". The `armed` fixture opens the
    gate without installing anything (installing belongs to `install()` and its own
    tests), so a section read straight from that fixture reports the layer, correctly.
    Fitting the real layer here would test `install()`, which other suites do far
    better; these two lines are exactly what a fitted scheduler reports, so the walk
    gets past link one and reaches the refusal.
    """
    monkeypatch.setattr(disk_restore, "fitted_methods_here",
                        lambda: list(disk_restore.SCHED_METHODS))
    disk_restore._state["sched_fitted"] = True


# --- 1. the gate: an incomplete claim is refused, by name ---------------------

def test_the_hybrid_geometry_refuses_when_no_contract_is_implementable():
    """A cacheable group with NO implementable contract: `{5}` alone is NOT an offer.

    This is the shape the gate was built for and it still refuses after 0.2.27. The
    manager here answers `has_positionally_stable_blocks = False` for groups 0-3 with
    NO `mamba_cache_mode` to explain it, which is a block shape this tier cannot price
    - not an align hand-off, not a scannable group - so claiming those tokens would be
    the 00:08 claim with nothing behind it. The refusal must be a NAMED, COUNTABLE
    reason (`incomplete_group_set`) and must name the groups that blocked it, because
    the payload is the only place the next on-call reader will look.

    `test_the_align_hybrid_groups_are_restorable_by_contract` is the other half: same
    geometry, `mamba_cache_mode="align"` present, and the gate opens by construction.
    """
    verdict = disk_restore._eligible_groups(hybrid(mode=None), BLOCK)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "incomplete_group_set"
    assert verdict["blocked_group_ids"] == [0, 1, 2, 3]
    assert verdict["restorable_group_ids"] == [5]
    assert verdict["prefix_cacheable_group_ids"] == [0, 1, 2, 3, 5]
    # The reason names the VERDICT and then the BLOCKERS, because `last_refusal` is
    # clipped at 160 characters and both must survive the clip.
    assert "offers NOTHING on this model" in verdict["reason"]
    assert "0, 1, 2, 3" in verdict["reason"]
    assert "no contract this tier implements" in verdict["reason"]
    assert len(verdict["reason"]) <= 160
    # And the contract vocabulary is the code's own, published per group: a reader can
    # tell "no contract" from "boundary state" without knowing what a MambaSpec is.
    assert verdict["blocked_kinds"] == {"0": "unstable", "1": "unstable",
                                        "2": "unstable", "3": "unstable"}
    service = disk_restore._state["group_serviceability"]
    assert service["group_contracts"]["0"]["contract"] == "unstable"
    assert service["boundary_state_group_ids"] == []
    assert service["positionally_stable_group_ids"] == [4, 5]


def test_the_align_hybrid_groups_are_restorable_by_contract():
    """THE GATE OPENS BY CONSTRUCTION, not by a flag: same fixture, `mode="align"`.

    Groups 0-3 answer the two predicates the live box answers (`stable=False`,
    `mamba_cache_mode="align"`), which is precisely the condition the fork's own
    `has_positionally_stable_blocks` override derives from
    (`single_type_kv_cache_manager.py:1494-1497`), so they are restorable and nothing
    is blocked. `blocked` is EMPTY because nothing is left over, not because a check
    was skipped - and `restorable_group_ids` is the whole cacheable set.
    """
    verdict = disk_restore._eligible_groups(hybrid(), BLOCK)
    assert verdict["ok"] is True
    assert verdict["groups"] == [0, 1, 2, 3, 5]
    assert verdict["boundary_group_ids"] == [0, 1, 2, 3]
    assert verdict["positional_group_ids"] == [5]
    assert verdict["contracts"] == {"0": "boundary_state", "1": "boundary_state",
                                    "2": "boundary_state", "3": "boundary_state",
                                    "5": "positional"}
    service = disk_restore._state["group_serviceability"]
    assert service["complete"] is True
    assert service["disk_offers_possible"] is True
    assert service["blocked_group_ids"] == []
    assert service["boundary_state_group_ids"] == [0, 1, 2, 3]
    assert service["unrestorable_cacheable_group_ids"] == []


def test_a_flagged_request_on_the_hybrid_model_gets_the_forks_answer_untouched(
        armed):
    """THE GATE END TO END: no offer, no allocation, no ledger entry, answer unchanged.

    The refusal is the same kind of refusal every other guard in this module is: the
    fork's own tuple comes back IDENTICALLY (`is`, not `==`), the CPU pool never sees a
    request, and the only thing that moves is a counter and a sentence.
    """
    fill(armed, seeds=[1, 2, 3, 4], groups=list(HYBRID_CACHEABLE))
    manager = hybrid(mode=None)
    ram_result = (0, False)
    assert offer(manager, Req(request_id="req-h", blocks=2), 0,
                 ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["refused"]["incomplete_group_set"] == 1
    assert counters["offers"] == 0
    assert counters["slots_allocated"] == 0
    assert manager.cpu_block_pool.allocated == [], ("a refusal must not reserve a "
                                                    "single destination slot")
    assert disk_restore._loads == {}
    assert manager._pending_cpu_hits == {}, "the fork's tuple is not ours to rewrite"
    assert "incomplete_group_set" in str(counters["last_refusal"])


def test_a_pure_attention_configuration_still_offers_normally(armed):
    """THE PROOF THE GATE IS NOT A MODEL SPECIAL CASE: nothing blocked, nothing changed.

    The sibling suite's default manager is a pure-attention shape - every cacheable
    group is whole-scheduler-block AND positionally stable - and it offers the same
    span, over the same groups, with the same per-block slot count it had before this
    gate existed. If this test ever starts refusing, the gate stopped being derived
    from the predicates and became a rule about a model.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()                       # base geometry: all groups stable
    verdict = disk_restore._eligible_groups(manager, BLOCK)
    assert verdict["ok"] is True
    assert verdict["groups"] == list(base.CACHEABLE) == [0, 1, 4]
    made = offer(manager, Req(request_id="req-a", blocks=1), 0, (0, False))
    assert made == (BLOCK, True), "a complete group set still offers a whole block"
    counters = disk_restore.counters()
    assert counters["offers"] == 1
    assert counters["refused"]["incomplete_group_set"] == 0
    service = counters["group_serviceability"]
    assert service["complete"] is True
    assert service["disk_offers_possible"] is True
    assert service["blocked_group_ids"] == []
    assert service["unrestorable_cacheable_group_ids"] == []
    # Every row of the requirement table is a receipt: needed, and filled.
    rows = counters["last_offer"]["groups_required"]
    assert [row["group"] for row in rows] == [0, 1, 4]
    assert all(row["restored"] and row["blocks_filled"] == row["blocks_needed"] == 1
               for row in rows)
    assert counters["last_offer"]["groups_unfilled"] == []
    assert counters["last_groups_required"]["offered"] is True


def test_making_the_gdn_groups_stable_makes_the_whole_model_serviceable(armed):
    """THE OTHER DIRECTION, WHICH IS WHY THIS IS A PREDICATE AND NOT A BAN.

    Same fixture, same store, only the flag flipped: with groups 0-3 also positionally
    stable, the eligible set is the whole cacheable set, nothing is blocked, and the
    disk offers a span across FIVE groups instead of one. A model whose recurrent
    state ever becomes addressable per block gets its disk tier back with no change to
    this file.
    """
    fill(armed, seeds=[1, 2, 3, 4], groups=list(HYBRID_CACHEABLE))
    manager = hybrid(stable_by_group={group: True for group in GDN_GROUPS})
    assert offer(manager, Req(request_id="req-s", blocks=1), 0,
                 (0, False)) == (BLOCK, True)
    counters = disk_restore.counters()
    assert counters["last_offer"]["groups"] == [0, 1, 2, 3, 5]
    assert counters["slots_allocated"] == 5, "one slot per group"
    assert counters["refused"]["incomplete_group_set"] == 0
    assert counters["last_groups_required"]["unfilled_group_ids"] == []


def test_the_veto_follows_the_predicates_and_not_a_group_list():
    """Move the flag, watch the veto move with it. No group id is hardcoded here.

    Three shapes, one rule: (a) one GDN group stable leaves the OTHER three blocking;
    (b) a cacheable group whose block size is not the scheduler's blocks the offer for
    the same reason - it cannot be restored, so it must not be claimed; (c) a scratch
    group (not cacheable) that happens to be unstable blocks NOTHING, because the RAM
    tier never claims it and neither do we.
    """
    one_fixed = disk_restore._eligible_groups(
        hybrid(mode=None, stable_by_group={0: True}), BLOCK)
    assert one_fixed["ok"] is False
    assert one_fixed["blocked_group_ids"] == [1, 2, 3], ("the veto is per group, not "
                                                         "per model")
    assert "1, 2, 3" in one_fixed["reason"]

    # (b) base geometry, group 2 is 1,600 tokens against a 3,200-token scheduler block.
    mismatched = Manager(cacheable=(0, 1, 2, 4))
    verdict = disk_restore._eligible_groups(mismatched, BLOCK)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "incomplete_group_set"
    assert verdict["blocked_group_ids"] == [2]
    assert verdict["blocked_kinds"] == {"2": "geometry"}
    assert "block_size 1600" in verdict["blocked_why"]["2"]

    # (c) the live shape's group 4: unstable-or-stable is IRRELEVANT, it is not
    # cacheable, so the RAM tier claims nothing for it and the veto must not fire.
    scratch = Manager(cacheable=(0, 1, 4))
    scratch.cpu_coordinator = fakes_debug.coordinator([
        fakes_debug.manager_item(group,
                                 fakes_debug.attention_spec(
                                     block_size=base.GROUP_SIZES[group]),
                                 stable=group != 2)
        for group in range(len(base.GROUP_SIZES))])
    assert disk_restore._eligible_groups(scratch, BLOCK)["ok"] is True


def test_no_eligible_group_at_all_keeps_its_own_refusal_name():
    """THE VOCABULARY IS ADDITIVE: when NOTHING qualifies, `no_eligible_group` answers.

    That reason already names instability as a cause, and the two must stay
    distinguishable: "no group can be restored at all" is a different fact from "some
    can, so an offer would be a partial claim" - the first is a geometry the merged
    tuple cannot express, the second is a model that must not be served from disk.
    """
    verdict = disk_restore._eligible_groups(
        hybrid(mode=None,
               stable_by_group={group: False for group in range(6)}), BLOCK)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "no_eligible_group"
    small = disk_restore._eligible_groups(Manager(group_sizes=(100,) * 6), BLOCK)
    assert small["reason_kind"] == "no_eligible_group"


# --- 2. the arithmetic and the requirement table, published ------------------

def test_the_serviceability_arithmetic_is_published_for_one_payload_read(armed,
                                                                          monkeypatch):
    """One JSON object that says the model is un-serviceable, with the group lists.

    `last_offer` says what the last offer WAS; this says what the model PERMITS, read
    from the same section, so nobody has to re-derive three manager predicates - or
    know what a MambaSpec is - to learn that the disk tier can never serve this
    architecture. `why_no_restore` carries the same verdict in prose.
    """
    fill(armed, seeds=[1, 2, 3, 4], groups=list(HYBRID_CACHEABLE))
    scheduler_here(monkeypatch)
    offer(Manager(), Req(request_id="req-x", blocks=1), 0, (0, False))
    # The lookup above was a complete (pure-attention) model; now the hybrid one, and
    # the payload must describe the LAST LOOKUP, refused as it was.
    offer(hybrid(mode=None), Req(request_id="req-y", blocks=1), 0, (0, False))
    section = disk_restore.scheduler_section()
    service = section["group_serviceability"]
    assert service["ok"] is True
    assert service["prefix_cacheable_group_ids"] == [0, 1, 2, 3, 5]
    assert service["positionally_stable_group_ids"] == [4, 5]
    assert service["unrestorable_cacheable_group_ids"] == [0, 1, 2, 3]
    assert service["restorable_group_ids"] == [5]
    assert service["blocked_group_ids"] == [0, 1, 2, 3]
    assert service["complete"] is False
    assert service["disk_offers_possible"] is False
    assert service["scheduler_block_size"] == BLOCK
    # A reader learns the REASON, not just the list.
    assert "align-mode" in service["blocked_why"]["0"]
    assert section["refused"]["incomplete_group_set"] == 1
    assert "CANNOT BE SERVED FROM DISK AT ALL" in section["why_no_restore"]
    assert "incomplete_group_set" in section["why_no_restore"]
    assert "incomplete_group_set" in section["refusal_reasons"]
    assert "boundary-state block" in section["refusal_reasons"]["incomplete_group_set"]
    # Every one of these is documented in the same payload, the section's convention.
    for key in ("group_serviceability", "last_groups_required", "last_aged_out",
                "no_read", "watchdog_fires", "invalid_reports", "aged_out_loads",
                "late_finished_reports", "restored_after_release",
                "disk_gpu_blocks_dma_verified", "witness_kind_now"):
        assert key in section, f"{key} is published"
        assert key in section["units"], f"{key} has a units entry"


def test_the_refusal_publishes_what_the_claimed_span_would_have_cost_each_group(
        armed, monkeypatch):
    """ITEM 7, THE INSTRUMENT. One object shows a half-covering claim by itself.

    The 00:08 request claimed 28,800 tokens = 9 scheduler blocks. Priced per group,
    that is 9 blocks for the full-attention group and ONE boundary-state block for
    each GDN group (the rest of that group's span is null blocks in the engine's own
    hit contract) - and the disk filled the full-attention ones and nothing else. The
    refusal path publishes exactly that table even though no offer was made, because
    the requirement is a fact about the claim, not about whether it happened.
    """
    fill(armed, seeds=list(range(1, 20)), groups=list(HYBRID_CACHEABLE))
    scheduler_here(monkeypatch)
    manager = hybrid(mode=None)
    nine_blocks = 9 * BLOCK
    request = Req(request_id="req-r", blocks=9, num_tokens=nine_blocks + 900)
    offer(manager, request, 0, (0, False))
    required = disk_restore.counters()["last_groups_required"]
    assert required["offered"] is False
    assert required["refused_as"] == "incomplete_group_set"
    assert required["claimed_blocks"] == 9
    assert required["claimed_tokens"] == nine_blocks
    by_group = {row["group"]: row for row in required["rows"]}
    assert by_group[5]["contract"] == "positional"
    assert by_group[5]["blocks_needed"] == 9
    assert by_group[5]["null_blocks"] == 0
    assert by_group[5]["restored"] is True
    for group in GDN_GROUPS:
        row = by_group[group]
        assert row["contract"] == "unstable"
        # A group with NO implementable contract says `blocks_needed: null` rather
        # than a guess. The 0.2.25 row for this shape claimed "1 block, 8 nulls", but
        # that arithmetic belongs to the align-mode contract (see
        # `test_disk_gdn_boundary_restore.py`), and pricing a group whose block shape
        # this tier does not implement is exactly the kind of number that reads as a
        # plan and turns out to be a wish.
        assert row["blocks_needed"] is None
        assert row["real_blocks"] is None
        assert row["restored"] is False
        assert row["blocks_filled"] == 0
    assert required["unfilled_group_ids"] == [0, 1, 2, 3]
    assert required["blocks_needed_total"] == 9
    assert required["blocks_filled_total"] == 9
    # And the prose in the payload carries the same arithmetic, so `why_no_restore`
    # alone already tells the story.
    why = disk_restore.scheduler_section()["why_no_restore"]
    assert "last_groups_required.rows" in why
    assert "0, 1, 2, 3" in why


def test_the_offer_path_prices_the_final_span_not_the_ceiling(armed):
    """THE RECEIPT MATCHES THE OFFER. A clamped offer's table describes the CLAMP.

    `last_groups_required` on the offer door is computed from the offer's final
    `blocks`, after the per-offer token cap, the slot cap and the pool's free count
    have all had their say, so the table can never describe a span larger than the one
    the scheduler was handed.
    """
    fill(armed, seeds=list(range(1, 20)))
    manager = Manager()
    request = Req(request_id="req-c", blocks=12, num_tokens=12 * BLOCK + 900)
    made = offer(manager, request, 0, (0, False))
    offered_blocks = made[0] // BLOCK
    required = disk_restore.counters()["last_groups_required"]
    assert required["offered"] is True
    assert required["claimed_blocks"] == offered_blocks
    for row in required["rows"]:
        assert row["blocks_needed"] == offered_blocks
        assert row["restored"] is True and row["blocks_filled"] == offered_blocks


def test_a_boot_with_the_switches_down_never_reaches_the_group_gate(switches_down):
    """A CLOSED BOOT IS BYTE-IDENTICAL: the group arithmetic is not even consulted.

    Since 0.2.31 there is no per-request door, so the gate itself is the first thing
    every lookup consults - and when both boot switches are down it closes BEFORE the
    group serviceability predicate runs. No group arithmetic, no `incomplete_group_set`
    counted, no `group_serviceability` written, no allocation. That ordering is the
    constraint: an operator who has not turned this feature on must not pay for its
    analysis on every request, and a refusal blamed on group shape would send them
    looking at the wrong thing entirely.
    """
    fill(switches_down, seeds=[1, 2, 3, 4], groups=list(HYBRID_CACHEABLE))
    manager = hybrid()
    ram_result = (0, False)
    assert offer(manager, Req(request_id="req-u", blocks=1), 0,
                 ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["refused"]["offers_disabled"] == 1
    assert counters["refused"]["incomplete_group_set"] == 0
    assert counters["group_serviceability"] is None, ("a closed boot must not "
                                                      "even reach the predicate")
    assert counters["last_groups_required"] is None
    assert counters["offers"] == 0


def test_the_refusal_counts_every_lookup_and_logs_once(armed, caplog):
    """Countable per lookup, ONE log line per episode.

    Under disk authority EVERY lookup is refused, so a refusal that logged per request
    would put a sentence in the engine log for each one. The counter is the
    measurement; `notice_once` is the announcement - and the announcement names the
    clamp-is-not-the-answer rule, because the next reader's first instinct is to
    "optimise" the refusal into a clamp.
    """
    fill(armed, seeds=[1, 2, 3, 4], groups=list(HYBRID_CACHEABLE))
    with caplog.at_level(logging.WARNING):
        for index in range(3):
            offer(hybrid(mode=None), Req(request_id=f"req-{index}", blocks=1),
                  0, (0, False))
    counters = disk_restore.counters()
    assert counters["refused"]["incomplete_group_set"] == 3
    assert counters["refusals_total"] == 3
    lines = [record.getMessage()
             for record in caplog.records
             if "cannot be served from disk" in record.getMessage()]
    assert len(lines) == 1, "one announcement per episode, three counters per lookup"
    assert "clamp" in lines[0]


def test_the_refusal_reason_is_defined_and_the_vocabulary_stays_closed():
    """Every refusal name has a WHY, and no old name was renamed or removed.

    `scheduler_section` publishes `refusal_reasons` for the whole `OFFER_REFUSALS`
    tuple and renders an absent entry as `undefined reason`, so the vocabulary and its
    documentation cannot drift apart silently - this test is that drift's alarm.
    """
    assert "incomplete_group_set" in disk_restore.OFFER_REFUSALS
    for name in disk_restore.OFFER_REFUSALS:
        assert disk_restore.OFFER_REFUSAL_WHY.get(name), f"{name} has no WHY entry"
    # The names that were there before 0.2.25 are all still there: the vocabulary grows.
    for name in ("no_disk_run", "offers_disabled", "unaligned_local_prefix",
                 "no_eligible_group", "no_worker_proof", "pin_release_unavailable"):
        assert name in disk_restore.OFFER_REFUSALS
    # The WHY for the new one explains the mechanism, not just the outcome.
    why = disk_restore.OFFER_REFUSAL_WHY["incomplete_group_set"]
    assert "boundary-state block" in why
    assert "SILENT no-op" in why
