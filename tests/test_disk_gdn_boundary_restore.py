r"""THE SECOND BLOCK CONTRACT: an align-mode GDN group served from the SSD tier.

Every other restore suite tests a model whose cacheable groups are all positionally
addressable. This file tests the model the live box actually runs
(`Qwen3.8-Flash-Next-W4A16-Merlin`): four GDN groups whose `MambaManager` answers
`has_positionally_stable_blocks = False` because `mamba_cache_mode == "align"`, one
full-attention group, and a QSA ring that is not prefix-cacheable at all. On that
geometry 0.2.25 refused every disk offer, and the reason it gave - "a group whose
blocks have no positional name has nothing to read back from disk" - is wrong about
the fork. What is NOT wrong is the invariant behind the refusal: an offer must have
bytes for EVERY prefix-cacheable group, because `get_num_new_matched_tokens` skips
prefill model-wide and `update_state_after_alloc` silently walks past a group with no
entry. Serving the GDN groups means proving the right NUMBER OF FILES for them, not
relaxing the gate.

THE SHAPE THIS SUITE PINS, in the order a reviewer should read it:

1. THE CONTRACT IS DERIVED, NEVER NAMED. No model name and no group id appears in the
   predicate; `has_positionally_stable_blocks` plus `mamba_cache_mode` say what a run
   costs the group (`_group_contract`), and an unreadable answer is `unknown`, which is
   a refusal.
2. AN OFFER THAT CLAIMS A RUN HAS A STATE FILE FOR EVERY ALIGN-MODE GROUP AT THE RUN'S
   END, and one KV file per span for every positional group - asserted on the merged
   tuple the fork actually consumes: `[null]*3 + [real]` for a 4-block run, and the
   real block paired with the run's LAST GPU block by the fork's own index arithmetic.
3. A RUN IS SHORTENED TO AN ANCHOR, NEVER TO A LENGTH. If the attention group has four
   blocks and only two of them have a GDN state file, the offer is two - not four, and
   not "four with the GDN groups left stale", which is the 00:08 wrong answer.
4. THE TWO NEW REFUSAL NAMES EXIST AND FIRE. `no_boundary_anchor` and `no_null_block`
   are the conditions that used to be silently survivable.
5. COPY-AT-COMPLETION. Now that a file can be SERVED, the spill must not read the slot
   later: the queue carries the bytes, and a job that carries only a slot number is
   refused. The byte cap that makes that safe is a named drop, not a silent one.
6. THE BYTE FIGURE. The per-block size the diagnostics publish is the RANK's measured
   slot width. The published `block_bytes_by_group` was the layer count multiplied
   into a spec page size that already contained the layer count, and the on-disk files
   are the proof - so the proof is a test, not a comment.

Everything runs against a `tmp_path` store and ctypes-backed fakes: no engine, no
torch, no CUDA, no network, no live box.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time

from kvtier_dev_plugin import (
    disk_read,
    disk_restore,
    disk_spill,
    worker_diag,
)
from tests import fakes_debug
from tests import test_disk_restore as base

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them, autouse ones included (a fresh boot per test, and a /tmp root that is allowed).
armed = base.armed                      # noqa: F811 - the gate open, the switch up
allow_tmp_root = base.allow_tmp_root    # noqa: F811 - autouse
fresh = base.fresh                      # noqa: F811 - autouse
fill = base.fill
offer = base.offer
vanish = base.vanish
BLOCK = base.BLOCK
Req = base.Req
WIDTH = base.WIDTH                      # one rank slot, and one spill file
DIGEST = base.DIGEST

# THE LIVE GEOMETRY (see `test_disk_restore_group_gate.py` for the transcription): a
# 3,200-token scheduler block, GDN groups 0-3 at that size, a QSA ring at 8 tokens, one
# full-attention group at 3,200. Group 4 is deliberately NOT cacheable: it is scratch.
HYBRID_SIZES = (BLOCK, BLOCK, BLOCK, BLOCK, 8, BLOCK)
HYBRID_CACHEABLE = (0, 1, 2, 3, 5)
GDN_GROUPS = (0, 1, 2, 3)
ATTENTION_GROUP = 5


def hybrid(mode="align", **kwargs):
    """The live box's manager: GDN groups answering `stable=False`, `mode="align"`.

    `mode=None` is the shape that must STILL refuse: not positionally stable for some
    reason that is not an align hand-off, so no name exists to read back. The default
    is the live answer, so a test that forgets the question is a test about the box.
    """
    manager = base.Manager(group_sizes=HYBRID_SIZES, cacheable=HYBRID_CACHEABLE,
                           **kwargs)
    items = []
    for group in range(len(HYBRID_SIZES)):
        if group in GDN_GROUPS:
            # A GDN group answers `has_positionally_stable_blocks = False` EITHER
            # way; what differs is whether it can say why. `mode="align"` says so
            # (the fork derives the flag from it) and buys the boundary-state
            # contract; `mode=None` leaves the flag unexplained, which is the shape
            # with no contract.
            spec = (fakes_debug.mamba_spec(block_size=HYBRID_SIZES[group],
                                           mamba_cache_mode=mode)
                    if mode is not None else
                    fakes_debug.attention_spec(block_size=HYBRID_SIZES[group]))
            items.append(fakes_debug.manager_item(group, spec, stable=False,
                                                  mamba_cache_mode=mode))
        else:
            items.append(fakes_debug.manager_item(
                group,
                fakes_debug.attention_spec(block_size=HYBRID_SIZES[group]),
                stable=True))
    manager.cpu_coordinator = fakes_debug.coordinator(items)
    return manager


def per_group_ids(manager, request_id="req-1"):
    """The merged tuple's CPU block ids, per group, in the fork's group order."""
    return [[int(block.block_id) for block in entry]
            for entry in manager._pending_cpu_hits[request_id][0]]


def pair_with_gpu_blocks(cpu_list, gpu_ids, start=0):
    """`update_state_after_alloc`'s pairing loop, verbatim in structure.

    `manager.py:460-478`: walk the group's CPU list by INDEX, skip the entries whose
    `is_null` is set, and hand `group_gpu_ids[start + i]` to the one that is not null.
    The index is NOT compacted when a null is skipped - that is the whole reason a
    boundary-state restore works, and the reason a list built by appending the real
    block first would land the state on the run's FIRST block and serve the request a
    recurrent state from the wrong place.
    """
    pairs = []
    for index, cpu_block in enumerate(cpu_list):
        if getattr(cpu_block, "is_null", False):
            continue
        pairs.append((int(cpu_block.block_id), gpu_ids[start + index]))
    return pairs


# --- 1. the contract is derived from the manager, not from a model name -------

def test_the_live_geometry_derives_two_contracts_and_names_no_model():
    """Groups 0-3 are `boundary_state`, group 5 is `positional`, and the code knows it
    from two attributes.

    This is the test that fails the loudest if the contract is ever reimplemented from
    a name: nothing here passes a model id, a layer count or a group index to the
    predicate, and the GDN answer comes from the same two fields the fork's own
    `has_positionally_stable_blocks` override is computed from.
    """
    manager = hybrid()
    contracts = {group: disk_restore._group_contract(manager, group)
                 for group in range(len(HYBRID_SIZES))}
    assert [contracts[group]["contract"] for group in GDN_GROUPS] == \
        [disk_restore.CONTRACT_BOUNDARY] * 4
    assert contracts[ATTENTION_GROUP]["contract"] == \
        disk_restore.CONTRACT_POSITIONAL
    # The non-cacheable ring is not the gate's problem, but its contract still has to
    # be answerable: "stable" is a per-group fact, not a per-model one.
    assert contracts[4]["contract"] == disk_restore.CONTRACT_POSITIONAL
    # The WHY is a sentence about the engine's shape, with the citation, because a
    # reader who disagrees with the gate should be able to check it against the fork.
    assert "1591-1607" in contracts[0]["reason"]
    assert contracts[0]["mamba_cache_mode"] == "align"
    # And the same fixture WITHOUT the align mode is the refusal shape, not a smaller
    # version of the same contract.
    stuck = hybrid(mode=None)
    assert disk_restore._group_contract(stuck, 0)["contract"] == \
        disk_restore.CONTRACT_UNSTABLE


def test_an_unreadable_predicate_answers_unknown_which_is_not_restorable():
    """A fork that renamed the flag gets `unknown`, and `unknown` is a refusal.

    The failure this guards is the quiet one: `getattr(entry, "...", None)` returning
    None and a predicate that reads None as False would turn every group on the box
    into an align-mode group the moment the fork renames a field - a model whose shape
    we do not know, offered as if we did.
    """
    manager = hybrid()
    del manager.cpu_coordinator.single_type_managers[0] \
        .has_positionally_stable_blocks
    verdict = disk_restore._group_contract(manager, 0)
    assert verdict["contract"] == disk_restore.CONTRACT_UNKNOWN
    assert verdict["stable"] is None
    blocked = disk_restore._eligible_groups(manager, BLOCK)
    assert blocked["ok"] is False
    assert blocked["reason"]
    # The reason is the OLD name, and the block shape is the NEW field: a reader of the
    # 0.2.25 payloads keeps their vocabulary and gains the diagnosis.
    assert blocked["reason_kind"] == "incomplete_group_set"
    assert blocked["blocked_kinds"]["0"] == disk_restore.CONTRACT_UNKNOWN
    assert blocked["blocked_group_ids"] == [0]


# --- 2. an offer on the real geometry: bytes for every cacheable group --------

def test_an_align_hybrid_run_offers_and_every_cacheable_group_gets_a_load(armed):
    """Four blocks on the live geometry: the offer stands, and all five groups appear.

    `loads` is the field an operator reads to answer "did the GDN groups get anything",
    so it is asserted here rather than inferred from a token count: group 5 has four
    real blocks and no nulls, every GDN group has ONE real block and three nulls, and
    no cacheable group has zero real blocks. A row with `real_blocks: 0` for an
    eligible group IS the 00:08 bug, in one field.
    """
    fill(armed, blocks=4, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    answer = offer(manager, Req(blocks=4), 0, (0, False))
    assert answer == (4 * BLOCK, True), disk_restore._state.get("last_reason")
    offer_witness = disk_restore._state["last_offer"]
    assert offer_witness["positional_groups"] == [ATTENTION_GROUP]
    assert offer_witness["boundary_groups"] == list(GDN_GROUPS)
    assert offer_witness["blocks"] == 4
    loads = {row["group"]: row for row in offer_witness["loads"]}
    assert loads[ATTENTION_GROUP]["real_blocks"] == 4
    assert loads[ATTENTION_GROUP]["null_blocks"] == 0
    for group in GDN_GROUPS:
        assert loads[group]["contract"] == disk_restore.CONTRACT_BOUNDARY
        assert loads[group]["real_blocks"] == 1
        assert loads[group]["null_blocks"] == 3
    # The claimed span costs 4 attention blocks plus ONE state block per GDN group:
    # 8 CPU slots, not 20. Pricing a boundary group per span would refuse offers this
    # tier can pay for; the counter is the arithmetic made checkable.
    assert disk_restore._state["slots_allocated"] == 8


def test_every_named_block_has_exactly_one_allocated_slot(armed):
    """THE INVARIANT, published as a number: named blocks == allocated slots.

    This is the check that caught a real defect in this change while it was being
    written. The first `_cut_run` rebuilt a shortened span from the run's already-merged
    `keys` and appended its own anchor to it, so a 4-block run on this geometry named
    12 blocks and allocated 8 slots: the boundary-state pair for the run's end was
    counted twice. `_merge_into_pending`'s allocation arithmetic refused the offer
    (`8 slots were allocated for 12 named block(s)`) rather than serving it - failing
    safe, which is the only reason this was a missing feature instead of a corruption.
    A pair-per-span count is the smallest statement of "we name what we allocate", so it
    is asserted at every length this suite offers.
    """
    for blocks in (1, 2, 4):
        # Files accumulate in this test's own root and the journal with them; the run
        # still stops at the request's own hash view, so `blocks` is the offer's length.
        fill(armed, blocks=blocks, groups=HYBRID_CACHEABLE)
        manager = hybrid()
        answer = offer(manager, Req(request_id=f"req-{blocks}", blocks=blocks), 0,
                       (0, False))
        assert answer == (blocks * BLOCK, True), disk_restore._state["last_reason"]
        identities = disk_restore._loads[f"req-{blocks}"]["identities"]
        assert len(identities) == blocks + len(GDN_GROUPS)
        assert len(identities) == len(manager.cpu_block_pool.allocated)
        # Every (group, span) pair appears once. A duplicate would be two readers of
        # one slot: the second load overwrites the first with a different file.
        pairs = [(entry["group"], entry["span"]) for entry in identities]
        assert len(pairs) == len(set(pairs))
        spans_for_gdn = [entry["span"] for entry in identities
                         if entry["group"] == 0]
        assert spans_for_gdn == [blocks - 1], "one state block, on the run's end"


def test_the_align_group_list_is_nulls_up_to_the_run_s_last_block(armed):
    """`[null, null, null, real]`, and the nulls are the POOL'S null block.

    Not `None`, not a fresh slot: the fork's `update_state_after_alloc` filters on
    `is_null` and its `_free_pending_cpu_hit` frees only the non-null entries, so a
    placeholder that is not the pool's null block would either be skipped wrongly or
    freed as a real block. Both are refcount bugs on the box, and the second one leaks
    a CPU block per flagged request per boot.
    """
    fill(armed, blocks=4, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    offer(manager, Req(blocks=4), 0, (0, False))
    ids = per_group_ids(manager)
    null_id = manager.cpu_block_pool.null_block.block_id
    for group in GDN_GROUPS:
        assert len(ids[group]) == 4
        assert ids[group][:3] == [null_id] * 3, \
            "the interior entries are the pool's null block, three of them"
        assert ids[group][3] != null_id
        assert ids[group][3] not in ids[group][:3]
    # The scratch ring and a group the model does not cache get nothing at all: the
    # per-group list count is still `num_groups`, or the fork's `range(num_groups)`
    # loop would IndexError on the first tuple it pops.
    assert ids[4] == []
    assert len(ids) == len(HYBRID_SIZES)
    # The pool handed out exactly 8 blocks and none of them is the null block.
    assert len(manager.cpu_block_pool.allocated) == 8
    assert manager.cpu_block_pool.null_block not in \
        manager.cpu_block_pool.allocated


def test_no_path_is_ever_emitted_for_a_null_block(armed):
    """The pool's null block is a placeholder, not a destination: no file may be read
    into it.

    `load_cpu_blocks`/`load_paths` are built from the identities, and an identity exists
    only for a block that was ALLOCATED. If a null ever acquired one, the worker would
    read a KV file into slot `null_block.block_id` - a slot every group's list points at
    at once, and the block the fork's own free path filters on. Cheap to assert, and it
    is the difference between a wrong answer and a corruption that outlives the request.
    """
    fill(armed, blocks=4, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    offer(manager, Req(blocks=4), 0, (0, False))
    null_id = manager.cpu_block_pool.null_block.block_id
    identities = disk_restore._loads["req-1"]["identities"]
    assert all(int(entry["cpu_block_id"]) != null_id for entry in identities)
    assert len(identities) == len(set(int(entry["cpu_block_id"])
                                      for entry in identities))
    # The emission is keyed by CPU slot id, so hand it a step whose load list is the
    # real slots PLUS the null block's id, and only the real ones may be attributed a
    # file. This is exactly what the fork's `load_cpu_blocks` is: the entries of the
    # merged tuple, index-aligned with the GPU half.
    null_id = manager.cpu_block_pool.null_block.block_id
    cpu = [int(entry["cpu_block_id"]) for entry in identities] + [null_id]
    step = base.Meta(load_event=7, cpu=cpu, gpu=[500 + i for i in range(len(cpu))],
                     reqs={7: ["req-1"]})
    made = disk_restore._build_load_paths(step)
    assert made["ok"] is True, made
    paths = step.load_paths
    assert named_count(paths) == len(identities)
    assert paths[-1] == "", "the null block was attributed a file it must never read"
    assert all(path for path in paths[:-1])


def named_count(paths) -> int:
    return len([path for path in paths if path])


def test_the_state_block_pairs_with_the_run_s_last_gpu_block(armed):
    """THE TEST THE WHOLE FEATURE HINGES ON.

    With the fork's own pairing loop, group 0's single real block must be paired with
    the run's LAST GPU block. Build the list the way a naive implementation would -
    real block first, nulls after - and the same pairing hands the recurrent state to
    the FIRST block, which is a wrong KV state that the model then uses to answer the
    prompt at `temperature=0`. So this asserts the pairing, not the list: the list is
    only interesting because of what the fork does with it.
    """
    fill(armed, blocks=4, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    offer(manager, Req(blocks=4), 0, (0, False))
    group_zero = manager._pending_cpu_hits["req-1"][0][0]
    gpu_ids = [900, 901, 902, 903]           # the run's four GPU blocks
    paired = pair_with_gpu_blocks(group_zero, gpu_ids)
    assert len(paired) == 1
    assert paired[0][1] == 903, "the boundary state must land on the run's end"
    # The positional group is unaffected: four pairs, in order, on all four blocks.
    paired_attention = pair_with_gpu_blocks(
        manager._pending_cpu_hits["req-1"][0][ATTENTION_GROUP], gpu_ids)
    assert [gpu for _cpu, gpu in paired_attention] == [900, 901, 902, 903]


def test_the_claim_is_whole_blocks_and_no_request_level_skip_is_invented(armed):
    """The offer raises the fork's token count in whole spans and changes nothing else.

    `get_num_new_matched_tokens` returns `(tokens, is_async)`; the second field is
    still True for a disk offer (the load is real and asynchronous) and the request's
    own `skip_reading_prefix_cache` is untouched - a plugin that set it would turn a
    KV-tier bug into an engine-wide one.
    """
    fill(armed, blocks=4, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    request = Req(blocks=4, num_tokens=4 * BLOCK + 1200)
    answer = offer(manager, request, 0, (0, False))
    tokens, is_async = answer
    assert tokens % BLOCK == 0
    assert tokens == 4 * BLOCK
    assert is_async is True
    assert request.skip_reading_prefix_cache is False
    # The partial tail is not claimed: 1200 tokens past the last whole block stay
    # prefill work, exactly as the fork leaves them.
    assert request.num_tokens - tokens == 1200


# --- 3. the run is anchored: the attention group does not get to run alone ----

def test_a_run_is_shortened_to_the_last_block_every_boundary_group_can_name(armed):
    """Group 5 has blocks 1-4; the GDN state files exist only at 1 and 2. THE OFFER IS
    TWO.

    This is the case the incident report asked for by name. Four is available for the
    attention group and would be the bigger number, the faster answer and a stale
    recurrent state at the end of it; two is the longest span whose end every
    align-mode group can actually name. The witness says why in one field:
    `positional_run_blocks` is the claim the attention group could have made alone.
    """
    fill(armed, blocks=2, groups=GDN_GROUPS)
    fill(armed, blocks=4, groups=(ATTENTION_GROUP,))
    manager = hybrid()
    answer = offer(manager, Req(blocks=4), 0, (0, False))
    assert answer == (2 * BLOCK, True)
    witness = disk_restore._state["last_offer"]
    assert witness["blocks"] == 2
    assert witness["positional_run_blocks"] == 4
    assert witness["run_reanchored"] is True
    ids = per_group_ids(manager)
    for group in GDN_GROUPS:
        assert len(ids[group]) == 2 and ids[group][1] != ids[group][0]
    assert len(ids[ATTENTION_GROUP]) == 2
    # And the state file that was named is the one for the run's END (block 2), not for
    # block 1 and not for the block 4 the attention group could have reached.
    identities = disk_restore._loads["req-1"]["identities"]
    paths = [entry["path"] for entry in identities if entry["group"] == 0]
    assert len(paths) == 1
    assert base.key_hex(2, 0)[:16] in paths[0]
    assert base.key_hex(4, 0)[:16] not in paths[0]


def test_a_state_file_missing_at_the_only_available_run_end_refuses(armed):
    """`no_boundary_anchor`: the positional groups have a run, and NO end of it is one
    every align-mode group can pay for.

    Not a shorter offer: a shorter offer would need a state file too, and there is
    none. Recomputing is the correct answer here, and the fork's own value must come
    back unconsumed - the refusal has to be indistinguishable from no plugin.
    """
    fill(armed, blocks=4, groups=(ATTENTION_GROUP,))
    manager = hybrid()
    fork_answer = (0, False)
    assert offer(manager, Req(blocks=4), 0, fork_answer) == fork_answer
    counters = disk_restore.counters()
    assert counters["refused"]["no_boundary_anchor"] == 1
    assert manager._pending_cpu_hits == {}
    assert manager.cpu_block_pool.allocated == []


def test_a_clamped_offer_cuts_to_an_anchor_and_never_between_two(armed,
                                                                 monkeypatch):
    """A cap that shortens the offer below the anchored end must move the END, not
    truncate the tail.

    Anchors at 2 and 4; the cap allows 3. A naive `keys[:3]` would claim three spans
    and put the state block on span 3, for which no group has a state file - the
    incident in miniature, created by a capacity clamp. The cut lands on 2.
    """
    fill(armed, blocks=2, groups=GDN_GROUPS)
    fill(armed, blocks=1, groups=GDN_GROUPS, seeds=[4])
    fill(armed, blocks=4, groups=(ATTENTION_GROUP,))
    monkeypatch.setattr(disk_restore, "MAX_SLOTS_PER_OFFER", 6)
    manager = hybrid()
    answer = offer(manager, Req(blocks=4), 0, (0, False))
    assert answer == (2 * BLOCK, True), "the cut must land on an anchored length"
    witness = disk_restore._state["last_offer"]
    assert witness["blocks"] == 2
    # 2 attention blocks + 4 state blocks = 6, exactly the cap: the clamp did the
    # arithmetic and then asked the run for a boundary it can still afford.
    assert disk_restore._state["slots_allocated"] == 6


def test_a_pool_without_a_null_block_refuses_rather_than_aiming_the_state(armed):
    """`no_null_block`: without the pool's placeholder the interior entries cannot be
    built, and the one state block would land on the run's FIRST GPU block.

    The refusal is the whole point - the alternative is a plausible restore whose KV
    is off by a whole run. And the refusal happens BEFORE any allocation, so nothing
    has to be undone: the pool is untouched and the fork's answer stands.
    """
    fill(armed, blocks=4, groups=HYBRID_CACHEABLE)
    pool = base.CpuPool()
    del pool.null_block
    manager = hybrid(pool=pool)
    fork_answer = (0, False)
    assert offer(manager, Req(blocks=4), 0, fork_answer) == fork_answer
    assert disk_restore.counters()["refused"]["no_null_block"] == 1
    assert pool.allocated == []
    assert manager._pending_cpu_hits == {}


def test_the_two_new_refusals_are_documented_and_in_the_vocabulary():
    """A refusal name with no WHY entry renders as `undefined reason` on the dashboard.

    `scheduler_section` publishes `refusal_reasons` for the whole `OFFER_REFUSALS`
    tuple, so this is the alarm that keeps the new names from becoming the kind of
    counter an operator has to reverse-engineer from the source.
    """
    for name in ("no_boundary_anchor", "no_null_block"):
        assert name in disk_restore.OFFER_REFUSALS
        assert disk_restore.OFFER_REFUSAL_WHY.get(name), f"{name} has no WHY entry"
    assert "state" in disk_restore.OFFER_REFUSAL_WHY["no_boundary_anchor"].lower()
    assert "null" in disk_restore.OFFER_REFUSAL_WHY["no_null_block"].lower()


# --- 4. the cost of a claim, published where a reader can check it ------------

def test_the_required_block_arithmetic_is_one_state_block_per_run(armed):
    """`_groups_required` at 4 blocks: 4 for the attention group, ONE per GDN group.

    The 0.2.25 row for these groups said "not restorable, 0 filled" and the 00:08
    payload said "g5 filled, nothing else was". Now the row says what the engine's own
    hit shape costs - `blocks_needed: 1, null_blocks: 3` - and `blocks_filled` counts
    it, so a reader comparing the two totals can see that the run is paid for.
    """
    fill(armed, blocks=4, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    offer(manager, Req(blocks=4), 0, (0, False))
    required = disk_restore._state["last_groups_required"]
    rows = {row["group"]: row for row in required["rows"]}
    assert rows[ATTENTION_GROUP]["blocks_needed"] == 4
    assert rows[ATTENTION_GROUP]["null_blocks"] == 0
    for group in GDN_GROUPS:
        assert rows[group]["contract"] == disk_restore.CONTRACT_BOUNDARY
        assert rows[group]["blocks_needed"] == 1
        assert rows[group]["null_blocks"] == 3
        assert rows[group]["real_blocks"] == 1
        assert rows[group]["restored"] is True
    assert required["unfilled_group_ids"] == []
    assert required["blocks_needed_total"] == 8
    assert required["blocks_filled_total"] == 8


def test_the_span_cost_of_each_contract_is_published_in_bytes(armed):
    """The writer states the cost model, in bytes, per contract.

    Both contracts cost one file of the same width; what differs is how many files a
    span buys. A single "bytes per span" field would be a lie about one of them, and a
    reader who counts spilled files concludes a GDN group is cheap to keep - true per
    hit, false as a capacity plan, because the spans it did not buy are spans it did
    not keep. So the two numbers are published side by side, with that sentence, and
    both are `null` until a fingerprint exists: an unknown width is not a zero width.
    """
    blind = disk_spill.section()["span_cost_bytes"]
    assert blind["positional_bytes_per_span"] is None
    assert blind["boundary_state_bytes_per_span"] is None
    assert "why_absent" in blind["meaning"]

    adopted = disk_spill.configure(base_fingerprint())
    assert adopted["configured"] is True, adopted
    cost = disk_spill.section()["span_cost_bytes"]
    assert cost["positional_bytes_per_span"] == WIDTH
    assert cost["boundary_state_bytes_per_span"] == WIDTH
    assert cost["units"]["positional_bytes_per_span"] == "bytes"
    # The sentence is the point: the two numbers being equal does NOT mean the two
    # contracts cost the same per span, and the field says so in the payload.
    assert "n-1 nulls" in cost["meaning"]["boundary_state_bytes_per_span"]
    assert "EVERY block of a run" in cost["meaning"]["positional_bytes_per_span"]


# --- 5. copy-at-completion: the queue carries BYTES, never a slot number ------

class Region:
    """A region tensor the RAM tier can RECYCLE: real memory, and writable mid-test.

    `read_slot_bytes` copies out of `data_ptr()`, so mutating a slot after a spill has
    been requested is exactly what a real store into a recycled CPU block does. That is
    the race, and this class is the only reason the regression test can exist without an
    engine.
    """

    def __init__(self, blocks, bytes_per_block):
        self.shape = (blocks, bytes_per_block)
        self._bytes_per_block = bytes_per_block
        self._buffer = ctypes.create_string_buffer(blocks * bytes_per_block)
        self.nbytes = blocks * bytes_per_block

    def data_ptr(self):
        return ctypes.addressof(self._buffer)

    def stride(self, axis):
        return self._bytes_per_block if axis == 0 else 1

    def element_size(self):
        return 1

    def fill_slot(self, slot, payload):
        assert len(payload) == self._bytes_per_block
        ctypes.memmove(ctypes.addressof(self._buffer) + slot *
                       self._bytes_per_block, payload, len(payload))

    def slot(self, slot):
        return ctypes.string_at(self.data_ptr() + slot * self._bytes_per_block,
                                self._bytes_per_block)


def base_fingerprint(**overrides) -> dict:
    """A fingerprint for THIS store: one rank, and a slot exactly `WIDTH` bytes wide."""
    made = {
        "model_id": "Qwen3.8-Flash-Next-W4A16-Merlin",
        "model_revision": "8f3c1d2e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d",
        "cache_dtype": "auto",
        "dtype": "torch.bfloat16",
        "total_bytes_per_block": WIDTH,
        "group_block_sizes": [1, 1, 1, 1, 1, 1],   # one entry per group
        "scheduler_block_size": BLOCK,
        "hash_block_size": base.HASH_BLOCK,
        "prefix_cacheable_group_ids": list(HYBRID_CACHEABLE),
        "prefix_caching_hash_algo": "sha256_cbor",
        "none_hash_seed": "0xd4e5f6a7b8c9d0e1",
        "rank": "PP0_TP0",
        "world_size": 4,
        "dcp_world_size": 1,
        "cp_world_size": 1,
    }
    made.update(overrides)
    return made


def spill_worker(region):
    return type("FakeWorker", (), {"cpu_kv_caches": {
        "language_model.model.layers.0.linear_attn": region}})()


def wait_for(path, limit=400):
    for _ in range(limit):
        if os.path.exists(path):
            return True
        time.sleep(0.005)
    return False


def test_the_bytes_that_land_are_the_bytes_the_store_completed_with(armed):
    """THE RACE, CLOSED, AND SHOWN CLOSED.

    A store completes into CPU slot 3 and the spill is requested. Before the writer
    thread gets the job, the RAM tier recycles slot 3 for the NEXT block - the ordinary
    thing a block pool does. The file must contain the FIRST block's bytes.

    Before this change the job carried `slot=3` and read the region on the writer
    thread, so it wrote the SECOND block's bytes under the FIRST block's name. While
    nothing could read a spilled file that was a wasted write; a file that can be
    SERVED makes it a wrong answer. The assertion is on the bytes, and the digest
    assertion next to it is the same fact from the file name's point of view: the name
    commits to a content, so the content had better be the one that was stored.
    """
    fingerprint = base_fingerprint()
    region = Region(8, WIDTH)
    first = bytes((index * 13 + 5) % 251 for index in range(WIDTH))
    second = bytes(0xEE for _ in range(WIDTH))
    region.fill_slot(3, first)
    worker = spill_worker(region)
    key = base.key_hex(1, 0)
    named = disk_spill.plan_slot_naming(key, fingerprint)
    assert named["ok"], named["reason"]
    assert named["bytes_per_file"] == WIDTH

    asked = disk_spill.request_spill(worker=worker, slot=3, key=key,
                                     fingerprint=fingerprint)
    assert asked["ok"] is True and asked["queued"] is True
    # The copy happened on the CALLER's thread, before this call returned: the answer
    # already knows how many bytes it took.
    assert asked["bytes"] == WIDTH
    assert int(disk_spill._state["queue_bytes"]) == WIDTH

    region.fill_slot(3, second)          # the RAM tier reuses the slot

    assert wait_for(named["path"]), "the writer thread never landed the file"
    with open(named["path"], "rb") as handle:
        landed = handle.read()
    assert landed == first, "the recycled slot's contents were written under the " \
                            "completed block's name"
    assert hashlib.sha256(landed).hexdigest() != hashlib.sha256(second).hexdigest()
    for _ in range(400):
        if int(disk_spill._state["queue_bytes"]) == 0:
            break
        time.sleep(0.005)
    assert int(disk_spill._state["queue_bytes"]) == 0, "the snapshot never left the " \
        "queue's byte accounting: the cap would fill and the tier would stop spilling"


def test_deferring_the_read_writes_the_next_block_s_bytes(armed):
    """The same mutation, the OLD code path, and a file that is now WRONG on purpose.

    `spill_slot(caches=..., slot=...)` still reads on the calling thread, because a
    synchronous caller does own the slot. This test keeps that door honest: it proves
    the race is real (the deferred read produces the recycled bytes) rather than
    theoretical, and it proves the snapshot path and the direct path are not the same
    promise. If someone ever routes the queue back through `caches`, this test is the
    one that says what they shipped.
    """
    fingerprint = base_fingerprint()
    region = Region(8, WIDTH)
    region.fill_slot(3, bytes((i * 13 + 5) % 251 for i in range(WIDTH)))
    worker = spill_worker(region)
    key = base.key_hex(2, 1)
    named = disk_spill.plan_slot_naming(key, fingerprint)
    assert named["ok"], named["reason"]
    region.fill_slot(3, bytes(0x77 for _ in range(WIDTH)))
    wrote = disk_spill.spill_slot(caches=worker.cpu_kv_caches, slot=3, key=key,
                                  fingerprint=fingerprint)
    assert wrote["ok"] is True, wrote
    assert wrote["ok"] is True
    with open(named["path"], "rb") as handle:
        assert handle.read() == bytes(0x77 for _ in range(WIDTH))


def test_a_queued_job_without_a_snapshot_is_refused_not_read(armed):
    """A job that carries only a slot number is a bug in its caller, and is counted.

    The `caches` fallback is gone from `_handle_job` deliberately. Leaving it in would
    keep the write-after-completion race one well-meaning refactor away, with no test
    failing - so the fallback's absence is itself asserted here.
    """
    fingerprint = base_fingerprint()
    region = Region(8, WIDTH)
    region.fill_slot(1, bytes((i * 7) % 251 for i in range(WIDTH)))
    key = base.key_hex(3, 2)
    named = disk_spill.plan_slot_naming(key, fingerprint)
    assert named["ok"], named["reason"]
    before = int(disk_spill._state["refused"]["bad_call"])
    disk_spill._handle_job({"slot": 1, "key": key, "fingerprint": fingerprint,
                            "caches": region})
    assert int(disk_spill._state["refused"]["bad_call"]) == before + 1
    assert "snapshot" in str(disk_spill._state["last_reason"])
    assert not os.path.exists(named["path"])


def test_a_snapshot_that_would_break_the_byte_cap_is_a_named_drop(armed,
                                                                  monkeypatch):
    """The queue is bounded in TWO currencies, and the second one has its own name.

    Copy-at-completion moved the queue's cost from a slot number to a whole slot of
    bytes (11,468,800 B on the live PP0), so `MAX_QUEUE_DEPTH` alone would allow
    gigabytes of held host memory exactly when the box is already behind. A refusal at
    the cap is a DROP with a counter - and the counter must not be `queue_full`,
    because "the queue is 256 deep" and "we would have had to hold another slot" are
    different diagnoses.
    """
    fingerprint = base_fingerprint()
    monkeypatch.setattr(disk_spill, "MAX_QUEUED_BYTES", WIDTH)
    region = Region(8, WIDTH)
    region.fill_slot(0, bytes(WIDTH))
    worker = spill_worker(region)
    first = disk_spill.request_spill(worker=worker, slot=0,
                                     key=base.key_hex(5, 0),
                                     fingerprint=fingerprint)
    assert first["ok"] is True, first
    region.fill_slot(1, bytes(WIDTH))
    second = disk_spill.request_spill(worker=worker, slot=1,
                                      key=base.key_hex(6, 0),
                                      fingerprint=fingerprint)
    assert second["ok"] is False
    assert second["reason_kind"] == "queue_bytes_full"
    assert disk_spill.counters()["refused"]["queue_bytes_full"] == 1
    # The unit table must not let a COUNT of drops be read as a VOLUME of memory.
    assert disk_spill.section()["refused"]["units"]["queue_bytes_full"] == \
        "refusals"
    assert disk_spill.section()["caps"]["max_queued_bytes"] == WIDTH


def test_a_failing_write_still_gives_its_bytes_back(armed, monkeypatch):
    """The byte accounting lives in a `finally`, so a failed write cannot leak the cap.

    A counter that only ever grows is the silent-failure shape this project has been
    bitten by before (a latch that outlived the attempt that set it cost 25,945
    watchdog cycles on the sibling box). Here the symptom would be a spill tier that
    stops spilling forever, with every counter reading "fine" except one.
    """
    fingerprint = base_fingerprint()
    region = Region(8, WIDTH)
    region.fill_slot(2, bytes(WIDTH))
    worker = spill_worker(region)
    monkeypatch.setattr(disk_spill.disk_layout, "write_committed_file",
                        lambda *a, **k: {"ok": False,
                                         "reason": "a filesystem that said no",
                                         "reason_kind": "write_failed",
                                         "final_exists": False})
    asked = disk_spill.request_spill(worker=worker, slot=2,
                                     key=base.key_hex(7, 3),
                                     fingerprint=fingerprint)
    assert asked["ok"] is True, asked
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if disk_spill.counters()["refused"]["write_failed"] >= 1:
            break
        time.sleep(0.005)
    assert disk_spill.counters()["refused"]["write_failed"] >= 1
    assert int(disk_spill._state["queue_bytes"]) == 0


# --- 6. task 0: the byte figure, settled by the bytes on disk -----------------

def test_a_spill_file_is_the_measured_slot_width_not_layers_times_page_size(armed):
    """The published `block_bytes_by_group` was 7x the file it describes.

    THE MEASUREMENT THAT DECIDED THE ARGUMENT, kept as an executable. The live
    diagnostics said group 0 was 80,281,600 B per block (7 layers x a 11,468,800 B
    page size) and the same rank's slot is 11,468,800 B - so the published figure is
    not a per-block size at all, it is the whole slot counted once per layer, because
    `MambaSpec.merge()` already summed `page_size_bytes` over the merged layers. Every
    spill file on the box is 11,468,800 B, and `MAX_BYTES_PER_READ` (64 MiB) was never
    the limit: the unit is under 13 MB.

    So this test asserts the file is the width the fingerprint describes, and that the
    OLD expression is strictly larger than the file it claimed to describe. A figure
    bigger than the slot cannot be a slot's price.
    """
    layers = 7
    fingerprint = base_fingerprint()
    region = Region(8, WIDTH)
    region.fill_slot(0, bytes((i * 3) % 251 for i in range(WIDTH)))
    wrote = disk_spill.spill_slot(caches=spill_worker(region).cpu_kv_caches, slot=0,
                                  key=base.key_hex(8, 0), fingerprint=fingerprint)
    assert wrote["ok"] is True, wrote
    size = os.path.getsize(wrote["path"])
    assert size == WIDTH == fingerprint["total_bytes_per_block"]
    assert size * layers > size          # the figure the publisher used to print
    assert size < int(disk_read.MAX_BYTES_PER_READ), \
        "a slot must fit inside one read: the read cap is not what limits a restore"


def test_the_publisher_stops_multiplying_the_layer_count_in(armed):
    """`block_bytes_by_group` is the spec's page size ALONE, with the layers separate.

    The field is fixed at the publisher, not documented around: a capacity number that
    is 7x too big is worse than no number, and the reader that wanted a product can
    take it from `layer_count_by_group` with its eyes open. `arena.bytes_per_block` is
    the measured cross-check that made the error visible in the first place.
    """
    spec = fakes_debug.mamba_spec(block_size=BLOCK, page_size_bytes=WIDTH)
    handler = fakes_debug.Obj(
        name="SimpleCPUOffloadWorker",
        kv_cache_config=fakes_debug.kv_cache_config([
            fakes_debug.group(spec, [f"model.layers.{i}.linear_attn"
                                     for i in range(7)])]))
    sizes, layers, note = worker_diag._block_bytes_by_group(handler)
    assert note is None
    assert sizes == [WIDTH], "the spec's number, undoubled"
    assert layers == [7]
    assert WIDTH * 7 not in sizes
