r"""Disk AUTHORITY: the disk decides every request's prefix - and nothing else does.

WHAT CHANGED IN 0.2.22, AND WHY IT IS A SEMANTIC CHANGE AND NOT A KNOB. Until then the
restore path filled the gap the host-RAM tier left: consult RAM, offer the disk's
continuation after it. That made the read path impossible to exercise. A real prompt
either hit RAM completely (nothing left to offer), or hit it PARTIALLY, which the
alignment rule refused as `ram_hit_not_block_aligned` - so proving the disk could
serve a single block meant making GPU and RAM forget the prefix first, ~1.3M tokens of
filler per attempt and seven failed runs. The rule is one switch with no new field:

  * `disk_authority` up -> the disk is the AUTHORITY. Each request's host-RAM hit is
    treated as zero tokens for the offer arithmetic and released, and the disk index
    decides the span from the local GPU-computed boundary.
  * only `offers_enabled` up -> byte-identical to the old behaviour. The RAM tier
    answers, the disk fills the gap.

WHAT CHANGED IN 0.2.31, AND WHY THIS FILE'S FIXTURES CHANGED NAME. Until then the rule
above was carried by a client-supplied marker
(`kv_transfer_params.kvtier_disk_restore = true`) and a `scope` argument chose whether
that marker was required. A request body must not decide whether its KV comes off
NVMe, so the marker and the scope are gone and the switch decides for every request or
for none. The marker is dead in the strongest sense a test can prove: `disk_restore`
never reads `kv_transfer_params` at all, so a request that still carries one changes
nothing anywhere in this suite.

The things these tests exist to pin, in the order they would hurt:

1. NO DOUBLE SERVE. The offer claims the disk span and NOTHING else: the returned
   count is the disk's, the merged tuple contains only our own slots, and the fork's
   pinned CPU blocks are released exactly once. The token span is never claimed twice
   and never left pinned.
2. `computed` IS NOT FORKED. It is the offset the fork's own DMA writes at
   (`manager.py:441-463` places entry `i` at `num_computed_tokens // group_block_size
   + i` into the request's whole block list). Zeroing it would aim disk bytes at the
   request's already-cached, ref-counted prefix blocks - blocks OTHER requests are
   reading - and leave the newly allocated region unwritten. So the switch overrides
   RAM, and never the local GPU prefix, and a test says so in the tuple itself.
3. EVERY REFUSAL IS SILENT AND CHEAP. A request the disk cannot serve leaves
   the fork's answer, the fork's pin and the CPU pool exactly as they were. No error
   reaches the client, nothing is allocated, nothing stalls.
4. THE SWITCH IS THE ONLY DOOR, and it is a JSON boolean set by the route op and nothing
   else. A request body is not a door, a `"true"` string is not a door, and raising the
   cheaper gap-fill switch is not a door to this one.
"""

from __future__ import annotations

import logging

import pytest
import test_disk_restore as base

from kvtier_dev_plugin import disk_restore

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them (pytest collects fixtures from imported names).
armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
armed_gap_fill = base.armed_gap_fill  # noqa: F811 - RAM first, the disk fills the gap
switches_down = base.switches_down  # noqa: F811 - autouse `armed` minus both switches
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

BLOCK = base.BLOCK
GROUP_SIZES = base.GROUP_SIZES
CACHEABLE = base.CACHEABLE
Manager = base.Manager
Req = base.Req
Slot = base.Slot
fill = base.fill
offer = base.offer
merged_cpu = base.merged_cpu


def ram_hit(manager, blocks, computed=0, request_id="req-1"):
    """Give the manager the fork's answer for a RAM hit, exactly as the fork would.

    `get_num_new_matched_tokens` pins the blocks it found (a `touch()` on the CPU pool)
    AND leaves the tuple in `_pending_cpu_hits`. Both halves are reproduced here,
    because the pin half is the one an ignoring offer has to undo.
    """
    lists = [[Slot(900 + group * 10 + i) for i in range(blocks)]
             for group in range(len(GROUP_SIZES))]
    manager._pending_cpu_hits[request_id] = (lists, blocks * BLOCK, computed)
    return lists


# --- 1. the switch is the authority ------------------------------------------

def test_a_gap_fill_request_with_a_full_ram_hit_gets_no_disk_offer(armed_gap_fill):
    """THE CONTROL HALF OF THE PAIR. Same store, same request, cheaper switch.

    The disk holds blocks 0..3 and RAM has all four, so RAM wins the span, the disk
    continuation would start at block 4, there is none, and the request keeps its RAM
    hit with `no_disk_run` counted. This is precisely the behaviour that made the read
    path untestable, and it is deliberately still the behaviour when only the gap-fill
    switch is up.
    """
    fill(armed_gap_fill, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 4)
    ram_result = (4 * BLOCK, False)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["refused"]["no_disk_run"] == 1
    assert counters["authority_lookups"] == 0
    assert counters["local_hit_ignored"] == 0
    assert counters["offers"] == 0
    assert counters["serving_as"] == "ram_first_gap_fill"


def test_the_authority_switch_gives_a_full_ram_hit_the_whole_span(armed):
    """THE OTHER HALF: the switch only. The disk index decides, from block 0.

    Nothing else is different - the same store, the same four spilled blocks, the same
    four-block RAM hit the fork pinned and reported. The two halves differ in which
    switch is up and in nothing else, which is what makes the switch, and not a request
    field, the thing under test.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 4)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (4 * BLOCK, False)) == (4 * BLOCK, True), (
        "the disk span is offered even though RAM had the entire prefix")
    counters = disk_restore.counters()
    assert counters["local_hit_ignored"] == 1
    assert counters["local_hit_tokens_ignored"] == 4 * BLOCK
    assert counters["refused"]["no_disk_run"] == 0
    assert counters["serving_as"] == "disk_authority"


def test_a_gap_fill_partial_ram_hit_is_refused_as_it_always_was(armed_gap_fill):
    """`ram_hit_not_block_aligned`, for the posture that consults RAM: unchanged.

    A 100-token RAM hit is a whole number of nothing, and the merged tuple's order
    would no longer mean "one whole block per entry". That rule stays exactly where it
    was for the gap-fill posture.
    """
    fill(armed_gap_fill, seeds=[1, 2])
    assert offer(Manager(), Req(request_id="req-1", blocks=2), 0,
                 (100, False)) == (100, False)
    assert disk_restore.counters()["refused"]["ram_hit_not_block_aligned"] == 1


def test_the_authority_switch_offers_the_span_that_refusal_used_to_block(armed):
    """THE REFUSAL THAT MADE THE DISK UNTESTABLE, now only reachable in gap-fill mode.

    Every small prompt the operator tried landed there: a RAM hit with a remainder, a
    refusal, and no way to exercise a disk read without evicting first. Under disk
    authority that request does not reach it, because the RAM hit it carries is not the
    thing being measured.
    """
    fill(armed, seeds=[1, 2])
    assert offer(Manager(), Req(request_id="req-1", blocks=2), 0,
                 (100, False)) == (2 * BLOCK, True)
    counters = disk_restore.counters()
    assert counters["refused"]["ram_hit_not_block_aligned"] == 0
    assert counters["local_hit_ignored"] == 1
    assert counters["local_hit_tokens_ignored"] == 100


def test_the_offer_uses_zero_ram_tokens_and_names_the_span_it_offered(armed):
    """`last_offer` is the structured difference between a forced read and a real one.

    `ram_tokens_reported` is what the fork found, `ram_tokens_used` is what the offer
    used, and `start_block` is where the disk index was asked. A forced read shows a
    non-zero reported count and a used count of exactly 0.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 2)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (4 * BLOCK, True)
    offer_view = disk_restore.counters()["last_offer"]
    assert offer_view["disk_authority"] is True
    assert offer_view["ram_tokens_reported"] == 2 * BLOCK
    assert offer_view["ram_tokens_used"] == 0
    assert offer_view["computed_tokens"] == 0
    assert offer_view["start_block"] == 0
    assert offer_view["blocks"] == 4
    assert offer_view["disk_tokens"] == 4 * BLOCK


# --- 2. what the marker does NOT touch ---------------------------------------

def test_computed_is_never_forced_to_zero_because_it_is_the_dma_offset(armed):
    """The third field of the merged tuple is where the FORK writes, not an opinion.

    `update_state_after_alloc` places external entry `i` at
    `num_computed_tokens // group_block_size + i` into the request's whole block list.
    Zeroing it would send disk bytes into the request's already-cached prefix blocks -
    blocks another request is reading right now - and leave the freshly allocated
    external region holding whatever was in those slots. So the disk span starts where
    the local prefix ENDS, and that boundary must survive the offer byte for byte.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    lists = ram_hit(manager, 1, computed=BLOCK)
    pinned = {block.block_id for group in lists for block in group}
    # computed = one block, so the disk answers for blocks 1..3 only.
    assert offer(manager, Req(request_id="req-1", blocks=4), BLOCK,
                 (BLOCK, False)) == (3 * BLOCK, True)
    lists, _hit_length, num_computed = manager._pending_cpu_hits["req-1"]
    assert num_computed == BLOCK, "the fork's local prefix must survive untouched"
    assert set(merged_cpu(manager)) & pinned == set(), (
        "no RAM block may appear in the merged tuple: the disk span is served by the "
        "disk, not by a second claim on the same tokens")


def test_the_disk_span_starts_at_the_local_prefix_not_at_zero(armed):
    """A flagged request with a real local prefix gets the span AFTER it.

    This is the honest limit of the marker: the scheduler had those tokens computed
    before it asked, and it placed blocks for them. `start_block` is the assertion
    that the module does not pretend otherwise.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    assert offer(manager, Req(request_id="req-1", blocks=4), BLOCK,
                 (0, False)) == (3 * BLOCK, True)
    assert disk_restore.counters()["last_offer"]["start_block"] == 1


# --- 3. no double serve, no leak ---------------------------------------------

def test_the_forks_pinned_ram_blocks_are_released_exactly_once(armed):
    """Ignoring a RAM hit means PUTTING BACK the blocks the fork pinned for it.

    `get_num_new_matched_tokens` `touch()`es every CPU block it found; the only things
    that ever release that are `update_state_after_alloc`, `request_finished`, and the
    stale-pop on the next lookup for the same id. A disk-authority offer overwrites the
    entry, so without a release the pins leak one span per flagged request - a slow way
    to empty the very pool the offer allocates its destinations from.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    lists = ram_hit(manager, 2)
    pinned = sorted(block.block_id for group in lists for block in group)
    pool = manager.cpu_block_pool
    free_before = pool.get_num_free_blocks()
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (4 * BLOCK, True)
    freed = [sorted(ids) for ids in pool.freed]
    assert pinned in freed, "the fork's pinned hit was released through the fork's own"
    assert freed.count(pinned) == 1, "released once, not twice"
    # ...and our own slots are NOT among them: they belong to the load now.
    allocated = {block.block_id for block in pool.allocated}
    for ids in freed:
        assert not set(ids) & allocated, (
            "the release must free the fork's blocks and never our destinations")
    assert pool.get_num_free_blocks() == (free_before - len(pool.allocated)
                                          + len(pinned)), (
        "the only blocks that came back to the pool are the fork's pinned ones: our "
        "destinations belong to the load now")
    assert disk_restore.counters()["pin_release_failures"] == 0


def test_a_manager_without_the_release_method_refuses_and_leaves_everything(armed,
                                                                            monkeypatch):
    """No `_free_pending_cpu_hit` means no disk authority, and NO leak.

    Refusing is the only safe answer: the alternative is offering disk bytes while a
    pin nobody can release stays on the CPU pool for the boot. The refusal happens
    BEFORE the index read, the allocation and any write to the manager, so the fork's
    tuple and its pin are exactly where it left them.
    """
    monkeypatch.delattr(Manager, "_free_pending_cpu_hit")
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 2)
    pool = manager.cpu_block_pool
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (2 * BLOCK, False), "the fork's answer"
    counters = disk_restore.counters()
    assert counters["refused"]["pin_release_unavailable"] == 1
    assert counters["pin_release_failures"] == 1
    assert counters["local_hit_ignored"] == 0, "nothing was ignored, nothing was given"
    assert counters["slots_allocated"] == 0, "no allocation to undo"
    assert pool.allocated == []
    assert manager._pending_cpu_hits["req-1"][1] == 2 * BLOCK, (
        "the fork's tuple is untouched, so the fork can still free its own pin")


def test_a_request_with_no_fork_tuple_costs_nothing_because_ram_is_not_consulted(armed):
    """Tokens the fork reported with no tuple behind them: nothing to release.

    The gap-fill path treats that as a fatal `ram_shape` (there is nothing to merge
    into). Under disk authority it is nothing at all: the RAM hit is not being used, so
    a missing tuple is a hit that was never pinned, and the disk offer stands.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (4 * BLOCK, True)
    counters = disk_restore.counters()
    assert counters["local_hit_ignored"] == 1
    assert counters["pin_release_failures"] == 0
    assert counters["refused"]["ram_shape"] == 0


# --- 4. the fallback is silent and cheap -------------------------------------

def test_a_request_the_disk_cannot_serve_keeps_the_forks_answer(armed):
    """THE FALLBACK PATH, in one assertion each: no offer, no error, no residue.

    Nothing is half-built when the disk has nothing to say. The tuple handed back is
    the fork's own object, the CPU pool is untouched, the fork's pin is intact for the
    fork to free, and the only trace is a counted refusal.
    """
    fill(armed, seeds=[9])                     # not a continuation of this request
    manager = Manager()
    ram_hit(manager, 1)
    pool = manager.cpu_block_pool
    ram_result = (BLOCK, True)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["refused"]["no_disk_run"] == 1
    assert counters["offers"] == 0
    assert counters["local_hit_ignored"] == 0
    assert counters["slots_allocated"] == 0
    assert pool.allocated == [] and pool.freed == [], (
        "a refused offer allocates nothing and releases nothing: the fork's pin stays "
        "the fork's to free")
    assert pool.get_num_free_blocks() == 1024


def test_a_request_below_one_block_falls_back_without_a_stall(armed):
    """The floor refuses a sub-block span, and the request recomputes like any other."""
    fill(armed, blocks=1)
    manager = Manager()
    request = Req(request_id="req-1", blocks=1, num_tokens=2000)
    assert offer(manager, request, 0, (0, False)) == (0, False)
    assert disk_restore.counters()["refused"]["disk_run_below_one_block"] == 1
    assert manager.cpu_block_pool.allocated == []


# --- 5. the switch is the only door -----------------------------------------

HOSTILE = ["true", "True", 1, 1.0, {}, [], [True], "1", None, object()]


@pytest.mark.parametrize("value", HOSTILE)
def test_a_request_carrying_the_retired_marker_changes_nothing(armed_gap_fill, value):
    """THE MARKER IS DEAD IN EVERY SPELLING, AND THE ONLY WAY TO PROVE IT IS TO SEND IT.

    Under 0.2.22 a dict carrying the boolean `True` here flipped this request into disk
    authority and the assertions below would have read the other way. Now the field is
    not read at all, so a request that still sends it - in any spelling, including the
    one that used to work - gets exactly what the same request with no field gets: the
    gap-fill refusal that the RAM hit decides.

    This is the regression test for the removal. A half-removal that left a
    `request.kv_transfer_params` read behind would pass every "authority works" test in
    the suite and fail this one, because the marker would move a counter that the switch
    is now the only writer of.
    """
    fill(armed_gap_fill, seeds=[1, 2])
    manager = Manager()
    request = Req(request_id="req-1", blocks=2)
    request.kv_transfer_params = {"kvtier_disk_restore": value}
    assert offer(manager, request, 0, (100, False)) == (100, False)
    counters = disk_restore.counters()
    assert counters["authority_lookups"] == 0
    assert counters["local_hit_ignored"] == 0
    assert counters["refused"]["ram_hit_not_block_aligned"] == 1


def test_the_retired_marker_cannot_revoke_authority_either(armed):
    """The other direction, because "no client can decline it" needs its own assertion.

    Same store, same full RAM hit, authority up: the offer stands whether or not the
    request carries the old marker, and it stands when the field holds junk that would
    once have been a parse case. A client cannot get out of the posture it was not
    asked to opt into, and a malformed field is never an error - the request never
    reaches code that looks at it.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    for index, params in enumerate(({"kvtier_disk_restore": True},
                                    {"kvtier_disk_restore": False},
                                    None, "kvtier_disk_restore",
                                    ["kvtier_disk_restore"], 7)):
        manager = Manager()
        ram_hit(manager, 4, request_id=f"req-{index}")
        request = Req(request_id=f"req-{index}", blocks=4)
        request.kv_transfer_params = params
        assert offer(manager, request, 0, (4 * BLOCK, False)) == (4 * BLOCK, True), (
            f"{params!r} must not change a posture the switch decided")
    counters = disk_restore.counters()
    assert counters["local_hit_ignored"] == 6
    assert counters["authority_lookups"] == 6
    assert counters["offers"] == 6


def test_the_gap_fill_switch_is_not_a_route_to_disk_authority(armed_gap_fill):
    """Raising the CHEAPER switch does not make the disk the authority.

    Two switches, two postures, and the pair must not blur: with only `offers_enabled`
    up, a request whose RAM hit covers the prompt keeps that hit and the disk offers
    nothing. This is what `armed_all` plus an unflagged request pinned before 0.2.31,
    and the reason `serving_as` exists - so a reader never has to infer which switch it
    is looking at.
    """
    fill(armed_gap_fill, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 4)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (4 * BLOCK, False)) == (4 * BLOCK, False), (
        "RAM covered the prompt, so the disk has no continuation to offer: exactly "
        "what the pre-0.2.22 path did")
    counters = disk_restore.counters()
    assert counters["refused"]["no_disk_run"] == 1
    assert counters["authority_lookups"] == 0
    assert counters["local_hit_ignored"] == 0
    assert counters["offers"] == 0
    assert counters["serving_as"] == "ram_first_gap_fill"


def test_serving_as_is_the_truth_table_of_the_two_switches(armed):
    """One word, four states, and a fresh boot reads `off`.

    `serving_as` is what the boot line, the counters and the section all publish, so the
    table has to hold in the module that owns it - not just in the docstring that
    describes it. `disk_authority` outranks the gap-fill switch on purpose: when both
    are up, the disk decides, and a reader who had to guess would guess wrong.
    """
    assert disk_restore.serving_as() == "disk_authority"
    disk_restore.set_offer_switch(True)
    assert disk_restore.serving_as() == "disk_authority", \
        "authority outranks gap fill: with both up, the disk decides"
    disk_restore.set_disk_authority(False)
    assert disk_restore.serving_as() == "ram_first_gap_fill"
    disk_restore.set_offer_switch(False)
    assert disk_restore.serving_as() == "off"
    assert disk_restore.disk_posture()["serving_as"] == "off"
    assert disk_restore.counters()["serving_as"] == "off"
    assert "serving_as=off" in disk_restore.boot_line()


def test_the_switch_opens_the_gate_by_itself(armed):
    """One POST, not two: raising authority is enough to serve, no second switch.

    `armed` raises ONLY `disk_authority` and leaves `offers_enabled` down, and the gate
    is open. If the gate ever required both, an operator following the README's single
    command would POST it and see `offers_disabled` forever - the exact shape of a
    "I turned it on and nothing happened" bug report.
    """
    assert disk_restore.disk_posture()["enabled"] is False
    assert disk_restore.gate()["ok"] is True
    assert disk_restore.gate()["reason"] is None


def test_only_a_json_boolean_moves_the_switch(armed):
    """`"true"`, `1`, `None`: refused, and the switch does NOT move.

    The old marker was lenient in the safe direction (a malformed value meant "not
    asked"). A boot switch cannot be lenient, because there is no safe direction: a
    curl typo must not change engine behaviour either way, so the setter validates and
    refuses rather than coercing. A switch that silently accepted `"false"` as truth
    would serve a whole boot from SSD on a string.
    """
    for hostile in ("true", "True", 1, 0, 1.0, None, [], {}, object()):
        verdict = disk_restore.set_disk_authority(hostile)
        assert verdict["ok"] is False, hostile
        assert disk_restore.disk_posture()["disk_authority"] is True, \
            f"{hostile!r} was refused but still moved the switch"
    changes_before = disk_restore.disk_posture()["disk_authority_changes"]
    assert disk_restore.set_disk_authority(False)["ok"] is True
    assert disk_restore.disk_posture()["disk_authority"] is False
    # One move counted, from the real change - and the refusals above added none.
    assert (disk_restore.disk_posture()["disk_authority_changes"]
            == changes_before + 1)


def test_a_boot_with_both_switches_down_costs_nothing_at_all(switches_down):
    """The guarantee a production boot depends on, and the incident that made it one.

    No gate pass, no index refresh, no allocation, no authority line in the ledger:
    with both switches down the plugin is indistinguishable from absent on the lookup
    path, and the refusal it does record names both ops that would open the gate. On
    2026-10-05 a boot keyed off the SPILL ARM instead of these switches put 123,474
    un-requested 160,000-token offers in front of live traffic; this is the assertion
    that says that boot cannot happen again while this test passes.
    """
    fill(switches_down, seeds=[1, 2, 3, 4])
    manager = Manager()
    request = Req(request_id="req-1", blocks=4)
    ram_result = (2 * BLOCK, True)
    assert offer(manager, request, 0, ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["serving_as"] == "off"
    assert counters["authority_lookups"] == 0
    assert counters["index_refreshes"] == 0
    assert counters["offers"] == 0
    assert counters["refused"]["offers_disabled"] == 1
    assert "op\":\"authority\"" in disk_restore.OFFER_REFUSAL_WHY[
        "offers_disabled"], \
        "the refusal must name the op that opens the gate, not just say 'disabled'"
    assert manager.cpu_block_pool.allocated == []


# --- 6. loudness -------------------------------------------------------------

def test_a_forced_read_says_so_in_the_log(armed, caplog):
    """An operator must be able to tell a forced read from a real one from the log."""
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 2)
    with caplog.at_level(logging.WARNING,
                         logger="kvtier_dev_plugin.disk_restore"):
        assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                     (2 * BLOCK, False)) == (4 * BLOCK, True)
    lines = [record.getMessage() for record in caplog.records
             if "from DISK" in record.getMessage()]
    assert lines, "a disk offer that overrode a RAM hit must be visible in the log"
    assert f"{2 * BLOCK}-token host-RAM hit" in lines[0]


def test_the_fallback_is_quiet(armed, caplog):
    """A flagged request the disk cannot serve adds no warning line.

    The loud rule belongs to the event that actually moved bytes. A miss is a routine
    recompute, and warning on it would train everyone to ignore the warning.
    """
    fill(armed, seeds=[9])
    with caplog.at_level(logging.WARNING,
                         logger="kvtier_dev_plugin.disk_restore"):
        assert offer(Manager(), Req(request_id="req-1", blocks=4), 0,
                     (0, False)) == (0, False)
    assert not [record for record in caplog.records
                if "from DISK" in record.getMessage()]


def test_the_published_section_carries_the_authority_fields(armed):
    """`restore.disk_authority` etc. are in the scheduler's own answer, not only in
    `counters()`. A payload a poller already reads is the difference between an
    answer and an experiment."""
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 2)
    offer(manager, Req(request_id="req-1", blocks=4), 0, (2 * BLOCK, False))
    section = disk_restore.scheduler_section()
    assert section["disk_authority"] == 1
    assert section["local_hit_ignored"] == 1
    assert section["local_hit_tokens_ignored"] == 2 * BLOCK
    assert section["pin_release_failures"] == 0
    assert section["last_offer"]["ram_tokens_used"] == 0
    assert "host-RAM" in section["last_local_hit_ignored"]
    for field in ("disk_authority", "local_hit_ignored",
                  "local_hit_tokens_ignored", "pin_release_failures",
                  "last_local_hit_ignored", "last_offer"):
        assert field in section["units"], f"{field} has no unit beside it"


def test_a_refusal_never_moves_the_ignored_hit_counters(armed):
    """`local_hit_ignored` counts OFFERS. A refusal leaves the fork's hit in place,
    so counting it there would claim a sacrifice the request never made."""
    fill(armed, seeds=[9])
    manager = Manager()
    ram_hit(manager, 2)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (2 * BLOCK, False)) == (2 * BLOCK, False)
    counters = disk_restore.counters()
    assert counters["disk_authority"] == 1, "the rule fired"
    assert counters["local_hit_ignored"] == 0, "but no offer was made on it"
    assert counters["local_hit_tokens_ignored"] == 0
    assert manager.cpu_block_pool.freed == []
