r"""The marker means "the DISK decides this request's prefix" - and nothing else does.

WHAT CHANGED IN 0.2.22, AND WHY IT IS A SEMANTIC CHANGE AND NOT A KNOB. Until now the
restore path filled the gap the host-RAM tier left: consult RAM, offer the disk's
continuation after it. That made the read path impossible to exercise. A real prompt
either hit RAM completely (nothing left to offer), or hit it PARTIALLY, which the
alignment rule refused as `ram_hit_not_block_aligned` - so proving the disk could
serve a single block meant making GPU and RAM forget the prefix first, ~1.3M tokens of
filler per attempt and seven failed runs. The rule now is one switch with no new field:

  * the marker present  -> the disk is the AUTHORITY. The fork's host-RAM hit is
    treated as zero tokens for the offer arithmetic, and the disk index decides the
    span from the local GPU-computed boundary.
  * no marker           -> byte-identical to before. The RAM tier answers, the disk
    fills the gap, and the existing client needs no change at all.

The three things these tests exist to pin, in the order they would hurt:

1. NO DOUBLE SERVE. The offer claims the disk span and NOTHING else: the returned
   count is the disk's, the merged tuple contains only our own slots, and the fork's
   pinned CPU blocks are released exactly once. The token span is never claimed twice
   and never left pinned.
2. `computed` IS NOT FORKED. It is the offset the fork's own DMA writes at
   (`manager.py:441-463` places entry `i` at `num_computed_tokens // group_block_size
   + i` into the request's whole block list). Zeroing it would aim disk bytes at the
   request's already-cached, ref-counted prefix blocks - blocks OTHER requests are
   reading - and leave the newly allocated region unwritten. So the marker overrides
   RAM, and never the local GPU prefix, and a test says so in the tuple itself.
3. EVERY REFUSAL IS SILENT AND CHEAP. A flagged request the disk cannot serve leaves
   the fork's answer, the fork's pin and the CPU pool exactly as they were. No error
   reaches the client, nothing is allocated, nothing stalls.

Plus the gate: the marker is the ONLY door to this, and it is strictly
`kv_transfer_params` being a dict with the boolean `True` in it. `scope=all` is not a
back door, and neither is any other spelling of "yes".
"""

from __future__ import annotations

import logging

import pytest
import test_disk_restore as base

from kvtier_dev_plugin import disk_restore

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them (pytest collects fixtures from imported names).
armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
armed_all = base.armed_all  # noqa: F811 - scope=all: a candidate request, no marker
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


# --- 1. the marker is the authority ------------------------------------------

def test_an_unmarked_request_with_a_full_ram_hit_gets_no_disk_offer(armed_all):
    """THE CONTROL HALF OF THE PAIR. Same store, same request, no marker.

    The disk holds blocks 0..3 and RAM has all four, so RAM wins the span, the disk
    continuation would start at block 4, there is none, and the request keeps its RAM
    hit with `no_disk_run` counted. This is precisely the behaviour that made the read
    path untestable, and it is deliberately still the behaviour for a request that did
    not ask.
    """
    fill(armed_all, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 4)
    ram_result = (4 * BLOCK, False)
    assert offer(manager, Req(request_id="req-1", blocks=4, restore=False), 0,
                 ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["refused"]["no_disk_run"] == 1
    assert counters["disk_authority"] == 0
    assert counters["local_hit_ignored"] == 0
    assert counters["offers"] == 0


def test_a_marked_request_with_a_full_ram_hit_gets_the_whole_span(armed_all):
    """THE OTHER HALF: marker only. The disk index decides, from block 0.

    Nothing else is different - the same fixture, the same four spilled blocks, the
    same four-block RAM hit the fork pinned and reported. `scope=all` here rather than
    `optin` so the two halves differ in the marker and in nothing else.
    """
    fill(armed_all, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 4)
    assert offer(manager, Req(request_id="req-1", blocks=4), 0,
                 (4 * BLOCK, False)) == (4 * BLOCK, True), (
        "the disk span is offered even though RAM had the entire prefix")
    counters = disk_restore.counters()
    assert counters["local_hit_ignored"] == 1
    assert counters["local_hit_tokens_ignored"] == 4 * BLOCK
    assert counters["refused"]["no_disk_run"] == 0


def test_an_unmarked_partial_ram_hit_is_refused_as_it_always_was(armed_all):
    """`ram_hit_not_block_aligned`, for the request that did not ask: unchanged.

    A 100-token RAM hit is a whole number of nothing, and the merged tuple's order
    would no longer mean "one whole block per entry". That rule stays exactly where it
    was for an unmarked request.
    """
    fill(armed_all, seeds=[1, 2])
    assert offer(Manager(), Req(request_id="req-1", blocks=2, restore=False), 0,
                 (100, False)) == (100, False)
    assert disk_restore.counters()["refused"]["ram_hit_not_block_aligned"] == 1


def test_a_marked_partial_ram_hit_offers_the_span_instead(armed):
    """THE REFUSAL THAT MADE THE DISK UNTESTABLE, now only reachable without a marker.

    Every small prompt the operator tried landed here: a RAM hit with a remainder, a
    refusal, and no way to exercise a disk read without evicting first. A flagged
    request does not reach it, because the RAM hit it carries is not the thing being
    measured.
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


def test_a_flagged_request_needs_no_fork_tuple_because_ram_is_not_consulted(armed):
    """Tokens the fork reported with no tuple behind them: nothing to release.

    The UNMARKED path treats that as a fatal `ram_shape` (there is nothing to merge
    into). Marked, it is nothing at all: the RAM hit is not being used, so a missing
    tuple is a hit that was never pinned, and the disk offer stands.
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

def test_a_flagged_request_the_disk_cannot_serve_keeps_the_forks_answer(armed):
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


def test_a_flagged_request_below_one_block_falls_back_without_a_stall(armed):
    """The floor refuses a sub-block span, and the request recomputes like any other."""
    fill(armed, blocks=1)
    manager = Manager()
    request = Req(request_id="req-1", blocks=1, num_tokens=2000)
    assert offer(manager, request, 0, (0, False)) == (0, False)
    assert disk_restore.counters()["refused"]["disk_run_below_one_block"] == 1
    assert manager.cpu_block_pool.allocated == []


# --- 5. the marker is the only door -----------------------------------------

HOSTILE = ["true", "True", 1, 1.0, {}, [], [True], "1", None, object()]


@pytest.mark.parametrize("value", HOSTILE)
def test_only_the_boolean_true_flags_a_request(armed_all, value):
    """`"true"`, `1`, a nested object: not asked, and NOT a route to disk authority.

    A client controls this field. Being loose about it means disk work on a request
    that never asked, so the rule is a real dict carrying the boolean `True`, and every
    other spelling must land on the unmarked path - here, the RAM-aligned refusal that
    proves the RAM hit is still the thing deciding the span. `armed_all` so the request
    is a CANDIDATE: the only difference between these and a flagged request must be the
    marker itself, not the scope keeping it out of the room.
    """
    fill(armed_all, seeds=[1, 2])
    manager = Manager()
    request = Req(request_id="req-1", blocks=2, restore=False)
    request.kv_transfer_params = {disk_restore.RESTORE_MARKER_KEY: value}
    assert offer(manager, request, 0, (100, False)) == (100, False)
    counters = disk_restore.counters()
    assert counters["disk_authority"] == 0
    assert counters["local_hit_ignored"] == 0
    assert counters["refused"]["ram_hit_not_block_aligned"] == 1


def test_kv_transfer_params_that_is_not_a_dict_is_not_a_request_that_asked(armed):
    """A list, a string or None in the engine's field: not flagged, never an error."""
    fill(armed, seeds=[1, 2])
    for params in (None, "kvtier_disk_restore",
                   [disk_restore.RESTORE_MARKER_KEY], 7):
        request = Req(request_id="req-1", blocks=2, restore=False)
        request.kv_transfer_params = params
        assert offer(Manager(), request, 0, (0, False)) == (0, False)
        counters = disk_restore.counters()
        assert counters["disk_authority"] == 0
        assert counters["offers"] == 0
    assert disk_restore.counters()["skipped_no_marker"] == 4


def test_scope_all_is_not_a_route_to_disk_authority(armed_all):
    """`scope=all` widens WHO MAY BE OFFERED, not WHO IS THE AUTHORITY.

    The marker is the switch. A candidate request that does not carry it keeps the
    pre-0.2.22 behaviour exactly: RAM first, the disk filling the gap - which is why
    this fixture exists, and why an operator cannot turn every live request into a
    disk-forced one by reaching for the scope knob.
    """
    fill(armed_all, seeds=[1, 2, 3, 4])
    manager = Manager()
    ram_hit(manager, 4)
    assert offer(manager, Req(request_id="req-1", blocks=4, restore=False), 0,
                 (4 * BLOCK, False)) == (4 * BLOCK, False), (
        "RAM covered the prompt, so the disk has no continuation to offer: exactly "
        "what an unmarked request did before 0.2.22")
    counters = disk_restore.counters()
    assert counters["refused"]["no_disk_run"] == 1
    assert counters["disk_authority"] == 0
    assert counters["local_hit_ignored"] == 0
    assert counters["offers"] == 0


def test_an_unmarked_request_in_optin_scope_costs_one_attribute_read(armed):
    """Nothing on the new path runs for a request that did not ask.

    No gate, no index refresh, no allocation, no counters from the authority rule.
    This is the guarantee the existing client depends on: the RAM tier answers it and
    the plugin is indistinguishable from absent.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    manager = Manager()
    request = Req(request_id="req-1", blocks=4, restore=False)
    ram_result = (2 * BLOCK, True)
    assert offer(manager, request, 0, ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["skipped_no_marker"] == 1
    assert counters["disk_authority"] == 0
    assert counters["index_refreshes"] == 0
    assert counters["offers"] == 0
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
