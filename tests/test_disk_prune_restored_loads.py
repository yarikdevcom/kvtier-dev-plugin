r"""STEP 2 OF RETIRING THE ARENA: pruning an adopted restore's destinations.

A disk restore reaches the GPU as a PAIR. `build_connector_meta` ships
`load_cpu_blocks` (the arena slot the bytes were read into) and `load_gpu_blocks` (the
window they belong to), and the fork's `start_load_kv` DMAs pair i to pair i. This suite
tests the flag that REMOVES those pairs once the restore has been adopted and its bytes
have landed - `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS`, default OFF - and it is ordered
by what hurts when it goes wrong, which is the one thing that makes this file different
from every other suite here: a mistake in these lists is not a crash and not a stall.
`copy_blocks` pairs `src[i]` with `dst[i]` POSITIONALLY (`cuda_mem_ops.py:244`), so a
prune that touches one side, or that names an index by arithmetic instead of by the
scheduler's own index-aligned identity, writes block K's bytes into block K+1's
destination and the engine answers confidently and wrongly.

THE ORDER A REVIEWER SHOULD READ IT IN:

1. OFF IS THE DEFAULT AND OFF IS UNCHANGED. With the variable unset the apply point
   produces the lists the previous release would have produced, entry for entry, in
   every state (nothing failed, something failed, nothing adopted) - and every prune
   counter stays exactly where a fresh boot left it.
2. THE SHIFT IS IMPOSSIBLE BY CONSTRUCTION, AND THE REFUSAL IS TESTED AS A REFUSAL. A
   call that hands the prune two different index sets is refused and hands back NO
   lists; the test then shows what the honoured call would have done, so the refusal is
   not a formality.
3. BOTH-SIDED IS CORRECT. The pairing is asserted PAIR BY PAIR against the pre-prune
   step, never by comparing lengths: equal lengths are exactly what a one-block shift
   still has.
4. THE GROUP BINDING SURVIVES (Gate B). The fork pairs group g's entry i with
   `block_ids_by_group[g][num_computed // group_block_sizes[g] + i]`, and the boundary
   groups differ from the positional group in BOTH their size and their hit shape
   (`[null] * (B - 1) + [real]`). The prune must name the entries the scheduler named,
   and the test asserts that the positional rule would have named DIFFERENT ones.
5. ADOPTED, NOT OFFERED. An offer of N blocks whose adoption is M prunes M and leaves
   the N-M destinations that arrived as RAM entries exactly where they were.
6. A FAILED READ KEEPS ITS RANGE, counted: the recompute path is what makes the tier
   safe, and the counter proves it was not traded away.
7. THE SHAPE AND THE LOG. A refused prune publishes the same key set as an accepted one,
   and the log says each episode once, never once per step.

Everything is the sibling suites' harness: a `tmp_path` store, ctypes-backed CPU slots,
the real reader thread settling on a bounded poll, and the fork's own pairing loop
reimplemented honestly rather than mocked.
"""

from __future__ import annotations

import pytest
import test_disk_restore as base

from kvtier_dev_plugin import (
    config,
    disk_gpu_write,
    disk_restore,
    disk_restore_verify,
)

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them, autouse ones included (a fresh boot per test, and a /tmp root that is allowed).
armed = base.armed                          # noqa: F811 - the gate open, the switch up
armed_gap_fill = base.armed_gap_fill        # noqa: F811 - RAM first, disk extends
allow_tmp_root = base.allow_tmp_root        # noqa: F811 - autouse
fresh = base.fresh                          # noqa: F811 - autouse: a fresh boot
fill = base.fill
offer = base.offer
vanish = base.vanish
settle = base.settle
bind = base.bind
merged_cpu = base.merged_cpu
BLOCK = base.BLOCK
Req = base.Req
Meta = base.Meta
Worker = base.Worker
Region = base.Region
Slot = base.Slot
Manager = base.Manager
DIGEST = base.DIGEST
PLACEHOLDER = base.PLACEHOLDER

# THE LIVE GEOMETRY (`test_disk_gdn_boundary_restore.py` transcribes it from the box):
# four GDN groups whose `MambaManager` is not positionally stable, a QSA ring that is
# not prefix-cacheable at all, and one full-attention group. The GDN groups' sizes are
# then CUT IN HALF below, because a boundary group whose `group_block_sizes` differs
# from the positional group's is the shape a positional offset gets wrong.
GDN = (0, 1, 2, 3)
ATTENTION = 5
HYBRID_CACHEABLE = (0, 1, 2, 3, ATTENTION)


@pytest.fixture
def pruning(monkeypatch):
    """THE FLAG, raised out loud. Nothing in this file may rely on the environment.

    The switch is read once per process and cached in the module's state, so the cache
    is dropped here in both directions: a test that forgot to name the flag gets the
    shipped OFF, and a test that named it gets it without leaking the decision into the
    next test file through the module-level cache.
    """
    monkeypatch.setenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, "1")
    disk_restore._state["prune_switch"] = None
    yield True
    disk_restore._state["prune_switch"] = None


@pytest.fixture
def off(monkeypatch):
    """The shipped posture, asserted rather than assumed, with the cache dropped."""
    monkeypatch.delenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, raising=False)
    disk_restore._state["prune_switch"] = None
    yield False
    disk_restore._state["prune_switch"] = None


@pytest.fixture
def delivered(monkeypatch):
    """THE STEP 3 PRECONDITION (0.2.52): the writer covers this step's entries.

    Since 0.2.52 the prune asks `disk_gpu_write.observe_apply` WHICH entries it put
    bytes
    into and prunes ONLY those. A test that raises the prune flag alone therefore prunes
    NOTHING - correctly, and the point of the whole change - so a test that wants to
    exercise the prune's GEOMETRY has to name a writer as well. That is what this
    fixture
    is: a stand-in for delivery mode at the one module boundary that needs one.

    WHAT IT STANDS IN FOR, and why that is not a test cheating: the real
    `disk_gpu_write.observe_apply` returns exactly this shape - `{"written": n,
    "written_indices": {survivor ids}, ...}` - and a separate test in
    `test_disk_gpu_write.py` proves the real function returns the ids of the entries it
    enqueued. This stand-in reproduces the CONTRACT (every applied, un-failed survivor
    is covered) without a CUDA context, a `load_stream` or a pinned pool, none of which
    exist
    in a unit test. It is the delivery-mode posture, and it is deliberately NOT the
    witness-mode posture: witness mode covers one entry in sixty-four and the tests
    below
    are about what happens when the plugin owns the delivery.

    A test that wants the OPPOSITE - "the prune flag is on and nothing was delivered" -
    names no fixture and gets it, which is
    `test_the_prune_flag_alone_prunes_nothing_...`.
    """
    calls: list = []

    def stand_in(worker, batch, failed):
        jobs = (batch or {}).get("jobs") or {}
        survivors = [job_id for job_id in jobs if job_id not in set(failed or {})]
        calls.append({"jobs": sorted(jobs), "survivors": sorted(survivors)})
        return {"seen": len(survivors), "written": len(survivors),
                "written_indices": set(survivors), "delivery_mode": True,
                "reason": None, "cause": None}

    monkeypatch.setattr(disk_restore.disk_gpu_write, "observe_apply", stand_in)
    # THE KILL-SWITCH, off: these tests are about the prune's GEOMETRY, and a latch left
    # fired by an earlier test in the same process would refuse every prune here and
    # make a geometry failure read like a kill-switch. `delivered` asserts it is unfired
    # so a leak from another file is caught here instead of in a confusing assertion
    # below.
    assert disk_restore.disk_gpu_write.kill_switch_engaged() is False, \
        "a fired kill-switch in the environment means an earlier test leaked its latch"
    monkeypatch.setattr(disk_restore.disk_gpu_write, "kill_switch_engaged",
                        lambda: False)
    yield calls


# --- the harness: be the fork for one step -----------------------------------

def hybrid(sizes=None, *, cacheable=HYBRID_CACHEABLE, **kwargs):
    """The live box's manager, with GDN groups answering `stable=False`, `mode="align"`.

    `sizes` defaults to the box's own (3,200 everywhere, an 8-token QSA ring); passing
    a smaller boundary size is how the Gate-B test gives a boundary group an offset that
    a positional derivation would get wrong.
    """
    sizes = list(sizes if sizes is not None
                 else [BLOCK, BLOCK, BLOCK, BLOCK, 8, BLOCK])
    manager = base.Manager(group_sizes=sizes, cacheable=list(cacheable), **kwargs)
    from tests import fakes_debug  # noqa: PLC0415 - the geometry fakes, locally bound
    items = []
    for group in range(len(sizes)):
        if group in GDN:
            items.append(fakes_debug.manager_item(
                group,
                fakes_debug.mamba_spec(block_size=sizes[group],
                                       mamba_cache_mode="align"),
                stable=False, mamba_cache_mode="align"))
        else:
            items.append(fakes_debug.manager_item(
                group, fakes_debug.attention_spec(block_size=sizes[group]),
                stable=True))
    manager.cpu_coordinator = fakes_debug.coordinator(items)
    return manager


def fork_pair(manager, request_id="req-1", blocks=1000, computed=0, event=11,
              take_per_group=None):
    """`update_state_after_alloc` (`manager.py:446-478`), and the metadata it implies.

    Returns the flattened `(gpu_ids, cpu_ids, owners)` the fork would ship, where
    `owners[index]` is the `(group, index_within_group, group_offset)` whose pair that
    is, computed by the fork's OWN rule: take `cdiv(num_external_tokens,
    group_block_sizes[g])` leading entries of group g, skip the entries whose `is_null`
    is set while KEEPING THE INDEX, and address the rest at
    `group_gpu_ids[num_computed // group_block_sizes[g] + i]`.

    `take_per_group` truncates each group's taken entries to that many LEADING ones,
    which is what the scheduler does when it adopts fewer blocks than were offered -
    the shape `test_an_offer_the_scheduler_truncated_prunes_only_what_arrived` needs.
    """
    merged = manager._pending_cpu_hits[request_id]
    lists, hit_length, num_computed = merged[0], merged[1], merged[2]
    assert num_computed == computed, (
        "the offer must carry the fork's local prefix through untouched, because the "
        "pairing offsets are computed from it")
    owners: list = []
    gpu_ids: list = []
    cpu_ids: list = []
    for group in range(len(lists)):
        if group not in manager.prefix_cacheable_group_ids:
            continue
        size = manager.group_block_sizes[group]
        take = -(-hit_length // size)                  # `cdiv`, as the fork does
        if take_per_group is not None:
            take = min(take, take_per_group)
        start = num_computed // size
        for index, block in enumerate(lists[group][:take]):
            if getattr(block, "is_null", False):
                continue
            gpu_ids.append(blocks + group * 100 + start + index)
            cpu_ids.append(int(block.block_id))
            owners.append((group, index, num_computed // size))
    return gpu_ids, cpu_ids, owners


def ship(manager, request_id="req-1", computed=0, event=11, ram=(), extra=0,
         take_per_group=None):
    """The metadata the scheduler would ship for the merged tuple, plus the pairing.

    `ram` appends host-RAM-only entries BEFORE the disk ones (path "" and no identity),
    and `extra` appends them AFTER: both are destinations this module must never touch,
    and having them on both sides of the disk range is what makes a shift visible.
    """
    gpu_ids, cpu_ids, owners = fork_pair(manager, request_id, computed=computed,
                                         event=event,
                                         take_per_group=take_per_group)
    cpu = list(cpu_ids)
    gpu = list(gpu_ids)
    disk_range = (len(ram), len(ram) + len(cpu))
    for index, (cpu_id, gpu_id) in enumerate(ram):
        cpu.insert(index, cpu_id)
        gpu.insert(index, gpu_id)
    for offset in range(extra):
        cpu.append(900 + offset)
        gpu.append(999_000 + offset)
    metadata = Meta(load_event=event, cpu=cpu, gpu=gpu,
                    reqs={event: [request_id]})
    disk_restore.build_load_paths(manager, metadata)
    return metadata, owners, disk_range


def worker_for(metadata, slots=512):
    """A worker with a real, wide-enough arena and the fork's pending-event set."""
    worker = Worker(regions={"g0": Region(slots=slots)})
    worker._pending_load_event_indices.add(metadata.load_event)
    bind(worker, metadata)
    settle(disk_restore._batch_id(metadata.load_event))
    return worker


# --- 1. off is the default, and off is the old behaviour ---------------------

def test_the_shipped_boot_prunes_nothing_and_says_which_switch_it_read(off):
    """DEFAULT OFF, in both senses: the answer is False and the step is untouched.

    Asserted before any store exists, because "with the flag off nothing changes" has to
    be true of a process that has never applied a restore, not only of one that has.
    """
    assert disk_restore.prune_restored_loads_enabled() is False
    state = disk_restore.prune_state()
    assert state["enabled"] is False
    assert state["switch"]["raw"] is None
    assert state["switch"]["env_var"] == config.ENV_DISK_PRUNE_RESTORED_LOADS
    for name, count in state["refused"].items():
        assert count == 0, name
    assert state["entries"] == 0 and state["steps"] == 0
    # Every refusal name is published even though nothing has happened, so a reader can
    # tell "never refused" from "this build has no refusal vocabulary".
    assert set(state["refused"]) == set(disk_restore.PRUNE_REFUSALS)


def test_a_typo_turns_the_prune_off_and_reports_itself(off, monkeypatch, caplog):
    """`tru` is not `true`: OFF, and the direction a typo moves in is the safe one.

    Same grammar as every other device-touching flag in the package, tested because the
    failure it protects against is a boot pruning load lists nobody asked it to prune.
    """
    monkeypatch.setenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, "tru")
    disk_restore._state["prune_switch"] = None
    assert disk_restore.prune_restored_loads_enabled() is False
    state = disk_restore.prune_state()
    assert state["switch"]["recognised"] is False
    assert state["switch"]["raw"] == "tru"
    assert any("neither an on word nor an off word" in line.message
               for line in caplog.records)


def test_flag_off_reproduces_the_previous_release_field_for_field(armed, off):
    """BYTE-IDENTICAL ON THE DEFAULT PATH, including the failed-read shape.

    The reference is not a snapshot of this build: the expected lists are recomputed
    here by the algorithm every release through 0.2.48 ran (`drop` is the set of failed
    entries, and every other entry stays, in order), and the produced metadata is
    compared to it entry by entry. The same scenario with the flag on is a different
    test below, so the two differ in exactly one thing: the prune.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager, ram=[(400, 70_000)], extra=1)
    before = (list(metadata.load_cpu_blocks), list(metadata.load_gpu_blocks))
    groups = sorted({group for group, _, _ in owners})
    # Fail exactly ONE disk entry, BEFORE the reader thread opens it: the file the
    # worker would read is gone, so that entry's read fails and every other entry of
    # the step lands. This is the shape the pre-0.2.49 prune loop was written for.
    vanish(armed, seeds=(1,), groups=(groups[0],))
    worker = worker_for(metadata)
    assert dict(disk_restore._slot_paths), \
        "the harness must have named real disk entries"
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True
    assert report["dropped"] == 1
    assert report["pruned"] == 0, "the flag is off: it may not prune anything"
    dropped = {index for index in range(len(before[0]))
               if before[0][index] not in metadata.load_cpu_blocks}
    expected_cpu = [value for index, value in enumerate(before[0])
                    if index not in dropped]
    expected_gpu = [value for index, value in enumerate(before[1])
                    if index not in dropped]
    assert metadata.load_cpu_blocks == expected_cpu
    assert metadata.load_gpu_blocks == expected_gpu
    assert metadata.load_paths == [""] * len(expected_cpu)
    state = disk_restore.prune_state()
    assert state["entries"] == 0 and state["steps"] == 0
    assert state["refusals_total"] == 0
    assert state["skipped_failed_entries"] == 0
    assert state["unwitnessed_entries"] == 0
    assert state["emptied_steps"] == 0
    assert state["last"] is None


def _slot_paths_snapshot() -> dict:
    return dict(disk_restore._slot_paths)
# --- 2. the shift is impossible: a one-sided prune is refused ----------------

def test_pruning_one_list_only_is_refused_and_no_list_is_handed_back():
    """THE HAZARD THIS FILE EXISTS FOR, constructed and refused.

    `copy_blocks` pairs by index (`cuda_mem_ops.py:244`), so honouring this call would
    leave the CPU side one entry shorter than the GPU side and pair every later block
    with the wrong destination. The assertion that matters is not `ok is False` alone:
    it is that no lists come back (a caller cannot use a refused result by accident)
    AND that the shift this prevented is real - computed below, so the refusal is shown
    to be preventing a wrong pairing rather than a cosmetic one.
    """
    cpu = [10, 11, 12, 13, 14]
    gpu = [20, 21, 22, 23, 24]
    verdict = disk_restore.prune_load_pairs(cpu, gpu, [2], [])
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "one_sided_prune"
    assert verdict["cpu"] is None and verdict["gpu"] is None, \
        "a refused prune must hand back nothing a caller could ship"
    assert verdict["pruned"] == 0
    # WHAT WOULD HAVE HAPPENED. Dropping index 2 from the CPU side alone pairs 13 with
    # 22 and 14 with 23 - two destinations holding the wrong block's bytes, silently.
    one_sided_cpu = [value for index, value in enumerate(cpu) if index != 2]
    # NOT `strict=True`, and deliberately: this zip is not our code, it is the FORK'S
    # pairing (`copy_blocks` pairs src[i] with dst[i] and stops at the shorter list), so
    # it must be modelled the way it behaves - silently, on unequal lengths.
    shifted = dict(zip(one_sided_cpu, gpu, strict=False))
    assert shifted == {10: 20, 11: 21, 13: 22, 14: 23}
    assert shifted[13] != 23, "the shift this refusal is preventing"
    # The same call with both sides moved together is accepted, and pairs 13 with 23.
    both = disk_restore.prune_load_pairs(cpu, gpu, [2])
    assert both["ok"] is True
    assert dict(zip(both["cpu"], both["gpu"],
                    strict=True)) == {10: 20, 11: 21, 13: 23, 14: 24}


def test_a_malformed_index_set_is_refused_rather_than_interpreted():
    """Negative, out-of-range, duplicated and over-cap index sets all refuse by name.

    None of these is a "best effort" case: a prune that guessed at a malformed set
    would be choosing which destination to shift.
    """
    cpu = list(range(6))
    gpu = [100 + value for value in cpu]
    for drop, kind in (([-1], "bad_index"),
                       ([99], "out_of_range"),
                       ([1, 1], "duplicate_index"),
                       (list(range(6)), "over_cap")):
        verdict = disk_restore.prune_load_pairs(cpu, gpu, drop, None,
                                                cap=3)
        assert verdict["ok"] is False, drop
        assert verdict["reason_kind"] == kind, (drop, verdict["reason"])
    # A step whose two sides already disagree in length cannot be pruned at all: there
    # is no correspondence to preserve, and guessing which list is right is the bug.
    broken = disk_restore.prune_load_pairs([1, 2, 3], [7, 8], [0])
    assert broken["ok"] is False
    assert broken["reason_kind"] == "length_mismatch"


def test_a_list_that_moved_under_us_refuses_the_whole_prune():
    """`destination_drift`: bind recorded ids the step no longer holds.

    The pairing the prune preserves is the pairing BIND saw. If anything rewrote the
    lists in between, the module's own evidence about which pair is which is stale, and
    a prune justified by stale evidence is the shift with extra steps.
    """
    cpu = [10, 11, 12]
    gpu = [20, 21, 22]
    expected = {0: (10, 20), 1: (11, 21), 2: (12, 22)}
    ok = disk_restore.prune_load_pairs(cpu, gpu, [0], None,
                                       groups={0: 5, 1: 5, 2: 5},
                                       expected=expected)
    assert ok["ok"] is True
    moved = dict(expected)
    moved[1] = (11, 999)                      # the GPU side was rewritten elsewhere
    drifted = disk_restore.prune_load_pairs(cpu, gpu, [0], None,
                                            groups={0: 5, 1: 5, 2: 5},
                                            expected=moved)
    assert drifted["ok"] is False
    assert drifted["reason_kind"] == "destination_drift"


def test_a_group_whose_two_sides_do_not_agree_refuses_the_prune():
    """GATE B'S CHECK ITSELF: a per-group delta that is not zero is a refusal.

    This is the in-band, `-O`-proof version of "assert the two lists still correspond":
    group 1 can confirm two CPU entries and only one GPU entry, and the module reports
    the delta instead of shipping it.
    """
    cpu = [10, 11, 12]
    gpu = [20, 21, 22]
    expected = {0: (10, 20), 1: (11, 21), 2: (12, 22)}
    groups = {0: 0, 1: 1, 2: 1}
    verdict = disk_restore.prune_load_pairs(cpu, gpu, [0], None, groups=groups,
                                            expected=expected)
    assert verdict["ok"] is True
    # BOTH MAPS COVER EVERY GROUP IN THE STEP, touched or not: a reader must be
    # able to tell "group 1 lost nothing" from "group 1 was not reported".
    assert verdict["pruned_by_group"] == {"0": 1, "1": 0}
    assert verdict["group_deltas"] == {"0": 0, "1": 0}
    # Now the same step with group 1's GPU half missing one entry the CPU half has.
    lopsided = {0: (10, 20), 1: (11, 21), 2: (12, None)}
    refused = disk_restore.prune_load_pairs(cpu, gpu, [0], None, groups=groups,
                                            expected=lopsided)
    assert refused["ok"] is False
    assert refused["reason_kind"] == "group_delta_mismatch"
    assert refused["group_deltas"].get("1") == 1


# --- 3. both-sided is correct: the survivors keep their partners --------------

def test_both_pruned_keeps_every_later_pair_with_its_own_source(
        pruning, delivered, armed):
    """THE POSITIVE CASE, asserted as a PAIRING and not as a length.

    The step is deliberately built with host-RAM destinations on BOTH sides of the disk
    range: a prune that removed the disk entries from one side would still leave two
    lists of equal length, and only a pair-by-pair comparison says whether the
    survivors kept the partners the scheduler gave them.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, disk_range = ship(manager, ram=[(400, 70_000)], extra=2)
    before = list(metadata.load_cpu_blocks), list(metadata.load_gpu_blocks)
    before_pairs = list(zip(before[0], before[1], strict=True))
    expected_pairs = [before_pairs[index] for index in range(len(before[0]))
                      if not disk_range[0] <= index < disk_range[1]]
    worker = worker_for(metadata)
    assert disk_restore.worker_before_load(worker)["pruned"] == len(owners)
    after = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    assert after == expected_pairs, (
        "a survivor's destination must be the one the scheduler paired with it")
    assert len(after) == len(before[0]) - len(owners)
    # The RAM entries are untouched, on both sides and in both positions.
    assert 400 in metadata.load_cpu_blocks and 70_000 in metadata.load_gpu_blocks
    assert 900 in metadata.load_cpu_blocks and 999_000 in metadata.load_gpu_blocks
    assert after[0] == before_pairs[0], "the leading RAM pair moved"
    assert after[-1] == before_pairs[-1], "the trailing RAM pair moved"
    # And the bytes really landed, because a prune runs AFTER the apply, not instead.
    assert disk_restore.counters()["reads_applied"] == len(owners)
    state = disk_restore.prune_state()
    assert state["entries"] == len(owners)
    assert state["skipped_failed_entries"] == 0
    assert state["last"]["entries_in_step"] == len(owners)
    assert state["last"]["pruned"] == len(owners)
    # NOTHING was left for the fork to copy here: the RAM entries are still submitted.
    assert state["emptied_steps"] == 0


def test_the_prune_reports_the_groups_it_removed_by_name(pruning, delivered, armed):
    """`pruned_by_group` is the group id READ from the job, never inferred.

    With the live group set an entry's group is the fact the whole Gate-B argument
    rests on, so the published per-group count is the visible proof it came from the
    scheduler's identity and not from an offset.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager)
    worker = worker_for(metadata)
    assert disk_restore.worker_before_load(worker)["ok"] is True
    by_group = disk_restore.prune_state()["last"]["groups"]
    assert {int(key) for key in by_group} == {group for group, _, _ in owners}
    assert sum(by_group.values()) == len(owners)


# --- 4. GATE B: the binding travels by list, not by value ---------------------

def test_a_cacheable_group_at_a_different_block_size_cannot_reach_the_prune(
        armed):
    """GATE B, THE STRUCTURAL HALF: a differently-sized group can never be OFFERED.

    The fork pairs group g's entry i with
    `block_ids_by_group[g][num_computed // group_block_sizes[g] + i]`, so a group whose
    `group_block_sizes[g]` differs from the scheduler's block size has a DIFFERENT
    offset from every other group - and the offer gate refuses such a model outright
    (`_group_serviceability`: "a slot for it would land at the wrong GPU offset"). That
    is the reason the prune never has to do group arithmetic at all: every group whose
    bytes can ever appear in `load_cpu_blocks` is at the scheduler's block size, and the
    rest are refused before a slot exists.

    This is the case the brief asked to be proved, proved where it is actually decided -
    and it is the reason the test below uses the box's real sizes instead of a
    contrived 1,600-token boundary group: this code cannot build one.
    """
    sizes = [1600, 1600, 1600, 1600, 8, BLOCK]
    fill(armed, seeds=[2, 3], groups=HYBRID_CACHEABLE)
    manager = hybrid(sizes, reqs_to_load={"req-1": object()})
    verdict = offer(manager, Req(request_id="req-1", blocks=4), BLOCK)
    assert verdict == (0, False), (
        "a boundary group at 1,600 tokens while the scheduler runs 3,200 was ACCEPTED: "
        "the gate that keeps Gate B unreachable has been relaxed")
    refusal = disk_restore.counters()["last_refusal"]
    assert refusal.startswith("incomplete_group_set"), refusal
    serviceable = disk_restore.counters()["group_serviceability"]
    assert serviceable["complete"] is False
    assert serviceable["blocked_group_ids"] == [0, 1, 2, 3]
    assert serviceable["blocked_kinds"]["0"] == "geometry"
    # The long form of WHY, per group, is where the sentence about offsets lives: the
    # refusal line itself is clipped, and the arithmetic is not.
    assert "wrong GPU offset" in serviceable["blocked_why"]["0"], \
        serviceable["blocked_why"]
    # No slot was reserved, so no destination exists, so there is nothing to prune.
    assert manager._pending_cpu_hits.get("req-1") is None


def test_a_boundary_group_keeps_its_own_entries_and_not_the_leading_ones(
        pruning, delivered, armed):
    """GATE B, THE OPERATIONAL HALF, on the live geometry (a Qwen3-Next-shaped model).

    Groups 0-3 are `boundary_state`: a hit run of B blocks contributes
    `[null] * (B - 1) + [real]`, ONE entry at index B-1, because the boundary state of a
    run lives at its END. Group 5 is `positional` and contributes all B entries, at
    indices 0..B-1. The fork skips the nulls while KEEPING THEIR INDICES, so the two
    kinds of group contribute a DIFFERENT number of entries at a DIFFERENT index within
    the group - and the flattened list position of a boundary entry is therefore not the
    positional group's position.

    Two assertions, and the second is what stops the test passing for a broken rule:
    (1) the pruned destinations are exactly the pairs the fork's own per-group formula
    produced, per group; (2) the rule this prune must NOT be - "each cacheable group
    contributes B leading entries" - names a set of GPU destinations that is not this
    step's, including ids that do not appear in the step at all.
    """
    spans = 2                      # the run the two seeded spans make
    fill(armed, seeds=[2, 3], groups=HYBRID_CACHEABLE)
    manager = hybrid(reqs_to_load={"req-1": object()})
    # The offer adopts as many spans as the store holds contiguously from `computed`,
    # which the two seeded spans give: TWO. A one-span run has no nulls in a boundary
    # group and would make the shape assertions below vacuous, so B is read back from
    # the merged tuple rather than assumed.
    assert offer(manager, Req(request_id="req-1", blocks=4), BLOCK) == (2 * BLOCK, True)
    metadata, owners, disk_range = ship(manager, computed=BLOCK)
    before = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    # THE SHAPE ITSELF, read off the harness: one entry per boundary group at index
    # B-1, and B entries for the positional group at indices 0..B-1.
    per_group: dict = {}
    for group, index_in_group, _offset in owners:
        per_group.setdefault(group, []).append(index_in_group)
    assert {group: per_group[group] for group in GDN} == {group: [spans - 1]
                                                          for group in GDN}
    assert per_group[ATTENTION] == list(range(spans))
    # (1) EVERY PAIR IS THE FORK'S OWN, computed with that group's own size.
    for index, (group, index_in_group, _offset) in enumerate(owners):
        _cpu_id, gpu_id = before[disk_range[0] + index]
        size = manager.group_block_sizes[group]
        assert gpu_id == 1000 + group * 100 + (BLOCK // size) + index_in_group, (
            f"group {group} entry {index_in_group} paired at {gpu_id}, not at its own "
            "offset: the harness stopped matching manager.py")
    worker = worker_for(metadata)
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True and report["pruned"] == len(owners)
    after = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    pruned = [index for index in range(len(before)) if before[index] not in after]
    assert pruned == list(range(disk_range[0], disk_range[1]))
    # THE DESTINATIONS REMOVED, recomputed from the fork's per-group formula rather than
    # from the harness, so this line fails if the prune ever removes the wrong ones.
    expected_gpu = [1000 + group * 100 + (BLOCK // manager.group_block_sizes[group]) + i
                    for group, i, _offset in owners]
    assert [before[index][1] for index in pruned] == expected_gpu, (
        "the destinations removed are not the ones the fork paired with the restored "
        "blocks")
    assert after == [] , "the whole step was disk entries, so nothing should remain"
    # (2) THE COUNTERFACTUAL. A prune that assumed B leading entries per cacheable group
    # would name `len(GDN) * spans + spans` = 10 destinations, and the boundary groups'
    # LEADING entry (index 0, the run's START) - GPU ids 1001/1101/1201/1301, none of
    # which is in this step. It would remove nothing it should and leave everything it
    # shouldn't.
    uniform_rule = len(GDN) * spans + spans
    assert len(owners) != uniform_rule, "the fixture no longer distinguishes the rules"
    leading_ids = {1000 + group * 100 + (BLOCK // manager.group_block_sizes[group])
                   for group in GDN}
    present = {gpu for _cpu, gpu in before}
    assert leading_ids.isdisjoint(present), (
        "the leading boundary entries are in this step, so this fixture cannot show "
        "that a leading-entry rule is wrong")
    pruned_ids = {before[index][1] for index in pruned}
    assert pruned_ids == set(expected_gpu)
    assert pruned_ids.isdisjoint(leading_ids)
    # The published per-group counts are the group ids READ from the jobs.
    assert disk_restore.prune_state()["last"]["groups"] == {
        str(group): len(per_group[group]) for group in sorted(per_group)}


def test_the_pair_function_never_looks_at_a_block_size(pruning):
    """`prune_load_pairs` is geometry-blind: an expectation at ANY offset works.

    The one place that computes an index from a block size is the place Gate B breaks,
    so this call pins the function's independence from it: two groups whose expected GPU
    ids are 1,000 apart, at list positions 1 apart, prune together or not at all. It
    also pins the STABLE KEY SET: a group the prune touched and a group it did not both
    appear in both maps, so a reader can tell "group 0 lost nothing" from "group 0 was
    not reported".
    """
    cpu = [10, 11, 12, 13]
    gpu = [20, 1020, 1040, 23]
    expected = {0: (10, 20), 1: (11, 1020), 2: (12, 1040), 3: (13, 23)}
    groups = {0: 0, 1: 1, 2: 1, 3: 0}
    verdict = disk_restore.prune_load_pairs(cpu, gpu, [1, 2], None, groups=groups,
                                            expected=expected)
    assert verdict["ok"] is True
    # THE STEP CARRIES ONE BLOCK PER GROUP (`take_per_group=1`), and the two
    # destinations
    assert verdict["pruned_by_group"] == {"0": 0, "1": 2}
    assert verdict["group_deltas"] == {"0": 0, "1": 0}


def test_the_prune_never_derives_an_index_from_tokens(pruning, delivered, armed):
    """A step whose GPU ids are meaningless still prunes the RIGHT entries.

    The identity of a disk entry is the path the scheduler named, and nothing else. If
    the prune ever computed an index from `blocks`, `block_size` or an id range, this
    step - where the GPU ids are deliberately unrelated to any block numbering - would
    prune the wrong ones or refuse, and never quietly succeed.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, disk_range = ship(manager)
    metadata.load_gpu_blocks = [7_777 * (index + 1)
                                for index in range(len(metadata.load_cpu_blocks))]
    worker = worker_for(metadata)
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True
    assert report["pruned"] == len(owners)
    assert len(metadata.load_cpu_blocks) == 0
    assert len(metadata.load_gpu_blocks) == 0


# --- 5. adopted, not offered --------------------------------------------------

def test_an_offer_the_scheduler_truncated_prunes_only_what_arrived(
        pruning, delivered, armed):
    """OFFERED 3 BLOCKS, THE STEP CARRIES 1: the other two destinations stay.

    The fork takes `cdiv(num_external_tokens, group_block_sizes[g])` LEADING entries per
    group, and `num_external_tokens` is the scheduler's decision - the adoption floor
    that runs on this box discards most short offers, so "offered N, shipped M" is the
    NORMAL shape here. A prune sized by the offer would delete destinations whose bytes
    never arrived, which is strictly worse than today: the engine would copy nothing
    into them and still report them loaded.
    """
    fill(armed, blocks=3)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=3)) == (3 * BLOCK, True)
    assert disk_restore.counters()["blocks_offered"] == 3
    # THE STEP CARRIES ONE BLOCK PER GROUP (`take_per_group=1`), and the two
    # destinations the scheduler did NOT take for those tokens are in the step as
    # host-RAM entries: the fork owns them, their bytes are not ours, and they must
    # survive untouched.
    metadata, owners, disk_range = ship(
        manager, take_per_group=1,
        ram=[(410 + index, 70_000 + index) for index in range(len(base.CACHEABLE))])
    before = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    assert len(owners) == len(base.CACHEABLE), "one entry per group reached the step"
    worker = worker_for(metadata)
    report = disk_restore.worker_before_load(worker)
    assert report["pruned"] == len(owners)
    after = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    assert after == [before[index] for index in range(len(before))
                     if not disk_range[0] <= index < disk_range[1]]
    # The not-adopted destinations are still there, WITH THE PARTNERS THEY ARRIVED WITH.
    assert dict(after) == {410 + index: 70_000 + index
                           for index in range(len(base.CACHEABLE))}
    state = disk_restore.prune_state()
    assert state["entries"] == len(owners)
    assert state["last"]["entries_in_step"] == len(owners)
    assert state["last"]["jobs_bound"] == len(owners)
    # The invariant the counters publish, in the order the pipeline loses blocks.
    assert state["last"]["pruned"] <= state["last"]["entries_in_step"] \
        <= state["last"]["jobs_bound"]
    assert disk_restore.counters()["blocks_offered"] == 3
    # The scheduler took one block per group of the three it offered, so the prune
    # removed one group's worth of destinations and left the rest of the offer alone.
    assert state["entries"] < disk_restore.counters()["blocks_offered"] \
        * len(base.CACHEABLE)


# --- 6. a failed restore keeps its range -------------------------------------

def _one_failed_read_step(root, request_id="req-1", event=11):
    """One step with ONE unreadable entry, run to the end of the apply point.

    Returns the report, the GPU ids the module will hand the engine as load errors, the
    surviving pairs, the pairs as they arrived, the fork's owners and the disk range.
    The file is deleted BEFORE the bind - that is the only moment a read can fail for a
    missing file, and the reader thread has already opened it once a batch is settling.
    """
    fill(root, seeds=(1, 2), groups=base.CACHEABLE)
    manager = Manager(reqs_to_load={request_id: object()})
    assert offer(manager, Req(request_id=request_id,
                              blocks=2)) == (2 * BLOCK, True)
    metadata, owners, disk_range = ship(manager, request_id=request_id, event=event,
                                        ram=[(400, 70_000)])
    before = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    # Fail the SECOND entry of the SECOND group: the kept entries sit on both sides of
    # it, so this is neither a suffix nor a prefix.
    vanish(root, seeds=(2,), groups=(owners[1][0],))
    worker = worker_for(metadata)
    report = disk_restore.worker_before_load(worker)
    invalid = set(disk_restore.take_invalid_block_ids())
    after = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    return report, invalid, after, before, owners, disk_range


def test_a_failed_restore_keeps_its_range_out_of_the_prune(pruning, delivered, armed):
    """THE RECOMPUTE PROPERTY, with the flag ON, asserted where it actually lives.

    WHERE THE PROPERTY LIVES, because the naive reading of "never prune a failed range"
    is wrong about this code base and the difference matters. Every release through
    0.2.48 REMOVED a failed entry from both load lists (`_worker_before_load`'s drop
    loop, `if index in drop_set: continue`) - the bytes never arrived, so copying the
    slot would move uninitialised memory into a KV window - and made the engine
    recompute those tokens by reporting the entry's GPU block through
    `take_invalid_block_ids()` (`_remember_invalid`, on the drop path). So "the range is
    kept" cannot mean "its pair stays in the lists": it means the failed entry is not
    part of the PRUNE'S population, and the load-error report the engine recomputes from
    is exactly the one the flag-off boot would have produced. Both halves are asserted,
    and the second one is asserted against a flag-off run of the same scenario below.
    """
    report, invalid, after, before, owners, disk_range = _one_failed_read_step(armed)
    failed_pair = before[disk_range[0] + 1]
    assert report["ok"] is True
    assert report["dropped"] == 1, "one read failed"
    # (1) THE FAILED ENTRY IS NOT IN THE PRUNE'S SET: one entry of six was refused, and
    # the counter says so by name rather than by an absent exception.
    assert report["pruned"] == len(owners) - 1
    state = disk_restore.prune_state()
    assert state["skipped_failed_entries"] == 1
    assert state["entries"] == len(owners) - 1
    assert state["last"]["skipped_failed"] == 1
    assert state["last"]["also_dropped_failed"] == 1
    assert sorted(state["last"]["pruned_indices"]) == sorted(
        index for index in range(disk_range[0], disk_range[1])
        if index != disk_range[0] + 1)
    assert state["last"]["removed_indices_total"] == report["dropped"] + \
        report["pruned"]
    # (2) THE RECOMPUTE ROUTE IS INTACT: the failed entry's GPU block is reported, so
    # `get_block_ids_with_load_errors` covers those tokens and the engine recomputes.
    assert failed_pair[1] in invalid
    # (3) THE SURVIVORS KEPT THEIR PARTNERS - here the single host-RAM entry, which this
    # module must never touch and must never shift.
    assert after == [(400, 70_000)]
    assert report["submitted"] is True


def test_the_engine_is_told_to_recompute_the_same_blocks_flag_off_and_flag_on(
        armed, delivered, monkeypatch):
    """THE NON-NEGOTIABLE, AS A COMPARISON: the flag may not change the recompute set.

    A property asserted only on the new path can be satisfied by a new path that broke
    the property. This runs the SAME scenario twice - the shipped off posture, then the
    prune - and requires the load-error GPU ids, the drop count and the applied reads to
    agree exactly. It is this test that fails if the prune ever decides to remove a
    failed entry's pair instead of leaving that population to the failure path.
    """
    report_off, invalid_off, after_off, _before, _owners, _range = \
        _one_failed_read_step(armed, request_id="req-off")
    assert report_off["pruned"] == 0, "this run has the flag off"
    monkeypatch.setenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, "1")
    disk_restore._state["prune_switch"] = None
    report_on, invalid_on, after_on, before, owners, disk_range = _one_failed_read_step(
        armed, request_id="req-on", event=12)
    assert report_on["pruned"] == len(owners) - 1, "this run has the flag on"
    # THE INVARIANT: what the engine is told to recompute is not the flag's business.
    # Same GPU block, same drop count, same applied reads.
    assert invalid_on == invalid_off
    assert report_on["dropped"] == report_off["dropped"] == 1
    assert report_on["applied"] == report_off["applied"]
    assert before[disk_range[0] + 1][1] in invalid_on
    # What DOES differ is exactly the prune: the entries whose bytes arrived left the
    # step in the flagged run and stayed in the shipped one, and the host-RAM entry that
    # this module must never touch is in BOTH.
    assert (400, 70_000) in after_off and (400, 70_000) in after_on
    assert len(after_off) > len(after_on)
    assert len(after_on) == len(after_off) - report_on["pruned"]



def test_a_step_where_every_read_failed_prunes_nothing_but_still_reports(
        pruning, armed):
    """All four failure shapes leave `prune_entries` at 0 and say WHY.

    `nothing_adopted` is a refusal NAME, not an error: with every read failed the right
    answer is the step the scheduler shipped, and the counter records that the prune was
    asked and correctly declined.
    """
    fill(armed, seeds=(1,), groups=base.CACHEABLE)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    metadata, owners, _ = ship(manager)
    # Every file the step would read is gone before the reader thread starts.
    vanish(armed, seeds=(1,), groups=tuple(sorted({group for group, _, _ in owners})))
    worker = Worker(regions={"g0": Region(slots=512)})
    worker._pending_load_event_indices.add(metadata.load_event)
    assert bind(worker, metadata)["ok"] is True
    settle(disk_restore._batch_id(metadata.load_event))
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True
    assert report["dropped"] == len(owners)
    assert report["pruned"] == 0
    assert disk_restore.prune_state()["entries"] == 0
    assert disk_restore.prune_state()["refused"]["nothing_adopted"] == 1
    assert metadata.load_cpu_blocks == [] and metadata.load_gpu_blocks == []


# --- 7. shape and log discipline ---------------------------------------------

def test_a_refused_prune_publishes_the_same_key_set_as_an_accepted_one(pruning,
                                                                        armed):
    """STABLE KEYS ACROSS STATES, the discipline every other section follows.

    A reader diffing two payloads must be able to tell "this did not happen" from "this
    build does not report it". The refused answer, the accepted answer and the
    never-consulted answer are all compared to the same key set.
    """
    refused = disk_restore.prune_load_pairs([1, 2, 3], [4, 5, 6], [0], [])
    accepted = disk_restore.prune_load_pairs([1, 2, 3], [4, 5, 6], [0])
    assert refused["ok"] is False and accepted["ok"] is True
    assert set(refused) == set(accepted)
    # And the module's published view has the same keys whether or not a step ran.
    quiet = disk_restore.prune_state()
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager)
    disk_restore.worker_before_load(worker_for(metadata))
    busy = disk_restore.prune_state()
    assert set(quiet) == set(busy)
    assert set(quiet["refused"]) == set(busy["refused"]) \
        == set(disk_restore.PRUNE_REFUSALS)
    assert set(quiet["units"]) >= {"entries", "refused", "unwitnessed_entries"}
    assert set(busy["last"]) == set(quiet["last"]) if quiet["last"] else True
    counters = disk_restore.counters()
    assert set(counters["prune_refused"]) == set(disk_restore.PRUNE_REFUSALS)
    for key in ("prune_switch", "prune_steps", "prune_entries", "prune_refused",
                "prune_refusals_total", "prune_skipped_failed_entries",
                "prune_unwitnessed_entries", "prune_emptied_steps", "last_prune"):
        assert key in counters, key
        assert str(counters["units"].get(key, "")).strip(), key
    # The section carries the view, so the verb and the snapshot agree.
    assert set(disk_restore.section()["load_prune"]) == set(busy)
    assert disk_restore.section()["load_prune_reasons"] == list(
        disk_restore.PRUNE_REFUSALS)


def test_the_prune_flag_alone_prunes_nothing_and_names_the_refusal(pruning, armed):
    """STEP 3'S DEFAULT, AND THE TEST THE OLD HAZARD TEST USED TO BE.

    THIS TEST REPLACES `test_the_hazard_is_counted_and_said_once_per_episode`, and the
    replacement is the point rather than a side effect. That test asserted that a pruned
    entry with no plugin-side write behind it was COUNTED (`unwitnessed_entries ==
    len(owners)`) and logged once per episode - it priced a hazard this flag walked into
    with its eyes open, because `disk_gpu_write` sampled one entry in 64 and the fork's
    copy was the only delivery. Step 3 removed the hazard instead of pricing it: the
    prune's population is filtered by MEMBERSHIP in the writer's enqueued set, so the
    scenario that test built (the prune flag on, no writer) now prunes NOTHING.
    Asserting
    the old numbers here would assert a behaviour the code no longer has.

    So the flag-on-no-writer step is asserted to be INERT, and to say why in its own
    name: `not_delivered`, not `nothing_adopted`. The distinction is what an operator
    reads - "this step had three adopted, successfully-read entries and the writer did
    not cover one of them" - and it is why the fork keeps every destination.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager)
    before = (list(metadata.load_cpu_blocks), list(metadata.load_gpu_blocks))
    worker = worker_for(metadata)
    report = disk_restore.worker_before_load(worker)
    assert report["pruned"] == 0, "nothing was delivered, so nothing may be pruned"
    assert report["submitted"] is True, "the fork still copies all of them"
    assert (list(metadata.load_cpu_blocks), list(metadata.load_gpu_blocks)) == before, \
        "the step's lists are exactly what the scheduler shipped"
    state = disk_restore.prune_state()
    assert state["entries"] == 0
    # THE REFUSAL IS NAMED, and it is the new name and not the old one. A step with real
    # adopted work and no delivery is a DIFFERENT operational story from a step with no
    # disk work, and `_prune_refused` silently folds an unknown `reason_kind` into
    # `nothing_adopted` - which is how a dead writer would hide behind a healthy zero.
    assert state["refused"]["not_delivered"] == 1, state["refused"]
    assert state["refused"]["nothing_adopted"] == 0
    assert state["last"]["reason_kind"] == "not_delivered"
    # THE INVARIANT, at this level: nothing was delivered, nothing was pruned, so the
    # tripwire cannot have moved.
    assert state["unwitnessed_entries"] == 0
    # AND IT SAYS SO, in the refusal's own sentence, naming the count it did not prune.
    assert "NONE was covered by a plugin-side GPU write" in state["last"]["reason"]
    assert state["last"]["adopted_entries"] == len(owners)
    assert state["last"]["uncovered_entries"] == len(owners)
    assert state["last"]["covered_entries"] == 0


def test_the_unwitnessed_hazard_is_now_unreachable_and_still_trips_a_broken_gate(
        pruning, armed, monkeypatch, caplog):
    """THE STRUCTURAL CLAIM, TESTED BOTH WAYS: `unwitnessed_entries` cannot move, and it
    STILL moves if the gate is broken.

    A test that only asserts `== 0` proves nothing, because 0 is also what a counter
    says
    when the code that was supposed to move it was deleted. So this runs the maximum-
    damage scenario the previous release walked into (prune on, entries adopted and
    read, a writer covering NONE of them) twice: once as shipped, where the counter must
    stay 0, and once with `_delivered` switched to "always covered" - the narrowest
    possible edit, the one line that is the whole safety argument - where the counter
    MUST move, at ERROR
    level.

    THE SECOND READING IS WHAT MAKES THE FIRST ONE MEAN ANYTHING. Without it the
    assertion would be indistinguishable from a tripwire nothing writes, which is the
    recurring bug
    class in this package: a number published in `units` and never incremented. It also
    depends on the tripwire reading `covered_ids` DIRECTLY instead of calling the same
    predicate - a tripwire that asks the predicate whether it passed would report 0 for
    a
    gate it was asked to bypass, and the mutation would pass as clean.
    """
    def step(event, request_id):
        manager = Manager(reqs_to_load={request_id: object()})
        assert offer(manager, Req(request_id=request_id, blocks=1)) == (BLOCK, True)
        metadata, owners, _ = ship(manager, request_id=request_id, event=event)
        disk_restore.worker_before_load(worker_for(metadata))
        return owners

    fill(armed, blocks=1)
    step(11, "req-1")
    state = disk_restore.prune_state()
    assert state["entries"] == 0, "nothing delivered, so nothing pruned"
    assert state["unwitnessed_entries"] == 0, "the gate holds, as it must"

    # THE MUTATION. Everything else - the population, the geometry, the counters, the
    # tripwire - is the shipped code, so the second reading tests the tripwire and not a
    # fixture. `_delivered` is the ONE predicate the gate calls, and the tripwire does
    # not.
    monkeypatch.setattr(disk_restore, "_delivered", lambda index, covered: True)
    disk_restore._notices.clear()          # a fresh episode, deliberately
    broken = step(12, "req-2")
    assert broken, "the scenario must still have adopted entries to prune"
    state2 = disk_restore.prune_state()
    assert state2["entries"] > 0, "the broken gate pruned, as a broken gate must"
    assert state2["unwitnessed_entries"] == state2["entries"], \
        "the tripwire did not count what the broken gate pruned"
    errors = [line for line in caplog.records
              if "NO plugin-side GPU write" in str(line.message)
              and line.levelname == "ERROR"]
    assert errors, "a broken gate must be said at ERROR, not warned about"
    # ONE LINE, NOT ONE PER STEP: the counter accumulates, the log does not.
    step(13, "req-3")
    again = disk_restore.prune_state()
    assert again["unwitnessed_entries"] > state2["unwitnessed_entries"], \
        "the counter counts entries, not episodes"
    errors2 = [line for line in caplog.records
               if "NO plugin-side GPU write" in str(line.message)
               and line.levelname == "ERROR"]
    assert len(errors2) == len(errors), "the line repeated per step"


def test_a_step_left_with_nothing_to_copy_is_counted(pruning, delivered, armed):
    """`emptied_steps`: the fork is left with no submission, so the event is released.

    This is the sharpest shape of the flag - every entry the fork would have copied is
    gone - and the phantom rule still has to hold: an index left in
    `_pending_load_event_indices` becomes reportable as soon as a LATER event completes,
    so `release_event` must take it out and report the requests itself.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager)
    worker = worker_for(metadata)
    report = disk_restore.worker_before_load(worker)
    assert report["pruned"] == len(owners)
    assert report["submitted"] is False
    assert metadata.load_cpu_blocks == [] and metadata.load_gpu_blocks == []
    assert worker._pending_load_event_indices == set(), \
        "a never-submitted load event must not stay pending: that is the phantom"
    assert report["released"] == 1
    state = disk_restore.prune_state()
    assert state["emptied_steps"] == 1
    assert disk_restore.counters()["phantom_guard_trips"] == 0


def test_a_fully_covered_step_keeps_its_last_entry_so_the_event_is_recorded(
        pruning, delivered, armed, monkeypatch):
    """THE EMPTIED-STEP GUARD, and the race it closes.

    THE HAZARD, stated because it is invisible in every other test in this file. The
    fork's `start_load_kv` submits a copy ONLY when `load_cpu_blocks` is non-empty, so a
    step the prune emptied records NO load event at all. The apply point then takes its
    existing release path - which is correct for the phantom bug it was written for, and
    which reports the requests finished IMMEDIATELY. And there is no ordering edge from
    the compute stream to `load_stream` anywhere in the fork's load path: loads never
    pass
    a `wait_event` (only stores do), so the next forward can read those KV windows while
    the plugin's own copies are still in flight. Nothing raises, nothing logs, the
    answer
    is wrong.

    The guard leaves the LAST covered entry in both lists. That entry makes the fork
    submit, the fork's event is recorded on the SAME stream AFTER the plugin's copies
    (CUDA streams are FIFO), and the event therefore witnesses them. The fork rewrites
    bytes the plugin just wrote with the same bytes, which is what makes the extra copy
    harmless rather than merely cheap.

    WHY THE `delivered` FIXTURE IS NOT ENOUGH: the guard asks the writer whether IT is
    delivering, and the stand-in fixture only reports coverage. So this test raises the
    writer's own delivery flag for real, which is the only way to reach the guarded
    state. THE GUARD IS ONLY FOR THAT STATE: the test next door runs the same
    fully-covered step with delivery NOT in force and still asserts the emptied-step
    release, because with no
    plugin-side copy there is nothing for the event to witness and the old behaviour is
    correct.
    """
    # BOTH writer flags: delivery mode is a LAYER on the writer, and with the writer off
    # `effective_delivery_mode()` is False - so the guard would not apply and this test
    # would be asserting the pre-0.2.52 emptied-step behaviour it is written to replace.
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    disk_gpu_write._state["env_resolved"] = False
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    disk_gpu_write._state["delivery_resolved"] = False
    assert disk_gpu_write.effective_delivery_mode() is True, \
        "the guarded state requires BOTH the writer and the delivery flag"
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager)
    before = list(zip(metadata.load_cpu_blocks, metadata.load_gpu_blocks,
                      strict=True))
    worker = worker_for(metadata)
    report = disk_restore.worker_before_load(worker)
    # ONE ENTRY FEWER PRUNED THAN THERE WERE COVERED, and the step is NOT empty.
    assert report["pruned"] == len(owners) - 1
    assert len(metadata.load_cpu_blocks) == 1 and len(metadata.load_gpu_blocks) == 1
    assert (metadata.load_cpu_blocks[0], metadata.load_gpu_blocks[0]) in before, \
        "the kept entry must be a pair the scheduler actually formed"
    assert report["submitted"] is True, "the fork must still copy, that is the point"
    assert report["released"] == 0, "the event is recorded, so nothing is released"
    state = disk_restore.prune_state()
    assert state["emptied_steps"] == 0, "the guard exists to keep this at zero"
    assert state["kept_last_entries"] == 1
    assert state["unwitnessed_entries"] == 0
    # THE KEPT PAIR IS STILL A PAIR: both lists lost the SAME indices, so the fork's
    # positional pairing cannot shift for the entry it does copy. `kept == 1` and that
    # one pair existing in the scheduler's own ordering is the whole claim.
    assert len(metadata.load_cpu_blocks) == len(metadata.load_gpu_blocks) == 1
    assert (metadata.load_cpu_blocks[0], metadata.load_gpu_blocks[0]) in before
    assert disk_restore.counters()["phantom_guard_trips"] == 0


def test_the_worker_report_and_the_snapshot_both_carry_the_numbers(
        pruning, delivered, armed):
    """A rank file that does not mention the prune is a rank file that hides it.

    Only a WORKER prunes (it is the process that applies a restore), so the flag's
    counters belong in the rank section as well as in the verb's payload - and the
    section still fits the byte budget the seam enforces.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager)
    disk_restore.worker_before_load(worker_for(metadata))
    report = disk_restore.worker_report()
    assert report.get("truncated") is not True, report
    assert report["prune"]["on"] is True
    assert report["prune"]["entries"] == len(owners)
    assert report["prune"]["emptied"] == 1
    assert disk_restore.prune_state()["entries"] == len(owners)
    assert disk_restore.counters()["prune_entries"] == len(owners)


def test_the_verifier_and_the_direct_writer_are_still_untouched_by_the_flag(
        pruning, delivered, armed):
    """The prune is a THIRD thing at the apply point, not a change to the two witnesses.

    `disk_restore_verify` and `disk_gpu_write` both return early on their own
    enablement test; a prune that made either of them run (or stopped them running)
    would put file reads and device copies on a boot that did not ask for them.
    """
    assert disk_restore_verify.effective_enabled() is False
    assert disk_gpu_write.effective_enabled() is False
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1))
    metadata, owners, _ = ship(manager)
    disk_restore.worker_before_load(worker_for(metadata))
    assert disk_gpu_write.counters()["applies_seen"] == 0
    assert disk_restore_verify.counters()["applies_seen"] == 0
    assert disk_restore.prune_state()["entries"] == len(owners)
