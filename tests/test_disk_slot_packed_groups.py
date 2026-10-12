r"""A group whose block ids are not pool block ids must be REFUSED, by name (0.2.71).

WHY THIS FILE EXISTS, AND WHY IT WAS WRITTEN BEFORE ANY OF IT COULD BE MEASURED.
The v2.5 side-cache layout (fork patch 0088, `kv_cache_utils._apply_side_cache_layout`)
changes two things a connector was never meant to see, and it gates itself off for
connectors precisely because of it - `_side_cache_layout_enabled` returns False whenever
`kv_transfer_config is not None`. We are preparing to relax that gate, so a folded boot
with OUR connector in it is a boot where these two shapes reach the tier for the first
time. This file is the prerequisite for that relaxation, not a follow-up to it.

THE TWO SHAPES, AND WHY EACH IS A MIS-ADDRESS RATHER THAN AN ERROR.

1. SLOT PACKING. `kv_cache_utils._choose_ring_slots` picks an S and sets
   `KVCacheGroupSpec.slots_per_block` / `KVCacheTensor.slots_per_block`;
   `single_type_kv_cache_manager.RingSlotPool` then hands out SLOTS of shared pool
   blocks and a slot's id is `pool_block_id * slots_per_block + slot`, while
   `KVCacheConfig.num_blocks_of` returns `num_blocks * tensor.slots_per_block`. The
   fork's own scheduler states the consequence when it walks managers to zero blocks
   (`v1/core/sched/scheduler.py`, patch 0088):

       if getattr(mgr, "slot_pool", None) is not None:
           # v2.5 slot-packed ring: its block ids are slot ids, not pool
           # block ids, and its pages are never zeroed.
           continue

   Every GPU address this tier forms is `base + block_id * stride` (see
   `blockdump.py`'s own description of the address it forms, and
   `disk_restore_verify`'s `gpu_window[gpu_block_id]`). Against a slot-packed group that
   arithmetic lands in somebody else's slot: the file is well-named, the read succeeds,
   every counter says `applied`, and the model continues from bytes that are not its
   state. That is this project's signature incident class - plausible output, not an
   error, the `Register Register Register...` of 2026-10-06 00:08 - so the answer is a
   refusal with its OWN counter name (`refused.slot_packed_group`), never a silent skip:
   a group dropped quietly from the eligible set is a claim made without it.

2. MIXED TP REPLICATION. The PLE fold merges a TP-replicated Mamba group into a sharded
   one and builds the container DIRECTLY, with the fork's comment saying
   `is_uniform_type` rejects mixed `tp_replicated` "on purpose (NIXL), and connectors
   never reach this path". Ours does reach it. Such a group has no single page shape,
   and taking `first_spec`'s values for the whole group prices thirteen layers by one of
   them - a number that reads as a plan. We cannot price it, so we refuse it
   (`refused.mixed_replication_group`), the same move `incomplete_group_set` already
   makes for a group with no implementable contract.

WHAT THESE TESTS DO NOT DO. They do not claim a folded boot is correct, or that a
slot-packed ring group could ever be served from disk. They pin that when the engine
SAYS one of these two things about a group, the tier names it and offers nothing - and
that a boot where the engine says NEITHER (every boot up to today) is byte-for-byte
unchanged, refusal counter names included.
"""

from __future__ import annotations

import logging

import fakes_debug
import pytest
import test_disk_restore as base

from kvtier_dev_plugin import disk_layout, disk_restore, disk_store

armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

BLOCK = base.BLOCK
Manager = base.Manager
Req = base.Req
fill = base.fill
offer = base.offer

# The live hybrid geometry with the QSA ring group (index 4) made PREFIX-CACHEABLE, the
# only way a slot-packed group can reach the gate at all: the fork packs the ring, the
# ring is `CircularBufferSpec` and its manager gets the `slot_pool`.
PACKED_GROUP = 4


def packed(*, group=PACKED_GROUP, slots=800, signal="slot_pool",
           cacheable=(0, 1, 2, 3, 4, 5), sizes=None, mode="align"):
    """A manager whose ring group answers the fork's slot-packing signals.

    `signal` picks WHICH of the fork's two signals the engine presents:
    `"slot_pool"` (the `RingSlotPool` the coordinator installs via
    `enable_slot_packing`, the very test the fork's scheduler skips on) or
    `"slots_per_block"` (the `KVCacheGroupSpec`/`KVCacheTensor` field). Both must work:
    one is the manager's answer, the other the config's, and a boot can plausibly carry
    one where the other is not reachable from the scheduler's view.
    """
    sizes = tuple(sizes or (BLOCK, BLOCK, BLOCK, BLOCK, BLOCK, BLOCK))
    manager = Manager(group_sizes=sizes, cacheable=cacheable)
    items = []
    for index in range(len(sizes)):
        if index == group:
            spec = fakes_debug.circular_buffer_spec(block_size=sizes[index])
            spec.prefix_cacheable = True
            item = fakes_debug.manager_item(index, spec, stable=True)
            if signal == "slot_pool":
                item.slot_pool = fakes_debug.Obj(
                    name="RingSlotPool", slots_per_block=slots)
            else:
                item.slots_per_block = slots
        elif index < 4:
            item = fakes_debug.manager_item(
                index,
                fakes_debug.mamba_spec(block_size=sizes[index],
                                       mamba_cache_mode=mode),
                stable=False, mamba_cache_mode=mode)
        else:
            item = fakes_debug.manager_item(
                index, fakes_debug.attention_spec(block_size=sizes[index]),
                stable=True)
        items.append(item)
    manager.cpu_coordinator = fakes_debug.coordinator(items)
    return manager


def merged(*, tp_replicated=(False, True), accessor="kv_cache_specs",
           cacheable=(0, 1, 2, 3, 5), sizes=None, mode="align"):
    """A manager whose GDN group is a PLE-fold merge: layers, mixed replication.

    The container is built the way `_fold_replicated_mamba_groups` builds it: one
    `UniformTypeKVCacheSpecs` holding per-layer `MambaSpec`s whose `tp_replicated`
    disagrees. `accessor="first_spec"` is the degraded container that can only show one
    layer - silence, which must NOT be read as a verdict either way.
    """
    sizes = tuple(sizes or (BLOCK, BLOCK, BLOCK, BLOCK, BLOCK, BLOCK))
    manager = Manager(group_sizes=sizes, cacheable=cacheable)
    items = []
    for index in range(len(sizes)):
        if index in (0, 1, 2, 3):
            spec = fakes_debug.mamba_spec(block_size=sizes[index],
                                          mamba_cache_mode=mode)
            if accessor == "kv_cache_specs":
                layers = {}
                for offset, flag in enumerate(tp_replicated):
                    layer = fakes_debug.mamba_spec(block_size=sizes[index],
                                                   mamba_cache_mode=mode)
                    layer.tp_replicated = flag
                    layers[f"layer{offset}"] = layer
                spec.kv_cache_specs = layers
            else:
                only = fakes_debug.mamba_spec(block_size=sizes[index],
                                              mamba_cache_mode=mode)
                only.tp_replicated = False
                spec.first_spec = only
            item = fakes_debug.manager_item(index, spec, stable=False,
                                            mamba_cache_mode=mode)
        elif index == 5:
            item = fakes_debug.manager_item(
                index, fakes_debug.attention_spec(block_size=sizes[index]),
                stable=True)
        else:
            item = fakes_debug.manager_item(
                index, fakes_debug.circular_buffer_spec(block_size=sizes[index]),
                stable=True)
        items.append(item)
    manager.cpu_coordinator = fakes_debug.coordinator(items)
    return manager


# --- 1. slot packing: the two fork signals, one named refusal ----------------

def test_a_slot_packed_group_is_refused_by_its_own_name():
    """The refusal is a NAMED, COUNTABLE reason, not a group quietly left out.

    Both halves matter: the name (`slot_packed_group`, distinct from the shared
    `incomplete_group_set`) because the `refused` map is the only place the next reader
    will look, and the sentence naming the group because "some group is not restorable"
    is not an answer when the reason is that its block ids are slot ids.
    """
    verdict = disk_restore._eligible_groups(packed(), BLOCK)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "slot_packed_group"
    assert verdict["blocked_group_ids"] == [PACKED_GROUP]
    assert verdict["blocked_kinds"] == {str(PACKED_GROUP): "slot_packed"}
    assert verdict["restorable_group_ids"] == [0, 1, 2, 3, 5]
    assert "slot ids, not pool block ids" in verdict["reason"]
    assert str(PACKED_GROUP) in verdict["reason"]
    assert len(verdict["reason"]) <= 160, "the verdict survives the last_refusal clip"
    service = disk_restore._state["group_serviceability"]
    assert service["complete"] is False
    assert service["disk_offers_possible"] is False
    # The published per-group row carries the engine's own number, so a reader learns
    # the packing width from the payload rather than from this file.
    assert service["group_contracts"][str(PACKED_GROUP)]["contract"] == "slot_packed"
    assert service["group_contracts"][str(PACKED_GROUP)]["slots_per_block"] == 800


def test_either_fork_signal_alone_is_enough_to_refuse():
    """`slot_pool` (the manager's answer) and `slots_per_block` (the config's).

    The scheduler-side reader tries the manager's `slot_pool` first because that is the
    test the fork's own scheduler uses, then the `slots_per_block` field because the
    coordinator sets it on the GROUP while the pool lives on the MANAGER - and a fork
    that has set one without exposing the other to this process is a fork we still must
    not mis-address. The width reported differs by signal, and both are the engine's
    number, not a guess.
    """
    by_pool = disk_restore._eligible_groups(packed(signal="slot_pool"), BLOCK)
    assert by_pool["reason_kind"] == "slot_packed_group"

    by_field = disk_restore._eligible_groups(packed(signal="slots_per_block"), BLOCK)
    assert by_field["reason_kind"] == "slot_packed_group"
    assert by_field["blocked_group_ids"] == [PACKED_GROUP]

    # A pool that exists but will not name its width is still a refusal: the FACT is
    # enough to refuse on, and the width is only ever reported, never needed.
    blind = packed(signal="slot_pool")
    blind.cpu_coordinator.single_type_managers[PACKED_GROUP].slot_pool.slots_per_block \
        = "many"
    verdict = disk_restore._eligible_groups(blind, BLOCK)
    assert verdict["reason_kind"] == "slot_packed_group"
    assert disk_restore._group_contract(blind, PACKED_GROUP)["slots_per_block"] is None


def test_the_packing_check_precedes_the_contract_check():
    """A slot-packed group is refused even when its CONTRACT would have been readable.

    The fixture's ring group answers `has_positionally_stable_blocks = True`, so with
    the packing signal removed it resolves to the `positional` contract and the gate
    opens.
    With it present, the same object resolves to `slot_packed`: addressing is asked
    before shape, because a perfectly-shaped group addressed at the wrong offset is the
    failure this file exists to prevent.
    """
    stable = packed()
    stable.cpu_coordinator.single_type_managers[PACKED_GROUP].slot_pool = None
    verdict = disk_restore._eligible_groups(stable, BLOCK)
    assert verdict["ok"] is True, "no packing signal, and the group is restorable"
    assert verdict["contracts"][str(PACKED_GROUP)] == "positional"
    assert disk_restore._group_contract(stable, PACKED_GROUP)["contract"] \
        == "positional"


def test_a_refused_offer_touches_nothing_and_leaves_the_forks_answer_alone(armed):
    """END TO END, THE SAME PROOF THE 0.2.25 GATE HAS: no offer, no slot, answer `is`.

    The counter moves and the sentence is logged; nothing else in the process is
    allowed to notice. The tuple comes back IDENTICALLY, and the CPU pool never sees a
    request - a refusal that allocated a destination slot is a refusal that lied about
    costing nothing.
    """
    fill(armed, seeds=[1, 2, 3, 4], groups=[0, 1, 2, 3, 4, 5])
    manager = packed()
    ram_result = (0, False)
    assert offer(manager, Req(request_id="req-pack", blocks=2), 0,
                 ram_result) is ram_result
    counters = disk_restore.counters()
    assert counters["refused"]["slot_packed_group"] == 1
    assert counters["refused"]["incomplete_group_set"] == 0, (
        "the specific name is the one that moves")
    assert counters["offers"] == 0
    assert counters["slots_allocated"] == 0
    assert manager.cpu_block_pool.allocated == []
    assert disk_restore._loads == {}
    assert manager._pending_cpu_hits == {}
    assert "slot_packed_group" in str(counters["last_refusal"])
    required = counters["last_groups_required"]
    assert required["offered"] is False
    assert required["refused_as"] == "slot_packed_group"
    # An un-priceable group says `null` rather than a guess, exactly as the `unstable`
    # contract already does: pricing a group we cannot address is the number that reads
    # as a plan and turns out to be a wish.
    by_group = {row["group"]: row for row in required["rows"]}
    assert by_group[PACKED_GROUP]["contract"] == "slot_packed"
    assert by_group[PACKED_GROUP]["blocks_needed"] is None
    assert by_group[PACKED_GROUP]["restored"] is False


def test_the_packing_refusal_counts_every_lookup_and_logs_once(armed, caplog):
    """Per-lookup counter, one WARNING per boot - the same discipline as the group gate.

    Under disk authority every lookup hits this refusal, so the counter is the
    measurement and the log line is the announcement.
    """
    fill(armed, seeds=[1, 2, 3, 4], groups=[0, 1, 2, 3, 4, 5])
    with caplog.at_level(logging.WARNING):
        for index in range(3):
            offer(packed(), Req(request_id=f"req-{index}", blocks=1), 0,
                  (0, False))
    counters = disk_restore.counters()
    assert counters["refused"]["slot_packed_group"] == 3
    assert counters["refusals_total"] == 3
    lines = [record.getMessage()
             for record in caplog.records
             if "slot ids, not pool block ids" in record.getMessage()]
    assert len(lines) == 1


# --- 2. mixed replication: no first_spec pricing, a refusal instead ---------

def test_a_merged_group_with_mixed_replication_is_refused_not_priced():
    """`tp_replicated` disagrees across the group's layers: refuse, do not average.

    `disk_restore._replication_uniformity` walks the container's `kv_cache_specs` (the
    same accessor `disk_store._sub_specs` knows) and fires ONLY on two distinct boolean
    answers - the positive-signal rule again. The fixture's group would otherwise
    resolve to the align-mode `boundary_state` contract from its `first_spec`, i.e.
    thirteen layers priced by one of them, which is exactly the mispricing the fork's
    `is_uniform_type` refuses to build.
    """
    contract = disk_restore._group_contract(merged(), 0)
    assert contract["contract"] == "mixed_replication"
    assert contract["tp_replicated_values"] == [False, True]
    assert "do not share one page shape" in contract["reason"]

    verdict = disk_restore._eligible_groups(merged(), BLOCK)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "mixed_replication_group"
    assert verdict["blocked_group_ids"] == [0, 1, 2, 3]
    assert set(verdict["blocked_kinds"].values()) == {"mixed_replication"}
    assert "do not share one page shape" in verdict["reason"]


def test_a_uniform_merge_and_a_first_spec_container_are_neither_refused_on_this():
    """THE CHECK FIRES ON DISAGREEMENT AND ON NOTHING ELSE.

    Two shapes that must NOT trip it: a container whose layers all agree (a legitimate
    merged group of one replication kind), and a container that only publishes
    `first_spec`. The second one is the important direction of the rule: one layer read
    from a thirteen-layer container cannot show a disagreement, so it is silence, and
    silence never refuses. It is also never BELIEVED - the row says `uniform=None` via
    the accessor that answered, which is why the scheduler's contract stays the thing
    that prices the group.
    """
    uniform = disk_restore._group_contract(merged(tp_replicated=(False, False)), 0)
    assert uniform["contract"] == "boundary_state", "an ordinary align group, priced"

    degraded = merged(accessor="first_spec")
    walked = disk_restore._replication_uniformity(
        degraded.cpu_coordinator.single_type_managers[0])
    assert walked["mixed"] is False and walked["uniform"] is None
    assert disk_restore._eligible_groups(degraded, BLOCK)["ok"] is True


# --- 3. no-regression: a boot where the engine says neither, unchanged ------

def test_a_boot_with_no_packing_signal_is_unchanged_including_its_refusal_name(armed):
    """THE OPT-IN RULE, PINNED. Every boot before v2.5 takes the `else` branch.

    Two assertions that together are the no-regression contract: the new counters exist
    and are ZERO on every other shape (a refusal name that never appears in a payload is
    indistinguishable from a refusal that was never reached), and the shapes that used
    to refuse under `incomplete_group_set` still refuse under `incomplete_group_set` -
    the name is chosen by the blocked KIND, not by a rewrite of the old verdict.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    assert offer(Manager(), Req(request_id="req-a", blocks=1), 0,
                 (0, False)) == (BLOCK, True)
    counters = disk_restore.counters()
    assert counters["offers"] == 1
    for name in (disk_restore.REFUSAL_SLOT_PACKED,
                 disk_restore.REFUSAL_MIXED_REPLICATION):
        assert name in counters["refused"], f"{name} is published from boot"
        assert counters["refused"][name] == 0, "no such group on this boot"
    # Every per-group row says the engine never mentioned packing or mixed replication.
    service = counters["group_serviceability"]
    assert all(row["slots_per_block"] in (None, 1)
               for row in service["group_contracts"].values())

    # The old refusal keeps its old name: same geometry the group-gate suite refuses on.
    from test_disk_restore_group_gate import hybrid
    assert disk_restore._eligible_groups(hybrid(mode=None), BLOCK)[
        "reason_kind"] == "incomplete_group_set"


def test_absent_and_benign_signals_never_fire():
    """`slots_per_block == 1`, an absent attribute, a non-int, and a raising read.

    `slots_per_block: int = 1` is the fork's DEFAULT on both `KVCacheGroupSpec` and
    `KVCacheTensor`, so a reader that treated "the attribute exists" as the signal would
    refuse every folded-engine boot and - worse - every boot of a fork that adopted the
    field without packing anything. Only a value GREATER THAN one, or a `slot_pool` that
    is not None, is a signal.
    """
    entry = fakes_debug.manager_item(0, fakes_debug.attention_spec(), stable=True)
    assert disk_restore._slot_packing(entry)["packed"] is False
    entry.slots_per_block = 1
    assert disk_restore._slot_packing(entry)["packed"] is False
    entry.slots_per_block = "800"
    assert disk_restore._slot_packing(entry)["packed"] is False
    entry.slots_per_block = True     # a bool is not a count
    assert disk_restore._slot_packing(entry)["packed"] is False
    entry.slot_pool = None
    assert disk_restore._slot_packing(entry)["packed"] is False

    class Raising:
        def __getattr__(self, name):
            raise RuntimeError("the fork renamed it")

    raising = Raising()
    assert disk_restore._slot_packing(raising)["packed"] is False
    assert disk_restore._replication_uniformity(raising)["mixed"] is False


# --- 4. the vocabulary stays closed and documented --------------------------

def test_both_new_refusal_names_are_defined_documented_and_shared():
    """The scheduler and the worker count the SAME fact under the SAME word.

    `OFFER_REFUSAL_WHY` carries a sentence for each, `scheduler_section` publishes the
    table, and `disk_store.REFUSAL_REASONS` - the spill side's own map, where tonight's
    real blocker surfaced as `no_key_for_slot` - carries both too, so a folded boot
    reports the group in the rank file as well as in the scheduler payload.
    """
    for name in (disk_restore.REFUSAL_SLOT_PACKED,
                 disk_restore.REFUSAL_MIXED_REPLICATION):
        assert name in disk_restore.OFFER_REFUSALS
        assert disk_restore.OFFER_REFUSAL_WHY.get(name), f"{name} has no WHY entry"
        assert name in disk_store.REFUSAL_REASONS
    assert "slot_pool_id" not in str(disk_restore.OFFER_REFUSALS)  # no stray spelling
    assert "slot ids, not pool block ids" in \
        disk_restore.OFFER_REFUSAL_WHY[disk_restore.REFUSAL_SLOT_PACKED]
    assert "first_spec" in disk_restore.OFFER_REFUSAL_WHY[
        disk_restore.REFUSAL_MIXED_REPLICATION]
    # The names the gate had before this change are all still there.
    for name in ("incomplete_group_set", "no_eligible_group", "no_disk_run",
                 "group_hit_truncated", "no_staging_rows"):
        assert name in disk_restore.OFFER_REFUSALS


def test_the_slot_packed_vocabulary_is_not_reused_for_a_real_miss(armed):
    """A refusal name is a FACT, and it must not be reachable by an ordinary miss.

    A cache miss, a missing key and an unimplementable contract all have their own
    counters. If any of those could also move `slot_packed_group`, the counter would
    stop being evidence that the engine SAID something about the group - which is the
    only thing that makes this refusal worth reading.
    """
    fill(armed, seeds=[1, 2, 3])
    manager = Manager()
    offer(manager, Req(request_id="req-miss", blocks=2), 0, (0, False))
    counters = disk_restore.counters()
    assert counters["refused"][disk_restore.REFUSAL_SLOT_PACKED] == 0
    assert counters["refused"][disk_restore.REFUSAL_MIXED_REPLICATION] == 0


# --- 5. the worker side: a slot-packed pair is never SPILLED, by name -------
#
# The scheduler refuses to OFFER a restore; this is the other half, and the more
# dangerous one, because spilling happens first in time. The worker copies a slot's
# bytes out of the group's region tensor at an offset formed from a block id, so for a
# slot-packed group the file it writes is a correctly-named file of the WRONG BYTES -
# and a future restore that reads it back answers plausibly. A well-named file of wrong
# bytes is the one artifact this tier must never be able to produce, so the pair is
# refused where the group is known: from the key's own 4-byte suffix, by the same
# `disk_layout.split_block_key` that names the file.

store = pytest.importorskip("test_disk_store")

# The sibling suite's own store fixture, under a name that says which suite it came
# from: it configures and arms a real store under tmp_path, which is what an
# end-to-end spill test needs and is not this file's to re-derive.
store_tier = store.tier  # noqa: F811 - re-exported as a fixture, not shadowed


@pytest.fixture(autouse=True)
def _reset_disk_store():
    """Undo everything this file does to `disk_store`'s module-level state.

    `conftest.py` resets the modules that have no other owner and explicitly leaves
    `disk_store` to "each module's own `reset_state_for_tests()`". The two sibling suites
    this file borrows fixtures from do not call it on every test, so a file that calls
    `observe_bind`/`on_store_completed` - as the spill tests below do - hands the next
    file a dirty retention map, a dirty `refused` table and a cached addressing view.
    That is not hypothetical: without this fixture the spill-shutdown and
    span-completeness suites fail in a full run and pass alone, which is the
    "leaks into a later file" bug class `conftest.py` exists to prevent. Autouse, so no
    test here can forget it.
    """
    disk_store.reset_state_for_tests()
    yield
    disk_store.reset_state_for_tests()
# The fork's `get_none_hash_seed()`, importable: without it the worker cannot read its
# own fingerprint and every spill refuses `fingerprint_incomplete` before the group is
# ever consulted, which would make the two spill tests below prove nothing.
none_hash_seed = store.none_hash_seed  # noqa: F811 - re-exported as a fixture


def store_handler(packed_group=None, *, mixed_group=None, slots=800):
    """`test_disk_store`'s own handler, with the fork's new field set on one group.

    The groups come from that suite's fixture, so the addressing view is read from the
    same object shape a real worker holds; only the v2.5 field is added, and only when
    the test asks for a packed or merged group. With both arguments None this IS the
    pre-0088 handler, which is what the no-regression test below needs.
    """
    role = store.handler()
    groups = role.kv_cache_config.kv_cache_groups
    if packed_group is not None:
        groups[packed_group].slots_per_block = slots
    if mixed_group is not None:
        spec = groups[mixed_group].kv_cache_spec
        sharded = fakes_debug.mamba_spec()
        replicated = fakes_debug.mamba_spec()
        replicated.tp_replicated = True
        spec.kv_cache_specs = {"a": sharded, "b": replicated}
    return role


def test_the_worker_reads_both_fork_signals_from_its_own_group_table():
    """`slots_per_block` on the group, and a container disagreeing on `tp_replicated`.

    The two lists are separate because the remedies are separate, and `signals` carries
    the evidence per group so a rank file can say WHY it refused without a reader having
    to know what a RingSlotPool is.
    """
    packed_role = store_handler(packed_group=1, slots=800)
    view = disk_store.unaddressable_group_ids(packed_role)
    assert view["ok"] is True
    assert view["slot_packed"] == [1]
    assert view["mixed_replication"] == []
    assert "slots_per_block == 800" in view["signals"]["1"]

    mixed_role = store_handler(mixed_group=2)
    view = disk_store.unaddressable_group_ids(mixed_role)
    assert view["slot_packed"] == []
    assert view["mixed_replication"] == [2]
    assert "tp_replicated" in view["signals"]["2"]

    # The default handler: the field does not exist, so both lists are empty and every
    # boot up to today is unaffected.
    plain = disk_store.unaddressable_group_ids(store.handler())
    assert plain["slot_packed"] == [] and plain["mixed_replication"] == []


def test_the_addressing_view_is_cached_per_config_and_a_failure_is_not_cached():
    """Boot-stable data, read on the step thread: cached on the config's identity.

    The same rule and the same reason as `_span_shapes_for`: the group table does not
    move within a boot, this runs per store event, and a FAILURE must heal on the next
    completion rather than be believed for the rest of the process.
    """
    role = store_handler(packed_group=0)
    first = disk_store.unaddressable_group_ids(role)
    assert disk_store._addressing_cache["view"] is first
    assert disk_store.unaddressable_group_ids(role) is first, "cached while unchanged"

    broken = store.handler()
    broken.kv_cache_config = None
    again = disk_store.unaddressable_group_ids(broken)
    assert again["ok"] is False
    assert disk_store._addressing_cache["config"] is not None, ("the failure did not "
                                                                "replace the cache")
    # THIS test writes a cache entry, and the cache is module-level between-call memory
    # on a step-thread path. `disk_store.reset_state_for_tests()` clears it, but this
    # file's fixtures come from the sibling suites and do not all call it, so undo it
    # here rather than trust the next file to reset what this one polluted.
    disk_store._addressing_cache["config"] = None
    disk_store._addressing_cache["view"] = None


def test_a_slot_packed_pair_is_never_written_and_the_refusal_is_counted_by_name(
        store_tier, none_hash_seed, monkeypatch):
    """THE SPILL GATE END TO END: no file, no queue job, one named counter.

    The key in the fixture names group 3, so group 3 is the one packed here; the pair is
    refused before the queue and therefore before any bytes are read out of the region,
    and `refused.slot_packed_group` is what a rank file will show. The mirror door is
    open for the test, so the refusal cannot be an artefact of a closed door.
    """
    with store.metadata_shape(True):
        role = store_handler(packed_group=3)
        step = store.metadata(store_cpu_blocks=[7], store_keys=[store.LONG_KEY])
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        report = disk_store.on_store_completed(role, [])
    assert report["spilled"] == 0
    counters = disk_store.counters()
    assert counters["refused"]["slot_packed_group"] == 1
    assert counters["refused"]["mixed_replication_group"] == 0
    assert counters["spill_queued"] == 0, "nothing reached the writer thread"
    assert "not addressed by pool block id" in str(counters["last_reason"]), \
        "the witness names the reason, not just the counter"


def test_an_ordinary_pair_still_spills_when_another_group_is_packed(store_tier,
                                                                     none_hash_seed,
                                                                     monkeypatch):
    """THE CHECK IS PER GROUP, NOT PER BOOT: a packed ring does not blind the tier.

    Same handler, key naming group 1 while group 3 is the slot-packed one: the pair
    spills exactly as it did before this gate existed. Without this direction the gate
    would be a kill switch wearing a predicate, and the live box's group 4 is exactly
    the kind of group a folded boot would pack.
    """
    key_group1 = store.LONG_KEY[:-8] + "00000001"
    assert disk_layout.split_block_key(key_group1)["group_id"] == 1
    with store.metadata_shape(True):
        role = store_handler(packed_group=3)
        step = store.metadata(store_cpu_blocks=[7], store_keys=[key_group1])
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        report = disk_store.on_store_completed(role, [])
    assert report["spilled"] == 1
    counters = disk_store.counters()
    assert counters["refused"]["slot_packed_group"] == 0
    assert counters["spill_queued"] == 1
