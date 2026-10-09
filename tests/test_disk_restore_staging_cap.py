r"""THE STAGING CAP (0.2.59): what sizes a restore, and WHO OWNS a staging row.

A restore's destination is a CPU row of `cpu_block_pool`, one per entry of the run, and
this suite is about the two questions that follow. (1) WHAT SIZES an offer: the arena's
free list (`arena`, the shipped default, which has CLIPPED an offer to what it could pay
for since 0.2.38) or a declared cap on rows in flight (`bounded`). (2) WHO HOLDS the row
while the copy runs - and this is where the first design of this feature was WRONG, so
the tests are ordered around it.

1. WHO OWNS A ROW. The design that shipped in this WIP reserved its rows at boot and
   handed those out. The deployed fork frees EVERY CPU row in a load's transfer when the
   load ends (`manager.py:1346`), so a reserved row's refcount reaches 0 and the pool
   takes it back - while the plugin still lists it as its own. That is one row on two
   free lists, and the next allocation hands the same bytes to a second restore: wrong
   output, no exception, and invisible until a box had been running for a week. These
   two tests are written to be readable against BOTH designs (they ask the module what
   rows it believes it owns, whatever it calls them), and the first one FAILS against
   the reserved-row design for exactly that reason.
2. THE ACCOUNTING PROOF, now run THROUGH THE FORK'S RELEASE. A success, a load error, an
   abort, an expiry and a cap eviction, every one of them followed by the fork's own
   `_cleanup_load_request`, and the pool's rows all come home exactly once with the cap
   back at its baseline. Asserting a baseline WITHOUT running the release is what let
   the hole above through review: the free list looked right because nothing had
   released anything yet.
3. THE CAP IS THE BOUND, NOT THE ARENA'S MOOD. An arena big enough for ONE offer - not
   for the worst prompt - carries a whole restore, and the offer's size stops depending
   on how much RAM the RAM tier happens to have left.
4. CLIPPING KEEPS THE COVERED SET HONEST. A run longer than the cap is CLIPPED, never
   waited for, and the ledger, the merged tuple and the emitted paths all carry the
   CLIPPED run and nothing else.
5. THE ROUTE, the deprecated `bounce` alias, and the per-OFFER pin that stops a mid-boot
   flip from re-pricing a run that is already staging.

Offline fakes only: no engine, no torch, no CUDA, no network, no live box.
"""

from __future__ import annotations

import ctypes

import fakes_debug
import pytest
import test_disk_restore as base

from kvtier_dev_plugin import (
    config,
    disk_layout,
    disk_restore,
    disk_store,
)

armed = base.armed                       # noqa: F811 - re-exported fixture
armed_gap_fill = base.armed_gap_fill     # noqa: F811
allow_tmp_root = base.allow_tmp_root     # noqa: F811 - autouse
fresh = base.fresh                       # noqa: F811 - autouse: a fresh boot
fill = base.fill
offer = base.offer
settle = base.settle
merged_cpu = base.merged_cpu
bind = base.bind
Meta = base.Meta
Worker = base.Worker
Manager = base.Manager
Req = base.Req
Region = base.Region
BLOCK = base.BLOCK
WIDTH = base.WIDTH
CACHEABLE = base.CACHEABLE
key_hex = base.key_hex

# One span of this model's positional groups: three cacheable groups, no boundary
# states, so ONE block of offer costs exactly this many staging rows.
PER_SPAN = len(CACHEABLE)


# --- selecting the posture ----------------------------------------------------

def bounded(monkeypatch, cap=12):
    """Select `bounded` for this boot and size the cap, out loud.

    The cap is set through the operator's own variable rather than by patching the
    constant, because it is read through `staging_rows_verdict()` and a test that
    patched the constant would be testing a number nothing reads.
    """
    monkeypatch.setenv(config.ENV_DISK_RESTORE_STAGING_ROWS, str(cap))
    verdict = disk_restore.set_restore_source("bounded")
    assert verdict["ok"] is True, verdict
    return verdict


def posture_this_build_offers(monkeypatch, cap=12):
    """Switch to the non-arena posture, by whatever words this build accepts.

    Deliberately tolerant, because the aliasing tests below state an INVARIANT (the
    module must not claim a row the pool has taken back) rather than a name: the same
    test must run against a build whose cap is called `..._STAGING_ROWS` and a build
    that still has `..._RING_SLOTS` and the `bounce` word, and it must FAIL on the
    invariant rather than skip or error on a missing name. A test that only knew the new
    names would pass on an old build by never entering the code it claims to be testing
    - which is precisely how the hole this file pins stayed alive through a green suite.
    """
    for var in (getattr(config, "ENV_DISK_RESTORE_STAGING_ROWS", None),
                getattr(config, "ENV_DISK_RESTORE_RING_SLOTS", None)):
        if var:
            monkeypatch.setenv(var, str(cap))
    for word in ("bounded", "bounce"):
        if disk_restore.set_restore_source(word)["ok"] is True:
            return word
    pytest.skip("this build offers no non-arena restore posture")


def cap_state():
    """The cap's published state, under whichever key this build publishes it."""
    counters = disk_restore.counters()
    for key in ("staging_cap", "restore_ring"):
        if key in counters:
            return counters[key]
    pytest.skip("this build publishes no staging budget")


# --- the rows the plugin believes it owns ------------------------------------

def plugin_claimed_rows():
    """EVERY CPU row this module currently believes it owns, whatever it calls them.

    A structural question, asked structurally: walk the module's cross-request state and
    collect any object that is a pool row. The point is that the answer must be EMPTY in
    a correct design - the plugin owns a BUDGET, `cpu_block_pool` owns the ROWS, and a
    module that names a row it does not hold is one `get_new_blocks()` away from handing
    two restores the same bytes. It is written as a walk so it finds the rows of a
    design that stores them under any name, in any shape, including one that has not
    been written yet.
    """
    found: set = set()

    def visit(value, depth=0):
        if depth > 6 or value is None:
            return
        block_id = getattr(value, "block_id", None)
        if block_id is not None and getattr(value, "is_null", False) is False:
            found.add(int(block_id))
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)

    for name in ("_staging", "_ring"):
        visit(getattr(disk_restore, name, None))
    return found


# --- the pool, with the refcount the real pool keeps -------------------------

class RefPool(base.CpuPool):
    """`cpu_block_pool` WITH ITS REFERENCE COUNT.

    The plain fake tracks a free list and no refcount, which is exactly why the hole
    this file pins was invisible: a row that the fork is still holding, and a row that
    is free, look identical. `BlockPool` distinguishes them - `get_new_blocks` takes a
    reference (`core/block_pool.py:688`), `touch` takes one and removes the row from the
    free list (`block_pool.py:760-762`), `free_blocks` drops one and puts the row ON the
    free list only when the count reaches 0 (`block_pool.py:786-798`) - and a test of
    ownership has to model that or it asserts nothing.
    """

    def get_new_blocks(self, count):
        taken = super().get_new_blocks(count)
        for block in taken:
            assert block.ref_cnt == 0, f"row {block.block_id} was not free"
            block.ref_cnt += 1
        return taken

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self._free = [row for row in self._free if row is not block]
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        back = []
        for block in blocks:
            if block.is_null:
                continue
            assert block.ref_cnt > 0, (
                f"row {block.block_id} freed while already free: a double free puts it "
                "on the free list twice, which is the same aliasing this file "
                "exists to "
                "prevent, arrived at from the other direction")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                back.append(block)
        self.freed.append([int(block.block_id) for block in blocks])
        self._free = back + self._free

    def free_ids(self):
        """Every row id currently on the pool's free list."""
        return {int(block.block_id) for block in self._free}

    def duplicates(self):
        """Rows appearing on the free list more than once - the aliasing, from the side.

        `free_blocks` puts a row on the list when its count reaches 0, so a row released
        twice by two owners lands there twice and two later owners are handed the same
        bytes. That is the failure this whole file is about, and it is visible in
        exactly one place: the shape of the free list itself.
        """
        seen, twice = set(), set()
        for block in self._free:
            bid = int(block.block_id)
            if bid in seen:
                twice.add(bid)
            seen.add(bid)
        return twice

    def ref(self, block_id):
        return self.blocks[int(block_id)].ref_cnt


class Fork:
    """The DEPLOYED fork's load lifecycle, transcribed from the box, not from memory.

    Two methods, both copied line-for-line out of
    `/home/user/vllm-38-next/vllm-workspace/vllm/v1/simple_kv_offload/manager.py`
    (md5 91a3aa6566710cb10e38af65fa33f318, read off the live box):

    * `update_state_after_alloc` (`manager.py:381-492`): takes the pending hit, TOUCHES
      the blocks the load will use (`:479`) and then releases the lookup pin
      (`_free_pending_cpu_hit`, `:481`).
    * `_cleanup_load_request` (`manager.py:1321-1355`): frees one reference for EVERY
      cpu
      block id in the transfer (`:1346-1349`), tail-first, whoever allocated it.

    That second one is the whole design constraint: the fork releases the transfer's
    rows unconditionally, so nothing outside the fork can keep one.
    """

    def __init__(self, manager):
        self.manager = manager
        self.pool = manager.cpu_block_pool

    def update_state_after_alloc(self, request_id, external_tokens=None):
        pending = self.manager._pending_cpu_hits.get(request_id)
        assert pending is not None, "the plugin never handed the fork a pending hit"
        lists, hit_length, _computed = pending
        tokens = hit_length if external_tokens is None else external_tokens
        taken = []
        for group, group_list in enumerate(lists):
            if group not in self.manager.prefix_cacheable_group_ids:
                continue                                  # manager.py:449-451
            size = self.manager.group_block_sizes[group]
            n_take = -(-tokens // size)                   # cdiv, manager.py:453
            taken.extend(block for block in group_list[:n_take]
                         if not block.is_null)
        self.pool.touch(taken)                            # manager.py:479
        self.manager._pending_cpu_hits.pop(request_id)
        self.manager._free_pending_cpu_hit(pending)       # manager.py:481
        return [int(block.block_id) for block in taken]

    def cleanup_load_request(self, cpu_block_ids):
        rows = [self.pool.blocks[int(bid)] for bid in reversed(cpu_block_ids)]
        self.pool.free_blocks(rows)                       # manager.py:1346-1349


# --- scenario helpers --------------------------------------------------------

def un_fuse():
    """Clear the failure fuse between scenarios (see `test_disk_restore_races.py`)."""
    disk_restore._suppressed.clear()
    disk_restore._suppressed_spans.clear()
    disk_restore._burned.clear()
    disk_restore._race_reoffer.clear()


def reported(*request_ids):
    return fakes_debug.Obj(name="KVConnectorOutput",
                           finished_recving=set(request_ids),
                           finished_sending=set())


def failed(*request_ids):
    return fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                           finished_sending=set(), failed_recving=set(request_ids),
                           invalid_block_ids=set())


def empty():
    return reported()


def stage(root, manager, fork, request_id, blocks=2, release=True):
    """Offer, let the fork TAKE the rows, adopt the load, and (by default) release it.

    THE ORDER IS THE FORK'S OWN: the plugin's offer writes `_pending_cpu_hits`, the
    scheduler allocates GPU blocks and calls `update_state_after_alloc` - which touches
    the taken rows and frees the pin - and only then does `build_connector_meta` adopt
    the load into a step. The release at the end is what the old suite skipped, and what
    a test must not skip: it is the step that puts a row back on the pool's free list.
    """
    offer(manager, Req(request_id=request_id, blocks=blocks))
    slots = merged_cpu(manager, request_id)
    taken = fork.update_state_after_alloc(request_id)
    manager._reqs_to_load.setdefault(request_id, object())
    disk_restore.build_load_paths(
        manager, Meta(load_event=7, cpu=slots,
                      gpu=[70_000 + index for index in range(len(slots))],
                      reqs={7: [request_id]}))
    result = {"slots": slots, "taken": taken}
    if release:
        fork.cleanup_load_request(taken)
    return result


def slot_path(root, seed, group):
    made = disk_layout.build_slot_path(str(root), base.key_hex(seed, group),
                                       base.PLACEHOLDER)
    assert made["ok"], made["reason"]
    return made["path"]


def scribble(region, slot):
    ctypes.memset(region.data_ptr() + slot * WIDTH, 0xEE, WIDTH)


# --- 1. who owns a row ------------------------------------------------------

def test_a_released_load_never_leaves_a_row_the_plugin_still_claims(armed, monkeypatch):
    """THE HOLE. A row the module claims must never be on the pool's free list.

    Driven through the fork's ACTUAL release path - `update_state_after_alloc` touches
    the taken rows and releases the lookup pin, then `_cleanup_load_request` frees one
    reference for every row in the transfer - because that release is what decides
    ownership, and a suite that asserted a free-list baseline without running it passed
    while a reserved row sat on two free lists at once.

    Three windows, because the fault is a matter of WHEN:
      a. while the load is in flight the pool cannot hand its rows to anyone;
      b. once the fork has released them, the module must not claim them either;
      c. whatever the module does claim must be un-allocatable from the pool.
    """
    root = armed
    posture_this_build_offers(monkeypatch, cap=PER_SPAN * 4)
    pool = RefPool(num_blocks=64)
    manager = Manager(pool=pool, reqs_to_load={"req-1": object()})
    fork = Fork(manager)
    fill(root, blocks=2)

    baseline_free = pool.free_ids()
    rows = stage(root, manager, fork, "load-1", blocks=2, release=False)
    assert rows["taken"], "the fork took no destination rows"

    # (a) IN FLIGHT: the fork's touch holds them, so the pool cannot hand them out.
    in_flight = set(rows["taken"])
    assert not (in_flight & pool.free_ids()), \
        f"a live load's rows are on the pool's free list: {sorted(in_flight)}"
    assert all(pool.ref(bid) >= 1 for bid in in_flight)

    # The plugin's positive report retires the load; then the fork releases the
    # transfer.
    disk_restore._state["reads_applied"] = len(rows["slots"])
    disk_restore.note_reported(manager, reported("load-1"))
    fork.cleanup_load_request(rows["taken"])

    # (b) THE CLAIM AND THE FREE LIST MUST NEVER OVERLAP. This is the whole failure
    # mode: an overlapping row is one `get_new_blocks()` away from being two restores'
    # destination at once - wrong output with no exception.
    overlap = plugin_claimed_rows() & pool.free_ids()
    assert not overlap, (
        f"the module claims rows {sorted(overlap)} as its own while the pool's free "
        f"list also holds them (posture "
        f"{disk_restore.effective_restore_source()!r}): a later allocation hands the "
        "same bytes to a second restore")

    # (c) The rows went HOME - all of them, once each, to a pool that started and ended
    # with the same free list. A restore must not leak a row or duplicate one. (The
    # `freed` LOG repeats legitimately: the fork's pin release and its transfer release
    # both free the same rows, at refcounts 2 and 1. It is the free LIST that must never
    # hold a row twice, because that is what puts two owners on one row.)
    assert pool.free_ids() == baseline_free, (
        f"the pool did not get every row back exactly once: "
        f"+{sorted(pool.free_ids() - baseline_free)} "
        f"-{sorted(baseline_free - pool.free_ids())}")
    assert not pool.duplicates(), \
        f"rows on the pool's free list twice: {sorted(pool.duplicates())}"
    assert not plugin_claimed_rows() & in_flight, \
        "the module still claims a row this load used"


def test_no_two_loads_are_ever_handed_the_same_staging_row(armed, monkeypatch):
    """The consequence, stated as the allocator sees it.

    Load A runs to completion and the fork releases its rows. Then load B is offered. If
    the module still claims any of A's rows, B's allocation can hand those same bytes to
    B - which is the aliasing, observed from the pool side rather than from the plugin's
    bookkeeping.
    """
    root = armed
    posture_this_build_offers(monkeypatch, cap=PER_SPAN * 4)
    pool = RefPool(num_blocks=64)
    manager = Manager(pool=pool, reqs_to_load={"req-1": object(), "req-2": object()})
    fork = Fork(manager)
    fill(root, blocks=2)
    fill(root, seeds=[2])

    first = stage(root, manager, fork, "req-1", blocks=2)
    disk_restore._state["reads_applied"] = len(first["slots"])
    disk_restore.note_reported(manager, reported("req-1"))
    claimed_for_a = plugin_claimed_rows()

    second = stage(root, manager, fork, "req-2", blocks=2)
    # Whatever the module claims for A, the pool must not have handed to B.
    shared = claimed_for_a & set(second["taken"])
    assert not shared, (
        f"row(s) {sorted(shared)} are claimed by the module for load A and were handed "
        f"to load B by the pool: two entries aliasing the same bytes, which is wrong "
        "output with no error to find")
    assert not pool.duplicates(), \
        f"rows on the pool's free list twice: {sorted(pool.duplicates())}"


# --- 2. the accounting proof, WITH the fork's release -----------------------

def test_every_terminal_path_returns_its_credit_and_every_row_goes_home(
        armed, monkeypatch):
    """Success, load error, abort, expiry and cap eviction - each one FOLLOWED BY THE
    FORK'S RELEASE - and both the cap and the pool return to their exact baseline.

    The previous version of this test asserted the ring's baseline WITHOUT running the
    fork's release, which is how a reserved row that the fork had already handed back
    looked like a healthy accounting proof. Now every load is released the way the fork
    releases it, so the baseline means something: rows home, credit returned, and the
    two lists disjoint.
    """
    root = armed
    bounded(monkeypatch, cap=PER_SPAN * 4)      # 4 spans' worth of rows
    rows = PER_SPAN * 2                          # one 2-block offer's rows
    pool = RefPool(num_blocks=64)
    manager = Manager(pool=pool)
    fork = Fork(manager)
    baseline_free = pool.free_ids()

    fill(root, blocks=2)
    stage(root, manager, fork, "boot")
    disk_restore._state["reads_applied"] = 1
    disk_restore.note_reported(manager, reported("boot"))

    # (a) a success, with an applied read behind it: a `returned` credit.
    un_fuse()
    stage(root, manager, fork, "ok")
    disk_restore._state["reads_applied"] = 1
    disk_restore.note_reported(manager, reported("ok"))

    # (b) a load error: the engine names the request in `failed_recving`.
    un_fuse()
    stage(root, manager, fork, "err")
    disk_restore.note_reported(manager, failed("err"))

    # (c) an abort: the fork no longer has a load state for it, so the sweep releases
    # the pin underneath us and no report will ever arrive.
    un_fuse()
    aborted = stage(root, manager, fork, "abrt", release=False)
    fork.cleanup_load_request(aborted["taken"])
    manager._reqs_to_load.pop("abrt")
    disk_restore.note_reported(manager, empty())

    # (d) an expired ledger: the load outlived its deadline unreported.
    un_fuse()
    stage(root, manager, fork, "expr")
    disk_restore._loads["expr"]["deadline_mono"] = 0.0
    disk_restore.note_reported(manager, empty())

    # (e) the ledger's cap: a load dropped to make room, with no news at all.
    un_fuse()
    monkeypatch.setattr(disk_restore, "MAX_TRACKED_LOADS", 1)
    stage(root, manager, fork, "cap1")
    ram = (0, False)
    assert disk_restore.offer(manager, Req(request_id="cap2", blocks=2), 0,
                               ram) == ram
    assert disk_restore.counters()["refused"]["too_many_loads"] == 1, \
        "the ledger's cap is a gate at the offer, not only an eviction"
    disk_restore._evict_oldest_load()

    cap = cap_state()
    # THE POOL'S BASELINE: every row home, none duplicated, nothing claimed.
    assert pool.free_ids() == baseline_free, (
        f"rows did not all come home: +{sorted(pool.free_ids() - baseline_free)} "
        f"-{sorted(baseline_free - pool.free_ids())}")
    assert not pool.duplicates(), (
        f"rows on the pool's free list twice: {sorted(pool.duplicates())} - two owners "
        "on one row, arrived at by releasing a row twice")
    assert not plugin_claimed_rows(), (
        f"the module still owns row ids {sorted(plugin_claimed_rows())} after every "
        "load ended: with the pool free to allocate them too, that is aliasing")
    assert cap["in_use_rows"] == 0, cap
    assert cap["in_use_requests"] == 0, cap
    # The leak counter stays where a fix leaves it: 0, and not because nothing moved.
    assert cap["unreturned_rows"] == 0, cap
    # Two loads reported positively; four are terminal without a report (error, abort,
    # expiry, eviction). Both kinds return the credit.
    assert cap["returned_rows"] == rows * 2, "boot + the clean success"
    assert cap["recovered_rows"] == rows * 4, "error + abort + expiry + eviction"
    assert disk_restore._loads == {}, "the cap-evicted load left the ledger"
    assert cap["rows_admitted"] == rows * 6, "six loads, each charging its own rows"
    # THE IDENTITY: every row ever charged is back on a positive report, back because
    # the load is terminal, given back unissued, or still charged. Anything else is a
    # leak.
    assert cap["rows_admitted"] == (cap["returned_rows"] + cap["recovered_rows"]
                                    + cap["unadmitted_rows"] + cap["in_use_rows"])
    assert disk_restore._slot_paths == {}, "no slot keeps a dead load's mapping"


def test_an_offer_that_never_reached_the_fork_gives_its_credit_back(armed, monkeypatch):
    """A charge that was never issued is not a leak either.

    `_release_staging` is the undo for an offer that fell apart AFTER being charged and
    BEFORE the fork held the tuple. It has to give back BOTH halves - the rows to the
    pool (the fork will never release rows it never saw) and the credit to the cap. A
    silent failure of either is a restore tier that gets a little smaller every day.
    """
    root = armed
    bounded(monkeypatch, cap=PER_SPAN * 2)
    fill(root, blocks=1)
    pool = RefPool(num_blocks=32)
    Manager(pool=pool, reqs_to_load={"req-1": object()})
    before_free = pool.free_ids()
    assert disk_restore._admit_staging("req-1", PER_SPAN)["ok"] is True
    allocated = disk_restore._allocate(pool, PER_SPAN)
    assert allocated["ok"] is True
    assert pool.free_ids() == before_free - {int(b.block_id)
                                             for b in allocated["blocks"]}
    disk_restore._release_staging(pool, allocated["blocks"], True, "req-1")
    cap = cap_state()
    assert pool.free_ids() == before_free, \
        "an offer's rows did not go back to the pool that allocated them"
    assert cap["unadmitted_rows"] == PER_SPAN
    assert cap["in_use_rows"] == 0 and cap["in_use_requests"] == 0
    assert cap["rows_admitted"] == cap["unadmitted_rows"]
    assert not plugin_claimed_rows()


# --- 3. the cap is the bound, not the arena's mood --------------------------

def test_a_bounded_restore_needs_an_arena_only_as_big_as_one_offer(armed, monkeypatch):
    """THE point of the posture, stated honestly.

    A whole restore still needs `spans * groups + boundary` arena rows per offer - there
    is no third place a destination can come from - so this does NOT assert that a
    restore happens with a zero-sized arena (it cannot, and the earlier version of this
    file asserted something the code could not do). What it asserts is the thing that
    was actually blocking the box: an arena sized for ONE offer carries a whole restore,
    with the RAM tier's reserve intact, even when a much bigger prompt is being
    restored. The arena stops being sized by the worst prompt and starts being sized by
    the cap.
    """
    root = armed
    cap = PER_SPAN * 4                     # room for 4 spans of rows
    bounded(monkeypatch, cap=cap)
    # The arena holds exactly what the cap needs plus the RAM tier's reserve: nothing
    # that a longer prompt could use, because a longer prompt cannot be staged anyway.
    pool = RefPool(num_blocks=cap + 8)
    manager = Manager(pool=pool, reqs_to_load={"req-1": object()})
    fill(root, blocks=2)

    offer(manager, Req(request_id="req-1", blocks=2))
    slots = merged_cpu(manager, "req-1")
    assert len(slots) == PER_SPAN * 2, "a full offer, from an arena one offer wide"
    counters = disk_restore.counters()
    assert counters["offers_staged_bounded"] == 1
    assert counters["offers_staged_arena"] == 0
    assert counters["refused"]["no_cpu_slots"] == 0
    # And the pool still has its reserve for the RAM tier: a restore does not eat the
    # arena to the last row.
    assert pool.get_num_free_blocks() >= 8, \
        "the restore took the arena past what the cap said it needed"


def test_a_larger_arena_does_not_make_a_bounded_offer_longer(armed, monkeypatch):
    """THE DIFFERENCE `bounded` BUYS, in one assertion.

    Under `arena` an offer's length floats with the pool's free count, so the same
    prompt restores to different depths at different moments of the day. Under `bounded`
    the cap decides, and doubling the arena changes NOTHING. This is the whole reason
    the cap is worth its code: a sizing claim you can check (`arena >= cap + reserve`)
    instead of a number that quietly moved.
    """
    root = armed
    cap = PER_SPAN * 2                     # exactly two spans' worth
    bounded(monkeypatch, cap=cap)
    fill(root, blocks=8)                   # wants 8 spans, the cap allows 2

    widths = []
    for index, pool_rows in enumerate((cap + 8, cap + 200)):
        un_fuse()
        request_id = f"wide-{index}"
        pool = RefPool(num_blocks=pool_rows)
        manager = Manager(pool=pool, reqs_to_load={request_id: object()})
        offer(manager, Req(request_id=request_id, blocks=8))
        assert request_id in disk_restore._loads, (
            f"the {pool_rows}-row arena refused the offer: "
            f"{ {k: v for k, v in disk_restore.counters()['refused'].items() if v} }")
        record = disk_restore._loads[request_id]
        widths.append(record["tokens"] // BLOCK)
        assert len(merged_cpu(manager, request_id)) == widths[-1] * PER_SPAN
        # Retire it: the cap charges rows for the life of a load, and the second offer
        # is a different prompt, not a second one waiting on the same budget.
        disk_restore._drop_load(request_id, "measured")
    assert widths[0] == widths[1] == 2, (
        f"a `bounded` offer grew with the arena ({widths}): then the cap is not the "
        "bound and the arena still decides")


def test_an_arena_with_nothing_free_refuses_by_name(armed, monkeypatch):
    """The honest limit of both postures: no rows, no restore.

    A destination that does not exist is not a smaller destination. Refusing with
    `alloc_refused` - and, under `bounded`, counting it in the cap's own sub-map as
    `pool_cannot_pay`, which is the payload's way of saying "your arena is smaller than
    your cap" - is the only correct answer, and it must not become a wait: a wait in the
    step thread is what kills the engine.
    """
    root = armed
    bounded(monkeypatch, cap=PER_SPAN * 4)
    pool = RefPool(num_blocks=PER_SPAN * 4)
    manager = Manager(pool=pool, reqs_to_load={"req-1": object()})
    fill(root, blocks=2)
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool.get_num_free_blocks() == 0, "the arena is empty on purpose"

    ram = (0, False)
    answer = disk_restore.offer(manager, Req(request_id="req-1", blocks=2), 0, ram)
    assert answer == ram, "a refusal hands the fork its own answer back untouched"
    counters = disk_restore.counters()
    assert counters["refused"]["alloc_refused"] == 1, counters["refused"]
    assert counters["bounded_refusal_reasons"]["pool_cannot_pay"] == 1, \
        "the payload must say the arena - not the cap - was what could not pay"
    assert counters["offers_staged_bounded"] == 0
    assert disk_restore._loads.get("req-1") is None, "nothing was committed"
    assert cap_state()["in_use_rows"] == 0, \
        "a refused offer must not keep its cap credit"


def test_a_zero_cap_refuses_every_bounded_offer(armed, monkeypatch):
    """`..._STAGING_ROWS=0` is a real posture and is honoured: nothing can be staged, so
    nothing is offered, and the refusal is counted by name.

    The same reasoning `cpu_slot_reserve_env` gives for honouring 0: the safe direction
    for a bound whose whole job is to make offers smaller.
    """
    root = armed
    bounded(monkeypatch, cap=0)
    fill(root, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    ram = (0, False)
    answer = disk_restore.offer(manager, Req(request_id="req-1", blocks=2), 0, ram)
    assert answer == ram
    counters = disk_restore.counters()
    assert counters["refused"]["no_staging_rows"] == 1
    assert counters["bounded_refused_offers"] == 1
    assert counters["bounded_refusal_reasons"]["cap_full"] == 1
    assert cap_state()["cap_rows"] == 0
    assert cap_state()["used"] is False, "a refused offer never spent any credit"


# --- 4. clipping ------------------------------------------------------------

def test_a_run_longer_than_the_cap_is_clipped_and_only_covers_what_it_delivered(
        armed, monkeypatch):
    """CLIPPED, never waited for - and the covered set can only name delivered spans.

    A shorter offer is safe ONLY if everything downstream is built from the clipped
    length: an offer that staged 2 spans but put 8 spans' identities into the load
    ledger would let the prune remove an entry nobody delivered bytes for, which is a
    GPU block served from whatever was left in it. That would be a wrong answer, not a
    test gap
    - so this asserts the ledger, the merged tuple and the emitted paths all carry the
    CLIPPED run and nothing else.
    """
    root = armed
    bounded(monkeypatch, cap=PER_SPAN * 2)      # room for 2 spans of rows
    fill(root, blocks=4)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=4))     # wants 4 spans
    record = disk_restore._loads["req-1"]
    clipped_blocks = record["tokens"] // BLOCK
    assert 0 < clipped_blocks < 4, \
        f"a 4-block run into a 2-row-per-span cap must be clipped: {record['tokens']}"
    assert record["cpu_block_ids"], "the clipped run still has destinations"
    assert len(record["cpu_block_ids"]) == clipped_blocks * PER_SPAN
    assert record["staging_bounded"] is True
    assert cap_state()["clipped_offers"] == 1
    # THE BOUND IS NAMED: an offer that came in short has to say which bound cut it.
    pricing = disk_restore.counters()["last_offer_pool"]
    assert "staging_rows" in (pricing.get("bounded_by") or []), pricing
    # And the arena's free count is a MEASUREMENT, not a null: under `bounded` the cap
    # decided the size, but the arena still supplied the rows.
    assert pricing["cpu_slots_free"] is not None, pricing
    assert pricing["restore_source"] == "bounded", pricing
    # The identities are the FIRST `clipped_blocks` spans of the run: nothing that was
    # not staged can appear in the covered set, because nothing was built for it.
    spans = sorted({entry["span"] for entry in record["identities"]})
    assert spans == list(range(clipped_blocks)), spans
    slots = merged_cpu(manager, "req-1")
    assert len(slots) == len(record["cpu_block_ids"])
    metadata = Meta(load_event=8, cpu=slots,
                    gpu=[80_000 + index for index in range(len(slots))],
                    reqs={8: ["req-1"]})
    disk_restore.build_load_paths(manager, metadata)
    emitted = [path for path in metadata.load_paths if path]
    assert len(emitted) == len(slots), \
        "every entry of the clipped run has a file, and there are no extra entries"
    assert len(metadata.load_paths) == len(slots), \
        "the step carries no entry the offer did not stage"
    # And the credit the clip did NOT take is still free: clipping is a shorter offer,
    # not a longer hold.
    assert cap_state()["in_use_rows"] == clipped_blocks * PER_SPAN


# --- 5. aliasing, end to end ------------------------------------------------

def test_two_in_flight_entries_own_disjoint_rows_and_cannot_dirty_each_other(
        armed, monkeypatch):
    """The design's central claim, tested rather than asserted in a comment.

    Two offers are staged and both loads are read into their rows through the real bind
    path; then one entry's rows are scribbled over, and the other entry's bytes are
    still exactly the bytes of the file it was read from. Every entry's row is its own
    for the whole life of its load, so the three readers that could otherwise see each
    other - the fork's copy, the readback and a DMA that has not finished - have nothing
    shared
    to read.
    """
    root = armed
    bounded(monkeypatch, cap=PER_SPAN * 2)      # exactly two spans' worth
    fill(root, blocks=1)
    fill(root, seeds=[2])
    pool = RefPool(num_blocks=32)
    first = Manager(pool=pool, reqs_to_load={"req-1": object(), "req-2": object()})
    rows_a = stage(root, first, Fork(first), "req-1", blocks=1,
                   release=False)["taken"]
    rows_b = stage(root, first, Fork(first), "req-2", blocks=1,
                   release=False)["taken"]
    assert rows_a and rows_b
    assert not (set(rows_a) & set(rows_b)), \
        f"two in-flight loads share a staging row: {rows_a} / {rows_b}"

    # One step, both requests, six entries: A reads span 1, B reads span 2. The region
    # is sized to cover the pool, because a row's id IS its slot index.
    paths, cpu, gpu = [], [], []
    for seed, rows in ((1, rows_a), (2, rows_b)):
        for group, row in zip(CACHEABLE, rows, strict=True):
            paths.append(slot_path(root, seed, group))
            cpu.append(row)
            gpu.append(60_000 + len(gpu))
    worker = Worker(regions={"g0": Region(slots=32, width=WIDTH)})
    metadata = Meta(load_event=11, cpu=cpu, gpu=gpu, paths=paths,
                    reqs={11: ["req-1", "req-2"]})
    assert bind(worker, metadata)["ok"] is True
    settle(disk_restore._batch_id(11))
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True and report["applied"] == len(CACHEABLE) * 2
    region = worker.cpu_kv_caches["g0"]
    for seed, rows in ((1, rows_a), (2, rows_b)):
        for group, row in zip(CACHEABLE, rows, strict=True):
            assert region.slot_bytes(row) == \
                bytes([(seed * 11 + group) % 251] * WIDTH)

    # Now the hazard: one entry's rows are overwritten. The OTHER entry's delivered
    # bytes are unchanged, and the apply verdict for its entries is unchanged - there is
    # no shared row to dirty.
    for row in rows_a:
        scribble(region, row)
    for group, row in zip(CACHEABLE, rows_b, strict=True):
        assert region.slot_bytes(row) == bytes([(2 * 11 + group) % 251] * WIDTH), \
            "an entry's delivered bytes changed because of ANOTHER entry's row"
    # The batch is closed: nothing re-applies these bytes and re-introduces the risk.
    assert disk_restore._wbatches == {}
    again = disk_restore.worker_before_load(worker)
    assert again["applied"] == 0, "a closed batch cannot re-apply"


# --- 6. the runtime route ---------------------------------------------------

def post(body):
    """One `/plugins/kvtier/disk` verb, as the route dispatches it."""
    return disk_store.disk_for_engine(None, dict(body))


def test_the_restore_source_route_moves_the_posture_and_clears_back_to_the_default(
        monkeypatch):
    """`bounded`, `arena`, and `source: null` handing the decision back to the boot.

    `source: null` is NOT "arena": it is "I am no longer deciding", which on a boot that
    came up with `VLLM_E2_TIER_DISK_RESTORE_SOURCE=bounded` must land on `bounded`.
    """
    monkeypatch.setenv(config.ENV_DISK_RESTORE_SOURCE, "bounded")
    assert disk_restore.effective_restore_source() == "bounded"

    answer = post({"op": "restore-source", "source": "arena"})
    assert answer["ok"] is True, answer
    assert answer["restore_source"]["posture"]["effective"] == "arena"
    assert answer["posture"]["effective"] == "arena"
    assert answer["posture"]["source"] == "route"
    assert "staging_cap" in answer, \
        "the cap travels with the posture: a caller must see what bounds an offer"

    back = post({"op": "restore-source", "source": None})
    assert back["ok"] is True, back
    assert back["restore_source"]["cleared"] is True, back
    assert back["posture"]["effective"] == "bounded", \
        "clearing the runtime choice hands the decision back to the boot, not to arena"
    assert back["posture"]["source"] == "env"


def test_the_deprecated_alias_resolves_to_bounded_and_says_so(monkeypatch):
    """`bounce` still works, and every payload says what it became.

    The alias exists so a drop-in written against the pre-0.2.59 word boots instead of
    getting a 400 - but a name that once meant "the plugin owns these rows" must not be
    allowed to keep meaning that quietly. So the posture comes out as `bounded` and the
    word used is published in `alias`, for the operator who greps for what he typed.
    """
    monkeypatch.delenv(config.ENV_DISK_RESTORE_SOURCE, raising=False)
    answer = post({"op": "restore-source", "source": "bounce"})
    assert answer["ok"] is True, answer
    assert answer["posture"]["effective"] == "bounded"
    assert answer["posture"]["alias"] == "bounce"
    assert answer["posture"]["raw"] == "bounce"
    assert answer["staging_cap"]["cap_rows"] == disk_restore.RESTORE_STAGING_ROWS

    monkeypatch.setenv(config.ENV_DISK_RESTORE_SOURCE, "bounce")
    verdict = disk_restore.restore_source_verdict()
    assert verdict["effective"] == "bounded"
    assert verdict["alias"] == "bounce", \
        "a boot-time alias has to be as visible as a runtime one"
    assert disk_restore.effective_restore_source() == "bounded"


def test_an_invalid_restore_source_is_refused_and_changes_nothing(monkeypatch):
    """A typo is a 400 and a no-op, in both directions, at both doors."""
    monkeypatch.delenv(config.ENV_DISK_RESTORE_SOURCE, raising=False)
    assert disk_restore.effective_restore_source() == "arena"
    answer = post({"op": "restore-source", "source": "bananas"})
    assert answer["ok"] is False, answer
    assert disk_restore.effective_restore_source() == "arena"
    assert disk_restore.counters()["restore_source_flips"] == 0

    monkeypatch.setenv(config.ENV_DISK_RESTORE_SOURCE, "sideways")
    verdict = disk_restore.restore_source_verdict()
    assert verdict["recognised"] is False and verdict["effective"] == "arena"
    assert "sideways" in str(verdict["reason"])


def test_a_request_keeps_the_posture_its_offer_chose_when_the_posture_flips_under_it(
        armed, monkeypatch):
    """THE PER-OFFER PIN.

    The posture is read once, on the offer, and written on the ledger record; the credit
    is returned from THAT record and never from the live posture. Without the pin a flip
    landing between an offer and its release would charge the cap for a posture that is
    no longer selected and return credit for one that was never charged - an accounting
    fault that reads forever like a leak, and worse, an offer priced one way and its
    rows released the other.
    """
    root = armed
    bounded(monkeypatch, cap=PER_SPAN * 4)
    pool = RefPool(num_blocks=64)
    manager = Manager(pool=pool)
    fork = Fork(manager)
    fill(root, blocks=2)

    stage(root, manager, fork, "flipping", blocks=2, release=False)
    flipped = disk_restore.set_restore_source("arena")
    assert flipped["ok"] is True
    record = disk_restore._loads["flipping"]
    assert record["restore_source"] == "bounded", \
        "the offer's posture was overwritten by the flip"
    assert record["staging_bounded"] is True
    assert cap_state()["in_use_rows"] == PER_SPAN * 2

    fork.cleanup_load_request(record["cpu_block_ids"])
    disk_restore._state["reads_applied"] = 1
    disk_restore.note_reported(manager, reported("flipping"))
    # The credit comes back even though the posture no longer says `bounded`: the pin,
    # not the live switch, decides who is owed it.
    assert cap_state()["in_use_rows"] == 0
    assert cap_state()["returned_rows"] == PER_SPAN * 2
    assert not plugin_claimed_rows()


def test_a_bounded_refusal_is_not_silently_re_staged_from_the_arena(armed, monkeypatch):
    """A `bounded` offer the cap cannot pay for is REFUSED, not quietly re-priced.

    The fallback would be the cheapest thing in the world to write and the hardest thing
    in the world to see: the arena would keep absorbing restores, an operator would
    believe the cap was holding it, and the pinned-RAM ceiling would be breached without
    a single error. So the refusal is by name, counted in the cap's own sub-map, and
    nothing is staged.
    """
    root = armed
    bounded(monkeypatch, cap=PER_SPAN)         # one span of rows, and one is in use
    fill(root, blocks=2)
    pool = RefPool(num_blocks=64)
    manager = Manager(pool=pool, reqs_to_load={"req-1": object(), "req-2": object()})
    fork = Fork(manager)
    stage(root, manager, fork, "req-1", blocks=1, release=False)
    un_fuse()

    ram = (0, False)
    answer = disk_restore.offer(manager, Req(request_id="req-2", blocks=2), 0, ram)
    assert answer == ram
    counters = disk_restore.counters()
    assert counters["refused"]["no_staging_rows"] == 1, counters["refused"]
    assert counters["bounded_refusal_reasons"]["cap_full"] == 1
    assert counters["offers_staged_bounded"] == 1, "only the first offer was staged"
    assert disk_restore._loads.get("req-2") is None, "the refusal committed nothing"
    # The first load still holds its rows - the refusal added nothing to the charge.
    assert cap_state()["in_use_rows"] == PER_SPAN


# --- 7. the environment ----------------------------------------------------

def test_the_cap_reads_both_env_names_and_says_which_won(monkeypatch):
    """The canonical name wins, the superseded one still works, and a conflict is named.

    The variable was called `..._RING_SLOTS` while the rows were meant to be owned. An
    operator with the old word in a drop-in must not get a silent default, and one with
    BOTH names set to different numbers must be told that his sizing claim is a coin
    flip - not quietly served one of the two.
    """
    monkeypatch.setenv(config.ENV_DISK_RESTORE_STAGING_ROWS, "5")
    assert disk_restore.staging_row_cap() == 5
    assert disk_restore.staging_rows_verdict()["source"] == "env"

    monkeypatch.delenv(config.ENV_DISK_RESTORE_STAGING_ROWS)
    monkeypatch.setenv(config.ENV_DISK_RESTORE_RING_SLOTS, "7")
    verdict = disk_restore.staging_rows_verdict()
    assert verdict["rows"] == 7, "the superseded name must still be honoured"
    assert verdict["legacy_env_var"] == config.ENV_DISK_RESTORE_RING_SLOTS

    monkeypatch.setenv(config.ENV_DISK_RESTORE_STAGING_ROWS, "5")
    verdict = disk_restore.staging_rows_verdict()
    assert verdict["rows"] == 5, "the new name must win"
    assert verdict["problems"], "both names set to different numbers is a complaint"

    monkeypatch.setenv(config.ENV_DISK_RESTORE_STAGING_ROWS, "plenty")
    verdict = disk_restore.staging_rows_verdict()
    assert verdict["rows"] == disk_restore.RESTORE_STAGING_ROWS
    assert verdict["source"] == "ignored" and verdict["problems"]


def test_the_cap_bounds_concurrency_and_the_pool_bounds_allocation(monkeypatch):
    """Two different numbers, and reading them as one is the mistake to prevent.

    The cap answers "how many staging rows may be charged at once"; the pool answers
    "how many can actually be handed out right now". With nothing charged, a cap of 2
    spans
    prices an offer at 2 spans whatever the pool's size (that is the determinism the cap
    buys over `arena`, whose offer length floats with the arena's free list); with
    everything charged, the same pricing has nothing left to give and says which bound.
    """
    bounded(monkeypatch, cap=PER_SPAN * 2)
    priced = disk_restore._price_against_cap(None, blocks=8, per_span=PER_SPAN,
                                             boundary_slots=0, bounds=[])
    assert priced["ok"] is True and priced["blocks"] == 2, priced
    assert priced["detail"]["spans_the_cap_can_carry"] == 2
    assert disk_restore._admit_staging("charging", PER_SPAN * 2)["ok"] is True
    full = disk_restore._price_against_cap(None, blocks=8, per_span=PER_SPAN,
                                           boundary_slots=0, bounds=[])
    assert full["ok"] is False and full["reason_kind"] == "no_staging_rows", full
