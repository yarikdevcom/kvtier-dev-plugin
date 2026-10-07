r"""THE SCHEDULER-SIDE BINDING ORACLE: is the file we offered the file that block wants?

The worker-side verifier (`test_disk_restore_verify.py`) answers "did the bytes arrive?"
and can never answer "was the right file chosen?", because by the time a worker applies
a restore the pairing decision has been made, reported and forgotten. This file tests
the other half, in its 0.2.41 shape: TWO hooks in the EngineCore. `observe_alloc`
hung on `update_state_after_alloc` captures the EXPECTATION - what group and which
position in that group's GPU block table the fork is about to pair with which CPU slot,
together with the request's own `block_hashes` - and `observe_offer`, at name emission,
compares the file name this plugin's ledger produced against that captured expectation
(the re-derivation runs through the engine's own `resolve_block_hashes` and
`make_block_hash_with_group_id`, on this module's thread).

WHEN THIS HOOK MOVED, AND WHY IT MUST NOT MOVE BACK. 0.2.40 read the request's GPU
block table from `scheduler_output.kv_connector_block_state` at name-emission time.
That table belongs to the CURRENT step (`sched/scheduler.py:1412-1425`, cleared at
`:1497`) while a load state is emitted in steps LATER than the one that allocated it -
and in production every sampled entry hit exactly that gap: `block_state_does_not_cover_
request`, zero comparisons, a decorative witness. THE REGRESSION TEST FOR THAT
FAILURE IS
`test_a_load_emitted_in_a_later_step_than_its_alloc_still_gets_a_verdict` below: a step
whose `kv_connector_block_state` is None, or does not carry the request, must still
produce verdicts, because the expectation was captured EARLY. If someone "simplifies"
the pair back into one hook, that test is the one that should break the loudest.

Ordering: the order of ways this oracle's answer could be worthless.

1. A CORRECT PAIRING MUST BE CONFIRMED at the real hook signature (a `blocks` whose
`get_block_ids()` answers per-group tables, a request carrying `block_hashes`,
`num_external_tokens`), or every negative case below is worthless. 2. A ONE-SLOT SHIFT
MUST BE CAUGHT, SIGNED. That is the whole reason this oracle exists. 3. A GROUP SWAP
MUST BE CAUGHT, IN BOTH SHAPES - the wrong suffix, and the right suffix over the wrong
group's hash view. 4. THE PRODUCTION STEP-MISMATCH MUST BE REPRODUCED AND BEATEN (see
above). 5. THE EXPECTATION LIFECYCLE IS COUNTED: capture, consumption, eviction of an
unconsumed pairing, and the per-alloc bound each have their own counter; an emitted
entry with no expectation is a NAMED unavailability, never a silent skip. 6. AN ORACLE
THAT CANNOT BE BUILT AT ALLOC TIME IS SAID OUT LOUD at alloc time. 7. WHAT THE ENGINE
THREADS PAY, and the promise that nothing is hashed there. 8. THE SECTION NEVER CHANGES
SHAPE, NEVER OUTGROWS ITS CAP, and names the hook that produced its numbers.

WHAT IS FAKE, STATED WHERE IT CANNOT BE MISLED: the manager exposes the fields the
capture reads (`_pending_cpu_hits`, `hash_block_size`, `group_block_sizes`,
`prefix_cacheable_group_ids`), `blocks.get_block_ids()` is a dict-shaped fake answering
per-group lists, and the pending tuple's `KVCacheBlock` fakes carry `block_id` and
`is_null` the way the real pool's do. The key arithmetic is NOT faked - `pack` and
`resolve` are the same two reimplementations `test_disk_restore.py` holds the OFFER path
to, imported rather than copied. One test drives the REAL wrapper (`disk_restore.
install` -> the wrapped `update_state_after_alloc` -> the fork's own pairing loop ->
`build_load_paths`) so the hook site is pinned, not assumed.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import time
import types

import pytest
from test_disk_restore import (
    BLOCK,
    CACHEABLE,
    DIGEST,
    HASH_BLOCK,
    WIDTH,
    Meta,
    Req,
    Slot,
    fill,
    offer,
    pack,
    raw_digest,
    resolve,
)
from test_disk_restore import Manager as OfferManager

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    backport,
    config,
    disk_layout,
    disk_restore,
    disk_restore_verify,
    disk_spill,
    disk_store,
    snapshot,
    worker_diag,
)

# --- the geometry: two groups, deliberately different block sizes -------------
G_SIZE = (BLOCK, 1600)        # 3200 B and 1600 B: 25 and 12 hashes per block
HASHES = [raw_digest(seed) for seed in range(100)]
VIEW = {size: resolve(HASHES, HASH_BLOCK, size) for size in G_SIZE}
TABLE = {0: [7001, 7002, 7003, 7004],
         1: [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008]}
CPU = {0: [5001, 5002, 5003],          # 5003 is past the cdiv trim, on purpose
       1: [6001, 6002, 6003, 6004, 6005]}
RID = "req-alloc-1"
TOKENS = 6400                 # cdiv -> 2 blocks of group 0, 4 of group 1
# The pairs the DEFAULT alloc captures: (group, index) -> (gpu, cpu).
DEFAULT_PAIRS = {**{(0, i): (TABLE[0][i], CPU[0][i]) for i in range(2)},
                 **{(1, i): (TABLE[1][i], CPU[1][i]) for i in range(4)}}


def offered(index: int, group: int, *, from_group=None, suffix=None) -> str:
    """The name a file for block `index` of `group` is offered under.

    `from_group` builds the name out of ANOTHER group's hash view - the failure a suffix
    check cannot see, because the suffix is written correctly while the digest
    underneath it came from a view resolved with the wrong `block_size`. `suffix` writes
    a DIFFERENT group id into the name, which is the shape a suffix check CAN see.
    """
    digest = VIEW[G_SIZE[group if from_group is None else from_group]][index]
    return bytes(pack(digest, group if suffix is None else suffix)).hex()


class Request:
    """`Request`, as far as the alloc hook reaches: `request_id` and `block_hashes`."""

    def __init__(self, request_id=RID, hashes=HASHES) -> None:
        self.request_id = request_id
        self.block_hashes = list(hashes)


class Blocks:
    """`KVCacheBlocks`: the one call the hook makes on it is `get_block_ids()`."""

    def __init__(self, tables) -> None:
        self.tables = tables          # {group: [gpu ids]} - group-ordered lists

    def get_block_ids(self):
        groups = max(self.tables) + 1 if self.tables else 0
        return [list(self.tables.get(group, [])) for group in range(groups)]


class Manager:
    """The manager fields the CAPTURE reads, by name, nothing else.

    `_reqs_to_load` is deliberately absent: 0.2.40 read it (and
    `_abandoned_reqs_to_load`) to find the request at EMISSION time, which is the
    timing that went blind in production. The 0.2.41 hook is handed the request as
    an argument and asks the manager for nothing but the geometry and the pinned
    hit it peeks.
    """

    def __init__(self, *, hash_block=HASH_BLOCK, sizes=G_SIZE, cacheable=(0, 1),
                 hits=None) -> None:
        self.hash_block_size = hash_block
        self.group_block_sizes = list(sizes)
        self.prefix_cacheable_group_ids = list(cacheable)
        self._pending_cpu_hits = dict(hits if hits is not None else {})


def pending(cpu=None, computed=0, *, nulls=()):
    """The `_pending_cpu_hits` tuple, in the fork's shape: (blocks, hit, computed).

    `nulls` are `(group, index)` positions whose `KVCacheBlock` is the pool's null
    placeholder - the align-mode shape `update_state_after_alloc` skips WITHOUT
    advancing the pairing index, which is the one fork quirk this observer must
    replicate exactly or mis-pair every boundary group.
    """
    cpu = CPU if cpu is None else cpu
    groups = []
    for group, ids in sorted(cpu.items()):
        blocks = []
        for index, block_id in enumerate(ids):
            block = Slot(block_id)
            if (group, index) in nulls:
                block.is_null = True
            blocks.append(block)
        groups.append(blocks)
    return (groups, TOKENS, computed)


class Output:
    """`SchedulerOutput`, with the 0.2.40 field either absent, None, or not covering.

    The oracle no longer reads it; the test hands it one anyway, because the
    production failure happened with a REAL one in hand.
    """

    def __init__(self, state=None) -> None:
        if state is not None:
            self.kv_connector_block_state = state


class BlockState:
    """The fork's connector channel: covers what it is told, answers None otherwise."""

    def __init__(self, tables=None) -> None:
        self.tables = dict(tables or {})

    def get_block_ids(self, request_id):
        table = self.tables.get(request_id)
        return None if table is None else [list(group) for group in table]


def entry(*, group=0, index=1, key=None, request_id=RID, gpu=None, cpu=None,
          span=0) -> dict:
    """One witness entry, shaped exactly as `_build_load_paths` writes it.

    The default pairs group 0's SECOND captured block with the name for that block, so
    the default is a CORRECT offer and every test states only the one thing it breaks.
    The (gpu, cpu) default is read from `DEFAULT_PAIRS`, i.e. it is a pairing the
    default alloc actually captures - the key the oracle actually looks up.
    """
    gpu_id, cpu_id = DEFAULT_PAIRS[(group, index)]
    return {"request_id": request_id, "cpu_block": (cpu_id if cpu is None else cpu),
            "gpu_block": (gpu_id if gpu is None else gpu),
            "group": group, "span": span,
            "key": offered(index, group) if key is None else key}


@pytest.fixture(autouse=True)
def sample_every_entry(monkeypatch):
    """Every test sees every offered entry, unless it is the test ABOUT the rate."""
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 1)
    return


@pytest.fixture
def armed_env(monkeypatch):
    """Armed by the environment - the posture a real boot uses. Starts a real thread."""
    monkeypatch.setenv(config.ENV_DISK_RESTORE_VERIFY, "1")
    return


@pytest.fixture
def armed(armed_env, monkeypatch):
    """The same, with `_ensure_thread` answering yes WITHOUT starting a thread."""
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread", lambda: True)
    return


@pytest.fixture
def key_helpers(monkeypatch):
    """`vllm.v1.core.kv_cache_utils` carrying the two helpers, as the fork's does."""
    module = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    module.make_block_hash_with_group_id = pack
    module.resolve_block_hashes = resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, module)
    return module


@pytest.fixture
def wide_step(monkeypatch):
    """Retention and the per-pass cap lifted, for tests about SOMETHING ELSE."""
    monkeypatch.setattr(disk_restore_verify, "MAX_RETAINED_PER_STEP", 16)
    monkeypatch.setattr(disk_restore_verify, "MAX_CHECKS_PER_PASS", 16)
    return


@pytest.fixture
def wired(key_helpers):
    """The hook is REPORTED fitted, the way `disk_restore.install` tells it.

    The published `scheduler_oracle_hook` is not inferred, it is told - so every
    oracle test wires it explicitly through the same door `disk_restore
    ._install_alloc_observer` uses, and one test below leaves it untold to pin what
    an EngineCore WITHOUT the wrapper publishes.
    """
    disk_restore_verify.note_alloc_hook(True, None)
    return True


@pytest.fixture
def stage(wired):
    """The alloc stage: the default request's pinned hit, and no capture yet."""
    manager = Manager(hits={RID: pending()})
    return {"manager": manager, "request": Request(),
            "blocks": Blocks({0: TABLE[0], 1: TABLE[1]})}


def alloc(stage, *, manager=None, request=None, blocks=None, tokens=TOKENS) -> dict:
    """One `update_state_after_alloc` argument set, through the module's alloc door."""
    return disk_restore_verify.observe_alloc(
        manager if manager is not None else stage["manager"],
        request if request is not None else stage["request"],
        blocks if blocks is not None else stage["blocks"],
        tokens)


def see(entries, manager=None) -> dict:
    """One scheduler step's witness, through the module's emission door.

    The `scheduler_output` handed along is one that carries NOTHING the oracle
    could read - the honest stand-in for the production step whose
    `kv_connector_block_state` did not cover the request.
    """
    return disk_restore_verify.observe_offer(
        manager, Output(object()), entries)


def one_pass() -> dict:
    """One pass, driven by the test. There is no worker in this process.

    The monotonic floor between passes is a production guard (the box must not pay
    two device scans in half a second); a test driving two passes in one phase is
    the one caller that must not be throttled, so the rewind lives HERE - tests
    that want to prove the floor itself set `last_pass_mono` explicitly.
    """
    disk_restore_verify._state["last_pass_mono"] = None
    return disk_restore_verify.run_pass()


def counts() -> dict:
    return disk_restore_verify.counters()


def reasons() -> list:
    return disk_restore_verify.view()["reasons"]


def newest() -> dict:
    """The newest scheduler verdict: the ring's last element."""
    samples = disk_restore_verify.section()["scheduler_mismatch_samples"]
    assert samples, "no scheduler mismatch has been published"
    return samples[-1]


def step(stage, entries, **kwargs) -> dict:
    """THE canonical two-hook step: capture, then emit. Returns the pass verdict."""
    verdict = alloc(stage, **kwargs)
    see(entries, stage["manager"])
    passed = one_pass()
    passed["captured"] = verdict["captured"]
    return passed


# --- 1: the positive control, without which every test below is hollow -------
def test_the_alloc_hook_captures_the_pairing_the_fork_arguments_describe(armed, stage):
    """The hook reads a REAL argument set and captures exactly the fork's pairs.

    The default step: 6400 external tokens over a 3200-byte group (2 blocks) and a
    1600-byte group (4 blocks, `cdiv`), prefix-cacheable gate ON for both, no nulls.
    SIX pairings; the third CPU block of group 0 is past the trim and must NOT be
    captured. Then the emitted entry for one of the captured pairs is a
    `scheduler_equal` - the oracle CAN confirm, not merely refuse.
    """
    captured = alloc(stage)
    assert captured["captured"] == 6
    after = counts()
    assert after["scheduler_allocs_observed"] == 1
    assert after["scheduler_expectations_captured"] == 6
    assert after["scheduler_expectations_dropped"] == 0
    assert after["scheduler_expectations_consumed"] == 0
    now = disk_restore_verify.view()["scheduler_expectations_now"]
    assert now == 6
    verdict = see([entry()])
    assert verdict["retained"] == 1
    passed = one_pass()
    assert passed["scheduler_checked"] == 1
    assert passed["scheduler_results"][0]["equal"] is True
    final = counts()
    assert final["scheduler_equal"] == 1
    assert final["scheduler_mismatch"] == 0
    assert final["scheduler_expectations_consumed"] == 1
    assert final["scheduler_oracle_unavailable"] == 0


def test_the_index_comes_from_the_alloc_pairing_not_from_the_offer_order(
        armed, stage, wide_step):
    """THE POINT OF THE WHOLE ROLE, as a positive test.

    Six emitted entries in an order with nothing to do with the capture's, across
    both groups, each named for the block ITS pairing says. A walk that used the
    ENTRY ORDER as the block index would call at least two of these a mismatch;
    reading (group, index) from the captured pairing calls every one a match.
    """
    alloc(stage)
    pairs = [(0, 0), (1, 3), (0, 1), (1, 0), (1, 1), (1, 2)]
    entries = [entry(group=group, index=index, span=number)
               for number, (group, index) in enumerate(pairs)]
    # Every entry carries the captured (gpu, cpu) pair of ITS (group, index) - the
    # key the oracle looks up - while the ORDER of the list has nothing to do with
    # the capture's order.
    assert see(entries)["retained"] == len(entries)
    passed = one_pass()
    assert passed["scheduler_checked"] == len(entries)
    assert counts()["scheduler_equal"] == len(entries)
    assert counts()["scheduler_mismatch"] == 0


# --- 2: the negative controls, now driven through the real hook signature ----
def test_a_one_slot_shift_is_a_mismatch_and_names_the_neighbour(armed, stage):
    """THE BUG THIS ORACLE EXISTS FOR, at the alloc hook. A shift moves BOTH lists.

    Every list-vs-list check in this plugin agrees with itself under a shift: the
    CPU slot count matches the GPU block count, the ledger's identity list has the
    same length as the fork's, and the worker's byte comparison compares a file
    against the GPU block that file was named for. Here the fork paired the CPU
    slot with the request's block 1 (captured at alloc, from `blocks.
    get_block_ids()`), and the file offered for that pairing carries the name of
    block 2 - the re-derived key says so, by name and by sign.
    """
    step(stage, [entry(group=0, index=1, key=offered(2, 0))])
    passed = counts()
    # (`step` already ran the pass that produced the verdict)
    assert passed["scheduler_mismatch"] == 1
    assert passed["scheduler_mismatch_neighbour_index"] == 1
    assert passed["scheduler_mismatch_stable"] == 1
    assert passed["scheduler_equal"] == 0
    sample = newest()
    assert sample["index"] == 1
    assert sample["key16"] != sample["expected_key16"]
    assert sample["expected_key16"] == disk_layout.split_block_key(
        offered(1, 0))["key16"]
    assert any(line.startswith("scheduler_mismatch:neighbour_index")
               and "one-slot shift" in line for line in reasons())


def test_a_shift_the_other_way_carries_its_sign(armed, stage):
    """The offset is SIGNED, because "which way did it move" is the diagnosis.

    A shift one way and a shift the other way have different causes (an off-by-one
    in the walk that names the files versus one in the walk that pairs them), and a
    sample that said only "a neighbour" would leave an operator to re-derive which.
    """
    step(stage, [entry(group=0, index=1, key=offered(0, 0))])
    sample = newest()
    assert sample["classification"] == "neighbour_index"
    assert sample["offset"] == -1
    assert counts()["scheduler_mismatch_neighbour_index"] == 1


def test_a_mismatch_beyond_the_window_is_reported_as_unknown_not_guessed(armed, stage):
    """The window is BOUNDED, and a name outside it gets its own honest word."""
    foreign = bytes(pack(raw_digest(9999), 0)).hex()
    step(stage, [entry(group=0, index=1, key=foreign)])
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_unknown"] == 1
    assert after["scheduler_mismatch_neighbour_index"] == 0
    assert after["scheduler_mismatch_other_group"] == 0
    assert newest()["classification"] == "unknown"


# --- 3: the group swap, in both of its shapes --------------------------------
def test_a_name_whose_group_suffix_contradicts_the_pairing_is_named_first(
        armed, stage):
    """The suffix is a fact about OUR OWN offer and needs no engine input at all.

    `kv_cache_utils.py:83` carries the group inside the key, so a file name ending
    in `g1` for a pairing the FORK made in group 0 (captured at alloc) is a group
    swap on the path that named the file - caught before any re-derivation.
    """
    step(stage, [entry(group=0, index=1, key=offered(1, 0, suffix=1))])
    assert counts()["scheduler_mismatch_other_group"] == 1
    sample = newest()
    assert sample["name_group"] == 1 and sample["group"] == 0
    assert any("a group swap on our own offer path" in line for line in reasons())


def test_a_name_built_from_another_groups_view_is_caught_too(armed, stage):
    """THE SHAPE A SUFFIX CHECK CANNOT SEE, and the reason the digest is compared."""
    assert VIEW[G_SIZE[0]][1] != VIEW[G_SIZE[1]][1]      # the two views differ here
    step(stage, [entry(group=0, index=1, key=offered(1, 0, from_group=1))])
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_other_group"] == 1
    assert after["scheduler_mismatch_neighbour_index"] == 0
    assert newest()["classification"] == "other_group"
    assert any("a group swap: the name carries group 1's digest" in line
               for line in reasons())


# --- 4: THE PRODUCTION REGRESSION: a later step than the alloc ---------------
def test_a_load_emitted_in_a_later_step_than_its_alloc_still_gets_a_verdict(
        armed, stage):
    """THE 0.2.40 FAILURE, REPRODUCED AND BEATEN.

    Live reading, one EngineCore boot with traffic: `scheduler_checked = 0`,
    `scheduler_oracle_unavailable = 2`, the reason
    `block_state_does_not_cover_request: the request id is not in this step's
    kv_connector_block_state.req_ids`. The load state was emitted in a step later
    than the one that allocated it, the block table 0.2.40 read is CURRENT-step
    only, and every sampled entry was blind. Here the emission step is that step
    twice over - once with `kv_connector_block_state = None`, once with a channel
    that does NOT carry the request - and the verdict still arrives, because it
    was captured at alloc time. If this test ever "simplifies" back to reading the
    emission step's table, this is the one that must fail.
    """
    alloc(stage)
    # (a) the step carries NO connector channel at all
    disk_restore_verify.observe_offer(stage["manager"], Output(None), [entry()])
    # (b) the step HAS one, and it does not cover this request - the production shape
    disk_restore_verify.observe_offer(
        stage["manager"], Output(BlockState({"some-other-request": [TABLE[0]]})),
        [entry()])
    passed = one_pass()
    assert passed["scheduler_checked"] == 2
    after = counts()
    assert after["scheduler_equal"] == 2
    assert after["scheduler_mismatch"] == 0
    assert after["scheduler_oracle_unavailable"] == 0
    # And the expectation SURVIVED the first comparison: consumption is a mark, not
    # a deletion, because a load state keeps being emitted until the worker confirms.
    assert after["scheduler_expectations_consumed"] == 1
    assert disk_restore_verify.view()["scheduler_expectations_now"] == 6


def test_the_section_names_the_hook_that_produced_its_numbers(armed, stage):
    """One glance answers "which hook ran": the section says update_state_after_alloc.

    0.2.40 published counters whose provenance was a lie of omission - they came
    from an observation point that could not see its inputs. The published section
    now carries the hook's name, taken from what `disk_restore.install` REPORTED
    through `note_alloc_hook`, not from what this module believes about its caller.
    """
    step(stage, [entry()])
    hook = disk_restore_verify.section()["scheduler_oracle_hook"]
    assert "update_state_after_alloc" in hook
    assert "NOT WIRED" not in hook


def test_an_engine_without_the_alloc_wrapper_says_NOT_WIRED_IN_ONE_GLANCE(
        armed, key_helpers):
    """No wrapper -> the oracle CANNOT run -> the section says exactly that.

    The counter alone would not do this: an EngineCore that never saw a disk load
    and an EngineCore whose hook never fitted both sit at zero comparisons. The
    hook line separates them, because it reports the INSTALL, not the traffic.
    """
    assert counts()["scheduler_steps_seen"] == 0
    section = disk_restore_verify.section()
    assert "NOT WIRED" in section["scheduler_oracle_hook"]
    # And with the hook unwired, an emitted entry is a NAMED unavailability - the
    # green zero this release exists to make impossible stays impossible even here.
    see([entry()])
    one_pass()
    assert counts()["scheduler_oracle_unavailable"] == 1
    reason = disk_restore_verify.view()["scheduler_unreachable_reason"]
    assert reason.startswith("no_captured_expectation:")
    assert counts()["scheduler_checked"] == 0


# --- 5: the expectation lifecycle, counted at every edge ----------------------
def test_an_emitted_entry_with_no_captured_expectation_is_a_named_unavailability(
        armed, stage):
    """"We never captured this pairing" is a refusal to COMPARE, and stays a refusal.

    The tempting collapse is to treat a missing expectation as an unremarkable
    skip, because most pairings legitimately never emit (a CPU hit with no disk
    file). But an EMITTED entry whose pairing was never captured is the oracle
    going blind again - so it counts, names itself, and never becomes a mismatch.
    """
    alloc(stage)
    see([entry(gpu=42424, cpu=5001)])     # a GPU block no alloc ever paired
    one_pass()
    after = counts()
    assert after["scheduler_oracle_unavailable"] == 1
    assert after["scheduler_mismatch"] == 0
    assert after["scheduler_checked"] == 0
    reason = disk_restore_verify.view()["scheduler_unreachable_reason"]
    assert reason.startswith("no_captured_expectation:")


def test_the_prefix_cacheable_gate_bounds_the_capture(armed, stage):
    """The observer applies the fork's OWN gate or every capture is a lie about groups.

    `manager.py:441-443` skips groups outside `prefix_cacheable_group_ids`; with
    only group 1 in the set, a step whose hit covers both groups must capture FOUR
    pairings (group 1's four) and not six, and an emitted entry paired in the
    gated-out group is a named unavailability, not a verdict.
    """
    manager = Manager(cacheable=(1,), hits={RID: pending()})
    captured = alloc(stage, manager=manager)
    assert captured["captured"] == 4
    see([entry(group=0, index=1), entry(group=1, index=2)])
    one_pass()
    after = counts()
    assert after["scheduler_equal"] == 1
    assert after["scheduler_oracle_unavailable"] == 1     # the gated group 0 entry
    assert after["scheduler_mismatch"] == 0


def test_a_null_cpu_block_is_skipped_without_advancing_the_pairing(armed, stage):
    """The fork skips `is_null` while KEEPING THE INDEX; so must the capture.

    `manager.py:464-468` is the whole reason an align-mode boundary restore lands
    its one real state block on the run's LAST GPU block: a null placeholder
    consumes a table position but not a pairing. A capture that compacted the list
    would pair the real block with the FIRST GPU id - a wrong-offset restore, and
    the oracle would confirm the WRONG expectation with a green equal.
    """
    cpu = {0: [5001, 5002, 5003], 1: [6001, 6002, 6003, 6004, 6005]}
    manager = Manager(hits={RID: pending(cpu, nulls=((0, 0),))})
    captured = alloc(stage, manager=manager)
    assert captured["captured"] == 5          # group 0 lost index 0, kept index 1
    # The surviving group-0 pairing is (index 1 -> GPU 7002, CPU 5002): emitted
    # under its true name, it is a `scheduler_equal` with index 1, not index 0.
    see([entry(group=0, index=1)])
    one_pass()
    assert counts()["scheduler_equal"] == 1
    assert disk_restore_verify.section()["scheduler_oracle"]["equal"] == 1
    # And the null's would-be pairing was never captured at all.
    see([entry(group=0, index=0, gpu=TABLE[0][0], cpu=5001)])
    one_pass()
    assert counts()["scheduler_oracle_unavailable"] == 1


def test_the_cdiv_truncation_bounds_the_capture(armed, stage):
    """A scheduler that accepted FEWER tokens pairs fewer blocks; the capture follows.

    The fork takes `cdiv(num_external_tokens, g_block_size)` LEADING entries
    (`manager.py:445-448`). Half the default tokens halves the capture: 1 block of
    group 0, 2 of group 1 - and an entry paired at the third block of group 1 was
    never captured, because the fork never pairs it.
    """
    captured = alloc(stage, tokens=3200)
    assert captured["captured"] == 3
    see([entry(group=1, index=2, gpu=TABLE[1][2], cpu=CPU[1][2])])
    one_pass()
    assert counts()["scheduler_oracle_unavailable"] == 1


def test_the_per_alloc_bound_is_counted_as_a_bound(armed, stage, monkeypatch):
    """More pairings than `MAX_EXPECTATIONS_PER_ALLOC`: bound hit, counted."""
    monkeypatch.setattr(disk_restore_verify, "MAX_EXPECTATIONS_PER_ALLOC", 2)
    captured = alloc(stage)
    assert captured["captured"] == 2
    after = counts()
    assert after["scheduler_expectations_capped"] == 1
    assert after["scheduler_expectations_captured"] == 2


def test_the_map_drops_oldest_first_and_counts_the_unconsumed(
        armed, stage, monkeypatch):
    """The expectation map is BOUNDED, and an eviction of a pair nobody compared is
    the named fact `scheduler_expectations_dropped` - never a silent forget.

    A cap of two with one alloc of six pairings: every insert past the second
    evicts the oldest, none of them consumed, so FOUR drops and SIX captures - the
    map is a bound, and the counters tell the whole story of where the pairings
    went.
    """
    monkeypatch.setattr(disk_restore_verify, "MAX_SCHED_EXPECTATIONS", 2)
    assert alloc(stage)["captured"] == 6
    after = counts()
    assert after["scheduler_expectations_captured"] == 6
    assert after["scheduler_expectations_dropped"] == 4
    assert disk_restore_verify.view()["scheduler_expectations_now"] == 2


def test_a_consumed_pairing_evicts_without_the_dropped_counter(armed, stage,
                                                               monkeypatch):
    """The `consumed` MARK is why `dropped` means "nobody ever compared it".

    Without the mark, every evicted pairing that had been compared would count as
    dropped and the one counter that distinguishes "abandoned load" from "normal
    churn" would read as noise. With it: capture ONE pairing, compare it, evict it
    with a fresh one, and the drop count does not move.
    """
    monkeypatch.setattr(disk_restore_verify, "MAX_SCHED_EXPECTATIONS", 1)
    monkeypatch.setattr(disk_restore_verify, "MAX_EXPECTATIONS_PER_ALLOC", 1)
    alloc(stage, tokens=1600)     # cdiv(1600,3200)=1 -> one pairing of group 0
    assert counts()["scheduler_expectations_captured"] == 1
    see([entry(group=0, index=0, gpu=TABLE[0][0], cpu=CPU[0][0])])
    one_pass()
    assert counts()["scheduler_expectations_consumed"] == 1
    assert counts()["scheduler_expectations_dropped"] == 0
    # A NEW pairing arrives with the map full of the consumed one: it evicts, and
    # the eviction of a CONSUMED expectation must not count as a drop.
    stage["blocks"] = Blocks({0: [7101, 7102, 7103, 7104], 1: TABLE[1]})
    alloc(stage, tokens=1600)
    assert counts()["scheduler_expectations_captured"] == 2
    assert counts()["scheduler_expectations_dropped"] == 0


# --- 6: an oracle that cannot be built AT ALLOC TIME is said out loud ---------
def test_a_manager_that_exposes_no_hash_block_size_is_named_at_alloc(armed, stage):
    """A fork-shape change on the geometry fields is an unavailability, at the door."""
    broken = Manager(hash_block=0, hits={RID: pending()})
    captured = alloc(stage, manager=broken)
    assert captured["captured"] == 0
    assert counts()["scheduler_oracle_unavailable"] == 1
    assert disk_restore_verify.view()["scheduler_unreachable_reason"] \
        .startswith("no_hash_block_size:")


def test_a_request_carrying_no_block_hashes_is_named_at_alloc(armed, stage):
    """The request is an ARGUMENT here; without its hashes, nothing is derivable."""
    request = Request()
    del request.block_hashes
    captured = alloc(stage, request=request)
    assert captured["captured"] == 0
    assert counts()["scheduler_oracle_unavailable"] == 1
    reason = disk_restore_verify.view()["scheduler_unreachable_reason"]
    assert reason.startswith("alloc_no_block_hashes:")


def test_blocks_without_get_block_ids_are_a_refusal_with_their_own_name(armed, stage):
    """If the fork moved the tables out of the argument, the hook says so here."""
    captured = alloc(stage, blocks=object())
    assert captured["captured"] == 0
    assert counts()["scheduler_oracle_refused"] == 1
    assert any(line.startswith("scheduler_oracle:alloc_blocks_shape")
               for line in reasons())


def test_the_common_alloc_carries_no_external_tokens_and_moves_no_counter(armed, stage):
    """The scheduler calls this for EVERY request; the no-load majority costs NOTHING.

    `sched/scheduler.py:1228` calls `update_state_after_alloc` unconditionally, and
    the fork itself returns at once when `num_external_tokens == 0`
    (`manager.py:406-416`). A counter for that would be a counter per scheduled
    request, which is noise, not information - so this asserts the ABSENCE.
    """
    assert alloc(stage, tokens=0) == {"captured": 0, "reason": "no_external_tokens"}
    assert alloc(stage, tokens=None) == {"captured": 0,
                                         "reason": "no_external_tokens"}
    empty = Manager(hits={})
    assert alloc(stage, manager=empty) == {"captured": 0,
                                           "reason": "no_pending_cpu_hit"}
    after = counts()
    assert all(value == 0 for value in after.values())


def test_a_hostile_alloc_and_a_hostile_entry_never_raise_into_the_step(armed, stage):
    """Both doors are on engine threads and must not escape. Garbage is counted."""
    assert alloc(stage, request=object())["reason"] == "no_request_id"
    assert counts()["scheduler_oracle_refused"] == 1
    hostile = ["not-a-dict", 7, {"group": "zero"}, {},
               {"request_id": RID, "group": 0, "gpu_block": "x", "key": "zz"}]
    verdict = see(hostile)
    assert verdict["seen"] == len(hostile)
    one_pass()
    after = counts()
    assert after["scheduler_oracle_refused"] >= 3
    assert after["scheduler_errors"] == 0
    assert after["scheduler_mismatch"] == 0


# --- 7: what the engine threads pay, and where the work happens ---------------
def test_the_step_pays_nothing_at_all_when_the_verifier_is_off(monkeypatch, stage):
    """DEFAULT OFF means OFF: no peek at the pending map, no manager read, no counter.

    Proved by instrumenting the two places this role touches foreign state.
    """
    def refuse(*args, **kwargs):
        raise AssertionError("the step reached the oracle while it was OFF")

    monkeypatch.setattr(disk_restore_verify, "_scheduler_sizes", refuse)
    monkeypatch.setattr(disk_restore_verify, "_peek_pending", refuse)
    assert disk_restore_verify.effective_enabled() is False
    assert alloc(stage) == {"captured": 0, "reason": "disabled"}
    assert see([entry(), entry(group=1)]) == {"seen": 0, "retained": 0,
                                              "reason": "disabled"}
    assert all(value == 0 for value in counts().values())


def test_the_capture_hashes_nothing_until_the_pass_runs(armed, stage, wide_step):
    """The engine threads run ints; the hashing runs on the module's own thread.

    Instrumented, not asserted from a docstring: `resolve_block_hashes` from the
    fake kv_cache_utils counts its calls, and none of them may happen until
    `run_pass` is on the clock.
    """
    module = sys.modules[disk_restore.KV_UTILS_MODULE]
    real = module.resolve_block_hashes
    calls = []

    def counting(block_hashes, hash_block_size, block_size):
        calls.append(block_size)
        return real(block_hashes, hash_block_size, block_size)

    module.resolve_block_hashes = counting
    alloc(stage)
    see([entry(), entry(group=1, index=2)])
    assert calls == [], "the step thread hashed"
    one_pass()
    assert calls, "the thread never hashed"


def test_an_unsampled_step_costs_only_counters(armed, stage, monkeypatch):
    """The sampling decision precedes every expectation lookup: the rate is a cost."""
    alloc(stage)
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 4)
    see([entry(index=index) for index in (0, 1)] +
        [entry(group=1, index=index) for index in (0, 1)])
    after = counts()
    assert after["scheduler_entries_seen"] == 4
    assert after["scheduler_sampled_out"] == 3
    assert after["scheduler_retained"] == 1


def test_no_thread_drops_the_emission_and_says_so_and_changes_no_offer(armed, stage,
                                                                       monkeypatch):
    """A record with no thread to read it is refused NOW, on the spot, and named."""
    alloc(stage)
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread", lambda: False)
    verdict = disk_restore_verify.observe_offer(stage["manager"], Output(object()),
                                                [entry()])
    assert verdict["retained"] == 0
    after = counts()
    assert after["scheduler_retained"] == 1
    assert after["scheduler_thread_unavailable"] == 1
    assert disk_restore_verify.section()["scheduler_jobs_now"] == 0
    assert one_pass()["scheduler_checked"] == 0


def test_the_thread_drains_offered_entries_without_being_asked(armed_env, key_helpers,
                                                               wired):
    """The real thread: captured on the step thread, verified on the module's own."""
    stage_manager = Manager(hits={RID: pending()})
    disk_restore_verify.observe_alloc(stage_manager, Request(),
                                      Blocks({0: TABLE[0], 1: TABLE[1]}), TOKENS)
    assert counts()["thread_started"] == 0
    verdict = disk_restore_verify.observe_offer(
        stage_manager, Output(object()),
        [entry(group=0, index=1, key=offered(2, 0))])
    assert verdict["retained"] == 1
    assert counts()["thread_started"] == 1
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not counts()["scheduler_mismatch"]:
        time.sleep(0.01)
    after = counts()
    assert after["scheduler_mismatch_neighbour_index"] == 1, "the thread never verified"
    assert disk_restore_verify.section()["scheduler_jobs_now"] == 0
    assert disk_restore_verify.status()["thread_alive"] is True
    assert any(thread.name == "kvtier-disk-restore-verify"
               for thread in threading.enumerate())
    disk_restore_verify.stop_thread()


def test_a_pass_with_no_worker_still_delivers_the_scheduler_verdicts(armed, stage):
    """A shutdown that takes the worker cannot strand the pairing verdicts."""
    step(stage, [entry(group=0, index=1, key=offered(2, 0))])
    # The pass inside `step` already ran with no worker registered:
    assert counts()["worker_gone"] == 1
    assert counts()["scheduler_mismatch_neighbour_index"] == 1


# --- 8: the stability witness and the sample ring -----------------------------
def test_the_stability_witness_refuses_a_moving_view_as_evidence(armed, stage,
                                                                 monkeypatch):
    """A re-derivation that disagrees with itself is a race, not a layout candidate."""
    real_resolve = resolve
    answers = [bytes.fromhex(offered(3, 0))[:32], bytes.fromhex(offered(2, 0))[:32]]

    def shifting(block_hashes, hash_block_size, block_size):
        out = list(real_resolve(block_hashes, hash_block_size, block_size))
        if answers and block_size == G_SIZE[0]:
            digest = answers.pop(0)
            out = [digest if index == 1 else item
                   for index, item in enumerate(out)]
        return out

    module = sys.modules[disk_restore.KV_UTILS_MODULE]
    monkeypatch.setattr(module, "resolve_block_hashes", shifting)
    step(stage, [entry(group=0, index=1, key=offered(2, 0))])
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_unstable"] == 1
    assert after["scheduler_mismatch_stable"] == 0
    assert newest()["stable"] is False


def test_the_mismatch_sample_ring_is_bounded_and_keeps_the_newest(armed, stage,
                                                                 wide_step,
                                                                 monkeypatch):
    """A step with three bad names publishes the last two, and counts all three."""
    monkeypatch.setattr(disk_restore_verify, "MAX_SCHED_MISMATCH_SAMPLES", 2)
    alloc(stage)
    entries = [entry(group=0, index=index, key=offered(index + 1, 0))
               for index in range(2)]
    entries.append(entry(group=1, index=1, key=offered(3, 1)))
    see(entries)
    one_pass()
    after = counts()
    assert after["scheduler_mismatch"] == 3, "every finding is counted"
    ring = disk_restore_verify.view()["scheduler_mismatch_samples"]
    assert len(ring) == 2, "the ring holds the NEWEST two and drops the oldest"
    published = disk_restore_verify.section()
    assert published["max_scheduler_mismatch_samples"] == 2
    # THE RING MAY TRIM FURTHER INSIDE THE PUBLISHED SECTION (the ladder sheds
    # illustration before explanation, and this section is one sample's width from
    # two samples plus reasons) - but the NEWEST finding always survives, and the
    # counters saw all three.
    shown = published["scheduler_mismatch_samples"]
    assert shown, "the newest finding must survive the ladder"
    assert shown[-1] == ring[-1]
    assert all(item in ring for item in shown)


# --- 9: the published shape, now naming its hook ------------------------------
def test_the_section_publishes_both_roles_and_never_changes_its_key_set(armed, stage):
    """One section, two roles, one key set - fresh, after a step, after a mismatch, fat.

    Five states are walked here, each one a way the old module could change shape
    under a reader: nothing seen, expectation captured but unconsumed, a verdict, a
    mismatch, and the oracle-unavailable path.
    """
    fresh = disk_restore_verify.section()
    assert fresh["roles"]["worker"] == disk_restore_verify.WORKER_ROLE
    assert fresh["roles"]["scheduler"] == disk_restore_verify.SCHEDULER_ROLE
    assert "never add the two" in fresh["roles"]["meaning"]
    assert "means the check is NOT running" in fresh["oracle_rule"]
    assert fresh["roles"]["scheduler_offer_point"] is False
    assert fresh["scheduler_oracle"]["unavailable"] == 0
    assert fresh["env_var"] == config.ENV_DISK_RESTORE_VERIFY
    # state 2: captured, unconsumed
    alloc(stage)
    captured = disk_restore_verify.section()
    assert captured["scheduler_expectations_now"] == 6
    assert captured["scheduler_expectations_captured"] == 6
    # state 3+4: a verdict, and a mismatch
    see([entry(group=0, index=1, key=offered(2, 0))])
    one_pass()
    moved = disk_restore_verify.section()
    assert moved["roles"]["scheduler_offer_point"] is True
    assert moved["scheduler_mismatch_samples"]
    # state 5: the unavailable path
    see([entry(group=0, index=0, gpu=42424, cpu=5001)])
    one_pass()
    for key in disk_restore_verify.scheduler_counters():
        assert key in moved
    published = disk_restore_verify.scheduler_counters()
    assert all(key.startswith("scheduler_") for key in published)
    assert set(published) == {key for key in fresh
                              if key.startswith("scheduler_")
                              and key in disk_restore_verify.counters()}
    # THE FAT PAYLOAD: the same keys, and still inside the cap the seam REFUSES at.
    disk_restore_verify._state["reasons"] = [
        f"tag{index}: " + "a persistent condition described at some length " * 4
        for index in range(disk_restore_verify.MAX_REASON_SAMPLES)]
    disk_restore_verify._state["boundary_missing_groups"].update(
        {index: 1 for index in range(40)})
    fat = disk_restore_verify.section()
    assert set(fat) == set(fresh)
    assert fat["roles"]["worker"] == disk_restore_verify.WORKER_ROLE, \
        "the role sentences are shed LAST: they are what the split exists to publish"
    for payload in (fresh, captured, moved, fat):
        encoded = json.dumps(payload, separators=(",", ":"), default=str)
        assert len(encoded.encode()) <= disk_restore_verify.MAX_SECTION_BYTES
        assert len(encoded.encode()) <= worker_diag.MAX_CONTRIBUTION_BYTES


def test_the_scheduler_role_extends_the_existing_section_rather_than_a_new_one():
    """The seam holds FIVE contributions and its allow-list has exactly five names."""
    assert worker_diag.MAX_CONTRIBUTIONS == 5
    state = worker_diag.contributions()
    assert len(state["allow_list"]) == worker_diag.MAX_CONTRIBUTIONS
    assert "disk_restore_verify" in state["allow_list"]
    assert len(state["registered"]) <= worker_diag.MAX_CONTRIBUTIONS
    assert not any("scheduler" in name for name in state["allow_list"])
    assert disk_restore_verify.MAX_SECTION_BYTES \
        <= worker_diag.MAX_CONTRIBUTION_BYTES


def test_the_snapshot_publishes_the_section_the_scheduler_answers_with(armed, stage):
    """`/snapshot` is the EngineCore's door to this role; it must carry the numbers."""
    step(stage, [entry(group=0, index=1, key=offered(2, 0))])
    out = snapshot.dev_state_sections()
    assert out["disk_restore_verify_enabled"] is True
    block = out["disk_restore_verify"]
    assert block["scheduler_mismatch_neighbour_index"] == 1
    assert "update_state_after_alloc" in block["scheduler_oracle_hook"]
    assert block["roles"]["scheduler_offer_point"] is True


def test_the_status_verb_reports_both_roles_with_the_same_numbers_as_the_section(
        armed, stage):
    """The verb is the EngineCore's only live door; it cannot disagree with the file."""
    step(stage, [entry(group=0, index=1, key=offered(2, 0))])
    status = disk_restore_verify.status()
    assert status["ok"] is True
    assert status["scheduler_mismatch_neighbour_index"] == 1
    assert status["scheduler_oracle_hook"] == \
        disk_restore_verify.section()["scheduler_oracle_hook"]


def test_the_witness_never_mutates_the_entries_it_is_handed(armed, stage, wide_step):
    """The witness is a reader. If it ever edited one, it would be part of the step."""
    entries = [entry(group=0, index=index) for index in range(2)] + \
              [entry(group=1, index=index) for index in range(4)]
    before = copy.deepcopy(entries)
    alloc(stage)
    see(entries)
    one_pass()
    assert entries == before
    # The pending hit the hook peeked must survive the peek exactly as the fork
    # left it to: the peek is the ONLY read of it before the fork pops it.
    retained_hits = stage["manager"]._pending_cpu_hits[RID]
    assert [block.block_id for block in retained_hits[0][0]] == CPU[0]


# --- the hook site: the real wrapper, not an assumed one ----------------------
class SimpleCPUOffloadScheduler(OfferManager):
    """A fork class with the REAL method names, so `disk_restore.install` fits for real.

    `update_state_after_alloc` below is the fork's pairing loop (`manager.py:404-489`)
    with the two pool `touch` calls dropped (this fake's pools have no touch, and an
    observer that never reaches the pools cannot care); everything the pairing decides
    - the pop, the gate, the cdiv trim, the start offset, the null skip - is verbatim.
    """

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)
        pending = self._pending_cpu_hits.pop(req_id, None)
        if num_external_tokens == 0 or pending is None:
            return
        cpu_hit_blocks_full, _, num_computed_tokens = pending
        cpu_hit_blocks = []
        for group in range(num_groups):
            if group not in self.prefix_cacheable_group_ids:
                cpu_hit_blocks.append([])
                continue
            group_size = self.group_block_sizes[group]
            take = -(-num_external_tokens // group_size)
            cpu_hit_blocks.append(cpu_hit_blocks_full[group][:take])
        gpu_block_ids, cpu_block_ids = [], []
        for group in range(num_groups):
            blocks_g = cpu_hit_blocks[group]
            if not blocks_g:
                continue
            group_size = self.group_block_sizes[group]
            gpu_ext_start = num_computed_tokens // group_size
            group_gpu_ids = block_ids_by_group[group]
            for index, cpu_block in enumerate(blocks_g):
                if cpu_block.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + index])
                cpu_block_ids.append(cpu_block.block_id)
        self._free_pending_cpu_hit((cpu_hit_blocks_full, 0, num_computed_tokens))
        self._reqs_to_load[req_id] = _Transfer(gpu_block_ids, cpu_block_ids)


class _Transfer:
    """`TransferMeta`, the two index-aligned lists."""

    def __init__(self, gpu, cpu):
        self.gpu_block_ids = list(gpu)
        self.cpu_block_ids = list(cpu)


@pytest.fixture
def offer_stage(armed, key_helpers, monkeypatch, tmp_path):
    """The OFFER path armed, exactly as `test_disk_restore.armed` arms it.

    Built on `armed` (not the bare environment) because these tests drive the
    wrapper directly and then run the pass themselves: with a live thread the
    thread and the test would race over one retained record.
    """
    root = tmp_path / "spill"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    # THE INSTALL RESOLVES ITS CLASS BY NAME FIRST (`_sched_class`), and an earlier
    # suite's fake `vllm.v1.simple_kv_offload.manager` would win that lookup and
    # take the install - silently wrapping a class this test never calls. The whole
    # point of the test is the REAL wrapper on THIS class, so the name-based door is
    # closed here, explicitly.
    monkeypatch.delitem(sys.modules, disk_restore.SCHED_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, disk_restore.WORKER_MODULE, raising=False)
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(disk_spill, "geometry_digest", lambda: DIGEST)
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": True, "count": 1,
                                             "ranks": ["PP0_TP0"], "reason": None,
                                             "ranks_armed": 1, "ranks_seen": 1,
                                             "reads_applied": 0, "reason_kind": None})
    monkeypatch.setattr(disk_store, "counters", lambda: {
        "geometry_digest": DIGEST,
        "fingerprint": {"total_bytes_per_block": WIDTH}, "rank": "PP0_TP0"})
    disk_restore.set_disk_authority(True)
    return root


def test_the_installed_wrapper_pairs_confirms_and_catches_a_shift(offer_stage,
                                                                  wide_step):
    """`install` -> wrapped `update_state_after_alloc` -> the fork's loop -> paths out.

    Nothing staged: the identities come out of `disk_restore.offer`, the CPU slots
    out of its own allocations, the GPU ids out of a `KVCacheBlocks` fake, and the
    metadata is built FROM THE FORK BODY'S OWN PAIRING LISTS. Then the same step is
    replayed with every GPU pairing rotated by one block - names unchanged, pairing
    moved - and the oracle has to call every one a shift. This is the test that the
    wrapper is where the install says it is, and `scheduler_oracle_hook` the test
    that it REPORTS the truth.
    """
    rid = "req-installed-1"
    blocks = 2
    fill(offer_stage, blocks=blocks)
    disk_restore.install(sched_cls=SimpleCPUOffloadScheduler)
    try:
        manager = SimpleCPUOffloadScheduler()
        request = Req(request_id=rid, blocks=blocks)
        tokens, _async = offer(manager, request)
        assert tokens == blocks * BLOCK
        per_group = {}
        position = 0
        for group in range(5):
            ids = []
            if group in CACHEABLE:
                ids = [9000 + position + step for step in range(blocks)]
                position += blocks
            per_group[group] = ids
        gpu_blocks = Blocks(per_group)
        # THE WRAPPED METHOD, called the way the scheduler calls it.
        manager.update_state_after_alloc(request, gpu_blocks, tokens)
        transfer = manager._reqs_to_load[rid]
        metadata = Meta(load_event=7, cpu=transfer.cpu_block_ids,
                        gpu=transfer.gpu_block_ids, reqs={7: [rid]})
        report = disk_restore.build_load_paths(
            manager, metadata, Output(BlockState({})))
        assert report["ok"] is True
        assert report["named"] == len(CACHEABLE) * blocks
        hook = disk_restore_verify.section()["scheduler_oracle_hook"]
        assert "update_state_after_alloc" in hook and "NOT WIRED" not in hook
        counts_before = disk_restore_verify.counters()
        assert counts_before["scheduler_expectations_captured"] == \
            len(CACHEABLE) * blocks
        one_pass()
        after = disk_restore_verify.counters()
        assert after["scheduler_equal"] == report["named"], after
        assert after["scheduler_mismatch"] == 0

        # THE CONTROL, ON THE SAME REAL NAMES: the files did not move, the PAIRING
        # did. The pending hit's per-group CPU lists are SWAPPED before the wrapped
        # alloc runs, so the fork pairs the block-2 slot with the request's FIRST
        # GPU position and vice versa - the exact one-slot shift the worker's byte
        # check cannot see (it compares each file against the block ITS OWN name
        # names) and the exact shift a list-vs-list check is blind to (both lists
        # keep their length and their alignment with each other). NO reset between
        # phases: the verdicts must read as a DELTA on one live boot, the way an
        # operator reads them, and a reset would force this test to re-stamp the
        # install verdict by hand - which is exactly the cover-up the hook line
        # exists to prevent.
        rid2 = "req-installed-2"
        manager = SimpleCPUOffloadScheduler()
        request = Req(request_id=rid2, blocks=blocks)
        offer(manager, request)
        for group in CACHEABLE:
            hit = manager._pending_cpu_hits[rid2][0][group]
            hit[0], hit[1] = hit[1], hit[0]
        manager.update_state_after_alloc(request, Blocks(per_group), tokens)
        transfer = manager._reqs_to_load[rid2]
        metadata = Meta(load_event=8, cpu=transfer.cpu_block_ids,
                        gpu=transfer.gpu_block_ids, reqs={8: [rid2]})
        report = disk_restore.build_load_paths(
            manager, metadata, Output(None))     # the later-step shape, again
        assert report["named"] == len(CACHEABLE) * blocks
        one_pass()
        moved = disk_restore_verify.counters()
        assert moved["scheduler_mismatch"] - after["scheduler_mismatch"] == \
            report["named"], moved
        assert moved["scheduler_mismatch_neighbour_index"] - \
            after["scheduler_mismatch_neighbour_index"] == report["named"]
        assert moved["scheduler_equal"] == after["scheduler_equal"]
    finally:
        disk_restore.uninstall()
        # THE ROUND TRIP IS THE ASSERTION: uninstall restores the methods AND
        # retracts the hook claim, so a reader who uninstalled never reads "fitted".
        assert "NOT WIRED" in disk_restore_verify.section()["scheduler_oracle_hook"]
        disk_restore.reset_state_for_tests()


def test_the_installed_wrapper_with_the_verifier_off_observes_nothing(offer_stage,
                                                                     monkeypatch):
    """The same install, disarmed: the SAME paths out, and no scheduler counter moved.

    The alloc wrapper is on the step path of EVERY request; this is the proof that
    unarmed, it is invisible - identical paths, identical fork behaviour, and the
    one honest counter it is allowed (none) stays at zero.
    """
    monkeypatch.delenv(config.ENV_DISK_RESTORE_VERIFY)
    rid = "req-disarmed-1"
    fill(offer_stage, blocks=2)
    disk_restore.install(sched_cls=SimpleCPUOffloadScheduler)
    try:
        manager = SimpleCPUOffloadScheduler()
        request = Req(request_id=rid, blocks=2)
        tokens, _async = offer(manager, request)
        per_group = {group: ([9000 + 2 * index + step for step in range(2)]
                             if group in CACHEABLE else [])
                     for index, group in enumerate(CACHEABLE)}
        rest = {group: per_group.get(group, []) for group in range(5)}
        manager.update_state_after_alloc(request, Blocks(rest), tokens)
        transfer = manager._reqs_to_load[rid]
        metadata = Meta(load_event=9, cpu=transfer.cpu_block_ids,
                        gpu=transfer.gpu_block_ids, reqs={9: [rid]})
        report = disk_restore.build_load_paths(manager, metadata, Output(None))
        assert report["ok"] is True
        assert report["named"] == len(CACHEABLE) * 2
        assert all(value == 0 for value in
                   disk_restore_verify.counters().values())
    finally:
        disk_restore.uninstall()
        disk_restore.reset_state_for_tests()


# --- 10: THE FORK'S LAZY HASH VIEW (0.2.42): THE SHAPE THAT WAS REFUSED -------
# Production, after the 0.2.41 alloc hook fixed the observation point:
# `scheduler_oracle_unavailable = 0`, `scheduler_expectations_consumed = 2`, and
# `scheduler_checked = 0` - because the last thing the oracle did was REFUSE the
# engine's answer: `resolve_shape: resolve_block_hashes answered
# BlockHashListWithBlockSize, not a sequence`. It is not a sequence. `BlockHashList` is
# the UNION `list[BlockHash] | BlockHashListWithBlockSize`
# (`vllm/v1/core/kv_cache_utils.py:2850`) and the second member is a lazy view that
# inherits from nothing (`:2783-2848`). Refusing an unrecognised shape was right; the
# recognised list was too short.
#
# THE ARITHMETIC THIS SECTION EXISTS TO PIN. `hash_block_size` on the live box is 40
# against `group_block_sizes = [3200, 3200, 3200, 3200, 8, 3200]`, so a serving group's
# view has `scale_factor = 3200 / 40 = 80`: `view[i]` is `raw[i*80 + 79]` and
# `len(view)` is `len(raw) // 80`. Indexing the raw list at `i` instead compares the
# digest of tokens `[0, 40)` against a file named for tokens `[0, 3200)` - a DIFFERENT
# TOKEN RANGE, which reads either as a storm of false mismatches or as an accidental
# agreement that looks like a passed safety check. `test_the_digest_compared_is_the
# views_and_never_the_raw_list` is the test that makes that swap fail loudly, and
# `test_the_views_collapsed_length_is_the_bound_not_the_raw_lists` is its twin.
LIVE_HASH_BLOCK = 40                                  # `tier.hash_block_size`, the box
LIVE_SIZES = [3200, 3200, 3200, 3200, 8, 3200]   # `tier.group_block_sizes`, the box
LIVE_CACHEABLE = (0, 1, 2, 3, 5)                      # group 4 is parked, never spilled
LIVE_SCALE = 3200 // LIVE_HASH_BLOCK                  # 80: this box's real ratio
LIVE_TOKENS = 6400                                    # -> 2 blocks per 3200-token group
LIVE_RID = "req-live-1"
LIVE_RAW = [raw_digest(seed) for seed in range(LIVE_SCALE * 8)]     # 8 view entries
LIVE_GPU = {group: [7000 + 100 * group + i for i in range(12)] for group in range(6)}
LIVE_CPU = {group: [5000 + 100 * group + i for i in range(12)] for group in range(6)}

# Every read of the underlying list the fake views have made, newest last. The ONLY way
# to tell "indexed the view" from "indexed the list the view reads" without trusting the
# module's own words.
LIVE_READS: list = []


class LiveHashView:
    """`BlockHashListWithBlockSize`, REPLICATED - never imported.

    Fork: `vllm/v1/core/kv_cache_utils.py:2783-2848`, whose whole protocol is
    `__len__ = len(block_hashes) // scale_factor`, `__getitem__(int) =
    _get_value_at(idx)` and `_get_value_at(idx) =
    block_hashes[(idx + 1) * scale_factor - 1]`, with `__init__` asserting
    `target_block_size % hash_block_size == 0`. It is deliberately NOT a `Sequence`,
    NOT registered as one, and NOT a list or tuple: those three facts are the shape the
    oracle had to learn to accept, and a fake that inherited `Sequence` would test
    something the fork is not. `_get_value_at` logs the index it read so a test can see
    the arithmetic rather than infer it.
    """

    def __init__(self, block_hashes, hash_block_size, target_block_size) -> None:
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]
        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self):
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> bytes:
        raw_index = (idx + 1) * self.scale_factor - 1
        LIVE_READS.append(raw_index)
        return self.block_hashes[raw_index]


def live_resolve(block_hashes, hash_block_size, block_size):
    """`resolve_block_hashes` (`kv_cache_utils.py:2859-2891`), replicated.

    Same three answers as the fork: the raw list when the sizes are equal, the SAME view
    when handed a view, and a new view otherwise - behind the fork's
    `assert block_size % hash_block_size == 0`, which is what raises `AssertionError`
    for group 4's 8-token ring against a 40-token hash.
    """
    if block_size == hash_block_size:
        return block_hashes
    if isinstance(block_hashes, LiveHashView):
        assert block_hashes.scale_factor == block_size // hash_block_size
        return block_hashes
    assert block_size % hash_block_size == 0
    return LiveHashView(block_hashes, hash_block_size, block_size)


def live_view(index: int, group: int, sizes=None) -> LiveHashView:
    """The live view for `group`, at the fork's own construction."""
    return LiveHashView(LIVE_RAW, LIVE_HASH_BLOCK,
                        (LIVE_SIZES if sizes is None else sizes)[group])


LIVE_PAIRS = {(group, index): (LIVE_GPU[group][index], LIVE_CPU[group][index])
              for group in range(6) for index in range(2)}


def live_offered(index: int, group: int, *, sizes=None, suffix=None,
                 from_group=None) -> str:
    """The name for block `index` of `group`, read out of the VIEW, not the list."""
    use = LIVE_SIZES if sizes is None else sizes
    source = group if from_group is None else from_group
    return bytes(pack(LiveHashView(LIVE_RAW, LIVE_HASH_BLOCK, use[source])[index],
                      group if suffix is None else suffix)).hex()


def raw_indexed(index: int, group: int) -> str:
    """The name a raw-LIST-indexing oracle would offer: `raw[index]`, nothing else.

    At scale 80 this is the digest for tokens `[0, 40)` while the file name claims
    tokens `[0, 3200)`. It is the trap, written down as a fixture.
    """
    return bytes(pack(LIVE_RAW[index], group)).hex()


def live_entry(*, group=0, index=1, key=None, sizes=None, gpu=None,
               cpu=None) -> dict:
    """One offered entry for the live geometry, correct unless `key` says otherwise.

    `gpu`/`cpu` override the DEFAULT pairing, which a test that shifts
    `num_computed_tokens` needs: the pairing the fork makes is then not the one
    `LIVE_PAIRS` holds, and the oracle's whole authority comes from looking up the
    pairing the fork actually made.
    """
    gpu_id, cpu_id = LIVE_PAIRS[(group, index)]
    return {"request_id": LIVE_RID,
            "cpu_block": (cpu_id if cpu is None else cpu),
            "gpu_block": (gpu_id if gpu is None else gpu),
            "group": group, "span": 0,
            "key": (live_offered(index, group, sizes=sizes) if key is None else key)}


@pytest.fixture
def live_stage(armed, wired, monkeypatch):
    """The box's geometry, a `resolve_block_hashes` that answers with the LAZY VIEW."""
    module = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    module.make_block_hash_with_group_id = pack
    module.resolve_block_hashes = live_resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, module)
    manager = Manager(hash_block=LIVE_HASH_BLOCK, sizes=LIVE_SIZES,
                      cacheable=LIVE_CACHEABLE,
                      hits={LIVE_RID: pending(cpu={group: LIVE_CPU[group]
                                                   for group in range(6)})})
    LIVE_READS.clear()
    return {"manager": manager,
            "request": Request(request_id=LIVE_RID, hashes=LIVE_RAW),
            "blocks": Blocks({group: LIVE_GPU[group] for group in range(6)})}


def live_step(stage, entries, tokens=LIVE_TOKENS, computed=0) -> dict:
    """Capture at the alloc hook, emit the offer, run the pass. One box-shaped step.

    THE READ LOG IS CLEARED HERE, AFTER the caller built its entry names: what
    `LIVE_READS` must show is the reads THE ORACLE made, and a fixture that built its
    own digest out of a view is not the oracle reading one.
    """
    stage["manager"]._pending_cpu_hits[LIVE_RID] = pending(
        cpu={group: LIVE_CPU[group] for group in range(6)}, computed=computed)
    LIVE_READS.clear()
    alloc(stage, tokens=tokens)
    see(entries, stage["manager"])
    return one_pass()


def key16_of(key_hex: str) -> str:
    return disk_layout.split_block_key(key_hex)["key16"]


def test_the_fakers_lazy_view_is_not_a_list_a_tuple_or_a_sequence():
    """THE PREMISE, tested before anything is concluded from it.

    Every test in this section rests on the fake having the shape the fork's has. If
    someone "simplifies" it into a list, or gives it a `Sequence` base, this fails and
    the section below stops meaning anything - which is the point.
    """
    import collections.abc

    view = live_view(1, 0)
    assert view.scale_factor == LIVE_SCALE == 80
    assert not isinstance(view, (list, tuple)), \
        "a list-shaped fake tests the 0.2.40 code path, not the one that broke"
    assert not isinstance(view, collections.abc.Sequence), \
        "the fork's view is NOT a Sequence; a fake that is one hides the bug"
    assert len(view) == len(LIVE_RAW) // LIVE_SCALE
    assert list(view)[1] == view[1]


def test_the_lazy_hash_view_is_compared_not_refused(live_stage):
    """THE PRODUCTION REGRESSION. A lazy non-`Sequence` view must produce a verdict.

    0.2.41 in production: `scheduler_expectations_consumed = 2`,
    `scheduler_oracle_refused = 2` with reason `resolve_shape`, `scheduler_checked = 0`.
    The view arrived, this module did not recognise it, and the witness ran decorative.
    """
    assert key16_of(live_offered(1, 0)) != key16_of(raw_indexed(1, 0)), \
        "the fixture has stopped distinguishing view-indexing from raw-indexing"
    live_step(live_stage, [live_entry(group=0, index=1)])
    passed = counts()
    assert passed["scheduler_checked"] == 1, \
        "a view this module cannot index is the 0.2.41 failure, back again"
    assert passed["scheduler_equal"] == 1
    assert passed["scheduler_oracle_refused"] == 0
    assert passed["scheduler_oracle_unavailable"] == 0
    assert passed["scheduler_mismatch"] == 0
    assert passed["scheduler_errors"] == 0


def test_the_digest_compared_is_the_views_and_never_the_raw_list(live_stage):
    """THE SWAP THAT WOULD LOOK LIKE A SAFETY CHECK, pinned from both directions.

    A "just accept the view and index it" fix that indexes the list underneath it does
    not crash and does not log anything odd: at scale 80 it compares the digest of
    tokens `[0, 40)` against a name covering tokens `[0, 3200)`. So this test pins the
    DIGEST THE ORACLE COMPARED, twice over - the correct pairing must verify (which a
    raw-indexing oracle would report as a mismatch), and the raw-indexed name must be
    REFUSED AS A MISMATCH whose expected digest is the VIEW's (which a raw-indexing
    oracle would report as an agreement). Either swap breaks one half, loudly.
    """
    # Half one: the view's digest is the one that verifies.
    live_step(live_stage, [live_entry(group=0, index=1)])
    assert counts()["scheduler_equal"] == 1
    assert LIVE_READS == [1 * LIVE_SCALE + LIVE_SCALE - 1], \
        "the oracle must read the underlying list THROUGH the view's arithmetic: " \
        f"view[1] is raw[159] at scale {LIVE_SCALE}, and the reads were {LIVE_READS}"
    AT_VIEW_BOUNDARIES = {(i + 1) * LIVE_SCALE - 1 for i in range(8)}

    # Half two: the name built from `raw[1]` is a mismatch, and the oracle says WHAT it
    # expected - the view's digest, by name.
    LIVE_READS.clear()
    live_step(live_stage, [live_entry(group=0, index=1, key=raw_indexed(1, 0))])
    passed = counts()
    assert passed["scheduler_mismatch"] == 1
    assert passed["scheduler_equal"] == 1, "the raw-indexed name must NOT agree"
    sample = newest()
    assert sample["key16"] == key16_of(raw_indexed(1, 0))
    assert sample["expected_key16"] == key16_of(live_offered(1, 0))
    assert sample["expected_key16"] != sample["key16"]
    assert 1 not in LIVE_READS, \
        f"raw[1] was read: the oracle indexed the list, not the view ({LIVE_READS})"
    # EVERY read, including the neighbour and other-group probes, landed on a view
    # boundary. A raw-indexing oracle reads raw[0], raw[1], raw[2], raw[3] here.
    outside = set(LIVE_READS) - AT_VIEW_BOUNDARIES
    assert not outside, \
        f"reads outside the view's arithmetic, i.e. raw-indexing: {sorted(outside)}"
    assert 159 in LIVE_READS


def test_the_views_collapsed_length_is_the_bound_not_the_raw_lists(live_stage):
    """`len(view)` is 80 times shorter, and THAT is the length the comparison honours.

    Block index 9 of group 0 exists in the raw list (640 entries) and does NOT exist in
    the view (8 entries), because a request whose hashes stop before block 9 has no
    digest for it. A raw-indexing oracle compares something; this one must refuse with
    `index_out_of_range` and compare nothing - the same arithmetic the fork's own
    `len(view)` bounds use when it publishes a block's hash.
    """
    assert len(live_view(9, 0)) == 8, "the fixture's view no longer collapses by 80"
    # `computed` 9*3200 makes the fork's own `start = num_computed // g_block_size`
    # pair this request's first external block with position 9 of group 0's table.
    live_step(live_stage, [live_entry(group=0, index=1, key=raw_indexed(9, 0),
                                      gpu=LIVE_GPU[0][9], cpu=LIVE_CPU[0][0])],
              computed=9 * 3200)
    passed = counts()
    assert passed["scheduler_index_out_of_range"] == 1
    assert passed["scheduler_oracle_refused"] == 1
    assert passed["scheduler_checked"] == 0, \
        "comparing past the view is comparing a digest for a token range no one named"
    assert passed["scheduler_mismatch"] == 0
    assert LIVE_READS == []


def test_the_negative_controls_still_fail_through_the_lazy_view(live_stage):
    """Shift, signed shift, and the suffix swap: caught through the view, not around it.

    Accepting a new answer shape must not have cost this oracle the three findings it
    was built for. Each is asserted at the live scale, where a wrong index is 80 tokens
    wide rather than one hash, so a near-miss in the arithmetic is a DIFFERENT RANGE.
    """
    LIVE_READS.clear()
    live_step(live_stage, [live_entry(group=0, index=1,
                                      key=live_offered(2, 0))])
    assert counts()["scheduler_mismatch_neighbour_index"] == 1
    assert newest()["offset"] == 1
    assert newest()["expected_key16"] == key16_of(live_offered(1, 0))

    LIVE_READS.clear()
    live_step(live_stage, [live_entry(group=0, index=1,
                                      key=live_offered(0, 0))])
    assert newest()["offset"] == -1, "the sign is the diagnosis, and it must survive"
    assert set(LIVE_READS) <= {(i + 1) * LIVE_SCALE - 1 for i in range(8)}, \
        "the classification walk must not read outside the view's arithmetic"

    live_step(live_stage, [live_entry(group=0, index=1,
                                      key=live_offered(1, 0, suffix=1))])
    assert counts()["scheduler_mismatch_other_group"] == 1
    assert newest()["name_group"] == 1 and newest()["group"] == 0


def test_a_name_from_another_groups_view_is_caught_through_the_lazy_view(live_stage):
    """THE SHAPE A SUFFIX CHECK CANNOT SEE, at two genuinely different view scales.

    The box's five serving groups all resolve at 3200, so their views are the same
    object's arithmetic and one group's digest IS another's - here the second group is
    1600 tokens (scale 40) so the two views differ, which is the only way this
    classification can ever fire. Group 4's 8 stays out of the comparison, contained by
    the next two tests.
    """
    # Group 1 at 2000 tokens (scale 50) rather than a neighbour of group 0's index 1:
    # its digest for block 1 is raw[99], which is NOT group 0's view at index 0, 2 or 3,
    # so the finding can only come from the other-group walk and not from the window.
    sizes = [3200, 2000, 3200, 3200, 8, 3200]
    assert live_view(1, 0, sizes).scale_factor == 80
    assert live_view(1, 1, sizes).scale_factor == 50
    assert live_offered(1, 0, sizes=sizes) != live_offered(1, 1, sizes=sizes)
    assert live_offered(1, 1, sizes=sizes) not in (live_offered(0, 0, sizes=sizes),
                                                   live_offered(2, 0, sizes=sizes),
                                                   live_offered(3, 0, sizes=sizes))
    stage = live_stage
    stage["manager"].group_block_sizes = list(sizes)
    live_step(stage, [live_entry(group=0, index=1, sizes=sizes,
                                 key=live_offered(1, 0, sizes=sizes, from_group=1))])
    passed = counts()
    assert passed["scheduler_mismatch_other_group"] == 1
    assert passed["scheduler_mismatch_neighbour_index"] == 0
    assert newest()["classification"] == "other_group"
    assert any("a group swap: the name carries group 1's digest" in line
               for line in reasons())


def test_group_4s_unviewable_ring_is_contained_and_never_raises(live_stage):
    """GROUP 4: 8-token blocks against a 40-token hash. No view can exist for it.

    `BlockHashListWithBlockSize.__init__` asserts `target % hash == 0`
    (`kv_cache_utils.py:2826`) and `resolve_block_hashes` asserts it again before
    building (`:2887`), so asking for group 4 raises `AssertionError` - and 8 % 40 is
    not 0. Group 4 is the parked QSA ring: not prefix-cacheable, never spilled, so the
    oracle has no business being pointed at it. Two facts must both hold: it is
    CONTAINED (nothing reaches the scheduler's step thread or kills the pass) and it is
    NAMED (a
    counter and a reason, because a skipped group with no counter is how a witness goes
    decorative again).
    """
    with pytest.raises(AssertionError):
        live_resolve(LIVE_RAW, LIVE_HASH_BLOCK, 8)     # the fork's own answer

    # A fork whose gate names group 4 pairs it, so the oracle is asked for it directly.
    stage = live_stage
    stage["manager"].prefix_cacheable_group_ids = [0, 4]
    live_step(stage, [live_entry(group=4, index=0, key=raw_indexed(0, 4))],
              tokens=16)
    passed = counts()
    # ORDERED: the escape first, so an observer that RAISES is reported as an escape
    # rather than as a missing counter, and the two failure modes stay distinguishable
    # in the one line a reader sees.
    assert passed["scheduler_errors"] == 0, \
        "an AssertionError escaping the observer reaches the step thread's caller"
    assert passed["scheduler_hash_view_unbuildable"] >= 1, \
        "a group the engine cannot view must be counted, never silently skipped"
    assert passed["scheduler_oracle_refused"] >= 1
    assert passed["scheduler_checked"] == 0
    assert passed["scheduler_mismatch"] == 0
    assert any(line.startswith("hash_view_unbuildable:") and "group 4" in line
               and "8-token" in line and "40-token" in line
               for line in reasons()), reasons()
    # Asking a SECOND time: the containment is not a one-shot, and nothing about it
    # may escalate into `scheduler_errors` as the pass keeps meeting the same group.
    live_step(stage, [live_entry(group=4, index=0, key=raw_indexed(0, 4))],
              tokens=16)
    again = counts()
    assert again["scheduler_errors"] == 0
    assert again["scheduler_hash_view_unbuildable"] >= 2


def test_group_4_in_the_classification_walk_does_not_lose_the_mismatch(live_stage):
    """The realistic path: a mismatch on group 0 walks EVERY other group, group 4 too.

    `_scheduler_classify` enumerates `group_block_sizes` to see whether a name carries
    another group's digest, so on this box group 4's unviewable 8 is reached on the way
    to a finding about group 0. The finding must survive - the walk is not allowed to
    turn into an exception or to swallow the verdict - and the group it could not
    consult must still be counted.
    """
    foreign = bytes(pack(raw_digest(9999), 0)).hex()
    live_step(live_stage, [live_entry(group=0, index=1, key=foreign)])
    passed = counts()
    assert passed["scheduler_errors"] == 0
    assert passed["scheduler_mismatch"] == 1
    assert passed["scheduler_mismatch_unknown"] == 1
    assert passed["scheduler_hash_view_unbuildable"] >= 1, \
        "group 4 was skipped by the walk without saying so"
    assert newest()["classification"] == "unknown"
    assert key16_of(foreign) == newest()["key16"]


def test_a_fork_that_asserts_on_a_viewable_size_is_contained_too(live_stage,
                                                                 monkeypatch):
    """THE BACKSTOP. An `AssertionError` from elsewhere is the same counted fact.

    The pre-check reads `target % hash` off the record, so on this box group 4 never
    reaches the fork. This test removes that shield's premise: a fork that notices the
    shape differently, or one run under `-O` where the assert is stripped and something
    downstream complains instead, must be met by the same containment and the same
    counter - not by `scheduler_errors`.
    """
    def asserting(block_hashes, hash_block_size, block_size):
        # 3200 % 40 IS 0, so the pre-check lets this through and the fork is the one
        # that says no - which is the branch that must still be contained.
        raise AssertionError("a fork that notices it HERE")

    monkeypatch.setattr(sys.modules[disk_restore.KV_UTILS_MODULE],
                        "resolve_block_hashes", asserting)
    live_step(live_stage, [live_entry(group=0, index=1)])
    passed = counts()
    assert passed["scheduler_hash_view_unbuildable"] == 1
    assert passed["scheduler_oracle_refused"] == 1
    assert passed["scheduler_checked"] == 0
    assert passed["scheduler_errors"] == 0
    assert passed["scheduler_mismatch"] == 0


def test_an_answer_that_is_indexable_by_nothing_is_still_refused(live_stage,
                                                                 monkeypatch):
    """THE OTHER HALF OF THE CAPABILITY CHECK: an unrecognised shape is REFUSED.

    Accepting the lazy view must not have meant accepting anything. An object with
    neither `len()` nor integer `__getitem__` is not a hash view, and comparing one is
    how a file gets read under the wrong name - so it keeps its named refusal and its
    counter, and nothing is compared.
    """
    class NotAView:
        """No `__len__`, no `__getitem__`: the shape `resolve_shape` was written for."""

    monkeypatch.setattr(sys.modules[disk_restore.KV_UTILS_MODULE],
                        "resolve_block_hashes", lambda *a: NotAView())
    live_step(live_stage, [live_entry(group=0, index=1)])
    passed = counts()
    assert passed["scheduler_oracle_refused"] == 1
    assert passed["scheduler_checked"] == 0
    assert passed["scheduler_hash_view_unbuildable"] == 0, \
        "an unrecognised shape is not the same fact as an unviewable group"
    assert passed["scheduler_mismatch"] == 0
    assert passed["scheduler_errors"] == 0


def test_the_plain_list_answer_still_works_at_scale_one(armed, wired, monkeypatch):
    """THE 0.2.40 CASE, UNREGRESSED: `hash_block_size == group block_size`.

    When the group is the hash granularity `resolve_block_hashes` returns the REQUEST'S
    OWN LIST (`kv_cache_utils.py:2866-2867`), so `view[i] is raw[i]`, the scale is 1,
    and the oracle must behave exactly as it did before it learned about the union. It
    also the case every test in sections 1-9 above runs at.
    """
    module = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    module.make_block_hash_with_group_id = pack
    module.resolve_block_hashes = live_resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, module)
    hashes = [raw_digest(seed) for seed in range(8)]
    manager = Manager(hash_block=3200, sizes=[3200, 3200, 3200, 3200, 8, 3200],
                      cacheable=(0, 1), hits={LIVE_RID: pending(
                          cpu={group: LIVE_CPU[group] for group in range(2)})})
    stage = {"manager": manager, "request": Request(request_id=LIVE_RID,
                                                    hashes=hashes),
             "blocks": Blocks({group: LIVE_GPU[group] for group in range(2)})}
    answer = live_resolve(hashes, 3200, 3200)
    assert isinstance(answer, list) and answer is hashes, \
        "at scale one the fork hands the list itself, and the oracle must accept that"
    live_step(stage, [{"request_id": LIVE_RID,
                       "cpu_block": LIVE_CPU[0][1], "gpu_block": LIVE_GPU[0][1],
                       "group": 0, "span": 0,
                       "key": bytes(pack(hashes[1], 0)).hex()}])
    passed = counts()
    assert passed["scheduler_checked"] == 1
    assert passed["scheduler_equal"] == 1
    assert passed["scheduler_oracle_refused"] == 0
    assert passed["scheduler_hash_view_unbuildable"] == 0


def test_the_section_keeps_its_key_set_across_the_view_shapes(live_stage, monkeypatch):
    """SIX STATES, ONE KEY SET - including the two this release adds.

    Nothing seen, an expectation captured and not consumed, a verdict from the lazy
    view, a REFUSED shape, the contained group-4 assertion, and the fat payload. A
    reader
    diffing two rank files must be able to tell "this never happened" from "this build
    does not report it", and the fat payload must still carry the same keys inside the
    cap the seam REFUSES at - which is what makes a new counter key safe to add at all.
    """
    fresh = disk_restore_verify.section()

    live_step(live_stage, [live_entry(group=0, index=1)])
    captured = disk_restore_verify.section()
    assert captured["scheduler_expectations_captured"] > 0

    foreign = bytes(pack(raw_digest(9999), 0)).hex()
    live_step(live_stage, [live_entry(group=0, index=1, key=foreign)])
    verdict = disk_restore_verify.section()
    assert verdict["scheduler_mismatch"] >= 1

    # THE REFUSED SHAPE: an answer with no index protocol at all. A refusal must not
    # be a state where keys go missing - it is the state a reader most needs to diff.
    class NotAView:
        pass

    monkeypatch.setattr(sys.modules[disk_restore.KV_UTILS_MODULE],
                        "resolve_block_hashes", lambda *a: NotAView())
    live_step(live_stage, [live_entry(group=0, index=1)])
    refused = disk_restore_verify.section()
    assert refused["scheduler_oracle_refused"] >= 1

    stage = live_stage
    stage["manager"].prefix_cacheable_group_ids = [0, 4]
    live_step(stage, [live_entry(group=4, index=0, key=raw_indexed(0, 4))], tokens=16)
    unviewable = disk_restore_verify.section()
    assert unviewable["scheduler_hash_view_unbuildable"] >= 1

    disk_restore_verify._state["reasons"] = [
        f"tag{index}: " + "a persistent condition described at some length " * 4
        for index in range(disk_restore_verify.MAX_REASON_SAMPLES)]
    disk_restore_verify._state["boundary_missing_groups"].update(
        {index: 1 for index in range(40)})
    fat = disk_restore_verify.section()

    for payload in (captured, verdict, refused, unviewable, fat):
        assert set(payload) == set(fat), \
            "the key set changed between two live states: a rank-file reader cannot " \
            "tell a zero from an absent counter"
    assert set(fresh) == set(fat)
    for payload in (fresh, captured, verdict, refused, unviewable, fat):
        encoded = json.dumps(payload, separators=(",", ":"), default=str)
        assert len(encoded.encode()) <= disk_restore_verify.MAX_SECTION_BYTES
        assert len(encoded.encode()) <= worker_diag.MAX_CONTRIBUTION_BYTES
    assert fat["roles"]["worker"] == disk_restore_verify.WORKER_ROLE, \
        "the trim ladder must shed illustration before the role sentences"
