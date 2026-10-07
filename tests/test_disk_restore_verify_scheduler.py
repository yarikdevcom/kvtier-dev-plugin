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
