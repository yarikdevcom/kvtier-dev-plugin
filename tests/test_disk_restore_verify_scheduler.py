r"""THE SCHEDULER-SIDE BINDING ORACLE: is the file we offered the file that block wants?

The worker-side verifier (`test_disk_restore_verify.py`) answers "did the bytes arrive?"
and can never answer "was the right file chosen?", because by the time a worker applies
a restore the pairing decision has been made, reported and forgotten. This file tests
the other half: the EngineCore, at the one moment it holds BOTH the file name its own
ledger produced AND the GPU block the fork put beside it, re-derives the name it SHOULD
have offered - through the engine's own `resolve_block_hashes` and
`make_block_hash_with_group_id` - and compares that with the name in the file.

Ordering: the order of ways this oracle's answer could be worthless.

1. A CORRECT PAIRING MUST BE CONFIRMED, not merely not-refused. If the positive case
does not produce `scheduler_equal`, every negative case below is worthless, because a
module that always refuses also always "catches" the shifts. 2. A ONE-SLOT SHIFT MUST BE
CAUGHT. That is the whole reason this oracle exists: a shift moves the two index-aligned
lists together, so every list-vs-list check the plugin already has agrees with itself
forever. The re-derived key disagrees by exactly one neighbour, and says which neighbour
and in which direction. 3. A GROUP SWAP MUST BE CAUGHT, IN BOTH SHAPES. A name whose
group SUFFIX contradicts the entry, and a name carrying the RIGHT suffix over the WRONG
GROUP'S VIEW of the same index - the second is invisible to a suffix check and is the
failure mode of a per-group hash view resolved with another group's `block_size`. 4. A
DESTINATION THE REQUEST DOES NOT OWN IS A FINDING. A GPU block outside the request's own
table for that group is the one input that comes from outside our lists, and it is
reported as a mismatch with its own name - never as a refusal, which would hide it among
the ordinary ones. 5. AN ORACLE THAT CANNOT BE BUILT IS SAID OUT LOUD. `block_hashes`,
the request, or the fork's block-table channel may be unreachable; each is a NAMED
unavailability with a published reason, and NONE of them is ever a mismatch. A check
that is not running must not read as a check that passed, and must not read as a finding
either. 6. IT MUST COST THE SCHEDULER NOTHING IT DID NOT SAY. One counter and one bound
`get_block_ids` per request id per step on the step thread; the hashing, the packing and
the neighbour walk all happen on this module's own thread. With the verifier OFF the
step pays NOTHING at all - proved by instrumenting the two readers of foreign state to
raise if called. 7. THE SECTION NEVER CHANGES SHAPE AND NEVER OUTGROWS ITS CAP, now that
it carries two roles' numbers, and a sixth contribution is not available: the seam holds
five.

WHAT IS FAKE, STATED WHERE IT CANNOT BE MISSED: `block_hashes` is a list of 32-byte
objects, the manager exposes the two manager fields `resolve_block_hashes` still needs,
and `kv_connector_block_state.get_block_ids` is a dict lookup answering a FRESH copy the
way the fork's does. The key arithmetic is NOT faked here - `pack` and `resolve` are the
same two reimplementations `test_disk_restore.py` holds the OFFER path to, imported
rather than copied, so this oracle and that offer cannot both agree with a bug of mine.
One test drives the REAL offer path (`disk_restore.offer` then
`disk_restore.build_load_paths`) so the hook site is pinned, not assumed.
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
    fill,
    merged_cpu,
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
RID = "req-offer-1"
# `get_block_ids` answers BY REQUEST, with one list per group in group order - so the
# fixture's table is keyed by request id and its value is the group-ordered lists.
TABLES = {RID: [list(TABLE[0]), list(TABLE[1])]}


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
    """`Request`, as far as this oracle reaches: one attribute, `block_hashes`."""

    def __init__(self, hashes=HASHES) -> None:
        self.block_hashes = list(hashes)


class LoadState:
    """`LoadRequestState`: the fork holds the request beside the load it described."""

    def __init__(self, request) -> None:
        self.request = request


class Manager:
    """The two manager fields the view needs, plus the fork's map of live loads."""

    def __init__(self, *, hash_block=HASH_BLOCK, sizes=G_SIZE,
                 loads=None, abandoned=None) -> None:
        self.hash_block_size = hash_block
        self.group_block_sizes = list(sizes)
        self._reqs_to_load = dict(loads if loads is not None
                                  else {RID: LoadState(Request())})
        self._abandoned_reqs_to_load = dict(abandoned or {})


class BlockState:
    """`KVConnectorBlockState`: a dict lookup answering a FRESH copy, like the fork's.

    The freshness matters to the test's honesty rather than to its result: the fork's
    `get_block_ids` goes through `kv_cache_manager.get_block_ids`, which rebuilds the
    tuple of lists per call - which is why holding one on this module's thread for a
    millisecond is a snapshot rather than a live view.
    """

    def __init__(self, tables=None, *, calls=None) -> None:
        self.tables = dict(TABLES if tables is None else tables)
        self.calls = calls if calls is not None else []

    def get_block_ids(self, request_id):
        self.calls.append(request_id)
        table = self.tables.get(request_id)
        return None if table is None else [list(group) for group in table]


class Output:
    """`SchedulerOutput`: the one field this role reads."""

    def __init__(self, state) -> None:
        self.kv_connector_block_state = state


ANY = object()          # "the caller named no GPU block": the DEFAULT is index 1


def entry(*, group=0, gpu_block=ANY, key=None, request_id=RID, cpu_block=700,
          span=0) -> dict:
    """One witness entry, shaped exactly as `_build_load_paths` writes it.

    The default pairs the entry with the request's SECOND block of that group and names
    it with the SECOND digest, so the default is a CORRECT offer and every test below
    states only the one thing it is breaking. `gpu_block=None` means an entry that
    arrived with no destination at all, which is a different fact from the default.
    """
    return {"request_id": request_id, "cpu_block": cpu_block,
            "gpu_block": (TABLE[group][1] if gpu_block is ANY else gpu_block),
            "group": group, "span": span,
            "key": offered(1, group) if key is None else key}


@pytest.fixture(autouse=True)
def sample_every_entry(monkeypatch):
    """Every test sees every offered entry, unless it is the test ABOUT the rate.

    The worker-side file pins the same thing for the same reason: with a rate of 32 only
    the first entry of a boot is sampled, and "the oracle refused this entry" would be
    indistinguishable from "the oracle never saw it" - the one confusion a diagnostic
    module must never cause.
    """
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 1)
    return


@pytest.fixture
def armed_env(monkeypatch):
    """Armed by the environment - the posture a real boot uses. Starts a real thread."""
    monkeypatch.setenv(config.ENV_DISK_RESTORE_VERIFY, "1")
    return


@pytest.fixture
def armed(armed_env, monkeypatch):
    """The same, with `_ensure_thread` answering yes WITHOUT starting a thread.

    The `disk_gpu_shadow` idiom, for the same reason: with a live thread the test and
    the thread race over one retained record, and the assertion that fails is whichever
    lost. The thread's own lifecycle has its own test, below, which does not use this
    fixture.
    """
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
    """Retention and the per-pass cap lifted, for tests about SOMETHING ELSE.

    `MAX_RETAINED_PER_STEP` is 2 and `MAX_CHECKS_PER_PASS` is 2 by design - a scheduler
    step must not be able to queue an afternoon of hashing - and the two tests that pin
    those bounds do it by name. The tests below need several entries checked in ONE pass
    to make a point about index derivation, so they lift the two bounds here, out in the
    open, rather than discovering the cap as a mystery failure.
    """
    monkeypatch.setattr(disk_restore_verify, "MAX_RETAINED_PER_STEP", 16)
    monkeypatch.setattr(disk_restore_verify, "MAX_CHECKS_PER_PASS", 16)
    return


@pytest.fixture
def stage(key_helpers):
    """The scheduler stage: a manager, a block state, an output - and a call log."""
    calls: list = []
    state = BlockState(calls=calls)
    return {"manager": Manager(), "state": state, "output": Output(state),
            "calls": calls}


def see(stage, entries) -> dict:
    """One scheduler step's witness, through the module's only scheduler door."""
    return disk_restore_verify.observe_offer(stage["manager"], stage["output"], entries)


def one_pass() -> dict:
    """One pass, driven by the test. There is no worker in this process."""
    return disk_restore_verify.run_pass()


def counts() -> dict:
    return disk_restore_verify.counters()


def reasons() -> list:
    return disk_restore_verify.view()["reasons"]


def newest() -> dict:
    """The newest scheduler verdict: the ring's last element.

    The section deliberately publishes no separate `scheduler_last_mismatch` key - the
    same object twice cost 440 B of a 4096 B budget to repeat a fact a reader can index.
    """
    samples = disk_restore_verify.section()["scheduler_mismatch_samples"]
    assert samples, "no scheduler mismatch has been published"
    return samples[-1]


# --- 1: the positive control, without which every test below is hollow -------
def test_a_correctly_paired_offer_is_confirmed_by_the_engines_own_key(armed, stage):
    """The name in the file IS the name the engine's own key machinery produces.

    The GPU block is `TABLE[0][1]`, i.e. the request's SECOND block for group 0, and the
    file offered for it carries the SECOND group-0 digest. The oracle must find the
    index from the block id rather than from the entry's position, so a pass here is a
    `scheduler_equal` and not a refusal.
    """
    verdict = see(stage, [entry()])
    assert verdict["seen"] == 1 and verdict["retained"] == 1
    assert counts()["scheduler_entries_seen"] == 1
    passed = one_pass()
    assert passed["scheduler_checked"] == 1
    assert passed["scheduler_results"][0]["equal"] is True
    after = counts()
    assert after["scheduler_checked"] == 1
    assert after["scheduler_equal"] == 1
    assert after["scheduler_mismatch"] == 0
    assert after["scheduler_oracle_unavailable"] == 0
    assert after["scheduler_oracle_refused"] == 0
    assert disk_restore_verify.section()["scheduler_mismatch_samples"] == []
    assert disk_restore_verify.scheduler_state()["equal"] == 1


def test_the_index_comes_from_the_gpu_block_not_from_the_position_in_the_offer(
        armed, stage, wide_step):
    """THE POINT OF THE WHOLE ROLE, as a positive test.

    Five entries for one request, in an order with nothing to do with the block table's,
    across both groups, with the GPU ids deliberately shuffled. Each name is the correct
    one for the block it is paired with, so a walk that used the ENTRY ORDER as the
    block index would call at least two of these a mismatch.
    """
    pairs = [(0, 3), (1, 2), (0, 0), (1, 7), (0, 2)]
    entries = [entry(group=group, gpu_block=TABLE[group][index],
                     key=offered(index, group), cpu_block=900 + number,
                     span=number)
               for number, (group, index) in enumerate(pairs)]
    assert see(stage, entries)["retained"] == len(entries)
    passed = one_pass()
    assert passed["scheduler_checked"] == len(entries)
    assert counts()["scheduler_equal"] == len(entries)
    assert counts()["scheduler_mismatch"] == 0
    # ONE table resolve per request id for the whole step, not one per entry.
    assert stage["calls"] == [RID]


# --- 2: the negative control that justifies the module -----------------------
def test_a_one_slot_shift_is_a_mismatch_and_names_the_neighbour(armed, stage):
    """THE BUG THIS ORACLE EXISTS FOR. A shift moves BOTH index-aligned lists together.

    Every list-vs-list check in this plugin agrees with itself under a shift: the CPU
    slot count matches the GPU block count, the ledger's identity list has the same
    length as the fork's, and the worker's byte comparison compares a file against the
    GPU block that file was named for. Here the file for block 2 is paired with the GPU
    block that is the request's block 1 - and the re-derived key says so, by name.
    """
    see(stage, [entry(gpu_block=TABLE[0][1], key=offered(2, 0))])
    passed = one_pass()
    verdict = passed["scheduler_results"][0]
    assert verdict["equal"] is False
    assert verdict["classification"] == "neighbour_index"
    assert verdict["offset"] == 1
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_neighbour_index"] == 1
    assert after["scheduler_mismatch_stable"] == 1
    assert after["scheduler_equal"] == 0
    sample = newest()
    assert sample["index"] == 1
    assert sample["key16"] != sample["expected_key16"]
    assert sample["expected_key16"] == disk_layout.split_block_key(
        offered(1, 0))["key16"]
    assert any(line.startswith("scheduler_mismatch:neighbour_index")
               and "one-slot shift" in line for line in reasons())


def test_a_shift_the_other_way_carries_its_sign(armed, stage):
    """The offset is SIGNED, because "which way did it move" is the diagnosis.

    A shift one way and a shift the other way have different causes (an off-by-one in
    the walk that names the files versus one in the walk that pairs them), and a sample
    that said only "a neighbour" would leave an operator to re-derive which.
    """
    see(stage, [entry(gpu_block=TABLE[0][2], key=offered(1, 0))])
    one_pass()
    sample = newest()
    assert sample["classification"] == "neighbour_index"
    assert sample["offset"] == -1
    assert counts()["scheduler_mismatch_neighbour_index"] == 1


def test_a_mismatch_beyond_the_window_is_reported_as_unknown_not_guessed(armed,
                                                                         stage):
    """The window is BOUNDED, and a name outside it gets its own honest word.

    A foreign digest matches no index, no neighbour and no other group. The tempting
    shortcut is to call anything that mismatches "a shift"; this does not, because a
    wrong classification is how a wrong fix ships.
    """
    foreign = bytes(pack(raw_digest(9999), 0)).hex()
    see(stage, [entry(gpu_block=TABLE[0][1], key=foreign)])
    one_pass()
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_unknown"] == 1
    assert after["scheduler_mismatch_neighbour_index"] == 0
    assert after["scheduler_mismatch_other_group"] == 0
    assert (newest()
            ["classification"] == "unknown")


# --- 3: the group swap, in both of its shapes --------------------------------
def test_a_name_whose_group_suffix_contradicts_the_entry_is_named_first(armed, stage):
    """The suffix is a fact about OUR OWN offer and needs no engine input at all.

    `kv_cache_utils.py:83` carries the group inside the key, so a file name ending in
    `g1` for an entry the fork paired with group 0 is a group swap on the path that
    named the file - caught here before any re-derivation is attempted.
    """
    see(stage, [entry(group=0, gpu_block=TABLE[0][1], key=offered(1, 0, suffix=1))])
    one_pass()
    assert counts()["scheduler_mismatch_other_group"] == 1
    sample = newest()
    assert sample["name_group"] == 1 and sample["group"] == 0
    assert any("a group swap on our own offer path" in line for line in reasons())


def test_a_name_built_from_another_groups_view_is_caught_too(armed, stage):
    """THE SHAPE A SUFFIX CHECK CANNOT SEE, and the reason the digest is compared.

    The suffix says group 0 and the entry IS group 0, so nothing about the name's shape
    is wrong: the digest underneath it is group 1's hash view of the same index, i.e.
    the two groups' views were resolved with each other's `block_size`. The oracle
    re-derives group 0's own view, disagrees, and finds the name in group 1's view at
    the same index.
    """
    assert VIEW[G_SIZE[0]][1] != VIEW[G_SIZE[1]][1]      # the two views differ here
    see(stage, [entry(group=0, gpu_block=TABLE[0][1],
                     key=offered(1, 0, from_group=1))])
    one_pass()
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_other_group"] == 1
    assert after["scheduler_mismatch_neighbour_index"] == 0
    sample = newest()
    assert sample["classification"] == "other_group"
    assert any("a group swap: the name carries group 1's digest" in line
               for line in reasons())


# --- 4: the destination the request does not own -----------------------------
def test_a_gpu_block_outside_the_requests_table_is_a_finding_not_a_refusal(armed,
                                                                           stage):
    """No expected key exists for this entry, and that is MORE alarming, not less.

    "Which token block is GPU block 42424?" has no answer inside this request's table,
    so the bytes are going somewhere this request never owned for this group. Counting
    it as a refusal would hide it among the ordinary ones; it gets a mismatch, a counter
    and a sample that names the table it searched.
    """
    see(stage, [entry(gpu_block=42424, key=offered(1, 0))])
    passed = one_pass()
    assert (passed["scheduler_results"][0]["classification"]
            == "gpu_block_not_in_table")
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_gpu_block_not_in_table"] == 1
    sample = newest()
    assert sample["gpu_block"] == 42424
    assert sample["index"] is None
    assert sample["index"] is None
    assert any("not one this request owns for group 0" in line for line in reasons())


# --- 5: an oracle that cannot be built is said out loud ----------------------
def test_a_step_without_the_forks_block_table_channel_says_so(armed, stage):
    """`kv_connector_block_state` is built only for a step that has a connector.

    Without it the GPU half of the pairing is unreadable here, which is an
    UNAVAILABILITY with a named reason - never a mismatch, and never silence.
    """
    stage["output"].kv_connector_block_state = None
    see(stage, [entry()])
    one_pass()
    after = counts()
    assert after["scheduler_oracle_unavailable"] == 1
    assert after["scheduler_mismatch"] == 0
    assert after["scheduler_checked"] == 0
    published = disk_restore_verify.section()["scheduler_oracle"]
    assert published["unavailable"] == 1
    assert "kv_connector_block_state is None" in published["unreachable_reason"]


def test_a_request_the_fork_no_longer_holds_is_named_not_assumed(armed, stage):
    """A load the fork has forgotten has no `block_hashes`, so nothing is re-derivable.

    An id in neither `_reqs_to_load` nor `_abandoned_reqs_to_load` is a named
    unavailability, because "we could not check" and "we checked and it was fine" are
    different sentences and only one of them is reassuring.
    """
    see(stage, [entry(request_id="req-gone")])
    one_pass()
    assert counts()["scheduler_oracle_unavailable"] == 1
    reason = disk_restore_verify.view()["scheduler_unreachable_reason"]
    assert reason.startswith("no_request_in_load_state:")
    assert counts()["scheduler_mismatch"] == 0


def test_an_abandoned_load_is_still_checked(armed, key_helpers):
    """The preemption copy is asked, so a preempted-but-continuing load is not a hole.

    This is the second lookup in `_load_request`, and the reason it is there: a
    preempted request's block table is exactly where a moved block would be found, so a
    role that read only the live map would be blind at the one moment it matters.
    """
    manager = Manager(loads={}, abandoned={"req-pre": LoadState(Request())})
    state = BlockState({**TABLES, "req-pre": [TABLE[0]]})
    disk_restore_verify.observe_offer(
        manager, Output(state),
        [entry(request_id="req-pre", gpu_block=TABLE[0][1], key=offered(2, 0))])
    one_pass()
    assert counts()["scheduler_mismatch_neighbour_index"] == 1
    assert counts()["scheduler_oracle_unavailable"] == 0


def test_an_unreachable_key_helper_is_a_named_unavailability(armed, stage,
                                                            monkeypatch):
    """No `kv_cache_utils` in this process means no oracle, and it must be VISIBLE.

    The module refuses to reimplement the engine's keying, so the only honest answer is
    "the check is not running", published with its reason and counted under the
    scheduler's own name - never collapsed into the worker's `oracle_unavailable`, which
    is expected forever and means nothing.
    """
    monkeypatch.delitem(sys.modules, disk_restore.KV_UTILS_MODULE)
    see(stage, [entry()])
    one_pass()
    after = counts()
    assert after["scheduler_oracle_unavailable"] == 1
    assert after["scheduler_mismatch"] == 0
    reason = disk_restore_verify.view()["scheduler_unreachable_reason"]
    assert reason.startswith("no_key_helpers:")


def test_a_manager_that_exposes_no_hash_block_size_is_named(armed, stage):
    """The view needs two manager fields; without them there is no view to re-derive.

    Named separately from "no block_hashes" because the operator's next step differs: a
    missing manager field is a fork-shape change, a missing `block_hashes` is a request.
    """
    stage["manager"] = Manager(hash_block=0)
    see(stage, [entry()])
    one_pass()
    after = counts()
    assert after["scheduler_oracle_unavailable"] == 1
    assert disk_restore_verify.view()["scheduler_unreachable_reason"].startswith(
        "no_hash_block_size:")


def test_an_entry_with_no_gpu_block_is_a_refusal_with_its_own_name(armed, stage):
    """The two lists are index-aligned by the fork's contract; if not, say so.

    `bind_connector_metadata` refuses a step whose CPU and GPU lists differ in length,
    so an entry arriving without a destination means that contract broke - a refusal
    named `no_gpu_block`, distinct from an unavailability, with no comparison attempted.
    """
    see(stage, [entry(gpu_block=None)])
    one_pass()
    after = counts()
    assert after["scheduler_oracle_refused"] == 1
    assert after["scheduler_mismatch"] == 0
    assert any(line.startswith("scheduler_oracle:no_gpu_block") for line in reasons())


def test_an_index_outside_the_resolved_view_counts_its_own_name(armed, stage):
    """A table longer than the request's hash view is a fork-shape change, named.

    Group 0 resolves 3200-byte blocks, so 100 hashes of 128 tokens reach four blocks;
    the fifth GPU id in the table has no digest at all. That is not a mismatch (there is
    no name to disagree with) and not nothing (the arithmetic the offer relied on has
    moved under us), so it gets a counter of its own and a refusal.
    """
    fifth = TABLE[0] + [7005]
    stage["state"].tables[RID] = [fifth, list(TABLE[1])]
    see(stage, [entry(group=0, gpu_block=7005, key=offered(1, 0))])
    one_pass()
    after = counts()
    assert after["scheduler_index_out_of_range"] == 1
    assert after["scheduler_oracle_refused"] == 1
    assert after["scheduler_mismatch"] == 0


def test_a_hostile_entry_never_raises_into_the_step(armed, stage):
    """`observe_offer` is called from the scheduler's step path and must not escape.

    Garbage in the witness is a counted oddity. The thing being protected here is the
    scheduler's step, not the oracle's opinion about it.
    """
    hostile = ["not-a-dict", 7, {"group": "zero"}, {},
               {"request_id": RID, "group": 0, "gpu_block": "x", "key": "zz"}]
    verdict = see(stage, hostile)
    assert verdict["seen"] == len(hostile)
    one_pass()
    after = counts()
    assert after["scheduler_oracle_refused"] >= 3
    assert after["scheduler_errors"] == 0
    assert after["scheduler_mismatch"] == 0


# --- 6: what it costs the step, and where the work happens -------------------
def test_the_step_pays_nothing_at_all_when_the_verifier_is_off(monkeypatch, stage):
    """DEFAULT OFF means OFF: no block table resolved, no manager read, no counter.

    Proved by instrumenting the two places this role touches foreign state, rather than
    by a counter agreeing with itself.
    """
    def refuse(*args, **kwargs):
        raise AssertionError("the step reached the oracle while it was OFF")

    monkeypatch.setattr(disk_restore_verify, "_scheduler_sizes", refuse)
    monkeypatch.setattr(disk_restore_verify, "_block_table", refuse)
    assert disk_restore_verify.effective_enabled() is False
    assert see(stage, [entry(), entry(group=1)]) == {"seen": 0, "retained": 0,
                                                     "reason": "disabled"}
    assert all(value == 0 for value in counts().values())
    assert stage["calls"] == []


def test_an_unsampled_step_resolves_no_block_table_at_all(armed, stage, monkeypatch):
    """The sampling decision is taken BEFORE the foreign read, so the rate is a cost.

    A rate of four means three quarters of a step's entries cost the scheduler a counter
    and nothing else - which is what makes this role affordable on a step that carried
    fifty disk loads.
    """
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 4)
    see(stage, [entry(cpu_block=700 + index) for index in range(4)])
    after = counts()
    assert after["scheduler_entries_seen"] == 4
    assert after["scheduler_sampled_out"] == 3
    assert after["scheduler_retained"] == 1
    # One resolve for the sampled entry; the three that were sampled out cost no table
    # copy.
    assert stage["calls"] == [RID]


def test_no_thread_drops_the_witness_and_says_so_and_changes_no_offer(armed, stage,
                                                                     monkeypatch):
    """A record with no thread to read it is refused NOW, on the spot, and named.

    The worker side's `refused_thread_unavailable` with the scheduler's prefix. What
    matters just as much is the second half: the offer itself is untouched, because the
    witness is called after the paths are written and owns no decision.
    """
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread", lambda: False)
    verdict = disk_restore_verify.observe_offer(stage["manager"], stage["output"],
                                                [entry()])
    assert verdict["retained"] == 0
    after = counts()
    assert after["scheduler_retained"] == 1
    assert after["scheduler_thread_unavailable"] == 1
    assert disk_restore_verify.section()["scheduler_jobs_now"] == 0
    assert one_pass()["scheduler_checked"] == 0


def test_the_thread_drains_offered_entries_without_being_asked(armed_env, key_helpers):
    """The real thread: retained on the step thread, verified on the module's own.

    The scheduler half is not allowed to hash on the step thread, so this is the test
    that the hand-off actually happens - the record leaves `scheduler_pending` because a
    thread the scheduler never saw consumed it. The wait is BOUNDED: a module that never
    drains is a failure, not a slow test.
    """
    state = BlockState()
    assert counts()["thread_started"] == 0
    verdict = disk_restore_verify.observe_offer(
        Manager(), Output(state),
        [entry(gpu_block=TABLE[0][1], key=offered(2, 0))])
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
    """A shutdown that takes the worker cannot strand the pairing verdicts.

    The scheduler half is drained FIRST in `run_pass` for exactly this reason: those
    verdicts are about the scheduler's own addressing, were taken while the scheduler
    was alive, and need no worker at all - while the worker's retained byte comparisons
    are dropped unread, because comparing one boot's file against another boot's GPU is
    meaningless.
    """
    assert disk_restore_verify.observe_offer(
        stage["manager"], stage["output"],
        [entry(gpu_block=TABLE[0][1], key=offered(2, 0))])
    passed = one_pass()
    assert passed["reason"] == "worker_gone"
    assert passed["scheduler_checked"] == 1
    assert counts()["scheduler_mismatch_neighbour_index"] == 1
    assert counts()["worker_gone"] == 1


def test_a_step_wider_than_the_bound_stops_looking_rather_than_slowing_down(
        armed, stage, monkeypatch):
    """The per-step bound is a BOUND: hitting it is normal, and it costs no finding.

    More request ids than `MAX_SCHED_RIDS_PER_STEP` means a table copy for each, so the
    excess is refused with a named reason while the ones inside the bound are still
    checked.
    """
    monkeypatch.setattr(disk_restore_verify, "MAX_SCHED_RIDS_PER_STEP", 2)
    # The per-step RETENTION bound must not be what stops the third entry: this test is
    # about the block-table bound, so retention is lifted above it.
    monkeypatch.setattr(disk_restore_verify, "MAX_RETAINED_PER_STEP", 8)
    monkeypatch.setattr(disk_restore_verify, "MAX_CHECKS_PER_PASS", 8)
    rids = [f"req-{index}" for index in range(4)]
    stage["state"].tables.update({rid: [TABLE[0]] for rid in rids})
    stage["manager"] = Manager(loads={rid: LoadState(Request()) for rid in rids})
    entries = [entry(request_id=rid, gpu_block=TABLE[0][1], key=offered(1, 0),
                     cpu_block=1000 + index) for index, rid in enumerate(rids)]
    assert see(stage, entries)["retained"] == 2
    one_pass()
    after = counts()
    assert after["scheduler_equal"] == 2
    assert after["scheduler_oracle_unavailable"] == 2
    assert after["scheduler_mismatch"] == 0
    assert any("request ids with disk entries" in line for line in reasons())


def test_the_stability_witness_refuses_a_moving_view_as_evidence(armed, stage,
                                                                 monkeypatch):
    """A re-derivation that disagrees with itself is a race, not a layout candidate.

    The scheduler's inputs are objects the engine mutates as a request runs - the whole
    reason a worker cannot answer this question about a name. So the scheduler side's
    stability check is "do the SAME captured inputs re-derive the SAME key", and when
    they do not, the finding is published UNSTABLE: counted, and never among the ones an
    operator acts on.
    """
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
    see(stage, [entry(gpu_block=TABLE[0][1], key=offered(2, 0))])
    one_pass()
    after = counts()
    assert after["scheduler_mismatch"] == 1
    assert after["scheduler_mismatch_unstable"] == 1
    assert after["scheduler_mismatch_stable"] == 0
    assert newest()["stable"] is False


def test_the_mismatch_sample_ring_is_bounded_and_keeps_the_newest(armed, stage,
                                                                 wide_step,
                                                                 monkeypatch):
    """A step that found ten bad names publishes the last three, and counts all ten.

    The ring is the section's only variable-size scheduler field, so its cap is what
    keeps a runaway boot from making the rank file unreadable - while the counters,
    which are the load-bearing numbers, see every one.
    """
    monkeypatch.setattr(disk_restore_verify, "MAX_SCHED_MISMATCH_SAMPLES", 2)
    entries = [entry(gpu_block=TABLE[0][index], key=offered(index + 1, 0),
                     cpu_block=700 + index) for index in range(3)]
    see(stage, entries)
    one_pass()
    after = counts()
    assert after["scheduler_mismatch"] == 3, "every finding is counted"
    ring = disk_restore_verify.view()["scheduler_mismatch_samples"]
    assert len(ring) == 2, "the ring holds the NEWEST two and drops the oldest"
    assert [item["index"] for item in ring] == [1, 2]
    published = disk_restore_verify.section()
    assert published["max_scheduler_mismatch_samples"] == 2
    assert len(published["scheduler_mismatch_samples"]) <= 2
    assert published["scheduler_mismatch_samples"][-1] == ring[-1]


# --- 7: the published shape, now carrying two roles --------------------------
def test_the_section_publishes_both_roles_and_never_changes_its_key_set(armed, stage):
    """One section, two roles, one key set - fresh, after a step, after a mismatch, fat.

    The rank file's whole value is that two boots' files can be DIFFED, which requires
    the keys to exist on the paths where there is nothing to say. `roles` and
    `oracle_rule` are load-bearing here: an unprefixed `oracle_unavailable` is expected
    forever in a worker and an emergency in the EngineCore, and a reader holding one
    payload and no sentence must not be left to infer which process wrote it.
    """
    fresh = disk_restore_verify.section()
    assert fresh["roles"]["worker"] == disk_restore_verify.WORKER_ROLE
    assert fresh["roles"]["scheduler"] == disk_restore_verify.SCHEDULER_ROLE
    assert "never add the two" in fresh["roles"]["meaning"]
    assert "means the check is NOT running" in fresh["oracle_rule"]
    assert fresh["roles"]["scheduler_offer_point"] is False
    assert fresh["scheduler_oracle"]["unavailable"] == 0
    assert fresh["env_var"] == config.ENV_DISK_RESTORE_VERIFY
    see(stage, [entry(gpu_block=TABLE[0][1], key=offered(2, 0))])
    one_pass()
    moved = disk_restore_verify.section()
    assert moved["roles"]["scheduler_offer_point"] is True
    assert moved["scheduler_mismatch_samples"]
    assert moved["neighbour_window"] == disk_restore_verify.NEIGHBOUR_WINDOW
    for key in disk_restore_verify.scheduler_counters():
        assert key in moved
    # Every published scheduler counter is prefixed, and every prefixed COUNTER in
    # the section is one of them. `scheduler_oracle`, `scheduler_jobs_now`,
    # `scheduler_last_mismatch` and `scheduler_mismatch_samples` are the section's own
    # structure rather than counters, and the comparison must not confuse the two.
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
    for payload in (fresh, moved, fat):
        encoded = json.dumps(payload, separators=(",", ":"), default=str)
        assert len(encoded.encode()) <= disk_restore_verify.MAX_SECTION_BYTES
        assert len(encoded.encode()) <= worker_diag.MAX_CONTRIBUTION_BYTES


def test_the_scheduler_role_extends_the_existing_section_rather_than_a_new_one():
    """The seam holds FIVE contributions and its allow-list has exactly five names.

    So a sixth module is not TRUNCATED - `worker_diag` REFUSES an unregistered
    contribution, which would make `disk_restore_verify` the section that vanishes from
    the rank file. This is the guard against a future change that "splits the scheduler
    role out into its own section": the roles live in one section, told apart by prefix,
    and the arithmetic of the cap is pinned here rather than discovered at a deploy.
    """
    assert worker_diag.MAX_CONTRIBUTIONS == 5
    state = worker_diag.contributions()
    assert len(state["allow_list"]) == worker_diag.MAX_CONTRIBUTIONS
    assert "disk_restore_verify" in state["allow_list"]
    assert len(state["registered"]) <= worker_diag.MAX_CONTRIBUTIONS
    assert not any("scheduler" in name for name in state["allow_list"])
    assert disk_restore_verify.MAX_SECTION_BYTES \
        <= worker_diag.MAX_CONTRIBUTION_BYTES


def test_the_snapshot_publishes_the_section_the_scheduler_answers_with(armed, stage):
    """`/snapshot` is the EngineCore's door to this role, so it must carry the numbers.

    The rank FILES are written by workers; the scheduler-side counters exist only in the
    EngineCore. If `/snapshot` did not publish the section, the offer-point oracle would
    be observable in exactly zero processes.
    """
    see(stage, [entry(gpu_block=TABLE[0][1], key=offered(2, 0))])
    one_pass()
    out = snapshot.dev_state_sections()
    assert out["disk_restore_verify_enabled"] is True
    block = out["disk_restore_verify"]
    assert block["scheduler_mismatch_neighbour_index"] == 1
    assert block["scheduler_mismatch_samples"][-1]["classification"] == \
        "neighbour_index"
    assert block["roles"]["scheduler_offer_point"] is True


def test_the_status_verb_reports_both_roles_with_the_same_numbers_as_the_section(
        armed, stage):
    """The verb is the operator's only live door in the EngineCore; it cannot disagree.

    `status()` builds its payload out of `section()`, so the numbers in a `/disk
    {"op":"restore-verify"}` answer, in `/snapshot` and in a rank file are the same
    expression - which is the only reason a reader can act on any of them.
    """
    see(stage, [entry(gpu_block=TABLE[0][1], key=offered(2, 0))])
    one_pass()
    status = disk_restore_verify.status()
    assert status["ok"] is True
    assert status["scheduler_mismatch_neighbour_index"] == 1
    assert status["scheduler_mismatch_neighbour_index"] == \
        disk_restore_verify.section()["scheduler_mismatch_neighbour_index"]


# --- the hook site: the real offer path, not an assumed one ------------------
@pytest.fixture
def offer_stage(armed, key_helpers, monkeypatch, tmp_path):
    """The OFFER path armed, exactly as `test_disk_restore.armed` arms it.

    Duplicated rather than imported because a fixture in another test module is not
    reachable, and the alternative - assuming the hook is called where the docstring
    says it is - is what this test exists to prevent. The five switches are the offer
    gate; the verifier's own arm comes from `armed`.
    """
    root = tmp_path / "spill"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
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

# --- the hook site: the real offer path, not an assumed one ------------------
def pairing(manager, metadata, tables, rid) -> dict:
    """One real `build_load_paths` step against a fake fork channel. The report."""
    return disk_restore.build_load_paths(manager, metadata, Output(BlockState(
        {rid: tables})))


def test_the_real_offer_path_confirms_its_own_pairing_and_catches_a_shift(offer_stage,
                                                                         wide_step):
    """`offer` -> `build_load_paths` -> `observe_offer`, with real identities.

    Nothing staged: the identities come out of the module's own walk, the paths out of
    its own slot map, and the GPU ids from a metadata built the way the fork fills one.
    Then the SAME step is replayed with every GPU pairing rotated by one block - the
    names unchanged, the pairing moved - and the oracle has to call every one of them a
    shift.

    This is the test that the hook is where the docstring says it is, and the test that
    the two roles agree: the names this module emitted are confirmed by the engine's own
    arithmetic in one case and contradicted in the other.
    """
    # A request id of its own: `disk_restore` remembers a request that has already had
    # its one disk offer, and the suite does not reset that module between tests.
    rid = "req-offer-pairing"
    blocks = 2
    fill(offer_stage, blocks=blocks)
    request = Req(request_id=rid, blocks=blocks)
    manager = OfferManager(reqs_to_load={rid: LoadState(request)})
    offer(manager, request)
    cpu = merged_cpu(manager, rid)
    gpu = [9000 + index for index in range(len(cpu))]
    metadata = Meta(load_event=7, cpu=cpu, gpu=gpu, reqs={7: [rid]})

    def tables(*, rotated=False):
        """The fork's channel: which GPU block sits at which index of which group.

        `merged_cpu` flattens group-major, so the entry for cacheable group `position`
        and run step `step` is `cpu[position * blocks + step]`; the run's step `step` is
        the engine's block index `step` for that group. `rotated` pairs each name with
        the OTHER block - a shift, which is precisely the failure that leaves every list
        the same length.
        """
        per_group = {}
        for position, group in enumerate(CACHEABLE):
            ids = [gpu[position * blocks + step] for step in range(blocks)]
            per_group[group] = list(reversed(ids)) if rotated else ids
        return [per_group.get(group, []) for group in range(5)]

    report = pairing(manager, metadata, tables(), rid)
    assert report["ok"] is True
    assert report["named"] == len(CACHEABLE) * blocks
    seen = counts()
    assert seen["scheduler_steps_seen"] == 1
    assert seen["scheduler_entries_seen"] == report["named"]
    one_pass()
    after = counts()
    assert after["scheduler_equal"] == report["named"], after
    assert after["scheduler_mismatch"] == 0
    assert disk_restore_verify.section()["scheduler_mismatch_samples"] == []

    # THE CONTROL, ON THE SAME REAL NAMES: the files did not move, the pairing did.
    disk_restore_verify.reset_state_for_tests()
    assert pairing(manager, metadata, tables(rotated=True), rid)["named"] == \
        report["named"]
    one_pass()
    moved = counts()
    assert moved["scheduler_mismatch"] == report["named"], moved
    assert moved["scheduler_mismatch_neighbour_index"] == report["named"]
    assert moved["scheduler_equal"] == 0


def test_the_real_offer_path_with_the_verifier_off_observes_nothing(offer_stage,
                                                                   monkeypatch):
    """The same step, disarmed: the SAME paths out, and no scheduler counter moved.

    The property this release must not break is that an unarmed verifier is
    indistinguishable from no plugin - and the paths ARE the restore, so "identical
    paths, no fork call made" is the strongest form of that claim available here.
    """
    rid = "req-offer-disarmed"
    monkeypatch.delenv(config.ENV_DISK_RESTORE_VERIFY)
    # THE SAME SHAPE as the paired test above, so the only difference between the two is
    # the verifier's arm: two blocks, three cacheable groups, six entries.
    fill(offer_stage, blocks=2)
    request = Req(request_id=rid, blocks=2)
    manager = OfferManager(reqs_to_load={rid: LoadState(request)})
    offer(manager, request)
    cpu = merged_cpu(manager, rid)
    metadata = Meta(load_event=9, cpu=cpu,
                    gpu=[9000 + index for index in range(len(cpu))], reqs={9: [rid]})
    tables = [[] for _ in range(5)]
    state = BlockState({rid: tables})
    report = disk_restore.build_load_paths(manager, metadata, Output(state))
    assert report["ok"] is True
    assert report["named"] == len(CACHEABLE) * 2
    assert all(value == 0 for value in counts().values())
    assert state.calls == [], "a disarmed verifier asked the fork for a block table"


def test_the_witness_never_mutates_the_entries_it_is_handed(armed, stage):
    """The witness is a reader. If it ever edited one, it would be part of the step.

    Deep equality after a full step is the cheapest way to keep that true - including
    the records the module retains for its own thread, which therefore hold COPIES of
    the hashes and the table rather than the engine's live objects.
    """
    entries = [entry(group=0, gpu_block=TABLE[0][index], key=offered(index, 0),
                     cpu_block=700 + index) for index in range(4)]
    before = copy.deepcopy(entries)
    see(stage, entries)
    one_pass()
    assert entries == before
    retained = disk_restore_verify.view()["scheduler_pending"]
    assert not any(item["hashes"] is HASHES for item in retained.values())
