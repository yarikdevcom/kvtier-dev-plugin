r"""THE ALLOC-TIME TRUNCATION GATE: is a claimed span left with nothing to pair?

WHAT THIS SUITE PINS, and why the predicate is `len(list) - n_take_g` rather than a
token count. `SimpleCPUOffloadScheduler.update_state_after_alloc` takes the per-group
CPU hit lists and trims them (`manager.py:437-454`):

    for g in range(num_groups):
        if g not in self.prefix_cacheable_group_ids:
            cpu_hit_blocks.append([]); continue
        g_block_size = self.group_block_sizes[g]
        n_take_g = cdiv(num_external_tokens, g_block_size)
        cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

The fork claims `num_external_tokens` external tokens and pairs the LEADING entries it
kept with `group_gpu_ids[gpu_ext_start + i]` (`manager.py:460-478`). So a group whose
list is LONGER than its take has trailing entries the slice discards - and for the
align-mode GDN groups this tier serves, the entry at the END of the list is the ONE
recurrent state block (`_merge_into_pending` builds `[null]*(B-1) + [state]` precisely
because the fork's pairing loop preserves the index of a skipped null). Discard it and
the group is claimed as loaded across its span while contributing no real block to the
pairing: a GDN/SSM recurrence resumed from a GPU block nobody wrote. That is the
2026-10-06 00:08 answer - wrong output that raises nothing - and no test in this repo,
and no line in the fork, tested the equality it depends on.

THE INVARIANT, stated as the fork states it implicitly: `n_take_g == len(list[g])`,
maintained from two sides this repo does not own -
`MambaManager.find_longest_cache_hit` sizes an align group's list by the ACCEPTED
length, and the scheduler adopts exactly what `get_num_new_matched_tokens` reported.
The gate is the observer that says when those two stop agreeing.

THE MUTATION TEST (`test_the_mutated_padding_line_...`) does not simulate a bad tuple:
it re-executes `_merge_into_pending` with ONE line changed - the null padding that
makes an align group's list exactly as long as the fork's take - and proves the counter
moves. Control and mutant run the same offer, so the only difference between "clean"
and "the counter moved" is that line.

Everything runs on ctypes fakes: no engine, no torch, no CUDA, no network, no box.
"""

from __future__ import annotations

import inspect
import logging
import textwrap

from test_disk_gdn_boundary_restore import (
    GDN_GROUPS,
    HYBRID_CACHEABLE,
    HYBRID_SIZES,
    hybrid,
    per_group_ids,
)
from test_disk_restore import (
    BLOCK,
    Req,
    Slot,
    fill,
    offer,
)
from test_disk_restore import (
    # THE FIXTURES, aliased rather than imported into a signature the way the sibling
    # GDN suite does it: `allow_tmp_root` and `fresh` are autouse (a /tmp spill root
    # this package must otherwise refuse, and a fresh boot per test), and `armed` is
    # the gate open plus the offer switch up. Importing them by name into a test's
    # parameter list reads as a redefinition, so they get the module-level alias the
    # established suite already uses for exactly this.
    allow_tmp_root as allow_tmp_root,  # noqa: F811 - autouse, never named below
)
from test_disk_restore import (
    armed as armed,  # noqa: F811 - the gate open, the switch up
)
from test_disk_restore import (
    fresh as fresh,  # noqa: F811 - autouse, a fresh boot
)

from kvtier_dev_plugin import disk_restore  # noqa: E402 - the module under test

# The live offer this suite drives: four scheduler blocks, no host-RAM hit, so the
# merged tuple is entirely ours and every entry in it is one we allocated.
RUN_BLOCKS = 4
OFFERED_TOKENS = RUN_BLOCKS * BLOCK          # 12,800 tokens
GPU_BASE = {group: 9000 + 100 * group for group in range(len(HYBRID_SIZES))}


class Blocks:
    """`KVCacheBlocks`: the one call the gate makes on it is `get_block_ids()`."""

    def __init__(self, count=RUN_BLOCKS, base=GPU_BASE) -> None:
        self.tables = [[base[group] + index for index in range(count)]
                       for group in range(len(HYBRID_SIZES))]

    def get_block_ids(self):
        return [list(table) for table in self.tables]


class Alloc:
    """One `update_state_after_alloc` argument set, with the fork's pop recorded."""

    def __init__(self, manager, request_id="req-1", *, tokens=OFFERED_TOKENS,
                 computed=0, blocks=None) -> None:
        self.manager = manager
        self.request = Req(request_id=request_id, blocks=RUN_BLOCKS)
        self.request.num_computed_tokens = computed
        self.blocks = blocks if blocks is not None else Blocks()
        self.tokens = tokens

    def see(self) -> dict:
        return disk_restore.observe_group_take_truncation(
            self.manager, self.request, self.blocks, self.tokens)


def un_fused() -> None:
    """Forget every disk fuse this test's own earlier episode pulled.

    A refusal fuses the SPAN, so a later offer over the same blocks is refused for the
    predecessor's reason. The cadence test below is about log lines per episode, not
    about the fuse (pinned in its own test), and leaving the fuse in place would make
    `live_offer` fail for a reason the test is not asserting.
    """
    disk_restore._suppressed.clear()
    disk_restore._suppressed_spans.clear()


def live_offer(request_id="req-1"):
    """A real 4-block offer on the live hybrid geometry; the manager keeps the tuple.

    THE FIXTURE THAT PRE-ARMS NOTHING: the offer goes through the module's own body on
    a manager the suite builds fresh, against files `fill` actually wrote. Nothing here
    stubs a thread, pre-fills a pool or hands the gate a tuple it did not build.
    """
    manager = hybrid()
    answer = offer(manager, Req(request_id=request_id, blocks=RUN_BLOCKS), 0,
                   (0, False))
    assert answer == (OFFERED_TOKENS, True), disk_restore._state["last_reason"]
    return manager


def counts() -> dict:
    return {key: value for key, value in disk_restore.counters().items()
            if key.startswith("group_take") or key.startswith("last_group_take")
            or key == "take_gap_episode"}


# --- 1. the predicate: `len(list) - n_take`, and nothing else ------------------

def test_an_exact_take_is_not_a_violation_at_all():
    """The fork's equality, directly: list 4, take 4, and the gate says nothing."""
    blocks = [[Slot(100 + index) for index in range(4)]]
    assert disk_restore.group_take_gap(blocks, [BLOCK], [(0, None)],
                                       OFFERED_TOKENS) == []


def test_a_shorter_list_is_not_a_violation_either():
    """The predicate is `hit_len - n_take > 0`, one direction only.

    A list SHORTER than the take is a different question (the fork pairs fewer entries
    and claims fewer tokens than it might); the brief's invariant is the DROP, and a
    gate that fired both ways would report a state the fork never produced.
    """
    blocks = [[Slot(100 + index) for index in range(3)]]
    assert disk_restore.group_take_gap(blocks, [BLOCK], [(0, None)],
                                       OFFERED_TOKENS) == []


def test_one_entry_longer_than_the_take_is_the_violation():
    """THE SHORTEST STATEMENT OF THE FINDING: `len(list) - n_take == 1`.

    And the tail is INSPECTED, not assumed: the record separates the entries dropped
    (`dropped`) from the ones that were REAL (`dropped_real`), because a dropped null
    costs nothing while a dropped state block costs the answer.
    """
    blocks = [[Slot(100 + index) for index in range(4)] + [Slot(500)]]
    found = disk_restore.group_take_gap(blocks, [BLOCK], [(0, None)],
                                        OFFERED_TOKENS)
    assert len(found) == 1
    assert found[0] == {
        "group": 0, "group_block_size": BLOCK, "hit_len": 5, "n_take": 4,
        "dropped": 1, "dropped_real": 1, "kept_real": 4,
        "unbacked_claim": False, "num_external_tokens": OFFERED_TOKENS}


def test_a_dropped_state_block_with_nothing_left_behind_is_the_wrong_answer():
    """THE FATAL SHAPE, in isolation: the tail holds the only real block.

    `[null, null, null, null, real]` trimmed to four entries keeps four nulls and
    drops the state: `kept_real == 0`, `dropped_real == 1`, so `unbacked_claim` is the
    one field that flips. This is the field the gate refuses on, and it is the only
    reading of the invariant that names a wrong answer rather than a shorter one.
    """
    null = Slot(0)
    null.is_null = True
    blocks = [[null, null, null, null, Slot(500)]]
    found = disk_restore.group_take_gap(blocks, [BLOCK], [(0, None)],
                                        OFFERED_TOKENS)
    assert found[0]["dropped"] == 1 and found[0]["dropped_real"] == 1
    assert found[0]["kept_real"] == 0
    assert found[0]["unbacked_claim"] is True


def test_a_trim_that_drops_only_null_padding_is_counted_but_is_not_fatal():
    """The other direction of the same arithmetic, and the reason the gate is picky.

    A partial adoption that trims our PADDING discards no bytes: the group still keeps
    a real block inside the claim, so the answer is right and the restore is merely
    shorter than offered. Refusing that would refuse the fork spending its own token
    budget, which is a gate nobody leaves on.
    """
    null = Slot(0)
    null.is_null = True
    blocks = [[Slot(100), Slot(101), Slot(102), Slot(103), null]]
    found = disk_restore.group_take_gap(blocks, [BLOCK], [(0, None)],
                                        OFFERED_TOKENS)
    assert found[0]["dropped"] == 1
    assert found[0]["dropped_real"] == 0
    assert found[0]["unbacked_claim"] is False


def test_only_the_prefix_cacheable_groups_are_audited_at_all():
    """The gate reads the fork's OWN gate, so the scratch ring is not a violation.

    Group 4 (the QSA ring, 8 tokens a block, not prefix-cacheable) gets `[]` from the
    fork's loop and can never be dropped. Auditing it would invent a finding on a group
    the fork deliberately hands nothing to.
    """
    blocks = [[[], [], [], [], [Slot(i) for i in range(50)], []]
              for _ in range(1)]
    found = disk_restore.group_take_gap(blocks[0], list(HYBRID_SIZES),
                                        [(g, None) for g in HYBRID_CACHEABLE],
                                        OFFERED_TOKENS)
    assert found == []


def test_an_unmeasurable_geometry_answers_cannot_measure_rather_than_a_guess():
    """No sizes, no gate, no tokens: `[]`, and the CALLER counts that separately.

    A gate that returns "no violations" for "I could not look" makes a clean reading
    indistinguishable from a blind one - the failure that hid the reconnect storm
    behind a `ready: false`. The empty list is therefore paired with
    `group_take_unavailable` at the door, never with a silent pass.
    """
    blocks = [[Slot(1), Slot(2), Slot(3), Slot(4), Slot(5)]]
    assert disk_restore.group_take_gap(blocks, [], [(0, None)],
                                       OFFERED_TOKENS) == []
    assert disk_restore.group_take_gap(blocks, [BLOCK], [],
                                       OFFERED_TOKENS) == []
    assert disk_restore.group_take_gap(blocks, [BLOCK], [(0, None)], 0) == []
    assert disk_restore.group_take_gap(None, [BLOCK], [(0, None)],
                                       OFFERED_TOKENS) == []


# --- 2. the door, on a clean live offer: the counter that proves it ran ---------

def test_a_live_offer_on_the_live_geometry_gives_the_gate_nothing_to_refuse(armed):
    """THE CONTROL FOR THE MUTATION TEST, and the reason `0` means something.

    A real offer, a real merged tuple, the fork's own `num_external_tokens` for it: the
    gate ran (`group_take_observations` moved), audited every cacheable group, and found
    no gap. Without this the mutant's non-zero reading would only prove the counter can
    count, not that the untouched code is clean.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = live_offer()
    assert per_group_ids(manager)[0] and len(per_group_ids(manager)[0]) == RUN_BLOCKS
    before = counts()
    verdict = Alloc(manager).see()
    after = counts()
    assert verdict["ok"] is True and verdict["violations"] == []
    assert after["group_take_observations"] == before["group_take_observations"] + 1
    assert after["group_take_dropped_groups"] == 0
    assert after["group_take_unbacked_groups"] == 0
    assert after["group_take_refusals"] == 0
    assert after["last_group_take_gap"] is None
    # The tuple the clean audit cleared is the one the fork consumes: five cacheable
    # groups of four entries, or the gate audited something the fork never sees.
    assert len(per_group_ids(manager)) == len(HYBRID_SIZES)


def test_a_manager_with_no_gate_is_counted_as_unavailable_not_as_clean(armed):
    """A manager that answers neither spelling of the gate cannot be audited.

    Asserted on `group_take_unavailable`, because a counter nobody increments and a
    counter that cannot move are the same reading to an operator unless the blind case
    has its own name.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = live_offer()
    del manager.prefix_cacheable_group_ids
    del manager.group_block_sizes
    verdict = Alloc(manager).see()
    after = counts()
    assert verdict["ok"] is True and verdict["violations"] == []
    assert after["group_take_unavailable"] == 1
    assert after["group_take_dropped_groups"] == 0


def test_an_alloc_with_no_pending_hit_cannot_be_a_drop_and_says_so(armed):
    """No `_pending_cpu_hits` entry means the fork has nothing to pair.

    The fork's own body logs "update_state_after_alloc called ... but no pending CPU
    hit" and returns; the gate counts the same case as unavailable rather than reading
    an empty list as "audited, clean".
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    verdict = Alloc(manager).see()
    after = counts()
    assert verdict["violations"] == []
    # ONE observation, counted as unavailable, and nothing else moved: the blind case
    # has its own name instead of hiding inside a clean zero.
    assert after["group_take_unavailable"] == 1
    assert after["group_take_observations"] == 1
    assert after["group_take_observations"] == 1


# --- 3. THE MUTATION: one line, and the counter moves --------------------------

def _mutated_merge(monkeypatch):
    """Re-execute `_merge_into_pending` with ONE line changed, and install it.

    The line is the align group's null padding:

        for _ in range(max(0, spans - 1)):
            lists[group].append(null_block)

    `spans - 1` is what makes `[null]*(B-1) + [state]` exactly `B` entries long -
    exactly `cdiv(B * group_block_size, group_block_size)`, exactly the fork's take -
    so the
    equality `n_take_g == len(list[g])` is THIS line's job on our side of the boundary.
    Changing it to `spans` changes nothing else: the same files, the same allocation
    arithmetic (a null is not allocated), the same identities, the same offer. The list
    is simply one entry longer than the take, which is the state the fork's own sizing
    is supposed to make unreachable.

    This is a mutation of the shipped source, not a hand-built tuple: if the line moves
    or is renamed, the lookup below fails loudly instead of testing a fiction.
    """
    shipped = disk_restore._merge_into_pending
    source = textwrap.dedent(inspect.getsource(shipped))
    needle = "for _ in range(max(0, spans - 1)):"
    assert source.count(needle) == 1, \
        "the padding line is not where the gate says it is"
    mutant = source.replace(needle, "for _ in range(max(0, spans)):", 1)
    namespace = dict(vars(disk_restore))
    # S102: the mutation IS the test - a source-level change to one shipped line,
    # compiled in memory, nothing on disk touched, and nothing taken from the mutant
    # but that one function.
    exec(compile(mutant, "<mutated _merge_into_pending>", "exec"),   # noqa: S102
        namespace)
    monkeypatch.setattr(disk_restore, "_merge_into_pending",
                        namespace["_merge_into_pending"])
    # THE WAY BACK. `monkeypatch.undo()` would undo the `armed` fixture's patches too
    # (the spill root among them), and a test that has to tear down its own fixtures to
    # get at the control is a test that cannot run one. Restoring the saved function is
    # the same undo, scoped to the one attribute the mutation touched.
    return shipped


def test_the_mutated_padding_line_moves_the_truncation_counter(armed, monkeypatch,
                                                               caplog):
    """THE MUTATION TEST, and the invariant's whole case in one assertion pair.

    Same offer, same files, same manager. UNMUTATED, the gate audits four entries
    against a take of four and finds nothing; with the padding line off by one, the
    align groups' lists are five long, the fork's slice keeps four, and the entry it
    discards is each group's ONLY state block - `kept_real == 0`, `dropped_real == 1`.
    If a change to this module could ever leave a claimed span with no real block
    behind, this is the counter that says so, and it says so HERE rather than in the
    output.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)

    # CONTROL: the shipped line, the same offer, nothing to refuse.
    clean = live_offer("req-control")
    assert Alloc(clean, "req-control").see()["violations"] == []
    assert disk_restore._state["group_take_dropped_groups"] == 0
    # THE CONTROL RAN THE SAME DOOR: the observation counter proves the clean tuple
    # was AUDITED, not skipped. (No state reset between the two runs: resetting would
    # also drop the boot switches `armed` raised, and the mutant must run in the same
    # posture the control did. The deltas below are what carry the comparison.)
    control = counts()

    # MUTANT: one line, and the counter moves.
    _mutated_merge(monkeypatch)
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = live_offer("req-mutant")
    padded = per_group_ids(manager, "req-mutant")
    for group in GDN_GROUPS:
        assert len(padded[group]) == RUN_BLOCKS + 1, \
            "the mutation must make the align list LONGER than the fork's take"

    with caplog.at_level(logging.WARNING):
        verdict = Alloc(manager, "req-mutant").see()
    state = disk_restore._state
    assert [item["group"] for item in verdict["violations"]] == list(GDN_GROUPS)
    for item in verdict["violations"]:
        assert item["hit_len"] == RUN_BLOCKS + 1
        assert item["n_take"] == RUN_BLOCKS
        assert item["dropped"] == 1 and item["dropped_real"] == 1
        assert item["kept_real"] == 0, "the state block is gone and nothing is left"
        assert item["unbacked_claim"] is True
    # THE COUNTERS THE OPERATOR READS. Every one of these is zero in the control run
    # above and non-zero here, on the strength of one changed character.
    assert (state["group_take_dropped_groups"]
            - control["group_take_dropped_groups"]) == len(GDN_GROUPS)
    assert (state["group_take_dropped_entries"]
            - control["group_take_dropped_entries"]) == len(GDN_GROUPS)
    assert (state["group_take_unbacked_groups"]
            - control["group_take_unbacked_groups"]) == len(GDN_GROUPS)
    assert state["group_take_refusals"] == 1
    assert state["refused"]["group_hit_truncated"] == 1
    assert state["last_group_take_gap"]["unbacked_groups"] == list(GDN_GROUPS)
    assert state["last_group_take_gap"]["our_entry_is_dropped"] is True
    # The refusal is DISCOVERABLE, not just logged: the taxonomy name and its why.
    assert "group_hit_truncated" in disk_restore.OFFER_REFUSALS
    assert "group_hit_truncated" in state["last_refusal"]
    lines = [rec.getMessage() for rec in caplog.records
             if "TRUNCATION GATE" in rec.getMessage()]
    assert len(lines) == 1, "one line per episode, and this episode just started"
    assert "00:08" in lines[0]


def test_the_refusal_pulls_the_fuse_and_reports_the_claimed_span_for_recompute(
        armed, monkeypatch):
    """What the gate DOES, as opposed to what it counts.

    Two actions, both in this module's own power and neither requiring a fork change:
    the request's disk fuse is pulled (so the next step cannot offer it into the same
    shape a second time), and the GPU blocks the scheduler has advanced past with no
    bytes behind them are put on the load-error channel, which is the only lever that
    makes the engine recompute instead of reading them as KV.

    The range reported is the CLAIMED span `[gpu_ext_start, gpu_ext_start + n_take)` of
    the UNBACKED groups - never the dropped tail's own blocks, which are outside the
    claim and belong to nobody's answer yet. Reporting those would truncate an
    unrelated request that later holds one of them.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    _mutated_merge(monkeypatch)
    manager = live_offer("req-fuse")
    alloc = Alloc(manager, "req-fuse")
    assert alloc.see()["refused"] is True

    # The fuse: named on the request, with the refusal's own name as its reason.
    assert disk_restore._state["suppressions"] == 1
    assert "req-fuse" in disk_restore._suppressed
    assert disk_restore._suppressed["req-fuse"] == "group_hit_truncated"
    # Our record is gone: no paths will be emitted, no read queued, nothing to prune.
    assert "req-fuse" not in disk_restore._loads
    assert disk_restore._state["loads_released"] == 1

    reported = disk_restore.take_invalid_block_ids()
    gpu = alloc.blocks.get_block_ids()
    expected = set()
    for group in GDN_GROUPS:
        expected.update(gpu[group][:RUN_BLOCKS])       # the claimed span, unwritten
    assert reported == expected
    # The attention group keeps its four real blocks, so its span is NOT reported: a
    # gate that truncated every group in the step would cost far more than the bug.
    assert not reported & set(gpu[5][:RUN_BLOCKS])
    # Drained, as `take_invalid_block_ids` promises: a stale id would truncate an
    # unrelated request holding that block later.
    assert disk_restore.take_invalid_block_ids() == set()


# --- 4. the episode, the vocabulary, and the door's own hygiene -----------------

def test_the_line_is_one_per_episode_and_a_clean_alloc_ends_the_episode(
        armed, monkeypatch, caplog):
    """Two sightings, one line; a clean step between them, two.

    This runs once per scheduled request per step. A line per sighting turns a data
    finding into the noise that buries it, and a line per boot means an operator who
    missed the first one never sees the second. The episode rule is the one every other
    rate-limited line in this module uses.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    shipped = _mutated_merge(monkeypatch)
    un_fused()
    first = live_offer("req-episode-a")
    Alloc(first, "req-episode-a").see()
    un_fused()
    second = live_offer("req-episode-b")
    Alloc(second, "req-episode-b").see()
    def seen():
        return [r for r in caplog.records if "TRUNCATION GATE" in r.getMessage()]
    assert len(seen()) == 1, "the second sighting must not repeat the line"

    # A clean alloc - the shipped padding, the same geometry - closes the episode.
    disk_restore._merge_into_pending = shipped
    un_fused()
    clean = live_offer("req-clean")
    Alloc(clean, "req-clean").see()
    assert len(seen()) == 1
    _mutated_merge(monkeypatch)
    un_fused()
    third = live_offer("req-episode-c")
    Alloc(third, "req-episode-c").see()
    assert len(seen()) == 2, "an episode that came back says it came back"
    # The counters keep counting the whole time: the line is rate-limited, the
    # evidence is not.
    assert disk_restore._state["group_take_dropped_groups"] == 3 * len(GDN_GROUPS)


def test_every_new_counter_declares_its_units_and_the_key_set_never_moves(armed):
    """THE STABLE KEY SET, pinned before anything has happened.

    The recurring failure this repo keeps paying for is a payload whose keys depend on
    what the boot did: a reader (or a diff, or an alert) sees a key vanish and cannot
    tell "never happened" from "not reported". These keys are in `_state` at import, so
    they are in the first read of a fresh process, and every one of them states its
    unit in words that match what increments it.
    """
    disk_restore.reset_state_for_tests()
    published = disk_restore.counters()
    units = published["units"]
    keys = ("group_take_observations", "group_take_dropped_groups",
            "group_take_dropped_entries", "group_take_unbacked_groups",
            "group_take_refusals", "group_take_unavailable",
            "last_group_take_gap")
    for key in keys:
        assert key in published, f"{key} is not published on a fresh boot"
        if key != "last_group_take_gap":
            assert published[key] == 0, f"{key} does not start at zero"
            assert key in units and units[key].strip(), f"{key} has no unit"
            assert "(unit:" in units[key], f"{key}'s unit is not stated"
    # The honest unit: the observations key is the DENOMINATOR, and says so, because
    # `dropped_groups: 0` is only evidence if something proves the gate looked.
    assert "DENOMINATOR" in units["group_take_observations"]
    assert "group_hit_truncated" in published["refused"]
    assert published["refused"]["group_hit_truncated"] == 0
    assert "group_hit_truncated" in disk_restore.OFFER_REFUSAL_WHY
    # THE SAME KEYS IN THE PUBLISHED DOCUMENT (0.2.54). The pin above walks
    # `counters()`, which copies `_state` wholesale and was therefore ALWAYS complete -
    # which is precisely why it could not catch that the route publishes
    # `prune_state()` instead, and `prune_state()` had never heard of the gate. Both
    # views are pinned from here on.
    document = published_prune()
    for key in PUBLISHED_TAKE_KEYS:
        assert key in document, f"{key} is not in the published load_prune document"
        if key != "last_group_take_gap":
            assert document[key] == 0
            assert "(unit:" in document["units"][key]


def test_the_gate_never_raises_even_when_the_fork_s_shape_is_nonsense(armed):
    """A gate on the alloc path cannot be allowed to break the step.

    Everything here is read off another process's objects: a list whose entries are
    strings, sizes that are not numbers, a request with no id. Each must come back as a
    status, not an exception - the fork's `update_state_after_alloc` runs on the step
    thread and an exception there is a lost step, which is how a diagnostic becomes an
    outage.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = live_offer("req-nonsense")
    manager._pending_cpu_hits["req-nonsense"] = (
        [["not", "blocks", "at", "all", None]] * len(HYBRID_SIZES)
        + [OFFERED_TOKENS, 0])
    manager.group_block_sizes = [None] * len(HYBRID_SIZES)
    assert Alloc(manager, "req-nonsense").see()["ok"] is True

    class NoId:
        pass

    # A request with no `request_id` at all, and a request id with no tuple under it:
    # neither may reach the fork as an exception.
    assert Alloc(live_offer("req-nonsense-2"), "no-such-request").see()["ok"] is True
    manager2 = hybrid()
    manager2.group_block_sizes = "not-a-list"
    assert disk_restore.observe_group_take_truncation(
        manager2, NoId(), Blocks(), OFFERED_TOKENS)["ok"] is True


def test_the_gate_runs_inside_the_alloc_wrapper_before_the_fork_pops_the_tuple(
        armed, monkeypatch):
    """THE WIRING, pinned: the call is in the wrapper, not only exported.

    `update_state_after_alloc` pops the pending tuple inside its own body
    (`manager.py:415-429`), so a gate fitted AFTER the fork - where every restore hook
    in this module lives, because a restore must not run before the fork decided -
    would find the numbers gone and report nothing forever. The wrapper therefore runs
    the gate first and still returns the fork's own value untouched.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    _mutated_merge(monkeypatch)
    manager = live_offer("req-wired")
    popped: list = []

    def fork_body(self, request, blocks, num_external_tokens):
        # The fork's own pop: after this line the tuple is gone.
        popped.append(self._pending_cpu_hits.pop(request.request_id, None))
        return "FORK RETURNED"

    wrapper = disk_restore.make_wrapper(disk_restore.METHOD_ALLOC, fork_body)
    assert getattr(wrapper, disk_restore.ORIGINAL_ATTR) is fork_body
    alloc = Alloc(manager, "req-wired")
    returned = wrapper(manager, alloc.request, alloc.blocks, alloc.tokens)
    assert returned == "FORK RETURNED", "the fork's return value is not ours to change"
    assert popped and popped[0] is not None
    # The gate saw the tuple BEFORE the pop, which is the only moment it exists.
    assert disk_restore._state["group_take_observations"] == 1
    assert disk_restore._state["group_take_unbacked_groups"] == len(GDN_GROUPS)
    assert disk_restore._state["group_take_refusals"] == 1


def test_a_request_the_tier_never_offered_is_reported_but_not_refused(armed,
                                                                      monkeypatch,
                                                                      caplog):
    """The gate reports an unbacked claim it did not cause, and refuses nothing.

    A pending tuple can exist with no `_loads` entry: the fork found its own host-RAM
    hit and this module never offered. There is no restore of ours to withdraw and no
    fuse to pull - refusing would count a disk refusal the disk never made. The
    counters and the load-error report still move, because the wrong answer is just as
    wrong when we did not cause it, and it is the reason the gate is worth having.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = hybrid()
    # A tuple in the fork's own shape: a RAM hit whose align group's state block sits
    # AFTER the take. Built by hand because this request never went through `offer`.
    null = Slot(0)
    null.is_null = True
    lists = [[Slot(600 + index) for index in range(RUN_BLOCKS)]
             for _ in range(len(HYBRID_SIZES))]
    # The align group's list is FOUR NULLS THEN ONE STATE BLOCK: five entries for a
    # take of four, so the slice keeps the nulls and discards the state - `kept_real`
    # falls to 0. This is the shape `_merge_into_pending` builds when its padding is
    # one longer than the run, reached here without mutating the module: the fork's
    # own `find_longest_cache_hit` may also hand back a list whose accepted run ends
    # one block past the token budget the scheduler finally adopted.
    for group in GDN_GROUPS:
        lists[group] = [null] * RUN_BLOCKS + [Slot(700 + group)]
    manager._pending_cpu_hits["req-foreign"] = (lists, OFFERED_TOKENS, 0)
    alloc = Alloc(manager, "req-foreign")
    verdict = alloc.see()
    state = disk_restore._state
    assert verdict["refused"] is False
    assert [item["unbacked_claim"] for item in verdict["violations"]] == \
        [True] * len(GDN_GROUPS)
    assert state["group_take_unbacked_groups"] == len(GDN_GROUPS)
    assert state["group_take_refusals"] == 0
    assert state["refused"]["group_hit_truncated"] == 0
    assert "req-foreign" not in disk_restore._suppressed
    assert state["last_group_take_gap"]["our_entry_is_dropped"] is False
    # The claimed spans are still reported: a block nobody wrote must not be read.
    gpu = alloc.blocks.get_block_ids()
    reported = disk_restore.take_invalid_block_ids()
    assert reported == set().union(*[set(gpu[group][:RUN_BLOCKS])
                                     for group in GDN_GROUPS])


# --- 5. THE PUBLISHED DOCUMENT: a counter that is not routed is not readable -----
#
# THE BLIND SPOT 0.2.53 SHIPPED WITH. Every counter above is asserted through
# `disk_restore._state` or `counters()` - and the HTTP document an operator reads is
# NOT `counters()`. Verified against the live box on 0.2.53: `GET
# /plugins/kvtier/snapshot`'s `disk_restore.load_prune` carried 21 keys and not one of
# them was the gate's, because the route publishes `scheduler_section()`, whose
# `load_prune` is `prune_state()`, and `prune_state()` builds its keys from the
# `prune_*` names by hand. So the suite was green, `counters()` was complete, and an
# operator watching the 00:08-shaped failure live had the gate's `units` prose
# describing numbers the document did not contain. THESE TESTS READ THE BUILDER.

# THE SIX KEYS THE PUBLISHED `load_prune` MUST ALWAYS CARRY. `group_take_refusals` is
# deliberately NOT among them: that event is ALREADY published once, under the refusal
# taxonomy name `refused["group_hit_truncated"]`, from the same two lines of code, and
# `test_the_refusal_is_counted_once_in_the_document` below pins that it stays that way.
# A second name for one event in one document is how a reader ends up adding it twice.
PUBLISHED_TAKE_KEYS = ("group_take_observations", "group_take_dropped_groups",
                       "group_take_dropped_entries", "group_take_unbacked_groups",
                       "group_take_unavailable", "last_group_take_gap")


def published_prune() -> dict:
    """THE `load_prune` BLOCK THE HTTP ROUTE SERVES - not the state dict.

    `snapshot.py` publishes `disk_restore.scheduler_section()` as the document's
    `disk_restore` section, so this call IS the HTTP-visible document builder. Reading
    `_state` here would repeat the mistake this section exists to prevent.
    """
    return disk_restore.scheduler_section()["load_prune"]


def test_the_published_document_carries_the_gate_in_every_state(armed, monkeypatch):
    """STABLE KEYS, NOW ON THE WIRE, in every state the module can be asked about.

    A key that appears only after the gate has fired is worse than no key: "the boot
    never hit the shape" and "this build does not report it" are the same reading on
    the page, and the second is the one that sends an operator to read the fork.
    Checked in the order a boot actually visits these: a fresh process, the armed
    posture with nothing done, the prune switch OFF, a clean audited alloc, and a gate
    that could not measure at all. Every one must carry the same key set, and the verb
    side must agree with the route.
    """
    def keys_of(view: dict) -> set:
        return {key for key in PUBLISHED_TAKE_KEYS if key in view}

    # (a) THE FIRST READ OF A FRESH BOOT (the `fresh` fixture's own posture): the
    # counters exist and read zero. NOT `reset_state_for_tests()` - that would also
    # drop the switches the `armed` fixture raised, and the states below need them.
    fresh_boot = published_prune()
    assert keys_of(fresh_boot) == set(PUBLISHED_TAKE_KEYS)
    for key in PUBLISHED_TAKE_KEYS:
        if key != "last_group_take_gap":
            assert fresh_boot[key] == 0, f"{key} does not start at zero on the wire"
    assert fresh_boot["last_group_take_gap"] is None

    # (b) ARMED, SWITCH UP, NOTHING DONE: the same keys, the same zeros.
    never_fired = published_prune()
    assert keys_of(never_fired) == set(PUBLISHED_TAKE_KEYS)
    assert never_fired["group_take_observations"] == 0
    # EVERY ONE OF THEM DECLARES ITS UNIT IN THIS DOCUMENT, not only in `counters()`:
    # the route serves `load_prune`'s own `units`, and a bare number there is the same
    # bug as a missing one.
    units = never_fired["units"]
    for key in PUBLISHED_TAKE_KEYS:
        assert str(units.get(key, "")).strip(), f"{key} reaches the wire with no unit"
        assert "(unit:" in units[key], f"{key}'s unit is not stated on the wire"

    # (c) THE PRUNE SWITCH OFF: a disarmed posture is a reading, not an absence.
    # A SCOPED context, not `monkeypatch.undo()`: the `armed` fixture's own patches
    # (the spill root among them) would go with it, and the two states after this one
    # still need them.
    with monkeypatch.context() as patch:
        patch.setattr(disk_restore, "prune_switch",
                      lambda: {"enabled": False, "raw": None,
                               "recognised": True, "source": "test",
                               "env_var":
                               "VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS"})
        disarmed = published_prune()
    assert disarmed["enabled"] is False
    assert keys_of(disarmed) == set(PUBLISHED_TAKE_KEYS)

    # (d) A CLEAN AUDITED ALLOC: the DENOMINATOR moves on the wire, which is the only
    # thing that makes the zeros above mean "audited and clean".
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    clean = live_offer("req-clean-wire")
    Alloc(clean, "req-clean-wire").see()
    audited = published_prune()
    assert audited["group_take_observations"] >= 1
    assert audited["group_take_dropped_groups"] == 0
    assert audited["last_group_take_gap"] is None

    # (e) THE GATE COULD NOT MEASURE: the blind case has its own name on the wire too.
    del clean.prefix_cacheable_group_ids
    del clean.group_block_sizes
    Alloc(clean, "req-clean-wire").see()
    blind = published_prune()
    assert blind["group_take_unavailable"] >= 1
    assert keys_of(blind) == set(PUBLISHED_TAKE_KEYS)

    # THE VERB SIDE AND THE ROUTE AGREE, key for key: one builder, two render sites.
    assert set(disk_restore.section()["load_prune"]) == set(audited)


def test_a_process_with_nothing_fitted_still_answers_the_gate_keys():
    """THE DEGRADED VIEW: a process that installed nothing answers in full.

    The state an operator meets when the layer never fitted - no class, no boot
    switches, no spill root - is the one where a missing key is most likely to be read
    as "this build has no gate". `prune_state()` builds its keys from `_state`, and
    every gate key is in `_state` at import, so the empty process answers zeros and a
    null and the verb side agrees with the route key for key.
    """
    disk_restore.reset_state_for_tests()
    view = disk_restore.scheduler_section()
    document = view["load_prune"]
    assert view["layer_installed_here"] is False
    assert {key for key in PUBLISHED_TAKE_KEYS if key in document} \
        == set(PUBLISHED_TAKE_KEYS)
    assert document["group_take_observations"] == 0
    assert document["last_group_take_gap"] is None
    for key in PUBLISHED_TAKE_KEYS:
        assert "(unit:" in document["units"][key]
    assert set(disk_restore.section()["load_prune"]) == set(document)


def test_the_gate_counters_move_in_the_published_document_not_only_the_state(
        armed, monkeypatch):
    """THE COUNTER AN OPERATOR CAN READ, proved by the same mutation as section 3.

    The mutation is the cheapest available way to make the gate fire for real on the
    live geometry, and the assertion is deliberately NOT on `_state`: 0.2.53's tests
    all read the dict, which is exactly how an unrouted counter survived a green
    suite. If the counter moves in `disk_restore._state` and not in
    `scheduler_section()["load_prune"]`, this fails - and that gap is the bug.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    before = published_prune()
    assert before["group_take_unbacked_groups"] == 0

    _mutated_merge(monkeypatch)
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = live_offer("req-wire")
    verdict = Alloc(manager, "req-wire").see()
    assert verdict["refused"] is True

    view = disk_restore.scheduler_section()
    published = view["load_prune"]
    assert published["group_take_observations"] >= 1
    assert published["group_take_dropped_groups"] == len(GDN_GROUPS)
    assert published["group_take_dropped_entries"] == len(GDN_GROUPS)
    assert published["group_take_unbacked_groups"] == len(GDN_GROUPS), \
        "the wrong-answer count is invisible on the wire"
    assert published["group_take_unavailable"] == 0
    gap = published["last_group_take_gap"]
    assert isinstance(gap, dict) and gap["unbacked_groups"] == list(GDN_GROUPS)
    assert gap["our_entry_is_dropped"] is True
    # THE REFUSAL IS COUNTED ONCE, in this same document, under the taxonomy name -
    # and NOT re-exposed beside it under the gate's own name.
    assert view["refused"]["group_hit_truncated"] == 1
    assert "group_take_refusals" not in published, \
        "one event, one name in one document"
    # And the pass-through is a copy: a reader mutating the document cannot move the
    # module's own record of the sighting.
    gap["request_id"] = "scribbled"
    assert disk_restore._state["last_group_take_gap"]["request_id"] != "scribbled"


# --- 6. WHY THE GATE WAS BLIND: the counter that must never climb in silence -----
#
# THE LIVE FINDING 0.2.55 EXISTS FOR. 0.2.54 published `group_take_unavailable`, it went
# live, and within five minutes a HEALTHY boot read `unavailable: 1` - and neither the
# log nor the payload could say which condition produced it, because `_take_gate`
# returned a bare None and the pending-hit branch beside it incremented the SAME
# counter. So the project's own `units` prose ("a boot where this is every observation
# has an unaudited take invariant wearing six clean zeros") was readable while the fact
# that decided it was not. These tests read the PUBLISHED DOCUMENT, for the reason
# section 5 exists: a cause that lives only in `disk_restore._state` is a cause nobody
# can read during an incident.

# THE SEVEN WAYS AN OBSERVATION GOES BLIND, exactly as the module declares them. If a
# new condition is added without a driver below, the last test in this section fails - a
# vocabulary with an unreachable name is how a counter starts lying again.
BLIND_PATHS = ("no_group_block_sizes", "no_cacheable_group_set", "gate_read_failed",
               "no_pending_hit_table", "empty_request_id",
               "no_pending_hit_for_request", "pending_hit_wrong_shape")


def blind_manager(reason, request_id="req-blind"):
    """A manager built to fail in EXACTLY one way, with the request id that hits it.

    Each branch deletes or corrupts ONE attribute, so the reason the gate reports is the
    condition the test named and not a neighbour that happened to be missing too.
    Nothing here hands the gate a tuple it did not have to look up.
    """
    manager = hybrid()
    if reason == "no_group_block_sizes":
        manager.group_block_sizes = []            # present, but empty: no gate at all
    elif reason == "no_cacheable_group_set":
        del manager.prefix_cacheable_group_ids    # the fake has no `block_ids` either
    elif reason == "gate_read_failed":
        class Explodes:
            def __iter__(self):
                raise RuntimeError("the fork's group set is not iterable")
        manager.prefix_cacheable_group_ids = Explodes()
    elif reason == "no_pending_hit_table":
        del manager._pending_cpu_hits             # not the manager we offer into
    elif reason == "pending_hit_wrong_shape":
        manager._pending_cpu_hits[request_id] = ["not", "the", "fork's", "tuple"]
    elif reason == "no_pending_hit_for_request":
        pass                                      # the table is there and empty
    elif reason == "empty_request_id":
        pass
    else:  # pragma: no cover - a guard against a name added without a driver
        raise AssertionError(f"no driver for blind reason {reason!r}")
    return manager


def re_armed() -> None:
    """Raise the switch `reset_state_for_tests()` drops, so `live_offer` still offers.

    The `armed` fixture raises DISK AUTHORITY as a call, not a patch, so a reset in the
    middle of a test leaves the boot offering nothing and `live_offer` would fail for a
    reason the test is not asserting. The existing suite makes the same note ("NOT
    `reset_state_for_tests()` - that would also drop the switches").
    """
    disk_restore.set_disk_authority(True)


def see_blind(reason, request_id="req-blind") -> dict:
    """One alloc that must come back blind for the named reason."""
    manager = blind_manager(reason, request_id=request_id)
    request_id = "" if reason == "empty_request_id" else request_id
    verdict = Alloc(manager, request_id).see()
    assert verdict["ok"] is True, "a blind observation must never break the step"
    return disk_restore.scheduler_section()["load_prune"]


def test_every_blind_condition_names_itself_in_the_published_document(armed):
    """THE FIX, ASSERTED ON THE WIRE: seven causes, seven distinct readings.

    Before 0.2.55 these were ONE number. Each case below drives one condition on a fresh
    module state and reads `scheduler_section()["load_prune"]` - the block the HTTP
    route serves - because the lesson of 0.2.53/0.2.54 is that a counter or a cause
    visible only in `_state` is a cause that can be entirely wrong while the suite is
    green.
    """
    for reason in BLIND_PATHS:
        disk_restore.reset_state_for_tests()
        re_armed()
        document = see_blind(reason)
        assert document["group_take_observations"] == 1, reason
        assert document["group_take_unavailable"] == 1, \
            f"{reason} did not count its blind look"
        counts = document["group_take_unavailable_reasons"]
        assert counts[reason] == 1, f"{reason} is not the reason the document reports"
        # ...and the OTHER names stayed zero: the cause is specific, not a catch-all
        # that every path happens to share (the failure the bare None was).
        assert sum(counts.values()) == document["group_take_unavailable"], \
            f"{reason}: the breakdown is not a partition of the total"
        last = document["last_group_take_unavailable_reason"]
        assert isinstance(last, dict) and last["reason"] == reason
        # THE OBJECT, because "some other manager type" is one of the candidate answers
        # and the payload must be able to say it without reading the code.
        assert last["manager"] == type(blind_manager(reason)).__name__
        assert last["at"] > 0


def test_the_benign_blindness_is_told_apart_from_the_unaudited_kind(armed):
    """THE JUDGEMENT `unavailable: 1` COULD NOT SUPPORT, in the payload's own words.

    A request with no host-RAM hit has no pending CPU hit by construction, so
    `no_pending_hit_for_request` is what an ordinary alloc looks like; every other name
    means the take invariant is unwatched. The two are opposite operational stories, and
    collapsing them is why the live reading was undecidable - so the distinction is
    pinned in the `units` prose and not only in a comment.
    """
    disk_restore.reset_state_for_tests()
    re_armed()
    benign = see_blind("no_pending_hit_for_request")
    disk_restore.reset_state_for_tests()
    re_armed()
    unaudited = see_blind("no_group_block_sizes")
    assert benign["group_take_unavailable"] == 1
    assert unaudited["group_take_unavailable"] == 1
    # Identical totals, opposite meanings: the total alone is NOT the reading.
    assert benign["group_take_unavailable"] == unaudited["group_take_unavailable"]
    assert benign["group_take_unavailable_reasons"]["no_pending_hit_for_request"] == 1
    assert unaudited["group_take_unavailable_reasons"]["no_group_block_sizes"] == 1
    for key in ("group_take_unavailable_reasons", "group_take_unavailable"):
        unit = benign["units"][key]
        assert "(unit:" in unit, f"{key} reaches the wire with no unit"
    assert "no_pending_hit_for_request" in benign["units"][
        "group_take_unavailable_reasons"]
    assert "UNWATCHED" in benign["units"]["group_take_unavailable_reasons"]
    # THE ORDINARY CASE MUST BE CALLED ORDINARY, or a healthy box reads as a sick one.
    assert "ordinary cache-miss alloc" in benign["units"][
        "group_take_unavailable_reasons"]


def test_many_blind_looks_log_once_and_the_counter_still_climbs(armed, caplog):
    """ONE LINE PER REASON PER EPISODE, with the count unaffected.

    This is the alloc path: one call per scheduled request per step, and a request with
    no host-RAM hit is blind by construction. A line per observation would print a line
    per request - the flood the `take_gap_episode` idiom exists to prevent, and the
    reason the 0.2.55 line could not simply be `logger.info` at the blind branch.

    THE COUNTER STILL CLIMBS: dedup is a LOG rule, never a counting rule. A notice that
    swallowed the count would be the same bug wearing a log line.
    """
    disk_restore.reset_state_for_tests()
    re_armed()
    manager = blind_manager("no_pending_hit_for_request")
    with caplog.at_level(logging.INFO):
        for _ in range(25):
            Alloc(manager, "req-blind").see()
    document = disk_restore.scheduler_section()["load_prune"]
    assert document["group_take_observations"] == 25
    assert document["group_take_unavailable"] == 25, \
        "the once-per-episode rule must not swallow the count"
    assert document["group_take_unavailable_reasons"][
        "no_pending_hit_for_request"] == 25
    lines = [r for r in caplog.records if "TRUNCATION GATE" in r.getMessage()]
    assert len(lines) == 1, \
        f"25 blind looks produced {len(lines)} lines; the episode rule is gone"
    line = lines[0].getMessage()
    # THE LINE NAMES THE CAUSE AND THE OBJECT, which is the whole point of the change.
    assert "no_pending_hit_for_request" in line
    assert "Manager" in line
    # THE LINE PRICES ITSELF AT THE MOMENT THE EPISODE OPENS (the payload carries the
    # running total): the first blind look says "1 of 1", and that is honest, because
    # the line is the episode's opening statement and not a status report.
    assert "1 of 1 blind look(s)" in line, \
        "the line must state its own count against the total"
    # AND THE ACTIONABLE POINTER SURVIVES `notice_once`'s 640-char truncation.
    assert "group_take_unavailable_reasons" in line


def test_a_measured_look_reopens_the_episode_so_a_returning_blindness_is_said_again(
        armed, caplog):
    """`notice_once` alone would cap this at one line per BOOT.

    "Blind at boot, then fine" and "fine, then blind" must not be the same reading, so a
    look that MEASURED closes the episode and the next blindness gets its own line.
    Pinned here because the dedup idiom, taken literally, would silently make the
    payload's most important distinction - is it still happening? - unloggable.
    """
    disk_restore.reset_state_for_tests()
    re_armed()
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    blind = blind_manager("no_pending_hit_for_request")
    with caplog.at_level(logging.INFO):
        Alloc(blind, "req-blind").see()
        Alloc(live_offer("req-clean"), "req-clean").see()   # a look that measured
        Alloc(blind, "req-blind").see()
    lines = [r for r in caplog.records if "TRUNCATION GATE" in r.getMessage()]
    assert len(lines) == 2, \
        "a blindness that came back must be said again, once"
    document = disk_restore.scheduler_section()["load_prune"]
    assert document["group_take_observations"] == 3
    assert document["group_take_unavailable"] == 2


def test_a_boot_that_never_looked_is_distinguishable_from_a_boot_blind_every_time(
        armed):
    """A LOOK IS NOT A MEASUREMENT, AND BOTH ARE NOT A BLIND LOOK.

    Three states a reader must tell apart from the published document alone:

      (a) never looked: `observations 0`, `unavailable 0` - the process is not the one
          that takes (a worker, or a boot with no allocs), which is a ROLE reading.
      (b) looked, all blind: `observations N`, `unavailable N` - it looked N times and
          measured nothing. An unaudited take invariant, or N cache-miss allocs, told
          apart by the reason names and by nothing else.
      (c) looked, measured: `observations N`, `unavailable 0` - the only state where the
          zeros below it are evidence of anything.

    A document that could not separate these would let (b) wear (c)'s clean zeros, which
    is the exact artifact the `units` prose warns about; and (a) must NOT read as
    damage, because a worker process sits in it permanently.
    """
    never_looked = disk_restore.scheduler_section()["load_prune"]
    assert never_looked["group_take_observations"] == 0
    assert never_looked["group_take_unavailable"] == 0
    assert sum(never_looked["group_take_unavailable_reasons"].values()) == 0
    assert never_looked["last_group_take_unavailable_reason"] is None

    disk_restore.reset_state_for_tests()
    re_armed()
    blind = blind_manager("no_pending_hit_for_request")
    for _ in range(3):
        Alloc(blind, "req-blind").see()
    all_blind = disk_restore.scheduler_section()["load_prune"]
    assert all_blind["group_take_observations"] == 3
    assert all_blind["group_take_unavailable"] == 3
    assert all_blind["last_group_take_unavailable_reason"] is not None

    disk_restore.reset_state_for_tests()
    re_armed()
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    Alloc(live_offer("req-clean"), "req-clean").see()
    measured = disk_restore.scheduler_section()["load_prune"]
    assert measured["group_take_observations"] == 1
    assert measured["group_take_unavailable"] == 0
    assert measured["last_group_take_unavailable_reason"] is None

    # THE THREE READINGS ARE PAIRWISE DISTINCT ON THE WIRE.
    assert (never_looked["group_take_observations"],
            never_looked["group_take_unavailable"]) == (0, 0)
    assert (all_blind["group_take_observations"],
            all_blind["group_take_unavailable"]) == (3, 3)
    assert (measured["group_take_observations"],
            measured["group_take_unavailable"]) == (1, 0)
    # A LOOK THAT MEASURED leaves no trace in the blind keys, so the two axes stay
    # independent: `observations` counts looks, `unavailable` counts blind ones, and
    # `observations - unavailable` is the only number that makes the zeros evidence.
    assert measured["group_take_unavailable_reasons"] == {
        name: 0 for name in BLIND_PATHS}


def test_the_new_cause_keys_are_stable_in_every_state_and_pair_their_units(
        armed, monkeypatch):
    """THE KEY/UNITS PIN, EXTENDED FOR THE CAUSE KEYS (0.2.55).

    The two render sites must carry the same keys (`snapshot.py` serves
    `scheduler_section()`, the verb serves `section()`, and 0.2.53's gap was exactly
    this pair disagreeing), and every key must reach the wire with a unit that states
    it.
    Checked in the order a boot visits these states: fresh, armed, disarmed, measured,
    blind, and degraded.
    """
    cause_keys = ("group_take_unavailable_reasons",
                  "last_group_take_unavailable_reason")
    route = disk_restore.scheduler_section()["load_prune"]
    verb = disk_restore.section()["load_prune"]
    assert set(route) == set(verb), "the route and the verb disagree on the key set"
    for key in cause_keys:
        assert key in route and key in verb, f"{key} is not published by both sites"
        assert "(unit:" in route["units"][key], f"{key} reaches the wire with no unit"
        assert route["units"][key].strip()
    # THE CLOSED VOCABULARY IS THE KEY SET OF THE BREAKDOWN, in every state.
    assert set(disk_restore.TAKE_GATE_UNAVAILABLE_REASONS) == set(BLIND_PATHS)
    assert set(route["group_take_unavailable_reasons"]) == set(BLIND_PATHS)

    with monkeypatch.context() as patch:
        patch.setattr(disk_restore, "prune_switch",
                      lambda: {"enabled": False, "raw": None,
                               "recognised": True, "source": "test",
                               "env_var":
                               "VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS"})
        disarmed = disk_restore.scheduler_section()["load_prune"]
    assert disarmed["enabled"] is False
    assert set(disarmed) == set(route)
    assert set(disarmed["group_take_unavailable_reasons"]) == set(BLIND_PATHS)

    disk_restore.reset_state_for_tests()
    re_armed()
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    Alloc(live_offer("req-clean"), "req-clean").see()
    measured = disk_restore.scheduler_section()["load_prune"]
    assert set(measured) == set(route)
    assert set(measured["group_take_unavailable_reasons"]) == set(BLIND_PATHS)
    assert measured["last_group_take_unavailable_reason"] is None

    blind = see_blind("no_group_block_sizes")
    assert set(blind) == set(route)
    assert set(blind["group_take_unavailable_reasons"]) == set(BLIND_PATHS)
    # THE DEGRADED VIEW: a process that fitted nothing still answers in full.
    disk_restore.reset_state_for_tests()
    re_armed()
    empty = disk_restore.scheduler_section()["load_prune"]
    assert set(empty) == set(route)
    assert set(disk_restore.section()["load_prune"]) == set(route)
    assert sum(empty["group_take_unavailable_reasons"].values()) == 0


def test_the_refusal_and_the_blind_breakdown_do_not_double_count_one_event(armed):
    """ONE EVENT, ONE NAME - the rule `group_take_refusals` follows, kept.

    A blind look is counted twice on purpose exactly once: in the total
    (`group_take_unavailable`) and in its cause (`..._reasons[name]`), which is why the
    breakdown sums to the total rather than exceeding it. A measured look that turns
    into a refusal must NOT touch the blind keys at all: it was not blind, and `refused
    [group_hit_truncated]` already carries that event once.
    """
    fill(armed, blocks=RUN_BLOCKS, groups=HYBRID_CACHEABLE)
    manager = live_offer("req-wire")
    verdict = Alloc(manager, "req-wire").see()
    document = disk_restore.scheduler_section()["load_prune"]
    assert verdict["refused"] is False
    assert document["group_take_unavailable"] == 0
    assert sum(document["group_take_unavailable_reasons"].values()) == 0
    assert document["group_take_observations"] == 1
    # The look was MEASURED: the observations axis moved and the blind axis did not,
    # which is the property that makes `dropped_groups: 0` evidence rather than silence.
    assert document["group_take_dropped_groups"] == 0
    assert "group_take_refusals" not in document, \
        "one event, one name in one document"


def test_every_reason_in_the_vocabulary_has_a_driver_and_an_explanation(armed):
    """NO UNREACHABLE NAME IN A CLOSED VOCABULARY.

    A reason with no driver is a name a future edit can add and never hit, and a name
    with no sentence logs half a line - which is how a counter starts climbing in
    silence again, this time with a label on it.
    """
    for reason in disk_restore.TAKE_GATE_UNAVAILABLE_REASONS:
        assert reason in BLIND_PATHS, \
            f"{reason} is in the vocabulary but no test can drive it"
        sentence = disk_restore.TAKE_GATE_UNAVAILABLE_WHY.get(reason)
        assert sentence and sentence.strip(), f"{reason} has no sentence for the log"
        assert sentence.startswith(("THE UNAUDITED CASE", "THE ORDINARY CASE")), \
            f"{reason} does not say whether the invariant is being watched"
