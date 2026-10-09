r"""SPAN COMPLETENESS AT THE HAND-OFF: is what we spilled restorable at all?

THE QUESTION THIS FILE ASKS. A span on the SSD tier is not one file, it is one file
per prefix-cacheable group, and the restore walk needs them in two DIFFERENT shapes
(`disk_restore._disk_run`, disk_restore.py:5829-5852):

  * a POSITIONAL group (live box: group 5, the attention KV) must have its file for
    EVERY block of the run - the walk stops at the first block missing one;
  * an ALIGN-MODE group (live box: groups 0-3, the recurrent state, whose contract is
    `[null_block] * (n - 1) + [one real state block]`) is needed at ONE block only:
    the run's END. That one file is the ANCHOR. With no anchor at any length the
    offer is refused by name - `no_boundary_anchor`, whose own words say "the
    positional groups have the whole run and NO run end has a boundary-state file in
    every align-mode group".

So a span that arrives with group 5 and no groups 0-3 can extend a run but can never
END one, and a span that arrives with groups 0-3 and no group 5 is not in any run at
all. The per-rank spill journals measured this over 71 hours: of 16,597 spans with an
attention spill, 3,326 (20.0%) never got their align state on ANY rank, and 2,281 got
their align state without attention - about a quarter of everything ever spilled,
dead on arrival, and the direct cause of the `no_boundary_anchor` refusals.

WHAT THIS FILE DOES NOT DO. It does not make a span complete. The align bytes for a
block are in host RAM only when the fork published a boundary hand-off for that
block, and the plugin's only handle on a store is the `(slot, key)` pairs of the
completed event (`disk_store.on_store_completed`, disk_store.py:1263). A pair is
either already in that list - and today it is already spilled - or the copy never
happened and no plugin-side code can reach the bytes. Widening that list is the
fork's decision, so this file measures instead of inventing.

WHAT IT DOES GUARANTEE, one test per numbered claim:

1. THE 20% BECOMES A LIVE NUMBER. Every hand-off is classified into `complete`,
   `spill_incomplete_state` (has attention, cannot anchor) and
   `spill_incomplete_attention` (has state, cannot be in a run), with
   `spill_spans_classified` as the denominator, so the rate is a division and not
   journal archaeology.
2. THE THREE CLASSES PARTITION THE SPANS. One hand-off, one count per span, never two
   classes for one span and never one event counted twice.
3. MEASUREMENT IS STRICTLY ADDITIVE. Classifying changes no spill: the same pairs
   queue the same slots whether the span is complete or orphaned, and no refusal
   counter moves. A `{5}`-only span may get its align state from a later hand-off, so
   refusing it would trade a maybe for a never.
4. AN UNREADABLE GEOMETRY COUNTS NOTHING AND SAYS WHY. The align/positional split is
   the FORK'S OWN reading of its group table (`prefix_cacheable` plus
   `mamba_cache_mode == "align"`, the rule `disk_restore._group_contract` applies to
   the manager). A config that does not answer is a zero with a reason - never a
   guess, and never an exception on the engine thread.
5. THE NUMBERS REACH THE DOORS. `counters`, `hook_section`, `section`, the rank
   file's `disk` block and the cross-rank `spill_rollup` totals, whose `units` map
   says `spans` for each of them.
"""

from __future__ import annotations

import fakes_debug as fake  # the shared fake engine structures
import test_disk_store as store_fixtures  # the module the fixtures live in
from test_disk_spill_rank_rollup import disk_section, write_rank
from test_disk_store import handler, metadata, metadata_shape, store_files

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    cpu_capacity,
    disk_spill,
    disk_store,
)

# THE FIXTURES, RE-EXPORTED BY NAME. `store_files`/`handler`/`metadata` are ordinary
# helpers and travel as plain imports; these three are FIXTURES, so they have to be
# names in this module's namespace for pytest to see them - and `tier` in particular
# is the armed store every test here runs against.
#
# `none_hash_seed` is the one that makes the difference between a test and a ceremony:
# without it the fingerprint builder refuses EVERY bind (`none_hash_seed` is a required
# field read from the fork's `kv_cache_utils`), so no spill path runs, every counter
# stays zero for the wrong reason, and a test of an unreachable counter goes green. It
# is the fixture-that-cannot-fail trap in its purest form, and it is why these lines
# are re-exports rather than a local re-implementation.
none_hash_seed = store_fixtures.none_hash_seed  # noqa: F811 - autouse: the fingerprint needs the fork's seed module
no_fake_distributed_state = store_fixtures.no_fake_distributed_state  # noqa: F811 - autouse: no fake parallel_state leaks into a rank answer
tier = store_fixtures.tier  # noqa: F811 - the configured, ARMED store under tmp_path

# THE LIVE GEOMETRY, as `/plugins/kvtier/disk?op=status` reported it on the box on
# 2026-10-09: six groups, `group_block_sizes` [3200,3200,3200,3200,8,3200], groups
# 0-3 `boundary_state` with `mamba_cache_mode: align`, group 4 positional but NOT
# prefix-cacheable (the parked QSA raw-key ring, which a restore never asks for),
# group 5 positional and prefix-cacheable. Every test builds this through the
# handler's own `kv_cache_config`, because that is the only place the plugin is
# allowed to learn it from.
LIVE_SIZES = [3200, 3200, 3200, 3200, 8, 3200]
LIVE_CACHEABLE = [True, True, True, True, False, True]
LIVE_MODE = ["align", "align", "align", "align", None, None]
ALIGN_GROUPS = (0, 1, 2, 3)
ATTENTION_GROUP = 5

SPAN_A = "a3f0b7c1d4e5f607" + "1" * 48      # one token boundary = one span
SPAN_B = "0f1e2d3c4b5a6978" + "2" * 48
SPAN_C = "1111222233334444" + "0" * 48


def key_for(digest: str, group: int) -> str:
    """The engine's group-qualified block hash for one span in one group.

    The digest, then the 4-byte big-endian group id - exactly as
    `kv_cache_utils.make_block_hash_with_group_id` packs it and as
    `disk_layout.split_block_key` reads it back. Nothing here invents an identity:
    the span's name is the digest the engine computed, and the group comes from the
    suffix, never from a position the plugin would have to guess.
    """
    return f"{digest}{group:08x}"


def live_handler(**overrides):
    """A worker handler whose group table is the live box's."""
    groups = [fake.Obj(kv_cache_spec=fake.Obj(block_size=size,
                                              prefix_cacheable=cacheable,
                                              mamba_cache_mode=mode))
              for size, cacheable, mode in zip(LIVE_SIZES, LIVE_CACHEABLE,
                                               LIVE_MODE, strict=True)]
    # `pop`, not `**overrides` on top: a test that wants a handler WITHOUT a group
    # table says `kv_cache_config=None`, and "the table is absent" is a different
    # world from "the table is the live box's".
    table = overrides.pop("kv_cache_config",
                          fake.Obj(kv_cache_groups=groups))
    return handler(kv_cache_config=table, **overrides)


def spill_a_handoff(role, pairs, event=7):
    """Bind one step, complete it, and return the hook's report.

    Slots stay inside the fake region (8 wide) and the pairing is the fork's:
    `store_cpu_blocks[i]` is `store_keys[i]`.
    """
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[slot for slot, _key in pairs],
                        store_keys=[key for _slot, key in pairs],
                        store_event=event)
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        return disk_store.on_store_completed(role, [])


def shape_counters() -> dict:
    state = disk_store.counters()
    return {name: state[name] for name in disk_store.SPAN_SHAPE_COUNTERS}


# --- 1. the classification is the fork's shape, from the fork's table ---------

def test_a_handoff_carrying_the_align_state_is_a_complete_span(tier):
    """Five groups for one span: it can be in a run AND end one.

    This is the 80% case and it must count ZERO incompleteness - a counter that
    cries wolf on a complete span is worse than no counter, because the next reader
    stops believing the 20%.
    """
    role = live_handler()
    pairs = [(slot, key_for(SPAN_A, group))
             for slot, group in enumerate(ALIGN_GROUPS + (ATTENTION_GROUP,))]
    report = spill_a_handoff(role, pairs)

    assert report["queued"] == 5, report
    assert shape_counters() == {"spill_spans_classified": 1,
                                "spill_spans_complete": 1,
                                "spill_incomplete_state": 0,
                                "spill_incomplete_attention": 0}
    assert report["spans"] == {"classified": 1, "complete": 1,
                               "incomplete_state": 0,
                               "incomplete_attention": 0}


def test_a_handoff_without_the_align_state_is_an_incomplete_state_span(tier):
    """THE 20% SHAPE: attention with no boundary state anywhere in the hand-off.

    On the live box this is the span behind
    `no_boundary_anchor: the positional groups have the whole run but NO run end
    within the offer's cap has a boundary-state file in every align-mode group`.
    """
    role = live_handler()
    report = spill_a_handoff(role, [(0, key_for(SPAN_A, ATTENTION_GROUP))])

    assert report["queued"] == 1, report
    assert shape_counters() == {"spill_spans_classified": 1,
                                "spill_spans_complete": 0,
                                "spill_incomplete_state": 1,
                                "spill_incomplete_attention": 0}


def test_partial_align_state_is_still_an_incomplete_span(tier):
    """Groups 0-2 without group 3 is not an anchor either.

    The walk's anchor test is `for group in boundary: ... if state != "hit": whole =
    False` (disk_restore.py:5829-5846): EVERY align-mode group must answer at the run
    end, because a run missing one group's recurrent state is "a claim on a recurrent
    state nobody wrote" - the 00:08 wrong-answer incident, not a shorter offer.
    """
    role = live_handler()
    pairs = [(slot, key_for(SPAN_A, group)) for slot, group in enumerate((0, 1, 2))]
    pairs.append((3, key_for(SPAN_A, ATTENTION_GROUP)))
    spill_a_handoff(role, pairs)

    assert shape_counters()["spill_incomplete_state"] == 1
    assert shape_counters()["spill_spans_complete"] == 0


def test_a_handoff_without_the_attention_block_is_an_incomplete_attention_span(tier):
    """The mirror shape: state with no positional file cannot be in a run at all.

    2,281 spans in the journals. It can never be walked over, so it can never be an
    anchor either - dead on arrival from the other side.
    """
    role = live_handler()
    pairs = [(slot, key_for(SPAN_A, group))
             for slot, group in enumerate(ALIGN_GROUPS)]
    spill_a_handoff(role, pairs)

    assert shape_counters() == {"spill_spans_classified": 1,
                                "spill_spans_complete": 0,
                                "spill_incomplete_state": 0,
                                "spill_incomplete_attention": 1}


def test_a_parked_group_does_not_make_a_span_look_complete(tier):
    """Group 4 is positional but NOT prefix-cacheable: the tier never asks for it.

    The live box's `prefix_cacheable_group_ids` is [0,1,2,3,5], and group 4 stays out
    of every restore. A span of group 4 alone is therefore not a span here: it
    classifies as nothing, not "complete", or the denominator would fill with files
    no reader will ever want.
    """
    role = live_handler()
    spill_a_handoff(role, [(0, key_for(SPAN_A, 4))])

    assert shape_counters() == {"spill_spans_classified": 0,
                                "spill_spans_complete": 0,
                                "spill_incomplete_state": 0,
                                "spill_incomplete_attention": 0}


# --- 2. the three classes partition the spans, and nothing double-counts ------

def test_the_three_classes_partition_every_span_of_a_handoff(tier):
    """One hand-off holding one of each shape: 3 = 1 + 1 + 1, nothing else.

    The counters exist to be divided, so the invariant is the partition rather than
    any single number, and a span in one class is in no other.
    """
    role = live_handler()
    pairs = [(slot, key_for(SPAN_A, group))
             for slot, group in enumerate(ALIGN_GROUPS + (ATTENTION_GROUP,))]
    pairs.append((5, key_for(SPAN_B, ATTENTION_GROUP)))
    pairs += [(6, key_for(SPAN_C, group)) for group in ALIGN_GROUPS[:2]]
    report = spill_a_handoff(role, pairs)

    counters = shape_counters()
    assert counters == {"spill_spans_classified": 3,
                        "spill_spans_complete": 1,
                        "spill_incomplete_state": 1,
                        "spill_incomplete_attention": 1}
    assert (counters["spill_spans_classified"] ==
            counters["spill_spans_complete"] +
            counters["spill_incomplete_state"] +
            counters["spill_incomplete_attention"])
    assert report["spans"]["classified"] == 3


def test_a_second_handoff_adds_once_and_a_repeated_event_adds_nothing(tier):
    """Once per HAND-OFF, and an event the fork has not drained is one hand-off.

    `_state["handled"]` already stops a completion spilling the same slot twice
    across steps; the completeness tally rides on that same de-duplication, or the
    dashboard's rate would depend on how often the fork polls its store stream.
    """
    role = live_handler()
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[0],
                        store_keys=[key_for(SPAN_A, ATTENTION_GROUP)],
                        store_event=9)
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        disk_store.on_store_completed(role, [])
        assert shape_counters()["spill_incomplete_state"] == 1
        # The same index, still sitting in the fork's dict on the next step.
        disk_store.on_store_completed(role, [])
        assert shape_counters()["spill_incomplete_state"] == 1
        # A DIFFERENT event with the same shape is a different hand-off: it adds.
        step2 = metadata(store_cpu_blocks=[1],
                         store_keys=[key_for(SPAN_B, ATTENTION_GROUP)],
                         store_event=10)
        disk_store.observe_bind(role, step2)
        role._completed_store_events = {step2.store_event: 1}
        disk_store.on_store_completed(role, [])
    assert shape_counters() == {"spill_spans_classified": 2,
                                "spill_spans_complete": 0,
                                "spill_incomplete_state": 2,
                                "spill_incomplete_attention": 0}


# --- 3. measurement is additive: it never costs a spill ----------------------

def test_classifying_never_costs_a_spill(tier):
    """THE ADDITIVE-ONLY RULE, proved by comparison rather than by assertion.

    A `{5}`-only span may receive its align state from a LATER hand-off, so refusing
    or dropping it would turn a maybe into a guaranteed loss - the prohibition this
    change is written under. The orphaned hand-off and the complete one must reach
    the same spill outcome, slot for slot, with every refusal counter still zero.
    """
    orphaned = live_handler()
    orphan_report = spill_a_handoff(orphaned,
                                    [(0, key_for(SPAN_A, ATTENTION_GROUP))])
    complete = live_handler()
    complete_report = spill_a_handoff(
        complete, [(slot, key_for(SPAN_A, group))
                   for slot, group in enumerate(ALIGN_GROUPS + (ATTENTION_GROUP,))],
        event=8)

    assert orphan_report["queued"] == 1 and orphan_report["spilled"] == 1
    assert complete_report["queued"] == 5 and complete_report["spilled"] == 5
    # Neither hand-off was refused, capped or dropped: the ONLY difference between
    # them is the tally, which is exactly what "additive" has to mean here.
    for event in orphan_report["events"] + complete_report["events"]:
        assert event.get("reason_kind") is None or event["spilled"] > 0
        assert event["spilled"] > 0, event
    assert disk_store.counters()["refusals_total"] == 0
    assert disk_store.counters()["slots_capped"] == 0
    assert sum(disk_spill.counters()["refused"].values()) == 0
    assert disk_spill.counters()["drops"] == 0


def test_the_files_of_an_incomplete_span_are_written_like_any_other(tier):
    """The orphan's bytes still land: counting changed the write path not at all.

    Counting incompleteness must not quietly become "don't bother writing what cannot
    anchor". The store is walked to prove it, not the report.
    """
    role = live_handler()
    spill_a_handoff(role, [(0, key_for(SPAN_A, ATTENTION_GROUP))])
    assert disk_spill.drain()["drained"] is True

    files = store_files(tier)
    assert len(files) == 1, files
    ((relative, payload),) = files.items()
    assert "-g5-" in relative and SPAN_A[:16] in relative
    assert payload == role.cpu_kv_caches["g0"].slot_bytes(0)
    assert disk_spill.counters()["spilled"] == 1


def test_the_tick_cap_does_not_change_what_the_hand_off_was(tier, monkeypatch):
    """The tally describes the HAND-OFF the engine made, not the truncated loop.

    `MAX_SLOTS_PER_TICK` bounds how much one step may hand to the writer thread; the
    span shape is a property of what the fork completed. Keeping those two separate
    is what stops the counter blaming the tier for its own cap.
    """
    monkeypatch.setattr(disk_store, "MAX_SLOTS_PER_TICK", 2)
    role = live_handler()
    pairs = [(slot, key_for(SPAN_A, group))
             for slot, group in enumerate(ALIGN_GROUPS + (ATTENTION_GROUP,))]
    report = spill_a_handoff(role, pairs)

    assert report["queued"] == 2
    assert disk_store.counters()["slots_capped"] == 3
    assert shape_counters() == {"spill_spans_classified": 1,
                                "spill_spans_complete": 1,
                                "spill_incomplete_state": 0,
                                "spill_incomplete_attention": 0}


# --- 4. an unreadable geometry counts nothing, and says why ------------------

def test_a_group_table_without_align_mode_counts_nothing_and_says_why(tier):
    """Today's `handler()` fake: prefix-cacheable groups, no `mamba_cache_mode`.

    Without the fork's own answer for what a group IS, "which group must the anchor
    name" is a guess, and a guessed denominator is worse than none. The counters stay
    at zero and `span_shapes` carries the reason; the spill itself is untouched,
    because a witness that gates the write is not a witness.
    """
    role = handler()   # GROUP_SIZES, prefix_cacheable for index < 3, no mode field
    report = spill_a_handoff(role, [(0, key_for(SPAN_A, 0)),
                                    (1, key_for(SPAN_A, 1))])

    assert report["queued"] == 2
    assert shape_counters() == {"spill_spans_classified": 0,
                                "spill_spans_complete": 0,
                                "spill_incomplete_state": 0,
                                "spill_incomplete_attention": 0}
    view = disk_store.hook_section()["span_shapes"]
    assert view["classified"] == 0
    assert view["reason"] and "align" in view["reason"]
    assert view["basis"] is None


def test_a_handler_without_a_group_table_counts_nothing(tier):
    """No group table: the classifier refuses, and the pre-existing gate already did.

    Worth stating out loud because it is the ONLY reason a worker with no
    `kv_cache_config` spills nothing today: the fingerprint builder needs the same
    table for `group_block_sizes`, so the bind is refused as `fingerprint_incomplete`
    before any completion happens. This change adds no second gate - the refusal
    reason is the old one, and the tally simply never gets an `ok` shape to count.
    """
    role = live_handler(kv_cache_config=None)
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[0],
                        store_keys=[key_for(SPAN_A, ATTENTION_GROUP)],
                        store_event=7)
        bound = disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        report = disk_store.on_store_completed(role, [])

    assert bound["reason_kind"] == "fingerprint_incomplete", bound
    assert report["queued"] == 0 and report["spans"]["classified"] == 0
    assert shape_counters() == {name: 0 for name in disk_store.SPAN_SHAPE_COUNTERS}
    refused = disk_store.span_shape_state(role)
    assert refused["ok"] is False and "kv_cache_groups" in refused["reason"]


def test_the_classifier_survives_pairs_it_cannot_read(tier):
    """Junk pairs are IGNORED: never an exception, never a span of their own.

    `on_store_completed` runs on the engine thread inside `get_finished`, where an
    `AttributeError` is a step fault on a box serving traffic. A 64-hex key carries no
    group id, an int is not a key, `None` is an unnamed slot, and a group id the table
    does not have is not a group - all four are dropped and counted as ignored.
    """
    shapes = disk_store.span_shape_state(live_handler())
    assert shapes["ok"] is True
    tally = disk_store.classify_handoff(
        ((0, SPAN_A),                      # digest with no group suffix
         (1, 12345),                       # not a string
         (2, key_for(SPAN_B, 9)),          # a group the table does not have
         (3, None),                        # a slot the engine could not name
         (4, key_for(SPAN_C, ATTENTION_GROUP))),   # the one real pair
        shapes)
    assert tally["spans"] == 1
    assert tally["incomplete_state"] == 1
    assert tally["incomplete_attention"] == 0
    assert tally["complete"] == 0
    assert tally["ignored_pairs"] == 4


# --- 5. the numbers reach the doors the dashboard reads ----------------------

def test_the_counters_are_published_in_the_hooks_own_sections(tier):
    """`counters()`, `hook_section()` and the snapshot's wiring block.

    `hook` is the block `op:status` and `/snapshot` publish, and a counter that lives
    only in `_state` is a counter nobody reads - the 0.2.25 lesson, restated in this
    module's own payload.
    """
    role = live_handler()
    spill_a_handoff(role, [(0, key_for(SPAN_A, ATTENTION_GROUP))])
    counters = disk_store.counters()
    hook = disk_store.hook_section()
    section = disk_store.section()

    assert counters["spill_incomplete_state"] == 1
    for name in disk_store.SPAN_SHAPE_COUNTERS:
        assert counters[name] == hook[name] == section[name], name
    assert hook["span_shapes"]["align_group_ids"] == list(ALIGN_GROUPS)
    assert hook["span_shapes"]["positional_group_ids"] == [ATTENTION_GROUP]
    assert hook["span_shapes"]["basis"]


def test_the_counters_travel_in_the_rank_file_and_roll_up_across_ranks(
        tier, monkeypatch, tmp_path):
    """The fleet's numbers, where `spilled` and `drops` live.

    Per-process counters on a four-rank boot are FOUR FACTS, which is why
    `spill_rollup` exists: the dashboard reads the summed view, so a completeness
    counter that only lands in the answering process repeats the 2026-10-05 "every
    number zero" failure with a nicer name.
    """
    from kvtier_dev_plugin import config

    monkeypatch.setitem(cpu_capacity._state, "reported", {})
    role = live_handler()
    spill_a_handoff(role, [(0, key_for(SPAN_A, ATTENTION_GROUP))])
    report = disk_store.worker_report()
    for name in disk_store.SPAN_SHAPE_COUNTERS:
        assert name in report, sorted(report)[:5]
    assert report["spill_incomplete_state"] == 1

    # Every field the rollup reads is a field the writer writes, in the same place.
    sources = {name: path for name, path, _unit in disk_store.RANK_ROLLUP_FIELDS}
    for name in disk_store.SPAN_SHAPE_COUNTERS:
        assert name in sources, "a rollup field the writer does not emit"
        assert sources[name] == ("disk", name)
        assert disk_store.RANK_ROLLUP_MEANINGS[name]

    directory = tmp_path / "fleet"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    for rank, state, attention, classified, complete in (
            ("PP0_TP0", 4, 1, 20, 15), ("PP0_TP1", 6, 2, 30, 22)):
        write_rank(directory, rank, disk=disk_section(
            spill_incomplete_state=state, spill_incomplete_attention=attention,
            spill_spans_classified=classified, spill_spans_complete=complete,
            files_written=classified * 5))
    rollup = disk_store.rank_rollup(force=True)

    assert rollup["totals"]["spill_incomplete_state"] == 10
    assert rollup["totals"]["spill_incomplete_attention"] == 3
    assert rollup["totals"]["spill_spans_classified"] == 50
    assert rollup["units"]["spill_incomplete_state"] == "spans"
    assert rollup["units"]["spill_spans_classified"] == "spans"
    assert rollup["kinds"]["spill_incomplete_state"] == "counter-since-boot"
    assert "anchor" in rollup["meanings"]["spill_incomplete_state"]


def test_the_two_readings_of_the_forks_align_mode_cannot_drift():
    """`disk_store` keeps its OWN copy of the align-mode string, so prove they match.

    `disk_restore.MAMBA_CACHE_MODE_ALIGN` is the restore side's reading of the same
    fork field, and the spill side cannot import it at module scope without the
    spill hook dragging the restore layer onto the step thread. That duplication is
    only safe while both spell the fork's value the same way.
    """
    from kvtier_dev_plugin import disk_restore

    assert disk_store.MAMBA_CACHE_MODE_ALIGN == disk_restore.MAMBA_CACHE_MODE_ALIGN


def test_the_classification_costs_nothing_on_a_step_with_nothing_to_say(tier):
    """An unclassified geometry repeated per completion is not a leak or a stall.

    `_span_shapes_for` retries a FAILED read on every hand-off (a worker that binds
    before its config exists must heal), so the failure path must stay one attribute
    walk per completion rather than a growing cache or a second exception path. A
    hundred hand-offs leave the cache holding one dict and the counters at zero.
    """
    role = handler()
    for event in range(1, 101):
        spill_a_handoff(role, [(0, key_for(SPAN_A, 0))], event=event)

    assert shape_counters() == {"spill_spans_classified": 0,
                                "spill_spans_complete": 0,
                                "spill_incomplete_state": 0,
                                "spill_incomplete_attention": 0}
    assert disk_store._span_shape_cache["shapes"] is not None
    assert disk_store._span_shape_cache["shapes"]["ok"] is False
    assert disk_store.counters()["span_shape_pairs_ignored"] == 0
