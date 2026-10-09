r"""RESOLVING `mamba_cache_mode` WHERE THE WORKER CAN ACTUALLY SEE IT.

THE PANEL THAT COULD NOT MOVE, AND WHY. Five counters (
`spill_spans_classified`, `spill_spans_complete`, `spill_incomplete_state`,
`spill_incomplete_attention`, `anchor_spills_completed`) and `disk_restore_verify`'s
align half (`boundary_groups_known`, `boundary.groups`) all derive from ONE predicate:
"is this group's spec in align mode?". Both modules asked the GROUP's spec -
`getattr(spec, "mamba_cache_mode", None) == "align"` - and on the live worker that
answers None for every group, so every prefix-cacheable group landed in `positional`,
no group landed in `align`, `span_shape_state()` refused, and `_tally_span_shapes`
contributed nothing (its fail-open contract: "report zeros you can explain"). Their
zeros were read on 2026-10-09 as "the anchor patch did nothing".

THE LIVE EVIDENCE THAT IT IS THE PREDICATE AND NOT A MISSING CONFIG, gathered
read-only on the box on 2026-10-09 (nothing here invents a shape; where a test
simulates one, its own docstring says so):

  * `/tmp/kv-tier-diagnostics/rank-PP0_TP0.json`, `disk_restore_verify.geometry`:
    `group_bytes: [11468800, 9830400, 9830400, 1638400, 13824, 11059200]` with
    `boundary_groups: []` and `boundary_note: "no group's spec reports 'align'
    mamba_cache_mode"` - the same predicate, witnessed by the same handler that
    publishes `disk.fingerprint_ready: true` off that very group table (which rules
    out "the worker has no config" and "prefix_cacheable is not a bool": the
    fingerprint builder reads both from the same six specs and refused nothing).
  * `GET /plugins/kvtier/kvconfig` (loopback, EngineCore process):
    `config.mamba_cache_mode: "align"`, groups 0-3 `spec_class: MambaSpec` with
    `spec.mamba_cache_mode: "align"` and `page_size_bytes: 1638400` each, group 4
    `CircularBufferSpec`, group 5 `MLAAttentionSpec`, `layer_count` [7,6,6,1,6,12].
  * THE TWO SIDES ARE DIFFERENT OBJECTS, and the fork says so in its own words:
    `v1/core/kv_cache_utils.py:1075-1077` - "whether a group carries an aggregated
    `UniformTypeKVCacheSpecs` (worker config) or a representative per-layer spec
    (scheduler config)" - and `_project_kv_cache_groups_to_worker`
    (`kv_cache_utils.py:2606-2614`) rebuilds the worker's group spec AS a
    `UniformTypeKVCacheSpecs`. The arithmetic agrees: the worker's group 0 is
    11,468,800 = 7 x 1,638,400, its group 3 is 1 x 1,638,400 and its group 5 is
    12 x 921,600 = 11,059,200 - and `UniformTypeKVCacheSpecs.page_size_bytes` IS
    `sum(spec.page_size_bytes ...)` (`v1/kv_cache_interface.py:1205`), while the
    container declares no `mamba_cache_mode` of its own (only `MambaSpec` declares it,
    `kv_cache_interface.py:998`, and `abstract.py:78` fills it from
    `vllm_config.cache_config.mamba_cache_mode`).

WHAT THAT LEAVES UNVERIFIED, STATED BECAUSE IT IS THE PREMISE OF THE FIX: no live
field publishes the worker's spec CLASS NAME. The arithmetic, the fork's projection
code and its comment are the evidence. The fix is therefore defensive in both
directions AND publishes the class names it observes, so the next boot reads its own
shape instead of an operator inferring it from page-size ratios. Tests here assert the
published strings as well as the classification: a panel that works again but still
cannot say what it read is half a fix.

WHAT THIS FILE GUARANTEES, one test per numbered claim:

1. THE MODE IS RESOLVED WHERE IT LIVES - the group's own field, else its sub-specs,
   else the engine-wide `cache_config`, else nothing - by ONE helper both sides call,
   and the output says which source answered.
2. THE ENGINE-WIDE FALLBACK IS LABELLED AS ENGINE-WIDE. It may never turn a group whose
   spec names no state class into an align group, and the basis says it is not
   per-group evidence.
3. NEITHER SOURCE IS TODAY'S BEHAVIOUR: zeros plus a readable reason, no exception, and
   a `BaseException` still propagates.
4. THE EXPLANATION REACHES THE RANK FILE AND THE ROLLUP: `span_shape_reason`,
   `span_shape_basis` and `span_shape_pairs_ignored` are written by `worker_report()`
   AND read by `_rank_file_rollup_row()`, because a field absent from that whitelist is
   silently never rolled up - no error, no gap, no `fields_rejected` note - which is
   how the dark panel stayed invisible for a day.
5. THE RESTORE-VERIFY SIDE CANNOT DRIFT from the store side.
6. OBSERVABILITY ONLY: the resolution touches no arm, no posture, no counter.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
import types

import fakes_debug as fake  # the shared fake engine structures
import test_disk_store as store_fixtures
import test_spill_span_completeness as span_fixtures
from test_disk_spill_rank_rollup import disk_section, write_rank

# THE FIXTURES, RE-EXPORTED BY NAME - the idiom `test_spill_span_completeness.py`
# itself uses, and for the same reason: `handler`/`metadata_shape` are ordinary
# helpers and travel as plain imports, while `tier` and the two AUTUSE fixtures have
# to be names in THIS module's namespace for pytest to see them. `none_hash_seed` in
# particular is the fixture-that-cannot-fail trap: without it the fingerprint builder
# refuses every bind, no spill path runs, and a test of an unreachable counter goes
# green for the wrong reason.
handler = store_fixtures.handler
metadata_shape = store_fixtures.metadata_shape
none_hash_seed = store_fixtures.none_hash_seed  # noqa: F811 - autouse: the seed
no_fake_distributed_state = store_fixtures.no_fake_distributed_state  # noqa: F811
tier = store_fixtures.tier  # noqa: F811 - the armed store under tmp_path

# THE LIVE GEOMETRY AND THE HAND-OFF HELPERS, from the file that pins the per-layer
# version of the same table (`LIVE_SIZES` is the box's per-group block size, and
# `spill_a_handoff` drives the real bind-and-complete path).
ATTENTION_GROUP = span_fixtures.ATTENTION_GROUP
LIVE_SIZES = span_fixtures.LIVE_SIZES
SPAN_A = span_fixtures.SPAN_A
key_for = span_fixtures.key_for
shape_counters = span_fixtures.shape_counters
spill_a_handoff = span_fixtures.spill_a_handoff

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    config,
    disk_restore_verify,
    disk_spill,
    disk_store,
    worker_diag,
)

# THE LIVE GROUP TABLE, as `/plugins/kvtier/kvconfig` reported it: six groups with
# block sizes [3200,3200,3200,3200,8,3200]; groups 0-3 prefix-cacheable state groups,
# group 4 the parked QSA ring (positional, NOT prefix-cacheable), group 5 the
# attention KV. Same numbers `test_spill_span_completeness.py` pins.
ALIGN_IDS = [0, 1, 2, 3]
MAMBA_LAYER_PAGE = 1_638_400          # one layer's state page, groups 0-3
ATTENTION_LAYER_PAGE = 921_600        # one layer's MLA page, group 5
SCRATCH_LAYER_PAGE = 2_304            # one layer of the parked ring, group 4
LAYER_COUNTS = [7, 6, 6, 1, 6, 12]    # the rank file's own per-group layer counts
CONTAINER_CLASS = "UniformTypeKVCacheSpecs"


# --- the worker's spec shape, built from the fork's shape --------------------

def sub_spec(cls: str, *, mode=None, block_size=3200, page=MAMBA_LAYER_PAGE,
             cacheable=True):
    """One PER-LAYER spec, as `get_kv_cache_spec` builds it (`abstract.py:70-86`).

    `MambaSpec` declares `mamba_cache_mode`; an attention spec declares no such field
    at all, which is the fork's way of saying "positional". The class name is carried
    for real - `fakes_debug.Obj(cls)` returns an instance of a class literally named
    `cls` - because publishing the class name is requirement 4 of this change.
    """
    fields = {"block_size": block_size, "prefix_cacheable": cacheable,
              "page_size_bytes": page}
    if mode is not None:
        fields["mamba_cache_mode"] = mode
    return fake.Obj(cls, **fields)


def container(subs, *, block_size=3200, expose="both"):
    """A MULTI-LAYER CONTAINER spec: what `UniformTypeKVCacheSpecs` is, and carries.

    NOT INVENTED. `kv_cache_interface.py:1185-1206` gives the container
    `kv_cache_specs: dict[str, KVCacheSpec]`, a `first_spec` property, a
    `prefix_cacheable` that is `all(...)` over the sub-specs and a `page_size_bytes`
    that SUMS them - and does NOT give it the `mamba_cache_mode` field `MambaSpec`
    declares at :998. `expose` covers a container variant that publishes one accessor
    and not the other, so the fix cannot depend on exactly one spelling.
    """
    fields = {"block_size": block_size,
              "prefix_cacheable": all(bool(getattr(s, "prefix_cacheable", False))
                                      for s in subs),
              "page_size_bytes": sum(int(s.page_size_bytes) for s in subs)}
    if expose in ("dict", "both"):
        fields["kv_cache_specs"] = {f"layers.{name}": s
                                    for name, s in enumerate(subs)}
    if expose in ("first", "both"):
        fields["first_spec"] = subs[0]
    return fake.Obj(CONTAINER_CLASS, **fields)


def live_sub_specs(group: int) -> list:
    """The per-layer specs the live box's group `group` aggregates."""
    if group in ALIGN_IDS:
        return [sub_spec("MambaSpec", mode="align", block_size=LIVE_SIZES[group])
                for _ in range(LAYER_COUNTS[group])]
    if group == 4:
        return [sub_spec("CircularBufferSpec", block_size=LIVE_SIZES[group],
                         page=SCRATCH_LAYER_PAGE, cacheable=False)
                for _ in range(LAYER_COUNTS[group])]
    return [sub_spec("MLAAttentionSpec", block_size=LIVE_SIZES[group],
                     page=ATTENTION_LAYER_PAGE)
            for _ in range(LAYER_COUNTS[group])]


def _wrap(specs, *, engine_mode="align", with_vllm_config=True):
    """A worker handler around one list of group specs, with the fork's `vllm_config`.

    `handler()` from `test_disk_store.py` supplies everything the fingerprint builder
    reads (`model_config`, `cache_config`, the parallel fields, the CPU region), so a
    container group table goes down the SAME spill path the per-layer one does - only
    the group specs differ.
    """
    groups = [fake.Obj(kv_cache_spec=spec) for spec in specs]
    role = handler(kv_cache_config=fake.Obj(kv_cache_groups=groups))
    if with_vllm_config:
        role.vllm_config.cache_config.mamba_cache_mode = engine_mode
    else:
        role.vllm_config = None
    return role


def container_handler(**overrides):
    """A worker whose six group specs are CONTAINERS - the shape the live worker has.

    Identical to `test_spill_span_completeness.live_handler()` in group count, block
    sizes and `prefix_cacheable` answers, and identical to the fork's worker in keeping
    `vllm_config` (`worker.py:40`). The one difference is the defect: no group spec
    carries `mamba_cache_mode`; its sub-specs do.
    """
    return _wrap([container(live_sub_specs(index), block_size=LIVE_SIZES[index])
                  for index in range(6)], **overrides)


def attention_container(**kwargs):
    """A container of attention sub-specs: positional, and silent about its mode."""
    return container([sub_spec("MLAAttentionSpec", page=ATTENTION_LAYER_PAGE)
                      for _ in range(3)], **kwargs)


# --- 1. the mode is resolved where it lives, and says where ------------------

def test_the_helper_resolves_a_mode_the_group_spec_does_not_carry():
    """`resolve_mamba_cache_mode` is the shared answer, and it names its source.

    Against the unpatched module there is no such function - which IS the defect: two
    modules each asked the container separately and each got None, so both panels went
    dark together.
    """
    resolved = disk_store.resolve_mamba_cache_mode(
        container(live_sub_specs(0)), None)

    assert resolved["mode"] == "align"
    assert resolved["source"] == "sub-specs"
    assert resolved["spec_class"] == CONTAINER_CLASS
    assert "MambaSpec" in resolved["sub_spec_classes"]
    assert "MambaSpec" in resolved["detail"]


def test_a_spec_that_declares_its_own_mode_is_still_read_directly():
    """THE SOURCE THAT WORKS TODAY KEEPS WORKING, AND WINS: the spec's own word.

    The scheduler-side config and every `MambaSpec` the fork builds answer the direct
    read (`kv_cache_interface.py:998`, filled from the engine-wide field at
    `abstract.py:78`), so the resolution must not start guessing when the group
    itself answered - here it says "align" while the engine-wide flag says "none".
    """
    resolved = disk_store.resolve_mamba_cache_mode(
        sub_spec("MambaSpec", mode="align"),
        fake.Obj(cache_config=fake.Obj(mamba_cache_mode="none")))

    assert resolved["mode"] == "align"
    assert resolved["source"] == "spec"


def test_the_live_worker_shape_yields_a_usable_basis_today_it_refuses():
    """THE BUG IN ONE ASSERTION. Six container groups must classify align [0-3].

    Against the unpatched tree `span_shape_state()` answers `ok: False` with "no
    prefix-cacheable group reports mamba_cache_mode == 'align'" - a reason published
    only through `hook_section()`, in the engine process, where the worker's cache is
    empty, so it could never be read. The fix must classify the same handler AND name
    the container and the sub-spec class it read, which is how the next boot tells us
    the worker's real spec shape.
    """
    shapes = disk_store.span_shape_state(container_handler())

    assert shapes["ok"] is True, shapes["reason"]
    assert shapes["align_group_ids"] == ALIGN_IDS
    assert shapes["positional_group_ids"] == [ATTENTION_GROUP]
    assert shapes["prefix_cacheable_group_ids"] == [0, 1, 2, 3, ATTENTION_GROUP]
    assert CONTAINER_CLASS in shapes["basis"]
    assert "MambaSpec" in shapes["basis"]
    assert "sub-spec" in shapes["basis"]


def test_a_container_exposing_only_first_spec_still_resolves():
    """A container publishing `first_spec` and not the dict is the same fact.

    `kv_cache_interface.py:1200-1205` gives the container BOTH accessors. Reading only
    one is a dependency on a fork not deleting a property, and a dark panel is exactly
    the cost of that dependency.
    """
    resolved = disk_store.resolve_mamba_cache_mode(
        container(live_sub_specs(3), expose="first"), None)

    assert resolved["mode"] == "align"
    assert resolved["source"] == "sub-specs"

    shapes = disk_store.span_shape_state(
        _wrap([container(live_sub_specs(index), block_size=LIVE_SIZES[index],
                         expose="first") for index in range(6)]))
    assert shapes["ok"] is True, shapes["reason"]
    assert shapes["align_group_ids"] == ALIGN_IDS


def test_the_basis_names_only_the_groups_that_were_actually_classified_align():
    """A group that resolves to a NON-align mode is never named in the align clause.

    Three containers: one layer says "align", one says "none", one says nothing. The
    first is align, the other two are positional - and the basis must list exactly the
    first in its "align ... from sub-specs" clause, because the basis is the field a
    reader uses to audit the classification and a surplus id there is a claim about a
    group the walk put in the other list.
    """
    specs = [container([sub_spec("MambaSpec", mode="align")]),
             container([sub_spec("MambaSpec", mode="none")]),
             container([sub_spec("MLAAttentionSpec",
                                 page=ATTENTION_LAYER_PAGE)])]
    shapes = disk_store.span_shape_state(_wrap(specs))

    assert shapes["ok"] is True, shapes["reason"]
    assert shapes["align_group_ids"] == [0]
    assert shapes["positional_group_ids"] == [1, 2]
    clause = shapes["basis"].split("align ", 1)[1].split(" from ", 1)[0]
    assert clause == "0", shapes["basis"]


def test_a_spill_on_the_worker_shape_is_classified_today_it_is_not(tier):
    """THE COUNTERS MOVE. A complete span on the container shape counts as complete.

    The end-to-end form of the bug: this is the same hand-off
    `test_spill_span_completeness.py::test_a_handoff_carrying_the_align_state_is_a_compl
    ete_span` proves on a per-layer table, and on the table the worker really has it
    counts nothing at all. Unpatched every number below is zero, which is precisely the
    reading that was mistaken for "the patch did nothing".
    """
    role = container_handler()
    pairs = [(slot, key_for(SPAN_A, group))
             for slot, group in enumerate(ALIGN_IDS + [ATTENTION_GROUP])]
    with metadata_shape(True):
        report = spill_a_handoff(role, pairs)

    assert report["queued"] == 5, report
    assert shape_counters() == {"spill_spans_classified": 1,
                                "spill_spans_complete": 1,
                                "spill_incomplete_state": 0,
                                "spill_incomplete_attention": 0}


# --- 2. the engine-wide fallback is labelled, and never overclaims -----------

def test_the_engine_wide_config_is_a_labelled_fallback_without_sub_specs():
    """No sub-specs anywhere: `cache_config.mamba_cache_mode` may answer, labelled.

    A SIMULATED SHAPE, SAID PLAINLY: a spec class named like a state spec
    (`MambaSpec`) that declares no `mamba_cache_mode` and aggregates nothing does not
    exist on the box today - `MambaSpec` declares the field with a "none" default
    (`kv_cache_interface.py:998`). This branch is the defensive one for a fork that
    moves the field, and it is the fallback requirement 1 asks for. What IS live is
    the value it reads: the box's `config.mamba_cache_mode` is "align", and
    `abstract.py:78` fills every MambaSpec's mode from that same field - so on this
    fork the two sources agree by construction, not by luck.

    The basis must say the claim is ENGINE-WIDE and NOT per-group. An engine-wide flag
    presented as a per-group fact is exactly what a `basis` field exists to prevent.
    """
    specs = [fake.Obj("MambaSpec", block_size=3200, prefix_cacheable=True,
                      page_size_bytes=MAMBA_LAYER_PAGE),
             fake.Obj("MLAAttentionSpec", block_size=3200, prefix_cacheable=True,
                      page_size_bytes=ATTENTION_LAYER_PAGE)]
    shapes = disk_store.span_shape_state(_wrap(specs, engine_mode="align"))

    assert shapes["ok"] is True, shapes["reason"]
    assert shapes["align_group_ids"] == [0]
    assert shapes["positional_group_ids"] == [1]
    assert "engine-wide" in shapes["basis"]
    assert "not per-group" in shapes["basis"]


def test_the_engine_wide_flag_never_calls_an_unrecognized_group_align():
    """THE OVERCLAIM THIS CHANGE MUST NOT MAKE.

    Engine-wide "align", plus groups whose spec names no state class and aggregates
    nothing that does: those groups stay POSITIONAL. The engine-wide flag says "this
    engine has align-mode state groups"; it does not say which. Marking the attention
    group as align would empty `positional`, make every span unclassifiable, and
    replace a dark panel with a wrong one.
    """
    shapes = disk_store.span_shape_state(
        _wrap([attention_container(expose="none"), attention_container(expose="none")],
              engine_mode="align"))

    assert shapes["ok"] is False
    assert shapes["align_group_ids"] == []
    assert shapes["positional_group_ids"] == []
    assert shapes["prefix_cacheable_group_ids"] == [0, 1]


def test_the_engine_wide_value_and_the_class_name_must_both_answer():
    """A state-named class alone is NOT align: the mode has to be the engine's answer.

    The fallback needs BOTH halves - a class name that says WHICH group, and the
    engine-wide mode that says WHETHER align. The positive case is asserted first so
    this test also fails against the unpatched tree; the negatives are the case where
    one half is missing (`vllm_config` absent, or prefix caching off, which is exactly
    when the fork sets the mode back to "none" at `models/config.py:651-652`) and
    nothing may then be classified.
    """
    state = fake.Obj("MambaSpec", block_size=3200, prefix_cacheable=True,
                     page_size_bytes=MAMBA_LAYER_PAGE)
    attention = fake.Obj("MLAAttentionSpec", block_size=3200, prefix_cacheable=True,
                         page_size_bytes=ATTENTION_LAYER_PAGE)

    both = disk_store.span_shape_state(_wrap([state, attention],
                                             engine_mode="align"))
    assert both["ok"] is True, both["reason"]
    assert both["align_group_ids"] == [0]

    for shapes in (disk_store.span_shape_state(_wrap([state, attention],
                                                    with_vllm_config=False)),
                   disk_store.span_shape_state(_wrap([state, attention],
                                                     engine_mode="none"))):
        assert shapes["ok"] is False
        assert shapes["align_group_ids"] == []
        assert "align" in shapes["reason"]


# --- 3. neither source: today's fail-open, and nothing raises ----------------

def test_neither_source_yields_align_and_the_reason_names_what_it_saw():
    """STILL FAILS OPEN, AND NOW EXPLAINS ITSELF.

    Containers whose sub-specs are attention specs and an engine-wide mode of "none":
    no align group exists, so the counters stay zero exactly as today - but the reason
    now carries the evidence a reader needed tonight: the spec classes it saw, and that
    the engine-wide flag was consulted and answered "none".
    """
    shapes = disk_store.span_shape_state(
        _wrap([attention_container(), attention_container()], engine_mode="none"))

    assert shapes["ok"] is False
    assert shapes["align_group_ids"] == []
    assert shapes["positional_group_ids"] == []
    assert shapes["prefix_cacheable_group_ids"] == [0, 1]
    assert "align" in shapes["reason"]
    assert CONTAINER_CLASS in shapes["reason"]
    assert "MLAAttentionSpec" in shapes["reason"]
    assert "engine-wide" in shapes["reason"]
    assert "none" in shapes["reason"]


def test_a_spec_that_raises_is_a_reason_and_a_base_exception_still_propagates():
    """THE CONTAINMENT BOUNDARY, one assertion on each side of it.

    `span_shape_state()` runs on the worker's step thread inside the `get_finished`
    wrapper. An `Exception` from an attribute walk is a diagnostic failure and must
    become a reason that NAMES it (a group whose evidence cannot be read is unknown,
    never silently positional); a `KeyboardInterrupt` is the engine speaking and must
    reach it - the same line `tests/test_containment.py` draws across the package.
    """
    class Hostile:
        def __init__(self, error) -> None:
            self.block_size = 3200
            self.prefix_cacheable = True
            self._error = error

        @property
        def mamba_cache_mode(self):
            raise self._error

    broken = disk_store.span_shape_state(_wrap([Hostile(RuntimeError("boom"))],
                                               with_vllm_config=False))
    assert broken["ok"] is False
    assert "RuntimeError" in broken["reason"]

    try:
        disk_store.span_shape_state(_wrap([Hostile(KeyboardInterrupt())],
                                          with_vllm_config=False))
    except KeyboardInterrupt:
        return
    raise AssertionError("the resolution swallowed a BaseException")


def test_sub_specs_that_disagree_are_refused_by_name_not_guessed():
    """A container whose layers disagree about the mode classifies NOTHING.

    On this fork every layer of a group takes its mode from the same engine-wide field
    (`abstract.py:78`), so a disagreement means the geometry is not what either side
    thinks it is. Guessing a denominator here would move the one ratio this exists to
    report in whichever direction the guess happened to point - the rule
    `span_shape_state`'s own docstring states, and this test keeps.
    """
    spec = container([sub_spec("MambaSpec", mode="align"),
                      sub_spec("MambaSpec", mode="none")])
    shapes = disk_store.span_shape_state(_wrap([spec, spec]))

    assert shapes["ok"] is False
    assert "disagree" in shapes["reason"]
    assert "align" in shapes["reason"] and "'none'" in shapes["reason"]


# --- 4. reason, basis and ignored pairs reach the rank file and the rollup ----

def test_the_rank_file_carries_the_reason_the_basis_and_the_ignored_pairs(tier):
    """`worker_report()` publishes what the engine-side view structurally cannot.

    `span_shape_view()` is built by `hook_section()`, which runs in the ENGINE process,
    where the worker's shape cache is empty - so the reason that explains a dark panel
    lived only in a payload that can never contain the worker's reason. The rank file is
    the door that can. Unpatched, the three keys below raise KeyError.
    """
    role = container_handler()
    with metadata_shape(True):
        spill_a_handoff(role, [(0, key_for(SPAN_A, ATTENTION_GROUP))])
    report = disk_store.worker_report()

    assert report["span_shape_pairs_ignored"] == 0
    assert isinstance(report["span_shape_basis"], str)
    assert "MambaSpec" in report["span_shape_basis"]
    assert len(report["span_shape_basis"]) <= 140, report["span_shape_basis"]
    assert report["span_shape_reason"] is None

    dark = handler()             # a table with no align evidence at all
    with metadata_shape(True):
        spill_a_handoff(dark, [(1, key_for(SPAN_A, 0))], event=9)
    reason = disk_store.worker_report()["span_shape_reason"]
    assert isinstance(reason, str) and "align" in reason


def test_the_pairs_this_module_could_not_read_are_counted_in_the_rank_file(tier):
    """`span_shape_pairs_ignored` was in `_state` and in `span_shape_view()` and in NO
    rank file - so a fleet-wide "the classifier is being fed garbage" was invisible
    unless you happened to be polling the one process that saw it.
    """
    role = container_handler()
    with metadata_shape(True):
        spill_a_handoff(role, [(0, SPAN_A),                  # no group suffix on it
                               (1, 12345),                   # not a key at all
                               (2, key_for(SPAN_A, 9)),      # a group with no file
                               (3, key_for(SPAN_A, ATTENTION_GROUP))])
    report = disk_store.worker_report()

    assert report["span_shape_pairs_ignored"] >= 1
    assert report["spill_spans_classified"] == 1
    assert report["spill_incomplete_state"] == 1


def test_the_rollup_rolls_up_the_reason_the_basis_and_the_ignored_pairs(
        tier, tmp_path, monkeypatch):
    """THE WHITELIST IS THE FAILURE MODE, and this test is aimed at it.

    A rank-file field that `_rank_file_rollup_row()` does not name is silently never
    rolled up: no error, no gap, no `fields_rejected` entry - just a total that is not
    there. That is exactly how the completeness panel stayed invisible for a day, so
    the round trip goes through the WRITER'S OWN output and back, the way
    `test_disk_spill_rank_rollup.py::test_the_writer_and_the_reader_agree_on_the_field_
    names` does for the counters that survived.
    """
    role = container_handler()
    with metadata_shape(True):
        spill_a_handoff(role, [(0, key_for(SPAN_A, 9)),        # a group with no file
                               (1, key_for(SPAN_A, ATTENTION_GROUP))])
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path))
    write_rank(tmp_path, "PP0_TP0", disk_store.worker_report())
    write_rank(tmp_path, "PP0_TP1", disk_section(span_shape_pairs_ignored=4))

    rollup = disk_store.rank_rollup(force=True)

    assert rollup["totals"]["span_shape_pairs_ignored"] == 5, rollup["totals"]
    rows = {row["rank"]: row for row in rollup["per_rank"]}
    assert "sub-spec" in rows["PP0_TP0"]["span_shape_basis"]
    assert rows["PP0_TP0"]["span_shape_reason"] is None
    assert rows["PP0_TP1"]["span_shape_pairs_ignored"] == 4
    assert rollup["units"]["span_shape_pairs_ignored"]
    assert rollup["meanings"]["span_shape_pairs_ignored"]


def test_the_rollup_distinguishes_a_missing_reason_from_an_empty_one(tmp_path,
                                                                     monkeypatch):
    """A rank whose panel is HEALTHY has no reason, and that is not a dropped field.

    `None` means "this boot resolved its geometry"; a string means "this is why it did
    not". Coercing one into the other would make a healthy rank look broken and a
    broken rank look healthy - the two readings this whole change is about.
    """
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path))
    write_rank(tmp_path, "PP0_TP0", disk_section(span_shape_reason=None,
                                                 span_shape_basis="spec>sub-specs"))
    write_rank(tmp_path, "PP0_TP1", disk_section(
        span_shape_reason="no prefix-cacheable group resolved 'align'"))

    rows = {row["rank"]: row
            for row in disk_store.rank_rollup(force=True)["per_rank"]}
    assert rows["PP0_TP0"]["span_shape_reason"] is None
    assert "align" in rows["PP0_TP1"]["span_shape_reason"]


def test_the_disk_section_still_fits_the_contribution_cap(tier):
    """THE NEW BYTES ARE BOUNDED, by a clip rather than by hope.

    `worker_diag` REFUSES a section over `MAX_CONTRIBUTION_BYTES` rather than
    truncating it, so a rank file that grew too verbose would lose its WHOLE `disk`
    section - every counter in it, including the ones that survived tonight. A long
    evidence string is therefore only allowed if it is clipped, and the clip is what
    this asserts, on the geometry that produces the longest reason (six groups, long
    class names, nothing resolved).
    """
    specs = [container([sub_spec(f"SomeVeryLongAttentionSpecNameForGroup{index}",
                                 page=ATTENTION_LAYER_PAGE)
                        for _ in range(12)], block_size=3200)
             for index in range(6)]
    role = _wrap(specs, engine_mode="none")
    with metadata_shape(True):
        spill_a_handoff(role, [(0, key_for(SPAN_A, ATTENTION_GROUP))])
    report = disk_store.worker_report()

    assert isinstance(report["span_shape_reason"], str)
    assert len(report["span_shape_reason"]) <= 400, report["span_shape_reason"]
    # THE CLIP MUST NOT COST THE POINT. The reason's value is the evidence at its end -
    # the class names and the engine-wide verdict - so a ceiling that truncated before
    # them would leave tonight's question unanswered in a slightly longer sentence.
    assert "SomeVeryLongAttentionSpecNameForGroup" in report["span_shape_reason"]
    assert "engine-wide" in report["span_shape_reason"]
    assert len(report["span_shape_basis"] or "") <= 140
    serialized = len(json.dumps(report))
    assert serialized < worker_diag.MAX_CONTRIBUTION_BYTES, serialized


# --- 5. the restore-verify side cannot drift ----------------------------------

def verify_worker(specs, *, engine_mode="align", with_vllm_config=True):
    """A worker with only what `_boundary_groups` reads: the group table, vllm_config.

    It is deliberately NOT `test_disk_restore_verify.VerifierWorker`: that fake exists
    to drive byte comparisons and its `CacheConfig` cannot express a container. The two
    attributes here are the whole surface the function touches, so this tests the
    predicate and nothing else.
    """
    return types.SimpleNamespace(
        kv_cache_config=fake.Obj(kv_cache_groups=[fake.Obj(kv_cache_spec=spec)
                                                  for spec in specs]),
        vllm_config=(fake.Obj(cache_config=fake.Obj(
            mamba_cache_mode=engine_mode)) if with_vllm_config else None))


def test_the_restore_verify_side_finds_the_boundary_groups_it_could_not_find():
    """THE SECOND HALF OF THE SAME BUG: `boundary_groups_known` was 0 on all four ranks.

    With no boundary group known, the verifier can never count
    `boundary_entry_missing` for a batch that restored the positional groups and
    dropped the state block - the truncation shape it was written to catch, inert
    forever. Unpatched this returns
    `([], "no group's spec reports 'align' mamba_cache_mode")`.
    """
    known, note = disk_restore_verify._boundary_groups(
        verify_worker([container(live_sub_specs(index), block_size=LIVE_SIZES[index])
                       for index in range(6)]))

    assert known == ALIGN_IDS
    assert note is None


def test_a_boundary_group_that_cannot_be_found_still_says_why_in_the_old_words():
    """The note keeps the sentence another test already greps for, and gains evidence.

    `test_disk_restore_verify.py::test_a_boot_with_no_align_group_counts_nothing`
    asserts the "no group's spec reports" prefix; a reader who knows how to grep the
    note must still find it, and must now also learn WHICH classes the worker has.
    """
    known, note = disk_restore_verify._boundary_groups(
        verify_worker([attention_container()], engine_mode="none"))

    assert known == []
    assert "no group's spec reports" in note
    assert CONTAINER_CLASS in note


def test_the_boundary_note_cannot_grow_past_its_own_rank_file_ceiling():
    """The evidence is worth bytes; the section being refused is not worth any.

    `disk_restore_verify` keeps itself under `MAX_SECTION_BYTES` (3920 of the seam's
    4096) because `worker_diag` REFUSES an over-cap section rather than trimming it -
    so a note that quoted thirty groups and forty class names would cost this module
    every counter it publishes, which is the dark panel again in a different file. The
    geometry below is the one that produces the longest note the walk can make.
    """
    worker = verify_worker(
        [fake.Obj(f"SomeVeryLongSpecClassNameForGroupNumber{index}",
                  block_size=3200, prefix_cacheable=True, page_size_bytes=1)
         for index in range(6)], engine_mode="none")
    known, note = disk_restore_verify._boundary_groups(worker)

    assert known == []
    assert len(note) <= disk_restore_verify.BOUNDARY_NOTE_CHARS, (len(note), note)
    assert note.startswith("no group's spec reports")
    # Compact, not merely short: the note names the spec class it saw, which is the one
    # fact that separates "this boot has no state group" from "this build could not
    # read the group's shape".
    assert "SomeVeryLongSpecClassNameForGroupNumber0" in note


def test_the_two_sides_cannot_disagree_about_which_groups_are_align():
    """ONE SHARED HELPER, PROVEN AS A FACT RATHER THAN A PROMISE.

    Two readings of one predicate is how this happened: `disk_store` had
    `span_shape_state()` and `disk_restore_verify` had `_boundary_groups()`, written
    against the same fork field, and both went dark together for the same reason. The
    behavioural half is equality over three geometries; the structural half is that
    the verifier must DELEGATE - it may name the mode only in prose, so a third
    predicate cannot appear beside them unnoticed.
    """
    cases = [
        ([container(live_sub_specs(index), block_size=LIVE_SIZES[index])
          for index in range(6)], "align"),
        ([fake.Obj("MambaSpec", block_size=3200, prefix_cacheable=True,
                   page_size_bytes=MAMBA_LAYER_PAGE),
          fake.Obj("MLAAttentionSpec", block_size=3200, prefix_cacheable=True,
                   page_size_bytes=ATTENTION_LAYER_PAGE)], "align"),
        ([attention_container()], "none"),
    ]
    for specs, mode in cases:
        shapes = disk_store.span_shape_state(_wrap(list(specs), engine_mode=mode))
        known, _note = disk_restore_verify._boundary_groups(
            verify_worker(list(specs), engine_mode=mode))
        assert known == shapes["align_group_ids"], (mode, shapes, known)

    source = inspect.getsource(disk_restore_verify._boundary_groups)
    assert "resolve_mamba_cache_mode" in source, "the verifier reads the mode itself"
    assert '"mamba_cache_mode"' not in source, "a second predicate was written"


# --- 6. observability only: no arm, no posture, no counter --------------------

def test_resolving_a_mode_changes_no_arm_no_posture_and_no_counter(tier):
    """The resolution is a READ. Nothing it observes may move a switch, an arm or one
    of its own counters: `span_shape_state()` and the helper are attribute walks, and
    the counters move only in `_note_span_shapes`.
    """
    role = container_handler()
    watched = ("spill_spans_classified", "spill_spans_complete",
               "span_shape_pairs_ignored", "anchor_spills_completed",
               "refusals_total", "errors", "fallbacks", "installed")
    before = {name: disk_store.counters()[name] for name in watched}
    before["spill_armed"] = disk_spill.is_armed()
    disk_store.span_shape_state(role)
    disk_store.resolve_mamba_cache_mode(container(live_sub_specs(0)),
                                        role.vllm_config)
    disk_restore_verify._boundary_groups(verify_worker(
        [container(live_sub_specs(index), block_size=LIVE_SIZES[index])
         for index in range(6)]))
    after = {name: disk_store.counters()[name] for name in watched}
    after["spill_armed"] = disk_spill.is_armed()

    assert after == before, (before, after)


def _executable_names(function) -> tuple:
    """(names, attributes) in a function's CODE, docstring and comments stripped.

    The `_executable_source` idiom of `tests/test_mamba_anchor_patch.py:819`, applied
    to one function: prose that names a switch in order to disclaim touching it must
    not fail the test that forbids touching it.
    """
    source = textwrap.dedent(inspect.getsource(function))
    body = [line for line in source.splitlines() if not line.strip().startswith("#")]
    tree = ast.parse("\n".join(body))
    doc = ast.get_docstring(tree, clean=False)
    if doc:
        statement = tree.body[0]
        drop = set(range(statement.lineno, statement.end_lineno + 1))
        body = [line for number, line in enumerate(body, start=1)
                if number not in drop]
    tree = ast.parse("\n".join(body))
    return ({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)},
            {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)},
            "\n".join(body))


def test_the_helper_reads_attributes_and_touches_nothing_else():
    """STRUCTURE: the resolution is an attribute walk, and the classification is too.

    No I/O, no engine object, no environment, no arm, no posture, no switch - in the
    CODE, not in the prose. These two functions run on the worker's step thread inside
    the fork's `get_finished`; anything but a bounded read there is a serving change.
    """
    forbidden = {"set_disk_authority", "set_offer_switch", "set_armed",
                 "disk_authority", "offers_enabled", "serving_as", "serve_as",
                 "open", "socket", "environ", "popen", "spawn", "system",
                 "suspend_server", "wake_and_start", "respawn_backend"}
    for function in (disk_store.resolve_mamba_cache_mode,
                     disk_store.span_shape_state):
        names, attrs, code = _executable_names(function)
        assert not (names & forbidden), (function.__name__, sorted(names & forbidden))
        assert not (attrs & forbidden), (function.__name__,
                                         sorted(attrs & forbidden))
        for word in ("os.", "open(", "socket", "environ", "subprocess"):
            assert word not in code, (function.__name__, word)
