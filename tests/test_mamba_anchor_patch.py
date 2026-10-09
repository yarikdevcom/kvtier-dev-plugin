r"""THE SECOND RUNTIME-PATCH TARGET: `MambaManager.reachable_block_mask`, one mark.

THE CLAIM THIS FILE EXISTS TO PROVE, and the trap it has to avoid. The patch is three
lines - `if mask and not mask[-1]: mask[-1] = True` - so the temptation is a test that
builds a mask which already ends True and asserts it is still True. That is the
anti-pattern this repo names explicitly (`test_in_tree_standdown.py:46-49`,
`test_disk_restore.py:3294`): a fake shaped so the assertion cannot fail. Every test
here is built the other way round, and the two that carry the behavioural claim open by
PROVING the unpatched tree does not already do the job:

  * `test_anchor_mask_marks_run_end` first asserts the FORK'S OWN TEXT returns a mask
    whose last element is False for a range whose last block is neither a segment end
    nor a reachable boundary - then installs, and only then asserts it is True. The
    same geometry, from the fork's captured source, not from a hand-shaped list.
  * `test_anchor_mask_fingerprint_refusal` mutates the fixture's source text and asserts
    the install refuses naming BOTH hashes, with its unmuted control (the unmutated
    text) installing in the same file - so "refused" cannot be an artifact of an install
    path that never works.

THE FIXTURES ARE THE FORK'S TEXT. `fixtures_mamba.py` is generated from
`/home/user/Source/experiment/vllm-live-repro@live` (the live-tree reconstruction: up-
stream v0.30.0 + the 58-patch series + our four kvtier hunks) by
`tools/gen_mamba_fixture.py`, and
`test_the_pinned_fingerprints_are_the_ones_the_fixture_carries` ties the plugin's
pin, the fixture's text and the file's md5 together, so a fingerprint
that drifted from the fork cannot be shipped silently. The class the tests patch is
built from that text and `install()` is the REAL one - so a green install test means the
live box's `MambaManager` will accept the patch, and a fork update means it will not.

WHAT THE PATCH MAY NEVER DO is pinned twice: once behaviourally (arm and posture
identical before and after an install and a mark) and once structurally, against this
module's own source - `test_nothing_about_serving_posture_changed`, the sibling of the
AST-pinned `test_disk_authority_switch.py:95/:191/:251`.

THE FOUR COUNTERS each get a door test, because a counter with no door is a variable:
`mamba_patch_installed` (backport section + BOOT line), `anchor_offers_total`
(`snapshot.dev_state_sections()`'s disk_store section), `anchor_spills_completed`
(`disk_store.rank_rollup()`'s totals, via the rank-file whitelist) and
`restore_anchor_hits_total` (`disk_restore.counters()` + the rank file's section).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re

import fixtures_mamba as FX
import pytest
import test_disk_restore as base
import test_disk_store as store_fixtures
import test_spill_span_completeness as spanfx

from kvtier_dev_plugin import (
    backport,
    config,
    disk_restore,
    disk_spill,
    disk_store,
    mamba_anchor,
    pr56810_sources,
    snapshot,
)

# THE FIXTURES, RE-EXPORTED BY NAME: pytest collects fixtures from imported names, and
# `tier` is the configured, ARMED spill store the spill-side tests drive, while `armed`
# is the restore suite's open gate. Re-exported rather than re-implemented, exactly as
# `test_spill_span_completeness.py` re-exports them for the same reason - a local copy
# that quietly fails to arm is a counter test that passes for the wrong reason.
tier = store_fixtures.tier  # noqa: F811 - the configured, ARMED store under tmp_path
none_hash_seed = store_fixtures.none_hash_seed  # noqa: F811 - the fingerprint's seed
no_fake_distributed_state = store_fixtures.no_fake_distributed_state  # noqa: F811
armed = base.armed  # noqa: F811 - the restore gate, open, with the fork helpers
fresh = base.fresh  # noqa: F811 - autouse: a fresh disk_restore boot per test.
# NOT optional here: `disk_restore._state` is deliberately absent from conftest's reset
# list (the restore suite owns it), and this file reads and moves
# `restore_anchor_hits_total`. Without `fresh` the restore-side tests inherit each
# other's counters, and the control test below fails on its predecessor's number -
# which is exactly the leak `test_disk_restore.py`'s comment "or counters leak" names.

# THE GEOMETRY THE MASK IS JUDGED AGAINST, in the live box's own numbers. `block_size`
# 1600 for the align-mode group, the scheduler/hash alignment 3200, and a retention
# interval of 6400 - which is TWO blocks per segment, so the segment-boundary marks land
# on blocks 1, 3, 5 ... A range of blocks [0, 5) therefore ends on block 4, one PAST the
# last segment boundary, and no reachable boundary is named: the fork's own text returns
# `[False, False, False, True, False]`, and `mask[-1]` is False. That is the shape the
# patch exists for, and the first assertion of the behaviour test is that this geometry
# really is it.
BLOCK = 1600
ALIGNMENT = 3200
RETENTION = 6400
SEGMENT_BLOCKS = RETENTION // BLOCK          # 2


class MambaSpec:
    """The `MambaSpec` field the mask reads: its block size. Nothing else.

    A stand-in for a dataclass this suite cannot import (no vllm), carrying the ONE
    attribute `reachable_block_mask` touches. `isinstance` is satisfied by the exec
    namespace's binding, not by a real fork class - the method's own text is real.
    """

    def __init__(self, block_size: int = BLOCK):
        self.block_size = block_size


def fork_mamba():
    """The fork-text `MambaManager`, freshly built. (cls, namespace)."""
    return FX.build_fork_mamba_class({"MambaSpec": MambaSpec,
                                      "KVCacheSpec": object,
                                      "Sequence": list})


def mask_kwargs(**overrides) -> dict:
    """One call's arguments, the live geometry as the default."""
    kwargs = {"start_block": 0, "end_block": 5, "alignment_tokens": ALIGNMENT,
              "kv_cache_spec": MambaSpec(BLOCK), "use_eagle": False,
              "retention_interval": RETENTION, "reachable_boundaries": ()}
    kwargs.update(overrides)
    return kwargs


def unpatched_mask(**overrides) -> list | None:
    """THE FORK'S VERDICT, from a class that has never been touched."""
    cls, _ns = fork_mamba()
    return cls.reachable_block_mask(**mask_kwargs(**overrides))


# --- the fingerprints, and the extraction that produced them ------------------

def test_the_extraction_method_reproduces_the_inspect_fingerprints():
    """The generator reads the fork's FILE; the plugin hashes `inspect.getsource()`.

    Those are two different extractions, and a fixture whose text differs from what
    `inspect.getsource()` yields on the box would make every production install refuse -
    a fixture that passes its own suite and fails in the field. The equivalence is not
    asserted, it is MEASURED: `fixtures_pr56810.py`'s four bodies were captured by
    `tools/gen_pr56810_fixture.py` under the vllm venv with real `inspect.getsource()`,
    and `pr56810_sources.py` ships the hashes of exactly that text. Re-running this
    file's extraction (AST line span, decorators included) over that text must
    reproduce every shipped hash - which is what makes the same extraction trustworthy
    for the Mamba method, captured from a tree rather than from a live interpreter.
    """
    def extract(src: str, class_name: str, method: str) -> str:
        tree = ast.parse(src)
        lines = src.splitlines(keepends=True)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for fn in node.body:
                    if (isinstance(fn, ast.FunctionDef)
                            and fn.name == method):
                        start = fn.lineno - 1
                        if fn.decorator_list:
                            start = min(d.lineno
                                        for d in fn.decorator_list) - 1
                        return "".join(lines[start:fn.end_lineno])
        raise AssertionError(f"{class_name}.{method} not found")

    import fixtures_pr56810 as fx56810
    for name, entry in pr56810_sources.SOURCES.items():
        text = fx56810.ORIGINAL_SOURCES[name]
        rejoined = extract("class SimpleCPUOffloadScheduler:\n" + text,
                           "SimpleCPUOffloadScheduler", name)
        assert hashlib.sha256(rejoined.encode()).hexdigest() \
            == entry["orig_sha256"], name
        assert rejoined == text, name


def test_the_pinned_fingerprints_are_the_ones_the_fixture_carries():
    """Pin == fixture text == the fork's own text, and the source is NAMED.

    The plugin's `ORIG_SHA256` must hash-match the generated fixture (or an install test
    would be driving a body the box does not have), and the fixture must carry the file
    md5 the three trees agree on - the identity that makes the pin a fact about the
    engine rather than about one checkout. `FINGERPRINT_SOURCE` naming the repro branch
    is the third half: after the next fork update, "where did this pin come from" has to
    be answerable from the payload.
    """
    body = FX.ORIGINAL_SOURCES[mamba_anchor.METHOD]
    assert hashlib.sha256(body.encode()).hexdigest() == FX.METHOD_SHA256
    assert mamba_anchor.ORIG_SHA256[mamba_anchor.METHOD] == FX.METHOD_SHA256
    assert FX.FORK_FILE_MD5 == "52a3c5ad0fc4e6911c69bb554b2d914a"
    assert FX.FORK_REF == "live"
    assert "vllm-live-repro@live" in mamba_anchor.FINGERPRINT_SOURCE
    # The target is the pure classmethod the task names - not a method of the offload
    # manager, whose live and dev copies DIFFER (live 91a3aa65 vs dev 26097a02) because
    # our own hunks live there, so no pin may be copied from that file's dev copy.
    assert mamba_anchor.TARGET_MODULE == \
        "vllm.v1.core.single_type_kv_cache_manager"
    assert mamba_anchor.TARGET_CLASS == "MambaManager"
    assert mamba_anchor.METHODS == (mamba_anchor.METHOD,)
    assert FX.METHOD_DECORATORS == ("classmethod",)
    # ...and the fork's text really is unpatched: no kvtier marker in it.
    assert "kvtier" not in body


# --- 1. the mask: the behaviour claim, with its anti-vacuity proof ------------

def test_anchor_mask_marks_run_end():
    """THE claim: the run's end block stops being False. In three steps, in order.

    1. THE UNPATCHED PROOF, on the fork's own captured text: `mask[-1] is False`, and
       not because a hand-built list said so. Block 4 is neither a segment boundary
       (those are 1 and 3 at this geometry) nor a named reachable boundary, so the
       fork's retention does not mark it - which is the `no_boundary_anchor` shape seen
       from the mask side.
    2. THE REAL INSTALL on that class (fingerprints match, so it is accepted).
    3. THE FLIP, and ONLY the flip: every earlier entry is byte-identical to the
       unpatched verdict, and a dense `None` mask is untouched.

    Without step 1 the rest is the vacuity this file's docstring names: a mask that
    already ended True would make step 3 true of the unpatched fork too.
    """
    before = unpatched_mask()
    assert before == [False, False, False, True, False], (
        "the fork's own text no longer leaves a cached range's LAST block unmarked - "
        "either the fork fixed this itself (then the patch is obsolete and must stand "
        "down) or this fixture drifted from it")
    assert before[-1] is False

    cls, _ns = fork_mamba()
    report = mamba_anchor.install(cls)
    assert report["installed"] is True, report
    assert report["verdict"] == mamba_anchor.VERDICT_PATCHED

    after = cls.reachable_block_mask(**mask_kwargs())
    assert after[-1] is True, "the patch did not mark the run's end block"
    assert after[:-1] == before[:-1], "the patch changed a block it must not touch"
    assert after == [False, False, False, True, True]

    # THE PATCH'S OWN CONTRIBUTION, counted once, for this one mark.
    assert mamba_anchor.counters()["anchor_offers_total"] == 1
    assert mamba_anchor.counters()["mamba_patch_installed"] is True

    # A DENSE VERDICT IS NOT A MASK. `None` is the fork saying "cache every block"
    # (dense retention, or no alignment constraint) and writing a last element into it
    # is not even representable - assert it comes back as None, not as [True].
    assert cls.reachable_block_mask(**mask_kwargs(retention_interval=None)) is None
    assert cls.reachable_block_mask(**mask_kwargs(alignment_tokens=None)) is None

    # AN EMPTY MASK IS NOT A RUN END. `end_block == start_block` gives the fork's
    # `[]`; extending it to one element would address a block the caller never asked a
    # mask for, so the patch must leave the length alone.
    assert cls.reachable_block_mask(**mask_kwargs(end_block=0)) == []


def test_anchor_mask_does_not_count_a_boundary_the_fork_already_marked():
    """The vacuity control for the counter, not just for the mask.

    A range that ends ON a segment boundary is already anchored by the fork's own
    retention: `mask[-1]` is True before any patch touched this class. Counting those
    would make `anchor_offers_total` measure the fork's retention interval instead of
    the patch's contribution - which is the difference between a counter and a number.
    """
    already = unpatched_mask(end_block=4)
    assert already[-1] is True, "this geometry no longer ends on a segment boundary"
    assert mamba_anchor.counters()["anchor_offers_total"] == 0, (
        "a probe of the UNPATCHED class must not move a counter")

    cls, _ns = fork_mamba()
    mamba_anchor.install(cls)
    cls.reachable_block_mask(**mask_kwargs(end_block=4))

    counters = mamba_anchor.counters()
    assert counters["anchor_offers_total"] == 0
    assert counters["mask_already_anchored"] == 1


def test_anchor_mask_is_idempotent_and_never_nests():
    """Three installs, ONE wrapper: a reload loop cannot nest masks or double-count.

    The mark count is the observable: if an install wrapped its own wrapper, one call
    through the mask would add one mark PER LAYER and `anchor_offers_total` would read
    3 for a single run end. Re-installing re-binds the fork's ORIGINAL instead, so the
    count stays one per call however many times `install()` ran.
    """
    cls, _ns = fork_mamba()
    mamba_anchor.install(cls)
    mamba_anchor.install(cls)
    mamba_anchor.install(cls)
    first = cls.reachable_block_mask(**mask_kwargs())
    counters = mamba_anchor.counters()
    assert counters["anchor_offers_total"] == 1, (
        f"one masked call after three installs counted "
        f"{counters['anchor_offers_total']} marks: the wrappers are nested")
    assert counters["reinstalls"] == 2
    assert counters["installs"] == 1
    # The wrapper's OWN marker is on the class entry, and only once.
    raw = cls.__dict__[mamba_anchor.METHOD]
    assert getattr(raw.__func__, mamba_anchor.OWNER_ATTR, None) \
        == mamba_anchor.OWNER_TAG
    second = cls.reachable_block_mask(**mask_kwargs())
    assert first == second
    assert mamba_anchor.counters()["anchor_offers_total"] == 2


# --- 2. the fingerprint: a stale body is a refusal, never a guess -------------

def _drifted_mamba():
    """The fork's text with ONE character changed inside the body.

    A reachable-boundary line gains a no-op, so the SHA changes while the method still
    runs - the exact shape of a harmless-looking fork edit, which is precisely what a
    fingerprint exists to catch.
    """
    drifted = FX.ORIGINAL_SOURCES[mamba_anchor.METHOD].replace(
        "        mask = [False] * (end_block - start_block)",
        "        mask = [False] * (end_block - start_block)  # fork edit")
    assert drifted != FX.ORIGINAL_SOURCES[mamba_anchor.METHOD]
    text = ("class MambaManager(SingleTypeKVCacheManagerStub):\n" + drifted)
    import linecache
    name = "<fixtures_mamba_drifted>"
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    module = {"SingleTypeKVCacheManagerStub": object,
              "MambaSpec": MambaSpec, "KVCacheSpec": object, "Sequence": list,
              "__name__": "fixtures_mamba_drifted", "__file__": name}
    exec(compile(text, name, "exec"), module, module)  # noqa: S102 - our own text
    cls = module["MambaManager"]
    cls.__module__ = "fixtures_mamba_drifted"
    return cls, hashlib.sha256(drifted.encode()).hexdigest()


def test_anchor_mask_fingerprint_refusal():
    """A moved body is REFUSED, and the refusal NAMES BOTH SHAS.

    "Stale fingerprint" has to be an operator-readable fact, not a silent `installed:
    False`: the reason names the live hash, the authoring-time hash and where the pin
    came from, so the next question ("what changed?") starts from an answer. And
    NOTHING is written to the class: a refusal that patched anyway would be the
    half-patch the docstring promises cannot happen.

    THE UNMUTED CONTROL is the second half of this test: the same install against the
    unmutated text installs. Without it, "refused" could be an artifact of an install
    path that never works for any class at all.
    """
    cls, live_sha = _drifted_mamba()
    before = cls.__dict__[mamba_anchor.METHOD]

    report = mamba_anchor.install(cls)
    assert report["installed"] is False
    assert report["verdict"] == mamba_anchor.VERDICT_REFUSED
    expected = mamba_anchor.ORIG_SHA256[mamba_anchor.METHOD]
    assert live_sha in report["reason"], report["reason"]
    assert expected in report["reason"], report["reason"]
    assert "vllm-live-repro@live" in report["reason"]
    assert mamba_anchor.counters()["verdict"] == mamba_anchor.VERDICT_REFUSED
    assert mamba_anchor.counters()["mamba_patch_installed"] is False
    assert mamba_anchor.counters()["anchor_offers_total"] == 0
    # THE CLASS IS UNTOUCHED: the very object in the class dict is the same one.
    assert cls.__dict__[mamba_anchor.METHOD] is before
    # ...and its mask still behaves like the fork's, because nothing was fitted.
    assert cls.reachable_block_mask(**mask_kwargs())[-1] is False
    # THE BOOT TOKEN SAYS SO, in the one word the operator greps.
    assert mamba_anchor.boot_token() == "MAMBA_ANCHOR[refused]"

    # THE CONTROL: same code path, unmutated text -> installed.
    control, _ns = fork_mamba()
    assert mamba_anchor.install(control)["installed"] is True
    assert control.reachable_block_mask(**mask_kwargs())[-1] is True
    assert mamba_anchor.boot_token() == "MAMBA_ANCHOR[patched]"


def test_anchor_mask_refuses_a_moved_or_reshaped_method():
    """Two more refusals that must not become a guess: moved, and no longer a
    classmethod.

    Both are the "the fork changed the calling convention" case that
    `install_connector_hook`
    refuses for the same reason: rebinding an inherited method onto this class would
    change the behaviour of every sibling that shares the base body, and a plain
    function bound where the fork calls a classmethod is a broken boot.
    """
    # (a) NOT ON THIS CLASS: an inherited body is a refusal, not a hint.
    class Base:
        @classmethod
        def reachable_block_mask(cls, **kwargs):
            return None

    class Inherited(Base):
        pass

    report = mamba_anchor.install(Inherited)
    assert report["installed"] is False
    assert "not defined on this class" in report["reason"]

    # (b) NOT A CLASSMETHOD ANY MORE.
    cls, _ns = fork_mamba()
    cls.reachable_block_mask = staticmethod(FX.ORIGINAL_SOURCES and
                                            _plain_body())
    report = mamba_anchor.install(cls)
    assert report["installed"] is False
    assert "not a classmethod" in report["reason"]


def _plain_body():
    """The fork's method as a plain function, for the "convention changed" refusal."""
    text = FX.ORIGINAL_SOURCES[mamba_anchor.METHOD].replace(
        "    @classmethod\n", "", 1)
    name = "<fixtures_mamba_plain>"
    import linecache
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    module = {"MambaSpec": MambaSpec, "KVCacheSpec": object, "Sequence": list,
              "__name__": "fixtures_mamba_plain", "__file__": name}
    exec(compile("class C:\n" + text, name, "exec"), module, module)  # noqa: S102
    return module["C"].__dict__["reachable_block_mask"]


def test_anchor_mask_refuses_a_foreign_owner_marker():
    """Another layer's marker on the body is REFUSED, never stacked on.

    `__kvtier_dev_*_owner__` is this package's convention; a marker that is not ours
    means a layer we do not know is fitted on the fork's cache-manager class, and
    hashing someone else's wrapper against our own pin is the misattribution
    `backport.py:521-540` documents. The marker's own body is left exactly as found.
    """
    cls, _ns = fork_mamba()
    raw = cls.__dict__[mamba_anchor.METHOD]
    # A plain attribute write, not setattr() with a constant name (B010): the point
    # is the attribute the production reader would find, not the mechanism.
    raw.__func__.__kvtier_dev_somebody_elses_owner__ = "them/1"

    report = mamba_anchor.install(cls)
    assert report["installed"] is False
    assert "another layer" in report["reason"]
    assert cls.__dict__[mamba_anchor.METHOD] is raw


# --- 3. the env gate: default on, `0` disables with a named reason ------------

def test_the_env_var_defaults_to_enabled(monkeypatch):
    """Unset is ENABLED, and the payload says the value came from the default."""
    monkeypatch.delenv(config.ENV_MAMBA_ANCHOR_PATCH, raising=False)
    verdict = config.mamba_anchor_patch()
    assert verdict["enabled"] is True
    assert verdict["source"] == "default"
    assert verdict["recognised"] is True
    cls, _ns = fork_mamba()
    assert mamba_anchor.install(cls)["installed"] is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "OFF", " 0 "])
def test_the_env_var_zero_disables_with_reason_env_refused(monkeypatch, value):
    """`0` is the rollback: nothing is fitted, and the reason is `env-refused`.

    Note what a disable is NOT: it is not a half-patch. `install()` returns before it
    resolves the class at all, so there is no state in which the wrapper is partly
    fitted, and the boot token distinguishes an operator's no from a fork's refusal.
    """
    monkeypatch.setenv(config.ENV_MAMBA_ANCHOR_PATCH, value)
    cls, _ns = fork_mamba()
    before = cls.__dict__[mamba_anchor.METHOD]

    report = mamba_anchor.install(cls)
    assert report["installed"] is False
    assert report["verdict"] == mamba_anchor.VERDICT_ENV_REFUSED
    assert "env-refused" in report["reason"]
    assert config.ENV_MAMBA_ANCHOR_PATCH in report["reason"]
    assert cls.__dict__[mamba_anchor.METHOD] is before
    assert mamba_anchor.boot_token() == "MAMBA_ANCHOR[env-refused]"
    assert mamba_anchor.is_installed() is False
    assert mamba_anchor.counters()["mamba_patch_installed"] is False
    # AND THE MASK STILL ANSWERS LIKE THE FORK, on a disabled boot.
    assert cls.reachable_block_mask(**mask_kwargs())[-1] is False


def test_an_unrecognised_value_stays_enabled_and_says_so(monkeypatch):
    """`flase` does not disable: the default direction is ON, and it is reported."""
    monkeypatch.setenv(config.ENV_MAMBA_ANCHOR_PATCH, "flase")
    verdict = config.mamba_anchor_patch()
    assert verdict["enabled"] is True
    assert verdict["recognised"] is False
    assert verdict["raw"] == "flase"
    cls, _ns = fork_mamba()
    assert mamba_anchor.install(cls)["installed"] is True


def test_a_process_without_the_class_reports_disabled_not_refused():
    """A worker or arg-parsing process has no `MambaManager`: that is `disabled`.

    Not a refusal, because nothing was wrong - the mask runs in the scheduler and
    nowhere else. Collapsing the two would make every worker's boot line look broken.
    """
    report = mamba_anchor.install(None)
    assert report["installed"] is False
    assert report["verdict"] == mamba_anchor.VERDICT_DISABLED
    assert "is imported in this process" in report["reason"]
    assert mamba_anchor.boot_token() == "MAMBA_ANCHOR[disabled]"


# --- 4. the spill side: an anchor that reaches the store ---------------------

def test_anchor_spill_reaches_disk(tier):
    """The align-mode pairs of a hand-off ARE the anchors that reached the store.

    Driven through the REAL hand-off path - `observe_bind` then `on_store_completed` -
    with the live box's group table (`spanfx.live_handler`: groups 0-3 align, group 4
    parked, group 5 positional) and block keys built the way the engine packs them
    (`spanfx.key_for`: the digest plus the 4-byte group id).

    THE CONTROL IS THE SECOND HAND-OFF, not an absent counter: an attention-only
    hand-off on the same armed store must leave the anchor counter where it was,
    because a span with no align group is exactly the shape that cannot anchor
    anything. Both numbers move together with `spill_spans_complete`, so the two
    counters cannot quietly disagree about the same bytes.
    """
    role = spanfx.live_handler()
    assert disk_store.counters()["anchor_spills_completed"] == 0

    complete = [(slot, spanfx.key_for(spanfx.SPAN_A, group))
                for slot, group in enumerate(spanfx.ALIGN_GROUPS +
                                             (spanfx.ATTENTION_GROUP,))]
    report = spanfx.spill_a_handoff(role, complete)
    assert report["queued"] == 5, report
    state = disk_store.counters()
    # FOUR align-mode files: groups 0-3 for one span. The attention block is not an
    # anchor (it is what a run walks OVER), so counting it would be a units bug.
    assert state["anchor_spills_completed"] == len(spanfx.ALIGN_GROUPS), state
    assert state["spill_spans_complete"] == 1

    # THE CONTROL: the 20% shape. Same armed store, attention only, no anchor at all.
    orphan = [(0, spanfx.key_for(spanfx.SPAN_B, spanfx.ATTENTION_GROUP))]
    spanfx.spill_a_handoff(role, orphan, event=8)
    state = disk_store.counters()
    assert state["spill_incomplete_state"] == 1
    assert state["anchor_spills_completed"] == len(spanfx.ALIGN_GROUPS), (
        "an attention-only hand-off moved the anchor counter: an incomplete span is "
        "not an anchor")

    # A PARKED GROUP IS NOT AN ANCHOR EITHER (group 4 is positional, not align).
    spanfx.spill_a_handoff(role, [(0, spanfx.key_for(spanfx.SPAN_C, 4))], event=9)
    assert disk_store.counters()["anchor_spills_completed"] == \
        len(spanfx.ALIGN_GROUPS)


def test_anchor_spills_completed_reaches_the_rank_rollup(diag_rank_dir, monkeypatch):
    """A counter missing from the rollup whitelist is SILENTLY NEVER ROLLED UP.

    Two halves, both necessary: the WRITER must put the field in the rank document
    (`disk_store.worker_report()`, read here as the real writer's output rather than as
    a hand-written fixture) and the READER's whitelist (`RANK_ROLLUP_FIELDS`, which
    `_rank_file_rollup_row()` walks) must dig it. Get either wrong and the totals carry
    `None` forever, which is the failure mode the task calls out by name.
    """
    from test_disk_spill_rank_rollup import disk_section, write_rank

    monkeypatch.setitem(disk_store._state, "anchor_spills_completed", 17)
    report = disk_store.worker_report()
    assert report["anchor_spills_completed"] == 17, (
        "the writer does not publish the field: the rollup can never see it")
    assert ("anchor_spills_completed", ("disk", "anchor_spills_completed"),
            "anchor files") in disk_store.RANK_ROLLUP_FIELDS
    assert "anchor_spills_completed" in disk_store.RANK_ROLLUP_MEANINGS

    for rank, count in (("PP0_TP0", 17), ("PP0_TP1", 23)):
        write_rank(diag_rank_dir, rank,
                   disk=disk_section(anchor_spills_completed=count))
    rollup = disk_store.rank_rollup(force=True)
    totals = rollup["totals"]
    assert totals["anchor_spills_completed"] == 40, totals
    assert rollup["units"]["anchor_spills_completed"] == "anchor files"

    # THE NEGATIVE CONTROL, and it is the failure the task warns about: a rank file
    # that does NOT carry the field yields None ("this rank never told us"), not a 0
    # folded into a confident total.
    for path in diag_rank_dir.glob("rank-*.json"):
        path.unlink()
    write_rank(diag_rank_dir, "PP0_TP0", disk=disk_section())
    assert disk_store.rank_rollup(force=True)["totals"][
        "anchor_spills_completed"] is None


# --- 5. the restore side: a run end that lands on an anchor ------------------

@pytest.fixture
def anchor_store(monkeypatch, armed):
    """The gate open, with an INDEX THAT ANSWERS FOR TWO GROUPS AND STOPS AT BLOCK 4.

    `disk_index.contains` is the one seam: block 0-3 named for the positional group and
    0-3 for the align group, block 4 missing from the positional walk. That is enough
    for `_disk_run` to walk a 4-block run and to price its END with an anchor, which is
    the event `restore_anchor_hits_total` counts - and enough for the mirror test, which
    withdraws the align name and gets `no_anchor`.
    """
    positional, boundary = 5, 0
    held = set()

    def fake_contains(key_hex, ranks=()):
        return key_hex in held

    monkeypatch.setattr(disk_restore.disk_index, "contains", fake_contains)
    monkeypatch.setattr(disk_restore, "_required_ranks", lambda: ())
    return {"positional": positional, "boundary": boundary, "held": held}


def _seed(anchor_store, blocks=4, groups=("positional", "boundary")):
    """The index's answer for `blocks` blocks, in the groups named.

    `groups` is the whole difference between an anchorable run and `no_boundary_anchor`:
    with both groups held, `_disk_run` walks four positional blocks AND finds an align
    name at the end; with only the positional group held, the walk still gets four
    blocks and NO end it can pay for. Same store, one withdrawn name, two verdicts -
    which is what makes the counter's control non-vacuous.
    """
    anchor_store["held"].clear()
    for index in range(blocks):
        for which in groups:
            anchor_store["held"].add(base.key_hex(index + 1, anchor_store[which]))
    return anchor_store


def test_anchor_restore_lands(armed_gap_fill_anchor, monkeypatch):
    """A run end with an anchor is COUNTED while the patch is live, and not without it.

    The run walk (`_disk_run`) is the only place that knows "this end block is one
    every align-mode group can name", so that is where the landing is recorded - and
    the counter is gated on the patch, so a zero on an unpatched boot is not a claim
    that no anchors exist. The control half runs the SAME walk with the patch reported
    as not installed and must see nothing.
    """
    cls, _ns = fork_mamba()
    mamba_anchor.install(cls)
    state = disk_restore.counters()
    assert state["restore_anchor_hits_total"] == 0

    run = _run_with_anchor()
    assert run["ok"] and run["blocks"] == 4, run
    assert run["boundary_groups"] == [0] and run["anchors"], run
    assert disk_restore.counters()["restore_anchor_hits_total"] == 1

    # THE CONTROL: the same walk, the same store, the patch not live here.
    mamba_anchor.reset_state_for_tests()
    before = disk_restore.counters()["restore_anchor_hits_total"]
    run2 = _run_with_anchor()
    assert run2["ok"] and run2["blocks"] == 4, run2
    assert disk_restore.counters()["restore_anchor_hits_total"] == before, (
        "the counter moved on a boot the patch is not live in")


@pytest.fixture
def anchor_store_no_anchor(armed):
    """The gate open, with the align group's names WITHDRAWN.

    The store holds the positional blocks and nothing for the align group: the
    `no_boundary_anchor` shape, which is the control for the restore-side counter.
    """
    class Only:
        positional = 5
        boundary = 0
        held: set = set()

    store = Only()
    store.held = set()
    return store


def _run_with_anchor():
    """One `_disk_run` over the seeded store: four blocks, group 0 anchoring the end."""
    request = base.Req(blocks=4)
    return disk_restore._disk_run(
        request, 0, [0, 5], base.BLOCK, base.HASH_BLOCK,
        {"pack": base.pack, "resolve": base.resolve},
        contracts={"0": "boundary_state", "5": "positional"},
        ranks=())


def test_a_run_without_an_anchor_lands_nowhere(anchor_store_no_anchor, monkeypatch):
    """The anti-vacuity control for the restore counter: NO anchor, NO count.

    Same walk, same helpers, same patched boot - only the align group's names are
    withdrawn from the index. `_disk_run` walks the positional run and finds no end it
    can price an anchor at, which is `no_boundary_anchor`. If this test counted, the
    counter would be measuring offers rather than anchors.
    """
    held = anchor_store_no_anchor.held
    held.add(base.key_hex(1, 5))
    held.add(base.key_hex(2, 5))
    held.add(base.key_hex(3, 5))
    held.add(base.key_hex(4, 5))
    monkeypatch.setattr(disk_restore.disk_index, "contains",
                        lambda key_hex, ranks=(): key_hex in held)
    monkeypatch.setattr(disk_restore, "_required_ranks", lambda: ())

    cls, _ns = fork_mamba()
    mamba_anchor.install(cls)
    assert disk_restore.counters()["restore_anchor_hits_total"] == 0

    run = _run_with_anchor()
    assert run["ok"] and run["blocks"] == 0, run
    assert run["no_anchor"] is True, run
    assert run["anchors"] == {}, run
    assert disk_restore.counters()["restore_anchor_hits_total"] == 0


# --- 6. the doors: every counter reaches a reader ----------------------------

def test_mamba_patch_installed_reaches_the_backport_section_and_the_boot_line():
    """The boot fact is published where the FIRST target's facts are published.

    `backport.section()` is what `snapshot.backport_section()` returns and what `/tier`
    answers with; `boot_line()` is the line the operator greps. Both must carry the
    verdict, and the line must carry it as the DISTINCT token - because the absence of
    a token and a token saying `disabled` are the same read, and neither may be
    confusable with `patched`.
    """
    cls, _ns = fork_mamba()
    mamba_anchor.install(cls)
    section = backport.section()
    assert section["mamba_anchor"]["installed"] is True
    counters = section["mamba_anchor"]["counters"]
    assert counters["mamba_patch_installed"] is True
    assert counters["anchor_offers_total"] == 0

    line = backport.boot_line({"mode": "auto", "backport": True, "guard": True,
                               "tier": True, "faults_at_boot": 0, "nested": [],
                               "refusals": [], "not_covered": [],
                               "guard_armed": False})
    assert "MAMBA_ANCHOR[patched]" in line
    # The restart verifier's grep is unaffected: it takes the LAST `tier=`-shaped
    # match, and this token has no `=` at all.
    assert re.findall(r"tier=[A-Za-z_]+", line)[-1] == "tier=ARMED"
    # Every `=`-shaped field name appears at most once (tests/test_auto_boot.py's rule).
    names = [part.split("=")[0] for part in line.split(" ") if "=" in part]
    assert names.count("mamba_anchor") == 0, "the token is not a field"

    mamba_anchor.reset_state_for_tests()
    assert "MAMBA_ANCHOR[disabled]" in backport.boot_line({"mode": "auto"})


def test_anchor_offers_total_reaches_the_dev_state_sections():
    """`snapshot.dev_state_sections()`'s disk_store section carries the mark count.

    The dev payload is the door a dashboard polls, and the counter lives in the
    scheduler - which is why the section also carries `mamba_anchor`'s own verdict and
    units, so a zero from the API process is readable as "that process runs no masks"
    rather than as "nothing was marked".
    """
    cls, _ns = fork_mamba()
    mamba_anchor.install(cls)
    cls.reachable_block_mask(**mask_kwargs())

    sections = snapshot.dev_state_sections()
    store_section = sections["disk_store"]
    assert store_section["mamba_anchor"]["counters"]["anchor_offers_total"] == 1
    assert store_section["mamba_anchor"]["units"]["anchor_offers_total"]

    # ...and the same number through the verb door, so the two cannot disagree.
    assert disk_store.hook_section()["mamba_anchor"]["counters"][
        "anchor_offers_total"] == 1
    assert disk_store.section()["mamba_anchor"]["counters"][
        "anchor_offers_total"] == 1


def test_restore_anchor_hits_total_reaches_counters_and_the_rank_file():
    """`disk_restore.counters()` publishes it with a unit, and so does the rank section.

    A rank file reports its OWN process's counters (a worker never runs the walk, so it
    honestly reports 0), and it is listed anyway: a reader comparing four rank files
    with the scheduler's section must be able to see the field belongs to the other
    process rather than infer a fault from a zero.
    """
    disk_restore.reset_state_for_tests()
    monkey_count = 3
    disk_restore._state["restore_anchor_hits_total"] = monkey_count
    counters = disk_restore.counters()
    assert counters["restore_anchor_hits_total"] == monkey_count
    assert counters["units"]["restore_anchor_hits_total"], \
        "a bare number in the payload"
    assert disk_restore.scheduler_section()["restore_anchor_hits_total"] == \
        monkey_count
    assert disk_restore.worker_report()["restore_anchor_hits_total"] == \
        monkey_count


def test_every_new_counter_carries_a_unit():
    """No bare numbers: each of the four arrives with a `units` entry at its door."""
    mamba_units = mamba_anchor.section()["units"]
    for name in ("mamba_patch_installed", "anchor_offers_total"):
        assert mamba_units.get(name), name
    for name, value in mamba_anchor.counters().items():
        if isinstance(value, int) and not isinstance(value, bool):
            assert mamba_units.get(name), f"{name} is a bare number in the payload"

    rollup_units = {field: unit for field, _src, unit
                    in disk_store.RANK_ROLLUP_FIELDS}
    assert rollup_units["anchor_spills_completed"] == "anchor files"
    assert "anchor_spills_completed" in disk_store.RANK_ROLLUP_MEANINGS


# --- 7. the claim that lets this patch exist at all --------------------------

def test_nothing_about_serving_posture_changed():
    """BEHAVIOUR: an install plus a mark moves no arm, no posture, no switch.

    The patch is spill-side: it can add data to the store and nothing else. This reads
    every posture and arm the plugin owns, installs, drives the mask, and reads them
    again - all of them must be the identical object, and the mask patch must not have
    armed, offered or spilled anything of its own accord.
    """
    before = {
        "disk_authority": disk_restore.counters()["disk_authority"],
        "serving_as": disk_restore.counters()["serving_as"],
        "offers_enabled": disk_restore.counters()["offers_enabled"],
        "ram_armed": backport.is_armed(),
        "spill_armed": disk_spill.is_armed(),
        "offers": disk_restore.counters()["offers"],
        "spill_queued": disk_store.counters()["spill_queued"],
    }
    cls, _ns = fork_mamba()
    report = mamba_anchor.install(cls)
    cls.reachable_block_mask(**mask_kwargs())

    after = {
        "disk_authority": disk_restore.counters()["disk_authority"],
        "serving_as": disk_restore.counters()["serving_as"],
        "offers_enabled": disk_restore.counters()["offers_enabled"],
        "ram_armed": backport.is_armed(),
        "spill_armed": disk_spill.is_armed(),
        "offers": disk_restore.counters()["offers"],
        "spill_queued": disk_store.counters()["spill_queued"],
    }
    assert after == before, f"the patch changed a posture: {(before, after)}"
    assert report["installed"] is True, "and the install worked, so the above is real"


def _executable_source(module) -> str:
    """A module's source with DOCSTRINGS and COMMENTS removed.

    The claim is about code, and a docstring that NAMES the switch to explain that the
    module never touches it must not fail the test that enforces the claim - while a
    real write, or a string key passed to something that writes, must. So the prose is
    stripped and the code is not: every docstring span (module, class, function) and
    every comment goes, everything else stays verbatim.
    """
    source = inspect.getsource(module)
    lines = source.splitlines()
    drop = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            # The docstring is the first statement's string constant.
            stmt = node.body[0]
            for line in range(stmt.lineno, stmt.end_lineno + 1):
                drop.add(line)
    kept = [line for number, line in enumerate(lines, start=1)
            if number not in drop and not line.lstrip().startswith("#")]
    return "\n".join(kept)


def test_the_modules_source_names_no_posture_or_arm_writer():
    """STRUCTURE: the same claim as an AST fact about `mamba_anchor.py` itself.

    Behavioural equality over one test run cannot rule out a door that fires on another
    path; the source can. Nothing in this module's CODE may name the serve switch, the
    offer switch, the RAM tier's arm or the spill arm - and it may not touch the
    environment directly either, so it cannot hand a later component a value it
    invented. Docstrings and comments are stripped first (`_executable_source`): this
    module's docstring names `disk_authority` in order to say it never touches it, and
    prose must not be what a structural test reads.
    """
    source = inspect.getsource(mamba_anchor)
    code = _executable_source(mamba_anchor)
    tree = ast.parse(source)
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden = {"set_disk_authority", "set_offer_switch", "set_armed",
                 "disk_authority", "offers_enabled", "serve_as", "serving_as"}
    assert not (named & forbidden), sorted(named & forbidden)
    assert not (attrs & forbidden), sorted(attrs & forbidden)
    # In the CODE (prose aside), neither switch's name appears at all - not as an
    # identifier, not as an attribute, not as a string key handed to anything.
    for word in sorted(forbidden):
        assert word not in code, word
    # Only the config reader is consulted for the gate, and never the environment.
    assert "os.environ" not in code
    assert "mamba_anchor_patch()" in code
    # The dependency is ONE-WAY and READ-ONLY, checked as an AST fact rather than a
    # string count (which would read comments and unit prose as calls). The restore
    # layer imports the patch module and touches exactly ONE attribute of it - the
    # liveness read that labels its counter. It never installs, never arms, never
    # switches: if a later edit reaches for anything else on this module, this fails.
    restore_tree = ast.parse(inspect.getsource(disk_restore))
    touched = {node.attr for node in ast.walk(restore_tree)
               if isinstance(node, ast.Attribute)
               and isinstance(node.value, ast.Name)
               and node.value.id == "mamba_anchor"}
    assert touched == {"is_installed"}, touched
    assert "mamba_anchor.install" not in inspect.getsource(disk_restore)
    assert "set_disk_authority" not in _executable_source(mamba_anchor)


def test_the_patch_does_not_reach_the_engine_or_the_disk():
    """No I/O, no engine objects, no network: the module is a pure wrapper plus state.

    The mask patch runs on the scheduler's cache path. An `open`, a `socket`, an
    `asyncio` primitive or a vllm import appearing here would be a side effect on the
    request path, and this file's whole argument for why the patch is safe to run does
    not survive one.
    """
    source = inspect.getsource(mamba_anchor)
    for forbidden in ("import vllm", "from vllm", "open(", "socket",
                      "threading", "asyncio", "subprocess", "requests."):
        assert forbidden not in source, forbidden
    imported = {alias.name for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {"__future__", "hashlib", "inspect", "logging", "sys"}


# --- the fixtures this file needs from its siblings --------------------------

@pytest.fixture
def armed_gap_fill_anchor(anchor_store):
    """The gate open AND the two-group store seeded, as one fixture.

    `armed` (from `test_disk_restore`) opens the gate and installs the fork helpers;
    `anchor_store` answers the index for the run this file walks. Requesting `armed`
    through `anchor_store` means a test cannot get the gate without the store.
    """
    _seed(anchor_store)
    return anchor_store


@pytest.fixture
def diag_rank_dir(tmp_path, monkeypatch):
    """The diagnostics directory the rank rollup reads."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    return directory
