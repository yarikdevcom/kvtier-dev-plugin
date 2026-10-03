"""The in-tree stand-down: when the engine's OWN file already carries #56810.

The runtime backport exists because the fix could not be landed in the engine's
source tree. It can be now (the tracked `config/box/kvtier-pr56810/` patch,
applied with `config/box/kvtier-patch.sh` in the llm-api repo), so the runtime
path must STAND DOWN when the in-tree fix is present - and must not, anywhere
else. Patching from outside a process we do not control has cost three
production incidents; this file is the contract that keeps the two mechanisms
from stacking on each other.

Five properties, all CPU-only, all driven through the REAL `backport.install()`
with fixture classes standing in for `SimpleCPUOffloadScheduler`:

1. stand-down      - an engine whose four bodies ARE the upstream text installs
                     NOTHING and reports `already_fixed_in_tree`
2. fallback intact - an unpatched engine still gets the runtime backport
3. no double-patching - a stand-down touches no class attribute, and repeating it
                     changes nothing
4. partial is a problem - a half-applied in-tree patch is REFUSED and disarmed,
                     never completed from a plugin
5. the layer lesson - a class WE patched at runtime classifies as `unpatched`,
                     never as fixed-in-tree (hashing our own wrapper and blaming
                     the fork is incident #3, and it must stay impossible)

Method text comes from the same two authoritative places as the rest of the
suite: the fork's own ORIGINAL bodies (`fixtures_pr56810`, fingerprint-verified)
and the plugin's stored #56810 bodies (`pr56810_sources.SOURCES[..]["source"]`).
"""

from __future__ import annotations

import hashlib
import inspect
import linecache
import types
from pathlib import Path

import fixtures_pr56810 as FX
import pytest

from kvtier_dev_plugin import backport, guard_mode, pr56810_sources

ORDER = ("__init__", "_estimate_lazy_target_blocks",
         "update_state_after_alloc", "_select_eager_blocks_to_store")

# Stand-in bodies for the kill-switch entry points. The fixture class is only
# supposed to be a `SimpleCPUOffloadScheduler` SHAPE, but `_install_switch`
# refuses a method that does not exist ("the fork renamed it") - so without
# these the "the kill switch still goes on" assertion could never be true. The
# bodies are irrelevant: a switch wrapper only adds a gated early return.
SWITCH_STUBS = """    def get_num_new_matched_tokens(self, *args, **kwargs):
        return None

    def build_connector_meta(self, *args, **kwargs):
        return None

    def has_pending_stores(self, *args, **kwargs):
        return False

    def _queue_finished_eager_store(self, *args, **kwargs):
        return None
"""

_builds = 0


@pytest.fixture(autouse=True)
def reset_backport_state():
    """`backport._state` is process-wide, and the reset fixture in
    `test_backport.py` is MODULE-scoped-by-definition, so it does not cover this
    file. Without its own reset, a stand-down count leaks across tests and every
    "we installed nothing" assertion passes or fails on test order instead of on
    the code. Same idiom: reset through the module's own function.
    """
    backport.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    yield
    backport.reset_state_for_tests()
    guard_mode.reset_state_for_tests()


def selector_namespace():
    """Module globals the original function closes over.

    Only what the ORIGINAL bodies reference; a name a test does not drive is not
    needed, because these fixtures are classified and installed, never called.
    """
    return {
        "resolve_block_hashes": lambda *a, **k: [],
        "_StoreAdmission": types.SimpleNamespace(READY="ready"),
        "BlockStoreMeta": dict,
        "maybe_convert_hash": lambda h: h,
        "cdiv": lambda a, b: -(-a // b),
        "logger": types.SimpleNamespace(warning=lambda *a, **k: None),
    }


def build_class(sources, ns=None):
    """Build a stand-in `SimpleCPUOffloadScheduler` from a per-method source map.

    Each built class gets its OWN pseudo filename: `inspect.getsource()` resolves
    through `linecache` by filename, so two classes sharing a filename would
    read each other's text and every hash in this file would be a lie.
    """
    global _builds  # noqa: PLW0603 - a module counter, the pytest idiom
    _builds += 1
    name = f"<in_tree_fixture_{_builds}>"
    src = ("class SimpleCPUOffloadScheduler:\n"
           + "".join(
               (sources[n] if sources[n].endswith("\n") else sources[n] + "\n")
               for n in ORDER)
           + SWITCH_STUBS)
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    module = dict(ns or {})
    module["__name__"] = "in_tree_fixture"
    module["__file__"] = name
    exec(compile(src, name, "exec"), module, module)  # noqa: S102
    cls = module["SimpleCPUOffloadScheduler"]
    cls.__module__ = "in_tree_fixture"
    return cls


def unpatched_class():
    return build_class(FX.ORIGINAL_SOURCES, selector_namespace())


def fixed_in_tree_class(overrides=None):
    """An engine whose file already carries the #56810 bodies."""
    sources = dict(FX.ORIGINAL_SOURCES)
    for name in ORDER:
        sources[name] = pr56810_sources.SOURCES[name]["source"]
    sources.update(overrides or {})
    return build_class(sources, selector_namespace())


def snapshot_attrs(cls):
    """Identity of every class attribute we care about - the "did we touch it?"
    witness. `id()` is enough: the objects are alive for the whole test."""
    return {name: id(cls.__dict__.get(name)) for name in ORDER}


# --- the operator's one line distinguishes the two steady states ------------

def live_manager_stand_in():
    """A stand-in for the LIVE `SimpleCPUOffloadScheduler` instance.

    `_live_object_check` reads attributes only, and the boot is what turns a
    class-level verdict into a tier, so the instance has to carry the additive
    attribute the same way a construct-first engine's would.
    """
    return types.SimpleNamespace(
        prefix_cacheable_group_ids=(0, 1, 2, 3, 4),
        cpu_kv_cache_config=types.SimpleNamespace(
            prefix_cacheable_group_ids=(0, 1, 2, 3, 4), num_blocks=8,
            kv_cache_groups=[]),
        kv_cache_config=None,
        block_size=3200, hash_block_size=3200,
        cpu_coordinator=types.SimpleNamespace(enable_partial_hash_hits=True),
        cpu_block_pool=object(), group_block_sizes=[3200],
        enable_kv_cache_events=False, _lazy_mode=False,
        _reqs_to_store={}, _pending_cpu_hits={}, _reqs_to_load={},
        _gpu_block_pool=object(),
    )


def test_the_boot_line_says_in_tree_rather_than_yes_on_a_stand_down():
    """`backport=YES` and `backport=IN_TREE` are different boxes.

    Both are fixed; only one is still being patched at runtime by a plugin. After
    this work IN_TREE is the state every box should reach, so the single line an
    operator greps for must be able to say which one it is looking at.
    """
    cls = fixed_in_tree_class()
    manager = live_manager_stand_in()
    report = backport.install(cls, manager=manager)
    assert report["already_fixed_in_tree"] is True

    boot = backport.boot_from_connector(cls, report, manager)
    assert boot["backport"] is True, boot
    assert boot["backport_source"] == "in_tree", boot
    assert "backport=IN_TREE" in boot["line"], boot["line"]
    assert "backport=YES" not in boot["line"]


def test_the_boot_line_still_says_yes_for_a_runtime_patch():
    cls = unpatched_class()
    manager = live_manager_stand_in()
    report = backport.install(cls, manager=manager)
    assert report["installed"] is True
    assert "already_fixed_in_tree" not in report

    boot = backport.boot_from_connector(cls, report, manager)
    assert boot["backport_source"] == "runtime", boot
    assert "backport=YES" in boot["line"], boot["line"]
    assert "backport=IN_TREE" not in boot["line"]


# --- the pinned post-patch fingerprints must describe a real release ---------

def test_the_package_version_matches_the_build_metadata():
    """The deploy script prints the installed version and the backport's sha256.

    The 0.1.0 that killed the engine and the 0.1.0 that fixed it were
    indistinguishable from the box, which is why the version exists at all; a
    `__version__` that has drifted from pyproject re-creates that blindness.
    """
    from kvtier_dev_plugin import __version__
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text()
    declared = [ln.split('"')[1] for ln in text.splitlines()
                if ln.startswith("version = ")]
    assert declared, "pyproject has no version"
    assert __version__ == declared[0], (
        f"__version__={__version__} but pyproject says {declared[0]}")


# --- the pinned post-patch fingerprints describe the stored bodies -------

def test_pinned_post_patch_fingerprints_match_the_stored_bodies():
    """`PATCHED_SHA256` is pinned, so a body edit must be re-pinned on purpose.

    These are the hashes `in_tree_state()` recognises as "already fixed". If one
    drifted from the body the plugin would install, the stand-down would either
    never fire or fire for the wrong text - and a silent stand-down is an
    unpatched engine facing live traffic.
    """
    assert set(pr56810_sources.PATCHED_SHA256) == set(backport.METHODS)
    for name, entry in pr56810_sources.SOURCES.items():
        want = pr56810_sources.PATCHED_SHA256[name]
        got = hashlib.sha256(entry["source"].encode()).hexdigest()
        assert got == want, f"{name}: pinned {want[:12]} != body {got[:12]}"
        # And the two states must stay distinguishable, or the classification
        # has nothing to decide between.
        assert want != entry["orig_sha256"], name


def test_the_fixed_bodies_hash_to_the_pinned_values_through_inspect():
    """The fixture path really does produce those hashes (no fixture cheat)."""
    cls = fixed_in_tree_class()
    state = backport.in_tree_state(cls)
    for name in ORDER:
        fn, _ = backport._underlying(cls, name)
        live = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        assert live == pr56810_sources.PATCHED_SHA256[name], name
        assert state["methods"][name]["live_sha256"] == live, name


# --- 1. classification ------------------------------------------------------

def test_an_unpatched_engine_classifies_as_unpatched():
    state = backport.in_tree_state(unpatched_class())
    assert state["checked"] is True
    assert state["all_fixed"] is False and state["partial"] is False
    assert state["fixed"] == []
    assert sorted(state["unpatched"]) == sorted(backport.METHODS)
    assert state["methods"]["__init__"]["live_sha256"] == \
        pr56810_sources.SOURCES["__init__"]["orig_sha256"]


def test_a_fully_patched_engine_classifies_as_fixed_in_tree():
    state = backport.in_tree_state(fixed_in_tree_class())
    assert state["checked"] is True
    assert state["all_fixed"] is True and state["partial"] is False
    assert state["reason"] == backport.IN_TREE_FIXED
    assert sorted(state["fixed"]) == sorted(backport.METHODS)
    assert state["unpatched"] == [] and state["other"] == []
    for name in ORDER:
        assert state["methods"][name]["state"] == backport.IN_TREE_FIXED, name


def test_a_partially_patched_engine_is_flagged_as_partial():
    """Two methods in the tree, two still on v0.30.0: the crash survives."""
    half = dict(FX.ORIGINAL_SOURCES)
    half["__init__"] = pr56810_sources.SOURCES["__init__"]["source"]
    half["_estimate_lazy_target_blocks"] = (
        pr56810_sources.SOURCES["_estimate_lazy_target_blocks"]["source"])
    state = backport.in_tree_state(build_class(half, selector_namespace()))
    assert state["all_fixed"] is False
    assert state["partial"] is True
    assert state["fixed"] == ["__init__", "_estimate_lazy_target_blocks"]
    assert sorted(state["unpatched"]) == ["_select_eager_blocks_to_store",
                                          "update_state_after_alloc"]
    # The reason must NAME the halves: "partially patched" without the names
    # sends the operator looking for a bug instead of a missing hunk.
    assert "__init__" in state["reason"]
    assert "update_state_after_alloc" in state["reason"]


def test_a_drifted_body_is_other_neither_fixed_nor_unpatched():
    """Stand-down must not swallow a fork change into a comfortable answer."""
    drifted = dict(FX.ORIGINAL_SOURCES)
    drifted["update_state_after_alloc"] = (
        FX.ORIGINAL_SOURCES["update_state_after_alloc"]
        .replace("req_id = request.request_id",
                 "req_id = request.request_id  # the fork edited this"))
    state = backport.in_tree_state(build_class(drifted, selector_namespace()))
    assert state["all_fixed"] is False and state["partial"] is False
    assert state["methods"]["update_state_after_alloc"]["state"] == "other"
    assert "neither" in state["methods"]["update_state_after_alloc"]["reason"]


def test_missing_and_unreadable_bodies_are_classified_not_guessed():
    state = backport.in_tree_state(None)
    assert state["checked"] is False and state["all_fixed"] is False

    class Bare:  # none of the four methods exist
        pass

    state = backport.in_tree_state(Bare)
    assert state["all_fixed"] is False and state["partial"] is False
    assert all(state["methods"][n]["state"] == "missing" for n in ORDER)


# --- 2. the stand-down itself ----------------------------------------------

def test_install_stands_down_and_reports_already_fixed_in_tree():
    cls = fixed_in_tree_class()
    before = snapshot_attrs(cls)

    report = backport.install(cls)

    assert snapshot_attrs(cls) == before, "a stand-down replaced an attribute"
    assert report["installed"] is True, report
    assert report["already_fixed_in_tree"] is True
    assert report["in_tree"]["all_fixed"] is True
    assert report["refused_methods"] == []
    assert report.get("auto_disarmed") is False
    for name in backport.METHODS:
        verdict = report["methods"][name]
        assert verdict["patched"] is True, name
        assert verdict["already_fixed_in_tree"] is True, name
        assert verdict["reason"].startswith(backport.IN_TREE_FIXED), name


def test_a_stand_down_installs_no_code_of_ours():
    """The claim "nothing was installed" is checked, not asserted.

    `installed: True` says the requirement is met; the marker and the attribute
    identity say we did not ALSO become a second, invisible patch of a file the
    operator now tracks. Both must hold at once.
    """
    cls = fixed_in_tree_class()
    before = snapshot_attrs(cls)
    backport.install(cls)

    assert snapshot_attrs(cls) == before, "a stand-down replaced a class attribute"
    for name in backport.METHODS:
        raw = cls.__dict__.get(name)
        fn = backport._function_of(raw) if raw is not None else None
        assert getattr(fn, backport.OWNER_ATTR, None) is None, name
    # The install counters count code WE fitted; a stand-down fitted none.
    assert backport._state["installs"] == 0
    assert backport._state["reinstalls"] == 0
    assert backport._state["in_tree_standdowns"] == 1
    assert backport._state["fully_installed"] is True


def test_stand_down_is_idempotent_and_never_double_patches():
    cls = fixed_in_tree_class()
    before = snapshot_attrs(cls)

    first = backport.install(cls)
    second = backport.install(cls)
    third = backport.install(cls)

    for report in (first, second, third):
        assert report["installed"] is True and report["already_fixed_in_tree"]
    assert snapshot_attrs(cls) == before
    assert backport._state["installs"] == 0
    assert backport._state["in_tree_standdowns"] == 3


def test_the_kill_switch_and_the_rest_of_the_plugin_still_install():
    """Only the four #56810 bodies stand down - the kill switch is unrelated.

    A fixed-in-tree engine still needs the arm/disarm gate, so the switch
    wrappers must go on exactly as they do on an unpatched engine.
    """
    cls = fixed_in_tree_class()
    report = backport.install(cls)
    for name in backport.SWITCH_METHODS:
        assert report["switch"][name].get("wrapped") is True, (
            f"{name}: {report['switch'][name]}")
    # And guard mode is untouched by the decision.
    assert report["geometry"] is not None


# --- 3. the runtime fallback stays intact -----------------------------------

def test_an_unpatched_engine_still_gets_the_runtime_backport():
    """The fallback for a box where the tracked patch was never applied."""
    cls = unpatched_class()
    before = snapshot_attrs(cls)

    report = backport.install(cls)

    assert report["installed"] is True, report
    assert "already_fixed_in_tree" not in report
    assert report["in_tree"]["all_fixed"] is False
    assert snapshot_attrs(cls) != before, "nothing was installed"
    for name in backport.METHODS:
        raw = cls.__dict__.get(name)
        fn = backport._function_of(raw) if raw is not None else None
        assert getattr(fn, backport.OWNER_ATTR, None) == backport.OWNER_TAG, name
    assert backport._state["installs"] == 1


def test_a_class_we_patched_at_runtime_is_never_claimed_as_fixed_in_tree():
    """Incident #3, made permanent: never hash our own wrapper.

    After a runtime install the LIVE bodies hash to the #56810 text - our copies
    of it. If `in_tree_state()` hashed what the class holds, it would report
    "already fixed in tree" for our own patch, stand down on a reload, and tell
    the operator the engine's file is fixed when it is not. Walking beneath our
    layers instead, the bottom is still the fork's v0.30.0 original.
    """
    cls = unpatched_class()
    backport.install(cls)

    state = backport.in_tree_state(cls)
    assert state["all_fixed"] is False, state
    assert state["fixed"] == [], state
    assert sorted(state["unpatched"]) == sorted(backport.METHODS), state
    for name in backport.METHODS:
        # Our layer is visible in the classification, and is not what was hashed.
        assert state["methods"][name]["layers"], name
        assert state["methods"][name]["live_sha256"] == (
            pr56810_sources.SOURCES[name]["orig_sha256"]), name


def test_a_fixed_engine_reinstalling_our_layers_does_not_claim_in_tree():
    """Guard/switch layers over an in-tree-fixed body still classify as fixed.

    The mirror of the test above: layers are fine to walk through, so a re-poll
    on a patched-and-guarded engine keeps standing down instead of flipping to
    "the fork changed this function".
    """
    cls = fixed_in_tree_class()
    backport.install(cls)          # stand-down: only the switch layers go on
    state = backport.in_tree_state(cls)
    assert state["all_fixed"] is True, state
    assert state["partial"] is False


# --- 4. a partial in-tree patch is a problem --------------------------------

def test_a_partially_patched_engine_is_refused_not_silently_completed():
    cls = fixed_in_tree_class({
        "update_state_after_alloc": FX.ORIGINAL_SOURCES[
            "update_state_after_alloc"],
        "_select_eager_blocks_to_store": FX.ORIGINAL_SOURCES[
            "_select_eager_blocks_to_store"],
    })
    before = snapshot_attrs(cls)

    report = backport.install(cls)

    assert report["installed"] is False, report
    assert "already_fixed_in_tree" not in report
    assert report["auto_disarmed"] is True
    assert report["disarm_reason"] == "incomplete_install"
    assert "PARTIALLY patched in-tree" in report["reason"], report["reason"]
    # It must say WHICH half is which, and point at the tracked patch.
    assert "__init__" in report["reason"]
    assert "update_state_after_alloc" in report["reason"]
    # Nothing was patched - a refusal must not half-do the thing it refuses.
    assert snapshot_attrs(cls) == before
    assert backport._state["installs"] == 0
    assert backport._state["fully_installed"] is False


def test_a_partial_engine_keeps_both_crash_routes_gated_while_refused():
    """The refusal is safe to sit on: the kill switch went on BEFORE it.

    Same ordering rule the geometry veto follows - a refused tier is a no-op, not
    a crash, because both routes into the selector are gated.
    """
    cls = fixed_in_tree_class({
        "update_state_after_alloc": FX.ORIGINAL_SOURCES[
            "update_state_after_alloc"],
    })
    report = backport.install(cls)
    assert report["installed"] is False
    for name in backport.SWITCH_METHODS:
        assert report["switch"][name].get("wrapped") is True, name


def test_a_missing_method_means_no_stand_down_and_is_refused_as_partial():
    """A class that does not even have all four bodies cannot be "fixed in tree".

    Dropping one method out of the class must not be read as "that one is fine".
    Three fixed bodies plus one missing is the SAME problem as a half-applied
    patch (the crash lives on in the missing half), so it is refused too - and
    `install()` never reaches the point of wrapping a class whose shape moved.
    """
    sources = dict(FX.ORIGINAL_SOURCES)
    for name in ORDER:
        sources[name] = pr56810_sources.SOURCES[name]["source"]
    sources["update_state_after_alloc"] = (
        "    def renamed_away(self):\n        return 1\n")
    cls = build_class(sources, selector_namespace())

    state = backport.in_tree_state(cls)
    assert state["all_fixed"] is False
    assert state["partial"] is True, state
    assert state["methods"]["update_state_after_alloc"]["state"] == "missing"
    assert "update_state_after_alloc" in state["reason"]

    report = backport.install(cls)
    assert report.get("already_fixed_in_tree") is None, report
    assert report["installed"] is False, report
    assert report["disarm_reason"] == "incomplete_install"


def test_an_unreadable_body_means_no_stand_down(monkeypatch):
    """No source to hash is a refusal, never a guess that it is probably fixed."""
    cls = fixed_in_tree_class()
    monkeypatch.setattr(backport, "fingerprint_of", lambda fn: None)
    state = backport.in_tree_state(cls)
    assert state["all_fixed"] is False and state["partial"] is False
    assert all(state["methods"][n]["state"] == "unreadable" for n in ORDER)

    monkeypatch.undo()
    monkeypatch.setattr(backport, "fingerprint_of", lambda fn: None)
    report = backport.install(cls)
    assert report.get("already_fixed_in_tree") is None, report
    assert report["installed"] is False, report
