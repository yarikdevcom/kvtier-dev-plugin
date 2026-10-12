r"""The SSM boot guard (`ssm_guard.py`): the tier refuses an SSM-recovery boot.

WHY THE TESTS ARE SHAPED LIKE THIS
----------------------------------
Two claims have to be proven, and only one of them is about the flag.

  1. THE SEMANTICS. Truthy refuses; ABSENT does not; present-and-false does not; an
     unreadable flag warns and installs. The absent direction is the one that bricks or
     does not brick every older engine, so it gets its own test and a name that says
     what it protects.
  2. THAT THE REFUSAL IS NOT THEATRE. An exception only protects anything if it reaches
     the operator, and this repo has a documented case of a gate that could not fail.
     The tier's own install hook runs inside `try/except Exception` by design
     (`backport._after_connector_created`: "nothing here may raise into
     `create_connector`"), so a refusal raised in there is a silent no-op. The
     propagation tests here drive the REAL fitted connector hook and assert both that
     the
     `ValueError` escapes AND that the swallowing layer's own error counters stayed at
     zero - proof it did not pass through the contained `except` on its way out. The AST
     test then pins the call site structurally, so a future "tidy-up" that moves the
     call
     inside that try fails here rather than in production.

Everything is stdlib-only and no vllm is imported: the fake engine objects are plain
objects carrying the real attribute names, the same posture as the rest of the suite.
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
import sys
import types

import pytest
import test_backport as TB  # noqa: F401 - the fork-class and connector fixtures

from kvtier_dev_plugin import backport, config, ssm_guard

GUARD_LOGGER = "kvtier_dev_plugin.ssm_guard"
ALL_FLAGS = list(ssm_guard.SSM_FLAG_ATTRS)


@pytest.fixture(autouse=True)
def reset_backport_state():
    """`backport._state` is process-wide; a leaked `fully_installed`/`armed` would make
    an escape test pass because a PREVIOUS test armed the tier. Same reset the boot
    tests use: the module's own `reset_state_for_tests()`, not a hand-copied field list.
    """
    saved = dict(backport._state)
    saved_verdicts = dict(backport._state["method_verdicts"])
    saved_switch = dict(backport._state["switch_verdicts"])
    backport.reset_state_for_tests()
    yield
    backport._state.clear()
    backport._state.update(saved)
    backport._state["method_verdicts"] = saved_verdicts
    backport._state["switch_verdicts"] = saved_switch


class CacheConfig:
    """A `CacheConfig` carrying exactly the flags a test names - and nothing else.

    Attributes are set per instance so an unnamed flag is genuinely ABSENT (a real fork
    without the feature), not `None` and not `False`. A name in `raises` is a property
    that explodes when read, which is the "our probe was confused" case.
    """

    def __init__(self, **flags):
        self._raises = set(flags.pop("raises", ()))
        for name, value in flags.items():
            setattr(self, name, value)

    def __getattr__(self, name):  # only consulted when nothing else provides it
        if name in self._raises:
            raise RuntimeError(f"the fork exploded while reading {name}")
        raise AttributeError(name)


def _config(cache) -> types.SimpleNamespace:
    return types.SimpleNamespace(cache_config=cache)


# --- 1. the semantics -------------------------------------------------------

@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_each_ssm_spelling_refuses_the_boot(flag):
    """A truthy flag of ANY of the three names costs the boot its tier."""
    with pytest.raises(ValueError) as exc:
        ssm_guard.boot_check(_config(CacheConfig(**{flag: True})))
    assert flag in str(exc.value), str(exc.value)


def test_a_fork_with_none_of_the_three_names_boots():
    """ABSENT IS NOT ON, THE LOAD-BEARING DIRECTION.

    An older fork has no ReplaySSM at all. If absence read as true, this plugin would
    refuse every engine that has no SSM - a guard that bricks the thing it is installed
    into. The verdict must be `clear`, not a warning and not a guess.
    """
    record = ssm_guard.probe(_config(CacheConfig(mamba_cache_mode="align")))
    assert record["verdict"] == ssm_guard.VERDICT_CLEAR, record
    assert record["attributes_absent"] == ALL_FLAGS
    assert not record["problems"]
    # ...and it installs, i.e. the boot step does not raise either.
    assert ssm_guard.boot_check(_config(CacheConfig()))["verdict"] \
        == ssm_guard.VERDICT_CLEAR


@pytest.mark.parametrize("flag", ALL_FLAGS)
def test_a_flag_that_is_present_and_false_is_not_ssm(flag):
    """Every normal boot: the field exists, nobody asked for it. No raise, no log."""
    record = ssm_guard.boot_check(_config(CacheConfig(**{flag: False})))
    assert record["verdict"] == ssm_guard.VERDICT_CLEAR
    assert record["attributes_seen"] == {flag: False}
    assert record["flag"] is None


def test_a_mixed_config_with_one_truthy_flag_refuses():
    """The refusal looks at all three, not just the first one that exists."""
    cache = CacheConfig(use_replayssm=False, use_kda_recoverssm=True)
    record = ssm_guard.probe(_config(cache))
    assert record["verdict"] == ssm_guard.VERDICT_REFUSED
    assert record["flag"] == "use_kda_recoverssm"
    with pytest.raises(ValueError):
        ssm_guard.boot_check(_config(cache))


@pytest.mark.parametrize("value", [1, "yes", [0]])
def test_a_truthy_value_of_any_kind_refuses(value):
    """The test is truthiness, not `is True`: a non-bool flag value is still on."""
    with pytest.raises(ValueError):
        ssm_guard.boot_check(_config(CacheConfig(use_replayssm=value)))


def test_a_probe_that_raises_warns_and_installs(caplog):
    """THE DELIBERATE FAIL-OPEN, PINNED SO IT IS NEVER "FIXED" INTO A REFUSAL.

    A flag we could not read is not a flag that is on. The engine's own
    `validate_mamba_cached_kernel` is the primary defence, and a plugin that refused to
    boot because its own probe was confused would be a new failure mode invented by the
    guard. The answer is one loud WARNING, and the warning must be loud enough that
    nobody reads it as a pass.
    """
    with caplog.at_level(logging.DEBUG, logger=GUARD_LOGGER):
        record = ssm_guard.boot_check(_config(CacheConfig(raises=("use_recoverssm",))))
    assert record["verdict"] == ssm_guard.VERDICT_INDETERMINATE, record
    assert any("use_recoverssm" in p for p in record["problems"]), record
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a fail-open that is silent is a fail-open nobody can find"
    text = " ".join(r.getMessage() for r in warnings)
    assert "NOT ESTABLISHED" in text
    assert ssm_guard.ENGINE_VALIDATOR in text


def test_a_config_we_cannot_reach_warns_and_installs(caplog):
    """No `vllm_config`, no `cache_config`, a `None` cache: three shapes, one answer.

    The hook tolerates a caller that hands it nothing (`_make_create_connector_hook`
    documents that), so "no config in reach" must never be a refusal either - but it
    is a
    DIFFERENT fact from "we read them all and they were clear", so it carries its own
    verdict and its own warning.
    """
    shapes = [None,
              types.SimpleNamespace(),
              types.SimpleNamespace(cache_config=None)]
    with caplog.at_level(logging.DEBUG, logger=GUARD_LOGGER):
        for shape in shapes:
            record = ssm_guard.boot_check(shape)
            assert record["verdict"] == ssm_guard.VERDICT_NO_CONFIG, (shape, record)
    assert len([r for r in caplog.records
                if r.levelno >= logging.WARNING]) == len(shapes)


def test_a_truthy_flag_outranks_an_unreadable_sibling():
    """Positive proof of the hazard wins over an unrelated probe failure."""
    cache = CacheConfig(use_recoverssm=True, raises=("use_replayssm",))
    record = ssm_guard.probe(_config(cache))
    assert record["verdict"] == ssm_guard.VERDICT_REFUSED
    assert record["flag"] == "use_recoverssm"
    assert record["problems"], "the failed read is still reported, not hidden"
    with pytest.raises(ValueError):
        ssm_guard.boot_check(_config(cache))


@pytest.mark.parametrize("env_value", ["1", "true", "yes"])
def test_an_inert_boot_warns_instead_of_refusing(monkeypatch, caplog, env_value):
    """`VLLM_E2_TIER_DEV_BOOT_INERT` is not an opt-out and is not treated as one.

    An inert boot installs and stops there - no guard, no arming (`boot_mode()`), so it
    spills no KV and there is no wrong claim to prevent. Refusing it would refuse a boot
    whose tier does nothing; installing it silently would hide the flag. Hence: do not
    raise, and say so at WARNING.
    """
    monkeypatch.setenv(config.ENV_BOOT_INERT, env_value)
    with caplog.at_level(logging.DEBUG, logger=GUARD_LOGGER):
        record = ssm_guard.boot_check(_config(CacheConfig(use_replayssm=True)))
    assert record["verdict"] == ssm_guard.VERDICT_REFUSED_INERT, record
    text = " ".join(r.getMessage() for r in caplog.records
                    if r.levelno >= logging.WARNING)
    assert "WOULD refuse" in text and config.ENV_BOOT_INERT in text


# --- 2. the refusal text ----------------------------------------------------

def test_the_message_carries_the_ordering_the_engine_guard_and_the_remedy():
    """Four things a reader needs, or the message is just an error.

    WHAT was found, WHY it is fatal (the boundary store vs the RecoverSSM commit, and
    that a wrong claim returns plausible output), WHO normally stops it - the sentence
    that tells a future reader DRAFT-0006 is in the tree - and the REMEDY.
    """
    with pytest.raises(ValueError) as exc:
        ssm_guard.boot_check(_config(CacheConfig(use_kda_recoverssm=True)))
    text = str(exc.value)
    assert "use_kda_recoverssm" in text
    assert "boundary" in text and "RecoverSSM commit" in text
    assert "plausible" in text, "the message must say WHY this is not a degrade case"
    assert ssm_guard.ENGINE_VALIDATOR in text
    assert ssm_guard.DRAFT_NAME in text, "this sentence is how a reader finds the draft"
    assert "Drop the flag" in text
    assert "no environment variable" in text, "the absence of an opt-out is a promise"


def test_no_environment_variable_can_bypass_the_guard(monkeypatch):
    """A sweep for the shape of every other switch in this package: no name opts out.

    Not a claim about a name we know of - a claim about the module, which must contain
    no
    `os.environ` read at all beyond the inert posture it borrows from `config`.
    """
    source = inspect.getsource(ssm_guard)
    assert "os.environ" not in source, source
    assert "getenv" not in source


# --- 3. PROPAGATION: the refusal reaches the operator -----------------------

def _boot_config(**flags) -> types.SimpleNamespace:
    """A `vllm_config` in the shape the factory is handed: cache_config + the field the
    sibling step reads, so a full boot through the hook stays realistic."""
    return types.SimpleNamespace(cache_config=CacheConfig(**flags),
                                 kv_transfer_config=None)


def test_the_refusal_escapes_the_connector_creation_hook(monkeypatch, caplog):
    """THE HEADLINE PROOF, driven through the real hook the plugin fits.

    `backport._after_connector_created` catches `Exception` on purpose. If the guard ran
    in there, this test would see a connector returned and an armed tier - the guard as
    decoration. Instead: the ValueError comes out of `create_connector`, the swallowing
    layer's own error counters stay at zero (so it never passed through that `except`),
    and nothing was installed or armed by the time it raised.
    """
    cls = TB._six_group_class()
    factory, connector_cls, built = TB._fake_vllm_modules(monkeypatch, cls)
    assert backport.install_connector_hook()["fitted"] is True

    with caplog.at_level(logging.DEBUG, logger=GUARD_LOGGER):
        with pytest.raises(ValueError) as exc:
            factory.create_connector(config=_boot_config(use_replayssm=True),
                                     role=types.SimpleNamespace(name="SCHEDULER"),
                                     kv_cache_config="KVC")

    assert "use_replayssm" in str(exc.value)
    assert isinstance(exc.value, ValueError), "the engine's veto is a ValueError too"
    # THE SWALLOWING LAYER DID NOT SEE IT: had the guard run inside the contained hook,
    # both of these would be 1 and the boot would have continued.
    assert backport._state["hook_errors"] == 0
    assert backport._state["hook_installs"] == 0, "nothing was installed"
    assert backport._state["fully_installed"] is False
    assert backport.is_armed() is False, "a refused boot arms no tier"
    assert len(built) == 1, "the connector was built and then refused, not half-built"
    assert isinstance(built[0], connector_cls)
    # The refusal is in the LOG too, not only in the traceback a pipe may wrap.
    assert any("SSM BOOT GUARD: REFUSED" in r.getMessage()
               for r in caplog.records if r.levelno >= logging.WARNING)


def test_a_worker_role_connector_with_the_flag_is_refused_too(monkeypatch):
    """The worker builds a tier connector too, and its refusal also kills the boot.

    Not a second rule: the same predicate. A worker-role refusal comes back through the
    executor as a worker error rather than as this exception, which is why the guard
    logs its own line before raising - the log line is the half that survives wrapping.
    """
    cls = TB._six_group_class()
    factory, _connector_cls, _built = TB._fake_vllm_modules(
        monkeypatch, cls, with_scheduler=False)
    backport.install_connector_hook()
    with pytest.raises(ValueError) as exc:
        factory.create_connector(config=_boot_config(use_replayssm=True),
                                 role=types.SimpleNamespace(name="WORKER"),
                                 kv_cache_config="KVC")
    assert "use_replayssm" in str(exc.value)
    assert backport._state["hook_errors"] == 0


def test_a_clean_boot_still_installs_and_arms_through_the_same_hook(monkeypatch):
    """The guard must not cost a normal boot anything: flags present-and-False, and the
    connector-creation boot runs exactly as it did before the guard existed."""
    cls = TB._six_group_class()
    factory, connector_cls, built = TB._fake_vllm_modules(monkeypatch, cls)
    backport.install_connector_hook()
    connector = factory.create_connector(
        config=_boot_config(use_replayssm=False, use_kda_recoverssm=False),
        role=types.SimpleNamespace(name="SCHEDULER"), kv_cache_config="KVC")
    assert isinstance(connector, connector_cls) and len(built) == 1
    assert backport._state["fully_installed"] is True
    assert backport._state["hook_installs"] == 1
    assert backport.is_armed() is True
    assert backport._state["hook_errors"] == 0


def test_a_fork_without_the_feature_boots_through_the_hook(monkeypatch):
    """The absent direction, at the integration layer: no flag attribute exists."""
    cls = TB._six_group_class()
    factory, _connector_cls, built = TB._fake_vllm_modules(monkeypatch, cls)
    backport.install_connector_hook()
    factory.create_connector(config=_boot_config(),
                             role=types.SimpleNamespace(name="SCHEDULER"),
                             kv_cache_config="KVC")
    assert len(built) == 1
    assert backport.is_armed() is True, "an SSM-less fork is the normal boot"


def test_a_foreign_connector_with_the_flag_is_not_this_plugins_business(monkeypatch):
    """No over-reach: with a foreign connector the tier never arms, so it spills
    nothing. Refusing somebody else's boot on a hazard this plugin cannot cause would
    make the plugin the reason a perfectly good engine will not start.
    """
    module = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.lmcache")

    class LMCacheConnectorV1:  # a foreign connector: no tier class in its MRO
        pass

    LMCacheConnectorV1.__module__ = module.__name__
    module.LMCacheConnectorV1 = LMCacheConnectorV1
    monkeypatch.setitem(sys.modules, module.__name__, module)
    factory_module = types.ModuleType(backport.CONNECTOR_FACTORY_MODULE)

    class KVConnectorFactory:
        @classmethod
        def create_connector(cls, config, role, kv_cache_config):  # noqa: ARG003
            return LMCacheConnectorV1()

    factory_module.KVConnectorFactory = KVConnectorFactory
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_FACTORY_MODULE,
                        factory_module)
    backport.install_connector_hook()
    connector = KVConnectorFactory.create_connector(
        config=_boot_config(use_replayssm=True),
        role=types.SimpleNamespace(name="SCHEDULER"), kv_cache_config=None)
    assert isinstance(connector, LMCacheConnectorV1)
    assert backport._state["hook_other_connector"] == 1
    assert backport._state["hook_errors"] == 0
    assert backport.is_armed() is False


# --- 4. the structural guarantee: nobody may move the call into the try ------

def _guard_call_sites():
    """Yield `(node, inside_try)` for every `boot_check(...)` call in the wrapper."""
    tree = ast.parse(inspect.getsource(backport._make_create_connector_hook))
    found = []

    def walk(node, inside_try):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Try):
                # The body and the finally-run block are the regions a bare
                # `except Exception` in THIS function would swallow a raise from; a
                # sibling statement before or after the try is not.
                for part in list(child.body) + list(child.finalbody or []):
                    walk(part, True)
                for handler in child.handlers:
                    walk(handler, inside_try)
                continue
            if isinstance(child, ast.Call):
                func = child.func
                if (isinstance(func, ast.Attribute) and func.attr == "boot_check"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "ssm_guard"):
                    found.append((child, inside_try))
            walk(child, inside_try)

    walk(tree, False)
    return found


def test_the_guard_is_called_outside_the_contained_try():
    """A green suite cannot see this corruption, so the AST has to.

    The whole value of a raising guard is that it is not called from inside
    `backport._after_connector_created`'s `except Exception`. That is a property of
    WHERE the call sits, and a future refactor that "tidies" the call into the
    contained region would leave every behavioural test above still able to pass - they
    would fail, but only if somebody remembered to re-run them against the moved call.
    This test fails on the move itself.
    """
    sites = _guard_call_sites()
    assert sites, "the connector-creation hook must call ssm_guard.boot_check at all"
    assert all(not inside for _node, inside in sites), \
        "ssm_guard.boot_check is called inside a try in the hook wrapper: a refusal " \
        "raised there is swallowed by the contained `except Exception` and the guard " \
        "becomes a gate that cannot fail"


def test_the_contained_hook_does_not_call_the_guard():
    """The reciprocal half: the guard must not also be reachable from inside the try.

    A second call site inside `_after_connector_created` would be a second, weaker,
    unraisable copy of the rule - and the one an operator would end up reading.
    """
    source = inspect.getsource(backport._after_connector_created)
    assert "ssm_guard" not in source


def test_the_guard_module_never_touches_the_engine_internals_it_refuses():
    """It reads three attribute names and nothing else: no torch, no vllm, no state.

    A boot-time check that imports the engine to answer a config question is a check
    that can fail for import reasons, which is the fail-open case wearing a costume.
    """
    source = inspect.getsource(ssm_guard)
    for banned in ("import torch", "from vllm", "import vllm", "disk_spill",
                   "disk_restore", "disk_store"):
        assert banned not in source, banned


# --- 5. THE BOOT LINE: `SSM[what]`, the verdict the log can actually show ------
#
# WHY THESE TESTS EXIST, AND WHY THEY LOOK LIKE THE LOAD_POLICY ONES
# ------------------------------------------------------------------
# The guard's SATISFIED case was one `logger.info`. Measured on the live engine
# 2026-10-22, plugin INFO on the connector-hook path does not reach
# `/tmp/vllm-serve.log` AT ALL: the hook's own INFO line (`connector hook saw a tier
# connector`) had 0 occurrences in a log carrying 2,775 plugin lines, and `SSM BOOT
# GUARD` had 0 too. So `SSM BOOT GUARD: clear` was un-readble in the one place an
# operator looks, and a boot where the guard ran and found no SSM was
# indistinguishable from a boot where the guard is not installed - the same class of
# defect this repo has now been bitten by twice (gates that could not be seen to work).
#
# The fix is the mechanism the boot line already uses for two other facts
# (`LOAD_POLICY[what:who]`, `MAMBA_ANCHOR[patched]`): a bracketed token on the WARNING
# line that DOES land, rendering the record the step wrote THIS boot. Hence these tests
# mirror `test_backport.py`'s LOAD_POLICY boot-line tests in structure: drive the real
# hook, read the line, and - the part that matters - prove the token follows the stored
# record rather than a constant. A token that always printed `clear` would satisfy every
# other kind of check and tell an operator the exact opposite of the truth.

def _guard_config(cache=None):
    """A `vllm_config` in the factory's shape, with or WITHOUT a `cache_config`.

    `cache=None` is the no-config shape the hook tolerates (a foreign caller, an older
    test), and it is how the `no-config` verdict is reached through the real path.
    """
    return types.SimpleNamespace(cache_config=cache, kv_transfer_config=None)


def _boot_with_guard(monkeypatch, config=None, **flags):
    """Boot through the REAL connector hook and report what the boot line said.

    Returns `(the guard's record from the boot payload, the boot line)`. `(None, line)`
    is the `not_run` answer: a boot that never handed the guard a tier connector.
    """
    cls = TB._six_group_class()
    factory, connector_cls, _built = TB._fake_vllm_modules(monkeypatch, cls)
    backport.install_connector_hook()
    if config is None:
        config = _boot_config(**flags)
    connector = factory.create_connector(
        config=config, role=types.SimpleNamespace(name="SCHEDULER"),
        kv_cache_config="KVC")
    assert isinstance(connector, connector_cls), "the engine still got its connector"
    boot = backport.section()["boot"]
    return boot.get("ssm_guard"), boot["line"]


def test_a_clean_boot_prints_ssm_clear_on_the_boot_line(monkeypatch):
    """THE FINDING THIS FIX ANSWERS: the healthy boot must be READABLE, not just safe.

    Before the token, a boot with no SSM flag logged nothing an operator could see and
    the log could not answer "did the guard run?". Now the one line the boot writes
    carries `SSM[clear]`, and the record behind it is in the payload.
    """
    record, line = _boot_with_guard(monkeypatch, use_replayssm=False,
                                    use_kda_recoverssm=False)
    assert record["verdict"] == ssm_guard.VERDICT_CLEAR, record
    assert "SSM[clear]" in line, line
    assert "SSM[not_run]" not in line


def test_a_boot_that_never_ran_the_guard_says_not_run_and_still_prints(monkeypatch):
    """A boot with no tier connector renders `not_run`, never the guard's own hope.

    The shape of `test_the_boot_reports_the_policy_without_the_step_claiming_anything`
    exactly: `boot_from_connector` is also called directly (the harness in
    `tests/test_auto_boot.py`), and a line that rendered `SSM[clear]` for a boot whose
    guard never ran would be the
    false provenance this token exists to avoid.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)
    boot = backport.boot_from_connector(cls, report, manager)
    assert boot["ssm_guard"] is None, "the payload key is present-but-null"
    assert "SSM[not_run]" in boot["line"], boot["line"]


def test_a_foreign_connector_boot_says_not_run(monkeypatch):
    """The other never-ran path: the guard's predicate is OUR connector, so a foreign
    one produces no record and the note line says so rather than reading as a pass."""
    module = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.lmcache")

    class LMCacheConnectorV1:
        pass

    LMCacheConnectorV1.__module__ = module.__name__
    module.LMCacheConnectorV1 = LMCacheConnectorV1
    monkeypatch.setitem(sys.modules, module.__name__, module)
    factory_module = types.ModuleType(backport.CONNECTOR_FACTORY_MODULE)

    class KVConnectorFactory:
        @classmethod
        def create_connector(cls, config, role, kv_cache_config):  # noqa: ARG003
            return LMCacheConnectorV1()

    factory_module.KVConnectorFactory = KVConnectorFactory
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_FACTORY_MODULE, factory_module)
    backport.install_connector_hook()
    KVConnectorFactory.create_connector(config=_boot_config(use_replayssm=True),
                                        role=types.SimpleNamespace(name="SCHEDULER"),
                                        kv_cache_config=None)
    line = backport.section()["boot"]["line"]
    assert "SSM[not_run]" in line, line
    assert "SSM[clear]" not in line


def test_a_refused_inert_boot_prints_ssm_refused_inert_on_the_boot_line(monkeypatch):
    """The refusal the guard does NOT raise, on the line that survives: `refused-inert`.

    Reached through the real hook with `VLLM_E2_TIER_DEV_BOOT_INERT` set, so this is a
    boot that installs and stops there - the verdict the operator must be able to see
    without reading a warning in a wall of engine output.
    """
    monkeypatch.setenv(config.ENV_BOOT_INERT, "1")
    record, line = _boot_with_guard(monkeypatch, use_replayssm=True)
    assert record["verdict"] == ssm_guard.VERDICT_REFUSED_INERT, record
    assert "SSM[refused-inert]" in line, line


def test_an_unreadable_flag_prints_ssm_indeterminate_on_the_boot_line(monkeypatch):
    """The fail-open is now visible. A fail-open nobody can see is an unlogged risk.

    A flag whose read raised leaves the boot installing, and the token says
    `indeterminate` - never `clear`, which would be a claim the guard did not make.
    """
    record, line = _boot_with_guard(
        monkeypatch, config=_guard_config(CacheConfig(raises=("use_recoverssm",))))
    assert record["verdict"] == ssm_guard.VERDICT_INDETERMINATE, record
    assert "SSM[indeterminate]" in line, line
    assert "SSM[clear]" not in line


def test_a_boot_with_no_cache_config_prints_ssm_no_config(monkeypatch):
    """No `cache_config` in reach is its own answer, on the line and in the payload."""
    record, line = _boot_with_guard(monkeypatch, config=_guard_config(None))
    assert record["verdict"] == ssm_guard.VERDICT_NO_CONFIG, record
    assert "SSM[no-config]" in line, line
    assert "SSM[clear]" not in line


def test_a_refused_record_reaches_the_boot_line_when_a_boot_runs(monkeypatch):
    """`SSM[refused]` renders from the guard's OWN refused record.

    A refusing boot normally dies before any line (the ValueError escapes the hook -
    proven above), which is the point of the guard. But the record is what the payload
    and any line built from it carry, and the worker-role refusal reaches the operator
    through the executor as a worker error rather than as this exception, so the
    verdict must be renderable from the record and not only from the traceback. The
    record here is `probe()`'s own, never a hand-written dict.
    """
    record = ssm_guard.probe(_boot_config(use_recoverssm=True))
    assert record["verdict"] == ssm_guard.VERDICT_REFUSED
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)
    boot = backport.boot_from_connector(cls, report, manager, ssm_record=record)
    assert boot["ssm_guard"]["flag"] == "use_recoverssm", boot["ssm_guard"]
    assert "SSM[refused]" in boot["line"], boot["line"]


@pytest.mark.parametrize("verdict", list(ssm_guard.VERDICTS))
def test_the_token_follows_the_record_and_never_a_constant(verdict):
    """THE TEST THAT KILLS A CONSTANT TOKEN, for every verdict in the vocabulary.

    Writes a chosen verdict into an otherwise realistic record and asserts the boot line
    carries THAT token and no other - so a token that always printed `clear`, or read
    the environment at print time, fails here for at least one of the five. The token is
    produced through the same read-back path the boot line uses (`ssm_guard_token`, then
    `boot_line`), not by calling the renderer's private helper.
    """
    record = ssm_guard._empty()
    record["verdict"] = verdict
    assert backport.ssm_guard_token({"ssm_guard": record}) == f"SSM[{verdict}]"
    line = backport.boot_line({"mode": "auto", "ssm_guard": record})
    assert f"SSM[{verdict}]" in line, line
    others = [v for v in ssm_guard.VERDICTS if v != verdict]
    for other in others:
        assert f"SSM[{other}]" not in line, (verdict, other, line)


def test_an_unnameable_verdict_renders_indeterminate_never_clear():
    """A record whose verdict is outside the closed vocabulary is not a pass.

    The pessimistic rule `lookup_ledger_token` states outright: the honest reading of a
    verdict this module cannot name is "it did not work". `SSM[clear]` over a corrupt
    record is the one wrong answer this token must never give.
    """
    for broken in ({"verdict": "patched"}, {"verdict": None}, {}, "not-a-record",
                   None, []):
        token = backport.ssm_guard_token({"ssm_guard": broken})
        assert token in ("SSM[not_run]", "SSM[indeterminate]"), (broken, token)
    assert backport.ssm_guard_token({}) == "SSM[not_run]"


def test_the_ssm_token_is_bracketed_and_never_a_candidate_tier_match():
    """The token shape rule every sibling token pins, for the same reason.

    The restart verifier in the llm-api repo pulls its PASS criterion with
    `grep -oE "tier=[A-Za-z_]+" | tail -1`, so a new token must carry no `=` and must
    not become the LAST `tier=`-shaped match on the line. `SSM[clear]` satisfies both: a
    bare `tier=ARMED` is still the last match with the token on the line.
    """
    line = backport.boot_line({"mode": "auto", "tier": True,
                               "ssm_guard": {"verdict": "clear"}})
    assert "SSM[clear]" in line
    assert "SSM=" not in line
    assert re.findall(r"tier=[A-Za-z_]+", line)[-1] == "tier=ARMED", line


def test_the_record_is_stored_in_the_boot_dict_not_reprinted_from_a_constant():
    """The provenance claim, tested: the token reads the boot's record, not the flag.

    Two moves make this a test and not an assertion. (1) Boot with no SSM flag and
    OVERWRITE the stored record with a different verdict: the next line rendered from
    that boot dict follows the record, proving print time reads the record. (2) Boot a
    config that DOES carry a flag the guard refuses - the guard's own probe path - and
    confirm the record's `flag` field is the one it read, which a constant could not
    produce.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)

    clean = ssm_guard.probe(_boot_config())
    boot = backport.boot_from_connector(cls, report, manager, ssm_record=clean)
    assert "SSM[clear]" in boot["line"]
    boot["ssm_guard"]["verdict"] = ssm_guard.VERDICT_INDETERMINATE
    tampered = backport.boot_line(boot)
    assert "SSM[indeterminate]" in tampered, tampered
    assert "SSM[clear]" not in tampered, "the token followed the record, not a constant"

    read = ssm_guard.probe(_boot_config(use_kda_recoverssm=True))
    boot2 = backport.boot_from_connector(cls, report, manager, ssm_record=read)
    assert boot2["ssm_guard"]["flag"] == "use_kda_recoverssm"
    assert "SSM[refused]" in backport.boot_line(boot2)


def test_the_token_appears_once_and_the_line_stays_one_bounded_line():
    """One token per boot line, and the line's own bounds still hold.

    `tests/test_auto_boot.py` pins that every `=`-shaped field name appears exactly
    once; this pins the same for the token, on a real boot line, and re-checks the
    no-newline / length bounds that keep the line greppable in an engine log.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)
    boot = backport.boot_from_connector(cls, report, manager,
                                        ssm_record=ssm_guard.probe(_boot_config()))
    line = boot["line"]
    assert line.count("SSM[") == 1, line
    assert "\n" not in line and len(line) < 2000, line


def test_a_worker_role_boot_line_carries_the_guard_verdict_too(monkeypatch):
    """The worker builds a tier connector and DOES run the guard, so its note line must
    carry the verdict - `not_run` there would be a line that lies by omission, which is
    the exact defect this token exists to remove.
    """
    cls = TB._six_group_class()
    factory, _connector_cls, built = TB._fake_vllm_modules(
        monkeypatch, cls, with_scheduler=False)
    backport.install_connector_hook()
    factory.create_connector(config=_boot_config(use_replayssm=False),
                             role=types.SimpleNamespace(name="WORKER"),
                             kv_cache_config="KVC")
    assert len(built) == 1, "the worker still got its connector"
    line = backport.section()["boot"]["line"]
    assert "SSM[clear]" in line, line
    assert "SSM[not_run]" not in line
