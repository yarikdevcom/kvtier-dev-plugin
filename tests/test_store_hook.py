"""The runtime store-policy hook: the veto, the guards, and the kill switch.

Nothing here imports vllm. The stub below is built by `exec`ing a source string
into a real module object so that its methods' `__globals__` is that module's
dict - which is what `store_hook._admissions()` resolves `_StoreAdmission` from,
exactly as the real call sites resolve it. The shape (five members, one block
argument, an enum return, two call sites that map a verdict to a counter) is
taken from the read-only fork at 10b8e07:

  manager.py:104-109   `_StoreAdmission`
  manager.py:125-132   `BoundaryStoreStats`
  manager.py:683-701   the boundary hand-off loop: verdict -> counter, and the
                       `if num_free <= 0` drop that NO verdict can produce
  manager.py:862-868   the eager per-group loop: any non-READY advances a cursor
  manager.py:932-946   `_classify_store_candidate` itself
"""

from __future__ import annotations

import ast
import sys
import time
import types
from pathlib import Path

import pytest

from kvtier_dev_plugin import config, engine_side, snapshot, store_hook
from kvtier_dev_plugin.policy_holder import PolicyHolder

PKG = Path(snapshot.__file__).resolve().parent

# --- the fork's scheduler side, faithfully shaped ---------------------------

FAKE_MANAGER_SOURCE = '''
"""Shaped after vllm/v1/simple_kv_offload/manager.py (a read-only fork)."""
from enum import Enum, auto


class _StoreAdmission(Enum):
    READY = auto()
    NULL_BLOCK = auto()
    NOT_HASHED = auto()
    IN_FLIGHT = auto()
    ALREADY_CACHED = auto()


class BoundaryStoreStats:
    """The eight counters of manager.py:125-132, as plain attributes.

    A plain class rather than a dataclass so the stub needs nothing from
    `sys.modules` at definition time; `vars()` of an instance is what the
    snapshot reads, and it looks the same.
    """

    def __init__(self):
        self.published = 0
        self.stored = 0
        self.dropped_cpu_full = 0
        self.dropped_request_gone = 0
        self.skipped_already_cached = 0
        self.skipped_in_flight = 0
        self.dropped_null_block = 0
        self.dropped_not_hashed = 0


class Pool:
    """BlockPool: the only thing the hook reads is the maintained free count."""

    def __init__(self, free):
        self.num_gpu_blocks = 4096
        self._free = free

    def get_num_free_blocks(self):
        return self._free


_ANY_HASH = object()
"""Sentinel: `Block(3, block_hash=None)` has to mean genuinely unhashed, so the
default hash cannot be spelled with None."""


class Block:
    """KVCacheBlock: block_id / block_hash / is_null, manager.py:933-938."""

    def __init__(self, block_id, block_hash=_ANY_HASH, is_null=False):
        self.block_id = block_id
        self.block_hash = (7000 + block_id if block_hash is _ANY_HASH
                           else block_hash)
        self.is_null = is_null


class SimpleCPUOffloadScheduler:
    def __init__(self, *, cpu_free=100, in_flight=(), cached=(), lazy=False):
        self.cpu_block_pool = Pool(cpu_free)
        self._in_flight_store_gpu_blocks = set(in_flight)
        self._cpu_cached_hashes = set(cached)
        self._lazy_mode = lazy
        self._target_free = 128
        self.block_size = 256
        self.hash_block_size = 64
        self.num_cpu_blocks = 4096
        self.boundary_store_stats = BoundaryStoreStats()

    def _classify_store_candidate(self, gpu_block):
        # The verbatim shape of manager.py:932-946.
        if gpu_block.is_null:
            return _StoreAdmission.NULL_BLOCK
        if gpu_block.block_hash is None:
            return _StoreAdmission.NOT_HASHED
        if gpu_block.block_id in self._in_flight_store_gpu_blocks:
            return _StoreAdmission.IN_FLIGHT
        if gpu_block.block_hash in self._cpu_cached_hashes:
            return _StoreAdmission.ALREADY_CACHED
        return _StoreAdmission.READY

    def get_boundary_store_stats(self):
        # manager.py:948 returns a `replace(...)` COPY; copying the fields is
        # the same promise, and it keeps the snapshot's tier section honest.
        copy = BoundaryStoreStats()
        for name in vars(self.boundary_store_stats):
            setattr(copy, name, getattr(self.boundary_store_stats, name))
        return copy

    def has_pending_stores(self):
        return False
'''

# A fork that renamed the enum: the hook must then dare to veto nothing.
RENAMED_SOURCE = FAKE_MANAGER_SOURCE.replace("_StoreAdmission",
                                             "_RenamedAdmission")


def build_module(source, name=store_hook.TARGET_MODULE):
    """A real module object, so methods' `__globals__` is its `__dict__`."""
    module = types.ModuleType(name)
    module.__file__ = "<fake simple_kv_offload.manager>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module


class FakeConnector:
    def __init__(self, manager) -> None:
        self.scheduler_manager = manager


class FakeScheduler:
    def __init__(self, manager) -> None:
        self.connector = FakeConnector(manager)


class FakeEngine:
    def __init__(self, manager) -> None:
        self.scheduler = FakeScheduler(manager)


def boundary_step_like_fork(module, manager, blocks, *, cpu_free=None):
    """manager.py:683-701: the only thing that turns a verdict into a counter.

    Returns (stats, gpu_block_ids the tier would copy). Kept in the test (not in
    the stub) so the mapping under test is the fork's, not ours.
    """
    admission = module._StoreAdmission
    stats = manager.boundary_store_stats
    num_free = (manager.cpu_block_pool.get_num_free_blocks()
                if cpu_free is None else cpu_free)
    stored = []
    for gpu_block in blocks:
        stats.published += 1
        verdict = manager._classify_store_candidate(gpu_block)
        if verdict is admission.NULL_BLOCK:
            stats.dropped_null_block += 1
            continue
        if verdict is admission.NOT_HASHED:
            stats.dropped_not_hashed += 1
            continue
        if verdict is admission.IN_FLIGHT:
            stats.skipped_in_flight += 1
            continue
        if verdict is admission.ALREADY_CACHED:
            stats.skipped_already_cached += 1
            continue
        if num_free <= 0:                       # manager.py:698: no verdict can
            stats.dropped_cpu_full += 1         # produce this one. See below.
            continue
        num_free -= 1
        stats.stored += 1
        stored.append(gpu_block.block_id)
    return stats, stored


# --- policy sources ---------------------------------------------------------

def policy_source(body, *, with_decide_store=True, version=1):
    """A policy file: `decide` always (a live deployment needs it), and
    optionally a `decide_store` whose body is `body`."""
    text = f'"""fixture policy"""\nVERSION = {version}\n\n\n' \
           f'def decide(view):\n    return {{"v": {version}}}\n'
    if with_decide_store:
        text += f'\n\ndef decide_store(ctx):\n    return {body}\n'
    return text


VETO = policy_source('"veto"')
ALLOW = policy_source('"allow"')
RAISES = '"""fixture policy"""\nVERSION = 1\n\n\n' \
         'def decide(view):\n    return 1\n\n\n' \
         'def decide_store(ctx):\n    raise RuntimeError("the policy is broken")\n'


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def hook_state():
    """Counters and the enum memo are process-wide; keep tests order-independent.

    Reset to zero at entry (another file's `engine_side.register()` test can
    install or refuse here exactly as this file can), restored at exit.
    """
    saved_state = dict(store_hook._state)
    saved_memo = list(store_hook._admission_cache)
    store_hook.reset_counters_for_tests()
    yield store_hook._state
    store_hook._state.update(saved_state)
    store_hook._admission_cache[:] = saved_memo


@pytest.fixture
def fake_manager_module(hook_state):
    """The manager module registered as if the connector had imported it."""
    module = build_module(FAKE_MANAGER_SOURCE)
    sys.modules[store_hook.TARGET_MODULE] = module
    yield module
    sys.modules.pop(store_hook.TARGET_MODULE, None)


@pytest.fixture
def policy_file(tmp_path):
    path = tmp_path / "policy.py"
    path.write_text(ALLOW, encoding="utf-8")
    return path


@pytest.fixture
def holder(policy_file):
    live = PolicyHolder(str(policy_file))
    assert live.ensure_loaded()["ok"] is True
    return live


@pytest.fixture
def patched(fake_manager_module, policy_file, holder):
    """A live manager whose class carries the hook, plus the pieces to inspect."""
    manager = fake_manager_module.SimpleCPUOffloadScheduler()
    report = store_hook.install(fake_manager_module.SimpleCPUOffloadScheduler,
                                lambda: holder)
    assert report["installed"] is True
    return types.SimpleNamespace(module=fake_manager_module, manager=manager,
                                 manager_cls=(
                                     fake_manager_module.SimpleCPUOffloadScheduler),
                                 holder=holder, policy_file=policy_file,
                                 engine=FakeEngine(manager), report=report)


# --- the stub is faithful ---------------------------------------------------

def test_the_stub_classifies_exactly_like_the_fork(fake_manager_module):
    module = fake_manager_module
    sched = module.SimpleCPUOffloadScheduler(in_flight={9}, cached={555})
    ready = module.Block(1)
    assert sched._classify_store_candidate(ready) is module._StoreAdmission.READY
    null = module.Block(2)
    null.is_null = True
    assert sched._classify_store_candidate(null) is \
        module._StoreAdmission.NULL_BLOCK
    unhashed = module.Block(3, block_hash=None)
    assert sched._classify_store_candidate(unhashed) is \
        module._StoreAdmission.NOT_HASHED
    assert sched._classify_store_candidate(module.Block(9)) is \
        module._StoreAdmission.IN_FLIGHT
    assert sched._classify_store_candidate(module.Block(10, block_hash=555)) is \
        module._StoreAdmission.ALREADY_CACHED


def test_an_unpatched_tier_stores_what_the_engine_allows(fake_manager_module):
    module = fake_manager_module
    sched = module.SimpleCPUOffloadScheduler()
    stats, stored = boundary_step_like_fork(module, sched,
                                            [module.Block(i) for i in range(3)])
    assert stored == [0, 1, 2] and stats.stored == 3


# --- the veto ---------------------------------------------------------------

def test_a_veto_stops_a_store_the_engine_would_have_made(patched):
    patched.policy_file.write_text(VETO, encoding="utf-8")
    assert patched.holder.reload()["ok"] is True

    module, manager = patched.module, patched.manager
    verdict = manager._classify_store_candidate(module.Block(4))
    assert verdict is module._StoreAdmission.ALREADY_CACHED
    assert store_hook.patch_report(manager)["vetoes"] == 1


def test_a_veto_is_recorded_under_the_only_decline_reason_the_fork_knows(patched):
    """The honest version of "the original path records the drop reason".

    `_StoreAdmission` has no member meaning "the policy said no", so a veto is
    reported as ALREADY_CACHED and the boundary loop charges it to
    `skipped_already_cached` - a SKIP, never a `dropped_*`. The true count is
    `store_hook.vetoes`, and the drop-on-full branch is unreachable from here.
    """
    patched.policy_file.write_text(VETO, encoding="utf-8")
    patched.holder.reload()
    module, manager = patched.module, patched.manager

    stats, stored = boundary_step_like_fork(module, manager,
                                            [module.Block(i) for i in range(4)])
    assert stored == [] and stats.stored == 0
    assert stats.skipped_already_cached == 4
    assert stats.dropped_cpu_full == 0          # NOT reachable via a veto
    assert stats.published == 4
    assert store_hook.patch_report(manager)["vetoes"] == 4


def test_an_allow_verdict_changes_nothing(patched):
    module, manager = patched.module, patched.manager
    stats, stored = boundary_step_like_fork(module, manager,
                                            [module.Block(i) for i in range(4)])
    assert stored == [0, 1, 2, 3] and stats.stored == 4
    report = store_hook.patch_report(manager)
    assert report["vetoes"] == 0 and report["policy_calls"] == 4


def test_the_policy_is_never_asked_about_a_block_the_engine_declines(patched):
    """Both halves of that rule matter: it keeps the eight counters honest (a
    veto could otherwise move a `dropped_not_hashed` into
    `skipped_already_cached`) and it keeps the per-call path off the policy for
    the majority of candidates on a busy box."""
    module, manager = patched.module, patched.manager
    patched.policy_file.write_text(VETO, encoding="utf-8")
    patched.holder.reload()
    manager._in_flight_store_gpu_blocks = {7}
    null = module.Block(1)
    null.is_null = True
    blocks = [null, module.Block(2, block_hash=None), module.Block(7),
              module.Block(3)]              # the only READY candidate
    verdicts = [manager._classify_store_candidate(b) for b in blocks]

    assert verdicts == [module._StoreAdmission.NULL_BLOCK,
                        module._StoreAdmission.NOT_HASHED,
                        module._StoreAdmission.IN_FLIGHT,
                        module._StoreAdmission.ALREADY_CACHED]
    report = store_hook.patch_report(manager)
    assert report["calls"] == 4 and report["policy_calls"] == 1
    assert report["vetoes"] == 1


# --- fail-open --------------------------------------------------------------

def test_a_policy_that_raises_leaves_the_verdict_and_counts_a_fallback(
        patched, reset_snapshot_state):
    module, manager = patched.module, patched.manager
    patched.policy_file.write_text(RAISES, encoding="utf-8")
    assert patched.holder.reload()["ok"] is True

    for i in range(3):
        assert manager._classify_store_candidate(module.Block(i)) is \
            module._StoreAdmission.READY

    report = store_hook.patch_report(manager)
    assert report["fallbacks"] == 3 and report["vetoes"] == 0
    assert reset_snapshot_state["errors"] == 3   # counted through snapshot._error


def test_a_holder_that_raises_cannot_reach_the_scheduler(fake_manager_module,
                                                         hook_state):
    module = fake_manager_module
    manager = module.SimpleCPUOffloadScheduler()

    def explode():
        raise RuntimeError("no holder for you")

    assert store_hook.install(module.SimpleCPUOffloadScheduler, explode)[
        "installed"] is True
    for i in range(2):
        assert manager._classify_store_candidate(module.Block(i)) is \
            module._StoreAdmission.READY
    # 3 = the one cold load a fresh install attempts, plus one per candidate.
    assert hook_state["fallbacks"] == 3


def test_a_base_exception_from_the_policy_is_not_swallowed(patched):
    """Package doctrine: a diagnostic swallows Exception, never BaseException."""
    module, manager = patched.module, patched.manager
    patched.policy_file.write_text(
        'VERSION = 1\n\n\ndef decide(view):\n    return 1\n\n\n'
        'def decide_store(ctx):\n    raise KeyboardInterrupt("ctrl-c")\n',
        encoding="utf-8")
    assert patched.holder.reload()["ok"] is True
    with pytest.raises(KeyboardInterrupt):
        manager._classify_store_candidate(module.Block(1))


@pytest.mark.parametrize("body,counted", [
    ('"veto."', True),                      # a typo is not a veto
    ('"VETO-OFF"', True),
    ('"no"', True),                         # not in either word set
    ("True", True),                        # never a veto, and visible
    ("False", True),
    ("1", True),
    ("[]", True),
    ('{"veto": True}', True),
    ('"allow"', False),
    ('"store"', False),
    ('""', False),
    ("None", False),
])
def test_only_a_listed_veto_word_vetoes_everything_else_is_the_engines_verdict(
        fake_manager_module, holder, policy_file, hook_state, body, counted):
    manager = fake_manager_module.SimpleCPUOffloadScheduler()
    assert store_hook.install(fake_manager_module.SimpleCPUOffloadScheduler,
                              lambda: holder)["installed"] is True
    policy_file.write_text(policy_source(body), encoding="utf-8")
    assert holder.reload()["ok"] is True

    for i in range(2):
        assert manager._classify_store_candidate(fake_manager_module.Block(i)) \
            is fake_manager_module._StoreAdmission.READY

    report = store_hook.patch_report(manager)
    assert report["vetoes"] == 0            # the engine's own verdict, always
    assert report["unrecognized"] == (2 if counted else 0)


@pytest.mark.parametrize("body,expected", [('"veto"', 1), ('"Veto"', 1),
                                           ('" BLOCK "', 1), ('"deny"', 1),
                                           ('"no-store"', 1)])
def test_veto_words_are_whole_words_case_and_space_insensitive(
        fake_manager_module, holder, policy_file, hook_state, body, expected):
    manager = fake_manager_module.SimpleCPUOffloadScheduler()
    assert store_hook.install(fake_manager_module.SimpleCPUOffloadScheduler,
                              lambda: holder)["installed"] is True
    policy_file.write_text(policy_source(body), encoding="utf-8")
    holder.reload()
    manager._classify_store_candidate(fake_manager_module.Block(1))
    assert store_hook.patch_report(manager)["vetoes"] == expected


def test_a_renamed_enum_means_the_hook_vetoes_nothing_at_all(hook_state):
    """An unknown return type is the one thing we must not guess at."""
    module = build_module(RENAMED_SOURCE)
    sys.modules[store_hook.TARGET_MODULE] = module
    try:
        manager = module.SimpleCPUOffloadScheduler()
        assert store_hook.install(module.SimpleCPUOffloadScheduler,
                                  lambda: _veto_holder())[
            "installed"] is True
        for i in range(3):
            assert manager._classify_store_candidate(module.Block(i)) is \
                module._RenamedAdmission.READY
        report = store_hook.patch_report(manager)
        assert report["ready_resolved"] is False
        assert report["vetoes"] == 0 and report["policy_calls"] == 0
    finally:
        sys.modules.pop(store_hook.TARGET_MODULE, None)


def _veto_holder():
    raise AssertionError("must not be consulted when the enum is unresolved")


# --- installation discipline ------------------------------------------------

def test_double_install_is_a_no_op_and_never_grows_a_wrapper_chain(
        fake_manager_module, holder, hook_state):
    cls = fake_manager_module.SimpleCPUOffloadScheduler
    true_original = cls.__dict__[store_hook.TARGET_METHOD]
    first = store_hook.install(cls, lambda: holder)
    second = store_hook.install(cls, lambda: holder)
    third = store_hook.install(cls, lambda: holder)

    assert first["installed"] is True and first["replaced"] is False
    assert second["replaced"] is True and third["replaced"] is True
    assert hook_state["installs"] == 1
    assert hook_state["reinstalls"] == 2
    # The wrapper installed second binds the FORK's function, not the wrapper
    # before it: a reload loop cannot nest wrappers deeper and deeper.
    installed = cls.__dict__[store_hook.TARGET_METHOD]
    assert getattr(installed, store_hook.ORIGINAL_ATTR) is true_original
    assert getattr(installed, store_hook.OWNER_ATTR) == store_hook.OWNER_TAG

    manager = cls()
    stats, stored = boundary_step_like_fork(fake_manager_module, manager,
                                            [fake_manager_module.Block(1)])
    assert stored == [1]                       # still just one pass-through


def test_a_foreign_owner_of_the_target_name_is_refused(fake_manager_module,
                                                       hook_state):
    """A name we did not install and that does not come from the fork's module
    is not ours to wrap. `attach_methods` can refuse any pre-existing name;
    here the name exists by definition, so the defining module is the tell."""
    cls = fake_manager_module.SimpleCPUOffloadScheduler

    def somebody_elses_hook(self, gpu_block):  # noqa: ARG001 - it is never run
        return "not yours"

    cls._classify_store_candidate = somebody_elses_hook
    report = store_hook.install(cls, must_not_be_consulted)
    assert report["installed"] is False
    assert "refusing to overwrite" in report["reason"]
    assert cls.__dict__[store_hook.TARGET_METHOD] is somebody_elses_hook
    assert hook_state["skips"] == 1


def test_a_marker_from_another_plugin_is_refused(fake_manager_module,
                                                hook_state):
    cls = fake_manager_module.SimpleCPUOffloadScheduler
    original = cls.__dict__[store_hook.TARGET_METHOD]

    def their_wrapper(self, gpu_block):
        return original(self, gpu_block)

    their_wrapper.__module__ = store_hook.TARGET_MODULE   # patched in place
    setattr(their_wrapper, store_hook.OWNER_ATTR, "some-other-plugin/1")
    cls._classify_store_candidate = their_wrapper

    report = store_hook.install(cls, must_not_be_consulted)
    assert report["installed"] is False
    assert "another plugin's owner marker" in report["reason"]
    assert cls.__dict__[store_hook.TARGET_METHOD] is their_wrapper


def must_not_be_consulted():
    """A holder getter that proves the install refused before touching it."""
    raise AssertionError("a refused install must not reach the holder")


def test_a_target_that_does_not_exist_is_refused_not_invented(hook_state):
    class Other:
        pass

    report = store_hook.install(Other, must_not_be_consulted)
    assert report["installed"] is False
    assert "the fork renamed it" in report["reason"]
    assert not hasattr(Other, store_hook.TARGET_METHOD)


def test_a_non_callable_target_is_refused(hook_state):
    class Other:
        _classify_store_candidate = 17

    report = store_hook.install(Other, must_not_be_consulted)
    assert report["installed"] is False
    assert "not callable" in report["reason"]


def test_no_module_and_no_import_of_vllm(hook_state):
    """`general_plugins` load in worker and arg-parsing processes: finding
    nothing must be a reported fact, and must never pull the tree in."""
    saved = sys.modules.pop(store_hook.TARGET_MODULE, None)
    try:
        assert "vllm" not in sys.modules
        assert store_hook.manager_class() is None
        report = store_hook.install(None, must_not_be_consulted)
        assert report["installed"] is False
        assert "absent" in report["reason"]
        assert "vllm" not in sys.modules        # we did not import it ourselves
    finally:
        if saved is not None:
            sys.modules[store_hook.TARGET_MODULE] = saved


@pytest.mark.parametrize("module_file", ["store_hook.py", "snapshot.py",
                                         "policy.py", "engine_side.py"])
def test_the_store_hook_path_pulls_in_nothing_heavy(module_file):
    """Same AST rule as test_snapshot.py, extended to the new module: an import
    that only runs conditionally still shows up here."""
    tree = ast.parse((PKG / module_file).read_text(encoding="utf-8"))
    heavy = {"torch", "vllm", "cuda", "numpy", "triton", "numba", "cupy"}
    pulled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pulled.add(node.module.split(".")[0])
    assert not (pulled & heavy), (module_file, sorted(pulled & heavy))


# --- where the patch actually lands -----------------------------------------

def test_ensure_installed_uses_the_live_managers_own_class(fake_manager_module,
                                                           holder, hook_state):
    manager = fake_manager_module.SimpleCPUOffloadScheduler()
    engine = FakeEngine(manager)
    report = store_hook.ensure_installed(engine, lambda: holder)
    assert report["installed"] is True
    assert report["class"] == "SimpleCPUOffloadScheduler"
    assert store_hook.patch_report(manager)["live_manager_patched"] is True


def test_an_engine_without_a_scheduler_still_finds_the_imported_module(
        fake_manager_module, holder, hook_state):
    class NoScheduler:
        pass

    report = store_hook.ensure_installed(NoScheduler(), lambda: holder)
    assert report["installed"] is True          # found via sys.modules instead
    assert store_hook.patch_report()["skip_reason"] is None


def test_an_engine_and_no_module_at_all_refuses_quietly(hook_state):
    saved = sys.modules.pop(store_hook.TARGET_MODULE, None)
    try:
        first = store_hook.ensure_installed(object(), must_not_be_consulted)
        second = store_hook.ensure_installed(object(), must_not_be_consulted)
        assert first["installed"] is False and second["installed"] is False
        # The dashboard polls this path every few seconds: one episode, one count.
        assert hook_state["skips"] == 1
    finally:
        if saved is not None:
            sys.modules[store_hook.TARGET_MODULE] = saved


def test_a_subclass_override_from_outside_the_fork_is_refused(hook_state):
    """An override written in a test/plugin module is foreign: refuse it, leave
    the subclass exactly as it was, and say why in the snapshot."""
    module = build_module(FAKE_MANAGER_SOURCE)
    base = module.SimpleCPUOffloadScheduler

    class Sub(base):
        def _classify_store_candidate(self, gpu_block):   # noqa: ARG002
            return "mine"

    report = store_hook.install(Sub, must_not_be_consulted)
    assert report["installed"] is False
    assert "refusing to overwrite" in report["reason"]
    assert Sub()._classify_store_candidate(module.Block(1)) == "mine"


def test_a_subclass_override_written_by_the_fork_gets_its_own_wrapper(
        fake_manager_module, holder, hook_state):
    """Same MRO trap, the other way round: a subclass the fork itself wrote must
    be shadowed on the SUBCLASS, not assumed covered by the base."""
    base = fake_manager_module.SimpleCPUOffloadScheduler

    def fork_side_override(self, gpu_block):
        return base._classify_store_candidate(self, gpu_block)

    fork_side_override.__module__ = store_hook.TARGET_MODULE
    Sub = type("Sub", (base,),
                {store_hook.TARGET_METHOD: fork_side_override})

    assert store_hook.install(base, lambda: holder)["installed"] is True
    report = store_hook.install(Sub, lambda: holder)
    assert report["installed"] is True and report["class"] == "Sub"
    assert Sub.__dict__[store_hook.TARGET_METHOD] is not \
        base.__dict__[store_hook.TARGET_METHOD]
    assert getattr(Sub.__dict__[store_hook.TARGET_METHOD],
                   store_hook.OWNER_ATTR) == store_hook.OWNER_TAG


# --- the ctx dict -----------------------------------------------------------

def test_the_ctx_is_the_documented_shape(patched, monkeypatch):
    seen = []
    real = store_hook.build_store_ctx

    def record(manager, gpu_block, verdict):
        ctx = real(manager, gpu_block, verdict)
        seen.append(ctx)
        return ctx

    monkeypatch.setattr(store_hook, "build_store_ctx", record)
    patched.manager._in_flight_store_gpu_blocks = {1, 2}
    patched.manager._classify_store_candidate(patched.module.Block(5))

    assert len(seen) == 1
    ctx = seen[0]
    assert set(ctx) == {"gpu_block_id", "is_null", "has_block_hash",
                        "engine_admission", "mode", "target_free_gpu_blocks",
                        "block_size", "hash_block_size", "num_cpu_blocks",
                        "cpu_free_blocks", "in_flight_store_blocks",
                        "manager_class"}
    assert ctx["gpu_block_id"] == 5
    assert ctx["engine_admission"] == "READY"
    assert ctx["mode"] == "eager"
    assert ctx["cpu_free_blocks"] == 100
    assert ctx["num_cpu_blocks"] == 4096
    assert ctx["block_size"] == 256 and ctx["hash_block_size"] == 64
    assert ctx["target_free_gpu_blocks"] == 128
    assert ctx["in_flight_store_blocks"] == 2
    assert ctx["manager_class"] == "SimpleCPUOffloadScheduler"
    assert ctx["is_null"] is False and ctx["has_block_hash"] is True
    assert all(isinstance(value, (int, bool, str, type(None)))
               for value in ctx.values())   # flat: no engine objects leak out


def test_the_ctx_reports_lazy_mode_which_a_veto_cannot_reach(patched):
    """The policy is told which mode the tier is in, because in lazy mode its
    veto has no effect at all (`manager.py:566-617` never classifies)."""
    patched.manager._lazy_mode = True
    patched.policy_file.write_text(
        'VERSION = 1\n\n\ndef decide(view):\n    return 1\n\n\n'
        'def decide_store(ctx):\n'
        '    return "veto" if ctx["mode"] == "lazy" else "allow"\n',
        encoding="utf-8")
    assert patched.holder.reload()["ok"] is True

    verdict = patched.manager._classify_store_candidate(patched.module.Block(1))
    assert verdict is patched.module._StoreAdmission.ALREADY_CACHED
    report = store_hook.patch_report(patched.manager)
    assert report["vetoes"] == 1
    assert report["live_manager_mode"] == "lazy"
    assert report["governs_current_mode"] is False


# --- observability ----------------------------------------------------------

def test_the_snapshot_reports_the_patch_state(patched):
    payload = snapshot.snapshot_for_engine(patched.engine, patched.holder)
    report = payload["store_hook"]
    assert report["patched"] is True
    assert report["class"] == "SimpleCPUOffloadScheduler"
    assert report["method"] == "_classify_store_candidate"
    assert report["module"] == store_hook.TARGET_MODULE
    assert report["installs"] == 1
    assert report["live_manager_patched"] is True
    assert report["live_instance_shadows_method"] is False
    assert report["governs_current_mode"] is True      # eager: the veto applies
    assert report["live_manager_mode"] == "eager"
    assert report["fallbacks"] == 0 and report["skips"] == 0
    assert report["veto_member"] == "ALREADY_CACHED"
    assert report["veto_charged_to_engine_counter"] == "skipped_already_cached"
    assert isinstance(snapshot.json_safe({"store_hook": report}), dict)


def test_lazy_mode_says_the_kill_switch_would_not_work(patched):
    patched.manager._lazy_mode = True          # manager.py:566-617 never asks us
    report = store_hook.patch_report(patched.manager)
    assert report["live_manager_mode"] == "lazy"
    assert report["governs_current_mode"] is False
    assert report["patched"] is True


def test_vetoes_since_last_report_is_a_delta(patched):
    patched.policy_file.write_text(VETO, encoding="utf-8")
    patched.holder.reload()
    for i in range(3):
        patched.manager._classify_store_candidate(patched.module.Block(i))
    assert store_hook.patch_report(patched.manager)[
        "vetoes_since_last_report"] == 3
    assert store_hook.patch_report(patched.manager)[
        "vetoes_since_last_report"] == 0
    patched.manager._classify_store_candidate(patched.module.Block(9))
    assert store_hook.patch_report(patched.manager)[
        "vetoes_since_last_report"] == 1


def test_no_manager_still_reports_the_hook(patched):
    payload = snapshot.snapshot_for_engine(FakeEngine(None), patched.holder)
    assert payload["store_hook"]["patched"] is True
    assert payload["store_hook"]["live_manager_patched"] is None
    assert payload["store_hook"]["governs_current_mode"] is None


def test_a_broken_report_is_data_not_a_crash(monkeypatch, patched,
                                             reset_snapshot_state):
    def explode(manager=None):
        raise RuntimeError("no report")

    monkeypatch.setattr(store_hook, "patch_report", explode)
    payload = snapshot.snapshot_for_engine(patched.engine, patched.holder)
    assert payload["store_hook"]["patched"] is False
    assert "report failed" in payload["store_hook"]["reason"]
    assert reset_snapshot_state["errors"] == 1


# --- the holder's half of the contract ------------------------------------

def test_the_holder_exposes_decide_store_only_when_the_policy_has_one(
        policy_file, hook_state):
    policy_file.write_text(policy_source('"veto"', with_decide_store=False),
                           encoding="utf-8")
    holder = PolicyHolder(str(policy_file))
    assert holder.ensure_loaded()["ok"] is True   # `decide` alone still loads
    assert holder.decide_store is None
    policy_file.write_text(VETO, encoding="utf-8")
    assert holder.reload()["ok"] is True
    assert holder.decide_store({}) == "veto"


def test_the_shipped_policy_ships_a_no_opinion_veto():
    """Deploying this must not change what the tier does. A shipped `decide`
    that observes and a `decide_store` that allows, both loadable from the one
    file the reload verb can ever read."""
    shipped = PolicyHolder(str(config.canonical_policy_path()))
    assert shipped.ensure_loaded()["ok"] is True
    assert shipped.decide_store({"gpu_block_id": 1}) == "allow"
    assert shipped.decide({"tier": {"present": False}})["verdict"] == "no-tier"


# --- the kill switch, end to end, with only the existing verbs --------------

class FakeEngineCore:
    """Stands in for EngineCore: the class the plugin attaches to."""


def test_the_kill_switch_needs_only_editing_policy_and_posting_a_reload(
        fake_manager_module, policy_file, hook_state, monkeypatch):
    """Requirement 8, as the operator will actually do it.

    `POST /plugins/kvtier/policy/reload` is dispatched by name
    (`getattr(self, method_name)`, core.py:1669-1671), so calling the attached
    method here IS the HTTP verb's effect on the engine.
    """
    manager = fake_manager_module.SimpleCPUOffloadScheduler()
    engine = FakeEngineCore()
    engine.scheduler = FakeScheduler(manager)
    holder = PolicyHolder(str(policy_file))
    monkeypatch.setattr(engine_side, "_holder", holder)
    engine_side.attach_methods(FakeEngineCore)
    try:
        # Step 0: the hook is not in place yet (the connector imported the
        # manager only after register() ran) and the tier is storing. The fork's
        # own method is in the class dict; what must be absent is OUR marker.
        assert not hasattr(
            vars(fake_manager_module.SimpleCPUOffloadScheduler)[
                store_hook.TARGET_METHOD], store_hook.OWNER_ATTR)
        stats, stored = boundary_step_like_fork(fake_manager_module, manager,
                                                [fake_manager_module.Block(i)
                                                 for i in range(3)])
        assert stored == [0, 1, 2]

        # The snapshot adopts the patch (it is the other verb that reaches the
        # engine, and the one the dashboard polls). `getattr(self, name)` is
        # literally how UTILITY dispatch resolves the method.
        first = getattr(engine, engine_side.config.METHOD_SNAPSHOT)()
        assert first["store_hook"]["patched"] is True
        assert first["store_hook"]["governs_current_mode"] is True

        # Step 1: make decide_store always veto.
        policy_file.write_text(VETO, encoding="utf-8")
        # Step 2: the ONE existing mutating verb.
        report = getattr(engine, engine_side.config.METHOD_RELOAD_POLICY)()
        assert report["ok"] is True

        stats, stored = boundary_step_like_fork(fake_manager_module, manager,
                                                [fake_manager_module.Block(i)
                                                 for i in range(3, 8)])
        assert stored == [], "the tier must stop parking new blocks immediately"
        assert stats.skipped_already_cached == 5

        after = getattr(engine, engine_side.config.METHOD_SNAPSHOT)()
        assert after["store_hook"]["vetoes"] == 5
        assert after["store_hook"]["vetoes_since_last_report"] == 5
        assert after["tier"]["boundary_store_stats"]["stored"] == 3  # unchanged
    finally:
        for name in (engine_side.config.METHOD_SNAPSHOT,
                     engine_side.config.METHOD_RELOAD_POLICY):
            if name in vars(FakeEngineCore):
                delattr(FakeEngineCore, name)


def test_the_wrapper_picks_up_a_reload_a_frozen_binding_never_would(
        fake_manager_module, holder, policy_file, hook_state):
    """Requirement 2's property, asserted through the hook rather than the holder.

    The wrapper holds the HOLDER and reads `holder.decide_store` per call, so
    the reloaded module's function is what runs. A `from policy import
    decide_store` binding would have frozen the first one forever.
    """
    manager = fake_manager_module.SimpleCPUOffloadScheduler()
    assert store_hook.install(fake_manager_module.SimpleCPUOffloadScheduler,
                              lambda: holder)["installed"] is True
    frozen = holder.decide_store               # the shape of a stale binding

    assert manager._classify_store_candidate(fake_manager_module.Block(1)) is \
        fake_manager_module._StoreAdmission.READY
    assert frozen({}) == "allow"

    policy_file.write_text(VETO, encoding="utf-8")
    assert holder.reload()["ok"] is True
    assert frozen({}) == "allow"               # frozen: still the old body
    assert manager._classify_store_candidate(fake_manager_module.Block(2)) is \
        fake_manager_module._StoreAdmission.ALREADY_CACHED


def test_the_store_path_is_microseconds(patched):
    """A smoke ceiling, like `build_us` in test_snapshot.py, not a claim.

    This runs once per candidate block inside a scheduler decision on the
    thread that serves users, so the shape of the budget is what matters.
    """
    module, manager = patched.module, patched.manager
    blocks = [module.Block(i) for i in range(64)]
    for source in (ALLOW, VETO):
        patched.policy_file.write_text(source, encoding="utf-8")
        patched.holder.reload()
        started = time.perf_counter()
        for _ in range(50):
            for block in blocks:
                manager._classify_store_candidate(block)
        elapsed = time.perf_counter() - started
        assert elapsed / (50 * len(blocks)) < 50e-6, (source, elapsed)


def test_a_veto_never_lets_the_hook_itself_raise(patched):
    """The one invariant the scheduler depends on, hammered from every side."""
    module, manager = patched.module, patched.manager
    nasty = [RAISES, policy_source('{"a": 1}'), VETO, ALLOW,
             'VERSION = 1\n\n\ndef decide(view):\n    return 1\n']
    for source in nasty:
        patched.policy_file.write_text(source, encoding="utf-8")
        patched.holder.reload()
        for i in range(4):
            verdict = manager._classify_store_candidate(module.Block(i))
        assert verdict in set(module._StoreAdmission)
