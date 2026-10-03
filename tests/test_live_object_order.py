"""REGRESSION: the install must be evaluated against the LIVE OBJECT, not a
pristine class (the production incidents of 2026-10-03).

Every pre-existing test installs on the class and THEN constructs - the
opposite of production, where `SimpleCPUOffloadScheduler` is built INSIDE
`create_connector`, before any of our layers are fitted. Two production bugs
came from that wrong order:

1. The additive `self.prefix_cacheable_group_ids` comes from the `__init__`
   wrapper, which never runs for an instance constructed before the wrapper
   exists. The patched methods then read a missing attribute off the only
   instance that matters (live traceback 2026-10-03 12:41, engine death):
   `AttributeError: 'SimpleCPUOffloadScheduler' object has no attribute
   'prefix_cacheable_group_ids'`.
2. A re-install while guard mode is armed fingerprinted GUARD'S OWN WRAPPER
   of `update_state_after_alloc` and reported "the fork changed this function
   since 10b8e07" - blaming the fork for our own layering, refusing the
   install, and auto-disarming the tier.

The live manager here is built with `cls.__new__(cls)` and the real
constructor's attribute set MINUS `prefix_cacheable_group_ids` (the UNPATCHED
fork constructor does not set it - fixtures_pr56810.py carries the fork's own
text, and it does not appear there). The class itself is the fork-text class,
so the real install path (fingerprints included) runs. Instantiating the real
fork class through its own `__init__` needs a full VllmConfig + GPU host; the
duck-typed instance covers exactly the property under test: an instance that
exists BEFORE the patch is fitted.
"""

from __future__ import annotations

import sys
import types

import fixtures_pr56810 as FX
import pytest
from test_backport import (
    GROUP_BLOCK_SIZES,
    HASH_BLOCK_SIZE,
    PCIDS,
    RING_BLOCK_SIZE,
    RING_GROUP,
    _Admission,
    _Config,
    _Coordinator,
    _GpuBlock,
    _GpuPool,
    _Group,
    _Pool,
    _Request,
    _State,
    selector_namespace,
)

from kvtier_dev_plugin import backport, guard_mode

TARGET = backport.TARGET_MODULE


@pytest.fixture(autouse=True)
def reset_states():
    saved = dict(backport._state)
    saved_verdicts = dict(backport._state["method_verdicts"])
    saved_switch = dict(backport._state["switch_verdicts"])
    saved_guard = dict(guard_mode._state)
    backport.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    yield
    backport._state.clear()
    backport._state.update(saved)
    backport._state["method_verdicts"] = saved_verdicts
    backport._state["switch_verdicts"] = saved_switch
    guard_mode._state.clear()
    guard_mode._state.update(saved_guard)
    sys.modules.pop(TARGET, None)
    sys.modules.pop(backport.METADATA_MODULE, None)


# --- production-shaped fixture: the fork-text class + a live instance --------

class _EmptyMetadata:
    def __init__(self):
        self.store_gpu_blocks = []
        self.store_cpu_blocks = []
        self.load_gpu_blocks = []
        self.load_cpu_blocks = []
        self.store_event = -1
        self.load_event = -1


def _fork_text_class():
    """The fork-text class WITH the four switch methods, nothing installed.

    The two stand-ins that lead into the selector are faithful to their call
    sites: `_queue_finished_eager_store` is manager.py:1169 and calls the
    selector at :1177, returning None; `build_connector_meta` is the per-step
    route (`prepare_store_specs` -> selector at :745), so it calls the
    selector too and returns a sentinel. Each RECORDS that the selector was
    reached, so a disarmed gate and a skipped ring group stay distinguishable
    from "the route never ran".
    """
    cls, module = FX.build_fork_class(selector_namespace())
    # The real class's __module__ - guard_mode.install composes over a body it
    # recognizes as the fork's own only under this name.
    cls.__module__ = TARGET
    reached = {"selector": 0}

    def _drive_selector(self, request):
        block_ids = []
        nxt = 1
        for g in range(len(GROUP_BLOCK_SIZES)):
            count = 4 if g != RING_GROUP else 1
            block_ids.append(list(range(nxt, nxt + count)))
            nxt += count
        state = _State(request, len(GROUP_BLOCK_SIZES))
        gpu, advanced, _meta = self._select_eager_blocks_to_store(
            state, tuple(block_ids))
        reached["selector"] += 1
        return gpu, advanced

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 4096, True

    def build_connector_meta(self, scheduler_output):
        # ROUTE 1: manager.py:488 -> prepare_store_specs -> :745 selector.
        self._last_store = _drive_selector(self, _Request(2))
        return "ENGINE METADATA"

    def has_pending_stores(self):
        return False

    def _queue_finished_eager_store(self, request, block_ids):
        # ROUTE 2: manager.py:1169 -> :1177 selector. Returns None either way.
        self._last_store = _drive_selector(self, request)
        return

    cls.get_num_new_matched_tokens = get_num_new_matched_tokens
    cls.build_connector_meta = build_connector_meta
    cls.has_pending_stores = has_pending_stores
    cls._queue_finished_eager_store = _queue_finished_eager_store
    # The real class's methods all report the fork's own module - the guard's
    # composition rule keys on it, so the stand-ins must be faithful here.
    for _fn in (get_num_new_matched_tokens, build_connector_meta,
                has_pending_stores, _queue_finished_eager_store):
        _fn.__module__ = TARGET
    for _name in ("__init__", "_estimate_lazy_target_blocks",
                  "update_state_after_alloc", "_select_eager_blocks_to_store"):
        _raw = cls.__dict__.get(_name)
        _f = _raw.__func__ if isinstance(_raw, staticmethod) else _raw
        _f.__module__ = TARGET
    # Class-level helpers the selector calls (the real fork class defines these;
    # the fixture assembles only the four PR #56810 methods).
    cls._classify_store_candidate = lambda self, gpu_block: _Admission.READY
    cls._cached_gpu_block = lambda self, hashes, idx, g: None
    cls._classify_store_candidate.__module__ = TARGET
    cls._cached_gpu_block.__module__ = TARGET
    cls.selector_reached = reached

    module = types.ModuleType(TARGET)
    module.SimpleCPUOffloadScheduler = cls
    sys.modules[TARGET] = module
    meta = types.ModuleType(backport.METADATA_MODULE)
    meta.SimpleCPUOffloadMetadata = _EmptyMetadata
    sys.modules[backport.METADATA_MODULE] = meta
    return cls


def build_live_manager(cls, *, with_pcids=None, cpu_config="real",
                       gpu_config="real"):
    """An instance of the fork-text class, constructed BEFORE the patch.

    `cls.__new__(cls)` + the unpatched constructor's attribute set, verbatim
    from fixtures_pr56810._ORIG_INIT: the attribute this backport ADDS is the
    one a pre-patch instance provably does not have (`with_pcids=None`).
    `cpu_config`/`gpu_config`: "real" = carries prefix_cacheable_group_ids,
    "bare" = does not, "absent" = the config attribute itself is missing.
    """
    mgr = cls.__new__(cls)
    mgr.vllm_config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(
        world_size=1, decode_context_parallel_size=1))
    mgr.kv_cache_config = _config_for(cpu_config if gpu_config == "absent"
                                      else gpu_config)
    if gpu_config == "absent":
        del mgr.kv_cache_config
    mgr.enable_kv_cache_events = False
    mgr.block_size = HASH_BLOCK_SIZE
    mgr.hash_block_size = HASH_BLOCK_SIZE
    mgr.cpu_kv_cache_config = _config_for(cpu_config)
    if cpu_config == "absent":
        del mgr.cpu_kv_cache_config
    mgr.num_cpu_blocks = 1024
    mgr.fa_gidx = 0
    mgr.fa_block_size = HASH_BLOCK_SIZE
    mgr.cpu_coordinator = _Coordinator(len(GROUP_BLOCK_SIZES))
    mgr.group_block_sizes = list(GROUP_BLOCK_SIZES)
    mgr.cpu_block_pool = _Pool(1000)
    mgr._gpu_block_pool = _GpuPool([_GpuBlock(i, f"bh{i}".encode())
                                    for i in range(4096)])
    mgr._reqs_to_load = {}
    mgr._load_event_to_reqs = {}
    mgr._pending_cpu_hits = {}
    mgr._lazy_mode = False
    mgr._cursor = None
    mgr._target_free = 0
    mgr._store_event_to_blocks = {}
    mgr._abandoned_store_event_to_blocks = {}
    mgr._reqs_to_store = {}
    mgr._store_event_to_reqs = {}
    mgr._in_flight_store_gpu_blocks = set()
    mgr._pending_finished_stores = []
    mgr._abandoned_reqs_to_load = {}
    mgr._load_event_counter = 0
    mgr._store_event_counter = 0
    if with_pcids is not None:
        mgr.prefix_cacheable_group_ids = tuple(with_pcids)
    return mgr


def _config_for(kind):
    if kind == "absent":
        return None
    cfg = _Config([_Group(b, b != RING_BLOCK_SIZE) for b in GROUP_BLOCK_SIZES],
                  PCIDS)
    if kind == "bare":
        del cfg.prefix_cacheable_group_ids
    return cfg


class _FakeEngine:
    def __init__(self, manager):
        self.scheduler = types.SimpleNamespace(
            connector=types.SimpleNamespace(scheduler_manager=manager))


def _assert_selector_skips_the_ring(mgr):
    """Both routes into the selector run, and NEITHER touches the ring group.

    Before the fix this raises the production AttributeError (the patched
    selector reads the missing attribute on its first loop statement); a
    blanket-skip fix would instead trip the divisibility poison-pill.
    """
    request = _Request(2)
    # Direct route.
    block_ids, nxt = [], 1
    for g in range(len(GROUP_BLOCK_SIZES)):
        count = 4 if g != RING_GROUP else 1
        block_ids.append(list(range(nxt, nxt + count)))
        nxt += count
    state = _State(request, len(GROUP_BLOCK_SIZES))
    gpu, advanced, _meta = mgr._select_eager_blocks_to_store(
        state, tuple(block_ids))
    ring_ids = set(block_ids[RING_GROUP])
    assert not (set(gpu) & ring_ids), "the ring group's blocks were stored"
    assert advanced[RING_GROUP] == 0

    backport.set_armed(True)  # the finish path is gated while disarmed
    # Route 2: _queue_finished_eager_store -> selector (manager.py:1177).
    assert mgr._queue_finished_eager_store(request, tuple(block_ids)) is None
    assert type(mgr).selector_reached["selector"] >= 1
    # Route 1: build_connector_meta -> prepare_store_specs -> selector (:745).
    assert mgr.build_connector_meta(None) == "ENGINE METADATA"
    assert type(mgr).selector_reached["selector"] >= 2


# --- BUG 1: construct first, install after (the production order) ------------

def test_construct_first_then_install_the_selector_must_run_on_the_live_object():
    """THE production repro: install AFTER construction must not crash.

    Fails on the unfixed code with the production traceback:
    `AttributeError: 'SimpleCPUOffloadScheduler' object has no attribute
    'prefix_cacheable_group_ids'` from the patched selector, because the
    `__init__` wrapper only fires for instances built after install.
    """
    cls = _fork_text_class()
    mgr = build_live_manager(cls)               # constructed BEFORE any patch
    assert not hasattr(mgr, "prefix_cacheable_group_ids")

    report = backport.ensure_installed(_FakeEngine(mgr))
    assert report["installed"] is True, report
    _assert_selector_skips_the_ring(mgr)
    # The install verdict must state the live-object check, and where the
    # back-filled value came from (upstream reads cpu_kv_cache_config).
    live = report.get("live_check") or backport._state.get("live_check") or {}
    assert live.get("checked") is True, live
    assert live.get("ok") is True, live
    assert "cpu_kv_cache_config" in str(live.get("source"))
    assert mgr.prefix_cacheable_group_ids == PCIDS


def test_backfill_falls_back_to_kv_cache_config_and_says_so():
    cls = _fork_text_class()
    mgr = build_live_manager(cls, cpu_config="bare", gpu_config="real")
    report = backport.ensure_installed(_FakeEngine(mgr))
    assert report["installed"] is True, report
    live = report["live_check"]
    assert live["ok"] is True
    assert live["source"].startswith("back-filled from kv_cache_config.")
    assert "fallback" in live["source"]
    _assert_selector_skips_the_ring(mgr)


def test_no_config_means_refuse_and_disarm_never_run_patched_code():
    """Neither config can yield the attribute: installed=false, disarmed, and
    arming is refused - the patched selector can never be reached."""
    cls = _fork_text_class()
    # kv_cache_config absent entirely: the geometry guard cannot veto (no
    # config to check), so this isolates the LIVE-OBJECT refusal path.
    mgr = build_live_manager(cls, cpu_config="bare", gpu_config="absent")
    report = backport.ensure_installed(_FakeEngine(mgr))
    assert report["installed"] is False
    assert report["auto_disarmed"] is True
    assert report["disarm_reason"] == "incomplete_install"
    assert "prefix_cacheable_group_ids" in report["reason"]
    assert backport.is_armed() is False
    res = backport.set_armed(True)
    assert res["refused"] is True
    assert res["kind"] == "incomplete_install"
    # Both routes to the selector stay gated by the kill switch (disarmed).
    request = _Request(2)
    assert mgr._queue_finished_eager_store(request, ((), (), (), (), (), ())) \
        is None
    assert isinstance(mgr.build_connector_meta(None), _EmptyMetadata)
    assert cls.selector_reached["selector"] == 0


def test_existing_instance_with_the_attribute_is_accepted_and_reported():
    """An instance built AFTER the wrapper (or already back-filled) passes the
    live check without being rewritten: the live attribute wins over any
    config, even one that reports no ids of its own."""
    cls = _fork_text_class()
    mgr = build_live_manager(cls, with_pcids=PCIDS, cpu_config="bare")
    report = backport.ensure_installed(_FakeEngine(mgr))
    assert report["installed"] is True, report
    assert report["live_check"]["ok"] is True
    assert "already" in str(report["live_check"]["source"]).lower()
    _assert_selector_skips_the_ring(mgr)


def test_install_verdict_asserts_required_attributes_present():
    """A live instance missing a constructor attribute the patched bodies read
    (not just the new one) is a refusal, named."""
    cls = _fork_text_class()
    mgr = build_live_manager(cls)
    del mgr.cpu_coordinator        # set by the REAL __init__; here: absent
    report = backport.ensure_installed(_FakeEngine(mgr))
    assert report["installed"] is False
    assert "cpu_coordinator" in report["reason"]
    assert backport.is_armed() is False


# --- BUG 2: re-install must not fingerprint our own guard wrapper ------------

def _arm_guard(cls):
    """Arm guard mode over whatever methods this class actually has.

    The fixture assembles four PR #56810 methods plus the four switch
    stand-ins; the guard's other five targets (prepare_store_specs, reset,
    ...) are legitimately absent and refused - that is the guard honestly
    declining to invent a method, not a composition failure. The one that
    matters here is `update_state_after_alloc`: it MUST be guarded (it is
    also a backport METHOD - the whole bug 2 lives in that overlap).
    """
    report = guard_mode.install(cls)
    assert report["installed"] is True, report
    assert "update_state_after_alloc" in report["attached"], report
    assert "update_state_after_alloc" not in report["skipped"], report
    guard_mode._state["armed"] = True


def test_reinstall_under_guard_keeps_installed_and_nesting():
    """install -> guard arm -> re-install: no refusal, no auto-disarm, guard
    stays on top of our copy. Fails on the unfixed code: the re-install
    fingerprints guard's own wrapper of update_state_after_alloc and reports
    'the fork changed this function since 10b8e07', refusing the install and
    auto-disarming the tier."""
    cls = _fork_text_class()
    mgr = build_live_manager(cls)
    engine = _FakeEngine(mgr)
    assert backport.ensure_installed(engine)["installed"] is True
    _arm_guard(cls)

    report = backport.ensure_installed(engine)   # what a verb re-run does
    assert report["refused_methods"] == [], report
    assert report["installed"] is True, report
    assert report["auto_disarmed"] is False, report
    # Nesting preserved, not replaced: guard on top, our copy under it.
    top = cls.__dict__["update_state_after_alloc"]
    assert getattr(top, guard_mode.OWNER_ATTR, None) == guard_mode.OWNER_TAG
    under = getattr(top, guard_mode.ORIGINAL_ATTR, None)
    assert under is not None
    assert getattr(under, backport.OWNER_ATTR, None) == backport.OWNER_TAG
    # The reporting fields must not disagree about what is live.
    section = backport.section(mgr)
    assert section["installed"] == section["patched"] == True  # noqa: E712


def test_install_over_guard_armed_first_order():
    """The reverse order: guard armed BEFORE the backport ever ran. The install
    must land UNDER the guard wrapper and still report installed=true."""
    cls = _fork_text_class()
    mgr = build_live_manager(cls)
    _arm_guard(cls)
    report = backport.ensure_installed(_FakeEngine(mgr))
    assert report["installed"] is True, report
    assert report["refused_methods"] == [], report
    top = cls.__dict__["update_state_after_alloc"]
    assert getattr(top, guard_mode.OWNER_ATTR, None) == guard_mode.OWNER_TAG
    under = getattr(top, guard_mode.ORIGINAL_ATTR, None)
    assert getattr(under, backport.OWNER_ATTR, None) == backport.OWNER_TAG
    _assert_selector_skips_the_ring(mgr)


def test_fingerprint_refusal_still_fires_for_a_genuinely_mutated_fork():
    """The fix must get MORE precise, not weaker: a real fork edit underneath
    our guard layers is still refused, with a message that says the body
    UNDERNEATH our layers changed - and it never claims the fork changed
    because of our own wrapper."""
    import linecache
    ns = selector_namespace()
    parts = []
    for name in ("__init__", "_estimate_lazy_target_blocks",
                 "update_state_after_alloc", "_select_eager_blocks_to_store"):
        text = FX.ORIGINAL_SOURCES[name]
        if name == "update_state_after_alloc":
            text = text.replace("def update_state_after_alloc(",
                                "def update_state_after_alloc(  # fork edit\n", 1)
        parts.append(text)
    src = "class SimpleCPUOffloadScheduler:\n" + "".join(parts)
    fname = "<mutated_fork_class>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    module = {"__name__": TARGET}
    module.update(ns)
    exec(compile(src, fname, "exec"), module, module)  # noqa: S102
    cls = module["SimpleCPUOffloadScheduler"]
    cls.__module__ = TARGET
    stub = types.ModuleType(TARGET)
    stub.SimpleCPUOffloadScheduler = cls
    sys.modules[TARGET] = stub

    mgr = build_live_manager(cls)
    # Arm the guard FIRST so the refused body sits under our own layer.
    _arm_guard(cls)
    report = backport.ensure_installed(_FakeEngine(mgr))
    assert report["installed"] is False
    assert "update_state_after_alloc" in report["refused_methods"]
    assert report["auto_disarmed"] is True
    assert report["disarm_reason"] == "incomplete_install"
    reason = report["reason"]
    # Honest wording: the body underneath our layers is what changed; and the
    # refusal must NOT read like the guard wrapper itself was blamed-as-fork...
    assert "under" in reason.lower()
    # ...while genuinely OUR layer over an UNCHANGED fork body is never
    # refused (covered by the two composition tests above).
    assert "update_state_after_alloc" in reason
