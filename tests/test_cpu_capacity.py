"""The per-rank CPU arena capacity clamp: the handshake, the cap, the surgery.

Nothing here imports vllm. The two stub layers are built to the fork's real
shapes at HEAD 10b8e07, and the shapes are the point:

  `KVConnectorBase_V1`, exec'd into a module NAMED `cpu_capacity.BASE_MODULE`,
  so the stub methods' `__module__` is the module the real ones are defined in
  (base.py:447-456 `get_handshake_metadata` -> None; base.py:716-731
  `set_xfer_handshake_metadata_pp_aware`, INCLUDING its ValueError for
  pp_rank > 0 - the trap that makes a naive carrier crash the boot).

  `BlockPool`/`FreeKVCacheBlockQueue`/`KVCacheBlock`, mirroring
  block_pool.py:162-196 (blocks, the free queue, the null block popped from its
  head), block_pool.py:664-676 (`get_new_blocks` and its shortfall ValueError),
  block_pool.py:802-819 (`evict_blocks`), block_pool.py:861-867
  (`get_num_free_blocks` -> the queue's maintained counter) and
  kv_cache_utils.py:368 (`remove()` raising on a block that is not in the queue).

The clamp is claimed to be "one-time surgery, then zero per-call cost", so the
tests do not merely check the counters: they drain the pool and prove no id at
or above the cap can ever come out of it.
"""

from __future__ import annotations

import json
import types

import pytest

from kvtier_dev_plugin import config, cpu_capacity, engine_side, snapshot

# --- the fork's connector base, with the fork's own module name -------------

FAKE_BASE_SOURCE = '''
"""Shaped after vllm/distributed/kv_transfer/kv_connector/v1/base.py."""


class KVConnectorBase_V1:
    """Both handshake methods, at the fork's shapes.

    `get_handshake_metadata` returns None for a connector that has no payload of
    its own (base.py:447-456, which is what SimpleCPUOffloadConnector inherits).
    `set_xfer_handshake_metadata_pp_aware` reproduces base.py:716-731 including
    the pp_rank > 0 ValueError, because that is what a plugin must not provoke.
    """

    payload = None

    def get_handshake_metadata(self):
        return self.payload

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        self.received = metadata
        if any(pp_rank != 0 for pp_rank, _ in metadata):
            raise ValueError(
                f"{type(self).__name__} received pp_rank > 0 handshake metadata "
                "but does not support PP-disaggregated KV transfer."
            )
        return None
'''


class Block:
    """KVCacheBlock: block_id / ref_cnt / is_null / the two free-list links."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.is_null = False
        self.block_hash = None
        self.prev_free_block = None
        self.next_free_block = None


class FreeQueue:
    """FreeKVCacheBlockQueue, transcribed from kv_cache_utils.py:208-410.

    The fake head and tail sentinels are the point, not decoration: the fork
    guarantees that every block IN the queue has both links non-None (the
    sentinels stand in for the missing neighbours) and every block outside it
    has both None. That invariant is exactly what `cpu_capacity._in_free_queue`
    tests, so a stub without sentinels would test something the fork does not do.
    `remove()` on a block that is not in the queue raises RuntimeError, as the
    fork's does.
    """

    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)
        for i, block in enumerate(blocks):
            if i > 0:
                block.prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                block.next_free_block = blocks[i + 1]
        self.fake_head = Block(-1)
        self.fake_tail = Block(-1)
        if self.num_free_blocks:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    def popleft(self):
        if self.fake_head.next_free_block is self.fake_tail:
            assert self.num_free_blocks == 0
            raise ValueError("No free blocks available")
        first = self.fake_head.next_free_block
        nxt = first.next_free_block
        self.fake_head.next_free_block = nxt
        nxt.prev_free_block = self.fake_head
        first.prev_free_block = first.next_free_block = None
        self.num_free_blocks -= 1
        return first

    def popleft_n(self, n):
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n
        current = self.fake_head.next_free_block
        ret = []
        for _ in range(n):
            ret.append(current)
            last = current
            current = current.next_free_block
            last.prev_free_block = last.next_free_block = None
        if current is not self.fake_tail:
            self.fake_head.next_free_block = current
            current.prev_free_block = self.fake_head
        return ret

    def remove(self, block):
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block):
        last = self.fake_tail.prev_free_block
        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = block
        self.num_free_blocks += 1

    def append_n(self, blocks):
        for block in blocks:
            self.append(block)

    def prepend_n(self, blocks):
        first = self.fake_head.next_free_block
        previous = self.fake_head
        for block in blocks:
            block.prev_free_block = previous
            previous.next_free_block = block
            previous = block
        previous.next_free_block = first
        first.prev_free_block = previous
        self.num_free_blocks += len(blocks)


class Pool:
    """BlockPool: the constructor's null block, the maintained free count, the
    shortfall ValueError, and evict_blocks clearing a hash."""

    def __init__(self, num_blocks):
        self.blocks = [Block(idx) for idx in range(num_blocks)]
        self.free_block_queue = FreeQueue(self.blocks)
        self.num_gpu_blocks = num_blocks
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.cached_block_hash_to_block = {}
        self.evicted = []

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    def get_new_blocks(self, num_blocks):
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        return self.free_block_queue.popleft_n(num_blocks)

    def cache(self, block, block_hash):
        block.block_hash = block_hash
        self.cached_block_hash_to_block[block_hash] = block

    def evict_blocks(self, block_ids):
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}.")
            block = self.blocks[block_id]
            self.evicted.append(block_id)
            if block.block_hash is not None:
                self.cached_block_hash_to_block.pop(block.block_hash, None)
                block.block_hash = None


class Config:
    """The manager's CPU `kv_cache_config`: it owns one field we mirror."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks


class Handler:
    """SimpleCPUOffloadWorker, as far as the handshake cares: arena size only."""

    def __init__(self, num_cpu_blocks):
        self.num_cpu_blocks = num_cpu_blocks


class Manager:
    """SimpleCPUOffloadScheduler: the CPU pool, its mirrors, a GPU pool to spare."""

    def __init__(self, num_cpu_blocks=200):
        self.cpu_block_pool = Pool(num_cpu_blocks)
        self.num_cpu_blocks = num_cpu_blocks
        self.cpu_kv_cache_config = Config(num_cpu_blocks)
        self._gpu_block_pool = Pool(500)


class Scheduler:
    def __init__(self, connector):
        self.connector = connector


class Engine:
    def __init__(self, connector):
        self.scheduler = Scheduler(connector)


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def cap_state():
    """Both process-wide state dicts, restored: these counters are shared."""
    saved = dict(cpu_capacity._state)
    saved_errors = dict(snapshot._state)
    cpu_capacity.reset_state_for_tests()
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0})
    yield cpu_capacity._state
    cpu_capacity._state.update(saved)
    snapshot._state.update(saved_errors)


@pytest.fixture
def base_module(cap_state):
    """A fresh stub of the fork's connector base class, exec'd into a module
    carrying the fork's module name (so `__module__` checks see the fork)."""
    module = types.ModuleType(cpu_capacity.BASE_MODULE)
    exec(FAKE_BASE_SOURCE, module.__dict__)  # noqa: S102 - our own source string
    return module


@pytest.fixture
def base_class(base_module):
    """The stub class itself - what `install()` takes and what we assert on."""
    return base_module.KVConnectorBase_V1


@pytest.fixture
def installed(base_class):
    """The stub base class with both wrappers installed."""
    report = cpu_capacity.install(base_class)
    assert report["installed"] is True, report
    return base_class


@pytest.fixture
def connector(installed):
    """One connector instance of the installed class: manager + worker handler."""
    manager = Manager(200)
    instance = installed()
    instance.scheduler_manager = manager
    instance.worker_handler = Handler(200)
    return instance


def handshake(connector, content):
    """Drive the installed scheduler-side wrapper the way `core.py:302` does."""
    return connector.set_xfer_handshake_metadata_pp_aware(content)


def carrier(capacity, **extra):
    entry = {cpu_capacity.CAPACITY_KEY: capacity}
    entry.update(extra)
    return entry


# --- the fork's own behaviour, before we touch it ---------------------------

def test_the_stub_base_class_behaves_like_the_fork(base_module):
    cls = base_module.KVConnectorBase_V1
    assert cls().get_handshake_metadata() is None
    assert cls().set_xfer_handshake_metadata_pp_aware({}) is None
    with pytest.raises(ValueError, match="pp_rank > 0"):
        cls().set_xfer_handshake_metadata_pp_aware({(1, 0): object()})


def test_the_stub_pool_behaves_like_the_forks_pool():
    pool = Pool(5)
    # The null block is id 0 and is not allocatable, so four blocks are free.
    assert pool.get_num_free_blocks() == 4
    assert pool.get_new_blocks(4) == pool.blocks[1:]
    with pytest.raises(ValueError, match="free blocks from the pool"):
        pool.get_new_blocks(1)


def test_the_stub_queue_refuses_a_double_remove():
    pool = Pool(4)
    queue = pool.free_block_queue
    queue.remove(pool.blocks[2])
    with pytest.raises(RuntimeError):
        queue.remove(pool.blocks[2])


def test_the_stub_queue_keeps_the_forks_link_invariant():
    """In the queue: both links non-None (the sentinels count). Out: both None."""
    pool = Pool(4)
    for block in pool.blocks[1:]:
        assert block.prev_free_block is not None
        assert block.next_free_block is not None
        assert cpu_capacity._in_free_queue(block) is True
    assert pool.get_new_blocks(1)[0].prev_free_block is None
    assert pool.null_block.prev_free_block is None
    assert cpu_capacity._in_free_queue(pool.null_block) is False


# --- installation -----------------------------------------------------------

def test_install_wraps_both_handshake_methods(installed, base_module):
    for name in (cpu_capacity.WORKER_METHOD, cpu_capacity.SCHEDULER_METHOD):
        method = getattr(base_module.KVConnectorBase_V1, name)
        assert getattr(method, cpu_capacity.OWNER_ATTR) == cpu_capacity.OWNER_TAG
    assert cpu_capacity._state["worker_installed"] is True
    assert cpu_capacity._state["scheduler_installed"] is True
    assert cpu_capacity.section()["installed"] is True


def test_ensure_installed_retries_only_when_the_class_was_never_found(
        cap_state, base_class, monkeypatch):
    """A late `import vllm` is the one case worth retrying."""
    import sys

    module = types.ModuleType(cpu_capacity.BASE_MODULE)
    monkeypatch.setitem(sys.modules, cpu_capacity.BASE_MODULE, module)
    assert cpu_capacity.ensure_installed()["installed"] is False
    assert cpu_capacity._state["installs"] == 0
    assert cpu_capacity._state["class"] is None

    module.KVConnectorBase_V1 = base_class  # the class arrives late
    assert cpu_capacity.ensure_installed()["installed"] is True
    reinstalls = cpu_capacity._state["reinstalls"]
    for _ in range(5):
        cpu_capacity.ensure_installed()
    assert cpu_capacity._state["reinstalls"] == reinstalls
    assert cpu_capacity._state["installs"] == 1
    assert sys.modules[cpu_capacity.BASE_MODULE] is module


def test_a_refused_class_is_not_retried_on_every_poll(cap_state, base_class):
    def foreign(self):
        return None

    setattr(foreign, cpu_capacity.OWNER_ATTR, "another-plugin/9")
    setattr(base_class, cpu_capacity.WORKER_METHOD, foreign)
    setattr(base_class, cpu_capacity.SCHEDULER_METHOD, foreign)
    report = cpu_capacity.install(base_class)
    assert report["installed"] is False
    for _ in range(5):
        cpu_capacity.ensure_installed()
    assert cpu_capacity._state["reinstalls"] == 0
    # Two episodes, one per distinct reason (each method is named in its own
    # message) - and the five retries below the first install added nothing.
    assert cpu_capacity._state["skips"] == 2


def test_install_finds_the_class_in_sys_modules_without_importing_it(
        cap_state, base_module, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, cpu_capacity.BASE_MODULE,
                       base_module)
    report = cpu_capacity.install()
    assert report["installed"] is True
    assert report["class"] == "KVConnectorBase_V1"
    assert "vllm" not in __import__("sys").modules


def test_a_missing_module_is_reported_and_nothing_is_imported(cap_state,
                                                              monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, cpu_capacity.BASE_MODULE,
                        raising=False)
    report = cpu_capacity.install()
    assert report["installed"] is False
    assert "not imported in this process" in report["reason"]
    assert cpu_capacity._state["skips"] == 1
    assert "vllm" not in __import__("sys").modules


def test_double_install_is_a_no_op_and_cannot_grow_a_wrapper_chain(installed,
                                                                   base_class,
                                                                   cap_state):
    first = getattr(base_class, cpu_capacity.WORKER_METHOD)
    report = cpu_capacity.install(base_class)
    assert report["attached"] == []
    assert set(report["replaced"]) == {cpu_capacity.WORKER_METHOD,
                                       cpu_capacity.SCHEDULER_METHOD}
    second = getattr(base_class, cpu_capacity.WORKER_METHOD)
    assert second is not first
    # The re-install re-bound the FORK's function, not the wrapper it replaced,
    # so one call still produces exactly one carrier and one counter bump.
    instance = base_class()
    instance.worker_handler = Handler(7)
    assert second(instance) == {cpu_capacity.CAPACITY_KEY: 7}
    assert cpu_capacity._state["worker_reports"] == 1
    assert cpu_capacity._state["reinstalls"] == 1


def test_a_foreign_owner_of_the_target_name_is_refused(cap_state, base_class):
    def foreign(self):
        return "not yours"

    setattr(foreign, cpu_capacity.OWNER_ATTR, "another-plugin/9")
    setattr(base_class, cpu_capacity.WORKER_METHOD, foreign)
    report = cpu_capacity.install(base_class)
    assert cpu_capacity.WORKER_METHOD in report["skipped"]
    assert "another plugin's owner marker" in (report["reason"] or "")
    assert getattr(base_class, cpu_capacity.WORKER_METHOD) is foreign
    assert cpu_capacity._state["worker_installed"] is False
    # The scheduler half still lands: a half pair is reported, not hidden.
    assert cpu_capacity._state["scheduler_installed"] is True
    assert cpu_capacity.SCHEDULER_METHOD in report["attached"]


def test_a_function_from_a_foreign_module_is_refused(cap_state, base_class):
    def foreign(self):  # defined in THIS test module, not in the fork
        return None

    setattr(base_class, cpu_capacity.SCHEDULER_METHOD, foreign)
    report = cpu_capacity.install(base_class)
    assert cpu_capacity.SCHEDULER_METHOD in report["skipped"]
    assert "neither ours nor the fork's own" in (report["reason"] or "")
    assert getattr(base_class, cpu_capacity.SCHEDULER_METHOD) is foreign
    assert cpu_capacity.section()["scheduler_installed"] is False


def test_a_renamed_method_is_refused_not_invented(cap_state, base_class):
    delattr(base_class, cpu_capacity.SCHEDULER_METHOD)
    report = cpu_capacity.install(base_class)
    assert cpu_capacity.SCHEDULER_METHOD in report["skipped"]
    assert "refusing to invent one" in (report["reason"] or "")
    # The worker half still installs: the two halves are independent.
    assert cpu_capacity._state["worker_installed"] is True


def test_a_non_callable_target_is_refused(cap_state, base_class):
    setattr(base_class, cpu_capacity.WORKER_METHOD, "a string")
    report = cpu_capacity.install(base_class)
    assert cpu_capacity.WORKER_METHOD in report["skipped"]
    assert "not callable" in (report["reason"] or "")


def test_install_never_lets_a_broken_class_break_the_boot(cap_state):
    class HostileMeta(type):
        def __getattr__(cls, name):  # every attribute read on the CLASS raises
            raise RuntimeError("the class itself is on fire")

    Hostile = HostileMeta("Hostile", (), {})
    report = cpu_capacity.install(Hostile)
    assert report["installed"] is False
    assert "install raised" in (report["reason"] or "")
    # Contained: the failure is a count and a report, not an exception, and not a
    # claim that anything was installed.
    assert cpu_capacity.section()["installed"] is False
    assert snapshot._state["errors"] >= 1


@pytest.fixture
def fake_connector_module(cap_state, monkeypatch):
    """The stub connector base module installed under the fork's module name, so
    `cpu_capacity.install()` finds it exactly as it does in a live process."""
    import sys

    module = types.ModuleType(cpu_capacity.BASE_MODULE)
    exec(FAKE_BASE_SOURCE, module.__dict__)  # noqa: S102
    monkeypatch.setitem(sys.modules, cpu_capacity.BASE_MODULE, module)
    return module


def test_register_installs_the_clamp_in_a_worker_that_never_imports_engine_core(
        enabled, monkeypatch, cap_state, fake_connector_module):
    """THE worker-process bug this ordering exists for.

    `register()` returns early when `vllm.v1.engine.core` is not imported, and a
    worker process never imports it - so if the capacity install sat after that
    return, no worker would ever report its arena, every rank would read as
    "failed to report", and the tier would park nothing forever. The install has
    to happen before the EngineCore lookup.
    """
    import sys

    cls = fake_connector_module.KVConnectorBase_V1
    monkeypatch.delitem(sys.modules, "vllm.v1.engine.core", raising=False)
    engine_side.register()
    assert cpu_capacity._state["installs"] == 1
    assert cpu_capacity.section()["worker_installed"] is True
    assert getattr(cls, cpu_capacity.OWNER_ATTR, None) is None
    assert getattr(getattr(cls, cpu_capacity.WORKER_METHOD),
                   cpu_capacity.OWNER_ATTR) == cpu_capacity.OWNER_TAG


def test_register_still_installs_the_clamp_next_to_the_engine_methods(
        enabled, monkeypatch, cap_state, fake_connector_module):
    import sys

    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core",
                        types.SimpleNamespace(EngineCore=type("EngineCore", (), {})))
    engine_side.register()
    assert cpu_capacity._state["installs"] == 1
    assert cpu_capacity.section()["installed"] is True


def test_register_is_inert_when_the_switch_is_off(monkeypatch, cap_state,
                                                  fake_connector_module):
    import sys

    monkeypatch.delitem(sys.modules, "vllm.v1.engine.core", raising=False)
    monkeypatch.setenv(config.ENV_ENABLE, "0")
    engine_side.register()
    assert cpu_capacity._state["installs"] == 0
    method = fake_connector_module.KVConnectorBase_V1.get_handshake_metadata
    assert getattr(method, cpu_capacity.OWNER_ATTR, None) is None


# --- the worker side --------------------------------------------------------

def test_a_connector_with_no_payload_returns_a_bare_carrier(installed):
    instance = installed()
    instance.worker_handler = Handler(166)
    assert instance.get_handshake_metadata() == {cpu_capacity.CAPACITY_KEY: 166}


def test_a_dict_payload_keeps_every_key_it_had_and_is_not_mutated(installed):
    payload = {"agent_ptr": 1234, "agent_addr": "cudaIpc"}
    instance = installed()
    instance.payload = payload
    instance.worker_handler = Handler(40)
    out = instance.get_handshake_metadata()
    assert out["agent_ptr"] == 1234
    assert out["agent_addr"] == "cudaIpc"
    assert out[cpu_capacity.CAPACITY_KEY] == 40
    assert payload == {"agent_ptr": 1234, "agent_addr": "cudaIpc"}  # untouched


def test_a_typed_payload_is_carried_and_survives_intact(installed):
    class Payload:  # a TypedDict-less connector payload, like NIXL's
        def __init__(self):
            self.rank = 3

    payload = Payload()
    instance = installed()
    instance.payload = payload
    instance.worker_handler = Handler(11)
    out = instance.get_handshake_metadata()
    assert out[cpu_capacity.PAYLOAD_KEY] is payload
    assert cpu_capacity._state["worker_foreign_payload"] == 1
    # ...and the scheduler side hands the ORIGINAL object to the fork's method.
    manager = Manager(50)
    instance.scheduler_manager = manager
    handshake(instance, {(0, 0): out})
    assert instance.received[(0, 0)] is payload


def test_a_scheduler_role_connector_says_nothing_at_all(installed):
    instance = installed()
    instance.payload = None
    assert instance.get_handshake_metadata() is None  # no worker_handler
    assert cpu_capacity._state["worker_reports"] == 0


def test_the_producers_own_exception_still_propagates(installed, cap_state):
    """A connector that raises must still raise through us.

    Swallowing it would hide a real fault behind a handshake that merely looks
    empty, and the engine would boot into a tier it believes is verified.
    """
    instance = installed()
    instance.worker_handler = Handler(5)

    def raiser(self):
        raise RuntimeError("the connector itself failed")

    setattr(installed, cpu_capacity.WORKER_METHOD,
            cpu_capacity._make_worker_wrapper(raiser))
    with pytest.raises(RuntimeError, match="connector itself failed"):
        instance.get_handshake_metadata()
    assert cpu_capacity._state["worker_fallbacks"] == 1
    assert snapshot._state["errors"] >= 1


def test_an_unregistered_worker_reports_zero_not_nothing(installed):
    instance = installed()
    instance.worker_handler = Handler(0)  # worker.py:52, register never ran
    assert instance.get_handshake_metadata() == {cpu_capacity.CAPACITY_KEY: 0}


# --- the scheduler side: the cap --------------------------------------------

def test_min_across_ranks_is_the_cap(connector):
    content = {(0, 0): carrier(187), (0, 1): carrier(187),
               (1, 0): carrier(166), (1, 1): carrier(166)}
    handshake(connector, content)
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["chosen_cap"] == 166
    assert section["cap_source"] == "min-across-ranks"
    assert section["zero_mode"] is False
    assert section["clamp_applied"] is True
    assert section["per_rank"] == {"0,0": 187, "0,1": 187,
                                   "1,0": 166, "1,1": 166}
    assert {"key": "1,0", "pp_rank": 1, "tp_rank": 0,
            "num_cpu_blocks": 166} in section["per_rank_list"]


def test_a_disagreement_never_picks_the_max(connector):
    handshake(connector, {(0, 0): carrier(100), (1, 0): carrier(40),
                          (1, 1): carrier(77)})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["chosen_cap"] == 40
    assert section["chosen_cap"] != max(100, 40, 77)


def test_the_cap_is_never_raised_above_the_schedulers_own_view(connector):
    # Every rank swears it can hold 5000; the aggregate view only ever sized 200.
    handshake(connector, {(0, 0): carrier(5000), (1, 0): carrier(5000)})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["chosen_cap"] == 200
    assert section["cap_source"] == "min-across-ranks-capped-by-scheduler-view"


def test_a_rank_reporting_nothing_parks_nothing(connector):
    """A rank that handed over a payload but no capacity is a rank we cannot
    trust, and the safe reading of "cannot trust" is park nothing. (It is keyed
    at pp_rank 0 because a REAL payload at pp_rank > 0 makes this fork's own
    base implementation raise - see test_the_originals_own_exception_..., which
    is the faithful test for that shape.)"""
    handshake(connector, {(0, 0): carrier(187), (0, 1): {"agent_ptr": 1}})
    manager = connector.scheduler_manager
    section = cpu_capacity.section(manager)
    assert section["zero_mode"] is True
    assert section["cap_source"] == "fail-safe-zero"
    assert section["chosen_cap"] == 0
    assert section["entries_without_capacity"] == ["(0, 1)"]
    assert manager.cpu_block_pool.get_num_free_blocks() == 0
    assert manager.num_cpu_blocks == 0


def test_a_rank_reporting_zero_parks_nothing(connector):
    handshake(connector, {(0, 0): carrier(187), (1, 0): carrier(0)})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["zero_mode"] is True
    assert "reported 0 blocks" in section["zero_reason"]
    assert connector.scheduler_manager.cpu_block_pool.get_num_free_blocks() == 0


def test_an_explicit_none_entry_counts_as_a_rank_that_said_nothing(connector):
    """`core.py:299` skips a worker that returned None, so this should not
    happen - but reading it as "that rank does not exist" would be the one
    mistake the split is not allowed to make."""
    handshake(connector, {(0, 0): carrier(166), (0, 1): None})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["zero_mode"] is True
    assert section["entries_without_capacity"] == ["(0, 1)"]
    assert connector.scheduler_manager.cpu_block_pool.get_num_free_blocks() == 0


def test_a_wrapped_typed_payload_is_reported_as_such(connector):
    class Payload:
        kind = "typed"

    payload = Payload()
    handshake(connector, {(0, 0): {cpu_capacity.CAPACITY_KEY: 120,
                                   cpu_capacity.PAYLOAD_KEY: payload},
                          (0, 1): carrier(110)})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["handshake_wrapped_payload"] is True
    assert section["chosen_cap"] == 110
    assert connector.received == {(0, 0): payload}


def test_a_payload_that_cannot_be_read_is_handed_over_and_parked(connector):
    """`metadata.items()` raising is the shape we cannot interpret: the fork gets
    the object untouched (we cannot improve on it), the clamp goes to zero, and
    nothing escapes into the engine."""
    class Unreadable(dict):
        def items(self):
            raise RuntimeError("not a mapping today")

    handshake(connector, Unreadable())
    assert isinstance(connector.received, Unreadable)
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["zero_mode"] is True
    assert section["clamp_applied"] is True
    assert section["entries_without_capacity"] == ["<unreadable payload>"]
    assert cpu_capacity._state["fallbacks"] >= 1
    assert snapshot._state["errors"] >= 1
    assert connector.scheduler_manager.cpu_block_pool.get_num_free_blocks() == 0


def test_no_reports_at_all_parks_nothing(connector):
    handshake(connector, {})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["zero_mode"] is True
    assert "no rank reported" in section["zero_reason"]
    assert connector.scheduler_manager.cpu_block_pool.get_num_free_blocks() == 0


def test_a_key_shape_we_cannot_read_parks_nothing(connector):
    handshake(connector, {0: carrier(166), "1,0": carrier(166)})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert section["zero_mode"] is True
    assert section["entries_unrecognized"] == 2


def test_a_bare_capacity_report_is_not_made_into_a_payload_entry(connector):
    """base.py:725 RAISES on a pp_rank > 0 entry, and this connector must never
    see one it did not produce: the carriers are ours, so they must contribute a
    capacity and NO entry. Without that the plugin would be the boot crash."""
    handshake(connector, {(0, 0): carrier(166), (1, 0): carrier(150)})
    assert connector.received == {}


def test_the_original_payload_reaches_the_original_method_intact(installed):
    class Real(installed):
        def get_handshake_metadata(self):
            return {"agent_ptr": 99}

    manager = Manager(200)
    instance = Real()
    instance.worker_handler = Handler(150)  # a payload AND a capacity, per rank
    instance.scheduler_manager = manager
    entry = instance.get_handshake_metadata()
    handshake(instance, {(0, 0): entry, (0, 1): entry})
    assert instance.received == {(0, 0): {"agent_ptr": 99},
                                 (0, 1): {"agent_ptr": 99}}
    assert cpu_capacity.CAPACITY_KEY not in json.dumps(
        {"0,0": instance.received[(0, 0)]}, default=str)
    assert instance.received[(0, 0)].get(cpu_capacity.CAPACITY_KEY) is None


def test_the_originals_own_exception_still_propagates(installed, cap_state):
    class Refusing(installed):
        def set_xfer_handshake_metadata_pp_aware(self, metadata):
            return installed.set_xfer_handshake_metadata_pp_aware(self, metadata)

    manager = Manager(200)
    instance = Refusing()
    instance.scheduler_manager = manager
    # Force the fork's own pp_rank>0 refusal with a REAL payload on rank 1.
    with pytest.raises(ValueError, match="pp_rank > 0"):
        handshake(instance, {(1, 0): {"agent_ptr": 1}})
    section = cpu_capacity.section(manager)
    assert "ValueError" in section["original_call_error"]
    # The clamp is still applied before the raise: a caller that survives the
    # handshake must not survive it uncapped.
    assert section["clamp_applied"] is True
    assert manager.cpu_block_pool.get_num_free_blocks() == 0
    assert snapshot._state["errors"] >= 1


def test_a_connector_that_is_not_ours_to_clamp_is_passed_through(installed,
                                                                 cap_state):
    """A connector with no offload manager (another connector's handshake) gets
    its payload untouched and no clamp, and that is NOT reported as a failure:
    `clamp_failed_open` is reserved for a manager we meant to clamp and could
    not."""
    instance = installed()
    instance.scheduler_manager = None
    handshake(instance, {(0, 0): {"agent_ptr": 5}})
    assert instance.received == {(0, 0): {"agent_ptr": 5}}
    section = cpu_capacity.section(None)
    assert section["clamp_applied"] is False
    assert section["zero_mode"] is False
    assert section["clamp_failed_open"] is False


def test_a_second_handshake_on_a_clamped_manager_changes_nothing(connector):
    handshake(connector, {(0, 0): carrier(166), (1, 0): carrier(166)})
    manager = connector.scheduler_manager
    free = manager.cpu_block_pool.get_num_free_blocks()
    handshake(connector, {(0, 0): carrier(999), (1, 0): carrier(999)})
    assert manager.cpu_block_pool.get_num_free_blocks() == free
    assert manager.num_cpu_blocks == 166
    assert cpu_capacity.section(manager)["chosen_cap"] == 166


# --- the clamp itself: the surgery, not a promise ---------------------------

def test_the_clamp_makes_the_pools_own_counters_agree_with_the_cap(connector):
    handshake(connector, {(0, 0): carrier(187), (1, 0): carrier(166)})
    manager = connector.scheduler_manager
    pool = manager.cpu_block_pool
    cap = 166
    assert manager.num_cpu_blocks == cap
    assert manager.cpu_kv_cache_config.num_blocks == cap
    assert pool.num_gpu_blocks == cap
    # The null block (id 0) is not allocatable, so a cap of 166 leaves 165 free.
    assert pool.get_num_free_blocks() == cap - 1
    section = cpu_capacity.section(manager)
    assert section["cpu_blocks_withheld"] == 200 - cap
    assert section["pool_free_after"] == cap - 1
    assert section["manager_num_cpu_blocks_before"] == 200


def test_id_cap_minus_one_is_allowed_and_id_cap_is_never_emitted(connector):
    handshake(connector, {(0, 0): carrier(166)})
    pool = connector.scheduler_manager.cpu_block_pool
    drained = pool.get_new_blocks(pool.get_num_free_blocks())
    ids = [block.block_id for block in drained]
    assert ids == list(range(1, 166))
    assert 166 not in ids
    assert max(ids) == 165
    with pytest.raises(ValueError, match="free blocks from the pool"):
        pool.get_new_blocks(1)
    assert cpu_capacity.section()["allocations_refused_by_cap"] == 1


def test_zero_mode_leaves_only_the_null_block_and_never_raises_on_its_own(
        connector):
    handshake(connector, {})
    manager = connector.scheduler_manager
    pool = manager.cpu_block_pool
    assert pool.get_num_free_blocks() == 0
    assert pool.num_gpu_blocks == 1
    assert manager.cpu_kv_cache_config.num_blocks == 1
    # The tier checks free space before every allocation (manager.py:698,
    # manager.py:600, manager.py:834), so zero free blocks is a no-op step.
    if pool.get_num_free_blocks() > 0:  # pragma: no cover - the guard is the tier's
        raise AssertionError("zero mode must leave nothing allocatable")
    assert pool.null_block.is_null is True
    assert pool.null_block.block_id == 0


def test_the_pool_keeps_every_block_object_so_ids_stay_nameable(connector):
    handshake(connector, {(1, 0): carrier(10)})
    pool = connector.scheduler_manager.cpu_block_pool
    assert len(pool.blocks) == 200
    assert pool.blocks[150].block_id == 150  # pool.blocks[bid] cannot IndexError


def test_a_cached_block_above_the_cap_is_evicted_before_it_is_withdrawn(
        connector):
    pool = connector.scheduler_manager.cpu_block_pool
    pool.cache(pool.blocks[190], "hash-190")
    pool.cache(pool.blocks[5], "hash-5")
    handshake(connector, {(1, 0): carrier(10)})
    assert "hash-190" not in pool.cached_block_hash_to_block
    assert pool.blocks[190].block_hash is None
    assert "hash-5" in pool.cached_block_hash_to_block  # below the cap, kept


def test_the_gpu_pool_never_acquires_anything_from_this_plugin(connector):
    handshake(connector, {(1, 0): carrier(50)})
    gpu = connector.scheduler_manager._gpu_block_pool
    assert cpu_capacity.POOL_MARKER not in vars(gpu)
    assert gpu.get_num_free_blocks() == 499
    assert gpu.num_gpu_blocks == 500
    # And the clamp is instance surgery on the CPU pool: the CLASS is untouched,
    # so the GPU pool's own get_new_blocks is the fork's.
    assert type(gpu).get_new_blocks is type(
        connector.scheduler_manager.cpu_block_pool).get_new_blocks


def test_an_allocated_block_above_the_cap_is_reported_not_hidden(connector):
    pool = connector.scheduler_manager.cpu_block_pool
    allocated = pool.get_new_blocks(1)[0]
    assert allocated.block_id == 1
    allocated.ref_cnt = 3  # pretend the tier already handed it out
    handshake(connector, {(1, 0): carrier(1)})
    section = cpu_capacity.section(connector.scheduler_manager)
    # id 1 is above the cap of 1 and is NOT in the free queue: it is counted.
    assert section["blocks_still_referenced_above_cap"] == 1


def test_a_manager_that_is_someone_elses_clamped_is_not_double_clamped(connector):
    manager = connector.scheduler_manager
    setattr(manager, cpu_capacity.MANAGER_MARKER, "another-plugin/9")
    handshake(connector, {(0, 0): carrier(166)})
    section = cpu_capacity.section(manager)
    assert section["clamp_applied"] is False
    assert section["clamp_failed_open"] is True


def test_a_pool_that_cannot_be_clamped_says_so_instead_of_claiming_zero_mode(
        connector, cap_state):
    """The one failure mode a report must never blur: a pool we could not clamp
    is `clamp_failed_open`, NOT `zero_mode`. "Parked nothing" would be a lie, and
    a lie here is an operator enabling a tier that can still overrun."""
    manager = connector.scheduler_manager
    empty = Pool.__new__(Pool)  # BlockPool cannot have 0 blocks; the reader must
    empty.blocks = []           # still survive one that somehow does
    empty.num_gpu_blocks = 0
    empty.free_block_queue = FreeQueue([])
    empty.get_num_free_blocks = lambda: 0
    manager.cpu_block_pool = empty
    handshake(connector, {(0, 0): carrier(166)})
    section = cpu_capacity.section(manager)
    assert section["clamp_applied"] is False
    assert section["zero_mode"] is False
    assert section["clamp_failed_open"] is True


def test_a_pool_that_raises_leaves_the_handshake_alive_and_says_so(installed,
                                                                   cap_state):
    class ExplodingPool(Pool):
        def evict_blocks(self, block_ids):
            raise RuntimeError("the pool is on fire")

    manager = Manager(50)
    manager.cpu_block_pool = ExplodingPool(50)
    instance = installed()
    instance.scheduler_manager = manager
    handshake(instance, {(0, 0): carrier(40)})  # must not raise
    section = cpu_capacity.section(manager)
    assert section["clamp_applied"] is False
    assert section["clamp_failed_open"] is True
    assert cpu_capacity._state["fallbacks"] >= 1
    assert snapshot._state["errors"] >= 1


# --- the last-resort adoption ------------------------------------------------

def test_adoption_parks_nothing_when_the_handshake_never_ran(cap_state):
    manager = Manager(200)
    engine = Engine(types.SimpleNamespace(scheduler_manager=manager))
    report = cpu_capacity.adopt_if_missed(engine)
    assert report["adopted"] is True
    assert manager.cpu_block_pool.get_num_free_blocks() == 0
    assert manager.num_cpu_blocks == 0
    section = cpu_capacity.section(manager)
    assert section["zero_mode"] is True
    assert section["adopted_by_snapshot"] is True
    assert section["zero_reason"].startswith("adopted from the read path")


def test_adoption_is_a_no_op_after_a_real_handshake(installed, cap_state):
    manager = Manager(200)
    instance = installed()
    instance.scheduler_manager = manager
    handshake(instance, {(0, 0): carrier(150), (1, 0): carrier(140)})
    engine = Engine(types.SimpleNamespace(scheduler_manager=manager))
    report = cpu_capacity.adopt_if_missed(engine)
    assert report["adopted"] is False
    assert "already ran" in report["reason"]
    assert manager.cpu_block_pool.get_num_free_blocks() == 139


def test_adoption_declines_an_engine_without_a_manager(cap_state):
    report = cpu_capacity.adopt_if_missed(Engine(None))
    assert report["adopted"] is False
    assert "no SimpleCPUOffloadScheduler" in report["reason"]
    assert cpu_capacity._state["zero_mode"] is False


def test_adoption_never_reclamps_a_manager_it_already_marked(cap_state):
    """The per-manager marker, not just the process counter: a second engine in
    one process must not re-withdraw from a pool that is already settled."""
    manager = Manager(200)
    engine = Engine(types.SimpleNamespace(scheduler_manager=manager))
    cpu_capacity.adopt_if_missed(engine)
    manager.cpu_block_pool.free_block_queue.append_n(
        manager.cpu_block_pool.blocks[1:])  # an operator "unclamped" it
    cpu_capacity._state["clamp_applied"] = False  # as if a new engine appeared
    cpu_capacity._state["handshake_seen"] = False
    report = cpu_capacity.adopt_if_missed(engine)
    assert report["adopted"] is False
    assert "clamp marker" in report["reason"]
    assert manager.cpu_block_pool.get_num_free_blocks() == 199


def test_adoption_of_an_engine_that_raises_is_data(cap_state):
    class Hostile:
        @property
        def scheduler(self):
            raise RuntimeError("no scheduler for you")

    report = cpu_capacity.adopt_if_missed(Hostile())
    assert report["adopted"] is False
    assert "adoption raised" in report["reason"]
    assert snapshot._state["errors"] >= 1


# --- the report -------------------------------------------------------------

REQUIRED_FIELDS = (
    "installed", "reason", "worker_installed", "scheduler_installed",
    "handshake_seen", "per_rank", "per_rank_list", "entries_without_capacity",
    "chosen_cap", "cap_source", "clamp_applied", "zero_mode", "zero_reason",
    "adopted_by_snapshot", "original_call_error", "cpu_blocks_withheld",
    "pool_free_after", "manager_num_cpu_blocks_after",
    "allocations_refused_by_cap", "clamp_failed_open", "live_manager_clamped",
    "live_cpu_pool", "error",
)


def test_the_section_reports_every_required_field_from_the_start():
    missing = [f for f in REQUIRED_FIELDS if f not in cpu_capacity.section()]
    assert not missing


def test_the_section_still_reports_every_field_after_a_failure(connector):
    handshake(connector, {(0, 0): carrier(166)})
    section = cpu_capacity.section(connector.scheduler_manager)
    assert not [f for f in REQUIRED_FIELDS if f not in section]


def test_the_degraded_section_has_every_key_by_construction():
    healthy = cpu_capacity.section(Manager(4))
    degraded = cpu_capacity.empty_section("the reader raised")
    assert set(healthy) == set(degraded)
    assert degraded["error"] == "the reader raised"


def test_the_section_survives_a_manager_that_raises_on_read(cap_state):
    class HostilePool:
        def __getattr__(self, name):
            raise RuntimeError("the pool is hostile")

    manager = types.SimpleNamespace(cpu_block_pool=HostilePool())
    section = cpu_capacity.section(manager)
    assert section["live_cpu_pool"] is None
    assert snapshot._state["errors"] >= 1


def test_the_section_encodes_for_the_wire(connector):
    handshake(connector, {(0, 0): carrier(187), (1, 0): carrier(166)})
    section = cpu_capacity.section(connector.scheduler_manager)
    # A tuple key would raise here, and the wire is the only thing standing
    # between this section and a msgpack failure inside the engine.
    assert json.loads(json.dumps(snapshot.json_safe({"cpu_capacity": section})))


def test_the_snapshot_carries_the_section(cap_state):
    manager = Manager(20)
    connector = types.SimpleNamespace(scheduler_manager=manager)
    view = snapshot.snapshot_for_engine(Engine(connector), _NullHolder())
    section = view["cpu_capacity"]
    assert section["per_rank"] == {}
    assert section["installed"] is False
    assert section["live_manager_clamped"] is False
    assert section["live_cpu_pool"]["free_blocks"] == 19
    assert not [f for f in REQUIRED_FIELDS if f not in section]


def test_a_broken_capacity_section_is_data_not_a_missing_field(cap_state,
                                                               monkeypatch):
    def explode(manager=None):
        raise RuntimeError("the section is on fire")

    monkeypatch.setattr(cpu_capacity, "section", explode)
    view = snapshot.snapshot_for_engine(Engine(types.SimpleNamespace(
        scheduler_manager=Manager(8))), _NullHolder())
    section = view["cpu_capacity"]
    assert section["error"].startswith("RuntimeError")
    assert not [f for f in REQUIRED_FIELDS if f not in section]


class _NullHolder:
    def ensure_loaded(self):
        return {"ok": False, "error": "no policy in this test", "traceback": None}

    @property
    def decide(self):
        return None


# --- discipline -------------------------------------------------------------

def test_the_module_pulls_in_nothing_heavy():
    import ast
    from pathlib import Path

    source = Path(cpu_capacity.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    pulled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pulled.add(node.module.split(".")[0])
    heavy = {"torch", "vllm", "cuda", "numpy", "triton", "numba", "cupy", "psutil"}
    assert not pulled & heavy, sorted(pulled & heavy)


def test_the_clamp_path_never_imports_vllm(installed, cap_state):
    import sys

    manager = Manager(50)
    instance = installed()
    instance.scheduler_manager = manager
    handshake(instance, {(0, 0): carrier(41), (1, 0): carrier(38)})
    cpu_capacity.adopt_if_missed(Engine(types.SimpleNamespace(
        scheduler_manager=manager)))
    assert "vllm" not in sys.modules
    assert "torch" not in sys.modules
