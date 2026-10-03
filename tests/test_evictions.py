"""The LRU-eviction counter: the arena's steady state, made countable.

Nothing here imports vllm. The pool stub is `exec`'d into a module NAMED
`vllm.v1.core.block_pool` so its methods report the fork's `__module__` - which
is the check `evictions.arm_pool` uses to refuse wrapping anything that is not
the engine's own function. The stub models the one contract that matters, from
the read-only fork at block_pool.py:726-750: `_maybe_evict_cached_block` returns
True IF AND ONLY IF a parked entry was actually discarded, resetting the hash.

Each test is a real behavior, not a snapshot of a dict: a parked block evicted
under pressure, a non-parked block that must NOT count, the group a discard is
attributed to, and every fail-closed path that turns a missing signal into a
reason instead of a lying zero.
"""

from __future__ import annotations

import types

import pytest

from kvtier_dev_plugin import engine_side, evictions, snapshot

# --- the fork's BlockPool, faithfully shaped (no vllm import) ---------------

FAKE_POOL_SOURCE = '''
"""Shaped after vllm/v1/core/block_pool.py (read-only fork), the eviction path."""


class KVCacheBlock:
    """block_pool.py: block_id / block_hash / reset_hash()."""

    def __init__(self, block_id, block_hash=None):
        self.block_id = block_id
        self.block_hash = block_hash

    def reset_hash(self):
        self.block_hash = None


class BlockPool:
    """The eviction contract, block_pool.py:726-750 + :588-606.

    `_maybe_evict_cached_block` returns True exactly when a hash was dropped,
    and `_remove_cached_block_hashes` drops the primary AND every extra hash
    registered for the block id, then resets. That is the whole reason the
    wrapper must PRE-READ the groups before the original resets them.
    """

    def __init__(self):
        self.cached_block_hash_to_block = {}
        self.cached_block_hashes_by_block = {}
        self.evict_calls = 0

    def park(self, block, primary, secondary=()):
        block.block_hash = primary
        self.cached_block_hash_to_block[primary] = block
        for extra in secondary:
            self.cached_block_hash_to_block[extra] = block
        if secondary:
            self.cached_block_hashes_by_block[block.block_id] = list(secondary)

    def _remove_cached_block_hashes(self, block):
        hashes = []
        if block.block_hash is not None:
            hashes.append(block.block_hash)
        hashes.extend(self.cached_block_hashes_by_block.pop(block.block_id, ()))
        if not hashes:
            return []
        removed = []
        for h in hashes:
            if self.cached_block_hash_to_block.pop(h, block.block_id) is not None:
                removed.append(h)
        block.reset_hash()
        return removed

    def _maybe_evict_cached_block(self, block):
        self.evict_calls += 1
        removed = self._remove_cached_block_hashes(block)
        if not removed:
            return False
        return True
'''


def make_pool_module():
    module = types.ModuleType(evictions.TARGET_MODULE)
    exec(FAKE_POOL_SOURCE, module.__dict__)  # noqa: S102 - our own source string
    return module


@pytest.fixture
def pool_type():
    return make_pool_module().BlockPool


@pytest.fixture
def block_type():
    return make_pool_module().KVCacheBlock


@pytest.fixture
def quiet():
    class QuietLogger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    saved = dict(evictions._state), dict(snapshot._state)
    evictions.reset_state_for_tests()
    snapshot.attach_logger(QuietLogger())
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0})
    yield
    snapshot.attach_logger(None)
    evictions._state.clear()
    evictions._state.update(saved[0])
    snapshot._state.clear()
    snapshot._state.update(saved[1])


# group ids are the last 4 big-endian bytes of a BlockHashWithGroupId.
def key(tag: int, group: int) -> bytes:
    return tag.to_bytes(8, "big") + group.to_bytes(4, "big")


def armed_pool(pool_type):
    pool = pool_type()
    report = evictions.arm_pool(pool)
    assert report["armed"] is True, report
    return pool


# --- the honest counting path -------------------------------------------------

def test_a_parked_block_evicted_counts_once_total_and_in_its_group(
        quiet, pool_type, block_type):
    pool = armed_pool(pool_type)
    block = block_type(7, key(500, 5))
    pool.park(block, key(500, 5))

    assert pool._maybe_evict_cached_block(block) is True

    state = evictions.counters()
    assert state["evictions_total"] == 1
    assert state["evicted_spans_per_group"] == {"5": 1}
    assert state["evicted_keys_per_group"] == {"5": 1}
    assert state["unknown_group_evictions"] == 0


def test_a_block_that_was_not_parked_is_not_an_eviction(quiet, pool_type, block_type):
    # The engine calls this on EVERY block handed out of the free queue; most are
    # already unhashed (refilled), and those must never be counted.
    pool = armed_pool(pool_type)
    block = block_type(9, block_hash=None)
    assert pool._maybe_evict_cached_block(block) is False
    assert evictions.counters()["evictions_total"] == 0


def test_a_coarse_fa_eviction_also_counts_its_fine_tick_keys(quiet, pool_type,
                                                              block_type):
    # THE transient question, made visible: evicting group 5's coarse block also
    # drops its secondary fine-tick keys, so group 5's KEY evictions run ahead of
    # its SPAN evictions. This is the balance that shows the arena is being eaten
    # by fine-graining entries the joint hit can never use.
    pool = armed_pool(pool_type)
    block = block_type(3, key(800, 5))
    pool.park(block, key(800, 5), secondary=[key(801, 5), key(802, 5)])

    assert pool._maybe_evict_cached_block(block) is True
    state = evictions.counters()
    assert state["evicted_spans_per_group"] == {"5": 1}       # one block
    assert state["evicted_keys_per_group"] == {"5": 3}        # primary + 2 fine
    # secondary total is a derivation of the two maps, kept from drifting:
    assert evictions._secondary_keys_total() == 2


def test_per_group_attribution_across_groups_and_secondary_of_other_group(
        quiet, pool_type, block_type):
    pool = armed_pool(pool_type)
    # A group-0 block aliased under a group-5 key: span to group 0, keys split.
    block = block_type(1, key(10, 0))
    pool.park(block, key(10, 0), secondary=[key(11, 5)])
    assert pool._maybe_evict_cached_block(block) is True
    state = evictions.counters()
    assert state["evicted_spans_per_group"] == {"0": 1}
    assert state["evicted_keys_per_group"] == {"0": 1, "5": 1}


def test_total_is_monotonic_across_many_evictions(quiet, pool_type, block_type):
    pool = armed_pool(pool_type)
    base = evictions.counters()["evictions_total"]
    for i in range(50):
        blk = block_type(i, key(2000 + i, 5))
        pool.park(blk, key(2000 + i, 5))
        assert pool._maybe_evict_cached_block(blk) is True
    assert evictions.counters()["evictions_total"] == base + 50
    assert evictions.counters()["evicted_spans_per_group"] == {"5": 50}


# --- the fail-closed paths ----------------------------------------------------

def test_an_unwrapped_pool_reports_null_not_zero(quiet):
    # The whole point: an unwrapped pool must NOT read as "the LRU evicted none".
    section = evictions.section(None)
    assert section["available"] is False
    assert section["live_pool_watched"] is False
    assert section["evictions_total"] is None
    assert section["evicted_spans_per_group"] is None
    assert section["reason"] and "null" in section["reason"]
    assert section["pure_read"] is False  # a wrap, said loudly


def test_unparseable_group_lands_in_unknown_never_a_guess(quiet, pool_type,
                                                           block_type):
    pool = armed_pool(pool_type)
    # A key too short to carry a group id: honest, not a fabricated group 0.
    block = block_type(4, b"\x00\x01")
    pool.park(block, b"\x00\x01")
    assert pool._maybe_evict_cached_block(block) is True
    state = evictions.counters()
    assert state["evictions_total"] == 1
    assert state["evicted_spans_per_group"] == {"unknown": 1}
    assert state["unknown_group_evictions"] == 1


def test_the_engines_own_exception_propagates_unchanged(quiet, pool_type):
    # A discard is the engine's own business: if the underlying raises, the
    # wrapper must re-raise it, not swallow it into a counter.
    pool = pool_type()

    class Boom(Exception):
        pass

    def explode(_block):
        raise Boom("engine allocator failure")

    wrapper = evictions._make_wrapper(explode, pool)
    with pytest.raises(Boom):
        wrapper(None)


def test_snapshot_view_carries_the_eviction_section(quiet, pool_type):
    # The build_view path must attach tier["evictions"] with the declared keys.
    mgr = _Mgr()
    mgr.num_cpu_blocks = 1331
    mgr._store_event_counter = 1441
    engine = _Engine(_Conn(mgr))
    view = snapshot.build_view(engine)
    section = view["tier"]["evictions"]
    assert set(section) == set(evictions.EXPECTED_FIELDS)
    # not armed here -> fail closed, not zero
    assert section["available"] is False
    assert section["evictions_total"] is None
    assert section["schema_version"] == 2


def test_counting_never_blocks_an_eviction(quiet, pool_type, block_type):
    # A group decode that raises must not stop the discard: the block still gets
    # evicted (True returned), the total still counts, and the miss is admitted
    # rather than invented - never a fabricated group and never a blocked eviction.
    pool = armed_pool(pool_type)

    class BadIndex(dict):
        def get(self, *a, **k):
            raise RuntimeError("corrupt secondary index")

    pool.cached_block_hashes_by_block = BadIndex()
    block = block_type(6, key(9, 5))
    pool.park(block, key(9, 5))
    assert pool._maybe_evict_cached_block(block) is True   # eviction happened
    state = evictions.counters()
    assert state["evictions_total"] == 1                    # still counted
    assert state["fallbacks"] >= 1                          # the miss is admitted
    assert state["unknown_group_evictions"] == 1            # not attributed


# --- adoption: instance-level, idempotent, refuses a foreign owner ------------

def test_adoption_is_idempotent_and_never_nests(quiet, pool_type, block_type):
    pool = pool_type()
    assert evictions.arm_pool(pool)["armed"] is True
    assert evictions.arm_pool(pool)["replaced"] is True
    assert evictions.arm_pool(pool)["replaced"] is True
    # A re-adopt re-binds the ORIGINAL, so one eviction counts once, not 3x.
    block = block_type(1, key(1, 5))
    pool.park(block, key(1, 5))
    assert pool._maybe_evict_cached_block(block) is True
    assert evictions.counters()["evictions_total"] == 1
    assert evictions.counters()["reinstalls"] == 2


def test_a_foreign_owner_is_refused_never_overwritten(quiet, pool_type):
    pool = pool_type()

    def foreign(_block):
        return False

    foreign.__module__ = evictions.TARGET_MODULE
    setattr(foreign, evictions.OWNER_ATTR, "some-other-plugin/1")
    setattr(pool, evictions.TARGET_METHOD, foreign)
    report = evictions.arm_pool(pool)
    assert report["armed"] is False
    assert "another owner" in report["reason"]
    # the foreign method is untouched:
    assert getattr(pool, evictions.TARGET_METHOD) is foreign


def test_a_function_not_from_the_fork_is_refused(quiet, pool_type):
    pool = pool_type()

    def not_the_engine(_block):
        return False

    not_the_engine.__module__ = "some.random.module"
    setattr(pool, evictions.TARGET_METHOD, not_the_engine)
    report = evictions.arm_pool(pool)
    assert report["armed"] is False
    assert "neither ours nor the fork" in report["reason"]


def test_uninstall_restores_the_pools_own_method(quiet, pool_type, block_type):
    pool = pool_type()
    own = getattr(pool, evictions.TARGET_METHOD)
    assert evictions.arm_pool(pool)["armed"] is True
    assert getattr(pool, evictions.TARGET_METHOD) is not own
    report = evictions.uninstall(pool)
    assert report["uninstalled"] is True
    # after uninstall a real eviction still works but is no longer counted
    evictions.reset_state_for_tests()
    block = block_type(1, key(1, 5))
    pool.park(block, key(1, 5))
    assert pool._maybe_evict_cached_block(block) is True
    assert evictions.counters()["evictions_total"] == 0


# --- adoption reaches only the CPU pool, never the GPU pool -------------------

class _Mgr:
    def __init__(self):
        self.cpu_block_pool = None
        self._gpu_block_pool = None


class _Conn:
    def __init__(self, mgr):
        self.scheduler_manager = mgr


class _Sched:
    def __init__(self, conn):
        self.connector = conn


class _Engine:
    def __init__(self, conn):
        self.scheduler = _Sched(conn)


def test_adopt_reaches_only_the_cpu_pool(quiet, pool_type):
    mgr = _Mgr()
    mgr.cpu_block_pool = pool_type()
    mgr._gpu_block_pool = pool_type()
    engine = _Engine(_Conn(mgr))
    report = evictions.adopt_if_missed(engine)
    assert report["armed"] is True
    # GPU pool is the SAME class but a different instance: it must be untouched.
    gpu_method = getattr(mgr._gpu_block_pool, evictions.TARGET_METHOD)
    assert getattr(gpu_method, evictions.OWNER_ATTR, None) != evictions.OWNER_TAG


def test_adopt_is_a_noop_when_already_armed(quiet, pool_type):
    mgr = _Mgr()
    mgr.cpu_block_pool = pool_type()
    engine = _Engine(_Conn(mgr))
    assert evictions.adopt_if_missed(engine)["armed"] is True
    second = evictions.adopt_if_missed(engine)
    assert second["armed"] is True and second["reason"] == "already armed"
    assert evictions.counters()["installs"] == 1


def test_adopt_reports_a_missing_pool_as_a_reason(quiet):
    engine = _Engine(_Conn(_Mgr()))   # cpu_block_pool is None
    report = evictions.adopt_if_missed(engine)
    assert report["armed"] is False
    assert "no cpu_block_pool" in report["reason"]


# --- the section rides the snapshot view --------------------------------------

def test_the_degraded_section_has_every_key_by_construction(quiet):
    healthy = evictions.section(None)
    degraded = evictions.empty_section("boom")
    assert set(healthy) == set(degraded)


def test_engine_side_wires_adoption():
    # The snapshot verb must arm the counter on the live pool, exactly as it arms
    # cpu_capacity: a real regression that a rename of the call site can't hide.
    import inspect
    src = inspect.getsource(engine_side._make_snapshot_method)
    assert "evictions.adopt_if_missed(self)" in src
