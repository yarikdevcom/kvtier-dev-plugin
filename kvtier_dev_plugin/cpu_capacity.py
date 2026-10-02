"""The per-rank CPU arena capacity clamp: what the scheduler may legally emit.

THE DEFECT THIS EXISTS TO CLOSE (measured on the live TP2xPP2 config, read out
of the read-only fork at HEAD 10b8e07)
-----------------------------------------------------------------------------
The scheduler sizes its CPU block-id space from ONE aggregate view:
``manager.py:270-272`` (``_derive_cpu_config``, ``kv_cache_tensors[0].size`` and
``num_blocks``) and assigns ``self.num_cpu_blocks`` at ``manager.py:167``.
Every worker sizes its OWN pinned arena from its OWN stage's bytes/block:
``worker.py:162`` (``num_cpu_blocks = cpu_capacity_bytes // total_bytes_per_block``).
Under ``VLLM_PP_LAYER_PARTITION=25,23`` with the MTP layer on the last stage the
two genuinely differ (11,468,800 B/block on PP0 vs 12,902,400 B/block on PP1),
so at ``--kv-offloading-size 8`` the scheduler emits ids up to 187 while the
last rank's arena holds 166 slots. ``cuda_mem_ops.py:243-249`` then computes
``src_bases[:, None] + ids[None, :] * bpb[:, None]`` with NO bounds check: an
out-of-range id is a write past the end of that rank's pinned host arena.

Why the worker must not just drop the copy: for a STORE a drop is a lost park,
but for a LOAD the scheduler has already counted those tokens as computed, so
dropping the load makes the model attend to garbage KV. The requirement is
therefore that the scheduler NEVER EMITS an out-of-range id.

THE MECHANISM (the fork's own PP-aware handshake, no fork edit)
---------------------------------------------------------------
``EngineCore.__init__`` (``core.py:288-302``) asks every worker for
``get_kv_connector_handshake_metadata()`` (``abstract.py:214-217``,
``gpu_worker.py:868-895``), merges the per-worker dicts into
``content: dict[(pp_rank, tp_rank), metadata]`` and hands it to
``kv_connector.set_xfer_handshake_metadata_pp_aware(content)``
(``base.py:716-731``). It runs after KV-cache registration (so
``worker.py:162`` has already run) and before the busy loop, i.e. before any
traffic. Three wrappers, installed by the existing ``general_plugins`` entry
point, which loads in the EngineCore process AND in every worker:

  1. worker side   - ``KVConnectorBase_V1.get_handshake_metadata``
     (``base.py:447-456``): call the original, then attach this rank's
     ``worker_handler.num_cpu_blocks`` (``worker.py:162``, exposed as the
     connector's ``worker_handler`` at
     ``simple_cpu_offload_connector.py:167``) to whatever it returned.
  2. scheduler side - ``KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware``
     (``base.py:716``): call the original FIRST with exactly the payload the
     fork would have delivered without this plugin, then read every reported
     capacity, take the minimum, and clamp.
  3. the clamp     - one-time surgery on the manager's CPU ``BlockPool`` (see
     ``_apply_cap``), plus the mirrors ``manager.num_cpu_blocks``
     (``manager.py:167``) and ``manager.cpu_kv_cache_config.num_blocks``.

WHY THE WRAPPERS SIT ON THE BASE CLASS, NOT THE CONNECTOR CLASS
---------------------------------------------------------------
Measured in this fork: at the moment ``load_general_plugins()`` runs
(``core.py:215-217``, and ``worker_base.py:286-288`` in a worker) the module
``...kv_connector.v1.base`` is ALREADY imported while
``...v1.simple_cpu_offload_connector`` and ``...simple_kv_offload.manager`` are
NOT - and the connector class is first created later, inside the Scheduler the
EngineCore constructs two lines before it calls the handshake. A connector-class
install would therefore either never land or land after the payload was already
consumed. ``SimpleCPUOffloadConnector`` overrides NEITHER method (it inherits
both), so wrapping the base class catches exactly the subclasses that do not
override, and every subclass that DOES override (``multi_connector.py:503``,
``nixl/connector.py:214``) shadows our wrapper and is untouched.

FAIL-SAFE: UNKNOWN MEANS PARK NOTHING
---------------------------------------------------------------
If no rank reports, a rank reports 0 or nothing at all, or the payload cannot be
interpreted, the clamp is applied with ``cap = 0``: the CPU pool keeps only its
null block, so ``get_num_free_blocks()`` returns 0 and every tier path takes its
own "CPU full" branch (``manager.py:698-701`` eager, ``manager.py:586/600``
lazy, ``manager.py:829/834`` the positional scan - all of them *before* any
``get_new_blocks``, so zero free blocks is a silent no-op, never a raise). A
tier that parks nothing is degraded; a tier that writes out of bounds is a
recall. Uncapped is never a fallback. The clamp can never stop the engine from
booting or serving: the handshake runs once at init, and the tier is optional
at every step.

HOUSE RULES
---------------------------------------------------------------
Stdlib only (``logging``, ``sys``); vllm is reached through ``sys.modules``
exactly like ``engine_side._engine_core_class()``. The clamp runs once, at init.
Nothing here raises into the engine: every entry point catches ``Exception``
(never ``BaseException``) and counts through ``snapshot._error``. Installs are
idempotent, marker-tagged, and refuse to clobber anything this plugin does not
own - the ``engine_side.attach_methods`` / ``store_hook.install`` discipline.
"""

from __future__ import annotations

import logging
import sys

from . import snapshot

logger = logging.getLogger("kvtier_dev_plugin.cpu_capacity")

# Resolved ONLY against an already-imported module (see the docstring).
BASE_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.base"
BASE_CLASS = "KVConnectorBase_V1"
CONNECTOR_MODULE = (
    "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector")

# Where the fork may legitimately define the two methods we wrap: the base class
# that defines them today, and the connector module, which is where the fork
# would move either one if the connector grew its own override. Anything else
# that owns the name is a third party and is refused.
FORK_OWNED_MODULES = (BASE_MODULE, CONNECTOR_MODULE)

WORKER_METHOD = "get_handshake_metadata"
SCHEDULER_METHOD = "set_xfer_handshake_metadata_pp_aware"

# In-band keys of our carrier. Both are namespaced so they cannot collide with a
# connector's own payload keys, and both are stripped before the payload reaches
# the fork's own method.
CAPACITY_KEY = "kvtier_dev_num_cpu_blocks"
PAYLOAD_KEY = "kvtier_dev_handshake_payload"

# Ownership markers, the same idiom as store_hook.
OWNER_ATTR = "__kvtier_dev_cap_owner__"
OWNER_TAG = "kvtier-dev-plugin/capacity-1"
ORIGINAL_ATTR = "__kvtier_dev_cap_original__"

# Markers on objects we mutate, so a second install is a no-op and a foreign one
# is visible.
MANAGER_MARKER = "__kvtier_dev_cap_applied__"
POOL_MARKER = "__kvtier_dev_cap_pool_counter__"

# The fork's own refusal text in BlockPool.get_new_blocks (block_pool.py:676),
# used to attribute a ValueError to a capacity shortfall and nothing else.
_SHORTFALL_TEXT = "free blocks from the pool"

_state = {
    # install
    "installs": 0,
    "reinstalls": 0,
    "skips": 0,
    "skip_reason": None,
    "class": None,
    "worker_installed": False,
    "scheduler_installed": False,
    # worker side (this half only ever fires in a worker process)
    "worker_reports": 0,
    "worker_foreign_payload": 0,
    "worker_fallbacks": 0,
    # scheduler side
    "handshake_calls": 0,
    "handshake_seen": False,
    "handshake_wrapped_payload": False,
    "reported": {},            # {"pp,tp": int | None}
    "entries_without_capacity": [],
    "entries_unrecognized": 0,
    "chosen_cap": None,
    "cap_source": None,        # "min-across-ranks" | "fail-safe-zero" | None
    "zero_mode": False,
    "zero_reason": None,
    "clamp_applied": False,
    "clamp_failed_open": False,
    "clamped_manager_class": None,
    "adopted_by_snapshot": False,
    "original_call_error": None,
    # clamp bookkeeping
    "pool_total_before": None,
    "pool_total_after": None,
    "cpu_blocks_withheld": None,
    "blocks_still_referenced_above_cap": None,
    "pool_free_after": None,
    "manager_num_cpu_blocks_before": None,
    "manager_num_cpu_blocks_after": None,
    "cpu_kv_config_num_blocks_after": None,
    "cap_limited_by_pool": None,
    "nothing_to_withdraw": None,
    # the refusal counter (the clamp itself is surgery: this only counts)
    "allocations_refused_by_cap": 0,
    "pool_value_errors": 0,
    "fallbacks": 0,
}


def reset_state_for_tests() -> None:
    """Counters are process-wide, like `snapshot._state` and `store_hook._state`."""
    _state.update({
        "installs": 0, "reinstalls": 0, "skips": 0, "skip_reason": None,
        "class": None, "worker_installed": False, "scheduler_installed": False,
        "worker_reports": 0, "worker_foreign_payload": 0, "worker_fallbacks": 0,
        "handshake_calls": 0, "handshake_seen": False,
        "handshake_wrapped_payload": False, "reported": {},
        "entries_without_capacity": [], "entries_unrecognized": 0,
        "chosen_cap": None, "cap_source": None, "zero_mode": False,
        "zero_reason": None, "clamp_applied": False,
        "clamp_failed_open": False,
        "clamped_manager_class": None, "adopted_by_snapshot": False,
        "original_call_error": None, "pool_total_before": None,
        "pool_total_after": None, "cpu_blocks_withheld": None,
        "blocks_still_referenced_above_cap": None, "pool_free_after": None,
        "manager_num_cpu_blocks_before": None,
        "manager_num_cpu_blocks_after": None,
        "cpu_kv_config_num_blocks_after": None,
        "cap_limited_by_pool": None, "nothing_to_withdraw": None,
        "allocations_refused_by_cap": 0, "pool_value_errors": 0,
        "fallbacks": 0,
    })


# --- resolving the fork's names without importing it ------------------------

def connector_base_class():
    """`KVConnectorBase_V1`, if this process already imported it. Never imports.

    Both methods we wrap are defined there and inherited by our connector, and
    that module is imported before `load_general_plugins()` in the engine
    process and in a worker alike (measured at HEAD 10b8e07) - which is the only
    reason this plugin can wrap anything before the handshake runs.
    """
    module = sys.modules.get(BASE_MODULE)
    if module is None:
        return None
    return getattr(module, BASE_CLASS, None)


def manager_of(engine):
    """The scheduler-side offload manager off a live engine, or None. Reads only."""
    scheduler = getattr(engine, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    return getattr(connector, "scheduler_manager", None)


def _looks_like_offload_manager(manager) -> bool:
    """The duck-type test for `SimpleCPUOffloadScheduler` (no import, no isinstance).

    `cpu_block_pool` is what we clamp and `num_cpu_blocks` is the mirror
    (`manager.py:167`, `manager.py:207`); a connector that carries a
    `scheduler_manager` of some other shape gets no clamp from us.
    """
    if manager is None:
        return False
    pool = getattr(manager, "cpu_block_pool", None)
    if pool is None or not isinstance(getattr(pool, "blocks", None), list):
        return False
    return _non_negative_int(getattr(manager, "num_cpu_blocks", None)) is not None


def _non_negative_int(value) -> int | None:
    """A real, usable block count, or None. `bool` is NOT an int here."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


# --- worker side ------------------------------------------------------------

def _rank_capacity(connector) -> int | None:
    """This rank's arena size, from the worker handler the connector holds.

    `simple_cpu_offload_connector.py:167` builds `worker_handler` for the
    WORKER role only, and `worker.py:162` fills `num_cpu_blocks` in
    `register_kv_caches()` - which `core.py` runs before it collects the
    handshake, so a value here is the size of the arena about to be copied into.
    0 is the constructor's initial value (`worker.py:52`) and means "registered
    never ran", which the scheduler side must read as nothing reported.
    """
    handler = getattr(connector, "worker_handler", None)
    if handler is None:
        return None
    return _non_negative_int(getattr(handler, "num_cpu_blocks", None))


# An optional zero/one-argument hook, called once per rank right after that rank
# reported its capacity. WHY IT IS HERE: it is the first moment in a WORKER
# process at which the offload connector - and therefore its worker handler and
# the module that class lives in - is guaranteed to exist, which is exactly what
# `worker_diag.install()` needs and what `engine_side.register()` (which runs
# earlier) cannot know. It stays a generic callback so this module keeps knowing
# nothing about who listens; `engine_side` wires `worker_diag` to it.
_worker_report_hook = None


def set_worker_report_hook(hook) -> None:
    """Install (or clear with None) the after-a-rank-reports hook. Contained."""
    global _worker_report_hook
    _worker_report_hook = hook


def _call_worker_report_hook(connector) -> None:
    hook = _worker_report_hook
    if hook is None:
        return
    try:
        hook(connector)
    except Exception as exc:  # noqa: BLE001 - a listener never breaks a handshake
        _state["worker_fallbacks"] += 1
        snapshot._error(exc, "worker report hook")


def _carrier(payload, capacity: int):
    """Attach the capacity to whatever the connector returned. Never mutates it.

    None payload    -> a bare carrier: this connector has no handshake payload
                       of its own today (`base.py:447-456` returns None and
                       `SimpleCPUOffloadConnector` does not override it), and
                       `gpu_worker.py:882` turns a non-None return into
                       `{(pp_rank, tp_rank): metadata}`, which is exactly the
                       channel we need.
    dict payload    -> a shallow copy plus our key, so the connector's own
                       object is never touched.
    anything else   -> a wrapper dict that also carries the original object under
                       PAYLOAD_KEY; the scheduler side unwraps it and hands the
                       ORIGINAL OBJECT to the fork's method, so a connector with
                       a typed payload sees precisely what it would have seen.
    """
    if payload is None:
        return {CAPACITY_KEY: capacity}
    if isinstance(payload, dict):
        merged = dict(payload)
        merged[CAPACITY_KEY] = capacity
        return merged
    _state["worker_foreign_payload"] += 1
    logger.warning("kvtier-dev-plugin: handshake payload is %s, not a dict; "
                   "carrying it under %r so the fork still receives it intact",
                   type(payload).__name__, PAYLOAD_KEY)
    return {CAPACITY_KEY: capacity, PAYLOAD_KEY: payload}


def _make_worker_wrapper(original):
    def get_handshake_metadata(self):
        try:
            payload = original(self)
            capacity = _rank_capacity(self)
            if capacity is None:
                # Not the offload connector (or no worker half in this role):
                # say nothing at all and let the fork's payload through untouched.
                return payload
            _state["worker_reports"] += 1
            _call_worker_report_hook(self)
            return _carrier(payload, capacity)
        except Exception as exc:  # noqa: BLE001 - but see the re-raise below
            # Count it, then re-raise: without this plugin the original's own
            # exception is what the engine would have seen, and swallowing a
            # connector error here would hide a real fault behind a handshake
            # that merely looks empty.
            _state["worker_fallbacks"] += 1
            snapshot._error(exc, "handshake metadata producer")
            raise

    setattr(get_handshake_metadata, OWNER_ATTR, OWNER_TAG)
    setattr(get_handshake_metadata, ORIGINAL_ATTR, original)
    get_handshake_metadata.__doc__ = (
        "kvtier-dev-plugin: the connector's own handshake metadata plus this "
        "rank's num_cpu_blocks, so the scheduler can clamp to what every rank "
        "can actually address.")
    return get_handshake_metadata


# --- the payload split ------------------------------------------------------

def _split_key(key):
    """(pp_rank, tp_rank) out of a content key, or None if it is not one."""
    if isinstance(key, tuple) and len(key) == 2:
        pp, tp = key
        if (isinstance(pp, int) and not isinstance(pp, bool)
                and isinstance(tp, int) and not isinstance(tp, bool)):
            return (pp, tp)
    return None


def _unwrap_entry(entry):
    """(capacity, payload, has_payload) for one handshake entry.

    `has_payload` decides whether the entry exists for the fork at all. That is
    the difference between "pass the connector's payload through" and "invent a
    rank the connector never reported": `gpu_worker.py:882` returns None for a
    connector whose `get_handshake_metadata()` is None, `core.py:299` skips
    those workers, and the base implementation at base.py:725 RAISES on an entry
    whose pp_rank is not 0. So a bare carrier (this connector has no payload of
    its own today) must contribute a capacity and NO entry - otherwise the
    plugin would be the thing that crashes the boot.
    """
    if entry is None:
        return None, None, False
    if not isinstance(entry, dict):
        # Not something we produced (a typed payload we were never able to
        # extend): it passes through untouched and reports no capacity.
        return None, entry, True
    capacity = _non_negative_int(entry.get(CAPACITY_KEY))
    if PAYLOAD_KEY in entry:
        # Our wrapper of a non-dict payload: hand back the ORIGINAL object.
        return capacity, entry[PAYLOAD_KEY], entry[PAYLOAD_KEY] is not None
    ours = {CAPACITY_KEY, PAYLOAD_KEY}
    rest = {k: v for k, v in entry.items() if k not in ours}
    if len(rest) == len(entry):
        # A dict the connector itself produced: the identical object the fork
        # would have passed, with nothing of ours inside it.
        return capacity, entry, True
    if not rest:
        # A bare carrier: capacity only, no entry.
        return capacity, None, False
    return capacity, rest, True


def _split_payload(metadata):
    """Separate "what the fork must see" from "what each rank can hold".

    Returns (payload, reported, without_capacity, unrecognized, wrapped):
      payload         the dict the fork would have received WITHOUT this plugin:
                      only the entries that carry a real connector payload, with
                      our in-band keys stripped and our wrapper unwrapped. Built
                      before the original is called, so the original sees exactly
                      its own pre-plugin argument - and nothing that only we
                      invented.
      reported        {"pp,tp": int} for every rank-keyed entry that reported.
      without_capacity  rank-keyed entries that reported nothing.
      unrecognized    entries whose key is not a (pp_rank, tp_rank) pair.
      wrapped         True if any entry was one of our wrapper dicts.
    """
    payload: dict = {}
    reported: dict = {}
    without: list = []
    unrecognized = 0
    wrapped = False
    items = metadata.items() if isinstance(metadata, dict) else []
    for key, entry in items:
        rank = _split_key(key)
        capacity, content, has_content = _unwrap_entry(entry)
        if has_content:
            payload[key] = content
        wrapped = wrapped or (isinstance(entry, dict) and PAYLOAD_KEY in entry)
        if rank is None:
            # A key that is not a (pp_rank, tp_rank) pair is not ours to read.
            unrecognized += 1
            continue
        if capacity is None:
            # Every rank-keyed entry that did not carry a capacity counts, down to
            # an explicit None: "one rank did not tell us" is the fail-safe
            # condition, and reading it as "that rank does not exist" would be
            # the one mistake this function must never make.
            without.append(_key_text(key))
        else:
            reported[f"{rank[0]},{rank[1]}"] = capacity
    return payload, reported, without, unrecognized, wrapped


def _key_text(key) -> str:
    try:
        return str(key)[:64]
    except Exception:  # noqa: BLE001 - a key that cannot be named is still data
        return "<unprintable key>"


def _interpret_cap(reported, without, unrecognized, manager):
    """The ceiling, or None for fail-safe zero. Never guesses, never takes max.

    min across the ranks is the only defensible reading: the ids the scheduler
    emits are shared by every rank, so the smallest arena is the binding one.
    The scheduler's own `num_cpu_blocks` is then the second term of the min, so
    a rank that reported an absurd number can never RAISE the ceiling above what
    the aggregate view already assumed.
    """
    if not reported:
        return None, "no rank reported a CPU arena capacity"
    if without:
        return None, (f"{len(without)} rank(s) reported no capacity "
                      f"({', '.join(without[:4])})")
    if unrecognized:
        return None, (f"{unrecognized} handshake entries are not keyed by "
                      f"(pp_rank, tp_rank) - the payload shape is not ours to "
                      f"interpret")
    values = sorted(reported.values())
    if values[0] <= 0:
        return None, f"a rank reported {values[0]} blocks"
    # The scheduler's own aggregate view is the second ceiling: the clamp may
    # only ever LOWER it, never raise it. (The pool's own block count is the
    # third, applied inside `_apply_cap` and reported as `cap_limited_by_pool`.)
    own = _non_negative_int(getattr(manager, "num_cpu_blocks", None))
    cap = values[0]
    source = "min-across-ranks"
    if own is not None and own < cap:
        cap = own
        source = "min-across-ranks-capped-by-scheduler-view"
    return cap, source


# --- the clamp itself -------------------------------------------------------

def _in_free_queue(block) -> bool:
    """Is this block a node of the free list? That is the queue's own test:
    `FreeKVCacheBlockQueue.remove()` (kv_cache_utils.py:~340) raises when either
    link is None, which is exactly the "not in the queue" state."""
    return (getattr(block, "prev_free_block", None) is not None
            and getattr(block, "next_free_block", None) is not None)


def _arm_refusal_counter(pool) -> bool:
    """Count the allocations the clamp causes the pool to refuse. Observability
    only - the CLAMP is the surgery below, this never decides anything.

    Instance-level, never class-level: the same BlockPool class backs the GPU
    pool, and the GPU pool must not acquire a wrapper from a KV-tier plugin.
    Cost: one extra Python frame per `get_new_blocks` call, which the tier
    makes a few times per scheduler step (manager.py:613, :705, :751, :1195),
    never per token. The fork's own guard (block_pool.py:676) still raises; we
    count and re-raise, so no caller's behaviour changes.
    """
    if getattr(pool, POOL_MARKER, None) == OWNER_TAG:
        return False
    original = getattr(pool, "get_new_blocks", None)
    if not callable(original):
        return False

    def get_new_blocks(num_blocks):
        try:
            return original(num_blocks)
        except ValueError as exc:
            if _SHORTFALL_TEXT in str(exc):
                _state["allocations_refused_by_cap"] += 1
            else:
                _state["pool_value_errors"] += 1
            raise

    setattr(get_new_blocks, OWNER_ATTR, OWNER_TAG)
    setattr(get_new_blocks, ORIGINAL_ATTR, original)
    try:
        pool.get_new_blocks = get_new_blocks
        setattr(pool, POOL_MARKER, OWNER_TAG)
    except Exception as exc:  # noqa: BLE001 - a pool that will not take a
        snapshot._error(exc, "refusal counter")  # counter is still clamped
        return False
    return True


def _apply_cap(manager, cap: int) -> dict:
    """One-time surgery, then zero per-call cost.

    Removing ids >= cap from the free structure at init makes every existing
    counter and code path correct BY CONSTRUCTION: `get_num_free_blocks()`
    (block_pool.py:861, the maintained `num_free_blocks` of the free queue) and
    `get_new_blocks()` (block_pool.py:664, which pops from that same queue)
    become right with nothing wrapped on the hot path. The queue supports O(1)
    removal of any node (`FreeKVCacheBlockQueue.remove`, kv_cache_utils.py), so
    this is a bounded loop over exactly the ids being withdrawn - no rebuild of
    the pool, no reallocation, and no ordering assumption: the queue's LRU order
    is left untouched for the ids that stay.

    Then the mirrors: `manager.num_cpu_blocks` (`manager.py:167`), the pool's own
    `num_gpu_blocks` (block_pool.py:173, which `get_usage()` and
    `reset_prefix_cache()` do arithmetic on) and the manager's CPU
    `kv_cache_config.num_blocks` (`manager.py:164`, a `replace()`-derived COPY of
    the GPU config, never the engine's own). The pool's `blocks` list keeps its
    full length on purpose: ids >= cap must stay *nameable* so that
    `pool.blocks[bid]` lookups of an id the fork already emitted cannot become
    an IndexError; they simply can never be allocated again.
    """
    pool = manager.cpu_block_pool
    blocks = pool.blocks
    record = {"applied": False, "reason": None}

    total = len(blocks)
    if total == 0:
        record["reason"] = "the CPU pool holds no blocks at all"
        return record
    # The pool's own length is a hard ceiling: an id the pool cannot name is not
    # a block, and a `num_cpu_blocks` larger than the pool would be a lie.
    effective = min(cap, total)
    record["requested_cap"] = cap
    record["effective_cap"] = effective
    record["cap_limited_by_pool"] = effective < cap

    # 1. Withdraw the ids from the prefix cache first, so a hash entry can never
    #    name a block that can no longer be allocated (a stale hit on an id we
    #    withdrew would become a LOAD from an arena slot we refuse to own).
    withdraw = set(range(effective, total))
    evict_blocks = getattr(pool, "evict_blocks", None)
    if withdraw and callable(evict_blocks):
        evict_blocks(withdraw)

    # 2. Withdraw them from the free structure. Ids that are neither free nor
    #    in the queue at this instant are counted as `still_referenced`: at the
    #    handshake nothing on the CPU pool can be allocated yet, so a non-zero
    #    number here means something reached the tier before the clamp and it
    #    belongs in the report, not in a guess.
    removed = 0
    still_referenced = 0
    queue = getattr(pool, "free_block_queue", None)
    for block in blocks[effective:]:
        if getattr(block, "is_null", False):
            continue
        if queue is not None and _in_free_queue(block):
            queue.remove(block)
            removed += 1
        if getattr(block, "ref_cnt", 0) > 0:
            still_referenced += 1

    # 3. The mirrors. The pool keeps at least the null block (block_pool.py:193
    #    pops id 0 as `null_block` and `get_usage()` divides by num_gpu_blocks-1),
    #    so a zero cap leaves it with exactly that one, non-allocatable block.
    record["pool_total_before"] = _non_negative_int(
        getattr(pool, "num_gpu_blocks", None))
    record["manager_num_cpu_blocks_before"] = _non_negative_int(
        getattr(manager, "num_cpu_blocks", None))
    pool.num_gpu_blocks = max(1, effective)
    manager.num_cpu_blocks = effective
    config = getattr(manager, "cpu_kv_cache_config", None)
    if config is not None:
        config.num_blocks = max(1, effective)
    record["cpu_kv_config_num_blocks_after"] = _non_negative_int(
        getattr(getattr(manager, "cpu_kv_cache_config", None), "num_blocks", None))
    record["manager_num_cpu_blocks_after"] = effective
    record["pool_total_after"] = max(1, effective)
    record["cpu_blocks_withheld"] = removed
    record["blocks_still_referenced_above_cap"] = still_referenced
    record["nothing_to_withdraw"] = effective >= total

    _arm_refusal_counter(pool)

    # 4. A claim we can prove, not a log line we hope somebody reads.
    free = None
    getter = getattr(pool, "get_num_free_blocks", None)
    if callable(getter):
        free = getter()
    record["pool_free_after"] = _non_negative_int(free)
    if still_referenced or (free is not None and _non_negative_int(free) is None):
        logger.warning("kvtier-dev-plugin: CPU block-id clamp to %d finished with "
                       "%d block(s) above the cap still referenced", effective,
                       still_referenced)
    record["applied"] = True
    return record


def _clamp_connector(connector, reported, without, unrecognized) -> dict:
    """Decide the cap for one connector and apply it. Never raises (caller guards).

    Returns a record with `applied` and, when nothing was clamped, `benign`:
    a connector that is not ours, or a manager already clamped by us, is a
    settled state - a foreign clamp marker and a failed surgery are not, and
    both end up reported as `clamp_failed_open`, because "we could not prove a
    ceiling" is exactly the state an operator must not be able to misread.
    """
    manager = getattr(connector, "scheduler_manager", None)
    marker = getattr(manager, MANAGER_MARKER, None) if manager is not None else None
    if not _looks_like_offload_manager(manager):
        return _decline("this connector carries no SimpleCPUOffloadScheduler to "
                        "clamp", benign=True)
    if marker == OWNER_TAG:
        return _decline("this manager is clamped already", benign=True)
    if marker is not None:
        _state["clamp_failed_open"] = True
        return _decline(f"{type(manager).__name__} carries another plugin's clamp "
                        f"marker: we refuse to double-clamp it, so no ceiling is "
                        f"proved", benign=False)

    cap, source = _interpret_cap(reported, without, unrecognized, manager)
    if cap is None:
        cap = 0
        _state["zero_mode"] = True
        _state["zero_reason"] = source
        _state["cap_source"] = "fail-safe-zero"
        logger.warning("kvtier-dev-plugin: CPU offload capacity UNKNOWN (%s) - "
                       "parking NOTHING: the CPU block pool is clamped to 0 "
                       "usable blocks. Uncapped ids would write past the end of "
                       "a rank's pinned arena.", source)
    else:
        _state["zero_mode"] = False
        _state["zero_reason"] = None
        _state["cap_source"] = source
        logger.warning(
            "kvtier-dev-plugin: CPU offload block ids clamped to %d "
            "(reported per rank: %s; scheduler's own view: %s)", cap,
            reported, getattr(manager, "num_cpu_blocks", None))

    record = _apply_cap(manager, cap)
    if not record.get("applied"):
        # The surgery declined. Retry the only ceiling that is always safe, and
        # say so plainly if even that failed: `zero_mode` must mean the pool IS
        # clamped to zero, never "we intended to".
        record = _apply_cap(manager, 0)
        if not record.get("applied"):
            _state["clamp_failed_open"] = True
            _state["zero_mode"] = False
            _state["zero_reason"] = None
            _state["cap_source"] = None
            logger.warning("kvtier-dev-plugin: the CPU block-id clamp FAILED "
                           "(%s) - no ceiling is proved and the tier keeps the "
                           "aggregate view's ids. Disable the tier.",
                           record.get("reason"))
            return record
        _state["zero_mode"] = True
        _state["zero_reason"] = (f"the reported clamp was refused "
                                 f"({source}), so the pool is parked at zero")
        _state["cap_source"] = "fail-safe-zero"

    effective = record["effective_cap"]
    setattr(manager, MANAGER_MARKER, OWNER_TAG)
    _state["clamp_applied"] = True
    _state["clamp_failed_open"] = False
    _state["clamped_manager_class"] = type(manager).__name__
    _state["chosen_cap"] = effective
    for key in ("pool_total_before", "pool_total_after", "cpu_blocks_withheld",
                "blocks_still_referenced_above_cap", "pool_free_after",
                "manager_num_cpu_blocks_before", "manager_num_cpu_blocks_after",
                "cpu_kv_config_num_blocks_after", "cap_limited_by_pool",
                "nothing_to_withdraw"):
        if key in record:
            _state[key] = record[key]
    logger.warning(
        "kvtier-dev-plugin: CPU block-id ceiling is %d; %d id(s) withdrawn from "
        "the CPU free pool (%d above the cap still referenced); pool reports "
        "%s free", effective, record.get("cpu_blocks_withheld"),
        record.get("blocks_still_referenced_above_cap"),
        record.get("pool_free_after"))
    return record


def _decline(reason: str, benign: bool) -> dict:
    """A clamp that did not happen, with the honesty flag the report needs."""
    if not benign:
        logger.warning("kvtier-dev-plugin: CPU capacity clamp NOT applied: %s", reason)
    return {"applied": False, "benign": benign, "reason": reason}


# --- scheduler side ---------------------------------------------------------

def _make_scheduler_wrapper(original):
    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        _state["handshake_calls"] += 1
        _state["handshake_seen"] = True
        payload = metadata
        reported: dict = {}
        without: list = []
        unrecognized = 0
        wrapped = False
        try:
            payload, reported, without, unrecognized, wrapped = (
                _split_payload(metadata))
        except Exception as exc:  # noqa: BLE001
            # A payload shape we cannot read is the unknown-capacity case: hand
            # the ORIGINAL object to the fork (we cannot improve on it) and clamp
            # to zero, because "no rank reported" is the only honest reading.
            _state["fallbacks"] += 1
            snapshot._error(exc, "handshake payload split")
            payload, reported, without = metadata, {}, ["<unreadable payload>"]

        _state["reported"] = dict(reported)
        _state["entries_without_capacity"] = list(without)
        _state["entries_unrecognized"] = unrecognized
        _state["handshake_wrapped_payload"] = bool(wrapped)

        # The fork's own handshake, on exactly the payload the fork would have
        # received without this plugin. A raise here is the fork's own behaviour
        # for that payload, so it propagates - except that the clamp is applied
        # first, because an engine that survives this call must not survive it
        # uncapped.
        try:
            original(self, payload)
        except Exception as exc:  # noqa: BLE001
            _state["fallbacks"] += 1
            _state["original_call_error"] = (
                f"{type(exc).__name__}: {str(exc)[:180]}")
            snapshot._error(exc, "handshake consumer")
            _contained_clamp(self, reported, without, unrecognized)
            raise

        _contained_clamp(self, reported, without, unrecognized)

    setattr(set_xfer_handshake_metadata_pp_aware, OWNER_ATTR, OWNER_TAG)
    setattr(set_xfer_handshake_metadata_pp_aware, ORIGINAL_ATTR, original)
    set_xfer_handshake_metadata_pp_aware.__doc__ = (
        "kvtier-dev-plugin: the connector's own PP-aware handshake, then the "
        "min-across-ranks clamp on the CPU block-id space. Contained: the clamp "
        "can degrade the tier to parking nothing, never corrupt a host arena.")
    return set_xfer_handshake_metadata_pp_aware


def _contained_clamp(connector, reported, without, unrecognized) -> None:
    """Apply the clamp, and never let its failure reach the engine.

    A clamp that raises is not a pass: `_clamp_connector` has already tried the
    zero ceiling on its own account, and a failure that escapes it sets
    `clamp_failed_open` here with the same meaning - no ceiling is proved. The
    handshake itself is never affected: the boot must survive this.
    """
    try:
        _clamp_connector(connector, reported, without, unrecognized)
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        _state["fallbacks"] += 1
        snapshot._error(exc, "CPU capacity clamp")
        _state["clamp_failed_open"] = True
        _state["zero_mode"] = False
        _state["zero_reason"] = None
        logger.warning("kvtier-dev-plugin: the CPU capacity clamp raised %s: %s - "
                       "no ceiling is proved. Disable the tier.",
                       type(exc).__name__, str(exc)[:160])


# --- last-resort adoption ---------------------------------------------------

def adopt_if_missed(engine) -> dict:
    """Fail-safe backstop: the tier exists, the handshake window is shut, so an
    unclamped pool is an unverified pool and must park nothing.

    WHY THIS IS SOUND: the handshake is collected inside `EngineCore.__init__`
    (`core.py:288-302`), while a utility handler - the only caller of this -
    runs from the busy loop (`core.py:1669-1671 -> :1712-1728`), which cannot
    start before `__init__` returns. So by the time this can run, a handshake
    that was going to arrive has arrived. If our scheduler wrapper never ran
    (the fork renamed the method, a connector that overrides it, an engine that
    skipped the call), the tier would otherwise keep the aggregate view's
    over-sized ids: the exact corruption this module exists to prevent.

    Same idiom as `store_hook.ensure_installed`: called from the snapshot and
    the reload verbs, marker-checked, and a no-op once the claim is settled.
    """
    report = {"adopted": False, "reason": None}
    try:
        if _state["clamp_applied"]:
            report["reason"] = "the handshake clamp already ran"
            return report
        if _state["handshake_seen"]:
            # The wrapper ran and declined to clamp: not our connector.
            report["reason"] = "the handshake ran and this is not our connector"
            return report
        manager = manager_of(engine)
        if not _looks_like_offload_manager(manager):
            report["reason"] = "no SimpleCPUOffloadScheduler on this engine"
            return report
        if getattr(manager, MANAGER_MARKER, None) is not None:
            report["reason"] = "this manager already carries a clamp marker"
            return report
        record = _clamp_connector(getattr(getattr(engine, "scheduler", None),
                                          "connector", None), {}, [], 0)
        report["adopted"] = bool(record.get("applied"))
        report["reason"] = (
            "the handshake never delivered a capacity, so the tier parks nothing"
            if report["adopted"] else str(record.get("reason")))
        if report["adopted"]:
            # Say WHERE the zero came from: a cap the handshake refused and a cap
            # this backstop had to invent are different incidents, and the
            # operator reading the snapshot must be able to tell them apart.
            _state["zero_reason"] = (
                f"adopted from the read path: {_state['zero_reason']}")
        if report["adopted"]:
            _state["adopted_by_snapshot"] = True
            logger.warning("kvtier-dev-plugin: adopted the fail-safe clamp from "
                           "a snapshot: no handshake capacity was ever reported")
        return report
    except Exception as exc:  # noqa: BLE001 - a diagnostic path, never fatal
        _state["fallbacks"] += 1
        snapshot._error(exc, "capacity adoption")
        report["reason"] = f"adoption raised: {type(exc).__name__}"[:160]
        return report


# --- installation -----------------------------------------------------------

def _wrap_one(cls, name, factory, report) -> str:
    """Install one wrapper on `cls`. Returns 'attached' | 'replaced' | 'skipped'.

    The ownership discipline of `store_hook.install`, with one difference forced
    by the target: the name we wrap is INHERITED, so `getattr(cls, name)` is the
    BASE class's function while `cls` is the base class - the defining module
    must therefore be allowed to be either module of the fork that may own the
    method today (FORK_OWNED_MODULES). A function defined anywhere else is a
    third party and is refused, never overwritten.
    """
    owner_name = getattr(cls, "__name__", type(cls).__name__)

    def refuse(reason):
        report["skipped"].append(name)
        # The FIRST reason is the report's: a half-installed pair must still say
        # why the other half did not land, so this cannot be read back out of
        # `_state` after a later success cleared it.
        report["skip_reasons"].append((name, reason))
        _refuse(reason)
        return "skipped"

    existing = getattr(cls, name, None)
    if existing is None:
        return refuse(f"{owner_name}.{name} does not exist: the fork renamed "
                      f"it, refusing to invent one")
    if not callable(existing):
        return refuse(f"{owner_name}.{name} is not callable "
                      f"({type(existing).__name__})")
    owner = getattr(existing, OWNER_ATTR, None)
    if owner is not None and owner != OWNER_TAG:
        held = getattr(existing, "__qualname__", type(existing).__name__)
        return refuse(f"{owner_name}.{name} carries another plugin's owner marker "
                      f"({held!r}), refusing to overwrite")
    replaced = owner == OWNER_TAG
    if not replaced:
        home = getattr(existing, "__module__", None)
        if home not in FORK_OWNED_MODULES:
            return refuse(f"{owner_name}.{name} is neither ours nor the fork's "
                          f"own: defined in {home!r}, not one of "
                          f"{FORK_OWNED_MODULES}, refusing to overwrite")
    if replaced:
        # Bind the FORK's function we stored, so a reload loop cannot nest.
        existing = getattr(existing, ORIGINAL_ATTR, existing)
    setattr(cls, name, factory(existing))
    report["replaced" if replaced else "attached"].append(name)
    return "replaced" if replaced else "attached"


def _refuse(reason: str) -> None:
    """One line per distinct reason, like store_hook._refuse: `register()` runs
    in every process and the absence of vllm is a fact, not a fault."""
    if _state["skip_reason"] == reason:
        logger.debug("kvtier-dev-plugin: CPU capacity clamp still not installed: "
                     "%s", reason)
        return
    _state["skips"] += 1
    _state["skip_reason"] = reason
    logger.info("kvtier-dev-plugin: CPU capacity clamp NOT installed: %s", reason)


def install(cls=None) -> dict:
    """Wrap the two handshake methods. Idempotent, never raises.

    Called from `engine_side.register()` in EVERY process: in the EngineCore
    process the scheduler-side half is what matters, in a worker the worker-side
    half is. Both halves are installed everywhere because `general_plugins`
    cannot tell the two apart and a wrapper in the wrong process is inert.
    """
    report = {"installed": False, "attached": [], "replaced": [], "skipped": [],
              "skip_reasons": [], "class": None, "reason": None}
    try:
        if cls is None:
            cls = connector_base_class()
        if cls is None:
            _refuse(f"{BASE_MODULE} is not imported in this process "
                    f"({BASE_CLASS} absent), nothing to wrap")
            report["reason"] = _state["skip_reason"]
            return report
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        _state["class"] = report["class"]

        worker = _wrap_one(cls, WORKER_METHOD, _make_worker_wrapper, report)
        scheduler = _wrap_one(cls, SCHEDULER_METHOD, _make_scheduler_wrapper,
                              report)
        _state["worker_installed"] = worker != "skipped"
        _state["scheduler_installed"] = scheduler != "skipped"
        if report["replaced"]:
            _state["reinstalls"] += 1
            logger.debug("kvtier-dev-plugin: capacity wrappers already on %s",
                         report["class"])
        if report["attached"] and not report["skipped"]:
            # A clean install clears the standing refusal; a HALF install keeps it,
            # because half a pair is still a refusal the operator must be able to
            # read (worker without scheduler = no clamp, scheduler without worker
            # = fail-safe zero).
            _state["skip_reason"] = None
        if report["attached"]:
            _state["installs"] += 1
            logger.info("kvtier-dev-plugin: CPU capacity clamp installed on "
                        "%s.%s / %s (%d attached, %d replaced, %d skipped)",
                        report["class"], WORKER_METHOD, SCHEDULER_METHOD,
                        len(report["attached"]), len(report["replaced"]),
                        len(report["skipped"]))
        report["installed"] = bool(report["attached"] or report["replaced"])
        if report["skip_reasons"]:
            report["reason"] = report["skip_reasons"][0][1]
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        _state["fallbacks"] += 1
        snapshot._error(exc, "capacity wrapper install")
        report["installed"] = False
        report["reason"] = f"install raised: {type(exc).__name__}: {exc}"[:200]
        return report


def ensure_installed() -> dict:
    """Retry the install if this process only imported vllm after we were loaded.

    Two cheap no-ops and one retry, in that order: once both halves are in place
    this is two `getattr` reads per snapshot, and once the class HAS been found
    and was still refused (a renamed method, a foreign marker, a half pair) a
    retry cannot change the answer - the fork does not edit itself while it runs
    - so asking again would only churn `reinstalls` on every dashboard poll.
    The case worth retrying is the one where `connector_base_class()` was None
    because `register()` ran before anything imported the connector tree.
    """
    if _state["worker_installed"] and _state["scheduler_installed"]:
        return {"installed": True, "reason": "already installed"}
    if _state["class"] is not None:
        return {"installed": False,
                "reason": _state["skip_reason"]
                          or "already attempted on this class"}
    return install()


# --- observability ----------------------------------------------------------


def _fields(**overrides) -> dict:
    """THE payload builder. One dict literal, so no path can be missing a field.

    `section()` adds the live-manager reads on top of it; `empty_section()` is the
    degraded view a caller falls back to, built through the SAME code, so the two
    key sets cannot drift (the `test_the_degraded_section_has_every_key` guard).
    """
    fields = {
        # installed, and why not
        "installed": bool(_state["worker_installed"]
                          and _state["scheduler_installed"]),
        "worker_installed": _state["worker_installed"],
        "scheduler_installed": _state["scheduler_installed"],
        "class": _state["class"],
        "module": BASE_MODULE,
        "worker_method": WORKER_METHOD,
        "scheduler_method": SCHEDULER_METHOD,
        "installs": _state["installs"],
        "reinstalls": _state["reinstalls"],
        "skips": _state["skips"],
        "reason": _state["skip_reason"],
        # what the workers said. Keyed "pp_rank,tp_rank": a tuple is not a
        # JSON/msgpack key, and a section that cannot be encoded on the wire is
        # worth nothing to the operator reading it from the live engine.
        "handshake_seen": _state["handshake_seen"],
        "handshake_calls": _state["handshake_calls"],
        "handshake_wrapped_payload": _state["handshake_wrapped_payload"],
        "per_rank": dict(_state["reported"]),
        "per_rank_list": _per_rank_list(),
        "entries_without_capacity": list(_state["entries_without_capacity"]),
        "entries_unrecognized": _state["entries_unrecognized"],
        "worker_reports": _state["worker_reports"],
        "worker_foreign_payload": _state["worker_foreign_payload"],
        "worker_fallbacks": _state["worker_fallbacks"],
        # the decision
        "chosen_cap": _state["chosen_cap"],
        "cap_source": _state["cap_source"],
        "clamp_applied": _state["clamp_applied"],
        "clamp_failed_open": _state["clamp_failed_open"],
        "zero_mode": _state["zero_mode"],
        "zero_reason": _state["zero_reason"],
        "adopted_by_snapshot": _state["adopted_by_snapshot"],
        "clamped_manager_class": _state["clamped_manager_class"],
        "original_call_error": _state["original_call_error"],
        # what the clamp physically did
        "pool_total_before": _state["pool_total_before"],
        "pool_total_after": _state["pool_total_after"],
        "cpu_blocks_withheld": _state["cpu_blocks_withheld"],
        "blocks_still_referenced_above_cap":
            _state["blocks_still_referenced_above_cap"],
        "pool_free_after": _state["pool_free_after"],
        "manager_num_cpu_blocks_before": _state["manager_num_cpu_blocks_before"],
        "manager_num_cpu_blocks_after": _state["manager_num_cpu_blocks_after"],
        "cpu_kv_config_num_blocks_after":
            _state["cpu_kv_config_num_blocks_after"],
        "cap_limited_by_pool": _state["cap_limited_by_pool"],
        "nothing_to_withdraw": _state["nothing_to_withdraw"],
        # and the hot-path evidence
        "allocations_refused_by_cap": _state["allocations_refused_by_cap"],
        "pool_value_errors": _state["pool_value_errors"],
        "fallbacks": _state["fallbacks"],
        "live_manager_clamped": None,
        "live_cpu_pool": None,
        # Set only by the degraded path; a healthy section says None.
        "error": None,
    }
    fields.update(overrides)
    return fields


def _per_rank_list() -> list:
    """The same reports as objects, so a reader never has to parse a key."""
    out = []
    for key, value in sorted(_state["reported"].items()):
        pp, _, tp = key.partition(",")
        out.append({"key": key,
                    "pp_rank": int(pp) if pp.isdigit() else None,
                    "tp_rank": int(tp) if tp.isdigit() else None,
                    "num_cpu_blocks": value})
    return out


def _live_view(manager) -> dict:
    """The two reads only the live manager can answer. Never raises."""
    try:
        clamped = getattr(manager, MANAGER_MARKER, None) == OWNER_TAG
        pool = getattr(manager, "cpu_block_pool", None)
        if pool is None:
            return {"live_manager_clamped": clamped, "live_cpu_pool": None}
        blocks = getattr(pool, "blocks", None)
        free = None
        getter = getattr(pool, "get_num_free_blocks", None)
        if callable(getter):
            free = _non_negative_int(getter())
        return {"live_manager_clamped": clamped,
                "live_cpu_pool": {
                    "num_gpu_blocks": _non_negative_int(
                        getattr(pool, "num_gpu_blocks", None)),
                    "free_blocks": free,
                    "block_objects": (_non_negative_int(len(blocks))
                                       if isinstance(blocks, list) else None),
                    "refusal_counter_armed": (
                        getattr(pool, POOL_MARKER, None) == OWNER_TAG)}}
    except Exception as exc:  # noqa: BLE001 - a pool that lies is data, not a 500
        snapshot._error(exc, "live CPU pool view")
        return {"live_manager_clamped": None, "live_cpu_pool": None}


def section(manager=None) -> dict:
    """The snapshot section: installed, what each rank said, the cap, the clamp.

    Every field is present on every path, including the ones that never got as
    far as reading a payload - a missing field here is the difference between
    "the tier is safe" and "the tier is safe, probably".
    """
    if manager is None:
        return _fields()
    return _fields(**_live_view(manager))


def empty_section(reason: str) -> dict:
    """The degraded section a caller falls back to. Same key set, by construction."""
    return _fields(error=str(reason)[:200])
