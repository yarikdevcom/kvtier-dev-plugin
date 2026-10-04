"""The GPU block pool's census: how much VRAM KV is matchable but unreferenced.

THE HOLE THIS CLOSES
--------------------
`BlockPool.get_usage()` (block_pool.py:876-888) is `1 - free/(total-1)`, and
`get_num_free_blocks()` (block_pool.py:861) is `free_block_queue.num_free_blocks` -
the count of blocks whose `ref_cnt` fell to 0. But a block whose refcount hit 0 is
NOT discarded: `free_blocks` (block_pool.py:772-798) sorts it into the free queue by
exactly ONE predicate, `block.block_hash is None` -> `prepend_n` (queue head, reused
first) versus everything else -> `append_n` (queue tail, LRU-aged). A queued block
that still carries a `block_hash` is therefore counted as FREE by the gauge while
remaining fully matchable: `get_cached_block` (block_pool.py:204) resolves hits
through `cached_block_hash_to_block` and never checks a refcount, and the entry only
dies in `_maybe_evict_cached_block` (block_pool.py:726), which a block reaches when
`get_new_blocks` (block_pool.py:664-688) hands it out. So "evicted" and "cached but
idle" are the SAME NUMBER on every existing gauge, and no consumer could tell them
apart. This module publishes the missing split.

THE THREE CLASSES, and the identity that keeps them honest
----------------------------------------------------------
Every block in `pool.blocks` is exactly one of:

  * `null_blocks`        - `is_null` (the block_id=0 placeholder; its refcount is
                           deliberately not maintained, so it is counted APART and
                           never folded into the free-queue reconciliation);
  * `referenced`         - `ref_cnt != 0`: owned by a live request;
  * `cached_idle`        - `ref_cnt == 0` AND `block_hash is not None`: sitting in
                           the free queue, counted free, STILL MATCHABLE - the
                           number this whole section exists to produce;
  * `free_uncached`      - `ref_cnt == 0` AND `block_hash is None`: genuinely free,
                           can never match a prefix.

`null_blocks + referenced + cached_idle + free_uncached == walked_blocks` is the
walk's own identity and is published as `sum_matches_walk`; `cached_idle +
free_uncached == free_queue_reported` is the engine's own counter agreeing with the
walk, published as `free_queue_reconciles`. A disagreement is a finding, not a
rounding error, so both are published rather than assumed.

WHY THE HEAD/TAIL ORDER MAKES `cached_idle` THE INTERESTING NUMBER
------------------------------------------------------------------
`free_blocks` prepends the UNHASHED (LIFO reuse for GPU locality, so they are
reused almost immediately) and appends the HASHED (FIFO/LRU). The head of the queue
is thus "doomed within a few allocations" and the tail is "aged, retained until
pressure reaches it" - and the tail is where `cached_idle` lives. The gauge cannot
see that difference; the census can, because the hashed/unhashed predicate is the
SAME predicate the queue's ordering is derived from.

PER GROUP, AND THE LIMIT OF WHAT A BLOCK KNOWS
----------------------------------------------
A block's group is not stored on it as a number: the group id is the LAST FOUR
big-endian bytes of its `BlockHashWithGroupId` (kv_cache_utils.py:74-93,
`make_block_hash_with_group_id`/`get_group_id`). So a HASHED block is attributable
to exactly one group - its primary hash is the only one carried on the block, the
fine-graining secondary hashes live in the pool's `cached_block_hashes_by_block` -
and the per-group maps below partition cleanly (they sum to the totals). An
UNHASHED block carries NO group identity at all, and cannot: every group allocates
from ONE shared `BlockPool` (`kv_cache_coordinator.py:98`, `block_pool =
BlockPool(num_gpu_blocks=kv_cache_config.num_blocks)`, reached by every group's
manager and bound to the tier as `_gpu_block_pool`), and `get_new_blocks` takes
`popleft_n(n)` distinct ids for n groups at once. A free unhashed block belongs to
whichever group asks for it NEXT. Hence `unhashed_per_group` is `null` WITH A REASON
and never a guess - and `free_uncached` is published only as a pool-wide count.

This is the honest limit of the data structure, and it matters for the question
being asked of this panel ("is the joint hit clamped by the state group's sparse
retention?"): the per-group HASHED counts are exactly what that question needs
(how many recurrent states are resident and matchable right now, against the
attention groups' thousands), while the per-group UNHASHED split - "is group 4's
pool 100% unhashed?" - is unanswerable from the pool and is reported as such rather
than approximated.

GROUP LABELS, DERIVED NEVER HARDCODED
--------------------------------------
Group indices are meaningless without their specs, and the numbering has already
been mis-stated once, so `group_labels` reads each group's spec off the live config
(`scheduler.kv_cache_config.kv_cache_groups[g].kv_cache_spec`) and publishes the
CLASS NAME plus the geometry (`block_size`, `page_size_bytes`,
`prefix_cacheable`, and `is_circular_buffer` decided by MRO class name exactly the
way the fork's own `is_circular_buffer()` does). Nothing here imports vllm: every
read is by attribute NAME, and a fork that renamed something publishes a `reason`
instead of a guess. `retention` carries the cache-config knobs that explain a sparse
state group (`prefix_cache_retention_interval`, `mamba_cache_mode`,
`enable_mamba_fine_grained_prefix_cache`, `prefix_match_unit`) when readable.

COST (this runs on the EngineCore loop thread - see snapshot.py's header)
-------------------------------------------------------------------------
One pass over `pool.blocks`, a plain `list[KVCacheBlock]` of length
`num_gpu_blocks` (317 on the boot this was written against): three attribute reads
per block (`ref_cnt`, `block_hash`, `is_null` - `block_hash` is a property returning
`_block_hash`, kv_cache_utils.py:202-208) and, for a hashed block, one
`int.from_bytes` of the last 4 bytes. No allocation per block beyond an entry in a
per-group counter, and no container growth bounded by anything but the group count.
O(blocks), no I/O, no CUDA, no torch, no vllm, and bounded above by
`MAX_WALKED_BLOCKS`: a pool larger than that is REFUSED outright rather than
partially counted, because a partial census is exactly the "prefix dressed as a
total" error class this repo refuses elsewhere. Measured on this machine, one block
at a time in CPython: ~0.6 us/block (185 us for the live 317-block pool, 1.7 ms for
3,170, 19 ms for 31,700) - linear, because nothing here is quadratic and the only
container that grows is bounded by the group count.

CONTAINMENT
-----------
Everything here is inside `try/except Exception` (never `BaseException`: a
KeyboardInterrupt or SystemExit must propagate, the same rule as
`snapshot.snapshot_for_engine`), counted through `snapshot._error`, and reported as
`available: False` with a `reason` and NULL counts - never a zero. A zero here would
read as "no VRAM prefix is retained", which is the lie this section exists to kill.
"""

from __future__ import annotations

import time

from . import snapshot

# The section's own contract version. Bumped when the KEY SET changes, so a reader
# can tell "a plugin that cannot see this" from "a plugin whose census refused".
SCHEMA_VERSION = 1

# Hard bounds. `MAX_WALKED_BLOCKS` is the walk bound (refuse, never truncate);
# `MAX_GROUPS` bounds the per-group maps and the label table.
#
# The walk costs ~0.6 us per block measured (median `walk_us`: 185 us at the live
# 317-block pool, 1.7 ms at 3,170, 19 ms at 31,700), and it runs on the thread that
# serves the utility call. 50,000 blocks therefore caps one census at ~30 ms of
# that thread - loud enough to be noticed in `build_us`, small enough that a poll
# every few seconds cannot turn a diagnostic into a stutter. A pool above the bound
# is REFUSED: a truncated census is a prefix dressed as a total.
MAX_WALKED_BLOCKS = 50_000
MAX_GROUPS = 64
MAX_STRING_CHARS = 96
MAX_LABEL_ATTRS = 8

# The pool handle this section reads, in resolution order. Both spellings name the
# SAME instance (base.py:475-476 binds `kv_cache_manager.block_pool`, which is
# `coordinator.block_pool`, onto the connector); the second is kept because a build
# that has the tier bound but not the manager shape must still be censusable.
POOL_SOURCES = (
    ("scheduler.kv_cache_manager.block_pool",
     lambda engine: getattr(getattr(getattr(engine, "scheduler", None),
                                    "kv_cache_manager", None), "block_pool", None)),
    ("scheduler.connector.scheduler_manager._gpu_block_pool",
     lambda engine: getattr(getattr(getattr(getattr(engine, "scheduler", None),
                                             "connector", None),
                                    "scheduler_manager", None),
                            "_gpu_block_pool", None)),
)

# The declared field set: every key `_fields` emits, and every key emitted is
# declared. tests/test_gpu_census.py pins this with the AST, both directions, the
# same way tests/test_monitor_schema.py pins `evictions.EXPECTED_FIELDS`.
EXPECTED_FIELDS = (
    "available", "reason", "error", "schema_version", "pure_read",
    "pool_source", "pool_class", "pool_enable_caching",
    "walked_blocks", "blocks_total", "walked_bound", "walk_us",
    "referenced", "referenced_hashed", "referenced_unhashed",
    "cached_idle", "free_uncached", "null_blocks", "negative_ref_cnt_blocks",
    "hashed_per_group", "hashed_idle_per_group", "hashed_referenced_per_group",
    "per_group_reason",
    "unhashed_per_group", "unhashed_per_group_reason",
    "hashed_groups_unattributed", "groups_truncated",
    "free_queue_reported", "free_queue_reconciles",
    "sum_matches_walk", "walked_matches_pool_size",
    "joint_hashed_blocks_min", "joint_hashed_blocks_min_meaning",
    "group_labels", "group_labels_reason",
    "retention", "retention_reason",
    "meaning", "cached_idle_meaning", "free_uncached_meaning",
)
# A degraded view is allowed to stop emitting nothing: the key set is constant so a
# consumer never has to branch on structure. Empty by design.
ALLOW_UNEMITTED: tuple = ()

MEANING = (
    "a per-block census of the GPU KV block pool, walked once per snapshot: the "
    "gauge's `free` is split into blocks that are still matchable (cached_idle) and "
    "blocks that can never match (free_uncached). Not a rate and not a lifetime: a "
    "cached_idle block is an eviction candidate the next time get_new_blocks reaches "
    "its queue position")
CACHED_IDLE_MEANING = (
    "ref_cnt == 0 AND block_hash is not None: in the free queue, counted FREE by "
    "BlockPool.get_usage(), and still resolvable by get_cached_block "
    "(block_pool.py:204 checks no refcount). This is VRAM prefix data the current "
    "gauge reports as free space, and it is queued at the queue's TAIL "
    "(free_blocks append_n's hashed blocks, block_pool.py:796-798), i.e. aged, not "
    "about-to-be-reused. It dies only in _maybe_evict_cached_block "
    "(block_pool.py:726) when get_new_blocks hands the block out")
FREE_UNCACHED_MEANING = (
    "ref_cnt == 0 AND block_hash is None: genuinely free and can never be a prefix "
    "cache hit. free_blocks prepend_n's these to the queue HEAD "
    "(block_pool.py:794-795, LIFO for GPU locality), so they are the blocks a busy "
    "engine reuses first - the opposite end of the queue from cached_idle")
UNHASHED_PER_GROUP_WHY = (
    "not derivable from the pool, and deliberately left null rather than "
    "approximated: a block carries no group number, only its hash (whose last 4 "
    "bytes ARE the group id, kv_cache_utils.py:74-93), and ALL groups allocate from "
    "ONE shared BlockPool (kv_cache_coordinator.py:98; get_new_blocks takes "
    "popleft_n(n) ids for n groups at once). An unhashed block belongs to whichever "
    "group allocates it NEXT, so per-group unhashed counts exist only as a total")
JOINT_MIN_MEANING = (
    "min over the prefix-cacheable groups of hashed_per_group: an UPPER BOUND on how "
    "many joint spans every cacheable group could answer at once, NOT a hit length "
    "and NOT a guarantee - hashed blocks are scattered across different prefixes, so "
    "the joint hit can be far shorter. It is published because a group whose hashed "
    "count is far below the attention groups CAPS the joint hit at that count, which "
    "is the one direction this figure is sound in")

# Attributes worth publishing per group spec. Read by name; absent means absent.
LABEL_SPEC_ATTRS = ("block_size", "page_size_bytes", "prefix_cacheable",
                    "tokens_per_state", "num_speculative_blocks",
                    "num_prefill_checkpoint_blocks", "blocks_per_request",
                    "sliding_window")
RETENTION_CONFIG_ATTRS = ("prefix_cache_retention_interval", "mamba_cache_mode",
                          "enable_mamba_fine_grained_prefix_cache",
                          "prefix_match_unit", "enable_prefix_caching",
                          "block_size")


def _fields(**overrides) -> dict:
    """THE builder for the section's key set. One dict, so a healthy view and a
    degraded view can never drift apart in shape (the evictions.py idiom)."""
    fields = {
        "available": False,
        "reason": None,
        "error": None,
        "schema_version": SCHEMA_VERSION,
        "pure_read": True,
        "pool_source": None,
        "pool_class": None,
        "pool_enable_caching": None,
        "walked_blocks": None,
        "blocks_total": None,
        "walked_bound": MAX_WALKED_BLOCKS,
        "walk_us": None,
        "referenced": None,
        "referenced_hashed": None,
        "referenced_unhashed": None,
        "cached_idle": None,
        "free_uncached": None,
        "null_blocks": None,
        "negative_ref_cnt_blocks": None,
        "hashed_per_group": None,
        "hashed_idle_per_group": None,
        "hashed_referenced_per_group": None,
        "per_group_reason": None,
        "unhashed_per_group": None,
        "unhashed_per_group_reason": UNHASHED_PER_GROUP_WHY,
        "hashed_groups_unattributed": None,
        "groups_truncated": None,
        "free_queue_reported": None,
        "free_queue_reconciles": None,
        "sum_matches_walk": None,
        "walked_matches_pool_size": None,
        "joint_hashed_blocks_min": None,
        "joint_hashed_blocks_min_meaning": JOINT_MIN_MEANING,
        "group_labels": None,
        "group_labels_reason": None,
        "retention": None,
        "retention_reason": None,
        "meaning": MEANING,
        "cached_idle_meaning": CACHED_IDLE_MEANING,
        "free_uncached_meaning": FREE_UNCACHED_MEANING,
    }
    fields.update(overrides)
    return fields


def empty_section(reason: str) -> dict:
    """The refused view: every count null, with the reason. Never a zero."""
    return _fields(available=False, reason=reason)


# --- group decoding, local: this module must never import vllm ----------------

def _group_of_key(key):
    """The group id packed at the END of a `BlockHashWithGroupId`, or None.

    Mirrors `kv_cache_utils.get_group_id` (:91) and `snapshot._group_of_key`; kept
    as a direct call into snapshot so the two cannot drift.
    """
    return snapshot._group_of_key(key)


def _count_value(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _scalar(value, max_chars: int = MAX_STRING_CHARS):
    """A JSON-safe scalar copy: ints/bools as-is, anything else stringified+clipped."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    return text[:max_chars] if len(text) > max_chars else text


# --- the walk ----------------------------------------------------------------

def _walk(blocks: list) -> dict:
    """One pass over the pool's block list. Raises on a block it cannot classify.

    Raising is the POINT: the caller turns it into a refusal, whereas publishing the
    counts gathered so far would publish a prefix of the pool as a total - the exact
    error class every `_refuse` in this repo exists to prevent.
    """
    referenced = referenced_hashed = referenced_unhashed = 0
    cached_idle = free_uncached = null_blocks = negative_ref_cnt = 0
    hashed_per_group: dict = {}
    hashed_idle_per_group: dict = {}
    hashed_referenced_per_group: dict = {}
    unattributed = 0
    truncated_groups = False

    for block in blocks:
        ref = getattr(block, "ref_cnt", None)
        if isinstance(ref, bool) or not isinstance(ref, int):
            raise TypeError(f"block ref_cnt is {type(ref).__name__}, not an int")
        key = getattr(block, "block_hash", None)
        is_null = getattr(block, "is_null", False) is True
        if ref < 0:
            negative_ref_cnt += 1

        hashed = key is not None
        group = _group_of_key(key) if hashed else None
        if hashed and group is None:
            # A hash this reader cannot decode into a group id: counted in the
            # totals, NOT in any group. Inventing a group would be the worse error.
            unattributed += 1

        # Exactly one bucket per block, in this order: the null placeholder's
        # refcount is deliberately unmaintained (block_pool.py:190-192), so it is
        # counted apart from everything else.
        if is_null:
            null_blocks += 1
        elif ref != 0:
            referenced += 1
            if hashed:
                referenced_hashed += 1
            else:
                referenced_unhashed += 1
        elif hashed:
            cached_idle += 1
        else:
            free_uncached += 1

        if hashed and group is not None:
            slot = str(group)
            if group in hashed_per_group or len(hashed_per_group) < MAX_GROUPS:
                hashed_per_group[slot] = hashed_per_group.get(slot, 0) + 1
                # The two splits cover the SAME block set as hashed_per_group, so
                # the identity sum(idle) + sum(referenced) == sum(hashed) holds.
                if is_null:
                    pass  # a null block is never cached; counted in hashed only
                elif ref != 0:
                    hashed_referenced_per_group[slot] = \
                        hashed_referenced_per_group.get(slot, 0) + 1
                else:
                    hashed_idle_per_group[slot] = \
                        hashed_idle_per_group.get(slot, 0) + 1
            else:
                truncated_groups = True

    if not truncated_groups:
        # Same key set in all three maps (a group with nothing idle reports 0), so a
        # consumer can zip them without a KeyError and the identity is checkable.
        for slot in hashed_per_group:
            hashed_idle_per_group.setdefault(slot, 0)
            hashed_referenced_per_group.setdefault(slot, 0)

    return {
        "referenced": referenced,
        "referenced_hashed": referenced_hashed,
        "referenced_unhashed": referenced_unhashed,
        "cached_idle": cached_idle,
        "free_uncached": free_uncached,
        "null_blocks": null_blocks,
        "negative_ref_cnt_blocks": negative_ref_cnt,
        "hashed_per_group": hashed_per_group,
        "hashed_idle_per_group": hashed_idle_per_group,
        "hashed_referenced_per_group": hashed_referenced_per_group,
        "hashed_groups_unattributed": unattributed,
        "groups_truncated": truncated_groups,
    }


def census(pool) -> dict:
    """The census of ONE GPU `BlockPool`. Contained: never raises, never a zero.

    `pool.blocks` is the whole pool (`BlockPool.__init__`, block_pool.py:173-175), so
    the walk covers every block by construction - which is why a refusal is the only
    acceptable answer to a partial walk.
    """
    if pool is None:
        return empty_section("no GPU block pool bound: neither "
                             "scheduler.kv_cache_manager.block_pool nor the tier "
                             "manager's _gpu_block_pool is reachable here")
    try:
        blocks = getattr(pool, "blocks", None)
        if blocks is None:
            return empty_section(f"{type(pool).__name__} exposes no `blocks`: the "
                                 f"fork renamed the pool's block list, refusing to "
                                 f"guess where the blocks live")
        if not isinstance(blocks, (list, tuple)):
            return empty_section(f"{type(pool).__name__}.blocks is "
                                 f"{type(blocks).__name__}, not a list: no walk "
                                 f"performed and no count invented")
        walked = len(blocks)
        if walked > MAX_WALKED_BLOCKS:
            # A partial census is a prefix wearing a total's name: refuse outright.
            return empty_section(f"pool holds {walked} blocks, above the "
                                 f"{MAX_WALKED_BLOCKS} walk bound: a partial census "
                                 f"would under-count, which is the error class this "
                                 f"section exists to remove")
        started = time.perf_counter()
        counts = _walk(list(blocks))
        walk_us = int((time.perf_counter() - started) * 1e6)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler, never a lie
        snapshot._error(exc, "gpu census walk")
        return empty_section(f"the block walk raised {type(exc).__name__}: "
                             f"{str(exc)[:160]} - no partial count published")

    out = _fields(available=True, **counts)
    out["walked_blocks"] = walked
    out["walk_us"] = walk_us
    out["pool_class"] = type(pool).__name__
    out["pool_enable_caching"] = getattr(pool, "enable_caching", None)
    out["blocks_total"] = _count_value(getattr(pool, "num_gpu_blocks", None))
    out["walked_matches_pool_size"] = (
        out["blocks_total"] == walked if out["blocks_total"] is not None else None)
    out["sum_matches_walk"] = (
        counts["null_blocks"] + counts["referenced"] + counts["cached_idle"]
        + counts["free_uncached"] == walked)

    free = None
    try:
        getter = getattr(pool, "get_num_free_blocks", None)
        if callable(getter):
            free = _count_value(getter())
    except Exception as exc:  # noqa: BLE001 - the engine's counter is a witness, not a dependency
        snapshot._error(exc, "gpu census get_num_free_blocks")
    out["free_queue_reported"] = free
    out["free_queue_reconciles"] = (
        counts["cached_idle"] + counts["free_uncached"] == free
        if free is not None else None)

    # Per-group maps: published only when they partition the totals cleanly. A map
    # that cannot be trusted is nulled WITH ITS REASON, never shipped as a partial.
    if counts["groups_truncated"]:
        reason = (f"more than {MAX_GROUPS} group ids appeared in the walk, so a min "
                  f"over the published groups would be a min over a subset")
    elif counts["hashed_groups_unattributed"]:
        reason = (f"{counts['hashed_groups_unattributed']} hashed blocks carried a "
                  f"hash this reader cannot decode into a group id "
                  f"(BlockHashWithGroupId ends with 4 big-endian group bytes): the "
                  f"per-group maps would not sum to the totals")
    else:
        reason = None
    if reason is not None:
        out["hashed_per_group"] = None
        out["hashed_idle_per_group"] = None
        out["hashed_referenced_per_group"] = None
        out["per_group_reason"] = reason
    return out


# --- the labels and the retention knobs, read off the live config -------------

def group_labels(engine) -> tuple[dict | None, str | None]:
    """{group index -> spec class + geometry}, or the reason it is unavailable.

    Group INDICES mean nothing without their specs and the numbering has been
    mis-stated before, so the class name is the label and it is READ, never assumed.
    Cheap by construction: one `getattr` chain per group, at most `MAX_GROUPS`
    groups, no `dir()` (unlike the kvconfig verb, which takes one `dir()` per spec -
    this runs on the loop thread, that one runs on a verb).
    """
    try:
        config = getattr(getattr(engine, "scheduler", None), "kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", None)
        if groups is None:
            return None, ("scheduler.kv_cache_config.kv_cache_groups is missing: no "
                          "group geometry to label the census with")
        total = len(groups)
        if total > MAX_GROUPS:
            return None, (f"{total} KV cache groups exceed the {MAX_GROUPS} label "
                          f"bound: the label table is refused, not truncated")
        out: dict = {}
        for index, group in enumerate(groups):
            spec = getattr(group, "kv_cache_spec", None)
            entry: dict = {"index": index,
                           "spec_class": type(spec).__name__ if spec is not None
                           else None}
            if spec is not None:
                attrs = {}
                for name in LABEL_SPEC_ATTRS[:MAX_LABEL_ATTRS]:
                    if hasattr(spec, name):
                        attrs[name] = _scalar(getattr(spec, name, None))
                entry["spec"] = attrs
                entry["is_circular_buffer"] = _is_circular_buffer(spec)
                entry["block_size"] = _count_value(getattr(spec, "block_size", None))
            entry["is_eagle_group"] = getattr(group, "is_eagle_group", None) is True
            entry["host_resident"] = getattr(group, "host_resident", None) is True
            out[str(index)] = entry
        return out, None
    except Exception as exc:  # noqa: BLE001 - labels are context, never a crash
        snapshot._error(exc, "gpu census group labels")
        return None, f"reading the group specs raised {type(exc).__name__}"


def _is_circular_buffer(spec) -> bool | None:
    """The fork's own MRO-by-class-name test (kvconfig.py:241-255), read locally."""
    try:
        return any(cls.__name__ == "CircularBufferSpec"
                   for cls in type(spec).__mro__)
    except Exception:  # noqa: BLE001 - a label we cannot decide stays unknown
        return None


def retention(scheduler) -> tuple[dict | None, str | None]:
    """The cache-config knobs that explain a sparse state group, or the reason.

    `prefix_cache_retention_interval` is why a state group hashes only boundary
    blocks, so it belongs in the same section as the counts it explains.
    """
    try:
        cache_config = getattr(scheduler, "cache_config", None)
        if cache_config is None:
            return None, "scheduler exposes no cache_config"
        out: dict = {}
        for name in RETENTION_CONFIG_ATTRS:
            if hasattr(cache_config, name):
                out[name] = _scalar(getattr(cache_config, name, None))
        config = getattr(scheduler, "kv_cache_config", None)
        if config is not None and hasattr(config, "prefix_cache_retention_interval"):
            out["kv_cache_config.prefix_cache_retention_interval"] = _scalar(
                getattr(config, "prefix_cache_retention_interval", None))
        if not out:
            return None, ("neither cache_config nor kv_cache_config carries any of "
                          "the retention attributes this reader knows")
        return out, None
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "gpu census retention")
        return None, f"reading the cache config raised {type(exc).__name__}"


def pool_of(engine) -> tuple[object, str | None]:
    """The GPU `BlockPool` and which spelling reached it. First hit wins."""
    for label, resolve in POOL_SOURCES:
        try:
            pool = resolve(engine)
        except Exception as exc:  # noqa: BLE001 - a property that raises is one miss
            snapshot._error(exc, f"gpu census pool lookup {label}")
            continue
        if pool is not None:
            return pool, label
    return None, None


def section(engine) -> dict:
    """The whole section: the walk, plus the labels that make its groups readable.

    Called by `snapshot.build_view` through a contained wrapper; a failure here is a
    reason there, and every path out of here is a dict with the declared key set.
    """
    pool, source = pool_of(engine)
    out = census(pool)
    # Which spelling reached the pool: both name the same instance in the fork, and a
    # reader must be able to tell which one this build answered through.
    out["pool_source"] = source
    scheduler = getattr(engine, "scheduler", None)
    labels, labels_reason = group_labels(engine)
    out["group_labels"] = labels
    out["group_labels_reason"] = labels_reason
    knobs, retention_reason = retention(scheduler)
    out["retention"] = knobs
    out["retention_reason"] = retention_reason
    out["joint_hashed_blocks_min"] = _joint_min(out, labels)
    return out


def _joint_min(out: dict, labels: dict | None) -> int | None:
    """min over the PREFIX-CACHEABLE groups' hashed counts, or None.

    Non-cacheable groups (the scratch ring) are excluded by their spec's own
    `prefix_cacheable` flag when the labels carry it; without labels the min would be
    over an unlabeled set, so None is published instead.
    """
    per_group = out.get("hashed_per_group")
    if not isinstance(per_group, dict) or not per_group or not labels:
        return None
    wanted = []
    for key, counts in per_group.items():
        label = labels.get(key)
        if not isinstance(label, dict):
            return None  # an unlabelled group cannot be excluded or included
        cacheable = label.get("spec", {}).get("prefix_cacheable")
        if cacheable is True:
            wanted.append(counts)
    if not wanted:
        return None
    return min(wanted)
