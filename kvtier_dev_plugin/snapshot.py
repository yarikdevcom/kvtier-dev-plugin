"""Read-only view of the scheduler-side KV tier. Stdlib imports only.

This module runs on the thread that serves traffic: an engine-side utility
method is dispatched from the EngineCore busy loop
(vllm/v1/engine/core.py:1669-1671 -> 1712-1728), the same thread that runs
`step_fn()` (core.py:1579). So:

  * imports are `json`, `os`, `time`, `secrets` - nothing else. No torch, no
    CUDA, no vllm, no numpy. `test_no_heavy_imports_in_read_side()` enforces
    it by AST. This mirrors `vllm/v1/simple_kv_offload/manager.py`, which
    contains zero torch references by design; the worker side
    (`simple_kv_offload/worker.py:293-294`) records on the live compute stream
    and is never touched from here.
  * every read is O(1) on the real objects: `len()` of a dict, a maintained
    counter (`BlockPool.get_num_free_blocks()` ->
    `free_block_queue.num_free_blocks`, block_pool.py:861), or a dataclass copy
    (`get_boundary_store_stats()` returns `replace(...)`, manager.py:947-948,
    so it is snapshot-safe by construction).
  * nothing here raises into the caller: `snapshot_for_engine()` is the
    containment boundary, in this fork's idiom
    (vllm/v1/core/sched/e2_alloc_log.py:143,151,201 with the capped `_error()`
    at :92). It catches `Exception`, never `BaseException`: a
    KeyboardInterrupt or SystemExit must not be swallowed by a diagnostic.

Field names below were read out of the fork at HEAD (10b8e07):
  manager.py  : `_lazy_mode`, `_target_free`, `block_size`, `hash_block_size`,
                `fa_block_size`, `fa_gidx`, `group_block_sizes`,
                `num_cpu_blocks`, `cpu_block_pool`, `_gpu_block_pool`,
                `_reqs_to_load`, `_reqs_to_store`, `_pending_cpu_hits`,
                `_load_event_to_reqs`, `_store_event_to_blocks`,
                `_abandoned_store_event_to_blocks`, `_abandoned_reqs_to_load`,
                `_pending_finished_stores`, `_in_flight_store_gpu_blocks`,
                `_store_event_pending_counts`, `_load_event_counter`,
                `_store_event_counter`, `boundary_store_stats`
  manager.py  : `has_pending_stores()` (:1119),
                `get_boundary_store_stats()` (:947),
                `BoundaryStoreStats` fields (:125-132)
  connector   : `scheduler_manager` (simple_cpu_offload_connector.py:129,157)
  block_pool  : `num_gpu_blocks` (:173), `get_num_free_blocks()` (:861)
"""

from __future__ import annotations

import json
import secrets
import time

PROTOCOL_VERSION = 1

# Containment state, in the style of e2_alloc_log._state: count every error
# forever, log at most the first MAX_ERROR_LOGS of them, and never let the
# writer itself become the second failure.
MAX_ERROR_LOGS = 20
_state = {"calls": 0, "errors": 0, "logged": 0, "epoch": None, "logger": None}

# The eight decline reasons the tier counts, manager.py:125-132. Kept as a
# tuple so a fork that renames one shows up as an `unknown_fields` entry here
# rather than as a silently missing number.
BOUNDARY_STAT_FIELDS = (
    "published",
    "stored",
    "dropped_cpu_full",
    "dropped_request_gone",
    "skipped_already_cached",
    "skipped_in_flight",
    "dropped_null_block",
    "dropped_not_hashed",
)


def attach_logger(logger) -> None:
    """Handed a logger by the engine-side installer; None keeps _error quiet."""
    _state["logger"] = logger


def error_counters() -> dict:
    return {"calls": _state["calls"], "errors": _state["errors"],
            "logged": _state["logged"]}


def epoch() -> str:
    """Per-process id: proves two snapshots came from the same engine boot.

    Generated lazily so importing this module costs no syscall and no entropy.
    """
    if _state["epoch"] is None:
        _state["epoch"] = secrets.token_hex(4)
    return _state["epoch"]


def _error(exc: BaseException, where: str) -> None:
    """Count, log up to a cap, and never raise. `exc` is never re-raised."""
    _state["errors"] += 1
    logger = _state["logger"]
    if logger is None or _state["logged"] >= MAX_ERROR_LOGS:
        return
    _state["logged"] += 1
    try:  # noqa: SIM105 - no contextlib import for the last resort
        logger.warning("[KVTIER_DEV] %s failed (%d/%d logged): %s: %s",
                       where, _state["logged"], MAX_ERROR_LOGS,
                       type(exc).__name__, exc)
    except Exception:  # noqa: BLE001, S110
        # Deliberate: the last-resort writer gets no contextlib import, exactly
        # like the nested `except Exception: pass` inside the fork's own
        # e2_alloc_log._error() at :92. A broken log must be silent, never fatal.
        pass


def _int_or_none(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _count(container) -> int | None:
    """len() of a real container, None if the attribute is missing/weird."""
    if container is None:
        return None
    try:
        return len(container)
    except TypeError:
        return None


def _pool_view(pool) -> dict:
    """Total/free for a BlockPool. Both reads are maintained counters."""
    if pool is None:
        return {"bound": False}
    free = None
    try:
        getter = getattr(pool, "get_num_free_blocks", None)
        if callable(getter):
            free = _int_or_none(getter())
    except Exception as exc:  # noqa: BLE001 - a pool that lies is data, not a crash
        _error(exc, "pool.get_num_free_blocks")
    return {"bound": True,
            "total_blocks": _int_or_none(getattr(pool, "num_gpu_blocks", None)),
            "free_blocks": free}


# --- the host-RAM parked-KV HASH INDEX ---------------------------------------
#
# THE QUESTION THIS ANSWERS: "how much partial KV is actually retrievable in host
# RAM right now?" `cpu_pool.free_blocks` cannot answer it, and reading the two
# numbers side by side without this block is actively misleading:
#
#     cpu_pool.total_blocks = 665   cpu_pool.free_blocks = 664   -> "1 in use"
#     boundary_store_stats.stored = 4372                          -> "4372 parked"
#
# Both are true. A parked CPU block is handed BACK TO THE ALLOCATOR when its store
# finishes (`BlockPool.free_blocks`/`unpin_blocks` only move it in the free queue;
# block_pool.py:705-741, :770-798), and what keeps it retrievable afterwards is its
# entry in `cpu_block_pool.cached_block_hash_to_block` - the hash index the fork
# itself looks hits up in (`get_one_block`, manager.py:605, :940, :1251). Allocator
# freeness does NOT mean the index is empty. Occupancy of the index is the number
# that means "retrievable".
#
# BOUNDED: one `len()` on a sized container, then at most `MAX_HASH_INDEX_SCAN`
# keys and `MAX_HASH_INDEX_SAMPLE` values for the per-group split and the sanity
# sample. Never a full walk of an index that could be 100k entries on a big host.
MAX_HASH_INDEX_SCAN = 20000
MAX_HASH_INDEX_GROUPS = 64
MAX_HASH_INDEX_SAMPLE = 32
# A map value is a KVCacheBlock, or a {block_id: block} dict when one key carries
# several blocks (the map does not de-duplicate, block_pool.py:47-51). A fan-out
# wider than this is a fork this reader does not understand, and it refuses the
# block view instead of counting a prefix of it.
MAX_BLOCKS_PER_KEY = 8
# The per-group block-id sets are the ONLY container this scan grows beyond a
# counter: at most one id is added per key, so the total is bounded by the key
# scan itself - MAX_HASH_INDEX_SCAN ints worst case, and in production the pool
# size, not the entry count, is the ceiling (665 blocks per rank here).
MAX_PARKED_BLOCK_IDS = MAX_HASH_INDEX_SCAN
HASH_INDEX_ATTR = "cached_block_hash_to_block"
HASH_INDEX_MEANING = (
    "parked entries in the host-RAM KV hash index, NOT allocator allocations: "
    "this is how much parked KV can be HIT. Distinct from cpu_pool.free_blocks "
    "- a parked CPU block is returned to the allocator when its store finishes, "
    "and the index entry is what keeps it retrievable, so a nearly-all-free pool "
    "can still hold thousands of retrievable blocks")
# The fork's own recycling contract, read out of the source rather than guessed:
# an entry is removed SYNCHRONOUSLY when its block is handed out again.
STALE_ENTRIES_VERDICT = "not_expected_on_pool_paths"
STALE_ENTRIES_WHY = (
    "BlockPool.get_new_blocks calls _maybe_evict_cached_block -> "
    "_remove_cached_block_hashes on every block it hands out "
    "(block_pool.py:664-688, :725-745), and unpin_blocks documents the blocks it "
    "releases as readable UNTIL reused (:705-723). So an entry does not outlive "
    "the bytes it points at on the pool's own paths: the entry is dropped at the "
    "moment of reuse, not kept over a later overwrite")
STALE_ENTRIES_NOT_RULED_OUT = (
    "a fork path that wrote CPU bytes into a block WITHOUT taking it from "
    "get_new_blocks would defeat the above, and this snapshot cannot see the copy "
    "itself; the sampled_values_without_hash counter is the empirical tripwire")


def _group_of_key(key):
    """The group id a `BlockHashWithGroupId` key belongs to, or None.

    The fork packs it as 4 big-endian unsigned bytes at the END of the key
    (kv_cache_utils.py:74-93, `make_block_hash_with_group_id`/`get_group_id`).
    Read locally rather than imported: this module must not import vllm.
    """
    if isinstance(key, (bytes, bytearray)) and len(key) > 4:
        # "> 4", not ">= 4": 4 bytes would be a group id with an EMPTY hash, which
        # is not a key the fork can produce (the hash comes first).
        return int.from_bytes(bytes(key)[-4:], "big")
    return None


def _entry_count(index):
    """(count, how, refusal_reason). `len()` first - BlockHashToBlockMap defines
    __len__ over its keys - then any own size/count method.

    A `len()` that RAISES is reported as that, not as "no size handle": the two
    need different answers from an operator, and a vague reason here would hide a
    pool that is actively misbehaving.
    """
    if hasattr(type(index), "__len__"):
        try:
            return len(index), "len", None
        except Exception as exc:  # noqa: BLE001 - a container that raises is data
            _error(exc, "hash index len")
            return (None, "len",
                    f"{type(index).__name__}.__len__ raised "
                    f"{type(exc).__name__}: no occupancy number invented")
    for name in ("num_entries", "count", "size"):
        getter = getattr(index, name, None)
        if callable(getter):
            try:
                return _int_or_none(getter()), name, None
            except Exception as exc:  # noqa: BLE001
                _error(exc, f"hash index {name}")
                return (None, name,
                        f"{type(index).__name__}.{name}() raised "
                        f"{type(exc).__name__}: no occupancy number invented")
    return (None, None,
            f"{type(index).__name__} exposes neither len() nor a size/count "
            f"method: no occupancy number invented")


def _hash_index_view(pool, hash_block_size) -> dict:
    """Occupancy of the CPU pool's parked-KV hash index. Read-only, contained.

    A missing attribute or an unsized container is a `reason` string, never a
    number: a missing reading is information, a fabricated one is a lie.
    """
    out: dict = {"available": False, "attribute": HASH_INDEX_ATTR,
                 "meaning": HASH_INDEX_MEANING,
                 "stale_entries": STALE_ENTRIES_VERDICT,
                 "stale_entries_why": STALE_ENTRIES_WHY,
                 "stale_entries_not_ruled_out": STALE_ENTRIES_NOT_RULED_OUT,
                 "retrievable_now_vs_retained": (
                     "an entry's block is an eviction candidate the moment its "
                     "store releases it, so this is retrievable NOW, not a "
                     "retention guarantee: the next stores can reclaim it")}
    if pool is None:
        out["reason"] = "no cpu_block_pool bound: no host-RAM tier in this process"
        return out
    try:
        index = getattr(pool, HASH_INDEX_ATTR, None)
        if index is None:
            out["reason"] = (f"{type(pool).__name__} has no {HASH_INDEX_ATTR}: the "
                             f"fork renamed the index, refusing to guess where "
                             f"parked KV is tracked")
            return out
        out["container"] = type(index).__name__
        count, counted_by, refusal = _entry_count(index)
        out["counted_by"] = counted_by
        if count is None:
            out["reason"] = refusal
            return out
        out["available"] = True
        # Entries, not blocks: a key may map to SEVERAL blocks (the map does not
        # de-duplicate, block_pool.py:47-51), so count the keys and say what they
        # are. A hit resolves through get_one_block(), so keys are exactly the
        # distinct prefixes that can be hit.
        out["entries"] = count
        if isinstance(hash_block_size, int) and not isinstance(hash_block_size, bool) \
                and hash_block_size > 0:
            out["hash_block_size"] = hash_block_size
            out["tokens_upper_bound"] = count * hash_block_size
            out["tokens_note"] = (
                f"entries x {hash_block_size}: an UPPER BOUND on distinct "
                "retrievable tokens (one index entry is one hash block; shared "
                "prefixes are counted once, duplicate parkings are not)")
        _hash_index_breakdown(index, out, pool)
        _hash_index_sanity_sample(index, out)
        return out
    except Exception as exc:  # noqa: BLE001 - a diagnostic never reaches the scheduler
        _error(exc, "hash index view")
        out["reason"] = f"reading {HASH_INDEX_ATTR} raised: " \
                        f"{type(exc).__name__}"[:200]
        return out


# --- parked BLOCKS and the primary/secondary key split ----------------------
#
# `per_group` counts KEYS of the hash index, and the groups do not mint keys at the
# same rate per parked span. The primary hash is one key per `block_size`-token
# span; the fine-graining group ALSO mints its intra-block secondary hashes, and
# they are inserted against the SAME `cpu_block` with `num_tokens=None`
# (manager.py:1039-1044) - dictionary entries that cost no block and no bytes. So
# `per_group` is inflated by a per-group divisor that is not in the payload, and a
# consumer that sums or mins it is wrong in a way it cannot discover.
#
# Measured live on 2026-10-05 (read-only GET /plugins/kvtier/snapshot, idle box):
# `per_group {0:94, 1:94, 2:94, 3:95, 5:317}` against `cpu_pool.total_blocks = 665`
# - group 5 carries 3.4x the keys of its siblings, and the ratio DRIFTS (317,
# 317, 318, 320, 320, 320 over six polls of one idle minute), which is exactly why
# no fixed divisor may be divided by anywhere downstream.
#
# What a consumer needs instead is spans, and spans ARE blocks: one parked span
# occupies exactly one CPU block per group (every group's GPU block goes into one
# list and `get_new_blocks()` takes that many DISTINCT ids via `popleft_n()`,
# manager.py:755-758 / block_pool.py:678). Hence distinct block ids per group.
PARKED_BLOCKS_MEANING = (
    "distinct CPU BLOCKS behind the index keys of each group, from the same bounded "
    "scan as `per_group`: one parked span occupies one block per group, so this is "
    "SPANS parked in the group. Unlike `per_group` it is not inflated by the "
    "fine-graining group's secondary hashes (same cpu_block, num_tokens=None, "
    "manager.py:1039-1044). min() over the PREFIX-CACHEABLE groups only is the "
    "joint depth a whole-model resume can use; x tier.block_size gives tokens")
PARKED_BLOCKS_SOURCE = (
    "one pass over the same key scan as `per_group`: per key, one map lookup plus "
    "at most MAX_BLOCKS_PER_KEY attribute reads, and one set.add of an int. The "
    "only container growth is the per-group id sets, bounded by "
    "MAX_PARKED_BLOCK_IDS ints in total (one id per scanned key at most, and in "
    "practice by the pool size - 665 blocks per rank here)")
KEY_SPLIT_MEANING = (
    "per-group split of `per_group` by what the entry IS. primary = the key the "
    "block itself carries (block.block_hash, the one whose num_tokens was recorded "
    "by set_block_hash); secondary = an extra key under which the SAME block is "
    "also indexed (BlockPool.cached_block_hashes_by_block, populated by "
    "_insert_block_hash at block_pool.py:641 from manager.py:1039-1044). "
    "primary[g] should equal parked_blocks_per_group[g]: that equality is the "
    "cross-check, and key_split_matches_blocks states it")


def _index_value_of(index, keys):
    """(fn(key) -> map value, how). Two handles, in preference order.

    The map's own dict is preferred because its value carries the key's FULL block
    set; the public `get_one_block()` is what a HIT resolves to, so a key holding
    several blocks is necessarily collapsed to one - and the payload has to say so.
    """
    if isinstance(keys, dict):
        return keys.get, "map value via _cache: a key's full block set is visible"
    get_one = getattr(index, "get_one_block", None)
    if callable(get_one):
        return (get_one,
                "get_one_block(): what a hit resolves to, so a key carrying "
                "several blocks is counted as the one a hit would use")
    return None, None


def _key_blocks_and_kind(key, value, by_block):
    """(block_ids | None, kind | None) for one index entry. O(1) attribute reads.

    `block_ids` is None when the value is not the shape the fork documents (then
    the whole block view is refused, not guessed). `kind` is None when the entry is
    neither the block's own key nor a member of that block's extra-hash set - the
    case the key split must NOT quietly file under "secondary".
    """
    if value is None:
        return None, None
    blocks = value
    if not isinstance(blocks, dict):
        blocks = (value,)
    if len(blocks) > MAX_BLOCKS_PER_KEY:
        return None, None
    ids: list = []
    own = False
    extra = False
    for block in (blocks.values() if isinstance(value, dict) else blocks):
        block_id = getattr(block, "block_id", None)
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            return None, None
        ids.append(block_id)
        if getattr(block, "block_hash", None) == key:
            own = True
        elif by_block is not None and key in by_block.get(block_id, ()):
            extra = True
    if own:
        return ids, "primary"
    if extra and by_block is not None:
        return ids, "secondary"
    return ids, None


def _hash_index_breakdown(index, out: dict, pool=None) -> None:
    """Per-group entry counts AND per-group parked BLOCK counts, from ONE bounded
    key scan. Best effort, and it says which handle it read and when it stopped.

    The key counts alone are not enough for a consumer: the groups mint keys at
    different rates per span (see PARKED_BLOCKS_MEANING), so a sum or a min over
    `per_group` is an upper bound wearing an occupancy's name. The block counts are
    what a consumer needs, and they come from this same pass at no extra scan cost.
    """
    keys = None
    handle = None
    for name in ("keys", HASH_INDEX_ATTR, "_cache"):
        getter = getattr(index, name, None)
        if callable(getter):
            try:
                keys = getter()
                handle = name + "()"
                break
            except Exception as exc:  # noqa: BLE001 - a handle that raises is a miss
                _error(exc, f"hash index {name}")
        elif isinstance(getter, dict):
            keys = getter
            handle = name
            break
    if keys is None:
        out["per_group"] = None
        out["per_group_reason"] = (
            f"no public keys() and no dict-like attribute to enumerate: "
            f"{type(index).__name__} exposes no key handle")
        _refuse(out, "parked_blocks_per_group",
                out["per_group_reason"])
        _refuse(out, "per_group_primary", out["per_group_reason"])
        out["per_group_secondary"] = None
        out["per_group_secondary_reason"] = out["per_group_reason"]
        return
    out["key_handle"] = handle
    if handle == "_cache":
        out["key_handle_note"] = (
            "read through the map's private _cache dict (dev-only, read-only): "
            "BlockHashToBlockMap has no public key iterator")
    value_of, blocks_via = _index_value_of(index, keys)
    by_block = getattr(pool, "cached_block_hashes_by_block", None)
    if not isinstance(by_block, dict):
        by_block = None
    groups: dict = {}
    block_sets: dict = {}
    primary: dict = {}
    secondary: dict = {}
    scanned = 0
    unparsable = 0
    truncated = False
    partial = False
    ids_stored = 0
    ids_capped = False
    unresolvable = 0
    unclassified = 0
    multi_block = 0
    try:
        for key in keys:
            if scanned >= MAX_HASH_INDEX_SCAN:
                truncated = True
                break
            scanned += 1
            group = _group_of_key(key)
            if group is None:
                unparsable += 1
                continue
            if len(groups) < MAX_HASH_INDEX_GROUPS or group in groups:
                groups[group] = groups.get(group, 0) + 1
                if value_of is None:
                    continue
                if ids_stored >= MAX_PARKED_BLOCK_IDS:
                    ids_capped = True
                    continue
                ids, kind = _key_blocks_and_kind(key, value_of(key), by_block)
                if ids is None:
                    # A value this reader cannot turn into block ids. Refused
                    # below, per field: a count of the keys it understood would
                    # look like a count of the index.
                    unresolvable += 1
                    continue
                if len(ids) > 1:
                    multi_block += 1
                bucket = block_sets.get(group)
                if bucket is None:
                    bucket = block_sets[group] = set()
                bucket.update(ids)
                ids_stored += len(ids)
                if kind is None:
                    unclassified += 1
                elif kind == "primary":
                    primary[group] = primary.get(group, 0) + 1
                else:
                    secondary[group] = secondary.get(group, 0) + 1
    except Exception as exc:  # noqa: BLE001 - a half-scan is still a reading
        _error(exc, "hash index per-group scan")
        partial = True
    out["entries_scanned"] = scanned
    out["scan_truncated"] = truncated
    out["per_group"] = ({str(name): groups[name] for name in sorted(groups)}
                        if groups else {})
    out["groups_truncated"] = len(groups) >= MAX_HASH_INDEX_GROUPS
    out["keys_without_group_id"] = unparsable
    if partial:
        out["per_group_partial"] = True
    blocks_ok = _publish_parked_blocks(
        out, groups, block_sets, truncated=truncated, partial=partial,
        groups_truncated=out["groups_truncated"], ids_capped=ids_capped,
        unresolvable=unresolvable, multi_block=multi_block,
        value_missing=value_of is None, blocks_via=blocks_via,
        scanned=scanned, unparsable=unparsable)
    _publish_key_split(out, groups, primary, secondary, by_block=by_block,
                       truncated=truncated, partial=partial,
                       unclassified=unclassified, blocks_ok=blocks_ok,
                       block_sets=block_sets, scanned=scanned,
                       unparsable=unparsable)


def _refuse(out: dict, field: str, reason: str) -> None:
    """Publish a field as null WITH its reason. Never a 0, never a guess: a missing
    reading is information, a fabricated one is a lie (and every `or 0` in this
    session's history turned into a number somebody believed)."""
    out[field] = None
    out[field + "_reason"] = reason


def _publish_parked_blocks(out: dict, groups: dict, block_sets: dict, *,
                           truncated: bool, partial: bool,
                           groups_truncated: bool, ids_capped: bool,
                           unresolvable: int, multi_block: int,
                           value_missing: bool, blocks_via: str | None,
                           scanned: int, unparsable: int) -> bool:
    """Emit `parked_blocks_per_group`, or the reason it is absent. True if emitted."""
    out["parked_blocks_meaning"] = PARKED_BLOCKS_MEANING
    out["parked_blocks_source"] = PARKED_BLOCKS_SOURCE
    field = "parked_blocks_per_group"
    if not groups:
        # An index that is genuinely empty is a real zero; an index whose keys this
        # reader could not attribute to a group is not.
        if scanned == 0:
            out[field] = {}
        else:
            _refuse(out, field,
                    f"{unparsable} of {scanned} scanned keys carried no parsable "
                    "group id and no key did: no group is claimable")
        return False
    if value_missing:
        _refuse(out, field,
                "the index exposes neither a _cache dict nor get_one_block(), so no "
                "block identity is reachable: keys can be counted, blocks cannot")
    elif truncated:
        _refuse(out, field,
                f"the key scan stopped at {MAX_HASH_INDEX_SCAN} entries "
                "(scan_truncated): a distinct-block count over a PREFIX of the "
                "index silently under-counts, which is the error class this field "
                "exists to remove")
    elif partial:
        _refuse(out, field, "the key scan raised mid-pass (per_group_partial): a "
                            "half-count of distinct blocks would be a number with "
                            "no stated coverage")
    elif groups_truncated:
        _refuse(out, field,
                f"more than {MAX_HASH_INDEX_GROUPS} groups were seen, so a min() "
                "over the published groups would be a min over a subset of them")
    elif ids_capped:
        _refuse(out, field,
                f"the block-id sets hit their {MAX_PARKED_BLOCK_IDS}-id bound; "
                "the count is a prefix, not the set")
    elif unresolvable:
        _refuse(out, field,
                f"{unresolvable} index values were neither a KVCacheBlock nor a "
                f"block dict with int ids (fan-out > {MAX_BLOCKS_PER_KEY} counts as "
                "ununderstood): counting the readable ones would mislabel a partial"
                " as a total")
    else:
        out[field] = {str(name): len(block_sets[name])
                      for name in sorted(block_sets)}
        out["parked_blocks_seen_via"] = blocks_via
        out["keys_backed_by_multiple_blocks"] = multi_block
        return True
    return False


def _publish_key_split(out: dict, groups: dict, primary: dict, secondary: dict, *,
                       by_block, truncated: bool, partial: bool,
                       unclassified: int, blocks_ok: bool,
                       block_sets: dict, scanned: int, unparsable: int) -> None:
    """Emit `per_group_primary`/`per_group_secondary`, or why they are absent.

    The discriminator is structural, not a divisor: a key is PRIMARY when the block
    it points at carries that key as its own `block_hash` (the one insert that got
    to call `set_block_hash`, and the only one that carried a real `num_tokens`),
    and SECONDARY when the pool's own extra-hash set for that block id contains it
    (`cached_block_hashes_by_block`, block_pool.py:641). Anything else is an
    unclassified entry and refuses the split outright.
    """
    out["per_group_key_split_meaning"] = KEY_SPLIT_MEANING
    if not groups:
        if scanned == 0:
            out["per_group_primary"] = {}
            out["per_group_secondary"] = {}
        else:
            _refuse(out, "per_group_primary",
                    f"{unparsable} of {scanned} scanned keys carried no parsable "
                    "group id and no key did: no group is claimable")
            out["per_group_secondary"] = None
            out["per_group_secondary_reason"] = out["per_group_primary_reason"]
        return
    if by_block is None:
        _refuse(out, "per_group_primary",
                "the CPU pool exposes no cached_block_hashes_by_block dict: a key "
                "that is not its block's own hash cannot be POSITIVELY identified "
                "as an extra-hash entry, and 'not primary' is not the same claim as "
                "'secondary'")
        out["per_group_secondary"] = None
        out["per_group_secondary_reason"] = out["per_group_primary_reason"]
        return
    if unclassified:
        _refuse(out, "per_group_primary",
                f"{unclassified} entries were neither their block's own hash nor a "
                "member of that block's extra-hash set - a fork this reader does "
                "not fully model, so the split is withheld")
        out["per_group_secondary"] = None
        out["per_group_secondary_reason"] = out["per_group_primary_reason"]
        return
    if truncated or partial:
        _refuse(out, "per_group_primary",
                "the key scan was truncated or raised mid-pass: a partial split "
                "would not sum to per_group and the cross-check would be noise")
        out["per_group_secondary"] = None
        out["per_group_secondary_reason"] = out["per_group_primary_reason"]
        return
    out["per_group_primary"] = {str(name): primary.get(name, 0)
                                for name in sorted(groups)}
    out["per_group_secondary"] = {str(name): secondary.get(name, 0)
                                  for name in sorted(groups)}
    if blocks_ok:
        # The cross-check the consumers asked for: one primary key per block, so the
        # two must agree entry for entry. A disagreement means the "one block per
        # span" premise (or a multi-block key) is what failed, not the arithmetic.
        out["key_split_matches_blocks"] = (
            out["per_group_primary"]
            == {str(name): len(block_sets[name]) for name in sorted(block_sets)})



def _hash_index_sanity_sample(index, out: dict) -> None:
    """Tripwire for a stale entry, from at most MAX_HASH_INDEX_SAMPLE values.

    A live entry's block still carries its hash and is not the null block. A
    sampled value that lost its hash while still indexed is exactly the "
    "stale entry" the count would otherwise silently include.
    """
    values = None
    for name in ("values", "_cache"):
        handle = getattr(index, name, None)
        try:
            if callable(handle):
                values = handle()
                break
            if isinstance(handle, dict):
                values = handle.values()
                break
        except Exception as exc:  # noqa: BLE001 - no sample is a missing reading
            _error(exc, f"hash index {name}")
    if values is None:
        out["sampled_entries"] = 0
        out["sample_reason"] = "no value handle on the index"
        return
    sampled = 0
    missing_hash = 0
    null_blocks = 0
    dicts_of_blocks = 0
    try:
        for value in values:
            if sampled >= MAX_HASH_INDEX_SAMPLE:
                break
            sampled += 1
            if isinstance(value, dict):
                # A key with more than one block behind it (no de-duplication).
                dicts_of_blocks += 1
                value = next(iter(value.values()), None)
            if value is None:
                missing_hash += 1
                continue
            if getattr(value, "is_null", False):
                null_blocks += 1
            if getattr(value, "block_hash", None) is None:
                missing_hash += 1
    except Exception as exc:  # noqa: BLE001 - a partial sample is still evidence
        _error(exc, "hash index sample")
        out["sample_partial"] = True
    out["sampled_entries"] = sampled
    out["sampled_values_without_hash"] = missing_hash
    out["sampled_null_blocks"] = null_blocks
    out["sampled_keys_with_multiple_blocks"] = dicts_of_blocks


def _copy_boundary_stats(stats) -> dict:
    """Copy the counter dataclass into plain ints (already a copy upstream)."""
    if stats is None:
        return {"available": False}
    try:
        mapping = vars(stats)
    except TypeError:
        return {"available": False,
                "reason": f"not a simple object: {type(stats).__name__}"}
    out: dict = {"available": True}
    for key, value in mapping.items():
        as_int = _int_or_none(value)
        out[key] = value if as_int is not None else repr(value)[:64]
    missing = [f for f in BOUNDARY_STAT_FIELDS if f not in out]
    unknown = [f for f in mapping
               if f not in BOUNDARY_STAT_FIELDS and f != "available"]
    if missing:
        out["missing_fields"] = missing
    if unknown:
        out["unknown_fields"] = unknown
    return out


def tier_view(manager) -> dict:
    """The scheduler-side offload manager, or a reason it is not there."""
    if manager is None:
        return {"present": False,
                "reason": "no scheduler_manager: the engine is running without "
                          "SimpleCPUOffloadConnector, or prefix caching is off"}

    gpu_pool = getattr(manager, "_gpu_block_pool", None)
    pending = None
    try:
        has_pending = getattr(manager, "has_pending_stores", None)
        if callable(has_pending):
            pending = bool(has_pending())
    except Exception as exc:  # noqa: BLE001
        _error(exc, "has_pending_stores")

    try:
        stats = _copy_boundary_stats(manager.get_boundary_store_stats())
    except Exception as exc:  # noqa: BLE001
        _error(exc, "get_boundary_store_stats")
        stats = {"available": False, "reason": "get_boundary_store_stats raised"}

    group_sizes = getattr(manager, "group_block_sizes", None)
    return {
        "present": True,
        "mode": "lazy" if getattr(manager, "_lazy_mode", False) else "eager",
        "target_free_gpu_blocks": _int_or_none(getattr(manager, "_target_free", None)),
        "block_size": _int_or_none(getattr(manager, "block_size", None)),
        "hash_block_size": _int_or_none(getattr(manager, "hash_block_size", None)),
        "fa_block_size": _int_or_none(getattr(manager, "fa_block_size", None)),
        "fa_group_index": _int_or_none(getattr(manager, "fa_gidx", None)),
        "group_block_sizes": (list(group_sizes) if group_sizes is not None else None),
        "num_cpu_blocks": _int_or_none(getattr(manager, "num_cpu_blocks", None)),
        "cpu_pool": _pool_view(getattr(manager, "cpu_block_pool", None)),
        # The retrievability half, next to the allocator half because the two must
        # be read together: `cpu_pool.free_blocks` is freeness, THIS is how much
        # parked KV a later request can actually hit.
        "cpu_hash_index": _hash_index_view(getattr(manager, "cpu_block_pool", None),
                                           _int_or_none(
                                               getattr(manager, "hash_block_size",
                                                       None))),
        "gpu_pool": _pool_view(gpu_pool),
        "pending_stores": pending,
        "counts": {
            "reqs_to_load": _count(getattr(manager, "_reqs_to_load", None)),
            "reqs_to_store": _count(getattr(manager, "_reqs_to_store", None)),
            "abandoned_reqs_to_load": _count(
                getattr(manager, "_abandoned_reqs_to_load", None)),
            "load_events_to_reqs": _count(
                getattr(manager, "_load_event_to_reqs", None)),
            "store_events": _count(getattr(manager, "_store_event_to_blocks", None)),
            "abandoned_store_events": _count(
                getattr(manager, "_abandoned_store_event_to_blocks", None)),
            "store_events_awaiting_workers": _count(
                getattr(manager, "_store_event_pending_counts", None)),
            "pending_finished_stores": _count(
                getattr(manager, "_pending_finished_stores", None)),
            "in_flight_store_gpu_blocks": _count(
                getattr(manager, "_in_flight_store_gpu_blocks", None)),
            "pending_cpu_hits": _count(getattr(manager, "_pending_cpu_hits", None)),
        },
        "event_counters": {
            "load_event_counter": _int_or_none(
                getattr(manager, "_load_event_counter", None)),
            "store_event_counter": _int_or_none(
                getattr(manager, "_store_event_counter", None)),
        },
        "boundary_store_stats": stats,
    }


def evictions_section(engine) -> dict:
    """The LRU-eviction view: the arena's steady state, made visible.

    Function-level import because `evictions` imports THIS module (for
    `_error`), the same cycle `store_hook_section`/`cpu_capacity_section` avoid
    this way. Contained: a section that cannot be built is a reason, never a
    500 - and it must never be a bare zero (an unwrapped pool reports nulls).
    """
    from . import evictions  # noqa: PLC0415 - cycle, see store_hook_section
    try:
        return evictions.section(engine)
    except Exception as exc:  # noqa: BLE001 - the shape must survive regardless
        _error(exc, "evictions section")
        return evictions.empty_section(f"{type(exc).__name__}: {exc}")


def gpu_census_section(engine) -> dict:
    """The GPU pool's own census: matchable-but-unreferenced blocks, per group.

    Why the gauge needs it: `BlockPool.get_usage()` counts every `ref_cnt == 0`
    block as free, and a queued block that still carries a `block_hash` remains
    matchable through `get_cached_block` (which checks no refcount) until
    `_maybe_evict_cached_block` discards it - so "evicted" and "cached but idle"
    were literally the same number. See gpu_census.py's header for the fork
    references and for the one thing the pool CANNOT answer (per-group unhashed
    counts, because all groups share one pool and an unhashed block has no group).

    Contained the same way every other section is: a census that cannot run is a
    `reason` with null counts, never a zero and never an exception into the engine.
    Function-level import because `gpu_census` imports THIS module.
    """
    from . import gpu_census  # noqa: PLC0415 - cycle, see store_hook_section
    try:
        return gpu_census.section(engine)
    except Exception as exc:  # noqa: BLE001 - the shape must survive regardless
        _error(exc, "gpu census section")
        return gpu_census.empty_section(f"{type(exc).__name__}: {exc}")


def build_view(engine) -> dict:
    """One pass over the engine's scheduler-side state. Reads only."""
    scheduler = getattr(engine, "scheduler", None)
    connector = getattr(scheduler, "connector", None)
    manager = getattr(connector, "scheduler_manager", None)
    tier = tier_view(manager)
    # The eviction counter rides the tier view (the arena lives here), but it is
    # the one tier field that needs the ENGINE (for its corroboration), so it is
    # attached below tier_view(manager) rather than inside it.
    tier["evictions"] = evictions_section(engine)
    # Same reason for the census: it resolves the pool through the engine (the
    # scheduler's own `kv_cache_manager.block_pool`) so it survives a boot whose
    # tier connector is absent, and its group labels come from `kv_cache_config`.
    tier["gpu_pool_census"] = gpu_census_section(engine)
    return {
        "version": PROTOCOL_VERSION,
        "epoch": epoch(),
        "ts": round(time.time(), 6),
        "mono": round(time.monotonic(), 6),
        "engine": {"class": type(engine).__name__},
        "scheduler": {
            "present": scheduler is not None,
            "class": type(scheduler).__name__ if scheduler is not None else None,
            "connector": type(connector).__name__ if connector is not None else None,
            "requests": _count(getattr(scheduler, "requests", None)),
            "running": _count(getattr(scheduler, "running", None)),
            "waiting": _count(getattr(scheduler, "waiting", None)),
        },
        "tier": tier,
    }


def policy_section(view, holder) -> dict:
    """Run the operator's policy over the view. Contained, and timed.

    `decide_us` is the evidence that the policy is cheap enough to sit on the
    engine loop thread: if it ever stops being microseconds, it shows up here.
    """
    started = time.perf_counter()
    try:
        report = holder.ensure_loaded()
        if not report.get("ok"):
            return {"loaded": False, "decide_us": None, "result": None,
                    "error": report.get("error"), "traceback": None}
        decide = holder.decide
        if decide is None:
            return {"loaded": True, "decide_us": None, "result": None,
                    "error": "policy module has no decide()", "traceback": None}
        result = decide(view)
        return {"loaded": True,
                "decide_us": int((time.perf_counter() - started) * 1e6),
                "result": result, "error": None, "traceback": None}
    except Exception as exc:  # noqa: BLE001 - BaseException must propagate
        _error(exc, "policy.decide")
        return {"loaded": True,
                "decide_us": int((time.perf_counter() - started) * 1e6),
                "result": None,
                "error": f"{type(exc).__name__}: {exc}"[:300], "traceback": None}


def json_safe(value: dict) -> dict:
    """Force the payload to msgpack/JSON primitives before it leaves the engine.

    The utility result is msgpack-encoded by the EngineCore output thread; a
    value it cannot encode would fail there, not here. A JSON round trip leaves
    only dict/list/str/int/float/bool/None behind. allow_nan=False also turns
    a stray NaN/inf into a contained error instead of JSON the HTTP layer
    cannot emit.
    """
    return json.loads(json.dumps(value, allow_nan=False,
                                 separators=(",", ":")))


def store_hook_section(engine) -> dict:
    """Whether the runtime store-policy hook is live, and what it has done.

    The import is function-level on purpose: `store_hook` imports THIS module
    (for `_error`), so a module-level import here would be a cycle. By the time
    a handler runs, both modules are fully loaded, and the lookup costs one
    `sys.modules` hit per snapshot - not per block, which is where the real
    per-call cost of the hook lives.
    """
    from . import store_hook  # noqa: PLC0415 - see above
    try:
        return store_hook.patch_report(store_hook.manager_of(engine))
    except Exception as exc:  # noqa: BLE001 - a missing report is data, not a 500
        _error(exc, "store hook report")
        return {"patched": False,
                "reason": f"report failed: {type(exc).__name__}"[:200]}


def cpu_capacity_section(engine) -> dict:
    """Whether the CPU block-id ceiling is clamped to what every rank can hold.

    The one section of the view that reports a SAFETY property rather than a
    measurement: `installed` / `reason` (and why not), the per-rank arena sizes
    the handshake carried, the chosen cap, whether the clamp was applied, whether
    the tier is in fail-safe zero mode, and how many allocations the cap has
    refused. Function-level import because `cpu_capacity` imports THIS module.
    """
    from . import cpu_capacity  # noqa: PLC0415 - see above
    manager = None
    try:
        manager = cpu_capacity.manager_of(engine)
    except Exception as exc:  # noqa: BLE001 - a manager we cannot reach is data
        _error(exc, "cpu capacity manager")
    try:
        return cpu_capacity.section(manager)
    except Exception as exc:  # noqa: BLE001 - the shape must survive regardless
        _error(exc, "cpu capacity section")
        return cpu_capacity.empty_section(f"{type(exc).__name__}: {exc}")


def backport_section(engine) -> dict:
    """The PR #56810 backport: is the ring-group fix live, and is the tier armed?

    Two questions in one section because they are read together. (1) Is the
    upstream divisibility fix actually installed - all four methods, or did a
    fingerprint refuse one? (2) Is the tier armed, i.e. is the connector allowed
    to schedule any store or load at all? `armed` is the kill-switch state and it
    belongs in the SAME section as the patch it governs: "patched and disarmed"
    and "not patched" look similar in a payload and mean very different things
    (one serves from GPU KV alone, the other crashes on a hybrid geometry).

    Function-level import because `backport` imports THIS module.
    """
    from . import backport  # noqa: PLC0415 - cycle, see store_hook_section
    manager = None
    try:
        manager = backport.manager_of(engine)
    except Exception as exc:  # noqa: BLE001 - a manager we cannot reach is data
        _error(exc, "backport manager")
    try:
        return backport.section(manager)
    except Exception as exc:  # noqa: BLE001 - the shape must survive regardless
        _error(exc, "backport section")
        return backport.empty_section(f"{type(exc).__name__}: {exc}")


def dev_state_sections() -> dict:
    """The DEVELOPMENT-state sections: guard mode, the trace ring, the worker writer,
    the lookup ledger.

    These are the four pieces of the plugin that CHANGE what the engine does (or
    put a wrapper on its path), so every snapshot publishes them. Guard mode in
    particular must be impossible to miss: `snapshot_for_engine` also copies it to
    the top level as `guard_mode`, because a guarded dev engine mistaken for a
    production one is the single worst way this plugin could hurt.

    Function-level imports again: `guard_mode` records into `trace_ring` and
    `kvconfig` reads `guard_mode`, so a module-level import here would be a cycle.
    """
    from . import guard_mode, step_trace, trace_ring, worker_diag  # noqa: PLC0415
    out: dict = {}
    try:
        out["guard"] = guard_mode.section()
        out["guard_mode"] = guard_mode.is_armed()
    except Exception as exc:  # noqa: BLE001 - a missing section is data, not a 500
        _error(exc, "guard section")
        out["guard"] = {"guard_mode": None, "error": f"{type(exc).__name__}"[:200]}
        out["guard_mode"] = None
    try:
        counters = trace_ring.counters()
        counters["step_observers"] = step_trace.counters()
        out["trace"] = counters
    except Exception as exc:  # noqa: BLE001
        _error(exc, "trace section")
        out["trace"] = {"error": f"{type(exc).__name__}"[:200]}
    try:
        out["worker_diag"] = worker_diag.section()
    except Exception as exc:  # noqa: BLE001
        _error(exc, "worker diag section")
        out["worker_diag"] = {"error": f"{type(exc).__name__}"[:200]}
    try:
        # The lookup/adopt ledger. Counters plus a BOUNDED preview: the payload is
        # polled every few seconds, so the ring itself stays behind the verb and
        # only `lookup_ledger.preview` (a handful of the newest records) travels.
        from . import lookup_ledger  # noqa: PLC0415 - cycle, see dev_state_sections
        out["lookup_ledger"] = lookup_ledger.section()
        out["lookup_ledger_armed"] = lookup_ledger.is_armed()
    except Exception as exc:  # noqa: BLE001
        _error(exc, "lookup ledger section")
        out["lookup_ledger"] = {"error": f"{type(exc).__name__}"[:200]}
        out["lookup_ledger_armed"] = None
    return out


def snapshot_for_engine(engine, holder) -> dict:
    """THE engine-side handler body. Never raises an Exception; reads only."""
    _state["calls"] += 1
    started = time.perf_counter()
    try:
        view = build_view(engine)
        view["policy"] = policy_section(view, holder)
        view["store_hook"] = store_hook_section(engine)
        view["cpu_capacity"] = cpu_capacity_section(engine)
        view["pr56810"] = backport_section(engine)
        view.update(dev_state_sections())
        # `dev_state_sections()` already put `guard_mode` at the top level: the
        # field a reader must never have to dig for. `guard` is the detail.
        view["build_us"] = int((time.perf_counter() - started) * 1e6)
        view["self"] = error_counters()
        return json_safe(view)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        _error(exc, "snapshot")
        # Hand-built from formatted strings only: this path cannot raise. The
        # guard question is still the first thing a reader wants, even here.
        return {"version": PROTOCOL_VERSION,
                "epoch": str(epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "guard_mode": _guard_mode_quietly(),
                "counters": {"calls": int(_state["calls"]),
                             "errors": int(_state["errors"]),
                             "logged": int(_state["logged"])}}


def _guard_mode_quietly():
    """guard_mode.is_armed() that cannot fail inside the containment path."""
    try:
        from . import guard_mode  # noqa: PLC0415 - cycle, see dev_state_sections
        return guard_mode.is_armed()
    except Exception:  # noqa: BLE001, S112 - the last resort stays silent
        return None


def reload_policy_on_engine(holder) -> dict:
    """THE engine-side reload handler body. Never raises; reports as data."""
    _state["calls"] += 1
    try:
        report = holder.reload()
        report["self"] = error_counters()
        report["epoch"] = epoch()
        return report
    except Exception as exc:  # noqa: BLE001
        _error(exc, "reload")
        return {"ok": False, "rolled_back": True, "kept": None, "applied": None,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "traceback": None, "self": {"calls": int(_state["calls"]),
                                            "errors": int(_state["errors"]),
                                            "logged": int(_state["logged"])}}
