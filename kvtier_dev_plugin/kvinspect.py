"""Capability 2: "show me block 412" and "what does this request hold?".

Two inspectors, both argument-taking, both bounded, both usable with NO connector
(the state this tier is being developed in): they read the engine's own GPU block
pool and per-request block tables, which exist whether or not a tier does.

WHERE EACH FIELD COMES FROM (fork at HEAD 10b8e07)
--------------------------------------------------
`scheduler.requests` (scheduler.py:228) is `dict[request_id -> Request]`.
`Request`: `num_computed_tokens` request.py:182, `num_tokens` :288,
`num_output_tokens` :296, `num_prompt_tokens` :141, `num_output_placeholders`
:160, `num_preemptions` :210, `status` :98, `block_hashes` :219 (computed at
`hash_block_size` granularity), `all_token_ids` :192.
Per-group block ids: `coordinator.single_type_managers[g].req_to_blocks[req_id]`
(single_type_kv_cache_manager.py:133), each entry a `KVCacheBlock`.
`KVCacheBlock` (kv_cache_utils.py:176): `block_id`, `ref_cnt`, `is_null`, the
`block_hash` property :203 (a `BlockHashWithGroupId`, i.e. RAW BYTES = the hash
plus a 4-byte big-endian group id, kv_cache_utils.py:76-92), the
`block_hash_num_tokens` property :207, and the free-list links
`prev_free_block`/`next_free_block` :200-201.
`BlockPool` (block_pool.py): `blocks` :178, `num_gpu_blocks` :173,
`free_block_queue` :184, `cached_block_hash_to_block` :187 with
`get_one_block(key)` :61 - the SAME lookup the tier performs at manager.py:940 -
and `get_num_free_blocks()` :861.
"In the free queue?" is the queue's own test: `FreeKVCacheBlockQueue.remove()`
raises unless both links are set, so both-set IS membership (the
`cpu_capacity._in_free_queue` test).

"HOW MANY OF ITS HASHES ARE COVERABLE" IS THE TIER'S OWN EXPRESSION
--------------------------------------------------------------------
The tier can only park a block whose hash it can view at that group's size, and
only up to what the request has actually computed. `manager.py:812-844` computes
exactly that:

    confirmed_tokens = request.num_computed_tokens - request.num_output_placeholders
    store_alignment  = hash_block_size if cpu_coordinator.enable_partial_hash_hits
                       else block_size
    aligned_tokens   = confirmed_tokens // store_alignment * store_alignment
    ready            = min(len(group_gpu_ids), aligned_tokens // group_size)
    resolved_hashes  = resolve_block_hashes(request.block_hashes, hash_block_size,
                                            group_size)   # <-- the assert

`hashes_at_group_size` is what `len(resolved_hashes)` WOULD be (the number of
`block_size`-granularity hashes derivable from the request's
`hash_block_size`-granularity list, i.e. `len // (group_size // hash_block_size)`
- and `None`, not 0, when the divisibility assert would fire first).
`tier_ready_blocks` recomputes the fork's `ready` here so a "the tier parked
nothing" investigation starts from numbers instead of guesses.

BOUNDS: ids are capped (`MAX_IDS` per list, requested cap honoured up to
`MAX_IDS_LIMIT`) and every capped list says how many it dropped. No unbounded
walk runs on the engine loop thread. Contained by `inspect_for_engine()`:
`Exception` yes, `BaseException` never.
"""

from __future__ import annotations

import time

from . import guard_mode, kvconfig, snapshot, trace_ring

PROTOCOL_VERSION = 1

MAX_IDS = 256
MAX_IDS_LIMIT = 1024
DEFAULT_IDS = 64
MAX_REQUEST_IDS = 128
MAX_GROUP_ENTRIES = 64
MAX_STRING_CHARS = 96

_HASH_KEY_GROUP_BYTES = 4  # kv_cache_utils.py:83 (4-byte big-endian group id)


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _ids_cap(args: dict) -> int:
    requested = kvconfig._int((args or {}).get("ids_cap"))
    if requested is None or requested <= 0:
        return DEFAULT_IDS
    return min(MAX_IDS_LIMIT, requested)


def _clip(text, chars: int = MAX_STRING_CHARS):
    if isinstance(text, str) and len(text) > chars:
        return text[:chars] + "..."
    return text


def _hash_key_view(key) -> dict:
    """Decode a `BlockHashWithGroupId` (raw bytes) into something encodable.

    Bytes are NOT msgpack/JSON safe as-is, so this is where they become hex. The
    split follows the fork's own helpers: `get_block_hash` is `key[:-4]` and
    `get_group_id` is the last 4 bytes big-endian (kv_cache_utils.py:86-92).
    """
    if key is None:
        return {"has_hash": False}
    if not isinstance(key, (bytes, bytearray)):
        return {"has_hash": True,
                "hash_unrenderable": type(key).__name__}
    raw = bytes(key)
    group_id = None
    hash_part = raw
    if len(raw) >= _HASH_KEY_GROUP_BYTES:
        group_id = int.from_bytes(raw[-_HASH_KEY_GROUP_BYTES:], "big",
                                  signed=False)
        hash_part = raw[:-_HASH_KEY_GROUP_BYTES]
    return {"has_hash": True,
            "hash_hex": hash_part[:16].hex(),
            "hash_bytes": len(hash_part),
            "hash_key_bytes": len(raw),
            "hash_group_id": group_id}


# --- the block inspector ----------------------------------------------------

def _block_state(pool, block, group) -> dict:
    """One KVCacheBlock, in the terms an operator asks about."""
    block_id = _int(getattr(block, "block_id", None))
    hash_view = _hash_key_view(getattr(block, "block_hash", None))
    in_free = (getattr(block, "prev_free_block", None) is not None
               and getattr(block, "next_free_block", None) is not None)
    in_cached = None
    if hash_view.get("has_hash") and isinstance(
            getattr(block, "block_hash", None), (bytes, bytearray)):
        cached_map = getattr(pool, "cached_block_hash_to_block", None)
        getter = getattr(cached_map, "get_one_block", None)
        if callable(getter):
            try:
                # The same O(1) keyed lookup the tier uses (manager.py:940).
                in_cached = getter(block.block_hash) is not None
            except Exception as exc:  # noqa: BLE001 - a map that lies is data
                snapshot._error(exc, "inspect.get_one_block")
    previous = getattr(block, "prev_free_block", None)
    following = getattr(block, "next_free_block", None)
    hash_group = hash_view.get("hash_group_id")
    out = {
        "block_id": block_id,
        "ref_cnt": _int(getattr(block, "ref_cnt", None)),
        "is_null": bool(getattr(block, "is_null", False)),
        "in_free_queue": in_free,
        "prev_free_block_id": _int(getattr(previous, "block_id", None)),
        "next_free_block_id": _int(getattr(following, "block_id", None)),
        "in_cached_map": in_cached,
        "block_hash_num_tokens": _int(
            getattr(block, "block_hash_num_tokens", None)),
        **hash_view,
    }
    if group is not None and hash_group is not None:
        out["hash_group_matches_request"] = hash_group == group
    return out


def _pool_summary(pool) -> dict:
    if pool is None:
        return {"bound": False}
    free = None
    getter = getattr(pool, "get_num_free_blocks", None)
    if callable(getter):
        try:
            free = _int(getter())
        except Exception as exc:  # noqa: BLE001
            snapshot._error(exc, "inspect.pool.get_num_free_blocks")
    blocks = getattr(pool, "blocks", None)
    return {"bound": True,
            "total_blocks": _int(getattr(pool, "num_gpu_blocks", None)),
            "free_blocks": free,
            "block_objects": kvconfig._length(blocks),
            "cached_entries": snapshot._count(
                getattr(pool, "cached_block_hash_to_block", None))}


def inspect_block(engine, group, block_id, ids_cap) -> dict:  # noqa: ARG001
    """One GPU block by id, plus the CPU-pool slot of the same id if a tier runs."""
    scheduler = getattr(engine, "scheduler", None)
    manager = getattr(scheduler, "kv_cache_manager", None)
    pool = getattr(manager, "block_pool", None)
    out: dict = {"verb": "block", "requested": {"block_id": block_id,
                                                "group": group},
                 "ids_cap": ids_cap, "pool": _pool_summary(pool)}
    blocks = getattr(pool, "blocks", None)
    if not isinstance(blocks, list) or block_id is None:
        out["found"] = False
        out["reason"] = ("no kv_cache_manager.block_pool.blocks list on this "
                         "engine, or no block_id given")
        return out
    if block_id < 0 or block_id >= len(blocks):
        out["found"] = False
        out["reason"] = (f"block_id {block_id} is not nameable: the pool holds "
                         f"{len(blocks)} block objects (ids 0..{len(blocks) - 1})")
        return out
    out["found"] = True
    out["block"] = _block_state(pool, blocks[block_id], group)
    out["group_geometry"] = _group_geometry(scheduler, group)
    tier_block = _tier_cpu_block(scheduler, block_id)
    if tier_block is not None:
        out["tier_cpu_slot"] = tier_block
    return out


def _group_geometry(scheduler, group) -> dict | None:
    """The geometry of the requested group, so a block answer is interpretable.

    A block id means little without the group it was asked about: the id space is
    shared by every group (ONE BlockPool for all of them, kv_cache_coordinator.py
    :98), so "block 412 in group 5" is only meaningful next to that group's block
    size and divisibility.
    """
    if group is None:
        return None
    managers = getattr(getattr(getattr(scheduler, "kv_cache_manager", None),
                               "coordinator", None), "single_type_managers", None)
    if not managers or group < 0 or group >= len(managers):
        return {"present": False, "reason": "no such group index"}
    manager_item = managers[group]
    spec = getattr(manager_item, "kv_cache_spec", None)
    block_size = _int(getattr(manager_item, "block_size", None))
    hash_block_size = _int(getattr(scheduler, "hash_block_size", None))
    return {"present": True, "group": group,
            "spec_class": type(spec).__name__ if spec is not None else None,
            "block_size": block_size,
            "hash_block_size": hash_block_size,
            "prefix_cacheable": getattr(spec, "prefix_cacheable", None),
            "layer_count": kvconfig._length(
                getattr(_group_of(scheduler, group), "layer_names", None)),
            "has_positionally_stable_blocks": getattr(
                manager_item, "has_positionally_stable_blocks", None),
            "divisibility": kvconfig.divisibility(block_size, hash_block_size)}


def _group_of(scheduler, group):
    """The KVCacheGroupSpec for a group index (kv_cache_interface.py:1421)."""
    groups = getattr(getattr(scheduler, "kv_cache_config", None),
                     "kv_cache_groups", None)
    if not groups or group < 0 or group >= len(groups):
        return None
    return groups[group]


def _tier_cpu_block(scheduler, block_id) -> dict | None:
    """The tier's CPU-pool slot with the same id, when a tier exists at all.

    A CPU block id and a GPU block id live in DIFFERENT pools and mean nothing to
    each other; the pairing is reported anyway because "did the slot the tier
    picked still hold a hash?" is a real question and the answer is cheap.
    """
    connector = getattr(scheduler, "connector", None)
    manager = getattr(connector, "scheduler_manager", None)
    if manager is None:
        return None
    pool = getattr(manager, "cpu_block_pool", None)
    blocks = getattr(pool, "blocks", None)
    if not isinstance(blocks, list):
        return {"present": False, "reason": "no cpu_block_pool.blocks"}
    if block_id is None or block_id < 0 or block_id >= len(blocks):
        return {"present": True, "found": False,
                "reason": f"id {block_id} is not nameable in the CPU pool "
                          f"({len(blocks)} objects)"}
    return {"present": True, "found": True,
            "pool": _pool_summary(pool),
            "block": _block_state(pool, blocks[block_id], None)}


# --- the request inspector --------------------------------------------------

def _request_counters(request) -> dict:
    return {
        "request_id": _clip(str(getattr(request, "request_id", "")), 64),
        "status_name": getattr(getattr(request, "status", None), "name", None),
        "status_value": _int(getattr(request, "status", None)),
        "num_tokens": _int(getattr(request, "num_tokens", None)),
        "num_prompt_tokens": _int(getattr(request, "num_prompt_tokens", None)),
        "num_output_tokens": _int(getattr(request, "num_output_tokens", None)),
        "num_computed_tokens": _int(getattr(request, "num_computed_tokens", None)),
        "num_output_placeholders": _int(
            getattr(request, "num_output_placeholders", None)),
        "num_preemptions": _int(getattr(request, "num_preemptions", None)),
        "all_token_ids": kvconfig._length(getattr(request, "all_token_ids", None)),
        "block_hashes": kvconfig._length(getattr(request, "block_hashes", None)),
        "skip_reading_prefix_cache": getattr(request, "skip_reading_prefix_cache",
                                             None),
    }


def _tier_alignment(scheduler) -> dict:
    """The tier's own store-alignment choice (manager.py:820-831)."""
    connector = getattr(scheduler, "connector", None)
    manager = getattr(connector, "scheduler_manager", None)
    if manager is None:
        return {"present": False,
                "reason": "no scheduler_manager: the engine is running without "
                          "SimpleCPUOffloadConnector, or prefix caching is off"}
    coordinator = getattr(manager, "cpu_coordinator", None)
    partial = getattr(coordinator, "enable_partial_hash_hits", None)
    block_size = _int(getattr(manager, "block_size", None))
    hash_block_size = _int(getattr(manager, "hash_block_size", None))
    alignment = hash_block_size if partial else block_size
    return {"present": True, "enable_partial_hash_hits": partial,
            "store_alignment": alignment, "block_size": block_size,
            "hash_block_size": hash_block_size}


def _group_request_view(counters, manager_item, group, hash_block_size,
                        alignment_info, ids_cap) -> dict:
    """One group's view of one request: its blocks, and what the tier could park."""
    table = getattr(manager_item, "req_to_blocks", None)
    request_id = counters["request_id"]
    blocks = None
    if table is not None:
        try:
            blocks = table.get(request_id) if hasattr(table, "get") \
                else table[request_id]
        except (KeyError, TypeError) as exc:  # noqa: BLE001 - a weird mapping is data
            snapshot._error(exc, "inspect.req_to_blocks")
            blocks = None
    ids = [getattr(block, "block_id", None) for block in (blocks or [])]
    shown = ids[:ids_cap]
    dropped = max(0, len(ids) - ids_cap)

    block_size = _int(getattr(manager_item, "block_size", None))
    hashes = _int(counters["block_hashes"])
    unit = _int(hash_block_size)
    scale = None
    hashes_at_group_size = None
    check = kvconfig.divisibility(block_size, hash_block_size)
    if check["divisible"] and block_size and unit:
        scale = block_size // unit
        if hashes is not None:
            hashes_at_group_size = hashes // scale

    confirmed = None
    computed = _int(counters["num_computed_tokens"])
    placeholders = _int(counters["num_output_placeholders"])
    if computed is not None and placeholders is not None:
        confirmed = computed - placeholders
    alignment = _int(alignment_info.get("store_alignment"))
    aligned = None
    if confirmed is not None and alignment:
        aligned = confirmed // alignment * alignment
    ready = None
    binding = None
    if block_size and aligned is not None:
        aligned_by_block = aligned // block_size
        candidates = {"block_count": len(ids),
                      "aligned_blocks": aligned_by_block}
        if hashes_at_group_size is not None:
            candidates["hashes_at_group_size"] = hashes_at_group_size
        elif check["would_fail"]:
            candidates["hashes_at_group_size"] = 0
        ready = min(candidates.values())
        binding = min(candidates, key=lambda key: candidates[key])
    return {
        "group": group,
        "manager_class": type(manager_item).__name__,
        "block_size": block_size,
        "spec_class": type(getattr(manager_item, "kv_cache_spec", None)).__name__,
        "prefix_cacheable": getattr(
            getattr(manager_item, "kv_cache_spec", None), "prefix_cacheable", None),
        "has_positionally_stable_blocks": getattr(
            manager_item, "has_positionally_stable_blocks", None),
        "block_count": len(ids),
        "block_ids": [_int(value) for value in shown],
        "block_ids_dropped": dropped,
        "divisibility": check,
        "hash_scale_factor": scale,
        "hashes_at_group_size": hashes_at_group_size,
        "coverable_blocks": (0 if check["would_fail"] else hashes_at_group_size),
        "tier_view": {"confirmed_tokens": confirmed,
                      "aligned_tokens": aligned,
                      "alignment": alignment,
                      "ready_blocks": ready,
                      "binding_term": binding,
                      "meaning": ("ready_blocks is the fork's own "
                                  "ready = min(len(group_gpu_ids), aligned_tokens "
                                  "// group_size) at manager.py:826, additionally "
                                  "capped by the hashes it can view at this size")},
    }


def inspect_request(engine, request_id, ids_cap) -> dict:
    """Everything one request holds, per group, plus what the tier could park."""
    scheduler = getattr(engine, "scheduler", None)
    requests = getattr(scheduler, "requests", None)
    out: dict = {"verb": "request", "requested": {"request_id": request_id},
                 "ids_cap": ids_cap}
    if not isinstance(requests, dict):
        out["found"] = False
        out["reason"] = "scheduler.requests is not a dict on this engine"
        return out
    out["scheduler"] = {"requests": len(requests),
                        "running": snapshot._count(getattr(scheduler, "running",
                                                           None)),
                        "waiting": snapshot._count(getattr(scheduler, "waiting",
                                                           None)),
                        "current_step": kvconfig._int(
                            getattr(scheduler, "current_step", None))}
    if request_id is None:
        out["found"] = False
        ids = list(requests)[:MAX_REQUEST_IDS]
        out["reason"] = ("no request_id given: pass {\"request_id\": \"...\"}, or "
                         "see known_request_ids")
        out["known_request_ids"] = [_clip(str(value), 64) for value in ids]
        out["known_request_ids_dropped"] = max(0, len(requests) - len(ids))
        return out
    request = requests.get(request_id)
    if request is None:
        ids = list(requests)[:MAX_REQUEST_IDS]
        out["found"] = False
        out["reason"] = (f"request_id {request_id!r} is not in scheduler.requests "
                         "(it is finished, or the id is wrong)")
        out["known_request_ids"] = [_clip(str(value), 64) for value in ids]
        out["known_request_ids_dropped"] = max(0, len(requests) - len(ids))
        return out
    out["found"] = True
    counters = _request_counters(request)
    out["request"] = counters
    managers = getattr(getattr(getattr(scheduler, "kv_cache_manager", None),
                               "coordinator", None), "single_type_managers", None)
    hash_block_size = _int(getattr(scheduler, "hash_block_size", None))
    alignment_info = _tier_alignment(scheduler)
    out["tier_alignment"] = alignment_info
    groups: list = []
    truncated = 0
    if managers:
        for group, manager_item in enumerate(list(managers)[:MAX_GROUP_ENTRIES]):
            groups.append(_group_request_view(counters, manager_item,
                                              group, hash_block_size,
                                              alignment_info, ids_cap))
        truncated = max(0, len(managers) - MAX_GROUP_ENTRIES)
    else:
        out["groups_reason"] = "no coordinator.single_type_managers on this engine"
    out["groups"] = groups
    out["groups_dropped"] = truncated
    return out


# --- the handler ------------------------------------------------------------

def usage_note() -> dict:
    """What the verb accepts, so a wrong body teaches instead of failing dark."""
    return {
        "verbs": {
            "block": '{"block_id": <int>, "group": <int?>, "ids_cap": <int?>}',
            "request": '{"request_id": "<str>"} or {"request_id": ""} for the id '
                       "list, or {}",
        },
        "ids_cap_default": DEFAULT_IDS,
        "ids_cap_max": MAX_IDS_LIMIT,
        "note": ("the first call that names a block_id inspects a block; naming a "
                 "request_id inspects a request; naming both inspects the block "
                 "and says so"),
    }


def inspect_for_engine(engine, args=None) -> dict:
    """THE engine-side handler body for the inspectors. Never raises an Exception.

    One method, two shapes: `{"block_id": ...}` inspects a GPU block,
    `{"request_id": ...}` inspects a request. Anything else answers with
    `usage_note()` rather than guessing which of the two the caller meant.
    """
    snapshot._state["calls"] += 1
    started = time.perf_counter()
    arguments = args if isinstance(args, dict) else {}
    try:
        ids_cap = _ids_cap(arguments)
        unknown = sorted(key for key in arguments
                         if key not in ("block_id", "group", "request_id",
                                        "ids_cap"))
        payload: dict
        block_id = _int(arguments.get("block_id"))
        request_id = arguments.get("request_id")
        if block_id is not None:
            payload = inspect_block(engine, _int(arguments.get("group")),
                                    block_id, ids_cap)
            if isinstance(request_id, str):
                payload["note"] = ("both block_id and request_id were given: the "
                                   "block was inspected, the request was not")
        elif isinstance(request_id, str) or request_id is None:
            payload = inspect_request(engine, request_id, ids_cap)
        else:
            payload = {"verb": "request", "found": False,
                       "reason": "request_id must be a string"}
        payload["version"] = PROTOCOL_VERSION
        payload["epoch"] = str(snapshot.epoch())
        payload["ts"] = round(time.time(), 6)
        payload["guard_mode"] = guard_mode.is_armed()
        payload["trace_armed"] = trace_ring.is_armed()
        if unknown:
            payload["unknown_args"] = unknown
        if not arguments:
            payload["usage"] = usage_note()
        payload["build_us"] = int((time.perf_counter() - started) * 1e6)
        payload["self"] = snapshot.error_counters()
        return snapshot.json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        snapshot._error(exc, "inspect")
        return {"version": PROTOCOL_VERSION, "verb": "inspect",
                "epoch": str(snapshot.epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "guard_mode": guard_mode.is_armed(),
                "counters": snapshot.error_counters(),
                "usage": usage_note()}
