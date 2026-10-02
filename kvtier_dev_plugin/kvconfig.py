"""Capability 1: the KV-cache GEOMETRY of the live engine, read while it serves.

WHY THIS EXISTS
---------------
The tier we are building parks per-group KV blocks in host RAM and restores them
on a prefix hit. The fork's own attempt dies on this model at the first request
with `assert block_size % hash_block_size == 0`
(`vllm/v1/core/kv_cache_utils.py:2888`, reached from
`simple_kv_offload/manager.py:842` via `resolve_block_hashes`, and asserted a
second time inside `BlockHashListWithBlockSize.__init__` at
`kv_cache_utils.py:2823`). The model is hybrid/mamba: six groups, five with a
3200-token block size, one with 8, one group a non-prefix-cacheable
`CircularBufferSpec` ring. That geometry is a *fact about the running engine*
that was only discoverable by crashing it. This module makes it a read.

WHAT IT READS (fork at HEAD 10b8e07; every name verified, none invented)
------------------------------------------------------------------------
`scheduler.kv_cache_config`               scheduler.py:105   (a KVCacheConfig)
  `.num_blocks`                           kv_cache_interface.py:1419
  `.kv_cache_groups[g]`                   kv_cache_interface.py:1421
  `.kv_cache_layout`                      kv_cache_interface.py:1435
  `.prefix_cache_retention_interval`      kv_cache_interface.py:1432
`KVCacheGroupSpec`                        kv_cache_interface.py:1392-1409
  `.layer_names` / `.kv_cache_spec` / `.is_eagle_group` / `.host_resident`
  / `.enable_kv_transfer` / `.role`
`KVCacheSpec`                             kv_cache_interface.py:156
  `.block_size` :161, `.prefix_cacheable` :164, `.tokens_per_state` :172,
  `.state_content_size_bytes` :181, `.page_size_bytes` :186, `.num_states` :195,
  `.block_table_token_alignment` :198
`AttentionSpec`                           kv_cache_interface.py:479
  `.num_kv_heads` / `.head_size` / `.head_size_v` / `.dtype` /
  `.page_size_padded` / `.num_head_slots` / `.state_content_bytes` /
  `.tokens_per_state`
`CircularBufferSpec`                      kv_cache_interface.py:865
  `prefix_cacheable` -> False :893, `uses_slot_mapping` -> False :897,
  `block_table_token_alignment` -> None :877, one block per request :885
`MambaSpec`                               kv_cache_interface.py:993
  `.shapes` / `.dtypes` / `.mamba_type` / `.mamba_cache_mode` /
  `.num_speculative_blocks` / `.num_prefill_checkpoint_blocks` /
  `.prefill_checkpoint_alignment` / `.num_heads` / `.tokens_per_state` /
  `.tp_replicated`
`scheduler.block_size` / `.hash_block_size`   scheduler.py:220 / :330
`scheduler.has_mamba_layers` / `.needs_kv_cache_zeroing`  scheduler.py:373 / :374
  (the scheduler's own PRECOMPUTED copies - the KVCacheConfig properties at
  kv_cache_interface.py:1513 / :1535 walk every layer of every group, and this
  runs on the thread that serves traffic)
`cache_config.mamba_cache_mode`           config/cache.py:178
`cache_config.prefix_match_unit`          config/cache.py:91 ("This equals to the
  `hash_block_size` used throughout the KV cache code")
`cache_config.enable_prefix_caching`, `.num_gpu_blocks`  config/cache.py:245
`kv_cache_manager.coordinator.single_type_managers[g]`
  `.block_size` single_type_kv_cache_manager.py:105, `.kv_cache_group_id` :139,
  `.has_positionally_stable_blocks` :62 (the Mamba "align" override at :1494 is
  exactly the case where a positional park is NOT safe), `.req_to_blocks` :133
`block_pool`  `.num_gpu_blocks` :173, `.blocks` :178, `.free_block_queue` :184,
  `.cached_block_hash_to_block` :187 (its `__len__` at :136 is maintained),
  `get_num_free_blocks()` :861
tier manager (only with a connector) `block_size`, `hash_block_size`,
  `group_block_sizes`, `num_cpu_blocks`, `fa_block_size`, `fa_gidx`

THE DIVISIBILITY FIELD IS THE POINT OF THE MODULE
-------------------------------------------------
Every group carries `divisibility`, computed with the SAME operands as the
fork's assert (the group's block size and `hash_block_size`), and the payload
carries a summary naming the groups that would fail. A geometry that breaks the
invariant is therefore readable from `GET /plugins/kvtier/kvconfig` before
anything is asked to run. The check is arithmetic on two ints we read; it cannot
make the engine run the crash.

HOUSE RULES: stdlib only, no torch/numpy/vllm, no file I/O, no unbounded walk.
Every list is capped and says how much it dropped. Contained by
`kvconfig_for_engine()`, which catches `Exception` and never `BaseException`.
"""

from __future__ import annotations

import time

from . import guard_mode, snapshot, trace_ring

PROTOCOL_VERSION = 1

# --- bounds. Every one exists because a bounded answer is the only kind this
# --- module is allowed to give on the thread that serves traffic.
MAX_GROUPS = 64
MAX_LAYER_NAMES = 32
MAX_GROUP_SIZES = 64
MAX_SPEC_ITEMS = 16
MAX_STRING_CHARS = 96

# The fork's own invariant, quoted so a reader of the payload can grep it.
DIVISIBILITY_RULE = ("assert block_size % hash_block_size == 0 "
                     "(vllm/v1/core/kv_cache_utils.py:2888, and again in "
                     "BlockHashListWithBlockSize.__init__ at :2823)")

# The spec attributes worth publishing. Read by name and reported only when the
# object actually carries them, so a Mamba group does not grow a wall of nulls
# and the ring group reports what makes it a ring.
SPEC_FIELDS = (
    # KVCacheSpec / AttentionSpec state-ish attributes.
    "tokens_per_state", "num_states", "state_content_size_bytes", "num_heads",
    "num_head_slots", "head_size", "head_size_v", "num_kv_heads", "dtype",
    "sliding_window", "attention_chunk_size", "page_size_padded",
    "page_size_bytes", "uses_slot_mapping", "block_table_token_alignment",
    "kv_quant_mode", "non_causal", "prefix_cacheable",
    # MambaSpec-only (kv_cache_interface.py:993-1004).
    "shapes", "dtypes", "mamba_type", "mamba_cache_mode",
    "num_speculative_blocks", "num_prefill_checkpoint_blocks",
    "prefill_checkpoint_alignment", "tp_replicated",
)


# --- small read helpers -----------------------------------------------------

def _int(value) -> int | None:
    """A real int, or None. `bool` is NOT an int here (a True block size is a lie)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _length(container) -> int | None:
    if container is None:
        return None
    try:
        return len(container)
    except TypeError:
        return None


def _attribute(owner, name):
    """getattr that never raises: a property that blows up is a missing field."""
    try:
        return getattr(owner, name, None)
    except Exception as exc:  # noqa: BLE001 - a lying property is data, not a crash
        snapshot._error(exc, f"kvconfig.{name}")
        return None


def _scalar(value, max_chars: int = MAX_STRING_CHARS):
    """A JSON/msgpack-safe rendering of one attribute value. Never raises.

    `tokens_per_state` is `int | Fraction` (kv_cache_interface.py:492) and a
    `Fraction` is not encodable, so it is rendered as `"num/den"` by duck-typing
    `numerator`/`denominator` (no `numbers` import, no torch). `dtype` is a
    `torch.dtype`, whose `str()` we may take because the object already exists
    and str() on it touches no CUDA. Bytes (a block hash) become a bounded hex
    prefix. Anything else is a clipped `repr`, so an unknown field still arrives.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "..."
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        return f"{numerator}/{denominator}"
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value)[:16].hex()
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)[:MAX_SPEC_ITEMS]
        out = [_scalar(item, 24) for item in items]
        if len(value) > MAX_SPEC_ITEMS:
            out.append(f"... {len(value) - MAX_SPEC_ITEMS} more")
        return out
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - an attribute that cannot be named is data
        return f"<unprintable {type(value).__name__}>"
    if text.startswith("<") and " at 0x" in text:
        return f"<{type(value).__name__}>"  # a default repr carries an address
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def _sample(values, cap: int) -> tuple[list, int]:
    """(capped list, dropped count) of a sized container. Never walks the tail."""
    total = _length(values)
    if total is None:
        return [], 0
    try:
        items = list(values)[:cap]
    except TypeError:
        return [], total
    return [_scalar(item) for item in items], max(0, total - cap)


# --- the divisibility check -------------------------------------------------

def divisibility(block_size, hash_block_size) -> dict:
    """The fork's assert, computed as data instead of crashing on it.

    `would_fail` is True only when both operands are real ints (the unit
    positive) and the remainder is non-zero. Anything less provable is `None`
    with a reason, because "unknown" and "safe" must never look the same here.
    """
    size = _int(block_size)
    unit = _int(hash_block_size)
    out: dict = {
        "group_block_size": size,
        "hash_block_size": unit,
        "divisible": None,
        "remainder": None,
        "scale_factor": None,
        "would_fail": None,
        "rule": DIVISIBILITY_RULE,
    }
    if size is None or unit is None:
        out["reason"] = ("an operand is missing or is not an int, so the fork's "
                         "assert cannot be evaluated from here")
        return out
    if unit <= 0:
        out["reason"] = "hash_block_size is not positive (the fork asserts first)"
        return out
    remainder = size % unit
    out["remainder"] = remainder
    out["divisible"] = remainder == 0
    out["would_fail"] = remainder != 0
    if remainder == 0:
        out["scale_factor"] = size // unit
    return out


# --- per-group geometry -----------------------------------------------------

def spec_view(spec, available: set) -> dict:
    """The spec of one group: class name plus the state-ish attributes it has.

    `available` is the ONE `dir()` we take per spec; presence is tested against
    it so a dataclass field with no class-level default is still seen, and a
    field the fork renamed is simply absent rather than a null.
    """
    if spec is None:
        return {"class": None}
    out: dict = {"class": type(spec).__name__}
    for name in SPEC_FIELDS:
        if name in available:
            out[name] = _scalar(_attribute(spec, name))
    return out


def is_circular_buffer(spec) -> bool | None:
    """Is this the ring group? Decided by CLASS NAME through the MRO.

    `isinstance` would need the import we are forbidden to do, and a fork
    subclass of `CircularBufferSpec` is still a ring: one block per request
    (`max_num_blocks_per_req` returns 1, kv_cache_interface.py:885), never
    prefix-cacheable (:893), no slot mapping (:897).
    """
    if spec is None:
        return None
    for cls in type(spec).__mro__:
        if getattr(cls, "__name__", "") == "CircularBufferSpec":
            return True
    return False


def group_view(index, group, hash_block_size, managers) -> dict:
    """One KV cache group, as an offload tier needs to see it."""
    spec = _attribute(group, "kv_cache_spec")
    try:
        available = set(dir(spec)) if spec is not None else set()
    except Exception as exc:  # noqa: BLE001 - dir() on a hostile object
        snapshot._error(exc, "kvconfig.spec dir()")
        available = set()
    block_size = _int(_attribute(spec, "block_size"))
    layer_names = _attribute(group, "layer_names")
    names, dropped = _sample(layer_names, MAX_LAYER_NAMES)
    manager = managers[index] if index < len(managers) else None
    manager_block_size = _int(_attribute(manager, "block_size"))
    out = {
        "index": index,
        "block_size": block_size,
        "spec_class": type(spec).__name__ if spec is not None else None,
        "page_size_bytes": _scalar(_attribute(spec, "page_size_bytes")),
        "prefix_cacheable": _attribute(spec, "prefix_cacheable"),
        "is_circular_buffer": is_circular_buffer(spec),
        "layer_count": _length(layer_names),
        "layer_names": names,
        "layer_names_dropped": dropped,
        "is_eagle_group": _attribute(group, "is_eagle_group"),
        "host_resident": _attribute(group, "host_resident"),
        "enable_kv_transfer": _attribute(group, "enable_kv_transfer"),
        "role": _scalar(_attribute(group, "role"), 32),
        "spec": spec_view(spec, available),
        "divisibility": divisibility(block_size, hash_block_size),
        "manager": {
            "present": manager is not None,
            "class": type(manager).__name__ if manager is not None else None,
            "block_size": manager_block_size,
            "kv_cache_group_id": _int(_attribute(manager, "kv_cache_group_id")),
            "has_positionally_stable_blocks": _attribute(
                manager, "has_positionally_stable_blocks"),
            "requests_with_blocks": snapshot._count(
                _attribute(manager, "req_to_blocks")),
        },
    }
    # A disagreement between the group's own spec and the manager that allocates
    # for it means a park under the wrong stride. Naming it here is cheaper than
    # finding it in a crash.
    if (block_size is not None and manager_block_size is not None
            and block_size != manager_block_size):
        out["manager_block_size_mismatch"] = {
            "spec_block_size": block_size,
            "manager_block_size": manager_block_size,
        }
    return out


# --- the surrounding state --------------------------------------------------

def block_pool_view(engine) -> dict:
    """The ONE GPU BlockPool the coordinator shares across all groups."""
    manager = _attribute(_attribute(engine, "scheduler"), "kv_cache_manager")
    pool = _attribute(manager, "block_pool")
    if pool is None:
        return {"bound": False}
    free = None
    getter = _attribute(pool, "get_num_free_blocks")
    if callable(getter):
        try:
            free = _int(getter())
        except Exception as exc:  # noqa: BLE001 - a pool that lies is data
            snapshot._error(exc, "kvconfig.pool.get_num_free_blocks")
    queue = _attribute(pool, "free_block_queue")
    return {
        "bound": True,
        "total_blocks": _int(_attribute(pool, "num_gpu_blocks")),
        "free_blocks": free,
        "block_objects": _length(_attribute(pool, "blocks")),
        "cached_entries": snapshot._count(
            _attribute(pool, "cached_block_hash_to_block")),
        "free_queue_nodes": _int(_attribute(queue, "num_free_blocks")),
    }


def managers_view(scheduler) -> tuple[dict, list]:
    """`coordinator.single_type_managers`, index-aligned with the groups.

    Managers are keyed by their own `kv_cache_group_id` when they carry one
    (single_type_kv_cache_manager.py:139), falling back to position, because the
    tier's `group_block_sizes` and our group list are positional and a silent
    misalignment would make every divisibility number below a lie.
    """
    manager = _attribute(scheduler, "kv_cache_manager")
    coordinator = _attribute(manager, "coordinator")
    managers = _attribute(coordinator, "single_type_managers")
    report = {"present": managers is not None, "count": 0, "truncated": False,
              "by_group_id": {}, "scheduler_block_size": None,
              "num_reprefillable_tokens": None, "eagle_group_ids": None,
              "cache_hit_alignment_tokens": None, "reason": None}
    if managers is None:
        report["reason"] = ("no kv_cache_manager.coordinator.single_type_managers "
                            "on this engine")
        return report, []
    try:
        items = list(managers)[:MAX_GROUPS]
        total = len(managers)
    except TypeError:
        report["reason"] = "single_type_managers is not sized/iterable"
        return report, []
    ordered = list(items)
    report.update({
        "count": total,
        "truncated": total > MAX_GROUPS,
        "scheduler_block_size": _int(_attribute(coordinator, "scheduler_block_size")),
        "num_reprefillable_tokens": _int(
            _attribute(coordinator, "num_reprefillable_tokens")),
        "cache_hit_alignment_tokens": _int(
            _attribute(coordinator, "cache_hit_alignment_tokens")),
        "eagle_group_ids": sorted(
            value for value in (_attribute(coordinator, "eagle_group_ids") or set())
            if isinstance(value, int) and not isinstance(value, bool)),
    })
    retained = _attribute(manager, "retained_hit_group_ids")
    if retained is not None:
        report["retained_hit_group_ids"] = _scalar(retained, 24)
    by_id = {}
    for position, item in enumerate(ordered):
        group_id = _int(_attribute(item, "kv_cache_group_id"))
        by_id[str(group_id if group_id is not None else position)] = {
            "position": position,
            "class": type(item).__name__,
            "block_size": _int(_attribute(item, "block_size")),
            "kv_cache_group_id": group_id,
            "has_positionally_stable_blocks": _attribute(
                item, "has_positionally_stable_blocks"),
        }
    report["by_group_id"] = by_id
    return report, ordered


def cache_config_view(scheduler) -> dict:
    """The CacheConfig knobs that interact with the geometry."""
    cache_config = _attribute(scheduler, "cache_config")
    if cache_config is None:
        return {"present": False,
                "reason": "scheduler exposes no cache_config"}
    return {
        "present": True,
        "mamba_cache_mode": _scalar(_attribute(cache_config, "mamba_cache_mode"), 16),
        "prefix_match_unit": _int(_attribute(cache_config, "prefix_match_unit")),
        "block_size": _int(_attribute(cache_config, "block_size")),
        "num_gpu_blocks": _int(_attribute(cache_config, "num_gpu_blocks")),
        "num_cpu_blocks": _int(_attribute(cache_config, "num_cpu_blocks")),
        "enable_prefix_caching": _attribute(cache_config, "enable_prefix_caching"),
        "enable_mamba_fine_grained_prefix_cache": _attribute(
            cache_config, "enable_mamba_fine_grained_prefix_cache"),
        "kv_cache_size_tokens": _int(_attribute(cache_config, "kv_cache_size_tokens")),
        # The knob the fork says must divide EVERY group's block size
        # (config/cache.py:91-99): same rule as hash_block_size, checked once.
        "prefix_match_unit_divisibility": divisibility(
            _int(_attribute(cache_config, "block_size")),
            _int(_attribute(cache_config, "prefix_match_unit"))),
    }


def tier_crosscheck(scheduler, groups) -> dict:
    """The tier manager's own view of the same geometry, and where they differ.

    With `connector: None` (the state this plugin is developed in) there is no
    manager and this says so in `reason`. When there is one, a
    `group_block_sizes` entry that disagrees with the group's spec, or a
    `hash_block_size` that disagrees with the scheduler's, is the precondition of
    the crash this module exists to make visible - so both are recomputed against
    the TIER's operands, not only the engine's.
    """
    connector = _attribute(scheduler, "connector")
    manager = _attribute(connector, "scheduler_manager")
    if manager is None:
        return {"present": False,
                "reason": "no scheduler_manager: the engine is running without "
                          "SimpleCPUOffloadConnector, or prefix caching is off"}
    sizes = _attribute(manager, "group_block_sizes")
    size_list = list(sizes) if isinstance(sizes, (list, tuple)) else []
    tier_sizes = [_int(value) for value in size_list][:MAX_GROUP_SIZES]
    tier_hash = _int(_attribute(manager, "hash_block_size"))
    scheduler_hash = _int(_attribute(scheduler, "hash_block_size"))
    scheduler_size = _int(_attribute(scheduler, "block_size"))
    tier_size = _int(_attribute(manager, "block_size"))
    per_group = []
    for index, entry in enumerate(groups):
        geometry_size = entry.get("block_size")
        tier_size_g = tier_sizes[index] if index < len(tier_sizes) else None
        per_group.append({
            "index": index,
            "geometry_block_size": geometry_size,
            "tier_block_size": tier_size_g,
            "agree": (geometry_size is not None and tier_size_g is not None
                      and geometry_size == tier_size_g),
            "tier_divisibility": divisibility(tier_size_g, tier_hash),
        })
    return {
        "present": True,
        "class": type(manager).__name__,
        "group_block_sizes": tier_sizes,
        "group_block_sizes_dropped": max(0, len(size_list) - MAX_GROUP_SIZES),
        "block_size": tier_size,
        "hash_block_size": tier_hash,
        "fa_block_size": _int(_attribute(manager, "fa_block_size")),
        "fa_group_index": _int(_attribute(manager, "fa_gidx")),
        "num_cpu_blocks": _int(_attribute(manager, "num_cpu_blocks")),
        "disagreements": {
            "hash_block_size": (None if tier_hash is None or scheduler_hash is None
                                else tier_hash != scheduler_hash),
            "block_size": (None if tier_size is None or scheduler_size is None
                           else tier_size != scheduler_size),
            "group_block_sizes": [entry["index"] for entry in per_group
                                  if entry["agree"] is False],
        },
        "per_group": per_group,
    }


def scheduler_view(scheduler, manager_report) -> dict:
    connector = _attribute(scheduler, "connector")
    return {
        "present": scheduler is not None,
        "class": type(scheduler).__name__ if scheduler is not None else None,
        "block_size": _int(_attribute(scheduler, "block_size")),
        "hash_block_size": _int(_attribute(scheduler, "hash_block_size")),
        "connector": (type(connector).__name__ if connector is not None else None),
        "requests": snapshot._count(_attribute(scheduler, "requests")),
        "running": snapshot._count(_attribute(scheduler, "running")),
        "waiting": snapshot._count(_attribute(scheduler, "waiting")),
        "current_step": _int(_attribute(scheduler, "current_step")),
        "has_mamba_layers": _attribute(scheduler, "has_mamba_layers"),
        "needs_kv_cache_zeroing": _attribute(scheduler, "needs_kv_cache_zeroing"),
        "managers": manager_report,
    }


def build_geometry(engine, groups, summary) -> dict:
    """The geometry block: totals, groups, the pool, and the divisibility summary."""
    kv_cache_config = _attribute(_attribute(engine, "scheduler"), "kv_cache_config")
    return {
        "num_blocks": _int(_attribute(kv_cache_config, "num_blocks")),
        "num_groups": _length(_attribute(kv_cache_config, "kv_cache_groups")),
        "groups": groups,
        "kv_cache_layout": _scalar(_attribute(kv_cache_config, "kv_cache_layout"), 32),
        "prefix_cache_retention_interval": _int(
            _attribute(kv_cache_config, "prefix_cache_retention_interval")),
        "transfer_group_ids": [entry["index"] for entry in groups
                               if entry.get("enable_kv_transfer") is True],
        "prefix_cacheable_group_ids": [entry["index"] for entry in groups
                                       if entry.get("prefix_cacheable") is True
                                       and entry.get("enable_kv_transfer") is True],
        "scratch_group_ids": [entry["index"] for entry in groups
                              if entry.get("prefix_cacheable") is False],
        "block_pool": block_pool_view(engine),
        "divisibility_summary": summary,
    }


def divisibility_summary(groups, hash_block_size) -> dict:
    """The answer to "will the tier's first store assert?", with the group indices."""
    failing = [entry["index"] for entry in groups
               if entry["divisibility"].get("would_fail") is True]
    unknown = [entry["index"] for entry in groups
               if entry["divisibility"].get("would_fail") is None]
    return {
        "hash_block_size": hash_block_size,
        "groups_checked": len(groups),
        "groups_failing": failing,
        "groups_unknown": unknown,
        "all_divisible": bool(groups) and not failing and not unknown,
        "any_fail": bool(failing),
        "rule": DIVISIBILITY_RULE,
        "meaning": ("a group listed in groups_failing makes "
                    "resolve_block_hashes(request.block_hashes, hash_block_size, "
                    "group_block_size) assert, which is what the tier hits on its "
                    "first store for that group"),
    }


def kvconfig_section(engine) -> dict:
    """The verb's payload, uncontained. `kvconfig_for_engine` contains it."""
    scheduler = _attribute(engine, "scheduler")
    hash_block_size = _int(_attribute(scheduler, "hash_block_size"))
    manager_report, managers = managers_view(scheduler)
    groups_raw = _attribute(_attribute(scheduler, "kv_cache_config"),
                            "kv_cache_groups")
    groups = []
    total = _length(groups_raw) or 0
    for index, group in enumerate(list(groups_raw or [])[:MAX_GROUPS]):
        groups.append(group_view(index, group, hash_block_size, managers))
    summary = divisibility_summary(groups, hash_block_size)
    summary["groups_reported"] = len(groups)
    summary["groups_dropped"] = max(0, total - len(groups))
    return {
        "version": PROTOCOL_VERSION,
        "verb": "kvconfig",
        "scheduler": scheduler_view(scheduler, manager_report),
        "config": cache_config_view(scheduler),
        "geometry": build_geometry(engine, groups, summary),
        "tier_crosscheck": tier_crosscheck(scheduler, groups),
    }


def kvconfig_for_engine(engine, args=None) -> dict:
    """THE engine-side handler body for `kvconfig`. Never raises an Exception.

    `args` is accepted for symmetry with the other argument-taking verbs and is
    deliberately unused: the geometry is one bounded pass over at most
    `MAX_GROUPS` groups, so there is no cheaper or bigger shape a caller could
    ask for that would justify a second payload shape to keep honest.
    """
    snapshot._state["calls"] += 1
    started = time.perf_counter()
    try:
        payload = kvconfig_section(engine)
        payload["epoch"] = str(snapshot.epoch())
        payload["ts"] = round(time.time(), 6)
        payload["guard_mode"] = guard_mode.is_armed()
        payload["trace_armed"] = trace_ring.is_armed()
        payload["args_ignored"] = sorted(args) if isinstance(args, dict) else []
        payload["build_us"] = int((time.perf_counter() - started) * 1e6)
        payload["self"] = snapshot.error_counters()
        return snapshot.json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        snapshot._error(exc, "kvconfig")
        return {"version": PROTOCOL_VERSION, "verb": "kvconfig",
                "epoch": str(snapshot.epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "guard_mode": guard_mode.is_armed(),
                "counters": snapshot.error_counters()}
