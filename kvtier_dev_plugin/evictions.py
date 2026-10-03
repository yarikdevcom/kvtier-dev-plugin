"""Honest LRU-eviction visibility for the host-RAM KV tier. Stdlib only.

THE HOLE THIS CLOSES
--------------------
At 99.9 % arena occupancy with the engine still parking (``store_event_counter``
climbing 863 -> 1376) the tier was silently evicting parked CPU spans to make
room, and NOTHING in the snapshot showed it: ``dropped_cpu_full`` stayed 0 and
every ``dropped*`` boundary counter stayed flat. Those counters count STORES THE
TIER DECLINED AT THE GATE; none of them counts a PARKED ENTRY THE LRU THREW AWAY
AFTER IT WAS ALREADY PARKED. "Is the tier full, or is retention shallow?" was
therefore an hour of argument, because the eviction signal was invisible.

THE ONE TRUTHFUL CHOKE POINT
----------------------------
``vllm/v1/core/block_pool.py:726`` ``BlockPool._maybe_evict_cached_block(block)``
returns ``True`` IF AND ONLY IF a block that was still indexed in
``cached_block_hash_to_block`` had its hash metadata reset (i.e. a parked entry
was actually discarded). It is the single place a parked entry dies, and it is
reached from exactly the two paths that discard one:

  * ``get_new_blocks`` (``block_pool.py:664``, loop at ``:686``) - a block handed
    out of the free queue to receive a NEW store, which is precisely "make room
    under pressure": the 99.9 %-occupancy regime.
  * ``evict_blocks`` (``block_pool.py:802``, loop at ``:819``) - an explicit
    discard (the capacity clamp's withdrawal, ``cpu_capacity._apply_cap``).

``move_block_hashes`` (``block_pool.py:650``) calls the lower
``_remove_cached_block_hashes`` DIRECTLY and must NOT be counted: it re-points a
live entry to a private copy, the entry stays cached, nothing is evicted. That is
why this module wraps ``_maybe_evict_cached_block`` (the boolean verdict) rather
than ``_remove_cached_block_hashes`` (which a re-point also calls).

WHY A WRAP, SAID LOUDLY
-----------------------
This is NOT a pure read. The pool keeps no monotonic eviction counter: the only
engine-side hook is ``KVCacheMetricsCollector.on_block_evicted``
(``kv_cache_metrics.py:71``), which is (a) SAMPLED at 1 % (``sample_rate`` at
``:49``, gated by ``should_sample_block`` at ``:59``) and (b) drained
(``drain_events`` at ``:93``), so it is a histogram of a 1 % subset, not a total.
A count of evictions therefore exists only if WE count it, at the one place the
verdict is produced. The alternative - reporting the sampled histogram as if it
were a total, or deriving ``stores - parked`` as an "eviction" number - is
exactly the drift-prone figure this operator asked to be spared. So: one
instance-level wrapper, it increments two monotonic integers and never changes
what the engine does.

IT FAILS CLOSED, NEVER GUESSSES
-------------------------------
* The wrapper calls the engine's own ``original(block)`` and returns its verdict
  UNCHANGED. If the engine raises, that raise propagates: a discard is the
  engine's own business, and swallowing it would corrupt the allocator.
* Everything this module ADDS (reading the block's group id, bumping counters)
  is inside ``try/except Exception`` and only ever moves a counter. A failed
  attribution increments ``attribution_failures`` and still lets the eviction
  through; it never raises into ``get_new_blocks``.
* If the live CPU pool is not wrapped (module renamed the method, a foreign owner
  took it first, a manager built before adoption), the section reports
  ``available: False`` with a ``reason`` and null counts - NOT a zero. A zero
  here would read as "the LRU has evicted nothing", the exact lie that made the
  steady state invisible.

INSTANCE-LEVEL, NEVER CLASS-LEVEL
---------------------------------
The SAME ``BlockPool`` class backs the GPU pool, and the GPU pool must never
acquire a wrapper from a KV-tier plugin. So the wrapper is set on the ONE
instance ``manager.cpu_block_pool`` (the ``cpu_capacity._arm_refusal_counter``
discipline, ``cpu_capacity.py:_arm_refusal_counter``), marker-tagged, idempotent,
with a stored ``ORIGINAL_ATTR`` so adoption can never nest.

PER GROUP, AND WHY THE FINE TICKS APPEAR HERE
---------------------------------------------
A parked span is one block whose PRIMARY hash carries one group id
(``kv_cache_utils.py:74-93``: the group id is the last 4 big-endian bytes).
Evicting a fine-graining group's coarse block ALSO drops that block's secondary
(fine-tick) keys - they live in ``cached_block_hashes_by_block[block_id]`` and
``_remove_cached_block_hashes`` (``block_pool.py:588-606``) removes the primary
AND every extra hash. So two views are published, both monotonic:

  * ``evicted_spans_per_group`` - one per discarded block, under its PRIMARY
    group: this mirrors ``snapshot``'s ``parked_blocks_per_group`` (also keyed by
    the primary group) so spans in == spans evicted is a real balance.
  * ``evicted_keys_per_group`` - one per discarded INDEX KEY, by the group baked
    into that key: this is the view that shows whether group 5's fine ticks are
    evicted alongside their coarse block.

Both are attributed from a cheap pre-read of attributes the block already
carries (one attribute read plus one dict ``.get``), never a scan.
"""

from __future__ import annotations

import logging

from . import snapshot

logger = logging.getLogger("kvtier_dev_plugin.evictions")

# Resolved only against the live pool instance; never an import of vllm.
TARGET_MODULE = "vllm.v1.core.block_pool"
TARGET_CLASS = "BlockPool"
TARGET_METHOD = "_maybe_evict_cached_block"

# Ownership markers, the store_hook / cpu_capacity idiom: a name we own is
# re-adoptable, a name somebody else owns is refused with a reason.
OWNER_ATTR = "__kvtier_dev_evict_owner__"
OWNER_TAG = "kvtier-dev-plugin/evictions-1"
ORIGINAL_ATTR = "__kvtier_dev_evict_original__"
POOL_MARKER = "__kvtier_dev_evict_armed__"

# The section's own contract version. The eviction fields are the schema-version-2
# addition to the tier view: a reader can tell a snapshot that CAN see evictions
# (this section, ``schema_version`` 2) from one predating the counter, and
# EXPECTED_FIELDS is the declared key set the AST test in tests/test_monitor_schema.py
# pins against what ``_fields`` actually emits (declared == emitted, both ways).
SCHEMA_VERSION = 2

# THE DECLARED FIELD SET. Every key ``_fields`` emits MUST be listed here and
# every key listed here MUST be emitted (or be in ALLOW_UNEMITTED). This is the
# repo's "every parsed key is declared, every declared key is parsed or
# allow-listed" contract, expressed the way this codebase expresses every other
# one: a single builder dict + a structural test, not a separate reader module.
EXPECTED_FIELDS = (
    "available",
    "reason",
    "schema_version",
    "pure_read",
    "wrap_method",
    "wrap_module",
    "wrap_scope",
    "installs",
    "reinstalls",
    "skips",
    "evictions_total",
    "evicted_spans_per_group",
    "evicted_keys_per_group",
    "evicted_secondary_keys_total",
    "attribution_failures",
    "unknown_group_evictions",
    "fallbacks",
    "live_pool_watched",
    "corroboration",
    "rank_files_can_corroborate",
    "meaning",
    "evicted_spans_meaning",
    "evicted_keys_meaning",
    "why_a_wrap",
    "error",
)
# Declared but deliberately absent from a degraded view is not allowed here; the
# list exists so the test can name any future field that legitimately stops being
# emitted instead of the test just failing. Empty today: every declared field is
# always emitted.
ALLOW_UNEMITTED: tuple = ()

_state: dict = {
    "installs": 0,
    "reinstalls": 0,
    "skips": 0,
    "skip_reason": None,
    # monotonic, process-wide, since the wrapper landed:
    "evictions_total": 0,
    "evicted_spans_per_group": {},   # {primary_group_str: int}
    "evicted_keys_per_group": {},    # {group_str: int}  (primary + secondary)
    "evicted_secondary_keys_total": 0,
    "attribution_failures": 0,
    "unknown_group_evictions": 0,
    "fallbacks": 0,
    "watched": False,                # is the live CPU pool instance wrapped
}


def reset_state_for_tests() -> None:
    """Process-wide state, like ``snapshot._state`` / ``cpu_capacity._state``."""
    _state.update({
        "installs": 0, "reinstalls": 0, "skips": 0, "skip_reason": None,
        "evictions_total": 0, "evicted_spans_per_group": {},
        "evicted_keys_per_group": {}, "evicted_secondary_keys_total": 0,
        "attribution_failures": 0, "unknown_group_evictions": 0,
        "fallbacks": 0, "watched": False,
    })


def counters() -> dict:
    """A copy of the monotonic counters for a test or the wire."""
    return dict(_state)


# --- group decoding, local, no vllm import ----------------------------------

def _group_of_key(key):
    """The group id packed at the END of a ``BlockHashWithGroupId``, or None.

    Mirrors ``kv_cache_utils.get_group_id`` (:91): 4 big-endian unsigned bytes.
    Read locally because this module must never import vllm; the same logic lives
    in ``snapshot._group_of_key`` for the same reason.
    """
    if isinstance(key, (bytes, bytearray)) and len(key) > 4:
        return int.from_bytes(bytes(key)[-4:], "big")
    return None


def _read_groups(pool, block):
    """(primary_group, [group, ...] for every key the discard will drop).

    A cheap PRE-READ of attributes the block already carries, done BEFORE the
    engine resets the hash. ``primary_group`` is the block's own key group; the
    second list is the primary plus every extra hash registered for this block id
    (``cached_block_hashes_by_block``), exactly the set
    ``_remove_cached_block_hashes`` removes. Raises only if the objects are the
    wrong shape, which the wrapper turns into ``attribution_failures``.
    """
    primary_key = getattr(block, "block_hash", None)
    primary_group = _group_of_key(primary_key)
    groups = []
    if primary_key is not None:
        groups.append(primary_group)
    by_block = getattr(pool, "cached_block_hashes_by_block", None)
    if isinstance(by_block, dict):
        block_id = getattr(block, "block_id", None)
        for extra in by_block.get(block_id, ()) or ():
            groups.append(_group_of_key(extra))
    return primary_group, groups


# --- the wrapper -------------------------------------------------------------

def _bump(bucket: dict, group) -> None:
    key = "unknown" if group is None else str(group)
    bucket[key] = bucket.get(key, 0) + 1


def _count_eviction(groups) -> None:
    """Record one confirmed eviction. `groups` is the pre-read (primary, [all]),
    or None when the pre-read could not run. Never raises: a mis-count must never
    be the reason an eviction did not happen."""
    _state["evictions_total"] += 1
    try:
        if groups is None:
            # A real eviction we could not attribute: count the total, admit the
            # miss, invent no group.
            _state["unknown_group_evictions"] += 1
            return
        primary_group, key_groups = groups
        _bump(_state["evicted_spans_per_group"], primary_group)
        if primary_group is None:
            _state["unknown_group_evictions"] += 1
        for group in key_groups:
            _bump(_state["evicted_keys_per_group"], group)
    except Exception as exc:  # noqa: BLE001 - counting never faults
        _state["attribution_failures"] += 1
        snapshot._error(exc, "eviction attribution")


def _make_wrapper(original, pool):
    def _maybe_evict_cached_block(block):
        # Pre-read the group BEFORE the engine resets the block's hash, but do
        # not let that read ever be the reason an eviction does not happen.
        try:
            groups = _read_groups(pool, block) if _truthy_hash(block) else None
        except Exception as exc:  # noqa: BLE001
            groups = None
            _state["fallbacks"] += 1
            snapshot._error(exc, "eviction pre-read")
        evicted = original(block)            # the engine's own verdict; may raise
        if evicted is True:
            _count_eviction(groups)
        return evicted

    setattr(_maybe_evict_cached_block, OWNER_ATTR, OWNER_TAG)
    setattr(_maybe_evict_cached_block, ORIGINAL_ATTR, original)
    _maybe_evict_cached_block.__doc__ = (
        "kvtier-dev-plugin: the pool's own _maybe_evict_cached_block, with a "
        "monotonic per-group eviction count. Contained: the engine's verdict is "
        "returned unchanged and a failed count never blocks an eviction.")
    return _maybe_evict_cached_block


def _truthy_hash(block) -> bool:
    """Is this block plausibly parked (so a discard of it is an eviction)?

    Only a hint for the pre-read; the AUTHORITATIVE signal is still the engine's
    own True return. False just means "skip the pre-read, count unattributed if
    it really was evicted".
    """
    try:
        return getattr(block, "block_hash", None) is not None
    except Exception:  # noqa: BLE001, S110 - the hint is best-effort
        return False


# --- adoption ----------------------------------------------------------------

def arm_pool(pool) -> dict:
    """Wrap the ONE CPU-pool instance's ``_maybe_evict_cached_block``.

    Never class-level (the GPU pool shares ``BlockPool``). Idempotent; refuses a
    foreign owner; never raises. ``reinstalls`` counts re-adoption over our own
    wrapper, which re-binds the ORIGINAL so a repeated snapshot cannot nest.
    """
    report = {"armed": False, "replaced": False, "reason": None}
    try:
        if pool is None:
            return _refuse(report, "no cpu_block_pool bound")
        current = getattr(pool, TARGET_METHOD, None)
        if current is None or not callable(current):
            return _refuse(report, f"{TARGET_CLASS}.{TARGET_METHOD} is missing or "
                                   f"not callable on {type(pool).__name__}")
        owner = getattr(current, OWNER_ATTR, None)
        if owner is not None and owner != OWNER_TAG:
            return _refuse(report, f"{TARGET_CLASS}.{TARGET_METHOD} carries "
                                   f"another owner marker, refusing to overwrite")
        replaced = owner == OWNER_TAG
        if not replaced:
            home = getattr(current, "__module__", None)
            if home != TARGET_MODULE:
                # A bound method reports the defining module on its __func__;
                # accept only the fork's own function as "the engine's".
                underlying = getattr(current, "__func__", current)
                home = getattr(underlying, "__module__", home)
            if home != TARGET_MODULE:
                return _refuse(report, f"{TARGET_METHOD} is neither ours nor the "
                                       f"fork's own (defined in {home!r}, not "
                                       f"{TARGET_MODULE!r})")
        original = getattr(current, ORIGINAL_ATTR, current) if replaced else current
        setattr(pool, TARGET_METHOD, _make_wrapper(original, pool))
        setattr(pool, POOL_MARKER, OWNER_TAG)
        _state["watched"] = True
        _state["skip_reason"] = None
        report["armed"] = True
        report["replaced"] = replaced
        if replaced:
            _state["reinstalls"] += 1
        else:
            _state["installs"] += 1
            logger.info("kvtier-dev-plugin: eviction counter armed on the CPU "
                        "pool %s.%s (instance-level, wrap not pure-read)",
                        TARGET_CLASS, TARGET_METHOD)
        return report
    except Exception as exc:  # noqa: BLE001 - a diagnostic never breaks a boot
        _state["fallbacks"] += 1
        snapshot._error(exc, "eviction arm")
        report["reason"] = f"arm raised: {type(exc).__name__}"[:160]
        return report


def _refuse(report: dict, reason: str) -> dict:
    if _state["skip_reason"] == reason:
        return report
    _state["skips"] += 1
    _state["skip_reason"] = reason
    logger.info("kvtier-dev-plugin: eviction counter NOT armed: %s", reason)
    report["reason"] = reason
    return report


def adopt_if_missed(engine) -> dict:
    """Arm the counter on the live CPU pool, once. Snapshot-driven, no-op after.

    The marker check runs first, so a wrapped pool costs one ``getattr`` per
    snapshot. If the pool cannot be wrapped the section says so honestly instead
    of reporting zeros. Same posture as ``cpu_capacity.adopt_if_missed``.
    """
    report = {"armed": False, "reason": None}
    try:
        manager = getattr(getattr(getattr(engine, "scheduler", None),
                                  "connector", None), "scheduler_manager", None)
        pool = getattr(manager, "cpu_block_pool", None)
        if pool is None:
            report["reason"] = "no cpu_block_pool on this engine"
            return report
        if getattr(pool, POOL_MARKER, None) == OWNER_TAG:
            report["armed"] = True
            report["reason"] = "already armed"
            _state["watched"] = True
            return report
        armed = arm_pool(pool)
        report["armed"] = bool(armed.get("armed"))
        report["reason"] = armed.get("reason")
        return report
    except Exception as exc:  # noqa: BLE001 - a missing section is data, not a 500
        _state["fallbacks"] += 1
        snapshot._error(exc, "eviction adoption")
        report["reason"] = f"adoption raised: {type(exc).__name__}"[:160]
        return report


def uninstall(pool) -> dict:
    """Restore the pool's own method. Never raises."""
    report = {"uninstalled": False, "reason": None}
    try:
        if pool is None:
            report["reason"] = "no pool"
            return report
        current = getattr(pool, TARGET_METHOD, None)
        if getattr(current, OWNER_ATTR, None) != OWNER_TAG:
            report["reason"] = "not ours to remove"
            return report
        setattr(pool, TARGET_METHOD, getattr(current, ORIGINAL_ATTR))
        try:  # noqa: SIM105 - no contextlib import for best-effort cleanup
            delattr(pool, POOL_MARKER)
        except Exception:  # noqa: BLE001, S110 - marker cleanup is best-effort
            pass
        _state["watched"] = False
        report["uninstalled"] = True
        return report
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "eviction uninstall")
        report["reason"] = f"{type(exc).__name__}"[:160]
        return report


# --- the snapshot section -----------------------------------------------------

def _fields(**overrides) -> dict:
    """THE section builder: one dict literal so no path can miss a field.

    ``section()`` and ``empty_section()`` both build through here, so a healthy
    and a degraded view carry the identical key set (the
    ``cpu_capacity._fields``/``test_the_degraded_section_has_every_key`` idiom -
    the one this codebase actually uses to guarantee "every declared key is
    emitted"; see ``EXPECTED_FIELDS`` and ``tests/test_monitor_schema.py``).
    """
    watched = bool(_state["watched"])
    fields = {
        # Honesty first: available is False UNLESS the live CPU pool is wrapped.
        "available": watched,
        "reason": None if watched else (_state["skip_reason"]
                                        or "the CPU pool is not wrapped: no "
                                           "eviction count exists, reporting "
                                           "null rather than a zero"),
        "schema_version": SCHEMA_VERSION,
        # The line the report must say loudly: this is a wrap, not a pure read.
        "pure_read": False,
        "wrap_method": TARGET_METHOD,
        "wrap_module": TARGET_MODULE,
        "wrap_scope": ("instance-level on manager.cpu_block_pool only; the GPU "
                       "pool shares BlockPool and is never wrapped"),
        "installs": int(_state["installs"]),
        "reinstalls": int(_state["reinstalls"]),
        "skips": int(_state["skips"]),
        # The counters. Null (not 0) when unwatched, so absence != "evicted none".
        "evictions_total": (int(_state["evictions_total"]) if watched else None),
        "evicted_spans_per_group": (
            dict(_state["evicted_spans_per_group"]) if watched else None),
        "evicted_keys_per_group": (
            dict(_state["evicted_keys_per_group"]) if watched else None),
        "evicted_secondary_keys_total": (
            _secondary_keys_total() if watched else None),
        "attribution_failures": int(_state["attribution_failures"]),
        "unknown_group_evictions": int(_state["unknown_group_evictions"]),
        "fallbacks": int(_state["fallbacks"]),
        "live_pool_watched": watched,
        "corroboration": None,        # filled in by section(engine), never here
        "rank_files_can_corroborate": (
            "no: the CPU pool and its LRU are scheduler-side; the per-rank worker "
            "files carry pool state as a proven floor only and hold no eviction "
            "signal by construction (worker_diag.py). The corroboration is "
            "scheduler-side, inside the snapshot (see `corroboration`)."),
        "meaning": ("monotonic since boot for THIS engine process and this boot's "
                    "epoch; a reset of the process restarts it at 0. It counts "
                    "parked entries the pool discarded, not stores it declined."),
        "evicted_spans_meaning": ("one per discarded block, attributed to the "
                                  "block's PRIMARY hash group: the span balance "
                                  "to snapshot.parked_blocks_per_group (also "
                                  "primary-keyed). A coarse group's spans being "
                                  "evicted is the joint-depth cost."),
        "evicted_keys_meaning": ("one per discarded INDEX KEY, by the group baked "
                                 "into the key: a coarse FA block evicted also "
                                 "drops its fine-tick keys, so group 5's key "
                                 "evictions run ahead of its span evictions."),
        "why_a_wrap": ("the pool keeps no monotonic eviction counter; the only "
                       "engine hook (KVCacheMetricsCollector.on_block_evicted) is "
                       "1% SAMPLED and drained, a subset histogram, not a total. "
                       "A total exists only if counted at the verdict site, which "
                       "is what this wrapper is. It changes no engine decision."),
        "error": None,
    }
    fields.update(overrides)
    return fields


def _secondary_keys_total() -> int:
    """Fine-tick (secondary) keys evicted: keys minus spans, floored at 0.

    Every discarded block drops its own primary key plus its secondary keys, so
    (keys - spans) is the secondary count. Kept a derivation, not a third
    counter, so the two primitive maps cannot disagree with it.
    """
    keys = sum(_state["evicted_keys_per_group"].values())
    spans = sum(_state["evicted_spans_per_group"].values())
    return max(0, keys - spans)


def _corroboration(engine) -> dict:
    """The scheduler-side cross-check, referencing fields already in the view.

    This is NOT a second eviction source - there isn't one. The occupancy half
    (free / total blocks) is ALREADY published as ``tier.cpu_pool`` and
    ``num_cpu_blocks`` right next to this section, and the "stores climbing"
    half is ``tier.event_counters.store_event_counter``. Re-reading the pool here
    would (a) duplicate those numbers and (b) add a second ``get_num_free_blocks``
    call, i.e. a second error path for a pool that lies - the snapshot promises a
    lying pool costs exactly ONE error, so corroboration reads only plain int
    attributes (``getattr`` with a default, never a call) and references the
    sibling fields by name instead of re-reading them.
    """
    manager = getattr(getattr(getattr(engine, "scheduler", None),
                              "connector", None), "scheduler_manager", None)
    if manager is None:
        return {"available": False, "reason": "no scheduler_manager"}
    try:
        return {
            "available": True,
            "num_cpu_blocks": _int_or_none(getattr(manager, "num_cpu_blocks", None)),
            "store_event_counter": _int_or_none(
                getattr(manager, "_store_event_counter", None)),
            "evictions_total": (int(_state["evictions_total"])
                                if _state["watched"] else None),
            "occupancy_and_free_are_at": (
                "tier.cpu_pool.free_blocks / tier.cpu_pool.total_blocks"),
            "note": ("occupancy = 1 - free/total; evictions_total climbing while "
                     "occupancy stays pinned is the steady state that was "
                     "invisible before: the LRU recycling parked spans to serve "
                     "new stores. store_event_counter is monotonic context, not "
                     "an independent check - there is no second eviction source."),
        }
    except Exception as exc:  # noqa: BLE001 - a corroboration that fails is data
        snapshot._error(exc, "eviction corroboration")
        return {"available": False,
                "reason": f"{type(exc).__name__}"[:160]}


def _int_or_none(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def section(engine=None) -> dict:
    """The tier-view section. Same key set on every path (via ``_fields``)."""
    base = _fields()
    if engine is not None:
        base["corroboration"] = _corroboration(engine)
    return base


def empty_section(reason: str) -> dict:
    """The degraded fallback. Identical key set, ``available`` False."""
    return _fields(available=False, reason=str(reason)[:200], error=str(reason)[:200])
