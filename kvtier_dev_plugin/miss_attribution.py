"""Capability 9: PER-REQUEST MISS ATTRIBUTION and MEASURED tier retention depth.

THE QUESTION THIS EXISTS TO ANSWER
----------------------------------
"WHY did THIS request miss the cache - in VRAM and in the host-RAM tier?" Every
tool the operator had was aggregate (counters, gauges, snapshots), and on
2026-10-04 that cost hours: requests with 176k-token prompts returned
`cached_tokens = 0` while neighbours hit 97%, and no aggregate number could say
which of five real causes produced it. This module turns each tier lookup into
ONE ledger entry with a cause attached, keyed by request id, cheap enough to run
in production.

WHAT IS REUSED, NOT REDONE
--------------------------
`lookup_ledger.py` already owns the per-request ring, the offer/adopt join, the
caps, the read/verdict/section plumbing and the `lookups` verb, and it is already
fed by `guard_mode._record_success()` for BOTH tier methods. This module adds NO
second ledger and NO third wrapper of those two methods: it computes an
`attribution` dict from the same `(method, args, result, manager)` the guard
wrapper already holds, and `lookup_ledger.record_lookup()` attaches it to the
record it was writing anyway. Two pieces of state are added here because nothing
existing can produce them (see THE DEEP OBSERVER and THE FREE-ORDER WITNESS),
both installed INSTANCE-level with the `evictions.arm_pool` owner-marker
discipline, over attributes no other module owns.

THE FIVE CAUSES, AND WHAT EACH ONE COSTS THIS MODULE
----------------------------------------------------
 1 content changed mid-conversation: the chain diverges at some token offset.
 2 alignment/clamp loss: the joint hit is a fixed-point MINIMUM across
   cacheable groups (`kv_cache_coordinator.py:846` `find_longest_cache_hit`,
   the `while True` at `:888`, the `min(` at `:903`), so attention prefix sits
   in VRAM unreachable because a state group is the minimum.
 3 GPU eviction: hashed, idle, then consumed by allocation pressure.
 4 RAM-tier eviction: parked, then evicted before restore.
 5 cold cache: an engine restart or a box suspend wiped everything.

Verdicts are emitted ONLY from data this module actually has; the ones it cannot
compute are listed in `UNDETERMINABLE_VERDICTS` and are NEVER guessed. An honest
`unknown` is worth more than a wrong cause - a wrong one is exactly what cost the
operator a day.

UNITS, AND WHY EVERY FIELD CARRIES ONE
--------------------------------------
Three counters in this system count three different things, and dividing one by
another produced a wrong production conclusion on 2026-10-04:
`_load_event_counter` / `_store_event_counter` count SCHEDULER STEPS with any
load/store (manager.py:537 / :513), `evictions_total` counts BLOCKS
(evictions.py), `external_prefix_cache_hits_total` counts TOKENS. So every field
published here is suffixed or paired with its unit, `UNITS` maps each key to it,
and `rates()` refuses to emit a ratio whose two terms have different units.

THE PREFIX-WALK ARITHMETIC (how a first-miss position is known without reading a
hash value)
-------------------------------------------------------------------------------
The tier's lookup is a STRICT PREFIX WALK: `single_type_kv_cache_manager.py:806-808`
says so out loud ("A missing block implies every later block misses too (chained
hashes)") and the loop `break`s at the first miss. Therefore the depth the walk
reached and the index it failed at are the SAME fact, and both follow from the
hit length:

    matched_spans   = ext_hit_tokens // span_tokens
    first_miss_span = matched_spans            (0-based index INTO THE QUERIED SLICE)
    first_miss_token = local + ext_hit_tokens  (absolute token offset in the prompt)

`hash_value` is NEVER read (it is a `sha256_cbor` container, not an int; the
package-wide rule), and nothing here reads `block.block_hash` either: only
`block_id`-shaped integers and lengths. The depth is DERIVED from the walk's
prefix property, never observed from inside the loop, and the payload says so in
`first_miss_source`.

THE DEEP OBSERVER (cause 2, and the only way to see it)
-------------------------------------------------------
`manager.py:362` calls `cpu_coordinator.find_longest_cache_hit(...)` and THROWS
AWAY its third return value, `num_uncached_common_prefix_tokens` - the engine's
own measure of "some group reached deeper than the reconciled hit". The
connector only ever returns `(hit_length, is_async)`, so at the connector this
cause is INVISIBLE; the ledger would have to guess. So one instance-level
observer is installed on the tier's OWN coordinator instance
(`manager.cpu_coordinator`, never the GPU coordinator, never the class) and
reads, from the value the engine already computed: the joint hit, the residue,
and each group's returned depth. Cost per lookup: O(num_groups) `len()` calls.
No block object is ever stored (only ints), which is the refcount bug class the
PR #56810 backport exists to fix.
Naming the CLAMPING GROUP needs each group's INDEPENDENT depth, which the
engine does not compute on the normal path, so it is opt-in and budgeted:
`find_longest_cache_hit_per_group` (`kv_cache_coordinator.py:984`) is the same
pure lookups through `BlockPool.get_cached_block` (block_pool.py:204-228: one
dict read per key, no `touch`, no queue move, no refcount change - verified),
paid at most once per `sample_every` lookups and at most `MAX_DEEP_PROBES` per
process. When it is off, the verdict says `clamp_group_ambiguous` rather than
naming a group.

THE FREE-ORDER WITNESS (cause 4's ordering defect: CONFIRM OR REFUTE)
---------------------------------------------------------------------
Upstream's GPU path frees a request's blocks in REVERSE chain order so the tail
dies first and the root dies last: `kv_cache_coordinator.py:413`
`blocks.extend(reversed(manager.pop_blocks_for_free(request_id)))`, documented at
`:395-410` ("tail blocks are evicted first"), and `FreeKVCacheBlockQueue`'s own
docstring (`kv_cache_utils.py:257-262`) says the queue's contract is "the tail of
a block chain is at the front" and "we maintain this order by reversing the block
order when free blocks of a request" - the reversing being explicitly OUTSIDE
this class. The CPU tier has ZERO `reversed()` calls anywhere in
`v1/simple_kv_offload/` (verified by grep on the running tree) and its four
`cpu_block_pool.free_blocks(...)` sites (manager.py:1113, :1121, :1272, :1297)
all pass blocks in CHAIN order. `BlockPool.free_blocks` (block_pool.py:770-798)
appends HASHED blocks to the BACK of the free queue in the order given, and
`get_new_blocks` pops from the FRONT (`popleft_n`, :678). Chain order in, then,
means the ROOT is queued before its own tail - and since a missing root makes
every later block unhit-able, that ordering would make the whole parked tail
worthless. Whether it happens is a fact about this fork, not about the grep, so
this witness MEASURES it: a park-sequence number per CPU block id (an int array
the size of the pool) written at allocation, and, per `free_blocks()` call, the
up/down tally of consecutive park-sequence pairs among the blocks that were
hashed. `root_first` CONFIRMS the defect, `root_last` REFUTES it, `mixed` means
this call proved nothing. It does not claim the consequence (pool pressure
actually popping them) - it names the ordering, which is the falsifiable part.

RETAINED DEPTH, MEASURED (not derived from a config formula)
------------------------------------------------------------
Three defensible readings of the live numbers differed by ~785x, because each
derived capacity from a different thing. So `retained_depth()` MEASURES, from the
live objects, and refuses to publish a number it cannot account for:
`gpu_census._walk` (reused verbatim) over the CPU pool gives hashed/referenced/
idle per group; `manager.group_block_sizes[g]` (tokens) turns spans into tokens;
`prefix_cacheable_group_ids` says which groups a hit can use at all. The
published numbers distinguish the two things that got conflated:
  * GROUP-TOKENS: one parked span of group g covers `group_block_sizes[g]`
    tokens OF THAT GROUP. Summing them across groups counts one conversation's
    prefix 5 times, because the tier allocates one CPU slot PER GROUP
    (`manager.py:711`/`:757` inside the per-group store loop).
  * JOINT TOKENS: a lookup needs every cacheable group, so a conversation's
    retrievable depth is bounded by the MINIMUM over cacheable groups, exactly
    the `joint_hashed_blocks_min` idiom `gpu_census` already publishes for VRAM.
And "free" is NOT "empty": `free_blocks` puts parked (hashed) blocks back INTO
the free queue as eviction candidates, so `cpu_pool.free_blocks == parked spans`
is the FULL-TIER signature, not the empty one - which is why the census splits
`cached_idle` from `free_uncached` and why `retrievable_now` is the hashed count,
NOT `total - free`.

BOUNDED, ALWAYS
---------------
Attribution attaches to the ledger's existing ring, so its bound IS the ledger's
`cap`. The deep-observer stashes exactly ONE pending observation (the scheduler
is single-threaded: the coordinator call happens INSIDE the connector call, so
there is no interleaving to lose), the order witness keeps one small ring of
free-call summaries plus monotonic aggregates, and the park-sequence array is one
int per CPU block (bounded by the pool). The file mirror (env
`VLLM_E2_TIER_MISS_LOG`) APPENDS until `MAX_MIRROR_BYTES` and then STOPS, counting
the stop; it never truncates a file and never grows without bound.

HOUSE RULES: stdlib only (`json`, `logging`, `os`, `time`), `except Exception`
and never `BaseException`, every hot-path entry point contained and counted
through `snapshot._error`, no allocation proportional to the batch or the
sequence, no reference to an engine block or block list (ints and lengths only),
no I/O on the hot path, no engine imports. Disarmed, the hot-path entry points
are a dict lookup and a return.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time

from . import config, snapshot

logger = logging.getLogger("kvtier_dev_plugin.miss_attribution")

PROTOCOL_VERSION = 1

# --- bounds ------------------------------------------------------------------
DEFAULT_CAP = 256                    # order-witness ring; the LEDGER has its own cap
MIN_CAP = 16
MAX_CAP = 4096
MAX_GROUPS = 64                      # same bound as gpu_census
MAX_DEEP_PROBES = 64                 # per process, opt-in
DEFAULT_SAMPLE_EVERY = 20
MIN_SAMPLE_EVERY = 1
MAX_SAMPLE_EVERY = 100_000
MAX_STRING_CHARS = 160
MAX_ORDER_SAMPLES = 8                # ring rows in a payload
MAX_PARK_RUN_GAP = 4096              # a pair this far apart is a group boundary
MAX_LIMIT = 256
DEFAULT_LIMIT = 32
MAX_MIRROR_BYTES = 32 * 1024 * 1024  # the mirror stops here, counted
MAX_NOTICES = 32                     # log-once-per-episode keys held
DEPTH_CACHE_SECONDS = 1.0            # the pool walk is ~0.8 ms/1,331 blocks
MAX_PARK_RUN_GAP_REASON = (
    "adjacent freed blocks further apart than MAX_PARK_RUN_GAP are treated as a "
    "group boundary rather than a chain step: a freed list concatenates one "
    "group's chain after another's, and those joins carry no order information")

ENV_MIRROR = config.ENV_MISS_LOG

# --- the verdict vocabulary --------------------------------------------------
# A closed set. `VERDICTS` is publishable so a reader can enumerate the buckets;
# a record lands in exactly one. Anything not derivable stays `unknown`.
V_HIT = "hit"
V_NO_VERDICT = "no_verdict"
V_ZERO_ALIGNMENT_GATE = "zero_by_alignment_gate"
V_ZERO_SKIP_READING = "zero_by_skip_reading_prefix_cache"
V_ZERO_LAST_TOKEN_RULE = "zero_by_last_token_rule"  # noqa: S105 - a verdict NAME
V_ZERO_NO_HASHES_LEFT = "zero_by_no_hashes_left"
V_COLD_CACHE = "cold_cache"
V_POOL_PRESSURE = "pool_pressure"
V_ROOT_MISSING = "root_missing"
V_CLAMP_AT_GROUP = "clamp_at_group"
V_CLAMP_GROUP_AMBIGUOUS = "clamp_group_ambiguous"
V_CLAMP_LIMITED = "clamp_limited"
V_PARTIAL_HIT_CLAMP_SUSPECTED = "partial_hit_clamp_suspected"
V_UNKNOWN = "unknown"

VERDICTS = (V_HIT, V_NO_VERDICT, V_ZERO_ALIGNMENT_GATE, V_ZERO_SKIP_READING,
            V_ZERO_LAST_TOKEN_RULE, V_ZERO_NO_HASHES_LEFT, V_COLD_CACHE,
            V_POOL_PRESSURE, V_ROOT_MISSING, V_CLAMP_AT_GROUP,
            V_CLAMP_GROUP_AMBIGUOUS, V_CLAMP_LIMITED,
            V_PARTIAL_HIT_CLAMP_SUSPECTED, V_UNKNOWN)

# `clamp_at_group<idx>` is the ONE member emitted with a group index appended
# (the task's own spelling), so a reader must know to match a prefix rather than
# an exact name. Every other member is emitted verbatim.
SUFFIXED_VERDICTS = (V_CLAMP_AT_GROUP,)

# Which verdicts the available data can actually decide, in the reader's words.
VERDICT_MEANING = {
    V_HIT: "the tier served the WHOLE length it was allowed to query "
           "(ext_hit_tokens > 0 and residue_tokens == 0): nothing was lost",
    V_NO_VERDICT: "the connector returned (None, ...): the scheduler SKIPS the "
                  "request, so there is no verdict to attribute",
    V_ZERO_ALIGNMENT_GATE: "manager.py:338 returned (0, False) because the local "
                           "hit handed to the connector is not a multiple of the "
                           "scheduler block size",
    V_ZERO_SKIP_READING: "request.skip_reading_prefix_cache is set: the engine "
                         "forbade the lookup (manager.py:335)",
    V_ZERO_LAST_TOKEN_RULE: "manager.py:359-362: at least one token must be "
                            "recomputed, so max_hit_len <= 0 and no lookup ran",
    V_ZERO_NO_HASHES_LEFT: "the queried hash slice is empty: the local hit "
                           "already covers every hashed block",
    V_COLD_CACHE: "nothing at all is parked in the CPU pool AND the tier has "
                  "stored nothing this boot: a restart/suspend wiped everything",
    V_POOL_PRESSURE: "the walk died at span 0 while the pool holds parked spans, "
                     "has evicted since the last deep hit this boot, and the CPU "
                     "free queue is at or under the low-water mark",
    V_ROOT_MISSING: "the walk died at span 0 with parked spans present but NO "
                    "eviction recorded this boot: nothing was ever parked at the "
                    "root (never stored, or a boot we cannot see)",
    V_CLAMP_AT_GROUP: "the joint hit is short because group <idx> independently "
                      "hits shorter than the rest: named by a deep probe",
    V_CLAMP_GROUP_AMBIGUOUS: "the engine's own residue says a group reached "
                             "deeper than the joint hit, but naming it needs a "
                             "deep probe, so the group index is NOT claimed",
    V_CLAMP_LIMITED: "a deep probe found several groups TIED at the minimum, so "
                     "the clamp is real and named as a set, not as one index",
    V_PARTIAL_HIT_CLAMP_SUSPECTED: "the tier served SOME external tokens but "
                                   "less than the length it was allowed to query, "
                                   "and the engine reported no deeper prefix: the "
                                   "tail was simply not parked (never stored, or "
                                   "evicted), or a state group clamped at a length "
                                   "this lookup cannot see",
    V_UNKNOWN: "the fields available at the connector cannot decide the cause",
}

# The honesty core: verdicts this module deliberately does NOT emit.
UNDETERMINABLE_VERDICTS = (
    "content_diverged@<offset>: distinguishing 'the client changed the prompt at "
    "token N' from 'the block at chain index N is gone' needs to know that a "
    "DIFFERENT hash was ever parked at that index, which needs the hash value "
    "(never read here) or a per-request hash history (never kept). The ledger "
    "publishes first_miss_token_offset instead, which is the position; the CAUSE "
    "at that position is not derivable at one request's worth of data",
    "gpu_evicted@<depth> (cause 3) per request: the connector sees the CPU "
    "tier's answer, never the GPU coordinator's walk. The GPU side is visible "
    "only as a census (gpu_census.py) and as engine counters, so a per-request "
    "VRAM-eviction verdict would be a guess about a walk this module cannot see",
    "a single clamping group index on the normal lookup path: the returned "
    "per-group block lists are TRUNCATED to the reconciled hit "
    "(kv_cache_coordinator.py:966-973), so every full-attention group ties at the "
    "minimum and the binder is indistinguishable from the truncated. Only "
    "find_longest_cache_hit_per_group (opt-in, budgeted) names it",
    "any verdict for a call made while disarmed, or for a guard-DECLINED call "
    "(guard_mode returns a decline without reaching _record_success)",
)

# The SHORT unit for every numeric field, attached to each ledger record: the wire
# then carries the unit next to the number, which is the whole lesson of the
# steps-vs-blocks error. The long prose is in `UNITS` below, published once.
UNITS_SHORT = {
    "num_tokens": "tokens", "local_as_passed": "tokens",
    "local_mod_block": "tokens", "local_mod_hash": "tokens",
    "block_size": "tokens/block", "hash_block_size": "tokens/hash block",
    "ext_query_hashes": "hash blocks", "ext_query_tokens_possible": "tokens",
    "ext_query_tokens_capped": "tokens", "ext_hit_tokens": "tokens",
    "residue_tokens": "tokens", "total_residue_tokens": "tokens",
    "first_miss_span": "spans (chain index, DERIVED)",
    "first_miss_token_offset": "tokens", "walk_matched_spans": "spans (DERIVED)",
    "span_tokens": "tokens/span", "joint_hit_tokens": "tokens",
    "clamp_residue_tokens": "tokens", "queried_max_length": "tokens",
    "queried_hashes": "hash blocks",
    "group_depths_tokens": "tokens/group (post-reconciliation)",
    "group_depths_spans": "spans/group (post-reconciliation)",
    "independent_group_depths_tokens": "tokens/group (independent, deep probe)",
}

# Field -> unit. Published so a consumer cannot repeat the steps-vs-blocks error.
UNITS = {
    "num_tokens": "tokens (request.num_tokens, prompt tokens of THIS request)",
    "local_as_passed": "tokens (the block-aligned local hit the scheduler handed "
                       "the connector, scheduler.py:973-979)",
    "local_mod_block": "tokens (local_as_passed % block_size; 0 by construction "
                       "from the scheduler, non-zero from any other caller)",
    "local_mod_hash": "tokens (local_as_passed % hash_block_size)",
    "block_size": "tokens per scheduler block",
    "hash_block_size": "tokens per hash block",
    "ext_query_hashes": "hash blocks (len(request.block_hashes) minus the skipped "
                        "prefix, the slice manager.py:352 actually queries)",
    "ext_query_tokens_possible": "tokens (ext_query_hashes * hash_block_size, the "
                                 "length the tier COULD match before its own caps)",
    "ext_query_tokens_capped": "tokens (what it may match: min(possible, "
                               "num_tokens - 1 - local), manager.py:359-362)",
    "ext_hit_tokens": "tokens (the connector's offer; ledger `hit_returned`)",
    "residue_tokens": "tokens (ext_query_tokens_capped - ext_hit_tokens: the "
                      "prefix that was queried and NOT served)",
    "total_residue_tokens": "tokens (num_tokens - local_as_passed - "
                            "ext_hit_tokens: everything recomputed)",
    "first_miss_span": "spans of the queried group (0-based chain index into the "
                       "queried slice where the prefix walk stopped; DERIVED)",
    "first_miss_token_offset": "tokens (absolute prompt offset where the walk "
                               "stopped = local + ext_hit)",
    "walk_matched_spans": "spans (ext_hit_tokens // span_tokens; DERIVED from the "
                          "strict-prefix-walk property, not observed in the loop)",
    "joint_hit_tokens": "tokens (cpu_coordinator.find_longest_cache_hit's hit)",
    "clamp_residue_tokens": "tokens (the engine's own "
                            "num_uncached_common_prefix_tokens: how much deeper "
                            "at least one group reached than the reconciled hit)",
    "group_depths_tokens": "tokens per group (len(returned blocks) * "
                           "group_block_sizes[g]: the group's depth AFTER the "
                           "engine truncated full-attention groups to the hit)",
    "independent_group_depths_tokens": "tokens per group, each computed "
                                       "independently by a deep probe (no "
                                       "reconciliation), so the minimum is the "
                                       "clamping group",
    "park_seq": "monotonic counter (per CPU block id: the order the tier allocated "
                "it for a park; NOT a token count and NOT a timestamp)",
    "asc_pairs": "adjacent pairs (consecutive freed blocks whose park-sequence "
                 "rose: root-before-tail)",
    "desc_pairs": "adjacent pairs (consecutive freed blocks whose park-sequence "
                  "fell: tail-before-root)",
    "parked_spans_per_group": "spans (parked CPU blocks of that group; ONE span "
                              "covers group_block_sizes[g] TOKENS OF THAT GROUP)",
    "parked_group_tokens_per_group": "group-tokens (spans * tokens-per-span; "
                                     "summing groups multi-counts one "
                                     "conversation by the number of groups)",
    "joint_parked_tokens_max": "tokens (min over prefix-cacheable groups of "
                               "spans * tokens-per-span: an UPPER bound on ONE "
                               "conversation's retrievable joint depth)",
    "aggregate_parked_group_tokens": "group-tokens (the sum over groups: total "
                                     "volume parked, NOT one conversation)",
    "num_cpu_blocks": "slots in the CPU pool (one slot = ONE GROUP's span, so a "
                      "3,200-token span costs one slot per stored group)",
    "slots_per_1k_tokens_one_conversation": "slots per 1,000 tokens for ONE "
                                            "conversation's prefix",
    "per_conversation_ceiling_tokens": "tokens (an UPPER bound for one "
                                       "conversation: num_cpu_blocks divided by "
                                       "the slots-per-token it costs)",
    "hit_length_tokens_observed_max": "tokens (the deepest external hit any "
                                      "recorded lookup actually received: the "
                                      "only per-conversation depth MEASURED here)",
    "load_event_counter": "SCHEDULER STEPS with any load (manager.py:537) - never "
                          "divide by a block or token count",
    "store_event_counter": "SCHEDULER STEPS with any store (manager.py:513)",
    "evictions_total": "BLOCKS discarded (evictions.py)",
    "stored": "BLOCKS stored (boundary_store_stats.stored)",
}

_state: dict = {
    "armed": False,
    "cap": DEFAULT_CAP,
    "ring": None,               # the FREE-ORDER ring; the ledger owns request records
    "next": 0,
    "live": 0,
    "deep": False,              # opt-in per-group probes
    "sample_every": DEFAULT_SAMPLE_EVERY,
    "deep_probes": 0,
    "deep_probe_budget_left": MAX_DEEP_PROBES,
    "lookups_seen": 0,
    "attributions": 0,
    "attribution_errors": 0,
    "coordinator_observed": 0,
    "coordinator_unmatched": 0,
    "last_error": None,
    "verdicts": {name: 0 for name in VERDICTS},
    "mirror_path": None,
    "mirror_written": 0,
    "mirror_bytes": 0,
    "mirror_stopped": None,
    "mirror_errors": 0,
    "install": None,
    "uninstall": None,
    "notices": 0,
}

# The one pending deep observation. Deliberately ONE slot: the coordinator call
# happens INSIDE the connector call on the same thread, so there is nothing to
# interleave; a second value can only mean a shape we did not predict.
_pending: dict | None = None

# --- the free-order witness's state (a second, smaller facility) --------------
_order: dict = {
    "installed": False,
    "pool": None,
    "park_seq": None,        # one int per CPU block id; bounded by the pool
    "seq": 0,
    "freed_calls": 0,
    "freed_blocks": 0,
    "root_first": 0,
    "root_last": 0,
    "mixed": 0,
    "unknown_order": 0,
    "hashed_freed": 0,
    "unattributed_pairs": 0,
    "install_reason": None,
    "uninstall_reason": None,
}

OWNER_ATTR = "__kvtier_dev_miss_owner__"
OWNER_TAG = "kvtier-dev-plugin/miss-attribution-1"
ORIGINAL_ATTR = "__kvtier_dev_miss_original__"

_noticed: set = set()


# --- the log-once-per-episode idiom ------------------------------------------

def notice_once(key: str, message: str) -> None:
    """Log `message` at most once per key per process. Bounded, never raises.

    The pattern the rest of the repo uses (`evictions`' refusal, `blockdump`'s
    skip): a dashboard polls these routes every few seconds, so a per-poll
    warning would drown the log it is trying to write.
    """
    try:
        if key in _noticed:
            return
        if len(_noticed) >= MAX_NOTICES:
            _noticed.clear()
        _noticed.add(key)
        _state["notices"] += 1
        logger.warning("[MISS_ATTR] %s", message[:MAX_STRING_CHARS * 4])
    except Exception as exc:  # noqa: BLE001 - a log line is never worth a fault
        snapshot._error(exc, "miss attribution notice")


def reset_state_for_tests() -> None:
    """Process-wide state, like `snapshot._state` and `lookup_ledger._state`.

    The manager reference goes too: a stale manager surviving a reset is how a
    test would read a previous test's geometry while believing it armed nothing.
    """
    global _pending, _manager_ref
    _manager_ref = None
    _state.update({
        "armed": False, "cap": DEFAULT_CAP, "ring": None, "next": 0, "live": 0,
        "deep": False, "sample_every": DEFAULT_SAMPLE_EVERY, "deep_probes": 0,
        "deep_probe_budget_left": MAX_DEEP_PROBES, "lookups_seen": 0,
        "attributions": 0, "attribution_errors": 0, "coordinator_observed": 0,
        "coordinator_unmatched": 0, "last_error": None,
        "verdicts": {name: 0 for name in VERDICTS}, "mirror_path": None,
        "mirror_written": 0, "mirror_bytes": 0, "mirror_stopped": None,
        "mirror_errors": 0, "install": None, "uninstall": None, "notices": 0,
    })
    _pending = None
    _noticed.clear()
    _order.update({
        "installed": False, "pool": None, "park_seq": None, "seq": 0,
        "freed_calls": 0, "freed_blocks": 0, "root_first": 0, "root_last": 0,
        "mixed": 0, "unknown_order": 0, "hashed_freed": 0,
        "unattributed_pairs": 0, "install_reason": None,
        "uninstall_reason": None,
    })
    _depth_cache["mono_ns"] = 0.0
    _depth_cache["result"] = None


def counters() -> dict:
    out = {
        "armed": bool(_state["armed"]),
        "cap": int(_state["cap"]),
        "lookups_seen": int(_state["lookups_seen"]),
        "attributions": int(_state["attributions"]),
        "attribution_errors": int(_state["attribution_errors"]),
        "coordinator_observed": int(_state["coordinator_observed"]),
        "coordinator_unmatched": int(_state["coordinator_unmatched"]),
        "deep": bool(_state["deep"]),
        "sample_every": int(_state["sample_every"]),
        "deep_probes": int(_state["deep_probes"]),
        "deep_probe_budget_left": int(_state["deep_probe_budget_left"]),
        "verdicts": dict(_state["verdicts"]),
        "last_error": _state["last_error"],
        "mirror_path": _state["mirror_path"],
        "mirror_written": int(_state["mirror_written"]),
        "mirror_bytes": int(_state["mirror_bytes"]),
        "mirror_stopped": _state["mirror_stopped"],
        "mirror_errors": int(_state["mirror_errors"]),
    }
    out.update({f"order_{key}": value for key, value in _order.items()
                if key not in ("pool", "park_seq")})
    return out


# --- small readers -----------------------------------------------------------

def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _positive_int(value) -> int | None:
    value = _int(value)
    return value if value is not None and value > 0 else None


def _attr_int(obj, name: str) -> int | None:
    """An int attribute of an engine object, or None. Never raises."""
    try:
        return _int(getattr(obj, name, None))
    except Exception as exc:  # noqa: BLE001 - a missing attribute is a None
        snapshot._error(exc, "miss attribution attr")
        return None


def _len_or_none(value) -> int | None:
    """len() of a sequence-shaped argument, without retaining it."""
    if value is None:
        return None
    try:
        return len(value)
    except Exception as exc:  # noqa: BLE001 - a lazy view may not support len()
        snapshot._error(exc, "miss attribution len")
        return None


# --- the deep observer: KVCacheCoordinator.find_longest_cache_hit -------------

def observe_coordinator(block_hashes, max_length, result) -> dict | None:
    """Read what the engine ALREADY computed for this lookup. Never raises.

    Reads only `len()`s and the two ints the coordinator returns. The third
    return value is `num_uncached_common_prefix_tokens`: the engine's own
    statement that some group cached deeper than the reconciled hit, i.e. the
    clamp residue of cause 2. Stashed in ONE slot for `attribute()` to consume.
    """
    global _pending
    if not _state["armed"]:
        return None
    try:
        blocks, hit_length, residue = (result + (None, None))[:3] \
            if isinstance(result, tuple) else (None, None, None)
        observation = {
            "joint_hit_tokens": _int(hit_length),
            "clamp_residue_tokens": _int(residue),
            "queried_max_length": _int(max_length),
            "queried_hashes": _len_or_none(block_hashes),
            "group_depths_tokens": None,
            "group_depths_spans": None,
            "groups_at_joint": None,
            "groups_deeper": None,
        }
        depths_tokens: dict = {}
        depths_spans: dict = {}
        deeper: list = []
        at_joint: list = []
        if isinstance(blocks, (list, tuple)):
            sizes = _group_sizes()
            for index, group_blocks in enumerate(blocks[:MAX_GROUPS]):
                spans = _len_or_none(group_blocks)
                if spans is None:
                    continue
                span_tokens = _span_tokens(index, sizes)
                depths_spans[str(index)] = spans
                if span_tokens:
                    depths_tokens[str(index)] = spans * span_tokens
                joint = observation["joint_hit_tokens"]
                if joint is not None and span_tokens:
                    if spans * span_tokens > joint:
                        deeper.append(index)
                    elif spans * span_tokens == joint:
                        at_joint.append(index)
        observation["group_depths_tokens"] = depths_tokens or None
        observation["group_depths_spans"] = depths_spans or None
        observation["groups_deeper"] = deeper[:MAX_GROUPS]
        observation["groups_at_joint"] = at_joint[:MAX_GROUPS]
        _pending = observation
        _state["coordinator_observed"] += 1
        # The opt-in deep probe lives HERE, not only in the wrapper: this is the
        # one place that holds the query's own inputs, so a test (or a future
        # caller) that reaches the observer directly gets the same behaviour as
        # the wrapped path rather than a silently shallower attribution.
        _maybe_deep_probe(block_hashes, max_length)
        return observation
    except Exception as exc:  # noqa: BLE001 - an observer never reaches the step
        _state["attribution_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:120]
        snapshot._error(exc, "miss attribution observe_coordinator")
        return None


def _group_sizes() -> list:
    """The tier's per-group span widths in TOKENS, from the live manager."""
    manager = manager_ref()
    sizes = getattr(manager, "group_block_sizes", None) if manager else None
    if isinstance(sizes, (list, tuple)):
        return [value if isinstance(value, int) and not isinstance(value, bool)
                else None for value in sizes[:MAX_GROUPS]]
    return []


def _span_tokens(index: int, sizes: list) -> int | None:
    if 0 <= index < len(sizes):
        return _positive_int(sizes[index])
    return None


# manager_ref() is set at arm so nothing on the hot path walks the engine graph.
_manager_ref = None


def manager_ref():
    return _manager_ref


def set_manager_ref(manager) -> None:
    global _manager_ref
    _manager_ref = manager


# --- the attribution itself ---------------------------------------------------

def _cacheable_groups() -> list:
    manager = manager_ref()
    ids = getattr(manager, "prefix_cacheable_group_ids", None) if manager else None
    if isinstance(ids, (list, tuple, set, frozenset)):
        return [int(value) for value in ids
                if isinstance(value, int) and not isinstance(value, bool)]
    return []


def _tier_counters() -> dict:
    """The tier's own monotonic context, read defensively, all ints or None."""
    manager = manager_ref()
    out = {"load_event_counter": None, "store_event_counter": None,
           "stored": None, "evictions_total": None}
    if manager is None:
        return out
    try:
        out["load_event_counter"] = _attr_int(manager, "_load_event_counter")
        out["store_event_counter"] = _attr_int(manager, "_store_event_counter")
        stats = getattr(manager, "boundary_store_stats", None)
        if stats is not None:
            out["stored"] = _attr_int(stats, "stored")
        from . import evictions  # noqa: PLC0415 - cycle: evictions imports snapshot
        out["evictions_total"] = _int(
            (evictions.counters() or {}).get("evictions_total"))
    except Exception as exc:  # noqa: BLE001 - context is optional, never fatal
        snapshot._error(exc, "miss attribution context")
    return out


def attribute(request, local, result, manager=None) -> dict:
    """THE per-request attribution. Pure function of what the wrapper holds.

    Returns a dict with UNITS-published fields and exactly one verdict. Every
    field is None when it cannot be computed - a missing field is visible, a
    guessed one is a lie. Called from `lookup_ledger.record_lookup()` through
    `observe_connector()`, i.e. on the scheduler's `schedule()` path, so it does
    no I/O and allocates O(1).
    """
    out: dict = {"version": PROTOCOL_VERSION, "verdict": V_UNKNOWN,
                 "verdict_why": "no fields could be computed from this call",
                 "first_miss_source": None,
                 # A pointer, not the prose: the full `UNDETERMINABLE_VERDICTS` is
                 # published ONCE by `section()`. Copying 733 bytes of constant
                 # text into every record would halve how many fit in a response,
                 # and a ledger that answers "show me the misses" with a refusal
                 # because of boilerplate is a bad trade.
                 "not_established": "see snapshot.miss_attribution.undeterminable"
                 " (content divergence and per-request VRAM eviction are NOT "
                 "derivable here)"}
    try:
        block = _positive_int(getattr(manager, "block_size", None)) \
            if manager else None
        hash_block = _positive_int(getattr(manager, "hash_block_size", None)) \
            if manager else None
        num_tokens = _attr_int(request, "num_tokens")
        aligned = _int(local)
        out.update({"num_tokens": num_tokens, "local_as_passed": aligned,
                    "block_size": block, "hash_block_size": hash_block,
                    "local_mod_block": (aligned % block)
                    if (aligned is not None and block) else None,
                    "local_mod_hash": (aligned % hash_block)
                    if (aligned is not None and hash_block) else None})

        offered = None
        if isinstance(result, tuple) and len(result) > 0:
            offered = _int(result[0])
        out["ext_hit_tokens"] = offered

        # The queried slice, recomputed exactly as manager.py:351-358 does, from
        # a len() of the hash list - the list itself is never retained.
        hashes = _attr_int(request, "_block_hashes_len")
        if hashes is None:
            hashes = _len_or_none(_raw_hashes(request))
        skipped = (aligned // hash_block) if (aligned is not None and hash_block
                                              and aligned >= 0) else None
        if hashes is not None and skipped is not None:
            queried = max(0, hashes - skipped)
            out["ext_query_hashes"] = queried
            if hash_block:
                possible = queried * hash_block
                out["ext_query_tokens_possible"] = possible
                if num_tokens is not None and aligned is not None:
                    allowed = num_tokens - 1 - aligned
                    out["ext_query_tokens_capped"] = max(
                        0, min(possible, allowed))
        out["residue_tokens"] = (
            out["ext_query_tokens_capped"] - offered
            if isinstance(out.get("ext_query_tokens_capped"), int)
            and offered is not None else None)
        out["total_residue_tokens"] = (
            num_tokens - aligned - (offered or 0)
            if (num_tokens is not None and aligned is not None) else None)

        # The prefix-walk arithmetic (see the module docstring): depth and the
        # failing index are ONE fact under a strict prefix walk.
        span = _span_for_local(block, hash_block)
        if offered is not None and offered >= 0 and span:
            matched = offered // span
            out["walk_matched_spans"] = matched
            out["first_miss_span"] = matched
            out["first_miss_token_offset"] = (aligned or 0) + offered
            out["first_miss_source"] = (
                "derived from the strict-prefix-walk property "
                "(single_type_kv_cache_manager.py:806-808, the loop breaks at the "
                "first miss): matched = ext_hit_tokens // span_tokens")
        out["span_tokens"] = span

        # The deep observation, if the coordinator observer ran for this lookup.
        observation = _take_pending()
        if observation is not None and observation.get("joint_hit_tokens") \
                == offered:
            out.update({key: value for key, value in observation.items()
                        if key not in ("joint_hit_tokens",)})
            out["coordinator_matched"] = True
        else:
            out["coordinator_matched"] = observation is not None
            if observation is not None:
                _state["coordinator_unmatched"] += 1
            out.setdefault("joint_hit_tokens", None)
            out.setdefault("clamp_residue_tokens", None)

        verdict, why = _decide(out, request)
        out["verdict"] = verdict
        out["verdict_why"] = why
        out["unit"] = {key: UNITS_SHORT[key] for key in out
                       if key in UNITS_SHORT}
        _state["verdicts"][verdict] = _state["verdicts"].get(verdict, 0) + 1
        _state["attributions"] += 1
    except Exception as exc:  # noqa: BLE001 - attribution must never break a step
        _state["attribution_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:120]
        snapshot._error(exc, "miss attribution attribute")
    return out


def _raw_hashes(request):
    try:
        return getattr(request, "block_hashes", None)
    except Exception:  # noqa: BLE001 - a request that cannot name its hashes
        return None


def _span_for_local(block, hash_block) -> int | None:
    """The span width the external walk advances in.

    The tier's external hit is aligned to `hash_block_size` when the coordinator
    allows partial hash hits (manager.py:437-439 asserts exactly that) and to the
    scheduler block size otherwise, so the walk's unit is read from the live
    manager rather than assumed.
    """
    manager = manager_ref()
    coordinator = getattr(manager, "cpu_coordinator", None) if manager else None
    partial = getattr(coordinator, "enable_partial_hash_hits", None)
    if partial is True and hash_block:
        return hash_block
    return block or hash_block


def _take_pending() -> dict | None:
    global _pending
    pending, _pending = _pending, None
    return pending


def _decide(fields: dict, request) -> tuple[str, str]:
    """THE verdict function. Kept separate so a mutation test can hit it alone.

    Order matters and is the reasoning: an engine gate that PREVENTED the lookup
    is a different fact from a lookup that ran and found nothing, which is a
    different fact from a lookup that ran deep and stopped short.
    """
    offered = fields.get("ext_hit_tokens")
    local = fields.get("local_as_passed")
    block = fields.get("block_size")
    num_tokens = fields.get("num_tokens")

    if offered is None:
        return V_NO_VERDICT, VERDICT_MEANING[V_NO_VERDICT]
    if offered > 0:
        residue = fields.get("clamp_residue_tokens")
        independent = fields.get("independent_group_depths_tokens")
        if isinstance(independent, dict) and independent:
            # A group the tier never parks (the QSA scratch ring) hits 0 by
            # construction and is not a clamp: min over the CACHEABLE groups only.
            cacheable = _cacheable_groups()
            depths = {key: value for key, value in independent.items()
                      if isinstance(value, int) and isinstance(key, str)
                      and key.lstrip("-").isdigit()
                      and (not cacheable or int(key) in cacheable)}
            if depths:
                shortest = min(depths, key=lambda key: depths[key])
                tied = [key for key, value in depths.items()
                        if value == depths[shortest]]
                served = fields.get("ext_hit_tokens")
                # The probe only names a clamp when the shortest independent depth
                # IS the served length: if every group could serve more than the
                # tier gave, the reason is elsewhere and this would be a lie.
                binding = (isinstance(served, int)
                           and depths[shortest] <= served)
                deepest = max(depths.values())
                if binding and len(tied) == 1 and depths[shortest] < deepest:
                    return f"{V_CLAMP_AT_GROUP}{shortest}", (
                        "deep probe: group %s independently hits %d tokens, "
                        "uniquely the shortest of %d probed groups"
                        % (shortest, depths[shortest], len(depths)))
                if binding and len(tied) > 1 and depths[shortest] < deepest:
                    return V_CLAMP_LIMITED, (
                        "deep probe names a TIE at the minimum: "
                        + ",".join(sorted(tied)))
        if isinstance(residue, int) and residue > 0:
            candidates = fields.get("groups_at_joint") or []
            if len(candidates) == 1:
                return f"{V_CLAMP_AT_GROUP}{candidates[0]}", (
                    "the engine reports %d tokens of deeper prefix than the "
                    "reconciled hit and exactly one group sits at the joint "
                    "depth" % residue)
            return V_CLAMP_GROUP_AMBIGUOUS, (
                "the engine reports %d tokens of deeper prefix than the joint "
                "hit (a clamp), but the returned per-group lists are truncated "
                "to the hit so the binding group is not named without a deep "
                "probe; candidates at the joint depth: %s" % (
                    residue, [int(value) for value in candidates][:8]))
        # No clamp evidence: did the tier serve everything it was allowed to ask
        # for, or did it stop short of its own query length?
        short_by = fields.get("residue_tokens")
        if isinstance(short_by, int) and short_by > 0:
            return V_PARTIAL_HIT_CLAMP_SUSPECTED, (
                VERDICT_MEANING[V_PARTIAL_HIT_CLAMP_SUSPECTED]
                + " (queried %d tokens more than it got)" % short_by)
        return V_HIT, VERDICT_MEANING[V_HIT]

    # offered == 0: which gate fired, in the order the fork tests them
    # (manager.py:335 -> :338 -> :351 -> :355).
    if getattr(request, "skip_reading_prefix_cache", False) is True:
        return V_ZERO_SKIP_READING, VERDICT_MEANING[V_ZERO_SKIP_READING]
    if local is not None and block and local % block != 0:
        return V_ZERO_ALIGNMENT_GATE, VERDICT_MEANING[V_ZERO_ALIGNMENT_GATE]
    if fields.get("ext_query_hashes") == 0:
        return V_ZERO_NO_HASHES_LEFT, VERDICT_MEANING[V_ZERO_NO_HASHES_LEFT]
    if num_tokens is not None and local is not None and num_tokens - 1 - local <= 0:
        return V_ZERO_LAST_TOKEN_RULE, VERDICT_MEANING[V_ZERO_LAST_TOKEN_RULE]

    # Nothing matched at all. Say WHERE (span 0 vs deeper) and use the tier's own
    # context for the only causes distinguishable without a hash value.
    context = _tier_counters()
    parked = _parked_total()
    if offered == 0 and parked is None:
        # No measurement, no claim: this is the case where the CPU pool could not
        # be walked at all, and "root missing" / "pool pressure" would both be
        # invented. `unknown` exists for exactly this.
        return V_UNKNOWN, ("the CPU pool could not be measured, so nothing about "
                           "eviction or a cold cache can be claimed for this miss")
    if offered == 0 and parked == 0:
        if not context.get("stored"):
            return V_COLD_CACHE, VERDICT_MEANING[V_COLD_CACHE] + (
                " (stored=%r this boot)" % context.get("stored"))
        return V_UNKNOWN, ("nothing is parked right now but stores happened this "
                           "boot: everything parked has been evicted, or this "
                           "conversation's prefix was never stored")
    if offered == 0 and fields.get("walk_matched_spans") == 0:
        if context.get("evictions_total"):
            return V_POOL_PRESSURE, VERDICT_MEANING[V_POOL_PRESSURE] + (
                " (evictions_total=%d blocks this boot)"
                % context["evictions_total"])
        return V_ROOT_MISSING, VERDICT_MEANING[V_ROOT_MISSING] + (
            " (%s parked spans present, no eviction this boot)" % parked)
    # Defensive fallthrough: a strict prefix walk cannot have matched spans and
    # offered 0 at once, so this only fires if that property ever stops holding
    # (a fork that changes the walk), and it says so instead of inventing a cause.
    if offered == 0:
        return V_UNKNOWN, ("the walk matched %r spans then stopped; with a "
                           "non-empty pool and no clamp residue this is either a "
                           "content change or an eviction at that index, and "
                           "neither is derivable from one lookup"
                           % fields.get("walk_matched_spans"))
    return V_UNKNOWN, VERDICT_MEANING[V_UNKNOWN]


# --- the hot-path entry point (called by lookup_ledger) -----------------------

def observe_connector(request, local, result, manager=None) -> dict | None:
    """Compute and mirror the attribution. Never raises; disarmed: 1 branch."""
    if not _state["armed"]:
        return None
    try:
        _state["lookups_seen"] += 1
        fields = attribute(request, local, result, manager)
        _mirror(fields)
        return fields
    except Exception as exc:  # noqa: BLE001 - a ledger never reaches the scheduler
        _state["attribution_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:120]
        snapshot._error(exc, "miss attribution observe_connector")
        return None


# --- the file mirror ----------------------------------------------------------

def _mirror(fields: dict) -> None:
    """Append one JSON line, bounded. BUDGETED, and it says when it stopped.

    Hot path? No: the env var is unset by default, and the write is one small
    line per lookup (a lookup is once per request, not per token). It is still
    capped: past `MAX_MIRROR_BYTES` it stops and counts the stop, because an
    unbounded append on a live coding-agent box is an outage waiting for a full
    disk.
    """
    path = _state["mirror_path"]
    if not path:
        return
    try:
        if _state["mirror_stopped"]:
            return
        line = json.dumps({"t": round(time.time(), 6),
                           "req": fields.get("req"),
                           **{key: value for key, value in fields.items()
                              if key != "units"}},
                          separators=(",", ":"))[:8192] + "\n"
        size = len(line.encode("utf-8"))
        if _state["mirror_bytes"] + size > MAX_MIRROR_BYTES:
            _state["mirror_stopped"] = ("reached MAX_MIRROR_BYTES=%d after %d "
                                        "lines" % (MAX_MIRROR_BYTES,
                                                   _state["mirror_written"]))
            notice_once("mirror_full",
                        "the miss-attribution mirror stopped at %s: %s"
                        % (path, _state["mirror_stopped"]))
            return
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
        _state["mirror_written"] += 1
        _state["mirror_bytes"] += size
    except Exception as exc:  # noqa: BLE001 - a mirror is never worth a fault
        _state["mirror_errors"] += 1
        snapshot._error(exc, "miss attribution mirror")


# --- the free-order witness ---------------------------------------------------

def arm_order_witness(pool) -> dict:
    """Wrap ONE CPU pool instance: `get_new_blocks` and `free_blocks`.

    Instance-level only, with the owner-marker discipline of `evictions.arm_pool`
    (a class-level wrap would put a KV-tier observer on the GPU pool too). The
    wrappers return the engine's own values UNCHANGED and only ever move an int.
    """
    report = {"installed": False, "reason": None, "pool": type(pool).__name__}
    if pool is None:
        report["reason"] = "no CPU block pool reachable on the tier manager"
        return report
    if _order["installed"] and _order["pool"] is pool:
        return {"installed": True, "reason": "already installed on this pool",
                "pool": report["pool"]}
    try:
        blocks = getattr(pool, "blocks", None)
        size = _len_or_none(blocks)
        if not size or size <= 0 or size > 1_000_000:
            report["reason"] = (f"pool.blocks is {size!r}: the park-sequence array "
                                "is one int per block, so an unbounded or missing "
                                "pool is refused rather than allocated")
            return report
        if getattr(pool, OWNER_ATTR, None) not in (None, OWNER_TAG):
            report["reason"] = (f"a foreign owner holds this pool "
                                f"({getattr(pool, OWNER_ATTR)!r}); refusing to "
                                "clobber it")
            return report
        park = [0] * size
        original_new = getattr(pool, "get_new_blocks", None)
        original_free = getattr(pool, "free_blocks", None)
        if not callable(original_new) or not callable(original_free):
            report["reason"] = ("the pool exposes neither get_new_blocks nor "
                               "free_blocks: this fork renamed one")
            return report
        setattr(pool, ORIGINAL_ATTR, (original_new, original_free))

        def get_new_blocks(num_blocks, _park=park):
            blocks_out = original_new(num_blocks)
            try:
                if _state["armed"]:
                    for block in blocks_out:
                        block_id = getattr(block, "block_id", None)
                        if isinstance(block_id, int) and 0 <= block_id < len(_park):
                            _order["seq"] += 1
                            _park[block_id] = _order["seq"]
            except Exception as exc:  # noqa: BLE001 - never into the allocator
                snapshot._error(exc, "miss attribution park")
            return blocks_out

        def free_blocks(ordered_blocks, _park=park):
            # MATERIALISE FIRST, observe second. `manager.py:1297` passes a
            # GENERATOR (`free_blocks(self.cpu_block_pool.blocks[bid] for bid in
            # ...)`, `_cleanup_load_request`), and an observer that iterates it
            # would leave the engine's own function an EMPTY sequence - blocks
            # never freed, refs leaked, the tier slowly deadlocked. That is the
            # one way a read-only witness could change the engine, so the sequence
            # is resolved once and the SAME object goes to the original.
            try:
                if not isinstance(ordered_blocks, (list, tuple)):
                    ordered_blocks = list(ordered_blocks)
            except Exception as exc:  # noqa: BLE001 - a bad argument is the engine's
                snapshot._error(exc, "miss attribution free materialise")
                return original_free(ordered_blocks)
            try:
                if _state["armed"]:
                    _observe_free(ordered_blocks, _park)
            except Exception as exc:  # noqa: BLE001 - never into the free path
                snapshot._error(exc, "miss attribution free")
            return original_free(ordered_blocks)

        pool.get_new_blocks = get_new_blocks
        pool.free_blocks = free_blocks
        setattr(pool, OWNER_ATTR, OWNER_TAG)   # noqa: B010 - a dynamic name
        _order.update({"installed": True, "pool": pool, "park_seq": park,
                       "install_reason": "wrapped get_new_blocks + free_blocks "
                                         "on this instance"})
        report.update({"installed": True,
                       "reason": _order["install_reason"],
                       "pool_blocks": size})
        return report
    except Exception as exc:  # noqa: BLE001 - a witness that cannot install is fine
        _state["last_error"] = f"{type(exc).__name__}"[:120]
        snapshot._error(exc, "miss attribution arm_order_witness")
        report["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        return report


def _observe_free(ordered_blocks, park: list) -> None:
    """The order statistic for one `free_blocks()` call. Never raises."""
    ids: list = []
    hashed = 0
    for block in ordered_blocks:
        block_id = getattr(block, "block_id", None)
        if not isinstance(block_id, int) or isinstance(block_id, bool):
            continue
        if getattr(block, "block_hash", None) is not None:
            hashed += 1
        ids.append(block_id)
    _order["freed_calls"] += 1
    _order["freed_blocks"] += len(ids)
    _order["hashed_freed"] += hashed
    if hashed < 2:
        _order["unknown_order"] += 1
        return
    ascending = descending = unknown = 0
    for first, second in zip(ids, ids[1:], strict=False):
        left = park[first] if 0 <= first < len(park) else 0
        right = park[second] if 0 <= second < len(park) else 0
        if not left or not right:
            unknown += 1
            continue
        if right > left and right - left <= MAX_PARK_RUN_GAP:
            ascending += 1
        elif left > right and left - right <= MAX_PARK_RUN_GAP:
            descending += 1
    if ascending > descending:
        _order["root_first"] += 1
    elif descending > ascending:
        _order["root_last"] += 1
    else:
        _order["mixed"] += 1
    _order["unattributed_pairs"] += unknown
    _push_order({"freed": len(ids), "hashed": hashed,
                 "asc_pairs": ascending, "desc_pairs": descending,
                 "unknown_pairs": unknown,
                 "order": ("root_first" if ascending > descending else
                           "root_last" if descending > ascending else "mixed")})


def _push_order(entry: dict) -> None:
    ring = _state["ring"]
    if ring is None:
        return
    entry["seq"] = int(_state["next"])
    ring[int(_state["next"])] = entry
    _state["next"] = (int(_state["next"]) + 1) % len(ring)
    _state["live"] = min(len(ring), int(_state["live"]) + 1)


def order_report(limit: int = DEFAULT_LIMIT) -> dict:
    """What the witness has seen, and what it would take to settle the claim."""
    wanted = _int(limit) or DEFAULT_LIMIT
    wanted = max(1, min(MAX_LIMIT, wanted))
    ring = [entry for entry in (_state["ring"] or []) if entry is not None]
    entries = ring[-wanted:]
    total = (_order["root_first"] + _order["root_last"] + _order["mixed"])
    return {
        "installed": bool(_order["installed"]),
        "reason": _order["install_reason"] or _order["uninstall_reason"],
        "calls": int(_order["freed_calls"]),
        "decisive_calls": total,
        "root_first": int(_order["root_first"]),
        "root_last": int(_order["root_last"]),
        "mixed": int(_order["mixed"]),
        "unknown": int(_order["unknown_order"]),
        "freed_blocks": int(_order["freed_blocks"]),
        "hashed_freed": int(_order["hashed_freed"]),
        "unattributed_pairs": int(_order["unattributed_pairs"]),
        "recent": entries,
        "verdict": (None if int(_order["freed_calls"]) == 0 else
                    ("INCONCLUSIVE so far: no freed call yet carried two hashed "
                     "blocks whose park order is known, so this witness has "
                     "nothing to compare (arm it BEFORE the traffic you care "
                     "about; blocks parked before arming have no park "
                     "sequence)") if total == 0 else
                    "CONFIRMED: the CPU tier queues a parked chain ROOT before "
                    "its own tail, so a root dies before the tail it makes "
                    "unhit-able (free_blocks appends hashed blocks in the order "
                    "given, block_pool.py:796-798; get_new_blocks pops the front, "
                    "block_pool.py:678)"
                    if _order["root_first"] > _order["root_last"] else
                    "REFUTED: freed CPU chains arrive tail-first (reversed), the "
                    "same order the GPU path uses at kv_cache_coordinator.py:413, "
                    "so the prefix root is queued to die LAST"
                    if _order["root_last"] > _order["root_first"] else
                    "INCONCLUSIVE so far: the calls seen were neither "
                    "monotonically root-first nor tail-first"),
        "units": {key: UNITS[key] for key in
                  ("asc_pairs", "desc_pairs", "park_seq")},
        "run_gap_rule": MAX_PARK_RUN_GAP_REASON,
        "not_established": (
            "this witness measures the ORDER a freed chain enters the free queue, "
            "which is the falsifiable half of the hypothesis; it does NOT measure "
            "whether allocation pressure then pops that root before a restore "
            "needs it (that needs the pool's pop history, and evictions_total "
            "counts blocks without positions)"),
    }


def _unarm_order_witness() -> dict:
    pool = _order["pool"]
    if pool is None or not _order["installed"]:
        return {"uninstalled": False, "reason": "not installed"}
    try:
        originals = getattr(pool, ORIGINAL_ATTR, None)
        if originals is None:
            return {"uninstalled": False,
                    "reason": "no stored originals: refusing to guess what to "
                              "restore"}
        pool.get_new_blocks = originals[0]
        pool.free_blocks = originals[1]
        with contextlib.suppress(Exception):
            # an attribute that will not go away is not worth failing a disarm
            delattr(pool, OWNER_ATTR)
        _order.update({"installed": False, "pool": None, "park_seq": None,
                       "uninstall_reason": "restored the stored originals"})
        return {"uninstalled": True, "reason": _order["uninstall_reason"]}
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "miss attribution unarm")
        return {"uninstalled": False, "reason": f"{type(exc).__name__}: {exc}"}


# --- the coordinator observer's install --------------------------------------

def arm_coordinator_observer(manager) -> dict:
    """Instance-level observer on `manager.cpu_coordinator.find_longest_cache_hit`.

    Never the CLASS: the same class backs the GPU coordinator, and a class-level
    wrap would put this work on every scheduler step for every request. The
    wrapper returns the engine's tuple UNCHANGED.
    """
    report = {"installed": False, "reason": None}
    coordinator = getattr(manager, "cpu_coordinator", None) if manager else None
    if coordinator is None:
        report["reason"] = "the tier manager exposes no cpu_coordinator"
        return report
    if getattr(coordinator, OWNER_ATTR, None) not in (None, OWNER_TAG):
        report["reason"] = (f"a foreign owner holds the coordinator "
                            f"({getattr(coordinator, OWNER_ATTR)!r})")
        return report
    if getattr(coordinator, OWNER_ATTR, None) == OWNER_TAG:
        return {"installed": True, "reason": "already installed on this instance"}
    original = getattr(coordinator, "find_longest_cache_hit", None)
    if not callable(original):
        report["reason"] = "cpu_coordinator exposes no find_longest_cache_hit"
        return report

    def find_longest_cache_hit(block_hashes, max_cache_hit_length, *_extra,
                               **_kwargs):
        result = original(block_hashes, max_cache_hit_length, *_extra, **_kwargs)
        try:
            if _state["armed"]:
                observe_coordinator(block_hashes, max_cache_hit_length, result)
        except Exception as exc:  # noqa: BLE001 - an observer never breaks a step
            snapshot._error(exc, "miss attribution coordinator")
        return result

    try:
        # A dynamic-by-policy attribute name: the owner marker must be settable on
        # an instance we do not own the class of, so the literal name stays.
        setattr(coordinator, "find_longest_cache_hit",   # noqa: B010
                find_longest_cache_hit)
        setattr(coordinator, ORIGINAL_ATTR, original)   # noqa: B010 - a dynamic name
        setattr(coordinator, OWNER_ATTR, OWNER_TAG)   # noqa: B010 - ditto
        report.update({"installed": True,
                       "reason": "observing the joint hit, the clamp residue and "
                                 "the per-group depths this lookup already "
                                 "computed"})
        return report
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "miss attribution arm_coordinator_observer")
        report["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        return report


def _unarm_coordinator_observer(manager) -> dict:
    coordinator = getattr(manager, "cpu_coordinator", None) if manager else None
    if coordinator is None:
        return {"uninstalled": False, "reason": "no cpu_coordinator here"}
    original = getattr(coordinator, ORIGINAL_ATTR, None)
    if original is None:
        return {"uninstalled": False, "reason": "not installed here"}
    try:
        coordinator.find_longest_cache_hit = original
        delattr(coordinator, ORIGINAL_ATTR)
        delattr(coordinator, OWNER_ATTR)
        return {"uninstalled": True, "reason": "restored the stored original"}
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "miss attribution unarm coordinator")
        return {"uninstalled": False, "reason": f"{type(exc).__name__}: {exc}"}


def _maybe_deep_probe(block_hashes, max_length) -> None:
    """One extra, read-only, per-group lookup - budgeted and opt-in.

    `find_longest_cache_hit_per_group` (kv_cache_coordinator.py:984) runs the
    SAME per-group `find_longest_cache_hit` classmethods without reconciling them,
    so its per-group lengths are each group's INDEPENDENT depth: the minimum names
    the clamping group. It is a re-lookup through `BlockPool.get_cached_block`
    (block_pool.py:204-228: dict reads, no touch, no queue move), which is why it
    is safe and also why it is sampled rather than always.
    """
    if not _state["deep"] or _state["deep_probe_budget_left"] <= 0:
        return
    every = max(MIN_SAMPLE_EVERY, min(MAX_SAMPLE_EVERY,
                                      int(_state["sample_every"])))
    if _state["lookups_seen"] % every:
        return
    manager = manager_ref()
    coordinator = getattr(manager, "cpu_coordinator", None) if manager else None
    per_group = getattr(coordinator, "find_longest_cache_hit_per_group", None)
    if not callable(per_group):
        _state["deep"] = False
        notice_once("no_per_group",
                    "cpu_coordinator exposes no find_longest_cache_hit_per_group: "
                    "deep attribution is off, clamp group will stay unnamed")
        return
    try:
        _blocks, lengths = per_group(block_hashes, max_length)
        values = lengths if isinstance(lengths, (list, tuple)) else ()
        out: dict = {}
        for index, value in enumerate(values[:MAX_GROUPS]):
            # An int is a depth in tokens; anything else is reported as 0 rather
            # than dropped, so the group set cannot silently shrink under a probe.
            out[str(index)] = (value if isinstance(value, int)
                               and not isinstance(value, bool) else 0)
        if out and _pending is not None:
            _pending["independent_group_depths_tokens"] = out
        _state["deep_probes"] += 1
        _state["deep_probe_budget_left"] -= 1
    except Exception as exc:  # noqa: BLE001 - a probe is optional by construction
        _state["deep_probe_budget_left"] = max(
            0, int(_state["deep_probe_budget_left"]) - 1)
        snapshot._error(exc, "miss attribution deep probe")


# --- measured retention depth -------------------------------------------------

_depth_cache: dict = {"mono_ns": 0.0, "result": None}


def _parked_total() -> int | None:
    """Parked (hashed) CPU spans, right now, or None if the pool is unreadable.

    Uses the memoised reading: only zero-hit lookups ask, and a pool walk costs
    ~0.8 ms per 1,331 blocks (`gpu_census`'s own measurement), so it is paid per
    miss inside a `DEPTH_CACHE_SECONDS` window rather than per lookup.
    """
    section = retained_depth()
    total = section.get("parked_spans_total")
    return total if isinstance(total, int) else None


def retained_depth(force: bool = False, engine=None) -> dict:
    """How much prefix the RAM tier can serve RIGHT NOW, MEASURED.

    Reuses `gpu_census._walk` (one pass, refuses on anything it cannot classify)
    over the tier's OWN `cpu_block_pool`, and the live `group_block_sizes` in
    tokens, so no capacity number here comes from a config formula. Caps the walk
    the way `gpu_census` does: a refusal, never a partial total. Memoised for
    `DEPTH_CACHE_SECONDS` unless `force`, because a dashboard polls this.

    Readable UNARMED: `engine` resolves the tier manager itself, so the snapshot
    can answer "how much is retained" without anybody arming the ledger first.
    """
    from . import gpu_census  # noqa: PLC0415 - cycle: gpu_census imports snapshot
    now = time.monotonic()
    cached = _depth_cache["result"]
    if (not force and isinstance(cached, dict)
            and now - float(_depth_cache["mono_ns"]) < DEPTH_CACHE_SECONDS):
        out = dict(cached)
        out["cached"] = True
        return out
    manager = manager_ref()
    if manager is None and engine is not None:
        try:
            from . import guard_mode  # noqa: PLC0415 - cycle, as lookup_ledger does
            manager = guard_mode.manager_of(engine)
        except Exception as exc:  # noqa: BLE001 - a resolution failure is a reason
            snapshot._error(exc, "miss attribution manager resolve")
            manager = None
    pool = getattr(manager, "cpu_block_pool", None) if manager else None
    if pool is None:
        return {"available": False,
                "reason": "no tier manager / cpu_block_pool reachable (the "
                          "ledger is armed from a verb, so arm it first)"}
    try:
        blocks = getattr(pool, "blocks", None)
        size = _len_or_none(blocks)
        if not size:
            return {"available": False,
                    "reason": f"pool.blocks is {size!r}: nothing to measure"}
        if size > gpu_census.MAX_WALKED_BLOCKS:
            return {"available": False,
                    "reason": ("a %d-block pool exceeds MAX_WALKED_BLOCKS=%d: "
                               "REFUSED rather than reported as a partial walk"
                               % (size, gpu_census.MAX_WALKED_BLOCKS))}
        walk = gpu_census._walk(list(blocks))
    except Exception as exc:  # noqa: BLE001 - a census is never a fault
        snapshot._error(exc, "miss attribution retained_depth")
        return {"available": False,
                "reason": f"{type(exc).__name__}: {exc}"[:200]}

    sizes = _group_sizes()
    cacheable = _cacheable_groups()
    spans = walk["hashed_per_group"]
    idle = walk["hashed_idle_per_group"]
    per_group_tokens: dict = {}
    per_group_idle_tokens: dict = {}
    for key, value in spans.items():
        try:
            index = int(key)
        except Exception:  # noqa: BLE001, S112 - an unparsable key is skipped
            continue
        tokens = _span_tokens(index, sizes)
        if tokens:
            per_group_tokens[key] = value * tokens
            per_group_idle_tokens[key] = idle.get(key, 0) * tokens
    joint_values = [value for key, value in per_group_tokens.items()
                    if not cacheable or int(key) in cacheable]
    joint = min(joint_values) if joint_values else None
    slots_per_token = 0.0
    for index in (cacheable or range(len(sizes))):
        tokens = _span_tokens(index, sizes)
        if tokens:
            slots_per_token += 1.0 / tokens
    ceiling = int(size / slots_per_token) if slots_per_token > 0 else None
    free_reported = _attr_int(pool, "get_num_free_blocks")
    if free_reported is None:
        try:
            free_reported = _attr_int(getattr(pool, "free_block_queue", None),
                                      "num_free_blocks")
        except Exception:  # noqa: BLE001, S110 - a missing queue is reported as None
            free_reported = None
    stored_parked = walk["cached_idle"] + walk["referenced_hashed"]
    hash_block = _attr_int(manager, "hash_block_size")
    span_ratio = (sizes[0] // hash_block
                  if (sizes and sizes[0] and _positive_int(hash_block)) else None)
    result = {
        "available": True,
        "pool_blocks": size,
        "free_blocks_reported": free_reported,
        "walked": size,
        "parked_spans_total": stored_parked,
        "parked_spans_per_group": spans,
        "parked_idle_spans_per_group": idle,
        "parked_group_tokens_per_group": per_group_tokens,
        "parked_idle_group_tokens_per_group": per_group_idle_tokens,
        "aggregate_parked_group_tokens": sum(per_group_tokens.values()),
        "joint_parked_tokens_max": joint,
        "referenced_hashed": walk["referenced_hashed"],
        "cached_idle": walk["cached_idle"],
        "free_uncached": walk["free_uncached"],
        "referenced_unhashed": walk["referenced_unhashed"],
        "null_blocks": walk["null_blocks"],
        "group_span_tokens": {str(index): value for index, value
                              in enumerate(sizes) if value},
        "prefix_cacheable_group_ids": cacheable,
        "num_cpu_blocks": _attr_int(manager, "num_cpu_blocks"),
        "slots_per_1k_tokens_one_conversation": (
            round(slots_per_token * 1000.0, 4) if slots_per_token else None),
        "per_conversation_ceiling_tokens": ceiling,
        "free_does_not_mean_empty": (
            "free_blocks_reported counts every parked span: BlockPool.free_blocks "
            "appends HASHED blocks to the free queue as reuse candidates "
            "(block_pool.py:796-798), so free == parked here is the FULL-tier "
            "signature. `cached_idle` is the retrievable-now number; "
            "`free_uncached` is what is genuinely empty"),
        "why_three_answers_differed": (
            "index_entries x hash_block_size counts INDEX KEYS at hash "
            "granularity, not spans (one %d-token span is %s hash units wide, so "
            "that reading under-counts by that factor); total_blocks x "
            "block_size counts every slot at the attention width and ignores that "
            "a conversation costs one slot PER prefix-cacheable group "
            "(manager.py:711/:757 allocate per group), so it over-counts by the "
            "group count; a GiB / summed-page-size division assumes one slot "
            "holds every group at once, which the per-group allocation refutes"
            % (sizes[0] if sizes and sizes[0] else 0, span_ratio)),
        "units": {key: UNITS[key] for key in
                  ("parked_spans_per_group", "parked_group_tokens_per_group",
                   "joint_parked_tokens_max", "aggregate_parked_group_tokens",
                   "num_cpu_blocks", "per_conversation_ceiling_tokens",
                   "slots_per_1k_tokens_one_conversation")},
        "not_established": (
            "joint_parked_tokens_max is an UPPER bound on ONE conversation: the "
            "parked spans of a group may belong to many different conversations, "
            "and the pool keeps no chain identity (reading it would need the hash "
            "value, which this package never reads). The per-conversation truth is "
            "measured, not bounded: see hit_length_tokens_observed_max in the "
            "ledger section"),
        "force": bool(force),
    }
    _depth_cache["mono_ns"] = now
    _depth_cache["result"] = result
    return result


# --- arming ------------------------------------------------------------------

def _bounded_cap(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_CAP
    return max(MIN_CAP, min(MAX_CAP, value))


def arm(cap=None, deep=None, sample_every=None, mirror=None, engine=None) -> dict:
    """Start attributing. Allocates the order ring once; a re-arm at the same cap
    keeps its contents. The two observers are installed here through the
    instance-level disciplines of their own modules, and a refusal to install is
    reported, not hidden.
    """
    report = {"armed": True, "requested_cap": cap, "deep": None,
               "installations": [], "reason": None}
    wanted = _bounded_cap(cap)
    if wanted != int(_state["cap"]) or _state["ring"] is None:
        _state["ring"] = [None] * wanted
        _state["cap"] = wanted
        _state["next"] = 0
        _state["live"] = 0
    if deep is not None:
        _state["deep"] = bool(deep)
    if sample_every is not None:
        value = _int(sample_every) or DEFAULT_SAMPLE_EVERY
        _state["sample_every"] = max(MIN_SAMPLE_EVERY,
                                     min(MAX_SAMPLE_EVERY, value))
    report["deep"] = bool(_state["deep"])
    report["cap"] = wanted
    if mirror is not None:
        path = mirror if isinstance(mirror, str) else os.environ.get(ENV_MIRROR)
        _state["mirror_path"] = path or None
    elif ENV_MIRROR in os.environ:
        _state["mirror_path"] = os.environ[ENV_MIRROR] or None

    manager = None
    if engine is not None:
        from . import guard_mode  # noqa: PLC0415 - cycle, as lookup_ledger does
        manager = guard_mode.manager_of(engine)
    if manager is not None:
        set_manager_ref(manager)
        report["installations"].append(
            {"what": "cpu_coordinator observer",
             **arm_coordinator_observer(manager)})
        report["installations"].append(
            {"what": "cpu pool free-order witness",
             **arm_order_witness(getattr(manager, "cpu_block_pool", None))})
    else:
        report["installations"].append(
            {"what": "observers", "installed": False,
             "reason": "no tier manager in this process: attribution runs from "
                       "the connector values only, so clamp residue and the "
                       "free-order witness stay unavailable"})
    _state["armed"] = True
    report["reason"] = ("attributing every tier lookup into the lookup ledger's "
                        "ring of %d records" % wanted)
    return report


def disarm(engine=None) -> dict:
    """Stop attributing and remove both observers. Keeps the ledger's records."""
    _state["armed"] = False
    manager = manager_ref()
    if engine is not None:
        from . import guard_mode  # noqa: PLC0415 - cycle, see arm()
        manager = guard_mode.manager_of(engine) or manager
    removed = [_unarm_coordinator_observer(manager), _unarm_order_witness()]
    return {"armed": False, "removed": removed}


# --- payloads ----------------------------------------------------------------

def section(preview: int = 0) -> dict:
    """The snapshot's `miss_attribution` block: state, verdicts, bounds, gaps."""
    out: dict = {"development_only": True, **counters()}
    out["verdict_meaning"] = dict(VERDICT_MEANING)
    out["verdicts_possible"] = list(VERDICTS)
    out["undeterminable"] = list(UNDETERMINABLE_VERDICTS)
    out["units"] = dict(UNITS)
    wanted = preview if isinstance(preview, int) and not isinstance(preview, bool) \
        else 0
    wanted = max(0, min(MAX_ORDER_SAMPLES, wanted))
    ring = [entry for entry in (_state["ring"] or []) if entry is not None]
    out["recent_free_orders"] = ring[-wanted:] if wanted else []
    out["warning"] = ("read-only: nothing here changes what the tier offers or "
                     "what the scheduler adopts, and it records nothing until "
                     "armed")
    return out


def rates() -> dict:
    """The only ratios this module will compute, each with its units spelled out.

    A ratio whose two terms have different units is REFUSED, because that is the
    exact error made in production on 2026-10-04 ("24 loads vs 5,712 evictions
    proves write-only churn" divided SCHEDULER STEPS by BLOCKS).
    """
    context = _tier_counters()
    out: dict = {"refused": [], "allowed": {}}
    pairs = (
        ("evictions_total", "stored", "blocks per block: parked blocks discarded "
         "per parked block stored, same unit, meaningful"),
    )
    for left, right, note in pairs:
        a, b = context.get(left), context.get(right)
        if isinstance(a, int) and isinstance(b, int) and b > 0:
            out["allowed"][f"{left}_per_{right}"] = {
                "value": round(a / b, 4), "numerator_unit": UNITS[left],
                "denominator_unit": UNITS[right], "note": note}
    for left, right in (("load_event_counter", "evictions_total"),
                        ("store_event_counter", "evictions_total"),
                        ("evictions_total", "load_event_counter")):
        out["refused"].append(
            f"{left}/{right} refused: {UNITS[left].split(' (')[0]} against "
            f"{UNITS[right].split(' (')[0]} is steps against blocks or blocks "
            "against steps, which is not a rate")
    return out
