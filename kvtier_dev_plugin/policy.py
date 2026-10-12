"""The hot-reloadable tier policy. THIS IS THE FILE YOU EDIT.

It is loaded, and re-loaded, by `policy_holder.py` and its `decide()` runs on
the EngineCore busy-loop thread (the thread that runs `step_fn()`,
vllm/v1/engine/core.py:1579). Therefore, in this file:

  * never import torch, CUDA, vllm, numpy or anything else heavy;
  * never sleep, never do file or network I/O, never allocate anything that
    scales with the batch;
  * never raise: the caller catches Exception and reports it as data, but a
    BaseException (SystemExit, KeyboardInterrupt) goes straight into the
    engine loop, and a raise costs you the whole `policy` field for that call;
  * read only. `view` is a plain JSON-safe dict built by `snapshot.py`; the
    engine's own objects are not reachable from here by design.

Reload resolves names at call time through the holder, so an edit to `decide`
here is picked up by the running engine without touching any import site. A
`from kvtier_dev_plugin.policy import decide` binding would NOT be - do not
add one anywhere in this package.

TWO HOOKS, TWO FREQUENCIES
--------------------------
`decide(view)`     -- OBSERVE. Once per `GET /plugins/kvtier/snapshot`, over a
                     JSON-safe view. Returns anything JSON-serialisable. Its
                     contract is unchanged and a live deployment uses it.
`decide_store(ctx)` -- STEER. Once per candidate GPU block on the tier's store
                     path (`manager.py:932`, via the wrapper in
                     `store_hook.py`), so it runs inside scheduler decisions.
                     Returns a VETO WORD. Read the contract below before
                     touching it: the cost model is nothing like `decide`'s.

THE `decide_store` CONTRACT
---------------------------
Input: one small flat `ctx` dict (no nested objects, no engine objects, no
block hashes - a block is identified by `gpu_block_id`):

    gpu_block_id           int | None  the KVCacheBlock id under consideration
    is_null                bool        engine's own first test (always False
                                       here: it would have declined already)
    has_block_hash         bool        ditto
    engine_admission       str | None  the `_StoreAdmission` NAME the engine
                                       itself returned. Always "READY": a block
                                       the engine already declines is never
                                       offered to you, so the eight
                                       BoundaryStoreStats drop reasons stay
                                       honest, and this stays cheap.
    mode                   str | None  "eager" | "lazy". THE VETO ONLY APPLIES
                                       IN "eager": lazy mode
                                       (`manager.py:566-617`) never calls the
                                       classification this hook wraps.
    target_free_gpu_blocks int | None  lazy-mode target, for context
    block_size             int | None  scheduler block size
    hash_block_size        int | None  hash granularity
    num_cpu_blocks         int | None  host-RAM tier capacity, in blocks
    cpu_free_blocks        int | None  free right now (an O(1) maintained count)
    in_flight_store_blocks int | None  blocks already scheduled for a copy
    manager_class          str         e.g. "SimpleCPUOffloadScheduler"

Every field is present but may be `None` if the fork renamed the source
attribute. Guard with `.get()`/`or` and never index into a field blindly.

Return value - interpreted CONSERVATIVELY, so a typo can never cause a store
that the engine would not have made:

    "veto" | "block" | "deny" | "no-store"
        -> do NOT store this block. Case and surrounding space insensitive. The
           engine is told `ALREADY_CACHED`, so the tier's own counter moves
           `skipped_already_cached` (a skip), never a `dropped_*`. The true
           count of your vetoes is `store_hook.vetoes` in the snapshot.
    None | "" | "allow" | "store" | "ok" | "pass" | "ready"
        -> do what the engine would have done.
    ANYTHING ELSE ("YES", True, 1, {}, [], a typo like "veto", a raised
    exception) -> ALSO do what the engine would have done. Strings outside both
    word sets and non-string results are counted as
    `store_hook.unrecognized` in the snapshot, so a broken veto is visible
    rather than silently inert; a raise is counted as `store_hook.fallbacks`.

THE KILL SWITCH (two steps, no restart, no new endpoint):

    1. make `decide_store` always veto - the body below is the whole switch:
           def decide_store(ctx):
               return "veto"
    2. `curl -X POST http://127.0.0.1:8000/plugins/kvtier/policy/reload`
           (no header: since 0.2.68 there is no token on this surface, and since
           0.2.72 no peer-address filter either - the ONLY gate is
           `VLLM_E2_TIER_DEV` being on, which means anything that can reach the
           port reaches these routes, and vLLM's `--api-key` does not cover
           `/plugins/*`.)

Then `GET /plugins/kvtier/snapshot` and read `store_hook.vetoes` (rising). New
parking stops immediately; a copy already handed to the workers still completes
and is freed normally. Restarting the engine keeps the veto only if the file
still says so.

DO NOT PROVE THE KILL SWITCH WITH `tier.boundary_store_stats.stored`: that
dataclass is documented upstream as "cumulative counters for BOUNDARY HAND-OFF
stores" (manager.py:113-123) and only the boundary loop increments it
(manager.py:682-723). The positional eager path (`_select_eager_blocks_to_store`,
manager.py:803-915) parks blocks without touching `published`/`stored` at all, so
`stored` is one store path out of two - flat `stored` does not prove parking
stopped, and "blocks stored" read off it undercounts real parking.
"""

# Bump this so `GET /plugins/kvtier/snapshot` visibly proves a reload landed.
VERSION = 3

# Below this share of the pool pinned by in-flight copies there is nothing to
# report. NOTE: this is PINNED-ness, not occupancy - see `decide`.
NEARLY_FULL_RATIO = 0.9


def _count(value):
    """A real counter, or None. Never a defaulted 0.

    The default IS the bug this helper exists to kill: `tier.get(k) or 0` on a
    key the snapshot does not emit silently turns "I do not know" into
    "0 free blocks", which is the most alarming answer the pool can give. A bool
    is not a count and a float is not a maintained counter, so both are None.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def decide(view):
    """Look at one KV-tier snapshot and return a JSON-safe verdict.

    `view` keys that exist today (everything is `.get()`-guarded because the
    engine may be running without the offload connector, or with a fork that
    renamed something). These are the names `snapshot.tier_view()` ACTUALLY
    emits - verified against the producer by
    `tests/test_policy_input_contract.py`, which fails if a name read here is
    not a name the snapshot writes:

      view["tier"]  -> {"present", "mode", "num_cpu_blocks",
                        "cpu_pool": {"bound", "total_blocks", "free_blocks"},
                        "cpu_hash_index": {"entries", "tokens_upper_bound", ...},
                        "gpu_pool", "pending_stores", "counts",
                        "boundary_store_stats", ...}
      view["scheduler"] -> {"connector", "requests", "running", "waiting"}
      view["epoch"], view["ts"], view["mono"]

    TWO THINGS THIS REPORTS, AND THE TRAP BETWEEN THEM
    --------------------------------------------------
    1. PINNED share: `cpu_pool.free_blocks` is the allocator's free-queue count.
       A parked CPU block is handed BACK to that queue when its copy finishes
       (`snapshot.py:144-167` reads the contract out of block_pool.py:705-741),
       and what keeps it retrievable is its entry in the hash index. So
       `total - free` is "blocks pinned by an in-flight copy right now", NOT
       "how full the arena is". The old code called it `cpu_used_ratio` and
       called the result "cpu-pool-nearly-full"; both names are a lie, and the
       ratio it printed (1.0) was fabricated from a key that does not exist.
       Occupancy from these two counters is impossible - do not reintroduce it.
    2. CAPACITY PRESSURE: `boundary_store_stats.dropped_cpu_full` is the only
       counter that means "a store was refused because the pool could not give a
       block". It is what a "is the tier full?" verdict must be built from.
    Retrievability itself is `cpu_hash_index.entries` / `tokens_upper_bound`,
    reported below for the reader to weigh against the run's token count.

    `cpu_free_blocks` is NOT a tier-view key - it is the name the per-block STORE
    CONTEXT gives the same number (`store_hook.build_store_ctx`). Reading it off
    `tier` used to be this file's behaviour and it produced
    `cpu_used_ratio: 1.0` on every pass, because `or 0` turned a missing key into
    zero free blocks. Numerator and denominator must come from the SAME pool
    object, so both come from `tier["cpu_pool"]`; `tier["num_cpu_blocks"]` is the
    manager's cap and only serves as the denominator fallback.

    Returns: any JSON-serialisable object. Keep it small.
    """
    tier = view.get("tier") or {}
    counts = tier.get("counts") or {}
    stats = tier.get("boundary_store_stats") or {}
    pool = tier.get("cpu_pool") or {}
    index = tier.get("cpu_hash_index") or {}

    total = _count(pool.get("total_blocks"))
    if total is None:
        total = _count(tier.get("num_cpu_blocks"))
    free = _count(pool.get("free_blocks"))

    # A missing input stays missing: no default, no fabricated number. The
    # verdict then says so instead of guessing in either direction.
    missing_inputs = []
    if total is None:
        missing_inputs.append("cpu_pool.total_blocks")
    if free is None:
        missing_inputs.append("cpu_pool.free_blocks")
    pinned = None
    pinned_ratio = None
    if not missing_inputs:
        if total <= 0:
            missing_inputs.append("cpu_pool.total_blocks (zero-capacity pool)")
        elif free > total:
            # More free than the pool has: the two counters disagree, so any
            # ratio computed from them is meaningless rather than optimistic.
            missing_inputs.append("cpu_pool.free_blocks exceeds total_blocks")
        else:
            pinned = total - free
            pinned_ratio = round(pinned / total, 4)

    dropped_cpu_full = _count(stats.get("dropped_cpu_full")) or 0
    drops = 0
    for key in ("dropped_cpu_full", "dropped_request_gone", "dropped_null_block",
                "dropped_not_hashed"):
        value = stats.get(key)
        if isinstance(value, int) and value > 0:
            drops += value

    if not tier.get("present"):
        # No tier at all: its pool keys are legitimately absent, so this wins
        # over the unknown-pinned verdict instead of screaming about it.
        verdict = "no-tier"
    elif dropped_cpu_full > 0:
        # The only true "the arena could not take a store" signal.
        verdict = "cpu-pool-full-dropping-stores"
    elif missing_inputs:
        verdict = "cpu-pool-pinned-unknown"
    elif pinned_ratio >= NEARLY_FULL_RATIO:
        # Nearly every block is PINNED by an in-flight copy. That is a
        # throughput signal, not an occupancy one: parked blocks are not pinned.
        verdict = "cpu-pool-pinned-high"
    elif drops > 0 and (stats.get("stored") or 0) == 0:
        verdict = "storing-nothing"
    else:
        verdict = "ok"

    return {
        "version": VERSION,
        "verdict": verdict,
        # None means "not derivable", and `missing_inputs` names why. It is
        # never a number invented from an absent key.
        "cpu_pinned_ratio": pinned_ratio,
        "cpu_pinned_blocks": pinned,
        "cpu_pool_total_blocks": total,
        "cpu_pool_free_blocks": free,
        # "Free" = unpinned, so these are the retrievability numbers to read
        # against a run's token count; the pool counters cannot answer it.
        "parked_entries": _count(index.get("entries")),
        "parked_tokens_upper_bound": _count(index.get("tokens_upper_bound")),
        "dropped_cpu_full": dropped_cpu_full,
        "missing_inputs": missing_inputs,
        "declined_stores": drops,
        "in_flight_store_blocks": counts.get("in_flight_store_gpu_blocks", 0),
    }


def decide_store(ctx):  # noqa: ARG001 - the contract's one argument, see above
    """Per-block store veto. Return a veto word to stop the tier parking it.

    Called once per candidate GPU block on the store path, on the engine loop
    thread, INSIDE the scheduler's decision. Contract, `ctx` shape and the
    kill-switch procedure are in this module's docstring; the interpretation
    lives in `store_hook.py` and is deliberately conservative: only a listed
    veto word stops a store, and everything else - including an exception -
    leaves the engine's own verdict in place.

    The shipped policy has no opinion. "allow" is a documented no-opinion
    word: do what the engine would have done. To stop the tier, replace the
    body with `return "veto"` and POST /plugins/kvtier/policy/reload.
    """
    return "allow"

