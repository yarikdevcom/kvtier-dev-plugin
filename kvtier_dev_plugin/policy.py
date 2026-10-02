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
    2. `curl -X POST -H "x-kvtier-dev-token: $TOK" \
           http://127.0.0.1:8000/plugins/kvtier/policy/reload`

Then `GET /plugins/kvtier/snapshot` and read `store_hook.vetoes` (rising) and
`tier.boundary_store_stats.stored` (flat). New parking stops immediately; a
copy already handed to the workers still completes and is freed normally.
Restarting the engine keeps the veto only if the file still says so.
"""

# Bump this so `GET /plugins/kvtier/snapshot` visibly proves a reload landed.
VERSION = 2


def decide(view):
    """Look at one KV-tier snapshot and return a JSON-safe verdict.

    `view` keys that exist today (everything is `.get()`-guarded because the
    engine may be running without the offload connector, or with a fork that
    renamed something):

      view["tier"]  -> {"present", "mode", "num_cpu_blocks", "cpu_free_blocks",
                        "pending_stores", "counts", "boundary_store_stats", ...}
      view["scheduler"] -> {"connector", "requests", "running", "waiting"}
      view["epoch"], view["ts"], view["mono"]

    Returns: any JSON-serialisable object. Keep it small.
    """
    tier = view.get("tier") or {}
    counts = tier.get("counts") or {}
    stats = tier.get("boundary_store_stats") or {}

    total = tier.get("num_cpu_blocks") or 0
    free = tier.get("cpu_free_blocks") or 0
    used_ratio = 0.0
    if total > 0:
        used_ratio = round((total - free) / total, 4)

    drops = 0
    for key in ("dropped_cpu_full", "dropped_request_gone", "dropped_null_block",
                "dropped_not_hashed"):
        value = stats.get(key)
        if isinstance(value, int) and value > 0:
            drops += value

    if not tier.get("present"):
        verdict = "no-tier"
    elif used_ratio >= 0.9:
        verdict = "cpu-pool-nearly-full"
    elif drops > 0 and stats.get("stored", 0) == 0:
        verdict = "storing-nothing"
    else:
        verdict = "ok"

    return {
        "version": VERSION,
        "verdict": verdict,
        "cpu_used_ratio": used_ratio,
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

