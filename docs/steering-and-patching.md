# Steering and runtime patching: reference

Moved from `README.md`. How the store veto (`store_hook.py`), the runtime kill
switch and the three runtime patches coexist. The dated narrative behind the
patches is in `DEVLOG.md`; the operator's runtime kill is in the runbook in
`README.md`.

## Steering and vetoing the tier at runtime (`store_hook.py`)

The offload tier is steered by wrapping one method of the scheduler-side
manager, `vllm/v1/simple_kv_offload/manager.py:932`:

```python
def _classify_store_candidate(self, gpu_block: "KVCacheBlock") -> _StoreAdmission
```

That is the single gate in front of every host-RAM store the tier makes in
**eager** mode; its two call sites (`manager.py:685` for boundary hand-offs and
`manager.py:862` for the per-group loop) both read "not `READY`" as "do not
store this block" and nothing else. The patch is a class-level wrapper installed
on the manager class, so the already-constructed live managers pick it up, and
it reaches the policy through the holder (`holder.decide_store`) on every call -
which is what makes a `POST /policy/reload` change tier behaviour with no
restart. It is installed by the reload verb and by the first snapshot after the
class exists (`register()` runs before the connector imports the manager), it is
idempotent, and a target owned by somebody else is refused with a log line
instead of being overwritten. It never imports `vllm`: if
`vllm.v1.simple_kv_offload.manager` is not imported in this process, nothing is
patched and the snapshot reports why.

The wrapper, in full (`store_hook._make_wrapper`):

```python
def _classify_store_candidate(self, gpu_block):
    _state["calls"] += 1
    verdict = original(self, gpu_block)
    try:
        ready, veto = _admissions(original)
        if ready is None or verdict is not ready:
            # Either the fork renamed the enum (so we dare not override
            # anything) or the engine already declined this block.
            return verdict
        answer = _policy_answer(self, gpu_block, verdict, get_holder())
        if answer is _VETO:
            _state["vetoes"] += 1
            return veto
        if answer is _UNRECOGNIZED:
            _state["unrecognized"] += 1
        return verdict
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        _state["fallbacks"] += 1
        snapshot._error(exc, "store hook")
        return verdict
```

The engine's own function runs first, always: it is the fallback, it is what
keeps the eight decline counters honest, and it is the cheap filter that keeps
the policy off the majority of candidates. `_admissions()` resolves `READY` and
`ALREADY_CACHED` out of the original function's own `__globals__` (memoised on
that dict's identity), because the call sites compare with
`is _StoreAdmission.X` looked up as a module global - a stale enum object would
match nothing and would silently turn every veto into a store.

The contract, in one table:

| `decide_store(ctx)` returns | effect |
|---|---|
| `"veto"`, `"block"`, `"deny"`, `"no-store"` (case/space insensitive) | the block is **not** stored |
| `None`, `""`, `"allow"`, `"store"`, `"ok"`, `"pass"`, `"ready"` | the engine keeps its own verdict |
| anything else - a typo, `True`, `1`, `{}`, a raised exception | the engine keeps its own verdict, and `store_hook.unrecognized` / `store_hook.fallbacks` counts it |

`ctx` is one flat dict of maintained numbers - `gpu_block_id`, `is_null`,
`has_block_hash`, `engine_admission` (always `"READY"`, see below), `mode`,
`target_free_gpu_blocks`, `block_size`, `hash_block_size`, `num_cpu_blocks`,
`cpu_free_blocks`, `in_flight_store_blocks`, `manager_class` - and any field may
be `None` if the fork renamed its source attribute. A block the engine already
declines is never offered to the policy, which keeps the eight
`BoundaryStoreStats` reasons honest and keeps the common path off your code.

### The kill switch (two steps, no restart, nothing new to call)

```bash
# 1. make the veto unconditional - the whole body of decide_store:
#        def decide_store(ctx):
#            return "veto"
#    in kvtier_dev_plugin/policy.py (the copy `status` reported)

# 2. the ONE existing mutating verb:
curl -s -X POST \
     http://127.0.0.1:8000/plugins/kvtier/policy/reload | python3 -m json.tool

# 3. prove it: vetoes rise, `stored` stops moving
curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["store_hook"]); print(d["tier"]["boundary_store_stats"])'
```

New parking stops on the next scheduler decision. A copy already handed to the
workers still completes and is freed normally. A restart keeps the veto only if
the file still says so (the hook cold-loads the policy the moment it takes
ownership of the store path, so a restarted engine does not run the tier
unvetted).

### What a veto really is - read this before trusting it

* `_StoreAdmission` (`manager.py:104-109`) has five members and **none of them
  means "the policy said no"**. A veto is reported to the engine as
  `ALREADY_CACHED`, so the tier's own counter moves `skipped_already_cached` (a
  *skip*, never a `dropped_*`). The real number is `store_hook.vetoes`.
* The **drop-on-full** branch (`manager.py:698-701`) cannot be produced from
  here at all: it is a capacity check the caller makes *after* classification
  returned `READY`, and there is no `CPU_FULL` member to return. A veto is never
  charged to `dropped_cpu_full`.
* A veto is a **skip, not a deferral**. Both call sites move past the block (the
  boundary hand-off is not retried, as the fork itself warns; the eager loop
  advances `num_stored_blocks[g]`), so a vetoed block is never offered again for
  that request. Clearing the veto resumes offloading what is still being
  produced; it does not re-offer what was skipped in between.
* **Lazy mode ignores the hook entirely.** `_prepare_lazy_store_specs`
  (`manager.py:566-617`) walks the GPU free queue and allocates CPU blocks
  directly without classifying, so with `lazy_offload` the kill switch does not
  stop stores. `store_hook.governs_current_mode` is `false` in that case, and it
  is also `false` if a subclass overrides the method.
* Ownership: an attribute marked with another plugin's marker, or a function not
  defined in `manager.py`'s own module, is refused. The one case this cannot see
  is a third party that patched the method in place from inside that module.

`GET /plugins/kvtier/snapshot` answers all of this under `store_hook`:
`patched`, `class`/`method`/`module`, `installs`/`reinstalls`/`skips` (distinct
refusal episodes, not polls) with `skip_reason`, `live_manager_patched` (does
the class of the manager that is actually running carry our marker),
`live_manager_mode`, `live_instance_shadows_method`, `governs_current_mode`,
`ready_resolved` (is the fork's enum still resolvable - if not, the hook vetoes
nothing), `decide_store_present`, `calls`, `policy_calls`, `vetoes`,
`vetoes_since_last_report`, `fallbacks`, `unrecognized`, and the two names that
say where a veto is charged: `veto_member` and
`veto_charged_to_engine_counter`.


### The kill switch (runtime, no restart)

```
curl -s -X POST localhost:8000/plugins/kvtier/tier \
     -d '{"op": "disarm"}'          # or "arm", or "status"
```

Disarmed, the connector is a strict no-op through **four** manager methods:
`get_num_new_matched_tokens` -> `(0, False)`, `build_connector_meta` -> an empty
`SimpleCPUOffloadMetadata`, `has_pending_stores` -> `False`, and
`_queue_finished_eager_store` -> `None` (queue nothing). The worker lives in
another process the HTTP call never reaches, so its no-op is delivered as DATA
(empty store/load lists, which `worker.py:266,289` copy) rather than as a second
patch. A tier that can only be turned off by a 4-minute reboot is not safe to
have on.

**Those four are the whole surface.** `_select_eager_blocks_to_store` - the method
that contains the assert - has exactly two callers in the fork, and both are now
behind a gate:

| route | chain | gate |
|---|---|---|
| 1, per step | `build_connector_meta` (`manager.py:488`) -> `prepare_store_specs` (`:552`) -> selector (`:745`) | the `build_connector_meta` wrapper |
| 2, per finished request | `request_finished_all_groups` (`:1161`) -> `_queue_finished_eager_store` (`:1169`) -> selector (`:1177`) | the `_queue_finished_eager_store` wrapper |

Route 2 had to be gated, and the reason is worth knowing before you trust a
disarm: the scheduler calls `update_state_after_alloc` for **every** scheduled
request (`scheduler.py:1228`, unconditional on `self.connector`), and the fork
registers `_reqs_to_store` *before* that method's `num_external_tokens == 0` early
return. So while the tier was disarmed and unpatched, every finished request still
reached the selector and crashed. The gate sits on the INNER method because
`request_finished_all_groups` returns the `(bool, dict|None)` that drives delayed
freeing and must never lie, while `_queue_finished_eager_store` returns `None`
whether or not it queued anything.

`snapshot.pr56810.disarm_limits` states what it cannot undo: an in-flight copy
already handed to workers finishes, blocks already parked in host RAM stay
parked (re-arming makes them reachable again), and the worker process is never
contacted. The limits about this module's own machinery:

* `enforced_by_wrappers_fitted_at` - the kill switch IS a set of wrappers, so it
  exists only where an install has fitted them. `install()` fits them FIRST, and
  the connector-creation hook runs that install when the tier connector is built,
  so on a normal tier boot `armed: false` is a gate from before the first request.
  What it does NOT cover: a tier connector the factory never builds as its own
  return value (one nested inside a `MultiConnector`, reported as
  `connector_hook.other_connector`) and a worker process, which builds no
  `scheduler_manager` and runs none of these methods.
* `finished_request_store_path` - now `GATED`, per the table above.
* `queued_before_disarm` - a `_pending_finished_stores` entry already queued at
  the instant of disarm stays queued and is drained by the next armed
  `build_connector_meta`.

`patched: false` and `armed: false` are different states. With the gate covering
both routes, `patched: false, armed: false` is a HARMLESS engine (it stores
nothing, by either route); `patched: false, armed: true` is the state that must
never exist, and `set_armed` is what makes it impossible.

### Coexisting with the plugin's other two patches

Three modules patch engine classes at runtime. They never touch the same
attribute, and each carries its own marker tag, so an overlap is detectable
rather than silent:

| Module | Class it patches | Members it owns | Marker tag |
|---|---|---|---|
| `cpu_capacity.py` | the connector BASE class | `get_handshake_metadata` (worker side), `set_xfer_handshake_metadata_pp_aware` (scheduler side); separately marks a block POOL (`POOL_MARKER`) and a manager INSTANCE (`MANAGER_MARKER`) | `kvtier-dev-plugin/capacity-1` |
| `store_hook.py` | `SimpleCPUOffloadScheduler` | `_classify_store_candidate` | `kvtier-dev-plugin/store-1` |
| `backport.py` | `SimpleCPUOffloadScheduler` | `__init__`, `_estimate_lazy_target_blocks`, `update_state_after_alloc`, `_select_eager_blocks_to_store`; plus the kill-switch wrappers `get_num_new_matched_tokens`, `build_connector_meta`, `has_pending_stores`, `_queue_finished_eager_store` | `kvtier-dev-plugin/backport-1` and `.../switch-1` |
| `backport.py` (again) | `KVConnectorFactory` | `create_connector` - one wrapper, on the factory, so the install above lands when the connector is built | `kvtier-dev-plugin/connector-hook-1` |

**There is no overlap to resolve, and that is a property worth keeping.** The
tag constants are deliberately distinct strings, and every member name is
owned by exactly one module. Each installer refuses to clobber a member carrying
a DIFFERENT tag (naming the owner in the refusal), so if a future change ever
picks a name another module already holds, the result is a logged refusal and a
`skipped:` entry - never a silently unwrapped patch. Do not merge the tags.

Notes a future editor needs:

* `backport` uses THREE tags on purpose: `backport-1` for the fingerprint-guarded
  bug fix, `switch-1` for the kill-switch wrappers (no fingerprint - they only add
  a gated early-return around the engine's own function and can never mispatch a
  moved body), and `connector-hook-1` for the factory wrapper. Keeping them apart
  is what lets `status` say "fix refused, switch live" - the state you want during
  a fingerprint drift - and lets a foreign hook on the factory be refused by name.
* `store_hook`'s veto and `backport`'s selector compose by dispatch, not by
  order: the replaced `_select_eager_blocks_to_store` calls
  `self._classify_store_candidate(...)`, which resolves through the instance at
  call time, so it reaches store_hook's wrapper whichever of the two landed
  first. Do not "optimise" that into a direct call.
* `guard_mode` composes with the kill-switch wrappers instead of refusing them
  (its `SIBLING_PACKAGE` rule), and it only ever unwinds a method whose OUTSIDE
  still carries its own marker: fit order either way, an arm/disarm cycle cannot
  remove the kill switch - both orders are pinned by tests
  (`test_the_finish_wrapper_survives_guard_mode_wrapping_on_top`,
  `test_the_switch_fitted_after_guard_mode_composes_and_still_declines`).
* Install ORDER in `register()` is `cpu_capacity.install()` ->
  `backport.install_connector_hook()` -> `store_hook.install()` ->
  `backport.install()`, and the verbs call the three
  `ensure_installed`/`adopt_if_missed` helpers in the same sequence. The hook is
  fitted early and unconditionally because it wraps the FACTORY, which is already
  imported wherever `EngineCore` is; the `backport.install()` inside `install()`
  itself fits the kill-switch wrappers before touching a single method body. That
  inner order IS load-bearing: a refused patch must never leave a route open.

Three different "offs", which are easy to confuse: `store_hook`'s policy veto is
a per-block SKIP inside a live tier; `guard_mode` turns an exception into a
decline (development only, masks bugs); the kill switch stops the connector
scheduling ANY store or load. Only the last one is a kill switch.

