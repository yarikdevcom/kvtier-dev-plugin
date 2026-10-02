# kvtier-dev-plugin

A development window onto a **live** vLLM engine's host-RAM KV-cache tier
(`SimpleCPUOffloadConnector`): read scheduler-side offload state over HTTP while
the engine serves traffic, and hot-reload your own policy module without
restarting the engine.

Out-of-tree. No file in the vLLM tree is edited, nothing is written to the GPU
box, and the engine-side code path never imports `torch`.

```
HTTP (loopback + allowlist)                 EngineCore loop thread (the one that runs step_fn)
GET  /plugins/kvtier/status        local  -> no engine call, answers while the engine is wedged
GET  /plugins/kvtier/snapshot  -- utility -> kvtier_dev_snapshot      (pure reads, contained)
POST /plugins/kvtier/policy/reload -- util-> kvtier_dev_reload_policy (re-reads policy.py)
```

One hop per request: `engine_client.engine_core.call_utility_async(name)`
(`vllm/v1/engine/core_client.py:1199-1214`), which is the same mechanism the
in-tree `/is_sleeping` dev route uses to read scheduler state. Enqueuing a
UTILITY message also wakes an idle engine (the IO thread's `input_queue`
`put_nowait`, `core.py:1904`, is the queue the loop blocks on at `core.py:1562`),
so these routes answer between requests.

## Layout

| File | Role |
|------|------|
| `kvtier_dev_plugin/config.py` | the env vars, the route prefix, the utility method names, and the policy-path validator |
| `kvtier_dev_plugin/engine_side.py` | `vllm.general_plugins` entry: attaches the two engine methods |
| `kvtier_dev_plugin/snapshot.py` | the read side + the containment boundary. Stdlib only |
| `kvtier_dev_plugin/policy_holder.py` | late-bound holder for the policy module; trial-load + swap, rollback on failure |
| `kvtier_dev_plugin/policy.py` | **the file you edit.** `decide(view)` observes, `decide_store(ctx)` vetoes |
| `kvtier_dev_plugin/store_hook.py` | the runtime patch on the tier's store-decision path: idempotent, non-clobbering, fail-open |
| `kvtier_dev_plugin/endpoint.py` | `vllm.endpoint_plugins` entry: the router, the guards, the 504 |
| `tests/` | CPU-only pytest; runs without vllm and without an engine |

## Install

Into the venv that runs the engine, editable so that the file you edit is the
file the engine reloads:

```bash
cd /home/user/Source/experiment/kvtier-dev-plugin
.venv/bin/python -m pip install -e .     # or whichever python runs vllm
.venv/bin/python -m pip check            # must stay clean
```

The package declares **no dependencies**: `fastapi`/`starlette` arrive with
vllm, and a `pip install -e .` must not churn the serving venv's package set.
Verified: installing it into an empty venv adds exactly one package.

Nothing activates yet - see the switches below.

## Switches

| Env var | Meaning |
|---------|---------|
| `VLLM_PLUGINS` | vLLM's own allowlist. **Endpoint plugins never load unless named here.** One name (`kvtier_dev`) covers both halves. |
| `VLLM_E2_TIER_DEV` | this plugin's switch. Unset/`0`/`false`/`off`/`no` = inert: no engine methods attached, HTTP answers 503. |
| `VLLM_E2_TIER_DEV_TOKEN` | required for the only mutating verb, **and for every read from a non-loopback peer**. Unset = reload refused (403) and no non-loopback read is served. Loopback reads do not need it. |
| `VLLM_E2_TIER_DEV_ALLOW` | extra peers beyond loopback: a comma-separated list of addresses or CIDRs. **Unset/empty = loopback only** (the pre-allowlist behaviour). A malformed entry is logged and skipped - never fatal, never an opening. See *Access model*. |
| `VLLM_E2_TIER_DEV_TIMEOUT` | seconds the HTTP layer waits for the engine before answering 504. Default `5.0`, clamped to `[0.05, 60]`. |

`VLLM_PLUGINS` parses as a comma-separated list; `VLLM_PLUGINS=""` is `[""]`
and matches nothing (it is not "unset").

> **Watch out.** `VLLM_PLUGINS` is one allowlist for every group, and
> `vllm.general_plugins` normally loads *everything it finds*. Naming only
> `kvtier_dev` therefore also disables vllm's own two general plugins
> (`lora_filesystem_resolver`, `lora_hf_hub_resolver`, `pyproject.toml:46-48`).
> If you use LoRA resolvers or any other general plugin, name them too:
> `VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver,kvtier_dev`

### Setting them for a systemd service

```bash
sudo systemctl edit vllm.service        # creates a drop-in override
```

```ini
[Service]
Environment=VLLM_PLUGINS=kvtier_dev
Environment=VLLM_E2_TIER_DEV=1
Environment=VLLM_E2_TIER_DEV_TOKEN=put-a-real-token-here
Environment=VLLM_E2_TIER_DEV_TIMEOUT=5
# Only if you need to reach it from other machines (see *Access model*):
Environment=VLLM_E2_TIER_DEV_ALLOW=192.168.2.64,192.168.2.120
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart vllm.service
```

Prefer `EnvironmentFile=/etc/vllm/kvtier-dev.env` (mode 0600) if you do not
want the token in `systemctl show` output. If you launch from a launcher script
instead: `export VLLM_PLUGINS=kvtier_dev VLLM_E2_TIER_DEV=1
VLLM_E2_TIER_DEV_TOKEN=...` before the `vllm serve` line - and note the
launcher is what needs restarting, not just the shell you curl from.

## Verify it is really mounted

```bash
curl -s http://127.0.0.1:8000/plugins/kvtier/status | python3 -m json.tool
```

`status` answers from the API-server process without touching the engine, so it
works even when the engine is idle or wedged, and it prints the resolved policy
path - proof of *which file* a reload would load.

`app.routes` is the check `docs/usage/security.md:347` recommends (a plugin route
can shadow a core route, and there is no conflict enforcement). On the fork's
FastAPI - pinned `>=0.133.0,<0.137.0` in `requirements/common.txt:15`, which
flattens included routers - against a scratch app:

```bash
python3 -c "
import argparse, asyncio
from fastapi import FastAPI
from kvtier_dev_plugin.endpoint import KVTierDevEndpointPlugin
app = FastAPI(); p = KVTierDevEndpointPlugin(); p.attach_router(app)
print(sorted(r.path for r in app.routes if 'kvtier' in getattr(r, 'path', '')))
"
# ['/plugins/kvtier/policy/reload', '/plugins/kvtier/snapshot', '/plugins/kvtier/status']
```

Against the **running** server, the equivalent that does not depend on the
FastAPI route-tree shape (newer FastAPI nests included routers behind an
`_IncludedRouter`) is the OpenAPI surface:

```bash
curl -s http://127.0.0.1:8000/openapi.json | python3 -c \
  'import json,sys; print(sorted(p for p in json.load(sys.stdin)["paths"] if "kvtier" in p))'
# ['/plugins/kvtier/policy/reload', '/plugins/kvtier/snapshot', '/plugins/kvtier/status']
```

and confirm nothing under `/v1` came from us: `GET /v1/models` must behave
exactly as it did before (the plugin adds no `/v1/*` route, and
tests/test_routes.py asserts it never can). (`app.routes` is also fine on the fork's FastAPI, which is
pinned to `>=0.133.0,<0.137.0` and flattens included routers; newer FastAPI
nests them behind an `_IncludedRouter`, so prefer the openapi check there.)

## Using it

```bash
TOK='put-a-real-token-here'
B=http://127.0.0.1:8000/plugins/kvtier

curl -s $B/snapshot | python3 -m json.tool | head -60
curl -s -X POST -H "x-kvtier-dev-token: $TOK" $B/policy/reload | python3 -m json.tool
```

From an allowlisted machine the same three calls work with the token on all of
them (`status` and `snapshot` need it off-box, loopback does not):

```bash
# on the engine's own machine, nothing to configure:
ssh llm curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot | python3 -m json.tool

# from 192.168.2.64 / 192.168.2.120, once VLLM_E2_TIER_DEV_ALLOW names you:
B=http://192.168.2.42:8000/plugins/kvtier
curl -s -H "x-kvtier-dev-token: $TOK" $B/status | python3 -m json.tool
curl -s -H "x-kvtier-dev-token: $TOK" $B/snapshot | python3 -m json.tool
```

`status` is the place to check what you actually opened: `allow_networks` is the
parsed list (an entry that failed to parse is simply absent from it), and
`loopback_only` answers "is this still closed to the LAN?".

`snapshot` returns the CPU pool totals/free, GPU pool totals/free,
`pending_stores`, the ten in-flight/queue lengths, the load/store event counters,
the eight `BoundaryStoreStats` decline reasons, scheduler lengths, the plugin's
per-boot `epoch` (proof that two snapshots came from the same boot), the policy's
own verdict plus `decide_us` and `build_us`, and the `store_hook` patch report
(below) that says whether the runtime veto is actually live.

The edit loop:

1. edit `kvtier_dev_plugin/policy.py` (the copy `status` reported);
2. `POST /plugins/kvtier/policy/reload`;
3. `GET /plugins/kvtier/snapshot` and look at `policy.result`.

Rules for `policy.py`: it runs **on the engine loop thread**. No `torch`, no
CUDA, no I/O, no sleeping, nothing that allocates per token. Return something
JSON-serialisable. `decide()` raising is contained and reported as
`policy.error` - but a `BaseException` (`SystemExit`, `KeyboardInterrupt`) is
deliberately *not* swallowed and will reach the engine loop. The reload itself
costs one `compile()` of a small file (tens of microseconds, reported as
`compile_us`) on that same thread.

`decide()` is the *observing* hook and runs once per snapshot. The steering hook,
`decide_store(ctx)`, runs once per candidate block - see the next section.

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
TOK='put-a-real-token-here'
# 1. make the veto unconditional - the whole body of decide_store:
#        def decide_store(ctx):
#            return "veto"
#    in kvtier_dev_plugin/policy.py (the copy `status` reported)

# 2. the ONE existing mutating verb:
curl -s -X POST -H "x-kvtier-dev-token: $TOK" \
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

## Making the tier safe to enable: the CPU arena capacity clamp (`cpu_capacity.py`)

The defect this exists to close (measured on the live TP2xPP2 layout,
`VLLM_PP_LAYER_PARTITION=25,23` + MTP on the last stage):

* the scheduler sizes its CPU block-id space from ONE aggregate view
  (`manager.py:270-272`, `manager.py:167`), while
* every worker sizes its OWN pinned arena from its OWN stage's bytes/block
  (`worker.py:162`). Here that is 11,468,800 B on PP0 vs 12,902,400 B on PP1, so
  at `--kv-offloading-size 8` the scheduler emits ids up to **187** while the
  last rank's arena holds **166** (always ~13%), and
* `cuda_mem_ops.py:243-249` computes `src_bases[:, None] + ids[None, :] *
  bpb[:, None]` with **no bounds check**: an out-of-range id is a write past the
  end of that rank's pinned host arena.

Dropping an out-of-range copy in the worker is not a fix: a dropped *store* is a
lost park, but a dropped *load* is the model attending to garbage KV, because the
scheduler already counted those tokens as computed. So the requirement is that
the scheduler **never emits** an out-of-range id.

The mechanism is the fork's own PP-aware handshake, which runs inside
`EngineCore.__init__` (`core.py:288-302`) after KV-cache registration and before
any traffic: `collective_rpc("get_kv_connector_handshake_metadata")`
(`abstract.py:214-217`, `gpu_worker.py:868-895`) collects one dict per worker,
they are merged into `content: dict[(pp_rank, tp_rank), metadata]`, and handed to
`kv_connector.set_xfer_handshake_metadata_pp_aware(content)`
(`base.py:716-731`). Three wrappers, installed by the existing
`vllm.general_plugins` entry point:

| side | wrapped | what it does |
|------|---------|--------------|
| worker | `KVConnectorBase_V1.get_handshake_metadata` (`base.py:447-456`) | calls the original, then attaches this rank's `worker_handler.num_cpu_blocks`. A `None` payload becomes a bare carrier; a `dict` payload becomes a **copy** plus our key; anything else is wrapped under a second key and unwrapped again on the scheduler side, so a connector with a typed payload still receives the original object. |
| scheduler | `KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware` (`base.py:716`) | calls the original **first**, with exactly the payload the fork would have received without the plugin (our in-band keys stripped, carriers contributing no entry at all - `base.py:725` raises on a `pp_rank > 0` entry this connector does not support, so an naive carrier would be a boot crash, not a clamp), then reads every reported capacity and clamps. |
| clamp | the manager's CPU `BlockPool` | one-time surgery at handshake time. |

**The clamp is surgery, not a hot-path check.** Ids `>= cap` are withdrawn from
the CPU pool's free structure once (`FreeKVCacheBlockQueue.remove` is O(1) per
node, and the LRU order of the ids that stay is untouched), and they are evicted
from the prefix-cache map first so no hash entry can name a block that can no
longer be allocated. `get_num_free_blocks()` (`block_pool.py:861`) and
`get_new_blocks()` (`block_pool.py:664`) then agree with the cap *by
construction*, every existing tier counter stays honest, and the per-step code
path costs nothing. The mirrors follow: `manager.num_cpu_blocks`
(`manager.py:167`), the pool's own `num_gpu_blocks` (`block_pool.py:173`, which
`get_usage()` and `reset_prefix_cache()` do arithmetic on) and the manager's CPU
`kv_cache_config.num_blocks` (`manager.py:164`, a `replace()`-derived copy -
never the engine's own GPU config). The pool's `blocks` list keeps its full
length on purpose, so `pool.blocks[bid]` for an id the fork already emitted
cannot become an `IndexError`. A tiny **instance-level** wrapper (never
class-level - the same `BlockPool` class backs the GPU pool) counts the
allocations the fork's own guard refuses; it decides nothing.

**Unknown means park nothing.** If no rank reports, a rank reports 0, a rank
reports without a capacity, or the payload is not keyed by `(pp_rank, tp_rank)`,
the cap is 0: the CPU pool keeps only its null block. Every tier allocation site
tests free space *before* allocating (`manager.py:698-701` eager,
`manager.py:586/600` lazy, `manager.py:829/834` the positional scan), so zero
free CPU blocks is a silent no-op step, never an exception, and the engine boots
and serves. Uncapped is never the fallback. As a backstop, the snapshot and
reload verbs call `adopt_if_missed()`: the handshake runs in `__init__` while a
utility handler can only run from the busy loop, so if the handshake never
delivered a ceiling by the first snapshot, the ceiling is invented at 0 and
reported as `adopted_by_snapshot`.

The whole answer is one object in the snapshot, `cpu_capacity`: `installed` and
`reason`, `per_rank` / `per_rank_list` (the arena sizes each rank reported,
keyed `"pp_rank,tp_rank"` because a tuple is not a msgpack key), `chosen_cap`,
`cap_source`, `clamp_applied`, `zero_mode` + `zero_reason`,
`adopted_by_snapshot`, `clamp_failed_open` (we could NOT prove a ceiling - the
field to page on), `cpu_blocks_withheld`, `pool_free_after`,
`manager_num_cpu_blocks_before/after`, `allocations_refused_by_cap`,
`entries_without_capacity`, `original_call_error`, and `live_cpu_pool` read off
the manager that is actually running.

## Access model

Read `docs/usage/security.md:349` first: `--api-key` only covers `/v1`, `/v2`,
`/inference`, `/cohere`, and the engine binds `0.0.0.0`. So:

1. **Prefix.** Everything is under `/plugins/kvtier`. The prefix is baked into
   the `APIRouter`, not added at mount time, so it cannot be dropped, and no
   `/v1/*` path is touched.
2. **Loopback, or an explicit allowlist - decided from the socket.** A peer is
   served only if it is loopback **or** contained in `VLLM_E2_TIER_DEV_ALLOW`:

   ```python
   not_loopback = loopback_denial(scope)          # scope['client'] only
   if not_loopback is None: return None           # loopback: always allowed
   address = _peer_address_unwrapped(scope)       # ::ffff:a.b.c.d -> a.b.c.d
   for network in config.allow_networks():        # VLLM_E2_TIER_DEV_ALLOW
       if address in network: return None
   return f"{not_loopback}, and it is not in the {ENV_ALLOW} allowlist"
   ```

   The whole operator-facing line:

   ```
   VLLM_E2_TIER_DEV_ALLOW=192.168.2.64,192.168.2.120
   ```

   (the proxy that serves real traffic, and the development workstation).
   `192.168.2.0/24` is accepted too - use a network only if you genuinely need
   every address in it, because a CIDR is exactly as wide as it says.

   **Unset or empty means loopback only**, which is the fail-closed default and
   the behaviour before the allowlist existed. Parsing is `ipaddress.ip_network
   (entry, strict=False)`, so a bare address and a CIDR both work. A malformed
   entry (`garbage`, `10.0.0.0/99`, a hostname) is logged as a warning and
   skipped: it never raises into the request path, and it never admits anyone.
   `localhost` is never resolved - name addresses.

   `scope["client"]` is the peer of the TCP connection. Headers are never read
   for this decision, so `X-Forwarded-For: 127.0.0.1` grants nothing. Verified
   both with constructed scopes and against a **real uvicorn socket** driven
   from a source-bound client socket (`tests/test_routes.py::
test_real_socket_*`).
   *Do not* put uvicorn's proxy-headers middleware in front of this: it rewrites
   `scope["client"]` from `X-Forwarded-For` and would hand the forgeability back.
3. **Token for every off-box request.** This is the part that changed when the
   surface became reachable off-box:

   * `POST /policy/reload` always requires `x-kvtier-dev-token` to equal
     `VLLM_E2_TIER_DEV_TOKEN` (`hmac.compare_digest`), loopback included, and is
     refused when that variable is unset.
   * **a read from a non-loopback peer requires that same token too.** A
     loopback read needs none, so `ssh llm curl http://127.0.0.1:8000/...` is
     unchanged.

   **The transport is plaintext HTTP with no TLS.** The token crosses the LAN in
   clear text and is sniffable by anything on that broadcast domain (and by any
   machine that can ARP-spoof it). The allowlist is the compensating control for
   that: it is what limits *who can even see a request enough to try*, so keep
   it as narrow as possible - individual `/32` addresses, not `/16`, and never
   `0.0.0.0/0`. Remove the entries when you are done; unset the variable and the
   surface is loopback-only again. If the token is ever exposed, rotate
   `VLLM_E2_TIER_DEV_TOKEN` and restart.

   Order on every route: prove the peer is allowed -> prove the plugin is
   enabled -> prove the token where it is required. A peer the guard rejects gets
   403 and learns nothing about the plugin's configuration (not even the
   contents of the allowlist, which are deliberately absent from the refusal).
4. **One mutating verb, one file.** The engine-side method takes **no
   arguments**, so nothing on the wire selects a module. The HTTP layer
   separately validates any `path` you send, and refuses anything that does not
   resolve to exactly `<package dir>/policy.py` (`../`, symlinks out of the
   tree, other file names, module names, `NUL`: all rejected). A rejected path
   is a 400 and the engine is never called.
4. **One mutating verb, one file.** The engine-side method takes **no
   arguments**, so nothing on the wire selects a module. The HTTP layer
   separately validates any `path` you send, and refuses anything that does not
   resolve to exactly `<package dir>/policy.py` (`../`, symlinks out of the
   tree, other file names, module names, `NUL`: all rejected). A rejected path
   is a 400 and the engine is never called.

## Hot reload: why it is built this way

`importlib.reload(M)` reuses `M` and updates `M.__dict__` in place, so a caller
that resolves a *module-global* name at call time picks up the new function
while a `from policy import decide` binding does not. Hence the holder: call
sites use `_POLICY.decide(...)`, which looks the function up on the live module
every time. `tests/test_reload.py::test_a_frozen_binding_is_stale_and_the_holder_is_not`
demonstrates both halves of that.

Two deliberate strengthening steps over the plain-`reload` design, both tested:

* **No bytecode cache in the loop.** The import machinery validates a cached
  `.pyc` against the source's `(mtime, size)`, so an edit that keeps the byte
  length and lands within the mtime tick reloads the **old** code and the reload
  looks like a no-op. Both are asserted in `test_reload.py` (`importlib` gets
  fooled by such an edit; the holder does not, because it `compile()`s the bytes
  it just read and writes no `.pyc` at all).
* **Rollback is structural.** A reload compiles into a *new* module object and
  only swaps a reference if that succeeded. A broken file therefore never becomes
  visible: the engine keeps the last good policy and you get
  `{"ok": false, "rolled_back": true, "error": ..., "traceback": ...}` as data.

## Inert, contained, cheap

* Inert: with `VLLM_E2_TIER_DEV` unset, `register()` returns before importing
  anything and every route answers 503 (after the peer check, so a peer that is
  not loopback and not allowlisted learns nothing).
* Contained: every engine-side body is wrapped in `try/except Exception` in this
  fork's idiom (`vllm/v1/core/sched/e2_alloc_log.py:143,151,201`, capped
  `_error()` at `:92`): failures are counted, the first 20 are logged, the log
  writer has its own `except Exception`. `BaseException` is never swallowed.
* No engine-side timeout exists (`future.result()` at `core_client.py:971`), so
  the HTTP layer bounds the call with `asyncio.wait_for` and answers **504**.
  Cancelling that await is safe: when the engine eventually answers,
  `_process_utility_output` pops the future and swallows the resulting
  `InvalidStateError` (`core_client.py:847-865`).
* No writes, no `reset_cache`, no mutation of the tier. The only side effect in
  the whole plugin is swapping the policy module reference.
* The worker's *compute* path is never referenced
  (`simple_kv_offload/worker.py:293-294` records on the live compute stream);
  `tests/test_snapshot.py` asserts the read side never names it and pulls in no
  heavy import. What the plugin does read on the worker side is one plain int,
  `worker_handler.num_cpu_blocks`, exactly once, inside the init-time handshake
  (`cpu_capacity.py`) - never on a per-step path.

## Uninstall / rollback

```bash
# 1. Stop exposing it: drop the two env vars (or set VLLM_E2_TIER_DEV=0) and
#    restart the engine. Routes 503 immediately; with VLLM_PLUGINS dropped from
#    the allowlist the endpoint plugin is not even loaded. To close only the
#    LAN half, unset VLLM_E2_TIER_DEV_ALLOW - the surface goes back to
#    loopback-only.
sudo systemctl daemon-reload && sudo systemctl restart vllm.service

# 2. Remove the package
.venv/bin/python -m pip uninstall -y kvtier-dev-plugin

# 3. Prove it is gone
.venv/bin/python -c "from importlib.metadata import entry_points; \
  print([e.value for g in ('vllm.general_plugins','vllm.endpoint_plugins') \
         for e in entry_points(group=g) if 'kvtier' in e.name])"   # -> []
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/plugins/kvtier/status  # 404
```

An editable install leaves only a `__editable__kvtier_dev_plugin` path entry and
a `.dist-info` in site-packages plus `kvtier_dev_plugin.egg-info` in this
directory; `pip uninstall` removes the former two. The engine never writes
anything anywhere: rolling back is env + uninstall, nothing else.

## Tests

```bash
cd /home/user/Source/experiment/kvtier-dev-plugin
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
```

328 tests, no engine and no vllm required. They cover: loopback accept/reject
(including a spoofed `X-Forwarded-For` from a real remote peer), the allowlist
(bare address, CIDR, IPv6, the `::ffff:` unwrap, unset = loopback-only, malformed
entries that neither raise nor admit, cache invalidation when the env shrinks),
the token on off-box reads *and* its continued absence for loopback reads, the
guard order (peer -> enabled -> token), the `status` payload's new
`allow_networks` / `loopback_only` / `non_loopback_peers_require_token` fields,
the token check and its fail-closed case, path rejection for the reload verb,
holder-based reload actually changing behaviour (same-size/same-mtime edit),
rollback with the traceback as data, containment (a raising policy and a raising
engine both come back as data, `BaseException` propagates), the 504/502/503
answers, and that the mount point is exactly the three `/plugins/kvtier/*`
paths. Three of them (`test_real_socket_*`) drive a real uvicorn listener over a
real TCP connection - one from loopback with no token, two from a source-bound
non-loopback client socket on this host's own LAN address (skipped on a box with
no such address).

The store-policy hook (`tests/test_store_hook.py`, 55 of them) runs against a
stub built by `exec`ing a source string into a real module object, so the
wrapper resolves `_StoreAdmission` out of that module's globals exactly as the
fork's own call sites do. It covers: the stub classifying like `manager.py:932`
before we touch it; a veto stopping a store the engine would have made; a veto
landing in `skipped_already_cached` and *never* in `dropped_cpu_full`; the policy
never being asked about a block the engine already declined; a raising policy and
a raising holder both keeping the engine's verdict while counting a fallback; a
`BaseException` still propagating; every unrecognised verdict (a typo, `True`,
`1`, `{}`) falling through and being counted; a renamed enum making the hook
veto nothing; double install being a no-op that cannot nest wrappers; a foreign
owner (foreign module, or another plugin's marker) being refused; the absence of
the module being reported without importing `vllm`; the `ctx` shape; the
snapshot's `store_hook` section (including `governs_current_mode` going false in
lazy mode); the per-call cost; and the kill-switch procedure end to end through
the two verbs the plugin actually exposes.

The capacity clamp (`tests/test_cpu_capacity.py`, 63 of them) runs against two
stub layers built to the fork's shapes: the connector base class `exec`'d into a
module NAMED `cpu_capacity.BASE_MODULE` (so the ownership check sees the module
the real methods are defined in, and a foreign `__module__` is refused), carrying
`base.py:716-731` including its `pp_rank > 0` `ValueError`; and a
`BlockPool`/`FreeKVCacheBlockQueue`/`KVCacheBlock` transcription with the fake
head/tail sentinels that make "is this block in the queue?" a decidable
question. It covers: min-across-ranks being the cap and a disagreement never
picking the max; the cap never being raised above the scheduler's own view; a
rank reporting nothing, reporting 0, or sending a payload without a capacity all
selecting zero mode instead of uncapped; a bare carrier contributing a capacity
and NO entry (the crash case); the original payload reaching the original method
intact, including a non-dict payload handed back as the identical object; the
original's own exception still propagating with the clamp applied first; double
install, foreign markers, renamed and non-callable targets; the drained pool
never yielding an id at or above the cap while `cap-1` is handed out; the GPU
pool acquiring nothing; a failed surgery reporting `clamp_failed_open` and never
a false `zero_mode`; and adoption.

Against the real fork classes (CPU-only, outside the fork's tree) run
`scratch/test_cpu_capacity_clamp_live.py` from the fork root:

```bash
cd /home/user/Source/experiment/vllm-dev
PYTHONPATH=/home/user/Source/experiment/vllm-testenv:\
/home/user/Source/experiment/kvtier-dev-plugin \
/home/user/Source/experiment/vllm-testenv/.venv/bin/python -m pytest \
    --noconftest -q /home/user/Source/experiment/scratch/test_cpu_capacity_clamp_live.py
# -> 9 passed
```

It builds a real `SimpleCPUOffloadConnector` (both roles) and a real
`SimpleCPUOffloadScheduler` with the fork's own test builders, installs the
wrappers on the real `KVConnectorBase_V1`, and asserts the pool ends up capped,
that no id at or above the cap can leave the real free queue, that the fork's own
consumer did not raise, and that `SimpleCPUOffloadConnector` never grew its own
copy of either method. The fork's own tier suite is unaffected: `84 passed`
(`tests/v1/simple_kv_offload/{test_scheduler,test_kv_events,test_hip_mem_ops}.py`).

## Known limits

* `snapshot` reads the **scheduler process** view (one EngineCore per TP2xPP2
  layout). Per-rank arenas arrive only as the handshake number each worker
  reports (`cpu_capacity.per_rank`); nothing else on the worker side is read,
  and no per-step worker path is touched.
* The clamp wraps the connector **base** class, so a connector that overrides a
  handshake method shadows that half of it: `MultiConnector` overrides both
  (`multi_connector.py:482`, `:503`) and `NixlConnector` the consumer
  (`nixl/connector.py:214`). A tier nested in a `MultiConnector` still gets the
  clamp - the forwarder at `multi_connector.py:503-507` calls each
  sub-connector's own method, which is ours by inheritance - but its worker half
  reports only the FIRST non-`None` sub-connector payload
  (`multi_connector.py:482-491`), so if that is not the offload connector's
  carrier no capacity arrives and the tier lands in fail-safe zero. Safe,
  degraded, visible in `zero_reason`. `adopt_if_missed` cannot see a nested
  manager at all (`scheduler_manager` is not on a `MultiConnector`), so it
  declines rather than guessing.
* `clamp_failed_open: true` is the state to read as "not safe": it means a
  manager we recognised could not be clamped, so the tier keeps the aggregate
  view's ids. Disable the tier if you see it. A `zero_mode: true` with
  `clamp_applied: true` is the *safe* degraded state.
* Fields are read with `getattr(..., None)`. If the fork renames one, the
  snapshot reports `null` (and names missing `BoundaryStoreStats` fields under
  `missing_fields`) rather than raising. Re-check `snapshot.py`'s header list
  against the fork after a rebase.
* A reload that lands while a request is in flight does not interrupt it, but it
  does briefly occupy the engine loop thread for one `compile()`.
* The store-policy hook governs the **eager** store path only. In `lazy_offload`
  mode (`manager.py:566-617`) the tier never classifies a candidate, so no veto
  can reach it and the kill switch does not stop stores;
  `store_hook.governs_current_mode` reports `false` rather than pretending.
* A veto cannot be reported as its own decline reason, because
  `_StoreAdmission` has no member for it: it is charged to
  `skipped_already_cached`, and the `dropped_cpu_full` branch is unreachable
  from the patch point by construction. See *What a veto really is*.
* Vetoed blocks are skipped for good within their request (both call sites move
  past them), so flipping the veto off does not re-offer them.
* `502` vs `504` on a route: `504` means the engine never answered within
  `VLLM_E2_TIER_DEV_TIMEOUT`; `502` means the engine answered with a failure.
  Neither cancels anything in the engine, and neither retries.
