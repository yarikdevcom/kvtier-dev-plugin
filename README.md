# kvtier_dev_plugin

A development window onto a **live** vLLM engine's host-RAM KV-cache tier
(`SimpleCPUOffloadConnector`): read scheduler-side offload state over HTTP while
the engine serves traffic, ask about one block or one request, trace the steps
around a decision, hot-reload your own policy module without restarting the
engine, and - in development only - keep the engine up while a tier bug is still
in it.

Out-of-tree. No file in the vLLM tree is edited, nothing is written to the GPU
box, and the engine-side code path never imports `torch`.

```
HTTP (loopback + allowlist)                 EngineCore loop thread (the one that runs step_fn)
GET  /plugins/kvtier/status        local  -> no engine call, answers while the engine is wedged
GET  /plugins/kvtier/snapshot  -- utility -> kvtier_dev_snapshot      (pure reads, contained)
GET  /plugins/kvtier/kvconfig  -- utility -> kvtier_dev_kvconfig      (geometry + divisibility)
GET  /plugins/kvtier/block?…   -- utility -> kvtier_dev_inspect       (one block, one group)
GET  /plugins/kvtier/request?… -- utility -> kvtier_dev_inspect       (one request, per group)
POST /plugins/kvtier/trace     -- utility -> kvtier_dev_trace         (arm/disarm/read the ring)
POST /plugins/kvtier/guard     -- utility -> kvtier_dev_guard         (DEV ONLY: catch + decline)
GET|POST /plugins/kvtier/tier  -- utility -> kvtier_dev_tier          (kill switch + PR #56810 status)
POST /plugins/kvtier/policy/reload -- util-> kvtier_dev_reload_policy (re-reads policy.py)

Worker process (per rank)        no utility channel exists, so a worker reports by file:
VLLM_E2_TIER_DIAG=…              -> rank-PP<n>_TP<m>.json, read by the proxy host
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
| `kvtier_dev_plugin/engine_side.py` | `vllm.general_plugins` entry: attaches the six engine methods, wires the logger and the worker-writer retry |
| `kvtier_dev_plugin/snapshot.py` | the read side + the containment boundary, and the `guard`/`trace`/`worker_diag` sections of the snapshot. Stdlib only |
| `kvtier_dev_plugin/kvconfig.py` | `kvconfig`: the KV-cache geometry and the divisibility verdict the tier's first store depends on |
| `kvtier_dev_plugin/kvinspect.py` | `inspect`: one block, or one request, per group, with every id list capped |
| `kvtier_dev_plugin/trace_ring.py` | the bounded event ring and the `trace` verb (arming installs the observers, disarming removes them) |
| `kvtier_dev_plugin/step_trace.py` | the two scheduler-step observers the ring reads from. Pure observers, no catching |
| `kvtier_dev_plugin/guard_mode.py` | **development only**: the tier exception capture-and-decline, its captures, and the `guard` verb |
| `kvtier_dev_plugin/worker_diag.py` | the worker-side per-rank JSON writer (files are how a worker reports; it has no utility channel) |
| `kvtier_dev_plugin/policy_holder.py` | late-bound holder for the policy module; trial-load + swap, rollback on failure |
| `kvtier_dev_plugin/policy.py` | **the file you edit.** `decide(view)` observes, `decide_store(ctx)` vetoes |
| `kvtier_dev_plugin/store_hook.py` | the runtime patch on the tier's store-decision path: idempotent, non-clobbering, fail-open |
| `kvtier_dev_plugin/backport.py` | the runtime backport of upstream PR #56810 (hybrid-geometry fix) plus the tier's runtime kill switch. Stdlib only |
| `kvtier_dev_plugin/pr56810_sources.py` | the patched function bodies, byte-exact, with the SHA-256 of the source each one replaces. Generated |
| `kvtier_dev_plugin/endpoint.py` | `vllm.endpoint_plugins` entry: the router, the guards, the argument specs, the 504 |
| `tests/` | CPU-only pytest; runs without vllm and without an engine (`fakes_debug.py` holds the shared fake engine structures) |

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
| `VLLM_E2_TIER_DEV_TOKEN` | required for the mutating verbs (`policy/reload`, and every shape of `trace` and `guard`), **and for every read from a non-loopback peer**. Unset = those requests refused (403) and no non-loopback read is served. Loopback reads do not need it - except `trace`/`guard`, which are token-gated as a whole because they can change engine dev state. |
| `VLLM_E2_TIER_DEV_ALLOW` | extra peers beyond loopback: a comma-separated list of addresses or CIDRs. **Unset/empty = loopback only** (the pre-allowlist behaviour). A malformed entry is logged and skipped - never fatal, never an opening. See *Access model*. |
| `VLLM_E2_TIER_DEV_TIMEOUT` | seconds the HTTP layer waits for the engine before answering 504. Default `5.0`, clamped to `[0.05, 60]`. |
| `VLLM_E2_TIER_DIAG` | the worker diagnostics directory (capability 5). **Unset = the worker writer is completely inert: no directory created, no file written, nothing wrapped in the worker.** Set it to a path both the worker ranks and the proxy host's reader can see. |
| `VLLM_E2_TIER_DIAG_INTERVAL` | how often a rank rewrites its file, in engine steps. Default `50`, clamped to `[1, 100000]`. A step that *changed* something writes immediately (subject to the floor below). |
| `VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS` | the rate floor between writes, in seconds. Default `1.0`, clamped to `[0, 3600]`; `0` means "no floor". |
| `VLLM_E2_TIER_DIAG_EPOCH` | pin the per-boot `epoch` string the reader uses to prove two facts came from the same boot. Unset = derived once per process and stable for its lifetime. |
| `VLLM_E2_TIER_DIAG_RANK` | pin this rank's id (`PP<n>_TP<m>`) instead of deriving it from the already-imported `vllm.distributed.parallel_state`. An illegal value means **no file at all**, because the reader refuses a file whose `rank` does not match its own name. |

There is deliberately **no env var for guard mode**. It can mask real bugs, so it
is off at every boot and reachable only through `POST /plugins/kvtier/guard` with
the token - see *Guard mode*.

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
# Worker diagnostics: leave this out and the worker writer never runs at all.
Environment=VLLM_E2_TIER_DIAG=/tmp/kv-tier-diagnostics
Environment=VLLM_E2_TIER_DIAG_INTERVAL=50
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
# ['/plugins/kvtier/block', '/plugins/kvtier/guard', '/plugins/kvtier/kvconfig',
#  '/plugins/kvtier/policy/reload', '/plugins/kvtier/request',
#  '/plugins/kvtier/snapshot', '/plugins/kvtier/status', '/plugins/kvtier/trace']
```

Against the **running** server, the equivalent that does not depend on the
FastAPI route-tree shape (newer FastAPI nests included routers behind an
`_IncludedRouter`) is the OpenAPI surface:

```bash
curl -s http://127.0.0.1:8000/openapi.json | python3 -c \
  'import json,sys; print(sorted(p for p in json.load(sys.stdin)["paths"] if "kvtier" in p))'
# ['/plugins/kvtier/block', '/plugins/kvtier/guard', '/plugins/kvtier/kvconfig',
#  '/plugins/kvtier/policy/reload', '/plugins/kvtier/request',
#  '/plugins/kvtier/snapshot', '/plugins/kvtier/status', '/plugins/kvtier/trace']
```

and confirm nothing under `/v1` came from us: `GET /v1/models` must behave
exactly as it did before (the plugin adds no `/v1/*` route, and
tests/test_routes.py asserts it never can). (`app.routes` is also fine on the fork's FastAPI, which is
pinned to `>=0.133.0,<0.137.0` and flattens included routers; newer FastAPI
nests them behind an `_IncludedRouter`, so prefer the openapi check there.)

## Using it

The whole surface, all under `/plugins/kvtier`:

| Route | Methods | Token | What it does |
|-------|---------|-------|--------------|
| `/status` | GET | off-box only | local facts; makes **no** engine call, so it answers while the engine is wedged |
| `/snapshot` | GET, POST | off-box only | the scheduler-side tier view (pools, counters, policy, `store_hook`, `guard`, `trace`) |
| `/kvconfig` | GET, POST | off-box only | the KV-cache geometry: per-group block sizes, spec classes, prefix-cacheability, and every group's divisibility verdict |
| `/block` | GET, POST | off-box only | one GPU block by id (and group), plus the tier's CPU slot of the same id |
| `/request` | GET, POST | off-box only | one request by id: its blocks per group and what the tier could park |
| `/trace` | GET, POST | **always** | the bounded event ring: read it, or arm / disarm / clear it |
| `/guard` | GET, POST | **always** | guard mode: state, captured faults, arm, disarm |
| `/policy/reload` | POST | **always** | reload `<package dir>/policy.py`, and adopt the store-policy patch |

**Arguments.** GET takes them as query parameters, POST as a JSON object body, and
both are validated against the SAME per-route table (`endpoint.ARG_SPECS`), so a
bound cannot be evaded by changing the method. An unrecognised key is a `400` that
never reaches the engine - including a stray `?cachebuster=1` on `/snapshot`. A
validated body is forwarded to the engine as ONE dict argument
(`call_utility_async(method, args)` -> `getattr(self, name)(*args)`,
`vllm/v1/engine/core.py:1669-1671`); a call with no arguments still sends none, so
nothing that worked before arguments changed. Which engine method a request reaches
is fixed by the route and can never come from the body.

```bash
TOK='put-a-real-token-here'
B=http://127.0.0.1:8000/plugins/kvtier

curl -s $B/snapshot | python3 -m json.tool | head -60
curl -s $B/kvconfig | python3 -m json.tool
curl -s "$B/block?block_id=412&group=5" | python3 -m json.tool
curl -s "$B/request?request_id=cmpl-a" | python3 -m json.tool
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"request_id":"cmpl-a","ids_cap":200}' \
     $B/request | python3 -m json.tool

# the ring: arm it, work, read it, disarm it (nothing is wrapped while disarmed)
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"arm","depth":256}' $B/trace
curl -s -H "x-kvtier-dev-token: $TOK" "$B/trace?limit=40" | python3 -m json.tool
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"disarm"}' $B/trace

# the one mutating verb of the old surface
curl -s -X POST -H "x-kvtier-dev-token: $TOK" $B/policy/reload | python3 -m json.tool
```

From an allowlisted machine the same calls work with the token on all of them
(`status` and `snapshot` need it off-box, loopback does not):

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
`loopback_only` answers "is this still closed to the LAN?". It also answers
"is this engine in a development mode?": `guard_mode` and `trace_armed` at the top
level, the same facts again in `dev_state` with their `source`, `epoch` and
`age_seconds`, and `worker_diagnostics` with the directory the worker writer was
told to use. Because `/status` never calls the engine, `guard_mode` is `null` until
this API process has seen an engine answer that says otherwise - it is never a
guessed `false`.

`snapshot` returns the CPU pool totals/free, GPU pool totals/free,
`pending_stores`, the ten in-flight/queue lengths, the load/store event counters,
the eight `BoundaryStoreStats` decline reasons, scheduler lengths, the plugin's
per-boot `epoch` (proof that two snapshots came from the same boot), the policy's
own verdict plus `decide_us` and `build_us`, the `store_hook` patch report (below)
that says whether the runtime veto is actually live, the `cpu_capacity` clamp
report, and the development sections - `guard_mode` (a top-level copy of the answer,
because a guarded engine must never be mistaken for a production one), `guard` (its
bounded previews), `trace` (the ring's counters and the step observer's), and
`worker_diag` (what this process knows about the per-rank file writer, with
`worker_scope` saying that the truth lives in the files).

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

## Reading the geometry before the first store (`kvconfig.py`)

One line of the fork decides whether the tier can store at all:

```python
assert block_size % hash_block_size == 0     # vllm/v1/core/kv_cache_utils.py:2888
```

and the tier reaches it through
`resolve_block_hashes(request.block_hashes, hash_block_size, group_block_size)`
(`vllm/v1/simple_kv_offload/manager.py`) on the **first store for that group** -
long after boot, far from the cause. `GET /plugins/kvtier/kvconfig` answers it as
data, per group, before anything is stored:

* `geometry.num_blocks`, `num_groups`, `kv_cache_layout`, and one entry per group:
  `block_size`, `spec_class`, `page_size_bytes`, `prefix_cacheable`,
  `is_circular_buffer`, `layer_count`/`layer_names` (capped, with
  `layer_names_dropped`), the group's `role`/`host_resident`/`enable_kv_transfer`,
  and the spec's own attributes (`spec`), reported only when the object carries
  them - so a Mamba group does not grow a wall of nulls and the ring group reports
  what makes it a ring.
* `geometry.groups[i].divisibility`: `divisible`, `remainder`, `scale_factor`,
  `would_fail`, plus the `rule` text. An operand that is missing or not an int
  gives `None` and a `reason`, never a misleading `true`: **"unknown" and "safe"
  must not look the same.**
* `geometry.divisibility_summary`: `groups_failing`, `groups_unknown`,
  `all_divisible`, `any_fail` - the yes/no an operator actually wants.
* `geometry.transfer_group_ids` / `prefix_cacheable_group_ids` /
  `scratch_group_ids`: the classification the tier needs (group index 4/5 in the
  live layout is the QSA raw-key ring - `CircularBufferSpec`, one block per
  request, `prefix_cacheable=False`, and recognised by CLASS NAME through the MRO
  because `isinstance` would need the import this plugin must not do).
* `tier_crosscheck`: the SAME geometry from the tier manager's side
  (`group_block_sizes`, `hash_block_size`, `fa_block_size`, `fa_gidx`), the
  per-group `agree` verdict, `disagreements`, and each group's divisibility
  recomputed against the TIER's operands. A spec whose `block_size` disagrees with
  the manager that allocates for it is reported as `manager_block_size_mismatch`
  - that one is a corruption, not a crash.
* `scheduler.managers`: `coordinator.single_type_managers` keyed by each manager's
  own `kv_cache_group_id` (with its position), because the tier's `group_block_sizes`
  and this list are positional and a silent misalignment would make every number
  above a lie.

It takes no arguments, reads only, and is bounded (`MAX_GROUPS = 64`, 32 layer
names per group); what it dropped is always counted rather than hidden.

## Asking about one block or one request (`kvinspect.py`)

A snapshot gives counters. These two questions are about one object:

```bash
curl -s "$B/block?block_id=412&group=5"      # ref_cnt, in_free_queue, prev/next free
                                             # ids, the decoded block hash
                                             # (hex + the group packed in its last
                                             # 4 bytes, kv_cache_utils.py:86-92),
                                             # in_cached_map via the same O(1)
                                             # lookup the tier uses (manager.py:940),
                                             # the group's geometry, and the tier's
                                             # CPU slot of the same id
curl -s "$B/request?request_id=cmpl-a"       # the request's counters, then per group:
                                             # block_ids (capped), divisibility,
                                             # hash_scale_factor, coverable_blocks
                                             # and the tier's own readiness terms
```

`ids_cap` bounds every id list (default 64, max 1024) and each list says how many
it dropped. A block id outside the pool, a group index that does not exist, a
request that has already finished: each answers with a `reason` that names the
range or the ids that DO exist.

`coverable_blocks` and `tier_view.ready_blocks` are the fork's arithmetic, not an
invention (`manager.py:820-831`):

```
confirmed = num_computed_tokens - num_output_placeholders
alignment = hash_block_size if enable_partial_hash_hits else block_size
aligned   = confirmed // alignment * alignment
ready     = min(len(group_gpu_block_ids), aligned // group_block_size)
```

plus one extra cap the fork's scheduler cannot see from inside: the number of
block hashes it can actually view at that group's size (`block_hashes //
(block_size // hash_block_size)`). The answer names `binding_term`, i.e. which of
those limits bound it - and a group whose division fails reports
`coverable_blocks: 0`, because the fork would assert before storing anything.

## The trace ring (`trace_ring.py` + `step_trace.py`)

An opt-in ring of scheduler/tier events, for the ten seconds around a decision you
cannot reproduce.

* **Disarmed it costs nothing**, and not because a branch is cheap: nothing is
  wrapped at all. `record()` returns on one test and there is no ring to write
  into; arming is what installs the observers, disarming removes them.
* **Armed, it is preallocated and fixed.** `arm(depth)` builds `depth` slots once
  (`512` default, clamped to `[16, 4096]`) and each record is a seven-field tuple
  `(seq, mono_ns, kind, group, n1, n2, text)` written at a cursor. No append, no
  growth, no lock, and `overwritten` counts what the wrap ate.
* **Reads are bounded.** `read(limit, after_seq)` walks backwards from the cursor
  (cost O(returned), never O(depth)), newest last, `limit` capped at 512. An
  incremental poller that fell behind gets `gap_records` instead of an illusion of
  contiguous history; `clear()` empties the ring but never the sequence, so a
  reader's cursor stays valid across a clear.
* **What lands in it:** `sched:step` (step number, scheduled tokens,
  running/waiting/requests) and `sched:output` from the two wrapped scheduler
  methods; `step:build_meta`, `step:store_specs`, `tier:load_offer`,
  `tier:load_accepted`, `tier:finish`, `tier:completion_report` and `tier:reset`
  from the tier's entry points; and `exception` for any guard capture. Each record
  is one dict of the seven slot fields plus its `age_ms`.
* The observers are pure: they call the fork's method first and return its value
  untouched, they catch nothing on the serving path, and a record that cannot be
  built is counted as a fallback. Installing and arming are separate decisions - an
  installed observer records nothing while the ring is disarmed.

`/trace` and `/guard` require the token in every shape, loopback included: a route
that can change engine dev state is token-gated as a whole, not per field.

## Guard mode: develop against a live engine without an outage (`guard_mode.py`)

> **DEVELOPMENT ONLY. Disarm it before any production boot.** Armed, a tier
> exception becomes a record plus a safe decline, which is exactly how you keep a
> live engine up while debugging - and also how a real bug gets masked: a declined
> store is silent, and a decline can leave tier state half-advanced. It has **no
> env var** (it is off at every boot), it is armed only by a tokened `POST
> /plugins/kvtier/guard`, and its state is echoed as `guard_mode` by `status` and
> by every verb payload.

```bash
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"armed":true}'  $B/guard
curl -s -H      "x-kvtier-dev-token: $TOK"        "$B/guard" | python3 -m json.tool
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"captures","limit":2}' $B/guard
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"armed":false}' $B/guard   # restore
```

Armed, the tier's entry points return the fork's own "nothing happened" value
instead of faulting - never an invented one:

| Method | Declined with | Source |
|--------|---------------|--------|
| `get_num_new_matched_tokens` | `(0, False)` | `manager.py:320` returns the same shape when it finds nothing |
| `update_state_after_alloc` | `None` | the method's normal fall-through |
| `build_connector_meta` | a fresh empty `SimpleCPUOffloadMetadata` (resolved from the original's own `__globals__`; if the fork renamed it there is **no** decline and the exception continues) | `metadata.py:16-38` |
| `prepare_store_specs` | `([], [], [], None)`, fresh lists per call | nothing to park this step |
| `request_finished`, `request_finished_all_groups` | `(False, None)` | `manager.py:1152` returns exactly this anyway |
| `take_events` | `()` | no events |
| `reset` | `False` ("the cache was not reset") | `base.py:748-760` documents `bool` |
| `update_connector_output` | **no decline: capture and re-raise** | a fake value here would corrupt state the scheduler trusts |

What a capture holds: the method, the exception type and a clipped message, a
bounded traceback, the arguments as data (a `Request` as its id and counters, a
`KVCacheBlocks` as per-group id heads), and the tier's own `manager_context` -
including `geometry.groups_failing_divisibility`, i.e. the arithmetic of the fork's
assert at the instant it failed. Captures live in a `deque(maxlen=8)` (overflow
counted, full text only via the `captures` op); the per-poll `snapshot` carries
bounded previews instead.

Two properties are absolute: `except Exception`, **never** `except BaseException`
(a `KeyboardInterrupt`, `SystemExit` or `asyncio.CancelledError` inside a tier call
is the engine's signal and always continues), and disarming restores the fork's own
functions by identity. If the trace ring is armed, disarming guard mode leaves the
wrappers in place - tracing still needs them, and traces capture-and-re-raise.

A wrapper installed by **this package's own** other module (today: the PR #56810
backport, which wraps `get_num_new_matched_tokens` and `build_connector_meta` too)
is composed with rather than refused: the guard goes on top of it, the install
report names those methods under `nested`, and uninstalling hands the sibling's
wrapper back exactly. Refusing it would have left the two most fault-prone entry
points unguarded while `arm` still reported "exceptions are captured and declined" -
a payload worse than a refusal, because it is confidently wrong. Anything owned by
anyone else is still refused outright, marker or no marker.

## Worker-side diagnostics (`worker_diag.py`)

A worker has no request path: `general_plugins` loads in the worker process, a
worker never imports `vllm.v1.engine.core`, and the UTILITY channel is
EngineCore-only. So a worker reports through a **file**: one per rank, into
`VLLM_E2_TIER_DIAG`, named `rank-PP<n>_TP<m>.json` for the proxy host's reader
(`kv_tier_diagnostics.py`) to pick up.

* **Inert by default.** With `VLLM_E2_TIER_DIAG` unset nothing is created, nothing
  is wrapped, and `install()` says the env var is unset. `engine_side.register()`
  normally runs before the connector imports its worker module, so it registers
  `worker_diag.ensure_installed` with the capacity handshake, which fires once this
  rank really has an offload handler.
* **Budgeted.** A write happens when either the step cadence
  (`VLLM_E2_TIER_DIAG_INTERVAL`, default 50) or a change in the handler's signature
  says so - and never more often than `VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS`
  (default 1 s). The first step always writes: "no file" must not be
  indistinguishable from "the tier is not running on this rank".
* **Atomic.** A temp file in the same directory, then `os.replace`; a document over
  32 KiB is trimmed (prose first, then the optional group list) and if it still does
  not fit the write is refused and counted. Nothing is ever truncated in place.
* **The fork's return values are untouched.** `bind_connector_metadata` counts the
  blocks this rank was handed; `get_finished` writes after the fork's own work, and
  both wrappers are contained, so a full disk cannot fail a step.
* **The schema is the reader's contract**: `version` 1, a per-boot `epoch`, `ts`,
  `role: "worker"`, `rank` matching the file name (the reader calls a mismatch
  `malformed`), `arena.bytes_total`/`bytes_used`/`num_blocks` (plus
  `block_bytes_by_group` = `len(layer_names) * page_size_bytes` per group),
  `counters`, and `writer` prose that states where each number came from.
  `arena.bytes_used` is a **proven floor** (blocks in flight x bytes per block), not
  the pool's occupancy, and says so; `counters.dropped_cpu_full` and
  `counters.boundary_stored` are deliberately absent because they are scheduler-side
  (`BoundaryStoreStats`) - the reader notes an absent counter and adds zero, which
  keeps the aggregate honest instead of inventing a reading.

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

## Making the tier bootable at all: the PR #56810 backport (`backport.py`)

The tier cannot be enabled on our model without this fix, so it is installed on
the way through every verb, not behind a switch.

Upstream issue #56396 / PR #56810 (merged 2026-09-20, NOT in the `v0.30.0` tag
this fork is on). `SimpleCPUOffloadScheduler` assumed every KV cache group's
block size is a multiple of the hash block size. Core only validates that over
the PREFIX-CACHEABLE groups (`kv_cache_utils.py:768-781`), so a hybrid model
dies in the store path:

```
manager.py:842 _select_eager_blocks_to_store -> resolve_block_hashes(hashes, 3200, 8)
kv_cache_utils.py:2888:  assert block_size % hash_block_size == 0
```

Our model is five `block_size=3200` groups plus the QSA raw-key ring, a
`CircularBufferSpec` of block size **8**. The ring's size is computed
(`qsa_cache.py:900-901`: `compress_ratio * cdiv(compress_ratio +
num_speculative_tokens, compress_ratio)` = `4 * cdiv(8, 4) = 8`), so it is not a
knob we can turn. It reaches the assert because the loop's only filter was
`has_positionally_stable_blocks`, which `CircularBufferManager` inherits as True
(unlike `MambaManager`).

**`--prefix-match-unit 4` or `8` is a dead end, and must not be re-proposed.**
It would make the sizes divisible by lowering `hash_block_size`, but (1) a
ring block is positionally UNSTABLE - a circular buffer reuses slots - so
hashing it in 8-token units creates prefix-cache entries that can hit against
stale ring content; the group is `prefix_cacheable=False` because its content
must never be matched, and the assert is telling us something true; and (2) it
multiplies prefix hashing by ~400x (every 8 tokens instead of every 3200, per
group, per step) for entries that can never be hit.

Four methods change, all inside `SimpleCPUOffloadScheduler`: record
`prefix_cacheable_group_ids`; iterate `prefix_cacheable_groups` when estimating
the lazy target; give a non-cacheable group `cpu_hit_blocks.append([])`; and
`continue` past a non-cacheable group in the store selector, placed BEFORE the
`len(gpu_block_ids) >= num_free` break so a scratch group can never consume CPU
blocks or emit `BlockStored`.

### Two guards, because a wrong patch here corrupts output silently

**Source fingerprints.** For each method the plugin stores the SHA-256 of
`inspect.getsource()` at fork HEAD `10b8e07`. At install time a mismatch means
the function moved: that method is REFUSED (reported per-method as
`patched: false` with the reason), never patched "close enough". Three methods
are replaced by a byte-identical copy of upstream's body exec'd in the
ORIGINAL function's `__globals__` (a wrapper cannot inject a `continue` into a
loop it cannot see); `__init__` is WRAPPED because its change is purely
additive and provably identical to setting the attribute after the original
returns.

**The geometry guard.** `manager.py:456-458` silently skips any group with no
parked blocks, so a *blanket* "skip whatever trips the assert" policy is an
output-corruption machine: drop a genuinely cacheable group and its KV is
simply missing on resume, with no error. `geometry_guard()` recomputes the set
the patch would skip from the live `KVCacheConfig` and REFUSES to install if any
of those groups reports `prefix_cacheable=True`, naming the group. If the patch
is already live on a class and a LATER config fails the guard, the tier is
auto-disarmed rather than left running (`auto_disarmed: true`).

### The kill switch (runtime, no restart)

```
curl -s -X POST localhost:8000/plugins/kvtier/tier -H "authorization: Bearer $TOK" \
     -d '{"op": "disarm"}'          # or "arm", or "status"
```

Disarmed, the connector is a strict no-op: `get_num_new_matched_tokens` ->
`(0, False)`, `build_connector_meta` -> an empty `SimpleCPUOffloadMetadata`,
`has_pending_stores` -> `False`. The worker lives in another process the HTTP
call never reaches, so its no-op is delivered as DATA (empty store/load lists,
which `worker.py:266,289` copy) rather than as a second patch. A tier that can
only be turned off by a 4-minute reboot is not safe to have on.

`snapshot.pr56810.disarm_limits` states what it cannot undo: an in-flight copy
already handed to workers finishes, blocks already parked in host RAM stay
parked (re-arming makes them reachable again), and the worker process is never
contacted. `patched: false` and `armed: false` are different states - the
former means a hybrid geometry will still crash the store path.

### Coexisting with the plugin's other two patches

Three modules patch engine classes at runtime. They never touch the same
attribute, and each carries its own marker tag, so an overlap is detectable
rather than silent:

| Module | Class it patches | Members it owns | Marker tag |
|---|---|---|---|
| `cpu_capacity.py` | the connector BASE class | `get_handshake_metadata` (worker side), `set_xfer_handshake_metadata_pp_aware` (scheduler side); separately marks a block POOL (`POOL_MARKER`) and a manager INSTANCE (`MANAGER_MARKER`) | `kvtier-dev-plugin/capacity-1` |
| `store_hook.py` | `SimpleCPUOffloadScheduler` | `_classify_store_candidate` | `kvtier-dev-plugin/store-1` |
| `backport.py` | `SimpleCPUOffloadScheduler` | `__init__`, `_estimate_lazy_target_blocks`, `update_state_after_alloc`, `_select_eager_blocks_to_store`; plus the kill-switch wrappers `get_num_new_matched_tokens`, `build_connector_meta`, `has_pending_stores` | `kvtier-dev-plugin/backport-1` and `.../switch-1` |

**There is no overlap to resolve, and that is a property worth keeping.** The
three tag constants are deliberately distinct strings, and every member name is
owned by exactly one module. Each installer refuses to clobber a member carrying
a DIFFERENT tag (naming the owner in the refusal), so if a future change ever
picks a name another module already holds, the result is a logged refusal and a
`skipped:` entry - never a silently unwrapped patch. Do not merge the tags.

Two notes a future editor needs:

* `backport` uses TWO tags on purpose: `backport-1` for the fingerprint-guarded
  bug fix and `switch-1` for the kill-switch wrappers, which have no fingerprint
  (they only add a gated early-return around the engine's own function and can
  never mispatch a moved body). Keeping them apart is what lets `status` say
  "fix refused, switch live" - the state you want during a fingerprint drift.
* `store_hook`'s veto and `backport`'s selector compose by dispatch, not by
  order: the replaced `_select_eager_blocks_to_store` calls
  `self._classify_store_candidate(...)`, which resolves through the instance at
  call time, so it reaches store_hook's wrapper whichever of the two landed
  first. Do not "optimise" that into a direct call.
* Install ORDER in `register()` is `cpu_capacity.install()` ->
  `store_hook.install()` -> `backport.install()`, and the verbs call the three
  `ensure_installed`/`adopt_if_missed` helpers in the same sequence. It is not
  load-bearing today (no shared members); it is what keeps the log readable and
  the capacity handshake settled before anything schedules a store.

Three different "offs", which are easy to confuse: `store_hook`'s policy veto is
a per-block SKIP inside a live tier; `guard_mode` turns an exception into a
decline (development only, masks bugs); the kill switch stops the connector
scheduling ANY store or load. Only the last one is a kill switch.

## Operator runbook: the KV tier is misbehaving, right now

Terse commands for standing at the box while something is wrong. Everything
below runs **on the box** (`ssh llm ...`) because `/plugins/kvtier/*` is
loopback-or-allowlist. The mutating/dev-state routes (`/tier`, `/guard`) need
the token in every shape, loopback included (`~/.local/share/kvtier-dev-token`,
mode 600, never in the repo). The plain reads (`/status`, `/snapshot`) are
token-free from loopback.

Set the token once per shell:

```bash
TOK=$(cat ~/.local/share/kvtier-dev-token)
H="x-kvtier-dev-token: $TOK"
```

### Engine crashed / crashes on the FIRST request

The first-request crash is the divisibility assert firing because the fix is
**not fully installed**. Confirm it in one line (the tier state lives in the
engine, so read the snapshot):

```bash
# installed=false or refused_methods non-empty => the fix is incomplete.
# armed=false there is the injector protecting you, not a bug.
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot' \
  | python3 -c 'import sys,json;p=json.load(sys.stdin)["pr56810"];print({k:p[k] for k in ("installed","armed","refused_methods","auto_disarmed","disarm_reason")})'
```

Read the last fault. The engine log is `/tmp/vllm-serve.log` (the journal copy
under systemd does NOT feed `logs.db`). The crash line to grep, and the
backport's own install/arm verdicts:

```bash
# the assert the incomplete fix lets through:
ssh llm "grep -a 'not divisible by tokens_per_hash' /tmp/vllm-serve.log | tail -3"
# did the backport install? was the tier force-disarmed or arm refused?
ssh llm "grep -aE 'PR #56810 backport (INSTALLED|NOT installed)|REFUSED to arm|KV tier (ARMED|DISARMED)' /tmp/vllm-serve.log | tail -10"
```

If guard mode was armed there is also a structured capture of the fault (guard
mode is OFF at every boot; there is no capture unless someone armed it):

```bash
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/guard \
  -d '{\"op\":\"captures\",\"limit\":2}'"   # capture.decline, ...geometry.groups_failing_divisibility
```

### Runtime kill: make the tier a strict no-op, NO restart (one curl)

```bash
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/tier \
  -d '{\"op\":\"disarm\"}'"
# prove it: armed=false, and the connector now reports no pending work
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot' \
  | python3 -c 'import sys,json;p=json.load(sys.stdin)["pr56810"];print("armed=",p["armed"])'
```

Disarmed, the connector reports `(0, False)` for prefix hits, hands the workers
an **empty** `SimpleCPUOffloadMetadata` (so `worker.py:266,289` copy nothing),
and claims no pending stores - the engine serves from GPU KV alone. **`disarm`
is never refused** (it is the safe direction). What it does NOT cover, verbatim
from `disarm_limits()` in the payload:

- the **worker process is never contacted** (it is a separate process; its no-op
  arrives as empty metadata, `worker_process_patched: false`);
- an **in-flight copy already handed to the workers finishes**
  (`in_flight_transfers_undone: false`);
- blocks **already parked in host RAM stay parked** and become reachable again
  if you re-arm (`parked_blocks_freed: false`);
- a store already inside `prepare_store_specs` for the CURRENT step completes;
- `has_pending_block_frees` is not wrapped - this connector does not override
  it, so it is already `False`.

There is deliberately **no `update_connector_output` entry** in `disarm_limits`:
that method is capture-and-reraise under *guard mode* (a development facility),
not something a disarm touches.

### Permanent off (survives the next boot)

The tier env for this experiment is a **systemd drop-in**, not the launcher
(the `KV_OFFLOAD_SIZE` knob in `config/box/serve-v2.2.sh` is commented out on the
box and stays that way). The drop-in is
`/etc/systemd/system/vllm-engine.service.d/kv-tier.conf`, carrying
`KV_OFFLOAD_SIZE=<GiB>` and `VLLM_USE_SIMPLE_KV_OFFLOAD=1`. To turn the tier off
permanently, remove those two lines (or the drop-in) and restart:

```bash
ssh llm 'sudo systemctl daemon-reload'          # after editing/removing kv-tier.conf
ssh llm '/usr/local/bin/pkexec /usr/local/bin/llm-engine-ctl restart'
```

A restart is a **~4-minute boot** (weights load after `systemctl` returns).
The unit runs under `Restart=no` on purpose: a crash is recovered by the **proxy
watchdog**, not systemd, so do not add `Restart=`. Until the restart the running
engine still has the arena allocated - use the runtime disarm above to stop it
scheduling right now.

### Confirm what state the engine is actually in

`GET /plugins/kvtier/status` makes **no engine call** (it answers while the
engine is wedged) and therefore does NOT show the live armed/installed state -
it only echoes the route note. The live truth is in `snapshot` (or
`GET /tier?op=status`, token) under `pr56810`:

```bash
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/tier?op=status' -H "$H" \
  | python3 -m json.tool | sed -n '/"pr56810"/,$p'
```

Where to look in `pr56810`:

- `installed` - the **all-or-nothing** verdict: `true` only if every required
  method is patched. `armed` must be `false` whenever this is `false`.
- `refused_methods` - the method(s) that were refused; each is also
  `methods.<name>.patched: false` with the reason in `methods.<name>.reason`.
- `geometry` - the geometry guard verdict (`geometry.ok`, and
  `geometry.cacheable_skipped` names any group it would wrongly skip).
- `disarm_reason` / `auto_disarmed` - whether the injector force-disarmed, and
  which refusal caused it: `"geometry"` vs `"incomplete_install"`.
- `arm_block` - if present (`{kind, reason}`), arming is currently refused and
  this says why; `null` means arming is allowed.

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
  not loopback and not allowlisted learns nothing). With `VLLM_E2_TIER_DIAG`
  unset the worker writer never wraps, never creates a directory and never writes.
* Contained: every engine-side body is wrapped in `try/except Exception` in this
  fork's idiom (`vllm/v1/core/sched/e2_alloc_log.py:143,151,201`, capped
  `_error()` at `:92`): failures are counted, the first 20 are logged, the log
  writer has its own `except Exception`. `BaseException` is never swallowed -
  which is also why guard mode declines `Exception`s and lets a `KeyboardInterrupt`,
  `SystemExit` or `CancelledError` through to the engine.
* No engine-side timeout exists (`future.result()` at `core_client.py:971`), so
  the HTTP layer bounds the call with `asyncio.wait_for` and answers **504**.
  Cancelling that await is safe: when the engine eventually answers,
  `_process_utility_output` pops the future and swallows the resulting
  `InvalidStateError` (`core_client.py:847-865`).
* No writes, no `reset_cache`, no mutation of the tier. The side effects are,
  exhaustively: swapping the policy module reference; the store-policy and capacity
  wrappers; a trace ring and its two scheduler wrappers while ARMED (removed on
  disarm); the guard wrappers while armed or traced (restored by identity on
  disarm); and the worker's diagnostics file when `VLLM_E2_TIER_DIAG` names a
  directory. Every one of those is opt-in, and the two that change engine behaviour
  (`guard`, tracing) say so in every payload.
* The worker's *compute* path is never referenced
  (`simple_kv_offload/worker.py:293-294` records on the live compute stream);
  `tests/test_snapshot.py` asserts the read side never names it and pulls in no
  heavy import, and the import check now walks **every** file in the package so a
  new verb module cannot smuggle in `torch`. What the worker side is asked for is
  one plain int inside the init-time handshake (`cpu_capacity.py`) and, when the
  diagnostics directory is configured, the plain fields `worker_diag.build_document`
  reads. Neither touches a CUDA API or a stream.

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

709 tests, no engine and no vllm required (`.venv/bin/python -m pytest -q`; the
count in this README is a snapshot - trust `pytest --collect-only`). They cover:
loopback accept/reject (including a spoofed `X-Forwarded-For` from a real remote
peer), the allowlist (bare address, CIDR, IPv6, the `::ffff:` unwrap, unset =
loopback-only, malformed entries that neither raise nor admit, cache invalidation
when the env shrinks), the token on off-box reads *and* its continued absence for
loopback reads, the token requirement on `trace`/`guard` in **every** shape, the
guard order (peer -> enabled -> token), the `status` payload's `allow_networks` /
`loopback_only` / `non_loopback_peers_require_token` fields and its `guard_mode` /
`dev_state` / `worker_diagnostics` fields (including `guard_mode` being `null` -
not `false` - until an engine has actually said something), path rejection for the
reload verb, holder-based reload actually changing behaviour (same-size/same-mtime
edit), rollback with the traceback as data, containment (a raising policy and a
raising engine both come back as data, `BaseException` propagates), the 504/502/503
answers, that every installed engine method takes exactly one optional argument and
answers under both dispatch shapes, and that the mount point is exactly the eight
`/plugins/kvtier/*` paths (a list derived from `config.ROUTES`, so a new route
cannot go unlisted, with the openapi surface checked against it too). Three of them
(`test_real_socket_*`) drive a real uvicorn listener over a real TCP connection -
one from loopback with no token, two from a source-bound non-loopback client socket
on this host's own LAN address (skipped on a box with no such address).

The debugger verbs add six files:

| File | Tests | What it pins down |
|------|-------|-------------------|
| `tests/test_kvconfig.py` | 40 | the divisibility table against the fork's assert (`kv_cache_utils.py:2888`), a Mamba group's different `hash_block_size`, the QSA ring recognised by class name, `prefix_cacheable` never inferred from `host_rotation`, the geometry-vs-manager disagreement, every cap honouring `dropped` counts, a missing operand reported as `unknown` and never as safe |
| `tests/test_kvinspect.py` | 41 | block/request inspection against a production-shaped fake: the packed block-hash decode, `in_cached_map` via the tier's own dict, `coverable_blocks` and `binding_term` recomputed by hand, the `ids_cap`/drop reporting, every "does not exist" answer naming what does |
| `tests/test_trace_ring.py` | 36 | the ring's fixed capacity and wrap accounting, `read` never walking O(depth), the `gap_records` a slow poller gets, `clear` not resetting `seq`, arming installing and disarming removing the observers, a disarmed ring costing no allocation and no wrap, the step observer returning the fork's value untouched |
| `tests/test_guard_mode.py` | 46 | every declined method returning the fork's documented no-op shape (and the one method that must re-raise), `BaseException` never swallowed, the metadata rebuild falling back to a re-raise if the fork renames its class, captures bounded with the full text only via the `captures` op, `guard_mode: true` in every payload, disarm restoring the originals by identity, tracing keeping the wrappers alive after a disarm |
| `tests/test_worker_diag.py` | 43 | the file schema against the reader's contract (`version`/`epoch`/`ts`/`role`/`rank`/`arena`/`counters`), the rank name matching the file name, an unknown rank meaning no file at all, the cadence and the time floor, "a change writes now", the empty pool still writing zeros, atomic replace and the no-truncate rule, the size cap trimming prose before data, a full disk failing a step never, `bytes_used` as a proven floor |
| `tests/test_routes_dev_verbs.py` | 57 | every argument rule twice (GET query and POST body) against one table, an unknown key `400`-ing without reaching the engine, a `413` before any parse, bool-is-not-int and int-is-not-bool, control characters refused, the route->method map being un-hijackable by a body, the empty body sending no positional argument, `guard`/`trace` never mutating the policy module, the `/status` mirror staying `null` until an engine speaks |
| `tests/test_dev_surface_e2e.py` | 29 | the routes wired to the REAL engine-side verbs on a fake engine, with the fork's own `len(args) <= len(parameters)` assert reproduced on the wire: every verb driven with and without an argument, every payload surviving JSON, a block/request read against the production-shaped geometry, a captured fault agreeing with the ring, the snapshot and the `/status` mirror, disarm restoring the fork's functions, and the guard composing with this package's own tier wrapper instead of leaving the fault-prone methods unguarded |

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

* **Guard mode is a development facility.** Armed, it turns a tier exception into a
  record plus a decline, so a real bug can hide behind a decline that looks like
  normal "nothing to do" - and `build_connector_meta`'s substitute is an *empty*
  metadata, which means the worker does nothing that step. It is off at every boot,
  needs the token, echoes `guard_mode: true` everywhere, and must be disarmed before
  a production boot. The same reasoning covers tracing: the ring is bounded and the
  observers are pure, but they are wrappers on the scheduler loop.
* `/status` mirrors `guard_mode` and `trace_armed`; it does not know them. They come
  from the last engine answer this API process saw and are `null` before the first
  one, so a `/status` right after an API restart says *unknown*, not *off* - and
  `status.dev_state.note` says exactly that, naming the boot default as a default
  rather than presenting it as an observation.
* Worker diagnostics are **self-reported per rank**. The numbers come from the
  worker's own plain fields (`num_cpu_blocks`, the `SimpleCPUBlockPool` totals,
  `cpu_to_gpu`/`gpu_to_cpu` lengths): the arena is never walked and no CUDA API is
  called, so `arena.bytes_used` is a proven floor from blocks in flight rather than
  the pool's occupancy, and `dropped_cpu_full`/`boundary_stored` are absent because
  they are scheduler-side counters. A file proves a worker *reached* a state; it is
  not a measurement of the arena.
* The worker writer resolves the connector class through the capacity handshake, so a
  rank that never reaches `register_kv_connector` (no KV connector role, or a
  connector overriding both handshake methods) simply never writes a file - reported
  as `install: {class: null, skip_reason: ...}`, never as a silent gap.
* `snapshot` reads the **scheduler process** view (one EngineCore per TP2xPP2
  layout). Per-rank arenas arrive only as the handshake number each worker reports
  (`cpu_capacity.per_rank`) and, when `VLLM_E2_TIER_DIAG` is set, as that rank's own
  file; no per-step worker path is touched.
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
