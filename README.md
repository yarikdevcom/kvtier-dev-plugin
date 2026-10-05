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
GET|POST /plugins/kvtier/tier  -- utility -> kvtier_dev_tier          (kill switch, off at boot; #56810)
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
| `kvtier_dev_plugin/engine_side.py` | `vllm.general_plugins` entry: attaches the eight engine methods, fits the connector-creation hook, wires the logger and the worker-writer retry |
| `kvtier_dev_plugin/snapshot.py` | the read side + the containment boundary, and the `guard`/`trace`/`worker_diag`/`lookup_ledger` sections of the snapshot, including the `tier.cpu_hash_index` parked-KV reading. Stdlib only |
| `kvtier_dev_plugin/lookup_ledger.py` | **development only**: the bounded ring that joins each `get_num_new_matched_tokens` offer to the `update_state_after_alloc` adopt that followed it, and the verdict that classifies the hits the scheduler did not adopt |
| `kvtier_dev_plugin/miss_attribution.py` | **development only**: the per-request *why did this request miss* attribution that rides the lookup ledger's records (query length, hit, residues, a derived first-miss position, a verdict from a fixed enum), the MEASURED host-RAM retention depth, and the free-order witness that confirms or refutes the prefix-root eviction hypothesis |
| `kvtier_dev_plugin/kvconfig.py` | `kvconfig`: the KV-cache geometry and the divisibility verdict the tier's first store depends on |
| `kvtier_dev_plugin/kvinspect.py` | `inspect`: one block, or one request, per group, with every id list capped |
| `kvtier_dev_plugin/trace_ring.py` | the bounded event ring and the `trace` verb (arming installs the observers, disarming removes them) |
| `kvtier_dev_plugin/step_trace.py` | the two scheduler-step observers the ring reads from. Pure observers, no catching |
| `kvtier_dev_plugin/guard_mode.py` | **development only**: the tier exception capture-and-decline, its captures, and the `guard` verb |
| `kvtier_dev_plugin/worker_diag.py` | the worker-side per-rank JSON writer (files are how a worker reports; it has no utility channel) |
| `kvtier_dev_plugin/disk_store.py` | **development only**: the worker half of the SSD spill tier - the fingerprint THIS rank builds, the `store_keys` resolver, the store-completion spill hook (composing over `worker_diag`/`blockdump`), and the engine half of the `/disk` verb. Drives `disk_spill.py` (the queue + writer) and `disk_layout.py` (the naming rules). Stdlib only |
| `kvtier_dev_plugin/policy_holder.py` | late-bound holder for the policy module; trial-load + swap, rollback on failure |
| `kvtier_dev_plugin/policy.py` | **the file you edit.** `decide(view)` observes, `decide_store(ctx)` vetoes |
| `kvtier_dev_plugin/store_hook.py` | the runtime patch on the tier's store-decision path: idempotent, non-clobbering, fail-open |
| `kvtier_dev_plugin/backport.py` | the runtime backport of upstream PR #56810 (hybrid-geometry fix), the tier's runtime kill switch, and the `create_connector` hook that makes both land before the first request. Stdlib only |
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
| `VLLM_E2_TIER_DISK_JOURNAL_DIR` | where the SSD tier's spill **journal** lives, apart from the rest of the diagnostics directory. **Unset = fall back to `VLLM_E2_TIER_DIAG`** (the pre-0.2.19 behaviour). Set it to storage that survives a reboot: the journal is the only map from a KV identity to a spilled file and cannot be rebuilt, while the KV files it describes outlive it. **The path does not have to exist** - the plugin creates it (mode `0750`, the mode the store's own shard directories get) on the way to the first journal line, once per directory. **Only the journal follows this var** - the spill arm file stays in `VLLM_E2_TIER_DIAG` and dies at boot, because an arm must not survive a reboot. A journal on tmpfs is warned about once and reported (`journal_dir=tmpfs` on the boot line), and a journal directory that cannot be created or written is reported (`journal_dir=unwritable`, `journal.writable: false`, `journal.directory_error`, `journal.write_errors`) - either way it is never a refusal to spill. |
| `VLLM_E2_TIER_DIAG_INTERVAL` | how often a rank rewrites its file, in engine steps. Default `50`, clamped to `[1, 100000]`. A step that *changed* something writes immediately (subject to the floor below). |
| `VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS` | the rate floor between writes, in seconds. Default `1.0`, clamped to `[0, 3600]`; `0` means "no floor". |
| `VLLM_E2_TIER_DIAG_EPOCH` | pin the per-boot `epoch` string the reader uses to prove two facts came from the same boot. Unset = derived once per process and stable for its lifetime. |
| `VLLM_E2_TIER_DIAG_RANK` | pin this rank's id (`PP<n>_TP<m>`) instead of deriving it from the already-imported `vllm.distributed.parallel_state`. An illegal value means **no file at all**, because the reader refuses a file whose `rank` does not match its own name. |
| `VLLM_E2_TIER_MISS_LOG` | the miss ledger's OPTIONAL file mirror: a path to append one JSON line per tier lookup to. **Unset = no file, no `open()`, no I/O.** Env-only on purpose: an HTTP argument naming a path the engine appends to would be a write surface this plugin does not have anywhere else. Bounded by `MAX_MIRROR_BYTES` (32 MiB), where it STOPS and counts the stop (`mirror_stopped`) rather than filling a disk. |
| `VLLM_E2_TIER_DEV_BOOT_INERT` | **the boot opt-out.** Unset (the default) = the boot brings the tier up *working*: at connector creation it installs the backport, arms guard mode, and arms the tier. Set (`1`/`true`/`yes`/`on`) = the old posture: patch, then stay disarmed and unguarded until a verb says otherwise. Same falsy set as `VLLM_E2_TIER_DEV`. It is the only env var that touches either facility, and it can only turn arming **off**. |

There is deliberately **no env var for guard mode** - no env var that turns it *on*.
The boot arms it (see *Booting the tier*), because an unguarded tier exception kills
the EngineCore, and `VLLM_E2_TIER_DEV_BOOT_INERT` is the one switch that stops the
boot doing so; there is no env var that arms guard mode on an inert boot, which can
only be done with `POST /plugins/kvtier/guard` and the token - see *Guard mode*.

There is equally **no env var that arms the KV tier on an inert boot**. The tier is
disarmed at plugin *load* (`backport.BOOT_ARMED = False`) because the plugin's
install cannot land before the manager class exists, and arming is gated on that
install being complete - so an env var that armed at plugin load would be an env var
that arms an unpatched engine. What the boot does instead, at CONNECTOR CREATION,
is arm the tier itself once the install verifies; `VLLM_E2_TIER_DEV_BOOT_INERT`
restores the fully manual posture, where `POST /plugins/kvtier/tier {"op":"arm"}` is
the only thing that arms it. See *Booting the tier*.

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
# ['/plugins/kvtier/block', '/plugins/kvtier/blockdump',
#  '/plugins/kvtier/disk', '/plugins/kvtier/guard', '/plugins/kvtier/kvconfig',
#  '/plugins/kvtier/lookups', '/plugins/kvtier/policy/reload',
#  '/plugins/kvtier/request', '/plugins/kvtier/snapshot',
#  '/plugins/kvtier/status', '/plugins/kvtier/tier', '/plugins/kvtier/trace']
```

Against the **running** server, the equivalent that does not depend on the
FastAPI route-tree shape (newer FastAPI nests included routers behind an
`_IncludedRouter`) is the OpenAPI surface:

```bash
curl -s http://127.0.0.1:8000/openapi.json | python3 -c \
  'import json,sys; print(sorted(p for p in json.load(sys.stdin)["paths"] if "kvtier" in p))'
# ['/plugins/kvtier/block', '/plugins/kvtier/blockdump',
#  '/plugins/kvtier/disk', '/plugins/kvtier/guard', '/plugins/kvtier/kvconfig',
#  '/plugins/kvtier/lookups', '/plugins/kvtier/policy/reload',
#  '/plugins/kvtier/request', '/plugins/kvtier/snapshot',
#  '/plugins/kvtier/status', '/plugins/kvtier/tier', '/plugins/kvtier/trace']
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
| `/tier` | GET, POST | **always** | the tier's runtime kill switch and the PR #56810 backport's state |
| `/lookups` | GET, POST | **always** | the lookup/adopt ledger: read, arm, disarm, verdict |
| `/blockdump` | GET, POST | **always** | the bounded raw-KV-byte dump: status, arm, disarm, dump (see `blockdump.py`) |
| `/disk` | GET, POST | **always** | the SSD spill tier: status, arm, disarm, `spill-one`, `ls` (see `disk_store.py`). A mutating `op` on GET is a `405` |
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
                                             #
                                             # THE ID IS BOUNDED PER POOL: `found`/
                                             # `block` is the GPU pool's answer and
                                             # `tier_cpu_slot` is the host-RAM pool's,
                                             # each validated against ITS OWN block
                                             # list (measured live: GPU 317 blocks,
                                             # CPU pool 1331 slots - so a slot above
                                             # the GPU count is nameable, and this is
                                             # how you find a slot to hand to
                                             # `spill-one`). `sections` says which
                                             # half answered; an id out of range for
                                             # a section names that pool and its size
                                             # in `reason` / `reason_kind`.
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

`/trace`, `/guard` and `/lookups` require the token in every shape, loopback
included: a route that can change engine dev state is token-gated as a whole, not
per field.

## The lookup ledger: where a host-RAM prefix hit goes missing (`lookup_ledger.py`)

The warning is unambiguous and the state behind it is not:

```
SimpleCPUOffloadScheduler: update_state_after_alloc called for req_id=chatcmpl-...
with no external tokens but get_num_new_matched_tokens() unexpectedly recorded
a pending CPU hit; releasing the stale pin.
```

The fork pins a CPU hit ONLY when `hit_length > 0` and then returns
`(hit_length, True)` (`manager.py:363-374`), so "a pending hit was recorded" means
the connector really did name a positive number of tokens - while the scheduler
really did arrive with `num_external_tokens == 0`, and
`vllm:external_prefix_cache_hits_total` did not move. Somewhere between "N tokens
are in host RAM" and "the scheduler adopts them", N becomes 0. The ledger records
the triple that says which rule did it, per request id:

| field | where it comes from |
|---|---|
| `local` | the second argument the scheduler passed to `get_num_new_matched_tokens` - **already block-aligned** (`scheduler.py:969-980`) |
| `hit_returned` | the first element of what the connector returned (`None` = "no verdict", and the scheduler skips the request) |
| `adopted` | the `num_external_tokens` the scheduler later passed to `update_state_after_alloc` (`scheduler.py:1228`) |

* **Armed only in development - and the recording is not a third wrapper.** It rides
  `guard_mode._record_success()`, which already has the method name, the arguments
  and the return value of both calls, on the armed *and* the disarmed path. `arm`
  installs those wrappers through `guard_mode.ensure_installed()` - the code that
  already composes over the backport's replacement, refuses a foreign owner marker
  and is idempotent - so the ledger owns **no class attribute of its own** and cannot
  clobber or double-stack a layer. One set of wrappers, three consumers (guard mode,
  the trace ring, the ledger), and `guard_mode.wrapper_customers()` is the question
  every removal asks first: a `/trace` disarm cannot blind the ledger, and neither
  can a `/guard` disarm.
* **Bounded, because this runs on the serving loop.** A ring of `cap` slots (`256`
  default, clamped to `[16, 4096]`) preallocated at arm, holding plain dicts.
  `dropped` counts what the wrap ate; `dropped_unjoined` the subset that can never be
  joined. The join index holds one entry per live record, so it is bounded by the
  same cap, and evicting a record takes its entry with it (a later adopt for that id
  is then honestly an `adopt_without_lookup`, never a stamp on someone else's record).
* **Joined by request id, never by call order.** The scheduler walks a queue, so any
  number of other requests' calls land between an offer and its adopt; a request can
  be skipped so no adopt ever arrives; and a `num_computed_tokens > 0` re-entry
  adopts without ever looking up (`scheduler.py:1046-1060`). All three are normal and
  each has its own bucket.
* **The verdict** (`op: verdict`) classifies every readable record: `adopted_full`,
  `adopted_partial`, `adopted_more_than_offered`, `dropped_to_zero`, `no_host_copy`,
  `no_verdict_from_connector`, `lookup_without_adopt`, `adopt_without_lookup`,
  `unparsable`. `dropped_to_zero` (`hit_returned > 0`, `adopted == 0`) is the case
  being hunted; each one carries `local_mod_block`,
  `min_partial_tail_the_rule_needs`, `implied_true_local_range` and
  `tail_rule_can_explain`.

Read that last field deliberately, because it is the honest part. The connector is
only ever handed `local - local % block_size`, so `local_mod_block` is 0 by
construction in every record that has a hit at all (the fork's own gate,
`manager.py:334-346`, returns `(0, False)` for an unaligned input), and the
scheduler's `partial_tail` is therefore **inferred, never observed**. What the data
can do is *refute*: the rule (`scheduler.py:996-1010`) drops the offer when
`partial_tail >= ext_tokens`, and a tail is by construction smaller than one block -
so a dropped hit at or above `block_size` cannot come from that rule at all
(`tail_rule_can_explain: false`), while one below it is consistent with a true local
hit somewhere in `[local + hit, local + block_size - 1]`.

```bash
T=$VLLM_E2_TIER_DEV_TOKEN
# arm it, let it see traffic, then ask the question
curl -s -H "x-kvtier-dev-token: $T" -X POST localhost:8000/plugins/kvtier/lookups \
     -H 'content-type: application/json' -d '{"op":"arm"}' | jq .reason
curl -s -H "x-kvtier-dev-token: $T" -X POST localhost:8000/plugins/kvtier/lookups \
     -H 'content-type: application/json' -d '{"op":"verdict"}' | jq '.counts, .dropped_examples'
curl -s -H "x-kvtier-dev-token: $T" localhost:8000/plugins/kvtier/lookups | jq .entries
curl -s -H "x-kvtier-dev-token: $T" -X POST localhost:8000/plugins/kvtier/lookups \
     -H 'content-type: application/json' -d '{"op":"disarm"}'   # removes the wrappers if alone
```

A bounded preview of the ring is in every `snapshot` under `lookup_ledger`, beside
its `not_covered` list: nothing before the arm is recorded, a guard-DECLINED call is
*not* recorded (that value never reached the caller - look in the guard captures), a
record lost to the cap is absent from a verdict, and an adopted N is the scheduler's
decision, not proof that a worker loaded N tokens.

## How much parked KV is retrievable (`snapshot.py`: `tier.cpu_hash_index`)

`cpu_pool.free_blocks` is not the answer, and the two numbers that look like a
contradiction are not one:

```json
"cpu_pool":             {"total_blocks": 665, "free_blocks": 664}
"boundary_store_stats": {"stored": 4372}
```

A parked CPU block is handed **back to the allocator** when its store finishes
(`BlockPool.free_blocks` / `unpin_blocks` only move it in the free queue,
`block_pool.py:705-741`, `:770-798`), and what keeps it retrievable afterwards is its
entry in `cpu_block_pool.cached_block_hash_to_block` - the index the fork itself
resolves hits through (`get_one_block`, `manager.py:605`, `:940`, `:1251`).
**Allocator freeness does not mean the index is empty.** `tier.cpu_hash_index` reads
that index:

| field | what it is |
|---|---|
| `entries` | `len()` of the map: distinct `(hash, group)` keys. The map does not de-duplicate (`block_pool.py:47-51`), so this is *hit-able prefixes*, not blocks - `sampled_keys_with_multiple_blocks` says when a key holds more than one |
| `tokens_upper_bound` | `entries × hash_block_size`, labelled an upper bound |
| `per_group` | distinct entries per KV-cache group, from the 4 packed big-endian bytes of each key (`kv_cache_utils.py:74-93`); a capped scan (`MAX_HASH_INDEX_SCAN`), and `key_handle` admits when it had to read the map's private `_cache`. **Keys, not spans** - see the granularity trap below |
| `parked_blocks_per_group` | distinct CPU **blocks** behind those keys, per group, from the *same* scan: one parked span occupies one block per group (`get_new_blocks` takes distinct ids, `manager.py:755-758` / `block_pool.py:678`), so this is **spans parked in the group**. `min()` over the prefix-cacheable groups × `tier.block_size` is the true joint coverage; `sum()` ÷ `cpu_pool.total_blocks` is the true occupancy. Refused (null + `parked_blocks_per_group_reason`) on a truncated/partial scan, a group-cap hit, the `MAX_PARKED_BLOCK_IDS` bound, or a value that is not a block - never a prefix, never a 0 |
| `per_group_primary` / `per_group_secondary` | the split of `per_group` by what each entry *is*: primary = the key the block itself carries (`block.block_hash`, the one insert that got a real `num_tokens`), secondary = an extra key the pool also files for that block (`cached_block_hashes_by_block`, `block_pool.py:641` fed by `manager.py:1039-1044`). Withheld whole when any entry is neither - "not primary" is not "secondary" |
| `key_split_matches_blocks`, `keys_backed_by_multiple_blocks` | the cross-check: one primary key per block, so the two maps must agree. `False` names the case that broke the premise (a key holding several blocks, `block_pool.py:47-51`) instead of leaving a consumer to divide by a guessed divisor |
| `sampled_values_without_hash`, `sampled_null_blocks` | the empirical tripwire for a stale entry, from at most 32 values |

On staleness the fork's contract is checkable rather than guessable:
`get_new_blocks` runs `_maybe_evict_cached_block` → `_remove_cached_block_hashes` on
every block it hands out (`block_pool.py:664-688`, `:725-745`) and `unpin_blocks`
documents the blocks it releases as readable *until reused*, so the reading says
`stale_entries: "not_expected_on_pool_paths"` - with
`stale_entries_not_ruled_out` naming the one thing that would defeat it (a write into
a CPU block that never went through `get_new_blocks`) and
`retrievable_now_vs_retained` saying what it does *not* claim: an entry's block is an
eviction candidate the moment its store releases it, so the count is *retrievable
now*, not *retained*. A missing attribute, an unsized container or a `len()` that
raises comes back as a `reason` string with no `entries` key at all - a missing
reading is information, a fabricated one is a lie.

### The granularity trap: `per_group` is not spans

The groups do **not** mint keys at the same rate per parked span. The primary hash is
one key per `block_size`-token span; the fine-graining group also mints its
intra-block hashes at the `hash_block_size` ticks, inserted against the **same**
`cpu_block` with `num_tokens=None` (`manager.py:1039-1044`) - dictionary entries that
cost no block and no bytes. Live, 2026-10-05, idle box:

```
per_group                  {0:98, 1:98, 2:98, 3:99, 5:308}   sum 701 keys
parked_blocks_per_group    {0:98, 1:98, 2:98, 3:99, 5: 77}   sum 470 blocks
occupancy of a 665-block arena   701/665 -> 1.00 (a lie)  vs  470/665 = 70.7%
joint depth a 5-group resume can use   min(per_group)=98 (over-reports 27%)  vs 77
```

There is **no divisor in the payload to divide by**, and the ratio is not even stable
(`per_group[5]` measured 317, 317, 318, 320, 320, 320 across six polls of one idle
minute), so a hardcoded `/4` is a fabrication waiting to be wrong. Consume
`parked_blocks_per_group`; the key fields stay for the cross-check and because a proxy
and a dashboard already read them.


## Why THIS request missed, and how much the RAM tier can retain (`miss_attribution.py`)

Aggregate instruments cannot answer "why did THIS 176k-token request get
`cached_tokens = 0` while its neighbour hit 97%". This capability turns every tier
lookup into ONE ledger entry - riding the lookup ledger's existing ring, keyed by
request id - with a cause attached, plus a measured answer to "how many tokens can
the RAM tier actually retain".

```bash
# arm it (token-gated, like every other shape of /lookups)
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/lookups \
  -H 'content-type: application/json' -d '{\"op\":\"arm\",\"cap\":256}'"

# the per-request attributions (optionally &verdict=root_missing)
ssh llm "curl -s -H '$H' 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=requests&limit=32'"
ssh llm "curl -s -H '$H' 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=depth'"   # measured retention
ssh llm "curl -s -H '$H' 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=order'"   # root-eviction witness
ssh llm "curl -s -H '$H' 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=rates'"   # unit-checked ratios only
```

`op=depth` answers **without arming anything**, and the same block is in
`snapshot.tier.cpu_retained_depth`: "how much can the tier retain" is a question
about the pool, not about the observer.

### The ledger entry (every numeric field carries its unit)

| Field | Unit | What it is |
|-------|------|-----------|
| `num_tokens` | tokens | `request.num_tokens` |
| `local_as_passed` | tokens | the block-aligned local hit the scheduler handed the connector (`scheduler.py:973-979`) |
| `local_mod_block`, `local_mod_hash` | tokens | `% block_size` / `% hash_block_size`; `local_mod_block` is 0 **by construction** from the scheduler, so a non-zero value means a different caller |
| `ext_query_hashes` | hash blocks | `len(request.block_hashes) - local/hash_block_size`: the slice `manager.py:352` actually queries |
| `ext_query_tokens_possible` | tokens | what the tier COULD match, before its own caps |
| `ext_query_tokens_capped` | tokens | what it MAY match: `min(possible, num_tokens - 1 - local)` (`manager.py:359-362`) |
| `ext_hit_tokens` | tokens | the connector's offer (the ledger's `hit_returned`) |
| `residue_tokens` | tokens | `capped - offered`: the prefix queried and NOT served |
| `total_residue_tokens` | tokens | `num_tokens - local - offered`: everything recomputed |
| `walk_matched_spans`, `first_miss_span` | spans | **derived**: a strict prefix walk (`single_type_kv_cache_manager.py:806-808`, "a missing block implies every later block misses too", the loop `break`s) makes depth and the failing index ONE fact, so `matched = offered // span_tokens` |
| `first_miss_token_offset` | tokens | the absolute prompt offset where the walk stopped |
| `joint_hit_tokens`, `clamp_residue_tokens` | tokens | from the tier's own `cpu_coordinator.find_longest_cache_hit`: the reconciled hit and `num_uncached_common_prefix_tokens`, which `manager.py:362` otherwise throws away |
| `group_depths_tokens`, `independent_group_depths_tokens` | tokens per group | after reconciliation (full-attention groups are TRUNCATED to the hit, `kv_cache_coordinator.py:966-973`), and from the opt-in deep probe, which does not reconcile |

`hash_value` is still never read anywhere in this package - and neither is
`block.block_hash`: only ids, lengths and counts.

### What may be claimed, and what may not

Verdicts: `hit`, `no_verdict`, `zero_by_alignment_gate`,
`zero_by_skip_reading_prefix_cache`, `zero_by_last_token_rule`,
`zero_by_no_hashes_left`, `cold_cache`, `pool_pressure`, `root_missing`,
`clamp_at_group<idx>`, `clamp_group_ambiguous`, `clamp_limited`,
`partial_hit_clamp_suspected`, `unknown` - all 14 are reachable, and the test suite
fails if one stops being so.

Not computable, and published as such (`undeterminable`) rather than guessed:

* **`content_diverged@<offset>`** - separating "the client changed the prompt at
  token N" from "the block at chain index N is gone" requires knowing that a
  DIFFERENT hash was ever parked at that index. That needs the hash value (never
  read here) or a per-request hash history (never kept). The ledger publishes the
  POSITION and refuses the cause.
* **per-request GPU eviction** - the connector sees the tier's answer, never the GPU
  coordinator's walk. VRAM is visible only as a census (`gpu_census.py`) and engine
  counters, so a per-request VRAM verdict would be a guess about a walk this module
  cannot see.
* **a single clamping group on the normal path** - the returned per-group lists are
  truncated to the reconciled hit, so every full-attention group ties at the minimum
  and the binder is indistinguishable from the truncated. Only
  `find_longest_cache_hit_per_group` (`kv_cache_coordinator.py:984`) names it, and it
  is a re-lookup through `BlockPool.get_cached_block` (dict reads, no `touch`, no
  queue move), so it is opt-in (`deep: true`), sampled (`sample_every`, default 20)
  and hard-capped (`MAX_DEEP_PROBES`, 64).

`op=rates` refuses any ratio that mixes units - scheduler steps
(`_load_event_counter`, `manager.py:537`) against blocks (`evictions_total`) against
tokens - because that division is exactly what produced a wrong production
conclusion.

### Measured retention: three answers 785x apart, resolved by measuring

The authoritative path in the running tree is
`num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // kv_cache_tensors[0].size)`
(`manager.py:273-275`); there is no `cpu_block_size = lcm(...) * sum(page_sizes)`
formula in it. The disputed part was never the pool size, it was what one slot
means:

* a CPU slot is **one GROUP's span**: the store loop allocates inside the per-group
  loop (`manager.py:711`, `:757`), so a 3,200-token prefix costs ONE SLOT PER
  prefix-cacheable group (five on this box - group 4, the QSA scratch ring, is
  skipped by `manager.py:838` and parks nothing).
* therefore `index_entries x hash_block_size` counts INDEX KEYS at 40-token
  granularity rather than spans and under-counts by `block_size / hash_block_size`
  (80x); `total_blocks x block_size` ignores the per-group multiplication
  (over-counts by the group count); and `GiB / summed page sizes` assumes one slot
  holds every group at once, which the per-group allocation refutes.
* `joint_parked_tokens_max` is the `min` over the prefix-cacheable groups of
  `spans x group_block_sizes[g]` - ONE conversation's retrievable depth;
  `aggregate_parked_group_tokens` is the total volume across conversations. On the
  measured box (1,331 slots, 1,330 parked, spans `[3200 x5, 8]`, per-group parked
  `[259,259,260,260,-,292]`): ~828,800 tokens of joint depth right now and a
  one-conversation ceiling (`per_conversation_ceiling_tokens`) of ~850,000 tokens.
  A 200k-token prefix costs ~313 of 1,331 slots, so the RAM tier is NOT structurally
  incapable of rescuing one.
* **`free` is not `empty`.** `BlockPool.free_blocks` appends HASHED blocks back into
  the free queue as reuse candidates (`block_pool.py:796-798`), so
  `cpu_pool.free_blocks == parked spans` is the FULL-tier signature. The retrievable
  number is `cached_idle` (+ `referenced_hashed`, pinned mid-restore but still
  parked), and `free_uncached` is what is genuinely empty.

### The free-order witness: a hypothesis, made falsifiable

`kv_cache_coordinator.py:413` frees the GPU path's blocks with `reversed(...)`, and
`FreeKVCacheBlockQueue`'s own docstring (`kv_cache_utils.py:257-262`) says the tail
of a chain must sit at the FRONT of the queue and that the reversing is done
OUTSIDE that class. `v1/simple_kv_offload/` contains **zero** `reversed()` calls,
and its four `cpu_block_pool.free_blocks(...)` sites (`manager.py:1113`, `:1121`,
`:1272`, `:1297`) pass chain order. `free_blocks` appends hashed blocks in the order
given and `get_new_blocks` pops the front - so chain order in would mean the parked
ROOT is queued before its own tail, and since a missing root makes every later block
unhit-able, the whole tail would be worthless. `op=order` MEASURES it: a
park-sequence number per CPU block id (one int array the size of the pool), then per
`free_blocks()` call the up/down tally of consecutive pairs, published as
`CONFIRMED`, `REFUTED`, or `INCONCLUSIVE` (blocks parked before arming carry no
sequence, so arm first). A confirming trace is `root_first > root_last` with
`asc_pairs` ~ the chain length per group and `desc_pairs` ~ the number of group
joins. What it does NOT claim is in the payload too: it measures the ORDER blocks
enter the free queue, not whether allocation pressure then pops that root before a
restore needs it.

### Cost, bounds, and what it never touches

Measured on this host: 0.07 us disarmed (one branch), 3.4 us median / 4.8 us p99
armed for a hit, 8.5 us including the coordinator observation, and 0.75 ms per pool
walk - memoised to one per second and paid only on a zero-hit lookup. The
attribution rides the ledger's ring (`cap` 16..4096, default 256), holds NO engine
block reference (proven with weak references: holding one is the refcount bug class
PR #56810 addresses), and `op=requests` REFUSES over `MAX_REQUESTS_BYTES` (200 KB,
about 100 attributed records at ~1.9 KB each) rather than silently shortening the
list of requests that missed.

## Guard mode: develop against a live engine without an outage (`guard_mode.py`)

> **DEVELOPMENT ONLY. Disarm it before any production boot.** Armed, a tier
> exception becomes a record plus a safe decline, which is exactly how you keep a
> live engine up while debugging - and also how a real bug gets masked: a declined
> store is silent, and a decline can leave tier state half-advanced. **The boot
> arms it**, immediately before it arms the tier, so no store or lookup ever runs
> unguarded; there is no env var that turns it on (`VLLM_E2_TIER_DEV_BOOT_INERT`
> turns the boot's arming off, and on such a boot `POST /plugins/kvtier/guard` is
> the only way to arm it), and its state is echoed as `guard_mode` by `status` and
> by every verb payload. **Because it is armed by default, the fault ring is the
> evidence that a boot was clean**: after every boot, read `POST /plugins/kvtier/guard
> {"op":"captures"}` and expect it empty. A tier that is armed, up, and reporting
> zero work usually means it is *declining* - and the captures say why.

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

## Dumping a few parked bytes to settle a byte-level question (`blockdump.py`)

Some questions are not answerable from counters. "A parked 3,200-token span costs 5
slots (one per prefix-cacheable group), yet `cuda_mem_ops.py:244-250` reads as
though one store writes `bpb` bytes for EVERY region at ONE GPU index - so does a
store made for group 0 also write the other groups' bytes there?" is a question
about BYTES. Sibling slots demonstrably hold different GPU block ids, so the data
is not simply duplicated; only real bytes settle it, and guessing risks silently
corrupting resumed prefixes.

So this capability reads a HANDFUL of real parked slots' bytes and nothing else.
It is split like `worker_diag`, because the bytes live in the worker (as a slice of
`SimpleCPUOffloadWorker.cpu_kv_caches[region]`, `worker.py:235-244`) and nothing in
a worker is reachable over the utility channel:

* the **EngineCore** half (`/blockdump`) validates against the caps and writes ONE
  small control file (`blockdump-request-<rank>.json`); it never returns bytes and
  never reads one;
* the **worker** half, on a wrapped `get_finished`, performs the read and writes
  ONE bounded response file (`blockdump-<rank>.json`) with base64 bytes and a
  per-block `sha256`. The operator collects that file - it is on the ENGINE host,
  in `$VLLM_E2_TIER_DIAG`.

**The read mutates nothing.** A slot's bytes are addressed by host-memory
arithmetic only - `data_ptr() + block_id * stride(0) * element_size + offset`,
`ctypes.string_at`. No CUDA (a stray context at 0.96 utilisation is an outage), no
GPU pool, and no `ref_cnt` / free-queue / hash-index change: the CPU block pool is
scheduler-side state this path never even looks at, which
`tests/test_blockdump.py::test_no_dump_path_even_looks_at_the_block_pool` proves
with a pool that records every attribute read on itself (and records none).

**DISARMED at every boot, and there is no env var that arms it** - the guard-mode
rule, applied to the one facility that reads KV bytes. `POST {"op":"arm"}` arms it
and the arm **expires** by itself (`seconds`, default 1800, max 7200). A worker
cannot be told anything over the utility channel, so the arm crosses to it as one
small file (`blockdump-arm.json`) that the worker re-reads at most once every
`POLL_INTERVAL_SECONDS` (2 s). Disarmed, a worker's whole cost is one `os.stat` per
poll: no open, no parse, no KV byte read, no response file. Deploy it, leave it
disarmed indefinitely, and the box does no dump work.

**The caps, and their units** (all reported in `caps` with a `units` map; every one
REFUSES with the numbers named and is counted - nothing is ever truncated):

| Cap | Value | Unit |
|-----|-------|------|
| `max_bytes_per_request` | 1 048 576 | bytes of raw KV bytes per request (pre-base64) |
| `max_blocks` | 8 | blocks (slots) per request |
| `max_response_bytes` | 4 194 304 | bytes of the response file (a worker step must not stall writing a giant file) |
| `max_requests_per_window` / `request_window_seconds` | 20 / 300.0 | dumps per rolling window |
| `arm_default_seconds` / `arm_max_seconds` | 1800 / 7200 | how long an arm lives |
| `poll_interval_seconds` | 2.0 | the worker's poll floor |
| one request per rank | - | a `dump` while one is pending is refused (and costs no window slot) |

A `length` is REQUIRED by `op: dump` (no default, `endpoint.ARG_SPECS` bound pinned
equal to the engine's own cap): the caller must ask for N bytes of one region at one
slot, so there is no path that dumps "the arena" and no path that leaks a span. The
`block_id` is a **slot index in that rank's arena tensor for that region**
(`0 .. num_cpu_blocks-1`), not a GPU block id; a wrong `group` is answered with the
region names this rank actually has.

The three-state capability marker is in `/plugins/kvtier/status` (`blockdump:
{deployed, armed, capability}`), in every `snapshot` (`snapshot.blockdump`) and in
the verb's own `armed_state`: `not_deployed` (no `VLLM_E2_TIER_DIAG`, no channel, no
hook) / `deployed_disarmed` / `armed_and_ready`. Because `status` makes no engine
call, an arm it has never observed is `deployed_armed_state_unobserved` - never a
guessed `false`, the same rule as `guard_mode`.

**Both `worker_diag` and `blockdump` wrap `get_finished`.** Neither refuses the
other: a sibling layer of this package is composed with, never clobbered (the
`guard_mode.SIBLING_PACKAGE` rule), in either install order, and the reports say
which methods are `nested`. `engine_side.register()` installs `worker_diag` first
and `blockdump` on top of it, in EVERY process - and, because the worker class does
not exist yet at plugin load, the real install in a worker is driven by the capacity
handshake hook (`engine_side._on_worker_report`), which calls the writer's retry
first and the dump's second, each contained on its own.

### The dump sequence (after a restart, to settle D vs P)

```bash
T='<the VLLM_E2_TIER_DEV_TOKEN value>'      # /blockdump needs the token in EVERY shape
H="x-kvtier-dev-token: $T"
B=http://127.0.0.1:8000/plugins/kvtier
D=/tmp/kv-tier-diagnostics                  # whatever VLLM_E2_TIER_DIAG is on the box

# 1. is it deployed, and disarmed?  (expect "capability": "deployed_disarmed")
ssh llm "curl -s -H '$H' $B/blockdump" | python3 -m json.tool

# 2. arm it for 30 minutes (a fresh boot is disarmed; nothing works until this)
ssh llm "curl -s -H '$H' -X POST -d '{\"op\":\"arm\",\"seconds\":1800}' $B/blockdump"

# 2b. if you do not know the region names yet: arm a made-up group and read the
#     refusal, which lists every region this rank has
ssh llm "curl -s -H '$H' -X POST -d '{\"op\":\"dump\",\"group\":\"list\",\"block_id\":0,\"length\":16}' $B/blockdump"
ssh llm "cat $D/blockdump-PP0_TP0.json" | python3 -m json.tool     # .reason names them

# 3. one dump per region, at the SAME slot index - collecting the file in between
#    (the response file is OVERWRITTEN by the next dump, and one is in flight/rank)
for R in 0 1 2 3 5; do
  ssh llm "curl -s -H '$H' -X POST -d '{\"op\":\"dump\",\"group\":\"$R\",\"block_id\":170,\"offset\":0,\"length\":4096}' $B/blockdump" >/dev/null
  sleep 2
  ssh llm "cat $D/blockdump-PP0_TP0.json" > dump-g$R-slot170.json
  ssh llm "rm -f $D/blockdump-PP0_TP0.json"
done

# 4. compare: identical bytes across regions at one slot index => one store wrote
#    every region there; different bytes => the regions are distinct
python3 -c "
import json,glob
for f in sorted(glob.glob('dump-g*-slot170.json')):
    d = json.load(open(f)); b = d['blocks'][0]
    print(f, d['group'], b['sha256'][:16], b['length_bytes'], 'bytes',
          'bpb', b['bytes_per_block'])"

# 5. stand down (or let the arm expire on its own)
ssh llm "curl -s -H '$H' -X POST -d '{\"op\":\"disarm\"}' $B/blockdump"
```

Two operational facts the payload states too: `get_finished` only runs while the
engine is **stepping**, so an idle engine answers at its next step rather than on a
timer; and the response file is overwritten by the next dump, so collect it (or
remove it) before arming another. To compare the SIBLING slots of one span instead
of one index across regions, read the per-group slot ids from
`GET $B/request?request_id=...` (or the live block tables) and dump each region at
its own slot id - same command, different `block_id`.

## Spilling parked KV to the engine's own disk (`disk_store.py` + `disk_spill.py`)

The host-RAM tier can only grow to what the box has. The SSD tier is the write side
of the next step down: when a store into a CPU slot **completes**, that slot's bytes
can be written to one file on the engine's own filesystem, named so that a
configuration change makes the old files un-findable rather than mis-readable:

```
<VLLM_E2_TIER_DISK_ROOT>/<key16[0:2]>/kvslot-<key16>-g<group>-<geodigest12>.bin
```

* The group id is **part of the identity**. Two KV-cache groups can hash the same
  token chain, so a `key16` alone does not say whose bytes a file holds; the older
  2-field shape `kvslot-<key16>-<geodigest12>.bin` named ONE file for two different
  payloads and is now refused as malformed (`disk_layout`).

* `key16` is the first 16 hex chars of the engine's block hash for that slot (the
  contract's `store_keys`, index-aligned with `store_cpu_blocks`; a slot the engine
  could not hash arrives as `""` and costs that one slot, never the batch).
* `geodigest12` is a digest of the fingerprint **the worker builds for itself**: the
  slot width it measures from its own region tensors and the rank it resolves, plus
  the rank-invariant facts read off `vllm_config`/`kv_cache_config` and the fork's
  `get_none_hash_seed()`. One scheduler cannot broadcast one digest that is four
  ranks' digests (`total_bytes_per_block` is 11,468,800 B on PP0 and 12,902,400 B on
  PP1), so the rank-local fields are read per rank - and that is what keeps a PP1
  worker from being able to form a PP0 file's name. `store_fingerprint` is an
  optional **cross-check**: `""` (what the engine sends today) means nothing, and a
  non-empty value that disagrees with a rank's own digest is a counted refusal.

**Off, twice over.** `VLLM_E2_TIER_DISK_ROOT` must name a real filesystem (the tmpfs
roots are refused by `disk_layout`) **and** the tier must be armed - no env var arms
it, and an arm expires. Unarmed, the two wrapped methods cost two `os.environ.get`
calls per step: no stat, no open, no copy, no queue, no writer thread.

**It needs the engine patch, and says whether it has it.** A file needs a key, and
today's `SimpleCPUOffloadMetadata` (`vllm/v1/simple_kv_offload/metadata.py`) carries
none. Every entry point then refuses with a reason, counts it, and writes zero bytes;
`engine_fields` in every answer says `present` / `absent` / `unknown` for this boot,
read from the metadata **type**, so the question is answerable without generating a
store. `no_key_for_slot` in the refusal table is that state, counted.

```bash
H="Authorization: Bearer $VLLM_E2_TIER_DEV_TOKEN"; B=http://127.0.0.1:8000/plugins/kvtier
# what can this boot do?
ssh llm "curl -s -H '$H' $B/disk" | python3 -m json.tool
# arm for ten minutes, then prove the path writes real bytes for one slot
curl -s -H "$H" -X POST -d '{"op":"arm","seconds":600}' $B/disk
# `key` is the engine's GROUP-QUALIFIED block hash: the 16-hex `key16` plus the
# group id as 8 hex chars (`key16 + f"{group:08x}"`, so 24 hex). A bare 16-hex key is
# refused - it cannot say which group's bytes these are. Find both halves with
# `GET /plugins/kvtier/block?block_id=<cpu slot>`: `tier_cpu_slot.block` carries
# `hash_hex` and `hash_group_id`, and any CPU slot is nameable up to the CPU pool's
# size (which is larger than the GPU pool's - the id is bounded per pool).
curl -s -H "$H" -X POST -d '{"op":"spill-one","slot":170,"key":"a3f0b7c1d4e5f60700000003"}' $B/disk
curl -s -H "$H" -X POST -d '{"op":"ls","limit":20}' $B/disk
ssh llm "curl -s -H '$H' -X POST -d '{"op":"disarm"}' $B/disk"
```

**No fsync, on purpose.** The spill writes are buffered: `write_committed_file`
keeps the `.part` + `O_CREAT|O_EXCL` + `os.replace` commit ordering (so *a committed
name means a complete file* against a torn write) but fsyncs nothing, so the page
cache owns device ordering and the writer thread never stalls on a platter. These KV
bytes are a cache, not a system of record, and the safety net is on the read side: a
journal entry whose bytes never reached the device is classified `file_missing` (or
`size_mismatch` / `short_read`) by `disk_read`, reported to the engine, and pruned out
of the DMA before a stale byte is copied. **False presence costs a recompute, never
wrong bytes.** `directory_synced: false` in a spill answer is therefore the normal
state, not a fault. (`sync=` stays as a capability for any caller that genuinely
wants a durable file; the spill tier does not use it.)

**The journal's own filesystem: `VLLM_E2_TIER_DISK_JOURNAL_DIR`.** The spill journal
(`disk-spill-journal-<rank>.jsonl`) is the only map from a KV identity to a file on
the store, and it cannot be rebuilt. Unset, it lives in `VLLM_E2_TIER_DIAG` as before;
set it to a path on storage that **survives a reboot** (e.g.
`/home/user/kv-tier/diag`) if you want the restore index to come back with its
entries. On 2026-10-05 a reboot destroyed a journal that lived in `/tmp` while 107 GB
of KV files survived on the RAID array, and the index came up at `entries: 0`. The
boot line prints `journal_dir=tmpfs|disk|unset|unknown|unwritable`, `GET /plugins/kvtier/disk`
answers `journal.fs`, and a tmpfs journal logs one warning per boot
(`journal_tmpfs_warnings` counts them). It is a warning, never a refusal: a
degraded-but-loud tier beats a silent one. **Only the journal moves** - the spill
arm file stays in the diagnostics directory, because an arm must die at a reboot (a
cold box booting with the tier armed exposes the restore path to live traffic with
nobody having asked). The arm now fails closed on its own contents as well: it
records the kernel boot id and its epoch with an `epoch_scope`, and a reader refuses a
file naming a different boot, with a reason that says `epoch mismatch` or `boot id
mismatch` (`arm_file_boot_refusals` in `arm_state`).

**The named path is created for you, and a journal that cannot be written is never
silent.** A path you set that does not exist yet used to mean every journal append
failed quietly: the KV file committed, the journal recorded nothing, and the index
never learned the file existed - the 2026-10-05 failure again, only self-inflicted,
and with no reboot to explain it. `disk_index.journal_path()` now creates the
journal directory (mode `0750`, matching the store's shard directories; `0775` would
be a new convention, not the existing one) on the way to the first line, once per
directory, and the writer still costs one `write()` per line. A directory that
cannot be created or written does **not** stop a spill - a logging problem does not
get to cost the whole tier - it makes the payload say so instead: the boot line
reads `journal_dir=unwritable` (deliberately a different word from `unknown`, which
means "we could not tell"), `journal.writable: false` with `journal.directory_error`
naming the errno, `journal.write_errors` counting the lines that never landed, and
`journal.last_error` holding the one-line reason. Those fields are on
`POST /disk {"op":"status"}`, on the snapshot's `disk_restore.index`, and in
`disk_index.section()`; `why_no_restore` names the same blindness in prose, because
"`entries: 0` over a store that is not empty" is the symptom and the cause is what
an operator needs. A failed creation is retried when a cheap `access` shows the path
has become writable, so a `chmod` mid-boot heals the tier without a restart.

`spill-one` is **relayed**: the EngineCore has no KV memory, so the ask becomes one
small file under `VLLM_E2_TIER_DIAG` that every rank answers for its own slot on its
next step, each writing its own receipt. The bytes are queued on the spill thread -
the receipt says `queued` and names the file it will land in, and `ls` is the proof.
`ls` is read-only and bounded (walk caps, `limit`, and a response-byte ceiling that
trims and says so).

**The restore path exists (`disk_restore.py`, since 0.2.14) and it does not touch the
hash index.** A prefix hit is resolved in the scheduler's
`cached_block_hash_to_block`, and only scheduler-side store-completion publishes into
it (the fork calls `_process_store_completion` at `manager.py:1028-1032` on the
deployed in-tree-fixed build, at `manager.py:988-992` at the fork HEAD 10b8e07 this
suite's fixture is generated from). A restore therefore does not try to make a lookup
hit: on a host-RAM shortfall the SCHEDULER offers the SSD's continuation as external
tokens, appends the CPU slots it allocated to the fork's own pending-hit tuple, and
the fork's `update_state_after_alloc` / `start_load_kv` do the pinning and the DMA.
A restored block helps exactly the request that paid for it; the next request reads
the file again (~10 ms per 3,200-token span against ~3.2 s to recompute it). The
worker rebuilds each name with its **own** digest, and a disagreement is a counted
failure, not a guess.

**A fresh boot offers nothing, and requests have to ask.** Two switches, both
dev-verb only, both per boot, neither reachable by an environment variable:

```bash
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"offers"}' $B/disk   # read
curl -s -X POST -H "x-kvtier-dev-token: $TOK" \
     -d '{"op":"offers","enabled":true,"scope":"optin"}' $B/disk             # arm offers
```

`enabled` is this boot's offer switch and it starts **false**: arming the spill tier
says bytes may be *written*, not that requests may be *served* from disk. `scope`
decides which requests once it is up - `optin` (the default) serves only a request
that asked for it, `all` is the whole-workload measurement posture. On 2026-10-05 a
spill arm raised for a write experiment was enough to put 123,474 un-requested
160,000-token offers in front of live traffic; these two switches are why an arm is no
longer one of those things.

`optin` rides a field the engine already has, so nothing new has to be plumbed. Send it
at the TOP LEVEL of the chat-completions body:

```json
{"kv_transfer_params": {"kvtier_disk_restore": true}}
```

That is the shape this engine reads: `kv_transfer_params` is a declared field of the
chat-completions request (`entrypoints/openai/chat_completion/protocol.py:143`), which
the engine folds into `extra_args["kv_transfer_params"]` there at `:717-719`, and
`vllm/v1/request.py:118-119` lifts it onto the `Request` our marker reads. Wrapping it
in `extra_args` yourself is NOT the same thing on this build: chat-completions declares
no `extra_args` body field and the request model ignores unknown fields, so
`{"extra_args": {...}}` is dropped in silence rather than refused. Only
`/v1/responses` and `/v1/completions` take `vllm_xargs`. The proxy passes either shape
through untouched, so it is the engine leg that decides - checked against the deployed
tree, not against upstream.

Strictly the JSON boolean `true` - `"true"`, `1` or an absent field mean "not asked",
and an unasked request is answered by the fork's own lookup with no journal refresh, no
index query and no allocation. `restore.skipped_no_marker` counts those, so "nothing
was offered because nobody asked" is readable and is never mixed up with a refusal.

**A failed disk load is never offered twice.** The engine recovers from a KV load
failure by resetting the request and rescheduling it, which asks
`get_num_new_matched_tokens` again - and the journal has not changed, so an amnesiac
restore answers with the same offer and the same failure. That loop is what starved
the box. `suppress_load` fuses the **request** (named exactly by
`connector_output.failed_recving`) and the **span** (its first block's key, so a
replayed prompt under a fresh request id is refused too), on any of: a failed load,
invalid block ids on a step that carried our slots, a load that outlived
`RESTORE_DEADLINE_SECONDS`, or a request returning at or before the point where it was
last offered. `suppressed_after_failure` is the refusal; `suppressed_hits` /
`suppressed_spans` / `suppressions` say which half fired. The table is bounded
(`MAX_SUPPRESSED_REQUESTS`, oldest-first), because a fuse that grows without limit is
its own leak.

**The scheduler refuses while any live rank refuses to serve.** A worker's rank file
publishes `disk_restore.armed` = *will this rank service a disk load*, and
`worker_proof()` is AND over the live ranks: one disarmed rank closes the whole
scheduler, because the load event spans every rank and the blocks the refusing rank
passes through stay un-filled while the fork has already committed to them. The
refusal is `reason_kind=scheduler_armed_workers_disarmed` with `ranks_armed=0/4`, in
`gate()`, in the boot line and in the counters. `installed` alone is not proof - that
is the field that lied on 2026-10-05, when four ranks were fitted, none was armed, and
the offers kept coming.

**`restorations` cannot rise without a read.** The fork reports a load finished for its
own host-RAM blocks too, so a finished report is not evidence of a restore. A
completion counts as a restoration only when an applied disk read can be shown -
this process's counter, or the sum the live worker rank files attest - and otherwise
counts as `no_read`. `verdict=CONFIRMED` needs both, which is why the witness line
carries `reads_applied=` and `no_read=`. The witness is emitted once per
**(verdict, reason-kind)**, never once per step: keying it on the sentence re-issued
it 1,048 times a second on the incident boot, and `witness_suppressed` counts the
withheld lines so a quiet log is not mistaken for a dead module.

**Why nothing was restored is one HTTP call.** `GET /plugins/kvtier/disk` (and the
`disk_restore` section of `/snapshot`) reports the SCHEDULER's own verdict - the layer
fitted in that process, the gate and its reason, the index size and its source, the
worker proof, and every offer refusal with its meaning - under `restore`, with
`restore.why_no_restore` naming the first missing link. The boot's **posture** is in
that section as fields, not only as text in the boot line: `offers_enabled`, `scope`,
`marker_key`, `offer_cap`, `ranks_armed`/`ranks_seen`, and the fuse counters
`suppressed_hits`/`suppressed_spans`/`suppressed_requests`/`suppressions`. A rank file's
`disk_restore.installed: true` describes a WORKER and cannot start a restore; only
the scheduler's half can, and `BOOT[auto] ... restore=YES|NO restore_gate=open|closed`
is that verdict at boot.

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
the way through every verb - while **the tier itself is behind a switch that is
OFF at boot** (see *Booting the tier* below).

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

### Booting the tier: the boot brings it up working (`VLLM_E2_TIER_DEV_BOOT_INERT`)

> **A tier-env boot ends with the tier ARMED and GUARDED, without a human in the
> loop - and it arms nothing it cannot verify.**

The install cannot land when the plugin *loads*: `load_general_plugins()` runs at
`core.py:217`, but the Scheduler - and with it the connector and the offload
manager - is not constructed until `core.py:260`, so `register()`'s install
attempt is *always* refused at boot (`cls is None`). For a while the answer was
"then the boot stays inert and a human arms it", which is safe but not a system:
every boot needed a curl, and a forgotten curl was a tier that silently parked
nothing. Both halves are now handled at the one moment the class exists:

* **`BOOT_ARMED = False` still holds at plugin load.** Nothing is armed before
  there is anything to arm, so no code path can ever arm an unpatched engine.
* **The install no longer waits for a verb.** `register()` fits a hook on
  `KVConnectorFactory.create_connector` (`factory.py:43`). When that returns a
  `SimpleCPUOffloadConnector` carrying a `scheduler_manager`, the hook runs the
  whole install against the class the connector module just imported
  (`simple_cpu_offload_connector.py:21`) - the first instant that class exists,
  still inside `Scheduler.__init__` (`scheduler.py:164`, reached from
  `core.py:260`), long before the first request - and it passes the manager's
  LIVE `kv_cache_config`, so the geometry guard gets a real check at boot rather
  than `checked: false`. The order inside `install()` is **wrappers FIRST, then
  the backport**, so a refused or exploded patch still leaves "wrappers fitted +
  disarmed", which gates *both* routes to the selector (see *The kill switch*).
  `pr56810.connector_hook` says what happened (`fitted`, `tier_connectors`,
  `worker_role_no_manager`, `installs_run`, `errors`, `last`), and
  `pr56810.patched_before_first_request` is the one-word answer.
* **Then the BOOT runs: guard, then arm.** `backport.boot_from_connector()` is the
  whole sequence, and it is strict in one direction only - it will not arm what it
  cannot verify:

  1. **Guard mode is installed and armed.** Its wrappers are *composed* over the
     backport's and the switch's (`nested`), never put under them, so there is no
     window in which a store or a lookup can run unguarded.
  2. **The tier is armed** - only if the install reported `installed: true` (the
     all-or-nothing verdict: every required method patched, every kill-switch
     wrapper fitted), **and** the live-object check passed on the connector's own
     `scheduler_manager` (the method *objects* it resolves are the ones carrying
     our markers, so an instance built before the patch was fitted cannot pass),
     **and** the guard is a layer of every entry point the class actually has.
  3. Anything else leaves the tier disarmed and puts the reason in
     `pr56810.boot.refusals` - never a silent no.

  Guard before arm is not a style preference: arming the tier without the guard
  means the first real tier bug is an EngineCore death instead of a recorded
  decline, and the arm gate therefore treats "the guard refused this method" as a
  reason not to arm.
* **One line says all of it.** `pr56810.boot.line`, also written to the engine log
  at WARNING so it survives without curling anything:

  `BOOT[auto] backport=YES guard=YES ram_tier=ARMED disk_tier=DISARMED tier=ARMED faults=0 guard_over=... not_covered=...`

  `tier=` is the **RAM/host tier** and stays exactly as it was - the restart verifier
  greps it as a PASS criterion - while `ram_tier=` repeats that verdict under an
  unambiguous name and `disk_tier=` reports the **disk spill tier**, whose arm file
  lives on tmpfs and is gone after every reboot. A boot can therefore legitimately
  print `tier=ARMED` next to a restore refusal reading "the disk tier is disarmed":
  two tiers, two arms, one line. The two named fields are printed *before* `tier=` so
  that a `grep -oE 'tier=[A-Za-z_]+' | tail -1` still lands on the bare field.

  `mode` is `auto` or `inert` and says which env posture ran; `refused=` names each
  gate that failed with its reason; `not_covered=` lists the methods the guard does
  **not** wrap (methods the fork's class does not define, or ones it inherits from
  outside the tier module) so the line never implies more coverage than it has;
  `faults=` is the guard's fault count at that instant, repeated as `faults_now`
  when you read the payload later.

```bash
# 1. boot the engine (tier env set, VLLM_E2_TIER_DEV_BOOT_INERT unset). Expect
#    BOTH lines in /tmp/vllm-serve.log:
#    "kvtier-dev-plugin: connector-creation hook fitted on KVConnectorFactory..."
#    "kvtier-dev-plugin: BOOT[auto] backport=YES guard=YES ram_tier=ARMED disk_tier=DISARMED tier=ARMED faults=0 ..."
#    and NOTHING else to do. `grep 'BOOT\[' /tmp/vllm-serve.log | tail -1` is the
#    whole boot verdict.

# 2. read it back. This no longer PATCHES or ARMS anything (the boot did), it
#    CONFIRMS it: expect installed: true, armed: true, boot.mode: "auto",
#    boot.ran: true, patched_before_first_request: true
ssh llm 'curl -s -H "$H" "http://127.0.0.1:8000/plugins/kvtier/tier?op=status"' \
  | python3 -c 'import sys,json;p=json.load(sys.stdin);b=p["pr56810"]["boot"];print(p["armed"],b["mode"],b["backport"],b["guard"],b["tier"],b["faults_at_boot"])'

# 3. prove the boot was CLEAN, not merely UP. A guard-arm means a tier exception
#    is now a DECLINE, so a broken tier looks healthy from `armed` alone.
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/guard \
  -d '{\"op\":\"captures\",\"limit\":5}'" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["count"],"faults");[print(c["method"],c["exception_type"],c["message"][:80]) for c in d["captures"]]'

# 4. send traffic.
```

Steps 2 and 3 are *confirmations*, not the mechanism. The verbs remain the
backstop - `ensure_installed` re-fits the hook and re-runs the install on `/tier`,
`/snapshot` and `/policy/reload`, which is what covers a process whose connector
factory was imported only after the plugin loaded, or a manager class that
appeared later. A boot where `armed: false` survives step 2 with the mode `auto` is
NOT "nobody has armed it yet": it is a refusal, and `boot.refusals` names the gate
(the install, the live-object check, or the guard's coverage of the class).

The deliberate shortcut in that design is worth naming: the guard covers the nine
entry points it knows and refuses to wrap what it cannot own, and the boot treats a
method the class does not define as *not covered* rather than as a failure -
otherwise a fork that renames or inherits a method would take the whole tier down.
That is why the BOOT line prints `not_covered=` instead of a bare `guard=YES`.

An `arm` is refused, never forced, by the boot or by a verb, and the refusal kind
tells you which gate you hit (`arm_block.kind`, `state.kind`, `disarm_reason`):

| kind | meaning | what to do |
|---|---|---|
| `incomplete_install` | a required method was refused (a fingerprint drift: the fork moved that function); names the method(s) | fix/re-record the source, then arm again |
| `geometry` | the guard vetoed this config: the patch would skip a group that is still `prefix_cacheable=True` | do **not** arm; boot without the tier or fix the geometry |
| `no_manager` | no `SimpleCPUOffloadScheduler` exists in this process - the tier connector is not configured, or the engine is not built yet | there is nothing to arm here; check the tier env of the engine you reached |

And because `armed: false` on a tier-env boot is now an *exception to explain*,
every payload says so where an operator will read it: `pr56810.boot` (the verdict
and its reasons), `pr56810.boot_default_armed`, `pr56810.boot_is_inert`,
`pr56810.arm_requires`, and `pr56810.operator_note` - a sentence that answers "the
engine booted with the tier on, why is it parking nothing?". The `/tier` verb
repeats `armed`, `boot_default_armed`, `operator_note` and a `next_step` at the top
level, next to `applied`.

### The kill switch (runtime, no restart)

```
curl -s -X POST localhost:8000/plugins/kvtier/tier -H "authorization: Bearer $TOK" \
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

### Every boot with the tier env set: the order traffic goes in

> **The boot arms the tier itself when the install verifies; the read-back proves
> it and the fault ring proves the boot was clean, not merely up.**

```bash
# 1. read the boot verdict. The BOOT installs, guards and arms (the connector-
#    creation hook runs all three when the tier connector is built, with the live
#    KVCacheConfig); this curl only CONFIRMS it - and it is still the read that
#    proves the claim. Expect installed:true, armed:true, boot.mode:"auto",
#    boot.backport/guard/tier all true, patched_before_first_request:true
ssh llm 'curl -s -H "$H" "http://127.0.0.1:8000/plugins/kvtier/tier?op=status"' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);b=d["pr56810"]["boot"];print(d["install"]["installed"],d["armed"],b["mode"],b["tier"],b["refusals"],d["operator_note"])'

# 2. prove it is CLEAN, not just armed. Guard mode is armed by the boot, so a tier
#    exception is a recorded DECLINE: `armed:true` alone cannot tell you the tier
#    is doing work. A non-zero count here means the tier is alive and refusing.
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/guard \
  -d '{\"op\":\"captures\",\"limit\":5}'" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["count"],"faults");[print(c["method"],c["exception_type"],c["message"][:80]) for c in d["captures"]]'

# 3. only a REFUSAL needs a verb now. If step 1 says armed:false with mode "auto",
#    read boot.refusals - it names the gate (install / live-object check / guard
#    coverage). To arm by hand once the cause is fixed:
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/tier \
  -d '{\"op\":\"arm\"}'" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["applied"],d["armed"],d.get("reason"),d["next_step"])'

# 4. now let traffic in.
```

Step 1 is a confirmation, not the patch or the arm: `install_connector_hook()` runs
the install inside `Scheduler.__init__` (`core.py:217` plugin load -> `core.py:260`
Scheduler -> `scheduler.py:164` `create_connector` -> first request), so an engine
that has never been spoken to is still a patched, guarded, armed engine. If step 1
answers `installed:false`, that is a finding, not a to-do: either the hook never saw
a tier connector in this process (`pr56810.connector_hook`: `fitted`,
`tier_connectors`, `other_connector`, `last`) or a method was refused
(`refused_methods`, a fingerprint drift). If it answers `installed:true, armed:false`
on an `auto` boot, `boot.refusals` says whether the live-object check or the guard's
coverage of the class stopped the arm - and in both cases the injector is what keeps
the engine alive, which is the point of gating rather than hoping.

What a disarmed tier now shields, stated where you will read it: BOTH callers of
`_select_eager_blocks_to_store` are gated - the three per-step entry points AND
the connector's finish path (`request_finished_all_groups` ->
`_queue_finished_eager_store`, `manager.py:1166` -> `manager.py:1177`), which used
to reach the selector ungated on every finished request. A disarmed tier is
therefore a real gate, not an intent - provided the wrappers exist, and the hook
is what makes them exist at boot rather than at first contact. The two cases the
wrappers genuinely cannot reach are the worker process (no `scheduler_manager` at
all, and it runs none of these methods) and a tier connector that is not the
object `create_connector` returns (a `MultiConnector` child, reported as
`other_connector`).

### Engine crashed / crashes on the FIRST request

On this geometry the first-request crash is the *unpatched* selector being reached
by one of its two callers. Since both callers are gated and the boot installs the
patch at connector creation, a crash here means one of exactly three things:

1. the plugin was not mounted at all (no `kvtier-dev-plugin:` line in the log -
   check *Verify it is really mounted*), so nothing gated or patched anything;
2. the tier connector was never built by the factory (a nested connector, or a
   tier wired up some other way), so the hook had nothing to install onto -
   `pr56810.connector_hook.last` says what it saw;
3. the tier is ARMED while `installed: false`, which is impossible by
   construction (`set_armed` refuses it, naming `incomplete_install`) - if you
   ever see it, that is the bug to report, not a thing to work around.

It is NOT what a correctly disarmed-but-armed-tier looks like - see the next
section for the inert posture, which now requires the opt-out.
Confirm it in one line (the tier state lives in the engine; the read also re-fits
the hook and re-runs the install, so it is the repair as well as the diagnosis):

```bash
# installed=false or refused_methods non-empty => the fix is incomplete.
# armed=false there is the injector protecting you, not a bug; armed=TRUE with
# installed=false must be impossible and is the thing to chase.
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot' \
  | python3 -c 'import sys,json;p=json.load(sys.stdin)["pr56810"];print({k:p[k] for k in ("installed","armed","refused_methods","auto_disarmed","disarm_reason","boot_default_armed")})'
```

Read the last fault. The engine log is `/tmp/vllm-serve.log` (the journal copy
under systemd does NOT feed `logs.db`). The crash line to grep, and the
backport's own install/arm verdicts:

```bash
# the assert the incomplete fix lets through:
ssh llm "grep -a 'not divisible by tokens_per_hash' /tmp/vllm-serve.log | tail -3"
# did the hook get fitted, did the install run at connector creation, was it
# refused? was the tier force-disarmed or an arm refused? did it boot inert (the
# expected line on every boot with the plugin enabled)?
ssh llm "grep -aE 'BOOT\\[(auto|inert)\\]|connector-creation hook (fitted|NOT fitted)|install run at CONNECTOR CREATION|connector-creation install was NOT completed|PR #56810 backport (INSTALLED|NOT installed)|REFUSED to arm|KV tier (ARMED|DISARMED)|boot posture (AUTO|INERT)' /tmp/vllm-serve.log | tail -12"
```

If guard mode was armed there is also a structured capture of the fault (guard
mode is OFF at every boot; there is no capture unless someone armed it):

```bash
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/guard \
  -d '{\"op\":\"captures\",\"limit\":2}'"   # capture.decline, ...geometry.groups_failing_divisibility
```

### Tier env is set but nothing is parked / `armed: false` after a boot

**On an `auto` boot this is a finding, not the expected state.** The boot arms the
tier itself once the install verifies, so an engine that booted with
`KV_OFFLOAD_SIZE` / `VLLM_USE_SIMPLE_KV_OFFLOAD` set and answers `armed: false` was
*refused*, and `pr56810.boot.refusals` names the gate that refused it. (The inert
posture is now a choice: `VLLM_E2_TIER_DEV_BOOT_INERT=1` produces it, and the BOOT
line says `BOOT[inert]` so the two are never confused.) So:

- `armed: false`, `boot.mode: "inert"` -> **normal, and ordered**: the operator
  asked for a manual boot. POST `{"op":"arm"}` when you want the tier.
- `armed: false`, `boot.mode: "auto"`, `boot.refusals` naming the guard -> the
  guard could not wrap an entry point the class has (a foreign owner marker, or an
  attribute that is not callable). The tier is unguarded, so arming it would be
  arming a crash; fix the ownership question the reason names.
- `armed: false`, `boot.mode: "auto"`, `boot.refusals` naming the live-object check
  -> the connector instance the engine is holding does not resolve to the patched
  methods (it was built before the patch was fitted). Read `pr56810.live_check`.
- `installed: false` -> the connector-creation install did not complete in this
  process. Read `pr56810.connector_hook` FIRST: `fitted: false` means the factory
  was not importable when the plugin loaded (the next verb re-fits it),
  `other_connector` means the factory returned something that is not the tier
  connector (a `MultiConnector` child), `worker_role_no_manager` is the normal
  worker-process answer. If the hook DID run and the install still failed,
  `refused_methods` names the fingerprint drift, and the tier is disarmed with
  both selector routes gated - park nothing, do not arm, fix the source.
- `armed: true`, `boot.tier: true`, but nothing parked -> read the guard's captures.
  An armed tier under an armed guard declines instead of crashing, so "armed and
  idle" is exactly what a live bug looks like from the outside.
- `armed: true` with `installed: false` -> impossible by construction; if you
  ever see it, that is the bug to report.

Confirm which one you are looking at:

```bash
ssh llm 'curl -s -H "$H" "http://127.0.0.1:8000/plugins/kvtier/tier?op=status"' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);b=d["pr56810"]["boot"];print(json.dumps({k:d[k] for k in ("armed","boot_default_armed","operator_note","next_step")}, indent=1)); print(json.dumps({k:b[k] for k in ("ran","mode","backport","guard","tier","refusals","faults_at_boot")}, indent=1)); print(d["install"])'
```

### Runtime kill: make the tier a strict no-op, NO restart (one curl)

Also the state the engine starts in: `disarm` is how you get back to the boot
posture without a reboot.

```bash
ssh llm "curl -s -X POST -H '$H' http://127.0.0.1:8000/plugins/kvtier/tier \
  -d '{\"op\":\"disarm\"}'"
# prove it: armed=false, and the connector now reports no pending work
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot' \
  | python3 -c 'import sys,json;p=json.load(sys.stdin)["pr56810"];print("armed=",p["armed"])'
```

Disarmed, the connector reports `(0, False)` for prefix hits, hands the workers
an **empty** `SimpleCPUOffloadMetadata` (so `worker.py:266,289` copy nothing),
claims no pending stores, and queues nothing on the finish path - the engine
serves from GPU KV alone. **`disarm` is never refused** (it is the safe
direction). What it does NOT cover, verbatim from `disarm_limits()` in the
payload:

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
- `boot` - **what the boot decided, in one place**: `ran`, `mode` (`auto`, or
  `inert` because `VLLM_E2_TIER_DEV_BOOT_INERT` is set) and `mode_why`, the three
  verdicts (`backport`, `guard`, `tier`), `refusals` (one string per gate that
  blocked the arm, each naming its reason), `not_covered` (entry points the guard
  does not wrap, so `guard: true` never over-claims), `nested` (methods where the
  guard was composed over another layer instead of replacing it), `live_check`,
  `manager`, `faults_at_boot` and `faults_now`, `note`, and `line` - the same
  single line the engine log carries. `ran: false` means this process has no tier
  to boot (worker rank, or a `MultiConnector`); `line` is written anyway, with the
  reason.
- `boot_default_armed` / `boot_is_inert` - the tier is disarmed at *plugin load* on
  purpose (nothing exists to arm yet), so these describe the load-time posture, not
  the boot's final state. `armed: false` after an `auto` boot is a refusal to read
  in `boot.refusals`; `operator_note` says which situation you are in, and the
  `/tier` payload adds `next_step`.
- `arm_requires` / `arm_block.kind` - arming needs the fix fully installed AND
  the geometry guard passing; a refusal kind is `geometry`, `no_manager` (no
  manager in this process - nothing to arm) or `incomplete_install`.
- `refused_methods` - the method(s) that were refused; each is also
  `methods.<name>.patched: false` with the reason in `methods.<name>.reason`.
- `geometry` - the geometry guard verdict (`geometry.ok`, and
  `geometry.cacheable_skipped` names any group it would wrongly skip).
- `disarm_reason` / `auto_disarmed` - whether the injector force-disarmed, and
  which refusal caused it: `"geometry"` vs `"incomplete_install"`.
- `arm_block` - if present (`{kind, reason}`), arming is currently refused and
  this says why; `null` means arming is allowed.
- `disarm_limits` - what the kill switch cannot undo. Both entries about the
  switch itself now read in its favour: the wrappers are fitted FIRST by the
  install the connector-creation hook runs, and the finish path is `GATED`, so
  both callers of the store selector are covered.
- `connector_hook` - how the patch actually landed: `fitted`, `calls`,
  `tier_connectors`, `worker_role_no_manager`, `other_connector`, `installs_run`,
  `errors`, `last` (a one-line verdict for the most recent connector built).
- `patched_before_first_request` - true once the hook has run an install in this
  process; the one-word answer to "was this engine patched before traffic?".

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
  wrappers; the PR #56810 replacements plus the four kill-switch wrappers on the
  offload scheduler and the one wrapper on `KVConnectorFactory.create_connector`
  (the hook is what installs them, at connector creation, and the same hook then
  arms the guard and the tier - `BOOT_ARMED = False` only means nothing is armed at
  *plugin load*, before there is a class to arm); a trace ring and its two
  scheduler wrappers while ARMED (removed on disarm); the guard wrappers while armed
  or traced
  (restored by
  identity on disarm); and the worker's diagnostics file when
  `VLLM_E2_TIER_DIAG` names a directory. Every one of those is opt-in, and the two
  that change engine behaviour (`guard`, tracing) say so in every payload.
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

1413 tests, no engine and no vllm required (`.venv/bin/python -m pytest -q`; the
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
answers under both dispatch shapes, and that the mount point is exactly the twelve
`/plugins/kvtier/*` paths (a list derived from `config.ROUTES`, so a new route
cannot go unlisted, with the openapi surface checked against it too). Three of them
(`test_real_socket_*`) drive a real uvicorn listener over a real TCP connection -
one from loopback with no token, two from a source-bound non-loopback client socket
on this host's own LAN address (skipped on a box with no such address).

The raw-byte dump (`tests/test_blockdump.py`) is the one that tests bytes rather
than claims, because the question it serves is a byte-level one: a real `ctypes`
buffer stands in for the pinned arena, so a known pattern written by the test must
come back through the response file identical, at an offset, with a matching
`sha256`. Around that it pins the discipline: every cap REFUSES with the numbers
named (a range that crosses the slot, a length over the cap, eight 200 KiB ranges
that would total 1.5 MiB, a whole-block read of a 2 MiB slot) and nothing is ever
clipped; a fresh boot does no dump work even with `VLLM_E2_TIER_DIAG` set for the
rank-file writer (a pending request file with no arm file is never opened, never
consumed, and answered by no file); non-perturbation is proven by a fake CPU block
pool that records every attribute read on itself and records NONE during a dump,
with `ref_cnt`, the free-queue order and the hash index identical after; a region
whose `data_ptr` raises and a region whose `__getattr__` raises are both contained
and BOTH consume the request, so one bad region is answered once instead of retried
every poll; the wrapper hands back the fork's object IDENTICALLY and does not
consume a generator result (the class of bug that would make the engine free
nothing); the arm expires on the EngineCore's monotonic clock and in the file a
worker reads, and a garbage/oversized arm file fails closed to disarmed; the
capability marker's three states are asserted in `section()`, in `snapshot` and in
`/status`, including that `/status` says `..._unobserved` rather than inventing
`armed: false`; and the wire bound on `length` is pinned equal to the engine's own
`MAX_BYTES_PER_REQUEST` rather than to a second copy of the literal.

The spill wiring (`tests/test_disk_store.py`) is the one that tests naming, because a
wrong name is the failure that cannot be repaired after the fact. The worker's
fingerprint is assembled from a fake handler that carries the fork's own names
(`cache_config.prefix_match_unit`, `kv_cache_groups[i].kv_cache_spec
.prefix_cacheable`, `parallel_config.decode_context_parallel_size`) and the fork's own
`get_none_hash_seed()` - including its non-hex `DEFAULT_NONE_HASH_SEED`, the value a
real unseeded boot returns; a field that cannot be read is refused BY NAME with the
paths tried and never defaulted, and the rebuild is rate-limited so an unreadable
config does not re-walk fifteen attribute paths every step. Two ranks of one
configuration get two digests and cannot form each other's file name, a changed model
id / hash algorithm / group layout moves the digest, and the seed moves it too. The
pairing rules are pinned as a refusal, not a heuristic: a length mismatch refuses the
whole step naming both numbers, while one `""` key costs its own slot and the rest of
the batch still spills. A completed store writes the region's own bytes - read out of
a real `ctypes` buffer by the same copy the worker performs - at the path the
fingerprint dictates, through the fork's `bind_connector_metadata` and
`get_finished`, with the fork's return value handed back by identity and the sibling
`worker_diag`/`blockdump` layers still installed underneath. `store_fingerprint` is
proved to be a cross-check and nothing else: `""` (today's engine) keeps the step, a
disagreement refuses it naming both digests, a non-digest is a separate counted
reason. An unpatched metadata is the AUTOUSE default of the file, so "correct on the
engine that exists" cannot be inherited from another test by accident, and the
no-engine-shape case reports `unknown` rather than `absent`. Every bound is asserted
as a counted event: the per-step hand-off cap, the retention cap (oldest step evicted
first, and completing it is an `unretained_event`, never a spill), the once-per-event
de-duplication, the scan cap that skips a huge completed map instead of copying it.
`spill-one` is exercised end to end across the relay - request file, rank poll, the
receipt's `queued` verdict and its `expected_path`, and the file really present with
the slot's bytes after a drain - with a second poll refusing to answer the same ask
twice, a malformed or wrong-version or expired request counted and ignored, and a
rank refusing an ask whose fingerprint is not its own. Nothing raises: every door is
opened with a tensor whose attributes explode, a `None` config, an `op` that is an
object; `KeyboardInterrupt` from the fork still escapes; and an AST walk asserts the
module imports only the standard library and its own package (function-level imports
included), so the engine is reached by NAME and never at import time.

The lookup ledger (`tests/test_lookup_ledger.py`) is the one that proves the
composition rules instead of asserting them: the cap and the monotonic `dropped`,
lookups and adopts interleaved in a different order and joined by request id, the
adopt that never had a lookup, the lookup whose request was skipped, a re-offer after
a failed allocation, a re-install that must leave exactly ONE guard layer (it walks
the marker chain and asserts the fork's function is still underneath), a
backport-replaced method with `backport._layers_of` saying `["guard", "backport"]`,
a foreign owner marker refused with the refusal QUOTED in the arm reason (not a
generic "no tier class"), a disarmed ledger that never even evaluates its arguments,
a trace disarm that cannot blind it, and a fork method that raises - which must
propagate exactly as it would unwrapped, because the ledger is not guard mode. The
parked-KV index reading is pinned in `tests/test_snapshot.py`: the live 664-free /
4372-entries pair reported side by side, per-group counts decoded from the packed key,
the scan and group caps honoured, a `len()` that raises named as that rather than as
a missing handle, and three separate ways to get a `reason` instead of a number.

Per-request miss attribution (`tests/test_miss_attribution.py`, 45 of them) is
pinned on the numbers and on the honesty of the gaps: the query length, the cap the
fork puts on it and the residue, each carrying the unit it claims; the derived
first-miss position and the walk property it is derived from; every verdict in the
enum reachable or the file fails; a clamp group named ONLY when the probe's minimum
actually binds the served length (and never the parked-nothing scratch ring);
`unknown` emitted when the CPU pool cannot be measured, instead of a confident
`root_missing`; that nothing recorded keeps an engine block alive (weak references,
because holding one is the PR #56810 bug class); the ledger bounded under 5,000
lookups; a `requests` read REFUSING over its byte budget instead of silently
returning a shorter list of "the requests that missed"; the measured retention split
into joint and aggregate from UNEQUAL per-group spans (a fixture where the groups tie
cannot tell a minimum from a maximum); `free != empty`, including the pinned-but-
parked spans; the free-order witness both CONFIRMING and REFUTING, and reporting
`INCONCLUSIVE` when it has seen nothing; the coordinator and pool wraps being
instance-level and restoring the stored originals; a ratio of scheduler steps against
blocks being refused outright; the file mirror env-gated and stopping at its cap; and
that a GENERATOR free list (`manager.py:1297`) reaches the engine's own
`free_blocks` INTACT - observing it first would leave the engine freeing nothing,
which is how a read-only witness would have become a leak; and ten source mutations
run against `_decide` and the retention arithmetic
(`offered > 0` -> `>=`, a residue sign flip, an alignment-gate sign flip,
`first_miss_span` off by one, joint `min` -> `max`, dropping the referenced spans
from the parked total, the deep-probe budget ignored, and the generator guard
        removed) each one shown to break a test.

The backport and the kill switch (`tests/test_backport.py`, 67 of them) drive the
fork's own
method text (fingerprints match, so the real install path runs) against the
production geometry, with a poison-pill `resolve_block_hashes` that reproduces
the assert - so "the patch works" and "the unpatched selector crashes" are
behavioural claims. The boot posture has its own section: a **fresh subprocess
that only imports the plugin is disarmed** (the fixtures reset state, so they
cannot prove the first instant of a real process); the boot-time install refusal
leaves the tier inert and reports `arm_block.kind: "no_manager"`; the cold-boot
`status` -> `arm` sequence installs then arms; arming seeks its own proof when a
caller skips `ensure_installed`; each refusal kind is refused, named, and
explained in the verb payload; a class-level install cannot forget a geometry
veto a live config already gave; a disarmed tier produces no store/load work as
observed **through the wrappers** (the fork's own no-op shapes, with the engine's
methods never called).

The invariant itself is proven rather than asserted, and paired with a control so
it cannot be vacuous: with the backport REFUSED (the store selector left
unpatched, the worst install there is) and the tier disarmed, a spy on
`_select_eager_blocks_to_store` records **no call at all** from
`request_finished_all_groups` or `build_connector_meta`; the mirror with the fix
installed and the tier armed sees both routes reach it (and skip the ring); and
the control - the same class with NO wrappers fitted - raises the production
assertion on both routes. The install ORDER is pinned by an install that explodes
halfway through the patch (all four wrappers are already on the class, the tier is
disarmed, both routes stay gated) and by a geometry veto, which now patches
nothing and therefore MUST get the gate. And the boot itself is tested without a
single HTTP verb: `register()` fits the factory hook, a fork-shaped
`create_connector(config=..., role=..., kv_cache_config=...)` returns the very
connector it built, and by then `installed` and every wrapper marker are true -
including a LIVE geometry check, which a class-only install can never do. The
worker role is a `no_manager` answer, not an error; a hook that raises is counted
and the engine still gets its connector; the fork's own `create_connector`
exceptions propagate untouched; a foreign hook marker is refused; and guard mode
composes with the new wrapper in BOTH fit orders, handing it back by identity.

Bringing the tier up working, unattended, is its own file
(`tests/test_auto_boot.py`, 30 of them), because the claim is a sequence and a
sequence needs ordering proven: a clean install arms guard mode and the tier with
**no verb anywhere in the test**; the guard is armed *before* the tier is armed
(spy-recorded, not asserted about, so a reorder fails here); a partial install, an
instance the live-object check cannot make safe, and an entry point the guard
refused to wrap each refuse to arm and name themselves in the payload and in the
one BOOT line; `not_covered` reports what the guard does not wrap without vetoing
the arm - including the production shape where `take_events`/`reset`/
`update_connector_output` are inherited from the connector base module and can
never carry a guard layer (an earlier gate read "attribute exists, therefore guard
it" and would have refused to arm on every real engine); an inherited limit does
not relax an *owned* unguarded method, which still blocks; a second boot composes
into exactly one guard layer with the backport's and switch's layers intact
beneath it; the opt-out resolves with the same falsy vocabulary as every other
flag and says so in the payload; and a raising tier method becomes a recorded
decline that survives a disarm, a re-arm and the next boot's `faults=N`.

The debugger verbs add six files:

| File | Tests | What it pins down |
|------|-------|-------------------|
| `tests/test_kvconfig.py` | 40 | the divisibility table against the fork's assert (`kv_cache_utils.py:2888`), a Mamba group's different `hash_block_size`, the QSA ring recognised by class name, `prefix_cacheable` never inferred from `host_rotation`, the geometry-vs-manager disagreement, every cap honouring `dropped` counts, a missing operand reported as `unknown` and never as safe |
| `tests/test_kvinspect.py` | 41 | block/request inspection against a production-shaped fake: the packed block-hash decode, `in_cached_map` via the tier's own dict, `coverable_blocks` and `binding_term` recomputed by hand, the `ids_cap`/drop reporting, every "does not exist" answer naming what does |
| `tests/test_trace_ring.py` | 36 | the ring's fixed capacity and wrap accounting, `read` never walking O(depth), the `gap_records` a slow poller gets, `clear` not resetting `seq`, arming installing and disarming removing the observers, a disarmed ring costing no allocation and no wrap, the step observer returning the fork's value untouched |
| `tests/test_guard_mode.py` | 49 | every declined method returning the fork's documented no-op shape (and the one method that must re-raise), `BaseException` never swallowed, the metadata rebuild falling back to a re-raise if the fork renames its class, captures bounded with the full text only via the `captures` op, `guard_mode: true` in every payload, disarm restoring the originals by identity, tracing keeping the wrappers alive after a disarm, **no env var that can arm it** (the boot does, and the opt-out can only turn that off) |
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
# -> 10 passed
```

It builds a real `SimpleCPUOffloadConnector` (both roles) and a real
`SimpleCPUOffloadScheduler` with the fork's own test builders, installs the
wrappers on the real `KVConnectorBase_V1`, and asserts the pool ends up capped,
that no id at or above the cap can leave the real free queue, that the fork's own
consumer did not raise, and that `SimpleCPUOffloadConnector` never grew its own
copy of either method. The fork's own tier suite is unaffected: `84 passed`
(`tests/v1/simple_kv_offload/{test_scheduler,test_kv_events,test_hip_mem_ops}.py`).

## Known limits

* **Guard mode is a development facility, and the boot arms it.** Armed, it turns a
  tier exception into a record plus a decline, so a real bug can hide behind a
  decline that looks like normal "nothing to do" - and `build_connector_meta`'s
  substitute is an *empty* metadata, which means the worker does nothing that step.
  That is the price of a boot that cannot be killed by a tier bug, and the trade is
  deliberate: the alternative is an unguarded `build_connector_meta` taking the
  EngineCore down on live traffic. The counterweight is evidence, not silence - the
  fault ring survives the boot, `boot.faults_at_boot`/`faults_now` compare across
  it, and the ring is named in the boot line itself. **Read
  `{"op":"captures"}` after every boot**: a tier that is armed, up, and doing no
  work is a tier that is declining. It needs the token to reach, echoes
  `guard_mode: true` everywhere, is opt-out-able only as a whole
  (`VLLM_E2_TIER_DEV_BOOT_INERT`), and must be disarmed before a production boot.
  The same reasoning covers tracing: the ring is bounded and the observers are pure,
  but they are wrappers on the scheduler loop.
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
