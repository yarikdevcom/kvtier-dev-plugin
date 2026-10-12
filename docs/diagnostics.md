# HTTP surface: the route table, argument rules and payload fields

Moved from `README.md`. Reference for the dev route surface: the route table and
the read-only commands an operator runs, the shared argument validation, and the
fields `/snapshot` publishes. The operator's emergency procedure ("the KV tier is
misbehaving, right now") stayed in the runbook in `README.md`; the per-module
instruments (`kvconfig`, `kvinspect`, the trace ring, `blockdump`) are in
`docs/reference-tools.md`.

**Arguments.** GET takes them as query parameters, POST as a JSON object body, and
both are validated against the SAME per-route table (`endpoint.ARG_SPECS`), so a
bound cannot be evaded by changing the method. An unrecognised key is a `400` that
never reaches the engine - including a stray `?cachebuster=1` on `/snapshot`. A
**POST reads BOTH forms** (since 0.2.21): `POST /disk?op=arm&seconds=3600` with an empty
body arms the tier, and a key given in the query string AND the body must carry the same
value or the answer is a `400` naming both - there is no silent winner, because until
0.2.21 the POST handlers read the body only and a query-string `op` was **ignored while
answering `ok: true`** (`?op=arm` ran `status`, the arm file kept its mtime, and the
response claimed success). A validated body is forwarded to the engine as ONE dict argument
(`call_utility_async(method, args)` -> `getattr(self, name)(*args)`,
`vllm/v1/engine/core.py:1669-1671`); a call with no arguments still sends none, so
nothing that worked before arguments changed. Which engine method a request reaches
is fixed by the route and can never come from the body.


`snapshot` returns the CPU pool totals/free, GPU pool totals/free,
`pending_stores`, the ten in-flight/queue lengths, the load/store event counters,
the eight `BoundaryStoreStats` decline reasons, scheduler lengths, the plugin's
per-boot `epoch` (proof that two snapshots came from the same boot), `plugin_version`
(the version of the code the ENGINE process imported; the top-level `version` beside
it is the document's SCHEMA and has been `1` all along), the policy's
own verdict plus `decide_us` and `build_us`, the `store_hook` patch report (below)
that says whether the runtime veto is actually live, the `cpu_capacity` clamp
report, and the development sections - `guard_mode` (a top-level copy of the answer,
because a guarded engine must never be mistaken for a production one), `guard` (its
bounded previews), `trace` (the ring's counters and the step observer's), and
`worker_diag` (what this process knows about the per-rank file writer, with
`worker_scope` saying that the truth lives in the files).


## Using it

The whole surface, all under `/plugins/kvtier`. **No route in this table requires a
token any more** (0.2.68 removed it from every route, in every shape): the ONLY gate
is the peer rule - loopback, or an address in `VLLM_E2_TIER_DEV_ALLOW`. And because
`/plugins/*` is **not** covered by vLLM's `--api-key` (that covers `/v1`, `/v2`,
`/inference` and `/cohere` only), the allowlist is not a second line of defence
behind a key - it is the whole defence. The table used to carry a `Token` column
with `off-box only` / `always` per row; it is gone because every answer is now
`none`, and a column of `none` is how a table starts lying.

| Route | Methods | What it does |
|-------|---------|--------------|
| `/status` | GET | local facts; makes **no** engine call, so it answers while the engine is wedged |
| `/snapshot` | GET, POST | the scheduler-side tier view (pools, counters, policy, `store_hook`, `guard`, `trace`) |
| `/kvconfig` | GET, POST | the KV-cache geometry: per-group block sizes, spec classes, prefix-cacheability, and every group's divisibility verdict |
| `/block` | GET, POST | one GPU block by id (and group), plus the tier's CPU slot of the same id |
| `/request` | GET, POST | one request by id: its blocks per group and what the tier could park |
| `/trace` | GET, POST | the bounded event ring: read it, or arm / disarm / clear it |
| `/guard` | GET, POST | guard mode: state, captured faults, arm, disarm |
| `/tier` | GET, POST | the tier's runtime kill switch and the PR #56810 backport's state |
| `/lookups` | GET, POST | the lookup/adopt ledger: read, arm, disarm, verdict |
| `/blockdump` | GET, POST | the bounded raw-KV-byte dump: status, arm, disarm, dump (see `blockdump.py`) |
| `/disk` | GET, POST | the SSD spill tier: status, arm, disarm, `spill-one`, `ls`, `contains`, `identities` (the paged, digest-only per-rank view of the index, 0.2.46 - the other half of the diff against `ls`), `offers` (RAM-first gap fill) and `authority` (every request served from disk), plus `failures` and `shadow` (the read-only byte-witness, see `disk_shadow.py`) and `gpu-shadow` (the GPU-direct shadow writer, see `disk_gpu_shadow.py`) and `restore-verify` (the restore-side byte verifier, see `disk_restore_verify.py`) and `gpu-write` (the GPU-direct write's counters and latch, see `disk_gpu_write.py` - its answer carries `scope`, because the apply point this op arms is code only a worker runs) and `restore-source` (0.2.59: where a restore's destination rows come from, `arena` or the plugin's staging ring; per boot, and its answer carries the ring's pressure beside the posture) and `adoptions` (0.2.61: WHICH request got how many tokens from
disk, newest first, `limit` up to the whole 256-record ring). A mutating `op` on GET is a `405` |
| `/policy/reload` | POST | reload `<package dir>/policy.py`, and adopt the store-policy patch |

```bash
B=http://127.0.0.1:8000/plugins/kvtier

curl -s $B/snapshot | python3 -m json.tool | head -60
curl -s $B/kvconfig | python3 -m json.tool
curl -s "$B/block?block_id=412&group=5" | python3 -m json.tool
curl -s "$B/request?request_id=cmpl-a" | python3 -m json.tool
curl -s -X POST -d '{"request_id":"cmpl-a","ids_cap":200}' \
     $B/request | python3 -m json.tool

# the ring: arm it, work, read it, disarm it (nothing is wrapped while disarmed)
curl -s -X POST -d '{"op":"arm","depth":256}' $B/trace
curl -s "$B/trace?limit=40" | python3 -m json.tool
curl -s -X POST -d '{"op":"disarm"}' $B/trace

# the mutating verb: no header, no token, only the peer rule
curl -s -X POST $B/policy/reload | python3 -m json.tool
```

From an allowlisted machine the SAME calls work unchanged - since 0.2.68 there is
nothing to add to them, off-box or on it:

```bash
# on the engine's own machine, nothing to configure:
ssh llm curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot | python3 -m json.tool

# from 192.168.2.64 / 192.168.2.120, once VLLM_E2_TIER_DEV_ALLOW names you:
B=http://192.168.2.42:8000/plugins/kvtier
curl -s $B/status | python3 -m json.tool
curl -s $B/snapshot | python3 -m json.tool
```

`status` is the place to check what you actually opened: `allow_networks` is the
parsed list (an entry that failed to parse is simply absent from it), and
`loopback_only` answers "is this still closed to the LAN?". It also answers
"is this engine in a development mode?": `guard_mode` and `trace_armed` at the top
level, the same facts again in `dev_state` with their `source`, `epoch` and
`age_seconds`, and `worker_diagnostics` with the directory the worker writer was
told to use. Because `/status` never calls the engine, `guard_mode` is `null` until
this API process has seen an engine answer that says otherwise - it is never a
guessed `false`. It also answers "which code is THIS process running?" with
`plugin_version` - the API server's own loaded version, not the engine's (see
*Which version is running* above).

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

Deep detail: see docs/steering-and-patching.md - "Steering and vetoing the tier at
runtime (`store_hook.py`)" (the section that used to follow this one), and
docs/diagnostics.md for the route argument rules and the `/snapshot` payload fields.


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
# arm it, let it see traffic, then ask the question
curl -s -X POST localhost:8000/plugins/kvtier/lookups \
     -H 'content-type: application/json' -d '{"op":"arm"}' | jq .reason
curl -s -X POST localhost:8000/plugins/kvtier/lookups \
     -H 'content-type: application/json' -d '{"op":"verdict"}' | jq '.counts, .dropped_examples'
curl -s localhost:8000/plugins/kvtier/lookups | jq .entries
curl -s -X POST localhost:8000/plugins/kvtier/lookups \
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
# arm it: no token, since 0.2.68 (the peer address is the gate)
ssh llm "curl -s -X POST http://127.0.0.1:8000/plugins/kvtier/lookups \
  -H 'content-type: application/json' -d '{\"op\":\"arm\",\"cap\":256}'"

# the per-request attributions (optionally &verdict=root_missing)
ssh llm "curl -s 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=requests&limit=32'"
ssh llm "curl -s 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=depth'"   # measured retention
ssh llm "curl -s 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=order'"   # root-eviction witness
ssh llm "curl -s 'http://127.0.0.1:8000/plugins/kvtier/lookups?op=rates'"   # unit-checked ratios only
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

### Cost, bounds, and what it never touches

Measured on this host: 0.07 us disarmed (one branch), 3.4 us median / 4.8 us p99
armed for a hit, 8.5 us including the coordinator observation, and 0.75 ms per pool
walk - memoised to one per second and paid only on a zero-hit lookup. The
attribution rides the ledger's ring (`cap` 16..4096, default 256), holds NO engine
block reference (proven with weak references: holding one is the refcount bug class
PR #56810 addresses), and `op=requests` REFUSES over `MAX_REQUESTS_BYTES` (200 KB,
about 100 attributed records at ~1.9 KB each) rather than silently shortening the
list of requests that missed.

Deep detail: see DEVLOG.md - "Measured retention: three answers 785x apart, resolved by
measuring" and "The free-order witness: a hypothesis, made falsifiable".


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
curl -s -X POST -d '{"armed":true}' $B/guard
curl -s "$B/guard" | python3 -m json.tool
curl -s -X POST -d '{"op":"captures","limit":2}' $B/guard
curl -s -X POST -d '{"armed":false}' $B/guard   # restore
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
* **It has a heartbeat (0.2.45; the beat is supervised since 0.2.47).** The rank file is
  this rank's PROOF OF LIFE, and `disk_restore` refuses to offer a disk restore on a file
  older than `RANK_FILE_FRESH_SECONDS` (900 s). Writing it only from `get_finished` meant
  an IDLE worker wrote nothing, its proof aged out, and the scheduler stopped serving from
  disk for a worker that was perfectly alive - so the worker writer keeps its own beat
  through the SAME write path a step uses, started at `install()` and not at the first
  step. Since 0.2.47 each beat runs in its own bounded thread (budget
  `HEARTBEAT_BEAT_BUDGET_SECONDS` 20 s, at most `HEARTBEAT_MAX_INFLIGHT_BEATS` 3 of them)
  and catches `BaseException`, so one contributor blocked on a lock cannot stop the proof
  and one raising `SystemExit` cannot end it. Since 0.2.51 the beat's DEADLINE and its
  write floor are both kept on `CLOCK_BOOTTIME` and the wait is polled in slices of
  `HEARTBEAT_RESUME_POLL_SECONDS` (5 s), because `CLOCK_MONOTONIC` - the clock CPython's
  `Event.wait()` waits on - does not advance while the box is suspended, and this box
  suspends: see *Silence is not death, part three*. An engine step may
  `wake_heartbeat()` when the supervisor is found behind its own deadline, and never runs
  a beat inline. **Read the state in the rank file -
  `writer.heartbeat.state`** (`beating`, `alive_not_beating`, `started_not_alive`,
  `never_started`, `not_applicable`): `heartbeat_alive` in `snapshot` describes the
  process that BUILT the snapshot - the EngineCore, reached over the utility hop, with
  the API server forwarding the payload unmodified - so for a worker it is `false` by
  construction either way. Hence the state lives in the file. See
  *Silence is not death, part two* below, and
  *An idle engine keeps its disk proof alive*.
* **The schema is the reader's contract**: `version` 1, a per-boot `epoch`, `ts`,
  `role: "worker"`, `rank` matching the file name (the reader calls a mismatch
  `malformed`), `arena.bytes_total`/`bytes_used`/`num_blocks` (plus
  `block_bytes_by_group` = `len(layer_names) * page_size_bytes` per group),
  `counters`, and `writer` prose that states where each number came from.
  Since 0.2.50 the file also carries `plugin_version` - the version of the code THAT
  WORKER imported, read live from `kvtier_dev_plugin.__version__` by the process that
  wrote the file, because a worker answers no route and this is the only place its
  build can be published. Three names in this file look like versions and are three
  different facts: `version` (the FILE SCHEMA, still 1), `writer.plugin` (a TAG whose
  trailing `/1` is part of its shape, not a release number), and `plugin_version` (the
  build). All three survive the byte-ceiling triage: `KEEP_KEYS` sheds a contributed
  section before it sheds the answer to "which build wrote this file?".
  `arena.bytes_used` is a **proven floor** (blocks in flight x bytes per block), not
  the pool's occupancy, and says so; `counters.dropped_cpu_full` and
  `counters.boundary_stored` are deliberately absent because they are scheduler-side
  (`BoundaryStoreStats`) - the reader notes an absent counter and adds zero, which
  keeps the aggregate honest instead of inventing a reading.

Deep detail: see DEVLOG.md - the "Silence is not death" entries (parts one, two and
three) and "The proof that arrived too late: a cold boot's first request found the gate
shut".

