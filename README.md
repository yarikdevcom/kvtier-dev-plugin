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
| `kvtier_dev_plugin/loaded_version.py` | the one accessor for "which code did THIS process load?" - `kvtier_dev_plugin.__version__`, read live, never `importlib.metadata`. It publishes `plugin_version`/`plugin_version_reason`/`plugin_version_scope` into `/snapshot`, `/status` and every rank file |
| `kvtier_dev_plugin/disk_store.py` | **development only**: the worker half of the SSD spill tier - the fingerprint THIS rank builds, the `store_keys` resolver, the store-completion spill hook (composing over `worker_diag`/`blockdump`), and the engine half of the `/disk` verb. Drives `disk_spill.py` (the queue + writer) and `disk_layout.py` (the naming rules). Stdlib only |
| `kvtier_dev_plugin/policy_holder.py` | late-bound holder for the policy module; trial-load + swap, rollback on failure |
| `kvtier_dev_plugin/policy.py` | **the file you edit.** `decide(view)` observes, `decide_store(ctx)` vetoes |
| `kvtier_dev_plugin/store_hook.py` | the runtime patch on the tier's store-decision path: idempotent, non-clobbering, fail-open |
| `kvtier_dev_plugin/backport.py` | the runtime backport of upstream PR #56810 (hybrid-geometry fix), the tier's runtime kill switch, and the `create_connector` hook that makes both land before the first request. Stdlib only |
| `kvtier_dev_plugin/pr56810_sources.py` | the patched function bodies, byte-exact, with the SHA-256 of the source each one replaces. Generated |
| `kvtier_dev_plugin/endpoint.py` | `vllm.endpoint_plugins` entry: the router, the guards, the argument specs, the 504 |
| `tests/` | CPU-only pytest; runs without vllm and without an engine (`fakes_debug.py` holds the shared fake engine structures) |

## Install

### From git - what a serving venv should carry

```bash
venv/bin/pip install "kvtier-dev-plugin @ git+https://github.com/yarikdevcom/kvtier-dev-plugin.git@v0.2.71"
venv/bin/pip check
```

Pin the ref in a serving venv. `v0.2.71` is a tag, and this command was run against
the public repository to confirm it resolves (`pip download --no-deps` of that exact
URL produced `kvtier-dev-plugin-0.2.71.zip`). An un-pinned URL installs whatever the
default branch happens to point at when the clone happens, which for a package that
wraps engine methods at import time is not a version you can reason about later.

`pip check` is not decoration: the package declares NO dependencies on purpose, so a
silent resolution that touches the serving venv's other pins is the failure to watch.

### Editable checkout - the development path

Into the venv that runs the engine, editable so that the file you edit is the
file the engine reloads:

```bash
cd /home/user/Source/experiment/kvtier-dev-plugin
.venv/bin/python -m pip install -e .     # or whichever python runs vllm
.venv/bin/python -m pip check            # must stay clean
```

The difference between the two is not cosmetic: this plugin wraps engine methods
at import time, so an editable checkout means a `git pull` inside that checkout
changes what the NEXT boot imports - which is the point while you develop and the
last thing you want in a venv that serves traffic.

The package declares **no dependencies**: `fastapi`/`starlette` arrive with
vllm, and a `pip install -e .` must not churn the serving venv's package set.
Verified: installing it into an empty venv adds exactly one package.

Nothing activates yet - see the switches below.

## Switches

| Env var | Meaning |
|---------|---------|
| `VLLM_PLUGINS` | vLLM's own allowlist. **Endpoint plugins never load unless named here.** One name (`kvtier_dev`) covers both halves. |
| `VLLM_E2_TIER_DEV` | this plugin's switch. Unset/`0`/`false`/`off`/`no` = inert: no engine methods attached, HTTP answers 503. |
| `VLLM_E2_TIER_DEV_TOKEN` | **NO LONGER A GATE (0.2.68).** It used to be required of the mutating verb, then of every read from a non-loopback peer, then of every shape of the six dev-state routes (`trace`, `guard`, `tier`, `lookups`, `blockdump`, `disk`) - loopback included, so even a plain `GET /trace` needed it. **The plugin no longer reads it for access control at all**, and a value set here is ignored. Setting it is not a risk and not a protection: the boot says so once at `WARNING` (`<VAR> is SET but is no longer read by this plugin`) and `/status` answers `token_gating: false` with `token_note` naming the release. Unset it, because a variable that reads like a control and is not one is how an operator ends up believing a door is shut. What gates this surface now is `VLLM_E2_TIER_DEV_ALLOW` below, and nothing else. |
| `VLLM_E2_TIER_DEV_ALLOW` | extra peers beyond loopback: a comma-separated list of addresses or CIDRs. **Unset/empty = loopback only** (the pre-allowlist behaviour). A malformed entry is logged and skipped - never fatal, never an opening. See *Access model*. |
| `VLLM_E2_TIER_DEV_TIMEOUT` | seconds the HTTP layer waits for the engine before answering 504. Default `5.0`, clamped to `[0.05, 60]`. |
| `VLLM_E2_TIER_DIAG` | the worker diagnostics directory (capability 5). **Unset = the worker writer is completely inert: no directory created, no file written, nothing wrapped in the worker.** Set it to a path both the worker ranks and the proxy host's reader can see. |
| `VLLM_E2_TIER_DISK_JOURNAL_DIR` | where the SSD tier's spill **journal** lives, apart from the rest of the diagnostics directory. **Unset = fall back to `VLLM_E2_TIER_DIAG`** (the pre-0.2.19 behaviour). Set it to storage that survives a reboot: the journal is the only map from a KV identity to a spilled file and cannot be rebuilt, while the KV files it describes outlive it. **The path does not have to exist** - the plugin creates it (mode `0750`, the mode the store's own shard directories get) on the way to the first journal line, once per directory. **Only the journal follows this var** - the spill arm file stays in `VLLM_E2_TIER_DIAG` and dies at boot, because an arm must not survive a reboot. A journal on tmpfs is warned about once and reported (`journal_dir=tmpfs` on the boot line), and a journal directory that cannot be created or written is reported (`journal_dir=unwritable`, `journal.writable: false`, `journal.directory_error`, `journal.write_errors`) - either way it is never a refusal to spill. |
| `VLLM_E2_TIER_DIAG_INTERVAL` | how often a rank rewrites its file, in engine steps. Default `50`, clamped to `[1, 100000]`. A step that *changed* something writes immediately (subject to the floor below). |
| `VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS` | the rate floor between writes, in seconds. Default `1.0`, clamped to `[0, 3600]`; `0` means "no floor". |
| `VLLM_E2_TIER_DIAG_EPOCH` | pin the per-boot `epoch` string the reader uses to prove two facts came from the same boot. Unset = derived once per process and stable for its lifetime. |
| `VLLM_E2_TIER_DIAG_RANK` | pin this rank's id (`PP<n>_TP<m>`) instead of deriving it from the already-imported `vllm.distributed.parallel_state`. An illegal value means **no file at all**, because the reader refuses a file whose `rank` does not match its own name. |
| `VLLM_E2_TIER_DISK_SETTLE_MS` | **how long one apply point may wait for the bytes it is about to apply, in milliseconds. Unset = `5000` since 0.2.32** (it was unset-means-never-wait through 0.2.31, which is why a default boot used to lose every raced read); `0` = never wait, and that is still reachable. It is a FLAT allowance - `min(SETTLE_MS, the load's own remaining deadline)` - and no arithmetic on bytes or throughput takes part in it; see *The restore race, measured, and the wait that stopped guessing* below. It is an allowance and NOT an arm - the restore path still needs the disk root, both arms, this boot's offer switch and a proven worker before it reads a byte. A value that will not parse falls back to the 5,000 ms default, says so in the log and on the payload (`env_problems`), and only that line names the variable that was ignored. |
| ~~`VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S`~~, ~~`VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S`~~ | **DELETED in 0.2.32.** These two fed the settle formula (`SETTLE_READ_BPS_DEFAULT`, `PREFILL_TOKENS_PER_S_DEFAULT`, `SETTLE_SAFETY_MARGIN`, `SETTLE_MIN_MS` are gone with them); the rule is flat now, so there is nothing left for a throughput assumption to compute. Reading them is a no-op - they were guesses, and the box proved it. See *The restore race, measured, and the wait that stopped guessing*. |
| `VLLM_E2_TIER_DISK_READ_THREADS` | **how many threads issue spill reads at once (0.2.24). Unset or `1` = ONE thread**, which is every boot before 0.2.24, byte-identically and pinned by a test. Clamped to `[1, 16]`; anything that is not a positive integer means "the operator said nothing" and is reported in `read.env_problems`. It is a thread count and nothing else - the queue stays FIFO, so dispatch order is submission order. **Raise the pool with it or it buys nothing on a big prompt**: in-flight bytes are bounded by the pool, not by the threads. Since 0.2.67 the count is *also* a runtime lever (`POST /disk {"op":"read-threads","threads":N}`), relayed to the worker ranks through `disk-read-threads.json` and still **per process and per boot** - so this variable remains the boot-time source of truth (a restart restores it on every rank), while `read.threads_source` says which owns the live count in a given process. See *Reading the spill in parallel*. |
| `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` | the bounce-buffer pool's byte ceiling, PER PROCESS. **Unset = `disk_read.MAX_BUFFERS_BYTES` = 96 MiB**, today's value, which holds **8** units of 11,468,800 B (7 of the 12,902,400 B width). This is what actually bounds in-flight work: a buffer is taken at *issue* and only returned at the apply point, so **one offer needs the pool to hold the whole offer at once** - and a big-prompt offer is 10 units, so the default misses it by two. **128 MiB (134217728) is the smallest setting that holds a whole 10-unit offer on both widths.** HOST-RAM COST: the pool is per process and every engine rank is a process, so 128 MiB here is 512 MiB of resident bytes on a 4-rank boot. Clamped at 8 GiB and says so (`pool.clamped`). See *Reading the spill in parallel*. |
| `VLLM_E2_TIER_DISK_CEILING_BUDGET_SECONDS` | **how many seconds one capacity-enforcement pass may spend (0.2.29). Unset = `disk_spill.CEILING_PASS_BUDGET_SECONDS` = 1.5 s.** The pass runs on the spill WRITER thread - the only thing draining the 256-job / 256 MiB spill queue - and before this release its worst case (100,000 directory entries read twice, 20,000 `stat`s, up to 4,096 unlinks) was bounded in files but not in TIME, so a full store could stall the queue and DROP spills at the moment the disk was fullest. When the budget expires the pass stops early and says so (`spill.ceiling_measurement.deadline_hit`, `plan_truncated`), and the next committed spill continues - a partial pass is this tier's normal shape, not an error. **`0` is refused** (a pass that does nothing reads as a configured ceiling that is not one) and a value that will not parse means the shipped default, reported in `env_problems`. See *The eviction clock is a touch clock, and nobody's neighbour is capacity*. |
| `VLLM_E2_TIER_DISK_EVICT_THREADS` | **how many threads `unlink()` at once inside one ceiling pass (0.2.29). Unset or `1` = ONE, the serial behaviour of every release before 0.2.29, and the shipped default.** Clamped to `[1, 8]`; the pool is built once per process (a restart is needed to change it, and `evict_pool.evict_threads_requested` beside `evict_threads_configured` shows when they disagree). It changes WHICH thread unlinks and nothing else: the plan is PARTITIONED by shard directory across the workers, so no path is ever attempted twice and the pass sheds the same files and bytes at 1 worker and at 4. **It defaults to 1 because parallel unlink measured ~38% SLOWER here** - see *The eviction clock is a touch clock*. |
| `VLLM_E2_TIER_DISK_INDEX_ENTRIES` | **how many spill identities the in-memory disk index may hold (0.2.45). Unset = `disk_index.MAX_INDEX_ENTRIES` = 262,144** (about 90 MB at the measured 342 bytes per entry, and the dict only grows to the identities this rank actually indexed). Before this release the cap was a hard-coded constant, and the store outgrows it in normal operation - a 148 GiB ceiling store runs ~30,000 files per rank and the index maps each one. When the cap is reached the index now evicts **LRU-by-touch** (the identity that has not been used leaves first, not the one that was learned first) and every drop is counted by REASON: `entries_dropped_lru`, `entries_dropped_unverified`, `entries_dropped_unparsable` balance `entries_dropped`, and `entries_dropped_file_present` names the one that matters - the index forgetting a file that is still on disk. Not an on/off switch: it bounds a memory cost, so lowering it trades cache knowledge for RAM and never disables the tier. A value that will not parse, or is not above 0, means the shipped default and is reported in `env_problems`. See *Silence is not death* below. |
| `VLLM_E2_TIER_MISS_LOG` | the miss ledger's OPTIONAL file mirror: a path to append one JSON line per tier lookup to. **Unset = no file, no `open()`, no I/O.** Env-only on purpose: an HTTP argument naming a path the engine appends to would be a write surface this plugin does not have anywhere else. Bounded by `MAX_MIRROR_BYTES` (32 MiB), where it STOPS and counts the stop (`mirror_stopped`) rather than filling a disk. |
| `VLLM_E2_TIER_DISK_SHADOW` | **the shadow verifier (0.2.36), and it is OFF by default.** Unset (or `0`/`false`/`off`/`no`) = nothing reads a GPU block and `disk_shadow` reports `enabled: false`. `1`/`true`/`yes`/`on` = on a completed store, one block in 32 is read back off the DEVICE and compared against the bytes the spill path captured from the arena, and the disagreement is published in `rank-<rank>.json` under `disk_shadow`. It reads REAL KV bytes on the engine's step thread, so the default is off for the same reason `guard_mode` and `blockdump` opt in: a launcher must not be able to turn it on by accident, and a value that is neither an on word nor an off word is OFF *and* reported (`env_recognised: false`) - the direction a typo never overrides. The value is read ONCE at first use (a live boot's environment does not change; the `POST /disk {"op":"shadow"}` latch is the live lever, for the process that answers). It never writes a file, never changes a spill or a restore decision, and its copies are bounded to `MAX_BLOCKS_PER_STEP` per bind window. See *Does the GPU still hold what the arena spilled?*. |
| `VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES` | **the GPU-direct shadow writer (0.2.37), also OFF by default, same grammar.** Unset (or `0`/`false`/`off`/`no`, or anything unrecognised) = no file is ever written by this module. `1`/`true`/`yes`/`on` = for a completed store the shadow verifier has just called `equal`, those GPU bytes are written a second time into a **separate** directory under the live store's own file name, and the two files are compared - the diff is the gate on the GPU -> NVMe route. It is never authoritative: the live store stays the only thing a restore loads, it never opens the store for writing, and a shadow root that *is* the store or is nested in it is REFUSED (`refused_root_conflict`), not merged. Everything it does runs on its own writer thread; the engine's step thread pays a `put_nowait`. The value is read once at first use; `POST /disk {"op":"gpu-shadow"}` latches only the answering process. See *Writing the block twice: the GPU-direct shadow*:. |
| `VLLM_E2_TIER_DISK_GPU_SHADOW_ROOT` | **where those shadow files go.** Unset = `<the live store root>/../gpu-shadow`, a sibling of the store (so one `cmp -r` compares the two trees and the store's own sweep never walks the evidence). Read once per process; validated like any other root, and subject to the same refusal. |
| `VLLM_E2_TIER_DISK_RESTORE_VERIFY` | **the RESTORE-side byte verifier (0.2.39), OFF by default, same grammar.** Unset (or `0`/`false`/`off`/`no`, or anything unrecognised) = no spill file is opened and no GPU block is read by this module. `1`/`true`/`yes`/`on` = a sampled restored entry is compared byte-for-byte: the leading `W_g` bytes of the spill file against the leading `W_g` bytes of the GPU block the restore loaded it into, with the file width and the group width checked first and the tail (the other groups' stale bytes) never compared at all. A disagreement is re-read before it is reported, so `mismatch_stable` (a layout candidate) is separable from `mismatch_gpu_changed_on_recheck` (the engine moved). Since **0.2.40** the same variable arms BOTH roles of the module: the worker's byte comparison above, and the scheduler-side **binding oracle** - which since **0.2.41** witnesses `update_state_after_alloc`, the hook where the fork DECIDES which GPU block each offered CPU block pairs with: the observer captures that pairing's expected file name at the alloc and the emitter compares it against the name it is about to ship (`scheduler_*` counters; a one-slot shift and a group swap are both named findings there; `scheduler_oracle_hook` says whether the witness is actually wired). One variable, because arming half a verifier would be the worst of both: it would publish numbers about a check that is not running. `POST /disk {"op":"restore-verify"}` latches only the answering process and its answer says so. It never changes a restore decision, and the engine's step thread pays a counter and a dict of ints - the file read and the device copy happen on its own bounded thread. See *Do the RESTORED bytes match the file?*. |
| `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE` | **the GPU-DIRECT WRITE (0.2.43, cold start fixed in 0.2.44), OFF by default, same grammar, and the one dev feature in this package that WRITES into live KV memory.** Unset (or `0`/`false`/`off`/`no`, or anything unrecognised) = no copy is issued, no memory is page-locked, no thread starts and no counter moves. `1`/`true`/`yes`/`on` = at the restore apply point, for a sampled entry whose bytes were just memmoved into a CPU slot, the plugin copies that slot into the GPU window the entry names (`gpu_block_id`) on the tier's own `load_stream`, BEFORE the fork's body runs - and reads the window back once before its own copy and once after, so the copy is witnessed rather than assumed. The fork's own arena-to-GPU copy still runs afterwards on that same stream and keeps the last word on every byte, so a wrong addressing guess here is a wrong WITNESS and not a wrong answer. The readback is compared against the SPILL FILE (the leading `W_g` only, the tail never), never against the CPU slot it copied from - `readback_mismatch_stable` is the single counter that names THIS module's addressing as the candidate, and `readback_mismatch_gpu_changed_on_recheck` is the expected race from which nothing follows. Pinned readback rows are allocated only on this module's own thread and released only when the tier's load watermark passes the entry's event; a row whose DMA state is unknowable is leaked (`buffers_leaked`) and never handed out again. Sampled at one entry in `sample_every_n_applied`, at most one write per step, and never an allocation on the step thread. The value is read once at first use; `POST /disk {"op":"gpu-write"}` latches only the answering process, and for this feature that process has no apply point at all - its answer says so in `scope` instead of letting a zero read as an all-clear. Since **0.2.44** an armed process starts the writer's own thread when the module registers (with the write attempt asking for it again as a safety net), because pinned rows can ONLY be allocated by that thread and 0.2.43 started it on the line after a row had already been taken - so it never started, and the module wrote nothing with `errors=0`. Read `liveness.state` to ask whether the witness is alive (`off` / `armed, no apply point seen in this process` / `armed but never wrote` / `writing`); `passes`, `pool_refills` and `buffers_allocated` all zero with `applies_seen` climbing is a writer that never ran. See *Writing the window ourselves, and watching it land*. |
| `VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE` | **how many CPU slots of the host-RAM arena's block pool an offer must NOT take (0.2.38). Unset = `disk_restore.CPU_SLOT_RESERVE` = 64, the value every release before 0.2.38 used.** An offer allocates its *destination* slots out of the same pool the RAM tier parks its own blocks in, and is clamped to `(free - reserve - boundary slots) // slots-per-span` spans - so on a shrunk arena the reserve is an offer-size lever, not a footnote: at the operator's 166 free slots per rank on the wide PP1 pair, reserve 64 leaves **98 spans (313,600 tokens)** and reserve 16 leaves **146**. `0` is **honoured** ("an offer may take the arena's last slot" - it leaves the RAM tier nothing to park, see the section). A non-integer or a negative falls back to 64 and logs once (`DISK[reserve] IGNORED`); there is deliberately **no ceiling**, because a reserve larger than the arena correctly declines every offer as `no_cpu_slots` instead of oversubscribing. The value in force, its origin (`env` / `default` / `ignored`) and the spans it leaves are published in `restore.limits`, `restore.offer_pool`, `last_offer_pool` and the boot line's `cpu_slot_reserve=64(default)`. See *Sizing the CPU-slot reserve an offer leaves alone*. |
| `VLLM_E2_TIER_DISK_RESTORE_SOURCE` | **WHERE A RESTORE'S DESTINATION ROWS COME FROM (0.2.59): `arena` (the default) or `bounded` (`bounce` is the same posture by its deprecated name).** `arena` is the behaviour every boot before this had: an offer allocates one CPU slot per restored span out of the host-RAM arena's `cpu_block_pool`, so a whole-prompt restore (82 spans + 4 boundary states = 86 slots) is what forces the pinned arena to be sized for the WORST PROMPT, permanently - the wall on a small-RAM box. `bounded` keeps that allocator and adds an ADMISSION CAP on the staging rows per rank: the plugin never owns a row, so the fork's release path cannot strand one. That is not a simplification of an ownership design, it is the reason one cannot be had - the fork's `_cleanup_load_request` releases a destination row's LAST reference (`v1/simple_kv_offload/manager.py:1346`, reached through the release the plugin's 0.2.52 guard installed at `disk_restore.py:8551`), so a row the plugin allocated for a load is back on the pool's free list the moment that load finishes, and a plugin that kept a private list of "its" rows after that would be claiming rows the allocator is free to hand to the next restore. `POST /disk {"op":"restore-source","source":"bounded"}` flips it at runtime (per boot, `survives_restart: false`; `"source":null` clears back to this variable) and an unrecognised value is REFUSED and changes nothing, at the route's enum and again in the setter. THE SOURCE IS PINNED PER OFFER, on the load record, so a mid-flight flip cannot stage half a run one way and half the other - the half-staged span is the wrong answer with no exception. A `bounded` run longer than the cap is CLIPPED (`bounded_by: ["staging_rows"]`) and one the cap cannot pay for is REFUSED with `no_staging_rows` - never waited for, because a wait in the step thread is what kills the engine, and a refused restore costs one slow request. A cap charge is returned when its own load ENDS, whichever way it ended (a positive `finished_recving`, a load error, an abort, an expiry, a cap eviction), counted apart as `returned_rows` / `recovered_rows`; there is NO timer and no age-based reuse, because reusing a row whose copy has not landed is a wrong answer that raises nothing. See `restore_source` and `staging_cap` in `/disk`. |
| `VLLM_E2_TIER_DISK_RESTORE_STAGING_ROWS` | **how many staging rows one rank may have charged to loads at once (0.2.59). Unset = `disk_restore.RESTORE_STAGING_ROWS` = 32.** `VLLM_E2_TIER_DISK_RESTORE_RING_SLOTS` is still read, as the name this shipped under first; the canonical name says what the number is, because the number is no longer a row count the plugin holds. The arithmetic for the default: 32 rows x 12,902,400 B (the widest rank window on the live 4-GPU box, PP1) = 412.9 MiB per rank = **1.61 GiB across four ranks**, inside the operator's 2 GiB ceiling - and that ceiling is now a CLAIM rather than the accident of an arena size, which is the one thing a cap does that `arena` clipping (0.2.38) cannot: `arena`'s offer length floats with the arena's free list, so a boot can serve today and not tomorrow at the same setting. **0 is a real posture and is honoured**: no offer can be staged, so `bounded` refuses every offer up front with `no_staging_rows` - the safe direction for a bound whose job is to make offers smaller. Negative, fractional and non-numeric are grammar errors: they fall back to 32, are named once in the engine log (`DISK[cap] IGNORED`) and stay visible in `staging_cap`'s `problems`. The cap never reserves anything, so `staging_cap` publishes `cap_rows` / `charged_rows` / `peak_charged_rows` / `in_flight_loads` and a `binding` verdict (`staging_cap` / `read_pool` / `neither_yet`) answering the only question that decides whether to grow it: has the cap clipped or refused more offers than the read pool has refused for capacity. The rows a charge describes are the arena's, so an arena that cannot pay still refuses with `no_cpu_slots` (`arena_shortfall`), and `over_cap` counts a boot where the charge exceeds the declared cap - a bug, since pricing is the only thing that raises a charge. |
| `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS` | **THE STEP-2 LOAD-PAIR PRUNE (0.2.49), OFF by default, same grammar.** Unset (or `0`/`false`/`off`/`no`, or anything unrecognised) = the restore apply point rewrites the load lists exactly as every release before this one did, byte for byte, and none of the `prune_*` counters move. `1`/`true`/`yes`/`on` = at the same apply point, an entry whose read SUCCEEDED and was adopted is REMOVED from BOTH `load_cpu_blocks` and `load_gpu_blocks` at the same index, in the one rewrite the module already owned, so the fork's `copy_blocks` no longer DMAs that arena slot to the GPU window. **This is the first flag in this package whose failure mode is a wrong ANSWER rather than a slower one**, for two reasons the design is built around: `copy_blocks` pairs `src[i]` with `dst[i]` POSITIONALLY (`cuda_mem_ops.py:227`, descriptors built at :239-248) and cannot see that one list got shorter: with EQUAL lengths a one-sided prune silently shifts every later block into the wrong destination, and with UNEQUAL ones `total` comes from the CPU side alone (:248) so the copy loop at :257 slices GPU descriptors past the end of a shorter array. Neither shape is an error the fork reports - which is why `prune_load_pairs` is the only door a pair leaves through, refuses a call whose two sides disagree (`one_sided_prune`, `length_mismatch`, `out_of_range`, `over_cap`, `group_delta_mismatch`, `destination_drift`) and hands back NO lists on a refusal; and a group's entry binds to `block_ids_by_group[g][num_computed // group_block_sizes[g] + i]`, so an index derived from tokens instead of from the scheduler's own index-aligned identity mis-names a `boundary_state` group - which is why the index set is read out of the bound jobs and never computed. An entry whose read FAILED is never in the prune's population (the apply point drops it and reports its GPU block as a load error exactly as it always has, so the engine recomputes: `prune_skipped_failed_entries`, and the load-error id set is identical with the flag on and off). **STEP 3 CHANGED THE SAFETY ARGUMENT (0.2.52), and it changed the flag's meaning**: an entry is now a candidate only if its index is in the set `disk_gpu_write.observe_apply` reported as ENQUEUED (`written_indices`). This is a MEMBERSHIP test and not a comparison of counts, and the difference is the whole point - the old arithmetic (`witnesses < candidates`) called a boot with three sampled writes and thirty adopted entries a gap of 27 and named nothing. Because the gate runs BEFORE the prune, `prune_unwitnessed_entries` is now a TRIPWIRE that cannot move: a non-zero reading means the membership test was broken, and every index it names is being served from uninitialised KV. `prune_emptied_steps` still counts the steps left with nothing for the fork to copy. WHY THE GATE IS THE ENQUEUE AND NOT THE READBACK: a readback verdict lands on the writer's daemon thread up to `JOB_MAX_AGE_SECONDS` (30 s) after the enqueue, while the prune runs SYNCHRONOUSLY inside `start_load_kv` and is applied in the same step - a synchronous decision cannot wait on asynchronous evidence, so the enqueue gates and the readback is the KILL-SWITCH that retires the flag later. CONSEQUENCE FOR AN OPERATOR: this flag alone does NOTHING unless `VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY` is also set, and it says so - every step is refused as `not_delivered` (a name of its own, because "no disk work this step" and "disk work with no writer" are opposite stories) and `prune_uncovered_entries` counts the destinations the fork is still carrying. Read once at first use; only a WORKER has an apply point, so an EngineCore answers `enabled: false` forever. See *Pruning the pairs an adopted restore no longer needs (`disk_restore.py`, 0.2.49)*. |
| `VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY` | **DELIVERY MODE (0.2.52), OFF by default, same grammar, and a LAYER on `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE` rather than a second writer.** Unset = the writer keeps its shipped witness posture (one entry in `sample_every_n_applied`, one write per step, a `min_interval_seconds` floor) and costs what it always cost. `1`/`true`/`yes`/`on` = EVERY applied entry of the step gets its own host-to-device copy, so the entries the prune may remove are a SUBSET of the entries this module enqueued. The three throttles are read from ONE dict (`limits_in_force`) so they cannot move independently - lifting the sampler and leaving the step cap at 1 is a "delivery" that writes one entry and counts `throttled_step` for the rest, reading as a broken feature. A fourth limit is NOT a throttle and is not lifted by this flag: the pinned pool holds `MAX_PINNED_ROWS_PER_REGION` rows and a witnessed write holds one row per lane, so it admits exactly ONE in-flight record. In delivery mode the COPY needs no pinned row (its source is the CPU arena slot, not a pinned buffer), so an unavailable row costs the WITNESS and never the write - `delivered_unwitnessed` counts copies that went unread back, and `no_pinned_buffer` stays a refusal this mode never takes. With the writer flag unset this flag is INERT and `delivery_mode.reason` names the missing flag. **THE KILL-SWITCH is what makes the mode safe to run**: a `readback_mismatch_stable` is same-execution evidence about our OWN copy (both readbacks were enqueued on our stream around our copy inside one `start_load_kv`, and the file side is the file this restore read), so it latches for the life of the process - the three throttles come back and `disk_restore` REFUSES every later prune as `kill_switch_fired`, restoring the fork's copy. `readback_mismatch_gpu_changed_on_recheck` (another stream wrote the window) and `readback_mismatch_unrechecked` (no before-state, so nothing is attributable) are counted in `kill_switch_declined` and NEVER latch: a switch that fires on a race is a feature that turns itself off for no reason, which is the 0.2.48 lesson from `disk_gpu_shadow` in a new lane. `kill_switch_fired + kill_switch_declined == readback_mismatch` always. Read once at first use. See *Delivering the window ourselves (0.2.52)*. |
| `VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS` | **THE ONE BOUND ON A DELIVERY STEP, in seconds (0.2.57). Unset, or anything that is not a finite number above 0 = `disk_gpu_write.DELIVERY_APPLY_TIMEOUT_SECONDS` = 10 s.** A delivery copy's source is the fork's **pageable** `cpu_kv_caches` (`v1/simple_kv_offload/worker.py:232`, no `pin_memory`), so `copy_(non_blocking=True)` does **not** return early: the whole step's bytes are synchronous PCIe on the engine's step thread, which is how a 198,400-token restore on 2026-10-09 parked a worker for minutes and EngineCore died on `RPC call to sample_tokens timed out`. **This is deliberately ONE generous wall-clock timeout and not a per-entry budget**: 0.2.56 shipped a 0.25 s step budget *and* a 64 MiB byte budget, and both are gone - the tier's house rule (learned on the disk read/write path) is one generous timeout instead of fine-grained caps, and a 0.25 s cap would defer ~57 of a normal 62-block restore, which turns the prune this mode exists to feed into a no-op. A guard that disables the feature is not a safety feature. Why 10 s: the engine's `No available shared memory broadcast block found in 60 seconds` is a WARNING CADENCE (`VLLM_RINGBUFFER_WARNING_INTERVAL`, `vllm/envs.py:20`) and the reader never gives up, while the deadline that actually ends the process is `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` = **300 s** (`vllm/envs.py:242`, passed by `sample_tokens`) - so 10 s sits ~30x inside the only real threshold and ~60x above the ~0.15 s a full 735 MB restore measured. When it expires the rest of the step's entries are DEFERRED (`delivery_deferred_entries` / `delivery_deferred_steps`) and stay out of `written_indices`, so the prune cannot remove them and the fork's copy delivers them: **a stop costs a missed optimization and never a hole**. `delivery_refused_entry_too_wide` (0.2.56's byte-budget refusal) is **REMOVED**, not left at zero. Read PER APPLY POINT, so a drop-in takes effect without a restart; clamped at `DELIVERY_APPLY_TIMEOUT_MAX` = 60 s and says so (`delivery_mode.timeout_clamped`), because past the engine's own warning cadence the number describes nothing real; `0` is refused rather than honoured (it would defer every entry of every step and read as "configured"). The value in force is published at `delivery_mode.timeout_seconds`, and whether the bound has ever bitten is `delivery_mode.step_seconds_peak` read beside it. See *Delivering the window ourselves (0.2.52)*. |
| `VLLM_E2_TIER_DEV_BOOT_INERT` | **the boot opt-out.** Unset (the default) = the boot brings the tier up *working*: at connector creation it installs the backport, arms guard mode, and arms the tier. Set (`1`/`true`/`yes`/`on`) = the old posture: patch, then stay disarmed and unguarded until a verb says otherwise. Same falsy set as `VLLM_E2_TIER_DEV`. It is the only env var that touches either facility, and it can only turn arming **off**. |
| `KVTIER_DISK_BOOT_ENABLE` | **the SSD tier's boot default (0.2.33), and it is ON by default.** Unset (or `1`/`true`/`yes`/`on`) = the boot that arms the RAM tier also **arms the disk spill tier (as a latch since 0.2.35: no expiry) and raises this boot's serve posture**, so the engine comes up `serving_as=disk_authority` with no POST at all. `0`/`false`/`off`/`no` = the posture of every release through 0.2.32: disarmed, `serving_as=off`, and the two `/disk` POSTs are the only lever. It is a **disable switch**, the same shape as `VLLM_E2_TIER_DEV_BOOT_INERT`: it can only take a boot back to manual, and `VLLM_E2_TIER_DEV_BOOT_INERT` (which arms nothing at all) outranks it. The decision is computed fresh from this variable at connector creation and from **nothing else** - it reads no arm file and no state an earlier boot wrote, so a `disk-spill-arm.json` surviving from a previous boot still arms nobody. **Since 0.2.35 there is nothing to renew and nothing to expire**: the arm is a latch, so what the boot raises stays raised until a human disarms it - and a human's `{"op":"disarm"}` (or `{"op":"authority","enabled":false}`) latches a veto that even a SECOND boot inside the same process cannot arm over. One line at boot prints the value, where it came from and how to switch it off (`KVTIER disk boot default=...`); `disk_restore.boot_default` is the payload. See *The SSD tier's boot default*. |
| `VLLM_E2_TIER_DEV_MAMBA_ANCHOR_PATCH` | **THE RUN-END ANCHOR PATCH (0.2.64), ON by default, and the SECOND runtime-patch target.** Unset (or `1`/`true`/`yes`/`on`) = the boot wraps `MambaManager.reachable_block_mask` (`single_type_kv_cache_manager.py:1610-1669`, a pure classmethod) so a cached range's LAST block is marked reachable: `if mask and not mask[-1]: mask[-1] = True`. That one mark is what carries an align-mode group's boundary state into the hash → offer → spill pipeline at the end of a run, which is the difference between a run that can be anchored from NVMe and the `no_boundary_anchor` refusal the spill-side counters measured at a 20.0% orphan rate. `0`/`false`/`off`/`no` = **`env-refused`**: nothing is fitted, no half-patch state exists, and the mask answers exactly as the fork's own body does. The grammar is `KVTIER_DISK_BOOT_ENABLE`'s (default ON, and a value in neither vocabulary stays ON and reports `recognised: false`). Like the #56810 backport it is **fingerprint-pinned**: the live body's `inspect.getsource()` SHA-256 is pinned from the live-tree reconstruction (`vllm-live-repro@live`, file md5 `52a3c5ad…`, identical on `built` and in `vllm-dev`), and a stale fingerprint is a REFUSAL naming both hashes and the pin's source, never a guess. **This target is spill-side only**: it can add data to the store and has no door to an arm, a serving posture or a request path – pinned twice, behaviourally and against this package's own AST, beside the AST-pinned serve-switch tests. The boot line carries the verdict as one token so an installed boot and a silently-absent one cannot be confused: `MAMBA_ANCHOR[patched|refused|disabled|env-refused]` (`disabled` = this process is not the scheduler, which is normal in a worker). Counters: `mamba_anchor.anchor_offers_total` (the marks the patch made, in the snapshot's `disk_store` section), `disk_store.anchor_spills_completed` (the anchor files the ranks stored, in the cross-rank rollup) and `disk_restore.restore_anchor_hits_total` (run ends found anchorable on a patched boot). The three are the two ends of one pipe and are deliberately NOT a per-block causal chain: a spill hand-off carries `(slot, block-hash)` pairs and no token position, so the per-block join from the mask to a file does not exist in-process. See *The run-end anchor patch (`mamba_anchor.py`, 0.2.64)*. |
| `VLLM_E2_TIER_DISK_BOOT_DRAIN_SECONDS` | **how many seconds the boot's journal drain may take (0.2.65). Unset, or anything that is not a finite number above 0, = `disk_index.BOOT_DRAIN_DEADLINE_SECONDS` = 45 s.** THE HARD CEILING ON A STARTUP COST, and the reason it is a deadline and not a byte count: a journal that grows without bound must never be able to stall a boot. Sized against the box's own measurement (the drain reads ~2.2 MB/s of journal, so 45 s permits ~99 MB of reading - ~57 MB of journal at the measured 1.73x line-cap amplification, ~2.3x the ~25 MB the incident boot had to read). Tonight's ~25 MB converges in ~11-12 modelled seconds, which is what the boot pays and a request no longer does. `0` is REFUSED, not honoured: a deadline of 0 drains nothing and reads as "configured". The value in force is `restore.index.drain_limits.boot_deadline_seconds` with its origin in `boot_deadline_source` (`default` / `env` / `env_ignored`), and a refused value reports its complaint in `problems` instead of defaulting silently. See *Who pays for the drain (`disk_index.py`, 0.2.65)*. |
| `VLLM_E2_TIER_DISK_BOOT_DRAIN_BYTES` | **how many bytes the boot's journal drain may read in total (0.2.65). Unset, or anything that is not an integer above 0, = `disk_index.BOOT_DRAIN_MAX_BYTES` = 64 MiB.** The 32 MiB it replaces could not have covered the incident boot even with an infinite deadline: a bounded pass reads a whole `MAX_READ_PER_REFRESH`-sized window but applies only the lines its line cap fits, and the cursor advances past APPLIED lines only, so the tail of every window is read again by the next pass - measured in this repo at **1.73x** on the incident's shape (28.0 MB read for 16.2 MB of journal, 4 ranks x 110k lines). 64 MiB covers 25 MB x 1.73 = ~43 MB with ~1.5x headroom. Past either bound the boot stops, `boot_drain.stopped_at` says WHICH bound, and the remainder keeps draining on the request path exactly as before. Grammar only (`config.disk_index_drain_env`); the policy is `disk_index.drain_limits()`. |
| `VLLM_E2_TIER_DISK_INDEX_REFRESH_BYTES` | **the STEADY-STATE cap: bytes one refresh after the boot drain may read across all journals (0.2.65). Unset, or anything that is not an integer above 0, = `disk_index.MAX_READ_PER_REFRESH` = 512 KiB - the number every release before this one used, UNCHANGED by 0.2.65 on purpose.** The offer path's work bound is what stops a burst of requests becoming a burst of reads; widening it is a decision an operator makes after the boot drain has proved itself, not a side effect of this release. It is the trickle's rate: `boot_drain.lag_bytes / this` is the number of refreshes a boot that stopped early still owes. Published as `restore.index.drain_limits.read_bytes_per_refresh` with `read_bytes_source`. |
| `VLLM_E2_TIER_DISK_SPILL_ON_STORE` | **MIRROR MODE (0.2.66): does a store completion alone queue the slot for NVMe? Unset is ENABLED; `0`/`false`/`off`/`no` close the door at boot (`source: "env"`); a word in neither vocabulary stays ENABLED and reports `recognised: false` with its complaint in `problems`.** ON is the default because the door only ever ADDS spill work and has no door to a serving posture, an arm or a request path - so the direction a typo can move in is the direction that keeps spilling, and `0` is the rollback. The cost was measured first: `slots_offered 3,076` against `spill_queued 2,556` on this boot's rank files, so mirroring everything is ~1.20x today's write volume. A mirror offer that would take the rank past its ceiling is refused AT THE DOOR (`disk_spill.mirror_headroom`, counted as `ceiling`) and never written. Read it at `spill_on_store` in `/disk op:status`, `/snapshot` or a rank file, with `spill_on_store_verdict()`'s `source`/`raw`/`recognised`/`problems`. See *The arena is a buffer, not a store (0.2.66)*. |
| `VLLM_E2_TIER_DISK_SHUTDOWN_FLUSH_SECONDS` | **how many seconds the spill queue may be waited on at STOP (0.2.66). Unset, or anything that is not a finite non-negative number, = `disk_spill.SHUTDOWN_FLUSH_SECONDS_DEFAULT` = 15 s; `0` means NO drain is installed at all; anything above `FLUSH_BUDGET_SECONDS_MAX` = 30 s is CLAMPED to it and says so (`clamped: true`).** A wait on the stop path needs a ceiling more than a listing does: the call runs on a thread that can be holding a suspend, so a value typed as `3600` for an experiment must not hold one for an hour. Sized against the box's real budgets (`TimeoutStopSec=90`, `--shutdown-timeout 60`, `KillMode=control-group`) - but read the caveat: on an idle box the executor's worker ladder (`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`, default 5 s, + 4 s) binds first at ~9 s. Fail-open: an expired budget costs one WARNING line and the stop proceeds. Read it at `spill.shutdown_flush` in `/snapshot` and `/disk op:status`. See *The automatic drain, and why it is not an `atexit` handler (0.2.66)*. |
| `VLLM_E2_TIER_DEV_LOOKUP_LEDGER_BOOT_ARM` | **does the boot arm the per-request lookup/adopt ring (0.2.66)? Unset is ENABLED; a disable word means the boot records nothing (`LOOKUP_LEDGER[env-refused]`).** OBSERVABILITY ONLY, and that is the whole design: a boot that cannot arm the ring boots, counts it in `boot_arm_refusals` and logs ONE line, and the verdict never joins `boot["refusals"]` - a missing diagnostic is not a tier that declined to serve. The verdict is on the boot line as a bracketed token (`armed` / `env-refused` / `refused` / `inert` / `not_run`), never `key=value`, so the restart verifier's `grep -oE "tier=[A-Za-z_]+" | tail -1` can never land on it. See *The lookup ledger arms at boot (0.2.66)*. |

There is deliberately **no env var for guard mode** - no env var that turns it *on*.
The boot arms it (see *Booting the tier*), because an unguarded tier exception kills
the EngineCore, and `VLLM_E2_TIER_DEV_BOOT_INERT` is the one switch that stops the
boot doing so; there is no env var that arms guard mode on an inert boot, which can
only be done with `POST /plugins/kvtier/guard` - see *Guard mode*.

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

Prefer `EnvironmentFile=/etc/vllm/kvtier-dev.env` if you would rather keep the
settings out of `systemctl show` output. If you launch from a launcher script
instead: `export VLLM_PLUGINS=kvtier_dev VLLM_E2_TIER_DEV=1` before the
`vllm serve` line - and note the launcher is what needs restarting, not just the
shell you curl from. Do **not** re-add `VLLM_E2_TIER_DEV_TOKEN` to a unit file: since
0.2.68 nothing reads it, and the boot warns about it (see *Access model*).

Deep detail: see DEVLOG.md - "The restore race, measured, and the wait that stopped
guessing", "The eviction clock is a touch clock, and nobody's neighbour is capacity",
"Booting the tier: the boot brings it up working", "The SSD tier's boot default"; and
docs/steering-and-patching.md for the runtime kill switch's full text.

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

## Which version is running - and which is only *installed*

Two different facts, and on a box that was upgraded after it booted only one of
them describes the engine:

| Field | Meaning | Where it appears |
|-------|---------|------------------|
| `plugin_version` | the version of the code **this process imported** (`kvtier_dev_plugin.__version__`, read when the payload is built) | `/status`, `/snapshot`, every `rank-*.json` (0.2.50+) |
| `version` | the **schema** version of the document - what a parser pins to answer "can I read this?". It has been `1` since the first release and still is | `/snapshot`, every `rank-*.json`, every verb payload |

Never read one for the other. Adding a code version by *redefining* `version`
would break every consumer that pins its schema meaning, so the two live under
different names - and `protocol_version` / `schema_version` inside a contributed
section stay those sections' own numbers too.

**And never read `importlib.metadata.version("kvtier-dev-plugin")` as "what is
running."** That answers which distribution is INSTALLED on the disk, i.e. what the
NEXT process to import this package will get. A long-running engine keeps the code
it imported at boot, so installed and loaded can legitimately disagree: they did
in this repository while 0.2.50 was being written (`installed = 0.2.49`,
`loaded = 0.2.50`). This is the field's only purpose - an instrument that printed
the installed metadata reported `plugin=0.2.49` for an engine that had been running
since before 0.2.49 existed, three times in one night.

### Three surfaces, three processes

The same field name, and three different processes answering for themselves. A
value from one is never a value for another - which is why each payload also
carries `plugin_version_scope` saying who is speaking, and
`plugin_version_reason` (non-null only when the answer is unknowable - never a
plausible default).

| Where | Which process builds it | Notes |
|-------|------------------------|-------|
| `GET /plugins/kvtier/status` -> `plugin_version` | the **API-server** process | makes no engine call, so this is NOT the engine's version |
| `GET /plugins/kvtier/snapshot` -> `plugin_version` | the **EngineCore** process (built on the loop thread; the API server forwards it unmodified) | one read-only UTILITY hop |
| `rank-PP<pp>_TP<tp>.json` -> `plugin_version` | that **worker** rank's process | a worker answers no route, so the file is the only channel it has |

The two HTTP answers disagreeing is a FINDING, not noise: it means the package was
upgraded and one process restarted while another kept running the code it imported
before the upgrade. Read both.

### The read-only commands

No restart, no engine write, no request traffic - `status` touches no engine at all
and `snapshot` is a pure read on the utility channel. The box's login shell is zsh,
so every remote path is quoted and the file listing uses `find`, never a bare glob.

```bash
# 1. the API-server process (no engine call at all)
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/status' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("api-server", d.get("plugin_version"), d.get("plugin_version_reason") or "")'

# 2. the EngineCore process (one read-only utility hop). The `version` beside it is
#    the SCHEMA, which stays 1.
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("engine", d.get("plugin_version"), "| schema", d.get("version"))'

# 3. every worker rank, from the files the workers write. The directory is the one
#    /status names in `worker_diagnostics.dir` (`/tmp/kv-tier-diagnostics` on the box).
#    `ssh -n` in the loop, or the first ssh eats the list on stdin and you get one
#    rank and a green exit.
ssh llm 'find /tmp/kv-tier-diagnostics -maxdepth 1 -name "rank-*.json"' \
  | while read -r f; do
      ssh -n llm "cat $f" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["rank"], d.get("plugin_version"), "schema", d["version"])'
    done
```

`.get()` everywhere, because a build older than 0.2.50 has no such key and the read
must print `None` rather than raise. That is exactly what these three commands did
on the live box on 2026-10-09, before this release was deployed: `api-server None`,
`engine None | schema 1`, and four ranks of `PPx_TPy None schema 1` - the cue for the
fallback below, not a bug in the reads.

The engine's own `BOOT[auto]` log line carries no version string and
`/plugins/kvtier/version` does not exist (both re-checked on the live box on
2026-10-09); the three reads above are the whole answer.

### Before 0.2.50: the fallback stays necessary

An engine booted on an older build publishes none of these fields, so its running
version can only be INFERRED from version-gated keys. That inference is the only
tool for an old boot and must not be forgotten:

- a **rank file's** `disk_gpu_shadow.pairs_cross_execution` present -> **0.2.48 or
  newer**. Read it from `rank-PP<pp>_TP<tp>.json`, NOT from `snapshot`: that section
  is a worker's own contribution and the snapshot has no `disk_gpu_shadow` key at all
  (confirmed on the live box on 2026-10-09 - the snapshot's top-level sections are
  `blockdump`, `cpu_capacity`, `disk_fill`, `disk_gpu_write`, `disk_restore`,
  `disk_restore_verify`, `disk_spill`, `disk_store`, `guard`, `lookup_ledger`,
  `policy`, `pr56810`, `scheduler`, `store_hook`, `tier`, `trace`, `worker_diag`).
- a rank file's `writer.heartbeat` present -> **0.2.47 or newer**.

Both were read live on 2026-10-09 on a box running such a build: its rank files
reported `writer.heartbeat` present and `disk_gpu_shadow.pairs_cross_execution = 2`,
so it was 0.2.48 or newer - while `/plugins/kvtier/version` answered 404,
`/snapshot` said `"version": 1`, `/status` had no key containing "version" at all
and only `"plugin": "kvtier_dev"` (a name), and every rank file's `version` was `1`.
That is the state of the art for an old boot: an inference you have to remember to
make is one you will eventually skip, which is what 0.2.50 fixed.

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

`/trace`, `/guard`, `/tier`, `/lookups`, `/blockdump` and `/disk` can change engine
dev state, and until 0.2.68 they were token-gated as a whole (not per field) in every
shape, loopback included. **That gate is removed.** What still separates an arm from a
read is the engine-side arm itself (`POST {"op":"arm"}`, an arm that expires or latches
per verb), the `safe_method` wall that refuses a mutating `op` on a `GET` with a 405,
and the peer allowlist - see *Access model*.

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
B=http://127.0.0.1:8000/plugins/kvtier
D=/tmp/kv-tier-diagnostics                  # whatever VLLM_E2_TIER_DIAG is on the box

# 1. is it deployed, and disarmed?  (expect "capability": "deployed_disarmed")
ssh llm "curl -s $B/blockdump" | python3 -m json.tool

# 2. arm it for 30 minutes (a fresh boot is disarmed; nothing works until this)
ssh llm "curl -s -X POST -d '{\"op\":\"arm\",\"seconds\":1800}' $B/blockdump"

# 2b. if you do not know the region names yet: arm a made-up group and read the
#     refusal, which lists every region this rank has
ssh llm "curl -s -X POST -d '{\"op\":\"dump\",\"group\":\"list\",\"block_id\":0,\"length\":16}' $B/blockdump"
ssh llm "cat $D/blockdump-PP0_TP0.json" | python3 -m json.tool     # .reason names them

# 3. one dump per region, at the SAME slot index - collecting the file in between
#    (the response file is OVERWRITTEN by the next dump, and one is in flight/rank)
for R in 0 1 2 3 5; do
  ssh llm "curl -s -X POST -d '{\"op\":\"dump\",\"group\":\"$R\",\"block_id\":170,\"offset\":0,\"length\":4096}' $B/blockdump" >/dev/null
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
ssh llm "curl -s -X POST -d '{\"op\":\"disarm\"}' $B/blockdump"
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
roots are refused by `disk_layout`) **and** this boot must have armed the tier. Since
0.2.33 the boot does arm it by default (`KVTIER_DISK_BOOT_ENABLE`, ON unless the operator
turned it off - see *The SSD tier's boot default*); what has not changed is that **no
runtime knob reaches the arm except the `/disk` verb**, and that since 0.2.35 the arm is a
**LATCH** - it does not expire, it is lowered by `{"op":"disarm"}` and by nothing else, and
it gates SERVING from disk as well as writing to it. Unarmed, the two wrapped methods cost two `os.environ.get`
calls per step: no stat, no open, no copy, no queue, no writer thread.

**It needs the engine patch, and says whether it has it.** A file needs a key, and
today's `SimpleCPUOffloadMetadata` (`vllm/v1/simple_kv_offload/metadata.py`) carries
none. Every entry point then refuses with a reason, counts it, and writes zero bytes;
`engine_fields` in every answer says `present` / `absent` / `unknown` for this boot,
read from the metadata **type**, so the question is answerable without generating a
store. `no_key_for_slot` in the refusal table is that state, counted.

```bash
B=http://127.0.0.1:8000/plugins/kvtier
# what can this boot do?
ssh llm "curl -s $B/disk" | python3 -m json.tool
# arm the tier, then prove the path writes real bytes for one slot. SINCE 0.2.35 THE ARM
# IS A LATCH: `seconds` is still accepted (an existing caller must not start failing) and
# does nothing - the verdict answers `seconds_ignored: true`. It comes down on `disarm`.
curl -s -X POST -d '{"op":"arm"}' $B/disk
# `key` is the engine's GROUP-QUALIFIED block hash: the 16-hex `key16` plus the
# group id as 8 hex chars (`key16 + f"{group:08x}"`, so 24 hex). A bare 16-hex key is
# refused - it cannot say which group's bytes these are. Find both halves with
# `GET /plugins/kvtier/block?block_id=<cpu slot>`: `tier_cpu_slot.block` carries
# `hash_hex` and `hash_group_id`, and any CPU slot is nameable up to the CPU pool's
# size (which is larger than the GPU pool's - the id is bounded per pool).
curl -s -X POST -d '{"op":"spill-one","slot":170,"key":"a3f0b7c1d4e5f60700000003"}' $B/disk
curl -s -X POST -d '{"op":"ls","limit":20}' $B/disk
# ONE IDENTITY, NOT THE WHOLE STORE (since 0.2.21). "Did the span I park actually land,
# and for WHICH RANKS?" - answered from the replayed journals in memory, never by
# walking the store (that is `ls`). One park span is one file PER RANK, so the answer is
# a set of ranks: `ranks_present: 2` on a boot with 4 armed ranks is a half-written span.
curl -s -X POST -d '{"op":"contains","key":"a3f0b7c1d4e5f60700000003"}' $B/disk
# the query-string form does the same thing (and since 0.2.21, so does every other op)
curl -s -X POST "$B/disk?op=contains&key=a3f0b7c1d4e5f60700000003"
# name the four rank FILES of one span without listing the whole store: `shard` is the
# first two hex of the key16 (the filter has always existed; 0.2.21 whitelists the key)
curl -s -X POST "$B/disk?op=ls&shard=a3f0&limit=20"
ssh llm "curl -s -X POST -d '{"op":"disarm"}' $B/disk"
```

**`contains` answers "I cannot answer" as its own verdict, with `present: null`.** Five
of its verdicts are not a "no": a key that cannot name an identity (a `400` at the route,
never a `present: false`), a process with no journal directory, an index that has replayed
nothing since boot, a replay that named no file at all, and a check that raised. That
distinction is the most valuable field on the verb: on 2026-10-05 `entries: 0` was read as
"my span is not on disk" for a whole afternoon when the fact was that nothing had been
replayed into that process yet. A present answer carries the entry's `bytes`, the journal
file it came from, and one row per rank (`rank`, `bytes`, `journal`, `seq`) - which is how
you tell a complete span from a partial one. Since 0.2.46 an identity that SOME ranks hold
and others do not is its own verdict, `partial_ranks`: `present: false` and `offerable:
false` because a restore reads one file per rank, with `ranks_missing` naming which, and
the surviving entry's `bytes`/`journal`/`rank` rows beside it - "three ranks hold this and
PP1_TP1 does not" is a different fact from "nothing holds this", and the offer path's
predicate is now the intersection over the ranks that have to read. What it cannot say is
each rank's geometry digest: the journal line is `<seq> <+|-> <key16> <bytes> <group>` and
names no digest, so naming the actual files is `op=ls` with a `shard` - or the index's own
claim list, `op=identities` - and the answer says so.

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

**A boot with the tier env set serves from disk, and one POST changes who is served.**
Since 0.2.33 the boot default raises the serve posture for you (`KVTIER_DISK_BOOT_ENABLE`,
ON unless the operator turned it off - see *The SSD tier's boot default*); the switches
themselves stay dev-verb only, per boot, and unreachable by a config file or the spill
arm, and `KVTIER_DISK_BOOT_ENABLE=0` restores the pre-0.2.33 posture where **a fresh boot
serves nothing from disk until you POST**:

```bash
curl -s -X POST -d '{"op":"authority"}' $B/disk  # read
curl -s -X POST \
     -d '{"op":"authority","enabled":true}' $B/disk        # EVERY request from SSD
curl -s -X POST \
     -d '{"op":"offers","enabled":true}' $B/disk           # RAM first, disk fills gaps
```

`op=authority` is the global **disk-authority** switch: every request's host-RAM hit is
treated as **zero tokens** for the offer arithmetic and released back to the pool, so the
disk index - not the RAM tier - decides where each request's prefix starts. It opens the
offer gate by itself, so that one POST is the whole change: no second command, no client
change, and no restart needed to turn it back off.

`op=offers` is the cheaper posture: RAM first, the disk extending only what the host-RAM
tier left short. With both up, authority wins - the disk decides.

Both start **false** in a process - a boot that *honours* the default raises one of them
by calling this module's own setter, and a boot that does not (or a box where the flag is
`0`) leaves them down - and `restore.serving_as` names the resulting posture as one word
(`off` / `disk_authority` / `ram_first_gap_fill`). Why the spill arm is not one of these:
on 2026-10-05 a spill arm raised for a write experiment was enough to put 123,474
un-requested 160,000-token offers in front of live traffic. Arming says bytes may be
*written*; these two say requests may be *served*.

**No request field decides any of this (0.2.31).** Until then the authority rule was
carried by a client-supplied marker, `kv_transfer_params.kvtier_disk_restore`, and a
`scope` argument (`optin` / `all`) chose whether that marker was required. A request body
must not decide whether its KV comes off NVMe - it is a payload the caller writes, and a
measurement rig had to make every probe carry it to get the behaviour an operator wanted.
So the marker, the `scope` argument and the fields it fed (`flagged`,
`skipped_no_marker`, `marker_key`, `requests_seen`) are retired: the route refuses
`scope` as an unsupported request key, `disk_restore` no longer reads
`request.kv_transfer_params` at all, and a client can neither ask for disk authority nor
decline it. Sending the old marker is a no-op that changes nothing.

**Why nothing was restored is one HTTP call.** `GET /plugins/kvtier/disk` (and the
`disk_restore` section of `/snapshot`) reports the SCHEDULER's own verdict - the layer
fitted in that process, the gate and its reason, the index size and its source, the
worker proof, and every offer refusal with its meaning - under `restore`, with
`restore.why_no_restore` naming the first missing link. The boot's **posture** is in
that section as fields, not only as text in the boot line: `disk_authority`,
`offers_enabled`, `serving_as`, `per_boot`/`survives_restart`,
`disk_authority_changes`/`offer_switch_changes`, the per-offer block bound
(`offer_block_cap_blocks`/`offer_block_cap_tokens`, which replaced the retired
`offer_cap` when 0.2.32 deleted the fixed 32,000-token cap),
`ranks_armed`/`ranks_seen`, the fuse counters
`suppressed_hits`/`suppressed_spans`/`suppressed_requests`/`suppressions`, and what the
last failure reports decided - `races_not_fused`, `fuses_on_data_kinds`,
`fuses_on_unknown_kinds`, `race_reoffers_granted`, `last_fuse_decision` and the
`failure_kind_classes` partition itself, so "why did this one fuse?" is a payload read. A rank file's
`disk_restore.installed: true` describes a WORKER and cannot start a restore; only
the scheduler's half can, and `BOOT[auto] ... restore=YES|NO restore_gate=open|closed`
is that verdict at boot.

**Running the experiment.** Boot as usual and read `apply_window.windows.bind_to_apply_ms`
- that distribution is the answer to "how big is the window", and since 0.2.32 the
default boot gathers it *while waiting*, so the same boot answers both "how big is the
race" and "did the wait win". Read `wait.settled_after_wait` against
`wait.dropped_after_wait`, `wait.bound` and `wait.allowance_spent`. A loss that is
`dropped_queued` is the wait doing its job late; a loss that is `dropped_running` with
`allowance_spent` true is a device slower than the allowance, and `settle_bound_deadline`
rising with it says the read's own expiry, not the cap, ended the wait.

### Reading the spill in parallel: threads, the pool, and the ordering question (0.2.24)

The read path used to run on exactly one thread (`kvtier-disk-read`), and the box says
what that costs: `dd bs=4M iflag=direct` over 50 spill files (610.7 MiB, read-only)
measured **1.32 GB/s with one reader and 5.14 GB/s with four** - 3.9x. The live engine
agreed from the inside: `reads_started=36`, `reads_applied=36`,
`read_seconds_max=0.0138`, i.e. 11,468,800 B / 13.8 ms ~ **0.83 GB/s per unit**,
serialized 36 deep inside a rank before the step could proceed. There are already four
ranks reading at once; within a rank the units were serial, and now they need not be.

**Two switches, both inert.** `VLLM_E2_TIER_DISK_READ_THREADS` (default **1**) and
`VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` (default **96 MiB**). With both unset the boot is
the 0.2.23 boot: one thread named `kvtier-disk-read`, the same queue, the same pool,
the same bytes - and `test_defaults_reproduce_the_single_thread_boot_exactly` pins it.

**The pool, not the threads, is the constraint - do the arithmetic before the threads.**
A job takes its bounce buffer at *issue* (`request_read`) and gives it back at the
*apply point* (`finish_batch`), so one offer needs the pool to hold that offer's units
**at the same time**. The arithmetic has two inputs, and both come from the code:

* **units per offer per rank = spans x eligible groups.** `per_block =
  len(groups['groups'])` (`disk_restore.py:2260`), and on the live boot that
  is **1**, not 5: g4 fails the block-size match and g0-g3 are excluded by
  `_group_positionally_stable`, so the offer names g5 alone (`last_offer.groups = [5]`).
  Spans are bounded by `MAX_RESTORE_BLOCKS_PER_OFFER = 128` blocks (0.2.35 raised it from
  64 so the backstop sits above a 1 GiB pool's ~93 buffers; 0.2.32 deleted the
  fixed `MAX_OFFER_TOKENS_PER_OFFER = 32_000`-token cap: the pool budget below is the
  bound that actually protects the engine, and the block walk is the principled one -
  how many spans one offer can name and stream inside its deadline). In practice neither
  number is what limits a live offer: the deepest contiguous run the store held on the
  boot the operator measured was **~11 spans**, because spill follows RAM-tier parking
  and the CPU arena is ~187 blocks per rank, so a rank is asked for ~11 units - one file
  per span. If g0-g3 return to positional stability, `per_block` becomes 5 and the same
  offer becomes 50 units; everything below says which number moves.
* **bytes per unit** = one rank's CPU slot: 11,468,800 B (the 3,200-token groups) or
  12,902,400 B otherwise. The live `bytes_applied / reads_applied` = 412,876,800 / 36 =
  **11,468,800 B exactly**, so the running geometry is the narrow one.

The pool is a byte ceiling, so what it buys in units is `floor(pool / unit)`:

| `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` | units of 11,468,800 B | units of 12,902,400 B | host RAM at 4 ranks |
|---|---|---|---|
| 100,663,296 (96 MiB, **the default**) | **8** | 7 | 384 MiB |
| 134,217,728 (128 MiB) | 11 | **10** | 512 MiB |
| 201,326,592 (192 MiB) | 17 | 15 | 768 MiB |

A 10-unit offer needs 114,688,000 B (109.4 MiB) on the narrow width and 129,024,000 B
(123.0 MiB) on the wide one, so **the default pool misses the big-prompt case by two
units - not by an order of magnitude.** That is the whole of the live failure:
`buffer_cap = 6` is 2 refusals per rank on 3 batches, and `incomplete = 24` is those
batches' **accepted** entries - 3 x 8 = 24 units that were read correctly and already
memmoved into their CPU slots - forfeited because `incomplete` is a batch-level flag
(`disk_restore.py:3674-3678`: one sibling's bind-time `buffer_cap` refusal adds
`failed.setdefault(job_id, "incomplete")` to every job in the batch). Nothing failed to
read; work that had already landed was discarded. **This release does not change that
rule** - the fix is up-front offer clamping plus per-entry (not per-batch) poison
inheritance, and that belongs in its own change with its own tests.

**The pool is the byte bound; `MAX_QUEUE_DEPTH` is the reader's JOB bound (256 since
0.2.35, 64 before).** These are different currencies and the smaller governs: the pool says
how many buffers exist (`floor(pool / unit)`, the row above) and the queue says how many
reads the process will accept at all. Since 0.2.35 the offer path prices
`min(pool jobs, queue jobs)` and `bounded_by` names which one cut it (`pool_budget` /
`read_queue`) - the ordered bound table sits with the disk-authority switches above. The queue is 256 because the
offer's own job ceiling is 256 CPU slots (one read per slot), so a maximum 128-span offer -
128 + 4 = 132 jobs - is submittable; a shallower queue does not shorten a restore, it
**holes** it at the first refused job. A queued job *does* hold its buffer, but that buffer
is billed to the pool's byte ceiling, so the queue's own cost is ~1.2 KB of descriptor per
slot (**~0.3 MB per rank process at 256 slots**), not a second RAM budget.

**Maximum useful thread count, from the code.** Threads do not create units. A buffer is
held from issue until the apply point, so the number of units that can sit in flight at
once is `floor(pool / unit)` **per worker process, across every open batch and at any
thread count** - and today's 96 MiB pool puts that at **8 (7 on the wide width)**. A
ninth thread has nothing to read and only adds `reader_starved_polls`, so **8 is the
maximum useful thread count on the default pool**. At **128 MiB** the ceiling is **11
narrow / 10 wide**, which is the whole offer; at **192 MiB** it is **17 / 15**. The other
ceiling is the offer itself: one request contributes 10 units, so threads above 10 only
help when several requests bind in the same step and share the pool - which is why 4 is
the sane first setting and 16 is a clamp, not a target.

What that buys in wall time, from the live `read_seconds_max = 0.0138 s` as the
per-unit bound: a 10-unit offer serialised is at most **138 ms**, 8 units issued at 96
MiB is at most 110 ms of reading that the step waits through, and four readers cut that
to about **28-35 ms** (the two refused units are lost at bind no matter how many threads
exist, which is why the pool is the first knob and not the second).

**What to set.** For the live big-prompt case, set the pool to 134,217,728 bytes
(128 MiB) and the threads to 4: 128 MiB is the smallest
power-of-two that holds a whole 10-unit offer on **both** widths (11 narrow, exactly 10
wide), costs 512 MiB of resident bytes across four worker processes on a box that
reported 99,646 MiB available with the 120 GB pinned tier already up, and the default
stays 96 MiB - an unset environment is still the 0.2.23 boot. 192 MiB is the setting for
a future 5-eligible-group offer (50 units needs 645,120,000 B, i.e. 640 MiB per rank,
2.5 GiB across four). Four threads, not eight: the box's own `dd iflag=direct` sweep
measured 3.9x at four readers, so four is where the array was still scaling; going past
the measured point is a decision to make from `parallel_efficiency`, not from a knob.

**The queue cap stays 64 and does not need to move.** The largest legal offer names 10
units today (50 if `per_block` reaches 5), under the cap, and a 96 MiB pool refuses the
NINTH unit long before the queue refuses the 65th - so the pool binds first in every
case the offer path can reach. With N readers the queue drains N-at-a-time, which makes the cap strictly
*less* likely to bind, never more; `queue_depth_peak` and `refused.queue_full` are
published so that claim can be checked instead of asserted.

**Changing the reader count without a restart (0.2.67).**
`POST /plugins/kvtier/disk {"op":"read-threads","threads":6}` moves the count in place,
in **both** directions, as often as you like, in one boot. It exists because the only
lever was `VLLM_E2_TIER_DISK_READ_THREADS`, and every move of 8 -> 16 -> 6 cost an engine
restart: a couple of minutes down, the previous arm's counters wiped, and every
comparison left resting on a short window. `{"op":"read-threads"}` alone **reads** the
posture (the house grammar of every mutating `/disk` op: a key present is an
instruction, a key absent is a question), and a `threads` that is not an integer in
`[1, 16]` is refused at the wire with a 400 and again by the verb, naming the range and
changing nothing; asking for the count the pool already has is an idempotent success;
only one resize may be in flight, and a second is refused rather than queued.

A **raise** spawns the missing slots. A **shrink is cooperative and therefore
asynchronous**: a surplus slot finishes the read it is holding, settles it, returns its
buffer, and exits at its next loop top - nothing is cancelled, no job changes owner, and
`resize.surplus_draining` says how many slots have not exited yet (`configured` 1 with
`alive` 4 for one `reader_poll_seconds` is a shrink landing, not a shrink refused). The
exit is a condition the slot reads about itself (`slot >= _reader_want`) and **cannot**
be a `_STOP` sentinel, because a sentinel is unaddressed: a queue cannot hand an item to
slot 7 specifically, so a sentinel posted for a surplus slot is one any reader may
consume - including one nobody asked to stop. `_ensure_thread` re-reads the limit at
every bind, so the override is applied inside `_resolve_limits` (the single place that
read goes through), which is what makes the count survive a re-bind instead of reverting
to the boot value on the next step - pinned by `test_the_runtime_count_survives_a_re_bind`.

**Two boundaries, and the verb says both in its own answer.** PER BOOT: a restart
restores the launcher's environment, `read_threads.env_requested` keeps what the boot
was given, `read_threads.source` is `env` or `verb`, and the `DISK[read]` announce line
stays the proof of what a boot was given - this verb does not replace the launcher, and
neither does the relay. PER PROCESS, RELAYED: a reader pool is process state, and this
door is answered by the process that holds the engine call - on a real boot the
EngineCore, which **never reads a spill file**. A worker is reachable by no door (that is
why `spill-one` writes a request file a worker polls), so 0.2.67 relays the ask the same
way, over its own file, and the ranks that apply restores are the processes it matters to
reach.

Deep detail: see docs/disk-tier.md for the eligibility arithmetic, the bound tables, the
journal's filesystem rules and the reader relay's internals; and DEVLOG.md for the dated
entries this tier grew from - "What a failed load costs", "The ceiling walk, the sample,
and the store that grew past its ceiling", "The claim must match the bytes", "The SSD
tier serves the GDN groups", "The door that showed zeros", "The buried journal", "The
eviction clock is a touch clock", "Four ranks, one ceiling", "An unattributable name
never buys capacity" and "The restore race, measured, and the wait that stopped
guessing".

## Operator runbook: the KV tier is misbehaving, right now

Terse commands for standing at the box while something is wrong. Everything
below runs **on the box** (`ssh llm ...`) because `/plugins/kvtier/*` is
loopback-or-allowlist. Nothing needs a header: since 0.2.68 the dev-state routes
(`/tier`, `/guard`, `/disk`, ...) answer a loopback peer exactly like the plain
reads do, so `~/.local/share/kvtier-dev-token` is only a receipt of an older
posture, not a credential. If a command below ever starts answering 403, the
question is "which address did the engine see", not "which token did I send".

### Every boot with the tier env set: the order traffic goes in

> **The boot arms the tier itself when the install verifies; the read-back proves
> it and the fault ring proves the boot was clean, not merely up.**

```bash
# 1. read the boot verdict. The BOOT installs, guards and arms (the connector-
#    creation hook runs all three when the tier connector is built, with the live
#    KVCacheConfig); this curl only CONFIRMS it - and it is still the read that
#    proves the claim. Expect installed:true, armed:true, boot.mode:"auto",
#    boot.backport/guard/tier all true, patched_before_first_request:true
ssh llm 'curl -s "http://127.0.0.1:8000/plugins/kvtier/tier?op=status"' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);b=d["pr56810"]["boot"];print(d["install"]["installed"],d["armed"],b["mode"],b["tier"],b["refusals"],d["operator_note"])'

# 2. prove it is CLEAN, not just armed. Guard mode is armed by the boot, so a tier
#    exception is a recorded DECLINE: `armed:true` alone cannot tell you the tier
#    is doing work. A non-zero count here means the tier is alive and refusing.
ssh llm "curl -s -X POST http://127.0.0.1:8000/plugins/kvtier/guard \
  -d '{\"op\":\"captures\",\"limit\":5}'" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["count"],"faults");[print(c["method"],c["exception_type"],c["message"][:80]) for c in d["captures"]]'

# 3. only a REFUSAL needs a verb now. If step 1 says armed:false with mode "auto",
#    read boot.refusals - it names the gate (install / live-object check / guard
#    coverage). To arm by hand once the cause is fixed:
ssh llm "curl -s -X POST http://127.0.0.1:8000/plugins/kvtier/tier \
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
ssh llm "curl -s -X POST http://127.0.0.1:8000/plugins/kvtier/guard \
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
ssh llm 'curl -s "http://127.0.0.1:8000/plugins/kvtier/tier?op=status"' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);b=d["pr56810"]["boot"];print(json.dumps({k:d[k] for k in ("armed","boot_default_armed","operator_note","next_step")}, indent=1)); print(json.dumps({k:b[k] for k in ("ran","mode","backport","guard","tier","refusals","faults_at_boot")}, indent=1)); print(d["install"])'
```

### Runtime kill: make the tier a strict no-op, NO restart (one curl)

Also the state the engine starts in: `disarm` is how you get back to the boot
posture without a reboot.

```bash
ssh llm "curl -s -X POST http://127.0.0.1:8000/plugins/kvtier/tier \
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
`GET /tier?op=status`) under `pr56810`:

```bash
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/tier?op=status' \
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

Deep detail: see DEVLOG.md - "The disk tier went off by itself on a quiet box (0.2.45;
corrected 0.2.47)".

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
3. **There is no third gate (0.2.68).** Until this release a bearer token stood
   behind the peer rule, in three shapes:

   * `POST /policy/reload` required `x-kvtier-dev-token` to equal
     `VLLM_E2_TIER_DEV_TOKEN` (compared with a constant-time digest), loopback
     included, and was refused when that variable was unset;
   * a read from a non-loopback peer required that same token (a loopback read
     needed none, so `ssh llm curl http://127.0.0.1:8000/...` was unchanged);
   * the six dev-state routes (`/trace`, `/guard`, `/tier`, `/lookups`,
     `/blockdump`, `/disk`) required it in EVERY shape - a loopback `GET /trace`
     included.

   **All three are removed**, by the operator's decision: the token made the surface
   painful to drive from the proxy and from scripts, and the engine is to be
   access-controlled later by other means. The consequence, stated plainly because it
   is the sentence this section exists to make impossible to miss:

   > **The peer allowlist is now the ONLY gate on this surface, and `/plugins/*` is
   > NOT covered by vLLM's `--api-key`** (that covers `/v1`, `/v2`, `/inference` and
   > `/cohere` only). A key in front of `/v1` protects nothing here. An address on
   > `VLLM_E2_TIER_DEV_ALLOW` reaches this surface unauthenticated and can arm guard
   > mode, arm the SSD spill tier, reload the policy and read parked KV bytes. Keep
   > the list at individual `/32` addresses, never `0.0.0.0/0`, and remove entries
   > when you are done - unset the variable and the surface is loopback-only again.

   What stands between a caller and engine state is therefore not a secret: the arm
   itself (a `POST {"op":"arm"}` per facility, expiring or latching per verb, and
   there is no env var that arms guard mode, the byte dump or `spill-one`), the
   `safe_method` wall (a mutating `/disk` `op` on a `GET` is a 405 before the engine
   is touched), the argument specs (`ARG_SPECS`, validated before the engine is
   reached, with the engine method a route may call fixed by the route and never by
   the wire), and `policy.py`'s own path validation. Those got MORE important in
   0.2.68, not less.

   **The transport is plaintext HTTP with no TLS**, as it always was. Nothing secret
   crosses it now - that is the one security gain in the removal.

   Order on every route: prove the peer is allowed -> prove the plugin is enabled ->
   run. A peer the guard rejects gets 403 and learns nothing about the plugin's
   configuration (not even the contents of the allowlist, which are deliberately
   absent from the refusal), and that 403 names the PEER rule and never a token,
   because there is no token left to name.

   **`VLLM_E2_TIER_DEV_TOKEN` does not go quietly.** A variable that reads like a
   control and gates nothing is a story an operator tells themselves, so the plugin
   says so where an operator looks. The boot logs ONE `WARNING` naming the variable,
   the release that stopped reading it and the gate that replaced it (`... is SET but
   is no longer read by this plugin (since 0.2.68) ...`; its VALUE is never logged),
   and `/status` answers `token_gating: false`, `token_header`,
   `non_loopback_peers_require_token: false`, `vllm_api_key_covers_these_routes:
   false` and a `token_note` carrying the same sentence. `token_configured` and
   `token_env` are still published with their literal meaning ("is the variable set",
   "which variable"), because a key vanishing from the payload is its own kind of
   silence. Setting the variable moves `token_configured` and nothing else anywhere on
   the surface - pinned by
   `tests/test_routes.py::test_setting_the_token_env_changes_the_report_and_nothing_else`.
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

Deep detail: see docs/testing.md - what each file in `tests/` guards.

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
  work is a tier that is declining. Reachable from any allowlisted peer since 0.2.68
  (there is no token to have), it echoes
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

## License

Apache License 2.0 — see [LICENSE](LICENSE), which is the canonical text byte for
byte with only the appendix's copyright line filled in, and [NOTICE](NOTICE) for the
third-party attributions.

The licence was chosen for compatibility rather than preference. This package is not a
fork of vLLM and vendors none of it, but it patches vLLM at runtime
(`kvtier_dev_plugin/backport.py`) and carries a patch series (`patches/`) whose hunks are
authored against forked vLLM trees and quote vLLM code in their context. vLLM is
Apache-2.0, so an Apache-2.0 licence here cannot conflict with the trees this plugin is
built to run against; a different permissive licence could.

This licence covers this repository only. It conveys nothing about the engine trees it
patches, the launcher scripts it is driven by, the host configuration it assumes, or any
model weight — all of which stay under whatever terms their own owners set.
