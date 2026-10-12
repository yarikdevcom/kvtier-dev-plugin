# kvtier_dev_plugin

<img src="docs/img/kvtier-tier.svg" alt="Architecture diagram. Data path: a request goes client, to the API server, to EngineCore (the scheduler, which owns the connector metadata and runs get_num_new_matched_tokens and update_state_after_alloc), to the four worker ranks. Below that sit three tiers: the GPU KV pool at block-size 4096 geometry, the pinned host-RAM arena behind SimpleCPUOffloadConnector sized by KV_OFFLOAD_SIZE, and the NVMe spill store at VLLM_E2_TIER_DISK_ROOT with its journal, which is the only file-to-block map. Spill runs downward and is solid; restore runs upward - solid from the RAM tier, which production serves on, and dashed from NVMe, whose restore gate stays boot-closed until the worker ranks prove their reads even though the posture in force is serving_as=disk_authority. A group rule decides what is restorable: group 5 at every block of a run, groups 0-3 at the run's last block (the anchor), group 4 never spilled, so one hole in the store forfeits every token behind it. Beside the data path, not inside it, sit the /plugins/kvtier/* dev routes, which run on the EngineCore loop thread, and the per-rank diagnostic files under /tmp/kv-tier-diagnostics whose sum - not the answering process's flat counters - is the fleet truth." width="100%">

_Data path: client -> API server -> EngineCore (the scheduler) -> the worker ranks, over three tiers - GPU KV pool, pinned host-RAM arena, NVMe spill store plus its journal. Spill runs down; restore runs up, and the disk restore is dashed because its gate opens only on worker-rank proofs. The dev routes and the per-rank diagnostic files are observability, beside the path and never in it._

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

## Where the documentation lives

This README keeps what you need at 02:00: what the package is, install, the
switches, how to prove the tier is really mounted, which version is running, the
access model, the known limits, and the **operator runbook** - "the KV tier is
misbehaving, right now". That ordering is deliberate: the runbook does not belong
one file away from the front door. The deep reference material lives in `docs/`:

| File | What is in it |
|------|---------------|
| `docs/disk-tier.md` | the NVMe tier in full: spilling parked KV to the engine's own disk, the parallel readers, the bounce-buffer pool and the ordering question, the eligibility arithmetic and the bound tables |
| `docs/diagnostics.md` | the HTTP surface: the route table and the read-only commands, the argument rules and the `/snapshot` fields, the lookup ledger, how much parked KV is retrievable, why THIS request missed, guard mode, the worker-side diagnostics |
| `docs/reference-tools.md` | the single-purpose tools: `kvconfig` (read the geometry before the first store), `kvinspect` (one block or one request), the trace ring, and `blockdump` (a few parked bytes to settle a byte-level question) |
| `docs/steering-and-patching.md` | steering and vetoing the tier at runtime (`store_hook.py`), the kill switch, and coexistence with the other runtime patches |
| `docs/testing.md` | the per-file ledger of `tests/`: what each file guards and why it exists |
| `DEVLOG.md` | the dated record - incidents, measurements, decisions and the wrong turns |

So a `See *Section*` inside the Switches table below names a section of one of
those files, not a section of this README.

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
| `VLLM_E2_TIER_DISK_ROOT` | **the SSD tier's store root - one file per KV slot, and the tier's existence condition.** Unset = the tier cannot exist: no directory is created, no thread starts, no byte is written (`config.py`). A root on tmpfs is REFUSED, because a store that a reboot erases is not a store. It is not an arm: naming a root makes the tier possible, `POST /plugins/kvtier/disk {"op":"arm"}` makes it happen. Keep the root **per tree** (per geometry, per `--block-size`, per digest) - one root under one ceiling means one tree's capacity pass is free to evict another's data. |
| `VLLM_E2_TIER_DISK_BYTES` | **the store's byte ceiling.** Above it files are shed LRU by `st_mtime` - a TOUCH clock since 0.2.29, meaning a write or a restore read that produced usable bytes, not a creation stamp. **Unset = no ceiling, and never zero** (a tier that deletes everything on boot would be a tier that cannot warm). Neither this nor the root arms anything; since 0.2.35 the arm is a latch with no expiry, so it comes down on a disarm POST and not on a clock. A store over its ceiling whose pass comes up short is back-pressure working as designed (`plan_short`), not a fault. |
| `VLLM_E2_TIER_DISK_JOURNAL_DIR` | where the SSD tier's spill **journal** lives, apart from the rest of the diagnostics directory. **Unset = fall back to `VLLM_E2_TIER_DIAG`** (the pre-0.2.19 behaviour). Set it to storage that survives a reboot: the journal is the only map from a KV identity to a spilled file and cannot be rebuilt, while the KV files it describes outlive it. **The path does not have to exist** - the plugin creates it (mode `0750`, the mode the store's own shard directories get) on the way to the first journal line, once per directory. **Only the journal follows this var** - the spill arm file stays in `VLLM_E2_TIER_DIAG` and dies at boot, because an arm must not survive a reboot. A journal on tmpfs is warned about once and reported (`journal_dir=tmpfs` on the boot line), and a journal directory that cannot be created or written is reported (`journal_dir=unwritable`, `journal.writable: false`, `journal.directory_error`, `journal.write_errors`) - either way it is never a refusal to spill. |
| `VLLM_E2_TIER_DIAG_INTERVAL` | how often a rank rewrites its file, in engine steps. Default `50`, clamped to `[1, 100000]`. A step that *changed* something writes immediately (subject to the floor below). |
| `VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS` | the rate floor between writes, in seconds. Default `1.0`, clamped to `[0, 3600]`; `0` means "no floor". |
| `VLLM_E2_TIER_DIAG_EPOCH` | pin the per-boot `epoch` string the reader uses to prove two facts came from the same boot. Unset = derived once per process and stable for its lifetime. |
| `VLLM_E2_TIER_DIAG_RANK` | pin this rank's id (`PP<n>_TP<m>`) instead of deriving it from the already-imported `vllm.distributed.parallel_state`. An illegal value means **no file at all**, because the reader refuses a file whose `rank` does not match its own name. |
| `VLLM_E2_TIER_DISK_SETTLE_MS` | **how long one apply point may wait for the bytes it is about to apply, in milliseconds. Unset = `5000` since 0.2.32** (it was unset-means-never-wait through 0.2.31, which is why a default boot used to lose every raced read); `0` = never wait, and that is still reachable. It is a FLAT allowance - `min(SETTLE_MS, the load's own remaining deadline)` - and no arithmetic on bytes or throughput takes part in it; see *The restore race, measured, and the wait that stopped guessing* below. It is an allowance and NOT an arm - the restore path still needs the disk root, both arms, this boot's offer switch and a proven worker before it reads a byte. A value that will not parse falls back to the 5,000 ms default, says so in the log and on the payload (`env_problems`), and only that line names the variable that was ignored. |
| ~~`VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S`~~, ~~`VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S`~~ | **DELETED in 0.2.32.** These two fed the settle formula (`SETTLE_READ_BPS_DEFAULT`, `PREFILL_TOKENS_PER_S_DEFAULT`, `SETTLE_SAFETY_MARGIN`, `SETTLE_MIN_MS` are gone with them); the rule is flat now, so there is nothing left for a throughput assumption to compute. Reading them is a no-op - they were guesses, and the box proved it. See *The restore race, measured, and the wait that stopped guessing*. |
| `VLLM_E2_TIER_DISK_READ_THREADS` | **how many threads issue spill reads at once (0.2.24). Unset or `1` = ONE thread**, which is every boot before 0.2.24, byte-identically and pinned by a test. Clamped to `[1, 16]`; anything that is not a positive integer means "the operator said nothing" and is reported in `read.env_problems`. It is a thread count and nothing else - the queue stays FIFO, so dispatch order is submission order. **Raise the pool with it or it buys nothing on a big prompt**: in-flight bytes are bounded by the pool, not by the threads. Since 0.2.67 the count is *also* a runtime lever (`POST /disk {"op":"read-threads","threads":N}`), relayed to the worker ranks through `disk-read-threads.json` and still **per process and per boot** - so this variable remains the boot-time source of truth (a restart restores it on every rank), while `read.threads_source` says which owns the live count in a given process. See *Reading the spill in parallel* in `docs/disk-tier.md`. |
| `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` | the bounce-buffer pool's byte ceiling, PER PROCESS. **Unset = `disk_read.MAX_BUFFERS_BYTES` = 96 MiB**, today's value, which holds **8** units of 11,468,800 B (7 of the 12,902,400 B width). This is what actually bounds in-flight work: a buffer is taken at *issue* and only returned at the apply point, so **one offer needs the pool to hold the whole offer at once** - and a big-prompt offer is 10 units, so the default misses it by two. **128 MiB (134217728) is the smallest setting that holds a whole 10-unit offer on both widths.** HOST-RAM COST: the pool is per process and every engine rank is a process, so 128 MiB here is 512 MiB of resident bytes on a 4-rank boot. Clamped at 8 GiB and says so (`pool.clamped`). See *Reading the spill in parallel* in `docs/disk-tier.md`. |
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
| `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS` | **THE STEP-2 LOAD-PAIR PRUNE (0.2.49), OFF by default, same grammar.** Unset (or `0`/`false`/`off`/`no`, or anything unrecognised) = the restore apply point rewrites the load lists exactly as every release before this one did, byte for byte, and none of the `prune_*` counters move. `1`/`true`/`yes`/`on` = at the same apply point, an entry whose read SUCCEEDED and was adopted is REMOVED from BOTH `load_cpu_blocks` and `load_gpu_blocks` at the same index, in the one rewrite the module already owned, so the fork's `copy_blocks` no longer DMAs that arena slot to the GPU window. **This is the first flag in this package whose failure mode is a wrong ANSWER rather than a slower one**, for two reasons the design is built around: `copy_blocks` pairs `src[i]` with `dst[i]` POSITIONALLY (`cuda_mem_ops.py:227`, descriptors built at :239-248) and cannot see that one list got shorter: with EQUAL lengths a one-sided prune silently shifts every later block into the wrong destination, and with UNEQUAL ones `total` comes from the CPU side alone (:248) so the copy loop at :257 slices GPU descriptors past the end of a shorter array. Neither shape is an error the fork reports - which is why `prune_load_pairs` is the only door a pair leaves through, refuses a call whose two sides disagree (`one_sided_prune`, `length_mismatch`, `out_of_range`, `over_cap`, `group_delta_mismatch`, `destination_drift`) and hands back NO lists on a refusal; and a group's entry binds to `block_ids_by_group[g][num_computed // group_block_sizes[g] + i]`, so an index derived from tokens instead of from the scheduler's own index-aligned identity mis-names a `boundary_state` group - which is why the index set is read out of the bound jobs and never computed. An entry whose read FAILED is never in the prune's population (the apply point drops it and reports its GPU block as a load error exactly as it always has, so the engine recomputes: `prune_skipped_failed_entries`, and the load-error id set is identical with the flag on and off). **STEP 3 CHANGED THE SAFETY ARGUMENT (0.2.52), and it changed the flag's meaning**: an entry is now a candidate only if its index is in the set `disk_gpu_write.observe_apply` reported as ENQUEUED (`written_indices`). This is a MEMBERSHIP test and not a comparison of counts, and the difference is the whole point - the old arithmetic (`witnesses < candidates`) called a boot with three sampled writes and thirty adopted entries a gap of 27 and named nothing. Because the gate runs BEFORE the prune, `prune_unwitnessed_entries` is now a TRIPWIRE that cannot move: a non-zero reading means the membership test was broken, and every index it names is being served from uninitialised KV. `prune_emptied_steps` still counts the steps left with nothing for the fork to copy. WHY THE GATE IS THE ENQUEUE AND NOT THE READBACK: a readback verdict lands on the writer's daemon thread up to `JOB_MAX_AGE_SECONDS` (30 s) after the enqueue, while the prune runs SYNCHRONOUSLY inside `start_load_kv` and is applied in the same step - a synchronous decision cannot wait on asynchronous evidence, so the enqueue gates and the readback is the KILL-SWITCH that retires the flag later. CONSEQUENCE FOR AN OPERATOR: this flag alone does NOTHING unless `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY` is also set, and it says so - every step is refused as `not_delivered` (a name of its own, because "no disk work this step" and "disk work with no writer" are opposite stories) and `prune_uncovered_entries` counts the destinations the fork is still carrying. Read once at first use; only a WORKER has an apply point, so an EngineCore answers `enabled: false` forever. See *Pruning the pairs an adopted restore no longer needs (`disk_restore.py`, 0.2.49)*. |
| `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY` | **DELIVERY MODE (0.2.52), OFF by default, same grammar, and a LAYER on `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE` rather than a second writer.** The variable NAME has `DIRECT` in it (`config.ENV_DISK_GPU_WRITE_DELIVERY` is the Python constant, and it holds `"VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY"`): set `VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY` and nothing reads it, because the constant's name is not the environment's name. The TIMEOUT below is the odd one out the other way - its spelling genuinely has no `DIRECT`. Unset = the writer keeps its shipped witness posture (one entry in `sample_every_n_applied`, one write per step, a `min_interval_seconds` floor) and costs what it always cost. `1`/`true`/`yes`/`on` = EVERY applied entry of the step gets its own host-to-device copy, so the entries the prune may remove are a SUBSET of the entries this module enqueued. The three throttles are read from ONE dict (`limits_in_force`) so they cannot move independently - lifting the sampler and leaving the step cap at 1 is a "delivery" that writes one entry and counts `throttled_step` for the rest, reading as a broken feature. A fourth limit is NOT a throttle and is not lifted by this flag: the pinned pool holds `MAX_PINNED_ROWS_PER_REGION` rows and a witnessed write holds one row per lane, so it admits exactly ONE in-flight record. In delivery mode the COPY needs no pinned row (its source is the CPU arena slot, not a pinned buffer), so an unavailable row costs the WITNESS and never the write - `delivered_unwitnessed` counts copies that went unread back, and `no_pinned_buffer` stays a refusal this mode never takes. With the writer flag unset this flag is INERT and `delivery_mode.reason` names the missing flag. **THE KILL-SWITCH is what makes the mode safe to run**: a `readback_mismatch_stable` is same-execution evidence about our OWN copy (both readbacks were enqueued on our stream around our copy inside one `start_load_kv`, and the file side is the file this restore read), so it latches for the life of the process - the three throttles come back and `disk_restore` REFUSES every later prune as `kill_switch_fired`, restoring the fork's copy. `readback_mismatch_gpu_changed_on_recheck` (another stream wrote the window) and `readback_mismatch_unrechecked` (no before-state, so nothing is attributable) are counted in `kill_switch_declined` and NEVER latch: a switch that fires on a race is a feature that turns itself off for no reason, which is the 0.2.48 lesson from `disk_gpu_shadow` in a new lane. `kill_switch_fired + kill_switch_declined == readback_mismatch` always. Read once at first use. See *Delivering the window ourselves (0.2.52)*. |
| `VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS` | **THE ONE BOUND ON A DELIVERY STEP, in seconds (0.2.57). Unset, or anything that is not a finite number above 0 = `disk_gpu_write.DELIVERY_APPLY_TIMEOUT_SECONDS` = 10 s.** Note the spelling: this name has NO `DIRECT` in it, while the delivery flag above it does (`VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY`) - the two are one word apart and only one of them is read by the other. A delivery copy's source is the fork's **pageable** `cpu_kv_caches` (`v1/simple_kv_offload/worker.py:232`, no `pin_memory`), so `copy_(non_blocking=True)` does **not** return early: the whole step's bytes are synchronous PCIe on the engine's step thread, which is how a 198,400-token restore on 2026-10-09 parked a worker for minutes and EngineCore died on `RPC call to sample_tokens timed out`. **This is deliberately ONE generous wall-clock timeout and not a per-entry budget**: 0.2.56 shipped a 0.25 s step budget *and* a 64 MiB byte budget, and both are gone - the tier's house rule (learned on the disk read/write path) is one generous timeout instead of fine-grained caps, and a 0.25 s cap would defer ~57 of a normal 62-block restore, which turns the prune this mode exists to feed into a no-op. A guard that disables the feature is not a safety feature. Why 10 s: the engine's `No available shared memory broadcast block found in 60 seconds` is a WARNING CADENCE (`VLLM_RINGBUFFER_WARNING_INTERVAL`, `vllm/envs.py:20`) and the reader never gives up, while the deadline that actually ends the process is `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` = **300 s** (`vllm/envs.py:242`, passed by `sample_tokens`) - so 10 s sits ~30x inside the only real threshold and ~60x above the ~0.15 s a full 735 MB restore measured. When it expires the rest of the step's entries are DEFERRED (`delivery_deferred_entries` / `delivery_deferred_steps`) and stay out of `written_indices`, so the prune cannot remove them and the fork's copy delivers them: **a stop costs a missed optimization and never a hole**. `delivery_refused_entry_too_wide` (0.2.56's byte-budget refusal) is **REMOVED**, not left at zero. Read PER APPLY POINT, so a drop-in takes effect without a restart; clamped at `DELIVERY_APPLY_TIMEOUT_MAX` = 60 s and says so (`delivery_mode.timeout_clamped`), because past the engine's own warning cadence the number describes nothing real; `0` is refused rather than honoured (it would defer every entry of every step and read as "configured"). The value in force is published at `delivery_mode.timeout_seconds`, and whether the bound has ever bitten is `delivery_mode.step_seconds_peak` read beside it. See *Delivering the window ourselves (0.2.52)*. |
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
only be done with `POST /plugins/kvtier/guard` - see *Guard mode* in `docs/diagnostics.md`.

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

### What one real boot actually carries (and the duplicate-flag trap)

The table above is the capability surface. This is one serving boot read from the
engine's own `/proc/<pid>/cmdline` and `/proc/<pid>/environ` - not from a unit file,
because `systemctl show -p Environment` reports a unit's INTENT and is not a reading
of any process.

Engine: `vllm serve <model-dir>` on v2.5.1, the tier mounted, with
`--block-size 4096 --prefix-match-unit 64 --prefix-cache-retention-interval 12288
--max-model-len 262144 --enable-prefix-caching --kv-offloading-size 8 --watermark
0.05 --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector",
"kv_role":"kv_both","kv_load_failure_policy":"recompute"}'`.

> **`--max-num-seqs` appears TWICE on the live command line: `8` first, then `4`.**
> argparse is last-wins, so the effective value is **4**. A reader who greps the first
> match reports a number the running engine does not obey. `--max-num-batched-tokens`,
> `--kv-cache-memory` and `--gpu-memory-utilization` are duplicated the same way by the
> wrapper, so the rule for reading ANY flag off `/proc/<pid>/cmdline` is: take the
> **last** occurrence (or the engine's own config dump), never the first grep hit.

`--use-replayssm` is **absent** from this boot (occurrence count 0). Where a doc says
the tier and that flag are refused together, that is a statement about the engine's
validator, not about this boot: this boot already runs without the flag.

Tier environment in force, against the default this table documents:

| Variable | Live value | Against the documented default |
|---|---|---|
| `VLLM_E2_TIER_DEV` | `1` | on (unset = inert) |
| `VLLM_E2_TIER_DEV_TIMEOUT` | `8.0` | default `5.0`, so this boot waits longer for the engine |
| `VLLM_E2_TIER_DEV_ALLOW` | one address (the proxy host) | unset = loopback only; this is the ONLY gate (0.2.68) |
| `VLLM_E2_TIER_DIAG` | `/tmp/kv-tier-diagnostics` | as documented; without it no rank file is ever written |
| `VLLM_E2_TIER_DISK_ROOT` | a per-tree root (`$HOME/kv-tier/disk-v25`) | the `-v25` suffix is the per-tree rule, not a version tag |
| `VLLM_E2_TIER_DISK_JOURNAL_DIR` | a per-tree journal (`$HOME/kv-tier/journal-v25`) | as documented: persistent storage, survives reboot |
| `VLLM_E2_TIER_DISK_BYTES` | `250000000000` (250 GB) | unset = no ceiling; this boot sets one |
| `KVTIER_DISK_BOOT_ENABLE` | `1` | **redundant but honest**: unset is already ON, so this states the default |
| `VLLM_E2_TIER_DISK_READ_THREADS` | `16` | default `1`; `16` is the clamp ceiling `[1, 16]` |
| `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` | `2147483648` (2 GiB) | default 96 MiB, clamp 8 GiB. **Per process**: 8 GiB transient across four ranks |
| `VLLM_E2_TIER_DISK_RESTORE_SOURCE` | `bounded` | default `arena`; this boot caps staging rows instead of trusting arena size |
| `VLLM_E2_TIER_DISK_RESTORE_STAGING_ROWS` | `86` | default `32`. 86 x 12,902,400 B = ~1.03 GiB/rank, ~4.1 GiB over four ranks - **above** the ~1.61 GiB the default's arithmetic quotes, so the 2 GiB figure in `config.py` is the default's number, not this boot's |
| `VLLM_E2_TIER_DISK_RESTORE_VERIFY` | `1` | default OFF: a byte witness is running on this boot |
| `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE` | `1` | default OFF |
| `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY` | `1` | default OFF. Note `DIRECT` in the name - the timeout next to it has none |
| `VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES` | `1` | default OFF |
| `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS` | `1` | default OFF, and **the one flag whose failure mode is a wrong answer**; it is only meaningful with delivery on, which this boot sets |
| `VLLM_E2_TIER_DISK_SETTLE_MS` | **not set** | so it runs the flat **5,000 ms** default (0.2.32). Unset no longer means "never wait" |

Four of the names in that boot's environment are **not this package's** and no plugin
code reads them (`VLLM_E2_SIDE_CACHE_LAYOUT`, `VLLM_E2_MAMBA_RETAIN_CHECKPOINT`,
`VLLM_E2_PLE_PULL_TRANSPORT`, `VLLM_E2_GUARD_MODE` - all belong to the engine fork's own
`VLLM_E2_*` family, and `VLLM_E2_GUARD_MODE` appears in `config.py` only as an example
of that family). Two of them are near-misses worth naming: `VLLM_E2_GUARD_MODE` is not
this plugin's guard mode (which has **no** env var at all - the boot arms it), and
`VLLM_E2_MAMBA_RETAIN_CHECKPOINT` is not the plugin's `VLLM_E2_TIER_DEV_MAMBA_ANCHOR_PATCH`.
`KV_OFFLOAD_SIZE` is the engine's own knob too; the plugin reasons about it and never
reads it as its own switch.

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
