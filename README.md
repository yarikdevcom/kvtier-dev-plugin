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
| `VLLM_E2_TIER_DISK_SETTLE_MS` | **how long one apply point may wait for the bytes it is about to apply, in milliseconds. Unset = `5000` since 0.2.32** (it was unset-means-never-wait through 0.2.31, which is why a default boot used to lose every raced read); `0` = never wait, and that is still reachable. It is a FLAT allowance - `min(SETTLE_MS, the load's own remaining deadline)` - and no arithmetic on bytes or throughput takes part in it; see *The restore race, measured, and the wait that stopped guessing* below. It is an allowance and NOT an arm - the restore path still needs the disk root, both arms, this boot's offer switch and a proven worker before it reads a byte. A value that will not parse falls back to the 5,000 ms default, says so in the log and on the payload (`env_problems`), and only that line names the variable that was ignored. |
| ~~`VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S`~~, ~~`VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S`~~ | **DELETED in 0.2.32.** These two fed the settle formula (`SETTLE_READ_BPS_DEFAULT`, `PREFILL_TOKENS_PER_S_DEFAULT`, `SETTLE_SAFETY_MARGIN`, `SETTLE_MIN_MS` are gone with them); the rule is flat now, so there is nothing left for a throughput assumption to compute. Reading them is a no-op - they were guesses, and the box proved it. See *The restore race, measured, and the wait that stopped guessing*. |
| `VLLM_E2_TIER_DISK_READ_THREADS` | **how many threads issue spill reads at once (0.2.24). Unset or `1` = ONE thread**, which is every boot before 0.2.24, byte-identically and pinned by a test. Clamped to `[1, 16]`; anything that is not a positive integer means "the operator said nothing" and is reported in `read.env_problems`. It is a thread count and nothing else - the queue stays FIFO, so dispatch order is submission order. **Raise the pool with it or it buys nothing on a big prompt**: in-flight bytes are bounded by the pool, not by the threads. See *Reading the spill in parallel*. |
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
| `VLLM_E2_TIER_DEV_BOOT_INERT` | **the boot opt-out.** Unset (the default) = the boot brings the tier up *working*: at connector creation it installs the backport, arms guard mode, and arms the tier. Set (`1`/`true`/`yes`/`on`) = the old posture: patch, then stay disarmed and unguarded until a verb says otherwise. Same falsy set as `VLLM_E2_TIER_DEV`. It is the only env var that touches either facility, and it can only turn arming **off**. |
| `KVTIER_DISK_BOOT_ENABLE` | **the SSD tier's boot default (0.2.33), and it is ON by default.** Unset (or `1`/`true`/`yes`/`on`) = the boot that arms the RAM tier also **arms the disk spill tier (as a latch since 0.2.35: no expiry) and raises this boot's serve posture**, so the engine comes up `serving_as=disk_authority` with no POST at all. `0`/`false`/`off`/`no` = the posture of every release through 0.2.32: disarmed, `serving_as=off`, and the two `/disk` POSTs are the only lever. It is a **disable switch**, the same shape as `VLLM_E2_TIER_DEV_BOOT_INERT`: it can only take a boot back to manual, and `VLLM_E2_TIER_DEV_BOOT_INERT` (which arms nothing at all) outranks it. The decision is computed fresh from this variable at connector creation and from **nothing else** - it reads no arm file and no state an earlier boot wrote, so a `disk-spill-arm.json` surviving from a previous boot still arms nobody. **Since 0.2.35 there is nothing to renew and nothing to expire**: the arm is a latch, so what the boot raises stays raised until a human disarms it - and a human's `{"op":"disarm"}` (or `{"op":"authority","enabled":false}`) latches a veto that even a SECOND boot inside the same process cannot arm over. One line at boot prints the value, where it came from and how to switch it off (`KVTIER disk boot default=...`); `disk_restore.boot_default` is the payload. See *The SSD tier's boot default*. |

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
| `/disk` | GET, POST | **always** | the SSD spill tier: status, arm, disarm, `spill-one`, `ls`, `contains`, `offers` (RAM-first gap fill) and `authority` (every request served from disk), plus `failures` and `shadow` (the read-only byte-witness, see `disk_shadow.py`) and `gpu-shadow` (the GPU-direct shadow writer, see `disk_gpu_shadow.py`) and `restore-verify` (the restore-side byte verifier, see `disk_restore_verify.py`) and `gpu-write` (the GPU-direct write's counters and latch, see `disk_gpu_write.py` - its answer carries `scope`, because the apply point this op arms is code only a worker runs). A mutating `op` on GET is a `405` |
| `/policy/reload` | POST | **always** | reload `<package dir>/policy.py`, and adopt the store-policy patch |

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
* **It has a heartbeat (0.2.45).** The rank file is this rank's PROOF OF LIFE, and
  `disk_restore` refuses to offer a disk restore on a file older than
  `RANK_FILE_FRESH_SECONDS` (900 s). Writing it only from `get_finished` meant an IDLE
  worker wrote nothing, its proof aged out, and the scheduler stopped serving from disk
  for a worker that was perfectly alive - so the worker writer now runs its own bounded
  daemon thread (`kvtier-worker-diag-heartbeat`, 60 s, floor
  `HEARTBEAT_MIN_INTERVAL_SECONDS` 5 s) through the SAME write path a step uses. Started
  at `install()`, not at the first step. Read `heartbeat_alive`,
  `last_heartbeat_age_seconds` and `heartbeat_beats_in_window`; see
  *An idle engine keeps its disk proof alive* below.
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
H="Authorization: Bearer $VLLM_E2_TIER_DEV_TOKEN"; B=http://127.0.0.1:8000/plugins/kvtier
# what can this boot do?
ssh llm "curl -s -H '$H' $B/disk" | python3 -m json.tool
# arm the tier, then prove the path writes real bytes for one slot. SINCE 0.2.35 THE ARM
# IS A LATCH: `seconds` is still accepted (an existing caller must not start failing) and
# does nothing - the verdict answers `seconds_ignored: true`. It comes down on `disarm`.
curl -s -H "$H" -X POST -d '{"op":"arm"}' $B/disk
# `key` is the engine's GROUP-QUALIFIED block hash: the 16-hex `key16` plus the
# group id as 8 hex chars (`key16 + f"{group:08x}"`, so 24 hex). A bare 16-hex key is
# refused - it cannot say which group's bytes these are. Find both halves with
# `GET /plugins/kvtier/block?block_id=<cpu slot>`: `tier_cpu_slot.block` carries
# `hash_hex` and `hash_group_id`, and any CPU slot is nameable up to the CPU pool's
# size (which is larger than the GPU pool's - the id is bounded per pool).
curl -s -H "$H" -X POST -d '{"op":"spill-one","slot":170,"key":"a3f0b7c1d4e5f60700000003"}' $B/disk
curl -s -H "$H" -X POST -d '{"op":"ls","limit":20}' $B/disk
# ONE IDENTITY, NOT THE WHOLE STORE (since 0.2.21). "Did the span I park actually land,
# and for WHICH RANKS?" - answered from the replayed journals in memory, never by
# walking the store (that is `ls`). One park span is one file PER RANK, so the answer is
# a set of ranks: `ranks_present: 2` on a boot with 4 armed ranks is a half-written span.
curl -s -H "$H" -X POST -d '{"op":"contains","key":"a3f0b7c1d4e5f60700000003"}' $B/disk
# the query-string form does the same thing (and since 0.2.21, so does every other op)
curl -s -H "$H" -X POST "$B/disk?op=contains&key=a3f0b7c1d4e5f60700000003"
# name the four rank FILES of one span without listing the whole store: `shard` is the
# first two hex of the key16 (the filter has always existed; 0.2.21 whitelists the key)
curl -s -H "$H" -X POST "$B/disk?op=ls&shard=a3f0&limit=20"
ssh llm "curl -s -H '$H' -X POST -d '{"op":"disarm"}' $B/disk"
```

**`contains` answers "I cannot answer" as its own verdict, with `present: null`.** Five
of its verdicts are not a "no": a key that cannot name an identity (a `400` at the route,
never a `present: false`), a process with no journal directory, an index that has replayed
nothing since boot, a replay that named no file at all, and a check that raised. That
distinction is the most valuable field on the verb: on 2026-10-05 `entries: 0` was read as
"my span is not on disk" for a whole afternoon when the fact was that nothing had been
replayed into that process yet. A present answer carries the entry's `bytes`, the journal
file it came from, and one row per rank (`rank`, `bytes`, `journal`, `seq`) - which is how
you tell a complete span from a partial one. What it cannot say is each rank's geometry
digest: the journal line is `<seq> <+|-> <key16> <bytes> <group>` and names no digest, so
naming the actual files is `op=ls` with a `shard`, and the answer says so.

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

**A boot with the tier env set serves from disk, and one POST changes who is served.**
Since 0.2.33 the boot default raises the serve posture for you (`KVTIER_DISK_BOOT_ENABLE`,
ON unless the operator turned it off - see *The SSD tier's boot default*); the switches
themselves stay dev-verb only, per boot, and unreachable by a config file or the spill
arm, and `KVTIER_DISK_BOOT_ENABLE=0` restores the pre-0.2.33 posture where **a fresh boot
serves nothing from disk until you POST**:

```bash
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"authority"}' $B/disk  # read
curl -s -X POST -H "x-kvtier-dev-token: $TOK" \
     -d '{"op":"authority","enabled":true}' $B/disk        # EVERY request from SSD
curl -s -X POST -H "x-kvtier-dev-token: $TOK" \
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

Eligibility bounds are untouched by the switch - the bounce-buffer pool budget, the
CPU-slot reserve, the reader queue's job count (`disk_read.MAX_QUEUE_DEPTH`, **256 jobs
since 0.2.35, 64 before**), the price walk's identity cap (`POOL_PRICE_MAX_IDENTITIES`,
**256 since 0.2.35, 64 before**), the per-offer block bound
(`MAX_RESTORE_BLOCKS_PER_OFFER`, **128 spans since 0.2.35, 64 before**; the fixed
32,000-token cap that used to sit beside it was deleted in 0.2.32) and "is this span
actually in the index" still decide how much any single request is offered. The block
bound is a BACKSTOP and has to stay sized above what the bounce-buffer pool can afford, or
it stops being a bound and becomes an undisclosed throughput limit: the operator's 1 GiB
pool affords ~93 buffers, and a 64-block cap sat *under* that - which is why 0.2.35 raised
it. Raising it does not raise `MAX_SLOTS_PER_OFFER` (256 CPU slots), which still binds
first for any multi-group model. Disk authority removes the *policy* stop, not the
physical ones.

**Every one of those job-shaped bounds is now priced, and `bounded_by` names the one that
bit.** An offer used to be sized by pool BYTES alone, while the reader enforces a JOB
count: `disk_read.request_read()` submits into one bounded queue, and the job past
`MAX_QUEUE_DEPTH` is refused `queue_full` - a `LOAD_CAPACITY_FAILURES` member, which means
the bytes are fine and the read never happened, so the block **arrives as nothing**. A
missing block is not a shorter restore, it is a **hole**, and coverage is usable only up to
the first hole: everything behind it is recomputed however much valid data sits there. That
is how a 62-block run came home as a 15-block restore and cost ~19,200 tokens. A 128-span
offer on the live geometry needs 128 + 4 = **132 jobs**, so a 64-job queue made a holed
restore the *expected* case on exactly the box that raises its pool to 1 GiB - and a
1 GiB pool tells the offer "~93 jobs fit", which is above 64 and below 132. The offer now
prices `min(pool jobs, queue jobs)` and `bounded_by` answers `pool_budget` or
`read_queue` depending on which one cut, because a payload that names the wrong bound
sends the operator to buy the wrong thing.

**THE THREE NUMBERS COME FROM ONE DERIVATION.** One offer allocates exactly one CPU slot
per read job (one spilled file fills one destination block), so `MAX_SLOTS_PER_OFFER`
(256) is the largest job count any offer can carry, and the two bounds that were at 64 are
now 256:

| constant | value | why that number | below it, the failure is |
|---|---|---|---|
| `MAX_SLOTS_PER_OFFER` | 256 slots | the offer's job ceiling: one read per slot | shorter offer, labelled `cpu_slots` |
| `disk_read.MAX_QUEUE_DEPTH` | 256 jobs | = the job ceiling, so a maximum offer is **submittable** | **a hole** at job 256, `bounded_by: read_queue` |
| `POOL_PRICE_MAX_IDENTITIES` | 256 lookups | = the job ceiling, so a maximum offer is priced **exactly** | under-priced offer -> over-offer -> **a hole** |

Extra queue slots above 256 buy nothing: only a *second concurrent offer* could fill
them, and concurrent offers are the open reservation arbiter (TODO 14), not something a
queue depth fixes. A capped identity walk is still refused the right to be optimistic:
since 0.2.35 a walk that hits its cap takes the **larger** of its sample and the box's own
width floor (`rank width:` or the fallback constant) and says `walk capped and widened`,
because a partial sample's maximum is not a maximum and an under-price is the direction
that holes a restore. `offers_price_walk_capped` counts it, and zero is the healthy value.

**The memory question, answered, because a queue slot is not free**: `request_read()`
takes the bounce buffer **before** it queues, so a *queued* job holds its buffer. That is
not a second memory budget, because every buffer comes out of the pool and `_take_buffer`
refuses at the pool's byte ceiling first - a 256-slot queue with a 1 GiB pool can never
make the process hold more than the pool. `test_a_queued_job_holds_its_buffer_so_the_pool_is_the_only_byte_budget`
pins it by the refusal that arrives (a 4-job pool against the 256-slot queue refuses the
surplus as `buffer_cap`, never `queue_full`). What the depth costs directly is one
descriptor plus its batch-ledger entry, ~1.2 KB per slot, so the queue itself is **~0.3 MB
per rank process** on top of the pool's own bytes (~1 GiB/worker at the operator's
setting). A 1 GiB pool with a 256-slot queue is therefore coherent in both directions: the
pool binds at 93 jobs, and the queue only becomes the operative bound on a box whose pool
exceeds ~2.7 GiB (256 x 11,468,800 B = 2,936,012,800 B) - and if it does, `bounded_by` says so.

**The ordered list of bounds that can shorten a restore**, with what each permits at the
live geometry (`per_span` = positional groups, four align-mode groups contribute one
boundary-state read each; block size 3,200 tokens; pool = 1 GiB priced at the 11,468,800 B
PP0 shape):

| bound | value | one-positional-group model | three-positional-group model |
|---|---|---|---|
| store depth (contiguous run in the index) | measured, not a constant | **~11 spans measured live** - the real limit | ~11 spans |
| bounce-buffer pool | 1 GiB / 11,468,800 B = 93 jobs | 93 - 4 = **89 spans** (284,800 tok) | (93 - 4) / 3 = **29 spans** |
| CPU slots, live free count | `(free - reserve - boundary) / per_span`, where **the reserve is `VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE` since 0.2.38 (default 64)** | the engine's own pool: **98 spans** (313,600 tok) at the operator's 166-slot PP1 arena, 119 at the 187-slot PP0 one. At **reserve 16**: 146 spans of arithmetic on PP1, which the 128-span walk cap below then trims to 128 | **32 spans** at 166 free slots (48 at reserve 16) - binds before 84 |
| CPU slots, the cap | `MAX_SLOTS_PER_OFFER` = 256 | (256-4)/1 = 252 spans | (256-4)/3 = **84 spans** |
| reader queue | `MAX_QUEUE_DEPTH` = 256 jobs | 252 spans (ties the slot cap exactly) | 84 spans (ties it) |
| block cap | `MAX_RESTORE_BLOCKS_PER_OFFER` = 128 | **128 spans** (409,600 tok), reachable only once the pool affords 132 jobs (~1.4 GiB = 1,513,881,600 B) | unreachable (84 < 128) |
| identity-walk width | `POOL_PRICE_MAX_IDENTITIES` = 256 | 132 identities named (one per job): never truncated | 256 named at 84 spans: exactly at the cap |

Read it as: the pool is the operative bound today on both geometries (89 and 29 spans),
the store's own depth is what actually shortens a live restore (~11 spans), and the queue
now ties the CPU-slot cap instead of undercutting it - so for a multi-group model the CPU
slot bound, at 84 spans (or 32 once the arena's *live* free count and the reserve are
taken off a shrunken 166-slot arena), is the wall a raised pool will meet next. Nothing
in this change moves `MAX_SLOTS_PER_OFFER` or its clamp semantics.

**The CPU-slot reserve in that row is an env var now (0.2.38), and it has a floor below
which it buys nothing.** Lowering `VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE` raises the
affordable span count by exactly the slots it gives back - 98 to 146 on the wide rank
between reserve 64 and 16 - but the positional walk stops at
`MAX_RESTORE_BLOCKS_PER_OFFER` (128 spans), so a reserve below ~34 on a 166-slot rank
(and ~55 on a 187-slot one) cannot make a *single* offer any longer: it only spends the
headroom the RAM tier was given. See *Sizing the CPU-slot reserve an offer leaves alone*.

**A failed disk load is not offered twice, and since 0.2.32 a raced load is.** The
engine recovers from a KV load failure by resetting the request and rescheduling it,
which asks `get_num_new_matched_tokens` again - and the journal has not changed, so an
amnesiac restore answers with the same offer and the same failure. That loop is what
starved the box. `suppress_load` fuses the **request** (named exactly by
`connector_output.failed_recving`) and the **span** (its first block's key, so a
replayed prompt under a fresh request id is refused too), on any of: a failed load whose
kinds say DATA, invalid block ids on a step that carried our slots with the same kinds,
a load that outlived `RESTORE_DEADLINE_SECONDS`, or a request returning at or before the
point where it was last offered with no re-offer left. `suppressed_after_failure` is the
refusal; `suppressed_hits` / `suppressed_spans` / `suppressions` say which half fired,
and `races_not_fused` / `fuses_on_data_kinds` / `fuses_on_unknown_kinds` say WHY a report
did or did not fuse (see *What a failed load costs*). The table is bounded
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

**A marked request is self-evident, and a closed gate is counted (both 0.2.21).**
**The posture is self-evident, and a closed gate is counted.**
`restore.disk_authority`, `restore.offers_enabled` and `restore.serving_as` are
published beside `restore.authority_lookups` and `restore.lookups`, so "is this boot
serving everyone from SSD, and did the rule actually fire?" is one payload read with no
control experiment: `lookups` moves for every lookup, `authority_lookups` only for one
that ran while disk authority was up. And a request turned away by the
**gate** now moves its own bucket - `refused.not_configured`, `refused.disarmed`,
`refused.tier_disarmed`, `refused.offers_disabled` - in the same region that counts
`workers_not_servicing` and `no_worker_proof`. Before, a gate decline wrote `last_reason`
and no counter, so an empty `refused` could not be told apart from "nothing refused us":
that ambiguity is what made an alignment-guard refusal look like the answer on
2026-10-05 when the gate had refused the request. `last_reason` is written exactly as
before (something may parse it), and the value handed back to the scheduler is the fork's
own tuple, unchanged - the change is bookkeeping, not behaviour.

**Disk authority is what makes the read path testable at all (0.2.22 as a marker rule,
0.2.31 as a boot switch).** With RAM consulted, an ordinary small
prompt either hit RAM completely (nothing left to offer, `disk_run_below_one_block`) or
hit it with a remainder (`ram_hit_not_block_aligned`) - so exercising one disk read
meant making GPU and RAM forget the prefix first, ~1.3M tokens of filler per attempt.
Under disk authority a request needs only that the span is in the index, which is why
the operator's one-command way to exercise the read path is now the authority op instead
of a marker on every probe. The posture is also what a whole-workload SSD measurement
needs, and it is now one decision an operator makes rather than one every client has to
cooperate with.

Three limits, stated because they are the ones an operator will hit:

* **Authority does not zero `num_computed_tokens`.** That number is the third field of
  the fork's pending tuple and `update_state_after_alloc` (`manager.py:441-463`) uses it
  as the *offset* into the request's whole block list - it is where the CPU-to-GPU copy
  writes. Zeroing it while the scheduler still counts a cached local prefix would DMA
  disk bytes into the request's already-cached, ref-counted GPU prefix blocks (shared
  with every other request reading them) and leave the newly allocated external region
  holding whatever was in those slots. So a request whose prefix the GPU already
  computed has nothing to restore and recomputes; the disk span starts where the local
  prefix ends, always.
* **Ignoring the RAM hit means releasing the pin it came with.**
  `get_num_new_matched_tokens` (`manager.py:360-372`) `touch()`es every CPU block it
  found and leaves the tuple in `_pending_cpu_hits`; a disk-authority offer overwrites
  that entry, so `disk_restore._release_fork_pin` calls the fork's own
  `_free_pending_cpu_hit` at the commit point - after the write, so every earlier
  failure path still leaves the fork's tuple intact. A manager without that method is
  refused by name (`refused.pin_release_unavailable`) rather than allowed to leak a
  pinned span per request. `pin_release_failures` must stay 0.
* **No token is claimed twice.** The offer's token count is the disk's alone (never
  `ram + disk`), and the merged tuple the fork consumes contains only our freshly
  allocated slots - no RAM block ids appear in it. The scheduler recomputes whatever
  neither tier covered.

Read it in the payload: `restore.last_offer` carries
`ram_tokens_reported` (what the fork found) beside `ram_tokens_used` (zero, while disk
authority is up), `start_block`, `blocks`, `disk_tokens`; `restore.authority_lookups`
counts the
lookups the rule fired on; `restore.local_hit_ignored` and
`restore.local_hit_tokens_ignored` count only OFFERS - a refusal leaves the fork's hit
exactly where the fork put it, and must not be counted as a sacrifice the request never
made. A forced read also logs one `DISK[restore] serving ... from DISK against a
N-token host-RAM hit` warning; a request the disk cannot serve stays silent and
leaves no allocation behind.

**The index now reads the newest spill lines first, and publishes how far behind it is
(0.2.22; the rule below was replaced by the 0.2.28 drain - kept here as history).** `refresh()` used to
advance a per-journal cursor from byte 0, sharing one
512 KiB / 8,192-line budget across every rank's journal. On four ~60k-line journals that
covered about a tenth of one rank per pass, so the lines the restore path was waiting
for - spills from the last few minutes, which is what "the GPU forgot this and RAM never
had it" means - were the last thing it would reach, and it was reached only by a request
that was itself waiting on the answer. Measured consequence: **0 of the 36 newest
identities in the store were in the index**, while `index.entries` sat at 6,892 claiming
files the ceiling had long since reclaimed.
The rule now: a refresh spends at most the same caps, but takes its window from the END
of each journal's unread run, and every journal gets a *share* of the budget rather than
whatever the previous rank left. Whatever that forfeits is older than everything it
applies - which is what makes forfeiting safe: a `+` we read is a file we saw written, a
`-` we read for an identity we never learned is a no-op, and a jump forgets the entries
its skipped region taught it (`_forget_rank`), so the index claims only what it read.
Nothing was forfeited silently: `index.lag_bytes` (unread journal bytes; 0 means every
cursor is at end-of-file), `index.lag_journals`, `index.catch_up_jumps`,
`index.bytes_skipped` and `index.lines_skipped` are published, a refresh skipped by the
1 s floor re-publishes the last *measured* lag with `lag_measured: false` rather than
claiming zero, and `restore.why_no_restore` names the lag when `no_disk_run` is the
answer. While `lag_bytes` is non-zero, a `not_present` from `op=contains` is a statement
about the reader and not about the store - and `op=contains` now carries those fields in
its own `index` view for exactly that reason.

**Why the offer path cannot stat the store itself.** The scheduler does not know the
file's name: four ranks name one identity four ways (the digest half covers rank-local
`rank` and `total_bytes_per_block`), which is why `_identity_path` sends a sentinel and
`_resolve_worker_path` rebuilds the name per rank. So a stale entry cannot be caught by
a `stat()` in the scheduler; it is caught where the name is known - the worker's read,
which classifies `file_missing`/`size_mismatch` and reports the GPU block as a load
error, and the scheduler recomputes. That is the cheap honest gate, and it costs one
wasted load round trip per stale entry, never wrong KV.

**The restore race, measured, and the wait that stopped guessing (0.2.23, flat since
0.2.32).** Reads are queued at `bind_connector_metadata` and applied at
`start_load_kv`, and the fork calls **both inside one engine step**
(`kv_connector_model_runner_mixin.py:80-93`). Because every disk hit is offered with
`load_kv_async=True`, `has_sync_kv_loads` stays false and the fork defers
`start_load_kv` to after the forward **launch** - so the window a read has to win is one
CPU-side forward launch, not a forward pass. Until now the only evidence about that
window was one counter (`dma_unsettled_entries`) plus a `read_seconds_max` of `0.0` that
reads as "no read ever ran" and actually means "no read that reached the kernel ever
finished" - a queued job's buffer is handed back before the reader thread dequeues it.

The apply point now publishes the window itself. `restore.apply_window.last` is the most
recent batch as the engine saw it: `elapsed_ms`, `accepted`/`expected`, `running`,
`queued`, `settled`, the **measured bind→apply gap** (`bind_to_apply_ms`, stamped at the
first line of `_worker_bind`), `bytes_asked` beside `bytes_at_stake` (the bytes still in
flight, which is what a wait is waiting for), `budget_ms`, `waited_ms`, `bound`,
`settled_after_wait`, and `allowance_spent` (did this batch use the whole grant and still
have bytes in flight - the field that replaced the arithmetic-based `over_recompute`,
which could not see a dropped load). `windows` carries
`min`/`max` **boot-wide**, an `avg`, and a `median` over the last `MAX_SETTLE_SAMPLES`
(512) observations that publishes `median_samples` so a truncated median can never be
misread as the boot's. `read_seconds_avg`/`read_seconds_max` sit beside
`waited_ms` on purpose: either number alone has no unit of comparison.

A drop now says **which kind** it is. `dropped_queued` counts entries whose read had not
started (the job sat in the queue and the engine got there first - thread scheduling on
the proxy host, and the kind a wait fixes); `dropped_running` counts entries whose read
was inside the kernel and still lost (a device slower than the step, which a wait only
fixes if the budget beats the read); an unreadable per-job state table is
`dropped_unattributed` and never folded into either. `applied_settled_immediately`
counts the entries the box already wins for free, so `settled_after_wait` over it is the
wait's hit rate.

The wait itself is **flat, bounded, and an allowance rather than an arm** - an unbounded
wait inside an engine step is a hang wearing a performance patch. The rule, as it reads
in `disk_restore.settle_budget`:

```
granted_ms = min(VLLM_E2_TIER_DISK_SETTLE_MS, the batch's own seconds_left)
             # the read's expiry, from LOAD_DEADLINE_SECONDS: the same clock that
             # cancels the read, so a wait can never outlive the bytes it waits for
```

That is the whole formula - one number the operator sets (`SETTLE_MS_DEFAULT`, **5,000
ms**, overridable by `VLLM_E2_TIER_DISK_SETTLE_MS`) and the load's own deadline, the only
other clock with a claim on the wait. `SETTLE_POLL_SECONDS = 0.0005` is a poll and never
a condition wait (the reader thread owns those bytes and the engine thread must not queue
behind its lock), `SETTLE_MS_MAX = 60000` clamps a typo with `ceiling_clamped` saying it
happened, and `0` is still reachable and still means never wait.

**Every term of the old formula was a guess, and the live box said so.** Through 0.2.31
the allowance was `clamp(bytes_in_flight / read_bytes_per_s * 1000 * safety_margin,
SETTLE_MIN_MS, cap)` - an assumed read throughput (it was the *write* rate, because the
read side had never been measured), an assumed safety margin, a floor of one scheduler
tick, and an assumed prefill rate used only to publish the cost a wait was racing. On
2026-10-06 a live request with a 28,583-token prompt had 12 jobs / 137.6 MB offered and
the formula granted **23.6 ms**; the measured bind→apply race gap for the batches it
lost was **89.7 / 97.1 / 129.7 / 135.8 ms**; the request recomputed **28,583 tokens
(12.26 s, `cached_tokens: 0`)** to save 23.6 ms, and the loss was `drop_running` - a
kernel read mid-flight, bytes perfectly good underneath it. A budget that gambles
against a 12-second recompute has no winning side, so the arithmetic is deleted:
`SETTLE_MIN_MS`, `SETTLE_SAFETY_MARGIN`, `SETTLE_READ_BPS_DEFAULT`,
`PREFILL_TOKENS_PER_S_DEFAULT`, `VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S` and
`VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S` are gone, and `bytes_at_stake` is PUBLISHED
beside the grant while deciding nothing - it is the quantity a reader of
`last_apply_window` wants, not an input. Which bound applied is published per batch as
`wait.bound` in exactly three values - `flat` (the allowance decided it), `deadline` (the
read's own expiry did, so a drop here is a deadline problem and not an allowance
problem), `off` (this boot does not wait) - and the three are counted boot-wide.

**The counter that could not see the loss is replaced by one that can.**
`budget_over_recompute` compared a grant with the arithmetic that produced it, so the
23.6 ms grant above reported `over_recompute: 0` the whole time it was costing twelve
seconds. `wait.allowance_spent` asks the only question a flat allowance can answer - did
this batch consume the WHOLE granted allowance and still have bytes in flight? - and
`settle_allowance_spent` is its boot-wide count. It is the field that says "5 s is not
enough on this box" without anyone measuring a throughput.

**The default is ON, which is a behaviour change on an unchanged command line.** Before
0.2.32 an unset variable meant never wait; now it means a 5 s ceiling, so a boot started
the old way will wait where it used to drop. The bound is the load's deadline, not the
cap, so a wait can never outlive the read - but a step that used to cost 0 ms of waiting
can now cost up to the race gap, which is exactly the trade the operator asked for ("give
it up to 5 s to read; only when something bad happens do we ignore"). `0` puts the old
boot back. An unset `VLLM_E2_TIER_DISK_SETTLE_MS` therefore announces what it settled on
once per boot (`DISK[settle] wait=ON allowance_ms=5000 (VLLM_E2_TIER_DISK_SETTLE_MS is
unset; the default since 0.2.32 is 5000 ms - set it to 0 for the pre-0.2.32
never-wait boot)`) - the line NAMING the ignored variable is mandatory, because "the
operator wrote 40 and the box waited 5 s" must never be a silent reading. A value that
will not parse falls back to the default and says so, naming the variable and the value,
once per boot, in `env_problems` as well as the log.

**What the counters exposed on the way.** Every failure on the live box arrived as
`invalid_block_ids` with an EMPTY `failed_recving`, and that path retired its ledger
entry with a hand-rolled `_loads.pop()` instead of `_drop_load` - so `loads_released`
never moved (the honest-looking reading was "the load was never registered" rather than
"it was registered and burned") and the load's slot→path mappings outlived it, which is
a slot the next `build_load_paths` will emit a path for while the fork may already be
using it as a host-RAM source. Both failure shapes now close the ledger through
`_drop_load` (or `_burn_load`, when the loss is a data failure and the request is fused
- see *What a failed load costs* below), and `no_read` deliberately does **not** move at
burn time (the load is already counted as a failure - moving both would double-book one
release). A `finished_recving` that arrives *after* a burn is counted as `no_read`
rather than vanishing, because it is the fork calling a load finished whose bytes this
plugin knows it never delivered.

**Running the experiment.** Boot as usual and read `apply_window.windows.bind_to_apply_ms`
- that distribution is the answer to "how big is the window", and since 0.2.32 the
default boot gathers it *while waiting*, so the same boot answers both "how big is the
race" and "did the wait win". Read `wait.settled_after_wait` against
`wait.dropped_after_wait`, `wait.bound` and `wait.allowance_spent`. A loss that is
`dropped_queued` is the wait doing its job late; a loss that is `dropped_running` with
`allowance_spent` true is a device slower than the allowance, and `settle_bound_deadline`
rising with it says the read's own expiry, not the cap, ended the wait.

### What a failed load costs: a race may lose its tokens, a fault loses the offer (0.2.32)

Through 0.2.31 every failure report called `suppress_load`, which black-holes the
**request** and fuses the **span** for the rest of the boot - whatever the failure was.
The reason string handed to that call already contained
`sorted(_recent_failure_kinds())`: the code had the answer in its hand and decided
without looking. The night the settle formula was measured, that meant a span that lost a
135.8 ms race in one group of one span cost a 28,583-token recompute **and**
`suppressed_requests: 3` - a boot-length fuse for a millisecond-scale timing loss, with
no expiry to wait out.

**The 26 failure kinds are partitioned, and the partition is the policy.**
`LOAD_TIMING_FAILURES` (`not_settled`, `deadline_expired`) and `LOAD_CAPACITY_FAILURES`
(`queue_full`, `buffer_cap`) say *the bytes are fine and the moment went wrong*; every
other name is `LOAD_DATA_FAILURES` - a wrong digest, a short read, a size mismatch, a
failed copy, a step whose lists disagree, a broken read path, or a loss nobody
attributed. `failure_kind_classes()` publishes the three lists with the decision rule
beside them so an operator can argue with the partition instead of reading source, and
`test_disk_restore_races.py` pins every name into exactly one class.

**The rule.** Any data kind visible means the fuse stays exactly where the 2026-10-05
incident put it (request and span, forever). Only timing/capacity kinds mean the request
keeps its next offer and the span is not implicated. NOTHING visible means `unknown`, and
`unknown` fuses: the scheduler learns a worker's kinds through the rank file, which lags
(`PROOF_CACHE_SECONDS`), so guessing "race" when the evidence has not arrived is how the
loop that starved the box comes back. Every decision is counted - `races_not_fused`,
`fuses_on_data_kinds`, `fuses_on_unknown_kinds`, `race_reoffers_granted` - and the last
one is written as a sentence in `last_fuse_decision`, because "why is `races_not_fused`
zero?" has to be answerable without a control experiment.

**The engine's report shape is not evidence.** `failed_recving` and `invalid_block_ids`
named the CHANNEL the bad news arrived on; `_note_load_failure` files any name outside
the `WORKER_FAILURES` vocabulary under `read_failed`, which is a data kind. A classifier
reading the plain ring would therefore see a corruption on every report ever written and
would never spare a race - the fix would be inert while looking alive. The classifier
reads only what the code that touched the bytes wrote (`where` = `apply`, `bind`,
`offer`, locally or carried by a live rank file) and excludes `where="report"`.

**The loop stays bounded without the permanent fuse.** A spared request gets exactly one
re-offer (`MAX_RACE_REOFFERS_PER_REQUEST`, spent when used, never refilled), so no
request is ever offered more than TWICE per boot whether its losses are races or faults -
against the 123,474 offers in 3.9 minutes that starved the box on 2026-10-05.

**The engine blocker, named rather than worked around: on this box a hole costs the whole
prompt, and a RACE costs it too.** The plugin-side half of the shrink is in place - an
entry without bytes is pruned, its GPU block reported, and the entries that landed stay in
the step, so the contiguous prefix survives the hole. What the engine does with the
reported blocks is not ours to choose: under
`--kv-transfer-config '{"kv_load_failure_policy":"recompute"}'` the scheduler
(`vllm/v1/core/sched/scheduler.py:3193-3215`) keeps `num_computed_tokens = 0` (the
assignment is at `:3205`) for a
request whose KV cache config has MORE THAN ONE group - this box runs 5 serviceable
groups, so all 12 reported blocks collapse to one contiguous prefix of ZERO tokens and
the whole prompt recomputes. The shrink-to-a-prefix mechanism exists only on the
single-group path (`scheduler.py:3217-3252`, with the truncation in
`_get_blocked_tp_blocks`). So on this deployment the flat allowance is not a way to
recompute a little less - it is the reason to wait at all: a race lost costs the full
recompute either way, and a 5 s wait costs the tokens only when it fails. `partial_load_verdict`
publishes that reading beside the counts.


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

**What to read on the next live run**, all of it under `restore.read`:

| field | the question it answers |
|-------|-------------------------|
| `threads_configured` vs `threads_alive` vs `threads_requested` | are the threads the operator asked for alive, and was the request clamped? |
| `threads[]` (`jobs_read`, `bytes_read`, `read_seconds`, `in_flight`, `peak_in_flight`, `starved_polls`) | is the queue spreading work, and did the units overlap? `peak_in_flight` can only be 1 with one thread, so it is the field that separates a parallel boot from a parallel-looking one |
| `throughput.bytes_per_reader_second` | the **device** rate. It should stay flat as threads rise |
| `throughput.bytes_per_second_wall` | the **wall** rate the engine step pays. It should rise, toward N x the single-reader figure |
| `throughput.parallel_efficiency` | reader seconds over wall seconds: ~1.0 with one reader, ~N when N truly read together. **1.0 with N configured is the signature of "threads configured, parallelism absent"** |
| `pool.held_bytes` / `peak_bytes` / `headroom_bytes` / `units_of_11468800` | was the pool the limit, and how many units could it ever hold? || `pool.measured_unit_bytes` / `units_of_measured_unit`, `concurrency.threads_useful_max` | the unit size this boot actually read and the thread count that pool could feed - `threads_configured` above this number is dead weight, and the field says so instead of a reader doing the division |
| `refused.buffer_cap` / `reader_starved_polls` / `concurrency.starved_on` | **starved on buffers** vs **starved on queue** vs (neither) - i.e. disk-bound |
| `units_in_flight_peak`, `queue_depth_peak` | the high-water marks for the whole boot |
| `disk_restore.worker_report()["readers"]` (rank file) | the same five facts from the only process that actually reads: `configured`, `alive`, `peak_units`, `starved_polls`, `buffer_refusals`, `out_of_order` |
| `disk_restore.worker_report()["spans"]` (rank file) | `seen` / `whole` / `partial` / `empty` / `hole_in_middle`, the span shape of what survived the apply point |

**The lock discipline, kept with N threads.** `_state_lock` is a plain (non-reentrant)
`Lock`, held only for dict updates - a few hundred nanoseconds - and never across a
queue wait, a `sleep` or a file read. That is why the engine's step thread can always
call `batch_status`/`batch_buffer`/`request_read` without queueing behind a kernel read,
and it is the same reason `_settle_wait` in `disk_restore` polls with `time.sleep`
instead of waiting on a condition (see `SETTLE_POLL_SECONDS`). Witness bookkeeping is
kept out of the lock entirely by pre-growing the thread/slot tables to the ceiling once,
so a reader thread's per-thread counters are indexed writes on pre-existing dicts. A
second rule joins it: **a reader claims its job's buffer inside the lock and is then its
only owner**. Before this change `_handle_read` read `job["buffer"]` *before* taking the
lock, so a `cancel_batch` that won the race returned that buffer to the pool and the
reader returned it a second time on the way out - the pool holding one buffer twice, and
the next two reads sharing it. That was reachable with one reader; four make it more so.

**Dead threads, cancellation, and the reading of "dead".** `_ensure_thread()` returns
False only when **not one** reader is alive, and `begin_batch` turns that into
`thread_dead` so a batch is never opened while nothing could read it (`reads_started`
cannot be nonzero with no reader). Three dead slots and one live one is a degraded boot
that still reads, so it publishes `threads_alive: 1` against `threads_configured: 4`
instead of refusing. A dead slot is respawned at the next `begin_batch`. Cancellation is
unchanged in meaning and safe with N: `cancel_batch`/`finish_batch` recycle QUEUED, DONE
and FAILED buffers and leave RUNNING alone (the kernel read cannot be stopped, and its
bytes are dropped by `_settle` because the batch is gone). `stop_reader()` now posts one
sentinel **per live thread** - one packet frees one waiter.

**The ordering question, which is a correctness question and is NOT decided here.**
Submission order is span-major and group-minor (`_merge_pending_hit` walks
`for span: for group:`), and a restore is only useful when a span is complete across its
groups. With ONE reader, completions followed submission order, so a batch caught
mid-flight was missing a **suffix** of spans - a hole at the end. With N readers the
answers interleave and the same mid-flight batch can show span 3 whole while span 1 is
still queued: **a hole in the middle**. Whether a partial restore may have its tokens
counted as computed by the engine is a separate, live question and this release decides
nothing about it - no pruning rule, no token arithmetic, no all-or-nothing logic is
touched. What it publishes is the shape, so that question is argued with measurements:

* `disk_read.batch_status(...).completion_order` - the job ids in the order they
  answered. Equal to `[0, 1, 2, ...]` exactly when the reads behaved like the
  single-thread boot; `out_of_order_completions` counts the answers that landed before a
  lower-numbered job's did (boot total: `completions_out_of_order`).
* `disk_restore.span_shape(batch, failed)` and the counters `spans_whole` /
  `spans_partial` / `spans_empty` / **`spans_hole_in_the_middle`**. The last one counts
  batches whose applied spans are NOT a prefix of the offer; with one reader that is
  arithmetically impossible, so **a rising count IS the concurrency finding**.
* The disorder is BOUNDED, and the bound is a property of the FIFO queue, not a hope.
  Dispatch order IS submission order - no job dequeues while an earlier one is still
  queued, so a job can only complete ahead of the earlier jobs **already in flight when
  it started**, which is at most `threads - 1` of them. The scatter is a window of N,
  never the whole batch. Pinned by `test_completion_order_under_four_threads_and_the_fifo_bound`,
  which measures the over-take from inside the fake read (the job states as each read
  begins) rather than inferring it from the completion list afterwards.

**What concurrency does NOT fix, and what it shifts.** `incomplete` is decided at bind:
`accepted < expected`, where `accepted` is fixed by `request_read` and `expected` by
`begin_batch`. A bind-time `buffer_cap` refusal therefore makes a batch incomplete
**whatever the thread count**, and the forfeit rule (`disk_restore.py:3674-3678`) takes
that batch's accepted entries with it - 24 on the live boot, every one of them read and
memmoved successfully. Threads are not the lever for that: the levers are a pool big
enough that nothing is refused, and up-front offer clamping with per-entry (not
per-batch) poison inheritance, which is deliberately not in this change.

What N readers does change is the shape of "too late", and the prediction is derivable
from the code rather than guessed. Readers pull eagerly, so N of them convert QUEUED to
RUNNING N times faster: at the apply point the units that have not landed are more often
**inside the kernel** than **still queued**. Both `dropped_queued` and `dropped_running`
should FALL in absolute terms (the whole batch drains sooner), but the ratio between
them moves the other way - `apply_window()`'s split is where to read it. And because the
same drain is faster, the settle wait (`SETTLE_CAP_MS`, derived from the measured
recompute cost and not from any thread count) should be *granted less often and for
less time*: `settle_waits_granted` and `waited_ms_avg` are the witnesses for that, and if
they do not move, the reads were never the thing that was late.

One live shape is worth pinning because the running boot is the degenerate case: with
`per_block = 1` (g5 only) every job in a batch names the SAME group, so the span walk
makes each entry its own span and `spans_partial` can never fire - the only shapes left
are whole and empty, and `applied_is_prefix` degenerates into "are the applied units a
prefix of the offer", which is precisely the question the token-counting analysis asks.
The multi-group shape that `spans_partial` was written for appears the day g0-g3 return
to positional stability. `test_the_single_eligible_group_shape_is_the_live_one` pins
both halves of that.

Tests for all of this live in `tests/test_disk_read_threads.py`: the pin that the
defaults are the old single-thread boot exactly, a barrier test that four readers really
do sit inside four reads at once (and that one reader cannot), cancellation and
pool-exhaustion behaviour under concurrency, the completion-order evidence above, and a
teardown that JOINS every reader thread - because a thread parked inside one test's
stand-in read wakes up calling the NEXT test's stand-in and charges its read to that
boot's counters, which is how the barrier test first went red with 3 reads attributed and
`reads_completed == 4`.

### The ceiling walk, the sample, and the store that grew past its ceiling (0.2.25)

The box measured this on a live boot: `du -sxb /home/user/kv-tier/disk` = **334.9 GiB**
against `VLLM_E2_TIER_DISK_BYTES` = 206,158,430,208 (**192 GiB**), **29,532 files**, and
the oldest surviving file **2.8 hours** old. Eviction was running, and the store was
still growing ~145 GiB an hour into 122 GiB of free space. Both halves were true at
once, because one bounded walk was being read as a measurement.

**What the walk was, and what it silently was not.** `_walk_store` stats at most
`MAX_WALK_FILES` (20,000) files, so on a bigger store it measures a *sample*. The old
walk took that sample depth-first - its file counter was never reset per shard, and its
own `truncated` flag broke the shard loop - so it measured the FIRST shard directories
and stopped. Two failures came out of that one shape, and they point the same way: the
byte total came from ~2/3 of the key space, so the trigger compared a floor against the
ceiling; and a file the walk never opened could not enter `lru_eviction_plan`, so the
`~1/3` of the key space it never reached became a sink that spilled into forever and
could never be shed. Measured on that shape: the old decision number was **218.7 GiB**
for a **326.2 GiB** store, which makes the *effective* ceiling **286 GiB** - and it
rises with the store, because the sampled fraction keeps shrinking.

**The fix is two bounded passes, and a sample that is honest about being a sample.**

1. `_count_store` counts the capacity-visible files with readdir only - no `stat` - so
   the walk knows the size of the population it is about to sample. Bounded by
   `MAX_WALK_ENTRIES` directory entries (100,000, five times the `stat` cap, per pass),
   and bounded in ENTRIES rather than files on purpose: the `stat` cap counted measured
   files, so before this the *iteration itself* was unbounded, and a directory of
   entries this tier will not measure walked to the end inside the spill writer's
   enforcement path. Cost, measured on the shape above: **29,532 entries in 0.099 s**,
   zero `stat` calls.
2. The measuring pass then stats a **proportional** sample: shard `i` is asked for
   `cap_files * files_i / total_files` of its files, so every shard is sampled at the
   same fraction of itself and the whole budget is spent however the files are spread.
   Proportional rather than even-per-shard because an even split wastes the budget on
   the small shards. The plan built from that sample can name a file in any shard, and
   enforcement runs after every committed write, so a partial pass is a pass and not a
   verdict: on the shape above the estimate lands **326.2 GiB against a 326.2 GiB
   truth (0.00% error)**, and the measured catch-down of the real store is several
   passes, each capped by `disk_layout.MAX_EVICTION_PLAN` = 4,096 files - see the table
   below.
3. `_enforce_ceiling` decides on `estimated_total_bytes` - the sample scaled by
   `total_files / files` - and never on the floor. A truncated walk that stat-ed
   *nothing* (`sample_unmeasured`) is declined outright: the estimate is kept, nothing
   is planned, and the reason says so, because a 0 read as "the store is empty" is the
   most dangerous number this module can emit.

**A store inside the caps is measured, not estimated, and byte-identically to before.**
When the walk sees every file, `estimated_total_bytes` IS `used_bytes` - the same
number, the same entries, the same eviction order. `test_a_store_inside_the_caps_is_
measured_and_not_scaled` and `test_the_eviction_order_of_a_measured_store_is_the_oldest_
first_order_it_always_was` pin that, because this tier is live and an eviction order
that moved for a 19,000-file store would not be an acceptable fix.

**The witness.** `spill.ceiling_measurement` (and the same fields in
`disk_spill.counters()` as `walk_*`) publishes `basis`
(`full_measurement` | `sample_scaled` | `sample_scaled_count_extrapolated` |
`sample_unmeasured`), `measured_bytes`, `sampled_files`, `total_files`,
`entries_examined`, `skipped_files`, `walk_truncated`, `count_truncated`,
`plan_truncated` and `ceiling_checks`. "The store is 170 GiB" and "I stat-ed 20,000 of
29,532 files and extrapolated 170 GiB" are now two different payloads. `skipped_files`
also finally keeps a promise the old walk made in a comment and never kept: files this
tier can neither measure nor shed - a `.part` left by a killed spill, a malformed name,
someone else's file - are counted instead of being invisible, which matters because
they are real bytes that the ceiling will never reclaim.

**The ceiling had to be woken up first.** `estimated_bytes` is a **per-process** counter
of this process's own committed writes and starts at 0, and the walk that could
correct it only ran once that counter *alone* reached the ceiling. A rank booting into
a store somebody else filled was therefore blind to it in proportion to how full it
already was: on this box a fresh boot had to spill **192 GiB of new files (17,384 of
them, ~48 minutes at the measured ~4 GiB/min)** before it was allowed to *look* at a
335 GiB store with 60 GiB of `/home` left. The filesystem fills long before the
ceiling wakes up, and honest scaling inside the walk buys nothing when the walk never
runs. So the first committed spill of a `(root, ceiling)` pair takes **one** bounded
walk and adopts its answer (`_maybe_seed_estimate`, writer thread only, 0.22 s once
per configuration, published as `ceiling_measurement.seed_walks`); a store that cannot
be read leaves the running estimate alone instead of writing a 0 into it, and is
attempted once, not per spill. With that, the recovery the ceiling exists to perform
actually happens: seed, then shed toward the target.

**Convergence, measured by replaying the live store.** By the time the file list was
taken the store had grown to **34,896 files / 395.8 GiB** (256 shard dirs, mean slot
12,178,780 B). Replaying that exact list - names and per-file sizes, rebuilt sparse in
a scratch directory - through the real `_maybe_seed_estimate` / `_walk_store` /
`_enforce_ceiling` / `lru_eviction_plan` / `execute_unlink_plan(dry_run=False)` at the
live 192 GiB ceiling:

| | old code | 0.2.25 |
|---|---|---|
| decision number | 225.4 GiB (20,000 files, DFS floor) | **395.8 GiB** (+0.01% vs truth) |
| ceiling actually enforced | 337 GiB | 192 GiB |
| passes to get under it | never (each pass was aimed at 202 GiB from a floor) | **5** |
| freed | - | **19,292 files / 223.0 GiB** |
| final store | - | 172.8 GiB = the `0.9 x` target, on the nose |
| `/home` free | 61 GiB | **284 GiB** |

Each pass sheds at most `MAX_EVICTION_PLAN` = 4,096 files (~47 GiB here) and cost
**0.456 s of writer-thread time** measured end to end (a 0.22 s walk plus the unlinks);
the whole 5-pass catch-down was **2.28 s**. Passes are triggered by committed writes, so
the catch-down needs 5 spills, and the steady state is one pass per ~1,700 files (~19 GiB)
- one walk every ~5 minutes at the measured ~4 GiB/min, about 0.1% of the writer thread.
One pass sheds ~11 minutes of growth: out-running the enforcement would take more than
4,096 files/minute - about 47 GiB/min. Measured on that filesystem with the store's own
layout: 4,096 `unlink()` = 0.050 s (12 us each); `disk_index` appends the removal line
without `fsync` (~4 us), so the unlinks are not the cost. Had the journal fsync'd per
line it would have been 1.4 ms x 4,096 = 5.6 s per pass - it does not, and that is
load-bearing for convergence, so it is stated rather than assumed. During the
catch-down the writer thread pays that cost per spill, which throttles the spill path
itself while the store is over the line; that self-throttling is the point, and it holds
at any spill rate below ~47 GiB/min.

**And the ceiling only runs while the tier is armed.** Enforcement is called from the
writer thread after a committed write, so a disarmed tier with a full store reclaims
nothing - correct by design (the ceiling is a property of the spill path, not a janitor),
but it means "`/home` is 99% full" and "spill is disarmed" together leave the 396 GiB
exactly where it is until the tier is armed again. At the measured pass cost, re-arming
reclaims 223 GiB in about 2.3 s of writer-thread time.

Tests: `tests/test_disk_ceiling_walk.py`, built on a fixture store whose bytes sit in
the LAST shard directories in `os.scandir` order - the exact shape the old walk was
blind to - plus a fresh-boot fixture whose own writes are one slot against a full
store. Every one of its 29 tests fails against the pre-fix code.

### The claim must match the bytes: a hybrid model gets no disk offer at all (0.2.25)

**A disk offer claims every prefix-cacheable group, so it must be able to restore every
one of them.** This is a correctness fix found on live traffic, not a hardening
exercise: at 00:08 on 2026-10-06 a disk-authority request on a 31,826-token prompt got a
mechanically perfect disk read - `reads_started=9 reads_applied=9 failed={}` per rank,
18 ms, settled on the first settle check - and answered `duct Register Register
Register...` at `temperature=0`, where the same prompt recomputed from scratch returned
the planted code word. Nothing failed on the way. The answer was wrong anyway.

The model is `Qwen3.8-Flash-Next-W4A16-Merlin`: groups 0-3 are `MambaSpec`/GDN under
`MambaManager` in `align` mode (7/6/6/1 layers), group 4 is a QSA ring at `block_size`
8, group 5 is the full-attention group (12 MLA layers), and the RAM tier's own
`prefix_cacheable_group_ids` is `[0,1,2,3,5]`. The restore path had narrowed its
eligible set to *prefix-cacheable AND whole-scheduler-block AND
`has_positionally_stable_blocks`*, which on this model is `{5}` - because
`MambaManager.has_positionally_stable_blocks` is False in align mode
(`vllm/v1/core/single_type_kv_cache_manager.py:1494`), and the fork's own spill path
skips those groups (`vllm/v1/simple_kv_offload/manager.py:874-876`). Then:

* the number that set produces is what `get_num_new_matched_tokens` returns
  (`manager.py:372`), and that return advances `num_computed_tokens` for the **whole
  model** - the engine skipped prefill of the GDN and QSA layers across 28,800 tokens;
* `update_state_after_alloc` builds load entries only for groups that have non-null CPU
  blocks, and its `n_ext_g == 0 -> continue` (`manager.py:445-470`) is a **silent
  no-op** - so groups 0-3 were claimed and never loaded, no group reported a failure,
  nothing came back through `get_block_ids_with_load_errors()`, and nothing recomputed
  the claimed range;
* what a claimed span actually needs from an align-mode GDN group is **one** block, not
  a span of KV: `MambaManager.find_longest_cache_hit`
  (`single_type_kv_cache_manager.py:1533-1610`) answers a hit with
  `[null_block] * (n - 1)` plus one real boundary-state block.

  **The sentence 0.2.25 concluded from that was wrong, and 0.2.27 is the correction.**
  What I wrote next was "that one block has no file name to read back", reasoning from
  the recurrent state being one running value per sequence. The fork disagrees in three
  places: the align hand-off names that block and hands it to the connector
  (`single_type_kv_cache_manager.py:2051-2080`, contract at `kv_cache_manager.py:907-921`),
  the tier stores it under exactly the hash a lookup will ask for (`manager.py:1050-1078`),
  and the store on the box proves it - one shard directory holds
  `kvslot-a36ec46fe316bd0d-g0-…`, `-g1-…`, `-g2-…`, `-g3-…` and `-g5-…` files for the
  SAME key16, five groups at one token boundary. `has_positionally_stable_blocks` is
  about whether offload *scans* may follow a block table (`:62-69`, `:1494-1497`: align
  mode "can null interior states and relocate speculative blocks in place"), not about
  whether a block is *named*. Nulling a block does not un-hash it.

So positional stability is a **necessary** condition (a name to read by) that was being
read as a **sufficient** one (a complete claim). The module already had the right rule
in the right place - `_disk_run`'s docstring: *"A block is restorable only when EVERY
eligible group has a file for it: the fork pairs per group, so one missing group's slot
would be a load entry with no bytes behind it - a whole-request recompute for a
partially-restored span."* The bug was never the rule. It was the **set** the rule was
applied to: narrow the set to `{5}` first and the safeguard is vacuous. The gate is that
same invariant applied to the set the **claim** covers.

**The gate: an incomplete group set claims nothing.** If any prefix-cacheable group
cannot be restored, the disk refuses the offer under its own countable name,
`refused.incomplete_group_set`, with the reason naming the blocking groups and the
sentence beginning with the verdict (because `last_refusal` is clipped at 160
characters). On this model the disk tier offers nothing, forever - that is the correct
behaviour, and it is stated as one rather than hedged. The predicate is derived entirely
from the managers (`prefix_cacheable_group_ids`, `group_block_sizes` against the
scheduler block size, `has_positionally_stable_blocks` per group), so a pure-attention
model has nothing blocked and is offered at **exactly as before**
(`test_a_pure_attention_configuration_still_offers_normally`).

**Superseded in 0.2.27, in its diagnosis only.** The sentences here promising "on this
model the disk tier offers nothing, forever", and the one saying the GDN groups have no
name to read by, were wrong about the fork; the gate now opens on this geometry because
the boundary-state contract is priced per run. What did NOT change is the invariant the
gate exists for - an offer must have bytes for every prefix-cacheable group - and a
group with no implementable contract still refuses under this same name. The next
section is what replaced them.

**Why this is a refusal and not a clamp - do not "optimise" it.** The tempting change is
to offer the intersection, the way the joint CPU lookup clamps. It is the same nothing,
twice paid. The live lookup ledger already shows the RAM path clamping on exactly these
groups: over the last 256 external lookups, 190 returned a hit and **4 were adopted**,
the typical record `local=134400 hit=1000 adopted=0` attributed
`partial_hit_clamp_suspected` - "a state group clamped at a length this lookup cannot
see". A clamped disk claim would be 1-3k tokens (about 0.3 s of prefill saved against
~20 ms of read and the whole claim risk) and would still be a claim whose boundary state
nobody produced. Refusing is the disk path agreeing with a rule the RAM path already
enforces, and it costs production almost literally nothing.

**Read it in the payload**, in the same section that carries `last_offer`:

* `group_serviceability` - the whole arithmetic of the last lookup: the cacheable set,
  the positionally stable set, the restorable set, `blocked_group_ids` with a
  `blocked_why` per group, and `complete`. `complete: false` means this model can never
  be served from disk, and an operator learns that from the payload without knowing what
  a `MambaSpec` is.
* `last_groups_required` - what a claim of `claimed_blocks` costs **each** cacheable
  group: `blocks_needed`, `null_blocks`, the `contract` it came from, and `restored`.
  The 00:08 claim priced this way reads "9 blocks needed in g5, 1 boundary-state block
  needed in each of g0-g3, 0 of those 4 filled", which is the whole incident in one
  object. It is published on the **refusal** door too (`offered: false`), because a
  requirement is a fact about the claim, not about whether it happened. The same rows
  ride along inside `last_offer.groups_required` on an offer that does happen.
* `why_no_restore` now says `THIS MODEL CANNOT BE SERVED FROM DISK AT ALL` with the
  groups that decided it, instead of leaving a reader at "no offer has been made, read
  `refused`".

**A boot that was not turned on touches nothing.** Both boot switches are read before
the gate's expensive work, so a boot with neither up never reaches the predicate at all
- no arithmetic computed, no index query, no journal refresh, no counter moved beyond
the one refusal bucket that names both ops, and the fork's own answer returned
identically (`test_a_boot_with_the_switches_down_touches_no_index_and_no_journal`).
Since 0.2.31 the population this covers is "a boot nobody switched on" rather than "a
request that did not carry the marker", and the guarantee is the same one the
2026-10-05 incident asked for.

Tests: `tests/test_disk_restore_group_gate.py` (the hybrid refusal, the pure-attention
control, the predicate-not-a-group-list proof, the requirement table on both doors) and
`tests/test_disk_restore_ledger_reporting.py` (the four reporting faults below).

### The SSD tier serves the GDN groups: a second block contract (0.2.27)

**The disk tier now restores a hybrid GDN + full-attention model instead of refusing
it.** The gate of 0.2.25 was correct about the invariant and wrong about the engine, and
this is the change the second half of that sentence asked for. Nothing about the
invariant moved: an offer still requires bytes for **every** prefix-cacheable group, and
a group whose block shape this tier cannot price still refuses under
`refused.incomplete_group_set`. What changed is that an align-mode `MambaManager` group
has a shape, so it is restorable - and the shape is what makes the offer cheap.

**Two contracts, derived per group from the manager itself.** `_group_contract()` reads
`has_positionally_stable_blocks` and `mamba_cache_mode` (from the manager, falling back
to its `kv_cache_spec`) and answers one of `positional`, `boundary_state`, `unstable`,
`geometry`, `outside`, `unknown`. No model name, no group id, no new switch: the same
two attributes the fork's own `has_positionally_stable_blocks` override is computed
from (`single_type_kv_cache_manager.py:1494-1497`), which is why the gate opens **by
construction** rather than by an opt-in.

| contract | a claimed run of `n` spans costs | where that comes from |
|---|---|---|
| `positional` | `n` real blocks, one file each | the fork pairs one CPU slot per group block |
| `boundary_state` | `[null]*(n-1)` plus **one** state block at the run's END | `find_longest_cache_hit` (`:1591-1607`), the align hand-off (`:2051-2080`), the store's own `-g0-` files |
| `unstable` | unknowable (`blocks_needed: null`) | not stable, not an align hand-off: no name to read by |
| `unknown` | unknowable, and refused | the predicate is unreadable - a rename, not a shape |

**A run is anchored, never truncated.** `_disk_run` walks the positional groups to the
first block any of them cannot name, then SHORTENS to the longest length whose end every
`boundary_state` group can name, and refuses with the new `no_boundary_anchor` if no
length qualifies. On the live geometry this is the difference between a right answer and
the 00:08 one: if group 5 holds blocks 1-4 and the GDN state files exist only at 1 and 2,
the offer is **two**. The witness says why in one field - `last_offer.positional_run_blocks`
is the length the attention group could have claimed alone, and `run_reanchored: true`
says the offer is shorter than that on purpose. Every later clamp goes through
`_cut_run`, which may only land on a length `_disk_run` proved anchored: a capacity clamp
that took `keys[:3]` would move the state block to a span no group has a state for, which
is the incident in miniature.

**The list the fork consumes is `[null, null, null, real]`.** `_merge_into_pending` builds
an align-mode group's entries from the CPU pool's own `null_block` - the same object the
fork's Mamba hit uses - and puts the one real state block last, because
`update_state_after_alloc` skips entries whose `is_null` is set **while keeping the
index** (`manager.py:460-478`). That index-preservation is the whole mechanism: build the
list the obvious way (real block first) and the recurrent state is paired with the run's
FIRST GPU block, which is a wrong KV state the model then uses to answer the prompt.
`test_the_state_block_pairs_with_the_run_s_last_gpu_block` asserts the pairing, not the
list, and `no_null_block` is the new refusal for a pool that cannot supply the
placeholder - the alternative is a plausible restore that is off by a whole run.

**Copy-at-completion: the queue carries bytes, not slot numbers.** The spill used to read
the CPU slot on `disk_spill`'s writer thread, and the module documented why that was
safe: no file could be read back, so a slot the RAM tier had recycled between completion
and the write cost a wasted write. Now a file CAN be served, and one wrong align-mode
boundary state answers confidently and wrongly at `temperature=0`. So `request_spill`
copies the slot synchronously, on the thread the completion arrived on, and the job
carries the snapshot; `_handle_job` REFUSES a job that carries a slot number without one,
so the old door cannot be reopened by accident. The cost is one memcpy of a rank slot per
spilled slot, bounded by the new `MAX_QUEUED_BYTES` (256 MiB): the queue is admitted on
two currencies now and the byte cap has its own drop name, `queue_bytes_full`, because
"the queue is 256 deep" and "we would have had to hold another slot" are different
diagnoses. `snapshot_failed` is the other new name - a slot that cannot be read is
reported where the slot still exists, not by the thread half a second later. Both the
race and its fix are tested (`test_the_bytes_that_land_are_the_bytes_the_store_completed_with`,
and `test_deferring_the_read_writes_the_next_block_s_bytes` keeps the old behaviour
provable rather than asserted).

**Task 0: what `block_bytes_by_group` was, settled by the files on disk.** The rank
diagnostics published `[80281600, 58982400, 58982400, 1638400, 82944, 132710400]` while
the same rank's spill files are **11,468,800 B** (12,902,400 on PP1) - so the published
per-block figure for group 0 was 7x the entire slot it described. The publisher computed
`len(group.layer_names) * spec.page_size_bytes`, but on the worker side of a merged
config `page_size_bytes` is already the group total: `MambaSpec.merge()` sums it across
the merged layers and `MambaSpec.page_size_bytes` multiplies by its own `num_speces`,
which the merge just folded in (`kv_cache_interface.py:1151-1160`, `:1135-1143`). The
error therefore differed per group (group 5 is an attention spec, merged by MAX) and per
PP rank - which is how two ranks published different "per-group" figures for one model.
Fixes, all at the publisher: the field is `page_size_bytes` **alone** with
`layer_count_by_group` beside it, `arena.bytes_per_block` publishes the rank's MEASURED
slot width as the cross-check, and the `meaning` strings name the mistake they prevent.
Two consequences for anyone reading the old payloads: the figures describe the merged
group's whole block, so no I/O path is affected (nothing reads that field), and
`MAX_BYTES_PER_READ` (64 MiB) was never the limit - the real unit is under 13 MB. The
arithmetic is pinned by tests, not prose: `test_a_spill_file_is_the_measured_slot_width_not_layers_times_page_size`
measures a written file against the figure, and `test_the_publisher_stops_multiplying_the_layer_count_in`
fails if the multiplication comes back.

**Capacity bookkeeping says what a span costs, per contract.** `disk_spill.section().span_cost_bytes`
publishes `positional_bytes_per_span` and `boundary_state_bytes_per_span` in the bytes the
ceiling is configured in, with the sentence that the two numbers being equal does not mean
the two contracts cost the same per span. A reader who counts spilled files concludes a GDN
group is cheap to keep: true per hit, false as a plan, because the `n-1` spans it did not
buy are spans it did not keep.

**Read it in the payload.** `last_offer` gained `positional_groups`, `boundary_groups`,
`run_blocks`, `positional_run_blocks`, `run_reanchored` and `loads` (per group:
`real_blocks`, `null_blocks`, `contract`); `group_serviceability` gained
`group_contracts`, `boundary_state_group_ids` and per-group `contract` rows;
`last_groups_required.blocks_needed` is `null` for a group with no implementable
contract instead of a guess of 1; `why_no_restore` names the contract vocabulary and says
outright that an align-mode GDN group is NOT in the blocked list.

**What is still refused, and why that is correct.** A `boundary_state` group whose run
end no engine ever spilled (a first-block-only chain, a request that starts at the run's
beginning with no state at any length) gets `no_boundary_anchor` and recomputes. A fork
that renames `has_positionally_stable_blocks` gets `unknown`, which is a refusal: a
restore we cannot justify is a restore we do not do. A GDN group that is unstable for a
reason that is not the align hand-off stays blocked, unchanged.

Tests: `tests/test_disk_gdn_boundary_restore.py` (21 tests: the derived contracts, the
offer end-to-end on the live geometry, the null-then-real list, the last-GPU-block
pairing, the anchor cut, both new refusals, the named-block/slot invariant that caught a
double-appended anchor during this change, copy-at-completion and its negative control,
the byte cap, and the byte figure settled against a written file).
`tests/test_disk_restore_group_gate.py` keeps the refusal, re-aimed at the
`unstable`/`unknown` shapes, and adds `test_the_align_hybrid_groups_are_restorable_by_contract`.

### The ledger's own honesty: aged-out loads, DMA-verified blocks, and a witness kind
that could not lie about the present (0.2.25)

The same 00:08 request had a second problem: its payload could not say what happened to
it. Four reporting faults, all fixed, none of them a change to a protective rule:

* **A load that aged out erased its own evidence.** `restorations` stayed 0 while 36
  reads were applied, because the ledger entry was released - by the fork's own *"no
  external tokens but `get_num_new_matched_tokens()` unexpectedly recorded a pending CPU
  hit; releasing the stale pin"* path, which the live engine logs in ~10% of its lines,
  or by the deadline - **before** `finished_recving` arrived. The report loop then found
  `_loads.get(request_id) is None` and `continue`d. Every report-less release now leaves
  a tombstone (`last_aged_out` names the door), and a late report is matched to it:
  `late_finished_reports`, `aged_out_with_applied_reads`, `restored_after_release`.
  `restorations` keeps its exact meaning - *reported while the ledger held it* - so
  nothing is inflated; the truth just stops being dropped. **The deadline did not move:**
  a read completes in 9-30 ms (`read_seconds_max` 0.018) against 45 s, about fifteen
  hundred times the read it guards, so the expiry was never a slow disk and a longer
  deadline would only slow the fuse that stops the re-offer loop of 2026-10-05.
* **Counters that existed and were never published.** `watchdog_fires`, `no_read`,
  `invalid_reports` and the aged-out family are now in the scheduler section with units,
  which is what turns "the release branch must have fired" from an inference into a
  reading - and distinguishes the fork's release door (`watchdog_fires: 0`) from our
  deadline door (`watchdog_fires: 1`). `invalid_reports` is drained on the **worker**
  side, so a scheduler's 0 is correct and its units entry says so.
* **The field that answers "were the GPU blocks actually written".**
  `disk_gpu_blocks_dma_verified` counts GPU blocks this module mapped for a disk span
  (their CPU slots carried our emitted paths - the one moment the scheduler sees our slot
  paired with the fork's destination block) that the fork then reported DMA-complete, in
  **blocks**, not requests. An offer cannot move it; only an adopted load reported
  finished can. A step whose `load_cpu_blocks` and `load_gpu_blocks` disagree maps
  nothing, because the worker's own bind would refuse that step.
* **`witness_kind` was a past sentence read as a present claim.** It was published as
  `offers_not_adopted` while `adopted=1`: emissions dedupe on `(verdict, kind)`, so a
  kind spoken before `_mark_adopted` stayed on the page after it. The verdict (`REFUTED`)
  was right; the kind was true at emission and false at read time. Both readings are
  published now - `witness_kind` (as of the last emission, unchanged) and
  `witness_kind_now` / `witness_verdict_now` (the same arithmetic on the current
  counters), plus `witness_kind_is_stale` when they disagree. `witness_now()` writes
  nothing, so polling cannot rewrite the record the log dedupes on.

Tests: `tests/test_disk_restore_ledger_reporting.py`.

**One release, two authors.** 0.2.25 also carries the ceiling-walk fix above; both
landed in the same uncommitted working tree (HEAD is 0.2.21 and 0.2.22-0.2.24 were never
committed), and neither change touches the other's files.

### The door that showed zeros: a cross-process rollup of the ranks' spill counters (0.2.26)

The reclaim was real and the payload said nothing happened. On the night of 2026-10-06
the store went **395.8 GiB / 34,896 files -> 161.5 GiB / 14,233 files** (234 GiB shed,
`/home` free 61 -> 293 GiB), and `POST /plugins/kvtier/disk {"op":"status"}` answered:

```json
{"spill": {"evictions": 0, "evicted_bytes": 0, "estimated_bytes": 0,
           "ceiling_checks": 0, "estimate_seed_walks": 0, "walk_basis": null,
           "walk_sampled_files": 0, "thread_alive": false}}
```

The truth was already on disk, in `/tmp/kv-tier-diagnostics/rank-PP0_TP0.json`:
`disk.evictions: 15237`, `disk.spill_queued: 79`, `disk.spill_errors: 0`,
`disk.ceiling_bytes: 206158430208`. Nothing was miscounted and nothing was lost.
`disk_spill._state` is **per process**; the spill writer thread and
`disk_spill._enforce_ceiling()` run in the processes that commit writes - the WORKERS,
four of them here (`PP0_TP0`, `PP0_TP1`, `PP1_TP0`, `PP1_TP1`) plus the scheduler - and
the `/disk` door is answered by whichever process the utility call landed on. That
process ran no walk, so every counter it published was honestly zero, and 0.2.25's
whitelist of `ceiling_checks` / `walk_*` - correct as far as it went - was unreadable in
practice because the publishing process was the wrong one. **A witness nobody can read
is not a witness.**

**What 0.2.26 adds: `spill_rollup`, a sibling of `spill`** (`disk_store.rank_rollup()`,
published by `op:status`). It reads the per-rank files `worker_diag` already writes in
`VLLM_E2_TIER_DIAG` and sums what each rank's own process counted. Read-only, no shared
memory, no new IPC channel, no new file name, no new thread, no new env var, no new
write. `spill` keeps its keys and its meanings exactly as they were (pinned by a
byte-identical test), because this adds a view and changes no policy.

```json
{"spill": {"evictions": 0, "thread_alive": false},
 "spill_rollup": {
   "basis": "rank_files_fleet_unknown",
   "totals": {"evictions": 30760, "spilled": 24474, "spill_queued": 143,
              "spill_errors": 2, "bytes_written": 337686041600,
              "queue_depth": 0},
   "totals_reported_by": {"evictions": 4, "spilled": 4, "queue_depth": 4},
   "totals_are_lower_bound": true,
   "ranks_counted": ["PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1"],
   "ranks_expected": null,
   "ranks_expected_source": "no capacity handshake has been recorded in this process",
   "ranks_missing": null,
   "complete": null,
   "per_rank": [{"rank": "PP0_TP0", "age_seconds": 1.4, "evictions": 15237,
                 "spilled": 11841, "queue_depth": 0, "ceiling_bytes": 206158430208,
                 "writer_thread_alive": true, "epoch": "9f2c1a07"}],
   "stale_ranks": [], "excluded_files": [], "excluded_total": 0,
   "ceiling_bytes": {"values": [206158430208], "agree": true, "min": 206158430208,
                     "max": 206158430208, "reported_by": 4},
   "writer_threads": {"alive": 4, "not_alive": 0, "unknown": 0},
   "this_process": {"rank": null, "rank_source": "...not imported here",
                    "counted_in_rollup": null}}}
```

**The totals are a sum of four processes' counters, not one fleet-wide counter**, and
they are labelled as one (`scope`, `kinds`: `evictions` is a `counter-since-boot`,
`queue_depth` is a `gauge`). There is deliberately no attempt at shared memory or a new
IPC channel: reading files the ranks already write is the same channel `disk_restore`'
`worker_proof()` already depends on, and the alternative is far riskier than the gap.

**`basis` says which files the numbers came from**, in the same spirit as
`ceiling_measurement.basis` above: `rank_files` (every rank the boot has, fresh),
`rank_files_partial` (something missing, stale, unreadable or over the read cap),
`rank_files_fleet_unknown` (every file that exists was fresh and counted, but this
process does not know the rank set), `rank_files_unusable`, `no_rank_files`,
`no_diagnostics_dir`, `directory_unreadable`, `rollup_failed`.

**A rank file that is missing or stale makes the rollup say so, never quietly agree.**
`totals_are_lower_bound` with `lower_bound_reason` naming the ranks; `ranks_missing`
lists an expected rank the directory holds **no file at all** for; `stale_ranks`
publishes a stale rank's last-known counters *outside* the totals, with its
`age_seconds`; `excluded_files` lists a file that exists and is unusable. A rank is
named in **at most one** of the three, so a reader adding them up can neither
double-count a gap nor miss one. A silent low total is the dangerous case: it is
indistinguishable from "the ceiling is holding", which is the exact misreading that cost
the 0.2.25 investigation, re-opened one level up. The freshness bar is
`RANK_ROLLUP_FRESH_SECONDS` (900 s, the same heartbeat bar `disk_restore` uses for its
proof) and a `ts` in the future is treated as *not* fresh, because these files cross a
process boundary.

**`ranks_expected` comes from the capacity handshake** (`cpu_capacity`, the one place
rank ids arrive in this process - the same rule `blockdump._expected_ranks()` uses, and
this calls it rather than writing a second one). When this process has recorded no
handshake, `ranks_expected` and `ranks_missing` are `null` and `complete` is `null`:
"I cannot say" is published as itself, never as an empty set, which would read as
"nothing was missing". And `this_process` says whether THIS process's own `spill`
numbers are *inside* the totals - the difference between "my process did nothing and
four others did" and "you are looking at one of the four", and the reason nobody can add
`spill` to `totals` and get a number wrong by exactly one rank.

**A rank file is untrusted input.** Size-capped, type-checked, per-field bounded, and
every rejection is *named*: `malformed`, `too_large`, `bad_version`, `mislabeled` (its
`rank` disagrees with its file name), `no_ts`, `clock_ahead`, `no_disk_section`, plus
per-field `fields_rejected` for a counter that is negative, non-integer or over
`RANK_ROLLUP_COUNTER_MAX`. A bad file costs that field for that rank only, and the door
still answers `ok: true` - never a 500 because a diagnostic file was truncated.

Costs: `op:status` now opens at most `MAX_RANK_ROLLUP_FILES` (32) files of at most 256
KiB each and writes nothing (pinned). There is no cache: the rollup is the truth about
the last file write, and freshness is the point of it. **This needs an engine restart to
take effect** - it is engine-side code reached through the utility call, so the running
boot keeps answering with the 0.2.25 payload until it is restarted into 0.2.26.

Tests: `tests/test_disk_spill_rank_rollup.py`.

### The buried journal: the index DRAINS now, and a partial index cannot masquerade as caught up (0.2.28)

**One defect, one deploy.** The 0.2.22 rule - take each journal's newest window and
forfeit everything older - was unrecoverable by construction and blind in exactly the
place that mattered. Measured on the box: four journals totalling **4,711,140 bytes /
132,099 records** with ~**6,290 net-live identities** (proven by a full front-to-back
replay) and 14,729 live files - and the index reported **`entries: 146`**, so
`no_disk_run` refused every offer for hours. Each first pass was bigger than its share
(131,072 of a 512 KiB budget over four journals), so it read the newest ~131 KiB,
`_forget_rank`ed everything older and landed the cursor at EOF - and a cursor only ever
advances, so a restart re-buried the same 1.4 MB, deterministically. The tail that night
was 93% removal lines (a ceiling reclamation), so the surviving window was 146
identities. `lag_bytes` sat at 0 because the cursor WAS at EOF, and `why_no_restore`
only spoke when `lag_bytes != 0`: the one blindness it was silent about was the one that
was happening. The plugin's own `op:contains` contradicted the store - identities whose
`+` sits early in a journal and was never retracted answered `not_present` while the
file sat committed on disk (`disk_spill` journals a `+` only after `os.replace` commits).

**The rule now: DRAIN, in bounded passes.** `_apply_journal` reads each journal FROM
ITS CURSOR FORWARD, applies whole lines in the file's own order, and when the budget
binds it RETURNS WITH THE CURSOR SHORT OF EOF - the next refresh continues where this
one stopped. Nothing is forfeited, nothing is forgotten, and the buried branch (and its
mid-line alignment probe) is gone: the "don't re-read history" optimization it traded
for - amnesia - is provided honestly by the cursor, which never moves backwards and
never re-reads a converged journal beyond its new tail. **A single refresh after the
first still reads at most `MAX_READ_PER_REFRESH` (512 KiB) and applies at most
`MAX_LINES_PER_REFRESH` (8,192) lines** - the offer-path bound is unchanged and now
auditable through the published `bytes_read`. The FIRST refresh of a process runs the
passes back to back under its own budget - `BOOT_DRAIN_MAX_BYTES` (32 MiB read) and
`BOOT_DRAIN_DEADLINE_SECONDS` (5 s wall) - so a cold boot converges inside the first
offer attempt instead of one share per second afterwards. Tonight's shape costs the
boot: ~4.71 MB read (each byte once, plus at most one line-capped window re-read),
~17 passes, well under a second with warm page cache, on the scheduler thread, inside
the first `get_num_new_matched_tokens` after engine start. Either bound ending the
drain early is normal: the index stays `converged: false`, says so everywhere, and
finishes through ordinary refreshes.

**A partial index can no longer read as a caught-up one.** `disk_index.section()`,
`counters()`, the `restore.index` payload and `op:contains`'s own view now publish
`converged` (every cursor at end-of-file), `boot_drained`, `unread_journals` (per-journal
unread BYTES, named), `bytes_read` and `entry_cap_hit`. `why_no_restore` fires its
blindness sentence on BOTH blindnesses - a drain in flight *and* forfeited history
(`catch_up_jumps`/`bytes_skipped` non-zero with a zero `lag_bytes`, the exact silent
shape) - and the `no_disk_run` refusal carries it too, so the answer reads "spilled but
not yet indexed", never "never spilled", unless the data says it cannot. At the
`MAX_INDEX_ENTRIES` cap the drain drops the OLDEST identities (the restore path wants
the recent spills), counts every drop in `entries_dropped`, flips `entry_cap_hit: true`
and logs ONE warning per boot - a capped index is loud, never a caught-up costume.

**The arm proof reads its own authority.** `disk_restore.armed` in a rank file is
written by `worker_diag._maybe_write` on cadence (50 steps) or a signature change - and
the signature never hashed the armed verdict. So an arm with no spill activity rewrote
nothing, and an idle box wrote nothing at all: the scheduler's `_read_worker_proof`
read a stale `armed: false` and refused with `workers_not_servicing` although every
worker would serve (reproduced live: 0/4 ranks "armed" for 120 s after an arm, then 4/4
seconds after a spill burst). The arm is a **rank-agnostic boot fact owned by the
EngineCore process that runs the scheduler**, so the proof now reads
`disk-spill-arm.json` directly through `disk_spill._read_arm_file` - boot fence and
expiry untouched, a reboot still cannot arm a cold box, and no rank file can arm a
disarmed tier by remembering `armed: true`. Rank files keep the facts they uniquely
know: `installed`, the new `worker_fitted`, `service_reason_kind: "not_configured"`
(a worker without its own disk root still vetoes the whole fleet, exactly as the
2026-10-05 incident demands) and `reads_applied`. A rank file's lagged `armed: false`
under a live arm file no longer vetoes - it is counted as `stale_arm_claims`. As
defence in depth the armed verdict is now folded into the writer's change signature
through a new `signal=` hook on the contribution seam, so an arm transition rewrites
the rank file on the next step; it is not the fix, because an idle box writes no rank
file at any cadence.

Costs: one bounded read of the arm file per `worker_proof()` cache miss (1 s floor) and
one signal evaluation per step inside the writer's existing signature test. Tests:
`tests/test_disk_index_journal_drain.py` (the incident shape, the monotone convergence,
the exact byte bounds, the cap's shout, both blindness sentences, the arm-proof and its
boot-fence refusals), `tests/test_disk_index_catchup.py` (rewritten to pin the drain),
`tests/test_worker_diag.py` (the signal seam), and the retargeted worker-proof tests in
`tests/test_disk_restore.py` / `tests/test_disk_restore_scope.py` (the incident is now
spelled `worker_fitted: false`; the old `armed: false` spelling is pinned as the
lockout it was).

### The eviction clock is a touch clock, and nobody's neighbour is capacity (0.2.29)

**What was asked**: "is the disk cache LRU, and does eviction happen non-blockingly?"
**What the source said**: no, and no - in three separate ways.

**1. The clock only ever saw writes.** Eviction ranks by `st_mtime` (`/home` is
mounted `relatime`, so `st_atime` is not a clock anyone can trust here), and until now
nothing moved `st_mtime` except a spill. A span restored a hundred times was therefore
evicled by a span written an hour later and never read - the tier preferentially threw
away the bytes its own restore path had just proven valuable. A restore now *promotes*:
`disk_read.promote_recency` bumps the file's mtime, so **recency means last touch, and a
touch is a write OR a successful restore read**. Three rules make that safe, and all
three are tested: it is **throttled** to one metadata write per file per
`TOUCH_MIN_INTERVAL_SECONDS` (60 s, on the monotonic clock; the timestamp *written* is
wall clock - writing the monotonic value would date every promoted file to 1970 and
evict the hottest spans first), the throttle map is **bounded** at 4,096 paths and
evolves its own oldest entry, and it is **fail-closed on the verdict** - `file_missing`,
`size_mismatch`, `short_read`, `not_ours` and `read_error` never promote, because
keeping a wrong-sized or unreadable file alive is strictly worse than evicting it. It
runs on the **reader thread** (`_reader_loop` → `_handle_read` → here, after the job has
settled), never on `get_finished` or the scheduler step. Promotion is an mtime and not a
side table on purpose: there is no new durable state and nothing to replay after a
restart. Watch it work in `read.recency_promotions` /
`read.recency_promotions_throttled` / `read.recency_promotion_failures`.

**2. Every rank shed its siblings' live data first.** Four ranks share ONE spill root,
and `lru_eviction_plan` was called without the `peer_digests` parameter it already
accepts. A digest that is not this boot's classifies as `stale_geometry` - "this boot can
never read it, it is pure capacity, **shed it first**" - so each rank's cheapest file to
delete was one of its three peers' live, loadable, hottest spans. Four evictors running
that rule against one shared root is a mutual-destruction loop, and a fill test is exactly
when it bites. The shed order is now
`stale_geometry`/`stale_key_shape` → **this boot's own files, oldest touch first** → a
**live peer rank's** files, and a rank *learns* its siblings' digests from the store walk
it already performs: a foreign digest with at least two committed files written inside
`PEER_DIGEST_FRESH_SECONDS` (120 s) is a rank that is writing right now.

**That protection was conditional, and 0.2.34 made it honest.** "Peers shed last" was
only ever true for a peer the evictor could NAME. The learning is a write-freshness
witness, not an identity: a sibling that is alive but has not written a spill in the last
120 s is not confirmed, and with nothing confirmed every foreign digest classified
`stale_geometry` and went FIRST. What was written at the time as "the fail-safe direction
is deliberate: if no digest qualifies, the pass behaves exactly as 0.2.28 did (foreign
first)" was the bug, not the safety property - on a shared root "foreign first" is four
ranks deleting each other's live KV data. See *An unattributable name never buys
capacity* below for the gate, the short plans it permits, and the witnesses it publishes.

**3. The pass had no deadline and one hand.** Up to 100,000 directory entries read
twice, 20,000 `stat` calls and 4,096 unlinks, serially, on the writer thread that is the
only thing draining a spill queue of 256 jobs / 256 MiB - so a full store dropped spills
while its evictor walked. The pass now has a **time budget** (`1.5 s`,
`VLLM_E2_TIER_DISK_CEILING_BUDGET_SECONDS`) honoured by the walk and the unlink loop
alike. Truncation happens **between whole per-directory batches**, never inside one, so
no plan is ever half-applied; the **first batch always runs**, because a pass that
released nothing only defers the work to a spill that will arrive into the same full
queue; and a submitted unlink is **always awaited and always collected**, so a truncated
pass can never strand a deleted file without its removal-journal record. The pass reports
`ceiling_measurement.pass_seconds`, `.deadline_hit`, `.deadline_hits` and
`.budget_seconds` so "is the ceiling keeping up with this fill rate?" is a reading and
not a hope.

**4. The pool, and why its default is 1.** The unlink batches (already grouped by shard
directory, which is both the filesystem-friendly shape and the partition that guarantees
no path is attempted twice) can fan out to a small named worker pool,
`kvtier-disk-evict`, built once per process. **Measured before shipping, on 20,000 files
across 64 subdirectories, three paired runs**: 1 thread deleted them in **0.402 s** and 4
threads in **0.557 s** - parallel `unlink()` is metadata work, and here it is ~38%
**SLOWER**, not faster. So the mechanism ships and the default stays serial
(`VLLM_E2_TIER_DISK_EVICT_THREADS`, unset = 1). *Caveat worth repeating*: that measurement
ran on the development host's `btrfs`/NVMe, because the serving box was not touched; if a
measurement on the serving array's `/dev/md1` (XFS) ever shows a real win, the knob is
there and the tests already pin that 1 worker and 4 workers shed identical files.

**No direct I/O anywhere, confirmed rather than assumed.** The read path is plain
buffered `open(path, "rb")` + `readinto`, and the write path is buffered
`O_WRONLY|O_CREAT|O_EXCL` + `os.replace`; there is no `O_DIRECT`/`O_SYNC`/`O_DSYNC` in
the spill or read path, and this release adds none. That is a deliberate choice kept
checked: this store reads at **25.4 GB/s from the page cache against 6.65 GB/s cold**, so
the page cache is the tier's largest single asset and `O_DIRECT` would discard it.

Tests: `tests/test_touch_promotion_and_peer_eviction.py` (the operator's request as a
behaviour - the read file survives the merely-written one; the throttle and its fake
clock; every failure verdict refusing to promote; the bounded map; a known peer's live
file not being the first thing shed, and an unconfirmable digest keeping the old
behaviour *and* being counted; a single sighting not being enough; the deadline stopping
the submission, releasing bytes, marking itself, and re-running to convergence without
deleting anything twice; the pool at 1 and 4 shedding the same files and never handing a
path to two workers).

### Four ranks, one ceiling: the fair share, and `disk_fill` (0.2.30)

**What the operator measured**: a four-rank boot sharing one spill root wrote 92 files
per rank and evicted **2,106 / 2,106 / 1,872 / 1,872** files - **7,956 files, 156 GiB
down to 68 GiB** - under a **148 GiB** ceiling whose `EVICT_TARGET_FRACTION` of 0.9 made
the one-pass target 133 GiB. One pass owed ~23 GiB. Four passes shed ~23 GiB each.

**What was wrong**: nothing miscounted. Every rank measured the WHOLE store (156 GiB),
every rank correctly wanted it at 133 GiB, and all four then acted on the SAME
directory. A shared store enforced per-rank against the whole ceiling has an effective
capacity of `ceiling / rank_count`, so the more ranks a boot has, the more of its own
cache it destroys.

**The rule now.** A rank sizes only the bytes it is ACCOUNTABLE for - its own geometry
digest's files, plus the files that belong to no live rank (`stale_geometry` /
`stale_key_shape`: a reader enforces an exact digest and byte-size match, so a mismatched
file can never be applied by anybody and every rank may shed it) - against
`ceiling / n_effective`, where `n_effective = 1 + (live peer digests its own walks
learned)`. A live sibling's bytes are then **neither counted nor sheddable**: its
entries leave the planner's input entirely (`peer_entries_excluded` counts what was
withheld), because the sibling accounts for them under its own share. The convergence is
arithmetic, not tuning: if every rank holds `own + dead <= 0.9 x C/N`, the store holds
`<= 0.9 x C`.

**The fail-safe direction, stated because it is the first question asked.** With no peer
learned, `n_effective` is 1, the share IS the whole ceiling, and the pass sheds byte for
byte what every release before this one shed - including a single-rank boot. The old
store-wide clause is kept INSIDE `_ceiling_trigger` (`estimated_bytes >= ceiling` still
starts a pass), so the new rule can only ever make a pass start SOONER, never later:
capacity safety cannot regress by construction, and `too timid` is not a direction this
exposes. The worst case it permits is a boot whose ranks are mutually invisible - each
holding up to `0.9 x ceiling` of unattributed bytes - which is bounded in practice by the
walk the deciding pass itself performs, and hard-bounded by the unchanged `statvfs`
floor (`MIN_FREE_BYTES`, refusal `no_free_space`).

**The witnesses.** `disk_spill.section().ceiling_share` publishes `accountable_bytes`,
`own_bytes`, `dead_bytes`, `peer_live_bytes`, `unattributed_bytes`, `share_bytes`,
`target_bytes`, `share_divisor`/`n_effective`, `share_source`, and
**`passes_declined`** - the count of passes that walked, found their accountable bytes
already inside their share, and shed nothing. That is the "one rank did the work and
three declined" signature, counted rather than logged, because the pass runs on the
spill writer thread. `passes_nothing_sheddable` is the different fact: over its share and
nothing it is allowed to shed (where the filesystem floor is the control).

**`disk_fill`: one block that answers "how full is the SSD tier, and is that a
problem"**, at the top level of `GET|POST /plugins/kvtier/snapshot` and `GET
/plugins/kvtier/status`. It is built from the rank files the WORKERS write (behind
`rank_rollup`'s one-second cache, keyed on a content signature, so a rank that wrote is
visible on the next read), it makes no engine call, and it emits rank entries only -
capped at `DISK_FILL_RANK_CAP` with `per_rank_truncated` said out loud. The keys a proxy
should write against:

| key | meaning |
|-----|---------|
| `used_bytes`, `ceiling_bytes`, `used_fraction`, `target_fraction`, `target_bytes`, `headroom_bytes`, `over_target`, `over_ceiling` | the fill, as arithmetic - no caller-side maths |
| `basis` | **how the size was measured**: `full_measurement`, `sample_scaled: n of m`, `no_rank_files`, `no_rank_reported_a_measurement` |
| `read_basis` / `ceiling_basis` | how complete the READ was / whose ceiling this is (`ranks_agree`, `ranks_disagree`, `this_process_env_only`) |
| `used_bytes_source`, `used_bytes_spread_bytes` | which rank the number came from, and how far the ranks' views were apart |
| `ranks_expected`, `ranks_reported`, `missing_ranks` | fleet coverage. `ranks_expected`/`missing_ranks` are **null, not `[]`**, when this process cannot know the rank set |
| `per_rank[]` | `rank`, `file_age_seconds`, `files_written`/`spilled`, `bytes_written`, `evictions`, `journal_removals`, `drops`, `queue_depth`, `recency_promotions`, `recency_promotions_throttled`, `ceiling_share_bytes`, `share_divisor`, `declined_because_already_under`, `estimated_bytes`, `walk_basis`, `writer_thread_alive` |
| `totals`, `totals_are_lower_bound`, `lower_bound_reason` | the fleet sum over FRESH ranks only, with the ranks it could not see named |
| `cross_process_verdict` | `shared_share` / `single_rank_share` / `ranks_disagree_on_the_share` / `no_ranks_reported`, with `ranks_acting`, `ranks_declining_names`, `declines_total`, `n_effective` |
| `per_process_note` | the sentence that says the flat `disk_spill` / `disk_store` counters beside it are THIS process's own and read zero on the API process |

`used_bytes` is the FRESHEST rank's estimate and NEVER a sum: four ranks measuring one
directory produced four views of one number, and adding them would be the same mistake
the ceiling just stopped making. A store nobody has measured is `used_bytes: null`, never
`0` - "empty" and "unmeasured" are different facts, and on a polled dashboard only one of
them is a lie.

**The flat sections now label themselves.** The zeros an operator reads at 03:00 are
usually not wrong, just mis-scoped, so `disk_spill.scope` and `disk_store.scope` state in
the payload that every counter in those blocks is THIS process's own - the writer thread,
the ceiling pass and the spill hook all live in the process that steps tokens, so on the
API process they are zero *by construction* - and point at `disk_fill` / `spill_rollup`
for the fleet's numbers.

Tests: `tests/test_ceiling_fair_share_multi_rank.py` (four ranks that all measured the
same pre-shed store and then all acted on it, the incident's interleaving: the store
lands at or under `0.9 x ceiling` AND the bytes freed are within ~1x of the overshoot -
the pre-0.2.30 code frees **4.0x** it, which is what the incident was; the second round
declines and deletes nothing; a lone rank still sheds to 9/10 of the whole ceiling, byte
for byte as before; the share's floor arithmetic; and the trigger firing no later than
the store-wide rule ever did), `tests/test_disk_fill_block.py` (the fill as arithmetic,
the `basis`/`read_basis`/`ceiling_basis` distinction, nulls instead of invented numbers,
missing ranks named, a stale rank aged, the size cap at 24 ranks, and the signature
cache reparsing when a rank file changes), and eleven tests in `tests/test_routes.py`
(the proxy peer on `192.168.2.64` reading the block with the token, a 403 without it, a
token not bypassing the allowlist, loopback needing no token, a spoofed
`X-Forwarded-For` buying nothing, the block travelling on `/snapshot` too, and an inert
plugin 503-ing before any fill read).

### An unattributable name never buys capacity: fail-closed eviction (0.2.34)

**What happened.** Four ranks share ONE spill root and tell files apart only by the
geometry digest in the name. The persistent journal at `/home/user/kv-tier/journal/`
carries **15,848 removal lines of size 11,468,800** - the block size of the OTHER rank
(PP0 is 11,468,800 B, PP1 is 12,902,400 B) - recorded by ranks that were not the writer.
A `file_missing` read then opened a hole in the middle of a restored prefix and
everything behind it, **19,200 tokens**, was recomputed.

**Why the peers were never known.** The peer map is not an identity channel, it is a
write-freshness witness learned from the walk the ceiling pass already performs
(`_observe_peer_digest`, one entry per foreign-digest file the sample stats). A digest is
only believed with `PEER_DIGEST_MIN_SIGHTINGS` (2) committed files written inside
`PEER_DIGEST_FRESH_SECONDS` (120 s), and a digest that ages out is **deleted** from the
map (`disk_spill._live_peer_digests`). So a rank that is alive but idle - which is most
of a serving box's life, and the normal state of three ranks while one fills the store -
is UNCONFIRMABLE, and the next pass runs with `peer_digests=()`. That is not a boot-order
bug: both call sites compute the tuple *after* the walk that would have fed it, so no
reordering helps - the input cannot support the claim. And with an empty peer map
`classify_name` has no peer to name, so a sibling's live restore answers
`stale_geometry`, which was bucket 0: shed before this rank's own files.

**The gate.** Within the ceiling/eviction path, a committed file whose name parses under
a digest that is **not this boot's**, seen while **no peer digest is confirmed**, is
`unattributable` and is never a candidate for unlinking (`disk_layout.attribute_name`, enforced
in `lru_eviction_plan`, again in `_attribute_walk_bytes`, and a third time in
`_validate_unlink` for `for_capacity=True` - nothing planned upstream is trusted at the
point of deletion). It is the unknown case of the rule this module has always had for
malformed names: **reported, counted, never deleted**. Still sheddable, deliberately:
this boot's own files; the pre-group `stale_key_shape` (our lineage, unloadable by any
group-aware reader); an in-flight `.part` (not a committed file, so nobody's restore);
and, when the peer map IS confirmed, any digest outside it - which is exactly what a
confirmed map means, and what keeps a store of old boots' geometry sheddable instead of
turning the ceiling into a comment.

**The consequence is a short plan, not a silent no-op.** If the ceiling cannot be met
without shedding unattributable files, the plan sheds what it is allowed to and comes up
**SHORT**: `short` / `short_by_bytes` / `excluded_unattributable` /
`excluded_unattributable_bytes` on the plan, and `plan_short`, `plan_short_bytes`,
`plan_short_reason` (one line, gate-named, not a stack trace) and the boot total
`passes_short` on `disk_spill.section().peer_ranks`. **Exceeding the ceiling is the
correct failure mode**: spilling back-pressures, the unchanged `statvfs` floor
(`MIN_FREE_BYTES`, refusal `no_free_space`) still guards the filesystem, and an operator
can act. Deleting a sibling's live restore corrupts a serving path in silence.

**The witnesses, published this time.** `peer_digests_live` / `peer_digests_unconfirmed`
/
`peer_digests_dropped` / `peer_files_planned` used to exist only inside
`disk_spill._state`, so the journal evidence could not be resolved into a mechanism. They
now travel in the per-rank file under **`disk.peer_eviction`** (`disk_store
.worker_report`), with the SAME key set on the healthy path, on a rank that never ran a
ceiling pass, on the path where the counters raised, and through the contribution seam's
byte-cap triage. `disk_spill.section().peer_ranks` is the full view, and
`disk_store.rank_rollup()` sums `peer_files_removed`, `peer_files_removed_known_peer`,
`peer_digests_unconfirmed`, `unattributable_entries_excluded` and `ceiling_passes_short`
across ranks, so the dev route answers it in one call.

**The removal is attributable.** A `-` journal line says an identity left the store; it
cannot say whose FILE that was, and a `-` from rank A followed by a `+` for the same
identity leaves the index offering an identity that physically exists only under another
rank's digest. So every unlink whose name carries a digest that is not this rank's
bumps **`peer_files_removed`** (and `peer_files_removed_bytes`), and the one that cannot
lie - the digest belonged to a rank this walk believed LIVE - bumps
**`peer_files_removed_known_peer`**. `last_reason` gets one sentence naming both. The
journal's five-field wire format (`seq op key16 size group`) is deliberately untouched:
`disk_index` parses it, it is durable shared state, and rank-aware identity is its own
bigger change with its own review.

**Two scope boundaries, pinned rather than accidental.** `execute_unlink_plan` with
`for_capacity=False` is the boot sweep's "the operator asked me to reclaim the bytes
nobody can read" path and keeps its old allowed set (it is not wired into production -
the ceiling pass is the only production deleter). And the gate cannot tell a sibling's
digest from THIS rank's own obsolete geometry: `geometry_digest` is a hash of the whole
fingerprint, so a name carries no rank to compare. A rank that restarts under a new
fingerprint therefore cannot shed its own pre-restart files while it cannot confirm
peers; the fix is a small durable "digests this rank has used" history, and it is a
follow-up, not a silent extra write on an engine path.

Tests: `tests/test_unattributable_eviction_fail_closed.py` (the incident as a behaviour on
a filesystem - a peer-blind pass leaves the sibling's files and reports the shortfall;
a confirmed peer's files are never in the destructive set at all; a digest outside a
confirmed map is dead capacity again; the planner's own arithmetic, and its stable key
set; the `.part` exception, pinned explicitly; the executor refusing an unattributable
path a caller handed it, and the sweep path keeping its own doctrine; the four paths of
`disk.peer_eviction` shipping one key set; the dev door summing the cross-rank removals;
and a removal under a known peer's digest being counted, not buried), plus the rewritten
doctrine tests in `tests/test_disk_layout.py` and
`tests/test_touch_promotion_and_peer_eviction.py`.

## Silence is not death: three clocks that were reading the wrong thing (0.2.45)

The operator's rule for this store is one sentence:

> **LRU/quota is the only deleter; silence is not death.**

Every file in the store obeys it - there is no TTL on a cached block, and the ceiling
pass is the only thing that unlinks. Three OTHER clocks in the tier did not obey it, and
all three cost real cache or real service on a quiet box. What they had in common: each
one treated "I have not heard from X recently" as "X is gone". A rank that has not
written recently is not a dead rank. An index entry nobody touched recently is not a
file that left the disk. A worker that has taken no step recently is not a dead worker.

**1. A quiet rank's bytes were the first casualty of a sibling's ceiling pass.** A rank
that has not spilled for a while has no fresh digest in its peers' maps: the peer map
aged its digest out at `PEER_DIGEST_FRESH_SECONDS` (120 s), the walk then found a file
named by a digest nobody could confirm, classified it `ATTRIBUTION_DEAD` ("a config-change
orphan"), and shed it in bucket 0 - ahead of the passing rank's OWN cold bytes. Measured
on the unpatched tree: a quiet rank with a live worker had 3 of 3 of its files unlinked
as dead bytes. A quiet rank is not a gone rank.

The fix separates *liveness* from *geometry*, because the two questions were being
answered by one lookup. Liveness now comes from the rank files (`rank-*.json`, which every
live worker rewrites every 60 s) and not from the peer digest map, so a rank that is alive
but quiet is LIVE and its current files are never `ATTRIBUTION_DEAD`. A foreign-GEOMETRY
file stays sheddable - the `stale_geometry` path is untouched and still runs ahead of this
rank's own cold bytes, because a file no engine on this box can read is genuinely dead
capacity. `peers_known` is derived, fail-closed: an empty peer map claims nothing, so
`attribute_name` refuses to call an identity dead on no evidence, while a rank that can
see its own rank files CAN reclaim real orphans.

*New, and deliberately separate so the two doctrines cannot be confused:*
`stale_geometry_removed` counts files removed as unreadable-geometry orphans (a config
change), `peer_files_removed_live` counts the unforgivable one - a live sibling's CURRENT
bytes - and on the fixed tree the first moves and the second stays exactly zero.
`identity_known` / `identity_available` / `liveness_source` in `disk.peer_eviction` name
which evidence the pass actually had, so "why did it delete that?" is a reading.
Tests: `tests/test_quiet_rank_bytes_are_not_dead_bytes.py`.

**2. The disk index forgot bytes that were still on disk.** `disk_index` capped its
identity map at a hard-coded count and evicted in INSERTION order, so a burst of restores
could push out the identity of the file that was just restored (the expensive one to
relearn) while a never-touched identity from the boot sweep stayed. The cap was also not
configurable: the store grows past it in normal operation.

Eviction is now LRU-by-**touch** (`OrderedDict.move_to_end`), a touch is recorded on the
same event that bumps a file's `st_mtime` (`disk_read.promote_recency`), so the file clock
and the index clock cannot drift apart, and the cap is
`VLLM_E2_TIER_DISK_INDEX_ENTRIES` (default 262,144 - about 90 MB at the measured 342
bytes per entry). A just-restored identity is the LAST to leave.

*The counter that makes the loss honest:* `entries_dropped` is now balanced exactly by
`entries_dropped_lru` (evicted by policy), `entries_dropped_unverified` (never checked
against the directory) and `entries_dropped_unparsable`. Distinguishing them matters
because only one of them means "we forgot a file that is still there":
`entries_dropped_file_present` counts the index forgetting a live file, and
`note_restored_path()` re-admits an identity a restore just proved exists. A drop is never
guessed at - an entry the bounded presence scan did not check is `unverified`, not
"absent". Tests: `tests/test_index_forgets_no_bytes_on_disk.py`.

**3. An idle engine stopped serving from disk.** `disk_restore` may only offer a disk
restore when a WORKER has attested that its restore layer is fitted (separate processes;
offering without the proof means `start_load_kv` DMAs CPU slots nobody filled and the
request is served from STALE KV). The attestation is the rank file and the reader refuses
one older than `RANK_FILE_FRESH_SECONDS` (900 s). Both halves are correct. But the file
was rewritten only by `get_finished` - an ENGINE STEP - so an idle worker's proof aged
out and the tier closed its disk path on a healthy box, reopening only once a burst of
steps arrived. Which is the worst possible moment: the first request after an idle spell
is the one that wanted the disk.

`worker_diag` now keeps its own proof alive from a bounded daemon thread
(`kvtier-worker-diag-heartbeat`, `HEARTBEAT_INTERVAL_SECONDS` 60 s = fifteen beats inside
the 900 s window, write floor 5 s), started at `install()` and NOT at the first step - the
0.2.44 cold-start lesson, because an idle worker is precisely a worker whose step-driven
starter never runs. The beat goes through the same `write_snapshot` a step uses, so a
reader can never tell the two files apart. The handler is held as a WEAK reference (a
diagnostic writer must not pin the CPU KV arena), and a beat with no handler still proves
liveness, because the FACT the file carries is "this process is alive".

What it does NOT do is weaken the gate: the window stays 900 s, so a worker that has
genuinely gone stops being believed on time. A PROOF IS NOT DATA - nothing here keeps a
cached byte alive, it keeps a live worker legible.

*New counters:* `heartbeats`, `heartbeat_threads`, `heartbeat_respawns` (a thread that
died and was re-armed by the next step), `heartbeat_floor_skips`, `heartbeat_errors`,
plus `heartbeat_alive` / `last_heartbeat_age_seconds` / `heartbeat_beats_in_window` in the
section, which answer "is this worker's proof being kept alive, and how tight is it?"
without arithmetic. Tests: `tests/test_idle_engine_keeps_its_disk_proof_alive.py`.

## Does the GPU still hold what the arena spilled? (`disk_shadow.py`, 0.2.36)

The spill path captures a slot by reading the **arena** - the worker's
`cpu_kv_caches`, the copy vLLM itself makes every step - and that is the only copy of
a block's bytes the tier ever writes. If that read captured the wrong bytes, or
captured them in an order the restore side does not reassemble, the tier would serve
silently wrong KV, and **no counter that exists today would move**: a byte-level
mismatch is invisible to a hit rate. The shadow verifier is the witness for exactly
that.

It is a measurement, not a gate. Default **OFF**, read-only, fail-open: it never
changes a spill or a restore decision, never refuses a write, never opens a file
handle for writing, and touches the GPU only to READ one block of it. At store **bind**
it retains the GPU block ids the engine named next to the CPU slots (ints only - never
a tensor, never the metadata object). At store **completion**, for a sampled subset, it
narrows that one block on the device, copies it to the host on the current stream, and
compares the bytes against the blob the spill path captured for the same block.

```bash
# the worker's switch is the environment, because the worker is another process:
#   VLLM_E2_TIER_DISK_SHADOW=1        (read once at first use; restart to change it)
# this op reads the counters, and arms the process that ANSWERS (it writes no file):
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"shadow"}' $B/disk
```

**A mismatch is a QUESTION, and the categories say which one.** A completed store's GPU
block is the engine's to reuse the moment the store finished, so the shadow re-reads the
block once and asks whether the GPU's own bytes moved between the two reads:

| counter | what it means | what you do |
|---|---|---|
| `equal` | the GPU and the arena agree | nothing; this is the number growing |
| `mismatch_gpu_changed_on_recheck` | the GPU rewrote the block between the reads | nothing: expected block reuse |
| `mismatch_stable` | the GPU held still and still differs | **this is the one** - a layout/ordering question |
| `mismatch_unrechecked` | the re-read could not run | look at `reasons` |
| `mismatch_rebound_later` | a block that mismatched was named again by a later bind | evidence of reuse, not proof of timing |
| `width_mismatch` | the GPU's region widths disagree with the arena's | a layout answer, never counted as staleness |
| `no_gpu_block` / `no_gpu_caches` / `no_blob` | no GPU id, no GPU tensors, or the spill refused | context for the sample rate, not a fault |

The last three are why there is no single "corruption" counter: "the two paths
disagreed", "there was nothing to compare" and "the engine reused the block" are three
different facts, and merging them into one integer would make the number useless on a
busy boot.

**What it costs, and where the bound is.** The copy runs on the engine's step thread, in
the same tick `MAX_SLOTS_PER_TICK` bounds. `MAX_BLOCKS_PER_STEP` (2) is the hard cap per
bind window - exceeding it increments `throttled_step` and does no work - with
`MIN_INTERVAL_SECONDS` (0.5, counted as `throttled_interval`) and 1-store-in-32 sampling
on top, so the cost is a constant rather than a function of the store rate. One block
per region is `narrow`ed BEFORE the copy, so it never moves a whole KV cache across the
link.

**Two invariants it exists to test.** The bytes of a slot are the sum over the region
tensors and the order of that sum is `sorted(name)`, never dict-insertion order (0.2.8
shipped that bug) - so the shadow does not re-implement the ordering: it hands the
copied one-block regions to `disk_spill.read_slot_bytes`, THE SPILL PATH'S OWN READER,
and a mismatch therefore cannot be "two readers summed differently". And because
everything downstream is `ctypes.string_at(data_ptr())`, a device pointer read as a host
address is a SIGSEGV rather than an exception, so the copy's result is checked against
its OWN `device` field before a byte is read (`_host_landed`).

The verdict goes into the worker's rank file through `worker_diag` as the `disk_shadow`
section - a guest section, never a second writer, never a second file - and its key set
is identical whether or not this process can read a GPU at all, so a reader diffing two
rank files can tell "moved" from "gone". `tests/test_disk_shadow.py` is the executable
spec; the device->host copy is the single expression it replaces, because that
environment has no CUDA device.

**Reading a mismatch in the file.** `last_mismatch` is a SHAPE, never bytes:
`category`, `slot`, `gpu_block`, `bytes`, `first_difference` - the byte offset where the
two blobs first differ, or `null` when they agree for the whole `scanned_bytes` window
(`MAX_DIFF_SCAN_BYTES`, 256 KB, is the scan cap, so "they differ past 256 KB" is an
honest answer and a 11.4 MB Python scan on the step thread is not) - and the two recheck
verdicts. It is never a trim target and never gains a key. `trimmed` names what the
section shed to fit its own `MAX_SECTION_BYTES` (3072, tighter than the seam's 4096, so
the writer never has to refuse it) and `section_trimmed` counts how many times that
changed shape; the counters are never shed, so a thin section is still a complete set of
numbers. If a section is over budget after every shed, the writer refuses it and the
worker log says `section STILL over its byte budget` - a missing `disk_shadow` key with
that line beside it is a size problem, not a missing facility.

Its `equal` verdict is also the *admission rule* for the next step of the route - writing
those GPU bytes to a second tree and diffing the two files - see *Writing the block twice:
the GPU-direct shadow* below.

## Writing the block twice: the GPU-direct shadow (`disk_gpu_shadow.py`, 0.2.37)

Entry 27 of the plan established that the host-RAM arena cannot simply be deleted: it is
the spill trigger, the byte source, the restore landing pad and the only GPU<->host
copier. Entry 28 decided the route - build a GPU-direct write and a GPU-direct read, and
prove each one as a shadow before letting either be authoritative. The verifier above
(`disk_shadow.py`) is step 1 and it said `equal` in production. **This module is step 2:**
the GPU-direct WRITE path, still not authoritative.

For a block the verifier has just called `equal`, this module writes those same GPU bytes
a second time, to a **separate directory**, under **exactly the name the live store uses**
for that block - and then compares the two files. The live store's file remains the only
thing any read path loads. Nothing in a spill, a restore, an offer or an eviction decision
reads a byte written here. Default OFF, fail-open, and its output is disposable evidence
with a retention cap of its own.

```bash
# the worker's switch is the environment, because the worker is the process that writes:
#   VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES=1     (read once at first use; restart to change)
#   VLLM_E2_TIER_DISK_GPU_SHADOW_ROOT=/path   (optional; default = <store>/../gpu-shadow)
# this op reads the counters, and arms the process that ANSWERS (which is usually NOT
# the worker - the answer says so in `scope`):
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"gpu-shadow"}' $B/disk
```

**Files land beside the store, never inside it.** With the box's store at
`/home/user/kv-tier/disk`, the shadow tree is `/home/user/kv-tier/gpu-shadow/<shard>/
kvslot-<key16>-g<group>-<geodigest12>.bin` - the same relative name, a different root. A
shadow root that IS the live store, or is nested in it in either direction, is REFUSED
(`refused_root_conflict`, one log line, nothing written): merging would put bytes in the
store's shard tree that a restore could load, which is the one outcome this module exists
to make impossible. `/a/store` and `/a/store2` are compared on path components and are
not confusable.

**The one command that reads the result.** Run it on the box, after the feature has been
on through some real traffic:

```bash
diff -rq /home/user/kv-tier/disk /home/user/kv-tier/gpu-shadow | grep -v '^Only in '
```

**When everything agrees it prints nothing.** `Only in ...` lines are expected and mean
nothing is wrong: the shadow tree holds only the sampled subset (one store in 32, capped
per bind window), so the live store always has files it does not. What must never appear
is a line of the form
`Files /home/user/kv-tier/disk/3f/kvslot-....bin and /home/user/kv-tier/gpu-shadow/3f/kvslot-....bin differ`
- one of those is a fault in the GPU-direct write path (or a live file that changed after
the spill), and the module raises the same alarm in the worker log when it sees one. The
counters cross-check the same claim without reading files:

| counter | what it means | what you do |
|---|---|---|
| `written`, `bytes_written` | shadow files committed | nothing; this is the number growing |
| `pairs_checked` / `pairs_identical` | files compared against the live store's file of the same name, and byte-identical | nothing; `identical == checked` is the gate |
| `pairs_differing` | a same-named pair disagrees | **this is the one** - look at `last_difference.first_difference` and `differing_names` |
| `pairs_size_disagree` | the same-named pair has different lengths | a truncated or half-written shadow file |
| `pairs_unpaired_expired` | the live partner never appeared | the spill refused, was evicted, or the name is not the same file |
| `pairs_shadow_missing` | the SHADOW file went away before its pair was compared (retention shed it, or the directory was cleared) | raise `MAX_SHADOW_FILES`/`MAX_SHADOW_BYTES` if you want every pair decided; it is never counted against the store |
| `refused_not_equal` | the verifier did not vouch for the block, so nothing was written | nothing; this is the admission rule working |
| `refused_not_checked` | there was nothing to compare (a device fault, a refused spill) | context, not a fault |
| `refused_root_conflict` | the shadow root is the live store or nested in it | fix the environment; nothing is written meanwhile |
| `refused_write_failed` / `refused_*` | the directory refused the write, the queue is full, the name disagreed | look at `reasons` |
| `retention_removed`, `walk_truncated` | the cap (`MAX_SHADOW_FILES`/`MAX_SHADOW_BYTES`) shed evidence, or the walk hit its bound | normal; the files are evidence, not data |

**What the diff proves, said exactly.** Only blocks the verifier called `equal` are
written, and the bytes written are the bytes that vouching read - so the two files agree
by construction *at capture time*, and the diff is not re-answering "do the GPU and the
arena hold the same bytes?" (the verifier answers that per block, live, in its own
counters). What the diff does prove is everything this module ADDS: that
`disk_layout.build_slot_path` under a different root produces the same relative name the
store wrote (the identity a restore looks up - the module checks the two naming sites
agree and writes nothing if they ever differ), that what landed is a whole slot and not
half of one, that the `.part`/`os.replace` discipline held under real traffic, and that
across a store turnover the GPU-direct tree and the arena tree are the same files.

**It inherits the ordering rule instead of re-implementing it.** The bytes come from
`disk_shadow._read_gpu_slot` - the ONE device->host seam in the package - which sums the
copied one-block regions through `disk_spill.read_slot_bytes`, the spill path's own
reader, in `sorted(name)` order. This module never walks `gpu_kv_caches`, never orders
anything and never forms an address, so it cannot disagree with the store about layout,
and there is no second device-copy mechanism to get wrong. The same reader's width check
is also what refuses the second trap in entry 28: the KV groups OVERLAY one backing
allocation (the slot is the MAX over groups, not the sum), so a design that walked group
specs would try to write a 43 MB file into an 11.5 MB name.

**It never blocks the engine.** The only function the step thread calls is `consider()`
- two dict lookups, a width check, a cap check, one `put_nowait`, and no `stat` or
`makedirs` anywhere in it. Naming, writing, comparing and pruning happen on this module's
own daemon writer thread, bounded at `MAX_QUEUE_JOBS`/`MAX_QUEUED_BYTES` for the queue,
`MAX_PAIRS_PER_PASS` pairs and one slot width of scanning per comparison pass, and
`MAX_SHADOW_FILES`/`MAX_SHADOW_BYTES` for the directory. Retention deletes through
`disk_layout.execute_unlink_plan`, which re-validates every path at the instant of
deletion - so "this never deletes anything outside its own root" is that proven
function's property, tested by planting a file in the live store with the same name as a
shadow file and watching which one disappears.

`tests/test_disk_gpu_shadow.py` (25 tests) is the executable spec: the default creates no
directory, an `equal` block produces a file whose bytes ARE the arena's, a block the
verifier did not vouch for produces none, the LIVE STORE is byte-for-byte what it would
have been without the feature, a device fault changes nothing on disk, a shadow root that
is the store is refused, a queue that is full refuses on the step thread instead of
growing or waiting, retention stays inside its own directory, a pair whose shadow file
retention removed is counted as ours going away and never as the store's absence, the
shadow file is the `sorted(name)` concatenation and not the insertion-order one, and the
published key set does not change shape.

## Do the RESTORED bytes match the file? (`disk_restore_verify.py`, 0.2.39-0.2.42)

The two modules above answer a WRITE question: after the arena spilled a block, does the
GPU still hold those bytes, and can those bytes be written straight to NVMe? This module
answers the READ question, which is the one nobody had answered: **after a disk restore,
does the GPU block hold what the spill file holds?** Default OFF, worker-side, read-only,
sampled, and it changes no restore decision.

Why the question is not already answered by the ledger. A disk restore is a chain of
**index-aligned lists**: the scheduler's `token_load_job_list` carries a
`(cpu_block_id, gpu_block_id)` per entry, `_worker_bind` walks the lists beside each other,
`disk_read` fills a CPU slot, the plugin memmoves that slot into the bounce buffer the
entry names, and the fork's `start_load_kv` DMAs the slot into `gpu_block_id`. Every step
is index arithmetic, and a **one-slot shift moves the aligned lists together** - the name
stays well-formed, the ledger row is complete, the counters agree, and the engine serves a
window that belongs to a different token boundary. Nothing in the stack compares the bytes
at the destination, which is exactly the check `disk_shadow` provides for the write path.

**The five assertions, in the order that makes each failure nameable on its own.**

1. **The file is window-wide.** `os.path.getsize == bytes_per_block` for this rank. A file
   that is 12 B (or double-wide) is a geometry or a rank-crossing answer, so it is refused
   as `file_not_window_wide` with BOTH widths in the reason and never compared - comparing
   under a width that cannot be right manufactures a finding.
2. **The group is narrower than the file.** A group whose own `page_size_bytes` exceeds the
   slot width would read past EOF and call the zeros a mismatch: `group_wider_than_file`,
   refused before the file is opened.
3. **The leading `W_g` bytes agree - and only those.** `W_g` is the GROUP's width, and the
   rest of the window is the other groups' bytes, stale by construction. So the file is
   read with `read(W_g)` (the tail is never read from disk at all) and the device bytes are
   sliced `[:W_g]` before they reach the compare. The refusal is structural, not a
   convention: a verifier that compared the whole window would report a mismatch on every
   narrow group forever, and the one real finding would drown in it. `tail_bytes_excluded`
   is published beside every verdict so the reader can see how much nobody claimed.
4. **The name comes from the blocks the restore was asked for** - the binding oracle. It
   re-derives the file's `key16` from `block_hashes` through the engine's OWN helpers
   (`make_block_hash_with_group_id` + `resolve_block_hashes`) and compares that against the
   name in the path. Name-vs-name cannot catch a one-slot shift, because a shift moves the
   index-aligned lists together and both strings stay well-formed; re-derivation from the
   blocks at the index actually used can. **On this fork a worker has no requests** - the
   fork's own metadata module states that "the worker never knows about request identities"
   - so in production every entry counts `oracle_unavailable` and the reason names the file
   that says so. That is the honest answer, published rather than a silent skip, and
   `set_oracle(provider)` is the seam that changes it the moment a source exists.
5. **The truncation shape.** A batch that restored the positional groups and named NOTHING
   for an align-mode group is the Gate B shape: `n_take_g` (`manager.py:445-448`) truncates
   each group's hit list, and an align group's list is `[null]*(n-1) + [one real block]`,
   so a truncation before the run's end removes that group's ONLY real entry while every
   positional group keeps its bytes. The verifier cannot see a missing entry, so it counts
   its ABSENCE (`boundary_entry_missing`, per group). The align groups come from the worker's
   own specs (`spec.mamba_cache_mode == "align"`, a field only `MambaSpec` carries); an entry
   that was present and whose read FAILED is excluded - that is `disk_restore`'s finding,
   already counted where it happened.

**Where the GPU bytes come from: `disk_shadow`'s seam and no other.**
`disk_shadow._read_gpu_slot` -> `_copy_slot_to_host` -> the host-landing guard ->
`disk_spill.read_slot_bytes`. There is no second device-copy mechanism in this package and
no `ctypes` read of a device address in this module (the module has no `import ctypes` at
all), because the hazard there is a segmentation fault rather than an exception, and because
one seam means the two witnesses cannot disagree about what a failed device read means. The
comparison also waits for the fork's own completion watermark (`worker._load_hwm >= event`):
the apply point runs BEFORE the DMA is issued, so comparing at the apply point would read
bytes that have not arrived; a worker with no watermark gets `no_load_hwm` and is never
compared.

**What it costs, and on which thread.** The engine's step thread pays `observe_apply()`,
which is bookkeeping only - a counter, a sample decision, and a dict of ints plus one path
string. No file I/O and no device read happen on a step thread; both happen on this module's
single lazy daemon thread, bounded to `MAX_CHECKS_PER_PASS` comparisons and
`MAX_RECHECKS_PER_PASS` re-reads per pass with a `MIN_INTERVAL_SECONDS` floor that COUNTS
(`throttled_interval`) instead of sleeping, because a floor that slept would put a wait on
whoever called it. Retention is `MAX_RETAINED_PER_STEP` per apply point inside
`MAX_RETAINED_JOBS` overall, dropping the OLDEST first: an entry that has waited longest has
had longest for its GPU block to be handed out again, so it is the least likely to be
evidence. Entry `i` is sampled every `SAMPLE_EVERY_N_RESTORES` (32nd), counted both ways
(`applies_seen` / `sampled_out`) so the sample rate is a rate over restores that actually
moved bytes.

**A disagreement is classified, never proclaimed.** A GPU block is handed out again at any
time, so a mismatch is re-read before it is reported: a re-read that differs is
`mismatch_gpu_changed_on_recheck` (the engine moved - an expected race), a re-read that held
still is `mismatch_stable` (a layout candidate, the ONLY one an operator acts on), and a
re-read that the device refused is `mismatch_unrechecked`. Collapsing those three would make
the number an operator acts on the number he cannot act on.

**How to turn it on.** It is a WORKER feature, so the environment is the switch that matters:

```bash
#   VLLM_E2_TIER_DISK_RESTORE_VERIFY=1        (read once at first use; restart to change)
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"restore-verify"}' $B/disk
```

The POST latches only the process that answers - which for this module is the EngineCore,
NOT the worker that applies a restore - so the answer carries `scope` and says so. It is
POST-only because it changes what the process does (reads device memory), and `op` is a
query parameter on that route, so a `<img>` preload must not be able to arm it. A non-boolean
`enabled` is refused (`JSON boolean` expected) and changes nothing.

**Reading it.** `rank-<rank>.json` gains a fifth contributed section, `disk_restore_verify`
(the seam's allow-list held five sections at the time; 0.2.43 has since taken the sixth slot
for `disk_gpu_write`, and the rank file still carries every field the
proxy host's reader requires inside its own byte ceiling). It publishes the whole counter set
with a stable key set, the geometry it read off its own worker (`window_bytes`,
`group_bytes`, the align group ids, and the note when a config has none), the pending queue,
the last mismatch with its offset and its excluded tail, and the oracle's state - and, since
0.2.40, the SCHEDULER half beside it: the `scheduler_*` counters (since 0.2.41 including the
alloc-hook and expectation-lifecycle counters and `scheduler_expectations_now`),
`scheduler_jobs_now`, `scheduler_mismatch_samples`, `scheduler_oracle`,
`scheduler_oracle_hook`, `roles` and `oracle_rule`. The rank FILE is
written by workers, so the scheduler's numbers reach a reader through `/snapshot` and
`/disk {"op":"restore-verify"}` in the EngineCore; the section is the same object either way.
The verb, `status()` and the section are built by ONE builder, so a number cannot differ by
door.

**Two limits, stated where they cannot be missed.** The truncation shape is counted at BATCH
granularity, so a batch containing two requests that both lost the same align entry counts
once - under-counting, never inventing. And the worker's binding oracle is unreachable in a
worker process on this fork, so `oracle_checked` stays 0 in production while
`oracle_unavailable` counts every verified entry: the recomputation is real code (tested
against the same key helpers the offer path is tested against) waiting for a provider, not a
stub.

## THE OTHER HALF OF THE SAME MODULE: did we offer the RIGHT file? (0.2.40; alloc hook since 0.2.41; the hash view accepted since 0.2.42)

Assertion 4 above - the binding oracle - is unreachable **in a worker**, and for 0.2.39
that was the honest end of the story. It is not the end of the check. The pairing that a
worker cannot see was made one process earlier, by the **scheduler** - and 0.2.40's first
attempt at witnessing it failed in production in a way worth stating exactly, because the
failure is the design.

**What 0.2.40 believed, and what the box said.** 0.2.40 asked at the emission point
(`_build_load_paths`): "this file goes to that GPU block - re-derive the name for that
block's index". Its input was `scheduler_output.kv_connector_block_state`, the fork's
connector-only view of block tables, attached at `sched/scheduler.py:1475` and cleared at
`:1497`. The first armed boot answered every sampled entry with
`scheduler_oracle_unavailable: block_state_does_not_cover_request` and `scheduler_checked`
stayed **0**: zero comparisons, exactly as if the oracle had not been fitted. The reason is
a step-boundary that only a live engine shows. `kv_connector_block_state` carries the block
tables of the requests the CURRENT step scheduled
(`scheduler.py:1412-1425`); `build_connector_meta` emits load states for EVERY request
still sitting in `manager._reqs_to_load` - and a load entry is emitted, and re-emitted, in
steps **later** than the one that allocated its GPU blocks (the request waits for its
`load_event`; the entry rides each step's metadata until the load completes). So at the one
moment the emitter could name the file, the table that would say which GPU block the
pairing used had already forgotten the request. The offer point can name but cannot see;
the alloc point can see but has named nothing. **No single connector call holds both**, and
a witness that pretends otherwise gets the verdict 0.2.40 got: nothing.

**The hook where the pairing is decided: `update_state_after_alloc`.** The fork's
`SimpleCPUOffloadScheduler.update_state_after_alloc(request, blocks, num_external_tokens)`
(`simple_kv_offload/manager.py:381`) IS the moment the pairing is made: it reads the GPU
block tables from its OWN argument (`blocks.get_block_ids()`), pops the CPU-hit record
`get_num_new_matched_tokens` left in `_pending_cpu_hits` (`:401`), gates on
`prefix_cacheable_group_ids` with a `None`-means-all rule, truncates each group to
`cdiv(num_external_tokens, group_block_sizes[g])`, skips `is_null_block` entries WITHOUT
advancing the index, and pairs `group_gpu_ids[num_computed_tokens // g_block_size + i]`
with the i-th CPU block (`:446-470`). Those pairs are handed to the worker as
`load_cpu_blocks` and never re-examined - a wrong pairing here is a file read into the
wrong GPU block, and no later check can see it. So 0.2.41's witness runs HERE, as a
read-only observer fitted by `disk_restore` (`METHOD_ALLOC`), replaying the fork's pairing
loop EXACTLY - the same gate, the same truncation, the same null-skip, the same index
arithmetic - and for every pair the fork will form, it captures the expectation that
pair's file must satisfy: the `(gpu_block, cpu_block)` key, the group, the block index, and
a clipped copy of the engine's own `block_hashes`. It never raises, never mutates, and
pops NOTHING - it peeks `_pending_cpu_hits`, because the fork pops it two lines later and
the next reader is the fork itself.

**Why the wrapper must observe BEFORE the original, which is not how the other wrappers
work:** the fork pops `_pending_cpu_hits[req_id]` at `manager.py:401` and registers the
request in `_reqs_to_load` at `:483-484` inside the same call, so after it returns the
table is gone and only the request survives; before it runs, the table is there and only
the request survives - except that at entry the request is an ARGUMENT. The observer reads
the arguments and the pre-pop pending state, which is the only complete view the hook
offers. (The offer path still passes `scheduler_output` through to the witness unchanged,
but since 0.2.41 nothing reads it there.)

**Then the comparison happens where the name is made** - `_build_load_paths`, unchanged
position, new input. For each entry the fork is about to emit `(cpu_block -> gpu_block)`,
the witness asks the captured map for that exact pair - identity, not position, because
"which GPU block did the fork pair THIS cpu block with" must survive re-emit steps. A hit
carries the group, the index and the hashes the fork itself used, and the verdict is the
same re-derivation 0.2.40 specified: resolve the engine's own
`resolve_block_hashes(captured block_hashes, manager.hash_block_size,
group_block_sizes[group])`, take `view[index]`, pack it with the group through the engine's
own `make_block_hash_with_group_id`, and compare 16-hex digests with the name in the
offered file. **Nothing in this module re-hashes or re-packs**: both calls go through
`disk_restore._key_helpers()`, the same resolver the OFFER path uses, so the code that
wrote the name and the code that re-derives it cannot disagree about which bytes are the
group.

**THE SHAPE OF THAT VIEW, AND WHY 0.2.41 RAN WITH `scheduler_checked = 0` (0.2.42).**
`resolve_block_hashes` does not answer a list. Its declared answer is the union
`BlockHashList = list[BlockHash] | BlockHashListWithBlockSize`
(`vllm/v1/core/kv_cache_utils.py:2850`), and the second member is a **lazy view that is
not a `Sequence`** - it neither inherits from nor registers
`collections.abc.Sequence` (`kv_cache_utils.py:2783-2848`). 0.2.40/0.2.41 checked
`isinstance(view, (list, tuple))`, refused the view, and the live boot reported exactly
what a decorative witness reports: `scheduler_expectations_consumed = 2`,
`scheduler_oracle_refused = 2` with reason
`resolve_shape: ... BlockHashListWithBlockSize, not a sequence`, and `scheduler_checked =
0`. Refusing a shape this module cannot index was the right behaviour; the list of
understood shapes was too short. **0.2.42 accepts the union by capability** - `len(view)`
must answer and integer `view[i]` must exist, both looked up on the `type`, the only place
Python looks them up - and still REFUSES, counted and reasoned, anything without that
protocol, because comparing a shape you cannot index is how a file gets read under the
wrong name.

**AND THE VIEW IS INDEXED, NEVER THE LIST UNDER IT - on this box they differ by a factor
of 80.** The live numbers are `hash_block_size = 40` against
`group_block_sizes = [3200, 3200, 3200, 3200, 8, 3200]`, so a serving group's view has
`scale_factor = 3200 / 40 = 80`: `view[i]` is `raw[(i + 1) * 80 - 1]` and `len(view)` is
`len(raw) // 80`. Indexing the raw list at `i` does not give a near miss - it gives the
digest of tokens `[0, 40)` compared against a file named for tokens `[0, 3200)`, a
DIFFERENT TOKEN RANGE, which would read as a storm of false mismatches or as an accidental
agreement that looks like a passed safety check. So `view[index]` is the only read in this
module and `view.block_hashes` is never touched, and the tests pin it from both sides: the
correct pairing must verify (a raw-indexing oracle calls it a mismatch), and a name built
from `raw[index]` must be REFUSED as a mismatch whose `expected_key16` is the view's; a
second test pins the collapsed length, because block 9 exists in a 640-entry list and does
not exist in the 8-entry view, and comparing past the view is comparing a digest for a
token range nobody named.

**A GROUP THE ENGINE CANNOT VIEW IS COUNTED, NEVER COMPARED.** Group 4's blocks are 8
tokens against a 40-token hash, and the fork's view-builder requires the target to be a
multiple of the hash granularity - it asserts it twice (`kv_cache_utils.py:2826`, `:2887`) -
so asking for group 4 raises `AssertionError`. Group 4 is the parked QSA ring: not
prefix-cacheable, never spilled, so the oracle has no business being asked about it; but
the classification walk enumerates EVERY other group of a mismatching record, so it walks
into that assert on this box on every finding. `_scheduler_view` contains it, names it
(`scheduler_hash_view_unbuildable`, with a reason that says which group and which
arithmetic), and never lets it reach the caller as an exception. A skipped group with no
counter is precisely how a witness goes decorative again, so the counter lives at the one
place that BUILDS a view - every call site, including the walk, is covered by
construction.

**Both negative controls fail loudly, driven through the real hook signature.** A
**one-slot shift** - the failure every list-vs-list check in this plugin is blind to,
because a shift moves the index-aligned lists together and every count still agrees - is
`scheduler_mismatch_neighbour_index`, and the sample carries the SIGNED offset (which way
it moved is the diagnosis). Its mirror is classified with the opposite sign. A **group
swap** is `scheduler_mismatch_other_group`, in both of its shapes: a name whose own 4-byte
group suffix contradicts the group the entry is paired with (a contradiction visible with
NO engine input, so it outranks the digest comparison), and a name with the RIGHT suffix
built from the WRONG GROUP'S HASH VIEW - what a per-group view resolved with another
group's `block_size` produces, which no suffix check can see. The controls are not
simulated: every negative test drives `observe_alloc` at its real signature
`(manager, request, blocks, num_external_tokens)` and a `build_load_paths` entry shaped the
way the fork shapes it, because a control that skips the arguments that decide the
pairing proves nothing. A name that matches none of these is `scheduler_mismatch_unknown` -
reported under that name rather than guessed at.

**The lifecycle of an expectation, because the memory bound lives there.** Captured at the
alloc hook into one insertion-ordered dict, capped at `MAX_SCHED_EXPECTATIONS` entries and
`MAX_EXPECTATIONS_PER_ALLOC` pairs per alloc (an alloc that overflows the per-alloc cap
counts `scheduler_expectations_capped`; evicting an expectation that was never consumed
counts `scheduler_expectations_dropped`). A consumed expectation is MARKED, not removed:
the entry that rode three emit steps must be checkable on all three, and only an
UNCONSUMED expectation can ever be evicted - so the bounded quantity is memory, never
correctness, and the bound holds at the alloc hook, which is the only place a burst of
traffic can pile expectations up before anyone compares them. `scheduler_expectations_now`
is the live gauge. The regression this buys is the production failure itself, and there is
a test for exactly it: alloc in step 1, emit in a LATER step - the check still compares,
because what it needs was captured at the alloc, and the emission step's
`SchedulerOutput` is irrelevant to it.

**What a shift costs, and what an absence means.** A captured expectation whose pair never
appears at the emitter is simply evicted (dropped) - allocs happen for requests whose loads
were sampled out, or cancelled, or never emitted, and that is normal. What must NOT happen
is an entry the emitter sampled but the map cannot answer: that is
`scheduler_oracle_unavailable` with a named reason (`this step's alloc hook captured no
pairing for this entry`, `no CPU hit was pending at update_state_after_alloc`,
`prefix_cacheable_group_ids excluded this manager's groups`,
`num_external_tokens truncated the pair before this entry`, ...). On the SCHEDULER side
that counter means the check is NOT running - an outage, not a fact of the fork. The
worker's `oracle_unavailable` stays exactly what 0.2.39 documented (a worker has no
requests; it grows and means nothing). The two counters are separate keys, the section
publishes `roles` and:

```jsonc
"oracle_rule": "worker oracle_unavailable is EXPECTED (a worker has no requests); scheduler_oracle_unavailable means the check is NOT running"
```

**The section names the hook, because "the oracle is fitted" was the claim that was
false.** `scheduler_oracle_hook` is `"update_state_after_alloc: fitted"` when
`disk_restore`'s observer is on the method and
`"update_state_after_alloc: NOT WIRED here"` when it is not - a renames-the-method fork, a
refused install, an uninstall all say NOT WIRED, and the install path reports its verdict
through `disk_restore_verify.note_alloc_hook()` rather than the section assuming one. The
observer is deliberately NOT part of `disk_restore.SCHED_METHODS`: a diagnostic witness must
never be able to cost the RESTORE layer its install, and it composes ON TOP of whatever
chain the method already carries (a bottom-insert would rebuild the chain through
`backport._rewrap`, which replaces the backport layer it finds there - fitting the observer
that way would silently delete a real patch while `backport` still believed it was fitted).

**What it costs the engine threads.** The alloc hook is the one place this module adds work
to a scheduler path, and it is bookkeeping only: the gate set, a dict `peek`, and at most
`MAX_EXPECTATIONS_PER_ALLOC` iterations storing 5 small ints per pair - no hashing, no
packing, no file or device, no copy of a block table beyond what `blocks.get_block_ids()`
already returns, no engine object retained (the hashes are a clipped copy). The step path
pays what it paid before: a sampling decision before any foreign state is read, and a dict
lookup for sampled entries. All hashing, packing and the neighbour walk happen on the same
single daemon thread the worker half already had; `run_pass()` drains the scheduler's ring
FIRST so a shutdown that takes the worker cannot strand the pairing verdicts.

**What a boot with one caught shift looks like** (the whole section is one object; this is
the scheduler half of it, plus the keys that interpret it):

```jsonc
{
  "scheduler_allocs_observed": 38, "scheduler_expectations_captured": 121,
  "scheduler_expectations_consumed": 119, "scheduler_expectations_dropped": 2,
  "scheduler_expectations_capped": 0, "scheduler_expectations_now": 4,
  "scheduler_steps_seen": 41, "scheduler_entries_seen": 126, "scheduler_sampled_out": 122,
  "scheduler_retained": 4, "scheduler_retained_dropped": 0, "scheduler_jobs_now": 0,
  "scheduler_checked": 4, "scheduler_equal": 3, "scheduler_mismatch": 1,
  "scheduler_mismatch_stable": 1, "scheduler_mismatch_unstable": 0,
  "scheduler_mismatch_neighbour_index": 1, "scheduler_mismatch_other_group": 0,
  "scheduler_mismatch_unknown": 0,
  "scheduler_oracle_unavailable": 0, "scheduler_oracle_refused": 0,
  "scheduler_hash_view_unbuildable": 0,
  "scheduler_index_out_of_range": 0, "scheduler_errors": 0,
  "scheduler_oracle_hook": "update_state_after_alloc: fitted",
  "scheduler_mismatch_samples": [
    {"request_id": "cmpl-a91f3c07b4-0", "group": 5, "index": 12,
     "key16": "0d6a1f47c2b58a90", "expected_key16": "77b2e0c4a19d3f65",
     "name_group": 5, "gpu_block": 1484, "classification": "neighbour_index",
     "offset": 1, "stable": true}],
  "scheduler_oracle": {"unreachable_reason": "nothing unavailable yet", "checked": 4,
                       "equal": 3, "mismatch": 1, "unavailable": 0, "refused": 0,
                       "index_out_of_range": 0},
  "roles": {"worker_apply_point": false, "scheduler_offer_point": true,
            "worker": "worker: the apply point",
            "scheduler": "scheduler: the offer point",
            "meaning": "unprefixed counters = a WORKER's apply point; scheduler_* = the EngineCore's offer point; never add the two"},
  "oracle_rule": "worker oracle_unavailable is EXPECTED (a worker has no requests); scheduler_oracle_unavailable means the check is NOT running",
  "neighbour_window": 2, "max_scheduler_mismatch_samples": 2,
  "reasons": ["scheduler_mismatch:neighbour_index: a one-slot shift (+1): the name carries the digest of block index 13 while the GPU block the fork paired it with is the engine's block index 12. A shift moves both index-aligned lists together, so it is invisible to every list-vs-list check this plugin has"]
}
```

The sample is the newest verdict and there is no separate `scheduler_last_mismatch` key.
`neighbour_window` says how far the classifier looked for the digest the name ACTUALLY
carries - a shift larger than the window is still a mismatch, it simply cannot say by how
much. Stability means something different for this role than for the worker's: the captured
hashes are a COPY, so the re-derivation cannot drift the way 0.2.40's live-object
re-derivation could; `scheduler_mismatch_unstable` now means the pair's captured bytes
disagreed with themselves on the re-walk - a thread bug, which is exactly as alarming as it
sounds.

**Its own named counters, all of them in the section's mandatory key set:**
`scheduler_steps_seen`, `scheduler_entries_seen`, `scheduler_sampled_out`,
`scheduler_retained`, `scheduler_retained_dropped`, `scheduler_thread_unavailable`,
`scheduler_checked`, `scheduler_equal`, `scheduler_mismatch`,
`scheduler_mismatch_stable`, `scheduler_mismatch_unstable`,
`scheduler_mismatch_neighbour_index`, `scheduler_mismatch_other_group`,
`scheduler_mismatch_unknown`, `scheduler_oracle_unavailable`, `scheduler_oracle_refused`,
`scheduler_hash_view_unbuildable` (a group the engine cannot view at its own block size -
group 4's 8 tokens against this box's 40-token hash - which the classification walk would
otherwise skip in silence), `scheduler_index_out_of_range`, `scheduler_errors`,
`scheduler_allocs_observed`,
`scheduler_expectations_captured`, `scheduler_expectations_consumed`,
`scheduler_expectations_dropped`, `scheduler_expectations_capped`.
`scheduler_mismatch_gpu_block_not_in_table` is GONE with the observation point it came
from: it existed because the emitter had to ASK whether a GPU block belonged to the
request; the alloc hook receives the pairing as arguments, so a GPU block outside the
group's table cannot be paired at all, and an entry whose expectation is structurally
unsound is a named `scheduler_oracle_refused`, not a re-derived finding.

**A section that bounds itself.** The budget is `MAX_SECTION_BYTES`, checked before the
section is handed to the seam (which refuses over-cap sections). The mandatory scheduler
content grew again in 0.2.41 - five expectation counters, the hook line and the gauge over
the retired classification - to ~3.5 KiB fresh, so `MAX_SECTION_BYTES` moved from 3800 to
3920 (0.2.42 added one more mandatory counter, `scheduler_hash_view_unbuildable`, and the
fresh section measures 3531 B against that cap, so no further move was needed), and that is
the whole seam story: the cap is 4096, the section ships its mandatory
keys FIRST, and `_fit()` sheds in a stated order (reason text, then the boundary
illustration lists, then the scheduler's mismatch samples oldest first, then the
explanatory prose, then the reasons). `roles`/`oracle_rule` are shed LAST because a reader
who cannot tell which process wrote a number cannot act on it, while a second mismatch
sample is only a nicer-to-read version of a fact the counters and the `reasons` ring carry
in full. A test walks every key of the fresh section against the post-trim view at full
load, so the ladder can never again shed a finding to buy room for prose.

**No new contribution, and that was forced.** As of 0.2.42 `worker_diag.MAX_CONTRIBUTIONS`
was 5 and its allow-list held exactly 5 names; `worker_diag` REFUSES an unregistered
contribution rather
than truncating it, so a sixth section would not have been "not truncated" - it would have
made `disk_restore_verify` the section missing from `diagnostics-rank.json`. Both roles are
therefore the SAME section, told apart by prefix, and the rank file gains no key it could
fail to write. The EngineCore's numbers are readable through `POST /disk
{"op":"restore-verify"}` and `/snapshot` (`dev_state_sections()` publishes the section);
no new op was added, and the scheduler role needs none. (The sixth slot exists now, and it is
the rule this paragraph states that decided how: `disk_gpu_write` took it in 0.2.43 by raising
`MAX_CONTRIBUTIONS` in the SAME change that registered the section, because the alternative was
a diagnostic silently absent from the file. The scheduler role still asks for nothing of its
own.)

`tests/test_disk_restore_verify_scheduler.py` (49 tests) is that spec: the correctly-paired
alloc is CONFIRMED at the emitter (without the positive case every negative case below is
hollow); a shift and its mirror are classified with the right SIGN; a digest from another
group's view and a name with a contradicting suffix are both `other_group`; EVERY negative
control is driven through `observe_alloc(manager, request, blocks, num_external_tokens)` -
the real signature, gate included - and through an entry shaped the way
`build_load_paths` shapes its own; the production regression has its own test (alloc in one
step, load emitted in a LATER step - the check still compares, and the same entry re-emitted
is checked again because a consumed expectation is not removed); the memory bound is proved
at the alloc hook (an expectation-per-alloc overflow and an LRU eviction are both counted,
and the map never exceeds `MAX_SCHED_EXPECTATIONS`); the gate refuses with a NAMED reason
and never a mismatch; `num_external_tokens` truncation, the `None`-gate-all rule and the
null-block skip match the fork's own loop line for line; the OFF flag means the hook counts
one line and stores nothing; the section's key set is identical fresh / after an alloc /
after an emit / after a mismatch / fat, and it fits the seam's cap under the five-section
arithmetic; and two tests drive the REAL `disk_restore.install` wrapper over
`update_state_after_alloc` feeding the fork's own pairing loop and then the real
`build_load_paths` - the hook site pinned by the product, not assumed, including the
install-then-uninstall round trip that must report NOT WIRED again. Since 0.2.42 the same
file holds the LAZY-VIEW section: a replicated `BlockHashListWithBlockSize` (never the
fork's class, never a `Sequence`) driven at this box's real 80-to-1 ratio, which pins that
the digest COMPARED is `view[index]` and never `raw[index]` (from both directions, plus the
collapsed `len(view)` as the bound), that the shift/signed-shift/group-swap controls still
fail through the view, that group 4's unviewable 8-token ring is contained and counted
rather than raised, that a shape with no index protocol is still refused, and that the
plain scale-one list answer is unregressed.

`tests/test_disk_restore_verify.py` (44 tests) is the executable spec: that the default reads
no file and no device and moves no counter (proved with the readers instrumented to RAISE,
not with the module reporting its own silence); that one flipped byte inside `W_g` is
`mismatch_stable` while a difference confined to the tail is `equal`; that a destination one
slot off FAILS rather than passes; that a reused block is classified on the re-read; that a
file which is not window-wide and a group wider than the window are refused without being
compared; that an entry waits for its DMA and a watermark-less worker is never compared; that
the oracle catches a shifted index name-vs-name cannot see, and counts itself unreachable
when no provider is installed; that the apply point touches no file and no device and barely
counts; and that two tests drive the REAL `worker_bind` -> `worker_before_load` path so the
hook site and the field names it reads are pinned by the product rather than by a fake.

## Writing the window ourselves, and watching it land (`disk_gpu_write.py`, 0.2.43-0.2.44)

Entry 28's route has a write side and a read side, and the two verifiers above each read the
GPU. This module is the write side of that route taken for real - and it is the **one dev
feature in this package that writes into live KV memory**, which is why it is default OFF,
why it is sampled, and why the interesting part of it is not the copy but the witness.

```bash
# the worker's switch is the environment, because the worker applies the restore and is the
# only process that can write a window (read once at first use; restart to change):
#   VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE=1
# this op READS: the counters, the pinned pool's live shape, the geometry the comparison
# used and the last verdicts. Its `scope` field says why an EngineCore will report
# `writes: 0` for ever - it has no apply point to write from:
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"gpu-write"}' $B/disk
# ... and this one lATCHES the process that answers. For this feature that is the wrong
# process: the write happens at the restore apply point, which only a worker runs, so arm
# the worker with the environment variable above and read the result here.
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"gpu-write","enabled":true}' $B/disk
```

**Why a wrong write here cannot corrupt anything.** At the restore apply point the plugin
knows the bytes (`key16`, the spill file, the CPU slot they were just memmoved into) and the
destination (`gpu_block_id`). So this module copies the arena slot into the GPU window itself,
on the tier's own `load_stream`, from inside the `start_load_kv` wrapper's call that runs
BEFORE the fork's body. The fork's own arena-to-GPU copy then still runs, on that same
stream, afterwards - so it has the last word on every byte. A module that guesses the
addressing wrongly puts the right bytes in the wrong place and is immediately overwritten;
what it cannot do is leave the KV cache wrong. The whole point of the feature is that the
mistake becomes *visible* instead of hypothetical.

**The witness is a pair of readbacks, and the file is the oracle.** Both readbacks are
enqueued on the same stream, one before our copy and one after, so on that stream they read
the window as it was before us and as our copy left it. The bytes they come back with are
compared against the **spill file**, never against the CPU slot the copy read from - a
test that compares the readback with its own source says `equal` for every write this module
can perform, which is a mirror and not a witness. Only the leading `W_g` bytes are compared,
for the same reason the verifier gives at length: the rest of the window belongs to other
groups and is stale by construction.

The before-state is what makes the disagreement decidable, and it is why there are two
mismatch counters and not one:

| what the readback says | counter | what it means |
|---|---|---|
| the window holds the file's `W_g` bytes | `readback_equal` | our copy landed where the entry said it would |
| the window is **exactly what it was before us** | `readback_mismatch_stable` | **the one to act on**: our copy went somewhere else. `last_verdict.shape` says whether it is `window-unchanged` (wrong block or wrong region) or `partial-span` (right place, wrong width) |
| the window changed, and not to what we wrote | `readback_mismatch_gpu_changed_on_recheck` | another stream owns the block now - the expected race, and acting on it would mean chasing the engine's normal traffic forever |
| a mismatch with no readable before-state | `readback_mismatch_unrechecked` | real disagreement, undiagnosable; it does not join the row above |

**The ordering is an observable, not a comment, and it has its own pair.** The premise of the
design is that this hook runs before the fork submits its copy, and the fork's own
`worker._load_events` ledger says whether the copy for this load event has been submitted yet.
So `readback_enqueued_before_original` moves only while that premise holds; if the hook ever
moves below the original, `event_already_submitted` and `readback_enqueued_after_original`
start moving, nothing is written, and the regression is loud rather than silent. A worker with
no ledger to read is `no_event_ledger`.

**Pinned rows are the cost, and they are released by the fork's event or not at all.** A
readback destination has to be page-locked or the read is synchronous on the step thread, and
`pin_memory=True` means a `cudaHostAlloc` - so rows are allocated ONLY on this module's own
thread, never at the apply point. The price is that the first apply after a boot finds an
empty pool and refuses; that is counted, published and correct, but since **0.2.44** it is
named as the COLD case (`pool_never_filled`, counted alongside `no_pinned_buffer`) instead of
sharing one name with the opposite diagnosis. A row handed to a `copy_(non_blocking=True)` is
still being written until the load event completes, so rows are held until the tier's own load
watermark passes the entry's event and are released by nothing else. An entry whose event never
passes has an unknowable DMA state: its rows go to `buffers_leaked` and are never handed out
again. That is the direction this module fails in - a witness that cannot afford a buffer
refuses, and a row of unknowable provenance is never a witness twice.

**THE COLD START, AND HOW TO RECOGNISE IT (0.2.44).** Rows exist only because this module's
thread allocates them, and in 0.2.43 that thread was started on the line AFTER a successful
`_take_rows` - so a cold pool refused the write, the refusal returned, and nothing ever started
the thread that could have filled the pool. The first write was unreachable, therefore every
write was: the live rank file said `applies_seen=64 writes=0 checked=0 passes=0 pool_refills=0
buffers_allocated=0 no_pinned_buffer=1 errors=0` and wrote nothing, ever, with `errors=0` the
reason nobody read it as a fault. The fix is both ends of that loop: an ARMED process starts
its writer thread when the module registers its contribution (so before the first restore, and
a thread born that way allocates nothing because no worker is registered yet), and the write
attempt asks for the thread at its TOP, before it touches the pool, so even an unregistered
module starts it on its first refusal. **An operator recognises the dead state in one field:
`liveness.state` is `armed but never wrote`, and `liveness.verdict` spells it out with the
numbers - `passes`, `pool_refills` and `buffers_allocated` all zero with `applies_seen`
climbing is a writer that never ran, and `thread_started=0` is the proof.** `liveness.state`
is one of `off` / `armed, no apply point seen in this process` (an EngineCore: no apply point,
so it can never write) / `armed but never wrote` / `writing`. The one refusal name that used to
carry two facts is now counted apart: `no_pinned_buffer` with `pool_never_filled` beside it
means "the witness thread has not warmed up, the next apply will be written", and `no_pinned_buffer`
alone means the pool filled and drained - rows held by records whose load event has not passed,
or leaked, which never recovers by waiting. The third member of that family, the in-flight cap,
was already separable through `in_flight_dropped`.

**What it costs when it is on**, all of it published in the section: one entry per
`sample_every_n_applied` (64) applies, at most `max_writes_per_step` (1) write per step and
one per `min_interval_seconds` (0.5), at most `max_checks_per_pass` (2) comparisons per pass,
at most `max_in_flight` (4) records waiting for their watermark, at most
`MAX_PINNED_ROWS_PER_REGION` (2) rows per region name and width. The step thread reads
dictionaries and enqueues copies; it allocates nothing, waits for nothing and cannot raise.
Every refusal is a name (`width_mismatch`, `region_pair_missing`, `no_gpu_block`,
`no_group_width`, `row_too_wide`, `stream_unavailable`, `refused_thread_unavailable`,
`write_failed`, ...) and leaves the entry ENTIRELY to the fork's copy, which was always going
to run.

**It copies by name, in the order the file is built in.** `sorted(name)`, GPU region `name`
paired with CPU region `name`, and the per-region widths must total the entry's own byte count
exactly - the fork's `build_params` assertion and `_plan_regions`' total, not a new invention.
A GPU region with no CPU region of the same name is `region_pair_missing`, because the only
other way to pair them is positionally, and one region's bytes in another region's window is
the silent corruption that rule exists to prevent.

`disk_gpu_write` is the **sixth** contributed section in `diagnostics-rank.json`, and the cap
had to move from 5 to 6 to let it exist: the seam refuses an unregistered section, so a sixth
section without the raise would have been silently absent - the worst place for a diagnostic
about live KV writes to be missing, because absence reads as "nothing to report". Six sections
can ask for 24 KiB of the 32 KiB ceiling, which is why contributions are what the byte triage
sheds first.

`tests/test_disk_gpu_write.py` (37 tests) is the executable spec. The default posture is
proved against seams that RAISE if touched, not against a counter agreeing with itself. One
flipped environment variable separates that from a write that puts the file's bytes in a GPU
block and a verdict that says so. The journal of submissions reads `pre-readback, our copy,
post-readback, the fork's copy` as a list, and running the fork first stops the ordering
counter while `event_already_submitted` starts. The negative controls are the point: shifting
only the copy's destination must report `readback_mismatch_stable` (the single counter that
means our addressing is wrong), halving only its extent must report the same counter with
`shape: partial-span`, letting another stream rewrite the window must report
`..._gpu_changed_on_recheck` instead, and moving the FILE must move the verdict with the copy
untouched - because a comparison against the copy's own source could not. The lifetime is
pinned as a fact about a pool: a slow watermark delays the release, provably refuses the next
write for want of a buffer, and ages out by leaking rather than recycling; and the step thread
is proved never to have called the allocator at all. The tail is refused in both directions, a
plan that does not total writes nothing, the two fake cache dicts are deliberately built in
different insertion orders so a positional pairing fails, and two tests drive the REAL
`worker_bind` -> `worker_before_load` path so the hook site, the field names and the order
against the fork's copy are pinned by the product rather than by a fake batch.

## Sizing the CPU-slot reserve an offer leaves alone (`disk_restore.py`, 0.2.38)

An offer does not own a KV byte, but it does own **destination slots**: the blocks a disk
restore reads INTO come out of the host-RAM arena's CPU block pool, the same pool the RAM
tier parks its own blocks in and the same pool the spill path reads OUT of. So an offer
that drained the pool would trade a recompute for a spill that can never happen - which
is why the offer path holds a **reserve** back and clamps itself to

```text
spans = (free_slots - reserve - boundary_slots) // slots_per_span
```

`free_slots` is the engine's own `cpu_block_pool.get_num_free_blocks()`, read exactly the
way the offer reads it; `slots_per_span` is the number of prefix-cacheable positional
groups (1 on the live Qwen3.8-Flash-Next box, whose four GDN groups are *boundary states*
and cost one slot each at the run's end); and `reserve` **was the constant 64 and is now
`VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE`**, defaulting to 64.

**Why it became a knob: the arena it was sized for is gone.** 64 slots was chosen against
a 64 GiB arena. The operator has since run the engine at 8 GiB, which leaves **187
allocatable slots per rank on the PP0 (narrow) pair and 166 on the PP1 (wide) one** - so
the reservation is 41% of what is left on the wide rank before the offer path sees a
slot, and the whole difference is restorable tokens that are already sitting on the SSD:

| free slots/rank | reserve 64 (shipped default) | reserve 32 | reserve 16 | reserve 0 |
|---|---|---|---|---|
| **166** (PP1, 8 GiB arena) | **98 spans = 313,600 tok** | 130 spans of arithmetic -> **128** (walk cap) | **146** -> **128** (walk cap) | 162 -> **128** (walk cap) |
| **187** (PP0, 8 GiB arena) | **119 spans = 380,800 tok** | 151 -> **128** (walk cap) | **167** -> **128** (walk cap) | 183 -> **128** (walk cap) |

Tokens are spans x the engine's 3,200-token scheduler block. **Read the cap column, not
the arithmetic column**: `MAX_RESTORE_BLOCKS_PER_OFFER` is 128 spans, so a *single* offer
cannot exceed 128 spans (409,600 tokens) whatever the reserve says. That makes **~34 slots
the useful floor on a 166-slot rank and ~55 on a 187-slot one**: below that, a smaller
reserve buys no longer offers and spends the RAM tier's headroom for nothing.

**And what the reserve is worth below that floor, stated as a failure mode** - because the
first reader of a "lower it" instruction will reach for 0:

- **0 is honoured**, and it means *"an offer may take the arena's last free slot."* The
  arithmetic stays correct, so this is a posture and not a dead knob - but a burst of disk
  offers then takes the arena to zero free, the RAM tier above has nowhere to park, the
  next spill has no destination, and an offer can even allocate the last slot out from
  under a request that was about to need it. Honourable on a box that restores everything
  from SSD and parks nothing in RAM; the reason the default is 64 and not 0.
- **A reserve at or above the free count declines offers** (`no_cpu_slots`), allocates
  nothing, and raises nothing. This is why there is **no ceiling** on the variable: the
  reserve can only ever make an offer *smaller*, so honouring an absurd value is the safe
  direction, while clamping it would silently replace the machine the operator asked for
  with the one they didn't.
- **A non-integer or a negative value falls back to 64 and logs once**, `DISK[reserve]
  IGNORED: ...; the shipped default 64 stands`. A negative is refused as grammar, not as
  policy: `free - (-1)` would hand the offer more slots than the pool has, and the fork's
  own `get_new_blocks` raises when a count exceeds the free list.

**What it looks like at boot** - the same line that prints the other per-offer bounds,
with its origin, because the number alone cannot answer "did my drop-in arrive?":

```text
DISK[restore] installed=YES ... offer_block_cap=128 offer_block_cap_tokens=409600 \
  cpu_slot_reserve=16(env) ...
```

`(default)` when the variable is unset, `(env)` when the environment's number won,
`(env-ignored)` when the grammar refused what was typed.

**Where to read it without grepping the log.** The value in force is published in the
restore section's `limits.cpu_slot_reserve` (with `cpu_slot_reserve_default` /
`_source` / `_env_var`), in `offer_pool` beside `jobs_the_pool_can_carry`, in the
per-offer `last_offer_pool` next to the bound it decided
(`cpu_slots_free`, `spans_the_cpu_slots_can_carry`, and `bounded_by: ["cpu_slots"]` when
it was the one that cut), in the boot verdict's `arena` dict, and inside the
`no_cpu_slots` refusal reason. So the question *"why was my offer only 98 jobs?"* is
answered by the payload: `166 free - 64 reserved - 4 boundary states = 98 spans of 1
slot`. The rank file deliberately gained nothing: it lives inside
`worker_diag.MAX_CONTRIBUTION_BYTES`.

`tests/test_disk_cpu_slot_reserve.py` (24 tests) is the executable spec: the unset boot
reserves 64 and the pool-with-no-room-above-64 refusal still fires, the affordability
table above is asserted as arithmetic at both ranks' real free counts, the live offer path
is measured at the real 12,902,400-byte width (98 spans at reserve 64, 146 at 16, a
difference of exactly the 48 slots freed), an over-large reserve declines without
allocating or raising, 0 is honoured, every invalid value falls back with one log line,
the value reaches every surface that names the bound, no key set moves when the knob
moves, and an AST scan pins that the bare constant is read by exactly one function - the
resolver - so the default can never become a second source of truth.

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
  4. **Then the disk tier's own boot default runs** (`KVTIER_DISK_BOOT_ENABLE`, ON by
     default since 0.2.33): it arms the SSD spill tier - as a LATCH since 0.2.35, with no
     expiry - and raises this boot's serve posture, so the box comes up `serving_as=disk_authority` with no
     POST. It calls the same two functions the `/disk` verb calls - `disk_spill.arm()` and
     `disk_restore.set_disk_authority()` - invents no third way to arm, runs after the
     RAM tier's arm (a disk arm on a boot that armed no tier is refused), and rolls the
     pair back to uniformly disarmed if either half fails. Its verdict is
     `pr56810.boot.disk_boot`, its token is `disk_boot=` on the BOOT line, its own log
     line is `KVTIER disk boot default=...`, and the whole design is in *The SSD tier's
     boot default (0.2.33)* below; `disk_restore.boot_default` in the snapshot (and
     `restore.boot_default` on every `/disk` answer) is where a poller reads it.

  Guard before arm is not a style preference: arming the tier without the guard
  means the first real tier bug is an EngineCore death instead of a recorded
  decline, and the arm gate therefore treats "the guard refused this method" as a
  reason not to arm.
* **One line says all of it.** `pr56810.boot.line`, also written to the engine log
  at WARNING so it survives without curling anything:

  `BOOT[auto] backport=YES guard=YES ram_tier=ARMED disk_tier=ARMED tier=ARMED faults=0 journal_dir=tmpfs arena=OK disk_boot=ARMED guard_over=... not_covered=... restore=YES restore_gate=closed restore_ranks=0 captures=...`

  `tier=` is the **RAM/host tier** and stays exactly as it was - the restart verifier
  greps it as a PASS criterion - while `ram_tier=` repeats that verdict under an
  unambiguous name and `disk_tier=` reports the **disk spill tier's arm**, whose file
  lives on tmpfs and is gone after every reboot. **What changed in 0.2.33 is what that
  field says on a default boot: `ARMED`, not `DISARMED`**, because the boot default armed
  it - and `disk_boot=` names the boot default's own verdict (`ARMED` / `OFF` /
  `NO_STORE` / `INERT` / `REFUSED` / `UNDECIDED`). A boot can still legitimately print
  `disk_tier=ARMED` next to a restore refusal (the worker proof is a separate gate -
  `restore_gate=closed` above says so): arming says bytes may be written, the serve
  posture says requests may be served, and the worker proof says a worker will fill the
  slots. The two named fields are printed *before* `tier=` so that a
  `grep -oE 'tier=[A-Za-z_]+' | tail -1` still lands on the bare field.

  `mode` is `auto` or `inert` and says which env posture ran; `refused=` names each
  gate that failed with its reason; `not_covered=` lists the methods the guard does
  **not** wrap (methods the fork's class does not define, or ones it inherits from
  outside the tier module) so the line never implies more coverage than it has;
  `faults=` is the guard's fault count at that instant, repeated as `faults_now`
  when you read the payload later.

```bash
# 1. boot the engine (tier env set, VLLM_E2_TIER_DEV_BOOT_INERT unset). Expect THREE
#    lines in /tmp/vllm-serve.log:
#    "kvtier-dev-plugin: connector-creation hook fitted on KVConnectorFactory..."
#    "kvtier-dev-plugin: KVTIER disk boot default=ON (KVTIER_DISK_BOOT_ENABLE unset -> default ON): HONOURED: ... serving_as=disk_authority with NO POST. ..."
#    "kvtier-dev-plugin: BOOT[auto] backport=YES guard=YES ram_tier=ARMED disk_tier=ARMED tier=ARMED ... disk_boot=ARMED ..."
#    and NOTHING else to do. `grep 'BOOT\[' /tmp/vllm-serve.log | tail -1` is the
#    whole boot verdict; `grep 'disk boot default' /tmp/vllm-serve.log | tail -1` is the
#    disk tier's half, and it names the way to turn the default off.

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

### The SSD tier's boot default (0.2.33) (`KVTIER_DISK_BOOT_ENABLE`)

> **A default boot arms the SSD tier and serves from it, with no POST - and the flag can
> only ever take a boot back to manual, never forward into serving.**

Until 0.2.32 every boot came up with the SSD tier doing nothing and stayed that way until
somebody POSTed twice - `{"op":"arm","seconds":7200}` and then
`{"op":"authority","enabled":true}`. Both are per boot, so every restart, every wake and
every "let me just restart it" silently reverted to a tier that serves nothing while
`serving_as` truthfully reported `off`. That is the most common way this feature looks
broken to a human, and it is what this release removes.

**One flag, ON by default.** `KVTIER_DISK_BOOT_ENABLE` unset (or `1`/`true`/`yes`/`on`)
means the boot that armed the RAM tier also arms the spill tier for the full cap and
raises the serve posture, so the engine comes up `serving_as=disk_authority`. Set
`0`/`false`/`off`/`no` and the boot is byte-for-byte the 0.2.32 boot: disarmed spill tier,
`serving_as=off`, `restore.gate.reason_kind: "disarmed"`, and the two POSTs are the only
lever. A value that is not a word in either vocabulary (`flase`) reads as the default -
**ON** - and says so on the boot line, because a typo that silently means "disabled" is a
box still serving from disk after somebody decided it should not.

**It is a disable switch, not an arm switch.** Every other facility here starts off and
is armed by a token-gated POST, and that rule is untouched: guard mode, the RAM tier's
arm, the byte dump and `spill-one` have no env var. `KVTIER_DISK_BOOT_ENABLE` has the
same shape as `VLLM_E2_TIER_DEV_BOOT_INERT` - it can only turn a boot effect **off** - and
`VLLM_E2_TIER_DEV_BOOT_INERT` outranks it, because an inert boot arms nothing at all and
the disk default obeys that (reported as `INERT`, not as a failure).

**THE PROHIBITION IT EXISTS TO RESPECT.** `llm-api/AGENTS.md` (the 2026-10-03 incident)
forbids a boot-time re-arm: *"A stale `disk-spill-arm.json` did NOT re-arm workers nine
hours later; ranks report `armed=False` on their own. Do not add a boot-time re-arm."* A
proxy-side "restore posture on engine-ready" hook was written for exactly that and was
deleted rather than left default-off. The line this feature walks, and the only line
there is: a **previous boot's persisted state decides nothing**. The boot's decision comes
from this one environment variable and from the boot's own measurements, and
`disk_boot.py` contains no vocabulary for reading an arm file - no `open`, no `json`, no
`_arm_path`, no `boot_id` - which `test_the_prohibition_structurally_the_module_never_reads_an_arm_file`
pins against the module's syntax tree, while
`test_the_prohibition_no_boot_time_re_arm_from_persisted_arm_state` pins it behaviourally
by planting an arm file that claims `"armed": true` and naming a **different kernel boot
id** (what a reboot actually leaves behind): the boot comes up disarmed with the flag `0`
and armed with the flag unset, and in both cases the planted file is refused by
`disk_spill`'s own boot fence - and with the flag `0` the boot's fail-safe **deletes** it,
because since 0.2.35 a fence that cannot see an engine restart is no longer enough. The
sibling test `test_the_boot_disarms_a_file_left_by_an_earlier_boot_on_one_kernel` plants
the file the fence CANNOT refuse (this kernel's boot id) and pins that it is gone. A surviving arm file still arms nobody; a restart is
still a rollback.

**The gates, and the fail-safe direction.** Every outcome except `ARMED` leaves this boot
uniformly disarmed, and `ARMED` requires **both** halves to have landed:

| gate | outcome | why it is a refusal |
|---|---|---|
| the flag says no | `OFF` | today's boot, exactly |
| `VLLM_E2_TIER_DEV_BOOT_INERT` set | `INERT` | that boot arms nothing, disk included |
| no `VLLM_E2_TIER_DISK_ROOT` | `NO_STORE` | nothing to arm - **not** a fault |
| the RAM tier did not arm | `REFUSED` | a disk offer needs scheduled load to ride on |
| the restore layer is not fitted in this process | `REFUSED` | it would write spilled KV and never read a byte |
| the host-RAM arena is tiny / has no pool / unreadable | `REFUSED` | the tier moves bytes through that arena both ways |
| the spill journal cannot be written where it lives | `REFUSED` | every spill would be a file no index can name |
| `disk_spill.arm()` refused | `REFUSED` | the store's own objection, reported verbatim |
| the arm could not be published to the ranks | `REFUSED` + rollback | an EngineCore-only arm is a half-armed fleet |
| the posture could not be raised | `REFUSED` + rollback | an armed tier that serves nothing is the state to avoid |

Nothing here can raise an engine's boot into an exception: the whole path is wrapped, and
an internal raise is reported as `REFUSED` with the exception type named.

**THE ARM IS A LATCH (0.2.35). There is no expiry, so there is nothing to renew.** Until
0.2.34 the arm was a timer (`ARM_DEFAULT_SECONDS` 1,800 s, `ARM_MAX_SECONDS` 7,200 s, both
now deleted) and this module carried a `renew_if_due()` tick that rode the scheduler's
offer path to beat the cap - a whole heartbeat mechanism whose only job was to keep a
clock from going dark. Deleting the clock deleted the tick, its constants, its counters
(`renewals`, `renewal_failures`, `last_renewal`) and its payload fields; `arm_state()`
reports `latched: true` and no time, and the boot view answers `arm_expires: false` where
it used to answer `seconds_left`.

**Why the timer went.** The operator's ruling, after the live box made the case: KV
persists until the LRU sheds it under the ceiling, and the ceiling eviction that sheds it
**runs from the spill writer path** - so an arm that expired froze *shedding* along with
growth, which is the opposite of the product: a tier that cannot shed is a tier that
refuses spills forever. The timer was not buying a bound either. At the live 11,468,800 B
slot and 200 files / 60 s one rank writes ~38.2 MB/s, which fills a 192 GiB ceiling in
about 90 minutes: a 7,200 s arm permits ~1.34 ceiling-fills and the 1,800 s default ~0.33.
Capacity is bounded independently of the arm - by the byte ceiling, the per-rank fair
share, LRU-by-touch, the `MIN_FREE_BYTES` free-space floor and the write-rate window - and
two of those mechanisms run **only while the arm is up**. The 2026-10-05 livelock the
expiry was partly credited with preventing is closed by six mechanisms that read no arm
expiry (the AND-over-ranks proof with per-rank service facts, the arm read live from one
authoritative file, the request and span fuses, the forward-progress fuse, the
contiguous-run offer bound, and a CONFIRMED witness that cannot be reached without proof).

**What the latch does not survive is a decision nobody made.** The boot fence in
`disk_spill._arm_boot_fence` still refuses an arm file naming a different **kernel** boot,
but `/proc/sys/kernel/random/boot_id` cannot see an ENGINE RESTART inside one long uptime
- four boots in a 2w3d uptime all look like "this boot" - and while the arm had a TTL, the
TTL is what made a file from boot #1 harmless to boot #2. The rule that replaces it is
**THE ONLY THING THAT ARMS A PROCESS IS THAT PROCESS'S OWN BOOT DECISION**: `boot_enable()`
calls `disk_spill.disarm()` on every outcome except `ARMED`, and `disarm()` deletes the
file. A boot that did not choose to arm leaves nothing behind to be misread.

**A human always wins, and the ordering is not negotiable - and since 0.2.35 it is the
veto that enforces it.** `POST /plugins/kvtier/disk {"op":"disarm"}` - or
`{"op":"authority","enabled":false}` - latches `disarmed_by_operator: true` with the
reason, and the boot decision **refuses to arm while it stands**. That is the veto's new
job: it used to gate `renew_if_due()` and nothing else, so when renewal was deleted it
guarded an empty set and could have gone with it. It is kept, for a stronger reason - with
a latch the path that can still re-arm a box behind its operator is `boot_enable()`
itself, and a second connector boot inside one process (which a re-instantiated engine
does) would otherwise raise an arm that, with no timer, would never come down. A manual
`{"op":"arm"}` lifts the veto: only a human at the verb does, and only because that human
asked for the tier back. A *refused* POST latches nothing, so a malformed curl cannot
silently kill the feature. And because the latch gates **both** directions, disarming
closes serving from disk as well as writing to it - `gate()` answers
`reason_kind: "disarmed"`, and `boot_enable`'s fail-safe and `_may_restore_worker` read
the same verdict.

**One line, at boot, naming how to turn it off.** It is WARNING-level, printed once per
process, and it is the answer to "what did this boot decide and how do I change it?"
without opening a source file. Verbatim, on a default boot:

```
kvtier-dev-plugin: KVTIER disk boot default=ON (KVTIER_DISK_BOOT_ENABLE unset -> default ON): HONOURED: armed the spill path and set this boot's serve posture, so it comes up serving_as=disk_authority with NO POST. The arm is a LATCH (0.2.35): it has no expiry, there is nothing to renew, and it comes down on a POST that disarms it and not on a clock. It gates SERVING from disk as well as writing to it, so the same POST stops both. to stop every future boot doing this: set KVTIER_DISK_BOOT_ENABLE=0 in the engine's environment and restart (VLLM_E2_TIER_DEV_BOOT_INERT=1 turns off the whole auto-boot, not just the disk tier); to stop THIS boot right now: POST /plugins/kvtier/disk {"op":"authority","enabled":false} and then POST /plugins/kvtier/disk {"op":"disarm"} - the arm is a LATCH, so that holds until someone arms again, and a manual disarm vetoes this module's own boot decision for the rest of the process too
```

```bash
# What the boot decided, three ways. All three read the same state through the same
# builder, so they cannot disagree: the boot line's `disk_boot=` token, the snapshot's
# `disk_restore.boot_default`, and `/disk`'s own `disk_boot_default`.
grep 'disk boot default' /tmp/vllm-serve.log | tail -1
curl -s -H "x-kvtier-dev-token: $TOK" $B/snapshot | python3 -c \
  'import sys,json;b=json.load(sys.stdin)["disk_restore"]["boot_default"];print(b["outcome"],b["honoured"],b["armed_now"],b["arm_expires"],b["disarmed_by_operator"])'
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"status"}' $B/disk | python3 -c \
  'import sys,json;print(json.load(sys.stdin)["disk_boot_default"]["reason"])'

# Take it off THIS boot, and permanently, in that order.
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"authority","enabled":false}' $B/disk
curl -s -X POST -H "x-kvtier-dev-token: $TOK" -d '{"op":"disarm"}' $B/disk
# ...and for the next boot: Environment=KVTIER_DISK_BOOT_ENABLE=0, then restart.
```

The payload is the same dict on every door (`disk_restore.boot_default` in the snapshot,
`restore.boot_default` on every `/disk` answer, and `disk_boot_default` again at the top
of a `/disk` `{"op":"status"}`), built by one function, and it carries **the same key set
on every path** - a boot that never ran the default, one that was refused, one whose flag
was `0` and one whose arm could not be read all answer with the same fields, so an
absence never has to be interpreted on the path where something went wrong.

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

### The disk tier went off by itself on a quiet box (0.2.45)

> **A rank file is a proof of life, and a proof of life needs a pulse.**

If `gate().reason_kind` is `no_worker_proof` (or the disk path simply stopped serving
after a quiet period) while the engine is healthy and idle, the rank files are stale, not
absent. Since 0.2.45 each worker writer keeps its own file alive on a 60 s daemon thread,
started at `install()` - so the question is whether that thread is running, which is a
read:

```bash
# The worker's own answer, per rank: the file's stamp, and how many beats it has
# written. A stamp older than 900 s with a live engine is the defect, not a fault.
ssh llm 'for f in /tmp/kv-tier-diagnostics/rank-*.json; do
  python3 -c "import json,sys,time; d=json.load(open(sys.argv[1])); c=d[\"counters\"]
  print(sys.argv[1], int(time.time()-d[\"ts\"]), c.get(\"heartbeats\"),
        c.get(\"heartbeat_errors\"), c.get(\"heartbeat_respawns\"))" "$f"
done'
```

* `heartbeat_alive: false`, or `heartbeats` frozen while `applies_seen`/steps climb, means
  the heartbeat is not running in that process - and `heartbeat_respawns` says whether a
  step has re-armed it (the step-driven path re-arms a thread this process can see is gone).
* `heartbeat_errors` climbing names a beat that raised (a filesystem, a contributor's
  producer); the beat is contained, so the loop survives and counts it.
* `heartbeat_beats_in_window` is the margin: the shipped 60 s inside 900 s is fifteen
  beats, so an operator who raises the interval is one missed tick from closing the disk
  path again, and this field is what makes that visible before it happens.
* **Before 0.2.45 this symptom had no cause to find.** An idle worker simply wrote nothing
  and the 900 s window elapsed. Nothing was broken; a heartbeat was missing.

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

2489 tests, no engine and no vllm required (`.venv/bin/python -m pytest -q`; the
count in this README is a snapshot - trust `pytest --collect-only`). The shadow
verifier is `tests/test_disk_shadow.py`: that the default reads no device memory, that
the sampling and the per-bind-window cap throttle instead of queueing, that a moving GPU
block is classified as staleness and a stable one as the layout candidate, that every
fault (no GPU caches, a device error, a width disagreement, a refused spill) is a
counter and never an exception on the step thread, and that the spill lands the same
bytes with the witness on as with it off. The GPU-direct shadow writer is
`tests/test_disk_gpu_shadow.py`: that the default creates no directory, that an `equal`
block produces a file whose bytes ARE the arena's bytes and a block the verifier did not
vouch for produces none, that the LIVE STORE is byte-for-byte what it would have been
without the feature, that a writer which raises or a root that cannot be written is a
counter and not a crash, that retention removes its own file and not a same-named file in
the store, that the shadow file is the `sorted(name)` concatenation and not the
insertion-order one, and that a fat published section sheds its illustrations rather than
its numbers. The restore-side verifier is `tests/test_disk_restore_verify.py`: that
the default opens no file and reads no device (proved with both readers instrumented
to raise), that a flipped byte inside `W_g` is a stable mismatch while a difference
confined to the tail is `equal`, that a destination one slot off FAILS rather than
passes, that a reused GPU block is classified on the re-read, that a file which is
not window-wide is refused rather than compared, that an entry waits for the fork's
DMA watermark, that the binding oracle catches a shifted index that name-vs-name
cannot see and counts itself unreachable when no provider is installed, and that two
tests drive the real `worker_bind` -> `worker_before_load` path so the hook site is
pinned by the product. The GPU-direct writer is `tests/test_disk_gpu_write.py`: that the
default issues no copy and no page-locked allocation (proved with the three device seams
instrumented to raise), that the submission journal reads pre-readback, our copy,
post-readback, then the fork's copy, that shifting only the copy's destination reports
`readback_mismatch_stable` while another stream rewriting the window reports
`..._gpu_changed_on_recheck`, that moving the spill FILE moves the verdict with the copy
untouched (so the comparison cannot be a mirror of its own source), that a pinned row is
released by the fork's load event and by nothing else and ages out by leaking rather than
being reused, that the step thread never calls the allocator at all, that a COLD armed boot
reaches a real write through the production entry point alone - no patched `_ensure_thread`,
no pre-filled pool, the product's own sample rate - because the cold refusal must start the
thread and the allocations must all land on the module's own thread and never on the step
thread's (the 0.2.44 regression), that `liveness` names which of the four states the writer is
in with the same key set in all four, and that a cold pool and a drained pool are two different
readings rather than one name, and that the region
pairing is by name in `sorted(name)` order - pinned by two fake caches deliberately built
in different insertion orders. The per-offer
bounds after the fixed 32,000-token cap was deleted are in
`tests/test_disk_offer_bounds.py`, with the affordability ladder that replaces the cap
as the guard against un-placeable offers. The 0.2.30
fair-share ceiling is in `tests/test_ceiling_fair_share_multi_rank.py`, the fill block in
`tests/test_disk_fill_block.py` and its access rules in `tests/test_routes.py`; the 0.2.29
touch-clock / peer-rank / deadline / evict-pool behaviour is in
`tests/test_touch_promotion_and_peer_eviction.py`. They cover:
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

The disk tier's **boot default** (`tests/test_disk_boot_default.py`) is written as the
two properties it must not lose. One half drives the REAL boot (install, guard, tier arm,
restore fit, arena verdict, then the disk default) and asserts the box comes up
`disk_authority` with an arm at the cap, an arm file naming this boot, and not one POST -
then asserts `KVTIER_DISK_BOOT_ENABLE=0` reproduces the 0.2.32 boot field for field, down
to the gate's `reason_kind`. The other half is the **prohibition**: an arm file from a
previous boot (a different kernel boot id, which is what a reboot leaves) is planted
claiming hours of life, and the boot's decision is shown to come from the flag and not
from the file - behaviourally, and structurally, by walking `disk_boot.py`'s syntax tree
for any name, attribute, string constant or call that could read persisted arm state.
Between them: a manual disarm beats the boot default - driven through the VERB and proved
against the only path that can still re-arm a latched tier, a SECOND `boot_enable()` in the
same process - the veto survives until a human lifts it by arming, an arm survives ten hours
of advanced clock (the inversion of the deleted expiry test), an old-format arm file with a
past `expires_ts` still arms (the upgrade path), `{"op":"arm","seconds":7200}` still arms and
reports the number `seconds_ignored`, a file planted with THIS kernel's boot id by an earlier
engine boot is deleted by a boot that did not arm, every exit of `boot_enable` that is not
`ARMED` provably routes through the fail-safe (an AST check on the function's `return`s),
every failed gate leaves the boot uniformly disarmed and prints exactly one line naming the
gate, and the status payload carries the same key set on every path including the one where
the arm cannot be read. The deleted renewal suite's seven tests went with the tick they
tested.

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
| `tests/test_quiet_rank_bytes_are_not_dead_bytes.py` | 12 | a rank that has not written for 300 s keeps its bytes while its worker is alive and publishing that digest (the defect, on a real filesystem); the peer map dropping a quiet rank's digest is not evidence of death; liveness from the RANK FILES, not the digest map; `stale_geometry` reclaim still runs and still runs ahead of this rank's own cold bytes; `stale_geometry_removed` and `peer_files_removed_live` separate the two doctrines; an empty peer map claims nothing, so no identity may be called dead on it; `identity_known`/`liveness_source` name the evidence a pass had |
| `tests/test_index_forgets_no_bytes_on_disk.py` | 13 | eviction is LRU-by-TOUCH and not insertion order (a just-restored identity is the last to leave, and on the unpatched tree it was the first); a read that promotes a file touches the index on the throttled path too; `VLLM_E2_TIER_DISK_INDEX_ENTRIES` configures the cap and a bad value falls back and reports; `entries_dropped` balances its three reason counters exactly; `entries_dropped_file_present` counts the index forgetting a live file and never guesses at an unchecked one (`unverified`); `note_restored_path` re-admits an identity a restore proved |
| `tests/test_idle_engine_keeps_its_disk_proof_alive.py` | 10 | install alone with ZERO engine steps produces and refreshes the rank file (the defect: only a step used to write it); `disk_restore.worker_proof` stays open across an idle spell and re-opens after a forced-stale file with no step in sight; the heartbeat starts at `install()` and never stacks a thread; it is a daemon with a findable name; the handler is held WEAKLY and a beat with no handler still writes; the write floor bounds forced beats; no `VLLM_E2_TIER_DIAG` means no thread and no file; the shipped interval fits the 900 s window with four beats to spare (pinned against the UNPATCHED default); the payload reports age and margin; a dead thread is respawned and counted |
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
