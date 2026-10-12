# Reference tools: `kvconfig`, `kvinspect`, the trace ring, `blockdump`

Moved from `README.md`. The four single-purpose instruments: read the KV-cache
geometry before the first store, ask about one block or one request, watch the
scheduler steps around a decision, and dump a few parked bytes to settle a
byte-level question. The route table, the argument rules and the `/snapshot`
payload fields are in `docs/diagnostics.md`; the operator's emergency procedure is
the runbook in `README.md`; the NVMe tier's own reference is `docs/disk-tier.md`.

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

