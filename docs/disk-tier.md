# Disk tier: reference (spill, restore, bounds, readers)

Moved from `README.md`. Reference material for the NVMe spill/restore tier that
is neither a newcomer's instructions nor a dated narrative: the eligibility
arithmetic, the bound tables, the journal's filesystem rules and the reader
relay's internals. For the story of how each rule was arrived at, see
`DEVLOG.md`.

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
did or did not fuse (see *What a failed load costs*). Both tables are bounded
(`MAX_SUPPRESSED_REQUESTS` for requests, `MAX_SUPPRESSED_SPANS` for spans since 0.2.60,
oldest-first), because a fuse that grows without limit is its own leak - the span table
used to be written with no bound at all.


**The scheduler refuses while any live rank refuses to serve.** A worker's rank file
publishes `disk_restore.armed` = *will this rank service a disk load*, and
`worker_proof()` is AND over the live ranks: one disarmed rank closes the whole
scheduler, because the load event spans every rank and the blocks the refusing rank
passes through stay un-filled while the fork has already committed to them. The
refusal is `reason_kind=scheduler_armed_workers_disarmed` with `ranks_armed=0/4`, in
`gate()`, in the boot line and in the counters. `installed` alone is not proof - that
is the field that lied on 2026-10-05, when four ranks were fitted, none was armed, and
the offers kept coming.


**Why the offer path cannot stat the store itself.** The scheduler does not know the
file's name: four ranks name one identity four ways (the digest half covers rank-local
`rank` and `total_bytes_per_block`), which is why `_identity_path` sends a sentinel and
`_resolve_worker_path` rebuilds the name per rank. So a stale entry cannot be caught by
a `stat()` in the scheduler; it is caught where the name is known - the worker's read,
which classifies `file_missing`/`size_mismatch` and reports the GPU block as a load
error, and the scheduler recomputes. That is the cheap honest gate, and it costs one
wasted load round trip per stale entry, never wrong KV.


**The relay, and why a receipt is a proof rather than an acknowledgement.** The ask goes
into `disk-read-threads.json` in the diagnostics directory - a SECOND channel, never
`spill-one`'s, because a spill ask is an event and a resize ask would silently overwrite
a pending one; each reader requires its own `role`/`op`, so a record from the other
channel is malformed data and never an instruction. A newer ask SUPERSEDES an unanswered
older one (the opposite of `spill-one`'s refusal to share, and right for a statement
about a count). A worker picks it up on the `get_finished` poll, under the same arm gate
and the same `POLL_INTERVAL_SECONDS` budget - **no new poller, thread or timer**, and on
a boot that never uses the relay the branch costs one failed `getsize` per poll.

The worker calls the SAME `set_read_threads` the in-process verb calls, so there is no
second copy of the count that a later bind could disagree with, and then writes
`disk-read-threads-receipt-<rank>.json` carrying **its own live counters read back after
the apply** (`configured`, `alive`, `surplus_draining`, `threads_source`). The caller does
not believe `refused: false`: it re-derives the verdict from those numbers, so a receipt
whose count does not equal the count it was asked for is reported `applied: false` with
the contradiction named. That is the difference between "a rank received it" and "a rank
applied it", and only the second one is counted.

`relay.status` is one of eight named answers, and the two that exist to refuse to lie are
`not yet answered` (the default — nothing is claimed because nothing was waited for) and
`answered_unverifiable` (ranks applied it and none refused, but no capacity handshake was
recorded in this process, so nobody can say who is missing - it is NOT the same answer as
`complete`). `ranks_answered`, `ranks_applied`, `ranks_refused` and `ranks_pending` split
the fleet four ways, and a rank that never spoke is kept apart from a rank that answered
`no`. `ok` is false whenever an ask was made and did not land (`no_rank_answered`,
`answered_refused`, `partial`) - **never a success that means "nobody saw it"** - with one
exception: a relay that could not be attempted at all (`no_channel`, `not_armed`) reports
its refusal without failing a call whose local half succeeded, unless you asked for the
ranks by name with `"ranks": true`, which gets the failure it asked for. A process with no
rank id (the asker) writes no receipt, so it can never answer its own ask.

**THE WAIT POLICY, and why the default is zero.** `/disk` dispatches this verb on the
EngineCore loop thread, and a rank reads the relay channel on its **step path** — a step
the same loop dispatches. So a wait in this handler waits on the very thing it is
stopping, and each second of it is serving that is not happening. (The first production
resize proved it: `no_rank_answered` at exactly the 3.0 s default, with every receipt
already on disk microseconds later.) Hence:

- **`wait_s` defaults to 0** — the ask is posted, the call returns at once with
  `ranks_pending` naming the silence, and a later `{"op":"read-threads"}` read of the
  receipts finishes the story. That read is the flow, not a fallback.
- **A wait is capped to the door's own budget** minus a margin, derived from
  `config.timeout_seconds()` (`_resize_wait_ceiling()`) — not a second hardcoded number,
  which is exactly how a 30 s ceiling came to sit against an 8 s door and answer
  `504 engine did not answer within 8.0s` over a resize all four ranks had applied.
  On top of that it is capped to a 1.0 s **grace**, the most a wait can ever be worth
  here: it can only catch a rank whose step was already in flight when the ask landed.
- **The cap is reported, never applied silently**: `wait_requested`, `wait_s` (what was
  honoured), `wait_capped` and `wait_cap.{requested,honoured,ceiling,why}`, plus
  `door_budget_seconds` / `wait_ceiling_seconds` so the numbers behind the decision are
  readable from the answer rather than from the source.

```bash
# READ the posture of the process that answers, plus who has answered the last ask
# (the read never relays: the channel carries instructions, and a rank's own count is
# already in its rank file):
curl -s -X POST -d '{"op":"read-threads"}' $B/disk
# RAISE or LOWER it, as often as you like, in one boot. `threads` outside [1, 16],
# or not an integer, is a 400 at the wire and a refusal here, and it changes nothing:
curl -s -X POST -d '{"op":"read-threads","threads":16}' $B/disk
curl -s -X POST -d '{"op":"read-threads","threads":6}' $B/disk
# ASK the fleet to move. The default posts the ask and returns at once (`wait_s: 0`),
# because a wait here holds the loop that dispatches the steps the ranks need to poll on.
# Read the receipts back with the plain `read-threads` call above:
curl -s -X POST -d '{"op":"read-threads","threads":6}' $B/disk
# the same request as a query string works too (POST honours BOTH halves):
curl -s -X POST "$B/disk?op=read-threads&threads=6"
# `wait_s` (integer seconds) is accepted up to the wire's 30 but HONOURED only up to the
# door's budget minus a margin, and to a 1.0 s grace, and the answer says what it did
# (`wait_capped`, `wait_cap.why`). Ask for 15 and you get the cap in the response, not a
# 504 over a resize that worked. `ranks: false` skips the workers and says so;
# `ranks: true` makes an impossible relay fail the call:
curl -s -X POST -d '{"op":"read-threads","threads":6,"wait_s":15}' $B/disk
curl -s -X POST -d '{"op":"read-threads","threads":6,"ranks":false}' $B/disk
# and what each WORKER is actually running is its own rank file, not this answer:
#   readers.configured / readers.threads_source / readers.last_resize
```

A partial answer is normal on a busy box and is reported as such: it names the ranks that
have not answered, and reading `{"op":"read-threads"}` again later finishes the story,
because the receipts on disk are the record and the ask they belong to is named by the ask
file rather than by one process's memory of writing it.

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
| `disk_restore.worker_report()["readers"]` (rank file) | the facts from the only process that actually reads: `configured`, `alive`, `peak_units`, `starved_polls`, `buffer_refusals`, `out_of_order`, the three recency-promotion counters, **the three rates** (`bytes_per_reader_second`, `bytes_per_second_wall`, `parallel_efficiency`), and since 0.2.67 `threads_source` / `last_resize` / `resize_refusals`. The rates were added because the rank file carried the *inputs* to a throughput arithmetic and not the arithmetic: deriving a rate from `bytes_per_file x reads / (read_seconds_avg x threads)` silently assumes every configured thread stayed busy, which is the one thing `parallel_efficiency` exists to falsify |
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

