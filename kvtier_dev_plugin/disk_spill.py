"""Stage 2 of the SSD tier: spill ONE KV slot's bytes to ONE file. Write side only.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
The RAM tier parks evicted GPU KV into pinned host-RAM slots that a worker reads
out of `cpu_kv_caches`. This module takes a slot's bytes out of those SAME tensors
and writes them to disk under a name that IS their identity
(`disk_layout.build_slot_path`:
`<root>/<shard>/kvslot-<key16>-g<group>-<geodigest12>.bin`).

IT ALSO KEEPS THE RESTORE INDEX FROM BEING STARVED. Every file this module commits
is journalled (`disk_index.record_spill`, called AFTER the `os.replace`, so a write
that never even reached its final name is never recorded) and every file its
ceiling plan deletes is journalled too (`record_removal`). Both happen on the
writer thread and both are strictly subordinate to the write: the SPILL is the
product and the journal is an index of it, so a journal that cannot be written costs
a missed restore and never a failed spill. `journal_failures` counts those, and the
spill still reports success.

AND IT DOES NOT FSYNC, ON PURPOSE. `spill_slot` commits with `sync=False`: buffered
writes, no fsync of the file and none of its directory, so the page cache owns
device ordering. These bytes are a CACHE, and the cost of losing one to a crash is
one recompute; `disk_read` classifies a journaled-but-absent file `file_missing`
(and a short one `size_mismatch`/`short_read`) and `disk_restore` prunes it out of
the DMA, so false presence costs a recompute and never wrong bytes. What is NOT
touched is the commit ordering (`O_CREAT|O_EXCL` then `os.replace`), which protects
against a TORN WRITE - a different property, and one this tier still guarantees.
See `disk_layout.write_committed_file`.

AND IT COUNTS GROUP COLLISIONS, because only a number settles how often two
KV-cache groups hash the same token chain. Two groups over one chain share a
`key16` and differ only in the group id - the discriminator the first version of
this name truncated away, which made them name ONE file with different bytes. When
this module is asked to spill a `key16` it has already spilled under a DIFFERENT
group id, it counts a `group_collision`, records both ids and (bounded) whether the
sibling file is actually on disk, and spills BOTH anyway: the names are distinct
now, so the event is a measurement and not a fault. It is readable in the snapshot's
`disk_spill`, in the `/plugins/kvtier/disk` `op:status` payload, and in a rank
file's `disk` section - no restart, no engine change, and nothing armed to see the
rate.

It is the WRITE side of a 5-stage design and THIS module has no read path, no
restore, no lookup and no re-publication of hashes - deliberately, because a hit is
resolved by a lookup in the scheduler's `cached_block_hash_to_block`, and the only
code that publishes into it is scheduler-side store-completion (the fork calls
`_process_store_completion` at `manager.py:1028-1032` on the deployed in-tree-fixed
build, at `manager.py:988-992` at the fork HEAD 10b8e07 this suite's fixture is
generated from; the method itself starts at `manager.py:1009` there and its
`cpu_block_pool._insert_block_hash` calls are at `manager.py:1026-1043`).

AN ARM IS A LATCH (0.2.35), AND IT MUST DIE AT A REBOOT, BY ITS OWN CONTENTS
----------------------------------------------------------------------------
THE ARM IS A LATCH, NOT A TIMER. It is raised by a decision and lowered by a
decision, and no clock lowers it. The operator's ruling, after the live box proved
the point: KV persists until the LRU sheds it under the ceiling - and the ceiling
eviction that sheds it runs from the SPILL WRITER PATH (`_enforce_ceiling` after a
committed write), so an arm that EXPIRED froze shedding as well as growth, which is
the opposite of the product. The timer could not bound the store at any setting
either: 200 files / 60 s at the live 11,468,800 B slot is ~38.2 MB/s per rank against
a 192 GiB ceiling, so ONE rank fills the store in ~90 minutes - a 7,200 s arm permits
~1.34 ceiling-fills and the 1,800 s default ~0.33. Capacity IS bounded, arm-
independently, by the ceiling, the fair share, LRU-by-touch, the `MIN_FREE_BYTES`
statvfs floor and the write-rate window. And the 2026-10-05 livelock the expiry was
partly credited with preventing is dead by six mechanisms that read no arm expiry:
the AND-over-ranks proof with per-rank service facts, the arm read live from ONE
authoritative file, the request/span fuses, the forward-progress fuse, the
contiguous-run offer bound, and a CONFIRMED witness that is unreachable without
proof. A lapse also closed every gate through the SAME `_read_arm_file` read, so it
could never have produced the scheduler-armed/workers-disarmed skew that caused the
loop; a latch cannot produce it either, and the latch's own truth is that
DISARMING STOPS SERVING FROM DISK, not only writing (spill write, `gate()`, the
rank proof's tier arm and `_may_restore_worker` all read the same verdict).

WHAT THE LATCH DOES NOT SURVIVE is a decision nobody made. The arm crosses the
process boundary as one small file in the diagnostics directory
(`ARM_FILE`), because the verb that arms runs in the EngineCore and the spill happens
in a worker. That file used to be checked for `armed` + a wallclock `expires_ts`
alone, so a file that survived a reboot armed the NEW boot for the rest of its life -
and a cold box booting with the disk tier armed exposes the restore path to live
traffic with no operator having asked, which is the precondition of the 2026-10-05
load-failure livelock. What actually killed it was `/tmp` being tmpfs, and "move the
diagnostics directory to disk so the journal survives" would have silently destroyed
that protection while fixing the journal.

So the arm fails closed on its own contents (`_arm_boot_fence`): the file records the
kernel BOOT ID and its epoch with an `epoch_scope`, and a reader refuses a file whose
boot identity provably differs from its own, with a reason that NAMES the mismatch.
The epoch is only evidence when its scope is `env` (an operator-set per-boot id);
with `epoch_scope: "process"` two workers of ONE boot legitimately disagree, and
consulting it there would disarm a healthy tier. A surviving arm file is therefore
harmless wherever it lives, and the journal can be moved to persistent storage
without arming anything (`VLLM_E2_TIER_DISK_JOURNAL_DIR`, `disk_index`).

THE FENCE SEPARATES REBOOTS, NOT ENGINE RESTARTS - AND THAT GAP IS CLOSED BY A
DECISION NOW THAT THE TTL IS GONE. `BOOT_ID_PATH` is one UUID per KERNEL boot, so
four EngineCore boots inside one 2w3d uptime all carry the same boot id, and an arm
file that survives an ENGINE RESTART names a boot the fence cannot tell from this
one. While the arm had a TTL, the TTL is what killed such a file; it has none now,
so the rule is: THE ONLY THING THAT ARMS A PROCESS IS THAT PROCESS'S OWN BOOT
DECISION. `disk_boot.boot_enable()` calls `disarm()` on EVERY outcome except `ARMED`,
and `disarm()` removes the file (`_remove_arm_file`, which already did), so a boot
that did not choose to arm leaves no file behind to arm anybody. Pinned by
`test_the_boot_disarms_an_arm_file_left_by_an_earlier_engine_boot_on_the_same_kernel`.

WHAT CHANGED IN 0.2.30: the ceiling stopped being able to be spent four times. The
store is ONE directory shared by every rank whose geometry digest appears in its file
names, and until now each rank compared the WHOLE store's bytes against the WHOLE
ceiling. Measured on the live box on 2026-10-22: four ranks, 92 files written each, and
2,106 / 2,106 / 1,872 / 1,872 files evicted - 7,956 files, 156 GiB down to 68 GiB, under
a 148 GiB ceiling whose one-pass target was 133 GiB. Nothing miscounted: each rank
measured 156 GiB correctly, each correctly wanted 133 GiB, and all four then acted on
the same directory, so the effective capacity of the store was `ceiling / rank_count`
and the more ranks a boot had, the more of its own cache it destroyed. A rank now sizes
only the bytes it is ACCOUNTABLE for - its own digest's files, plus the dead geometry no
live rank owns (a mismatched file can never be applied, so those bytes are worth zero to
everybody and shed first) - against `ceiling / n_effective`, where `n_effective` is 1
plus the live peer digests this rank's walks already learn (0.2.29). A live sibling's
bytes are then neither counted nor sheddable: the sibling owns them. The fail-safe
direction is stated because it is the first question this change will be asked: with NO
peer learned, `n_effective` is 1 and the share is the WHOLE ceiling - every release's
behaviour, byte for byte - and the old store-wide trigger is kept as a floor so a pass
can only ever start SOONER than before, never later: too timid is not a direction this
exposes. See
`disk_spill._attribute_walk_bytes`, `_share_of_ceiling`, `_ceiling_trigger` and
`ceiling_share` in the section payload.

WHAT CHANGED IN 0.2.25: the capacity ceiling stopped being able to relax itself.
The store reached 334.9 GiB under a 192 GiB ceiling (29,532 files, ~145 GiB grown in
the hour after boot and ~4 GiB/min - 240 GiB/hour - in the burst measured as `/home`
fell 80 GiB -> 60 GiB in five minutes, with 122 GiB of `/home` left) because ONE
bounded walk was being read as a measurement twice over: it stat-ed at most
`MAX_WALK_FILES` files, and it took that sample DEPTH-FIRST - the file counter was
never reset per shard and `truncated`
broke the shard loop - so it measured the first shard directories and stopped. The
byte total handed to `_enforce_ceiling` was then a FLOOR taken from ~2/3 of the key
space (218.7 GiB reported for a 326.2 GiB store), which does not delay the trigger so
much as RESCALE it: the ceiling actually enforced was `ceiling / sample_rate` = 286
GiB, and the divisor grows as the store does. And a file the walk never opened cannot
enter `lru_eviction_plan`, so the third of the key space it never reached became a
sink that took spills forever and shed nothing - eviction running (oldest surviving
file 2.8 h) and structurally blind at the same time. The walk now runs TWO bounded
passes: `_count_store` counts the store's files with readdir only (no `stat`), and the
measuring pass stats a PROPORTIONAL sample - the same fraction of every shard - so the
ceiling decides on `estimated_total_bytes` (the sample scaled to the store; identical
to `used_bytes` for any store inside the caps, byte for byte) and a plan can name a
file in any shard. A truncated plan is PARTIAL by design and converges over the
passes that follow, because enforcement runs after every committed write. What the
measurement was is published, not implied: `ceiling_measurement.basis`,
`sampled_files`/`total_files`, `measured_bytes`, `entries_examined`, `skipped_files`,
`plan_truncated`, `ceiling_checks`. And the more important fix: `estimated_bytes` is
a PER-PROCESS counter of this process's own writes, starting at 0, and the walk that
could correct it only ran once that counter ALONE reached the ceiling - so a rank
booting into a store somebody else filled had to spill 192 GiB of NEW files (17,384
of them, ~48 min at 4 GiB/min) before it was allowed to LOOK at a 335 GiB store with
60 GiB of `/home` left - and the store has since grown to 396 GiB with 61 GiB free. The
filesystem fills long before the ceiling wakes up, and an
honest walk buys nothing when it never runs. The first committed spill of a (root,
ceiling) pair now takes ONE bounded walk and adopts its answer (`_maybe_seed_estimate`,
writer thread only, once per configuration, published as
`ceiling_measurement.seed_walks`). See `disk_spill._walk_store` and README
*The ceiling walk, the sample, and the store that grew past its ceiling*.

WHAT CHANGED IN 0.2.14: a lookup does not have to hit for a restore to happen.
`disk_restore` reads these files back WITHOUT touching that hash index, by extending
the fork's own `get_num_new_matched_tokens` answer with spans whose bytes it puts
into CPU slots the fork then DMAs - so no engine change was needed, and a restored
block helps exactly the request that paid for it instead of becoming matchable by
everyone. "Restore" is therefore live (see `disk_restore.py` and the `restore`
section of `/plugins/kvtier/disk`), and a file written here remains verifiable
(`verify_slot`) and disposable (`Deleting a file is always safe`).

THE HARD PART: THE WORKER HAS THE BYTES, THE SCHEDULER HAS THE HASH
-------------------------------------------------------------------
A content-addressed name needs both. Investigated in the order the brief demands:

1. DOES THE METADATA THAT ALREADY REACHES THE WORKER CARRY A KEY? No.
   `SimpleCPUOffloadMetadata` (`vllm/v1/simple_kv_offload/metadata.py:16-38`) is
   `load_event`, `load_gpu_blocks`, `load_cpu_blocks`, `load_event_to_reqs`,
   `store_event`, `store_gpu_blocks`, `store_cpu_blocks`, `need_flush` - the store
   half is exactly the `(store_gpu_blocks, store_cpu_blocks)` index pair plus an
   event id. `build_connector_meta` (`manager.py:494-556`) builds it from
   `prepare_store_specs`, and in LAZY mode that returns `(gpu_ids, cpu_ids, [],
   None)` (`manager.py:626`) - no hashes at all. No hash, no group id, no per-slot
   key crosses to the worker anywhere.

2. CAN THE SCHEDULER BUILD THE NAME AND THREAD IT THROUGH THAT SAME OBJECT? It
   HAS what it needs, but not a place to put it. At meta-build time the scheduler
   holds each stored block's hash (`manager.py:606` reads `node.block_hash` in lazy
   mode; `_classify_store_candidate` at `manager.py:940-954` refuses an unhashed
   block in eager mode; `make_block_hash_with_group_id` is imported at
   `manager.py:34`), so it could compute `<key16>` per slot. But the metadata
   object has no field for it, and the object is not passed by reference: the
   worker's copy is rebuilt from the wire type
   (`core/sched/output.py:279` declares `kv_connector_metadata`; the decoder is
   `msgspec.convert(result, result_type, ...)` at `v1/serial_utils.py:387`), so a
   field that is not DECLARED on the fork's dataclass is not part of the contract.
   Setting an attribute on the instance from a wrapper is a hack whose behaviour
   depends on which fields msgspec keeps - exactly the "invented channel" the
   brief forbids.

3. SO: THE CORE IS BUILT AGAINST AN INJECTED RESOLVER. (And read THE DIGEST below:
   the digest half of this analysis was WRONG as first written, and the wiring that
   replaced it has each worker compute its own. The wrong reasoning is kept because
   it is the reasoning someone will be tempted by again.) `request_spill(worker,
   slot, key=..., resolver=...)` takes the key from the caller, or from a
   `slot -> key16` callable handed in, and refuses with a counter when neither
   answers. Everything except the wiring is implemented and tested. THE SINGLE
   ENGINE CHANGE THAT UNBLOCKS THE WIRING: add fields to `SimpleCPUOffloadMetadata`
   (metadata.py) - `store_keys: list[str]`, the hex group-qualified block hash per
   stored slot, index-aligned with `store_cpu_blocks` (a slot the engine could not
   hash gets `""`, never a dropped entry); `store_fingerprint: str`, an OPTIONAL
   cross-check digest sent once per store batch (`""` means nothing, and it is
   `""` today); and `load_paths: list[str]` for the restore milestone - then
   populate them in `build_connector_meta` (manager.py), which already holds those
   hashes. THE WIRING FOR IT IS `disk_store.py`: it wraps the worker's
   `bind_connector_metadata`/`get_finished`, reads those fields off the metadata the
   worker already binds, truncates each hash to the layout's 16-hex key, and calls
   `request_spill(worker, slot, key=...)` - the NAME coming from the fingerprint
   this worker built and adopted for itself, not from the broadcast. On an
   unpatched engine the hook refuses and counts (`no_key_for_slot`) and writes
   nothing, which is the point: the same build is correct on both engines.
THE DIGEST: ONE COMPUTING SITE PER FIELD, NOT ONE PER BOOT. The first version of
this contract said "never compute the digest worker-side; the scheduler's
`store_fingerprint` is the only source of truth". That is IMPOSSIBLE, and shipping
it would have caused the exact failure it was written to prevent:
`FINGERPRINT_FIELDS` include `total_bytes_per_block` (the live box measures
11,468,800 B on PP0 and 12,902,400 B on PP1) and `rank` (`PP<n>_TP<m>`), while
there is ONE `SimpleCPUOffloadScheduler` broadcasting ONE `store_fingerprint` to
four ranks. No scheduler-computed digest can equal four different rank digests, so
every rank would write files the others could never resolve. The rule that replaces
it is narrower and actually satisfiable: each FIELD has exactly one computing site.
The rank-local fields (the measured slot width, this rank's id) are the worker's;
the rank-invariant ones are read from the config objects the worker already holds;
and every site calls the same `disk_layout.fingerprint_digest`. The metadata's
`store_fingerprint` survives as an optional CROSS-CHECK only - a non-empty one that
disagrees with this rank's own digest is a counted refusal, and the empty string
the unpatched and freshly-patched engine both send means nothing at all.

SAFETY ENVELOPE (the same posture as `blockdump.py`, which this mirrors)
------------------------------------------------------------------------
  * OFF BY DEFAULT, twice over: `VLLM_E2_TIER_DISK_ROOT` must name a real
    filesystem root (`/tmp`, `/dev/shm` and `/run` are refused by
    `disk_layout.validate_root` - a tmpfs "disk tier" spends the RAM the tier
    exists to relieve) AND the facility must be ARMED. There is no env var that
    arms it: an env var may only ever turn a dev facility OFF, which is the
    `guard_mode` rule, and the same rule `blockdump` follows.
  * NO HOOK IS INSTALLED IN THIS MODULE. It wraps no fork method and touches no
    fork object, so every entry point here returns the fork's value untouched by
    construction, not by promise. The engine hook lives in `disk_store.py`, which
    composes over the existing worker wrappers and keeps this module's shape.
  * NEVER RAISES INTO THE ENGINE. Every public entry point catches `Exception`
    (never `BaseException`) and counts. A disk tier that can crash inference is
    worse than no disk tier.
  * NEVER BLOCKS THE ENGINE THREAD. `request_spill` validates cheaply, then puts
    on a BOUNDED queue with `put_nowait` and returns. Full means a DROP and a
    counter, never growth. The bytes are read and written on this module's own
    single thread, started lazily on the first accepted job - never at import, so
    a fresh boot starts no thread and writes no byte.
  * BOUNDS EVERYWHERE. A region tensor is refused unless `shape[0] * stride(0) *
    element_size() == nbytes` (the same guard `blockdump.read_block_bytes` runs,
    and for the same reason: `ctypes.string_at` past the end of an allocation is a
    SIGSEGV that `except Exception` cannot catch, and it takes serving down). The
    slot index is bounded by the region's own block count, the file's bytes by the
    fingerprint's `total_bytes_per_block`, the queue by `MAX_QUEUE_DEPTH`, the
    write rate by `MAX_FILES_PER_WINDOW`, a sweep/walk by its caps.
  * THE IDENTITY IS NEVER OPTIONAL. A spill is named by a fingerprint whose fields
    all validate (this module's `configure`, fed worker-side by `disk_store` out of
    its own measurements), and no door here accepts a digest from elsewhere. A
    fingerprint whose region bytes do not add up to `total_bytes_per_block` is
    refused: that mismatch means the fingerprint does not describe these bytes,
    which is the one case a content-addressed name cannot survive - and on this box
    it is also the check that stops one PP stage spilling another's bytes.

WHAT IS CAPPED, AND IN WHAT UNIT
--------------------------------
  * `MAX_QUEUE_DEPTH` (jobs)             - the drop line, bounded at the object.
  * `MAX_FILES_PER_WINDOW` / `SPILL_WINDOW_SECONDS` (files / seconds) - the write
    rate, so a spill storm cannot become a disk stall for real traffic.
  * bytes per file = the fingerprint's `total_bytes_per_block`, hard-refused above
    it, and bounded by `disk_layout.MAX_BLOCK_BYTES` (1 GiB).
  * `MIN_FREE_BYTES` (bytes)             - the `statvfs` floor: refuse rather than
    fill the filesystem, reserving the slot we are about to write.
  * `MAX_SHARD_DIRS` (shard DIRECTORIES), `MAX_WALK_FILES` (FILES `stat`-ed) and
    `MAX_WALK_ENTRIES` (directory entries READ, no `stat`) - the three bounds of
    ONE capacity walk, in three currencies, because the walk does two jobs: it
    COUNTS the store's files (readdir only) and MEASURES a bounded SAMPLE of them
    (one `stat` each). When a bound stops it the walk says so AND scales the sample
    by `total_files / sampled_files`, so the ceiling is decided against an estimate
    of the whole store and not against a floor taken from whatever the walk happened
    to reach. A truncated walk that reports only its floor is the defect this bound
    set exists to prevent (see `_enforce_ceiling`), and the bound stays small on
    purpose: this walk runs inside the spill writer's enforcement path, and a walk
    that stats hundreds of thousands of files is its own I/O storm.
  * the capacity ceiling (bytes, configured) - LRU by `st_mtime`, which since 0.2.29
    is a TOUCH clock: a spill writes it and a restore that produced usable bytes bumps
    it (`disk_read.promote_recency`, throttled). `st_atime` is NOT usable because
    `/home` is mounted `relatime` here, so recency is promoted explicitly rather than
    hoped for. Shed order is unreadable bytes, then this boot's own oldest touch, then
    a live PEER RANK's files - see `disk_layout.SHED_ORDER_BY_CLASSIFICATION`. The pass
    that enforces the ceiling is bounded in TIME as well as in files
    (`CEILING_PASS_BUDGET_SECONDS`, `VLLM_E2_TIER_DISK_CEILING_BUDGET_SECONDS`) and
    unlinks on a small worker pool (`VLLM_E2_TIER_DISK_EVICT_THREADS`).

Stdlib only (`ctypes`, `os`, `queue`, `threading`, `time`, `hashlib`, `json`). No
torch and no vllm: the tensors are addressed through the attributes the fork's
object already exposes, read BY NAME, the `worker_diag`/`blockdump` posture. The
on-disk contract - names, the digest, the commit protocol, the eviction order, the
space gate - lives in `disk_layout.py` and is not re-implemented here.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from . import (
    blockdump,
    config,
    disk_index,
    disk_layout,
    disk_read,
    snapshot,
    worker_diag,
)

# WHY WARNING AND NOT INFO, stated here as well as in `disk_store`: a plugin logger is
# not one of the `vllm.*` loggers the engine configures, so an INFO line from this
# module
# never reaches the log the operator greps. The shutdown drain's one line is exactly the
# kind of verdict that must be readable after the fact - it answers "did the backlog
# make
# it before the stop?" from `/tmp/vllm-serve.log`, and a line nobody can see is the same
# as no line.
_logger = logging.getLogger("kvtier_dev_plugin.disk_spill")

# --- the caps, and the unit of every one of them ----------------------------

MAX_QUEUE_DEPTH = 256                # JOBS in flight; past it, a job is DROPPED
# BYTES of snapshotted slots the queue may hold at once. THE SECOND HALF OF THE SAME
# BOUND, and the one the copy-at-completion change made load-bearing: a job used to
# carry a SLOT NUMBER (a few bytes) and read the bytes on the writer thread, so a deep
# queue cost nothing. Now a job carries the bytes themselves - one rank slot,
# 11,468,800 B on the live PP0 and 12,902,400 on PP1 - so the depth cap alone would
# allow 256 x 12.9 MB = 3.3 GB of held host memory at exactly the moment the box is
# already behind. So a job is admitted on BOTH currencies and a refusal is a DROP with
# its own name (`queue_bytes_full`). 256 MiB is about twenty slots: enough to keep the
# writer thread busy through a normal burst (the write-rate window is 200 files per
# 60 s, and the writer drains at SSD speed) and small enough that a stalled disk costs
# dropped spills, never growth.
MAX_QUEUED_BYTES = 256 * 1024 * 1024

# THE SPAN COST, IN BYTES, PER BLOCK CONTRACT. Why the writer publishes a cost model at
# all, when `disk_restore` is the one that walks spans: an align-mode/boundary-state
# group costs ONE file per RUN while a positional group costs one file per SPAN, so a
# reader who counts files concludes a GDN group is "cheap to keep" - true per hit,
# and false as a plan (it is cheap because the run's other spans are nulls, i.e.
# it kept nothing at all for them).
# Publishing both numbers side by side, in the currency the ceiling is configured in, is
# cheaper than being asked again.
SPAN_COST_MEANING = {
    "positional_bytes_per_span": "ONE file of this width for EVERY block of a run: a"
                                 " positionally stable group names every span, so a run"
                                 " of n spans costs n files",
    "boundary_state_bytes_per_span": "ONE file of this width for the LAST block of"
                                     " a run ONLY: an align-mode group hands off its"
                                     " recurrent state as a boundary-state block, so a"
                                     " hit of n spans costs n-1 nulls and this one"
                                     " file. Cheap per hit; NOT a cheaper capacity"
                                     " plan, because the n-1 spans it did not buy are"
                                     " spans it did not keep",
    "measured_from": "this rank's fingerprint total_bytes_per_block - the width every"
                     " spill file has here, and the width disk_read reads back",
    "unit": "bytes",
}
SPILL_WINDOW_SECONDS = 60.0          # SECONDS in the write-rate window
MAX_FILES_PER_WINDOW = 200           # FILES written per window

# THE FLUSH BUDGET (0.2.66). A flush is a bounded poll of THIS module's own queue, so
# its cost is one `qsize()` per interval and the only thing that can make it slow is
# the budget itself. `SHUTDOWN_FLUSH_SECONDS_DEFAULT` is the wait the stop path may
# spend getting the queued bytes onto NVMe before it hands the process back to the
# fork's teardown; `FLUSH_BUDGET_SECONDS_MAX` is the ceiling on ANY flush, whoever asked
# (the verb's caller or the environment), because a wait on the stop path is the one
# place where a typo costs a suspend rather than a byte.
SHUTDOWN_FLUSH_SECONDS_DEFAULT = 15.0
FLUSH_BUDGET_SECONDS_MAX = 30.0
FLUSH_POLL_SECONDS = 0.02            # SECONDS between looks at the queue depth
SHUTDOWN_FLUSH_ATTR = "kvtier_dev_shutdown_flush"    # the wrapper's own marker

# WHY THE DRAIN IS HOOKED TO THE CONNECTOR AND NOT TO `atexit`, read out of the live
# fork rather than assumed: a worker ends its ORDERLY shutdown in
# `v1/executor/multiproc_executor.py:_exit_worker_process` with `os._exit(exit_code)`,
# and `os._exit` skips `atexit`, every pending `finally` and interpreter finalization.
# A drain registered only with `atexit` would therefore never run in the process that
# OWNS the spill queue (the queue is built in the worker, by `request_spill`, which runs
# inside the worker's `get_finished`) - silently, on exactly the path it was written
# for.
# The worker's teardown does, however, call the connector: `Worker.shutdown()`
# (`v1/worker/gpu_worker.py:1699`) -> `ensure_kv_transfer_shutdown()`
# (`distributed/kv_transfer/kv_transfer_state.py:97`) ->
# `_KV_CONNECTOR_AGENT.shutdown()`,
# and that runs BEFORE the `os._exit`. So the connector's own `shutdown` is the seam
# that exists in the right process at the right moment, and `atexit` stays as the
# secondary seam for the processes that really do exit normally (an in-process boot).
SHUTDOWN_FLUSH_NOTE = (
    "the drain runs on the connector's own shutdown (the worker's teardown calls it "
    "before os._exit) and on atexit secondarily, because os._exit skips atexit and "
    "the spill queue lives in the worker process")
MIN_FREE_BYTES = 1024 * 1024 * 1024  # BYTES the filesystem must keep free
MAX_REGIONS = 64                     # REGION TENSORS making up one slot
MAX_STRING_CHARS = 160
MAX_SHARD_DIRS = 288                 # shard directories a capacity walk will open
MAX_WALK_FILES = 20_000              # FILES a capacity walk will stat
# Directory ENTRIES a capacity walk may read, PER PASS (the walk has two passes, so
# a whole walk reads at most twice this many entries). Five times the `stat` cap,
# because reading an entry is readdir - which the walk pays for anyway, and which
# hands back the `d_type` byte that answers `is_file()` without a syscall - while
# stat-ing one is a syscall. The bound is deliberately NOT expressed in files: it
# bounds the WORK, and a store stuffed with entries this tier will not measure is
# exactly the store whose walk would otherwise run to the end of the directory (the
# `stat` cap counts MEASURED files only, so before this bound the iteration itself
# was unbounded). Five times the stat cap also makes the count EXACT for every store
# this box can physically hold: 100,000 files at the widest slot measured here
# (12.9 MB) is 1.2 TiB, and past that the count extrapolates and says so.
MAX_WALK_ENTRIES = 100_000           # directory entries a capacity walk will read
EVICT_TARGET_FRACTION = 0.9          # FRACTION of the ceiling eviction aims under

# THE CEILING PASS'S TIME BUDGET (0.2.29). One enforcement pass walks the store and
# then unlinks, and it runs on the SPILL WRITER THREAD - the only thing that drains
# `MAX_QUEUE_DEPTH`/`MAX_QUEUED_BYTES`. Its worst case was unbounded in TIME: two
# passes of `MAX_WALK_ENTRIES` (100,000) directory entries, `MAX_WALK_FILES` (20,000)
# `stat` calls and up to `MAX_EVICTION_PLAN` (4,096) unlinks, i.e. seconds, during
# which the queue backs up and spills are DROPPED - on the one path (a full disk)
# where dropping spills is most expensive. So the pass now gets a deadline and stops
# early when it expires, reporting itself partial. A partial pass is the shape this
# module already relies on for a store bigger than its caps: enforcement runs after
# EVERY committed write, so the next spill runs the next pass against a store whose
# directory slots the last pass just freed, and the passes converge. Truncation only
# ever SHRINKS the destructive set, and it happens between whole per-directory
# `execute_unlink_plan` calls - never inside one, so no plan is half-applied.
# The overshoot is bounded: the first directory's unlinks always run (at most
# `MAX_UNLINK_PLAN` of them), because a pass that released nothing has to be
# re-run by something else, and only a committed spill re-runs it.
CEILING_PASS_BUDGET_SECONDS = 1.5    # SECONDS one pass may spend, by default
CEILING_BUDGET_SECONDS_MAX = 60.0    # the clamp: an env typo must not restore an
                                     # unbounded pass while claiming to bound it

# THE LIVE PEER RANKS (0.2.29, re-pointed 0.2.45). Four ranks share ONE spill root and
# each rank's evictor sees the other three's committed files as names carrying a
# digest that is not its own. `disk_layout.classify_name` can call those files
# `peer_rank` (live data, shed LAST) instead of `stale_geometry` (pure capacity, shed
# FIRST) - but only if it is TOLD the sibling digests, and nothing in the engine hands
# them over. They are therefore LEARNED, and since 0.2.45 they are learned from TWO
# witnesses:
#
# 1. WRITE ACTIVITY (this map): a foreign digest with committed files written inside
#    `PEER_DIGEST_FRESH_SECONDS` is a rank that is WRITING RIGHT NOW.
# 2. THE RANK FILE (`_peer_identity_view`): a live worker publishes its own current
#    geometry digest in `rank-<rank>.json`, which is already this tier's liveness proof.
#
# WITNESS 1 ALONE WAS THE DEFECT THIS RELEASE REMOVES. A digest is dropped from this map
# when it goes quiet (`PEER_DIGEST_FRESH_SECONDS` with no write), and a digest outside a
# map that is still CONFIRMED by some other sibling was `ATTRIBUTION_DEAD` - "positively
# nobody's" - which `SHED_ORDER_BY_CLASSIFICATION` puts at bucket 0, the first thing
# every busy rank unlinked. So a rank that simply had no request for two minutes had its
# live, restorable KV treated as free capacity, and 5,910 removals in one boot on the
# live box came from exactly this path. The write-activity witness stays (it is a PROOF
# of liveness, and it is what makes the protection work on a box with no diagnostics
# directory) but it is no longer the only one, and its silence is no longer read as
# death.
#
# WHAT STILL DECIDES `dead`, and why that is not an age rule over cached bytes: a
# digest NO live rank publishes and NO live rank is writing. That is what a
# configuration change
# leaves behind (the operator moved `KV_OFFLOAD_SIZE` 64 -> 8 GiB on 2026-10-07: slot
# width moved, so the digest moved, and every pre-change file became
# unservable-by-anybody), and what a rank whose WORKER has gone leaves behind. Both are
# statements about IDENTITY and about a PROCESS's liveness, never about how long a byte
# has been sitting on the disk.
#
# FAIL-SAFE DIRECTION, and it is deliberate: if no digest qualifies, the pass behaves
# exactly as it did before 0.2.29 (foreign first) unless the identity view says it
# cannot name every live rank - in which case foreign committed bytes are
# `unattributable`, are withheld, and the ceiling is allowed to come up SHORT. The worst
# case of being too conservative is a spill refusal and a filesystem an operator can
# see; the worst case of being too free is a `file_missing` hole in a serving path. A
# digest is only believed with `PEER_DIGEST_MIN_SIGHTINGS` fresh files, so one stray
# recent file from an old boot cannot protect a whole digest.
PEER_DIGEST_FRESH_SECONDS = 120.0    # SECONDS a foreign digest counts as "writing now"
PEER_DIGEST_MIN_SIGHTINGS = 2        # FRESH committed files before a digest is believed
MAX_PEER_DIGESTS = 32                # DIGESTS remembered (a 4-rank fleet needs 3)

#
# THE SECOND WITNESS: A LIVE WORKER'S OWN PUBLISHED IDENTITY (0.2.45)
# -----------------------------------------------------------------------------
# `worker_diag` writes one small JSON file per rank, atomically, into the diagnostics
# directory (`VLLM_E2_TIER_DIAG`, the same directory the ARM FILE lives in), and
# `disk_store.worker_report()` publishes that rank's CURRENT geometry digest inside
# it. That file is already this tier's proof that a fitted worker is alive -
# `disk_restore` refuses every disk offer without it - so reading the digest out of it
# costs nothing new and asks a question that is already being answered somewhere.
#
# WHY THIS IS THE WITNESS THAT FIXES THE QUIET RANK. Write activity asks "is this rank
# SPILLING?", which a healthy rank with a warm prefix stops doing indefinitely; a rank
# file asks "is this rank's WORKER ALIVE?", which a healthy rank answers for as long
# as it lives. A rank whose worker has gone stops rewriting its file, so its digest
# stops being protected and its orphaned bytes revert to `stale_geometry` and are
# reclaimed under the quota by whoever can prove ownership - which is how a genuinely-
# gone rank's bytes are eventually freed WITHOUT a timer over cached bytes. (0.2.45
# also makes the file a real heartbeat - `worker_diag` now rewrites it on its own
# thread, so an idle engine's proof stays true instead of expiring. That is what makes
# this witness usable at all.)
# THREE-VALUED BY DESIGN. `known` means "I can name every live rank's identity right
# now", and it is the flag `attribute_name` needs to call a foreign digest dead. A
# fresh rank file WITHOUT a readable digest (a mixed-version fleet, a rank whose
# report raised) makes it FALSE, because "not in my list" is not evidence of absence.
# A rank that cannot read the directory at all falls back to the write-activity
# reading, which is exactly the pre-0.2.45 behaviour: this change never makes the
# stale-geometry reclaim path harder to reach than it was, it only stops it from being
# reached by silence alone.
PEER_RANK_FILE_FRESH_SECONDS = 900.0
# SECONDS a rank file may be old and still vouch for its writer's IDENTITY. Deliberately
# the same number as `disk_restore.RANK_FILE_FRESH_SECONDS`: one file, one liveness
# window, two readers. A bound over a PROCESS's liveness, not a lifetime over cached
# bytes - and pinned equal by `tests/test_quiet_rank_bytes_are_not_dead_bytes.py`, so a
# future edit to one cannot silently drift from the other.
MAX_PEER_RANK_FILES = 8              # RANK FILES probed (disk_restore's own bound)
MAX_PEER_RANK_FILE_BYTES = 256 * 1024  # BYTES one rank file may reach
PEER_IDENTITY_CACHE_SECONDS = 1.0    # SECONDS one identity read stands for
MAX_PEER_DIGEST_CHARS = 64           # CHARS of a published digest accepted from a file
# UPPER BOUND on a peer's published block width, read from ITS OWN rank file
# (`arena.bytes_per_block`) and used to weight a shared ceiling. A block wider than
# this is a corrupt or hostile file, not a wide model: refusing it costs the block-
# equal share and falls back to the equal-byte one, which is the pre-0.2.58 shape.
MAX_PEER_WIDTH_BYTES = 1 << 30       # BYTES (1 GiB) one rank's block may be

# THE FAIR SHARE: ONE CEILING IS NOT FOUR CEILINGS (0.2.30)
# -----------------------------------------------------------------------------
# THE DEFECT, MEASURED ON THE LIVE BOX (the night of 2026-10-22). Four ranks share ONE
# spill root, and every one of them ran a ceiling pass whose decision number was the
# WHOLE STORE'S bytes compared against the WHOLE CEILING. Measured, per rank, from the
# rank files: 92 files written each and 2,106 / 2,106 / 1,872 / 1,872 files evicted -
# 7,956 files, 156 GiB -> 68 GiB, against a 148 GiB ceiling with
# `EVICT_TARGET_FRACTION = 0.9`, i.e. ONE pass should have shed ~23 GiB and four passes
# shed ~4 x 23 GiB. Nothing miscounted: each rank correctly measured a 156 GiB store,
# each correctly wanted 133 GiB, and all four then acted on the same directory. The
# effective capacity of a shared store enforced this way is `ceiling / rank_count`, and
# the more ranks a boot has, the more of its own cache it destroys.
#
# THE RULE NOW. A rank sizes only the bytes it is ACCOUNTABLE for - its own geometry
# digest's files, plus the files that belong to nobody live (dead geometry from a
# previous boot, which no rank can read and every rank may shed) - against
# `ceiling / n_effective`, where `n_effective = 1 + (live peer digests learned)`. The
# CONVERGENCE is arithmetic, not tuning: if every rank holds `own + dead/n <= 0.9 * C/N`
# then the store holds `sum(own) + dead <= 0.9 * C`. Tonight's four ranks each get 33.3
# GiB of budget instead of 133 GiB, and the store lands on the target instead of a
# quarter past it.
#     WHY THE DEAD BYTES ARE COUNTED IN FULL AGAINST EVERY RANK (rather than as a 1/N
# share): a dead file is worth zero to every rank - the reader enforces an exact byte
# size and digest match, so a mismatched file can NEVER be applied - and counting it
# fully makes every rank shed the worthless bytes first, which is the eager direction.
# The cost is that N ranks can over-shed a large dead population, and every byte of
# that over-shed is a byte nobody could ever have read back. Counting it not at all
# would be the dangerous choice: a store full of old geometry would stop triggering the
# ceiling, which is 0.2.25's bug wearing a new hat.
#     WHY A LIVE PEER'S BYTES ARE NOT COUNTED AND NOT SHEDDABLE: a live sibling is
# accounting for exactly those bytes under its OWN share. Counting them here would
# charge them twice and shed them twice; keeping them in the planner's entry list would
# let a scaled estimate (which is deliberately conservative) reach into them. A rank
# therefore cannot now delete a live peer's file at all - the peer's entries are removed
# from the plan (`peer_entries_excluded`) - and if a sibling misbehaves and overruns its
# share, the `MIN_FREE_BYTES` statvfs floor is what protects the filesystem, which is
# the invariant the brief refuses to let regress.
#     THE FAIL-SAFE DIRECTION, stated because it is the question the change will be
# asked: with NO peer learned, `n_effective` is 1 and the share is the WHOLE ceiling -
# i.e. every release's behaviour, byte for byte, never a timid hole at boot. The trigger
# keeps the old store-wide clause as a floor (`estimated_bytes >= ceiling` still starts
# a pass) so the new rule can only ever make a pass EARLIER, never later: capacity
# safety cannot regress by construction. The worst case that permits is a 4-rank boot
# whose ranks are mutually invisible - each holding up to 0.9 x ceiling of bytes it
# cannot attribute to a live sibling, so the store up to ~3.6 x ceiling - bounded in
# practice by the fact that the very pass that decides runs a walk that learns its peers
# (so the window is one spill wide), and bounded hard by the `no_free_space` refusal.
SHARE_SOURCE_PEERS = "peers_learned"       # n_effective = 1 + live peers
SHARE_SOURCE_NO_PEERS = "no_live_peer_seen"  # n_effective = 1: today's behaviour
SHARE_SOURCE_NO_DIGEST = "no_geometry_digest"  # cannot attribute: whole ceiling

# HOW A SHARE WAS SIZED (0.2.58). A share equal in BYTES is not a share equal in BLOCKS
# when the ranks' block widths differ - and on the live box they do: PP0 ranks spill
# 11,468,800-byte blocks and PP1 ranks 12,902,400-byte ones, so one byte budget handed
# the PP0 ranks 4,493 blocks against the PP1 ranks' 3,994. A restore needs a block on
# EVERY rank, so those 499 surplus PP0 blocks are disk space holding something that can
# never be served. `block_equal` says the width-weighted arithmetic ran; `equal_bytes`
# says a width was missing (no fingerprint yet, or a live sibling whose rank file
# publishes none) and the old equal-byte split was used instead of a guessed width.
SHARE_BASIS_BLOCK_EQUAL = "block_equal"      # shares sized by width, blocks equal
SHARE_BASIS_EQUAL_BYTES = "equal_bytes"      # pre-0.2.58 arithmetic: bytes equal

# The group-collision witness: which group ids this process has seen spill each
# `key16`. Three caps, because this is a witness and not a database. The map is
# what makes a collision observable in O(1) per spill with no directory scan on the
# write path; each cap bounds what the witness itself costs.
MAX_TRACKED_DIGESTS = 4_096          # key16s remembered (oldest dropped first)
MAX_GROUPS_PER_DIGEST = 8            # group ids remembered for ONE key16
MAX_COLLISION_SAMPLES = 8            # collision EVENTS kept for a payload
MAX_COLLISION_SCAN = 4_096           # directory entries a sibling probe will read

# The arm. In-process here (a worker's truth is the arm FILE, the `blockdump`
# one-way-per-process problem again), with a bounded, stat-first read so an armed
# worker's gate costs nothing. SINCE 0.2.35 THE ARM IS A LATCH: there is no TTL
# constant left to validate, because nothing about the arm is measured in time -
# `ARM_DEFAULT_SECONDS` / `ARM_MIN_SECONDS` / `ARM_MAX_SECONDS` are deleted, and
# `disk_boot`'s renewal path went with them (see the module docstring for the
# ruling and the arithmetic).
ARM_FILE = "disk-spill-arm.json"
MAX_ARM_FILE_BYTES = 4096
ARM_FILE_POLL_SECONDS = 1.0          # SECONDS a cached file verdict is trusted
# THE BOOT FENCE. `/proc/sys/kernel/random/boot_id` is one UUID per kernel boot, the
# same string in every process on the box, so it is the boot identity an ARM FILE can
# be checked against without the operator configuring anything. Bounded read, never
# a raise; an unreadable one is an absence of knowledge, not a disarm.
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
MAX_BOOT_ID_CHARS = 64
MAX_EPOCH_CHARS = 128

WRITER_TAG = "kvtier-dev-plugin/disk-spill-1"
PROTOCOL_VERSION = 1

# EVERY COUNTER HERE IS ONE PROCESS'S, AND A READER MUST BE TOLD (0.2.30).
# `disk_spill`'s state is a module global, and the writer thread and the ceiling pass
# run in the process that COMMITS writes - a WORKER, four of them on a PP2xTP2 boot.
# The process that answers `/snapshot` is a different process, which runs no writer
# thread, so its `spilled`, `evictions` and `bytes_written` are zero BY CONSTRUCTION.
# On 2026-10-22 an operator read `disk_spill.spilled: 0` while four ranks had written
# 92 files each and evicted 7,956 between them, and was misled by it twice. The fix is
# not a shared counter (no new IPC and no shared memory on a serving box); it is a RANK
# ROLLUP beside these numbers, plus this label on them, so a zero that is true of one
# process cannot be read as a zero that is true of the tier.
PER_PROCESS_SCOPE_NOTE = (
    "PER-PROCESS: every counter in this block is THIS process's own `disk_spill` "
    "state. The writer thread and the ceiling pass run in the WORKER processes, so a "
    "zero here says 'this process spilled nothing', never 'the tier spilled nothing' "
    "- on the API or EngineCore process these are zero by construction. For the "
    "fleet's numbers read the rank rollup beside this block (`disk_fill.per_rank` / "
    "`disk_fill.totals` in /snapshot, `spill_rollup` in `op:status`), which sums each "
    "rank's own counters from the rank files and names which ranks it summed and how "
    "stale each one was.")
PER_PROCESS_COUNTERS = ("requests", "accepted", "spilled", "duplicates",
                        "bytes_written", "write_seconds", "read_seconds", "drops",
                        "evictions", "evicted_bytes", "estimated_bytes", "errors",
                        "queue_depth", "queue_depth_peak", "queue_bytes",
                        "queue_bytes_peak", "window_files", "thread_started",
                        "group_collisions", "journal_records", "journal_removals",
                        "journal_failures", "journal_write_errors", "write_errors")


def _per_process_units(units: dict) -> dict:
    """The unit map, with the PER-PROCESS label written into the units themselves.

    A reader that only looks at `units` (a dashboard's habit) still cannot mistake
    `spilled: 0` for "nothing spilled": the dimension it is labelled with says whose
    zero it is.
    """
    out = dict(units)
    for name in PER_PROCESS_COUNTERS:
        if name in out:
            out[name] = f"{out[name]} (this process only)"
    return out

# Refusal reasons are a closed vocabulary: a counter per reason, so "nothing
# happened" is always attributable rather than inferred.
REFUSAL_REASONS = ("not_configured", "disarmed", "no_key_for_slot",
                   "fingerprint_missing", "fingerprint_invalid", "bad_digest",
                   "no_region_tensors", "region_not_contiguous",
                   "slot_out_of_range", "region_width_mismatch",
                   "read_failed", "no_free_space", "over_rate_window",
                   "queue_full", "write_failed", "bad_key", "bad_call",
                   # A job is admitted on TWO currencies now - see MAX_QUEUED_BYTES:
                   # the depth cap (`queue_full`) and this byte cap. A reader must be
                   # able to tell "the queue is 256 deep" from "we would have had to
                   # slot's bytes and chose not to".
                   "queue_bytes_full",
                   # The snapshot could not be taken: a slot the region map cannot
                   # answer, or a region narrower than the slot. Since the copy moved to
                   # store completion this is reported HERE, where the slot still
                   # exists, and not by the writer thread half a second later as
                   # `write_failed`.
                   "snapshot_failed")

_state: dict = {
    # configuration and the arm
    "configured": False,
    "root": None,
    "ceiling_bytes": None,
    "armed": False,
    "arm_file_verdict": None,
    "arm_file_checked_mono": None,
    # How often an arm file was refused for belonging to another boot: the number
    # that says "the boot fence is working", and - if it ever climbs with no reboot
    # - "the boot identity is not what I thought it was".
    "arm_file_boot_refusals": 0,
    # the fingerprint this process writes under (boot-constant)
    "fingerprint": None,
    "geometry_digest": None,
    # the queue and its thread
    "thread_started": 0,
    "thread_alive": False,
    "queue_depth": 0,
    "queue_depth_peak": 0,
    # THE FLUSH TALLIES (0.2.66). A flush is a bounded WAIT on this queue, added for
    # mirror mode: a stop that happens while the queue still holds work loses exactly
    # the bytes the change exists to protect. `flush_calls` counts the calls this
    # process answered, `flush_drained`/`flush_timed_out` split them by outcome, and
    # `flush_bytes_written` is the bytes that landed *inside those waits* - never the
    # boot total, because a caller that asked "did MY backlog arrive" must be able to
    # read that off one answer. `flush_errors` is the fail-open tally: a flush that
    # raised, which must be visible and must never be a fault on the stop path.
    "flush_calls": 0,
    "flush_drained": 0,
    "flush_timed_out": 0,
    "flush_bytes_written": 0,
    "flush_errors": 0,
    "flush_last": None,
    # THE SHUTDOWN DRAIN'S verdict, for the boot-line/verb readers.
    "shutdown_flush": None,
    # Bytes of SNAPSHOTTED slots held by queued jobs. Load-bearing since the copy moved
    # to store completion: the queue's memory is now bytes x depth, so admission is
    # checked in both currencies (see MAX_QUEUED_BYTES).
    "queue_bytes": 0,
    "queue_bytes_peak": 0,
    "window_start_mono": None,
    "window_files": 0,
    # counters
    "requests": 0,
    "accepted": 0,
    "spilled": 0,
    "duplicates": 0,
    "bytes_written": 0,
    "write_seconds": 0.0,
    "read_seconds": 0.0,
    "drops": 0,
    "refused": {reason: 0 for reason in REFUSAL_REASONS},
    "evictions": 0,
    "evicted_bytes": 0,
    "estimated_bytes": 0,
    "walk_truncated": False,
    # THE HONESTY OF THE MEASUREMENT THE CEILING WAS DECIDED ON. `estimated_bytes`
    # alone cannot distinguish "the store IS this big" from "I stat-ed 20,000 files
    # and multiplied by 1.48". On the production box a 192 GiB ceiling faced a
    # 326 GiB store while the walk reported 218.7 GiB - a floor taken from a
    # depth-first sample of the FIRST shards - which rescaled the ceiling this module
    # enforced to 192/0.671 = 286 GiB, and left the shards it never opened unable to
    # be evicted at all. The distinction is the whole point of the witness.
    # `walk_*` describes the LAST walk the ceiling logic ran; `ceiling_checks` counts
    # how many enforcement passes it took.
    "walk_sampled_files": 0,
    "walk_total_files": 0,
    "walk_measured_bytes": 0,
    "walk_entries_examined": 0,
    "walk_skipped_files": 0,
    "walk_count_truncated": False,
    "walk_basis": None,
    "plan_truncated": False,
    "ceiling_checks": 0,
    # THE PASS'S OWN CLOCK (0.2.29). `ceiling_budget_seconds` is the budget in force
    # (env or default), `ceiling_pass_seconds` how long the LAST pass actually took,
    # `ceiling_deadline_hit` whether it ran out of time. The three together answer the
    # only question that matters during a fill test: "is the ceiling keeping up, or is
    # every pass being cut off before it finishes?" A `plan_truncated` that is True
    # with `ceiling_deadline_hit` False is the plan cap; with it True, the deadline.
    "ceiling_budget_seconds": None,
    "ceiling_pass_seconds": 0.0,
    "ceiling_deadline_hit": False,
    # ...and the boot total behind the last-pass bool: `ceiling_deadline_hits` is how
    # many passes this process has ever had to cut off, which is the number that says
    # "the budget is too small for this store" rather than "one pass was unlucky".
    "ceiling_deadline_hits": 0,
    "walk_deadline_hit": False,
    # THE LIVE PEER RANKS (0.2.29), as observed by the walks this module runs.
    # `peer_digest_sightings` counts every foreign-digest committed file the measuring
    # pass saw; `peer_digest_learned` counts digests that ever qualified as a live
    # sibling's; `peer_digests_live` is how many were believed live at the LAST
    # ceiling decision; `peer_digests_unconfirmed` counts the ceiling decisions taken
    # with NONE (today's behaviour, and the number that says "this rank could not tell
    # its peers' live data from dead capacity"); `peer_files_planned` how many files
    # the last plan nevertheless chose to shed from a live sibling; `peer_digests_
    # dropped` how many were forgotten because the map is bounded.
    "peer_digest_sightings": 0,
    "peer_digest_learned": 0,
    "peer_digests_live": 0,
    "peer_digests_unconfirmed": 0,
    "peer_digests_dropped": 0,
    "peer_files_planned": 0,
    # THE FAIL-CLOSED EVICTION GATE (0.2.34). An UNATTRIBUTABLE file - a committed
    # file under a digest that is not this boot's, seen while no peer digest is
    # confirmed - is never a candidate for unlinking, so a pass can come up SHORT of
    # its target. Both halves must be readable: what was withheld (`unattributable_
    # entries_excluded`, a boot total, plus the last pass's reading) and what that
    # cost (`ceiling_passes_short` / `ceiling_plan_short*`). The alternative was the
    # cross-rank deletion incident: 15,848 removal lines of a sibling's block size,
    # and a `file_missing` hole 19,200 tokens deep, found in the journal hours later.
    # `peer_files_removed` is Fix 3: files this rank unlinked whose name carries a
    # digest that is NOT its own - legitimate only when the peer was known and the
    # plan had reached the last bucket, and the number that says so at a glance.
    "unattributable_entries_last_pass": 0,
    "unattributable_bytes_last_pass": 0,
    "unattributable_entries_excluded": 0,
    # THE GRACE AND THE PENDING CLAIM (0.2.62). The grace's pair mirrors the gate's
    # above: entries and bytes the PLANNER withheld because the touch clock says they
    # were used inside the window, plus the boot total. The two refusal counters are
    # the EXECUTOR's: files a plan named that the executor refused to delete because
    # they were in the grace, or because this process has a read in flight for them.
    # A boot total with no last-pass twin would make a single unlucky pass
    # indistinguishable from a rank that is permanently short of sheddable files.
    "grace_entries_last_pass": 0,
    "grace_bytes_last_pass": 0,
    "grace_entries_excluded": 0,
    "grace_unlink_refused": 0,
    "grace_unlink_refused_last_pass": 0,
    "claim_unlink_refused": 0,
    "claim_unlink_refused_last_pass": 0,
    "claim_paths_last_pass": 0,
    "claim_set_truncated": False,
    "ceiling_passes_short": 0,
    "ceiling_plan_short": False,
    "ceiling_plan_short_bytes": 0,
    "ceiling_plan_short_reason": None,
    "peer_files_removed": 0,
    "peer_files_removed_bytes": 0,
    "peer_files_removed_last_pass": 0,
    "peer_files_removed_known_peer": 0,
    # WHICH JUSTIFICATION PAID FOR EACH REMOVAL (0.2.45). The operator's question,
    # read off one line: "did I delete 76 GB of a LIVE SIBLING's bytes, or 76 GB
    # of bytes a CONFIGURATION CHANGE orphaned?" Those two used to be the same
    # number. Every removal is still counted in `peer_files_removed`; the two
    # below say which classification licensed it, and they are disjoint, so
    # `peer_files_removed` minus the two of them is the residue a reader has to
    # explain (`peer_files_removed_known_peer` is the subset of
    # `peer_files_removed`, never a third bucket).
    "stale_geometry_removed": 0,
    "stale_geometry_removed_bytes": 0,
    "stale_geometry_removed_last_pass": 0,
    "stale_key_shape_removed": 0,
    "stale_key_shape_removed_bytes": 0,
    # A LIVE rank's file, unlinked, by a capacity pass. Since 0.2.45 the planner
    # cannot put one in a plan and the executor refuses it, so these two are the
    # alarm: they are the incident counter, spelled separately from the cleanup
    # counter above.
    "peer_files_removed_live": 0,
    "peer_files_removed_live_bytes": 0,
    # THE IDENTITY VIEW (0.2.45): what this rank could SEE about who is alive, at
    # its last `_peer_identity_view()` read. `peer_liveness_source` names the
    # witness that decided (`rank_files`, or `write_activity` when no rank file
    # was readable here at all) and `peer_identity_known` is the flag that lets a
    # foreign digest be called dead.
    "peer_liveness_source": None,
    "peer_identity_reads": 0,
    "peer_identity_available": False,
    "peer_identity_known": False,
    "peer_identity_digests": 0,
    "peer_identity_own_published": False,
    "peer_identity_files_unnamed": 0,
    "peer_identity_files_stale": 0,
    "peer_identity_files_unreadable": 0,
    "peer_identity_reason": None,
    # THE FAIR SHARE (0.2.30): WHOSE BYTES THIS RANK IS ACCOUNTABLE FOR. Split out of
    # the one store-wide `estimated_bytes`, because that single number is what the four
    # ranks each read as "156 GiB, over my 148 GiB ceiling" and each acted on.
    # `own_bytes` is this geometry digest's files, `foreign_dead_bytes` belongs to no
    # geometry, shed first by everyone), `peer_live_bytes` is a live sibling's and is
    # neither counted nor shed. `accountable = own + dead` is the number the ceiling is
    # compared against, and `share_bytes` is its budget.
    "estimated_own_bytes": 0,
    "estimated_foreign_dead_bytes": 0,
    "estimated_peer_live_bytes": 0,
    "estimated_unattributed_bytes": 0,
    # False until a walk has attributed the store's bytes by ownership. While False the
    # rank cannot know its own share of a store it did not write, so it decides on
    # `estimated_bytes` against the WHOLE ceiling - every earlier release's behaviour,
    # which is the eager direction and the fail-safe one.
    "bytes_attributed": False,
    "share_bytes": None,
    # HOW MANY BLOCKS `share_bytes` can actually hold on THIS rank, and the width that
    # derived it (0.2.58). A restore needs one block on EVERY rank, so a share that is
    # equal in BYTES across ranks of unequal width is a share that is unequal in the
    # only currency a restore spends. `share_blocks` is None when no width is known.
    "share_blocks": None,
    "share_own_width": None,
    # HOW MANY of the live peers this rank had a published width for when it sized the
    # share. `share_divisor - 1` is the number it needed: fewer, and the sum of widths
    # is a guess, so the arithmetic falls back to equal bytes and this field is the
    # reason an operator can read instead of infer.
    "share_width_peers": None,
    # `block_equal` when the shares were weighted by width, `equal_bytes` when a width
    # was missing and the pre-0.2.58 arithmetic was used - the honest way to say "this
    # rank's budget is not known to be block-comparable to its siblings'".
    "share_basis": SHARE_BASIS_EQUAL_BYTES,
    "share_divisor": None,
    "share_source": None,
    "share_target_bytes": None,
    # THE "ANOTHER RANK GOT THERE FIRST" WITNESS. This runs on the writer thread, so it
    # is a counter and not a log line: "four ranks, one pass did the work, three
    # declined" is the signature of the fair share working, and it has to be readable
    # live during a fill test rather than reconstructed from four files by hand.
    "ceiling_passes_declined": 0,
    "ceiling_declined_accountable_bytes": None,
    "ceiling_declined_target_bytes": None,
    # Entries the pass REMOVED from the destructive set because a live peer owns them:
    # the counter that says "a rank had work to do and deliberately did not do it".
    "peer_entries_excluded": 0,
    # A pass that found itself over its target and had nothing it was allowed to shed.
    # Distinct from `ceiling_passes_declined`, which means "already under".
    "ceiling_passes_nothing_sheddable": 0,
    # THE UNLINK POOL (0.2.29). The count is the shape of the mechanism; the rest are
    # its evidence. `evict_threads_requested` is what the environment asked for and
    # `evict_threads_configured` what the pool built (they differ across a restart, and
    # the pool is deliberately built once), `evict_threads_alive` how many of its worker
    # threads exist right now, `evict_fallbacks` how often the pool could not be built
    # (or a submission failed) and the batch ran on the spill thread instead - the
    # count that says "parallel eviction was configured and did not happen".
    "evict_threads_requested": None,
    "evict_threads_configured": 0,
    "evict_threads_alive": 0,
    "evict_batches_submitted": 0,
    "evict_batches_serial": 0,
    "evict_files_unlinked": 0,
    "evict_files_failed": 0,
    "evict_fallbacks": 0,
    "evict_peak_in_flight": 0,
    # A FRESH PROCESS DOES NOT KNOW THE STORE IT BOOTED INTO. `estimated_bytes` is a
    # per-process counter of THIS process's own committed writes and starts at 0, and
    # the walk that could correct it only runs once that counter alone reaches the
    # ceiling. The seed breaks that circle: the first committed spill of a boot takes
    # ONE bounded walk and adopts its answer. `estimate_seed_key` is the (root,
    # ceiling) pair already seeded, so a reconfigure re-seeds and a steady boot does
    # not walk twice; `estimate_seed_walks` is how many times it has.
    "estimate_seed_key": None,
    "estimate_seed_walks": 0,
    "errors": 0,
    # the group-collision witness (see MAX_TRACKED_DIGESTS) and the journal
    "group_collisions": 0,
    "group_collision_samples": [],
    "digest_groups": {},
    "digest_groups_dropped": 0,
    "digest_group_overflow": 0,
    "sibling_probe_failures": 0,
    "journal_records": 0,
    "journal_failures": 0,
    "journal_removals": 0,
    "last_path": None,
    "last_reason": None,
    "last_error": None,
}


def reset_state_for_tests() -> None:
    """Back to a fresh boot: nothing configured, disarmed, nothing counted.

    The queue and the thread are stopped and dropped, not just un-counted: a test
    that spilled must not leave a daemon thread draining a queue the next test
    cannot see, because that thread would keep mutating the NEW test's `_state`
    through jobs it queued. The stop is sent BEFORE the references are dropped for
    exactly that reason, and it is best-effort: a test that never spilled never
    started a thread and has nothing to join.
    """
    global _queue, _writer_thread
    try:
        stop_writer(timeout_seconds=2.0)
    except Exception as exc:  # noqa: BLE001 - a reset must never raise into a fixture
        # Recorded rather than swallowed: a writer that will not stop is a fact a
        # test reader can see, not a silence.
        snapshot._error(exc, "disk_spill reset")
    _queue = None
    _writer_thread = None
    _state.update({
        "configured": False, "root": None, "ceiling_bytes": None,
        "armed": False, "arm_file_verdict": None,
        "arm_file_checked_mono": None, "arm_file_boot_refusals": 0,
        "fingerprint": None, "geometry_digest": None,
        "thread_started": 0, "thread_alive": False,
        "queue_depth": 0, "queue_depth_peak": 0,
        # The flush tallies: a reset that left these behind would let one test read
        # another test's wait as its own, including the bytes it claimed to have seen.
        "flush_calls": 0, "flush_drained": 0, "flush_timed_out": 0,
        "flush_bytes_written": 0, "flush_errors": 0, "flush_last": None,
        "shutdown_flush": None,
        "queue_bytes": 0, "queue_bytes_peak": 0,
        "window_start_mono": None, "window_files": 0,
        "requests": 0, "accepted": 0, "spilled": 0, "duplicates": 0,
        "bytes_written": 0, "write_seconds": 0.0, "read_seconds": 0.0,
        "drops": 0, "refused": {reason: 0 for reason in REFUSAL_REASONS},
        "evictions": 0, "evicted_bytes": 0, "estimated_bytes": 0,
        "walk_truncated": False, "errors": 0,
        # The measurement-honesty witness: a reset that left one of these behind
        # would let one test's walk be read as another test's ceiling decision.
        "walk_sampled_files": 0, "walk_total_files": 0,
        "walk_measured_bytes": 0, "walk_entries_examined": 0,
        "walk_skipped_files": 0, "walk_count_truncated": False,
        "walk_basis": None, "plan_truncated": False, "ceiling_checks": 0,
        # The pass's own clock and the live-peer witness: a reset that left these
        # behind would let one test read another test's deadline as its own.
        "ceiling_budget_seconds": None, "ceiling_pass_seconds": 0.0,
        "ceiling_deadline_hit": False, "ceiling_deadline_hits": 0,
        "walk_deadline_hit": False,
        "peer_digest_sightings": 0, "peer_digest_learned": 0,
        "peer_digests_live": 0, "peer_digests_unconfirmed": 0,
        "peer_digests_dropped": 0, "peer_files_planned": 0,
        # The fail-closed gate's witnesses (0.2.34): a reset that left these behind
        # would hand the next test a shortfall it never caused, or a withheld count
        # that makes a clean pass look like it could not attribute its own store.
        "unattributable_entries_last_pass": 0,
        "unattributable_bytes_last_pass": 0,
        "unattributable_entries_excluded": 0, "ceiling_passes_short": 0,
        # The grace and the pending claim (0.2.62): a reset that left these behind
        # hands the next test another test's withheld tally.
        "grace_entries_last_pass": 0, "grace_bytes_last_pass": 0,
        "grace_entries_excluded": 0, "grace_unlink_refused": 0,
        "grace_unlink_refused_last_pass": 0, "claim_unlink_refused": 0,
        "claim_unlink_refused_last_pass": 0, "claim_paths_last_pass": 0,
        "claim_set_truncated": False,
        "ceiling_plan_short": False, "ceiling_plan_short_bytes": 0,
        "ceiling_plan_short_reason": None, "peer_files_removed": 0,
        "peer_files_removed_bytes": 0, "peer_files_removed_last_pass": 0,
        "peer_files_removed_known_peer": 0,
        # The removal JUSTIFICATIONS and the identity view (0.2.45): a reset
        # that left these behind would hand the next test another test's
        # justification tally, or a `peer_identity_known: true` it never
        # earned - and the second one is a licence to delete a foreign digest.
        "stale_geometry_removed": 0, "stale_geometry_removed_bytes": 0,
        "stale_geometry_removed_last_pass": 0, "stale_key_shape_removed": 0,
        "stale_key_shape_removed_bytes": 0,
        "peer_files_removed_live": 0, "peer_files_removed_live_bytes": 0,
        "peer_liveness_source": None,
        "peer_identity_reads": 0, "peer_identity_available": False,
        "peer_identity_known": False, "peer_identity_digests": 0,
        "peer_identity_own_published": False,
        "peer_identity_files_unnamed": 0, "peer_identity_files_stale": 0,
        "peer_identity_files_unreadable": 0, "peer_identity_reason": None,
        # The fair share (0.2.30). Left behind by a reset, one test's peer count would
        # hand the next test a share of 1/N it never earned, and a share that is too
        # small over-evicts - the exact defect this release exists to remove.
        "estimated_own_bytes": 0, "estimated_foreign_dead_bytes": 0,
        "estimated_peer_live_bytes": 0, "estimated_unattributed_bytes": 0,
        "bytes_attributed": False, "share_bytes": None,
        "share_blocks": None, "share_own_width": None,
        "share_width_peers": None,
        "share_basis": SHARE_BASIS_EQUAL_BYTES,
        "share_divisor": None, "share_source": None,
        "share_target_bytes": None, "ceiling_passes_declined": 0,
        "ceiling_declined_accountable_bytes": None,
        "ceiling_declined_target_bytes": None, "peer_entries_excluded": 0,
        "ceiling_passes_nothing_sheddable": 0,
        # The eviction pool (0.2.29): shut it down and forget its shape, exactly as
        # the reader threads are dropped, so no worker of one test mutates another's.
        "evict_threads_configured": 0, "evict_threads_alive": 0,
        "evict_batches_submitted": 0, "evict_files_unlinked": 0,
        "evict_files_failed": 0, "evict_fallbacks": 0, "evict_batches_serial": 0,
        "estimate_seed_key": None, "estimate_seed_walks": 0,
        "group_collisions": 0, "group_collision_samples": [],
        "digest_groups": {}, "digest_groups_dropped": 0,
        "digest_group_overflow": 0, "sibling_probe_failures": 0,
        "journal_records": 0, "journal_failures": 0, "journal_removals": 0,
        "last_path": None, "last_reason": None, "last_error": None,
    })
    # The peer witness belongs to a boot: a digest remembered from an earlier test
    # would let that test's store protect this test's files from eviction.
    with _peer_lock:
        _peer_sightings.clear()
    # And so does the identity read's cache: a view of "who is alive" carried across a
    # test boundary is a licence to call a foreign digest dead, which is destructive.
    with _identity_lock:
        _peer_identity_cache["view"] = None
        _peer_identity_cache["mono"] = None
    # And so does the eviction pool: a worker left alive would unlink files the next
    # test believes it is only planning about.
    shutdown_evict_pool()


def counters() -> dict:
    """A copy for a payload. The collision WITNESS is published as a COUNT.

    `digest_groups` is the map itself, bounded at `MAX_TRACKED_DIGESTS` key16s -
    several KiB of text when full, which no polled payload should carry. Its count
    is the fact a reader wants; the capped `group_collision_samples` are the detail
    that is worth publishing.
    """
    out = dict(_state)
    out["digest_groups"] = int(len(_state["digest_groups"]))
    out["group_collision_samples"] = [dict(row)
                                      for row in _state["group_collision_samples"]]
    return out


def _fail(reason, **extra) -> dict:
    """The same failure shape as `disk_layout`: a reason, and no fake numbers.

    Local rather than imported from `disk_layout` because these payloads are this
    module's contract (they carry `reason_kind`, which the layout layer has no
    use for), and a shared helper that two modules bend in different directions is
    worse than ten duplicated lines.
    """
    out = {"ok": False, "reason": str(reason)}
    out.update(extra)
    return out


def _ok(units=None, **fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    out["units"] = dict(units or {})
    return out


# --- configuration: the two independent switches ----------------------------

def disk_root():
    """The configured store root, or None. Read per call, never at import.

    Unset is the DEFAULT and means the tier cannot exist here: no directory is
    created, no thread is started, no byte is written. The tmpfs roots are refused
    by `disk_layout.validate_root`, which is also where a relative path stops
    being an accident waiting for a different cwd.
    """
    raw = os.environ.get(config.ENV_DISK_ROOT, "").strip()
    if not raw:
        return None
    checked = disk_layout.validate_root(raw)
    if not checked["ok"]:
        return None
    return checked["absolute"]


def _slot_width_bytes():
    """The width ONE spill file has on this rank, or None before a fingerprint exists.

    The fingerprint's `total_bytes_per_block` - the same figure `read_slot_bytes`
    requires and `disk_read` reads back - so a cost published from here cannot drift
    from the bytes actually written. None is returned rather than 0: a cost of zero is a
    number a reader will multiply, and an unknown width is not.
    """
    fingerprint = _state.get("fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    width = fingerprint.get("total_bytes_per_block")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        return None
    return int(width)


def slot_width_bytes():
    """PUBLIC reading of this rank's file width, for a caller that must PRICE a read.

    `disk_restore` asks this before it makes an offer: a read costs one bounce buffer
    of exactly this many bytes, so "how many reads can the pool carry" is arithmetic
    over this figure and nothing else. Public as a thin wrapper rather than a second
    implementation - the fingerprint stays the single source, and a process with no
    fingerprint (an EngineCore, which has no KV tensors to measure) keeps answering
    None so its caller's conservative fallback stays honest.
    """
    return _slot_width_bytes()


def capacity_ceiling_bytes():
    """The configured byte ceiling, or None for "no ceiling".

    A garbage value falls back to NO ceiling rather than to 0: a 0 ceiling would
    mean "evict everything you see", which is a destructive reading of a typo.
    """
    raw = os.environ.get(config.ENV_DISK_BYTES, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def configured() -> dict:
    """What the environment says, refreshed. Never creates anything."""
    root = disk_root()
    ceiling = capacity_ceiling_bytes()
    _state["configured"] = root is not None
    _state["root"] = root
    _state["ceiling_bytes"] = ceiling
    return {"configured": root is not None, "root": root,
            "ceiling_bytes": ceiling,
            "env_root": config.ENV_DISK_ROOT, "env_ceiling": config.ENV_DISK_BYTES,
            "units": disk_layout._units("ceiling_bytes")}


def capability() -> str:
    """The three-state marker a reader distinguishes: never a bare bool.

    `not_configured`     - no root: nothing this module does can ever become work.
    `configured_disarmed`- the root exists, no arm: still zero disk work.
    `armed_and_ready`    - armed (the latch is up, and nothing lowers it but a
                           decision): accepted jobs spill.
    """
    if not configured()["configured"]:
        return "not_configured"
    return "armed_and_ready" if is_armed() else "configured_disarmed"


def _arm_path():
    directory = config.diagnostics_dir()
    if directory is None:
        return None
    return os.path.join(str(directory), ARM_FILE)


def is_armed() -> bool:
    """This process's arm: the in-process latch first, then the arm FILE.

    SINCE 0.2.35 THIS IS A LATCH AND NOT A TIMER. Nothing here reads a clock: the
    arm is raised by `arm()` and lowered by `disarm()`, and it stays exactly where a
    decision left it. What that means operationally, stated where a reader cannot
    miss it: THE ARM GATES BOTH DIRECTIONS, so `disarm()` stops SERVING from disk as
    well as writing to it (`request_spill` refuses, `disk_restore.gate()` refuses,
    the rank proof's tier arm refuses, `_may_restore_worker` refuses). It is the
    master switch, not a write-only knob.

    The file is the `blockdump` one-way-per-process problem again: the verb that
    arms runs in the EngineCore, and the spill happens in a worker, so the arm
    crosses as one small bounded file. Its verdict is cached for
    `ARM_FILE_POLL_SECONDS` because this is consulted per spill request, not per
    diagnostic poll. The file's OWN lifetime rule is not a timer either: what the
    reader refuses is a file that provably belongs to a different BOOT
    (`_arm_boot_fence`), never a file that has simply been sitting there.
    """
    if _state["armed"]:
        return True
    now = time.monotonic()
    checked = _state["arm_file_checked_mono"]
    if isinstance(checked, (int, float)) and (now - checked) < ARM_FILE_POLL_SECONDS:
        return bool(_state["arm_file_verdict"])
    _state["arm_file_checked_mono"] = now
    verdict, reason = _read_arm_file(_arm_path())
    _state["arm_file_verdict"] = verdict
    if not verdict:
        _state["last_reason"] = reason
    return bool(verdict)


def _boot_id() -> str | None:
    """This boot's kernel boot id, or None. Bounded, and never raises.

    The one boot identity that needs no operator configuration and is identical in
    every process of the boot (unlike `snapshot.epoch()`, which is per-process unless
    `VLLM_E2_TIER_DIAG_EPOCH` is set). A container that re-mints its boot id gets a
    stricter fence, never a weaker one: the only direction a mismatch can move is
    DISARMED.
    """
    try:
        with open(BOOT_ID_PATH, "rb") as handle:
            raw = handle.read(MAX_BOOT_ID_CHARS + 1)
    except OSError:
        return None
    text = raw.decode("utf-8", errors="replace").strip()
    if not text or len(text) > MAX_BOOT_ID_CHARS or "\n" in text:
        return None
    return text


def _arm_boot_fence(document) -> str | None:
    """Why this arm file belongs to a DIFFERENT BOOT, or None to believe it.

    WHY THIS EXISTS. The arm file used to be checked for `armed` and `expires_ts`
    only, so a surviving file armed a REBOOTED box for the rest of its 1800 s - and a
    cold boot with the disk tier armed exposes the restore path to live traffic with
    no operator having asked, which is the precondition of the 2026-10-05
    load-failure livelock. Today the file survives because the diagnostics directory
    is tmpfs; a fix that moved the whole directory to disk (to save the JOURNAL from
    the same fate) would have silently destroyed that invariant. So the arm fails
    closed ON ITS OWN CONTENTS, and the directory it lives in stops mattering.

    Two fences, each applied only when it can actually PROVE something, because a
    false refusal silently stops the tier from spilling:

    * the BOOT EPOCH, when the file says its epoch was the operator's per-boot one
      (`epoch_scope: "env"`). Then the epoch IS a boot identity and a disagreement is
      proof. With `epoch_scope: "process"` the epoch is a per-process token - which
      two processes of ONE boot legitimately disagree on - so it is not evidence of
      anything and is not consulted; that is what the boot id below is for.
    * the KERNEL BOOT ID, when both the file and this process have one.

    An arm file written by a plugin older than this change carries neither field: no
    comparison is possible, so it is believed (the same behaviour it had), and the
    gap closes for every file this version writes. Refusing such a file would also
    break the documented "arm an engine restart survives" workflow, which is a
    feature - it is a REBOOT the arm must not survive, and a reboot moves the boot
    id.
    """
    scope = document.get("epoch_scope")
    if scope == "env":
        documented = document.get("epoch")
        if isinstance(documented, str) and documented:
            ours, our_scope = config.diagnostics_epoch()
            if documented != ours:
                return (f"the arm file names a different boot: epoch mismatch "
                        f"(the file says {documented[:MAX_EPOCH_CHARS]!r} with "
                        f"scope env; this boot is {ours[:MAX_EPOCH_CHARS]!r} with "
                        f"scope {our_scope})")
    documented_boot = document.get("boot_id")
    ours_boot = _boot_id()
    if (isinstance(documented_boot, str) and documented_boot and ours_boot
            and documented_boot != ours_boot):
        return (f"the arm file is from a previous boot: boot id mismatch (the file "
                f"says {documented_boot[:MAX_BOOT_ID_CHARS]!r}, this boot is "
                f"{ours_boot[:MAX_BOOT_ID_CHARS]!r})")
    return None


def _read_arm_file(path):
    """(armed, reason) from the arm file. Fail CLOSED: any doubt is disarmed.

    SINCE 0.2.35 THE FILE CARRIES NO EXPIRY AND THIS READER READS NO CLOCK. The
    version that had one refused a file whose wallclock `expires_ts` had passed; that
    check is deleted, along with the requirement that the field exist at all, because
    the arm is a latch (see the module docstring: an expired arm froze LRU shedding
    as well as growth, and no setting of the timer bounded the store). A file that
    still carries `expires_ts` - one written by an older build - parses and is
    believed on the same terms as any other, and its timestamp is not consulted.

    THE ONE LIFETIME RULE THAT REMAINS IS THE BOOT: `_arm_boot_fence` refuses an arm
    written by a boot that is not this one (a reboot must not arm a cold box), and
    its reason NAMES the mismatch, because "disarmed and I do not know why" is how an
    armed-looking box ends up quietly doing nothing. The fence is load-bearing and
    unrelated to the timer: it separates REBOOTS. It does NOT separate engine
    restarts inside one kernel boot, and with the timer gone nothing in this reader
    does either - what closes that gap is the boot's own decision calling `disarm()`
    (`disk_boot.boot_enable`), which removes the file.
    """
    if path is None:
        return False, (f"{config.ENV_DIAG_DIR} is not set, so no arm file can "
                       "reach a worker")
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return False, "no arm file: disarmed"
    except OSError as exc:  # noqa: BLE001 - unreadable is disarmed, never a fault
        return False, f"the arm file could not be stat'ed: {type(exc).__name__}"
    if size > MAX_ARM_FILE_BYTES:
        return False, (f"the arm file is {size} bytes (cap "
                       f"{MAX_ARM_FILE_BYTES}); refusing to trust it")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_ARM_FILE_BYTES + 1)
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a broken arm file is a disarm
        return False, f"the arm file did not parse: {type(exc).__name__}"
    if not isinstance(document, dict) or document.get("armed") is not True:
        return False, "the arm file does not say armed"
    stale = _arm_boot_fence(document)
    if stale:
        _state["arm_file_boot_refusals"] += 1
        return False, stale
    return True, "armed by file"


def arm(seconds=None) -> dict:
    """Raise the latch here, and (when a diagnostics dir exists) by file.

    THE ARM IS A LATCH (0.2.35): it stays up until `disarm()` takes it down or the
    process exits. Nothing about it expires, and nothing in this module reads a
    clock to decide it.

    Refused when nothing is configured: an armed tier with no root is a knob that
    lies, so the arm requires the root, exactly as `blockdump.arm` requires a
    channel. There is NO env var that arms this - the same rule as guard mode and
    the byte dump: a launcher script must not be able to turn a facility that
    writes real KV bytes to disk on.

    `seconds` IS STILL ACCEPTED AND NOW DOES NOTHING. It stays in the signature
    because the call an operator already makes (`POST {"op":"arm","seconds":7200}`)
    must not start failing on the day this ships - the countdown that used to read
    the number is a separate, sequenced removal in the proxy repo. The verdict says
    `seconds_ignored: true` rather than pretending the value meant something, and the
    validation that existed only to bound it is gone with the TTL.
    """
    state = configured()
    if not state["configured"]:
        _state["refused"]["not_configured"] += 1
        return {"armed": False, "capability": "not_configured",
                "latched": True, "seconds_ignored": seconds is not None,
                "reason": f"{config.ENV_DISK_ROOT} is not set (or names a refused "
                          "path), so there is nowhere to write and arming would "
                          "only be a lie"}
    now = time.time()
    _state["armed"] = True
    _state["arm_file_checked_mono"] = None
    written = _write_arm_file(now)
    return {"armed": True, "capability": "armed_and_ready",
            "latched": True, "seconds_ignored": seconds is not None,
            "arm_file": written.get("path"),
            "arm_file_written": bool(written.get("written")),
            "arm_file_reason": written.get("reason"),
            "reason": None}


def disarm() -> dict:
    """Lower the latch here AND delete the file the ranks read.

    TOTAL, AND TOTAL IN BOTH DIRECTIONS: with the latch down nothing spills AND
    nothing is served from disk, because the spill writer, `disk_restore.gate()`, the
    rank proof's tier arm and `_may_restore_worker` all read this one verdict. There
    is no expiry left to lapse and no renewal left to undo it, so this is the only way
    an arm comes down short of a process exit - and because it removes the file, an
    operator who has disarmed leaves NOTHING behind for a later boot or a later
    process to resurrect (`_remove_arm_file`).
    """
    was = bool(_state["armed"])
    _state["armed"] = False
    _state["arm_file_verdict"] = None
    _state["arm_file_checked_mono"] = None
    removed = _remove_arm_file()
    return {"armed": False, "was_armed": was, "capability": capability(),
            "arm_file_removed": removed,
            "reason": "spills are refused again, AND serving from disk is refused "
                      "(one latch gates both); a queued job still lands, so drain "
                      "before trusting a zero"}


def arm_state() -> dict:
    """The arm as one stable-key view. It reports a LATCH, so it has no time left to
    report: `seconds_left`, `expires_at_wallclock`, `default_seconds` and
    `max_seconds` went out with the timer, and the reader that counted them down (the
    proxy's countdown) is a separate, sequenced removal.
    """
    armed = is_armed()
    return {"armed": armed, "capability": capability(),
            "latched": True,
            # THE BOOT FENCE, in the view an operator reads when the tier is quiet:
            # how many arm files were refused for naming another boot, and the last
            # reason the file verdict said no - which names the epoch or boot-id
            # mismatch instead of only saying "disarmed".
            "arm_file_boot_refusals": int(_state["arm_file_boot_refusals"]),
            "arm_file_reason": _state["last_reason"],
            "boot_id": _boot_id(),
            "units": disk_layout._units("arm_file_boot_refusals"),
            "note": ("an arm is a LATCH: a decision raises it and a decision lowers "
                     "it, and no clock does either (0.2.35 retired the TTL, the "
                     "default, the cap and the renewal that existed only to beat "
                     "them). The store is bounded by the byte ceiling, the per-rank "
                     "fair share, LRU-by-touch, the free-space floor and the "
                     "write-rate window - all of which work with the arm UP, and two "
                     "of which run ONLY while it is up. No env var arms this "
                     "facility; disarming stops SERVING from disk as well as writing "
                     "to it; and an arm file must also name THIS boot (the kernel "
                     "boot id, and the per-boot epoch when the operator sets one), so "
                     "a reboot disarms a surviving file instead of arming a cold box "
                     "with the restore path open to live traffic. An engine RESTART "
                     "inside one kernel boot is invisible to that fence, so a boot "
                     "that did not choose to arm deletes the file it did not ask "
                     "for: see disk_boot.boot_enable")}


def _write_arm_file(now: float) -> dict:
    path = _arm_path()
    if path is None:
        return {"written": False, "reason": "no diagnostics directory"}
    # The boot identity the reader will check itself against (`_arm_boot_fence`).
    # `epoch` is the OPERATOR's per-boot value when they set one and this process's
    # token otherwise, and `epoch_scope` says which - without the scope the reader
    # cannot tell "a different boot" apart from "a different worker of this boot",
    # and a false refusal would disarm a healthy tier. NO `expires_ts` AND NO
    # `seconds`: the arm is a latch, and a reader that silently ignored a lifetime
    # field would be worse than one that never had it.
    epoch, scope = config.diagnostics_epoch()
    document = {"version": PROTOCOL_VERSION, "writer": WRITER_TAG,
                "armed": True, "armed_ts": round(now, 6),
                "latched": True, "root": _state["root"],
                "epoch": str(epoch), "epoch_scope": scope, "pid": os.getpid(),
                "boot_id": _boot_id()}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as handle:
            handle.write(json.dumps(document).encode("utf-8"))
        os.replace(path + ".tmp", path)
    except OSError as exc:
        return {"written": False, "path": path,
                "reason": f"the arm file could not be written: {exc}"}
    return {"written": True, "path": path, "reason": None}


def _remove_arm_file() -> bool:
    path = _arm_path()
    if path is None:
        return False
    try:
        os.remove(path)
    except FileNotFoundError:
        return True
    except OSError as exc:  # noqa: BLE001 - a file we cannot remove is reported
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return False
    return True


# --- the fingerprint this process writes under ------------------------------

def configure(fingerprint) -> dict:
    """Validate and adopt the fingerprint that names every file we write.

    Boot-constant by nature (model, dtype, group layout, rank, world sizes), so it
    is set once by the wiring - and refused here, loudly, rather than discovered by
    a half-written store later. A None fingerprint CLEARS it, and with it the
    digest: without a digest this module cannot name a file at all, which is the
    same "refuse rather than guess" rule `disk_layout.validate_fingerprint` applies
    to the fields themselves.
    """
    if fingerprint is None:
        _state["fingerprint"] = None
        _state["geometry_digest"] = None
        return {"configured": False, "reason": "the fingerprint was cleared"}
    digested = disk_layout.fingerprint_digest(fingerprint)
    if not digested["ok"]:
        # Fail CLOSED: a rejected fingerprint leaves no digest behind, so a caller
        # that ignores this report cannot name a file with the PREVIOUS boot's
        # identity. Half a fingerprint is worse than none.
        _state["fingerprint"] = None
        _state["geometry_digest"] = None
        _state["refused"]["fingerprint_invalid"] += 1
        _state["last_reason"] = str(digested["reason"])[:MAX_STRING_CHARS]
        return {"configured": False, "reason": str(digested["reason"])}
    _state["fingerprint"] = digested["fingerprint"]
    _state["geometry_digest"] = digested["geometry_digest"]
    return {"configured": True,
            "geometry_digest": digested["geometry_digest"],
            "total_bytes_per_block": digested["total_bytes_per_block"],
            "num_groups": digested["num_groups"],
            "fingerprint": disk_layout.fingerprint_section(
                digested["fingerprint"]),
            "reason": None}


def geometry_digest():
    return _state["geometry_digest"]


# --- the reader: one slot's bytes, from the worker's region tensors ---------

def _region_span(tensor, slot: int, *, max_bytes: int) -> dict:
    """(base address, bytes) of slot `slot` in one region, or a refusal.

    THE CONTIGUITY GUARD IS NOT AN OPTIMISATION. The address formed below is
    `data_ptr() + slot * stride(0) * element_size()`, which assumes the region's
    slots tile the allocation with no holes and nothing between the last slot and
    the end of the buffer. That is exactly what a contiguous tensor guarantees -
    and exactly what `blockdump.read_block_bytes` checks, for the reason stated
    there: `ctypes.string_at` past the end of the allocation is a SIGSEGV, not a
    Python exception, and no `except Exception` can save a worker process. So the
    invariant is checked against the tensor's OWN byte count rather than trusted.
    """
    for name in ("data_ptr", "stride", "element_size"):
        if not callable(getattr(tensor, name, None)):
            return _fail(f"the region has no {name}() to read a slot out of")
    shape = tuple(getattr(tensor, "shape", ()) or ())
    if len(shape) < 1:
        return _fail("the region is 0-dimensional")
    blocks = int(shape[0])
    if not 0 <= slot < blocks:
        return _fail(f"slot {slot} is outside the region's {blocks} blocks")
    stride0 = tensor.stride(0)
    itemsize = tensor.element_size()
    if not isinstance(stride0, int) or not isinstance(itemsize, int) \
            or isinstance(stride0, bool) or isinstance(itemsize, bool) \
            or stride0 <= 0 or itemsize <= 0:
        return _fail("stride(0) or element_size() is not a positive int")
    bytes_per_block = int(stride0) * int(itemsize)
    if bytes_per_block > max_bytes:
        return _fail(f"this region alone is {bytes_per_block} bytes, past the "
                     f"slot's {max_bytes}-byte budget")
    nbytes = getattr(tensor, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(nbytes, bool) \
            and nbytes != blocks * bytes_per_block:
        return _fail(f"the region is not a flat run of {blocks} slots of "
                     f"{bytes_per_block} B (shape implies "
                     f"{blocks * bytes_per_block} B, nbytes is {nbytes} B); "
                     "refusing to form an address that could leave the "
                     "allocation")
    base = tensor.data_ptr()
    if not isinstance(base, int) or isinstance(base, bool) or base <= 0:
        return _fail("data_ptr() is not a usable address")
    return _ok(units=disk_layout._units("bytes_per_file"),
               base=base + slot * bytes_per_block, bytes_per_file=bytes_per_block,
               blocks=blocks)


def read_slot_bytes(caches, slot: int, total_bytes_per_block: int) -> dict:
    """One slot's bytes across every region of THIS rank, concatenated by name.

    `caches` is the worker's `cpu_kv_caches` (`worker.py:226-235`) - on the live
    box exactly ONE tensor per rank, deduped by `data_ptr` in
    `worker.py:unique_gpu_caches`, but the loop is written for N so a future fork
    shape does not silently spill half a slot. ORDER is `sorted(name)`, documented
    by the fingerprint's width check below: the same set of regions in the same
    boot always concatenates the same way, which is what makes the file's bytes a
    function of the slot and nothing else.

    The total must equal `total_bytes_per_block` EXACTLY. A slot that is smaller
    than the fingerprint claims is a fingerprint describing a different engine -
    writing it under this name would be a mislabeled file, and the mismatch is
    exactly the PP0/PP1 confusion the width field exists to catch.
    """
    if not isinstance(caches, dict) or not caches:
        return _fail("cpu_kv_caches is not a dict of region tensors (the worker "
                     "has not registered its KV caches)")
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        return _fail(f"slot must be a non-negative int, got {slot!r} (unit: slots)")
    if not isinstance(total_bytes_per_block, int) or isinstance(
            total_bytes_per_block, bool) or total_bytes_per_block <= 0:
        return _fail("total_bytes_per_block must be a positive int (unit: bytes)")
    names = sorted(str(name) for name in caches)
    if len(names) > MAX_REGIONS:
        return _fail(f"the region set has {len(names)} tensors; the cap is "
                     f"{MAX_REGIONS}")
    chunks: list[bytes] = []
    total = 0
    widths: dict = {}
    for name in names:
        try:
            span = _region_span(caches[name], slot,
                                max_bytes=total_bytes_per_block)
        except Exception as exc:  # noqa: BLE001 - a hostile tensor is a refusal
            return _fail(f"region {name!r}: inspecting it raised "
                         f"{type(exc).__name__}", region=name,
                         reason_kind="read_failed")
        if not span["ok"]:
            return _fail(f"region {name!r}: {span['reason']}",
                         region=name,
                         reason_kind=("region_not_contiguous"
                                      if "not a flat run" in str(span["reason"])
                                      else "slot_out_of_range"))
        widths[name] = span["bytes_per_file"]
        try:
            raw = ctypes.string_at(span["base"], span["bytes_per_file"])
        except Exception as exc:  # noqa: BLE001 - an unreadable region is data
            return _fail(f"region {name!r}: the host read failed: "
                         f"{type(exc).__name__}", region=name,
                         reason_kind="read_failed")
        if len(raw) != span["bytes_per_file"]:
            return _fail(f"region {name!r}: the host read returned "
                         f"{len(raw)} of {span['bytes_per_file']} bytes",
                         region=name, reason_kind="read_failed")
        chunks.append(raw)
        total += len(raw)
    if total != total_bytes_per_block:
        return _fail(f"the slot's regions total {total} bytes but the fingerprint "
                     f"declares {total_bytes_per_block}: these bytes are not what "
                     "this fingerprint describes",
                     reason_kind="region_width_mismatch",
                     widths={name: widths[name] for name in names})
    return _ok(units=disk_layout._units("bytes_per_file"),
               bytes=b"".join(chunks), bytes_per_file=total,
               regions=names, widths=widths,
               sha256=hashlib.sha256(b"".join(chunks)).hexdigest())


def sample_slot_bytes(worker, group, block_id: int, length_bytes: int) -> dict:
    """A bounded sample of one region's slot, through BLOCKDUMP'S OWN reader.

    Deliberately not a second implementation: `blockdump.read_block_bytes` is the
    path that was proven against real bytes on the live box, so a "did the spill
    and the dump agree?" question is answered by the two readers over the same
    tensor rather than by an argument. The 1 MiB per-request cap there is why this
    exists for samples while `read_slot_bytes` (whole slots) is here.
    """
    caches = getattr(worker, "cpu_kv_caches", None)
    if not isinstance(caches, dict) or not caches:
        return _fail("cpu_kv_caches is not present yet")
    found = blockdump._lookup_region(caches, group)
    if found is None:
        return _fail(f"group {group!r} is not a cpu_kv_caches region; available: "
                     f"{sorted(str(name) for name in caches)[:32]}")
    read = blockdump.read_block_bytes(found[1], block_id, 0, length_bytes)
    if not read.get("ok"):
        return _fail(str(read.get("reason")))
    return _ok(units=disk_layout._units("bytes_per_file", "length_bytes"),
               bytes=read["bytes"], sha256=read["sha256"],
               bytes_per_file=read["bytes_per_block"],
               length_bytes=read["length_bytes"], region=str(found[0]))


# --- the synchronous core: plan, read, commit, shed -------------------------

def measured_slot_bytes(caches) -> dict:
    """The width of one slot as THIS rank's tensors declare it: the sum of every
    region's `stride(0) * element_size()`.

    THIS is what a fingerprint's `total_bytes_per_block` is built FROM
    (`disk_store.build_worker_fingerprint`): the slot width is rank-local - the
    live box measures 11,468,800 B on PP0 and 12,902,400 B on PP1 - so the one
    fingerprint field that differs per rank is read per rank and never accepted
    from a broadcast. A fingerprint whose declared width disagrees with these
    tensors is refused downstream by `read_slot_bytes`, which is the check that
    stops a PP0 fingerprint from spilling PP1's bytes.
    """
    if not isinstance(caches, dict) or not caches:
        return _fail("cpu_kv_caches is not a dict of region tensors")
    names = sorted(str(name) for name in caches)
    if len(names) > MAX_REGIONS:
        return _fail(f"the region set has {len(names)} tensors; the cap is "
                     f"{MAX_REGIONS}")
    total = 0
    widths: dict = {}
    for name in names:
        tensor = caches[name]
        try:
            shape = tuple(getattr(tensor, "shape", ()) or ())
            if len(shape) < 1:
                return _fail(f"region {name!r} is 0-dimensional")
            stride0 = tensor.stride(0)
            itemsize = tensor.element_size()
        except Exception as exc:  # noqa: BLE001 - a hostile tensor is a refusal
            return _fail(f"region {name!r}: inspecting it raised "
                         f"{type(exc).__name__}")
        if isinstance(stride0, bool) or isinstance(itemsize, bool) \
                or not isinstance(stride0, int) or not isinstance(itemsize, int) \
                or stride0 <= 0 or itemsize <= 0:
            return _fail(f"region {name!r}: stride(0) or element_size() is not a "
                         "positive int")
        width = stride0 * itemsize
        if width > disk_layout.MAX_BLOCK_BYTES:
            return _fail(f"region {name!r} is {width} bytes per slot, past the "
                         f"{disk_layout.MAX_BLOCK_BYTES}-byte cap")
        widths[name] = width
        total += width
    if total <= 0 or total > disk_layout.MAX_BLOCK_BYTES:
        return _fail(f"the slot totals {total} bytes, outside (0, "
                     f"{disk_layout.MAX_BLOCK_BYTES}]")
    return _ok(units=disk_layout._units("bytes_per_file"),
               bytes_per_file=total, widths=widths, regions=names)


def plan_slot_naming(key, fingerprint=None) -> dict:
    """The name and the width a slot's bytes WOULD be written under - no bytes read.

    Split out of `plan_slot_write` for the copy-at-completion path: the bytes are
    snapshotted on the engine thread when the store event COMPLETES, and the name is
    derived later, on the writer thread, from the identity carried with them. The two
    halves must still be validated by the same code, so this is the one place the
    fingerprint check, the digest computation and the path build live, and
    `plan_slot_write` calls it rather than repeating it.

    Same single naming mode as `plan_slot_write`: the digest comes from the fingerprint
    THIS process was configured with, and a missing or invalid one is a refusal. No
    "hand me a digest" door - the digest is a function of rank-local facts and a
    second computing site is how files become un-findable.
    """
    adopted = fingerprint if fingerprint is not None else _state["fingerprint"]
    if adopted is None:
        return _fail("no fingerprint is configured: a file whose identity is "
                     "unknown must not be written",
                     reason_kind="fingerprint_missing")
    configured()
    digested = disk_layout.fingerprint_digest(adopted)
    if not digested["ok"]:
        return _fail(str(digested["reason"]), reason_kind="fingerprint_invalid")
    named = disk_layout.build_slot_path(_state["root"] or "", key,
                                        digested["geometry_digest"])
    if not named["ok"]:
        return _fail(str(named["reason"]), reason_kind="bad_key")
    return _ok(units=disk_layout._units("bytes_per_file", "ceiling_bytes",
                                        "slot", "block_id", "group_id"),
               path=named["path"], part_path=named["part_path"],
               directory=named["directory"], filename=named["filename"],
               key16=named["key16"], group_id=named["group_id"],
               group_field=named["group_field"], shard=named["shard"],
               geometry_digest=named["geometry_digest"],
               bytes_per_file=int(digested["total_bytes_per_block"]))


def plan_slot_write(caches, slot, key, fingerprint=None) -> dict:
    """Everything that can be refused, checked BEFORE any byte is written.

    Returns the validated plan (path, part path, expected bytes, digest) or a
    refusal naming the check that fired. No I/O, no allocation of the slot's
    bytes: the queue's job is a plan, so a job that cannot succeed is dropped
    before the writer thread touches the disk.

    ONE NAMING MODE, deliberately: the digest comes from the fingerprint THIS
    process was configured with, and a missing or invalid one is a refusal
    (`fingerprint_missing` / `fingerprint_invalid`). There is no "hand me a digest"
    door, because the digest is a function of rank-local facts - the slot width and
    the rank are this process's own readings (`disk_store` builds the fingerprint
    from the tensor stride and `worker_diag.rank_id()`), and accepting a digest
    from anywhere else would put two computing sites on the SAME field, which is
    precisely how files become un-findable.

    `key` must be the engine's FULL group-qualified block hash: `build_slot_path`
    splits it into the digest and the `g<group>` name component, so a key that does
    not carry the group suffix is a refusal (`bad_key`) rather than the ambiguous
    pre-fix name. That is the identity bug this closes, enforced at the door that
    writes.
    """
    adopted = fingerprint if fingerprint is not None else _state["fingerprint"]
    if adopted is None:
        return _fail("no fingerprint is configured: a file whose identity is "
                     "unknown must not be written",
                     reason_kind="fingerprint_missing")
    named = plan_slot_naming(key, adopted)
    if not named["ok"]:
        return _fail(str(named["reason"]),
                     reason_kind=named.get("reason_kind", "bad_key"))
    read = read_slot_bytes(caches, slot, named["bytes_per_file"])
    if not read["ok"]:
        return _fail(str(read["reason"]),
                     reason_kind=read.get("reason_kind", "read_failed"))
    return _ok(units=disk_layout._units("bytes_per_file", "ceiling_bytes",
                             "slot", "block_id", "group_id"),
               path=named["path"], part_path=named["part_path"],
               directory=named["directory"], filename=named["filename"],
               key16=named["key16"], group_id=named["group_id"],
               group_field=named["group_field"],
               shard=named["shard"],
               geometry_digest=named["geometry_digest"],
               bytes_per_file=read["bytes_per_file"], sha256=read["sha256"],
               slot=slot, blob=read["bytes"])


def _space_target(path: str) -> str:
    """The nearest EXISTING ancestor of `path`, for the free-space read.

    A first-ever spill has no store directory yet, and `statvfs` on a missing path
    reports an honest "unknown" - which, read as a refusal, would mean the tier
    could never create its own root. Walking up is not a guess: the filesystem
    under the nearest ancestor IS the filesystem the child will be created on.
    """
    current = path
    for _ in range(8):
        if os.path.isdir(current):
            return current
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return current
        current = parent
    return current


def _note_digest_group(key16, group_id, filename, shard) -> dict | None:
    """Remember which group spilled a `key16`, and count it when two disagree.

    Returns a bounded collision record when this spill's group id differs from a
    group id already recorded for the same `key16` - two KV-cache groups whose
    prefix hash landed on the same digest - and None otherwise. Called on the spill
    thread only, and does no I/O unless a collision actually happened.

    WHY A MEMORY MAP AND NOT A DIRECTORY SCAN EVERY SPILL: the rate is what we are
    measuring, so the check must cost nothing on the 99.99% of spills with no
    collision. The map is per-process and per-boot, which is exactly the window the
    operator reads (arm the tier today, read the counter tomorrow); a sibling file
    from a previous boot is not silently ignored either, because the collision
    probe looks at the shard directory when the map fires and records what is really
    on disk.

    The spill is NEVER refused by this: both files are legitimately distinct now,
    and refusing one would trade a measured collision for an unmeasured one.
    """
    groups = _state["digest_groups"].get(key16)
    if groups is None:
        while len(_state["digest_groups"]) >= MAX_TRACKED_DIGESTS:
            oldest = next(iter(_state["digest_groups"]))
            _state["digest_groups"].pop(oldest, None)
            _state["digest_groups_dropped"] += 1
        _state["digest_groups"][key16] = {int(group_id): 1}
        return None
    if group_id in groups:
        groups[int(group_id)] += 1
        return None
    # Deterministic and honest: the highest group id ALREADY recorded for this
    # digest. Computed before the new id is added, or the collision would be
    # reported as a collision between the new group and itself.
    prior = max(groups)
    if len(groups) >= MAX_GROUPS_PER_DIGEST:
        # A key16 with more distinct groups than the witness holds is itself the
        # finding; say so with a counter rather than grow the witness without bound.
        _state["digest_group_overflow"] += 1
    else:
        groups[int(group_id)] = 1
    _state["group_collisions"] += 1
    record = {"key16": key16,
              "group_id_a": int(prior), "group_id_b": int(group_id),
              "filename": str(filename)[:disk_layout.MAX_NAME_CHARS],
              "shard": str(shard)[:8],
              "sibling_on_disk": _sibling_on_disk(shard, key16, filename),
              "groups_seen": sorted(groups),
              "ts": round(time.time(), 6),
              "units": disk_layout._units("group_id_a", "group_id_b")}
    _state["group_collision_samples"] = (list(_state["group_collision_samples"])
                                         + [record])[-MAX_COLLISION_SAMPLES:]
    _state["last_reason"] = (f"group collision on {key16}: spilled for group "
                             f"{group_id} after group {prior} used the same digest")
    return record


def _sibling_on_disk(shard, key16, filename):
    """Is a DIFFERENT group's file for this `key16` actually in the shard? Runs only
    on a counted collision, so the cost is bounded by the collision rate.

    `None` means "could not look", never "is not there": an unreadable directory is
    a different fact from an absent file, and the counter must not invent the
    comfortable one.
    """
    root = _state["root"]
    if not root or not shard:
        return None
    directory = os.path.join(str(root), str(shard))
    prefix = f"{disk_layout.SLOT_FILE_PREFIX}{key16}-"
    try:
        seen = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                if seen >= MAX_COLLISION_SCAN:
                    return None
                seen += 1
                name = entry.name
                if (name.startswith(prefix) and name != str(filename)
                        and name.endswith(disk_layout.SUFFIX_DATA)):
                    return True
        return False
    except OSError as exc:
        _state["sibling_probe_failures"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return None


def _journal_write(path, size_bytes, *, removal: bool = False) -> dict:
    """Journal one committed file or one deleted file. The spill does not care.

    THE ONE RULE: this never changes the caller's verdict. A journal failure is
    counted (`journal_failures`) and reported in `last_reason`, and the bytes stay
    on disk either way - the spill is the product and this is an index of it, so the
    worst outcome here is a file the scheduler does not know about, which costs a
    recompute and nothing else.
    """
    try:
        if removal:
            recorded = disk_index.record_removal(path, size_bytes)
        else:
            recorded = disk_index.record_spill(path, size_bytes)
        if recorded.get("ok"):
            if removal:
                _state["journal_removals"] += 1
            else:
                _state["journal_records"] += 1
        else:
            _state["journal_failures"] += 1
            _state["last_reason"] = ("the spill journal did not record "
                                     f"{os.path.basename(str(path))}: "
                                     f"{str(recorded.get('reason'))[:96]}")
        return recorded
    except Exception as exc:  # noqa: BLE001 - an index never breaks a write
        _state["journal_failures"] += 1
        _state["errors"] += 1
        snapshot._error(exc, "disk_spill journal")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return {"ok": False, "reason": f"journaling raised {type(exc).__name__}"}


def spill_slot(caches=None, slot=None, key=None, fingerprint=None,
               blob=None) -> dict:
    """Read one slot and commit it. The whole write path, synchronously.

    Used by the writer thread and callable directly (a test, or a future synchronous
    caller). It NEVER raises: every failure is a classified dict, and `final_exists:
    False` on a failed write is the property `disk_layout` guarantees and this
    function preserves.

    TWO WAYS TO HAND OVER THE BYTES, and the difference between them is a race, not a
    style choice:

      * `blob=` - the bytes, already copied. THE PATH THE ENGINE HOOK USES since the
        restore tier landed: the copy happens at store COMPLETION, on the engine
        thread, in `disk_store.on_store_completed`. The snapshot's width is still
        checked against the fingerprint, so bytes that are not a whole slot are
        refused exactly as `read_slot_bytes` refuses a narrow slot.
      * `caches=` + `slot=` - read the slot HERE, now. Correct only when the caller
        knows the slot cannot be recycled before this call returns - true of a test and
        of a synchronous caller, and FALSE of a queued job: a CPU slot the RAM tier has
        finished storing into is available to the next store, so a job that carried a
        slot NUMBER would write the NEXT block's bytes under the PREVIOUS block's NAME.
        For an attention block that costs one wrong KV block; for an align-mode GDN
        boundary state one wrong state produces a confident wrong answer at
        `temperature=0`. That is why `request_spill` snapshots and never defers the
        read.

    Two things happen AFTER a successful commit and neither can affect it: the group
    collision witness is updated (`group_collisions`, see
    `_note_digest_group`) and the file is journalled (`disk_index.record_spill`,
    AFTER `write_committed_file` did its `os.replace`, so a write that never landed
    is never recorded as having landed). A duplicate is not journalled: the file was
    already there, which means it was already recorded - possibly by a boot whose
    journal this one continues.

    `directory_synced` in the answer is `False` on this path and that is the NORMAL
    state, not an anomaly: the spill tier does not fsync (see the module docstring),
    so it reports honestly that neither the file nor its directory was forced to the
    device. A reader must never treat it as a failure.
    """
    started = time.monotonic()
    if blob is not None:
        # COPY-AT-COMPLETION: the bytes came with the job, so what lands on disk is
        # what the store event completed with, whatever the RAM tier has since done
        # with the slot. The width check is the same one `read_slot_bytes` runs.
        plan = plan_slot_naming(key, fingerprint)
        if plan["ok"]:
            payload = bytes(blob)
            if len(payload) != int(plan["bytes_per_file"]):
                return {"ok": False,
                        "reason": f"the snapshot is {len(payload)} bytes but this "
                                  f"rank's slot is {plan['bytes_per_file']}: these "
                                  "bytes are not what this fingerprint describes",
                        "reason_kind": "region_width_mismatch",
                        "final_exists": False}
            plan = _ok(path=plan["path"], part_path=plan["part_path"],
                       directory=plan["directory"], filename=plan["filename"],
                       key16=plan["key16"], group_id=plan["group_id"],
                       group_field=plan["group_field"], shard=plan["shard"],
                       geometry_digest=plan["geometry_digest"],
                       bytes_per_file=plan["bytes_per_file"],
                       sha256=hashlib.sha256(payload).hexdigest(),
                       slot=slot, blob=payload)
    else:
        plan = plan_slot_write(caches, slot, key, fingerprint)
    if not plan["ok"]:
        return {"ok": False, "reason": plan["reason"],
                "reason_kind": plan.get("reason_kind", "read_failed")}
    space = disk_layout.space_check(_space_target(plan["path"]),
                                    watermark_bytes=MIN_FREE_BYTES,
                                    extra_bytes=plan["bytes_per_file"])
    if not space["ok"] or space["below_watermark"]:
        return {"ok": False, "final_exists": False,
                "reason": (f"the filesystem is not usable for a "
                           f"{plan['bytes_per_file']}-byte write: "
                           f"{space['reason']}"),
                "reason_kind": "no_free_space"}
    # NO FSYNC ON THE SPILL PATH, ON PURPOSE. `sync=False` means buffered writes
    # and no fsync of the file or its directory: the page cache owns device
    # ordering. These are CACHE bytes, so the only cost of a lost write is one
    # recompute, and the property that makes it free is that a journal entry whose
    # bytes never reached the device is classified `file_missing` (or
    # `size_mismatch`/`short_read`) at read time by `disk_read._read_file`,
    # reported to the engine, and pruned out of the DMA by `disk_restore` before
    # any stale byte is copied - so false presence costs a recompute and never
    # wrong bytes. `O_CREAT|O_EXCL` and the `os.replace` commit ordering stay
    # exactly as they were: those protect against TORN WRITES, which is a
    # different property from durability and is not what this flag removes.
    written = disk_layout.write_committed_file(plan["path"], plan["blob"],
                                               sync=False)
    elapsed = round(time.monotonic() - started, 6)
    if not written["ok"]:
        return {"ok": False, "final_exists": bool(written.get("final_exists")),
                "reason": written["reason"], "reason_kind": "write_failed",
                "path": plan["path"]}
    collision = _note_digest_group(plan["key16"], plan["group_id"],
                                   plan["filename"], plan["shard"])
    journalled = None
    if not written["duplicate"]:
        journalled = _journal_write(plan["path"], int(written["write_bytes"]))
    return {"ok": True, "duplicate": bool(written["duplicate"]),
            "path": plan["path"], "key16": plan["key16"],
            "group_id": plan["group_id"], "group_field": plan["group_field"],
            "shard": plan["shard"], "filename": plan["filename"],
            "bytes_written": int(written["write_bytes"]),
            "bytes_per_file": plan["bytes_per_file"], "sha256": plan["sha256"],
            "slot": slot, "elapsed_seconds": elapsed,
            "group_collision": collision,
            "journalled": bool(journalled and journalled.get("ok")),
            "journal_reason": (journalled or {}).get("reason"),
            "directory_synced": written.get("directory_synced"),
            "reason": written.get("reason"),
            "units": disk_layout._units("bytes_written", "bytes_per_file",
                                        "group_id")}


# --- the queue and its thread ------------------------------------------------

_queue = None
_writer_thread = None
_thread_lock = threading.Lock()


def _get_queue():
    """The one bounded queue. Created on first accepted job, never at import."""
    global _queue
    with _thread_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        return _queue


def _ensure_thread():
    """Start the module's single writer thread, once, lazily. Never at import."""
    global _writer_thread
    with _thread_lock:
        alive = _writer_thread is not None and _writer_thread.is_alive()
        _state["thread_alive"] = bool(alive)
        if alive:
            return True
        try:
            thread = threading.Thread(target=_writer_loop,
                                      name="kvtier-disk-spill", daemon=True)
            thread.start()
        except Exception as exc:  # noqa: BLE001 - no thread is a refusal, not a fault
            _state["refused"]["write_failed"] += 1
            snapshot._error(exc, "disk_spill thread start")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            return False
        _writer_thread = thread
        _state["thread_started"] += 1
        _state["thread_alive"] = True
        return True


_STOP = object()


def _writer_loop() -> None:
    """The drain loop. ONE thread, and the only place a byte is written.

    A job is a plan already validated by `spill_slot`'s own re-check; the loop
    never trusts the caller's `caches` to still be the same object (the worker may
    have re-registered its caches), which is why a job carries the worker's cache
    MAPPING rather than a copy of its bytes and the read happens here.
    """
    local = _get_queue()
    while True:
        try:
            job = local.get(timeout=0.5)
        except queue.Empty:
            continue
        if job is _STOP:
            local.task_done()
            break
        try:
            _handle_job(job)
        except Exception as exc:  # noqa: BLE001 - a job never kills the thread
            _state["errors"] += 1
            snapshot._error(exc, "disk_spill job")
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        finally:
            # The snapshot leaves memory when the job leaves the queue - whether the
            # write landed, failed, or the loop is draining on shutdown. It sits in the
            # finally and not in the success branch on purpose: a byte counter that only
            # ever grows fills the cap permanently and the tier quietly stops spilling,
            # the same silent-failure shape that cost 25,945 watchdog cycles on the
            # llm-api box for a latch that outlived the attempt that set it.
            _state["queue_depth"] = max(0, int(_state["queue_depth"]) - 1)
            _state["queue_bytes"] = max(0, int(_state["queue_bytes"])
                                        - int(job.get("bytes") or 0))
            local.task_done()


def _handle_job(job) -> None:
    """One queued job, on the spill thread. The bytes arrive as `blob` - the snapshot
    `request_spill` copied at store completion.

    There is deliberately NO `caches` fallback here. A job that arrived through the
    queue has no right to read a slot it does not own: the RAM tier reuses a CPU slot
    as soon as its store completes, so reading it later writes the NEXT block's bytes
    under THIS block's name. For an attention block that is one wrong KV block; for an
    align-mode GDN boundary state, one wrong state answers confidently and wrongly at
    `temperature=0`. A job without a snapshot is a caller bug and is reported as one.
    """
    started = time.monotonic()
    if job.get("blob") is None:
        _state["refused"]["bad_call"] += 1
        _state["last_reason"] = (
            "the job carries no snapshot: request_spill copies the bytes at store "
            "completion and a queued job must not read a slot it does not own"
        )[:MAX_STRING_CHARS]
        return
    result = spill_slot(key=job.get("key"), fingerprint=job.get("fingerprint"),
                        slot=job.get("slot"), blob=job.get("blob"))
    _state["read_seconds"] = round(float(_state["read_seconds"])
                                   + (time.monotonic() - started), 6)
    if not result.get("ok"):
        kind = str(result.get("reason_kind", "write_failed"))
        if kind in _state["refused"]:
            _state["refused"][kind] += 1
        else:
            _state["refused"]["write_failed"] += 1
        _state["last_reason"] = str(result.get("reason"))[:MAX_STRING_CHARS]
        return
    if result.get("duplicate"):
        _state["duplicates"] += 1
        _state["last_reason"] = "a committed file with this identity was already " \
                                "there"
        return
    _state["spilled"] += 1
    _state["bytes_written"] = int(_state["bytes_written"]) + int(
        result["bytes_written"])
    _state["write_seconds"] = round(float(_state["write_seconds"])
                                    + float(result["elapsed_seconds"]), 6)
    _state["estimated_bytes"] = int(_state["estimated_bytes"]) + int(
        result["bytes_written"])
    # Every byte THIS process commits is written under THIS rank's geometry digest, so
    # the fair-share's own-bytes bucket tracks the same writes the store-wide counter
    # does (0.2.30). Without this the share would only ever be re-measured by a walk and
    # a rank would fly blind between passes about how much of its budget it had spent.
    _state["estimated_own_bytes"] = int(_state["estimated_own_bytes"]) + int(
        result["bytes_written"])
    _state["last_path"] = result["path"]
    # A fresh process knows nothing about the store it booted into; give the ceiling
    # that knowledge ONCE before it decides, or it cannot see a store it did not fill.
    _maybe_seed_estimate()
    _enforce_ceiling()


def _publish_walk(walk: dict) -> None:
    """Record the shape of the walk the ceiling logic just ran, so the decision that
    came out of it can be read back afterwards. ONE place, used by the seed walk and
    by the enforcement walk alike: a witness that only described half the walks would
    be the same trap as a payload that cannot tell a sample from a measurement.
    """
    _state["walk_truncated"] = bool(walk["truncated"])
    _state["walk_count_truncated"] = bool(walk["count_truncated"])
    _state["walk_sampled_files"] = int(walk["files"])
    _state["walk_total_files"] = int(walk["total_files"])
    _state["walk_measured_bytes"] = int(walk["used_bytes"])
    _state["walk_entries_examined"] = int(walk["entries_examined"])
    _state["walk_skipped_files"] = int(walk["skipped_files"])
    _state["walk_basis"] = str(walk["measurement_basis"])
    _state["walk_deadline_hit"] = bool(walk.get("deadline_hit"))


def _maybe_seed_estimate() -> None:
    """Adopt the store's own size once per configuration, from ONE bounded walk.

    WHY THIS IS LOAD-BEARING AND NOT A REFINEMENT. `estimated_bytes` is a per-process
    counter of THIS process's own committed writes and starts at 0, and the walk that
    could correct it only ran once that counter ALONE reached the ceiling. So a rank
    booting into a store someone else filled was blind to it in proportion to how
    full it already was: on the box tonight a fresh boot had to spill 192 GiB of NEW
    files (17,384 of them, ~48 minutes at the measured ~4 GiB/min) before it was
    allowed to LOOK at a 335 GiB store with only 60 GiB of `/home` left to grow into
    (the store reached 396 GiB with 61 GiB free before this shipped). The filesystem
    fills long before the ceiling wakes up, and no amount of honest
    scaling inside the walk changes that, because the walk never runs. The trigger
    being a write counter is the reason the ceiling could not rescue the store it was
    armed to protect.

    THE COST IS ONE WALK. It happens on the first committed spill of a (root, ceiling)
    pair - 0.224 s measured replaying the live 34,896-file store, on the writer thread,
    where the walk's whole cost model already lives - and after that the fast path is
    exactly what it was: no walk, no plan and no unlink until this rank's accountable
    bytes cross its share of the ceiling (or the store crosses the whole ceiling). A
    reconfigure (a different root or ceiling) re-seeds; a steady boot does
    not walk twice. A walk that could not measure anything (a missing store, or the
    entry-bound starved `sample_unmeasured`) leaves the running estimate alone rather
    than writing a 0 into it - the same rule `_enforce_ceiling` applies to its own walk.
    """
    root = _state["root"]
    ceiling = _state["ceiling_bytes"]
    if not root or not ceiling:
        return
    key = f"{root}|{ceiling}"
    if _state["estimate_seed_key"] == key:
        return
    # Claim the seed BEFORE the walk: one attempt per configuration, so a store that
    # cannot be read does not buy a walk on every spill of the boot.
    _state["estimate_seed_key"] = key
    _state["estimate_seed_walks"] = int(_state["estimate_seed_walks"]) + 1
    walk = _walk_store(root)
    if not walk["ok"]:
        _state["last_reason"] = (f"the estimate could not be seeded: "
                                 f"{str(walk['reason'])[:96]}")
        return
    _publish_walk(walk)
    if walk["truncated"] and int(walk["files"]) == 0:
        return
    # THE SEED DOES THE APPORTIONMENT TOO (0.2.30). A rank booting into a store the
    # filled must learn its share before it decides anything, and the walk that measures
    # the store is the same walk that learns which foreign digests are alive - so the
    # first ceiling decision of a boot is a fair-share decision, not a store-wide one.
    # That ordering is the whole reason the fail-safe direction is cheap: by the time a
    # pass is expensive enough to matter, this rank has already walked the store once.
    attribution = _attribute_walk_bytes(walk, *_peer_liveness())
    # THE WIDTH-WEIGHTED SHARE (0.2.58). This is the first decision of a boot, and the
    # widths needed to make it block-equal are already in hand: this rank's from its own
    # fingerprint, each live sibling's from the rank file the identity view opened a few
    # lines above. No width, no weighting - see `_share_of_ceiling`.
    _apply_attribution(attribution, int(walk["estimated_total_bytes"]),
                       _share_of_ceiling(
                           int(ceiling), attribution["divisor"],
                           attribution["source"],
                           own_width=_slot_width_bytes(),
                           peer_widths=_peer_widths_for_share(
                               attribution["peers"])))


# --- the live peer ranks: learning which foreign digests are somebody's data ---

# digest -> {"sightings": int, "fresh_sightings": int, "last_mtime_s": float}.
# Its OWN lock, never nested with anything else in this module: `_walk_store` is
# called from the spill writer thread AND from a polled utility verb (`disk_store`'s
# `ls`), so this map can have two authors, while `_state` itself keeps the
# single-writer posture it has always had.
_peer_sightings: dict = {}
_peer_lock = threading.Lock()


def _observe_peer_digest(seen_digest, mtime_ns: int) -> None:
    """Record that a file under a FOREIGN digest exists and when it was written.

    Called from the walk's measuring pass for every capacity-visible file whose parsed
    digest is not this boot's. Cost: one dict lookup per such file, no syscall - the
    `stat` and the parse the pass already paid are the only real work here.

    This only OBSERVES. Whether a digest may be called a LIVE sibling's is
    `_peer_liveness`' decision, made at the moment the plan is built, from this map AND
    the rank files (`_peer_identity_view`) - because one witness going quiet proves
    nothing on its own. Since 0.2.45 a digest that is NOT believed live is not
    automatically free capacity either: it is only free when this rank can name every
    live rank's identity, and a rank that cannot is told so (`peer_liveness_source`,
    `peer_identity_known`) and withholds those bytes instead of shedding them.
    """
    own = _state["geometry_digest"]
    if not isinstance(seen_digest, str) or not seen_digest \
            or seen_digest == own:
        return
    key = seen_digest.lower()
    wall_now = time.time()
    written = int(mtime_ns) / 1_000_000_000.0
    with _peer_lock:
        row = _peer_sightings.get(key)
        if row is None:
            if len(_peer_sightings) >= MAX_PEER_DIGESTS:
                # Bounded witness. This map is one of TWO liveness witnesses, so a
                # digest dropped here can still be protected by its owner's rank file;
                # and since 0.2.45 a digest no witness believes is not automatically
                # sheddable either - it is `unattributable` unless this rank can name
                # every live rank. The cap therefore costs a fair share, never data.
                _state["peer_digests_dropped"] += 1
                return
            row = {"sightings": 0, "fresh_sightings": 0, "last_mtime_s": 0.0,
                   "learned": False}
            _peer_sightings[key] = row
        row["sightings"] = int(row["sightings"]) + 1
        if written > float(row["last_mtime_s"]):
            row["last_mtime_s"] = written
        if wall_now - written <= PEER_DIGEST_FRESH_SECONDS:
            row["fresh_sightings"] = int(row["fresh_sightings"]) + 1
    _state["peer_digest_sightings"] += 1


def _live_peer_digests() -> tuple:
    """Which foreign digests this rank's WALK says are writing now, as a sorted tuple.

    TWO conditions, both required, and both observable rather than assumed: at least
    `PEER_DIGEST_MIN_SIGHTINGS` committed files seen inside the freshness window, and
    the most recently written of them inside it. This is the WRITE-ACTIVITY witness: it
    can say "alive" with certainty (a file was just written under that digest) and
    it can
    NEVER say "dead" with certainty, which is the whole 0.2.45 point - a rank with a
    warm
    prefix stops spilling, and a rank that stops spilling is not a rank that has gone.
    Ageing a digest out of THIS map is therefore only a statement about write
    traffic: it
    stops counting toward this rank's fair share, and `_peer_identity_view` decides
    separately whether the bytes under it are still anybody's.

    Empty means "nothing is writing", and `_peer_liveness` turns that into either "I
    know
    who is alive" (the rank files named them) or "I cannot tell" - never, as it used to,
    into "those bytes belong to nobody".
    """
    wall_now = time.time()
    live: list = []
    with _peer_lock:
        for digest, row in list(_peer_sightings.items()):
            fresh = int(row["fresh_sightings"])
            age = wall_now - float(row["last_mtime_s"])
            if fresh >= PEER_DIGEST_MIN_SIGHTINGS \
                    and age <= PEER_DIGEST_FRESH_SECONDS:
                live.append(digest)
                if not row.get("learned"):
                    # First time this digest has qualified. The counter is the number
                    # of siblings this rank has ever recognised, so `peer_digests_live`
                    # (a snapshot) and `peer_digest_learned` (a boot total) answer
                    # different questions and neither is a guess.
                    row["learned"] = True
                    _state["peer_digest_learned"] += 1
                continue
            # Forget it. A dead rank's digest must not sit in the map vetoing the
            # shedding of its now-dead files forever after it stopped writing.
            _peer_sightings.pop(digest, None)
        _state["peer_digests_live"] = len(live)
        if not live:
            _state["peer_digests_unconfirmed"] += 1
    return tuple(sorted(live))


# --- the second witness: a live worker's own published identity (0.2.45) ----

# {"mono": float, "view": dict} - the last identity read, so one ceiling pass costs ONE
# directory read however many times `_peer_liveness` is asked. `disk_restore` caches its
# read of the SAME files for the same reason (`PROOF_CACHE_SECONDS`).
_peer_identity_cache: dict = {"mono": None, "view": None}
_identity_lock = threading.Lock()


def _peer_rank_files() -> list:
    """The rank-file paths to probe, sorted, capped. Never raises; [] means none.

    The same directory the ARM FILE lives in (`VLLM_E2_TIER_DIAG`), and the same naming
    contract `worker_diag` writes: `rank-<PPn_TPm>.json`. Hidden files (the `.tmp`
    halves
    of an atomic replace) are skipped by name, and the cap is `disk_restore`'s own bound
    on how many it will probe, so a directory full of junk cannot make this unbounded.
    """
    directory = config.diagnostics_dir()
    if directory is None:
        return []
    try:
        names = sorted(os.listdir(str(directory)))
    except OSError:
        return []
    out = []
    for name in names:
        if not name.startswith(worker_diag.RANK_FILE_PREFIX) \
                or not name.endswith(worker_diag.RANK_FILE_SUFFIX) \
                or name.startswith("."):
            continue
        out.append(os.path.join(str(directory), name))
        if len(out) >= MAX_PEER_RANK_FILES:
            break
    return out


def _published_digest(document) -> str | None:
    """The digest one rank document says is its CURRENT one, or None. Never raises.

    `disk_store.worker_report()` publishes it as `disk.digest`, and it is the same value
    `fingerprint_digest` computed from this rank's own fingerprint - which is what
    makes it
    a statement about IDENTITY ("the bytes I write now carry this digest") rather than a
    statement about traffic. Untrusted input: a non-string, an over-long or a
    non-lowercase-hex value is refused, and refusing it is a fail-CLOSED choice (an
    unreadable identity means "I cannot name every live rank", never "that digest is
    nobody's").
    """
    try:
        section = document.get("disk")
        if not isinstance(section, dict):
            return None
        digest = section.get("digest")
        if not isinstance(digest, str):
            return None
        text = digest.strip().lower()
        if not text or len(text) > MAX_PEER_DIGEST_CHARS:
            return None
        if any(character not in "0123456789abcdef" for character in text):
            return None
        return text
    except Exception:  # noqa: BLE001 - an unreadable identity is a fact, not a fault
        return None


def _published_width(document) -> int | None:
    """The block width one rank document says ITS spill files have, or None.

    `worker_diag.build_document` publishes it as `arena.bytes_per_block` - the width
    measured off that rank's own region tensors, the same figure `disk_spill`
    fingerprints into its geometry digest and `disk_read` reads back. It lives in the
    `arena` section, NOT the `disk` one: the disk section publishes no width at all, so
    a reader that asks for `disk.bytes_per_block` reads a None and concludes the box has
    no widths to sum.

    Untrusted input, so the bounds are the point: a non-int, a bool, a zero, a negative
    or an absurd width is refused, and refusing it is NOT a failure - it costs the
    block-equal share and the caller falls back to the equal-byte split it shipped with
    before 0.2.58.
    """
    try:
        section = document.get("arena")
        if not isinstance(section, dict):
            return None
        width = section.get("bytes_per_block")
        if isinstance(width, bool) or not isinstance(width, int):
            return None
        if width <= 0 or width > MAX_PEER_WIDTH_BYTES:
            return None
        return int(width)
    except Exception:  # noqa: BLE001 - an unreadable width is an absence, not a fault
        return None


def _peer_identity_view(*, force: bool = False) -> dict:
    """WHO IS ALIVE and what digest each live rank writes under, from the rank files.

    Returns `{"available": bool, "known": bool, "digests": tuple, "peer_widths":
    {digest: bytes}, "own_published": bool,
    "files_read": n, "files_unnamed": n, "files_stale": n, "files_unreadable": n,
    "reason": str}`. Bounded by `MAX_PEER_RANK_FILES` x `MAX_PEER_RANK_FILE_BYTES`,
    cached
    for `PEER_IDENTITY_CACHE_SECONDS`, and it never raises: every failure is a
    counter and
    a reason, because a rank that cannot read the directory must keep spilling.

    THE THREE READINGS, and the `known` flag is the whole point of the function:
      * `available`, `known` - the directory yielded at least one FRESH rank
      document and
        every fresh one published a readable digest. This rank can name every live rank,
        so a foreign digest outside the set is a digest nobody writes under: an
        old boot's
        or a configuration change's, i.e. `stale_geometry`, and reclaimable.
      * `available`, NOT `known` - at least one FRESH rank file has no readable
      digest (a
        mixed-version fleet, a contributor that raised). "Not in my list" is not
        evidence
        of absence, so no foreign committed digest may be called dead and the
        ceiling may
        come up SHORT.
      * NOT `available` - no fresh rank document here at all (no `VLLM_E2_TIER_DIAG`, a
        worker that has not written once, a directory this process cannot read). The
        caller falls back to the write-activity reading, which is exactly the pre-0.2.45
        behaviour: the stale-geometry reclaim path is never made harder to reach by this
        change, it is only no longer reachable through SILENCE.

    A STALE rank file (older than `PEER_RANK_FILE_FRESH_SECONDS`, or stamped in the
    future) is neither a protection nor an unnameable: it is a worker that has stopped
    heartbeating, so its digest stops being protected and its orphaned bytes become
    reclaimable. That is how a genuinely-gone rank frees its bytes without a timer over
    cached bytes - the bound is on a PROCESS's liveness proof.
    """
    cached = _peer_identity_cache.get("view")
    mono = _peer_identity_cache.get("mono")
    if not force and isinstance(cached, dict) \
            and isinstance(mono, (int, float)) \
            and (time.monotonic() - mono) < PEER_IDENTITY_CACHE_SECONDS:
        return cached
    own = _state["geometry_digest"]
    own_text = own.lower() if isinstance(own, str) else None
    view = {"available": False, "known": False, "digests": (),
            "peer_widths": {},
            "own_published": False, "files_read": 0, "files_unnamed": 0,
            "files_stale": 0, "files_unreadable": 0, "files_missing_ts": 0,
            "reason": None}
    live: set = set()
    widths: dict = {}
    wall_now = time.time()
    paths = _peer_rank_files()
    for path in paths:
        try:
            with open(path, "rb") as handle:
                raw = handle.read(MAX_PEER_RANK_FILE_BYTES + 1)
            if len(raw) > MAX_PEER_RANK_FILE_BYTES:
                # Unreadable by the proxy host's reader too, so it proves nothing.
                view["files_unreadable"] += 1
                continue
            document = json.loads(raw.decode("utf-8", "replace"))
            if not isinstance(document, dict):
                view["files_unreadable"] += 1
                continue
        except Exception:  # noqa: BLE001 - one bad file is not evidence, that's all
            view["files_unreadable"] += 1
            continue
        stamp = document.get("ts")
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
            view["files_missing_ts"] += 1
            view["files_unreadable"] += 1
            continue
        age = wall_now - float(stamp)
        if age < 0.0 or age > PEER_RANK_FILE_FRESH_SECONDS:
            # A worker that is not heartbeating any more: the future-stamp refusal is
            # `disk_restore._read_rank_file`'s rule for the same file, kept identical so
            # the two readers of one file cannot disagree about who is alive.
            view["files_stale"] += 1
            continue
        view["files_read"] += 1
        view["available"] = True
        digest = _published_digest(document)
        if digest is None:
            view["files_unnamed"] += 1
            continue
        if digest == own_text:
            view["own_published"] = True
            continue
        live.add(digest)
        # THE WIDTH THAT RANK SPILLS AT, from the same file already in memory: no extra
        # open, no extra read, and the same freshness rule as its digest, so a width can
        # never be read from a rank file too stale to vouch for the rank.
        width = _published_width(document)
        if width is not None:
            widths[digest] = width
    view["digests"] = tuple(sorted(live))
    view["peer_widths"] = {digest: widths[digest] for digest in sorted(widths)}
    if view["available"]:
        view["known"] = view["files_unnamed"] == 0
        view["reason"] = (
            f"{view['files_read']} fresh rank document(s): "
            f"{len(view['digests'])} foreign digest(s) published, "
            f"{view['files_unnamed']} could not name their digest"
            + ("" if view["known"] else ", so no foreign digest may be called dead"))
    else:
        view["reason"] = (
            "no fresh rank document here "
            f"({len(paths)} file(s) probed: {view['files_stale']} stale, "
            f"{view['files_unreadable']} unreadable), so identity is decided by write "
            "activity alone - the pre-0.2.45 reading")
    with _identity_lock:
        _peer_identity_cache["view"] = view
        _peer_identity_cache["mono"] = time.monotonic()
    _state["peer_identity_reads"] = int(_state["peer_identity_reads"]) + 1
    _state["peer_identity_known"] = bool(view["known"])
    _state["peer_identity_available"] = bool(view["available"])
    _state["peer_identity_digests"] = len(view["digests"])
    _state["peer_identity_own_published"] = bool(view["own_published"])
    _state["peer_identity_files_unnamed"] = int(view["files_unnamed"])
    _state["peer_identity_files_stale"] = int(view["files_stale"])
    _state["peer_identity_files_unreadable"] = int(view["files_unreadable"])
    _state["peer_identity_reason"] = str(view["reason"])[:MAX_STRING_CHARS]
    return view


def _peer_liveness() -> tuple:
    """`(live_peer_digests, peers_known)` - the two witnesses, combined honestly.

    `live` is the UNION: a digest is a live sibling's if a sibling is writing under
    it OR
    a live worker publishes it as its current digest. A union is the only honest shape,
    because each witness can produce a false negative and neither can produce a false
    positive about liveness - a rank that has not spilled for an hour still has a
    worker,
    and a rank whose rank file has not landed yet still has files being written.

    `peers_known` is `attribute_name`'s gate on calling a foreign digest `dead`:
    true only
    when the rank files named every live rank this rank can see, or - when there are no
    rank files to read at all - when the write-activity map is non-empty, which is what
    every release up to 0.2.44 used and what a box with no `VLLM_E2_TIER_DIAG` keeps.
    """
    written = _live_peer_digests()
    view = _peer_identity_view()
    live = tuple(sorted(set(written) | set(view["digests"])))
    if view["available"]:
        known = bool(view["known"])
        source = "rank_files"
    else:
        known = bool(written)
        source = "write_activity"
    _state["peer_liveness_source"] = source
    _state["peer_digests_live"] = len(live)
    if not live:
        _state["peer_digests_unconfirmed"] += 1
    return live, known


# --- the fair share: whose bytes is this rank accountable for? ---------------

def _scale_to_store(sample: int, used: int, scaled_total: int) -> int:
    """One class's sample bytes, scaled by the walk's OWN scaling factor.

    The walk already publishes the number the whole-store estimate came from
    (`estimated_total_bytes`) and the number it was scaled out of (`used_bytes`), so
    reusing that ratio for the per-owner classes cannot disagree with it: for a full
    measurement the two are equal and every class comes back byte for byte, which is the
    property every store inside the walk's caps rests on.
    """
    if used <= 0 or scaled_total <= used:
        return int(sample)
    return int(int(sample) * int(scaled_total) // int(used))


def _attribute_walk_bytes(walk: dict, peers, peers_known: bool = True) -> dict:
    """WHO owns the bytes one walk measured, and therefore what this rank's share is.

    Returns the per-owner byte totals (scaled to the store the way the walk scaled its
    own total), the entries this rank is ALLOWED to shed, and the divisor for the share.
    Never raises and never plans: this is the measurement half of the decision, and
    `disk_layout.lru_eviction_plan` stays the only owner of the ORDER.

    THE FOUR BUCKETS, and the rule for each:
      * `current` (this geometry digest) -> OWN. Counted in full against this rank's
        share: these are the bytes this rank wrote for its own readers.
      * `stale_geometry` / `stale_key_shape` -> DEAD. Belongs to no live rank: a reader
        enforces an exact digest AND byte-size match, so a mismatched file can never be
        applied by anybody, and every rank may shed them. Counted against EVERY rank's
        share, so the worthless bytes go first whoever wakes up (the constant block
        above says why a 1/N share of them is the wrong trade).
      * `peer_rank` (a digest learned to be a live sibling's) -> NEITHER counted NOR
        sheddable. The sibling accounts for those bytes under its own share; charging
        them here as well would make two ranks shed one file's worth of pressure, and
        leaving them in the planner's entry list would let a deliberately conservative
        scaled estimate reach into a live sibling's data.
      * anything else (a name this tier cannot classify, or - since 0.2.34 - a
        committed file under a FOREIGN digest seen while NO peer digest is confirmed)
        -> UNATTRIBUTED: counted as capacity, never handed to the planner. The second
        case is the fail-closed gate. Such a file is either an old boot's dead bytes
        or a live sibling's restore, and the name cannot say which while the peer map
        is empty; the pre-0.2.34 code counted it as DEAD and shed it first, which is
        how four ranks on one root deleted each other's live data (15,848 removal
        lines of the sibling's block size, and a `file_missing` hole 19,200 tokens
        deep). Its BYTES stay in `accountable_bytes` exactly as they did - they are on
        disk and they are this rank's problem to see - only the licence to delete
        them is gone, so the ceiling comes up SHORT and says so.
    """
    digest = _state["geometry_digest"]
    entries = [row for row in (walk.get("entries") or []) if isinstance(row, dict)]
    used = int(walk.get("used_bytes") or 0)
    scaled_total = int(walk.get("estimated_total_bytes") or 0)
    if not isinstance(digest, str) or not digest:
        # No fingerprint means no ownership can be established for ANY file here, so
        # there is nothing to apportion. Fall back to the whole store against the whole
        # ceiling - what every release before 0.2.30 did, and the direction that can
        # only ever shed MORE, never less.
        return {"attributed": False, "own_bytes": 0, "dead_bytes": 0,
                "peer_bytes": 0, "unattributed_bytes": scaled_total,
                "accountable_bytes": scaled_total, "peers": tuple(peers or ()),
                "divisor": 1, "source": SHARE_SOURCE_NO_DIGEST,
                "sheddable_entries": entries, "peer_entries": 0,
                # No fingerprint means the gate has no digest to compare against, so
                # this rank claims NOTHING about who owns any file here and the
                # planner's own `refused` classification is what stops every unlink
                # downstream (counted there as `skipped`). Inventing an attribution
                # here would be the guess this branch exists to avoid.
                "unattributable_entries": 0, "unattributable_bytes": 0}
    own = dead = peer = unattributed = 0
    sheddable: list = []
    peer_entries = 0
    unattributable_entries = 0
    for row in entries:
        size = int(row.get("size_bytes") or 0)
        name = row.get("filename") or os.path.basename(str(row.get("path") or ""))
        # ONE `parse_path` per entry, shared with the classification the attribution
        # derives: the walk can hand this loop 20,000 entries inside a 1.5 s pass
        # budget, and a redundant parse of a name is ~80 ms of that budget spent on
        # nothing. `attribute_name` asks `classify_name` the read-question and adds
        # the eviction-question on top of it, so this is one call, not two verdicts
        # that could disagree.
        owner = disk_layout.attribute_name(
            name, digest, peers,
            parsed=disk_layout.parse_path(name),
            peers_known=peers_known)["attribution"]
        if owner == disk_layout.ATTRIBUTION_OWN:
            own += size
            sheddable.append(row)
        elif owner == disk_layout.ATTRIBUTION_DEAD:
            dead += size
            sheddable.append(row)
        elif owner == disk_layout.ATTRIBUTION_PEER:
            peer += size
            peer_entries += 1
        else:
            # THE GATE'S OWN BUCKET: counted as capacity, never handed to the planner.
            unattributed += size
            unattributable_entries += 1
    return {"attributed": True,
            "own_bytes": _scale_to_store(own, used, scaled_total),
            "dead_bytes": _scale_to_store(dead, used, scaled_total),
            "peer_bytes": _scale_to_store(peer, used, scaled_total),
            "unattributed_bytes": _scale_to_store(unattributed, used, scaled_total),
            "accountable_bytes": _scale_to_store(own + dead + unattributed, used,
                                                 scaled_total),
            "peers": tuple(peers or ()), "divisor": 1 + len(peers or ()),
            "source": SHARE_SOURCE_PEERS if peers else SHARE_SOURCE_NO_PEERS,
            "sheddable_entries": sheddable, "peer_entries": int(peer_entries),
            # THE GATE'S WITNESS, in the same scaled currency as the byte buckets
            # above: how many files this rank could have shed and deliberately did
            # not, and what they were worth. `unattributed_bytes` above already
            # carries the bytes; this is the ENTRY COUNT, which is what a plan that
            # came up short needs an operator to be able to see.
            "unattributable_entries": int(unattributable_entries),
            "unattributable_bytes": _scale_to_store(unattributed, used,
                                                    scaled_total)}


def _usable_width(value) -> int | None:
    """A block width this module is willing to do arithmetic with, or None.

    One rule for a rank's own width and a peer's, so the two can never be held to
    different standards by the same function. None means "unknown", never "zero": a
    zero width would make `share / width` a division by nothing and a weighted share
    a division by an under-count.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0 or value > MAX_PEER_WIDTH_BYTES:
        return None
    return int(value)


def _peer_widths_for_share(peers):
    """EVERY live peer's published block width, in peer order; None if one is missing.

    Reads the identity view this pass has already taken (`_peer_liveness()` calls it a
    few lines above, and it is cached for `PEER_IDENTITY_CACHE_SECONDS`), so this costs
    no extra file I/O - it is a dictionary lookup into documents this rank already
    opened and parsed.

    The all-or-nothing return is the whole design: a sum of widths that is missing a
    rank is SMALLER than the true sum, which makes every rank's share too BIG - the one
    direction that over-evicts a shared store. So a missing width does not buy an
    estimate, it buys the pre-0.2.58 equal-byte split and a `share_basis` that says so.
    """
    live = [str(digest).strip().lower() for digest in (peers or ())]
    if not live:
        return ()
    published = _peer_identity_view().get("peer_widths") or {}
    out = []
    for digest in live:
        width = _usable_width(published.get(digest))
        if width is None:
            return None
        out.append(width)
    return tuple(out)


def _share_of_ceiling(ceiling, divisor: int, source: str,
                      own_width=None, peer_widths=None) -> dict:
    """This rank's slice of a ceiling that `divisor` ranks share. Pure: pinnable.

    `divisor == 1` returns the ceiling and the target UNCHANGED, byte for byte, which is
    what makes the single-rank case (one rank on the box, or a boot that cannot see a
    sibling) identical to every release before this one. The floor division is
    deliberate: a share rounded UP would hand N ranks a combined budget larger than the
    ceiling they were configured with.

    EQUAL IN BLOCKS, NOT IN BYTES (0.2.58). With `own_width` and one width per live peer
    supplied, the split is done in blocks first:

        blocks    = ceiling // sum(widths)          # whole blocks EVERY rank can hold
        share_i   = blocks * width_i                # that many of MY blocks, in bytes

    which is the width-weighted form `ceiling * width_i / sum(widths)` with the floor
    taken ONCE, on the shared block count, so the four shares are the same number of
    blocks by construction and `sum(share_i) <= ceiling` still holds. Why bytes are the
    wrong currency: a restore is only servable when the block is present on EVERY rank,
    so the tier's useful capacity is MIN blocks across ranks, and an equal-byte split on
    the live box's widths (PP0 11,468,800 B, PP1 12,902,400 B) gave the PP0 ranks 4,493
    blocks against PP1's 3,994 - 499 PP0 blocks per rank that occupy disk and can never
    be served. Measured retention-vs-claims was PP0 0.886 / PP1 0.776, which is that
    3,994/4,493 ratio to within a rounding error.

    WITHOUT the widths (`own_width` missing, or any live peer's width unpublished) this
    returns EXACTLY the equal-byte share it always did, with `share_blocks` derived from
    the width where one is known and `share_basis` saying `equal_bytes`. Guessing a
    missing width would size a share from a number nobody measured, and a sum that is
    missing a rank makes every share too big - the over-eviction direction.
    """
    safe = max(1, int(divisor))
    ceiling = int(ceiling)
    width = _usable_width(own_width)
    widths = None
    if width is not None and safe >= 2 and peer_widths is not None:
        peers = [_usable_width(item) for item in peer_widths]
        if len(peers) == safe - 1 and all(item is not None for item in peers):
            widths = [item for item in peers if item is not None]
    blocks = None
    basis = SHARE_BASIS_EQUAL_BYTES
    share = 0
    if widths is not None:
        total = width + sum(widths)
        whole = ceiling // total
        if whole >= 1:
            # One floor, taken on the quantity every rank must agree on.
            share = whole * width
            blocks = whole
            basis = SHARE_BASIS_BLOCK_EQUAL
    if blocks is None:
        share = int(ceiling // safe)
        if share <= 0:
            # A ceiling smaller than the rank count is a misconfiguration, not a licence
            # to evict everything: 1 byte is a budget, 0 means "delete the store". The
            # same rule keeps a ceiling narrower than one block from yielding a 0-byte
            # share, and keeps the block-equal path from ever being entered at all.
            share = 1
        elif width is not None:
            # DERIVED, not promised: what this byte budget can actually hold. Read
            # together with `share_basis`, this is the field that shows the asymmetry
            # the fix removes (4,493 / 4,493 / 3,994 / 3,994 before, 3,994 x 4 after).
            blocks = share // width
    return {"ceiling_bytes": int(ceiling), "divisor": safe,
            "share_bytes": share,
            "target_bytes": int(share * EVICT_TARGET_FRACTION),
            # HOW MANY WHOLE BLOCKS `share_bytes` is worth on this rank (None when no
            # width is known here), the width that priced it, how many live peers
            # published one, and which arithmetic produced the budget.
            "share_blocks": blocks, "own_width_bytes": width,
            "peer_widths_known": len(widths) if widths is not None else 0,
            "share_basis": basis, "source": source}


def _apply_attribution(attribution: dict, decision_bytes: int,
                       share: dict) -> None:
    """Publish WHOSE bytes these are, so the next trigger and every payload agree.

    ONE place, because the trigger (`_ceiling_trigger`) and the decision
    (`_run_ceiling_pass`) must read the same split; two writers of these counters is how
    a share and a measurement start disagreeing.
    """
    _state["estimated_bytes"] = int(decision_bytes)
    _state["bytes_attributed"] = bool(attribution["attributed"])
    _state["estimated_own_bytes"] = int(attribution["own_bytes"])
    _state["estimated_foreign_dead_bytes"] = int(attribution["dead_bytes"])
    _state["estimated_peer_live_bytes"] = int(attribution["peer_bytes"])
    _state["estimated_unattributed_bytes"] = int(attribution["unattributed_bytes"])
    _state["share_bytes"] = int(share["share_bytes"])
    # THE SHARE'S SHAPE, not just its size (0.2.58): the block count, the width that
    # priced it and the basis, so a reader can check four ranks for block-equality
    # without knowing any of their geometries.
    _state["share_blocks"] = share.get("share_blocks")
    _state["share_own_width"] = share.get("own_width_bytes")
    _state["share_width_peers"] = int(share.get("peer_widths_known") or 0)
    _state["share_basis"] = str(share.get("share_basis")
                                or SHARE_BASIS_EQUAL_BYTES)
    _state["share_divisor"] = int(share["divisor"])
    _state["share_source"] = str(share["source"])
    _state["share_target_bytes"] = int(share["target_bytes"])


def _accountable_bytes() -> int:
    """The bytes THIS rank is accountable for. Never a walk, never a syscall.

    Before the first attribution the rank cannot tell its own bytes from a store it did
    not write, so it reports the store-wide number - the pre-0.2.30 behaviour, and the
    eager direction.
    """
    if not _state["bytes_attributed"]:
        return int(_state["estimated_bytes"])
    return (int(_state["estimated_own_bytes"])
            + int(_state["estimated_foreign_dead_bytes"])
            + int(_state["estimated_unattributed_bytes"]))


def _ceiling_trigger() -> dict:
    """Should a pass run, and against which number? Reads counters only, no walk.

    TWO clauses, and the second is the whole reason this cannot regress capacity safety:
      * `over_share` - the new fair share: my accountable bytes reached my slice.
      * `over_store_ceiling` - the pre-0.2.30 rule, kept verbatim: the store-wide
        estimate reached the WHOLE ceiling. With no peers learned the share IS the
        whole ceiling, so the two clauses coincide; with peers learned the first one
        fires strictly EARLIER. A pass can therefore only ever start sooner than it did
        in every release before this one, never later.
    """
    ceiling = _state["ceiling_bytes"]
    if not ceiling:
        return {"run": False, "reason": "no ceiling configured"}
    accountable = _accountable_bytes()
    share = (int(_state["share_bytes"]) if (_state["bytes_attributed"]
                                            and _state["share_bytes"])
             else int(ceiling))
    over_share = accountable >= share
    over_store = int(_state["estimated_bytes"]) >= int(ceiling)
    return {"run": bool(over_share or over_store),
            "ceiling_bytes": int(ceiling), "share_bytes": share,
            "accountable_bytes": accountable, "over_share": over_share,
            "over_store_ceiling": over_store,
            "reason": ("fair share reached" if over_share else
                       ("store-wide ceiling reached" if over_store else
                        "under the fair share and the store-wide ceiling"))}


def mirror_headroom(width_bytes: int | None = None) -> dict:
    """Would ONE more mirrored slot take this rank past its ceiling? A QUESTION, not a
    decision: it counts nothing, arms nothing and refuses nothing. The caller - the
    mirror door in `disk_store` - decides and counts.

    WHY THE DOOR EXISTS AT ALL. The live store sits at 961 GB of a 1 TB soft ceiling and
    the filesystem behind it has ~400 GB free, so the one way mirror mode becomes an
    outage is a mirror that outruns the evictor and fills `/home`. The evictor is not
    the guard: it runs AFTER a committed write, so an offer that is already queued is
    too late to ask. This is the check that happens BEFORE the bytes are copied.

THE CURRENCY IS THE ONE THE CEILING ITSELF DECIDES ON, so the door and the evictor
cannot disagree about where the store stands: `_accountable_bytes()` (this rank's own
bytes plus the dead-geometry and unattributable bytes it is accountable for, or the
store-wide estimate before any attribution - the eager direction, deliberately) plus
`_state["queue_bytes"]`, the snapshotted bytes already queued and therefore already
promised. Omitting the queued half would let 256 jobs x 11.4 MB walk past a ceiling they
had already spent.

    A MISSING OR UNPARSEABLE CEILING MEANS NO CEILING, and this inherits that verdict
    from `capacity_ceiling_bytes()` rather than restating it: "0" would mean "evict
    everything you see", so a typo must never become a destructive reading - and here it
    would additionally mean "refuse every mirror offer", which is a typo turning the
    tier off. `basis` names which number was used, because a refusal a reader cannot
    reconstruct is a refusal he has to re-derive by hand.
    """
    out = {"ok": True, "reason_kind": None, "reason": None,
           "ceiling_bytes": None, "accountable_bytes": 0, "queued_bytes": 0,
           "width_bytes": None, "projected_bytes": 0, "basis": "no ceiling configured"}
    try:
        ceiling = capacity_ceiling_bytes()
        out["ceiling_bytes"] = ceiling
        if not ceiling:
            return out
        width = int(width_bytes) if isinstance(width_bytes, int) \
            and not isinstance(width_bytes, bool) and width_bytes > 0 else 0
        accountable = int(_accountable_bytes())
        queued = int(_state["queue_bytes"])
        out["width_bytes"] = int(width_bytes) if width else None
        out["accountable_bytes"] = accountable
        out["queued_bytes"] = queued
        out["projected_bytes"] = accountable + queued + width
        out["basis"] = ("accountable+queued+width vs ceiling_bytes"
                        if width else
                        "accountable+queued vs ceiling_bytes (no width given)")
        if out["projected_bytes"] > int(ceiling):
            out["ok"] = False
            out["reason_kind"] = "ceiling"
            out["reason"] = (
                f"one more mirrored slot would put this rank at "
                f"{out['projected_bytes']} B accountable+queued against a "
                f"{ceiling} B ceiling (accountable {accountable} B, already "
                f"queued {queued} B, this slot {width} B): the offer is refused "
                f"at the door and no byte is copied")
        return out
    except Exception as exc:  # noqa: BLE001 - a headroom question is never a fault
        snapshot._error(exc, "disk_spill mirror headroom")
        out["ok"] = True
        out["basis"] = f"unreadable ({type(exc).__name__}): no ceiling is enforced"
        return out


def _ceiling_budget_seconds() -> dict:
    """The pass's time budget in force: the operator's, else the shipped default.

    Read per pass, like this module's other env readers (`disk_root`,
    `capacity_ceiling_bytes`), so a test's monkeypatch and an operator's drop-in both
    take effect without a restart. An unusable value is NOT a second default here -
    `config.ceiling_env` already refused it and said why - it is the constant, which
    is the behaviour the release shipped with.
    """
    problems: list = []
    budget = CEILING_PASS_BUDGET_SECONDS
    try:
        env = config.ceiling_env()
        problems = list(env.get("problems") or [])
        if env.get("budget_seconds") is not None:
            budget = float(env["budget_seconds"])
    except Exception as exc:  # noqa: BLE001 - a budget is never worth a fault
        snapshot._error(exc, "disk_spill ceiling budget")
        problems = [f"the environment could not be read: {type(exc).__name__}"]
    clamped = budget > CEILING_BUDGET_SECONDS_MAX
    if clamped:
        budget = CEILING_BUDGET_SECONDS_MAX
    return {"seconds": budget, "clamped": clamped, "problems": problems}


def _deadline_expired(deadline) -> bool:
    """Has the pass's budget run out? `None` (no budget) never expires.

    ONE place, so the count pass, the measure pass and the unlink loop all answer the
    same question with the same clock. `time.monotonic()`: a wall-clock adjustment
    must never be able to stretch or cut a pass that is already running.
    """
    return deadline is not None and time.monotonic() >= deadline


def _enforce_ceiling() -> None:
    """Shed files, oldest TOUCH first, when the estimate crosses the ceiling.

    THE NUMBER THIS DECIDES ON IS AN ESTIMATE OF THE WHOLE STORE, not the bytes the
    walk happened to `stat`. A walk that hits `MAX_WALK_FILES` reports `truncated`
    and a `used_bytes` that is a FLOOR, and the first version of this function wrote
    that floor straight back into `estimated_bytes`. Both halves of the ceiling then
    break together, in the direction that fills the disk.

    Measured on the live shape (29,532 files, 326.2 GiB of slot bytes, a 192 GiB
    ceiling): the old walk stat-ed 19,804 files and reported **218.7 GiB** - a floor
    1/3 under the truth, because it read the FIRST shard directories and stopped. A
    floor does not just delay the trigger, it RESCALES it: the ceiling this function
    actually enforced was `192 GiB / 0.671` = **286 GiB**, and the divisor grows as
    the store does (at 40,000 files it is 2x, at 100,000 it is 5x), so the bigger the
    store got the more headroom the ceiling quietly gave it. And the second half is
    worse than a late trigger: a file the walk never opened cannot enter an eviction
    plan, so the ~1/3 of the key space it never reached was an UNREACHABLE SINK - it
    took spills forever and shed nothing, which is why the oldest surviving file was
    2.8 hours old while the store grew 145 GiB an hour. Two rules follow, and both
    are load-bearing:

      1. the decision uses `walk["estimated_total_bytes"]` - the sample scaled by
         `total_files / files`. It is identical to `used_bytes`, byte for byte, for
         any walk that saw every file, which is every store inside the caps; and
      2. a truncated walk's plan is allowed to be PARTIAL. It sheds the oldest files
         it COULD see toward the target and returns; enforcement runs after every
         committed write, the next pass samples a store whose directory slots were
         just freed by the deletes, and the passes converge. One pass was never
         going to be enough on a store this far over, and pretending otherwise is
         what made the trigger silent rather than merely slow.

    The estimate is per-process and monotonic-increasing until a walk replaces it
    with a measurement, so the walk only happens at the moment capacity actually
    matters. `disk_layout.lru_eviction_plan` owns the ORDER (foreign digests, then
    oldest mtime); executing it here is the only destructive action in this module,
    and deleting a file is always safe: the cost is a recompute, never a wrong
    answer - there is no read path that could serve a file we deleted. NOTHING here
    relaxes `execute_unlink_plan`'s guards: it still re-validates every path against
    the CURRENT digest, still needs `for_capacity` to touch this boot's own files,
    still caps its unlinks per directory, and still deletes only what it proved it
    may delete. A partial plan is a SMALLER destructive set, never a wider one, and
    every file it deletes is journalled (`_journal_write(..., removal=True)`) - an
    un-journalled removal makes the index offer a dead file, which costs a reported
    load failure instead of a plain recompute.

    WHAT CHANGED IN 0.2.30: the number above is no longer the number the ceiling is
    compared against. Four ranks sharing ONE root each measured the WHOLE store (156
    GiB), each compared it against the WHOLE ceiling (148 GiB) and each shed down to
    0.9 x that target, so one pass's worth of relief was paid FOUR times over: 7,956
    files evicted and 156 GiB -> 68 GiB where 23 GiB was the plan. A shared store
    enforced that way has an effective capacity of `ceiling / rank_count`. A rank now
    sizes only the bytes it is ACCOUNTABLE for - its own digest's files plus the dead
    geometry no live rank owns - against `ceiling / n_effective`, where `n_effective`
    is 1 plus the live peer digests this rank's walks have learned (0.2.29 already
    learns them, so the rank count comes from work the pass was doing anyway). See
    `_attribute_walk_bytes`, `_share_of_ceiling` and `_ceiling_trigger`; the fail-safe
    direction (no peers learned => the whole ceiling, exactly as before) and its worst
    case are documented on the constant block above.
    """
    ceiling = _state["ceiling_bytes"]
    if not ceiling:
        return
    # THE GATE, and it is a fair share now rather than the store-wide total. The
    # store-wide clause is kept INSIDE the trigger, so this can only ever run a pass
    # SOONER than every release before 0.2.30 did, never later.
    if not _ceiling_trigger()["run"]:
        return
    root = _state["root"]
    if not root:
        return
    # THE PASS'S DEADLINE, started here - after the gates, so a pass that was going to
    # do nothing does not spend a clock - and honoured by BOTH halves: the walk (which
    # stops sampling) and the unlink loop (which stops between whole per-directory
    # plans). It is a bound on the WORK, not a cancellation: whatever was measured is
    # still a measurement and whatever was unlinked was journalled.
    started = time.monotonic()
    budget = _ceiling_budget_seconds()
    deadline = started + float(budget["seconds"])
    _state["ceiling_budget_seconds"] = round(float(budget["seconds"]), 6)
    # False at the start of every pass: the witness must describe THIS pass, not last
    # pass's timeout. The two places that set it True are inside the pass (the walk
    # sampling, the unlink loop), which is where the fact is actually discovered.
    _state["ceiling_deadline_hit"] = False
    for problem in budget["problems"]:
        _state["last_reason"] = str(problem)[:MAX_STRING_CHARS]
    try:
        _run_ceiling_pass(root, int(ceiling), deadline, budget)
    finally:
        # Published on EVERY exit - including the early returns inside - because a
        # witness that only described the passes that reached the end is a witness for
        # the happy path only.
        _state["ceiling_pass_seconds"] = round(time.monotonic() - started, 6)
        if _state["ceiling_deadline_hit"]:
            _state["ceiling_deadline_hits"] = (
                int(_state["ceiling_deadline_hits"]) + 1)


# --- the unlink pool: the ceiling pass's hands (0.2.29) ----------------------
#
# WHY A POOL HERE AND NOT A BIGGER QUEUE. The pass unlinks up to `MAX_EVICTION_PLAN`
# files, grouped by shard directory, and until now it did them one at a time on the
# spill writer thread. `unlink()` is a metadata operation: on a journaling filesystem
# the interesting part is that several deletions can be IN FLIGHT at once instead of
# each one being waited for, which is exactly what `os.unlink` in N threads gives you
# and what a serial loop cannot. No direct I/O is involved anywhere in this tier and
# none is added here: the read path is plain buffered `open()`/`readinto` and that is
# deliberate (this store reads at 25.4 GB/s from the page cache against 6.65 GB/s
# cold, so the page cache is the tier's biggest asset and `O_DIRECT` would discard it).
#
# WHY THE PLAN IS PARTITIONED AND NOT SHARED. One path is attempted by exactly one
# worker: the plan is split by shard directory (the grouping the serial loop already
# had, which is also the right shape for the filesystem) and each batch is handed to
# one worker whole. Nothing races on a path, so `evictions`/`evicted_bytes` cannot
# double-count, and the pass computes the SAME bytes for 1 worker and for 8.
#
# WHY THE WORKERS DO NO BOOKKEEPING. A worker calls `disk_layout.execute_unlink_plan`
# (pure: validate, unlink, return its own report) and touches NO module state. Every
# `_state` update and every journal write happens on the spill thread that collected
# the results, in submission order. That is what keeps this change from turning the
# writer thread's single-writer counters into a race, and it is why the journalling
# order is unchanged whether the unlinks ran in parallel or not.
MAX_EVICT_THREADS = 8                # workers one process may build
EVICT_THREADS_DEFAULT = 1            # 1 = today's serial behaviour, measured first
EVICT_THREAD_PREFIX = "kvtier-disk-evict"

_evict_pool: ThreadPoolExecutor | None = None
_evict_pool_size = 0
_evict_pool_lock = threading.Lock()


def evict_thread_count_alive() -> int:
    """How many `kvtier-disk-evict` worker threads exist in this process right now.

    A count, not a bool, and read from the thread table rather than from a counter:
    the same posture as `disk_read.thread_alive_count()`, so a payload cannot report
    workers that a shutdown already took away.
    """
    try:
        return sum(1 for thread in threading.enumerate()
                   if thread.name.startswith(EVICT_THREAD_PREFIX)
                   and thread.is_alive())
    except Exception:  # noqa: BLE001 - a witness is never worth a fault
        return 0


def _evict_worker_count() -> dict:
    """The worker count to build, from the environment, clamped and explained."""
    wanted = EVICT_THREADS_DEFAULT
    problems: list = []
    try:
        env = config.ceiling_env()
        problems = list(env.get("problems") or [])
        if env.get("evict_threads") is not None:
            wanted = int(env["evict_threads"])
    except Exception as exc:  # noqa: BLE001 - a thread count is never worth a fault
        snapshot._error(exc, "disk_spill evict threads")
        problems = [f"the environment could not be read: {type(exc).__name__}"]
    clamped = wanted > MAX_EVICT_THREADS
    configured = max(1, min(MAX_EVICT_THREADS, wanted))
    return {"requested": wanted, "configured": configured, "clamped": clamped,
            "problems": problems}


def _get_evict_pool() -> ThreadPoolExecutor | None:
    """The process's unlink pool, built on first use. ONE pool, never rebuilt.

    Built lazily because a boot that never crosses its ceiling must not start threads
    it never uses, and built ONCE because a pool is a thread-count commitment:
    rebuilding it per pass would pay thread creation inside the pass the pool exists
    to shorten and would make the reported count mean nothing. If the operator changes
    the env var, the payload shows `evict_threads_requested` against
    `evict_threads_configured` and the next restart honours it.

    None means "no pool": the caller runs the batch inline on the spill thread, which is
    today's behaviour and the fallback for a machine that will not start a thread.
    """
    global _evict_pool, _evict_pool_size
    with _evict_pool_lock:
        if _evict_pool is not None:
            return _evict_pool
        want = _evict_worker_count()
        _state["evict_threads_requested"] = int(want["requested"])
        for problem in want["problems"]:
            _state["last_reason"] = str(problem)[:MAX_STRING_CHARS]
        if want["requested"] <= 1:
            # The serial default. Recorded so "configured 1" is a fact a payload can
            # distinguish from "the pool failed to build" (`evict_fallbacks`).
            _state["evict_threads_configured"] = 1
            _state["evict_threads_alive"] = 0
            return None
        try:
            _evict_pool = ThreadPoolExecutor(
                max_workers=want["configured"],
                thread_name_prefix=EVICT_THREAD_PREFIX)
            _evict_pool_size = want["configured"]
        except Exception as exc:  # noqa: BLE001 - no pool means serial, not no eviction
            snapshot._error(exc, "disk_spill evict pool")
            _state["evict_fallbacks"] += 1
            _state["last_error"] = (
                f"evict pool: {type(exc).__name__}")[:MAX_STRING_CHARS]
            _evict_pool = None
            _evict_pool_size = 0
            _state["evict_threads_configured"] = 1
            return None
        _state["evict_threads_configured"] = int(_evict_pool_size)
        _state["evict_threads_alive"] = evict_thread_count_alive()
        return _evict_pool


def shutdown_evict_pool() -> None:
    """Drop the pool, waiting for every submitted unlink. For a reset and shutdown.

    Waits, never cancels: an unlink that was submitted has to be tracked, and a file
    that vanishes without the journal hearing about it is the exact index error this
    tier treats as its worst outcome (it costs a reported load failure rather than a
    plain recompute). The pool is then dropped so the next boot re-reads the count.
    """
    global _evict_pool, _evict_pool_size
    with _evict_pool_lock:
        pool = _evict_pool
        _evict_pool = None
        _evict_pool_size = 0
    if pool is not None:
        try:
            pool.shutdown(wait=True, cancel_futures=False)
        except Exception as exc:  # noqa: BLE001 - a shutdown is best-effort
            snapshot._error(exc, "disk_spill evict pool shutdown")
    _state["evict_threads_configured"] = 0
    _state["evict_threads_alive"] = evict_thread_count_alive()


def _run_evict_batches(batches: list, *, peers, peers_known: bool = True,
                       deadline, protected=frozenset()) -> tuple:
    """Unlink every batch, on the pool if there is one. Returns per-batch reports.

    `batches` is a list of `(directory, paths)` pairs, DISJOINT by construction (they
    came from grouping one plan by `os.path.dirname`), which is what makes "no path is
    attempted twice" a structural fact rather than a hope.

    Returns `(reports, attempted, truncated_by_deadline)` where `reports` is a list of
    `(directory, execute_unlink_plan_report | None)` IN SUBMISSION ORDER, `attempted`
    the number of paths actually handed to an executor, and the third whether the pass's
    TIME budget stopped the submission.

    THE DEADLINE AND THE ORDERING THAT MAKES TRUNCATION SAFE. The budget is checked
    BEFORE A SUBMISSION, never after: a batch that has been submitted is ALWAYS awaited
    and ALWAYS collected, because a submitted-but-uncollected unlink is a deleted file
    the journal never heard about. So "stop early" means "stop handing out new work",
    and the in-flight wave is awaited whole. Work advances in waves of at most
    `pool size` batches, each wave fully collected before the next is considered, which
    bounds both the futures alive at once and the amount of work that can outrun the
    deadline (one wave, each batch capped at `MAX_UNLINK_PLAN` unlinks).

    A worker NEVER touches module state. It calls `execute_unlink_plan`, which
    validates and unlinks and returns its own report; every counter and every journal
    write happens in the caller, on the spill thread, in submission order - so the
    journalled removal
    order is the plan order whether the unlinks ran in parallel or not, and the writer
    thread's counters stay single-writer.
    """
    reports: list = []
    attempted = 0
    truncated = False
    pool = _get_evict_pool()
    wave = max(1, int(_evict_pool_size)) if pool is not None else 1
    index = 0
    while index < len(batches):
        if pool is None:
            # No pool: inline, one batch at a time, deadline honoured between batches.
            # The FIRST batch always runs, exactly as the serial loop always did, so a
            # pass that was cut off still releases bytes.
            directory, paths = batches[index]
            index += 1
            attempted += len(paths)
            _state["evict_batches_serial"] += 1
            reports.append((directory, _unlink_batch(directory, paths, peers,
                                                     peers_known,
                                                     protected=protected)))
            if _deadline_expired(deadline):
                truncated = bool(batches[index:])
                break
            continue
        chunk = batches[index:index + wave]
        index += len(chunk)
        futures: list[tuple[str, Future, int]] = []
        for directory, paths in chunk:
            try:
                futures.append((directory,
                                pool.submit(_unlink_batch, directory, paths, peers,
                                            peers_known, protected),
                                len(paths)))
            except Exception as exc:  # noqa: BLE001 - no pool means inline, not no work
                snapshot._error(exc, "disk_spill evict submit")
                _state["evict_fallbacks"] += 1
                attempted += len(paths)
                _state["evict_batches_serial"] += 1
                reports.append((directory, _unlink_batch(directory, paths, peers,
                                                          peers_known,
                                                          protected=protected)))
        if futures:
            _state["evict_batches_submitted"] += len(futures)
            _state["evict_peak_in_flight"] = max(
                int(_state["evict_peak_in_flight"]), len(futures))
        attempted += sum(size for _d, _f, size in futures)
        # THE WAIT. In submission order, so `reports` is in plan order whatever order
        # the unlinks finished in. `future.result()` re-raises a worker's exception on
        # this thread; `_unlink_batch` catches everything itself, so the only thing that
        # can arrive here is a failure of the pool machinery, which is reported as a
        # batch that did nothing rather than being allowed to break the pass.
        for directory, future, _size in futures:
            try:
                reports.append((directory, future.result()))
            except Exception as exc:  # noqa: BLE001 - a lost worker loses its batch
                snapshot._error(exc, "disk_spill evict batch")
                _state["evict_files_failed"] += 1
                reports.append((directory, None))
        if index < len(batches) and _deadline_expired(deadline):
            truncated = True
            break
    return reports, attempted, truncated


def _unlink_batch(directory, paths, peers, peers_known: bool = True,
                  protected=frozenset()) -> dict | None:
    """ONE directory's unlink batch, on whatever thread owns it. Never raises.

    `execute_unlink_plan` re-validates every path against the CURRENT digest and the
    peer digests before it deletes anything, and a directory that is itself a symlink is
    refused outright; none of that is weakened by running it on a worker. The peer list
    and the `peers_known` flag are handed in, so a worker never reads the mutable
    witness
    map or the identity cache - which is also what keeps a pass self-contained: the
    decision and the deletion answer to the SAME view of who is alive.
    """
    try:
        return disk_layout.execute_unlink_plan(
            directory, paths, dry_run=False,
            geometry_digest=_state["geometry_digest"], peer_digests=peers,
            for_capacity=True, peers_known=peers_known, protected=protected)
    except Exception as exc:  # noqa: BLE001 - a worker that raises loses its own batch
        snapshot._error(exc, "disk_spill unlink batch")
        return None


def _owner_of_name(path, peers) -> str:
    """Which owner's bucket one deleted path came out of: own / dead / peer / other.

    Name-only, so it costs a parse and nothing else, and it is the same classification
    the planner used a moment earlier. Used for the post-pass bookkeeping only: the
    decision was already made against `_attribute_walk_bytes`'s split.
    """
    try:
        name = os.path.basename(str(path or ""))
        kind = disk_layout.classify_name(name, _state["geometry_digest"],
                                        peers)["classification"]
    except Exception:  # noqa: BLE001 - a bookkeeping helper never breaks a pass
        return "unattributed"
    if kind == disk_layout.NAME_CURRENT:
        return "own"
    if kind in (disk_layout.NAME_STALE_GEOMETRY, disk_layout.NAME_STALE_KEY_SHAPE):
        return "dead"
    if kind == disk_layout.NAME_PEER_RANK:
        return "peer"
    return "unattributed"


def _foreign_digest_of(path):
    """The geometry digest in this name, if it is NOT this rank's. Else None.

    THE ATTRIBUTIBILITY OF ONE REMOVAL. The question a reader of the journal cannot
    answer from a `-` line: a removal by rank A of a file whose digest is rank B's is
    the event that leaves the index claiming an identity that physically exists only
    under somebody else's name, and after it happens the next restore hits a
    `file_missing` hole in the middle of a prefix. So the tier counts it, in a
    counter, on the thread that did the deleting - not as a log line a busy fill test
    would drown, and NOT as a change to the journal's wire format, whose five fields
    (`seq op key16 size group`) the restore index parses and which no reader can
    grow mid-flight. Name-only: one parse, no syscall. None means "this name is this
    boot's, carries no digest, or could not be read" - a bookkeeping helper never
    invents a foreign digest out of a parse it could not finish.
    """
    try:
        parsed = disk_layout.parse_path(os.path.basename(str(path or "")))
        seen = parsed.get("geometry_digest")
        own = _state["geometry_digest"]
    except Exception:  # noqa: BLE001 - a bookkeeping helper never breaks a pass
        return None
    if not isinstance(seen, str) or not seen:
        return None
    if not isinstance(own, str) or not own:
        return None
    lowered = seen.lower()
    return None if lowered == own.lower() else lowered


def _peer_digest_set(peers) -> set:
    """The confirmed live peer digests, lower-cased, as a set. Never raises."""
    try:
        return {str(digest).lower() for digest in (peers or ())}
    except Exception:  # noqa: BLE001 - a bookkeeping helper never breaks a pass
        return set()


def _removal_justification(path, peers, peers_known) -> str:
    """WHICH RULE licensed one unlinked file, as one closed-vocabulary name.

    The operator's question, from the live box: `peer_files_removed = 2955` with
    `peer_files_removed_known_peer = 0` cannot tell "I deleted 38 GB of a live sibling's
    current bytes" apart from "I deleted 38 GB of bytes an arena-size change orphaned".
    Those are different facts and the second one is CORRECT cleanup, so every
    removal now
    also names its justification, using `disk_layout.attribute_name` - the SAME function
    the gate used a moment earlier, so a justification can never disagree with the
    decision that made the file eligible:

      * `stale_geometry`  - a foreign digest no live rank publishes and no live rank is
                            writing: the
                            configuration-change case.
                            Shed-first by design.
      * `stale_key_shape` - the pre-group key shape, this plugin's own unloadable
      lineage.
      * `peer`            - a digest a LIVE rank owns. Structurally unreachable for a
                            capacity pass since 0.2.45 (a peer's entries never enter the
                            destructive set), so a non-zero count here is an incident.
      * `own`             - this rank's own digest (not a cross-rank removal at all).
      * `unnamed`         - a name carrying no readable digest, or one this tier cannot
                            classify. Counted so the totals balance and never guessed.

    Name-only: one parse, no syscall.
    """
    try:
        name = os.path.basename(str(path or ""))
        verdict = disk_layout.attribute_name(
            name, _state["geometry_digest"], peers, peers_known=peers_known)
        attribution = verdict["attribution"]
        if attribution == disk_layout.ATTRIBUTION_OWN:
            return "own"
        if attribution == disk_layout.ATTRIBUTION_PEER:
            return "peer"
        if verdict["classification"] == disk_layout.NAME_STALE_KEY_SHAPE:
            return "stale_key_shape"
        if attribution == disk_layout.ATTRIBUTION_DEAD:
            return "stale_geometry"
        return "unnamed"
    except Exception:  # noqa: BLE001 - a bookkeeping helper never breaks a pass
        return "unnamed"


def _run_ceiling_pass(root, ceiling: int, deadline, budget: dict) -> None:
    """One enforcement pass: walk, decide, shed. Deadline-aware, partial by design.

    Split out of `_enforce_ceiling` so the deadline bookkeeping has exactly one owner
    (the caller's `finally`) and this body reads as the three steps it performs.
    """
    # THE LAST PASS'S GATE WITNESS starts empty. These are per-pass readings, and a
    # value left over from the pass before would be read as this one's - including
    # the two early returns below, which make no decision at all and so must report
    # neither a shortfall nor a withheld set.
    _state["peer_files_planned"] = 0
    _state["peer_files_removed_last_pass"] = 0
    _state["unattributable_entries_last_pass"] = 0
    _state["unattributable_bytes_last_pass"] = 0
    _state["ceiling_plan_short"] = False
    _state["ceiling_plan_short_bytes"] = 0
    _state["ceiling_plan_short_reason"] = None
    _state["grace_entries_last_pass"] = 0
    _state["grace_bytes_last_pass"] = 0
    _state["grace_unlink_refused_last_pass"] = 0
    _state["claim_unlink_refused_last_pass"] = 0
    _state["claim_paths_last_pass"] = 0
    _state["claim_set_truncated"] = False
    walk = _walk_store(root, deadline=deadline)
    if not walk["ok"]:
        _state["last_reason"] = str(walk["reason"])[:MAX_STRING_CHARS]
        return
    if walk.get("deadline_hit"):
        # Say it where the plan's own truncation is recorded, so one field answers
        # "why is this pass partial" and it is never mistaken for the plan cap.
        _state["plan_truncated"] = True
        _state["ceiling_deadline_hit"] = True
    # THE SIBLING DIGESTS, decided NOW and handed to the planner AND the executor:
    # the union of the two liveness witnesses, plus the flag that says whether
    # this rank can name every live rank's identity (`peers_known`). Empty is no
    # longer the fail-safe direction on its own - with `peers_known` false a
    # foreign committed digest is UNATTRIBUTABLE and withheld
    # (`peer_digests_unconfirmed` counts the fact), and with it true a foreign
    # digest is the dead geometry a configuration change left behind, which must
    # stay reclaimable. Since 0.2.30 the same tuple does a second job: it IS the
    # rank count, because `n_effective` is 1 plus the live peers, and that is what
    # turns a 148 GiB ceiling shared by four ranks into four 37 GiB budgets
    # instead of four ranks each defending 148 GiB of one directory.
    peers, peers_known = _peer_liveness()
    # THE DECISION NUMBER. `estimated_total_bytes` equals `used_bytes` whenever the
    # walk measured the whole store, so a store inside the caps cannot tell this
    # from the old code; on a truncated walk it is the sample scaled to the store.
    decision_bytes = int(walk["estimated_total_bytes"])
    if walk["truncated"] and int(walk["files"]) == 0:
        # A bounded walk that stat-ed NOTHING is not a measurement, and the one
        # thing it must never do is write a 0 (or a floor) into the estimate: that
        # is how the ceiling goes quiet forever over a store it cannot see, which is
        # the bug this function exists to fix. So: publish the walk's shape, say why
        # in the reason, keep the estimate where it was, and plan nothing - the next
        # spill asks again. `walk_skipped_files` is the counter that explains it when
        # the store is full of entries this tier will not measure.
        _publish_walk(walk)
        _state["last_reason"] = (
            f"the ceiling could not be measured: {walk['measurement_basis']} after "
            f"{int(walk['entries_examined'])} entries "
            f"({int(walk['skipped_files'])} not measurable by this tier), so the "
            "estimate was kept and nothing was planned")
        return
    # WHOSE BYTES THESE ARE, and therefore what this rank's slice of the ceiling is.
    # This is the half the 2026-10-22 over-eviction was missing: four ranks, one
    # directory, one store-wide number each read as its own problem.
    attribution = _attribute_walk_bytes(walk, peers, peers_known)
    # EQUAL IN BLOCKS, NOT BYTES (0.2.58): the same ceiling, split so that every rank
    # can hold the SAME number of blocks. A restore needs a block on all four ranks, so
    # the surplus blocks an equal-byte split hands a narrow-rank box are disk space
    # holding data that can never be served. The peer widths come from the rank files
    # this pass has already opened for their digests, and a missing one falls back to
    # the equal-byte share rather than a guessed width.
    share = _share_of_ceiling(int(ceiling), attribution["divisor"],
                              attribution["source"],
                              own_width=_slot_width_bytes(),
                              peer_widths=_peer_widths_for_share(
                                  attribution["peers"]))
    accountable = int(attribution["accountable_bytes"])
    target = int(share["target_bytes"])
    _state["ceiling_checks"] = int(_state["ceiling_checks"]) + 1
    _apply_attribution(attribution, decision_bytes, share)
    _state["peer_entries_excluded"] = int(_state["peer_entries_excluded"]) + int(
        attribution["peer_entries"])
    # THE GATE'S OUTCOME, published before any decision is made about it: how many
    # shed candidates this rank withheld because it could not attribute them, and what
    # they were worth. A boot total AND a last-pass reading, because "one unlucky
    # pass" and "this rank has been blind for an hour" are different facts and the
    # operator has to be able to tell them apart from one read of one rank file.
    withheld = int(attribution["unattributable_entries"])
    withheld_bytes = int(attribution["unattributable_bytes"])
    _state["unattributable_entries_last_pass"] = withheld
    _state["unattributable_bytes_last_pass"] = withheld_bytes
    if withheld:
        _state["unattributable_entries_excluded"] = int(
            _state["unattributable_entries_excluded"]) + withheld
    # Publish HOW the decision was made, not just what it was. A reader of
    # `estimated_bytes` must be able to tell "the store is 170 GiB" from "I stat-ed
    # 20,000 of 29,532 files and extrapolated 170 GiB from them": see `section()`.
    _publish_walk(walk)
    if walk["truncated"]:
        _state["last_reason"] = (
            f"ceiling decided on {walk['measurement_basis']}: "
            f"{int(walk['files'])} of {int(walk['total_files'])} file(s) stat-ed, "
            f"{int(walk['used_bytes'])} measured bytes scaled to "
            f"{decision_bytes}")
    if accountable <= target:
        # THE DECLINE, counted and not logged, because this runs on the spill writer
        # thread and the signature it needs to show is arithmetic: "four ranks, one pass
        # did the work, three declined". It means THIS rank's accountable bytes (its own
        # files plus the dead geometry no live rank owns) are already at or under its
        # slice - usually because a sibling's pass got there first, which is precisely
        # the behaviour the fair share exists to produce.
        _state["ceiling_passes_declined"] = int(_state["ceiling_passes_declined"]) + 1
        _state["ceiling_declined_accountable_bytes"] = int(accountable)
        _state["ceiling_declined_target_bytes"] = int(target)
        _state["last_reason"] = (
            f"the pass declined: {accountable} accountable byte(s) "
            f"(own {attribution['own_bytes']} + dead "
            f"{attribution['dead_bytes']}) are at or under this rank's "
            f"{target}-byte share of a {int(ceiling)}-byte ceiling across "
            f"{share['divisor']} rank(s), so nothing was shed")[:MAX_STRING_CHARS]
        return
    plan = disk_layout.lru_eviction_plan(
        attribution["sheddable_entries"],
        geometry_digest=_state["geometry_digest"],
        ceiling_bytes=int(share["share_bytes"]), used_bytes=accountable,
        target_fraction=EVICT_TARGET_FRACTION, peer_digests=peers,
        peers_known=peers_known)
    _state["plan_truncated"] = bool(plan.get("truncated"))
    # THE GRACE'S WITHHOLDING, published before the short/nothing-sheddable branches
    # below, because a pass that withheld its way to a SHORT plan is exactly the pass a
    # reader needs to be able to explain. Counted once per entry: the attribution pass
    # withheld nothing here (it filters by owner, not by clock), so the planner's number
    # is the pass's number and there is nothing to reconcile.
    plan_grace = int(plan.get("excluded_grace") or 0)
    if plan_grace:
        _state["grace_entries_last_pass"] = plan_grace
        _state["grace_bytes_last_pass"] = int(plan.get("excluded_grace_bytes") or 0)
        _state["grace_entries_excluded"] = int(
            _state["grace_entries_excluded"]) + plan_grace
    # The planner ran the same gate over the entries it was handed, so it normally
    # reports nothing further to withhold (the attribution already filtered them).
    # Adding the two is exact and never double-counts: whichever half removed an
    # entry is the half that counted it, and a caller that handed the planner an
    # unfiltered list would still see its exclusions here.
    plan_excluded = int(plan.get("excluded_unattributable") or 0)
    if plan_excluded:
        _state["unattributable_entries_excluded"] = int(
            _state["unattributable_entries_excluded"]) + plan_excluded
        _state["unattributable_entries_last_pass"] = int(
            _state["unattributable_entries_last_pass"]) + plan_excluded
    if not plan["ok"] or not plan["paths"]:
        # Over its share and nothing it is ALLOWED to shed: a different fact from the
        # decline above, and the one case where the filesystem floor (`MIN_FREE_BYTES`,
        # refusal `no_free_space`) is the control that keeps the box alive. Counted, and
        # said, because "the tier wanted to shed and could not" must never be quiet.
        # Since 0.2.34 this is also the shape a BLIND rank takes on a store full of
        # siblings: short by design, over the ceiling by design, and loud.
        _state["ceiling_passes_nothing_sheddable"] = (
            int(_state["ceiling_passes_nothing_sheddable"]) + 1)
        why = str(plan.get("reason") or "no evictable entry named")[:96]
        _state["ceiling_plan_short"] = True
        _state["ceiling_plan_short_bytes"] = max(0, accountable - target)
        # LEADS WITH THE GATE, because this string is stored bounded: a reason whose
        # first 150 characters are arithmetic and whose explanation is the tail is a
        # reason that reads as "nothing was evictable" when the truth is "nothing was
        # ALLOWED". The arithmetic follows the explanation, never precedes it.
        # THE COUNT IS THE ATTRIBUTION'S, not the planner's: `_attribute_walk_bytes`
        # already withheld these entries, so the plan it was handed reports none. A
        # reason that quoted the planner here would say "0 file(s) withheld" over a
        # payload field saying 2 - the one thing this sentence exists to prevent.
        _state["ceiling_plan_short_reason"] = (
            f"UNATTRIBUTABLE: {int(_state['unattributable_entries_last_pass'])} "
            "file(s) withheld (no peer digest confirmed), so the plan named "
            f"nothing and came up short by {max(0, accountable - target)} byte(s)"
        )[:MAX_STRING_CHARS]
        _state["ceiling_passes_short"] = int(_state["ceiling_passes_short"]) + 1
        _state["last_reason"] = (
            f"over the {target}-byte share by {accountable - target} byte(s) but the "
            f"plan named nothing: {why}")[:MAX_STRING_CHARS]
        return
    _state["peer_files_planned"] = int(plan.get("peer_rank") or 0)
    if plan.get("short"):
        # THE SHORT PLAN. It shed what it was allowed to shed and is still over its
        # share, because the rest of the store belongs to names it cannot prove are
        # its own to delete. That is the intended failure mode: the ceiling is
        # exceeded, spilling back-pressures, and `last_reason` says which count was
        # withheld and why. Not an error, not a raise, never a silent no-op.
        _state["ceiling_plan_short"] = True
        _state["ceiling_plan_short_bytes"] = int(plan.get("short_by_bytes") or 0)
        _state["ceiling_plan_short_reason"] = (
            f"UNATTRIBUTABLE: {int(_state['unattributable_entries_last_pass'])} "
            "file(s) withheld (no peer digest confirmed), so the plan shed "
            f"{int(plan.get('planned') or 0)} file(s) and came up short by "
            f"{int(plan.get('short_by_bytes') or 0)} byte(s)")[:MAX_STRING_CHARS]
        _state["ceiling_passes_short"] = int(_state["ceiling_passes_short"]) + 1
        _state["last_reason"] = _state["ceiling_plan_short_reason"]
    by_directory: dict = {}
    for path in plan["paths"]:
        by_directory.setdefault(os.path.dirname(path), []).append(path)
    # THE BATCHES, in plan order. Grouping by directory is both the filesystem-friendly
    # shape (a shard's entries leave one directory) and the partition that guarantees no
    # path is attempted by two workers: the groups are disjoint by construction.
    batches = list(by_directory.items())
    # THE PENDING CLAIMS, read ONCE per pass: the paths this process's reader threads
    # have promised to apply. Same process, so no channel and no new state - the reader
    # is the only witness, and an age window cannot express it (a file being read right
    # now may have been touched a minute ago, or an hour ago on its first service).
    claims = disk_read.inflight_paths()
    _state["claim_paths_last_pass"] = int(claims["count"])
    _state["claim_set_truncated"] = bool(claims["truncated"])
    reports, attempted, cut_off = _run_evict_batches(
        batches, peers=peers, peers_known=peers_known, deadline=deadline,
        protected=claims["paths"])
    freed = 0
    unlinked = 0
    failed = 0
    # FREED BYTES BY OWNER, so the fair-share estimate comes down by the right amount.
    # A rank that shed 10 of its OWN files must not have its `dead` estimate shrink, and
    # a byte shed from a live sibling (which the plan can no longer name) would show up
    # here as a fact rather than as a silent miscount.
    freed_by_owner = {"own": 0, "dead": 0, "peer": 0, "unattributed": 0}
    peer_set = _peer_digest_set(peers)
    for directory, done in reports:
        # COLLECTION, on the spill thread, in submission order. This is where every
        # counter and every journal write happens, so the removal journal is written in
        # plan order whether the unlinks ran on one thread or on eight, and the writer
        # thread's counters stay single-writer.
        if not isinstance(done, dict) or not done.get("ok"):
            # A batch that did not report: nothing was proved deleted, so nothing is
            # journalled and nothing is counted as freed. Counted, never silent.
            failed += 1
            _state["last_reason"] = (
                f"an unlink batch for {os.path.basename(str(directory))} returned no "
                "report; its files were not journalled as removed")[:MAX_STRING_CHARS]
            continue
        for action in done.get("actions") or []:
            if not isinstance(action, dict):
                continue
            outcome = action.get("action")
            if outcome == "refused":
                # THE EXECUTOR'S OWN TWO RULES, counted by NAME rather than by
                # string-matching the sentence (`rule` is the machine-readable tag
                # `execute_unlink_plan` copies from `_validate_unlink`). A plan that
                # named a file and did not delete it is a pass that came in under its
                # target for a reason, and the reason is which of the two rules fired:
                # the clock (grace) or an in-flight read (pending claim).
                rule = action.get("rule")
                if rule == disk_layout.UNLINK_RULE_GRACE:
                    _state["grace_unlink_refused"] = int(
                        _state["grace_unlink_refused"]) + 1
                    _state["grace_unlink_refused_last_pass"] = int(
                        _state["grace_unlink_refused_last_pass"]) + 1
                elif rule == disk_layout.UNLINK_RULE_CLAIM:
                    _state["claim_unlink_refused"] = int(
                        _state["claim_unlink_refused"]) + 1
                    _state["claim_unlink_refused_last_pass"] = int(
                        _state["claim_unlink_refused_last_pass"]) + 1
            if outcome == "unlinked":
                unlinked += 1
                removed_size = int(action.get("size_bytes") or 0)
                freed_by_owner[_owner_of_name(action.get("path"), peers)] += (
                    removed_size)
                # FIX 3: THE REMOVAL'S OWNER, COUNTED. A `-` line in the journal says
                # an identity left the store; it cannot say whose FILE that was, and
                # the difference is the whole cross-rank question (rank A removing a
                # file named under rank B's digest is what leaves the index offering
                # an identity that exists nowhere). So the tier counts it here, per
                # pass and per boot, instead of leaving the next reader to grep
                # 15,848 lines and compare block sizes by hand. `peer_files_removed`
                # with a non-zero `peer_files_planned` is the legitimate case (a
                # KNOWN peer's bucket, reached because nothing else could meet the
                # target); `peer_files_removed` with the plan short of its target is
                # the case to look at.
                foreign = _foreign_digest_of(action.get("path"))
                if foreign is not None:
                    _state["peer_files_removed"] = int(
                        _state["peer_files_removed"]) + 1
                    _state["peer_files_removed_bytes"] = int(
                        _state["peer_files_removed_bytes"]) + removed_size
                    _state["peer_files_removed_last_pass"] = int(
                        _state["peer_files_removed_last_pass"]) + 1
                    if foreign in peer_set:
                        # THE UNAMBIGUOUS ONE. `peer_files_removed` also counts an
                        # old boot's geometry, which may well be this same rank's own
                        # predecessor; this counter has no such ambiguity - the digest
                        # belonged to a rank this walk believed LIVE at the moment the
                        # file was deleted. Non-zero is the incident signature.
                        _state["peer_files_removed_known_peer"] = int(
                            _state["peer_files_removed_known_peer"]) + 1
                    # WHICH RULE PAID FOR IT (0.2.45). `peer_files_removed` alone could
                    # not tell a live sibling's bytes from bytes a configuration change
                    # orphaned, and the two read in opposite directions: one is the
                    # incident, the other is the reclaim path working as designed. The
                    # justification comes from `attribute_name`, the same function the
                    # gate used, so it cannot disagree with the decision.
                    justification = _removal_justification(action.get("path"),
                                                           peers, peers_known)
                    if justification == "stale_geometry":
                        _state["stale_geometry_removed"] = int(
                            _state["stale_geometry_removed"]) + 1
                        _state["stale_geometry_removed_bytes"] = int(
                            _state["stale_geometry_removed_bytes"]
                        ) + removed_size
                        _state["stale_geometry_removed_last_pass"] = int(
                            _state["stale_geometry_removed_last_pass"]) + 1
                    elif justification == "stale_key_shape":
                        _state["stale_key_shape_removed"] = int(
                            _state["stale_key_shape_removed"]) + 1
                        _state["stale_key_shape_removed_bytes"] = int(
                            _state["stale_key_shape_removed_bytes"]) + removed_size
                    elif justification == "peer":
                        # A LIVE rank's file, unlinked. Since 0.2.45 a capacity pass
                        # cannot plan one, so this is the counter to alert on.
                        _state["peer_files_removed_live"] = int(
                            _state["peer_files_removed_live"]) + 1
                        _state["peer_files_removed_live_bytes"] = int(
                            _state["peer_files_removed_live_bytes"]) + removed_size
                # EVERY file this plan actually deleted is journalled, so the
                # scheduler's index stops offering it within one refresh. Without this
                # half the journal the index would keep offering files whose bytes the
                # ceiling already reclaimed - the one direction of index error that
                # costs a reported load failure instead of a plain recompute.
                _journal_write(action.get("path"), removed_size, removal=True)
            elif outcome in ("refused", "gone", "error"):
                failed += 1
    freed = sum(int(row["counts"].get("planned_bytes") or 0)
                for _d, row in reports
                if isinstance(row, dict) and row.get("ok"))
    _state["evict_files_unlinked"] = int(_state["evict_files_unlinked"]) + unlinked
    _state["evict_files_failed"] = int(_state["evict_files_failed"]) + failed
    if cut_off:
        # THE DEADLINE, at BATCH GRANULARITY, and never before the first batch has run.
        # One batch is the unit that is all-or-nothing (one `execute_unlink_plan` call
        # re-validates and deletes its own list), so stopping BETWEEN batches can only
        # ever drop whole directories from the destructive set - a half-applied plan is
        # the one thing this pass must never be. The first batch always runs so a pass
        # that was cut off still RELEASES BYTES, and a submitted batch is always awaited
        # and collected, so a truncation can never strand a deleted file without its
        # journal record.
        _state["plan_truncated"] = True
        _state["ceiling_deadline_hit"] = True
        _state["last_reason"] = (
            f"the ceiling pass stopped at its {budget['seconds']}s budget after "
            f"{len(reports)} of {len(batches)} directory batch(es) of a "
            f"{int(plan['planned'])}-file plan; the next spill continues"
        )[:MAX_STRING_CHARS]
    # `evictions` counts the files this pass ATTEMPTED, which is exactly
    # `plan["planned"]` whenever the whole plan ran - the pre-0.2.29 number,
    # unchanged. With a deadline it counts what was really handed to a worker rather
    # than what was planned and never touched: the alternative would inflate the
    # eviction count by every file the budget declined to attempt. `evicted_bytes` was
    # always the bytes actually freed.
    _state["evictions"] = int(_state["evictions"]) + int(attempted)
    _state["evicted_bytes"] = int(_state["evicted_bytes"]) + freed
    # The estimate comes DOWN by what was really freed, off the DECISION number: on
    # a full walk that is the measurement, exactly as it always was. The per-owner
    # estimates come down by the same rule, each against its OWN bucket: the fair share
    # is only honest if the number it was decided on tracks the bytes that left.
    _state["estimated_bytes"] = max(0, decision_bytes - freed)
    _state["estimated_own_bytes"] = max(
        0, int(_state["estimated_own_bytes"]) - freed_by_owner["own"])
    _state["estimated_foreign_dead_bytes"] = max(
        0, int(_state["estimated_foreign_dead_bytes"])
        - freed_by_owner["dead"])
    _state["estimated_peer_live_bytes"] = max(
        0, int(_state["estimated_peer_live_bytes"]) - freed_by_owner["peer"])
    _state["estimated_unattributed_bytes"] = max(
        0, int(_state["estimated_unattributed_bytes"])
        - freed_by_owner["unattributed"])
    if int(_state["peer_files_removed_last_pass"]) and not _state[
            "ceiling_plan_short"]:
        # THE CROSS-RANK REMOVAL, IN WORDS. A legitimate pass (it reached bucket 2
        # with its peers KNOWN and its own files gone) still deserves one sentence in
        # `last_reason`: this is the one eviction outcome that can surprise an
        # operator reading a single rank file, and the counter alone does not say
        # "known peer" from "we were somehow blind". A short plan already overwrites
        # `last_reason` with its own, more urgent sentence, which names the withheld
        # count - so do not clobber it here.
        _state["last_reason"] = (
            f"this pass removed {int(_state['peer_files_removed_last_pass'])} "
            f"file(s) whose geometry digest is NOT this rank's "
            f"(peer_files_removed={int(_state['peer_files_removed'])} since boot, "
            f"{int(_state['peer_files_removed_bytes'])} byte(s)); the plan had "
            f"{int(_state['peer_files_planned'])} of them as known-peer bucket "
            "entries")[:MAX_STRING_CHARS]


def _capacity_visible(parsed: dict) -> bool:
    """Does a name that parsed this way occupy capacity this tier owns?

    ONE definition, used by the counting pass and the measuring pass alike, because
    those two MUST describe the same set of files: the scale factor is
    `total_files / sampled_files`, and scaling a sample drawn from one population by
    the size of a different one is a number, not a measurement. The rule is the one
    the measuring pass has always applied - a committed file with our name shape
    counts (including the pre-group `stale_key_shape`, which is ours, holds real
    bytes and is unlinkable), while a `.part` (an in-flight write, never a committed
    file) and a name that is not ours at all do not.
    """
    return bool(not parsed["partial"] and (
        parsed["ok"]
        or parsed["classification"] == disk_layout.NAME_STALE_KEY_SHAPE))


def _name_is_ours(name) -> bool:
    """The cheap half of `disk_layout.parse_path`: our prefix AND our suffix.

    The counting pass runs this BEFORE the full parse, because a store that has
    collected other programs' files - or a shard of `.part` cruft - must not pay a
    ~2.6 microsecond parse per entry to discover that none of them are ours. A name
    that passes here can still be refused by `parse_path`: this is a filter, not a
    verdict, and the verdict stays `parse_path`'s. It cannot be the verdict, because
    a filter that disagrees with the parse in EITHER direction would silently move
    the population the sample is scaled against.
    """
    return (isinstance(name, str) and name.startswith(disk_layout.SLOT_FILE_PREFIX)
            and name.endswith(disk_layout.SUFFIX_DATA))


def _count_store(root, *, cap_dirs: int, max_entries: int,
                 deadline=None) -> dict:
    """How many capacity-visible files the store holds, WITHOUT one `stat()`.

    WHY A SECOND PASS EXISTS. The measuring pass stats at most `MAX_WALK_FILES`, so
    on a big store it measures a SAMPLE - and a sample is only usable against a byte
    ceiling if you know what it was a sample OF. Counting entries answers that at
    readdir cost: `os.scandir` hands back the `d_type` byte from the same `getdents`
    buffer the measuring pass reads anyway, so `is_symlink()`/`is_file()` cost no
    syscall on ext4, XFS and tmpfs, and the only per-entry work is `_name_is_ours`
    plus, for a name that looks like ours, one `parse_path`.

    THE BOUND, AND ITS WORST CASE. `max_entries` directory ENTRIES across the whole
    store and `cap_dirs` shard directories: at the cap, ~100,000 entry iterations,
    `cap_dirs + 1` `opendir` calls, ZERO `stat` calls and at most `max_entries`
    `parse_path` calls (~0.26 s of CPU at the cap; measured 0.099 s for all 29,532
    entries of the live-shape store). It runs ONLY from `_walk_store`, which
    `_enforce_ceiling` calls only once the estimate has crossed the ceiling - so the
    worst case lands on the enforcement path that is already stat-ing 20,000 files
    and unlinking up to 4,096, not on a spill. Expressing the bound in ENTRIES rather
    than files is the point: it bounds the WORK, where the `stat` cap bounds the
    syscalls.

    A shard whose iteration raised, or that was cut short by the entry cap, is
    reported as UNKNOWN (`None`) and never as zero - "I could not read it" must not
    reach the ceiling as "it is empty". Unknown shards are extrapolated by the mean
    of the shards fully counted and the result is FLAGGED (`count_truncated`), never
    presented as a count: the shard is the first two hex chars of a block hash, so
    the population is hash-distributed and a mean over the shards seen IS the
    store's own estimate of the shards not seen.
    """
    try:
        top = list(os.scandir(root))
    except OSError as exc:
        return _fail(f"the store could not be listed: {type(exc).__name__}")
    shard_dirs = [entry for entry in top if entry.is_dir(follow_symlinks=False)]
    shards_all = len(shard_dirs)
    walked = shard_dirs[:cap_dirs]
    paths = [entry.path for entry in walked]
    counts: list = []
    examined = 0
    skipped = 0
    tally = 0
    full_shards = 0
    counting_stopped = False
    deadline_hit = False
    for shard in walked:
        if counting_stopped:
            counts.append(None)
            continue
        if _deadline_expired(deadline):
            # The pass's TIME budget, not its entry budget. Same treatment on
            # purpose: shards we did not read are UNKNOWN, never zero, and are
            # extrapolated from the mean of the ones we did - so a deadline truncates
            # the COUNT exactly as an entry cap does and the caller's honesty witness
            # (`count_truncated`) covers both.
            deadline_hit = True
            counting_stopped = True
            counts.append(None)
            continue
        count = 0
        unknown = False
        try:
            with os.scandir(shard.path) as children:
                for child in children:
                    if examined >= max_entries:
                        unknown = True
                        break
                    if _deadline_expired(deadline):
                        deadline_hit = True
                        unknown = True
                        break
                    if child.is_symlink() or not child.is_file(follow_symlinks=False):
                        continue
                    examined += 1
                    # The cheap filter first, then the parse: see `_name_is_ours`.
                    if not _name_is_ours(child.name) or not _capacity_visible(
                            disk_layout.parse_path(child.name)):
                        # Not invisible: a store can hold real bytes this tier will
                        # neither measure nor shed (a `.part` left by a killed spill,
                        # a malformed name, someone else's file), and `skipped_files`
                        # is the counter that says how many. The old walk promised
                        # this number in a comment and never kept it.
                        skipped += 1
                        continue
                    count += 1
        except OSError:
            unknown = True
        if unknown:
            counting_stopped = True
            counts.append(None)
            continue
        full_shards += 1
        tally += count
        counts.append(count)
    truncated = counting_stopped or shards_all > cap_dirs
    if truncated:
        # Ceiling division: an extrapolation that rounds DOWN would round the
        # ceiling's own trigger DOWN with it.
        mean = -(-tally // full_shards) if full_shards else 0
        total = tally + (mean * (shards_all - full_shards))
        total = max(total, tally)
    else:
        mean = 0
        total = tally
    return _ok(units=disk_layout._units("total_files", "entries_examined",
                                        "skipped_files", "shard_dirs_total",
                                        "shard_dirs_walked",
                                        "shard_dirs_counted",
                                        "mean_files_per_shard"),
               shard_paths=paths, counts=counts, total_files=int(total),
               entries_examined=int(examined), skipped_files=int(skipped),
               shard_dirs_total=int(shards_all),
               shard_dirs_walked=len(walked),
               shard_dirs_counted=int(full_shards),
               mean_files_per_shard=int(mean),
               count_truncated=bool(truncated), deadline_hit=bool(deadline_hit),
               reason=None)


def _sample_quotas(count: dict, cap_files: int) -> list:
    """A fair share of the `stat` budget for every shard directory the count saw.

    THE SHAPE OF THE SAMPLE IS THE OTHER HALF OF THE FIX. "Stat until the cap, then
    stop" walks depth-first: it measures whatever the FIRST shards hold, is blind to
    the rest for both jobs a walk does (the byte total comes from one corner of the
    key space and a file never seen cannot be evicted), and truncates on a store
    whose bytes live further down the directory list. So the budget is spread as a
    uniform RATE instead: shard `i` is asked for `cap_files * files_i / total_files`
    of its files, which samples EVERY shard at the same fraction of itself and spends
    the whole budget however the files are distributed. Proportional rather than
    even-per-shard, because an even split wastes the budget on the small shards and a
    proportional one cannot; a shard with files in it always gets at least 1, because
    skipping a shard entirely is the same blindness in a smaller hat. A shard whose
    count is unknown is treated as holding the mean of the shards that are known.

    `None` means "no per-shard bound": the whole store is inside the budget and this
    is a measurement, not a sample. That is the branch every store under the caps
    takes, and the reason its numbers cannot have moved.
    """
    counts = count["counts"]
    total = int(count["total_files"])
    if not count["count_truncated"] and total <= cap_files \
            and int(count["shard_dirs_total"]) <= int(count["shard_dirs_walked"]):
        return [None] * len(counts)
    mean = int(count["mean_files_per_shard"])
    if mean <= 0:
        mean = 1
    quotas: list = []
    for known in counts:
        files = mean if known is None else int(known)
        if files <= 0:
            quotas.append(0)
            continue
        share = (cap_files * files) // total if total > 0 else files
        quotas.append(max(1, int(share)))
    return quotas


def _walk_absence(reason) -> dict:
    """The shape of a walk that could not read the store at all.

    Every key present, every measurement an ABSENCE rather than a plausible zero: a
    caller must not be able to read "0 bytes" out of a failed walk and conclude the
    store is empty. (`ok` is False and no caller in this repository reads a number
    from a failed walk, which is exactly what this function makes unnecessary.)
    """
    return _fail(reason, entries=[], used_bytes=None, files=None,
                 total_files=None, estimated_total_bytes=None,
                 entries_examined=None, skipped_files=None,
                 shard_dirs_total=None, shard_dirs_walked=None,
                 truncated=True, count_truncated=True,
                 measurement_basis="unavailable")


def _walk_store(root, *, max_files=None, max_shard_dirs=None,
                max_entries=None, deadline=None) -> dict:
    """Measure the store, bounded, across the WHOLE shard space. No links followed.

    Three caps rather than one - shard DIRECTORIES (288; the space is 256, so the
    cap only fires on garbage), FILES statted, and directory ENTRIES read - because
    the walk does two jobs and they cost different currencies. Two bounded passes,
    EACH bounded by `max_entries` entries, so the whole walk reads at most twice
    that many directory entries and stats at most `max_files` of them:

      1. `_count_store` counts the capacity-visible files with readdir only, so the
         walk knows how big the population it is about to sample is; and
      2. the measuring pass stats a PROPORTIONAL SAMPLE of it - the same fraction of
         every shard, never a depth-first run into the first ones - and reports
         `used_bytes` (the bytes it actually stat-ed, the same number it always
         reported) AND `estimated_total_bytes` (that sample scaled by
         `total_files / files`), which is the number a byte ceiling belongs to.

    A walk that saw every file reports the two as the SAME number, so for a store
    inside the caps nothing here moves: same bytes, same entries, same order, same
    eviction plan. When it truncated, `truncated`, `count_truncated`, `files` and
    `total_files` all say so and the basis names how the estimate was made - a
    reader must never be able to mistake "I sampled 20,000 files and extrapolated"
    for "the store is this big".

    `max_files` / `max_shard_dirs` / `max_entries` TIGHTEN the caps for a caller
    that must be cheap (the polled snapshot section stats a handful of files; the
    `/disk` verb's `ls` uses the full caps). They may only lower them: a caller
    cannot widen a walk this module decided was the safe size.
    """
    cap_files = MAX_WALK_FILES if max_files is None else min(MAX_WALK_FILES,
                                                             int(max_files))
    cap_dirs = MAX_SHARD_DIRS if max_shard_dirs is None \
        else min(MAX_SHARD_DIRS, int(max_shard_dirs))
    cap_entries = MAX_WALK_ENTRIES if max_entries is None \
        else min(MAX_WALK_ENTRIES, int(max_entries))
    count = _count_store(root, cap_dirs=cap_dirs, max_entries=cap_entries,
                         deadline=deadline)
    if not count["ok"]:
        return _walk_absence(count["reason"])
    entries: list[dict] = []
    used = 0
    counted = 0
    measured_entries = 0
    entries_capped = False
    deadline_hit = bool(count.get("deadline_hit"))
    budget = cap_files
    for path, quota in zip(count["shard_paths"], _sample_quotas(count, cap_files),
                           strict=True):
        share = budget if quota is None else min(int(quota), budget)
        if share <= 0:
            continue
        got = 0
        try:
            with os.scandir(path) as children:
                for child in children:
                    if got >= share:
                        break
                    if measured_entries >= cap_entries:
                        entries_capped = True
                        break
                    if _deadline_expired(deadline):
                        # The pass ran out of TIME. Half a shard measured is a partial
                        # measurement, which is the same fact an OSError below records
                        # - `counted < total_files` is what marks it - so the walk
                        # reports itself truncated and the caller's partial-plan rule
                        # takes over. Nothing here is a zero for an absent measurement.
                        deadline_hit = True
                        break
                    if child.is_symlink() or not child.is_file(follow_symlinks=False):
                        continue
                    measured_entries += 1
                    if not _name_is_ours(child.name):
                        continue
                    # Parsed BEFORE the stat, which the old walk did the other way
                    # round: a shard of names that are not ours must not buy a
                    # syscall each, and the accepted SET is identical either way.
                    parsed = disk_layout.parse_path(child.name)
                    if not _capacity_visible(parsed):
                        continue
                    try:
                        info = child.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    counted += 1
                    got += 1
                    budget -= 1
                    used += int(info.st_size)
                    # THE SIBLING WITNESS (0.2.29): a committed file under a digest that
                    # is not this boot's, with the time it was written. This is the only
                    # place the tier learns that a peer rank is alive on this root, and
                    # it costs no extra syscall - the parse and the `stat` are both
                    # already paid above.
                    _observe_peer_digest(parsed.get("geometry_digest"),
                                         int(info.st_mtime_ns))
                    entries.append({"path": os.path.join(path, child.name),
                                    "filename": child.name,
                                    "size_bytes": int(info.st_size),
                                    "mtime_ns": int(info.st_mtime_ns),
                                    # The same units idiom `disk_layout`'s own sweep
                                    # entries carry, so a walk entry dropped into any
                                    # payload arrives with its dimensions named.
                                    "units": disk_layout._units("size_bytes",
                                                                "mtime_ns")})
        except OSError:
            # Half a shard is a partial measurement, and `counted < total_files`
            # below is what records it as one.
            continue
    total_files = int(count["total_files"])
    truncated = bool(count["count_truncated"]) or entries_capped \
        or deadline_hit or counted < total_files
    if not truncated:
        # The walk saw every capacity-visible file. The estimate IS the measurement,
        # byte for byte, which is the property every store inside the caps rests on.
        scaled = used
        basis = "full_measurement"
    elif counted == 0:
        # A bounded walk that measured NOTHING. Its `used_bytes` is a floor of zero,
        # and zero is the most dangerous number this module can emit: read as
        # "the store is empty" it silences the ceiling forever. Report the absence
        # and let the caller decline to act on it (see `_enforce_ceiling`).
        scaled = used
        basis = "sample_unmeasured"
    else:
        # The scaling. Uniform slot widths make this close to exact; the honest
        # word for it is still "estimate", and `measurement_basis` says so. The
        # population is at least the sample, so an unknown count can never scale a
        # measurement DOWN.
        population = max(total_files, counted)
        scaled = max(used, used * population // counted)
        basis = ("sample_scaled_count_extrapolated" if count["count_truncated"]
                 else "sample_scaled")
    return _ok(units=disk_layout._units("used_bytes", "files", "total_files",
                                        "estimated_total_bytes",
                                        "entries_examined", "skipped_files",
                                        "shard_dirs_total", "shard_dirs_walked"),
               entries=entries, used_bytes=used, files=counted,
               total_files=total_files, estimated_total_bytes=int(scaled),
               entries_examined=int(count["entries_examined"]) + measured_entries,
               skipped_files=int(count["skipped_files"]),
               shard_dirs_total=int(count["shard_dirs_total"]),
               shard_dirs_walked=int(count["shard_dirs_walked"]),
               truncated=bool(truncated),
               count_truncated=bool(count["count_truncated"]),
               deadline_hit=bool(deadline_hit),
               measurement_basis=basis, reason=None)


def request_spill(caches=None, slot=None, key=None, *, worker=None,
                  fingerprint=None, resolver=None, shadow=None) -> dict:
    """THE entry point the store-completion hook calls. Cheap, bounded, never blocks.

    `key` is the 16-hex-char, group-qualified block hash for this slot, or None;
    when it is None the module's injected `resolver(slot)` is asked. A slot the
    resolver cannot name is REFUSED AND COUNTED (`no_key_for_slot`) - that is the
    shape of the wiring gap, made visible instead of guessed at (see the
    docstring's option 3). `caches` may be the worker's mapping directly or a
    worker object, whose `cpu_kv_caches` is read by NAME (the `worker_diag`
    posture).

    The expensive half (reading 11.4 MB and writing it) is handed to the thread, so
    the engine thread pays a dict lookup, an arm check and one `put_nowait`. A full
    queue is a DROP with a counter - the queue never grows, because an unbounded
    queue on a box that is already slow is a memory leak with a nicer name.

    `shadow` is the optional hand-off to `disk_shadow`, the read-only witness that
    compares a completed store's GPU bytes against these same bytes. It is a dict, it
    is filled with `blob` by reference below, and NOTHING else in this function reads
    or writes it: no import of that module, no call into it, no behavior of this
    function depends on it. A caller that passes nothing (every caller today except the
    store-completion hook) changes nothing about this call.
    """
    _state["requests"] += 1
    try:
        state = configured()
        if not state["configured"]:
            _state["refused"]["not_configured"] += 1
            return _refused("not_configured",
                            f"{config.ENV_DISK_ROOT} is not set or names a refused "
                            "path: no disk work is possible")
        if not is_armed():
            _state["refused"]["disarmed"] += 1
            return _refused("disarmed",
                            "the spill path is disarmed: the arm is a LATCH and "
                            "stays disarmed until someone arms it")
        adopted = fingerprint if fingerprint is not None else _state["fingerprint"]
        if adopted is None:
            _state["refused"]["fingerprint_missing"] += 1
            return _refused("fingerprint_missing",
                            "no fingerprint is configured for this boot: nothing "
                            "here names a file without an identity to name it under")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            _state["refused"]["bad_call"] += 1
            return _refused("bad_call", f"slot must be a non-negative int, got "
                                        f"{slot!r}")
        resolved = key if key is not None else None
        if resolved is None and callable(resolver):
            try:
                resolved = resolver(slot)
            except Exception as exc:  # noqa: BLE001 - a resolver's fault is a refusal
                _state["refused"]["no_key_for_slot"] += 1
                snapshot._error(exc, "disk_spill resolver")
                return _refused("no_key_for_slot",
                                f"the resolver raised {type(exc).__name__}")
        if resolved is None:
            _state["refused"]["no_key_for_slot"] += 1
            return _refused("no_key_for_slot",
                            "nothing named this slot's block hash: the engine does "
                            "not carry a key to the worker (see the module "
                            "docstring's option 3)")
        region_map = caches
        if region_map is None and worker is not None:
            region_map = getattr(worker, "cpu_kv_caches", None)
        if not isinstance(region_map, dict) or not region_map:
            _state["refused"]["no_region_tensors"] += 1
            return _refused("no_region_tensors",
                            "no cpu_kv_caches mapping was handed in")
        if not _window_open(time.monotonic()):
            _state["refused"]["over_rate_window"] += 1
            return _refused("over_rate_window",
                            f"the write-rate window allows {MAX_FILES_PER_WINDOW} "
                            f"files per {SPILL_WINDOW_SECONDS}s")
        # COPY-AT-COMPLETION. The bytes are read HERE, synchronously, by the thread the
        # store completion arrived on - that is the whole point. A CPU slot the RAM tier
        # has finished storing into is available to the NEXT store, so a job that
        # deferred this read would write whatever block was stored into that slot next
        # under the name of the block that completed. The cost is one memcpy of a rank
        # slot on the engine thread (11,468,800 B on the live PP0, 12,902,400 on PP1);
        # the bound on that cost is MAX_QUEUED_BYTES, checked BEFORE the copy, and every
        # failure here is a drop with its own name rather than an exception on the
        # engine's thread.
        named = plan_slot_naming(resolved, adopted)
        if not named["ok"]:
            kind = str(named.get("reason_kind") or "bad_key")
            if kind not in REFUSAL_REASONS:
                kind = "bad_key"
            _state["refused"][kind] = int(_state["refused"].get(kind, 0)) + 1
            return _refused(kind, str(named["reason"]))
        width = int(named["bytes_per_file"])
        if int(_state["queue_bytes"]) + width > MAX_QUEUED_BYTES:
            _state["drops"] = int(_state["drops"]) + 1
            _state["refused"]["queue_bytes_full"] += 1
            return _refused("queue_bytes_full",
                            f"holding one more slot ({width} B) would exceed the "
                            f"{MAX_QUEUED_BYTES} B the queue may hold; the job was "
                            "dropped, not queued")
        read = read_slot_bytes(region_map, int(slot), width)
        if not read["ok"]:
            _state["refused"]["snapshot_failed"] += 1
            _state["last_error"] = str(read["reason"])[:MAX_STRING_CHARS]
            return _refused("snapshot_failed", str(read["reason"]))
        job = {"blob": read["bytes"], "slot": int(slot), "key": resolved,
               "fingerprint": adopted, "bytes": width,
               "queued_ts": round(time.time(), 6)}
        # THE SHADOW WITNESS GETS THE SAME BYTES, BY REFERENCE. `shadow` is a plan
        # `disk_shadow.snapshot()` decided and `disk_store` handed down: filling this
        # one key is the ENTIRE interface between the two modules - no import, no
        # call, no second copy of 11 MB. It happens after the read and before the
        # queue put, so the comparison runs against what this call decided to write,
        # at the earliest moment those bytes exist; a job that is refused above leaves
        # the plan's `blob` None, which `disk_shadow.settle` counts as "nothing to
        # compare" rather than as a mismatch. Ignored entirely when no shadow is
        # watching.
        if isinstance(shadow, dict):
            shadow["blob"] = read["bytes"]
        pending = _get_queue()
        if not _ensure_thread():
            _state["refused"]["write_failed"] += 1
            return _refused("write_failed", "the writer thread could not start")
        try:
            pending.put_nowait(job)
        except queue.Full:
            _state["drops"] = int(_state["drops"]) + 1
            _state["refused"]["queue_full"] += 1
            return _refused("queue_full",
                            f"the queue is at its {MAX_QUEUE_DEPTH}-job cap: the "
                            "job was dropped, not queued")
        _state["accepted"] += 1
        depth = int(_state["queue_depth"]) + 1
        _state["queue_depth"] = depth
        _state["queue_bytes"] = int(_state["queue_bytes"]) + width
        _state["queue_bytes_peak"] = max(int(_state["queue_bytes_peak"]),
                                         int(_state["queue_bytes"]))
        _state["queue_depth_peak"] = max(int(_state["queue_depth_peak"]), depth)
        return {"ok": True, "queued": True, "slot": int(slot),
                "path": None, "reason": None, "bytes": width,
                "note": "the bytes were copied at store completion; the file is "
                        "written on the spill thread"}
    except Exception as exc:  # noqa: BLE001 - nothing here may reach the engine
        _state["errors"] += 1
        snapshot._error(exc, "disk_spill request")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return _refused("bad_call", f"the spill request raised "
                                    f"{type(exc).__name__}")


def _refused(kind: str, reason: str) -> dict:
    _state["last_reason"] = str(reason)[:MAX_STRING_CHARS]
    return {"ok": False, "queued": False, "reason_kind": kind, "reason": reason,
            "dropped": kind == "queue_full"}


def _window_open(now: float) -> bool:
    """The write-rate window: files per rolling window, restart-empty when it lapses."""
    start = _state.get("window_start_mono")
    if not isinstance(start, (int, float)) \
            or (now - start) >= SPILL_WINDOW_SECONDS:
        _state["window_start_mono"] = now
        _state["window_files"] = 0
    if int(_state.get("window_files") or 0) >= MAX_FILES_PER_WINDOW:
        return False
    _state["window_files"] = int(_state.get("window_files") or 0) + 1
    return True


def drain(timeout_seconds: float = 5.0) -> dict:
    """Wait for the queue to empty. A test and shutdown helper, not an engine path.

    Bounded: it reports `drained: False` rather than waiting forever, because a
    wedged writer must stay visible as a wedged writer.
    """
    local = _queue
    if local is None:
        return {"drained": True, "reason": "no queue was ever created",
                "queued": 0}
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while local.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    return {"drained": local.unfinished_tasks == 0,
            "queued": int(local.unfinished_tasks),
            "reason": None if local.unfinished_tasks == 0
                      else "the queue was still working at the timeout"}


def stop_writer(timeout_seconds: float = 2.0) -> dict:
    """Ask the thread to exit. Nothing here is relied on at shutdown today."""
    global _writer_thread
    local = _queue
    if local is None:
        return {"stopped": True, "reason": "no thread was ever started"}
    try:
        local.put_nowait(_STOP)
    except queue.Full:
        return {"stopped": False, "reason": "the queue is full; try again"}
    thread = _writer_thread
    if thread is not None and thread.is_alive():
        thread.join(max(0.0, float(timeout_seconds)))
    alive = bool(thread is not None and thread.is_alive())
    _state["thread_alive"] = alive
    return {"stopped": not alive, "reason": None if not alive
            else "the thread did not exit inside the timeout"}


def _queue_backlog() -> int:
    """Jobs queued OR being written RIGHT NOW. Never raises; 0 when no queue exists.

    `unfinished_tasks` rather than `qsize()`: a job the writer has taken but not
    finished is not on the queue and IS still unwritten, so a flush that reported
    `drained: true` on `qsize() == 0` would be reporting a file that is mid-`write()`.
    """
    local = _queue
    if local is None:
        return 0
    try:
        return int(local.unfinished_tasks)
    except Exception:  # noqa: BLE001 - a queue that cannot answer is treated as empty
        return 0


def shutdown_flush_verdict() -> dict:
    """The SHUTDOWN DRAIN's budget, resolved from the environment. Never raises.

    Same three-door honesty as `disk_restore.staging_rows_verdict()`: the shipped
    default, the boot environment, and a value the grammar refused reported with its
    complaint while the default stands. `0` is the operator's off switch and is legal,
    because a drain is an addition to the stop path that he must be able to take back.
    """
    out = {"seconds": float(SHUTDOWN_FLUSH_SECONDS_DEFAULT),
           "default_seconds": float(SHUTDOWN_FLUSH_SECONDS_DEFAULT),
           "maximum": float(FLUSH_BUDGET_SECONDS_MAX), "enabled": True,
           "source": "default", "raw": None, "recognised": True, "clamped": False,
           "problems": [], "env_var": config.ENV_DISK_SHUTDOWN_FLUSH_SECONDS,
           "note": SHUTDOWN_FLUSH_NOTE}
    try:
        env = config.disk_shutdown_flush_env()
    except Exception as exc:  # noqa: BLE001 - a budget read is never worth a fault
        snapshot._error(exc, "disk_spill shutdown flush verdict")
        return out
    out["raw"] = env.get("raw")
    out["source"] = str(env.get("source") or "default")
    out["recognised"] = bool(env.get("recognised", True))
    out["clamped"] = bool(env.get("clamped"))
    out["problems"] = [str(problem) for problem in (env.get("problems") or [])]
    seconds = env.get("seconds")
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        out["seconds"] = max(0.0, min(float(FLUSH_BUDGET_SECONDS_MAX), float(seconds)))
    out["enabled"] = out["seconds"] > 0.0
    return out


# "No budget was asked for", as a value. A sentinel rather than `None` because `None`
# is a value a caller can SEND (`{"op":"flush","timeout_seconds":null}`), and those two
# facts must not mean the same thing: an absent argument takes the environment's budget,
# while an argument that is not a number is a malformed request and is refused.
FLUSH_BUDGET_UNSET = object()


def _flush_budget(requested) -> tuple:
    """(seconds, problem) for ONE flush. A non-numeric ask is a PROBLEM, never a wait.

    The budget is a WAIT that somebody asked for, so the two ways to get it wrong are
    asymmetric: a missing ask takes the environment's value (that is what the env var is
    for), while a malformed ask must be REFUSED - silently falling back to "wait 20
    seconds" would put a wait on the caller's path that nobody requested.
    """
    verdict = shutdown_flush_verdict()
    if requested is FLUSH_BUDGET_UNSET:
        return verdict["seconds"], None, verdict
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        return 0.0, (f"timeout_seconds must be a number of seconds, got "
                     f"{requested!r}"), verdict
    try:
        asked = float(requested)
    except (TypeError, ValueError):  # noqa: BET001 - defensive: a weird numeric type
        return 0.0, (f"timeout_seconds must be a number of seconds, got "
                     f"{requested!r}"), verdict
    if asked != asked or asked < 0.0:      # NaN, or a negative wait
        return 0.0, (f"timeout_seconds must be 0 or a positive number, got "
                     f"{asked}"), verdict
    if asked > FLUSH_BUDGET_SECONDS_MAX:
        # The caller asked for more than the ceiling allows: it gets the ceiling AND
        # sees
        # that it did, on the same `clamped` field the environment's clamp reports. A
        # bound that is applied in silence is a bound a caller can never discover.
        verdict = dict(verdict)
        verdict["clamped"] = True
        verdict["problems"] = list(verdict["problems"]) + [
            f"timeout_seconds={asked} is above the "
            f"{FLUSH_BUDGET_SECONDS_MAX} s ceiling; the ceiling stands"]
        return float(FLUSH_BUDGET_SECONDS_MAX), None, verdict
    return asked, None, verdict


def flush(budget_seconds=FLUSH_BUDGET_UNSET) -> dict:
    """Wait, BOUNDED, for the spill queue to reach empty. Never raises, never blocks
    longer than the budget, and returns in milliseconds when there is nothing to do.

    WHY THIS EXISTS. Mirror mode made the spill asynchronous by design, so a stop that
    happens the instant the queue still holds work loses exactly the data the change
    exists to protect. This is the bounded wait that closes that gap - a MIRROR, not a
    flush procedure: nobody has to know how the bytes got there, they only have to ask
    whether the queue is empty.

    WHAT IT IS NOT: a dump of the arena. It waits for jobs already in the queue; it
    never queues anything itself, never arms or disarms, never touches a posture, and
    never writes a byte on the caller's thread. The mechanism is one `unfinished_tasks`
    read per `FLUSH_POLL_SECONDS`, which is why the empty case costs a comparison and a
    dict build rather than a sleep.

    THE NAMED REASONS, because a caller that cannot tell the causes cannot decide: a
    budget that expired is `timeout` (the backlog is real, the disk is slow), a queue
    with no live writer is `writer_down` (those jobs will NEVER land, so waiting is a
    hang - this returns promptly), a budget of 0 is `disabled` (the operator's own off
    switch), and a no-seam answer is `not_installed`. `ok` is about the CALL, so a
    timeout is `ok: true, drained: false`: a proxy that read `ok` as "the bytes are
    safe" would suspend on a queue that still holds the conversation.
    """
    started = time.monotonic()
    budget, problem, verdict = _flush_budget(budget_seconds)
    answer = {"ok": True, "drained": False, "queue_depth_before": 0,
              "queue_depth_after": 0, "bytes_written": 0, "elapsed_seconds": 0.0,
              "timed_out": False, "reason": None, "budget_seconds": None,
              "clamped": bool(verdict["clamped"]), "budget_source": verdict["source"],
              "budget_problems": list(verdict["problems"]),
              "poll_seconds": FLUSH_POLL_SECONDS, "writer_alive":
              bool(_state["thread_alive"]), "mechanism": SHUTDOWN_FLUSH_NOTE}
    # THE DEPTH IS MEASURED FIRST, before any refusal branch: a caller whose budget was
    # rejected still needs to see what was sitting on the queue when it asked, because
    # that is the number it is deciding about. `_queue_backlog()` owns its own failure
    # (a queue that cannot answer reads as empty), so no guard is needed here.
    answer["queue_depth_before"] = _queue_backlog()
    if problem is not None:
        answer["ok"] = False
        answer["reason"] = problem
        answer["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return answer
    answer["budget_seconds"] = float(budget)
    _state["flush_calls"] = int(_state["flush_calls"]) + 1
    try:
        before = answer["queue_depth_before"]
        bytes_before = int(_state["bytes_written"])
        if before <= 0:
            # THE COMMON CASE, and the one the suspend path runs on every sleep: look
            # first, so "nothing to do" costs a comparison instead of a sleep.
            answer["drained"] = True
            answer["queue_depth_after"] = 0
            answer["bytes_written"] = 0
            _state["flush_drained"] = int(_state["flush_drained"]) + 1
            answer["elapsed_seconds"] = round(time.monotonic() - started, 6)
            _state["flush_last"] = dict(answer)
            return answer
        if budget <= 0.0:
            answer["reason"] = "disabled"
            answer["queue_depth_after"] = before
            answer["elapsed_seconds"] = round(time.monotonic() - started, 6)
            _state["flush_last"] = dict(answer)
            return answer
        thread = _writer_thread
        if not (thread is not None and thread.is_alive()):
            # Nobody is going to drain this. Say so now rather than after the budget:
            # the caller's decision (proceed with the stop) is the same either way, and
            # a suspend that waits 15 s to learn it is a suspend 15 s late.
            answer["reason"] = "writer_down"
            answer["queue_depth_after"] = before
            answer["elapsed_seconds"] = round(time.monotonic() - started, 6)
            _state["flush_last"] = dict(answer)
            return answer
        deadline = started + float(budget)
        while _queue_backlog() > 0 and time.monotonic() < deadline:
            time.sleep(FLUSH_POLL_SECONDS)
        after = _queue_backlog()
        answer["queue_depth_after"] = after
        answer["bytes_written"] = max(0, int(_state["bytes_written"]) - bytes_before)
        answer["elapsed_seconds"] = round(time.monotonic() - started, 6)
        answer["drained"] = after == 0
        if answer["drained"]:
            _state["flush_drained"] = int(_state["flush_drained"]) + 1
        else:
            answer["timed_out"] = True
            answer["reason"] = "timeout"
            _state["flush_timed_out"] = int(_state["flush_timed_out"]) + 1
        _state["flush_bytes_written"] = int(_state["flush_bytes_written"]) + int(
            answer["bytes_written"])
        _state["flush_last"] = dict(answer)
        return answer
    except Exception as exc:  # noqa: BLE001 - a flush never breaks a stop or a step
        snapshot._error(exc, "disk_spill flush")
        _state["flush_errors"] = int(_state["flush_errors"]) + 1
        answer["ok"] = False
        answer["reason"] = f"the flush raised {type(exc).__name__}"
        answer["queue_depth_after"] = _queue_backlog()
        answer["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return answer


# The ONE atexit registration this module ever makes, so a repeated install cannot
# stack handlers. `atexit` is the SECONDARY seam (see `SHUTDOWN_FLUSH_NOTE`): it runs in
# a process that exits normally and NOT in a worker, which ends with `os._exit`.
_atexit_registered = [False]
_atexit_lock = threading.Lock()


def drain_at_shutdown(budget_seconds=None) -> dict:
    """The stop-path entry: ONE bounded flush, one log line, and never a raise.

    Fail-open is the whole contract here. A budget that expires costs one WARNING and
    the stop proceeds; a drain that could hold a suspend hostage would be a diagnostic
    causing the outage it measures. The log is ONE line per process, at WARNING like the
    boot line, because the question "did the backlog make it before the stop?" is asked
    after the fact from the engine log.

    `budget_seconds` is an OPTIONAL OVERRIDE for a caller that knows it has less time
    than the environment allows; the ceiling is honoured either way, so an override can
    shorten a drain and can never lengthen one past `FLUSH_BUDGET_SECONDS_MAX`.
    """
    verdict = shutdown_flush_verdict()
    if not verdict["enabled"]:
        return {"ok": True, "drained": True, "reason": "disabled",
                "budget_seconds": 0.0, "skipped": True}
    asked = budget_seconds
    if isinstance(asked, bool) or not isinstance(asked, (int, float)):
        asked = None
    budget = (float(verdict["seconds"]) if asked is None
              else max(0.0, min(float(FLUSH_BUDGET_SECONDS_MAX), float(asked))))
    answer = flush(budget)
    answer["budget_source"] = ("override" if asked is not None
                               else verdict["source"])
    if answer["drained"] and not int(answer["queue_depth_before"]):
        # NOTHING TO DRAIN, SO NOTHING TO SAY. This is the common case at a stop (the
        # queue drained long ago), and the line exists to answer "did the backlog make
        # it?" - a question that has no content when there was no backlog. The
        # empty-case
        # answer still travels in the return value for a caller that asked directly.
        answer["skipped"] = True
        return answer
    answer.setdefault("skipped", False)
    if answer["drained"]:
        _logger.warning("kvtier-dev-plugin: SHUTDOWN FLUSH drained the spill queue "
                       "(%d job(s), %d B written) in %.3f s of a %.1f s budget",
                       int(answer["queue_depth_before"]),
                       int(answer["bytes_written"]), float(answer["elapsed_seconds"]),
                       float(answer["budget_seconds"] or 0.0))
    else:
        _logger.warning("kvtier-dev-plugin: SHUTDOWN FLUSH did NOT drain the spill "
                       "queue (reason=%s, %d job(s) left, %d B written) after "
                       "%.3f s; the stop proceeds anyway and those bytes are lost "
                       "with the process (budget %.1f s, %s)",
                       answer["reason"], int(answer["queue_depth_after"]),
                       int(answer["bytes_written"]), float(answer["elapsed_seconds"]),
                       float(answer["budget_seconds"] or 0.0),
                       verdict["env_var"])
    return answer


def install_shutdown_flush(connector) -> dict:
    """Chain a bounded drain onto the connector's own `shutdown`, ONCE, in this process.

    WHY THE CONNECTOR, AND NOT `atexit` (the finding that decided this, read out of the
    live fork): the spill queue is built in the WORKER process - `request_spill` runs
    inside the worker's `get_finished` - and a worker ends its orderly shutdown in
    `multiproc_executor.py:_exit_worker_process` with `os._exit`, which skips `atexit`,
    every pending `finally` and interpreter finalization. A drain registered only with
    `atexit` would therefore be silent in the one process that owns the queue. The
    worker's teardown DOES reach the connector, however: `Worker.shutdown()` ->
    `ensure_kv_transfer_shutdown()` -> `_KV_CONNECTOR_AGENT.shutdown()`, and that runs
    BEFORE the `os._exit`. So the connector instance the plugin already sees at creation
    is the seam that exists in the right process at the right moment.

    AN INSTANCE WRAP, NOT A CLASS PATCH. It is the object the fork handed us, wrapped
    once and marked, so it cannot clobber a sibling layer, cannot double-wrap on a
    retry, and cannot survive to the next connector. The chain is honest: the drain
    runs FIRST (what follows it in the fork is `destroy_model_parallel()` and then the
    exit), the original is called afterwards with its arguments, and the ORIGINAL's own
    exception still propagates - that failure is the fork's story and not ours to
    swallow. Our own failure is owned here: counted in `flush_errors`, logged once, and
    the fork's teardown runs regardless.
    """
    report = {"installed": False, "already_installed": False, "reason": None,
              "reason_kind": None, "budget_seconds": None, "atexit": False,
              "seams": [], "target": type(connector).__name__}
    try:
        verdict = shutdown_flush_verdict()
        report["budget_seconds"] = float(verdict["seconds"])
        if not verdict["enabled"]:
            # The rollback is structural, not a budget of zero: a wrapper that runs and
            # does nothing is still a wrapper that can fail on the stop path.
            report["reason_kind"] = "disabled"
            report["reason"] = (f"{verdict['env_var']}={verdict['raw']!r} asked for no "
                                "shutdown drain, so nothing is installed")
            return report
        if connector is None:
            report["reason_kind"] = "no_connector"
            report["reason"] = "no connector object was handed over, so there is no " \
                               "lifecycle seam to chain onto"
            return report
        existing = getattr(connector, "shutdown", None)
        if existing is not None and getattr(existing, SHUTDOWN_FLUSH_ATTR, None) \
                == SHUTDOWN_FLUSH_ATTR:
            report["already_installed"] = True
            report["reason"] = "this connector already carries the drain"
            return report
        if not callable(existing):
            report["reason_kind"] = "no_seam"
            report["reason"] = (f"{report['target']} has no callable `shutdown` to "
                                "chain onto (a fork rename, or a connector shape this "
                                "release has not seen)")
            return report
        original = existing

        def shutdown(*args, **kwargs):
            try:
                drain_at_shutdown()
            except Exception as exc:  # noqa: BLE001 - the fork's teardown must still run
                snapshot._error(exc, "disk_spill shutdown flush")
                _state["flush_errors"] = int(_state["flush_errors"]) + 1
            return original(*args, **kwargs)

        setattr(shutdown, SHUTDOWN_FLUSH_ATTR, SHUTDOWN_FLUSH_ATTR)
        connector.shutdown = shutdown
        # THE MARKER ON THE OBJECT TOO, so "is this connector carrying the drain?" is
        # one
        # `getattr` for any reader, not a walk into the method's attributes - and the
        # idempotence check above reads the method's marker, which is the one that
        # matters
        # for not wrapping twice.
        setattr(connector, SHUTDOWN_FLUSH_ATTR, SHUTDOWN_FLUSH_ATTR)
        report["installed"] = True
        report["seams"].append("connector.shutdown")
        with _atexit_lock:
            if not _atexit_registered[0]:
                atexit.register(drain_at_shutdown)
                _atexit_registered[0] = True
        report["atexit"] = bool(_atexit_registered[0])
        if report["atexit"]:
            report["seams"].append("atexit")
        report["reason"] = ("the drain is chained onto the connector's own shutdown "
                            "and registered with atexit; atexit does not run in a "
                            "worker (os._exit), which is why the connector seam is the "
                            "primary one")
        _state["shutdown_flush"] = {"installed": True,
                                    "budget_seconds": float(verdict["seconds"]),
                                    "budget_source": verdict["source"],
                                    "seams": list(report["seams"]),
                                    "target": report["target"]}
        return report
    except Exception as exc:  # noqa: BLE001 - an install failure never costs a connector
        snapshot._error(exc, "disk_spill install shutdown flush")
        report["installed"] = False
        report["reason_kind"] = "error"
        report["reason"] = f"the install raised {type(exc).__name__}"
        return report


def shutdown_flush_state() -> dict:
    """What this process installed, if anything. Never raises, creates nothing."""
    state = _state.get("shutdown_flush")
    verdict = shutdown_flush_verdict()
    if isinstance(state, dict):
        out = dict(state)
        out["enabled"] = bool(verdict["enabled"])
        out["budget_configured_seconds"] = float(verdict["seconds"])
        out["problems"] = list(verdict["problems"])
        out["note"] = SHUTDOWN_FLUSH_NOTE
        return out
    return {"installed": False, "enabled": bool(verdict["enabled"]),
            "budget_seconds": float(verdict["seconds"]),
            "budget_configured_seconds": float(verdict["seconds"]),
            "budget_source": verdict["source"], "seams": [], "target": None,
            "problems": list(verdict["problems"]), "note": SHUTDOWN_FLUSH_NOTE}


def verify_slot(key, fingerprint=None) -> dict:
    """Read back what we wrote, and check its length. The ONLY read path here.

    Stage 2 is allowed exactly this much reading: enough to prove a spill landed
    with the bytes we handed it. There is deliberately no "find a slot to load"
    helper, because there is nothing that could act on a hit. `key` is the FULL
    group-qualified block hash - the same identity `plan_slot_write` named the file
    with, so a verification cannot succeed against another group's bytes.
    """
    try:
        adopted = fingerprint if fingerprint is not None else _state["fingerprint"]
        if adopted is None:
            return _fail("no fingerprint is configured")
        configured()
        digested = disk_layout.fingerprint_digest(adopted)
        if not digested["ok"]:
            return _fail(str(digested["reason"]))
        named = disk_layout.build_slot_path(_state["root"] or "", key,
                                            digested["geometry_digest"])
        if not named["ok"]:
            return _fail(str(named["reason"]))
        try:
            size = os.path.getsize(named["path"])
        except FileNotFoundError:
            return _fail("the file is not there", exists=False,
                         path=named["path"], bytes_read=0)
        cap = int(digested["total_bytes_per_block"])
        if size > cap:
            return _fail(f"the file is {size} bytes, past the fingerprint's "
                         f"{cap}-byte slot width", exists=True,
                         path=named["path"], bytes_read=0)
        with open(named["path"], "rb") as handle:
            blob = handle.read(cap)
        return _ok(units=disk_layout._units("read_bytes", "bytes_on_disk"),
                   exists=True, path=named["path"], read_bytes=len(blob),
                   sha256=hashlib.sha256(blob).hexdigest(),
                   truncated=size > len(blob), bytes_on_disk=int(size))
    except Exception as exc:  # noqa: BLE001 - a verification is never a fault
        snapshot._error(exc, "disk_spill verify")
        return _fail(f"verification failed: {type(exc).__name__}")


# --- the status view ---------------------------------------------------------

def journal_view(*, fs: dict | None = None,
                 index: dict | None = None) -> dict:
    """The spill journal's own block: where it lives, and what landed in it.

    ONE SHAPE, published in two places (the snapshot's `spill.journal` and the
    `/disk` verb's `journal`), so a `POST /disk {"op":"status"}` answer and a
    snapshot can never disagree about the number that means "the scheduler's index
    was never told about these files".

    `writable: false` with its `directory_error` is the failure this block exists
    to make impossible to miss: the journal directory was named by
    `VLLM_E2_TIER_DISK_JOURNAL_DIR` and cannot be created or written, so every
    spill this rank makes commits a KV file and records nothing. The spill is not
    refused for that (a logging problem does not get to cost the whole tier) - it
    is said, here, in the payload an operator queries.

    State only, plus `disk_index`'s per-directory cached filesystem verdict and the
    one `access` its writability costs; no walk, no statvfs. `fs`/`index` let a
    caller that already read them reuse the read.
    """
    fs = disk_index.journal_fs_state() if fs is None else fs
    index = disk_index.counters() if index is None else index
    return {"directory": disk_index.journal_directory(),
            "directory_env": config.ENV_DISK_JOURNAL_DIR,
            "fallback_env": config.ENV_DIAG_DIR,
            # THE JOURNAL'S OWN VOICE: the count of lines that never landed, the
            # last reason, and whether the directory can be written at all.
            "write_errors": int(index["journal_write_errors"]),
            "last_error": index["journal_last_error"],
            "writable": fs.get("writable"),
            "directory_error": fs.get("directory_error"),
            "fs": fs}

def section() -> dict:
    """The snapshot view. STABLE IN SHAPE, and it reads NO files.

    `snapshot.dev_state_sections()` runs on every polled status, so this reports
    only this process's own state and the configuration - never a directory walk,
    never a statvfs, never the store's size. The counters are the point: spilled,
    refused PER REASON, bytes, seconds, drops, evictions. (The journal's filesystem
    and writability are the two reads it does make, both of them per-directory
    cached or one syscall, because "where does the map live" is a first-class
    counter's-equal: it is the difference between an empty index and a lost store.)
    """
    state = configured()
    # ONE read of the journal's own view, reused by every journal field below.
    # `disk_index`'s counters are THIS process's, and that is the right process:
    # the writer thread that could not write a line runs here. `journal_fs_state()`
    # adds one `access` on the journal directory (deliberately uncached, so an
    # operator who chmods the path is reflected on the next poll) on top of the
    # mount-table read it already cached per directory.
    journal_fs = disk_index.journal_fs_state()
    journal_index = disk_index.counters()
    return {
        "capability": capability(),
        # WHICH PROCESS these numbers belong to, placed beside the counters themselves
        # because the mistake it prevents was made twice on one night.
        "scope": PER_PROCESS_SCOPE_NOTE,
        "configured": bool(state["configured"]),
        "root": state["root"],
        "ceiling_bytes": state["ceiling_bytes"],
        # THE COST MODEL, in bytes, next to the ceiling it feeds. See SPAN_COST_MEANING:
        # the two per-span costs are published SEPARATELY because a single "bytes per
        # span" number would be a lie about one of the two contracts.
        "span_cost_bytes": ({
            "positional_bytes_per_span": _slot_width_bytes(),
            "boundary_state_bytes_per_span": _slot_width_bytes(),
            "meaning": SPAN_COST_MEANING,
            "units": disk_layout._units("positional_bytes_per_span",
                                        "boundary_state_bytes_per_span"),
        } if _slot_width_bytes() is not None else {
            "positional_bytes_per_span": None,
            "boundary_state_bytes_per_span": None,
            "meaning": dict(SPAN_COST_MEANING,
                            why_absent="no fingerprint is configured, so this rank "
                                       "does not know the width of a file it would "
                                       "write"),
            "units": disk_layout._units("positional_bytes_per_span",
                                        "boundary_state_bytes_per_span"),
        }),
        "armed": bool(is_armed()),
        "arm_state": arm_state(),
        "fingerprint": disk_layout.fingerprint_section(_state["fingerprint"]),
        "geometry_digest": _state["geometry_digest"],
        "requests": int(_state["requests"]),
        "accepted": int(_state["accepted"]),
        "spilled": int(_state["spilled"]),
        "duplicates": int(_state["duplicates"]),
        "bytes_written": int(_state["bytes_written"]),
        "write_seconds": round(float(_state["write_seconds"]), 6),
        "read_seconds": round(float(_state["read_seconds"]), 6),
        "drops": int(_state["drops"]),
        "refused": dict(_state["refused"],
                        **{"units": disk_layout._units(*REFUSAL_REASONS)}),
        "evictions": int(_state["evictions"]),
        "evicted_bytes": int(_state["evicted_bytes"]),
        "estimated_bytes": int(_state["estimated_bytes"]),
        "walk_truncated": bool(_state["walk_truncated"]),
        # HOW the last ceiling decision was measured. `estimated_bytes` above is the
        # number the ceiling was compared against, and on a store bigger than the
        # walk's caps it is an EXTRAPOLATION from a sample - which is the right
        # decision to have made, and a different claim from "I measured the store".
        # Without this block a reader cannot tell "the store is 170 GiB" from "I
        # stat-ed 20,000 of 29,532 files and scaled what I saw", and a witness that
        # cannot make that distinction is not a witness: it is how a 334.9 GiB store
        # sat under a 192 GiB ceiling while every payload said the estimate was fine.
        "ceiling_measurement": {
            # "full_measurement" | "sample_scaled"
            # | "sample_scaled_count_extrapolated" | None (no walk yet).
            "basis": _state["walk_basis"],
            "measured_bytes": int(_state["walk_measured_bytes"]),
            "sampled_files": int(_state["walk_sampled_files"]),
            "total_files": int(_state["walk_total_files"]),
            "entries_examined": int(_state["walk_entries_examined"]),
            # Files in the store this tier can neither measure nor shed: a `.part`
            # left by a killed spill, a malformed name, someone else's file. They
            # are capacity, they are invisible to `estimated_bytes`, and this is the
            # counter that stops them being invisible to a reader.
            "skipped_files": int(_state["walk_skipped_files"]),
            "walk_truncated": bool(_state["walk_truncated"]),
            "count_truncated": bool(_state["walk_count_truncated"]),
            # The last plan hit `disk_layout.MAX_EVICTION_PLAN`: the pass was
            # partial by design and the next spill runs another one. Since 0.2.29 it is
            # ALSO true when the pass ran out of TIME - `deadline_hit` says which.
            "plan_truncated": bool(_state["plan_truncated"]),
            # THE PASS'S OWN CLOCK (0.2.29). `budget_seconds` is the bound in force,
            # `pass_seconds` how long the last pass actually took, `deadline_hit`
            # whether it was cut off. With `plan_truncated` these three answer "is the
            # ceiling keeping up with the fill rate, or is every pass being truncated at
            # its budget?" - which during a load experiment is the difference between a
            # tier that is shedding and one that is pretending.
            "budget_seconds": _state["ceiling_budget_seconds"],
            "pass_seconds": round(float(_state["ceiling_pass_seconds"]), 6),
            "deadline_hit": bool(_state["ceiling_deadline_hit"]),
            "deadline_hits": int(_state["ceiling_deadline_hits"]),
            "walk_deadline_hit": bool(_state["walk_deadline_hit"]),
            "ceiling_checks": int(_state["ceiling_checks"]),
            # The FIRST walk of a boot, which is what lets a fresh process see a
            # store it did not write. Zero here with a full store and a quiet
            # `estimated_bytes` is the story of a ceiling that never woke up.
            "seed_walks": int(_state["estimate_seed_walks"]),
            # THE TWO 0.2.62 RULES, as numbers. `grace_entries_last_pass` is what the
            # PLANNER withheld from this rank's shed candidates because the touch clock
            # put them inside `disk_layout.UNLINK_GRACE_SECONDS`; `grace_unlink_refused`
            # is what the EXECUTOR refused on a path a plan had already named (the two
            # differ when the touch landed between the walk and the unlink, which is the
            # race the rule exists for). `claim_paths_last_pass` is how many paths this
            # process had reads in flight for when the pass ran, and
            # `claim_unlink_refused` how many of those a plan had named anyway.
            # A pass with `plan_short: true` and a non-zero grace count is a pass that
            # came up short because the store's young files were the only ones left -
            # which is the intended failure mode, not a fault.
            "grace_seconds": disk_layout.UNLINK_GRACE_SECONDS,
            "grace_entries_last_pass": int(_state["grace_entries_last_pass"]),
            "grace_bytes_last_pass": int(_state["grace_bytes_last_pass"]),
            "grace_entries_excluded": int(_state["grace_entries_excluded"]),
            "grace_unlink_refused": int(_state["grace_unlink_refused"]),
            "grace_unlink_refused_last_pass": int(
                _state["grace_unlink_refused_last_pass"]),
            "claim_unlink_refused": int(_state["claim_unlink_refused"]),
            "claim_unlink_refused_last_pass": int(
                _state["claim_unlink_refused_last_pass"]),
            "claim_paths_last_pass": int(_state["claim_paths_last_pass"]),
            "claim_set_truncated": bool(_state["claim_set_truncated"]),
            "units": disk_layout._units("measured_bytes", "sampled_files",
                                        "total_files", "entries_examined",
                                        "skipped_files", "ceiling_checks",
                                        "seed_walks", "deadline_hits",
                                        "budget_seconds", "pass_seconds",
                                        "grace_seconds", "grace_entries_last_pass",
                                        "grace_bytes_last_pass",
                                        "grace_entries_excluded",
                                        "grace_unlink_refused",
                                        "grace_unlink_refused_last_pass",
                                        "claim_unlink_refused",
                                        "claim_unlink_refused_last_pass",
                                        "claim_paths_last_pass"),
        },
        # THE FAIR SHARE (0.2.30): THE NUMBER THE CEILING WAS ACTUALLY COMPARED
        # AGAINST, and who else is sharing it. This is the block that makes tonight's
        # over-eviction readable while it is happening: four ranks sharing a root each
        # used to defend the WHOLE ceiling over one directory, and the only way to see
        # that is broken is to see the share each rank believes it owns.
        "ceiling_share": {
            # The bytes this rank is accountable for, and the budget they are measured
            # against. Since 0.2.58 `share_bytes` is the width-weighted share - the
            # whole ceiling divided by the SUM of the ranks' block widths, times this
            # rank's
            # own width - so every rank holds the same NUMBER of blocks (see
            # `share_blocks`/`share_basis` below); with no width known it is the older
            # `ceiling / n_effective`. `target_bytes` is the share times
            # `EVICT_TARGET_FRACTION`, which is what a pass sheds toward.
            "accountable_bytes": _accountable_bytes(),
            "own_bytes": int(_state["estimated_own_bytes"]),
            "dead_bytes": int(_state["estimated_foreign_dead_bytes"]),
            "peer_live_bytes": int(_state["estimated_peer_live_bytes"]),
            "unattributed_bytes": int(_state["estimated_unattributed_bytes"]),
            "share_bytes": _state["share_bytes"],
            # EQUAL IN BLOCKS, NOT BYTES (0.2.58). `share_blocks` is the number this
            # rank's budget can actually hold; `share_basis` says whether the widths
            # made that number EQUAL ACROSS RANKS by construction (`block_equal`) or
            # merely
            # derived from a byte budget (`equal_bytes`). This is the pair an operator
            # reads to answer "does a block exist on all four ranks at the same depth?"
            # without knowing any rank's geometry.
            "share_blocks": _state["share_blocks"],
            "own_width_bytes": _state["share_own_width"],
            "peer_widths_known": _state["share_width_peers"],
            "share_basis": _state["share_basis"],
            "target_bytes": _state["share_target_bytes"],
            "share_divisor": _state["share_divisor"],
            # 1 + live peers, which is the ONLY honest rank count this module can get
            # without an engine change: the walk that measures the store is the walk
            # that learns who is writing to it. `no_live_peer_seen` means the rank
            # behaves exactly as every release before 0.2.30 did (the whole ceiling).
            "n_effective": (1 + int(_state["peer_digests_live"]) if
                            _state["bytes_attributed"] else None),
            "share_source": _state["share_source"],
            "attributed": bool(_state["bytes_attributed"]),
            # THE "ONE RANK DID THE WORK, THREE DECLINED" WITNESS. A counter, not a log
            # line, because the pass runs on the spill writer thread and the signature
            # has to be watchable live across four rank files.
            "passes_declined": int(_state["ceiling_passes_declined"]),
            "declined_accountable_bytes": _state["ceiling_declined_accountable_bytes"],
            "declined_target_bytes": _state["ceiling_declined_target_bytes"],
            "passes_nothing_sheddable": int(
                _state["ceiling_passes_nothing_sheddable"]),
            "peer_entries_excluded": int(_state["peer_entries_excluded"]),
            "store_estimate_bytes": int(_state["estimated_bytes"]),
            "note": ("the store is SHARED by every rank whose fingerprint digest "
                     "appears in it, so a rank that sizes the WHOLE store against the "
                     "WHOLE ceiling makes the effective capacity ceiling/rank_count: "
                     "measured on the live box on 2026-10-22, four ranks each shed "
                     "~23 GiB of the same 23 GiB (7,956 files, 156 GiB -> 68 GiB under "
                     "a 148 GiB ceiling whose one-pass target was 133 GiB). "
                     "`accountable_bytes` = this rank's own files + the dead geometry "
                     "no live rank owns, and `peer_live_bytes` is deliberately "
                     "EXCLUDED from it: a live sibling accounts for those under its "
                     "own share, and they are no longer in this rank's destructive set "
                     "at all (`peer_entries_excluded` counts what was withheld). With "
                     "no peer learned, `n_effective` is 1 and the share is the whole "
                     "ceiling - every earlier release's behaviour, so the fail-safe "
                     "direction is 'sheds like before', never 'too timid to protect "
                     "the filesystem'."),
            "units": disk_layout._units("accountable_bytes", "own_bytes",
                                        "dead_bytes", "peer_live_bytes",
                                        "unattributed_bytes", "share_bytes",
                                        "share_blocks", "own_width_bytes",
                                        "peer_widths_known", "share_basis",
                                        "target_bytes", "share_divisor",
                                        "n_effective", "passes_declined",
                                        "declined_accountable_bytes",
                                        "declined_target_bytes",
                                        "passes_nothing_sheddable",
                                        "peer_entries_excluded", "store_estimate_bytes",
                                        "ceiling_bytes"),
        },
        # THE PEERS THIS RANK CAN SEE, and the pool that does the unlinks (0.2.29).
        # Both belong beside the ceiling witness because both only mean anything next
        # to a decision: a peer digest you never showed the planner protected nothing,
        # and a thread count you never measured is a wish.
        "peer_ranks": {
            "live_digests": int(_state["peer_digests_live"]),
            "learned_digests": int(_state["peer_digest_learned"]),
            "foreign_files_seen": int(_state["peer_digest_sightings"]),
            "decisions_without_peers": int(_state["peer_digests_unconfirmed"]),
            "files_planned_last_pass": int(_state["peer_files_planned"]),
            "peer_digests_dropped": int(_state["peer_digests_dropped"]),
            # THE FAIL-CLOSED GATE'S OWN OUTCOME (0.2.34), beside the peer map it
            # depends on: what was WITHHELD because it could not be attributed, and
            # whether the plan therefore came up SHORT of its target.
            "unattributable_entries_last_pass": int(
                _state["unattributable_entries_last_pass"]),
            "unattributable_bytes_last_pass": int(
                _state["unattributable_bytes_last_pass"]),
            "unattributable_entries_excluded": int(
                _state["unattributable_entries_excluded"]),
            "plan_short": bool(_state["ceiling_plan_short"]),
            "plan_short_bytes": int(_state["ceiling_plan_short_bytes"]),
            "plan_short_reason": _state["ceiling_plan_short_reason"],
            "passes_short": int(_state["ceiling_passes_short"]),
            # FIX 3: THE ATTRIBUTIBLE REMOVAL. Files this rank unlinked whose name
            # carries a digest that is NOT its own - the count that answers, in one
            # read, the question a `-` journal line cannot answer at all. Named for
            # what it means to a reader: with the gate in place this is only ever
            # reachable through the KNOWN-peer bucket, so every one of these is a
            # cross-rank removal.
            "peer_files_removed": int(_state["peer_files_removed"]),
            "peer_files_removed_bytes": int(_state["peer_files_removed_bytes"]),
            "peer_files_removed_last_pass": int(
                _state["peer_files_removed_last_pass"]),
            # THE UNAMBIGUOUS CASE: the digest belonged to a rank this walk believed
            # LIVE. `peer_files_removed` can also name this rank's OWN older geometry,
            # which is not a cross-rank removal at all; this one cannot lie.
            "peer_files_removed_known_peer": int(
                _state["peer_files_removed_known_peer"]),
            # WHICH RULE PAID FOR EACH REMOVAL (0.2.45), so the two
            # readings of `peer_files_removed` can be told apart from one
            # read: `stale_geometry_removed`  = cleanup of bytes a
            # CONFIGURATION CHANGE orphaned (a digest no live rank
            # publishes and none is writing). This is the reclaim path
            # working, and it is the number the operator's 2026-10-07
            # arena-size change is expected to produce.
            # `peer_files_removed_live` = a file belonging to a rank that
            # is ALIVE. A capacity pass cannot plan one since 0.2.45, so
            # this is the alarm. They are disjoint, and
            # `peer_files_removed` is their superclass.
            "stale_geometry_removed": int(_state["stale_geometry_removed"]),
            "stale_geometry_removed_bytes": int(
                _state["stale_geometry_removed_bytes"]),
            "stale_geometry_removed_last_pass": int(
                _state["stale_geometry_removed_last_pass"]),
            "stale_key_shape_removed": int(_state["stale_key_shape_removed"]),
            "stale_key_shape_removed_bytes": int(
                _state["stale_key_shape_removed_bytes"]),
            "peer_files_removed_live": int(_state["peer_files_removed_live"]),
            "peer_files_removed_live_bytes": int(
                _state["peer_files_removed_live_bytes"]),
            # THE IDENTITY VIEW (0.2.45): WHO THIS RANK COULD SEE, and on
            # what evidence. `source` names the witness that decided
            # (`rank_files` when a live worker's published identity was
            # readable, `write_activity` when nothing fresh was readable
            # here and the pre-0.2.45 reading stands). `identity_known` is
            # the flag that lets a foreign digest be called dead at all:
            # it says "every live rank I can see, I can also NAME". When
            # it is false a foreign committed file is UNATTRIBUTABLE, is
            # withheld, and the ceiling is allowed to come up SHORT.
            "source": _state["peer_liveness_source"],
            "identity_reads": int(_state["peer_identity_reads"]),
            "identity_available": bool(_state["peer_identity_available"]),
            "identity_known": bool(_state["peer_identity_known"]),
            "identity_published_digests": int(_state["peer_identity_digests"]),
            "identity_own_published": bool(_state["peer_identity_own_published"]),
            "identity_files_unnamed": int(_state["peer_identity_files_unnamed"]),
            "identity_files_stale": int(_state["peer_identity_files_stale"]),
            "identity_files_unreadable": int(
                _state["peer_identity_files_unreadable"]),
            "identity_reason": _state["peer_identity_reason"],
            "fresh_seconds": PEER_DIGEST_FRESH_SECONDS,
            "min_sightings": PEER_DIGEST_MIN_SIGHTINGS,
            "peer_digest_cap": MAX_PEER_DIGESTS,
            "rank_file_fresh_seconds": PEER_RANK_FILE_FRESH_SECONDS,
            "note": (
                "`live_digests` are foreign digests this rank has positive evidence "
                "for: a sibling is writing under them NOW "
                "(`PEER_DIGEST_FRESH_SECONDS` of committed files) OR a live worker "
                "publishes them as its current digest in its rank file. Their files "
                "shed LAST, and never for capacity. SILENCE IS NOT DEATH (0.2.45): a "
                "rank that stops SPILLING drops out of the write-activity witness, and "
                "that alone no longer makes its bytes free capacity - it takes a rank "
                "file that stopped being rewritten (a worker that is gone) or a digest "
                "nobody writes under at all (a configuration change) to move those "
                "bytes to `stale_geometry`. `source` + `identity_known` say how this "
                "rank decided: with `identity_known: false` it cannot name every live "
                "rank, so a foreign digest outside its map is UNATTRIBUTABLE, is "
                "withheld (`unattributable_entries_last_pass`, "
                "`unattributable_entries_excluded`) and the plan may come up SHORT "
                "(`plan_short`, `plan_short_bytes`, `passes_short`). Exceeding the "
                "ceiling is the intended consequence: spilling back-pressures and an "
                "operator can act, while deleting a sibling's live restore corrupts a "
                "serving path in silence. TO ANSWER 'IS AN IDLE RANK BEING PROTECTED "
                "OR SHED?': read `identity_published_digests` (an idle rank still "
                "publishes -> protected while it is idle, `live_digests` counts it) "
                "against `stale_geometry_removed` / "
                "`stale_geometry_removed_bytes` (an idle rank whose WORKER has gone, "
                "its bytes being reclaimed as orphaned) and "
                "`peer_files_removed_live` (an idle rank's bytes being shed while it "
                "is still alive - since 0.2.45 that is a bug, not a behaviour, and it "
                "is counted separately from the cleanup case so the two cannot be "
                "confused). This rank's own files and the bytes positively claimed by "
                "no live rank stay sheddable whatever the map says, so a single-rank "
                "box never comes up short for want of a peer."),
            "units": disk_layout._units("live_digests", "learned_digests",
                                        "foreign_files_seen",
                                        "decisions_without_peers",
                                        "files_planned_last_pass",
                                        "peer_digests_dropped", "fresh_seconds",
                                        "min_sightings", "peer_digest_cap",
                                        "rank_file_fresh_seconds",
                                        "identity_reads",
                                        "identity_published_digests",
                                        "identity_files_unnamed",
                                        "identity_files_stale",
                                        "identity_files_unreadable",
                                        "stale_geometry_removed",
                                        "stale_geometry_removed_bytes",
                                        "stale_geometry_removed_last_pass",
                                        "stale_key_shape_removed",
                                        "stale_key_shape_removed_bytes",
                                        "peer_files_removed_live",
                                        "peer_files_removed_live_bytes",
                                        "unattributable_entries_last_pass",
                                        "unattributable_bytes_last_pass",
                                        "unattributable_entries_excluded",
                                        "plan_short_bytes", "passes_short",
                                        "peer_files_removed",
                                        "peer_files_removed_bytes",
                                        "peer_files_removed_last_pass",
                                        "peer_files_removed_known_peer"),
        },
        "evict_pool": {
            "evict_threads_requested": _state["evict_threads_requested"],
            "evict_threads_configured": int(_state["evict_threads_configured"]),
            "evict_threads_alive": int(evict_thread_count_alive()),
            "evict_threads_cap": MAX_EVICT_THREADS,
            "batches_submitted": int(_state["evict_batches_submitted"]),
            "batches_serial": int(_state["evict_batches_serial"]),
            "files_unlinked": int(_state["evict_files_unlinked"]),
            "files_failed": int(_state["evict_files_failed"]),
            "pool_fallbacks": int(_state["evict_fallbacks"]),
            "peak_in_flight_batches": int(_state["evict_peak_in_flight"]),
            "note": ("the pool unlinks the plan's per-directory batches; "
                     "`evict_threads_configured` 1 means the serial path of every "
                     "release before 0.2.29. `pool_fallbacks` is the count that says "
                     "parallel eviction was asked for and did not happen; "
                     "`files_failed` counts refusals, vanished files and errors, "
                     "exactly as the serial loop always reported them."),
            "units": disk_layout._units("evict_threads_requested",
                                        "evict_threads_configured",
                                        "evict_threads_alive", "evict_threads_cap",
                                        "batches_submitted", "batches_serial",
                                        "files_unlinked", "files_failed",
                                        "pool_fallbacks",
                                        "peak_in_flight_batches"),
        },
        "errors": int(_state["errors"]),
        # The collision witness and the journal, next to the counters they qualify:
        # `group_collisions` is the rate the tier was armed to measure, and
        # `journal_failures` is why the restore index may lag a store that is fine.
        "group_collisions": int(_state["group_collisions"]),
        "group_collision_samples": [dict(row) for row
                                    in _state["group_collision_samples"]],
        "digest_groups_tracked": int(len(_state["digest_groups"])),
        "digest_groups_cap": MAX_TRACKED_DIGESTS,
        "digest_groups_dropped": int(_state["digest_groups_dropped"]),
        "digest_group_overflow": int(_state["digest_group_overflow"]),
        "sibling_probe_failures": int(_state["sibling_probe_failures"]),
        "journal_records": int(_state["journal_records"]),
        "journal_removals": int(_state["journal_removals"]),
        "journal_failures": int(_state["journal_failures"]),
        # THE SAME FAILURE, IN THE WORDS THE INDEX USES, on the verb an operator
        # actually queries: `journal_write_errors` is the index's own count of lines
        # it was never told about and `journal_last_error` is the one-line reason.
        # A spill whose journal line fails COSTS DISK SPACE, so the tier keeps
        # spilling - but it does not get to be quiet about it.
        "journal_write_errors": int(journal_index["journal_write_errors"]),
        "journal_last_error": journal_index["journal_last_error"],
        # False here means every spill this rank makes is invisible to the
        # scheduler's restore index: the number that explains an empty index on a
        # store that is anything but empty.
        "journal_directory_writable": journal_fs.get("writable"),
        "journal_directory": disk_index.journal_directory(),
        # THE JOURNAL'S FILESYSTEM, named where an operator reading this verb will
        # look: `fs.on_tmpfs: true` means the map dies at reboot while the KV files
        # it describes survive (2026-10-05: 107 GB spilled, `entries: 0`). The fix
        # is `VLLM_E2_TIER_DISK_JOURNAL_DIR`; the arm file stays behind on purpose.
        "journal": journal_view(fs=journal_fs, index=journal_index),
        "thread_started": int(_state["thread_started"]),
        "thread_alive": bool(_state["thread_alive"]),
        "queue_depth": int(_state["queue_depth"]),
        # A queued job holds the slot's BYTES now (copy-at-completion), so depth alone
        # cannot tell a reader what the queue costs. Both currencies are published, and
        # `queue_bytes_full` in `refused` is the drop that the byte cap caused.
        "queue_bytes": int(_state["queue_bytes"]),
        "queue_bytes_peak": int(_state["queue_bytes_peak"]),
        "queue_depth_peak": int(_state["queue_depth_peak"]),
        "queue_max": MAX_QUEUE_DEPTH,
        # THE FLUSH TALLIES (0.2.66): the bounded waits this process answered, and what
        # they saw land. Beside the queue's own numbers on purpose - a flush is a
        # statement ABOUT that queue, and the split between `flush_drained` and
        # `flush_timed_out` is the difference between "the stop was safe" and "the stop
        # lost the backlog". Per-process, like every counter in this block.
        "flush_calls": int(_state["flush_calls"]),
        "flush_drained": int(_state["flush_drained"]),
        "flush_timed_out": int(_state["flush_timed_out"]),
        "flush_bytes_written": int(_state["flush_bytes_written"]),
        "flush_errors": int(_state["flush_errors"]),
        "shutdown_flush": shutdown_flush_state(),
        "window_files": int(_state.get("window_files") or 0),
        "caps": {"max_queue_depth": MAX_QUEUE_DEPTH,
                 # The queue's SECOND admission currency. Published as a cap because a
                 # reader who sees `refused.queue_bytes_full` grow must be able to tell
                 # the limit from the leak, and because since copy-at-completion the
                 # queue's memory cost is this number, not `max_queue_depth` x nothing.
                 "max_queued_bytes": MAX_QUEUED_BYTES,
                 "spill_window_seconds": SPILL_WINDOW_SECONDS,
                 "max_files_per_window": MAX_FILES_PER_WINDOW,
                 "min_free_bytes": MIN_FREE_BYTES,
                 # The three bounds of one capacity walk, published next to the
                 # ceiling they decide: `estimated_bytes` is a sample as soon as
                 # `ceiling_measurement.total_files` passes `max_walk_files`, and an
                 # operator reading that should be able to see the cap it hit.
                 "max_shard_dirs": MAX_SHARD_DIRS,
                 "max_walk_files": MAX_WALK_FILES,
                 "max_walk_entries": MAX_WALK_ENTRIES,
                 "max_bytes_per_file": (
                     _state["fingerprint"]["total_bytes_per_block"]
                     if isinstance(_state["fingerprint"], dict) else None),
                 "evict_target_fraction": EVICT_TARGET_FRACTION,
                 "units": disk_layout._units(
                     "max_queue_depth", "max_queued_bytes", "spill_window_seconds",
                     "max_files_per_window", "min_free_bytes",
                     "max_bytes_per_file", "evict_target_fraction",
                     "max_shard_dirs", "max_walk_files", "max_walk_entries")},
        "naming": {"template": disk_layout.FILENAME_TEMPLATE,
                   "root": state["root"],
                   "shard": f"<key16[0:{disk_layout.SHARD_CHARS}]>",
                   "key": ("the FULL group-qualified block hash: <key16> is its "
                           "first 16 hex chars and g<group> is the 4-byte suffix "
                           "the engine appended, split by "
                           "disk_layout.split_block_key - a key too short to carry "
                           "the suffix is refused, never truncated into an "
                           "ambiguous name")},
        "inert_when_off": ("no thread, no directory created, no byte written: the "
                           "queue is not even constructed until a job is accepted"),
        "wiring": ("the store-completion hook lives in `disk_store.py`: it wraps "
                   "the worker's bind_connector_metadata/get_finished, reads the "
                   "per-slot keys off the metadata, builds THIS rank's fingerprint "
                   "from its own measured slot width and rank id (configure()), and "
                   "calls request_spill. On an UNPATCHED engine (no store_keys on "
                   "SimpleCPUOffloadMetadata, metadata.py:16-38) that hook refuses "
                   "and counts, and this module is never reached."),
        "read_path": ("none by design: a hit is resolved in the scheduler's "
                      "cached_block_hash_to_block, and only scheduler-side "
                      "store-completion publishes into it (manager.py:1017-1044)"),
        "last_path": _clip(_state["last_path"]),
        "last_reason": _clip(_state["last_reason"]),
        "last_error": _clip(_state["last_error"]),
        "units": _per_process_units(disk_layout._units(
            "bytes_written", "evicted_bytes",
            # THE FLUSH TALLIES (0.2.66), declared with the block they belong to. The
            # payload unit test exists because `flush_bytes_written` is a volume and
            # `flush_calls` is a count, and a reader who confuses them reads a 4 GB
            # write
            # total as 4 GB of waits.
            "flush_calls", "flush_drained", "flush_timed_out", "flush_errors",
            "flush_bytes_written", "budget_seconds", "budget_configured_seconds",
            "budget_max_seconds",
            "estimated_bytes", "write_seconds",
            "read_seconds", "ceiling_bytes", "requests",
            "accepted", "spilled", "duplicates",
            "drops", "evictions", "errors", "queue_depth",
            "queue_depth_peak", "queue_bytes",
            "queue_bytes_peak", "queue_max", "window_files",
            "thread_started", "thread_alive",
            "group_collisions", "digest_groups_tracked",
            "digest_groups_cap",
            "digest_groups_dropped",
            "digest_group_overflow",
            "sibling_probe_failures", "journal_records",
            "journal_removals", "journal_failures",
            "journal_write_errors", "write_errors")),
    }


def _clip(text, chars: int = MAX_STRING_CHARS):
    if isinstance(text, str) and len(text) > chars:
        return text[:chars] + "..."
    return text
