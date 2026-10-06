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

AN ARM MUST DIE AT A REBOOT, BY ITS OWN CONTENTS
-------------------------------------------------
The arm crosses the process boundary as one small file in the diagnostics directory
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

import ctypes
import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from . import blockdump, config, disk_index, disk_layout, snapshot

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

# THE LIVE PEER RANKS (0.2.29). Four ranks share ONE spill root and each rank's
# evictor sees the other three's committed files as names carrying a digest that is
# not its own. `disk_layout.classify_name` can call those files `peer_rank` (live
# data, shed LAST) instead of `stale_geometry` (pure capacity, shed FIRST) - but only
# if it is TOLD the sibling digests, and nothing in the engine hands them over.
# They are therefore LEARNED from the walk this pass already does: a foreign digest
# with committed files written inside `PEER_DIGEST_FRESH_SECONDS` is a rank that is
# WRITING RIGHT NOW, which is the only definition of "live" this module can observe.
#
# FAIL-SAFE DIRECTION, and it is deliberate: if no digest qualifies, the pass behaves
# exactly as it did before 0.2.29 (foreign first). The worst case of being too
# conservative is therefore the old behaviour - peers' live files get shed - and it is
# NEVER "cannot release capacity", because this boot's own files and the truly
# unreadable ones stay sheddable whatever the peer map says. The opposite direction
# (assume every foreign digest is a live sibling) is the dangerous one: a store of
# dead geometry from old boots would become unsheddable and the ceiling would stop
# being a ceiling. A digest is only believed with `PEER_DIGEST_MIN_SIGHTINGS` fresh
# files, so one stray recent file from an old boot cannot protect a whole digest.
PEER_DIGEST_FRESH_SECONDS = 120.0    # SECONDS a foreign digest counts as "writing now"
PEER_DIGEST_MIN_SIGHTINGS = 2        # FRESH committed files before a digest is believed
MAX_PEER_DIGESTS = 32                # DIGESTS remembered (a 4-rank fleet needs 3)

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
SHARE_SOURCE_UNATTRIBUTED = "not_measured_yet"  # no walk has run in this boot

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
# worker's gate costs nothing.
ARM_DEFAULT_SECONDS = 1800.0
ARM_MIN_SECONDS = 5.0
ARM_MAX_SECONDS = 7200.0
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
    "arm_expires_mono": None,
    "arm_expires_wall": None,
    "arm_seconds": None,
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
        "armed": False, "arm_expires_mono": None, "arm_expires_wall": None,
        "arm_seconds": None, "arm_file_verdict": None,
        "arm_file_checked_mono": None, "arm_file_boot_refusals": 0,
        "fingerprint": None, "geometry_digest": None,
        "thread_started": 0, "thread_alive": False,
        "queue_depth": 0, "queue_depth_peak": 0,
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
        # The fair share (0.2.30). Left behind by a reset, one test's peer count would
        # hand the next test a share of 1/N it never earned, and a share that is too
        # small over-evicts - the exact defect this release exists to remove.
        "estimated_own_bytes": 0, "estimated_foreign_dead_bytes": 0,
        "estimated_peer_live_bytes": 0, "estimated_unattributed_bytes": 0,
        "bytes_attributed": False, "share_bytes": None,
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
    `armed_and_ready`    - armed and inside its lifetime: accepted jobs spill.
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
    """This process's arm: in-process, expired lazily, then the arm FILE.

    The expiry is checked on EVERY read (the `intentional_stop_active()` lesson -
    a lapsed arm must never answer `True` because nobody swept). The file is the
    `blockdump` one-way-per-process problem again: the verb that arms runs in the
    EngineCore, and the spill happens in a worker, so the arm crosses as one small
    bounded file. Its verdict is cached for `ARM_FILE_POLL_SECONDS` because this is
    consulted per spill request, not per diagnostic poll.
    """
    deadline = _state["arm_expires_mono"]
    if _state["armed"] and isinstance(deadline, (int, float)) \
            and time.monotonic() >= deadline:
        _state["armed"] = False
        _state["last_reason"] = "the arm expired"
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

    `armed` and `expires_ts` are the two checks the first versions had. THE THIRD IS
    THE BOOT: `_arm_boot_fence` refuses an arm written by a boot that is not this one
    (a reboot must not arm a cold box), and its reason NAMES the mismatch, because
    "disarmed and I do not know why" is how an armed-looking box ends up quietly
    doing nothing.
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
    expires = document.get("expires_ts")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        return False, "the arm file has no numeric expires_ts"
    if time.time() >= float(expires):
        return False, "the arm expired"
    stale = _arm_boot_fence(document)
    if stale:
        _state["arm_file_boot_refusals"] += 1
        return False, stale
    return True, "armed by file"


def arm(seconds=None) -> dict:
    """Arm the spill path here, and (when a diagnostics dir exists) by file.

    Refused when nothing is configured: an armed tier with no root is a knob that
    lies, so the arm requires the root, exactly as `blockdump.arm` requires a
    channel. There is NO env var that arms this - the same rule as guard mode and
    the byte dump: a launcher script must not be able to turn a facility that
    writes real KV bytes to disk on.
    """
    wanted = ARM_DEFAULT_SECONDS
    if seconds is not None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return {"armed": False, "reason": "seconds must be a number"}
        wanted = max(ARM_MIN_SECONDS, min(ARM_MAX_SECONDS, float(seconds)))
    state = configured()
    if not state["configured"]:
        _state["refused"]["not_configured"] += 1
        return {"armed": False, "capability": "not_configured",
                "reason": f"{config.ENV_DISK_ROOT} is not set (or names a refused "
                          "path), so there is nowhere to write and arming would "
                          "only be a lie"}
    now = time.time()
    _state["armed"] = True
    _state["arm_expires_mono"] = time.monotonic() + wanted
    _state["arm_expires_wall"] = now + wanted
    _state["arm_seconds"] = wanted
    _state["arm_file_checked_mono"] = None
    written = _write_arm_file(now, wanted)
    return {"armed": True, "seconds": wanted, "capability": "armed_and_ready",
            "arm_file": written.get("path"),
            "arm_file_written": bool(written.get("written")),
            "arm_file_reason": written.get("reason"),
            "reason": None}


def disarm() -> dict:
    was = bool(_state["armed"])
    _state["armed"] = False
    _state["arm_expires_mono"] = None
    _state["arm_expires_wall"] = None
    _state["arm_seconds"] = None
    _state["arm_file_verdict"] = None
    _state["arm_file_checked_mono"] = None
    removed = _remove_arm_file()
    return {"armed": False, "was_armed": was, "capability": capability(),
            "arm_file_removed": removed,
            "reason": "spills are refused again; a queued job still lands, so "
                      "drain before trusting a zero"}


def arm_state() -> dict:
    armed = is_armed()
    remaining = None
    deadline = _state["arm_expires_wall"]
    if armed and isinstance(deadline, (int, float)):
        remaining = round(max(0.0, deadline - time.time()), 3)
    return {"armed": armed, "capability": capability(),
            "seconds_left": remaining,
            "expires_at_wallclock": (round(float(deadline), 6)
                                     if isinstance(deadline, (int, float))
                                     and armed else None),
            "default_seconds": ARM_DEFAULT_SECONDS,
            "max_seconds": ARM_MAX_SECONDS,
            # THE BOOT FENCE, in the view an operator reads when the tier is quiet:
            # how many arm files were refused for naming another boot, and the last
            # reason the file verdict said no - which names the epoch or boot-id
            # mismatch instead of only saying "disarmed".
            "arm_file_boot_refusals": int(_state["arm_file_boot_refusals"]),
            "arm_file_reason": _state["last_reason"],
            "boot_id": _boot_id(),
            "units": disk_layout._units("seconds_left", "default_seconds",
                                        "max_seconds", "expires_at_wallclock",
                                        "arm_file_boot_refusals"),
            "note": ("an arm expires on its own; no env var arms this facility, "
                     "and a fresh boot does zero disk work. An arm file must also "
                     "name THIS boot (the kernel boot id, and the per-boot epoch "
                     "when the operator sets one), so a reboot disarms a surviving "
                     "file instead of arming a cold box with the restore path "
                     "open to live traffic")}


def _write_arm_file(now: float, seconds: float) -> dict:
    path = _arm_path()
    if path is None:
        return {"written": False, "reason": "no diagnostics directory"}
    # The boot identity the reader will check itself against (`_arm_boot_fence`).
    # `epoch` is the OPERATOR's per-boot value when they set one and this process's
    # token otherwise, and `epoch_scope` says which - without the scope the reader
    # cannot tell "a different boot" apart from "a different worker of this boot",
    # and a false refusal would disarm a healthy tier.
    epoch, scope = config.diagnostics_epoch()
    document = {"version": PROTOCOL_VERSION, "writer": WRITER_TAG,
                "armed": True, "armed_ts": round(now, 6),
                "expires_ts": round(now + seconds, 6),
                "seconds": round(seconds, 3), "root": _state["root"],
                "epoch": str(epoch), "epoch_scope": scope, "pid": os.getpid(),
                "boot_id": _boot_id()}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        raw = json.dumps(document, separators=(",", ":")).encode("utf-8")
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
        return {"written": True, "path": path, "bytes": len(raw)}
    except Exception as exc:  # noqa: BLE001 - an unpublished arm is data
        snapshot._error(exc, "disk_spill arm file")
        return {"written": False, "path": path,
                "reason": f"{type(exc).__name__}: {exc}"[:MAX_STRING_CHARS]}


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
    attribution = _attribute_walk_bytes(walk, _live_peer_digests())
    _apply_attribution(attribution, int(walk["estimated_total_bytes"]),
                       _share_of_ceiling(int(ceiling), attribution["divisor"],
                                         attribution["source"]))


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

    This only OBSERVES. Whether a digest is believed to belong to a LIVE sibling is
    `_live_peer_digests`' decision, made at the moment the plan is built, so a digest
    learned at 10:00 cannot still be protecting a peer's files at 18:00 after that
    rank stopped writing.
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
                # Bounded witness. Dropping a sighting costs the fail-safe direction
                # (the digest is not believed, so its files shed first), never the
                # dangerous one.
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
    """Which foreign digests this rank may call a LIVE sibling's, as a sorted tuple.

    TWO conditions, both required, and both observable rather than assumed: at least
    `PEER_DIGEST_MIN_SIGHTINGS` committed files seen inside the freshness window, and
    the most recently written of them inside it. A digest that stops being written
    ages out of the map here - a rank that died an hour ago is no longer a live
    sibling, and its files go back to being the pure capacity they actually are.

    Empty means "I cannot tell", and the caller must then pass NOTHING to
    `disk_layout`: that is today's behaviour, the fail-safe direction, and it is
    counted (`peer_digests_unconfirmed`) rather than left as a silence.
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


def _attribute_walk_bytes(walk: dict, peers) -> dict:
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
      * anything else (a name this tier cannot classify) -> UNATTRIBUTED: counted as
        capacity, never handed to the planner, which would only refuse it.
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
                "sheddable_entries": entries, "peer_entries": 0}
    own = dead = peer = unattributed = 0
    sheddable: list = []
    peer_entries = 0
    for row in entries:
        size = int(row.get("size_bytes") or 0)
        name = row.get("filename") or os.path.basename(str(row.get("path") or ""))
        kind = disk_layout.classify_name(name, digest, peers)["classification"]
        if kind == disk_layout.NAME_CURRENT:
            own += size
            sheddable.append(row)
        elif kind in (disk_layout.NAME_STALE_GEOMETRY,
                      disk_layout.NAME_STALE_KEY_SHAPE):
            dead += size
            sheddable.append(row)
        elif kind == disk_layout.NAME_PEER_RANK:
            peer += size
            peer_entries += 1
        else:
            unattributed += size
    return {"attributed": True,
            "own_bytes": _scale_to_store(own, used, scaled_total),
            "dead_bytes": _scale_to_store(dead, used, scaled_total),
            "peer_bytes": _scale_to_store(peer, used, scaled_total),
            "unattributed_bytes": _scale_to_store(unattributed, used, scaled_total),
            "accountable_bytes": _scale_to_store(own + dead + unattributed, used,
                                                 scaled_total),
            "peers": tuple(peers or ()), "divisor": 1 + len(peers or ()),
            "source": SHARE_SOURCE_PEERS if peers else SHARE_SOURCE_NO_PEERS,
            "sheddable_entries": sheddable, "peer_entries": int(peer_entries)}


def _share_of_ceiling(ceiling, divisor: int, source: str) -> dict:
    """This rank's slice of a ceiling that `divisor` ranks share. Pure: pinnable.

    `divisor == 1` returns the ceiling and the target UNCHANGED, byte for byte, which is
    what makes the single-rank case (one rank on the box, or a boot that cannot see a
    sibling) identical to every release before this one. The floor division is
    deliberate: a share rounded UP would hand N ranks a combined budget larger than the
    ceiling they were configured with.
    """
    safe = max(1, int(divisor))
    share = int(int(ceiling) // safe)
    if share <= 0:
        # A ceiling smaller than the rank count is a misconfiguration, not a licence to
        # evict everything: 1 byte is a budget, 0 means "delete the store".
        share = 1
    return {"ceiling_bytes": int(ceiling), "divisor": safe,
            "share_bytes": share,
            "target_bytes": int(share * EVICT_TARGET_FRACTION), "source": source}


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


def _run_evict_batches(batches: list, *, peers, deadline) -> tuple:
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
            reports.append((directory, _unlink_batch(directory, paths, peers)))
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
                                pool.submit(_unlink_batch, directory, paths, peers),
                                len(paths)))
            except Exception as exc:  # noqa: BLE001 - no pool means inline, not no work
                snapshot._error(exc, "disk_spill evict submit")
                _state["evict_fallbacks"] += 1
                attempted += len(paths)
                _state["evict_batches_serial"] += 1
                reports.append((directory, _unlink_batch(directory, paths, peers)))
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


def _unlink_batch(directory, paths, peers) -> dict | None:
    """ONE directory's unlink batch, on whatever thread owns it. Never raises.

    `execute_unlink_plan` re-validates every path against the CURRENT digest and the
    peer digests before it deletes anything, and a directory that is itself a symlink is
    refused outright; none of that is weakened by running it on a worker. The peer list
    is a tuple, handed in, so a worker never reads the mutable witness map.
    """
    try:
        return disk_layout.execute_unlink_plan(
            directory, paths, dry_run=False,
            geometry_digest=_state["geometry_digest"], peer_digests=peers,
            for_capacity=True)
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


def _run_ceiling_pass(root, ceiling: int, deadline, budget: dict) -> None:
    """One enforcement pass: walk, decide, shed. Deadline-aware, partial by design.

    Split out of `_enforce_ceiling` so the deadline bookkeeping has exactly one owner
    (the caller's `finally`) and this body reads as the three steps it performs.
    """
    walk = _walk_store(root, deadline=deadline)
    if not walk["ok"]:
        _state["last_reason"] = str(walk["reason"])[:MAX_STRING_CHARS]
        return
    if walk.get("deadline_hit"):
        # Say it where the plan's own truncation is recorded, so one field answers
        # "why is this pass partial" and it is never mistaken for the plan cap.
        _state["plan_truncated"] = True
        _state["ceiling_deadline_hit"] = True
    # THE SIBLING DIGESTS, decided NOW (freshness is the point) and handed to the
    # planner AND the executor. Empty is the fail-safe direction: the plan then sees a
    # foreign digest as `stale_geometry` and sheds it first, exactly as it did before
    # 0.2.29, and `peer_digests_unconfirmed` counts the fact. Since 0.2.30 the same
    # tuple does a second job: it IS the rank count, because `n_effective` is 1 plus the
    # live peers, and that is what turns a 148 GiB ceiling shared by four ranks into
    # four 37 GiB budgets instead of four ranks each defending 148 GiB of one directory.
    peers = _live_peer_digests()
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
    attribution = _attribute_walk_bytes(walk, peers)
    share = _share_of_ceiling(int(ceiling), attribution["divisor"],
                              attribution["source"])
    accountable = int(attribution["accountable_bytes"])
    target = int(share["target_bytes"])
    _state["ceiling_checks"] = int(_state["ceiling_checks"]) + 1
    _apply_attribution(attribution, decision_bytes, share)
    _state["peer_entries_excluded"] = int(_state["peer_entries_excluded"]) + int(
        attribution["peer_entries"])
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
        target_fraction=EVICT_TARGET_FRACTION, peer_digests=peers)
    _state["plan_truncated"] = bool(plan.get("truncated"))
    if not plan["ok"] or not plan["paths"]:
        # Over its share and nothing it is ALLOWED to shed: a different fact from the
        # decline above, and the one case where the filesystem floor (`MIN_FREE_BYTES`,
        # refusal `no_free_space`) is the control that keeps the box alive. Counted, and
        # said, because "the tier wanted to shed and could not" must never be quiet.
        _state["ceiling_passes_nothing_sheddable"] = (
            int(_state["ceiling_passes_nothing_sheddable"]) + 1)
        why = str(plan.get("reason") or "no evictable entry named")[:96]
        _state["last_reason"] = (
            f"over the {target}-byte share by {accountable - target} byte(s) but the "
            f"plan named nothing: {why}")[:MAX_STRING_CHARS]
        return
    _state["peer_files_planned"] = int(plan.get("peer_rank") or 0)
    by_directory: dict = {}
    for path in plan["paths"]:
        by_directory.setdefault(os.path.dirname(path), []).append(path)
    # THE BATCHES, in plan order. Grouping by directory is both the filesystem-friendly
    # shape (a shard's entries leave one directory) and the partition that guarantees no
    # path is attempted by two workers: the groups are disjoint by construction.
    batches = list(by_directory.items())
    reports, attempted, cut_off = _run_evict_batches(batches, peers=peers,
                                                     deadline=deadline)
    freed = 0
    unlinked = 0
    failed = 0
    # FREED BYTES BY OWNER, so the fair-share estimate comes down by the right amount.
    # A rank that shed 10 of its OWN files must not have its `dead` estimate shrink, and
    # a byte shed from a live sibling (which the plan can no longer name) would show up
    # here as a fact rather than as a silent miscount.
    freed_by_owner = {"own": 0, "dead": 0, "peer": 0, "unattributed": 0}
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
            if outcome == "unlinked":
                unlinked += 1
                freed_by_owner[_owner_of_name(action.get("path"), peers)] += int(
                    action.get("size_bytes") or 0)
                # EVERY file this plan actually deleted is journalled, so the
                # scheduler's index stops offering it within one refresh. Without this
                # half the journal the index would keep offering files whose bytes the
                # ceiling already reclaimed - the one direction of index error that
                # costs a reported load failure instead of a plain recompute.
                _journal_write(action.get("path"),
                               int(action.get("size_bytes") or 0), removal=True)
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
                  fingerprint=None, resolver=None) -> dict:
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
                            "the spill path is disarmed (arm expires by itself)")
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
            "units": disk_layout._units("measured_bytes", "sampled_files",
                                        "total_files", "entries_examined",
                                        "skipped_files", "ceiling_checks",
                                        "seed_walks", "deadline_hits",
                                        "budget_seconds", "pass_seconds"),
        },
        # THE FAIR SHARE (0.2.30): THE NUMBER THE CEILING WAS ACTUALLY COMPARED
        # AGAINST, and who else is sharing it. This is the block that makes tonight's
        # over-eviction readable while it is happening: four ranks sharing a root each
        # used to defend the WHOLE ceiling over one directory, and the only way to see
        # that is broken is to see the share each rank believes it owns.
        "ceiling_share": {
            # The bytes this rank is accountable for, and the budget they are measured
            # against. `share_bytes` is `ceiling / n_effective`; `target_bytes` is that
            # share times `EVICT_TARGET_FRACTION`, which is what a pass sheds toward.
            "accountable_bytes": _accountable_bytes(),
            "own_bytes": int(_state["estimated_own_bytes"]),
            "dead_bytes": int(_state["estimated_foreign_dead_bytes"]),
            "peer_live_bytes": int(_state["estimated_peer_live_bytes"]),
            "unattributed_bytes": int(_state["estimated_unattributed_bytes"]),
            "share_bytes": _state["share_bytes"],
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
            "fresh_seconds": PEER_DIGEST_FRESH_SECONDS,
            "min_sightings": PEER_DIGEST_MIN_SIGHTINGS,
            "peer_digest_cap": MAX_PEER_DIGESTS,
            "note": ("`live_digests` are foreign digests whose committed files are "
                     "being written now, so their files shed LAST instead of first. "
                     "`decisions_without_peers` counts the ceiling decisions taken "
                     "with none known, which is the pre-0.2.29 behaviour (foreign "
                     "first) and the fail-safe direction: it never means unable to "
                     "evict, because this rank's own files always remain sheddable."),
            "units": disk_layout._units("live_digests", "learned_digests",
                                        "foreign_files_seen",
                                        "decisions_without_peers",
                                        "files_planned_last_pass",
                                        "peer_digests_dropped", "fresh_seconds",
                                        "min_sightings", "peer_digest_cap"),
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
