"""Restore spilled KV bytes off the SSD tier, through the fork's OWN load path.

WHAT THIS IS
------------
`disk_spill` writes one file per (group-qualified block key, rank) and
`disk_index` tells the SCHEDULER process which of those files exist. This module
closes the loop: when the host-RAM tier comes up short on a prefix, it offers the
SSD's continuation as external tokens, allocates destination CPU slots, has
`disk_read` fetch the bytes on its own thread, and lets the fork's existing
CPU->GPU DMA carry them to the GPU. Nothing here invents a transfer: the fork's
`update_state_after_alloc`, `build_connector_meta`, `start_load_kv` and
`get_finished` do the bookkeeping, the pinning and the copy, because a plugin that
duplicates that bookkeeping is a plugin that leaks refs behind the engine's back.

IT DOES NOT PUBLISH ANYTHING. A restored block is NOT inserted into
`cached_block_hash_to_block` / `_insert_block_hash`, so no later request can find
it; a later miss simply re-reads the file (at the measured 11.5 GB/s that is ~10 ms
per 3,200-token span against ~3.2 s to recompute those tokens). The scope decision
matters: mutating the scheduler's hash index is the one action in this feature
that could corrupt OTHER requests' lookups, and it buys nothing correctness-wise.
Cross-boot adoption is likewise absent - `disk_index` indexes what THIS boot's
ranks journalled, so a fresh boot restores nothing until it has spilled.

WHO IS SERVED FROM DISK IS A BOOT POSTURE, NOT A REQUEST FIELD (0.2.31).
Two per-boot switches answer the one question the read path has: may this boot serve
requests from disk, and if so, WHOSE answer decides a request's own prefix. Both are
import-time False, and both are written only by `set_disk_authority()` and
`set_offer_switch()` over the `/disk` POST - and, since 0.2.33, by this package's own
boot default (`disk_boot.py`), which calls that same setter when
`KVTIER_DISK_BOOT_ENABLE` is at its default. A restart hands the decision back to the
boot default either way; a request field, a config file, the spill arm and every
failure-recovery path still cannot raise either switch.

  * `disk_authority_enabled` - THE DISK IS THE AUTHORITY FOR EVERY REQUEST. The fork's
    host-RAM hit is treated as ZERO tokens for the offer arithmetic, so the disk index
    - not whatever RAM happened to hold - decides what is offered, starting at the
    local GPU-computed boundary. One POST turns it on for every request; a restart
    turns it off; nothing else in this process can turn it on. It opens the offer gate
    by itself, because "serve everyone from disk, with disk deciding" is one decision
    and not two.
  * `offers_enabled` - the disk may EXTEND what the host-RAM tier left short: RAM
    first, the disk filling the gap. The pre-0.2.22 shape, kept because it is the
    mode that spends the fewest disk bytes.

There is NO per-request decider. Until 0.2.31 the authority rule was carried by a
client-supplied marker (`kv_transfer_params.kvtier_disk_restore`), which meant a
request body decided whether its KV came off SSD - acceptable for a measurement, not
for production. The marker, its constant, the `flagged`/`skipped_no_marker` counters,
the `skipped_no_marker` refusal and the `optin`/`all` scope are all GONE: no client
can ask for disk service and no client can decline it.

`computed` is never forced to zero under either switch, because it is the offset the
fork's own copy writes at, not an opinion: zeroing it would aim the copy at the
request's already-cached, ref-counted prefix blocks (shared with every other request
reading them) and leave the newly allocated external region unwritten. The scheduler's
local hit is not this module's to un-learn, so a disk span always starts where the
local prefix ends; what the authority switch changes is the host-RAM hit, which the
fork keeps in exactly one place (`_pending_cpu_hits`) and releases through exactly
one method (`_free_pending_cpu_hit`).

THE THREE ENGINE FACTS THIS IS BUILT ON (all read from the deployed fork)
------------------------------------------------------------------------
1. THE IDENTITY IS DERIVABLE, EXACTLY. The store path spilled
   `gpu_block_pool.blocks[i].block_hash.hex()`, and `block_pool.cache_block_hashes`
   builds that value as `make_block_hash_with_group_id(block_hashes[i], group)`
   where `block_hashes = resolve_block_hashes(request.block_hashes,
   hash_block_size, block_size)` (`core/block_pool.py:269-271, 283, 287`;
   `core/kv_cache_utils.py:74` packs the 4-byte big-endian group suffix, and
   `:2859` is the view that re-indexes the request's hash list at a group's own
   block size). So the key for group `g` at absolute block index `i` is
   re-computed here by CALLING THOSE TWO ENGINE FUNCTIONS - never by re-implementing
   the packing or the view, which is the same rule `disk_layout.split_block_key`
   keeps on the unpacking side. If either function is not importable, this module
   offers nothing.
2. THE MERGED TUPLE IS THE ONLY SOUND WAY IN. `get_num_new_matched_tokens`
   (`manager.py:323-379`) pins its CPU hit and stashes
   `_pending_cpu_hits[req_id] = (cpu_hit_blocks, hit_length, num_computed_tokens)`;
   `update_state_after_alloc` (`manager.py:381-492`) pops it, takes
   `cdiv(num_external_tokens, group_block_size)` LEADING blocks per prefix-cacheable
   group, pairs them with GPU blocks at `num_computed_tokens // group_block_size + i`,
   touches both sides and creates the `LoadRequestState`. Appending our freshly
   allocated destination slots to the END of each group's list and raising the offer
   therefore makes the FORK do the touch, the free of the untaken remainder
   (`_free_pending_cpu_hit`, `manager.py:1307`) and the state bookkeeping - which is
   why this module wraps nothing on that method and owns no refs of its own.
3. SILENCE IS DEATH. `WAITING_FOR_REMOTE_KVS` is released only when the worker
   reports the request id in `finished_recving` (`scheduler.py:3068-3084`) and the
   fork has NO engine timeout for that state: a load that never reports is
   re-prepended forever. So every path out of a disk load ends in a report. The
   failure report is the pair the engine already supports -
   `get_block_ids_with_load_errors()` (`kv_connector/v1/base.py:405-423`, whose
   docstring requires the request to be reported finished anyway) plus
   `finished_recving`, which `scheduler.py:2008-2015` turns into "truncate to the
   last valid prefix and recompute" under the box's
   `kv_load_failure_policy=recompute`. A missing file is therefore a recompute, and
   a recompute is exactly what would have happened without this module.

THE ADOPTION FLOOR IS 3,200 TOKENS, AND IT IS NOT NEGOTIABLE
------------------------------------------------------------
Measured on live traffic: 142 of 158 offers of 40-3,080 tokens were DISCARDED by
the scheduler (it adopts external tokens only in whole scheduler blocks), and the
single offer at or above 3,200 (19,200) was adopted complete. An offer below one
scheduler block is therefore worse than no offer: it parks a request in
`WAITING_FOR_REMOTE_KVS` for a load the scheduler will throw away. Hence
`whole_block_tokens()` - `floor(matched / block_size) * block_size` - and the rule
that an offer of 0 is not an offer.

THE PHANTOM-REPORT HAZARD, AND WHY IT IS CLOSED
-----------------------------------------------
The worker reports "loaded" from `_pending_load_event_indices` ∩ the
`_poll_stream_events` high-water mark (`worker.py:330-352`), and the watermark is
MONOTONE over event indices. So a load event that is *dropped* (never submitted to
the DMA) but left in `_pending_load_event_indices` becomes reportable as soon as a
LATER event completes - the engine would then believe bytes it never received are
valid KV. That is the phantom, and the invariant this module maintains is:

    every index still in `_pending_load_event_indices` had `launch_copy` called
    with it.

The only place this module can skip a submission is `start_load_kv`, when nothing
survives the disk-entry pruning; `release_event()` there discards the index from
that set in the SAME call and reports the req ids itself, and
`_phantom_guard()` re-checks it after the fact and counts a trip if the invariant
is ever violated (a trip means "we nearly believed garbage", and it is the one
counter that must stay 0).

NEVER BLOCK THE ENGINE THREAD, AND NEVER WAIT FOR A READ
---------------------------------------------------------
Reads are submitted at `bind_connector_metadata` and applied at `start_load_kv`,
which is one forward pass later: a read that has NOT settled by then is treated as
a failure (its entries are pruned from the DMA, so no stale bytes are copied, and
the request recomputes). A bounded wait on the engine thread was rejected - this is
a best-effort optimization, and making the serving loop wait for NVMe converts a
restore miss into latency for every other request. `dma_unsettled_entries` is the
counter that says whether that race dominates on a given box.

THE GATE IS TWO SWITCHES, AND NEITHER IS AN ENV VAR FOR RESTORE
---------------------------------------------------------------
A restore needs `VLLM_E2_TIER_DISK_ROOT` (configured) AND `disk_spill`'s arm file
AND the tier's own arm. A fresh boot with the root set but unarmed emits no offer,
allocates no CPU slot, reads no file and installs no behaviour change. Arming the
spill tier (`POST /plugins/kvtier/disk {"op":"arm"}`, which expires) is what arms
the restore, deliberately: the two halves must never be separable in operation,
because restoring bytes that the same switch stopped being written is nonsense.

Stdlib only (`ctypes`, `os`, `sys`, `time`). Every public door is total: it catches
`Exception` (never `BaseException`, so `GeneratorExit` and `CancelledError`
propagate), counts, and returns a classified dict or the value it was handed.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time

from . import (
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_spill,
    disk_store,
    snapshot,
)

_logger = logging.getLogger("kvtier_dev_plugin.disk_restore")

# --- install targets ---------------------------------------------------------

SCHED_MODULE = "vllm.v1.simple_kv_offload.manager"
SCHED_CLASS = "SimpleCPUOffloadScheduler"
WORKER_MODULE = "vllm.v1.simple_kv_offload.worker"
WORKER_CLASS = "SimpleCPUOffloadWorker"
CONNECTOR_MODULE = ("vllm.distributed.kv_transfer.kv_connector.v1."
                    "simple_cpu_offload_connector")
CONNECTOR_CLASS = "SimpleCPUOffloadConnector"
KV_UTILS_MODULE = "vllm.v1.core.kv_cache_utils"

METHOD_MATCHED = "get_num_new_matched_tokens"
METHOD_BUILD_META = "build_connector_meta"
METHOD_UPDATE_OUTPUT = "update_connector_output"
METHOD_BIND = "bind_connector_metadata"
METHOD_START_LOAD = "start_load_kv"
METHOD_GET_FINISHED = "get_finished"
METHOD_LOAD_ERRORS = "get_block_ids_with_load_errors"

SCHED_METHODS = (METHOD_MATCHED, METHOD_BUILD_META, METHOD_UPDATE_OUTPUT)
WORKER_METHODS = (METHOD_BIND, METHOD_START_LOAD, METHOD_GET_FINISHED)

# WHY A HALF IS NOT AN INSTALL. The scheduler half and the worker half land in
# DIFFERENT PROCESSES, and a `installed=YES` printed in a worker says nothing about
# the process that makes the offer. `role_here()` names the halves THIS process has
# classes for, so every verdict this module prints is a verdict about a role.
ROLE_SCHEDULER = "scheduler"
ROLE_WORKER = "worker"
ROLE_BOTH = "scheduler+worker"
ROLE_NONE = "none"

OWNER_ATTR = "__kvtier_dev_diskrestore_owner__"
OWNER_TAG = "kvtier-dev-plugin/disk-restore-1"
ORIGINAL_ATTR = "__kvtier_dev_diskrestore_original__"
LAYER_KIND = "restore"          # the name `backport._layer_kinds()` knows us by
SIBLING_PACKAGE = (__package__ or "kvtier_dev_plugin") + "."

FIELD_LOAD_PATHS = "load_paths"
FIELD_LOAD_CPU = "load_cpu_blocks"
FIELD_LOAD_GPU = "load_gpu_blocks"
FIELD_LOAD_EVENT = "load_event"
FIELD_LOAD_EVENT_TO_REQS = "load_event_to_reqs"

# The scheduler's path entry is an IDENTITY carried in path shape: it knows the
# key and the group, and it knows NOTHING about the rank that will read the file
# (the geometry digest covers the slot width, which is 11,468,800 B on PP0 and
# 12,902,400 B on PP1, plus the rank). So the digest half of the name it sends is
# this sentinel, and the worker REPLACES it with its own digest - see
# `_resolve_worker_path`, where a REAL digest that disagrees is a counted failure
# rather than a silent choice. It is not a file name this boot could ever open:
# `disk_layout.compute_geometry_digest` cannot produce twelve zeros, so a path that
# still carries it is, by construction, unreadable.
IDENTITY_DIGEST_PLACEHOLDER = "0" * disk_layout.GEOMETRY_DIGEST_CHARS

# "We never got an answer about what the fork pinned", as distinct from `None`, which
# means "we asked and the table genuinely holds nothing for this request". The
# difference decides whether a flagged request may be given the disk: an unreadable
# pin table is a pin that cannot be released, and a pin that cannot be released must
# not be walked away from.
_PIN_UNREAD = object()

# --- the proof that the WORKER half of this path is live ---------------------
# THE ONE ASYMMETRY THIS MODULE CANNOT SEE FOR ITSELF. The scheduler and the workers
# are different processes, and an offer is only SAFE if a worker will fill the CPU
# slots the offer allocates. If the scheduler half installed and the worker half did
# not (a fork rename of one worker method, the plugin loaded in one process only), the
# fork's own `start_load_kv` would DMA slots nobody had filled and the request would be
# served from STALE KV - the one failure mode of this design that is silent, so it may
# not be left to luck.
#
# The proof is the rank file the worker ALREADY writes: `worker_diag` publishes this
# module's `disk_restore` section in `rank-*.json`, so a file that is FRESH and says
# `installed: YES` is a worker attesting that its layer is fitted. It adds NO new
# precondition (that directory is already the only way this process can know what
# exists on disk) and it is deliberately not a handshake: no socket, no new file name,
# no engine channel.
RANK_FILE_PREFIX = "rank-"
RANK_FILE_SUFFIX = ".json"
MAX_RANK_FILES_PROBED = 8
# The reader's own ceiling (`llm-api/kv_tier_diagnostics.py`): a file over it is
# unreadable THERE too, so reading more here would buy nothing.
MAX_RANK_FILE_BYTES = 256 * 1024
# A rank file is a heartbeat, not a lease: `worker_diag` rewrites it as the worker
# steps, so this is generous on purpose (an idle engine writes rarely) while still
# refusing to believe a worker that has been gone for an hour.
RANK_FILE_FRESH_SECONDS = 900.0
PROOF_CACHE_SECONDS = 1.0

# --- the bounds, and the unit of every one of them ---------------------------

RESTORE_DEADLINE_SECONDS = 45.0
"""SECONDS an in-flight disk load may stay unreported before the watchdog reports
it as a failure. Arithmetic: the measured cost of one 3,200-token span is ~10 ms
(4 slots x 11,468,800 B = 45.9 MB at the 11.5 GB/s this store measured), and an
offer is capped at `MAX_RESTORE_BLOCKS_PER_OFFER` spans, so the largest legitimate
read is 128 x 45.9 MB = 5.87 GB = ~510 ms of streaming at that 11.5 GB/s - and 45 s
is an ~88x margin on it. (The cap went 64 -> 128 in 0.2.35 because the OPERATOR'S
POOL grew past 64; see the constant for why a backstop below the pool's affordance
is a throughput limit and not a bound. The margin is what makes that raise safe on
time alone: it was ~175x at 64 spans and it is still ~88x here.)

It is ALSO `disk_read.LOAD_DEADLINE_SECONDS` (30 s, the reader's own promise) +
50%: the batch deadline must always fire first, so the watchdog only ever sees the
case the batch deadline could not reach (a batch that settled into nothing, or a
step that never reached `start_load_kv`). 45 s stays far below the point at which an
operator calls a parked request a hang, and a stall it fires on would otherwise be
permanent, because the fork has no timeout for WAITING_FOR_REMOTE_KVS."""

MAX_RESTORE_BLOCKS_PER_OFFER = 128   # SCHEDULER BLOCKS offered per request
#
# THE TOKEN CAP THAT USED TO SIT HERE IS DELETED (0.2.32), and the deletion is the
# whole change. Up to and including 0.2.31 this module fixed the per-offer ceiling at
# 32,000 tokens (10 spans of a 3,200-token block), and that number was an ARTIFICIAL
# ceiling on restoring a prefix from disk: below what the process can carry, below what
# the bounce-buffer pool can afford at the operator's live 256 MiB setting, and below
# the run depth the store reaches anyway. It existed for a real reason - the 2026-10-05
# boot offered EXACTLY 160,000 tokens per request 123,474 times, the engine could not
# place them, and every one came back as a load failure that rescheduled the request -
# but since 0.2.31 the offer path PRICES that reason DIRECTLY: the reader's own
# `disk_read.afford_reads()` says how many bounce buffers the pool can hold at once, the
# offer is cut to it, and `bounded_by` names which bound bit (`pool_budget` /
# `cpu_slots` / `block_cap`). A crude token constant was only ever the accidental guard,
# and the pool-aware offer path replaced it on purpose.
#
# THE RISK KNOWINGLY ACCEPTED BY THIS DELETION (the operator's decision, recorded here
# so the code can see it): with the constant gone there is NO reservation arbiter, so
# two simultaneous offers each compute affordability against the pool's CURRENT free
# bytes, both can believe they fit, and together they over-commit - the open TODO 14.
# Accepted because the live offer rate is 4 offers in 332 lookups, so overlap is
# near-zero, and the crude per-request cap was the only accidental guard against it.
# What is NOT accepted, and is not bought by this change: raising the offer rate is a
# different trade, and the arbiter (see `disk_read.pool_state()` / `afford_reads()`:
# reserve at adoption, release at `finish_batch`) must be scheduled before that.
#
# THE BOUND THAT REMAINS IS THE BLOCK WALK - AND IT IS A BACKSTOP, NOT THE OPERATIVE
# LIMIT, SO IT HAS TO BE SIZED AGAINST THE POOL AND NOT AGAINST A THEORY.
#
# `_disk_run` stops its positional walk at `MAX_RESTORE_BLOCKS_PER_OFFER` blocks, so no
# offer can name more than `MAX_RESTORE_BLOCKS_PER_OFFER x block_size` tokens at any
# prompt length. WHAT ACTUALLY SIZES AN OFFER IS THE POOL: `disk_read.afford_reads()`
# says how many bounce buffers `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES` can hold at once
# and the offer is cut to that, which is why `bounded_by` names `pool_budget` on a
# healthy box more often than `block_cap`. The two numbers must be sized COHERENTLY or
# the backstop silently becomes the limit, and that is exactly what happened on the live
# box: the operator raised the pool to 1 GiB, which affords 1 GiB / 11,468,800 B = ~93
# buffers, and a run of 62 spilled blocks was cut to 15 by pool arithmetic at the old
# setting and then met the 64-block wall as the NEXT ceiling. A cap below what the pool
# affords is not a safety bound any more; it is an undisclosed throughput limit, and it
# is the same mistake the deleted 32,000-token cap above made in a different unit.
#
# So 0.2.35 raises it to 128 - above the ~93 buffers a 1 GiB pool affords, which puts
# the OPERATIVE bound back where it belongs (in the pool, which the engine can actually
# afford) and leaves this constant doing what a backstop is for: bounding a run the
# positional walk could name on a box whose pool was configured far above its RAM.
#
# THE COST OF RAISING IT, stated because a backstop you cannot price is not a bound:
#   * TIME. 128 spans is ~5.87 GB and ~510 ms of streaming at the measured 11.5 GB/s -
#     ~88x inside `RESTORE_DEADLINE_SECONDS` (45 s), ~10x inside the 5,000 ms settle
#     allowance, and `disk_read.LOAD_DEADLINE_SECONDS` (30 s) still fires first.
#   * THE MISSING ARBITER, which is the real one. There is NO reservation arbiter
#     between ranks or requests: two simultaneous offers each price affordability
#     against the pool's CURRENT free bytes, both can believe they fit, and together
#     they over-commit (the open TODO 14 above). Doubling the cap DOUBLES the worst
#     case of that over-commit, so it is not free. What keeps it tolerable is measured,
#     not assumed: 4 offers in 332 lookups on the live box, so overlap is near-zero.
#     RAISING THE OFFER RATE IS A DIFFERENT TRADE and must not be done on this change's
#     strength - the arbiter (reserve at adoption, release at `finish_batch`; see
#     `disk_read.pool_state()` / `afford_reads()`) has to be scheduled before that.
#   * THE READER'S QUEUE, sized independently and NOT priced by the offer path. A read
#     is handed to `disk_read.request_read()` in a tight loop at bind, and that queue is
#     `MAX_QUEUE_DEPTH` = 64 JOBS queued-or-in-flight; job 65 of one load event is
#     refused `queue_full` (returned to the pool, counted in `LOAD_CAPACITY_FAILURES`,
#     and the block it was for arrives as nothing). The offer prices affordability in
#     POOL BYTES ONLY (`afford_reads()` -> `budget // size`), so a 1 GiB pool says "93
#     jobs" to the offer path while the reader's own queue says 64. AT THE LIVE 96 MiB
#     DEFAULT THIS IS UNREACHABLE (96 MiB affords 8 buffers), and it was already 4 jobs
#     over at 64 spans x 1 positional group + 4 boundary states = 68; raising the cap to
#     128 spans moves it to 132 jobs, so the gap becomes the expected case on a box that
#     raises its pool. IT IS AN OPERATOR DECISION, NOT A TUNE: either the queue rises
#     with the pool (one bounded pile per rank process, ~11.5 MB of buffer each, already
#     paid for by the pool budget) or the offer path prices `MAX_QUEUE_DEPTH` next to
#     the pool budget and the ceiling it names in `bounded_by` stops being a half-
#     truth. The
#     second is the honest one; neither belongs in this change unasked.
#   * CPU SLOTS, unchanged: `MAX_SLOTS_PER_OFFER` (256) is a separate bound and it
#     binds FIRST on a multi-group hybrid, where 128 blocks x 4 groups is 512 slots and
#     `cpu_slots` cuts the offer to 64 blocks anyway. Raising this constant therefore
#     changes nothing for a 4-group model and everything for the live single-cacheable-
#     group one (128 + 4 boundary states = 132 slots, inside 256).
#
# Neither bound is what stops a 109k-token prompt (128 x 3,200 = 409,600 tokens is far
# above it): what does is bounce-buffer affordability and the store's own depth - the
# deepest contiguous run the operator measured on the live store was ~11 spans, because
# spill follows RAM-tier parking and the CPU arena is ~187 blocks per rank. Raising a
# backstop removes a ceiling nobody needed; it does not make long prefixes restorable.
DEFAULT_OFFER_BLOCK_SIZE_TOKENS = 3_200
"""TOKENS: the scheduler block size used to NAME the block cap in tokens before the
first lookup reveals the engine's real one. The boot line and the payload publish a
token figure, and both run in processes that will never make an offer (a worker rank)
- so they need a number, and the honest one is the live box's block size with its
source published beside it (`block_size_source=fallback`). The offer path always passes
the manager's real `block_size`, so no offer has ever been sized by this constant."""

MAX_SLOTS_PER_OFFER = 256            # CPU slots allocated per request
MAX_TRACKED_LOADS = 64               # REQUESTS with an outstanding disk load
MAX_TRACKED_SLOTS = 4096             # cpu-slot -> identity entries
MAX_INVALID_BLOCKS = 4096            # GPU block ids awaiting one report
MAX_STRING_CHARS = 160
MAX_BOOT_LINE_CHARS = 160
CPU_SLOT_RESERVE = 64                # CPU slots the offer must leave free
MAX_PATHS_PER_STEP = 512             # load_paths entries one step may carry
MIN_WITNESS_LOG_SECONDS = 30.0       # SECONDS between witness re-emissions
MAX_SUPPRESSED_REQUESTS = 4096       # REQUESTS black-holed after a failed load
MAX_RACE_REOFFERS_PER_REQUEST = 1    # RE-OFFERS one request may claim after a load it
                                     # LOST A RACE (0.2.32): a spent allowance, never
                                     # refilled, so a race costs at most two offers
MAX_RANK_COUNTER = 1_000_000_000     # ceiling on a counter read from a rank file
MAX_SPANS_REPORT = 16                # PARTIAL SPANS one shape report may name

# --- what an offer costs the bounce-buffer pool (0.2.31) ---------------------
#
# THE CONSTRAINT THE OFFER PATH NEVER PRICED. A read does not land in a KV slot: it
# lands in a host-RAM bounce buffer that `disk_read` takes at ISSUE and returns at
# the APPLY POINT - one forward pass later (`disk_read.counters()`'s pool note: one
# offer needs the pool to hold the whole offer). So the number of JOBS an offer
# carries (spans x positional groups + one boundary-state block per align-mode
# group, the same `needed` the CPU-slot arithmetic uses) times the bytes in one
# spilled file must fit the pool, or the surplus jobs are refused at bind as
# `buffer_cap` - which the scheduler never hears about, because the only worker-to-
# scheduler channel is a set of GPU block ids.
#
# THE LIVE FAILURE THIS PRICES. The pool is `VLLM_E2_TIER_DISK_READ_BUFFER_BYTES`,
# 134,217,728 B per PROCESS on the box; one file is 11,468,800 B there, so 11 jobs fit
# and a 65,000-token offer wanted 13 (9 span blocks of group 5 plus one boundary-state
# block in each of groups 0-3). The surplus died at bind, was pruned at the apply
# point into `invalid_block_ids`, rescheduled the request as a KV load failure, and
# `note_reported` then read that same set back and fused the request forever.
#
# THE FALLBACK PRICE IS THE LARGER FILE, ON PURPOSE. The current build writes TWO
# legitimate widths CONCURRENTLY because the slot width is rank-local and the digest
# carries it: 11,468,800 B on PP0 and 12,902,400 B on PP1 (`disk_spill
# measured_slot_bytes`). An offer spans EVERY rank and the pool is per process, so
# pricing a PP0 offer and being wrong by 1.13x is exactly the cliff this constant
# exists to remove: when no rank file has published its width yet, price the bigger
# shape and cap a little harder. When they have, use the largest published width -
# still the conservative answer, and exact for the rank with the widest slots.
POOL_PRICE_BYTES_PER_JOB_FALLBACK = 12_902_400

MAX_LOAD_FAILURE_RECORDS = 32        # refused/failed LOADS this process remembers
MAX_FAILURES_PER_BIND = 4            # records ONE bind may write (a big step refuses
                                     # many jobs at once; the ring wants the episode)
MAX_FAILURES_PER_RANK_FILE = 2       # records ONE rank file may carry (byte budget)
MAX_FAILURES_PUBLISHED = 6           # records a section or an op surfaces
MAX_FAILURE_STRING_CHARS = 80        # a string inside one failure record
REFUSAL_LOG_MIN_SECONDS = 30.0       # SECONDS between load-failure lines of one kind
POOL_PRICE_MAX_IDENTITIES = 64       # index lookups ONE price walk may make

# --- the settle wait, and the one number that sizes it -----------------------
#
# THE RULE, IN ONE LINE (0.2.32): `granted = min(SETTLE_MS, the load's own remaining
# deadline)`. Nothing else. The operator's decision, verbatim: "give it up to 5 s to
# read; only when something bad happens do we ignore; do not compute how much time it
# needs."
#
# WHAT USED TO SIT HERE, AND WHY IT IS GONE. Until 0.2.31 the allowance was COMPUTED:
# bytes still in flight divided by an ASSUMED read throughput, times a safety margin,
# floored at one scheduler tick and capped by the environment variable. Every one of
# those inputs was a guess, and the boot measured the guesses to be wrong in the
# direction that costs a request: on 2026-10-06 a 28,583-token prompt was offered 12
# jobs / 137.6 MB and the formula granted 23.6 ms for reads whose measured bind-to-apply
# race gap was 89.7 / 97.1 / 129.7 / 135.8 ms. The request recomputed 28,583 tokens
# (12.26 s, `cached_tokens: 0`) to save 23.6 ms of a step. A formula that guesses the
# speed of the device is a formula that loses to the device, and the loss is paid by the
# request that never asked for a special favour.
#
# THE COUNTER THAT COULD NOT SEE THAT IS DELETED WITH IT. `budget_over_recompute`
# compared the granted wait against the granted budget's own arithmetic, so a load that
# was dropped at 23.6 ms and then cost a 12-second recompute read as `over_recompute: 0`
# - a clean bill of health for the bug. What remains is the honest version: how many
# batches spent the WHOLE allowance and still did not settle (`settle_allowance_spent`).
#
# WHAT THE DEFAULT MEANS NOW. `SETTLE_MS_DEFAULT` was 0.0 - never wait - on the argument
# that a step that blocks on NVMe by accident is worse than a tier that never reads.
# That argument still holds and the operator accepted its cost deliberately: a flat
# 5,000 ms is permission to block the worker's load path for up to five seconds, and it
# is bounded by the load's own deadline (`disk_read.LOAD_DEADLINE_SECONDS`, 30 s) and by
# `SETTLE_MS_MAX` below, and `VLLM_E2_TIER_DISK_SETTLE_MS=0` still means never wait and
# is still reachable. The wait is a POLL with three exits (the bytes, the allowance, the
# read's own expiry), so the five seconds is a ceiling on a wait that usually ends in
# single-digit milliseconds - `settled_on_arrival` counts the ones that cost nothing.
SETTLE_MS_DEFAULT = 5_000.0           # MILLISECONDS; the flat allowance one apply point
                                      # may wait for the bytes it is about to apply
SETTLE_MS_MAX = 60_000.0              # a cap that is not this is a typo, not a budget
SETTLE_POLL_SECONDS = 0.0005          # SECONDS between looks while waiting; half a
                                      # millisecond is below every interesting read
                                      # time and above a busy-wait's cost
MAX_SETTLE_SAMPLES = 512              # WINDOW samples kept for a median; min/max are
                                      # boot-wide regardless

# --- the two boot switches ----------------------------------------------------
# WHO IS SERVED FROM DISK, AND BY WHOSE ANSWER. Both are per-boot in-process booleans,
# both import-time False, and the ONLY writers are `set_disk_authority()` and
# `set_offer_switch()`. Neither reads the environment, and the test suite asserts that
# by AST; no config file, no arm file, no recovery path and no request field can raise
# either one.
#
# SINCE 0.2.33 THERE IS A THIRD CALLER OF THE SETTER, AND NOT A THIRD KNOB: the
# package's own boot default (`disk_boot.py`) calls `set_disk_authority(True)` at
# connector creation when `KVTIER_DISK_BOOT_ENABLE` is unset. That is the same two
# writers, the same gate, the same log line - it is the operator's own POST, issued by
# the boot on his behalf because every restart used to silently revert to a tier that
# serves nothing. The variable DISABLES that (0/false/off/no); it cannot raise anything
# the boot default is not already on, which is the property
# `VLLM_E2_TIER_DEV_BOOT_INERT` has always had. What it must NEVER become is a read of a
# PREVIOUS boot's arm file or persisted posture: see `disk_boot.py`'s prohibition and
# the 2026-10-03 entry in the sibling llm-api repo.
#
# WHY THERE ARE TWO AND NOT ONE. They answer different questions and a boot can sit in
# either posture:
#
#   disk_authority = True  -> THE DISK IS THE AUTHORITY FOR EVERY REQUEST. Each
#     request's host-RAM hit is treated as ZERO tokens for the offer arithmetic and
#     released back to the pool, so the disk index decides the span from the local
#     GPU-computed boundary. This is "serving everyone from SSD": the expensive
#     posture, the one that must be confirmable at a glance, and the one that opens
#     the offer gate BY ITSELF - "serve every request from disk, with disk deciding"
#     is a single decision and must not need two POSTs.
#   offers_enabled = True  -> the disk may EXTEND a request whose host-RAM prefix came
#     up short: RAM first, the disk filling the gap. The cheaper posture, and the
#     pre-0.2.22 shape.
#
# WHY THERE IS NO THIRD KNOB AND NO REQUEST FIELD. Until 0.2.31 the authority rule
# was carried by a client-supplied marker,
# `kv_transfer_params.kvtier_disk_restore = true`, and `scope` chose whether that
# marker was required (`optin`) or ignored (`all`). That meant a REQUEST BODY decided
# whether its KV came off SSD - fine for a measurement, not acceptable approaching
# production. The marker, its constant, the `flagged`/`skipped_no_marker` counters, the
# `skipped_no_marker` refusal reason and the `optin`/`all` scope are all retired here:
# with the marker gone, `scope`'s question ("WHICH requests?") has exactly one answer
# ("every request"), so the knob is retired rather than kept as a constant. What is
# left is the axis that always mattered: WHAT the disk may do.

DISK_AUTHORITY_DEFAULT = False
"""THE IMPORT-TIME value of the switch, stated as a constant so it is the assertion.

A process that has run no boot gives disk authority to NOBODY, and that stays true in
0.2.33: what changed is that a boot may now RAISE the switch by calling
`set_disk_authority(True)` when `KVTIER_DISK_BOOT_ENABLE` is at its default (see
`disk_boot.py`). The constant stays False for two reasons that are both load-bearing.
First, an import that turned serving behaviour on would make the plugin's import itself
a side effect, and this package's import is read by the read side, by tests, and by a
proxy host that is not serving anything. Second, `disk_boot`'s fail-safe direction is
"every outcome except ARMED leaves this boot disarmed", and that is only expressible
against a default-off starting point.

What the constant does NOT mean any more: "this boot will not be serving from disk". For
that question the payload has `serving_as`, and the log has the one
`KVTIER disk boot default=` line. `tests/test_disk_authority_switch.py` pins the value,
the writer set, and that no configuration vocabulary names this switch.
"""

OFFER_REFUSALS = ("not_configured", "disarmed", "tier_disarmed", "bad_call",
                  "bad_result", "unaligned_local_prefix", "skip_read_requested",
                  "no_tokens_left", "no_block_hashes", "no_key_helpers",
                  "no_eligible_group", "ram_shape", "no_disk_run",
                  "disk_run_below_one_block", "no_cpu_slots", "alloc_refused",
                  "no_channel", "path_build_failed", "index_refresh_failed",
                  "too_many_loads", "ram_hit_not_block_aligned",
                  "no_worker_proof", "offers_disabled", "workers_not_servicing",
                  "suppressed_after_failure",
                  "pin_release_unavailable", "incomplete_group_set",
                  "no_boundary_anchor", "no_null_block", "no_pool_budget")

# --- what a group's blocks ARE, as a claim reads them (0.2.27) ----------------
#
# THE TWO CONTRACTS THE FORK ACTUALLY HAS. A prefix-cache hit is a list of CPU
# blocks per group, and how long that list is - and how many of its entries carry
# bytes - is decided by the group's own manager class, not by this module and not
# by the model's name:
#
#   `positional`      `has_positionally_stable_blocks is True` (the base answer,
#                     `single_type_kv_cache_manager.py:62-69`). One real block per
#                     scheduler block of the claimed run, each named by that block's
#                     own group-qualified cumulative hash. Full attention, sliding
#                     window, and every group the fork's own positional spill scan
#                     (`manager.py:872-876`) is willing to walk.
#
#   `boundary_state`  `mamba_cache_mode == "align"`, which is EXACTLY the condition
#                     the fork's own override uses to answer
#                     `has_positionally_stable_blocks = False`
#                     (`single_type_kv_cache_manager.py:1494-1497`). Such a group's
#                     hit for a run of N blocks is
#                     `[null_block] * (N - 1) + [cached]` (`:1591-1607`): ONE real
#                     recurrent state at the END of the run and nothing for the
#                     interior. It is NOT unnamed. The align hand-off publishes the
#                     state block's own group-qualified hash (`:2051-2080`,
#                     `:1955-1963`) and the tier's store completion registers the CPU
#                     block under that same hash (`manager.py:1050-1078`), which is
#                     the name a lookup computes
#                     (`resolve_block_hashes(block_hashes, hash_block_size,
#                     block_size)[i]`, `:1591-1601`) and the name this module already
#                     resolves a spilled file from. The bytes exist, they are
#                     findable, and ONE of them per run end is the whole cost.
#
#   `unstable`        False positional stability for a reason that is NOT the align
#                     hand-off. No contract this module can implement: the gate
#                     refuses, loudly, because a claim would cover a group whose
#                     blocks nobody names.
#
# THE ONE RULE ALL THREE SERVE: a claim is made only for bytes this tier can NAME
# and READ BACK for EVERY eligible group. 0.2.25 enforced it by refusing every
# hybrid model; 0.2.27 enforces it by implementing the second contract, so the gate
# opens BY CONSTRUCTION on a model whose cacheable groups all have a contract and
# stays shut - with the same `incomplete_group_set` refusal, word for word - on
# anything with a cacheable group that has none.
CONTRACT_POSITIONAL = "positional"
CONTRACT_BOUNDARY = "boundary_state"
CONTRACT_UNSTABLE = "unstable"
CONTRACT_GEOMETRY = "geometry"
CONTRACT_OUTSIDE = "outside"
CONTRACT_UNKNOWN = "unknown"
CONTRACTS_IMPLEMENTABLE = (CONTRACT_POSITIONAL, CONTRACT_BOUNDARY)

# The one value of `MambaSpec.mamba_cache_mode` the fork reads as "I hand off
# boundary states" (`single_type_kv_cache_manager.py:1497`, `:1516`, `:1703`,
# `:2051`), spelled exactly as the fork spells it.
MAMBA_CACHE_MODE_ALIGN = "align"

# The align-mode hit shape needs real null blocks between a run's start and its
# boundary (`[null_block] * (N - 1)`), and the null block is the CPU pool's own -
# the same object the fork's own Mamba hit puts in those slots (`BlockPool.null_block`,
# read at `single_type_kv_cache_manager.py:142`). A pool that does not expose one
# cannot have a boundary group restored at all.
NULL_BLOCK_ATTR = "null_block"

# EVERY REFUSAL SPELLED OUT. `refused` says WHICH counter moved; this says what the
# counter MEANS, so "why is there no restore?" is answerable from one payload read
# instead of from the source. A name with no entry here is a bug in this table, and
# `scheduler_section()` publishes the gap as `<name>: "undefined reason"` rather
# than hiding it.
OFFER_REFUSAL_WHY = {
    "not_configured": "no store to read: VLLM_E2_TIER_DISK_ROOT is unset in this "
                      "process, so there is no directory a file could come from",
    "disarmed": "the spill tier is disarmed (its arm file is absent, unreadable or "
                "fenced to another boot - never 'expired': the arm is a latch), and "
                "restore is deliberately not separately armable",
    "tier_disarmed": "the KV tier itself is disarmed, so no lookup reaches the "
                     "connector at all",
    "bad_call": "the scheduler handed this wrapper something unusable (a request "
                "id, a computed-token count or a result shape it cannot read)",
    "bad_result": "the fork's own get_num_new_matched_tokens() returned a shape "
                  "this module cannot extend",
    "unaligned_local_prefix": "the local prefix does not end on a boundary every "
                              "group can address, so a disk span could not be "
                              "appended to it",
    "skip_read_requested": "the request opted out of external reads (the fork's "
                           "own skip_read flag)",
    "no_tokens_left": "the whole prompt is already computed, so nothing is left "
                      "to serve from disk",
    "no_block_hashes": "the request carries no block hashes to name a file with",
    "no_key_helpers": "vllm.v1.core.kv_cache_utils is not importable here, so the "
                      "engine's own key-packing/view functions cannot be called "
                      "(this module refuses to re-implement them)",
    "no_eligible_group": "no prefix-cacheable group has block_size == the "
                         "scheduler block size under a contract this tier can "
                         "restore (positional, or an align-mode boundary state), so "
                         "a merged tuple cannot be extended one whole block at a "
                         "time",
    "ram_shape": "the host-RAM hit's per-group block lists do not add up to its "
                 "own token count, so the merged tuple could not be trusted",
    "no_disk_run": "the index holds no contiguous span at the next block for every "
                   "eligible group (the usual answer for a prefix that was never "
                   "spilled, or that a restart dropped from the index)",
    "disk_run_below_one_block": "the disk run is shorter than one scheduler block, "
                                "so offering it would park the request for a load "
                                "the scheduler discards",
    "no_cpu_slots": "the CPU pool cannot spare a whole span's slots above the "
                    "reserve, so a destination does not exist",
    "alloc_refused": "the CPU pool refused the allocation outright",
    "no_channel": "no diagnostics directory, so no identity could be handed to a "
                  "worker",
    "path_build_failed": "the file name for an identity was rejected by the layout "
                         "rules",
    "index_refresh_failed": "the scheduler-side index could not be refreshed from "
                            "the journals, so a stale view was refused instead of "
                            "read",
    "too_many_loads": "the tracked disk-load cap is already reached",
    "ram_hit_not_block_aligned": "the host-RAM hit is not a whole number of "
                                 "scheduler blocks, so appending disk blocks would "
                                 "break the fork's own alignment assert",
    "no_worker_proof": "no fresh rank file attests a fitted worker half, so the "
                       "CPU slots an offer allocates might never be filled - the "
                       "one refusal that exists to prevent serving stale KV",
    "offers_disabled": "neither boot switch is up: no offer switch "
                       "(`{\"op\":\"offers\",\"enabled\":true}`, RAM first and the "
                       "disk filling the gap) and no disk authority "
                       "(`{\"op\":\"authority\",\"enabled\":true}`, every request "
                       "served from SSD with disk deciding). Both are per boot, both "
                       "start OFF, because arming the spill tier for a different "
                       "experiment once put an un-requested disk offer in front of "
                       "live traffic",
    "workers_not_servicing": "the scheduler is armed but at least one live rank "
                             "file says its worker half will NOT service a disk "
                             "load (disarmed in its own process, or a boot the "
                             "arm file outlived): offering would allocate slots "
                             "nobody fills, the fork would report the load failed "
                             "and the request would be rescheduled - the exact "
                             "loop that starved the box on 2026-10-05",
    "pin_release_unavailable": "disk authority is up and the fork had already "
                               "PINNED host-RAM blocks for this request's own hit, "
                               "but this build's manager exposes no "
                               "`_free_pending_cpu_hit` to put them back: offering "
                               "disk bytes instead would strand those pins for the "
                               "rest of the boot, so the request keeps its RAM hit "
                               "and the disk stays out of it",
    "suppressed_after_failure": "a disk load already failed for THIS request (or "
                                "for the span its offer was built from), and a "
                                "failed load must never be re-offered: the "
                                "reschedule that follows calls "
                                "get_num_new_matched_tokens again, and re-offering "
                                "the same span is an infinite loop, not a retry",
    "incomplete_group_set": "THE CLAIM-MUST-MATCH-THE-BYTES GATE (0.2.25; opened "
                            "for align-mode groups by 0.2.27). At least one group "
                            "CAN be restored, but another prefix-cacheable group "
                            "CANNOT, and offering the restorable subset claims "
                            "every group: get_num_new_matched_tokens advances "
                            "num_computed_tokens for the WHOLE model, "
                            "update_state_after_alloc's `n_ext_g == 0 -> continue` "
                            "is a SILENT no-op for the group we did not put in the "
                            "tuple, and nothing recomputes the claimed range "
                            "because a group that was never in the tuple reports "
                            "no failure. An align-mode Mamba/GDN group is NO LONGER "
                            "a blocker: its hit contract is [null_block]*(n-1) plus "
                            "ONE real boundary-state block "
                            "(`single_type_kv_cache_manager.py:1591-1607`), that one "
                            "block IS named by the align hand-off, and this tier "
                            "spills and reads it now. What still fires this refusal "
                            "is a cacheable group with NO implementable contract: a "
                            "non-align group that is not positionally scannable, a "
                            "group outside the group table, or one whose block_size "
                            "is not the scheduler's. Refusing costs one recompute; "
                            "offering serves a wrong answer, which is what the "
                            "live box did on 2026-10-06 at 00:08: a mechanically "
                            "perfect read (9 of 9 applied per rank, no failures, "
                            "18 ms) that answered garbage because the GDN "
                            "boundary state was claimed and never filled",
    "no_boundary_anchor": "the positional groups have files for the whole run but NO "
                          "run end within the offer's cap has a boundary-state file "
                          "in EVERY align-mode group, so the one state block those "
                          "groups need does not exist at any length this offer could "
                          "claim. A run without it is a claim on a recurrent state "
                          "nobody wrote - the 00:08 shape - so the run is refused "
                          "rather than shortened into a lie",
    "no_null_block": "an align-mode group is restorable but the CPU pool exposes no "
                     "`null_block`, so the `[null_block]*(n-1)` half of the fork's "
                     "own hit shape cannot be built; offering the real state block "
                     "alone would land it on the run's FIRST block instead of its "
                     "last, which is a wrong state read back as a right one",
    "no_pool_budget": "the bounce-buffer pool cannot carry even ONE job of this "
                      "offer's byte shape, so no offer is made at all. A read lands "
                      "in a host-RAM buffer taken at issue and returned a forward "
                      "pass later, so an offer the pool cannot hold IN FULL is "
                      "refused at bind as `buffer_cap`, its GPU blocks are reported "
                      "as load errors, the request is rescheduled and then fused by "
                      "`note_reported`: the 0.2.30 shape, where the scheduler made "
                      "an offer the worker could not physically service. Refusing "
                      "costs a recompute; offering costs the request its RAM hit "
                      "too. The reason carries the byte figures (pool budget, price "
                      "per job, jobs the offer needed) because publishing those is "
                      "the whole point: nobody could see them before",
}
WORKER_FAILURES = ("path_count_mismatch", "bad_path", "identity_disagreement",
                   "digest_disagreement", "fingerprint_incomplete", "no_channel",
                   "read_failed", "file_missing", "size_mismatch", "short_read",
                   "read_error", "not_ours", "not_settled", "slot_plan_failed",
                   "copy_failed", "batch_cap", "queue_full", "buffer_cap",
                   "bad_batch", "bad_job", "bad_size", "incomplete",
                   "deadline_expired", "thread_dead", "not_tracked", "abandoned")

# --- WHAT A FAILURE KIND COSTS (0.2.32) --------------------------------------
#
# WHY THIS PARTITION EXISTS. A span that fails to land used to cost the SAME no matter
# WHY: the GPU block went to `get_block_ids_with_load_errors`, the engine reset the
# request, and `suppress_load` black-holed the request and the span for the rest of the
# boot. That is the right cost for BAD BYTES and the wrong cost for a RACE. On
# 2026-10-06 a 23.6 ms settle grant lost a 135.8 ms race in one group of one span and
# the bill was a 28,583-token recompute PLUS `suppressed_requests: 3` - a boot-length
# fuse for a millisecond-scale timing loss. The fuse call site already wrote a reason
# containing `sorted(_recent_failure_kinds())`, so the code knew which it was and
# decided without looking.
#
# THE RULE: a timing loss may cost the tokens and must not cost the next offer. A data
# failure keeps the fuse exactly where it was.
#
# `LOAD_TIMING_FAILURES`: the bytes are FINE, the read simply had not landed when the
# engine arrived (`not_settled`, and its deadline-blowing sibling `deadline_expired`).
# `LOAD_CAPACITY_FAILURES`: the bytes are FINE and the read was never issued, because a
# bounded pool or queue was full at that instant - a momentary condition that the NEXT
# offer re-prices, not corruption. `LOAD_DATA_FAILURES`: everything else - a wrong
# digest, a short read, a size mismatch, a failed copy, a step whose lists disagree, an
# unattributable loss. These keep the fuse; the partition is published in
# `failure_kind_classes()` so an operator can check it against a payload, and
# `test_disk_restore_races.py` pins every one of the 26 names into exactly one class.
#
# THE JUDGEMENT CALLS, spelled out so they can be argued with:
#   * `batch_cap` is DATA-class. It means an offer was made larger than the worker's
#     batch table - an internal bound violation, and a retry would reproduce it.
#   * `not_tracked`, `abandoned` and `read_failed` are DATA-class too. They are
#     UNATTRIBUTABLE, and "nobody saw what happened" is not evidence of a race: fusing
#     on the unknown case is the fail-closed direction, and it is what keeps the
#     2026-10-05 loop un-reproducible.
#   * `no_channel` / `thread_dead` are DATA-class: the read path itself is broken,
#     which is "something bad" in the operator's own words.
LOAD_TIMING_FAILURES = ("not_settled", "deadline_expired")
LOAD_CAPACITY_FAILURES = ("queue_full", "buffer_cap")
LOAD_DATA_FAILURES = tuple(
    name for name in WORKER_FAILURES
    if name not in LOAD_TIMING_FAILURES and name not in LOAD_CAPACITY_FAILURES)


def failure_kind_class(kind) -> str:
    """Which of the three classes one failure name belongs to.

    An unknown name answers `"data"`: `WORKER_FAILURES` is a closed vocabulary and a
    name outside it means something drifted, which is not a race we should keep
    offering into.
    """
    name = str(kind or "")
    if name in LOAD_TIMING_FAILURES:
        return "timing"
    if name in LOAD_CAPACITY_FAILURES:
        return "capacity"
    return "data"


def _attributable_failure_kinds(limit: int = 6) -> set:
    """The failure kinds that name a CAUSE, excluding this process's own report echoes.

    WHY THIS IS NOT `_recent_failure_kinds`. `note_reported` writes a record for the
    REPORT SHAPE the engine used (`failed_recving`, `invalid_block_ids`) before it
    decides anything, and those two names are not in the `WORKER_FAILURES` vocabulary,
    so `_note_load_failure` files them under `read_failed` - a DATA kind. Reading the
    plain ring here would therefore make every report look like a corruption from the
    moment the first one arrived, and a race could never be spared: the fix would be
    inert and would read as if it worked. This reader keeps what the code that TOUCHED
    THE BYTES wrote (`where` = `apply`, `bind`, `offer`, here or carried by a live rank
    file) and drops the records that only name the channel the bad news arrived on.

    A kind outside the `WORKER_FAILURES` vocabulary is dropped too, so an empty set
    means "no cause is on the table" and `classify_failure_kinds` answers `unknown`.
    """
    kinds: set = set()
    try:
        wanted = max(1, int(limit))
        rows = ([dict(item) for item in list(_load_failures)[:wanted]]
                + [dict(item) for item in
                   list(_state.get("rank_failure_records") or [])[:wanted]])
        for item in rows:
            if str(item.get("where") or "") == "report":
                continue
            kind = item.get("kind")
            if kind and str(kind) in WORKER_FAILURES:
                kinds.add(str(kind))
    except Exception as exc:  # noqa: BLE001 - a decision is never worth a fault
        snapshot._error(exc, "disk_restore attributable failure kinds")
    return kinds


def classify_failure_kinds(kinds) -> str:
    """What a SET of failure kinds says about the loss: `data`, `timing`, `unknown`.

    THREE ANSWERS, ONE RULE. Any data kind present means DATA - one corrupted read in
    the window outranks any number of races, so the fuse stays. Otherwise, any timing or
    capacity kind means TIMING. An empty set means UNKNOWN, and UNKNOWN FUSES: the
    scheduler's knowledge of a worker's kinds arrives through the rank file, so a loss
    whose worker has not yet been read looks exactly like a loss nobody diagnosed, and
    guessing "race" there is how the loop that starved the box comes back.
    """
    seen = {str(kind) for kind in (kinds or ()) if kind}
    if not seen:
        return "unknown"
    if any(failure_kind_class(name) == "data" for name in seen):
        return "data"
    if any(failure_kind_class(name) in ("timing", "capacity") for name in seen):
        return "timing"
    return "data"


def failure_kind_classes() -> dict:
    """THE PARTITION, as a payload read, with the decision rule beside it.

    Published because the partition IS the policy for what a failed load costs, and a
    reader who cannot see it cannot argue with it. `unattributable_are_data` names the
    fail-closed branch so a quiet `races_not_fused` can be read as "we could not tell"
    and not as "there were no races".
    """
    return {"timing": list(LOAD_TIMING_FAILURES),
            "capacity": list(LOAD_CAPACITY_FAILURES),
            "data": list(LOAD_DATA_FAILURES),
            "rule": ("any data kind visible -> fuse; only timing/capacity kinds -> "
                     "spare the request and the span and let the one-per-request "
                     "re-offer decide; nothing visible -> fuse (unknown is not proof "
                     "of a race)"),
            "read_from": ("the records the code that touched the bytes wrote "
                         "(`where` = apply/bind/offer, here or carried by a live rank "
                         "file). The engine's REPORT SHAPES (`failed_recving`, "
                         "`invalid_block_ids`) are excluded: they name the CHANNEL the "
                         "bad news arrived on, and `_note_load_failure` files an "
                         "unknown name under `read_failed` - a data kind - so reading "
                         "them would make every report look like a corruption"),
            "unattributable_are_data": True,
            "kinds_total": len(WORKER_FAILURES)}


def _grant_race_reoffer(request_id) -> None:
    """THE SMALLEST LOOP RULE (0.2.32): one re-offer per request after a race.

    Not a TTL and not a window: a per-request allowance of exactly
    `MAX_RACE_REOFFERS_PER_REQUEST` re-offers, SPENT when used and never renewed.
    Without it, removing the fuse would remove the only bound on the retry loop - a race
    on a sick device would re-offer forever, five seconds of waiting a step at a time.
    With it, a request that loses the race twice gets no third offer and the loop is
    bounded at two, whichever way the loss goes.
    """
    try:
        key = str(request_id or "")[:256]
        if not key:
            return
        if key not in _race_reoffer and len(_race_reoffer) >= MAX_SUPPRESSED_REQUESTS:
            _race_reoffer.pop(next(iter(_race_reoffer)), None)   # oldest first
        # `setdefault`, not assignment: a request whose allowance is already SPENT (0)
        # must not have it refilled by the second race it caused.
        _race_reoffer.setdefault(key, MAX_RACE_REOFFERS_PER_REQUEST)
    except Exception as exc:  # noqa: BLE001 - an allowance is never worth a fault
        _logger.debug("kvtier-dev-plugin: DISK[load-fail] re-offer allowance not "
                      "granted for %s: %s", request_id, type(exc).__name__)


def _spend_race_reoffer(request_id) -> bool:
    """Claim this request's one re-offer, or say it has none left."""
    try:
        key = str(request_id or "")[:256]
        if not key or int(_race_reoffer.get(key, 0)) < 1:
            return False
        _race_reoffer[key] = 0
        _state["race_reoffers_granted"] += 1
        return True
    except Exception:  # noqa: BLE001 - an allowance is never worth a fault
        return False

_state: dict = {
    # install / posture. `installed` is THIS PROCESS's verdict about the halves it
    # could reach; `sched_fitted` is the one that answers "can this process make an
    # offer?" and `methods_fitted` names what is actually wrapped here.
    "installed": False, "class": None, "installs": 0, "reinstalls": 0,
    "uninstalls": 0, "skip_reason": None, "nested": [], "errors": 0,
    "fallbacks": 0, "boot_line": None,
    "sched_fitted": False, "worker_fitted": False, "methods_fitted": [],
    "install_refusals": 0, "install_deferrals": 0, "missing_methods": [],
    "install_refusal_line": None, "install_deferral_line": None,
    # WHO IS SERVED FROM DISK, AND WHOSE ANSWER DECIDES A REQUEST'S OWN PREFIX. Two
    # independent per-boot switches, both import-time False, both written only by the
    # two setters - by the `/disk` dev verb, and since 0.2.33 by this package's own boot
    # default calling the same setter (`disk_boot.py`; `KVTIER_DISK_BOOT_ENABLE` turns
    # that off). Arming the SPILL tier is neither of them:
    # on 2026-10-05 an arm raised for a different experiment offered 19.7 BILLION
    # tokens and starved the harness, which is why offering is its own decision and
    # `disk_authority` (the expensive posture: every request's KV comes off SSD and its
    # host-RAM hit is given back) is its own decision on top of it.
    "offers_enabled": False, "disk_authority": DISK_AUTHORITY_DEFAULT,
    "offer_switch_changes": 0, "disk_authority_changes": 0, "ranks_seen": 0,
    # scheduler-side request ledger. `lookups` (below) is the population; the
    # request-scoped `requests_seen` it used to sit beside is retired with the
    # marker - it existed as the denominator of `flagged` + `skipped_no_marker`, and
    # `lookups` is incremented at the same statement.
    "suppressed_hits": 0, "suppressed_spans": 0, "suppressions": 0,
    # WHICH FAILURE KIND DECIDED THE FUSE (0.2.32). `suppress_load` used to be called
    # by every failure report whether the read had LOST A RACE or the bytes were BAD -
    # and the reason string it was handed already named the kinds (`sorted
    # (_recent_failure_kinds())`), so the code knew and decided without looking. These
    # three say which side of `failure_kind_classes()` each decision fell on:
    # `races_not_fused` is a timing loss that cost the request its tokens and NOT its
    # next offer; `fuses_on_data_kinds` is the fuse kept for corruption, short reads and
    # DMA faults; `fuses_on_unknown_kinds` is the fail-closed branch (no kind visible -
    # usually a worker whose rank file has not been read yet), which fuses exactly as a
    # data failure does because "nobody saw it" is not proof of a race.
    "races_not_fused": 0, "fuses_on_data_kinds": 0, "fuses_on_unknown_kinds": 0,
    "race_reoffers_granted": 0,       # second offers allowed by the one-per-request
                                      # race allowance (the loop bound that replaced the
                                      # boot-length fuse for a timing loss)
    "last_fuse_decision": None,       # the last such decision, as a sentence
    # THE AUTHORITY RULE, IN USE. Under the switch every request hands the DISK the
    # decision about its own prefix: the fork's host-RAM hit is treated as zero tokens
    # for the offer arithmetic. `authority_lookups` counts the lookups that took that
    # rule (the switch itself is `disk_authority`, above); the other two count the ones
    # where a real RAM hit was thrown away and how many tokens it was, so "was this
    # read forced from disk, or did RAM have nothing?" is one payload read and not an
    # inference.
    "authority_lookups": 0, "local_hit_ignored": 0,
    "local_hit_tokens_ignored": 0,
    "pin_release_failures": 0,
    "last_local_hit_ignored": None,
    # THE LAST OFFER, AS STRUCTURE: the numbers a forced read and an ordinary one
    # disagree about, side by side. One dict, overwritten, clipped; never a ledger.
    "last_offer": None,
    # WHAT A CLAIM COSTS EVERY GROUP, published for the LAST priced lookup, offered or
    # refused (0.2.25). `last_offer` can only describe an offer that happened; the
    # 00:08 incident was an offer that should not have happened, so the requirement
    # arithmetic is published on BOTH doors. Rows are per prefix-cacheable group: what
    # the claimed range needs from it (`blocks_needed`/`null_blocks`), the contract that
    # arithmetic came from, and whether the disk fills it. See `_groups_required`.
    "last_groups_required": None,
    # WHICH GROUPS THIS MODEL CAN EVER SERVE FROM DISK (0.2.25). Recomputed on every
    # lookup that reaches the group arithmetic, whether it offered or refused, so the
    # answer to "is this model un-serviceable from disk?" is one payload read and not
    # a re-derivation of three manager predicates by hand. See `_group_serviceability`.
    "group_serviceability": None,
    # THE LAST LOAD THAT LEFT THE LEDGER WITHOUT A REPORT, in one sentence: which sweep
    # door released it, whether its slots had reached a worker, whether a disk read had
    # been applied, and how many GPU blocks it had mapped. With `last_refusal` and
    # `last_offer` this is the third thing an incident actually asks for.
    "last_aged_out": None,
    # scheduler-side ledger
    "lookups": 0, "offers": 0, "offered_tokens": 0, "blocks_offered": 0,
    "adopted": 0, "adopted_tokens": 0, "restorations": 0,
    # THE LEDGER'S OWN HONESTY (0.2.25). A load can leave the ledger three ways
    # without ever being correlated with a completion report: the fork stopped owning
    # it (`_reqs_to_load` no longer names it - the shape of the fork's own
    # "no external tokens but a pending CPU hit: releasing the stale pin" line), or
    # its deadline expired. Either way a `finished_recving` that arrives afterwards
    # used to find `_loads.get(request_id) is None` and `continue`: silently. On the
    # live box that cost the 00:08 investigation its only evidence - 36 reads applied,
    # `loads_released=1`, `restorations=0`, and nothing in the payload saying the
    # report had arrived too late rather than never. `restorations` keeps its exact
    # meaning ("reported while the ledger held it"); these count the OTHER case.
    "aged_out_loads": 0,             # ledger releases with no report correlated
    "aged_out_with_applied_reads": 0,  # ...of those, ones a read WAS applied for:
                                       # bytes landed, the report came too late
    "late_finished_reports": 0,      # a `finished_recving` whose ledger entry was
                                     # already gone, matched to its release
    "restored_after_release": 0,     # ...of those, ones whose release showed an
                                     # adopted load with bytes applied behind it
    # THE ONLY FIELD THAT COUPLES OUR CLAIM TO THE FORK'S DMA (0.2.25). "We offered
    # these GPU blocks" and "the fork finished these GPU blocks" are two facts that
    # meet in exactly one place: a `finished_recving` id matching a ledger record that
    # names the GPU blocks our emitted paths wrote into. Counted in GPU BLOCKS, not
    # requests, so a 9-block offer that finishes adds 9.
    "disk_gpu_blocks_dma_verified": 0,
    "disk_gpu_blocks_released_unreported": 0,  # ...mapped for a disk span, released
                                               # by the ledger, never reported
    "refused": {r: 0 for r in OFFER_REFUSALS},
    "refusals_total": 0, "last_refusal": None, "last_reason": None,
    "paths_emitted_steps": 0, "path_entries": 0, "path_entries_empty": 0,
    "path_length_violations": 0, "slots_allocated": 0, "stale_slots_dropped": 0,
    "watchdog_fires": 0, "index_refreshes": 0, "index_refresh_errors": 0,
    "loads_released": 0, "ranks_proven": 0, "proof_reads": 0,
    # scheduler-side honesty: a completion is a RESTORATION only when this process
    # or a proven rank can show an applied disk read for it; a completion with no
    # read behind it is `no_read` and is never counted as a restoration.
    "no_read": 0, "disk_reads_applied": 0, "ranks_reads_applied": 0,
    # `offer_clamped` (0.2.31 and earlier) is DELETED with the fixed 32,000-token cap:
    # with no configured cap there is no clamp to count. See the note at
    # MAX_RESTORE_BLOCKS_PER_OFFER.
    # worker-side ledger
    "binds": 0, "disk_binds": 0, "reads_started": 0, "reads_applied": 0,
    "bytes_applied": 0, "dma_unsettled_entries": 0,
    # THE RACE, AS A MEASUREMENT AND NOT AN INFERENCE. `dma_unsettled_entries` says
    # how many entries lost the race; these say HOW FAR they lost it by, which is the
    # only thing that decides whether waiting can fix it at all. The distinction
    # between "the job never started" and "the job was mid-flight" is the difference
    # between a thread that never got the CPU and a disk that was too slow, and until
    # 0.2.23 both were counted as `not_settled`.
    "apply_observations": 0,          # batches seen at the apply point
    "apply_settled_on_arrival": 0,    # ...of those, already settled (the race, won)
    "applied_settled_immediately": 0,  # ENTRIES applied without any wait
    "dropped_queued": 0,              # ENTRIES dropped whose read had not started
    "dropped_running": 0,             # ENTRIES dropped whose read was mid-flight
    "dropped_unattributed": 0,        # ...dropped unsettled, state unreadable
    "bind_to_apply_total_ms": 0.0,    # sums for the averages; min/median/max are
    "bind_to_apply_min_ms": None,     # recomputed from the bounded samples plus a
    "bind_to_apply_max_ms": None,     # boot-wide running min/max
    "apply_elapsed_total_ms": 0.0,
    "apply_elapsed_min_ms": None,
    "apply_elapsed_max_ms": None,
    # THE SETTLE WAIT (0.2.23), all of it inert while the cap is 0.
    "settle_waits_granted": 0,        # batches given a non-zero budget
    "settle_ms_total": 0.0,           # ms actually waited (sum -> avg)
    "settle_ms_max": None,            # the worst single wait this boot
    "settle_budget_total_ms": 0.0,
    "settle_budget_max_ms": None,
    "settled_after_wait": 0,          # batches that settled during their wait
    "dropped_after_wait": 0,          # ...still unsettled when the wait expired
    "settle_bytes_waited": 0,         # bytes the waits were waiting for
    "settle_bound_flat": 0,           # which bound decided the allowance, per the
    "settle_bound_deadline": 0,       #   three-way rule in `settle_budget`
    "settle_bound_off": 0,            # (`off` = the allowance is 0: never wait)
    "settle_allowance_spent": 0,      # batches that spent the WHOLE granted allowance
                                      # and still had bytes in flight: the honest wait
                                      # metric, replacing `settle_budget_over_recompute`
    "last_apply_window": None,        # the most recent batch's full window
    "settle_notice": None,            # the once-per-boot line, as data
    # THE SHAPE OF A PARTIAL RESTORE (0.2.24), measured and not decided. A span needs
    # EVERY group to be worth anything; with one reader an interrupted batch was
    # always missing a suffix of spans, and with N readers it can be missing one in
    # the middle. Whether a partial restore may have its tokens counted as computed is
    # NOT decided here - these are the numbers that question gets asked with.
    "spans_observed": 0,              # spans seen at the apply point
    "span_entries_observed": 0,       # disk entries those spans were made of
    "spans_whole": 0,                 # ...every group applied
    "spans_partial": 0,               # ...some groups applied (the new shape)
    "spans_empty": 0,                 # ...nothing applied
    "spans_hole_in_the_middle": 0,    # batches whose applied spans are NOT a prefix:
                                      # impossible with one reader, so this counter
                                      # rising IS the concurrency finding
    "last_span_shape": None,          # the most recent batch's shape, as data
    # THE OFFER'S PRICE IN BOUNCE BUFFERS (0.2.31). `offer_pool_capped` counts offers
    # SHORTENED to what the pool can carry and `offer_pool_refused` the lookups given
    # no offer at all because not one job fits. `last_offer_pool` is the arithmetic of
    # the most recent pricing, so "did this offer come in short, and against which
    # bound?" is a payload read and not an inference from a job count.
    "offer_pool_capped": 0, "offer_pool_refused": 0,
    "last_offer_pool": None,
    # WHICH OFFER BOUND EXISTS IN THIS BOOT (0.2.32). `offer_block_capped` counts the
    # offers the per-offer BLOCK cap stopped - the positional walk hitting
    # `MAX_RESTORE_BLOCKS_PER_OFFER` while the request still had tokens left - the
    # successor to the deleted `offer_clamped`, and the ONLY per-offer cap counter left
    # now that the fixed 32,000-token clamp is gone. `offer_block_size_seen` is the
    # engine's real scheduler block size, remembered so the payload and the boot line
    # can state the block bound in tokens in a process that has made no offer yet.
    "offer_block_capped": 0,
    "offer_block_size_seen": None,
    # THE LOAD-FAILURE RING (0.2.31): what the worker said when a read did not happen,
    # in the process that can READ it. `load_failures` counts records written and
    # `load_failures_overwritten` the ones the ring recycled; `ranks_load_failures`
    # is how many arrived from live rank files, the `ranks_reads_applied` idiom.
    "load_failures": 0, "load_failures_overwritten": 0,
    "ranks_load_failures": 0, "failure_log_lines": 0, "failure_log_suppressed": 0,
    "ranks_bytes_per_file_max": None,
    # The records the LIVE rank files carried the last time the proof read them -
    # the transport, refreshed with the proof's own 1 s cache, not a second cache.
    "rank_failure_records": [],
    # THE BLOCK IDS THE FORK REPORTED, kept. `note_reported` used to take
    # `len(ids)` and drop the ids, which is why a hunt for WHICH of our blocks were
    # the load errors had no answer in any payload.
    "invalid_block_ids_seen": 0, "last_invalid_block_ids": None,
    "reads_failed": {r: 0 for r in WORKER_FAILURES},
    "invalid_reports": 0, "finished_released": 0, "worker_watchdog_fires": 0,
    "phantom_guard_trips": 0, "region_plan_errors": 0, "worker_errors": 0,
    "load_error_installs": 0,
    "witness_verdict": None, "witness_reason": None, "witness_kind": None,
    "witness_emissions": 0,
    "witness_suppressed": 0,
}
"""`reads_failed` is the WORKER's per-entry ledger; the scheduler-side `no_read`
and `disk_reads_applied` are the per-OFFER verdicts the witness is allowed to use:
an offer is a restoration only if a read was applied for it."""

# Scheduler side: request id -> load record. Bounded by MAX_TRACKED_LOADS.
_loads: dict = {}
# Scheduler side: cpu block id -> (request id, path). Bounded by MAX_TRACKED_SLOTS.
_slot_paths: dict = {}
# Scheduler side: request id -> why it will get no further disk offer, for the life
# of the request. A load that failed and is re-offered is an infinite loop (see
# `suppress_load`), so the answer has to be remembered, not re-derived.
_suppressed: dict = {}
# Scheduler side: request id -> why its load BURNED (a failure report), kept after the
# ledger entry is gone so a later `finished_recving` for the same request can be
# counted as `no_read` instead of vanishing. Same bound as the fuse: a request that
# burns and then never reports is one short string.
_burned: dict = {}
# Scheduler side: the same for a disk SPAN (first block's key:group of the offered
# run), because a request the engine reset comes back with the same hashes and a
# span that failed once will fail for whoever asks for it next.
_suppressed_spans: dict = {}
# Scheduler side: request id -> how its load LEFT THE LEDGER without a report (which
# sweep branch, whether the slots had reached a worker, whether a disk read had been
# applied, and how many GPU blocks it had mapped). The `_burned` idiom applied to the
# releases that are not failures: a `finished_recving` that arrives after a release is
# not nothing, and without this tombstone it is `continue`d away. Same bound as the
# fuse: one short record per request that aged out.
_aged_out: dict = {}
# Scheduler side: request id -> computed tokens at the moment we last offered it disk
# tokens. A request that returns at or before that point made no progress, which is
# what a failed load looks like from the scheduler's seat.
_offered_at: dict = {}
# Scheduler side: request id -> how many re-offers it may still claim after a load it
# LOST A RACE rather than corrupted (`_grant_race_reoffer`). 0 means the allowance is
# spent; a data failure never puts a request in here at all, it fuses it.
_race_reoffer: dict = {}

# Worker side: request id -> in-flight record (the watchdog's table), and the
# batches those records share. Both bounded; `_invalid` is drained once per step.
_wloads: dict = {}
_wbatches: dict = {}
_invalid: dict = {}
_logged_lines: list = []
_witness_state: dict = {"line": None, "emitted_mono": None, "key": None}

# The window's samples. A median needs the population, min/max do not, so min/max
# are boot-wide running values in `_state` while the median is computed over the last
# `MAX_SETTLE_SAMPLES` observations and published AS that (with the count it covers,
# so a median over 512 of 40,000 observations can never be misread as the boot's).
_apply_samples: dict = {"bind_to_apply_ms": [], "apply_elapsed_ms": [],
                        "settle_ms": [], "settle_budget_ms": []}
# Once-per-boot notices, the `miss_attribution.notice_once` idiom: a wait that is on
# has to be said out loud once, and saying it at the apply point every step would be
# the flood that already cost this module its log.
_notices: set = set()


def notice_once(key: str, message: str) -> bool:
    """Log `message` at most once per key per process. Bounded, never raises.

    The repo's own idiom (`disk_index.notice_once`, `miss_attribution.notice_once`):
    a line on the engine path is printed once per episode, because the alternative -
    one line per step - is how a diagnostic stops being read. Returns True when it
    actually logged, so the caller can count the episode and not the call.
    """
    try:
        if key in _notices:
            return False
        if len(_notices) >= 64:
            _notices.clear()
        _notices.add(key)
        _logger.warning("kvtier-dev-plugin: %s", str(message)[:640])
        return True
    except Exception as exc:  # noqa: BLE001 - a log line is never worth a fault
        snapshot._error(exc, "disk_restore notice")
        return False


# Worker/scheduler side: the LOAD-FAILURE RING (0.2.31). One record per refused or
# failed load read, newest first, recycled at `MAX_LOAD_FAILURE_RECORDS`. It exists
# because the reasons already existed and went nowhere: `WORKER_FAILURES` names 26
# kinds, every one of them counted in the process that READ the file, and the only
# channel back to the scheduler is a set of GPU block ids - so `reads_failed` was
# structurally zero on EngineCore and a hunt for "why did this read not happen" had
# to be settled by reading four rank files by hand.
_load_failures: list = []
# Kind -> monotonic of the last line logged for it: the rate limit that lets a line
# be printed per EPISODE and never per step.
_failure_log_at: dict = {}

# The pointer a record uses instead of pasting the provenance essay into every row,
# the `miss_attribution.attribute` idiom (`not_established` points at `section()`).
PROVENANCE_POINTER = "see the disk verb's failures op: provenance_meaning"


def _failure_text(value, chars: int = MAX_FAILURE_STRING_CHARS):
    """A clipped string for one field of a failure record, or None."""
    if value is None:
        return None
    return str(value)[:chars]


def _failure_int(value):
    """A plain int for one field of a failure record, or None. Bool is not an int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - an unreadable number is no number
        return None
    return int(value) if text.lstrip("-").isdigit() else None


def _failure_time(value):
    """A timestamp for a record: int or float, never a bool, never a guess.

    `_failure_int` is deliberately integer-only - it is what a block id or a byte
    count uses, and a float byte count means the writer is confused. Journal stamps are
    `time.time()` floats, so provenance needs a second, wider reader: reading them with
    the int-only one produced `learned_at: null` for an entry the index holds, which is
    the field that tells an operator whether these bytes are from this boot or from a
    spill days old.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _failure_field(record: dict, name: str) -> str:
    """One field of a record as log text, with None spelled as `?`."""
    value = record.get(name)
    return "?" if value is None else str(value)


def _identity_pair(raw, group) -> tuple:
    """(key16, group_id) from EITHER a full group-qualified hash or a bare key16.

    WHY THE RECORD NORMALIZES RATHER THAN PASSING THROUGH. The two callers hand in
    different spellings of one identity: the offer path reads `run["keys"]`, whose
    entries carry the engine's FULL group-qualified block hash, while a worker's
    record comes from a spilled PATH, which `disk_layout` splits into a 16-hex
    `key16` plus a group. A payload field called `key16` must hold a key16, and the
    index's own identity rule (`disk_index.identity_of`) is built from one - feed it
    a 64-character hash and every lookup misses, so provenance would answer "never
    seen" about a file this index journalled itself. Reusing `split_block_key` is the
    point: the journal's writer and this reader then cannot drift into two spellings.

    An unreadable string comes back as its own clipped text with the group untouched:
    a record about an identity this module cannot parse is still a record, and the
    provenance lookup refuses it on its own terms.
    """
    text = _failure_text(raw, 128)
    group_id = _failure_int(group)
    if not text:
        return None, group_id
    split = disk_layout.split_block_key(text)
    if split.get("ok"):
        return _failure_text(split.get("key16"), 64), (
            group_id if group_id is not None else _failure_int(
                split.get("group_id")))
    return _failure_text(text, 64), group_id


def _block_provenance(key16, group) -> dict | None:
    """What THIS process's index can say about the bytes behind one block.

    WHAT IT CANNOT SAY, AND WHY THE FIELD IS NOT NAMED `attribution`. The journal
    line is `<seq> <+|-> <key16> <bytes> <group>` (`disk_index.JOURNAL_FIELDS`, five
    fields) and `record_spill` is called from the writer thread with a PATH and a
    SIZE: no request id reaches the journal at all, on either side of the spill, so
    "which request wrote this block" is not knowable from anything this plugin holds.
    The lookup ledger's `attribution` object is about the CURRENT lookup (num_tokens,
    the local hit, the tokens offered, the prefix-walk verdict) and never named a
    writer either. Inventing a writer field here would be a guess wearing a name.

    What IS knowable, and is what the operator needs to tell "restored from bytes I
    spilled an hour ago" from "restored from bytes this prompt just wrote":
      * key16 + group, which IS the content identity - the same prompt and settings
        under a restart produce the SAME key, so reuse is visible in the key itself;
      * which rank's journal claims the identity, and its position (seq) in that
        journal, from `disk_index.rank_claims`, and the size/learn stamp of the entry
        those claims resolved to from `disk_index.entry_for` (the accessor that takes
        a `key16` + group, which is what a record carries; `lookup()` needs the
        engine's full hash and a record does not have it);
      * `learned_at`, the time THIS index applied the spill line. That is a LEARN
        time, not a write time - a cold boot's drain stamps files written days ago
        with the drain's clock - and the record says so rather than implying more.

    Never raises; None when the identity cannot be read or was never given.
    """
    try:
        key = _failure_text(key16, 64)
        group_id = _failure_int(group)
        if not key or group_id is None:
            return None
        identity = disk_index.identity_of(key, group_id)
        claims = disk_index.rank_claims(identity)
        entry = disk_index.entry_for(key, group_id)
        rows = []
        newest_seq = None
        newest_rank = None
        newest_bytes = None
        for name in sorted(claims)[:4]:
            row = claims[name] or {}
            seq = _failure_int(row.get("seq"))
            rows.append({"rank": _failure_text(name, 32),
                         "bytes": _failure_int(row.get("bytes")),
                         "journal_seq": seq})
            if seq is not None and (newest_seq is None or seq > newest_seq):
                # The LAST claim wins, exactly as it does when the index replays a
                # journal: a rank that spilled the same identity twice is the author
                # of the bytes the entry stands for now.
                newest_seq, newest_rank = seq, name
                newest_bytes = _failure_int(row.get("bytes"))
        seen = None
        for row in claims.values():
            stamp = _failure_time((row or {}).get("seen_ts"))
            if stamp is not None and (seen is None or stamp > seen):
                seen = stamp
        known = bool(entry.get("present"))
        size = _failure_int(entry.get("bytes"))
        learned = _failure_time(entry.get("seen_ts"))
        if learned is None:
            learned = seen
        return {"key16": key, "group": group_id,
                "known_to_this_index": known,
                "bytes": size if size is not None else newest_bytes,
                "writer_rank": _failure_text(
                    newest_rank if newest_rank is not None
                    else (sorted(claims)[0] if claims else None), 32),
                "writer_journal_seq": newest_seq,
                "learned_at": learned,
                "learned_age_seconds": (round(time.time() - learned, 3)
                                        if learned is not None else None),
                "ranks": rows,
                "learned_at_is_learn_time_not_write_time": True,
                "meaning": PROVENANCE_POINTER}
    except Exception as exc:  # noqa: BLE001 - provenance is never worth a fault
        snapshot._error(exc, "disk_restore failure provenance")
        return None


def _log_failure_episode(kind, line: str) -> bool:
    """One load-failure line per kind per `REFUSAL_LOG_MIN_SECONDS`. Never per step.

    THE LINE THE OPERATOR ASKED FOR. Nothing in any log explained why a read did not
    happen: `witness_line` speaks once per verdict change and `suppress_load` speaks
    about the FUSE, not the cause. This is the cause, at the moment it happens, with
    the byte figures in it - rate-limited because a refusal that repeats every step
    is the 1,048-lines-a-second flood that already cost this module its log.
    """
    try:
        key = str(kind or "?")[:48]
        now = time.monotonic()
        fired = _failure_log_at.get(key)
        if isinstance(fired, (int, float)) \
                and (now - float(fired)) < REFUSAL_LOG_MIN_SECONDS:
            _state["failure_log_suppressed"] += 1
            return False
        if len(_failure_log_at) >= 32:
            _failure_log_at.clear()
        _failure_log_at[key] = now
        _state["failure_log_lines"] += 1
        _logger.warning("kvtier-dev-plugin: DISK[load-fail] %s", line[:640])
        return True
    except Exception as exc:  # noqa: BLE001 - a log line is never worth a fault
        snapshot._error(exc, "disk_restore failure log")
        return False


def _note_load_failure(kind, *, where: str, reason: str = "", request_id=None,
                       span=None, group=None, contract=None, key16=None,
                       gpu_block_id=None, cpu_block_id=None, job_id=None,
                       bytes_needed=None, jobs_needed=None, jobs_fit=None,
                       pool_budget_bytes=None, pool_committed_bytes=None,
                       provenance: bool = True) -> dict:
    """Record one refused or failed load read, and say it out loud (rate-limited).

    `provenance` is the caller's opt-out for a record whose identity is not a block
    identity at all (a whole-step refusal, a dropped queue entry): looking one up
    would cost an index read to produce a `known_to_this_index: false`.

    THE VOCABULARY IS THE EXISTING ONE: `kind` is a name from `WORKER_FAILURES`, so a
    record and the `reads_failed` counter can never disagree about what happened - a
    fourth spelling of "short read" is how a counter stops being evidence.

    For `buffer_cap` specifically the record carries the arithmetic that decides it:
    `pool_budget_bytes`, `pool_committed_bytes`, `bytes_needed` (the price of ONE
    job), `jobs_needed` (what the offer asked for) and `jobs_fit`. Those five answer
    "is the pool too small for an offer this big" (budget vs needed) and "is some-
    thing holding buffers it is not returning" (committed) without a second boot.

    Bounded, never raises, and cheap: one dict, one append, one monotonic read.
    """
    record: dict = {}
    try:
        name = str(kind or "read_failed")[:48]
        _state["load_failures"] += 1
        key_field, group_field = _identity_pair(key16, group)
        record = {"seq": int(_state["load_failures"]),
                  "ts": round(time.time(), 6),
                  "kind": name if name in WORKER_FAILURES else "read_failed",
                  "kind_as_reported": name,
                  "where": _failure_text(where, 24),
                  "request_id": _failure_text(request_id, 48),
                  "span": _failure_text(span, 80),
                  "group": group_field,
                  "contract": _failure_text(contract, 24),
                  "key16": key_field,
                  "gpu_block_id": _failure_int(gpu_block_id),
                  "cpu_block_id": _failure_int(cpu_block_id),
                  "job_id": _failure_int(job_id),
                  "bytes_needed": _failure_int(bytes_needed),
                  "jobs_needed": _failure_int(jobs_needed),
                  "jobs_fit": _failure_int(jobs_fit),
                  "pool_budget_bytes": _failure_int(pool_budget_bytes),
                  "pool_committed_bytes": _failure_int(pool_committed_bytes),
                  "reason": _failure_text(reason, 200),
                  "rank": None,
                  "block_provenance": (_block_provenance(key_field, group_field)
                                       if provenance and key_field else None)}
        _load_failures.insert(0, record)
        while len(_load_failures) > MAX_LOAD_FAILURE_RECORDS:
            _load_failures.pop()
            _state["load_failures_overwritten"] += 1
        parts = [f"kind={record['kind']}", f"where={record['where']}",
                 f"req={_failure_field(record, 'request_id')}",
                 f"group={_failure_field(record, 'group')}",
                 f"key16={_failure_field(record, 'key16')}",
                 f"gpu_block={_failure_field(record, 'gpu_block_id')}"]
        if record["bytes_needed"] is not None:
            parts.append(f"bytes_needed={record['bytes_needed']}")
        if record["jobs_needed"] is not None:
            parts.append(f"jobs_needed={record['jobs_needed']} "
                         f"jobs_fit={record['jobs_fit']}")
        if record["pool_budget_bytes"] is not None:
            parts.append(f"pool_budget={record['pool_budget_bytes']} "
                         f"pool_committed={record['pool_committed_bytes']}")
        prov = record.get("block_provenance") or {}
        if prov:
            parts.append(f"block_rank={prov.get('writer_rank') or '?'} "
                         f"block_seq={prov.get('writer_journal_seq')} "
                         f"learned_age_s={prov.get('learned_age_seconds')}")
        _log_failure_episode(record["kind"], " ".join(parts) + " "
                             + _failure_text(reason, 160))
        return record
    except Exception as exc:  # noqa: BLE001 - a record never breaks a step
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore load failure record")
        return record


def _last_invalid_blocks(limit: int = 8) -> list:
    """The GPU block ids of the most recent load-error report, bounded for a line."""
    try:
        return list(((_state.get("last_invalid_block_ids") or {}).get("blocks"))
                    or [])[:int(limit)]
    except Exception:  # noqa: BLE001 - a sentence is never worth a fault
        return []


def _recent_failure_kinds(limit: int = 6) -> set:
    """The failure kinds seen most recently, here or in a live rank file.

    One line of an operator's log should say WHICH kind the worker blamed, even when
    the line itself was written by the scheduler - and the scheduler's own
    `reads_failed` is zero by construction, so the ring and the rank files are the
    only places that know.
    """
    kinds: set = set()
    try:
        for item in list(_load_failures)[:int(limit)]:
            if item.get("kind"):
                kinds.add(str(item["kind"]))
        for item in list(_state.get("rank_failure_records") or [])[:int(limit)]:
            if item.get("kind"):
                kinds.add(str(item["kind"]))
    except Exception as exc:  # noqa: BLE001 - a sentence is never worth a fault
        snapshot._error(exc, "disk_restore recent failure kinds")
    kinds.discard(None)
    return kinds


def load_failures(limit=None) -> list:
    """The ring's records, newest first. Bounded by `MAX_LOAD_FAILURE_RECORDS`.

    THIS PROCESS ONLY. A worker writes the refusals it sees and the scheduler cannot,
    so a reader that wants the fleet's records wants `load_failure_report()`, which
    folds in what the live rank files carry (the `ranks_reads_applied` transport).
    """
    try:
        wanted = _int(limit) if limit is not None else MAX_FAILURES_PUBLISHED
        if wanted is None or wanted < 1:
            wanted = MAX_FAILURES_PUBLISHED
        wanted = min(int(wanted), MAX_LOAD_FAILURE_RECORDS)
        return [dict(item) for item in _load_failures[:wanted]]
    except Exception as exc:  # noqa: BLE001 - a payload field never breaks a poll
        snapshot._error(exc, "disk_restore load_failures")
        return []


def load_failure_report(limit=None) -> dict:
    """The load-failure records an operator can read: this process, plus every rank.

    Cross-process the way this module already does it - the rank file. A worker's
    `worker_report()` publishes its last `MAX_FAILURES_PER_RANK_FILE` records inside
    its (byte-capped) `disk_restore` section and `_read_worker_proof()` reads them
    with the same bounded, defensive, fresh-from-disk walk that carries
    `reads_applied` today; this merges the two sources newest-first.

    Provenance is attached HERE for records that came from a worker, because the
    worker's own `disk_index` is empty by construction (the index is the scheduler's)
    and the SCHEDULER is the process holding the index when this runs. The identity
    in the record is a content hash, so the lookup is the same identity the worker
    was reading bytes for.
    """
    wanted = _int(limit)
    if wanted is None or wanted < 1:
        wanted = MAX_FAILURES_PUBLISHED
    wanted = min(int(wanted), MAX_LOAD_FAILURE_RECORDS * (MAX_RANK_FILES_PROBED + 1))
    records = [dict(item) for item in _load_failures[:wanted]]
    ranks: list = []
    for item in _state.get("rank_failure_records") or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if row.get("block_provenance") is None and row.get("key16"):
            row["block_provenance"] = _block_provenance(row.get("key16"),
                                                        row.get("group"))
        ranks.append(row)
    merged = sorted(records + ranks,
                    key=lambda item: (item.get("ts") or 0.0,
                                       item.get("seq") or 0), reverse=True)
    return {"ok": True,
            "records": merged[:wanted],
            "records_here": len(records),
            "records_from_rank_files": int(_state["ranks_load_failures"] or 0),
            "ring_depth": MAX_LOAD_FAILURE_RECORDS,
            "ring_written": int(_state["load_failures"]),
            "ring_overwritten": int(_state["load_failures_overwritten"]),
            "newest_first": True,
            "kinds": sorted(WORKER_FAILURES),
            "provenance_meaning": (
                "key16+group IS the content identity: the same prompt and the same "
                "settings under a restart produce the SAME key, so a restore can be "
                "served by bytes an older boot (or another session, or another "
                "request) spilled. Which REQUEST wrote a block is NOT knowable from "
                "the store: a journal line is `<seq> <+|-> <key16> <bytes> <group>` "
                "and `record_spill` is handed a path and a size, so no request id "
                "ever reaches the journal. The closest real provenance is what these "
                "records carry - the rank whose journal claims the identity, that "
                "line's position in the rank's journal, and when THIS process's index "
                "learned it (a learn time, not a write time: a cold boot's drain "
                "stamps old files with the drain's clock)."),
            "note": ("a worker writes the records (it is the only process that reads "
                     "a file) and this report merges what its rank file carried. "
                     "`reads_failed` on EngineCore is zero by construction: that is "
                     "why the records, not the counter, are the evidence.")}


def reset_state_for_tests() -> None:
    """Back to a fresh boot: no ledgers, no counters, no installed layer."""
    global _loads, _slot_paths, _wloads, _wbatches, _invalid, _logged_lines
    global _install_refusal_lines, _sched_cls_seen, _suppressed, _suppressed_spans
    global _offered_at, _burned, _aged_out, _load_failures, _failure_log_at
    global _race_reoffer
    _loads = {}
    _slot_paths = {}
    _wloads = {}
    _wbatches = {}
    _invalid = {}
    _suppressed = {}
    _suppressed_spans = {}
    _burned = {}
    _aged_out = {}
    _offered_at = {}
    _race_reoffer = {}
    _logged_lines = []
    _install_refusal_lines = []
    _sched_cls_seen = None
    del _load_failures[:]
    _failure_log_at.clear()
    for samples in _apply_samples.values():
        del samples[:]
    _notices.clear()
    _witness_state.update({"line": None, "emitted_mono": None, "key": None})
    _proof_cache.update({"mono": None, "result": None})
    _deferred_finished.clear()
    for key, value in {
            "installed": False, "class": None, "installs": 0, "reinstalls": 0,
            "uninstalls": 0, "skip_reason": None, "nested": [], "errors": 0,
            "fallbacks": 0, "boot_line": None, "sched_fitted": False,
            "worker_fitted": False, "methods_fitted": [], "install_refusals": 0,
            "install_deferrals": 0, "missing_methods": [],
            "install_refusal_line": None, "install_deferral_line": None,
            "lookups": 0, "offers": 0, "offered_tokens": 0,
            "blocks_offered": 0, "adopted": 0, "adopted_tokens": 0,
            "restorations": 0, "refusals_total": 0, "last_refusal": None,
            "last_reason": None, "paths_emitted_steps": 0, "path_entries": 0,
            "path_entries_empty": 0, "path_length_violations": 0,
            "slots_allocated": 0, "stale_slots_dropped": 0, "watchdog_fires": 0,
            "index_refreshes": 0, "index_refresh_errors": 0, "loads_released": 0,
            "ranks_proven": 0, "proof_reads": 0, "no_read": 0,
            "disk_reads_applied": 0, "ranks_reads_applied": 0,
            "binds": 0, "disk_binds": 0, "reads_started": 0, "reads_applied": 0,
            "bytes_applied": 0, "dma_unsettled_entries": 0, "invalid_reports": 0,
            "finished_released": 0, "worker_watchdog_fires": 0,
            "phantom_guard_trips": 0, "region_plan_errors": 0,
            "worker_errors": 0, "load_error_installs": 0,
            "apply_observations": 0, "apply_settled_on_arrival": 0,
            "applied_settled_immediately": 0, "dropped_queued": 0,
            "dropped_running": 0, "dropped_unattributed": 0,
            "bind_to_apply_total_ms": 0.0, "bind_to_apply_min_ms": None,
            "bind_to_apply_max_ms": None, "apply_elapsed_total_ms": 0.0,
            "apply_elapsed_min_ms": None, "apply_elapsed_max_ms": None,
            "settle_waits_granted": 0, "settle_ms_total": 0.0,
            "settle_ms_max": None, "settle_budget_total_ms": 0.0,
            "settle_budget_max_ms": None, "settled_after_wait": 0,
            "dropped_after_wait": 0, "settle_bytes_waited": 0,
            "settle_bound_flat": 0, "settle_bound_deadline": 0,
            "settle_bound_off": 0, "settle_allowance_spent": 0,
            "last_apply_window": None,
            "settle_notice": None,
            "spans_observed": 0, "span_entries_observed": 0, "spans_whole": 0,
            "spans_partial": 0, "spans_empty": 0, "spans_hole_in_the_middle": 0,
            "last_span_shape": None,
            "offer_pool_capped": 0, "offer_pool_refused": 0,
            "last_offer_pool": None,
            "offer_block_capped": 0, "offer_block_size_seen": None,
            "load_failures": 0, "load_failures_overwritten": 0,
            "ranks_load_failures": 0, "failure_log_lines": 0,
            "failure_log_suppressed": 0,
            "ranks_bytes_per_file_max": None,
            "rank_failure_records": [],
            "invalid_block_ids_seen": 0, "last_invalid_block_ids": None,
            "witness_verdict": None, "witness_reason": None,
            "witness_kind": None, "witness_emissions": 0,
            "witness_suppressed": 0,
            "offers_enabled": False, "disk_authority": DISK_AUTHORITY_DEFAULT,
            "offer_switch_changes": 0, "disk_authority_changes": 0,
            "ranks_seen": 0,
            "suppressed_hits": 0, "suppressed_spans": 0, "suppressions": 0,
            "races_not_fused": 0, "fuses_on_data_kinds": 0,
            "fuses_on_unknown_kinds": 0, "race_reoffers_granted": 0,
            "last_fuse_decision": None,
            "authority_lookups": 0, "local_hit_ignored": 0,
            "local_hit_tokens_ignored": 0, "pin_release_failures": 0,
            "last_local_hit_ignored": None, "last_offer": None,
            "last_groups_required": None,
            "group_serviceability": None, "last_aged_out": None,
            "aged_out_loads": 0,
            "aged_out_with_applied_reads": 0, "late_finished_reports": 0,
            "restored_after_release": 0, "disk_gpu_blocks_dma_verified": 0,
            "disk_gpu_blocks_released_unreported": 0}.items():
        _state[key] = value
    _state["refused"] = {name: 0 for name in OFFER_REFUSALS}
    _state["reads_failed"] = {name: 0 for name in WORKER_FAILURES}


def settle_budget(bytes_at_stake: int = 0,
                  seconds_left: float | None = None) -> dict:
    """How long ONE apply point may wait for the bytes it is about to apply.

    THE RULE, IN ONE LINE (0.2.32):

        granted_ms = min(SETTLE_MS, the load's own remaining deadline)

    `SETTLE_MS` (`SETTLE_MS_DEFAULT`, 5,000 ms, overridable by
    `VLLM_E2_TIER_DISK_SETTLE_MS`) is the FLAT allowance, and the deadline is the one
    `disk_read.begin_batch` set from `LOAD_DEADLINE_SECONDS` - the same clock that
    cancels the read, so a wait can never outlive the bytes it is waiting for. Nothing
    else decides. `0` still means never wait and is still reachable.

    WHY THERE IS NO FORMULA ANY MORE. Through 0.2.31 the allowance was computed from
    the bytes still in flight over an ASSUMED read throughput, times a safety margin,
    floored at a scheduler tick and capped by the env var - and every term was a guess.
    The live box settled the argument on 2026-10-06: 12 jobs / 137.6 MB were offered,
    the formula granted 23.6 ms, the measured bind-to-apply race gap for the lost
    batches was 89.7 / 97.1 / 129.7 / 135.8 ms, and the request recomputed 28,583 tokens
    (12.26 s, `cached_tokens: 0`) to save 23.6 ms. The operator's ruling is the rule
    above: "give it up to 5 s to read; only when something bad happens do we ignore; do
    not compute how much time it needs." `SETTLE_SAFETY_MARGIN`,
    `SETTLE_READ_BPS_DEFAULT`, `PREFILL_TOKENS_PER_S_DEFAULT`, `SETTLE_MIN_MS` and the
    two env knobs that fed them are deleted with this docstring paragraph - a knob
    nothing computes on is worse than no knob, because a payload will still report it.

    `bytes_at_stake` is PUBLISHED AND NOTHING ELSE. It is the bytes still in flight at
    the apply point (a batch that delivered nine of ten slots is waiting for one), which
    is the quantity a reader of `last_apply_window` needs beside the wait - but it moves
    no part of the grant, and the same is true of the offered-token estimate: it is a
    description of the batch, not an input to the wait.

    WHICH BOUND APPLIED is published because the three answers mean three different
    things: `off` = the allowance is 0, this boot never waits; `flat` = the operator's
    allowance decided it, the ordinary case; `deadline` = the read's own expiry decided
    it, so the wait was cut short by the clock that is cancelling the read and a drop
    here is a DEADLINE problem, not an allowance problem.

    WHAT IS MEASURED INSTEAD OF GUESSED: `settle_allowance_spent` counts the batches
    that consumed the WHOLE granted allowance and still had bytes in flight - the honest
    successor to `budget_over_recompute`, which compared a grant against its own
    arithmetic and therefore read `over_recompute: 0` for exactly the 23.6 ms loss
    above.
    """
    env = config.settle_env()
    raw_cap = env["cap_ms"] if env["cap_ms"] is not None else SETTLE_MS_DEFAULT
    cap_ms = max(0.0, min(float(raw_cap), SETTLE_MS_MAX))
    payload = max(0, int(bytes_at_stake or 0))
    if cap_ms <= 0.0:
        granted, bound = 0.0, "off"
    else:
        granted, bound = cap_ms, "flat"
    deadline_ms = None
    if seconds_left is not None:
        try:
            deadline_ms = max(0.0, float(seconds_left)) * 1000.0
        except (TypeError, ValueError):
            # An unreadable `seconds_left` is NO deadline, not a deadline of zero: the
            # grant stays the flat allowance. (The pre-0.2.32 body compared None with a
            # float here and raised TypeError; `test_grant_is_the_flat_cap_or_the_loads_
            # own_deadline` is the test that found it.)
            deadline_ms = None
        if deadline_ms is not None and deadline_ms < granted:
            granted, bound = deadline_ms, "deadline"
    return {"enabled": cap_ms > 0.0, "cap_ms": round(cap_ms, 3),
            # THE DEFAULT MUST WEAR ITS NAME. Since 0.2.32 an unset
            # `VLLM_E2_TIER_DISK_SETTLE_MS` means a 5,000 ms allowance instead of never
            # waiting, so a boot that waits on a command line that never asked to wait
            # has to be able to say WHY in one line - otherwise "the operator wrote 40
            # and the box waited 5 s" and "nobody set anything" are the same payload,
            # and only one of them is a mis-set drop-in.
            "cap_from_default": env["cap_ms"] is None and not env["problems"],
            # Named, not swallowed: an absurd allowance is clamped and the reader is
            # told a ceiling did it, so a typo in a systemd drop-in is a visible clamp
            # and not a mystery wait at 3 a.m.
            "ceiling_clamped": float(raw_cap) > SETTLE_MS_MAX,
            "bytes_at_stake": payload,
            "allowance_ms": round(cap_ms, 3),
            "budget_ms": round(granted, 3), "bound": bound,
            "deadline_ms": None if deadline_ms is None else round(deadline_ms, 3),
            "env_problems": list(env["problems"])}


def _settle_wait(batch_id, status: dict, budget_ms: float) -> tuple[dict, float]:
    """Poll `batch_id` until it settles, expires, or the budget is gone.

    A POLL, not a condition wait, and deliberately so: `disk_read`'s reader thread
    finishes jobs on its own and publishes them under its lock, and the apply point
    runs on the engine's step thread - a `Condition.wait` here would put the engine's
    step behind a lock held by a thread doing kernel reads, which is the same
    mistake in a new costume. `time.sleep` yields the GIL, so the reader runs between
    looks; `SETTLE_POLL_SECONDS` (0.5 ms) is far below the ~10 ms the smallest offer
    reads in and far above the cost of one context switch.

    BOUNDED THREE WAYS BY CONSTRUCTION, because an unbounded wait inside an engine
    step is a hang wearing a performance patch: the granted budget, the batch's own
    `seconds_left`, and `disk_read`'s deadline expiry - whichever ends the loop first.
    Never wait with the state lock held; the caller must not hold it.
    """
    started = time.monotonic()
    ceiling = max(0.0, float(budget_ms)) / 1000.0
    while True:
        if (status.get("settled") or status.get("incomplete")
                or status.get("deadline_expired") or not status.get("known", True)):
            break
        left = ceiling - (time.monotonic() - started)
        seconds_left = status.get("seconds_left")
        if isinstance(seconds_left, (int, float)):
            left = min(left, float(seconds_left))
        if left <= 0:
            break
        time.sleep(min(SETTLE_POLL_SECONDS, left))
        status = disk_read.batch_status(batch_id)
    return status, (time.monotonic() - started) * 1000.0


def _offered_tokens_estimate(batch: dict) -> tuple[int | None, str]:
    """Tokens this batch is offering, from what a WORKER rank is able to see.

    The offered-token count is a scheduler-side fact (`_loads[rid]["tokens"]`) and
    the scheduler is a different PROCESS - the number cannot be handed over without
    widening the metadata the fork owns, which this release does not do. What a rank
    CAN see is the batch's own shape: one disk entry per (span, group), so

        spans  = entries / distinct groups in the batch
        tokens = spans * scheduler_block_size        (from the store fingerprint)

    which is an ESTIMATE and is published named as such. It is exact for the uniform
    case the loader builds today (every span carries every cacheable group) and drifts
    only where the scheduler truncated a span - and the direction is published with
    it, so a reader can tell `derived` from `unknown` and never from a guess.
    """
    jobs = batch.get("jobs") or {}
    if not jobs:
        return None, "no disk entries in the batch"
    groups = {int(job.get("group") or 0) for job in jobs.values()}
    try:
        fingerprint = disk_store.counters().get("fingerprint") or {}
        block_tokens = int(fingerprint.get("scheduler_block_size") or 0)
    except Exception:        # noqa: BLE001 - an estimate is never worth a fault
        block_tokens = 0
    if block_tokens <= 0:
        return None, "scheduler_block_size is not in the store fingerprint"
    spans = len(jobs) // max(1, len(groups))
    return spans * block_tokens, (
        f"{len(jobs)} entries / {len(groups)} groups x {block_tokens} tokens/block")


def _note_apply_window(window: dict) -> None:
    """Fold one completed apply-point observation into the counters.

    Kept separate from the observation so the MEASUREMENT is not gated on the wait:
    the numbers that decide whether waiting is worth anything are exactly the numbers
    a default boot (cap 0, never waiting) produces. Every field is a copy of
    something already read, the sample lists are bounded, and nothing here can raise
    or wait - it runs on the engine's step thread.

    NO LOCK, deliberately, in common with every other counter in this module: the
    apply point runs on the worker's step thread and nothing else writes these keys,
    so a lock here would only add the re-entry hazard `test_no_nonreentrant_lock_
    reentry` exists to catch.
    """
    gap = window.get("bind_to_apply_ms")
    elapsed = window.get("elapsed_ms")
    wait_granted = bool(window.get("wait_granted"))
    _state["apply_observations"] += 1
    if window.get("settled_on_arrival"):
        _state["apply_settled_on_arrival"] += 1
    _state["last_apply_window"] = {key: value for key, value in window.items()
                                   if key not in ("status", "job_states")}
    for key, value in (("bind_to_apply", gap), ("apply_elapsed", elapsed)):
        if value is None:
            continue
        _state[f"{key}_total_ms"] += float(value)
        low = _state[f"{key}_min_ms"]
        high = _state[f"{key}_max_ms"]
        if low is None or value < low:
            _state[f"{key}_min_ms"] = float(value)
        if high is None or value > high:
            _state[f"{key}_max_ms"] = float(value)
    waited = window.get("waited_ms") or 0.0
    if waited:
        _state["settle_ms_total"] += float(waited)
        if _state["settle_ms_max"] is None or waited > _state["settle_ms_max"]:
            _state["settle_ms_max"] = float(waited)
    budget = window.get("budget_ms")
    if wait_granted:
        _state["settle_waits_granted"] += 1
        _state["settle_budget_total_ms"] += float(budget or 0.0)
        if (budget
                and (_state["settle_budget_max_ms"] is None
                     or budget > _state["settle_budget_max_ms"])):
            _state["settle_budget_max_ms"] = float(budget)
    if window.get("settled_after_wait"):
        _state["settled_after_wait"] += 1
    if window.get("dropped_after_wait"):
        _state["dropped_after_wait"] += 1
    if wait_granted:
        # Bytes the arithmetic was actually asked to wait for. Not counted when the
        # wait was off or the batch had already settled: a default boot has bytes in
        # flight on every disk step, and summing them into a "waited" field would
        # report waiting that never happened.
        _state["settle_bytes_waited"] += int(window.get("bytes_at_stake") or 0)
    bound = window.get("bound")
    if bound in ("flat", "deadline", "off"):
        _state[f"settle_bound_{bound}"] += 1
    if window.get("allowance_spent"):
        _state["settle_allowance_spent"] += 1
    for name, value in (("bind_to_apply_ms", gap), ("apply_elapsed_ms", elapsed),
                        ("settle_ms", window.get("waited_ms")),
                        ("settle_budget_ms", window.get("budget_ms"))):
        if value is None:
            continue
        samples = _apply_samples[name]
        if len(samples) < MAX_SETTLE_SAMPLES:
            samples.append(float(value))


def _median(values: list) -> float | None:
    """The median of the retained samples, or None when nothing was retained."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 3)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 3)


def apply_window() -> dict:
    """The bind-to-apply window, and every wait this boot spent, as one dict.

    THE QUESTION THIS EXISTS TO ANSWER, stated once so the fields have a referent:
    the fork calls `bind_connector_metadata` and `start_load_kv` in the same step
    (`kv_connector_model_runner_mixin.py:80-93`), and because every disk hit is
    offered with `load_kv_async=True` the fork defers `start_load_kv` to after the
    forward LAUNCH - so the gap is one CPU-side forward launch, NOT one forward pass.
    Whether the reads win that gap has been an inference from one counter
    (`dma_unsettled_entries`) and a `read_seconds_max` of 0.0 that was read as "no
    read ever ran" when it actually means "no read that reached the kernel ever
    finished". These are the measurements.

    `min`/`max` are BOOT-WIDE. `median` is over the last `MAX_SETTLE_SAMPLES`
    observations and says so in `median_samples` - a median silently truncated to the
    first 512 steps would be a number that changes meaning without changing its name.
    """
    state = dict(_state)
    samples = {name: list(values) for name, values in _apply_samples.items()}
    out = {"observations": state["apply_observations"],
           "settled_on_arrival": state["apply_settled_on_arrival"],
           "samples_retained": MAX_SETTLE_SAMPLES, "windows": {}}
    for key, label in (("bind_to_apply", "bind_to_apply_ms"),
                       ("apply_elapsed", "apply_elapsed_ms")):
        count = state["apply_observations"] or 1
        held = samples[f"{key}_ms"]
        out["windows"][label] = {
            "min_ms": (None if state[f"{key}_min_ms"] is None
                       else round(state[f"{key}_min_ms"], 3)),
            "max_ms": (None if state[f"{key}_max_ms"] is None
                       else round(state[f"{key}_max_ms"], 3)),
            "avg_ms": round(state[f"{key}_total_ms"] / count, 3),
            "median_ms": _median(held),
            "median_samples": len(held)}
    held = samples["settle_ms"]
    budgets = samples["settle_budget_ms"]
    granted = state["settle_waits_granted"] or 0
    posture = settle_budget(0)
    out["wait"] = {
        # THE POSTURE, before any batch: what the environment says today, so a reader
        # can tell "the wait is off" from "the wait is on and nothing needed it".
        "posture": {key: posture[key] for key in
                    ("enabled", "cap_ms", "allowance_ms", "cap_from_default",
                     "ceiling_clamped",
                     "env_problems")},
        "waits_granted": granted,
        "settled_after_wait": state["settled_after_wait"],
        "dropped_after_wait": state["dropped_after_wait"],
        "allowance_spent": state["settle_allowance_spent"],
        "bytes_waited": state["settle_bytes_waited"],
        "waited_ms": {"avg": (None if not granted else
                              round(state["settle_ms_total"] / granted, 3)),
                      "max": (None if state["settle_ms_max"] is None
                              else round(state["settle_ms_max"], 3)),
                      "median": _median(held), "median_samples": len(held)},
        "budget_ms": {"avg": (None if not granted else
                              round(state["settle_budget_total_ms"] / granted, 3)),
                      "max": (None if state["settle_budget_max_ms"] is None
                              else round(state["settle_budget_max_ms"], 3)),
                      "median": _median(budgets), "median_samples": len(budgets)},
        "bound": {"flat": state["settle_bound_flat"],
                  "deadline": state["settle_bound_deadline"],
                  "off": state["settle_bound_off"]},
        "last": state["last_apply_window"],
        "notice": state["settle_notice"],
        # The posture's units. `cap_from_default` is the field that answers "did the
        # operator set this, or is this the shipped default?" - a question that did not
        # exist before 0.2.32, when unset meant OFF.
        "units": {"enabled": "does this boot wait at the apply point at all",
                  "cap_ms": "milliseconds, the FLAT allowance (`SETTLE_MS`)",
                  "allowance_ms": "milliseconds, the same cap restated as the policy "
                                  "word rather than the env-var word",
                  "cap_from_default": "true when "
                                      "`VLLM_E2_TIER_DISK_SETTLE_MS` is unset and the "
                                      "5,000 ms default decided it - the field that "
                                      "separates 'nobody set it' from 'the value was "
                                      "ignored'",
                  "ceiling_clamped": "true when the env value exceeded "
                                     "`SETTLE_MS_MAX` and was clamped",
                  "env_problems": "the environment's own complaints, verbatim",
                  "waits_granted": "batches the code actually sat in front of",
                  "allowance_spent": "batches that used the WHOLE allowance and still "
                                     "had bytes in flight",
                  "bound": "which bound stopped the wait, per boot: `flat` (the "
                           "allowance), `deadline` (the read's own expiry), `off` "
                           "(no wait at all)"}}
    out["recompute"] = {"applied_bytes": state["bytes_applied"],
                        "dma_unsettled_entries": state["dma_unsettled_entries"],
                        "read_seconds_avg": read_latency()[0],
                        "read_seconds_max": read_latency()[1]}
    return out


def counters() -> dict:
    """A copy of the counters for a payload. Never the ledgers themselves."""
    out = dict(_state)
    out["refused"] = dict(_state["refused"])
    out["reads_failed"] = dict(_state["reads_failed"])
    # THE POSTURE, beside the numbers it explains: "offers: 0" means something
    # different under `off` than under `disk_authority`, and the reader should not
    # have to combine two booleans to find out which.
    out["serving_as"] = serving_as()
    out["tracked_loads"] = len(_loads)
    out["tracked_slots"] = len(_slot_paths)
    out["tracked_worker_loads"] = len(_wloads)
    out["pending_invalid_blocks"] = len(_invalid)
    out["suppressed_requests"] = len(_suppressed)
    out["methods_fitted"] = list(_state["methods_fitted"])
    out["missing_methods"] = list(_state["missing_methods"])
    out["role_here"] = role_here()
    return counters_view(out)


def counters_view(out: dict) -> dict:
    out["units"] = {
        "lookups": "lookups reaching the restore path",
        "offers": "requests a disk offer was made for",
        "offered_tokens": "tokens offered beyond the host-RAM hit",
        "installs": "times the restore layer was fitted in this process",
        "install_refusals": "install passes refused in this process because a "
                            "class was there and a method was not (or was "
                            "refused): each one has a line in the log",
        "install_deferral_line": "the last deferral line: what this process said "
                                 "when it had nothing to hook yet",
        "install_deferrals": "install attempts that found no tier class imported "
                             "in this process yet: the EXPECTED state at plugin "
                             "load, counted separately so a healthy boot never "
                             "reports a refusal it did not have",
        "blocks_offered": "scheduler blocks offered from disk",
        "errors": "internal exceptions caught and counted by this module",
        "fallbacks": "degradations to the fork's own behaviour, each with a reason",
        "reinstalls": "times install() found the layer already fitted",
        "uninstalls": "times the layer was taken back off the fork in this process",
        "load_error_installs": "times the load-error report was fitted on a class",
        "adopted": "requests whose disk slots reached the worker",
        "adopted_tokens": "tokens the scheduler adopted from disk",
        "restorations": "loads with a DISK READ APPLIED behind them: a finished "
                        "report with no read is counted as `no_read` instead, so "
                        "this counter cannot rise on a boot where nothing was "
                        "actually read from disk",
        "no_read": "finished load reports with NO applied disk read behind them "
                   "(an adopted offer that no worker filled): refusals of the "
                   "counter, never restorations",
        "aged_out_loads": "loads the LEDGER released without ever correlating a "
                          "`finished_recving` for them - the fork stopped owning the "
                          "request (its own stale-pin release), the deadline expired, "
                          "or a cap eviction forgot it. A release is a statement about "
                          "the ledger, not about the bytes (unit: loads)",
        "aged_out_with_applied_reads": "of those releases, the ones whose slots had "
                                       "reached a worker AND a disk read had been "
                                       "applied in this boot: bytes were on their way "
                                       "when the ledger let go, so a later report is "
                                       "expected and is counted by name (unit: loads)",
        "late_finished_reports": "`finished_recving` ids that arrived for a load the "
                                 "ledger had already released and were MATCHED to that "
                                 "release. Before 0.2.25 this exact case was the bare "
                                 "`continue` that made 36 applied reads look like "
                                 "`restorations=0` with nothing to explain it (unit: "
                                 "reports)",
        "restored_after_release": "of those late reports, the ones whose "
                                  "release showed an adopted load with an applied "
                                  "disk read behind it: a restoration the ledger "
                                  "could not count as one, counted here instead of "
                                  "in `restorations`, which keeps meaning 'reported "
                                  "while the ledger held it' (unit: loads)",
        "disk_gpu_blocks_dma_verified": "GPU BLOCKS this module mapped for a disk span "
                                        "(their CPU slots carried our emitted paths) "
                                        "and the fork then reported DMA-complete. THE "
                                        "ONLY FIELD THAT COUPLES OUR CLAIM TO THE "
                                        "FORK'S COPY: an offer cannot move it, only an "
                                        "adopted load reported finished can (unit: GPU "
                                        "blocks, not requests)",
        "disk_gpu_blocks_released_unreported": "GPU blocks that were mapped for a disk "
                                               "span and left the ledger with no "
                                               "report either way - the complement of "
                                               "`disk_gpu_blocks_dma_verified`, and "
                                               "the honest half of 'were the blocks "
                                               "written?' (unit: GPU blocks)",
        "last_aged_out": "the last report-less ledger release in one sentence: which "
                         "door released it, whether it was adopted, whether a read was "
                         "applied, and how many GPU blocks it had mapped",
        "last_groups_required": "WHAT A CLAIM COSTS EVERY CACHEABLE GROUP (0.2.25), "
                                "priced on the offer door AND the refusal door: the "
                                "claimed range, `offered`, and a row per "
                                "prefix-cacheable group (`blocks_needed`, "
                                "`null_blocks`, `contract`, `restored`). Derived only "
                                "from `group_serviceability` and the claim's geometry, "
                                "so it reads no manager and knows no model",
        "group_serviceability": "THE GROUP ARITHMETIC OF THE LAST LOOKUP, as data: "
                                "which groups are prefix-cacheable, which are "
                                "positionally stable, which are restorable, which "
                                "BLOCKED THE OFFER AND WHY, and `complete` - whether "
                                "every prefix-cacheable group can be restored at all. "
                                "`complete: false` means this model can NEVER be "
                                "served from disk (a hybrid GDN + full-attention model "
                                "in align mode is the live case) and an operator "
                                "learning that from the payload need not read the "
                                "source or know the architecture",
        "disk_reads_applied": "scheduler-side applied disk reads (0 in a "
                              "scheduler-only process by construction)",
        "ranks_reads_applied": "applied disk reads attested by live worker rank "
                               "files at the last proof read",
        "offers_enabled": "this boot's gap-fill switch: False until the dev verb "
                          "raises it, so an armed spill tier alone cannot put a "
                          "disk offer in front of live traffic. Up on its own it "
                          "means RAM first and the disk filling the gap; "
                          "`disk_authority` is the switch that makes the disk "
                          "decide instead",
        "disk_authority": "this boot's DISK-AUTHORITY switch, and the whole answer "
                          "to 'is everyone being served from SSD?': true means the "
                          "disk, not the host-RAM hit, decides EVERY request's own "
                          "prefix. False at every fresh boot, set only by "
                          "`{\"op\":\"authority\",\"enabled\":true}`, and a restart "
                          "turns it off",
        "serving_as": "the two switches read as one verdict: `off` (nothing is "
                      "served from disk), `disk_authority` (EVERY request, with the "
                      "disk deciding its prefix) or `ram_first_gap_fill` (only the "
                      "prefix the host-RAM tier left short). Derived from the two "
                      "switches by `disk_posture()` and published nowhere else",
        "disk_authority_changes": "times the disk-authority switch was raised or "
                                  "dropped in this boot: a posture moved once per "
                                  "experiment, so a rising count is someone "
                                  "deciding mid-flight",
        "offer_switch_changes": "times this boot's gap-fill switch was raised or "
                                "dropped: a boot that never offered and never "
                                "changed hands says OFF and 0",
        "authority_lookups": "lookups that ran under disk authority, so the disk "
                             "and not the host-RAM tier decided the span (unit: "
                             "lookups). While the switch is up this is the offer "
                             "path's whole population; while it is down it stays 0",
        "local_hit_ignored": "OFFERS committed under disk authority whose host-RAM "
                             "hit was non-zero and was discarded to let the disk "
                             "span start at the local prefix instead. A refusal "
                             "counts here NEVER: a refused offer leaves the RAM hit "
                             "exactly where the fork put it",
        "local_hit_tokens_ignored": "tokens of host-RAM hit discarded by those "
                                    "offers: what the request gave up to exercise "
                                    "the disk path (unit: tokens)",
        "pin_release_failures": "times the fork's pinned host-RAM blocks could not "
                                "be released when disk authority chose the disk over "
                                "them: each one is a pin that may outlive the "
                                "request, so it must stay 0 (an unavailable "
                                "`_free_pending_cpu_hit` refuses up front instead)",
        "last_local_hit_ignored": "the last discarded host-RAM hit in the words an "
                                  "operator reads: request id, the tokens the fork "
                                  "reported, the block the disk span starts at",
        "witness_kind": "the stable name of the last verdict: the emission key is "
                       "(verdict, kind) and never the sentence, because the sentence "
                       "quotes counters. AS OF THE LAST EMISSION, NOT NOW: it is the "
                       "name the logged line carried, and a counter can move without "
                       "changing the verdict, so read `witness_kind_now` for the same "
                       "arithmetic on today's numbers (0.2.25: "
                       "`witness_kind=offers_not_adopted` was published beside "
                       "`adopted=1` on the live box)",
        "witness_verdict": "the last restore-posture verdict this process spoke "
                           "(CONFIRMED / NO READ / ...): a verdict, never a count, "
                           "and read beside `reads_applied` rather than instead of "
                           "it - on 2026-10-05 a CONFIRMED verdict sat next to "
                           "reads_applied=0",
        "witness_reason": "the sentence the witness last gave for its verdict; "
                          "prose for a human, never a number to aggregate",
        "suppressed_hits": "lookups refused because the REQUEST is suppressed after "
                           "a failed disk load (the loop that starved the box)",
        "suppressed_requests": "request ids currently black-holed for disk "
                               "service in this boot (the fuse table's live "
                               "size, not a counter)",
        "suppressed_spans": "offers refused because the disk SPAN was implicated in "
                            "a failed load, whichever request asked for it",
        "ranks_seen": "live rank files that spoke about disk service at all, "
                      "armed or not: ranks below this means a partial fleet, and a "
                      "partial fleet refuses every offer",
        "ranks_armed": "rank files that will SERVICE a disk load: the proof is AND "
                       "over these, so one disarmed rank closes the whole scheduler",
        "suppressions": "requests black-holed for disk service in this boot",
        "races_not_fused": "failure reports classified a RACE (a timing or capacity "
                           "kind, no data kind visible) and therefore cost the request "
                           "its tokens but NOT its next offer (0.2.32)",
        "fuses_on_data_kinds": "fuses raised with a DATA kind visible - corruption, a "
                               "short read, a bad digest, a failed copy: the fuse the "
                               "loop-killer was built for, kept",
        "fuses_on_unknown_kinds": "fuses raised because NO failure kind was visible "
                                  "(fail closed: the scheduler reads a worker's kinds "
                                  "from its rank file, so an undiagnosed loss is not "
                                  "proof of a race)",
        "race_reoffers_granted": "second offers granted by the one-per-request race "
                                 "allowance: the loop bound that replaced the "
                                 "boot-length fuse for a timing loss (a request is "
                                 "offered at most twice, race or not)",
        "refusals_total": "refusals", "paths_emitted_steps": "steps",
        "path_entries": "load_paths entries naming a file",
        "path_entries_empty": "load_paths entries naming nothing",
        "path_length_violations": "steps where the length rule was broken",
        "slots_allocated": "CPU slots allocated as disk destinations",
        "stale_slots_dropped": "mappings dropped as their load died",
        "watchdog_fires": "scheduler-side ledger expiries",
        "index_refreshes": "journal refreshes",
        "index_refresh_errors": "raised refreshes",
        "ranks_proven": "rank files attesting a live worker half",
        "proof_reads": "RANK FILES read for that proof - nothing to do with KV "
                       "reads: see `reads_applied` for the applied disk reads",
        "loads_released": "loads removed from the ledger",
        "binds": "worker binds observed", "disk_binds": "binds carrying disk work",
        "reads_started": "file reads queued", "reads_applied": "reads memmoved",
        "bytes_applied": "bytes", "dma_unsettled_entries": "load entries not DMA'd",
        # THE RACE, MEASURED (0.2.23). Averages are recomputed from the sums in
        # `apply_window()`; only whole things are listed here as ints.
        "apply_observations": "batches seen at the apply point (`start_load_kv`)",
        "apply_settled_on_arrival": "of those, batches already settled when the apply "
                                    "point looked: the restore race, WON",
        "applied_settled_immediately": "disk entries memmoved at an apply point that "
                                       "found the batch settled and waited nothing",
        "dropped_queued": "disk entries dropped unsettled whose read had NOT STARTED "
                          "(the job sat in the queue): a thread-scheduling loss, and "
                          "the kind a wait can fix",
        "dropped_running": "disk entries dropped unsettled whose read was RUNNING "
                           "(mid-kernel when the engine arrived): a device-speed loss, "
                           "which a wait only fixes if the budget exceeds the read",
        "dropped_unattributed": "disk entries dropped unsettled whose per-job state "
                                "could not be read: unknown, and kept separate so the "
                                "other two stay a claim they can be trusted to make",
        "spans_observed": "spans seen at the apply point (a span is one scheduler "
                          "block across every cacheable group, so it restores whole "
                          "or is worth nothing to the engine)",
        "span_entries_observed": "disk entries those spans were made of (unit: "
                                 "entries; entries_observed >= spans_observed)",
        "spans_whole": "spans whose EVERY group had bytes applied",
        "spans_partial": "spans where SOME groups applied and others did not: the "
                         "shape a partial restore takes once readers are concurrent",
        "spans_empty": "spans where nothing at all applied (the harmless shape: the "
                       "engine recomputes the block and nothing claims it was read)",
        "spans_hole_in_the_middle": "batches whose applied spans are NOT a prefix of "
                                    "the offer. With one reader thread this is "
                                    "arithmetically impossible (submission order is "
                                    "span-major and completions followed it), so a "
                                    "rising count IS the measurement that N readers "
                                    "changed the SHAPE of an interrupted restore and "
                                    "not only its speed. Counted, never acted on",
        "last_span_shape": "the most recent batch's span shape as data: spans, whole, "
                           "partial, empty, `applied_is_prefix`, and which span "
                           "indices came away partial (bounded by MAX_SPANS_REPORT "
                           "with `truncated` naming the rest)",
        "offer_pool_capped": "offers SHORTENED to the job count the bounce-buffer pool "
                             "can hold at once (one read holds its buffer from issue "
                             "to the apply point, so an offer needs the pool to hold "
                             "ALL of its buffers); the bound that binds in practice "
                             "now that the fixed token cap is deleted",
        "offer_pool_refused": "lookups given no disk offer at all because the pool "
                              "cannot carry one span's reads: the refusal that "
                              "replaces the 0.2.30 shape, where the offer was made, "
                              "refused at bind as `buffer_cap` and fused the request",
        "offer_block_capped": "offers the per-offer BLOCK cap bound: the positional "
                              "walk stopping at MAX_RESTORE_BLOCKS_PER_OFFER while the "
                              "request still had tokens left. The only per-offer CAP "
                              "counter left since 0.2.32 deleted the fixed "
                              "32,000-token clamp (`offer_clamped`)",
        "last_offer_pool": "the most recent offer's bounce-buffer arithmetic: bytes "
                           "per job and where that price came from, the pool budget "
                           "and what was committed, jobs needed versus jobs the pool "
                           "can carry, and `bounded_by`",
        "load_failures": "load-failure RECORDS written by this process (one per "
                         "refused or failed read), bounded by the ring cap",
        "load_failures_overwritten": "records the ring recycled at its cap",
        "ranks_load_failures": "load-failure records read from live worker rank "
                               "files (the transport `ranks_reads_applied` uses)",
        "failure_log_lines": "DISK[load-fail] lines this process logged, one per "
                             "episode per kind",
        "failure_log_suppressed": "lines the rate limit suppressed, so a quiet log is "
                                  "provably a quiet tier and not a silent one",
        "invalid_block_ids_seen": "GPU block ids the fork reported as load errors, "
                                  "counted over the boot instead of discarded",
        "last_invalid_block_ids": "the ids from the most recent report (bounded)",
        "settle_waits_granted": "batches given a non-zero settle budget",
        "settled_after_wait": "batches that settled during their wait: the waits that "
                              "bought bytes",
        "dropped_after_wait": "batches still unsettled when their wait expired: the "
                              "waits that were spent for nothing",
        "settle_bytes_waited": "bytes a settle wait was waiting for, summed over "
                               "batches (published alongside the flat allowance, which "
                               "it does not decide)",
        "settle_bound_flat": "allowances decided by the flat cap (`SETTLE_MS`): the "
                             "ordinary regime since 0.2.32 deleted the proportional "
                             "arithmetic",
        "settle_bound_deadline": "allowances cut short by the read's own deadline: the "
                                 "wait was stopped by the clock that cancels the read, "
                                 "so a drop here is a deadline problem and not an "
                                 "allowance problem",
        "settle_bound_off": "apply points that saw the allowance at 0 (never wait): "
                            "the population of a boot that switched the wait off",
        "settle_allowance_spent": "batches that consumed the WHOLE granted allowance "
                                  "and still had bytes in flight - the honest wait "
                                  "metric, "
                                  "and the successor to `budget_over_recompute`, which "
                                  "could not see a dropped load at all",
        "invalid_reports": "GPU block ids reported as load errors",
        "finished_released": "requests released without a DMA",
        "witness_emissions": "witness lines logged in this boot",
        "witness_suppressed": "witness emissions withheld because the verdict and "
                              "reason had already been said (the flood fix: "
                              "counters move every step, the verdict does not)",
        "worker_watchdog_fires": "worker-side load expiries",
        "phantom_guard_trips": "invariant violations (must stay 0)",
        "region_plan_errors": "slot address plans that failed",
        "worker_errors": "raised worker-path exceptions",
        "tracked_loads": "requests with an outstanding disk load",
        "tracked_slots": "cpu-slot mappings held",
        "tracked_worker_loads": "worker watchdog entries",
        "pending_invalid_blocks": "GPU block ids awaiting their report",
    }
    return out


# --- which role is THIS process? ---------------------------------------------

_sched_cls_seen = None
"""The manager class this process was ASKED to fit, when the name missed.

`_class_of` resolves by name out of `sys.modules`, which is right for the box - and
it is the only reason a boot can say "no tier class is imported here" while holding
the live manager object in its hand. The connector-creation boot has that object
(`type(connector.scheduler_manager)`), so it hands it over and the fit does not
depend on the module being reachable under the name this file expects. A fork that
relocated the module must not be able to turn a fitted scheduler into an
offer-less one by moving a file.
"""


def _sched_class(known=None):
    """The scheduler class THIS process can fit: by name, else the one in hand."""
    cls = _class_of(SCHED_MODULE, SCHED_CLASS)
    if cls is None and known is not None \
            and getattr(known, "__name__", None) == SCHED_CLASS:
        cls = known if isinstance(known, type) else None
    elif cls is None and _sched_cls_seen is not None \
            and getattr(_sched_cls_seen, "__name__", None) == SCHED_CLASS:
        cls = _sched_cls_seen
    return cls


def role_here() -> str:
    """Which halves of the tier this process has classes for. Never raises.

    `scheduler` is the EngineCore: the process the fork calls
    `get_num_new_matched_tokens` on, and therefore the ONLY process in which a
    restore can begin. `worker` is a rank: it can move bytes but can never make an
    offer. A worker printing `installed=YES` proves nothing about the scheduler -
    which is exactly the reading this field exists to make impossible.
    """
    try:
        sched = _sched_class() is not None
        worker = _class_of(WORKER_MODULE, WORKER_CLASS) is not None
    except Exception as exc:  # noqa: BLE001 - a role read never breaks a step
        snapshot._error(exc, "disk_restore role_here")
        return ROLE_NONE
    if sched and worker:
        return ROLE_BOTH
    if sched:
        return ROLE_SCHEDULER
    if worker:
        return ROLE_WORKER
    return ROLE_NONE


def sched_fitted_here() -> bool:
    """Is the OFFER-MAKING half wrapped in THIS process, right now?

    Checked against the class, not against `_state`, because the state can be older
    than the class: `backport._rewrap` can rebuild a method from a new base and
    leave our layer off it. `installed=YES` in a rank file and `false` here is the
    exact signature of tonight's failure, and this is the field that shows it.
    """
    try:
        cls = _sched_class()
        if cls is None:
            return False
        return all(_layer_is_ours(cls.__dict__.get(name))
                   or _restore_layer_beneath(cls.__dict__.get(name))
                   for name in SCHED_METHODS)
    except Exception as exc:  # noqa: BLE001 - a verdict of False is a refusal
        snapshot._error(exc, "disk_restore sched_fitted_here")
        return False


def _restore_layer_beneath(method) -> bool:
    """True when our layer sits under a guard/switch wrapper on the same name."""
    if method is None:
        return False
    try:
        from . import backport  # noqa: PLC0415 - the layer vocabulary lives there
        layers, _bottom = backport._layers_of(method)
    except Exception as exc:  # noqa: BLE001 - an unreadable chain is "not fitted"
        snapshot._error(exc, "disk_restore layer walk")
        return False
    return any(kind == LAYER_KIND for kind, _ in layers)


def fitted_methods_here() -> list:
    """The scheduler methods our layer is actually on in this process, in order."""
    try:
        cls = _sched_class()
        if cls is None:
            return []
        return [name for name in SCHED_METHODS
                if _layer_is_ours(cls.__dict__.get(name))
                or _restore_layer_beneath(cls.__dict__.get(name))]
    except Exception as exc:  # noqa: BLE001 - a name list is never a step fault
        snapshot._error(exc, "disk_restore fitted_methods_here")
        return []


# --- small helpers ------------------------------------------------------------

def _fail(reason, **extra) -> dict:
    out = {"ok": False, "reason": str(reason)[:MAX_STRING_CHARS]}
    out.update(extra)
    return out


def _ok(**fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    return out


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int(value) -> int | None:
    if _plain_int(value):
        return int(value)
    if isinstance(value, (str, bytes)) and str(value).isdigit():
        return int(value)
    return None


def _clip(text, chars: int = MAX_STRING_CHARS):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= chars else text[:chars] + "..."


def _note(reason) -> None:
    _state["last_reason"] = _clip(reason)


def _refuse(kind: str, reason: str) -> None:
    """Count an offer refusal under its closed-vocabulary name."""
    name = kind if kind in _state["refused"] else "bad_call"
    _state["refused"][name] += 1
    _state["refusals_total"] += 1
    _state["last_refusal"] = _clip(f"{name}: {reason}")
    _note(reason)


def _worker_fail(kind: str, reason: str) -> dict:
    name = kind if kind in _state["reads_failed"] else "read_failed"
    _state["reads_failed"][name] += 1
    _note(reason)
    # THE WHOLE-STEP REFUSALS, recorded too. A bind that queued nothing because the
    # identity is unreadable or the metadata's lists disagree is the biggest "why did
    # no read happen" shape there is, and until now it moved a counter in the worker
    # and said nothing anywhere the scheduler or the operator could read. Rate-limited
    # with everything else here, so a gate that stays closed costs one line per window
    # and not one per step.
    _note_load_failure(name, where="bind", reason=reason)
    return _fail(reason, reason_kind=name)


# --- the gate -----------------------------------------------------------------

def serving_as() -> str:
    """The two boot switches, read as ONE verdict. Derived here and nowhere else.

    A payload with two booleans invites the reader to do the arithmetic, and the
    arithmetic is the part that matters: "am I serving everyone from SSD?" must be one
    field read. Exactly three answers, and they are exhaustive.
    """
    if _state["disk_authority"]:
        # Authority implies offering: a boot that made the disk the authority has
        # decided to serve from disk, so the gap-fill switch cannot contradict it.
        return "disk_authority"
    if _state["offers_enabled"]:
        return "ram_first_gap_fill"
    return "off"


def disk_posture() -> dict:
    """This boot's disk posture: who may be served, and whose answer decides. Never
    raises, reads no file.

    THREE facts, all per-boot, all written by `set_disk_authority()` and
    `set_offer_switch()` - the `/disk` dev verb over a POST, and since 0.2.33 this
    package's own boot default calling the same functions - and all lost by a restart,
    which hands the decision back to the boot default. That is stated in the payload
    because an operator deciding "is this boot safe to leave running?" needs to know
    whether the posture survives a crash. It does not, deliberately: an unhealthy boot
    comes back the way every boot now comes back, by the flag and the boot's own gates,
    and a boot whose gates fail comes back giving disk authority to nobody.

    `disk_authority` - THE DISK IS THE AUTHORITY FOR EVERY REQUEST. Each request's
      host-RAM hit is treated as zero tokens for the offer arithmetic and released, so
      the disk index decides the span from the local GPU-computed boundary. This is
      "serving everyone from SSD": it opens the offer gate BY ITSELF, so one POST is
      the whole change, and nothing - no config file, no arm, no recovery path, no
      request field - can raise it instead of the setter (which the dev verb calls, and
      since 0.2.33 the boot default calls too).
    `offers_enabled` - the cheaper posture: RAM first, the disk filling the gap.
      Arming the spill tier is not this: on 2026-10-05 an arm raised for a spill
      measurement was enough to put an un-requested 160,000-token disk offer in front
      of live traffic 123,474 times, because the restore path keyed off the spill arm.
    `serving_as` - the two switches collapsed into one verdict, so a poller never has
      to guess which of them it was looking at.
    """
    return {"disk_authority": bool(_state["disk_authority"]),
            "enabled": bool(_state["offers_enabled"]),
            "serving_as": serving_as(),
            "per_boot": True,
            "survives_restart": False,
            "disk_authority_changes": int(_state["disk_authority_changes"]),
            "offer_switch_changes": int(_state["offer_switch_changes"]),
            "authority_lookups": int(_state["authority_lookups"]),
            "lookups": int(_state["lookups"]),
            "suppressed_requests": len(_suppressed)}


def disk_boot_section() -> dict:
    """THE BOOT DEFAULT BEHIND THIS BOOT'S POSTURE (0.2.33), for the payload. Never
    raises.

    `disk_restore` owns the two serve switches; `disk_boot` owns the decision about what
    a FRESH BOOT should do with them, and whether the arm it raised is still latched.
    (0.2.35 removed the "why renewal is or is not running" half of that: there is no
    renewal, because the arm has no expiry.) This is a pass-through of
    `disk_boot.view()` and nothing else - no recomputation, no
    re-decision - because a payload that re-derived a posture claim from somewhere else
    is how two views start disagreeing in production.

    The except branch returns `disk_boot.static_view()`, the SAME key set built without
    the live arm reads, so a caller that cannot read the arm still gets every field: the
    degraded view is never thinner than the healthy one.
    """
    try:
        from . import disk_boot  # noqa: PLC0415 - it imports this module's posture
        return disk_boot.view()
    except Exception as exc:  # noqa: BLE001 - a view is never worth a fault
        snapshot._error(exc, "disk_restore disk_boot view")
        try:
            from . import disk_boot  # noqa: PLC0415 - same reason
            return disk_boot.static_view()
        except Exception:  # noqa: BLE001 - nothing here is worth a second fault
            return {}


def _switch_verdict(field, enabled, changed, line) -> dict:
    """The one validation and logging path both boot switches share.

    A real boolean or nothing: a payload that says `"true"` or `1` is REFUSED rather
    than obeyed, because a typo in a curl must not change engine behaviour in either
    direction. Both switches are per-boot and neither survives a restart, and the
    caller's response says so rather than leaving it in the docstring.

    `line` is a TEMPLATE with one `%s`, filled in HERE - after the state has moved -
    because every one of these sentences ends by naming the resulting `serving_as`, and
    a caller that formatted it before calling would report the posture the boot is
    LEAVING. A log line that names the wrong posture is worse than no log line: it is
    the record an operator trusts at 3 a.m.
    """
    if not isinstance(enabled, bool):
        return _fail(f"{field} wants a JSON true or false, got "
                     f"{type(enabled).__name__}")
    changed_now = bool(_state[field]) != enabled
    _state[field] = enabled
    if changed_now:
        _state[changed] += 1
    _logger.warning("kvtier-dev-plugin: %s", line % serving_as())
    # `field` and `enabled` travel with the verdict because the route op's response is
    # the operator's receipt: `ok: true` without "which field, to what" is the same
    # ambiguity the switch was introduced to remove.
    return _ok(field=field, enabled=enabled, changed=changed_now,
               posture=disk_posture(),
               per_boot=True, survives_restart=False)


def set_disk_authority(enabled) -> dict:
    """THE global disk-authority switch: the disk decides EVERY request's prefix.

    Import-time value False, and this function is the ONLY writer - reached from the
    `/disk` route op over a POST, and since 0.2.33 from this package's own boot default
    (`disk_boot.py`, gated on `KVTIER_DISK_BOOT_ENABLE`, a variable whose whole point is
    that it DISABLES: it never warms anything up). Not a config file, not the spill arm,
    not any failure-recovery path and not a request field: an unhealthy boot stays off,
    and the boot default that can turn it on is computed fresh from the environment at
    connector creation rather than restored from anything a previous boot wrote.

    Raising it opens the offer gate by itself, so one call serves every request from
    disk; dropping it while `offers_enabled` is down puts the gate back to
    `serving_as=off`. Lowering it from the VERB is an operator's veto and latches the
    boot default's renewal (`disk_boot.note_operator_disarm`, called by the route op and
    NOT by this function) - a manual disarm always beats the boot default.
    """
    return _switch_verdict(
        "disk_authority", enabled, "disk_authority_changes",
        ("DISK[restore] disk_authority=ON for this boot: EVERY request is served from "
         "disk and its host-RAM hit is given back (serving_as=%s). A process starts "
         "with it OFF; this package's own boot default may raise it "
         f"({config.ENV_DISK_BOOT_ENABLE} unset = ON since 0.2.33 - read the "
         "'KVTIER disk boot default=' line), the POST route op changes it either way, "
         "and a restart hands the decision back to the boot default")
        if enabled else
        # The OFF line has to say what is left serving, because turning this off does
        # not necessarily stop serving: if the gap-fill switch is up, the boot keeps
        # offering with RAM leading. A line that only announced "OFF" would read as
        # "nothing is served from disk" on a boot that still is.
        ("DISK[restore] disk_authority=OFF for this boot: the disk no longer decides "
         "any request's prefix and no host-RAM hit is given back (serving_as=%s). The "
         f"{config.ROUTE_PREFIX}{config.ROUTE_DISK} POST changes it, a restart hands "
         f"the decision back to the boot default ({config.ENV_DISK_BOOT_ENABLE}), and "
         "turning it off from that POST latches the boot default's renewal for this "
         "boot, so nothing re-arms a box you just took out of disk service"))


def set_offer_switch(enabled) -> dict:
    """Raise or drop THIS BOOT's gap-fill switch: RAM first, the disk fills the gap.

    Deliberately not an environment variable and not the spill arm: the whole point is
    that a boot nobody asked to serve disk restores does not serve them. Accepts a real
    boolean only.
    """
    return _switch_verdict(
        "offers_enabled", enabled, "offer_switch_changes",
        # The direction word is concatenated rather than %-substituted because the
        # template's ONE `%s` belongs to `_switch_verdict`, which fills the posture in
        # AFTER the write (see there). A half-formatted template is a TypeError.
        "DISK[restore] offers=" + ("ENABLED" if enabled else "disabled")
        + " for this boot (serving_as=%s) - a fresh boot starts with them OFF and "
        "only this verb changes it")


def arena_verdict(manager) -> dict:
    """Can the host-RAM arena THIS process can see service even the smallest offer?

    WHY THE DISK TIER DEPENDS ON THE ARENA AT ALL. Nothing here owns a KV byte. Spill
    reads bytes OUT of the arena's pinned CPU tensors (`disk_spill.read_slot_bytes`
    over the worker's `cpu_kv_caches`) and is TRIGGERED by the RAM tier's own
    store-completion events; restore allocates its destination slots FROM the arena's
    CPU block pool (`cpu_block_pool`, the same pool whose free count clamps an offer
    at `no_cpu_slots`) and memmoves into them before the fork DMAs them to GPU. So an
    arena that is absent, or clamps to a handful of slots per rank (a mis-set
    `KV_OFFLOAD_SIZE`, a missing `VLLM_USE_SIMPLE_KV_OFFLOAD`, or the fork's own
    `max(1, ...)` clamp) leaves the disk tier STRUCTURALLY INERT while its arm file
    still says ARMED - a healthy-looking boot with a dead tier.

    WHAT THIS CAN SEE, AND WHAT IT REFUSES TO GUESS. The pool is the scheduler-side
    half of the arena and the manager handed in is the connector's own, so the free
    count is read exactly the way the offer path reads it. The worker's region tensors
    are NOT visible from EngineCore; their absence is already reported where it is
    known (`fingerprint_incomplete` in a rank file, `no_worker_proof` in the gate) and
    this function says `no_manager` rather than inventing a worker-side verdict it
    cannot take. A pool that cannot hold `CPU_SLOT_RESERVE + 1` free blocks can never
    pass the offer arithmetic, which is the smallest claim this module ever makes.

    Never raises, and reports its own failure as `unreadable` rather than as `ok`: a
    boot verdict that swallows its error is how a dead tier reads as a live one.
    """
    out = {"state": "no_manager", "pool_total_blocks": None,
           "free_blocks": None, "cpu_slot_reserve": CPU_SLOT_RESERVE,
           "blocks_usable_for_an_offer": None,
           "reason": "no connector manager was handed to the boot verdict, so no "
                     "CPU block pool can be read here"}
    try:
        if manager is None:
            # No connector handed to the boot verdict is a DIFFERENT fact from a
            # connector with no arena: the first means "this process cannot see an
            # arena to judge" (a worker, a nested connector), the second means "the
            # tier has no host-RAM arena", which is a fault. Blaming a class that was
            # never there would cry wolf on every process that legitimately has no
            # scheduler.
            return out
        pool = getattr(manager, "cpu_block_pool", None)
        if pool is None:
            named = (type(manager).__name__ if manager is not None else "nothing")
            out["reason"] = (f"{named} exposes no `cpu_block_pool`: the disk tier "
                             "has no host-RAM arena to spill out of or restore "
                             "into, so it cannot move one byte however firmly its "
                             "arm is raised")
            out["state"] = "no_pool"
            return out
        free = _int(pool.get_num_free_blocks())
        total = _int(len(getattr(pool, "blocks", ()) or ()))
        out["free_blocks"] = free
        out["pool_total_blocks"] = total
        if free is None:
            out["state"] = "unreadable"
            out["reason"] = ("the CPU block pool exposes no readable "
                             "`get_num_free_blocks()`, so the arena's size is unknown "
                             "and an offer cannot be priced against it")
            return out
        usable = free - CPU_SLOT_RESERVE
        out["blocks_usable_for_an_offer"] = int(usable)
        if usable < 1:
            out["state"] = "tiny"
            out["reason"] = (f"the arena has {free} free CPU block(s) and "
                             f"{CPU_SLOT_RESERVE} are reserved, leaving {usable} for a "
                             "disk offer: one span needs at least one destination "
                             "slot, so every offer dies at `no_cpu_slots`. Check "
                             "KV_OFFLOAD_SIZE / VLLM_USE_SIMPLE_KV_OFFLOAD: a tier "
                             "whose arena clamps to a slot or two per rank cannot be "
                             "serviced from disk at any pool size")
            return out
        out["state"] = "ok"
        out["reason"] = (f"the arena has {free} free of "
                         f"{total if total is not None else '?'} CPU block(s), "
                         f"{usable} usable once the "
                         f"{CPU_SLOT_RESERVE}-block reserve is held: an offer can be "
                         "given a destination slot")
        return out
    except Exception as exc:  # noqa: BLE001 - a verdict is never worth a fault
        snapshot._error(exc, "disk_restore arena verdict")
        return {"state": "unreadable", "pool_total_blocks": None,
                "free_blocks": None, "cpu_slot_reserve": CPU_SLOT_RESERVE,
                "blocks_usable_for_an_offer": None,
                "reason": f"reading the CPU block pool raised "
                          f"{type(exc).__name__}"}


def gate() -> dict:
    """May THIS process make/serve a disk restore? Configured AND armed AND armed.

    Three switches, no new one: the root (`VLLM_E2_TIER_DISK_ROOT`, which
    `disk_spill` owns and validates), the spill tier's arm file (a LATCH since 0.2.35 -
    no expiry, no renewal, nothing that lowers it but a decision), and the KV tier's own
    arm. A fresh boot answers `disarmed` and the wrappers cost an env read. Reading the
    arm file is `disk_spill`'s cached verdict, so this is not a per-step `open()`.

    SAY IT PLAINLY, BECAUSE THE NAME SUGGES A WRITE KNOB AND THIS IS A SERVE DOOR: this
    gate means DISARMING THE SPILL TIER STOPS SERVING FROM DISK. It is not a
    write-only switch, and an operator who disarms to "stop the writes" also closes
    every restore - which is the intended coupling (the release's whole load-failure
    story is that the ceiling eviction which sheds bytes runs from the spill writer, so
    an arm that could lapse would freeze eviction too).
    """
    root = disk_spill.disk_root()
    if root is None:
        return _fail(f"{config.ENV_DISK_ROOT} is not set: there is no store a "
                     "restored file could be read from",
                     reason_kind="not_configured")
    if not disk_spill.is_armed():
        return _fail("the disk tier is disarmed: the restore path obeys the spill "
                     "tier's arm, and the arm is a LATCH (0.2.35) - it does not "
                     "expire, so it stays down until someone POSTs "
                     "{\"op\":\"arm\"}. Disarming stops serving from disk as "
                     "well as writing to it, by design",
                     reason_kind="disarmed")
    try:
        from . import backport  # noqa: PLC0415 - lazy: backport imports us lazily
        tier_armed = bool(backport.is_armed())
    except Exception as exc:  # noqa: BLE001 - an unreadable tier is a closed gate
        snapshot._error(exc, "disk_restore tier arm")
        return _fail(f"the tier's arm could not be read: {type(exc).__name__}",
                     reason_kind="tier_disarmed")
    if not tier_armed:
        return _fail("the KV tier itself is disarmed, so the connector schedules "
                     "no load for a disk offer to ride on",
                     reason_kind="tier_disarmed")
    if not (_state["disk_authority"] or _state["offers_enabled"]):
        # The boot default, and the ONLY thing that keeps this gate shut on an armed
        # box. Reaching here means the spill tier is armed AND the RAM tier is armed,
        # which is a state a *different* experiment puts up - and on 2026-10-05 that
        # state was enough to put an un-requested 160,000-token disk offer in front of
        # live traffic 123,474 times. Which posture serves, and whether any does, is
        # decided per boot - by the dev verb, or by the boot default that calls the same
        # setter (`disk_boot.py`) when `KVTIER_DISK_BOOT_ENABLE` is at its default - and
        # by nothing else.
        # Both ops are spelled inside the first 160 characters on purpose: `_fail`
        # clips a reason at MAX_STRING_CHARS, and a sentence whose second half is
        # clipped away stops being the answer to "how do I turn this on?".
        return _fail("neither boot switch is up: {\"op\":\"authority\",\"enabled\":"
                     "true} serves every request from SSD; {\"op\":\"offers\","
                     "\"enabled\":true} fills only what host RAM left short",
                     reason_kind="offers_disabled")
    proof = worker_proof()
    if not proof["ok"]:
        kind = proof.get("reason_kind")
        if kind == "scheduler_armed_workers_disarmed":
            return _fail(proof["reason"],
                         reason_kind="scheduler_armed_workers_disarmed",
                         ranks_armed=f"{proof.get('ranks_armed', 0)}/"
                                     f"{proof.get('ranks_seen', 0)}")
        if kind == "tier_arm_disarmed":
            # 0.2.28: the proof read the ARM FILE and it says no (absent, unreadable,
            # or fenced to another boot - never "expired", since 0.2.35 took the timer
            # out). The gate counts this under the same `disarmed` refusal the arm
            # check above uses - the authority is the file either way, and the reason
            # says which read got there first.
            return _fail(proof["reason"], reason_kind="disarmed")
        return _fail(proof["reason"], reason_kind="no_worker_proof")
    return _ok(root=root, ranks=proof["ranks"])


_proof_cache: dict = {"mono": None, "result": None}


def worker_proof(force: bool = False) -> dict:
    """Has a WORKER process attested that its restore layer is fitted?

    THE SCHEDULER CANNOT SEE THE WORKER. They are separate processes, and an offer is
    only safe if a worker will fill the CPU slots the offer allocates: if the
    scheduler half installed and the worker half did not (a fork rename of one worker
    method, the plugin loaded in one process only), the fork's own `start_load_kv`
    would DMA slots nobody had filled and the request would be served from STALE KV -
    the one failure mode of this design that is silent. So the offer requires proof.

    The proof is TWO things (0.2.28): the rank files the workers ALREADY write -
    `worker_diag` publishes this module's `disk_restore` section in `rank-*.json`,
    so a file that is FRESH and says `installed: YES` and `worker_fitted: true` is
    a worker attesting that its layer is live - and the spill tier's ARM FILE,
    read live (`_tier_arm_verdict`), because the tier arm is a rank-agnostic boot
    fact that a cadence-written mirror cannot be the authority for. It adds no new
    precondition (that directory is already the only way this process can know what
    exists on disk) and it is not a handshake: no socket, no new file name, no engine
    channel. Cached for `PROOF_CACHE_SECONDS`; bounded reads; never raises.
    """
    now = time.monotonic()
    if not force and _proof_cache["result"] is not None \
            and isinstance(_proof_cache["mono"], (int, float)) \
            and (now - _proof_cache["mono"]) < PROOF_CACHE_SECONDS:
        return _proof_cache["result"]
    result = _read_worker_proof()
    _proof_cache["mono"] = now
    _proof_cache["result"] = result
    _state["ranks_proven"] = int(result["count"])
    _state["ranks_seen"] = int(result.get("ranks_seen") or 0)
    return result


def _may_restore_worker() -> dict:
    """The worker's view of the same gate (there is no scheduler in this process)."""
    root = disk_spill.disk_root()
    if root is None:
        return _fail(f"{config.ENV_DISK_ROOT} is not set",
                     reason_kind="not_configured")
    if not disk_spill.is_armed():
        return _fail("the disk tier is disarmed", reason_kind="disarmed")
    return _ok(root=root)


def worker_service_verdict() -> dict:
    """Will THIS process, as a worker, service a disk load? The rank file's answer.

    This is deliberately NOT `gate()`. `gate()` is the SCHEDULER's question (is the
    tier armed here, do the ranks prove a worker?) and a worker process fails it for
    reasons that have nothing to do with filling a slot - the RAM tier's arm is
    per-process and the arm verb reaches the EngineCore, so a worker that would serve
    every read perfectly would answer `armed:false` forever, and a worker would be
    judging itself against files it did not write. What `bind()` checks before it
    touches a slot is `_may_restore_worker()` plus this process having fitted the
    layer, so that is what a rank file reports.

    On 2026-10-05 the scheduler offered 160,000 tokens 123,474 times to workers that
    would not bind. The rank file was the only place that could have said so, and the
    proof read looked at `installed` and ignored `armed`.
    """
    verdict = _may_restore_worker()
    if not verdict["ok"]:
        return verdict
    if not _state["worker_fitted"]:
        return _fail("the restore layer is not fitted in this worker process, so "
                     "nothing would bind the paths an offer allocates slots for",
                     reason_kind="worker_half_not_fitted")
    return _ok(root=verdict["root"])


def _rank_failure_records(raw) -> list:
    """Validate the `last_failures` one rank file carried. Defensive, bounded, safe.

    The field arrives from ANOTHER PROCESS, so it is treated as untrusted input of the
    same kind the reader already treats `reads_applied` as: every string clipped, every
    number checked with `_int`, the list capped, and anything not shaped like a record
    dropped. A hostile or half-written file must not be able to put a fake reason in
    front of the operator, and must not be able to make this reader raise.
    """
    out: list = []
    try:
        if isinstance(raw, (str, bytes, dict)) or raw is None:
            return out
        for item in list(raw)[:MAX_FAILURES_PER_RANK_FILE]:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            row = {"kind": (_failure_text(kind, 48)
                            if kind in WORKER_FAILURES else "read_failed"),
                   "kind_as_reported": _failure_text(kind, 48),
                   "where": _failure_text(item.get("where"), 24),
                   "request_id": _failure_text(item.get("request_id"), 48),
                   "span": _failure_text(item.get("span"), 80),
                   "group": _failure_int(item.get("group")),
                   "contract": _failure_text(item.get("contract"), 24),
                   "key16": _failure_text(item.get("key16"), 64),
                   "gpu_block_id": _failure_int(item.get("gpu_block_id")),
                   "job_id": _failure_int(item.get("job_id")),
                   "bytes_needed": _failure_int(item.get("bytes_needed")),
                   "jobs_needed": _failure_int(item.get("jobs_needed")),
                   "jobs_fit": _failure_int(item.get("jobs_fit")),
                   "pool_budget_bytes": _failure_int(item.get("pool_budget_bytes")),
                   "pool_committed_bytes": _failure_int(
                       item.get("pool_committed_bytes")),
                   "reason": _failure_text(item.get("reason"), 200),
                   "ts": (_failure_int(item.get("ts")) or 0),
                   "seq": _failure_int(item.get("seq")),
                   "rank": _failure_text(item.get("rank"), 32),
                   "source": "rank file",
                   "block_provenance": None}
            out.append(row)
    except Exception as exc:  # noqa: BLE001 - one bad field is not evidence
        snapshot._error(exc, "disk_restore rank failure records")
        return out[:MAX_FAILURES_PER_RANK_FILE]
    return out


def _rank_file_proves_restore(path) -> dict | None:
    """What one rank file says about ITS worker: a verdict dict, or None.

    None means "this file is not evidence of anything" (unreadable, too big, no
    stamp, stale, wrong shape) - it neither proves nor vetoes. A readable file always
    yields a verdict with the PER-RANK facts, because "fitted but will not serve" is
    a DIFFERENT and much more important answer than "nothing here": a rank that is
    fitted and cannot serve passes every disk entry through as though it had none,
    and the blocks the fork already committed to stay stale.

    WHICH FACTS ARE PER-RANK (0.2.28). The file's `installed` and `worker_fitted`
    and its `reads_applied` are the only facts a rank file uniquely knows; the
    TIER ARM is not one of them. The arm is a rank-agnostic boot fact owned by the
    EngineCore process that runs the scheduler and reaches the workers as
    `disk-spill-arm.json` - the very file `_read_worker_proof` now reads directly.
    The rank file's own `armed` verdict is kept for reporting and for ONE veto it
    can still legitimately carry: `service_reason_kind: "not_configured"` (this
    worker's own `VLLM_E2_TIER_DISK_ROOT` is unset - a per-process env fact no
    arm file can see). It does NOT veto for `disarmed`: a worker's arm verdict is
    its own cached read of the same file the scheduler is reading live, and a rank
    file is written on cadence - `armed: false` in a rank file is more often
    YESTERDAY's truth than this minute's. That staleness was the whole of the
    `workers_not_servicing` lockout reproduced live on 2026-10-20: 0/4 ranks
    "armed" for 120 s after an arm with no spill activity, then 4/4 within seconds
    of a spill burst, because `_signature` never hashed the armed verdict.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read(MAX_RANK_FILE_BYTES + 1)
        if len(raw) > MAX_RANK_FILE_BYTES:
            return None  # unreadable by the reader too, so not evidence of anything
        document = json.loads(raw)
        if not isinstance(document, dict):
            return None
        stamp = document.get("ts")
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
            return None
        age = time.time() - float(stamp)
        if age < 0.0:
            # A CLOCK AHEAD OF OURS is not freshness: a file stamped an hour in the
            # future would sail past the staleness test forever and vouch for a worker
            # that died with it. Rank files live on a directory shared with another
            # host's clock, so the skew is real and has to be one-sided.
            return None
        if age > RANK_FILE_FRESH_SECONDS:
            return None  # a heartbeat from a worker that may be gone
        section = document.get("disk_restore")
        if not isinstance(section, dict):
            return None
        rank = document.get("rank")
        name = str(rank)[:32] if rank is not None else str(path.name)[:64]
        installed = section.get("installed") is True
        armed = section.get("armed") is True
        # THE PER-RANK SERVICE FACTS, three-valued on purpose: `None` means this
        # file predates the field, and an absent field is NOT evidence of a fault
        # (a file from a mixed-version fleet that never measured it must not veto a
        # worker for a fact it never refused). `False` is a worker's own NO.
        fitted = section.get("worker_fitted")
        fitted = fitted if isinstance(fitted, bool) else None
        kind = section.get("service_reason_kind")
        kind = kind if isinstance(kind, str) else None
        # `reads_applied` is a worker's own counter, published for the honesty check.
        # It arrives from another process, so it is clamped before it is summed: a
        # hostile or broken writer must not be able to put a fake total in front of
        # the witness. `disk_store` has a `_bounded_int` with a floor of 1 (it
        # validates operator knobs); this one wants 0-as-valid, so it uses `_int`.
        reads = _int(section.get("reads_applied"))
        if reads is None or reads < 0:
            reads = 0
        # WHAT ONE READ COSTS THIS RANK (0.2.31): its own `total_bytes_per_file`, the
        # figure the offer's bounce-buffer cap is priced with. Clamped like every other
        # number read out of another process, and None when the file predates the field
        # - which is the mixed-fleet case the fallback price exists for.
        width = _int(section.get("bytes_per_file"))
        if width is None or width < 0:
            width = None
        return {"rank": name, "installed": installed,
                "worker_fitted": fitted, "service_reason_kind": kind,
                "armed": installed and armed,
                "reads_applied": min(reads, MAX_RANK_COUNTER),
                "bytes_per_file": width,
                "last_failures": _rank_failure_records(
                    section.get("last_failures")),
                "reason": _clip(section.get("reason"), 160)}
    except Exception as exc:  # noqa: BLE001 - one bad file is not evidence, that's all
        snapshot._error(exc, "disk_restore rank file")
        return None


def _tier_arm_verdict() -> tuple:
    """(armed, reason) from the spill tier's ARM FILE, read LIVE. Never raises.

    THE AUTHORITATIVE SIDE OF THE ARM (0.2.28). The arm is issued by `arm()` in the
    EngineCore - the very process that runs the scheduler and calls this reader -
    and crosses to the workers as `disk-spill-arm.json`. Asking RANK FILES whether
    the tier is armed asked four lagged mirrors of one fact each: they are rewritten
    on cadence (`worker_write_interval_steps`, default 50 steps) or on a signature
    change that did not include the armed verdict, so an arm transition with no
    spill activity was invisible to the proof until the next spill burst, and an
    idle box wrote nothing at all. Reading the arm file through `disk_spill`'s own
    `_read_arm_file` keeps the boot-fence semantics EXACTLY as they are (a reboot
    must not arm a cold box: the epoch/boot-id fences and every fail-closed shape
    are `_read_arm_file`'s, untouched), and it is one bounded read behind the
    `worker_proof` cache's own 1 s floor.
    """
    try:
        return disk_spill._read_arm_file(disk_spill._arm_path())
    except Exception as exc:  # noqa: BLE001 - an unreadable arm is no arm at all
        snapshot._error(exc, "disk_restore tier arm read")
        return False, f"the arm file could not be read: {type(exc).__name__}"


def _read_worker_proof() -> dict:
    """Read the rank files NOW (no cache). See `worker_proof`.

    TWO SOURCES, each asked only what it uniquely knows (0.2.28). THE TIER ARM
    comes from the arm file, read live through `disk_spill._read_arm_file` (the
    authoritative copy, boot-fenced, written by this very process when the operator
    armed it - see `_tier_arm_verdict` for why the rank files stopped being asked).
    The PER-RANK facts - `installed`, `worker_fitted`, `not_configured`,
    `reads_applied` - come from the rank files, which are the only place they live.

    The verdict is still AND over the ranks that are live. A rank that is not fitted
    VETOES the offer even when other ranks are ready: the fork's load event spans
    every rank, so the ready ones cannot make the refusing one's blocks valid - that
    is precisely the shape of the 2026-10-05 incident (4 ranks, 0 of them servicing,
    123,474 offers). What is no longer a veto is a rank file whose only refusal is
    its own lagged `armed: false`: when the arm file (the fact's authority, read
    live) says armed, that file is reporting yesterday. The proof counts those as
    `stale_arm_claims` and says so, rather than locking the tier out for the life of
    an idle spell - the `workers_not_servicing` lockout reproduced live on 2026-10-20
    (0/4 ranks "armed" for 120 s after an arm, then 4/4 seconds after a spill burst,
    because a rank file is rewritten on cadence and its signature never hashed the
    armed verdict).
    """
    directory = config.diagnostics_dir()
    if directory is None:
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": 0, "reads_applied": 0, "tier_armed": False,
                "stale_arm_claims": 0, "arm_file_reason": None,
                "reason_kind": "no_worker_proof",
                "reason": f"{config.ENV_DIAG_DIR} is not set, so no worker's rank "
                          "file can be read to prove the worker half of this path "
                          "is fitted"}
    try:
        names = sorted(name for name in os.listdir(directory)
                       if name.startswith(RANK_FILE_PREFIX)
                       and name.endswith(RANK_FILE_SUFFIX))
    except OSError as exc:
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": 0, "reads_applied": 0, "tier_armed": False,
                "stale_arm_claims": 0, "arm_file_reason": None,
                "reason_kind": "no_worker_proof",
                "reason": "the diagnostics directory could not be listed: "
                          f"{type(exc).__name__}"}
    if not names:
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": 0, "reads_applied": 0, "tier_armed": False,
                "stale_arm_claims": 0, "arm_file_reason": None,
                "reason_kind": "no_worker_proof",
                "reason": "no worker has written a rank file yet, so nothing proves "
                          "the worker half of this path is fitted"}
    tier_armed, arm_reason = _tier_arm_verdict()
    armed, silent = [], []
    stale_claims = 0
    reads = 0
    # THE TWO FACTS THE OFFER'S PRICE AND THE OPERATOR'S DIAGNOSIS READ OFF THE SAME
    # READ (0.2.31). `widths` is every live rank's own `bytes_per_file`, and the offer
    # cap uses the WIDEST of them: an offer spans every rank, the bounce-buffer pool is
    # per process, and pricing the widest slot is the only cap that cannot be
    # optimistic for somebody. `failures` is the load-failure records the files carried,
    # the transport `reads_applied` already uses, because the reasons live in the worker
    # and the operator reads the scheduler.
    widths: list = []
    failures: list = []
    probed = names[:MAX_RANK_FILES_PROBED]
    for name in probed:
        said = _rank_file_proves_restore(directory / name)
        if said is None:
            continue
        width = said.get("bytes_per_file")
        if isinstance(width, int) and not isinstance(width, bool) and width > 0:
            widths.append(int(width))
        for record in said.get("last_failures") or []:
            if len(failures) >= MAX_RANK_FILES_PROBED * MAX_FAILURES_PER_RANK_FILE:
                break
            row = dict(record)
            row["rank"] = said["rank"]
            failures.append(row)
        if not said["installed"]:
            silent.append(f"{said['rank']}: not fitted")
            continue
        if said["worker_fitted"] is False:
            silent.append(f"{said['rank']}: restore layer not fitted in this "
                          "worker process")
            continue
        if said["service_reason_kind"] == "not_configured":
            # A per-WORKER env fact no arm file can see: this rank has no disk
            # root, so it cannot name a path, so it cannot bind. A real veto.
            silent.append(f"{said['rank']}: no disk root configured in this worker")
            continue
        if not tier_armed:
            silent.append(f"{said['rank']}: the tier arm file says "
                          f"{arm_reason}")
            continue
        if not said["armed"]:
            # The file's OWN armed verdict is false while the arm file - the
            # authority for this rank-agnostic fact, read live - says armed. The
            # file was written before the arm (or before its writer's cache
            # turned); it is stale evidence, not a refusal.
            stale_claims += 1
        reads += said["reads_applied"] or 0
        armed.append(said["rank"])
    _state["proof_reads"] += len(probed)
    armed.sort()
    seen = len(armed) + len(silent)
    _state["ranks_seen"] = seen
    _state["ranks_reads_applied"] = reads
    _state["ranks_load_failures"] = len(failures)
    _state["rank_failure_records"] = sorted(
        failures, key=lambda item: (item.get("ts") or 0, item.get("seq") or 0),
        reverse=True)[:MAX_FAILURES_PUBLISHED]
    widest = max(widths) if widths else None
    _state["ranks_bytes_per_file_max"] = widest
    if not tier_armed:
        # THE ARM FILE OUTRANKS EVERY RANK FILE (0.2.28). No rank file can arm a
        # disarmed tier by claiming `armed: true` - the arm carries the boot fence
        # (and, until 0.2.35, an expiry), and a boot that never armed must not offer
        # whatever the rank files remember from the last armed window. The refusal
        # reason is the arm file's own, which is what keeps the boot fence honest.
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": seen, "reads_applied": reads,
                "tier_armed": False, "stale_arm_claims": stale_claims,
                "arm_file_reason": arm_reason,
                "reason_kind": "tier_arm_disarmed",
                "reason": f"the disk tier's arm file does not arm this boot "
                          f"({arm_reason}): the arm is a boot fact read from its "
                          "authoritative copy, and rank files cannot substitute "
                          "for it (a reboot must not arm a cold box)"}
    if armed and not silent:
        return {"ok": True, "count": len(armed), "ranks": armed[:8],
                "ranks_armed": len(armed), "ranks_seen": seen,
                "reads_applied": reads, "tier_armed": True,
                "bytes_per_file_max": widest,
                "bytes_per_file_ranks": len(widths),
                "load_failures": list(_state["rank_failure_records"]),
                "stale_arm_claims": stale_claims, "arm_file_reason": None,
                "reason_kind": None, "reason": None}
    if silent and armed:
        return {"ok": False, "count": len(armed), "ranks": armed[:8],
                "ranks_armed": len(armed), "ranks_seen": seen,
                "reads_applied": reads, "tier_armed": True,
                "bytes_per_file_max": widest,
                "bytes_per_file_ranks": len(widths),
                "load_failures": list(_state["rank_failure_records"]),
                "stale_arm_claims": stale_claims, "arm_file_reason": None,
                "reason_kind": "scheduler_armed_workers_disarmed",
                "reason": f"{len(armed)} of {seen} live rank file(s) serve disk and "
                          f"{len(silent)} do not ({'; '.join(silent)[:160]}): the "
                          "load event spans every rank, so the ones that serve would "
                          "still leave blocks the fork committed to "
                          f"(ranks_armed={len(armed)}/{seen})"}
    if silent:
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": seen, "reads_applied": reads,
                "tier_armed": True, "stale_arm_claims": stale_claims,
                "arm_file_reason": None,
                "reason_kind": "scheduler_armed_workers_disarmed",
                "reason": f"the scheduler is armed and {len(silent)} live rank "
                          f"file(s) refuse disk service "
                          f"({'; '.join(silent)[:160]}): an offer would allocate CPU "
                          "slots nothing fills, the fork would report the load failed "
                          "and reschedule the request - ranks_armed="
                          f"0/{seen}, the exact shape of the 2026-10-05 loop"}
    return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
            "ranks_seen": 0, "reads_applied": reads, "tier_armed": True,
            "stale_arm_claims": stale_claims, "arm_file_reason": None,
            "reason_kind": "no_worker_proof",
            "reason": f"{len(probed)} rank file(s) were read and none reports a "
                      "fitted disk restore layer (disk_restore.installed) - so the "
                      "worker half of this path is not live, and offering disk "
                      "tokens would DMA slots nobody filled"}


# --- the identity: which file holds the next span? ----------------------------

def whole_block_tokens(matched, block_size) -> dict:
    """`floor(matched / block_size) * block_size`, and why a partial block is 0.

    THE OFFER ARITHMETIC IS A SEPARATE DOOR because it is the rule the live box
    measured: with a 3,200-token scheduler block, 3,199 matched tokens must offer
    NOTHING (a sub-block offer is discarded by the scheduler while parking the
    request), 3,200 offers 3,200, and 6,500 offers 3,200. Exposing the arithmetic
    means a test can pin it without a fake engine, and a reader can find the rule
    by name instead of in the middle of a loop.
    """
    if not _plain_int(block_size) or block_size <= 0:
        return _fail(f"block_size must be a positive int, got {block_size!r} "
                     "(unit: tokens)")
    matched_tokens = _int(matched)
    if matched_tokens is None or matched_tokens < 0:
        return _fail(f"matched must be a non-negative int, got {matched!r} "
                     "(unit: tokens)")
    blocks = matched_tokens // int(block_size)
    return _ok(blocks=blocks,
               tokens=blocks * int(block_size),
               discarded_tokens=matched_tokens - blocks * int(block_size),
               block_size=int(block_size))


def offer_block_ceiling(block_size=None) -> dict:
    """THE bound one offer may carry, as the blocks that define it and the tokens they
    are worth (0.2.32).

    WHY THIS EXISTS AT ALL: the field it replaces, `offer_cap`, advertised 32,000
    tokens - the fixed cap deleted in this release - and once the cap is gone a field
    with that name and that number would be reporting a bound that constrains nothing.
    So this publishes the bound that DOES constrain: `MAX_RESTORE_BLOCKS_PER_OFFER`
    scheduler blocks, which `_disk_run`'s positional walk stops at, because one offer
    can only name that many spans and stream them inside `RESTORE_DEADLINE_SECONDS`.
    The token figure is derived from the engine's own block size and carries the source
    of that block size, because a payload read in a process that has made no offer (a
    worker rank) cannot honestly claim to know the engine's block size yet.

    Read-only, never raises, and derives rather than stores: the block size is the
    engine's fact and the payload must agree with the next offer, not with the last one.
    """
    size = _int(block_size)
    size_source = "engine"
    if not size or size <= 0:
        size = _int(_state.get("offer_block_size_seen"))
        size_source = "last-lookup"
    if not size or size <= 0:
        size = DEFAULT_OFFER_BLOCK_SIZE_TOKENS
        size_source = "fallback"
    return {"ok": True,
            "blocks": MAX_RESTORE_BLOCKS_PER_OFFER,
            "tokens": MAX_RESTORE_BLOCKS_PER_OFFER * int(size),
            "block_size": int(size),
            "block_size_source": size_source,
            # THE HONEST READING, in the payload rather than only in a comment: this
            # bound is not what a long prefix hits. Pool affordance
            # (`offer_pool.jobs_the_pool_can_carry`) is the ceiling that binds in
            # practice, and on the live store the depth the spill tier reached (~11
            # spans, because spill follows RAM-tier parking) binds earlier still.
            "note": ("a per-offer BLOCK bound, not a token cap: the deleted fixed "
                     "32,000-token cap is gone and nothing replaced it, so the bound "
                     "that binds in practice is bounce-buffer affordability "
                     "(`bounded_by` says which bound cut any given offer)")}


def _key_helpers() -> dict:
    """The engine's OWN two packers/view-builders, or a refusal.

    `make_block_hash_with_group_id` (`kv_cache_utils.py:74`) and
    `resolve_block_hashes` (`kv_cache_utils.py:2859`) are resolved from
    `sys.modules` exactly the way `backport._empty_metadata` resolves the metadata
    class: never imported here, and refused when absent. Using them is the whole
    safety argument for deriving keys at all - a plugin that re-implemented either
    would eventually disagree with the engine about which bytes are the group, and
    that disagreement is a file read under the wrong name.
    """
    module = sys.modules.get(KV_UTILS_MODULE)
    if module is None:
        return _fail(f"{KV_UTILS_MODULE} is not imported in this process, so the "
                     "group-qualified key cannot be built",
                     reason_kind="no_key_helpers")
    pack = getattr(module, "make_block_hash_with_group_id", None)
    resolve = getattr(module, "resolve_block_hashes", None)
    if not callable(pack) or not callable(resolve):
        return _fail(f"{KV_UTILS_MODULE} carries no callable "
                     "make_block_hash_with_group_id/resolve_block_hashes: this "
                     "fork renamed the key machinery, and guessing it is how a "
                     "file gets read under the wrong name",
                     reason_kind="no_key_helpers")
    return _ok(pack=pack, resolve=resolve)


def _group_serviceability(manager, block_size: int, sizes: list,
                          wanted: list) -> dict:
    """THE WHOLE GROUP ARITHMETIC, as data, for one payload read (0.2.25/0.2.27).

    Three predicates decide whether a model can be served from disk at all -
    `prefix_cacheable_group_ids`, `group_block_sizes` against the scheduler block
    size, and the per-group BLOCK CONTRACT (`_group_contract`, which reads the
    fork's `has_positionally_stable_blocks` and `mamba_cache_mode`) - and until now a
    reader had to re-derive all three from a live manager to answer "why does the
    disk offer nothing?". This publishes the derivation instead: which groups the RAM
    tier says are prefix-cacheable, which contract each one has, which are restorable,
    and WHICH BLOCKED THE OFFER AND WHY.

    Every group in the table gets a contract read, not just the cacheable ones, so the
    payload shows the shape of the model (a boundary-state group that is not
    cacheable is exactly the fact that distinguishes "nothing is hashable" from "the
    hashable ones are not cacheable"). Each predicate is read ONCE per group per
    call: an exception inside a contract read is an error line, and reading the same
    flag twice would report a failure that happened once as two.

    `complete` is the property the gate is built on: every prefix-cacheable group has
    a contract this tier implements. A pure-attention model satisfies it, and always
    did; a hybrid GDN + full-attention model satisfies it too now that the align-mode
    contract is implemented, and a model with a group that has NO implementable
    contract does not, and cannot be.
    """
    contracts: dict = {}
    for group in range(len(sizes)):
        contracts[group] = _group_contract(manager, group)
    positional = [group for group in contracts
                  if contracts[group]["contract"] == CONTRACT_POSITIONAL]
    boundary = [group for group in contracts
                if contracts[group]["contract"] == CONTRACT_BOUNDARY]
    restorable: list = []
    blocked: dict = {}
    kinds: dict = {}
    for group in wanted:
        contract = (contracts[group]["contract"] if group in contracts
                    else CONTRACT_OUTSIDE)
        if group < 0 or group >= len(sizes):
            blocked[group] = "outside the group table"
            kinds[group] = CONTRACT_OUTSIDE
        elif _int(sizes[group]) != block_size:
            blocked[group] = (f"block_size {sizes[group]} != the scheduler "
                              f"block_size {block_size}, so a slot for it would "
                              "land at the wrong GPU offset")
            kinds[group] = CONTRACT_GEOMETRY
        elif contract not in CONTRACTS_IMPLEMENTABLE:
            blocked[group] = contracts[group]["reason"]
            kinds[group] = contract
        else:
            restorable.append(group)
    unrestorable = [group for group in wanted
                    if group >= len(sizes) or contracts[group]["contract"]
                    not in CONTRACTS_IMPLEMENTABLE]
    by_group = {str(group): {
        "contract": contracts[group]["contract"] if group in contracts
        else CONTRACT_OUTSIDE,
        "has_positionally_stable_blocks": contracts[group]["stable"]
        if group in contracts else None,
        "mamba_cache_mode": contracts[group]["mamba_cache_mode"]
        if group in contracts else None}
        for group in list(contracts)[:16]}
    return _ok(prefix_cacheable_group_ids=list(wanted)[:16],
               group_contracts=by_group,
               positionally_stable_group_ids=sorted(positional)[:16],
               boundary_state_group_ids=sorted(boundary)[:16],
               unrestorable_cacheable_group_ids=sorted(set(unrestorable))[:16],
               restorable_group_ids=restorable[:16],
               blocked_group_ids=sorted(blocked)[:16],
               blocked_kinds={str(group): kinds[group]
                              for group in sorted(blocked)[:16]},
               blocked_why={str(group): _clip(text, 200)
                            for group, text in sorted(blocked.items())[:8]},
               scheduler_block_size=int(block_size),
               group_block_sizes=[_int(item) for item in sizes[:16]],
               complete=not blocked,
               disk_offers_possible=bool(restorable) and not blocked)


def _groups_required(serviceability: dict, blocks: int) -> dict:
    """What a claim of `blocks` scheduler blocks COSTS every prefix-cacheable group.

    THE INSTRUMENT THAT WOULD HAVE FOUND THE 00:08 INCIDENT BY ITSELF (item 7 of that
    investigation, and the reason it is permanent): "the claimed 32,000 tokens need 10
    blocks in g5 and 1 real boundary-state block in each of g0-g3, and this restore
    filled g5 only" is ONE JSON object, readable without an agent investigation.

    THE ARITHMETIC IS THE ENGINE'S, NOT OURS. Every number here comes out of
    `group_serviceability` - which is `prefix_cacheable_group_ids`,
    `group_block_sizes`, the scheduler block size and `has_positionally_stable_blocks`
    - so this reads NO manager a second time and knows no model. The one contract that
    is not pure geometry is the align-mode state group's, and it is the engine's own:
    `MambaManager.find_longest_cache_hit` (`single_type_kv_cache_manager.py:1533-1610`)
    answers a hit with `computed.extend([block_pool.null_block] * i); computed.append(
    cached)` - NULLS for every interior block and ONE real state block at the boundary,
    early-stopping on the last match (`:1578-1582`, `:1601-1607`). That is why the row
    for a non-positional group says `blocks_needed: 1, null_blocks: blocks - 1`: the
    claim needs one block from it, not a span of KV. A group whose row cannot be
    derived says `contract: "unknown"` and `blocks_needed: null` rather than a guess.

    Bounded: at most 16 rows, and `blocks` is capped by the caller at the offer's own
    ceiling, so the numbers in here are always smaller than the offer's.
    """
    blocks = max(1, min(_int(blocks) or 1, MAX_RESTORE_BLOCKS_PER_OFFER))
    if not isinstance(serviceability, dict):
        # A caller with no arithmetic to price (no lookup has reached the predicate yet)
        # gets an empty table, not an exception on the offer path: this is a published
        # field, and a missing field is better than a refused offer.
        return {"claimed_blocks": int(blocks), "claimed_tokens": 0, "rows": [],
                "unfilled_group_ids": [], "blocks_needed_total": 0,
                "blocks_filled_total": 0}
    block_size = _int(serviceability.get("scheduler_block_size")) or 0
    sizes = serviceability.get("group_block_sizes") or []
    restorable = set(serviceability.get("restorable_group_ids") or ())
    declared = serviceability.get("group_contracts") or {}
    kinds = serviceability.get("blocked_kinds") or {}
    rows: list = []
    unfilled: list = []
    for group in (serviceability.get("prefix_cacheable_group_ids") or [])[:16]:
        group_size = (sizes[group]
                      if 0 <= group < len(sizes) else None)
        entry = declared.get(str(group)) or {}
        kind = str(entry.get("contract") or kinds.get(str(group))
                   or CONTRACT_UNKNOWN)
        needed: int | None
        nulls: int
        real: int | None
        if kind == CONTRACT_POSITIONAL and group_size:
            # One real block per scheduler block of the run: the group's blocks may
            # be smaller than the scheduler's (a `geometry` group is priced the same
            # way because it would need the same number of them), and the claim
            # cannot be satisfied without all of them.
            needed = max(1, (blocks * block_size) // int(group_size))
            nulls = 0
            real = needed
        elif kind == CONTRACT_GEOMETRY and group_size:
            # Position-addressable in principle, at a block size this offer cannot
            # express: it still needs its blocks per the same arithmetic, and the
            # disk fills none of them.
            needed = max(1, (blocks * block_size) // int(group_size))
            nulls = 0
            real = needed
        elif kind == CONTRACT_BOUNDARY:
            # ONE real state block per claimed run, not a run of KV: the align-mode
            # contract is `[null_block] * (n - 1)` then the cached boundary state
            # (`single_type_kv_cache_manager.py:1591-1607`), which is why the row
            # says `blocks_needed: 1, null_blocks: blocks - 1`.
            needed = 1
            nulls = max(0, blocks - 1)
            real = 1
        else:
            needed = None
            nulls = 0
            real = None
        filled = needed if (group in restorable and needed) else 0
        if not filled:
            unfilled.append(group)
        rows.append({"group": int(group), "block_size": group_size,
                     "contract": _clip(kind, 24),
                     "blocks_needed": needed, "null_blocks": nulls,
                     "real_blocks": real,
                     "restored": bool(group in restorable),
                     "blocks_filled": int(filled)})
    return {"claimed_blocks": int(blocks),
            "claimed_tokens": int(blocks * block_size),
            "rows": rows,
            "unfilled_group_ids": unfilled[:16],
            "blocks_needed_total": sum(int(row["blocks_needed"] or 0)
                                       for row in rows),
            "blocks_filled_total": sum(int(row["blocks_filled"] or 0)
                                       for row in rows)}


def _eligible_groups(manager, block_size: int,
                     claim_blocks: int | None = None) -> dict:
    """The groups a disk span can be restored for - and the veto that needs ALL of them.

    A group qualifies when it is prefix-cacheable AND its block size IS the
    scheduler block size AND it is positionally stable. All three halves are
    load-bearing. Prefix-cacheable, because `update_state_after_alloc`
    (`manager.py:440-446`) discards every other group's CPU hit before pairing, so a
    slot offered for a scratch group is a slot the fork never pairs and never frees.
    Equal to the scheduler block size, because the merged tuple's entries are consumed
    in ORDER, one whole group block each: a group whose blocks are smaller than the
    scheduler block would put our slots at the wrong GPU offsets while the offer
    arithmetic stayed a whole number of scheduler blocks. Positionally stable, because
    a spilled file is named after a hash derived from the request's tokens at a fixed
    position, so a group whose blocks have no positional name has nothing to read
    back from (`_group_positionally_stable`).

    THEN THE GATE, WHICH IS THE POINT OF THIS FUNCTION (0.2.25). Qualifying was
    treated as sufficient when it is only necessary. The number this function's
    output feeds is the one `get_num_new_matched_tokens` returns to the scheduler
    (`manager.py:372`), and that return advances `num_computed_tokens` for the WHOLE
    model - the engine skips prefill of every group across the claimed tokens, not
    just the ones we can name. `update_state_after_alloc` then builds load entries
    only for the groups with non-null CPU blocks (`manager.py:452-459`), so a group
    outside the eligible set is CLAIMED AND NEVER LOADED. A span held by a full-
    attention group while a recurrent group is not restorable is therefore not a
    partial restore: it is a correct read of some layers feeding a model whose other
    layers are being asked to continue from whatever their slots happened to hold,
    and at `temperature=0` it answers confidently and wrongly.

    So a non-empty eligible set is not enough: the offer must be COMPLETE over the
    groups the RAM tier itself considers prefix-cacheable, or it must claim nothing.
    On the live hybrid box (`Qwen3.8-Flash-Next-W4A16-Merlin`, groups 0-3 GDN in
    align mode, group 4 a non-cacheable QSA ring, group 5 full attention, cacheable
    `[0,1,2,3,5]`) that means the disk tier offers NOTHING, forever, on this model -
    which is the correct answer and is stated as one in `incomplete_group_set`, not
    hedged. A pure-attention model has nothing blocked, so nothing changes for it.

    THIS MODULE ALREADY KNEW THE RULE. `_disk_run`'s docstring says it plainly: "A
    block is restorable only when EVERY eligible group has a file for it: the fork
    pairs per group, so one missing group's slot would be a load entry with no bytes
    behind it - a whole-request recompute for a partially-restored span." The bug was
    never the rule; it was the SET the rule was applied to. With the eligible set
    narrowed to `{5}` first, that safeguard is VACUOUS - there is only one group left
    to check, and it always has a file. This gate is the same invariant applied to the
    set the CLAIM actually covers.

    AND IT IS THE RAM TIER'S RULE TOO. The live lookup ledger's verdicts say the joint
    CPU lookup is clamped by exactly these groups: 190 of 256 lookups returned a hit
    and only 4 were adopted, with the typical record `local=134400 hit=1000 adopted=0`
    attributed `partial_hit_clamp_suspected` - "a state group clamped at a length this
    lookup cannot see". The RAM path will not claim more than every group can supply;
    the disk path claimed 28,800 tokens because it asked about one group. This gate is
    the disk path agreeing with the RAM path, not a new restriction on the box.

    `claim_blocks` is the number of scheduler blocks the caller was about to offer, so
    the refusal can publish what those tokens WOULD have required of every group
    (`_groups_required`). It is an upper bound the caller clips to the offer cap; None
    means a caller that is only asking the predicate, which gets the arithmetic with a
    one-block claim.
    """
    try:
        sizes = list(getattr(manager, "group_block_sizes", ()) or ())
    except Exception as exc:  # noqa: BLE001 - an unreadable layout is a refusal
        snapshot._error(exc, "disk_restore group_block_sizes")
        return _fail(f"group_block_sizes could not be read: {type(exc).__name__}",
                     reason_kind="bad_call")
    cacheable = getattr(manager, "prefix_cacheable_group_ids", None)
    if not isinstance(cacheable, (list, tuple, set, frozenset)):
        return _fail("prefix_cacheable_group_ids is not a collection, so no group "
                     "can be proven restorable", reason_kind="no_eligible_group")
    try:
        wanted = sorted({int(item) for item in cacheable})
    except Exception as exc:  # noqa: BLE001 - a hostile id list is a refusal
        snapshot._error(exc, "disk_restore prefix_cacheable_group_ids")
        return _fail(f"prefix_cacheable_group_ids is unreadable: "
                     f"{type(exc).__name__}", reason_kind="no_eligible_group")
    serviceability = _group_serviceability(manager, block_size, sizes, wanted)
    # THE ARITHMETIC, PUBLISHED. Written on every lookup that gets this far, offered
    # or refused, so the payload describes the LAST LOOKUP'S model rather than the
    # last successful one. A manager whose inputs are unreadable never reaches this
    # line and leaves the previous read standing.
    _state["group_serviceability"] = serviceability
    groups = serviceability["restorable_group_ids"]
    declared = serviceability.get("group_contracts") or {}
    contracts = {str(group): str((declared.get(str(group)) or {})
                                 .get("contract") or CONTRACT_UNKNOWN)
                 for group in groups}
    if not groups:
        return _fail(f"no prefix-cacheable group has block_size == the scheduler "
                     f"block_size {block_size} under a contract this tier can "
                     f"restore (group sizes are {sizes[:8]}, cacheable "
                     f"{wanted[:8]}): the merged CPU-hit tuple cannot be extended "
                     "one whole block at a time for this geometry",
                     reason_kind="no_eligible_group",
                     group_block_sizes=sizes[:8],
                     prefix_cacheable_group_ids=wanted[:8])
    blocked = serviceability["blocked_group_ids"]
    if blocked:
        # THE GATE. Named, countable, and self-explanatory: the blocked ids come FIRST
        # in the sentence because `last_refusal` is clipped, and the structured
        # arithmetic goes out as extras so a caller that wants the lists does not read
        # them out of prose. Blocked groups are named BEFORE the restorable ones for
        # the same reason.
        #
        # DO NOT "OPTIMISE" THIS INTO A CLAMP. The tempting change is to offer the
        # intersection instead - clamp the span down to what the joint group set can
        # supply, the way the CPU lookup does. It is the same nothing, twice paid: the
        # live box's clamped external hits are 1,000-3,000 tokens, which is below the
        # point where a disk read earns anything (about 0.3 s of prefill saved against
        # ~20 ms of read plus the whole claim risk), and it is still a claim whose
        # align-mode boundary state nobody produced - `update_state_after_alloc`'s
        # `n_ext_g == 0 -> continue` is a SILENT no-op, and the only lever back to
        # recompute is someone reporting a load error for a group that was never in the
        # tuple, which nobody does. Clamp-to-nothing and refuse-because-nothing end up
        # in the same place; only the refusal says why in the witness. The per-group
        # requirement table below is what a future reader needs to re-derive that.
        required = _groups_required(serviceability, claim_blocks or 1)
        required["offered"] = False
        required["refused_as"] = "incomplete_group_set"
        _state["last_groups_required"] = required
        kinds = set((serviceability.get("blocked_kinds") or {}).values())
        why_bits = []
        if CONTRACT_UNSTABLE in kinds:
            # KEPT SHORT ON PURPOSE: the whole sentence is clipped at 160 characters
            # and this clause carries the mechanism, so it has to fit before the cut.
            # The long form lives in `blocked_why` and in `OFFER_REFUSAL_WHY`.
            why_bits.append("no contract this tier implements: not scannable and "
                            "not an align-mode boundary hand-off")
        if CONTRACT_UNKNOWN in kinds:
            why_bits.append("its block shape could not be read, so it cannot be "
                            "proved restorable")
        if CONTRACT_GEOMETRY in kinds:
            why_bits.append("its block_size is not the scheduler's")
        if CONTRACT_OUTSIDE in kinds:
            why_bits.append("it is outside the group table")
        short = "; ".join(why_bits) or "not restorable"
        listed = ", ".join(str(item) for item in blocked)
        notice_once(
            "incomplete_group_set",
            f"DISK[restore] this model cannot be served from disk: "
            f"prefix-cacheable groups {blocked} are not restorable while only "
            f"{groups} are, and an offer claims EVERY prefix-cacheable group "
            "because get_num_new_matched_tokens skips prefill model-wide while "
            "update_state_after_alloc silently skips the group that has no entry. "
            "Refusing every disk offer on this geometry; recompute is the only "
            "correct answer here, and a clamp to the restorable subset would be the "
            "same nothing with the risk still taken.")
        # THE CONCLUSION FIRST, BECAUSE THE SENTENCE IS CLIPPED. `last_refusal` carries
        # 160 characters, so the part a reader must not miss is the verdict itself and
        # the groups that caused it - both land inside the clip - while the clause that
        # explains WHY they block it, and the restorable subset, are the part allowed to
        # fall off the end. `group_serviceability` and `last_groups_required` publish
        # the same facts uncut for anyone who wants the arithmetic.
        return _fail(f"the disk offers NOTHING on this model: prefix-cacheable "
                     f"group(s) {listed} cannot be restored ({short}), so offering "
                     f"the restorable subset {groups} alone would claim them as "
                     "computed",
                     reason_kind="incomplete_group_set",
                     blocked_group_ids=blocked[:16],
                     blocked_kinds=dict(serviceability.get("blocked_kinds") or {}),
                     blocked_why=dict(serviceability.get("blocked_why") or {}),
                     restorable_group_ids=groups[:16],
                     prefix_cacheable_group_ids=wanted[:8],
                     groups_required=required["rows"][:16],
                     unfilled_group_ids=required["unfilled_group_ids"][:16])
    return _ok(groups=groups, num_groups=len(sizes),
               contracts=contracts,
               boundary_group_ids=[group for group in groups
                                   if contracts.get(str(group))
                                   == CONTRACT_BOUNDARY],
               positional_group_ids=[group for group in groups
                                     if contracts.get(str(group))
                                     == CONTRACT_POSITIONAL],
               group_block_sizes=[int(sizes[group]) for group in groups])


def _group_contract(manager, group: int) -> dict:
    """What a claimed run COSTS this group, derived from the group's own manager.

    THE FUNCTION THAT REPLACES THE 0.2.25 VETO, and the reason the veto was wrong is
    worth keeping in prose because it is the exact bug the box served at 00:08 on
    2026-10-06. The old `_group_positionally_stable` answered a yes/no question and
    `_eligible_groups` used the answer as a FILTER - "offer on the groups that pass".
    Its docstring's conclusion was that a group whose blocks "have no positional name"
    "has nothing to read back from". The fork contradicts that sentence about its own
    align-mode groups, three times over:

      * `has_positionally_stable_blocks` is about SCANNING, not about NAMING -
        `single_type_kv_cache_manager.py:62-69` defines it as "whether positional
        offload scans can follow this block table", and the align override at
        `:1494-1497` explains why align-mode fails that test: "align-mode Mamba can
        null interior states and relocate speculative blocks in place". Nulling and
        relocating a block does not un-hash it.
      * the align hand-off NAMES the state block and hands it to the connector
        (`:2051-2080`, `:1955-1963`), with the contract at
        `kv_cache_manager.py:907-921`: "A connector reads the referenced blocks -
        never resolving them positionally";
      * and the tier stores it under exactly the hash a lookup will ask for
        (`manager.py:1050-1078` inserts under `gpu_block.block_hash` with
        `num_tokens = block_hash_num_tokens`; `:1591-1601` looks up
        `resolve_block_hashes(block_hashes, hash_block_size, block_size)[i]`). The
        live store proves it: one shard directory holds `kvslot-a36ec46fe316bd0d-g0-…`,
        `-g1-…`, `-g2-…`, `-g3-…` and `-g5-…` files for the SAME key16 - the same
        token boundary, named per group, five groups on disk.

    So the flag is not a veto, it is a SHAPE. A positional group's hit list is one
    real block per scheduler block; an align-mode group's hit list is
    `[null_block] * (n - 1) + [one real state block]` (`:1591-1607`). Both are named;
    they cost different numbers of files; and a claim must be backed by the RIGHT
    NUMBER OF FILES FOR EVERY GROUP, which is what `_disk_run` now enforces.

    Everything is read from the manager, so no model name and no group id appears
    here: `has_positionally_stable_blocks` (the fork's own predicate) plus
    `mamba_cache_mode` (the fork's own reason for answering False, read from the
    manager and, failing that, its `kv_cache_spec`). An unreadable manager, a renamed
    flag or an unknown mode answers `unknown`, which is a refusal: a restore we cannot
    justify is a restore we do not do.
    """
    entry: object = None
    try:
        coordinator = getattr(manager, "cpu_coordinator", None)
        single = getattr(coordinator, "single_type_managers", None)
        entry = single[group] if single is not None else None
    except Exception as exc:  # noqa: BLE001 - an unverifiable group has no contract
        snapshot._error(exc, "disk_restore group contract")
        return _ok(contract=CONTRACT_UNKNOWN, stable=None, mamba_cache_mode=None,
                   reason=f"the manager for group {group} could not be read "
                          f"({type(exc).__name__}), so its block shape is unknown")
    stable = getattr(entry, "has_positionally_stable_blocks", None)
    if stable is None:
        spec = getattr(entry, "kv_cache_spec", None)
        stable = getattr(spec, "has_positionally_stable_blocks", None)
    mode = getattr(entry, "mamba_cache_mode", None)
    if not isinstance(mode, str):
        mode = getattr(getattr(entry, "kv_cache_spec", None), "mamba_cache_mode",
                       None)
    if not isinstance(mode, str):
        mode = None
    if stable is True:
        return _ok(contract=CONTRACT_POSITIONAL, stable=True,
                   mamba_cache_mode=mode,
                   reason="positionally stable: one real block per scheduler block "
                          "of the claimed run, each named by that block's own "
                          "group-qualified hash")
    if stable is False and mode == MAMBA_CACHE_MODE_ALIGN:
        return _ok(contract=CONTRACT_BOUNDARY, stable=False, mamba_cache_mode=mode,
                   reason="align-mode Mamba/GDN group: a claimed run costs it "
                          "[null_block]*(n-1) plus ONE real boundary-state block at "
                          "the run's end (single_type_kv_cache_manager.py:1591-1607), "
                          "and that block is named by the align hand-off, so the "
                          "disk file for it exists and is findable")
    if stable is False:
        return _ok(contract=CONTRACT_UNSTABLE, stable=False, mamba_cache_mode=mode,
                   reason="not positionally stable and not an align-mode boundary "
                          "hand-off, so no name exists that a restore could read "
                          "back: an implementation this module does not have")
    return _ok(contract=CONTRACT_UNKNOWN, stable=stable, mamba_cache_mode=mode,
               reason=f"has_positionally_stable_blocks is {stable!r} rather than a "
                      "bool: the fork renamed or removed the predicate and guessing "
                      "a block shape from it is how a claim gets made for bytes "
                      "nobody wrote")


def _group_block_key(resolved: dict, lengths: dict, index: int, group: int,
                     helpers: dict) -> tuple:
    """One (group, block) -> the identity a spilled file for it is named under.

    Four answers, because "no file" and "no hash" and "the hash is the wrong shape"
    are three different facts and only one of them is a prefix we never spilled:

      `("hit", key_hex)`   the index says a file for this group's block exists;
      `("miss", None)`     the block is addressable and simply is not on disk - a run
                           boundary for a positional group, a non-anchor for a
                           boundary group, and a recompute either way;
      `("stop", None)`     this request's hash view does not reach block `index`, or
                           reading it raised: the run cannot extend past here;
      `("refuse", verdict)` the view holds something that is not a block hash, or the
                           engine's own packer raised. That is a bug we must be able
                           to SEE, so it is a named refusal rather than a quiet run
                           end - the same distinction `disk_read` draws between a
                           missing file and a malformed one.
    """
    if index >= int(lengths.get(group) or 0):
        return "miss", None
    try:
        digest = resolved[group][index]
    except Exception as exc:  # noqa: BLE001 - a hole in the view is a stop
        snapshot._error(exc, "disk_restore hash view")
        return "stop", None
    if not isinstance(digest, bytes):
        return "refuse", _fail(
            f"the engine's block hash at index {index} is "
            f"{type(digest).__name__}, not bytes: the group suffix can only be "
            "packed onto raw digests", reason_kind="no_block_hashes")
    try:
        packed = helpers["pack"](digest, group)
        key_hex = bytes(packed).hex()
    except Exception as exc:  # noqa: BLE001 - a packer fault is a refusal
        snapshot._error(exc, "disk_restore make_block_hash_with_group_id")
        return "refuse", _fail(f"make_block_hash_with_group_id raised "
                               f"{type(exc).__name__}",
                               reason_kind="no_key_helpers")
    if not disk_index.contains(key_hex):
        return "miss", None
    return "hit", key_hex


def _disk_run(request, start_block: int, groups: list, block_size: int,
              hash_block_size: int, helpers: dict, contracts: dict | None = None,
              max_blocks: int | None = None) -> dict:
    """The contiguous span of scheduler blocks the SSD tier can supply.

    THIS FUNCTION IS WHERE THE CORRECTNESS OF A CLAIM LIVES. A run is a promise that
    every eligible group has bytes for it, and the two block contracts (`_group_
    contract`) make that promise cost different things, so the rule is per contract:

      * a `positional` group needs a file for EVERY block of the run, because the fork
        pairs one CPU slot per group block and each of those slots receives the KV the
        engine will not recompute;
      * a `boundary_state` group needs a file at the run's LAST block and nowhere
        else, because its hit list is `[null_block] * (n - 1) + [one state block]`
        (`single_type_kv_cache_manager.py:1591-1607`) - the interior entries carry no
        bytes by the engine's own design. Requiring a file for every block of such a
        group would truncate every run on this model to zero (no engine spills a state
        per block); requiring one only at the end is exactly what the engine does
        spill, which is why the live store holds `…-g0-…` through `…-g5-…` files at
        the SAME key16.

    So the run is: walk the positional groups to the first block any of them cannot
    name, then SHORTEN to the longest length whose end block every boundary group CAN
    name. If no length qualifies, there is no run: that is `no_boundary_anchor`, and
    it is a refusal rather than a shorter claim because the shorter claim would be a
    claim on a recurrent state nobody wrote. This is the invariant the box paid for on
    2026-10-06 at 00:08: a run that names bytes for the attention group and none for
    the align-mode groups is not a partial restore, it is a wrong answer at
    temperature=0.

    `max_blocks` is the caller's own ceiling (it defaults to the module's); the anchor
    scan stays inside it so a clamped offer can never be handed a run whose boundary
    it cannot pay for. `anchors` publishes every length that WOULD have worked, which
    is what `_cut_run` uses when a later clamp shortens the offer further.
    """
    block_hashes = getattr(request, "block_hashes", None)
    if block_hashes is None:
        return _fail("the request carries no block_hashes",
                     reason_kind="no_block_hashes")
    declared = contracts or {}
    positional: list = []
    boundary: list = []
    for group in groups:
        kind = declared.get(str(group)) or CONTRACT_POSITIONAL
        if kind == CONTRACT_BOUNDARY:
            boundary.append(group)
        elif kind == CONTRACT_POSITIONAL:
            positional.append(group)
        else:
            # `_eligible_groups` only hands out implementable contracts, so arriving
            # here means the two readings of the model disagree - which is a refusal,
            # not a group to guess about.
            return _fail(f"group {group} has contract {kind!r}, which this run "
                         "cannot price", reason_kind="incomplete_group_set")
    try:
        resolved = {}
        for group in groups:
            # THE view the engine itself uses: `resolve_block_hashes(block_hashes,
            # hash_block_size, block_size)` is what `block_pool.cache_block_hashes`
            # (`block_pool.py:269-271`) indexes when it publishes a block's hash, so
            # index `i` of this view is exactly the digest the spilled file for group
            # `g`'s block `i` was named after - for an align-mode group too, because
            # the hand-off publishes that block's own hash (`manager.py:1050-1078`).
            # `_eligible_groups` has already proved every group here has this block
            # size, so one view serves them all.
            resolved[group] = helpers["resolve"](block_hashes, hash_block_size,
                                                 block_size)
        lengths = {group: len(resolved[group]) for group in groups}
    except Exception as exc:  # noqa: BLE001 - a view we cannot build is a refusal
        snapshot._error(exc, "disk_restore resolve_block_hashes")
        return _fail(f"resolve_block_hashes raised {type(exc).__name__}",
                     reason_kind="no_block_hashes")
    limit = MAX_RESTORE_BLOCKS_PER_OFFER
    if max_blocks is not None:
        limit = max(0, min(limit, int(max_blocks)))
    # --- 1. the positional run: one file per block, per group ------------------
    pos_keys: list = []
    walk_capped = False
    while len(pos_keys) < limit:
        index = start_block + len(pos_keys)
        block_keys: list = []
        complete = True
        for group in positional:
            state, key_hex = _group_block_key(resolved, lengths, index, group,
                                              helpers)
            if state == "refuse":
                return key_hex
            if state != "hit":
                complete = False
                break
            block_keys.append((group, key_hex))
        if not complete:
            break
        pos_keys.append(block_keys)
    else:
        # WHILE/ELSE, and the distinction IS the point: this branch runs only when the
        # loop ended on its CONDITION, so the walk stopped because it reached `limit`
        # and not because a block was missing from the store. That is the per-offer
        # BLOCK bound binding - the one that survives the deleted token cap - and
        # `bounded_by` can name it instead of leaving a reader to infer a cap from a
        # block count that happens to be 64 (0.2.32).
        walk_capped = True
    # --- 2. the anchors: run ends every boundary group can pay for -------------
    anchors: dict = {}
    for length in range(1, len(pos_keys) + 1):
        index = start_block + length - 1
        pairs: list = []
        whole = True
        for group in boundary:
            state, key_hex = _group_block_key(resolved, lengths, index, group,
                                              helpers)
            if state == "refuse":
                return key_hex
            if state != "hit":
                whole = False
                break
            pairs.append((group, key_hex))
        if whole:
            anchors[str(length)] = pairs
    # --- 3. the run, clamped to the longest anchored length --------------------
    if boundary:
        valid = sorted(int(length) for length in anchors)
        blocks = valid[-1] if valid else 0
    else:
        blocks = len(pos_keys)
        anchors = {}
    if blocks <= 0:
        return _ok(blocks=0, keys=[], view_lengths=lengths,
                   positional_groups=positional, boundary_groups=boundary,
                   positional_run=len(pos_keys), anchors={},
                   clamped_to_anchor=False, walk_capped=walk_capped,
                   no_anchor=bool(boundary) and len(pos_keys) > 0)
    # THE TWO VIEWS, and the difference is a bug that was actually shipped once in this
    # change: `keys` is the run AS OFFERED (the last span carrying the boundary-state
    # pairs too), while `positional_keys` is the walk before that merge. A clamp that
    # rebuilds a shorter span from `keys` and then appends its own anchor gets the
    # boundary pairs TWICE - 12 named blocks for 8 allocated slots on the live
    # geometry - which `_merge_into_pending`'s allocation check refuses, so it fails
    # safe, but it fails. Every cut therefore rebuilds from `positional_keys`.
    positional_keys = [list(pairs) for pairs in pos_keys[:blocks]]
    keys = [list(pairs) for pairs in positional_keys]
    keys[blocks - 1] = keys[blocks - 1] + list(anchors.get(str(blocks)) or [])
    return _ok(blocks=blocks, keys=keys, positional_keys=positional_keys,
               view_lengths=lengths,
               positional_groups=positional, boundary_groups=boundary,
               positional_run=len(pos_keys), anchors=anchors,
               clamped_to_anchor=bool(boundary) and blocks != len(pos_keys),
               walk_capped=walk_capped,
               no_anchor=False)


def _cut_run(run: dict, length: int) -> dict:
    """Shorten an offered run to at most `length` blocks, KEEPING ITS BOUNDARY.

    A clamp that just truncates `run["keys"]` is the 00:08 bug in miniature: the real
    boundary-state block sits on the run's LAST block, so cutting the tail off moves
    the claim to a length no group has a state file for and the align-mode groups get
    nulls over a range nobody computed. So a cut only ever lands on an ANCHOR length
    `_disk_run` already proved complete, and when no anchor is short enough the cut
    fails and the offer is refused. Positional-only runs (no boundary group) have no
    anchors to keep and cut freely, exactly as they always did.
    """
    wanted = min(int(length), int(run.get("blocks") or 0))
    boundary = list(run.get("boundary_groups") or [])
    if wanted <= 0:
        return _fail(f"a run of at most {length} blocks leaves nothing to offer",
                     reason_kind="disk_run_below_one_block")
    if not boundary:
        return _ok(blocks=wanted,
                   keys=[list(pairs) for pairs in (run.get("keys") or [])[:wanted]],
                   clamped_from=int(run.get("blocks") or 0), cut_to=wanted)
    valid = sorted(int(key) for key in (run.get("anchors") or {}))
    fitting = [value for value in valid if value <= wanted]
    if not fitting:
        return _fail(f"the shortest run every align-mode group has a boundary state "
                     f"for is longer than the {wanted} block(s) this offer can pay "
                     f"for (anchored lengths: {valid[:16]})",
                     reason_kind="no_boundary_anchor")
    cut = fitting[-1]
    # REBUILT, NOT TRUNCATED: the positional walk, then this length's own anchor. Taking
    # `run["keys"]` (which already carries the anchor of the run's ORIGINAL end) and
    # appending another anchor is what named a boundary state twice for one span.
    walk = run.get("positional_keys")
    if walk is None:
        walk = [pairs for pairs in (run.get("keys") or [])]
    keys = [list(pairs) for pairs in walk[:cut]]
    keys[cut - 1] = keys[cut - 1] + list((run.get("anchors") or {})[str(cut)])
    return _ok(blocks=cut, keys=keys, clamped_from=int(run.get("blocks") or 0),
               cut_to=cut)


# --- the scheduler's door 1: offer the SSD's continuation ---------------------

def offer(manager, request, num_computed_tokens, ram_result):
    """Widen the fork's `(tokens, is_async)` with the SSD's contiguous span.

    NO RENEWAL TICK ANY MORE (0.2.35). Until 0.2.34 this function began with
    `disk_boot.renew_if_due()` - a bounded tick that re-armed the spill path before the
    gate read it, because the arm had a TTL and a lapse here cost a request its disk
    restore. The TTL is gone (the arm is a latch; see `disk_spill`'s module docstring),
    so the tick is gone: a latch has no lapse to forestall, and a heartbeat that
    re-armed from the request path would have been a way to re-arm a box an operator had
    just disarmed.

    SAYING IT HERE BECAUSE THERE WERE TWO OF THEM. The same eleven lines sat in BOTH
    offer entry points, and the one this change's first pass missed kept calling the
    deleted function on EVERY lookup - its own `except Exception` turned that into a
    `snapshot._error()` counter nobody asserted on, and the suite stayed green. The
    containment is right and stays; the guard is
    `test_an_offer_cycle_records_no_plugin_error`, which fails if any door on this path
    files an internal error while doing ordinary work.


    Called by the wrapper AFTER the fork's own body, with its result. Returns the
    result to hand the scheduler: on EVERY refusal path that is the fork's value,
    byte-identical and unconsumed - which is the property that makes an unarmed or
    confused restore indistinguishable from no plugin at all.

    On success it rewrites `manager._pending_cpu_hits[request_id]` into the MERGED
    tuple (the fork's CPU hit, then our freshly allocated destination slots, per
    group in the fork's own order) and returns the raised offer, so every piece of
    refcounting, pairing and state creation stays the fork's.
    """
    _state["lookups"] += 1
    try:
        return _offer(manager, request, num_computed_tokens, ram_result)
    except Exception as exc:  # noqa: BLE001 - a restore never breaks a schedule
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore offer")
        _refuse("bad_call", f"the offer path raised {type(exc).__name__}")
        return ram_result


def suppress_load(request_id: str, reason: str, span=None) -> bool:
    """Never offer disk service to this request (and, if named, this span) again.

    THE LOOP KILLER. A failed load is not a reason to try again on the next step: the
    engine's recovery from a KV load failure resets the request's computed tokens and
    reschedules it, which calls `get_num_new_matched_tokens` again, which asks us the
    same question - and the index has not changed, so we would answer the same way.
    On 2026-10-05 that loop ran 123,474 times in 3.9 minutes (520 offers a second) and
    starved the harness into a watchdog reset. Nothing in the fork bounds it, so the
    bound has to live here.

    Two keys, because the failure has two reachable attributions:
      * the REQUEST id, which `connector_output.failed_recving` names exactly;
      * the SPAN (the first block's key16:group of the run we offered), which is what
        a request that returns with no progress can be matched against when the fork
        reports only invalid block ids and those ids are GPU-side.

    Bounded and never raising: this table is a fuse, not a database.
    """
    try:
        key = str(request_id or "")[:256]
        if not key:
            return False
        if key not in _suppressed and len(_suppressed) >= MAX_SUPPRESSED_REQUESTS:
            # Oldest-first eviction: a fuse that grows without bound in a long boot is
            # its own leak, and forgetting the oldest suppression costs one more offer.
            _suppressed.pop(next(iter(_suppressed)), None)
        fresh = key not in _suppressed
        _suppressed[key] = _clip(reason, 200)
        if fresh:
            _state["suppressions"] += 1
        if span:
            _suppressed_spans[str(span)[:80]] = _clip(reason, 200)
        _logger.warning("kvtier-dev-plugin: DISK[restore] no further disk offer for "
                       "request %s (span %s): %s", key or "?",
                       str(span)[:40] or "-", _clip(reason, 160))
        return True
    except Exception as exc:  # noqa: BLE001 - a fuse must not break the scheduler
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore suppress_load")
        return False


def _suppressed_span(run) -> str | None:
    """The span key of a run offered by `_disk_run`: its first block's first group.

    The FIRST block identifies the span well enough to be useful: a run that starts
    at the same scheduler block for the same groups is the same offer, and the fused
    span is exactly the offer that failed. Keying on the first block rather than the
    whole list means a shorter or longer run that overlaps it is also refused, which
    is the safe direction: a missed restore costs a recompute, a re-offered failure
    costs the whole engine.
    """
    try:
        first = (run.get("keys") or [None])[0]
        if not first:
            return None
        group, key_hex = first[0][0], first[0][1]
        return f"{str(key_hex)[:16]}:{int(group)}"
    except Exception as exc:  # noqa: BLE001 - a key we cannot read is no key
        snapshot._error(exc, "disk_restore span key")
        return None


def _pool_price_bytes(run_keys=None, blocks: int = 0) -> tuple:
    """(bytes ONE read needs, where that figure came from) - the offer's unit price.

    THREE SOURCES, TRIED IN ORDER OF PRECISION, AND THE ANSWER IS CONSERVATIVE AT
    EACH STEP: the largest of the shapes actually in play for the identities this
    offer would read. A read costs one bounce buffer of exactly the file's size, so an
    under-estimate is exactly the cliff this release exists to remove.

      1. THE EXACT FIGURE, from the index. Every journal line records the size the
         writer committed (`disk_index.entry_bytes`, which moves no witness counter),
         so the identities this very offer would read carry their own widths. This is
         the per-group answer the pool wants, and it is what a store holding TWO shapes
         looks like from the scheduler: the offer is priced at the bigger one.
      2. THE RANK WIDTHS, from the rank files: `total_bytes_per_block` as each worker
         measured it, the widest live one winning. EngineCore cannot measure this
         itself (it has no KV tensors), an offer spans every rank, and the pool is per
         process - so the widest rank is the only cap that cannot be optimistic for
         somebody. The live box measures 11,468,800 B on PP0 and 12,902,400 B on PP1
         at the same moment.
      3. THE CONSTANT `POOL_PRICE_BYTES_PER_JOB_FALLBACK`, the larger of those two
         shapes, for an early offer with neither an indexed identity nor a rank file to
         ask. Capping a little harder is the safe direction: a missed restore costs a
         recompute, an over-priced offer costs the request its RAM hit AND a
         reschedule.
    """
    exact: list = []
    walked = 0
    truncated = False
    try:
        for entry_list in list(run_keys or [])[:max(0, int(blocks))]:
            for identity in entry_list or []:
                if walked >= POOL_PRICE_MAX_IDENTITIES:
                    truncated = True
                    break
                size = disk_index.entry_bytes(identity[1])
                walked += 1
                if size:
                    exact.append(int(size))
            if truncated:
                break
    except Exception as exc:  # noqa: BLE001 - a price is never worth a fault
        snapshot._error(exc, "disk_restore pool price index walk")
    if exact:
        return (max(exact),
                f"exact: {max(exact)} B, the largest file size this index journalled "
                f"for the {walked} identit(ies) this offer would read"
                + (f" (walk capped at {POOL_PRICE_MAX_IDENTITIES})" if truncated
                   else ""))
    candidates: list = []
    try:
        local = _int(disk_spill.slot_width_bytes())
        if local is not None and local > 0:
            candidates.append((local, "this process's own fingerprint "
                                      f"total_bytes_per_block={local}"))
    except Exception as exc:  # noqa: BLE001 - a price is never worth a fault
        snapshot._error(exc, "disk_restore pool price local width")
    try:
        proof = worker_proof()
        widest = _int(proof.get("bytes_per_file_max"))
        if widest is None:
            # A proof that answered before it read any rank file (no diagnostics
            # directory, no files yet) carries no width at all; the state the last
            # real read left behind is the same fact, one cache refresh old.
            widest = _int(_state.get("ranks_bytes_per_file_max"))
        if widest is not None and widest > 0:
            candidates.append((widest, f"the widest slot width published by a live "
                                       f"rank file ({widest} bytes, "
                                       f"{int(proof.get('bytes_per_file_ranks') or 0)}"
                                       " rank(s) published)"))
    except Exception as exc:  # noqa: BLE001 - as above
        snapshot._error(exc, "disk_restore pool price rank widths")
    if candidates:
        best = max(candidates, key=lambda item: item[0])
        return (int(best[0]),
                f"rank width: {int(best[0])} B, measured by {best[1]}; no indexed "
                "identity named a size for this run")
    return (POOL_PRICE_BYTES_PER_JOB_FALLBACK,
            f"fallback: {POOL_PRICE_BYTES_PER_JOB_FALLBACK} B, the larger of the two "
            f"shapes this build writes (the smaller is 11,468,800), because nothing "
            "named a size for this run - no indexed identity, no rank file width, no "
            "local fingerprint")


def _offer_pool_view() -> dict:
    """The bounce-buffer pricing as it stands, plus the last offer that used it.

    Recomputed here rather than stored, because the POOL side moves (buffers are held
    and returned) while the PRICE side only moves when a rank file changes. A reader
    that sees `jobs_the_pool_can_carry` below the job count of `last_offer_pool` is
    looking at an offer made when the pool could carry it and a pool that has since
    filled - which is the concurrent-offers case this release does NOT yet solve.
    """
    price, source = _pool_price_bytes()
    afford = disk_read.afford_reads(price)
    return {"bytes_per_job": int(price),
            "price_source": _clip(source, 160),
            "pool_budget_bytes": int(afford.get("pool_budget_bytes") or 0),
            "pool_committed_bytes": int(afford.get("pool_committed_bytes") or 0),
            "jobs_the_pool_can_carry": int(afford.get("jobs_at_empty_pool") or 0),
            "jobs_that_fit_now": int(afford.get("jobs_now") or 0),
            "fallback_price_bytes": POOL_PRICE_BYTES_PER_JOB_FALLBACK,
            "capped_offers": int(_state["offer_pool_capped"]),
            "refused_offers": int(_state["offer_pool_refused"]),
            "last_offer": _state["last_offer_pool"],
            "note": ("an offer is capped to `jobs_the_pool_can_carry` because a read "
                     "holds its bounce buffer from issue until the apply point, so "
                     "ONE offer needs the pool to hold ALL of its buffers at once. "
                     "Two full-size offers in the same step still divide the pool "
                     "between them, and the surplus is refused at bind as "
                     "`buffer_cap`: this view publishes `jobs_that_fit_now` so that "
                     "case is at least visible")}


def _span_parts(run) -> tuple:
    """(block identity, group_id) of a run's FIRST block, for a record's provenance.

    The same first block `_suppressed_span` keys the span fuse on, so a record and a
    fuse name one and the same identity.

    The identity is clipped at 128 characters and NOT at the 64 a payload field uses:
    a run's key is the engine's group-qualified hash (64 digest + 8 group), and a clip
    at 64 would slice off the group and hand `_identity_pair` a string that is neither
    a key16 nor a full key - which `split_block_key` then reads as a digest whose group
    suffix is `9fa0a1a2`, refuses it, and the provenance lookup that follows reports
    "never seen" about a file this index journalled itself.
    """
    try:
        first = (run.get("keys") or [None])[0]
        if not first:
            return None, None
        group, key_hex = first[0][0], first[0][1]
        return _failure_text(key_hex, 128), _int(group)
    except Exception as exc:  # noqa: BLE001 - no identity is only a weaker record
        snapshot._error(exc, "disk_restore span parts")
        return None, None


def _offer(manager, request, num_computed_tokens, ram_result) -> dict | tuple:
    # NO REQUEST-SCOPED CHECK COMES FIRST, and that is deliberate. Until 0.2.31 the
    # first thing here was a read of `request.kv_transfer_params`, because a client
    # field decided whether this request was served from disk. Nothing about a request
    # decides that any more: the boot posture decided it when the operator POSTed the
    # `/disk` verb, and the gate below is the cheapest true statement about that.
    opened = gate()
    if not opened["ok"]:
        kind = opened.get("reason_kind")
        if kind == "offers_disabled":
            # Counted, because a boot that is armed everywhere but here is exactly the
            # state an operator needs to see from the counters.
            _refuse("offers_disabled", opened["reason"])
        elif kind == "scheduler_armed_workers_disarmed":
            _refuse("workers_not_servicing", opened["reason"])
        elif kind == "no_worker_proof":
            _refuse("no_worker_proof", opened["reason"])
        elif kind in ("not_configured", "disarmed", "tier_disarmed"):
            # THE CLOSED GATE, COUNTED BY NAME (0.2.21). This branch used to call
            # `_note()` and leave `refused` untouched, so `refused == {}` was ambiguous
            # between "nothing refused us" and "the gate refused every request of the
            # boot without a trace" - and that ambiguity cost the whole afternoon of
            # 2026-10-05, which read an empty `refused` as evidence that the ALIGNMENT
            # guard had declined the request when the gate had declined it. A closed
            # gate is not a judgement about THIS request (the comment here used to say
            # so) but it IS the reason no offer was made, and the `refused` dict has to
            # be readable on its own. `_refuse` sets `last_reason` exactly as `_note`
            # did, and the value returned to the scheduler is unchanged.
            _refuse(kind, opened["reason"])
        else:
            # A gate reason this module has no bucket for. Still `last_reason` only, so
            # a gate that grows a fourth kind shows up as an absent counter rather than
            # being filed under someone else's name.
            _note(opened["reason"])
        return ram_result
    request_id = str(getattr(request, "request_id", "") or "")
    if request_id in _suppressed:
        _state["suppressed_hits"] += 1
        _refuse("suppressed_after_failure",
                f"a disk load already failed for this request "
                f"({_clip(_suppressed.get(request_id), 120)}) and re-offering the "
                "same span to a request the engine just rescheduled is a loop, not "
                "a retry")
        return ram_result
    if not isinstance(ram_result, tuple) or len(ram_result) < 2:
        _refuse("bad_result", f"the fork returned {type(ram_result).__name__} "
                              "instead of (tokens, is_async)")
        return ram_result
    ram_tokens = ram_result[0]
    if ram_tokens is None:
        ram_tokens = 0
    if not _plain_int(ram_tokens) or ram_tokens < 0:
        _refuse("bad_result", f"the fork reported {ram_tokens!r} matched tokens")
        return ram_result
    # DISK AUTHORITY IS A BOOT POSTURE, NOT A REQUEST FIELD (0.2.31). With the switch
    # up, this request does not get "whatever the RAM tier left over": its host-RAM hit
    # is treated as ZERO for the offer arithmetic, so the disk index decides the span
    # from the local GPU-computed boundary. Everything downstream reads `ram_tokens`,
    # so this one assignment is the whole change of authority - and `computed` is
    # deliberately NOT touched, because it is the offset the fork's DMA writes at
    # (zeroing it would aim the copy at the request's already-cached, shared prefix
    # blocks). With the switch down the fork's answer stands and the disk only fills
    # the gap it left, which is what `offers_enabled` alone means.
    #
    # There is no per-request input to this decision any more. Until 0.2.31 the line
    # below read `request.kv_transfer_params`, so a client body chose whether its KV
    # came off SSD; the answer now comes from `_state`, and no request can raise it,
    # lower it, or opt out of it.
    #
    # The fork keeps its side of a RAM hit in exactly one place, so ignoring the hit
    # means releasing exactly one thing (`_release_fork_pin`, below). Until that
    # release happens NOTHING here has consumed the fork's answer: every refusal
    # below still `return ram_result`, and a request that gets no disk offer keeps
    # the RAM hit the fork gave it.
    ram_tokens_reported = int(ram_tokens)
    disk_authority = bool(_state["disk_authority"])
    fork_pending: object = _PIN_UNREAD
    if disk_authority:
        _state["authority_lookups"] += 1
        if ram_tokens_reported > 0:
            if not callable(getattr(manager, "_free_pending_cpu_hit", None)):
                # Refuse BEFORE any allocation, index read or state write: without
                # the fork's own release path, taking the disk's word would strand
                # a pin on every one of those CPU blocks for the rest of the boot.
                _state["pin_release_failures"] += 1
                _refuse("pin_release_unavailable",
                        f"the fork reported {ram_tokens_reported} host-RAM tokens "
                        "and this manager has no _free_pending_cpu_hit() to put "
                        "them back, so the disk cannot be made the authority here "
                        "without leaking the pin")
                return ram_result
            try:
                table = getattr(manager, "_pending_cpu_hits", None)
                fork_pending = (table.get(request_id) if table is not None
                                else None)
            except Exception as exc:  # noqa: BLE001 - unknown pins are not ours
                snapshot._error(exc, "disk_restore read pinned hit")
                _state["pin_release_failures"] += 1
                _refuse("pin_release_unavailable",
                        f"the fork's pinned host-RAM hit could not be read "
                        f"({type(exc).__name__}), and a pin that cannot be read "
                        "cannot be released")
                return ram_result
        ram_tokens = 0
    block_size = _int(getattr(manager, "block_size", None))
    hash_block_size = _int(getattr(manager, "hash_block_size", None))
    computed = _int(num_computed_tokens)
    if not block_size or block_size <= 0 or not hash_block_size:
        _refuse("bad_call", "the manager exposes no usable block_size/"
                            "hash_block_size")
        return ram_result
    if computed is None or computed < 0 or computed % block_size != 0:
        # The fork's own guard (`manager.py:334-347`) refuses an unaligned local
        # prefix for the same reason: an external suffix cannot start mid-block.
        _refuse("unaligned_local_prefix", f"{computed} local tokens is not a whole "
                                          f"number of {block_size}-token blocks")
        return ram_result
    if request_id in _offered_at and computed <= int(_offered_at[request_id]):
        # THE LOOP FUSE, second half, and it needs `computed` to say anything. The
        # engine's recovery from a KV load failure puts the request back at the door
        # with its computed tokens RESET, so a request we already offered comes back at
        # the same place or earlier having made no progress - and the index has not
        # changed, so the honest answer to the second asking is the first answer plus a
        # failure. Only a request that has moved FORWARD gets a second offer.
        #
        # ONE EXCEPTION, BOUNDED TO ONE (0.2.32): a request whose first load lost a
        # RACE (a timing or capacity kind, and nothing bad in the window) holds exactly
        # `MAX_RACE_REOFFERS_PER_REQUEST` re-offers, spent here and never refilled - see
        # `_grant_race_reoffer`. Without it, deleting the permanent fuse for a race
        # would delete the loop's only bound; with it, the loop stays bounded at TWO
        # offers per request whether the losses are races or faults. A data failure
        # never reaches this branch with an allowance, because `_fuse_or_spare` fused
        # it at the report.
        if _spend_race_reoffer(request_id):
            _state["last_fuse_decision"] = (
                f"req={request_id[:48]} where=second_asking verdict=timing "
                f"reoffer=granted (one per request, spent)")[:400]
            _note(f"re-offering {request_id[:48]} at {computed} computed tokens: its "
                  f"first load lost a race and its one re-offer is unspent (the "
                  f"allowance is now spent)")
        else:
            _state["suppressed_hits"] += 1
            suppress_load(request_id, f"it came back for a second disk offer at "
                                      f"{computed} computed tokens having reached "
                                      f"{_offered_at[request_id]} on the first one")
            _refuse("suppressed_after_failure",
                    f"{request_id[:48]} was already offered disk tokens at "
                    f"{_offered_at[request_id]} computed tokens and is asking again at "
                    f"{computed}: the engine rescheduled it, so the first load did not "
                    f"land, and its one race re-offer is spent ({request_id[:48]} "
                    "would be offered a third time - that is the loop that starved "
                    "the box)")
            return ram_result
    if ram_tokens % block_size != 0:
        # A fine-grained (hash-aligned) RAM hit would interleave with our blocks
        # inside a group block, so the merged list's ORDER would no longer mean
        # "one whole block per entry". Decline rather than mis-pair. Under disk
        # authority `ram_tokens` is already the 0 this cannot refuse, which is the
        # whole point: a partial RAM hit used to be a refusal that made the disk
        # path untestable on any prompt that was not a whole number of blocks.
        _refuse("ram_hit_not_block_aligned", f"the host-RAM hit is {ram_tokens} "
                                             f"tokens, not a multiple of the "
                                             f"{block_size}-token scheduler block")
        return ram_result
    if getattr(request, "skip_reading_prefix_cache", False):
        _refuse("skip_read_requested", "the request asked not to read the prefix "
                                       "cache, so the fork returned nothing and "
                                       "neither may we")
        return ram_result
    num_tokens = _int(getattr(request, "num_tokens", None))
    if num_tokens is None:
        _refuse("bad_call", "the request carries no usable num_tokens")
        return ram_result
    max_hit_len = num_tokens - 1 - computed
    # THE CEILING THE REFUSAL HAS TO PRICE. `_eligible_groups` answers a question about
    # the MODEL, but its refusal is worth reading only if it says what the tokens this
    # request was about to be offered would have cost every group, so the offer's own
    # bound travels into the call. An upper bound, clipped to the same caps the offer
    # itself obeys; the real offer re-prices it with its final `blocks` below.
    claim_blocks = max(1, min(max_hit_len // block_size,
                              MAX_RESTORE_BLOCKS_PER_OFFER))
    if max_hit_len <= 0:
        # The fork's own rule (`manager.py:353-357`): at least one token must be
        # recomputed, or the sampler has nothing to sample.
        _refuse("no_tokens_left", f"{computed} computed of {num_tokens} tokens "
                                  "leaves nothing to restore")
        return ram_result
    groups = _eligible_groups(manager, block_size, claim_blocks)
    if not groups["ok"]:
        _refuse(groups.get("reason_kind") or "no_eligible_group", groups["reason"])
        return ram_result
    helpers = _key_helpers()
    if not helpers["ok"]:
        _refuse("no_key_helpers", helpers["reason"])
        return ram_result
    if len(_loads) >= MAX_TRACKED_LOADS:
        _refuse("too_many_loads", f"{len(_loads)} disk loads are already in flight "
                                  f"(cap {MAX_TRACKED_LOADS})")
        return ram_result
    refreshed = _refresh_index()
    if not refreshed["ok"]:
        _refuse("index_refresh_failed", refreshed["reason"])
        return ram_result
    if not request_id or len(request_id) > 256:
        _refuse("bad_call", "the request id is unusable")
        return ram_result

    start_block = (computed // block_size) + (ram_tokens // block_size)
    contracts = groups.get("contracts") or {}
    boundary_groups = list(groups.get("boundary_group_ids") or [])
    positional_groups = list(groups.get("positional_group_ids") or [])
    # THE SLOT BILL, PER CONTRACT (0.2.27). A positional group costs one destination
    # CPU slot per span; an align-mode group costs ONE real slot for the whole run -
    # its boundary state - and `[null]*(n-1)` placeholders that cost nothing because
    # the pool's own null block is not allocated. Pricing a boundary group per span
    # would refuse offers the tier can pay for; pricing it per RUN and forgetting the
    # nulls would land the state on the wrong GPU block, so both halves are here.
    per_span = len(positional_groups)
    boundary_slots = len(boundary_groups)
    run = _disk_run(request, start_block, groups["groups"], block_size,
                    hash_block_size, helpers, contracts=contracts)
    if not run["ok"]:
        _refuse(run.get("reason_kind") or "no_block_hashes", run["reason"])
        return ram_result
    if run["blocks"] <= 0:
        if run.get("no_anchor"):
            # The positional groups have the whole run and NO run end has a state
            # file in every align-mode group. This is the 00:08 shape and it is a
            # named refusal, not a shorter offer.
            _refuse("no_boundary_anchor",
                    f"groups {positional_groups} hold a contiguous run from block "
                    f"{start_block} ({run['positional_run']} block(s)) but no run "
                    f"end has a boundary-state file in every align-mode group "
                    f"{boundary_groups}, so the one block those groups need for a "
                    "claim does not exist")
            return ram_result
        _refuse("no_disk_run", f"the SSD tier holds no contiguous span starting at "
                               f"block {start_block} for all of groups "
                               f"{groups['groups']}"
                               + _index_blindness()["sentence"])
        return ram_result
    # The SPAN half of the loop fuse: this exact span already failed for somebody.
    span = _suppressed_span(run)
    if span and span in _suppressed_spans:
        _state["suppressed_spans"] += 1
        _refuse("suppressed_after_failure",
                f"the span starting at block {start_block} already failed a disk "
                f"load ({_clip(_suppressed_spans.get(span), 120)}): offering it to "
                "this request would fail the same way and cost it its RAM hit too")
        return ram_result
    matched = min(run["blocks"] * block_size, max_hit_len)
    floored = whole_block_tokens(matched, block_size)
    if not floored["ok"]:
        _refuse("bad_call", floored["reason"])
        return ram_result
    disk_tokens = int(floored["tokens"])
    # WHICH BOUND SHORTENED THIS OFFER, recorded as it happens. THREE bounds are left
    # on this path and they have three different fixes - the per-offer BLOCK walk
    # (`MAX_RESTORE_BLOCKS_PER_OFFER`), the CPU-slot pool, and the bounce-buffer pool -
    # so an offer that came in short has to say WHICH one bit, or a job count is the
    # only evidence and a job count cannot be read.
    #
    # THE FIXED 32,000-TOKEN CAP AND ITS CLAMP WERE DELETED HERE (0.2.32), along with
    # `offer_clamped`: with no configured token cap there is nothing to clamp and
    # nothing to count, and `bounded_by` can no longer contain `offer_cap`. THE RISK
    # THAT DELETION ACCEPTS (the operator's decision, not an oversight): that cap was
    # the only accidental guard against the still-open TODO 14 - with no reservation
    # arbiter, two simultaneous offers each price affordability against the pool's
    # CURRENT free bytes, both believe they fit, and together they over-commit. Accepted
    # because the live offer rate is 4 offers in 332 lookups, so overlap is near-zero;
    # raising that rate is a different trade and needs the arbiter (reserve at adoption,
    # release at `finish_batch`) scheduled first. See MAX_RESTORE_BLOCKS_PER_OFFER.
    bounds: list = []
    # Remembered for `offer_block_ceiling()`, which publishes the block bound in tokens
    # to processes that have no manager of their own to ask (a worker rank's payload).
    _state["offer_block_size_seen"] = int(block_size)
    if run.get("walk_capped") and max_hit_len > run["blocks"] * block_size:
        # The walk stopped at the block cap with tokens still left to restore, so the
        # block cap - not the pool and not the CPU slots - is what ended this run. The
        # `max_hit_len` test is what keeps the label honest: a run that happened to be
        # exactly 64 blocks long because that is all the store HOLDS says nothing here
        # (the store bounded it, and `no_disk_run`/`run_blocks` already say that).
        bounds.append("block_cap")
        _state["offer_block_capped"] += 1
    if disk_tokens <= 0:
        _refuse("disk_run_below_one_block", f"{matched} matched tokens is below one "
                                            f"{block_size}-token scheduler block "
                                            f"({floored['discarded_tokens']} "
                                            "tokens discarded): offering it would "
                                            "park the request for a load the "
                                            "scheduler will throw away")
        return ram_result

    blocks = disk_tokens // block_size
    needed = blocks * per_span + boundary_slots
    if needed > MAX_SLOTS_PER_OFFER:
        blocks = max(0, (MAX_SLOTS_PER_OFFER - boundary_slots)) // max(1, per_span)
        needed = blocks * per_span + boundary_slots
        disk_tokens = blocks * block_size
        # NAME IT. Until 0.2.35 this clamp shortened the offer and left no mark in
        # `bounded_by`, so a multi-group offer read as `["block_cap"]` while the bound
        # that actually cut it was the CPU-slot cap - one entry short, and the entry
        # that names the fix (`MAX_SLOTS_PER_OFFER`, or a model with fewer groups).
        # Raising the block cap 64 -> 128 is what made the gap common rather than
        # theoretical: at 128 spans x 3 positional groups the slot bound is the
        # OPERATIVE limit for every multi-group model, so an unlabelled clamp is now the
        # normal answer instead of a corner. Guarded, because the engine's own free-pool
        # clamp below reports the SAME label and the list is a set of names in order.
        if "cpu_slots" not in bounds:
            bounds.append("cpu_slots")
    if blocks <= 0:
        _refuse("no_cpu_slots", f"one span needs {per_span} CPU slots plus "
                                f"{boundary_slots} boundary slot(s) and the cap is "
                                f"{MAX_SLOTS_PER_OFFER}")
        return ram_result
    pool = getattr(manager, "cpu_block_pool", None)
    free = None
    try:
        free = _int(pool.get_num_free_blocks())
    except Exception as exc:  # noqa: BLE001 - an unreadable pool is a refusal
        snapshot._error(exc, "disk_restore get_num_free_blocks")
        _refuse("no_cpu_slots", f"the CPU pool's free count is unreadable: "
                                f"{type(exc).__name__}")
        return ram_result
    if free is None:
        _refuse("no_cpu_slots", "the CPU pool exposes no get_num_free_blocks()")
        return ram_result
    affordable = max(0, (free - CPU_SLOT_RESERVE - boundary_slots)) // max(1, per_span)
    if affordable <= 0:
        _refuse("no_cpu_slots", f"the CPU pool has {free} free blocks, "
                                f"{CPU_SLOT_RESERVE} are reserved and "
                                f"{boundary_slots} go to boundary states, so not "
                                "even one span can be given a destination")
        return ram_result
    if affordable < blocks:
        blocks = affordable
        needed = blocks * per_span + boundary_slots
        disk_tokens = blocks * block_size
        bounds.append("cpu_slots")
    # THE BOUNCE-BUFFER BILL, PRICED BEFORE THE OFFER EXISTS (0.2.31). Every clamp
    # above asks how many DESTINATIONS the offer can have; none of them asked how many
    # READS the process can carry. A read lands in a host-RAM bounce buffer taken at
    # issue and returned at the apply point - one forward pass later - so an offer of
    # N jobs needs the pool to hold N buffers AT ONCE, and `disk_read` REFUSES rather
    # than queueing when it cannot (`request_read`'s `buffer_cap`). The consequence of
    # not pricing it was the 0.2.30 failure: the scheduler offered 13 jobs to a pool
    # that fits 11, the surplus died at bind, the apply point pruned them into
    # `invalid_block_ids`, the engine reset the request's computed tokens and resched
    # uled it, and `note_reported` then read that same set back and fused the request
    # for good. The refusal reason and the record below carry the byte figures, which
    # is the part that was never publishable.
    price_bytes, price_source = _pool_price_bytes(run.get("keys"), blocks)
    afford = disk_read.afford_reads(price_bytes)
    jobs_ceiling = int(afford["jobs_at_empty_pool"]) if afford.get("ok") else 0
    jobs_wanted = int(needed)
    pool_pricing = {"bytes_per_job": int(price_bytes),
                    "price_source": _clip(price_source, 160),
                    "pool_budget_bytes": int(afford.get("pool_budget_bytes") or 0),
                    "pool_committed_bytes": int(afford.get("pool_committed_bytes")
                                                or 0),
                    "jobs_the_offer_needed": jobs_wanted,
                    "jobs_the_pool_can_carry": jobs_ceiling,
                    "jobs_that_fit_now": int(afford.get("jobs_now") or 0),
                    "bytes_the_offer_needs": jobs_wanted * int(price_bytes),
                    # THE PER-OFFER BOUND THAT STOOD WHEN THIS OFFER WAS PRICED, beside
                    # the pool arithmetic `bounded_by` names it next to (0.2.32): a
                    # reader asking "was this offer short because of a cap or because of
                    # the pool" gets both numbers from the one field. BLOCKS, because
                    # the deleted fixed token cap is not coming back under a new name.
                    "block_cap_blocks": MAX_RESTORE_BLOCKS_PER_OFFER,
                    "block_cap_tokens": int(offer_block_ceiling(block_size)["tokens"]),
                    "bounded_by": bounds}
    _state["last_offer_pool"] = dict(pool_pricing)
    # ONE SPAN IS THE MINIMUM OFFER, and a span costs `per_span` positional reads plus
    # the boundary-state reads the run cannot be claimed without. Below that the only
    # honest answer is NO OFFER AT ALL - not a shorter offer, because a shorter offer
    # is what got refused at bind last time.
    if jobs_ceiling < per_span + boundary_slots:
        _state["offer_pool_refused"] += 1
        first_key, first_group = _span_parts(run)
        _note_load_failure("buffer_cap", where="offer",
                           reason=f"the pool budget {jobs_ceiling} job(s) of "
                                  f"{price_bytes} B cannot hold one span "
                                  f"({per_span} positional + {boundary_slots} "
                                  "boundary-state reads), so no offer is made",
                           request_id=request_id, span=span, group=first_group,
                           key16=first_key, bytes_needed=price_bytes,
                           jobs_needed=jobs_wanted, jobs_fit=jobs_ceiling,
                           pool_budget_bytes=int(afford.get("pool_budget_bytes")
                                                 or 0),
                           pool_committed_bytes=int(afford.get("pool_committed_bytes")
                                                    or 0))
        _refuse("no_pool_budget",
                f"the pool holds {jobs_ceiling} read(s) of {price_bytes} bytes "
                f"(budget {int(afford.get('pool_budget_bytes') or 0)} B) while one "
                f"span needs {per_span} positional + {boundary_slots} "
                f"boundary-state read(s): offering the {jobs_wanted}-job run would "
                "have it refused at bind as buffer_cap, which costs the request its "
                f"RAM hit AND a reschedule. Price: {price_source}")
        return ram_result
    if jobs_ceiling < needed:
        capped = (jobs_ceiling - boundary_slots) // max(1, per_span)
        if capped <= 0:
            # Same arithmetic as above, reached from the other side. With one or more
            # positional groups the check above already guarantees `capped >= 1` (a
            # ceiling of at least `per_span + boundary_slots` leaves a whole span after
            # the states), so this fires on the one shape where `per_span` is zero: a
            # model whose every cacheable group is a boundary-state group. Refusing is
            # still the right answer for it, and the sentence says which half ate the
            # pool.
            _state["offer_pool_refused"] += 1
            first_key, first_group = _span_parts(run)
            _note_load_failure("buffer_cap", where="offer",
                               reason="the pool can hold the boundary-state reads but "
                                      "not one span on top of them",
                               request_id=request_id, span=span, group=first_group,
                               key16=first_key, bytes_needed=price_bytes,
                               jobs_needed=jobs_wanted, jobs_fit=jobs_ceiling,
                               pool_budget_bytes=int(afford.get("pool_budget_bytes")
                                                     or 0),
                               pool_committed_bytes=int(afford
                                                        .get("pool_committed_bytes")
                                                        or 0))
            _refuse("no_pool_budget",
                    f"the pool holds {jobs_ceiling} read(s) of {price_bytes} bytes "
                    f"and the {boundary_slots} boundary-state block(s) this run must "
                    f"claim leave no room for one span's {per_span} positional "
                    f"read(s) (budget {int(afford.get('pool_budget_bytes') or 0)} B): "
                    f"no offer is made. Price: {price_source}")
            return ram_result
        blocks = capped
        needed = blocks * per_span + boundary_slots
        disk_tokens = blocks * block_size
        bounds.append("pool_budget")
        pool_pricing["jobs_offered"] = int(needed)
        pool_pricing["bytes_offered"] = int(needed * price_bytes)
        pool_pricing["jobs_dropped"] = int(jobs_wanted - needed)
        _state["offer_pool_capped"] += 1
        _state["last_offer_pool"] = dict(pool_pricing)
    # THE BOUNDARY-SAFE CUT. Every clamp above shortens the OFFER, and with an
    # align-mode group in the set a shortening is not a truncation: the one real
    # state block rides on the run's LAST block, so a cut that moves the end moves
    # the claim to a length no group has a state for. `_cut_run` only ever lands on a
    # length `_disk_run` proved anchored, and refuses if none is short enough. (For a
    # positional-only model this is the plain `run["keys"][:blocks]` it always was.)
    cut = _cut_run(run, blocks)
    if not cut["ok"]:
        _refuse(cut.get("reason_kind") or "no_boundary_anchor", cut["reason"])
        return ram_result
    if cut["blocks"] != blocks:
        blocks = cut["blocks"]
        needed = blocks * per_span + boundary_slots
        disk_tokens = blocks * block_size
    keys = cut["keys"]
    # The null block the align-mode groups' interior entries are made of. The fork's
    # own Mamba hit uses the CPU pool's `null_block` for exactly those slots, and
    # `update_state_after_alloc` skips entries whose `is_null` is set while KEEPING
    # THE INDEX, which is what lands the real state on `group_gpu_ids[start+B-1]`.
    null_block = getattr(pool, NULL_BLOCK_ATTR, None) if boundary_groups else None
    if boundary_groups and (
            null_block is None or not getattr(null_block, "is_null", False)):
        _refuse("no_null_block", "the CPU pool exposes no usable `null_block` "
                                 f"(is_null={getattr(null_block, 'is_null', None)!r}), "
                                 "so the align-mode groups' interior entries cannot "
                                 "be built and their one state block would land on "
                                 "the run's first GPU block instead of its last")
        return ram_result
    allocated = _allocate(pool, needed)
    if not allocated["ok"]:
        _refuse("alloc_refused", allocated["reason"])
        return ram_result
    merged = _merge_into_pending(manager, request_id, ram_tokens, computed,
                                 block_size, groups["groups"],
                                 groups["num_groups"], allocated["blocks"],
                                 keys, opened["root"], contracts=contracts,
                                 null_block=null_block)
    if not merged["ok"]:
        _release(pool, allocated["blocks"])
        _refuse(merged.get("reason_kind") or "ram_shape", merged["reason"])
        return ram_result
    # THE MERGE IS HANDED TO THE FORK: it pops this tuple in
    # `update_state_after_alloc`, takes the leading blocks per group, touches both
    # sides and frees the untaken remainder. We own no refs of our own.
    try:
        manager._pending_cpu_hits[request_id] = merged["pending"]
    except Exception as exc:  # noqa: BLE001 - a manager we cannot write is a no-op
        _release(pool, allocated["blocks"])
        snapshot._error(exc, "disk_restore pending_cpu_hits")
        _refuse("ram_shape", f"_pending_cpu_hits could not be written: "
                             f"{type(exc).__name__}")
        return ram_result
    # THE ENTRY IS OURS NOW, so the fork's pinned hit is the one thing left to put
    # back, and this is the only place it can be put back. The ordering is the whole
    # safety argument for the authority rule: BEFORE this write the fork's tuple is
    # still in the table and every failure path above returns the fork's own answer
    # with its pin intact (no leak, no double free, the request keeps its RAM hit);
    # AFTER it, nothing else in the fork will ever release those blocks, so releasing
    # here is the difference between an ignored hit and a leaked one.
    if disk_authority and ram_tokens_reported > 0:
        if isinstance(fork_pending, tuple) and fork_pending:
            released = _release_fork_pin(manager, fork_pending)
            if not released["ok"]:
                # The offer already stands: the fork holds OUR tuple and will free
                # our slots in its own time. Unwinding a committed offer to undo one
                # failed release is the worse error, so this is a counted, shouted
                # leak and not a refusal.
                _state["pin_release_failures"] += 1
                _logger.warning(
                    "kvtier-dev-plugin: DISK[restore] LEAKED host-RAM pin for %s: "
                    "the fork's %d-token hit could not be released (%s); those CPU "
                    "blocks stay pinned for the boot",
                    request_id[:48], ram_tokens_reported,
                    released.get("reason"))
        _state["local_hit_ignored"] += 1
        _state["local_hit_tokens_ignored"] += ram_tokens_reported
        _state["last_local_hit_ignored"] = _clip(
            f"{request_id[:48]}: the fork's {ram_tokens_reported}-token host-RAM "
            f"hit was treated as 0 (disk is the authority), so the disk span starts "
            f"at block {computed // block_size}")
        _logger.warning("kvtier-dev-plugin: DISK[restore] serving %s from DISK "
                        "against a %d-token host-RAM hit: computed=%d start_block=%d "
                        "offered=%d blocks=%d",
                        request_id[:48], ram_tokens_reported, computed,
                        computed // block_size, disk_tokens, blocks)
    _register_load(request_id, merged, disk_tokens, span=span)
    # Remember where this offer left the request, so a re-ask at the SAME point can
    # be recognised as a reschedule rather than progress (see the loop fuse below).
    _offered_at[request_id] = computed
    if len(_offered_at) > MAX_SUPPRESSED_REQUESTS:
        _offered_at.pop(next(iter(_offered_at)), None)
    _state["offers"] += 1
    _state["offered_tokens"] += disk_tokens
    _state["blocks_offered"] += blocks
    _state["slots_allocated"] += needed
    # WHAT THE CLAIM COSTS EVERY CACHEABLE GROUP (item 7, 0.2.25). Priced with the
    # offer's FINAL `blocks`, after every clamp, so the table cannot describe a span
    # bigger than the one being made. On a model the gate lets through this is a list
    # of `restored: true` rows and reads as a receipt; the 00:08 shape - "g5 filled, a
    # boundary state block needed in each of g0-g3 and filled by nobody" - is what the
    # refusal path publishes now that the offer cannot happen here at all.
    required = _groups_required(_state["group_serviceability"], blocks)
    required["offered"] = True
    # THE LOADS ACTUALLY BUILT, per group, read off the merged tuple itself. The
    # gate's promise is "no claim without bytes for every group", and the operator's
    # acceptance test reads this to see it: `groups_required[].restored` says every
    # eligible group was restorable, and `loads` says how many REAL blocks (bytes on
    # disk behind them) and how many null entries (no bytes, by the engine's own
    # design) each group was handed. A row whose `real_blocks` is 0 for an eligible
    # group would be the 00:08 bug visible in one field, which is why it is published.
    loads = merged.get("loads") or {}
    required["loads"] = [{"group": int(group),
                          "contract": _clip(str(contracts.get(str(group))
                                                     or CONTRACT_POSITIONAL), 24),
                          "real_blocks": int((loads.get(str(group))
                                              or {}).get("real") or 0),
                          "null_blocks": int((loads.get(str(group))
                                              or {}).get("null") or 0)}
                         for group in groups["groups"][:16]]
    _state["last_groups_required"] = required
    # The pool pricing, restated with the offer's FINAL shape (the boundary-safe cut
    # may have shortened it again), so the figures beside `bounded_by` in a payload
    # describe the offer that happened and not the one first proposed.
    _state["last_offer_pool"] = dict(pool_pricing,
                                     jobs_offered=int(needed),
                                     bytes_offered=int(needed * price_bytes),
                                     blocks_offered=int(blocks),
                                     bounded_by=list(bounds))
    # THE OFFER, AS STRUCTURE: a forced-from-disk read and an ordinary one disagree
    # about exactly these numbers, so `last_offer` is what tells them apart in a
    # payload without reading the log.
    _state["last_offer"] = {
        "request_id": _clip(request_id, 48),
        "disk_authority": bool(disk_authority),
        "computed_tokens": int(computed),
        "ram_tokens_reported": int(ram_tokens_reported),
        "ram_tokens_used": int(ram_tokens),
        "start_block": int(start_block),
        "blocks": int(blocks),
        "disk_tokens": int(disk_tokens),
        "groups": list(groups["groups"])[:8],
        "positional_groups": positional_groups[:8],
        "boundary_groups": boundary_groups[:8],
        "run_blocks": int(run["blocks"]),
        # TRUE WHEN THE OFFER IS SHORTER THAN THE POSITIONAL RUN. Two ways that
        # happens and an operator must not have to tell them apart by arithmetic:
        # `_disk_run` walked further than every align-mode group could close (a state
        # file is missing at the tail), or a later clamp cut the offer below an
        # anchored end. Either way the number next to it is the length the attention
        # group alone could have claimed, which is the number that would have been the
        # 00:08 wrong answer.
        "run_reanchored": bool(run.get("clamped_to_anchor")) or bool(
            cut.get("clamped_from", 0) > cut.get("cut_to", 0)),
        "positional_run_blocks": int(run.get("positional_run") or 0),
        "groups_required": required["rows"][:16],
        "groups_unfilled": required["unfilled_group_ids"][:16],
        "loads": required["loads"][:16],
    }
    _note(f"offered {disk_tokens} disk tokens for {request_id[:48]} in {blocks} "
          f"span(s) across {len(groups['groups'])} group(s) "
          f"({len(positional_groups)} positional, {len(boundary_groups)} "
          f"boundary-state, {needed} CPU slot(s))"
          + (f" [disk authority: local hit of {ram_tokens_reported} tokens "
             "not consulted]" if disk_authority else ""))
    emit_witness()
    return (int(ram_tokens) + disk_tokens, True)


def _refresh_index() -> dict:
    """Let the scheduler learn what the ranks spilled since it last asked.

    Bounded inside `disk_index.refresh` (one stat per journal, a byte cursor, a
    line cap, a 1 s floor; the FIRST refresh of the process additionally runs the
    boot drain under `BOOT_DRAIN_MAX_BYTES` / `BOOT_DRAIN_DEADLINE_SECONDS`, so a
    cold boot with deep journals converges inside the first offer attempt instead
    of staying blind for a minute of one-share refreshes) and never raising there;
    the try/except is for a fork that renamed something under it.
    """
    try:
        out = disk_index.refresh()
        _state["index_refreshes"] += 1
        if not out.get("ok", True):
            _state["index_refresh_errors"] += 1
            return _fail(str(out.get("reason")))
        return _ok()
    except Exception as exc:  # noqa: BLE001 - an unreadable index offers nothing
        _state["index_refresh_errors"] += 1
        snapshot._error(exc, "disk_restore refresh")
        return _fail(f"the disk index refresh raised {type(exc).__name__}")


def _index_blindness() -> dict:
    """Why this index may not be able to answer "is it on disk?", from memory only.

    Two blindnesses, and the distinction is the whole point (the `entries: 146`
    incident, 2026-10-20): a DRAIN still in flight says "spilled but not yet
    indexed" - the named bytes WILL be read - while a FORFEITED history (the 0.2.22
    reader, or a boot of an older plugin) says "not knowable for the life of this
    process", because a cursor only ever advances. Both get a sentence, because
    both mean `no_disk_run` cannot be read as "never spilled". That is the exact
    blind spot the night cost hours on: after a jump `lag_bytes` is legitimately 0,
    so a check written only on `lag_bytes != 0` was silent about the one shape that
    mattered. Reads `disk_index.index_lag()` - memory, never a stat, never a scan -
    and never raises.
    """
    try:
        lag = disk_index.index_lag()
    except Exception as exc:  # noqa: BLE001 - a view is never worth a fault
        snapshot._error(exc, "disk_restore index blindness")
        return {"blind": False, "sentence": "", "lag": None}
    unread = lag.get("unread_journals") or {}
    names = "; ".join(f"{name}: {int(row)} unread byte(s)"
                      for name, row in list(unread.items())[:4])
    omitted = int(lag.get("unread_journals_omitted") or 0)
    if omitted > 0:
        names = (f"{names}; {omitted} more journal(s) unnamed"
                 if names else f"{omitted} journal(s) unnamed")
    behind = (not bool(lag.get("converged"))) or int(lag.get("lag_bytes") or 0) > 0
    jumps = int(lag.get("catch_up_jumps") or 0)
    skipped = int(lag.get("bytes_skipped") or 0)
    forfeited = jumps > 0 or skipped > 0
    if behind:
        sentence = (f"THE INDEX IS NOT CAUGHT UP: {int(lag.get('lag_bytes') or 0)} "
                    f"unread journal byte(s) over "
                    f"{int(lag.get('lag_journals') or 0)} journal(s) "
                    f"({names or 'no per-journal names measured yet'}); the drain "
                    "is still reading, spans living in an unread region are NOT "
                    "KNOWABLE to this index, and `no_disk_run` must therefore be "
                    "read as 'spilled but not yet indexed', never as 'never "
                    "spilled'.")
    elif forfeited:
        sentence = (f"THE INDEX FORFEITED HISTORY (catch_up_jumps: {jumps}, "
                    f"bytes_skipped: {skipped} bytes): the skipped regions of "
                    "these journals are NOT KNOWABLE for the life of this process "
                    "(a cursor only ever advances), so `no_disk_run` must be read "
                    "as 'spilled, possibly, but unread', never as 'never spilled'.")
    else:
        return {"blind": False, "sentence": "", "lag": lag}
    return {"blind": True, "sentence": sentence, "lag": lag}


def _allocate(pool, count: int) -> dict:
    """Destination CPU slots, through the pool API the store path uses.

    `get_new_blocks` RAISES when the count exceeds the free list
    (`core/block_pool.py:664-676`), so the free count is checked first and the call
    is still contained. NO touch(): the allocation is itself one reference, which
    the fork's `_free_pending_cpu_hit` releases and its
    `update_state_after_alloc` re-touches for the blocks it takes - adding a ref of
    our own would leak one per slot, forever.
    """
    if count <= 0:
        return _ok(blocks=[])
    try:
        blocks = list(pool.get_new_blocks(count))
    except Exception as exc:  # noqa: BLE001 - the pool says no with an exception
        snapshot._error(exc, "disk_restore get_new_blocks")
        return _fail(f"the CPU pool refused {count} blocks: {type(exc).__name__}")
    if len(blocks) != count:
        _release(pool, blocks)
        return _fail(f"the CPU pool returned {len(blocks)} of {count} blocks")
    for block in blocks:
        if _int(getattr(block, "block_id", None)) is None:
            _release(pool, blocks)
            return _fail("an allocated CPU block carries no usable block_id")
    return _ok(blocks=blocks)


def _release(pool, blocks) -> None:
    """Give slots back when a merge fails AFTER allocation. Tail-first, the order
    `manager._free_pending_cpu_hit` and the coordinator both use (freeing a chain
    root-first lets the root be evicted before the deeper blocks pinned for it)."""
    if pool is None or not blocks:
        return
    try:
        pool.free_blocks(list(reversed(list(blocks))))
    except Exception as exc:  # noqa: BLE001 - a failed undo is counted, not fatal
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore release")


def _release_fork_pin(manager, pending) -> dict:
    """Put back the host-RAM blocks the fork pinned for a hit we are discarding.

    WHY THIS IS NECESSARY, NOT OPTIONAL. `get_num_new_matched_tokens`
    (`manager.py:360-372`) does not merely return a number: on a hit it `touch()`es
    every CPU block it found and leaves the tuple in `_pending_cpu_hits[request_id]`.
    Exactly three things ever free it - `update_state_after_alloc`,
    `request_finished`, and the stale-pop at the top of the next lookup for the same
    id. A disk-authority offer OVERWRITES that entry with our own slots, so without
    this release those refs are never returned: one pinned span of CPU blocks per
    flagged request, forever, which is a slow way to empty the pool the restore path
    itself allocates its destinations from.

    The release goes through the FORK's own method, and the plugin never calls
    `free_blocks` on blocks it did not allocate - the rule `_allocate` states for the
    other direction. A manager without the method, or a release that raises, is a
    refusal at the door that has committed nothing yet; from the commit site below it
    can only be a counted, logged line, because the offer is already the fork's.
    """
    releaser = getattr(manager, "_free_pending_cpu_hit", None)
    if not callable(releaser):
        return _fail("this manager exposes no _free_pending_cpu_hit()", reason_kind=(
            "pin_release_unavailable"))
    try:
        releaser(pending)
    except Exception as exc:  # noqa: BLE001 - reported, never raised at the fork
        snapshot._error(exc, "disk_restore release fork pin")
        return _fail(f"_free_pending_cpu_hit raised {type(exc).__name__}",
                     reason_kind="pin_release_unavailable")
    return _ok()


def _identity_path(root: str, key_hex: str) -> dict:
    """The identity in path shape: the scheduler's half of the name.

    `build_slot_path` is the ONE name builder, so the key16/g<group> components are
    exactly the ones `disk_spill` wrote; only the digest half is the sentinel,
    because it is rank-local and the scheduler has no business guessing it.
    """
    named = disk_layout.build_slot_path(root, key_hex,
                                        IDENTITY_DIGEST_PLACEHOLDER)
    if not named["ok"]:
        return _fail(str(named["reason"]))
    return _ok(path=named["path"], key16=named["key16"],
               group_id=named["group_id"])


def _merge_into_pending(manager, request_id, ram_tokens, ram_computed, block_size,
                       groups, num_groups, allocated, keys, root,
                       contracts: dict | None = None,
                       null_block=None) -> dict:
    """Append our slots to the fork's per-group CPU-hit lists, BY CONTRACT.

    Shape contract with `manager.py:381-492`: element 0 is a sequence with ONE entry
    per group (so the fork's `for g in range(num_groups)` cannot IndexError), our
    slots go at the END of each eligible group's list (the fork consumes LEADING
    entries, so "RAM first, then disk" is the token order it will pair), and the tuple
    becomes `(merged_lists, ram_tokens + disk_tokens, num_computed_tokens)` with the
    third field untouched - the fork needs that local prefix to place the external
    blocks and must not have it re-derived.

    THE PART 0.2.27 ADDED, and it is the whole reason a GDN boundary restore is a
    restore rather than a wrong answer. The fork pairs entries with this loop
    (`manager.py:460-478`):

        for i, cpu_blk in enumerate(cpu_blocks_g):
            if cpu_blk.is_null:
                continue
            gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])

    so an entry is addressed by its INDEX and a null costs its index and nothing
    else. That is what makes the align-mode hit shape land correctly: a boundary group
    gets `[null] * (B - 1) + [one real state block]`, exactly the shape the fork's own
    `MambaManager.find_longest_cache_hit` builds
    (`single_type_kv_cache_manager.py:1591-1607`), and the real state is therefore
    DMA'd into `group_gpu_ids[start + B - 1]` - the state at the END of the claimed
    run, which is the only state that describes those tokens. Appending the state at
    position 0 instead, or omitting the nulls, would aim a 31 MB recurrent state at
    the wrong block and answer confidently and wrongly. So:

      * `positional` group      -> B real blocks, one per span (unchanged);
      * `boundary_state` group  -> (B - 1) of the pool's OWN `null_block`, then the
                                   one real state block, and the real block must be
                                   the pair `_disk_run` put on the LAST span or this
                                   function refuses.

    A RAM hit whose per-group length disagrees with its own token count is REFUSED:
    the whole merge rests on "leading entries = the RAM hit, one per whole group
    block", and if that is not true the pairing would put our bytes at GPU offsets
    belonging to someone else's tokens.
    """
    declared = contracts or {}
    ram_blocks = ram_tokens // block_size
    spans = len(keys)
    # How many REAL (bytes-backed) entries each group is being handed, and at which
    # span each of them belongs. Counted from `keys`, which `_disk_run` built under
    # the contract rule, so a disagreement between the two is caught here rather than
    # reaching the fork.
    real_at: dict = {}
    for position, span in enumerate(keys):
        for group, _key_hex in span:
            real_at.setdefault(int(group), []).append(position)
    wanted = sum(len(positions) for positions in real_at.values())
    if len(allocated) != wanted:
        return _fail(f"{len(allocated)} slots were allocated for {wanted} named "
                     f"block(s) across {spans} span(s) and {len(groups)} group(s)",
                     reason_kind="ram_shape")
    pending = None
    try:
        table = getattr(manager, "_pending_cpu_hits", None)
        if table is not None:
            pending = table.get(request_id)
    except Exception as exc:  # noqa: BLE001 - an unreadable table is "no RAM hit"
        snapshot._error(exc, "disk_restore read pending")
    if ram_tokens > 0:
        if not isinstance(pending, tuple) or len(pending) < 3:
            return _fail(f"the fork reported {ram_tokens} host-RAM tokens but left "
                         f"no _pending_cpu_hits tuple for {request_id[:48]}, so "
                         "there is nothing to merge into", reason_kind="ram_shape")
        first = pending[0]
        if not isinstance(first, (list, tuple)):
            return _fail("the fork's pending CPU hit is not a per-group sequence",
                         reason_kind="ram_shape")
        if len(first) != num_groups:
            return _fail(f"the fork's pending CPU hit has {len(first)} group "
                         f"entries, not {num_groups}", reason_kind="ram_shape")
        lists = [list(entry or []) for entry in first]
    else:
        lists = [[] for _ in range(num_groups)]
    for group in groups:
        if group >= len(lists):
            return _fail(f"group {group} is outside the fork's {len(lists)} group "
                         "lists", reason_kind="ram_shape")
        if len(lists[group]) != ram_blocks:
            return _fail(f"group {group}'s host-RAM hit is {len(lists[group])} "
                         f"blocks but the offer says {ram_blocks}, so leading "
                         "entries are not the RAM hit", reason_kind="ram_shape")
    identities: list = []
    loads: dict = {}
    # First the boundary groups' interior entries, so the one real state block that
    # arrives with the LAST span lands after them. The fork pairs by INDEX and skips
    # nulls with the index preserved (`manager.py:460-478`), which is what puts that
    # state on `group_gpu_ids[start + B - 1]`.
    for group in groups:
        if (declared.get(str(group)) or CONTRACT_POSITIONAL) != CONTRACT_BOUNDARY:
            continue
        positions = real_at.get(int(group), [])
        if len(positions) != 1:
            return _fail(f"group {group} is an align-mode boundary group but the run "
                         f"named {len(positions)} state block(s) for it; the fork's "
                         "hit shape is exactly one, so merging this run would aim "
                         "the state at the wrong GPU block", reason_kind="ram_shape")
        if positions[0] != spans - 1:
            return _fail(f"group {group}'s boundary state is named for span "
                         f"{positions[0]} of {spans}, not the last one, so the state "
                         "would be paired with a GPU block that is not the run's "
                         "end", reason_kind="ram_shape")
        if null_block is None:
            return _fail("an align-mode group needs the CPU pool's null_block for "
                         "its interior entries and none was supplied",
                         reason_kind="no_null_block")
        for _ in range(max(0, spans - 1)):
            lists[group].append(null_block)
    for group in groups:
        contract = declared.get(str(group)) or CONTRACT_POSITIONAL
        loads[str(group)] = {
            "real": len(real_at.get(int(group), [])),
            "null": (max(0, spans - 1) if contract == CONTRACT_BOUNDARY else 0),
            "contract": contract}
    # Then the real blocks, SPAN-MAJOR, exactly as they were allocated: pairing an
    # allocation with the wrong identity is the one mistake a content-addressed store
    # cannot survive, so the order that named them is the order that pairs them.
    index = 0
    for position, span in enumerate(keys):
        for group, key_hex in span:
            named = _identity_path(root, key_hex)
            if not named["ok"]:
                return _fail(f"the identity path for {key_hex[:20]} could not be "
                             f"built: {named['reason']}",
                             reason_kind="path_build_failed")
            block = allocated[index]
            index += 1
            lists[int(group)].append(block)
            identities.append({"group": int(group), "key": key_hex,
                               "cpu_block_id": int(block.block_id),
                               "path": named["path"], "span": int(position),
                               "contract": _clip(str(declared.get(str(group))
                                                          or CONTRACT_POSITIONAL),
                                                 24)})
    if index != wanted:
        return _fail(f"{index} identities built for {wanted} slots",
                     reason_kind="ram_shape")
    hit_length = ram_tokens + len(keys) * block_size
    return _ok(pending=(tuple(lists), hit_length, ram_computed),
               identities=identities, hit_length=hit_length, loads=loads)


def _register_load(request_id: str, merged: dict, disk_tokens: int,
                   span: str | None = None) -> None:
    """Remember the load until the worker reports it, or the ledger expires it.

    The mapping `cpu_block_id -> path` is what later fills `load_paths`, and its
    lifetime is deliberately SHORTER than the fork's ownership of those slots:
    `build_load_paths` drops every mapping whose request has left
    `_reqs_to_load` BEFORE it emits anything, so a slot the fork has already freed
    can never be attributed a file in the step it is re-used. That ordering is the
    guard against the one corruption this path could cause - reading a file into a
    slot that has become someone else's host-RAM source block.
    """
    while len(_loads) >= MAX_TRACKED_LOADS:
        _evict_oldest_load()
    _loads[request_id] = {
        "request_id": request_id,
        "tokens": int(disk_tokens),
        "cpu_block_ids": [int(item["cpu_block_id"])
                          for item in merged["identities"]],
        "identities": list(merged["identities"]),
        "offered_mono": round(time.monotonic(), 6),
        # The span this offer was built from, so a failure later can fuse the SPAN
        # and not only the request that happened to ask for it.
        "span": str(span)[:80] if span else None,
        "deadline_mono": round(time.monotonic()
                               + RESTORE_DEADLINE_SECONDS, 6),
        "adopted": False,
        "emitted": False,
        # THE GPU SIDE OF OUR OWN CLAIM (0.2.25). Filled in by `_build_load_paths`,
        # which is the one moment the scheduler sees a CPU slot we allocated paired
        # with the GPU block the fork will DMA it into. Until now the ledger knew only
        # the CPU half, so "were the GPU blocks actually written?" had no answer at
        # all: the fork's completion report names a REQUEST, and this is the only
        # place that couples it to block ids.
        "gpu_block_ids": [],
    }
    for item in merged["identities"]:
        if len(_slot_paths) >= MAX_TRACKED_SLOTS:
            _note(f"the slot ledger is at its {MAX_TRACKED_SLOTS} cap: a span of "
                  "this offer will not be findable (a missed restore, never a "
                  "wrong one)")
            break
        _slot_paths[int(item["cpu_block_id"])] = (request_id, item["path"])


def _evict_oldest_load() -> None:
    """Drop the oldest tracked load (a dropped ledger entry costs a restore)."""
    if not _loads:
        return
    oldest = next(iter(_loads))
    # Remembered like every other report-less release: a cap eviction is the ledger
    # giving up on a load, and a `finished_recving` that arrives for it afterwards
    # still deserves a name rather than a silent `continue`.
    _remember_release(oldest, _loads[oldest], "the ledger is at its cap")
    _drop_load(oldest, "the ledger is at its cap")


def _drop_load(request_id: str, why: str) -> None:
    record = _loads.pop(request_id, None)
    if record is None:
        return
    for cpu_block_id in record.get("cpu_block_ids") or ():
        held = _slot_paths.get(int(cpu_block_id))
        if held is not None and held[0] == request_id:
            _slot_paths.pop(int(cpu_block_id), None)
    _state["loads_released"] += 1
    _note(f"{request_id[:48]}: {why}")


def _remember_release(request_id: str, record: dict, how: str) -> None:
    """Record HOW a load left the ledger without a report, so a late report has a
    witness.

    A ledger release is not a judgement about whether the bytes landed, and two of the
    doors out of the ledger are decisions about the LEDGER and not about the data: the
    fork stopped owning the request (`_reqs_to_load` no longer names it - which is
    exactly the shape of the fork's own `update_state_after_alloc ... no external
    tokens but get_num_new_matched_tokens() unexpectedly recorded a pending CPU hit;
    releasing the stale pin` line, logged by the live engine in 10 of every 100 of its
    lines), and our own deadline expired. In both cases the worker may ALREADY have
    read the files, and the fork may call the load finished afterwards - and until
    0.2.25 that report reached `note_reported`, found `_loads.get(request_id) is None`
    and was `continue`d away with no counter moved at all. That is how a boot with 36
    applied reads reported `loads_released=1, restorations=0` and left the on-call
    reader to INFER that the report had arrived late rather than never.

    The evidence stored is the strongest available AT RELEASE: `adopted` (our CPU slots
    reached a worker's `load_cpu_blocks`) and whether any disk read had been applied in
    this boot (`_a_read_was_applied`, boot-wide because no channel could make it
    per-request). That is deliberately NOT a claim that this load's bytes landed; it is
    the same evidence `restorations` is allowed to use, recorded while it was still
    readable. Bounded oldest-first, like every other table here.
    """
    try:
        adopted = bool(record.get("adopted"))
        with_reads = adopted and _a_read_was_applied()
        gpu_blocks = len(record.get("gpu_block_ids") or ())
        if len(_aged_out) >= MAX_SUPPRESSED_REQUESTS:
            _aged_out.pop(next(iter(_aged_out)), None)      # oldest first
        _aged_out[request_id] = {
            "how": _clip(how, 160),
            "adopted": adopted,
            "reads_applied": with_reads,
            "tokens": _int(record.get("tokens")) or 0,
            "gpu_blocks": gpu_blocks,
            "released_mono": round(time.monotonic(), 6),
        }
        _state["aged_out_loads"] += 1
        if with_reads:
            _state["aged_out_with_applied_reads"] += 1
        _state["disk_gpu_blocks_released_unreported"] += gpu_blocks
        _state["last_aged_out"] = _clip(
            f"{request_id[:48]}: released because {how}; "
            f"adopted={str(adopted).lower()} "
            f"disk_reads_applied={str(with_reads).lower()} "
            f"gpu_blocks_mapped={gpu_blocks}", 240)
    except Exception as exc:  # noqa: BLE001 - a tombstone never breaks a sweep
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore remember release")


def _note_late_report(request_id: str) -> bool:
    """A completion report for a load the ledger already released: name it, don't
    drop it.

    THE REPORTING BUG THIS EXISTS TO FIX (0.2.25). The measured window says it is not a
    deadline that is too short: a disk read on the live box completes in 9-30 ms
    (`read_seconds_max` 0.018 at 00:08, 9-12 ms waited of a 139 ms settle budget) while
    `RESTORE_DEADLINE_SECONDS` is 45 - about fifteen hundred times the read it is
    guarding. So an expiry means the REPORT never arrived inside 45 s, and the
    protective answer is to lengthen nothing: the fuse has to stay fast, because a load
    that does not report IS the shape of the loop that starved the box on 2026-10-05.
    What was wrong is only that the late truth was thrown away.

    A tombstone that says "adopted, and a disk read had been applied" plus a late
    `finished_recving` is as close to "the bytes were on those GPU blocks" as this
    module is ever going to get for a load it already released, so it counts
    `restored_after_release` and its mapped GPU blocks join
    `disk_gpu_blocks_dma_verified` (the fork DID report them complete). It never
    touches `restorations`, which keeps meaning "reported while the ledger held it".
    """
    tombstone = _aged_out.pop(request_id, None)
    if tombstone is None:
        return False
    _state["late_finished_reports"] += 1
    gpu_blocks = int(tombstone.get("gpu_blocks") or 0)
    if tombstone.get("reads_applied"):
        _state["restored_after_release"] += 1
        _state["disk_gpu_blocks_dma_verified"] += gpu_blocks
    _logger.warning(
        "kvtier-dev-plugin: DISK[restore] late finished report for %s: the ledger "
        "released it (aged_out=%s, adopted=%s, reads_applied=%s, gpu_blocks=%d, "
        "tokens=%d) - a restoration the ledger could not count, now counted by name",
        request_id[:48], _clip(tombstone.get("how"), 80),
        tombstone.get("adopted"), tombstone.get("reads_applied"), gpu_blocks,
        int(tombstone.get("tokens") or 0))
    return True


def _sweep_loads(manager) -> int:
    """Forget loads the fork no longer owns, or that outlived their deadline.

    Called at the top of `build_load_paths` - i.e. BEFORE any path is emitted - and
    on every completion report. A request that was aborted, preempted-away or
    simply never adopted leaves a ledger entry whose CPU slots the fork has already
    returned to the pool; keeping that mapping is the only way this module could
    ever point a read at a slot it does not own, so the sweep is a correctness
    mechanism and not housekeeping.
    """
    dropped = 0
    now = time.monotonic()
    try:
        live = getattr(manager, "_reqs_to_load", None)
    except Exception as exc:  # noqa: BLE001 - an unreadable table stops the sweep
        snapshot._error(exc, "disk_restore sweep live")
        live = None
    for request_id in list(_loads):
        record = _loads.get(request_id)
        if record is None:
            continue
        if live is not None and request_id not in live:
            # THE FORK RELEASED THE CLAIM UNDERNEATH US. `update_state_after_alloc`
            # returns early (releasing its own pin) whenever it is handed no external
            # tokens, and then `_reqs_to_load` never holds this request - so the load
            # state is gone, ours is stale, and the mapping has to go with it. The
            # tombstone is what keeps a later `finished_recving` from vanishing; the
            # gate in `_eligible_groups` is what stops us MAKING this claim at all on
            # a model whose groups cannot all be restored.
            _remember_release(request_id, record,
                              "the fork no longer has a load state for it "
                              "(aborted, never adopted, or its own stale-pin "
                              "release)")
            _drop_load(request_id, "the fork no longer has a load state for it "
                                   "(aborted, or never adopted)")
            dropped += 1
        elif now >= float(record["deadline_mono"]):
            _state["watchdog_fires"] += 1
            # A load that never reported inside its deadline is a failed load: the
            # fork has no timeout of its own for WAITING_FOR_REMOTE_KVS, so the only
            # thing standing between this request and another offer is this fuse.
            #
            # THE DEADLINE VALUE IS NOT THE BUG (0.2.25). A read completes in 9-30 ms
            # on this box and the budget is 45 s, so this branch cannot be firing
            # because the disk is slow; it fires because no report arrives. Lengthening
            # it would only slow the fuse that stops the re-offer loop, so the value
            # stays and the REPORTING is what changed: the release is remembered by
            # `_remember_release`, so a `finished_recving` that turns up afterwards is
            # counted as `late_finished_reports` / `restored_after_release` instead of
            # being dropped in silence.
            suppress_load(request_id, "its load outlived the restore deadline "
                                      "without a report",
                          span=_span_key_of(record))
            _remember_release(request_id, record,
                              "its ledger deadline expired without a report")
            _drop_load(request_id, "its ledger deadline expired without a report")
            dropped += 1
    if dropped:
        _state["stale_slots_dropped"] += dropped
    return dropped


# --- the scheduler's door 2: name the files, index-aligned --------------------

def build_load_paths(manager, metadata) -> dict:
    """Fill `metadata.load_paths` for the fork's own `load_cpu_blocks`.

    THE LENGTH RULE IS THE CONTRACT: one entry per load entry, always, with ""
    where nothing is on disk. A list that merely omitted the un-restorable entries
    would mis-align every index the worker resolves, and a mis-aligned index is a
    file read into the wrong slot - the one failure mode this whole design exists to
    make unreachable. So the length is asserted on the way out and a violation is
    counted and blanked rather than shipped.

    Mutates the metadata in place and returns a report; the WRAPPER returns the
    fork's object untouched either way, so a refusal here is invisible to the fork.
    """
    _sweep_loads(manager)
    try:
        return _build_load_paths(metadata)
    except Exception as exc:  # noqa: BLE001 - a step without paths is a normal step
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore build_load_paths")
        return _fail(f"the path emission raised {type(exc).__name__}")


def _build_load_paths(metadata) -> dict:
    cpu_blocks = getattr(metadata, FIELD_LOAD_CPU, None)
    if not isinstance(cpu_blocks, (list, tuple)):
        return _ok(emitted=False, reason="the step carries no load_cpu_blocks")
    paths = [""] * len(cpu_blocks)
    if not cpu_blocks:
        # Publish the (empty) field anyway: the worker's test must be "does any
        # entry name a file", never "does this attribute exist on this boot".
        _set_load_paths(metadata, paths)
        return _ok(emitted=False, entries=0, reason="the step has no load entries")
    named = 0
    if len(cpu_blocks) > MAX_PATHS_PER_STEP:
        _note(f"a step carried {len(cpu_blocks)} load entries; the cap is "
              f"{MAX_PATHS_PER_STEP}, so nothing was attributed")
    elif _slot_paths:
        # THE GPU HALF, read alongside the CPU half of the same index. The fork's own
        # two lists are index-aligned by construction (the worker's `bind_connector_
        # metadata` refuses a step where they are not: "load_cpu_blocks has N and
        # load_gpu_blocks has M"), so a step whose lists disagree says NOTHING about
        # block ids and is skipped rather than guessed at - a wrong pairing here would
        # attribute a finished DMA to a block this module never wrote.
        gpu_blocks = getattr(metadata, FIELD_LOAD_GPU, None)
        gpu_aligned = (isinstance(gpu_blocks, (list, tuple))
                       and len(gpu_blocks) == len(cpu_blocks))
        for index, value in enumerate(cpu_blocks):
            cpu_block_id = _int(value)
            if cpu_block_id is None:
                continue
            held = _slot_paths.get(cpu_block_id)
            if held is None:
                continue
            paths[index] = held[1]
            named += 1
            record = _loads.get(held[0])
            if record is not None:
                record["emitted"] = True
                if gpu_aligned:
                    gpu_id = _int(gpu_blocks[index])
                    mapped = record.get("gpu_block_ids")
                    if (gpu_id is not None and isinstance(mapped, list)
                            and gpu_id not in mapped):
                        mapped.append(gpu_id)
    if len(paths) != len(cpu_blocks):
        # Unreachable by construction; counted rather than trusted.
        _state["path_length_violations"] += 1
        _set_load_paths(metadata, [""] * len(cpu_blocks))
        return _fail("the length rule was violated, so every entry was blanked")
    _set_load_paths(metadata, paths)
    _state["path_entries"] += named
    _state["path_entries_empty"] += len(paths) - named
    if named:
        _state["paths_emitted_steps"] += 1
        _mark_adopted()
    return _ok(emitted=bool(named), entries=len(paths), named=named)


def _set_load_paths(metadata, paths) -> None:
    try:
        setattr(metadata, FIELD_LOAD_PATHS, paths)
    except Exception as exc:  # noqa: BLE001 - a frozen metadata is a no-op
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore set load_paths")


def _mark_adopted() -> None:
    """Count the offers that reached a worker as ADOPTED.

    "Adopted" here means the scheduler put our CPU slot ids into a step's
    `load_cpu_blocks`, which is the only observable proof that the fork accepted the
    external tokens: otherwise the live box's adoption floor (whole scheduler blocks
    only) makes an offer vanish between `get_num_new_matched_tokens` and
    `update_state_after_alloc` with nothing to measure.
    """
    for record in list(_loads.values()):
        if record["adopted"] or not record["emitted"]:
            continue
        record["adopted"] = True
        _state["adopted"] += 1
        _state["adopted_tokens"] += int(record["tokens"])


# --- the scheduler's door 3: completions, and the ledger's expiry -------------

def _fuse_or_spare(request_id, span, *, where: str, why: str) -> dict:
    """WHAT A FAILED LOAD COSTS, decided from the failure kinds the code can see.

    THE BUG THIS CLOSES (2026-10-06). Every failure report used to call `suppress_load`
    - black-holing the REQUEST and fusing the SPAN for the rest of the boot - whatever
    the failure was, while the reason string it handed that call already contained
    `sorted(_recent_failure_kinds())`. The code had the answer in its hand and decided
    without looking, so a span that lost a 135.8 ms race in one group of one span cost a
    28,583-token recompute AND the permanent fuse (`suppressed_requests: 3`).

    THE RULE, in the operator's words: "give it up to 5 s to read; only when something
    bad happens do we ignore". A timing loss costs the tokens (the engine still
    recomputes them - see `LOAD_TIMING_FAILURES` for what the engine does with the
    reported blocks) and does NOT cost the next offer. A data failure keeps the fuse
    exactly where it was, and so does the UNKNOWN case: a loss nobody diagnosed is not
    proof of a race, because the scheduler learns a worker's kinds from the rank file
    and that transport lags (`PROOF_CACHE_SECONDS`). The first loss of an episode
    therefore usually fuses and the losses after it do not - which is the honest shape
    of what this process can know, and it is published in `last_fuse_decision`.

    A spared request is not unsentinelled: it gets exactly one re-offer through
    `_grant_race_reoffer`, spent when used, so the loop that starved the box stays
    bounded at two offers per request however the losses fall.
    """
    kinds = sorted(_attributable_failure_kinds())
    verdict = classify_failure_kinds(kinds)
    fused = verdict != "timing"
    if fused:
        _state["fuses_on_data_kinds" if verdict == "data"
               else "fuses_on_unknown_kinds"] += 1
        suppress_load(request_id, f"{why} (failure kinds visible: "
                                  f"{kinds or ['none seen']}; classified {verdict})",
                      span=span)
    else:
        _state["races_not_fused"] += 1
        _grant_race_reoffer(request_id)
    _state["last_fuse_decision"] = (
        f"req={str(request_id)[:48]} where={where} verdict={verdict} "
        f"kinds={kinds or ['none seen']} "
        + ("fused (no further offer)" if fused
           else "spared: one re-offer granted, span not fused"))[:400]
    return {"verdict": verdict, "fused": fused, "kinds": kinds,
            "span": span, "request_id": request_id}


def note_reported(manager, connector_output) -> dict:
    """A load was reported finished: those disk bytes are now live on the GPU.

    THIS is the counter the witness needs (`restorations`): not "we offered", not
    "we read a file", but a request whose disk-sourced slots were DMA'd and whose
    load the worker reported complete. The counterfactual is inherent - those tokens
    were beyond the GPU hit (the scheduler computed it and came up short) and beyond
    the host-RAM hit (this module only extends a hit that ran out) - so a
    restoration is a recompute that did not happen.

    TWO THINGS IT ALSO DOES, both learned the hard way on 2026-10-05:

    * It reads the FAILURE halves of the same report (`failed_recving`, and
      `invalid_block_ids` when the fork names blocks but no request) and burns the
      request and the span. A report that says "this load did not land" has to cost
      the next offer, or the reschedule re-offers the same span forever.
    * It will not call a finished load a restoration without a read behind it. The
      fork reports a load finished for its OWN host-RAM blocks too, so
      `finished_recving` on a request whose disk slots nobody filled looked exactly
      like success: the live box counted 123,161 "restorations" with
      `reads_applied=0` in the same line.
    """
    out = {"released": 0, "restorations": 0, "failures": 0, "no_read": 0,
           "races_not_fused": 0}
    try:
        reported = {str(item) for item in
                    (getattr(connector_output, "finished_recving", None) or ())}
        failed = {str(item) for item in
                  (getattr(connector_output, "failed_recving", None) or ())}
        invalid = _invalid_block_ids(connector_output)
        for request_id in sorted(failed):
            # EXACT attribution: the fork names the request whose load failed.
            record = _loads.get(request_id)
            _note_load_failure(
                "failed_recving", where="report",
                reason="the engine named this request's KV load as failed "
                       f"(newest local/rank failure kinds: "
                       f"{sorted(_recent_failure_kinds()) or ['none seen']})",
                request_id=request_id, span=_span_key_of(record),
                provenance=False)
            # A RACE AND A FAULT NO LONGER COST THE SAME (0.2.32): see
            # `_fuse_or_spare`. A spared request closes its ledger with `_drop_load`
            # rather than `_burn_load`, so it is offered again exactly once.
            decision = _fuse_or_spare(request_id, _span_key_of(record),
                                      where="failed_recving",
                                      why="the engine reported this request's KV "
                                          "load failed")
            if decision["fused"]:
                # THE LEDGER MUST CLOSE. Until 0.2.23 this line was `_loads.pop(...)`,
                # a hand-rolled duplicate of `_drop_load` that skipped its two side
                # effects: the `loads_released` counter and the release of this load's
                # `_slot_paths` entries. On a box where EVERY failure arrives in this
                # shape, `loads_released` sat at 0 forever while the fork's
                # `_reqs_to_load` drained - which is exactly the on-call reading "the
                # load was never registered at all" instead of "it was registered and
                # burned", and it left slot->path mappings naming a request that no
                # longer had a load.
                _burn_load(request_id, "the engine reported this request's KV load "
                                       "failed")
            else:
                _drop_load(request_id, "the engine reported this request's KV load "
                                       "failed and the loss was a race: the request "
                                       "keeps its next offer")
                out["races_not_fused"] += 1
            out["failures"] += 1
        if invalid and not failed:
            # The block-id-only shape: we own the load-error report, so on a step
            # where a disk load was in flight these are ours. Deliberately
            # over-broad - a missed restore costs a recompute, a missed failure
            # feeds the loop that starved the box.
            _note_load_failure(
                "invalid_block_ids", where="report",
                reason=(f"{invalid} GPU block id(s) came back as load errors on a "
                        "step carrying disk slots "
                        "(blocks: "
                        f"{(_last_invalid_blocks(8))}; "
                        "newest local/rank failure kinds: "
                        f"{sorted(_recent_failure_kinds()) or ['none seen']})"),
                gpu_block_id=(_last_invalid_blocks(1) or [None])[0],
                provenance=False)
            for request_id in list(_loads):
                if _loads[request_id].get("adopted"):
                    # A RACE AND A FAULT NO LONGER COST THE SAME (0.2.32). This is the
                    # shape the live box actually produced (66 requests with
                    # `invalid_block_ids=1..3` and an empty `failed_recving`), so this
                    # is where tonight's 12 ms race became a boot-length fuse.
                    decision = _fuse_or_spare(
                        request_id, _span_key_of(_loads.get(request_id)),
                        where="invalid_block_ids",
                        why=f"{invalid} GPU block id(s) came back as load errors on a "
                            f"step that carried this request's disk slots")
                    if decision["fused"]:
                        # The suppression decision for a FAULT is unchanged; only the
                        # bookkeeping was ever wrong (`_loads.pop` meant
                        # `loads_released` never moved by one).
                        _burn_load(request_id,
                                   f"{invalid} GPU block id(s) reported as load errors "
                                   "on a step carrying this request's disk slots")
                    else:
                        _drop_load(request_id,
                                   f"{invalid} GPU block id(s) reported as load errors "
                                   "on a step carrying this request's disk slots, and "
                                   "the loss was a race: the request keeps its next "
                                   "offer")
                        out["races_not_fused"] += 1
                    out["failures"] += 1
        for request_id in sorted(reported):
            if request_id in _burned:
                # A report that arrives AFTER the load was burned. Nothing can undo
                # the burn (the request is fused, so no re-offer can follow it), but
                # the report is not nothing either: it is the fork calling a load
                # finished whose bytes this plugin knows it never delivered. That is
                # `no_read`'s definition, and dropping it silently would leave
                # `no_read` under-counting the one shape of lie this counter exists
                # to catch. It is NOT a release - the load was already released when
                # it burned - so `released` stays untouched.
                _state["no_read"] += 1
                out["no_read"] += 1
                _burned.pop(request_id, None)
                continue
            record = _loads.get(request_id)
            if record is None:
                # THE VANISH, CLOSED (0.2.25). This branch used to be a bare
                # `continue`, which is how 36 applied reads and one finished report
                # produced `restorations=0` and nothing else in the payload: the
                # ledger entry had already been released (by the fork's own stale-pin
                # release, or by the deadline) and the report that proved the bytes
                # were there arrived after it. `_note_late_report` matches the id to
                # the tombstone the release left and counts the outcome by name.
                _note_late_report(request_id)
                continue
            if request_id in failed:
                _drop_load(request_id, "the worker reported it failed")
                out["released"] += 1
                continue
            if not record["adopted"]:
                _drop_load(request_id, "the worker reported it finished without the "
                                       "slots ever reaching a step")
                out["released"] += 1
                continue
            if not _a_read_was_applied():
                # The honest version of what the live box called a restoration.
                _state["no_read"] += 1
                out["no_read"] += 1
                suppress_load(request_id, "the load was reported finished with no "
                                          "applied disk read behind it",
                              span=_span_key_of(record))
                _drop_load(request_id, "reported finished with NO applied disk "
                                       "read: counted as no_read, not as a "
                                       "restoration")
                out["released"] += 1
                continue
            _state["restorations"] += 1
            out["restorations"] += 1
            # A SUCCESS CLEARS THE ALLOWANCE (0.2.32): the race allowance exists to give
            # a request ONE more chance at the bytes it lost, and a request whose load
            # landed has no use for it. Freeing the row is also what keeps the table
            # bounded in a long boot that offers a great many requests.
            _race_reoffer.pop(request_id, None)
            # THE BLOCKS, NOT JUST THE REQUEST (0.2.25). This is the one place the
            # plugin's own claim ("these GPU blocks are the ones our disk span
            # feeds") and the fork's own verdict ("this request's load is DMA'd")
            # are in the same hand, so it is the only place that can count them
            # together. A load that was never emitted to a step has no gpu ids
            # mapped and adds nothing, which is the point: an offer cannot inflate
            # it, only an adopted load reported complete can.
            _state["disk_gpu_blocks_dma_verified"] += len(
                record.get("gpu_block_ids") or ())
            _drop_load(request_id, "the worker reported its load finished")
            out["released"] += 1
        _sweep_loads(manager)
        if out["restorations"] or out["failures"]:
            emit_witness()
        return out
    except Exception as exc:  # noqa: BLE001 - a completion never breaks a step
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore note_reported")
        return _fail(f"the completion report raised {type(exc).__name__}")


def _invalid_block_ids(connector_output) -> int:
    """How many GPU blocks this step reported as load errors. -1 if unreadable.

    AND WHICH ONES. The count is what the burn decision has always used, but the IDS
    were thrown away - which is why "were these OUR blocks, and which of our spans do
    they belong to?" had no answer in any payload while the plugin was the only thing
    that knew the mapping. They are remembered (bounded, clipped to ints) beside the
    count; nothing about the decision changes.
    """
    try:
        ids = getattr(connector_output, "invalid_block_ids", None)
        if not ids:
            return 0
        seen = []
        for item in list(ids)[:MAX_TRACKED_SLOTS]:
            value = _int(item)
            if value is not None:
                seen.append(int(value))
        _state["invalid_block_ids_seen"] = int(_state["invalid_block_ids_seen"]) \
            + len(list(ids))
        _state["last_invalid_block_ids"] = {
            "count": len(list(ids)),
            "blocks": seen[:64],
            "truncated": len(seen) > 64,
            "ts": round(time.time(), 6)}
        return len(ids)
    except Exception as exc:  # noqa: BLE001 - an unreadable report says nothing
        snapshot._error(exc, "disk_restore invalid_block_ids")
        return -1


def _span_key_of(record) -> str | None:
    """The span an in-flight load was built from, for the span half of the fuse."""
    if not record:
        return None
    try:
        span = record.get("span")
        return str(span)[:80] if span else None
    except Exception as exc:  # noqa: BLE001 - no key is only a weaker fuse
        snapshot._error(exc, "disk_restore load span")
        return None


def _a_read_was_applied() -> bool:
    """Was there any evidence at all that a disk byte reached a CPU slot for this?

    Two sources, because the scheduler process and the worker process are different
    processes and the scheduler NEVER reads a file itself: its own `reads_applied`
    stays 0 by construction, and the worker publishes its count in its rank file.
    So the scheduler-side evidence is (this process's counter) OR (the sum the live
    rank files reported at the last proof read). Neither is per-request - there is no
    channel that could make it per-request - so this is the weakest statement that
    still separates "disk reads are happening on this box" from "nothing has ever
    been read", which is precisely the distinction a witness line has to get right.

    A boot where workers serve reads normally keeps this true; a boot where the
    scheduler offers and the workers refuse (2026-10-05) keeps it false forever, and
    `restorations` stays 0 and the witness says REFUTED instead of CONFIRMED.
    """
    return applied_reads_evidence()["reads_applied"] >= 1


# --- the worker's doors -------------------------------------------------------

def _own_identity(worker) -> dict:
    """THIS rank's digest and slot width - the half of the name only a worker knows.

    `disk_store.adopt_fingerprint` is the ONE builder (it reads the slot width off
    this process's own tensors and the rank id off `worker_diag`), so the restore
    path cannot name a file a spill could never have written. A second digest site
    is exactly what `disk_spill.plan_slot_write`'s docstring refuses to allow.
    """
    try:
        store = disk_store.counters()
        digest = store.get("geometry_digest") or disk_spill.geometry_digest()
        fingerprint = store.get("fingerprint") or {}
        width = _int(fingerprint.get("total_bytes_per_block"))
        if not digest or not width:
            adopted = disk_store.adopt_fingerprint(worker)
            if not adopted.get("adopted"):
                return _fail(str(adopted.get("reason")),
                             reason_kind="fingerprint_incomplete")
            store = disk_store.counters()
            digest = store.get("geometry_digest")
            width = _int((store.get("fingerprint") or {})
                         .get("total_bytes_per_block"))
        if not digest:
            return _fail("the fingerprint was adopted but no digest came out of it",
                         reason_kind="fingerprint_incomplete")
        if not width or width <= 0:
            return _fail("this rank's slot width is unknown, so no read has an "
                         "expected size to enforce",
                         reason_kind="fingerprint_incomplete")
        return _ok(digest=str(digest), bytes=int(width))
    except Exception as exc:  # noqa: BLE001 - no identity, no read
        snapshot._error(exc, "disk_restore own identity")
        return _fail(f"the fingerprint build raised {type(exc).__name__}",
                     reason_kind="fingerprint_incomplete")


def _resolve_worker_path(scheduler_path, digest: str) -> dict:
    """Re-derive THIS rank's path from the identity the scheduler sent.

    The scheduler is authoritative about the IDENTITY (key16 + group: it read the
    engine's own block hashes) and the worker is authoritative about the GEOMETRY
    (the digest, which covers this rank's slot width). `_identity_path` sent the
    digest half as the sentinel, so the normal case is "replace the sentinel". A
    path carrying a REAL digest is the cross-check: if it is not ours, two sites
    computed the same field and disagree, which is a counted failure and never a
    silent choice of either one.

    The name is rebuildable from key16 + group alone because
    `build_slot_path`/`split_block_key` truncate the digest to KEY_DIGEST_CHARS on
    the way in: `key16 + 8 hex of group` re-splits to the same key16 and the same
    group, hence exactly the file name the spill wrote.
    """
    parsed = disk_layout.parse_path(scheduler_path)
    if not parsed["ok"] or not parsed.get("ours"):
        return _fail(f"the scheduler sent a name this plugin could not have "
                     f"written: {parsed.get('reason')}",
                     reason_kind="bad_path")
    if parsed.get("partial"):
        return _fail("the scheduler sent an in-flight .part name",
                     reason_kind="bad_path")
    key16 = parsed.get("key16")
    group_id = parsed.get("group_id")
    if not isinstance(key16, str) or not _plain_int(group_id):
        return _fail("the scheduler's name carries no resolvable identity",
                     reason_kind="identity_disagreement")
    sent = parsed.get("geometry_digest")
    if sent not in (None, IDENTITY_DIGEST_PLACEHOLDER) and sent != digest:
        return _fail(f"the scheduler's path says geometry {sent} but this rank's "
                     f"own digest is {digest}: two sites computed the same field "
                     "and disagree, so neither names a file",
                     reason_kind="digest_disagreement")
    root = disk_spill.disk_root()
    if root is None:
        return _fail(f"{config.ENV_DISK_ROOT} is not set in this worker",
                     reason_kind="no_channel")
    rebuilt = disk_layout.build_slot_path(root, key16 + f"{int(group_id):08x}",
                                          digest)
    if not rebuilt["ok"]:
        return _fail(f"this rank could not name the file: {rebuilt['reason']}",
                     reason_kind="bad_path")
    if sent == digest:
        # Not the sentinel: a second digest site exists somewhere. Harmless here,
        # and worth one line, because it is the early sign of exactly the
        # two-sites-one-field bug the sentinel rule prevents.
        _note("the scheduler sent a path already carrying this rank's digest")
    return _ok(path=rebuilt["path"], key16=key16, group_id=int(group_id),
               digest=digest)


def worker_bind(worker, metadata) -> dict:
    """Queue one read per disk-sourced load entry. Cheap, bounded, never raises.

    THE ORDER IS THE WHOLE DESIGN: reads are queued HERE (bind) and applied at
    `start_load_kv`, one forward pass later, so the engine thread pays a
    `put_nowait` per entry and never waits for NVMe. An entry whose read has not
    settled by then is dropped and recomputed - see the module docstring.
    """
    _state["binds"] += 1
    try:
        return _worker_bind(worker, metadata)
    except Exception as exc:  # noqa: BLE001 - a bind never breaks a forward pass
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore worker bind")
        return _fail(f"the bind raised {type(exc).__name__}")


def _worker_bind(worker, metadata) -> dict:
    # Stamped BEFORE anything else this function does, so the bind-to-apply gap the
    # apply point reports is the gap between the fork's two call sites and not the
    # time our own bookkeeping took.
    bind_mono = time.monotonic()
    paths = _metadata_paths(metadata)
    if not paths or not any(paths):
        return _ok(queued=0, reason="no load entry on this step names a file")
    _state["disk_binds"] += 1
    gate = _may_restore_worker()
    if not gate["ok"]:
        return _worker_fail(gate.get("reason_kind") or "disarmed", gate["reason"])
    cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
    gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    if event is None or event < 0:
        return _worker_fail("bad_batch", "the step has disk work but no load event")
    if len(paths) != len(cpu_blocks) or len(gpu_blocks) != len(cpu_blocks):
        # The alignment IS the contract; a mismatch cannot be repaired by guessing
        # which list is wrong. Every disk entry on this step becomes untracked (and
        # is therefore pruned at `start_load_kv` and never DMA'd) while the fork's
        # RAM-only entries proceed untouched.
        return _worker_fail("path_count_mismatch",
                            f"load_paths has {len(paths)} entries, "
                            f"{FIELD_LOAD_CPU} has {len(cpu_blocks)} and "
                            f"{FIELD_LOAD_GPU} has {len(gpu_blocks)}: the index "
                            "alignment is gone, so nothing on this step can be "
                            "attributed")
    identity = _own_identity(worker)
    if not identity["ok"]:
        return _worker_fail(identity.get("reason_kind") or "fingerprint_incomplete",
                            identity["reason"])
    digest = identity["digest"]
    expect_bytes = identity["bytes"]
    indices = [index for index, path in enumerate(paths) if path]
    batch_id = _batch_id(event)
    opened = disk_read.begin_batch(batch_id, len(indices),
                                   deadline_seconds=disk_read.LOAD_DEADLINE_SECONDS)
    if not opened.get("ok"):
        kind = str(opened.get("reason_kind") or "")
        return _worker_fail(kind if kind in _state["reads_failed"] else "bad_batch",
                            str(opened.get("reason")))
    jobs: dict = {}
    failures: dict = {}
    # THE REQUEST(S) THIS EVENT BELONGS TO, read BEFORE the loop so a failure can name
    # one. `load_event_to_reqs` is per EVENT, so when a step put two requests in the
    # same load event the attribution is genuinely ambiguous and the record says None
    # rather than guessing the first id and letting it look like evidence.
    req_ids = _event_req_ids(metadata, event)
    bind_request_id = (req_ids[0] if len(req_ids) == 1 else None)
    recorded = 0
    for job_id in indices:
        resolved = _resolve_worker_path(paths[job_id], digest)
        if not resolved["ok"]:
            kind = str(resolved.get("reason_kind") or "bad_path")
            failures[job_id] = (kind if kind in _state["reads_failed"]
                                else "bad_path")
            _state["reads_failed"][failures[job_id]] += 1
            if recorded < MAX_FAILURES_PER_BIND:
                recorded += 1
                _note_load_failure(
                    failures[job_id], where="bind", reason=str(resolved.get("reason")),
                    request_id=bind_request_id, group=resolved.get("group_id"),
                    key16=resolved.get("key16"),
                    job_id=job_id, bytes_needed=expect_bytes,
                    cpu_block_id=_int(cpu_blocks[job_id]),
                    gpu_block_id=_int(gpu_blocks[job_id]),
                    jobs_needed=len(indices), jobs_fit=len(jobs))
            continue
        asked = disk_read.request_read(batch_id, job_id, resolved["path"],
                                       expect_bytes)
        if not asked.get("ok"):
            kind = str(asked.get("reason_kind") or "read_failed")
            failures[job_id] = (kind if kind in _state["reads_failed"]
                                else "read_failed")
            _state["reads_failed"][failures[job_id]] += 1
            if recorded < MAX_FAILURES_PER_BIND:
                recorded += 1
                # THE `buffer_cap` FIGURES, from the refusal that made them. This is
                # the record the 0.2.30 hunt needed: the pool's budget, what was
                # already committed, what this read cost, and how many jobs the offer
                # asked the pool to hold against how many fit.
                _note_load_failure(
                    failures[job_id], where="bind",
                    reason=str(asked.get("reason")),
                    request_id=bind_request_id, group=resolved.get("group_id"),
                    key16=resolved.get("key16"), job_id=job_id,
                    bytes_needed=asked.get("bytes_needed") or expect_bytes,
                    jobs_needed=asked.get("jobs_needed") or len(indices),
                    jobs_fit=(asked.get("jobs_that_fit") if asked.get("jobs_that_fit")
                              is not None else len(jobs)),
                    pool_budget_bytes=asked.get("pool_budget_bytes"),
                    pool_committed_bytes=asked.get("pool_committed_bytes"),
                    cpu_block_id=_int(cpu_blocks[job_id]),
                    gpu_block_id=_int(gpu_blocks[job_id]))
            continue
        jobs[job_id] = {"path": resolved["path"], "key16": resolved["key16"],
                        "group": resolved["group_id"],
                        "cpu_block_id": _int(cpu_blocks[job_id]),
                        "gpu_block_id": _int(gpu_blocks[job_id]),
                        "bytes": expect_bytes}
        _state["reads_started"] += 1
    deadline = round(time.monotonic() + RESTORE_DEADLINE_SECONDS, 6)
    _wbatches[batch_id] = {"batch_id": batch_id, "jobs": jobs,
                           "failures": failures, "event": event,
                           "req_ids": req_ids,
                           "deadline_mono": deadline,
                           # THE OTHER END OF THE WINDOW. `start_load_kv` runs one
                           # CPU-side forward launch after this call in the same step
                           # (`kv_connector_model_runner_mixin.py:80-93`, and our
                           # loads are all `load_kv_async=True` so
                           # `has_sync_kv_loads` is False), so this timestamp is what
                           # makes "did the read have time?" a measured number
                           # instead of an inference from a dropped-by-default
                           # counter. Monotonic, because a wall clock that steps
                           # would be reported as a race the box never had.
                           "bind_mono": bind_mono}
    while len(_wbatches) > disk_read.MAX_BATCHES:
        _evict_oldest_batch()
    for request_id in _wbatches[batch_id]["req_ids"]:
        while len(_wloads) >= MAX_TRACKED_LOADS:
            _evict_oldest_worker_load()
        _wloads[request_id] = {"batch": batch_id, "deadline_mono": deadline,
                               "released": False}
    _note(f"queued {len(jobs)} disk read(s) for load event {event} "
          f"({len(failures)} immediate failure(s))")
    return _ok(queued=len(jobs), failed=len(failures), batch=batch_id)


def _batch_id(event) -> str:
    return f"kvtier-restore-{int(event)}"


def _metadata_paths(metadata) -> list:
    raw = getattr(metadata, FIELD_LOAD_PATHS, None)
    if raw is None or isinstance(raw, (str, bytes)):
        return []
    try:
        return [item if isinstance(item, str) else "" for item in raw]
    except Exception as exc:  # noqa: BLE001 - an unreadable field is no field
        snapshot._error(exc, "disk_restore read load_paths")
        return []


def _event_req_ids(metadata, event: int) -> list:
    try:
        table = getattr(metadata, FIELD_LOAD_EVENT_TO_REQS, None) or {}
        raw = list(table.get(event) or [])
        return [str(item)[:256] for item in raw][:MAX_TRACKED_LOADS]
    except Exception as exc:  # noqa: BLE001 - an unreadable map costs a report only
        snapshot._error(exc, "disk_restore read load_event_to_reqs")
        return []


def _plan_regions(worker, cpu_block_id: int, total_bytes: int) -> dict:
    """(address, bytes) per region of one CPU slot, in the SPILL's own order.

    Mirrors `disk_spill.read_slot_bytes` exactly - `sorted(name)`, one
    `_region_span` per region, widths totalling `total_bytes` - because the file's
    byte order IS that iteration order. A plan that totals anything else is a
    fingerprint describing a different engine, and applying half a slot is the
    corruption this function exists to make impossible: it returns a refusal and the
    caller applies NOTHING.
    """
    caches = getattr(worker, "cpu_kv_caches", None)
    if not isinstance(caches, dict) or not caches:
        return _fail("the worker has no cpu_kv_caches to write into",
                     reason_kind="slot_plan_failed")
    names = sorted(str(name) for name in caches)
    if len(names) > disk_spill.MAX_REGIONS:
        return _fail(f"{len(names)} regions exceed the {disk_spill.MAX_REGIONS} "
                     "cap", reason_kind="slot_plan_failed")
    spans: list = []
    total = 0
    for name in names:
        try:
            span = disk_spill._region_span(caches[name], cpu_block_id,
                                           max_bytes=total_bytes)
        except Exception as exc:  # noqa: BLE001 - a hostile tensor is a refusal
            snapshot._error(exc, "disk_restore region span")
            return _fail(f"region {name!r}: {type(exc).__name__}",
                         reason_kind="slot_plan_failed")
        if not span["ok"]:
            return _fail(f"region {name!r}: {span['reason']}",
                         reason_kind="slot_plan_failed")
        spans.append((int(span["base"]), int(span["bytes_per_file"])))
        total += int(span["bytes_per_file"])
    if total != int(total_bytes):
        return _fail(f"the slot's regions total {total} bytes but the file is "
                     f"{total_bytes}: not these bytes, never applied",
                     reason_kind="slot_plan_failed")
    return _ok(spans=spans, total=total)


def _apply_buffer(worker, job: dict, buffer, total_bytes: int) -> dict:
    """One memmove per region, on the engine thread, once the read has settled.

    This is the moment `disk_read`'s docstring reserved for it: the only point at
    which a byte may enter a CPU slot, because it is synchronous, on the thread that
    owns the slot, and reached only with a buffer whose byte count was verified
    against this rank's slot width. A cancelled read's buffer is never reached,
    which is what makes "we stopped this load" and "the slot is clean" one fact.
    """
    plan = _plan_regions(worker, int(job["cpu_block_id"]), total_bytes)
    if not plan["ok"]:
        return plan
    try:
        source = ctypes.addressof(buffer)
        offset = 0
        for base, width in plan["spans"]:
            ctypes.memmove(base, source + offset, width)
            offset += width
        if offset != int(total_bytes):
            return _fail(f"moved {offset} of {total_bytes} bytes into the slot",
                         reason_kind="copy_failed")
    except Exception as exc:  # noqa: BLE001 - a failed copy is a load failure
        snapshot._error(exc, "disk_restore memmove")
        return _fail(f"the copy raised {type(exc).__name__}",
                     reason_kind="copy_failed")
    return _ok(bytes=int(total_bytes))


def abandon_disk_entries(worker, why: str) -> dict:
    """FAIL CLOSED: prune every disk entry of this step and report all of them.

    Called when OUR OWN prep raised. The disk entries in `load_cpu_blocks` were
    allocated for bytes we have not verified, and the fork's `start_load_kv` runs
    right after this hook returns - so leaving the lists alone means the fork DMAs
    whatever those CPU slots happened to contain and reports the load as finished.
    That is the one silent-corruption shape this module can produce by itself, and
    it is exactly what the `except Exception` path used to do (a probe with
    `batch_status` made to raise left `load_cpu_blocks == [0, 1, 2]`, the event
    submitted, and no load error reported). Recompute is always correct, so on any
    failure of our own the whole step's disk work is abandoned and reported.
    """
    metadata = getattr(worker, "_connector_metadata", None)
    if metadata is None:
        return _ok(abandoned=0, reason="no connector metadata on this step")
    paths = _metadata_paths(metadata)
    disk_indices = [index for index, path in enumerate(paths) if path]
    if not disk_indices:
        return _ok(abandoned=0, reason="no disk entries to abandon")
    cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
    gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
    keep = [index for index in range(min(len(cpu_blocks), len(gpu_blocks)))
            if index not in set(disk_indices)]
    for index in disk_indices:
        if index < len(gpu_blocks):
            gpu_id = _int(gpu_blocks[index])
            if gpu_id is not None:
                _remember_invalid(gpu_id)
        _state["reads_failed"]["abandoned"] += 1
    _replace_load_lists(metadata, [cpu_blocks[i] for i in keep],
                        [gpu_blocks[i] for i in keep])
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    batch_id = _batch_id(event) if event is not None else None
    batch = _wbatches.pop(batch_id, None) if batch_id else None
    submitted = bool(list(getattr(metadata, FIELD_LOAD_CPU, None) or []))
    released = 0
    if not submitted:
        released = release_event(worker, metadata, batch, why)
    _phantom_guard(worker, metadata, submitted=submitted)
    for request_id in (batch or {}).get("req_ids") or _event_req_ids(metadata,
                                                                      event or -1):
        _wloads.pop(request_id, None)
    if batch_id is not None:
        disk_read.cancel_batch(batch_id, why)
    _note(f"abandoned {len(disk_indices)} disk entry/entries: {why}")
    return _ok(abandoned=len(disk_indices), released=released)


def worker_before_load(worker) -> dict:
    """Apply settled bytes, prune the rest, and never DMA a stale slot.

    Runs BEFORE the fork's `start_load_kv`, which is the call that submits the
    CPU->GPU copy. Per disk entry, three outcomes: its bytes are in a verified
    bounce buffer and get memmoved into the slot; or it failed for any reason
    (missing, short, size mismatch, digest disagreement, queue drop, never settled,
    expired, unattributable) and its entry is REMOVED from this step's load lists,
    which both stops the stale DMA and lets its GPU block id be reported as a load
    error.

    When pruning leaves nothing to copy, the fork's `start_load_kv` returns without
    submitting an event - and a load event that was never submitted must NOT be left
    in `_pending_load_event_indices`, because the completion watermark is monotone
    over indices and a later completion would make this one reportable as "loaded".
    `release_event` does the removal and the report in the same call, and
    `_phantom_guard` re-checks the invariant afterwards and counts any trip.
    """
    try:
        return _worker_before_load(worker)
    except Exception as exc:  # noqa: BLE001 - a load prep never breaks a forward
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore before_load")
        try:
            abandon_disk_entries(worker, f"the load prep raised "
                                         f"{type(exc).__name__}")
        except Exception as exc2:  # noqa: BLE001 - the fallback cannot be the fault
            snapshot._error(exc2, "disk_restore abandon")
            _state["worker_errors"] += 1
        return _fail(f"the load prep raised {type(exc).__name__}")


def _burn_load(request_id: str, why: str) -> None:
    """Retire a load the engine said failed, THROUGH THE LEDGER, and remember it.

    Two jobs, one of which is a bug fix and one of which is honesty:

    1. `_drop_load` is the only door out of `_loads`. The failure paths used to call
       `_loads.pop()` directly, which skipped both side effects of the door - the
       `loads_released` counter and the release of this load's `_slot_paths` entries.
       A slot still mapped to a request that has no load is a slot the next
       `build_load_paths` will happily emit a path for, and bytes read into a CPU
       block the fork is using as a RAM-tier source are how a wrong byte reaches a
       GPU block. The mapping did not survive forever (`_sweep_loads` frees it once
       the request leaves the fork's `_reqs_to_load`), but "until the sweep notices"
       is a window, and this module does not have windows.
    2. The request is now in `_burned`, so if a `finished_recving` for the same id
       arrives later it is counted as `no_read` instead of being dropped on the
       floor: the fork called a load finished whose bytes we know did not land, and
       that is exactly the claim this counter exists to refuse.
    """
    if request_id not in _burned:
        if len(_burned) >= MAX_SUPPRESSED_REQUESTS:
            _burned.pop(next(iter(_burned)), None)      # oldest first
        _burned[request_id] = why[:240]
    _drop_load(request_id, why)


def _observe_apply_point(batch_id, batch: dict, status: dict,
                         apply_mono: float) -> tuple[dict, dict]:
    """Measure the window, wait if the operator asked to, and say what happened.

    Returns `(status, window)`, where `status` is the batch's state as of AFTER any
    wait, so the caller's classification is always the freshest answer and never a
    stale one. With the cap at its default 0 nothing is waited, nothing is slept, no
    extra lock is taken beyond the one `batch_status` already takes, and the returned
    `status` is the same object that came in - the observation is arithmetic over
    numbers the apply point already had.
    """
    bind_mono = batch.get("bind_mono")
    gap_ms = None
    if isinstance(bind_mono, (int, float)) and bind_mono > 0:
        gap_ms = round(max(0.0, apply_mono - float(bind_mono)) * 1000.0, 3)
    elapsed = status.get("elapsed_seconds")
    tokens, tokens_basis = _offered_tokens_estimate(batch)
    jobs = batch.get("jobs") or {}
    bytes_asked = sum(int(job.get("bytes") or 0) for job in jobs.values())
    # WHAT THE WAIT IS WAITING FOR. Since 0.2.32 the allowance is FLAT, so the bytes
    # still in flight decide nothing about the grant: they are PUBLISHED beside it, and
    # they are the bytes STILL IN FLIGHT rather than the size of the offer - a batch
    # that has delivered nine of ten slots is waiting for one slot. Every entry in one
    # batch is the same width (the store's slot width for this fingerprint), so the
    # outstanding bytes are exact rather than estimated, and `bytes_asked` is published
    # next to it so a reader can see both numbers.
    pending = int(status.get("pending") or 0)
    per_entry = (bytes_asked / len(jobs)) if jobs else 0.0
    bytes_at_stake = int(round(per_entry * min(pending, len(jobs))))
    budget = settle_budget(bytes_at_stake, status.get("seconds_left"))
    settled_on_arrival = bool(status.get("settled"))
    window = {"settled_on_arrival": settled_on_arrival,
              "bind_to_apply_ms": gap_ms,
              "elapsed_ms": (None if elapsed is None
                             else round(float(elapsed) * 1000.0, 3)),
              "known": bool(status.get("known", True)),
              "accepted": status.get("accepted"), "expected": status.get("expected"),
              "running": status.get("running"), "queued": status.get("queued"),
              "settled": settled_on_arrival,
              "bytes_asked": bytes_asked,
              "offered_tokens_estimate": tokens, "tokens_basis": tokens_basis}
    window.update({key: value for key, value in budget.items()
                   if key != "enabled"})
    if budget["env_problems"]:
        # A typo in the cap is reported once, because "the wait did not happen"
        # silently is how a mis-set drop-in becomes a week of confusion.
        if notice_once(f"settle-env-{'; '.join(budget['env_problems'])[:120]}",
                       f"DISK[settle] IGNORED: "
                       f"{'; '.join(budget['env_problems'])[:300]}"):
            _state["settle_notice"] = ("ignored: "
                                       + "; ".join(budget["env_problems"]))[:300]
    elif budget["enabled"]:
        key = f"settle-on-{budget['cap_ms']}-{budget['cap_from_default']}"
        origin = (f"{config.ENV_DISK_SETTLE_MS} is UNSET, so this is the 0.2.32 "
                  f"default: set it to 0 for the pre-0.2.32 never-wait boot"
                  if budget["cap_from_default"] else
                  f"{config.ENV_DISK_SETTLE_MS} is set by the environment")
        line = (f"DISK[settle] wait=ON cap_ms={budget['cap_ms']} ({origin}; FLAT "
                f"allowance since 0.2.32: granted = min(cap, the load's own "
                f"remaining deadline); no throughput or prefill-rate arithmetic "
                f"anymore. One line per distinct setting; the apply point waits "
                f"inside the engine step)")
        if notice_once(key, line):
            _state["settle_notice"] = line
    waited_ms = 0.0
    settled_after_wait = False
    # ONE CONDITION DECIDES THE WAIT, and the same condition is published as
    # `wait_granted` so the counter and the behaviour cannot drift apart: a batch
    # counted as granted a wait is a batch the code actually sat in front of.
    # Never wait on a batch that cannot settle - nothing pending means the loop would
    # spend the budget for nothing, and an expired deadline means the reader has
    # already given these bytes up.
    granted = bool(budget["enabled"] and budget["budget_ms"] > 0
                   and status.get("known", True) and not settled_on_arrival
                   and pending > 0 and not status.get("deadline_expired"))
    window["wait_granted"] = granted
    if granted:
        status, waited_ms = _settle_wait(batch_id, status, budget["budget_ms"])
        waited_ms = round(waited_ms, 3)
        settled_after_wait = bool(status.get("settled"))
    window["waited_ms"] = waited_ms
    window["settled_after_wait"] = settled_after_wait
    window["dropped_after_wait"] = bool(budget["enabled"] and waited_ms > 0
                                        and not status.get("settled"))
    # THE HONEST WAIT METRIC (0.2.32). `budget_over_recompute` compared a grant with
    # the arithmetic that produced it, so the 23.6 ms grant that lost a 135.8 ms race
    # and then cost a 12-second recompute read as `over_recompute: 0`. This one asks
    # the only question a flat allowance can answer: did the batch consume the WHOLE
    # allowance and still not have its bytes? The 1 ms tolerance is because the poll
    # loop leaves when the remaining time drops below a tick, not at zero.
    window["allowance_spent"] = bool(granted and not settled_after_wait
                                     and waited_ms + 1.0 >= float(budget["budget_ms"]))
    window["settled_now"] = bool(status.get("settled"))
    states = ({} if not status.get("known", True) or not status.get("pending")
              else disk_read.job_states(batch_id))
    window["job_states"] = states.get("states") or {}
    window["job_states_truncated"] = bool(states.get("truncated"))
    _note_apply_window(window)
    return status, window


def _classify_dropped_entry(window: dict, job_id: int) -> str:
    """Which kind of "too late" this entry is, from the batch's own state table.

    `not_settled` covered two different bugs and told the operator neither: a job
    still QUEUED means the reader thread never got to it (a scheduling problem on the
    proxy host - fixable by waiting, which is what the settle wait is for), while a
    job RUNNING means the kernel read was in flight and lost (a device that is slower
    than the engine's step - only a bigger budget or a faster disk fixes it). An
    unreadable state table is reported as `dropped_unattributed` rather than guessed
    into one of the two, because a counter that silently absorbs the unknown case is
    how a measurement becomes a story.
    """
    state = (window.get("job_states") or {}).get(int(job_id))
    if state == disk_read.QUEUED:
        _state["dropped_queued"] += 1
        return "dropped_queued"
    if state == disk_read.RUNNING:
        _state["dropped_running"] += 1
        return "dropped_running"
    _state["dropped_unattributed"] += 1
    return "dropped_unattributed"


def _apply_failure_sentence(kind, window: dict) -> str:
    """One sentence for an apply-point loss, using the batch's own state table.

    `not_settled` covers two different faults and the window already knows which one
    this was (see `_classify_dropped_entry`), so the record says QUEUED (the reader
    never got to it - a wait fixes it) or RUNNING (the kernel read was in flight and
    lost - only a bigger budget or a faster disk fixes it) rather than re-deriving it
    from a counter afterwards. Anything else quotes its own classification name.
    """
    name = str(kind or "read_failed")
    if name != "not_settled":
        return (f"the read was asked for and its bytes were not in a buffer at "
                f"`start_load_kv`, classified {name}; the GPU block is reported as a "
                "load error so the engine recomputes it")
    states = window.get("job_states") or {}
    return (f"the read had not settled at `start_load_kv` "
            f"(settled_on_arrival={bool(window.get('settled_on_arrival'))}, "
            f"waited_ms={window.get('waited_ms')}, bind_to_apply_ms="
            f"{window.get('bind_to_apply_ms')}; job state "
            f"{sorted({str(value) for value in states.values()})[:4]})")


def _batch_span(batch_jobs: dict) -> str | None:
    """The span key of a worker batch: its lowest job's `key16:group`.

    The same spelling `_suppressed_span` uses for a scheduler-side run, so a worker's
    record and the scheduler's fuse name one identity and can be matched by eye.
    """
    try:
        for job_id in sorted(batch_jobs):
            job = batch_jobs[job_id] or {}
            key16, group = job.get("key16"), _int(job.get("group"))
            if key16 and group is not None:
                return f"{str(key16)[:16]}:{int(group)}"
        return None
    except Exception:  # noqa: BLE001 - a missing span is only a weaker record
        return None


def span_shape(batch: dict, failed) -> dict:
    """Group one batch's disk entries into SPANS and say which spans came whole.

    WHY THIS IS PUBLISHED AND WHY IT CHANGES NOTHING. A restore only feeds the engine
    a usable prefix when a span is complete ACROSS its groups, and the apply point
    prunes per ENTRY - so "what survived" has a shape, and that shape is now an
    input to a live investigation into whether a partial restore may have its tokens
    counted as computed. Until 0.2.24 the shape was implicitly known: reads were
    issued and completed in submission order by one thread, and submission order is
    span-major / group-minor (`_merge_pending_hit` walks `for span: for group:`), so
    an interrupted batch could only ever be missing a SUFFIX of spans. With N reader
    threads the answers interleave and the missing piece can be a span in the middle.

    This function MEASURES that. It decides nothing, prunes nothing, and touches no
    token count and no control flow - the all-or-nothing question belongs to another
    change and must stay there.

    SPAN RECOGNITION, spelled out because the job id is a step-wide index and not a
    span number: entries arrive in order and a span is a maximal run of STRICTLY
    INCREASING group ids, so the run ends where the group id stops advancing. That is
    why a request's RAM entries (empty paths, so no job) and a second request sharing
    the step cannot confuse it, while `job_id // groups` would.

    Bounded: the walk is over this batch's jobs, which `disk_read.MAX_JOBS_PER_BATCH`
    caps, and `partial_spans` names at most `MAX_SPANS_REPORT` of them.

    ONE INTERPRETIVE CAVEAT, because this number will be read by someone deciding
    whether a partial restore may be counted: the batch this is called with is the
    RESTORE-side one, whose `jobs` hold only entries that were ACCEPTED at bind. An
    entry refused at bind (`buffer_cap`, `bad_path`, a stale key) is ABSENT rather
    than failed - the scheduler was told at bind and the DMA never asked for it - so
    it does not make its span partial. `failed` is the apply-time set: bytes that were
    asked for and did not arrive. That is the distinction the token question needs.
    """
    jobs = batch.get("jobs") or {}
    refused = {int(item) for item in (failed or {}) if _is_int_like(item)}
    if not jobs:
        return {"spans": 0, "entries": 0, "entries_applied": 0, "spans_whole": 0,
                "spans_partial": 0, "spans_empty": 0, "applied_is_prefix": None,
                "partial_spans": [], "truncated": False, "reason":
                    "no disk entries in this batch"}
    # ONE int-keyed copy, so the walk below is a linear pass over a bounded batch
    # rather than a lookup per entry (a hand-built batch with string keys would
    # otherwise turn this into a scan per job, on the step thread).
    by_id: dict = {}
    for key, value in jobs.items():
        if _is_int_like(key):
            by_id[int(key)] = value
        else:
            continue
    spans: list = []
    current: list = []
    last_group: int | None = None
    ungrouped = 0
    for job_id in sorted(by_id):
        group = _int((by_id.get(job_id) or {}).get("group"))
        if group is None:
            ungrouped += 1
            group = 0
        if last_group is not None and group <= last_group:
            spans.append(current)
            current = []
        current.append(job_id)
        last_group = group
    if current:
        spans.append(current)
    whole = partial = empty = applied_entries = 0
    partial_spans: list = []
    prefix_broken = False
    applied_is_prefix = True
    for index, members in enumerate(spans):
        hit = [job_id for job_id in members if job_id not in refused]
        applied_entries += len(hit)
        if not hit:
            empty += 1
            prefix_broken = True
        elif len(hit) == len(members):
            whole += 1
            if prefix_broken:
                applied_is_prefix = False
        else:
            partial += 1
            # A partial span AFTER a break is as much a hole as a whole one is: the
            # applied bytes are not a prefix of the span sequence either way, and a
            # caller that treated this as a prefix would count tokens for a middle.
            if prefix_broken:
                applied_is_prefix = False
            prefix_broken = True
            if len(partial_spans) < MAX_SPANS_REPORT:
                partial_spans.append(int(index))
    return {"spans": len(spans), "entries": len(by_id),
            "entries_applied": int(applied_entries),
            "spans_whole": int(whole), "spans_partial": int(partial),
            "spans_empty": int(empty),
            # True when every span that got anything is followed by nothing but
            # empty spans: the shape ONE reader produced for free, because a batch
            # interrupted mid-flight was always missing a SUFFIX. False is the new
            # shape - a hole in the middle - and it can only appear with N readers.
            "applied_is_prefix": bool(applied_is_prefix),
            "partial_spans": partial_spans,
            "truncated": bool(partial > len(partial_spans)),
            "reason": (f"{ungrouped} entries named no group and were counted as "
                       "group 0" if ungrouped else None)}


def _is_int_like(value) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _note_span_shape(shape: dict) -> None:
    """Fold one span-shape observation into the counters. Counting ONLY.

    No lock, in common with every other counter here, and nothing that can wait or
    raise; it runs on the worker's step thread at the apply point.
    """
    spans = int(shape.get("spans") or 0)
    if not spans:
        return
    _state["spans_observed"] += spans
    _state["span_entries_observed"] += int(shape.get("entries") or 0)
    _state["spans_whole"] += int(shape.get("spans_whole") or 0)
    _state["spans_partial"] += int(shape.get("spans_partial") or 0)
    _state["spans_empty"] += int(shape.get("spans_empty") or 0)
    if not shape.get("applied_is_prefix", True):
        _state["spans_hole_in_the_middle"] += 1
    _state["last_span_shape"] = dict(shape)


def _worker_before_load(worker) -> dict:
    apply_mono = time.monotonic()
    metadata = getattr(worker, "_connector_metadata", None)
    if metadata is None:
        return _ok(applied=0, reason="no connector metadata on this step")
    paths = _metadata_paths(metadata)
    if not paths or not any(paths):
        return _ok(applied=0, reason="no disk entries on this step")
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    batch_id = _batch_id(event) if event is not None else None
    batch = _wbatches.pop(batch_id, None) if batch_id else None
    # A batch we never bound has no window to report, and an empty one must never
    # make the classification below crash or guess: `_classify_dropped_entry` reads
    # it for the per-entry state and finds "unattributed", which is the truth.
    window: dict = {"job_states": {}, "job_states_truncated": False}
    cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
    gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
    disk_indices = [index for index, path in enumerate(paths) if path]
    if batch is None:
        # Bind registered no batch (disarmed at bind, a bind-time refusal, or a
        # worker-side restart). Every disk entry is then unattributable, and
        # unattributable bytes must never be DMA'd: prune them all.
        batch = {"batch_id": batch_id, "jobs": {}, "failures": {}, "event": event,
                 "req_ids": (_event_req_ids(metadata, event)
                             if event is not None else []),
                 "deadline_mono": time.monotonic()}
        failed = {index: "not_tracked" for index in disk_indices}
        status = {}
    else:
        status = disk_read.batch_status(batch_id)
        # THE WINDOW, MEASURED. This is where a read either has its bytes or does
        # not, so it is the only place the bind-to-apply gap can be observed from;
        # `batch` carries the timestamp `_worker_bind` stamped when it queued them.
        # With the settle cap at its default the call rewrites nothing about the
        # control flow below - same status object, no sleep - and the whole effect
        # on the boot is a handful of counters and one dict.
        status, window = _observe_apply_point(batch_id, batch, status, apply_mono)
        failed = dict(batch["failures"])
        # `disk_read` classified the failure on its own ledger (`file_missing`,
        # `size_mismatch`, `short_read`, `read_error`, `not_ours`). Carry the NAME
        # through: "the LRU is evicting faster than we read" (file_missing) and
        # "the file is a different geometry than this boot's" (size_mismatch) and
        # "the disk returned half a slot" (short_read) are three different operator
        # actions, and flattening them into `read_failed` made the counter that is
        # supposed to explain the recomputes explain nothing. The report is capped
        # (`disk_read.MAX_JOBS_REPORT`), so a job past the cap keeps the generic
        # name - the fine grain is best-effort, the failure report is not.
        classified = {}
        for item in status.get("failures") or []:
            try:
                classified[int(item["job"])] = str(item["reason_kind"])
            except (TypeError, ValueError, KeyError):
                continue
        if len(paths) != len(cpu_blocks) or len(gpu_blocks) != len(cpu_blocks):
            # THE STEP IS UNATTRIBUTABLE. `load_paths`, `load_cpu_blocks` and
            # `load_gpu_blocks` are built together by the fork and always share a
            # length, so a disagreement means the metadata is not the object the
            # scheduler built - and with the pairing unknown, we cannot say which GPU
            # block a disk file's bytes belong to, nor even which RAM entries are
            # safe. Pruning selectively would be a guess. So the whole step's load is
            # abandoned and EVERY GPU block it named is reported as a load error:
            # recompute is always correct, and a step that recomputes is a step that
            # serves the right answer slightly later.
            failed = {index: "path_count_mismatch" for index
                      in range(len(cpu_blocks))}
            for value in gpu_blocks:
                gpu_id = _int(value)
                if gpu_id is not None:
                    _remember_invalid(gpu_id)
            _state["path_length_violations"] += 1
        else:
            for job_id, job in batch["jobs"].items():
                if job_id in failed:
                    continue
                if not status.get("known"):
                    failed[job_id] = "not_tracked"
                    continue
                buffer = disk_read.batch_buffer(batch_id, job_id)
                if buffer is None:
                    # Not DONE: still in flight, or failed. Both mean "these bytes do
                    # not exist". WHICH of them they are is the whole finding of
                    # 0.2.23: a job still QUEUED lost to thread scheduling (a wait
                    # fixes it), a job RUNNING lost to a device slower than the step
                    # (only a bigger budget or a faster disk fixes it), and until now
                    # both were the same `not_settled` line.
                    if status.get("pending"):
                        failed[job_id] = "not_settled"
                        _state["dma_unsettled_entries"] += 1
                        _classify_dropped_entry(window, job_id)
                    else:
                        reason_kind = classified.get(job_id)
                        failed[job_id] = (reason_kind
                                          if reason_kind in _state["reads_failed"]
                                          else "read_failed")
                    continue
                outcome = _apply_buffer(worker, job, buffer, int(job["bytes"]))
                if not outcome["ok"]:
                    kind = outcome.get("reason_kind")
                    failed[job_id] = (kind if kind in _state["reads_failed"]
                                      else "copy_failed")
                    continue
                if window.get("settled_on_arrival") and not window.get("waited_ms"):
                    # The counter the settle-wait decision is made against: these are
                    # the entries the box already wins for free, and the ratio of
                    # `settled_after_wait` to this number is the wait's hit rate.
                    _state["applied_settled_immediately"] += 1
                _state["reads_applied"] += 1
                _state["bytes_applied"] += int(outcome["bytes"])
        if status.get("incomplete"):
            # Fewer reads than declared: a caller bug, and a batch that quietly had
            # fewer reads than slots is exactly how a stale slot reaches the DMA.
            for job_id in batch["jobs"]:
                failed.setdefault(job_id, "incomplete")
        if status.get("deadline_expired"):
            for job_id in batch["jobs"]:
                failed.setdefault(job_id, "deadline_expired")
    drop = sorted({int(job_id) for job_id in failed
                   if 0 <= int(job_id) < len(cpu_blocks)})
    # THE SHAPE OF WHAT SURVIVED, counted and nothing else. `failed` is final here and
    # the prune below has not run yet, so this is exactly the set the DMA will see.
    # It cannot change the outcome: `span_shape` reads, `_note_span_shape` counts.
    shape = span_shape(batch, failed)
    _note_span_shape(shape)
    window["span_shape"] = shape
    drop_set = set(drop)
    already_counted = set(batch["failures"]) if batch else set()
    batch_jobs = batch.get("jobs") or {}
    # THE APPLY-POINT LOSSES, recorded with the numbers. Bytes that were asked for and
    # did not arrive are the failures the engine then reports as load errors, so this
    # is the last moment the plugin knows WHY. Bounded per step
    # (`MAX_FAILURES_PER_BIND`) because one bad step can drop forty entries and a ring
    # full of the forty-first is a ring that recorded nothing useful.
    logged = 0
    for job_id in drop:
        if job_id not in already_counted:
            # Bind-time failures were counted when they happened; counting them again
            # here would make one failed read two, and the counters are the only
            # evidence an operator has about which stage is losing reads.
            kind = failed.get(job_id)
            _state["reads_failed"][kind if kind in _state["reads_failed"]
                                   else "read_failed"] += 1
            if logged < MAX_FAILURES_PER_BIND:
                logged += 1
                job = batch_jobs.get(job_id) or {}
                _note_load_failure(
                    kind if kind in WORKER_FAILURES else "read_failed",
                    where="apply",
                    reason=_clip(_apply_failure_sentence(kind, window), 200),
                    request_id=(batch.get("req_ids") or [None])[0]
                    if len(batch.get("req_ids") or []) == 1 else None,
                    span=_batch_span(batch_jobs), group=job.get("group"),
                    key16=job.get("key16"),
                    gpu_block_id=_int(job.get("gpu_block_id")),
                    cpu_block_id=_int(job.get("cpu_block_id")), job_id=job_id,
                    bytes_needed=_int(job.get("bytes")),
                    jobs_needed=len(batch_jobs),
                    jobs_fit=len(batch_jobs) - len(drop))
        if job_id < len(gpu_blocks):
            gpu_id = _int(gpu_blocks[job_id])
            if gpu_id is not None:
                _remember_invalid(gpu_id)
    if drop:
        kept_cpu = []
        kept_gpu = []
        for index in range(min(len(cpu_blocks), len(gpu_blocks))):
            if index in drop_set:
                continue
            kept_cpu.append(cpu_blocks[index])
            kept_gpu.append(gpu_blocks[index])
        _replace_load_lists(metadata, kept_cpu, kept_gpu)
    if batch_id is not None:
        disk_read.finish_batch(batch_id)
    submitted = bool(list(getattr(metadata, FIELD_LOAD_CPU, None) or []))
    released = 0
    if drop and not submitted:
        released = release_event(worker, metadata, batch,
                                 "every load entry on this step came from disk and "
                                 "none had bytes to copy")
    _phantom_guard(worker, metadata, submitted=submitted)
    for request_id in batch["req_ids"]:
        record = _wloads.get(request_id)
        if record is not None and record.get("batch") not in (None, batch_id):
            # A NEWER bind for the same request owns the record now; this step may
            # not retire another step's watchdog coverage.
            continue
        # THE WORKER WATCHDOG HAS NOTHING LEFT TO DO FOR THIS REQUEST, and keeping
        # the record was a real bug, not untidiness. Its one job is a load whose
        # step never reached `start_load_kv`; here the step DID reach it, so this
        # request's outcome is already final: either the fork submitted what
        # survived the pruning and will report the event itself, or
        # `release_event` reported it just now. A record left behind fired ~45 s
        # later and pushed `finished_recving` for a request the engine had long
        # since moved on from - a release nobody asked for, a `finished_released`
        # counter that lies, and (for a load still in flight on a stalled stream)
        # a claim that bytes were loaded before the copy that carries them had
        # finished. That last shape is the phantom this module exists to prevent,
        # arriving from the watchdog instead of from the watermark.
        _wloads.pop(request_id, None)
    applied = len([job_id for job_id in batch["jobs"] if job_id not in failed])
    kinds = sorted({str(failed.get(job_id)) for job_id in drop})[:3]
    _note(f"applied {applied} of {len(batch['jobs'])} disk read(s); dropped "
          f"{len(drop)} for {kinds or '-'}")
    return _ok(applied=applied, dropped=len(drop), released=released,
               submitted=submitted)


def _replace_load_lists(metadata, cpu_blocks, gpu_blocks) -> None:
    """Rewrite a step's load lists with the surviving entries, still aligned."""
    try:
        setattr(metadata, FIELD_LOAD_CPU, list(cpu_blocks))
        setattr(metadata, FIELD_LOAD_GPU, list(gpu_blocks))
        setattr(metadata, FIELD_LOAD_PATHS, [""] * len(cpu_blocks))
    except Exception as exc:  # noqa: BLE001 - an immutable step is a no-op
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore prune")


def _remember_invalid(gpu_block_id: int) -> None:
    if len(_invalid) >= MAX_INVALID_BLOCKS:
        _invalid.pop(next(iter(_invalid)), None)
    _invalid[int(gpu_block_id)] = round(time.monotonic(), 6)


def _phantom_guard(worker, metadata, submitted: bool) -> dict:
    """THE hazard check: an event that was never submitted must not stay pending.

    `worker._pending_load_event_indices` is what `get_finished` mines, and it mines
    it with `index <= watermark` where the watermark is monotone. So leaving a
    never-submitted index in that set is how a LATER completion gets reported as
    "this request's KV arrived" - the engine believing bytes it never received are
    valid KV. Every path that skips a submission calls `release_event` in the same
    breath; this re-checks the pair and COUNTS a trip when the invariant is broken
    (and closes it, because the right response to the hazard is to remove the index,
    not merely to notice it).
    """
    try:
        event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
        pending = getattr(worker, "_pending_load_event_indices", None)
        if event is None or event < 0 or not isinstance(pending, set):
            return _ok(checked=False)
        if submitted or event not in pending:
            return _ok(checked=True)
        _state["phantom_guard_trips"] += 1
        pending.discard(event)
        for request_id in _event_req_ids(metadata, event):
            record = _wloads.get(request_id)
            if record is not None:
                record["released"] = True
            _defer_finish(request_id)
            _state["finished_released"] += 1
        _note(f"PHANTOM GUARD: load event {event} was not submitted but was still "
              "pending; it was removed and its requests released as failures")
        return _fail("the invariant was violated, and closed", tripped=True)
    except Exception as exc:  # noqa: BLE001 - a guard that raises is not a guard
        snapshot._error(exc, "disk_restore phantom guard")
        return _fail(f"the guard raised {type(exc).__name__}")


_deferred_finished: set = set()

# KNOWN GAP, DELIBERATELY NOT CLOSED HERE (the single next action for this path).
# `finished_recving` and `get_block_ids_with_load_errors` are produced by two
# different engine calls in a step, and this module cannot choose their order: the
# model runner reads the load errors at `v1/worker/gpu/kv_connector.py:116` and
# mines `get_finished` separately. A release made INSIDE `get_finished` (this
# watchdog's, which runs there) therefore has its GPU block ids drained by the
# NEXT step's read, so the scheduler sees "the load finished" one step before it
# sees "truncate these blocks and recompute them". Under
# `kv_load_failure_policy=recompute` the truncation does arrive, but one step of
# tokens is generated from KV that was about to be invalidated. The sound fix is a
# per-request pairing: hold a release in `_deferred_finished` until the drain that
# carries ITS OWN block ids has happened, with a bounded wait (never park a
# request waiting for a report). It needs a new piece of state and a deadline, and
# the release-vs-report ordering must be settled against the fork's real step
# order before writing it - the hazard is recorded here, and pinned by
# `test_a_watchdog_expiry_releases_exactly_once`, so the next change has a test to
# extend rather than a mystery to rediscover.


def _defer_finish(request_id: str) -> None:
    if len(_deferred_finished) >= MAX_TRACKED_LOADS:
        _deferred_finished.clear()
    _deferred_finished.add(str(request_id))


def release_event(worker, metadata, batch, why: str) -> int:
    """Abandon a load event, REPORTING it finished instead of going silent.

    Two things happen together and neither is optional: the index leaves
    `_pending_load_event_indices` (the phantom fix) and every request of that event
    is queued for `finished_recving` (the "never report" fix, which with no engine
    timeout for WAITING_FOR_REMOTE_KVS would park the request for the life of the
    boot). The read batch is cancelled so its bounce buffers are dropped - a
    cancelled read never writes a destination, which is `disk_read`'s whole reason
    for existing.
    """
    event = _int(getattr(metadata, FIELD_LOAD_EVENT, None))
    req_ids = list((batch or {}).get("req_ids") or [])
    if event is not None and event >= 0:
        try:
            pending = getattr(worker, "_pending_load_event_indices", None)
            if isinstance(pending, set):
                pending.discard(event)
        except Exception as exc:  # noqa: BLE001 - best effort, then report anyway
            snapshot._error(exc, "disk_restore discard pending")
        if not req_ids:
            req_ids = _event_req_ids(metadata, event)
    released = 0
    for request_id in req_ids:
        record = _wloads.get(request_id)
        if record is not None:
            record["released"] = True
        _defer_finish(request_id)
        _state["finished_released"] += 1
        released += 1
    batch_id = (batch or {}).get("batch_id")
    if batch_id is not None:
        disk_read.cancel_batch(batch_id, why)
        _wbatches.pop(batch_id, None)
    # MAKE THE STEP SAFE, not just silent. `release_event` is usually called from
    # `start_load_kv`, which has already pruned the entries it dropped and rewritten
    # the lists; from the watchdog it has not, and the disk entries would still be
    # sitting in `load_cpu_blocks` for the next `start_load_kv` of this metadata to
    # DMA - stale bytes, reported as loaded. So every disk entry this release speaks
    # for leaves the step's lists here, and its GPU block is reported if the caller
    # has not already reported it. This also erases the disk identities (via
    # `_replace_load_lists`), so no later pass can re-release the same load or report
    # the same block ids twice - `_handle_invalid_blocks` attributes an id to
    # whichever request holds that block NOW.
    try:
        paths = _metadata_paths(metadata)
        if any(paths):
            cpu_blocks = list(getattr(metadata, FIELD_LOAD_CPU, None) or [])
            gpu_blocks = list(getattr(metadata, FIELD_LOAD_GPU, None) or [])
            owned = set((batch or {}).get("jobs") or {})
            keep: list = []
            for index in range(min(len(cpu_blocks), len(gpu_blocks))):
                if not paths[index]:
                    keep.append(index)
                    continue
                if index not in owned:
                    gpu_id = _int(gpu_blocks[index])
                    if gpu_id is not None:
                        _remember_invalid(gpu_id)
            _replace_load_lists(metadata, [cpu_blocks[i] for i in keep],
                                [gpu_blocks[i] for i in keep])
    except Exception as exc:  # noqa: BLE001 - an immutable step is a no-op
        snapshot._error(exc, "disk_restore release prune")
    _note(f"released load event {event} as a failure: {why}")
    return released


def take_invalid_block_ids() -> set:
    """Drain the GPU block ids whose disk read failed. ONE read, then empty.

    The engine asks every step (`v1/worker/gpu/kv_connector.py:116`) and
    `_handle_invalid_blocks` maps an id to whichever request currently holds it, so
    a stale id would truncate an UNRELATED request later. Draining is therefore part
    of the safety story, not a convenience.
    """
    out = {int(block_id) for block_id in _invalid}
    _invalid.clear()
    if out:
        _state["invalid_reports"] += len(out)
    return out


def worker_get_finished(worker, result) -> tuple:
    """Add this module's releases to the fork's `(sending, finished_recving)`.

    The fork's value is returned UNTOUCHED on the happy path: the only mutation is a
    union when this module has something to report, so an unarmed or idle restore
    gives back the identical object.
    """
    try:
        sweep_watchdog(worker)
        if not _deferred_finished:
            return result
        sending = result[0] if isinstance(result, tuple) and len(result) > 0 else None
        recving = (result[1] if isinstance(result, tuple) and len(result) > 1
                   else None)
        merged = set(recving or ()) | set(_deferred_finished)
        _deferred_finished.clear()
        for request_id in merged:
            _wloads.pop(request_id, None)
        return (sending, merged or None)
    except Exception as exc:  # noqa: BLE001 - a report never breaks a step
        _state["worker_errors"] += 1
        snapshot._error(exc, "disk_restore get_finished")
        _deferred_finished.clear()
        return result


def sweep_watchdog(worker) -> dict:
    """Expire in-flight disk loads. Bounded table, every fire counted, never silent.

    THE LAST RESORT, and the reason a disk load cannot become a permanent stall: the
    batch deadline and the settle check cover the ordinary failures, and this covers
    the case where a step never reached `start_load_kv` (a no-forward step, or an
    exception upstream of us) so nobody ever looked at the batch. On expiry the load
    is reported as a failure for every GPU block it owned and its requests are
    released - a recompute under `kv_load_failure_policy=recompute`, and under no
    policy at all still a request that continues.
    """
    fired = 0
    now = time.monotonic()
    pending = getattr(worker, "_pending_load_event_indices", None)
    for request_id in list(_wloads):
        record = _wloads.get(request_id)
        if record is None:
            continue
        if record.get("released"):
            _wloads.pop(request_id, None)
            continue
        if now < float(record["deadline_mono"]):
            continue
        batch_id = record.get("batch")
        batch = _wbatches.pop(batch_id, None) if batch_id else None
        metadata = getattr(worker, "_connector_metadata", None)
        event = (batch or {}).get("event")
        live_step = (metadata is not None and event is not None
                     and _int(getattr(metadata, FIELD_LOAD_EVENT, None)) == event)
        if batch is not None:
            # Every GPU block this load owned: without the report the scheduler keeps
            # the blocks it allocated for bytes that are never coming. Entries that
            # failed at submit are not in `jobs`; `release_event`'s prune picks those
            # up from the step itself when the step is still live.
            for job in (batch.get("jobs") or {}).values():
                gpu_id = _int((job or {}).get("gpu_block_id"))
                if gpu_id is not None:
                    _remember_invalid(gpu_id)
        if batch is not None and live_step:
            # ONE RELEASE PATH. `release_event` removes the index, reports every
            # request of the event, cancels the read batch AND prunes the step's
            # disk entries - which matters here as much as in `start_load_kv`: the
            # sweep decides the load failed, and a metadata still naming those files
            # would be pruned and released a SECOND time by the next pass over the
            # same step (a second `finished_recving` for one load, and a second
            # report of the same GPU block ids, which the scheduler attributes to
            # whichever request holds those blocks by then).
            release_event(worker, metadata, batch,
                          "the restore watchdog expired the load")
        else:
            if batch is not None:
                disk_read.cancel_batch(batch_id,
                                       "the restore watchdog expired the load")
            _defer_finish(request_id)
            _state["finished_released"] += 1
        if batch is not None and not live_step:
            event = batch.get("event")
            if event is not None and isinstance(pending, set):
                pending.discard(event)
        _state["worker_watchdog_fires"] += 1
        _wloads.pop(request_id, None)
        fired += 1
    if fired:
        _note(f"the restore watchdog released {fired} load(s)")
    return _ok(fired=fired)


def _evict_oldest_batch() -> None:
    oldest = next(iter(_wbatches))
    batch = _wbatches.pop(oldest, None)
    if batch is not None:
        disk_read.cancel_batch(oldest, "the batch table is at its cap")


def _evict_oldest_worker_load() -> None:
    if not _wloads:
        return
    _wloads.pop(next(iter(_wloads)), None)
    _note("the worker ledger is at its cap; the oldest entry was dropped")


# --- the witness --------------------------------------------------------------

def witness_verdict() -> dict:
    """Can this boot PROVE a disk restore served a request? One verdict, one line.

    CONFIRMED needs TWO facts, not one:

    * `restorations >= 1` - a request whose disk-sourced CPU slots reached the
      worker's `load_cpu_blocks` and whose load was reported `finished_recving`; and
    * `reads_applied >= 1` somewhere in this boot (here, or attested by a live worker
      rank file) - because the fork reports a load finished for its OWN host-RAM
      blocks too, so a finished report on a request whose disk slots nobody filled
      is indistinguishable from success.

    The live box on 2026-10-05 printed `verdict=CONFIRMED reads_applied=0`
    8,244 times while serving zero disk bytes. One of those two facts was missing and
    nothing checked the other.

    REFUTED names why it never happened, in the order a reader should check: the
    gate never opened, nothing was offered, offers were not adopted (the
    3,200-token floor or a clamp), an adopted load never reported, or it reported
    without a single disk byte behind it.
    """
    refused = {name: int(count) for name, count in _state["refused"].items()
               if count}
    reads = applied_reads_evidence()
    restorations = int(_state["restorations"])
    verdict, reason, kind = _verdict_and_kind(refused, reads, restorations)
    # The stable NAME of a verdict is everything before the first colon:
    # `offers_not_adopted: 3 offer(s) ...` and `... 4 offer(s) ...` are the same thing
    # to say. Keying an emission on the sentence is exactly the 1,048-lines-a-second
    # flood - the sentence quotes counters, and counters move every step.
    _state["witness_verdict"] = verdict
    _state["witness_reason"] = reason
    _state["witness_kind"] = kind
    return {"verdict": verdict, "reason": reason, "kind": kind,
            "restorations": restorations,
            "reads_applied": int(reads["reads_applied"]),
            "reads_from": reads["source"],
            "offers": int(_state["offers"]),
            "offered_tokens": int(_state["offered_tokens"]),
            "adopted": int(_state["adopted"]),
            "adopted_tokens": int(_state["adopted_tokens"]),
            "no_read": int(_state["no_read"]),
            "refused": refused}


def _verdict_and_kind(refused: dict, reads: dict, restorations: int) -> tuple:
    """THE VERDICT ARITHMETIC, in one place, for a caller that stores it and one that
    only looks. `(verdict, reason, kind)`.

    Split out of `witness_verdict` for `witness_now()`: a payload read must be able to
    ask "what would the witness say THIS instant?" without writing the three
    as-of-emission fields that `emit_witness()` dedupes on. A read path that mutated
    them would both rewrite history and change what the next emission decides to log.
    """
    verdict = "CONFIRMED" if restorations >= 1 and reads["reads_applied"] >= 1 \
        else "REFUTED"
    reason = "-"
    if verdict == "REFUTED":
        reason = _refuted_reason(refused, reads)
    kind = "CONFIRMED" if verdict == "CONFIRMED" else reason.split(":", 1)[0]
    return verdict, reason, kind


def witness_now() -> dict:
    """What the witness WOULD say this instant, touching no state (0.2.25).

    WHY THIS EXISTS. `witness_verdict`/`witness_kind` are frozen at the last emission,
    which dedupes on `(verdict, kind)` and only speaks when one of them changes - and
    on the live box at 00:08 that published `witness_kind=offers_not_adopted` next to
    `adopted=1`. The kind was TRUE when it was spoken (the offer had not been adopted
    yet) and FALSE by the time anyone read it, because `_mark_adopted` had since run
    and no verdict change had followed to overwrite the sentence. The verdict itself
    (`REFUTED`) was right; the kind read as a live claim and was not one.

    So the payload carries both: the as-of-emission kind, which is the line that was
    logged, and this, which is the same arithmetic on the counters as they stand.
    Neither is allowed to mutate the other.
    """
    try:
        refused = {name: int(count) for name, count in _state["refused"].items()
                   if count}
        reads = applied_reads_evidence()
        restorations = int(_state["restorations"])
        verdict, reason, kind = _verdict_and_kind(refused, reads, restorations)
        return {"verdict": verdict, "kind": kind, "reason": reason,
                "restorations": restorations,
                "reads_applied": int(reads["reads_applied"]),
                "adopted": int(_state["adopted"]),
                "stale_as_of_emission": bool(
                    _state["witness_kind"] is not None
                    and _state["witness_kind"] != kind)}
    except Exception as exc:  # noqa: BLE001 - a payload field never breaks a poll
        snapshot._error(exc, "disk_restore witness_now")
        return {"verdict": None, "kind": None, "reason": None,
                "restorations": 0, "reads_applied": 0, "adopted": 0,
                "stale_as_of_emission": False}


def applied_reads_evidence() -> dict:
    """Applied disk reads this boot, from either process, with the source named.

    The scheduler never reads a file itself - the worker does - so its own counter is
    0 by construction and the worker's number only crosses the process boundary
    through the rank file. Naming WHICH source answered is the point: `reads=0`
    because the scheduler has no fingers is a different fact from `reads=0` because
    nothing was ever read, and only the second one is news.
    """
    local = int(_state["reads_applied"] or 0) + int(_state["disk_reads_applied"] or 0)
    ranks = int(_state["ranks_reads_applied"] or 0)
    if local > 0:
        return {"reads_applied": local, "source": "this process"}
    if ranks > 0:
        return {"reads_applied": ranks, "source": "worker rank file"}
    return {"reads_applied": 0,
            "source": ("nothing has applied a disk read in this boot"
                       if _state["sched_fitted"] else
                       "this process runs neither half")}


def partial_load_verdict() -> dict:
    """What a PARTIAL load actually does to a request, as data and as a sentence.

    WHY THIS EXISTS. "restore is live ... 0 load(s) reported disk bytes back" printed
    beside `ranks_reads_applied=42` read as a contradiction, and it was not one: the
    reads happened and the request still got nothing. The reason is in the apply point
    and the engine's own recovery, and it is worth stating because it decides what a
    partial load is WORTH:

      1. Each dropped entry is PRUNED from `load_cpu_blocks`/`load_gpu_blocks`, and its
         GPU block goes through `get_block_ids_with_load_errors`.
      2. If ANY entry survived, the fork submits the DMA for the survivors. So the
         bytes that did arrive really are written into GPU blocks.
      3. The engine's `_handle_invalid_blocks` then resets the request's computed
         tokens and RESCHEDULES it. A KV prefix is a prefix: everything from the first
         invalid block onward is recomputed, so an applied read that sits after a hole
         is work done for bytes the request will overwrite.
      4. `note_reported` reads that same invalid set back and decides what it COSTS
         (0.2.32, `failure_kind_classes`): a DATA kind fuses the request and the span,
         a RACE kind (`not_settled`, `deadline_expired`, `buffer_cap`, `queue_full`)
         spares both and grants the request exactly one re-offer. Either way the offer
         is wasted for THAT request while its prefix was in flight: reads after the
         first hole are bytes the request will recompute over.
      5. THE SHRINK THE ENGINE WILL NOT LET US DO. A hole in ONE group of ONE span
         should be truncatable to the contiguous landed prefix, and the fork CAN do
         that - but only for a request with exactly one KV cache group
         (`vllm/v1/core/sched/scheduler.py:3217-3252`). This box runs a hybrid model,
         so `req_block_ids_per_group` has more than one entry and the branch above it
         (`scheduler.py:3193-3215`) sets `num_computed_tokens = 0`: the WHOLE prefix,
         not the tail, is recomputed. That is why a 23.6 ms grant became a 12.26 s
         recompute, and no plugin-side change can shorten it: the invalid-block channel
         is the only channel the worker has for "these bytes are not here", and on this
         layout that channel means "recompute everything".

    The span counters say how much of that was a hole in the middle rather than a
    missing suffix, which is the shape N reader threads introduced.
    """
    reads = applied_reads_evidence()
    applied = int(reads["reads_applied"])
    restorations = int(_state["restorations"] or 0)
    no_read = int(_state["no_read"] or 0)
    spans_partial = int(_state["spans_partial"] or 0)
    spans_whole = int(_state["spans_whole"] or 0)
    spans_empty = int(_state["spans_empty"] or 0)
    holes = int(_state["spans_hole_in_the_middle"] or 0)
    dropped = (int(_state["dropped_queued"] or 0)
               + int(_state["dropped_running"] or 0)
               + int(_state["dropped_unattributed"] or 0))
    if not applied:
        shape = "no_reads_applied"
        sentence = ("nothing has been memmoved into a CPU slot in this boot, so "
                    "there is no partial load to talk about")
    elif restorations:
        shape = "applied_and_reported"
        sentence = (f"{applied} read(s) applied from {reads['source']} and "
                    f"{restorations} load(s) reported disk bytes back: that pairing "
                    "is the restoration this module counts as a real restore")
    else:
        shape = ("applied_but_unreported" if not no_read
                 else "applied_reported_without_disk")
        sentence = (
            f"{applied} read(s) reached a CPU slot from {reads['source']}, "
            f"{restorations} load(s) reported disk bytes back "
            f"({no_read} report(s) came back with NO disk read behind them): the "
            "reads are real work and the restoration is not. A PARTIAL load is the "
            "likely shape - the survivors were DMA'd and the refused ones were "
            "reported as load errors, so the engine reset the request's computed "
            "tokens and rescheduled it (on a multi-group layout that reset is to "
            "ZERO, not to the first hole: see `partial_load_verdict`'s item 5), and "
            f"the fuse decision now reads the failure kinds - races not fused: "
            f"{int(_state['races_not_fused'])}, fuses on data kinds: "
            f"{int(_state['fuses_on_data_kinds'])}, on unknown kinds: "
            f"{int(_state['fuses_on_unknown_kinds'])}. "
            "That means an offer whose jobs did not all fit is WASTED for the "
            "request even when its prefix landed: reads after the first hole are "
            "bytes the request will recompute over.")
    return {"shape": shape, "reads_applied": applied,
            "reads_from": reads["source"], "restorations": restorations,
            "no_read": no_read, "entries_dropped_at_apply": dropped,
            "spans_whole": spans_whole, "spans_partial": spans_partial,
            "spans_empty": spans_empty, "hole_in_the_middle": holes,
            "sentence": sentence}


def _refuted_reason(refused: dict, reads: dict) -> str:
    """The first link in the chain that did not hold, named."""
    if _state["offers"] == 0:
        if refused:
            worst = max(refused.items(), key=lambda item: item[1])[0]
            return (f"gate_closed_or_refused:{worst} (no offer was ever made; the "
                    "refusals name why)")
        return ("never_offered: the restore path ran and offered nothing, so the "
                "SSD tier never had a contiguous span to offer")
    if _state["adopted"] == 0:
        return (f"offers_not_adopted: {_state['offers']} offer(s) reached the "
                "scheduler and none reached a worker (below the whole-scheduler-"
                "block adoption floor, or clamped to zero)")
    if not _state["restorations"] and not _state["no_read"]:
        # Nothing has been REPORTED yet for an adopted load: that is a different
        # failure from "reported, but with no read behind it", and the order matters
        # because a scheduler process always has reads_applied == 0 by construction.
        return ("adopted_never_reported: disk slots reached the worker but no load "
                "was reported finished (watchdog fires: "
                f"{int(_state['worker_watchdog_fires'])}, phantom trips: "
                f"{int(_state['phantom_guard_trips'])})")
    if _state["no_read"] and not _state["restorations"]:
        return (f"no_disk_read_behind_the_report: {_state['no_read']} load(s) came "
                "back finished with no applied disk read at all, which is the "
                "signature of slots that were allocated and never filled ("
                f"{_clip(reads['source'], 60)})")
    if not reads["reads_applied"]:
        return (f"adopted_without_a_read: {_state['adopted']} load(s) reached the "
                "worker with no applied disk read anywhere in this boot "
                f"({_clip(reads['source'], 60)}) - the reports the fork makes for "
                "its own host-RAM blocks are not evidence of a restore")
    # Reached only when a partial restoration happened and something else is still
    # short of the bar; say what the numbers are rather than invent a category. The
    # span shape travels into the sentence because "partial" without it cannot be
    # acted on: a missing suffix is one reader being slow, a hole in the middle is N
    # readers answering out of order.
    shape = partial_load_verdict()
    return (f"partial: {int(_state['restorations'])} restoration(s) with "
            f"{int(reads['reads_applied'])} applied read(s) and "
            f"{int(_state['no_read'])} report(s) with no read behind them "
            f"(spans whole={shape['spans_whole']} partial={shape['spans_partial']} "
            f"empty={shape['spans_empty']} hole_in_middle="
            f"{shape['hole_in_the_middle']})")


def witness_line() -> str:
    r"""THE greppable line. Fixed field order; bounded; never raises.

    `grep 'DISK[restore-witness]'` answers "does the restore serve requests, or did
    it never run" without a verb and without a judgement call.
    """
    try:
        view = witness_verdict()
        reads = {"reads_applied": view["reads_applied"], "source": view["reads_from"]}
        parts = ["DISK[restore-witness]",
                 f"verdict={view['verdict']}",
                 f"restorations={view['restorations']}",
                 f"offers={view['offers']}",
                 f"offered_tokens={view['offered_tokens']}",
                 f"adopted={view['adopted']}",
                 f"adopted_tokens={view['adopted_tokens']}",
                 f"reads_applied={int(reads['reads_applied'])}",
                 f"no_read={int(_state['no_read'])}",
                 f"suppressed={int(_state['suppressions'])}",
                 f"disk_authority={'on' if _state['disk_authority'] else 'off'}",
                 f"serving_as={serving_as()}",
                 f"offers_switch={'on' if _state['offers_enabled'] else 'off'}",
                 f"watchdog_fires={int(_state['worker_watchdog_fires'])}",
                 f"phantom_trips={int(_state['phantom_guard_trips'])}"]
        refused = view["refused"]
        head = ",".join(f"{name}:{count}"
                        for name, count in sorted(refused.items())[:4])
        parts.append(f"refused={head or 'none'}")
        if len(refused) > 4:
            parts.append(f"refused_more={len(refused) - 4}")
        parts.append(f"reason={_clip(view['reason'], MAX_BOOT_LINE_CHARS)}")
        line = " ".join(parts)
    except Exception as exc:  # noqa: BLE001 - the witness cannot be allowed to fail
        snapshot._error(exc, "disk_restore witness_line")
        line = ("DISK[restore-witness] verdict=REFUTED "
                f"unreadable={type(exc).__name__}")
    return line[:MAX_BOOT_LINE_CHARS * 4]


def emit_witness(force: bool = False) -> dict:
    """Log the witness when its VERDICT OR REASON changes, and never more than a
    stated rate.

    The dedupe key is `(verdict, reason)`, NOT the whole line. Keying on the line is
    what put this module in the log 1,048 times a second on the incident boot:
    `offered_tokens` and `adopted_tokens` move on every single step, so the "changed
    verdict" test fired forever and the flood itself became the second outage. The
    verdict and its reason are what an operator reads; the counters are readable from
    the payload whenever they want numbers.

    A verdict change is logged immediately even mid-window (that is the event worth
    reading); a REPEAT of a verdict already out is what the window suppresses.
    """
    try:
        line = witness_line()
        key = (_state["witness_verdict"], _state["witness_kind"])
        now = time.monotonic()
        fired = _witness_state["emitted_mono"]
        if _witness_state["key"] == key and not force:
            _state["witness_suppressed"] += 1
            if not isinstance(fired, (int, float)) \
                    or (now - fired) < MIN_WITNESS_LOG_SECONDS:
                return {"logged": False, "reason": "unchanged verdict",
                        "line": line}
        _witness_state["key"] = key
        _witness_state["line"] = line
        _witness_state["emitted_mono"] = now
        _state["witness_emissions"] += 1
        _logger.warning("kvtier-dev-plugin: %s", line)
        return {"logged": True, "reason": None, "line": line}
    except Exception as exc:  # noqa: BLE001 - a log line never breaks a step
        snapshot._error(exc, "disk_restore emit_witness")
        return {"logged": False, "reason": f"logging raised {type(exc).__name__}",
                "line": None}


# --- the boot line and the rank file ------------------------------------------

def boot_line() -> str:
    r"""`DISK[restore] installed=... role=... methods=... armed=... ...`.

    The pair to `DISK[spill]`: that line says whether bytes could be WRITTEN, this
    one whether they could be READ BACK. Same posture, same reason - the refusal
    paths are the ones an operator needs, and a plugin that only logs success is a
    plugin whose failure looks exactly like idleness.

    `role=` and `methods=` are what make the line readable across a distributed
    boot. Four ranks and one scheduler each print one of these, and a bare
    `installed=YES` from a rank used to be indistinguishable from the same verdict
    from the scheduler - while only the scheduler's half can produce an offer. The
    label set is unique by construction, and the fields name the CLASS and the
    METHODS wrapped in the process that printed the line.
    """
    try:
        # ROLE-ARMEDED `armed`. On 2026-10-05 four worker ranks printed `armed=NO` in
        # this line while the scheduler printed `armed=YES`, and the field meant two
        # different things in the same log: for a scheduler it answers "may I offer",
        # for a worker it must answer "will I serve" (worker_service_verdict). A
        # worker fails the SCHEDULER gate for reasons that have nothing to do with
        # serving - its process never received the tier arm - so printing the
        # scheduler's answer from a worker made an honest worker look broken and let
        # a genuinely broken pairing look like ordinary idleness.
        service = worker_service_verdict()
        opened = gate() if role_here() != "worker" else service
        parts = ["DISK[restore]"]
        parts.append("installed={}".format("YES" if _state["installed"] else "NO"))
        parts.append("class={}".format(_state["class"] or "none"))
        parts.append("role={}".format(role_here()))
        parts.append("methods={}".format(_methods_field(fitted_methods_here())))
        parts.append("armed={}".format("YES" if opened["ok"] else "NO"))
        parts.append("root={}".format(_clip(disk_spill.disk_root(), 60) or "-"))
        parts.append("index={}".format(int(disk_index.counters()["entries"])))
        parts.append("digest={}".format(_clip(disk_spill.geometry_digest(), 16)
                                        or "none"))
        # THE POSTURE, on the line, because it is the first thing to ask about when
        # the counters say nothing happened. `disk_authority=ON` is the field that says
        # "this boot serves EVERY request from SSD and gives its host-RAM hit back" -
        # a posture an operator must be able to confirm at a glance, and one no request
        # can influence. `serving_as` is the same two switches read as one verdict, and
        # `offer_block_cap`/`ranks_armed` are the bounds that still apply under it.
        parts.append("disk_authority={}".format(
            "ON" if _state["disk_authority"] else "OFF"))
        parts.append(f"serving_as={serving_as()}")
        parts.append("offers_switch={}".format(
            "ON" if _state["offers_enabled"] else "OFF"))
        # THE PER-OFFER BOUND, on the line, in BLOCKS and in the tokens they are worth.
        # `offer_cap=32000` used to sit here, and after 0.2.32 deletes that cap the
        # field would be reporting a bound that constrains nothing - so it is gone, and
        # what replaces it is the bound that DOES constrain: the block walk. Blocks
        # first because the block count is the real quantity; the token figure is only
        # that many blocks times the engine's block size, whose source a payload reader
        # can check at `limits.offer_block_ceiling.block_size_source`.
        ceiling = offer_block_ceiling()
        parts.append("offer_block_cap={}".format(int(ceiling["blocks"])))
        parts.append("offer_block_cap_tokens={}".format(int(ceiling["tokens"])))
        parts.append("authority_lookups={}".format(
            int(_state["authority_lookups"])))
        parts.append("suppressed={}".format(int(_state["suppressions"])))
        parts.append("offers={}".format(int(_state["offers"])))
        parts.append("offered_tokens={}".format(int(_state["offered_tokens"])))
        parts.append("reads_applied={}".format(int(_state["reads_applied"])))
        parts.append("ranks_armed={}/{}".format(int(_state["ranks_proven"]),
                                                 int(_state["ranks_seen"]
                                                     or _state["ranks_proven"])))
        parts.append("deadline={}s".format(RESTORE_DEADLINE_SECONDS))
        refused = {name: int(count) for name, count in _state["refused"].items()
                   if count}
        head = ",".join(f"{name}:{count}"
                        for name, count in sorted(refused.items())[:4])
        parts.append(f"refused={head or 'none'}")
        reason = None if opened["ok"] else opened["reason"]
        if reason is None and _state["skip_reason"]:
            reason = _state["skip_reason"]
        if reason:
            parts.append(f"reason={_clip(reason, MAX_BOOT_LINE_CHARS)}")
        line = " ".join(parts)
    except Exception as exc:  # noqa: BLE001 - the fallback cannot fail
        snapshot._error(exc, "disk_restore boot_line")
        _state["fallbacks"] += 1
        line = (f"DISK[restore] installed={'YES' if _state['installed'] else 'NO'} "
                f"unreadable={type(exc).__name__}")
    return line[:MAX_BOOT_LINE_CHARS * 4]


def _methods_field(names: list, cap: int = 62) -> str:
    """`a+b+Nmore`, never a name cut in half. A truncated label misreads as a fact."""
    text = "+".join(names)
    if not names:
        return "none"
    if len(text) <= cap:
        return text
    kept: list = []
    used = 0
    for name in names:
        if used + len(name) + 1 > cap - 8:
            break
        kept.append(name)
        used += len(name) + 1
    return "+".join(kept) + f"+{len(names) - len(kept)}more"


def emit_boot_line() -> dict:
    """Log `boot_line()` once per distinct verdict per process."""
    try:
        line = boot_line()
        _state["boot_line"] = line
        if line in _logged_lines:
            return {"logged": False, "reason": "this verdict is already logged",
                    "line": line}
        _logged_lines.append(line)
        del _logged_lines[:-4]
        _logger.warning("kvtier-dev-plugin: %s", line)
        return {"logged": True, "reason": None, "line": line}
    except Exception as exc:  # noqa: BLE001 - a log line never breaks an install
        snapshot._error(exc, "disk_restore boot line")
        return {"logged": False, "reason": f"logging raised {type(exc).__name__}",
                "line": None}


def read_latency() -> tuple:
    """(avg, max) seconds per file read, from `disk_read`'s own ledger."""
    try:
        state = disk_read.counters()
        reads = int(state.get("reads_completed") or 0)
        seconds = float(state.get("read_seconds") or 0.0)
        avg = round(seconds / reads, 6) if reads else 0.0
        return avg, round(float(state.get("max_read_seconds") or 0.0), 6)
    except Exception as exc:  # noqa: BLE001 - a missing latency is (0, 0)
        snapshot._error(exc, "disk_restore read latency")
        return 0.0, 0.0


def _compact_failure(record: dict) -> dict:
    """One load-failure record, shrunk to what a rank file's byte budget can carry.

    The rank file's `disk_restore` section is a guest in a `MAX_CONTRIBUTION_BYTES`
    document that `worker_diag` REFUSES outright when it is exceeded, so the records
    that cross the process boundary are the small ones: the provenance the worker
    could not compute anyway (its own index is empty) is dropped and re-derived by the
    scheduler, which is the process holding the index. The five `buffer_cap` figures
    are NEVER dropped - they are the reason the record exists.
    """
    out: dict = {}
    for field in ("seq", "ts", "kind", "where", "request_id", "span", "group",
                  "contract", "key16", "gpu_block_id", "job_id", "bytes_needed",
                  "jobs_needed", "jobs_fit", "pool_budget_bytes",
                  "pool_committed_bytes", "reason"):
        out[field] = record.get(field)
    return out


def worker_report() -> dict:
    """This module's worker verdict for the rank file, degraded to fit if needed."""
    from . import worker_diag  # noqa: PLC0415 - the seam owns its own cap
    try:
        # `armed` here is what the SCHEDULER reads to decide whether to offer at all,
        # so it must be the worker's own service verdict and never `gate()`: the
        # scheduler's gate asks for a tier arm and a rank proof that a worker process
        # cannot satisfy, so `gate()` would have every honest rank publish
        # `armed:false` forever, and the incident boot's `armed:NO` was this same
        # field meaning the wrong question. See `worker_service_verdict`.
        service = worker_service_verdict()
        report = {"section": "disk_restore", "writer": OWNER_TAG,
                  "installed": bool(_state["installed"]),
                  # THE PER-RANK SERVICE FACTS the scheduler's proof reads (0.2.28):
                  # whether THIS process fitted the worker half, and the kind of the
                  # service verdict's refusal. The `armed` field stays for reporting
                  # and for backward-reading, but the TIER ARM is no longer asked of
                  # rank files - the scheduler reads `disk-spill-arm.json` live
                  # (`_tier_arm_verdict`); this file's armed claim is a lagged mirror
                  # of a fact its owner does not own.
                  "worker_fitted": bool(_state["worker_fitted"]),
                  "service_reason_kind": service.get("reason_kind"),
                  "class": _clip(_state["class"], 60),
                  "armed": bool(service["ok"]),
                  "armed_reason": _clip(service.get("reason"), 100),
                  "binds": int(_state["binds"]),
                  "disk_binds": int(_state["disk_binds"]),
                  "reads_started": int(_state["reads_started"]),
                  "reads_applied": int(_state["reads_applied"]),
                  "bytes_applied": int(_state["bytes_applied"]),
                  "dma_unsettled": int(_state["dma_unsettled_entries"]),
                  "failed": {name: int(count) for name, count
                             in _state["reads_failed"].items() if count},
                  "invalid_reports": int(_state["invalid_reports"]),
                  "finished_released": int(_state["finished_released"]),
                  "watchdog_fires": int(_state["worker_watchdog_fires"]),
                  "phantom_trips": int(_state["phantom_guard_trips"]),
                  "deadline_seconds": RESTORE_DEADLINE_SECONDS,
                  "read_seconds_avg": read_latency()[0],
                  "read_seconds_max": read_latency()[1],
                  # THE RACE AND THE WAIT, in the one place an operator reads worker
                  # state. `waited_ms` sits next to `read_seconds_avg` on purpose: a
                  # wait is only interpretable against how long the reads actually
                  # take, and the pair is the finding either way it goes. Kept
                  # deliberately compact - this section is a guest in a byte-capped
                  # file (`worker_diag.MAX_CONTRIBUTION_BYTES`) and the full
                  # distribution, per-batch record and token estimate live in
                  # `apply_window()` on the snapshot/verb side.
                  "race": {"obs": int(_state["apply_observations"]),
                           "settled_on_arrival":
                               int(_state["apply_settled_on_arrival"]),
                           "gap_ms": {"min": _state["bind_to_apply_min_ms"],
                                      "max": _state["bind_to_apply_max_ms"]},
                           "elapsed_ms": {"min": _state["apply_elapsed_min_ms"],
                                          "max": _state["apply_elapsed_max_ms"]},
                           "applied_now": int(_state["applied_settled_immediately"]),
                           "drop_queued": int(_state["dropped_queued"]),
                           "drop_running": int(_state["dropped_running"]),
                           "drop_unknown": int(_state["dropped_unattributed"])},
                  "reason": _clip(_state["last_reason"], 100),
                  "note": ("restores READ the SSD tier. armed is this rank's own "
                           "answer to 'will I service a disk load' and the scheduler "
                           "refuses every offer while any live rank says no; "
                           "reads_applied counts slots whose bytes reached a CPU "
                           "slot; dma_unsettled counts those that arrived too late "
                           "and were recomputed. race splits that last number into "
                           "drop_queued (the read never started) and drop_running (it "
                           "was mid-kernel); phantom_trips must stay 0: a trip means "
                           "a load event that was never submitted was about to be "
                           "reported as loaded.")}
        waited = int(_state["settle_waits_granted"])
        # The wait lives in its own key rather than inside the literal above because its
        # average needs a division that would have to be inlined at eight characters of
        # indentation, and a per-batch record only exists once a batch has been seen.
        report["settle"] = {
            "waits": waited,
            "settled_after_wait": int(_state["settled_after_wait"]),
            "dropped_after_wait": int(_state["dropped_after_wait"]),
            "waited_ms_avg": (None if not waited else
                              round(_state["settle_ms_total"] / waited, 3)),
            "waited_ms_max": _state["settle_ms_max"],
            "budget_ms_max": _state["settle_budget_max_ms"],
            "allowance_spent": int(_state["settle_allowance_spent"]),
            "last": ({key: value for key, value in
                      (_state["last_apply_window"] or {}).items()
                      if key in ("bytes_asked", "bytes_at_stake", "budget_ms",
                                 "waited_ms", "bound", "settled",
                                 "settled_after_wait", "allowance_spent")}
                     or None)}
        # THE CONCURRENCY AND THE SHAPE, in the rank file because a WORKER is the only
        # process that reads, and both are worker-side facts. A handful of ints on
        # purpose: this section is a guest in a byte-capped file, and the per-thread
        # table, the throughput rates and the span-by-span record live in
        # `disk_read.section()` / `apply_window()` on the snapshot and verb side.
        #   * readers.configured vs alive tells an operator whether the threads they
        #     asked for exist; peak_units == 1 with configured > 1 says the reads were
        #     never actually concurrent; starved_polls and buffer_refusals name WHICH
        #     side starved (queue/pool vs disk).
        #   * spans.hole_in_middle is the concurrency finding about PARTIAL restores,
        #     in the place a rank can report it: impossible for one reader, so a
        #     non-zero count next to configured > 1 is the measurement.
        read_state = disk_read.counters()
        report["readers"] = {
            "configured": int(read_state["threads_configured"]),
            "alive": int(read_state["threads_alive"]),
            "peak_units": int(read_state["units_in_flight_peak"]),
            "starved_polls": int(read_state["reader_starved_polls"]),
            "buffer_refusals": int((read_state["refused"] or {})
                                   .get("buffer_cap") or 0),
            "out_of_order": int(read_state["completions_out_of_order"]),
            # THE RECENCY PROMOTION (0.2.29), in the place a rank can report it: the
            # tier's LRU clock only moves if these numbers move, so "is promotion
            # working?" is one field rather than a guess during a load run.
            "recency_promotions": int(read_state.get("recency_promotions") or 0),
            "recency_promotions_throttled": int(
                read_state.get("recency_promotions_throttled") or 0),
            "recency_promotion_failures": int(
                read_state.get("recency_promotion_failures") or 0),
        }
        # THE UNLINK POOL (0.2.29), beside the reader threads it is modelled on: the
        # spill tier's eviction workers live in this same process, and "did parallel
        # eviction run?" belongs next to "did parallel reads run?".
        spill_state = disk_spill.counters()
        report["evict"] = {
            "configured": int(spill_state.get("evict_threads_configured") or 0),
            "alive": disk_spill.evict_thread_count_alive(),
            "batches_submitted": int(
                spill_state.get("evict_batches_submitted") or 0),
            "files_unlinked": int(spill_state.get("evict_files_unlinked") or 0),
            "files_failed": int(spill_state.get("evict_files_failed") or 0),
            "pool_fallbacks": int(spill_state.get("evict_fallbacks") or 0),
            # The boot total, not the last pass's bool: a rank that once ran out of
            # time has told the operator the ceiling is not keeping up with the fill.
            "deadline_hits": int(spill_state.get("ceiling_deadline_hits") or 0),
        }
        report["spans"] = {"seen": int(_state["spans_observed"]),
                           "whole": int(_state["spans_whole"]),
                           "partial": int(_state["spans_partial"]),
                           "empty": int(_state["spans_empty"]),
                           "hole_in_middle": int(
                               _state["spans_hole_in_the_middle"])}
        # WHAT ONE READ COSTS THIS RANK, so the scheduler can price an offer against
        # its bounce-buffer pool with a real number instead of a guess. The fingerprint
        # owns the figure; None means this rank has no fingerprint yet and the offer
        # falls back to the conservative price (`POOL_PRICE_BYTES_PER_JOB_FALLBACK`).
        report["bytes_per_file"] = disk_spill.slot_width_bytes()
        # THE REASONS, ACROSS THE PROCESS BOUNDARY. `failed` above counts them; these
        # say WHICH read, for WHICH request, with WHICH byte figures - the numbers a
        # scheduler process can never have in its own counters because it does not read
        # files. Newest first, and few: the byte budget belongs to the whole section.
        report["last_failures"] = [_compact_failure(item) for item
                                   in _load_failures[:MAX_FAILURES_PER_RANK_FILE]]
        if len(json.dumps(report, default=str)) > worker_diag.MAX_CONTRIBUTION_BYTES:
            return {"section": "disk_restore", "writer": OWNER_TAG,
                    "truncated": True,
                    "installed": bool(_state["installed"]),
                    "worker_fitted": bool(_state["worker_fitted"]),
                    "service_reason_kind": service.get("reason_kind"),
                    "armed": bool(service["ok"]),
                    "reads_applied": int(_state["reads_applied"]),
                    "bytes_applied": int(_state["bytes_applied"]),
                    "phantom_trips": int(_state["phantom_guard_trips"])}
        return report
    except Exception as exc:  # noqa: BLE001 - a report is never a step fault
        snapshot._error(exc, "disk_restore worker report")
        return {"section": "disk_restore", "writer": OWNER_TAG,
                "report_error": type(exc).__name__}


def register_contribution() -> dict:
    """Offer the restore verdict to the rank-file writer. Idempotent.

    The `signal` is DEFENCE IN DEPTH for the stale-arm hole (0.2.28): it folds the
    armed verdict into `worker_diag`'s change signature, so an arm or fit transition
    forces a rewrite on the next step instead of waiting for the cadence or a spill
    to move a counter. It does NOT close the idle-box hole on its own (a box with no
    finished requests runs no `get_finished` and writes nothing at any cadence),
    which is exactly why the proof stopped depending on the rank files for the arm
    in the first place (`_tier_arm_verdict`).
    """
    from . import worker_diag  # noqa: PLC0415 - one writer, and no import cycle
    return worker_diag.register_contribution(
        "disk_restore", worker_report, owner=OWNER_TAG,
        signal=lambda: (int(bool(disk_spill.is_armed())),
                        int(bool(_state["worker_fitted"])),
                        int(bool(_state["installed"]))))


# --- install / uninstall ------------------------------------------------------

_REUSED = object()

# --- install / uninstall: the wrappers ----------------------------------------

def _layer_is_ours(method) -> bool:
    return getattr(method, OWNER_ATTR, None) == OWNER_TAG


def _layer_is_foreign(method) -> bool:
    """True only for a wrapper carrying ANOTHER PACKAGE's owner marker.

    The marker name is the whole question: our own layers (this module's, `backport`'s)
    carry `OWNER_ATTR` set to this package's tag, so anything that carries it set to
    something else is a layer whose semantics we cannot reason about and must not wrap.
    Layers that carry no marker at all - the fork's own function, or a sibling's
    `functools.wraps` composition - are ordinary callables to compose with, which is
    what lets four wrappers of `get_finished` coexist.
    """
    marker = getattr(method, OWNER_ATTR, None)
    return marker is not None and marker != OWNER_TAG


def _stamp(wrapper, name: str, original=None) -> None:
    setattr(wrapper, OWNER_ATTR, OWNER_TAG)
    if original is not None:
        setattr(wrapper, ORIGINAL_ATTR, original)
    wrapper.__module__ = __name__
    wrapper.__name__ = name


def make_wrapper(name: str, original):
    """One scheduler entry point, with the restore hook AFTER the fork's body.

    Public because `backport._rewrap` rebuilds our layers on top of a new base and
    has to rebuild THIS one exactly as `install` does - the composition rule that
    keeps `guard_mode` outermost, so an exception raised inside this module reaches
    the guard's fault counter instead of becoming a step that silently did nothing.
    Returns None for a method this module has no hook for (a refusal, never a guess).
    """
    if name == METHOD_MATCHED:
        def wrapper(self, request, num_computed_tokens):
            result = original(self, request, num_computed_tokens)
            return offer(self, request, num_computed_tokens, result)
    elif name == METHOD_BUILD_META:
        def wrapper(self, scheduler_output):
            metadata = original(self, scheduler_output)
            build_load_paths(self, metadata)   # contained: never raises
            return metadata                    # THE FORK'S OBJECT, UNTOUCHED
    elif name == METHOD_UPDATE_OUTPUT:
        def wrapper(self, connector_output):
            result = original(self, connector_output)
            note_reported(self, connector_output)
            return result                      # UNTOUCHED (the fork returns None)
    else:
        return None
    _stamp(wrapper, name, original)
    wrapper.__doc__ = (
        "kvtier-dev-plugin SSD restore layer: the tier's own "
        f"{name}, with the disk tier consulted after the fork's body. Refuses and "
        "counts unless a root is configured AND the spill tier is armed AND the "
        "tier is armed; the fork's return value passes through unchanged on every "
        "refusal path.")
    return wrapper


def _wrapped_bind(original):
    def bind_connector_metadata(self, metadata):
        original(self, metadata)        # THE FORK FIRST: it owns the event sets
        worker_bind(self, metadata)     # contained: never raises
    _stamp(bind_connector_metadata, METHOD_BIND, original)
    return bind_connector_metadata


def _wrapped_start_load(original):
    def start_load_kv(self, *args, **kwargs):
        worker_before_load(self)        # apply settled bytes, prune the rest
        return original(self, *args, **kwargs)
    _stamp(start_load_kv, METHOD_START_LOAD, original)
    return start_load_kv


def _wrapped_get_finished(original):
    def get_finished(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return worker_get_finished(self, result)
    _stamp(get_finished, METHOD_GET_FINISHED, original)
    return get_finished


_WORKER_WRAPPERS = {METHOD_BIND: _wrapped_bind,
                    METHOD_START_LOAD: _wrapped_start_load,
                    METHOD_GET_FINISHED: _wrapped_get_finished}


def _wrapped_load_errors(original):
    def get_block_ids_with_load_errors(self, *args, **kwargs):
        """The fork's own answer UNION the disk reads that failed. Drained here."""
        try:
            mine = take_invalid_block_ids()
        except Exception as exc:  # noqa: BLE001 - a report never breaks a step
            snapshot._error(exc, "disk_restore load errors")
            mine = set()
        try:
            theirs = original(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the fork's fault stays the fork's
            snapshot._error(exc, "disk_restore load errors original")
            theirs = set()
        if not mine:
            return theirs
        try:
            return set(theirs or ()) | mine
        except Exception as exc:  # noqa: BLE001 - our set alone is still a report
            snapshot._error(exc, "disk_restore load errors union")
            return set(mine)
    _stamp(get_block_ids_with_load_errors, METHOD_LOAD_ERRORS, original)
    return get_block_ids_with_load_errors


def _install_worker_method(cls, name: str):
    """Wrap one worker method. A sibling layer is composed with, never refused.

    THE ORDERING RULE THAT MAKES FOUR WRAPPERS OF ONE METHOD WORK: this module lands
    OUTERMOST, so `worker_before_load` runs BEFORE the layers under it and
    `worker_get_finished` sees the value they produced. A wrapper carrying a marker of
    OURS is reused, a wrapper carrying ANOTHER PACKAGE's marker is refused outright,
    and anything else (the fork's own function, or a sibling composed with
    `functools.wraps` so `__module__` reads as vLLM's) is composed with.
    """
    current = cls.__dict__.get(name)
    if current is None:
        return (f"{cls.__name__}.{name} is not in the class dict, so the fork "
                f"changed shape")
    if _layer_is_ours(current):
        return _REUSED
    if not callable(current):
        return f"{cls.__name__}.{name} is not callable"
    if _layer_is_foreign(current):
        return (f"{cls.__name__}.{name} carries a wrapper from another owner: "
                "refusing to stack on an unknown layer")
    if getattr(current, "__module__", None) != cls.__module__:
        _state["nested"].append(f"{name}({getattr(current, '__name__', '?')})")
    setattr(cls, name, _WORKER_WRAPPERS[name](current))
    return None


def _install_sched_method(cls, name: str):
    """Wrap one scheduler entry point UNDER our existing layers.

    Uses `backport`'s own layer walk so the guard stays ON TOP: when the class
    attribute is already a guard or kill-switch layer, the restore layer is inserted
    at the BOTTOM (directly over the fork's body) and the layers above are rebuilt.
    That is the only ordering in which `guard_mode`'s fault counter can see an
    exception raised here.
    """
    from . import backport  # noqa: PLC0415 - the layer vocabulary lives there
    current = cls.__dict__.get(name)
    if current is None:
        current = getattr(cls, name, None)
    if current is None:
        return f"{cls.__name__}.{name} does not exist: the fork renamed it"
    if not callable(current):
        return f"{cls.__name__}.{name} is not callable"
    if _layer_is_ours(current):
        return _REUSED
    layers, bottom = backport._layers_of(current)
    if layers:
        kinds = [kind for kind, _ in layers]
        if LAYER_KIND in kinds:
            _state["nested"].append(f"{name}(already {LAYER_KIND} under {kinds})")
            return _REUSED
        if bottom is None or not callable(bottom):
            return (f"the ownership chain under our layers ({kinds}) does not end "
                    f"in a callable body")
        base = make_wrapper(name, bottom)
        if base is None:
            return f"{cls.__name__}.{name} has no restore hook"
        setattr(cls, name, backport._rewrap(name, base, layers))
        _state["nested"].append(f"{name}(under {'+'.join(kinds)})")
        return None
    if getattr(current, OWNER_ATTR, None) is not None \
            and _layer_is_foreign(current):
        return f"{cls.__name__}.{name} carries a foreign owner marker"
    base = make_wrapper(name, current)
    if base is None:
        return f"{cls.__name__}.{name} has no restore hook"
    setattr(cls, name, base)
    return None


def _install_pass(cls, names, installer, report) -> bool:
    """Run one install pass. True to continue, False when a refusal is fatal.

    A refusal is COUNTED AND LOGGED, not merely returned. This closes the failure
    mode that hid a bug for hours earlier in this project: an install that finds no
    method to wrap returns cleanly, a boot with nothing wrapped looks exactly like a
    boot with nothing to restore, and the only evidence that the fork renamed a
    method is a string in a dict nobody reads. `install_refusals` is the counter and
    `DISK[restore-install]` is the line; the suite asserts both, so a refusal cannot
    go silent in either direction.
    """
    global _reused_this_install
    for name in names:
        refusal = installer(cls, name)
        if refusal is _REUSED:
            _reused_this_install = True
            continue
        if refusal:
            report["reason"] = refusal
            _state["skip_reason"] = refusal
            _state["install_refusals"] += 1
            missing = f"{cls.__name__}.{name}"
            if missing not in _state["missing_methods"]:
                _state["missing_methods"].append(missing)
            emit_install_refusal(f"{cls.__name__}: {refusal}")
            return False
    return True


_reused_this_install = False

_install_refusal_lines: list = []


def _emit_install_line(reason: str, *, deferred: bool) -> dict:
    """ONE line per distinct install outcome per process; deferrals at INFO.

    THE WORDING SPLIT, and it matters more than it looks. At plugin load the tier
    classes are legitimately not imported yet - that is the ORDER of the engine, not a
    fault - and printing that at WARNING as `installed=NO reason=...` made the first
    line of every healthy boot look like a failure, so the real refusals (a fork that
    renamed a method, a wrapper we refuse to stack on) were indistinguishable from
    routine startup noise. Two shapes now:

      `deferred=YES refused=NO` at INFO    - nothing to hook YET. The boot is fine and
                                             `install()` will be called again by the
                                             process that can see the class.
      `deferred=NO refused=YES` at WARNING - the class was there and we could not fit
                                             it. This is the line to read.
    """
    try:
        flag = ("deferred=YES refused=NO" if deferred
                else "deferred=NO refused=YES")
        line = (f"DISK[restore-install] installed=NO role={role_here()} {flag} "
                f"reason={_clip(reason, MAX_BOOT_LINE_CHARS)}")
        if deferred:
            _state["install_deferral_line"] = line
        else:
            _state["install_refusal_line"] = line
        if line in _install_refusal_lines:
            return {"logged": False, "reason": "this line is already logged",
                    "line": line}
        _install_refusal_lines.append(line)
        del _install_refusal_lines[:-4]
        if deferred:
            _logger.info("kvtier-dev-plugin: %s", line)
        else:
            _logger.warning("kvtier-dev-plugin: %s", line)
        return {"logged": True, "reason": None, "line": line}
    except Exception as exc:  # noqa: BLE001 - a log line never breaks an install
        snapshot._error(exc, "disk_restore install refusal line")
        return {"logged": False, "reason": f"logging raised {type(exc).__name__}",
                "line": None}


def emit_install_refusal(reason: str) -> dict:
    """A real install refusal: the class was there and the method was not. WARNING."""
    return _emit_install_line(reason, deferred=False)


def emit_install_deferral(reason: str) -> dict:
    """Nothing to hook in this process YET. INFO, and worded as a deferral."""
    return _emit_install_line(reason, deferred=True)


def install(refit: bool = False, sched_cls=None) -> dict:
    """Fit the restore layer onto whichever halves of the tier this process has.

    Idempotent, never raises, and announces its verdict in a `finally` - the same
    posture as `disk_store.install`, for the same reason: the refusal is the line an
    operator needs, and a quiet install failure looks exactly like a boot with nothing
    to restore.

    THE HALVES INSTALL APART, because the processes do not share modules: an
    EngineCore has the manager (and the connector) but never imports the worker,
    while a worker has the worker and the connector but never the manager. Requiring
    both here would mean the layer could never fit anywhere. The cross-process risk
    that tolerance opens - a scheduler offering slots no worker will fill - is closed
    by `gate()` requiring `worker_proof()`, not by this function.

    IT IS CALLED WHEN THE CLASSES EXIST, NOT WHEN THE PLUGIN LOADS. `register()`
    calls it once, and in the EngineCore that is BEFORE the connector imports the
    manager (`cpu_capacity`'s docstring measures exactly that ordering), so the
    plugin-load call is a no-op there by construction. The retry that matters is
    driven by the process that can see the class it needs: the worker half by the
    handshake hook (`engine_side._on_worker_report`, which fires inside the
    WORKER-side `get_handshake_metadata` and therefore only in a worker), and the
    SCHEDULER half by `adopt_if_missed()` from the connector-creation boot
    (`backport.boot_from_connector`) and from the snapshot verb. Wiring the scheduler
    half to the worker's hook is the bug that made four ranks print `installed=YES`
    while the one process that could make an offer printed nothing: a worker fitting
    the SCHEDULER class is a wrapper no interpreter will ever call.
    `refit=True` re-runs the passes even when this process already believes it is
    fitted: `backport._rewrap()` rebuilds a method from a new base and can leave our
    layer off it, and a belief that survives the layer being gone is exactly how a
    half-wrapped class reads as healthy. Our own layers answer `_REUSED`, so a refit
    cannot double-stack - it can only fill a hole.

    `sched_cls` is the manager class a caller is HOLDING (the connector-creation
    boot has it). It is accepted only under the name this module wraps, and it is
    remembered, so a process whose manager arrived as an object rather than as a
    `sys.modules` entry can still be asked, afterwards, whether it can offer.
    """
    global _sched_cls_seen
    if _state["installed"] and not refit:
        return {"installed": True, "reason": "already installed",
                "class": _state["class"]}
    report = {"installed": False, "reason": None, "class": None}
    global _reused_this_install
    _reused_this_install = False
    worker_refusal = None
    try:
        sched_cls = _sched_class(sched_cls)
        worker_cls = _class_of(WORKER_MODULE, WORKER_CLASS)
        if sched_cls is None and worker_cls is None:
            report["reason"] = (
                f"neither {SCHED_MODULE}.{SCHED_CLASS} nor "
                f"{WORKER_MODULE}.{WORKER_CLASS} is imported in this process: "
                "there is nothing to hook here")
            _state["skip_reason"] = report["reason"]
            # Counted and logged like any other refusal - but as a DEFERRAL, not a
            # refusal. "Nothing imported" is the EXPECTED answer at plugin load in
            # every process (`register()` runs before the connector imports its
            # modules), and a healthy boot must not read as one that was refused.
            # What it must never be is the last thing said about an install that
            # never came back: the line prints once, and `install_deferrals` says
            # how many times this process asked and found nothing.
            _state["install_deferrals"] += 1
            emit_install_deferral(report["reason"])
            return report
        halves = []
        if sched_cls is not None:
            if not _install_pass(sched_cls, SCHED_METHODS, _install_sched_method,
                                 report):
                return report
            halves.append(sched_cls.__name__)
        if worker_cls is not None:
            if not _install_pass(worker_cls, WORKER_METHODS, _install_worker_method,
                                 report):
                return report
            error_refusal = _install_load_errors()
            if error_refusal:
                # NOT merely a gap: without the load-error report a failed read has
                # no way to say so, and a request would continue on KV that was never
                # restored. The worker half is refused rather than half-fitted; the
                # scheduler half, if this process also has one, is harmless on its
                # own only because `gate()` demands a worker's proof, and a worker
                # whose layer refused cannot produce that proof.
                _uninstall_worker_half()
                worker_refusal = (
                    f"the load-error report could not be fitted ({error_refusal}): "
                    "a failed read would have no way to say so, so the worker "
                    "half of the restore layer is refused")
                report["reason"] = worker_refusal
                _state["skip_reason"] = worker_refusal
                if not halves:
                    return report
            else:
                halves.append(worker_cls.__name__)
        if _reused_this_install:
            _state["reinstalls"] += 1
        else:
            _state["installs"] += 1
        _state["installed"] = True
        _state["class"] = "+".join(halves)
        # WHICH HALF, IN THIS PROCESS. `installed` alone was the reading that let a
        # worker's `YES` stand in for the scheduler's: record the halves and the
        # method names, and every line and payload this module prints can say which
        # of the two it is talking about.
        _state["sched_fitted"] = sched_cls is not None
        _state["worker_fitted"] = worker_cls is not None and not worker_refusal
        _state["methods_fitted"] = fitted_methods_here()
        if sched_cls is not None:
            # Remembered so `sched_fitted_here()` can answer for a class that is not
            # reachable by name (see `_sched_cls_seen`).
            _sched_cls_seen = sched_cls
        # A REFUSED HALF MUST STAY VISIBLE. `skip_reason` is what `boot_line()`
        # prints when the gate is open, so clearing it here used to erase the only
        # record that the worker half was refused - and "the worker half was
        # refused" is the one refusal an operator has to be able to see without a
        # post-mortem, because it means a failed read in this process can never be
        # reported. The scheduler half fitted is still reported as fitted (that is
        # true, and it is inert without a worker's proof); the refusal rides along
        # in the same sentence.
        _state["skip_reason"] = worker_refusal
        register_contribution()
        success = (f"{len(SCHED_METHODS)} scheduler and "
                   f"{len(WORKER_METHODS)} worker methods consult the SSD tier "
                   f"where this process has them ({'+'.join(halves)}); inert "
                   "until a root is set AND the spill arm AND the tier arm are "
                   "raised")
        if worker_refusal:
            success = f"{success}; {worker_refusal}"
        report.update({"installed": True, "class": _state["class"],
                       "reason": success})
        return report
    except Exception as exc:  # noqa: BLE001 - installation never breaks a boot
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore install")
        report["reason"] = f"install raised {type(exc).__name__}"
        _state["skip_reason"] = report["reason"]
        return report
    finally:
        emit_boot_line()


def _class_of(module_name: str, class_name: str):
    """A tier class this process has imported, or None. Never raises."""
    module = sys.modules.get(module_name)
    candidate = getattr(module, class_name, None) if module is not None else None
    return candidate if isinstance(candidate, type) else None


def _install_load_errors():
    """Fit `get_block_ids_with_load_errors` on the connector (worker process).

    The fork's connector does not define it - it inherits the base's `return set()`
    (`kv_connector/v1/base.py:405`), which the model runner calls every step
    (`v1/worker/gpu/kv_connector.py:116`). Installing an override on the FORK'S class
    is legitimate only when the inherited body is vLLM's own, so the home module is
    checked; a third-party report is refused rather than overwritten.
    """
    module = sys.modules.get(CONNECTOR_MODULE)
    cls = getattr(module, CONNECTOR_CLASS, None) if module else None
    if not isinstance(cls, type):
        return (f"{CONNECTOR_MODULE}.{CONNECTOR_CLASS} is not imported here, so "
                "load errors have nowhere to be reported")
    current = cls.__dict__.get(METHOD_LOAD_ERRORS)
    if current is not None and _layer_is_ours(current):
        _state["load_error_installs"] += 1
        return None
    if current is not None and not callable(current):
        return f"{cls.__name__}.{METHOD_LOAD_ERRORS} is not callable"
    inherited = current if current is not None else getattr(cls, METHOD_LOAD_ERRORS,
                                                            None)
    if not callable(inherited):
        return (f"{cls.__name__} exposes no {METHOD_LOAD_ERRORS}, so the engine "
                "would never learn which blocks failed to load")
    home = getattr(inherited, "__module__", "") or ""
    if not home.startswith("vllm."):
        # Checked for an INHERITED body and for an existing override alike: an
        # override that is neither ours nor vLLM's belongs to another package, and
        # wrapping it would silently adopt its facts as ours - or drop them.
        return (f"{METHOD_LOAD_ERRORS} is defined in {home!r}, not in vllm's "
                "own tree: refusing to overwrite a third party's load-error "
                "report")
    setattr(cls, METHOD_LOAD_ERRORS, _wrapped_load_errors(inherited))
    _state["load_error_installs"] += 1
    return None


def ensure_installed(connector=None) -> dict:  # noqa: ARG001 - the handshake's object
    """The worker-handshake entry point: install, ignoring the connector object.

    `engine_side._on_worker_report` calls every listener as `hook(connector)`; this
    module resolves its classes BY NAME from the already-imported modules rather
    than reaching into an instance, so the argument is genuinely unused - the same
    honest signature `disk_store.ensure_installed` has.

    THIS IS THE WORKER'S ENTRY POINT. It is reached only from the worker-side
    handshake producer (`cpu_capacity._call_worker_report_hook`), so a scheduler
    that depends on this call alone never installs - see `adopt_if_missed()` for the
    scheduler-side door and the reason it exists.
    """
    return install()


def adopt_if_missed(manager_cls=None) -> dict:
    """THE SCHEDULER-SIDE ENTRY POINT: fit the layer if this process can and has not.

    Called from the connector-creation boot (`backport.boot_from_connector`) and
    from the snapshot verb, because those run in the process the fork calls
    `get_num_new_matched_tokens` on, and they run AFTER the connector has imported
    the manager - which is the first moment the scheduler class exists at all. The
    cheap exit is a marker check: `install()` returns at once when the layer is on,
    and when the manager is not imported here it is two `sys.modules` lookups, so
    this is safe on a path a dashboard polls every few seconds.

    It never arms anything: the gate is `gate()`, and a fitted-but-disarmed layer is
    the normal, cost-free state of a fresh boot.

    `manager_cls` is the manager class the caller is holding, when it has one: the
    connector-creation boot passes `type(connector.scheduler_manager)` so the fit
    does not hinge on the fork's module being importable under the name this file
    expects. The snapshot verb passes nothing and gets the name-based lookup.
    """
    if _state["installed"] and sched_fitted_here():
        return {"installed": True, "reason": "already installed", "adopted": False,
                "class": _state["class"]}
    if _sched_class(manager_cls) is None \
            and _class_of(WORKER_MODULE, WORKER_CLASS) is None:
        return {"installed": bool(_state["installed"]), "adopted": False,
                "class": _state["class"],
                "reason": "no tier class is imported in this process yet, so there "
                          "is nothing here to fit"}
    report = install(refit=True, sched_cls=manager_cls)
    report["adopted"] = not _reused_this_install
    return report


def _uninstall_worker_half() -> None:
    """Undo the worker methods only - used when the load-error report cannot fit.

    The scheduler half is deliberately NOT undone here: in an EngineCore process the
    two halves are independent, and a half-removed worker layer would leave the class
    in a state no later install could reason about. The scheduler half is made safe by
    the gate, not by teardown.
    """
    cls = _class_of(WORKER_MODULE, WORKER_CLASS)
    if cls is None:
        return
    for name in WORKER_METHODS:
        current = cls.__dict__.get(name)
        if current is None or not _layer_is_ours(current):
            continue
        original = getattr(current, ORIGINAL_ATTR, None)
        if callable(original):
            setattr(cls, name, original)


def uninstall() -> dict:
    """Put the fork's methods back exactly. Test teardown and a clean undo."""
    report = {"uninstalled": False, "restored": [], "reason": None}
    try:
        for module_name, class_name, names in (
                (SCHED_MODULE, SCHED_CLASS, SCHED_METHODS),
                (WORKER_MODULE, WORKER_CLASS, WORKER_METHODS)):
            module = sys.modules.get(module_name)
            cls = getattr(module, class_name, None) if module else None
            if not isinstance(cls, type):
                continue
            for name in names:
                current = cls.__dict__.get(name)
                if current is None or not _layer_is_ours(current):
                    continue
                original = getattr(current, ORIGINAL_ATTR, None)
                if not callable(original):
                    # A restore layer UNDER a rebuilt guard has no restorable
                    # original of its own to hand back: the layer above it does, and
                    # restoring here would delete the guard. Say so, and leave it.
                    report["reason"] = (f"{cls.__name__}.{name} has no restorable "
                                        "original (a layer of ours sits above it)")
                    continue
                setattr(cls, name, original)
                report["restored"].append(f"{cls.__name__}.{name}")
        connector_module = sys.modules.get(CONNECTOR_MODULE)
        connector_cls = (getattr(connector_module, CONNECTOR_CLASS, None)
                         if connector_module else None)
        if isinstance(connector_cls, type):
            current = connector_cls.__dict__.get(METHOD_LOAD_ERRORS)
            if current is not None and _layer_is_ours(current):
                original = getattr(current, ORIGINAL_ATTR, None)
                if callable(original):
                    setattr(connector_cls, METHOD_LOAD_ERRORS, original)
                else:
                    delattr(connector_cls, METHOD_LOAD_ERRORS)
                report["restored"].append(f"connector.{METHOD_LOAD_ERRORS}")
        _state["uninstalls"] += 1
        _state["installed"] = False
        _state["sched_fitted"] = sched_fitted_here()
        _state["worker_fitted"] = False
        _state["methods_fitted"] = fitted_methods_here()
        report["uninstalled"] = bool(report["restored"])
        if not report["uninstalled"] and not report["reason"]:
            report["reason"] = "no layer of ours was installed on any class"
        return report
    except Exception as exc:  # noqa: BLE001 - a teardown never raises
        snapshot._error(exc, "disk_restore uninstall")
        report["reason"] = f"uninstall raised {type(exc).__name__}"
        return report


# --- the view ----------------------------------------------------------------

def section() -> dict:
    """This module's state for a payload or a verb. Reads no files, never raises."""
    view = counters()
    view["gate"] = gate()
    view["deadline_seconds"] = RESTORE_DEADLINE_SECONDS
    view["limits"] = {
        "max_restore_blocks_per_offer": MAX_RESTORE_BLOCKS_PER_OFFER,
        "max_slots_per_offer": MAX_SLOTS_PER_OFFER,
        "max_tracked_loads": MAX_TRACKED_LOADS,
        "max_tracked_slots": MAX_TRACKED_SLOTS,
        "max_paths_per_step": MAX_PATHS_PER_STEP,
        "cpu_slot_reserve": CPU_SLOT_RESERVE,
        "identity_digest_placeholder": IDENTITY_DIGEST_PLACEHOLDER,
        # THE BOUND ONE OFFER MAY CARRY (0.2.32), as the blocks that define it plus the
        # tokens they are worth and where that block size came from. The fixed
        # 32,000-token cap it replaces is DELETED, not renamed: a number without its
        # derivation is what made the old cap impossible to argue about from a payload,
        # so the whole `offer_block_ceiling()` dict travels here.
        "offer_block_ceiling": offer_block_ceiling(),
    }
    average, maximum = read_latency()
    view["read_latency_seconds"] = {"avg": average, "max": maximum}
    # THE SHAPE OF A PARTIAL RESTORE, beside the read latency and the wait that
    # produce it. Counting only: `span_shape` decides nothing about tokens or pruning,
    # and the all-or-nothing question belongs to a different change.
    view["spans"] = {
        "observed": int(_state["spans_observed"]),
        "entries_observed": int(_state["span_entries_observed"]),
        "whole": int(_state["spans_whole"]),
        "partial": int(_state["spans_partial"]),
        "empty": int(_state["spans_empty"]),
        "hole_in_the_middle": int(_state["spans_hole_in_the_middle"]),
        "last": _state["last_span_shape"],
        "note": ("a span is one scheduler block across every cacheable group, so it "
                 "is either restored whole or worth nothing to the engine. "
                 "`hole_in_the_middle` counts batches whose applied spans are NOT a "
                 "prefix of the offer: with one reader thread that is arithmetically "
                 "impossible, so a rising count is direct evidence that N readers "
                 "changed the shape of an interrupted restore, not only its speed.")}
    # THE WINDOW AND THE WAIT, next to the read latency they are measured against:
    # `waited_ms` without `read_seconds` beside it is a number with no unit of
    # comparison, and the whole point of the settle-wait fields is to be comparable.
    view["apply_window"] = apply_window()
    # WHY A READ DID NOT HAPPEN, in the payload. `reads_failed` counts the kinds; this
    # names the requests, the blocks and - for `buffer_cap` - the five numbers that
    # decided it. THIS PROCESS ONLY: the cross-process view is `load_failure_report()`
    # and the scheduler's `recent_load_failures`.
    view["load_failures"] = {"written": int(_state["load_failures"]),
                             "overwritten": int(_state["load_failures_overwritten"]),
                             "ring_depth": MAX_LOAD_FAILURE_RECORDS,
                             "log_lines": int(_state["failure_log_lines"]),
                             "log_suppressed": int(_state["failure_log_suppressed"]),
                             "log_min_seconds": REFUSAL_LOG_MIN_SECONDS,
                             "recent": load_failures()},
    # WHAT AN OFFER COSTS THE POOL, as this process currently prices it. The same
    # arithmetic the offer cap uses, so a reader can check an offer against the pool
    # without re-deriving `afford_reads` by hand.
    view["offer_pool"] = _offer_pool_view()
    view["witness"] = witness_verdict()
    view["witness_line"] = witness_line()
    view["index"] = disk_index.section()
    view["read"] = disk_read.section()
    view["worker_proof"] = worker_proof()
    view["offer_refusal_reasons"] = list(OFFER_REFUSALS)
    view["worker_failure_reasons"] = list(WORKER_FAILURES)
    view["note"] = ("the SSD restore path: it EXTENDS the fork's own load with "
                    "spans read back from disk, publishes nothing into any hash "
                    "index, and reports every failure as a load error so the "
                    "scheduler recomputes instead of trusting bytes it never "
                    "received. Inert until a root is set AND the spill tier's arm "
                    "AND the tier's arm are raised.")
    return view


def scheduler_section() -> dict:
    """THE one-call answer to "why is there no restore?", from THIS process.

    WHY THIS EXISTS. The restore path lives in two processes and only ONE of them -
    the scheduler - can start a restore, yet everything this module published about
    itself was readable as a single verdict. A boot in which four ranks fitted the
    layer and the scheduler fitted nothing printed four `installed=YES` lines and no
    answer, and the truth was only reachable by counting process names in a 200,000
    line log. This section is the same facts split by role, in the payload a poller
    already reads, with the chain walked for the reader: the layer, the gate, the
    index, the worker proof, the offers - `why_no_restore` names the first link that
    is missing, so one `GET /plugins/kvtier/snapshot` (or the `/disk` verb) settles
    the question without a grep.

    Reads no KV byte, walks no directory beyond the rank files `worker_proof()`
    already reads through its cache, and never raises.
    """
    here = fitted_methods_here()
    missing = [name for name in SCHED_METHODS if name not in here]
    opened = gate()
    proof = worker_proof()
    index = disk_index.section()
    state = _state
    # THE POSTURE, FROM ITS OWNERS. `disk_posture()` owns the two boot switches and
    # `counters()` owns every suppression number; this section re-publishes them and
    # recomputes nothing, so a payload and a verb can never disagree about a posture.
    posture = disk_posture()
    tally = counters()
    # THE PER-OFFER BOUND, computed ONCE for the whole payload (0.2.32). Like
    # `_offer_pool_view()`, this is a re-derivation rather than a stored value: the
    # engine's block size is the authority and a payload that published a cached figure
    # could disagree with the next offer. ONE call per payload read, not one per field.
    offer_ceiling = offer_block_ceiling()
    # THE INDEX'S OWN BLINDNESS, computed once for every branch below (0.2.28).
    blindness = _index_blindness()
    # THE VERDICT AS OF THIS INSTANT, for `witness_*_now`. Computed here, not stored "
    # anywhere: `witness_now()` writes nothing, so a payload poll cannot rewrite the "
    # as-of-emission fields the log dedupe depends on.
    now = witness_now()
    offers = int(state["offers"])
    entries = int(index.get("entries") or 0)
    # From the PROOF, not only from the counter: `worker_proof()` publishes the
    # count it read and a caller that patched the door (a test, or a future cached
    # verdict source) must still be reported honestly.
    ranks = int(proof.get("count") or state["ranks_proven"] or 0)
    installed_here = bool(_state["sched_fitted"]) and not missing
    last_refusal_line = (state["install_refusal_line"]
                         or state["skip_reason"] or "none")
    if not installed_here:
        why = ("the scheduler half of the restore layer is not fitted in THIS "
               f"process (wrapped here: {here or 'none'}; role here: "
               f"{role_here()}). No offer is possible: "
               f"{SCHED_METHODS[0]} is not ours here. Refusals this boot: "
               f"{int(state['install_refusals'])} "
               f"(deferrals: {int(state['install_deferrals'])}); last reason: "
               f"{last_refusal_line}")
    elif not opened["ok"]:
        why = (f"the gate is closed: {opened['reason']}")
    elif entries == 0:
        why = ("the scheduler-side index holds no identities, so nothing is known "
               f"to be on disk (journals tracked: "
               f"{int(index.get('journals_tracked') or 0)}; "
               f"{index.get('journal_directory')}). A fresh boot knows nothing "
               "until its ranks have spilled.")
        # THE BLINDNESS, NAMED, IN THE SENTENCE AN OPERATOR READS FIRST. An empty
        # index over a store that is not empty has exactly two loud causes: the
        # writers could not record their lines, or the journal directory cannot be
        # written at all. Say which, because "entries: 0" alone is a mystery that
        # costs an afternoon (2026-10-05: 107 GB of KV bytes and no map).
        missed = int(index.get("journal_write_errors") or 0)
        if index.get("journal_directory_writable") is False:
            why += (" The JOURNAL DIRECTORY CANNOT BE WRITTEN ("
                    f"{index.get('journal_directory_error')}), so no spill this "
                    "boot makes can ever be indexed.")
        elif missed:
            why += (f" {missed} journal line(s) could not be recorded "
                    f"(journal_last_error: {index.get('journal_last_error')}): "
                    "those files exist and this index cannot name them.")
        # "entries: 0 over journals that are not empty" has a third cause besides
        # those two (0.2.28): the drain has simply not REACHED the spill lines yet.
        # Say it, because "A fresh boot knows nothing until its ranks have spilled"
        # is a lie about a box that spilled a long time ago.
        if blindness["blind"]:
            why += " " + blindness["sentence"]
    elif ranks == 0:
        why = (f"no rank file proves a fitted worker half ({proof.get('reason')}): "
               "offering disk tokens would DMA CPU slots nobody fills, so the "
               "refusal is the design working")
    elif offers == 0:
        why = ("the layer is fitted, the gate is open, the index holds "
               f"{entries} identities and {ranks} rank(s) are proven - but no "
               "offer has been made. Read `refused`: the per-request refusals "
               "(usually `no_disk_run`: the spilled spans do not continue the "
               "prefix this request asked for)")
        # THE ANSWER BEHIND THE ANSWER (0.2.28: the WHOLE blindness, not just the
        # lag). A span spilled minutes ago is invisible to an index that has not
        # read its journal's drain, and a span whose lines an OLDER build forfeited
        # is invisible forever - and after a forfeit `lag_bytes` is legitimately 0,
        # which is how the `entries: 146` over 132,099 records stayed silent for
        # hours. `_index_blindness` names the bytes, the journals and the skips,
        # and says plainly that such spans are NOT KNOWABLE, so `no_disk_run` reads
        # as "spilled but not yet indexed" rather than "never spilled".
        if blindness["blind"]:
            why += " " + blindness["sentence"]
        # THE ARCHITECTURAL ANSWER, IN THE FIRST SENTENCE (0.2.25). An operator who
        # reads `why_no_restore` at 3 a.m. should not have to know what a MambaSpec is,
        # let alone that its recurrent state has no per-block file name: say that the
        # MODEL is un-serviceable, name the groups that decided it, and name the
        # refusal that will be counting every lookup.
        service = state["group_serviceability"]
        if service and not service.get("complete", True):
            why += (f" THIS MODEL CANNOT BE SERVED FROM DISK AT ALL: "
                    f"prefix-cacheable group(s) "
                    f"{service.get('blocked_group_ids')} have no block contract this "
                    f"tier can price (see `group_serviceability.group_contracts`) "
                    f"while {service.get('restorable_group_ids')} do, and a disk "
                    "offer claims EVERY prefix-cacheable group because "
                    "get_num_new_matched_tokens skips prefill model-wide. An "
                    "align-mode GDN group is NOT in this list: its boundary-state "
                    "contract is restorable (0.2.27). Every lookup here is refused "
                    "as `incomplete_group_set` and recompute is the only correct "
                    "answer on this architecture.")
            needed = state["last_groups_required"] or {}
            if needed.get("unfilled_group_ids"):
                why += (f" What the refused claim needs: "
                        f"{needed.get('claimed_blocks')} block(s) "
                        f"({needed.get('claimed_tokens')} tokens) would need "
                        f"{needed.get('blocks_needed_total')} block(s) across the "
                        f"cacheable groups and fills none of "
                        f"{needed.get('unfilled_group_ids')} - see "
                        "`last_groups_required.rows` for the per-group arithmetic "
                        "(align-mode state groups need ONE boundary block each, the "
                        "rest of the span is null blocks in that group anyway).")
    else:
        why = (f"restore is live in this process: {offers} offer(s), "
               f"{int(state['offered_tokens'])} tokens, "
               f"{int(state['blocks_offered'])} scheduler blocks; "
               f"{int(state['restorations'])} load(s) reported disk bytes back. "
               "An offer that is not adopted is a scheduler decision, not this "
               "module's: see lookup_ledger.")
        # THE SENTENCE MUST NOT STOP BEING TRUE WHEN THE READS MOVE. `restorations`
        # is the counter that reads as success, and on the live box it sat at 0 next
        # to `ranks_reads_applied=42` while the sentence explained nothing: the reads
        # had happened and the requests had still been rescheduled. When those two
        # disagree, say what a partial load does to a request instead of leaving the
        # reader to reconcile two numbers from different processes.
        partial = partial_load_verdict()
        if partial["reads_applied"] and not partial["restorations"]:
            why += " " + partial["sentence"]
            if int(state["offer_pool_refused"]) or int(state["offer_pool_capped"]):
                why += (f" The bounce-buffer pool has bounded this boot's offers: "
                        f"{int(state['offer_pool_capped'])} shortened, "
                        f"{int(state['offer_pool_refused'])} refused outright "
                        "(see `offer_pool`).")
            kinds = sorted(_recent_failure_kinds())
            if kinds:
                why += f" Worker-side failure kinds seen: {kinds[:6]}."
    return {
        # WHICH PROCESS ANSWERED. The single most important field on the page: a
        # `true` here and a rank file's `installed: true` are different facts about
        # different processes, and only this one can start a restore.
        "layer_installed_here": installed_here,
        "role": role_here(),
        "class_here": _state["class"],
        "sched_fitted": bool(_state["sched_fitted"]),
        "worker_fitted_here": bool(_state["worker_fitted"]),
        "methods_expected": list(SCHED_METHODS),
        "methods_here": here,
        "methods_missing": missing,
        "install_refusals": int(state["install_refusals"]),
        "install_deferrals": int(state["install_deferrals"]),
        "install_refusal_line": state["install_refusal_line"],
        "missing_methods": list(state["missing_methods"]),
        "installs": int(state["installs"]),
        "reinstalls": int(state["reinstalls"]),
        "gate": {"ok": bool(opened["ok"]), "reason": opened.get("reason"),
                 "reason_kind": opened.get("reason_kind")},
        # THE POSTURE AS DATA, not as a sentence inside `boot_line`. Until now the only
        # place any of this was printed was that line, and a dashboard cannot grep a
        # sentence at 3 a.m.: "is the switch up, which switch, how many ranks are
        # armed, is anything fused" was only answerable by reading prose. Every value
        # is a pass-through from the function that owns it - `disk_posture()`,
        # `counters()`, `worker_proof()`, and for the cap the module constant the boot
        # line already prints - and none of them is derived here.
        #
        # `disk_authority` is the one an operator reads to answer "is this boot serving
        # EVERY request from SSD?", and `per_boot` / `survives_restart` say what its
        # lifetime is: raised only by `POST /plugins/kvtier/disk {"op":"authority"}`,
        # lost by a restart, so a boot that crashed and came back gives disk authority
        # to nobody.
        "disk_authority": bool(posture["disk_authority"]),
        "offers_enabled": bool(posture["enabled"]),
        "serving_as": posture["serving_as"],
        # THE BOOT DEFAULT BEHIND THE POSTURE (0.2.33): whether
        # `KVTIER_DISK_BOOT_ENABLE` asked for this boot to come up armed and serving,
        # whether it was honoured, and whether an operator disarm has latched a veto
        # this process cannot lift by itself. (0.2.35: the arm is a latch, so the view
        # answers `arm_expires: false` where it used to answer a countdown.)
        # `disk_boot.view()` answers with every key on every path, so a degraded boot
        # cannot hand back a thinner payload than a healthy one - and nothing here
        # recomputes any of it: the view is the same dict the `/disk` verb answers
        # with, from the module that owns the decision.
        "boot_default": disk_boot_section(),
        "per_boot": True,
        "survives_restart": False,
        "disk_authority_changes": int(posture["disk_authority_changes"]),
        "offer_switch_changes": int(posture["offer_switch_changes"]),
        # THE BOUND THAT REPLACES `offer_cap`, and `offer_cap` IS GONE (0.2.32): it
        # advertised the fixed 32,000-token cap this release deletes, and a field with
        # that name reporting a number that constrains nothing is worse than no field.
        # What is published instead is the bound that does constrain - the per-offer
        # BLOCK walk - in blocks (the real quantity) and in the tokens those blocks are
        # worth, with the block size and its source so a reader can see the token figure
        # is derived, not measured. `offer_block_capped` is the counter of offers that
        # bound here, and for every offer that came in short for a DIFFERENT reason,
        # `bounded_by` names the one that bit (`pool_budget` / `cpu_slots` /
        # `block_cap`).
        # THE HONEST READING, and the payload says it because the number invites the
        # opposite one: this bound is not what makes a long prefix restorable. Pool
        # affordance binds far below it in practice, and on the live store the depth the
        # spill tier reached (~11 spans) binds earlier still.
        "offer_block_cap_blocks": int(offer_ceiling["blocks"]),
        "offer_block_cap_tokens": int(offer_ceiling["tokens"]),
        "offer_block_cap_block_size": int(offer_ceiling["block_size"]),
        "offer_block_cap_block_size_source": str(offer_ceiling["block_size_source"]),
        "offer_block_capped": int(state["offer_block_capped"]),
        "ranks_armed": int(proof.get("ranks_armed") or 0),
        "ranks_seen": int(proof.get("ranks_seen") or 0),
        "suppressed_hits": int(tally["suppressed_hits"]),
        "suppressed_spans": int(tally["suppressed_spans"]),
        "suppressed_requests": int(tally["suppressed_requests"]),
        # The lifetime total beside the live table size: one failed load whose fuse has
        # since aged out and a hundred of them both read as `suppressed_requests=0`.
        "suppressions": int(tally["suppressions"]),
        "offers": offers,
        "tokens_offered": int(state["offered_tokens"]),
        "blocks_offered": int(state["blocks_offered"]),
        "slots_allocated": int(state["slots_allocated"]),
        "restorations": int(state["restorations"]),
        "adopted": int(state["adopted"]),
        "adopted_tokens": int(state["adopted_tokens"]),
        # THE DIRECT EVIDENCE THAT A READ HAPPENED, passed through from `counters()`
        # with no recomputation here.
        #
        # WHY THESE ARE IN THE PAYLOAD AT ALL. `restorations` is the counter an
        # experiment reads as success, and on 2026-10-05 it reported CONFIRMED while
        # nothing was read. The honest applied-read counter that existed in memory
        # (`reads_applied`) was never sent, so a predicate had to choose between the
        # costume counter and the indirect `adopted_tokens`. `reads_applied` means
        # literally "a read was memmoved into a slot", and it is now reachable from
        # the same one call as everything else. `disk_reads_applied` /
        # `ranks_reads_applied` are the scheduler-side per-offer verdict and the
        # worker-attested total; the three `witness_*` are the verdict allowed to use
        # them. `loads_released` completes the ledger: the reads that finished and were
        # let go.
        "reads_applied": int(tally["reads_applied"]),
        "disk_reads_applied": int(tally["disk_reads_applied"]),
        "ranks_reads_applied": int(tally["ranks_reads_applied"]),
        "loads_released": int(tally["loads_released"]),
        "lookups": int(tally["lookups"]),
        # WHAT THE AUTHORITY SWITCH DID. The switch hands the disk the decision about
        # every request's own prefix; these four say whether that rule fired, whether it
        # cost a real host-RAM hit, and how many tokens that was. `last_offer` is the
        # structured proof a forced read and an ordinary one disagree about: a real
        # restore has `ram_tokens_reported` at or below what it used, a forced one
        # shows the hit the disk overrode.
        "authority_lookups": int(tally["authority_lookups"]),
        "local_hit_ignored": int(tally["local_hit_ignored"]),
        "local_hit_tokens_ignored": int(tally["local_hit_tokens_ignored"]),
        "pin_release_failures": int(tally["pin_release_failures"]),
        "last_local_hit_ignored": tally["last_local_hit_ignored"],
        "last_offer": tally["last_offer"],
        # WHAT THE OFFER COSTS THE BOUNCE-BUFFER POOL (0.2.31). The bound that keeps an
        # offer inside what THIS process can carry, the arithmetic behind it, and the
        # two counters that say how often it bound. Since 0.2.32 deletes the fixed
        # 32,000-token cap this is the binding bound in practice: with the block walk
        # the only cap left above it, `last_offer_pool.bounded_by` naming `pool_budget`
        # is the normal answer for a long prefix - and the ONLY guard against the
        # 2026-10-05 shape of un-placeable offers, since nothing arbitrates two
        # simultaneous offers against the pool's current free bytes (TODO 14).
        "offer_pool": _offer_pool_view(),
        "offers_pool_capped": int(tally["offer_pool_capped"]),
        "offers_pool_refused": int(tally["offer_pool_refused"]),
        "last_offer_pool": tally["last_offer_pool"],
        # WHY A READ DID NOT HAPPEN, across the fleet. The records the workers wrote
        # (carried by their rank files, the `ranks_reads_applied` transport) merged
        # with this process's own, newest first, each carrying its block provenance.
        "recent_load_failures": load_failure_report()["records"],
        "load_failure_kinds": sorted(_recent_failure_kinds()),
        # WHAT EACH KIND IS ALLOWED TO COST (0.2.32). The partition is the policy for
        # what a failed load costs, so it travels in the payload rather than only in a
        # comment: a reader can check `not_settled` against `races_not_fused` and see
        # whether a box is losing reads to a race it survives or to bytes it does not.
        "failure_kind_classes": failure_kind_classes(),
        "races_not_fused": int(tally["races_not_fused"]),
        "fuses_on_data_kinds": int(tally["fuses_on_data_kinds"]),
        "fuses_on_unknown_kinds": int(tally["fuses_on_unknown_kinds"]),
        "race_reoffers_granted": int(tally["race_reoffers_granted"]),
        "last_fuse_decision": tally["last_fuse_decision"],
        "load_failures_here": int(tally["load_failures"]),
        "load_failures_from_ranks": int(tally["ranks_load_failures"]),
        # THE BLOCK IDS THE FORK BLAMED, kept instead of discarded, so a scheduler-side
        # burn can be matched to a worker-side record by block id.
        "invalid_block_ids_seen": int(tally["invalid_block_ids_seen"]),
        "last_invalid_block_ids": tally["last_invalid_block_ids"],
        # WHAT A PARTIAL LOAD DOES, in data and in a sentence.
        "partial_load": partial_load_verdict(),
        # WHAT THE LAST PRICED CLAIM COSTS EVERY CACHEABLE GROUP (item 7, 0.2.25), on
        # the refusal door as well as the offer door: `last_offer` cannot describe an
        # offer that correctly never happened, and this can. `offered: false` plus rows
        # with `restored: false` IS the 00:08 failure, in one object.
        "last_groups_required": tally["last_groups_required"],
        # WHICH GROUPS THIS MODEL CAN EVER SERVE FROM DISK (0.2.25). The one object
        # that answers "is the disk tier un-serviceable on this architecture, and how "
        # do you know?" without re-deriving three manager predicates by hand: the "
        # cacheable set, the stable set, the restorable set, and which groups blocked "
        # the offer with a reason each. Beside `last_refusal` because `refused." "
        # incomplete_group_set` is the counter this arithmetic explains.
        "group_serviceability": tally["group_serviceability"],
        "last_aged_out": tally["last_aged_out"],
        # THE LEDGER'S OWN HONESTY, PUBLISHED (0.2.25). `no_read` and `watchdog_fires`
        # existed in `counters()` and were never sent, so the release-without-
        # restoration branch had to be INFERRED from `suppressions`/`loads_released` on
        # the 00:08 incident instead of read. Same rule as every field here: pass-
        # through from `counters()`, recomputed nowhere.
        "no_read": int(tally["no_read"]),
        "watchdog_fires": int(tally["watchdog_fires"]),
        "invalid_reports": int(tally["invalid_reports"]),
        "aged_out_loads": int(tally["aged_out_loads"]),
        "aged_out_with_applied_reads": int(tally["aged_out_with_applied_reads"]),
        "late_finished_reports": int(tally["late_finished_reports"]),
        "restored_after_release": int(tally["restored_after_release"]),
        # THE ONLY COUPLING BETWEEN OUR CLAIM AND THE FORK'S DMA (0.2.25), in GPU
        # BLOCKS: the blocks our emitted paths fed, that the fork then reported
        # complete. `restorations` counts requests; this counts the blocks that were
        # actually written, which is the question a KV-tier experiment is really asking.
        "disk_gpu_blocks_dma_verified": int(
            tally["disk_gpu_blocks_dma_verified"]),
        "disk_gpu_blocks_released_unreported": int(
            tally["disk_gpu_blocks_released_unreported"]),
        "witness_verdict": tally["witness_verdict"],
        "witness_reason": tally["witness_reason"],
        "witness_kind": tally["witness_kind"],
        # THE SAME VERDICT, COMPUTED FROM THE COUNTERS AS THEY STAND (0.2.25). The two "
        # above are AS OF THE LAST EMISSION (that is what the log-line dedupe keys on) "
        # and can name a state the boot has already left: on the live box "
        # `witness_kind=offers_not_adopted` was published while `adopted=1`. Both are "
        # published now, labelled for what they are, and neither mutates the other.
        "witness_verdict_now": now["verdict"],
        "witness_kind_now": now["kind"],
        "witness_reason_now": now["reason"],
        "witness_kind_is_stale": now["stale_as_of_emission"],
        "refused": {name: int(state["refused"].get(name) or 0)
                    for name in OFFER_REFUSALS},
        "refusal_reasons": {name: OFFER_REFUSAL_WHY.get(
            name, "undefined reason: this refusal has no entry in "
                   "OFFER_REFUSAL_WHY") for name in OFFER_REFUSALS},
        "last_refusal": state["last_refusal"],
        "index": {"entries": entries,
                  "source": index.get("journal_directory"),
                  "bytes_indexed": int(index.get("bytes_indexed") or 0),
                  "journals_tracked": int(index.get("journals_tracked") or 0),
                  "refreshes": int(state["index_refreshes"]),
                  "refresh_errors": int(state["index_refresh_errors"]),
                  # THE INDEX'S OWN BLINDNESS, counted and named: lines the writers
                  # could not record (so files exist that this index cannot name)
                  # and whether the journal directory can be written at all. An
                  # `entries: 0` over a store that is not empty is explained by
                  # these two fields or it is not explained at all.
                  "journal_write_errors": int(
                      index.get("journal_write_errors") or 0),
                  "journal_last_error": index.get("journal_last_error"),
                  "journal_directory_writable": index.get(
                      "journal_directory_writable"),
                  # HOW MUCH OF THE JOURNAL THIS INDEX HAS NOT READ. While
                  # `lag_bytes` is non-zero a `not_present` from the index is a
                  # statement about the READER, not about the store - the shape that
                  # made 36 of 36 just-spilled identities look absent on 2026-10-05.
                  "lag_bytes": int(index.get("lag_bytes") or 0),
                  "lag_journals": int(index.get("lag_journals") or 0),
                  "catch_up_jumps": int(index.get("catch_up_jumps") or 0),
                  "bytes_skipped": int(index.get("bytes_skipped") or 0),
                  "lines_skipped": int(index.get("lines_skipped") or 0),
                  # THE DRAIN, PASSED THROUGH (0.2.28): `converged` is the answer to
                  # "may this index answer 'not present' at all?", and
                  # `unread_journals` names what it has not read yet. A snapshot
                  # reader must not have to recompute either from a lag number.
                  "converged": bool(index.get("converged")),
                  "boot_drained": bool(index.get("boot_drained")),
                  "unread_journals": dict(index.get("unread_journals") or {}),
                  "bytes_read": int(index.get("bytes_read") or 0),
                  "entry_cap_hit": bool(index.get("entry_cap_hit")),
                  "blind": blindness["blind"],
                  "blindness": blindness["sentence"] or None,
                  "last_reason": index.get("last_reason")},
        "ranks_proven": ranks,
        "digests_expected": ranks,
        "ranks": list(proof.get("ranks") or []),
        "proof_reads": int(state["proof_reads"]),
        "proof_reason": proof.get("reason"),
        "deadline_seconds": RESTORE_DEADLINE_SECONDS,
        "boot_line": _state["boot_line"],
        "why_no_restore": why,
        "units": {
            "install_refusals": "install passes refused here because a method was "
                                "missing or refused",
            "install_deferrals": "attempts that found no tier class imported yet "
                                 "(expected at plugin load, not a refusal)",
            "installs": "times the layer was fitted in this process",
            "reinstalls": "times a pass found our layer already on the method",
            "offers": "requests this process offered a disk span to",
            "tokens_offered": "tokens offered beyond the host-RAM hit",
            "blocks_offered": "scheduler blocks offered from disk",
            "slots_allocated": "CPU slots allocated as disk destinations",
            "restorations": "loads that reported disk bytes back",
            "adopted": "requests whose disk slots reached a worker",
            "adopted_tokens": "tokens the scheduler adopted from disk",
            "refused": "refusals, one counter per reason",
            "last_refusal": "refusals",
            "index.entries": "identities known to be on disk",
            "index.bytes_indexed": "bytes",
            "index.journals_tracked": "journal files",
            "index.refreshes": "journal replays",
            "index.refresh_errors": "raised replays",
            "index.journal_write_errors": "journal lines the writers could not "
                                          "record: spilled files this index was "
                                          "never told about, so they cannot be "
                                          "offered (never a wrong claim, always a "
                                          "recompute)",
            "index.lag_bytes": "bytes of journal this process has NOT read behind "
                               "its cursor. While it is non-zero the index cannot "
                               "name the newest spills, so a `not_present` describes "
                               "the reader and not the store",
            "index.lag_journals": "rank journals still carrying unread bytes",
            "index.catch_up_jumps": "times a journal was too deep for one refresh "
                                    "to reach its tail, so the older bytes were "
                                    "forfeited and that rank's entries forgotten",
            "index.bytes_skipped": "journal bytes forfeited to reach the tail (unit: "
                                   "bytes, never a guessed line count)",
            "index.lines_skipped": "journal lines permanently unread: the lines "
                                   "after a journal that stopped being trustworthy "
                                   "(the drain can never repair this one)",
            "index.converged": "true when every journal cursor sits at "
                               "end-of-file; while false a `not_present` from the "
                               "index describes the READER, not the store - the "
                               "drain has not reached the line yet",
            "index.boot_drained": "whether the first-refresh drain has run in "
                                  "this process",
            "index.unread_journals": "per-journal bytes behind the cursor, named "
                                     "(unit: bytes per journal; the drain in flight "
                                     "made visible as files, not just a total)",
            "index.bytes_read": "bytes this process has read out of the journals "
                                "since boot (unit: bytes) - the refresh bound is "
                                "auditable through it",
            "index.entry_cap_hit": "true once the identity cap dropped even one "
                                   "entry: the index answers `not_present` for "
                                   "files that ARE on disk; a capped index is not "
                                   "a caught-up index",
            "index.blind": "true when this index cannot answer 'is it on disk?' "
                           "for spans living in an unread or forfeited region - "
                           "the one reading that turns `no_disk_run` from 'never "
                           "spilled' into 'spilled but not yet indexed'",
            "index.blindness": "the sentence `why_no_restore` prints when `blind` "
                               "is true (names, bytes and journals included)",
            "ranks_proven": "rank files attesting a fitted worker half",
            "digests_expected": "distinct geometry digests a restore must satisfy "
                                "(one per proven rank: the WORKER names the file "
                                "with its own digest, so four ranks are four names "
                                "per identity, not one)",
            "proof_reads": "RANK FILES read for that proof - NOT KV reads. "
                           "`reads_applied` is the applied-disk-read counter; this "
                           "one counts how many rank files `worker_proof()` opened, "
                           "and the two must never be read as the same kind of "
                           "number",
            "reads_applied": "reads memmoved into a CPU slot (THE direct evidence "
                             "that a disk read was applied; pass-through from "
                             "counters()). In a scheduler-only process this stays 0 "
                             "by construction - the worker reads - so read it "
                             "beside `ranks_reads_applied` and `witness_verdict`",
            "disk_reads_applied": "scheduler-side applied disk reads (0 in a "
                                  "scheduler-only process by construction), from "
                                  "counters()",
            "ranks_reads_applied": "applied disk reads attested by live worker rank "
                                   "files at the last proof read, from counters()",
            "offer_pool": "what an offer costs the BOUNCE-BUFFER POOL: the bytes one "
                          "read needs, the pool's budget, how many jobs it can carry, "
                          "and the last offer's pricing with `bounded_by` naming which "
                          "constraint shortened it (`pool_budget` = bounce buffers, "
                          "`cpu_slots` = CPU blocks, `block_cap` = the per-offer block "
                          "walk). `offer_cap` is RETIRED with the fixed 32,000-token "
                          "cap: no configured token cap exists to bound an offer",
            "offers_pool_capped": "offers SHORTENED to the job count the bounce-buffer "
                                  "pool can carry, from counters()",
            "offers_pool_refused": "lookups given NO disk offer because the pool "
                                   "cannot carry even one span's reads (`no_pool_"
                                   "budget`), from counters()",
            "last_offer_pool": "the byte arithmetic of the most recent pricing: price "
                               "per job and its source, the pool budget, jobs needed "
                               "versus jobs the pool can carry, and `bounded_by`",
            "recent_load_failures": "the load-failure records an operator can read "
                                    "without opening a rank file: this process's ring "
                                    "merged with what the live rank files carried, "
                                    "newest first, each with its block provenance",
            "load_failure_kinds": "the failure kinds seen most recently (this "
                                  "process's ring plus the live rank files), from "
                                  "`_recent_failure_kinds`",
            "failure_kind_classes": "THE PARTITION of `WORKER_FAILURES` into timing "
                                    "/ capacity / data, with the decision rule beside "
                                    "it: "
                                    "a RACE may cost a request its tokens and must not "
                                    "cost its next offer, a DATA failure keeps the "
                                    "fuse, and an UNKNOWN report fuses (fail closed)",
            "races_not_fused": "failure reports classified a race, so no request fuse "
                               "and no span fuse was written, from counters()",
            "fuses_on_data_kinds": "fuses raised with a data kind visible (corruption, "
                                   "short read, bad digest, failed copy), from "
                                   "counters()",
            "fuses_on_unknown_kinds": "fuses raised with NO failure kind visible - the "
                                      "fail-closed branch, and the reason a quiet "
                                      "`races_not_fused` can mean 'could not tell' and "
                                      "not 'there were no races', from counters()",
            "race_reoffers_granted": "second offers granted by the one-per-request "
                                     "race allowance: a request is offered at most "
                                     "TWICE per boot, race or fault, from counters()",
            "last_fuse_decision": "the last failure report's decision as a sentence: "
                                  "the request, where the report arrived, the kinds "
                                  "visible, and whether it fused or spared the offer",
            "load_failures_here": "load-failure records written by THIS process, from "
                                  "counters()",
            "load_failures_from_ranks": "load-failure records read from live rank "
                                        "files, from counters()",
            "invalid_block_ids_seen": "GPU block ids the fork reported as load errors "
                                      "over the boot (the count `note_reported` used "
                                      "to take and discard), from counters()",
            "last_invalid_block_ids": "the ids themselves from the most recent report "
                                      "(bounded), so a scheduler-side burn can be "
                                      "matched to a worker-side record by block id",
            "partial_load": "what a PARTIAL load does to a request, as data and as a "
                            "sentence: applied reads, restorations, the span shape, "
                            "and the statement that bytes after the first hole are "
                            "recomputed by the engine and the request is fused",
            "loads_released": "loads removed from the ledger, from counters()",
            "authority_lookups": "lookups that ran under disk authority, so the disk "
                              "and not the host-RAM tier decided the span, from "
                              "counters(). Every one of these "
                              "was offered at `computed // block_size`, i.e. "
                              "the disk span starts where the LOCAL GPU prefix ends - "
                              "the local hit is the one thing the switch cannot "
                              "un-learn, because the scheduler placed those blocks "
                              "before it ever asked us. It stays 0 while "
                              "`disk_authority` is false (unit: lookups)",
            "local_hit_ignored": "offers committed under disk authority whose "
                                 "host-RAM hit was non-zero and was discarded; a "
                                 "refusal never moves this, because a refused offer "
                                 "leaves the fork's hit exactly where the fork put "
                                 "it (unit: offers)",
            "local_hit_tokens_ignored": "host-RAM tokens those offers gave up to "
                                        "let the disk serve the span (unit: tokens)",
            "pin_release_failures": "times the fork's pinned host-RAM blocks could "
                                    "not be released after a disk-authority offer: "
                                    "pinned CPU blocks that may outlive the request, "
                                    "so this must stay 0 (unit: failures)",
            "last_local_hit_ignored": "the last discarded host-RAM hit in one "
                                      "sentence (request id, the tokens the fork "
                                      "reported, the block the disk span starts at)",
            "last_offer": "THE LAST OFFER AS STRUCTURE: `disk_authority`, "
                          "`computed_tokens`, `ram_tokens_reported` (what the fork "
                          "found), `ram_tokens_used` (what the offer used), "
                          "`start_block`, `blocks`, `disk_tokens`, `groups`, and "
                          "`groups_required` - one row per prefix-cacheable group "
                          "saying what the claimed range costs THAT group and "
                          "whether the disk fills it. How a read forced from disk is "
                          "told apart from one the disk had to supply anyway, and how "
                          "a claim that only half-covers the model would have shown "
                          "itself",
            "group_serviceability": "THE GROUP ARITHMETIC OF THE LAST LOOKUP (0.2.25), "
                                    "from counters(): the prefix-cacheable set, the "
                                    "positionally stable set, the restorable set, "
                                    "`blocked_group_ids` with a `blocked_why` for "
                                    "each, and `complete`. `complete: false` means a "
                                    "prefix-cacheable group can NEVER be restored for "
                                    "any token on this model, so a disk offer would "
                                    "claim more than it delivers and the tier refuses "
                                    "to make one: the whole of "
                                    "`refused.incomplete_group_set`, readable without "
                                    "knowing what a MambaSpec is",
            "last_aged_out": "the last load the ledger released with no report "
                             "correlated, in one sentence (which door, adopted or "
                             "not, reads applied or not, GPU blocks mapped)",
            "last_groups_required": "THE REQUIREMENT ARITHMETIC OF THE LAST PRICED "
                                    "CLAIM, offered OR refused (0.2.25): "
                                    "`claimed_blocks`/`claimed_tokens`, `offered`, and "
                                    "one row per prefix-cacheable group - "
                                    "`blocks_needed`, `null_blocks`, the `contract` "
                                    "that arithmetic came from, and `restored`. A "
                                    "state group's row says `blocks_needed: 1` with "
                                    "`null_blocks: claimed-1` because "
                                    "`MambaManager.find_longest_cache_hit` answers a "
                                    "hit as nulls plus ONE boundary state block "
                                    "(`single_type_kv_cache_manager.py:1533-1610`) - "
                                    "the claim needs one block from it, not a span of "
                                    "KV. This is the object that shows a half-covering "
                                    "claim without anyone reading the source",
            "no_read": "finished reports with no applied disk read behind them, from "
                       "counters(): the counter that keeps a finished report from "
                       "being mistaken for a restoration, and the reason "
                       "`restorations: 0` next to a nonzero `adopted` is an answer "
                       "and not a mystery",
            "watchdog_fires": "scheduler-side ledger expiries, from counters(): the "
                              "deadline branch of the sweep. Read beside "
                              "`invalid_reports` and `no_read`, it tells an operator "
                              "WHICH report-less release happened instead of leaving "
                              "them to infer one from `loads_released`",
            "invalid_reports": "GPU block ids DRAINED as load errors by this "
                               "process, from counters(). The drain lives on the "
                               "WORKER side (`take_invalid_block_ids`, called by the "
                               "model runner), so a scheduler process reports 0 here "
                               "BY CONSTRUCTION and the live rank file's "
                               "`disk_restore.invalid_reports` is the number that "
                               "moved - published in this section so a reader sees the "
                               "zero and the explanation together, not a gap (unit: "
                               "block ids)",
            "aged_out_loads": "loads released by the ledger with no report "
                              "correlated (fork release, deadline, or cap eviction), "
                              "from counters(): a release describes the LEDGER, not "
                              "the bytes (unit: loads)",
            "aged_out_with_applied_reads": "of those, releases whose slots had "
                                           "reached a worker and a disk read had been "
                                           "applied: bytes were in flight when the "
                                           "ledger let go (unit: loads)",
            "late_finished_reports": "finished reports that arrived for a load the "
                                     "ledger had already released, MATCHED to that "
                                     "release instead of dropped, from counters(). "
                                     "Before 0.2.25 this branch moved no counter at "
                                     "all, which is how 36 applied reads presented as "
                                     "`restorations=0` and nothing else (unit: "
                                     "reports)",
            "restored_after_release": "late reports whose release showed an adopted "
                                      "load with bytes applied behind it: a "
                                      "restoration the ledger could not count in "
                                      "`restorations`, counted here (unit: loads)",
            "disk_gpu_blocks_dma_verified": "GPU BLOCKS this module mapped for a disk "
                                            "span and the fork then reported "
                                            "DMA-complete, from counters(): the only "
                                            "field coupling our claim to the fork's "
                                            "copy. An offer cannot move it; only an "
                                            "adopted load reported finished can "
                                            "(unit: GPU blocks)",
            "disk_gpu_blocks_released_unreported": "GPU blocks mapped for a disk span "
                                                   "and released with no report "
                                                   "either way - the complement of "
                                                   "`disk_gpu_blocks_dma_verified` "
                                                   "(unit: GPU blocks)",
            "lookups": "lookups reaching the restore path at all, from counters(): "
                       "the entry count of the offer path, counted BEFORE the gate, "
                       "so a `lookups` that rises while `offers` sits at 0 "
                       "is the refusal vocabulary's job to explain",
            "witness_verdict": "the last restore-posture verdict this process spoke "
                               "(a verdict, not a count). Read it BESIDE "
                               "`reads_applied`, never instead of it: on 2026-10-05 "
                               "the witness said CONFIRMED while reads_applied was 0",
            "witness_reason": "the sentence the witness last gave for its verdict: "
                              "prose for a human, never a number to aggregate",
            "witness_kind": "the stable name of the last verdict (the emission key is "
                            "(verdict, kind), never the sentence), from counters(). "
                            "AS OF THE LAST EMISSION and NOT as of now: it is the "
                            "name the logged line carried. Read `witness_kind_now` "
                            "beside it",
            "witness_verdict_now": "the verdict the same arithmetic gives on the "
                                   "counters AS THEY STAND (0.2.25): a payload read "
                                   "that writes nothing, so polling cannot rewrite "
                                   "the as-of-emission fields the log dedupes on",
            "witness_kind_now": "`witness_kind`'s question answered now: the kind "
                                "`_refuted_reason` names against the CURRENT "
                                "`offers`/`adopted`/`restorations`/`no_read`. On the "
                                "live box at 00:08 the as-of-emission kind said "
                                "`offers_not_adopted` while `adopted=1` - true when "
                                "spoken, false when read, and this field is what "
                                "makes the two impossible to confuse",
            "witness_reason_now": "the sentence behind `witness_verdict_now`, for "
                                  "this instant",
            "witness_kind_is_stale": "true when the as-of-emission kind no longer "
                                     "matches the kind the current counters would "
                                     "give: a flag saying 'read the `_now` pair, the "
                                     "frozen one is history' (a verdict that has since "
                                     "been re-emitted clears it by itself)",
            "deadline_seconds": "SECONDS an unreported load may live before the "
                                "watchdog reports it as a failure",
            "methods_here": "scheduler methods our layer is on in this process",
            "methods_expected": "scheduler methods it must be on to offer at all",
            "disk_authority": "this boot's DISK-AUTHORITY switch, read from "
                              "disk_posture(): FALSE at every fresh boot, raised only "
                              "by `POST /plugins/kvtier/disk {\"op\":\"authority\","
                              "\"enabled\":true}`, and turned back off by a restart. "
                              "True means EVERY request this boot is served from disk "
                              "with the disk deciding its own prefix - no client can "
                              "ask for it and no client can decline it.",
            "offers_enabled": "this boot's gap-fill switch, from disk_posture(): "
                              "false until the dev verb raises it, so an armed spill "
                              "tier alone cannot put a disk offer in front of live "
                              "traffic. Up on its own it means RAM first and the disk "
                              "extending the shortfall.",
            "serving_as": "the two switches as one verdict, from disk_posture(): "
                          "`off`, `disk_authority` or `ram_first_gap_fill`",
            "per_boot": "true, and it is a statement about the switches: both are "
                        "in-process per-boot state set only by the POST op, so "
                        "nothing here outlives the process",
            "survives_restart": "false: a restart - clean or after a crash - gives "
                                "disk authority to NOBODY, which is why an unhealthy "
                                "boot cannot come back serving from SSD",
            "disk_authority_changes": "times the disk-authority switch moved this "
                                      "boot, from counters(): a posture moved once "
                                      "per experiment (unit: changes)",
            "offer_switch_changes": "times the gap-fill switch moved this boot, from "
                                    "counters() (unit: changes)",
            # `offer_cap` IS RETIRED, with the fixed 32,000-token cap it advertised:
            # leaving a field of that name in the payload reporting a number that no
            # longer constrains anything would be the worst kind of stale field, so the
            # name is gone and the four fields below carry the bound that DOES bind.
            "offer_block_cap_blocks": "SCHEDULER BLOCKS: the per-offer bound that "
                                      "replaces the deleted fixed 32,000-token cap - "
                                      "MAX_RESTORE_BLOCKS_PER_OFFER, where the "
                                      "`_disk_run` positional walk stops, because that "
                                      "is how many spans one offer can name and stream "
                                      "inside RESTORE_DEADLINE_SECONDS",
            "offer_block_cap_tokens": "tokens those blocks are worth "
                                      "(offer_block_cap_blocks x the scheduler block "
                                      "size): a DERIVED figure, not a cap anyone "
                                      "configured, and far above what a real offer "
                                      "reaches - bounce-buffer affordability is the "
                                      "bound that binds, and on the live store the "
                                      "spill tier's depth (~11 spans) binds earlier",
            "offer_block_cap_block_size": "tokens in one scheduler block, the "
                                          "multiplier behind `offer_block_cap_tokens`, "
                                          "from the manager's own `block_size` once a "
                                          "lookup has run",
            "offer_block_cap_block_size_source": "`engine` (a lookup in this process "
                                                 "supplied it), `last-lookup` "
                                                 "(remembered from one), or `fallback` "
                                                 "(DEFAULT_OFFER_BLOCK_SIZE_TOKENS, in "
                                                 "a process that has made no offer yet "
                                                 "- a worker rank, typically)",
            "offer_block_capped": "offers the per-offer BLOCK cap bound, from "
                                  "counters(): the positional walk hit "
                                  "MAX_RESTORE_BLOCKS_PER_OFFER with tokens still left "
                                  "to restore. The only per-offer cap counter left now "
                                  "that `offer_clamped` is retired with the token cap; "
                                  "read it beside `offers_pool_capped` (the "
                                  "bounce-buffer pool), which is the bound that binds "
                                  "in practice",
            "ranks_armed": "live rank files that WILL SERVICE a disk load, from the "
                           "same worker_proof() read as `ranks_proven`: the proof is "
                           "AND over these, so one disarmed rank closes the whole "
                           "scheduler",
            "ranks_seen": "live rank files that spoke about disk service at all, "
                          "armed or not, from the same worker_proof() read",
            "suppressed_hits": "lookups refused because the REQUEST is suppressed "
                               "after a failed disk load (the loop that starved the "
                               "box), from counters()",
            "suppressed_spans": "offers refused because the disk SPAN was implicated "
                                "in a failed load, whichever request asked for it, "
                                "from counters()",
            "suppressed_requests": "request ids currently black-holed for disk "
                                   "service in this boot (the fuse table's live size, "
                                   "not a counter), from counters()",
            "suppressions": "requests black-holed for disk service at any point in "
                            "this boot (the lifetime total, from counters()); read "
                            "beside suppressed_requests, which is the table's live "
                            "size",
        },
        "note": ("THE SCHEDULER'S OWN ANSWER: the scheduler half, the gate, the "
                 "index and the worker proof, walked in that order, with "
                 "`why_no_restore` naming the first link that is missing. A rank "
                 "file's `disk_restore.installed` is a DIFFERENT process's verdict "
                 "and cannot start a restore; this one can."),
    }
