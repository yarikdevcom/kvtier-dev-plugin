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

THE MARKER DECIDES THE SPAN (0.2.22), AND THAT IS THE WHOLE READ PATH'S TESTABILITY.
A request carrying `kv_transfer_params.kvtier_disk_restore = true` has the fork's
host-RAM hit treated as ZERO tokens for the offer arithmetic, so the disk index - not
whatever RAM happened to hold - decides what is offered, starting at the local
GPU-computed boundary. Without that rule the disk could only ever fill the gap RAM
left, so a small prompt either hit RAM wholly (nothing to offer) or hit it with a
remainder (a refusal), and exercising one disk read cost ~1.3M tokens of eviction
filler. A request WITHOUT the marker is unaffected: RAM first, disk second, byte for
byte as before. `computed` is never forced to zero, because it is the offset the fork's
own copy writes at, not an opinion - see the RESTORE_MARKER_KEY docstring for the
exact corruption that forcing it would cause.

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
read is 64 x 45.9 MB = 2.94 GB = ~255 ms of streaming - and 45 s is a ~175x margin
on that. It is ALSO `disk_read.LOAD_DEADLINE_SECONDS` (30 s, the reader's own
promise) + 50%: the batch deadline must always fire first, so the watchdog only
ever sees the case the batch deadline could not reach (a batch that settled into
nothing, or a step that never reached `start_load_kv`). 45 s stays far below the
point at which an operator calls a parked request a hang, and a stall it fires on
would otherwise be permanent, because the fork has no timeout for
WAITING_FOR_REMOTE_KVS."""

MAX_RESTORE_BLOCKS_PER_OFFER = 64    # SCHEDULER BLOCKS offered per request
# THE OFFER'S REAL CEILING, IN TOKENS. 64 blocks of a 3,200-token scheduler block is
# 204,800 tokens, and the block/slot caps alone let one request ask for the whole
# prompt: on 2026-10-05 a live boot offered EXACTLY 160,000 tokens (50 blocks) on
# every one of 123,474 offers, the engine could not place them, and every one of
# them came back as a load failure that rescheduled the request. A cap that the
# request's own length can reach is not a cap. 32,000 tokens is 10 scheduler blocks,
# ~144 MB at the measured 14,416,000 B per 3,200-token span, ~13 ms of streaming
# against a 45 s deadline: enough to be worth doing, small enough to be placeable.
MAX_OFFER_TOKENS_PER_OFFER = 32_000  # TOKENS offered per request (10 spans)
MAX_SLOTS_PER_OFFER = 256            # CPU slots allocated per request
MAX_TRACKED_LOADS = 64               # REQUESTS with an outstanding disk load
MAX_TRACKED_SLOTS = 4096             # cpu-slot -> identity entries
MAX_INVALID_BLOCKS = 4096            # GPU block ids awaiting one report
MAX_REFUSAL_REASONS = 24             # distinct refusal reasons remembered
MAX_STRING_CHARS = 160
MAX_BOOT_LINE_CHARS = 160
CPU_SLOT_RESERVE = 64                # CPU slots the offer must leave free
MAX_PATHS_PER_STEP = 512             # load_paths entries one step may carry
MIN_WITNESS_LOG_SECONDS = 30.0       # SECONDS between witness re-emissions
MAX_SUPPRESSED_REQUESTS = 4096       # REQUESTS black-holed after a failed load
MAX_RANK_COUNTER = 1_000_000_000     # ceiling on a counter read from a rank file
MAX_SPANS_REPORT = 16                # PARTIAL SPANS one shape report may name

# --- the settle wait, and the arithmetic that sizes it -----------------------
#
# WHY A CAP AND NOT A BUDGET. The wait has to be proportional to the work in flight,
# because the two ends of the range are orders of magnitude apart. Measured on the live
# box: a 9,400-token prompt recomputes in 2.4 s through the proxy, and a 32,000-token
# offer is 10 spans x 5 groups x 11,468,800 B = ~573 MB per rank. So the recompute a
# wait is racing is ~2.5 s for a 3-span x 5-group offer (~9,600 tokens, ~172 MB) and
# ~8.2 s for a 10-span one (~32,000 tokens, ~573 MB), while the read that avoids it is
# ~15 ms and ~50 ms at the store's measured 11.5 GB/s. A flat budget sized for the big
# offer is ~3x what the small one needs; the only flat number that is ever obviously
# safe against recompute (eight seconds) is a net loss on every offer, paid inside the
# step by requests that never asked for disk. The rule the fields below encode is
# therefore: WAIT ONLY WHILE THE WAIT IS CHEAPER THAN THE RECOMPUTE, and publish the
# boundary rather than hide it.
#
# `SETTLE_MS_DEFAULT` is 0.0 for a reason that outlives this experiment: the plugin
# ships into a box serving real traffic, and a step that blocks on NVMe by accident
# is worse than a tier that never reads. Every number below is inert until an
# operator names a cap in the environment.
SETTLE_MS_DEFAULT = 0.0               # MILLISECONDS; 0 = never wait (today's boot)
SETTLE_MS_MAX = 60_000.0              # a cap that is not this is a typo, not a budget
SETTLE_MIN_MS = 0.5                   # FLOOR: a wait shorter than one scheduler
                                      # timer tick cannot be observed, so grant the
                                      # tick and let the poll loop leave early
SETTLE_SAFETY_MARGIN = 2.0            # the read is the FLOOR of the budget, not the
                                      # estimate: queue hand-off, thread wakeup and
                                      # the bounce-buffer copy are all on top of the
                                      # kernel's read time
SETTLE_READ_BPS_DEFAULT = 11_500_000_000.0   # BYTES/S assumed while nobody has
                                             # measured the READ side; see the env
                                             # var's comment in config.py
PREFILL_TOKENS_PER_S_DEFAULT = 9400.0 / 2.4  # TOKENS/S: the two measured samples,
                                             # 9,400 tokens recomputed in 2.4 s
SETTLE_POLL_SECONDS = 0.0005          # SECONDS between looks while waiting; half a
                                      # millisecond is below every interesting read
                                      # time and above a busy-wait's cost
MAX_SETTLE_SAMPLES = 512              # WINDOW samples kept for a median; min/max are
                                      # boot-wide regardless

# --- the request-scoped gate --------------------------------------------------

RESTORE_MARKER_KEY = "kvtier_disk_restore"
"""THE opt-in key, read out of `request.kv_transfer_params`.

The channel is the engine's own: `vllm/v1/request.py:103` declares
`self.kv_transfer_params: dict[str, Any] | None` and `:118-119` fills it from
`sampling_params.extra_args.get("kv_transfer_params")`, and
`get_num_new_matched_tokens(self, request, num_computed_tokens)` receives the
request object - so the opt-in is readable at exactly the decision point, with no
new socket, file, env var or side table. The fork's own connector reads nothing
from it (`grep -rn kv_transfer_params vllm/v1/simple_kv_offload/` is empty on the
box), so taking it over does not collide with fork behaviour.

STRICTNESS, spelled out because a client can send anything here: a request is
flagged when `kv_transfer_params` is a real `dict` AND its value under this key is
the boolean `True`. Not `"true"`, not `1`, not a nested object - a caller who means
it sends JSON `true`. Everything else is "not flagged", counted, never raised.

WHAT THE MARKER MEANS (0.2.22): THE DISK IS THE AUTHORITY FOR THAT REQUEST. It is
not a hint that a disk offer would be welcome. A flagged request's host-RAM hit is
treated as ZERO tokens for the offer arithmetic, so the disk index decides the span
from the local GPU-computed boundary rather than filling the gap the RAM tier left.
That is the only way the read path can be exercised at all: with RAM consulted, a
small prompt either hits RAM (nothing left to offer) or hits it partially, which the
old alignment rule refused outright - so proving the disk could serve ONE block meant
evicting ~1.3M tokens of filler first. An UNFLAGGED request is not touched by any of
this: it gets the fork's own answer, byte for byte, which is how the existing client
keeps running on the RAM tier with no client-side change.

WHY `computed` IS NOT ALSO FORCED TO ZERO. `update_state_after_alloc`
(`manager.py:441-463`) places the external blocks at
`num_computed_tokens // group_block_size + i` INTO THE REQUEST'S WHOLE BLOCK LIST
(the scheduler passes `kv_cache_manager.get_blocks(request_id)`,
`scheduler.py:1228-1232`). The third field of the pending tuple is therefore not an
opinion - it is the offset the DMA writes at. Zeroing it while the scheduler still
believes `computed` tokens are cached would aim the copy at the request's ALREADY
CACHED, ref-counted prefix blocks (shared with every other request reading them) and
leave the newly allocated external region unwritten: wrong KV, plus a write into
blocks this request does not own. The scheduler's local hit is not this module's to
un-learn, so the disk span starts where the local prefix ends. The RAM hit IS ours to
ignore, because the fork keeps it in exactly one place, `_pending_cpu_hits`, and
releases it through one method, `_free_pending_cpu_hit` (see `_release_fork_pin`).
"""

RESTORE_SCOPE_OPTIN = "optin"
RESTORE_SCOPE_ALL = "all"
RESTORE_SCOPES = (RESTORE_SCOPE_OPTIN, RESTORE_SCOPE_ALL)
RESTORE_SCOPE_DEFAULT = RESTORE_SCOPE_OPTIN

OFFER_REFUSALS = ("not_configured", "disarmed", "tier_disarmed", "bad_call",
                  "bad_result", "unaligned_local_prefix", "skip_read_requested",
                  "no_tokens_left", "no_block_hashes", "no_key_helpers",
                  "no_eligible_group", "ram_shape", "no_disk_run",
                  "disk_run_below_one_block", "no_cpu_slots", "alloc_refused",
                  "no_channel", "path_build_failed", "index_refresh_failed",
                  "too_many_loads", "ram_hit_not_block_aligned",
                  "no_worker_proof", "offers_disabled", "workers_not_servicing",
                  "skipped_no_marker", "suppressed_after_failure",
                  "pin_release_unavailable", "incomplete_group_set")

# EVERY REFUSAL SPELLED OUT. `refused` says WHICH counter moved; this says what the
# counter MEANS, so "why is there no restore?" is answerable from one payload read
# instead of from the source. A name with no entry here is a bug in this table, and
# `scheduler_section()` publishes the gap as `<name>: "undefined reason"` rather
# than hiding it.
OFFER_REFUSAL_WHY = {
    "not_configured": "no store to read: VLLM_E2_TIER_DISK_ROOT is unset in this "
                      "process, so there is no directory a file could come from",
    "disarmed": "the spill tier is disarmed (its arm file is absent or expired), "
                "and restore is deliberately not separately armable",
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
    "no_eligible_group": "no prefix-cacheable, positionally-stable group has "
                         "block_size == the scheduler block size, so a merged "
                         "tuple cannot be extended one whole block at a time",
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
    "offers_disabled": "the operator has not raised this boot's offer switch "
                       "(`POST /plugins/kvtier/disk {\"op\":\"enable\"}`): the "
                       "restore path stays out of the scheduling decision until "
                       "someone chooses that in this boot, because arming the "
                       "spill tier for a different experiment once put an "
                       "un-requested disk offer in front of live traffic",
    "workers_not_servicing": "the scheduler is armed but at least one live rank "
                             "file says its worker half will NOT service a disk "
                             "load (disarmed in its own process, or a boot the "
                             "arm file outlived): offering would allocate slots "
                             "nobody fills, the fork would report the load failed "
                             "and the request would be rescheduled - the exact "
                             "loop that starved the box on 2026-10-05",
    "skipped_no_marker": "scope is `optin` and this request did not ask for a "
                         f"disk restore (`extra_args.kv_transfer_params."
                         f"{RESTORE_MARKER_KEY} = true` in the request body): the "
                         "fork's own answer is returned untouched, which is the "
                         "default posture for every request that does not ask: "
                         "no marker means the host-RAM tier answers it exactly as "
                         "it did before this module existed",
    "pin_release_unavailable": "the request asked to be served from disk and the "
                               "fork had already PINNED host-RAM blocks for its "
                               "own hit, but this build's manager exposes no "
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
    "incomplete_group_set": "THE CLAIM-MUST-MATCH-THE-BYTES GATE (0.2.25). At "
                            "least one group CAN be restored, but another "
                            "prefix-cacheable group CANNOT, and offering the "
                            "restorable subset claims every group: "
                            "get_num_new_matched_tokens advances "
                            "num_computed_tokens for the WHOLE model, "
                            "update_state_after_alloc's `n_ext_g == 0 -> continue` "
                            "is a SILENT no-op for the group we did not put in the "
                            "tuple, and nothing recomputes the claimed range "
                            "because a group that was never in the tuple reports "
                            "no failure. The usual cause is a group that is "
                            "prefix-cacheable but not positionally stable: an "
                            "align-mode Mamba/GDN group's own hit contract is "
                            "[null_block]*(n-1) plus ONE real boundary-state block "
                            "(`single_type_kv_cache_manager.py:1533-1610`), so a "
                            "claimed span costs it ONE block - not a span of KV - "
                            "and that one block has no positional file name to "
                            "read back, because the recurrent state is one runn"
                            "ing value per sequence. Refusing costs one recompute; "
                            "offering serves a wrong answer, which is what the "
                            "live box did on 2026-10-06 at 00:08: a mechanically "
                            "perfect read (9 of 9 applied per rank, no failures, "
                            "18 ms) that answered garbage because the GDN "
                            "boundary state was claimed and never filled",
}
WORKER_FAILURES = ("path_count_mismatch", "bad_path", "identity_disagreement",
                   "digest_disagreement", "fingerprint_incomplete", "no_channel",
                   "read_failed", "file_missing", "size_mismatch", "short_read",
                   "read_error", "not_ours", "not_settled", "slot_plan_failed",
                   "copy_failed", "batch_cap", "queue_full", "buffer_cap",
                   "bad_batch", "bad_job", "bad_size", "incomplete",
                   "deadline_expired", "thread_dead", "not_tracked", "abandoned")

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
    # THE OFFER POSTURE, per boot, dev-verb only. `offers_enabled` is False at every
    # boot: arming the SPILL tier must not be enough to put a disk offer in front of
    # live traffic (on 2026-10-05 an arm raised for a different experiment offered
    # 19.7 BILLION tokens and starved the harness). `scope` then decides which
    # requests may be offered at all once the switch is up.
    "offers_enabled": False, "scope": RESTORE_SCOPE_DEFAULT,
    "scope_changes": 0, "offer_switch_changes": 0, "ranks_seen": 0,
    # scheduler-side request ledger (the request-scoped gate)
    "requests_seen": 0, "flagged": 0, "skipped_no_marker": 0,
    "suppressed_hits": 0, "suppressed_spans": 0, "suppressions": 0,
    # THE MARKER'S AUTHORITY (0.2.22). A flagged request hands the DISK the decision
    # about its own prefix: the fork's host-RAM hit is treated as zero tokens for the
    # offer arithmetic. `disk_authority` counts the lookups that took that rule; the
    # other two count the ones where a real RAM hit was thrown away and how many
    # tokens it was, so "was this read forced from disk, or did RAM have nothing?" is
    # one payload read and not an inference.
    "disk_authority": 0, "local_hit_ignored": 0, "local_hit_tokens_ignored": 0,
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
    "offer_clamped": 0,
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
    "settle_bound_min": 0,            # which bound decided the budget, per the
    "settle_bound_proportional": 0,   #   three-way rule in `settle_budget`
    "settle_bound_cap": 0,
    "settle_bound_deadline": 0,
    "settle_budget_over_recompute": 0,  # batches whose budget exceeded the recompute
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


def reset_state_for_tests() -> None:
    """Back to a fresh boot: no ledgers, no counters, no installed layer."""
    global _loads, _slot_paths, _wloads, _wbatches, _invalid, _logged_lines
    global _install_refusal_lines, _sched_cls_seen, _suppressed, _suppressed_spans
    global _offered_at, _burned, _aged_out
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
    _logged_lines = []
    _install_refusal_lines = []
    _sched_cls_seen = None
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
            "disk_reads_applied": 0, "ranks_reads_applied": 0, "offer_clamped": 0,
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
            "settle_bound_min": 0, "settle_bound_proportional": 0,
            "settle_bound_cap": 0, "settle_bound_deadline": 0,
            "settle_budget_over_recompute": 0, "last_apply_window": None,
            "settle_notice": None,
            "spans_observed": 0, "span_entries_observed": 0, "spans_whole": 0,
            "spans_partial": 0, "spans_empty": 0, "spans_hole_in_the_middle": 0,
            "last_span_shape": None,
            "witness_verdict": None, "witness_reason": None,
            "witness_kind": None, "witness_emissions": 0,
            "witness_suppressed": 0,
            "offers_enabled": False, "scope": RESTORE_SCOPE_DEFAULT,
            "scope_changes": 0, "offer_switch_changes": 0, "ranks_seen": 0,
            "requests_seen": 0,
            "flagged": 0, "skipped_no_marker": 0, "suppressed_hits": 0,
            "suppressed_spans": 0, "suppressions": 0,
            "disk_authority": 0, "local_hit_ignored": 0,
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


def settle_budget(bytes_at_stake: int, offered_tokens: int | None = None,
                  seconds_left: float | None = None) -> dict:
    """How long ONE apply point may wait, and what that wait is racing.

    THE FORMULA, as it reads in code:

        proportional_ms = bytes_at_stake / read_bytes_per_s * 1000 * SAFETY_MARGIN
        budget_ms       = clamp(proportional_ms, SETTLE_MIN_MS, cap_ms)

    then clamped again by the batch's own deadline (`seconds_left`, which
    `disk_read.begin_batch` set from `LOAD_DEADLINE_SECONDS`) - a wait that outlives
    the deadline it is waiting under is not a wait, it is a second bug.

    `bytes_at_stake` is the bytes STILL IN FLIGHT at the apply point, not the size of
    the offer: a batch that has delivered nine of ten slots is waiting for one slot, and
    sizing its budget on ten would make the budget a function of the offer instead of a
    function of the wait.

    WHY PROPORTIONAL. The read is the floor of the budget and bytes are the only
    quantity that scales with it: the measured offer is 11,468,800 B per (span,
    group) per rank, so a 3-span x 5-group offer is ~172 MB and a 10-span offer is
    ~573 MB, while the recompute they avoid is ~2.5 s (9,600 tokens) and ~8.2 s
    (32,000 tokens) at the measured 3,917 tokens/s. A flat budget sized for the 573 MB
    case is about three times what a 3-span offer needs, and the only FLAT number that
    is ever obviously safe against recompute - eight seconds - is a net loss on every
    offer, paid inside the step by requests that never asked for disk. Bytes-per-second
    is a guess until this release measures it (the 11.5 GB/s figure is the WRITE rate);
    `SAFETY_MARGIN` covers the parts the estimate omits - queue hand-off, thread wakeup,
    the bounce-buffer copy - and the env var replaces the guess with a measurement once
    there is one.

    WHICH BOUND APPLIED is published, because the three answers mean three
    different things to the operator: `off` = the cap is 0, this boot never waits (the
    shipped default); `min` = a batch too small to measure, the floor is what it got;
    `proportional` = the arithmetic decided, which is the intended regime; `cap` = the
    operator's ceiling bound it, so a `dropped_after_wait` here is a cap that is too
    small; `deadline` = the read's own expiry decided, i.e. the wait was told to stop by
    the same clock that cancels the read.

    The recompute side is published as `recompute_ms` with `budget_over_recompute`
    set when the granted wait is MORE than the recompute it avoids - a wait there is
    a regression by construction even when it wins the race, and batches like that are
    counted (`settle_budget_over_recompute`) instead of being left for a human to
    notice at 3 a.m.
    """
    env = config.settle_env()
    raw_cap = env["cap_ms"] if env["cap_ms"] is not None else SETTLE_MS_DEFAULT
    cap_ms = max(0.0, min(float(raw_cap), SETTLE_MS_MAX))
    read_bps = (env["read_bytes_per_s"] if env["read_bytes_per_s"]
                else SETTLE_READ_BPS_DEFAULT)
    prefill_tps = (env["prefill_tokens_per_s"] if env["prefill_tokens_per_s"]
                   else PREFILL_TOKENS_PER_S_DEFAULT)
    payload = max(0, int(bytes_at_stake or 0))
    proportional_ms = (payload / read_bps) * 1000.0 * SETTLE_SAFETY_MARGIN
    if cap_ms <= 0.0:
        granted, bound = 0.0, "off"
    elif proportional_ms >= cap_ms:
        granted, bound = cap_ms, "cap"
    elif proportional_ms <= SETTLE_MIN_MS:
        granted, bound = SETTLE_MIN_MS, "min"
    else:
        granted, bound = proportional_ms, "proportional"
    deadline_ms = None
    if seconds_left is not None:
        try:
            deadline_ms = max(0.0, float(seconds_left)) * 1000.0
        except (TypeError, ValueError):
            deadline_ms = None
        if deadline_ms < granted:
            granted, bound = deadline_ms, "deadline"
    recompute_ms = None
    if offered_tokens:
        recompute_ms = (int(offered_tokens) / prefill_tps) * 1000.0
    return {"enabled": cap_ms > 0.0, "cap_ms": round(cap_ms, 3),
            # Named, not swallowed: an absurd cap is clamped and the reader is told a
            # ceiling did it, so a typo in a systemd drop-in is a visible clamp and not
            # a mystery wait at 3 a.m.
            "ceiling_clamped": float(raw_cap) > SETTLE_MS_MAX,
            "bytes_at_stake": payload, "read_bytes_per_s": read_bps,
            "safety_margin": SETTLE_SAFETY_MARGIN, "min_ms": SETTLE_MIN_MS,
            "proportional_ms": round(proportional_ms, 3),
            "budget_ms": round(granted, 3), "bound": bound,
            "deadline_ms": None if deadline_ms is None else round(deadline_ms, 3),
            "offered_tokens_estimate": None if not offered_tokens
                                      else int(offered_tokens),
            "prefill_tokens_per_s": round(prefill_tps, 1),
            "recompute_ms": None if recompute_ms is None else round(recompute_ms, 1),
            "budget_over_recompute": (None if recompute_ms is None or granted <= 0
                                      else granted > recompute_ms),
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
    if bound in ("min", "proportional", "cap", "deadline"):
        _state[f"settle_bound_{bound}"] += 1
    if window.get("budget_over_recompute"):
        _state["settle_budget_over_recompute"] += 1
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
                    ("enabled", "cap_ms", "ceiling_clamped", "min_ms",
                     "safety_margin", "read_bytes_per_s", "prefill_tokens_per_s",
                     "env_problems")},
        "waits_granted": granted,
        "settled_after_wait": state["settled_after_wait"],
        "dropped_after_wait": state["dropped_after_wait"],
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
        "bound": {"min": state["settle_bound_min"],
                  "proportional": state["settle_bound_proportional"],
                  "cap": state["settle_bound_cap"],
                  "deadline": state["settle_bound_deadline"]},
        "budget_over_recompute": state["settle_budget_over_recompute"],
        "last": state["last_apply_window"],
        "notice": state["settle_notice"]}
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
        "offers_enabled": "this boot's offer switch: False until the dev verb "
                          "raises it, so an armed spill tier alone cannot put a "
                          "disk offer in front of live traffic",
        "scope": "which requests may be offered at all: `optin` (only a request "
                 "carrying the kvtier_disk_restore marker) or `all`",
        "scope_changes": "times the scope was changed in this boot: a knob that "
                         "moves once per experiment, so anything above 1-2 in a "
                         "production log means someone was deciding mid-flight",
        "offer_switch_changes": "times this boot's offer switch was raised or "
                                "dropped: a boot that never offered and never "
                                "changed hands says OFF and 0",
        "requests_seen": "lookups observed by the request-scoped gate",
        "flagged": "requests carrying the kvtier_disk_restore marker",
        "skipped_no_marker": "requests left exactly as the fork would have left "
                             "them because they did not ask for a disk restore",
        "disk_authority": "flagged lookups where the DISK was made the authority "
                          "for the request's own prefix, so the host-RAM hit did "
                          "not decide the span (0.2.22: the marker's meaning)",
        "local_hit_ignored": "OFFERS committed on a flagged request whose host-RAM "
                             "hit was non-zero and was discarded to let the disk "
                             "span start at the local prefix instead. A refusal "
                             "counts here NEVER: a refused offer leaves the RAM hit "
                             "exactly where the fork put it",
        "local_hit_tokens_ignored": "tokens of host-RAM hit discarded by those "
                                    "offers: what the request gave up to exercise "
                                    "the disk path (unit: tokens)",
        "pin_release_failures": "times the fork's pinned host-RAM blocks could not "
                                "be released when a flagged request chose disk over "
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
        "offer_clamped": "offers shortened to MAX_OFFER_TOKENS_PER_OFFER: an offer "
                         "that would have been bigger than one request may put at "
                         "once (the 160,000-token offers of 2026-10-05 clamp here)",
        "suppressions": "requests black-holed for disk service in this boot",
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
        "settle_waits_granted": "batches given a non-zero settle budget",
        "settled_after_wait": "batches that settled during their wait: the waits that "
                              "bought bytes",
        "dropped_after_wait": "batches still unsettled when their wait expired: the "
                              "waits that were spent for nothing",
        "settle_bytes_waited": "bytes a settle wait was waiting for, summed over "
                               "batches (the quantity the budget is proportional to)",
        "settle_bound_min": "budgets decided by the floor (`SETTLE_MIN_MS`): batches "
                            "too small for the arithmetic to say anything",
        "settle_bound_proportional": "budgets decided by bytes-over-throughput: the "
                                     "intended regime",
        "settle_bound_cap": "budgets clamped by the operator's cap: an unsettled batch "
                            "counted here is a cap that is too small",
        "settle_bound_deadline": "budgets clamped by the read's own deadline: the wait "
                                 "was stopped by the clock that cancels the read",
        "settle_budget_over_recompute": "batches granted a wait LONGER than the "
                                        "recompute it avoids: a loss by construction "
                                        "even when the read lands",
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
    return _fail(reason, reason_kind=name)


# --- the gate -----------------------------------------------------------------

def offer_posture() -> dict:
    """This boot's offer posture: the switch and the scope. Read-only, never raises.

    Two independent facts, both per-boot, both set only by the dev verb:

    `enabled` - may this boot offer disk restores AT ALL. False at every fresh boot.
      Arming the spill tier is not this: on 2026-10-05 an arm raised for a spill
      measurement was enough to put an un-requested 160,000-token disk offer in front
      of live traffic 123,474 times, because the restore path keyed off the spill
      arm. It is its own decision now.

    `scope` - WHICH requests, once the switch is up. `optin` (the default) offers to
      a request only if it carries `kv_transfer_params.kvtier_disk_restore = true`;
      `all` offers to every request, which is what a measurement of the whole
      workload wants and what no production boot should be left in.
    """
    return {"enabled": bool(_state["offers_enabled"]),
            "scope": _state["scope"],
            "marker_key": RESTORE_MARKER_KEY,
            "flagged": _state["flagged"], "requests_seen": _state["requests_seen"],
            "skipped_no_marker": _state["skipped_no_marker"],
            "suppressed_requests": len(_suppressed)}


def set_offer_switch(enabled) -> dict:
    """Raise or drop THIS BOOT's offer switch. The dev verb's doing, never a file.

    Deliberately not an environment variable and not the spill arm: the whole point
    is that a boot that nobody asked to serve disk restores does not serve them.
    Accepts a real boolean only - a payload that says `"true"` or `1` is refused
    rather than obeyed, because a typo in a curl should not change engine behaviour.
    """
    if not isinstance(enabled, bool):
        return _fail(f"the offer switch wants a JSON true or false, got "
                     f"{type(enabled).__name__}")
    changed = bool(_state["offers_enabled"]) != enabled
    _state["offers_enabled"] = enabled
    if changed:
        _state["offer_switch_changes"] += 1
    _logger.warning("kvtier-dev-plugin: DISK[restore] offers %s for this boot "
                   "(scope=%s) - a fresh boot starts with them OFF and only this "
                   "verb changes it", "ENABLED" if enabled else "disabled",
                   _state["scope"])
    return _ok(posture=offer_posture(), changed=changed)


def set_scope(scope) -> dict:
    """Which requests may be offered at all: `optin` (default) or `all`.

    A misspelled scope is refused, not defaulted: silently falling back to `optin`
    when someone typed `"all"` with a typo would leave a measurement measuring
    nothing, and the operator would only see it in the results.
    """
    if not isinstance(scope, str) or scope not in RESTORE_SCOPES:
        return _fail(f"scope must be one of {list(RESTORE_SCOPES)}, got {scope!r}")
    changed = _state["scope"] != scope
    _state["scope"] = scope
    if changed:
        _state["scope_changes"] += 1
    _logger.warning("kvtier-dev-plugin: DISK[restore] scope=%s - %s", scope,
                   "only a request carrying "
                   f"kv_transfer_params.{RESTORE_MARKER_KEY}=true is offered" if
                   scope == RESTORE_SCOPE_OPTIN else
                   "EVERY request is a candidate, which is a measurement posture")
    return _ok(posture=offer_posture(), changed=changed)


def request_flagged(request) -> bool:
    """Did THIS request ask for a disk restore? One attribute read; never raises.

    Strict on purpose: a real `dict` carrying the marker key with the boolean `True`.
    `"true"`, `1`, `1.0`, a nested object and a `kv_transfer_params` that is not a
    dict are all "not asked". A client controls this field, and the cost of being
    loose about it is disk work on a request that never asked for it.

    This is the first thing the offer path looks at, before the gate and before any
    file or index access, so an unflagged request in `optin` scope costs the fork's
    own lookup plus this one attribute read. `tests` pin that with a lookup spy.
    """
    try:
        params = getattr(request, "kv_transfer_params", None)
        return isinstance(params, dict) and params.get(RESTORE_MARKER_KEY) is True
    except Exception as exc:  # noqa: BLE001 - a hostile request is not a reason to raise
        snapshot._error(exc, "disk_restore request marker")
        return False


def gate() -> dict:
    """May THIS process make/serve a disk restore? Configured AND armed AND armed.

    Three switches, no new one: the root (`VLLM_E2_TIER_DISK_ROOT`, which
    `disk_spill` owns and validates), the spill tier's arm file (which expires), and
    the KV tier's own arm. A fresh boot answers `disarmed` and the wrappers cost an
    env read. Reading the arm file is `disk_spill`'s cached verdict, so this is not
    a per-step `open()`.
    """
    root = disk_spill.disk_root()
    if root is None:
        return _fail(f"{config.ENV_DISK_ROOT} is not set: there is no store a "
                     "restored file could be read from",
                     reason_kind="not_configured")
    if not disk_spill.is_armed():
        return _fail("the disk tier is disarmed: the restore path obeys the spill "
                     "tier's arm, which expires on its own",
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
    if not _state["offers_enabled"]:
        # The boot default. Reaching here means the spill tier is armed AND the RAM
        # tier is armed, which is a state a *different* experiment puts up - and on
        # 2026-10-05 that state was enough to put an un-requested 160,000-token disk
        # offer in front of live traffic 123,474 times. Offering is its own decision,
        # made per boot with the dev verb, and nothing else can make it.
        return _fail("this boot has not been told to offer disk restores: arm it "
                     "with POST /plugins/kvtier/d {\"op\":\"offers\","
                     "\"enabled\":true} when a measurement actually wants them",
                     reason_kind="offers_disabled")
    proof = worker_proof()
    if not proof["ok"]:
        kind = proof.get("reason_kind")
        if kind == "scheduler_armed_workers_disarmed":
            return _fail(proof["reason"],
                         reason_kind="scheduler_armed_workers_disarmed",
                         ranks_armed=f"{proof.get('ranks_armed', 0)}/"
                                     f"{proof.get('ranks_seen', 0)}")
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

    The proof is the rank file the worker ALREADY writes: `worker_diag` publishes this
    module's `disk_restore` section in `rank-*.json`, so a file that is FRESH and says
    `installed: YES` is a worker attesting that its layer is live. It adds no new
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


def _rank_file_proves_restore(path) -> dict | None:
    """What one rank file says about ITS worker: a verdict dict, or None.

    None means "this file is not evidence of anything" (unreadable, too big, no
    stamp, stale, wrong shape) - it neither proves nor vetoes. A readable file always
    yields a verdict with `installed`/`armed`, because "fitted but will not serve" is
    a DIFFERENT and much more important answer than "nothing here": a rank that is
    fitted and disarmed passes every disk entry through as though it had none, and
    the blocks the fork already committed to stay stale.

    Absent or non-boolean `armed` reads as NO. A file from a plugin that predates the
    field cannot vouch for a fact it never measured - fail closed, which is also what
    makes a mixed-version fleet safe.
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
        # `reads_applied` is a worker's own counter, published for the honesty check.
        # It arrives from another process, so it is clamped before it is summed: a
        # hostile or broken writer must not be able to put a fake total in front of the
        # witness. `disk_store` has a `_bounded_int` with a floor of 1 (it validates
        # operator knobs); this one wants 0-as-valid, so it uses this module's `_int`.
        reads = _int(section.get("reads_applied"))
        if reads is None or reads < 0:
            reads = 0
        return {"rank": name, "installed": installed, "armed": installed and armed,
                "reads_applied": min(reads, MAX_RANK_COUNTER),
                "reason": _clip(section.get("reason"), 160)}
    except Exception as exc:  # noqa: BLE001 - one bad file is not evidence, that's all
        snapshot._error(exc, "disk_restore rank file")
        return None


def _read_worker_proof() -> dict:
    """Read the rank files NOW (no cache). See `worker_proof`.

    The verdict is AND over the ranks that are live. A rank that is fitted but says it
    will not serve disk VETOES the offer even when other ranks are ready: the fork's
    load event spans every rank, so the ready ones cannot make the refusing one's
    blocks valid - that is precisely the shape of the 2026-10-05 incident (4 ranks,
    0 of them servicing, 123,474 offers).
    """
    directory = config.diagnostics_dir()
    if directory is None:
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": 0, "reads_applied": 0,
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
                "ranks_seen": 0, "reads_applied": 0,
                "reason_kind": "no_worker_proof",
                "reason": "the diagnostics directory could not be listed: "
                          f"{type(exc).__name__}"}
    if not names:
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": 0, "reads_applied": 0,
                "reason_kind": "no_worker_proof",
                "reason": "no worker has written a rank file yet, so nothing proves "
                          "the worker half of this path is fitted"}
    armed, silent = [], []
    reads = 0
    probed = names[:MAX_RANK_FILES_PROBED]
    for name in probed:
        said = _rank_file_proves_restore(directory / name)
        if said is None:
            continue
        if not said["installed"]:
            silent.append(f"{said['rank']}: not fitted")
            continue
        if not said["armed"]:
            silent.append(f"{said['rank']}: {said['reason'] or 'will not serve'}")
            continue
        reads += said["reads_applied"] or 0
        armed.append(said["rank"])
    _state["proof_reads"] += len(probed)
    armed.sort()
    seen = len(armed) + len(silent)
    _state["ranks_seen"] = seen
    _state["ranks_reads_applied"] = reads
    if armed and not silent:
        return {"ok": True, "count": len(armed), "ranks": armed[:8],
                "ranks_armed": len(armed), "ranks_seen": seen,
                "reads_applied": reads, "reason_kind": None, "reason": None}
    if silent and armed:
        return {"ok": False, "count": len(armed), "ranks": armed[:8],
                "ranks_armed": len(armed), "ranks_seen": seen,
                "reads_applied": reads,
                "reason_kind": "scheduler_armed_workers_disarmed",
                "reason": f"{len(armed)} of {seen} live rank file(s) serve disk and "
                          f"{len(silent)} do not ({'; '.join(silent)[:160]}): the "
                          "load event spans every rank, so the ones that serve would "
                          "still leave blocks the fork committed to "
                          f"(ranks_armed={len(armed)}/{seen})"}
    if silent:
        return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
                "ranks_seen": seen, "reads_applied": reads,
                "reason_kind": "scheduler_armed_workers_disarmed",
                "reason": f"the scheduler is armed and {len(silent)} live rank "
                          f"file(s) refuse disk service "
                          f"({'; '.join(silent)[:160]}): an offer would allocate CPU "
                          "slots nothing fills, the fork would report the load failed "
                          "and reschedule the request - ranks_armed="
                          f"0/{seen}, the exact shape of the 2026-10-05 loop"}
    return {"ok": False, "count": 0, "ranks": [], "ranks_armed": 0,
            "ranks_seen": 0, "reads_applied": reads,
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
    """THE WHOLE GROUP ARITHMETIC, as data, for one payload read (0.2.25).

    Three predicates decide whether a model can be served from disk at all -
    `prefix_cacheable_group_ids`, `group_block_sizes` against the scheduler block
    size, and `has_positionally_stable_blocks` per group - and until now a reader had
    to re-derive all three from a live manager to answer "why does the disk offer
    nothing?". This publishes the derivation instead: which groups the RAM tier says
    are prefix-cacheable, which groups are positionally stable, which are restorable,
    and WHICH BLOCKED THE OFFER AND WHY.

    Every group in the table gets a stability read, not just the cacheable ones, so
    the payload shows the shape of the model (a stable-but-scratch group is exactly
    the fact that distinguishes "nothing is stable" from "the stable ones are not
    cacheable"). Each predicate is read ONCE per group per call: an exception inside
    `_group_positionally_stable` is an error line, and reading the same flag twice
    would report a failure that happened once as two.

    `complete` is the property the gate is built on: every prefix-cacheable group is
    restorable. A pure-attention model satisfies it and is offered at exactly as
    before; a hybrid GDN + full-attention model does not, and cannot be, because a
    recurrent state has no per-block name.
    """
    stable: list = []
    for group in range(len(sizes)):
        if _group_positionally_stable(manager, group):
            stable.append(group)
    restorable: list = []
    blocked: dict = {}
    kinds: dict = {}
    for group in wanted:
        if group < 0 or group >= len(sizes):
            blocked[group] = "outside the group table"
            kinds[group] = "outside"
        elif _int(sizes[group]) != block_size:
            blocked[group] = (f"block_size {sizes[group]} != the scheduler "
                              f"block_size {block_size}, so a slot for it would "
                              "land at the wrong GPU offset")
            kinds[group] = "geometry"
        elif group not in stable:
            blocked[group] = ("has no positionally stable blocks, so a claimed "
                              "span costs it ONE boundary-state block and that "
                              "block has no per-block file name to read back (an "
                              "align-mode Mamba/GDN group keeps ONE running state "
                              "per sequence, not a per-block array, and no token "
                              "hash can address it)")
            kinds[group] = "unstable"
        else:
            restorable.append(group)
    unstable = [group for group in wanted if group not in stable]
    return _ok(prefix_cacheable_group_ids=list(wanted)[:16],
               positionally_stable_group_ids=stable[:16],
               unrestorable_cacheable_group_ids=unstable[:16],
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
    kinds = serviceability.get("blocked_kinds") or {}
    rows: list = []
    unfilled: list = []
    for group in (serviceability.get("prefix_cacheable_group_ids") or [])[:16]:
        group_size = (sizes[group]
                      if 0 <= group < len(sizes) else None)
        kind = ("positional" if group in restorable
                else str(kinds.get(str(group)) or "unknown"))
        needed: int | None
        nulls: int | None
        if kind == "positional" and group_size:
            needed = max(1, (blocks * block_size) // int(group_size))
            nulls = 0
        elif kind == "geometry" and group_size:
            # Position-addressable in principle, at a block size this offer cannot
            # express: it still needs its blocks per the same arithmetic, and the
            # disk fills none of them.
            needed = max(1, (blocks * block_size) // int(group_size))
            nulls = 0
        elif kind == "unstable":
            needed = 1
            nulls = max(0, blocks - 1)
        else:
            needed = None
            nulls = None
        filled = needed if (group in restorable and needed) else 0
        if not filled:
            unfilled.append(group)
        rows.append({"group": int(group), "block_size": group_size,
                     "contract": _clip(kind, 24),
                     "blocks_needed": needed, "null_blocks": nulls,
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
    if not groups:
        return _fail(f"no prefix-cacheable group has block_size == the scheduler "
                     f"block_size {block_size} (group sizes are {sizes[:8]}, "
                     f"cacheable {wanted[:8]}): the merged CPU-hit tuple cannot be "
                     "extended one whole block at a time for this geometry",
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
        if "unstable" in kinds:
            # KEPT SHORT ON PURPOSE: the whole sentence is clipped at 160 characters
            # and this clause carries the mechanism, so it has to fit before the cut.
            # The long form lives in `blocked_why` and in `OFFER_REFUSAL_WHY`.
            why_bits.append("no positional name: each span costs ONE boundary-state "
                            "block and the disk has none")
        if "geometry" in kinds:
            why_bits.append("its block_size is not the scheduler's")
        if "outside" in kinds:
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
               group_block_sizes=[int(sizes[group]) for group in groups])


def _group_positionally_stable(manager, group: int) -> bool:
    """Can this group's blocks be addressed by a POSITIONAL hash at all?

    The fork gates its own spill path on exactly this flag
    (`manager.py:874-876` on the deployed in-tree-fixed build, `manager.py:835-836`
    at the fork HEAD 10b8e07 this suite's fixture is generated from:
    `single_type_managers[g].has_positionally_stable_blocks` -> `continue`), and
    the reason carries over untouched: a spilled file is named after a hash derived
    from the request's tokens at a fixed position, so a group
    whose block content is not a function of position (a sliding window that
    rewrites its own blocks, for instance) has no stable name to restore from.
    Reading such a group back would paste bytes computed for different positions
    into the KV, which no recompute policy can undo. Unreadable counts as unstable:
    a restore we cannot justify is a restore we do not do.

    THE CONCLUSION THE OLDER CODE DREW FROM THIS, WRITTEN DOWN SO IT CANNOT BE DRAWN
    AGAIN (0.2.25). This flag was used as a FILTER - "offer on the groups that pass"
    - and the fork's own skip at `manager.py:874-876` looked like company for that.
    It is not: the fork is deciding what it may SPILL, where dropping a group costs a
    file that was never written. Here the same flag decides what may be CLAIMED, and a
    claim is not per group - `get_num_new_matched_tokens` returning a hit length makes
    the engine skip prefill for EVERY group at once. So a stable group is a necessary
    condition (a name to read by) and never a sufficient one, and a model with any
    prefix-cacheable group this function answers False for has no complete disk offer
    available at all. That is what `_eligible_groups`'s `incomplete_group_set` veto
    enforces; on the live hybrid box (`MambaManager` in `align` mode for groups 0-3)
    it means the disk tier offers nothing, which is the truth about the architecture
    and not a bug to route around.
    """
    try:
        coordinator = getattr(manager, "cpu_coordinator", None)
        single = getattr(coordinator, "single_type_managers", None)
        entry = single[group] if single is not None else None
        flag = getattr(entry, "has_positionally_stable_blocks", None)
        return flag is True
    except Exception as exc:  # noqa: BLE001 - an unverifiable group is not eligible
        snapshot._error(exc, "disk_restore has_positionally_stable_blocks")
        return False


def _disk_run(request, start_block: int, groups: list, block_size: int,
              hash_block_size: int, helpers: dict) -> dict:
    """The contiguous span of scheduler blocks the SSD tier can supply.

    A block is restorable only when EVERY eligible group has a file for it: the
    fork pairs per group, so one missing group's slot would be a load entry with no
    bytes behind it - a whole-request recompute for a partially-restored span. The
    run therefore stops at the first block any group cannot name.
    """
    block_hashes = getattr(request, "block_hashes", None)
    if block_hashes is None:
        return _fail("the request carries no block_hashes",
                     reason_kind="no_block_hashes")
    try:
        resolved = {}
        for group in groups:
            # THE view the engine itself uses: `resolve_block_hashes(block_hashes,
            # hash_block_size, block_size)` is what `block_pool.cache_block_hashes`
            # (`block_pool.py:269-271`) indexes when it publishes a block's hash, so
            # index `i` of this view is exactly the digest the spilled file for group
            # `g`'s block `i` was named after. `_eligible_groups` has already proved
            # every group here has this block size, so one view serves them all.
            resolved[group] = helpers["resolve"](block_hashes, hash_block_size,
                                                 block_size)
        lengths = {}
        for group in groups:
            lengths[group] = len(resolved[group])
    except Exception as exc:  # noqa: BLE001 - a view we cannot build is a refusal
        snapshot._error(exc, "disk_restore resolve_block_hashes")
        return _fail(f"resolve_block_hashes raised {type(exc).__name__}",
                     reason_kind="no_block_hashes")
    keys: list = []
    run = 0
    while run < MAX_RESTORE_BLOCKS_PER_OFFER:
        index = start_block + run
        block_keys: list = []
        complete = True
        for group in groups:
            if index >= lengths[group]:
                complete = False
                break
            try:
                digest = resolved[group][index]
            except Exception as exc:  # noqa: BLE001 - a hole in the view is a stop
                snapshot._error(exc, "disk_restore hash view")
                complete = False
                break
            if not isinstance(digest, bytes):
                return _fail(f"the engine's block hash at index {index} is "
                             f"{type(digest).__name__}, not bytes: the group "
                             "suffix can only be packed onto raw digests",
                             reason_kind="no_block_hashes")
            try:
                packed = helpers["pack"](digest, group)
                key_hex = bytes(packed).hex()
            except Exception as exc:  # noqa: BLE001 - a packer fault is a refusal
                snapshot._error(exc, "disk_restore make_block_hash_with_group_id")
                return _fail(f"make_block_hash_with_group_id raised "
                             f"{type(exc).__name__}", reason_kind="no_key_helpers")
            if not disk_index.contains(key_hex):
                complete = False
                break
            block_keys.append((group, key_hex))
        if not complete:
            break
        keys.append(block_keys)
        run += 1
    return _ok(blocks=run, keys=keys, view_lengths=lengths)


# --- the scheduler's door 1: offer the SSD's continuation ---------------------

def offer(manager, request, num_computed_tokens, ram_result):
    """Widen the fork's `(tokens, is_async)` with the SSD's contiguous span.

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
    _state["requests_seen"] += 1
    try:
        return _offer(manager, request, num_computed_tokens, ram_result)
    except Exception as exc:  # noqa: BLE001 - a restore never breaks a schedule
        _state["errors"] += 1
        snapshot._error(exc, "disk_restore offer")
        _refuse("bad_call", f"the offer path raised {type(exc).__name__}")
        return ram_result


def _skip_unless_flagged(request) -> bool:
    """True when this request must be handed straight back to the fork, at no cost.

    THE CHEAPEST CHECK IN THE PATH, and it has to be first: an unflagged request in
    `optin` scope must cost the fork's own lookup plus one attribute read - no gate,
    no arm file, no rank directory, no journal, no allocation. Reading the gate first
    would put a cached `open()` and a `listdir()` in front of every request the plugin
    has no business touching, which is the opposite of what a scope switch is for.

    It reports itself only when this boot is actually offering (`offers_enabled`), so
    a disarmed boot does not fill its counters with `skipped_no_marker` for requests
    that would have been refused anyway, and `why_no_restore()` keeps answering the
    gate's question first for anyone who asks it directly.
    """
    if not _state["offers_enabled"] or _state["scope"] != RESTORE_SCOPE_OPTIN:
        return False
    if request_flagged(request):
        _state["flagged"] += 1
        return False
    _state["skipped_no_marker"] += 1
    return True


def suppression_lookups() -> int:
    """How many requests are black-holed for disk service right now. For a test."""
    return len(_suppressed)


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


def _offer(manager, request, num_computed_tokens, ram_result) -> dict | tuple:
    if _skip_unless_flagged(request):
        # `optin` scope and this request never asked: not a refusal worth a counter
        # beyond `skipped_no_marker`, and not one byte of the fork's answer changed.
        return ram_result
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
    # THE MARKER IS THE AUTHORITY OVER THIS REQUEST'S OWN PREFIX (0.2.22). A flagged
    # request does not get "whatever the RAM tier left over": its host-RAM hit is
    # treated as ZERO for the offer arithmetic, so the disk index decides the span
    # from the local GPU-computed boundary. Everything downstream reads `ram_tokens`,
    # so this one assignment is the whole change of authority - and `computed` is
    # deliberately NOT touched, because it is the offset the fork's DMA writes at
    # (see the RESTORE_MARKER_KEY docstring: zeroing it would aim the copy at the
    # request's already-cached, shared prefix blocks).
    #
    # The fork keeps its side of a RAM hit in exactly one place, so ignoring the hit
    # means releasing exactly one thing (`_release_fork_pin`, below). Until that
    # release happens NOTHING here has consumed the fork's answer: every refusal
    # below still `return ram_result`, and a request that gets no disk offer keeps
    # the RAM hit the fork gave it.
    ram_tokens_reported = int(ram_tokens)
    disk_authority = request_flagged(request)
    fork_pending: object = _PIN_UNREAD
    if disk_authority:
        _state["disk_authority"] += 1
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
        # with its computed tokens RESET, so a request we already offered comes back
        # at the same place or earlier having made no progress - and the index has
        # not changed, so the honest answer to the second asking is the first answer
        # plus a failure. Only a request that has moved FORWARD gets a second offer.
        _state["suppressed_hits"] += 1
        suppress_load(request_id, f"it came back for a second disk offer at "
                                  f"{computed} computed tokens having reached "
                                  f"{_offered_at[request_id]} on the first one")
        _refuse("suppressed_after_failure",
                f"{request_id[:48]} was already offered disk tokens at "
                f"{_offered_at[request_id]} computed tokens and is asking again at "
                f"{computed}: the engine rescheduled it, so the first load did not "
                "land, and a second offer is the loop that starved the box")
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
    run = _disk_run(request, start_block, groups["groups"], block_size,
                    hash_block_size, helpers)
    if not run["ok"]:
        _refuse(run.get("reason_kind") or "no_block_hashes", run["reason"])
        return ram_result
    if run["blocks"] <= 0:
        _refuse("no_disk_run", f"the SSD tier holds no contiguous span starting at "
                               f"block {start_block} for all of groups "
                               f"{groups['groups']}")
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
    if disk_tokens > MAX_OFFER_TOKENS_PER_OFFER:
        # Clamped, not truncated mid-span: whole spans only, so the fork's own
        # "one whole block per entry" rule still holds after the clamp. This is the
        # ceiling that 160,000-token offers blew straight through.
        _state["offer_clamped"] += 1
        disk_tokens = whole_block_tokens(MAX_OFFER_TOKENS_PER_OFFER,
                                         block_size)["tokens"]
    if disk_tokens <= 0:
        _refuse("disk_run_below_one_block", f"{matched} matched tokens is below one "
                                            f"{block_size}-token scheduler block "
                                            f"({floored['discarded_tokens']} "
                                            "tokens discarded): offering it would "
                                            "park the request for a load the "
                                            "scheduler will throw away")
        return ram_result

    blocks = disk_tokens // block_size
    per_block = len(groups["groups"])
    needed = blocks * per_block
    if needed > MAX_SLOTS_PER_OFFER:
        blocks = MAX_SLOTS_PER_OFFER // per_block
        needed = blocks * per_block
        disk_tokens = blocks * block_size
    if disk_tokens <= 0:
        _refuse("no_cpu_slots", f"one span needs {per_block} CPU slots and the cap "
                                f"is {MAX_SLOTS_PER_OFFER}")
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
    affordable = max(0, (free - CPU_SLOT_RESERVE)) // per_block
    if affordable <= 0:
        _refuse("no_cpu_slots", f"the CPU pool has {free} free blocks and "
                                f"{CPU_SLOT_RESERVE} are reserved, so not even one "
                                "span can be given a destination")
        return ram_result
    if affordable < blocks:
        blocks = affordable
        needed = blocks * per_block
        disk_tokens = blocks * block_size
    # Every clamp above shortens the OFFER, so the resolved key list is cut to
    # match: merging more identities than there are slots would be an alignment
    # failure at the door that allocates, and one fewer would silently restore a
    # shorter span than was offered.
    keys = run["keys"][:blocks]
    allocated = _allocate(pool, needed)
    if not allocated["ok"]:
        _refuse("alloc_refused", allocated["reason"])
        return ram_result
    merged = _merge_into_pending(manager, request_id, ram_tokens, computed,
                                 block_size, groups["groups"],
                                 groups["num_groups"], allocated["blocks"],
                                 keys, opened["root"])
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
    _state["last_groups_required"] = required
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
        "groups_required": required["rows"][:16],
        "groups_unfilled": required["unfilled_group_ids"][:16],
    }
    _note(f"offered {disk_tokens} disk tokens for {request_id[:48]} in {blocks} "
          f"span(s) across {per_block} group(s)"
          + (f" [disk authority: local hit of {ram_tokens_reported} tokens "
             "not consulted]" if disk_authority else ""))
    emit_witness()
    return (int(ram_tokens) + disk_tokens, True)


def _refresh_index() -> dict:
    """Let the scheduler learn what the ranks spilled since it last asked.

    Bounded inside `disk_index.refresh` (one stat per journal, a byte cursor, a
    line cap, a 1 s floor) and never raising there; the try/except is for a fork
    that renamed something under it.
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
                        groups, num_groups, allocated, keys, root) -> dict:
    """Append our slots to the fork's per-group CPU-hit lists.

    Shape contract with `manager.py:381-492`: element 0 is a sequence with ONE
    entry per group (so the fork's `for g in range(num_groups)` cannot IndexError),
    our slots go at the END of each eligible group's list (the fork consumes
    LEADING entries, so "RAM first, then disk" is the token order it will pair),
    and the tuple becomes `(merged_lists, ram_tokens + disk_tokens,
    num_computed_tokens)` with the third field untouched - the fork needs that
    local prefix to place the external blocks and must not have it re-derived.

    A RAM hit whose per-group length disagrees with its own token count is REFUSED:
    the whole merge rests on "leading entries = the RAM hit, one per whole group
    block", and if that is not true the pairing would put our bytes at GPU offsets
    belonging to someone else's tokens.
    """
    per_block = len(groups)
    ram_blocks = ram_tokens // block_size
    needed = len(keys) * per_block
    if len(allocated) != needed:
        return _fail(f"{len(allocated)} slots were allocated for {needed}",
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
    index = 0
    for span in keys:
        for group, key_hex in span:
            named = _identity_path(root, key_hex)
            if not named["ok"]:
                return _fail(f"the identity path for {key_hex[:20]} could not be "
                             f"built: {named['reason']}",
                             reason_kind="path_build_failed")
            block = allocated[index]
            index += 1
            lists[group].append(block)
            identities.append({"group": group, "key": key_hex,
                               "cpu_block_id": int(block.block_id),
                               "path": named["path"]})
    if index != needed:
        return _fail(f"{index} identities built for {needed} slots",
                     reason_kind="ram_shape")
    hit_length = ram_tokens + len(keys) * block_size
    return _ok(pending=(tuple(lists), hit_length, ram_computed),
               identities=identities, hit_length=hit_length)


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
    out = {"released": 0, "restorations": 0, "failures": 0, "no_read": 0}
    try:
        reported = {str(item) for item in
                    (getattr(connector_output, "finished_recving", None) or ())}
        failed = {str(item) for item in
                  (getattr(connector_output, "failed_recving", None) or ())}
        invalid = _invalid_block_ids(connector_output)
        for request_id in sorted(failed):
            # EXACT attribution: the fork names the request whose load failed.
            record = _loads.get(request_id)
            suppress_load(request_id, "the engine reported this request's KV load "
                                      "failed",
                          span=_span_key_of(record))
            # THE LEDGER MUST CLOSE. Until 0.2.23 this line was `_loads.pop(...)`, a
            # hand-rolled duplicate of `_drop_load` that skipped its two side
            # effects: the `loads_released` counter and the release of this load's
            # `_slot_paths` entries. On a box where EVERY failure arrives in this
            # shape, `loads_released` sat at 0 forever while the fork's `_reqs_to_load`
            # drained - which is exactly the on-call reading "the load was never
            # registered at all" instead of "it was registered and burned", and it
            # left slot->path mappings naming a request that no longer had a load.
            _burn_load(request_id, "the engine reported this request's KV load "
                                   "failed")
            out["failures"] += 1
        if invalid and not failed:
            # The block-id-only shape: we own the load-error report, so on a step
            # where a disk load was in flight these are ours. Deliberately
            # over-broad - a missed restore costs a recompute, a missed failure
            # feeds the loop that starved the box.
            for request_id in list(_loads):
                if _loads[request_id].get("adopted"):
                    suppress_load(request_id, f"{invalid} GPU block id(s) came "
                                              "back as load errors on a step that "
                                              "carried this request's disk slots",
                                  span=_span_key_of(_loads.get(request_id)))
                    # The live box's whole failure signature: 66 requests with
                    # `invalid_block_ids=1..3` and an empty `failed_recving`, so this
                    # is the path that decided the run - and the path whose
                    # `_loads.pop` meant `loads_released` never moved by one. The
                    # suppression decision is unchanged; only the bookkeeping is.
                    _burn_load(request_id, f"{invalid} GPU block id(s) reported as "
                                           "load errors on a step carrying this "
                                           "request's disk slots")
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
    """How many GPU blocks this step reported as load errors. -1 if unreadable."""
    try:
        ids = getattr(connector_output, "invalid_block_ids", None)
        return len(ids) if ids else 0
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
    for job_id in indices:
        resolved = _resolve_worker_path(paths[job_id], digest)
        if not resolved["ok"]:
            kind = str(resolved.get("reason_kind") or "bad_path")
            failures[job_id] = (kind if kind in _state["reads_failed"]
                                else "bad_path")
            _state["reads_failed"][failures[job_id]] += 1
            continue
        asked = disk_read.request_read(batch_id, job_id, resolved["path"],
                                       expect_bytes)
        if not asked.get("ok"):
            kind = str(asked.get("reason_kind") or "read_failed")
            failures[job_id] = (kind if kind in _state["reads_failed"]
                                else "read_failed")
            _state["reads_failed"][failures[job_id]] += 1
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
                           "req_ids": _event_req_ids(metadata, event),
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
    # WHAT THE WAIT IS WAITING FOR. The budget is sized on the bytes STILL IN FLIGHT,
    # not the whole batch: a batch that has delivered nine of ten slots does not get a
    # ten-slot budget, and a batch that has delivered all of them gets nothing at all.
    # Every entry in one batch is the same width (the store's slot width for this
    # fingerprint), so the outstanding bytes are exact rather than estimated - and
    # `bytes_asked` is published next to it so a reader can see both numbers.
    pending = int(status.get("pending") or 0)
    per_entry = (bytes_asked / len(jobs)) if jobs else 0.0
    bytes_at_stake = int(round(per_entry * min(pending, len(jobs))))
    budget = settle_budget(bytes_at_stake, tokens, status.get("seconds_left"))
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
        key = (f"settle-on-{budget['cap_ms']}-{budget['read_bytes_per_s']}"
               f"-{budget['safety_margin']}")
        line = (f"DISK[settle] wait=ON cap_ms={budget['cap_ms']} "
                f"min_ms={budget['min_ms']} margin=x{budget['safety_margin']} "
                f"read_bps={budget['read_bytes_per_s']:.0f} "
                f"prefill_tokens_per_s={budget['prefill_tokens_per_s']} "
                f"(one line per distinct setting; the apply point waits inside the "
                f"engine step)")
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
    for job_id in drop:
        if job_id not in already_counted:
            # Bind-time failures were counted when they happened; counting them again
            # here would make one failed read two, and the counters are the only
            # evidence an operator has about which stage is losing reads.
            kind = failed.get(job_id)
            _state["reads_failed"][kind if kind in _state["reads_failed"]
                                   else "read_failed"] += 1
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
    # short of the bar; say what the numbers are rather than invent a category.
    return (f"partial: {int(_state['restorations'])} restoration(s) with "
            f"{int(reads['reads_applied'])} applied read(s) and "
            f"{int(_state['no_read'])} report(s) with no read behind them")


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
                 f"scope={_state['scope']}",
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
        # the counters say nothing happened: this boot's offer switch, the scope it
        # offers under, and the marker an opt-in request has to carry.
        parts.append(f"offers_switch={'ON' if _state['offers_enabled'] else 'OFF'}")
        parts.append(f"scope={_state['scope']}")
        parts.append(f"marker={RESTORE_MARKER_KEY}")
        parts.append(f"offer_cap={int(MAX_OFFER_TOKENS_PER_OFFER)}")
        parts.append("flagged={}".format(int(_state["flagged"])))
        parts.append("skipped_no_marker={}".format(int(_state["skipped_no_marker"])))
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
            "over_recompute": int(_state["settle_budget_over_recompute"]),
            "last": ({key: value for key, value in
                      (_state["last_apply_window"] or {}).items()
                      if key in ("bytes_asked", "budget_ms", "waited_ms", "bound",
                                 "settled", "settled_after_wait", "recompute_ms")}
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
            "out_of_order": int(read_state["completions_out_of_order"])}
        report["spans"] = {"seen": int(_state["spans_observed"]),
                           "whole": int(_state["spans_whole"]),
                           "partial": int(_state["spans_partial"]),
                           "empty": int(_state["spans_empty"]),
                           "hole_in_middle": int(
                               _state["spans_hole_in_the_middle"])}
        if len(json.dumps(report, default=str)) > worker_diag.MAX_CONTRIBUTION_BYTES:
            return {"section": "disk_restore", "writer": OWNER_TAG,
                    "truncated": True,
                    "installed": bool(_state["installed"]),
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
    """Offer the restore verdict to the rank-file writer. Idempotent."""
    from . import worker_diag  # noqa: PLC0415 - one writer, and no import cycle
    return worker_diag.register_contribution("disk_restore", worker_report,
                                             owner=OWNER_TAG)


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
    # THE POSTURE, FROM ITS OWNERS. `offer_posture()` owns the switch/scope/marker and
    # `counters()` owns every suppression number; this section re-publishes them and
    # recomputes nothing, so a payload and a verb can never disagree about a posture.
    posture = offer_posture()
    tally = counters()
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
        # THE ANSWER BEHIND THE ANSWER. A span spilled minutes ago is invisible to
        # an index that has not read its journal's tail, and `no_disk_run` cannot
        # tell "never spilled" apart from "spilled, unread". Say which this is.
        lag = int(index.get("lag_bytes") or 0)
        if lag:
            why += (f" The index is {lag} byte(s) of journal BEHIND the newest "
                    "spill lines, so a span spilled since the last refresh is not "
                    "knowable here yet (catch_up_jumps: "
                    f"{int(index.get('catch_up_jumps') or 0)}, bytes_skipped: "
                    f"{int(index.get('bytes_skipped') or 0)}).")
        # THE ARCHITECTURAL ANSWER, IN THE FIRST SENTENCE (0.2.25). An operator who
        # reads `why_no_restore` at 3 a.m. should not have to know what a MambaSpec is,
        # let alone that its recurrent state has no per-block file name: say that the
        # MODEL is un-serviceable, name the groups that decided it, and name the
        # refusal that will be counting every lookup.
        service = state["group_serviceability"]
        if service and not service.get("complete", True):
            why += (f" THIS MODEL CANNOT BE SERVED FROM DISK AT ALL: "
                    f"prefix-cacheable group(s) "
                    f"{service.get('blocked_group_ids')} cannot be restored while "
                    f"{service.get('restorable_group_ids')} can, and a disk offer "
                    "claims EVERY prefix-cacheable group because "
                    "get_num_new_matched_tokens skips prefill model-wide. Every "
                    "lookup here is refused as `incomplete_group_set` and recompute "
                    "is the only correct answer on this architecture.")
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
        # sentence at 3 a.m.: "is the switch up, under what scope, how many ranks are
        # armed, is anything fused" was only answerable by reading prose. Every value
        # is a pass-through from the function that owns it - `offer_posture()`,
        # `counters()`, `worker_proof()`, and for the cap the module constant the boot
        # line already prints - and none of them is derived here.
        "offers_enabled": bool(posture["enabled"]),
        "scope": posture["scope"],
        "marker_key": posture["marker_key"],
        "offer_cap": int(MAX_OFFER_TOKENS_PER_OFFER),
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
        # them. `loads_released` and `skipped_no_marker` complete the ledger: the
        # reads that finished and were let go, and the lookups that were left alone
        # because nobody asked.
        "reads_applied": int(tally["reads_applied"]),
        "disk_reads_applied": int(tally["disk_reads_applied"]),
        "ranks_reads_applied": int(tally["ranks_reads_applied"]),
        "loads_released": int(tally["loads_released"]),
        "skipped_no_marker": int(tally["skipped_no_marker"]),
        # THE MARKER'S OWN EVIDENCE (0.2.21). All three existed in `counters()` and in
        # `counters_view`'s units but were never published here, so PROVING a marker
        # reached the scheduler meant inventing a control experiment: send a PAIRED
        # unflagged request and watch `skipped_no_marker` move for the control and not
        # for the marked one - one extra request through a live engine to learn one
        # bit. `flagged` rising with the marked request while `requests_seen` rises for
        # both makes a marked request self-evident from one payload read, which is the
        # whole purpose of this section.
        "flagged": int(tally["flagged"]),
        "requests_seen": int(tally["requests_seen"]),
        "lookups": int(tally["lookups"]),
        # WHAT THE MARKER DID (0.2.22). A flagged request hands the disk the decision
        # about its own prefix; these four say whether that rule fired, whether it
        # cost a real host-RAM hit, and how many tokens that was. `last_offer` is the
        # structured proof a forced read and an ordinary one disagree about: a real
        # restore has `ram_tokens_reported` at or below what it used, a forced one
        # shows the hit the disk overrode.
        "disk_authority": int(tally["disk_authority"]),
        "local_hit_ignored": int(tally["local_hit_ignored"]),
        "local_hit_tokens_ignored": int(tally["local_hit_tokens_ignored"]),
        "pin_release_failures": int(tally["pin_release_failures"]),
        "last_local_hit_ignored": tally["last_local_hit_ignored"],
        "last_offer": tally["last_offer"],
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
            "index.lines_skipped": "journal lines passed over without being applied",
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
            "loads_released": "loads removed from the ledger, from counters()",
            "skipped_no_marker": "requests left exactly as the fork would have "
                                 "left them because they did not ask for a disk "
                                 "restore (`optin` scope without the marker), from "
                                 "counters(): the nothing-was-offered-because-nobody-"
                                 "asked case, kept apart from a refusal",
            "disk_authority": "flagged lookups where the disk, not the host-RAM "
                              "tier, decided the span (0.2.22). Every one of these "
                              "was offered at `computed // block_size`, i.e. the "
                              "disk span starts where the LOCAL GPU prefix ends - "
                              "the local hit is the one thing the marker cannot "
                              "un-learn, because the scheduler placed those blocks "
                              "before it ever asked us",
            "local_hit_ignored": "offers committed on a flagged request whose "
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
            "flagged": "requests that reached the offer path CARRYING the "
                       "kvtier_disk_restore marker, from counters(): the evidence a "
                       "marker survived the engine's own field-folding and arrived - "
                       "without it a `flagged: 0` and a `skipped_no_marker` that never "
                       "moved were the same observation twice",
            "requests_seen": "lookups observed by the request-scoped gate, from "
                             "counters(): the population `flagged` and "
                             "`skipped_no_marker` are drawn from, so the two must sum "
                             "to it while the scope is `optin`",
            "lookups": "lookups reaching the restore path at all, from counters(): "
                       "the entry count of the offer path, counted BEFORE the gate and "
                       "the scope, so a `lookups` that rises while `offers` sits at 0 "
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
            "offers_enabled": "this boot's offer switch, read from "
                              "offer_posture(): false until the dev verb raises it, "
                              "so an armed spill tier alone cannot put a disk offer "
                              "in front of live traffic",
            "scope": "which requests may be offered at all, from offer_posture(): "
                     "`optin` (only a request carrying `marker_key`) or `all`",
            "marker_key": "the `kv_transfer_params` key an opt-in request must carry "
                          "in `optin` scope, from offer_posture()",
            "offer_cap": "tokens: the most one offer may carry, clamped at "
                         "MAX_OFFER_TOKENS_PER_OFFER (the 160,000-token offers of "
                         "2026-10-05 clamp here)",
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
