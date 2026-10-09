"""The SSD tier's engine wiring: when a store COMPLETES in a worker, spill it.

This is the layer that turns `disk_spill.request_spill` from a ready-but-unwired
engine into a spill that happens. It owns three things and nothing else: the
worker-side store-completion hook, the token-gated `/plugins/kvtier/disk` verb,
and the snapshot section that says whether THIS boot can spill at all.

THE ENGINE IT TOUCHES (live tree, read, never edited)
-----------------------------------------------------
`vllm/v1/simple_kv_offload/` - that directory, not
`distributed/kv_transfer/kv_connector/v1/simple_cpu_offload/`, which does not
exist in this fork. The metadata type is `.../metadata.py`
(`SimpleCPUOffloadMetadata`), the worker handler is `.../worker.py`
(`SimpleCPUOffloadWorker`), and the engine patch that adds the fields is authored
as TWO directories with a mandatory apply order (`kvtier-disk-restore-meta`
declares the dataclass fields, `kvtier-disk-restore` fills them; the reverse order
is a `TypeError` on the first store). Nothing here imports vllm: every name is read
off an already-imported module or off the object the fork handed us, the
`worker_diag` posture.

WHERE THE WORKER LEARNS THAT A STORE COMPLETED (read out of the live fork)
-------------------------------------------------------------------------
The fork tells a worker nothing about a store except through CUDA events:

  * `bind_connector_metadata` (`worker.py:246-252`) keeps the step's metadata on
    the handler (`self._connector_metadata`) and adds `metadata.store_event` to
    `_pending_store_event_indices`. THIS is the only moment the per-slot lists
    (`store_cpu_blocks`, and on a patched engine `store_keys`) are in the worker's
    hands.
  * `wait_for_save` (`worker.py:276-303`) submits the copy tagged with that same
    `store_event` index.
  * `get_finished` (`worker.py:305-336`) polls the store stream; for every pending
    index at or below the new high-water mark it discards it from
    `_pending_store_event_indices` and records `self._completed_store_events[j] = 1`
    (`worker.py:330-334`). `build_connector_worker_meta` then drains that dict into
    the scheduler (`worker.py:338-346`).

So THE COMPLETION FACT APPEARS INSIDE `get_finished`, as the difference the fork
makes to `_completed_store_events`, while the key/slot pairing needed to name a
file existed only back at `bind_connector_metadata`. That is why this module wraps
BOTH methods and keeps its own bounded record between them:

  * the bind wrapper copies `(store_cpu_blocks, store_keys)` out of the metadata for
    the step's `store_event` - tuples of ints and hex strings, never a tensor and
    never the metadata object itself;
  * the `get_finished` wrapper diffs `_completed_store_events` around the fork's own
    call and, for each newly-completed event, hands `(slot, full block hash)` pairs
    to
    `disk_spill.request_spill`.

Composition, not replacement. Both methods already carry wrappers from this package
(`worker_diag` writes the per-rank file from `get_finished` and counts submissions
from the bind; `blockdump` answers byte dumps from `get_finished`). An install that
refused a sibling layer would make the INSTALL ORDER decide whether the rank files
exist - the `worker_diag`/`blockdump` lesson, and the reason
`engine_side._on_worker_report` installs this one LAST so it composes on top. The
fork's return value is handed back UNTOUCHED AND UNCONSUMED.

THE DIGEST: ONE COMPUTING SITE PER FIELD, NOT ONE PER BOOT
---------------------------------------------------------
An earlier draft of this contract said "never compute the digest worker-side; the
scheduler's `store_fingerprint` is the only source". That is IMPOSSIBLE, and
shipping it would have produced the exact failure it was meant to prevent.
`disk_layout.FINGERPRINT_FIELDS` include `total_bytes_per_block` (the live box
measures 11,468,800 B on PP0 and 12,902,400 B on PP1) and `rank` (`PP<n>_TP<m>`),
while there is ONE `SimpleCPUOffloadScheduler` broadcasting ONE `store_fingerprint`
to four ranks. No scheduler-computed digest can equal four different rank digests:
every rank would write under a name the others could never resolve, which is
permanently un-findable data - the one failure mode the naming scheme exists to
avoid.

What replaced it is narrower and satisfiable: **each field has exactly one computing
site.** `build_worker_fingerprint` assembles the fingerprint in the worker, and

  * the RANK-LOCAL fields are this process's own readings: `total_bytes_per_block`
    from the region tensors' strides (`disk_spill.measured_slot_bytes`), `rank` from
    `worker_diag.rank_id()` (the same place `blockdump` resolves a rank), `dtype`
    from the tensors that hold the bytes, and the group layout from
    `kv_cache_config.kv_cache_groups`;
  * the rank-INVARIANT fields are read by name off the objects the worker already
    holds (`vllm_config.model_config`, `.cache_config`, `.parallel_config`) and from
    `get_none_hash_seed()` on the already-imported `vllm.v1.core.kv_cache_utils`;
  * every site - this module, the tests, any future scheduler-side reader - calls
    the same `disk_layout.compute_geometry_digest`. One digest function, one field
    list, and nothing guessed: a field this worker cannot read is a counted
    `fingerprint_incomplete` refusal, never a made-up default.

`store_fingerprint` survives as an optional CROSS-CHECK. It arrives `""` today. An
empty value means NOTHING (it must never be read as "no fingerprint"); a non-empty
one that disagrees with this rank's own digest is a counted `fingerprint_mismatch`
refusal, because then two sites did compute the same field and something is wrong.

The property that had to survive the change is the one that ever mattered: **a
configuration change produces a different filename**, so stale bytes are
un-findable rather than mis-read. That holds worker-side exactly as before, and the
PP isolation is now carried by the width and rank fields themselves: a PP1 worker
cannot form a PP0 file's name because `total_bytes_per_block` and `rank` differ
(`tests/test_disk_store.py` pins both directions).

AN UNPATCHED ENGINE IS THE NORMAL CASE, AND IT IS VISIBLE
---------------------------------------------------------
On today's box the metadata carries none of the new fields. Every entry point here
then refuses with a reason and a counter and writes ZERO bytes:

    no_key_for_slot        `store_keys` absent, or the whole list empty
    key_count_mismatch     len(store_keys) != len(store_cpu_blocks)  (whole step)
    bad_key_text           one entry is not a usable hex hash (that ONE slot only)
    fingerprint_incomplete a fingerprint field could not be read in this worker
    fingerprint_mismatch   a non-empty store_fingerprint disagreed with this rank

and `engine_fields()` in every payload answers `present` / `absent` for this boot by
asking the already-imported metadata CLASS whether it declares `store_keys`. That
verdict - not a log grep - is how the operator knows whether the engine patch is
live. (`unknown` is the third answer, for a process that has not imported the
module, e.g. an API server. It is never collapsed into `absent`.)

SAFETY ENVELOPE (the `blockdump` posture, because this writes where that only reads)
------------------------------------------------------------------------------------
  * OFF BY DEFAULT, twice over: `VLLM_E2_TIER_DISK_ROOT` must name a real root AND
    `disk_spill` must be ARMED (no env var arms it; the arm is a LATCH, so it is
    armed by a decision and disarmed by one). Unarmed, the
    bind wrapper's whole cost is two `os.environ.get` calls and the `get_finished`
    wrapper adds one more: no stat, no open, no copy, no queue, no thread.
  * NEVER RAISES INTO THE ENGINE. Every wrapper calls the fork's method FIRST and
    then runs its own work inside `except Exception` (never `BaseException`: a
    `KeyboardInterrupt` or a `GeneratorExit` must still escape). Every public
    function returns a dict.
  * NEVER BLOCKS AN ENGINE THREAD. Nothing on the per-step path writes a file or
    reads a tensor. The only I/O a step may do is the spill-one relay's bounded
    control-file read, at most once per `POLL_INTERVAL_SECONDS`, and only while the
    whole facility is armed with a diagnostics directory configured. Bytes reach
    disk only on `disk_spill`'s own writer thread.
  * BOUNDED MEMORY. Retained steps: `MAX_RETAINED_EVENTS` events of
    `MAX_SLOTS_PER_EVENT` slots. De-duplication memory: `MAX_HANDLED_EVENTS`.
    Per-step hand-off: `MAX_SLOTS_PER_TICK` spill requests; the rest are counted,
    never queued. Fingerprint rebuilds: at most one per `FINGERPRINT_RETRY_SECONDS`
    while one is failing, so a worker whose config is unreadable does not re-walk
    fifteen attribute paths every step.
  * CAPS ON THE VERB. `ls` walks at most `disk_spill.MAX_SHARD_DIRS`/
    `MAX_WALK_FILES` and returns at most `limit` (cap `MAX_LS_FILES`) entries
    newest-first, and a payload is trimmed to `MAX_RESPONSE_BYTES` with
    `truncated: true` rather than allowed to grow. `ls` is read-only: the only
    destructive action in this design is `disk_spill`'s own LRU eviction.
  * PP STAGES CANNOT ADDRESS EACH OTHER'S FILES. The digest carries this rank's id
    and its own measured slot width, so a PP0 name cannot be produced by a PP1
    worker; and `disk_spill.read_slot_bytes` refuses a slot whose regions do not
    total the width the fingerprint declared, so even a hand-copied name would be
    refused at the read rather than spill the wrong bytes.

WHAT THIS DOES NOT CLAIM
------------------------
Nothing about restore any more - it landed in the same wheel. The two sentences that
used to sit here were a debt note, and both halves of it are paid, so they are
replaced by what actually holds:

  * THE READ PATH IS `disk_restore.py`, and this module's `/disk` verb answers with its
    section for exactly that reason: "it spilled 400 files and restored none" was
    unreadable while the two halves were read from two places.
  * THE RACE THIS NOTE WARNED ABOUT IS CLOSED, at the layer that could close it. The
    bytes used to be read on `disk_spill`'s writer thread, so a CPU slot the RAM tier
    recycled between completion and the write would land as its NEW contents under the
    OLD name. With no read path that was a wasted write, not a wrong answer. Now that a
    file CAN be served, one wrong align-mode GDN boundary state is a confident wrong
    answer at `temperature=0`, so `request_spill` copies the slot SYNCHRONOUSLY, on the
    thread the completion arrived on, and the queue carries bytes instead of slot
    numbers. The cost is one memcpy of a rank slot per spilled slot, bounded by
    `disk_spill.MAX_QUEUED_BYTES`; `disk_spill._handle_job` refuses a job that carries a
    slot number without a snapshot rather than reading one it does not own.

What remains genuinely not claimed: a file is not flushed (no fsync, by design, so a
crash leaves a `part` file or a committed file and never a half-written one), and the
hook never decides to spill on its own - the spill needs the armed hook AND
`disk_spine` AND the engine's per-layer store decision, each of which answers separately
in this payload.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import sys
import time

from . import (
    config,
    disk_gpu_shadow,
    disk_index,
    disk_layout,
    disk_shadow,
    disk_spill,
    mamba_anchor,
    snapshot,
)

# The engine's logging posture is why this module logs at WARNING, not INFO: a
# plugin logger is not one of the `vllm.*` loggers the engine configures, so an
# INFO line from here never reaches the log an operator greps. Measured on the box
# on 2026-10-05: `/tmp/vllm-serve.log` carries every `logger.warning` this plugin
# emits (the `BOOT[auto]` line, the connector-hook line, `KV tier ARMED`) and NOT
# ONE of its `logger.info` lines - zero matches for "SSD spill", "diagnostics
# writer", "attached to EngineCore". A verdict logged at INFO is a verdict nobody
# can read, and "the plugin never tried" was indistinguishable from "it tried and
# refused" for exactly that reason.
_logger = logging.getLogger("kvtier_dev_plugin.disk_store")
_logged_lines: list = []

# --- the fork's names, resolved only against an already-imported module ------

TARGET_MODULE = "vllm.v1.simple_kv_offload.worker"
TARGET_CLASS = "SimpleCPUOffloadWorker"
METADATA_MODULE = "vllm.v1.simple_kv_offload.metadata"
METADATA_CLASS = "SimpleCPUOffloadMetadata"
SEED_MODULE = "vllm.v1.core.kv_cache_utils"      # get_none_hash_seed(), :147
PARALLEL_STATE_MODULE = "vllm.distributed.parallel_state"

METHOD_BIND_METADATA = "bind_connector_metadata"
METHOD_GET_FINISHED = "get_finished"
WRAPPED_METHODS = (METHOD_BIND_METADATA, METHOD_GET_FINISHED)

# A layer of THIS package on the same method (`worker_diag`, `blockdump`) is
# composed with, never refused; anything else is refused outright.
SIBLING_PACKAGE = (__package__ or "kvtier_dev_plugin") + "."

OWNER_ATTR = "__kvtier_dev_diskstore_owner__"
OWNER_TAG = "kvtier-dev-plugin/disk-store-2"
ORIGINAL_ATTR = "__kvtier_dev_diskstore_original__"

# --- the wire contract, as names --------------------------------------------

FIELD_STORE_KEYS = "store_keys"
FIELD_STORE_FINGERPRINT = "store_fingerprint"
FIELD_LOAD_PATHS = "load_paths"
INVALID_JOB_ID = -1                 # metadata.py:12
# A key the engine could not fill is the empty string, never a dropped entry
# (contract), so a placeholder costs ONE slot and not the whole step.
KEY_PLACEHOLDER = ""

# --- the caps, and the unit of every one of them ----------------------------

MAX_RETAINED_EVENTS = 32            # STEPS remembered between bind and completion
MAX_SLOTS_PER_EVENT = 4096          # SLOTS copied out of one step's metadata
MAX_COMPLETED_SCAN = 512            # ENTRIES of `_completed_store_events` we diff
MAX_SLOTS_PER_TICK = 64             # spill requests handed over per step
MAX_HANDLED_EVENTS = 256            # EVENTS remembered as already spilled
FINGERPRINT_RETRY_SECONDS = 5.0     # SECONDS between rebuilds while one is failing
MAX_MISSING_REPORTED = 16           # unreadable fields named in one report
MAX_SOURCES_PER_FIELD = 6           # candidate read paths tried for one field

POLL_INTERVAL_SECONDS = 2.0         # SECONDS between spill-one relay polls
REQUEST_TTL_SECONDS = 120.0         # SECONDS a spill-one request stays live
MAX_BOOT_LINE_CHARS = 160           # CHARS of one refusal reason in the boot line
PROTOCOL_VERSION = 2                # 1: the one-digest-per-boot contract (impossible)
ROLE = "disk-spill-one"
WRITER_TAG = "kvtier-dev-plugin/disk-store-2"

PENDING_FILE = "disk-spill-one.json"
RECEIPT_FILE = "disk-spill-one-receipt-{rank}.json"
RECEIPT_PREFIX = "disk-spill-one-receipt-"
FILE_SUFFIX = ".json"
MAX_PENDING_FILE_BYTES = 2048
MAX_RECEIPT_FILE_BYTES = 1024
MAX_RANK_FILES = 32                 # receipt files one status read will open
MAX_SERVED_RECORDS = 64             # per-process memory of answered requests

# --- the cross-process spill ROLLUP: reading the ranks' own rank files --------
# The counters in `disk_spill._state` are PER PROCESS, and the processes are many:
# the spill writer thread and the ceiling enforcement run in every WORKER, while the
# `/plugins/kvtier/disk` door is answered by whichever process the utility call
# landed on. On a 4-rank boot there are four independent spill states, and the one
# that answers the HTTP call may have done none of the work. So the door reads the
# per-rank files the WORKER ALREADY WRITES (`worker_diag`) and sums what each rank
# saw - no shared memory, no new IPC, no new file: an honest sum of separate
# processes' counters, labelled as one. See `rank_rollup` and README *The door that
# showed zeros*.
RANK_FILE_PREFIX = "rank-"              # worker_diag's name; READ here, never written
RANK_FILE_SUFFIX = ".json"              # `rank-PP<pp>_TP<tp>.json`
MAX_RANK_ROLLUP_FILE_BYTES = 256 * 1024  # the reader's own ceiling, same rule as proof
MAX_RANK_ROLLUP_FILES = 32              # rank files one status read will open
MAX_RANK_ROLLUP_ROWS = 16               # per-rank rows in one rollup payload
MAX_RANK_ROLLUP_EXCLUDED = 16           # excluded files NAMED in one rollup payload
RANK_ROLLUP_COUNTER_MAX = 1_000_000_000_000  # 1 TiB: past this a counter is not a fact
RANK_ROLLUP_FRESH_SECONDS = 900.0       # the same heartbeat bar as disk_restore's proof
# HOW OFTEN THE RANK FILES ARE RE-READ. One second, the SAME idiom and the same number
# as `disk_restore.PROOF_CACHE_SECONDS`: `/snapshot` and `/status` are polled by a
# dashboard every few seconds, and both now carry a rank-derived view, so without a
# cache every poll would open up to `MAX_RANK_ROLLUP_FILES` JSON files. The cache is
# keyed on a CONTENT SIGNATURE (the directory plus each rank file's size and mtime), not
# on the clock alone, so a rank that writes is picked up on the very next read and a
# burst of polls against a quiet fleet costs one parse. Bounded staleness is the point:
# a signature miss reparses immediately, and a hit is never older than this.
RANK_ROLLUP_CACHE_SECONDS = 1.0
DISK_FILL_RANK_CAP = 16                 # rank entries in ONE `disk_fill` block

MAX_LS_FILES = 200                  # ENTRIES in an `ls` response
MAX_LS_LIMIT_DEFAULT = 50
MAX_IDENTITIES_LIMIT = 200          # IDENTITIES in an `identities` page
MAX_RANK_NAME_CHARS = 32            # CHARS of a rank label in a verb argument
MAX_FILENAME_DISPLAY = 200          # CHARS of a path shown in a response
MAX_RESPONSE_BYTES = 256 * 1024     # BYTES: any verb payload's hard ceiling
MAX_STRING_CHARS = 160

DISK_OPS = ("status", "arm", "disarm", "spill-one", "ls", "contains", "identities",
             "offers", "authority", "failures", "shadow", "gpu-shadow",
             "restore-verify", "gpu-write", "restore-source", "adoptions")
MUTATING_DISK_OPS = ("arm", "disarm", "spill-one", "offers", "authority", "shadow",
                     "gpu-shadow", "restore-verify", "gpu-write", "restore-source")
"""`offers` and `authority` are mutating and POST-only: each raises or drops ONE of
this boot's two disk switches. Neither is the arm. Arming the spill tier says bytes
may be WRITTEN; these say requests may be SERVED from disk, which on 2026-10-05 was a
difference of 123,474 un-requested 160,000-token offers.

`authority` is the one that means "serving everyone from SSD": the disk, not the
request's host-RAM hit, decides every request's own prefix. It replaced the per-request
opt-in marker (and the `scope` argument that went with it) in 0.2.31, because a request
body must not decide whether its KV comes off NVMe."""

# This module's own refusal vocabulary. `no_key_for_slot` and `bad_digest` share a
# WORD with `disk_spill.REFUSAL_REASONS` because they are the same operational
# statement made one level higher. `delegated_refusals` is a MIRROR (the same
# refusals `disk_spill.refused` counts), never to be summed with them.
REFUSAL_REASONS = ("no_key_for_slot", "key_count_mismatch", "bad_key_text",
                   "no_store_slots", "over_slot_cap",
                   "unretained_event", "fingerprint_incomplete",
                   "fingerprint_mismatch", "bad_digest", "bad_call", "not_armed",
                   "not_configured", "no_channel", "expired", "over_tick_cap")

# --- SPAN COMPLETENESS AT THE HAND-OFF ----------------------------------------
#
# WHY THESE COUNTERS EXIST. A span on the SSD tier is one file PER PREFIX-CACHEABLE
# GROUP, and the restore walk needs those files in two different shapes
# (`disk_restore._disk_run`, disk_restore.py:5829-5852): a POSITIONAL group's file for
# every block of the run, and an ALIGN-MODE group's file for ONE block - the run's END.
# That one file is the anchor, and without it the offer dies as `no_boundary_anchor`
# ("the positional groups have the whole run and NO run end has a boundary-state file
# in every align-mode group"). So a hand-off that carries the attention block and not
# the align state produces a span that can extend a run but never end one, and one that
# carries the align state without attention produces a span no run can walk over.
#
# THE MEASURED COST, from the per-rank spill journals over 71 hours on the live box
# (2026-10-06 22:26 to 2026-10-09 21:32, 150k lines per rank, seq starting at 1):
# 16,597 spans received an attention spill and 3,326 of them (20.0%) received NO align
# state on ANY rank, while 2,281 received align state without attention - together
# about a quarter of everything ever spilled, dead on arrival, unanchored by
# construction. The rate is FLAT across the whole journal (20.4% in the first 5,000
# attention spills, 20.1% by the 65,000th), so this is not a regression: it is the
# steady-state shape of the fork's boundary hand-off supply since the align-mode tier
# landed (`9f8153d`, 2026-10-06), and it was invisible because the only way to see it
# was to fold four journals by hand.
#
# WHY MEASURING IS ALL THIS MODULE DOES. The align bytes for a block are in host RAM
# only when the fork published a boundary hand-off for it: the connector allocates a
# CPU block inside that loop and nowhere else (`manager.py:701-745` of the live tree),
# and the metadata that reaches this process is flat - `store_cpu_blocks` plus
# `store_keys`, with the group recoverable only from the key's 4-byte suffix
# (`metadata.py:35-37`; `disk_layout.split_block_key`). So a pair is either already in
# `record["pairs"]` below - and today it is already spilled - or the copy never
# happened, and there is nothing here that can reach the state without reading GPU
# memory on the step thread. Widening the supply is a fork change; making the 20% a
# live number is this change.
#
# THE RULE THAT MAKES IT SAFE: strictly additive. A `{5}`-only span may receive its
# align state from a LATER hand-off, so a span that looks incomplete here is a maybe,
# and refusing it would make it a never. Nothing below gates, drops, reorders or
# refuses a spill; the classification reads `record["pairs"]`, counts, and stops.
SPAN_SHAPE_COUNTERS = ("spill_spans_classified", "spill_spans_complete",
                       "spill_incomplete_state", "spill_incomplete_attention")

# THE ONE VALUE, duplicated on purpose. `disk_restore.MAMBA_CACHE_MODE_ALIGN`
# (disk_restore.py:1612) reads the same string from the manager; disk_restore IMPORTS
# this module, so a hot-path `from . import disk_restore` inside the spill hook would
# make the spill's cheapest witness depend on loading the heaviest layer in the
# package. `test_the_two_readings_of_the_forks_align_mode_cannot_drift` keeps the two
# copies honest, which is the cheap direction for that trade.
MAMBA_CACHE_MODE_ALIGN = "align"

# WHERE the split was read from, published beside the numbers so a reader can tell
# "the geometry says X" from "the plugin guessed". The base string is the source that
# has always worked; `resolve_mamba_cache_mode` APPENDS the source it actually used,
# because a per-group claim read from an engine-wide flag has to be readable as one.
SPAN_SHAPE_BASIS = ("kv_cache_groups[i].kv_cache_spec."
                    "{prefix_cacheable,mamba_cache_mode}")

# THE THREE PLACES a group's mode can be true of, in the order they are tried, each
# optional. They are published inside `basis`/`reason` rather than kept private, so
# "how do you know this group is a state group?" is answerable from the payload.
MODE_FROM_SPEC = "spec"
MODE_FROM_SUB_SPECS = "sub-specs"
MODE_FROM_CACHE_CONFIG = "engine-wide cache_config"
MODE_FROM_AMBIGUOUS = "sub-specs disagree"

# A class name is EVIDENCE about a spec the same way the fork reads one
# (`gpu_census._is_circular_buffer`'s MRO-by-class-name, `kvconfig.py:241-255`): it
# says WHICH group, never WHETHER align mode is on. Only a real
# `mamba_cache_mode == "align"` may classify a group, and the name marker below is
# consulted solely to decide which group an ENGINE-WIDE "align" may be attributed to.
STATE_CLASS_MARKERS = ("Mamba",)

# BOUNDS on the evidence strings. `worker_diag.MAX_CONTRIBUTION_BYTES` refuses (here:
# trims) an over-long rank section, so a reason that named thirty groups and forty
# classes would cost every counter beside it. These are the reason strings' ceiling,
# and `test_the_disk_section_still_fits_the_contribution_cap` holds the line; the class
# cap is TWO because the live pair - the container and the sub-spec class inside it - is
# the fact worth naming, and a wider list pushes the engine-wide verdict past the clip.
MODE_EVIDENCE_CLASSES = 2
MODE_CLASS_CHARS = 40
MODE_MRO_CLASSES = 8
MODE_SUB_SPECS_SCANNED = 64
SPAN_SHAPE_REASON_CHARS = 400
SPAN_SHAPE_BASIS_CHARS = 140
SPAN_SHAPE_ID_CHARS = 32

# The last readable answer, and the config object it was read out of. A holder rather
# than two module globals because this module has already been crashed twice by a
# `global` declared in the wrong place (`test_global_shadowing.py`).
_span_shape_cache: dict = {"config": None, "shapes": None}

_state: dict = {
    # the hook
    "installed": False,
    "installs": 0,
    "reinstalls": 0,
    "uninstalls": 0,
    "class": None,
    "skip_reason": None,
    "nested": [],
    "binds": 0,
    "steps_idle": 0,
    "completions": 0,
    "events_retained": 0,
    "retained_dropped": 0,
    "retained_now": 0,
    "slots_offered": 0,
    "spill_queued": 0,
    "delegated_refusals": 0,
    "slots_capped": 0,
    "scan_skipped": 0,
    # THE COMPLETENESS TALLY (see SPAN_SHAPE_COUNTERS above). Three mutually
    # exclusive classes plus their denominator, so the orphan rate is a division
    # rather than a journal-folding exercise: `spill_incomplete_state` counts the
    # spans that can never ANCHOR a run, `spill_incomplete_attention` the ones no run
    # can walk over, `spill_spans_complete` the rest, and `spill_spans_classified`
    # every span the tier was able to classify at all.
    "spill_spans_classified": 0,
    "spill_spans_complete": 0,
    "spill_incomplete_state": 0,
    "spill_incomplete_attention": 0,
    # THE ANCHORS THAT REACHED THE STORE (0.2.64). One per ALIGN-MODE pair a hand-off
    # carried: an align-mode group appears in a hand-off only at a block the fork's
    # mask let survive, and those are exactly the boundary-state ANCHOR files a run end
    # is priced with. This is the store-side landing count of the blocks
    # `mamba_anchor`'s `anchor_offers_total` marked reachable at the mask - the two
    # ends of the same pipe, each measured where it is observable.
    #
    # WHY IT IS NOT FILTERED BY "DID THE PATCH MARK THIS ONE". A hand-off carries
    # `(slot, block-hash)` pairs and nothing else (`_remember_event`): no token
    # position, no request id, so no per-block causal join from the mask to a file
    # exists in-process. The patch's own effect is therefore read as this counter's
    # RATE against `spill_incomplete_state` MOVING when
    # `VLLM_E2_TIER_DEV_MAMBA_ANCHOR_PATCH` flips - the A/B the box has to measure.
    # The counter is also unconditional in THIS process for a second, sharper reason:
    # the workers do the spilling and the patch is fitted in the scheduler, so a worker
    # that gated on a local install flag would report zero forever.
    "anchor_spills_completed": 0,
    # Pairs this module could not read (no group suffix, not a key, a group the table
    # does not have). They are IGNORED, never a span of their own, and published so an
    # all-zero tally can be told apart from a classifier being fed garbage.
    "span_shape_pairs_ignored": 0,
    "refused": {reason: 0 for reason in REFUSAL_REASONS},
    "refusals_total": 0,
    "last_refusal": None,
    "boot_line": None,
    "fallbacks": 0,
    "errors": 0,
    # the rank-local fingerprint this process names files under
    "fingerprint": None,
    "geometry_digest": None,
    "fingerprint_sources": {},
    "fingerprint_missing": [],
    "fingerprint_attempts": 0,
    "fingerprint_attempted_mono": None,
    "fingerprint_normalized_none": [],
    # the engine-field probe (worker: the last metadata bound; any process: the
    # metadata CLASS as it was imported here)
    "keys_observed_on_a_step": None,
    "fingerprint_observed_on_a_step": None,
    "load_paths_observed_on_a_step": None,
    # the spill-one relay
    "polls": 0,
    "poll_suppressed": 0,
    "last_poll_mono": None,
    "pending_stat": None,
    "pending_reads": 0,
    "pending_malformed": 0,
    "adopted": 0,
    "served": {},
    "handled": {},
    "requests_written": 0,
    "fulfilled": 0,
    "fulfil_errors": 0,
    "receipt_errors": 0,
    "pending_seq": None,
    "last_rank": None,
    "last_reason": None,
    "last_error": None,
    "last_path": None,
}


def counters() -> dict:
    """A copy for a status payload. Nothing big, nothing secret, no file reads."""
    return dict(_state)


def reset_state_for_tests() -> None:
    """Back to a fresh boot. Counters only: this module starts no thread and owns no
    queue (the writer thread belongs to `disk_spill`, whose own reset stops it)."""
    _state.update({
        "installed": False, "installs": 0, "reinstalls": 0, "uninstalls": 0,
        "class": None, "skip_reason": None, "nested": [],
        "binds": 0, "steps_idle": 0, "completions": 0,
        "events_retained": 0, "retained_dropped": 0, "retained_now": 0,
        "slots_offered": 0, "spill_queued": 0, "delegated_refusals": 0,
        "slots_capped": 0, "scan_skipped": 0,
        "spill_spans_classified": 0, "spill_spans_complete": 0,
        "spill_incomplete_state": 0, "spill_incomplete_attention": 0,
        "anchor_spills_completed": 0,
        "span_shape_pairs_ignored": 0,
        "refused": {reason: 0 for reason in REFUSAL_REASONS},
        "refusals_total": 0, "last_refusal": None, "boot_line": None,
        "fallbacks": 0, "errors": 0,
        "fingerprint": None, "geometry_digest": None,
        "fingerprint_sources": {}, "fingerprint_missing": [],
        "fingerprint_attempts": 0, "fingerprint_attempted_mono": None,
        "fingerprint_normalized_none": [],
        "keys_observed_on_a_step": None,
        "fingerprint_observed_on_a_step": None,
        "load_paths_observed_on_a_step": None,
        "polls": 0, "poll_suppressed": 0, "last_poll_mono": None,
        "pending_stat": None, "pending_reads": 0, "pending_malformed": 0,
        "adopted": 0, "requests_written": 0, "fulfilled": 0,
        "fulfil_errors": 0, "receipt_errors": 0, "pending_seq": None,
        "last_rank": None, "last_reason": None, "last_error": None,
        "last_path": None,
    })
    _state["served"] = {}
    _state["handled"] = {}
    _span_shape_cache["config"] = None
    _span_shape_cache["shapes"] = None
    _logged_lines.clear()
    _retained.clear()


# The one piece of between-call memory that is not a counter: the (slot, key) pairs
# pairs kept between `bind_connector_metadata` and store completion. Module-level
# because it is worker state that outlives one call, BOUNDED by `_remember_event`
# because a map that grows with load is an outage with a timer. The event-dedup set
# lives in `_state["handled"]` (same shape, same cap idiom, visible in `counters()`).
_retained: dict = {}


# --- small helpers ------------------------------------------------------------

def _fail(reason, **extra) -> dict:
    out = {"ok": False, "reason": str(reason)}
    out.update(extra)
    return out


def _ok(**fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    return out


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _clip(text, chars: int = MAX_STRING_CHARS):
    if isinstance(text, str) and len(text) > chars:
        return text[:chars] + "..."
    return text


def _count(reason: str, detail: str | None = None) -> None:
    """One counted refusal, and the last one remembered. Never raises.

    `refusals_total` and `last_refusal` exist because a rank file has to answer
    "did the tier refuse, and how many times" in ONE reading. A per-reason table
    whose non-zero entry is buried among fifteen zeros is how a live box looked
    idle: the counters were right and nobody could see them.
    """
    key = reason if reason in _state["refused"] else "bad_call"
    _state["refused"][key] = int(_state["refused"][key]) + 1
    _state["refusals_total"] = int(_state["refusals_total"]) + 1
    _state["last_refusal"] = {"reason": key,
                              "count": int(_state["refused"][key]),
                              "detail": _clip(detail, 90) if detail else None}
    if detail is not None:
        _state["last_reason"] = _clip(detail)


def _remember(key: str, value) -> None:
    """Replace a bounded dict-valued piece of state rather than mutating it.

    The `blockdump._remember_record` idiom: a test fixture that saved a shallow copy
    of `_state` gets its OWN dict back, so a leak cannot cross a test.
    """
    current = dict(_state.get(key) or {})
    current[str(value)] = round(time.time(), 6)
    cap = MAX_HANDLED_EVENTS if key == "handled" else MAX_SERVED_RECORDS
    while len(current) > cap:
        current.pop(next(iter(current)))
    _state[key] = current


def _may_touch_disk() -> bool:
    """The two-switch gate, asked on the per-step path. Never creates anything.

    Root set (an `os.environ.get`) and the arm (`disk_spill.is_armed()`, whose file
    verdict is cached for `disk_spill.ARM_FILE_POLL_SECONDS`). Unarmed this is two
    cheap lookups and at most one `stat` per second. A caller that had work to do
    counts its own refusal; the gate itself stays silent.
    """
    if disk_spill.disk_root() is None:
        return False
    return bool(disk_spill.is_armed())


# --- the engine-field probe ---------------------------------------------------

def _declares(cls, name: str) -> bool:
    """Does this metadata TYPE carry `name`? Three honest routes, one answer.

    A dataclass with a mutable default (`field(default_factory=list)`) has NO class
    attribute for it, so `hasattr` alone would call a patched engine unpatched. The
    annotation and the dataclass field list both see it; `hasattr` catches a plain
    attribute or a test double that only assigns in `__init__`.
    """
    if cls is None:
        return False
    try:
        if dataclasses.is_dataclass(cls) and any(
                field.name == name for field in dataclasses.fields(cls)):
            return True
    except Exception as exc:  # noqa: BLE001 - a broken type is an unknown, not a fault
        snapshot._error(exc, "disk_store field probe")
    annotations = getattr(cls, "__annotations__", None)
    if isinstance(annotations, dict) and name in annotations:
        return True
    return hasattr(cls, name)


def metadata_class():
    """`SimpleCPUOffloadMetadata` if this process imported it, else None."""
    module = sys.modules.get(METADATA_MODULE)
    if module is None:
        return None
    return getattr(module, METADATA_CLASS, None)


def engine_fields() -> dict:
    """Is THIS BOOT's metadata carrying the new fields? `present|absent|unknown`.

    THE QUESTION IS `store_keys`, and only `store_keys`: it is the field the spill
    cannot work without. `store_fingerprint` is a cross-check the patched engine
    sends as `""`, so its presence changes nothing about whether a spill is
    possible, and `load_paths` belongs to restore. All three are still reported,
    because "the patch is installed but half of it is missing" is exactly the state
    an operator needs to see.

    Two kinds of evidence, and the payload says which answered:
      * the TYPE, in this process: the already-imported metadata class declares
        `store_keys`. That is the "is the engine patch live" question, answerable
        before a single store happens - the whole point of reading it from a
        snapshot.
      * an INSTANCE, worker-side only: the last metadata object this process bound
        actually had the attribute (`observed_on_a_step`). Runtime proof, and the
        one an EngineCore process can never produce.
    `unknown` means this process has not imported the metadata module; it is never
    collapsed into `absent`, the same discipline as the byte dump's capability.
    """
    cls = metadata_class()
    if cls is None:
        verdict = "unknown"
        evidence = (f"{METADATA_MODULE} is not imported in this process, so the "
                    f"{METADATA_CLASS} shape is unknowable here")
        class_probe: object = "not imported"
    else:
        keys = _declares(cls, FIELD_STORE_KEYS)
        fingerprint = _declares(cls, FIELD_STORE_FINGERPRINT)
        paths = _declares(cls, FIELD_LOAD_PATHS)
        class_probe = {"store_keys": keys, "store_fingerprint": fingerprint,
                       "load_paths": paths}
        verdict = "present" if keys else "absent"
        evidence = (f"{cls.__name__} declares store_keys={keys}, "
                    f"store_fingerprint={fingerprint}, load_paths={paths}")
    observed = _state["keys_observed_on_a_step"]
    if observed is True:
        verdict = "present"
        evidence = "a metadata object this process bound carried store_keys"
    return {"verdict": verdict,
            "store_keys": verdict == "present",
            "evidence": _clip(evidence),
            "observed_on_a_step": observed,
            "fingerprint_observed_on_a_step":
                _state["fingerprint_observed_on_a_step"],
            "class_probe": class_probe,
            "note": ("`absent` means this boot's SimpleCPUOffloadMetadata has no "
                     "store_keys: the spill path refuses and writes nothing, and "
                     "that is the expected state until the engine patch is live. "
                     "`unknown` means this process cannot see the type at all. "
                     "store_fingerprint is a cross-check, not a requirement: the "
                     'engine sends "" and that means nothing.')}


def observe_metadata_fields(metadata) -> dict:
    """Record what THIS metadata object carries. Contained; reads attributes only."""
    keys = hasattr(metadata, FIELD_STORE_KEYS)
    fingerprint = hasattr(metadata, FIELD_STORE_FINGERPRINT)
    paths = hasattr(metadata, FIELD_LOAD_PATHS)
    _state["keys_observed_on_a_step"] = bool(keys)
    _state["fingerprint_observed_on_a_step"] = bool(fingerprint)
    _state["load_paths_observed_on_a_step"] = bool(paths)
    return {"ok": True, "reason": None, "store_keys": bool(keys),
            "store_fingerprint": bool(fingerprint), "load_paths": bool(paths)}


# --- the rank-local fingerprint: what THIS worker names files under ----------

# Each rank-invariant field, and where a worker may read it from, in order. These
# are the FORK'S names, verified against the live tree (config/model.py:127/204,
# config/cache.py:73/91/111/132, config/parallel.py:126/338/357,
# v1/kv_cache_interface.py:1392-1410), and a name that does not answer is a REPORTED
# missing field - never a default. Two paths for a field is a fallback for a fork
# that renamed one; more than a few would be a guess with a table on it.
CONFIG_SOURCES = {
    "model_id": (("vllm_config", "model_config", "model"),
                 ("vllm_config", "model_config", "served_model_name")),
    "model_revision": (("vllm_config", "model_config", "revision"),),
    "cache_dtype": (("vllm_config", "cache_config", "cache_dtype"),),
    "prefix_caching_hash_algo": (
        ("vllm_config", "cache_config", "prefix_caching_hash_algo"),),
    "scheduler_block_size": (("vllm_config", "cache_config", "block_size"),),
    "hash_block_size": (("vllm_config", "cache_config", "prefix_match_unit"),
                        ("vllm_config", "cache_config", "block_size")),
    "world_size": (("vllm_config", "parallel_config", "world_size"),
                   ("__world_group__",)),
    "dcp_world_size": (
        ("vllm_config", "parallel_config", "decode_context_parallel_size"),
        ("__dcp_group__",)),
    "cp_world_size": (
        ("vllm_config", "parallel_config", "prefill_context_parallel_size"),
        ("__pcp_group__",)),
}

# The rank-local fields: computed here, because these are exactly the ones a single
# scheduler cannot broadcast to four ranks.
LOCAL_FIELDS = ("total_bytes_per_block", "rank", "dtype", "group_block_sizes",
                "prefix_cacheable_group_ids", "none_hash_seed")

# A fork-legal `None` that has one stable rendering. `model_revision=None` is the
# normal state of a model checked out by revision name rather than hash, and a
# fingerprint that refused it would make the tier unusable on every ordinary boot;
# a refusal would also not be SAFER, since the field's purpose (a weight change
# under the same name must not revalidate old files) survives any stable text.
NONE_REPLACEMENTS = {"model_revision": "none"}


def _read_path(handler, path) -> tuple:
    """(value, source) for one dotted candidate, or (None, what was tried)."""
    current = handler
    walked = []
    for part in path:
        if part == "__world_group__":
            return _group_world_size("get_world_group")
        if part == "__dcp_group__":
            return _group_world_size("get_dcp_group")
        if part == "__pcp_group__":
            return _group_world_size("get_pcp_group")
        if current is None:
            return None, f"{'.'.join(walked)} is absent"
        current = getattr(current, part, None)
        walked.append(part)
        if current is None:
            return None, f"{'.'.join(walked)} is absent"
    return current, ".".join(walked)


def _group_world_size(getter: str) -> tuple:
    """A parallel group's world size, from the ALREADY-IMPORTED parallel_state.

    The rule `worker_diag.rank_id()` applies to `rank_in_group`: a worker has
    distributed state, and this module never imports the module itself. A scheduler
    or API process raises inside those getters (no `init_process_group`), which is
    the honest "not readable here", not an error.
    """
    module = sys.modules.get(PARALLEL_STATE_MODULE)
    if module is None:
        return None, f"{PARALLEL_STATE_MODULE} is not imported here"
    function = getattr(module, getter, None)
    if not callable(function):
        return None, f"{PARALLEL_STATE_MODULE}.{getter} is missing"
    try:
        group = function()
    except Exception as exc:  # noqa: BLE001 - no distributed state is a fact, not a fault
        return None, f"{getter}() refused: {type(exc).__name__}"
    world = _int(getattr(group, "world_size", None))
    if world is None or world < 1:
        return None, f"{getter}().world_size is not a positive int"
    return world, f"parallel_state.{getter}().world_size"


def _group_fields(handler) -> tuple:
    """(group_block_sizes, prefix_cacheable_group_ids, source, refusal).

    From `handler.kv_cache_config.kv_cache_groups` (`worker.py:41` keeps it and
    `register_kv_caches` reads it at `worker.py:101-106`), index-ORDERED: the
    fingerprint's group list is positional identity, so an unordered set here would
    fold two different layouts into one digest.
    """
    kv_config = getattr(handler, "kv_cache_config", None)
    groups = getattr(kv_config, "kv_cache_groups", None)
    if not groups:
        return None, None, None, ("kv_cache_config.kv_cache_groups is absent or "
                                  "empty")
    sizes = []
    cacheable = []
    for index, group in enumerate(list(groups)):
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            return None, None, None, f"group {index} has no kv_cache_spec"
        size = _int(getattr(spec, "block_size", None))
        if size is None or size < 1:
            return None, None, None, (f"group {index} kv_cache_spec.block_size is "
                                      "not a positive int")
        sizes.append(size)
        cacheable_attr = getattr(spec, "prefix_cacheable", None)
        if not isinstance(cacheable_attr, bool):
            return None, None, None, (f"group {index} kv_cache_spec."
                                      "prefix_cacheable is not a bool")
        if cacheable_attr:
            cacheable.append(index)
    if len(sizes) > disk_layout.MAX_NUM_GROUPS:
        return None, None, None, (f"{len(sizes)} groups exceeds the cap "
                                  f"{disk_layout.MAX_NUM_GROUPS}")
    return sizes, cacheable, "kv_cache_config.kv_cache_groups[*].kv_cache_spec", None


def build_worker_fingerprint(handler) -> dict:
    """The fingerprint THIS worker's bytes belong to, or the fields it could not read.

    Never a partial answer and never a default: a missing field is reported by name
    with the paths tried, and `disk_layout.validate_fingerprint` is the final judge
    (it refuses missing AND unknown fields, so this table cannot drift wider than
    the identity the file names are built from). `disk_spill.configure` is the only
    thing that turns a result here into a naming authority.
    """
    caches = getattr(handler, "cpu_kv_caches", None)
    fingerprint: dict = {}
    sources: dict = {}
    missing: list = []
    normalized_none: list = []

    # 1. the slot width: measured from these tensors, never declared elsewhere.
    measured = disk_spill.measured_slot_bytes(caches)
    if not measured["ok"]:
        missing.append({"field": "total_bytes_per_block",
                        "tried": ["cpu_kv_caches regions: "
                                  + str(measured["reason"])]})
    else:
        fingerprint["total_bytes_per_block"] = measured["bytes_per_file"]
        sources["total_bytes_per_block"] = (
            "measured: sum(region.stride(0) * region.element_size()) over "
            f"{len(measured['regions'])} region(s)")

    # 2. this rank's id, from the process that has distributed state. The one rank
    # rule lives in worker_diag, so this calls it rather than re-deriving PP_TP.
    from . import worker_diag  # noqa: PLC0415 - the one rank rule, not two
    rank, rank_source = worker_diag.rank_id()
    if rank is None:
        missing.append({"field": "rank", "tried": [str(rank_source)]})
    else:
        fingerprint["rank"] = rank
        sources["rank"] = f"worker_diag.rank_id() ({rank_source})"

    # 3. the dtype of the bytes themselves; the config's claim is the fallback.
    dtype = None
    dtype_source = None
    if isinstance(caches, dict) and caches:
        first = caches[sorted(str(name) for name in caches)[0]]
        claimed = getattr(first, "dtype", None)
        if claimed is not None:
            dtype = str(claimed)
            dtype_source = "cpu_kv_caches region .dtype (the bytes' own dtype)"
    if dtype is None:
        value, source = _read_path(handler, ("vllm_config", "model_config",
                                             "dtype"))
        if value is not None:
            dtype = str(value)
            dtype_source = f"{source} (the config's claim, not the tensor's)"
    if dtype:
        fingerprint["dtype"] = dtype
        sources["dtype"] = dtype_source
    else:
        missing.append({"field": "dtype",
                        "tried": ["cpu_kv_caches region .dtype",
                                  "vllm_config.model_config.dtype"]})

    # 4. the group layout.
    sizes, cacheable, group_source, group_refusal = _group_fields(handler)
    if group_refusal:
        missing.append({"field": "group_block_sizes", "tried": [group_refusal]})
        missing.append({"field": "prefix_cacheable_group_ids",
                        "tried": [group_refusal]})
    else:
        fingerprint["group_block_sizes"] = sizes
        fingerprint["prefix_cacheable_group_ids"] = cacheable
        sources["group_block_sizes"] = group_source
        sources["prefix_cacheable_group_ids"] = group_source

    # 5. the NONE_HASH seed, from the fork's own accessor. It is NOT a constant:
    # vllm/v1/core/kv_cache_utils.py:147 returns the PYTHONHASHSEED-derived value
    # for a non-cryptographic hasher and DEFAULT_NONE_HASH_SEED otherwise, so a
    # digest that ignored it would call two boots of one config identical. It is
    # published in the fingerprint block (the same view `disk_spill.section()`
    # publishes) because a file name cannot be reconstructed without it; the surface
    # that carries it is dev state behind the dev token.
    seed_module = sys.modules.get(SEED_MODULE)
    getter = getattr(seed_module, "get_none_hash_seed", None) if seed_module else None
    seed_reason = None
    seed = None
    if callable(getter):
        try:
            seed = getter()
        except Exception as exc:  # noqa: BLE001 - an unreadable seed is a refusal
            seed_reason = f"get_none_hash_seed() raised {type(exc).__name__}"
        if seed is not None:
            fingerprint["none_hash_seed"] = seed
            sources["none_hash_seed"] = f"{SEED_MODULE}.get_none_hash_seed()"
    if seed is None:
        missing.append({"field": "none_hash_seed",
                        "tried": [seed_reason or f"{SEED_MODULE} is not imported "
                                  "here, or has no get_none_hash_seed()"]})

    # 6. the rank-invariant config fields, by the table above.
    for field, candidates in CONFIG_SOURCES.items():
        if len(candidates) > MAX_SOURCES_PER_FIELD:
            missing.append({"field": field,
                            "tried": [f"{len(candidates)} candidate paths; the cap "
                                      f"is {MAX_SOURCES_PER_FIELD}"]})
            continue
        for path in candidates:
            value, source = _read_path(handler, path)
            if value is None and field in NONE_REPLACEMENTS:
                fingerprint[field] = NONE_REPLACEMENTS[field]
                sources[field] = (f"{source}; the fork's value is None, rendered "
                                  "as the stable text 'none'")
                normalized_none.append(field)
                break
            if value is None:
                continue
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                fingerprint[field] = value
                sources[field] = source
                break
            fingerprint[field] = str(value)
            sources[field] = f"{source} (stringified {type(value).__name__})"
            break
        else:
            missing.append({"field": field,
                            "tried": [f"{'.'.join(path)}: unreadable"
                                      for path in candidates]})

    missing = missing[:MAX_MISSING_REPORTED]
    if missing:
        return _fail("this worker cannot read every fingerprint field, so it "
                     "cannot name a file: "
                     + ", ".join(str(item["field"]) for item in missing),
                     missing_fields=missing,
                     fingerprint_partial=fingerprint,
                     note=("the digest is built from these fields; guessing one "
                           "would name a file whose identity is a lie, so the "
                           "spill is refused and counted instead"))
    checked = disk_layout.validate_fingerprint(fingerprint)
    if not checked["ok"]:
        return _fail(f"the fingerprint this worker assembled is not a valid "
                     f"identity: {checked['reason']}",
                     missing_fields=[{"field": "validation",
                                      "tried": [str(checked["reason"])]}],
                     fingerprint_partial=fingerprint)
    digest = disk_layout.compute_geometry_digest(checked["fingerprint"])
    return _ok(fingerprint=checked["fingerprint"], geometry_digest=digest,
               sources=sources, normalized_none=normalized_none,
               total_bytes_per_block=checked["total_bytes_per_block"],
               rank=checked["fingerprint"]["rank"])


def fingerprint_report() -> dict:
    """The fingerprint's STATE, for a verb or a snapshot. Publishes no seed."""
    configured = _state["fingerprint"]
    return {"configured": configured is not None,
            "geometry_digest": _state["geometry_digest"],
            "rank": (configured or {}).get("rank"),
            "total_bytes_per_block": (configured or {}).get(
                "total_bytes_per_block"),
            "dtype": (configured or {}).get("dtype"),
            "sources": dict(_state["fingerprint_sources"]),
            # Which fields are THIS rank's own, published so a reader of a digest can
            # see why four ranks have four digests for one configuration - the exact
            # misunderstanding the first version of this contract fell into.
            "rank_local_fields": list(LOCAL_FIELDS),
            "missing_fields": list(_state["fingerprint_missing"]),
            "normalized_none": list(_state["fingerprint_normalized_none"]),
            "attempts": int(_state["fingerprint_attempts"]),
            "fingerprint": disk_layout.fingerprint_section(configured)
            if isinstance(configured, dict) else None,
            "note": ("the worker builds this itself: the slot width it measures and "
                     "the rank it resolves are the fields a single scheduler cannot "
                     "broadcast to four ranks. The seed appears here in its canonical "
                     "rendering - the SAME view `disk_spill.section()` publishes, "
                     "because a file name cannot be reconstructed without it and two "
                     "views of one identity is how a naming argument starts. The "
                     "mitigation is the surface: this is dev state, behind the dev "
                     "token, loopback-only by default - and with a non-cryptographic "
                     "hasher the seed is per-process random, not a shared secret.")}


def adopt_fingerprint(handler, force: bool = False) -> dict:
    """Build this rank's fingerprint once per boot and hand it to `disk_spill`.

    `disk_spill.configure` is the only naming authority, so a spill is only ever
    possible after this has succeeded here. Retries are rate-limited while failing
    (`FINGERPRINT_RETRY_SECONDS`), because the alternative is re-walking fifteen
    attribute paths on every bound step of a worker whose config is unreadable - and
    the answer to an unreadable config is a counted refusal, not a busy loop.
    """
    now = time.monotonic()
    if _state["fingerprint"] is not None and not force:
        return _ok(adopted=True, cached=True,
                   geometry_digest=_state["geometry_digest"],
                   reason="this rank's fingerprint is already the naming authority")
    last = _state["fingerprint_attempted_mono"]
    if (not force and isinstance(last, (int, float))
            and (now - last) < FINGERPRINT_RETRY_SECONDS):
        return _fail("the fingerprint build is rate-limited after its last failure",
                     retry_in_seconds=round(FINGERPRINT_RETRY_SECONDS - (now - last),
                                            3),
                     reason_kind="fingerprint_incomplete",
                     missing_fields=list(_state["fingerprint_missing"]))
    _state["fingerprint_attempted_mono"] = now
    _state["fingerprint_attempts"] = int(_state["fingerprint_attempts"]) + 1
    built = build_worker_fingerprint(handler)
    if not built["ok"]:
        _state["fingerprint_missing"] = list(built["missing_fields"])
        _count("fingerprint_incomplete", built["reason"])
        return _fail(built["reason"], reason_kind="fingerprint_incomplete",
                     missing_fields=list(built["missing_fields"]))
    adopted = disk_spill.configure(built["fingerprint"])
    if not adopted.get("configured"):
        _state["fingerprint_missing"] = [{"field": "disk_spill.configure",
                                          "tried": [str(adopted.get("reason"))]}]
        _count("fingerprint_incomplete", str(adopted.get("reason")))
        return _fail(f"disk_spill refused this fingerprint: "
                     f"{adopted.get('reason')}",
                     reason_kind="fingerprint_incomplete")
    _state["fingerprint"] = built["fingerprint"]
    _state["geometry_digest"] = built["geometry_digest"]
    _state["fingerprint_sources"] = built["sources"]
    _state["fingerprint_normalized_none"] = built["normalized_none"]
    _state["fingerprint_missing"] = []
    return _ok(adopted=True, cached=False,
               geometry_digest=built["geometry_digest"],
               rank=built["rank"],
               total_bytes_per_block=built["total_bytes_per_block"],
               reason="this rank's fingerprint is now the naming authority")


# --- the resolver: metadata -> (slots, group-qualified keys, cross-check) ----

def file_key(value) -> tuple:
    """The file-name key for one engine block hash: `(full key | None, why)`.

    `BlockHashWithGroupId` is BYTES - a digest plus a 4-byte big-endian group id
    (`kv_cache_utils.py:86-92`) - and the engine sends its hex. THE WHOLE VALUE IS
    THE KEY: it is normalised (stripped, lower-cased) and passed on UNTRUNCATED,
    because `disk_layout.build_slot_path` splits it into the digest and the
    `g<group>` name component. Truncating it to 16 hex characters here - which is
    what this function used to do, citing `disk_layout.validate_key16` as the
    authority - is precisely the bug: the group id sits at the END of the value, so
    the leading 16 characters are digest only, and two prefix-cacheable groups over
    one token chain then named the same file with different bytes.

    A value too short to carry the 4-byte suffix is REFUSED, not truncated: a
    16-hex key cannot be attributed to a group, so it has no identity to write
    under. An entry the engine could not fill arrives as the empty string, which
    means THAT SLOT has no identity: it is refused and counted on its own, and the
    rest of the step still spills. That is the difference between a placeholder (the
    contract's shape) and a peer bug (a length mismatch), which refuses the whole
    step.
    """
    if not isinstance(value, str):
        return None, f"a key must be a hex string, got {type(value).__name__}"
    split = disk_layout.split_block_key(value)
    if not split["ok"]:
        return None, str(split["reason"])
    return split["key_hex"], None


def resolve_store_keys(metadata) -> dict:
    """The per-slot keys for one metadata object, or a classified refusal.

    Pure: no I/O, no allocation beyond tuples, no engine import. THE ORDER IS
    PRESERVED AND THE LENGTH IS ASSERTED, because index i of `store_keys` names
    index i of `store_cpu_blocks` and nothing else can re-establish that pairing
    once the two lists are in different hands.

    Returns `{ok, slots, keys, placeholders, count, expected_digest}` - `keys` being
    the FULL group-qualified block hashes (normalised hex) with `None` for a slot the
    engine could not name, and `expected_digest` the cross-check value or None - or
    `{ok: False, reason_kind, reason}` with the numbers named.
    """
    if metadata is None:
        _count("bad_call", "no metadata object was handed in")
        return _fail("no connector metadata is bound in this worker",
                     reason_kind="bad_call")
    observe_metadata_fields(metadata)
    slots_raw = getattr(metadata, "store_cpu_blocks", None)
    keys_raw = getattr(metadata, FIELD_STORE_KEYS, None)
    if slots_raw is None:
        _count("bad_call", "the metadata has no store_cpu_blocks")
        return _fail("the metadata has no store_cpu_blocks to name",
                     reason_kind="bad_call")
    if not isinstance(slots_raw, (list, tuple)):
        _count("bad_call", f"store_cpu_blocks is {type(slots_raw).__name__}")
        return _fail(f"store_cpu_blocks is {type(slots_raw).__name__}, not a list "
                     "of slot ids", reason_kind="bad_call")
    count = len(slots_raw)
    if count == 0:
        # Not an error: a step with nothing to store. Counted, because "the tier
        # parked nothing" must be attributable rather than inferred.
        _count("no_store_slots", "this step stored no blocks, so there is nothing "
                                 "to spill")
        return _fail("this step stored no blocks", reason_kind="no_store_slots",
                     count=0)
    if keys_raw is None:
        _count("no_key_for_slot",
               f"store_keys is absent from this boot's metadata: the engine does "
               f"not carry a block hash to the worker ({count} slots unnamed)")
        return _fail(f"store_keys is absent from this boot's metadata: {count} "
                     f"stored slots have no key, so nothing can be named (the "
                     f"engine patch adds the field; see disk_store's docstring)",
                     reason_kind="no_key_for_slot", count=count, keys=0)
    if not isinstance(keys_raw, (list, tuple)):
        _count("key_count_mismatch",
               f"store_keys is {type(keys_raw).__name__}, not a list")
        return _fail(f"store_keys is {type(keys_raw).__name__}, not a list of hex "
                     "block hashes", reason_kind="key_count_mismatch",
                     count=count, keys=None)
    key_count = len(keys_raw)
    if key_count == 0:
        _count("no_key_for_slot",
               f"store_keys is empty for {count} stored slots: the engine says it "
               "has no keys for this step")
        return _fail(f"store_keys is empty while store_cpu_blocks holds {count} "
                     "slots: nothing can be named",
                     reason_kind="no_key_for_slot", count=count, keys=0)
    if key_count != count:
        # A PEER BUG, and the one place a "be liberal" reading would corrupt the
        # store: pairing 3 keys with 5 slots by truncation writes someone else's
        # bytes under the wrong name. Refuse the whole step, name both numbers.
        _count("key_count_mismatch",
               f"store_keys has {key_count} entries for {count} stored slots")
        return _fail(f"store_keys has {key_count} entries but store_cpu_blocks has "
                     f"{count}: the pairing cannot be recovered, so this step is "
                     f"refused whole rather than truncated",
                     reason_kind="key_count_mismatch", count=count, keys=key_count)
    if count > MAX_SLOTS_PER_EVENT:
        _count("over_slot_cap",
               f"one step stored {count} blocks; the cap is {MAX_SLOTS_PER_EVENT}")
        return _fail(f"this step stored {count} blocks; the cap is "
                     f"{MAX_SLOTS_PER_EVENT}, so the step is refused rather than "
                     f"partially remembered", reason_kind="over_slot_cap",
                     count=count)
    keys = []
    placeholders = 0
    for value in keys_raw:
        resolved, why = file_key(value)
        keys.append(resolved)
        if resolved is None:
            placeholders += 1
            if value != KEY_PLACEHOLDER:
                _count("bad_key_text",
                       f"a store_keys entry is not a usable block hash: {why}")
    # THE CROSS-CHECK, not the naming source. Empty (the normal case, and what a
    # half-applied patch sends) means NOTHING; a non-empty one must be a digest and
    # must equal this rank's own, or two sites computed the same field.
    expected = None
    raw_expected = getattr(metadata, FIELD_STORE_FINGERPRINT, None)
    if isinstance(raw_expected, str) and raw_expected.strip():
        checked = disk_layout.validate_geometry_digest(raw_expected)
        if not checked["ok"]:
            _count("bad_digest", str(checked["reason"]))
            return _fail(f"store_fingerprint is not a usable digest: "
                         f"{checked['reason']}", reason_kind="bad_digest",
                         count=count)
        expected = checked["value"]
    return _ok(slots=tuple(slots_raw), keys=tuple(keys),
               placeholders=placeholders, count=count, expected_digest=expected)


# --- the between-call memory --------------------------------------------------

def _remember_event(event, pairs, digest, count) -> dict:
    """Keep one step's slot/key pairs until its store event completes. Bounded."""
    existing = int(len(_retained))
    if event in _retained:
        existing -= 1                     # a replacement is not growth
    _retained[event] = {"pairs": pairs, "digest": digest, "count": count,
                        "ts": round(time.time(), 6)}
    _state["events_retained"] = int(_state["events_retained"]) + 1
    while len(_retained) > MAX_RETAINED_EVENTS:
        oldest = next(iter(_retained))
        dropped = _retained.pop(oldest)
        _state["retained_dropped"] = int(_state["retained_dropped"]) + 1
        _state["last_reason"] = (
            f"event {oldest} ({dropped['count']} slots) was evicted from retention "
            f"before its store completed: those spills are lost, and a lost spill "
            f"is only ever a missed file")
    _state["retained_now"] = int(len(_retained))
    return _ok(retained=existing + 1, event=event, count=count, digest=digest,
               reason=None)


def retained_view() -> dict:
    """The retention map's SHAPE, never its keys: bounded and hash-free."""
    return {"events": int(len(_retained)),
            "capacity": MAX_RETAINED_EVENTS,
            "slots": int(sum(int(record.get("count") or 0)
                             for record in _retained.values())),
            "slot_capacity_per_event": MAX_SLOTS_PER_EVENT,
            "dropped": int(_state["retained_dropped"]),
            "note": ("the (slot, block hash) pairs kept between "
                     "bind_connector_metadata and store completion. Only the shape "
                     "is reported: the keys are "
                     "block hashes, and a snapshot is not a place to publish them.")}


# --- the per-step path: bind, then completion --------------------------------

def observe_bind(handler, metadata) -> dict:
    """The bind entry: probe the fields, then remember the step's slot/key pairs.

    Called from the wrapper with the metadata the fork just bound, AFTER the fork's
    own body ran. Never raises; every failure is a counted dict.
    """
    _state["binds"] = int(_state["binds"]) + 1
    try:
        observe_metadata_fields(metadata)
        if not _may_touch_disk():
            return _ok(retained=False,
                       reason="the tier is not configured and armed, so nothing is "
                              "remembered and no byte will be written")
        event = _int(getattr(metadata, "store_event", None))
        if event is None or event <= INVALID_JOB_ID:
            # The NORMAL idle step: the fork only allocates a store event when it has
            # blocks to park, so a step without one is not a refusal and must not
            # inflate a refusal counter on a healthy box. It is still counted, under
            # its own name, because "the tier saw 40,000 steps and stored nothing" is
            # a fact an operator is entitled to see.
            _state["steps_idle"] = int(_state["steps_idle"]) + 1
            return _ok(retained=False,
                       reason="this step has no store event, so it has nothing to "
                              "spill (the fork allocates one only when it parks "
                              "blocks)")
        resolved = resolve_store_keys(metadata)
        if not resolved["ok"]:
            return resolved
        if resolved["placeholders"] == resolved["count"]:
            _count("no_key_for_slot",
                   f"every one of the {resolved['count']} stored slots has an empty "
                   f"block hash")
            return _fail(f"every stored slot has an empty block hash "
                         f"({resolved['count']} of {resolved['count']}): nothing can "
                         f"be named", reason_kind="no_key_for_slot",
                         count=resolved["count"])
        # This rank's own digest, built once per boot from its own measurements.
        own = adopt_fingerprint(handler)
        if not own["ok"]:
            return _fail(own["reason"], reason_kind="fingerprint_incomplete",
                         missing_fields=own.get("missing_fields"))
        digest = own["geometry_digest"]
        expected = resolved["expected_digest"]
        if expected is not None and expected != digest:
            _count("fingerprint_mismatch",
                   f"store_fingerprint says {expected} but this rank's own digest "
                   f"is {digest}")
            return _fail(f"store_fingerprint says {expected} while THIS rank's "
                         f"digest - built from its own slot width and rank id - is "
                         f"{digest}: two sites computed the same field and "
                         f"disagree, so nothing is spilled under either name",
                         reason_kind="fingerprint_mismatch",
                         scheduler_digest=expected, worker_digest=digest)
        # strict=True is a SECOND assertion of the pairing, not the first:
        # resolve_store_keys already refused unequal lengths, and if that ever
        # regressed, silently zipping to the shorter list is exactly the
        # wrong-bytes-under-the-right-name failure this module exists to prevent.
        pairs = tuple(zip(resolved["slots"], resolved["keys"], strict=True))
        named = tuple(pair for pair in pairs if pair[1] is not None)
        # THE SHADOW WITNESS SEES THE SAME BIND. Contained and default-off: it keeps
        # the GPU block ids v0.2.35 names next to these CPU slots so it can read the
        # GPU's own bytes at completion and ask whether the arena captured the right
        # ones (`disk_shadow`). It cannot change what is remembered here: the store's
        # own pairing is computed above and returned below, untouched.
        disk_shadow.observe_bind(handler, metadata)
        return _remember_event(event, named, digest, resolved["count"])
    except Exception as exc:  # noqa: BLE001 - a diagnostic never breaks a bind
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        snapshot._error(exc, "disk_store bind observation")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return _fail(f"the bind observation raised {type(exc).__name__}")


def _class_name(spec) -> str:
    """A spec's class name, bounded. Never raises: `type()` cannot, the slice cannot."""
    return type(spec).__name__[:MODE_CLASS_CHARS]


def _class_chain(spec) -> list:
    """The class names on a spec's own MRO, bounded, or [] if the reflection fails.

    The fork's own precedent for identifying a spec by its class rather than by an
    import (`gpu_census._is_circular_buffer`, `kvconfig.py:241-255`): a plugin that
    imports vllm types onto the step thread is a plugin that can fail a step. The names
    come from each class's OWN `__name__` - a spec is an instance, so its MRO entries
    are classes and `type(cls).__name__` would answer "type" for every one of them.
    """
    try:
        names = []
        for cls in type(spec).__mro__:
            name = getattr(cls, "__name__", None)
            if isinstance(name, str) and name:
                names.append(name[:MODE_CLASS_CHARS])
        return names[:MODE_MRO_CLASSES]
    except Exception:  # noqa: BLE001 - a name we cannot read is not a fault
        return []


def _sub_specs(spec) -> tuple:
    """(the per-layer specs a container aggregates, the accessor that yielded them).

    A WORKER's group entry is the AGGREGATED spec the fork projects to it: where the
    EngineCore's scheduler holds a representative per-layer spec, `kv_cache_utils.py:
    1075-1077` says the worker's config carries "an aggregated
    UniformTypeKVCacheSpecs", and `_project_kv_cache_groups_to_worker`
    (`kv_cache_utils.py:2606-2614`) rebuilds each group's spec as exactly that. The
    container declares no `mamba_cache_mode` of its own - `MambaSpec` does, at
    `kv_cache_interface.py:998` - and it publishes its layers in `kv_cache_specs` with
    a `first_spec` shortcut (`kv_cache_interface.py:1191`, `:1200-1205`). BOTH
    spellings are tried: depending on exactly one is how a panel goes dark again.

    The scan is bounded at `MODE_SUB_SPECS_SCANNED`: the population is one entry per
    LAYER, and a witness on the step thread must be able to price itself.
    """
    try:
        direct = getattr(spec, "kv_cache_specs", None)
        if isinstance(direct, dict) and direct:
            values = [item for item in list(direct.values())[:MODE_SUB_SPECS_SCANNED]
                      if item is not None]
            if values:
                return values, "kv_cache_specs"
        elif isinstance(direct, (list, tuple)) and direct:
            values = [item for item in list(direct)[:MODE_SUB_SPECS_SCANNED]
                      if item is not None]
            if values:
                return values, "kv_cache_specs"
        first = getattr(spec, "first_spec", None)
        if first is not None:
            return [first], "first_spec"
    except Exception:  # noqa: BLE001 - an unreadable container says "nothing found"
        return [], None
    return [], None


def _cache_config_mode(vllm_config) -> tuple:
    """(the engine-wide `mamba_cache_mode`, or None, and why it is None).

    The worker keeps the config it was built with (`worker.py:40`), and the fork fills
    EVERY MambaSpec's mode from this same field (`abstract.py:78`), so on this fork the
    two sources agree by construction. It is ENGINE-WIDE all the same: it can say the
    engine runs align-mode state groups, never which group is one.
    """
    try:
        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is None:
            return None, "no vllm_config.cache_config in this process"
        mode = getattr(cache_config, "mamba_cache_mode", None)
        if mode is None:
            return None, "cache_config.mamba_cache_mode is absent"
        if not isinstance(mode, str):
            return None, (f"cache_config.mamba_cache_mode is "
                          f"{type(mode).__name__}, not a str")
        return mode, None
    except Exception as exc:  # noqa: BLE001 - a config that refuses is a fact
        return None, f"reading cache_config raised {type(exc).__name__}"


def resolve_mamba_cache_mode(spec, vllm_config=None) -> dict:
    """One group's `mamba_cache_mode`, read wherever it is actually kept.

    WHY THIS EXISTS, AND WHY IT IS ONE FUNCTION. `span_shape_state()` here and
    `disk_restore_verify._boundary_groups()` both asked the GROUP's spec, and on the
    live worker both got None for all six groups - so the whole 0.2.63 span-completeness
    panel and the verifier's align half reported zeros that were structurally incapable
    of moving. Fixing one and not the other would leave the same predicate written twice
    and read twice; this is the only place in the package that asks.

    THREE SOURCES, each optional, tried in this order:
      1. the group's OWN spec field, which is the fork's answer and always wins; a
         `MambaSpec` declares it (`kv_cache_interface.py:998`) and an attention spec
         declares no such field at all, which is the fork's way of saying positional;
      2. the CONTAINER's per-layer sub-specs (`_sub_specs`), which is where the mode
         lives on a worker whose group entry aggregates seven layers into one spec;
      3. the ENGINE-WIDE `cache_config.mamba_cache_mode`, used only for a group whose
         spec class (or MRO, or sub-spec class) NAMES a state spec, because an
         engine-wide flag can say WHETHER but not WHICH. Every result from this source
         is labelled in `detail` so no reader can mistake it for per-group evidence.

    A group whose sub-specs DISAGREE is `ambiguous`, and a walk that cannot read the
    spec at all is `error`: neither is silently positional, because a guessed
    denominator would move the one ratio this feeds in whichever direction the guess
    pointed. `Exception` is contained into those two verdicts; `BaseException` is not
    caught and reaches the engine - the line `tests/test_containment.py` draws.

    Returns `{mode, source, spec_class, sub_spec_classes, detail, ambiguous, error}`,
    where `mode` is the fork's string (or the spec's own non-str value, which the
    caller reports by type) and `detail` is one line naming what was looked at.
    """
    out: dict = {"mode": None, "source": None, "spec_class": _class_name(spec),
                 "sub_spec_classes": [], "detail": "", "ambiguous": False,
                 "error": None}
    try:
        declared = getattr(spec, "mamba_cache_mode", None)
        if declared is not None:
            out.update(mode=declared, source=MODE_FROM_SPEC,
                       detail=(f"{out['spec_class']} declares mamba_cache_mode="
                               f"{declared!r}"))
            return out
        subs, accessor = _sub_specs(spec)
        classes: list = []
        modes: list = []
        for sub in subs:
            name = _class_name(sub)
            if name not in classes:
                classes.append(name)
            value = getattr(sub, "mamba_cache_mode", None)
            if isinstance(value, str) and value not in modes:
                modes.append(value)
        out["sub_spec_classes"] = classes
        engine, engine_note = _cache_config_mode(vllm_config)
        where = (f"{out['spec_class']} via {accessor} ({', '.join(classes)})"
                 if accessor else f"{out['spec_class']}, no sub-spec accessor")
        if len(modes) > 1:
            out.update(ambiguous=True, source=MODE_FROM_AMBIGUOUS,
                       detail=(f"{len(subs)} sub-specs of {where} disagree about "
                               f"mamba_cache_mode: {modes}"))
            return out
        if modes:
            out.update(mode=modes[0], source=MODE_FROM_SUB_SPECS,
                       detail=(f"{where} declare mamba_cache_mode={modes[0]!r}"))
            return out
        names = _class_chain(spec) + classes
        state_named = any(marker in name for name in names
                          for marker in STATE_CLASS_MARKERS)
        if engine is not None and state_named:
            out.update(mode=engine, source=MODE_FROM_CACHE_CONFIG,
                       detail=(f"ENGINE-WIDE cache_config.mamba_cache_mode="
                               f"{engine!r} attributed to {out['spec_class']} by its "
                               "class name, not a per-group field"))
            return out
        out["detail"] = (
            f"no mode found: {where} declared none"
            + (f"; engine-wide cache_config.mamba_cache_mode={engine!r} but no class "
               "names a state spec" if engine is not None else
               f"; engine-wide cache_config unreadable ({engine_note})"))
        return out
    except Exception as exc:  # noqa: BLE001 - a reason, never a step fault
        out["error"] = type(exc).__name__
        out["detail"] = (f"reading {out['spec_class']}.{MODE_FROM_SUB_SPECS} raised "
                         f"{type(exc).__name__}")
        return out


def _span_evidence(vllm_config, resolutions) -> str:
    """One bounded line naming WHAT THE WALK SAW, for a reason that has to explain it.

    This is the self-diagnostic half of the change: tonight the worker's spec shape had
    to be INFERRED from `page_size_bytes` ratios because nothing published it. The
    class names and the source that answered are the facts a reader needs to know
    whether the resolution is reading the real thing or a simulation of it.
    """
    classes: list = []
    for entry in resolutions:
        for name in ([entry.get("spec_class")]
                     + list(entry.get("sub_spec_classes") or ())):
            if name and name not in classes:
                classes.append(name)
    engine, engine_note = _cache_config_mode(vllm_config)
    verdict = (repr(engine) if engine is not None
               else f"unreadable ({engine_note})")
    return (f"spec classes {', '.join(classes[:MODE_EVIDENCE_CLASSES]) or 'none'}; "
            f"modes answered for {len(resolutions)} group(s); engine-wide "
            f"cache_config.mamba_cache_mode={verdict}")


def _span_reason(text: str, vllm_config, resolutions) -> str:
    """A refusal, and the evidence that explains it, in one line."""
    return f"{text}; evidence: {_span_evidence(vllm_config, resolutions)}"


def _span_ids(ids) -> str:
    """Group ids as a bounded label, for a basis that stays rank-file sized."""
    return ",".join(str(int(value)) for value in list(ids)[:8])[
        :SPAN_SHAPE_ID_CHARS]


def _span_basis(resolved: dict, from_sub_specs: list,
                from_cache_config: list) -> str:
    """WHERE the split was read from, spelled out of the evidence rather than asserted.

    The base names the fields; each suffix names the source actually used and the class
    names it read, so a reader of a rank file can tell a per-group `mamba_cache_mode`
    from a container's sub-specs from an ENGINE-WIDE flag attributed by class name -
    the three claims have different strengths and the basis is where that shows.
    """
    basis = SPAN_SHAPE_BASIS
    if from_sub_specs:
        shapes = []
        for index in from_sub_specs:
            entry = resolved[index]
            layers = (entry["sub_spec_classes"] or ["?"])[0]
            label = f"{entry['spec_class']}>{layers}"
            if label not in shapes:
                shapes.append(label)
        basis += (f"; align {_span_ids(from_sub_specs)} from sub-specs: "
                  f"{', '.join(shapes)}")
    if from_cache_config:
        basis += (f"; align {_span_ids(from_cache_config)} from engine-wide "
                  "cache_config.mamba_cache_mode (not per-group evidence)")
    return basis


def span_shape_state(handler) -> dict:
    """Which prefix-cacheable groups must be in a run, and which supply its anchor.

    READ FROM THE FORK'S OWN GROUP TABLE, the same object `_group_fields` reads for the
    fingerprint (`handler.kv_cache_config.kv_cache_groups`, kept by `worker.py:41`):
    `kv_cache_spec.prefix_cacheable` says whether a restore asks for the group at all
    (group 4, the parked QSA ring, says no), and a `mamba_cache_mode` of `"align"` says
    it is a boundary-state group. The mode is read by `resolve_mamba_cache_mode`, NOT by
    a direct `getattr` here, because on a worker the group's spec is the AGGREGATED
    container the fork projects to it and the mode lives on its sub-specs - which is how
    the whole 0.2.63 panel reported zeros that could never move (see that helper's
    docstring, and `tests/test_span_shape_spec_resolution.py`).

    EVERY failure is an answer with a reason, never an exception and never a default:
    a geometry whose shape cannot be read classifies NOTHING. A guessed denominator
    would be worse than no number - it would move the one ratio this exists to report,
    in whichever direction the guess happened to point. Every reason carries the
    evidence the walk saw (spec classes, which source answered, the engine-wide mode),
    because a panel that reads zero has to be able to say why from the rank file.
    """
    base = {"ok": False, "basis": None, "align_group_ids": [],
            "positional_group_ids": [], "prefix_cacheable_group_ids": [],
            "reason": None}
    groups = getattr(getattr(handler, "kv_cache_config", None),
                     "kv_cache_groups", None)
    if not groups:
        return {**base, "reason": "kv_cache_config.kv_cache_groups is absent or "
                                  "empty, so which group a restore must find at a "
                                  "run's end is unknown"}
    listed = list(groups)
    if len(listed) > disk_layout.MAX_NUM_GROUPS:
        return {**base, "reason": f"{len(listed)} groups exceeds the cap "
                                  f"{disk_layout.MAX_NUM_GROUPS}"}
    align: list = []
    positional: list = []
    cacheable: list = []
    resolved: dict = {}          # group index -> the resolution, for the strings below
    from_sub_specs: list = []    # align groups whose mode came from a container
    from_cache_config: list = []  # align groups CLAIMED from the engine-wide flag
    vllm_config = getattr(handler, "vllm_config", None)
    for index, group in enumerate(listed):
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            return {**base, "reason": f"group {index} has no kv_cache_spec, so its "
                                      "block shape is unknown"}
        cacheable_attr = getattr(spec, "prefix_cacheable", None)
        if not isinstance(cacheable_attr, bool):
            return {**base, "reason": f"group {index} kv_cache_spec."
                                      "prefix_cacheable is not a bool"}
        if not cacheable_attr:
            continue
        cacheable.append(index)
        entry = resolve_mamba_cache_mode(spec, vllm_config)
        resolved[index] = entry
        if entry["error"]:
            return {**base, "prefix_cacheable_group_ids": cacheable,
                    "reason": _span_reason(
                        f"group {index} kv_cache_spec.{entry['spec_class']} could not "
                        f"be read: {entry['error']}, so its block shape is unknown",
                        vllm_config, list(resolved.values()))}
        if entry["ambiguous"]:
            return {**base, "prefix_cacheable_group_ids": cacheable,
                    "reason": _span_reason(
                        f"group {index}'s sub-specs disagree about the mode, so which "
                        f"group a run must find its anchor in is unknown: "
                        f"{entry['detail']}", vllm_config, list(resolved.values()))}
        mode = entry["mode"]
        if mode is None:
            # Neither the spec, nor its layers, nor the config said "align": this is
            # not a state group HERE. An attention spec declares no mode at all, and
            # that is the fork's own way of saying positional.
            positional.append(index)
            continue
        if not isinstance(mode, str):
            return {**base, "reason": f"group {index} kv_cache_spec."
                                      "mamba_cache_mode is "
                                      f"{type(mode).__name__}, not a str"}
        if mode == MAMBA_CACHE_MODE_ALIGN:
            align.append(index)
            # THE BASIS RECORDS HOW EACH ALIGN GROUP WAS EARNED - and only for the
            # groups that were classified align. A group that resolved to "none" via
            # the container or the config is positional, and naming it in an "align
            # from ..." clause would be a wrong claim in the one field a reader trusts
            # to tell the strengths of the evidence apart.
            if entry["source"] == MODE_FROM_SUB_SPECS:
                from_sub_specs.append(index)
            elif entry["source"] == MODE_FROM_CACHE_CONFIG:
                from_cache_config.append(index)
        else:
            positional.append(index)
    if not cacheable:
        return {**base, "reason": "no prefix-cacheable group exists, so there is no "
                                  "span here for a restore to want"}
    if not align:
        # THE REASON A PANEL IS DARK, and the text a rank file carries. Kept short on
        # purpose: `span_shape_publication` clips the whole line, and the EVIDENCE at
        # its end (the spec classes, the engine-wide mode) is the part that pays for
        # the field - a long explanation that pushed the evidence past the clip would
        # have made this change's own diagnostic unreadable.
        return {**base, "prefix_cacheable_group_ids": cacheable,
                "reason": _span_reason(
                    "no prefix-cacheable group resolved mamba_cache_mode == 'align', "
                    "so the group a run must find its boundary-state anchor in is "
                    "unknown", vllm_config, list(resolved.values()))}
    if not positional:
        return {**base, "prefix_cacheable_group_ids": cacheable,
                "reason": "no prefix-cacheable group is positional, so no span could "
                          "ever be part of a contiguous run"}
    return {"ok": True, "basis": _span_basis(resolved, from_sub_specs,
                                             from_cache_config),
            "align_group_ids": align,
            "positional_group_ids": positional,
            "prefix_cacheable_group_ids": cacheable, "reason": None}


def _span_shapes_for(handler) -> dict:
    """The cached `span_shape_state()`. The group table is boot-stable; the cache is
    only to keep a witness that runs once per completed store from re-reading fifteen
    attributes each time. A FAILURE is not cached (a worker that binds before its
    config exists heals on the next completion), and the cache is keyed on the config
    object's identity, so a rebuilt config is re-read rather than believed.
    """
    config_object = getattr(handler, "kv_cache_config", None)
    cached = _span_shape_cache["shapes"]
    if (cached is not None and cached.get("ok")
            and _span_shape_cache["config"] is config_object):
        return cached
    shapes = span_shape_state(handler)
    _span_shape_cache["config"] = config_object
    _span_shape_cache["shapes"] = shapes
    return shapes


def classify_handoff(pairs, shapes) -> dict:
    """Group one hand-off's `(slot, key)` pairs into spans and classify each one.

    PURE: no file, no engine object, no state, no allocation beyond a dict bounded by
    `MAX_SLOTS_PER_EVENT` (the cap on pairs a hand-off can hold), so the loop cannot
    grow with the workload. The span's identity is `disk_layout.split_block_key`'s
    `key16` - the same digest the spill journal records, and the only position-free
    name a span has: nothing here synthesizes a key or a position, and a pair whose key
    cannot be read is IGNORED and counted, because a half-resolved identity is the one
    thing this module is not allowed to write or count.

    THE PARTITION, which is the invariant the counters depend on:
      * `incomplete_attention` - no positional group at all. Not walkable over, so it
        can never be an anchor either.
      * `complete` - every prefix-cacheable positional group AND every align group.
      * `incomplete_state` - some attention, but a shape that cannot anchor (a missing
        align group, as the walk demands ALL of them at the run's end, or a missing
        second positional group on a geometry that has one).
    `spans` counts every span classified, so `spans == complete + incomplete_state +
    incomplete_attention` holds on every call.
    """
    tally = {"spans": 0, "complete": 0, "incomplete_state": 0,
             "incomplete_attention": 0, "ignored_pairs": 0,
             "align_anchor_pairs": 0}
    if not isinstance(shapes, dict) or not shapes.get("ok"):
        return tally
    align = {int(group) for group in (shapes.get("align_group_ids") or ())}
    positional = {int(group) for group in (shapes.get("positional_group_ids") or ())}
    seen: dict = {}
    for pair in pairs or ():
        try:
            key = pair[1]
        except (TypeError, IndexError, KeyError):
            tally["ignored_pairs"] += 1
            continue
        if not isinstance(key, str):
            tally["ignored_pairs"] += 1
            continue
        split = disk_layout.split_block_key(key)
        if not split.get("ok"):
            tally["ignored_pairs"] += 1
            continue
        group = int(split["group_id"])
        if group not in align and group not in positional:
            # A parked group (the QSA ring) or one this table does not have: a file
            # no restore will ever ask for, so it is not evidence about a span.
            tally["ignored_pairs"] += 1
            continue
        entry = seen.setdefault(split["key16"], (set(), set()))
        if group in align:
            # ONE ANCHOR, counted where it is PROVEN: an align-mode file for a span is
            # the boundary state a run end is priced with (`_disk_run`'s anchor test
            # asks for exactly this file and nothing else from that group), so a pair
            # in an align group IS an anchor reaching the store. Counted per pair, not
            # per span, because a run end needs one per align group and a span with two
            # of four is two anchors and an unanchored run.
            tally["align_anchor_pairs"] += 1
        (entry[1] if group in align else entry[0]).add(group)
    for present_positional, present_align in seen.values():
        tally["spans"] += 1
        if not present_positional:
            tally["incomplete_attention"] += 1
        elif (present_positional == positional and present_align == align):
            tally["complete"] += 1
        else:
            tally["incomplete_state"] += 1
    return tally


def _note_span_shapes(tally) -> dict:
    """Add one hand-off's tally to the boot counters. Nothing else - no policy."""
    for state_key, tally_key in (("spill_spans_classified", "spans"),
                                 ("spill_spans_complete", "complete"),
                                 ("spill_incomplete_state", "incomplete_state"),
                                 ("spill_incomplete_attention",
                                  "incomplete_attention"),
                                 ("anchor_spills_completed", "align_anchor_pairs"),
                                 ("span_shape_pairs_ignored", "ignored_pairs")):
        _state[state_key] = int(_state[state_key]) + int(tally.get(tally_key) or 0)
    return {"classified": int(tally.get("spans") or 0),
            "complete": int(tally.get("complete") or 0),
            "incomplete_state": int(tally.get("incomplete_state") or 0),
            "incomplete_attention": int(tally.get("incomplete_attention") or 0)}


def _tally_span_shapes(handler, pairs) -> dict:
    """One hand-off, classified and added to the boot counters. Returns the per-hand-off
    numbers so the caller can put them in its own report.

    A geometry this module cannot read contributes NOTHING: no counter moves, and the
    `span_shapes` block carries the reason. That is the whole fail-open contract of
    this witness - a boot whose group table is unreadable reports zeros it can explain,
    never zeros that look like a healthy tier.
    """
    shapes = _span_shapes_for(handler)
    if not shapes.get("ok"):
        return {"classified": 0, "complete": 0, "incomplete_state": 0,
                "incomplete_attention": 0}
    return _note_span_shapes(classify_handoff(pairs, shapes))


def span_shape_view() -> dict:
    """The tally's surroundings: what was classified, and on what authority.

    Built on every path, so a payload from a boot that could not read its group table
    has the same keys as one that could - and `classified: 0` beside a `reason` is a
    different fact from `classified: 0` beside a healthy basis.
    """
    cached = _span_shape_cache["shapes"] or {}
    return {"basis": cached.get("basis"),
            "reason": cached.get("reason"),
            "align_group_ids": list(cached.get("align_group_ids") or ()),
            "positional_group_ids": list(cached.get("positional_group_ids") or ()),
            "prefix_cacheable_group_ids": list(
                cached.get("prefix_cacheable_group_ids") or ()),
            "classified": int(_state["spill_spans_classified"]),
            "pairs_ignored": int(_state["span_shape_pairs_ignored"]),
            "note": ("a span is COMPLETE when it holds every prefix-cacheable "
                     "positional group and every align-mode group; "
                     "`spill_incomplete_state` spans can never anchor a run end and "
                     "`spill_incomplete_attention` spans can never be walked over, so "
                     "their bytes are written and never served")}


def span_shape_publication() -> dict:
    """THE THREE RANK-FILE KEYS about the span geometry, clipped to their ceilings.

    WHY THREE KEYS AND NOT JUST THE COUNTERS: `span_shape_view()` belongs to
    `hook_section()`, which runs in the ENGINE process, where the worker's shape cache
    is empty - so the reason that explains a dark panel could never travel in the one
    payload that carried it. The rank file is the door that CAN carry it:

      * `span_shape_reason` - WHY this rank's span counters are zero, in its own words;
      * `span_shape_basis` - WHERE the geometry was read from, naming the spec classes
        it saw, so the next boot reports the worker's real shape instead of an operator
        inferring it from `page_size_bytes` ratios;
      * `span_shape_pairs_ignored` - pairs the classifier could not read, which is the
        other explanation for a tally of nothing.

    The clips are load-bearing: `worker_diag` caps ONE rank section at
    `MAX_CONTRIBUTION_BYTES`, so a reason long enough to overflow it would cost every
    counter beside it - the dark panel again, with more words.
    """
    cached = _span_shape_cache["shapes"] or {}
    reason = cached.get("reason")
    basis = cached.get("basis")
    return {"span_shape_reason": (_clip(reason, SPAN_SHAPE_REASON_CHARS - 3)
                                  if isinstance(reason, str) else None),
            "span_shape_basis": (_clip(basis, SPAN_SHAPE_BASIS_CHARS - 3)
                                 if isinstance(basis, str) else None),
            "span_shape_pairs_ignored": int(_state["span_shape_pairs_ignored"])}


def completed_snapshot(handler):
    """The keys of `_completed_store_events`, or None when we must not bother.

    Taken BEFORE the fork's `get_finished` runs: that call is what ADDS the newly
    completed indices, so the only way to see the difference is to know the before.
    None (not an empty set) means "no diff this step", which is the state whenever
    the tier cannot do work, nothing is retained, or the dict is implausibly large -
    the last case is counted rather than copied.
    """
    if not _may_touch_disk():
        return None
    if not _retained:
        return None
    completed = getattr(handler, "_completed_store_events", None)
    if not isinstance(completed, dict):
        return None
    if len(completed) > MAX_COMPLETED_SCAN:
        _state["scan_skipped"] = int(_state["scan_skipped"]) + 1
        return None
    return frozenset(completed.keys())


def on_store_completed(handler, before) -> dict:
    """Spill the slots each newly-completed store event wrote. Never raises.

    `before` is the pre-call key set from `completed_snapshot`. An event is spilled
    ONCE (`_state["handled"]`), its retention record is then dropped, and at most
    `MAX_SLOTS_PER_TICK` spill requests leave this step: `disk_spill`'s queue and
    write window are the real bounds, and this is the bound that keeps the engine
    thread's work bounded whatever they are set to.
    """
    if before is None:
        return _ok(spilled=0, reason="no completion diff was taken")
    report = {"ok": True, "reason": None, "events": [], "spilled": 0, "queued": 0,
              "spans": {"classified": 0, "complete": 0, "incomplete_state": 0,
                        "incomplete_attention": 0}}
    try:
        completed = getattr(handler, "_completed_store_events", None)
        if not isinstance(completed, dict):
            return report
        handled = dict(_state.get("handled") or {})
        new_events = [index for index in completed
                      if index not in before and str(index) not in handled]
        handed = 0
        for event in sorted(new_events, key=lambda value: str(value)):
            record = _retained.pop(event, None)
            _state["completions"] = int(_state["completions"]) + 1
            # `_remember` REPLACES the dict (the `blockdump._remember_record`
            # idiom), so the local `handled` snapshot above goes stale by design: an
            # event appears once per call, and the set exists to stop a repeat ACROSS
            # calls - the case where the fork has not yet drained
            # `_completed_store_events` into its worker metadata.
            _remember("handled", event)
            handled[str(event)] = round(time.time(), 6)
            _state["retained_now"] = int(len(_retained))
            if record is None:
                _count("unretained_event",
                       f"store event {event} completed with no retained slot/key "
                       f"record (retention held {MAX_RETAINED_EVENTS} events, or "
                       f"the tier was unarmed when it was bound)")
                report["events"].append({"event": event, "spilled": 0,
                                         "reason_kind": "unretained_event"})
                continue
            # THE COMPLETENESS TALLY, taken over the WHOLE hand-off before the tick
            # cap can truncate it: the span's shape is a fact about what the fork
            # completed, not about how much of it this step agreed to write. Contained
            # twice over - `classify_handoff` and `span_shape_state` refuse rather than
            # raise, and this guard is the belt to that braces - because a witness that
            # faults the step thread is not a witness. It changes nothing about the
            # spills below: no gate reads a tally, and an incomplete span is spilled
            # exactly like a complete one.
            try:
                tallied = _tally_span_shapes(handler, record["pairs"])
                for field, value in tallied.items():
                    report["spans"][field] = int(report["spans"][field]) + value
            except Exception as exc:  # noqa: BLE001 - a tally never faults a step
                snapshot._error(exc, "disk_store span shape")
                _state["errors"] = int(_state["errors"]) + 1
                _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            spilled_here = 0
            for slot, key in record["pairs"]:
                if handed >= MAX_SLOTS_PER_TICK:
                    remaining = len(record["pairs"]) - spilled_here
                    _state["slots_capped"] = int(_state["slots_capped"]) + remaining
                    _count("over_tick_cap",
                           f"{remaining} slots of event {event} were not offered "
                           f"this step (cap {MAX_SLOTS_PER_TICK} per step)")
                    break
                _state["slots_offered"] = int(_state["slots_offered"]) + 1
                handed += 1
                spilled_here += 1
                # THE SHADOW'S COMPARISON WINDOW. `snapshot` decides (default OFF, then
                # sampled and bounded) whether THIS store is compared, and returns a
                # plan `request_spill` fills with the bytes it captured - so the
                # comparison runs against the blob the spill decided to write, on this
                # thread, at the earliest moment it exists. `settle` never raises and
                # never changes `answer`: a spill's fate is decided entirely by the
                # spill path below.
                shadow = disk_shadow.snapshot(handler=handler, event=event, slot=slot,
                                              key=key)
                answer = disk_spill.request_spill(worker=handler, slot=slot, key=key,
                                                  shadow=shadow)
                verdict = disk_shadow.settle(shadow, answer)
                # THE GPU-DIRECT SHADOW WRITER SEES THE SAME VERDICT. Contained and
                # default-off: it is handed the plan and the verdict it just produced,
                # and writes the GPU's bytes to a SECOND tree (never this store) for the
                # blocks the verifier called `equal` (`disk_gpu_shadow`). It is a step-
                # thread function only in name - no I/O, one `put_nowait` - and it can
                # neither change `answer` nor delay this loop by a file write.
                disk_gpu_shadow.consider(shadow, verdict)
                if answer.get("ok"):
                    _state["spill_queued"] = int(_state["spill_queued"]) + 1
                    report["queued"] = int(report["queued"]) + 1
                else:
                    _state["delegated_refusals"] = \
                        int(_state["delegated_refusals"]) + 1
                    _state["last_reason"] = _clip(answer.get("reason"))
            report["spilled"] = int(report["spilled"]) + spilled_here
            report["events"].append({"event": event, "spilled": spilled_here,
                                     "slots": int(record["count"]),
                                     "digest": record["digest"]})
        return report
    except Exception as exc:  # noqa: BLE001 - a spill never breaks a step
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        snapshot._error(exc, "disk_store completion hook")
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return _fail(f"the completion hook raised {type(exc).__name__}")


# --- the spill-one relay: one EngineCore ask, answered by every rank ----------

def _pending_path(directory=None):
    directory = config.diagnostics_dir() if directory is None else directory
    if directory is None:
        return None
    return os.path.join(str(directory), PENDING_FILE)


def _next_seq() -> int:
    last = _state.get("pending_seq")
    seq = int(last) + 1 if isinstance(last, int) and not isinstance(last, bool) \
        else 1
    _state["pending_seq"] = seq
    return seq


def _record_key(record) -> str:
    """A worker's de-duplication key for one ask. `epoch` alone cannot work, because
    a worker can never learn the EngineCore's boot id (the `worker_diag` epoch
    problem, in the other direction)."""
    return (f"{record.get('epoch')}:{record.get('seq')}"
            f":{record.get('requested_ts')}")


def _read_pending_record(path) -> tuple:
    """(record | None, reason). One bounded, validated read of an untrusted file."""
    if path is None:
        return None, f"{config.ENV_DIAG_DIR} is not set: no channel to a worker"
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return None, "no spill-one request"
    except OSError as exc:  # noqa: BLE001 - unreadable is reported, never a fault
        return None, f"the request could not be stat'ed: {type(exc).__name__}"
    if size > MAX_PENDING_FILE_BYTES:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, (f"the request is {size} bytes (cap "
                      f"{MAX_PENDING_FILE_BYTES}); refusing to trust it")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_PENDING_FILE_BYTES + 1)
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a broken record is data, not a fault
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, f"the request did not parse: {type(exc).__name__}"
    _state["pending_reads"] = int(_state["pending_reads"]) + 1
    if not isinstance(document, dict):
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, "the request is not an object"
    if document.get("version") != PROTOCOL_VERSION:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, (f"the request is protocol version "
                      f"{document.get('version')!r}, this build speaks "
                      f"{PROTOCOL_VERSION}")
    if _int(document.get("seq")) is None or document["seq"] <= 0:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, "the request has no positive integer seq"
    for field in ("requested_ts", "expires_ts"):
        value = document.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
            return None, f"the request has no numeric {field}"
    if _int(document.get("slot")) is None:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, "the request names no slot"
    if not disk_layout.split_block_key(document.get("key"))["ok"]:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, ("the request's key is not a group-qualified block hash "
                      "(a digest plus the engine's 4-byte group suffix)")
    # The fingerprint is an OPTIONAL cross-check here too: empty is normal, and a
    # value that is not a digest is a lie about the naming identity.
    fingerprint = document.get("fingerprint")
    if fingerprint not in (None, "") and not \
            disk_layout.validate_geometry_digest(fingerprint)["ok"]:
        _state["pending_malformed"] = int(_state["pending_malformed"]) + 1
        return None, ("the request's fingerprint is neither empty nor 12 hex "
                      "characters")
    return document, None


def _write_receipt(rank, record, *, refused, reason, queued, path) -> dict:
    """One rank's verdict on the spill-one ask. Small, atomic, never raises."""
    report = {"written": False, "path": None, "reason": None}
    directory = config.diagnostics_dir()
    if directory is None or rank is None:
        report["reason"] = "no diagnostics directory or rank to write as"
        return report
    document = {"version": PROTOCOL_VERSION, "role": ROLE, "rank": rank,
                "writer": WRITER_TAG, "epoch": str(snapshot.epoch()),
                "ts": round(time.time(), 6),
                "request_seq": _int(record.get("seq")),
                "slot": _int(record.get("slot")),
                "key": _clip(record.get("key"), 80),
                "fingerprint": _clip(record.get("fingerprint"), 32),
                "refused": bool(refused), "reason": _clip(reason),
                "queued": bool(queued), "expected_path": _clip(path),
                "scope": ("THIS rank's answer only. The ask is rank-agnostic, so "
                          "every rank that polls it spills its OWN slot with that "
                          "index under ITS OWN digest and writes a receipt of its "
                          "own; a rank missing here did not poll in time, which is "
                          "not the same as it having no data.")}
    path_out = os.path.join(str(directory), RECEIPT_FILE.format(rank=rank))
    try:
        raw = _dump(document)
        if len(raw) > MAX_RECEIPT_FILE_BYTES:
            report["reason"] = (f"the receipt is {len(raw)} bytes (cap "
                                f"{MAX_RECEIPT_FILE_BYTES})")
            return report
        tmp = f"{path_out}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path_out)
        report.update({"written": True, "path": path_out})
        return report
    except Exception as exc:  # noqa: BLE001 - a verdict file never breaks a step
        _state["receipt_errors"] = int(_state["receipt_errors"]) + 1
        snapshot._error(exc, "disk_store receipt")
        report["reason"] = f"{type(exc).__name__}"
        return report


def fulfil_spill_one(handler, record=None) -> dict:
    """Answer one spill-one ask in THIS worker: name it, queue it, receipt it.

    The bytes are NOT written here: `disk_spill.request_spill` is a `put_nowait` and
    the writer thread does the read and the write, because a synchronous 11.4 MB
    read-plus-fsync on the worker's step thread is exactly the engine stall this
    module's rules forbid. The receipt therefore says `queued`, names THIS rank's
    digest and the file it will land in, and the operator confirms it with `op: ls`.
    """
    directory = config.diagnostics_dir()
    if directory is None:
        # A worker with no channel cannot be asked anything, and that is a counted
        # fact rather than a silent None: `no_channel` in a refusal table is how an
        # operator learns the rank never saw the ask, as opposed to refusing it.
        _count("no_channel", f"{config.ENV_DIAG_DIR} is unset, so this worker has "
                             f"no channel to read a spill-one request from")
        return {"written": False, "reason": f"{config.ENV_DIAG_DIR} is unset"}
    from . import worker_diag  # noqa: PLC0415 - one rank rule, not two
    rank, _source = worker_diag.rank_id()
    if record is None:
        return {"written": False, "reason": "no request record"}
    key = _record_key(record)
    if key in (_state.get("served") or {}):
        return {"written": True, "reason": "already served by this rank",
                "refused": False, "rank": rank}
    _state["fulfilled"] = int(_state["fulfilled"]) + 1
    refused = False
    reason = None
    queued = False
    expected = None
    try:
        if not _may_touch_disk():
            refused, reason = True, ("the disk tier is not configured and armed in "
                                     "this worker, so writing a slot is not "
                                     "permitted by the tier's own gate")
            _count("not_armed" if disk_spill.disk_root() is not None
                   else "not_configured", reason)
        else:
            own = adopt_fingerprint(handler)
            if not own["ok"]:
                refused, reason = True, own["reason"]
            elif (record.get("fingerprint") not in (None, "")
                    and record["fingerprint"] != own["geometry_digest"]):
                # THE CROSS-CHECK, and it is answered AFTER the fingerprint was
                # adopted so the message can name this rank's real digest: a refusal
                # that reads `own digest is None` on a rank that has one sends the
                # operator to the wrong machine. Adopting first also means a rank
                # that cannot name a file never queues a job for one.
                refused, reason = True, (
                    f"the request carries fingerprint {record['fingerprint']} but "
                    f"THIS rank's own digest is {own['geometry_digest']}: its "
                    f"identity was computed by two sites and disagrees, so nothing "
                    f"is written under either name")
                _count("fingerprint_mismatch", reason)
            else:
                answer = disk_spill.request_spill(worker=handler,
                                                  slot=record["slot"],
                                                  key=record["key"])
                queued = bool(answer.get("ok"))
                refused = not queued
                reason = answer.get("reason")
                # The name the writer thread WILL use. Computed here from the
                # fingerprint this rank adopted - `request_spill`'s queued answer
                # carries no path on purpose (the queue can drop the job, so a path
                # in it would be a promise), which is exactly why the receipt says
                # `queued` and points at `op: ls` for proof.
                named = disk_layout.build_slot_path(
                    disk_spill.disk_root(), record["key"], own["geometry_digest"])
                expected = named.get("path") if named.get("ok") else None
                if queued:
                    _state["last_path"] = _clip(expected)
                else:
                    _state["delegated_refusals"] = \
                        int(_state["delegated_refusals"]) + 1
    except Exception as exc:  # noqa: BLE001 - a spill-one ask never breaks a step
        _state["fulfil_errors"] = int(_state["fulfil_errors"]) + 1
        snapshot._error(exc, "disk_store spill-one fulfilment")
        refused = True
        reason = f"fulfilment raised {type(exc).__name__}"
    _remember("served", key)
    _state["last_rank"] = rank
    _state["last_reason"] = _clip(reason)
    written = _write_receipt(rank, record, refused=refused, reason=reason,
                             queued=queued, path=expected)
    return {"written": bool(written["written"]), "refused": refused, "rank": rank,
            "queued": queued, "reason": reason, "expected_path": expected,
            "receipt": written["path"], "receipt_reason": written["reason"]}


def poll_spill_one(handler) -> dict:
    """The engine-thread half of spill-one: at most one bounded control read, then
    one rank's receipt. Bounded by time, the arm, and a configured directory."""
    if not config.diagnostics_dir() or not _may_touch_disk():
        _state["poll_suppressed"] = int(_state["poll_suppressed"]) + 1
        return _ok(polled=False, reason="no diagnostics dir, or the tier is not "
                                        "configured and armed")
    now = time.monotonic()
    last = _state.get("last_poll_mono")
    if isinstance(last, (int, float)) and (now - last) < POLL_INTERVAL_SECONDS:
        return _ok(polled=False, reason="polled recently")
    _state["last_poll_mono"] = now
    _state["polls"] = int(_state["polls"]) + 1
    outcome = {"polled": True, "adopted": False, "reason": None}
    try:
        path = _pending_path()
        record, reason = _read_pending_record(path)
        outcome["reason"] = reason
        if record is not None:
            expires = float(record["expires_ts"])
            if time.time() >= expires:
                outcome["reason"] = "the request had expired"
                _state["pending_stat"] = "expired"
                _count("expired", f"seq {record['seq']} expired before this rank "
                                  f"answered it")
            else:
                key = _record_key(record)
                if key in (_state.get("served") or {}):
                    outcome["reason"] = "already served by this rank"
                    _state["pending_stat"] = "served"
                else:
                    answered = fulfil_spill_one(handler, record)
                    outcome["adopted"] = True
                    outcome["answer"] = answered
                    _state["adopted"] = int(_state["adopted"]) + 1
                    _state["pending_stat"] = ("refused" if answered["refused"]
                                              else "queued")
    except Exception as exc:  # noqa: BLE001 - a control poll never breaks a step
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        snapshot._error(exc, "disk_store spill-one poll")
        outcome["reason"] = f"poll raised {type(exc).__name__}"
    return outcome


def _op_spill_one(body: dict) -> dict:
    """Write one pending request and read the receipts that arrive in the window."""
    directory = config.diagnostics_dir()
    if directory is None:
        return {"ok": False, "reason": f"{config.ENV_DIAG_DIR} is not set, so there "
                                       f"is no channel to a worker; point it at the "
                                       f"directory the engine's rank files are in"}
    checked_key = disk_layout.split_block_key(body.get("key"))
    if not checked_key["ok"]:
        return {"ok": False, "reason": f"the key is not a usable block hash: "
                                       f"{checked_key['reason']}"}
    if not disk_spill.is_armed():
        # THE ARM IS THE ASK'S PRECONDITION, on both sides of the process boundary.
        # This runs in the EngineCore and `is_armed()` consults the arm FILE that a
        # worker also reads, so the check is honest cross-process and not a read of
        # another process's memory: writing a request that every rank must refuse
        # would leave a file a worker has to chew through for nothing.
        return {"ok": False,
                "reason": "the spill tier is not armed in this process (and the arm "
                          "file a worker reads is absent, unreadable or fenced to "
                          "another boot - never 'expired': the arm is a latch): arm "
                          "it with op:arm first, because a request every rank must "
                          "refuse is not a diagnostic, it is litter"}
    slot = _int(body.get("slot"))
    if slot is None or slot < 0:
        return {"ok": False, "reason": "slot must be a non-negative CPU slot index"}
    # The fingerprint is OPTIONAL and a cross-check only: the empty string is the
    # normal value (the engine sends "") and means "name it with whatever each rank
    # computes for itself", which is exactly the corrected contract.
    fingerprint = body.get("fingerprint", "")
    fingerprint = "" if fingerprint is None else str(fingerprint).strip()
    if fingerprint:
        checked_digest = disk_layout.validate_geometry_digest(fingerprint)
        if not checked_digest["ok"]:
            return {"ok": False,
                    "reason": f"the optional fingerprint is not 12 hex characters: "
                              f"{checked_digest['reason']}"}
        fingerprint = checked_digest["value"]
    try:
        wait_s = float(body.get("wait_s") or 3.0)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "wait_s must be a number"}
    wait_s = max(0.0, min(30.0, wait_s))
    seq = _next_seq()
    now = time.time()
    record = {"version": PROTOCOL_VERSION, "role": ROLE, "op": "spill-one",
              "seq": seq, "epoch": str(snapshot.epoch()),
              "requested_ts": round(now, 6),
              "expires_ts": round(now + REQUEST_TTL_SECONDS, 6),
              "slot": slot, "key": checked_key["key_hex"],
              "fingerprint": fingerprint,
              "writer": WRITER_TAG,
              "scope": ("rank-agnostic: every worker that polls this answers it for "
                        "ITS OWN slot with that index, under ITS OWN digest, and "
                        "writes a receipt of its own. A rank missing from the "
                        "receipts did not poll in time, which is NOT the same as it "
                        "having no data."),
              "note": ("the spill is QUEUED on each rank's disk_spill writer thread, "
                       "not completed synchronously: confirm with op:ls")}
    path = _pending_path(directory)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        raw = _dump(record)
        if len(raw) > MAX_PENDING_FILE_BYTES:
            return {"ok": False, "reason": f"the request is {len(raw)} bytes (cap "
                                           f"{MAX_PENDING_FILE_BYTES})"}
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - reported, never raised at a caller
        snapshot._error(exc, "disk_store spill-one request")
        for candidate in (tmp, path):
            # The half-written request must not survive to be adopted by a worker:
            # a request the writer already gave up on, answered, is worse than none.
            with contextlib.suppress(OSError):
                os.unlink(candidate)
        return {"ok": False, "reason": f"the request could not be written: "
                                       f"{type(exc).__name__}"}
    _state["requests_written"] = int(_state["requests_written"]) + 1
    expected, expected_source = _expected_ranks()
    deadline = time.monotonic() + wait_s
    receipts = _read_receipts(directory, seq)
    while time.monotonic() < deadline:
        receipts = _read_receipts(directory, seq)
        # With no handshake recorded, completeness is NOT claimed: the window is
        # waited out in full rather than breaking at the first receipt (or, worse,
        # at an empty expected set, which would mean "answered" before anyone spoke).
        if expected and len(receipts) >= len(expected):
            break
        time.sleep(0.05)
    return {"ok": True, "reason": f"the request was written and the receipts read "
                                  f"for {round(wait_s, 1)}s; a spill is QUEUED on "
                                  f"each rank's writer thread, so confirm the file "
                                  f"with op:ls",
            "seq": seq, "path": path,
            "expected_ranks": expected, "expected_ranks_source": expected_source,
            "ranks_answered": sorted(receipts),
            "completeness": ("unknown" if not expected
                             else ("complete" if len(receipts) >= len(expected)
                                   else "partial")),
            "receipts": receipts}


def _expected_ranks() -> tuple:
    """(rank ids | None, how that is known). THE ONE RULE LIVES IN `blockdump`.

    The EngineCore is the process that must say who did not answer, and it is also
    the process that cannot resolve a rank itself (no `init_process_group`) - the
    rank SET is only knowable from the capacity handshake. `blockdump` already owns
    that reading, so this calls it instead of writing a second rule that can drift:
    two answers for "who should have answered" is how a completeness claim becomes
    a coin flip.
    """
    from . import blockdump  # noqa: PLC0415 - the handshake's rank set, one rule
    try:
        return blockdump._expected_ranks()
    except Exception as exc:  # noqa: BLE001 - a wait budget is never a verb failure
        return None, f"the rank set was unreadable: {type(exc).__name__}"


def _read_receipt_file(path):
    try:
        if os.path.getsize(path) > MAX_RECEIPT_FILE_BYTES:
            return None
        with open(path, "rb") as handle:
            return json.loads(handle.read(MAX_RECEIPT_FILE_BYTES + 1).decode("utf-8"))
    except Exception:  # noqa: BLE001 - a bad receipt is data, not a fault
        return None


def _read_receipts(directory, seq: int) -> dict:
    """The ranks that answered THIS seq, keyed by rank. An unreadable directory is
    NO receipts and a remembered reason, never a fake `error` entry: this dict's
    LENGTH is compared against the expected rank set, so a phantom key could turn an
    I/O failure into a "complete" claim. The caller reports the wait window; the
    reason stays in `_state["last_reason"]` for `op:status` to surface.
    """
    out = {}
    try:
        names = sorted(os.listdir(str(directory)))
    except OSError as exc:
        _state["last_reason"] = _clip(
            f"the receipts could not be listed: {type(exc).__name__}")
        return {}
    for name in names[:MAX_RANK_FILES]:
        if not (name.startswith(RECEIPT_PREFIX) and name.endswith(FILE_SUFFIX)):
            continue
        document = _read_receipt_file(os.path.join(str(directory), name))
        if not document or document.get("request_seq") != seq:
            continue
        rank = document.get("rank")
        if not isinstance(rank, str) or not rank:
            _state["last_reason"] = _clip(
                f"a receipt ({name}) named no rank, so it is not counted as an "
                f"answer")
            continue
        out[rank] = {
            "refused": bool(document.get("refused")),
            "queued": bool(document.get("queued")),
            "reason": document.get("reason"),
            "expected_path": document.get("expected_path"),
            "digest": document.get("fingerprint")}
    return out


# --- the verb ------------------------------------------------------------------

def _dump(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def _trim(response: dict, budget: int | None = None) -> dict:
    """Guarantee a response cannot grow past its budget.

    The listing is the only unbounded part of any answer, so it is what gets eaten,
    from the tail, and the payload SAYS it was trimmed: a truncated `ls` that looks
    complete is worse than a short one, because the operator draws conclusions about
    what the store holds from the rows that are missing.

    The budget is read at CALL time, not bound as a default: a constant captured in
    the signature cannot be moved by anything, including the test that exists to
    prove the ceiling holds.
    """
    budget = MAX_RESPONSE_BYTES if budget is None else budget
    raw = _dump(response)
    if len(raw) <= budget:
        return response
    listing = response.get("files")
    while listing and len(_dump(response)) > budget:
        listing.pop()
        response["truncated"] = True
        response["count"] = len(listing)
        response["note"] = (f"the response exceeded {budget} bytes and was trimmed "
                            f"from {len(raw)} bytes")
        # The op's own sentence may claim "3 of 3 file(s)"; after a trim it would be
        # a completeness claim about rows that are no longer in the payload, so the
        # sentence is marked rather than left to mislead.
        if isinstance(response.get("reason"), str):
            response["reason"] = response["reason"] + " (TRIMMED)"
    if len(_dump(response)) > budget:
        response = {"ok": False, "reason": "the response would exceed "
                                           f"{budget} bytes"}
    return response


def _bounded_int(value, default, cap, floor=1) -> int:
    try:
        number = int(value) if value is not None else default
    except (TypeError, ValueError):
        number = default
    return max(floor, min(cap, number))


def _validate_shard(shard) -> tuple:
    """`(shard | None, why)`: the two-hex-char directory prefix, or a refusal.

    The LENGTH and the ALPHABET come from `disk_layout` rather than being spelled out
    here, because a shard is a piece of the file name and a second definition of one
    is how a listing ends up looking in a directory no writer writes to.
    """
    if not isinstance(shard, str):
        return None, f"a shard must be text, got {type(shard).__name__}"
    text = shard.strip().lower()
    if len(text) != disk_layout.SHARD_CHARS:
        return None, (f"a shard is {disk_layout.SHARD_CHARS} hex chars; "
                      f"{shard!r} is {len(text)}")
    if any(char not in disk_layout.HEX_DIGITS for char in text):
        return None, "a shard is hexadecimal (it is the head of a block hash)"
    return text, None


def _op_ls(body: dict) -> dict:
    """The listing. Read-only, bounded in files, dirs, entries, and bytes.

    The shard filter is applied to the walk's entries rather than pushed into
    `_walk_store`: the walk has no shard argument (the capacity check always wants the
    whole store) and inventing one would mean editing the module that owns the caps
    for a convenience of this verb. The caps bound the walk either way.

    A store directory that does not exist is an EMPTY listing, not a failure: the
    tier creates nothing until its first write (that is what "off means off" costs the
    filesystem), so a fresh boot - or a booted tier that has never seen pressure - has
    no directory, and `ls` saying so is a normal answer, not an error one.
    """
    shard = body.get("shard")
    if shard not in (None, ""):
        checked_shard, why = _validate_shard(shard)
        if checked_shard is None:
            return {"ok": False, "reason": f"the shard is not usable: {why}"}
        shard = checked_shard
    limit = _bounded_int(body.get("limit"), MAX_LS_LIMIT_DEFAULT, MAX_LS_FILES)
    root = disk_spill.disk_root()
    if root is None:
        return {"ok": True, "reason": f"{config.ENV_DISK_ROOT} is not set, so there "
                                      f"is nothing to list", "count": 0,
                "files": [], "root": None, "read_only": True}
    if not os.path.isdir(root):
        return {"ok": True,
                "reason": (f"no store exists at {root} yet: nothing has been spilled "
                           f"by this tier, which creates no directory until its "
                           f"first write"),
                "count": 0, "files": [], "root": root, "shard": shard or "all",
                "read_only": True,
                "ceiling_bytes": disk_spill.capacity_ceiling_bytes(),
                "units": disk_layout._units("ceiling_bytes")}
    walk = disk_spill._walk_store(root, max_files=disk_spill.MAX_WALK_FILES,
                                  max_shard_dirs=disk_spill.MAX_SHARD_DIRS)
    if not walk["ok"]:
        return {"ok": False, "root": root,
                "reason": f"the store could not be measured: {walk['reason']}",
                "read_only": True}
    entries = list(walk["entries"])
    if shard not in (None, ""):
        entries = [row for row in entries
                   if os.path.basename(os.path.dirname(str(row["path"])))
                   == str(shard).lower()]
    entries.sort(key=lambda row: (-int(row.get("mtime_ns") or 0),
                                  str(row.get("filename"))))
    files = [{"name": str(row["filename"])[-MAX_FILENAME_DISPLAY:],
              "bytes": int(row["size_bytes"]),
              "mtime_ns": int(row["mtime_ns"])} for row in entries[:limit]]
    return {"ok": True,
            "reason": (f"{len(files)} of {len(entries)} file(s); the store holds "
                       f"{int(walk['used_bytes'])} bytes measured over "
                       f"{int(walk['files'])} file(s)"),
            "count": len(files), "files": files, "root": root,
            "shard": shard or "all", "walked": int(walk["files"]),
            "walk_used_bytes": int(walk["used_bytes"]),
            "ceiling_bytes": disk_spill.capacity_ceiling_bytes(),
            "walk_capped": bool(walk["truncated"]), "limit": limit,
            "truncated": len(files) < len(entries) or bool(walk["truncated"]),
            "read_only": True,
            "units": disk_layout._units("walk_used_bytes", "ceiling_bytes",
                                        "mtime_ns")}


def contains_key_problem(key) -> str | None:
    """Why this `key` cannot be asked about, or None when it can.

    ONE definition, used by the ROUTE (to answer a bad key with a `400` before the
    engine is touched) and by the VERB (to answer it as data when something calls the
    engine-side function directly). "You asked badly" and "it is not there" must never
    share a response: a caller that reads `present: false` for a malformed key goes
    looking for a missing spill instead of a missing eight hex characters.

    The parsing is `disk_layout.split_block_key`'s, so there is no second rule for what
    a key is: 16 hex of digest plus 8 hex of big-endian group id, at least
    `disk_layout.MIN_KEY_HEX_CHARS` characters.
    """
    if key is None or key == "":
        return ('op=contains needs a "key": the engine\'s GROUP-QUALIFIED block hash '
                f'(the {disk_layout.KEY_DIGEST_CHARS}-hex key16 plus the group id as '
                f"{disk_layout.GROUP_SUFFIX_HEX_CHARS} hex chars, so at least "
                f"{disk_layout.MIN_KEY_HEX_CHARS} hex characters)")
    split = disk_layout.split_block_key(key)
    if not split["ok"]:
        return f"the key cannot name an identity: {split['reason']}"
    return None


def _op_contains(body: dict) -> dict:
    """"Is this identity on the disk, and for which ranks?" - answered from the index.

    THE QUESTION A STORE SIZE CANNOT ANSWER. Before this verb the only evidence that a
    span had been parked was "the store grew engine-wide", which attributes nothing to
    the span you asked about. `disk_index.membership()` answers it from the replayed
    journals, in memory, per RANK.

    It deliberately does NOT walk the store: `ls` already does that (and takes the walk
    caps with it), and a membership test that scanned every file would turn a polled
    endpoint into an I/O storm over a 2,720-file shard set. It also does not move the
    witness's `hits`/`misses` - see the docstring there.

    The one thing this CANNOT say is the geometry digest each rank named its file with:
    the journal line is `<seq> <+|-> <key16> <bytes> <group>` and carries no digest.
    Naming the actual files is `op=ls` with a `shard`, and the answer says so.
    """
    key = body.get("key")
    problem = contains_key_problem(key)
    if problem is not None:
        return {"ok": False, "op": "contains", "read_only": True,
                "reason": problem, "present": None,
                "verdict": disk_index.VERDICT_KEY_REJECTED,
                "key": str(key)[:MAX_STRING_CHARS] if key is not None else None}
    answer = disk_index.membership(key)
    answer["read_only"] = True
    answer["key"] = str(key)[:MAX_STRING_CHARS]
    answer["next_step"] = (
        "the files this identity names are "
        "kvslot-<key16>-g<group>-<geodigest12>.bin, ONE PER RANK; the journal "
        "carries no digest, so name them with {\"op\":\"ls\","
        "\"shard\":\"<shard>\"} (it walks the store, this verb does not). Compare "
        "ranks_present with restore.digests_expected in this same answer: fewer "
        "ranks than the boot has armed means a partially-written span.")
    return answer


# --- the cross-process rollup: what the RANKS' spill counters saw --------------
#
# WHY THIS EXISTS (the 2026-10-06 reclaim, read out of a payload that said 0).
# `disk_spill._state` is per-process. The spill writer thread and `_enforce_ceiling()`
# run in the process that COMMITS WRITES - a worker - and on this boot there are four
# of them (PP0_TP0, PP0_TP1, PP1_TP0, PP1_TP1) plus the scheduler. The `/disk` door is
# answered by whichever process the utility call landed on, and that process ran no
# walk. So the live reclaim of 395.8 GiB / 34,896 files down to 161.5 GiB / 14,233
# files - 234 GiB shed in one night - was reported by `op:status` as
# `evictions: 0, evicted_bytes: 0, ceiling_checks: 0, thread_alive: false`, while
# `rank-PP0_TP0.json` was carrying `disk.evictions: 15237`. The whitelist that
# publishes `ceiling_checks` and the `walk_*` fields is correct and unreadable in
# practice: the publishing process is the wrong one, and a witness nobody can read is
# not a witness.
#
# WHAT THIS IS NOT. It is not a shared counter and it does not become one: four
# processes with four `evictions` counters, summed, is FOUR FACTS, not one fact times
# four. There is no shared memory and no new IPC channel here (both far riskier than
# reading files the ranks already write), and `spill` above stays this process's
# verbatim counters - this adds a VIEW and changes no policy.
#
# THE ONE RULE THAT MAKES IT SAFE TO READ. A rank file is UNTRUSTED INPUT: another
# process wrote it, it can be half-written, truncated by a crash, or a leftover from a
# boot that is gone. Every file is size-capped, type-checked and per-field bounded, and
# anything that is not fresh-and-usable is NAMED and EXCLUDED rather than folded into a
# total. A total that cannot account for every rank the boot has is published as a
# LOWER BOUND with the ranks it is missing named - the same rule that stopped a
# truncated ceiling walk reporting its floor as a measurement.

# (published name, WHERE in the rank document to read it, unit). `files_written` is the
# worker's name for `disk_spill`'s `spilled` counter - see `worker_report`, which is the
# writer of that key - and the read-path counters live one section DOWN, in
# `disk_restore.readers`, because `disk_read`'s state is reported by the restore layer's
# rank contribution (it is the layer that owns the reader threads and the LRU touch).
# A source is therefore a PATH of keys into the rank document, so one reader serves both
# sections and no field gets a bespoke read that skips the bounds.
RANK_ROLLUP_FIELDS = (("evictions", ("disk", "evictions"), "files"),
                      ("spilled", ("disk", "files_written"), "files"),
                      ("spill_queued", ("disk", "spill_queued"), "requests"),
                      ("spill_errors", ("disk", "spill_errors"), "writes"),
                      ("bytes_written", ("disk", "bytes_written"), "bytes"),
                      ("queue_depth", ("disk", "queue_depth"),
                       "requests waiting NOW"),
                      # THE 0.2.30 ADDITIONS, all of them fields the operator asked to
                      # be able to see without opening four files: what the ceiling pass
                      # did (`journal_removals` is the journalled half of an eviction,
                      # `drops` the spills that never became files) and whether the LRU
                      # clock is actually moving (`recency_promotions`).
                      ("journal_removals", ("disk", "journal_removals"), "files"),
                      ("drops", ("disk", "drops"), "jobs"),
                      # THE COMPLETENESS OF THE SPILLS (0.2.63). Three classes of span
                      # and their denominator, summed across the ranks that wrote them.
                      # The orphan rate is `spill_incomplete_state / classified` - the
                      # spans whose bytes are on the SSD and can never anchor a run end
                      # - and it is published as counts, not a percentage, because four
                      # ranks with different volumes do not average into one rate.
                      ("spill_spans_classified",
                       ("disk", "spill_spans_classified"), "spans"),
                      ("spill_spans_complete",
                       ("disk", "spill_spans_complete"), "spans"),
                      ("spill_incomplete_state",
                       ("disk", "spill_incomplete_state"), "spans"),
                      ("spill_incomplete_attention",
                       ("disk", "spill_incomplete_attention"), "spans"),
                      # THE STORE-END OF THE RUN-END ANCHOR PIPE (0.2.64), summed
                      # across the ranks that wrote them: boundary-state files that
                      # reached the store. Compare with the scheduler's
                      # `mamba_anchor.anchor_offers_total` (published by
                      # `disk_store.section()`, not by a rank file - a worker never
                      # runs the mask), and read the pair as the two ends of one pipe
                      # rather than as a per-block cause and its effect.
                      ("anchor_spills_completed",
                       ("disk", "anchor_spills_completed"), "anchor files"),
                      # THE PAIRS THE CLASSIFIER COULD NOT READ (no group suffix, not a
                      # key, a group the table does not have). Counted since 0.2.63 and
                      # published in `span_shapes` only, which is the engine process's
                      # view of ITS OWN empty cache - so a fleet feeding the classifier
                      # garbage looked identical to a fleet with clean pairs.
                      ("span_shape_pairs_ignored",
                       ("disk", "span_shape_pairs_ignored"), "pairs"),
                      ("recency_promotions",
                       ("disk_restore", "readers", "recency_promotions"), "reads"),
                      ("recency_promotions_throttled",
                       ("disk_restore", "readers",
                        "recency_promotions_throttled"), "reads"),
                      # THE CROSS-RANK WITNESS (0.2.34). `peer_files_removed` summed
                      # across ranks is "how many files left this store under a
                      # digest that was NOT the remover's" - the question the journal
                      # cannot answer from a `-` line, and the one that decides
                      # whether an eviction storm is LRU behaviour or four ranks
                      # deleting each other's live restores. `passes_short` is the
                      # other half: a rank that came up SHORT because it withheld
                      # what it could not attribute.
                      ("peer_files_removed",
                       ("disk", "peer_eviction", "peer_files_removed"), "files"),
                      ("peer_files_removed_known_peer",
                       ("disk", "peer_eviction",
                        "peer_files_removed_known_peer"), "files"),
                      ("unattributable_entries_excluded",
                       ("disk", "peer_eviction",
                        "unattributable_entries_excluded"), "files"),
                      ("peer_digests_unconfirmed",
                       ("disk", "peer_eviction", "peer_digests_unconfirmed"),
                       "passes"),
                      ("ceiling_passes_short",
                       ("disk", "peer_eviction", "passes_short"), "passes"))

RANK_ROLLUP_MEANINGS = {
    "evictions": "files THIS RANK's ceiling pass removed from the store (monotonic "
                 "since that rank's boot); the reclaim's real size is the sum here, "
                 "not `spill.evictions` in the process answering the call",
    "spilled": "files this rank wrote (the rank file's `files_written`, which is "
               "disk_spill's `spilled` counter)",
    "spill_queued": "requests this rank handed to its writer thread",
    "spill_errors": "writes this rank's thread failed",
    "bytes_written": "bytes this rank wrote since its boot - NOT the store's current "
                     "size: an eviction removes bytes this counter never subtracts. "
                     "For how full the store IS, read `disk_fill.used_bytes`, which is "
                     "the ranks' measured estimate, not this write total",
    "queue_depth": "requests waiting in this rank's queue RIGHT NOW (a gauge, summed "
                   "as a fleet-wide instantaneous depth)",
    "journal_removals": "removal lines this rank wrote to the restore journal, "
                        "i.e. the files its ceiling pass deleted AND told the index "
                        "about; it tracks `evictions` and lags it only when the "
                        "journal refused",
    "drops": "spills this rank's tier REFUSED to queue (queue full, byte cap, arm "
             "off): a spill that never became a file, so `spilled` alone cannot see "
             "it",
    "spill_spans_classified": "spans this hand-off layer could classify (every "
                              "attention spill, every align-state spill, counted once "
                              "per store hand-off): the DENOMINATOR of the two rates "
                              "beside it, never a rate on its own",
    "spill_spans_complete": "spans that arrived with EVERY group a restore asks for - "
                            "each one is a span that can be walked over and can end a "
                            "run",
    "spill_incomplete_state": "spans that arrived with the attention block and NOT "
                              "with the boundary-state anchor every align-mode group "
                              "needs at the run's end. Their bytes are written and "
                              "counted against the ceiling, and no offer can use them: "
                              "this is the `no_boundary_anchor` refusal, seen at spill "
                              "time instead of at the lookup that fails. Summed, "
                              "divided by `spill_spans_classified`, this is the "
                              "orphan rate the journals measured at 20.0%",
    "spill_incomplete_attention": "spans that arrived with align-state groups and NO "
                                  "positional file, so no contiguous run can include "
                                  "them at all - the mirror of the orphan above, and "
                                  "equally unservable",
    "span_shape_pairs_ignored": "(slot, key) pairs this rank's classifier could not "
                                "read - no group suffix on the key, a key that is not "
                                "a string, or a group the group table does not list - "
                                "and therefore IGNORED, never a span of their own. A "
                                "zero-span tally beside a non-zero count here is a "
                                "CLASSIFIER BEING FED SOMETHING ELSE, not an idle "
                                "tier; a zero beside a `span_shape_reason` is a "
                                "geometry this build could not read. Added because the "
                                "count lived in `_state` and in the engine's own view "
                                "and in NO cross-rank door, so the one process that "
                                "saw it was the one nobody polled",
    "anchor_spills_completed": "boundary-state ANCHOR files this rank's hand-offs "
                              "wrote: one per align-mode pair, which is the block a "
                              "run end is priced with (`_disk_run` asks that group for "
                              "nothing else). Summed, this is how many anchors the "
                              "fleet actually stored - the store-side half of the "
                              "0.2.64 mask patch, to be read beside the scheduler's "
                              "`mamba_anchor.anchor_offers_total`. It is NOT filtered "
                              "by whether the patch marked the individual block: a "
                              "hand-off carries `(slot, block-hash)` pairs only, so no "
                              "per-block join exists, and the patch's effect is the "
                              "rate this moves at against `spill_incomplete_state` "
                              "when the env flag flips",
    "recency_promotions": "reads that moved a file's eviction clock "
                          "(disk_read.promote_recency): if this is not moving, the "
                          "tier is running a write-ordered LRU and shedding what it "
                          "keeps reading",
    "recency_promotions_throttled": "promotions skipped by the per-file throttle - the "
                                    "count that says promotion is working and bounded, "
                                    "as opposed to absent",
    "peer_files_removed": "files this rank unlinked whose name carries a geometry "
                          "digest that is NOT its own (0.2.34). Summed, this is the "
                          "cross-rank removal count the journal cannot give: it "
                          "includes this rank's OWN obsolete geometry, so read it "
                          "with the field below, and read a rising total beside "
                          "`ceiling_passes_short` as the peer witness failing rather "
                          "than as LRU behaviour",
    "peer_files_removed_known_peer": "the unambiguous half: removals of a file whose "
                                     "digest belonged to a rank this walk believed "
                                     "LIVE at the moment of deletion. Summed across "
                                     "ranks, a non-zero total IS the cross-rank "
                                     "deletion signature - the number that used to "
                                     "mean reading 15,848 journal lines and "
                                     "comparing block sizes by hand",
    "unattributable_entries_excluded": "files a rank WITHHELD from its destructive "
                                       "set because it could not attribute them - no "
                                       "peer digest confirmed, a foreign digest on a "
                                       "committed file. This is the fail-closed gate "
                                       "doing its job, and the number that says how "
                                       "much capacity the fleet declined to buy",
    "peer_digests_unconfirmed": "ceiling decisions this rank took with NO live peer "
                                "digest - the state that used to mean 'shed the "
                                "foreign files first' and now means 'withhold them "
                                "and come up short'",
    "ceiling_passes_short": "passes this rank finished SHORT of its target, by "
                            "design, rather than unlink a file it could not prove "
                            "was its own to delete; the ceiling is expected to be "
                            "exceeded while this rises",
}
RANK_ROLLUP_GAUGES = ("queue_depth",)
"""Fields that are INSTANTANEOUS, not monotonic-since-boot. Summing a gauge across
ranks answers "how much is waiting across the fleet right now", which is a different
kind of number from a counter, so it gets its own label rather than a footnote."""


def _rank_rollup_int(value):
    """A counter from another process, or None. Bounded, and never a guess."""
    number = _int(value)
    if number is None or number < 0 or number > RANK_ROLLUP_COUNTER_MAX:
        return None
    return number



def _dig(document, path):
    """One value out of a rank document by KEY PATH, or None. Never raises.

    The rank file is UNTRUSTED input written by another process: every level has to
    be a mapping before it can be indexed, and a `disk_restore` that turned out to be
       a string is an absent field rather than a TypeError on a polled path.
    """
    node = document
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _rank_file_rollup_row(path, name: str) -> tuple:
    """What ONE rank's file says its process did: (row | None, status, detail).

    `status` is the verdict a reader sees by name: `counted` (fresh and usable),
    `stale`, `too_large`, `malformed`, `bad_version`, `mislabeled`, `no_ts`,
    `clock_ahead`, `no_disk_section`, `unreadable`. Only `counted` reaches a total; a
    `stale` row still carries its last-known numbers so a reader can see what the rank
    was doing before it stopped writing - under its own name, never inside a total.
    """
    from . import worker_diag  # noqa: PLC0415 - one rank-shape rule, not two
    try:
        size = os.path.getsize(path)
    except OSError as exc:  # noqa: BLE001 - it vanished between listing and read
        return None, "unreadable", f"stat failed: {type(exc).__name__}"
    if size > MAX_RANK_ROLLUP_FILE_BYTES:
        return None, "too_large", (f"{size} bytes (cap "
                                   f"{MAX_RANK_ROLLUP_FILE_BYTES}); not read")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_RANK_ROLLUP_FILE_BYTES + 1)
        document = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - a half-written file is not a fault
        return None, "malformed", f"did not parse: {type(exc).__name__}"
    if not isinstance(document, dict):
        return None, "malformed", f"top level is {type(document).__name__}"
    version = document.get("version")
    if version != worker_diag.SCHEMA_VERSION:
        return None, "bad_version", (f"schema {version!r}, this build reads "
                                     f"{worker_diag.SCHEMA_VERSION}")
    rank = document.get("rank")
    claimed = name[len(RANK_FILE_PREFIX):-len(RANK_FILE_SUFFIX)]
    if not isinstance(rank, str) or not worker_diag._valid_rank(rank):
        return None, "mislabeled", f"its own `rank` field is {rank!r}"
    if rank != claimed:
        return None, "mislabeled", (f"the file says {rank!r} and the name says "
                                    f"{claimed!r}")
    stamp = document.get("ts")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return None, "no_ts", "`ts` is absent or not numeric, so age is unknowable"
    age = time.time() - float(stamp)
    if age < 0.0:
        # A CLOCK AHEAD OF OURS is not freshness (disk_restore applies the same
        # one-sided rule to its proof): a file stamped an hour in the future would
        # sail past the staleness test forever, and rank files cross a host boundary.
        return None, "clock_ahead", f"stamped {-age:.1f}s in the future"
    section = document.get("disk")
    if not isinstance(section, dict):
        return None, "no_disk_section", ("no `disk` section: that rank's worker never "
                                         "registered the spill tier's report")
    epoch = document.get("epoch")
    row = {"rank": rank, "file": _clip(name, MAX_FILENAME_DISPLAY),
           "age_seconds": round(age, 3),
           "epoch": _clip(epoch, 40) if isinstance(epoch, str) else None}
    rejected = []
    for field, source, _unit in RANK_ROLLUP_FIELDS:
        # A SOURCE IS A KEY PATH now (`disk_restore.readers` holds the read-side
        # counters), so one walk of the untrusted document serves both sections under
        # the same bounds.
        raw_value = _dig(document, source)
        value = _rank_rollup_int(raw_value)
        row[field] = value
        if value is None and raw_value is not None:
            rejected.append(field)
    alive = section.get("writer_thread_alive")
    row["writer_thread_alive"] = alive if isinstance(alive, bool) else None
    row["ceiling_bytes"] = _rank_rollup_int(section.get("ceiling_bytes"))
    # THE FAIR SHARE, AS THAT RANK SAW IT (0.2.30). NOT summed - a share is a budget,
    # and summing four ranks' budgets re-invents the number that over-evicted the store.
    # Published per rank so "four ranks, one did the work, three declined" is one glance
    # rather than four files opened by hand.
    row["ceiling_share_bytes"] = _rank_rollup_int(
        section.get("ceiling_share_bytes"))
    # THE SAME SHARE, COUNTED IN BLOCKS (0.2.58). NOT summed, for the same reason the
    # byte share is not: it is a budget. Read across the four rows it is the equality a
    # restore depends on - a block is only servable when it is present on EVERY rank, so
    # four equal `ceiling_share_bytes` over four UNEQUAL `ceiling_share_blocks` is the
    # signature of a tier withholding capacity it cannot serve.
    row["ceiling_share_blocks"] = _rank_rollup_int(
        section.get("ceiling_share_blocks"))
    basis_share = section.get("share_basis")
    row["share_basis"] = (_clip(basis_share, 24)
                          if isinstance(basis_share, str) else None)
    row["share_divisor"] = _rank_rollup_int(section.get("ceiling_divisor"))
    row["declined_because_already_under"] = _rank_rollup_int(
        section.get("ceiling_passes_declined"))
    # HOW FULL THE STORE IS, per the rank that last measured it, and on what basis -
    # "the store is 68 GiB" and "I stat-ed 20,000 files and scaled" are different
    # claims, and `disk_fill` must carry the distinction through, so the file must.
    row["estimated_bytes"] = _rank_rollup_int(section.get("estimated_bytes"))
    basis = section.get("ceiling_measurement_basis")
    row["walk_basis"] = _clip(basis, 48) if isinstance(basis, str) else None
    # WHY A RANK'S SPAN PANEL IS ZERO, READ OUT OF THAT RANK'S OWN FILE. Modelled on
    # `share_basis` above: a STRING field, so it is published per rank and never summed
    # - and, like every other field here, it has to be NAMED to exist. A rank-file key
    # that this reader does not name is silently never rolled up: no error, no gap, no
    # `fields_rejected` entry, just a view in which the fact was never available. That
    # is exactly how the span-completeness panel stayed invisible for a day, so the
    # reason and the basis travel with the counters they explain.
    span_reason = section.get("span_shape_reason")
    row["span_shape_reason"] = (_clip(span_reason, SPAN_SHAPE_REASON_CHARS)
                                if isinstance(span_reason, str) else None)
    span_basis = section.get("span_shape_basis")
    row["span_shape_basis"] = (_clip(span_basis, SPAN_SHAPE_BASIS_CHARS)
                               if isinstance(span_basis, str) else None)
    row["fields_rejected"] = rejected
    if rejected:
        row["fields_rejected_note"] = _clip(
            f"{rejected} carried a value that is not a non-negative int under "
            f"{RANK_ROLLUP_COUNTER_MAX}, so they are excluded from the totals")
    if age > RANK_ROLLUP_FRESH_SECONDS:
        return row, "stale", f"{age:.0f}s old (fresh within " \
                             f"{RANK_ROLLUP_FRESH_SECONDS:.0f}s)"
    return row, "counted", None


# --- the one-second cache: what a polled path is allowed to cost -------------
#
# `worker_proof()` set the pattern this copies (`disk_restore.PROOF_CACHE_SECONDS`,
# 1.0 s): the rank files are the ONLY way a process knows what the other processes did,
# and `/snapshot` and `/status` are polled. Reading them per request would put up to
# `MAX_RANK_ROLLUP_FILES` JSON parses on a polled path; reading them never would make
# the view a lie. So: one parse per second AT MOST, keyed on a CONTENT SIGNATURE (each
# rank file's name, size and mtime) so that a rank which wrote is reparsed on the very
# next read rather than up to a second later. A clock-only TTL would be the wrong shape
# here: it would make a test (and an operator) wait for a value that is already on disk.
_rollup_cache: dict = {"mono": None, "signature": None, "result": None}


def _rank_file_signature():
    """(directory, ((name, size, mtime_ns), ...)) for the rank files, or None.

    None means "the directory cannot be listed", in which case NOTHING is cached: an
    unreadable diagnostics directory must stay as cheap to re-check as it is to
    check, and a cached failure would make a directory that came back look broken for
    a second.
    """
    directory = config.diagnostics_dir()
    if directory is None:
        return ("no-dir", ())
    rows = []
    try:
        with os.scandir(str(directory)) as entries:
            for entry in entries:
                if not (entry.name.startswith(RANK_FILE_PREFIX)
                        and entry.name.endswith(RANK_FILE_SUFFIX)):
                    continue
                stat = entry.stat(follow_symlinks=False)
                rows.append((entry.name, int(stat.st_size),
                             int(stat.st_mtime_ns)))
    except OSError:
        return None
    return (str(directory), tuple(sorted(rows)))


def rank_rollup(force: bool = False) -> dict:
    """THE CROSS-PROCESS VIEW of the spill tier, behind the one-second rank cache.

    `force=True` re-reads (the `op:status` door is a human asking a question right
    now); the polled paths (`/snapshot`, `/status`, and therefore `disk_fill`) take
    the cached answer when the rank files have not changed. Neither path can raise.
    """
    signature = None
    try:
        signature = _rank_file_signature()
    except Exception as exc:  # noqa: BLE001 - a cache key is never worth a fault
        snapshot._error(exc, "disk_store rank rollup signature")
    now = time.monotonic()
    cached = _rollup_cache.get("result")
    if (not force and cached is not None and signature is not None
            and _rollup_cache.get("signature") == signature
            and isinstance(_rollup_cache.get("mono"), (int, float))
            and (now - float(_rollup_cache["mono"])) < RANK_ROLLUP_CACHE_SECONDS):
        return cached
    result = _rank_rollup_guarded()
    _rollup_cache["mono"] = now
    _rollup_cache["signature"] = signature
    _rollup_cache["result"] = result
    return result


def reset_rank_rollup_cache_for_tests() -> None:
    """Drop the one-second cache. A test that rewrites rank files must not be served a
    previous test's fleet, and a reset is the only honest way to guarantee that."""
    _rollup_cache["mono"] = None
    _rollup_cache["signature"] = None
    _rollup_cache["result"] = None


def _rank_rollup_guarded() -> dict:
    """THE CROSS-PROCESS VIEW of the spill tier: what every rank's own counters saw.

    The public entry point, and the only thing about it that is not a reading: it
    cannot raise. `_rank_rollup_inner` builds the whole payload from ONE skeleton, so a
    degraded rollup and a healthy one ship the same keys; the outer guard below exists
    only for the case where even that skeleton fails, and answers the handful of keys a
    reader must always have rather than a 500 on the door.

    Published by `op:status` as `spill_rollup` (a SIBLING of `spill`, which stays
    this process's counters byte-for-byte). It answers the question the door could not
    answer on 2026-10-06: "this process enforced nothing" and "the enforcement happened
    in a process whose counters I am not looking at" look IDENTICAL in the `spill`
    block, and the second answer is the default on a multi-rank boot.

    READ-ONLY. It writes nothing, starts nothing, reads nothing but the rank files
    `worker_diag` already writes, bounded by `MAX_RANK_ROLLUP_FILES` opens of
    `MAX_RANK_ROLLUP_FILE_BYTES` each.
    """
    try:
        return _rank_rollup_inner()
    except Exception as exc:  # noqa: BLE001 - the door answers 200, always
        snapshot._error(exc, "disk_store rank rollup")
        return {"basis": "rollup_failed", "degraded": True, "complete": None,
                "totals": {}, "totals_reported_by": {},
                "totals_are_lower_bound": True, "ranks_counted": [],
                "ranks_expected": None, "ranks_missing": None,
                "reason": _clip(f"the rollup could not be built: "
                                f"{type(exc).__name__}", MAX_STRING_CHARS)}


def _rank_rollup_inner() -> dict:
    """Build and fill the rollup. ONE skeleton for every outcome, healthy or not, so a
    degraded payload and a complete one can never disagree about which keys exist.
    """
    out: dict = {"basis": None,
                 "totals": {field: None for field, _s, _u in RANK_ROLLUP_FIELDS},
                 "totals_reported_by": {field: 0
                                        for field, _s, _u in RANK_ROLLUP_FIELDS},
                 "totals_are_lower_bound": True,
                 "lower_bound_reason": None,
                 "ranks_counted": [],
                 "ranks_expected": None,
                 "ranks_expected_source": None,
                 "ranks_missing": None,
                 "rank_files_found": 0,
                 "listing_truncated": False,
                 "per_rank": [],
                 "per_rank_truncated": False,
                 "stale_ranks": [],
                 "excluded_files": [],
                 "excluded_total": 0,
                 "excluded_truncated": False,
                 "reason": None,
                 "complete": None,
                 "ceiling_bytes": None,
                 "writer_threads": None,
                 "distinct_epochs": 0,
                 "epoch_note": ("epochs are PER-PROCESS unless VLLM_E2_TIER_DIAG_EPOCH "
                                "is set (worker_diag), so more than one here is normal "
                                "on a multi-rank boot and is NOT evidence of two "
                                "boots"),
                 "this_process": {"rank": None, "rank_source": None,
                                  "counted_in_rollup": None,
                                  "note": ("this process could not resolve its own "
                                           "rank id, so it cannot say whether its "
                                           "`spill` counters are inside the totals or "
                                           "describe a different process")},
                 "fresh_seconds": RANK_ROLLUP_FRESH_SECONDS,
                 "degraded": False,
                 "kinds": {field: ("gauge" if field in RANK_ROLLUP_GAUGES
                                   else "counter-since-boot")
                           for field, _s, _u in RANK_ROLLUP_FIELDS},
                 "units": {field: unit for field, _src, unit in RANK_ROLLUP_FIELDS},
                 "meanings": dict(RANK_ROLLUP_MEANINGS),
                 "scope": ("totals are the SUM of each rank's OWN per-process "
                           "counters, read from its rank file; `spill` in this same "
                           "payload is THIS process's counters and nothing else. A "
                           "worker and the process answering this call are different "
                           "processes, so the two disagreeing is the expected case, "
                           "not a contradiction."),
                 "note": ("A total here is an honest sum over the ranks whose files "
                          "were fresh, never a fleet-wide counter: separate "
                          "processes, separate counters, no shared memory and no new "
                          "IPC. Where a rank is missing, stale or unreadable the "
                          "totals are a LOWER BOUND and `ranks_missing`/"
                          "`stale_ranks`/`excluded_files` name it, because a "
                          "confidently low total is the failure this view exists to "
                          "stop - it is the same claim a truncated ceiling walk used "
                          "to make about its own floor.")}
    try:
        filled = _fill_rank_rollup(out)
        _settle_local_overlap(filled)
        return filled
    except Exception as exc:  # noqa: BLE001 - a degraded rollup still answers 200
        snapshot._error(exc, "disk_store rank rollup")
        out["basis"] = "rollup_failed"
        out["degraded"] = True
        out["reason"] = _clip(f"the rollup raised {type(exc).__name__}",
                              MAX_STRING_CHARS)
        return out


def _settle_local_overlap(out: dict) -> None:
    """Say whether THIS process's own `spill` counters are inside the totals.

    `spill` and `totals` are two views that may describe the SAME process or two
    different ones, and a reader that adds them gets a number wrong by exactly one
    rank. Every return path lands here, so the answer is never a `null` the caller has
    to guess at - it is the difference between "my process did nothing and four others
    did" and "you are looking at one of the four", which is the same question the whole
    rollup exists to answer.
    """
    local = out.get("this_process")
    if not isinstance(local, dict):
        return
    rank = local.get("rank")
    if not isinstance(rank, str):
        return
    counted = rank in out["ranks_counted"]
    local["counted_in_rollup"] = counted
    local["note"] = (
        f"this process IS {rank} and its file is in the totals: `spill` and `totals` "
        "overlap, so never add them" if counted else
        f"this process is {rank} and its file is NOT in the totals, so `spill` and "
        "`totals` describe different processes and may legitimately differ")


def _mark_rollup_gaps(out: dict, expected, counted_ranks: list,
                      listed_ranks=None) -> None:
    """Name the ranks this rollup did NOT read, and set `complete` from that.

    THREE FACTS, kept apart on purpose, because collapsing them is how a gap becomes
    invisible: `ranks_missing` lists an expected rank the directory holds NO FILE for;
    `stale_ranks` lists a rank whose file exists but is too old to describe the present;
    `excluded_files` lists a file that exists and is unusable. A rank appears in at most
    one of the three, so a reader adding them up cannot double-count a gap or miss one.

    `ranks_missing: null` means the rank SET is unknown (this process cannot say), and
    `[]` means every rank the boot names has a file. The two must never be conflated: "I
    do not know the fleet" and "the fleet is fully accounted for" are opposite claims,
    and a reader can only tell them apart if both are spelled out.
    """
    missing = None
    if isinstance(expected, (list, tuple)):
        missing = sorted(set(expected) - set(listed_ranks or ()))
    out["ranks_missing"] = missing
    counted = set(counted_ranks)
    out["complete"] = (None if missing is None else
                       bool(counted) and set(expected) <= counted
                       and not out["excluded_total"])


def _rollup_gap_reason(out: dict, missing, expected) -> str:
    """WHY these totals are a lower bound, naming every rank that is not in them."""
    reasons = []
    if missing:
        reasons.append(f"no file at all from {', '.join(missing[:8])}")
    if out["stale_ranks"]:
        names = ", ".join(str(entry.get("rank"))
                          for entry in out["stale_ranks"][:8])
        reasons.append(f"stale, so excluded from the totals: {names}")
    if out["excluded_files"]:
        names = ", ".join(str(entry.get("file"))
                          for entry in out["excluded_files"][:8])
        reasons.append(f"unreadable or unusable: {names}")
    if out["listing_truncated"]:
        reasons.append(f"only {MAX_RANK_ROLLUP_FILES} of "
                       f"{out['rank_files_found']} files were opened")
    if expected is None:
        reasons.append("the rank set is unknown here, so a rank that never wrote a "
                       "file is invisible to this view as well")
    return "; ".join(reasons)


def _fill_rank_rollup(out: dict) -> dict:
    """Read the directory and fill `rank_rollup`'s payload. Contained by its caller."""
    from . import worker_diag  # noqa: PLC0415 - one rank rule, not two
    local_rank, local_source = worker_diag.rank_id()
    out["this_process"] = {"rank": local_rank, "rank_source": local_source,
                           "counted_in_rollup": None,
                           "note": ("this process resolved its rank to "
                                    f"{local_rank!r}; whether that rank's file was "
                                    "counted is decided below")}
    # The rank SET is asked for BEFORE the directory is touched, so even a payload that
    # can read nothing can still say which ranks it therefore cannot see.
    expected, expected_source = _expected_ranks()
    out["ranks_expected"] = expected
    out["ranks_expected_source"] = expected_source
    directory = config.diagnostics_dir()
    if directory is None:
        out["basis"] = "no_diagnostics_dir"
        out["degraded"] = True
        _mark_rollup_gaps(out, expected, [])
        out["lower_bound_reason"] = _clip(_rollup_gap_reason(out, out["ranks_missing"],
                                                             expected), 400)
        out["reason"] = (f"{config.ENV_DIAG_DIR} is not set, so no rank writes a file "
                         "and nothing can be rolled up: the zeros in `spill` are all "
                         "this view can see, and they are NOT evidence that no rank "
                         "evicted anything")
        return out
    try:
        names = sorted(entry.name for entry in os.scandir(str(directory))
                       if entry.is_file()
                       and entry.name.startswith(RANK_FILE_PREFIX)
                       and entry.name.endswith(RANK_FILE_SUFFIX))
    except OSError as exc:  # noqa: BLE001 - unreadable directory is data, not a 500
        out["basis"] = "directory_unreadable"
        out["degraded"] = True
        _mark_rollup_gaps(out, expected, [])
        out["lower_bound_reason"] = _clip(_rollup_gap_reason(out, out["ranks_missing"],
                                                             expected), 400)
        out["reason"] = (f"the diagnostics directory could not be listed: "
                         f"{type(exc).__name__}")
        return out
    out["rank_files_found"] = len(names)
    if len(names) > MAX_RANK_ROLLUP_FILES:
        out["listing_truncated"] = True
        names = names[:MAX_RANK_ROLLUP_FILES]
    if not names:
        out["basis"] = "no_rank_files"
        out["degraded"] = True
        _mark_rollup_gaps(out, expected, [])
        out["lower_bound_reason"] = _clip(_rollup_gap_reason(out, out["ranks_missing"],
                                                             expected), 400)
        out["reason"] = ("no rank file exists yet, so there is NO EVIDENCE of "
                         "enforcement in any process - which is a different claim "
                         "from `spill.evictions: 0` in this one")
        return out
    counted: list = []
    listed: list = []
    for name in names:
        # The rank a FILE NAME claims, whether or not the file is readable: it decides
        # "no file at all" (ranks_missing) versus "a file that told us nothing".
        claimed = name[len(RANK_FILE_PREFIX):-len(RANK_FILE_SUFFIX)]
        if worker_diag._valid_rank(claimed):
            listed.append(claimed)
        row, status, detail = _rank_file_rollup_row(
            os.path.join(str(directory), name), name)
        if status == "counted":
            counted.append(row)
            continue
        record = {"file": _clip(name, MAX_FILENAME_DISPLAY), "status": status}
        if detail is not None:
            record["reason"] = _clip(detail, MAX_STRING_CHARS)
        if row is not None:
            record.update({key: row[key] for key in
                           ("rank", "age_seconds", "evictions", "spilled",
                            "spill_queued", "spill_errors", "bytes_written",
                            "queue_depth", "journal_removals", "drops",
                            "recency_promotions",
                            "recency_promotions_throttled", "ceiling_bytes",
                            "ceiling_share_bytes", "share_divisor",
                            "declined_because_already_under",
                            "estimated_bytes", "walk_basis",
                            "writer_thread_alive")})
        elif worker_diag._valid_rank(claimed):
            # The CONTENT told us nothing, but the NAME still claims a rank, and a
            # reader scanning "which of my ranks is unaccounted for" needs that name
            # here rather than having to line file names up by hand. Labelled as the
            # file's claim, because an unreadable body cannot vouch for it.
            record["rank"] = claimed
            record["rank_source"] = "the file name only; the body was unreadable"
        if status == "stale":
            out["stale_ranks"].append(record)
        else:
            out["excluded_files"].append(record)
    out["excluded_total"] = len(out["excluded_files"]) + len(out["stale_ranks"])
    out["excluded_files"] = out["excluded_files"][:MAX_RANK_ROLLUP_EXCLUDED]
    out["stale_ranks"] = out["stale_ranks"][:MAX_RANK_ROLLUP_EXCLUDED]
    out["excluded_truncated"] = out["excluded_total"] > (len(out["excluded_files"])
                                                         + len(out["stale_ranks"]))
    out["ranks_counted"] = sorted(row["rank"] for row in counted)
    _mark_rollup_gaps(out, expected, out["ranks_counted"], listed)
    if not counted:
        out["basis"] = "rank_files_unusable"
        out["degraded"] = True
        out["lower_bound_reason"] = _clip(_rollup_gap_reason(out, out["ranks_missing"],
                                                             expected), 400)
        out["reason"] = (f"{len(names)} rank file(s) were read and NONE was fresh "
                         "and usable: the totals are null rather than zero, and "
                         "`excluded_files`/`stale_ranks` name why")
        return out
    rows = sorted(counted, key=lambda row: row["rank"])
    out["per_rank"] = rows[:MAX_RANK_ROLLUP_ROWS]
    out["per_rank_truncated"] = len(rows) > MAX_RANK_ROLLUP_ROWS
    for field, _source, _unit in RANK_ROLLUP_FIELDS:
        values = [row[field] for row in rows if row[field] is not None]
        out["totals"][field] = sum(values) if values else None
        out["totals_reported_by"][field] = len(values)
    out["kinds"] = {field: ("gauge" if field in RANK_ROLLUP_GAUGES
                            else "counter-since-boot")
                    for field, _source, _unit in RANK_ROLLUP_FIELDS}
    ceilings = [row["ceiling_bytes"] for row in rows
                if row["ceiling_bytes"] is not None]
    out["ceiling_bytes"] = {
        "values": sorted(set(ceilings)),
        "min": min(ceilings) if ceilings else None,
        "max": max(ceilings) if ceilings else None,
        "agree": len(set(ceilings)) <= 1,
        "reported_by": len(ceilings),
        "note": ("a CEILING is a configured LIMIT, not a counter: summing it would be "
                 "a number about arithmetic, not about the box. Disagreement here "
                 "means two ranks booted against different limits, which is exactly "
                 "the state in which one rank keeps writing while another evicts")}
    alive = [row["writer_thread_alive"] for row in rows]
    out["writer_threads"] = {"alive": alive.count(True),
                             "not_alive": alive.count(False),
                             "unknown": alive.count(None),
                             "note": ("this is the field tonight's payload could not "
                                      "say: `spill.thread_alive: false` in the process "
                                      "answering the call, while every rank's writer "
                                      "thread was alive in its own")}
    out["distinct_epochs"] = len({row["epoch"] for row in rows
                                  if isinstance(row["epoch"], str)})
    out["epoch_note"] = ("epochs are PER-PROCESS unless VLLM_E2_TIER_DIAG_EPOCH is "
                         "set (worker_diag), so more than one here is normal on a "
                         "multi-rank boot and is NOT evidence of two boots (a rank "
                         "file from a PREVIOUS boot is caught by `ts`, not by this)")
    if local_rank is not None:
        out["this_process"]["rank"] = local_rank
    has_gap = bool(out["ranks_missing"]) or bool(out["stale_ranks"]) \
        or bool(out["excluded_files"]) or bool(out["listing_truncated"])
    if out["complete"] is True:
        out["basis"] = "rank_files"
        out["totals_are_lower_bound"] = False
    elif has_gap:
        out["basis"] = "rank_files_partial"
        out["degraded"] = True
        out["lower_bound_reason"] = _clip(
            _rollup_gap_reason(out, out["ranks_missing"], expected), 400)
    else:
        # Every file that EXISTS was counted and fresh; what is unknown is whether some
        # rank never wrote one. That is not a defect in this reading, but it does make
        # the totals a lower bound - and it must not be reported under the same word as
        # "a rank file is stale", which is a fact about a rank rather than about this
        # process's ignorance of the fleet.
        out["basis"] = "rank_files_fleet_unknown"
        out["lower_bound_reason"] = _clip(
            f"the totals cover the {len(out['ranks_counted'])} rank file(s) that "
            f"exist, but the rank set is unknown here ({expected_source}), so a rank "
            "that never wrote a file is invisible to this view", 400)
    return out


DISK_FILL_RANK_FIELDS = ("files_written", "bytes_written", "evictions",
                         "journal_removals", "recency_promotions",
                         "recency_promotions_throttled", "drops", "queue_depth",
                         "declined_because_already_under")
"""THE RANK ENTRY'S COUNTERS, and the whole list on purpose: this block is polled by a
   dashboard, so it carries per-rank counters and nothing else - no file lists, no
   sample arrays, no paths. A rank entry is ~10 short fields; `DISK_FILL_RANK_CAP`
   bounds how many entries one response can carry."""

DISK_FILL_PER_PROCESS_NOTE = (
    "THE FLAT `disk_spill` / `disk_store` COUNTERS BESIDE THIS BLOCK ARE THIS "
    "PROCESS'S OWN, and on the process that answers /snapshot they read zero by "
    "construction: the writer thread and the ceiling pass run in the WORKER "
    "processes. `disk_fill` is the fleet's answer, summed from the per-rank files. "
    "`disk_spill.spilled: 0` next to `disk_fill.totals.spilled: 368` is not a "
    "contradiction - it is one idle process and four busy ones. Never add the two "
    "views, and never read the flat one as the tier.")


def _disk_fill_row_entry(row: dict) -> dict:
    """ONE rank's entry: the counters, the age, and the fair-share verdict. Capped."""
    entry = {"rank": row["rank"],
             "file_age_seconds": row.get("age_seconds")}
    for field in DISK_FILL_RANK_FIELDS:
        entry[field] = row.get(field)
    # `files_written` is what the RANK FILE and the operator's question call it; the
    # rollup row publishes the same number as `spilled`. BOTH names are published, in
    # this entry and in `totals`, because per-rank and fleet-wide views that use
    # different names for one counter is how a consumer starts adding them twice. The
    # two are the same integer or both null - never one and not the other.
    entry["spilled"] = row.get("spilled")
    entry["files_written"] = row.get("spilled")
    # The fair-share verdict, per rank: what budget this rank thought it owned, over how
    # many ranks, and how many times it declined because a sibling got there first.
    entry["ceiling_share_bytes"] = row.get("ceiling_share_bytes")
    # The share in the currency a restore spends, and the arithmetic that produced it
    # (0.2.58): `block_equal` means the ranks' budgets were made equal by width, so the
    # block counts below are comparable rather than merely coincident.
    entry["ceiling_share_blocks"] = row.get("ceiling_share_blocks")
    entry["share_basis"] = row.get("share_basis")
    entry["share_divisor"] = row.get("share_divisor")
    entry["declined_because_already_under"] = row.get(
        "declined_because_already_under")
    entry["estimated_bytes"] = row.get("estimated_bytes")
    entry["walk_basis"] = row.get("walk_basis")
    entry["writer_thread_alive"] = row.get("writer_thread_alive")
    return entry


def _disk_fill_verdict(rows: list, ceilings: list, shares: list,
                       divisors: list) -> dict:
    """DO THE RANKS' PASSES AGREE? The `cross_process_verdict` block.

    This is how the 0.2.30 fix is WATCHED rather than assumed. Before the fix, four
    ranks each defended the whole ceiling over one directory and all four shed; after
    it, the ranks report the same share (`ceiling / n_effective`), usually exactly
    one of them has work to do, and the other three COUNT the decline.

    `declined_because_already_under` climbing across three of four rank files while
    `evictions` climbs in one is the signature of the fix working; four ranks
    evicting the same bytes with no declines is the signature of it not.
    """
    counted = [row for row in rows if isinstance(row, dict)]
    # THE BLOCK COMPARISON (0.2.58). Bytes are the budget's currency; blocks are the
    # SERVICABLE one, because a restore needs the block on every rank. Distinct values
    # here is the defect, and one value is the fix - readable from one polled block
    # rather than by dividing four files by four geometries.
    blocks = sorted({int(row["ceiling_share_blocks"]) for row in counted
                     if row.get("ceiling_share_blocks") is not None})
    bases = sorted({str(row["share_basis"]) for row in counted
                    if isinstance(row.get("share_basis"), str)})
    with_evictions = [row for row in counted
                      if (row.get("evictions") or 0) > 0]
    declining = [row for row in counted
                 if (row.get("declined_because_already_under") or 0) > 0]
    declines = sum(int(row.get("declined_because_already_under") or 0)
                   for row in counted)
    if not counted:
        return {"verdict": "no_ranks_reported",
                "ranks_with_evictions": 0, "ranks_declining": 0,
                "declines_total": 0,
                "note": ("no fresh rank file reported, so no claim is made about the "
                         "fleet: this is not evidence that the ranks agree")}
    # DO THE RANKS AGREE? In BLOCKS, since 0.2.58 - and the change of currency here is
    # not cosmetic. A block-equal split gives a WIDE rank MORE bytes than a narrow one
    # by design (same depth, wider block), so a rule that flags "different share bytes"
    # would call every healthy mixed-geometry boot a disagreement and teach an operator
    # to ignore the verdict. Bytes are still the currency when NO rank published a block
    # count, which is the mixed-version / pre-0.2.58 case the old rule was written for.
    if len(ceilings) > 1 or (blocks and (len(blocks) > 1 or len(bases) > 1)) \
            or (not blocks and len(shares) > 1):
        return {"verdict": "ranks_disagree_on_the_share",
                "ranks_with_evictions": len(with_evictions),
                "ranks_declining": len(declining),
                "declines_total": int(declines),
                "share_bytes_values": sorted(shares),
                "share_blocks_values": blocks,
                "share_basis_values": bases,
                "ceiling_bytes_values": sorted(ceilings),
                "blocks_equal": bool(len(blocks) <= 1),
                "note": (("the ranks report DIFFERENT BLOCK COUNTS for one store, or "
                          "ranks that split it with different arithmetic: a shared "
                          "store whose ranks hold different depths is the state where "
                          "a restore finds the block on three ranks and not the "
                          "fourth")
                         if blocks else
                         ("the ranks report DIFFERENT shares (or ceilings) for what "
                          "should be one store: at least one rank is running a "
                          "different build or a different ceiling, and a shared store "
                          "with two budgets is the state where one rank keeps writing "
                          "while another evicts"))}
    solo = bool(divisors) and max(divisors) <= 1
    n_effective = max(divisors) if divisors else None
    if solo:
        note = ("every rank sized the store against the WHOLE ceiling: either this "
                "is a single-rank boot (correct) or the ranks cannot see each other's "
                "digests, which is the fail-safe direction and costs capacity")
    elif declining and len(with_evictions) <= max(1, len(counted) // 2):
        note = (f"the ceiling is split {n_effective} ways, one rank's pass did the "
                f"work and {len(declining)} declined because the store was already "
                f"under their share - the 0.2.30 fix visible in the counters"
                + (f"; every rank holds the same {blocks[0]} blocks"
                   if len(blocks) == 1 else ""))
    else:
        note = (f"the ceiling is split {n_effective} ways and "
                f"{len(with_evictions)} of {len(counted)} ranks are over their own "
                "share: normal while the store fills, and not the old defect (which "
                "had ALL of them over the WHOLE ceiling and zero declines)")
    return {"verdict": ("single_rank_share" if solo else "shared_share"),
            "ranks_with_evictions": len(with_evictions),
            # EQUAL IN BLOCKS, NOT BYTES (0.2.58): one distinct block count across the
            # ranks that published one, and the arithmetic they used. `blocks_equal`
            # false alongside `share_basis_values: ["block_equal"]` would mean the
            # widths lied;
            # `equal_bytes` means at least one rank could not see a width, which costs
            # capacity and not correctness.
            "share_blocks": blocks[0] if len(blocks) == 1 else None,
            "share_blocks_values": blocks,
            "blocks_equal": bool(len(blocks) <= 1),
            "share_basis": bases[0] if len(bases) == 1 else None,
            "share_basis_values": bases,
            "ranks_declining": len(declining),
            "ranks_acting": [row["rank"] for row in with_evictions][:8],
            "ranks_declining_names": [row["rank"] for row
                                      in declining][:8],
            "declines_total": int(declines),
            # `share_bytes` is only ONE NUMBER when the ranks agree on it; with the
            # block-equal split a mixed-geometry boot legitimately has two (same depth,
            # different widths), and naming the smaller one "the share" would be a lie.
            # The values are published either way, so the reader never guesses.
            "share_bytes": shares[0] if len(shares) == 1 else None,
            "share_bytes_values": sorted(shares),
            "n_effective": n_effective,
            "note": _clip(note, 400),
            }


def _fill_skeleton() -> dict:
    """ONE KEY SET, built once. Both the healthy and the degraded fill come
    through here, so a reader can never meet two shapes of this block.

    The degraded block has the SAME keys with null values: `available: false`
    plus a `reason` is the only difference, because a caller that reads
    `fill["ceiling_bytes"]` on a route must not KeyError on the one day the
    read failed.
    """
    return {"available": False, "reason": None,
                 # THE FILL, in the currency the ceiling is configured in.
                 "used_bytes": None, "ceiling_bytes": None,
                 "used_fraction": None, "target_fraction": None,
                 "target_bytes": None, "headroom_bytes": None,
                 "over_target": None, "over_ceiling": None,
                 # HOW the fill was measured. THE OPERATOR ASKED FOR THIS FIELD BY NAME
                 # and the distinction is the whole reason it is here: "the store is 68
                 # GiB" and "I stat-ed 20,000 of 34,896 files and scaled" are different
                 # claims, so the basis travels WITH the number, exactly as
                 # `disk_spill.ceiling_measurement.basis` does. `read_basis` says how
                 # COMPLETE the read was and `ceiling_basis` where the ceiling came
                 # from, because a fill number has three claims behind it and folding
                 # them into one word is how a partial read starts looking like a
                 # measurement.
                 "basis": None, "read_basis": None, "ceiling_basis": None,
                 "used_bytes_source": None, "used_bytes_spread_bytes": None,
                 # WHO REPORTED. A count and a list, so "3 of 4 ranks reported" is a
                 # field rather than something a reader infers from a missing key.
                 # `ranks_expected` / `missing_ranks` are null (never empty) when this
                 # process cannot know the rank set: absence of knowledge is not a fact
                 # about the fleet, and a `[]` here would be a lie.
                 "ranks_expected": None, "ranks_reported": 0,
                 "missing_ranks": None, "ranks_stale": [],
                 "ranks_unusable": [], "ranks_unusable_total": 0,
                 "per_rank": [], "per_rank_truncated": False,
                 "totals": {field: None for field, _s, _u in RANK_ROLLUP_FIELDS},
                 "totals_are_lower_bound": True,
                 "lower_bound_reason": None,
                 "cross_process_verdict": None,
                 "fresh_seconds": RANK_ROLLUP_FRESH_SECONDS,
                 "per_process_note": DISK_FILL_PER_PROCESS_NOTE,
                 "scope": ("fleet-wide, read from the rank files the WORKERS write; "
                           "see per_process_note for what the flat per-process "
                           "counters beside this block do and do not mean"),
                 "note": ("the SSD tier's fill as the ranks themselves measured it: "
                          "`used_fraction` over `target_fraction` is the one "
                          "comparison that matters, and `ranks_reported` / "
                          "`missing_ranks` say how much of the fleet it covers"),
                 "units": {"used_bytes": "bytes", "ceiling_bytes": "bytes",
                           "used_fraction": "fraction of the ceiling",
                           "target_fraction": "fraction of the ceiling",
                           "target_bytes": "bytes", "headroom_bytes": "bytes",
                           "used_bytes_spread_bytes": "bytes",
                           "ranks_expected": "ranks", "ranks_reported": "ranks"},
                 }


def disk_fill_unavailable(reason: str) -> dict:
    """The fill block for a caller that could not build it at all - same keys."""
    out = _fill_skeleton()
    out["basis"] = "fill_failed"
    out["read_basis"] = "fill_failed"
    out["reason"] = _clip(f"the fill could not be built: {reason}",
                          MAX_STRING_CHARS)
    return out


def disk_fill() -> dict:
    """HOW FULL IS THE SSD TIER, AND IS THAT A PROBLEM - in one read-only block.

    THE QUESTION THIS EXISTS TO ANSWER WITHOUT ARITHMETIC ACROSS SECTIONS. The
    operator's words on 2026-10-22, after a night of reading four rank files by hand:
    "I just want to see fill." Everything the answer needs is here: how many bytes
    the store holds (as the ranks measured it, with the basis they measured it on),
    the ceiling and the 0.9 target it is kept under, one entry per rank that reported
    with the age of its file, which ranks did NOT report, and whether the ranks'
    ceiling passes agree with each other.

    READ-ONLY and CAPPED: it is a view over `rank_rollup()`'s already-cached rank read
    (no new file I/O per request, no walk, no statvfs), it emits rank entries only - no
    file lists, no samples - and it cannot raise: a failure returns the same keys with
    nulls and a `reason`, never a 500 on a polled door.

    `used_bytes` is the FRESHEST rank's store estimate, not a sum: four ranks
    measuring one directory produced four views of one number, and adding them would
    be the same mistake the ceiling pass just stopped making.
    `used_bytes_spread_bytes` says how far apart those four views were, because a
    wide spread means the ranks are measuring at different times or one is running a
    different walk bound.
    """
    out = _fill_skeleton()
    try:
        rollup = rank_rollup()
        _fill_disk_fill(out, rollup)
        return out
    except Exception as exc:  # noqa: BLE001 - a polled door answers 200 with nulls
        snapshot._error(exc, "disk_store disk fill")
        out["basis"] = "fill_failed"
        out["read_basis"] = "fill_failed"
        out["reason"] = _clip(f"the fill could not be built: "
                              f"{type(exc).__name__}", MAX_STRING_CHARS)
        return out


def _fill_disk_fill(out: dict, rollup: dict) -> None:
    """Fill the skeleton from ONE rollup. Mutates `out`, and never raises upward."""
    rows = [row for row in (rollup.get("per_rank") or []) if isinstance(row, dict)]
    out["ranks_expected"] = (len(rollup["ranks_expected"])
                             if isinstance(rollup.get("ranks_expected"),
                                           (list, tuple)) else None)
    # WHO REPORTED. THE COUNT COMES FROM THE ROLLUP'S COUNTED RANKS, not from the length
    # of the (capped) entry list: a 24-rank boot whose entries are capped at 16 must
    # still say 24 reported, because a short list that also lowers the count is the
    # "missing rank folded into a low total" mistake this block exists to avoid.
    counted = rollup.get("ranks_counted")
    out["ranks_reported"] = (len(counted) if isinstance(counted, (list, tuple))
                             else len(rows))
    out["missing_ranks"] = (list(rollup["ranks_missing"])
                            if isinstance(rollup.get("ranks_missing"),
                                          (list, tuple)) else None)
    # HOW COMPLETE THE READ WAS, in the same vocabulary `rank_rollup` publishes, so a
    # caller that reads either block sees one word for one fact.
    out["read_basis"] = rollup.get("basis")
    out["ranks_stale"] = [{"rank": entry.get("rank"),
                           "file_age_seconds": entry.get("age_seconds"),
                           "evictions": entry.get("evictions"),
                           "files_written": entry.get("spilled"),
                           "reason": _clip(entry.get("reason"), 96)}
                          for entry in (rollup.get("stale_ranks") or [])
                          if isinstance(entry, dict)][:DISK_FILL_RANK_CAP]
    excluded = [entry for entry in (rollup.get("excluded_files") or [])
                if isinstance(entry, dict)]
    out["ranks_unusable_total"] = int(rollup.get("excluded_total") or 0)
    out["ranks_unusable"] = [{"file": entry.get("file"),
                              "rank": entry.get("rank"),
                              "status": entry.get("status"),
                              "reason": _clip(entry.get("reason"), 96)}
                             for entry in excluded][:DISK_FILL_RANK_CAP]
    for field, _source, _unit in RANK_ROLLUP_FIELDS:
        out["totals"][field] = (rollup.get("totals") or {}).get(field)
    # The same two names for one counter, here as in `per_rank`: `spilled` is the
    # rollup's name, `files_written` is the rank file's, and neither view may use only
    # one of them.
    out["totals"]["files_written"] = out["totals"]["spilled"]
    out["totals_are_lower_bound"] = bool(rollup.get("totals_are_lower_bound"))
    out["lower_bound_reason"] = _clip(rollup.get("lower_bound_reason"), 400)
    entries = [_disk_fill_row_entry(row) for row in rows][:DISK_FILL_RANK_CAP]
    out["per_rank"] = entries
    out["per_rank_truncated"] = (len(rows) > DISK_FILL_RANK_CAP
                                 or bool(rollup.get("per_rank_truncated")))
    # THE FILL. The ceiling is the ranks' configured value when they agree on it, and
    # this process's environment otherwise (labelled either way, because a ceiling read
    # from the answering process is a weaker claim than one every rank agrees on).
    ceilings = sorted({int(row["ceiling_bytes"]) for row in rows
                       if row.get("ceiling_bytes") is not None})
    env_ceiling = disk_spill.capacity_ceiling_bytes()
    if len(ceilings) == 1:
        out["ceiling_bytes"] = ceilings[0]
        out["ceiling_basis"] = "ranks_agree"
    elif len(ceilings) > 1:
        out["ceiling_bytes"] = max(ceilings)
        out["ceiling_basis"] = "ranks_disagree"
    elif env_ceiling:
        out["ceiling_bytes"] = int(env_ceiling)
        out["ceiling_basis"] = "this_process_env_only"
    else:
        out["ceiling_basis"] = "no_ceiling_known"
    # THE MEASUREMENT, from the freshest rank that has one. NEVER a sum, NEVER a zero:
    # four ranks measuring one directory produced four views of one number, and a rank
    # that has not walked reports NO estimate rather than a 0 that would read as an
    # empty store.
    measured = [row for row in rows if row.get("estimated_bytes") is not None]
    if measured:
        freshest = min(measured, key=lambda row: float(
            row.get("age_seconds") or 0.0))
        out["used_bytes"] = int(freshest["estimated_bytes"])
        # THE OPERATOR'S `basis`: "the store is 68 GiB" versus "I stat-ed 20,000 of
        # 34,896 files and scaled them". It comes from the SAME rank the number came
        # from, because a basis from a different rank is a different measurement.
        out["basis"] = freshest.get("walk_basis") or "basis_not_reported"
        out["used_bytes_source"] = {"rank": freshest["rank"],
                                    "file_age_seconds":
                                        freshest.get("age_seconds"),
                                    "walk_basis": freshest.get("walk_basis")}
        values = [int(row["estimated_bytes"]) for row in measured]
        out["used_bytes_spread_bytes"] = max(values) - min(values)
        out["available"] = True
    elif rows:
        out["basis"] = "no_rank_reported_a_measurement"
        out["reason"] = (f"{len(rows)} rank(s) reported counters but none carried a "
                         "store estimate: either no rank has run a ceiling pass yet "
                         "or its rank file predates the 0.2.30 fields. `used_bytes` "
                         "is null, not 0, because an empty store and an unmeasured "
                         "one are different facts")
    else:
        out["basis"] = "no_rank_files"
        out["reason"] = ("no fresh rank file exists, so the store's size is UNKNOWN "
                         "here - which is a different claim from `disk_spill."
                         "estimated_bytes: 0` in the per-process block")
    ceiling = out["ceiling_bytes"]
    if ceiling:
        fraction = disk_spill.EVICT_TARGET_FRACTION
        out["target_fraction"] = fraction
        out["target_bytes"] = int(int(ceiling) * fraction)
        if out["used_bytes"] is not None:
            out["used_fraction"] = round(out["used_bytes"] / float(ceiling), 6)
            out["headroom_bytes"] = int(ceiling) - int(out["used_bytes"])
            out["over_target"] = bool(out["used_bytes"] > int(out["target_bytes"]))
            out["over_ceiling"] = bool(out["used_bytes"] > int(ceiling))
    shares = sorted({int(row["ceiling_share_bytes"]) for row in rows
                     if row.get("ceiling_share_bytes") is not None})
    divisors = [int(row["share_divisor"]) for row in rows
                if row.get("share_divisor") is not None]
    out["cross_process_verdict"] = _disk_fill_verdict(rows, ceilings, shares,
                                                      divisors)


def _op_status(_body: dict) -> dict:
    from . import (
        disk_boot,  # noqa: PLC0415 - it imports disk_restore, which imports us
        worker_diag,  # noqa: PLC0415 - one rank rule, not two
    )
    rank, source = worker_diag.rank_id()
    state = disk_spill.counters()
    # ONE read of the index's journal view, reused by the fields below.
    journal_fs = disk_index.journal_fs_state()
    journal_index = disk_index.counters()
    pending_path = _pending_path()
    _record, pending_reason = _read_pending_record(pending_path)
    return {"ok": True, "reason": None, "op": "status", "rank": rank,
            "rank_source": source, "armed": disk_spill.is_armed(),
            "capability": disk_spill.capability(),
            "root": disk_spill.disk_root(),
            "ceiling_bytes": disk_spill.capacity_ceiling_bytes(),
            "arm_state": disk_spill.arm_state(),
            # THE SSD TIER'S BOOT DEFAULT (0.2.33), on the door an operator reads first:
            # whether the flag asked for a boot that comes up armed and serving, whether
            # this boot honoured it, whether renewal has run, and whether a manual
            # disarm has latched. `disk_boot.view()` builds every key on every path, so
            # a degraded boot's answer has the same shape as a healthy one.
            "disk_boot_default": disk_boot.view(),
            "geometry_digest": disk_spill.geometry_digest(),
            "engine_fields": engine_fields(),
            "worker_fingerprint": fingerprint_report(),
            # disk_spill's own counters, verbatim: this verb and `snapshot` must
            # report the SAME numbers for the same mechanism, and a renamed or
            # re-derived counter is how two views start disagreeing in production.
            "spill": {key: state[key] for key in
                      ("requests", "accepted", "spilled", "duplicates",
                       "bytes_written", "drops", "evictions", "evicted_bytes",
                       "estimated_bytes", "errors", "thread_started",
                       "thread_alive", "queue_depth", "queue_depth_peak",
                       "window_files", "group_collisions", "journal_records",
                       "journal_removals", "journal_failures",
                       # HOW the ceiling was measured, not just where it stands. An
                       # `estimated_bytes` of 396 GiB and one of 396 GiB-extrapolated-
                       # from-20,000-of-34,896-files are different claims, and this is
                       # the door an operator reads when the disk is 99% full; the
                       # witness that says "the ceiling never woke up" is
                       # `estimate_seed_walks: 0` over a full store.
                       "ceiling_checks", "estimate_seed_walks", "walk_basis",
                       "walk_sampled_files", "walk_total_files",
                       "walk_measured_bytes", "walk_entries_examined",
                       "walk_skipped_files", "walk_truncated",
                       "walk_count_truncated", "plan_truncated")},
            # THE SAME MECHANISM, SEEN FROM OUTSIDE THIS PROCESS. Everything above -
            # `spill.evictions`, `ceiling_checks`, the `walk_*` fields - is THIS
            # process's `disk_spill._state`, and on a multi-rank boot the processes
            # that RUN the ceiling pass and the writer thread are the WORKERS, not
            # whoever is answering this call. That is why a boot that shed 234 GiB
            # answered this door with every counter at zero: the fields were published
            # correctly and read from the wrong process, which is the same thing as
            # not publishing them. Additive by contract: `spill` keeps its keys and
            # its meanings exactly as they were, and this block only ever adds a view.
            "spill_rollup": rank_rollup(),
            # THE JOURNAL, SAYING IT OUT LOUD. `spill.journal_failures` is this
            # process's count of refusals it was handed; `journal.write_errors` is
            # the INDEX's count of lines it never got, and `journal.writable:
            # false` is the reason a store full of KV bytes can come up with an
            # empty index over it. The spill commits and the journal records
            # nothing - a cost in disk space, deliberately not a cost in tier -
            # so the payload is where that trade becomes visible.
            "journal": disk_spill.journal_view(fs=journal_fs, index=journal_index),
            "journal_write_errors": int(journal_index["journal_write_errors"]),
            "journal_last_error": journal_index["journal_last_error"],
            "journal_directory_writable": journal_fs.get("writable"),
            "spill_refused": dict(state["refused"]),
            "group_collisions": int(state["group_collisions"]),
            "group_collision_samples": list(state["group_collision_samples"]),
            "queue_max": disk_spill.MAX_QUEUE_DEPTH,
            "spill_one": {"pending": pending_path, "pending_state": pending_reason,
                          "requests_written": int(_state["requests_written"]),
                          "poll_interval_seconds": POLL_INTERVAL_SECONDS,
                          "ttl_seconds": REQUEST_TTL_SECONDS},
            "hook": hook_section(include_fingerprint=False),
            "retained": retained_view()}


def _op_arm(body: dict) -> dict:
    """`disk_spill.arm`, with the body UNPACKED.

    The verb's table calls every op with the whole body, so an op whose engine
    function takes a scalar has to be adapted here rather than handed the dict: a
    body passed where `seconds` belongs fails closed (`seconds must be a number`),
    which is safe but leaves an operator arming a tier with a message about types.

    THE BOOT DEFAULT'S OTHER HALF (0.2.33): a hand arm LIFTS the operator's disarm
    latch. This door is the human's, so an operator who disarmed and then armed by hand
    has asked for the tier back - and since 0.2.35 the latch it clears is not a renewal
    veto but a VETO OVER THIS PROCESS'S OWN BOOT DECISION (`disk_boot._gate_verdict`),
    so leaving it set would mean a later connector boot in this process refusing to arm
    a box its operator just chose to serve from disk. A refused arm lifts nothing.

    `seconds` IS ACCEPTED AND IGNORED (0.2.35). It stays in the wire contract because an
    existing `{"op":"arm","seconds":7200}` must not start failing on the day this
    ships; `disk_spill.arm()` reports `seconds_ignored: true` rather than pretending the
    number bought anything.
    """
    from . import disk_boot  # noqa: PLC0415 - it owns the boot default's latch
    verdict = disk_spill.arm(body.get("seconds"))
    if verdict.get("armed"):
        disk_boot.note_operator_arm("POST {\"op\":\"arm\"}")
    return verdict


def _op_disarm(_body: dict) -> dict:
    """`disk_spill.disarm`, and the operator's veto over the boot default (0.2.33).

    A manual disarm is an INSTRUCTION, and since 0.2.33 there is a boot default for it
    to override - so this door latches `disk_boot`. Since 0.2.35 the latch is what stops
    this PROCESS'S OWN BOOT DECISION from re-arming (renewal is gone: the arm is a
    latch, so there is nothing to renew and nothing to lapse). That is the
    non-negotiable ordering, and it is stronger than it was: a human at the verb beats
    the plugin's own boot posture for the life of the process, and with no timer there
    is also no deadline after which the veto quietly stops mattering.
    """
    from . import disk_boot  # noqa: PLC0415 - it owns the boot default's latch
    verdict = disk_spill.disarm()
    disk_boot.note_operator_disarm("POST {\"op\":\"disarm\"}")
    return verdict


def _op_offers(body: dict) -> dict:
    """Raise/drop this boot's GAP-FILL switch: RAM first, the disk fills the gap.

    `{"op":"offers"}` alone reads the posture. `{"op":"offers","enabled":true}` raises
    the switch, which makes this boot serve EVERY request whose host-RAM prefix came up
    short - there is no per-request filter any more, and `{"op":"authority"}` is the
    switch that makes the disk decide instead.

    Two separate switches on purpose, and neither is the spill arm: the incident boot
    had a healthy spill arm (raised for a spill measurement) and that alone was enough
    to put disk offers in front of live traffic. A fresh boot answers `enabled: false`
    and no restart, config file or environment variable changes that - only this verb
    does, and a restart undoes it.
    """
    from . import disk_restore  # noqa: PLC0415 - it imports this module
    return _apply_switch(body, "offers_enabled", disk_restore.set_offer_switch)


def _op_restore_source(body: dict) -> dict:
    """WHAT SIZES A RESTORE'S DESTINATION ROWS: the arena's free list, or a cap.

    `{"op":"restore-source"}` reads the posture; `{"op":"restore-source",
    "source":"bounded"}` prices every restore of this boot against the staging cap, and
    `{"op":"restore-source","source":"arena"}` puts the pricing back on the arena's free
    list exactly as before. `{"op":"restore-source","source":null}` clears the runtime
    choice and hands the decision back to the boot default
    (`VLLM_E2_TIER_DISK_RESTORE_SOURCE`).

    BOTH POSTURES TAKE THEIR ROWS FROM THE ARENA. That is the honest shape of this verb,
    and it is not what the first version of it promised: the fork releases a load's CPU
    rows when the load ends, so the plugin cannot own a private set of them (the
    reference audit in `disk_restore.py` names the line). `bounded` bounds how MANY rows
    may be in flight, which is what lets the arena be sized from the cap; it does not
    move WHERE a row comes from, because nothing can.

    PER BOOT, AND NOT PERSISTED - the same lifetime rule as `op:authority`: a restart
    returns the posture to the environment variable, so this verb is a lever you can
    pull on a serving box and a restart undoes. That is the point: an operator can size
    the pinned arena at roughly `cap + reserve` instead of at the worst prompt, and roll
    back in one POST if the cap turns out to be the constraint.

    A REQUEST ALREADY IN FLIGHT KEEPS THE POSTURE ITS OFFER CHOSE. The posture is read
    once per offer and stored on the ledger record, so flipping this verb between the
    offer and the copy cannot price a run one way and return its cap credit the other -
    an accounting fault that would look like a leak forever, and it is unrepresentable
    here rather than merely unlikely.

    A value that is neither word (nor the deprecated alias `bounce`) is refused and
    changes NOTHING, reported with the vocabulary so a typo in a curl cannot move a
    serving box in either direction.
    """
    from . import disk_restore  # noqa: PLC0415 - it imports this module
    out = {"ok": True, "posture": disk_restore.restore_source_verdict(),
           "staging_cap": disk_restore.restore_staging_state(),
           "per_boot": True, "survives_restart": False}
    if "source" in body:
        verdict = disk_restore.set_restore_source(body["source"])
        out["restore_source"] = verdict
        if not verdict.get("ok"):
            out["ok"] = False
            out["reason"] = f"restore_source: {verdict.get('reason')}"
        out["posture"] = disk_restore.restore_source_verdict()
        out["staging_cap"] = disk_restore.restore_staging_state()
    return out


def _op_authority(body: dict) -> dict:
    """THE global disk-authority switch: the disk decides EVERY request's prefix.

    `{"op":"authority"}` alone reads the posture; `{"op":"authority","enabled":true}`
    turns it on for every request this boot, and `{"op":"authority","enabled":false}`
    turns it back off. One POST is the whole change: raising it opens the offer gate by
    itself, so an operator does not have to raise two switches to serve everyone from
    SSD. Eligibility bounds are untouched by it - the pool budget, the CPU slots, the
    offer cap and "is the span actually spilled" all still decide how much any single
    request is offered.

    WHAT THIS REPLACED. Until 0.2.31 the authority rule was carried by a
    client-supplied marker (`kv_transfer_params.kvtier_disk_restore = true`) and a
    `scope` argument chose whether that marker was required. A request body must not
    decide whether its KV comes off NVMe, so the marker, the `scope` argument and the
    `flagged` / `skipped_no_marker` / `marker_key` fields it fed are all retired. No
    client can ask for disk authority and no client can decline it.

    And the lifetime, stated in the response because "is this posture sticky?" is the
    first question a posture this expensive deserves: per boot, lost by a restart, and
    raised by nothing except this POST - or, since 0.2.33, by this boot's own DEFAULT
    (`KVTIER_DISK_BOOT_ENABLE`, on unless the operator turned it off), which is the same
    setter and is reported beside it.

    LOWERING IT IS AN OPERATOR'S VETO (0.2.33). `enabled: false` latches `disk_boot`, so
    nothing in this process - since 0.2.35 including its own boot decision, and there is
    no renewal left to fear - will put the arm back under an operator who has just taken
    the box out of disk service. A manual disarm always beats the boot default.
    """
    from . import (
        disk_boot,  # noqa: PLC0415 - it owns the boot default's latch
        disk_restore,  # noqa: PLC0415 - it imports this module
    )
    verdict = _apply_switch(body, "disk_authority",
                            disk_restore.set_disk_authority)
    applied = verdict.get("disk_authority") or {}
    if applied.get("changed") and applied.get("enabled") is False:
        disk_boot.note_operator_disarm(
            "POST {\"op\":\"authority\",\"enabled\":false}")
    return verdict


def _apply_switch(body: dict, field: str, apply_switch) -> dict:
    """Read the posture, and apply `enabled` to one boot switch if it was sent.

    Reading is the default because an operator checking what this boot will do must not
    change it. But a key that is PRESENT is an instruction: `{"op":"authority",
    "enabled":null}` is a malformed command, not a read, and obeying it as a read would
    leave the operator holding an `ok: true` for a switch that never moved. A value that
    is not a real boolean is refused by the setter, so a typo in a curl cannot change
    engine behaviour in either direction.
    """
    from . import disk_restore  # noqa: PLC0415 - it imports this module
    out = {"ok": True, "posture": disk_restore.disk_posture(),
           "per_boot": True, "survives_restart": False}
    if "enabled" in body:
        verdict = apply_switch(body["enabled"])
        if not verdict.get("ok"):
            return {"ok": False, "reason": f"{field}: {verdict.get('reason')}",
                    "posture": disk_restore.disk_posture(),
                    "per_boot": True, "survives_restart": False}
        out[field] = verdict
    out["posture"] = disk_restore.disk_posture()
    return out


def _op_failures(body: dict) -> dict:
    """Why a disk LOAD did not happen, in the process that cannot see it happen.

    `{"op":"failures"}` reads the load-failure ring; `{"op":"failures","limit":N}` reads
    at most N. A read-only op, so a safe method reaches it (the mutating-op rule is
    for the ops that write).

    WHY AN OP AT ALL. Every reason already existed as a name in `WORKER_FAILURES` and
    a counter in the process that READ the file, and the only worker-to-scheduler
    channel is a set of GPU block ids - so `disk_restore.reads_failed` on EngineCore
    was structurally zero and the hunt for one failed read cost hours of reading rank
    files by hand. This is those files' contents, merged, newest first, with the byte
    figures a `buffer_cap` refusal decided on and the writing block's provenance.
    """
    from . import disk_restore  # noqa: PLC0415 - it imports this module
    return disk_restore.load_failure_report(limit=body.get("limit"))


def _op_adoptions(body: dict) -> dict:
    """WHICH REQUEST got how many tokens from disk, newest first (0.2.61).

    `{"op":"adoptions"}` reads the newest `MAX_ADOPTIONS_PUBLISHED` adoption records;
    `{"op":"adoptions","limit":N}` reads at most N (capped at
    `MAX_ADOPTION_RECORDS`, so a caller can ask for the whole ring in one read). A
    read-only op, so a safe method reaches it.

    WHY AN OP AT ALL. The polled snapshot carries only the newest 48 rows to keep its
    ~52 KiB document from inflating further, and 48 rows is 20 seconds of history in
    the busiest minute this box has ever measured. The ring holds 256; this is how a
    hunt that started an hour ago gets the rest of them. It is ALSO the only door for
    a reader that must distinguish "this request adopted nothing" from "the ring
    already recycled it": the answer is the report's `adopted`/`adopted_tokens`
    boot totals read against `ring_dropped`, never the absence of a row.
    """
    from . import disk_restore  # noqa: PLC0415 - it imports this module
    return disk_restore.disk_adoption_report(limit=body.get("limit"))


def _op_shadow(body: dict) -> dict:
    """The SHADOW VERIFIER: read its counters, or arm it in this process.

    `{"op":"shadow"}` alone READS (the counters, the enablement, the region list and
    widths, the limits in force). `{"op":"shadow","enabled":true|false}` latches the
    switch - the same shape as the two boot switches above: a key that is PRESENT is an
    instruction and a key that is absent is a question, and a value that is not a real
    boolean is refused rather than guessed at.

    IT IS LISTED AS MUTATING even though it writes nothing, because it CHANGES WHAT THE
    PROCESS DOES (a worker that was comparing nothing starts reading GPU blocks off the
    device). A browser that can issue a GET must not be able to do that, exactly as
    with `offers` and `authority`.

    THE SCOPE IS SAID IN THE ANSWER. This module writes no control file - it opens no
    file handle for writing anywhere, which is its whole safety story - so a POST that
    arrives at the EngineCore latches the EngineCore. The process that spills, and so
    the process that can compare a GPU block with a blob, is a WORKER, and a worker is
    armed with `VLLM_E2_TIER_DISK_SHADOW=1` in the engine's environment. `scope` says
    that, instead of letting an operator believe he armed the worker.
    """
    body = body if isinstance(body, dict) else {}
    if "enabled" in body:
        value = body["enabled"]
        if not isinstance(value, bool):
            return {"ok": False,
                    "reason": f"enabled must be a JSON boolean, got {value!r}: "
                              "nothing changed",
                    "shadow": disk_shadow.status()}
        applied = disk_shadow.set_enabled(value)
        applied["changed"] = True
        return {"ok": True, "reason": None, "shadow": applied,
                "note": ("the latch is this process only; the worker that spills is "
                         f"armed with {config.ENV_DISK_SHADOW}=1")}
    return {"ok": True, "reason": None, "shadow": disk_shadow.status()}


def _op_gpu_shadow(body: dict) -> dict:
    """The GPU-DIRECT SHADOW WRITER: read its counters, or arm it in this process.

    `{"op":"gpu-shadow"}` alone READS (the counters, the enablement, where the files go
    and why they might not go there, the comparison's verdicts, the retention's
    measurements). `{"op":"gpu-shadow","enabled":true|false}` latches the switch - the
    same shape as `_op_shadow` above and for the same reasons: a key that is PRESENT is
    an instruction, a key that is absent is a question, and a value that is not a real
    boolean is refused rather than guessed at.

    IT IS LISTED AS MUTATING, and here the case is stronger than `shadow`'s: it writes
    no control file either, but arming this makes a process WRITE KV FILES from bytes
    read off the device. A browser that can issue a GET must not be able to do that,
    exactly as with `offers`, `authority` and `shadow`.

    THE SCOPE IS SAID IN THE ANSWER, and it matters more here than anywhere else on this
    verb: the POST latches the process that ANSWERS (usually the EngineCore), while the
    process that spills, compares and writes is a WORKER, armed with
    `VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES=1` in the engine's environment. `scope` says so
    instead of letting an operator believe he armed a writer that is still off.
    """
    body = body if isinstance(body, dict) else {}
    if "enabled" in body:
        value = body["enabled"]
        if not isinstance(value, bool):
            return {"ok": False,
                    "reason": f"enabled must be a JSON boolean, got {value!r}: "
                              "nothing changed",
                    "gpu_shadow": disk_gpu_shadow.status()}
        applied = disk_gpu_shadow.set_enabled(value)
        applied["changed"] = True
        return {"ok": True, "reason": None, "gpu_shadow": applied,
                "note": ("the latch is this process only; the worker that spills and "
                         "writes is armed with "
                         f"{config.ENV_DISK_GPU_SHADOW_WRITES}=1")}
    return {"ok": True, "reason": None, "gpu_shadow": disk_gpu_shadow.status()}


def _op_restore_verify(body: dict) -> dict:
    """THE RESTORE-SIDE VERIFIER: read its counters, or arm it in this process.

    `{"op":"restore-verify"}` alone READS (the counters, the enablement, the geometry
    the comparisons used, the oracle's reachability, the limits in force).
    `{"op":"restore-verify","enabled":true|false}` latches the switch - the same shape
    as `shadow` and `gpu-shadow` above: a key that is PRESENT is an instruction, a key
    that is absent is a question, and a value that is not a real boolean is refused
    rather than guessed at.

    IT IS LISTED AS MUTATING, and here the case is `shadow`'s rather than
    `gpu-shadow`'s: it writes NOTHING anywhere, but it CHANGES WHAT THE PROCESS DOES - a
    worker that was comparing nothing starts reading restored GPU blocks off the device
    and opening spill files for reading on its own thread. A browser that can issue a
    GET must not be able to do that, exactly as with `offers`, `authority` and `shadow`.

    THE SCOPE IS SAID IN THE ANSWER, and for this op it matters more than for the other
    two: what is being armed lives at the RESTORE APPLY POINT, which is code only a
    WORKER runs. This module writes no control file (it opens no file handle for writing
    anywhere, which is its whole safety story), so a POST that arrives at the EngineCore
    latches a process that will never apply a restore, and it reports `checked: 0`
    forever for that reason alone. `scope` says so, instead of letting an operator read
    a zero as an all-clear. The worker is armed with
    `VLLM_E2_TIER_DISK_RESTORE_VERIFY=1`.
    """
    # Lazy for the same reason `_op_failures` reaches `disk_restore` lazily: the restore
    # half owns this module's registration, and the import is one dict lookup per call
    # on a path an operator drives by hand.
    from . import disk_restore_verify  # noqa: PLC0415 - it is installed by disk_restore
    body = body if isinstance(body, dict) else {}
    if "enabled" in body:
        value = body["enabled"]
        if not isinstance(value, bool):
            return {"ok": False,
                    "reason": f"enabled must be a JSON boolean, got {value!r}: "
                              "nothing changed",
                    "restore_verify": disk_restore_verify.status()}
        applied = disk_restore_verify.set_enabled(value)
        applied["changed"] = True
        return {"ok": True, "reason": None, "restore_verify": applied,
                "note": ("the latch is this process only; the worker that applies a "
                         "restore is armed with "
                         f"{config.ENV_DISK_RESTORE_VERIFY}=1")}
    return {"ok": True, "reason": None,
            "restore_verify": disk_restore_verify.status()}


def _op_gpu_write(body: dict) -> dict:
    """THE GPU-DIRECT WRITE: read its counters, or arm it in this process.

    `{"op":"gpu-write"}` alone READS (the enablement, the limits, the pinned pool's live
    shape, the counters, the geometry the comparison used and the last verdicts).
    `{"op":"gpu-write","enabled":true|false}` latches the switch - the same shape as
    `_op_restore_verify` above and for the same reasons: a key that is PRESENT is an
    instruction, a key that is absent is a question, and a value that is not a real
    boolean is refused rather than guessed at.

    IT IS LISTED AS MUTATING, and the case is the strongest on this verb: it writes no
    control file, but arming it makes the process issue ITS OWN host->device copy into a
    live KV window at the restore apply point. A browser that can issue a GET must not
    be able to do that, exactly as with `offers`, `authority`, `shadow` and
    `gpu-shadow`.

    THE SCOPE IS SAID IN THE ANSWER, and for this op it matters more than for any other:
    the write happens at the RESTORE APPLY POINT, which is code only a WORKER runs. This
    module writes no control file, so a POST that arrives at the EngineCore latches a
    process that can never write a window, and it will report `writes: 0` forever for
    that reason alone. `scope` says so instead of letting an operator read a zero as an
    all-clear. The worker is armed with
    `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE=1`.
    """
    # Lazy for the same reason `_op_restore_verify` reaches `disk_restore_verify`
    # lazily: the restore half owns this module's registration, and the import is one
    # dict lookup per call on a path an operator drives by hand.
    from . import disk_gpu_write  # noqa: PLC0415 - it is installed by disk_restore
    body = body if isinstance(body, dict) else {}
    if "enabled" in body:
        value = body["enabled"]
        if not isinstance(value, bool):
            return {"ok": False,
                    "reason": f"enabled must be a JSON boolean, got {value!r}: "
                              "nothing changed",
                    "gpu_write": disk_gpu_write.status()}
        applied = disk_gpu_write.set_enabled(value)
        applied["changed"] = True
        return {"ok": True, "reason": None, "gpu_write": applied,
                "note": ("the latch is this process only; the worker that applies a "
                         "restore is armed with "
                         f"{config.ENV_DISK_GPU_DIRECT_WRITE}=1")}
    return {"ok": True, "reason": None,
            "gpu_write": disk_gpu_write.status()}


def _op_identities(body: dict) -> dict:
    """The OTHER SIDE OF THE DIFF: what the index believes, per rank, digests only.

    WHY A `contains` AND AN `ls` WERE NOT ENOUGH. The one question the whole box
    investigation turned on was "which identities does the index believe, per RANK, and
    which files are actually on the store?" - and neither verb could answer it alone.
    `contains` names ONE identity and never enumerates; `ls` names files and knows
    nothing about ranks. Diffing them by hand needs a list, and a list of 262,144
    identities in one payload is not a diagnostic, it is an outage. So this is bounded
    (a page of at most `MAX_IDENTITIES_LIMIT`), paged (pass `after` to continue),
    filterable (`rank`, and `state` = all | offerable | partial) and DIGEST-ONLY: rows
    carry a key16, a group, a shard and the rank labels, never a path. The digest each
    rank named its own file with is not in the journal (the line is
    `<seq> <+|-> <key16> <bytes> <group>`), so this verb deliberately does not name
    files - pairing it with `op=ls` is the operator's join, and the README says so.

    It walks no store, moves no counter and reads no file: the index is per-process
    memory and this endpoint is polled, so the same rule that keeps `contains` from
    scanning a shard keeps this from turning a dashboard into an I/O storm.
    """
    limit = _bounded_int(body.get("limit"), MAX_LS_LIMIT_DEFAULT,
                         MAX_IDENTITIES_LIMIT)
    after = body.get("after")
    if after not in (None, ""):
        after = str(after)[:64]
        if not all(char.isalnum() or char in ":g_" for char in after):
            return {"ok": False, "op": "identities", "read_only": True,
                    "reason": "the `after` cursor must be an identity from the "
                              "previous page (a key16, a colon, and a group field)"}
    else:
        after = None
    rank = body.get("rank")
    rank = None if rank in (None, "") else str(rank)[:MAX_RANK_NAME_CHARS]
    which = str(body.get("which") or body.get("state") or "all").strip().lower()
    if which not in ("all", "offerable", "partial"):
        return {"ok": False, "op": "identities", "read_only": True,
                "reason": 'which must be "all", "offerable" or "partial"',
                "which": ["all", "offerable", "partial"]}
    page = disk_index.scan_identities(after=after, limit=limit, rank=rank,
                                      state=which)
    page["read_only"] = True
    page["limit"] = int(page["limit"])
    page["next_step"] = (
        "pair this with {\"op\":\"ls\",\"shard\":\"<shard>\"}: the store names every "
        "file kvslot-<key16>-g<group>-<geodigest12>.bin, one per rank, and the "
        "geodigest is the rank's own. An identity this verb reports WITHOUT a rank, "
        "for which `ls` nevertheless shows that rank's digest, is a withheld file; "
        "one it reports WITH a rank the store no longer holds is a false present. "
        "Page with `after` until `truncated` is false to make it a census.")
    page["sampled"] = False
    return page


_DISK_OPS = {"status": _op_status, "arm": _op_arm, "disarm": _op_disarm,
             "spill-one": _op_spill_one, "ls": _op_ls, "contains": _op_contains,
             "identities": _op_identities,
             "offers": _op_offers, "authority": _op_authority,
             "failures": _op_failures, "shadow": _op_shadow,
             "gpu-shadow": _op_gpu_shadow,
             "restore-verify": _op_restore_verify,
             # LAST, to match `DISK_OPS`: the status route advertises that tuple and a
             # test asserts the tuple and this table are the same list IN ORDER, so an
             # op
             # inserted mid-table is a route advertising an order it does not dispatch
             # in.
             "gpu-write": _op_gpu_write,
             "restore-source": _op_restore_source,
             # LAST, for the same in-order reason: per-request disk-adoption
             # attribution out of the scheduler's bounded ring (0.2.61).
             "adoptions": _op_adoptions}


def disk_for_engine(engine=None, args: dict | None = None) -> dict:  # noqa: ARG001 - the factory binds it as a method; see the docstring
    """The whole `/plugins/kvtier/disk` verb. Token-gated at `endpoint.py`; never
    raises. `op` defaults to `status`; a wrong name is refused, never guessed.

    THE OP IS MUTATING, so a caller that reached this function through a safe method
    is refused before any op runs: a browser's `<img>`/`<script>` preload issues a
    GET with no custom header and no CORS preflight, and the `op` is a QUERY
    PARAMETER, so the URL that spills a slot is as short as the URL that reads
    status. `endpoint.route_disk` therefore answers a safe method with 405.

    `engine` is the bound EngineCore, taken and unused: the sibling `blockdump`
    handler's signature is the convention the factory binds against, and this tier
    needs nothing from the engine - the CPU slots it spills live in the WORKER, and
    the only engine-side state it touches is `disk_spill`'s own gate and the request
    file a worker polls.
    """
    body = args if isinstance(args, dict) else {}
    op = str(body.get("op") or "status").strip().lower()
    if op not in DISK_OPS:
        return {"ok": False,
                "reason": f"unknown op {op!r}; the ops are {', '.join(DISK_OPS)}",
                "ops": list(DISK_OPS)}
    if op in MUTATING_DISK_OPS and body.get("safe_method"):
        return {"ok": False,
                "reason": f"op '{op}' mutates and will not run from a method a "
                          f"browser may issue without a preflight; send it as POST",
                "op": op}
    try:
        response = _DISK_OPS[op](body)
    except Exception as exc:  # noqa: BLE001 - the verb is contained by contract
        snapshot._error(exc, "disk_store verb")
        _state["errors"] = int(_state["errors"]) + 1
        response = {"ok": False, "op": op,
                    "reason": f"the op raised {type(exc).__name__}: {exc}"}
    if not isinstance(response, dict):
        response = {"ok": False, "reason": "an op returned a non-object result",
                    "op": op}
    response.setdefault("op", op)
    response["engine_fields"] = engine_fields()
    if "armed" not in response:
        response["armed"] = disk_spill.is_armed()
    # THE RESTORE SIDE OF THE SAME STORE, from THIS process. `/disk` is where an
    # operator comes to ask about the SSD tier, and before this the verb described
    # the write side only - so "it spilled 400 files and restored none" had no
    # answer here either. `scheduler_section()` is the SCHEDULER talking about its
    # own layer (the half that makes the offer), with `why_no_restore` naming the
    # first missing link. Contained: a restore verdict never breaks the spill verb.
    try:
        from . import disk_restore  # noqa: PLC0415 - it imports this module
        response.setdefault("restore", disk_restore.scheduler_section())
    except Exception as exc:  # noqa: BLE001 - a missing section is data
        snapshot._error(exc, "disk_store restore section")
        response.setdefault("restore", {"error": f"{type(exc).__name__}"[:200]})
    # Trim LAST, after the fields this wrapper adds. A budget enforced before the
    # shared fields were appended would be a guarantee the caller never receives:
    # `engine_fields` alone is a few hundred bytes, so an op that obediently trimmed
    # its own listing could still ship over the ceiling.
    return _trim(response)


# --- what a WORKER publishes about this module: the rank-file contribution ----

def _nonzero_refusals() -> dict:
    """Only the refusal reasons that happened, with their counts.

    Fifteen keys of which fourteen are zero is the reason this tier's silence was
    unreadable for a whole boot; a contribution that repeats the zeros spends the
    rank file's byte budget on nothing.
    """
    return {name: int(count) for name, count in _state["refused"].items()
            if count}


def worker_report() -> dict:
    """THE worker's disk verdict, for the rank file `worker_diag` writes.

    This is the answer to the question a boot could not answer on 2026-10-05: the
    engine wrote zero files while four ranks parked KV, and nothing in the worker
    said why. So the report names the gate (configured/armed/root), the identity
    (`digest`, or the fields it could not read), the engine fields AS OBSERVED ON AN
    ACTUAL STEP (`engine_fields.observed_on_a_step`: "the class declares
    `store_keys`" and "a metadata object this worker bound carried it" are
    different claims and only the second one means a spill can be named), the
    per-reason refusals, and the bytes that actually landed.

    BOUNDED and self-checking: `worker_diag` refuses a section over its byte cap,
    so a report that would be refused degrades ITSELF (drops the readable detail
    first, then the field names, says `truncated`) rather than costing the worker
    its whole section. Never raises; a JSON-serialisable dict in every path.

    ONE BLOCK IS STRUCTURE, NOT DETAIL: `peer_eviction` (0.2.34) ships on the
    healthy path, on the path where no ceiling pass ever ran, on the path where the
    counters raised, and through the byte-cap triage - the same keys every time, so
    the cross-rank witness can be read from four files by a reader that never has to
    ask whether a field is allowed to be missing.
    """
    try:
        state = disk_spill.counters()
        fields = engine_fields()
        missing = [str(item.get("field"))
                   for item in list(_state["fingerprint_missing"])[:8]
                   if isinstance(item, dict)]
        report = {"section": "disk", "writer": WRITER_TAG,
                  "protocol_version": PROTOCOL_VERSION,
                  "installed": bool(_state["installed"]),
                  "class": _state["class"],
                  "install_reason": _clip(_state["skip_reason"], 120),
                  "installs": int(_state["installs"]),
                  "reinstalls": int(_state["reinstalls"]),
                  "configured": bool(state["configured"]),
                  "armed": bool(disk_spill.is_armed()),
                  "root": _clip(disk_spill.disk_root(), 120),
                  "ceiling_bytes": disk_spill.capacity_ceiling_bytes(),
                  # THE FAIR SHARE AS THIS RANK LAST SIZED IT (0.2.30). Published per
                  # rank on purpose: "four ranks, one pass did the work, three declined"
                  # is only checkable if each rank says what budget it thought it owned.
                  # `ceiling_divisor` is `n_effective` (this rank plus the live siblings
                  # its own walk learned); 1 means it saw no sibling - the fail-safe
                  # value, and the one an operator must be able to spot from the door.
                  "ceiling_share_bytes": (int(state["share_bytes"])
                                          if state.get("share_bytes") else None),
                  # THE SHARE IN THE CURRENCY A RESTORE SPENDS (0.2.58): how many whole
                  # blocks that budget holds on THIS rank. Four ranks whose
                  # `ceiling_share_bytes` agree but whose `ceiling_share_blocks` do not
                  # are four ranks whose surplus can never be served, because a restore
                  # needs the block on every one of them. Null when this rank has no
                  # measured block width (a fingerprint-less process), never 0.
                  "ceiling_share_blocks": (int(state["share_blocks"])
                                           if state.get("share_blocks") else None),
                  "share_basis": (str(state.get("share_basis"))[:24]
                                  if state.get("share_basis") else None),
                  "ceiling_divisor": (int(state["share_divisor"])
                                      if state.get("share_divisor") else None),
                  "ceiling_passes_declined": int(
                      state.get("ceiling_passes_declined") or 0),
                  # HOW FULL THE STORE IS, per this rank's last walk, and on what basis:
                  # `disk_fill` cannot publish `used_bytes` unless a rank file carries
                  # it, and cannot label the claim unless the basis travels with it.
                  "estimated_bytes": (int(state["estimated_bytes"])
                                      if state.get("estimated_bytes") is not None
                                      else None),
                  "ceiling_measurement_basis": _clip(state.get("walk_basis"), 48),
                  "digest": _state["geometry_digest"],
                  "fingerprint_ready": _state["fingerprint"] is not None,
                  "fingerprint_missing": missing,
                  "binds": int(_state["binds"]),
                  "steps_idle": int(_state["steps_idle"]),
                  "completions": int(_state["completions"]),
                  "slots_offered": int(_state["slots_offered"]),
                  "spill_queued": int(_state["spill_queued"]),
                  # THE COMPLETENESS OF THIS RANK'S SPILLS, so the fleet's rate is a
                  # sum of four honest per-rank counters rather than the answering
                  # process's own view of a job it did not do (the 2026-10-05 "every
                  # number zero" failure, which is the reason this file exists).
                  "spill_spans_classified": int(_state["spill_spans_classified"]),
                  "spill_spans_complete": int(_state["spill_spans_complete"]),
                  "spill_incomplete_state": int(_state["spill_incomplete_state"]),
                  "spill_incomplete_attention": int(
                      _state["spill_incomplete_attention"]),
                  # THE STORE-END OF THE 0.2.64 PIPE: boundary-state anchor files this
                  # rank's hand-offs actually wrote. `mamba_anchor` cannot travel in
                  # this file because the patch is fitted in the scheduler and the
                  # mask never runs in a worker, so the scheduler's own mark count is
                  # published by `disk_store.section()`, not by a rank file.
                  "anchor_spills_completed": int(
                      _state["anchor_spills_completed"]),
                  # WHY A RANK'S SPAN PANEL IS DARK, AND ON WHAT AUTHORITY IT WAS READ.
                  # The reason and the basis live in THIS process; `span_shapes` (the
                  # view that used to hold them) is built by `hook_section()` in the
                  # engine process, where the cache behind it is empty, so a rank file
                  # without these keys could report zeros and never say why. Clipped -
                  # see `span_shape_publication`.
                  **span_shape_publication(),
                  "retained_events": int(len(_retained)),
                  "refused": _nonzero_refusals(),
                  "refusals_total": int(_state["refusals_total"]),
                  "last_refusal": _state["last_refusal"],
                  "files_written": int(state["spilled"]),
                  "duplicates": int(state["duplicates"]),
                  "group_collisions": int(state["group_collisions"]),
                  "group_collision_groups": _collision_digest_list(state),
                  "journal_records": int(state["journal_records"]),
                  "journal_failures": int(state["journal_failures"]),
                  "journal_removals": int(state["journal_removals"]),
                  "bytes_written": int(state["bytes_written"]),
                  "drops": int(state["drops"]),
                  "evictions": int(state["evictions"]),
                  # THE PEER-ATTRIBUTION WITNESS (0.2.34), as its own nested block.
                  # These counters existed in `disk_spill` and appeared NOWHERE a
                  # reader could reach them - not the rank file, not the dev route -
                  # so when the persistent journal proved cross-rank deletion was
                  # happening (15,848 removal lines of a sibling's 11,468,800-byte
                  # block size, recorded by ranks that were not its writer), nothing
                  # could answer the one question that separated "a rank shed a KNOWN
                  # peer's file because nothing else could meet its target" from "a
                  # rank could not name its peers at all". Nested, in the same style
                  # as `disk_restore`'s `readers` block, so one key set serves every
                  # path. `disk_spill.section()["peer_ranks"]` is the full view.
                  "peer_eviction": _peer_eviction_block(state),
                  "queue_depth": int(state["queue_depth"]),
                  "writer_thread_alive": bool(state["thread_alive"]),
                  "spill_errors": int(state["errors"]),
                  "hook_errors": int(_state["errors"]),
                  "fallbacks": int(_state["fallbacks"]),
                  "engine_fields": {"verdict": fields["verdict"],
                                    "store_keys": fields["store_keys"],
                                    "observed_on_a_step":
                                        fields["observed_on_a_step"],
                                    "fingerprint_observed_on_a_step":
                                        fields["fingerprint_observed_on_a_step"],
                                    "class_probe": fields["class_probe"]},
                  "reason": _clip(_state["last_reason"], 120),
                  "note": ("THE DISK TIER'S OWN VERDICT, FROM THIS RANK. "
                           "files_written is the only proof bytes landed; "
                           "refusals_total > 0 with files_written == 0 means the "
                           "tier ran and REFUSED, and refused names the reason. "
                           "group_collisions counts spills whose key16 this rank "
                           "had already spilled under a DIFFERENT group id - the "
                           "cross-group digest collisions the name now separates, "
                           "and the rate the tier is armed to measure. "
                           "engine_fields.observed_on_a_step is the runtime proof "
                           "that a metadata object carried store_keys - the class "
                           "declaring the field proves nothing about a step.")}
        return _degrade_report(report)
    except Exception as exc:  # noqa: BLE001 - a report is never a step fault
        snapshot._error(exc, "disk_store worker report")
        return {"section": "disk", "writer": WRITER_TAG,
                "installed": bool(_state["installed"]),
                # THE SAME KEY SET, on the path where the counters could not be read.
                # A reader that walks `disk.peer_eviction` in every rank file must
                # never get a KeyError because one rank's internals raised; every
                # field is present at its never-happened value.
                "peer_eviction": _peer_eviction_block(None),
                "report_error": type(exc).__name__}


# THE BLOCK'S FIELDS, ONE TABLE: (payload key, `disk_spill` state key, default).
# One table so the healthy path, a rank that never ran a ceiling pass and a report
# whose counters exploded cannot ship three different shapes of the same witness.
_PEER_EVICTION_FIELDS = (
    ("peer_digests_live", "peer_digests_live", 0),
    ("peer_digests_learned", "peer_digest_learned", 0),
    ("peer_digests_unconfirmed", "peer_digests_unconfirmed", 0),
    ("peer_digests_dropped", "peer_digests_dropped", 0),
    ("peer_digest_sightings", "peer_digest_sightings", 0),
    ("peer_files_planned", "peer_files_planned", 0),
    ("peer_files_removed", "peer_files_removed", 0),
    ("peer_files_removed_bytes", "peer_files_removed_bytes", 0),
    ("peer_files_removed_known_peer", "peer_files_removed_known_peer", 0),
    # WHICH RULE PAID FOR A REMOVAL (0.2.45). `peer_files_removed` cannot tell a live
    # sibling's bytes from the orphans a configuration change left behind, and those two
    # read in opposite directions - one is the incident, the other is the reclaim path
    # working. They are disjoint, so a reader can subtract.
    ("stale_geometry_removed", "stale_geometry_removed", 0),
    ("stale_geometry_removed_bytes", "stale_geometry_removed_bytes", 0),
    ("stale_key_shape_removed", "stale_key_shape_removed", 0),
    ("peer_files_removed_live", "peer_files_removed_live", 0),
    ("peer_files_removed_live_bytes", "peer_files_removed_live_bytes", 0),
    # THE IDENTITY VIEW: WHO THIS RANK COULD SEE when it decided.
    ("identity_published_digests", "peer_identity_digests", 0),
    ("identity_files_unnamed", "peer_identity_files_unnamed", 0),
    ("identity_files_stale", "peer_identity_files_stale", 0),
    ("identity_files_unreadable", "peer_identity_files_unreadable", 0),
    ("unattributable_entries_excluded", "unattributable_entries_excluded", 0),
    ("unattributable_entries_last_pass", "unattributable_entries_last_pass", 0),
    ("unattributable_bytes_last_pass", "unattributable_bytes_last_pass", 0),
    ("passes_short", "ceiling_passes_short", 0),
)


def _peer_eviction_block(state) -> dict:
    """THE PEER-ATTRIBUTION WITNESS FOR THE RANK FILE. Same keys on every path.

    `state` is `disk_spill.counters()` or anything that is not a dict - a rank that
    never ran a ceiling pass, or a `counters()` that raised, gets the same keys at
    their never-happened values rather than a thinner document. The prose lives in
    `disk_spill.section()["peer_ranks"]`, which is the version the `/disk` door
    serves; this is the compact cross-process copy, and the byte cap of the
    contribution seam is why it does not carry a second copy of that paragraph.
    """
    source = state if isinstance(state, dict) else {}
    block: dict = {}
    for name, key, default in _PEER_EVICTION_FIELDS:
        raw = source.get(key, default)
        block[name] = int(raw) if isinstance(raw, int) and not isinstance(
            raw, bool) else default
    block["plan_short"] = bool(source.get("ceiling_plan_short") or False)
    block["plan_short_bytes"] = int(source.get("ceiling_plan_short_bytes") or 0)
    # THE THREE-STATE VERDICT, not just its consequences. `identity_known` is what
    # licensed calling a foreign digest dead: with it false, the numbers above were
    # produced by a rank that could NOT name every live rank, and a reader who does not
    # know that will misread a withheld removal as a missing one.
    # `liveness_source` names
    # the witness that decided (`rank_files` / `write_activity`), so a box with no
    # diagnostics directory is visible from the rank file alone.
    block["identity_available"] = bool(source.get("peer_identity_available") or False)
    block["identity_known"] = bool(source.get("peer_identity_known") or False)
    source_name = source.get("peer_liveness_source")
    block["liveness_source"] = (str(source_name)[:32]
                                if isinstance(source_name, str) else None)
    return block


def _collision_digest_list(state: dict) -> list:
    """The collision samples, COMPACTED for the rank file's byte cap.

    The section payloads carry the full records; a contribution to someone else's
    file publishes one short string per event (`key16:gA/gB`) so the rate and the
    identities are readable without spending the seam's 4 KiB on it. A truncated
    sample list is still an honest count, because `group_collisions` above is the
    counter and this is only its evidence.
    """
    out = []
    for row in list(state.get("group_collision_samples") or [])[:4]:
        if not isinstance(row, dict):
            continue
        out.append(f"{str(row.get('key16'))[:16]}"
                   f":g{row.get('group_id_a')}/g{row.get('group_id_b')}"
                   f"{'/disk' if row.get('sibling_on_disk') else ''}")
    return out


def _degrade_report(report: dict) -> dict:
    """Keep the report inside the seam's byte cap by shedding detail, not truth."""
    from . import worker_diag  # noqa: PLC0415 - the seam owns its own cap
    limit = worker_diag.MAX_CONTRIBUTION_BYTES
    for step in ("detail", "fields", "minimal"):
        if len(json.dumps(report, default=str)) <= limit:
            return report
        if step == "detail":
            report = dict(report)
            report["fingerprint_missing"] = []
            report["note"] = "(trimmed to fit the rank file's byte cap)"
        elif step == "fields":
            report = dict(report)
            report["last_refusal"] = None
            report["reason"] = None
            report["install_reason"] = None
        else:
            return {"section": "disk", "writer": WRITER_TAG,
                    "truncated": True, "installed": report["installed"],
                    "armed": report["armed"],
                    "refusals_total": report["refusals_total"],
                    "files_written": report["files_written"],
                    # THE ONE BLOCK THE TRIAGE KEEPS. It is ~13 ints, it is the
                    # witness for an incident that cost a serving path, and dropping
                    # it would make the byte-cap path ship a different key set from
                    # the healthy one - the rule this file's own tests exist to hold.
                    "peer_eviction": report.get("peer_eviction")
                    or _peer_eviction_block(None),
                    "note": "the report exceeded the seam's byte cap"}
    return report


def register_contribution() -> dict:
    """Offer this module's verdict to the rank-file writer. Idempotent.

    Registered only where the module is RELEVANT: `install()` calls it after it has
    found the worker class, so an EngineCore or API-server process (which writes no
    rank file and has nothing to say about a spill) registers nothing and the
    section is simply absent - never an empty object that would make every rank
    file on every boot claim a disk tier it does not have.
    """
    from . import worker_diag  # noqa: PLC0415 - one writer, and no import cycle
    return worker_diag.register_contribution("disk", worker_report,
                                             owner="kvtier-dev-plugin/disk-store")


def boot_line() -> str:
    r"""ONE greppable line: what the disk tier can and did do in THIS process.

    The pair to `BOOT[auto]`, which says nothing about the disk tier at all. Fixed
    field order, bounded values, and the refusal table inline: `grep DISK\[spill\]`
    must answer "did the worker try, and why did it not write" without a verb.

    It is a LOG line and deliberately not a payload field: `op:status` already
    carries the digest at its top level, and pasting this line into that payload
    would put a third copy of one identity in one document (see the `hook_section`
    one-copy rule).

    It never raises. The line is the fallback for a reader that cannot call a verb,
    so a line that fails because a module it reads is broken - the exact situation it
    exists to describe - would be the worst possible failure mode.
    """
    try:
        fields = engine_fields()
        refused = _nonzero_refusals()
        parts = ["DISK[spill]"]
        parts.append("installed={}".format("YES" if _state["installed"] else "NO"))
        parts.append("class={}".format(_state["class"] or "none"))
        parts.append("configured={}".format("YES" if disk_spill.disk_root()
                                            else "NO"))
        parts.append("root={}".format(_clip(disk_spill.disk_root(), 80) or "-"))
        parts.append("tier={}".format("ARMED" if disk_spill.is_armed()
                                      else "DISARMED"))
        parts.append("engine_fields={}".format(fields["verdict"]))
        parts.append("keys_on_step={}".format(_state["keys_observed_on_a_step"]))
        parts.append("digest={}".format(_state["geometry_digest"] or "none"))
        parts.append("binds={}".format(int(_state["binds"])))
        parts.append("files={}".format(int(disk_spill.counters()["spilled"])))
        parts.append("refused={}".format(",".join(f"{name}:{count}"
                                                  for name, count in
                                                  sorted(refused.items())[:4])
                     or "none"))
        if refused and len(refused) > 4:
            parts.append("refused_more={}".format(len(refused) - 4))
        reason = _state["last_reason"] or _state["skip_reason"]
        if reason:
            parts.append("reason={}".format(_clip(reason, MAX_BOOT_LINE_CHARS)))
        line = " ".join(parts)
    except Exception as exc:  # noqa: BLE001 - the fallback cannot be allowed to fail
        snapshot._error(exc, "disk_store boot_line")
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        line = (f"DISK[spill] installed={'YES' if _state['installed'] else 'NO'} "
                f"unreadable={type(exc).__name__}")
    return line[:MAX_BOOT_LINE_CHARS * 3]


def emit_boot_line() -> dict:
    """Log `boot_line()` where the operator will actually read it. Once per distinct
    verdict per process: install runs at plugin load AND again off the handshake,
    and a per-step or per-retry echo of the same line is noise, not a signal."""
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
        snapshot._error(exc, "disk_store boot line")
        return {"logged": False, "reason": f"logging raised {type(exc).__name__}",
                "line": None}


# --- install / uninstall: the engine hook ------------------------------------

def _layer_is_ours(method) -> bool:
    return getattr(method, OWNER_ATTR, None) == OWNER_TAG


def _layer_is_sibling(method) -> bool:
    origin = getattr(method, "__module__", "") or ""
    return origin.startswith(SIBLING_PACKAGE)


def _wrapped_bind(original):
    def bind_connector_metadata(self, metadata):
        original(self, metadata)             # THE FORK FIRST; its result is unused
        observe_bind(self, metadata)         # contained: never raises
    bind_connector_metadata.__name__ = METHOD_BIND_METADATA
    return bind_connector_metadata


def _wrapped_get_finished(original):
    def get_finished(self, *args, **kwargs):
        before = completed_snapshot(self)    # BEFORE the fork: it does the completing
        result = original(self, *args, **kwargs)
        on_store_completed(self, before)
        poll_spill_one(self)
        return result                        # UNTOUCHED AND UNCONSUMED
    get_finished.__name__ = METHOD_GET_FINISHED
    return get_finished


_WRAPPERS = {METHOD_BIND_METADATA: _wrapped_bind,
             METHOD_GET_FINISHED: _wrapped_get_finished}

# "This method was already ours." A sentinel rather than None so `install()` can
# count ONE reinstall per install call: a reader asking "did the reload re-attach or
# stack a second layer?" wants 1, not 2 (the number of methods this hook wraps).
_REUSED = object()


def _install_method(cls, name: str) -> str:
    """Wrap one method. Returns a reason, None on success, or `_REUSED` when the
    class already carries THIS owner's layer."""
    current = cls.__dict__.get(name)
    if current is None:
        return (f"{cls.__name__}.{name} is not in the class dict, so the fork "
                f"changed shape")
    if _layer_is_ours(current):
        return _REUSED            # counted once per install(), not per method
    if not callable(current):
        return f"{cls.__name__}.{name} is not callable"
    if _layer_is_sibling(current):
        _state["nested"].append(f"{name}({current.__name__})")
    elif getattr(current, OWNER_ATTR, None) is not None:
        return (f"{cls.__name__}.{name} already carries a wrapper from another "
                f"owner: refusing rather than stacking on an unknown layer")
    wrapper = _WRAPPERS[name](current)
    setattr(wrapper, OWNER_ATTR, OWNER_TAG)
    # The original goes on the WRAPPER, not on the class: TWO methods are wrapped,
    # and a single class-level slot would mean the second install overwrote the
    # first one's undo handle - uninstall would then restore `get_finished`'s body
    # under the name `bind_connector_metadata`. An attribute that survives install
    # and is read exactly once, at undo.
    setattr(wrapper, ORIGINAL_ATTR, current)
    wrapper.__module__ = __name__
    setattr(cls, name, wrapper)
    return None


def install() -> dict:
    """Wrap the two methods on the live class. Idempotent; never raises.

    No env-var gate, unlike `worker_diag`/`blockdump`: the per-step cost when the
    tier is unarmed is a couple of `os.environ.get` calls, and the gate that matters
    (`_may_touch_disk`) is the one that can be changed at runtime by the verb. The
    two-switch rule still holds - no root, no arm, no spill.

    Every outcome announces itself. The line goes in a `finally` because the refusal
    paths ARE the ones worth reading: on 2026-10-05 this function succeeded, the
    engine's metadata could not name a file, and nothing in the log said so.
    """
    if _state["installed"]:
        return {"installed": True, "reason": "already installed",
                "class": _state["class"]}
    report = {"installed": False, "reason": None, "class": None}
    try:
        module = sys.modules.get(TARGET_MODULE)
        if module is None:
            report["reason"] = (f"{TARGET_MODULE} is not imported in this process, "
                                f"so there is no worker class to hook")
            _state["skip_reason"] = report["reason"]
            return report
        cls = getattr(module, TARGET_CLASS, None)
        if not isinstance(cls, type):
            report["reason"] = f"{TARGET_MODULE}.{TARGET_CLASS} is not a class"
            _state["skip_reason"] = report["reason"]
            return report
        reused = False
        for name in WRAPPED_METHODS:
            refusal = _install_method(cls, name)
            if refusal is _REUSED:
                reused = True
                continue
            if refusal:
                report["reason"] = refusal
                _state["skip_reason"] = refusal
                return report
        if reused:
            # A reload that found its own layer: counted as a REINSTALL and not also
            # as an install, so `installs` reads "boots that attached a layer" and
            # `reinstalls` reads "installs that found one already there". Two names
            # for one event would let a reader add them and invent a second wrapper.
            _state["reinstalls"] = int(_state["reinstalls"]) + 1
        else:
            _state["installs"] = int(_state["installs"]) + 1
        _state["installed"] = True
        _state["class"] = cls.__name__
        _state["skip_reason"] = None
        # A worker only: this is the process that can see the spill gate, and
        # `worker_diag` is the only writer of the rank file, so the verdict goes to it
        # rather than becoming a fourth file or a second writer.
        register_contribution()
        # THE SHADOW'S VERDICT TRAVELS WITH THE SPILL'S. Both are the worker's own
        # answer about the same spill path, both are relevant only in a process that
        # has this hook, and `worker_diag` remains the only writer of the rank file.
        disk_shadow.register_contribution()
        # THE GPU-DIRECT SHADOW WRITER'S VERDICT TRAVELS WITH BOTH. It reports the same
        # spill path from the same process, and `worker_diag` remains the only writer of
        # the rank file. This is the FOURTH and last section the seam's own
        # `MAX_CONTRIBUTIONS` allows, so adding a fifth means raising that cap in the
        # same change that adds it - a silent refusal at register time is how a section
        # goes missing.
        disk_gpu_shadow.register_contribution()
        report.update({"installed": True, "class": cls.__name__,
                       "reason": (f"{cls.__name__}.{METHOD_BIND_METADATA} and "
                                  f"{cls.__name__}.{METHOD_GET_FINISHED} now spill "
                                  f"on store completion; the tier stays inert until "
                                  f"a root is set AND the arm is raised")})
        return report
    except Exception as exc:  # noqa: BLE001 - installation never breaks a boot
        snapshot._error(exc, "disk_store install")
        _state["fallbacks"] = int(_state["fallbacks"]) + 1
        _state["errors"] = int(_state["errors"]) + 1
        report["reason"] = f"install raised {type(exc).__name__}"
        _state["skip_reason"] = report["reason"]
        return report
    finally:
        # `finally`, not the success path: the verdict a reader needs is the one that
        # says NO. Contained by `emit_boot_line` itself.
        emit_boot_line()


def ensure_installed(connector=None) -> dict:  # noqa: ARG001 - the handshake's object
    """The worker-handshake entry point: install, ignoring the connector it is handed.

    `engine_side._on_worker_report` calls every listener as `hook(connector)`, so this
    wrapper exists for that signature - and the argument is genuinely unused, because
    this module resolves the worker CLASS BY NAME from the already-imported module
    rather than reaching into an instance it was handed. Accepting the object and
    ignoring it is the honest version of the contract; the siblings (`worker_diag`,
    `blockdump`) take the same optional argument for the same reason.
    """
    return install()


def boot_note() -> str:
    """One line for the boot log: what this hook can do on THIS boot, in words.

    A log line is the only place the two halves of the posture meet - "the hook is on
    the class" and "this boot's engine cannot name a file" - and an operator reading a
    boot log should not have to call a route to learn that the tier is inert, or why.
    """
    counters = _state
    fields = engine_fields()["verdict"]
    if not counters["installed"]:
        posture = ("NOT installed ("
                   f"{counters['skip_reason'] or 'never attempted here'})")
    else:
        posture = (f"installed on {counters['class']}, wrapped over "
                   f"{len(counters['nested'])} sibling layer(s)")
    gate = "armed" if disk_spill.is_armed() else "disarmed"
    if disk_spill.disk_root() is None:
        gate += ", NO root configured"
    return (f"{posture}; engine_fields={fields}, tier {gate}. Inert until a root is "
            f"set AND the arm is raised, and inert on an engine whose metadata "
            f"carries no store_keys: a spill needs a key to name its file.")


def uninstall() -> dict:
    """Put the fork's methods back exactly. Test teardown and a clean undo."""
    report = {"uninstalled": False, "reason": None}
    try:
        module = sys.modules.get(TARGET_MODULE)
        cls = getattr(module, TARGET_CLASS, None) if module else None
        if not isinstance(cls, type):
            report["reason"] = "the worker class is not imported here"
            return report
        restored = 0
        for name in WRAPPED_METHODS:
            current = cls.__dict__.get(name)
            if current is None or not _layer_is_ours(current):
                continue
            original = getattr(current, ORIGINAL_ATTR, None)
            if callable(original):
                setattr(cls, name, original)
                restored += 1
        _state["uninstalls"] = int(_state["uninstalls"]) + 1
        _state["installed"] = False
        report.update({"uninstalled": True,
                       "reason": f"{restored} of {len(WRAPPED_METHODS)} methods "
                                 f"restored to the fork's own"})
        return report
    except Exception as exc:  # noqa: BLE001 - a teardown never raises
        snapshot._error(exc, "disk_store uninstall")
        report["reason"] = f"uninstall raised {type(exc).__name__}"
        return report


def hook_section(include_fingerprint: bool = True) -> dict:
    """The hook's STATE. A layer a snapshot reports, so it must never raise or lie.

    `include_fingerprint=False` drops the nested fingerprint block for a caller that
    publishes the report at its own top level (`op:status` does): one payload, one
    copy of one identity. A duplicated fingerprint is not harmless - the day the two
    copies disagree, a reader has to decide which one named the file.
    """
    return {"role": ROLE, "writer": WRITER_TAG,
            "protocol_version": PROTOCOL_VERSION,
            "installed": bool(_state["installed"]),
            "installs": int(_state["installs"]),
            "reinstalls": int(_state["reinstalls"]),
            "uninstalls": int(_state["uninstalls"]),
            "class": _state["class"], "skip_reason": _state["skip_reason"],
            "refusals_total": int(_state["refusals_total"]),
            "last_refusal": _state["last_refusal"],
            "wraps": list(_state["nested"]),
            "methods": list(WRAPPED_METHODS),
            "binds": int(_state["binds"]),
            "steps_idle": int(_state["steps_idle"]),
            "completions": int(_state["completions"]),
            "slots_offered": int(_state["slots_offered"]),
            "spill_queued": int(_state["spill_queued"]),
            "delegated_refusals": int(_state["delegated_refusals"]),
            "slots_capped": int(_state["slots_capped"]),
            "scan_skipped": int(_state["scan_skipped"]),
            # HOW MUCH OF WHAT THIS PROCESS SPILLED CAN EVER BE SERVED. The live form
            # of a number that took four journals and a Python session to produce
            # (20.0% of attention spans arriving without the align state that would let
            # them anchor a run); `span_shapes` says what authority the split was read
            # under, so a zero here is either "nothing orphaned" or "could not tell",
            # and the payload says which.
            "spill_spans_classified": int(_state["spill_spans_classified"]),
            "spill_spans_complete": int(_state["spill_spans_complete"]),
            "spill_incomplete_state": int(_state["spill_incomplete_state"]),
            "spill_incomplete_attention": int(_state["spill_incomplete_attention"]),
            # THE OTHER END OF THE 0.2.64 PIPE: what reached the store, against what
            # the scheduler's mask patch marked reachable (`mamba_anchor` below). Read
            # the two together, and never as a per-block cause-and-effect ratio - the
            # counter's own comment in `_state` says why the join is not representable.
            "anchor_spills_completed": int(_state["anchor_spills_completed"]),
            "mamba_anchor": mamba_anchor.section(),
            "span_shapes": span_shape_view(),
            "refused": dict(_state["refused"]),
            "refusal_reasons": list(REFUSAL_REASONS),
            "fallbacks": int(_state["fallbacks"]), "errors": int(_state["errors"]),
            "worker_fingerprint": fingerprint_report()
            if include_fingerprint else "see the top level",
            "engine_fields": engine_fields(),
            "retained": retained_view(),
            "polls": int(_state["polls"]),
            "poll_suppressed": int(_state["poll_suppressed"]),
            "pending_reads": int(_state["pending_reads"]),
            "pending_malformed": int(_state["pending_malformed"]),
            "receipt_errors": int(_state["receipt_errors"]),
            "requests_written": int(_state["requests_written"]),
            "fulfilled": int(_state["fulfilled"]),
            "last_rank": _state["last_rank"], "last_reason": _state["last_reason"],
            "last_path": _state["last_path"], "last_error": _state["last_error"],
            "caps": {"retained_events": MAX_RETAINED_EVENTS,
                     "slots_per_event": MAX_SLOTS_PER_EVENT,
                     "slots_per_tick": MAX_SLOTS_PER_TICK,
                     "handled_events": MAX_HANDLED_EVENTS,
                     "fingerprint_retry_seconds": FINGERPRINT_RETRY_SECONDS,
                     "ls_files": MAX_LS_FILES,
                     "poll_interval_seconds": POLL_INTERVAL_SECONDS}}


DISK_STORE_PROCESS_SCOPE_NOTE = (
    "PER-PROCESS: the wiring counters in this block (wraps, binds, steps_idle, "
    "completions, slots_offered, spill_queued, refusals_total, fallbacks, and the "
    "`spill` sub-block) belong to THIS process's own module state. The hook wraps the "
    "scheduler-side manager, so they are non-zero only in the process that steps the "
    "tokens; on the API process - and on any process that never served a step - they "
    "are zero BY CONSTRUCTION, which is not a claim that the tier spilled nothing. "
    "For the fleet's numbers read the rank rollup: `disk_fill` (in /snapshot and "
    "/status), which sums each rank's own counters from the rank files it wrote, names "
    "the ranks it summed, and says how stale each one was.")


def section() -> dict:
    """The snapshot section: the wiring's state, and whether THIS BOOT can name a
    file. Reads no files and walks no directory, like `disk_spill.section()`.

    Deliberately does NOT restate the spill mechanism's counters as its own: `spill`
    is a SUBSET of `disk_spill.counters()` and `refused` is this hook's own refusals.
    The two fail differently - "no root configured" is disk_spill's, "the engine
    never sent a key" is this module's - and a reader that conflates them will chase
    the wrong one.
    """
    state = disk_spill.counters()
    armed = disk_spill.is_armed()
    root = disk_spill.disk_root()
    fields = engine_fields()
    return {"role": ROLE, "writer": WRITER_TAG,
            # THE LABEL THAT COST AN OPERATOR A NIGHT OF READING ZEROS (2026-10-22).
            # Every number in this block is THIS process's own wiring state, and the
            # hook runs where the scheduler runs - so on the API process, and on any
            # process that never stepped a token, they are zero BY CONSTRUCTION. The
            # fleet's numbers are in `disk_fill` beside it. Say this in the payload
            # rather than in a README a reader has already given up on.
            "scope": DISK_STORE_PROCESS_SCOPE_NOTE,
            "protocol_version": PROTOCOL_VERSION,
            "installed": bool(_state["installed"]),
            "class": _state["class"], "skip_reason": _state["skip_reason"],
            "refusals_total": int(_state["refusals_total"]),
            "last_refusal": _state["last_refusal"],
            "wraps": list(_state["nested"]),
            "methods": list(WRAPPED_METHODS),
            "armed": armed, "configured": bool(state["configured"]),
            "capability": disk_spill.capability(),
            "root": root, "engine_fields": fields,
            "worker_fingerprint": fingerprint_report(),
            "binds": int(_state["binds"]),
            "steps_idle": int(_state["steps_idle"]),
            "completions": int(_state["completions"]),
            "slots_offered": int(_state["slots_offered"]),
            "spill_queued": int(_state["spill_queued"]),
            "delegated_refusals": int(_state["delegated_refusals"]),
            "slots_capped": int(_state["slots_capped"]),
            # The completeness tally, again (see `hook_section`): three classes and
            # their denominator, so a snapshot and an `op:status` answer the same
            # question with the same numbers.
            "spill_spans_classified": int(_state["spill_spans_classified"]),
            "spill_spans_complete": int(_state["spill_spans_complete"]),
            "spill_incomplete_state": int(_state["spill_incomplete_state"]),
            "spill_incomplete_attention": int(_state["spill_incomplete_attention"]),
            # THE OTHER END OF THE 0.2.64 PIPE, in the door a dashboard polls: what
            # reached the store against what the scheduler's mask marked reachable
            # (`mamba_anchor` below). Same keys in both doors, so a verb and a
            # snapshot cannot disagree about a counter.
            "anchor_spills_completed": int(_state["anchor_spills_completed"]),
            "mamba_anchor": mamba_anchor.section(),
            "refused": dict(_state["refused"]),
            "refusal_reasons": list(REFUSAL_REASONS),
            "fallbacks": int(_state["fallbacks"]),
            "spill": {key: state[key] for key in
                      ("spilled", "accepted", "requests", "drops", "bytes_written",
                       "evictions", "errors", "group_collisions",
                       "journal_records", "journal_failures")},
            "retained": retained_view(),
            "spill_one": {"requests_written": int(_state["requests_written"]),
                          "fulfilled": int(_state["fulfilled"]),
                          "poll_interval_seconds": POLL_INTERVAL_SECONDS,
                          "ttl_seconds": REQUEST_TTL_SECONDS},
            "wiring": ("THE ENGINE CALLS THIS. `bind_connector_metadata` remembers "
                       "each step's (slot, block hash) pairs and `get_finished` spills "
                       "them when the store event completes; both compose with "
                       "worker_diag and blockdump and hand the fork's return value "
                       "back untouched. Names come from the fingerprint THIS worker "
                       "builds for itself (its own measured slot width, its own "
                       "rank), because one scheduler digest cannot be four ranks' "
                       "digests; store_fingerprint is a cross-check only, and "
                       "disagreeing with it is a counted refusal. Bytes are read on "
                       "disk_spill's writer thread, never on an engine thread. "
                       "Inert until a root is set AND the arm is raised; "
                       f"engine_fields={fields['verdict']} on this boot."),
            "note": "the SSD tier's wiring: what it can name, and why it cannot yet."}
