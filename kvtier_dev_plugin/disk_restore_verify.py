"""THE RESTORE-SIDE VERIFIER: are the bytes now in a GPU window the bytes in the file?

WHAT THIS ASKS, AND WHY IT IS A DIFFERENT QUESTION THAN `disk_shadow`'s
-------------------------------------------------------------------------------
`disk_shadow` (0.2.36) watches the STORE direction: a block the engine has just written
to the GPU is compared with the bytes the spill path decided to write, so it proves that
what we SHIPPED is what the GPU HELD. This module watches the RESTORE direction, which
is where the correctness risk actually lives on the live box: the offer path claims a
prefix is on SSD, the reader pulls a file into a bounce buffer, the plugin memmoves that
buffer into a CPU KV slot, and then the FORK's own `start_load_kv` DMAs that slot into a
GPU block. Every one of those steps has an index in it. One index off by one and the
engine serves a window that belongs to a different token boundary - with every name in
the file system still perfectly well-formed, because a one-slot shift moves the
index-aligned lists together.

THE VERDICT THIS PROVES, per sampled restored entry:

file[0 : W_g]  ==  gpu_window[gpu_block_id][0 : W_g]

where `gpu_window[gpu_block_id]` is `base + gpu_block_id * bytes_per_block` and `W_g` is
THE GROUP'S OWN block width (`arena.block_bytes_by_group[group]`, i.e.
`spec.page_size_bytes` - `worker_diag._block_bytes_by_group`, the same function that
publishes that field, so this verifier's `W_g` and the rank file's cannot drift apart).

THE TAIL IS REFUSED, AND THAT IS NOT A CAVEAT, IT IS AN ASSERTION. All six groups alias
from byte 0 of the SAME region (`worker_diag._block_bytes_by_group`'s own words:
`block_bytes[3] == 1_638_400 < block_bytes[0]` on the live PP0), so the window is as
wide as the WIDEST group and a narrow group owns only its leading `W_g` bytes. The rest
of the window belongs to the other groups and holds whatever THEY last left there -
stale by construction, and stale in a way that is correct. Comparing `[W_g,
bytes_per_block)` would produce false mismatches by construction, on every entry of
every narrow group, forever. So the comparison is built from slices that do not contain
the tail: the file read is `read(W_g)` (the tail is never even read from disk) and the
device bytes are sliced to `[:W_g]` before the compare. `tail_bytes_excluded` in
`last_mismatch` publishes how much was deliberately not looked at, so a reader can tell
"we compared 1,638,400 bytes and they matched" from "we compared 11,468,800".

THE FIVE THINGS IT CHECKS, and what each one would mean if it failed
------------------------------------------------------------------------------- 1.
`file_size == bytes_per_block` - the rank's own slot width (`disk_spill.
measured_slot_bytes` / the fingerprint's `total_bytes_per_block`; 11,468,800 on PP0 and
12,902,400 on PP1 are DIFFERENT NUMBERS, so a file that is the other rank's width is a
PP-crossing bug and `file_not_window_wide` is how it shows up). A file that is not
window-wide is REFUSED, not compared: a width disagreement has an explanation and a
comparison against it explains nothing. 2. `W_g <= file_size` (`group_wider_than_file`).
A group wider than the file it is named in is a geometry contradiction; comparing would
read past EOF and call the zeros a mismatch. 3. the leading-`W_g` byte equality, above,
with staleness classified by a re-read. 4. THE BINDING ORACLE. The name in the path is
derived from a hash, so name-vs-name is blind. The oracle recomputes the group's
expected `key16` for the GPU index the fork ACTUALLY used, from the engine's own hash
derivation - `resolve_block_hashes(request.block_hashes, hash_block_size, group_size)`
and `make_block_hash_with_group_id(digest, group)`, the same two helpers the fork uses
for the same purpose (`manager.py:897-900`, and the per-group view at
`manager.py:1591-1601`) and the same two `disk_restore._key_helpers()` resolves. It then
requires the recomputed `key16` to equal the `key16` in the restore path's file name.
REACHABILITY IS SAID HONESTLY, NOT FAKED: a WORKER never sees a request. The fork's
worker metadata carries `load_event_to_reqs` - request IDs - and the file that defines
it says so out loud: "Job->req_id translation is handled by the scheduler-side manager
... the worker never knows about request identities"
(`vllm/v1/simple_kv_offload/metadata.py`). So `block_hashes` is not reachable from this
process on this fork, and every entry counts `oracle_unavailable`. That is a counted,
named, published outcome, not a skip: the counter is how an operator learns the binding
check is not running, and `oracle_note` names the ONE thing that would make it run -
carrying the per-entry hash view to the worker, or arming the seam below from a process
that has requests. The recomputation itself is real code, exercised by tests, behind
`set_oracle()`: a fork patch that carries the view, or a harness, hands this module a
provider and the check goes live with no change here. 5. THE TRUNCATION SHAPE
(`boundary_entry_missing`). The fork truncates each group's hit list with `n_take_g =
cdiv(num_external_tokens, g_block_size)` (`manager.py:445-448`), and a `boundary_state`
group's hit list is `[null_block] * (n - 1) + [one real state block]`
(`disk_restore._group_contract`'s cite of `single_type_kv_cache_manager.py:1591-1607`).
So a truncation that lands before the run's last block drops the boundary group's ONLY
real entry while every positional group keeps its entries - a restore of a prefix whose
state block never arrived, which is the Gate B shape. Worker-side, a boundary group is
one whose spec says `mamba_cache_mode == "align"` (a field only `MambaSpec` carries,
`kv_cache_interface.py:992-998`, and the same value `disk_restore._group_contract` reads
as the fork's own reason). A batch that restored entries for some groups and NOTHING for
a known boundary group is counted. THE GRANULARITY IS BATCH-LEVEL AND SAY SO: one batch
can hold spans from several requests, so a boundary entry that DID arrive for one of
them masks the one that did not. It under-counts; it cannot over-count in the way that
matters, because "no entry for a group this boot knows is align-mode, in a batch that
restored something" is a fact about that batch.

WHERE THE WORK HAPPENS, AND WHAT THE STEP THREAD PAYS
------------------------------------------------------------------------------- Nothing
in this module runs file I/O or a device copy on the engine's threads. The step thread
pays ONE call, at the apply point in `disk_restore._worker_before_load`, after
`span_shape` has made `failed` final: enablement, a counter, a sampling cursor, a
bounded scan, a handful of int comparisons, and at most `MAX_RETAINED_PER_STEP` small
dicts of ints and one path string. It reads no file, touches no tensor, and issues no
CUDA call - so there is nothing there to block a step, and the interval floor and the
per-pass bound apply to the thread that DOES the work.

The work runs on this module's single daemon thread (the `disk_gpu_shadow` idiom: one
thread, lazily started, every pass wrapped in `_contained` so a raise can never kill
it). A pass reads at most `MAX_CHECKS_PER_PASS` retained entries, honouring
`MIN_INTERVAL_SECONDS` between passes, so the worst case this module adds to a live box
is `MAX_CHECKS_PER_PASS` file reads of `W_g` bytes and the same number of ONE-BLOCK
device copies, once per `MIN_INTERVAL_SECONDS`. That is deliberately LESS device traffic
than `disk_shadow` already accepts in production (`MAX_BLOCKS_PER_STEP` per bind
window), and the copies are window-wide exactly as theirs are.

THE GPU READ IS `disk_shadow`'S, REUSED, NOT RE-IMPLEMENTED. Everything the verifier
reads off the device goes through `disk_shadow._read_gpu_slot`, which is
`_gpu_region_map` + `_copy_slot_to_host` (`narrow(0, block, 1).contiguous().cpu()`) +
`_host_landed` + `disk_spill.read_slot_bytes`. There is no second device-copy mechanism
here and no `ctypes` read of a device address - the guard that turns a would-be SIGSEGV
into a `gpu_read_failed` (`disk_shadow._host_landed`) is inherited by construction, and
a test that replaces `disk_shadow._copy_slot_to_host` redirects this module too, which
is the only reason the device half of these checks can be tested at all.

WHY IT WAITS FOR THE DMA. The apply point runs BEFORE the fork's `start_load_kv` issues
the CPU->GPU copy, so reading the GPU block there would compare against bytes that have
not arrived yet. The fork records the copy as a CUDA event tagged with
`metadata.load_event` (`worker.py:263-275`) and moves `_load_hwm` past that index when
the event completes (`worker.py:365-381`). So a retained entry is verified only once
`worker._load_hwm >= event` - a real witness that the DMA landed, not a sleep. A worker
that exposes no `_load_hwm` at all is `no_load_hwm`, and an entry whose event never
completes is `dropped_waiting_dma`; neither is ever compared, because comparing before
the DMA is the one way to invent a mismatch.

A DISAGREEMENT IS A QUESTION. Only `mismatch_stable` (the GPU's own bytes did not move
between the two reads) is a layout candidate. A GPU block is handed out again by the
scheduler at any time, so a mismatch whose re-read differs is the engine having reused
the destination - `mismatch_gpu_changed_on_recheck`, an expected race, not corruption.
That classification is the whole reason the re-read exists, and it is `disk_shadow`'s,
kept identical so the two witnesses cannot disagree about what a mismatch means.

DEFAULT OFF, AND IT STAYS OFF. This reads real KV bytes off the device and opens spill
files for reading on a live box; `VLLM_E2_TIER_DISK_RESTORE_VERIFY` must name it,
exactly as `ENV_DISK_SHADOW` and `ENV_DISK_GPU_SHADOW_WRITES` must. The POST op latches
the process that answers and writes no control file (this module opens no file handle
for writing anywhere - its whole safety story). It never changes a restore decision: it
reads the batch the apply point already decided on, and cannot prune, reorder or re-file
anything.

THE SCHEDULER-SIDE ROLE (0.2.40, MOVED at 0.2.41): THE CHECK THAT CAN SEE A ONE-SLOT
SHIFT
------------------------------------------------------------------------------- The
worker-side oracle above is unreachable where it lives, and the reason is structural
(`metadata.py`: a worker has no requests). The SCHEDULER side of the same plugin does
have requests, and it is the side that DECIDES the pairing. So this module has a second
role, armed by the SAME `VLLM_E2_TIER_DISK_RESTORE_VERIFY` (the EngineCore inherits the
environment just as the workers do - one drop-in line arms both roles, and every counter
name says which role produced it: every scheduler-side counter begins `scheduler_`).

THE COMPARISON, per sampled offered entry:

key16 in the file name we offered for the pairing (GPU block G, CPU slot C) ==  key16
of `view_g[j]` packed with g,

where (g, j) is the EXPECTATION the fork itself created when it paired G with C:
group `g`, and position `j` in the request's own GPU block table for that group. The
expectation is captured at ALLOC time (`observe_alloc`, hung on
`update_state_after_alloc`, the one method where the fork decides the pairing: its loop
IS `group_gpu_ids[gpu_ext_start + i]`, `manager.py:460-475`), and compared at NAME-EMIT
time (`observe_offer`, hung on `build_connector_meta` via `disk_restore
.build_load_paths`). `view_g` is the engine's own `resolve_block_hashes(request
.block_hashes, hash_block_size, group_block_sizes[g])` - re-derived on this module's
thread, never on the step thread. THE VIEW IS THE ONLY THING INDEXED, AND IT IS NOT A
LIST ON THIS BOX (0.2.42): that call answers `list[BlockHash] |
BlockHashListWithBlockSize` (`kv_cache_utils.py:2850`) and the lazy arm is NOT a
`Sequence`, so a shape check for a sequence refused it and the witness ran with
`scheduler_checked = 0`; and its index arithmetic is scaled (`view[i]` is
`raw[i * (group_size // hash_block_size) + stride]`, 80-to-1 with this box's 40-token
hashes against 3200-token groups), so indexing the raw list instead of the view compares
digests for a DIFFERENT TOKEN RANGE, not a near miss. See `_scheduler_view`. That is the
whole point: a one-slot shift moves
`load_cpu_blocks` and `load_gpu_blocks` together, so every comparison BETWEEN them is
blind, and the answer has to come from outside them - and the ONE moment "which GPU
block for which span" is actually decided is the alloc loop, not the metadata walk.

WHY `kv_connector_block_state` WAS THE WRONG OBSERVATION POINT (the 0.2.40 finding,
live in production, not theoretical). 0.2.40 read the request's GPU block table from
`scheduler_output.kv_connector_block_state` at the EMIT point. That table is THIS
STEP'S (`sched/scheduler.py:1412-1425`, cleared at `:1497`), but a load state in
`_reqs_to_load` can be emitted in a LATER step than the one that allocated it - so the
request id legitimately is not in the current step's `req_ids`, and every sampled entry
hit `block_state_does_not_cover_request`: the oracle shipped decorative, checked
nothing, and only stayed trustworthy because it counted its own blindness instead of
reporting a green zero. The alloc hook fixes this structurally: the expectation is
captured when the pairing is MADE, so it does not matter when - or over how many steps
- the load is emitted. The block-state channel is NOT kept as a second source: it reads
the same fork facts at the later, wrong moment, and its one independent signal (the
request's current-step table) adds a comparison the alloc-time expectation already
answers at the authoritative moment. Its counters are gone with it, and
`scheduler_oracle_hook` in the published section names the hook that runs.

WHAT THE ALLOC HOOK READS (all of it arguments or manager fields, nothing invented): in
`update_state_after_alloc(self, request, blocks, num_external_tokens)` -
`request.block_hashes` (the engine's own hash list, referenced, never copied, never
hashed on the step thread), `blocks.get_block_ids()` (the per-group GPU block-id tables
for THIS request, the very lists whose fork indexes), and the manager's
`_pending_cpu_hits[req_id]` PEEKED (a `.get`, never the fork's `.pop`) for the CPU hit
blocks and `num_computed_tokens` - plus `hash_block_size`, `group_block_sizes` and
`prefix_cacheable_group_ids`, the same fields the fork's own loop reads at
`manager.py:433-475`. The observer REPLICATES the fork's pairing loop (cdiv truncation,
the prefix-cacheable gate, the `is_null` skip that keeps the index) so the captured
(g, j) is the fork's decision, not a guess at it. It mutates NOTHING and its whole
loop is bounded by `MAX_EXPECTATIONS_PER_ALLOC` per call and `MAX_SCHED_EXPECTATIONS`
in the map (LRU, oldest evicted).

THE EXPECTATION LIFECYCLE IS COUNTED, BECAUSE AN ORACLE THAT SILENTLY HAS NOTHING TO
COMPARE IS THE FAILURE MODE THIS RELEASE EXISTS TO KILL. `scheduler_allocs_observed`
counts allocs that carried a real external load; `scheduler_expectations_captured` the
pairings written into the map; `scheduler_expectations_consumed` the sampled emitted
entries that matched one; `scheduler_expectations_dropped` the ones EVICTED UNCONSUMED
(a load abandoned, a request finished, or plain churn - the vast majority of pairings
have no disk file and are never expected to be consumed); `scheduler_expectations_
capped` the per-alloc bound hit. An emitted entry whose (GPU block, CPU slot) pairing
has NO captured expectation is a NAMED `scheduler_oracle_unavailable`
(`no_captured_expectation`), never a silent skip: a check that cannot run says so in a
counter an operator can see.

WHAT A FAILURE MEANS, and it is three different things with three different fixes, so
the classification is a name and not a number: `neighbour_index` (the name carries the
digest of the block at index j +/- 1..N: a one-slot shift, the finding this role exists
for), `other_group` (the digest is real but belongs to another group's view of the same
index, or the group suffix INSIDE the name contradicts the group THE FORK PAIRED at
alloc time), and `unknown`. A name's OWN group suffix outranks the digest comparison,
so a suffix that contradicts the pairing is a mismatch even when the digest part
matches exactly: the file is named by BOTH halves (`disk_layout.build_slot_path` writes
the group into the name), so a `key16`-only equality would call a wrong group a match.
Like the worker side's `mismatch_stable`, a scheduler mismatch is re-derived before it
is called stable: `scheduler_mismatch_stable` means the same captured inputs re-derived
the same expected key, so the disagreement is not a half-mutated `block_hashes` view;
`_unstable` means the engine's own inputs moved under us and the sample is not
evidence.

WHAT THE ENGINE THREADS PAY, AND WHAT THEY DO NOT. `observe_alloc` and `observe_offer`
are the only calls on engine threads. `observe_alloc` (once per SCHEDULED request - the
scheduler calls `update_state_after_alloc` unconditionally, `scheduler.py:1228` - and
it returns at once for the overwhelming majority that carry no external load) pays,
for a real external load: enablement, a peek at the manager's own pending map, and a
bounded int-only loop replicating the fork's pairing - the fork runs the same loop one
layer below, so this doubles a walk the step was already taking, bounded by
`MAX_EXPECTATIONS_PER_ALLOC`. NO hashing, no file I/O, no GPU work, no list copy
beyond the small per-group id ranges it indexes. `observe_offer` pays enablement,
counters, a sample decision, one dict lookup per sampled entry and at most
`MAX_RETAINED_PER_STEP` small dicts. `resolve_block_hashes` +
`make_block_hash_with_group_id` + the key split + the neighbour walk all run on the
module's ONE existing daemon thread, off the scheduler's way, bounded by
`MAX_CHECKS_PER_PASS` and the same `MIN_INTERVAL_SECONDS` floor. The references the
step thread hands that thread are `block_hashes` list references and ints - no tensor,
no file handle; the expectation map holds at most `MAX_SCHED_EXPECTATIONS` entries and
the pending ring at most `MAX_RETAINED_JOBS` records for at most `JOB_MAX_AGE_SECONDS`.

WHY IT LIVES IN THIS MODULE AND NOT A NEW ONE: `worker_diag.MAX_CONTRIBUTIONS` is 5 and
the contribution allow-list is EXACTLY those five names, so a sixth section is refused
at register time and an operator would read a rank file that silently lacks the verdict.
The scheduler role therefore reports into the same `disk_restore_verify` section, with
every one of its counters prefixed so the two roles can never be added together by
accident, and with `roles` and `oracle_rule` in the payload saying WHICH process
produced which number - a sentence rather than a comment, because the two roles'
`oracle_unavailable` counters mean opposite things (a worker's is a permanent fact of
the fork and says nothing; a scheduler's says the addressing check is not running where
it COULD run) and a reader holding one payload must not have to infer which one he has.

One published shape note, because it was a deliberate subtraction: there is no
`scheduler_last_mismatch` key. `scheduler_mismatch_samples` is the ring of the newest
`MAX_SCHED_MISMATCH_SAMPLES` verdicts and its last element IS the newest, and publishing
the same object twice cost 440 B of a section whose budget the seam ENFORCES by refusal.
The samples carry numbers and no sentence for the same reason - the sentence is the
`scheduler_mismatch:<classification>` line in `reasons`, once per finding, in full.

WHAT IT NEVER DOES
------------------------------------------------------------------------------- * writes
any file, anywhere, ever (the only file operation is a read of a path the restore path
itself already chose to read); * mutates the engine: it reads a request, a pending-hit
tuple it PEEKS (a `.get`, never the fork's `.pop`), a `KVCacheBlocks` it only calls
`get_block_ids()` on, and a metadata it was handed, and returns nothing to any of them.
It cannot prune, reorder or re-file an entry; the offer-point call site is after the
paths have been written, and the alloc-point call site runs before the fork's body and
cannot change what that body decides; *
changes a restore outcome - `observe_apply` cannot raise, cannot prune and is called
after every decision the apply point makes; * blocks the step thread (no I/O, no CUDA,
no torch symbol, no unbounded loop); * imports `torch`/`vllm`/`cuda`/`numpy` (the same
package invariant `tests/test_snapshot.py::test_no_heavy_imports_in_the_read_side`
enforces; the device read needs no symbol, because it is `disk_shadow`'s method call); *
keeps a reference to a worker it cannot prove is alive (a weakref; a worker that is gone
is `worker_gone`, and the pass stops rather than copying from freed memory).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import weakref

from . import (
    config,
    disk_layout,
    disk_shadow,
    disk_spill,
    disk_store,
    worker_diag,
)

logger = logging.getLogger("kvtier_dev_plugin.disk_restore_verify")

MODULE = "disk_restore_verify"
OWNER_TAG = "kvtier-dev-plugin/disk-restore-verify-1"

# --- THE BOUNDS ----------------------------------------------------------------
# One restored entry in SAMPLE_EVERY_N_RESTORES. The population is the entries the apply
# point ACTUALLY applied (not offered, not bound), so the rate is a rate over restores
# that really moved bytes. At 32, a 10,000-restore boot samples ~312 entries - enough to
# catch a systematic one-slot shift on the first hundred and little enough that a
# 1-in-32 sample of a 45-entry batch is one entry, not four.
SAMPLE_EVERY_N_RESTORES = 32

# The step thread's bound. Retention is a dict of ints; this cap exists so a batch of
# forty-five entries cannot make the module's queue its own forty-five jobs before the
# sampler has a chance to thin them.
MAX_RETAINED_PER_STEP = 2

# Retained-but-unverified entries. Anything past this drops its OLDEST entry: the oldest
# is the one whose GPU block has had the longest to be handed out again, so it is the
# most likely to answer `..._on_recheck` and the least likely to be evidence.
MAX_RETAINED_JOBS = 16

# Entries verified per pass, and per-pass re-reads. At PASS_TICK_SECONDS this is a hard
# ceiling of four window-wide device copies and four bounded file reads per second, on a
# box that is already serving traffic.
MAX_CHECKS_PER_PASS = 2
MAX_RECHECKS_PER_PASS = 1

# The pass floor, and the thread's tick. The floor is what makes a burst of steps cheap:
# retention is per step, verification is not.
MIN_INTERVAL_SECONDS = 0.5
PASS_TICK_SECONDS = 0.5

# How long an entry may wait for its load event to complete. Past this the DMA is not
# coming (the fork's own restore deadline is `disk_restore.RESTORE_DEADLINE_SECONDS`,
# well under it), and a stale entry's GPU block is somebody else's by now.
JOB_MAX_AGE_SECONDS = 30.0

# The apply point's scan bound, on the population rather than the batch: a batch is
# already bounded by `disk_read.MAX_JOBS_PER_BATCH`, and this is the belt that keeps a
# future, larger batch from putting an unbounded loop on the step thread.
MAX_ENTRIES_SCANNED = 64

MAX_REASON_SAMPLES = 6
MAX_REASON_CHARS = 120

# --- THE SCHEDULER-SIDE ROLE'S BOUNDS ----------------------------------------
# How far the classification looks for the digest a name ACTUALLY carries. A shift of
# more than this is still reported as a MISMATCH (that is the finding); the window only
# decides whether the sample can also say BY HOW MUCH, which is the difference between
# "the start block is off by one" and "the start block is off, somewhere".
NEIGHBOUR_WINDOW = 2

# Published mismatch samples: NEWEST LAST, oldest dropped. There is deliberately no
# separate `scheduler_last_mismatch` key - the ring's last element IS the newest
# verdict, and publishing it twice cost 440 B of a 4096 B section budget to say the same
# thing. TWO is the cap, and the arithmetic behind it is this section's own: the
# mandatory content (sixty-four counters, two roles' oracle states, and the sentences
# that tell the roles apart) measures most of 3.8 KiB, so the illustration a reader
# gets is what is
# left under `MAX_SECTION_BYTES` - and a section the seam REFUSES for being 100 B fat
# carries NO samples at all, which is the failure mode the cap exists to prevent. The
# counters, which are the load-bearing numbers, see every single mismatch.
MAX_SCHED_MISMATCH_SAMPLES = 2
MAX_SCHED_ID_CHARS = 32            # a request id, clipped the way `disk_restore` clips

# THE EXPECTATION MAP (0.2.41). `MAX_EXPECTATIONS_PER_ALLOC` bounds what ONE
# `update_state_after_alloc` may write (a 128k-token restore over a 1600-token group
# would otherwise put eighty dicts on one alloc, and the map is LRU-thrashed by exactly
# such steps); hitting it is a bound, counted once per alloc as
# `scheduler_expectations_capped`, never a finding. `MAX_SCHED_EXPECTATIONS` is the
# global LRU cap: the oldest entry goes first, and if it goes UNCONSUMED that is the
# `scheduler_expectations_dropped` counter - the named fact that the pairing this map
# remembered was abandoned, finished, or outrun. At a few hundred bytes a record (ints,
# one shared `group_block_sizes` copy, one `block_hashes` LIST REFERENCE) the cap is a
# fraction of a megabyte, held for at most a few seconds of step churn.
MAX_EXPECTATIONS_PER_ALLOC = 256
MAX_SCHED_EXPECTATIONS = 2048

# The first-difference scan, and why it is chunked: `memcmp` per 4 KiB block, so the
# only Python-level byte loop runs inside the ONE chunk that differs. A full-window
# Python scan of 11.4 MB to produce an offset is not worth the offset.
MAX_DIFF_SCAN_BYTES = 262_144

# The published lists, and the section's own byte budget. See `_fit`.
MAX_BOUNDARY_GROUPS_PUBLISHED = 8
TRIM_BOUNDARY_GROUPS = 4
TRIM_REASON_CHARS = 42
# 3072 was the budget while this section described ONE role; 3800 was the 0.2.40
# number (the scheduler role lives in THIS section because
# `worker_diag.MAX_CONTRIBUTIONS` is 5 and its allow-list is exactly five names, so a
# sixth is refused at register time and the verdict would be SILENTLY ABSENT from the
# rank file). The 0.2.41 expectation lifecycle and the hook-name line pushed the
# mandatory content to ~3660 B, which under 3800 left no room for even one mismatch
# sample - the trim ladder then shed the section's own findings as a structural
# certainty, which is a different way of shipping a decorative witness. The new number
# keeps 176 B of margin under the seam's own `MAX_CONTRIBUTION_BYTES` = 4096, the only
# ceiling that can make this key disappear; the trim ladder protects that margin when
# a worker's live geometry and six fat reasons arrive.
MAX_SECTION_BYTES = 3920

# THE CEILING ON THE BOUNDARY NOTE ITSELF. The note is published in this module's rank
# section, whose own budget above has 176 bytes of margin to the seam's cap - so the
# evidence `_boundary_groups` gained is CLIPPED, not hoped for, and names at most
# `BOUNDARY_NOTE_CLASSES` spec classes.
BOUNDARY_NOTE_CHARS = 120
BOUNDARY_NOTE_CLASSES = 2

# The fork's own reason for calling a group's blocks positionally unstable. It is read
# through `disk_store.resolve_mamba_cache_mode`, NOT by a `getattr` on the group's spec:
# only `MambaSpec` declares the field (`kv_cache_interface.py:998`), and a worker's
# group entry is the aggregated container the fork projects to it, whose own attribute
# is absent for EVERY group. An absent attribute on the CONTAINER therefore means
# "unknown here", never "not an align-mode group" - the reading that left
# `boundary_groups_known` at 0 on all four ranks (see `_boundary_groups`).
MAMBA_CACHE_MODE_ALIGN = "align"

# THE COUNTER KEYS. A stable set, always all present: an operator diffing two rank files
# must be able to tell "this never happened" from "this build does not report it", which
# is the property that made `disk_restore.load_failure_report` worth writing.
_COUNTER_KEYS = (
    # the apply point's own accounting
    "applies_seen",
    "sampled_out",
    "retained",
    "retained_dropped",
    "refused_thread_unavailable",
    "geometry_unavailable",
    "batches_seen",
    # the truncation shape (assertion 5, Gate B)
    "boundary_groups_known",
    "boundary_entry_missing",
    # the verification thread
    "thread_started",
    "passes",
    "throttled_interval",
    "worker_gone",
    "dropped_waiting_dma",
    "no_load_hwm",
    # the refusals, each of which is a refusal to COMPARE
    "no_window_width",
    "no_group_width",
    "file_missing",
    "file_not_window_wide",
    "group_wider_than_file",
    "file_short",
    # the device read, in `disk_shadow`'s own categories
    "gpu_read_ok",
    "gpu_read_failed",
    "no_gpu_caches",
    "no_gpu_block",
    "width_mismatch",
    # the verdicts
    "checked",
    "equal",
    "mismatch",
    "rechecks",
    "mismatch_gpu_changed_on_recheck",
    "mismatch_stable",
    "mismatch_unrechecked",
    # the binding oracle
    "oracle_checked",
    "oracle_equal",
    "oracle_mismatch",
    "oracle_unavailable",
    "oracle_refused",
    # --- THE SCHEDULER-SIDE ORACLE (0.2.40, moved to the alloc hook at 0.2.41). Every
    # one of these begins `scheduler_` because a number that can be confused with the
    # worker's own is worse than a missing one: the worker's `oracle_unavailable` is
    # EXPECTED in production, while `scheduler_oracle_unavailable` means the oracle did
    # not get built far enough to be useful. Never add the two together.
    #
    # THE EXPECTATION LIFECYCLE (0.2.41) is counted, because "the oracle ran" and "the
    # oracle found nothing to look at" must never read the same. `allocs_observed` is
    # the alloc hook seeing a real external load; `expectations_captured` the pairings
    # it wrote; `expectations_consumed` the sampled emitted entries that matched one;
    # `expectations_dropped` the ones evicted UNCONSUMED (an abandoned load, a finished
    # request, or plain churn - most pairings have no disk file and are legitimately
    # never consumed); `expectations_capped` the per-alloc bound hit.
    "scheduler_allocs_observed",
    "scheduler_expectations_captured",
    "scheduler_expectations_consumed",
    "scheduler_expectations_dropped",
    "scheduler_expectations_capped",
    "scheduler_steps_seen",
    "scheduler_entries_seen",
    "scheduler_sampled_out",
    "scheduler_retained",
    "scheduler_retained_dropped",
    "scheduler_thread_unavailable",
    "scheduler_checked",
    "scheduler_equal",
    "scheduler_mismatch",
    "scheduler_mismatch_stable",
    "scheduler_mismatch_unstable",
    "scheduler_mismatch_neighbour_index",
    "scheduler_mismatch_other_group",
    "scheduler_mismatch_unknown",
    "scheduler_oracle_unavailable",
    "scheduler_oracle_refused",
    # A group the engine cannot view at its OWN block size (group 4 here: 8 tokens
    # against a 40-token hash). Counted at the one place that builds a view, so a group
    # the classification walk skips is never a SILENT skip - the distinction the whole
    # 0.2.40 lesson is about.
    "scheduler_hash_view_unbuildable",
    "scheduler_index_out_of_range",
    "scheduler_errors",
    "section_trimmed",
    "errors",
)

# The three scheduler-side mismatch classifications, and the counter each one moves.
# (0.2.41 removed `gpu_block_not_in_table` with its observation point: at the alloc
# hook the index comes FROM the fork's pairing, so "a destination outside the request's
# table" has no observable moment left - the failure that replaced it, an emitted entry
# whose pairing was never captured, is a NAMED `scheduler_oracle_unavailable`, not a
# mismatch.)
SCHED_CLASS_SHIFT = "neighbour_index"
SCHED_CLASS_GROUP = "other_group"
SCHED_CLASS_UNKNOWN = "unknown"
_SCHED_CLASS_COUNTERS = {
    SCHED_CLASS_SHIFT: "scheduler_mismatch_neighbour_index",
    SCHED_CLASS_GROUP: "scheduler_mismatch_other_group",
    SCHED_CLASS_UNKNOWN: "scheduler_mismatch_unknown",
}

# What the published section says about who produced a number, so a reader of a rank
# file and a reader of `/snapshot` cannot read the same payload as the same process.
# These strings are in the PAYLOAD, so they are priced in bytes: the seam refuses a
# section over `MAX_CONTRIBUTION_BYTES` rather than truncating it, and every hundred
# characters of prose is a hundred characters no counter could use.
WORKER_ROLE = "worker: the apply point"
SCHEDULER_ROLE = "scheduler: the offer point"
ROLE_MEANING = ("unprefixed=worker apply point; scheduler_*=EngineCore offer point; "
                "never add the two")
# The one sentence that keeps the two oracles from being read as one facility. A
# worker's `oracle_unavailable` is a fact of the fork; a `scheduler_oracle_unavailable`
# means the check is not running where it could - the distinction an operator acts on.
ORACLE_RULE = ("worker oracle_unavailable is EXPECTED (a worker has no requests); "
               "scheduler_oracle_unavailable means the check is NOT running")
SCHED_NOTHING_UNAVAILABLE = "nothing unavailable yet"

# WHICH HOOK PRODUCES THE `scheduler_*` NUMBERS, published so "the oracle ran" and
# "the oracle found nothing to look at" are one glance apart. 0.2.40 published numbers
# from an observation point that could not see its own inputs (see the module
# docstring); the least a rewrite of that failure can promise is to name, IN THE
# PAYLOAD, the hook that captured the expectations it is comparing.
SCHED_HOOK_FITTED = "update_state_after_alloc: fitted"
SCHED_HOOK_UNWIRED = "update_state_after_alloc: NOT WIRED here"


def _fresh_state() -> dict:
    """A brand-new `_state`, so the reset list and the declaration cannot drift."""
    return {
        "latch": False,          # the POST op's decision; the environment is the floor
        "latch_set": False,
        "env_resolved": False,   # the environment is read ONCE, see `_resolve_env`
        "env_enabled": False,
        "env_raw": None,
        "env_recognised": True,
        "seq": 0,                # retained-record ids
        "pending": {},           # {id: record}, ordered; ints and one path string
        "sample_cursor": 0,
        "last_pass_mono": None,
        "pass_used": 0,
        "recheck_used": 0,
        "worker": None,          # a weakref, see `_resolve_worker`
        "geometry": None,        # the last geometry read, published verbatim
        "boundary_groups": [],   # the align-mode group ids this worker declared
        "boundary_missing_groups": {},   # {group: count}, bounded
        "last_boundary": None,
        "last_mismatch": None,
        "last_oracle": None,
        "oracle": None,          # (provider, note), see `set_oracle`
        # THE SCHEDULER-SIDE ROLE: its own ring, its own cursor, its own samples. The
        # thread is SHARED (one daemon thread for both roles, so this module can never
        # grow a second one by accident). THE EXPECTATION MAP (0.2.41) is insertion-
        # ordered on purpose: the LRU eviction is `popitem(last=False)`-by-first-key
        # over that order, and `consumed` is a mark, not a deletion - a pairing that
        # was compared may be compared again when the same load state is emitted in a
        # later step, and an entry that was consumed must never be counted as dropped.
        "scheduler_seq": 0,
        "scheduler_pending": {},
        "scheduler_sample_cursor": 0,
        "scheduler_mismatch_samples": [],
        "scheduler_last_oracle": None,
        "scheduler_unreachable_reason": None,
        "scheduler_expectations": {},
        "scheduler_alloc_hook_fitted": False,
        "scheduler_alloc_hook_reason": None,
        "notes": {}, "reasons": [], "reason_seen": set(), "last_trim": None,
        "counters": {key: 0 for key in _COUNTER_KEYS},
    }


_state: dict = _fresh_state()
_lock = threading.Lock()

# The thread and its stop flag are MODULE GLOBALS, not `_state` entries: `_state` is
# what `conftest.py` snapshots and restores around every test, and a thread reference
# restored into the next test is a thread the next test cannot stop.
_thread = None
_thread_lock = threading.Lock()
_stop = None


# --- THE HOUSEKEEPING ----------------------------------------------------------
def _count(name: str, amount: int = 1) -> None:
    # `.get(name, 0)` rather than a subscript: the pass runs on this module's thread
    # while `conftest.py` may be clearing `_state` underneath it, and a diagnostic that
    # cannot be written must never be what raises.
    _state["counters"][name] = int(_state["counters"].get(name, 0)) + int(amount)


def _note_once(key: str, message: str, level: str = "warning") -> bool:
    """One log line per episode, never one per pass - the `disk_shadow` idiom."""
    if _state["notes"].get(key):
        return False
    _state["notes"][key] = True
    logger.log(getattr(logging, level.upper()), "kvtier-dev-plugin: %s", message)
    return True


def _reason(tag: str, detail: str) -> None:
    """Record WHY something was refused, once per distinct tag, and log it once.

    A persistent condition (a worker with no `_load_hwm`, a store whose files are one
    rank's width short) must be legible in the rank file as a SENTENCE, not as an
    anonymous growing integer - the integers say how often, only the text says what.
    """
    if tag in _state["reason_seen"]:
        return
    _state["reason_seen"].add(tag)
    text = str(detail)
    if len(text) > MAX_REASON_CHARS:
        text = text[:MAX_REASON_CHARS - 3] + "..."
    _state["reasons"].append(f"{tag}: {text}")
    del _state["reasons"][MAX_REASON_SAMPLES:]
    _note_once(f"reason:{tag}", f"[disk_restore_verify] {tag}: {text}")


# --- ENABLEMENT ----------------------------------------------------------------
def _resolve_env() -> dict:
    """Read `VLLM_E2_TIER_DISK_RESTORE_VERIFY` ONCE per process, and say so.

    The grammar is `config.disk_restore_verify_enable()` - the same DEFAULT-OFF shape as
    `disk_shadow_enable()`. The read is cached because a live boot's workers inherit an
    environment that does not change, and re-reading it per apply point would put an
    `os.environ` lookup on the engine's step thread for no information. The POST latch
    is the live switch, and the two are reported separately by `enable_state()`.
    """
    if not _state["env_resolved"]:
        read = config.disk_restore_verify_enable()
        _state["env_raw"] = read["raw"]
        _state["env_enabled"] = bool(read["enabled"])
        _state["env_recognised"] = bool(read["recognised"])
        _state["env_resolved"] = True
        if read["enabled"]:
            _note_once("env-on",
                       f"[disk_restore_verify] ENABLED by "
                       f"{config.ENV_DISK_RESTORE_VERIFY}={read['raw']!r}: in a WORKER "
                       f"one restored entry in {SAMPLE_EVERY_N_RESTORES} has its "
                       f"leading group width compared between the file and the GPU "
                       f"window; in the EngineCore one OFFERED entry in the same "
                       f"{SAMPLE_EVERY_N_RESTORES} has its file name re-derived from "
                       f"the engine's own block_hashes at the block index the fork "
                       f"paired it with - at most {MAX_CHECKS_PER_PASS} per role per "
                       f"pass on this module's thread",
                       level="info")
        elif read["raw"] is not None and not read["recognised"]:
            _reason("env_value_unrecognised",
                    f"{config.ENV_DISK_RESTORE_VERIFY}={read['raw']!r} is neither "
                    f"an on word nor an off word; the verifier is OFF, the direction a "
                    f"typo never overrides")
    return {"enabled": bool(_state["env_enabled"]),
            "source": ("environment (read once at first use)"
                       if _state["env_raw"] is not None else
                       "default (off); the environment is read once at first use"),
            "raw": _state["env_raw"],
            "recognised": bool(_state["env_recognised"]),
            "env_var": config.ENV_DISK_RESTORE_VERIFY}


def enable_state() -> dict:
    """One stable-key view of the enablement, including WHO decided it."""
    env = _resolve_env()
    return {"enabled": bool(env["enabled"] or _state["latch"]),
            "env_enabled": bool(env["enabled"]),
            "latch": bool(_state["latch"]),
            "latch_set": bool(_state["latch_set"]),
            "source": ("post op" if _state["latch"] else env["source"]),
            "env_var": config.ENV_DISK_RESTORE_VERIFY, "env_raw": env["raw"],
            "env_recognised": bool(env["recognised"])}


def effective_enabled() -> bool:
    """Verification happens only if the environment says yes or the POST latch does."""
    if _state["latch"]:
        return True
    return bool(_resolve_env()["enabled"])


def set_enabled(on: bool) -> dict:
    """The POST op's decision: IN-PROCESS ONLY, and said out loud in the answer.

    No control file is written - this module opens no file handle for writing anywhere -
    so a POST that arrives at the EngineCore latches the EngineCore. Since 0.2.40 that
    is NOT the nothing it used to be: the EngineCore is where the SCHEDULER-side runs,
    so the latch arms the offer-point oracle (`scheduler_*` counters) in the answering
    process. What the latch still cannot do is arm a WORKER's byte comparison, because a
    worker is a different process - that one is armed with
    `VLLM_E2_TIER_DISK_RESTORE_VERIFY=1` in the engine's environment. `scope` says both
    halves instead of letting an operator believe he armed the worker.
    """
    with _lock:
        _state["latch"] = bool(on)
        _state["latch_set"] = True
        answer = enable_state()
    answer["scope"] = ("this process only: the scheduler-side oracle in the process "
                       "answers this op), while the worker that applies a restore is "
                       f"armed with {config.ENV_DISK_RESTORE_VERIFY}=1")
    if on:
        _note_once("latch-on",
                   "[disk_restore_verify] enabled in this process by the POST op "
                   "(scope: the answering process - its scheduler-side offer oracle - "
                   "and not the worker that applies a restore)",
                   level="info")
    return answer


# --- THE GEOMETRY (read on the step thread, ints only) -------------------------
def _group_widths(worker) -> tuple:
    """Per-group block widths, and the note `worker_diag` attached to them.

    `worker_diag._block_bytes_by_group` is THE source: it is the function that publishes
    `arena.block_bytes_by_group` into the rank file, so using it here means the `W_g`
    this module compares and the `W_g` a reader sees in `diagnostics-rank.json` are the
    same expression. Its own docstring carries the reason that function is
    `page_size_bytes` ALONE (a merged `MambaSpec` already sums its layers, and
    multiplying again by `len(layer_names)` is what made two PP ranks publish different
    per-group figures), so re-deriving widths here would be the second site that
    disagrees with the first.
    """
    try:
        sizes, _layers, note = worker_diag._block_bytes_by_group(worker)
    except Exception:  # noqa: BLE001 - an unreadable config is a named refusal
        return None, "the worker's kv_cache_config could not be read"
    if not sizes:
        return None, str(note or "the worker declared no kv_cache_groups")
    return sizes, None


def _boundary_note(known, resolutions, scanned) -> str | None:
    """The note for a boot with no boundary group, COMPACT by construction.

    It lands in this module's rank section, which keeps itself under
    `MAX_SECTION_BYTES` (3920 of the seam's 4096), so the evidence is class NAMES and a
    status word - the fact that decides whether "no boundary group" means a positional
    boot or a shape this build could not read - not a quotation of every attribute walk.
    """
    if known:
        return None
    classes: list = []
    flags: list = []
    for entry in resolutions:
        for name in ([entry.get("spec_class")]
                     + list(entry.get("sub_spec_classes") or ())):
            if name and name not in classes:
                classes.append(name)
        verdict = entry.get("error") or ("layers disagree"
                                         if entry.get("ambiguous") else None)
        if verdict and verdict not in flags:
            flags.append(verdict)
    note = (f"no group's spec reports {MAMBA_CACHE_MODE_ALIGN!r} mamba_cache_mode"
            f"; saw {', '.join(classes[:BOUNDARY_NOTE_CLASSES]) or 'no specs'}"
            + (f" ({', '.join(flags[:1])})" if flags else "")
            + f" in {scanned} group(s)")
    return (note[:BOUNDARY_NOTE_CHARS - 3] + "..."
            if len(note) > BOUNDARY_NOTE_CHARS else note)


def _boundary_groups(worker) -> tuple:
    """The group ids whose blocks are align-mode state blocks, or a named reason.

    THE PREDICATE IS NOT DUPLICATED HERE, and that is the point. This function used to
    ask `spec.mamba_cache_mode == "align"` itself - the same question
    `disk_store.span_shape_state()` asks - and BOTH readings went dark together on the
    live worker, because a worker's group entry is the AGGREGATED spec the fork
    projects to it (`kv_cache_utils.py:1075-1077`, with
    `_project_kv_cache_groups_to_worker` at `:2606-2614` rebuilding it as a
    `UniformTypeKVCacheSpecs`) and the mode lives on its SUB-SPECS
    (`kv_cache_interface.py:998`, filled from `cache_config` at `abstract.py:78`). One
    shared resolver (`disk_store.resolve_mamba_cache_mode`) now answers for both sides,
    so the two panels cannot disagree and cannot drift apart again: the live rank file
    carried `boundary_groups: []` with this function's own note beside a populated
    `group_bytes`, which is the fingerprint of that shared bug.

    With no align group known this module cannot count `boundary_entry_missing` at all,
    so the truncation shape it was written to catch was structurally invisible - the
    same failure as the spill-side completeness counters, in the other process.
    """
    groups = getattr(getattr(worker, "kv_cache_config", None), "kv_cache_groups", None)
    if not groups:
        return [], "the worker declares no kv_cache_groups"
    vllm_config = getattr(worker, "vllm_config", None)
    listed = list(groups)[:worker_diag.MAX_GROUPS]
    known = []
    resolutions = []
    for index, group in enumerate(listed):
        resolved = disk_store.resolve_mamba_cache_mode(
            getattr(group, "kv_cache_spec", None), vllm_config)
        resolutions.append(resolved)
        if resolved["mode"] == MAMBA_CACHE_MODE_ALIGN:
            known.append(int(index))
    return known, _boundary_note(known, resolutions, len(listed))


def _refresh_geometry(worker) -> dict | None:
    """The numbers this module compares against, read on the step thread, cached.

    THREE READS, ALL ATTRIBUTE WALKS: the group widths (`spec.page_size_bytes` per
    group), the align-mode group ids, and this rank's slot width (`disk_spill.
    measured_slot_bytes`, the same measurement a fingerprint's `total_bytes_per_block`
    is built from). No CUDA call, no file, no tensor data - `measured_slot_bytes`
    multiplies `stride(0) * element_size()`.
    """
    sizes, width_note = _group_widths(worker)
    boundary, boundary_note = _boundary_groups(worker)
    arena = disk_spill.measured_slot_bytes(getattr(worker, "cpu_kv_caches", None))
    geometry = {"group_bytes": list(sizes) if sizes else None,
                "group_bytes_note": width_note,
                "boundary_groups": boundary,
                "boundary_note": boundary_note,
                "window_bytes": (int(arena["bytes_per_file"])
                                 if arena.get("ok") else None),
                "window_note": (None if arena.get("ok")
                                else str(arena.get("reason"))),
                "window_rule": "sum(region.stride(0) * element_size)",
                "read_mono": time.monotonic()}
    with _lock:
        _state["geometry"] = geometry
        _state["boundary_groups"] = boundary
    if sizes is None:
        with _lock:
            if not _state["counters"].get("geometry_unavailable"):
                _count("geometry_unavailable")
        _reason("group_widths_unavailable",
                width_note or "the per-group block widths are unreadable")
        return None
    return geometry


def _note_boundary_shape(batch, failed, groups_present: set,
                         failed_groups: set | None = None) -> None:
    """Assertion 5, at batch granularity: a boundary group that restored NOTHING.

    Called with the batch's SURVIVING group ids (the entries the apply point actually
    applied). A batch that applied entries and named no align-mode group is the Gate B
    shape: the fork truncated that group's hit list with `n_take_g`
    (`manager.py:445-448`) below its single real entry, so the positional groups' bytes
    were restored and the state block that makes them addressable was not.

    `failed_groups` are EXCLUDED from the missing set on purpose. An align entry that
    was there and whose read failed (`file_missing`, `not_settled`) is a different
    finding, already counted where it happened; reporting its absence here as well would
    make one failure two findings in two files, and would point at a truncation that did
    not happen. `entries_failed` in `last_boundary` keeps the distinction visible.
    """
    excluded = set(failed_groups or ())
    with _lock:
        known = list(_state["boundary_groups"])
    if not known or not groups_present:
        return
    missing = [group for group in known
               if group not in groups_present and group not in excluded]
    if not missing:
        return
    with _lock:
        _count("boundary_entry_missing", len(missing))
        _count("boundary_groups_known", len(known))
        for group in missing:
            tracked = _state["boundary_missing_groups"]
            tracked[group] = int(tracked.get(group, 0)) + 1
            while len(tracked) > MAX_BOUNDARY_GROUPS_PUBLISHED:
                tracked.pop(next(iter(tracked)))
        _state["last_boundary"] = {
            "event": _int_or_none((batch or {}).get("event")),
            "groups_present": sorted(int(value) for value in groups_present),
            "boundary_groups": [int(value) for value in known],
            "missing": [int(value) for value in missing],
            "entries_failed": len(failed or {}),
        }
    _reason("boundary_entry_missing",
            f"a batch restored groups {sorted(int(v) for v in groups_present)} and "
            f"nothing for align-mode group(s) {missing}: the fork's `n_take_g` "
            f"truncation (`manager.py:445-448`) dropped that group's only real entry")


# --- THE STEP THREAD'S ONLY CALL ----------------------------------------------
def observe_apply(worker, batch, failed) -> dict:
    """The apply point's witness: remember what was restored, and count the shape.

    THE ONLY ENTRY POINT ON AN ENGINE THREAD, and it does bookkeeping only. Called by
    `disk_restore._worker_before_load` after `span_shape` has made `failed` final, so
    the entries in `batch["jobs"]` that are not in `failed` are exactly the ones whose
    bytes were memmoved into the CPU slot this step - the ones the fork is about to DMA
    into the GPU blocks named beside them.

    IT CANNOT CHANGE THE STEP. It reads, counts, and retains; it never prunes, reorders,
    renames or re-files anything, and the call site is after every decision the apply
    point makes. Every path is inside a `try` that turns a raise into a counter, because
    a witness that can break a restore is not a witness, it is an outage.

    `{"seen": N, "retained": N, "reason": ...}` for a caller that wants the numbers; the
    counters are the durable record.
    """
    try:
        if not effective_enabled():
            return {"seen": 0, "retained": 0, "reason": "disabled"}
        jobs = (batch or {}).get("jobs") or {}
        if not jobs:
            return {"seen": 0, "retained": 0, "reason": "no entries"}
        survivors = [(job_id, job) for job_id, job in jobs.items()
                     if job_id not in (failed or {})][:MAX_ENTRIES_SCANNED]
        if not survivors:
            return {"seen": 0, "retained": 0, "reason": "nothing applied"}
        # The groups whose entry the apply point DROPPED (its read failed): the shape
        # check below must not read a dropped entry as a truncated list.
        failed_groups: set = set()
        for job_id in (failed or {}):
            group = _int_or_none((jobs.get(job_id) or {}).get("group")) \
                if isinstance(jobs.get(job_id), dict) else None
            if group is not None:
                failed_groups.add(int(group))
        with _lock:
            _state["worker"] = weakref.ref(worker) if worker is not None else None
            _count("batches_seen")
        geometry = _refresh_geometry(worker)
        event = _int_or_none((batch or {}).get("event"))
        groups_present: set = set()
        seen = 0
        retained = 0
        for _job_id, job in survivors:
            if not isinstance(job, dict):
                continue
            seen += 1
            group = _int_or_none(job.get("group"))
            if group is not None:
                groups_present.add(int(group))
            # Counted BEFORE the sample decision and before every geometry test:
            # `applies_seen` is the denominator of the sample rate, and an operator
            # dividing `verified_equal` by it is asking "how much of my disk-restore
            # traffic did the verifier look at?". Counting only the sampled entries
            # would make that fraction 1/1 forever, and counting only the ones with
            # readable geometry would make it vanish on the boots whose config is odd -
            # the same
            # reason `disk_gpu_shadow` counts `pairs_unpaired` before it pairs anything.
            _count("applies_seen")
            if geometry is None:
                # Already counted, once, as `geometry_unavailable` with the reason the
                # config gave. Retaining here would also count `no_group_width` once per
                # entry, which turns one unreadable config into a number that grows with
                # traffic and says nothing more than the first line already said.
                continue
            with _lock:
                cursor = int(_state["sample_cursor"])
                _state["sample_cursor"] = cursor + 1
                selected = (cursor % max(1, SAMPLE_EVERY_N_RESTORES)) == 0
            if not selected:
                with _lock:
                    _count("sampled_out")
                continue
            if retained >= MAX_RETAINED_PER_STEP:
                # The per-step bound is a BOUND, and hitting it is normal on a wide
                # batch: counted, not reasoned (one reason per step would be a reason
                # per step).
                with _lock:
                    _count("sampled_out")
                continue
            record = _record(job, group, event)
            if record is None:
                continue      # `_record` counted and named the refusal itself
            if not _retain(record):
                break
            retained += 1
        _note_boundary_shape(batch, failed, groups_present, failed_groups)
        return {"seen": seen, "retained": retained,
                "reason": None if geometry else "group widths unreadable"}
    except Exception as exc:  # noqa: BLE001 - never into the engine's step path
        with _lock:
            _count("errors")
        _reason("observe_apply", f"the apply-point witness raised {type(exc).__name__}")
        return {"seen": 0, "retained": 0, "reason": f"error_{type(exc).__name__}"}


def _record(job: dict, group, event) -> dict | None:
    """The retained job: ints, one path string, and the geometry to compare against.

    NOTHING HERE IS A TENSOR OR A FILE HANDLE. The path is the one `disk_read` was
    already asked to read (`_worker_bind` resolved it), so the verifier reads a file the
    restore path itself chose - it never resolves a name of its own, and a name it
    resolved itself would be a second naming site, which is how files become
    un-findable.
    """
    path = job.get("path")
    if not isinstance(path, str) or not path:
        with _lock:
            _count("no_window_width")
        return None
    width = _int_or_none(job.get("bytes"))
    if not width or width <= 0:
        with _lock:
            _count("no_window_width")
        _reason("no_window_width",
                "the restore entry declares no `bytes`, so there is no window width to "
                "check the file against (assertion 1 cannot be evaluated)")
        return None
    with _lock:
        group_bytes = _state["geometry"].get("group_bytes") if _state.get("geometry") \
            else None
        _state["seq"] = int(_state["seq"]) + 1
        record_id = int(_state["seq"])
    group_width = None
    if isinstance(group_bytes, list) and group is not None \
            and 0 <= int(group) < len(group_bytes):
        group_width = _int_or_none(group_bytes[int(group)])
    if not group_width or group_width <= 0:
        with _lock:
            _count("no_group_width")
        _reason("no_group_width",
                f"group {group} has no readable `page_size_bytes`, so W_g is unknown "
                "and the comparison would have to guess where the group's bytes stop")
        return None
    gpu_block = _int_or_none(job.get("gpu_block_id"))
    if gpu_block is None or gpu_block < 0:
        with _lock:
            _count("no_gpu_block")
        return None
    return {"id": record_id, "path": path,
            "key16": (str(job.get("key16"))[:32] if job.get("key16") else None),
            "group": int(group) if group is not None else None,
            "gpu_block": int(gpu_block),
            "cpu_block": _int_or_none(job.get("cpu_block_id")),
            "bytes": int(width), "group_bytes": int(group_width),
            "event": event, "mono": time.monotonic()}


def _retain(record: dict) -> bool:
    """Queue one record, and start the thread that will read it. False = dropped.

    The thread is started HERE rather than by a pass, exactly as `disk_gpu_shadow`'s
    `consider` does: a record retained with no thread to read it is a record that will
    be dropped with no explanation, so the refusal is taken now, on the spot, and
    counted as `refused_thread_unavailable`.
    """
    with _lock:
        pending = _state["pending"]
        while len(pending) >= MAX_RETAINED_JOBS:
            # Oldest first: the entry that has waited longest has had the longest for
            # its GPU block to be handed out again, so it is the least likely to be
            # evidence.
            pending.pop(next(iter(pending)))
            _count("retained_dropped")
        pending[record["id"]] = record
        _count("retained")
    if not _ensure_thread():
        with _lock:
            _count("refused_thread_unavailable")
            _state["pending"].pop(record["id"], None)
        return False
    return True


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --- THE THREAD ----------------------------------------------------------------
def _ensure_thread() -> bool:
    """Start the module's single verifier thread, once, lazily. Never at import."""
    global _thread, _stop
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return True
        try:
            stop = threading.Event()
            thread = threading.Thread(target=_loop, args=(stop,),
                                      name="kvtier-disk-restore-verify", daemon=True)
            thread.start()
        except Exception:  # noqa: BLE001 - no thread is a refusal, not a fault
            return False
        _stop = stop
        _thread = thread
        with _lock:
            _count("thread_started")
        return True


def _contained(action):
    """Run one pass so that NOTHING it does can kill the thread.

    Not defensive for its own sake: `conftest.py` clears this module's `_state` IN PLACE
    while a pass from the previous test may still be running, and a thread that walks a
    half-cleared dictionary raises a `KeyError` that has nothing to do with the product.
    A loop that dies there stops verifying silently and every later entry piles up until
    the queue drops them all - which reads as "the verifier found nothing", the one
    answer worse than a wrong one.
    """
    try:
        return action()
    except Exception as exc:  # noqa: BLE001 - a pass never kills the verifier thread
        with _lock:
            _count("errors")
        _reason("pass", f"the verification pass raised {type(exc).__name__}")
        return None


def _loop(stop) -> None:
    """The tick: wait `PASS_TICK_SECONDS`, then ONE bounded pass. Nothing else, ever."""
    while not stop.wait(PASS_TICK_SECONDS):
        _contained(run_pass)


def _resolve_worker(worker=None):
    """The worker to read GPU bytes from, or None. A weakref, on purpose.

    The KV cache tensors are owned by the worker (or by the model runner behind it), and
    the thread that copies from them is not the thread that frees them. Holding a strong
    reference would keep freed-but-alive tensors alive; holding NONE would mean not
    being able to tell "the engine is gone" from "I have never seen one". A weakref
    gives the third answer: `worker_gone`, counted, and the pass stops.
    """
    if worker is not None:
        return worker
    with _lock:
        ref = _state["worker"]
    if ref is None:
        return None
    try:
        return ref()
    except Exception:  # noqa: BLE001 - a dead or broken ref is simply no worker
        return None


# --- THE COMPARISON ------------------------------------------------------------
def _first_difference(left: bytes, right: bytes) -> tuple:
    """`(offset, scanned_bytes)`: where two heads first differ, and how far we looked.

    `disk_shadow`'s discipline, chunked so the compare is `memcmp` and the only Python
    byte loop runs inside the ONE 4 KiB chunk that differs. `offset` is None when the
    two agree for the whole scanned window - which is a real answer: "they differ past
    256 KiB" is what an operator can act on, and a full-width Python scan is not worth
    an offset.
    """
    limit = min(len(left), len(right), MAX_DIFF_SCAN_BYTES)
    for start in range(0, limit, 4096):
        stop = min(start + 4096, limit)
        if left[start:stop] == right[start:stop]:
            continue
        for offset in range(start, stop):
            if left[offset] != right[offset]:
                return offset, stop
        return None, stop
    if len(left) != len(right):
        return min(len(left), len(right)), limit
    return None, limit


def _read_file_head(path: str, width: int, group_bytes: int) -> dict:
    """The FILE side of the comparison: the leading W_g bytes, and the width facts.

    Assertion 1 (`file_size == bytes_per_block`) and assertion 2 (`W_g <= file_size`)
    are decided HERE, from `stat` and from this read - never inferred from the journal
    or the fingerprint, because the question is what is on the disk under this name NOW.

    The read is `read(W_g)`: the tail is never read. A file that is window-wide and a
    group that owns 1.6 MB of it means 9.8 MB of other groups' bytes we have no claim to
    compare.
    """
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return {"ok": False, "kind": "file_missing",
                "reason": "the file the restore path read is gone (the LRU evicted it "
                          "between the read and the verification)"}
    except OSError as exc:
        return {"ok": False, "kind": "file_missing",
                "reason": f"the spill file could not be stat'd ({type(exc).__name__})"}
    if int(size) != int(width):
        return {"ok": False, "kind": "file_not_window_wide",
                "reason": f"the file is {int(size)} B and this rank's window is "
                          f"{int(width)} B: a spill file is exactly one slot wide, so "
                          "this is a geometry or a rank-crossing answer, not a byte "
                          "answer, and it is refused rather than compared",
                "file_size": int(size)}
    if int(group_bytes) > int(size):
        return {"ok": False, "kind": "group_wider_than_file",
                "reason": f"group's own width {int(group_bytes)} B exceeds the file's "
                          f"{int(size)} B: comparing would read past EOF and call the "
                          "zeros a mismatch",
                "file_size": int(size)}
    try:
        with open(path, "rb") as handle:
            head = handle.read(int(group_bytes))
    except OSError as exc:
        return {"ok": False, "kind": "file_missing",
                "reason": f"the spill file could not be opened "
                          f"({type(exc).__name__})"}
    if len(head) < int(group_bytes):
        return {"ok": False, "kind": "file_short",
                "reason": f"the file read returned {len(head)} B of a requested "
                          f"{int(group_bytes)} B head", "file_size": int(size)}
    return {"ok": True, "head": head, "bytes": len(head), "file_size": int(size)}


def _classify_mismatch(worker, record, gpu_head: bytes, file_head: bytes) -> dict:
    """A disagreement: re-read the GPU block ONCE and ask whether the GPU moved.

    The re-read is the ONLY thing that separates a layout bug from the engine reusing
    the destination block, and it is bounded per pass (`MAX_RECHECKS_PER_PASS`) because
    a box whose every entry mismatches must not turn the verifier into a device-copy
    loop.
    """
    rechecked = False
    changed = None
    reason = None
    with _lock:
        if int(_state["recheck_used"]) < MAX_RECHECKS_PER_PASS:
            _state["recheck_used"] = int(_state["recheck_used"]) + 1
            _count("rechecks")
            rechecked = True
    if rechecked:
        again = disk_shadow._read_gpu_slot(worker, record["gpu_block"], record["bytes"])
        if again.get("ok"):
            changed = bytes(again["bytes"][:record["group_bytes"]]) != gpu_head
        else:
            rechecked = False
            reason = again.get("reason")
            _reason(f"recheck:{again.get('kind')}",
                    f"the re-read that would classify a mismatch failed: {reason}")
    offset, scanned = _first_difference(gpu_head, file_head)
    with _lock:
        if not rechecked:
            _count("mismatch_unrechecked")
            category = "mismatch_unrechecked"
        elif changed:
            _count("mismatch_gpu_changed_on_recheck")
            category = "mismatch_gpu_changed_on_recheck"
        else:
            _count("mismatch_stable")
            category = "mismatch_stable"
        _state["last_mismatch"] = {
            "category": category, "gpu_block": int(record["gpu_block"]),
            "group": record["group"], "event": record["event"],
            "key16": record["key16"],
            "bytes": int(record["bytes"]),
            "compared_bytes": int(record["group_bytes"]),
            # HOW MUCH WAS REFUSED. The tail is other groups' stale bytes by
            # construction; publishing the excluded span is what lets a reader tell a
            # narrow-group compare from a full-window one. Nothing here ever compared
            # it.
            "tail_bytes_excluded": int(record["bytes"]) - int(record["group_bytes"]),
            "first_difference": offset, "scanned_bytes": int(scanned),
            "rechecked": bool(rechecked), "gpu_changed_on_recheck": changed,
        }
    return {"checked": True, "equal": False, "category": category,
            "first_difference": offset, "scanned_bytes": scanned, "reason": reason}


def _verify_record(worker, record: dict) -> dict:
    """One retained entry, on this module's thread. Never raises."""
    try:
        return _verify_record_inner(worker, record)
    except Exception as exc:  # noqa: BLE001 - an entry never kills the pass
        with _lock:
            _count("errors")
        _reason("verify", f"verifying one entry raised {type(exc).__name__}")
        return {"checked": False, "reason": f"error_{type(exc).__name__}"}


def _verify_record_inner(worker, record: dict) -> dict:
    """The five assertions, in the order that makes each failure nameable on its own."""
    event = record.get("event")
    hwm = _int_or_none(getattr(worker, "_load_hwm", None))
    if hwm is None:
        # Without the fork's own completion watermark there is no way to say the DMA
        # landed, and comparing before it is how a verifier invents a mismatch.
        with _lock:
            _count("no_load_hwm")
        _reason("no_load_hwm",
                "this worker exposes no `_load_hwm`, so nothing here can say whether "
                "the fork's CPU->GPU copy has completed; entries are never compared, "
                "because a comparison that races the DMA is not evidence")
        return {"checked": False, "reason": "no_load_hwm"}
    if event is None or hwm < int(event):
        if (time.monotonic() - float(record.get("mono") or 0.0)) > JOB_MAX_AGE_SECONDS:
            with _lock:
                _count("dropped_waiting_dma")
            return {"checked": False, "reason": "dropped_waiting_dma"}
        return {"checked": False, "reason": "waiting_for_dma"}
    with _lock:
        group_bytes = int(record["group_bytes"])
        width = int(record["bytes"])
    if group_bytes > width:
        # The geometry contradiction, caught before the file is opened: a group wider
        # than the slot cannot fit in the window the restore read it into.
        with _lock:
            _count("group_wider_than_file")
        _reason("group_wider_than_file",
                f"group {record['group']}'s own width {group_bytes} B exceeds this "
                f"rank's window {width} B: the group's `page_size_bytes` and the "
                "measured slot disagree, so there is no correct number of bytes to "
                "compare")
        return {"checked": False, "reason": "group_wider_than_file"}
    file_side = _read_file_head(record["path"], width, group_bytes)
    if not file_side["ok"]:
        kind = file_side["kind"]
        with _lock:
            _count(kind)
        _reason(kind, file_side["reason"])
        return {"checked": False, "reason": kind, "detail": file_side["reason"]}
    # THE GPU SIDE, THROUGH `disk_shadow`'S SEAM AND NO OTHER: `_gpu_region_map` ->
    # `_copy_slot_to_host` -> `_host_landed` -> `disk_spill.read_slot_bytes`. The
    # category names below are `disk_shadow`'s, so the two witnesses cannot disagree
    # about what a failed device read means, and a test that replaces the seam redirects
    # this read too.
    read = disk_shadow._read_gpu_slot(worker, int(record["gpu_block"]), width)
    if not read.get("ok"):
        kind = read.get("kind")
        if read.get("reason_kind") == "region_width_mismatch":
            kind = "width_mismatch"
        kind = kind if kind in _COUNTER_KEYS else "gpu_read_failed"
        with _lock:
            _count(kind)
        _reason(f"gpu:{kind}", str(read.get("reason")))
        return {"checked": False, "reason": kind,
                "detail": str(read.get("reason"))}
    with _lock:
        _count("gpu_read_ok")
    gpu = bytes(read["bytes"])
    # ONLY THE LEADING W_g OF EACH SIDE. Both slices are built here, and nothing
    # downstream sees a longer one: the file head was read with `read(W_g)` and the
    # device bytes are cut to `[:W_g]` before they reach the compare or the re-read. The
    # remainder of the window is the other groups' bytes and stale by construction.
    gpu_head = gpu[:group_bytes]
    file_head = bytes(file_side["head"])[:group_bytes]
    if gpu_head == file_head:
        with _lock:
            _count("checked")
            _count("equal")
        oracle = _oracle_check(record)
        return {"checked": True, "equal": True, "bytes": group_bytes,
                "oracle": oracle, "tail_bytes_excluded": width - group_bytes}
    verdict = _classify_mismatch(worker, record, gpu_head, file_head)
    verdict["oracle"] = _oracle_check(record)
    return verdict


# --- THE BINDING ORACLE (assertion 4) ------------------------------------------
def set_oracle(provider, note: str | None = None) -> dict:
    """Install (or remove, with None) the source of `block_hashes` for this process.

    THE SEAM, AND WHY IT IS A SEAM. The oracle needs a request's `block_hashes`, and a
    WORKER HAS NO REQUESTS: the fork's worker metadata carries `load_event_to_reqs`, and
    the module that defines it states the rule - "Job->req_id translation is handled by
    the scheduler-side manager ... the worker never knows about request identities"
    (`vllm/v1/simple_kv_offload/metadata.py`). So on this fork nothing can be installed
    here from inside a worker, `oracle_unavailable` counts every entry, and the view
    says so. It is not a stub: the recomputation below is what runs the moment a
    provider IS installed, by a fork patch that carries the per-entry hash view to the
    worker (the smallest change that would make the binding check live in production),
    or by a test.

    A provider answers one retained entry with the ENGINE'S OWN inputs:
    `{"block_hashes": [...], "hash_block_size": int, "group_size": int, "index": int}`,
    where `index` is the index INTO THE RESOLVED VIEW for the GPU block this entry named
    - the arithmetic the fork itself does at `manager.py:888-900`. Anything else is
    `oracle_refused`, never a silent skip.
    """
    with _lock:
        if provider is None:
            _state["oracle"] = None
        elif not callable(provider):
            _state["oracle"] = None
            _reason("oracle_not_callable",
                    "set_oracle was handed something that is not callable, so the "
                    "binding oracle stays unavailable rather than guessing at it")
        else:
            _state["oracle"] = (provider, (str(note)[:160] if note else None))
    state = oracle_state()
    if state["provider"]:
        _note_once("oracle-on",
                   "[disk_restore_verify] the binding oracle has a provider: restored "
                   "entries now have their file name re-derived from block_hashes",
                   level="info")
    return state


def oracle_state() -> dict:
    """The oracle's reachability, published. A stable key set on every path."""
    with _lock:
        installed = _state["oracle"]
        counters = _state["counters"]
        last = _state["last_oracle"]
        return {"provider": bool(installed),
                "note": (installed[1] if installed else None),
                "reachable": bool(installed),
                "unreachable_reason": (None if installed else
                                       # IN THE PAYLOAD, so it is priced in bytes like
                                       # every other string here: the file, the reason
                                       # and the consequence, and nothing else.
                                       "a worker receives no request identities "
                                       "(vllm/v1/simple_kv_offload/metadata.py), so "
                                       "block_hashes is unreachable where a restore is "
                                       "applied; entries count oracle_unavailable "
                                       "until set_oracle() installs a provider"),
                "checked": int(counters["oracle_checked"]),
                "equal": int(counters["oracle_equal"]),
                "mismatch": int(counters["oracle_mismatch"]),
                "unavailable": int(counters["oracle_unavailable"]),
                "refused": int(counters["oracle_refused"]),
                "last": copy.deepcopy(last) if last else None}


def oracle_expect_key16(source: dict, group) -> dict:
    """The engine's own derivation of the `key16` a file for THIS block must carry.

    Four calls, all of them to code this package does not own:
      1. `resolve_block_hashes(block_hashes, hash_block_size, group_size)` - the
      per-group
         hash view, the same expression `disk_restore._disk_run` builds
         (`disk_restore.py:4045-4048`) and the fork builds for its lookup
         (`manager.py:1591-1601`);
      2. `make_block_hash_with_group_id(digest, group)` - the group-qualified key, the
      same
         call the fork makes when it stores a block (`manager.py:897-900`);
      3. `disk_layout.split_block_key(...)` - the ONE place this package reads the
      4-byte
         group suffix, which is what turns the key's hex into the `key16` a file name
         carries;
      4. and both helpers come from `disk_restore._key_helpers()`, which refuses rather
         than guesses if the fork renamed them. Nothing here re-hashes or re-packs: a
         plugin that invented its own derivation would eventually disagree with the
         engine about which bytes are the group, and that disagreement is a file read
         under the wrong name - the exact failure this oracle is supposed to CATCH, so
         it cannot be allowed to be the thing that causes it.
    """
    if not isinstance(source, dict):
        return {"ok": False, "kind": "oracle_source_shape",
                "reason": "the oracle answered with something that is not a dict"}
    missing = [key for key in ("block_hashes", "hash_block_size", "group_size", "index")
               if source.get(key) is None]
    if missing:
        return {"ok": False, "kind": "oracle_source_shape",
                "reason": f"the oracle's answer carries no {', '.join(missing)}"}
    from . import disk_restore  # noqa: PLC0415 - it imports nothing that imports us
    helpers = disk_restore._key_helpers()
    if not helpers.get("ok"):
        return {"ok": False, "kind": "no_key_helpers",
                "reason": str(helpers.get("reason"))}
    try:
        view = helpers["resolve"](source["block_hashes"],
                                  int(source["hash_block_size"]),
                                  int(source["group_size"]))
        index = int(source["index"])
        if index < 0 or index >= len(view):
            return {"ok": False, "kind": "oracle_index_out_of_range",
                    "reason": f"index {index} is outside a resolved view of "
                              f"{len(view)} entries"}
        digest = view[index]
        if not isinstance(digest, bytes):
            return {"ok": False, "kind": "oracle_not_bytes",
                    "reason": f"the engine's block hash at index {index} is "
                              f"{type(digest).__name__}, not bytes"}
        key_hex = bytes(helpers["pack"](digest, int(group))).hex()
    except Exception as exc:  # noqa: BLE001 - the engine's helpers are not ours to trust
        return {"ok": False, "kind": "oracle_raised",
                "reason": f"the engine's own key machinery raised "
                          f"{type(exc).__name__}"}
    split = disk_layout.split_block_key(key_hex)
    if not split.get("ok"):
        return {"ok": False, "kind": "oracle_key_shape",
                "reason": f"the recomputed key is not a block hash: "
                          f"{split.get('reason')}"}
    return {"ok": True, "key16": split["key16"], "key_hex": key_hex}


def _oracle_check(record: dict) -> dict:
    """Ask the oracle what name THIS GPU block should have been restored under."""
    with _lock:
        installed = _state["oracle"]
    if not installed:
        with _lock:
            _count("oracle_unavailable")
        return {"state": "oracle_unavailable",
                "reason": "no provider is installed in this process"}
    provider = installed[0]
    entry = {key: record.get(key) for key in
             ("gpu_block", "group", "key16", "cpu_block", "bytes", "event")}
    try:
        answer = provider(entry)
    except Exception as exc:  # noqa: BLE001 - a provider's fault is a counted outcome
        with _lock:
            _count("oracle_refused")
        _reason("oracle_raised", f"the oracle provider raised {type(exc).__name__}")
        return {"state": "oracle_refused", "reason": f"{type(exc).__name__}"}
    if answer is None:
        with _lock:
            _count("oracle_unavailable")
        return {"state": "oracle_unavailable",
                "reason": "the provider has no block_hashes for this entry"}
    expected = oracle_expect_key16(answer, record.get("group"))
    if not expected["ok"]:
        with _lock:
            _count("oracle_refused")
        _reason(expected["kind"], expected["reason"])
        return {"state": "oracle_refused", "reason": expected["reason"]}
    equal = expected["key16"] == record.get("key16")
    with _lock:
        _count("oracle_checked")
        _count("oracle_equal" if equal else "oracle_mismatch")
        _state["last_oracle"] = {"equal": bool(equal),
                                 "expected_key16": expected["key16"],
                                 "name_key16": record.get("key16"),
                                 "group": record.get("group"),
                                 "gpu_block": record.get("gpu_block"),
                                 "event": record.get("event")}
    return {"state": "oracle_equal" if equal else "oracle_mismatch",
            "expected_key16": expected["key16"]}


# --- THE SCHEDULER-SIDE ROLE (0.2.40, moved at 0.2.41): THE BINDING ORACLE ----
# Everything below runs in the EngineCore. TWO engine-thread doors: `observe_alloc`
# (hung on `update_state_after_alloc` by `disk_restore`'s wrapper, BEFORE the fork's
# body) captures what the fork is about to pair; `observe_offer` (called by
# `disk_restore._build_load_paths` after the paths are written) compares the names we
# emitted against those captured expectations. Everything above in this file is a
# worker's apply point.
def _sched_unavailable(tag: str, detail: str) -> dict:
    """The oracle could not be BUILT for this entry: counted, reasoned, published.

    THE most important thing this role can say, and it is said as a sentence. A
    `scheduler_oracle_unavailable` that grows with traffic means the addressing check is
    not running in the one process that can run it - which is a different fact from a
    worker's `oracle_unavailable` (that one is expected forever) and must never be
    collapsed into it.
    """
    text = str(detail)[:MAX_REASON_CHARS]
    with _lock:
        _count("scheduler_oracle_unavailable")
        _state["scheduler_unreachable_reason"] = f"{tag}: {text}"
    _reason(f"scheduler_oracle:{tag}", text)
    return {"state": "scheduler_oracle_unavailable", "reason": tag}


def _sched_refused(tag: str, detail: str) -> dict:
    """The oracle was built and answered badly: a refusal, never a silent skip."""
    with _lock:
        _count("scheduler_oracle_refused")
    _reason(f"scheduler_oracle:{tag}", str(detail))
    return {"state": "scheduler_oracle_refused", "reason": tag}


def _scheduler_sizes(manager) -> dict:
    """`resolve_block_hashes`' two remaining arguments, read off the manager.

    `hash_block_size` and `group_block_sizes` are the manager's own fields (the same two
    the fork feeds its own `resolve_block_hashes` at `manager.py:842-844` and the same
    two `_disk_run` uses for the offer), so this module derives no block geometry of its
    own.
    """
    hash_block = _int_or_none(getattr(manager, "hash_block_size", None))
    sizes = getattr(manager, "group_block_sizes", None)
    if not hash_block or hash_block <= 0:
        return {"ok": False, "kind": "no_hash_block_size",
                "reason": "the manager exposes no usable `hash_block_size`, so the "
                          "engine's own hash view cannot be built"}
    if not isinstance(sizes, (list, tuple)) or not sizes:
        return {"ok": False, "kind": "no_group_block_sizes",
                "reason": "the manager exposes no `group_block_sizes`, so the view for "
                          "this entry's group cannot be built"}
    cleaned = [_int_or_none(value) for value in list(sizes)[:worker_diag.MAX_GROUPS]]
    return {"ok": True, "hash_block_size": int(hash_block),
            "group_block_sizes": cleaned}


# --- THE ALLOC-TIME HOOK (0.2.41): THE EXPECTATION CAPTURE ---------------------
# The 0.2.40 offer point read the request's GPU block table from
# `scheduler_output.kv_connector_block_state`, and that table is THIS STEP'S
# (`sched/scheduler.py:1412-1425`, cleared at `:1497`) while a load state is
# emitted in steps LATER than the one that allocated it. In production every
# sampled entry landed on exactly those later steps, every lookup answered
# `block_state_does_not_cover_request`, and the oracle checked nothing while
# counting its own blindness. The hook below moves the observation to where the
# PAIRING IS DECIDED - `update_state_after_alloc` (`manager.py:381-489`) - which
# is where every input the oracle needs is an ARGUMENT.

def note_alloc_hook(fitted: bool, reason=None) -> dict:
    """Record the ALLOC HOOK's state, called by `disk_restore.install`/`uninstall`.

    This is what makes `scheduler_oracle_hook` in the published section a FACT
    rather than a claim: the module does not infer whether the hook is wired, it
    is told by the only installer. Until an install says otherwise the answer is
    "not wired" - the honest state of a worker process (which runs no allocs at
    all) and of any process whose tier class was never fitted.
    """
    with _lock:
        _state["scheduler_alloc_hook_fitted"] = bool(fitted)
        _state["scheduler_alloc_hook_reason"] = (
            None if fitted or reason is None
            else str(reason)[:MAX_REASON_CHARS])
        return {"fitted": bool(fitted)}


def _cdiv(a: int, b: int) -> int:
    """The fork's own ceiling division (`manager.py:445`), local on purpose.

    Not imported: the fork does not export it, and this observer must stay
    answerable on a fork that moved it.
    """
    return -(-int(a) // int(b))


def _alloc_gate(manager):
    """The manager's prefix-cacheable set as a frozenset, or None for no gate.

    The fork's pairing loop SKIPS groups outside this set (`manager.py:441-443`),
    and the observer must apply the same gate or it captures pairings the fork
    will never emit - which would turn `scheduler_expectations_dropped` into
    churn noise and dilute the one counter that is supposed to mean "a pairing
    this map remembered was abandoned". A fork without the field has no gate;
    that is answered here, not guessed at.
    """
    gate = getattr(manager, "prefix_cacheable_group_ids", None)
    if gate is None:
        return None
    try:
        return frozenset(int(g) for g in gate)
    except Exception:  # noqa: BLE001 - an unreadable gate is no gate; the fork's
        # own loop will fail on it there, where the guard can see it
        return None


def _peek_pending(manager, request_id):
    """`manager._pending_cpu_hits[request_id]`, PEEKED - never popped.

    The fork POPS the entry when its body runs (`manager.py:404`), which is why
    the wrapper calls `observe_alloc` BEFORE the fork's body and why this is a
    `.get`: a peek is the only read that survives the call it observes. The
    returned tuple is the engine's live object; this module reads it
    (`[0]` per-group hit lists, `[2]` num_computed_tokens) and mutates nothing.
    """
    table = getattr(manager, "_pending_cpu_hits", None)
    if not isinstance(table, dict):
        return None
    try:
        pending = table.get(request_id)
    except Exception:  # noqa: BLE001,S112 - an unreadable ledger holds nothing
        return None
    if not isinstance(pending, (tuple, list)) or len(pending) < 3:
        return None
    return pending


def observe_alloc(manager, request, blocks, num_external_tokens) -> dict:
    """THE ALLOC-TIME DOOR (0.2.41): capture what the fork is about to pair.

    Called by the `update_state_after_alloc` wrapper BEFORE the fork's body. The
    scheduler calls that method for EVERY scheduled request
    (`sched/scheduler.py:1228`), so this runs per scheduled request and returns
    at once for the overwhelming majority that carry no external load. For a
    real external load it runs a bounded, int-only loop replicating the fork's
    own pairing walk (`manager.py:433-475`: prefix-cacheable gate, cdiv
    truncation of the per-group hit, start at `num_computed_tokens //
    g_block_size`, `is_null` skipped WITHOUT advancing the pairing), and writes
    `(gpu_block, cpu_block) -> (group, index in that group's table, the engine's
    block_hashes)` into the bounded LRU expectation map. It hashes nothing,
    reads no file, touches no GPU and mutates NOTHING: the pending hit is
    peeked, not popped, `blocks` is only asked for `get_block_ids()`, and it
    returns nothing to the caller. Never raises; the wrapper hands it to the
    scheduler's step path.
    """
    try:
        if not effective_enabled():
            return {"captured": 0, "reason": "disabled"}
        request_id = getattr(request, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            _sched_refused("alloc_no_request_id",
                           "update_state_after_alloc was handed a request with no "
                           "request id, so no expectation can be attached to any "
                           "pairing it makes")
            return {"captured": 0, "reason": "no_request_id"}
        if not isinstance(num_external_tokens, int) \
                or isinstance(num_external_tokens, bool) \
                or num_external_tokens <= 0:
            # THE COMMON CASE: the scheduler calls this for every request, and
            # the fork itself returns at once (`manager.py:406-416`). No counter
            # moves at all for it - this is the step thread's cost for a request
            # that loads nothing.
            return {"captured": 0, "reason": "no_external_tokens"}
        pending = _peek_pending(manager, request_id)
        if pending is None:
            # The fork warns and skips the load for this too
            # (`manager.py:417-426`): no pairing happens, so there is nothing to
            # expect and nothing to say.
            return {"captured": 0, "reason": "no_pending_cpu_hit"}
        cpu_hits = pending[0]
        num_computed = _int_or_none(pending[2])
        if not isinstance(cpu_hits, (list, tuple)) or num_computed is None \
                or num_computed < 0:
            _sched_refused("alloc_pending_shape",
                           "the manager's pending-CPU-hit entry for this request "
                           "is not the fork's (blocks, hit_length, "
                           "num_computed_tokens) triple, so the pairing cannot be "
                           "replicated from it")
            return {"captured": 0, "reason": "pending_shape"}
        sizes = _scheduler_sizes(manager)
        if not sizes["ok"]:
            _sched_unavailable(sizes["kind"], sizes["reason"])
            return {"captured": 0, "reason": sizes["kind"]}
        hashes = getattr(request, "block_hashes", None)
        if not isinstance(hashes, (list, tuple)) or not hashes:
            _sched_unavailable("alloc_no_block_hashes",
                               "the request the fork handed to "
                               "update_state_after_alloc carries no "
                               "block_hashes, so there is no engine key to "
                               "re-derive for the GPU blocks it pairs")
            return {"captured": 0, "reason": "no_block_hashes"}
        getter = getattr(blocks, "get_block_ids", None)
        if not callable(getter):
            _sched_refused("alloc_blocks_shape",
                           "the KVCacheBlocks the scheduler handed over carries no "
                           "callable get_block_ids: the fork moved the per-group "
                           "GPU tables out of the argument the pairing loop "
                           "indexes")
            return {"captured": 0, "reason": "blocks_shape"}
        try:
            gpu_tables = getter()
        except Exception as exc:  # noqa: BLE001 - the fork's objects are not ours
            _sched_refused("alloc_get_block_ids_raised",
                           f"blocks.get_block_ids() raised {type(exc).__name__}")
            return {"captured": 0, "reason": "get_block_ids_raised"}
        if not isinstance(gpu_tables, (list, tuple)) or not gpu_tables:
            _sched_refused("alloc_blocks_table",
                           f"blocks.get_block_ids() answered "
                           f"{type(gpu_tables).__name__}, not per-group block-id "
                           f"tables")
            return {"captured": 0, "reason": "blocks_table"}
        gate = _alloc_gate(manager)
        group_sizes = sizes["group_block_sizes"]
        pairs: list = []
        capped = False
        num_groups = min(len(gpu_tables), len(cpu_hits), len(group_sizes))
        for group in range(num_groups):
            if gate is not None and group not in gate:
                continue
            group_size = _int_or_none(group_sizes[group])
            if not group_size or group_size <= 0:
                continue
            group_gpu = gpu_tables[group]
            group_cpu = cpu_hits[group]
            if not isinstance(group_gpu, (list, tuple)) \
                    or not isinstance(group_cpu, (list, tuple)):
                continue
            start = num_computed // group_size
            for index in range(_cdiv(num_external_tokens, group_size)):
                if len(pairs) >= MAX_EXPECTATIONS_PER_ALLOC:
                    capped = True
                    break
                if index >= len(group_cpu):
                    break
                cpu_block = group_cpu[index]
                if getattr(cpu_block, "is_null", False):
                    continue        # the fork skips a null WITHOUT advancing the
                    # pairing, and so do we: the index is `start + index`, not the
                    # position in the compacted list (`manager.py:464-468`).
                position = start + index
                cpu_id = _int_or_none(getattr(cpu_block, "block_id", None))
                if cpu_id is None:
                    continue
                if position >= len(group_gpu):
                    # The fork's own line would IndexError on this index one
                    # layer down: the accepted load reaches past the table the
                    # scheduler allocated. Same name the thread side uses.
                    with _lock:
                        _count("scheduler_index_out_of_range")
                    break
                gpu_id = _int_or_none(group_gpu[position])
                if gpu_id is None:
                    break
                pairs.append((gpu_id, cpu_id, group, position))
            if capped:
                break
        if capped:
            with _lock:
                _count("scheduler_expectations_capped")
        captured = _capture_expectations(request_id, pairs, hashes, sizes)
        return {"captured": captured, "reason": "capped" if capped else None}
    except Exception as exc:  # noqa: BLE001 - never into the scheduler's step path
        with _lock:
            _count("scheduler_errors")
        _reason("observe_alloc",
                f"the alloc-time witness raised {type(exc).__name__}")
        return {"captured": 0, "reason": f"error_{type(exc).__name__}"}


def _capture_expectations(request_id, pairs, hashes, sizes: dict) -> int:
    """Write the captured pairings into the bounded LRU expectation map.

    Keyed by `(gpu_block, cpu_block)` - the exact pair the fork will emit in its
    metadata's two index-aligned lists, and the only key that can tell two
    pairings of one GPU block (in two groups, or two runs) apart. A re-capture of
    the same pair REPLACES it at the map's newest end: a resumed request
    re-allocing gets its NEW expectation, not the one the first attempt left.
    The `block_hashes` list is held by REFERENCE - the same list the stability
    witness re-derives from, and the reason the map is capped rather than aged:
    a pinned reference to a dead request's hashes is a bounded memory fact, and
    the cap is the bound.
    """
    if not pairs:
        return 0
    now = time.monotonic()
    group_sizes = list(sizes["group_block_sizes"])
    captured = 0
    with _lock:
        _count("scheduler_allocs_observed")
        expectations = _state["scheduler_expectations"]
        for gpu_id, cpu_id, group, position in pairs:
            while len(expectations) >= MAX_SCHED_EXPECTATIONS:
                oldest = next(iter(expectations))
                evicted = expectations.pop(oldest)
                if not evicted.get("consumed"):
                    _count("scheduler_expectations_dropped")
            key = (gpu_id, cpu_id)
            expectations.pop(key, None)      # re-insert at the newest end
            expectations[key] = {
                "rid": request_id[:MAX_SCHED_ID_CHARS], "group": int(group),
                "index": int(position), "hashes": hashes,
                "hash_block_size": int(sizes["hash_block_size"]),
                "group_block_sizes": group_sizes, "mono": now,
                "consumed": False}
            _count("scheduler_expectations_captured")
            captured += 1
    return captured


def _take_expectation(gpu_block: int, cpu_block: int) -> dict | None:
    """The captured expectation for one emitted pairing, marked consumed. NOT removed.

    Consumption is a MARK, not a pop, for the very failure mode that made 0.2.40
    decorative: a load state is emitted in EVERY step until the worker confirms
    the load, so the same pairing can be sampled twice - and the second sampling
    deserves the same verdict, not a "no_captured_expectation". The map is bounded
    by eviction, not by consumption.
    """
    with _lock:
        record = _state["scheduler_expectations"].get((int(gpu_block),
                                                       int(cpu_block)))
        if record is None:
            return None
        if not record.get("consumed"):
            record["consumed"] = True
            _count("scheduler_expectations_consumed")
        return dict(record)


def _scheduler_record(entry) -> dict | None:
    """One offered entry PLUS the expectation captured for its pairing.

    The name we offered and the group our ledger believed (from the entry
    `disk_restore._build_load_paths` wrote) on one side; the fork's own
    alloc-time decision - group, and the position in that group's GPU block
    table, plus the `block_hashes` to re-derive from - on the other. What makes
    this oracle able to see a shift is that the (group, index) side comes from
    the alloc hook and NOT from either list in the metadata: a shift moves both
    lists together, and neither of them can contradict the other. NOTHING IS
    HASHED HERE; `hashes` travels by reference to the module's one thread.
    """
    if not isinstance(entry, dict):
        return None
    request_id = entry.get("request_id")
    gpu_block = _int_or_none(entry.get("gpu_block"))
    cpu_block = _int_or_none(entry.get("cpu_block"))
    key_hex = entry.get("key")
    if not isinstance(request_id, str) or not request_id:
        _sched_refused("no_request_id", "the offer witness handed an entry with no "
                                        "request id, so it can be neither "
                                        "attributed nor reported")
        return None
    if not isinstance(key_hex, str) or not key_hex:
        _sched_refused("no_key", "the ledger's identity for this entry carries no "
                                 "key, so there is no offered name to re-derive")
        return None
    split = disk_layout.split_block_key(key_hex)
    if not split.get("ok"):
        _sched_refused("offered_key_shape", f"the key we offered this entry under is "
                                            f"not a block hash: {split.get('reason')}")
        return None
    if gpu_block is None:
        _sched_refused("no_gpu_block", "this step's metadata carries no GPU block id "
                                       "beside this CPU slot (the two lists are not "
                                       "aligned), so there is no destination to ask "
                                       "about")
        return None
    if cpu_block is None:
        _sched_refused("no_cpu_block", "this step's metadata carries no CPU slot id "
                                       "beside this GPU block, so there is no "
                                       "pairing to look an expectation up for")
        return None
    expectation = _take_expectation(gpu_block, cpu_block)
    if expectation is None:
        _sched_unavailable(
            "no_captured_expectation",
            "no captured expectation covers this (GPU block, CPU slot) pairing: "
            "the alloc hook is not wired in this process (`scheduler_oracle_hook` "
            "says so), the expectation was evicted under the map cap "
            "(`scheduler_expectations_dropped` counts that), the per-alloc cap "
            "stopped it (`scheduler_expectations_capped`), or this load was never "
            "paired by update_state_after_alloc at all")
        return None
    group = _int_or_none(expectation.get("group"))
    index = _int_or_none(expectation.get("index"))
    hashes = expectation.get("hashes")
    group_sizes = expectation.get("group_block_sizes") or ()
    if group is None or group < 0 or index is None or index < 0 \
            or not isinstance(hashes, (list, tuple)) or not hashes:
        _sched_unavailable("captured_expectation_incomplete",
                           "the expectation captured at alloc time lacks a group, "
                           "an index, or the engine's block_hashes, so there is "
                           "nothing for the thread to re-derive from it")
        return None
    group_size = (group_sizes[group] if group < len(group_sizes) else None)
    if not group_size:
        _sched_unavailable("no_group_block_size",
                           f"the manager gave no usable `group_block_sizes` entry "
                           f"for group {group} at alloc time, so the engine's view "
                           f"of it cannot be built at emit time")
        return None
    with _lock:
        _state["scheduler_seq"] = int(_state["scheduler_seq"]) + 1
        record_id = int(_state["scheduler_seq"])
    return {"id": record_id,
            "rid": str(expectation.get("rid") or "")[:MAX_SCHED_ID_CHARS],
            "group": int(group), "span": _int_or_none(entry.get("span")),
            "key": key_hex[:72], "key16": split["key16"],
            "name_group": int(split["group_id"]),
            "cpu_block": int(cpu_block), "gpu_block": int(gpu_block),
            "index": int(index), "hashes": hashes,
            "hash_block_size": int(expectation.get("hash_block_size") or 0),
            "group_bytes": int(group_size),
            "group_block_sizes": list(group_sizes),
            "mono": time.monotonic()}


def _retain_scheduler(record: dict) -> bool:
    """Queue one scheduler-side record on the module's ONE thread. False = dropped."""
    with _lock:
        pending = _state["scheduler_pending"]
        while len(pending) >= MAX_RETAINED_JOBS:
            pending.pop(next(iter(pending)))
            _count("scheduler_retained_dropped")
        pending[record["id"]] = record
        _count("scheduler_retained")
    if not _ensure_thread():
        with _lock:
            _count("scheduler_thread_unavailable")
            _state["scheduler_pending"].pop(record["id"], None)
        return False
    return True


# --- THE OFFER POINT (0.2.40): THE NAME COMPARISON -----------------------------
# The second half of the check: `disk_restore._build_load_paths` hands us the
# entries whose file this plugin's ledger NAMED, and each one is compared against
# the expectation the alloc hook captured for its pairing. The hashing happens on
# this module's thread (`_verify_scheduler_inner`), never here.
def observe_offer(manager, scheduler_output, entries) -> dict:  # noqa: ARG001
    """THE SCHEDULER DOOR: does the name we offered match the captured expectation?

    Called by `disk_restore._build_load_paths` after the length rule has been
    satisfied and the paths have been written, with the entries of this step that
    are OURS (the ones whose CPU slot this module's ledger named a file for).
    Bookkeeping only: it cannot change the step, cannot prune an entry and cannot
    reorder a list, and every path inside is a counter.

    `manager` and `scheduler_output` are ACCEPTED AND NOT READ (0.2.41). 0.2.40
    resolved the request's GPU block table from
    `scheduler_output.kv_connector_block_state` HERE, and that table is the
    CURRENT step's - so a load state emitted in a later step than its alloc
    (the normal life of a multi-step load) had no table, and the oracle went
    blind exactly when it mattered. The expectation now arrives EARLY, from
    `observe_alloc`, and leaves the emission timing irrelevant. The parameters
    stay so the call site does not have to know the oracle stopped reading them.

    The sampling decision is taken BEFORE any expectation is touched, so an
    unarmed or unsampled step costs this role nothing at all.
    """
    try:
        if not effective_enabled():
            return {"seen": 0, "retained": 0, "reason": "disabled"}
        if not entries:
            return {"seen": 0, "retained": 0, "reason": "no entries"}
        with _lock:
            _count("scheduler_steps_seen")
        scanned = list(entries)[:MAX_ENTRIES_SCANNED]
        seen = 0
        retained = 0
        for entry in scanned:
            seen += 1
            with _lock:
                _count("scheduler_entries_seen")
                cursor = int(_state["scheduler_sample_cursor"])
                _state["scheduler_sample_cursor"] = cursor + 1
                selected = (cursor % max(1, SAMPLE_EVERY_N_RESTORES)) == 0
                room = retained < MAX_RETAINED_PER_STEP
            if not selected or not room:
                # The per-step bound is a BOUND: hitting it is normal on a wide
                # step and one reason per step would be a reason per step.
                with _lock:
                    _count("scheduler_sampled_out")
                continue
            record = _scheduler_record(entry)
            if record is None:
                continue      # the refusal was counted and named where it happened
            if not _retain_scheduler(record):
                break
            retained += 1
        return {"seen": seen, "retained": retained, "reason": None}
    except Exception as exc:  # noqa: BLE001 - never into the scheduler's step path
        with _lock:
            _count("scheduler_errors")
        _reason("observe_offer",
                f"the scheduler-side witness raised {type(exc).__name__}")
        return {"seen": 0, "retained": 0, "reason": f"error_{type(exc).__name__}"}


def _scheduler_helpers() -> dict:
    """The engine's OWN two view-builders, or a named unavailability.

    `disk_restore._key_helpers()` is the one resolver of `make_block_hash_with_group_id`
    and `resolve_block_hashes` in this package, so the offer that NAMED the file and
    this check that RE-DERIVES the name go through the same two functions. Nothing here
    re-hashes or re-packs: a plugin that invented its own derivation would eventually
    disagree with the engine about which bytes are the group, and that disagreement is a
    file read under the wrong name - the exact failure this oracle exists to catch.
    """
    from . import disk_restore  # noqa: PLC0415 - it imports nothing that imports us
    try:
        helpers = disk_restore._key_helpers()
    except Exception as exc:  # noqa: BLE001 - a resolver that raises is a refusal
        return {"ok": False, "kind": "key_helpers_raised",
                "reason": f"_key_helpers raised {type(exc).__name__}"}
    if not helpers.get("ok"):
        return {"ok": False, "kind": helpers.get("reason_kind") or "no_key_helpers",
                "reason": str(helpers.get("reason"))}
    return {"ok": True, "pack": helpers["pack"], "resolve": helpers["resolve"]}


# THE HASH VIEW IS A UNION, AND IT IS NOT A SEQUENCE (0.2.42, the reason the witness
# was decorative in production). `resolve_block_hashes` is declared to answer
# `BlockHashList`, and `BlockHashList` is `list[BlockHash] | BlockHashListWithBlockSize`
# (`kv_cache_utils.py:2850`). The second arm is a LAZY VIEW THAT IS NOT A SEQUENCE: it
# neither inherits from nor registers `collections.abc.Sequence`
# (`kv_cache_utils.py:2783-2848`), so 0.2.40/0.2.41's `isinstance(view, (list, tuple))`
# refused it on a box whose groups are NOT the hash granularity - which is THIS box, and
# the live result was `scheduler_checked = 0` beside `scheduler_oracle_refused = 2`,
# reason `resolve_shape: ... BlockHashListWithBlockSize, not a sequence`. Refusing a
# shape this module does not understand was the RIGHT move; the understood shape list
# was simply too small. What the comparison needs is two operations, so those two
# operations are now the test: `len(view)` and integer `view[i]`.
#
# AND THE VIEW IS INDEXED, NEVER THE LIST UNDERNEATH IT. The lazy view's own arithmetic
# is `_get_value_at(i) = raw[(i + 1) * scale_factor - 1]` and `__len__ = len(raw) //
# scale_factor` (`kv_cache_utils.py:2845-2848`, `:2824-2825`), with `scale_factor =
# target_block_size // hash_block_size`. On the live box `hash_block_size` is 40 and the
# serving groups are 3200-token blocks, so the factor is 80: `view[i]` is `raw[i*80+79]`
# and the view is 80 times SHORTER than the list it reads. Indexing the raw list at `i`
# therefore does not produce a near miss, it produces a digest for tokens `[0, 40)`
# against a file named for tokens `[0, 3200)` - a DIFFERENT TOKEN RANGE, which would
# read as a storm of false mismatches or as accidental agreement. Everything here goes
# through `view[index]`; `view.block_hashes` is never read.
def _hash_view_is_indexable(view) -> bool:
    """May this `resolve_block_hashes` answer be compared? Capability, not lineage.

    Checked on `type(view)`, because that is where Python looks up `__len__` and
    `__getitem__` for `len(view)` / `view[i]`; and `len()` is CALLED, not just looked
    for, because a view whose length cannot be computed has no usable protocol (a view
    built with `scale_factor == 0` under `python -O`, where the fork's assert is
    stripped, divides by zero exactly here).
    """
    kind = type(view)
    if not callable(getattr(kind, "__getitem__", None)) \
            or not callable(getattr(kind, "__len__", None)):
        return False
    try:
        length = len(view)
    except Exception:  # noqa: BLE001 - an unusable length is not a view to compare
        return False
    return isinstance(length, int) and not isinstance(length, bool) and length >= 0


def _scheduler_view(helpers: dict, record: dict, group_bytes=None,
                    viewed_group=None) -> dict:
    """One call to the engine's `resolve_block_hashes`, or a named refusal.

    `viewed_group` names the group whose block size is being resolved WHEN IT IS NOT
    the record's own: the classification walk asks every OTHER group for its view, and a
    refusal that blamed the record's group would send an operator to the wrong one.

    THE GROUP-4 SHAPE IS CONTAINED HERE, AND NAMED. `group_block_sizes` on this box is
    `[3200, 3200, 3200, 3200, 8, 3200]` against a `hash_block_size` of 40, and the
    fork's view-builder requires the target to be a MULTIPLE of the hash granularity,
    asserting it twice (`kv_cache_utils.py:2826`, in
    `BlockHashListWithBlockSize.__init__`, and `:2887`, in `resolve_block_hashes`).
    Group 4 is the 13,824 B QSA ring: parked, never prefix-cacheable, never spilled, so
    the oracle should never be pointed at it - but the classification walk enumerates
    EVERY group of the record, so a 40-into-8
    group is asked for on every mismatch this box produces, and the fork's answer is an
    `AssertionError`. This module does not re-derive that arithmetic (a group block size
    is not required to be a multiple of the hash granularity by anything except this
    one assert, and a plugin that guessed the rule would refuse views the fork is happy
    to build); it ASKS, and contains what comes back. `AssertionError` is named on its
    own - `scheduler_hash_view_unbuildable`, with a reason - because on this box it has
    exactly one cause and an operator needs to see that cause, not a generic
    `resolve_raised`. Nothing here reaches the caller as an exception.
    """
    target = int(record["group_bytes"] if group_bytes is None else group_bytes)
    named = record.get("group") if viewed_group is None else viewed_group
    hash_block = record.get("hash_block_size")
    try:
        view = helpers["resolve"](record["hashes"], record["hash_block_size"], target)
    except AssertionError as exc:  # noqa: BLE001 - the fork's own refusal, contained
        # The fork's `target % hash == 0` assert, fired by a group this box cannot view
        # at its own block size. It is not an engine fault and must not read as one:
        # group 4 on this box walks into it on EVERY mismatch classification, so the
        # generic `resolve_raised` name would bury a standing fact under what is
        # supposed to mean "the engine's helper broke".
        with _lock:
            _count("scheduler_hash_view_unbuildable")
        detail = (f"resolve_block_hashes refused group {named}'s "
                  f"{target}-token blocks on a {hash_block}-token hash "
                  f"(AssertionError{f': {exc}'[:60]}): the fork's own refusal that "
                  "this group has no view at its own block size")
        _reason("hash_view_unbuildable", detail)
        return {"ok": False, "kind": "resolve_asserted", "reason": detail}
    except Exception as exc:  # noqa: BLE001 - the engine's helper is not ours to trust
        return {"ok": False, "kind": "resolve_raised",
                "reason": f"resolve_block_hashes raised {type(exc).__name__}"}
    if not _hash_view_is_indexable(view):
        return {"ok": False, "kind": "resolve_shape",
                "reason": f"resolve_block_hashes answered {type(view).__name__}, "
                          f"which answers neither `len()` nor integer `__getitem__`, "
                          "so it is a shape this module refuses to compare (the fork's "
                          "declared answer is `list[BlockHash] | "
                          "BlockHashListWithBlockSize`, and a lazy non-`Sequence` view "
                          "IS one of the two legitimate shapes)"}
    return {"ok": True, "view": view}


def _scheduler_key16(helpers: dict, view, index, group) -> dict:
    """The `key16` the engine says block `index` of group `group` is named under."""
    if index is None or index < 0 or index >= len(view):
        return {"ok": False, "kind": "index_out_of_range",
                "reason": f"index {index} is outside a resolved view of "
                          f"{len(view)} entries"}
    digest = view[index]
    if not isinstance(digest, bytes):
        return {"ok": False, "kind": "digest_not_bytes",
                "reason": f"the engine's block hash at index {index} is "
                          f"{type(digest).__name__}, not bytes"}
    try:
        key_hex = bytes(helpers["pack"](digest, int(group))).hex()
    except Exception as exc:  # noqa: BLE001 - the packer is not ours either
        return {"ok": False, "kind": "pack_raised",
                "reason": f"make_block_hash_with_group_id raised "
                          f"{type(exc).__name__}"}
    split = disk_layout.split_block_key(key_hex)
    if not split.get("ok"):
        return {"ok": False, "kind": "key_shape",
                "reason": f"the recomputed key is not a block hash: "
                          f"{split.get('reason')}"}
    return {"ok": True, "key16": split["key16"], "group_id": split["group_id"]}


def _scheduler_classify(record: dict, helpers: dict, view) -> tuple:
    """Which addressing failure this is: shift, group, or neither (yet).

    The order is the order an operator acts in. The name's OWN group suffix is checked
    first because it is a fact about our own run that needs no engine input at all: the
    engine carries the group INSIDE the key (`kv_cache_utils.py:83`, read back by
    `disk_layout.split_block_key`, the one place this package reads that suffix), so a
    name whose suffix contradicts the group the entry is paired with is a group swap
    inside our own walk. THEN the neighbour window, then the other groups' views of the
    same index.
    """
    group = int(record["group"])
    index = record["index"]
    if record.get("name_group") is not None and int(record["name_group"]) != group:
        return (SCHED_CLASS_GROUP, None,
                f"a group swap on our own offer path: the name we offered ends with "
                f"group id {record['name_group']} while the entry is paired with group "
                f"{group}. The group a key belongs to is carried inside the key "
                "itself, so this contradicts itself without any engine input")
    for delta in range(1, NEIGHBOUR_WINDOW + 1):
        for candidate in (int(index) + delta, int(index) - delta):
            found = _scheduler_key16(helpers, view, candidate, group)
            if found["ok"] and found["key16"] == record["key16"]:
                return (SCHED_CLASS_SHIFT, (delta if candidate > index else -delta),
                        f"a one-slot shift ({delta:+d}): the name carries"
                        f" the digest of block index {candidate} while the GPU"
                        f" block the fork paired it with is the engine's block index "
                        f"{index}. A shift moves both index-aligned lists"
                        " together, so it is invisible to every list-vs-list check this"
                        " plugin has")
    try:
        for other, size in enumerate(record.get("group_block_sizes") or ()):
            if other == group or not size:
                continue
            other_view = _scheduler_view(helpers, record, group_bytes=size,
                                         viewed_group=other)
            if not other_view["ok"]:
                # NOT a silent skip: the one place a view is BUILT counts its own
                # failures (`scheduler_hash_view_unbuildable` moves for group 4's
                # 40-into-8 shape on every classification walk that reaches it), so a
                # group this walk cannot consult is a counted group.
                continue
            found = _scheduler_key16(helpers, other_view["view"], index, other)
            if found["ok"] and found["key16"] == record["key16"]:
                return (SCHED_CLASS_GROUP, None,
                        f"a group swap: the name carries group {other}'s digest for "
                        f"block index {index} while the entry is paired with group "
                        f"{group}. The digest is real but belongs to another group's "
                        "hash view, so a file is read for one group and restored as "
                        "another")
    except Exception:  # noqa: BLE001 - a classification walk never loses the mismatch
        return (SCHED_CLASS_UNKNOWN, None,
                "the mismatch is real but its classification walk raised, so it is "
                "reported under its own name rather than guessed at")
    return (SCHED_CLASS_UNKNOWN, None,
            "no shape this module has a word for: the name's digest matches neither "
            f"index {index} for group {group}, nor a neighbour within "
            f"{NEIGHBOUR_WINDOW}, nor another group's view of that index - a wrong "
            "name reported under its own name rather than guessed at")


def _scheduler_sample(record: dict, expected=None, classification=None, offset=None,
                      stable=None) -> dict:
    """The diagnosable shape of one scheduler-side verdict, clipped and bounded.

    Every field here answers a question an operator asks of a bad name: WHICH REQUEST,
    which group, which block, what the file was named for, what the engine says it
    should have been named for, and what KIND of wrong that is - with `offset` the shift
    and `index` the position the pairing implied.

    It carries NO sentence, and that is a byte-budget decision rather than an oversight:
    the explanation of a scheduler mismatch is the `scheduler_mismatch:<classification>`
    line in `reasons`, which says it once per finding in full, while a sample's numbers
    are what makes a rank file actionable. `span`/`cpu_block`/`table_len` are out for
    the same reason - the ledger and the path already carry them.
    """
    return {"request_id": record["rid"], "group": record["group"],
            "key16": record["key16"], "expected_key16": expected,
            "name_group": record.get("name_group"),
            "gpu_block": record["gpu_block"],
            "index": record["index"],
            "classification": classification, "offset": offset, "stable": stable}


def _scheduler_mismatch(record: dict, classification: str, detail: str,
                        expected=None, offset=None) -> dict:
    """A scheduler-side finding: classify it, re-derive it, publish the sample."""
    stable = None
    if expected is not None:
        # THE SCHEDULER SIDE'S STABILITY WITNESS. The worker re-reads a GPU block
        # because the engine hands blocks out again; the scheduler's inputs are two
        # Python objects the engine mutates as a request runs (`block_hashes` grows,
        # and a preemption can move a table), so the equivalent question is whether the
        # SAME captured inputs re-derive the SAME expected key. If they do not, the
        # disagreement came from reading the view mid-mutation and the sample is not
        # evidence.
        helpers = _scheduler_helpers()
        again = None
        if helpers["ok"] and record.get("index") is not None:
            view = _scheduler_view(helpers, record)
            if view["ok"]:
                again = _scheduler_key16(helpers, view["view"], record["index"],
                                         record["group"])
        stable = bool(again and again["ok"]
                      and again["key16"] == expected)
    sample = _scheduler_sample(record, expected=expected,
                               classification=classification, offset=offset,
                               stable=stable)
    with _lock:
        _count("scheduler_mismatch")
        _count(_SCHED_CLASS_COUNTERS.get(classification, "scheduler_mismatch_unknown"))
        _count("scheduler_mismatch_stable" if stable
               else "scheduler_mismatch_unstable")
        samples = _state["scheduler_mismatch_samples"]
        samples.append(dict(sample))
        while len(samples) > MAX_SCHED_MISMATCH_SAMPLES:
            samples.pop(0)
    _reason(f"scheduler_mismatch:{classification}", detail)
    return {"checked": True, "equal": False, "state": "scheduler_mismatch",
            "classification": classification, "stable": stable, "offset": offset,
            "expected_key16": expected, "reason": detail[:MAX_REASON_CHARS]}


def _verify_scheduler_record(record: dict) -> dict:
    """One retained offered entry, on this module's thread. Never raises."""
    try:
        return _verify_scheduler_inner(record)
    except Exception as exc:  # noqa: BLE001 - an entry never kills the pass
        with _lock:
            _count("scheduler_errors")
        _reason("scheduler_verify",
                f"verifying one scheduler entry raised {type(exc).__name__}")
        return {"checked": False, "reason": f"error_{type(exc).__name__}"}


def _verify_scheduler_inner(record: dict) -> dict:
    """The comparison, in the order that makes each refusal nameable on its own.

    There is no "the destination is not in the request's table" branch any more,
    and there cannot be: at the alloc hook the index comes FROM the fork's own
    pairing decision, so the old `gpu_block_not_in_table` shape has no observable
    moment left at this site. Its replacement - an emitted entry whose pairing was
    never captured - is a NAMED `scheduler_oracle_unavailable` at the door
    (`_scheduler_record`), never a silent skip and never a guessed mismatch.
    """
    if (time.monotonic() - float(record.get("mono") or 0.0)) > JOB_MAX_AGE_SECONDS:
        with _lock:
            _count("scheduler_retained_dropped")
        return {"checked": False, "reason": "stale"}
    if record.get("index") is None:
        # Unreachable by construction (the expectation's index is an int or the
        # record was refused at the door); counted as a refusal, never a mismatch.
        return _sched_refused("no_captured_index",
                              "a retained scheduler record reached the thread with "
                              "no captured index, which the door guarantees it "
                              "cannot have")
    helpers = _scheduler_helpers()
    if not helpers["ok"]:
        _sched_unavailable(helpers["kind"], helpers["reason"])
        return {"checked": False, "reason": helpers["kind"]}
    view = _scheduler_view(helpers, record)
    if not view["ok"]:
        if view["kind"] == "index_out_of_range":
            with _lock:
                _count("scheduler_index_out_of_range")
        _sched_refused(view["kind"], view["reason"])
        return {"checked": False, "reason": view["kind"]}
    expected = _scheduler_key16(helpers, view["view"], record["index"],
                                record["group"])
    if not expected["ok"]:
        if expected["kind"] == "index_out_of_range":
            with _lock:
                _count("scheduler_index_out_of_range")
        _sched_refused(expected["kind"], expected["reason"])
        return {"checked": False, "reason": expected["kind"]}
    # THE NAME HAS TWO HALVES, AND BOTH ARE CHECKED. `key16` is the digest part, and it
    # is what the equality comparison compares - but the FILE is named by the digest AND
    # the group (`disk_layout.build_slot_path` writes the group into the file name), so
    # a name whose own suffix names a different group than the entry is paired with
    # points at a different file, whatever its digest says. A `key16`-only equality
    # would call that a match; it is the one contradiction this module can see with no
    # engine input at all, so it outranks the digest comparison rather than following
    # it.
    name_conflict = (record.get("name_group") is not None
                     and int(record["name_group"]) != int(record["group"]))
    if expected["key16"] == record["key16"] and not name_conflict:
        with _lock:
            _count("scheduler_checked")
            _count("scheduler_equal")
            _state["scheduler_last_oracle"] = {
                "equal": True, "expected_key16": expected["key16"],
                "name_key16": record["key16"], "group": record["group"],
                "index": record["index"], "gpu_block": record["gpu_block"],
                "request_id": record["rid"]}
        return {"checked": True, "equal": True, "state": "scheduler_equal",
                "index": record["index"], "key16": record["key16"],
                "expected_key16": expected["key16"]}
    classification, offset, detail = _scheduler_classify(record, helpers, view["view"])
    return _scheduler_mismatch(record, classification, detail,
                               expected=expected["key16"], offset=offset)


def _drain_scheduler(batch) -> tuple:
    """Verify the scheduler records this pass took. Returns (results, comparisons)."""
    results: list = []
    checked = 0
    if not batch:
        return results, checked
    with _lock:
        for record in batch:
            _state["scheduler_pending"].pop(record["id"], None)
    for record in batch:
        verdict = _verify_scheduler_record(record)
        if verdict.get("checked"):
            checked += 1
        results.append({"id": record["id"], **verdict})
    return results, checked


def scheduler_state() -> dict:
    """The scheduler-side oracle's state. A stable key set on every path.

    NOTE THE ASYMMETRY WITH the worker's `oracle_state()`, and it is deliberate: the
    worker's `unreachable_reason` is a PERMANENT fact of the fork (`metadata.py` says a
    worker has no request identities), so it is a fixed sentence. The scheduler's
    `unreachable_reason` is a FAULT: it names why the oracle could not be built on the
    one process that CAN build it, and a non-None value there means the addressing check
    is not running. When nothing has been unavailable yet, the field says so rather than
    reading like a missing feature.

    It carries NO `last` and NO role string, on purpose: `scheduler_mismatch_samples`
    and `roles` already publish those, and a section whose byte budget is enforced by
    REFUSAL cannot afford to say the same thing twice.
    """
    with _lock:
        counters = _state["counters"]
        return {"unreachable_reason": (_state["scheduler_unreachable_reason"]
                                       or SCHED_NOTHING_UNAVAILABLE),
                "checked": int(counters["scheduler_checked"]),
                "equal": int(counters["scheduler_equal"]),
                "mismatch": int(counters["scheduler_mismatch"]),
                "unavailable": int(counters["scheduler_oracle_unavailable"]),
                "refused": int(counters["scheduler_oracle_refused"]),
                "index_out_of_range": int(counters["scheduler_index_out_of_range"])}


# --- THE PASS ------------------------------------------------------------------
def run_pass(worker=None) -> dict:
    """At most `MAX_CHECKS_PER_PASS` retained entries PER ROLE. One pass; never raises.

    The thread's body, and the function a test drives directly (the `disk_gpu_shadow`
    idiom: monkeypatch `_ensure_thread` to keep the thread off and call this).

    THE SCHEDULER HALF GOES FIRST, on purpose: it needs no worker, so a `worker_gone` in
    the middle of a shutdown cannot strand the offered-entry verdicts - which are the
    verdicts about the SCHEDULER's own addressing, taken while the scheduler was alive.
    """
    try:
        if not effective_enabled():
            return {"ran": False, "reason": "disabled", "checked": 0,
                    "scheduler_checked": 0}
        now = time.monotonic()
        with _lock:
            last = _state["last_pass_mono"]
            if last is not None and (now - float(last)) < MIN_INTERVAL_SECONDS:
                _count("throttled_interval")
                return {"ran": False, "reason": "throttled_interval", "checked": 0,
                        "scheduler_checked": 0}
            _state["last_pass_mono"] = now
            _state["pass_used"] = 0
            _state["recheck_used"] = 0
            batch = [record for record in list(_state["pending"].values())
                     [:MAX_CHECKS_PER_PASS]]
            sched_batch = [record for record
                           in list(_state["scheduler_pending"].values())
                           [:MAX_CHECKS_PER_PASS]]
            if not batch and not sched_batch:
                return {"ran": True, "reason": "nothing retained", "checked": 0,
                        "scheduler_checked": 0}
            _count("passes")
        drained = _contained(lambda: _drain_scheduler(sched_batch)) or ([], 0)
        sched_results, sched_checked = drained
        target = _resolve_worker(worker)
        if target is None:
            # Every retained entry goes with it: they were retained against a worker
            # that is gone, and re-verifying them against a future one would compare one
            # boot's file against another boot's GPU. The SCHEDULER records are
            # untouched by that reasoning, which is why they were drained above.
            with _lock:
                _count("worker_gone")
                dropped = len(_state["pending"])
                _state["pending"] = {}
            _reason("worker_gone",
                    f"the worker this module retained is gone, so {dropped} retained "
                    "entries were dropped unread")
            return {"ran": True, "reason": "worker_gone", "checked": 0,
                    "scheduler_checked": sched_checked,
                    "scheduler_results": sched_results}
        results = []
        still_waiting: list = []
        for record in batch:
            verdict = _verify_record(target, record)
            if verdict.get("reason") == "waiting_for_dma":
                still_waiting.append(record)
                continue
            with _lock:
                _state["pending"].pop(record["id"], None)
            results.append({"id": record["id"], **verdict})
        if still_waiting:
            with _lock:
                for record in still_waiting:
                    _state["pending"][record["id"]] = record
        return {"ran": True, "reason": None,
                "checked": sum(1 for item in results if item.get("checked")),
                "waiting": len(still_waiting), "results": results,
                "scheduler_checked": sched_checked,
                "scheduler_results": sched_results}
    except Exception as exc:  # noqa: BLE001 - a pass never escapes its thread
        with _lock:
            _count("errors")
        _reason("run_pass", f"the pass raised {type(exc).__name__}")
        return {"ran": False, "reason": f"error_{type(exc).__name__}", "checked": 0}


# --- THE VIEW ------------------------------------------------------------------
def counters() -> dict:
    with _lock:
        return dict(_state["counters"])


def scheduler_counters() -> dict:
    """Only the `scheduler_`-prefixed ones, for a reader who must not add them to the
    worker's."""
    with _lock:
        return {key: int(_state["counters"][key]) for key in _COUNTER_KEYS
                if key.startswith("scheduler_")}


def _dump_size(payload) -> int:
    """The byte count `worker_diag` will measure, in ITS encoding (`separ=(",", ":")`,
    `default=str`), because that is the number the cap is applied to."""
    return len(json.dumps(payload, separators=(",", ":"),
                          default=str).encode("utf-8"))


def _counted_trim(out: dict) -> dict:
    """Count a trim ONCE PER SHAPE OF TRIM, and write the count back into the payload.

    A trim is a real event, but `section()` is also what a polling dashboard calls every
    few seconds - so an un-deduplicated counter would tick with the polling and nobody
    could read it. What is worth counting is a CHANGE: the section began shedding a
    different set of detail than it did last time. `trimmed` in the payload still says
    exactly what is missing NOW, and the count is written back INTO the payload because
    `out`'s counters were snapshotted before this trim happened.
    """
    signature = tuple(out["trimmed"])
    with _lock:
        if signature != _state["last_trim"]:
            _state["last_trim"] = signature
            _count("section_trimmed")
        out["section_trimmed"] = int(_state["counters"]["section_trimmed"])
    return out


def _fit(out: dict) -> dict:
    """Never hand the writer a section it would REFUSE - and say what was shed.

    `worker_diag` refuses an over-cap section rather than truncating it, so an over-fat
    payload's failure mode is that the ENTIRE `disk_restore_verify` key vanishes from
    the rank file on the boot with the most to report. So the section bounds itself, in
    the order a reader acts on, numbered in the body: the reason TEXT first (the leading
    tag survives - the tag is what names the category), then the boundary-group
    illustration lists (the COUNTS above them never move), then the scheduler's mismatch
    SAMPLES oldest first, then the explanatory PROSE (clipped, never removed as keys),
    then the reasons themselves.

    Illustration goes before explanation on purpose. `roles`/`oracle_rule` are what make
    an unprefixed `oracle_unavailable` readable as "expected, this is a worker" and not
    as "the check is not running here", and that distinction is the 0.2.40 role split; a
    second mismatch sample is a nicer-to-read version of a fact the counters and the
    `reasons` ring already carry in full.

    The counters and `last_mismatch` are NEVER trim targets: they are why the section
    exists, they are fixed-size, and a trimmed section that gained or lost a key is a
    section a reader cannot diff. Every top-level key survives every step, which is what
    keeps "the key set never changes" true under pressure too - a clipped value is not a
    missing key.
    """
    trimmed = out["trimmed"] = []
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return out
    # 1. THE REASON TEXT FIRST. The leading tag survives - the tag is what names the
    # category - so a clipped reason still says WHICH condition is standing.
    out["reasons"] = [str(line)[:TRIM_REASON_CHARS] for line in out["reasons"]]
    trimmed.append(f"reason_chars:{TRIM_REASON_CHARS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    # 2. THE BOUNDARY GROUPS' ILLUSTRATION LISTS. The COUNTS above them never move, so
    # this costs the reader a list of which groups and buys back a hundred bytes.
    boundary = out["boundary"]
    if isinstance(boundary, dict):
        boundary["missing_by_group"] = dict(list(boundary["missing_by_group"].items())
                                            [:TRIM_BOUNDARY_GROUPS])
        boundary["groups"] = list(boundary["groups"])[:TRIM_BOUNDARY_GROUPS]
        trimmed.append(f"boundary:{TRIM_BOUNDARY_GROUPS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    # 3. THE SCHEDULER'S SAMPLES, oldest first. They are the scheduler half's only
    # variable-size field and their findings are DUPLICATED in kind: every one is a
    # counter, and their sentences are in `reasons`. They go BEFORE the explanatory
    # prose, because the sentences telling a worker's number from a scheduler's are what
    # the 0.2.40 role split exists to publish, and a reader who cannot tell which
    # process wrote a number cannot act on it - while one sample instead of two still
    # leaves the newest finding and every number.
    samples = out.get("scheduler_mismatch_samples")
    if isinstance(samples, list) and len(samples) > 1:
        out["scheduler_mismatch_samples"] = samples[-1:]
        trimmed.append("scheduler_samples:1")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    if isinstance(samples, list) and samples:
        out["scheduler_mismatch_samples"] = []
        trimmed.append("scheduler_samples")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    # 4. THE EXPLANATORY PROSE, clipped (never removed as keys). Every counter, every
    # oracle number and every last-verdict field survives this step untouched.
    clipped = 0
    for key in ("note", "oracle_rule"):
        if isinstance(out.get(key), str) and len(out[key]) > TRIM_REASON_CHARS:
            out[key] = out[key][:TRIM_REASON_CHARS]
            clipped += 1
    for name in ("oracle", "scheduler_oracle"):
        block = out.get(name)
        reason = block.get("unreachable_reason") if isinstance(block, dict) else None
        if isinstance(reason, str) and len(reason) > TRIM_REASON_CHARS:
            block["unreachable_reason"] = reason[:TRIM_REASON_CHARS]
            clipped += 1
    if isinstance(out.get("roles"), dict):
        for key in ("worker", "scheduler", "meaning"):
            value = out["roles"].get(key)
            if isinstance(value, str) and len(value) > TRIM_REASON_CHARS:
                out["roles"][key] = value[:TRIM_REASON_CHARS]
                clipped += 1
    if clipped:
        trimmed.append(f"notes:{TRIM_REASON_CHARS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    out["reasons"] = []
    out["boundary"]["last"] = None
    trimmed.append("reasons_dropped")
    out = _counted_trim(out)
    if _dump_size(out) > MAX_SECTION_BYTES:
        # Every shed has been taken and it is still over budget, so the writer will
        # refuse the section and this key simply will not be in the rank file. Nothing
        # is left to shed without shedding a NUMBER, so the payload goes out honest and
        # the LOG says what the file cannot.
        _note_once("section-over-budget",
                   f"[disk_restore_verify] section STILL over its byte budget after "
                   f"every trim step ({_dump_size(out)} B over {MAX_SECTION_BYTES} B): "
                   "the rank writer will refuse it, so `disk_restore_verify` will be "
                   "absent from diagnostics-rank.json", level="error")
    return out


def geometry_view() -> dict:
    """The numbers the comparisons used, or the stable placeholder."""
    with _lock:
        saved = _state["geometry"]
    if isinstance(saved, dict):
        return copy.deepcopy(saved)
    return {"group_bytes": None, "group_bytes_note": None,
            "boundary_groups": [], "boundary_note": None,
            "window_bytes": None, "window_note": None,
            "window_rule": "sum(region.stride(0) * element_size)", "read_mono": None}


def section() -> dict:
    """The `disk_restore_verify` block: EVERY KEY IS ALWAYS PRESENT.

    The counters, the enablement, the limits in force, the geometry the comparison used,
    the boundary-group accounting, the oracle's reachability, and the shape of the last
    mismatch - for BOTH roles: the worker's apply-point counters are unprefixed and the
    EngineCore's offer-point counters all begin `scheduler_`, and `roles`/`oracle_rule`
    say which process produces which. The degraded view is built by this same function,
    which is what makes "the key set does not change when there is no GPU" a structural
    property rather than a copy.
    """
    with _lock:
        out = {key: int(_state["counters"][key]) for key in _COUNTER_KEYS}
        out["reasons"] = copy.deepcopy(_state["reasons"])
        out["retained_jobs_now"] = len(_state["pending"])
        out["scheduler_jobs_now"] = len(_state["scheduler_pending"])
        out["last_mismatch"] = (copy.deepcopy(_state["last_mismatch"])
                                if _state["last_mismatch"] else None)
        # THE SCHEDULER'S VERDICTS, as a ring only. There is no
        # `scheduler_last_mismatch` key: the ring's last element is the newest verdict,
        # and publishing the same object twice cost 440 B of a 4096 B budget to repeat a
        # fact a reader can index.
        out["scheduler_mismatch_samples"] = copy.deepcopy(
            _state["scheduler_mismatch_samples"])
        missing_by_group = dict(_state["boundary_missing_groups"])
        boundary_groups = list(_state["boundary_groups"])
        last_boundary = copy.deepcopy(_state["last_boundary"])
    enable = enable_state()
    out["enabled"] = enable["enabled"]
    out["enable_source"] = enable["source"]
    out["env_var"] = config.ENV_DISK_RESTORE_VERIFY
    out["env_raw"] = enable["env_raw"]
    out["env_recognised"] = enable["env_recognised"]
    out["latch"] = enable["latch"]
    out["sample_rate"] = 1.0 / float(max(1, SAMPLE_EVERY_N_RESTORES))
    out["sample_every_n_restores"] = int(SAMPLE_EVERY_N_RESTORES)
    out["max_retained_per_step"] = int(MAX_RETAINED_PER_STEP)
    out["max_retained_jobs"] = int(MAX_RETAINED_JOBS)
    out["max_checks_per_pass"] = int(MAX_CHECKS_PER_PASS)
    out["min_interval_seconds"] = float(MIN_INTERVAL_SECONDS)
    out["job_max_age_seconds"] = float(JOB_MAX_AGE_SECONDS)
    out["geometry"] = geometry_view()
    out["boundary"] = {"groups": boundary_groups,
                       "missing_by_group": missing_by_group,
                       "last": last_boundary}
    out["oracle"] = oracle_state()
    # --- THE SCHEDULER-SIDE ROLE, in the same section and under its own prefix. See the
    # module docstring for why a separate contribution is not available (the seam holds
    # five and its allow-list is five), and for the reachability proof of the two inputs
    # this half reads that the worker half cannot.
    out["scheduler_oracle"] = scheduler_state()
    # WHICH HOOK PRODUCES THESE NUMBERS. The 0.2.40 lesson, published: numbers from
    # an unwired or blind observation point are worse than no numbers, so the
    # section names the hook (`update_state_after_alloc`, the allocation-time
    # capture) whenever it is actually fitted, and says NOT WIRED when it is not -
    # "the oracle ran" and "the oracle found nothing to look at" are one glance
    # apart. The expectation map's live size is published beside it as
    # `scheduler_expectations_now`: the state between "ran" and "nothing to see".
    with _lock:
        fitted = bool(_state["scheduler_alloc_hook_fitted"])
        hook_reason = _state["scheduler_alloc_hook_reason"]
        out["scheduler_expectations_now"] = len(_state["scheduler_expectations"])
    out["scheduler_oracle_hook"] = (
        SCHED_HOOK_FITTED if fitted else
        (SCHED_HOOK_UNWIRED if not hook_reason else
         f"{SCHED_HOOK_UNWIRED}; install said: {hook_reason}"[:160]))
    out["roles"] = {"worker_apply_point": bool(out["batches_seen"]),
                    "scheduler_offer_point": bool(out["scheduler_steps_seen"]),
                    "worker": WORKER_ROLE, "scheduler": SCHEDULER_ROLE,
                    "meaning": ROLE_MEANING}
    out["oracle_rule"] = ORACLE_RULE
    out["neighbour_window"] = int(NEIGHBOUR_WINDOW)
    out["max_scheduler_mismatch_samples"] = int(MAX_SCHED_MISMATCH_SAMPLES)
    out["thread_alive"] = bool(_thread is not None and _thread.is_alive())
    out["section"] = MODULE
    out["writer"] = OWNER_TAG
    out["note"] = ("a mismatch is a QUESTION: only mismatch_stable is a layout "
                   "candidate; the tail is other groups' by construction")
    return _fit(out)


def worker_report() -> dict:
    """The rank-JSON contribution. Builds every key, always; owns no file."""
    return section()


def register_contribution() -> dict:
    """Offer this module's verdict to the rank-file writer. Idempotent.

    Registered by `disk_restore.install()` beside its own `disk_restore` section, i.e.
    only where a restore can actually be applied. The rank FILES are written by workers
    (a worker is the process with a `PP<n>_TP<m>` identity to name a file after), so in
    practice this is the worker's section. THE SCHEDULER-SIDE ROLE still needs no
    contribution of its own: the EngineCore answers `/disk {"op":"restore-verify"}` and
    `/snapshot`, and `snapshot.dev_state_sections()` publishes the SAME `section()`
    there, so both roles are readable through one shape and neither process invents a
    second file, a second contract or a sixth contribution.
    """
    return worker_diag.register_contribution(
        MODULE, worker_report,
        # OWNER_TAG's claim, kept true: this reports a STATE, never an age. What moves
        # with no counter moving is the ENABLEMENT and the ORACLE - either can change
        # between two writes of the file with nothing compared in between, and a reader
        # must be able to see that without diffing the file by hand. The two scheduler
        # terms join them for one reason: a scheduler-side MISMATCH is a finding worth a
        # rewrite the second it happens, while a role that has merely seen its first
        # step changes the signature once per boot and nothing after that.
        owner=f"{OWNER_TAG} state, never a window",
        signal=lambda: (int(bool(enable_state()["enabled"])),
                        int(bool(oracle_state()["provider"])),
                        int(bool(scheduler_state()["mismatch"])),
                        int(bool(scheduler_counters()["scheduler_steps_seen"]))))


def status() -> dict:
    """The verb's view: the enablement, the limits, the geometry, both roles' counters,
    both oracle states and the last verdicts.

    `section()` builds it, so the verb, `worker_report()` and the rank file cannot
    disagree. In the EngineCore - the process that answers this verb - this is the ONLY
    door to the scheduler-side oracle, which is why 0.2.40 added no new op.
    """
    out = {"ok": True, "reason": None, "module": MODULE}
    out.update(section())
    return out


def stop_thread(timeout_seconds: float = 2.0) -> dict:
    """Ask the verifier thread to stop, and say whether it did."""
    with _thread_lock:
        thread = _thread
        stop = _stop
    if stop is not None:
        stop.set()
    if thread is None:
        return {"stopped": True, "reason": "no thread was ever started"}
    alive = thread.is_alive()
    if alive:
        thread.join(timeout=max(0.0, float(timeout_seconds)))
        alive = thread.is_alive()
    return {"stopped": not alive,
            "reason": (None if not alive else
                       f"the thread was still alive after {timeout_seconds}s "
                       "(it is a daemon, so it dies with the process)")}


def reset_state_for_tests() -> None:
    """Back to a fresh boot. The thread is STOPPED, not just un-counted.

    A daemon thread left verifying against a worker from the previous test keeps
    mutating the NEXT test's `_state` through entries it retained, which turns a green
    suite into a property of file order (`disk_gpu_shadow.reset_state_for_tests` says
    the same thing about its writer for the same reason).
    """
    global _thread, _stop
    stopped = stop_thread(timeout_seconds=2.0)
    if not stopped["stopped"]:
        # Logged, not swallowed: a thread that will not stop is the one reset failure
        # that can still write into the NEXT test's state, and a silent reset is how a
        # suite becomes order-dependent without anyone being able to say why.
        logger.warning("kvtier-dev-plugin: [disk_restore_verify] reset found a thread "
                       "that would not stop: %s", stopped["reason"])
    _thread = None
    _stop = None
    # IN PLACE, not a rebind: `conftest.py`'s isolation fixture snapshots this object
    # before the reset and restores it afterwards.
    with _lock:
        _state.clear()
        _state.update(_fresh_state())


def view() -> dict:
    """The internal state, deep-copied: for tests and for a human reading a live
    process. A caller cannot mutate the retention it is looking at."""
    with _lock:
        return copy.deepcopy({"pending": _state["pending"],
                              "geometry": _state["geometry"],
                              "boundary_groups": _state["boundary_groups"],
                              "boundary_missing_groups":
                                  _state["boundary_missing_groups"],
                              "last_boundary": _state["last_boundary"],
                              "last_mismatch": _state["last_mismatch"],
                              "reasons": list(_state["reasons"]),
                              "oracle_note": (_state["oracle"][1]
                                              if _state["oracle"] else None),
                              "scheduler_pending": _state["scheduler_pending"],
                              "scheduler_expectations_now":
                                  len(_state["scheduler_expectations"]),
                              "scheduler_alloc_hook_fitted":
                                  bool(_state["scheduler_alloc_hook_fitted"]),
                              "scheduler_mismatch_samples":
                                  _state["scheduler_mismatch_samples"],
                              "scheduler_unreachable_reason":
                                  _state["scheduler_unreachable_reason"]})
