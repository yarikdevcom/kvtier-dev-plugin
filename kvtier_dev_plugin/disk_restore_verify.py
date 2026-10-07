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
-------------------------------------------------------------------------------
  1. `file_size == bytes_per_block` - the rank's own slot width (`disk_spill.
     measured_slot_bytes` / the fingerprint's `total_bytes_per_block`; 11,468,800 on PP0
     and 12,902,400 on PP1 are DIFFERENT NUMBERS, so a file that is the other rank's
     width is a PP-crossing bug and `file_not_window_wide` is how it shows up). A file
     that is not window-wide is REFUSED, not compared: a width disagreement has an
     explanation and a comparison against it explains nothing.
  2. `W_g <= file_size` (`group_wider_than_file`). A group wider than the file it is
     named in is a geometry contradiction; comparing would read past EOF and call the
     zeros a mismatch.
  3. the leading-`W_g` byte equality, above, with staleness classified by a re-read.
  4. THE BINDING ORACLE. The name in the path is derived from a hash, so name-vs-name is
     blind. The oracle recomputes the group's expected `key16` for the GPU index the
     fork ACTUALLY used, from the engine's own hash derivation -
     `resolve_block_hashes(request.block_hashes, hash_block_size, group_size)` and
     `make_block_hash_with_group_id(digest, group)`, the same two helpers the fork uses
     for the same purpose (`manager.py:897-900`, and the per-group view at
     `manager.py:1591-1601`) and the same two `disk_restore._key_helpers()` resolves. It
     then requires the recomputed `key16` to equal the `key16` in the restore path's
     file name. REACHABILITY IS SAID HONESTLY, NOT FAKED: a WORKER never sees a request.
     The fork's worker metadata carries `load_event_to_reqs` - request IDs - and the
     file that defines it says so out loud: "Job->req_id translation is handled by the
     scheduler-side manager ... the worker never knows about request identities"
     (`vllm/v1/simple_kv_offload/metadata.py`). So `block_hashes` is not reachable from
     this process on this fork, and every entry counts `oracle_unavailable`. That is a
     counted, named, published outcome, not a skip: the counter is how an operator
     learns the binding check is not running, and `oracle_note` names the ONE thing that
     would make it run - carrying the per-entry hash view to the worker, or arming the
     seam below from a process that has requests. The recomputation itself is real code,
     exercised by tests, behind `set_oracle()`: a fork patch that carries the view, or a
     harness, hands this module a provider and the check goes live with no change here.
  5. THE TRUNCATION SHAPE (`boundary_entry_missing`). The fork truncates each group's
     hit list with `n_take_g = cdiv(num_external_tokens, g_block_size)`
     (`manager.py:445-448`), and a `boundary_state` group's hit list is `[null_block] *
     (n - 1) + [one real state block]` (`disk_restore._group_contract`'s cite of
     `single_type_kv_cache_manager.py:1591-1607`). So a truncation that lands before the
     run's last block drops the boundary group's ONLY real entry while every positional
     group keeps its entries - a restore of a prefix whose state block never arrived,
     which is the Gate B shape. Worker-side, a boundary group is one whose spec says
     `mamba_cache_mode == "align"` (a field only `MambaSpec` carries,
     `kv_cache_interface.py:992-998`, and the same value `disk_restore._group_contract`
     reads as the fork's own reason). A batch that restored entries for some groups and
     NOTHING for a known boundary group is counted. THE GRANULARITY IS BATCH-LEVEL AND
     SAY SO: one batch can hold spans from several requests, so a boundary entry that
     DID arrive for one of them masks the one that did not. It under-counts; it cannot
     over-count in the way that matters, because "no entry for a group this boot knows
     is align-mode, in a batch that restored something" is a fact about that batch.

WHERE THE WORK HAPPENS, AND WHAT THE STEP THREAD PAYS
-------------------------------------------------------------------------------
Nothing in this module runs file I/O or a device copy on the engine's threads. The step
thread pays ONE call, at the apply point in `disk_restore._worker_before_load`, after
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

WHAT IT NEVER DOES
-------------------------------------------------------------------------------
  * writes any file, anywhere, ever (the only file operation is a read of a path the
    restore path itself already chose to read);
  * changes a restore outcome - `observe_apply` cannot raise, cannot prune and is called
    after every decision the apply point makes;
  * blocks the step thread (no I/O, no CUDA, no torch symbol, no unbounded loop);
  * imports `torch`/`vllm`/`cuda`/`numpy` (the same package invariant
    `tests/test_snapshot.py::test_no_heavy_imports_in_the_read_side` enforces; the
    device read needs no symbol, because it is `disk_shadow`'s method call);
  * keeps a reference to a worker it cannot prove is alive (a weakref; a worker that is
    gone is `worker_gone`, and the pass stops rather than copying from freed memory).
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

# The first-difference scan, and why it is chunked: `memcmp` per 4 KiB block, so the
# only Python-level byte loop runs inside the ONE chunk that differs. A full-window
# Python scan of 11.4 MB to produce an offset is not worth the offset.
MAX_DIFF_SCAN_BYTES = 262_144

# The published lists, and the section's own byte budget. See `_fit`.
MAX_BOUNDARY_GROUPS_PUBLISHED = 8
TRIM_BOUNDARY_GROUPS = 4
TRIM_REASON_CHARS = 42
MAX_SECTION_BYTES = 3072

# The fork's own reason for calling a group's blocks positionally unstable, read from
# the group's spec on the worker (`kv_cache_interface.py:992-998`: only `MambaSpec`
# carries this field, so an absent attribute means "not an align-mode state group", not
# "unknown").
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
    "section_trimmed",
    "errors",
)


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
                       f"{config.ENV_DISK_RESTORE_VERIFY}={read['raw']!r}: one "
                       f"restored entry in {SAMPLE_EVERY_N_RESTORES} has its leading "
                       f"group width compared between the file and the GPU window, at "
                       f"most {MAX_CHECKS_PER_PASS} per pass on this module's thread",
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
    so a POST that arrives at the EngineCore latches the EngineCore, where no restore is
    ever applied. The process that applies restores is a WORKER, and a worker is armed
    with `VLLM_E2_TIER_DISK_RESTORE_VERIFY=1` in the engine's environment. `scope` says
    that instead of letting an operator believe he armed the worker.
    """
    with _lock:
        _state["latch"] = bool(on)
        _state["latch_set"] = True
        answer = enable_state()
    answer["scope"] = ("this process only; the worker that restores is armed with "
                       f"{config.ENV_DISK_RESTORE_VERIFY}=1")
    if on:
        _note_once("latch-on",
                   "[disk_restore_verify] enabled in this process by the POST op "
                   "(scope: the answering process, not where a restore applies)",
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


def _boundary_groups(worker) -> tuple:
    """The group ids whose blocks are align-mode state blocks, or a named reason.

    The predicate is the one the WORKER can evaluate: `mamba_cache_mode` is a field only
    `MambaSpec` carries (`kv_cache_interface.py:992-998`), so an attention spec answers
    None and an align-mode state group answers `"align"` - the same value
    `disk_restore._group_contract` reads (from the manager, falling back to the spec) as
    the fork's own reason for calling a group's blocks positionally unstable. The
    manager itself is EngineCore-only, so the spec is all this process has and it is
    enough.
    """
    groups = getattr(getattr(worker, "kv_cache_config", None), "kv_cache_groups", None)
    if not groups:
        return [], "the worker declares no kv_cache_groups"
    known = []
    for index, group in enumerate(list(groups)[:worker_diag.MAX_GROUPS]):
        mode = getattr(getattr(group, "kv_cache_spec", None), "mamba_cache_mode", None)
        if mode == MAMBA_CACHE_MODE_ALIGN:
            known.append(int(index))
    if not known:
        return [], (f"no group's spec reports {MAMBA_CACHE_MODE_ALIGN!r} "
                    "mamba_cache_mode")
    return known, None


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
                                       "a worker receives no request identities "
                                       "(vllm/v1/simple_kv_offload/metadata.py), so "
                                       "request.block_hashes is unreachable from the "
                                       "process that applies a restore; every entry "
                                       "counts oracle_unavailable until set_oracle() "
                                       "installs a provider"),
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


# --- THE PASS ------------------------------------------------------------------
def run_pass(worker=None) -> dict:
    """At most `MAX_CHECKS_PER_PASS` retained entries. ONE bounded pass; never raises.

    The thread's body, and the function a test drives directly (the `disk_gpu_shadow`
    idiom: monkeypatch `_ensure_thread` to keep the thread off and call this).
    """
    try:
        if not effective_enabled():
            return {"ran": False, "reason": "disabled", "checked": 0}
        now = time.monotonic()
        with _lock:
            last = _state["last_pass_mono"]
            if last is not None and (now - float(last)) < MIN_INTERVAL_SECONDS:
                _count("throttled_interval")
                return {"ran": False, "reason": "throttled_interval", "checked": 0}
            _state["last_pass_mono"] = now
            _state["pass_used"] = 0
            _state["recheck_used"] = 0
            batch = [record for record in list(_state["pending"].values())
                     [:MAX_CHECKS_PER_PASS]]
            if not batch:
                return {"ran": True, "reason": "nothing retained", "checked": 0}
            _count("passes")
        target = _resolve_worker(worker)
        if target is None:
            # Every retained entry goes with it: they were retained against a worker
            # that is gone, and re-verifying them against a future one would compare one
            # boot's file against another boot's GPU.
            with _lock:
                _count("worker_gone")
                dropped = len(_state["pending"])
                _state["pending"] = {}
            _reason("worker_gone",
                    f"the worker this module retained is gone, so {dropped} retained "
                    "entries were dropped unread")
            return {"ran": True, "reason": "worker_gone", "checked": 0}
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
                "waiting": len(still_waiting), "results": results}
    except Exception as exc:  # noqa: BLE001 - a pass never escapes its thread
        with _lock:
            _count("errors")
        _reason("run_pass", f"the pass raised {type(exc).__name__}")
        return {"ran": False, "reason": f"error_{type(exc).__name__}", "checked": 0}


# --- THE VIEW ------------------------------------------------------------------
def counters() -> dict:
    with _lock:
        return dict(_state["counters"])


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
    the order a reader acts on: the reason TEXT first (the leading tag survives - the
    tag is what names the category), then the boundary-group illustration lists (the
    COUNTS above them never move), then the reasons themselves. The counters and
    `last_mismatch` are NEVER trim targets: they are why the section exists, they are
    fixed-size, and a trimmed section that gained or lost a key is a section a reader
    cannot diff. Every top-level key survives every step, which is what keeps "the key
    set never changes" true under pressure too.
    """
    trimmed = out["trimmed"] = []
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return out
    out["reasons"] = [str(line)[:TRIM_REASON_CHARS] for line in out["reasons"]]
    trimmed.append(f"reason_chars:{TRIM_REASON_CHARS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    boundary = out["boundary"]
    if isinstance(boundary, dict):
        boundary["missing_by_group"] = dict(list(boundary["missing_by_group"].items())
                                            [:TRIM_BOUNDARY_GROUPS])
        boundary["groups"] = list(boundary["groups"])[:TRIM_BOUNDARY_GROUPS]
        trimmed.append(f"boundary:{TRIM_BOUNDARY_GROUPS}")
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
    mismatch. The degraded view is built by this same function, which is what makes "the
    key set does not change when there is no GPU" a structural property rather than a
    copy.
    """
    with _lock:
        out = {key: int(_state["counters"][key]) for key in _COUNTER_KEYS}
        out["reasons"] = copy.deepcopy(_state["reasons"])
        out["retained_jobs_now"] = len(_state["pending"])
        out["last_mismatch"] = (copy.deepcopy(_state["last_mismatch"])
                                if _state["last_mismatch"] else None)
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
    out["thread_alive"] = bool(_thread is not None and _thread.is_alive())
    out["section"] = MODULE
    out["writer"] = OWNER_TAG
    out["note"] = ("a mismatch is a QUESTION, not a verdict: only mismatch_stable is a "
                   "layout candidate, and only the leading W_g bytes of the window are "
                   "compared, because the tail belongs to the other groups by "
                   "construction (tail_bytes_excluded says how much)")
    return _fit(out)


def worker_report() -> dict:
    """The rank-JSON contribution. Builds every key, always; owns no file."""
    return section()


def register_contribution() -> dict:
    """Offer this module's verdict to the rank-file writer. Idempotent.

    Registered by `disk_restore.install()` beside its own `disk_restore` section, i.e.
    only where a restore can actually be applied. An EngineCore or an API-server process
    registers neither, so its rank file carries no `disk_restore_verify` key at all
    rather than an empty object claiming a facility that process does not have.
    """
    return worker_diag.register_contribution(
        MODULE, worker_report,
        # OWNER_TAG's claim, kept true: this reports a STATE, never an age. What moves
        # with no counter moving is the ENABLEMENT and the ORACLE - either can change
        # between two writes of the file with nothing compared in between, and a reader
        # must be able to see that without diffing the file by hand.
        owner=f"{OWNER_TAG} state, never a window",
        signal=lambda: (int(bool(enable_state()["enabled"])),
                        int(bool(oracle_state()["provider"]))))


def status() -> dict:
    """The verb's view: the enablement, the limits, the geometry, the oracle and the
    counters. `section()` builds it, so the verb and the rank file cannot disagree."""
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
                                              if _state["oracle"] else None)})
