"""Shadow verification of the spill path against the GPU's own KV bytes.

WHAT THIS IS. The spill path (`disk_spill.request_spill`) captures a slot by reading
the ARENA - the worker's `cpu_kv_caches`, the copy vLLM itself makes every step - and
that is the only copy of a block's bytes the tier ever writes. If the arena read
captured the wrong bytes, or captured them in an order the restore side does not
reassemble, the tier would serve silently wrong KV and nothing in the existing
telemetry would notice: a byte-level mismatch is invisible to a hit-rate counter.
THIS MODULE IS THE PARANOID WITNESS for exactly that. At store BIND time it retains
the GPU block ids the engine named alongside the CPU slots (ints only - never a
tensor, never the metadata object). At store COMPLETION, for a sampled subset, it
copies the GPU's bytes for that same block into a host buffer and compares them with
the blob the arena path captured for that same block. When they disagree, the
operator sees it in `diagnostics-rank.json` under `disk_shadow`.

WHAT IT IS NOT. Not a correctness gate: it never changes a spill or restore decision,
never refuses a write, never opens a file handle for writing, never touches the GPU
except to READ one block of it. It is a MEASUREMENT - default OFF, fail-open,
read-only.

COMPARE AGAINST THE BLOB, NOT THE FILE. The spill writer is a daemon thread, so when
the comparison runs the file may not exist yet. The blob is what the spill path
DECIDED TO WRITE, which is the exact thing under test. The interface between the two
modules is one dict key: `disk_store` builds a plan, hands it to `request_spill`, and
`request_spill` fills in `plan["blob"]` by reference when it captures bytes - the
bytes are never copied a second time, and `disk_spill` never calls into this module.

THE ORDER RULE, RESTATED BECAUSE IT IS THE WHOLE POINT. A slot's bytes are the `sum`
over the region tensors and the sum's order is `sorted(name)` - NEVER dict-insertion
order (0.2.8 shipped that bug). This module does not re-implement the ordering: it
copies each region's block to the host and hands the resulting one-block map to
`disk_spill.read_slot_bytes()`, THE SAME READER THE SPILL PATH USES, at index 0. The
shadow therefore cannot disagree with the spill path about ordering by construction,
and a mismatch means the GPU and the arena genuinely hold different bytes - not that
two readers summed in two different orders.

STALENESS, AND WHY ONE "corruption" COUNTER WOULD BE A LIE. A GPU KV block is the
engine's to reuse the moment the store finished, and the shadow reads it AFTER
completion, so it may legitimately have changed. On a mismatch the shadow therefore
RE-READS the GPU block once (bounded) and asks whether the GPU's own bytes moved
between the two reads:

    mismatch_gpu_changed_on_recheck  the GPU moved under us: staleness, expected
    mismatch_stable                  the GPU held still and still differs: the
                                     candidate for a real layout/ordering fault
    mismatch_unrechecked             the recheck could not run (budget or copy fault)
    mismatch_rebound_later           a block that mismatched was named again by a
                                     later bind: direct evidence it was handed out

The re-read is the PRIMARY discriminator because it needs no timing assumption, no
lock and no engine cooperation, and it separates "the GPU changed" from "our layout
is wrong". The rebound-later witness is secondary: it proves reuse happened, not that
it happened BEFORE the read. Its failure modes are named where they are counted - a
block reused and rewritten to BYTE-IDENTICAL content between the two reads would look
stable (which can only weaken a mismatch's classification, never invent one), and a
block reused AFTER the re-read is reported as `mismatch_stable`. A `width_mismatch` is
NOT staleness and is never counted as such: the GPU's region widths disagreeing with
the arena's is a layout question, not a timing one.

GPU READS, AND THE SEAM. THIS MODULE IMPORTS NOTHING HEAVY - not at import, not
lazily. `tests/test_snapshot.py::test_no_heavy_imports_in_the_read_side` walks the AST
of every module file and refuses a `torch`/`vllm`/`cuda`/`numpy` import ANYWHERE in it,
"so it cannot be satisfied by an import that only runs conditionally", and a
device->host copy needs no `torch` symbol: `narrow`, `contiguous` and `cpu` are methods
on the tensor the engine already built. The single expression that touches a GPU tensor
lives in `_copy_slot_to_host()`, which is the seam a test replaces, and every copy is
bounded by `MAX_BLOCKS_PER_STEP` and `MIN_INTERVAL_SECONDS` because this work happens
on the engine's step thread, inside the same tick as `get_finished`.

READING A DEVICE ADDRESS AS A HOST ONE IS A SEGFAULT, NOT AN EXCEPTION, so the one
check that cannot be delegated to an `except` is that the copy LANDED: `_host_landed`
refuses a copied region that does not report a `cpu` device, before
`disk_spill.read_slot_bytes` forms any address. It is the same reason
`disk_spill._region_span` checks a region's own byte count instead of trusting its
shape.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time

from . import config, disk_spill

MODULE = "disk_shadow"
OWNER_TAG = "kvtier-dev-plugin/disk-shadow-1"
logger = logging.getLogger(MODULE)

# --- THE LIMITS, EACH WITH WHAT IT PROTECTS ----------------------------------
#
# How many store binds' GPU block ids are remembered. One entry per bind, holding
# `{cpu slot: gpu block id}` - ints, not tensors. `disk_store` spills at completion,
# one or two ticks after the bind, so 16 events is several engine steps of slack.
# An eviction is not a wrong answer: it loses one comparison (`retained_dropped`, and
# then `no_gpu_block`), which is the honest failure shape for a bounded buffer.
MAX_RETAINED_EVENTS = 16
#
# THE BOUND THE STEP THREAD NEEDS. This copy runs on the engine's step thread, in the
# same tick `disk_store.MAX_SLOTS_PER_TICK` bounds. Two blocks is 2 slot widths of
# device->host traffic (2 x 11,468,800 B on the live PP0 moves in ~1 ms at 22 GB/s),
# a rounding error on a step. "Compare everything" is not the goal; the sampler is
# what makes the cost a constant instead of a function of the store rate.
MAX_BLOCKS_PER_STEP = 2
#
# The minimum monotonic gap between two read episodes - the `disk_read` touch-throttle
# idiom, with `throttled_interval` as its published counter. A store burst completes
# many events on one tick; without this, a 1-in-32 decision could still put a copy on
# EVERY step of a busy boot.
MIN_INTERVAL_SECONDS = 0.5
#
# One re-read per read episode. The re-read exists to CLASSIFY a mismatch, not to
# measure how fast the engine rewrites a block; unbounded re-reads on a block that is
# being rewritten would turn one mismatch into a copy loop on the step thread.
MAX_RECHECKS_PER_STEP = 1
#
# Deterministic 1-in-N sampling on a monotonic counter. Deterministic (no RNG) so a
# test can force a hit and a comparison is reproducible across boots. 32 leaves a
# busy boot comparing hundreds of blocks while the shadow stays a constant cost.
SAMPLE_EVERY_N_STORES = 32
#
# How many mismatched GPU block ids are kept as reuse witnesses. Ints in a FIFO; 64 is
# a rounding error in a worker's memory and bounds the only cross-step state here.
MAX_TRACKED_MISMATCH_BLOCKS = 64
#
# The failure reasons an operator can read. Capped, oldest kept, so a persistent
# failure stays visible instead of being pushed out by a flood of identical ones.
MAX_REASON_SAMPLES = 8
#
# How long ONE reason may be. This is not cosmetics: the section is a GUEST in
# `rank-<rank>.json`, and `worker_diag.MAX_CONTRIBUTION_BYTES` REFUSES a section over
# the cap rather than truncating it - so 8 unbounded exception strings from a driver
# would delete the whole `disk_shadow` section from the file an operator is reading.
MAX_REASON_CHARS = 120
#
# How far the "where do they first differ?" scan looks. A chunked `bytes` compare is
# memcmp (C speed), so 256 KB is microseconds; scanning all 11.4 MB byte-by-byte in
# Python would put ~1 s on the step thread for the sake of an offset nobody needs.
MAX_DIFF_SCAN_BYTES = 262_144
#
# Published region names. A real boot has one region per KV-bearing layer (dozens);
# the view publishes the count, the ORDER RULE, the total width and the first few
# names, which is everything a reader needs to answer "which order did you sum in?".
MAX_REGIONS_PUBLISHED = 8
#
# THE SECTION'S OWN BYTE BUDGET - TIGHTER THAN THE SEAM'S, ON PURPOSE.
# `worker_diag.MAX_CONTRIBUTION_BYTES` (4096) is the cap that REFUSES a section, and a
# refusal is invisible in the counters: the key simply is not in the file. Measured with
# every reason at its cap, eight long region names with their widths, a full
# `last_mismatch` and six-digit counters, the section came to 3974 bytes - 3% of
# headroom, so ONE more counter name or a longer reason would have deleted the witness
# on the boot that most needed it. So `section()` never hands the writer anything over
# this budget; it sheds its own detail, in a declared order, and says what it shed in
# `trimmed`. 3072 = three quarters of the seam's cap, which is room for roughly fifteen
# more counter names before the trim path is ever taken.
MAX_SECTION_BYTES = 3072
# What the trim steps replace, and in what order. The COUNTERS are never shed: they are
# the small, always-present part, and the numbers are the reason the section exists.
TRIM_REASON_CHARS = 48
TRIM_REGIONS_PUBLISHED = 4

# THE STABLE KEY SET. `section()` builds every one of these on every call, in every
# state - no `gpu_kv_caches`, no usable GPU shape, a device fault mid-copy, a width
# disagreement - because a reader diffing two rank files must be able to tell "moved"
# from "gone", and the degraded view is built by the same code as the healthy one.
_COUNTER_KEYS = (
    "binds",
    "bind_missing_gpu_ids",
    "gpu_ids_retained",
    "retained_dropped",
    "stores_seen",
    "sampled_out",
    "throttled_step",
    "throttled_interval",
    "no_gpu_block",
    "no_gpu_caches",
    "no_blob",
    "gpu_read_ok",
    "gpu_read_failed",
    "width_mismatch",
    "section_trimmed",
    "checked",
    "equal",
    "mismatch",
    "rechecks",
    "mismatch_gpu_changed_on_recheck",
    "mismatch_stable",
    "mismatch_unrechecked",
    "mismatch_rebound_later",
    "errors",
)

_state: dict = {
    "latch": False,           # the POST op's decision; the environment is the floor
    "latch_set": False,       # has the POST said anything at all in this process
    "env_resolved": False,    # the environment is read ONCE, see `_resolve_env`
    "env_enabled": False,
    "env_raw": None,
    "env_recognised": True,
    "retained": {},           # {event: {"slots": {cpu: gpu}, "ts": float}}
    "retained_order": [],
    "step_seq": 0,            # a bind is the step marker: the per-step budget's clock
    "step_budget_step": -1,
    "step_used": 0,
    "recheck_used": 0,
    "sample_cursor": 0,
    "last_read_mono": None,
    "mismatch_blocks": {},    # {gpu block id: True}, insertion-ordered
    "reasons": [],
    "reason_seen": set(),
    "last_trim": None,        # the trim SHAPE last counted, so polling is not an event
    "notes": {},
    "regions": None,          # the last region view built from a real worker
    # The SHAPE of the last mismatch: where the two blobs first differ, how far the
    # scan looked, which block it was. Never the block hash - `disk_store` refuses to
    # publish keys in a snapshot, and this is a snapshot.
    "last_mismatch": None,
    "counters": {key: 0 for key in _COUNTER_KEYS},
}
_lock = threading.Lock()


def _note_once(key: str, message: str, level: str = "warning") -> bool:
    """One log line per episode, never one per tick - the `disk_boot` idiom, for the
    same reason: a worker's step thread must not repeat one warning forever."""
    if _state["notes"].get(key):
        return False
    _state["notes"][key] = True
    logger.log(getattr(logging, level.upper()), "kvtier-dev-plugin: %s", message)
    return True


def _count(name: str, amount: int = 1) -> None:
    _state["counters"][name] = int(_state["counters"][name]) + int(amount)


def _reason(tag: str, detail: str) -> None:
    """Record WHY something failed, once per distinct tag, and log it once. A
    persistent condition (a `gpu_kv_caches` shape that cannot be read, a device fault)
    must be legible in the rank file, not an anonymous growing integer.

    CLIPPED, because the section is a guest in someone else's file: see
    `MAX_REASON_CHARS`.
    """
    if tag in _state["reason_seen"]:
        return
    _state["reason_seen"].add(tag)
    text = str(detail)
    if len(text) > MAX_REASON_CHARS:
        text = text[:MAX_REASON_CHARS - 3] + "..."
    _state["reasons"].append(f"{tag}: {text}")
    del _state["reasons"][MAX_REASON_SAMPLES:]
    _note_once(f"reason:{tag}", f"[disk_shadow] {tag}: {text}")


# --- ENABLEMENT ---------------------------------------------------------------
def _resolve_env() -> dict:
    """Read `VLLM_E2_TIER_DISK_SHADOW` ONCE per process, and say so in the view.

    The reading is `config.disk_shadow_enable()` - the DEFAULT-OFF grammar, same shape
    as the tier's own boot flag and the opposite default. What is added here is the
    CADENCE: a live boot's workers inherit an environment that does not change, so
    re-reading it per comparison would put an `os.environ` lookup on the engine's step
    thread for no information, and the POST latch already exists as the live switch.
    `source` names that honestly instead of hiding it.
    """
    if not _state["env_resolved"]:
        read = config.disk_shadow_enable()
        _state["env_raw"] = read["raw"]
        _state["env_enabled"] = bool(read["enabled"])
        _state["env_recognised"] = bool(read["recognised"])
        _state["env_resolved"] = True
        if read["enabled"]:
            _note_once("env-on",
                       f"[disk_shadow] ENABLED by {config.ENV_DISK_SHADOW}="
                       f"{read['raw']!r}: one GPU block in "
                       f"{SAMPLE_EVERY_N_STORES} completed stores is compared against "
                       f"the spill path's blob, at most {MAX_BLOCKS_PER_STEP} per "
                       f"bind window", level="info")
        elif read["raw"] is not None and not read["recognised"]:
            _reason("env_value_unrecognised",
                    f"{config.ENV_DISK_SHADOW}={read['raw']!r} is neither an on word "
                    f"nor an off word; the shadow is OFF, which is the direction a "
                    f"typo never overrides")
    return {"enabled": bool(_state["env_enabled"]),
            # Both branches say READ ONCE, because both are the same cadence. What the
            # operator needs from this field is "exporting the variable now will not
            # take effect", whichever way the first read went.
            "source": ("environment (read once at first use)"
                       if _state["env_raw"] is not None else
                       "default (off); the environment is read once at first use"),
            "raw": _state["env_raw"],
            "recognised": bool(_state["env_recognised"]),
            "env_var": config.ENV_DISK_SHADOW}


def effective_enabled() -> bool:
    """Shadow work happens only if the environment says yes or the POST latch does."""
    if _state["latch"]:
        return True
    return bool(_resolve_env()["enabled"])


def enable_state() -> dict:
    """One stable-key view of the enablement, including WHO decided it."""
    env = _resolve_env()
    return {"enabled": bool(env["enabled"] or _state["latch"]),
            "env_enabled": bool(env["enabled"]),
            "latch": bool(_state["latch"]),
            "latch_set": bool(_state["latch_set"]),
            "source": ("post op" if _state["latch"] else env["source"]),
            "env_var": config.ENV_DISK_SHADOW, "env_raw": env["raw"],
            "env_recognised": bool(env["recognised"])}


def set_enabled(on: bool) -> dict:
    """The POST op's decision: IN-PROCESS ONLY, and said out loud in the answer.

    No control file is written, deliberately: this module's contract is that it opens
    no file handle for writing anywhere, and `disk_store._apply_switch` already
    establishes the same scope for the same reason ("the persistent path is the env
    var"). `scope` is published because the process that spills is a DIFFERENT process
    from the EngineCore that answers the POST: the operator turns this on for a worker
    with the environment variable, and uses the POST to arm the process answering.
    """
    with _lock:
        _state["latch"] = bool(on)
        _state["latch_set"] = True
        answer = enable_state()
    answer["scope"] = ("this process only; a worker process needs "
                       f"{config.ENV_DISK_SHADOW}=1 in the engine's environment")
    if on:
        _note_once("latch-on",
                   "[disk_shadow] enabled in this process by the POST op "
                   "(scope: the answering process)")
    return answer


# --- THE COPY SEAM ------------------------------------------------------------
#
# WHY THERE IS NO `import torch` IN THIS FILE, LAZILY OR OTHERWISE. The brief asked
# for a lazy torch import inside the copy. The package has an existing, deliberate
# invariant that forbids it: `tests/test_snapshot.py::
# test_no_heavy_imports_in_the_read_side` walks the AST of EVERY module file and fails
# on a `torch`/`vllm`/`cuda`/`numpy` import ANYWHERE in it - its docstring says the AST
# walk exists precisely so "it cannot be satisfied by an import that only runs
# conditionally". Evading that with `importlib.import_module("torch")` would satisfy
# the letter of the check while defeating its stated purpose, so it is not done here.
# It also turns out to be unnecessary: a device->host copy needs no `torch` SYMBOL.
# `narrow`, `contiguous` and `cpu` are methods on the tensor the engine already built,
# so the copy below pulls in nothing at module import OR at call time.
#
# THE COST, STATED HONESTLY: there is no `import torch` failure left to name as its own
# reason category, because nothing imports torch. The state that would have produced it
# - a process with no GPU library - cannot have `gpu_kv_caches` either, so it is
# reported as `no_gpu_caches`, and a device fault during the copy is counted as
# `gpu_read_failed` with the exception's own name in the reason.


def _copy_slot_to_host(tensor, index):
    """ONE BLOCK OF ONE REGION, DEVICE -> HOST. The only expression in this module
    that touches a GPU tensor, and the seam a test replaces.

    `narrow` then `contiguous` then `cpu` is a blocking copy on the CURRENT CUDA
    stream: nothing is queued ahead of the engine's own work and only this copy is
    synchronised. Narrowing BEFORE the copy is the point - copying the region to the
    host and then indexing it would move the whole KV cache to read one block.

    The result is a one-block CPU tensor, which is region-shaped (`data_ptr`, `stride`,
    `element_size`, `nbytes`) - exactly what `disk_spill.read_slot_bytes` reads, so the
    host half of the comparison runs through the spill path's own reader.

    It raises when there is no usable device, when the copy is refused, or when the
    object is not a tensor at all; `_read_gpu_slot` turns every raise into a counted
    failure and never into an exception on the engine's thread.
    """
    return tensor.narrow(0, int(index), 1).contiguous().cpu()


def _host_landed(host) -> bool:
    """Did the copy actually arrive in HOST memory? THE ONE GUARD THAT CANNOT BE
    DELEGATED TO AN `except`.

    Everything downstream is `ctypes.string_at(data_ptr())`, and a device pointer read
    as a host address is a SIGSEGV, not a Python exception - no `except Exception` can
    save the worker process. That is the same reason `disk_spill._region_span` checks a
    region's own byte count instead of trusting its shape, and it is checked here on
    the COPY'S OWN `device` field: anything that does not say `cpu` is refused before a
    single byte is read through an address. An object with no `device` at all (a
    ctypes-backed test region) is host memory by construction and passes.
    """
    device = getattr(host, "device", None)
    if device is None:
        return True
    return "cpu" in str(device).lower()


def _gpu_region_map(worker) -> dict | None:
    """The worker's GPU region tensors, or None. An attribute read and nothing more:
    the tensors are never copied, sampled out of, or retained by this module."""
    caches = getattr(worker, "gpu_kv_caches", None)
    if not isinstance(caches, dict) or not caches:
        return None
    return caches


def _read_gpu_slot(worker, gpu_block: int, width: int) -> dict:
    """One GPU block's slot bytes in the SPILL PATH's order, or a named refusal.

    The copy is per region into a map under the same names; the concatenation, the
    width check and the ctypes read are `disk_spill.read_slot_bytes` at index 0. Never
    raises: every fault comes back as `{ok: False, kind, reason}`.
    """
    caches = _gpu_region_map(worker)
    if caches is None:
        return {"ok": False, "kind": "no_gpu_caches",
                "reason": "the worker exposes no gpu_kv_caches"}
    names = sorted(str(name) for name in caches)
    if len(names) > disk_spill.MAX_REGIONS:
        return {"ok": False, "kind": "gpu_read_failed",
                "reason": f"the GPU reports {len(names)} regions, above the "
                          f"{disk_spill.MAX_REGIONS} the reader accepts"}
    host: dict = {}
    for name in names:
        region = caches[name]
        shape = getattr(region, "shape", None)
        try:
            blocks = int(tuple(shape)[0]) if shape is not None else None
        except (TypeError, ValueError, IndexError):
            blocks = None
        if blocks is None:
            return {"ok": False, "kind": "gpu_read_failed",
                    "reason": f"GPU region {name} has no usable [blocks][width] "
                              "shape to index"}
        if not 0 <= int(gpu_block) < blocks:
            return {"ok": False, "kind": "no_gpu_block",
                    "reason": f"GPU block {gpu_block} is outside {name}'s "
                              f"{blocks} blocks"}
        try:
            host[name] = _copy_slot_to_host(region, gpu_block)
        except Exception as exc:  # noqa: BLE001 - a device fault is a counted answer
            # No CUDA driver, an illegal block, a copy the allocator refuses, a host
            # allocation that will not fit: each names itself in the reason, and none
            # of them is allowed to reach the step thread as an exception.
            return {"ok": False, "kind": "gpu_read_failed",
                    "reason": f"the device->host copy of {name} failed "
                              f"({type(exc).__name__}: {exc})"}
        if not _host_landed(host[name]):
            # THE GUARD AGAINST A SEGMENTATION FAULT, not against an exception: see
            # `_host_landed`. A copy that reports a non-host device is refused here
            # rather than handed to `ctypes.string_at` below.
            return {"ok": False, "kind": "gpu_read_failed",
                    "reason": f"the copy of {name} did not land on the host "
                              f"(device={getattr(host[name], 'device', None)!r}): "
                              "refusing to read a device address as a host one"}
    # THE ORDER RULE IS NOT RE-IMPLEMENTED HERE: same reader, same sorted-name order,
    # index 0 of a one-block map. `reason_kind == "region_width_mismatch"` below means
    # the GPU's widths disagree with the arena's - `width_mismatch`, NOT staleness.
    return disk_spill.read_slot_bytes(host, 0, int(width))


# --- CAPTURE AT BIND ----------------------------------------------------------
def observe_bind(handler, metadata) -> None:
    """Retain the GPU block ids a store bind named, aligned with its CPU slots.

    INTS ONLY. The bind wrapper in `disk_store` deliberately retains neither tensors
    nor the metadata object and this keeps that rule: what is kept per event is
    `{cpu slot: gpu block id}` plus a timestamp. Never raises.

    `store_gpu_blocks` is the 0.2.35 field, index-aligned with `store_cpu_blocks`
    (`metadata.py`). On an engine that does not carry it the absence is the normal
    state and is counted as `bind_missing_gpu_ids`, never treated as a fault.
    """
    try:
        if not effective_enabled():
            return
        cpu_blocks = getattr(metadata, "store_cpu_blocks", None)
        gpu_blocks = getattr(metadata, "store_gpu_blocks", None)
        if not isinstance(cpu_blocks, (list, tuple)) \
                or not isinstance(gpu_blocks, (list, tuple)):
            with _lock:
                _count("bind_missing_gpu_ids")
            _reason("bind_without_gpu_ids",
                    "this bind carries no store_gpu_blocks, so there is no GPU id to "
                    "read (an engine older than the 0.2.35 field, or a connector "
                    "that does not name its source blocks)")
            return
        slots: dict = {}
        missing = 0
        if len(cpu_blocks) != len(gpu_blocks):
            # The engine patch guarantees these two are index-aligned; if they ever
            # are not, there is no alignment to retain and PAIRING THEM ANYWAY (which
            # a plain `zip` would do, silently, to the shorter list) is the
            # wrong-block-by-luck this witness exists to catch. Refuse the bind.
            with _lock:
                _count("bind_missing_gpu_ids")
            _reason("gpu_id_count_mismatch",
                    f"this bind named {len(cpu_blocks)} CPU slots but "
                    f"{len(gpu_blocks)} GPU blocks; the two are documented as "
                    "index-aligned (metadata.py), so no pairing here is trusted")
            return
        for cpu, gpu in zip(cpu_blocks, gpu_blocks, strict=True):
            if cpu is None or gpu is None:
                missing += 1
                continue
            try:
                slots[int(cpu)] = int(gpu)
            except (TypeError, ValueError):
                missing += 1
        event = getattr(metadata, "store_event", None)
        if missing:
            _reason("gpu_id_incomplete",
                    f"{missing} of {len(cpu_blocks)} stored slots in this bind named "
                    f"no GPU block id, so those slots cannot be shadow-checked "
                    f"(an engine that fills `store_gpu_blocks` only partly)")
        with _lock:
            if missing:
                _count("bind_missing_gpu_ids")
            _count("binds")
            # A bind IS the step marker, so the per-step budget has a clock that does
            # not depend on a timer or a thread.
            _state["step_seq"] += 1
            _capture_regions(handler)
            if event is None:
                return
            _state["retained"].pop(event, None)
            _state["retained"][event] = {"slots": slots, "ts": time.monotonic()}
            _state["retained_order"].append(event)
            _count("gpu_ids_retained", len(slots))
            # THE REUSE WITNESS: a block that mismatched and is named again by a later
            # bind has demonstrably been handed back out. Evidence of reuse, not of
            # when the rewrite happened - see the module docstring.
            for gpu in set(slots.values()):
                if _state["mismatch_blocks"].pop(gpu, None) is not None:
                    _count("mismatch_rebound_later")
            while len(_state["retained_order"]) > MAX_RETAINED_EVENTS:
                _state["retained"].pop(_state["retained_order"].pop(0), None)
                _count("retained_dropped")
    except Exception as exc:  # noqa: BLE001 - never into the engine's bind path
        with _lock:
            _count("errors")
        _reason("observe_bind", f"the bind observation raised {type(exc).__name__}")


def _capture_regions(worker) -> None:
    """Remember WHICH regions and widths a comparison would use, from a real worker.

    Cached in module state rather than rebuilt at report time because the rank-file
    writer has no worker in hand (`worker_diag` calls a producer with no arguments),
    and because publishing the region list the LAST OBSERVED BOOT used is the answer to
    the question a mismatch asks. Never probes when no worker is handed in.
    """
    if worker is None:
        return
    gpu = _gpu_region_map(worker)
    arena = getattr(worker, "cpu_kv_caches", None)
    names = sorted(str(name) for name in (gpu if gpu else arena or {}))
    if not names:
        return
    measured = disk_spill.measured_slot_bytes(arena) \
        if isinstance(arena, dict) and arena else {}
    view = {"order_rule": "sorted(name)", "count": len(names),
            "names": names[:MAX_REGIONS_PUBLISHED],
            "truncated": len(names) > MAX_REGIONS_PUBLISHED,
            "multi_region": len(names) > 1,
            "total_bytes_per_block": measured.get("bytes_per_file"),
            "widths": None,
            "source": ("gpu_kv_caches" if gpu else
                       "cpu_kv_caches (the arena; the GPU side was unreadable)")}
    widths = measured.get("widths")
    if isinstance(widths, dict):
        view["widths"] = {str(key): int(value)
                          for key, value in sorted(widths.items())
                          [:MAX_REGIONS_PUBLISHED]}
    _state["regions"] = view
    if len(names) > 1:
        # A multi-region boot is exactly the configuration that produced the 0.2.8
        # order-rule bug, so it is news at BOOT, not news at the first mismatch.
        _note_once("multi-region",
                   f"[disk_shadow] MULTI-REGION boot: {len(names)} regions "
                   f"({names[0]} ... {names[-1]}), slot width "
                   f"{view['total_bytes_per_block']} B, summed in sorted(name) "
                   f"order by the spill path and by this checker", level="warning")


# --- DECIDE AT COMPLETION -----------------------------------------------------
def snapshot(handler, event, slot: int, key=None) -> dict | None:
    """Decide whether THIS store gets compared, and reserve the budget. Never raises.

    Returns a plan for `settle()`, or None for "not this time". The plan exists because
    the bytes being compared against are only known inside `disk_spill.request_spill`,
    which fills `plan["blob"]` by reference. Nothing is read from the GPU here.
    """
    try:
        if not effective_enabled():
            return None
        with _lock:
            _state["sample_cursor"] += 1
            # EVERY store this module was asked about, sampled or not: the arithmetic
            # an operator reads is `stores_seen = sampled_out + everything else`, and a
            # denominator that only counted the survivors would make the sample rate
            # unreadable from the file it is published in.
            _count("stores_seen")
            if _state["sample_cursor"] % max(1, SAMPLE_EVERY_N_STORES):
                _count("sampled_out")
                return None
            _capture_regions(handler)
            step = _state["step_seq"]
            if _state["step_budget_step"] != step:
                _state["step_budget_step"] = step
                _state["step_used"] = 0
                _state["recheck_used"] = 0
            if _state["step_used"] >= MAX_BLOCKS_PER_STEP:
                # THE BOUND THE STEP THREAD NEEDS: counted, dropped, never deferred to
                # a later step where it would arrive at the wrong cost.
                _count("throttled_step")
                return None
            now = time.monotonic()
            last = _state["last_read_mono"]
            if last is not None and (now - last) < MIN_INTERVAL_SECONDS:
                _count("throttled_interval")
                return None
            record = _state["retained"].get(event)
            gpu_block = None if not record else record["slots"].get(int(slot))
            if gpu_block is None:
                _count("no_gpu_block")
                _reason("no_gpu_block",
                        "a sampled store had no retained GPU block id: its bind was "
                        "evicted, or the engine named no source block for it")
                return None
            _state["step_used"] += 1
            _state["last_read_mono"] = now
            return {"handler": handler, "event": event, "slot": int(slot),
                    "key": key, "gpu_block": int(gpu_block), "blob": None}
    except Exception as exc:  # noqa: BLE001 - never into the engine's step path
        with _lock:
            _count("errors")
        _reason("snapshot", f"the sampling decision raised {type(exc).__name__}")
        return None


def settle(plan, answer=None) -> dict | None:
    """Compare a plan against the bytes the spill path captured. Never raises.

    Called by `disk_store.on_store_completed` immediately after `request_spill`
    returned, on the same thread, so `plan["blob"]` is the arena bytes that spill
    decided to write - or None when it refused, which is its own counted category and
    never a mismatch.
    """
    if not plan:
        return None
    try:
        with _lock:
            blob = plan.get("blob")
            if blob is None:
                # The spill path captured nothing: a full queue, a bad key, an armed
                # gate that closed. There is nothing to compare, and that is NOT
                # evidence about the bytes.
                _count("no_blob")
                return {"checked": False,
                        "reason": "the spill path captured no bytes"
                                  + (f" ({answer.get('reason_kind')})"
                                     if isinstance(answer, dict)
                                     and answer.get("reason_kind") else "")}
            width = len(blob)
            gpu_block = int(plan["gpu_block"])
            handler = plan.get("handler")
        read = _read_gpu_slot(handler, gpu_block, width)
        with _lock:
            kind = read.get("kind")
            if not read.get("ok"):
                if kind == "no_gpu_caches":
                    _count("no_gpu_caches")
                elif kind == "no_gpu_block":
                    _count("no_gpu_block")
                elif read.get("reason_kind") == "region_width_mismatch":
                    # A width disagreement is a LAYOUT answer. Counting it as
                    # staleness, or as a GPU read failure, would hide a real fault
                    # behind a number an operator has learned to ignore.
                    _count("width_mismatch")
                else:
                    # Every device fault lands here - no driver, a refused copy, an
                    # object that is not a tensor - distinguished by the reason tag.
                    _count("gpu_read_failed")
                _reason(f"read:{kind}", str(read.get("reason")))
                return {"checked": False, "reason": read.get("reason"), "kind": kind}
            _count("gpu_read_ok")
            gpu = read["bytes"]
        if gpu == blob:
            with _lock:
                _count("checked")
                _count("equal")
            return {"checked": True, "equal": True, "bytes": width}
        return _classify(handler, gpu_block, gpu, blob, width,
                         int(plan["slot"]))
    except Exception as exc:  # noqa: BLE001 - never into the engine's step path
        with _lock:
            _count("errors")
        _reason("settle", f"the comparison raised {type(exc).__name__}")
        return {"checked": False,
                "reason": f"the comparison raised {type(exc).__name__}"}


def _first_difference(left: bytes, right: bytes) -> tuple:
    """`(offset, scanned_bytes)`: where two blobs first differ, and how far we looked.

    Chunked, so the compare is `memcmp` and the only Python-level byte loop is inside
    the ONE 4 KiB chunk that differs. `offset` is None when the two agree for the whole
    scanned window (or are identical), which is a real answer: "they differ somewhere
    past 256 KB" is what an operator can act on, and a full-slot Python scan of 11.4 MB
    on the step thread is not worth an offset.
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


def _classify(handler, gpu_block: int, gpu: bytes, blob: bytes,
              width: int, slot: int) -> dict:
    """A mismatch: re-read the GPU block ONCE and ask whether the GPU moved."""
    rechecked = False
    changed = None
    reason = None
    with _lock:
        if _state["recheck_used"] < MAX_RECHECKS_PER_STEP:
            _state["recheck_used"] += 1
            _count("rechecks")
            rechecked = True
    if rechecked:
        again = _read_gpu_slot(handler, gpu_block, width)
        if again.get("ok"):
            changed = again["bytes"] != gpu
        else:
            rechecked = False
            reason = again.get("reason")
            _reason(f"recheck:{again.get('kind')}",
                    f"the re-read that would classify a mismatch failed: {reason}")
    offset, scanned = _first_difference(gpu, blob)
    with _lock:
        _count("checked")
        _count("mismatch")
        _track_mismatch(gpu_block)
        if not rechecked:
            _count("mismatch_unrechecked")
            category = "mismatch_unrechecked"
        elif changed:
            _count("mismatch_gpu_changed_on_recheck")
            category = "mismatch_gpu_changed_on_recheck"
        else:
            _count("mismatch_stable")
            category = "mismatch_stable"
        # The SHAPE of the last mismatch, kept for the rank file. The block hash is
        # deliberately NOT published here: `disk_store.observe_bind` refuses to put
        # keys in a snapshot, and this is a snapshot.
        _state["last_mismatch"] = {"category": category, "slot": int(slot),
                                   "gpu_block": int(gpu_block), "bytes": int(width),
                                   "first_difference": offset,
                                   "scanned_bytes": int(scanned),
                                   "rechecked": bool(rechecked),
                                   "gpu_changed_on_recheck": changed}
    return {"checked": True, "equal": False, "category": category,
            "first_difference": offset, "scanned_bytes": scanned,
            "reason": reason}


def _track_mismatch(gpu_block: int) -> None:
    """Remember a mismatched GPU block id as a reuse witness (bounded FIFO)."""
    _state["mismatch_blocks"][gpu_block] = True
    while len(_state["mismatch_blocks"]) > MAX_TRACKED_MISMATCH_BLOCKS:
        _state["mismatch_blocks"].pop(next(iter(_state["mismatch_blocks"])))


# --- THE VIEW -----------------------------------------------------------------
def _region_view() -> dict:
    """The published region block. A stable key set even when no worker was ever seen:
    the shape a report must keep so a reader can tell "moved" from "never seen"."""
    saved = _state["regions"]
    if isinstance(saved, dict):
        return copy.deepcopy(saved)
    return {"order_rule": "sorted(name)", "count": 0, "names": [],
            "truncated": False, "multi_region": False,
            "total_bytes_per_block": None, "widths": None, "source": None}


def counters() -> dict:
    with _lock:
        return dict(_state["counters"])


def _dump_size(payload) -> int:
    """The byte count `worker_diag` will measure. The SAME encoding as its `_dump`
    (`separ=(",", ":")`, `default=str`), because that is the number the cap is applied
    to and a different separator would silently disagree with it."""
    return len(json.dumps(payload, separators=(",", ":"),
                          default=str).encode("utf-8"))


def _fit(out: dict) -> dict:
    """Never hand the writer a section it would REFUSE - and say what was shed.

    `worker_diag` refuses an over-cap section rather than truncating it (right: half a
    section reads as a fact with a piece missing), so the failure mode of a fat payload
    is that the ENTIRE `disk_shadow` key disappears from the rank file - on the boot
    that has eight distinct faults to report and a six-digit counter, which is
    precisely the boot someone is reading the file about. So the section bounds
    ITSELF, in an order that keeps what a reader acts on:

      1. the reason TEXT, first 48 chars of each (the TAG at its front survives, and the
         tag is the part that names the category);
      2. the published region names and widths, to four of each (`count`, `order_rule`
         and `total_bytes_per_block` survive);
      3. the reason list itself, emptied (the counters above it are the reason this
         section exists, and they are NEVER shed - they are fixed-size).

    `last_mismatch` is NEVER a trim target and never gains a key here. It carries only
    numbers (`category`, `slot`, `gpu_block`, `bytes`, `first_difference`,
    `scanned_bytes`, and the two recheck verdicts); there is no hex window in it, so a
    step that "dropped the hex windows" would have had nothing to drop and would have
    ADDED two keys to the payload it was supposed to shrink - making a trimmed section a
    different SHAPE from an untrimmed one, which is the promise this module does not get
    to break. The offset a reader acts on is `first_difference`, and it survives every
    step because nothing here touches that block.

    `trimmed` records the steps taken, so a thin section says why it is thin instead of
    looking like a quiet boot. Every top-level key survives every step, which is what
    keeps this module's "the key set never changes" promise true under pressure too.
    """
    trimmed = out["trimmed"] = []
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return out
    # Reasons are one clipped STRING each ("tag: detail"), so the first shed is the
    # detail: the tag survives, which is the part that names the category.
    out["reasons"] = [str(line)[:TRIM_REASON_CHARS] for line in out["reasons"]]
    trimmed.append(f"reason_chars:{TRIM_REASON_CHARS}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    regions = out["regions"]
    if isinstance(regions, dict):
        names = list(regions.get("names") or [])[:TRIM_REGIONS_PUBLISHED]
        regions["names"] = names
        widths = regions.get("widths")
        if isinstance(widths, dict):
            regions["widths"] = {key: widths[key] for key in names if key in widths}
        if regions.get("count"):
            regions["truncated"] = True
        trimmed.append(f"regions:{TRIM_REGIONS_PUBLISHED}")
    if _dump_size(out) <= MAX_SECTION_BYTES:
        return _counted_trim(out)
    out["reasons"] = []
    trimmed.append("reasons_dropped")
    out = _counted_trim(out)
    if _dump_size(out) > MAX_SECTION_BYTES:
        # EVERY SHED HAS BEEN TAKEN AND IT IS STILL OVER BUDGET, so the writer will
        # refuse the section and the `disk_shadow` key simply will not be in the rank
        # file. There is nothing left to shed without shedding a NUMBER, so the payload
        # goes out honest and the LOG says what the file cannot: the silent absence this
        # function exists to prevent is exactly what a reader must be able to find.
        _note_once("section-over-budget",
                   f"[disk_shadow] section STILL over its byte budget after every trim "
                   f"step ({_dump_size(out)} B over {MAX_SECTION_BYTES} B): the rank "
                   "writer will refuse it, so `disk_shadow` will be absent from "
                   "diagnostics-rank.json", level="error")
    return out


def _counted_trim(out: dict) -> dict:
    """Count a trim ONCE PER SHAPE OF TRIM, and write the count back into the payload.

    A trim is a real event - the faults this module has to report no longer fit in the
    space it was allotted - but `section()` is also what a polling dashboard calls every
    few seconds, so an un-deduplicated counter would be a number that ticks up with the
    polling and that nobody can read. What is worth counting is a CHANGE: the section
    began shedding a different set of detail than it did last time. `trimmed` in the
    payload still says exactly what is missing NOW.

    The counter is written BACK INTO THE PAYLOAD, not only into state: `out`'s counters
    were snapshotted at the top of `section()`, so the trim caused by THIS build would
    otherwise publish `trimmed: [...]` beside a stale `section_trimmed` - a payload that
    contradicts itself about whether it was trimmed.
    """
    signature = tuple(out["trimmed"])
    with _lock:
        if signature != _state["last_trim"]:
            _state["last_trim"] = signature
            _count("section_trimmed")
        out["section_trimmed"] = int(_state["counters"]["section_trimmed"])
    return out


def section() -> dict:
    """The `disk_shadow` block: EVERY KEY IS ALWAYS PRESENT.

    The counters, the gauges, the limits in force, who enabled it, and the region list
    and widths the comparisons use. Building the degraded view from this same function
    is what makes "the key set does not change when the GPU is missing" a structural
    property rather than a copy.
    """
    with _lock:
        out = {key: int(_state["counters"][key]) for key in _COUNTER_KEYS}
        # DEEP copy: `_fit` shortens the reason TEXT in place when the section is over
        # its budget, and a shallow list here would trim the module's OWN stored rows.
        out["reasons"] = copy.deepcopy(_state["reasons"])
        out["retained_events_now"] = len(_state["retained"])
        out["mismatch_blocks_tracked"] = len(_state["mismatch_blocks"])
        # Present on every path, including the one where nothing ever mismatched:
        # "no mismatch yet" and "this build does not report it" must not look alike.
        out["last_mismatch"] = (copy.deepcopy(_state["last_mismatch"])
                                if _state["last_mismatch"] else None)
    enable = enable_state()
    out["enabled"] = enable["enabled"]
    out["enable_source"] = enable["source"]
    out["env_var"] = config.ENV_DISK_SHADOW
    out["env_raw"] = enable["env_raw"]
    out["env_recognised"] = enable["env_recognised"]
    out["latch"] = enable["latch"]
    out["sample_rate"] = 1.0 / float(max(1, SAMPLE_EVERY_N_STORES))
    out["sample_every_n_stores"] = int(SAMPLE_EVERY_N_STORES)
    out["max_blocks_per_step"] = int(MAX_BLOCKS_PER_STEP)
    out["max_rechecks_per_step"] = int(MAX_RECHECKS_PER_STEP)
    out["min_interval_seconds"] = float(MIN_INTERVAL_SECONDS)
    out["max_retained_events"] = int(MAX_RETAINED_EVENTS)
    out["regions"] = _region_view()
    out["section"] = MODULE
    out["writer"] = OWNER_TAG
    out["note"] = ("a mismatch is a QUESTION, not a verdict: only mismatch_stable is "
                   "a layout candidate, and mismatch_gpu_changed_on_recheck is an "
                   "expected race with the engine's own block reuse")
    # THE SELF-IMPOSED BUDGET, applied last, after every other key exists: the writer's
    # cap refuses a fat section outright, and a witness that vanishes on a bad boot is
    # worse than one that sheds its illustrations and keeps its numbers.
    return _fit(out)


def worker_report() -> dict:
    """The rank-JSON contribution. Builds every key, always; owns no file and keeps no
    cross-request state beyond this module's own counters."""
    return section()


def register_contribution() -> dict:
    """Offer this module's verdict to the rank-file writer. Idempotent.

    Registered by `disk_store.install()` right after the `disk` section, i.e. only
    where the spill hook exists: an EngineCore or an API-server process registers
    neither, so its rank file carries no `disk_shadow` key at all instead of an empty
    object claiming a facility that process does not have.
    """
    from . import worker_diag  # noqa: PLC0415 - one writer, and no import cycle
    return worker_diag.register_contribution(
        MODULE, worker_report,
        # OWNER_TAG's own claim, kept true: this reports a STATE, never an age. What
        # genuinely moves with no counter moving is the ENABLEMENT - the POST latch can
        # arm the shadow between two writes of the file, and a reader must see that
        # without diffing the file by hand.
        owner=f"{OWNER_TAG} state, never a window",
        signal=lambda: (int(enable_state()["enabled"]),
                        int(bool(_state["env_enabled"]))))


def status() -> dict:
    """The verb's view: enablement, the limits in force, the regions, the counters."""
    out = {"ok": True, "reason": None, "module": MODULE}
    out.update(section())
    return out


def reset_state_for_tests() -> None:
    with _lock:
        _state["latch"] = False
        _state["latch_set"] = False
        _state["env_resolved"] = False
        _state["env_enabled"] = False
        _state["env_raw"] = None
        _state["env_recognised"] = True
        _state["retained"] = {}
        _state["retained_order"] = []
        _state["step_seq"] = 0
        _state["step_budget_step"] = -1
        _state["step_used"] = 0
        _state["recheck_used"] = 0
        _state["sample_cursor"] = 0
        _state["last_read_mono"] = None
        _state["mismatch_blocks"] = {}
        _state["last_mismatch"] = None
        _state["reasons"] = []
        _state["reason_seen"] = set()
        _state["last_trim"] = None
        _state["notes"] = {}
        _state["regions"] = None
        _state["counters"] = {key: 0 for key in _COUNTER_KEYS}


def view() -> dict:
    """The internal state, deep-copied, for tests and for a human reading a live
    process: a caller cannot mutate the retention it is looking at."""
    with _lock:
        return copy.deepcopy({"retained": _state["retained"],
                              "retained_order": list(_state["retained_order"]),
                              "mismatch_blocks": list(_state["mismatch_blocks"]),
                              "reasons": list(_state["reasons"]),
                              "regions": _state["regions"]})
