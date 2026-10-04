"""Capability 6: a read-only, bounded dump of a few PARKED KV blocks' raw bytes.

WHY THIS EXISTS, AND WHY IT IS SHAPED THIS WAY
----------------------------------------------
The arena is the scarce resource (99.9% occupancy, LRU recycling parked spans),
so "can the parked bytes be compressed?" is a real capacity question, and the
honest answer needs REAL parked bf16 KV bytes - synthetic noise compresses to
~1.0x by construction and would give a useless answer. So this module reads a
HANDFUL of real parked blocks' bytes and returns their metadata plus a sha256.

THE QUESTION IT IS CURRENTLY FUNDING (D vs P)
---------------------------------------------
A parked 3,200-token span costs 5 slots of the CPU pool (one per prefix-cacheable
group), yet `cuda_mem_ops.py:244-250` reads as though every `(gpu_id, cpu_id)`
store writes `bpb` bytes for EVERY region at ONE GPU index - which would mean a
store made on behalf of group 0 also writes other groups' bytes there. Counters
cannot settle that (sibling slots demonstrably hold different GPU block ids, so
the data is not simply duplicated); only bytes can. So the verb dumps an
explicit, bounded BYTE RANGE of one slot for one named region, and the operator
compares regions at the same GPU block id.

THE HARD FACT THAT DICTATES THE DESIGN
--------------------------------------
The raw bytes of a parked block live in the **worker process**, as a slice of
`SimpleCPUOffloadWorker.cpu_kv_caches[name]` - one pinned host tensor per GPU
region, shape `(num_cpu_blocks, ...)`, allocated at `worker.py:235-244` and
filled by the DMA copy in `copy_backend`. The HTTP dev routes reach the
**EngineCore** through the UTILITY channel (`engine_side.py`; core.py:1669-1671),
and NOTHING in a worker is reachable over that channel - the same reason
`worker_diag.py` exists ("a worker therefore writes one small JSON file per
rank"). So this capability is split, exactly like `worker_diag`:

  * an HTTP route + EngineCore handler (`blockdump_for_engine`) that validates a
    request against the caps, echoes the SCHEDULER-side geometry (which IS
    visible there), and ARMS a bounded request by writing one small control file;
  * a worker-side fulfilment (`fulfil_request`, run from a wrapped
    `get_finished`) that reads the bytes straight out of the pinned host buffers
    and writes ONE small bounded response file.

THE OPERATOR COLLECTS THE FILE. A raw block is ~11-12.9 MB (PP0/PP1), so bytes are
NEVER put on the plaintext utility channel. The EngineCore answer is metadata plus
the response file's path; the bytes (base64, with a sha256 per block) land in the
response file the worker writes. `payload.bytes_source` says exactly this in every
response.

WHY READING A BLOCK'S BYTES HERE MUTATES NOTHING (the Q2 safety contract)
-------------------------------------------------------------------------
A parked block's bytes are addressed by pure HOST-MEMORY arithmetic: the tensor's
`data_ptr()` plus `block_id * stride(0) * element_size()`, read with
`ctypes.string_at`. It is a read of an already-resident pinned buffer:

  * NO GPU access, NO CUDA call - it never touches the GPU pools or an engine
    request, and creates no CUDA context (a stray context at 0.96 utilization is
    an outage; this path cannot make one);
  * NO refcount change and NO LRU perturbation: the refcounts and the free-queue
    (the LRU) live on the SCHEDULER-side `cpu_block_pool` `KVCacheBlock` objects
    (`manager.py:207`), which are objects the worker does not hold and this read
    does not touch. The worker's `cpu_kv_caches[name]` is a raw storage tensor,
    not a pool, so indexing a block out of it cannot move an entry in or out of
    any cache or free list. There is no `get_new_blocks`, no `touch`, no `free`,
    no `_insert_block_hash` on this path at all - and no LOOKUP of the pool
    either, which is what `test_no_dump_path_even_looks_at_the_block_pool` pins.

So the safe way to read a block's bytes involves no mutation, and this module
does it that way. If a future fork shape made a byte read require any refcount
or LRU change, the contract is to STOP and report, not to do it.

ARMED VS DISARMED: A FRESH BOOT DOES NO DUMP WORK
-------------------------------------------------
This is a diagnostic on a box that serves real traffic, so the facility is
DISARMED at load and there is deliberately NO env var that arms it (the
`guard_mode`/`VLLM_E2_TIER_DEV_BOOT_INERT` rule: an env var may only ever turn a
dev facility OFF). `POST /plugins/kvtier/blockdump {"op":"arm"}` arms it, `
{"op":"disarm"}` withdraws that, and an arm EXPIRES by itself after
`ARM_DEFAULT_SECONDS` so a forgotten arm cannot leave the door open across a
shift. The EngineCore holds the arm in memory (checked on every verb, expired
lazily on a monotonic clock); a WORKER cannot be told anything over the utility
channel, so the arm crosses to it as one small file (`ARM_FILE`) - the same
one-way-per-process problem `worker_diag` solves with a file, in the other
direction.

The cost of being disarmed is therefore: one `os.stat` of the arm file per
`POLL_INTERVAL_SECONDS` (default 2 s) per worker, and NOTHING else - no open, no
parse, no read of a KV byte, no response file. With `VLLM_E2_TIER_DIAG` unset the
hook is not even installed, which is the `not_deployed` state.

WHAT IS CAPPED, AND IN WHAT UNIT
--------------------------------
  * `MAX_BYTES_PER_REQUEST` (1 MiB, bytes)  - raw bytes per dump request, before
    base64. A range longer than this is REFUSED, naming the cap; nothing is ever
    served truncated.
  * `MAX_BLOCKS` (8, blocks)                - blocks per request.
  * `MAX_REQUESTS_PER_WINDOW` (20, requests per `REQUEST_WINDOW_SECONDS` (300 s))
    - arms of the dump verb. Past it, `refused: rate limit`, counted in
    `window_refused`. Never silently dropped.
  * `MAX_RESPONSE_BYTES` (4 MiB, bytes)     - the response file's size; a
    document that would exceed it is refused rather than written (a worker step
    must never stall writing a giant file).
  * `MAX_ARM_FILE_BYTES` (4 KiB, bytes)     - the arm file read by the worker.
  * `ARM_DEFAULT_SECONDS`/`ARM_MAX_SECONDS` (1800/7200 s) - the arm's lifetime.
  * one request in flight per rank - a second `dump` while one is pending is
    refused (`pending request`), so nothing queues up behind a wedged worker.

A `length` is REQUIRED by the `dump` verb: the caller must ask for an explicit
number of bytes. This module has no path that dumps "a whole arena", and no path
that reads a block without being told how far to read.

THE TOKEN IS NEVER PRINTED OR LOGGED. The route is gated by the same dev token
`/trace` and `/guard` use (`require_dev_state_token`, `hmac.compare_digest` in
`endpoint.py`); this module reads no token and logs none - it never even sees it.
NO USER CONTENT EVER LEAVES THROUGH HERE. These are KV bytes, not tokens: the
plugin's rule is that the diagnostic plane carries no prompt or completion
content, which is why a caller must name an explicit byte length and cannot ask
for "whatever is in this span".

Stdlib only (`base64`, `ctypes`, `hashlib`, `json`, `os`, `sys`, `time`). No
torch and no vllm: the tensors are addressed through the attributes/methods the
fork's object already exposes, read BY NAME, the `worker_diag` posture. Every
entry point catches `Exception`, never `BaseException`, and counts through
`snapshot._error`.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import sys
import time

from . import config, snapshot

# The worker half lives on the fork's worker class, wrapped with the exact
# owner-marker discipline `worker_diag` uses on the same class/method.
TARGET_MODULE = "vllm.v1.simple_kv_offload.worker"
TARGET_CLASS = "SimpleCPUOffloadWorker"
METHOD_GET_FINISHED = "get_finished"
WRAPPED_METHODS = (METHOD_GET_FINISHED,)

# A function defined in THIS package (a sibling layer such as worker_diag's
# wrapper on the SAME method) is composed with, never refused: refusing it would
# mean "the rank-file writer is installed, so the block dump can never be", which
# is a lie the other way round. A marker or definition owned by ANYONE else is
# still refused outright. Same rule as `guard_mode.SIBLING_PACKAGE`.
SIBLING_PACKAGE = (__package__ or "kvtier_dev_plugin") + "."

OWNER_ATTR = "__kvtier_dev_blockdump_owner__"
OWNER_TAG = "kvtier-dev-plugin/blockdump-1"
ORIGINAL_ATTR = "__kvtier_dev_blockdump_original__"

PROTOCOL_VERSION = 2          # 2: ranges (offset_bytes/length_bytes) in the pair
ROLE = "blockdump"
WRITER_TAG = "kvtier-dev-plugin/blockdump-1"

# --- the caps, and the unit of every one of them ----------------------------
MAX_BLOCKS = 8                    # blocks per request (a handful, never the arena)
DEFAULT_BLOCKS = 4                # reported in `caps`, never a silent wire default
MAX_BYTES_PER_REQUEST = 1024 * 1024   # BYTES of raw KV bytes per request (pre-base64)
MAX_BYTES = 256 * 1024 * 1024     # BYTES: the ceiling on ONE block's own size
MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # BYTES: the response file's size cap
MAX_ARM_FILE_BYTES = 4096         # BYTES: the arm file, read by a worker
MAX_STRING_CHARS = 160
MAX_AVAILABLE_NAMES = 32          # region names echoed back on an unknown group

# The dump rate window: requests per rolling window, in SECONDS.
REQUEST_WINDOW_SECONDS = 300.0
MAX_REQUESTS_PER_WINDOW = 20

# How long an armed state lives, in SECONDS. An arm is not a subscription.
ARM_DEFAULT_SECONDS = 1800.0
ARM_MIN_SECONDS = 5.0
ARM_MAX_SECONDS = 7200.0

# The worker's poll floor, in SECONDS. While disarmed this costs one `os.stat`.
POLL_INTERVAL_SECONDS = 2.0
# How long a pending request may sit before a worker treats it as stale, in
# SECONDS. A request from a dead boot must not be fulfilled by a later one.
REQUEST_TTL_SECONDS = 120.0

# The three files (names are a fixed, rank-keyed shape under VLLM_E2_TIER_DIAG).
REQUEST_FILE = "blockdump-request-{rank}.json"
RESPONSE_FILE = "blockdump-{rank}.json"
ARM_FILE = "blockdump-arm.json"

# `op` values of the verb. `dump` is the only one that arms a request.
BLOCKDUMP_OPS = ("status", "arm", "disarm", "dump")
MUTATING_BLOCKDUMP_OPS = ("arm", "disarm", "dump")

_state: dict = {
    # install / role
    "enabled": False,
    "installed": False,
    "class": None,
    "installs": 0,
    "reinstalls": 0,
    "uninstalls": 0,
    "skips": 0,
    "skip_reason": None,
    "nested": [],
    # the arm, in THIS process (the EngineCore holds it; a worker mirrors the file)
    "armed": False,
    "arm_expires_mono": None,
    "arm_expires_wall": None,
    "arm_seconds": None,
    "arms": 0,
    "disarms": 0,
    "arm_errors": 0,
    # the dump window (EngineCore side)
    "window_start_mono": None,
    "window_requests": 0,
    "window_refused": 0,
    "requests_armed": 0,
    "pending_refused": 0,
    # the worker side
    "polls": 0,
    "poll_suppressed": 0,
    "last_poll_mono": None,
    "requests_read": 0,
    "fulfilled": 0,
    "fulfil_errors": 0,
    "refused": 0,
    "dropped": 0,
    "fallbacks": 0,
    "last_response_path": None,
    "last_error": None,
    "last_poll_reason": None,
}


def reset_state_for_tests() -> None:
    """Back to a fresh boot: not installed, not armed, nothing counted."""
    _state.update({
        "enabled": False, "installed": False, "class": None, "installs": 0,
        "reinstalls": 0, "uninstalls": 0, "skips": 0, "skip_reason": None,
        "nested": [],
        "armed": False, "arm_expires_mono": None, "arm_expires_wall": None,
        "arm_seconds": None, "arms": 0, "disarms": 0, "arm_errors": 0,
        "window_start_mono": None, "window_requests": 0, "window_refused": 0,
        "requests_armed": 0, "pending_refused": 0,
        "polls": 0, "poll_suppressed": 0, "last_poll_mono": None,
        "requests_read": 0, "fulfilled": 0,
        "fulfil_errors": 0, "refused": 0, "dropped": 0, "fallbacks": 0,
        "last_response_path": None, "last_error": None, "last_poll_reason": None,
    })


def counters() -> dict:
    return dict(_state)


def deployed() -> bool:
    """Is there a channel to a worker at all? `not_deployed` = there is not."""
    return config.diagnostics_dir() is not None


def is_armed() -> bool:
    """This process's arm, expired lazily on a monotonic clock.

    A reader must never get a confident `True` from an arm that lapsed while
    nobody looked, which is the `intentional_stop_active()` lesson: the expiry is
    checked HERE, on every read, not by a background sweep.
    """
    deadline = _state["arm_expires_mono"]
    if _state["armed"] and isinstance(deadline, (int, float)) \
            and time.monotonic() >= deadline:
        _state["armed"] = False
        _state["disarms"] += 1
        _state["last_poll_reason"] = "the arm expired"
    return bool(_state["armed"])


def capability() -> str:
    """The three-state marker a reader distinguishes: never a bare bool.

    `not_deployed`          - no diagnostics directory, so no channel to a worker
                              and no hook anywhere. Nothing this verb does can
                              ever become work here.
    `deployed_disarmed`     - the channel exists, the hook can be installed, and
                              no arm is live: zero dump work.
    `armed_and_ready`       - armed and inside its lifetime; a request armed by
                              `op: dump` will be answered by a worker within
                              `POLL_INTERVAL_SECONDS` of its next poll.
    """
    if not deployed():
        return "not_deployed"
    return "armed_and_ready" if is_armed() else "deployed_disarmed"


def arm_state() -> dict:
    """The arm as data: live, and if so for how much longer (in SECONDS)."""
    armed = is_armed()
    remaining = None
    deadline = _state["arm_expires_wall"]
    if armed and isinstance(deadline, (int, float)):
        remaining = round(max(0.0, deadline - time.time()), 3)
    return {"armed": armed,
            "capability": capability(),
            "seconds_left": remaining,
            "expires_at_wallclock": (round(float(deadline), 6)
                                     if isinstance(deadline, (int, float))
                                     and armed else None),
            "default_seconds": ARM_DEFAULT_SECONDS,
            "max_seconds": ARM_MAX_SECONDS,
            "window_seconds": REQUEST_WINDOW_SECONDS,
            "max_requests_per_window": MAX_REQUESTS_PER_WINDOW,
            "window_requests": int(_state["window_requests"]),
            "window_refused": int(_state["window_refused"]),
            "note": ("an arm expires on its own; there is no env var that arms "
                     "this facility, and a fresh boot does no dump work")}


def arm(seconds=None) -> dict:
    """Arm the dump facility, in THIS process, and tell the worker by file.

    Refused when there is no channel (`not_deployed`): an armed EngineCore whose
    worker can never see the arm would be a knob that lies.
    """
    wanted = ARM_DEFAULT_SECONDS
    if seconds is not None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return {"armed": False, "reason": "seconds must be a number"}
        wanted = max(ARM_MIN_SECONDS, min(ARM_MAX_SECONDS, float(seconds)))
    if not deployed():
        _state["skips"] += 1
        return {"armed": False,
                "capability": "not_deployed",
                "reason": f"{config.ENV_DIAG_DIR} is not set: no channel to a "
                          "worker, so arming would only be a lie"}
    now = time.time()
    _state["armed"] = True
    _state["arm_expires_mono"] = time.monotonic() + wanted
    _state["arm_expires_wall"] = now + wanted
    _state["arm_seconds"] = wanted
    _state["arms"] += 1
    written = _write_arm_file(now, wanted)
    return {"armed": True, "seconds": wanted,
            "capability": "armed_and_ready",
            "arm_file": written.get("path"),
            "arm_file_written": bool(written.get("written")),
            "arm_file_reason": written.get("reason"),
            "worker_sees_it_within_seconds": POLL_INTERVAL_SECONDS,
            "reason": None}


def disarm() -> dict:
    """Withdraw the arm here, and remove the file so a worker stands down too."""
    was = bool(_state["armed"])
    _state["armed"] = False
    _state["arm_expires_mono"] = None
    _state["arm_expires_wall"] = None
    _state["arm_seconds"] = None
    if was:
        _state["disarms"] += 1
    removed = _remove_arm_file()
    return {"armed": False, "was_armed": was,
            "capability": capability(),
            "arm_file_removed": removed,
            "reason": "dumps are refused again; a worker stands down within its "
                      "next poll"}


# --- the arm file (EngineCore writes it, a worker reads it) ------------------

def _arm_path():
    directory = config.diagnostics_dir()
    if directory is None:
        return None
    return os.path.join(str(directory), ARM_FILE)


def _write_arm_file(now: float, seconds: float) -> dict:
    """Atomically publish the arm. Contained: a failed arm file is reported."""
    path = _arm_path()
    if path is None:
        return {"written": False, "reason": "no diagnostics directory"}
    document = {"version": PROTOCOL_VERSION, "role": ROLE,
                "writer": WRITER_TAG, "armed": True,
                "armed_ts": round(now, 6),
                "expires_ts": round(now + seconds, 6),
                "seconds": round(seconds, 3),
                "epoch": str(snapshot.epoch()), "pid": os.getpid()}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        raw = json.dumps(document, separators=(",", ":")).encode("utf-8")
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
        return {"written": True, "path": path, "bytes": len(raw)}
    except Exception as exc:  # noqa: BLE001 - an arm that cannot be published is data
        _state["arm_errors"] += 1
        snapshot._error(exc, "blockdump arm file")
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
        _state["arm_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return False
    return True


def _read_arm_file(path):
    """(armed, reason) from the arm file, as a worker reads it.

    Cheap first: a `stat` decides whether to open at all, so a box that has never
    been armed pays one `os.stat` per poll and never opens a file. The read is
    size-capped before it happens, and a file from another boot or a broken one
    means DISARMED - the fail-closed direction for a diagnostic.
    """
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
    if not isinstance(document, dict):
        return False, "the arm file is not an object"
    if document.get("armed") is not True:
        return False, "the arm file does not say armed"
    expires = document.get("expires_ts")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        return False, "the arm file has no numeric expires_ts"
    if time.time() >= float(expires):
        return False, "the arm expired"
    return True, "armed"


# --- small validators --------------------------------------------------------

def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _clip(text, chars: int = MAX_STRING_CHARS):
    if isinstance(text, str) and len(text) > chars:
        return text[:chars] + "..."
    return text


def _lookup_region(caches, group):
    """(key, tensor) for a region name, accepting "3" and 3 alike, or None.

    The keys of `cpu_kv_caches` are the fork's region names; an operator reading a
    geometry dump has the group INDEX, and the two are the same digits for our
    five prefix-cacheable groups. Trying both is a lookup in a dict the fork
    already handed us - a pure read, and it is what makes an off-by-string a
    refusal that names the available regions rather than a confusing miss.
    """
    if not isinstance(caches, dict) or not caches:
        return None
    if group in caches:
        return group, caches[group]
    if isinstance(group, str) and group.strip().lstrip("-").isdigit():
        as_int = int(group.strip())
        if as_int in caches:
            return as_int, caches[as_int]
    elif isinstance(group, int) and not isinstance(group, bool):
        as_text = str(group)
        if as_text in caches:
            return as_text, caches[as_text]
    return None


# --- the pure reader: a parked block's bytes, by host-memory address ---------

def read_block_bytes(tensor, block_id: int, offset_bytes: int = 0,
                     length_bytes: int | None = None) -> dict:
    """One bounded byte range of one block, from a pinned host tensor.

    A pure host-memory read: `data_ptr() + block_id*stride(0)*element_size +
    offset`, `ctypes.string_at`, exactly `length_bytes` of it. Returns
    {ok, bytes, sha256, dtype, shape, element_size, bytes_per_block, offset_bytes,
    length_bytes} or {ok: False, reason}. Never imports torch, never calls CUDA,
    never touches a pool, a refcount or the LRU (see the module docstring).

    A range that would cross the end of the block, or that exceeds
    `MAX_BYTES_PER_REQUEST`, is REFUSED with the numbers named - never clipped to
    fit, because a silently short range is exactly the "D vs P" false negative
    this reader exists to avoid.
    """
    if not callable(getattr(tensor, "data_ptr", None)):
        return {"ok": False, "reason": "the region has no data_ptr() to read"}
    if not callable(getattr(tensor, "stride", None)):
        return {"ok": False, "reason": "the region has no stride() to index"}
    if not callable(getattr(tensor, "element_size", None)):
        return {"ok": False, "reason": "the region has no element_size()"}
    try:
        shape = tuple(getattr(tensor, "shape", ()) or ())
        if len(shape) < 1:
            return {"ok": False, "reason": "the region is 0-dimensional"}
        blocks = int(shape[0])
        if block_id < 0 or block_id >= blocks:
            return {"ok": False,
                    "reason": f"block_id {block_id} is outside the region's "
                              f"{blocks} blocks"}
        stride0 = _int(tensor.stride(0))
        element_size = _int(tensor.element_size())
        if not stride0 or not element_size:
            return {"ok": False,
                    "reason": "stride(0) or element_size() is not a positive int"}
        bytes_per_block = stride0 * element_size
        if bytes_per_block <= 0 or bytes_per_block > MAX_BYTES:
            return {"ok": False,
                    "reason": f"block size {bytes_per_block} B is not in (0, "
                              f"{MAX_BYTES}]"}
        # CONTIGUITY, and why it is a gate rather than an assumption.
        # The address we are about to form is `base + block_id * stride0 * itemsize
        # + offset`, i.e. it assumes the region's slots tile the allocation with no
        # holes and nothing between the last slot and the end of the buffer. That is
        # exactly true for a contiguous tensor, and it is what the fork's own copy
        # path assumes too (`cuda_mem_ops.py` computes src/dst addresses the same
        # way), so a non-contiguous region here would mean the tier itself is
        # already computing wrong addresses.
        # We still refuse it, because this is the one read on this box whose failure
        # mode cannot be caught: `ctypes.string_at` past the end of the allocation
        # is a SIGSEGV, not a Python exception, and `except Exception` below cannot
        # save a worker process - it takes serving down. A shape where
        # `shape[0] * stride(0) * itemsize != nbytes` is precisely a shape where the
        # arithmetic above can leave the buffer, so the invariant is checked against
        # the tensor's own byte count rather than trusted.
        nbytes = getattr(tensor, "nbytes", None)
        if isinstance(nbytes, int) and nbytes != blocks * bytes_per_block:
            return {"ok": False,
                    "reason": f"the region is not a flat run of {blocks} slots of "
                              f"{bytes_per_block} B (shape implies "
                              f"{blocks * bytes_per_block} B, nbytes is {nbytes} B); "
                              "refusing to form an address that could leave the "
                              "allocation"}
        offset = _int(offset_bytes)
        if offset is None or offset < 0:
            return {"ok": False, "reason": "offset_bytes must be a non-negative int"}
        if length_bytes is None:
            length = bytes_per_block - offset
        else:
            length = _int(length_bytes)
            if length is None or length <= 0:
                return {"ok": False,
                        "reason": "length_bytes must be a positive int (this verb "
                                  "never reads an implicit range)"}
        if length > MAX_BYTES_PER_REQUEST:
            return {"ok": False,
                    "reason": f"the range is {length} bytes; the hard cap is "
                              f"{MAX_BYTES_PER_REQUEST} bytes per request"}
        if offset >= bytes_per_block or offset + length > bytes_per_block:
            return {"ok": False,
                    "reason": f"offset {offset} + length {length} does not fit "
                              f"inside the block's {bytes_per_block} bytes"}
        base = _int(tensor.data_ptr())
        if base is None or base <= 0:
            return {"ok": False, "reason": "data_ptr() is not a usable address"}
        # The read itself. string_at copies exactly `length` bytes from an address
        # inside a buffer the fork already allocated and pinned.
        raw = ctypes.string_at(base + block_id * bytes_per_block + offset, length)
    except Exception as exc:  # noqa: BLE001 - an unreadable region is data, not a fault
        return {"ok": False, "reason": f"the host read failed: {type(exc).__name__}"}
    if len(raw) != length:
        # A short read would be a silent truncation, which is the one outcome a
        # byte-level investigation cannot afford. Report it as a refusal instead.
        return {"ok": False,
                "reason": f"the host read returned {len(raw)} of {length} bytes"}
    return {
        "ok": True,
        "bytes": raw,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "shape": [int(dim) for dim in shape],
        "element_size": element_size,
        "bytes_per_block": bytes_per_block,
        "offset_bytes": offset,
        "length_bytes": length,
    }


def _available(caches) -> list:
    """The region names, sorted and capped - what a wrong `group` is answered with.

    A refusal that names the alternatives is the difference between an operator
    finishing the D-vs-P question and filing another question about it.
    """
    return sorted(str(name) for name in caches)[:MAX_AVAILABLE_NAMES]


def dump_blocks(worker, group, block_ids, offset_bytes: int = 0,
                length_bytes: int | None = None) -> dict:
    """Read a HARD-CAPPED set of byte ranges out of the worker's arena.

    `group` selects the `cpu_kv_caches` region (`worker.py:235`, the same key the
    DMA backend copies between); `block_ids` are the slots to read at
    `offset_bytes`/`length_bytes`. Refused WHOLE - never partially, never
    truncated - if it names more than MAX_BLOCKS blocks, the region is unknown, or
    the raw total would exceed `MAX_BYTES_PER_REQUEST`.
    """
    requested = list(block_ids or [])
    if not requested:
        return {"refused": True, "reason": "no block ids were named"}
    if len(requested) > MAX_BLOCKS:
        return {"refused": True,
                "reason": f"a request names {len(requested)} blocks; the hard cap "
                          f"is {MAX_BLOCKS} (this route never dumps the arena)"}
    caches = getattr(worker, "cpu_kv_caches", None)
    if not isinstance(caches, dict) or not caches:
        return {"refused": True,
                "reason": "cpu_kv_caches is not present yet (the worker has not "
                          "registered its KV caches)"}
    if group is None:
        return {"refused": True,
                "reason": "a group (region name) must be named; available: "
                          f"{_available(caches)}"}
    found = _lookup_region(caches, group)
    if found is None:
        return {"refused": True,
                "reason": f"group {group!r} is not a cpu_kv_caches region; "
                          f"available: {_available(caches)}"}
    key, tensor = found
    blocks = []
    total_bytes = 0
    for block_id in requested:
        index = _int(block_id)
        if index is None:
            return {"refused": True, "reason": f"block id {block_id!r} is not an int"}
        read = read_block_bytes(tensor, index, offset_bytes, length_bytes)
        if not read.get("ok"):
            return {"refused": True,
                    "reason": f"block {index} in group {key!r}: "
                              f"{read.get('reason')}"}
        total_bytes += read["length_bytes"]
        if total_bytes > MAX_BYTES_PER_REQUEST:
            return {"refused": True,
                    "reason": f"the request would return {total_bytes} bytes; the "
                              f"hard cap is {MAX_BYTES_PER_REQUEST} bytes per "
                              "request (never served truncated)"}
        blocks.append({"group": str(key), "block_id": index,
                       "offset_bytes": read["offset_bytes"],
                       "length_bytes": read["length_bytes"],
                       "bytes_returned": read["length_bytes"],
                       "sha256": read["sha256"],
                       "dtype": read["dtype"], "shape": read["shape"],
                       "element_size_bytes": read["element_size"],
                       "bytes_per_block": read["bytes_per_block"],
                       "encoding": "base64",
                       "b64": base64.b64encode(read["bytes"]).decode("ascii")})
    return {"refused": False, "group": str(key), "blocks": blocks,
            "total_bytes": total_bytes,
            "blocks_requested": len(requested)}


# --- the scheduler-side verb (the HTTP route reaches this) -------------------

def _group_geometry(scheduler, group_index):
    """That group's geometry, from the scheduler's own CPU config, if visible."""
    if group_index is None or scheduler is None:
        return None
    manager = getattr(getattr(scheduler, "connector", None), "scheduler_manager",
                      None)
    if manager is None:
        return None
    sizes = getattr(manager, "group_block_sizes", None)
    out = {"num_cpu_blocks": _int(getattr(manager, "num_cpu_blocks", None))}
    if isinstance(sizes, (list, tuple)) and 0 <= group_index < len(sizes):
        out["group_block_size"] = _int(sizes[group_index])
    return out


def _window_open(now: float) -> bool:
    """Is this request inside the rate window? Advances the window's counters.

    A rolling window on a monotonic clock, deliberately dumb: when it lapses it
    restarts empty. `window_refused` counts the refusals so "nothing happened" is
    always attributable rather than inferred.
    """
    start = _state["window_start_mono"]
    if not isinstance(start, (int, float)) or (now - start) >= REQUEST_WINDOW_SECONDS:
        _state["window_start_mono"] = now
        _state["window_requests"] = 0
    if _state["window_requests"] >= MAX_REQUESTS_PER_WINDOW:
        _state["window_refused"] += 1
        return False
    _state["window_requests"] += 1
    return True


def _pending_request_path(rank):
    directory = config.diagnostics_dir()
    if directory is None or rank is None:
        return None
    return os.path.join(str(directory), REQUEST_FILE.format(rank=rank))


def _pending_is_live(path) -> bool:
    """True if a request file is still inside its TTL. Stat-cheap, fail-open to
    "not live" (which lets a fresh request overwrite a stale one instead of
    wedging the channel forever behind a worker that died)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception:  # noqa: BLE001 - unreadable is stale, and stale is replaceable
        return False
    expires = document.get("expires_ts") if isinstance(document, dict) else None
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        return False
    return time.time() < float(expires)


def blockdump_for_engine(engine, args=None) -> dict:
    """THE engine-side handler body for `/blockdump`. Never raises an Exception.

    `op` selects the shape: `status` (the default, and all a GET sends), `arm`,
    `disarm`, `dump`. This half validates against the caps and writes the one
    small control file a worker answers; it CANNOT reach worker memory and never
    pretends to. `bytes_source` says so in every payload.
    """
    snapshot._state["calls"] += 1
    started = time.perf_counter()
    arguments = args if isinstance(args, dict) else {}
    payload: dict = {"version": PROTOCOL_VERSION, "verb": "blockdump"}
    try:
        op = arguments.get("op", "status")
        op = op.strip().lower() if isinstance(op, str) else None
        payload["op"] = op
        payload["caps"] = {
            "max_bytes_per_request": MAX_BYTES_PER_REQUEST,
            "max_blocks": MAX_BLOCKS,
            "max_response_bytes": MAX_RESPONSE_BYTES,
            "request_window_seconds": REQUEST_WINDOW_SECONDS,
            "max_requests_per_window": MAX_REQUESTS_PER_WINDOW,
            "arm_default_seconds": ARM_DEFAULT_SECONDS,
            "arm_max_seconds": ARM_MAX_SECONDS,
            "poll_interval_seconds": POLL_INTERVAL_SECONDS,
            "units": {"max_bytes_per_request": "bytes",
                      "max_blocks": "blocks",
                      "max_response_bytes": "bytes",
                      "request_window_seconds": "seconds",
                      "max_requests_per_window": "requests",
                      "poll_interval_seconds": "seconds"},
        }
        payload["bytes_source"] = (
            "raw parked bytes live in the WORKER (cpu_kv_caches) and are NOT "
            "returned over the utility channel; the worker writes them (base64, "
            "per-block sha256) to the response file named here. This answer is "
            "metadata + the armed request, and mutates nothing.")
        payload["ops"] = list(BLOCKDUMP_OPS)
        payload["armed_state"] = arm_state()

        if op not in BLOCKDUMP_OPS:
            payload["refused"] = (f"unknown op {arguments.get('op')!r}; expected "
                                  f"one of {list(BLOCKDUMP_OPS)}")
        elif op == "arm":
            payload.update(arm(arguments.get("seconds")))
        elif op == "disarm":
            payload.update(disarm())
        elif op == "status":
            payload["status"] = counters()
            payload["request_armed"] = False
            payload["response_file"] = _response_path_for_rank()
        else:  # "dump"
            payload.update(_arm_request(engine, arguments))
        unknown = sorted(key for key in arguments
                         if key not in ("op", "group", "block_id", "block_ids",
                                        "offset", "offset_bytes", "length",
                                        "length_bytes", "seconds"))
        payload["armed_state"] = arm_state()
        if unknown:
            payload["unknown_args"] = unknown
        payload["guard_mode"] = None
        payload["blockdump_armed"] = is_armed()
        payload["ts"] = round(time.time(), 6)
        payload["epoch"] = str(snapshot.epoch())
        payload["build_us"] = int((time.perf_counter() - started) * 1e6)
        return snapshot.json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler loop
        snapshot._error(exc, "blockdump")
        return {"version": PROTOCOL_VERSION, "verb": "blockdump",
                "epoch": str(snapshot.epoch()),
                "blockdump_armed": bool(_state["armed"]),
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def _response_path_for_rank():
    """This rank's response file path, or None with no channel. Read-only."""
    directory = config.diagnostics_dir()
    if directory is None:
        return None
    from . import worker_diag  # noqa: PLC0415 - shared rank/diagnostics
    rank, _source = worker_diag.rank_id()
    if rank is None:
        return None
    return os.path.join(str(directory), RESPONSE_FILE.format(rank=rank))


def _arm_request(engine, arguments) -> dict:
    """Validate a dump request and write the one control file.

    Refused with the numbers named, never clipped to fit.
    """
    out: dict = {"request_armed": False, "response_file": None}
    if not is_armed():
        _state["refused"] += 1
        out["refused"] = ("the dump facility is disarmed (capability "
                          f"{capability()!r}); POST "
                          f"{config.ROUTE_PREFIX}{config.ROUTE_BLOCKDUMP} "
                          '{"op":"arm"} first - a fresh boot does no dump work '
                          "by design")
        return out
    directory = config.diagnostics_dir()
    if directory is None:
        _state["refused"] += 1
        out["refused"] = (f"{config.ENV_DIAG_DIR} is not set: with no diagnostics "
                          "directory there is no channel to the worker, so "
                          "nothing was armed")
        out["capability"] = "not_deployed"
        return out
    raw_ids = arguments.get("block_ids")
    if raw_ids is None:
        raw_ids = [arguments.get("block_id")]
    block_ids = [value for value in (raw_ids or [])
                 if not isinstance(value, (str, bytes)) or str(value).isdigit()]
    block_ids = [int(value) if isinstance(value, str) else value
                 for value in block_ids]
    group = arguments.get("group")
    group = group if isinstance(group, str) else (
        str(group) if group is not None else None)
    offset = arguments.get("offset_bytes", arguments.get("offset", 0))
    length = arguments.get("length_bytes", arguments.get("length"))
    offset = _int(offset) if offset is not None else 0
    length = _int(length)
    out["requested"] = {"group": group, "block_ids": block_ids[:MAX_BLOCKS + 1],
                        "count": len(block_ids),
                        "offset_bytes": offset, "length_bytes": length}
    scheduler = getattr(engine, "scheduler", None)
    index = None
    if group is not None and group.strip().lstrip("-").isdigit():
        index = int(group.strip())
    out["group_geometry"] = _group_geometry(scheduler, index)

    if not block_ids:
        _state["refused"] += 1
        out["refused"] = ('no block_id was named; pass {"group": "<region>", '
                          '"block_id": N, "offset": 0, "length": 4096}')
        return out
    if len(block_ids) > MAX_BLOCKS:
        _state["refused"] += 1
        out["refused"] = (f"the request names {len(block_ids)} blocks; the hard "
                          f"cap is {MAX_BLOCKS}")
        return out
    if length is None:
        _state["refused"] += 1
        out["refused"] = (f"an explicit length is required (1 to "
                          f"{MAX_BYTES_PER_REQUEST} bytes); this verb never reads "
                          "a range nobody asked for")
        return out
    if length <= 0 or length > MAX_BYTES_PER_REQUEST:
        _state["refused"] += 1
        out["refused"] = (f"length {length} is not in [1, "
                          f"{MAX_BYTES_PER_REQUEST}] bytes")
        return out
    if offset is None or offset < 0:
        _state["refused"] += 1
        out["refused"] = "offset must be a non-negative int (bytes)"
        return out
    if any(_int(value) is None for value in block_ids):
        _state["refused"] += 1
        out["refused"] = "every block id must be an int"
        return out

    from . import worker_diag  # noqa: PLC0415 - shared rank/diagnostics
    rank, rank_source = worker_diag.rank_id()
    if rank is None:
        _state["refused"] += 1
        out["refused"] = (f"this process resolved no rank id ({rank_source}); the "
                          "request file is rank-keyed, so nothing was armed")
        return out

    request_path = _pending_request_path(rank)
    if request_path is not None and os.path.exists(request_path):
        if _pending_is_live(request_path):
            # Checked BEFORE the rate window: a refusal that armed nothing must not
            # cost the caller a window slot.
            _state["pending_refused"] += 1
            _state["refused"] += 1
            out["refused"] = (f"a request for rank {rank} is already pending in "
                              f"{request_path}; one dump is in flight per rank, "
                              "collect the response file (or wait for its TTL to "
                              f"expire, {int(REQUEST_TTL_SECONDS)}s)")
            out["pending_request_file"] = request_path
            return out
        out["superseded_pending_request"] = request_path

    if not _window_open(time.monotonic()):
        _state["refused"] += 1
        out["refused"] = (f"the rate window allows {MAX_REQUESTS_PER_WINDOW} "
                          f"dumps per {REQUEST_WINDOW_SECONDS}s; this request was "
                          f"refused (window_refused="
                          f"{int(_state['window_refused'])})")
        return out

    now = time.time()
    request = {"version": PROTOCOL_VERSION, "rank": rank, "group": group,
               "block_ids": block_ids[:MAX_BLOCKS],
               "offset_bytes": offset, "length_bytes": length,
               "armed_ts": round(now, 6),
               "expires_ts": round(now + REQUEST_TTL_SECONDS, 6),
               "epoch": str(snapshot.epoch())}
    path = os.path.join(str(directory), REQUEST_FILE.format(rank=rank))
    try:
        os.makedirs(str(directory), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(request, handle, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - a failed arm is data, not a 500
        _state["refused"] += 1
        snapshot._error(exc, "blockdump arm request")
        out["refused"] = (f"could not write the request file: "
                          f"{type(exc).__name__}")
        return out
    _state["requests_armed"] += 1
    out["request_armed"] = True
    out["request_file"] = path
    out["response_file"] = os.path.join(str(directory),
                                        RESPONSE_FILE.format(rank=rank))
    out["answer_within_seconds"] = POLL_INTERVAL_SECONDS
    out["how_to_read_the_bytes"] = (
        f"cat {out['response_file']} - written by the worker on its next "
        f"get_finished (within ~{POLL_INTERVAL_SECONDS}s of this arming). TWO "
        "operational facts: get_finished only runs while the engine is STEPPING, "
        "so an idle engine answers at its next step rather than on a timer; and "
        "this response file is OVERWRITTEN by the next dump, so collect it (or rm "
        "it) before arming another. The request file is removed once answered.")
    out["list_the_region_names"] = (
        "the region names are the worker's cpu_kv_caches keys, visible only "
        "there: arm a dump with any made-up group and the response file's reason "
        "field names every region this rank has. That costs one window slot.")
    out["refused"] = None
    return out


# --- the worker half: fulfil the armed request, write a bounded file ---------

def _read_request(directory, rank):
    path = os.path.join(str(directory), REQUEST_FILE.format(rank=rank))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return None, None
    except Exception as exc:  # noqa: BLE001 - an unreadable request is a skip
        _state["skip_reason"] = f"request unreadable: {type(exc).__name__}"
        return None, None
    return raw, path


def _request_is_live(request) -> bool:
    """A request from a dead boot must never be answered by a later worker."""
    if not isinstance(request, dict):
        return False
    expires = request.get("expires_ts")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        return False
    return time.time() < float(expires)


def fulfil_request(handler, request=None) -> dict:
    """If this rank has a pending, in-bounds request, answer it in ONE file.

    Runs from a wrapped get_finished, AFTER the fork's own work (the
    worker_diag discipline), so it can never reorder what the scheduler is told.
    Reads the bytes with the pure reader above - no refcount, no LRU, no CUDA -
    writes the response atomically, and removes the request so it fires once.

    A dump that RAISES is answered the same way as one that is refused: a
    response file with the reason, and the request consumed. Leaving it behind
    would retry a broken tensor on every poll forever, which turns one bad region
    into an endless error loop on a serving box.
    """
    directory = config.diagnostics_dir()
    if directory is None:
        return {"written": False, "reason": f"{config.ENV_DIAG_DIR} is unset"}
    from . import worker_diag  # noqa: PLC0415
    rank, _source = worker_diag.rank_id()
    if rank is None:
        return {"written": False, "reason": "no rank id"}
    request_path = os.path.join(str(directory), REQUEST_FILE.format(rank=rank))
    if request is None:
        request, request_path = _read_request(directory, rank)
    if request is None:
        return {"written": False, "reason": "no pending request"}
    _state["requests_read"] += 1
    if not _request_is_live(request):
        # Consume it: a stale request is not work, and re-reporting it every poll
        # would be a log line per poll forever.
        _state["dropped"] += 1
        _remove_quietly(request_path)
        return {"written": False, "reason": "the request expired before it was "
                                            "answered; it was dropped, not served"}
    try:
        result = dump_blocks(handler, request.get("group"),
                             request.get("block_ids"),
                             request.get("offset_bytes", 0),
                             request.get("length_bytes"))
    except Exception as exc:  # noqa: BLE001 - a broken region is a refusal, not a step fault
        _state["fulfil_errors"] += 1
        _state["refused"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "blockdump dump")
        result = {"refused": True,
                  "reason": f"the dump raised {type(exc).__name__}; the request "
                            "was consumed so it will not retry"}
    document = {"version": PROTOCOL_VERSION, "role": ROLE, "rank": rank,
                "ts": round(time.time(), 6),
                "epoch": str(snapshot.epoch()),
                "writer": WRITER_TAG,
                "refused": bool(result.get("refused")),
                "reason": result.get("reason"),
                "group": result.get("group"),
                "total_bytes": int(result.get("total_bytes") or 0),
                "request": {"group": request.get("group"),
                            "block_ids": list(request.get("block_ids") or []),
                            "offset_bytes": request.get("offset_bytes", 0),
                            "length_bytes": request.get("length_bytes")},
                "caps": {"max_bytes_per_request": MAX_BYTES_PER_REQUEST,
                         "max_blocks": MAX_BLOCKS,
                         "max_response_bytes": MAX_RESPONSE_BYTES},
                "units": {"total_bytes": "bytes of raw KV bytes (pre-base64)",
                          "offset_bytes": "bytes from the start of the slot",
                          "length_bytes": "bytes returned for that block",
                          "bytes_per_block": "bytes per slot in that region",
                          "element_size_bytes": "bytes per element of the dtype"},
                "blocks": []}
    for block in result.get("blocks", []):
        # The base64 payload is the ONLY place the bytes leave the worker; the
        # per-block sha256 lets the collector verify the download.
        document["blocks"].append(dict(block))
    raw = json.dumps(document, separators=(",", ":"), default=str).encode("utf-8")
    if len(raw) > MAX_RESPONSE_BYTES:
        # Refuse rather than write: a multi-megabyte synchronous write inside a
        # worker step is a stall, and a stall on this box is a latency spike for
        # real traffic. Count it, say why, and consume the request.
        _state["refused"] += 1
        _remove_quietly(request_path)
        return {"written": False,
                "reason": f"the response would be {len(raw)} bytes (cap "
                          f"{MAX_RESPONSE_BYTES}); refused, request consumed"}
    response_path = os.path.join(str(directory), RESPONSE_FILE.format(rank=rank))
    try:
        tmp = f"{response_path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, response_path)
        _remove_quietly(request_path)   # fire once; a re-arm rewrites the request
        _state["fulfilled"] += 1
        _state["last_response_path"] = response_path
        return {"written": True, "path": response_path, "bytes": len(raw),
                "refused": bool(document["refused"])}
    except Exception as exc:  # noqa: BLE001 - a file is never a worker-step fault
        _state["fulfil_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "blockdump fulfil")
        return {"written": False, "reason": _state["last_error"]}


def _remove_quietly(path) -> None:
    """Remove a control file, ignoring its absence. Fire-once bookkeeping."""
    try:  # noqa: SIM105 - no contextlib import for best-effort cleanup
        os.remove(path)
    except OSError:
        pass


def _tick(handler) -> dict:
    """The per-step entry: poll-cheap, and does NOTHING while disarmed.

    The whole cost of being disarmed is one `os.stat` of the arm file per
    `POLL_INTERVAL_SECONDS` per worker: no open, no parse, no byte read, no
    response file. The arm lives in a file because a worker is unreachable over
    the utility channel (`worker_diag`'s reason, in the other direction).
    """
    directory = config.diagnostics_dir()
    if directory is None:
        _state["last_poll_reason"] = f"{config.ENV_DIAG_DIR} is not set"
        return {"written": False, "reason": _state["last_poll_reason"]}
    now = time.monotonic()
    last = _state["last_poll_mono"]
    if isinstance(last, (int, float)) and (now - last) < POLL_INTERVAL_SECONDS:
        _state["poll_suppressed"] += 1
        return {"written": False, "reason": "not due"}
    _state["last_poll_mono"] = now
    _state["polls"] += 1
    armed, reason = _read_arm_file(_arm_path())
    _state["armed"] = armed
    _state["last_poll_reason"] = reason
    if not armed:
        return {"written": False, "reason": reason}
    from . import worker_diag  # noqa: PLC0415 - shared rank/diagnostics
    rank, _source = worker_diag.rank_id()
    if rank is None:
        _state["last_poll_reason"] = "no rank id"
        return {"written": False, "reason": "no rank id"}
    request_path = os.path.join(str(directory), REQUEST_FILE.format(rank=rank))
    if not os.path.exists(request_path):
        _state["last_poll_reason"] = "armed, nothing pending"
        return {"written": False, "reason": "armed, nothing pending"}
    return fulfil_request(handler)


def worker_class():
    module = sys.modules.get(TARGET_MODULE)
    if module is None:
        return None
    return getattr(module, TARGET_CLASS, None)


def _make_get_finished(original):
    def get_finished(self, finished_req_ids):
        result = original(self, finished_req_ids)   # the fork's own work, first
        try:
            _tick(self)
        except Exception as exc:  # noqa: BLE001 - a file never breaks a step
            _state["fallbacks"] += 1
            snapshot._error(exc, "blockdump trigger")
        # The fork's return value is handed back UNTOUCHED and UNCONSUMED: a
        # wrapper that iterated a generator result would free nothing downstream.
        return result
    return get_finished


def install(cls=None) -> dict:
    """Wrap the worker's get_finished. Idempotent, never raises, INERT by default.

    Two independent things make it inert, and both are required:
    `VLLM_E2_TIER_DIAG` must name a directory (otherwise there is no channel to a
    worker at all, and nothing is wrapped or created), and the facility must be
    ARMED over HTTP (otherwise the wrapper's per-step cost is one monotonic
    comparison and, once per poll, one `os.stat`). A fresh boot therefore does no
    dump work even on a box where the diagnostics directory is configured for the
    rank-file writer.

    A sibling layer of this package on the same method (worker_diag's wrapper,
    installed first in production) is composed with, not refused - and the report
    says which methods are nested.
    """
    report = {"installed": False, "attached": [], "replaced": [], "skipped": [],
              "nested": [], "class": None, "reason": None}
    try:
        if config.diagnostics_dir() is None:
            report["reason"] = (f"{config.ENV_DIAG_DIR} is not set: no channel to "
                                "the worker, so nothing is wrapped")
            _state["skip_reason"] = report["reason"]
            _state["skips"] += 1
            return report
        if cls is None:
            cls = worker_class()
        if cls is None:
            report["reason"] = (f"{TARGET_MODULE} is not imported here "
                                f"({TARGET_CLASS} absent)")
            return report
        report["class"] = getattr(cls, "__name__", type(cls).__name__)
        name = METHOD_GET_FINISHED
        original = getattr(cls, name, None)
        if original is None or not callable(original):
            report["reason"] = f"{report['class']}.{name} is missing or not callable"
            return report
        owner = getattr(original, OWNER_ATTR, None)
        if owner is not None and owner != OWNER_TAG:
            report["reason"] = (f"{report['class']}.{name} carries another "
                                "plugin's owner marker")
            return report
        replaced = owner == OWNER_TAG
        if not replaced:
            home = getattr(original, "__module__", None)
            if home != TARGET_MODULE and not str(home or "").startswith(
                    SIBLING_PACKAGE):
                report["reason"] = (f"{name} is neither ours nor the fork's own: "
                                    f"defined in {home!r}")
                return report
            if home != TARGET_MODULE:
                # A layer of this package (worker_diag's writer on the same
                # method): go on top of it, so both facilities run.
                report["nested"].append(name)
            report["attached"].append(name)
        else:
            original = getattr(original, ORIGINAL_ATTR, original)
            report["replaced"].append(name)
        wrapped = _make_get_finished(original)
        setattr(wrapped, OWNER_ATTR, OWNER_TAG)
        setattr(wrapped, ORIGINAL_ATTR, original)
        wrapped.__name__ = name
        wrapped.__doc__ = ("kvtier-dev-plugin: the worker's own get_finished, plus "
                           "a bounded read-only block dump fulfilment "
                           "(VLLM_E2_TIER_DIAG, and only while armed over "
                           "/plugins/kvtier/blockdump). Contained; nothing changes "
                           "what the fork returns; a byte read mutates no pool, "
                           "refcount or LRU.")
        setattr(cls, name, wrapped)
        report["installed"] = True
        _state["installed"] = True
        _state["enabled"] = True
        _state["class"] = report["class"]
        if report["nested"]:
            _state["nested"] = list(report["nested"])
        if report["attached"]:
            _state["installs"] += 1
            if not report["skipped"]:
                _state["skip_reason"] = None
        if report["replaced"]:
            _state["reinstalls"] += 1
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        _state["fallbacks"] += 1
        snapshot._error(exc, "blockdump install")
        report["reason"] = f"install raised: {type(exc).__name__}"[:MAX_STRING_CHARS]
        return report


def ensure_installed(connector=None) -> dict:
    """Retry once the worker's offload module exists. Marker-cheap no-ops first.

    The same handshake-driven retry as `worker_diag.ensure_installed` (the class
    does not exist when `engine_side.register()` runs), and it MUST be called
    after worker_diag's, so the writer's layer is the one already on the method
    when this one composes over it.
    """
    if _state["installed"]:
        return {"installed": True, "reason": "already installed"}
    if _state["class"] is not None:
        return {"installed": False,
                "reason": _state["skip_reason"] or "already attempted"}
    handler = getattr(connector, "worker_handler", None) if connector is not None \
        else None
    return install(type(handler) if handler is not None else None)


def uninstall(cls=None) -> dict:
    report = {"uninstalled": False, "reason": None}
    try:
        if cls is None:
            cls = worker_class()
        if cls is None:
            report["reason"] = f"{TARGET_CLASS} is not imported here"
            return report
        current = getattr(cls, METHOD_GET_FINISHED, None)
        if getattr(current, OWNER_ATTR, None) != OWNER_TAG:
            report["reason"] = "not ours to remove"
            return report
        setattr(cls, METHOD_GET_FINISHED, getattr(current, ORIGINAL_ATTR))
        _state["uninstalls"] += 1
        _state["installed"] = False
        report["uninstalled"] = True
        return report
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "blockdump uninstall")
        report["reason"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        return report


def section() -> dict:
    """The snapshot/status view of this capability, stable in shape, NO file IO.

    This runs on every snapshot (the dashboard polls it), so it reads no files and
    touches no directory: what it reports is THIS process's own state plus the
    configuration. The arm's truth for a worker is the arm file, and the worker's
    own counters are visible only in its own process - `worker_scope` says so.
    """
    directory = config.diagnostics_dir()
    return {
        "capability": capability(),
        "deployed": bool(deployed()),
        "configured": directory is not None,
        "dir": str(directory) if directory is not None else None,
        "armed": bool(is_armed()),
        "armed_state": arm_state(),
        "installed": bool(_state["installed"]),
        "class": _state["class"],
        "nested_in": list(_state["nested"]),
        "installs": int(_state["installs"]),
        "arms": int(_state["arms"]),
        "disarms": int(_state["disarms"]),
        "fulfilled": int(_state["fulfilled"]),
        "requests_read": int(_state["requests_read"]),
        "requests_armed": int(_state["requests_armed"]),
        "refused": int(_state["refused"]),
        "dropped": int(_state["dropped"]),
        "window_refused": int(_state["window_refused"]),
        "polls": int(_state["polls"]),
        "fulfil_errors": int(_state["fulfil_errors"]),
        "fallbacks": int(_state["fallbacks"]),
        "caps": {"max_bytes_per_request": MAX_BYTES_PER_REQUEST,
                 "max_blocks": MAX_BLOCKS,
                 "max_response_bytes": MAX_RESPONSE_BYTES,
                 "request_window_seconds": REQUEST_WINDOW_SECONDS,
                 "max_requests_per_window": MAX_REQUESTS_PER_WINDOW,
                 "poll_interval_seconds": POLL_INTERVAL_SECONDS},
        "disarmed_cost": ("one os.stat of the arm file per poll interval per "
                          "worker; no open, no read, no response file"),
        "arm_transport": (f"the worker learns the arm from {ARM_FILE} in the "
                          "diagnostics directory: a worker is unreachable over "
                          "the utility channel"),
        "worker_scope": ("bytes are read and written in a WORKER process; a "
                         "non-worker process reports its own arm and the "
                         "configured values only"),
        "last_response_path": _clip(_state["last_response_path"]),
        "last_error": _clip(_state["last_error"]),
        "read_only": ("the byte read is host-memory only: no CUDA, no GPU pool, "
                      "no refcount or LRU change"),
    }
