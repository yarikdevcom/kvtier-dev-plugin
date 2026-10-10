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
    visible there), and ARMS a bounded request by writing one small rank-agnostic
    pending record (it cannot resolve a rank id - see the relay below);
  * a worker-side fulfilment (`fulfil_request`, run from a wrapped
    `get_finished`) that reads the bytes straight out of the pinned host buffers
    and writes ONE small bounded response file.

THE OPERATOR COLLECTS THE FILE. A raw block is ~11-12.9 MB (PP0/PP1), so bytes are
NEVER put on the plaintext utility channel. The EngineCore answer is metadata plus
the response file's path; the bytes (base64, with a sha256 per block) land in the
response file the worker writes. `payload.bytes_source` says exactly this in every
response.

THE RELAY: WHY THE REQUEST FILE CARRIES NO RANK (the 0.2.7 live failure)
------------------------------------------------------------------------
The first release keyed the request file by rank (`blockdump-request-<rank>.json`)
had the ROUTE resolve the rank with `worker_diag.rank_id()`. On the box that is
structurally impossible: the route is served by the EngineCore/scheduler process,
which is the ONE role with no distributed state to ask - `get_pp_group()` there
raises `AssertionError` (no `init_process_group` in that role), so `rank_id()`
answers `(None, "parallel_state refused to answer: AssertionError")` and every
`dump` refused with it. The process that receives the request cannot name the file,
and the process that could name it never sees the request. `op: arm` worked only
because arming needs no rank. Faked rank resolution hid it: a fake that answers a
question the real process cannot answer turns a bug into a passing test.

So the request crosses as a RANK-AGNOSTIC pending record and the rank is resolved
where rank resolution actually works:

    HTTP route (EngineCore)  ->  blockdump-pending.json      (no rank in the name)
    every worker, per step   ->  adopts it into blockdump-request-<rank>.json
                              ->  fulfils it unchanged
                              ->  writes blockdump-<rank>.json        (the bytes)
                              ->  writes blockdump-receipt-<rank>.json (the verdict)

which keeps the proven one-way channel (EngineCore arms, worker fulfils) and adds
no new direction. The pending record is NOT consumed by a worker - several ranks
must each get their turn - so it lives until the next arming supersedes it or its
TTL lapses; "one in flight" therefore means one pending RECORD, not one per rank.

WHO ANSWERED, AND WHO DID NOT. Under PP2xTP2 four ranks poll the same record and
each writes its own two files, so "the answer" is a set. `GET/POST /blockdump`
(`op: status`) reports it per rank from the RECEIPT files: which ranks answered the
pending seq, which refused and why (a rank whose region set lacks the group answers
"I do not have it", which is an ANSWER, not a failure), which are missing, and
`complete` - which is only ever `True` when this process knows the rank set from
the capacity handshake (`cpu_capacity` records every `(pp_rank, tp_rank)` the
EngineCore collected, the one place a scheduler process can learn ranks it cannot
resolve itself). Unknown rank set, or any rank missing, is reported as such: absent
is never zero and never silently complete. Receipts are read rather than response
files because a response can be 4 MiB and this runs on the EngineCore loop thread;
a receipt is capped at `MAX_RECEIPT_FILE_BYTES` (1 KiB).

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
  * one ask in flight per RECORD - a second `dump` while the live pending record has
    been answered by NO rank is refused (`already pending`), which is the same
    protection the per-rank design had (nothing queues up behind a wedged worker).
    Once at least one rank has answered, the next `dump` supersedes the record and
    reports the per-rank answers it replaced, because a pending record is never
    consumed by a worker (every rank must get its turn).

A `length` is REQUIRED by the `dump` verb: the caller must ask for an explicit
number of bytes. This module has no path that dumps "a whole arena", and no path
that reads a block without being told how far to read.

THE PEER ALLOWLIST IS THE GATE (0.2.68). This route is reached through the ONE
gate this surface has - the ASGI peer address, loopback or
`VLLM_E2_TIER_DEV_ALLOW` - and nothing else. The bearer token that used to be
demanded of `/trace`, `/guard` and this route in every shape is GONE (`see
endpoint.guard_route), which is exactly why the bounds above are the safety: an
explicit `length`, one region, one slot, an arm that expires, a rate window. This
module reads no secret and logs none - it never even sees one.
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

PROTOCOL_VERSION = 3          # 2: ranges; 3: the rank-agnostic pending relay + receipts
ROLE = "blockdump"
RECEIPT_ROLE = "blockdump-receipt"
WRITER_TAG = "kvtier-dev-plugin/blockdump-1"

# --- the caps, and the unit of every one of them ----------------------------
MAX_BLOCKS = 8                    # blocks per request (a handful, never the arena)
MAX_BYTES_PER_REQUEST = 1024 * 1024   # BYTES of raw KV bytes per request (pre-base64)
MAX_BYTES = 256 * 1024 * 1024     # BYTES: the ceiling on ONE block's own size
MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # BYTES: the response file's size cap
MAX_ARM_FILE_BYTES = 4096         # BYTES: the arm file, read by a worker
MAX_PENDING_FILE_BYTES = 4096     # BYTES: the pending record the EngineCore writes
MAX_RECEIPT_FILE_BYTES = 1024     # BYTES: a rank's verdict file, read by the verb
MAX_RANK_FILES = 32               # rank-keyed files listed by one status read
MAX_SERVED_RECORDS = 64           # per-process memory of records already relayed
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

# The three roles of file (names are a fixed shape under VLLM_E2_TIER_DIAG). The
# PENDING one carries NO rank component - that is the whole point of the relay, and
# a rank in this name would put the bug back (`blockdump.py`'s docstring).
PENDING_FILE = "blockdump-pending.json"
REQUEST_FILE = "blockdump-request-{rank}.json"
RESPONSE_FILE = "blockdump-{rank}.json"
RECEIPT_FILE = "blockdump-receipt-{rank}.json"
ARM_FILE = "blockdump-arm.json"
# The name a receipt starts with, for the per-rank listing (no globbing surprise:
# `blockdump-<rank>.json` and `blockdump-request-<rank>.json` share a prefix).
RECEIPT_PREFIX = RECEIPT_FILE.split("{")[0]
FILE_SUFFIX = ".json"

# `op` values of the verb. `dump` is the only one that arms a request.
BLOCKDUMP_OPS = ("status", "arm", "disarm", "dump")

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
    "pending_seq": None,
    "last_pending_file": None,
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
    # the relay (worker side): records this process has already looked at, so one
    # pending record is served ONCE per rank even though it stays on disk for the
    # other ranks to pick up.
    "served": {},
    "adopted": 0,
    "pending_reads": 0,
    "pending_malformed": 0,
    "pending_stat": None,
    "last_rank": None,
    "last_seq": None,
    "receipt_errors": 0,
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
        "requests_armed": 0, "pending_refused": 0, "pending_seq": None,
        "last_pending_file": None,
        "polls": 0, "poll_suppressed": 0, "last_poll_mono": None,
        "requests_read": 0, "fulfilled": 0,
        "fulfil_errors": 0, "refused": 0, "dropped": 0, "fallbacks": 0,
        "last_response_path": None, "last_error": None, "last_poll_reason": None,
        "served": {}, "adopted": 0, "pending_reads": 0,
        "pending_malformed": 0, "pending_stat": None, "last_rank": None,
        "last_seq": None, "receipt_errors": 0,
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


def _pending_record_path(directory=None):
    """THE relay's inbound file. Deliberately rank-free: the process that writes
    it cannot resolve a rank, which is the whole bug this relay exists to fix."""
    directory = config.diagnostics_dir() if directory is None else directory
    if directory is None:
        return None
    return os.path.join(str(directory), PENDING_FILE)


def _next_seq() -> int:
    """The next pending record id, unique for this boot.

    A counter, not a timestamp: a worker needs only to tell "this record" from
    "that record", and a bare wall-clock value could repeat if the operator armed
    twice inside one clock tick. It is NOT a global ordering - a second EngineCore
    boot restarts at 1 - which is why a worker's de-duplication key pairs it with
    the boot epoch and the arming timestamp (`_record_key`).
    """
    last = _state.get("pending_seq")
    seq = int(last) + 1 if isinstance(last, int) and not isinstance(last, bool) else 1
    _state["pending_seq"] = seq
    return seq


def _record_key(record) -> str:
    """A worker's de-duplication key for one pending record.

    `epoch` alone is not enough: `snapshot.epoch()` is per-PROCESS, and a worker
    can never learn the EngineCore's boot id (`worker_diag`'s epoch problem, in the
    other direction), while an operator-pinned epoch is shared across boots. So the
    key is the triple this process actually observed, and a re-armed record (same
    seq, newer `requested_ts`) is correctly treated as new work.
    """
    return (f"{record.get('epoch')}:{record.get('seq')}"
            f":{record.get('requested_ts')}")


def _seen_record(key: str) -> bool:
    return key in dict(_state.get("served") or {})


def _remember_record(key: str) -> None:
    """Remember a record, bounded. Replaces the dict instead of mutating it, so a
    test fixture that saved a shallow copy of `_state` gets its own copy back."""
    seen = dict(_state.get("served") or {})
    seen[key] = round(time.time(), 6)
    while len(seen) > MAX_SERVED_RECORDS:
        seen.pop(next(iter(seen)))
    _state["served"] = seen


def _read_pending_record(path) -> tuple:
    """(record | None, reason): one bounded, validated read of the pending file.

    The file is untrusted input to a worker step - a different process writes it -
    so it is size-capped BEFORE the read and structurally validated after, the
    `_read_arm_file` posture. The caps themselves (`MAX_BLOCKS`, the byte cap) are
    deliberately NOT re-implemented here: an over-cap record is adopted and answered
    with `dump_blocks`' refusal naming the numbers, because silence is a worse
    answer than a refusal the operator can read.
    """
    if path is None:
        return None, f"{config.ENV_DIAG_DIR} is not set: no channel"
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return None, "no pending record"
    except OSError as exc:  # noqa: BLE001 - unreadable is reported, never a fault
        return None, f"the pending record could not be stat'ed: " \
                     f"{type(exc).__name__}"
    if size > MAX_PENDING_FILE_BYTES:
        _state["pending_malformed"] += 1
        return None, (f"the pending record is {size} bytes (cap "
                      f"{MAX_PENDING_FILE_BYTES}); refusing to trust it")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_PENDING_FILE_BYTES + 1)
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a broken record is data, not a fault
        _state["pending_malformed"] += 1
        return None, f"the pending record did not parse: {type(exc).__name__}"
    _state["pending_reads"] += 1
    if not isinstance(document, dict):
        _state["pending_malformed"] += 1
        return None, "the pending record is not an object"
    if document.get("version") != PROTOCOL_VERSION:
        _state["pending_malformed"] += 1
        return None, (f"the pending record is protocol version "
                      f"{document.get('version')!r}, this build speaks "
                      f"{PROTOCOL_VERSION}")
    if _int(document.get("seq")) is None or document.get("seq") <= 0:
        _state["pending_malformed"] += 1
        return None, "the pending record has no positive integer seq"
    for field in ("requested_ts", "expires_ts"):
        value = document.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _state["pending_malformed"] += 1
            return None, f"the pending record has no numeric {field}"
    ids = document.get("block_ids")
    if not isinstance(ids, (list, tuple)) or not ids:
        _state["pending_malformed"] += 1
        return None, "the pending record names no block ids"
    if document.get("group") is None or not isinstance(document.get("group"),
                                                       (str, int)):
        _state["pending_malformed"] += 1
        return None, "the pending record names no group"
    return document, None


def _expected_ranks() -> tuple:
    """(rank ids | None, how that is known). The scheduler CANNOT resolve a rank id
    and yet it is the process that must say who did not answer.

    The one place rank ids arrive in this process is the capacity handshake: the
    EngineCore collects `content: dict[(pp_rank, tp_rank), metadata]`
    (`core.py:288-302`) and `cpu_capacity` records every entry on the scheduler
    side. So the rank SET is knowable here even though no rank IS. With no
    handshake recorded the answer is None - "nobody told me" - and the per-rank view
    then refuses to call anything complete rather than guessing at an empty set.
    """
    try:
        from . import cpu_capacity  # noqa: PLC0415 - the handshake's rank set
        reported = cpu_capacity._state.get("reported")
        ranks = []
        for key in (reported or {}):
            pp, _, tp = str(key).partition(",")
            if pp.isdigit() and tp.isdigit():
                ranks.append(f"PP{pp}_TP{tp}")
        if not ranks:
            return None, ("no capacity handshake has been recorded in this process, "
                          "so the rank set is unknown - completeness cannot be "
                          "claimed, only counted")
        return sorted(set(ranks)), ("the capacity handshake's "
                                    "(pp_rank, tp_rank) keys (cpu_capacity)")
    except Exception as exc:  # noqa: BLE001 - a view is never a verb failure
        return None, f"the handshake view was unreadable: {type(exc).__name__}"


def _receipt_rank(name: str):
    """The rank a receipt FILE NAME claims, or None. Shape-checked by the same rule
    the proxy host's reader applies to the rank files (`worker_diag._valid_rank`)."""
    if not name.startswith(RECEIPT_PREFIX) or not name.endswith(FILE_SUFFIX):
        return None
    rank = name[len(RECEIPT_PREFIX):-len(FILE_SUFFIX)]
    from . import worker_diag  # noqa: PLC0415 - one rank-shape rule, not two
    return rank if worker_diag._valid_rank(rank) else None


def _read_receipt(path, rank, pending) -> dict:
    """One rank's verdict, as data. Bounded, and never raises into the verb."""
    entry = {"rank": rank, "path": path, "seq": None, "refused": None,
             "reason": None, "group": None, "total_bytes": None, "blocks": None,
             "ts": None, "response_file": None, "response_written": None,
             "for_pending_seq": False, "status": "malformed"}
    try:
        size = os.path.getsize(path)
        if size > MAX_RECEIPT_FILE_BYTES:
            entry["reason"] = (f"{size} bytes (cap {MAX_RECEIPT_FILE_BYTES}); "
                               "not read")
            return entry
        with open(path, "rb") as handle:
            raw = handle.read(MAX_RECEIPT_FILE_BYTES + 1)
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - one bad receipt is one bad rank
        entry["reason"] = f"unreadable: {type(exc).__name__}"
        return entry
    if not isinstance(document, dict):
        entry["reason"] = "not an object"
        return entry
    claimed = document.get("rank")
    if claimed != rank:
        # The proxy host's reader calls a file whose rank does not match its name
        # `mislabeled`, and for the same reason: a verdict attributed to the wrong
        # rank is worse than no verdict.
        entry["status"] = "mislabeled"
        entry["reason"] = f"the file name says {rank!r}, the document says " \
                          f"{str(claimed)[:40]!r}"
        return entry
    entry["seq"] = _int(document.get("seq"))
    entry["refused"] = document.get("refused") if isinstance(
        document.get("refused"), bool) else None
    entry["reason"] = _clip(document.get("reason"), 200)
    entry["group"] = _clip(document.get("group"), 40)
    entry["total_bytes"] = _int(document.get("total_bytes"))
    entry["blocks"] = _int(document.get("blocks"))
    ts = document.get("ts")
    entry["ts"] = round(float(ts), 6) if isinstance(ts, (int, float)) \
        and not isinstance(ts, bool) else None
    entry["response_file"] = _clip(document.get("response_file"), 200)
    entry["response_written"] = document.get("response_written") if isinstance(
        document.get("response_written"), bool) else None
    if entry["seq"] is None or entry["refused"] is None:
        entry["status"] = "no seq or verdict: unverifiable"
        return entry
    # A receipt counts for the pending record only if it is stamped AT OR AFTER the
    # arming. Epochs cannot be compared across the relay (they are per-process on
    # both ends), so the wall clock on the shared host is the only ordering left -
    # and a receipt from an earlier boot therefore necessarily predates the ask.
    if pending is not None and entry["seq"] == _int(pending.get("seq")) \
            and entry["ts"] is not None \
            and entry["ts"] >= float(pending.get("requested_ts", 0.0)):
        entry["for_pending_seq"] = True
        entry["status"] = "refused" if entry["refused"] else "answered"
    else:
        entry["status"] = "stale_seq"
    return entry


def per_rank_view(pending=None) -> dict:
    """WHO answered the pending record - per rank, and never silently complete.

    THE RULE: absent is never zero and never complete. A rank with no fresh receipt
    is listed under `missing_ranks`, `complete` stays False, and when this process
    does not know the rank set at all `complete` is None with a reason - a reader
    must never mistake "nobody told me" for "everybody answered".

    This reads files (one bounded <= 1 KiB read per rank, never a 4 MiB response)
    so it belongs to the explicit verb and NEVER to `section()`, which is on the
    dashboard's polled path.
    """
    view = {
        "pending_seq": _int(pending.get("seq")) if isinstance(pending, dict) else None,
        "pending_live": bool(pending is not None and _request_is_live(pending)),
        "pending_file": _pending_record_path(),
        "expected_ranks": None, "expected_ranks_source": None,
        "expected_count": None, "receipts": [], "receipt_files": 0,
        "answered_ranks": [], "answered_with_bytes": [], "refused_ranks": [],
        "missing_ranks": None, "unverifiable_receipts": [],
        "answered_for_pending_seq": 0, "listing_truncated": False,
        "complete": None, "completeness_note": None,
        "note": ("per-rank state, read from the receipt files (a response file can "
                 "be megabytes; a receipt is 1 KiB). A rank that answered "
                 "refused did its job: it said it has no such region."),
    }
    directory = config.diagnostics_dir()
    if directory is None:
        view["completeness_note"] = (f"{config.ENV_DIAG_DIR} is not set: no channel, "
                                     "so no rank can have answered")
        return view
    expected, expected_source = _expected_ranks()
    view["expected_ranks"] = expected
    view["expected_ranks_source"] = expected_source
    view["expected_count"] = len(expected) if expected is not None else None
    try:
        names = sorted(entry.name for entry in os.scandir(str(directory))
                       if entry.is_file()
                       and entry.name.startswith(RECEIPT_PREFIX)
                       and entry.name.endswith(FILE_SUFFIX))
    except OSError as exc:
        view["completeness_note"] = (
            f"the directory could not be listed: {type(exc).__name__}")
        return view
    view["receipt_files"] = len(names)
    if len(names) > MAX_RANK_FILES:
        view["listing_truncated"] = True
        names = names[:MAX_RANK_FILES]
    for name in names:
        rank = _receipt_rank(name)
        if rank is None:
            view["unverifiable_receipts"].append(
                {"file": _clip(name, 120), "status": "mislabeled file name",
                 "reason": "the name does not carry a PP<n>_TP<m> rank"})
            continue
        receipt = _read_receipt(os.path.join(str(directory), name), rank, pending)
        view["receipts"].append(receipt)
        if receipt["status"] == "answered":
            view["answered_ranks"].append(rank)
            view["answered_with_bytes"].append(rank)
        elif receipt["status"] == "refused":
            view["answered_ranks"].append(rank)
            view["refused_ranks"].append(rank)
        else:
            view["unverifiable_receipts"].append(
                {"file": _clip(name, 120), "status": receipt["status"],
                 "reason": receipt["reason"]})
    answered = set(view["answered_ranks"])
    view["answered_for_pending_seq"] = len(answered)
    if expected is None:
        view["complete"] = None
        view["missing_ranks"] = None
        view["completeness_note"] = (
            "the rank set is unknown here, so completeness CANNOT be claimed: "
            f"{view['answered_for_pending_seq']} rank(s) have answered the pending "
            "seq and any other rank may simply have no file yet")
    elif not view["pending_live"]:
        view["missing_ranks"] = sorted(set(expected) - answered)
        view["complete"] = None
        view["completeness_note"] = (
            "no live pending record: these receipts answer an earlier seq, so "
            "completeness is not a meaningful claim")
    else:
        view["missing_ranks"] = sorted(set(expected) - answered)
        view["complete"] = not view["missing_ranks"]
        view["completeness_note"] = (
            "every rank the handshake named has answered this seq" if
            view["complete"] else
            f"{len(view['missing_ranks'])} of {len(expected)} named ranks have not "
            "answered this seq yet (a worker only polls while the engine is "
            "stepping, so 'not yet' is normal, and 'never' means that rank's hook "
            "is not installed)")
    return view


def blockdump_for_engine(engine, args=None) -> dict:
    """THE engine-side handler body for `/blockdump`. Never raises an Exception.

    `op` selects the shape: `status` (the default, and all a GET sends), `arm`,
    `disarm`, `dump`. This half validates against the caps and writes the one small
    rank-agnostic pending record that every worker answers; it CANNOT reach worker
    memory, cannot resolve a rank id, and never pretends to either. `bytes_source`
    and `relay` say so in every payload. `status` additionally reads the per-rank
    receipts (bounded, <= 1 KiB each) - the verb is an explicit call, unlike
    `section()`, which is on the dashboard's polled path and touches no files.
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
            "returned over the utility channel; every worker that polls the pending "
            "record writes them (base64, per-block sha256) to its own "
            f"{RESPONSE_FILE.format(rank='<rank>')}. This answer is metadata + the "
            "armed request, and mutates nothing.")
        payload["relay"] = {
            "pending_file": _pending_record_path(),
            "direction": ("route (EngineCore, NO rank id available here) -> "
                          f"{PENDING_FILE} -> every worker, which resolves ITS OWN "
                          "rank, adopts the record into "
                          f"{REQUEST_FILE.format(rank='<rank>')} and answers "
                          f"{RESPONSE_FILE.format(rank='<rank>')} plus a "
                          f"{RECEIPT_FILE.format(rank='<rank>')}"),
            "note": ("the request file is rank-keyed but the ROUTE is served by the "
                     "process that cannot resolve a rank id, so the request crosses "
                     "rank-agnostically and the rank is named where naming one "
                     "actually works. That split is why the answer is a SET of "
                     "per-rank files, reported under `per_rank`."),
        }
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
            pending, _reason = _read_pending_record(_pending_record_path())
            payload["status"] = counters()
            payload["request_armed"] = False
            payload["pending_request"] = pending
            payload["per_rank"] = per_rank_view(pending)
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


def _arm_request(engine, arguments) -> dict:
    """Validate a dump request and write the one rank-agnostic pending record.

    Refused with the numbers named, never clipped to fit. NO RANK IS RESOLVED HERE,
    and that is not an oversight: this runs in the EngineCore, the one role whose
    `parallel_state` refuses to answer (`get_pp_group()` raises `AssertionError`
    without a distributed init), so an implementation that named a rank-keyed file
    here could never arm anything. The rank is resolved by the worker that picks the
    record up (`_adopt_pending_record`), where it works.
    """
    out: dict = {"request_armed": False, "request_file": None}
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

    expected, expected_source = _expected_ranks()
    out["expected_ranks"] = expected
    out["expected_ranks_source"] = expected_source

    path = _pending_record_path(directory)
    existing, _reason = _read_pending_record(path)
    if existing is not None and _request_is_live(existing):
        # Checked BEFORE the rate window: a refusal that armed nothing must not cost
        # the caller a window slot. The gate is "has ANY rank answered this record
        # yet?", which is what the relay makes both possible and necessary: a
        # pending record is deliberately NOT consumed (every rank needs its turn),
        # so the old "the worker took it, so the channel is free" signal is gone.
        # An unanswered live record means the ask is genuinely in flight - or the
        # worker is wedged, which is the case one-in-flight exists to stop, and it
        # behaves exactly as it did before. Once somebody has answered, replacing the
        # record is the intended next step, and it is REPORTED with the per-rank
        # answers it replaces, so a partial answer is never silently dropped.
        answers = per_rank_view(existing)
        if not answers["answered_for_pending_seq"]:
            _state["pending_refused"] += 1
            _state["refused"] += 1
            out["refused"] = (
                f"a dump request (seq {existing.get('seq')}) is already pending in "
                f"{path} and NO rank has answered it yet; one ask is in flight per "
                "record. Wait for a rank to answer (the engine must be stepping) or "
                "let its TTL expire "
                f"({int(REQUEST_TTL_SECONDS)}s) - collect the per-rank response "
                "files before arming again, they are overwritten per rank")
            out["pending_request_file"] = path
            out["pending_seq"] = _int(existing.get("seq"))
            out["per_rank"] = answers
            return out
        out["superseded_pending_request"] = path
        out["superseded_answers"] = {
            "seq": answers["pending_seq"],
            "answered_ranks": answers["answered_ranks"],
            "refused_ranks": answers["refused_ranks"],
            "missing_ranks": answers["missing_ranks"],
            "note": ("the record this arming replaces, and who had answered it. A "
                     "rank listed as missing there did not answer the PREVIOUS ask "
                     "and will not; a rank that has not polled since will answer "
                     "this seq instead."),
        }

    if path is not None and os.path.exists(path):
        # Stale or malformed: replaceable, and saying so keeps "why did the old one
        # vanish" answerable.
        out["superseded_pending_request"] = path

    if not _window_open(time.monotonic()):
        _state["refused"] += 1
        out["refused"] = (f"the rate window allows {MAX_REQUESTS_PER_WINDOW} "
                          f"dumps per {REQUEST_WINDOW_SECONDS}s; this request was "
                          f"refused (window_refused="
                          f"{int(_state['window_refused'])})")
        return out

    now = time.time()
    seq = _next_seq()
    record = {"version": PROTOCOL_VERSION, "seq": seq, "group": group,
              "block_ids": block_ids[:MAX_BLOCKS],
              "offset_bytes": offset, "length_bytes": length,
              "requested_ts": round(now, 6),
              "armed_ts": round(now, 6),        # the field the pair has always had
              "expires_ts": round(now + REQUEST_TTL_SECONDS, 6),
              "epoch": str(snapshot.epoch()), "writer": WRITER_TAG,
              "ttl_seconds": REQUEST_TTL_SECONDS}
    raw = json.dumps(record, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_PENDING_FILE_BYTES:
        # Named numbers, never a clipped list: a truncated `block_ids` would be a
        # silently different request than the one the operator asked for.
        _state["refused"] += 1
        out["refused"] = (f"the pending record would be {len(raw)} bytes; the cap "
                          f"is {MAX_PENDING_FILE_BYTES} bytes, so nothing was "
                          "armed (shorten the request; nothing here truncates)")
        return out
    try:
        os.makedirs(str(directory), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - a failed arm is data, not a 500
        _state["refused"] += 1
        snapshot._error(exc, "blockdump arm request")
        out["refused"] = (f"could not write the pending record: "
                          f"{type(exc).__name__}")
        return out
    _state["requests_armed"] += 1
    _state["last_pending_file"] = path
    out["request_armed"] = True
    out["request_file"] = path
    out["request_seq"] = seq
    out["response_file"] = None
    out["response_files"] = {
        "pattern": os.path.join(str(directory),
                                RESPONSE_FILE.format(rank="<rank>")),
        "receipt_pattern": os.path.join(str(directory),
                                        RECEIPT_FILE.format(rank="<rank>")),
        "one_per_rank_answering": True,
        "expected_ranks": expected,
        "note": ("every rank that polls this record answers its OWN pair of files, "
                 "so the answer is a set: `GET /blockdump` (op status) reports "
                 "`per_rank` - who answered, who refused and why, and who is "
                 "missing. A missing rank is never reported as no data."),
    }
    out["answer_within_seconds"] = POLL_INTERVAL_SECONDS
    out["how_to_read_the_bytes"] = (
        f"cat {os.path.join(str(directory), RESPONSE_FILE.format(rank='<rank>'))} "
        "- written by EACH worker that polls this record, on its next "
        f"get_finished (within ~{POLL_INTERVAL_SECONDS}s of this arming). THREE "
        "operational facts: get_finished only runs while the engine is STEPPING, "
        "so an idle engine answers at its next step rather than on a timer; each "
        "rank's response file is OVERWRITTEN by the next dump, so collect them (or "
        "rm them) before arming another; and this pending record is NOT removed by "
        "a worker (every rank must get its turn) - the next arming replaces it and "
        "reports in `superseded_answers` which ranks had answered the record it "
        "replaced.")
    out["list_the_region_names"] = (
        "the region names are each worker's cpu_kv_caches keys, visible only "
        "there: arm a dump with any made-up group and every rank's response file "
        "has a reason field naming the regions THAT rank has. That costs one "
        "window slot.")
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


def _write_receipt(rank, request, *, refused, reason, group, total_bytes, blocks,
                   response_path, response_written) -> dict:
    """One rank's VERDICT for one pending seq, in a file small enough to read.

    WHY THIS FILE EXISTS: the response file can legitimately be 4 MiB of base64, and
    the only process that must answer "who has not answered yet?" is the verb, which
    runs on the EngineCore loop thread. Reading four 4 MiB responses to build that
    answer is not an acceptable cost on that thread; reading four <= 1 KiB receipts
    is. The response carries the bytes, the receipt carries the state, and both name
    the same `seq` so a reader can tell which ask a verdict belongs to.
    """
    directory = config.diagnostics_dir()
    report = {"written": False, "path": None, "reason": None}
    if directory is None or rank is None:
        report["reason"] = "no diagnostics directory or no rank"
        return report
    receipt = {"version": PROTOCOL_VERSION, "role": RECEIPT_ROLE,
               "writer": WRITER_TAG, "rank": rank,
               "seq": _int(request.get("seq")) if isinstance(request, dict) else None,
               "epoch": str(snapshot.epoch()), "ts": round(time.time(), 6),
               "refused": bool(refused), "reason": _clip(reason),
               "group": _clip(group, 40), "total_bytes": int(total_bytes or 0),
               "blocks": int(blocks or 0),
               "response_file": response_path,
               "response_written": bool(response_written),
               "pid": os.getpid()}
    try:
        raw = json.dumps(receipt, separators=(",", ":"),
                         default=str).encode("utf-8")
        if len(raw) > MAX_RECEIPT_FILE_BYTES:
            # Trim the prose, never a number: a receipt without a verdict or a seq
            # is unverifiable, which is a worse report than a short reason.
            receipt["reason"] = _clip(_clip(reason), 48)
            raw = json.dumps(receipt, separators=(",", ":"),
                             default=str).encode("utf-8")
        if len(raw) > MAX_RECEIPT_FILE_BYTES:
            _state["receipt_errors"] += 1
            report["reason"] = (f"the receipt would be {len(raw)} bytes (cap "
                                f"{MAX_RECEIPT_FILE_BYTES}); not written")
            return report
        path = os.path.join(str(directory), RECEIPT_FILE.format(rank=rank))
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, path)
        report.update({"written": True, "path": path, "bytes": len(raw)})
        return report
    except Exception as exc:  # noqa: BLE001 - a verdict file never breaks a step
        _state["receipt_errors"] += 1
        snapshot._error(exc, "blockdump receipt")
        report["reason"] = f"{type(exc).__name__}"
        return report


def fulfil_request(handler, request=None) -> dict:
    """If this rank has a live, in-bounds request, answer it: bytes + a verdict.

    Runs from a wrapped get_finished, AFTER the fork's own work (the
    worker_diag discipline), so it can never reorder what the scheduler is told.
    Reads the bytes with the pure reader above - no refcount, no LRU, no CUDA -
    writes the response atomically, and removes THIS RANK's request so it fires once
    per rank. The shared pending record is not touched here; that is
    `_adopt_pending_record`'s business, and every rank gets its own turn at it.

    A dump that RAISES is answered the same way as one that is refused: a response
    file and a receipt with the reason, and the request consumed. Leaving it behind
    would retry a broken tensor on every poll forever, which turns one bad region
    into an endless error loop on a serving box.
    """
    directory = config.diagnostics_dir()
    if directory is None:
        return {"written": False, "reason": f"{config.ENV_DIAG_DIR} is unset"}
    from . import worker_diag  # noqa: PLC0415
    rank, rank_source = worker_diag.rank_id()
    if rank is None:
        return {"written": False, "reason": f"no rank id ({rank_source})"}
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
    if result.get("refused"):
        # A rank that cannot serve the request has still ANSWERED, and its reason is
        # the per-rank fact that makes "2 of 4 ranks hold this region" readable.
        _state["refused"] += 1
    document = {"version": PROTOCOL_VERSION, "role": ROLE, "rank": rank,
                "ts": round(time.time(), 6),
                "epoch": str(snapshot.epoch()),
                "writer": WRITER_TAG,
                # Which ask this answers. A rank-keyed file with no seq could be
                # yesterday's, and "yesterday's answer" is exactly what a byte
                # comparison must never mistake for a fresh one.
                "request_seq": _int(request.get("seq")),
                "scope": ("THIS rank's answer only. Every rank that polls the "
                          "pending record writes its own file, so the answer to one "
                          "ask is a SET; a rank missing here did not answer, which "
                          "is not the same as it having no data."),
                "refused": bool(result.get("refused")),
                "reason": result.get("reason"),
                "group": result.get("group"),
                "total_bytes": int(result.get("total_bytes") or 0),
                "request": {"seq": _int(request.get("seq")),
                            "group": request.get("group"),
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
        # real traffic. Count it, say why, and consume the request. The VERDICT
        # still ships (in the receipt): a rank that silently produced nothing is
        # indistinguishable from a rank that never polled.
        _state["refused"] += 1
        _remove_quietly(request_path)
        reason = (f"the response would be {len(raw)} bytes (cap "
                  f"{MAX_RESPONSE_BYTES}); refused, request consumed")
        receipt = _write_receipt(rank, request, refused=True, reason=reason,
                                 group=document.get("group"), total_bytes=0,
                                 blocks=0, response_path=None,
                                 response_written=False)
        return {"written": False, "reason": reason, "receipt": receipt}
    response_path = os.path.join(str(directory), RESPONSE_FILE.format(rank=rank))
    try:
        tmp = f"{response_path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, response_path)
        _remove_quietly(request_path)   # fire once, FOR THIS RANK
        _state["fulfilled"] += 1
        _state["last_response_path"] = response_path
        receipt = _write_receipt(
            rank, request, refused=bool(document["refused"]),
            reason=document.get("reason"), group=document.get("group"),
            total_bytes=int(document["total_bytes"]),
            blocks=len(document["blocks"]), response_path=response_path,
            response_written=True)
        return {"written": True, "path": response_path, "bytes": len(raw),
                "refused": bool(document["refused"]), "receipt": receipt}
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


def _pending_gate(path):
    """`(mtime_ns, size)` of the pending record, or None: the cheap test that lets an
    ARMED worker skip re-reading a record it has already dealt with. Without it an
    armed worker parses the same few hundred bytes every poll, forever, on a box
    where nobody is watching."""
    if path is None:
        return None
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (int(info.st_mtime_ns), int(info.st_size))


def _adopt_pending_record(directory, rank) -> dict:
    """Relay step 2: the rank-agnostic pending record becomes THIS rank's request.

    THIS is where a rank id is resolved, because this is where one exists: a worker
    has distributed state and the EngineCore does not (the live 0.2.7 failure).
    Adoption copies the record into the rank-keyed request file, which is what lets
    `fulfil_request` stay exactly as it was written - fire once, per rank - and it is
    also why the shared record is NEVER consumed here: every other rank must get its
    turn at the same ask.

    A record is dealt with exactly once per rank (`_record_key`), including one that
    is stale or unreadable: an expired leftover is IGNORED rather than deleted -
    this process reached it too late, which does not entitle it to hide the record
    from a rank that could still serve it.
    """
    report = {"adopted": False, "rank": rank, "seq": None, "reason": None}
    path = _pending_record_path(directory)
    gate = _pending_gate(path)
    if gate is not None and gate == _state.get("pending_stat"):
        report["reason"] = ("the pending record has not changed since this rank "
                            "last relayed it")
        return report
    record, reason = _read_pending_record(path)
    if record is None:
        report["reason"] = reason
        if reason and reason != "no pending record":
            # Unreadable is a verdict too: remember it, so one broken file is one
            # counted event and not one parse (and one log line) per poll.
            _remember_record(f"malformed:{gate}")
            _state["pending_stat"] = gate
        return report
    if not _request_is_live(record):
        _state["dropped"] += 1
        _remember_record(_record_key(record))
        _state["pending_stat"] = gate
        report["reason"] = ("the pending record expired before this rank reached "
                            "it; ignored, not served, and left in place for the "
                            "ranks that can still answer it")
        return report
    key = _record_key(record)
    if _seen_record(key):
        _state["pending_stat"] = gate
        report["reason"] = (f"seq {record.get('seq')} was already relayed by this "
                            "rank (fire once per rank)")
        return report
    request = {"version": PROTOCOL_VERSION, "rank": rank,
               "seq": _int(record.get("seq")), "group": record.get("group"),
               "block_ids": list(record.get("block_ids") or []),
               "offset_bytes": _int(record.get("offset_bytes")) or 0,
               "length_bytes": _int(record.get("length_bytes")),
               "armed_ts": record.get("armed_ts", record.get("requested_ts")),
               "requested_ts": record.get("requested_ts"),
               "expires_ts": record.get("expires_ts"),
               "epoch": record.get("epoch"),
               "relay": {"from": PENDING_FILE, "relayed_by_rank": rank,
                         "relayed_ts": round(time.time(), 6)}}
    request_path = os.path.join(str(directory), REQUEST_FILE.format(rank=rank))
    try:
        raw = json.dumps(request, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_PENDING_FILE_BYTES:
            # Named numbers, and remembered so this is not retried every poll.
            _state["refused"] += 1
            _remember_record(key)
            _state["pending_stat"] = gate
            report["reason"] = (f"the adopted request would be {len(raw)} bytes "
                                f"(cap {MAX_PENDING_FILE_BYTES}); refused")
            return report
        tmp = f"{request_path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, request_path)
    except Exception as exc:  # noqa: BLE001 - an adoption is never a worker-step fault
        _state["fulfil_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "blockdump adopt")
        _state["pending_stat"] = None    # not a verdict: try this record again
        report["reason"] = (f"the pending record could not be adopted: "
                            f"{type(exc).__name__}")
        return report
    _remember_record(key)
    _state["pending_stat"] = gate
    _state["adopted"] += 1
    _state["last_rank"] = rank
    _state["last_seq"] = _int(record.get("seq"))
    report.update({"adopted": True, "seq": _int(record.get("seq")), "reason": None})
    return report


def _tick(handler) -> dict:
    """The per-step entry: poll-cheap, and does NOTHING while disarmed.

    The whole cost of being disarmed is one `os.stat` of the arm file per
    `POLL_INTERVAL_SECONDS` per worker: no open, no parse, no byte read, no
    response file - and no read of the pending record either, because the arm gate
    comes first and returns before it. The arm lives in a file because a worker is
    unreachable over the utility channel (`worker_diag`'s reason, in the other
    direction).

    Armed, one more bounded read joins the poll (the pending record, capped at
    `MAX_PENDING_FILE_BYTES`), skipped outright while its stat does not move.
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
    rank, source = worker_diag.rank_id()
    if rank is None:
        # In the real fork a worker always has distributed state; if this ever fires
        # on the box it says WHY in one line instead of failing silently.
        _state["last_poll_reason"] = f"no rank id ({source})"
        return {"written": False, "reason": f"no rank id ({source})"}
    relay = _adopt_pending_record(directory, rank)
    request_path = os.path.join(str(directory), REQUEST_FILE.format(rank=rank))
    if not relay.get("adopted") and not os.path.exists(request_path):
        # Nothing new, and nothing left over from an adoption that failed to answer.
        _state["last_poll_reason"] = relay.get("reason") or "armed, nothing pending"
        return {"written": False, "reason": _state["last_poll_reason"]}
    report = fulfil_request(handler)
    report["relay"] = relay
    _state["last_poll_reason"] = (relay.get("reason")
                                  or (f"answered rank {rank}'s adoption"
                                      if report.get("written")
                                      else report.get("reason")))
    return report


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
        "adopted": int(_state["adopted"]),
        "pending_reads": int(_state["pending_reads"]),
        "pending_malformed": int(_state["pending_malformed"]),
        "receipt_errors": int(_state["receipt_errors"]),
        "records_served": len(dict(_state.get("served") or {})),
        "last_seq": _state["last_seq"],
        "last_rank": _state["last_rank"],
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
        "request_transport": (
            f"the request crosses the other way as {PENDING_FILE}, with NO rank in "
            "its name: this process cannot resolve a rank id (parallel_state "
            "refuses to answer in the EngineCore role), so each worker resolves its "
            "own and adopts the record into "
            f"{REQUEST_FILE.format(rank='<rank>')} - fire once per rank, so every "
            "rank answers the same ask"),
        "per_rank": ("the answer to one ask is a SET of per-rank files; who "
                     "answered, who refused and who is missing is reported by the "
                     "verb (op status) from the bounded "
                     f"{RECEIPT_FILE.format(rank='<rank>')} files - never here, "
                     "because this runs on a polled path"),
        "worker_scope": ("bytes are read and written in a WORKER process; a "
                         "non-worker process reports its own arm and the "
                         "configured values only"),
        "last_response_path": _clip(_state["last_response_path"]),
        "last_error": _clip(_state["last_error"]),
        "read_only": ("the byte read is host-memory only: no CUDA, no GPU pool, "
                      "no refcount or LRU change"),
    }
