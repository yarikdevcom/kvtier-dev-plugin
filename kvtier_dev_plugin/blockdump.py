"""Capability 6: a read-only, bounded dump of a few PARKED KV blocks' raw bytes.

WHY THIS EXISTS, AND WHY IT IS SHAPED THIS WAY
----------------------------------------------
The arena is the scarce resource (99.9% occupancy, LRU recycling parked spans),
so "can the parked bytes be compressed?" is a real capacity question, and the
honest answer needs REAL parked bf16 KV bytes - synthetic noise compresses to
~1.0x by construction and would give a useless answer. So this module reads a
HANDFUL of real parked blocks' bytes and returns their metadata plus a sha256.

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

  * an HTTP route + EngineCore handler (`dump_for_engine`) that reads the
    SCHEDULER-side geometry (group, dtype, shape, element_size, bytes per block)
    - all of which ARE visible there - and ARMS a bounded dump request; and
  * a worker-side fulfilment (`fulfil_request`, run from a wrapped
    `get_finished`) that reads a hard-capped number of blocks' bytes straight out
    of the pinned host buffers and writes ONE small bounded response file.

THE OPERATOR COLLECTS THE FILE. A raw block is ~11-12.9 MB (PP0/PP1); eight of
them are ~100 MB and are never put on the plaintext utility channel. So the
EngineCore answer is metadata + the request it armed + the response file's path;
the bytes (base64, with a sha256 per block) land in the response file the worker
writes. `payload.bytes_source` says exactly this in every response.

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
    no `_insert_block_hash` on this path at all.

So the safe way to read a block's bytes involves no mutation, and this module
does it that way. If a future fork shape made a byte read require any refcount
or LRU change, the contract is to STOP and report, not to do it.

BOUNDED AND CAPPED, NEVER THE WHOLE ARENA
-----------------------------------------
`MAX_BLOCKS` (8) and `MAX_BYTES` (256 MiB) are hard caps. A request naming more
than `MAX_BLOCKS` is refused with an HONEST error (`refused`) rather than served
partially or as garbage. `MAX_BLOCKS` is per response; there is no "all". The
request file is a single control file, and the response is a single file, so
there is no path here that walks the arena.

THE TOKEN IS NEVER PRINTED OR LOGGED. The route is gated by the same dev token
`/trace` and `/guard` use (`require_dev_state_token`, `hmac.compare_digest` in
`endpoint.py`); this module reads no token and logs none - it never even sees it.

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

OWNER_ATTR = "__kvtier_dev_blockdump_owner__"
OWNER_TAG = "kvtier-dev-plugin/blockdump-1"
ORIGINAL_ATTR = "__kvtier_dev_blockdump_original__"

PROTOCOL_VERSION = 1
ROLE = "blockdump"
WRITER_TAG = "kvtier-dev-plugin/blockdump-1"

# Hard caps. A parked block is ~11-12.9 MB; 8 is "a handful", not "the arena".
MAX_BLOCKS = 8
DEFAULT_BLOCKS = 4
MAX_BYTES = 256 * 1024 * 1024          # 256 MiB of raw bytes per response
MAX_STRING_CHARS = 160

# The two files (names are a fixed, rank-keyed shape under VLLM_E2_TIER_DIAG).
REQUEST_FILE = "blockdump-request-{rank}.json"
RESPONSE_FILE = "blockdump-{rank}.json"


_state: dict = {
    "enabled": False,
    "installed": False,
    "class": None,
    "installs": 0,
    "reinstalls": 0,
    "uninstalls": 0,
    "skips": 0,
    "skip_reason": None,
    "requests_read": 0,
    "fulfilled": 0,
    "fulfil_errors": 0,
    "fallbacks": 0,
    "last_response_path": None,
    "last_error": None,
}


def reset_state_for_tests() -> None:
    _state.update({
        "enabled": False, "installed": False, "class": None, "installs": 0,
        "reinstalls": 0, "uninstalls": 0, "skips": 0, "skip_reason": None,
        "requests_read": 0, "fulfilled": 0, "fulfil_errors": 0,
        "fallbacks": 0, "last_response_path": None, "last_error": None,
    })


def counters() -> dict:
    return dict(_state)


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _clip(text, chars: int = MAX_STRING_CHARS):
    if isinstance(text, str) and len(text) > chars:
        return text[:chars] + "..."
    return text


# --- the pure reader: a parked block's bytes, by host-memory address ---------

def read_block_bytes(tensor, block_id: int) -> dict:
    """One block's raw bytes from a pinned host tensor, WITHOUT mutating anything.

    A pure host-memory read: base + block_id*stride(0)*element_size, ctypes.
    Returns {ok, bytes, sha256, dtype, shape, element_size, bytes_per_block} or
    {ok: False, reason}. Never imports torch, never calls CUDA, never touches a
    pool, a refcount or the LRU (see the module docstring for why).
    """
    if not callable(getattr(tensor, "data_ptr", None)):
        return {"ok": False, "reason": "the region has no data_ptr() to read"}
    if not callable(getattr(tensor, "stride", None)):
        return {"ok": False, "reason": "the region has no stride() to index"}
    if not callable(getattr(tensor, "element_size", None)):
        return {"ok": False, "reason": "the region has no element_size()"}
    shape = tuple(getattr(tensor, "shape", ()) or ())
    if len(shape) < 1:
        return {"ok": False, "reason": "the region is 0-dimensional"}
    if block_id < 0 or block_id >= int(shape[0]):
        return {"ok": False,
                "reason": f"block_id {block_id} is outside the region's "
                          f"{shape[0]} blocks"}
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
    base = _int(tensor.data_ptr())
    if base is None or base <= 0:
        return {"ok": False, "reason": "data_ptr() is not a usable address"}
    offset = block_id * bytes_per_block
    try:
        # The read itself. string_at copies exactly bytes_per_block bytes from
        # an address inside a buffer the fork already allocated and pinned.
        raw = ctypes.string_at(base + offset, bytes_per_block)
    except Exception as exc:  # noqa: BLE001 - an unreadable address is data, not a fault
        return {"ok": False, "reason": f"the host read failed: {type(exc).__name__}"}
    return {
        "ok": True,
        "bytes": raw,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "shape": [int(dim) for dim in shape],
        "element_size": element_size,
        "bytes_per_block": bytes_per_block,
    }


def dump_blocks(worker, group, block_ids) -> dict:
    """Read a HARD-CAPPED set of blocks straight out of the worker's arena.

    `group` selects the `cpu_kv_caches` region by name (`worker.py:235`, the same
    key the DMA backend copies between); `block_ids` is the list to read. Refused
    whole - never partially, never garbage - if it names more than MAX_BLOCKS,
    the group is unknown, or the raw total would exceed MAX_BYTES.
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
                          f"{sorted(str(name) for name in caches)[:32]}"}
    if group not in caches:
        return {"refused": True,
                "reason": f"group {group!r} is not a cpu_kv_caches region"}
    tensor = caches[group]
    blocks = []
    total_bytes = 0
    for block_id in requested:
        index = _int(block_id)
        if index is None:
            return {"refused": True, "reason": f"block id {block_id!r} is not an int"}
        read = read_block_bytes(tensor, index)
        if not read.get("ok"):
            return {"refused": True,
                    "reason": f"block {index} in group {group!r}: "
                              f"{read.get('reason')}"}
        total_bytes += len(read["bytes"])
        if total_bytes > MAX_BYTES:
            return {"refused": True,
                    "reason": f"the response would exceed {MAX_BYTES} bytes "
                              f"(hard cap)"}
        blocks.append({"group": group, "block_id": index,
                       "sha256": read["sha256"], "bytes": len(read["bytes"]),
                       "dtype": read["dtype"], "shape": read["shape"],
                       "element_size": read["element_size"],
                       "bytes_per_block": read["bytes_per_block"],
                       "b64": base64.b64encode(read["bytes"]).decode("ascii")})
    return {"refused": False, "group": group, "blocks": blocks,
            "total_bytes": total_bytes,
            "blocks_requested": len(requested)}


# --- the scheduler-side handler (the HTTP route reaches this) ----------------

def _group_geometry(scheduler, group):
    """That group's geometry, from the scheduler's own CPU config, if visible."""
    if group is None or scheduler is None:
        return None
    manager = getattr(getattr(scheduler, "connector", None), "scheduler_manager",
                      None)
    if manager is None:
        return None
    sizes = getattr(manager, "group_block_sizes", None)
    out = {"num_cpu_blocks": _int(getattr(manager, "num_cpu_blocks", None))}
    if isinstance(sizes, (list, tuple)) and group < len(sizes):
        out["group_block_size"] = _int(sizes[group])
    return out


def dump_for_engine(engine, args=None) -> dict:
    """The EngineCore half: geometry + an ARMED bounded dump request. NEVER bytes.

    It cannot reach worker memory, so it does not pretend to: it validates the
    request against the caps, echoes the scheduler-side geometry, and - when the
    diagnostics directory is configured - writes the one small control file the
    worker fulfils on its next step. The bytes (base64 + per-block sha256) appear
    in the response file, whose path this returns. It never mutates the engine.
    """
    snapshot._state["calls"] += 1
    started = time.perf_counter()
    arguments = args if isinstance(args, dict) else {}
    payload: dict = {"version": PROTOCOL_VERSION, "verb": "blockdump"}
    try:
        raw_ids = arguments.get("block_ids")
        if raw_ids is None and _int(arguments.get("block_id")) is not None:
            raw_ids = [arguments["block_id"]]
        block_ids = list(raw_ids) if isinstance(raw_ids, (list, tuple)) else []
        group = arguments.get("group")
        group = group if isinstance(group, str) else (
            str(group) if group is not None else None)
        too_many = len(block_ids) > MAX_BLOCKS
        payload["requested"] = {"group": group,
                                "block_ids": block_ids[:MAX_BLOCKS + 1],
                                "count": len(block_ids)}
        payload["caps"] = {"max_blocks": MAX_BLOCKS, "max_bytes": MAX_BYTES,
                           "default_blocks": DEFAULT_BLOCKS}
        payload["bytes_source"] = (
            "raw parked bytes live in the WORKER (cpu_kv_caches) and are NOT "
            "returned over the utility channel; the worker writes them (base64, "
            "per-block sha256) to the response file named here. This answer is "
            "metadata + the armed request, and mutates nothing.")
        scheduler = getattr(engine, "scheduler", None)
        payload["group_geometry"] = _group_geometry(scheduler, group if group
                                                    is not None else 0) \
            if group is not None else None

        directory = config.diagnostics_dir()
        payload["request_armed"] = False
        payload["response_file"] = None
        rank = None
        if directory is not None:
            from . import worker_diag  # noqa: PLC0415 - shared rank/diagnostics
            rank, _ = worker_diag.rank_id()
        if too_many:
            payload["refused"] = (f"the request names {len(block_ids)} blocks; "
                                  f"the hard cap is {MAX_BLOCKS}")
        elif not block_ids:
            payload["refused"] = ("no block_ids were named; pass "
                                  '{\"group\": \"<region>\", "block_ids": [...]}')
        elif directory is None:
            payload["refused"] = (f"{config.ENV_DIAG_DIR} is not set: with no "
                                  "diagnostics directory there is no channel to "
                                  "the worker, so nothing was armed")
        elif rank is None:
            payload["refused"] = ("this process resolved no rank id; the request "
                                  "file is rank-keyed, so nothing was armed")
        else:
            request = {"version": PROTOCOL_VERSION, "rank": rank,
                       "group": group, "block_ids": block_ids[:MAX_BLOCKS],
                       "armed_ts": round(time.time(), 6),
                       "epoch": str(snapshot.epoch())}
            path = os.path.join(str(directory), REQUEST_FILE.format(rank=rank))
            try:
                os.makedirs(str(directory), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(request, handle, separators=(",", ":"))
                os.replace(tmp, path)
                payload["request_armed"] = True
                payload["request_file"] = path
                payload["response_file"] = os.path.join(
                    str(directory), RESPONSE_FILE.format(rank=rank))
            except Exception as exc:  # noqa: BLE001 - a failed arm is data, not a 500
                payload["refused"] = f"could not write the request file: " \
                                     f"{type(exc).__name__}"
                snapshot._error(exc, "blockdump arm")
        payload["guard_mode"] = None
        payload["ts"] = round(time.time(), 6)
        payload["epoch"] = str(snapshot.epoch())
        payload["build_us"] = int((time.perf_counter() - started) * 1e6)
        return snapshot.json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler loop
        snapshot._error(exc, "blockdump")
        return {"version": PROTOCOL_VERSION, "verb": "blockdump",
                "epoch": str(snapshot.epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300]}


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


def fulfil_request(handler) -> dict:
    """If this rank has a pending, in-bounds request, answer it in ONE file.

    Runs from a wrapped get_finished, AFTER the fork's own work (the
    worker_diag discipline), so it can never reorder what the scheduler is told.
    Reads the bytes with the pure reader above - no refcount, no LRU, no CUDA -
    writes the response atomically, and removes the request so it fires once.
    """
    directory = config.diagnostics_dir()
    if directory is None:
        return {"written": False, "reason": f"{config.ENV_DIAG_DIR} is unset"}
    from . import worker_diag  # noqa: PLC0415
    rank, _source = worker_diag.rank_id()
    if rank is None:
        return {"written": False, "reason": "no rank id"}
    request, request_path = _read_request(directory, rank)
    if request is None:
        return {"written": False, "reason": "no pending request"}
    _state["requests_read"] += 1
    result = dump_blocks(handler, request.get("group"),
                         request.get("block_ids"))
    document = {"version": PROTOCOL_VERSION, "role": ROLE, "rank": rank,
                "ts": round(time.time(), 6),
                "epoch": str(snapshot.epoch()),
                "writer": WRITER_TAG,
                "refused": result.get("refused", False),
                "reason": result.get("reason"),
                "group": result.get("group"),
                "total_bytes": result.get("total_bytes", 0),
                "blocks": []}
    for block in result.get("blocks", []):
        # The base64 payload is the ONLY place the bytes leave the worker; the
        # per-block sha256 lets the collector verify the download.
        document["blocks"].append(dict(block))
    raw = json.dumps(document, separators=(",", ":"), default=str).encode("utf-8")
    response_path = os.path.join(str(directory), RESPONSE_FILE.format(rank=rank))
    try:
        tmp = f"{response_path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            handle.write(raw)
        os.replace(tmp, response_path)
        try:
            os.remove(request_path)   # fire once; a re-arm rewrites the request
        except OSError:
            pass
        _state["fulfilled"] += 1
        _state["last_response_path"] = response_path
        return {"written": True, "path": response_path, "bytes": len(raw),
                "refused": bool(result.get("refused"))}
    except Exception as exc:  # noqa: BLE001 - a file is never a worker-step fault
        _state["fulfil_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "blockdump fulfil")
        return {"written": False, "reason": _state["last_error"]}


def _make_get_finished(original):
    def get_finished(self, finished_req_ids):
        result = original(self, finished_req_ids)   # the fork's own work, first
        try:
            fulfil_request(self)
        except Exception as exc:  # noqa: BLE001 - a file never breaks a step
            _state["fallbacks"] += 1
            snapshot._error(exc, "blockdump trigger")
        return result
    return get_finished


def worker_class():
    module = sys.modules.get(TARGET_MODULE)
    if module is None:
        return None
    return getattr(module, TARGET_CLASS, None)


def install(cls=None) -> dict:
    """Wrap the worker's get_finished. Idempotent, never raises, INERT by default.

    Inert unless VLLM_E2_TIER_DIAG names a diagnostics directory: with no
    directory there is no channel to the worker, so nothing is wrapped and no
    file is ever created.
    """
    report = {"installed": False, "attached": [], "replaced": [], "skipped": [],
              "class": None, "reason": None}
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
            if home != TARGET_MODULE:
                report["reason"] = (f"{name} is neither ours nor the fork's own: "
                                    f"defined in {home!r}")
                return report
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
                           "(VLLM_E2_TIER_DIAG). Contained; nothing changes what "
                           "the fork returns; a byte read mutates no pool, refcount "
                           "or LRU.")
        setattr(cls, name, wrapped)
        report["installed"] = True
        _state["installed"] = True
        _state["enabled"] = True
        _state["class"] = report["class"]
        if report["attached"]:
            _state["installs"] += 1
        if report["replaced"]:
            _state["reinstalls"] += 1
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin must not break a boot
        _state["fallbacks"] += 1
        snapshot._error(exc, "blockdump install")
        report["reason"] = f"install raised: {type(exc).__name__}"[:MAX_STRING_CHARS]
        return report


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
    """The snapshot/status view of this capability, stable in shape."""
    directory = config.diagnostics_dir()
    return {
        "configured": directory is not None,
        "dir": str(directory) if directory is not None else None,
        "enabled": bool(_state["enabled"]),
        "installed": bool(_state["installed"]),
        "class": _state["class"],
        "installs": int(_state["installs"]),
        "fulfilled": int(_state["fulfilled"]),
        "requests_read": int(_state["requests_read"]),
        "fulfil_errors": int(_state["fulfil_errors"]),
        "caps": {"max_blocks": MAX_BLOCKS, "max_bytes": MAX_BYTES},
        "worker_scope": ("bytes are read and written in a WORKER process; a "
                         "non-worker process reports the configured values only"),
        "last_response_path": _clip(_state["last_response_path"]),
        "read_only": ("the byte read is host-memory only: no CUDA, no GPU pool, "
                      "no refcount or LRU change"),
    }
