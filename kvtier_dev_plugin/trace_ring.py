"""Capability 3: a bounded, opt-in ring of recent tier/scheduler events.

WHAT IT IS
----------
A preallocated fixed-depth ring of tiny tuples, written from the wrapper in
`guard_mode.py` (the only place that sees the tier's per-step entry points) and
read/cleared through the argument-taking `POST /plugins/kvtier/trace` verb. It
answers "what did the tier do in the last few hundred steps, and in what order"
- the question you cannot ask after the fact from a single snapshot.

WHEN DISARMED IT COSTS ONE BRANCH
---------------------------------
`record()` opens with `if not _state["armed"]: return`: a dict lookup, a pointer
compare, a return. No tuple is built, no argument is even evaluated by the
caller's own code path (the wrapper passes literals and already-live objects).
Better still, `guard_mode` does not INSTALL its wrappers until something is
armed, so in the default state the fork's methods run unwrapped and this module
is not on the hot path at all.

BUDGET
------
Armed: a handful of records per scheduler step (one per tier entry point, plus
one step tick from `build_connector_meta`), each one tuple into a preallocated
slot. Depth is fixed at arm time (`MAX_DEPTH`), never grows, and the oldest
entries are overwritten and counted (`overwritten`), so an armed ring can never
become a memory leak in a long-running engine.

HOUSE RULES: stdlib only (no time module tricks beyond monotonic), no I/O, no
allocation proportional to the batch, `Exception` caught and counted through
`snapshot._error`, never `BaseException`.
"""

from __future__ import annotations

import time

from . import config, route_policy, snapshot

PROTOCOL_VERSION = 1

DEFAULT_DEPTH = 512
MIN_DEPTH = 16
MAX_DEPTH = 4096
DEFAULT_LIMIT = 64
MAX_LIMIT = 512
MAX_STRING_CHARS = 96

# One slot, seven fields, in this order. A tuple (not a dict) so an armed record
# costs one small allocation and reads below stay trivial to convert.
#   (seq, mono_ns, kind, group, n1, n2, text)
SLOT_FIELDS = ("seq", "mono_ns", "kind", "group", "n1", "n2", "text")

_state: dict = {
    "armed": False,
    "depth": DEFAULT_DEPTH,
    "ring": None,          # allocated on first arm, fixed length from then on
    "next": 0,             # write cursor
    "seq": 0,              # monotone sequence, never reset by clear()
    "written": 0,          # records ever written
    "overwritten": 0,      # records lost to the ring wrapping
    "live": 0,             # records currently readable
    "arms": 0,
    "disarms": 0,
    "clears": 0,
    "reads": 0,
    "record_errors": 0,
    "last_kind": None,
}


def _ring() -> list:
    """The preallocated slots. Built once per arm, never resized while armed."""
    if _state["ring"] is None:
        _state["ring"] = [None] * int(_state["depth"])
    return _state["ring"]


def is_armed() -> bool:
    return bool(_state["armed"])


def counters() -> dict:
    """Cheap copy for the snapshot: what the ring is, not what is in it."""
    return {
        "armed": bool(_state["armed"]),
        "depth": int(_state["depth"]),
        "written": int(_state["written"]),
        "overwritten": int(_state["overwritten"]),
        "live": int(_state["live"]),
        "seq": int(_state["seq"]),
        "arms": int(_state["arms"]),
        "disarms": int(_state["disarms"]),
        "clears": int(_state["clears"]),
        "reads": int(_state["reads"]),
        "record_errors": int(_state["record_errors"]),
        "last_kind": _state["last_kind"],
    }


def reset_state_for_tests() -> None:
    """Counters and slots are process-wide, like `snapshot._state`."""
    _state.update({"armed": False, "depth": DEFAULT_DEPTH, "ring": None,
                   "next": 0, "seq": 0, "written": 0, "overwritten": 0,
                   "live": 0, "arms": 0, "disarms": 0, "clears": 0, "reads": 0,
                   "record_errors": 0, "last_kind": None})


# --- the hot path -----------------------------------------------------------

def record(kind: str, group=None, n1=None, n2=None, text=None) -> None:
    """Append one event. Disarmed: one branch, zero allocation. Never raises.

    Callers pass things that already exist (ints, None, a short literal). A
    string is clipped to `MAX_STRING_CHARS` here rather than in the caller, so no
    caller can hand us a megabyte by accident.
    """
    if not _state["armed"]:
        return
    try:
        ring = _ring()
        depth = len(ring)
        if depth == 0:  # cannot happen; a broken arm must not raise into the tier
            return
        _state["seq"] += 1
        index = _state["next"]
        if _state["ring"][index] is not None:
            _state["overwritten"] += 1
        else:
            _state["live"] += 1
        if isinstance(text, str) and len(text) > MAX_STRING_CHARS:
            text = text[:MAX_STRING_CHARS] + "..."
        ring[index] = (_state["seq"], time.monotonic_ns(), kind, group, n1, n2,
                       text)
        _state["next"] = (index + 1) % depth
        _state["written"] += 1
        _state["last_kind"] = kind
    except Exception as exc:  # noqa: BLE001 - a diagnostic ring never reaches the tier
        _state["record_errors"] += 1
        snapshot._error(exc, "trace record")


# --- arming -----------------------------------------------------------------

def _bounded_depth(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_DEPTH
    return max(MIN_DEPTH, min(MAX_DEPTH, value))


def arm(depth=None) -> dict:
    """Start recording. Allocates the slots once; a re-arm with the same depth
    keeps what is already in the ring."""
    wanted = _bounded_depth(depth)
    report = {"armed": True, "requested_depth": depth, "depth": wanted,
              "reallocated": False, "reason": None}
    if wanted != int(_state["depth"]):
        _state["ring"] = [None] * wanted
        _state["depth"] = wanted
        _state["next"] = 0
        _state["live"] = 0
        report["reallocated"] = True
    elif _state["ring"] is None:
        _ring()
    if not _state["armed"]:
        _state["arms"] += 1
        _state["armed"] = True
        report["reason"] = f"recording into a {wanted}-slot ring"
    else:
        report["reason"] = ("already armed" + (" at a new depth"
                                               if report["reallocated"] else ""))
    return report


def disarm() -> dict:
    """Stop recording. The slots are KEPT so a re-arm can still be read."""
    was = bool(_state["armed"])
    _state["armed"] = False
    if was:
        _state["disarms"] += 1
    return {"armed": False, "was_armed": was,
            "reason": "recording stopped; the ring keeps its contents until cleared"}


def clear() -> dict:
    """Forget the contents. Never the arming, never the `seq` (a reader's cursor
    must stay monotone across a clear, or an incremental poll re-reads the world)."""
    depth = int(_state["depth"])
    _state["ring"] = [None] * depth
    _state["next"] = 0
    _state["live"] = 0
    _state["clears"] += 1
    return {"cleared": True, "depth": depth, "seq": int(_state["seq"]),
            "armed": bool(_state["armed"])}


# --- reading ----------------------------------------------------------------

def read(limit=None, after_seq=None) -> dict:
    """The newest `limit` records (optionally only those after `after_seq`).

    Scans backwards from the write cursor, so the cost is O(returned), never
    O(depth). Newest is LAST, so a caller can append to its own log in order.
    """
    wanted = DEFAULT_LIMIT if limit is None else limit
    if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted <= 0:
        wanted = DEFAULT_LIMIT
    wanted = min(wanted, MAX_LIMIT)
    cursor = None
    if after_seq is not None and not isinstance(after_seq, bool) \
            and isinstance(after_seq, int):
        cursor = after_seq
    ring = _state["ring"]
    entries: list = []
    truncated = False
    if ring:
        depth = len(ring)
        live = min(int(_state["live"]), depth)
        position = int(_state["next"])
        scanned = 0
        # At most `live` slots exist; walk back no further than that, and stop as
        # soon as `wanted` entries are in hand (older ones cannot be wanted).
        while scanned < live and len(entries) < wanted:
            position = (position - 1) % depth
            slot = ring[position]
            scanned += 1
            if slot is None:
                continue
            if cursor is not None and slot[0] <= cursor:
                break
            entry = dict(zip(SLOT_FIELDS, slot, strict=True))
            entry["age_ms"] = round((time.monotonic_ns() - slot[1]) / 1e6, 3)
            entries.append(entry)
        entries.reverse()
        truncated = int(_state["live"]) > len(entries)
    _state["reads"] += 1
    out = {"armed": bool(_state["armed"]), "count": len(entries),
           "limit": wanted, "after_seq": cursor, "entries": entries,
           "truncated": truncated,
           "dropped_since_last_read": int(_state["overwritten"])}
    if cursor is not None and int(_state["seq"]) > cursor + len(entries):
        # The gap between the cursor and the newest seq is exactly what the
        # reader missed while it was not looking; say so instead of letting a
        # poller believe it saw a contiguous history.
        out["gap_records"] = int(_state["seq"]) - cursor - len(entries)
    return out


# --- the verb ---------------------------------------------------------------

TRACE_OPS = ("read", "arm", "disarm", "clear", "status")


def _install_observers(engine) -> dict:
    """Put the recorders in place. Function-level imports break the cycle
    (`guard_mode` records into this module, so it cannot be imported at top
    level here)."""
    from . import guard_mode, step_trace  # noqa: PLC0415 - see above
    return {"tier_wrappers": guard_mode.ensure_installed(engine),
            "step_observers": step_trace.install(engine)}


def _remove_observers(engine) -> dict:
    from . import guard_mode, step_trace  # noqa: PLC0415
    reports = {"step_observers": step_trace.uninstall(engine)}
    # Ask EVERY consumer, not just guard mode: the same class attributes feed the
    # guard, this ring and the lookup ledger, and `disarm()` above has already
    # cleared the trace flag, so a non-empty list means someone else is still
    # recording through these wrappers.
    customers = guard_mode.wrapper_customers()
    if customers:
        reports["tier_wrappers"] = {
            "uninstalled": False,
            "reason": ("still needed by: " + ", ".join(customers)
                       + "; the tier wrappers stay in place")}
    else:
        manager = guard_mode.manager_of(engine)
        cls = type(manager) if manager is not None else guard_mode.manager_class()
        reports["tier_wrappers"] = guard_mode.uninstall(cls)
    return reports


def trace_for_engine(engine, args=None) -> dict:
    """THE engine-side handler body for `trace`. Never raises an Exception.

    `op` selects the shape: `read` (the default, and all a GET ever sends),
    `arm` (optionally with `depth`), `disarm`, `clear`, `status`. Arming is what
    INSTALLS the observers (`step_trace` + the tier wrappers), disarming is what
    removes them again - which is how "costs essentially nothing when disarmed"
    is actually true rather than a claim about how cheap an empty branch is.
    """
    snapshot._state["calls"] += 1
    started = time.perf_counter()
    arguments = args if isinstance(args, dict) else {}
    from . import guard_mode  # noqa: PLC0415 - cycle: guard_mode records into us
    # THE SAFE-METHOD BACKSTOP, engine-side. `GET /trace?op=arm` is a 405 at the route;
    # this is the same refusal asked of the verb, so a mutation cannot land because the
    # HTTP half one day forgot to check. Mirrors `disk_store.disk_for_engine`.
    refused = route_policy.safe_method_refusal(config.ROUTE_TRACE, arguments)
    if refused is not None:
        return refused
    try:
        op = arguments.get("op", "read")
        # A non-string `op` is None, which is not in TRACE_OPS: a wrong type is a
        # bad `op`, never a crash and never a silently defaulted read.
        op = op.strip().lower() if isinstance(op, str) else None
        if op not in TRACE_OPS:
            payload = {"op": None, "error": (
                f"unknown op {arguments.get('op')!r}; expected one of "
                f"{list(TRACE_OPS)}")}
        elif op == "arm":
            report = arm(arguments.get("depth"))
            report["install"] = _install_observers(engine)
            payload = {"op": "arm", **report}
        elif op == "disarm":
            report = disarm()
            report["uninstall"] = _remove_observers(engine)
            payload = {"op": "disarm", **report}
        elif op == "clear":
            payload = {"op": "clear", **clear()}
        elif op == "status":
            payload = {"op": "status", "counters": counters()}
        else:
            payload = {"op": "read", **read(arguments.get("limit"),
                                            arguments.get("after_seq"))}
        unknown = sorted(key for key in arguments
                         if key not in ("op", "depth", "limit", "after_seq")
                         and key != route_policy.SAFE_METHOD_KEY)
        payload["version"] = PROTOCOL_VERSION
        payload["epoch"] = str(snapshot.epoch())
        payload["ts"] = round(time.time(), 6)
        payload["guard_mode"] = guard_mode.is_armed()
        payload["trace_armed"] = is_armed()
        if unknown:
            payload["unknown_args"] = unknown
        payload["build_us"] = int((time.perf_counter() - started) * 1e6)
        payload["self"] = snapshot.error_counters()
        return snapshot.json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        snapshot._error(exc, "trace")
        return {"version": PROTOCOL_VERSION, "verb": "trace",
                "epoch": str(snapshot.epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "counters": snapshot.error_counters()}
