"""Capability 8: the LOOKUP LEDGER - the offer/adopt pair, joined per request id.

THE QUESTION THIS EXISTS TO ANSWER
----------------------------------
The live engine logs, repeatedly:

    SimpleCPUOffloadScheduler: update_state_after_alloc called for
    req_id=chatcmpl-... with no external tokens but
    get_num_new_matched_tokens() unexpectedly recorded a pending CPU hit

The fork's `get_num_new_matched_tokens` pins a CPU hit ONLY when `hit_length >
0` and then returns `(hit_length, True)` (manager.py:363-374), so "a pending hit
was recorded" means the connector DID return a positive token count. Yet the
scheduler's next call arrives with `num_external_tokens == 0`, and
`vllm:external_prefix_cache_hits_total` does not move. Between "the connector
says N tokens are in host RAM" and "the scheduler adopts them" the number
becomes zero, and nothing in the current surface says which rule did it or with
what inputs. So record the pair and let the data name the rule:

    local           what the scheduler passed IN  (already block-aligned)
    hit_returned    what the connector gave BACK  (offer)
    adopted         what the scheduler passed to update_state_after_alloc

A triple with `hit_returned > 0 and adopted == 0` is the drop we are hunting.

WHERE THE RECORDER SITS (and why not a third wrapper)
-----------------------------------------------------
NOT its own wrapper. `guard_mode._record_success()` is called by the guard
wrapper for BOTH methods on BOTH the armed and the disarmed path, with the
method name, the arguments and the returned value already in hand - the ledger
is one more reader of values the wrapper already has. Composing a third
independent wrapper here would mean a third install/uninstall/refusal path over
the same two class attributes that `backport.py` (replacements + kill switch)
and `guard_mode.py` (guard) already layer, for zero extra information.

Consequences of sitting on the guard wrapper, all of them deliberate:

  * arming the ledger INSTALLS the guard wrappers (`guard_mode.ensure_installed`)
    when nothing else did, and `guard_mode.install()` is the code that already
    composes over a backport/switch layer, refuses a foreign owner marker, and is
    idempotent. The ledger owns no class attribute at all, so it cannot clobber a
    layer and cannot double-wrap.
  * removal is coordinated: `guard_mode.wrapper_customers()` names every armed
    consumer, and an uninstaller that still has one leaves the wrappers alone
    (the trace ring already had this problem with guard mode; a third consumer
    makes the shared predicate mandatory, not a courtesy).
  * the recording works for a manager constructed LONG BEFORE the wrappers were
    fitted, because the wrapper is a class attribute and Python resolves it per
    call. Nothing here reads `self` except to ask it for `block_size`.

WHAT IT CANNOT SEE, AND SAYS SO
-------------------------------
The wrapper sits at the CONNECTOR, so it sees the scheduler's input AFTER the
scheduler aligned it: `scheduler.py:969-980` computes `partial_tail = local %
block_size` and passes `local - partial_tail`. So `local_mod_block` in a record
is 0 by construction whenever the connector reported a hit at all (the fork's
own gate, manager.py:334-346, returns `(0, False)` for an unaligned input), and
the scheduler's `partial_tail` is therefore INFERRED, never observed. The verdict
reports the RANGE a tail-rule drop implies and whether the tail rule can explain
the record at all, instead of pretending to have measured it. `not_covered`
lists this and the other blind spots (a guard-DECLINED call is not recorded,
nothing before arm is recorded, a record lost to the cap is not in a verdict).

BOUNDED, ALWAYS
---------------
One preallocated list of `cap` slots (16..4096, 256 by default) allocated at arm
and never resized while armed, holding small plain dicts. The join index
(`req_id -> (slot, seq)`) is bounded by the same cap because every entry belongs
to a live record; a hard re-check evicts the oldest and counts it anyway, so a
logic error costs records, never memory. `dropped` is monotonic across the
process and counts records lost to the ring wrapping; `dropped_unjoined` is the
subset that can never be joined again. This runs on the serving loop.

HOUSE RULES: stdlib only (`time`), `except Exception` and never `BaseException`,
every hot-path entry point contained and counted through `snapshot._error`, no
allocation proportional to the batch or the sequence, no I/O, no imports of the
engine. Disarmed, `record_from_call()` is one dict lookup and a return.
"""

from __future__ import annotations

import json
import logging
import time

from . import config, snapshot

# WARNING, for the reason every other module in this package gives: a plugin logger is
# not one of the `vllm.*` loggers the engine configures, so an INFO line from here never
# reaches `/tmp/vllm-serve.log`, which is the only engine log an operator reads. The
# BOOT-ARM line is a refusal notice, and a refusal nobody can read is a silent failure.
_logger = logging.getLogger("kvtier_dev_plugin.lookup_ledger")

PROTOCOL_VERSION = 1

# The two methods whose pair this is. Names as the fork spells them
# (vllm/v1/core/sched/scheduler.py:977 and :1228), because they are also the
# names `guard_mode` hands us.
METHOD_LOOKUP = "get_num_new_matched_tokens"
METHOD_ADOPT = "update_state_after_alloc"
LEDGER_METHODS = (METHOD_LOOKUP, METHOD_ADOPT)

DEFAULT_CAP = 256
MIN_CAP = 16
MAX_CAP = 4096
DEFAULT_PREVIEW = 6
MAX_PREVIEW = 32
DEFAULT_LIMIT = 32
MAX_LIMIT = 256
MAX_VERDICT_SCAN = 4096          # never more than one full ring
MAX_VERDICT_SAMPLES = 12
MAX_REQ_ID_CHARS = 96

KIND_LOOKUP = "lookup"
KIND_ADOPT_ONLY = "adopt_only"

# The closed vocabulary of `verdict()`. A reader must be able to enumerate the
# buckets from the payload, and every record lands in exactly one of them.
CLASSIFICATIONS = ("adopted_full", "adopted_partial", "adopted_more_than_offered",
                   "dropped_to_zero", "no_host_copy", "no_verdict_from_connector",
                   "lookup_without_adopt", "adopt_without_lookup", "unparsable")

# What the fork does with each offer (scheduler.py:982-1010), quoted so the
# verdict's reasoning travels with the payload instead of living in a terminal.
SCHEDULER_ADOPTION_RULE = (
    "if partial_tail and ext_tokens > partial_tail: adopt ext_tokens; "
    "elif partial_tail: force 0; else: adopt ext_tokens "
    "(partial_tail = local % block_size, and local is the UNALIGNED local hit, "
    "which the connector never sees)")

_state: dict = {
    "armed": False,
    "cap": DEFAULT_CAP,
    "ring": None,        # allocated at arm, fixed length from then on
    "next": 0,           # write cursor
    "seq": 0,            # monotonic, never reset by clear(): a reader's cursor
    "records": 0,        # records ever written
    "live": 0,           # records currently readable
    "dropped": 0,        # monotonic: records lost because the ring wrapped
    "dropped_unjoined": 0,   # ... of which lookups that can never be joined
    "pending_now": 0,    # lookups waiting for their adopt right now
    "pending_evicted": 0,    # the can't-happen hard-bound eviction, counted
    "lookups": 0,
    "adopts": 0,
    "joins": 0,
    "relookups": 0,          # a second lookup for a req_id still pending
    "adopts_without_lookup": 0,
    "join_refusals": 0,      # a slot recycled under a pending entry
    "arms": 0,
    "disarms": 0,
    "clears": 0,
    "reads": 0,
    "verdicts": 0,
    "record_errors": 0,
    "last_error": None,
    "install": None,
    "uninstall": None,
    # THE BOOT ARM (0.2.66). `boot_arms` counts boots that armed the ring,
    # `boot_arm_refusals` counts boots that could not and BOOTS ANYWAY (this is
    # observability: a missing diagnostic is never a tier that declined to serve, so
    # neither number ever joins `boot["refusals"]`). `boot_verdict` is the last
    # verdict THIS PROCESS reached, and `armed_at` is when the ring actually started
    # recording - the field that distinguishes "armed from the first request" from
    # "someone POSTed arm in the middle of the boot", which is the whole lesson of the
    # incident that made this a boot step.
    "boot_arms": 0,
    "boot_arm_refusals": 0,
    "boot_verdict": None,
    "boot_reason": None,
    "armed_at": None,
}

# THE BOOT ARM'S CLOSED VERDICT VOCABULARY (0.2.66), rendered as
# `LOOKUP_LEDGER[verdict]` on the boot line. Bracketed and never `key=value` for the
# reason `mamba_anchor.boot_token` gives: the restart verifier in llm-api pulls its
# PASS criterion with `grep -oE "tier=[A-Za-z_]+" | tail -1`, and a token that happens
# to contain an `=` becomes a candidate match for a grep that was never meant to see it.
VERDICT_ARMED = "armed"                # the ring is recording from this instant
VERDICT_ENV_REFUSED = "env-refused"    # the operator asked for no ring, by name
VERDICT_REFUSED = "refused"            # something stopped the arm; the boot is intact
VERDICT_INERT = "inert"                # an inert boot installs nothing, ring included
VERDICT_NOT_RUN = "not_run"            # this boot never reached the step
BOOT_VERDICTS = (VERDICT_ARMED, VERDICT_ENV_REFUSED, VERDICT_REFUSED, VERDICT_INERT,
                 VERDICT_NOT_RUN)

# req_id -> (slot, seq of the record waiting for its adopt). Bounded by `cap`:
# see the module docstring. A dict, not a deque, because an adopt for ANY older
# request must find its lookup, which is the whole reason this is not a stack.
_pending: dict = {}


def reset_state_for_tests() -> None:
    """Process-wide state, like `snapshot._state` and `trace_ring._state`."""
    _state.update({
        "armed": False, "cap": DEFAULT_CAP, "ring": None, "next": 0, "seq": 0,
        "records": 0, "live": 0, "dropped": 0, "dropped_unjoined": 0,
        "pending_now": 0, "pending_evicted": 0, "lookups": 0, "adopts": 0,
        "joins": 0, "relookups": 0, "adopts_without_lookup": 0,
        "join_refusals": 0, "arms": 0, "disarms": 0, "clears": 0, "reads": 0,
        "verdicts": 0, "record_errors": 0, "last_error": None,
        "install": None, "uninstall": None,
        # THE BOOT ARM'S STATE TOO (0.2.66). A reset that left `boot_verdict` behind
        # would let one test read another test's boot verdict as its own, and
        # `armed_at` with it - which is the field a reader uses to decide whether the
        # ring was recording when the request he is investigating arrived.
        "boot_arms": 0, "boot_arm_refusals": 0, "boot_verdict": None,
        "boot_reason": None, "armed_at": None,
    })
    _pending.clear()


def is_armed() -> bool:
    return bool(_state["armed"])


def counters() -> dict:
    """Cheap copy for the snapshot: what the ledger IS, not what is in it."""
    return {
        "armed": bool(_state["armed"]),
        "cap": int(_state["cap"]),
        "records": int(_state["records"]),
        "live": int(_state["live"]),
        "seq": int(_state["seq"]),
        "dropped": int(_state["dropped"]),
        "dropped_unjoined": int(_state["dropped_unjoined"]),
        "pending_now": int(_state["pending_now"]),
        "pending_evicted": int(_state["pending_evicted"]),
        "lookups": int(_state["lookups"]),
        "adopts": int(_state["adopts"]),
        "joins": int(_state["joins"]),
        "relookups": int(_state["relookups"]),
        "adopts_without_lookup": int(_state["adopts_without_lookup"]),
        "join_refusals": int(_state["join_refusals"]),
        "arms": int(_state["arms"]),
        "disarms": int(_state["disarms"]),
        "clears": int(_state["clears"]),
        "reads": int(_state["reads"]),
        "verdicts": int(_state["verdicts"]),
        "record_errors": int(_state["record_errors"]),
        "last_error": _state["last_error"],
        # THE BOOT ARM (0.2.66): was the ring recording from this boot's first
        # request, and if not, which of the five verdicts says why.
        "boot_arms": int(_state["boot_arms"]),
        "boot_arm_refusals": int(_state["boot_arm_refusals"]),
        "boot_verdict": (_state["boot_verdict"]
                         if _state["boot_verdict"] in BOOT_VERDICTS else None),
        "boot_reason": _state["boot_reason"],
        "armed_at": _state["armed_at"],
    }


# --- internals ---------------------------------------------------------------

def _ring() -> list:
    if _state["ring"] is None:
        _state["ring"] = [None] * int(_state["cap"])
    return _state["ring"]


def _int_or_none(value) -> int | None:
    """A real int, or None. A bool is NOT an int here (True != 1 for a count)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _clip(value, chars: int = MAX_REQ_ID_CHARS):
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        # An int id is not what the fork uses, but str() of it is cheap and keeps
        # the value; only for an UNKNOWN object type do we fall back to the name.
        return str(value)
    if not isinstance(value, str):
        # NEVER str() of an arbitrary object on the serving loop: a fork type
        # with a heavy __repr__ would be paid per request. A name is enough.
        value = type(value).__name__
    if len(value) > chars:
        return value[:chars] + "..."
    return value


def _req_id(request) -> str:
    """The request id, bounded, never raising. Unknown is a literal, not None."""
    try:
        return _clip(getattr(request, "request_id", None)) or "<unknown>"
    except Exception as exc:  # noqa: BLE001 - a request we cannot name is still a request
        _state["record_errors"] += 1
        snapshot._error(exc, "lookup ledger req_id")
        return "<unknown>"


def _release_slot(index: int) -> None:
    """Account for the record the cursor is about to overwrite."""
    ring = _ring()
    old = ring[index]
    if old is None:
        return
    _state["dropped"] += 1
    if old["kind"] == KIND_LOOKUP and old["adopted"] is None:
        _state["dropped_unjoined"] += 1
        held = _pending.get(old["req"])
        if held is not None and held[0] == index and held[1] == old["seq"]:
            del _pending[old["req"]]
            _state["pending_now"] = len(_pending)
    _state["live"] = max(0, int(_state["live"]) - 1)


def _take_slot() -> int:
    ring = _ring()
    depth = len(ring)
    if depth == 0:            # a broken arm must never raise into the step loop
        return -1
    index = int(_state["next"])
    _release_slot(index)
    _state["next"] = (index + 1) % depth
    _state["live"] += 1
    return index


def _write(record: dict) -> dict | None:
    index = _take_slot()
    if index < 0:
        return None
    record["seq"] = int(_state["seq"])
    _state["seq"] += 1
    _state["records"] += 1
    ring = _ring()
    ring[index] = record
    record["_slot"] = index
    return record


# --- the hot path ------------------------------------------------------------

def record_lookup(req_id, local, hit_returned, load_kv_async=None,
                  block_size=None, hash_block_size=None, attribution=None) -> None:
    """One offer: what came in, what went back. Never raises; disarmed: 1 branch.

    `hit_returned` is None when the connector gave no verdict at all (the fork's
    `(None, False)`, which makes the scheduler SKIP the request - no adopt ever
    follows), and 0 when it simply found no host copy.

    `attribution` is the `miss_attribution.attribute()` dict for this same call
    (capability 9). It rides on THIS record rather than a second ledger because
    the question it answers - "why did THIS request miss" - is per request id, and
    this ring is already keyed by request id.
    """
    if not _state["armed"]:
        return
    try:
        identity = _clip(req_id) or "<unknown>"
        aligned = _int_or_none(local)
        block = _int_or_none(block_size)
        record = _write({
            "seq": 0,
            "req": identity,
            "kind": KIND_LOOKUP,
            "local": aligned,
            "hit_returned": _int_or_none(hit_returned),
            "load_kv_async": bool(load_kv_async) if load_kv_async is not None
            else None,
            "adopted": None,
            "local_mod_block": (aligned % block)
            if (aligned is not None and block) else None,
            "block_size": block,
            "hash_block_size": _int_or_none(hash_block_size),
            "attribution": attribution if isinstance(attribution, dict) else None,
            "mono_ns": time.monotonic_ns(),
        })
        if record is None:
            return
        _state["lookups"] += 1
        if _pending.pop(identity, None) is not None:
            # A second offer for a request still waiting: the fork does this on
            # purpose (manager.py:326-329 releases a stale pin "e.g. retry after
            # a failed allocate_slots"), and the adopt pairs with the LATEST one.
            _state["relookups"] += 1
        _pending[identity] = (record["_slot"], record["seq"])
        _state["pending_now"] = len(_pending)
        if len(_pending) > int(_state["cap"]):
            # Cannot happen while every entry owns a live slot; paid for only if
            # it ever does, and then it costs records, not memory.
            oldest = min(_pending.items(), key=lambda item: item[1][1])[0]
            _pending.pop(oldest, None)
            _state["pending_evicted"] += 1
            _state["pending_now"] = len(_pending)
    except Exception as exc:  # noqa: BLE001 - a ledger never reaches the scheduler
        _state["record_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:120]
        snapshot._error(exc, "lookup ledger record_lookup")


def record_adopt(req_id, num_external_tokens, block_size=None) -> None:
    """The scheduler's decision for a request. Never raises.

    Joined by REQUEST ID, not by call order: the scheduler walks a queue, so any
    number of other requests' lookups and adopts land in between, and a request
    can be skipped so that no adopt ever arrives.
    """
    if not _state["armed"]:
        return
    try:
        identity = _clip(req_id) or "<unknown>"
        adopted = _int_or_none(num_external_tokens)
        _state["adopts"] += 1
        held = _pending.pop(identity, None)
        _state["pending_now"] = len(_pending)
        if held is not None:
            slot, seq = held
            ring = _ring()
            record = ring[slot] if 0 <= slot < len(ring) else None
            if (record is not None and record.get("_slot") == slot
                    and record.get("seq") == seq and record["kind"] == KIND_LOOKUP
                    and record["adopted"] is None):
                record["adopted"] = adopted
                if block_size is not None and record["block_size"] is None:
                    record["block_size"] = _int_or_none(block_size)
                _state["joins"] += 1
                return
            # The slot was recycled under the pending entry: say so, and fall
            # through so the adopt is still counted as one without a lookup.
            _state["join_refusals"] += 1
        # An adopt with no lookup of ours is EXPECTED noise, not an anomaly:
        # scheduler.py:1046-1060 skips the lookup entirely for a WAITING request
        # whose num_computed_tokens is already > 0 (an async load's second pass,
        # a resumed streaming session, a post-preemption re-entry) and then calls
        # update_state_after_alloc with 0 anyway.
        _state["adopts_without_lookup"] += 1
        _write({"seq": 0, "req": identity, "kind": KIND_ADOPT_ONLY,
                "local": None, "hit_returned": None, "load_kv_async": None,
                "adopted": adopted, "local_mod_block": None,
                "block_size": _int_or_none(block_size),
                "hash_block_size": None, "mono_ns": time.monotonic_ns()})
    except Exception as exc:  # noqa: BLE001 - a ledger never reaches the scheduler
        _state["record_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:120]
        snapshot._error(exc, "lookup ledger record_adopt")


def record_from_call(method, args, result, manager=None) -> None:
    """THE hook `guard_mode._record_success()` calls. Contained; never raises.

    Arguments are read POSITIONALLY because that is how both callers spell them:
    the scheduler (`scheduler.py:977`, `:1228`) and the connector's forwarders
    (`simple_cpu_offload_connector.py:242-262`) both pass positionally. A keyword
    call would arrive as an empty `args` and record `local: null` /
    `adopted: null`, which is a visible hole rather than a wrong number.
    """
    if not _state["armed"]:
        return
    try:
        arguments = args or ()
        block_size = getattr(manager, "block_size", None) if manager else None
        if method == METHOD_LOOKUP:
            request = arguments[0] if len(arguments) > 0 else None
            local = arguments[1] if len(arguments) > 1 else None
            hit = result[0] if isinstance(result, tuple) and len(result) > 0 \
                else None
            async_flag = result[1] if isinstance(result, tuple) and len(result) > 1 \
                else None
            record_lookup(_req_id(request), local, hit, async_flag, block_size,
                          getattr(manager, "hash_block_size", None) if manager
                          else None,
                          attribution=_attribute_safely(request, local, result,
                                                        manager))
        elif method == METHOD_ADOPT:
            request = arguments[0] if len(arguments) > 0 else None
            adopted = arguments[2] if len(arguments) > 2 else None
            record_adopt(_req_id(request), adopted, block_size)
    except Exception as exc:  # noqa: BLE001 - the hook is not the scheduler's problem
        _state["record_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:120]
        snapshot._error(exc, "lookup ledger hook")


def _attribute_safely(request, local, result, manager):
    """`miss_attribution.observe_connector()`, contained twice over.

    The attribution is an ADDITION to a record that is written anyway; nothing it
    can do is worth losing the ledger's own row, so the whole call is wrapped and
    a failure lands in the ledger's error counter with the record still written.
    """
    try:
        from . import miss_attribution  # noqa: PLC0415 - cycle: it reads our ring
        return miss_attribution.observe_connector(request, local, result, manager)
    except Exception as exc:  # noqa: BLE001 - the ledger outlives the attribution
        _state["record_errors"] += 1
        snapshot._error(exc, "lookup ledger attribution hook")
        return None


# --- reading -----------------------------------------------------------------

def _preview(record: dict, strip_mono: bool = True) -> dict:
    """A copy without the internal slot key, with an age instead of a stamp."""
    out = {key: value for key, value in record.items() if key != "_slot"}
    if strip_mono:
        mono = out.pop("mono_ns", None)
        if isinstance(mono, int):
            out["age_ms"] = round((time.monotonic_ns() - mono) / 1e6, 3)
    return out


def _live_records() -> list:
    ring = _ring()
    if not ring:
        return []
    depth = len(ring)
    live = min(int(_state["live"]), depth)
    position = int(_state["next"])
    out: list = []
    scanned = 0
    while scanned < live:
        position = (position - 1) % depth
        slot = ring[position]
        scanned += 1
        if slot is not None:
            out.append(slot)
    out.reverse()           # oldest first, so a reader appends in order
    return out


def read(limit=None) -> dict:
    """The newest `limit` records, oldest first. Bounded by MAX_LIMIT."""
    wanted = DEFAULT_LIMIT if limit is None else limit
    if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted <= 0:
        wanted = DEFAULT_LIMIT
    wanted = min(wanted, MAX_LIMIT)
    live = _live_records()
    window = live[-wanted:]
    _state["reads"] += 1
    return {"armed": bool(_state["armed"]), "count": len(window),
            "limit": wanted, "live": int(_state["live"]),
            "entries": [_preview(record) for record in window],
            "truncated": len(live) > len(window),
            "dropped": int(_state["dropped"])}


# --- per-request miss attribution (capability 9) -----------------------------

MAX_REQUESTS_BYTES = 200_000        # the whole payload's budget, refused not cut
MAX_ATTRIBUTED_LIMIT = 128          # records per `requests` read, at most


def _fit_count(entries: list, budget: int, overhead: int = 512) -> int:
    """How many of `entries` fit in `budget`, from the largest one's size."""
    if not entries:
        return 0
    largest = max(len(json.dumps(entry, separators=(",", ":"), default=str))
                  for entry in entries)
    return max(0, (budget - overhead) // max(1, largest))


def requests(limit=None, verdict: str | None = None) -> dict:
    """The attributed lookup records, oldest first, with the verdicts attached.

    Reads the SAME ring as `read()` - one ledger, one join, one bound - and adds
    nothing to it. Two bounds protect the response: `MAX_ATTRIBUTED_LIMIT` records
    and `MAX_REQUESTS_BYTES` bytes. When the caller's `limit` cannot fit, this
    REFUSES with the largest limit that would, because a silently shortened list
    of "the requests that missed" is how a diagnostic starts lying.
    """
    wanted = DEFAULT_LIMIT if limit is None else limit
    if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted <= 0:
        wanted = DEFAULT_LIMIT
    wanted = min(wanted, MAX_ATTRIBUTED_LIMIT)
    live = [record for record in _live_records() if record["kind"] == KIND_LOOKUP]
    if isinstance(verdict, str) and verdict:
        live = [record for record in live
                if str((record.get("attribution") or {}).get("verdict",
                                                              "")) == verdict]
    window = live[-wanted:]
    preview = [_preview(record) for record in window]
    fit = _fit_count(preview, MAX_REQUESTS_BYTES)
    _state["reads"] += 1
    if preview and fit < len(preview):
        return {"armed": bool(_state["armed"]), "refused": (
            f"{len(preview)} attributed records do not fit the "
            f"{MAX_REQUESTS_BYTES}-byte budget; the largest limit that does is "
            f"{fit}"), "limit_requested": wanted, "would_fit": fit,
            "entries": [], "count": 0, "live": int(_state["live"]),
            "dropped": int(_state["dropped"])}
    return {"armed": bool(_state["armed"]), "count": len(preview),
            "limit": wanted, "live": int(_state["live"]),
            "entries": preview,
            "truncated": len(live) > len(preview),
            "dropped": int(_state["dropped"]),
            "estimated_bytes": sum(len(json.dumps(entry, separators=(",", ":"),
                                                  default=str))
                                   for entry in preview),
            "budget_bytes": MAX_REQUESTS_BYTES,
            "verdict_filter": verdict if isinstance(verdict, str) else None,
            "units": "see snapshot.miss_attribution.units: every *_tokens field "
                     "is tokens, *_spans is spans, *_hash is hash blocks"}


# --- the verdict -------------------------------------------------------------

def _dropped_detail(record: dict) -> dict:
    """What a hit>0 / adopted==0 record can and cannot tell us about the rule.

    The fork's rule (scheduler.py:996-1010) drops the offer when
    `partial_tail >= ext_tokens`, where `partial_tail = local % block_size` and
    `local` is the scheduler's UNALIGNED local hit. Two things follow, and both
    are computed here because the connector only ever sees `local - partial_tail`:

      * the rule can only fire when `0 < ext_tokens < block_size` (a tail is by
        construction smaller than a block), so a dropped hit AT OR ABOVE the
        block size is NOT the tail rule and something else zeroed it;
      * the true local hit must have lived in
        `[local + hit, local + block_size - 1]`, which is a range, not a number.
    """
    hit = record.get("hit_returned")
    local = record.get("local")
    block = record.get("block_size")
    can = None
    implied = None
    if isinstance(hit, int) and isinstance(block, int) and block > 0:
        can = hit < block
        # The range is only emitted when it is NOT empty: `[52000, 35199]` would
        # be an inverted interval pretending to be data. An empty range is exactly
        # the refutation, and `tail_rule_can_explain: false` already says it.
        if can and isinstance(local, int):
            implied = [local + max(hit, 1), local + block - 1]
    return {
        "seq": record.get("seq"),
        "req": record.get("req"),
        "hit_returned": hit,
        "local_as_passed": local,
        "block_size": block,
        "local_mod_block": record.get("local_mod_block"),
        "partial_tail_observed": None,   # NOT visible at the connector: never fake it
        "min_partial_tail_the_rule_needs": hit,
        "implied_true_local_range": implied,
        "tail_rule_can_explain": can,
        "hash_block_size": record.get("hash_block_size"),
        "hit_mod_hash_block": (hit % record["hash_block_size"])
        if isinstance(hit, int) and isinstance(record.get("hash_block_size"), int)
        and record["hash_block_size"] > 0 else None,
    }


def _classify(record: dict) -> tuple[str, dict | None]:
    kind = record.get("kind")
    if kind == KIND_ADOPT_ONLY:
        return "adopt_without_lookup", None
    hit = record.get("hit_returned")
    adopted = record.get("adopted")
    if hit is None:
        # (None, False): "I cannot tell you yet". The scheduler pops the request
        # and prepends it (scheduler.py:982-988) - the request is SKIPPED, so no
        # adopt follows and this is not a drop.
        return "no_verdict_from_connector", None
    if adopted is None:
        return "lookup_without_adopt", None
    if hit == 0:
        return "no_host_copy", None
    if adopted == hit:
        return "adopted_full", None
    if adopted == 0:
        return "dropped_to_zero", _dropped_detail(record)
    if 0 < adopted < hit:
        return "adopted_partial", None
    if adopted > hit:
        # Impossible under the fork's rules; its own bucket so it is never
        # quietly folded into "adopted_partial".
        return "adopted_more_than_offered", None
    return "unparsable", None


def verdict(limit=None) -> dict:
    """Classify every readable record. Bounded, contained, and honest about gaps.

    The bucket that matters is `dropped_to_zero`: `hit_returned > 0` and the
    scheduler adopted 0. Its detail block carries `local % block_size` AND the
    derived range, so the partial-tail rule can be CONFIRMED (a hit below one
    block, consistent with a tail at least as long) or REFUTED (a hit at or above
    one block: that rule cannot produce this) straight from the payload.
    """
    wanted = limit if isinstance(limit, int) and not isinstance(limit, bool) \
        else MAX_VERDICT_SCAN
    wanted = max(1, min(MAX_VERDICT_SCAN, int(_state["cap"]), wanted))
    live = _live_records()
    scanned = live[-wanted:]
    counts = {name: 0 for name in CLASSIFICATIONS}
    samples: list = []
    for record in scanned:
        try:
            name, detail = _classify(record)
        except Exception as exc:  # noqa: BLE001 - one bad record is not a bad verdict
            snapshot._error(exc, "lookup ledger classify")
            name, detail = "unparsable", None
        counts[name] = counts.get(name, 0) + 1
        if name in ("dropped_to_zero", "adopted_more_than_offered") \
                and len(samples) < MAX_VERDICT_SAMPLES:
            samples.append(detail if detail is not None else _preview(record))
        elif name == "adopt_without_lookup" and len(samples) < MAX_VERDICT_SAMPLES \
                and counts[name] <= 3:
            samples.append(_preview(record))
    _state["verdicts"] += 1
    return {"armed": bool(_state["armed"]),
            "records_scanned": len(scanned),
            "records_live": int(_state["live"]),
            "counts": counts,
            "dropped_examples": samples,
            "rule_under_test": SCHEDULER_ADOPTION_RULE,
            "truncated": len(live) > len(scanned),
            "not_covered": list(NOT_COVERED),
            "dropped": int(_state["dropped"]),
            "pending_now": int(_state["pending_now"])}


NOT_COVERED = (
    "the scheduler's pre-alignment local hit and therefore partial_tail: the "
    "connector is handed `local - local % block_size` (scheduler.py:969-980), so "
    "local_mod_block is 0 by construction and the tail rule is inferred, not seen",
    "calls made while the ledger was disarmed, including every boot before the arm",
    "a guard-DECLINED call: guard_mode returns a decline without reaching "
    "_record_success, so look in the guard captures, not here",
    "records lost to the cap: `dropped` counts them and they are absent from a "
    "verdict, so a long sleep between the offer and the adopt can hide the pair",
    "the request's later fate: an adopt of N is the scheduler's offer, not proof "
    "the worker loaded N tokens",
    "requests the scheduler skipped before the connector (max_loras, encoder "
    "pending, budget): no lookup, no record, no evidence either way",
)


# --- arming ------------------------------------------------------------------

def _bounded_cap(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_CAP
    return max(MIN_CAP, min(MAX_CAP, value))


def arm(cap=None) -> dict:
    """Start recording. Allocates once; a re-arm at the same cap keeps contents.

    Does NOT install anything itself - `lookups_for_engine()` does that, so that
    a test can arm the ring without an engine in the process.
    """
    wanted = _bounded_cap(cap)
    report = {"armed": True, "requested_cap": cap, "cap": wanted,
              "reallocated": False, "reason": None}
    if wanted != int(_state["cap"]):
        _state["ring"] = [None] * wanted
        _state["cap"] = wanted
        _state["next"] = 0
        _state["live"] = 0
        _pending.clear()
        _state["pending_now"] = 0
        report["reallocated"] = True
    elif _state["ring"] is None:
        _ring()
    if not _state["armed"]:
        _state["arms"] += 1
        _state["armed"] = True
        # WHEN recording started, in wall-clock terms. The field that separates "armed
        # from the first request of this boot" from "armed by a POST at 16:40, so the
        # 16:34 request was never recorded" - the distinction the boot arm exists to
        # make, and worthless unless the moment is stamped where the arm happens.
        _state["armed_at"] = round(time.time(), 3)
        report["reason"] = f"recording the lookup/adopt pair into a {wanted}-slot ring"
    else:
        report["reason"] = "already armed" + (
            " at a new cap" if report["reallocated"] else "")
    return report


def boot_arm(guard_wrappers_installed: bool = True,
             inert: bool = False) -> dict:
    """ARM THE RING AT BOOT. Fail-open in the strongest sense: this function may not
    change the boot it is observing, and the only thing a refusal may cost is the ring.

    WHY A BOOT STEP RATHER THAN A POST. The ledger was disarmed on the boot where the
    16:34 request came back with 0.0% reuse after a 15-hour suspend, and its refusal
    reason was unrecoverable: the ring records NOTHING while disarmed, so the question
    "why did that offer adopt 0 tokens" had no answer and took two investigations to
    approximate. An observability ring that must be remembered is an observability ring
    that is disarmed for exactly the incident you will care about.

    THE ORDER OF THE GATES, and each gate's reason for being first:

      * INERT first, because `VLLM_E2_TIER_DEV_BOOT_INERT` is the operator's explicit
        "install and stop there" and arming a ring on that boot would be doing work the
        opt-out refused. It is NOT a refusal and is not counted.
      * the ENVIRONMENT second, because `0` is the rollback and must be distinguishable
        from a failure. Also not counted: an operator asked for no ring.
      * the GUARD WRAPPERS third. The ring is fed by `record_from_call`, which the guard
        wrappers call, so arming it on a boot that installed no wrappers would arm a
        recorder that nothing writes into - a payload saying `armed: true` over an empty
        ring, which is the worst shape of a lie this module could tell. So the
        dependency is ASKED, and the answer says which side failed.
      * `arm()` last, inside the guard, because a ring that cannot be allocated is a
        fact to count and log - not an exception to hand to a boot.

    WHAT THIS FUNCTION MUST NEVER DO: join `boot["refusals"]`, change the tier's arm, or
    touch a serve posture. `tests/test_lookup_ledger_boot_arm.py` pins that against this
    function's own source, and its behavioural test compares every other boot verdict
    field for field against the same boot with the ledger armed.
    """
    out = {"armed": False, "verdict": VERDICT_REFUSED, "reason": None,
           "cap": int(_state["cap"]), "recognised": True, "raw": None,
           "problems": [], "env_var": config.ENV_LOOKUP_LEDGER_BOOT_ARM,
           "note": ("a per-request offer/adopt ring, armed by the boot so the first "
                    "lookup of this boot is a record and not a counted miss")}
    if inert:
        out["verdict"] = VERDICT_INERT
        out["reason"] = (f"{config.ENV_BOOT_INERT} is set: this boot installs and "
                         "stops there, so no ring is armed and no lookup is recorded")
        _state["boot_verdict"] = VERDICT_INERT
        _state["boot_reason"] = out["reason"]
        return out
    try:
        gate = config.lookup_ledger_boot_arm()
    except Exception as exc:  # noqa: BLE001 - a posture read never costs a boot
        snapshot._error(exc, "lookup_ledger boot arm gate")
        gate = {"enabled": True, "source": "default", "raw": None,
                "recognised": True,
                "problems": [f"the read raised {type(exc).__name__}"]}
    out["raw"] = gate.get("raw")
    out["recognised"] = bool(gate.get("recognised", True))
    out["problems"] = [str(problem) for problem in (gate.get("problems") or [])]
    if not gate.get("enabled", True):
        out["verdict"] = VERDICT_ENV_REFUSED
        out["reason"] = (f"{config.ENV_LOOKUP_LEDGER_BOOT_ARM}="
                         f"{gate.get('raw')!r} asked for no ledger, so this boot "
                         "records no lookup/adopt pair")
        _state["boot_verdict"] = VERDICT_ENV_REFUSED
        _state["boot_reason"] = out["reason"]
        return out
    if not guard_wrappers_installed:
        # THE DEPENDENCY, ASKED RATHER THAN ASSUMED. A ring nothing writes into is worse
        # than no ring: it makes an empty answer look like a measured one.
        out["verdict"] = VERDICT_REFUSED
        out["reason"] = ("no guard wrappers are installed on this manager, so nothing "
                         "would write into the ring: arming it here would publish "
                         "`armed: true` over a ring that can never hold a record")
        _state["boot_arm_refusals"] = int(_state["boot_arm_refusals"]) + 1
        _state["boot_verdict"] = VERDICT_REFUSED
        _state["boot_reason"] = out["reason"]
        _logger.warning("kvtier-dev-plugin: LOOKUP LEDGER not armed at boot: %s. The "
                        "boot is otherwise unaffected - this costs a diagnostic, not a "
                        "tier.", out["reason"])
        return out
    try:
        verdict = arm()
    except Exception as exc:  # noqa: BLE001 - an observability arm never costs a boot
        snapshot._error(exc, "lookup_ledger boot arm")
        out["verdict"] = VERDICT_REFUSED
        out["reason"] = (f"{type(exc).__name__}: the ring could not be armed "
                         f"({exc})")
        _state["boot_arm_refusals"] = int(_state["boot_arm_refusals"]) + 1
        _state["boot_verdict"] = VERDICT_REFUSED
        _state["boot_reason"] = out["reason"]
        # ONE line per episode, at WARNING like the boot line, because a diagnostic that
        # could not be fitted has to be readable from the log - and one line per BOOT is
        # the discipline that keeps it a notice instead of a flood.
        _logger.warning("kvtier-dev-plugin: LOOKUP LEDGER not armed at boot: %s. The "
                        "boot is otherwise unaffected - this costs a diagnostic, not a "
                        "tier.", out["reason"])
        return out
    out["armed"] = bool(verdict.get("armed")) and bool(_state["armed"])
    out["cap"] = int(_state["cap"])
    out["armed_at"] = _state["armed_at"]
    out["reallocated"] = bool(verdict.get("reallocated"))
    out["verdict"] = VERDICT_ARMED if out["armed"] else VERDICT_REFUSED
    out["reason"] = str(verdict.get("reason") or "the ring was armed")
    if out["verdict"] == VERDICT_ARMED:
        _state["boot_arms"] = int(_state["boot_arms"]) + 1
    else:
        _state["boot_arm_refusals"] = int(_state["boot_arm_refusals"]) + 1
    _state["boot_verdict"] = out["verdict"]
    _state["boot_reason"] = out["reason"]
    return out


def disarm() -> dict:
    """Stop recording. The slots are KEPT so a reader can still see the last ones."""
    was = bool(_state["armed"])
    _state["armed"] = False
    if was:
        _state["disarms"] += 1
    return {"armed": False, "was_armed": was,
            "reason": ("recording stopped; the ring keeps its contents until "
                       "cleared, and nothing more is recorded")}


def clear() -> dict:
    """Forget the contents and the join index. Never the `seq`, never the arming."""
    cap = int(_state["cap"])
    _state["ring"] = [None] * cap
    _state["next"] = 0
    _state["live"] = 0
    _pending.clear()
    _state["pending_now"] = 0
    _state["clears"] += 1
    return {"cleared": True, "cap": cap, "seq": int(_state["seq"]),
            "armed": bool(_state["armed"])}


# --- the snapshot section ----------------------------------------------------

def section(preview: int = DEFAULT_PREVIEW) -> dict:
    """The snapshot's `lookup_ledger` block: counters, a bounded preview, gaps."""
    wanted = preview if isinstance(preview, int) and not isinstance(preview, bool) \
        else DEFAULT_PREVIEW
    wanted = max(0, min(MAX_PREVIEW, wanted))
    live = _live_records()
    out: dict = {"development_only": True, **counters()}
    out["methods"] = list(LEDGER_METHODS)
    out["preview"] = [_preview(record) for record in live[-wanted:]]
    out["preview_truncated"] = len(live) > len(out["preview"])
    out["pending_requests"] = sorted(_pending)[: 8]
    out["pending_requests_truncated"] = len(_pending) > 8
    out["not_covered"] = list(NOT_COVERED)
    out["attributed_records"] = sum(
        1 for record in _live_records()
        if isinstance(record.get("attribution"), dict))
    out["miss_attribution"] = _attribution_section()
    out["warning"] = ("read-only: the ledger changes nothing about what the "
                      "scheduler adopts, and it records nothing until armed")
    return out


def _attribution_section() -> dict:
    """The attribution block, contained: a broken observer cannot break a snapshot."""
    try:
        return _attribution().section()
    except Exception as exc:  # noqa: BLE001 - the snapshot must answer regardless
        snapshot._error(exc, "lookup ledger attribution section")
        return {"available": False, "reason": f"{type(exc).__name__}"[:120]}


# --- the verb ----------------------------------------------------------------

LOOKUPS_OPS = ("read", "arm", "disarm", "clear", "status", "verdict",
               "requests", "depth", "order", "rates")


def lookups_for_engine(engine, args=None) -> dict:
    """THE engine-side handler body for `lookups`. Never raises an Exception.

    `arm` installs the tier wrappers through `guard_mode.ensure_installed()` -
    the SAME entry point that already composes over the backport/switch layers
    and refuses a foreign owner. `disarm` removes them only when no other
    consumer needs them (`guard_mode.wrapper_customers()`), which is what stops
    a trace-ring disarm from silently blinding this ledger.
    """
    snapshot._state["calls"] += 1
    started = time.perf_counter()
    arguments = args if isinstance(args, dict) else {}
    from . import guard_mode  # noqa: PLC0415 - cycle: guard_mode records into us
    try:
        op = arguments.get("op", "read")
        op = op.strip().lower() if isinstance(op, str) else None
        if op not in LOOKUPS_OPS:
            payload = {"op": None, "error": (
                f"unknown op {arguments.get('op')!r}; expected one of "
                f"{list(LOOKUPS_OPS)}")}
        elif op == "arm":
            report = arm(arguments.get("cap"))
            report["install"] = guard_mode.ensure_installed(engine)
            _state["install"] = report["install"]
            if not report["install"].get("installed"):
                report["reason"] = (
                    (report["reason"] or "") + " | the ring is armed but NOTHING "
                    "is wrapped yet, so nothing will be recorded: "
                    + _install_refusal_summary(report["install"]))
            report["miss_attribution"] = _arm_attribution(engine, arguments)
            payload = {"op": "arm", **report}
        elif op == "disarm":
            report = disarm()
            report["uninstall"] = _uninstall_if_alone(engine)
            _state["uninstall"] = report["uninstall"]
            report["miss_attribution"] = _disarm_attribution(engine)
            payload = {"op": "disarm", **report}
        elif op == "clear":
            payload = {"op": "clear", **clear()}
        elif op == "status":
            payload = {"op": "status", "counters": counters(),
                       "wrapper_customers": guard_mode.wrapper_customers(),
                       "miss_attribution": _attribution().counters()}
        elif op == "verdict":
            payload = {"op": "verdict", **verdict(arguments.get("limit"))}
        elif op == "requests":
            payload = {"op": "requests",
                       **requests(arguments.get("limit"),
                                  arguments.get("verdict"))}
        elif op == "depth":
            payload = {"op": "depth", "retained_depth": _retained_depth(engine)}
        elif op == "order":
            payload = {"op": "order", "free_order": _order_report(
                arguments.get("limit"))}
        elif op == "rates":
            payload = {"op": "rates", "rates": _rates()}
        else:
            payload = {"op": "read", **read(arguments.get("limit"))}

        unknown = sorted(key for key in arguments
                         if key not in ("op", "cap", "limit", "verdict", "deep",
                                        "sample_every"))
        payload["version"] = PROTOCOL_VERSION
        payload["epoch"] = str(snapshot.epoch())
        payload["ts"] = round(time.time(), 6)
        payload["guard_mode"] = guard_mode.is_armed()
        payload["trace_armed"] = _trace_armed()
        payload["lookup_ledger_armed"] = is_armed()
        if unknown:
            payload["unknown_args"] = unknown
        payload["build_us"] = int((time.perf_counter() - started) * 1e6)
        payload["self"] = snapshot.error_counters()
        return snapshot.json_safe(payload)
    except Exception as exc:  # noqa: BLE001 - never into the scheduler
        snapshot._error(exc, "lookups")
        return {"version": PROTOCOL_VERSION, "verb": "lookups",
                "epoch": str(snapshot.epoch()),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "counters": snapshot.error_counters()}


def _attribution():
    """`miss_attribution`, imported at call time (it reads this module's ring)."""
    from . import miss_attribution  # noqa: PLC0415 - cycle, see _attribute_safely
    return miss_attribution


def _arm_attribution(engine, arguments) -> dict:
    """Arm capability 9 on the same verb, because it is the same question.

    `deep`/`sample_every` are optional arm-time arguments. The FILE MIRROR is
    deliberately env-only (`VLLM_E2_TIER_MISS_LOG`): an HTTP argument that names a
    path the engine then appends to is a write surface this plugin does not have
    anywhere else, and no diagnostic is worth becoming one. A failure here is
    reported in the payload and cannot fail the ledger's own arming.
    """
    try:
        return _attribution().arm(cap=arguments.get("cap"),
                                  deep=arguments.get("deep"),
                                  sample_every=arguments.get("sample_every"),
                                  engine=engine)
    except Exception as exc:  # noqa: BLE001 - the ledger stays armed regardless
        snapshot._error(exc, "lookups arm attribution")
        return {"armed": False, "error": f"{type(exc).__name__}"[:120]}


def _disarm_attribution(engine) -> dict:
    try:
        return _attribution().disarm(engine)
    except Exception as exc:  # noqa: BLE001, S110 - same containment as the arm
        snapshot._error(exc, "lookups disarm attribution")
        return {"armed": True, "error": f"{type(exc).__name__}"[:120]}


def _retained_depth(engine=None) -> dict:
    try:
        return _attribution().retained_depth(engine=engine)
    except Exception as exc:  # noqa: BLE001 - a measurement is never a 500
        snapshot._error(exc, "lookups depth")
        return {"available": False, "reason": f"{type(exc).__name__}"[:120]}


def _order_report(limit) -> dict:
    try:
        return _attribution().order_report(limit)
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "lookups order")
        return {"installed": False, "reason": f"{type(exc).__name__}"[:120]}


def _rates() -> dict:
    try:
        return _attribution().rates()
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "lookups rates")
        return {"allowed": {}, "refused": [f"{type(exc).__name__}"[:120]]}


def _install_refusal_summary(install) -> str:
    """WHY nothing is wrapped, in the install report's own words. Never raises.

    Worth the code because "armed but recording nothing" is the one state an
    operator must be able to diagnose from the payload alone, and "no tier class
    is visible" would be a lie when the real reason is a foreign owner marker or a
    fork that renamed the method. The install report already says which; quote it,
    bounded.
    """
    try:
        detail = [item for item in (install.get("skip_detail") or [])
                  if isinstance(item, dict)]
        if detail:
            head = "; ".join(f"{item.get('method')}: {item.get('reason')}"
                             for item in detail[:3])
            if len(detail) > 3:
                head += f"; +{len(detail) - 3} more refused"
            return head[:400]
        return str(install.get("reason")
                   or "no tier class in this process, and no reason given")[:300]
    except Exception:  # noqa: BLE001 - a summary of a refusal must not be a fault
        return "the install report could not be read"


def _uninstall_if_alone(engine) -> dict:
    """Remove the tier wrappers only if this ledger was their last consumer."""
    from . import guard_mode  # noqa: PLC0415 - cycle, see above
    customers = guard_mode.wrapper_customers()
    if customers:
        return {"uninstalled": False,
                "reason": ("still needed by: " + ", ".join(customers) +
                           "; the wrappers stay in place")}
    manager = guard_mode.manager_of(engine)
    cls = type(manager) if manager is not None else guard_mode.manager_class()
    return guard_mode.uninstall(cls)


def _trace_armed() -> bool:
    """The trace ring's state, for the payload's dev-state echo. Never raises."""
    try:
        from . import trace_ring  # noqa: PLC0415 - cycle: guard_mode records into it
        return trace_ring.is_armed()
    except Exception:  # noqa: BLE001, S112 - a dev-state echo is not worth an error
        return False
