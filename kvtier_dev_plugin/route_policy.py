"""WHICH VERBS CHANGE STATE, and what stops them running from a safe method.

ONE LIST OF TRUTH. Before this file the question "does this `op` change state?" had
one answer per route: `/disk` asked `disk_store.MUTATING_DISK_OPS` (and refused a
mutating `op` on a GET with 405, with an engine-side `safe_method` backstop behind
it), while `/trace`, `/guard`, `/tier`, `/lookups` and `/blockdump` each answered it
nowhere at all - so `GET /plugins/kvtier/guard?op=arm` armed guard mode, `GET
/tier?op=arm` armed the KV tier, `GET /lookups?op=arm` installed wrappers on the
tier, `GET /trace?op=clear` emptied the ring and `GET /blockdump?op=arm` armed a
facility that reads parked KV bytes.

WHY A GET IS THE PROBLEM, when the caller has already been admitted. A browser
issues a GET with no custom header and no CORS preflight, so any page that can make
a client load an image can issue these URLs; and every one of these routes takes its
`op` from a QUERY PARAMETER, so the URL that arms a facility is as short as the URL
that reads its status. A state change must arrive on a method the browser has to
ask permission for - that is the whole argument, and it has nothing to do with
who is calling: since the peer-address gate was removed, these routes answer whoever
can reach the port, and the safe-method rule is one of the few walls left.

THE SHAPE OF THE RULE:
  * `mutating_op()` names the state change an argument dict asks for, or None;
  * the HTTP half answers 405 (see `refusal_detail`) BEFORE the engine is touched,
    and sends `safe_method: true` when it forwards a read;
  * the ENGINE half refuses too (`safe_method_refusal`), mirroring
    `disk_store.disk_for_engine`, so a future HTTP bug cannot put a mutating `op` on
    a safe method and have it land. A refusal in one layer only is a refusal that
    depends on the other layer staying correct.

`/guard` needs one extra rule: `{"armed": true|false}` is the ergonomic spelling of
arm/disarm (see `guard_mode.guard_for_engine`), so on that route the PRESENCE of the
`armed` key is itself the state change, whatever `op` says. `MUTATING_ARGUMENTS` is
where that is written down, so the two layers cannot disagree about it either.

`/disk` is deliberately absent from `MUTATING_OPS`: its list lives with the verb in
`disk_store.MUTATING_DISK_OPS` (the same tuple the dispatch table and several tests
are pinned against), and this module reads it from there rather than copying it. A
copy is how two answers to one question start to differ.
"""

from __future__ import annotations

from . import config

# The argument key the HTTP layer sets on a read it forwarded from a method a browser
# may issue without a preflight. One name, defined once, for the route that sets it
# and the five verbs that honour it.
SAFE_METHOD_KEY = "safe_method"

# Every state-changing op of every route whose op list lives with this module. Read
# this as "arming, disarming, clearing, and anything that starts a thread or writes a
# file": a read of a ring, a counter or a posture is never in here.
#
#  * `/trace`   - `arm` installs the scheduler/tier observers, `disarm` removes them,
#                 `clear` empties the ring. `read`/`status` are pure reads.
#  * `/guard`   - `arm`/`disarm` flip guard mode (and `armed`, see MUTATING_ARGUMENTS).
#                 `status`/`captures` read.
#  * `/tier`    - `arm`/`disarm` flip the tier's kill switch. `status` re-runs the
#                 install check but changes no posture (and is the route's read
#                 default, so treating it as mutating would break every GET).
#  * `/lookups` - `arm` installs the ledger's wrappers, `disarm` may remove them,
#                 `clear` empties the ring. `depth`/`order`/`rates`/`verdict`/
#                 `requests`/`read`/`status` are reads.
#  * `/blockdump` - `arm`/`disarm` the facility, and `dump` too: it writes the pending
#                 request file every worker picks up, which makes FOUR processes read
#                 real KV bytes and write files. It is the route's only "read" that
#                 changes anything, and the rate window it charges is state.
MUTATING_OPS: dict[str, tuple[str, ...]] = {
    config.ROUTE_TRACE: ("arm", "disarm", "clear"),
    config.ROUTE_GUARD: ("arm", "disarm"),
    config.ROUTE_TIER: ("arm", "disarm"),
    config.ROUTE_LOOKUPS: ("arm", "disarm", "clear"),
    config.ROUTE_BLOCKDUMP: ("arm", "disarm", "dump"),
}

# Keys whose PRESENCE is a state change regardless of `op`, per route.
MUTATING_ARGUMENTS: dict[str, tuple[str, ...]] = {
    config.ROUTE_GUARD: ("armed",),
}

# The op a route runs when the caller named none, so `mutating_op()` can judge the
# request the way the route will. A default is always a read (that is what makes it
# safe to default), so nothing here can turn a bare GET into a mutation.
DEFAULT_OPS: dict[str, str] = {
    config.ROUTE_TRACE: "read",
    config.ROUTE_GUARD: "status",
    config.ROUTE_TIER: "status",
    config.ROUTE_LOOKUPS: "read",
    config.ROUTE_BLOCKDUMP: "status",
    config.ROUTE_DISK: "status",
}


def _disk_mutating_ops() -> tuple[str, ...]:
    """`/disk`'s list, from its owner. Imported at call time: `disk_store` is the
    engine-side half and this module is imported by the HTTP layer, which must stay
    loadable on its own."""
    from . import disk_store  # noqa: PLC0415 - lazy by idiom, and breaks no cycle
    return tuple(disk_store.MUTATING_DISK_OPS)


def mutating_ops(route: str) -> tuple[str, ...]:
    """The state-changing ops of one route, from its ONE list of truth."""
    if route == config.ROUTE_DISK:
        return _disk_mutating_ops()
    return MUTATING_OPS.get(route, ())


def mutating_op(route: str, args) -> str | None:
    """A human-readable label for the state change `args` asks for, else None.

    None means "this is a read", and includes a missing `op` (the route's default,
    which is always a read) and an `op` outside the route's vocabulary - a name no
    verb implements cannot change state, and the route's own argument spec has
    usually refused it long before here.

    The label is built for a refusal message: `op 'arm'` for an op, `the 'armed'
    argument` for the `/guard` spelling. Returning a label rather than a bool is what
    lets the 405 name the thing it refused, which is the difference between a
    refusal an operator can act on and one they have to reproduce to understand.
    """
    arguments = args if isinstance(args, dict) else {}
    raw = arguments.get("op")
    op = raw.strip().lower() if isinstance(raw, str) else None
    ops = mutating_ops(route)
    if op in ops:
        return f"op {op!r}"
    for key in MUTATING_ARGUMENTS.get(route, ()):
        if key in arguments:
            return f"the {key!r} argument"
    return None


def post_route(route: str) -> str:
    """The full path the caller must use instead."""
    return config.ROUTE_PREFIX + route


def refusal_detail(route: str, label: str) -> str:
    """The 405 body: names the state change and the POST route that runs it."""
    return (f"{label} changes state and will not run from GET; send it as "
            f"POST {post_route(route)} (this route's GET runs read-only ops only)")


def safe_method_refusal(route: str, args) -> dict | None:
    """The ENGINE-SIDE backstop payload, or None when this call may run.

    Same question as `mutating_op`, answered the same way, and asked by every verb
    handler behind a route that has a safe method: `disk_store.disk_for_engine` has
    done this since the safe-method rule was introduced, and the other four verbs
    must not be the ones that rely on the HTTP layer never making a mistake. The
    payload carries BOTH refusal vocabularies (`reason` as `/disk` words it, `error`
    as the ring verbs do) so no reader has to know which dialect a refusal is
    written in.
    """
    arguments = args if isinstance(args, dict) else {}
    if not arguments.get(SAFE_METHOD_KEY):
        return None
    label = mutating_op(route, arguments)
    if label is None:
        return None
    reason = (f"{label} mutates and will not run from a method a browser may issue "
              f"without a preflight; send it as POST {post_route(route)}")
    return {"ok": False, "op": arguments.get("op"), "mutating_op": True,
            "status_code": 405, "reason": reason, "error": reason}
