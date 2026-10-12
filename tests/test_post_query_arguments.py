r"""A POST's QUERY STRING is read, and a refused request leaves nothing behind.

THE DEFECT THIS SUITE KILLS (0.2.21). Every POST handler used to build its arguments
from the JSON BODY ALONE (`body_arguments`) while every GET built them from the query
string - and the routes are documented for both forms, so an operator who typed the
shortest true thing:

    POST /plugins/kvtier/disk?op=arm&seconds=3600

got `{"ok": true, ..., "op": "status", "arm_state": {"armed": false}}`. The query
string was never read, the verb ran `status`, and the answer said SUCCESS. The arm
file kept its old mtime; nothing was armed; the operator believed a live-code path was
open. For a verb whose whole job is to open and close a path (arm / disarm / the offer
switch), an unexecuted operation reporting success is worse than an error: the inverse
case is a `disarm` that "succeeded" while the tier stayed armed.

So these tests are written backwards from a normal suite. The interesting assertion in
each one is about the ABSENCE of an effect - the arm file's bytes and mtime, the offer
switch's value, and the number of engine calls - not about the response text. A 400
that also armed the tier would pass a suite that only reads the response.

Coverage:
  1. the query form is HONOURED, end to end, with the REAL verb behind a fake utility
     channel (a fake that returns a canned dict would prove nothing about side effects);
  2. query + body: agreement runs once, disagreement is a 400 that does nothing;
  3. an unknown query key on a POST is refused, not ignored (the same accepted-then-
     ignored shape, one level down);
  4. the safe-method wall is untouched: `GET ?op=arm` is still a 405 and still arms
     nothing - and since 0.2.72 it is the MATRIX, not one route: `/trace`, `/guard`,
     `/tier`, `/lookups` and `/blockdump` all refused a state change on nothing at all
     before this release, and all four answer 405 now, with the engine-side backstop
     behind each of them;
  5. the fix is not `/disk`-only: `/blockdump?op=arm` honours it too, because leaving
     five routes with the same trap would be a half fix.
"""

from __future__ import annotations

import hashlib
import os

import pytest
from test_routes import (
    BLOCKDUMP,
    DISK,
    FakeAsyncLLM,
    FakeEngineCoreClient,
    build_app,
    client_for,
    token_headers,
)

from kvtier_dev_plugin import (
    blockdump,
    config,
    disk_layout,
    disk_restore,
    disk_spill,
    disk_store,
    route_policy,
)

pytest.importorskip("fastapi")

KEY = "a3f0b7c1d4e5f607" + "0" * 8


class RealVerbEngine:
    """A fake utility channel that RUNS the real `disk` verb.

    `test_routes.FakeEngineCoreClient` returns a canned dict, which is the right fake
    for routing tests and useless here: the point of this suite is that a refused
    request changes no file. The side effects have to be real, so the hop calls
    `disk_store.disk_for_engine` exactly as `engine_side` binds it.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if method == config.METHOD_DISK:
            return disk_store.disk_for_engine(None, args[0] if args else {})
        return {"ok": True, "method": method}


@pytest.fixture(autouse=True)
def plugin_on(monkeypatch):
    """Live, plus the token variable set: 0.2.68 removed the gate it used to feed,
    and every call in this file still sends the header - so the file measures that a
    leftover variable and a stale header change no answer, including the 405 that is
    now the only wall in front of a mutating `op` on a GET."""
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A configured spill tier with a diagnostics directory: an arm file is possible.

    The tmpfs refusal is relaxed for the PREFIX LIST only, because pytest's `tmp_path`
    lives under /tmp; the rule itself is pinned in `test_disk_spill.py`.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    root = tmp_path / "kv-store"
    diag = tmp_path / "diag"
    diag.mkdir()
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(64 * 1024 * 1024))
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    disk_spill.reset_state_for_tests()
    disk_restore.reset_state_for_tests()
    yield {"root": root, "diag": diag}
    disk_spill.reset_state_for_tests()
    disk_restore.reset_state_for_tests()


def engine_app(engine):
    return client_for(build_app(FakeAsyncLLM(engine)))


def arm_file():
    return disk_spill._arm_path()


def arm_file_state():
    """`(exists, sha256, mtime_ns)` of the arm file: its IDENTITY, not its verdict.

    A request that "did nothing" must not have touched the bytes or the clock. Reading
    the file rather than `disk_spill.is_armed()` is deliberate: the in-memory verdict is
    what a same-process `arm()` changes, and the whole point of the file is that a
    WORKER process reads it - so a test that only checked the verdict could pass while
    the file a worker obeys was rewritten.
    """
    path = arm_file()
    if path is None or not os.path.exists(path):
        return (False, None, None)
    with open(path, "rb") as handle:
        raw = handle.read()
    return (True, hashlib.sha256(raw).hexdigest(),
            os.stat(path).st_mtime_ns)


def arm_for_a_minute():
    """Arm through the REAL verb and prove it took.

    Every "a refused request changed nothing" test below needs a state that COULD have
    been changed, or its most interesting assertion is vacuous: comparing the absence
    of an arm file against a request that also armed nothing proves nothing.
    """
    answer = disk_store.disk_for_engine(None, {"op": "arm", "seconds": 600})
    assert answer["armed"] is True, answer
    assert arm_file_state()[0] is True, "arming wrote no arm file"


def blockdump_arm_state():
    path = blockdump._arm_path()
    if path is None or not os.path.exists(path):
        return (False, None, None)
    with open(path, "rb") as handle:
        raw = handle.read()
    return (True, hashlib.sha256(raw).hexdigest(), os.stat(path).st_mtime_ns)


# --- 1. the query form is honoured ------------------------------------------

def test_a_query_string_op_arms_the_tier_for_real(world):
    """The exact request that used to lie, asked again.

    Before the fix this answered `ok: true` with `"op": "status"` and left the arm file
    unwritten. Now it runs `arm`, and BOTH halves of the claim are checked: the
    response says `arm`, and the file that a worker process obeys exists.
    """
    engine = RealVerbEngine()
    before = arm_file_state()
    response = engine_app(engine).post(f"{DISK}?op=arm&seconds=600",
                                       headers=token_headers())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["op"] == "arm"
    assert body["armed"] is True, body
    assert disk_spill.is_armed() is True
    assert disk_spill.is_armed() is True
    after = arm_file_state()
    assert after[0] is True, "the arm file a WORKER obeys was never written"
    assert after != before


def test_a_query_string_op_and_a_body_combine_into_one_call(world):
    """`?op=arm` with `{"seconds": 3600}`: the parts come from different places and
    arrive as ONE argument dict - one hop, one op, no half-applied request."""
    engine = RealVerbEngine()
    response = engine_app(engine).post(f"{DISK}?op=arm", json={"seconds": 3600},
                                       headers=token_headers())
    assert response.status_code == 200, response.text
    assert response.json()["op"] == "arm"
    assert response.json()["armed"] is True
    assert engine.calls == [(config.METHOD_DISK,
                             ({"op": "arm", "seconds": 3600},))]
    assert disk_spill.is_armed() is True


def test_a_query_string_disarm_really_disarms(world):
    """The inverse of the dangerous case: a `disarm` that only looked like it worked
    leaves a live tier armed, so this one is checked against the file too."""
    engine = RealVerbEngine()
    arm_for_a_minute()
    assert arm_file_state()[0] is True
    response = engine_app(engine).post(f"{DISK}?op=disarm", headers=token_headers())
    assert response.status_code == 200
    assert response.json()["op"] == "disarm"
    assert disk_spill.is_armed() is False
    assert arm_file_state()[0] is False


def test_an_empty_body_with_a_query_op_is_not_a_bad_request(world):
    """`POST ?op=status` with no body at all: there is no JSON to complain about."""
    engine = RealVerbEngine()
    response = engine_app(engine).post(f"{DISK}?op=status", headers=token_headers())
    assert response.status_code == 200, response.text
    assert response.json()["op"] == "status"


# --- 2. query AND body: agree or be refused ---------------------------------

def test_a_key_in_both_places_must_agree_and_runs_once(world):
    """`?seconds=3600` and `{"seconds": 3600}` AGREE (the int rule coerces the query
    string), so this is one request, not two, and not a conflict."""
    engine = RealVerbEngine()
    response = engine_app(engine).post(f"{DISK}?op=arm&seconds=3600",
                                       json={"op": "arm", "seconds": 3600},
                                       headers=token_headers())
    assert response.status_code == 200, response.text
    assert engine.calls == [(config.METHOD_DISK,
                             ({"op": "arm", "seconds": 3600},))]
    assert disk_spill.is_armed() is True


def test_a_disagreement_between_query_and_body_is_a_400_with_no_effect(world):
    """THE PRECEDENCE DECISION, tested. There is no "body wins"/"query wins" rule,
    because a silent winner is the bug being fixed in miniature: the loser would be
    ignored without a word. `?op=arm` beside `{"op":"disarm"}` therefore says nothing
    and DOES NOTHING - the arm file is byte-identical, the offer switch unmoved, and
    the verb is never called at all."""
    engine = RealVerbEngine()
    arm_for_a_minute()
    disk_restore.set_offer_switch(True)
    before = arm_file_state()
    posture_before = disk_restore.disk_posture()["enabled"]
    response = engine_app(engine).post(f"{DISK}?op=arm", json={"op": "disarm"},
                                       headers=token_headers())
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "op" in detail and "disagree" in detail
    assert arm_file_state() == before
    assert disk_restore.disk_posture()["enabled"] == posture_before
    assert engine.calls == [], "the engine was reached: the request was not refused"


def test_a_disagreeing_scalar_is_refused_too(world):
    """Same rule for a value, not just for the `op`: two different `seconds` for one
    arm is a question, not a request."""
    engine = RealVerbEngine()
    before = arm_file_state()
    response = engine_app(engine).post(f"{DISK}?op=arm&seconds=600",
                                       json={"seconds": 3600},
                                       headers=token_headers())
    assert response.status_code == 400
    assert "seconds" in response.json()["detail"]
    assert arm_file_state() == before
    assert engine.calls == []


# --- 3. a refused mutating request leaves nothing behind --------------------

def test_an_out_of_range_seconds_on_a_query_arm_is_refused_and_writes_nothing(world):
    """`seconds` is bounded by the route's spec (max 7200). A query-string form that
    used to be IGNORED could not fail this way at all; now it is validated, so the
    request is refused - and the ARM THAT WAS ALREADY THERE is untouched, byte for
    byte and mtime for mtime.

    This is the test the brief asked for: the assertion is about the file, and the
    strongest line in it is that the engine was never called.
    """
    engine = RealVerbEngine()
    arm_for_a_minute()
    before = arm_file_state()
    assert before[0] is True
    response = engine_app(engine).post(f"{DISK}?op=arm&seconds=99999",
                                       headers=token_headers())
    assert response.status_code == 400
    assert "seconds" in response.json()["detail"]
    assert arm_file_state() == before, "a refused arm rewrote the arm file"
    assert engine.calls == [], "the verb ran: the refusal happened after the write"


def test_an_unknown_mutating_op_in_a_query_is_refused_and_writes_nothing(world):
    """`?op=disarm-please` is outside the enum. Before the fix the query string was
    dropped on the floor and this answered `ok: true` (it ran `status`)."""
    engine = RealVerbEngine()
    arm_for_a_minute()
    before = arm_file_state()
    response = engine_app(engine).post(f"{DISK}?op=disarm-please",
                                       headers=token_headers())
    assert response.status_code == 400
    assert arm_file_state() == before
    assert disk_spill.is_armed() is True
    assert engine.calls == []


def test_a_refused_offers_request_leaves_the_offer_switch_exactly_as_it_was(world):
    """The offer switch is the OTHER live-code path (it decides whether a disk offer
    goes in front of live traffic), so it gets the same treatment: a bad `enabled`
    value on the query string is a 400, and the switch does not move in either
    direction."""
    engine = RealVerbEngine()
    for wanted in (True, False):
        disk_restore.set_offer_switch(wanted)
        before = disk_restore.disk_posture()
        response = engine_app(engine).post(f"{DISK}?op=offers&enabled=maybe",
                                           headers=token_headers())
        assert response.status_code == 400, response.text
        assert "enabled" in response.json()["detail"]
        assert disk_restore.disk_posture() == before
        assert engine.calls == []


def test_an_unknown_query_key_on_a_post_is_refused_not_ignored(world):
    """The same accepted-then-ignored shape one level down: `?op=status&shardd=ab` used
    to run `status` and answer `ok: true` about a key nobody read. A POST is no longer
    a place where a typo in an argument costs nothing - the GET rule applies."""
    engine = RealVerbEngine()
    response = engine_app(engine).post(f"{DISK}?op=status&shardd=ab",
                                       headers=token_headers())
    assert response.status_code == 400
    assert "shardd" in response.json()["detail"]
    assert engine.calls == []


# --- 4. the safe-method wall is untouched -----------------------------------

def test_a_mutating_op_on_get_is_still_a_405_that_arms_nothing(world):
    """Honouring a query `op` on POST must not leak into the safe method: a browser
    issues a GET with no preflight and no custom header, so the URL that arms the tier
    stays refused there. The 405 is the route's guard and the engine-side `safe_method`
    refusal is the backstop behind it - both must still hold."""
    engine = RealVerbEngine()
    before = arm_file_state()
    response = engine_app(engine).get(f"{DISK}?op=arm&seconds=600",
                                      headers=token_headers())
    assert response.status_code == 405
    assert arm_file_state() == before
    assert disk_spill.is_armed() is False
    assert engine.calls == []


def test_the_engine_side_safe_method_refusal_still_holds(world):
    """The backstop, asked directly: even a caller that reaches the verb with
    `safe_method` set cannot run a mutating op."""
    for op in disk_store.MUTATING_DISK_OPS:
        answer = disk_store.disk_for_engine(None, {"op": op,
                                                   "safe_method": True})
        assert answer["ok"] is False, op
        assert "will not run from a method" in answer["reason"]
    assert arm_file_state() == (False, None, None)


# --- 5. the class, not one route --------------------------------------------

def test_blockdump_arms_from_a_query_string_too(world):
    """`/blockdump` had the identical shape (`op` is a query parameter on its GET, and
    its POST read the body only). Fixing one route and leaving five with the same trap
    would leave the same lie on the surface, so the merge lives in the shared
    `post_arguments` helper and this proves it is not `/disk`-specific."""
    engine = RealVerbEngine()
    before = blockdump_arm_state()
    assert blockdump.is_armed() is False
    response = engine_app(engine).post(f"{BLOCKDUMP}?op=arm&seconds=600",
                                       headers=token_headers())
    assert response.status_code == 200, response.text
    # The canned engine answer stands in for the blockdump verb (this fake only runs
    # the disk verb); what is proved here is that the QUERY STRING reached the hop.
    assert engine.calls == [(config.METHOD_BLOCKDUMP,
                             ({"op": "arm", "seconds": 600},))]
    assert blockdump_arm_state() == before


# --- 6. the wall is a MATRIX now, not one route (0.2.72) --------------------
#
# `/disk` had this rule since the day the safe-method idea was introduced. The other
# five state-changing routes had NOTHING: `GET /guard?op=arm` armed guard mode, `GET
# /tier?op=arm` flipped the tier's kill switch, `GET /lookups?op=arm` installed
# wrappers on the live tier, `GET /trace?op=clear` emptied the ring, `GET
# /blockdump?op=arm` armed a facility that reads parked KV bytes out of a running
# engine. Each route takes its `op` from a QUERY PARAMETER, so in every case the URL
# that changes the engine is as short as the URL that reads its status - and a browser
# fetches a URL with no preflight and no custom header. Removing the peer-address gate
# made this the load-bearing wall, which is why it is asserted per route rather than
# "once, in the helper": the one-route version is exactly how five routes ended up
# without it.

MATRIX_ROUTES = {
    config.ROUTE_TRACE: ("arm", "disarm", "clear"),
    config.ROUTE_GUARD: ("arm", "disarm"),
    config.ROUTE_TIER: ("arm", "disarm"),
    config.ROUTE_LOOKUPS: ("arm", "disarm", "clear"),
    config.ROUTE_BLOCKDUMP: ("arm", "disarm", "dump"),
}


@pytest.mark.parametrize("route,op", sorted(
    (route, op) for route, ops in MATRIX_ROUTES.items() for op in ops))
def test_a_mutating_op_on_get_is_a_405_on_every_route_and_touches_no_engine(route,
                                                                            op):
    """The 405 half of the wall, over the whole matrix: refused, named, and the
    engine never heard about it.

    `engine.calls == []` is the assertion that matters. A 405 that armed something is
    a worse answer than a 200 that did not, because the operator reads the status code
    and the arm file keeps its secret. The detail must name the op it refused and the
    POST route that runs it - a refusal an operator cannot act on is a bug report.
    """
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.get(f"{config.ROUTE_PREFIX}{route}?op={op}",
                          headers=token_headers())
    assert response.status_code == 405, (route, op, response.text)
    assert "will not run from GET" in response.json()["detail"]
    assert f"POST {config.ROUTE_PREFIX}{route}" in response.json()["detail"]
    assert op in response.json()["detail"]
    assert engine.calls == [], (route, op, engine.calls)


def test_the_guard_armed_spelling_is_refused_on_get_too():
    """`/guard`'s ergonomic spelling is the case an op list cannot see.

    `{"armed": true}` is arm/disarm in another word (see `guard_mode.guard_for_engine`),
    so on this route the PRESENCE of the key is the state change whatever `op` says -
    `GET /guard?armed=true` used to flip guard mode on, and guard mode masks real bugs.
    The rule lives in `route_policy.MUTATING_ARGUMENTS`, shared by both layers, so
    neither can be right about it while the other is wrong.
    """
    engine = FakeEngineCoreClient()
    client = client_for(build_app(FakeAsyncLLM(engine)))
    for spelling in ("armed=true", "armed=false", "armed=1", "armed=0"):
        response = client.get(f"{config.ROUTE_PREFIX}{config.ROUTE_GUARD}?{spelling}",
                              headers=token_headers())
        assert response.status_code == 405, (spelling, response.text)
        assert "'armed'" in response.json()["detail"], spelling
        assert "will not run from GET" in response.json()["detail"]
    assert engine.calls == []


def test_a_read_on_get_is_still_a_read_on_every_route():
    """The wall must not become a wall on reads: every route's default and every
    declared read op still answers 200 from a GET.

    A 405 that also blocks reads is a refusal that gets disabled within a day, and the
    `status` op on `/tier` is the case the rule's author nearly got wrong: `status`
    re-runs the install check, which is work, but it changes no posture, so it is a
    read. If a future op is added to a route's mutating list and is really a read, this
    is the test that says so.
    """
    engine = FakeEngineCoreClient(result={"version": 1})
    client = client_for(build_app(FakeAsyncLLM(engine)))
    reads = {
        config.ROUTE_TRACE: (None, "read", "status"),
        config.ROUTE_GUARD: (None, "status", "captures"),
        config.ROUTE_TIER: (None, "status"),
        config.ROUTE_LOOKUPS: (None, "read", "status", "verdict", "depth"),
        config.ROUTE_BLOCKDUMP: (None, "status"),
        config.ROUTE_DISK: (None, "status"),
    }
    for route, ops in reads.items():
        for op in ops:
            url = f"{config.ROUTE_PREFIX}{route}" + (f"?op={op}" if op else "")
            response = client.get(url, headers=token_headers())
            assert response.status_code == 200, (route, op, response.text)


@pytest.mark.parametrize("route,op", sorted(
    (route, op) for route, ops in MATRIX_ROUTES.items() for op in ops))
def test_the_engine_side_verb_refuses_a_stamped_mutation_on_every_route(route, op):
    """The backstop, asked of each verb directly - the four that had none until 0.2.72
    included.

    The utility channel answers a refusal as DATA (a payload with `ok: false`), not as
    an exception, so a refusal that lives only in the HTTP layer means a future HTTP
    bug lands the mutation. Each verb is called the way `engine_side` binds it, with
    `safe_method` set, and must decline with both refusal vocabularies present: `reason`
    (the `/disk` dialect) and `error` (the ring verbs'), so no caller has to know which
    family a refusal was written in. The wording is pinned because operators grep for
    it, and because "will not run from a method" is the phrase that tells a reader this
    was a method refusal and not a failure.
    """
    from kvtier_dev_plugin import backport, guard_mode, lookup_ledger, trace_ring

    handlers = {
        config.ROUTE_TRACE: trace_ring.trace_for_engine,
        config.ROUTE_GUARD: guard_mode.guard_for_engine,
        config.ROUTE_TIER: backport.tier_for_engine,
        config.ROUTE_LOOKUPS: lookup_ledger.lookups_for_engine,
        config.ROUTE_BLOCKDUMP: blockdump.blockdump_for_engine,
    }
    answer = handlers[route](None, {"op": op, "safe_method": True})
    assert answer["ok"] is False, (route, op, answer)
    assert answer["mutating_op"] is True, (route, op, answer)
    assert answer["status_code"] == 405, (route, op, answer)
    assert "will not run from a method" in answer["reason"], (route, op, answer)
    assert "will not run from a method" in answer["error"], (route, op, answer)
    assert f"POST {config.ROUTE_PREFIX}{route}" in answer["reason"], (route, op, answer)


def test_the_engine_side_refusal_asks_the_armed_key_too_and_nothing_else():
    """Both halves of `/guard`'s rule, from the verb's own side: `{"armed": true}`
    stamped is refused, and the same call WITHOUT the stamp is not refused by this
    layer.

    The second half is the one that keeps the backstop from becoming the gate: a POST
    carries no `safe_method` stamp, and a verb that refused an unstamped mutation would
    take away the route that exists to run it. Refusing on the stamp is what lets the
    HTTP 405 and this payload be two answers to one question instead of two questions.
    """
    from kvtier_dev_plugin import guard_mode

    refused = guard_mode.guard_for_engine(None, {"armed": True,
                                                 "safe_method": True})
    assert refused["ok"] is False and refused["mutating_op"] is True
    assert "'armed'" in refused["reason"]
    assert route_policy.safe_method_refusal(config.ROUTE_GUARD,
                                           {"armed": True}) is None
    assert route_policy.safe_method_refusal(config.ROUTE_GUARD,
                                           {"op": "status"}) is None
    assert route_policy.safe_method_refusal(config.ROUTE_GUARD,
                                           {"op": "arm"}) is None


def test_the_http_and_engine_halves_cannot_disagree_about_which_ops_mutate():
    """One list, both layers, asserted as a property rather than a convention.

    The failure this prevents is silent and specific: a new mutating `op` added to a
    dispatch table but not to the route's policy answers 200 from a GET (the HTTP half
    does not know it mutates) and then runs - the engine half checks the same list, so
    it does not catch it either. Reading the ops from `route_policy` here, and
    `/disk`'s from its own module, means the matrix above and the refusal below are
    generated from the same source the product uses.
    """
    from kvtier_dev_plugin import disk_store

    assert route_policy.mutating_ops(config.ROUTE_DISK) == tuple(
        disk_store.MUTATING_DISK_OPS), "the route must not keep a copy of /disk's list"
    for route, ops in MATRIX_ROUTES.items():
        assert route_policy.mutating_ops(route) == ops, route
        for op in ops:
            label = route_policy.mutating_op(route, {"op": op})
            assert label == f"op {op!r}", (route, op, label)
            assert "will not run from GET" in route_policy.refusal_detail(route, label)
            assert route_policy.safe_method_refusal(
                route, {"op": op, "safe_method": True}) is not None, (route, op)
    # A read, a missing op and an unknown op are NOT mutations, on every route.
    for route in list(MATRIX_ROUTES) + [config.ROUTE_DISK]:
        for args in ({}, {"op": route_policy.DEFAULT_OPS[route]}, {"op": "nonsense"},
                     {"limit": 5}):
            assert route_policy.mutating_op(route, args) is None, (route, args)
    # `/disk`'s default and every route's default must be reads, or a bare GET mutates.
    assert route_policy.mutating_op(config.ROUTE_DISK,
                                    {"op": "status"}) is None
    assert set(route_policy.DEFAULT_OPS) == set(MATRIX_ROUTES) | {config.ROUTE_DISK}
