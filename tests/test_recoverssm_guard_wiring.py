r"""The DRAFT-0007 LOAD SIDE: the connector publishes, and un-publishes.

WHAT THIS SUITE IS FOR
----------------------
`recoverssm.py.patch` ships the collision predicate and `mamba_hybrid.py.patch`
publishes the commit side. Both were inert: nothing called
`publish_inflight_load_blocks`, so the in-flight set was empty forever and
`guard_armed()` was False forever. `kv_connector.py.patch` is the piece that
arms it - the connector's load submit hands the destination block ids to the
guard, and the connector's completion report takes them back. This file is what
proves THAT hunk does what it says, because a guard that reports itself as
protection while doing nothing is worse than no guard at all.

TWO KINDS OF TEST, AND WHY BOTH ARE HERE
----------------------------------------
*BEHAVIOURAL*, on the shipped bytes. Like
`test_recoverssm_load_collision_guard.py`, nothing is re-implemented: the `+`
lines of the hunk are carved out and exec'd, so the code under test is the code
that ships. The carve is textual because the patched module imports `torch` and
`vllm.*` at module scope and cannot be imported here at all (the plugin imports
without an engine); the carved methods need only `os`, `json`, a logger, and two
module constants, all of which the hunk itself provides or which are stdlib.
The sink on the other side of the wiring is the REAL `RecoverSSMState`
extracted from `recoverssm.py.patch` by the sibling suite - so a publish
travels through the shipped connector code into the shipped predicate and the
shipped exception, end to end, in one process, with no engine.

*STRUCTURAL*, on the patch text. The three call sites (capture at bind, publish
at the submit, retire at the completion report) are one line each inside
existing methods - `pre_forward`, `_start_load_kv`, `post_forward` - whose
bodies cannot be exec'd here without a connector, a torch context and a GPU. A
behavioural test cannot reach them without an engine, so those three are
parse-the-patch tests and are labelled STRUCTURAL where they are defined. They
exist for one specific failure: someone edits `_start_load_kv` for another
reason, the publish call falls out of the hunk, and the guard silently stops
being armed while every behavioural test in this file keeps passing. A
structural test is weaker than a behavioural one and says so; it is not absent.

THE ONE PROPERTY THE WHOLE FILE DEFENDS
---------------------------------------
Publish, then collision; un-publish, then no collision. If un-publishing did not
work the set would only grow, and the guard would raise on a serving path over a
load that finished forty steps ago - a false alarm, which on this box is an
outage. Both halves are tested with the real predicate on the other end.

Nothing here touches an engine, the GPU box, or a network.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time
import typing

import pytest

import test_recoverssm_load_collision_guard as guard_side

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAFT = ROOT / "patches" / "drafts" / "DRAFT-0007-recoverssm-load-collision-assertion"
CONNECTOR = DRAFT / "kv_connector.py.patch"
RUNNER = DRAFT / "model_runner.py.patch"

RecoverSSMState = guard_side.RecoverSSMState
Collision = guard_side.RecoverSSMLoadCollisionError


# --------------------------------------------------------------------------
# Extraction: the shipped bytes, or nothing.
# --------------------------------------------------------------------------
def _added(path: pathlib.Path) -> str:
    if not path.exists():  # pragma: no cover - the draft was moved or renamed
        pytest.skip(f"{path.relative_to(ROOT)} is not in the tree")
    return "".join(
        line[1:]
        for line in path.read_text().splitlines(keepends=True)
        if line.startswith("+") and not line.startswith("+++")
    )


ADDED = _added(CONNECTOR)

# The module constants the hunk adds. Read from the patch and never restated
# here, so changing the bound changes the test instead of drifting from it.
CONSTANTS = dict(
    (m.group(1), int(m.group(2)))
    for m in re.finditer(r"^(KVTIER_\w+) = (\d+)$", ADDED, re.M)
)


def _class_body() -> str:
    """The `ActiveKVConnector` guard methods: the LAST `set_load_collision_sink`
    def (the interface's stub is the first) through the end of the hunk, minus
    the module-level tail line."""
    marker = "    def set_load_collision_sink(self, model_state: Any) -> None:"
    assert ADDED.count(marker) == 2, (
        "expected the interface stub and the connector's own override; a rename "
        "here means these tests would exec a class the patch does not build"
    )
    body = ADDED[ADDED.rindex(marker):]
    body = body.split("NO_OP_KV_CONNECTOR = KVConnector()")[0]
    return body.rstrip("\n") + "\n"


class _SilentLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg, *args):  # noqa: ANN001, ANN002 - a stand-in
        self.lines.append(str(msg) % args if args else str(msg))

    warning = info


LOGGER = _SilentLogger()
NS: dict = {
    "__name__": "wiring_under_test",
    "Any": typing.Any,
    "os": os,
    "json": json,
    "logger": LOGGER,
    **CONSTANTS,
}
exec("class ActiveKVConnector:\n" + _class_body(), NS)
ActiveKVConnector = NS["ActiveKVConnector"]


# The hunk's `__init__` additions are FRAGMENTS inside a constructor that calls
# `get_kv_transfer_group()`, so they cannot be exec'd. This mirrors them, and
# `test_the_mirror_initialises_every_name_the_hunk_initialises` is what stops the
# mirror from inventing state the patch does not create.
def _connector(sink=None) -> "ActiveKVConnector":
    obj = ActiveKVConnector.__new__(ActiveKVConnector)
    obj._kvtier_guard_sink = sink
    obj._kvtier_inflight_loads = {}
    obj._kvtier_pending_load = None
    obj._kvtier_guard_counters = {
        "publishes": 0,
        "unpublishes": 0,
        "expired_unproven": 0,
        "pending_replaced": 0,
        "unprovable_events": 0,
        "pushes_to_guard": 0,
        "checks_at_submit": 0,
        "state_side_unknown": 0,
        "witness_writes": 0,
        "witness_write_failures": 0,
    }
    obj._kvtier_guard_rank_tag = None
    obj._kvtier_guard_armed_logged = False
    obj._kvtier_witness_events = None
    obj._kvtier_witness_changes = 0
    return obj


class _Meta:
    """Exactly the fields the hunk reads off `SimpleCPUOffloadMetadata`."""

    def __init__(self, load_event=-1, load_gpu_blocks=(), load_event_to_reqs=None):
        self.load_event = load_event
        self.load_gpu_blocks = list(load_gpu_blocks)
        self.load_event_to_reqs = dict(load_event_to_reqs or {})


def _submit(conn, event, blocks, reqs=("req-a",)) -> None:
    """The lifecycle exactly as the shipped call sites drive it: capture at bind
    time, publish at the submit."""
    conn._kvtier_capture_load_submit(
        _Meta(load_event=event, load_gpu_blocks=blocks,
              load_event_to_reqs={event: list(reqs)})
    )
    conn._kvtier_publish_inflight_loads()


def _sink(state_blocks=None) -> "RecoverSSMState":
    state = RecoverSSMState.__new__(RecoverSSMState)
    state._state_blocks = state_blocks
    state._inflight_load_blocks = None
    return state


# `guard_armed()` is in the shipped hunk but not in the sibling suite's carved
# host (it is a two-line read), so it is pinned by its body here and the same
# expression is evaluated below. A rename that turns it into something else
# fails the pin instead of being silently re-implemented by this file.
ARMED_IS_A_TRUTHINESS_TEST = re.search(
    r"def guard_armed\(self\)[^\n]*\n(?:.*\n)*?        return bool"
    r"\(self\._inflight_load_blocks\)",
    guard_side.SRC,
)


def _armed(sink) -> bool:
    assert ARMED_IS_A_TRUTHINESS_TEST is not None, (
        "`guard_armed` no longer reads as `return bool(self._inflight_load_blocks)" 
        "`; re-read the hunk instead of trusting this file's reading of it"
    )
    return bool(sink._inflight_load_blocks)


@pytest.fixture(autouse=True)
def _witness_in_tmp_path(tmp_path, monkeypatch):
    """Keep the witness file inside `tmp_path`. The hunk writes to
    `VLLM_E2_TIER_DIAG` when set and `/tmp` when not, and a test suite must
    neither litter `/tmp` nor read another process's file."""
    monkeypatch.setenv("VLLM_E2_TIER_DIAG", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------
# 1. THE WIRING EXISTS, AND IN THE RIGHT ORDER.
#    STRUCTURAL: a parse of the shipped patch, not a behaviour - see the header.
# --------------------------------------------------------------------------
def _hunk_with(needle: str) -> str:
    hunks = re.split(r"^@@[^\n]*\n", CONNECTOR.read_text(), flags=re.M)
    hits = [h for h in hunks if needle in h]
    assert hits, f"no hunk of the shipped patch contains {needle!r}"
    return hits[0]


def test_the_bind_site_captures_the_ids_it_is_about_to_launch() -> None:
    # STRUCTURAL. An ADDED line, after the metadata is bound, because the ids
    # come off that object. Before the sync-launch branch, which is why one call
    # covers the sync submit, the async one, and the `no_forward` path.
    hunk = _hunk_with("self._kvtier_capture_load_submit(")
    assert any(ln.startswith("+") and "_kvtier_capture_load_submit(" in ln
               for ln in hunk.splitlines()), "the capture must be added, not context"
    assert hunk.index("bind_connector_metadata(") < hunk.index(
        "self._kvtier_capture_load_submit(")
    assert hunk.index("self._kvtier_capture_load_submit(") < hunk.index(
        "if scheduler_output.has_sync_kv_loads:")


def test_the_submit_site_publishes_before_the_copy_is_launched() -> None:
    # STRUCTURAL, and the single most important line in this file: without it the
    # guard is inert and every behavioural test here still passes. The publish
    # must be inside `_start_load_kv` (the one funnel both load paths share) and
    # ahead of the call that launches the copy.
    hunk = _hunk_with("self._kvtier_publish_inflight_loads()")
    assert any(ln.startswith("+") and "_kvtier_publish_inflight_loads()" in ln
               for ln in hunk.splitlines())
    assert hunk.index("self._pending_load_kwargs = None") < hunk.index(
        "self._kvtier_publish_inflight_loads()"), "must run once per submit"
    assert hunk.index("self._kvtier_publish_inflight_loads()") < hunk.index(
        "self.kv_connector.start_load_kv("), (
        "the guard must see the destination before the copy is issued, so a "
        "collision stops the step before a byte moves"
    )


def test_the_completion_site_retires_from_the_engine_s_own_verdict() -> None:
    # STRUCTURAL. The finished/failed recving sets are the only completion proof
    # that reaches this process, so the retire must read them out of the SAME
    # `transfer_results` the output is built from - the call that asks the
    # connector what finished. It is placed with the lines that consume that
    # object, which is what the hunk can show; the arguments are values, not the
    # step's metadata, so it does not depend on the metadata still being bound.
    hunk = _hunk_with("self._kvtier_retire_completed(")
    assert any(ln.startswith("+") and "self._kvtier_retire_completed(" in ln
               for ln in hunk.splitlines())
    assert hunk.index("output.failed_recving = transfer_results.failed_recving") < hunk.index(
        "self._kvtier_retire_completed("), (
        "the retire must live in the block that consumes the connector's verdict"
    )
    call = hunk[hunk.index("self._kvtier_retire_completed("):].split(")")[0]
    assert "transfer_results.finished_recving" in call, call
    assert "transfer_results.failed_recving" in call, (
        "a failed load writes nothing more either, so it un-arms the same way"
    )


def test_the_runner_registers_the_sink_where_it_builds_the_connector() -> None:
    # STRUCTURAL, in the other patch. This is the only place in the engine that
    # holds both sides; if the call is dropped the connector never has a sink,
    # which the witness reports as `guard_not_armed_no_recover_ssm_sink` and
    # which is exactly the inert state this draft was written to end.
    if not RUNNER.exists():  # pragma: no cover
        pytest.skip(f"{RUNNER.relative_to(ROOT)} is not in the tree")
    text = RUNNER.read_text()
    added = [ln for ln in text.splitlines() if ln.startswith("+")]
    assert any("set_load_collision_sink(self.model_state)" in ln for ln in added), (
        "the model runner must register the sink at the connector's construction "
        "site - nothing else in the engine can reach both sides"
    )
    body = text[text.index("def initialize_kv_cache"):] if (
        "def initialize_kv_cache" in text) else text
    assert body.index("get_kv_connector(self.vllm_config") < body.index(
        "set_load_collision_sink(self.model_state)")


def test_the_registration_is_unconditional_and_the_no_op_connector_answers_it() -> None:
    # The call must sit at the `if is_profiling:` level, not inside the `else`,
    # so a profile boot registers "no sink" rather than skipping registration;
    # and the interface must answer the call, because a `hasattr` guard at the
    # call site would silently skip the arm on any connector refactor.
    text = CONNECTOR.read_text()
    stub = _hunk_with("class ActiveKVConnector(KVConnector):")
    assert "def set_load_collision_sink" in stub, (
        "KVConnector must carry the stub so the runner can register "
        "unconditionally"
    )
    runner_text = RUNNER.read_text() if RUNNER.exists() else ""
    assert "\n+        self.kv_connector.set_load_collision_sink(" in runner_text, (
        "the runner's call must be at eight spaces - outside the if/else"
    )


# --------------------------------------------------------------------------
# 2. THE MIRROR IS FAITHFUL.
# --------------------------------------------------------------------------
def test_the_mirror_initialises_every_name_the_hunk_initialises() -> None:
    import inspect

    mirrored = set(re.findall(r"obj\.(_kvtier_\w+)", inspect.getsource(_connector)))
    shipped = set(re.findall(r"self\.(_kvtier_\w+): [^=]+ = ", ADDED))
    assert shipped, "the hunk declares no guard state at all - read the patch"
    assert shipped <= mirrored, f"the hunk initialises {shipped - mirrored} that the " \
        "test's mirror does not, so the class under test is not the one shipping"


def test_the_constants_came_from_the_patch() -> None:
    assert CONSTANTS.get("KVTIER_LOAD_INFLIGHT_MAX_SUBMITS", 0) > 1, CONSTANTS
    assert "KVTIER_GUARD_WITNESS_EVERY" in CONSTANTS, CONSTANTS


# --------------------------------------------------------------------------
# 3. PUBLISH ARMS THE GUARD; THE COLLISION RAISES THROUGH THE SHIPPED PREDICATE.
# --------------------------------------------------------------------------
def test_a_load_still_in_flight_arms_the_commit_of_a_later_step() -> None:
    # The async path launches the copy AFTER this step's RecoverSSM commit, so
    # the collision it can create is one no commit of this step can see: the ids
    # must survive until the transfer is proven over, and arm the NEXT step's
    # commit. Clearing per step would be the silent direction of wrong.
    sink = _sink(state_blocks=None)     # nothing committed yet: the submit is quiet
    conn = _connector(sink=sink)
    _submit(conn, event=3, blocks=[900, 431])
    assert _armed(sink) is True
    assert not (sink._inflight_load_blocks and sink._state_blocks), (
        "the commit-side gate must stay shut while nothing is committed"
    )
    assert conn._kvtier_guard_counters["checks_at_submit"] == 0
    # ... and the commit side publishes on a later step, naming block 431.
    sink._state_blocks = {7: 431}
    assert sink._inflight_load_blocks and sink._state_blocks, (
        "the shipped gate in `commit_step` is `_inflight_load_blocks and "
        "_state_blocks`; both sides are published here"
    )
    with pytest.raises(Collision) as exc:
        RecoverSSMState.assert_no_load_collision(
            sink._state_blocks, sink._inflight_load_blocks, side="commit"
        )
    msg = str(exc.value)
    assert guard_side.MARKER in msg and "row 7" in msg and "block 431" in msg, msg
    assert "RecoverSSM commit is the SECOND" in msg, msg


def test_the_submit_side_raises_when_the_commit_side_got_there_first() -> None:
    # The other direction: the state blocks are already published when the load
    # is submitted, so the SUBMIT is the second side to see the collision, and
    # the publish itself must raise - not defer it to a commit that may never
    # happen.
    sink = _sink(state_blocks=[(7, 431)])
    conn = _connector(sink=sink)
    with pytest.raises(Collision) as exc:
        _submit(conn, event=3, blocks=[431])
    assert "load submit is the SECOND" in str(exc.value), str(exc.value)


def test_disjoint_ids_arm_the_guard_and_raise_nothing() -> None:
    sink = _sink(state_blocks={7: 431})
    conn = _connector(sink=sink)
    _submit(conn, event=3, blocks=[900, 901])
    assert sink._inflight_load_blocks == frozenset({900, 901})
    assert conn._kvtier_guard_counters["publishes"] == 1
    assert conn._kvtier_guard_counters["checks_at_submit"] == 1


# --------------------------------------------------------------------------
# 4. UN-PUBLISH: the property that keeps a guard off a serving path.
# --------------------------------------------------------------------------
def test_retiring_on_completion_disarms_the_guard() -> None:
    sink = _sink(state_blocks={})           # nothing committed yet: submit is quiet
    conn = _connector(sink=sink)
    _submit(conn, event=4, blocks=[431], reqs=("req-a",))
    assert _armed(sink) is True
    # The commit side publishes on a later step, and the armed set is what it
    # would intersect: the collision is live until the transfer is proven over.
    sink._state_blocks = {7: 431}
    assert sink._inflight_load_blocks and sink._state_blocks
    conn._kvtier_retire_completed({"req-a"}, set())
    assert conn._kvtier_inflight_loads == {}
    assert _armed(sink) is False, (
        "a completed load must stop arming the guard, or the set only grows and "
        "the guard fires later over a transfer that finished long ago"
    )
    assert not (sink._inflight_load_blocks and sink._state_blocks), (
        "the commit-side gate must close, or the next commit raises on stale ids"
    )
    assert conn._kvtier_guard_counters["unpublishes"] == 1


def test_a_load_shared_by_two_requests_is_not_retired_by_the_first() -> None:
    conn = _connector(sink=_sink(state_blocks={}))
    _submit(conn, event=5, blocks=[700], reqs=("req-a", "req-b"))
    conn._kvtier_retire_completed({"req-a"}, set())
    assert conn._kvtier_inflight_loads, "partial completion proves nothing"
    conn._kvtier_retire_completed({"req-b"}, set())
    assert conn._kvtier_inflight_loads == {}, (
        "both requests have now been reported done, in whichever order the "
        "engine happened to report them"
    )
    assert conn._kvtier_guard_counters["unpublishes"] == 1


def test_a_failed_load_retires_exactly_like_a_successful_one() -> None:
    # `failed_recving` is what the scheduler feeds `_handle_failed_recving`; to
    # this process it means the same thing - the copy will write nothing more.
    conn = _connector(sink=_sink(state_blocks={}))
    _submit(conn, event=6, blocks=[701], reqs=("req-c",))
    conn._kvtier_retire_completed(set(), {"req-c"})
    assert conn._kvtier_inflight_loads == {}


def test_nothing_retired_changes_nothing() -> None:
    conn = _connector(sink=None)
    _submit(conn, event=7, blocks=[702], reqs=("req-d",))
    conn._kvtier_retire_completed(set(), set())
    assert len(conn._kvtier_inflight_loads) == 1


def test_a_bound_load_that_never_submits_is_never_published() -> None:
    # `pre_forward` binds metadata on every step; only the submit makes ids in
    # flight. A step that binds and never launches must leave the guard cold.
    conn = _connector(sink=_sink(state_blocks={7: 431}))
    conn._kvtier_capture_load_submit(
        _Meta(load_event=9, load_gpu_blocks=[431], load_event_to_reqs={9: ["r"]})
    )
    assert conn._kvtier_inflight_loads == {}
    assert conn._kvtier_guard_sink._inflight_load_blocks is None


def test_a_stale_bind_note_does_not_leak_into_the_next_submit() -> None:
    # The step that bound a load is followed by a step that bound none: the
    # first step's destination must not be published as the second's.
    conn = _connector(sink=None)
    conn._kvtier_capture_load_submit(
        _Meta(load_event=1, load_gpu_blocks=[431], load_event_to_reqs={1: ["r"]})
    )
    conn._kvtier_capture_load_submit(_Meta())
    assert conn._kvtier_pending_load is None
    assert conn._kvtier_guard_counters["pending_replaced"] == 1


def test_an_unprovable_entry_expires_loudly_rather_than_forever() -> None:
    # Bounded on purpose: an entry whose completion is never proven would
    # otherwise keep the guard armed for the rest of the boot. Expiry is an
    # UNDER-report, so it is counted and must be visible, not absorbed.
    conn = _connector(sink=None)
    bound = CONSTANTS["KVTIER_LOAD_INFLIGHT_MAX_SUBMITS"]
    _submit(conn, event=1, blocks=[800], reqs=("req-e",))
    for step in range(bound):
        conn._kvtier_capture_load_submit(_Meta())
        conn._kvtier_publish_inflight_loads()
    assert conn._kvtier_inflight_loads == {}, conn._kvtier_inflight_loads
    assert conn._kvtier_guard_counters["expired_unproven"] == 1


# --------------------------------------------------------------------------
# 5. THE GUARD'S EXCEPTION IS NOT A DIAGNOSTIC.
# --------------------------------------------------------------------------
def test_a_collision_is_never_swallowed_by_the_counter_or_witness_path() -> None:
    class _Boom:
        _state_blocks = {3: 42}

        def publish_inflight_load_blocks(self, blocks, *, side="load_submit"):
            raise Collision("boom through the wiring")

    conn = _connector(sink=_Boom())
    with pytest.raises(Collision):
        _submit(conn, event=2, blocks=[42])
    # And the state it was accumulating is still recorded: the raise must not
    # look like a rollback.
    assert conn._kvtier_inflight_loads


def test_a_witness_that_cannot_be_written_still_serves(monkeypatch) -> None:
    # A diagnostic never outranks serving - but it must SAY it failed, or an
    # unwritten witness becomes an assumption of "armed, nothing found".
    monkeypatch.setenv("VLLM_E2_TIER_DIAG", "/proc/definitely/not/a/directory")
    conn = _connector(sink=_sink(state_blocks={}))
    _submit(conn, event=8, blocks=[900], reqs=("req-f",))
    assert conn._kvtier_guard_counters["witness_write_failures"] == 1
    assert conn._kvtier_inflight_loads, "the guard kept working despite the failure"
    assert any("RECOVERSSM-GUARD witness" in line for line in LOGGER.lines), LOGGER.lines


# --------------------------------------------------------------------------
# 6. ARMED IS OBSERVABLE, AND A ZERO IS NOT A PASS.
# --------------------------------------------------------------------------
def _witness(tmp_path, conn):
    path = conn._kvtier_guard_witness_path()
    assert pathlib.Path(path).parent == tmp_path
    return json.loads(pathlib.Path(path).read_text())


def test_the_witness_distinguishes_no_sink_from_no_loads_from_loads(tmp_path) -> None:
    # Three states a reader must not confuse, and none of them differs in a
    # number: no RecoverSSM to arm, armed with nothing submitted, and armed with
    # loads seen. The verdict is WRITTEN DOWN, not inferred from a zero.
    no_sink = _connector(sink=None)
    _submit(no_sink, event=1, blocks=[10], reqs=("r1",))
    assert _witness(tmp_path, no_sink)["verdict"] == "guard_not_armed_no_recover_ssm_sink"

    armed = _connector(sink=_sink())
    armed._kvtier_write_guard_witness(force=True)
    assert _witness(tmp_path, armed)["verdict"] == "armed_no_load_submit_yet"

    loaded = _connector(sink=_sink(state_blocks={}))
    _submit(loaded, event=2, blocks=[11], reqs=("r2",))
    seen = _witness(tmp_path, loaded)
    assert seen["verdict"] == "armed_loads_observed", seen
    assert seen["counters"]["publishes"] == 1
    assert seen["in_flight_blocks"] == 1


def test_the_witness_path_names_the_process_and_never_impersonates_a_rank_file(tmp_path) -> None:
    conn = _connector(sink=None)
    name = os.path.basename(conn._kvtier_guard_witness_path())
    assert name.startswith("recoverssm-guard-"), name
    assert not name.startswith("rank-"), (
        "the reader that walks this directory keys on `rank-`; a guard file that "
        "looked like a rank document would be parsed as one"
    )
    assert name.endswith(".json")
    assert name != "recoverssm-guard-.json"  # a tag is always filled in


def test_the_witness_is_sampled_not_per_step(tmp_path) -> None:
    # The hunk runs on the step loop, so it may not turn a diagnostic into
    # per-step file I/O: a rewrite happens when the outstanding count moves or
    # after a bound of accumulated changes, and NOT on every push.
    conn = _connector(sink=_sink(state_blocks={}))
    _submit(conn, event=1, blocks=[1], reqs=("a",))
    before = conn._kvtier_guard_counters["witness_writes"]
    assert before == 1, "the first submit must produce a witness at all"
    for _ in range(50):
        conn._kvtier_push_guard()
    assert conn._kvtier_guard_counters["witness_writes"] == before, (
        "an unchanged guard rewrote the file 50 times"
    )
    bound = CONSTANTS["KVTIER_GUARD_WITNESS_EVERY"]
    conn._kvtier_witness_changes = bound - 1
    conn._kvtier_write_guard_witness()
    assert conn._kvtier_guard_counters["witness_writes"] == before
    conn._kvtier_witness_changes = bound
    conn._kvtier_write_guard_witness()
    assert conn._kvtier_guard_counters["witness_writes"] == before + 1


def test_a_missing_witness_file_is_a_different_claim_than_a_zero(tmp_path) -> None:
    # Documented contract: no file at all means no load submit has ever run in
    # this process - NOT "the guard ran and found nothing". Pinned so a future
    # change cannot quietly make absence and emptiness the same state.
    conn = _connector(sink=_sink())
    assert not os.path.exists(conn._kvtier_guard_witness_path())
    conn._kvtier_write_guard_witness(force=True)
    payload = _witness(tmp_path, conn)
    assert payload["verdict"] == "armed_no_load_submit_yet", payload
    assert payload["counters"]["publishes"] == 0
    assert "NOT the same claim" in payload["meaning"], (
        "the file must carry its own reading rule: an operator finds it without "
        "finding this test"
    )


# --------------------------------------------------------------------------
# 7. THE UNARMED PATH COSTS NOTHING, AGAINST A REFERENCE MEASURED ALONGSIDE IT.
# --------------------------------------------------------------------------
def test_a_step_with_no_load_and_nothing_outstanding_does_almost_nothing() -> None:
    # A RELATIVE reference, timed in the same loop on the same box: the claim is
    # "an ordinary step pays about one Python call here", and a constant ceiling
    # copied from another machine is not an instrument.
    conn = _connector(sink=_sink())
    n = 200_000

    def noop(_self):
        return None

    t0 = time.perf_counter()
    for _ in range(n):
        noop(conn)
    base = (time.perf_counter() - t0) / n * 1e6

    t0 = time.perf_counter()
    for _ in range(n):
        conn._kvtier_publish_inflight_loads()
    guard = (time.perf_counter() - t0) / n * 1e6
    assert guard < base + 1.0, f"unarmed path {guard:.3f}us vs a bare call {base:.3f}us"


def test_registration_takes_the_guard_off_the_model_state_or_says_it_refused(tmp_path) -> None:
    class _Model:
        pass

    conn = _connector()
    conn.set_load_collision_sink(_Model())          # no `recoverssm` at all
    assert conn._kvtier_guard_sink is None
    assert _witness(tmp_path, conn)["verdict"] == "guard_not_armed_no_recover_ssm_sink"

    without_the_method = _Model()
    without_the_method.recoverssm = object()        # a member that is not a guard
    conn.set_load_collision_sink(without_the_method)
    assert conn._kvtier_guard_sink is None, (
        "registering something without `publish_inflight_load_blocks` would look "
        "armed and raise nothing, ever"
    )

    armed = _sink()
    with_guard = _Model()
    with_guard.recoverssm = armed
    conn.set_load_collision_sink(with_guard)
    assert conn._kvtier_guard_sink is armed
    assert _witness(tmp_path, conn)["verdict"] == "armed_no_load_submit_yet"
