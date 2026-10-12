r"""The RecoverSSM / KV-connector collision guard (DRAFT-0007), tested on its own bytes.

WHAT THIS SUITE IS FOR
----------------------
`--use-replayssm` and a KV connector on one boot is refused today by
`vllm/config/vllm.py::validate_mamba_cached_kernel` (DRAFT-0006 asks to relax that).
Relaxing a refusal is a permission; this suite is about the thing that has to exist
beside the permission: a guard that FIRES if the two mechanisms ever write the same
state block in one step, because both outcomes of that collision are a plausible
continuation and neither raises anywhere else in the engine - the deferred RecoverSSM
commit overwrites state a load just restored with one derived from a checkpoint that
predates the load, or the load overwrites a state the commit had already advanced.
This repo has live evidence of that failure class (an unsound KV claim producing a
fluent `Register Register Register...` loop at temperature=0 where the recomputed
answer was correct), so a model answer cannot be the witness and neither can a
counter. The witness is an exception that cannot be misread.

WHY IT IS TESTABLE AT ALL, AND HOW THE TEST CANNOT DRIFT
--------------------------------------------------------
The guard lives in the ENGINE tree (`vllm/v1/worker/gpu/model_states/recoverssm.py`),
which this repo does not import - so the obvious way to write this file is to
re-implement the predicate here and test the re-implementation. That test would stay
green while the shipped hunk rotted, which is the specific lie this repo has incident
history for. Instead every byte under test is EXTRACTED FROM THE DRAFT'S PATCH FILE:
the `+` lines are carved out and executed, so a change to the hunk changes what these
tests run, and a hunk that stops carrying the guard fails here instead of silently
passing a test of a function nobody ships.

The carve is textual on purpose. The patched module imports `torch`, `triton` and
`vllm.*` at module scope, so it cannot be imported here at all (the plugin imports
without an engine - see `conftest.py`); the guard's own code needs none of them, only
`typing.Any` for its annotations, which is why it can be exec'd in a bare namespace.

THE ORDER A REVIEWER SHOULD READ IT IN:

1. THE GUARD RAISES, AND NAMES THE ROW, THE BLOCK AND THE SIDE. Not an `assert`:
   a bare `assert` vanishes under `python -O`, and this is a production-path
   invariant, so one test pins that the shipped lines contain no statement-level
   `assert` and another pins the exception type and its message contents.
2. BOTH DIRECTIONS ARE CHECKED. Whoever sees the collision second must raise: the
   commit (connector ids published first) and the submit hook (state blocks
   published first). A guard on one side only is silent on the other, and which
   side is second is a property of the step order, not a choice.
3. UNPUBLISHED MEANS INERT, NOT COLLIDING. Nothing is published until the two
   call-site hunks land, so the guard's default state must never raise - and the
   test pins the sentence that keeps that honest (`__init__` sets both to None).
4. THE COMMIT-SIDE GUARD RUNS BEFORE ANY STATE IS WRITTEN. The hunk puts the check
   ahead of the `for metadata in step:` loop that calls `commit_recoverssm_state`;
   after it, the collision would already have happened.
5. THE FAST PATH IS CHEAP, MEASURED AGAINST A CONCURRENT REFERENCE. The step loop
   runs thousands of times a minute on the thread that steps the engine - the
   plugin's dev-door lesson is that work here stalls serving - so the no-load
   fast path is priced against a no-op call measured in the same loop on the same
   box, never against a constant ceiling copied from someone else's machine.
6. THE PREDICATE DOES NOT ALLOCATE PER STEP. `tracemalloc` over many no-collision
   calls, because the shipped comment claims it and a set comprehension would
   silently contradict that comment.

Nothing here touches an engine, the GPU box, or a network.
"""

from __future__ import annotations

import pathlib
import re
import time
import tracemalloc
import typing

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAFT = ROOT / "patches" / "drafts" / "DRAFT-0007-recoverssm-load-collision-assertion"
PATCH = DRAFT / "recoverssm.py.patch"

MARKER = "RECOVERSSM/KV-CONNECTOR STATE COLLISION"


# --------------------------------------------------------------------------
# Extraction: the shipped bytes, or nothing.
# --------------------------------------------------------------------------
def _added_by_hunk() -> list[str]:
    """Added lines (marker stripped) grouped by `@@` hunk, in file order."""
    if not PATCH.exists():  # pragma: no cover - the draft was moved or renamed
        pytest.skip(f"{PATCH.relative_to(ROOT)} is not in the tree")
    hunks: list[list[str]] = []
    cur: list[str] | None = None
    for line in PATCH.read_text().splitlines(keepends=True):
        if line.startswith("@@"):
            cur = []
            hunks.append(cur)
            continue
        if cur is None or line.startswith("+++") or not line.startswith("+"):
            continue
        cur.append(line[1:])
    return ["".join(h) for h in hunks if h]


HUNKS = _added_by_hunk()
SRC = "".join(HUNKS)


def _hunk_containing(needle: str) -> str:
    for hunk in HUNKS:
        if needle in hunk:
            return hunk
    pytest.fail(f"no hunk of the shipped patch contains {needle!r} - the guard was "
                "renamed or removed and these tests are testing nothing")
    raise AssertionError  # unreachable; quiets the type checker


def _carve_member(text: str, first: str) -> str:
    """One class member: its decorators and body, bounded by the next sibling
    member or by the end of the hunk (the last member in a hunk has no next)."""
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith(first))
    except StopIteration:
        pytest.fail(f"the shipped hunk no longer contains {first!r}")
        raise AssertionError
    while lines[start - 1].startswith("    @"):  # keep decorators
        start -= 1
    sibling = re.compile(r"^    (@|def |async def |class )")
    # Scan from AFTER the `def` line: a decorator rewind puts the member's own `def`
    # at start+1, and that line is not its own next sibling.
    end = next(i for i in range(start, len(lines)) if lines[i].lstrip().startswith(first))
    end += 1
    while end < len(lines) and not (lines[end].strip() and sibling.match(lines[end])):
        end += 1
    return "".join(lines[start:end])


def _carve_exception(text: str) -> str:
    """The module-level exception class, from its `class` line through its own
    `MARKER` assignment - the anchor that bounds it, since the lines added after it
    in the same file are INDENTED and so cannot be bounded by column 0."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("class RecoverSSMLoadCollisionError"))
    end = next(i for i, ln in enumerate(lines[start:], start)
               if ln.lstrip().startswith("MARKER = "))
    return "".join(lines[start:end + 1])


# The exception class and the guard, exec'd from the patch's own bytes.
_EXC_SRC = _carve_exception(_hunk_containing("class RecoverSSMLoadCollisionError"))
_MEMBERS = _hunk_containing("def find_load_collision(")
_STATIC_SRC = _carve_member(_MEMBERS, "def find_load_collision(")
_CLASSMETHOD_SRC = _carve_member(_MEMBERS, "def assert_no_load_collision(")
_PUBLISH_SRC = _carve_member(_MEMBERS, "def publish_state_blocks(")
_PUBLISH2_SRC = _carve_member(_MEMBERS, "def publish_inflight_load_blocks(")

# The host is the class the hunk patches; its body is the hunk's methods verbatim.
# `__init__`/`reset` are FRAGMENTS in the hunk (context lines around a few added
# assignments) and cannot be exec'd, so the host mirrors them - and
# `test_init_declares_the_two_id_sets` pins that the mirror is still faithful, which
# is what stops this file from testing a class the patch does not build.
HOST = (
    "class RecoverSSMState:\n"
    + _STATIC_SRC
    + _CLASSMETHOD_SRC
    + _PUBLISH_SRC
    + _PUBLISH2_SRC
)
NS: dict = {"__name__": "guard_under_test", "Any": typing.Any}
exec(_EXC_SRC + "\n" + HOST, NS)

RecoverSSMState = NS["RecoverSSMState"]
RecoverSSMLoadCollisionError = NS["RecoverSSMLoadCollisionError"]


def _host(state=None, inflight=None) -> "RecoverSSMState":
    obj = RecoverSSMState.__new__(RecoverSSMState)
    obj._state_blocks = state
    obj._inflight_load_blocks = inflight
    return obj


# --------------------------------------------------------------------------
# 1. It raises, and the message is an instruction, not a shrug.
# --------------------------------------------------------------------------
def test_finds_the_first_row_and_block_claimed_by_both_sides() -> None:
    committed = {7: 431, 8: 432, 9: 433}
    hit = RecoverSSMState.find_load_collision(committed, frozenset({433, 900}))
    assert hit == (9, 433)


def test_no_collision_is_none_and_names_nothing() -> None:
    assert RecoverSSMState.find_load_collision({7: 431}, frozenset({432})) is None


def test_accepts_pairs_as_well_as_a_mapping() -> None:
    # The caller with a GPU-resident gather has pairs, not a dict; both must work.
    assert RecoverSSMState.find_load_collision([(3, 55), (4, 56)], {56}) == (4, 56)


def test_raises_a_real_exception_naming_row_block_and_side() -> None:
    with pytest.raises(RecoverSSMLoadCollisionError) as exc:
        RecoverSSMState.assert_no_load_collision(
            {7: 431}, frozenset({431}), side="commit"
        )
    msg = str(exc.value)
    assert MARKER in msg, "an operator must be able to tell this from any other crash"
    assert " row 7 " in msg, msg
    assert "block 431" in msg, msg
    assert "SECOND side" in msg, msg


def test_the_two_sides_blame_different_sides() -> None:
    with pytest.raises(RecoverSSMLoadCollisionError) as a:
        RecoverSSMState.assert_no_load_collision({7: 431}, {431}, side="commit")
    with pytest.raises(RecoverSSMLoadCollisionError) as b:
        RecoverSSMState.assert_no_load_collision({7: 431}, {431}, side="load_submit")
    assert "RecoverSSM commit is the SECOND" in str(a.value), str(a.value)
    assert "load submit is the SECOND" not in str(a.value), str(a.value)
    assert "load submit is the SECOND" in str(b.value), str(b.value)
    assert "RecoverSSM commit is the SECOND" not in str(b.value), str(b.value)


# --------------------------------------------------------------------------
# A bare assert would vanish under `python -O`. This is not one.
# --------------------------------------------------------------------------
def test_the_carve_actually_got_the_guard_it_claims_to_run() -> None:
    # If the extractor ever returns an empty or wrong block, the tests below would
    # exec a class that is not the shipped guard - so the extraction is pinned first.
    assert "MARKER" in _EXC_SRC and _EXC_SRC.startswith("class RecoverSSMLoadCollisionError")
    for src, name in ((_STATIC_SRC, "find_load_collision"),
                      (_CLASSMETHOD_SRC, "assert_no_load_collision"),
                      (_PUBLISH_SRC, "publish_state_blocks"),
                      (_PUBLISH2_SRC, "publish_inflight_load_blocks")):
        assert f"def {name}(" in src, name
    # A member block must not run past its own member into the next one.
    assert "def publish_state_blocks" not in _CLASSMETHOD_SRC
    assert "side=\"commit\"" not in _PUBLISH2_SRC


def test_is_a_real_exception_type_not_an_assert() -> None:
    assert issubclass(RecoverSSMLoadCollisionError, RuntimeError)
    # `raise X` survives `python -O`; `assert` does not. Pinned on the shipped text,
    # not on behaviour, because the difference only shows up on an optimised boot.
    assert "raise RecoverSSMLoadCollisionError(" in SRC
    statement_asserts = [
        ln for ln in SRC.splitlines() if re.match(r"^\s*assert\b", ln)
    ]
    assert statement_asserts == [], statement_asserts


# --------------------------------------------------------------------------
# 2/3. Both directions, and the inert-until-published contract.
# --------------------------------------------------------------------------
def test_publish_reports_and_checks_the_ids_it_was_given() -> None:
    host = _host(state={7: 431})
    with pytest.raises(RecoverSSMLoadCollisionError):
        host.publish_inflight_load_blocks([900, 431])  # list in, checked
    assert host._inflight_load_blocks == frozenset({900, 431})


def test_publish_of_disjoint_ids_is_quiet_and_stores_them() -> None:
    host = _host(state={7: 431})
    host.publish_inflight_load_blocks([900, 901])
    assert host._inflight_load_blocks == frozenset({900, 901})


def test_nothing_published_collides_with_nothing() -> None:
    # Before the two call-site hunks land, nothing publishes: the guard must be
    # inert, not a boot-time crash. `None` on both sides is the shipped default.
    host = _host()
    assert host._inflight_load_blocks is None and host._state_blocks is None
    if host._inflight_load_blocks and host._state_blocks:  # the commit-side gate
        pytest.fail("unpublished state must not arm the guard")


def test_empty_inflight_is_not_a_collision() -> None:
    host = _host(state={7: 431}, inflight=frozenset())
    if host._inflight_load_blocks and host._state_blocks:
        pytest.fail("an empty in-flight set must leave the step alone")


def test_init_declares_the_two_id_sets() -> None:
    # The host above mirrors the hunk's `__init__` additions; this pins the mirror.
    # If the hunk stops declaring either name, the host is inventing state and the
    # rest of this file is testing a class the patch does not build.
    assert "_state_blocks" in SRC and "_inflight_load_blocks" in SRC
    for name in ("_state_blocks", "_inflight_load_blocks"):
        assert re.search(rf"self\.{name}(: Any)? = None", SRC), name


# --------------------------------------------------------------------------
# 4. The commit-side guard precedes the write it is guarding.
# --------------------------------------------------------------------------
def test_the_commit_guard_sits_before_the_commit_loop() -> None:
    hunk = PATCH.read_text()
    guard = hunk.index("side=\"commit\"")
    loop = hunk.index("for metadata in step:")
    write = hunk.index("postprocess_meta = metadata.commit_recoverssm_state")
    assert guard < loop < write, (
        "the guard must run before commit_recoverssm_state writes the row"
    )


def test_the_hunk_lands_in_the_class_the_fingerprinter_hashes() -> None:
    # kvtier-patch.sh fingerprints METHOD bodies of `klass`; a hunk that drifted to a
    # module-level function would report UNKNOWN provenance instead of a verdict.
    text = PATCH.read_text()
    # The hunk header's function-context line names the enclosing class, so
    # `klass=RecoverSSMState` is a reading of the patch, not a wish.
    assert re.search(r"^@@ .* @@ class RecoverSSMState:", text, re.M), (
        "every hunk must sit inside the klass the manifest declares"
    )
    # Which METHOD, proven by adjacency rather than by a line number (this repo bans
    # line-number citations): the guard shares a hunk with `commit_step`'s own PP
    # filter - a CONTEXT line, so it is the pre-image speaking, not the patch - and it
    # sits ahead of that method's commit loop. Only `commit_step` has both lines.
    raw = re.split(r"^@@[^\n]*\n", PATCH.read_text(), flags=re.M)
    with_guard = [b for b in raw if 'side="commit"' in b]
    assert len(with_guard) == 1, "the commit-side guard must live in exactly one hunk"
    body = with_guard[0]
    assert "num_sampled = torch.where(" in body, (
        "the guard's hunk must be anchored in commit_step's own body"
    )
    assert body.index('side="commit"') < body.index("for metadata in step:")


def test_the_commit_guard_is_an_added_line_not_mere_context() -> None:
    added = [ln for ln in PATCH.read_text().splitlines() if ln.startswith("+")]
    assert any('self.assert_no_load_collision(' in ln for ln in added), added[:3]
    assert any(ln.startswith("+        if self._inflight_load_blocks") for ln in added)


# --------------------------------------------------------------------------
# 5/6. Hot-path cost, against a reference measured in the same loop.
# --------------------------------------------------------------------------
def _timed(fn, args, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return (time.perf_counter() - t0) / n * 1e6


def test_fast_path_costs_about_one_python_call() -> None:
    # A RELATIVE reference, measured in the same loop on the same box: the claim is
    # "the guard is a few Python calls", and a constant ceiling copied from another
    # machine is not an instrument. A no-op method call is the floor.
    host = _host()
    n = 200_000

    def noop(_self):
        return None

    base = _timed(noop, (host,), n)
    armed = _timed(
        lambda h: (h._inflight_load_blocks and h._state_blocks), (host,), n
    )
    assert armed < base + 0.5, f"fast path {armed:.3f}us vs a bare call {base:.3f}us"


def test_scanned_path_is_bounded_by_the_rows_it_scans() -> None:
    # Worst realistic case: every request slot committed, a full window in flight,
    # and no collision (so the whole scan runs). Priced against the same-shape loop
    # doing nothing but the membership test, so a slow box cannot fail this.
    state = {r: 1000 + r for r in range(512)}
    inflight = frozenset(range(50_000, 54_096))
    n = 5_000
    guard = _timed(
        lambda: RecoverSSMState.find_load_collision(state, inflight), (), n
    )
    floor = _timed(lambda: all(b not in inflight for b in state.values()), (), n)
    assert guard < 4 * floor + 20, f"scan {guard:.3f}us vs floor {floor:.3f}us"


def test_the_predicate_allocates_nothing_per_step() -> None:
    # The shipped comment says a set comprehension was rejected because it allocates
    # per step. This is the witness for that sentence.
    state = {r: 1000 + r for r in range(512)}
    inflight = frozenset(range(50_000, 50_320))
    tracemalloc.start()
    for _ in range(20_000):
        RecoverSSMState.find_load_collision(state, inflight)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 4096, f"per-step allocation of {peak} bytes contradicts the comment"
