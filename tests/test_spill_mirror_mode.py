r"""MIRROR MODE: a block that lands in the arena is queued for NVMe AT ONCE.

THE FAILURE THIS EXISTS FOR (measured 2026-10-12). The proxy STOPS the backend in
order to suspend the box, so every prefix still resident in the host-RAM arena when
nobody needed the slots dies with the process. One conversation - 207,861 tokens at
01:22, returned at 16:34 after a single 15-hour suspend - came back with **0.0%
reuse**, `reads_applied` 0 for the whole fresh process, while the spill path had
evicted NOTHING. There was no data to lose because nothing had been written: a block
reached NVMe only when the arena needed its slot back.

Mirror mode makes the arena a BUFFER instead of a store: the store-completion
hand-off queues its `(cpu_slot, block_hash)` pairs to the SAME spill queue the
evictor uses, and eviction stays purely the capacity valve. The cost was measured
before the decision (`slots_offered 3,076` against `spill_queued 2,556`: ~1.20x today's
write volume), so this file does not re-litigate the design. It pins the four things
that could make it an outage:

1. THE DOOR IS AN HONEST ENVIRONMENT DOOR. Default ON; a disable word closes it; a
   value in neither vocabulary leaves it OPEN and says so, the same grammar as
   `config.mamba_anchor_patch()` and `disk_restore.staging_rows_verdict()`. A refused
   value is reported with its complaint and the default stands - never a crash.
2. IT CANNOT WRITE PAST THE CEILING. The live store sits at 961 GB of a 1 TB soft
   ceiling on a filesystem with ~400 GB free, and an unparseable ceiling means NO
   ceiling (`disk_spill.capacity_ceiling_bytes`: "0" would mean "evict everything").
   So a mirror offer that would take this rank's accountable bytes past the ceiling is
   REFUSED AT THE DOOR: `request_spill` is never called, no bytes are copied, no file
   is written, and the refusal is counted under its own name. A mirror that outruns
   the evictor and fills `/home` is the only way this change becomes an outage.
3. IT CANNOT BLOCK THE SERVING PATH. The door's cost is one arithmetic comparison and
   the queue's own `put_nowait`. A full queue, or a writer that is not running, is a
   COUNTED refusal - a request never waits on NVMe.
4. IT IS SPILL-SIDE ONLY. Structurally, not by behaviour: the door's own source names
   no serve posture, no offer switch, no tier arm and no spill arm, and it cannot make
   a request be served. The AST pins below are the same class of pin
   `tests/test_mamba_anchor_patch.py` puts on the anchor patch, and for the same
   reason: behavioural equality over one run cannot rule out a door that fires on
   another path.

Plus the plumbing that has burned this project before: a counter that is not in the
rank-file whitelist is SILENTLY NEVER ROLLED UP (the span-completeness panel was
invisible for a day that way), so every new number is asserted in the writer, in the
whitelist and in a rollup built from two rank files.
"""

from __future__ import annotations

import ast
import inspect
import logging
import time

import pytest
import test_disk_store as store_fixtures  # the module the fixtures live in
from test_disk_spill import SLOT_BYTES
from test_disk_spill_rank_rollup import disk_section, write_rank
from test_disk_store import handler, metadata, metadata_shape, store_files

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    backport,
    config,
    disk_restore,
    disk_spill,
    disk_store,
)

# THE FIXTURES, RE-EXPORTED BY NAME (the `test_spill_span_completeness.py` idiom):
# `none_hash_seed` in particular is the fixture that cannot fail - without the fork's
# `get_none_hash_seed()` module the fingerprint builder refuses every bind, no spill
# path runs, and a test of an unreachable counter goes green.
none_hash_seed = store_fixtures.none_hash_seed  # noqa: F811 - autouse: the fingerprint needs the fork's seed module
no_fake_distributed_state = store_fixtures.no_fake_distributed_state  # noqa: F811 - autouse: no fake parallel_state leaks into a rank answer
tier = store_fixtures.tier  # noqa: F811 - the configured, ARMED store under tmp_path
unpatched_engine = store_fixtures.unpatched_engine  # noqa: F811 - autouse

KEY_A = "a3f0b7c1d4e5f607" + "1" * 48 + "00000003"
KEY_B = "0f1e2d3c4b5a6978" + "2" * 48 + "00000003"
KEY_C = "1111222233334444" + "0" * 48 + "00000003"


def spill_a_handoff(role, pairs, event=7):
    """Bind one step, complete it, and return the hook's report (the fork's order)."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[slot for slot, _key in pairs],
                        store_keys=[key for _slot, key in pairs],
                        store_event=event)
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        return disk_store.on_store_completed(role, [])


def mirror_state() -> dict:
    counters = disk_store.counters()
    return {name: counters[name] for name in
            ("spill_on_store_queued", "spill_on_store_bytes",
             "spill_on_store_refused", "spill_on_store_refused_total")}


# --- 1. the door: default ON, honest about who set it -------------------------

def test_the_mirror_door_is_open_by_default_and_names_its_door():
    """Unset means ON, and the verdict says WHICH door said so.

    ON is the whole point of the release, so it is asserted with nothing configured:
    no store, no arm, no hand-off. A boot that inherited nothing from its environment
    must be the boot the release ships.
    """
    verdict = disk_store.spill_on_store_verdict()
    assert verdict["enabled"] is True
    assert verdict["source"] == "default"
    assert verdict["raw"] is None
    assert verdict["recognised"] is True
    assert verdict["env_var"] == config.ENV_DISK_SPILL_ON_STORE
    assert verdict["problems"] == []
    # The door's own default is a named constant, not a literal in the payload, so a
    # flip is a visible diff rather than a word in a dict.
    assert verdict["default"] is disk_store.SPILL_ON_STORE_DEFAULT
    assert disk_store.SPILL_ON_STORE_DEFAULT is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "OFF", " 0 "])
def test_a_disable_word_closes_the_door_and_says_env(monkeypatch, value):
    """`0` is the rollback, and it is `source: "env"` so a rank file can tell an
    operator's NO from nobody's answer."""
    monkeypatch.setenv(config.ENV_DISK_SPILL_ON_STORE, value)
    verdict = disk_store.spill_on_store_verdict()
    assert verdict["enabled"] is False
    assert verdict["source"] == "env"
    assert verdict["recognised"] is True
    assert verdict["raw"] == value


@pytest.mark.parametrize("value", ["flase", "2", "nope", "-1"])
def test_an_unrecognised_word_leaves_the_door_open_and_reports_the_complaint(
        monkeypatch, value):
    """A typo does not close the door, and the payload says the word was not read.

    The direction is the one that keeps spilling, for `disk_boot_enable()`'s reason:
    a reader must see that the value he wrote is not a disable word, rather than being
    handed a payload that reads as intent. The complaint travels in `problems` and the
    raw word in `raw`; nothing raises.
    """
    monkeypatch.setenv(config.ENV_DISK_SPILL_ON_STORE, value)
    verdict = disk_store.spill_on_store_verdict()
    assert verdict["enabled"] is True
    assert verdict["source"] == "default"
    assert verdict["recognised"] is False
    assert verdict["raw"] == value
    assert verdict["problems"], "an unrecognised value must arrive with a complaint"
    assert config.ENV_DISK_SPILL_ON_STORE in verdict["problems"][0]


# --- 2. a store completion queues its pairs ----------------------------------

def test_a_store_completion_queues_its_pairs_through_the_mirror_door(tier):
    """The hand-off's pairs reach the queue, and the mirror's own counters see them.

    Both halves matter: `spill_queued` already existed (it counted the queue), while
    `spill_on_store_queued`/`_bytes` are the door's tally - the number that answers
    "is mirror mode actually running on this rank, and at what byte cost", which is
    the ratio the design was priced at.
    """
    role = handler()
    report = spill_a_handoff(role, [(6, KEY_A), (7, KEY_B)])
    assert report["queued"] == 2, report
    state = disk_store.counters()
    assert state["slots_offered"] == 2
    assert state["spill_queued"] == 2
    assert state["spill_on_store_queued"] == 2
    assert state["spill_on_store_bytes"] == 2 * SLOT_BYTES
    assert state["spill_on_store_refused_total"] == 0
    # The bytes really land: the door queues, the writer writes.
    assert disk_spill.drain(timeout_seconds=10.0)["drained"] is True
    assert len(store_files(tier)) == 2


def test_the_door_closed_by_environment_lets_no_pair_through_at_all(tier,
                                                                     monkeypatch):
    """Door off: the spill module is never even ASKED, and nothing is started.

    `disk_spill.counters()["requests"] == 0` is the assertion that matters, and it is
    the one an unpatched tree fails: counting only "no file was written" would also
    pass if the pairs had been queued and dropped. With the door closed no snapshot is
    copied, no writer thread starts, no file appears, and the refusal is counted under
    the door's own `disabled` reason.
    """
    monkeypatch.setenv(config.ENV_DISK_SPILL_ON_STORE, "0")
    role = handler()
    report = spill_a_handoff(role, [(6, KEY_A), (7, KEY_B)])
    assert report["queued"] == 0, report
    state = disk_store.counters()
    assert state["spill_queued"] == 0
    assert state["spill_on_store_queued"] == 0
    assert state["spill_on_store_bytes"] == 0
    # The hand-off census still counts the slots: `slots_offered` is the denominator
    # the mirror ratio is read against, and a closed door must not shrink it.
    assert state["slots_offered"] == 2
    assert state["spill_on_store_refused"]["disabled"] == 2
    assert state["spill_on_store_refused_total"] == 2
    # `delegated_refusals` counts refusals the SPILL PATH answered. The door answered
    # these, so the two counters stay separable.
    assert state["delegated_refusals"] == 0
    spill = disk_spill.counters()
    assert spill["requests"] == 0, "the door let a pair through to the spill module"
    assert spill["thread_started"] == 0, "a closed door started the writer thread"
    assert store_files(tier) == {}


# --- 3. the ceiling: refused at the door, nothing written ---------------------

def test_a_mirror_offer_at_the_ceiling_is_refused_and_writes_nothing(
        tier, monkeypatch):
    """A mirror offer that would take the rank past its ceiling is never queued.

    The ceiling is the change's one real risk (961 GB of a 1 TB soft ceiling, ~400 GB
    free on the filesystem behind it), so this test names the numbers rather than a
    boolean: the store is already AT the ceiling, one more slot would exceed it, and
    the proof that nothing was written is that `disk_spill` was never asked, no bytes
    were copied, and the store directory is still empty.
    """
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(2 * SLOT_BYTES))
    disk_spill.configured()   # refresh the ceiling the door will read
    disk_spill._state["estimated_bytes"] = 2 * SLOT_BYTES   # already at the ceiling
    headroom = disk_spill.mirror_headroom(SLOT_BYTES)
    assert headroom["ok"] is False
    assert headroom["reason_kind"] == "ceiling"
    assert headroom["ceiling_bytes"] == 2 * SLOT_BYTES
    assert headroom["projected_bytes"] == 3 * SLOT_BYTES

    role = handler()
    report = spill_a_handoff(role, [(6, KEY_A), (7, KEY_B)])
    assert report["queued"] == 0, report
    state = disk_store.counters()
    assert state["spill_on_store_refused"]["ceiling"] == 2, state
    assert state["spill_on_store_queued"] == 0
    spill = disk_spill.counters()
    assert spill["requests"] == 0, "a ceiling-refused offer still reached request_spill"
    assert spill["accepted"] == 0
    assert spill["bytes_written"] == 0
    assert store_files(tier) == {}, "a ceiling-refused mirror offer wrote a file"


def test_no_ceiling_configured_means_no_ceiling_and_the_door_stays_open(
        tier, monkeypatch):
    """An unparseable or absent ceiling means NO ceiling, deliberately.

    `disk_spill.capacity_ceiling_bytes()` refuses to read a typo as 0, because a 0
    ceiling means "evict everything you see". The door must inherit that exact
    verdict and not invent its own floor: a mirror door that refused everything
    because the ceiling failed to parse would be a typo turning the tier off.
    """
    monkeypatch.setenv(config.ENV_DISK_BYTES, "not-a-number")
    assert disk_spill.capacity_ceiling_bytes() is None
    headroom = disk_spill.mirror_headroom(SLOT_BYTES)
    assert headroom["ok"] is True
    assert headroom["ceiling_bytes"] is None
    role = handler()
    report = spill_a_handoff(role, [(6, KEY_A)])
    assert report["queued"] == 1
    assert disk_store.counters()["spill_on_store_refused"]["ceiling"] == 0


# --- 4. a full queue cannot block the step ------------------------------------

def test_a_full_queue_counts_queue_full_and_cannot_block_the_step(
        tier, monkeypatch):
    """Queue full, or writer down: a counted refusal, in milliseconds, never a wait.

    The writer is faked as "not running" so the filled queue stays filled - the real
    thread would drain the filler and the test would prove nothing. `put_nowait` is
    the only queue call on this path, so the elapsed bound is a pin on the mechanism,
    not a stopwatch wish: an engine step must never wait on NVMe.
    """
    monkeypatch.setattr(disk_spill, "MAX_QUEUE_DEPTH", 2)
    monkeypatch.setattr(disk_spill, "_ensure_thread", lambda: True)
    pending = disk_spill._get_queue()
    for index in range(2):
        pending.put_nowait({"blob": b"x", "slot": index, "key": KEY_C,
                            "bytes": SLOT_BYTES})
    role = handler()
    started = time.monotonic()
    report = spill_a_handoff(role, [(6, KEY_A), (7, KEY_B)])
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"the spill door blocked the step for {elapsed:.3f}s"
    assert report["queued"] == 0
    state = disk_store.counters()
    assert state["spill_on_store_refused"]["queue_full"] == 2
    assert state["spill_on_store_queued"] == 0
    # The spill path's own drop counter still sees its own currency: the door does not
    # hide a queue drop from the mechanism that owns the queue.
    assert disk_spill.counters()["drops"] == 2
    assert disk_spill.counters()["spilled"] == 0
    assert store_files(tier) == {}


# --- 5. duplicates are absorbed, not double-written ---------------------------

def test_a_duplicate_through_the_mirror_path_is_absorbed_not_double_written(tier):
    """The SAME block offered twice costs one file, because the dedup is the writer's.

    Two completed events carrying the same `(slot, key)` both pass the door - the door
    is not a second dedup, and a second dedup would be a second mechanism. The
    absorption happens where it already happened before this change: the writer finds
    the committed name and reports `duplicate`, so `spilled` and `bytes_written` move
    once. That is the machinery that absorbed 1,552 duplicate writes for free.
    """
    role = handler()
    spill_a_handoff(role, [(6, KEY_A)], event=7)
    spill_a_handoff(role, [(6, KEY_A)], event=8)
    assert disk_spill.drain(timeout_seconds=10.0)["drained"] is True
    spill = disk_spill.counters()
    assert spill["spilled"] == 1, spill
    assert spill["duplicates"] == 1, spill
    assert spill["bytes_written"] == SLOT_BYTES
    assert len(store_files(tier)) == 1
    # The door's own tally counted BOTH offers, which is the honest claim: the door
    # queued two, the writer absorbed one. Reading it as "2 files" would be a units
    # bug, and the pair of counters beside it is what makes the reading wrong.
    assert disk_store.counters()["spill_on_store_queued"] == 2
    assert disk_store.counters()["spill_on_store_bytes"] == 2 * SLOT_BYTES


# --- 6. spill-side only: behaviour and structure ------------------------------

def test_the_mirror_door_moves_no_posture_no_arm_and_no_serving_state(tier):
    """A mirrored hand-off changes nothing about serving, in this run.

    `mirror on` must not be able to raise a serve posture, flip the offer switch, arm
    the spill tier or the RAM tier, or serve a request. Behavioural half here; the
    structural half below is the one that generalises.
    """
    before = {"posture": disk_restore.disk_posture(),
              "counters": {key: disk_restore.counters()[key]
                           for key in ("disk_authority", "offers_enabled",
                                       "authority_lookups", "offers")},
              "spill_armed": disk_spill.is_armed(),
              "ram_armed": backport.is_armed()}
    role = handler()
    report = spill_a_handoff(role, [(6, KEY_A), (7, KEY_B)])
    assert report["queued"] == 2
    assert disk_restore.disk_posture() == before["posture"]
    assert {key: disk_restore.counters()[key] for key in before["counters"]} \
        == before["counters"]
    assert disk_spill.is_armed() == before["spill_armed"]
    assert backport.is_armed() == before["ram_armed"]


FORBIDDEN_POSTURE_WORDS = ("set_disk_authority", "set_offer_switch", "set_armed",
                           "disk_authority", "offers_enabled", "serving_as",
                           "serve_as", "boot_enable", "disk_boot", "arm(",
                           "disarm(", "set_restore_source")


def _code_only(func) -> str:
    """A function's source with its docstring gone: the CODE is what is pinned.

    THE ONE LINE THAT MAKES THIS A TEST AND NOT A FORMALITY: `ast.get_docstring` answers
    about the node you hand it, and the node parsed out of `inspect.getsource(func)` is
    a
    Module whose first statement is a FunctionDef - so asking the Module for its
    docstring
    returns None, nothing is stripped, and the assertion silently degrades into "this
    function's English does not contain the word". Ask the FunctionDef instead, and
    assert
    the strip removed something.
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    target = tree.body[0]
    doc = ast.get_docstring(target, clean=False)
    if doc:
        stmt = target.body[0]
        lines = source.splitlines()
        kept = [line for number, line in enumerate(lines, start=1)
                if not (stmt.lineno <= number <= stmt.end_lineno)]
        assert len(kept) < len(lines), "the docstring strip stripped nothing"
        source = "\n".join(kept)
    return source


@pytest.mark.parametrize("func", [disk_store.mirror_admission,
                                  disk_store.note_mirror_refusal,
                                  disk_store.spill_on_store_verdict])
def test_the_mirror_door_names_no_posture_or_arm_writer_in_its_own_source(func):
    """STRUCTURE: the door is spill-side, proved against its source, not its behaviour.

    A behavioural test shows one run; the claim is about every path. The same class of
    pin `tests/test_mamba_anchor_patch.py` puts on the anchor patch, including the
    docstring-strip (this module's prose NAMES the posture switches to say the door
    never touches them, and prose must not be what a structural test reads).
    """
    code = _code_only(func)
    for word in FORBIDDEN_POSTURE_WORDS:
        assert word not in code, f"{func.__name__} names {word!r}"


# --- 7. the numbers reach the doors -------------------------------------------

def test_the_mirror_counters_reach_the_rank_file_and_the_rollup(tier, monkeypatch,
                                                                tmp_path):
    """A counter missing from the rollup whitelist is SILENTLY NEVER ROLLED UP.

    Three halves, all necessary: the WRITER (`worker_report`, the rank file's `disk`
    block), the READER's whitelist (`RANK_ROLLUP_FIELDS`, walked by
    `_rank_file_rollup_row`) and a rollup built from TWO rank files that actually sums.
    This is exactly how the span-completeness panel stayed invisible for a day.
    """
    monkeypatch.setitem(disk_store._state, "spill_on_store_queued", 11)
    monkeypatch.setitem(disk_store._state, "spill_on_store_bytes", 11 * 4096)
    monkeypatch.setitem(disk_store._state, "spill_on_store_refused",
                        {"ceiling": 3, "queue_full": 2, "disabled": 0, "other": 1})
    monkeypatch.setitem(disk_store._state, "spill_on_store_refused_total", 6)
    report = disk_store.worker_report()
    assert report["spill_on_store_queued"] == 11, "the writer does not publish it"
    assert report["spill_on_store_bytes"] == 11 * 4096
    assert report["spill_on_store_refused_total"] == 6
    names = {field for field, _source, _unit in disk_store.RANK_ROLLUP_FIELDS}
    for field in ("spill_on_store_queued", "spill_on_store_bytes",
                  "spill_on_store_refused_total"):
        assert field in names, f"{field} is never rolled up"
        assert field in disk_store.RANK_ROLLUP_MEANINGS

    directory = tmp_path / "diag"
    # The `tier` fixture already created the diagnostics directory, so this is a no-op
    # when the run is honest and a creation when it is not - `exist_ok` rather than a
    # FileExistsError that would read as a bug in the test.
    directory.mkdir(exist_ok=True)
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    for rank, queued in (("PP0_TP0", 11), ("PP0_TP1", 7)):
        write_rank(directory, rank, disk=disk_section(
            spill_on_store_queued=queued,
            spill_on_store_bytes=queued * 4096,
            spill_on_store_refused_total=6))
    totals = disk_store.rank_rollup(force=True)["totals"]
    assert totals["spill_on_store_queued"] == 18, totals
    assert totals["spill_on_store_bytes"] == 18 * 4096
    assert totals["spill_on_store_refused_total"] == 12


def test_the_verdict_is_published_in_every_door_that_shows_the_spill_counters(tier):
    """`spill_on_store` travels with the counters that were published beside it.

    `hook_section`, `section` and the rank file are three readers of one door; the
    payload has to say WHICH door set the value (default, env, refused-env) or an
    operator reading a rank file cannot tell the shipped default from his own flag.
    """
    hook = disk_store.hook_section()
    section = disk_store.section()
    assert hook["spill_on_store"]["enabled"] is True
    assert section["spill_on_store"]["enabled"] is True
    for block in (hook, section, disk_store.worker_report()):
        for key in ("spill_on_store_queued", "spill_on_store_bytes",
                    "spill_on_store_refused"):
            assert key in block, key
    # The verb's own door, on the same page as the spill block it belongs to.
    status = disk_store.disk_for_engine(None, {"op": "status"})
    assert status["ok"] is True
    assert status["spill_on_store"]["enabled"] is True
    assert "spill_on_store_queued" in status["hook"]


# --- 8. release discipline: said, and not done --------------------------------

def test_the_release_discipline_note_says_the_arena_release_is_not_this_change(
        caplog):
    """The docstring must name what mirror mode makes possible - and deny doing it.

    If the arena is a buffer whose bytes are on disk promptly, its slots become
    releasable sooner. That is the NEXT change, and it needs its own evidence; the
    note is here so the next reader does not find the idea undocumented, or find it
    implemented in a release that promised not to.
    """
    note = disk_store.SPILL_ON_STORE_RELEASE_NOTE
    assert "arena" in note and "release" in note.lower()
    assert "does not" in note, "the note must deny doing it in this change"
    # And the note is reachable from the payload, not only from the source: a reader
    # with only the JSON gets the same sentence.
    assert disk_store.section()["spill_on_store"]["release_note"] == note


# --- 9. the door never faults -------------------------------------------------

def test_the_door_survives_a_spill_module_that_raises(tier, monkeypatch, caplog):
    """A spill path that raises is a counted refusal, never an exception on the step.

    `on_store_completed` runs inside the worker's `get_finished`. The door is the new
    code in that loop, so the module's existing "never breaks a step" rule applies to
    it with force: the failure must be counted, logged, and the fork's return value
    must still be handed back.
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("the spill module is on fire")

    monkeypatch.setattr(disk_spill, "mirror_headroom", explode)
    with caplog.at_level(logging.WARNING):
        report = spill_a_handoff(handler(), [(6, KEY_A), (7, KEY_B)])
    assert report["ok"] is False
    assert "RuntimeError" in report["reason"]
    state = disk_store.counters()
    # THE MODULE'S OWN WITNESS FOR A CONTAINED RAISE is `fallbacks` plus `last_error`
    # (`errors` counts the hook's own internal faults, a different thing, and inventing
    # a
    # second counter for one event is how two counters start disagreeing) - and the
    # snapshot's error ring, which is where the traceback actually goes.
    assert state["fallbacks"] == 1
    assert state["last_error"] == "RuntimeError"
    assert store_files(tier) == {}, "a broken guard must not write"
