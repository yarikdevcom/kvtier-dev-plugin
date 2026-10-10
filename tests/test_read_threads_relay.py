r"""The reader-count RELAY (0.2.67): ask the ranks to resize, believe only receipts.

`read-threads` moves a thread pool, and a thread pool is PROCESS state. The process
that answers a `/disk` call is the EngineCore, which never reads a spill file - the
ranks that apply restores are the only processes whose reader count matters, and a
worker has no utility channel (that is why `spill-one` writes a request FILE). So a
verb that cannot reach them is a knob on the wrong box, and the answer's
`workers_reachable_over_this_door: false` was honest but useless. This module's job is
the relay that fixes that, and the failures it must make impossible are ordered by what
each one costs:

1. **NEVER A SUCCESS THAT MEANS "NOBODY SAW IT".** The worst outcome is an operator
   reading `ok: true` after a resize that not one rank adopted. So an application is
   PROVEN, per rank, from that rank's own live counters read back into its receipt
   (`configured`/`alive`/`threads_source`) - and a receipt that claims success while
   disagreeing with itself is reported as NOT applied. "Received" is never a verdict.
2. **AN UNANSWERED ASK IS NOT A REFUSED ASK.** `ranks_pending` names the ranks that
   never spoke; `ranks_refused` names the ones that did and said no. Collapsing them
   loses the difference between "this rank's poll is dead" and "this rank rejected
   the value".
3. **THE TWO CHANNELS MUST NEVER BE CONFUSABLE.** A resize request living in
   `spill-one`'s single-slot file would silently replace a pending spill ask, and a
   spill record read as a resize record would resize a pool from someone else's
   bytes. Both files are read with a REQUIRED `role`/`op` match, so a record from the
   other channel is malformed data, never an instruction.
4. **THE STEP THREAD PAYS NOTHING FOR A CHANNEL NOBODY IS USING.** No new poller,
   thread or timer: the branch rides the `get_finished` poll's existing arm and
   interval gates, and when no rank is being asked it costs one failed `getsize`.
5. **THE OVERRIDE IS THE ONE PLACE A BIND READS.** The relay lands the ask through
   `disk_read.set_read_threads`, the same function the in-process verb uses, so there
   is no second copy of the count that a later bind could disagree with.

Everything is one process playing both ends (the fork-shaped fake worker, the real
request/receipt files under `tmp_path`), because the relay's contract is the FILE: a
real rank and the fake write byte-identical records. No engine, no vllm, no box.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import pytest
import test_disk_store as store_fixtures  # the module the fixtures live in
from test_disk_read_threads import _stop_all_readers
from test_disk_spill import SLOT_BYTES, FakeRegion

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    config,
    cpu_capacity,
    disk_read,
    disk_restore,
    disk_spill,
    disk_store,
    snapshot,
)

# THE FIXTURES, ONE COPY. `tier` (a configured, ARMED store under tmp_path with a rank)
# and `worker_class` (the fork's worker installed under the fork's module name) are
# imported rather than rewritten here: a second copy of the arm-the-store-and-pin-the-
# rank ceremony would drift from the first the moment the first was fixed, and every
# claim the relay makes about "an armed worker" would rest on a fixture nobody keeps.
fork_worker_instance = store_fixtures.fork_worker_instance
tier = store_fixtures.tier  # noqa: F811 - the configured, ARMED store under tmp_path
worker_class = store_fixtures.worker_class  # noqa: F811 - the fake fork's worker

REQUEST_FILE = "disk-read-threads.json"


@pytest.fixture(autouse=True)
def fresh_reader():
    """A fresh pool per test, with no reader thread carried across the boundary.

    The same reason `test_disk_read_threads.py` gives at length: a reader thread is
    bound to the queue OBJECT it was handed and `_state` is a module global, so a
    thread that leaves this test alive will serve the NEXT test's boot. Teardown stops
    every reader the process still has - including a slot this module lowered away and
    never published, which is exactly the surplus a relay test creates.
    """
    disk_read.reset_state_for_tests()
    yield
    _stop_all_readers(strict=True)
    disk_read.reset_state_for_tests()


@pytest.fixture(autouse=True)
def fresh_rank_set():
    """Restore the capacity handshake's rank set around every test.

    `expected_ranks` below has to WRITE that state for the completeness assertions to
    mean anything, and `_state` is a module global: a test that left two ranks
    recorded would hand the next test a rank set it never asked for, and a
    `completeness: unknown` assertion would be reading the previous test's handshake.
    """
    saved = dict(cpu_capacity._state.get("reported") or {})
    yield
    cpu_capacity._state["reported"] = saved


def diag_dir() -> pathlib.Path:
    return pathlib.Path(config.diagnostics_dir())


def request_record(**changes) -> dict:
    """A well-formed resize ask, as the caller writes it, with holes on demand.

    Built here rather than by calling the verb because the point of half of these
    tests is a file the CALLER would never have written: expired, oversized, or a
    spill-one record parked in the wrong file.
    """
    now = time.time()
    record = {"version": disk_store.PROTOCOL_VERSION,
              "role": disk_store.RESIZE_ROLE, "op": "read-threads",
              "seq": 7, "epoch": "epoch-under-test",
              "requested_ts": round(now, 6),
              "expires_ts": round(now + disk_store.REQUEST_TTL_SECONDS, 6),
              "threads": 6,
              "writer": disk_store.WRITER_TAG}
    record.update(changes)
    return record


def write_request(**changes) -> pathlib.Path:
    path = diag_dir() / REQUEST_FILE
    path.write_text(json.dumps(request_record(**changes)), encoding="utf-8")
    return path


def receipt_files() -> list[str]:
    return sorted(name for name in os.listdir(str(diag_dir()))
                  if name.startswith(disk_store.RESIZE_RECEIPT_PREFIX))


def poll(worker=None) -> dict:
    """One step-path poll, with the resize half pulled out.

    `poll_spill_one` is the ONLY poll on the step path, so the sibling branch has to
    show up in its answer; asserting on the same dict is how the test proves the
    branch rode the existing poll instead of a second one.
    """
    outcome = disk_store.poll_spill_one(worker)
    return outcome, (outcome.get("resize") or {})


def posted(worker=None) -> dict:
    return poll(worker)[1]


def read_receipt(rank="PP0_TP0") -> dict:
    path = diag_dir() / disk_store.RESIZE_RECEIPT_FILE.format(rank=rank)
    return json.loads(path.read_text(encoding="utf-8"))


def expected_ranks(*ranks) -> None:
    """The capacity handshake's answer, the ONLY way a caller knows who should have
    answered. Written directly because no handshake runs in a test, and restored by the
    autouse fixture above because `_state` is a module global.
    """
    cpu_capacity._state["reported"] = {}
    for rank in ranks:
        pp, _, tp = rank.replace("PP", "").replace("_TP", ",").partition(",")
        cpu_capacity._state["reported"][f"{pp},{tp}"] = 1


# --- the channel is its own, and cannot be confused for spill-one ----------------

def test_the_relay_has_its_own_file_and_its_own_receipt_prefix(tier):
    """Two channels in one directory is only safe if their NAMES cannot overlap."""
    assert disk_store.RESIZE_REQUEST_FILE == REQUEST_FILE
    assert disk_store.RESIZE_REQUEST_FILE != disk_store.PENDING_FILE
    assert disk_store.RESIZE_ROLE != disk_store.ROLE
    assert disk_store.RESIZE_RECEIPT_PREFIX == "disk-read-threads-receipt-"
    # A spill-one receipt must never be picked up by the resize reader and vice
    # versa: one prefix is not a prefix of the other.
    assert not disk_store.RESIZE_RECEIPT_PREFIX.startswith(
        disk_store.RECEIPT_PREFIX)
    assert not disk_store.RECEIPT_PREFIX.startswith(
        disk_store.RESIZE_RECEIPT_PREFIX)
    assert disk_store._resize_request_path() is not None
    assert disk_store._resize_request_path().endswith(REQUEST_FILE)


def test_a_spill_one_record_in_the_resize_file_is_refused_and_applies_nothing(tier):
    """THE confusion test, both directions.

    A spill-one record parked where a resize is read must be malformed data. The
    alternative is the worst kind of accident: a rank resizing its pool because it
    read someone else's spill ask.
    """
    disk_read.set_read_threads(4)
    before = disk_read.counters()["threads_configured"]
    spill_record = {"version": disk_store.PROTOCOL_VERSION, "role": disk_store.ROLE,
                    "op": "spill-one", "seq": 3, "epoch": "e",
                    "requested_ts": time.time(), "expires_ts": time.time() + 60,
                    "slot": 1, "key": "a3f0b7c1d4e5f607" + ":" + "0" * 8,
                    "fingerprint": "", "writer": disk_store.WRITER_TAG}
    (diag_dir() / REQUEST_FILE).write_text(json.dumps(spill_record), encoding="utf-8")
    adopted = posted()["adopted"]
    assert adopted is False
    assert disk_read.counters()["threads_configured"] == before
    assert receipt_files() == []


def test_a_resize_record_in_the_spill_one_file_is_refused_and_QUESTIONS_nothing(
        tier, worker_class):
    """The other direction: `poll_spill_one` must not adopt a resize ask as a spill.

    So the resize record is written INTO `spill-one`'s file, which is exactly the
    accident the two-name design prevents - and it is refused structurally: the spill
    reader demands a slot and a block key, and a resize record has neither. That is a
    stronger guarantee than a `role` string both sides could be edited into agreeing
    about.
    """
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    disk_store.install()
    (diag_dir() / disk_store.PENDING_FILE).write_text(json.dumps(request_record()),
                                                      encoding="utf-8")
    outcome = disk_store.poll_spill_one(worker)
    assert outcome["polled"] is True
    assert outcome["adopted"] is False
    assert outcome["reason"] and "slot" in outcome["reason"]
    assert not disk_store._state["last_path"]
    # ... and the resize channel saw nothing, because nothing is in ITS file.
    assert outcome["resize"]["adopted"] is False
    assert disk_read.counters()["threads_configured"] == 1
    assert receipt_files() == []


def test_a_second_ask_supersedes_the_first_rather_than_being_refused(tier):
    """The single-slot semantics, and why they differ from spill-one's.

    A spill ask is an EVENT: overwriting one loses a diagnostic, which is why
    spill-one refuses to share its slot. A resize ask is a STATEMENT OF INTENT about
    a count, so the newest ask is the whole truth and the older one is superseded -
    refusing a second resize because one is "in flight" would make the knob sticky.
    """
    write_request(seq=1, threads=4)
    write_request(seq=2, threads=9)
    answer = posted()
    assert answer["adopted"] is True
    assert disk_read.counters()["threads_configured"] == 9
    # ONE receipt, for the newest ask. A rank answering the superseded seq too would
    # make the caller's completeness arithmetic a coin flip.
    assert len(receipt_files()) == 1
    assert read_receipt()["request_seq"] == 2


# --- the poll branch, its gates, and its cost -----------------------------------

def test_the_ask_is_adopted_on_the_step_path_by_the_same_poll(tier, worker_class):
    """No new poller: one `get_finished` call adopts both channels' asks.

    The fake worker's `get_finished` is the engine's own call site, so this is the
    proof that the branch runs on a real step and not on a thread invented for it.
    """
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    assert disk_store.install()["installed"] is True
    write_request(threads=5)
    worker.get_finished(set())
    assert disk_read.counters()["threads_configured"] == 5
    assert read_receipt()["configured"] == 5


def test_the_sibling_branch_shares_the_interval_gate_and_the_arm_gate(tier):
    """The two gates spill-one has are the two gates the branch inherits.

    A second poll inside `POLL_INTERVAL_SECONDS` does not re-read the channel (one
    budget, not two), and an unarmed tier adopts nothing at all.
    """
    write_request(seq=1, threads=6)
    first, resize = poll()
    assert first["polled"] is True and resize["adopted"] is True
    assert read_receipt()["to"] == 6
    write_request(seq=8, threads=11)
    second, resize2 = poll()
    assert second["polled"] is False
    assert resize2 == {}
    assert disk_read.counters()["threads_configured"] == 6   # not 11 yet
    # The interval has passed; the arm has not. Disarm and the channel is invisible.
    disk_store._state["last_poll_mono"] = 0.0
    disk_spill.disarm()
    third, resize3 = poll()
    assert third["polled"] is False and resize3 == {}
    assert disk_read.counters()["threads_configured"] == 6
    # Still ONE receipt, for the ask that was answered: no poll ran, none was written.
    assert len(receipt_files()) == 1


def test_an_unarmed_tier_never_reads_the_channel_at_all(tier):
    """An unarmed worker must not chew a control file: the arm gate comes first."""
    disk_spill.disarm()
    write_request()
    outcome, resize = poll()
    assert outcome["polled"] is False
    assert resize == {}
    assert disk_store._state["resize_reads"] == 0
    assert receipt_files() == []


def test_an_untouched_channel_costs_one_failed_stat_and_no_state(tier):
    """The branch must be free when nobody is asking.

    One `getsize` that raises `FileNotFoundError`, a counted `absent`, no receipt, no
    thread touched - the cost a step thread pays on every boot that never uses the
    relay.
    """
    disk_read.set_read_threads(2)
    before = dict(disk_store.counters())
    outcome, resize = poll()
    assert outcome["polled"] is True
    assert resize["adopted"] is False
    assert resize["reason"] and "no resize request" in resize["reason"]
    assert disk_store.counters()["resize_reads"] == before["resize_reads"]
    assert disk_store.counters()["resize_absent"] == before["resize_absent"] + 1
    assert disk_read.counters()["threads_configured"] == 2
    assert receipt_files() == []


# --- proof of APPLICATION, not receipt -------------------------------------------

def test_a_rank_applies_the_ask_and_its_receipt_proves_it_from_its_own_counters(tier):
    """What makes a receipt a PROOF rather than an echo of the request.

    The receipt's `configured` is read out of THIS process's live `disk_read` state
    AFTER the apply, not copied from the request - which is what lets the caller (and
    the reader of a rank file) tell "I moved to 6" from "somebody asked for 6".
    """
    disk_read.set_read_threads(2)
    write_request(threads=6)
    answer = posted()
    assert answer["adopted"] is True
    receipt = read_receipt()
    assert receipt["role"] == disk_store.RESIZE_ROLE
    assert receipt["request_seq"] == 7
    assert receipt["requested_threads"] == 6
    assert receipt["refused"] is False and receipt["reason"] is None
    assert receipt["from"] == 2 and receipt["to"] == 6
    assert receipt["configured"] == 6
    assert receipt["threads_source"] == "verb"
    assert receipt["epoch"] == str(snapshot.epoch())
    # The receipt and the live state cannot disagree, because they are the same read.
    live = disk_read.counters()
    assert receipt["configured"] == live["threads_configured"]
    assert receipt["alive"] == live["threads_alive"]


def test_a_receipt_that_disagrees_with_its_own_counters_is_not_counted_as_applied(
        tier):
    """THE anti-lie test for the relay's whole claim.

    A rank could be patched, or a future writer could copy the request into the
    receipt, and the caller would report an application that never happened. The
    caller therefore RE-DERIVES the verdict from the receipt's own numbers and calls
    a self-contradicting one `not applied`, naming the contradiction.
    """
    write_request(threads=6)
    posted()
    receipt_path = diag_dir() / disk_store.RESIZE_RECEIPT_FILE.format(rank="PP0_TP0")
    forged = read_receipt()
    forged["configured"] = 1          # "applied", but its own counter says otherwise
    forged["threads_source"] = "env"
    receipt_path.write_text(json.dumps(forged), encoding="utf-8")
    receipts = disk_store._read_resize_receipts(str(diag_dir()), 7)
    assert set(receipts) == {"PP0_TP0"}
    entry = receipts["PP0_TP0"]
    assert entry["applied"] is False
    assert entry["verdict"] and "own counters" in entry["verdict"]


def test_an_ask_for_the_count_a_rank_already_runs_says_changed_false(tier):
    """An idempotent apply is a SUCCESS that changed nothing, and must say so.

    `last_resize.ok` is True on that path, so a receipt that inferred `changed` from it
    would claim a movement that never happened - and an operator who divides one
    `changed: true` by a poll count to estimate how fast the fleet reconfigures is
    estimating from a number the receipt invented. `applied` stays true: the rank IS at
    the count the ask asked for, which is the question the verdict answers.
    """
    disk_read.set_read_threads(6)
    write_request(threads=6)
    posted()
    receipt = read_receipt()
    assert receipt["refused"] is False
    assert receipt["changed"] is False
    assert receipt["from"] == 6 and receipt["to"] == 6
    entry = disk_store._read_resize_receipts(str(diag_dir()), 7)["PP0_TP0"]
    assert entry["applied"] is True


def test_a_rank_that_refused_is_counted_as_answered_and_not_applied(tier):
    """`refused` and `pending` are different operational statements."""
    write_request(threads=99)          # the FILE says 99; no local validation runs
    answer = posted()
    assert answer["adopted"] is True   # adopted means "answered", not "obeyed"
    receipt = read_receipt()
    assert receipt["refused"] is True
    assert receipt["reason_kind"] == "out_of_range"
    assert receipt["configured"] == 1  # nothing moved
    receipts = disk_store._read_resize_receipts(str(diag_dir()), 7)
    assert receipts["PP0_TP0"]["applied"] is False
    assert receipts["PP0_TP0"]["refused"] is True


# --- the verb, relayed ------------------------------------------------------------

def ask(**body) -> dict:
    return disk_store.disk_for_engine(None, {"op": "read-threads", **body})


def test_the_verb_relays_and_reports_per_rank_application(tier):
    """The round trip the operator actually drives: ask, poll, read the receipts back.

    `wait_s: 0` means "do not wait for an answer", and the answer to that must be a
    status that CLAIMS nothing (`not yet answered`) rather than a failure or - worse -
    a success. The verdict arrives on the next read, from the receipts on disk.
    """
    expected_ranks("PP0_TP0")
    asked = ask(threads=6, wait_s=0)
    assert asked["ok"] is True
    assert asked["relay"]["attempted"] is True
    assert asked["relay"]["status"] == "not yet answered"
    assert asked["relay"]["ranks_answered"] == []
    assert asked["relay"]["ranks_pending"] == ["PP0_TP0"]
    posted()
    read = ask()
    relay = read["relay"]
    assert relay["ranks_answered"] == ["PP0_TP0"]
    assert relay["ranks_applied"] == ["PP0_TP0"]
    assert relay["completeness"] == "complete"
    assert relay["asked"]["threads"] == 6, "the read names the ask it is reporting on"
    assert read["relay"]["receipts"]["PP0_TP0"]["configured"] == 6


def test_an_unanswered_relay_is_never_reported_as_a_success(tier):
    """The failure the operator named, pinned as a wire behaviour.

    A local apply is still reported truthfully (`resize.ok`), but the verb as a whole
    must not read as success when its purpose - the ranks - produced silence after the
    window the caller asked to be waited out. And this one blocks even WITHOUT `ranks`
    spelled out: an ask that was sent and ignored is the exact lie, whatever the caller
    left in or out of the body.
    """
    asked = ask(threads=6, wait_s=1)
    relay = asked["relay"]
    assert asked["ok"] is False
    assert asked["reason"] and "no rank answered" in asked["reason"]
    assert relay["status"] == "no_rank_answered"
    assert relay["refusal"] == "relay_no_answer"
    assert relay["ranks_answered"] == []
    assert relay["ranks_applied"] == []
    assert asked["resize"]["ok"] is True      # the local half really did move
    assert asked["local"]["configured"] == 6


def test_a_partial_answer_names_the_ranks_that_have_not_answered(tier):
    """Which ranks are silent, rather than a timeout that reads as a failure.

    The caller waits the window it was given and stops: it does not block a caller who
    asked for one second, and it never guesses that a silent rank refused. The first
    phase is pure silence (nobody polled yet), the second is one rank answered while the
    other still has not - and the difference between those two is what the payload is
    for.
    """
    expected_ranks("PP0_TP0", "PP1_TP0")
    started = time.monotonic()
    asked = ask(threads=6, wait_s=1)
    assert time.monotonic() - started < 2.5, "the window is the caller's, not longer"
    assert asked["ok"] is False
    relay = asked["relay"]
    assert relay["status"] == "no_rank_answered"
    assert relay["completeness"] == "partial"
    assert relay["ranks_pending"] == ["PP0_TP0", "PP1_TP0"]
    assert relay["ranks_answered"] == []
    assert asked["reason"] and "PP1_TP0" in asked["reason"]
    posted()
    read = ask()
    relay = read["relay"]
    assert relay["completeness"] == "partial"
    assert relay["ranks_answered"] == ["PP0_TP0"]
    assert relay["ranks_applied"] == ["PP0_TP0"]
    assert relay["ranks_pending"] == ["PP1_TP0"]
    assert relay["status"] == "partial"


def test_a_relay_without_a_channel_says_so_and_fails_only_an_explicit_ask(
        tier, monkeypatch):
    """No diagnostics directory means no channel to a worker, and no rank can be asked.

    Two verdicts, because two intents. A caller that left `ranks` out gets `ok: true`
    with the local number it actually moved and a refusal naming the environment
    variable to fix - failing the whole verb here would bury a real change under a
    complaint about a change that could not be made. A caller that wrote
    `"ranks": true` asked a question that cannot be answered, and gets `ok: false`.
    """
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    defaulted = ask(threads=6)
    assert defaulted["ok"] is True
    assert defaulted["relay"]["attempted"] is False
    assert defaulted["relay"]["status"] == "not attempted"
    assert defaulted["relay"]["refusal"] == "no_channel"
    assert defaulted["local"]["configured"] == 6
    assert config.ENV_DIAG_DIR in defaulted["relay"]["reason"]

    explicit = ask(threads=6, ranks=True)
    assert explicit["ok"] is False
    assert explicit["relay"]["refusal"] == "no_channel"
    assert explicit["local"]["configured"] == 6
    assert config.ENV_DIAG_DIR in explicit["reason"]


def test_a_relay_without_an_arm_writes_no_litter(tier):
    """`not_armed` is a different refusal from `no_channel`, because the fix differs.

    And the precondition is enforced on the CALLER's side too, not only in the rank's
    poll: an ask that every rank must refuse is not a diagnostic, it is a file some
    worker has to chew through for nothing.
    """
    disk_spill.disarm()
    armed = ask(threads=8, ranks=True)
    assert armed["ok"] is False
    assert armed["relay"]["refusal"] == "not_armed"
    assert armed["relay"]["status"] == "not attempted"
    assert armed["relay"]["attempted"] is False
    assert armed["local"]["configured"] == 8
    assert not (diag_dir() / REQUEST_FILE).exists(), "no litter for a refused relay"


def test_an_answer_without_a_known_rank_set_claims_no_completeness(tier):
    """One rank obeying with nobody able to say how many ranks exist is NOT `complete`.

    The same rule spill-one already lives by (`completeness: unknown`), and the reason
    is stronger here: `complete` on a 4-rank boot where one rank's poll is dead would
    be the exact lie this relay exists to prevent.
    """
    ask(threads=6, wait_s=0)
    posted()
    read = ask()
    assert read["relay"]["ranks_applied"] == ["PP0_TP0"]
    assert read["relay"]["completeness"] == "unknown"
    assert read["relay"]["status"] == "answered_unverifiable"
    assert read["ok"] is True
    assert read["relay"]["ranks_pending_unknown"] is True


def test_ranks_false_asks_the_local_process_only_and_says_so(tier):
    """The opt-out exists, and it is loud rather than a silent default flip."""
    asked = ask(threads=5, ranks=False)
    assert asked["ok"] is True
    assert asked["relay"]["status"] == "not requested"
    assert asked["relay"]["attempted"] is False
    assert asked["relay"]["ranks_note"]
    assert not (diag_dir() / REQUEST_FILE).exists()


def test_a_read_never_relays_and_points_at_the_rank_files_instead(tier):
    """The channel carries an INSTRUCTION. A question is answered locally.

    A rank's count is already visible in its own rank file (`threads_source`,
    `last_resize`), so inventing a query kind would add a second receipt meaning for
    a fact the fleet already publishes.
    """
    read = ask()
    assert read["ok"] is True
    assert read["relay"]["status"] == "not requested"
    assert read["relay"]["attempted"] is False
    assert read["relay"]["ranks_note"] and "rank file" in read["relay"]["ranks_note"]


def test_a_relayed_resize_lands_the_same_override_a_bind_reads(tier):
    """No parallel state variable: the relay calls `set_read_threads`, full stop.

    Three later binds must not resurrect the launcher's value, and `env_requested`
    must keep reporting what the boot was actually given.
    """
    write_request(threads=6)
    posted()
    for step in range(3):
        disk_read.begin_batch(f"bind{step}", 1)
        disk_read.finish_batch(f"bind{step}")
    counters = disk_read.counters()
    assert counters["threads_configured"] == 6
    assert counters["threads_source"] == "verb"
    assert counters["threads_requested"] in (None, 1)
    assert disk_restore.worker_report()["readers"]["threads_source"] == "verb"


def test_a_relayed_rank_stops_reporting_env_in_its_own_rank_file(tier):
    """The rank file must stop lying.

    Before the relay a rank's readers section could only ever say `env`, because
    nothing could ever ask it. After one adopted ask the REAL rank-file builder says
    `verb` and carries its OWN last_resize, and a rank that was never asked still says
    `env` - which is the contrast that makes the field worth reading.
    """
    assert disk_restore.worker_report()["readers"]["threads_source"] == "env"
    write_request(threads=6)
    posted()
    readers = disk_restore.worker_report()["readers"]
    assert readers["threads_source"] == "verb"
    assert readers["configured"] == 6
    assert readers["last_resize"]["to"] == 6
    assert readers["last_resize"]["ok"] is True
    assert readers["last_resize"]["from"] == 1


def test_a_process_with_no_rank_writes_no_receipt(tier, monkeypatch):
    """Otherwise the EngineCore answers its own ask and inflates `ranks_answered`.

    The asking process has no rank id, so its receipt would be an "answer" from a
    rank the completeness arithmetic never expected.
    """
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)
    import kvtier_dev_plugin.worker_diag as worker_diag
    monkeypatch.setattr(worker_diag, "rank_id", lambda: (None, "test"))
    write_request(threads=4)
    answer = posted()
    assert answer["adopted"] is True
    assert answer["answer"]["rank"] is None
    assert answer["answer"]["written"] is False
    assert "no rank id" in answer["answer"]["receipt_reason"]
    assert disk_read.counters()["threads_configured"] == 4   # applied anyway
    assert receipt_files() == []


# --- malformed data, expiry, and the vocabulary ----------------------------------

@pytest.mark.parametrize(("changes", "needle"), (
    ({"role": "disk-spill-one"}, "role"),
    ({"op": "spill-one"}, "op"),
    ({"version": 1}, "version"),
    ({"seq": 0}, "seq"),
    ({"requested_ts": "now"}, "requested_ts"),
    ({"expires_ts": None}, "expires_ts"),
    ({"threads": "six"}, "threads"),
    ({"threads": None}, "threads"),
))
def test_a_malformed_ask_is_counted_and_never_applied(tier, changes, needle):
    """An untrusted file is data. It cannot move a pool and it cannot be silent."""
    disk_read.set_read_threads(3)
    write_request(**changes)
    answer = posted()
    assert answer["adopted"] is False
    assert needle in answer["reason"]
    assert disk_read.counters()["threads_configured"] == 3
    assert disk_store.counters()["resize_malformed"] >= 1
    assert receipt_files() == []


def test_an_oversized_ask_is_refused_without_being_parsed(tier):
    disk_read.set_read_threads(3)
    path = diag_dir() / REQUEST_FILE
    path.write_bytes(b"[" + b" " * (disk_store.MAX_RESIZE_REQUEST_BYTES + 10) + b"]")
    answer = posted()
    assert answer["adopted"] is False and "bytes" in answer["reason"]
    assert disk_read.counters()["threads_configured"] == 3


def test_an_expired_ask_is_never_applied(tier):
    """A resize adopted three hours late is a surprise, not a service."""
    disk_read.set_read_threads(3)
    write_request(threads=9, requested_ts=time.time() - 400,
                  expires_ts=time.time() - 1)
    answer = posted()
    assert answer["adopted"] is False and "expired" in answer["reason"]
    assert disk_read.counters()["threads_configured"] == 3
    assert disk_store.counters()["resize_expired"] >= 1


def test_a_rank_fulfilment_with_the_gate_down_answers_rather_than_raising(tier):
    """The one path the poll's gate hides, exercised directly.

    `fulfil_read_threads` is also callable straight (a test, or whatever a later release
    wires to it), and a rank whose gate is down must ANSWER - a receipt naming
    `not_armed` - rather than raise on a `last_resize` a never-touched pool does not
    have. A raise here would be counted as `poll raised` and the caller would report a
    rank that had its say as one that said nothing.
    """
    disk_spill.disarm()
    answer = disk_store.fulfil_read_threads(request_record(threads=6))
    assert answer["refused"] is True
    assert answer["reason_kind"] == "not_armed"
    assert answer["written"] is True
    receipt = read_receipt()
    assert receipt["refused"] is True and receipt["reason_kind"] == "not_armed"
    assert receipt["from"] is None and receipt["to"] is None
    assert receipt["changed"] is False
    assert receipt["configured"] == 1, "the rank's real count, not the asked-for one"
    entry = disk_store._read_resize_receipts(str(diag_dir()), 7)["PP0_TP0"]
    assert entry["applied"] is False and entry["refused"] is True


def test_an_ask_already_answered_by_this_rank_is_not_answered_twice(tier):
    """De-duplication, so a 2 s poll cannot rewrite the receipt forever."""
    write_request(threads=6)
    assert posted()["adopted"] is True
    disk_store._state["last_poll_mono"] = 0.0
    moved = disk_read.set_read_threads(4)      # something else moved the pool
    assert moved["ok"] is True
    second = posted()
    assert second["adopted"] is False
    assert "already" in second["reason"]
    assert disk_read.counters()["threads_configured"] == 4


def test_spill_one_honours_an_explicit_zero_window(tier):
    """`wait_s: 0` means "do not wait", for BOTH channels' callers.

    `float(body.get("wait_s") or 3.0)` read an explicit 0 as "absent" and blocked for
    three seconds anyway - the one value a caller is most likely to ask for on purpose,
    and now reachable from the wire because `wait_s` had to be added to `ARG_SPECS` for
    the relay. Fixed in the same change as the relay precisely because the relay made it
    reachable: shipping a window argument that lies about zero is worse than not
    shipping it, and the two channels' windows must be read the same way.
    """
    started = time.monotonic()
    answer = disk_store.disk_for_engine(None, {"op": "spill-one", "slot": 1,
                                               "key": store_fixtures.KEY,
                                               "wait_s": 0})
    assert answer["ok"] is True
    assert time.monotonic() - started < 1.0, "a zero window must not block the caller"


def test_the_relay_refusal_vocabulary_is_named_and_does_not_widen_any_other(
        tier):
    """Every relay refusal must survive `_count`, which folds an unknown name into
    `bad_call` - so the vocabulary is a named tuple that is a SUBSET of the door's.
    """
    assert disk_store.RELAY_REFUSALS
    assert set(disk_store.RELAY_REFUSALS) <= set(disk_store.REFUSAL_REASONS), \
        "a refusal _count() cannot name is a refusal that reports as bad_call"
    # The two capability refusals are the ONLY thing `ok` forgives a caller that did not
    # name the ranks, so they are a named set rather than a string comparison inlined
    # where the verdict is composed.
    assert disk_store.RELAY_CAPABILITY_REFUSALS == ("no_channel", "not_armed")
    assert set(disk_store.RELAY_CAPABILITY_REFUSALS) <= set(disk_store.RELAY_REFUSALS)
    assert set(disk_store.RELAY_STATUSES) == {
        "not requested", "not attempted", "not yet answered", "no_rank_answered",
        "answered_unverifiable", "answered_refused", "partial", "complete"}
    # The three value refusals stay exactly three, and the spill path's table is
    # untouched by a relay that has nothing to do with it.
    assert disk_read.RESIZE_REFUSALS == ("bad_type", "out_of_range", "in_progress")
    assert "relay_no_answer" in disk_store.REFUSAL_REASONS
    assert "relay_unwritable" in disk_store.REFUSAL_REASONS


def test_the_status_publishes_the_relay_counters_and_the_file_names(tier):
    """An operator must be able to ask "has anyone ever relayed anything here?".

    Same numbers in the door's `status` and the rank file's disk section, so a verb
    and a snapshot cannot disagree about the tally.
    """
    status = disk_store.disk_for_engine(None, {"op": "status"})
    relay = status["read_threads_relay"]
    assert relay["request_file"] == REQUEST_FILE
    assert relay["receipt_prefix"] == disk_store.RESIZE_RECEIPT_PREFIX
    assert relay["requests_written"] == 0
    for key in ("polls", "reads", "absent", "malformed", "adopted", "expired",
                "receipt_errors"):
        assert key in relay, key
    write_request(threads=4)
    posted()
    after = disk_store.disk_for_engine(None, {"op": "status"})["read_threads_relay"]
    assert after["adopted"] == 1
    assert after["reads"] == 1
    section = disk_store.section()
    assert section["read_threads_relay"]["adopted"] == 1
    # The door and the rank file answer the SAME tallies (one `_state`, two doors), and
    # the door's refusal table can name every relay failure without folding one into
    # `bad_call`.
    assert section["read_threads_relay"]["reads"] == after["reads"]
    assert set(section["refused"]) >= set(disk_store.RELAY_REFUSALS)


# --- the wait is bounded by the door, and by what a wait can actually achieve ----
#
# WHY THIS SECTION EXISTS (the live box found it). The `/disk` door wraps every engine
# hop in `asyncio.wait_for(..., config.timeout_seconds())` (`endpoint.py:315`) and
# answers 504 when that budget runs out, while the engine-side handler runs ON THE
# ENGINECORE LOOP THREAD (`engine_side.py:99`, `engine_side.py:308`, and the
# `input_queue` note at `engine_side.py:39-41`: these handlers are woken by the same
# queue the busy loop blocks on, which is why they "must be O(microseconds) reads").
# A `time.sleep` wait loop here therefore (a) can outlive the door and report a 504 over
# a resize every rank applied, and (b) holds the very loop that dispatches the steps on
# which the ranks poll the channel - a wait that cannot be answered by construction.
# These tests are the pin that keeps both true.


def test_no_resize_call_can_outlive_the_door_budget(tier, monkeypatch):
    """THE GUARD: whatever a caller asks for, the handler returns inside the door.

    A wait that runs past the door's budget does not report a slow answer, it reports
    `504 engine did not answer within Ns` over a resize every rank has already applied -
    a knob that teaches the operator to distrust a success. The honoured wait plus the
    margin must fit the budget, and the whole call must return inside the budget, for
    every shape of ask: the default, one inside the cap, one far above it, and one that
    is not a number at all.
    """
    monkeypatch.setenv(config.ENV_TIMEOUT, "8")
    budget = config.timeout_seconds()
    for requested in (None, 0, 0.5, 15, 10000, "nonsense"):
        body = {"op": "read-threads", "threads": 6}
        if requested is not None:
            body["wait_s"] = requested
        started = time.monotonic()
        answer = disk_store.disk_for_engine(None, body)
        elapsed = time.monotonic() - started
        wait = answer["relay"]["wait_s"]
        assert wait + disk_store.DOOR_MARGIN_SECONDS <= budget, (
            f"wait_s={requested!r} honoured {wait}s against a {budget}s door")
        assert elapsed < budget, (
            f"wait_s={requested!r} held the handler {elapsed:.2f}s of a {budget}s "
            "door budget: the caller would get a 504 over a resize that worked")


def test_the_default_ask_posts_the_ask_and_returns_without_waiting(tier):
    """The default is POST-AND-RETURN, because waiting on this thread cannot help.

    The answer comes from a rank's step path, and steps are dispatched by the loop this
    handler is running on - so a default wait is a stall of serving that buys a
    guaranteed silence. The default is therefore 0.0: write the ask, name the ranks that
    have not answered, return in milliseconds.
    """
    expected_ranks("PP0_TP0", "PP1_TP0")
    started = time.monotonic()
    asked = ask(threads=6)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"the default ask blocked the handler for {elapsed:.2f}s"
    assert asked["relay"]["wait_s"] == 0.0
    assert asked["relay"]["wait_requested"] is None, "nothing was asked for"
    assert asked["relay"]["status"] == "not yet answered"
    assert asked["relay"]["ranks_pending"] == ["PP0_TP0", "PP1_TP0"]
    assert asked["ok"] is True
    # and it says where the answer will come from, rather than implying a wait would
    # have produced it
    assert "step" in asked["relay"]["note"]


def test_a_wait_above_the_cap_is_honoured_down_and_says_so(tier, monkeypatch):
    """A caller asking for more than the door can carry gets a capped wait and a reason.

    Not a 400 (the ask is legitimate), and not the wait it asked for (which would be a
    504 over a working resize). The response states the requested number, the honoured
    number, the cap, and which knob produced the cap.
    """
    monkeypatch.setenv(config.ENV_TIMEOUT, "8")
    expected_ranks("PP0_TP0")
    asked = ask(threads=6, wait_s=15)
    relay = asked["relay"]
    assert relay["wait_requested"] == 15.0
    assert relay["wait_capped"] is True
    assert relay["wait_s"] == disk_store.RESIZE_WAIT_GRACE_SECONDS
    assert relay["wait_cap"]["ceiling"] == disk_store.RESIZE_WAIT_GRACE_SECONDS
    assert config.ENV_TIMEOUT in relay["wait_cap"]["why"]
    assert relay["ranks_pending"] == ["PP0_TP0"]


def test_the_cap_follows_the_door_and_not_a_second_hardcoded_number(monkeypatch,
                                                                    tier):
    """The ceiling is DERIVED from the door's own budget, in the same source it reads.

    A second hardcoded number is how 30s came to sit against an 8s door in the first
    place, so the ceiling is a function of `config.timeout_seconds()`: a tighter door
    means a tighter wait, and a door tighter than the grace means no wait at all rather
    than a 504.
    """
    monkeypatch.setenv(config.ENV_TIMEOUT, "8")
    assert disk_store._resize_wait_ceiling() == disk_store.RESIZE_WAIT_GRACE_SECONDS
    monkeypatch.setenv(config.ENV_TIMEOUT, "1.2")
    # 1.2s door, 0.5s margin -> 0.7s of room, which is under the grace, so the grace is
    # NOT the binding limit and the honoured wait must fit the door anyway.
    assert disk_store._resize_wait_ceiling() == pytest.approx(0.7)
    assert ask(threads=6, wait_s=15)["relay"]["wait_s"] == pytest.approx(0.7)
    monkeypatch.setenv(config.ENV_TIMEOUT, "0.4")
    # A door tighter than the margin for the response gets NO wait at all: a wait that
    # leaves no room to answer is a 504 with extra steps.
    assert disk_store._resize_wait_ceiling() == 0.0
    assert ask(threads=6, wait_s=15)["relay"]["wait_s"] == 0.0


def test_posting_then_reading_ends_complete_with_nobody_blocked(tier, monkeypatch):
    """THE FLOW THAT REPLACES THE WAIT: ask, serve, read - and the read says complete.

    One process plays both ranks because the de-duplication memory is per-process
    ("this rank already answered"), so the served table is cleared between the two
    answers; every other part of the path is the real one, writing real receipt files
    with distinct rank ids. The assertion that matters is the pair together:
    `completeness: complete` AND the whole sequence inside the door's budget.
    """
    expected_ranks("PP0_TP0", "PP1_TP0")
    started = time.monotonic()
    asked = ask(threads=6)
    assert asked["relay"]["completeness"] == "partial"
    disk_store.poll_spill_one(None)
    # The second rank, in this same test process: a fresh rank id and a fresh
    # per-process "already served" memory, which is what a second process would bring.
    # The interval gate is also reset - one process cannot legally poll twice in the
    # same 2 s window, and here it is playing two ranks, not one rank twice.
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP1_TP0")
    disk_store._state["resize_served"] = {}
    disk_store._state["last_poll_mono"] = 0.0
    assert disk_store.poll_spill_one(None)["polled"] is True
    disk_store.poll_spill_one(None)
    read = ask()
    elapsed = time.monotonic() - started
    relay = read["relay"]
    assert relay["completeness"] == "complete"
    assert relay["ranks_applied"] == ["PP0_TP0", "PP1_TP0"]
    assert relay["ranks_pending"] == []
    assert elapsed < config.timeout_seconds(), (
        f"the whole post-then-read flow took {elapsed:.2f}s; nobody should be blocked "
        f"for a second")
    assert read["ok"] is True


def test_the_wait_policy_is_published_where_an_operator_will_ask(tier, monkeypatch):
    """The door's budget, the grace and the default are all readable from the verb.

    An operator deciding what to send must not have to read the source to learn that a
    long wait is impossible here; and the same numbers belong in the rank file, so a
    snapshot and a verb cannot disagree about the policy in force.
    """
    monkeypatch.setenv(config.ENV_TIMEOUT, "8")
    relay = ask(threads=6)["relay"]
    assert relay["door_budget_seconds"] == 8.0
    assert relay["wait_grace_seconds"] == disk_store.RESIZE_WAIT_GRACE_SECONDS
    assert relay["wait_default_seconds"] == disk_store.RESIZE_WAIT_SECONDS_DEFAULT
    assert relay["wait_default_seconds"] == 0.0
    section = disk_store.section()["read_threads_relay"]
    assert section["wait_default_seconds"] == 0.0
    assert section["door_budget_seconds"] == 8.0
    assert section["wait_ceiling_seconds"] == pytest.approx(
        disk_store._resize_wait_ceiling())


def test_spill_one_still_offers_its_own_thirty_second_window(tier):
    """THE PRE-EXISTING INSTANCE, pinned so it cannot be "fixed" by accident.

    `spill-one` answers the same 30-second ceiling against the same door, on the same
    loop thread, with the same self-deadlock - and its DEFAULT is 3.0s of sleep on the
    EngineCore loop. That is reported, not changed here: the operator wants to decide it
    separately. This test fails the moment anyone changes spill-one's window without
    saying so, which is the point.
    """
    assert disk_store.SPILL_ONE_WAIT_SECONDS_MAX == 30.0
    assert disk_store.SPILL_ONE_WAIT_SECONDS_DEFAULT == 3.0
    assert disk_store.SPILL_ONE_WAIT_SECONDS_DEFAULT > (
        disk_store.RESIZE_WAIT_SECONDS_DEFAULT)


def test_the_relay_refusal_counters_move_only_on_a_real_failure(tier, monkeypatch):
    """`relay_no_answer` counts a FAILED ASK, and a clean round trip counts nothing.

    The tallies are read from `section()`, the rank file's half, because that is where
    this module's own refusal table is published (`op:status` carries the spill
    mechanism's table, a different one). The point of pinning it is the false positive:
    a caller that asked for no wait leaves an ask in flight, and counting that as a
    failure would make the counter useless for finding the real ones.
    """
    clean = disk_store.section()
    assert clean["refused"]["relay_no_answer"] == 0
    ask(threads=6, wait_s=0)
    after = disk_store.section()
    assert after["refused"]["relay_no_answer"] == 0
    ask(threads=6, wait_s=1)
    after = disk_store.section()
    assert after["refused"]["relay_no_answer"] == 1
    assert after["refused"]["relay_unwritable"] == 0
    monkeypatch.setattr(disk_store.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    broken = ask(threads=7)
    assert broken["ok"] is False
    assert broken["relay"]["refusal"] == "relay_unwritable"
    assert broken["resize"]["ok"] is True       # the local apply is unaffected
    assert disk_store.section()["refused"]["relay_unwritable"] == 1
