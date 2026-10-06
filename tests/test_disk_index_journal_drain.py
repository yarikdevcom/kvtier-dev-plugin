r"""THE 0.2.28 REGRESSION: a deep journal must be DRAINED, never buried.

THE NIGHT THIS EXISTS FOR. On the live box: four journals totalling 4,711,140
bytes / 132,099 records, ~6,290 net-live identities (proven by a full front-to-back
replay), 14,729 live files. The index reported `entries: 146`. Every journal was
bigger than its share of one refresh, so the 0.2.22 rule took the BURIED branch on
the first pass: read the newest 131,072 bytes, `_forget_rank` everything older,
land the cursor at EOF. The live counters said `catch_up_jumps: 4`,
`bytes_skipped: 4,186,980`, and `last_reason` named the jump - while `lag_bytes`
sat at a smug 0 and `why_no_restore` only spoke when `lag_bytes != 0`. The window
was 93% removal lines (a ceiling reclamation had just journalled ~15,000 `-`), so
what survived was 146 identities. It was UNRECOVERABLE in place (cursors only
advance; the sole reset is a truncation that append-only journals never do) and a
restart did not help (1.4 MB is still deeper than one share, so the first pass
buried again, deterministically). `no_disk_run` refused every offer for hours.

This suite pins the fix (the front-to-back DRAIN) from the operator's side:

* THE TEST THAT WOULD HAVE CAUGHT TONIGHT: four ~1 MB journals whose target
  identity is added at a LOW seq and never removed. One `refresh(force=True)` -
  the boot drain - and `contains()` sees it. Under the buried rule this answers
  False; that is the whole bug in one assertion.
* `entries` after convergence equals the TRUE NET-LIVE count (adds minus removes)
  over the whole journal, with tonight's shape (~1,900 removals vs 146 adds per
  journal) so no window-based implementation passes by luck.
* Convergence is monotone and observable: partial, then more, then complete; the
  per-call byte bound is an EXACT number, not a feeling.
* A converged journal reads only its new tail; an add-then-remove never
  resurrects.
* The memory cap drops-oldest and SHOUTS (one log line per episode), and
  `entry_cap_hit` makes a capped index unreadable as a caught-up one.
* `why_no_restore` / the `no_disk_run` refusal text name the skip/jump and the
  drain even when `lag_bytes == 0` - the exact blind spot that cost the hours.

Stdlib plus the plugin; `tmp_path` for every file; no engine, no threads.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_restore,
    disk_spill,
)

GROUP = 0
WIDTH = 11_468_800
TARGET_KEY16 = "0000d1c0d1c00000"


def key_for(index: int) -> str:
    return f"{index:016x}"


def present(key16: str, group: int = GROUP) -> bool:
    return disk_index.contains(key16 + "0" * 32 + f"{group:08x}")


def journal_named(directory, rank: str) -> str:
    return os.path.join(str(directory),
                        f"{disk_index.JOURNAL_PREFIX}{rank}"
                        f"{disk_index.JOURNAL_SUFFIX}")


def line(seq: int, op: str, key16: str) -> str:
    return (f"{seq}{disk_index.LINE_SEP}{op}{disk_index.LINE_SEP}"
            f"{key16}{disk_index.LINE_SEP}{WIDTH}{disk_index.LINE_SEP}{GROUP}\n")


def seed_incident_journal(path, *, adds: int, removal_pairs: int) -> None:
    """One journal in TONIGHT'S shape, ~1 MB, written once for speed.

    The first `adds` records are `+` at LOW sequence numbers, never removed - the
    target identity is among them at the very front. On top of that,
    `removal_pairs` add/remove PAIRS (the shape a ceiling reclamation journalled
    that night: ~15k removals crowding the tail). The buried rule's newest-window
    saw mostly `-`; the drain applies the file in order and the keepers survive.
    """
    lines = []
    seq = 1
    keeper_keys = [key_for(index) for index in range(adds)]
    lines.append(line(seq, disk_index.SPILL_LINE, TARGET_KEY16))
    seq += 1
    for key in keeper_keys:
        lines.append(line(seq, disk_index.SPILL_LINE, key))
        seq += 1
    churn_base = 0xC0000000
    for offset in range(removal_pairs):
        churn = f"{churn_base + offset:016x}"
        lines.append(line(seq, disk_index.SPILL_LINE, churn))
        seq += 1
        lines.append(line(seq, disk_index.REMOVE_LINE, churn))
        seq += 1
    blob = "".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(blob)


@pytest.fixture(autouse=True)
def fresh_index(tmp_path, monkeypatch):
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    disk_index.reset_state_for_tests()
    disk_restore._proof_cache.update({"mono": None, "result": None})
    yield directory
    disk_index.reset_state_for_tests()
    disk_restore._proof_cache.update({"mono": None, "result": None})


RANKS = ("PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1")


# --- the regression that would have caught tonight ----------------------------

def test_one_refresh_finds_an_early_add_in_four_deep_journals(fresh_index):
    """FOUR ~1 MB JOURNALS, TARGET ADDED AT SEQ 1, ONE REFRESH, `contains` TRUE.

    This is the exact box shape of the incident: per journal ~1 MB (8x the old
    131,072-byte share), target added at a low seq and never removed. Under the
    buried rule the first pass jumped to each tail, `_forget_rank`ed the history
    that had just read the target's `+`, and `contains()` answered False while the
    file sat committed on disk. Under the drain the first refresh converges (the
    boot budget covers 4 MB in one go) and the answer is True. The assertion that
    the buried path can NEVER run again (`catch_up_jumps == 0`, `bytes_skipped ==
    0`) travels with it, because a fix that converged by re-burying would fail
    HERE even if it passed the membership question.
    """
    per_journal_adds = 12_000
    per_journal_pairs = 12_000            # ~1 MB of lines per journal, 93% churn
    for rank in RANKS:
        seed_incident_journal(journal_named(fresh_index, rank),
                              adds=per_journal_adds,
                              removal_pairs=per_journal_pairs)
    total_bytes = sum(os.path.getsize(journal_named(fresh_index, rank))
                      for rank in RANKS)
    assert total_bytes > 4 * disk_index.MAX_READ_PER_REFRESH, (
        "the fixture stopped meaning anything: each journal must be far deeper "
        "than one refresh's budget")
    assert present(TARGET_KEY16) is False, "nothing has been read yet"

    out = disk_index.refresh(force=True)  # THE ONE REFRESH, as the offer path does
    assert out["ok"] and out["converged"] is True, out
    assert present(TARGET_KEY16) is True, (
        "the identity added at seq 1 and never removed is the exact span tonight's "
        "box had on disk and could not see; one refresh must answer True")

    counters = disk_index.counters()
    assert counters["catch_up_jumps"] == 0
    assert counters["bytes_skipped"] == 0
    # The net-live truth: the target plus its keepers; every churn pair cancelled.
    assert counters["entries"] == per_journal_adds + 1, (
        "`entries` is the net-live count over the WHOLE journal - adds minus "
        "removes - not the newest-window count the buried rule reported")
    for index in range(per_journal_adds):
        assert present(key_for(index)) is True


def test_convergence_is_monotone_and_every_bound_is_a_number(fresh_index,
                                                             monkeypatch):
    """partial -> more -> complete, with the truth at each step and a hard bound.

    The boot budget is shrunk so convergence LEGITIMATELY takes more than one
    refresh - the task's clause - and then: every non-boot refresh reads at most
    `MAX_READ_PER_REFRESH` bytes (an exact number, asserted from the published
    `bytes_read`), applies at most `MAX_LINES_PER_REFRESH` lines, the lag strictly
    decreases, entries never fall on an adds-only journal, `unread_journals` names
    the file and its bytes the whole way, and `converged` flips exactly once.
    """
    monkeypatch.setattr(disk_index, "BOOT_DRAIN_MAX_BYTES", 128 * 1024)
    monkeypatch.setattr(disk_index, "BOOT_DRAIN_DEADLINE_SECONDS", 60.0)
    path = journal_named(fresh_index, "PP0_TP0")
    with open(path, "w", encoding="utf-8") as handle:
        seq = 1
        for index in range(20_000):
            handle.write(line(seq, disk_index.SPILL_LINE, key_for(index)))
            seq += 1
    seen_partial = False
    last_lag = None
    last_entries = 0
    steps = 0
    while True:
        out = disk_index.refresh(force=True)
        steps += 1
        assert out["ok"], out
        if steps > 1:
            # Post-boot passes: the ONE-pass steady-state bound, as an exact number.
            assert out["bytes_read"] <= disk_index.MAX_READ_PER_REFRESH, (
                f"a steady refresh read {out['bytes_read']} bytes, over the "
                f"{disk_index.MAX_READ_PER_REFRESH}-byte cap")
            assert out["lines"] <= disk_index.MAX_LINES_PER_REFRESH
        if not out["converged"]:
            seen_partial = True
            assert out["lag_bytes"] > 0
            assert disk_index.counters()["unread_journals"], (
                "a partial index must NAME its unread journals, not just a total")
        if last_lag is not None:
            assert out["lag_bytes"] < last_lag, (
                "every refresh must strictly advance the drain")
            assert out["entries"] >= last_entries, (
                "an adds-only drain must never lose ground")
        last_lag = int(out["lag_bytes"])
        last_entries = int(out["entries"])
        if out["converged"]:
            break
        assert steps < 100, "the drain must converge in a computable number of steps"
    assert seen_partial, "the fixture must actually have been deeper than one drain"
    assert disk_index.counters()["entries"] == 20_000
    assert present(key_for(0)) is True, "the FIRST line is the one the jump killed"


# --- the shape that hid the truth: mostly-removals ----------------------------

def test_entries_equals_net_live_over_a_majority_of_removals(fresh_index):
    """Tonight's ratio: ~1,900 `-` per 146 keepers, and the index lands on 146+1.

    A window-based implementation could pass an adds-only test by luck (the tail
    window is all adds); this one has no lucky window - the newest ~131 KiB is
    ~93% removal lines, which is exactly why the buried rule reported 146 for the
    WHOLE store instead of the truth. The drain applies every line once, in order,
    and reports the net-live count exactly.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_incident_journal(path, adds=146, removal_pairs=1_900)
    disk_index.refresh(force=True)
    counters = disk_index.counters()
    assert counters["converged"] is True
    assert counters["entries"] == 147, "146 keepers + the target; churn cancels out"
    assert counters["spills"] == 1 + 146 + 1_900
    assert counters["removals"] == 1_900
    assert counters["bytes_skipped"] == 0


def test_a_converged_then_idle_journal_never_resurrects_or_re_reads(fresh_index):
    """Convergence is a state, not a re-reading habit: the tail alone is read next.

    `+ X` before convergence, `- X` after, one refresh: X stays ABSENT (no
    duplicate application, no resurrection from any later pass), and the refresh
    read exactly the appended bytes - the cursor, not a windowing trick, is what
    stops history from being re-read.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(line(1, disk_index.SPILL_LINE, TARGET_KEY16))
        for index in range(2_000):
            handle.write(line(2 + index, disk_index.SPILL_LINE, key_for(index)))
    disk_index.refresh(force=True)
    assert present(TARGET_KEY16) is True
    before = disk_index.counters()["bytes_read"]
    appended = line(2_002, disk_index.REMOVE_LINE, TARGET_KEY16)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(appended)
    out = disk_index.refresh(force=True)
    assert out["bytes_read"] == len(appended.encode()), (
        "a converged journal reads ONLY its new tail - no history re-read")
    assert disk_index.counters()["bytes_read"] == before + len(appended.encode())
    assert present(TARGET_KEY16) is False
    again = disk_index.refresh(force=True)
    assert again["applied"] is False
    assert present(TARGET_KEY16) is False, "and no later pass resurrects it"


# --- the memory caps, loudly ---------------------------------------------------

def test_a_beyond_capacity_store_drops_oldest_and_says_so_once(fresh_index,
                                                               monkeypatch,
                                                               caplog):
    """The cap is drop-oldest, counted, AND loud - never a caught-up costume.

    At `MAX_INDEX_ENTRIES` the converged drain keeps the NEWEST identities (the
    restore path wants recent spills), counts every drop in `entries_dropped`,
    flips `entry_cap_hit` to true, and logs ONE warning naming the cap. What it
    must never do is present a capped index as a caught-up one: the section and
    the counters both carry the flag.
    """
    monkeypatch.setattr(disk_index, "MAX_INDEX_ENTRIES", 200)
    path = journal_named(fresh_index, "PP0_TP0")
    with open(path, "w", encoding="utf-8") as handle:
        seq = 1
        for index in range(1_000):
            handle.write(line(seq, disk_index.SPILL_LINE, key_for(index)))
            seq += 1
    with caplog.at_level("WARNING"):
        disk_index.refresh(force=True)
    counters = disk_index.counters()
    assert counters["entries"] == 200
    assert counters["entries_dropped"] == 800
    assert counters["entry_cap_hit"] is True
    assert disk_index.section()["entry_cap_hit"] is True
    assert present(key_for(999)) is True, "the cap keeps the NEWEST identities"
    assert present(key_for(0)) is False, "the oldest are the ones it dropped"
    shouts = [row for row in caplog.text.splitlines() if "identity cap" in row]
    assert len(shouts) == 1, (
        "the cap shouts once per boot, not once per dropped identity")


# --- the sentences an operator reads at 3 a.m. ---------------------------------

def _write_rank(directory, rank, *, installed=True, fitted=True):
    (directory / f"rank-{rank}.json").write_text(json.dumps({
        "version": 1, "epoch": "aabbccdd", "ts": round(time.time(), 6),
        "role": "worker", "rank": rank, "arena": {}, "counters": {},
        "writer": {"epoch_scope": "process"},
        "disk_restore": {"installed": installed, "armed": True,
                         "worker_fitted": fitted,
                         "reads_applied": 0}}))


@pytest.fixture
def restore_state(tmp_path, monkeypatch):
    """A scheduler-side boot: diag dir, rank files, arm file, arms raised."""
    diag = tmp_path / "diag2"
    diag.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setattr(config, "diagnostics_dir", lambda: diag)
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())

    def arm():
        # A LATCH (0.2.35): armed is all the file has to say. No `expires_ts`, because
        # the reader ignores it and writing one here would be testing an inert field.
        (diag / disk_spill.ARM_FILE).write_text(json.dumps(
            {"armed": True, "latched": True}))

    def disarm_file():
        (diag / disk_spill.ARM_FILE).unlink(missing_ok=True)

    return diag, arm, disarm_file


def test_why_no_restore_names_the_drain_even_when_lag_bytes_is_zero(restore_state,
                                                                    monkeypatch):
    """THE BLIND SPOT, pinned in the sentence an operator reads first.

    `why_no_restore` used to speak only when `lag_bytes != 0` - and after a buried
    jump the cursor sits at EOF so `lag_bytes` is legitimately 0: precisely the
    case that went silent for hours. Now the sentence fires on BOTH blindnesses:
    a drain in flight (lag > 0, journals named) and forfeited history (jumps or
    skipped bytes > 0 with lag 0), and it says in as many words that spans in an
    unread or skipped region are NOT KNOWABLE, so `no_disk_run` reads as 'spilled
    but not yet indexed', not 'never spilled'.

    The chain in `scheduler_section` walks layer -> gate -> index -> ranks ->
    offers, so the test raises the earlier links through their own published
    doors (the same objects every other test in this repo patches) and pins the
    index-blindness sentence in the `why_no_restore` string itself.
    """
    diag, arm, _ = restore_state
    arm()
    for rank in RANKS:
        _write_rank(diag, rank)
    monkeypatch.setattr(disk_restore, "fitted_methods_here",
                        lambda: list(disk_restore.SCHED_METHODS))
    monkeypatch.setattr(disk_restore, "gate",
                        lambda: {"ok": True, "reason": None, "reason_kind": None})
    monkeypatch.setattr(
        disk_restore, "worker_proof",
        lambda force=False: {"ok": True, "count": 4, "ranks": list(RANKS),
                             "ranks_armed": 4, "ranks_seen": 4,
                             "reads_applied": 0, "reason_kind": None,
                             "reason": None})
    monkeypatch.setitem(disk_restore._state, "sched_fitted", True)
    monkeypatch.setitem(disk_restore._state, "offers", 0)

    # Case A: a DRAIN IN FLIGHT (lag > 0). Shrink the refresh budget so one pass
    # leaves a real unread run, then ask the section the question.
    journal = journal_named(diag, "PP0_TP0")
    with open(journal, "w", encoding="utf-8") as handle:
        seq = 1
        for index in range(30_000):
            handle.write(line(seq, disk_index.SPILL_LINE, key_for(index)))
            seq += 1
    # A small second journal, so the named-unread set is not a single anonymous
    # total: the sentence must NAME the draining journals.
    shallow = journal_named(diag, "PP1_TP0")
    with open(shallow, "w", encoding="utf-8") as handle:
        handle.write(line(1, disk_index.SPILL_LINE, TARGET_KEY16))

    original = disk_index.MAX_READ_PER_REFRESH
    original_boot = disk_index.BOOT_DRAIN_MAX_BYTES
    disk_index.MAX_READ_PER_REFRESH = 4096      # force a partial drain
    disk_index.BOOT_DRAIN_MAX_BYTES = 8192      # cap the boot drain too
    try:
        first = disk_index.refresh(force=True)
    finally:
        disk_index.MAX_READ_PER_REFRESH = original
        disk_index.BOOT_DRAIN_MAX_BYTES = original_boot
    assert first["converged"] is False and first["lag_bytes"] > 0

    section = disk_restore.scheduler_section()
    why = section["why_no_restore"]
    assert "NOT CAUGHT UP" in why, why
    assert str(disk_index.counters()["lag_bytes"]) in why.replace(",", ""), (
        "the sentence names the unread BYTES")
    assert "disk-spill-journal-" in why, (
        "the sentence NAMES the draining journals")
    assert "not knowable" in why.lower()
    assert "spilled but not yet indexed" in why
    assert section["index"]["blind"] is True
    assert section["index"]["converged"] is False

    # Case B: FORFEITED HISTORY with lag_bytes == 0 (the shape that was silent).
    disk_index.reset_state_for_tests()
    disk_index._state["catch_up_jumps"] = 4
    disk_index._state["bytes_skipped"] = 4_186_980
    disk_index._state["converged"] = True       # the LIE the buried rule told
    disk_index._state["boot_drained"] = True
    disk_index._state["refreshes"] = 3
    why = disk_restore.scheduler_section()["why_no_restore"]
    assert "FORFEITED HISTORY" in why, why
    assert "4186980" in why.replace(",", "")
    assert "never as 'never spilled'" in why


def test_no_disk_run_refusal_distinguishes_unindexed_from_never_spilled(
        restore_state, monkeypatch):
    """The refusal text itself says WHICH unknown it is, from data already at hand.

    With a drain in flight, a `no_disk_run` refusal must carry the blindness
    sentence: 'spilled but not yet indexed', not a bare 'the SSD tier holds no
    contiguous span' that reads as 'never spilled'. No scan of the store is
    performed to find out - the distinction comes from `disk_index.index_lag()`,
    which is pure memory.
    """
    diag, arm, _ = restore_state
    arm()
    for rank in RANKS:
        _write_rank(diag, rank)
    journal = journal_named(diag, "PP0_TP0")
    with open(journal, "w", encoding="utf-8") as handle:
        seq = 1
        for index in range(30_000):
            handle.write(line(seq, disk_index.SPILL_LINE, key_for(index)))
            seq += 1
    from kvtier_dev_plugin import disk_index as di  # noqa: PLC0415 - same module
    original = di.MAX_READ_PER_REFRESH
    original_boot = di.BOOT_DRAIN_MAX_BYTES
    di.MAX_READ_PER_REFRESH = 4096
    di.BOOT_DRAIN_MAX_BYTES = 8192
    try:
        first = di.refresh(force=True)
    finally:
        di.MAX_READ_PER_REFRESH = original
        di.BOOT_DRAIN_MAX_BYTES = original_boot
    assert first["converged"] is False

    verdict = disk_restore._index_blindness()
    assert verdict["blind"] is True
    sentence = verdict["sentence"]
    assert "NOT CAUGHT UP" in sentence
    assert "spilled but not yet indexed" in sentence
    assert "never spilled" in sentence       # the negation is explicit

    # And once the index IS caught up, the refusal stops claiming blindness: the
    # sentence must not be attached to a `no_disk_run` from a converged index.
    while not di.refresh(force=True)["converged"]:
        pass
    assert disk_restore._index_blindness()["blind"] is False


# --- the arm proof (change 3) ---------------------------------------------------

def test_worker_proof_passes_with_stale_rank_armed_false_and_a_live_arm_file(
        restore_state):
    """THE 2026-10-20 LOCKOUT, as a test: rank files say armed:false, arm file says
    armed - the proof must PASS and count the stale claims, not refuse.

    The lockout: an arm with no spill activity rewrote no rank file (the writer
    runs on cadence and a signature change, and the signature never hashed the
    armed verdict), so the proof read four honest-but-stale `armed: false` files
    and refused with `workers_not_servicing` for 120 seconds although every worker
    would have served. The armed fact is rank-agnostic and its authority is the
    arm file; the rank files keep only the per-rank facts they uniquely know.
    """
    diag, arm, disarm = restore_state
    for rank in RANKS:
        (diag / f"rank-{rank}.json").write_text(json.dumps({
            "version": 1, "epoch": "aabbccdd", "ts": round(time.time(), 6),
            "role": "worker", "rank": rank, "arena": {}, "counters": {},
            "writer": {"epoch_scope": "process"},
            "disk_restore": {"installed": True, "armed": False,
                             "armed_reason": "the disk tier is disarmed",
                             "worker_fitted": True,
                             "service_reason_kind": "disarmed",
                             "reads_applied": 2}}))
    # WITHOUT the arm file: refuse, and name the arm file as the authority.
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "tier_arm_disarmed"
    assert "arm file" in verdict["reason"]

    arm()
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is True, verdict["reason"]
    assert verdict["ranks_armed"] == 4 and verdict["ranks_seen"] == 4
    assert verdict["reads_applied"] == 8
    assert verdict["stale_arm_claims"] == 4, (
        "the stale `armed: false` claims are COUNTED and published, not trusted "
        "and not silently ignored")
    assert verdict["tier_armed"] is True

    # AN EXPIRED-LOOKING FILE NO LONGER REFUSES (0.2.35): the proof passes on a file
    # whose `expires_ts` is in the past, because the arm is a LATCH and the reader reads
    # no clock. Removing that check is what keeps a box that has been serving for hours
    # from going disarmed on a timestamp nothing honours any more.
    (diag / disk_spill.ARM_FILE).write_text(json.dumps(
        {"armed": True, "expires_ts": time.time() - 60}))
    assert disk_restore.worker_proof(force=True)["ok"] is True

    # The absence still refuses - that is the real negative, and the fence below is the
    # other one (a file from another boot).
    (diag / disk_spill.ARM_FILE).unlink()
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "tier_arm_disarmed"

    # FOREIGN BOOT (the boot fence, unchanged): refuses even with `armed: true`.
    (diag / disk_spill.ARM_FILE).write_text(json.dumps(
        {"armed": True, "expires_ts": time.time() + 600,
         "boot_id": "not-this-boot" * 4}))
    boot_ok = True
    try:
        with open(disk_spill.BOOT_ID_PATH, "r", encoding="utf-8") as handle:
            boot_ok = bool(handle.read().strip())
    except OSError:
        boot_ok = False
    verdict = disk_restore.worker_proof(force=True)
    if boot_ok:
        assert verdict["ok"] is False, (
            "a reboot must not arm a cold box: the fence is `_read_arm_file`'s "
            "and the proof inherits it untouched")
        assert verdict["reason_kind"] == "tier_arm_disarmed"
    else:
        # No boot id readable here: the fence cannot PROVE a mismatch, and
        # `disarm()` semantics are unchanged - the file's other checks still rule.
        # Remove it and the refusal must come back for `no arm file`.
        assert verdict["ok"] is True
    disarm()
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "tier_arm_disarmed"


def test_worker_proof_still_vetoes_the_per_rank_facts_it_uniquely_knows(
        restore_state):
    """The AND over live ranks survives - it just reads it from the RIGHT fields.

    A rank whose worker half is NOT fitted, or that has no disk root of its own,
    still vetoes the whole offer exactly as the 2026-10-05 incident demands, with
    the same `scheduler_armed_workers_disarmed` kind and the same `ranks_armed=n/m`
    split. What changed is only WHICH field says no: per-rank fit/config facts
    (which the rank file uniquely knows) veto; the rank-agnostic armed verdict no
    longer does, because the proof now owns its authoritative copy.
    """
    diag, arm, _ = restore_state
    arm()

    def write(rank, **section):
        base = {"installed": True, "armed": False, "worker_fitted": True,
                "reads_applied": 1}
        base.update(section)
        (diag / f"rank-{rank}.json").write_text(json.dumps({
            "version": 1, "epoch": "aabbccdd", "ts": round(time.time(), 6),
            "role": "worker", "rank": rank, "arena": {}, "counters": {},
            "writer": {"epoch_scope": "process"},
            "disk_restore": base}))

    write("PP0_TP0")
    write("PP1_TP0", worker_fitted=False)
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "scheduler_armed_workers_disarmed"
    assert verdict["ranks_armed"] == 1 and verdict["ranks_seen"] == 2
    assert "not fitted in this worker" in verdict["reason"]

    write("PP1_TP0", worker_fitted=True, service_reason_kind="not_configured")
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert "no disk root" in verdict["reason"]

    write("PP2_TP0", installed=False)
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert "not fitted" in verdict["reason"]

    # A stale rank file (past `RANK_FILE_FRESH_SECONDS`) is not evidence at all:
    # the idle-box half of the hole is honestly reported as "no proof", never as
    # "armed by a heartbeat from a worker that may be gone".
    (diag / "rank-PP0_TP0.json").unlink()
    (diag / "rank-PP1_TP0.json").unlink()
    (diag / "rank-PP2_TP0.json").write_text(json.dumps({
        "version": 1, "epoch": "aabbccdd", "ts": time.time() - 3600.0,
        "role": "worker", "rank": "PP2_TP0", "arena": {}, "counters": {},
        "writer": {"epoch_scope": "process"},
        "disk_restore": {"installed": True, "armed": True,
                         "worker_fitted": True, "reads_applied": 1}}))
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "no_worker_proof"
