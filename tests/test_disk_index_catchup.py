r"""The reader DRAINS every journal front to back, in bounded passes, and says so.

WHY THIS SUITE EXISTS, TWICE. On 2026-10-05 the disk store held 5,707 files and the
restore path was offered none of the 36 newest identities: `refresh()` advanced a
per-journal cursor FROM BYTE 0 and shared one 512 KiB budget across four ~60k-line
journals, so the newest lines were the last thing a reader would reach. The 0.2.22
answer - read each journal's TAIL and FORFEIT everything older - fixed that afternoon
and caused the next incident: on 2026-10-20 four journals totalling 4,711,140 bytes /
132,099 records with ~6,290 net-live identities reported `entries: 146`, because every
first pass BURIED the history (jump to the tail, `_forget_rank`, cursor at EOF), the
jump landed `lag_bytes` at 0 so every payload read as caught up, and a restart
re-buried deterministically. `no_disk_run` refused every offer for hours.

The 0.2.28 rule is a DRAIN, and these tests pin it:

1. A journal deeper than one pass is read FROM ITS CURSOR FORWARD and the refresh
   RETURNS WITH THE CURSOR SHORT OF EOF. Nothing is forfeited, nothing forgotten;
   the next refresh continues where this one stopped.
2. THE FIRST REFRESH DRAINS TO EOF under its own budget (`BOOT_DRAIN_MAX_BYTES`,
   `BOOT_DRAIN_DEADLINE_SECONDS`) so a cold boot converges before the first offer.
3. EVERY LATER REFRESH IS ONE BOUNDED PASS: at most `MAX_READ_PER_REFRESH` bytes and
   `MAX_LINES_PER_REFRESH` lines, and the bytes it READ are published (`bytes_read`)
   so the bound is assertable, not inferable.
4. CONVERGENCE IS MONOTONIC AND MEASURED: partial, then more, then `converged:
   true`; `lag_bytes`/`unread_journals` tell the truth at each step; a converged
   journal then reads only its new tail.
5. EVERY JOURNAL GETS A SHARE, so the fleet drains together - a span needs every
   rank, and a uniformly-behind drain is honest where a one-rank-caught-up index
   would offer a load that must fail.
6. NOTHING IS SILENT: `converged`, `lag_bytes`, `lag_journals`, `unread_journals`,
   `bytes_read`, and the legacy `catch_up_jumps`/`bytes_skipped` (which a draining
   build can never raise) are published; a refresh inside the age floor re-publishes
   the last MEASURED lag rather than claiming zero.
7. THE CAPS ARE THE CAPS. Nothing here widens a steady-state budget; the tests
   exercise the drain by writing journals deeper than the existing caps, and the
   memory caps are pinned as LOUD, not silent.

Stdlib plus the plugin; `tmp_path` for every file; no engine, no threads.
"""

from __future__ import annotations

import os

import pytest

from kvtier_dev_plugin import config, disk_index, disk_layout

GROUP = 0
WIDTH = 11_468_800


def key_for(index: int) -> str:
    """A distinct, valid identity for line `index` (16 hex, the wire shape)."""
    return f"{index:016x}"


def identity_for(index: int, group: int = GROUP) -> str:
    return disk_index.identity_of(key_for(index), group)


def present(index: int, group: int = GROUP) -> bool:
    """The offer path's own question, asked through the offer path's own function."""
    return disk_index.contains(key_for(index) + "0" * 32 + f"{group:08x}")


def journal_named(directory, rank: str) -> str:
    return os.path.join(str(directory),
                        f"{disk_index.JOURNAL_PREFIX}{rank}"
                        f"{disk_index.JOURNAL_SUFFIX}")


def line_for(seq: int, op: str, index: int, group: int = GROUP) -> str:
    return (f"{seq}{disk_index.LINE_SEP}{op}{disk_index.LINE_SEP}"
            f"{key_for(index)}{disk_index.LINE_SEP}{WIDTH}"
            f"{disk_index.LINE_SEP}{group}\n")


def seed_journal(path, count: int, *, start_seq: int = 1, group: int = GROUP,
                 removal_every: int = 0) -> int:
    """Append `count` well-formed spill lines to the journal file.

    Written directly rather than through `journal_append` because these tests need
    tens of thousands of lines to be genuinely deeper than the caps, and the writer
    would spend that time stat'ing and seeding. The bytes are the same bytes the
    writer would have produced, so the reader is tested against its real contract.
    APPEND mode, because "the journal grew" is the event under test as often as "the
    journal exists". `removal_every` interleaves `-` lines (removing the identity the
    same line index would have added, which the drain must then keep ABSENT).
    """
    lines = []
    seq = start_seq
    for index in range(count):
        op = disk_index.SPILL_LINE
        if removal_every and index and index % removal_every == 0:
            op = disk_index.REMOVE_LINE
        lines.append(line_for(seq, op, index, group))
        seq += 1
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("".join(lines))
    return seq - 1


@pytest.fixture(autouse=True)
def fresh_index(tmp_path, monkeypatch):
    """A diagnostics directory and a per-process reset, as every index test needs."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    disk_index.reset_state_for_tests()
    yield directory
    disk_index.reset_state_for_tests()


# --- 1/2. the drain, and the boot that runs it --------------------------------

def test_the_boot_drain_reads_the_whole_journal_and_sees_the_newest(fresh_index):
    """A 20k-line journal, ONE refresh: every identity is there, nothing forfeited.

    The boot drain repeats the bounded pass under its own budget, so the first
    refresh of the process is the last one a cold boot needs. The newest identity is
    visible, the OLDEST is visible too (the 0.2.22 reader forfeited it), and the
    forfeit counters stay at zero because the drain has no forfeit path at all.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    total = 20_000
    seed_journal(path, total)
    assert present(total - 1) is False, "nothing has been read yet"
    out = disk_index.refresh(force=True)
    assert out["ok"] and out["applied"], out
    assert present(total - 1) is True, "the newest identity is known"
    assert present(0) is True, (
        "the identity the tail-window rule forfeited is the one the drain reads "
        "FIRST - that is the whole difference between the two rules")
    assert out["converged"] is True, out
    counters = disk_index.counters()
    assert counters["entries"] == total
    assert counters["bytes_skipped"] == 0
    assert counters["catch_up_jumps"] == 0
    assert counters["lines_skipped"] == 0
    assert counters["bytes_read"] >= os.path.getsize(path), (
        "the drain read every byte at least once")
    assert counters["bytes_read"] <= 3 * os.path.getsize(path), (
        "a window the LINE cap truncates has its unapplied tail re-read by the next "
        "pass (the cursor only advances past APPLIED lines) - the amplification is "
        "bounded because every truncated pass still advances exactly "
        "`line_budget` lines; 3x is the ceiling this line shape can reach")


def test_entries_after_a_drain_is_the_true_net_live_count(fresh_index):
    """Adds minus removes, across the WHOLE journal, not the newest window.

    Tonight's shape at one-thousandth scale: 146 long-lived identities added at LOW
    sequence numbers, then ~1,900 add/remove pairs piled on top - so the tail window
    the old reader took was 93% removal lines and the index reported what the window
    happened to hold. The drain applies the file in order and lands on the truth.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    keepers = 146
    churn = 1_900
    seq = seed_journal(path, keepers)              # keepers: + at low seq, never -
    # churn keys are distinct from the keepers' keys: added and removed as pairs.
    with open(path, "a", encoding="utf-8") as handle:
        next_seq = seq + 1
        for offset in range(churn):
            key = f"{9_000_000 + offset:016x}"
            handle.write(f"{next_seq}{disk_index.LINE_SEP}{disk_index.SPILL_LINE}"
                         f"{disk_index.LINE_SEP}{key}{disk_index.LINE_SEP}{WIDTH}"
                         f"{disk_index.LINE_SEP}{GROUP}\n")
            next_seq += 1
            handle.write(f"{next_seq}{disk_index.LINE_SEP}{disk_index.REMOVE_LINE}"
                         f"{disk_index.LINE_SEP}{key}{disk_index.LINE_SEP}{WIDTH}"
                         f"{disk_index.LINE_SEP}{GROUP}\n")
            next_seq += 1
    out = disk_index.refresh(force=True)
    assert out["converged"] is True, out
    assert disk_index.counters()["entries"] == keepers, (
        "net-live is what a converged index holds: every keeper, no churn residue")
    for index in range(keepers):
        assert present(index) is True


# --- 3. the steady-state bound is a NUMBER, not a vibe ------------------------

def test_every_refresh_after_the_boot_is_one_bounded_pass(fresh_index, monkeypatch):
    """With the boot budget shrunk, the drain advances one hard-capped pass a time.

    A single non-boot refresh reads AT MOST `MAX_READ_PER_REFRESH` bytes and applies
    AT MOST `MAX_LINES_PER_REFRESH` lines - exact numbers, asserted from the
    published `bytes_read`, not from a "smallish" feeling. Convergence is monotone
    and its steps tell the truth: `lag_bytes` strictly decreases, `unread_journals`
    names the journal and its unread bytes while it drains, and `converged` flips
    exactly once, at the end.
    """
    monkeypatch.setattr(disk_index, "BOOT_DRAIN_MAX_BYTES", 256 * 1024)
    monkeypatch.setattr(disk_index, "BOOT_DRAIN_DEADLINE_SECONDS", 60.0)
    path = journal_named(fresh_index, "PP0_TP0")
    size = 0
    seed_journal(path, 20_000)
    size = os.path.getsize(path)
    first = disk_index.refresh(force=True)
    assert first["bytes_read"] <= 256 * 1024, (
        "the boot drain is bounded by its own byte budget")
    assert first["converged"] is False
    assert first["lag_bytes"] > 0
    lag = first["lag_bytes"]
    learned = disk_index.counters()["entries"]
    assert learned > 0, "a partial drain is a partial index, not an empty one"
    unread = disk_index.counters()["unread_journals"]
    assert unread.get(os.path.basename(path)) == lag, (
        "the payload names the journal and its unread bytes, exactly")
    passes = 1
    while not disk_index.counters()["converged"]:
        out = disk_index.refresh(force=True)
        passes += 1
        assert out["bytes_read"] <= disk_index.MAX_READ_PER_REFRESH, (
            "a non-boot refresh never reads more than one full budget")
        assert out["lines"] <= disk_index.MAX_LINES_PER_REFRESH
        assert out["lag_bytes"] < lag, "the drain advances every pass, never stalls"
        assert disk_index.counters()["entries"] >= learned
        lag = out["lag_bytes"]
        learned = disk_index.counters()["entries"]
        assert passes < 200, "the drain converged in a computable number of passes"
    assert disk_index.counters()["entries"] == 20_000
    assert disk_index.counters()["bytes_read"] >= size, (
        "the drain read every byte at least once; bytes past a line-capped window "
        "are honestly re-counted when the next pass re-reads them, because the "
        "cursor only ever advances past APPLIED lines")


def test_the_line_cap_pauses_the_pass_and_the_next_one_continues(fresh_index):
    """A window deeper than the LINE cap stops applying - it does NOT skip.

    The 0.2.22 rule forfeited the OLDEST lines of the window when the line cap
    bound; under the drain the cap stops the pass EARLY (lines in file order from
    the cursor) and leaves the rest unread for the next refresh. No line is ever
    passed over: `lines_skipped` stays 0 because nothing was skipped.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    # One journal, byte-budget enough to carry everything, so ONLY the line cap can
    # bind: 40-byte lines x 10k < 512 KiB.
    seed_journal(path, 10_000)
    monkey_off = min(disk_index.MAX_READ_PER_REFRESH, os.path.getsize(path))
    assert monkey_off == os.path.getsize(path), (
        "the fixture stopped meaning anything: bytes must not bind before lines")
    out = disk_index.refresh(force=True)   # boot drain: several passes, then EOF
    assert out["converged"] is True
    counters = disk_index.counters()
    assert counters["lines_skipped"] == 0, "nothing is passed over any more"
    assert counters["dropped"]["line_cap"] >= 1, (
        "the line cap did bind during the drain - it pauses, it does not forfeit")


# --- 4. what the applied order is allowed to do -------------------------------

def test_an_add_then_remove_stays_absent_across_passes(fresh_index):
    """The drain's one ordering promise: a `-` read after its `+` KEEPS it gone.

    Split across passes deliberately: the identity's `+` is drained in one refresh
    and its `-` in a later one, so a resurrection by a subsequent pass is not merely
    "the last window said so". The 0.2.22 rule preserved this inside one window;
    the drain must preserve it across windows, which is the property the "no
    duplicate application" clause of the fix rests on.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    monkey_note = 20_000
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(line_for(1, disk_index.SPILL_LINE, 0))
        for index in range(2, monkey_note):     # padding past the first window
            handle.write(line_for(index, disk_index.SPILL_LINE, index))
    first = disk_index.refresh(force=True)      # the boot drain takes it all
    assert first["converged"] is True
    assert present(0) is True
    seed_journal_at_removal(path, monkey_note)
    second = disk_index.refresh(force=True)
    assert second["removals"] == 1
    assert present(0) is False, "the removal line wins, in whichever pass it lands"
    third = disk_index.refresh(force=True)
    assert third["applied"] is False
    assert present(0) is False, "and a later pass must not resurrect it"


def seed_journal_at_removal(path, seq):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line_for(seq, disk_index.REMOVE_LINE, 0))


def test_a_converged_journal_then_reads_only_its_new_tail(fresh_index):
    """No re-reading history: `bytes_read` after convergence equals the append.

    The "don't re-read history" optimization the buried branch claimed is provided
    by the cursor: a converged journal's next pass reads exactly the bytes appended
    since, applies them, and returns to EOF. If this ever regressed, a deep journal
    would silently cost 512 KiB of re-read per refresh AND replay old lines.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 5_000)
    disk_index.refresh(force=True)
    before = disk_index.counters()["bytes_read"]
    appended = line_for(5_001, disk_index.SPILL_LINE, 5_000)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(appended)
    out = disk_index.refresh(force=True)
    assert out["bytes_read"] == len(appended.encode()), (
        "one append in, one append read: the cursor is the no-re-read rule")
    assert disk_index.counters()["bytes_read"] == before + len(appended.encode())
    assert out["converged"] is True
    assert present(5_000) is True


# --- 5. fair share ------------------------------------------------------------

def test_one_deep_journal_cannot_starve_another_rank(fresh_index, monkeypatch):
    """Every journal gets a SHARE of the budget, so the fleet drains together.

    A span is restorable only when EVERY rank has journalled it, so an index caught
    up for one rank and behind for three is worse than uniformly behind: it turns
    "the file is not there" into an offered load that fails. With the boot budget
    shrunk so the deep rank cannot finish, the shallow rank STILL finishes in the
    same passes - the share divides the budget, it is not whatever the previous
    rank left over.

    SINCE 0.2.46 THE OFFER ANSWER FOLLOWS THE DRAIN PER RANK, which sharpens what this
    test can assert. The shallow rank's identity is READ in the same pass (the entry
    exists, the fair share did its job), but it is not OFFERABLE while the deep rank's
    cursor sits short of it: presence is an intersection over the ranks that have to
    read, and a rank whose journal has not been drained that far cannot be proved to
    hold the file. The fair share is what makes that intersection honest, and the last
    assertion is the pair of them working: once the deep journal converges, the span
    becomes offerable.
    """
    monkeypatch.setattr(disk_index, "BOOT_DRAIN_MAX_BYTES", 128 * 1024)
    deep = journal_named(fresh_index, "PP0_TP0")
    shallow = journal_named(fresh_index, "PP1_TP1")
    seed_journal(deep, 15_000)
    seed_journal(shallow, 300)
    out = disk_index.refresh(force=True)
    assert out["journals"] == 2
    assert disk_index.entry_for(key_for(299), GROUP)["present"] is True, (
        "the shallow rank's newest identity is READ in the same pass a deep rank is "
        "being drained - the share protects it")
    assert present(299) is False, (
        "read is not offerable: the deep rank's cursor has not reached this identity, "
        "so its own file for it is unproven, and a restore reads one file per rank")
    unread = disk_index.counters()["unread_journals"]
    assert os.path.basename(shallow) not in unread, "the shallow journal is at EOF"
    assert os.path.basename(deep) in unread, "the deep journal is honestly draining"
    for _ in range(40):
        if disk_index.refresh(force=True)["converged"]:
            break
    assert disk_index.counters()["converged"] is True, "the drain finishes"
    assert present(299) is True, (
        "the fair share plus the intersection: once BOTH ranks' journals have been "
        "drained to this identity, it is offerable")


# --- 6. silence is the bug ----------------------------------------------------

def test_a_refresh_that_did_not_look_reports_the_last_measured_lag(fresh_index):
    """The age floor must not report `lag_bytes: 0` as if it had looked."""
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000)
    disk_index.refresh(force=True)
    seed_journal(path, 200, start_seq=20_001)
    skipped = disk_index.refresh()          # inside the floor: no read happens
    assert skipped["applied"] is False
    assert skipped["lag_measured"] is False, "this number is the previous pass's"


def test_the_lag_field_moves_and_returns_to_zero(fresh_index):
    """`lag_bytes` says how far behind the reader is, and 0 means caught up.

    Pinned at both ends: non-zero while a journal has unread bytes (the drain in
    flight), and exactly zero once every cursor sits at end-of-file. A small journal
    is the zero case at boot, because nothing was behind the cursor at all.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 50)
    out = disk_index.refresh(force=True)
    assert out["lag_bytes"] == 0 and out["lag_journals"] == 0, out
    assert out["converged"] is True
    assert out["bytes_skipped"] == 0 and out["jumps"] == 0
    seed_journal(path, 200, start_seq=51)
    again = disk_index.refresh(force=True)
    assert again["lag_bytes"] == 0, "one small append is still one cheap pass"
    assert again["spills"] == 200


def test_membership_publishes_the_drain_next_to_its_answer(fresh_index, monkeypatch):
    """`op=contains` answers with the reader's own freshness beside it.

    An operator reading `not_present` cannot otherwise tell "never spilled" from
    "this process has not read that line yet". The fields must be present in the
    SAME view object the verdict comes from, not only in a section nobody polls.
    """
    monkeypatch.setattr(disk_index, "BOOT_DRAIN_MAX_BYTES", 128 * 1024)
    monkeypatch.setattr(disk_index, "BOOT_DRAIN_DEADLINE_SECONDS", 60.0)
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000)
    disk_index.refresh(force=True)           # partial on purpose
    view = disk_index.membership(key_for(0) + "0" * 32 + f"{GROUP:08x}")
    assert view["index"]["converged"] is False
    assert view["index"]["lag_bytes"] > 0
    assert view["index"]["unread_journals"], "the draining journal is NAMED"
    assert disk_index.section()["lag_bytes"] == view["index"]["lag_bytes"], (
        "the verb and the snapshot must never disagree about the same number")


# --- 7. the caps stay, and a stopped journal stays honest ----------------------

def test_the_entry_cap_drops_oldest_and_cries_about_it(fresh_index, monkeypatch,
                                                       caplog):
    """A converged store bigger than the cap is LOUD, not silently truncated.

    The cap's policy is drop-oldest (insertion order) and every drop is counted;
    0.2.28 adds the part that was missing: once it bites, `entry_cap_hit` is true
    and the log says so ONCE per boot - a capped index answers `not_present` for
    files that are on disk, and that must never read as a caught-up index.
    """
    monkeypatch.setattr(disk_index, "MAX_INDEX_ENTRIES", 100)
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 500)
    with caplog.at_level("WARNING"):
        disk_index.refresh(force=True)
    counters = disk_index.counters()
    assert counters["entries"] == 100
    assert counters["entries_dropped"] == 400
    assert counters["entry_cap_hit"] is True
    lines = [row for row in caplog.text.splitlines()
             if "identity cap" in row]
    assert len(lines) == 1, "the cap shouts once per episode, not 400 times"


def test_a_stopped_journal_reports_lag_zero_and_names_itself(fresh_index):
    """A malformed line STOPS that journal; the drain calls it converged, honestly.

    Converged means "no further read is POSSIBLE from this process", not "I read
    every byte" - the stopped journal is published in `journals_stopped` beside it,
    and its unread bytes are permanently unread. The lines after the stop are the
    ONE remaining `lines_skipped` truth: they are skipped forever, not deferred.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(line_for(1, disk_index.SPILL_LINE, 1))
        handle.write("99 not-a-journal-line\n")
        handle.write(line_for(3, disk_index.SPILL_LINE, 3))
    out = disk_index.refresh(force=True)
    assert out["converged"] is True, out
    section = disk_index.section()
    assert section["journals_stopped"], "the stopped file is named, not swallowed"
    assert section["lines_skipped"] >= 1, (
        "the lines after the stop are permanently unread and counted as skipped")
    assert disk_index.lookup(key_for(1) + "0" * 32 + f"{GROUP:08x}")["present"] \
        is True, "what was applied before the stop stays applied"


def test_a_partial_last_line_is_never_applied_and_never_crossed(fresh_index):
    """A line the writer is halfway through is nobody's fact yet.

    The cursor advances only past COMPLETE lines, so a torn tail is retried by the
    next refresh - the invariant that lets the drain read from a raw byte offset
    without ever needing the alignment probe the tail-window rule paid for.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 10)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("11 + aaaaaaaaaaaaaaaa")     # no newline: not a fact yet
    out = disk_index.refresh(force=True)
    assert out["entries"] == 10
    assert out["lag_bytes"] == len("11 + aaaaaaaaaaaaaaaa")
    assert disk_index.counters()["converged"] is False
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{disk_index.LINE_SEP}{WIDTH} {GROUP}\n")
    again = disk_index.refresh(force=True)
    assert again["converged"] is True
    assert again["spills"] == 1
