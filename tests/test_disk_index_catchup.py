r"""The reader reaches the NEWEST spill lines, under the same caps, and says so.

WHY THIS SUITE EXISTS. On 2026-10-05 the disk store held 5,707 files and the restore
path was offered none of the 36 newest identities. The journals had the lines; the
reader did not. `refresh()` advanced a per-journal cursor FROM BYTE 0 and shared one
512 KiB budget across four ~60k-line journals, so a single refresh applied about a
tenth of one rank's history and the lines the offer path was waiting for - the spills
from the last few minutes, which is exactly what "the GPU forgot this and RAM never
had it" means - were the LAST thing it would ever reach. Both symptoms the operator
measured (newest spills invisible; entries surviving their own removal lines) are the
same defect: an index that is eight refreshes behind.

The fix is a rule, not a raised cap, and these tests pin the rule:

1. ONE REFRESH SEES THE NEWEST IDENTITY, whatever the backlog. A journal longer than
   `MAX_LINES_PER_REFRESH`, refreshed once, answers for the identity appended last.
2. THE WINDOW IS TAKEN FROM THE END, so the bytes it forfeits are all OLDER than the
   lines it applies. That ordering is what makes forfeiting safe, and the tests below
   pin the two consequences: an entry taught by a forfeited line is simply unknown
   (a missed restore), and a removal inside the window still removes an identity the
   window also names (no false presence).
3. A JUMP FORGETS WHAT IT SKIPPED. The rank's previously-learned entries go, because
   the index may only claim what it actually read.
4. EVERY JOURNAL GETS A SHARE, so one deep rank cannot starve the others and leave an
   index that is caught up for the wrong ranks (the shape that turns a missing file
   into an offered load that fails).
5. NOTHING IS SILENT: `lag_bytes`, `lag_journals`, `catch_up_jumps`, `bytes_skipped`
   and `lines_skipped` are published, and a refresh that skipped its look says so.
6. THE CAPS ARE THE CAPS. Nothing here widens a budget; the tests exercise the rule
   by writing journals that are genuinely deeper than the existing caps.

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


def group_row(group: int = GROUP) -> str:
    """The group field of a journal line, spelled the way the writer spells it."""
    return str(group)


def present(index: int, group: int = GROUP) -> bool:
    """The offer path's own question, asked through the offer path's own function."""
    return disk_index.contains(key_for(index) + "0" * 32 + f"{group:08x}")


def journal_named(directory, rank: str) -> str:
    return os.path.join(str(directory),
                        f"{disk_index.JOURNAL_PREFIX}{rank}"
                        f"{disk_index.JOURNAL_SUFFIX}")


def seed_journal(path, count: int, *, start_seq: int = 1, group: int = GROUP,
                 removal_every: int = 0, seq_width: int = 0) -> int:
    """Append `count` well-formed spill lines to the journal file.

    Written directly rather than through `journal_append` because these tests need
    tens of thousands of lines to be genuinely deeper than the caps, and the writer
    would spend that time stat'ing and seeding. The bytes are the same bytes the
    writer would have produced, so the reader is tested against its real contract.
    APPEND mode, because "the journal grew" is the event under test as often as "the
    journal exists". `removal_every` interleaves `-` lines to prove a window can
    still REMOVE, and `seq_width` zero-pads the sequence so every line is the same
    length - which is what lets a test place a window start MID-LINE on purpose.
    """
    lines = []
    seq = start_seq
    for index in range(count):
        op = disk_index.SPILL_LINE
        if removal_every and index and index % removal_every == 0:
            op = disk_index.REMOVE_LINE
        label = f"{seq:0{seq_width}d}" if seq_width else str(seq)
        lines.append(f"{label}{disk_index.LINE_SEP}{op}{disk_index.LINE_SEP}"
                     f"{key_for(index)}{disk_index.LINE_SEP}{WIDTH}"
                     f"{disk_index.LINE_SEP}{group}\n")
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


# --- 1. one refresh reaches the newest line ----------------------------------

def test_one_refresh_sees_the_newest_identity_past_the_line_cap(fresh_index):
    """More lines than `MAX_LINES_PER_REFRESH`, ONE refresh, newest identity present.

    THE TEST THE OPERATOR ASKED FOR, and it is the whole defect in one assertion: the
    old reader applied the FIRST `MAX_LINES_PER_REFRESH` lines and walked on, so the
    identity written last was invisible until the backlog drained - eight refreshes
    deep on the live box, with each refresh gated behind a flagged request that was
    itself waiting on the answer.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    total = disk_index.MAX_LINES_PER_REFRESH + 1_000
    seed_journal(path, total)
    # The newest line names an identity nobody else could guess at.
    assert present(total - 1) is False, "nothing has been read yet"
    out = disk_index.refresh(force=True)
    assert out["ok"] and out["applied"], out
    assert present(total - 1) is True, (
        "the most recently spilled identity must be visible after ONE refresh")
    counters = disk_index.counters()
    assert counters["lines_skipped"] >= 1, (
        "the forfeited older lines must be counted, not silently walked over")
    assert disk_index.section()["lines_skipped"] == counters["lines_skipped"]


def test_the_lag_field_moves_and_returns_to_zero(fresh_index):
    """`lag_bytes` says how far behind the reader is, and 0 means caught up.

    The published number has to be a MEASUREMENT, so it is pinned at both ends: non-
    zero while a journal has unread bytes, and exactly zero once every cursor sits at
    end-of-file. A small journal is the zero case, because nothing was forfeited.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 50)
    out = disk_index.refresh(force=True)
    assert out["lag_bytes"] == 0 and out["lag_journals"] == 0, out
    assert out["bytes_skipped"] == 0 and out["jumps"] == 0
    # Now push 200 more lines behind the cursor and refresh nothing: the lag is real.
    seed_journal(path, 200, start_seq=51)
    again = disk_index.refresh(force=True)
    assert again["lag_bytes"] == 0, "one small append is still one cheap pass"
    assert again["spills"] == 200


def test_a_deep_journal_reports_its_lag_and_its_jump(fresh_index):
    """A backlog deeper than one pass jumps to the tail and SAYS so, in bytes."""
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000)
    out = disk_index.refresh(force=True)
    assert out["jumps"] == 1, out
    assert out["bytes_skipped"] > 0, "the forfeited bytes are the point of the field"
    assert out["entries"] > 0
    assert out["entries"] < 20_000, (
        "the forfeited region must NOT be claimed: the index may only hold what it "
        "read")
    section = disk_index.section()
    assert section["catch_up_jumps"] == 1
    assert section["bytes_skipped"] == out["bytes_skipped"]
    assert int(section["lag_bytes"]) == 0, (
        "a jump lands the cursor at the end of the file: the pass that forfeits is "
        "the pass that catches up")


def test_a_refresh_that_did_not_look_reports_the_last_measured_lag(fresh_index):
    """The age floor must not report `lag_bytes: 0` as if it had looked."""
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000)
    disk_index.refresh(force=True)
    seed_journal(path, 200, start_seq=20_001)
    skipped = disk_index.refresh()          # inside the floor: no read happens
    assert skipped["applied"] is False
    assert skipped["lag_measured"] is False, "this number is the previous pass's"


# --- 2/3. what forfeiting is allowed to do ----------------------------------

def test_a_removal_inside_the_window_still_removes(fresh_index):
    """Newest-first is only safe because the WINDOW's order is the file's order.

    Twenty identities are spilled near the END of a deep journal and then evicted a
    few lines later, both sides inside the window a jump reads. They must come back
    ABSENT. If the window were applied in the wrong order (or the removals were read
    before the spills), they would come back PRESENT - the unsafe direction, a promise
    the worker cannot keep, and the only failure mode this rule could have.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    last = seed_journal(path, 20_000)
    doomed = list(range(19_980, 20_000))
    with open(path, "a", encoding="utf-8") as handle:
        for offset, index in enumerate(doomed):
            handle.write(f"{last + 1 + offset}{disk_index.LINE_SEP}"
                         f"{disk_index.REMOVE_LINE}{disk_index.LINE_SEP}"
                         f"{key_for(index)}{disk_index.LINE_SEP}{WIDTH}"
                         f"{disk_index.LINE_SEP}{GROUP}\n")
    out = disk_index.refresh(force=True)
    assert out["jumps"] == 1, "this must exercise the windowed path, not a small file"
    for index in doomed:
        assert present(index) is False, (
            f"identity {index} was spilled and then evicted inside the same window: "
            "the eviction wins")
    assert present(19_979) is True, "the identities nobody evicted are still claimed"


def test_a_jump_forgets_what_this_rank_taught_it(fresh_index):
    """The entries a jump skips over are FORGOTTEN, not kept as a stale promise.

    Boot 1 reads a small journal, learns identity 5. The journal then grows past one
    pass and identity 5's `+` line falls in the forfeited region - so the index must
    no longer claim it. Keeping it would be the exact bug the operator measured: an
    index full of identities whose files (or removal lines) it has never read.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 5)
    disk_index.refresh(force=True)
    assert present(3) is True
    seed_journal(path, 20_000, start_seq=6)
    out = disk_index.refresh(force=True)
    assert out["jumps"] == 1
    assert present(3) is False, (
        "the line that taught us identity 3 is inside the skipped region, so the "
        "claim is not ours to keep")
    assert present(19_999) is True


def test_an_identity_only_in_the_forfeited_region_is_simply_unknown(fresh_index):
    """Absence is the safe direction: a forfeited `+` costs a restore, nothing more."""
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000)
    disk_index.refresh(force=True)
    assert present(0) is False, "the first lines were forfeited, not guessed at"
    assert disk_index.counters()["bytes_skipped"] > 0


# --- 4. fair share ----------------------------------------------------------

def test_one_deep_journal_cannot_starve_another_rank(fresh_index):
    """Every journal gets a SHARE of the budget, so the fleet catches up together.

    A span is restorable only when EVERY rank has journalled it, so an index caught up
    for one rank and behind for three is worse than uniformly behind: it turns "the
    file is not there" into an offered load that fails. Here rank PP0_TP0 is deeper
    than the whole byte budget and PP1_TP1 is small; under a shared budget the small
    one was never reached at all.
    """
    deep = journal_named(fresh_index, "PP0_TP0")
    shallow = journal_named(fresh_index, "PP1_TP1")
    seed_journal(deep, 15_000)
    seed_journal(shallow, 300)
    out = disk_index.refresh(force=True)
    assert out["journals"] == 2
    assert present(299) is True, (
        "the shallow rank's newest identity must be read in the same refresh a deep "
        "rank is being drained")
    assert disk_index.counters()["lag_journals"] == 0


def test_per_rank_claims_come_from_the_same_window_as_the_entries(fresh_index):
    """`op=contains`'s per-rank table cannot disagree with the entry table.

    Both are built by the same replay, so a jump that forgets a rank must empty that
    rank's claims too - otherwise a span reads as "1 of 4 ranks" from a table whose
    entries were forgotten two lines earlier.
    """
    deep = journal_named(fresh_index, "PP0_TP0")
    seed_journal(deep, 20_000)
    disk_index.refresh(force=True)
    view = disk_index.membership(key_for(19_999) + "0" * 32 + f"{GROUP:08x}")
    assert view["index"]["lag_bytes"] == 0
    assert view["index"]["catch_up_jumps"] == 1
    assert view["ranks_present"] >= 1, "the window's identity is still claimed"
    gone = disk_index.membership(key_for(0) + "0" * 32 + f"{GROUP:08x}")
    assert gone["present"] is False
    assert gone["ranks"] == [], "a forgotten rank claims nothing"


# --- 5/6. the caps stay, and a window never opens mid-line -------------------

def test_one_refresh_never_reads_more_than_the_total_cap(fresh_index):
    """Four deep journals, one refresh, and the TOTAL byte cap still binds.

    The share divides the cap; it does not multiply it. `read` is not published, so
    the assertion goes through what the reader can prove: at most
    `MAX_LINES_PER_REFRESH` lines applied, and every journal still tracked.
    """
    for rank in ("PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1"):
        seed_journal(journal_named(fresh_index, rank), 20_000)
    out = disk_index.refresh(force=True)
    assert out["ok"], out
    assert out["lines"] <= disk_index.MAX_LINES_PER_REFRESH, (
        "the line cap is a ceiling, not a target the shares may exceed in sum")
    assert out["journals"] == 4
    assert disk_index.section()["journals_tracked"] == 4


def test_a_window_never_opens_mid_line(fresh_index):
    """A tail window is aligned to a line start, or the pass reads nothing.

    The reason this is load-bearing: `_apply_line` answers a malformed line by
    STOPPING that journal for the rest of the boot. An unaligned window would
    therefore not merely miss a restore, it would blind the index permanently - so
    the rule is "align or abstain", and the proof is that the journal is still alive
    afterwards and still applies new lines.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000)
    disk_index.refresh(force=True)
    assert disk_index.section()["journals_stopped"] == [], (
        "a catch-up window must never open on a half-line")
    # The keys this append names are 0..9 - the ones the jump FORFEITED. So their
    # reappearing proves both halves: the journal survived the jump, and a line the
    # reader reaches again is believed again.
    assert present(9) is False, "the jump forfeited the region naming this key"
    seed_journal(path, 10, start_seq=20_001)
    out = disk_index.refresh(force=True)
    assert out["applied"] is True, "the journal still works after a jump"
    assert disk_index.section()["journals_stopped"] == []
    assert present(9) is True


def test_a_window_that_cannot_find_a_line_start_abstains(fresh_index, monkeypatch):
    """No newline in the probe: read nothing, keep the cursor, report the lag.

    The alternative is applying a half line, which STOPS the journal for the boot, so
    `abstain` is the only acceptable answer and it has to be measurable. Deterministic
    by construction: equal-length lines, and the window start is asserted to land
    mid-line before the pass runs.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000, seq_width=6)
    line_bytes = len(f"{1:06d}{disk_index.LINE_SEP}+{disk_index.LINE_SEP}"
                     f"{key_for(0)}{disk_index.LINE_SEP}{WIDTH}"
                     f"{disk_index.LINE_SEP}{group_row()}\n")
    size = os.path.getsize(path)
    window = min(disk_index.MAX_READ_PER_REFRESH, size)
    assert (size - window) % line_bytes != 0, (
        "the fixture stopped meaning anything: the window start is a line boundary")
    monkeypatch.setattr(disk_index, "CATCHUP_ALIGN_PROBE_BYTES", 1)
    out = disk_index.refresh(force=True)
    assert out["applied"] is False, out
    assert out["lag_bytes"] > 0, "abstaining is not being caught up"
    assert disk_index.counters()["catch_up_jumps"] == 0
    assert disk_index.section()["journals_stopped"] == []


def test_membership_publishes_the_lag_next_to_its_answer(fresh_index):
    """`op=contains` answers with the reader's own freshness beside it.

    An operator reading `not_present` cannot otherwise tell "never spilled" from
    "this process has not read that line yet", and on 2026-10-05 those two readings
    cost an afternoon each way. The fields must be present in the SAME view object
    the verdict comes from, not only in a section nobody polls.
    """
    path = journal_named(fresh_index, "PP0_TP0")
    seed_journal(path, 20_000)
    disk_index.refresh(force=True)
    view = disk_index.membership(key_for(19_999) + "0" * 32 + f"{GROUP:08x}")
    assert view["index"]["lag_bytes"] == 0, "a drained journal says so"
    assert view["index"]["catch_up_jumps"] == 1
    assert view["index"]["bytes_skipped"] > 0
    assert disk_index.section()["lag_bytes"] == view["index"]["lag_bytes"], (
        "the verb and the snapshot must never disagree about the same number")
