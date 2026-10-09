r"""THE BOOT DRAINS THE SPILL JOURNAL BEFORE THE ENGINE SERVES (0.2.65).

THE NIGHT THIS EXISTS FOR. The box restarted at 23:00:43. Ninety seconds later a
request recomputed 129,285 tokens because the restore tier answered `blind: true`:
the journal replay was still running, so the index could not say whether the span was
on disk. The replay finished around 23:12 - eleven minutes after the boot, and about
six seconds of actual reading.

THE CAUSE WAS NEVER A DEADLOCK. `disk_restore._refresh_index()` runs on the offer path
BEFORE blindness is consulted (blindness is refusal text appended later), so the drain
advances regardless of the blind flag. What was wrong is that the drain is
TRAFFIC-DRIVEN and its boot phase is bounded by a 5-second deadline: the boot spent its
5 s with 2.3 MB of journal still unread, and the remainder dribbled out at
`MAX_READ_PER_REFRESH` (512 KiB) per ARRIVING REQUEST at a 1 s floor - ~11 minutes for
nothing.

WHAT THIS SUITE PINS:

1. THE FIX, AS A DEFAULT: a journal of tonight's shape converges INSIDE the boot's own
   budget, with no env var set and no traffic. The clock is the box's measured journal
   throughput (2.2 MB/s), because the deadline being tested is a wall-clock one; the
   model is pinned to be MORE demanding than the shipped default needs and LESS
   demanding than the deadline the tree shipped with (5.0 s) allowed - the assertion
   below says both numbers out loud, and the same test fails against the unpatched tree
   for exactly the reason tonight happened.
2. THE DEADLINE IS A CEILING, NOT A SUGGESTION: a journal deeper than the deadline stops
   at it, is REPORTED as not-converged, does not raise, and the rest keeps draining on
   the request path. A journal of any size cannot stall a boot.
3. THE BUDGET IS A BUDGET: the byte bound stops the same way.
4. A VALUE THE GRAMMAR REFUSES LEAVES THE DEFAULT STANDING AND NAMES ITS COMPLAINT in
   the payload - never a silent default, never a crash.
5. UNSET IS TODAY: the shipped numbers, the same steady-state per-refresh byte cap, and
   a drain that is READ-ONLY (the journals' bytes are identical afterwards).
6. THE OBSERVABILITY: the record is in the same index section of the `/disk` payload as
   `converged`/`lag_bytes`, and the boot line carries a bracketed token.
7. IT TOUCHES NO ARM AND NO POSTURE - behaviourally and as an AST fact about this
   module's own source, the same two-way pinning `test_mamba_anchor_patch.py` and
   `test_disk_authority_switch.py` use.
8. IT NEVER RAISES INTO A BOOT, and `BaseException` still travels.
9. IT LANDS AT CONNECTOR CREATION: the fork's own `create_connector` returns a connector
   whose index is ALREADY drained, so the seconds are paid by the boot and not by a
   user's request.
10. AN INERT BOOT DRAINS NOTHING.

Stdlib plus the plugin; `tmp_path` for every file; no engine, no threads, no vllm.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import re
import sys
import types

import pytest
import test_backport as TB
import test_disk_restore as base

from kvtier_dev_plugin import (
    backport,
    config,
    disk_index,
    disk_restore,
    disk_spill,
    guard_mode,
    snapshot,
)

# Re-exported as fixtures, not shadowed: a fresh boot per test, and a tmp_path the
# store-root validation is allowed to live under.
fresh = base.fresh  # noqa: F811
allow_tmp_root = base.allow_tmp_root  # noqa: F811

GROUP = 0
WIDTH = 11_468_800

# THE BOX'S MEASURED JOURNAL THROUGHPUT, in bytes read per second. The operator's
# arithmetic for tonight: 25 MB of journal at ~2.2 MB/s is ~11 s of reading, and the
# boot spent 5 of them. Any clock model in this file is this one number; inventing a
# second rate would make the deadline test measure the fixture instead of the box.
MEASURED_DRAIN_BYTES_PER_S = 2_200_000

# WHAT THE TREE SHIPPED WITH until 0.2.65. Named here so the deep-fixture test can say
# "this is the bound tonight hit", and so nobody can quietly re-tune a default back
# under the fixture without that test explaining itself differently.
RETIRED_BOOT_DRAIN_DEADLINE_SECONDS = 5.0

# THE FIXTURE'S JOURNAL: 4 ranks (one file per rank, as on the box), 110k lines each,
# keys cycling over 60k identities (a re-spill of a live identity is a new `+` line with
# a new seq - which is what a real journal is mostly made of). The key space is sized
# against the INDEX's own caps rather than by taste: 60k identities x 4 ranks = 240,000
# (identity, rank) pairs, inside `MAX_RANK_RECORDS` (262,144), and the convergence test
# asserts `rank_records_dropped == 0` so a fixture that drifted past a cap could not
# silently turn this suite into a test of the cap. ~16 MB: MORE than one refresh-pass of
# lines by a factor of ~30, and deep enough that at 2.2 MB/s its drain costs more wall
# time than the 5.0 s the boot used to allow.
DEEP_JOURNALS = 4
DEEP_LINES_PER_JOURNAL = 110_000
DEEP_KEY_SPACE = 60_000

LINE = "{} {} {} {} {}\n"


def key_for(index: int) -> str:
    return f"{index:016x}"


def present(index: int, group: int = GROUP) -> bool:
    """The offer path's own question, through the offer path's own function."""
    return disk_index.contains(key_for(index) + "0" * 32 + f"{group:08x}")


def journal_named(directory, rank: str) -> str:
    return os.path.join(str(directory), f"{disk_index.JOURNAL_PREFIX}{rank}"
                                       f"{disk_index.JOURNAL_SUFFIX}")


def line(seq: int, op: str, index: int, group: int = GROUP) -> str:
    """ONE WIRE LINE: `<seq> <+|-> <key16> <bytes> <group>` (disk_index.py:275)."""
    return LINE.format(seq, op, key_for(index), WIDTH, group)


def write_journal(path, lines: int, *, start_seq: int = 1,
                  key_space: int = 1) -> int:
    """`lines` well-formed spill lines, keys cycling over `key_space` identities."""
    with open(path, "w", encoding="utf-8") as handle:
        for offset in range(lines):
            handle.write(line(start_seq + offset, disk_index.SPILL_LINE,
                              offset % key_space))
    return start_seq + lines


def digest_of(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --- fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def journal_dir(tmp_path, monkeypatch):
    """A diagnostics directory per test, and a per-process index, as default."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    disk_index.reset_state_for_tests()
    yield directory
    disk_index.reset_state_for_tests()


@pytest.fixture(scope="module")
def deep_journal(tmp_path_factory):
    """TONIGHT'S JOURNAL, written once for the module: 4 ranks, ~16 MB, 440k lines.

    Shared and READ-ONLY: the drain must never write to a journal, and test 5 asserts
    that with a digest. `tmp_path_factory` keeps the ~3 s it costs to parse 440k lines
    from being paid once per test.
    """
    directory = tmp_path_factory.mktemp("deep-journal")
    seq = 1
    total = 0
    for rank in range(DEEP_JOURNALS):
        path = journal_named(directory, f"PP{rank}_TP0")
        seq = write_journal(path, DEEP_LINES_PER_JOURNAL, start_seq=seq,
                            key_space=DEEP_KEY_SPACE)
        total += os.path.getsize(path)
    return {"directory": str(directory), "bytes": total,
            "lines": DEEP_JOURNALS * DEEP_LINES_PER_JOURNAL,
            "digests": {name: digest_of(os.path.join(str(directory), name))
                        for name in os.listdir(str(directory))}}


@pytest.fixture
def deep_dir(deep_journal, monkeypatch):
    """Point the journal directory at tonight's shape."""
    monkeypatch.setenv(config.ENV_DIAG_DIR, deep_journal["directory"])
    return deep_journal


class _DrainClock:
    """A monotonic clock that moves as fast as the box reads journal bytes.

    The bound under test is a WALL-CLOCK deadline, and this host reads the same journal
    three times faster than the NVMe array on the box (measured here: ~10 MB/s of reads
    against the box's ~2.2 MB/s). A test that measured a deadline in this host's time
    would be measuring the machine it ran on - so the clock advances by
    `bytes read / MEASURED_DRAIN_BYTES_PER_S` after every bounded pass, which is the
    operator's own arithmetic (25 MB at 2.2 MB/s is 11 s) applied to the bytes the pass
    actually pulled out of the journals.
    """

    def __init__(self) -> None:
        self.now = 1_000.0
        self.passes = 0
        self.bytes_read = 0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:  # the module also stamps wall-clock `seen_ts`
        return 1_700_000_000.0 + self.now

    def consume(self, read_bytes: int) -> None:
        self.bytes_read += int(read_bytes)
        self.now += int(read_bytes) / MEASURED_DRAIN_BYTES_PER_S
        self.passes += 1

    @property
    def elapsed(self) -> float:
        return self.now - 1_000.0


@pytest.fixture
def box_paced(monkeypatch):
    """Run `disk_index` on the box's clock: every byte read costs 1/2.2 ms."""
    clock = _DrainClock()
    real_pass = disk_index._refresh_pass

    def paced_pass(directory, names, byte_budget, unread_out):
        result = real_pass(directory, names, byte_budget, unread_out)
        clock.consume(result["read"])
        return result

    monkeypatch.setattr(disk_index, "time", clock)
    monkeypatch.setattr(disk_index, "_refresh_pass", paced_pass)
    return clock


# --- 1. the default fixes the observed case ----------------------------------

def test_tonight_s_journal_converges_inside_the_shipped_boot_budget(deep_dir,
                                                                    box_paced):
    """No env var, no traffic: the boot drain converges with `lag_bytes: 0`.

    THIS IS THE TEST THAT FAILS AGAINST THE UNPATCHED TREE, and it fails for the reason
    the box failed: the fixture is deep enough that at the box's measured 2.2 MB/s the
    drain costs more than the 5.0 s the tree shipped with, so the unpatched first
    refresh stops blind and `converged` is False. The assertion below states both
    numbers so the fixture cannot silently drift into a size that would pass either way.
    """
    net_seconds = deep_dir["bytes"] / MEASURED_DRAIN_BYTES_PER_S
    assert net_seconds > RETIRED_BOOT_DRAIN_DEADLINE_SECONDS, (
        f"a {deep_dir['bytes']} byte journal at the box's measured "
        f"{MEASURED_DRAIN_BYTES_PER_S} B/s is {net_seconds:.1f} s of reading: this "
        "fixture must be deeper than the retired 5.0 s boot budget or this test "
        "proves nothing")
    assert net_seconds < disk_index.BOOT_DRAIN_DEADLINE_SECONDS, (
        "and shallower than the deadline the release ships, or this test would be "
        "asserting a boot the release does not promise")

    out = disk_index.refresh(force=True)

    assert out["ok"] is True, out
    assert out["converged"] is True, (
        f"the drain must CONVERGE at boot: {out['lag_bytes']} unread bytes over "
        f"{out['lag_journals']} journal(s) after {box_paced.elapsed:.1f} modelled "
        "seconds")
    assert out["lag_bytes"] == 0
    assert out["unread_journals"] == {}
    record = disk_index.boot_drain_view()
    assert record["ran"] is True and record["converged"] is True
    assert record["stopped_at"] == "converged"
    # The proof that the margin is real and not an artefact of a fast host: the modelled
    # boot cost is MORE than the deadline the tree used to ship.
    assert record["elapsed_seconds"] > RETIRED_BOOT_DRAIN_DEADLINE_SECONDS, record
    assert record["bytes_read"] == out["bytes_read"]
    assert record["lines"] == out["lines"] == deep_dir["lines"]
    assert disk_index.boot_drain_token(record) == "JOURNAL_DRAIN[converged]"
    # Convergence means the INDEX, not just the cursors: every identity the journals
    # name is answerable, and the entry cap never fired on the way.
    counters = disk_index.counters()
    assert counters["entries"] == DEEP_KEY_SPACE
    assert counters["entry_cap_hit"] is False
    assert counters["rank_records_dropped"] == 0, (
        "the fixture must stay inside the index's own caps, or this test would be "
        "measuring a withheld span instead of a drained journal")
    assert counters["bytes_skipped"] == 0 and counters["catch_up_jumps"] == 0
    assert present(0) is True and present(DEEP_KEY_SPACE - 1) is True


# --- 2/3. the bounds bind, and say so ----------------------------------------

def test_the_deadline_stops_a_deeper_journal_and_the_request_path_finishes_it(
        deep_dir, box_paced, monkeypatch):
    """A journal no deadline can finish must STOP, REPORT, and never raise.

    `VLLM_E2_TIER_DISK_BOOT_DRAIN_SECONDS` is the knob; the assertion that fails against
    the unpatched tree is the one that says the boot stopped at 3 modelled seconds and
    published `stopped_at: "deadline"` - unpatched, the variable is unknown to the
    grammar, the 5.0 s constant applies, and this fixture converges anyway, which is
    exactly the "silently ignored knob" failure this repo refuses to ship.
    """
    monkeypatch.setenv(config.ENV_DISK_BOOT_DRAIN_SECONDS, "3")
    limits = disk_index.drain_limits()
    assert limits["boot_deadline_seconds"] == pytest.approx(3.0)
    assert limits["boot_deadline_source"] == "env"

    out = disk_index.refresh(force=True)

    assert out["ok"] is True, "a deadline is a stop, never an exception"
    assert out["converged"] is False, "3 s of reading cannot finish this journal"
    assert out["lag_bytes"] > 0
    record = disk_index.boot_drain_view()
    assert record["ran"] is True
    assert record["stopped_at"] == "deadline"
    assert record["converged"] is False
    assert record["deadline_seconds"] == pytest.approx(3.0)
    assert record["error"] is None
    assert 3.0 <= record["elapsed_seconds"] < 4.5, (
        f"the deadline must STOP the drain, not merely be noticed: "
        f"{record['elapsed_seconds']}s elapsed for a 3s deadline")
    assert disk_index.boot_drain_token(record) == "JOURNAL_DRAIN[deadline]"
    # WHAT REMAINS KEEPS DRAINING ON THE REQUEST PATH, one bounded pass at a time - the
    # behaviour tonight fell back on. The boot gave up; the index did not.
    lag = out["lag_bytes"]
    unread = disk_index.counters()["unread_journals"]
    assert unread, "the payload must name what it has not read"
    for _ in range(2):
        again = disk_index.refresh(force=True)
        assert again["bytes_read"] <= limits["read_bytes_per_refresh"], (
            "a post-boot refresh is still ONE bounded pass: the offer path's work "
            "bound is not what this release changes")
        assert again["lag_bytes"] < lag, "the drain advances every pass"
        lag = again["lag_bytes"]


def test_the_byte_budget_stops_at_its_budget_and_reports_the_budget(
        deep_dir, box_paced, monkeypatch):
    """A byte bound, honoured to the byte: the boot cannot read past what it declared.

    Fails against the unpatched tree the same way the deadline test does - an unknown
    variable, a drain that ignores the bound, and a `stopped_at` field that does not
    exist.
    """
    budget = 1024 * 1024
    monkeypatch.setenv(config.ENV_DISK_BOOT_DRAIN_BYTES, str(budget))
    limits = disk_index.drain_limits()
    assert limits["boot_bytes"] == budget
    assert limits["boot_bytes_source"] == "env"

    out = disk_index.refresh(force=True)

    assert out["ok"] is True
    assert out["converged"] is False
    assert budget // 2 <= out["bytes_read"] <= budget, (
        "the budget is the budget: the boot drains up to it and stops AT it")
    record = disk_index.boot_drain_view()
    assert record["stopped_at"] == "byte_budget"
    assert record["max_bytes"] == budget
    assert disk_index.boot_drain_token(record) == "JOURNAL_DRAIN[budget]"


# --- 4/5. the knobs are grammar, not folklore --------------------------------

def test_an_unparseable_value_leaves_the_default_standing_and_names_its_complaint(
        monkeypatch):
    """A typo in a systemd drop-in costs a knob, never a boot and never silence.

    The shape is `staging_rows_verdict()`'s: the default stands, `source` says
    `env_ignored`, and the complaint names the variable AND the raw value. Against the
    unpatched tree there is no resolver to ask, so this fails with an AttributeError -
    which is the honest failure of a test for a feature that does not exist.
    """
    monkeypatch.setenv(config.ENV_DISK_BOOT_DRAIN_SECONDS, "soon")
    monkeypatch.setenv(config.ENV_DISK_BOOT_DRAIN_BYTES, "64mb")
    monkeypatch.setenv(config.ENV_DISK_INDEX_REFRESH_BYTES, "-1")

    limits = disk_index.drain_limits()

    assert limits["boot_deadline_seconds"] == pytest.approx(
        disk_index.BOOT_DRAIN_DEADLINE_SECONDS)
    assert limits["boot_bytes"] == disk_index.BOOT_DRAIN_MAX_BYTES
    assert limits["read_bytes_per_refresh"] == disk_index.MAX_READ_PER_REFRESH
    for key in ("boot_deadline_source", "boot_bytes_source", "read_bytes_source"):
        assert limits[key] == "env_ignored", limits
    joined = " | ".join(limits["problems"])
    for name in (config.ENV_DISK_BOOT_DRAIN_SECONDS,
                 config.ENV_DISK_BOOT_DRAIN_BYTES,
                 config.ENV_DISK_INDEX_REFRESH_BYTES):
        assert name in joined, f"the complaint must name {name}: {joined}"
    for raw in ("'soon'", "'64mb'", "'-1'"):
        assert raw in joined, f"the raw value {raw} must be quoted back: {joined}"
    # And the drain still runs, under the shipped numbers.
    assert disk_index.refresh(force=True)["ok"] is True


def test_a_byte_knob_below_the_floor_is_clamped_with_a_reason(journal_dir, monkeypatch):
    """`VLLM_E2_TIER_DISK_INDEX_REFRESH_BYTES=10` is well-formed and useless: a window
    below one rank's spill burst drains nothing useful while reading as "configured" to
    whoever reads `stopped_at` next. So the floor is honoured AND the clamp published -
    the `entry_cap()` "CLAMPED WITH A REASON" doctrine, applied because this
    is an operator bound - and the clamp is what a refresh HONOORS, so the drain
    still makes progress instead of reading 10 bytes a pass.

    FAILS against the unpatched tree: no floor, no clamp, no `*_source` field at all."""
    monkeypatch.setenv(config.ENV_DISK_BOOT_DRAIN_BYTES, "10")
    monkeypatch.setenv(config.ENV_DISK_INDEX_REFRESH_BYTES, "10")
    limits = disk_index.drain_limits()
    assert limits["boot_bytes_source"] == limits["read_bytes_source"] == "env_clamped"
    assert limits["boot_bytes"] == disk_index.DRAIN_BYTES_FLOOR
    assert limits["read_bytes_per_refresh"] == disk_index.DRAIN_BYTES_FLOOR
    problems = limits["problems"]
    assert len(problems) == 2, problems
    assert all("clamped" in problem for problem in problems), problems
    assert all("'10'" in problem for problem in problems), problems
    # A deadline gets NO floor above zero: one second of boot reading is legit
    # (if futile) choice, a 10-byte window is a misunderstanding.
    assert limits["boot_deadline_source"] == "default"
    write_journal(journal_named(journal_dir, "PP0_TP0"), 40)
    disk_index.refresh(force=True)
    boot = disk_index.counters()["boot_drain"]
    assert boot["stopped_at"] == "byte_budget", (
        "the clamped floor must be the number the boot drain actually honours")
    assert 0 < boot["bytes_read"] <= disk_index.DRAIN_BYTES_FLOOR
    assert boot["lines"] < 40 and boot["lag_bytes"] > 0
    assert boot["max_bytes"] == disk_index.DRAIN_BYTES_FLOOR
    # ... and the clamped floor still DRAINS: each following refresh reads the floor and
    # moves the cursor, so a tiny-but-legal window makes progress rather than stalling.
    for _ in range(12):
        out = disk_index.refresh(force=True)
    assert out["converged"] is True, out
    assert out["lag_bytes"] == 0, out
    journal = journal_named(journal_dir, "PP0_TP0")
    assert disk_index.counters()["bytes_read"] >= os.path.getsize(journal), (
        "every journal byte was eventually read at the clamped window size - the clamp "
        "slows the drain, it does not stop it")


def test_the_environment_honours_a_well_formed_value_and_says_env(journal_dir,
                                                                   monkeypatch):
    """The other door: a good value lands, is the number in force, and says `env`."""
    monkeypatch.setenv(config.ENV_DISK_BOOT_DRAIN_SECONDS, "12.5")
    monkeypatch.setenv(config.ENV_DISK_BOOT_DRAIN_BYTES, "8388608")
    monkeypatch.setenv(config.ENV_DISK_INDEX_REFRESH_BYTES, "65536")
    limits = disk_index.drain_limits()
    assert limits["boot_deadline_seconds"] == pytest.approx(12.5)
    assert limits["boot_bytes"] == 8 * 1024 * 1024
    assert limits["read_bytes_per_refresh"] == 65536
    assert limits["boot_deadline_source"] == limits["boot_bytes_source"] == \
        limits["read_bytes_source"] == "env"
    assert limits["problems"] == []
    # The steady-state cap is a number the REFRESH honours, not only one the payload
    # prints: with the cap at 64 KiB, one boot pass reads at most that much.
    write_journal(journal_named(journal_dir, "PP0_TP0"), 6_000)
    out = disk_index.refresh(force=True)
    assert out["bytes_read"] > 0
    assert out["bytes_read"] <= 8 * 65536, (
        "the boot drain multiplies the per-pass cap by passes, so the bound this "
        "asserts is the cap TIMES the passes it was allowed, and the per-pass bound "
        "itself is pinned by the non-boot assertion below")
    after = disk_index.refresh(force=True)
    assert after["bytes_read"] <= 65536, "a non-boot pass honours the env cap"


def test_an_unset_environment_is_todays_numbers_and_the_drain_reads_only(
        journal_dir):
    """UNSET IS TODAY: the shipped numbers, the same per-refresh cap, no writes.

    Three claims, all of which the unpatched tree cannot answer because two of them
    live in new payload fields:
      * every knob reports `source: default` and the constant it names;
      * the steady-state read cap is STILL 512 KiB and the line cap STILL 8192 - this
        release widens the BOOT and deliberately does not widen the offer path;
      * the drain is READ-ONLY: the journal files are byte-identical afterwards, so a
        boot that drains cannot be a boot that corrupted its own map.
    """
    limits = disk_index.drain_limits()
    assert limits["boot_deadline_source"] == "default"
    assert limits["boot_bytes_source"] == "default"
    assert limits["read_bytes_source"] == "default"
    assert limits["read_bytes_per_refresh"] == disk_index.MAX_READ_PER_REFRESH
    assert disk_index.MAX_READ_PER_REFRESH == 512 * 1024, (
        "the per-refresh cap this release leaves ALONE, byte for byte")
    assert disk_index.MAX_LINES_PER_REFRESH == 8_192
    assert limits["problems"] == []

    path = journal_named(journal_dir, "PP0_TP0")
    write_journal(path, 30_000)
    before = digest_of(path)

    record = disk_index.drain_journal_at_boot()
    assert record["ran"] is True and record["converged"] is True
    assert digest_of(path) == before, "the drain read the journal and changed nothing"

    # The steady-state pass after the boot is still ONE pass under the SAME cap.
    with open(path, "a", encoding="utf-8") as handle:
        for offset in range(30_000):
            handle.write(line(100_000 + offset, disk_index.SPILL_LINE,
                              DEEP_KEY_SPACE + offset))
    after = disk_index.refresh(force=True)
    assert after["bytes_read"] <= disk_index.MAX_READ_PER_REFRESH
    assert after["converged"] is False, "one bounded pass cannot eat 4 MB"


# --- 6. observability ---------------------------------------------------------

def test_the_record_and_its_limits_reach_the_disk_verb_s_index_section(journal_dir):
    """The fields live BESIDE `converged`/`lag_bytes` - the section already read.

    The path a dashboard reads is `GET /plugins/kvtier/disk` -> `restore.index`, so the
    new block is placed there rather than in a section nobody polls. This fails against
    the unpatched tree as a KeyError on `boot_drain`: the payload has no such word.
    """
    write_journal(journal_named(journal_dir, "PP0_TP0"), 2_000)
    disk_index.drain_journal_at_boot()

    section = disk_index.section()
    assert set(section["boot_drain"]) >= {"ran", "stopped_at", "bytes_read",
                                          "converged", "elapsed_seconds",
                                          "lag_bytes", "deadline_seconds",
                                          "max_bytes", "error"}
    assert isinstance(section["boot_drain"]["bytes_read"], int)
    assert isinstance(section["boot_drain"]["converged"], bool)
    assert isinstance(section["boot_drain"]["elapsed_seconds"], float)
    assert section["boot_drain"]["bytes_read"] > 0
    assert section["drain_limits"]["boot_deadline_seconds"] == pytest.approx(
        disk_index.BOOT_DRAIN_DEADLINE_SECONDS)

    verb = disk_restore.scheduler_section()["index"]
    assert verb["boot_drain"] == section["boot_drain"]
    assert verb["drain_limits"] == section["drain_limits"]
    # BESIDE the fields it explains, not in a new corner of its own.
    assert {"converged", "lag_bytes", "blind"}.issubset(verb)


# --- 7. posture ---------------------------------------------------------------

def test_the_drain_moves_no_arm_and_no_posture(journal_dir):
    """BEHAVIOUR: draining the journal changes no arm, no posture, no switch.

    The drain runs in the same moment as the boot, so it must be provably unable to
    raise a serve posture the boot did not decide. Every switch this package owns is
    read before and after.
    """
    write_journal(journal_named(journal_dir, "PP0_TP0"), 1_500)

    def posture():
        return {
            "disk_posture": disk_restore.disk_posture(),
            "gate": disk_restore.gate(),
            "spill_armed": disk_spill.is_armed(),
            "tier_armed": backport.is_armed(),
            "guard_armed": guard_mode.is_armed(),
            "restore_layer_installed": disk_restore.counters()["sched_fitted"],
        }

    before = posture()
    record = disk_index.drain_journal_at_boot()
    assert record["ran"] is True
    assert posture() == before, "the drain changed a posture"


def test_the_modules_source_names_no_posture_or_arm_writer():
    """STRUCTURE: the same claim as an AST fact about `disk_index.py` and the boot.

    `test_mamba_anchor_patch.py` pins this for the second patch target and
    `test_disk_authority_switch.py` pins the writers of the switch across the package;
    this is the same instrument pointed at the drain. Docstrings and comments are
    stripped first - this module's prose names `disk_authority` to say it never touches
    it, and prose must not be what a structural test reads.
    """
    def executable(module) -> str:
        tree = ast.parse(inspect.getsource(module))
        keep = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                keep.append(ast.get_docstring(node, clean=False))
        source = inspect.getsource(module)
        for doc in filter(None, keep):
            source = source.replace(doc, "")
        return source

    forbidden = {"set_disk_authority", "set_offer_switch", "set_armed",
                 "disk_authority", "offers_enabled", "serving_as"}
    code = executable(disk_index)
    tree = ast.parse(inspect.getsource(disk_index))
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (named & forbidden), sorted(named & forbidden)
    assert not (attrs & forbidden), sorted(attrs & forbidden)
    for word in sorted(forbidden):
        assert word not in code, word
    # The boot's own drain step reaches the index and nothing else: one call, on the
    # module that owns the journal, and it may not name a switch either.
    boot_code = executable(backport)
    boot_tree = ast.parse(inspect.getsource(backport))
    drain = next(fn for fn in ast.walk(boot_tree)
                 if isinstance(fn, ast.FunctionDef)
                 and fn.name == "_boot_journal_drain")
    touched = {node.attr for node in ast.walk(drain)
               if isinstance(node, ast.Attribute)}
    assert "drain_journal_at_boot" in touched, (
        "the boot's journal step must go through the module that owns the journal")
    assert not ({node.attr for node in ast.walk(drain)
                 if isinstance(node, ast.Attribute)} & forbidden)
    assert not ({node.id for node in ast.walk(drain)
                 if isinstance(node, ast.Name)} & forbidden)
    assert "set_disk_authority" not in boot_code


# --- 8. the fail-open contract -----------------------------------------------

def test_a_drain_that_raises_is_counted_logged_and_leaves_todays_behaviour(
        journal_dir, monkeypatch):
    """NEVER RAISE INTO A BOOT - and `BaseException` still travels.

    The contract is `mamba_anchor.install()`'s and the backport installs': a contained
    step that fails leaves the boot exactly where it was, counts the failure, and says
    so once. A drain that raised out of `create_connector` would be a plugin that
    cannot boot an engine - a worse outage than the blind index it exists to fix.
    """
    write_journal(journal_named(journal_dir, "PP0_TP0"), 400)
    errors_before = int(snapshot._state["errors"])

    def explode(*_args, **_kwargs):
        raise RuntimeError("the journal reader caught fire")

    monkeypatch.setattr(disk_index, "refresh", explode)
    record = disk_index.drain_journal_at_boot()

    assert record["ran"] is False, "a failed drain did not drain"
    assert record["error"] == "RuntimeError", record
    assert record["stopped_at"] == "error"
    assert disk_index.boot_drain_token(record) == "JOURNAL_DRAIN[error]"
    assert int(snapshot._state["errors"]) > errors_before, (
        "a contained exception is a counted exception: this is the ledger the "
        "containment tests read")
    # Today's behaviour means the OFFER PATH is untouched: the next refresh is the one
    # the request path always made, and it is not marked as having run at boot.
    assert disk_index.counters()["boot_drained"] is False

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(disk_index, "refresh", interrupt)
    with pytest.raises(KeyboardInterrupt):
        disk_index.drain_journal_at_boot()


# --- 9/10. where the drain happens -------------------------------------------

@pytest.fixture
def boot_state():
    """The boot's own process state, restored: this file runs the REAL boot."""
    saved_bp = dict(backport._state)
    saved_verdicts = dict(backport._state["method_verdicts"])
    saved_switch = dict(backport._state["switch_verdicts"])
    saved_guard = dict(guard_mode._state)
    backport.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    yield
    backport._state.clear()
    backport._state.update(saved_bp)
    backport._state["method_verdicts"] = saved_verdicts
    backport._state["switch_verdicts"] = saved_switch
    guard_mode._state.clear()
    guard_mode._state.update(saved_guard)


def _fake_factory(cls, seen):
    """The fork's factory shape (`scheduler.py:164`'s call), as `test_backport` builds
    it.

    The construction body records the drain's state, which is the whole evidence this
    test needs: the fork's own constructor runs FIRST, so the drain necessarily lands
    between the connector being built and the connector being handed to the scheduler -
    which is inside `Scheduler.__init__` and before the engine's first step.
    """
    class SimpleCPUOffloadConnector:
        def __init__(self, config, role, kv_cache_config):
            self.role = role
            self.kv_cache_config = kv_cache_config
            self.worker_handler = None
            self.scheduler_manager = (TB._production_manager(cls)
                                      if getattr(role, "name", role) == "SCHEDULER"
                                      else None)
            seen["ran_during_construction"] = (
                disk_index.boot_drain_view()["ran"])

    SimpleCPUOffloadConnector.__module__ = backport.CONNECTOR_MODULE
    module = types.ModuleType(backport.CONNECTOR_MODULE)
    module.SimpleCPUOffloadConnector = SimpleCPUOffloadConnector

    class KVConnectorFactory:
        @classmethod
        def create_connector(cls, config, role, kv_cache_config):
            return SimpleCPUOffloadConnector(config, role, kv_cache_config)

    factory_module = types.ModuleType(backport.CONNECTOR_FACTORY_MODULE)
    factory_module.KVConnectorFactory = KVConnectorFactory
    return KVConnectorFactory, SimpleCPUOffloadConnector, module, factory_module


def test_the_drain_lands_at_connector_creation_before_the_engine_holds_one(
        journal_dir, boot_state, monkeypatch):
    """THE WHOLE CLAIM, IN CODE: the connector is handed over with a converged index.

    `create_connector` is what the scheduler calls inside `Scheduler.__init__`
    (`scheduler.py:164`), and this plugin wraps it, so anything the hook does happens
    after the fork built the object and before the caller receives it - which is long
    before the first step and long before any request. The assertion that fails against
    the unpatched tree is the boot's own `journal_drain` record (the key does not exist)
    and the token on the BOOT line (the word is not in the line).
    """
    write_journal(journal_named(journal_dir, "PP0_TP0"), 400)
    cls = TB._six_group_class()
    seen: dict = {}
    factory, connector_cls, connector_module, factory_module = _fake_factory(cls, seen)
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_MODULE, connector_module)
    monkeypatch.setitem(sys.modules, backport.CONNECTOR_FACTORY_MODULE, factory_module)

    assert disk_index.boot_drain_view()["ran"] is False, "nothing has been built yet"
    assert backport.install_connector_hook()["fitted"] is True

    connector = factory.create_connector(config="CFG",
                                         role=types.SimpleNamespace(name="SCHEDULER"),
                                         kv_cache_config="KVC")

    assert isinstance(connector, connector_cls), "the engine got its own object"
    assert connector.scheduler_manager is not None
    assert seen["ran_during_construction"] is False, (
        "the fork's constructor must run BEFORE the drain, or the drain is not "
        "hanging off the connector-creation hook")
    record = disk_index.boot_drain_view()
    assert record["ran"] is True and record["converged"] is True
    assert record["bytes_read"] > 0
    boot = backport._state["boot"]
    assert boot["journal_drain"]["ran"] is True
    assert boot["journal_drain"]["trigger"] == "connector creation"
    line = boot["line"]
    assert "JOURNAL_DRAIN[converged]" in line, line
    # The restart verifier's grep is unaffected: it takes the LAST `tier=` match and
    # a bracketed token has no `=` to be mistaken for one.
    assert re.findall(r"tier=[A-Za-z_]+", line)[-1] == "tier=ARMED", line
    assert line.count("JOURNAL_DRAIN[") == 1
    assert backport.section()["boot"]["journal_drain"]["bytes_read"] > 0


def test_an_inert_boot_drains_nothing(journal_dir, boot_state, monkeypatch):
    """`VLLM_E2_TIER_DEV_BOOT_INERT` means the boot changes nothing - not even this.

    The drain is read-only and moves no arm, but it does cost boot seconds, and inert
    is the operator's explicit "install and stop there". It fails against the unpatched
    tree because the inert boot has no `journal_drain` record to read at all.
    """
    write_journal(journal_named(journal_dir, "PP0_TP0"), 400)
    monkeypatch.setenv(config.ENV_BOOT_INERT, "1")
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)
    boot = backport.boot_from_connector(cls, report, manager)

    assert boot["mode"] == "inert"
    assert boot["journal_drain"]["ran"] is False
    assert boot["journal_drain"]["stopped_at"] == "inert"
    assert "JOURNAL_DRAIN[inert]" in boot["line"], boot["line"]
    assert disk_index.counters()["boot_drained"] is False, (
        "an inert boot read no journal byte at all")
    assert disk_index.counters()["bytes_read"] == 0
