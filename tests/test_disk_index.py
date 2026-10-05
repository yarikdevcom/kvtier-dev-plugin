r"""The spill journal: the writer's records, and the replay that becomes an index.

`disk_index.py` had a reader and no writer for one release - nothing called
`record_spill`, so the scheduler's index was starved and the whole restore path had
nothing to look in. These tests pin BOTH halves, in the order a failure would cost
the operator:

1. A JOURNAL LINE NAMES AN IDENTITY, NOT A KEY16. `<key16>` alone is what two
   prefix-cacheable groups over one token chain share, so an index keyed by it
   would offer one group's file for another group's request. Every record carries
   the group, the replay keys on `<key16>:g<group>`, and a lookup handed a bare
   digest misses with a reason instead of guessing.
2. THE ROUND TRIP IS THE WIRING. `record_spill(path)` on a real committed file ->
   a line on disk -> `refresh()` -> `lookup(full_key)` present, with the same bytes,
   and the path the SPILLING rank would rebuild is the file that exists.
   `record_removal(path)` -> the entry is gone after the next refresh.
3. A REFUSAL IS NOT A CORRUPTION. A pre-group (2-field) file name, a .part name, a
   garbage key, a garbage size: each is refused with a reason, writes no line, and
   raises nothing - because the caller is a spill that must not care.
4. THE SEQUENCE SURVIVES A RESTART. The journal file outlives the process that wrote
   it, so boot 2 must not restart the sequence at 1: the reader's one rule about a
   journal it cannot trust is to stop applying it, which would un-wire the index
   after the first restart. The seed comes from the file's own tail.
5. THE READER STOPS RATHER THAN GUESSES: truncated, malformed, out-of-sequence and
   over-cap all stop that rank's journal and count why, and the index never claims a
   file it did not see written.

Stdlib plus the plugin; `tmp_path` for every file; no engine, no threads.
"""

from __future__ import annotations

import json
import os

import pytest

from kvtier_dev_plugin import config, disk_index, disk_layout

KEY16 = "a3f0b7c1d4e5f607"
GROUP = 3
KEY = KEY16 + "1" * 48 + f"{GROUP:08x}"
KEY_GROUP_7 = KEY16 + "1" * 48 + f"{7:08x}"
RANK = "PP0_TP0"
GEO = "abcdef123456"


@pytest.fixture
def journal_dir(tmp_path, monkeypatch):
    """A diagnostics directory and a rank, which is all a journal needs to exist."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANK)
    return directory


@pytest.fixture(autouse=True)
def fresh_index(journal_dir, monkeypatch):
    """The index is per-process memory: no test inherits another's entries, cursors,
    sequences or `journal_seeded` flag. Teardown repeats the reset so a leak is not
    pushed at the next file's tests either.

    The tmpfs refusal is cleared for the PREFIX LIST ONLY (pytest's tmp_path lives
    under /tmp) because some tests write a real slot file to name; the rule itself
    belongs to `disk_layout`, and the index never consults it - it takes a path.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    disk_index.reset_state_for_tests()
    yield
    disk_index.reset_state_for_tests()


def name_for(key=KEY, geo=GEO):
    made = disk_layout.build_slot_name(key, geo)
    assert made["ok"], made["reason"]
    return made["filename"]


def write_slot(root, key=KEY, geo=GEO, size=11_468_800):
    """A committed slot file at the name this identity dictates."""
    made = disk_layout.build_slot_path(root, key, geo)
    assert made["ok"], made["reason"]
    written = disk_layout.write_committed_file(made["path"], b"\0" * size,
                                               sync=False)
    assert written["ok"], written["reason"]
    return made


def read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


# --- the seam: a journal only ever describes what it was told ---------------

def test_a_committed_spill_is_journalled_and_replays_to_the_same_identity(
        journal_dir):
    """The wiring, end to end: path -> line -> index -> lookup by the engine's key."""
    store = journal_dir / "store"
    made = write_slot(store)
    report = disk_index.record_spill(made["path"], 11_468_800)
    assert report["ok"] is True, report["reason"]
    path = disk_index.journal_path()["path"]
    lines = read_lines(path)
    assert len(lines) == 1
    seq, op, key_text, size_text, group_text = lines[0].split(" ")
    assert (seq, op) == ("1", "+")
    assert key_text == KEY16 and size_text == "11468800"
    assert group_text == str(GROUP)

    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"] is True and refreshed["applied"] is True
    assert refreshed["entries"] == 1

    looked = disk_index.lookup(KEY)
    assert looked["present"] is True
    assert looked["key16"] == KEY16 and looked["group_id"] == GROUP
    assert looked["bytes"] == 11_468_800
    assert looked["rank"] == RANK
    # The index publishes no path (four ranks name one identity four ways), so the
    # reader rebuilds the name IT would have written - and that name is the file.
    rebuilt = disk_layout.build_slot_path(store, KEY, GEO)
    assert os.path.exists(rebuilt["path"])
    assert disk_index.contains(KEY) is True


def test_removal_is_journalled_and_the_entry_disappears(journal_dir):
    store = journal_dir / "store"
    made = write_slot(store)
    disk_index.record_spill(made["path"], 1000)
    disk_index.refresh(force=True)
    assert disk_index.lookup(KEY)["present"] is True
    # The file goes first, then the record: the journal describes reality.
    os.remove(made["path"])
    removed = disk_index.record_removal(made["path"], 1000)
    assert removed["ok"] is True
    lines = read_lines(disk_index.journal_path()["path"])
    assert lines[-1] == f"2 - {KEY16} 1000 {GROUP}"
    assert disk_index.refresh(force=True)["removals"] == 1
    assert disk_index.lookup(KEY)["present"] is False
    assert disk_index.counters()["bytes_indexed"] == 0


def test_two_groups_over_one_digest_get_two_entries_and_never_each_others_bytes(
        journal_dir):
    """THE identity bug, one layer above the file name.

    One `key16`, two groups: two records, two index entries, and a lookup for one
    never answers for the other. With the index keyed on `key16` alone the second
    spill would have OVERWRITTEN the first entry, and the scheduler would have
    offered group 3's file to a request for group 7's bytes.
    """
    store = journal_dir / "store"
    first = write_slot(store, KEY)
    second = write_slot(store, KEY_GROUP_7)
    disk_index.record_spill(first["path"], 1000)
    disk_index.record_spill(second["path"], 2000)
    refreshed = disk_index.refresh(force=True)
    assert refreshed["entries"] == 2
    assert disk_index.lookup(KEY)["bytes"] == 1000
    assert disk_index.lookup(KEY_GROUP_7)["bytes"] == 2000
    assert disk_index.counters()["bytes_indexed"] == 3000
    # Removing one identity leaves the other intact.
    disk_index.record_removal(first["path"], 1000)
    disk_index.refresh(force=True)
    assert disk_index.lookup(KEY)["present"] is False
    assert disk_index.lookup(KEY_GROUP_7)["present"] is True


def test_a_bare_key16_is_never_a_lookup_and_never_an_entry(journal_dir):
    """A key with no group is not "an entry with an unknown group": it is absent.

    Answering `present` to that is the corruption this identity exists to prevent,
    so both doors refuse it - the lookup, and the offer loop's cheaper `contains`.
    """
    store = journal_dir / "store"
    disk_index.record_spill(write_slot(store)["path"], 10)
    disk_index.refresh(force=True)
    miss = disk_index.lookup(KEY16)
    assert miss["present"] is False
    assert miss["ok"] is False
    assert "group" in miss["reason"]
    assert disk_index.contains(KEY16) is False
    assert disk_index.remove(KEY16)["ok"] is False
    # These doors are diagnostic; only `find_on_disk` moves the scheduler's
    # counters, so a bare digest can neither book a hit nor quietly become a miss
    # the operator would read as "we looked, and it was not there".
    assert disk_index.counters()["hits"] == 0
    assert disk_index.counters()["misses"] == 0


def test_a_pre_group_file_name_is_refused_and_writes_no_line(journal_dir):
    """The 827 ambiguous files cannot enter the index, by construction.

    `record_spill` takes a PATH, and `parse_path` names the pre-group shape
    `stale_key_shape` rather than parsing it - so there is no code path by which a
    file with no group in its name becomes an entry. It is refused, counted as a
    journal failure by the caller, and the spill it came from is unaffected.
    """
    legacy = journal_dir / f"kvslot-{KEY16}-{GEO}.bin"
    legacy.write_bytes(b"ambiguous bytes")
    refused = disk_index.record_spill(legacy, 13)
    assert refused["ok"] is False
    assert "not ours to journal" in refused["reason"]
    assert not os.path.exists(disk_index.journal_path()["path"])
    part = journal_dir / (name_for() + disk_layout.SUFFIX_PART)
    part.write_bytes(b"half a file")
    assert disk_index.record_spill(part, 11)["ok"] is False


@pytest.mark.parametrize("junk", [None, object(), 7, b"/tmp/x", "",
                                  "/nowhere/kvslot-nope.bin",
                                  f"kvslot-{KEY16}-{GEO}.bin"])
def test_record_doors_refuse_garbage_and_never_raise(journal_dir, junk):
    for door in (disk_index.record_spill, disk_index.record_removal):
        verdict = door(junk, 10)
        assert isinstance(verdict, dict)
        assert verdict["ok"] is False
        assert json.dumps(verdict, default=str)


def test_journal_append_refuses_garbage_inputs(journal_dir):
    for junk in (None, object(), 7, "", "zz", KEY16, "g3"):
        verdict = disk_index.journal_append("+", junk)
        assert verdict["ok"] is False and json.dumps(verdict, default=str)
    assert disk_index.journal_append("?", KEY)["ok"] is False
    assert disk_index.journal_append("+", KEY, -1)["ok"] is False
    assert disk_index.journal_append(
        "+", KEY, disk_index.MAX_JOURNAL_SIZE_BYTES + 1)["ok"] is False
    assert disk_index.journal_append("+", KEY16, 10, 99_999)["ok"] is False
    assert disk_index.journal_append("+", KEY16, 10, GROUP)["ok"] is True


def test_every_public_door_survives_a_hostile_argument(journal_dir):
    """Nothing in this module may raise: `refresh()` runs on the lookup path."""
    doors = [lambda: disk_index.lookup(None), lambda: disk_index.lookup(object()),
             lambda: disk_index.lookup(b"x" * 40), lambda: disk_index.contains(7),
             lambda: disk_index.remove(None), lambda: disk_index.remove([]),
             disk_index.counters, disk_index.section, disk_index.journal_directory,
             lambda: disk_index.refresh(), lambda: disk_index.refresh(force=True),
             lambda: disk_index.forget_all("a test said so"),
             lambda: disk_index.journal_path("PP0_TP0"),
             lambda: disk_index.journal_path("../evil"),
             lambda: disk_index.journal_path(None),
             lambda: disk_index.record_spill(None, None),
             lambda: disk_index.identity_of(KEY16, GROUP)]
    for door in doors:
        verdict = door()
        assert json.dumps(verdict, default=str) is not None


def test_a_rank_id_can_never_name_a_file_outside_the_directory(journal_dir):
    """The rank id arrives from an environment variable, so it is untrusted text.

    Some shapes are refused outright (a separator, a NUL, an over-long or empty id).
    A name like `..` is NOT refused - it is inside the character class - and it does
    not need to be: the id is EMBEDDED in a longer file name, so the file lands in
    the diagnostics directory like any other. The invariant an operator cares about
    is that the path never leaves the directory, so this proves THAT for every input
    instead of trusting a blocklist.
    """
    for bad in ("..", ".", "PP0/TP0", "PP0\\TP0", "PP0\nTP0", "x" * 40, "",
                "../../etc/passwd", "/etc/passwd", "PP0 TP0"):
        verdict = disk_index.journal_path(bad)
        if verdict["ok"]:
            assert os.path.dirname(verdict["path"]) == str(journal_dir), bad
        else:
            assert verdict["path"] is None, bad
    legal = disk_index.journal_path("PP9_TP9")
    assert legal["ok"] is True
    assert os.path.dirname(legal["path"]) == str(journal_dir)


def test_without_a_diagnostics_directory_there_is_no_channel_and_no_raise(
        monkeypatch):
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    assert disk_index.journal_directory() is None
    # A WELL-FORMED name, so this exercises the missing channel and not the parse.
    verdict = disk_index.record_spill(f"/tmp/{name_for()}", 10)
    assert verdict["ok"] is False
    assert "not set" in verdict["reason"]
    assert not os.path.exists(f"/tmp/{name_for()}")
    # "The spill happened and the index was never told" is exactly what
    # `journal_write_errors` means, so a missing channel counts - it is the most
    # common reason a restore index is empty on a box that enabled the tier but not
    # diagnostics, and a counter that stayed at zero would hide it. The reason names
    # the variable to set, and `disk_spill` mirrors the tally as `journal_failures`.
    assert disk_index.counters()["journal_write_errors"] == 1
    assert disk_index.counters()["journal_writes"] == 0
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"] is True and refreshed["applied"] is False
    assert "not set" in refreshed["reason"]


def test_a_process_with_no_rank_cannot_name_its_own_journal(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path))
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)
    verdict = disk_index.journal_path()
    assert verdict["ok"] is False and verdict["path"] is None
    # The spill still happened; only the index cannot be told about it.
    assert disk_index.record_spill(f"{tmp_path}/{name_for()}", 10)["ok"] is False


# --- the sequence has to survive the process that started it -----------------

def test_a_second_boot_continues_the_journal_sequence_instead_of_restarting_it(
        journal_dir):
    """The cliff that would have silently un-wired this index after one restart.

    The journal file lives in the diagnostics directory and OUTLIVES the worker;
    `_state` does not. Booting again and appending seq 1 to a file that already
    reached seq 3 makes the sequence DECREASE, and the reader's one rule about a
    journal it cannot trust is to stop applying it - permanently. So the first
    append of a process seeds its sequence from the file's own tail.
    """
    store = journal_dir / "store"
    disk_index.record_spill(write_slot(store, KEY)["path"], 10)
    disk_index.record_spill(write_slot(store, KEY_GROUP_7)["path"], 20)
    disk_index.refresh(force=True)
    # A new process: the same journal file, no memory of the sequences it holds.
    disk_index.reset_state_for_tests()
    again = disk_index.record_spill(write_slot(store)["path"], 30)
    assert again["ok"] is True
    assert again["seq"] == 3, again
    lines = read_lines(disk_index.journal_path()["path"])
    assert [line.split(" ")[0] for line in lines] == ["1", "2", "3"]
    # And the reader keeps up with it, because nothing went backwards.
    refreshed = disk_index.refresh(force=True)
    assert refreshed["applied"] is True
    assert disk_index.counters()["stopped"] == {}
    assert disk_index.counters()["dropped"]["bad_seq"] == 0


def test_a_journal_whose_tail_has_no_sequence_is_not_appended_to(journal_dir):
    """When the seed cannot be PROVEN, the writer stops rather than guess.

    A guessed sequence could decrease, and a decrease stops this rank's journal for
    the rest of the boot. A missing index entry costs a restore; that is the safe
    direction, and the failure is counted and reported to the same payload.
    """
    path = disk_index.journal_path()["path"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("garbage that is not a journal line at all\n")
    verdict = disk_index.record_spill(f"{journal_dir}/{name_for()}", 10)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "journal_error"
    assert disk_index.counters()["journal_write_errors"] == 1


# --- the reader: bounded, and it stops rather than guesses ------------------

def _write_journal(directory, text, rank=RANK):
    path = directory / f"{disk_index.JOURNAL_PREFIX}{rank}" \
                       f"{disk_index.JOURNAL_SUFFIX}"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_malformed_line_stops_that_rank_s_journal_and_is_counted(journal_dir):
    """A field that is not the number it claims to be stops THIS rank's journal.

    Applying the records around a corrupt one would build an index from a partly
    understood history. What was already applied stays applied, and the reason names
    the line rather than leaking a bare `int()` message into a payload.
    """
    _write_journal(journal_dir,
                   f"1 + {KEY16} 100 {GROUP}\n"
                   f"2 + {KEY16} broken 3\n")
    disk_index.refresh(force=True)
    counters = disk_index.counters()
    assert counters["dropped"]["bad_size"] == 1
    assert list(counters["stopped"].values()), counters["stopped"]
    assert "non-integer" in str(counters["stopped"])
    # The line BEFORE the malformed one is real and stays: good records are not
    # discarded because a later one is bad.
    assert disk_index.lookup(KEY)["present"] is True
    assert disk_index.lookup(KEY)["bytes"] == 100


def test_a_pre_group_journal_line_stops_the_replay_rather_than_guessing(
        journal_dir):
    """A 4-field line names an identity this index cannot hold, so it is refused."""
    _write_journal(journal_dir,
                   f"1 + {KEY16} 100\n"
                   f"2 + {KEY16} 200 {GROUP}\n")
    disk_index.refresh(force=True)
    assert disk_index.counters()["dropped"]["malformed_line"] == 1
    assert disk_index.lookup(KEY)["present"] is False


def test_a_decreasing_sequence_stops_the_replay(journal_dir):
    _write_journal(journal_dir,
                   f"1 + {KEY16} 100 {GROUP}\n"
                   f"1 + {'0f1e2d3c4b5a6978'} 200 7\n")
    disk_index.refresh(force=True)
    counters = disk_index.counters()
    assert counters["dropped"]["bad_seq"] == 1
    assert counters["spills"] == 1
    # The rank's journal is now stopped: a journal that contradicted itself once is
    # not consulted again, because the alternative is applying records out of order.
    assert "does not increase" in str(counters["stopped"])


def test_a_group_id_outside_the_engine_s_reach_stops_the_replay(journal_dir):
    _write_journal(journal_dir, f"1 + {KEY16} 100 999999\n")
    disk_index.refresh(force=True)
    assert disk_index.counters()["dropped"]["bad_group"] == 1
    assert disk_index.lookup(KEY)["present"] is False


def test_a_truncated_journal_drops_that_rank_s_entries_and_replays_from_the_top(
        journal_dir):
    store = journal_dir / "store"
    disk_index.record_spill(write_slot(store)["path"], 100)
    disk_index.refresh(force=True)
    assert disk_index.lookup(KEY)["present"] is True
    path = disk_index.journal_path()["path"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("")                       # someone truncated it
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"] is True
    assert disk_index.counters()["dropped"]["journal_shrank"] == 1
    assert disk_index.counters()["bytes_indexed"] == 0
    assert disk_index.lookup(KEY)["present"] is False


def test_a_line_that_is_only_partially_written_is_not_applied(journal_dir):
    """The reader stops at the last newline: a line in flight is nobody's fact."""
    _write_journal(journal_dir, f"1 + {KEY16} 100 {GROUP}\n2 + {KEY16} 2")
    disk_index.refresh(force=True)
    assert disk_index.lookup(KEY)["bytes"] == 100
    # The rest of the line arrives "later", and the cursor had not consumed it. What
    # it completes into - `2 + <key16> 2 7` - is a legal record for a DIFFERENT
    # identity (group 7), so group 3 keeps its 100 bytes: an identity another group
    # wrote can never be overwritten by this one.
    with open(disk_index.journal_path()["path"], "a", encoding="utf-8") as handle:
        handle.write(" 7\n")
    disk_index.refresh(force=True)
    assert disk_index.lookup(KEY)["bytes"] == 100
    assert disk_index.lookup(KEY_GROUP_7)["bytes"] == 2
    assert disk_index.counters()["spills"] == 2


def test_the_entry_cap_drops_the_oldest_and_says_so(journal_dir, monkeypatch):
    monkeypatch.setattr(disk_index, "MAX_INDEX_ENTRIES", 3)
    store = journal_dir / "store"
    for index in range(6):
        key = f"{index:016x}" + "0" * 48 + "00000003"
        disk_index.record_spill(write_slot(store, key)["path"], 10)
    disk_index.refresh(force=True)
    counters = disk_index.counters()
    assert counters["entries"] == 3
    assert counters["entries_dropped"] == 3


def test_the_size_cap_stops_the_journal_and_counts_it(journal_dir, monkeypatch):
    monkeypatch.setattr(disk_index, "MAX_JOURNAL_BYTES", 40)
    store = journal_dir / "store"
    first = disk_index.record_spill(write_slot(store, KEY)["path"], 10)
    assert first["ok"] is True
    stopped = None
    for index in range(1, 6):
        key = f"{index:016x}" + "0" * 48 + "00000003"
        stopped = disk_index.record_spill(write_slot(store, key)["path"], 10)
    assert stopped["ok"] is False
    assert stopped["reason_kind"] == "journal_capped"
    counters = disk_index.counters()
    assert counters["journal_full"] is True
    assert counters["dropped"]["journal_capped"] >= 1


def test_a_refresh_is_bounded_and_an_idle_store_costs_almost_nothing(journal_dir):
    _write_journal(journal_dir, f"1 + {KEY16} 100 {GROUP}\n")
    first = disk_index.refresh(force=True)
    assert first["applied"] is True
    second = disk_index.refresh(force=True)
    assert second["applied"] is False              # nothing new to read
    assert second["entries"] == 1
    # The age floor is what keeps a per-request lookup path off the disk.
    third = disk_index.refresh()
    assert third["applied"] is False
    assert "the floor is" in third["reason"]
    assert disk_index.counters()["refresh_skipped"] == 1


def test_the_section_publishes_the_index_shape_and_its_units(journal_dir):
    _write_journal(journal_dir, f"1 + {KEY16} 100 {GROUP}\n")
    disk_index.refresh(force=True)
    section = disk_index.section()
    assert json.dumps(section, default=str)
    assert section["entries"] == 1
    assert section["journal_fields"] == 5
    assert "key16" not in json.dumps(section["note"]) or "g<group>" in \
        section["note"]
    assert "identities" in section["units"]["entries"]
    assert section["journal_directory"] == str(journal_dir)


# --- where the journal lives, and what filesystem that is -------------------
# The journal is the ONLY map from a KV identity to a spilled file, and it cannot
# be rebuilt. So its filesystem is a first-class fact: `VLLM_E2_TIER_DISK_JOURNAL_DIR`
# puts it on storage that survives a reboot, and `journal_fs_state()` says out loud
# when it is still sitting in `/tmp`.

REAL_FORBIDDEN_ROOTS = disk_layout.FORBIDDEN_ROOT_PREFIXES

MOUNTS_TMPFS = ("/dev/nvme0n1p1 / ext4 rw 0 0\n"
                "tmpfs /tmp tmpfs rw,nosuid,nodev 0 0\n")
MOUNTS_DISK = ("/dev/nvme0n1p1 / ext4 rw 0 0\n"
               "/dev/md0 /home/user/kv-tier ext4 rw 0 0\n")


def fake_mounts(tmp_path, monkeypatch, text=None):
    """A stand-in `/proc/mounts`, so the tmpfs answer is deterministic here."""
    path = tmp_path / "mounts"
    if text is None:
        path.unlink(missing_ok=True)
        monkeypatch.setattr(disk_layout, "PROC_MOUNTS_PATH", str(path))
        return path
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(disk_layout, "PROC_MOUNTS_PATH", str(path))
    return path


def test_the_journal_directory_has_its_own_variable_and_a_diagnostics_fallback(
        journal_dir, tmp_path, monkeypatch):
    """Unset = exactly the pre-0.2.19 behaviour; set = the journal moves, alone."""
    monkeypatch.delenv(config.ENV_DISK_JOURNAL_DIR, raising=False)
    assert disk_index.journal_directory() == str(journal_dir)
    for inert in ("", "   "):
        monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, inert)
        assert disk_index.journal_directory() == str(journal_dir)
    elsewhere = tmp_path / "persistent"
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(elsewhere))
    assert disk_index.journal_directory() == str(elsewhere)
    # And the diagnostics directory itself did NOT move: the rank files, the arm
    # file and the dump receipts still use it. Only the journal follows the new
    # variable, because only the journal is rebuild-proof-by-necessity.
    assert config.diagnostics_dir() == journal_dir
    assert disk_index.journal_path()["path"].startswith(str(elsewhere))


def test_every_journal_path_is_composed_at_the_one_read_point(journal_dir, tmp_path,
                                                               monkeypatch):
    """Append, tail-seed and the refresh glob all resolve through one function.

    A second composition site is how a journal ends up written where the reader
    will never look - which is indistinguishable from "nothing was spilled".
    """
    elsewhere = tmp_path / "persistent"
    diag = journal_dir
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANK)
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(elsewhere))
    # THE GREP, made executable: the journal file NAME is composed in exactly one
    # place, and `config.disk_journal_dir()` has exactly one caller in the whole
    # plugin. A second site is how a journal ends up written where no reader looks.
    plugin_dir = os.path.dirname(os.path.abspath(disk_index.__file__))
    with open(disk_index.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert source.count('{JOURNAL_PREFIX}{text}{JOURNAL_SUFFIX}') == 1
    callers = []
    for name in sorted(os.listdir(plugin_dir)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(plugin_dir, name), "r",
                  encoding="utf-8") as handle:
            text = handle.read()
        if "disk_journal_dir" in text:
            callers.append(name)
    assert callers == ["config.py", "disk_index.py"], callers

    store = tmp_path / "store"
    made = write_slot(store)
    assert disk_index.record_spill(made["path"], 11_468_800)["ok"] is True
    path = disk_index.journal_path()["path"]
    assert os.path.dirname(path) == str(elsewhere)
    assert os.listdir(elsewhere) == [os.path.basename(path)]
    assert list(diag.iterdir()) == [], "the diagnostics dir stays untouched"
    # The refresh globs THE JOURNAL DIRECTORY, so it finds what the append wrote.
    refreshed = disk_index.refresh(force=True)
    assert refreshed["applied"] is True, refreshed
    assert disk_index.lookup(KEY)["present"] is True
    # The tail-seed reads the SAME file: a second boot continues the sequence.
    disk_index.reset_state_for_tests()
    assert disk_index.journal_append("+", "b" + KEY16[1:], 10, GROUP)["seq"] > 1


def test_a_journal_on_tmpfs_is_warned_about_once_and_counted(tmp_path, monkeypatch,
                                                             caplog):
    """The 2026-10-05 failure made visible: a map on a filesystem that dies at boot.

    Loud, once, and counted - a dashboard polls these views every few seconds, so a
    per-poll warning would drown the log it is trying to write.
    """
    fake_mounts(tmp_path, monkeypatch, MOUNTS_TMPFS)
    journal = tmp_path / "journal-on-tmpfs"
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(journal))
    first = disk_index.journal_fs_state()
    assert first["on_tmpfs"] is True, first
    assert first["label"] == "tmpfs"
    assert first["fstype"] == "tmpfs" and first["mount_point"] == "/tmp"
    assert first["method"] == "mounts"
    assert first["warned"] is True
    warnings = [rec.getMessage() for rec in caplog.records
                if "spill JOURNAL directory" in rec.getMessage()]
    assert len(warnings) == 1, warnings
    for _ in range(5):
        again = disk_index.journal_fs_state()
        assert again["warned"] is False, "once per episode, not once per poll"
    assert disk_index.counters()["journal_tmpfs_warnings"] == 1
    assert len([rec for rec in caplog.records
                if "spill JOURNAL directory" in rec.getMessage()]) == 1
    # And it reaches the surfaces an operator actually reads.
    section = disk_index.section()
    assert section["journal_on_tmpfs"] is True
    assert section["journal_fs"]["label"] == "tmpfs"
    assert section["journal_directory"] == str(journal)
    assert disk_index.journal_fs_label() == "tmpfs"


def test_a_journal_on_a_real_filesystem_says_disk_and_stays_quiet(tmp_path,
                                                                  monkeypatch,
                                                                  caplog):
    fake_mounts(tmp_path, monkeypatch, MOUNTS_DISK)
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, "/home/user/kv-tier/diag")
    verdict = disk_index.journal_fs_state()
    assert verdict["on_tmpfs"] is False and verdict["label"] == "disk"
    assert verdict["fstype"] == "ext4"
    assert verdict["warned"] is False
    assert disk_index.counters()["journal_tmpfs_warnings"] == 0
    assert not [rec for rec in caplog.records if "JOURNAL" in rec.getMessage()]


def test_a_more_specific_mount_beats_the_forbidden_name(tmp_path, monkeypatch):
    """A bind-mounted `/tmp` on platter must NOT be warned about.

    A warning that cries wolf is a warning that gets ignored, and this one has to
    be believed: it is the only thing standing between a reboot and an empty index.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES",
                        REAL_FORBIDDEN_ROOTS)
    fake_mounts(tmp_path, monkeypatch,
                "tmpfs /tmp tmpfs rw 0 0\n/dev/md0 /tmp/kv-tier ext4 rw 0 0\n")
    verdict = disk_layout.tmpfs_verdict("/tmp/kv-tier/diag")
    assert verdict["ok"] is True
    assert verdict["on_tmpfs"] is False and verdict["method"] == "mounts"
    assert verdict["mount_point"] == "/tmp/kv-tier"


def test_the_store_root_name_rule_still_answers_without_a_mount_table(tmp_path,
                                                                      monkeypatch):
    """No `/proc/mounts` (a container, a weird host): fall back to the NAME list."""
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES",
                        REAL_FORBIDDEN_ROOTS)
    fake_mounts(tmp_path, monkeypatch)              # the file does not exist
    for path in ("/tmp/kv-tier", "/dev/shm/diag", "/var/run/x"):
        verdict = disk_layout.tmpfs_verdict(path)
        assert verdict["on_tmpfs"] is True, verdict
        assert verdict["method"] == "name", verdict


def test_an_unreadable_mount_table_is_an_unknown_and_never_a_guess(tmp_path,
                                                                   monkeypatch):
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    fake_mounts(tmp_path, monkeypatch)
    verdict = disk_layout.tmpfs_verdict(tmp_path / "somewhere")
    assert verdict["ok"] is True
    assert verdict["on_tmpfs"] is None and verdict["method"] == "unknown"
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(tmp_path / "somewhere"))
    assert disk_index.journal_fs_label() == "unknown"
    assert disk_index.journal_fs_state()["warned"] is False


def test_the_tmpfs_verdict_refuses_junk_and_never_raises(tmp_path, monkeypatch):
    fake_mounts(tmp_path, monkeypatch, MOUNTS_TMPFS)
    for junk in (None, 7, b"bytes", "", "nul\x00"):
        verdict = disk_layout.tmpfs_verdict(junk)
        assert verdict["ok"] is False
        assert verdict["on_tmpfs"] is None
