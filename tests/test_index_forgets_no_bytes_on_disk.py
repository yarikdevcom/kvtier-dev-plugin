r"""THE INDEX MUST NOT FORGET BYTES THAT ARE STILL ON DISK.

THE OPERATOR'S RULE THIS FILE EXISTS FOR.

    "remove expiration time on disk cache etc etc, so it stays infinite till lru CACHED"

A block spilled to the disk store persists INDEFINITELY and may be discarded only by
LRU/quota. The store's FILES obey that (the ceiling is the only deleter, `st_mtime` is a
touch clock). The in-memory INDEX that maps a KV identity to a file did not, in a way
that is easy to miss because nothing is destroyed: when the index reaches
`MAX_INDEX_ENTRIES` it evicts `next(iter(entries))` - THE FIRST KEY INSERTED - and the
file it described stays on the disk, whole, readable, and unreachable. `contains()`
answers False, `no_disk_run` refuses the offer, and the only trace is a counter named
`entries_dropped` that cannot tell "the file was unlinked ages ago" (harmless
bookkeeping)
apart from "a whole 11 MB span is sitting on my RAID and I have stopped offering it"
(the
missed restore the operator is paying for).

THREE DEFECTS, THREE FIXES, ALL PINNED HERE.

1. THE EVICTION ORDER IS INSERTION, NOT LRU. An entry that has been restored a thousand
   times sits at the front of the queue forever, because nothing on a touch moves it.
   The
   index therefore evicts its MOST-USEFUL entries first - the exact inverse of an
   LRU, and
   the inverse of what the file side does (`disk_read.promote_recency` bumps
   `st_mtime` on
   a read that produced bytes). The fix is the same idea, in memory: a touch moves the
   entry to the back of the queue. A touch is (a) a spill line that re-names an identity
   the index already holds (it was just WRITTEN) or (b) a restore that produced usable
   bytes - the SAME event `promote_recency` promotes the FILE for, so both clocks mean
   "last touch: a write or a successful read", never "last time I was inserted".

2. THE CAP IS NOT THE OPERATOR'S TO SET. It is a module constant, so an operator who
   measures 12,738 live files on a 148 GiB ceiling and wants headroom must patch the
   plugin. `VLLM_E2_TIER_DISK_INDEX_ENTRIES` is the fix: the grammar lives in `config`
   (a typo means the shipped default, never a broken index) and the policy - the floor,
   the ceiling and the reason a clamp happened - lives here, next to the memory it buys
   (~342 bytes per entry, measured, so the shipped default costs at most ~90 MB AT the
   cap and nothing before it: the dict grows to the identities actually indexed).

3. `entries_dropped` CANNOT EXPLAIN ITSELF. This file's title question - was a dropped
   entry's file still on disk? - is answered by a BOUNDED presence check: the shards
   that
   a cap-drop actually touched are listed at most `PRESENCE_SCAN_MAX_SHARDS_PER_REFRESH`
   times per refresh, and the answer lands in `entries_dropped_file_present` /
   `_file_absent`. When there is no store root to ask, or the scan budget is spent, the
   drop goes to `entries_dropped_unverified` rather than being silently counted either
   way. A counter that guesses is worse than one that says it did not look.

Real journals, real committed files, the real replay path. No engine, no torch, no
threads, no network - and no sleeping: only file contents and mtimes are invented.
"""

from __future__ import annotations

import os

import pytest

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_restore,
)

GROUP = 3
WIDTH = 11_468_800
RANK = "PP0_TP0"


def key_for(index: int, shard: str = "0a") -> str:
    """A key16 whose first byte is `shard`, so a test can choose the shard.

    SIXTEEN hex characters, the way `disk_layout.validate_key16` demands: two for the
    shard and fourteen for the index. Getting this length wrong does not fail loudly -
    the journal's line is refused as `bad_key` and the index comes up empty, which is
    exactly how this helper's own first version presented the bug.
    """
    assert len(shard) == disk_layout.SHARD_CHARS == 2
    return f"{shard}{index:014x}"


def full_key(key16: str, group: int = GROUP) -> str:
    return key16 + "0" * 32 + f"{group:08x}"


def present(key16: str, group: int = GROUP) -> bool:
    return disk_index.contains(full_key(key16, group))


def journal_path(directory, rank: str = RANK) -> str:
    return os.path.join(str(directory), f"{disk_index.JOURNAL_PREFIX}{rank}"
                        f"{disk_index.JOURNAL_SUFFIX}")


def line(seq: int, op: str, key16: str) -> str:
    return (f"{seq}{disk_index.LINE_SEP}{op}{disk_index.LINE_SEP}"
            f"{key16}{disk_index.LINE_SEP}{WIDTH}{disk_index.LINE_SEP}{GROUP}\n")


def append_lines(path, lines) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("".join(lines))


def counters() -> dict:
    return disk_index.counters()


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_index(tmp_path, monkeypatch):
    """A fresh index, AND the shipped cap restored afterwards.

    The second half is not ceremony: the cap is a module attribute these tests lower
    to 2
    or 4 to make the eviction observable, and a leaked `MAX_INDEX_ENTRIES = 3` makes the
    very next test in the file see a three-entry index and fail as if the plugin were
    broken. That is how this file's first run reported "the default cap is 3".
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.delenv(config.ENV_DISK_INDEX_ENTRIES, raising=False)
    shipped = disk_index.MAX_INDEX_ENTRIES
    disk_index.reset_state_for_tests()
    yield
    disk_index.MAX_INDEX_ENTRIES = shipped
    disk_index.reset_state_for_tests()
    disk_restore.reset_state_for_tests()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A journal directory AND a real store root with real committed files."""
    diag = tmp_path / "diag"
    diag.mkdir()
    root = tmp_path / "kv-store"
    root.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANK)
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    disk_restore.reset_state_for_tests()
    return {"diag": diag, "root": root}


def seed(diag, seq_start: int, key16s, *, op: str = disk_index.SPILL_LINE):
    """Append spill (or removal) lines for `key16s` and return the next seq."""
    path = journal_path(diag)
    body = [line(seq_start + offset, op, key16)
            for offset, key16 in enumerate(key16s)]
    append_lines(path, body)
    return seq_start + len(key16s)


def drain(diag) -> dict:
    return disk_index.refresh(force=True)


def slot_file(root, key16: str, group: int = GROUP, *, digest: str = "a" * 12,
              size: int = 4096) -> str:
    """A committed spill file of the shape a real spill writes, in its shard."""
    made = disk_layout.build_slot_path(str(root), full_key(key16, group), digest)
    assert made["ok"], made["reason"]
    written = disk_layout.write_committed_file(made["path"], b"x" * size,
                                               sync=False)
    assert written["ok"], written["reason"]
    return made["path"]


# ===========================================================================
# 1. THE EVICTION ORDER IS LRU-BY-TOUCH, NOT INSERTION ORDER
# ===========================================================================

def test_a_touched_entry_outlives_a_cold_one_inserted_before_it(store):
    """The defect in one assertion: a RESTORED entry was the first one evicted.

    Cap 3, three identities, a restore for the OLDEST-inserted one, then a fourth
    arrives. An LRU throws out the entry nobody has touched for the longest time - the
    middle one. The shipped code throws out the one that was just restored, because it
    evicts `next(iter(entries))` and a touch never moved anything.

    THE COST OF GETTING THIS BACKWARDS IS NOT A MISS ONCE. Under a full index the
    hot spans are the ones a ceiling pass will not evict and a busy prefix keeps
    reading, so an insertion-ordered index steadily shovels exactly those spans out of
    the map while keeping the cold ones - the index becomes an anti-cache.
    """
    diag = store["diag"]
    monkey = disk_index
    monkey.MAX_INDEX_ENTRIES = 3
    a, b, c = key_for(1), key_for(2), key_for(3)
    seq = seed(diag, 1, [a, b, c])
    drain(diag)
    assert present(a) and present(b) and present(c)

    # A restore produces bytes for `a`: the SAME event that bumps the file's st_mtime.
    assert disk_index.lookup(full_key(a))["present"] is True

    seed(diag, seq, [key_for(4)])
    drain(diag)

    assert present(a), ("an entry that was just touched is the LAST thing this index "
                        "should forget: it evicted the restored span instead")
    assert not present(b), "the least-recently-touched entry is the one that goes"
    assert present(c)
    assert counters()["entries_touched"] >= 1, (
        "the LRU clock has to be observable, or a reader cannot tell a touch from a "
        "no-op")


def test_a_re_written_identity_is_a_touch(store):
    """A second spill line for an identity is a WRITE, which is a touch too.

    The refresh branch of `_insert` updates the size and the stamp in place and returns,
    which leaves the entry wherever it was in the queue. Re-spilling an identity is the
    store saying "I just used this key again", so it must move to the back like any
    other
    touch. This is the touch that works on a rank that never restores from disk (a pure
    writer), so it is not a bonus - without it the LRU only moves for readers.
    """
    diag = store["diag"]
    disk_index.MAX_INDEX_ENTRIES = 3
    a, b, c = key_for(5), key_for(6), key_for(7)
    seq = seed(diag, 1, [a, b, c])
    drain(diag)

    # `a` is re-spilled (a second write of the same identity), then a 4th arrives.
    seq = seed(diag, seq, [a, key_for(8)])
    drain(diag)

    assert present(a), ("re-spilling an identity must promote it: the file was just "
                        "written, which is the strongest possible statement that it is "
                        "not the index's cheapest memory")
    assert not present(b)
    assert present(c)


def test_a_successful_restore_promotes_the_entry(store):
    """The read-side touch, wired to the one place that KNOWS a read succeeded.

    `disk_read` already owns the "these bytes were usable" decision: it calls
    `disk_read.promote_recency(path)` there and nowhere else. The index hook is called
    from the SAME line, so the file clock and the index clock cannot drift into meaning
    different things - and a FAILED read (`file_missing`, `size_mismatch`) reaches
    neither, because keeping a wrong-sized file alive is worse than evicting it.
    """
    diag, root = store["diag"], store["root"]
    disk_index.MAX_INDEX_ENTRIES = 3
    a, b, c = key_for(9), key_for(10), key_for(11)
    seq = seed(diag, 1, [a, b, c])
    drain(diag)

    # The real hook, given the real file name, exactly as the reader thread holds it.
    path = slot_file(root, a)
    assert disk_index.note_restored_path(path)["promoted"] is True

    seed(diag, seq, [key_for(12)])
    drain(diag)
    assert present(a) and not present(b), \
        "a restore must move the entry in the same queue the file moves in"


def test_the_reader_calls_the_index_touch_on_a_read_that_produced_bytes(store):
    """THE WIRING, not just the function: a real promoted read touches the index.

    A hook nobody calls is a counter nobody can trust, so this asserts the call site
    rather than the callee. It uses the tier's own promotion call -
    `disk_read.promote_recency` - and asserts the index was told about that path.
    """
    diag, root = store["diag"], store["root"]
    key16 = key_for(13)
    seed(diag, 1, [key16])
    drain(diag)
    path = slot_file(root, key16)

    seen: list = []
    real = disk_index.note_restored_path

    def spy(given):
        seen.append(str(given))
        return real(given)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(disk_read.disk_index, "note_restored_path", spy)
    try:
        # THE CONTRACT UNDER TEST: `promote_recency` is the single place the tier admits
        # a file was just used, so it is where the index must be told too.
        disk_read.promote_recency(path)
    finally:
        monkey.undo()

    assert seen == [path], ("a promoted read must tell the index; a touch that lives "
                            "only in disk_index is a clock that never moves in "
                            "production")


# ===========================================================================
# 2. THE CAP IS THE OPERATOR'S TO SET
# ===========================================================================

def test_the_operators_env_var_sets_the_cap(store):
    """`VLLM_E2_TIER_DISK_INDEX_ENTRIES` moves the cap, and says that it did.

    The shipped default must stay honest about its cost, so the payload names the source
    of the number the index is honouring: `entry_cap_source` is `env` only when the
    operator set it and the value was usable.
    """
    os.environ[config.ENV_DISK_INDEX_ENTRIES] = "5000"
    try:
        cap = disk_index.entry_cap()
        assert cap["value"] == 5000, cap
        assert cap["source"] == "env", cap
        assert cap["clamped"] is False, cap
        assert disk_index.section()["entry_cap"] == 5000
        assert disk_index.section()["entry_cap_source"] == "env"
    finally:
        del os.environ[config.ENV_DISK_INDEX_ENTRIES]
    cap = disk_index.entry_cap()
    assert cap["source"] == "default"
    assert cap["value"] == disk_index.MAX_INDEX_ENTRIES


def test_the_default_cap_covers_the_store_it_runs_against(store):
    """The shipped default must not be a number an operator has to raise.

    MEASURED, from this repo's own numbers: the live store holds 12,738 files across
    four
    ranks at a 148 GiB ceiling, and a store can reach ~30,000 files, each identity
    journalled per rank. 65,536 leaves room but a fleet with a bigger ceiling or more
    ranks walks into it, and walking into it means silently un-offering real data. The
    default is therefore one order of magnitude above the observed need, at ~342
    bytes an
    entry (measured: a 20,000-entry index with a real identity string and a real entry
    dict costs 6.8 MB) - under 90 MB AT the cap, and the dict only grows to the
    identities
    actually indexed. An operator who wants less can ask for less with the env var; that
    is the point of the other test.
    """
    assert disk_index.MAX_INDEX_ENTRIES >= 262_144, disk_index.MAX_INDEX_ENTRIES
    assert disk_index.entry_cap()["value"] == disk_index.MAX_INDEX_ENTRIES


def test_a_typoed_cap_is_refused_and_said_not_honoured(store, monkeypatch):
    """Grammar problems go to `problems`, and the shipped default stands.

    The split is this repo's own (`settle_env` / `ceiling_env`): `config` parses and
    never
    interprets, the consumer applies policy. A value below the FLOOR or above the
    CEILING
    is clamped with a reason rather than trusted, because the two failure directions are
    "the index forgets nearly everything" and "the scheduler process eats the box".
    """
    for raw in ("", "not-a-number", "0", "-5"):
        monkeypatch.setenv(config.ENV_DISK_INDEX_ENTRIES, raw)
        cap = disk_index.entry_cap()
        assert cap["value"] == disk_index.MAX_INDEX_ENTRIES, (raw, cap)
        assert cap["source"] == "default", (raw, cap)
        # THE GRAMMAR'S SIDE OF THE SAME ANSWER: a value it refuses is reported where a
        # reader looks, and never silently becomes a number.
        parsed = config.disk_index_env()
        assert parsed["entries"] is None, raw
        if raw:
            assert parsed["problems"], f"{raw} was ignored without saying so"
        else:
            assert parsed["problems"] == []
    monkeypatch.setenv(config.ENV_DISK_INDEX_ENTRIES, "3")
    clamped = disk_index.entry_cap()
    assert clamped["clamped"] is True
    assert clamped["value"] == disk_index.INDEX_ENTRIES_FLOOR
    assert clamped["reason"], "a clamp with no reason is indistinguishable from a no-op"
    # A clamp is GRAMMATICALLY valid (a positive int), so `config` reports no
    # problem and
    # the index owns the disagreement. That is the split, and it is pinned here so a
    # future edit cannot move the policy into the parser and lose the reason.
    assert config.disk_index_env()["entries"] == 3
    assert config.disk_index_env()["problems"] == []
    monkeypatch.setenv(config.ENV_DISK_INDEX_ENTRIES, "999999999")
    high = disk_index.entry_cap()
    assert high["clamped"] is True and high["value"] == disk_index.INDEX_ENTRIES_CEILING
    assert high["reason"]


# ===========================================================================
# 3. A DROP MUST SAY WHETHER THE FILE WAS STILL THERE
# ===========================================================================

def test_a_dropped_entry_whose_file_is_still_on_disk_is_a_missed_restore(store):
    """THE HEADLINE NUMBER. `entries_dropped` alone could not tell these two apart.

    Two cap drops, one file present on the store and one file already gone. The count
    that matters - "real KV is on disk and the index stopped offering it" - must be its
    own number, because the other one is not a loss at all.
    """
    diag, root = store["diag"], store["root"]
    disk_index.MAX_INDEX_ENTRIES = 2
    keep_live, keep_gone, first, second = (key_for(21), key_for(22), key_for(23),
                                           key_for(24))
    slot_file(root, keep_live)      # on disk: dropping THIS entry is the incident
    # `keep_gone` is journalled but its file was reclaimed; nothing to find.
    seed(diag, 1, [keep_live, keep_gone, first, second])
    report = drain(diag)
    assert report["ok"] is True, report

    got = counters()
    assert got["entries_dropped"] >= 1, got
    assert got["entries_dropped_file_present"] >= 1, (
        "at least one dropped entry names a file that is still on the store: this is "
        "the number the operator asked for, and it means capacity is being wasted on "
        "spans the index refuses to offer")
    assert (got["entries_dropped_file_present"]
            + got["entries_dropped_file_absent"]
            + got["entries_dropped_unverified"] == got["entries_dropped"]), (
        "the three must balance the total: a drop that was neither verified nor "
        "explicitly unverified is a drop nobody accounted for")
    assert not present(keep_live) or got["entries_dropped_file_present"] >= 1
    section = disk_index.section()
    assert section["entries_dropped_file_present"] \
        == got["entries_dropped_file_present"]


def test_a_removal_line_is_not_a_drop_and_never_reaches_the_presence_scan(store):
    """A `-` line is BOOKKEEPING, not a loss: the file is gone, the index agrees.

    Folding removals into the drop accounting would make the headline number unreadable
    - a busy reclamation journalled ~15,000 `-` lines in one pass on the live box, and
    counting those as "the index forgot a file" would bury the real losses by four
    orders of magnitude.
    """
    diag, root = store["diag"], store["root"]
    a, b = key_for(31), key_for(32)
    slot_file(root, a)
    seq = seed(diag, 1, [a, b])
    drain(diag)
    assert present(a) and present(b)
    seed(diag, seq, [a], op=disk_index.REMOVE_LINE)
    drain(diag)

    got = counters()
    assert not present(a)
    assert got["entries_dropped"] == 0, \
        "a removal the journal named is not a cap drop and must not be counted as one"
    assert got["entries_dropped_file_present"] == 0
    assert got["removals"] >= 1


def test_an_unverifiable_drop_says_it_did_not_look(store, monkeypatch):
    """No store root, or a spent scan budget, means UNVERIFIED - never a guess.

    The presence check is a bounded directory listing of the shards a drop touched. A
    rank whose store root this process cannot see must not report those drops as
    "absent" (that would claim the files were gone, which is the comfortable wrong
    answer), and a burst of thousands of drops must not turn into thousands of scans on
    the scheduler's refresh path.
    """
    diag = store["diag"]
    disk_index.MAX_INDEX_ENTRIES = 2
    keys = [key_for(40 + offset) for offset in range(6)]
    seed(diag, 1, keys)
    drain(diag)
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    # THE BASELINE, taken after the root went away: the wave below is what must be
    # unverified, and a claim about "all drops" would be a claim about the wave ABOVE it
    # too - which did have a store root and did get checked.
    before = counters()
    seed(diag, 100, [key_for(60 + offset) for offset in range(4)])
    drain(diag)
    got = counters()
    drops = int(got["entries_dropped"]) - int(before["entries_dropped"])
    scans = int(got["presence_scans"]) - int(before["presence_scans"])
    unverified = (int(got["entries_dropped_unverified"])
                  - int(before["entries_dropped_unverified"]))
    present = (int(got["entries_dropped_file_present"])
               - int(before["entries_dropped_file_present"]))
    absent = (int(got["entries_dropped_file_absent"])
              - int(before["entries_dropped_file_absent"]))
    assert drops >= 1, (before, got)
    assert unverified == drops, (
        "with no store root the honest answer is 'did not look': counting these as "
        "absent would tell the operator the disk is clean when nothing checked it")
    assert present == 0 and absent == 0
    assert scans == 0, "no root means no listing, not a listing that answers 'absent'"


def test_the_presence_scan_is_bounded_per_refresh(store):
    """A burst of drops must not become a burst of directory listings.

    The cap can bite thousands of times in one drain (the 0.2.28 incident was 800 drops
    in a refresh), so the scan is capped per refresh and the excess is unverified - the
    same shape as every other bounded probe in this module: a bounded answer plus a
    counter, never an unbounded walk of a directory that may hold 100k files.
    """
    diag, root = store["diag"], store["root"]
    disk_index.MAX_INDEX_ENTRIES = 4
    wave = [key_for(70 + offset, shard=f"{offset % 16:02x}")
            for offset in range(40)]
    for key in wave[:4]:
        slot_file(root, key)
    seed(diag, 1, wave)
    drain(diag)
    got = counters()
    assert got["entries_dropped"] >= 30, got
    assert got["presence_scans"] <= disk_index.PRESENCE_SCAN_MAX_SHARDS_PER_REFRESH, (
        f"one refresh listed {got['presence_scans']} shards: a cap-drop burst must not "
        "turn into an unbounded directory walk on the scheduler's refresh path")
    assert (got["entries_dropped_file_present"]
            + got["entries_dropped_file_absent"]
            + got["entries_dropped_unverified"] == got["entries_dropped"])


# ===========================================================================
# 4. THE INVARIANTS THE FIX MUST NOT BREAK
# ===========================================================================

def test_the_cap_still_bounds_memory_after_a_thousand_touches(store):
    """An LRU must not become an excuse for an unbounded dict.

    Touches only reorder; they never admit. Whatever the mix of inserts, re-spills and
    restores, `entries` stays at or under the effective cap and every admission past the
    cap drops exactly one entry.
    """
    diag = store["diag"]
    disk_index.MAX_INDEX_ENTRIES = 8
    seq = seed(diag, 1, [key_for(100 + offset) for offset in range(8)])
    drain(diag)
    for round_index in range(60):
        seq = seed(diag, seq, [key_for(200 + round_index)])
        drain(diag)
        disk_index.lookup(full_key(key_for(100 + (round_index % 8))))
    got = counters()
    assert got["entries"] <= 8, got
    assert got["entries"] == 8
    assert got["entries_dropped"] == 60, (
        "every admission past the cap drops exactly one entry: a touch that grew the "
        "index would be a memory leak with a nice name")
    assert disk_index.section()["entry_cap_hit"] is True


def test_the_index_still_drops_nothing_the_journal_removed_and_keeps_no_ghosts(store):
    """The LRU works on LIVE entries only: a `-` line takes the entry out entirely.

    Regression guard for the reordering itself: a move-to-end that forgot a removal
    would keep an entry the journal already unlinked - a ghost the offer path would
    offer for a file that is gone, which is the corruption direction, not the waste
    direction.
    """
    diag, root = store["diag"], store["root"]
    disk_index.MAX_INDEX_ENTRIES = 4
    keys = [key_for(120 + offset) for offset in range(3)]
    slot_file(root, keys[0])
    seq = seed(diag, 1, keys)
    drain(diag)
    for key in keys:
        disk_index.lookup(full_key(key))
    seed(diag, seq, [keys[1]], op=disk_index.REMOVE_LINE)
    drain(diag)
    assert present(keys[1]) is False
    assert counters()["entries"] == 2
    seed(diag, seq + 1, [key_for(130), key_for(131), key_for(132)])
    drain(diag)
    assert counters()["entries"] == 4
    assert present(keys[1]) is False, \
        "a removed identity must not be resurrected by the LRU reordering"
