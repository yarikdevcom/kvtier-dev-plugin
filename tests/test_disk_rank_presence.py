r"""PRESENCE IS PER RANK. A peer's journal may neither withdraw nor resurrect MY file.

WHY THIS FILE EXISTS, AND WHAT IT COST. A parked span is not one file, it is ONE FILE
PER RANK (`kvslot-<key16>-g<group>-<geodigest12>.bin`, and every rank writes the name
with its OWN geometry digest), and a restore reads all of them. So "is this span on
disk?" is a per-rank SET, never a bool. Until 0.2.46 the index kept that set for
display (`rank_lines`, reported by `op=contains` as `ranks`) and answered the offer path
from a single GLOBAL table (`entries`), fed by all four journals. Both directions of the
lie were live on the box:

  * FALSE PRESENT. My rank's ceiling pass deletes its slice and appends `- <key16>`;
    a peer that still holds its slice refreshes the identity on its next `+`, and the
    GLOBAL entry reappears. The offer path asks `contains()`, sees "present", offers the
    span - and my rank is handed a read for a file it unlinked. That is the box's
    `failed={'file_missing': 5}` and `hole_in_middle`: a promise about bytes that
    rank no longer had, made by another rank's bookkeeping.
  * FALSE ABSENT. The mirror: one rank's `-` line popped the global entry and dropped
    EVERY rank's claim, so the three ranks that still held readable files were reported
    as holding nothing. Measured on the box as ~442 identities per rank - about 19.5 GB
    of parked KV that no rank was allowed to read.

THE RULE THIS SUITE PINS, in the order the failure misleads:

1. A PEER'S `-` CANNOT WITHDRAW MY CLAIM. The removal line's rank comes from the journal
   FILE the line was read out of, and it retires THAT rank's claim and nothing else. The
   entry survives while any rank still backs it (it is the bytes/cap/LRU view), and the
   offer answer says which rank is missing rather than pretending none of them has
   anything (`ranks_missing`).
2. A PEER'S `+` CANNOT RESURRECT A FILE MY RANK DELETED. An insert records the
   inserting rank's claim; it cannot manufacture a claim for a rank whose `-` line this
   index has already applied. This is the false-present fix, and the test asserts it
   through the offer predicate itself.
3. A RUN TRUNCATES AT THE FIRST BLOCK ANY RANK CANNOT SUPPLY - it does not vanish. The
   positional walk already stopped at the first miss; what was missing was a per-rank
   answer for the miss, so a hole at rank 3's fourth block hid behind three peers'
   claims and the whole run got promised. Eight blocks with rank 3 missing the fourth
   now offers THREE, and the refusal names the rank.
4. THE WITHHELD COUNT GOES TO ZERO against ground truth: replay each journal on its own
   and the set of (identity, rank) pairs the index refuses to attribute is empty - at
   HEAD it is every pair of every rank that was not the last one to speak.
5. THE CONDITION IS MEASURABLE, read-only, bounded: `op=identities` enumerates the
   index's per-rank claims (digests only, capped, paged) to diff against `op=ls`, and
   the audit counters name the divergence live.

Nothing here knows a model or a box. Stdlib plus the plugin; `tmp_path` for every file;
no engine, no threads, no store walk except where a test WRITES files to diff against.
"""

from __future__ import annotations

import os

import pytest
import test_disk_restore as base

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_restore,
    disk_spill,
    disk_store,
)

# The heavy fixtures live in the sibling module; importing the names is how this suite
# reuses them (pytest collects fixtures from imported names).
armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

BLOCK = base.BLOCK
Req = base.Req
key_hex = base.key_hex
Manager = base.Manager

# The box's own slot width, so a journal's size field means what it means in production.
WIDTH = 11_468_800
RANKS = ("PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1")
# One geometry digest per rank, the way the box has them: the SAME key16 named four
# different ways. A restore needs all four names, which is the whole premise of D1.
DIGESTS = {rank: f"{index + 1:012x}" for index, rank in enumerate(RANKS)}
GROUP = 5


# --- journals, written the way FOUR ranks write them -------------------------

def journal_path(rank) -> str:
    """One rank's journal file name: the rank is IN THE NAME, and that is the whole
    reason a format change is not needed to make presence per-rank."""
    directory = disk_index.journal_directory()
    assert directory is not None, "the journal_dir fixture did not run"
    return os.path.join(str(directory),
                        f"{disk_index.JOURNAL_PREFIX}{rank}"
                        f"{disk_index.JOURNAL_SUFFIX}")


def write_journal(rank, lines, start=1):
    """One rank's journal, in the journal's own 5-field line format."""
    with open(journal_path(rank), "w", encoding="utf-8") as handle:
        for offset, (op, key16, size, group) in enumerate(lines):
            handle.write(f"{start + offset} {op} {key16} {size} {group}\n")
    return journal_path(rank)


def append_lines(rank, lines, start):
    """Append to one rank's journal: what that rank does on its next spill/evict."""
    with open(journal_path(rank), "a", encoding="utf-8") as handle:
        for offset, (op, key16, size, group) in enumerate(lines):
            handle.write(f"{start + offset} {op} {key16} {size} {group}\n")
    return start + len(lines)


def seeded(ranks=RANKS, blocks=1, group=GROUP, missing=None, start_seq=1):
    """`blocks` identities claimed by every rank in `ranks`.

    `missing` maps a rank to the block SEEDS that rank does NOT claim - the shape of a
    ceiling pass that took one rank's slice and left the others' standing.
    """
    missing = missing or {}
    for rank in ranks:
        lines = []
        for seed in range(1, blocks + 1):
            if seed in missing.get(rank, ()):
                continue
            lines.append((disk_index.SPILL_LINE, key_hex(seed, group)[:16], WIDTH,
                          group))
        write_journal(rank, lines, start=start_seq)
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    return refreshed


def truth(ranks=RANKS, group=GROUP):
    """GROUND TRUTH: replay every journal ON ITS OWN and answer identity -> ranks.

    This is the answer the index must give, computed by a rule that cannot share a bug
    with it: a rank's own journal, read alone, is exactly the set of files that rank
    holds. Deliberately naive (last line per identity wins), because a ground truth that
    needs the module's own semantics is not a ground truth.
    """
    held: dict = {}
    for rank in ranks:
        path = journal_path(rank)
        if not os.path.exists(path):
            continue
        claim: dict = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) != 5:
                    continue
                _seq, op, key16, _size, group_text = fields
                identity = disk_index.identity_of(key16, int(group_text))
                claim[identity] = (op == disk_index.SPILL_LINE)
        for identity, keeping in claim.items():
            if keeping:
                held.setdefault(identity, set()).add(rank)
    return held


def denied_claims(ranks=RANKS) -> list:
    """(identity, rank) pairs the rank's own journal claims and the index denies it.

    THE 442, AS A PREDICATE. This is the measurement the box's ~442-identities-per-rank
    number was: a rank holds a readable file, and the index will not attribute it to
    that rank. It must be empty, always, and at HEAD it is not.
    """
    out = []
    for identity, holders in truth(ranks).items():
        claims = set(disk_index.rank_claims(identity))
        for rank in sorted(holders - claims):
            out.append((identity, rank))
    return sorted(out)


def contains_for(key, ranks):
    """`contains(key, ranks)`, asked the way the offer path will ask it.

    A wrapper rather than a bare call so that a build whose predicate takes no rank set
    fails with the SENTENCE that explains the defect, not a TypeError: the predicate
    answering for one global table is the whole bug, and the failure should say so.
    """
    try:
        return disk_index.contains(key, ranks)
    except TypeError as exc:
        raise AssertionError(
            "the offer predicate answers for one GLOBAL table: `contains()` takes no "
            f"rank set ({exc}), so another rank's journal decides what my rank is "
            "told it may read") from exc


def presence_of(identity, ranks):
    """The per-rank presence verdict, or an assertion that no such answer exists."""
    try:
        return disk_index.presence(identity, ranks)
    except AttributeError as exc:
        raise AssertionError(
            "the index has no per-rank presence answer at all: there is no "
            f"`presence()` to ask WHICH rank holds nothing ({exc}), so 'absent' cannot "
            "be told apart from 'one rank evicted its slice'") from exc


def counted(counters, key):
    """A counter the fix must publish, named in the failure when it is not there."""
    assert key in counters, (
        f"`{key}` is not published, so an operator cannot tell 'this box never meets "
        "the condition' from 'nobody ever counted it'")
    return counters[key]


def published(section, key):
    """A published field, named in the failure when the section does not carry it."""
    assert key in section, (
        f"`{key}` is not published, so a reader cannot see what an answer was "
        "computed from")
    return section[key]


def complete_identities(ranks=RANKS) -> list:
    """Identities EVERY rank in `ranks` holds, per ground truth."""
    wanted = set(ranks)
    return sorted(identity for identity, holders in truth(ranks).items()
                  if holders >= wanted)


def key_of(identity) -> str:
    """The group-qualified key an identity string names, as a hex block key."""
    key16, group = str(identity).split(disk_index.IDENTITY_SEP)
    return key16 + f"{int(group[1:]):08x}"


@pytest.fixture(autouse=True)
def journal_dir(tmp_path, monkeypatch):
    """A journal directory for tests that need no gate.

    A per-rank presence question needs at least one journal file on disk, and the
    journals get their OWN directory here (`ENV_DISK_JOURNAL_DIR`, which the index
    prefers over the diagnostics directory) so this fixture and the `armed` fixture -
    which makes its own diagnostics directory and writes its own rank file - cannot
    trip over each other's files.
    """
    directory = tmp_path / "journals"
    directory.mkdir(exist_ok=True)
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANKS[0])
    return directory


# --- 1. a peer's `-` cannot withdraw my rank's own live file -----------------

def test_a_peer_removal_leaves_every_other_rank_claim_and_the_entry_alive():
    """THE FALSE-ABSENT FIX, at the two tables the answer is made of.

    Rank PP1_TP1 unlinks ITS slice and says so. The other three ranks hold their files,
    so their claims and the entry that prices them must survive. At HEAD the removal
    handler called `_drop_identity_ranks(identity)`, which popped the WHOLE per-rank
    table for the identity, and popped the global entry with it: three ranks holding
    readable files were reported as holding nothing, which is the ~442.
    """
    seeded(blocks=1)
    identity = disk_index.identity_of(key_hex(1, GROUP)[:16], GROUP)
    assert set(disk_index.rank_claims(identity)) == set(RANKS), "the setup"
    append_lines("PP1_TP1", [(disk_index.REMOVE_LINE,
                              key_hex(1, GROUP)[:16], 0, GROUP)], start=2)
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"] and refreshed["removals"] == 1, refreshed

    claims = set(disk_index.rank_claims(identity))
    assert claims == {"PP0_TP0", "PP0_TP1", "PP1_TP0"}, (
        "a peer's removal retired ranks that never removed anything: "
        f"{sorted(claims)}")
    entry = disk_index.entry_for(key_hex(1, GROUP)[:16], GROUP)
    assert entry["present"] is True, (
        "the entry that prices three ranks' files vanished because a fourth rank "
        f"unlinked its own copy: {entry}")
    assert disk_index.counters()["bytes_indexed"] == WIDTH, (
        "bytes_indexed must be debited once, when the LAST rank claim goes, not on "
        "the first peer removal")


def test_membership_names_the_rank_that_is_missing_instead_of_reporting_nothing():
    """The verb must say WHICH rank the span is waiting for, not just "absent".

    `present` is the OFFER answer (an intersection), so a 3-of-4 span answers `false` -
    and a `false` with no name attached is the answer that costs an afternoon, because
    it reads exactly like "never spilled". `ranks_present` 3 with `ranks_missing`
    ["PP1_TP1"] is the same safety and a different investigation.
    """
    seeded(blocks=1)
    append_lines("PP1_TP1", [(disk_index.REMOVE_LINE,
                              key_hex(1, GROUP)[:16], 0, GROUP)], start=2)
    assert disk_index.refresh(force=True)["ok"]
    answer = disk_index.membership(key_hex(1, GROUP))
    assert answer["ranks_present"] == 3, answer
    assert answer["ranks_missing"] == ["PP1_TP1"], answer
    assert answer["ranks_expected"] == list(RANKS), answer
    assert answer["offerable"] is False, answer
    assert answer["present"] is False, answer
    assert answer["verdict"] == disk_index.VERDICT_PARTIAL_RANKS, answer


def test_the_offer_predicate_is_an_intersection_over_the_ranks_that_must_read():
    """`contains(key, ranks)` answers for THE RANKS ASKED ABOUT and nobody else.

    Three ranks hold the span, one does not. Asked about the three, the answer is
    offerable; asked about all four, it is not, and `ranks_missing` says why. A peer
    outside the set can neither add to the answer nor subtract from it - that is the
    whole of D1 in one predicate.
    """
    seeded(blocks=1)
    append_lines("PP1_TP1", [(disk_index.REMOVE_LINE,
                              key_hex(1, GROUP)[:16], 0, GROUP)], start=2)
    assert disk_index.refresh(force=True)["ok"]
    key = key_hex(1, GROUP)
    three = ("PP0_TP0", "PP0_TP1", "PP1_TP0")
    assert contains_for(key, three) is True, (
        "three ranks hold the files a restore of those three ranks would read, and "
        "the predicate refused them")
    assert contains_for(key, RANKS) is False
    verdict = presence_of(disk_index.identity_of(key[:16], GROUP), RANKS)
    assert verdict["ranks_missing"] == ["PP1_TP1"], verdict
    assert contains_for(key, ("PP0_TP0", "PP1_TP1")) is False, (
        "a rank that removed its slice must never count as present for a set that "
        "includes it")


# --- 2. a peer's `+` cannot resurrect a file my rank deleted -----------------

def test_a_peer_spill_cannot_resurrect_a_file_my_rank_deleted():
    """THE FALSE-PRESENT FIX, asserted through the offer predicate.

    My rank's ceiling pass deletes its slice (`-`). A peer that still holds its slice
    then re-spills the same identity (`+`) - which is ordinary behaviour, and exactly
    what a live rank does every time it writes a block another rank has kept. At HEAD
    that `+` refreshed the GLOBAL entry and `contains()` said "present", so the offer
    path promised a span my rank could not read: a read for a file that is gone, a hole
    in the middle of a claimed run, and a prefix destroyed. The intersection cannot be
    moved by a peer: my rank's absence is a fact the peer's journal does not get a vote
    on.
    """
    seeded(blocks=1)
    identity = disk_index.identity_of(key_hex(1, GROUP)[:16], GROUP)
    append_lines("PP0_TP0", [(disk_index.REMOVE_LINE,
                              key_hex(1, GROUP)[:16], 0, GROUP)], start=2)
    assert disk_index.refresh(force=True)["ok"]
    assert disk_index.contains(key_hex(1, GROUP)) is False, "after my own removal"
    append_lines("PP1_TP1", [(disk_index.SPILL_LINE,
                              key_hex(1, GROUP)[:16], WIDTH, GROUP)], start=2)
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"] and refreshed["spills"] == 1, refreshed

    assert disk_index.contains(key_hex(1, GROUP)) is False, (
        "a peer's spill made a span offerable that my rank deleted: this is the "
        "file_missing/hole_in_middle promise")
    assert set(disk_index.rank_claims(identity)) == {"PP0_TP1", "PP1_TP0", "PP1_TP1"}, (
        "my rank's claim must stay retired after a peer re-spills the same identity")
    verdict = presence_of(identity, RANKS)
    assert verdict["ranks_missing"] == ["PP0_TP0"], verdict
    assert disk_index.contains(key_hex(1, GROUP), ("PP0_TP1", "PP1_TP0",
                                                  "PP1_TP1")) is True


# --- 3. the run truncates at the hole instead of promising it ---------------

@pytest.fixture
def armed4(armed, monkeypatch):
    """The gate open with FOUR ranks proven, the live box's shape.

    `armed` proves one rank because one rank is all its own tests need; a per-rank
    presence question needs the rank set the offer path will read it from, and the
    offer path reads that set from the worker proof it is already required to have.
    """
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": True, "count": 4,
                                             "ranks": list(RANKS), "reason": None,
                                             "ranks_armed": 4, "ranks_seen": 4,
                                             "reads_applied": 0,
                                             "reason_kind": None})
    return armed


def test_a_hole_at_one_rank_truncates_the_run_instead_of_destroying_the_prefix(
        armed4):
    """Eight blocks on four ranks, rank 3 missing the FOURTH: THREE blocks are offered.

    THE SHAPE THE BOX PAID FOR. A positional run is a promise of one file per rank per
    block; a hole at block 4 means blocks 4..8 cannot be claimed, and blocks 1..3 can.
    At HEAD the walk saw the GLOBAL presence, which three peers kept true for block 4,
    so it promised all eight and rank 3 was handed a read for a file it did not have -
    a hole in the middle of a claimed run, which is not a partial restore but a wrong
    answer. Truncation is the ONLY safe shortening here: the prefix is still a promise
    every rank can keep.
    """
    seeded(blocks=8, missing={"PP1_TP1": {4}})
    request = Req(blocks=8)
    run = disk_restore._disk_run(request, 0, [GROUP], BLOCK, base.HASH_BLOCK,
                                 {"pack": base.pack, "resolve": base.resolve},
                                 contracts={str(GROUP): "positional"})
    assert run["ok"], run
    assert run["blocks"] == 3, (
        f"a run across a hole at rank PP1_TP1's 4th block was offered as "
        f"{run['blocks']} blocks; the answer must stop at the hole")
    assert int(run["positional_run"]) == 3, run


def test_the_refusal_names_the_rank_the_store_is_waiting_for(armed4):
    """A run that cannot start says WHICH rank has nothing, in the refusal sentence.

    `no_disk_run` alone reads as "the store is empty", which sends an operator to the
    capacity ceiling. The block that stopped the walk and the rank missing from it are
    the two facts that end the investigation, so they travel in the run's own data.
    """
    seeded(blocks=8, missing={"PP1_TP1": {1}})
    request = Req(blocks=8)
    run = disk_restore._disk_run(request, 0, [GROUP], BLOCK, base.HASH_BLOCK,
                                 {"pack": base.pack, "resolve": base.resolve},
                                 contracts={str(GROUP): "positional"})
    assert run["ok"], run
    assert run["blocks"] == 0, run
    assert "PP1_TP1" in str(run.get("stop_detail") or ""), run


# --- 4. the withheld count goes to 0 -----------------------------------------

def test_no_rank_holds_a_file_the_index_refuses_to_attribute_to_it():
    """A 442-style divergence, reproduced and MEASURED against ground truth.

    Six identities, four ranks, then two ranks retire some of their own slices - the
    ceiling passes that made the box's numbers diverge rank by rank. Ground truth is
    every journal replayed ALONE; the index must attribute every ground-truth claim to
    exactly its rank, so `denied_claims()` is empty. At HEAD a single `-` line anywhere
    retired every rank's claim for that identity, so this is not empty by a mile: it is
    the ~442 identities per rank, in miniature.
    """
    seeded(blocks=6)
    dropped = {"PP0_TP1": (2, 5), "PP1_TP0": (3, 4, 6)}
    for rank, seeds in dropped.items():
        lines = [(disk_index.REMOVE_LINE, key_hex(seed, GROUP)[:16], 0, GROUP)
                 for seed in seeds]
        append_lines(rank, lines, start=7)
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"] and refreshed["removals"] == 5, refreshed

    claims = truth()
    assert len(claims) == 6, claims
    denied = denied_claims()
    assert denied == [], (
        f"{len(denied)} rank claim(s) the index refuses to attribute - the withheld-"
        f"identity condition, e.g. {denied[:3]}")


def test_an_identity_every_rank_still_holds_is_offerable():
    """The other half of the same measurement: what is complete IS offered.

    Withheld is only worth counting if completeness pays: every identity ground truth
    says all four ranks hold must answer the offer predicate True.
    """
    seeded(blocks=6)
    for rank, seeds in {"PP0_TP1": (2, 5), "PP1_TP0": (3, 4, 6)}.items():
        append_lines(rank, [(disk_index.REMOVE_LINE, key_hex(seed, GROUP)[:16], 0,
                             GROUP) for seed in seeds], start=7)
    assert disk_index.refresh(force=True)["ok"]
    complete = complete_identities()
    assert complete, "the fixture must leave at least one complete identity"
    withheld = [identity for identity in complete
                if not contains_for(key_of(identity), RANKS)]
    assert withheld == [], f"{len(withheld)} complete span(s) withheld: " \
                           f"{withheld[:3]}"


# --- 5. the condition is measurable, read-only and bounded ------------------

def write_store_for(root, claims):
    """Write the store AS IT IS: one file per (identity, rank) ground truth claims.

    The fixture's honesty depends on this. Each file is named with the CLAIMING rank's
    own
    geometry digest, which is what the box does, so `op=ls` and the index become two
    independent descriptions of the same bytes - and a disagreement between them is the
    measurement, in either direction (a file the index denies its rank: withheld; a rank
    the index believes with no file: the false present).
    """
    made = []
    for identity, holders in sorted(claims.items()):
        key = key_of(identity)
        for rank in sorted(holders):
            built = disk_layout.build_slot_path(root, key, DIGESTS[rank])
            assert built["ok"], built["reason"]
            os.makedirs(os.path.dirname(built["path"]), exist_ok=True)
            with open(built["path"], "wb") as handle:
                handle.write(b"\0" * 16)
            made.append(built["filename"])
    return made


def test_the_identities_verb_enumerates_per_rank_claims_and_agrees_with_the_store(
        armed, monkeypatch):
    """`op=identities` diffed against `op=ls` says 0 withheld, where HEAD says 6.

    THE CENSUS AN OPERATOR CAN ACTUALLY RUN. The verb names what the index believes,
    per rank, digests only and capped; `op=ls` names the files. Pairing them by
    (key16, group, digest) answers the question the whole box investigation was built
    around, in BOTH directions: a file whose rank the index denies (withheld capacity -
    a rank holding a readable file nobody will offer) and a rank the index believes
    whose
    file is gone (the false present that reaches a worker as `file_missing`). The store
    is
    written from ground truth, one file per claim, so the two views describe the same
    bytes twice. At HEAD the verb does not exist at all, and the condition it measures
    is
    the ~442.
    """
    root = armed
    seeded(blocks=6)
    for rank, drop in {"PP0_TP1": (2,), "PP1_TP0": (3,)}.items():
        append_lines(rank, [(disk_index.REMOVE_LINE, key_hex(seed, GROUP)[:16], 0,
                             GROUP) for seed in drop], start=7)
    assert disk_index.refresh(force=True)["ok"]
    claims = truth()
    assert len(claims) == 6, claims
    written = write_store_for(root, claims)
    assert len(written) == sum(len(holders) for holders in claims.values())

    listed = disk_store.disk_for_engine(None, {"op": "ls", "limit": 200})
    assert listed["ok"], listed
    digest_to_rank = {DIGESTS[rank]: rank for rank in RANKS}
    on_disk: dict = {}
    for row in listed["files"]:
        parts = row["name"].split("-")
        if len(parts) != 4:
            continue
        key16, group_field, digest = parts[1], parts[2], parts[3][:12]
        rank = digest_to_rank.get(digest)
        if rank is None:
            continue
        on_disk.setdefault(f"{key16}{disk_index.IDENTITY_SEP}{group_field}",
                           set()).add(rank)
    assert on_disk, "the fixture wrote no files this listing can see"

    page = disk_store.disk_for_engine(None, {"op": "identities", "limit": 200})
    assert page["ok"], page
    assert page["read_only"] is True, page
    believed: dict = {}
    for row in page["identities"]:
        believed[row["identity"]] = set(row["ranks"])
    withheld = sorted((identity, sorted(ranks - believed.get(identity, set())))
                      for identity, ranks in on_disk.items()
                      if ranks - believed.get(identity, set()))
    assert withheld == [], (
        f"{sum(len(ranks) for _i, ranks in withheld)} file(s) on disk whose rank the "
        f"index denies: {withheld[:3]}")
    phantom = sorted((identity, sorted(ranks - on_disk.get(identity, set())))
                     for identity, ranks in believed.items()
                     if ranks - on_disk.get(identity, set()))
    assert phantom == [], (
        f"rank claim(s) with no file behind them - the false present, {phantom[:3]}")


def test_the_identities_verb_is_bounded_paged_and_walks_no_store(armed, monkeypatch):
    """The enumeration is capped, paged by a cursor, and reads no directory.

    The index is per-process memory and this endpoint is polled; a verb that walked the
    store to answer would turn a dashboard into an I/O storm over a 2,720-file shard
    set. So: `limit` is honoured, `truncated`/`next` let an operator page to the end,
    the rows carry digests and never paths, and the store walk helper is made to explode
    to prove the verb never reaches for it.
    """
    seeded(blocks=4)
    monkeypatch.setattr(disk_spill, "_walk_store",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("op=identities must not walk the store")))
    first = disk_store.disk_for_engine(None, {"op": "identities", "limit": 2})
    assert first["ok"], first
    assert first["count"] <= 2, first
    assert first["limit"] == 2, first
    assert first["truncated"] is True, first
    assert first["next"], first
    rows = first["identities"]
    assert all("identity" in row and "ranks" in row and "path" not in row
               for row in rows), rows
    second = disk_store.disk_for_engine(None, {"op": "identities", "limit": 2,
                                               "after": first["next"]})
    assert second["ok"], second
    assert {row["identity"] for row in first["identities"]}.isdisjoint(
        {row["identity"] for row in second["identities"]}), "the cursor must advance"
    assert "identities" in disk_store.DISK_OPS
    assert "identities" not in disk_store.MUTATING_DISK_OPS


def test_the_index_publishes_the_rank_set_it_intersects_over_and_a_sampled_audit():
    """The rank set, its source, and the sampled audit, all published.

    An intersection answer is only readable if the reader can see WHAT it intersected:
    `ranks_expected` and `ranks_expected_source` travel with the count. The audit is the
    live measurement of the condition, bounded by a sample size and a refresh cadence
    and NAMED as sampled, because an operator watching a box needs a rate they can
    trust rather than a scan that stops the offer path.
    """
    seeded(blocks=3)
    append_lines("PP1_TP1", [(disk_index.REMOVE_LINE,
                              key_hex(2, GROUP)[:16], 0, GROUP)], start=4)
    assert disk_index.refresh(force=True)["ok"]
    try:
        audit = disk_index.audit_now()
    except AttributeError as exc:
        raise AssertionError(
            "the condition has no live measurement: no sampled audit exists, so "
            f"'is the rank divergence being withheld correctly?' has no answer "
            f"without an operator scanning the store ({exc})") from exc
    assert audit["sampled"] is True, audit
    assert audit["sampled_identities"] > 0, audit
    assert audit["sampled_partial_ranks"] >= 1, audit
    section = disk_index.section()
    assert published(section, "ranks_expected") == list(RANKS), section
    assert published(section, "ranks_expected_source") == "journal files", section
    assert published(section, "sampled_runs") >= 1, section


def test_a_peer_removal_is_counted_as_the_event_it_used_to_be():
    """`claims_kept_past_peer_removal`: the fix counts its own interventions.

    Every `-` line that arrives while ANOTHER rank still claims the identity is one
    false-absent that used to happen. Counting it at apply time costs nothing (no scan,
    no stat), and it answers "did this fix change anything on my box?" without an
    operator having to diff two verbs.
    """
    seeded(blocks=2)
    append_lines("PP1_TP1", [(disk_index.REMOVE_LINE,
                              key_hex(1, GROUP)[:16], 0, GROUP),
                             (disk_index.REMOVE_LINE,
                              key_hex(2, GROUP)[:16], 0, GROUP)], start=3)
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    counters = disk_index.counters()
    assert counters["dropped"]["bad_seq"] == 0, ("the fixture wrote a journal whose "
                                                 "sequence did not increase: "
                                                 f"{counters['stopped']}")
    # THE UNIT IS THE (identity, rank) PAIR, because that is the unit the box measured:
    # two removal lines, each of which used to withdraw the files of three OTHER ranks.
    assert counted(counters, "claims_kept_past_peer_removal") == 6, counters
    assert counted(counters, "claims_removed") == 2, counters


def test_the_last_removal_drops_the_entry_once_and_the_bytes_balance():
    """When the LAST rank claim goes, the entry goes with it - and the bytes balance.

    The entry table is the bytes/cap/LRU view: it must not outlive every claim, or a
    dropped entry would keep `bytes_indexed` and the presence scan lying about files
    nobody holds. Removing all four ranks' slices leaves nothing: no entry, no rank
    rows, `bytes_indexed` 0 (the debit happens once, against the entry's own size), and
    an offer answer of False.
    """
    seeded(blocks=1)
    identity = disk_index.identity_of(key_hex(1, GROUP)[:16], GROUP)
    for rank in RANKS:
        append_lines(rank, [(disk_index.REMOVE_LINE,
                             key_hex(1, GROUP)[:16], 0, GROUP)], start=2)
    assert disk_index.refresh(force=True)["ok"]
    assert disk_index.rank_claims(identity) == {}
    assert disk_index.entry_for(key_hex(1, GROUP)[:16], GROUP)["present"] is False
    assert disk_index.counters()["bytes_indexed"] == 0
    assert disk_index.contains(key_hex(1, GROUP)) is False


def test_a_removal_from_a_rank_that_holds_no_claim_costs_nothing_and_is_counted():
    """A duplicate or out-of-order `-` must not debit the entry twice.

    A journal replayed from a cursor, or a rank that reports an eviction it never
    journalled as a spill, produces a `-` for a claim this rank does not hold. Under a
    global entry table that removal was the ONLY thing that mattered and it popped the
    entry (double-debiting `bytes_indexed` across a pair of such lines). Per rank, the
    line retires nothing: it is counted and the tables do not move.
    """
    seeded(blocks=1)
    before = disk_index.counters()
    append_lines("PP1_TP1", [(disk_index.REMOVE_LINE,
                              key_hex(1, GROUP)[:16], 0, GROUP)], start=2)
    append_lines("PP1_TP1", [(disk_index.REMOVE_LINE,
                              key_hex(1, GROUP)[:16], 0, GROUP)], start=3)
    assert disk_index.refresh(force=True)["ok"]
    after = disk_index.counters()
    assert after["bytes_indexed"] == before["bytes_indexed"], (
        "a removal that retired no claim moved the byte accounting")
    assert counted(after, "rank_claims_unmatched") == 1, after
    assert set(disk_index.rank_claims(
        disk_index.identity_of(key_hex(1, GROUP)[:16], GROUP))) == {"PP0_TP0",
                                                                   "PP0_TP1",
                                                                   "PP1_TP0"}


def test_the_rank_set_a_rank_the_caller_names_alone_answers_for_one_rank():
    """A caller-named rank set is used AS GIVEN: no union, no silent widening.

    The offer path asks about the ranks the boot has ARMED (that is who will read). A
    journal from a rank that is not armed - a dead rank, or last boot's topology - must
    not veto the living ones, and an armed rank with no journal must veto everything.
    Both halves are the same predicate, so both are pinned here.
    """
    seeded(ranks=RANKS, blocks=1)
    key = key_hex(1, GROUP)
    assert contains_for(key, RANKS[:3]) is True
    assert contains_for(key, RANKS) is True
    # An armed rank with no journal at all: it has spilled nothing, so it holds nothing
    # to read, so a span that needs it is not a span this boot can restore.
    assert contains_for(key, RANKS + ("PP2_TP0",)) is False
    verdict = presence_of(disk_index.identity_of(key[:16], GROUP),
                          RANKS + ("PP2_TP0",))
    assert verdict["ranks_missing"] == ["PP2_TP0"], verdict


def test_the_expected_set_grows_when_a_new_journal_appears_and_withholds():
    """A rank that starts spilling AFTER the fact withholds spans until it claims them.

    The index's own rank set comes from the journal FILES it reads, and it only ever
    grows within a boot: a new journal is a new rank that must read, and until that
    rank's journal names the identity the honest answer is "not offerable". Relaxing
    here is how a false present comes back.
    """
    seeded(ranks=RANKS[:1], blocks=1)
    key = key_hex(1, GROUP)
    assert disk_index.contains(key) is True, "one journal, one rank, complete"
    assert published(disk_index.section(), "ranks_expected") == ["PP0_TP0"]
    write_journal("PP0_TP1", [])
    assert disk_index.refresh(force=True)["ok"]
    assert published(disk_index.section(), "ranks_expected") == ["PP0_TP0",
                                                                  "PP0_TP1"]
    assert disk_index.contains(key) is False, (
        "a rank whose journal exists and names nothing must veto, not be ignored")
    verdict = disk_index.membership(key)
    assert verdict["ranks_missing"] == ["PP0_TP1"], verdict
