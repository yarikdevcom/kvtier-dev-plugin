r"""The `/disk` door's CROSS-PROCESS ROLLUP: what the ranks' spill counters saw.

WHY THIS FILE EXISTS (the 2026-10-06 reclaim). The store was reclaimed live - 395.8
GiB / 34,896 files down to 161.5 GiB / 14,233 files, `/home` free 61 -> 293 GiB - and
`POST /plugins/kvtier/disk {"op":"status"}` answered:

    evictions: 0   evicted_bytes: 0   estimated_bytes: 0   ceiling_checks: 0
    estimate_seed_walks: 0   walk_basis: null   thread_alive: false

Every number zero, on a box that had just shed 234 GiB. The truth was in
`/tmp/kv-tier-diagnostics/rank-PP0_TP0.json`: `disk.evictions: 15237`,
`disk.spill_queued: 79`, `disk.ceiling_bytes: 206158430208`. Nothing was miscounted:
`disk_spill._state` is PER PROCESS, the ceiling pass and the writer thread run in the
WORKERS (four of them on this boot), and the process that answered the HTTP call had
done none of that work. So 0.2.25's whitelist - publishing `ceiling_checks` and the
`walk_*` fields - is correct and unreadable in practice, and a witness nobody can read
is not a witness.

`disk_store.rank_rollup()` reads the rank files the workers already write and publishes
`spill_rollup` as a SIBLING of `spill`. What this file is obliged to prove:

1. TONIGHT'S SIGNATURE, REVERSED. A door whose own counters are all zeros over rank
   files that are not yields a NON-ZERO rollup naming the ranks it came from.
2. A MISSING RANK IS NAMED, not folded into a confidently low total - the same failure
   mode as a truncated ceiling walk reporting its floor as a measurement, which is the
   bug 0.2.25 fixed one level down.
3. A STALE FILE IS A STALE FILE. An old `ts` excludes that rank from the totals and
   says so, with the rank's last-known numbers published separately under its own name.
4. `spill` IS UNTOUCHED. Byte-identical with and without rank files: this adds a view
   and changes no counter, no policy and no walk bound.
5. A RANK FILE IS UNTRUSTED INPUT. Truncated JSON, a non-object, an oversized file, a
   wrong `version`, a `rank` disagreeing with the file name, a counter that is a string
   or negative or absurd, a missing `ts`, a `ts` in the future, a file with no `disk`
   section: each is classified and excluded, and the door still answers `ok: true`.
6. NOTHING IS WRITTEN, NOTHING IS STARTED, NOTHING RAISES.

The fixtures are `test_disk_restore.py`'s rank-file idiom (write the document
`worker_diag` writes, into `VLLM_E2_TIER_DIAG`) and `test_blockdump.py`'s
`handshake_names_ranks` for the expected rank set; no new harness.
"""

from __future__ import annotations

import json
import time

import pytest

from kvtier_dev_plugin import (
    config,
    cpu_capacity,
    disk_spill,
    disk_store,
    worker_diag,
)

RANKS = ("PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1")
# The live box's ceiling: `VLLM_E2_TIER_DISK_BYTES=206158430208` (192 GiB).
CEILING = 206158430208


@pytest.fixture
def diag(tmp_path, monkeypatch):
    """The diagnostics directory, named the way production names it."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    return directory


# --- the fixtures: rank files, exactly as the worker writes them --------------

@pytest.fixture(autouse=True)
def no_leaked_handshake(monkeypatch):
    """No rank set from another file's handshake.

    `cpu_capacity._state` is not in conftest's reset list, so a leaked handshake from
    another test file would silently make this file's `ranks_expected` non-empty and
    turn an "unknown fleet" assertion into a passing test that tested nothing.
    """
    monkeypatch.setitem(cpu_capacity._state, "reported", {})


def disk_section(**overrides) -> dict:
    """The `disk` section's shape, in `disk_store.worker_report`'s key names.

    Written out rather than produced, so a rename on either side is a test failure;
    `test_the_writer_and_the_reader_agree_on_the_field_names` closes that loop from the
    other direction, against the writer's real output.
    """
    section = {"section": "disk", "writer": "kvtier-dev-plugin/disk-store-2",
               "installed": True, "armed": True, "ceiling_bytes": CEILING,
               "files_written": 0, "bytes_written": 0, "evictions": 0,
               "spill_queued": 0, "spill_errors": 0, "queue_depth": 0,
               "writer_thread_alive": True}
    section.update(overrides)
    return section


def rank_document(rank, disk=None, *, ts=None, epoch="aabbccdd") -> dict:
    """One rank file, with the fields the reader REQUIRES, as `worker_diag` writes."""
    document = {"version": worker_diag.SCHEMA_VERSION, "epoch": epoch,
                "ts": round(time.time(), 6) if ts is None else ts,
                "role": "worker", "rank": rank,
                "arena": {"bytes_total": 0, "bytes_used": 0, "num_blocks": 0},
                "counters": {"stored": 0, "restored_hits": 0,
                             "store_events_pending": 0, "load_events_pending": 0},
                "writer": {"plugin": "kvtier-dev-plugin/1", "writes": 1,
                           "epoch_scope": "process"},
                "error": None}
    document["disk"] = disk_section() if disk is None else disk
    return document


def write_rank(directory, rank, disk=None, **kwargs):
    path = directory / f"rank-{rank}.json"
    path.write_text(json.dumps(rank_document(rank, disk, **kwargs)),
                    encoding="utf-8")
    return path


def rewrite(path, **changes) -> None:
    """One field changed in a file already written the honest way."""
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    path.write_text(json.dumps(document), encoding="utf-8")


def handshake_names_ranks(monkeypatch, *ranks) -> None:
    """The rank SET, as the EngineCore learns it: `cpu_capacity`'s handshake keys.

    The helper `test_blockdump.py` uses for the same reason - setting that dict is
    honest (the route process really does see it) and wrapping the handshake method is
    `test_cpu_capacity.py`'s business, not this file's.
    """
    monkeypatch.setitem(cpu_capacity._state, "reported",
                        {f"{pp},{tp}": 100 for pp, tp in ranks})


def totals(payload: dict) -> dict:
    return payload["spill_rollup"]["totals"]


# --- 1. tonight's signature: the door is zero and the ranks are not -----------

def test_tonights_signature_a_zero_door_over_a_working_fleet(diag):
    """THE regression this release exists for, as one assertion.

    Four ranks, all with real work in their own counters; this process (the one
    answering the call) with none. Before 0.2.26 the payload could only say "zero", and
    the second answer - "the work happened in a process you are not looking at" - was
    not expressible at all.
    """
    write_rank(diag, "PP0_TP0", disk_section(evictions=15237, files_written=11841,
                                             spill_queued=79, spill_errors=0,
                                             bytes_written=163686041600))
    write_rank(diag, "PP0_TP1", disk_section(evictions=15010, files_written=11500,
                                             spill_queued=61, spill_errors=0,
                                             bytes_written=159000000000))
    write_rank(diag, "PP1_TP0", disk_section(evictions=412, files_written=900,
                                             spill_queued=3, spill_errors=2,
                                             bytes_written=12000000000))
    write_rank(diag, "PP1_TP1", disk_section(evictions=101, files_written=233,
                                             spill_queued=0, spill_errors=0,
                                             bytes_written=3000000000))

    answer = disk_store.disk_for_engine(None, {"op": "status"})

    assert answer["ok"] is True
    # The door's own counters, still zero, still truthful about themselves.
    assert answer["spill"]["evictions"] == 0
    assert answer["spill"]["ceiling_checks"] == 0
    assert answer["spill"]["thread_alive"] is False
    # And the fleet, which is the fact 2026-10-06 was missing.
    assert totals(answer)["evictions"] == 15237 + 15010 + 412 + 101
    assert totals(answer)["spilled"] == 11841 + 11500 + 900 + 233
    assert totals(answer)["spill_queued"] == 79 + 61 + 3
    assert totals(answer)["spill_errors"] == 2
    assert totals(answer)["bytes_written"] == (163686041600 + 159000000000
                                               + 12000000000 + 3000000000)
    assert answer["spill_rollup"]["ranks_counted"] == list(RANKS)
    assert answer["spill_rollup"]["excluded_total"] == 0
    assert answer["spill_rollup"]["basis"] == "rank_files_fleet_unknown"
    # Per-rank, so a reader can see the split (PP0 did the work; PP1 barely did).
    per_rank = {row["rank"]: row for row in answer["spill_rollup"]["per_rank"]}
    assert per_rank["PP0_TP0"]["evictions"] == 15237
    assert per_rank["PP1_TP1"]["evictions"] == 101
    assert per_rank["PP0_TP0"]["age_seconds"] < 5.0


def test_the_rollup_names_this_process_and_says_it_is_not_the_fleet(diag):
    """`scope` and `kinds` stop the two views being read as one number."""
    write_rank(diag, "PP0_TP0")
    rollup = disk_store.rank_rollup()
    assert "SUM of each rank's OWN per-process counters" in rollup["scope"]
    assert "THIS process's counters" in rollup["scope"]
    assert rollup["this_process"]["rank"] is None  # no distributed state in a test
    assert rollup["this_process"]["counted_in_rollup"] is None
    assert rollup["kinds"]["queue_depth"] == "gauge"
    assert rollup["kinds"]["evictions"] == "counter-since-boot"
    assert rollup["units"]["bytes_written"] == "bytes"
    assert rollup["meanings"]["spilled"]


def test_a_complete_fleet_says_complete_and_its_totals_are_not_a_bound(diag,
                                                                       monkeypatch):
    """(b) from the brief: every rank file is fresh, and the payload can SAY so."""
    handshake_names_ranks(monkeypatch, (0, 0), (0, 1), (1, 0), (1, 1))
    for rank in RANKS:
        write_rank(diag, rank, disk_section(evictions=5))
    rollup = disk_store.rank_rollup()
    assert rollup["basis"] == "rank_files"
    assert rollup["complete"] is True
    assert rollup["totals_are_lower_bound"] is False
    assert rollup["ranks_missing"] == []
    assert rollup["ranks_expected"] == list(RANKS)
    assert "capacity handshake" in rollup["ranks_expected_source"]
    assert rollup["totals"]["evictions"] == 20
    assert rollup["degraded"] is False
    assert rollup["this_process"]["counted_in_rollup"] is None


# --- 2. a missing rank is named, not silently missing from a low total --------

def test_a_missing_rank_file_is_partial_and_names_the_rank(diag, monkeypatch):
    """THE dangerous case, pinned.

    A rollup that reported the 3-file sum with no flag is indistinguishable from "the
    fleet evicted this much", and the reader's next move is to conclude the ceiling is
    holding. So the sum is published (it is what we know) AND marked a lower bound with
    the missing rank named - the rule the ceiling walk learned the hard way.
    """
    handshake_names_ranks(monkeypatch, (0, 0), (0, 1), (1, 0), (1, 1))
    for rank in ("PP0_TP0", "PP0_TP1", "PP1_TP0"):
        write_rank(diag, rank, disk_section(evictions=100))

    answer = disk_store.disk_for_engine(None, {"op": "status"})
    rollup = answer["spill_rollup"]
    assert rollup["basis"] == "rank_files_partial"
    assert rollup["degraded"] is True
    assert rollup["complete"] is False
    assert rollup["ranks_missing"] == ["PP1_TP1"]
    assert rollup["totals_are_lower_bound"] is True
    assert "PP1_TP1" in rollup["lower_bound_reason"]
    # The sum is still the honest sum of what DID report - never padded, never guessed.
    assert rollup["totals"]["evictions"] == 300
    assert rollup["totals_reported_by"]["evictions"] == 3
    assert rollup["ranks_counted"] == ["PP0_TP0", "PP0_TP1", "PP1_TP0"]


def test_an_unknown_rank_set_never_claims_completeness(diag):
    """No handshake in this process: the rollup may not call itself complete.

    This is the route-process/worker split in the flesh - the process answering the call
    may genuinely not know how many ranks the boot has. Saying so is the whole
    difference between a lower bound and a lie.
    """
    write_rank(diag, "PP0_TP0", disk_section(evictions=7))
    rollup = disk_store.rank_rollup()
    assert rollup["ranks_expected"] is None
    assert rollup["ranks_expected_source"]
    assert rollup["complete"] is None
    assert rollup["totals_are_lower_bound"] is True
    assert rollup["basis"] == "rank_files_fleet_unknown"
    assert "rank set is unknown" in rollup["lower_bound_reason"]
    assert rollup["totals"]["evictions"] == 7


# --- 3. staleness: an old heartbeat is excluded, and named --------------------

def test_a_stale_rank_file_is_excluded_and_named(diag, monkeypatch):
    """An old file is a rank that may be gone: its numbers may not enter a total.

    The stale rank's last-known counters are STILL published (under `stale_ranks`, with
    the age), because "PP1_TP0 was evicting hard and then stopped" is exactly the story
    an operator needs - published where they cannot be mistaken for current.
    """
    handshake_names_ranks(monkeypatch, (0, 0), (1, 0))
    write_rank(diag, "PP0_TP0", disk_section(evictions=10))
    write_rank(diag, "PP1_TP0", disk_section(evictions=99999),
               ts=time.time() - (disk_store.RANK_ROLLUP_FRESH_SECONDS + 1200))

    answer = disk_store.disk_for_engine(None, {"op": "status"})
    rollup = answer["spill_rollup"]
    assert rollup["basis"] == "rank_files_partial"
    assert rollup["complete"] is False
    assert rollup["totals"]["evictions"] == 10, \
        "a stale rank's numbers are not evidence about the present"
    assert rollup["totals_reported_by"]["evictions"] == 1
    assert [row["rank"] for row in rollup["stale_ranks"]] == ["PP1_TP0"]
    stale = rollup["stale_ranks"][0]
    assert stale["status"] == "stale"
    assert stale["evictions"] == 99999, "the last-known numbers are still published"
    assert stale["age_seconds"] > disk_store.RANK_ROLLUP_FRESH_SECONDS
    assert "PP1_TP0" in rollup["lower_bound_reason"]
    assert "stale" in rollup["lower_bound_reason"]
    assert rollup["fresh_seconds"] == disk_store.RANK_ROLLUP_FRESH_SECONDS


def test_a_clock_ahead_stamp_is_not_freshness(diag):
    """A file stamped in the FUTURE would stay "fresh" forever; it is excluded instead.

    Rank files cross a process (and in production a host) boundary, so clock skew is
    real and the rule has to be one-sided - the rule `disk_restore`'s proof read already
    applies to these same files.
    """
    write_rank(diag, "PP0_TP0", disk_section(evictions=42), ts=time.time() + 3600)
    rollup = disk_store.rank_rollup()
    assert rollup["basis"] == "rank_files_unusable"
    assert rollup["excluded_files"][0]["status"] == "clock_ahead"
    assert rollup["totals"]["evictions"] is None, \
        "null, not zero: no rank was counted"


def test_a_missing_ts_is_named_not_assumed_fresh(diag):
    write_rank(diag, "PP0_TP0", disk_section(evictions=1))
    rewrite(diag / "rank-PP0_TP0.json", ts=None)
    rollup = disk_store.rank_rollup()
    assert rollup["excluded_files"][0]["status"] == "no_ts"
    assert rollup["totals"]["evictions"] is None


# --- 4. `spill` is untouched: this is additive or it is not allowed -----------

def test_the_spill_block_is_byte_identical_with_and_without_rank_files(diag):
    """(d) from the brief, as a byte comparison.

    `spill` is what other readers (the proxy's KV-tier reader, the dashboards) already
    parse, with the meanings 0.2.25 published for `ceiling_checks` and the `walk_*`
    fields. Those keys and values are this release's compatibility contract: the rollup
    is a SIBLING and may not reach inside.
    """
    without = disk_store.disk_for_engine(None, {"op": "status"})
    baseline = json.dumps(without["spill"], sort_keys=True, default=str)

    for rank in RANKS:
        write_rank(diag, rank, disk_section(evictions=1234, files_written=5678,
                                            bytes_written=99, spill_queued=7,
                                            spill_errors=1, queue_depth=3))
    with_ranks = disk_store.disk_for_engine(None, {"op": "status"})

    assert json.dumps(with_ranks["spill"], sort_keys=True,
                      default=str) == baseline
    assert set(with_ranks["spill"]) == set(without["spill"])
    # A sibling, never a guest: the rollup does not live inside `spill`.
    assert "spill_rollup" in with_ranks and "spill_rollup" in without
    assert "rank_totals" not in with_ranks["spill"]
    assert with_ranks["spill_rollup"]["totals"]["evictions"] == 1234 * 4


def test_the_rollup_ships_the_same_keys_whatever_happened(diag, monkeypatch,
                                                          tmp_path):
    """A degraded view and a healthy one ship the SAME keys, or readers guess.

    The house rule the proxy host's reader applies to these same files: build the
    degraded shape through the same code as the healthy one, so the two cannot drift.
    """
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "empty-diag"))
    shapes = [disk_store.rank_rollup()]                    # no such directory
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    shapes.append(disk_store.rank_rollup())                 # no rank files yet
    (diag / "rank-broken.json").write_text("{ not json")
    shapes.append(disk_store.rank_rollup())                 # unusable files
    write_rank(diag, "PP0_TP0", disk_section(evictions=3))
    (diag / "rank-broken.json").unlink()
    handshake_names_ranks(monkeypatch, (0, 0))
    shapes.append(disk_store.rank_rollup())                 # complete

    assert [shape["basis"] for shape in shapes] == [
        "directory_unreadable", "no_rank_files", "rank_files_unusable", "rank_files"]
    keys = set(shapes[0])
    for shape in shapes[1:]:
        assert set(shape) == keys, sorted(set(shape) ^ keys)


# --- 5. a rank file is untrusted input ---------------------------------------

@pytest.mark.parametrize("raw, status", (
    ("{ not json at all", "malformed"),
    ("", "malformed"),
    ("[1, 2, 3]", "malformed"),
    ('{"version": 1, "ts": 1}', "mislabeled"),      # no `rank` field at all
))
def test_an_unreadable_rank_file_is_data_and_never_a_crash(diag, raw, status):
    (diag / "rank-PP9_TP9.json").write_text(raw)
    write_rank(diag, "PP0_TP0", disk_section(evictions=5))
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    assert answer["ok"] is True, "the door answers 200 with a partial rollup, never 500"
    rollup = answer["spill_rollup"]
    assert rollup["totals"]["evictions"] == 5
    assert rollup["excluded_total"] == 1
    assert rollup["excluded_files"][0]["status"] == status
    assert rollup["basis"] == "rank_files_partial"


def test_absurd_and_wrong_type_counters_are_rejected_for_that_rank_only(diag):
    """A hostile or broken writer cannot put a fake total in front of the witness.

    Every field is bounded on its own, so one rank's garbage costs THAT FIELD for THAT
    RANK and is named in `fields_rejected`; the other rank's honest numbers still sum,
    and `totals_reported_by` says the total spans one file rather than two.
    """
    write_rank(diag, "PP0_TP0", disk_section(evictions=-5, spill_queued="lots",
                                             bytes_written=10 ** 18,
                                             files_written=11, spill_errors=0))
    write_rank(diag, "PP0_TP1", disk_section(evictions=8, spill_queued=2,
                                             bytes_written=1000, files_written=4,
                                             spill_errors=1))
    rollup = disk_store.rank_rollup()
    row = next(r for r in rollup["per_rank"] if r["rank"] == "PP0_TP0")
    assert row["evictions"] is None and row["spill_queued"] is None
    assert row["bytes_written"] is None
    assert sorted(row["fields_rejected"]) == ["bytes_written", "evictions",
                                              "spill_queued"]
    assert row["fields_rejected_note"]
    assert rollup["totals"]["evictions"] == 8
    assert rollup["totals_reported_by"]["evictions"] == 1
    assert rollup["totals"]["spilled"] == 11 + 4
    assert rollup["totals"]["spill_errors"] == 1
    assert rollup["totals_reported_by"]["spill_errors"] == 2


def test_a_version_this_build_does_not_read_is_excluded_by_name(diag):
    """A future writer's file is not this reader's data - but it is REPORTED data."""
    write_rank(diag, "PP0_TP0", disk_section(evictions=9))
    rewrite(diag / "rank-PP0_TP0.json", version=worker_diag.SCHEMA_VERSION + 1)
    rollup = disk_store.rank_rollup()
    assert rollup["excluded_files"][0]["status"] == "bad_version"
    assert rollup["totals"]["evictions"] is None


def test_a_rank_field_disagreeing_with_the_file_name_is_mislabeled(diag):
    """The name and the `rank` field are ONE fact in `worker_diag`; a disagreement means
    a hand-edited or half-renamed file, and a row keyed by the wrong rank would put
    another rank's evictions under this rank's name."""
    path = write_rank(diag, "PP0_TP0", disk_section(evictions=9))
    rewrite(path, rank="PP1_TP1")
    rollup = disk_store.rank_rollup()
    assert rollup["excluded_files"][0]["status"] == "mislabeled"
    assert "PP1_TP1" in rollup["excluded_files"][0]["reason"]
    assert rollup["totals"]["evictions"] is None


def test_a_file_too_big_to_trust_is_not_read(diag):
    """The byte ceiling is the first bound, because reading is the thing to avoid."""
    huge = rank_document("PP0_TP1", disk_section(evictions=1))
    huge["disk"]["note"] = "y" * disk_store.MAX_RANK_ROLLUP_FILE_BYTES
    (diag / "rank-PP0_TP1.json").write_text(json.dumps(huge), encoding="utf-8")
    rollup = disk_store.rank_rollup()
    assert rollup["excluded_files"][0]["status"] == "too_large"
    assert rollup["totals"]["evictions"] is None
    assert rollup["basis"] == "rank_files_unusable"


def test_a_file_with_no_disk_section_says_that_rank_never_registered_it(diag):
    """A rank file whose writer never contributed `disk` is ABSENT evidence, named.

    `worker_diag` deliberately omits a section a module never registered, so "no `disk`
    section" means that worker had no spill tier fitted - a fact about that rank, not a
    zero to be summed.
    """
    write_rank(diag, "PP0_TP0")
    rewrite(diag / "rank-PP0_TP0.json", disk=None)
    rollup = disk_store.rank_rollup()
    assert rollup["excluded_files"][0]["status"] == "no_disk_section"
    assert "never" in rollup["excluded_files"][0]["reason"]
    assert rollup["totals"]["evictions"] is None


def test_the_rollup_is_bounded_and_says_so(diag, monkeypatch):
    """`MAX_RANK_ROLLUP_FILES` is a real cap on a polled door, and a capped listing is
    a lower bound like any other gap."""
    monkeypatch.setattr(disk_store, "MAX_RANK_ROLLUP_FILES", 3)
    for index in range(6):
        write_rank(diag, f"PP0_TP{index}", disk_section(evictions=10))
    rollup = disk_store.rank_rollup()
    assert rollup["rank_files_found"] == 6
    assert rollup["listing_truncated"] is True
    assert rollup["complete"] is None
    assert rollup["totals_are_lower_bound"] is True
    assert "files were opened" in rollup["lower_bound_reason"]
    assert rollup["totals"]["evictions"] == 30, "the cap bounds the read, not the claim"


def test_no_diagnostics_dir_is_no_evidence_rather_than_zero():
    """The unset case: nulls and a reason, so nobody reads it as "nothing evicted"."""
    rollup = disk_store.rank_rollup()
    assert rollup["basis"] == "no_diagnostics_dir"
    assert rollup["degraded"] is True
    assert rollup["totals"]["evictions"] is None
    assert rollup["totals_reported_by"]["evictions"] == 0
    assert rollup["complete"] is None
    assert config.ENV_DIAG_DIR in rollup["reason"]
    assert "NOT evidence" in rollup["reason"]


def test_no_rank_files_yet_is_a_different_claim_from_zero(diag):
    rollup = disk_store.rank_rollup()
    assert rollup["basis"] == "no_rank_files"
    assert rollup["rank_files_found"] == 0
    assert rollup["totals"]["evictions"] is None
    assert "NO EVIDENCE" in rollup["reason"]


def test_an_unreadable_directory_is_a_view_not_a_500(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "not-there"))
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    assert answer["ok"] is True
    assert answer["spill_rollup"]["basis"] == "directory_unreadable"
    assert answer["spill_rollup"]["totals"]["evictions"] is None


def test_a_rollup_that_raises_still_answers_the_door(diag, monkeypatch):
    """Even a bug in the reader may not become a 500 on the door."""
    def explode(*_args, **_kwargs):
        raise ZeroDivisionError("a future reader divided by zero")
    monkeypatch.setattr(disk_store, "_rank_file_rollup_row", explode)
    write_rank(diag, "PP0_TP0", disk_section(evictions=1))
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    assert answer["ok"] is True
    assert answer["spill_rollup"]["basis"] == "rollup_failed"
    assert answer["spill_rollup"]["degraded"] is True
    assert answer["spill"]["evictions"] == 0, "the op's own fields are untouched"


# --- 6. read-only, and the fields that made tonight legible ------------------

def test_the_rollup_writes_nothing_and_creates_no_file(diag):
    write_rank(diag, "PP0_TP0", disk_section(evictions=2))
    before = sorted(entry.name for entry in diag.iterdir())
    disk_store.rank_rollup()
    disk_store.disk_for_engine(None, {"op": "status"})
    assert sorted(entry.name for entry in diag.iterdir()) == before


def test_the_writer_and_the_reader_agree_on_the_field_names(diag, monkeypatch):
    """Round trip: `worker_report`'s OWN output, read back by the rollup.

    The rank file's `disk` section is written by `disk_store.worker_report` under names
    that are NOT all the same as `disk_spill`'s (`files_written` for `spilled`), so the
    reader is pinned against the writer's real output rather than against a fixture that
    could drift along with it.
    """
    disk_spill._state["spilled"] = 4321
    disk_spill._state["bytes_written"] = 555_000
    disk_spill._state["evictions"] = 777
    disk_spill._state["errors"] = 3
    disk_spill._state["queue_depth"] = 9
    disk_store._state["spill_queued"] = 12
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(CEILING))
    section = disk_store.worker_report()
    assert section["evictions"] == 777 and section["files_written"] == 4321
    write_rank(diag, "PP0_TP0", section)

    rollup = disk_store.rank_rollup()
    assert rollup["totals"]["evictions"] == 777
    assert rollup["totals"]["spilled"] == 4321
    assert rollup["totals"]["bytes_written"] == 555_000
    assert rollup["totals"]["spill_errors"] == 3
    assert rollup["totals"]["spill_queued"] == 12
    assert rollup["totals"]["queue_depth"] == 9
    assert rollup["ceiling_bytes"]["values"] == [CEILING]
    assert rollup["ceiling_bytes"]["agree"] is True
    assert rollup["writer_threads"]["not_alive"] == 1, \
        "the round trip carries the writer's own thread verdict (no thread in a test)"
    assert rollup["complete"] is None     # the fleet is unknown here, so never True


def test_the_writer_threads_are_alive_in_the_ranks_even_when_here_there_is_none(diag):
    """Tonight's `thread_alive: false`, corrected by the fleet's own answer."""
    for rank in ("PP0_TP0", "PP0_TP1"):
        write_rank(diag, rank, disk_section(writer_thread_alive=True))
    write_rank(diag, "PP1_TP0", disk_section(writer_thread_alive=False))
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    assert answer["spill"]["thread_alive"] is False
    threads = answer["spill_rollup"]["writer_threads"]
    assert threads["alive"] == 2 and threads["not_alive"] == 1
    assert threads["unknown"] == 0
    rewrite(diag / "rank-PP1_TP0.json", disk=None)
    rollup = disk_store.rank_rollup()
    assert rollup["excluded_files"][0]["status"] == "no_disk_section"
    assert rollup["writer_threads"]["alive"] == 2
    assert rollup["writer_threads"]["not_alive"] == 0, \
        "a rank with no `disk` section is excluded, not counted as a dead thread"


def test_a_ceiling_disagreement_between_ranks_is_named(diag):
    """Two ranks under different limits is a real state, and it explains a store that
    grows on one rank while another evicts. A ceiling is not summable, so it is
    published as the SET of them with the disagreement said out loud."""
    write_rank(diag, "PP0_TP0", disk_section(ceiling_bytes=CEILING))
    write_rank(diag, "PP0_TP1", disk_section(ceiling_bytes=CEILING // 2))
    rollup = disk_store.rank_rollup()
    assert rollup["ceiling_bytes"]["agree"] is False
    assert rollup["ceiling_bytes"]["min"] == CEILING // 2
    assert rollup["ceiling_bytes"]["max"] == CEILING
    assert rollup["ceiling_bytes"]["values"] == [CEILING // 2, CEILING]
    assert "configured LIMIT" in rollup["ceiling_bytes"]["note"]
    assert rollup["ceiling_bytes"]["reported_by"] == 2


def test_the_epoch_field_is_published_but_never_accused(diag):
    """Worker epochs are per-process unless the operator pins one, so a disagreement is
    NORMAL here - and must not be read as "two boots", which is what the proxy host's
    reader would conclude. `ts` is what catches a previous boot's leftover file."""
    write_rank(diag, "PP0_TP0", disk_section(evictions=1), epoch="1111aaaa")
    write_rank(diag, "PP0_TP1", disk_section(evictions=2), epoch="2222bbbb")
    rollup = disk_store.rank_rollup()
    assert rollup["distinct_epochs"] == 2
    assert "PER-PROCESS" in rollup["epoch_note"]
    assert rollup["totals"]["evictions"] == 3


def test_a_previous_boots_leftover_is_stale_not_current(diag, monkeypatch):
    """The leftover-file case a rank file really does produce: the directory survives a
    restart, the counters in it do not describe this boot."""
    handshake_names_ranks(monkeypatch, (0, 0))
    write_rank(diag, "PP0_TP0", disk_section(evictions=4), epoch="oldboot1")
    rewrite(diag / "rank-PP0_TP0.json",
            ts=time.time() - (disk_store.RANK_ROLLUP_FRESH_SECONDS * 3))
    rollup = disk_store.rank_rollup()
    assert rollup["totals"]["evictions"] is None
    assert rollup["basis"] == "rank_files_unusable"
    assert rollup["complete"] is False
    # The file EXISTS, so this is not a missing rank; it is a stale one, and naming it
    # in only one of the two lists is what keeps the gap countable.
    assert rollup["ranks_missing"] == []
    assert [row["rank"] for row in rollup["stale_ranks"]] == ["PP0_TP0"]
    assert "PP0_TP0" in rollup["lower_bound_reason"]


def test_a_worker_answering_its_own_door_says_the_two_views_overlap(diag, monkeypatch):
    """The double-count trap: when the door IS a rank, `spill` and `totals` overlap.

    A reader that adds `spill.evictions` to `totals.evictions` gets a number wrong by
    exactly one rank, so the payload has to say which of the two cases it is looking
    at - and it must keep saying so on the paths that never reach the counting loop.
    """
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    handshake_names_ranks(monkeypatch, (0, 0))
    disk_spill._state["evictions"] = 40
    write_rank(diag, "PP0_TP0", disk_section(evictions=40))
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    rollup = answer["spill_rollup"]
    assert answer["rank"] == "PP0_TP0"
    assert rollup["this_process"]["counted_in_rollup"] is True
    assert "never add them" in rollup["this_process"]["note"]
    assert rollup["complete"] is True
    assert rollup["totals"]["evictions"] == answer["spill"]["evictions"] == 40

    # Pull that rank's file and the same door says the OPPOSITE about itself.
    (diag / "rank-PP0_TP0.json").unlink()
    rollup = disk_store.rank_rollup()
    assert rollup["basis"] == "no_rank_files"
    assert rollup["this_process"]["counted_in_rollup"] is False
    assert "different processes" in rollup["this_process"]["note"]


def test_missing_stale_and_unusable_are_three_different_lists(diag, monkeypatch):
    """A rank appears in AT MOST ONE of `ranks_missing` / `stale_ranks` /
    `excluded_files`, and the right words are used for each.

    "No file for PP1_TP1", "PP1_TP0's file is hours old" and "rank-PP0_TP1.json will not
    parse" are three different operator actions; a single bucket called `missing` would
    make the first two look the same and the third invisible.
    """
    handshake_names_ranks(monkeypatch, (0, 0), (0, 1), (1, 0), (1, 1))
    write_rank(diag, "PP0_TP0", disk_section(evictions=1))
    (diag / "rank-PP0_TP1.json").write_text("{ truncated", encoding="utf-8")
    write_rank(diag, "PP1_TP0", disk_section(evictions=2),
               ts=time.time() - (disk_store.RANK_ROLLUP_FRESH_SECONDS + 60))
    # PP1_TP1: nothing written at all.

    rollup = disk_store.rank_rollup()
    assert rollup["ranks_missing"] == ["PP1_TP1"]
    assert [row["rank"] for row in rollup["stale_ranks"]] == ["PP1_TP0"]
    assert [row["file"] for row in rollup["excluded_files"]] == ["rank-PP0_TP1.json"]
    assert rollup["ranks_counted"] == ["PP0_TP0"]
    assert rollup["complete"] is False
    assert rollup["totals_are_lower_bound"] is True
    reason = rollup["lower_bound_reason"]
    assert "no file at all from PP1_TP1" in reason
    assert "stale" in reason and "PP1_TP0" in reason
    assert "rank-PP0_TP1.json" in reason
    # Each rank is named once: the three lists are disjoint.
    named = ([row["rank"] for row in rollup["stale_ranks"]]
             + [row.get("rank") for row in rollup["excluded_files"] if row.get("rank")])
    assert len(named) == len(set(named)) == 2
