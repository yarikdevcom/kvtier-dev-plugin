r"""`disk_fill`: ONE BLOCK answering "how full is the SSD tier, and is that a
problem".

WHY THIS FILE EXISTS. On 2026-10-22 the operator read the tier's state across four rank
files and one `/snapshot` and still had to do the arithmetic by hand, because the block
that had the numbers was the WRONG PROCESS's: `disk_spill.spilled` and
`disk_store.spill.evictions` are per-process counters and the process answering HTTP
runs no writer thread, so they read zero while four ranks had written 92 files each and
evicted 7,956 between them. `disk_fill` is the answer that question in ONE read-only
block: how many bytes the store holds and on what basis, the ceiling and the 0.9 target
it is kept under, one entry per rank with the age of the file it came from, which ranks
did NOT report, and whether the ranks' ceiling passes agree with each other.

WHAT EACH TEST OWNS:

* THE FILL ITSELF, as arithmetic: `used_fraction` over `target_fraction`, and
  `target_bytes` = `0.9 x ceiling`. A dashboard must not have to derive either.
* `basis` IS PRESERVED, and the distinction is the point: a rank that statted every
  file and a rank that sampled 20,000 of 34,896 and scaled them make different claims,
  and `read_basis` (how complete the READ was) is a separate claim from `basis` (how the
  SIZE was measured) and a third from `ceiling_basis` (whose ceiling this is).
* NOTHING IS FABRICATED. No rank file => `used_bytes: null`, not 0. Ranks that reported
  counters but never ran a ceiling pass => null and a reason. A missing rank is NAMED in
  `missing_ranks` rather than silently lowering a total. `ranks_expected` is null (not
  `[]`) when this process cannot know the fleet.
* THE BLOCK IS BOUNDED. It is polled by a dashboard every few seconds, so it carries
  rank entries and nothing else - no file lists, no sample arrays - with a hard cap on
  rank entries and an assertion on the serialised size.
* THE CROSS-PROCESS VERDICT, including the 0.2.30 signature: one rank's pass did the
  work and three DECLINED because the store was already under their share. Four ranks
  evicting the same bytes with zero declines is the old defect, and this block is where
  an operator can see that without opening four files.
* THE PER-PROCESS LABEL LIVES INSIDE THE BLOCK. A reader that never leaves `disk_fill`
  still cannot mistake `disk_spill.spilled: 0` for a tier that spilled nothing.
* THE ONE-SECOND RANK CACHE reparses when a rank file CHANGES, so a burst of polls
  costs one parse and a rank that just wrote is not invisible for a second.
* IT IS ADDITIVE. Every key that existed in `/snapshot` before this block still exists
  with the same meaning, and the flat per-process blocks are unchanged.

No engine, no network, no filesystem walk of a real store: rank files are written the
way `worker_diag` writes them, into `VLLM_E2_TIER_DIAG`, exactly as
`test_disk_spill_rank_rollup.py` does.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from kvtier_dev_plugin import (
    config,
    cpu_capacity,
    disk_store,
    snapshot,
    worker_diag,
)

RANKS = ("PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1")
CEILING = 148 * 1024 ** 3           # the live box's spill ceiling, 158,913,789,952
USED = 156 * 1024 ** 3              # where the store actually stood on 2026-10-22
TARGET = int(CEILING * 0.9)


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_fleet(tmp_path, monkeypatch):
    """A diagnostics directory, no leaked handshake, no leaked one-second cache."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.delenv(config.ENV_DISK_BYTES, raising=False)
    # `cpu_capacity._state` is not in conftest's reset list, and a handshake leaked
    # from another file would turn every "the fleet is unknown here" assertion below
    # into a test that asserts nothing.
    monkeypatch.setitem(cpu_capacity._state, "reported", {})
    disk_store.reset_rank_rollup_cache_for_tests()
    yield directory
    disk_store.reset_rank_rollup_cache_for_tests()


def rank_document(rank, disk=None, restore=None, *, ts=None,
                  epoch="aabbccdd") -> dict:
    """One rank file, with the fields the reader REQUIRES."""
    document = {"version": worker_diag.SCHEMA_VERSION, "epoch": epoch,
                "ts": round(time.time(), 6) if ts is None else ts,
                "role": "worker", "rank": rank, "error": None}
    document["disk"] = disk if disk is not None else {}
    if restore is not None:
        document["disk_restore"] = restore
    return document


def write_rank(directory, rank, *, disk=None, restore=None, **kwargs):
    """Write one rank file and hand back its path."""
    path = directory / f"rank-{rank}.json"
    document = rank_document(rank, disk, restore, **kwargs)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def disk_block(**overrides) -> dict:
    """A rank's `disk` section carrying the fill-relevant fields.

    Numbers are the incident's, scaled: one rank did the work (2,106 evictions) and its
    siblings declined, which is what the fair share is supposed to look like.
    """
    section = {"section": "disk", "writer": "kvtier-dev-plugin/disk-store-2",
               "installed": True, "armed": True, "ceiling_bytes": CEILING,
               "ceiling_share_bytes": CEILING // 4, "ceiling_divisor": 4,
               "ceiling_passes_declined": 0,
               "estimated_bytes": USED,
               "ceiling_measurement_basis": "full_measurement",
               "files_written": 92, "bytes_written": 92 * 13_142_016,
               "evictions": 2106, "journal_removals": 2106, "drops": 0,
               "spill_queued": 79, "spill_errors": 0, "queue_depth": 0,
               "writer_thread_alive": True}
    section.update(overrides)
    return section


def restore_block(**overrides) -> dict:
    """The `disk_restore` contribution's reader half, where the touch clock lives."""
    readers = {"recency_promotions": 4100,
               "recency_promotions_throttled": 585, "reads_issued": 5000}
    readers.update(overrides)
    return {"section": "disk_restore", "readers": readers}


def healthy_fleet(directory, ranks=RANKS, **overrides) -> dict:
    """Four ranks, fresh files, one ceiling: the state the block must explain.

    THE SHAPES ARE DELIBERATELY UNEVEN, because the incident was uneven: ONE rank's pass
    did the shedding and the other three DECLINED because the store was already under
    their share. Four ranks all evicting is the old defect, and a fixture that shows it
    would let the verdict test pass for the wrong reason.
    """
    written = {}
    for index, rank in enumerate(ranks):
        acting = index == 0
        written[rank] = write_rank(
            directory, rank,
            disk=disk_block(evictions=(2106 if acting else 0),
                            journal_removals=(2106 if acting else 0),
                            ceiling_passes_declined=(0 if acting else index * 6),
                            **overrides),
            restore=restore_block(),
            # Newest first for rank 0, then descending, so "the freshest file" is a
            # fact a test can point at rather than one it inherits from dict order.
            ts=time.time() - index)
    return written


def size_of(payload: dict) -> int:
    return len(json.dumps(payload))


# ===========================================================================
# THE FILL, as arithmetic
# ===========================================================================

def test_the_fill_answers_without_arithmetic(clean_fleet):
    """used / ceiling / fraction / target, all four, in one read.

    The operator's complaint was not that the numbers were missing - it was that
    answering "is the tier full?" meant reading four files and doing the multiplication.
    So the assertion here is the arithmetic itself.
    """
    healthy_fleet(clean_fleet)
    fill = disk_store.disk_fill()
    assert fill["available"] is True, fill["reason"]
    assert fill["used_bytes"] == USED, fill["used_bytes"]
    assert fill["ceiling_bytes"] == CEILING, fill["ceiling_bytes"]
    assert fill["ceiling_basis"] == "ranks_agree", fill["ceiling_basis"]
    assert fill["target_fraction"] == pytest.approx(0.9)
    assert fill["target_bytes"] == TARGET
    assert fill["used_fraction"] == pytest.approx(USED / CEILING, abs=1e-6)
    # 156 GiB against a 148 GiB ceiling: over the ceiling, therefore over the target.
    assert fill["over_target"] is True and fill["over_ceiling"] is True
    assert fill["headroom_bytes"] == CEILING - USED
    assert fill["headroom_bytes"] < 0


def test_an_under_ceiling_store_reports_headroom_and_no_alarm(clean_fleet):
    """The other side of the same arithmetic, so `over_target` is a real verdict."""
    low = 100 * 1024 ** 3
    healthy_fleet(clean_fleet, estimated_bytes=low)
    fill = disk_store.disk_fill()
    assert fill["used_bytes"] == low
    assert fill["over_target"] is False and fill["over_ceiling"] is False
    assert fill["headroom_bytes"] == CEILING - low > 0
    assert fill["used_fraction"] < 0.9


# ===========================================================================
# THE BASIS DISTINCTION, and the refusal to invent a number
# ===========================================================================

def test_a_scaled_sample_is_not_reported_as_a_measurement(clean_fleet):
    """`basis` travels WITH `used_bytes`, from the rank that produced it.

    "The store is 68 GiB" and "I stat-ed 20,000 of 34,896 files and extrapolated" are
    different claims, and the second one is the one an operator acts on differently.
    """
    for index, rank in enumerate(RANKS):
        # The SAMPLED rank is also the FRESHEST one, so the block cannot be accused of
        # taking the basis from a different rank than the number.
        scaled = rank == "PP0_TP0"
        write_rank(clean_fleet, rank,
                   disk=disk_block(ceiling_measurement_basis=(
                       "sample_scaled: 20000 of 34896" if scaled
                       else "full_measurement")),
                   restore=restore_block(), ts=time.time() - index)
    fill = disk_store.disk_fill()
    assert fill["used_bytes_source"]["rank"] == "PP0_TP0", fill["used_bytes_source"]
    assert fill["basis"].startswith("sample_scaled"), fill["basis"]
    # ... and `read_basis` is a DIFFERENT claim: all four files were readable.
    assert fill["read_basis"].startswith("rank_files"), fill["read_basis"]
    assert fill["basis"] != fill["read_basis"]


def test_no_rank_files_yields_a_null_and_a_reason_never_a_zero(clean_fleet):
    """THE FABRICATION RULE. An unmeasured store is `null`, not 0.

    `used_bytes: 0` would render as "the tier is empty" on a dashboard, which is the
    opposite of true on a box whose workers have not written a rank file yet.
    """
    fill = disk_store.disk_fill()
    assert fill["available"] is False
    assert fill["used_bytes"] is None, fill["used_bytes"]
    assert fill["basis"] == "no_rank_files", fill["basis"]
    assert fill["reason"] and "UNKNOWN" in fill["reason"], fill["reason"]
    assert fill["ranks_reported"] == 0
    # A ceiling with no measurement is still a claim worth making, but a FRACTION of a
    # number that does not exist is not.
    assert fill["used_fraction"] is None
    assert fill["over_target"] is None and fill["over_ceiling"] is None


def test_counters_without_a_measurement_are_reported_as_such(clean_fleet):
    """A rank file from before 0.2.30 has counters but no store estimate.

    It is counted (its evictions reach the totals) and the fill stays null: a partial
    read must never become a number.
    """
    write_rank(clean_fleet, "PP0_TP0",
               disk={"installed": True, "ceiling_bytes": CEILING,
                     "files_written": 92, "bytes_written": 1_000_000,
                     "evictions": 2106, "writer_thread_alive": True},
               restore=restore_block())
    fill = disk_store.disk_fill()
    assert fill["available"] is False
    assert fill["used_bytes"] is None
    assert fill["basis"] == "no_rank_reported_a_measurement", fill["basis"]
    assert fill["ranks_reported"] == 1
    assert fill["totals"]["evictions"] == 2106
    assert fill["per_rank"][0]["evictions"] == 2106
    assert fill["per_rank"][0]["estimated_bytes"] is None


def test_a_missing_rank_is_named_and_an_unknown_fleet_is_null_not_empty(
        clean_fleet, monkeypatch):
    """`ranks_expected` / `ranks_reported` / `missing_ranks`, all three.

    With a handshake the block can say WHICH rank is silent. Without one it says null:
    an empty list would be a claim that nobody is missing, which is not what an empty
    handshake knows.
    """
    monkeypatch.setitem(cpu_capacity._state, "reported",
                        {f"{pp},{tp}": 100 for pp, tp in
                         (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"))})
    healthy_fleet(clean_fleet, ranks=("PP0_TP0", "PP0_TP1", "PP1_TP0"))
    fill = disk_store.disk_fill()
    assert fill["ranks_expected"] == 4, fill["ranks_expected"]
    assert fill["ranks_reported"] == 3
    assert fill["missing_ranks"] == ["PP1_TP1"], fill["missing_ranks"]
    assert fill["totals_are_lower_bound"] is True
    assert fill["lower_bound_reason"]

    # The same store with no handshake: the fleet is unknown, and that is said.
    monkeypatch.setitem(cpu_capacity._state, "reported", {})
    disk_store.reset_rank_rollup_cache_for_tests()
    unknown = disk_store.disk_fill()
    assert unknown["ranks_expected"] is None, unknown["ranks_expected"]
    assert unknown["missing_ranks"] is None, unknown["missing_ranks"]
    assert unknown["ranks_reported"] == 3


def test_a_stale_or_unusable_rank_file_is_named_with_its_age(clean_fleet,
                                                             monkeypatch):
    """Silence is reported by NAME and AGE, never folded into a low total."""
    monkeypatch.setitem(cpu_capacity._state, "reported",
                        {f"{pp},{tp}": 100 for pp, tp in
                         (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"))})
    healthy_fleet(clean_fleet, ranks=("PP0_TP0", "PP0_TP1"))
    # A rank that stopped writing an hour ago: stale, excluded from the totals.
    write_rank(clean_fleet, "PP1_TP0", disk=disk_block(), ts=time.time() - 3600)
    # A rank whose file is not JSON at all.
    (clean_fleet / "rank-PP1_TP1.json").write_text("{ not json", encoding="utf-8")
    fill = disk_store.disk_fill()
    assert fill["ranks_reported"] == 2, fill["ranks_reported"]
    stale = {row["rank"]: row for row in fill["ranks_stale"]}
    assert "PP1_TP0" in stale, fill["ranks_stale"]
    assert stale["PP1_TP0"]["file_age_seconds"] > 3000, stale["PP1_TP0"]
    unusable = {row["rank"] for row in fill["ranks_unusable"]}
    assert "PP1_TP1" in unusable, fill["ranks_unusable"]
    # Only the acting rank evicted, and it is one of the two FRESH files: the stale and
    # the unusable rank contribute nothing to a total, by name.
    assert fill["totals"]["evictions"] == 2106, fill["totals"]


# ===========================================================================
# WHO DID THE WORK: the cross-process verdict
# ===========================================================================

def test_the_verdict_shows_one_rank_acting_and_three_declining(clean_fleet):
    """The 0.2.30 fix, WATCHED rather than assumed.

    Four ranks each defending the whole ceiling is the defect; one rank's pass shedding
    while three count a decline is the fix. Both states are visible in this block, and
    only because each rank publishes its own share and its own decline count.
    """
    healthy_fleet(clean_fleet)      # declines 0 / 6 / 12 / 18, evictions in rank 1 only
    verdict = disk_store.disk_fill()["cross_process_verdict"]
    assert verdict["verdict"] == "shared_share", verdict
    assert verdict["n_effective"] == 4, verdict
    assert verdict["share_bytes"] == CEILING // 4, verdict
    assert verdict["ranks_with_evictions"] == 1, verdict
    assert verdict["ranks_acting"] == ["PP0_TP0"], verdict
    assert verdict["ranks_declining"] == 3, verdict
    assert verdict["declines_total"] == 36, verdict
    assert "0.2.30 fix" in verdict["note"], verdict["note"]


def test_the_verdict_says_solo_when_a_rank_cannot_see_a_sibling(clean_fleet):
    """Divisor 1 everywhere: correct on a single-rank boot, and the fail-safe on a
    multi-rank boot whose ranks cannot see each other. The block says which it is."""
    write_rank(clean_fleet, "PP0_TP0",
               disk=disk_block(ceiling_share_bytes=CEILING, ceiling_divisor=1),
               restore=restore_block())
    verdict = disk_store.disk_fill()["cross_process_verdict"]
    assert verdict["verdict"] == "single_rank_share", verdict
    assert verdict["n_effective"] == 1, verdict
    assert "fail-safe" in verdict["note"], verdict["note"]


def test_the_verdict_names_a_disagreement_between_the_ranks(clean_fleet):
    """One store, two budgets: a rank on another build or another ceiling is a fault."""
    write_rank(clean_fleet, "PP0_TP0", disk=disk_block(), restore=restore_block())
    write_rank(clean_fleet, "PP0_TP1",
               disk=disk_block(ceiling_share_bytes=CEILING // 2),
               restore=restore_block())
    verdict = disk_store.disk_fill()["cross_process_verdict"]
    assert verdict["verdict"] == "ranks_disagree_on_the_share", verdict
    assert verdict["share_bytes_values"] == [CEILING // 4, CEILING // 2], verdict


def test_no_ranks_reports_no_verdict_rather_than_an_agreeing_one(clean_fleet):
    """An empty directory is not evidence that the fleet agrees."""
    verdict = disk_store.disk_fill()["cross_process_verdict"]
    assert verdict["verdict"] == "no_ranks_reported", verdict
    assert verdict["declines_total"] == 0


# ===========================================================================
# WHOSE COUNTERS THESE ARE: the label lives inside the block
# ===========================================================================

def test_the_per_process_note_lives_inside_the_block(clean_fleet):
    """A reader who never leaves `disk_fill` is told what the FLAT counters mean.

    The mistake this prevents was made twice on one night on a live box, which is why
    the sentence is inside the block rather than in a README nobody opens at 02:00.
    """
    healthy_fleet(clean_fleet)
    fill = disk_store.disk_fill()
    note = fill["per_process_note"]
    assert "THIS PROCESS'S OWN" in note and "zero by construction" in note, note
    assert fill["scope"].startswith("fleet-wide"), fill["scope"]
    assert fill["totals"]["evictions"] == 2106      # one acting rank, three decliners
    assert fill["totals"]["files_written"] == 4 * 92
    assert fill["totals"]["spilled"] == fill["totals"]["files_written"]
    # The flat, per-process counters are untouched and still zero here, which is the
    # exact combination the note exists to make readable.
    assert disk_store.counters()["spill_queued"] == 0
    assert disk_store.disk_fill()["totals"]["spilled"] == 4 * 92

def test_every_per_rank_field_the_operator_asked_for_is_present(clean_fleet):
    """The per-rank entry's shape: the counters, the age, and the share verdict."""
    healthy_fleet(clean_fleet)
    entry = disk_store.disk_fill()["per_rank"][0]
    for field in ("rank", "file_age_seconds", "files_written", "bytes_written",
                  "evictions", "journal_removals", "recency_promotions",
                  "recency_promotions_throttled", "drops", "queue_depth",
                  "declined_because_already_under", "ceiling_share_bytes",
                  "share_divisor", "estimated_bytes", "walk_basis",
                  "writer_thread_alive"):
        assert field in entry, (field, sorted(entry))
    assert entry["files_written"] == 92
    assert entry["journal_removals"] == 2106
    # The read-side counters live one section DOWN in the rank file, in
    # `disk_restore.readers`: the restore layer owns the reader threads and the touch.
    assert entry["recency_promotions"] == 4100, entry
    assert entry["recency_promotions_throttled"] == 585, entry
    assert entry["file_age_seconds"] < 60, entry


def test_a_rank_file_without_the_restore_section_reports_nulls_not_zeroes(
        clean_fleet):
    """Absent is not zero. A rank that never contributed the restore section has no
    touch clock to report, and publishing 0 would claim promotion is not running.
    """
    write_rank(clean_fleet, "PP0_TP0", disk=disk_block())     # no `disk_restore`
    entry = disk_store.disk_fill()["per_rank"][0]
    assert entry["recency_promotions"] is None, entry
    assert entry["recency_promotions_throttled"] is None, entry
    assert entry["files_written"] == 92


# ===========================================================================
# SIZE DISCIPLINE: a polled block carries counters, not inventories
# ===========================================================================

def test_the_block_is_bounded_however_many_ranks_report(clean_fleet):
    """24 rank files, one capped entry list, a bounded serialised size.

    `/snapshot` is polled every few seconds by a dashboard. Rank entries only, capped at
    `DISK_FILL_RANK_CAP`, with the truncation said out loud.
    """
    ranks = tuple(f"PP{pp}_TP{tp}" for pp in range(6) for tp in range(4))
    healthy_fleet(clean_fleet, ranks=ranks)
    fill = disk_store.disk_fill()
    # The COUNT is the fleet; the ENTRY LIST is the cap. A block that reported 16 for a
    # 24-rank boot would be the "missing rank silently folded into a low total" bug
    # wearing a different hat, so `ranks_reported` counts and `per_rank_truncated` says
    # the listing is short.
    assert fill["ranks_reported"] == len(ranks), fill["ranks_reported"]
    assert len(fill["per_rank"]) == disk_store.DISK_FILL_RANK_CAP
    assert fill["per_rank_truncated"] is True
    assert size_of(fill) < 64 * 1024, size_of(fill)
    # No inventory: the block names files only where a rank could not be read.
    assert "files" not in fill and "samples" not in fill, sorted(fill)
    for entry in fill["per_rank"]:
        assert all(not isinstance(value, (list, dict)) for value in entry.values()), \
            entry


# ===========================================================================
# THE CACHE: one parse per second, and none when nothing changed
# ===========================================================================

def test_the_cache_reparses_when_a_rank_file_changes(clean_fleet):
    """A signature cache, not a clock cache: a rank that wrote is read NOW.

    The cache is `rank_rollup`'s (the parse and the file reads), which `disk_fill`
    derives from - so identity is asserted on the rollup and the EFFECT is asserted on
    the fill. An unchanged fleet costs one parse; a fleet that just wrote costs a new
    one on the same tick, which a clock-based TTL would have got wrong.
    """
    healthy_fleet(clean_fleet)
    first = disk_store.rank_rollup()
    assert disk_store.rank_rollup() is first, "an unchanged fleet must cost one parse"
    assert disk_store.disk_fill()["used_bytes"] == USED
    path = clean_fleet / "rank-PP0_TP0.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["disk"]["estimated_bytes"] = USED + 1024 ** 3
    path.write_text(json.dumps(document), encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 1))
    assert disk_store.rank_rollup() is not first
    assert disk_store.disk_fill()["used_bytes"] == USED + 1024 ** 3


# ===========================================================================
# ADDITIVE: the block joins the snapshot, and changes nothing in it
# ===========================================================================

def test_the_block_is_published_in_the_snapshot_and_changes_no_other_key(clean_fleet):
    """/snapshot grows ONE key. Every key that was there before is still there."""
    healthy_fleet(clean_fleet)
    sections = snapshot.dev_state_sections()
    assert "disk_fill" in sections
    for key in ("disk_spill", "disk_store", "disk_restore", "blockdump",
                "disk_spill_armed", "disk_store_installed"):
        assert key in sections, key
    assert "error" not in sections["disk_fill"], sections["disk_fill"]
    assert sections["disk_fill"]["used_bytes"] == USED
    # The per-process sections beside it are unchanged and still say what they said.
    assert sections["disk_spill"]["scope"], "the per-process label must stay"
    assert sections["disk_spill"]["spilled"] == 0
    # `disk_store` is the block that READ as zeros during the incident, so it carries
    # the same label: a flat `spill_queued: 0` on the API process is not a fact about
    # the tier, and the sentence that says so must live in the payload.
    store_scope = sections["disk_store"]["scope"]
    assert store_scope.startswith("PER-PROCESS"), store_scope
    assert "disk_fill" in store_scope, store_scope


def test_the_fill_never_raises_even_when_the_rollup_does(clean_fleet, monkeypatch):
    """A polled door answers 200 with nulls and a reason. Never a 500."""
    def explode(*_a, **_k):
        raise RuntimeError("the rank reader fell over")
    monkeypatch.setattr(disk_store, "rank_rollup", explode)
    fill = disk_store.disk_fill()
    assert fill["available"] is False
    assert fill["read_basis"] == "fill_failed", fill["read_basis"]
    assert "RuntimeError" in fill["reason"], fill["reason"]
    # Same key set as a healthy block: a degraded answer cannot drift from a good one.
    healthy_fleet(clean_fleet)
    monkeypatch.undo()
    disk_store.reset_rank_rollup_cache_for_tests()
    assert set(fill) == set(disk_store.disk_fill())
    # ... and the block `/status` falls back to when the module call itself raised is
    # built from the SAME skeleton, so `fill["ceiling_bytes"]` cannot KeyError on the
    # one day the read failed. That is why the fallback exists as a function.
    unavailable = disk_store.disk_fill_unavailable("ImportError")
    assert set(unavailable) == set(fill), set(unavailable) ^ set(fill)
    assert unavailable["available"] is False
    assert unavailable["used_bytes"] is None
    assert unavailable["basis"] == "fill_failed"
    assert "ImportError" in unavailable["reason"]
