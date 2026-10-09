r"""PER-REQUEST DISK-ADOPTION ATTRIBUTION (0.2.61): the ring, its cap, its shape.

`_mark_adopted()` has always known "this request adopted 89,600 tokens from disk" and
thrown the request identity away the moment it added the number to the boot-scoped
`adopted_tokens`. A dashboard that shows a TABLE OF REQUESTS cannot fill its
"from disk" column from a boot total, so the number now goes into a bounded ring that
the snapshot payload publishes. These tests are the contract for that ring:

1. THE CAP IS A CAP AND THE DROPS ARE COUNTED. 256 records, oldest recycled,
   `adoptions_dropped` moving one-for-one with the recycles, and `seq` staying
   monotone so a reader can see the gap instead of assuming a contiguous history.
2. A RECORD CONTAINS THE ATTRIBUTION AND ITS PROVENANCE: the tokens, the block/span
   counts, the boot `epoch`, both clocks, and which process/rank wrote it.
3. NOTHING ADOPTED ANSWERS AN EMPTY LIST PLUS THE COUNTERS - never a fabricated row
   with `tokens: 0`, which is the difference between "nobody adopted anything" and
   "every request came from disk and took nothing".
4. A BROKEN WRITE NEVER REACHES THE ENGINE. The writer runs on the scheduler step path
   inside `_mark_adopted()`; an exception there has killed this engine before, and one
   thing we learned the hard way is never to wait or raise inside the step thread. A
   record that cannot be built is counted (`adoption_record_errors`) and the boot's
   aggregate counters still move.

Also pinned: the deeper ring is reachable through the `disk` verb's `adoptions` op
(the polled payload carries only the newest 48 rows), the new numbers all carry units,
the published window stays inside its byte budget, and `snapshot.py`'s module docstring
- the wire contract the reader in `llm-api/kv_tier_diagnostics.py` is written against -
names every field this ring publishes.
"""

from __future__ import annotations

import json

import pytest

from kvtier_dev_plugin import disk_restore, disk_store, snapshot


@pytest.fixture(autouse=True)
def reset_restore_state():
    """The ring is process-wide boot state, exactly like `_loads`."""
    disk_restore.reset_state_for_tests()
    yield
    disk_restore.reset_state_for_tests()


def _adopt(request_id, tokens, *, spans=2, groups=3, source="bounded"):
    """One ledger row through the ONLY door that records an adoption."""
    disk_restore._loads[request_id] = {
        "request_id": request_id,
        "tokens": tokens,
        "cpu_block_ids": list(range(spans * groups)),
        "identities": [{"group": g, "span": s, "key": f"{'a' * 16}",
                        "cpu_block_id": (s * groups) + g}
                       for s in range(spans) for g in range(groups)],
        "offered_mono": 1.0,
        "deadline_mono": 2.0,
        "adopted": False,
        "emitted": True,
        "gpu_block_ids": [],
        "restore_source": source,
    }


def _adopt_one(request_id, tokens, **kwargs):
    _adopt(request_id, tokens, **kwargs)
    disk_restore._mark_adopted()


# --- 1: the cap, and the drops it owes an account for -----------------------

def test_the_ring_holds_exactly_its_depth_and_counts_every_drop():
    total = disk_restore.MAX_ADOPTION_RECORDS + 44
    for index in range(total):
        _adopt_one(f"req-{index}", 1000 + index)
    assert len(disk_restore._adoptions) == disk_restore.MAX_ADOPTION_RECORDS
    assert disk_restore._adoptions.maxlen == disk_restore.MAX_ADOPTION_RECORDS
    assert disk_restore._state["adoptions"] == total
    assert disk_restore._state["adoptions_dropped"] == 44


def test_the_oldest_rows_leave_and_the_newest_are_never_the_ones_dropped():
    total = disk_restore.MAX_ADOPTION_RECORDS + 10
    for index in range(total):
        _adopt_one(f"req-{index}", 1000 + index)
    rows = disk_restore.disk_adoptions(limit=disk_restore.MAX_ADOPTION_RECORDS)
    assert rows[0]["request_id"] == f"req-{total - 1}", "newest FIRST"
    assert rows[0]["seq"] == total
    # Exactly the 10 oldest were recycled, and nothing newer.
    surviving = {row["request_id"] for row in rows}
    assert {f"req-{index}" for index in range(10)}.isdisjoint(surviving)
    assert "req-10" in surviving and f"req-{total - 1}" in surviving
    # `seq` runs over records EVER written, so the gap is visible rather than implied.
    assert [row["seq"] for row in rows[:5]] == [total, total - 1, total - 2,
                                                total - 3, total - 4]


def test_an_adopted_request_is_recorded_once_however_often_the_mark_runs():
    _adopt_one("req-once", 89600)
    disk_restore._mark_adopted()
    disk_restore._mark_adopted()
    assert disk_restore._state["adoptions"] == 1
    assert disk_restore._state["adopted"] == 1
    assert disk_restore._state["adopted_tokens"] == 89600


def test_an_offer_that_was_never_adopted_is_never_recorded():
    disk_restore._loads["req-offered-only"] = {
        "request_id": "req-offered-only", "tokens": 3200, "identities": [],
        "cpu_block_ids": [], "adopted": False, "emitted": False,
        "gpu_block_ids": []}
    disk_restore._mark_adopted()
    assert disk_restore.disk_adoptions() == []
    assert disk_restore._state["adopted"] == 0


# --- 2: what a record owes the reader ---------------------------------------

def test_a_record_carries_the_attribution_and_its_provenance():
    _adopt_one("0b9c1f2e-0000-4000-8000-000000000000", 12800, spans=4, groups=3)
    row = disk_restore.disk_adoptions()[0]
    assert set(row) == {"seq", "request_id", "tokens", "blocks", "spans", "epoch",
                        "ts", "mono", "rank", "role", "restore_source"}
    assert row["request_id"] == "0b9c1f2e-0000-4000-8000-000000000000"
    # THE FIELD THIS RELEASE EXISTS FOR.
    assert row["tokens"] == 12800
    assert row["blocks"] == 12, "one entry per group per span"
    assert row["spans"] == 4
    assert row["epoch"] == snapshot.epoch(), "a row must name the boot that knew it"
    assert row["restore_source"] == "bounded"
    assert isinstance(row["ts"], float) and row["ts"] > 1_600_000_000
    assert isinstance(row["mono"], float) and row["mono"] > 0
    # A restore is adopted by the SCHEDULER, which has no TP rank: the honest answer
    # is a null rank beside a role that says which process this is.
    assert row["rank"] is None
    assert row["role"] == disk_restore.role_here()


def test_a_request_id_that_fits_is_not_truncated_because_it_is_a_join_key():
    # Measured: the longest `requests.id` in 88,780 proxy rows is 36 (UUIDv4), so the
    # 48-char clip never bites a real id and a reader can join on it exactly.
    identifier = "0" * 36
    _adopt_one(identifier, 3200)
    assert disk_restore.disk_adoptions()[0]["request_id"] == identifier


# --- 3: the empty state, which is the state a reader sees first -------------

def test_nothing_adopted_answers_an_empty_list_and_counters_never_a_row():
    section = disk_restore.scheduler_section()
    assert section["recent_adoptions"] == []
    assert section["adoptions_recorded"] == 0
    assert section["adoptions_dropped"] == 0
    assert section["adoption_record_errors"] == 0
    assert section["adopted"] == 0 and section["adopted_tokens"] == 0
    # The two sentences a UI must surface rather than paraphrase, so "no rows" can
    # never be drawn as "0 tokens from disk".
    assert "PERISHABLE" in section["adoption_perishable"]
    assert "NEVER MEANS ZERO" in section["adoption_perishable"]
    assert "one record per REQUEST" in section["adoption_meaning"]
    report = disk_restore.disk_adoption_report()
    assert report["records"] == [] and report["ring_written"] == 0


def test_the_reports_key_set_does_not_move_between_an_empty_and_a_full_ring():
    empty = set(disk_restore.disk_adoption_report())
    _adopt_one("req-keyset", 3200)
    assert set(disk_restore.disk_adoption_report()) == empty
    # ... and the section's adoption keys do not move either.
    empty_keys = {key for key in disk_restore.scheduler_section()
                  if "adoption" in key or key == "recent_adoptions"}
    _adopt_one("req-keyset-2", 6400)
    assert {key for key in disk_restore.scheduler_section()
            if "adoption" in key or key == "recent_adoptions"} == empty_keys


def test_a_request_absent_from_the_ring_is_not_evidence_of_a_zero():
    _adopt_one("req-present", 89600)
    rows = disk_restore.disk_adoptions()
    assert [row["request_id"] for row in rows] == ["req-present"]
    report = disk_restore.disk_adoption_report()
    # The ONLY basis a reader may use for a zero claim: the exact boot totals, plus
    # the counter that says whether rows were lost. Both are present even when the
    # row for the request in question is not.
    assert report["adopted"] == 1 and report["adopted_tokens"] == 89600
    assert report["ring_dropped"] == 0
    assert "PERISHABLE" in report["perishable"]


# --- 4: a broken write never reaches the engine -----------------------------

def test_a_writer_that_raises_is_counted_and_never_reaches_the_caller(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("the boot id is on strike")

    monkeypatch.setattr(snapshot, "epoch", explode)
    disk_restore._note_disk_adoption({"request_id": "req-bad", "tokens": 100,
                                      "identities": [], "cpu_block_ids": []})
    assert disk_restore._state["adoption_record_errors"] == 1
    assert len(disk_restore._adoptions) == 0, "no half-built row survives"


def test_a_ledger_row_that_is_not_a_mapping_is_refused_not_raised():
    for bad in (None, "a string", 7, ["a", "list"]):
        disk_restore._note_disk_adoption(bad)   # must not raise
    assert disk_restore._state["adoption_record_errors"] == 4
    assert len(disk_restore._adoptions) == 0


def test_a_malformed_ledger_row_still_records_its_tokens_and_no_lies():
    disk_restore._note_disk_adoption({
        "request_id": 12345,                     # not a string
        "tokens": "3200",                        # a numeric string
        "identities": ["not", "a", "dict"],      # uncountable spans
        "cpu_block_ids": [1, 2, 3],
        "restore_source": None})
    row = disk_restore.disk_adoptions()[0]
    assert row["tokens"] == 3200
    assert row["request_id"] == "12345"
    assert row["spans"] == 0
    assert row["blocks"] == 3, "identities unreadable, so the slot count is counted"
    assert row["restore_source"] is None
    assert disk_restore._state["adoption_record_errors"] == 0


def test_a_broken_ring_cannot_cost_the_boot_its_aggregate_counters(monkeypatch):
    """THE ORDERING THAT MATTERS: `adopted`/`adopted_tokens` move BEFORE the ring.

    A ring that cannot take a row is allowed to lose the per-request row and nothing
    else. If the ring's write ever moved above the counter updates, a diagnostic would
    be silently rewriting the boot's adoption totals - the exact class of bug the
    `restorations`/`reads_applied` mismatch of 2026-10-05 was.
    """
    monkeypatch.setattr(disk_restore, "_adoptions", _FakeRing(broken=True))
    _adopt_one("req-counter-safe", 89600)
    assert disk_restore._state["adopted"] == 1
    assert disk_restore._state["adopted_tokens"] == 89600
    assert disk_restore._state["adoption_record_errors"] == 1
    assert disk_restore.disk_adoptions() == []


class _FakeRing:
    """A stand-in for the deque: the ring's own failure modes, without a C method.

    `deque.append` is a read-only C slot, so it cannot be monkeypatched; this carries
    the same surface the writer touches (`len`, `append`, `popleft`, `clear`,
    `reversed`) and raises where a test needs it to.
    """

    maxlen = disk_restore.MAX_ADOPTION_RECORDS

    def __init__(self, *, broken: bool = False, unreading: bool = False):
        self.rows: list = []
        self.broken = broken
        self.unreading = unreading

    def __len__(self):
        return len(self.rows)

    def append(self, row):
        if self.broken:
            raise RuntimeError("the deque is full of glass")
        self.rows.append(row)

    def popleft(self):
        self.rows.pop(0)

    def clear(self):
        self.rows = []

    def __reversed__(self):
        if self.unreading:
            raise RuntimeError("the ring will not be read")
        return reversed(self.rows)


def test_the_ring_read_is_contained_when_the_ring_itself_is_broken(monkeypatch):
    monkeypatch.setattr(disk_restore, "_adoptions", _FakeRing(unreading=True))
    assert disk_restore.disk_adoptions() == []
    report = disk_restore.disk_adoption_report()
    assert report["records"] == []
    # The degraded report still carries the full key set: a reader must never have to
    # branch on which fields a broken answer happens to include.
    assert {"records", "ring_written", "ring_dropped", "adopted",
            "adopted_tokens", "perishable"} <= set(report)


def test_a_bad_limit_is_a_default_never_an_exception():
    _adopt_one("req-limit", 3200)
    for bad in ("many", -1, 0, True, None, {}, 1.5):
        assert isinstance(disk_restore.disk_adoptions(bad), list)


# --- the published window, and what it costs --------------------------------

def test_the_payload_carries_the_newest_window_and_the_exact_counters():
    total = disk_restore.MAX_ADOPTION_RECORDS + 44
    for index in range(total):
        _adopt_one(f"req-{index}", 1000 + index)
    section = disk_restore.scheduler_section()
    assert len(section["recent_adoptions"]) == disk_restore.MAX_ADOPTIONS_PUBLISHED
    assert section["recent_adoptions"][0]["seq"] == total
    assert section["adoptions_recorded"] == total
    assert section["adoptions_dropped"] == 44
    assert section["adoption_ring_depth"] == disk_restore.MAX_ADOPTION_RECORDS
    assert section["adoptions_published"] == disk_restore.MAX_ADOPTIONS_PUBLISHED


def test_the_published_window_stays_inside_its_stated_byte_budget():
    """The cap is justified in bytes as well as in records.

    48 rows measured 11,371 B against a 51,839 B snapshot document (+21.9%). The
    budget below is that figure with room for a longer id and a future field: a
    payload a dashboard polls every few seconds must not creep, and the only way to
    keep it from creeping is a number a test can fail.
    """
    for index in range(disk_restore.MAX_ADOPTIONS_PUBLISHED):
        _adopt_one(f"{'c' * 36}-{index}", 89600)
    rows = disk_restore.scheduler_section()["recent_adoptions"]
    assert len(rows) == disk_restore.MAX_ADOPTIONS_PUBLISHED
    assert len(json.dumps(rows)) < 16 * 1024


def test_every_new_number_arrives_with_a_unit():
    counters = disk_restore.counters()
    units = disk_restore.counters_view(dict(counters))["units"]
    for key in ("adoptions", "adoptions_dropped", "adoption_record_errors"):
        assert key in counters, key
        assert units.get(key), f"{key} reaches the payload with no unit"
    section = disk_restore.scheduler_section()
    for key in ("adoptions_recorded", "adoptions_dropped",
                "adoption_record_errors", "adoption_ring_depth",
                "adoptions_published"):
        assert key in section, key
        assert section["units"].get(key), f"{key} is a bare number in the payload"


# --- the deeper ring's door -------------------------------------------------

def test_the_adoptions_op_reaches_the_deeper_ring_the_payload_omits():
    total = disk_restore.MAX_ADOPTION_RECORDS + 10
    for index in range(total):
        _adopt_one(f"req-{index}", 1000 + index)
    answer = disk_store.disk_for_engine(None, {"op": "adoptions"})
    assert answer["ok"] is True
    assert len(answer["records"]) == disk_restore.MAX_ADOPTIONS_PUBLISHED
    deep = disk_store.disk_for_engine(None, {"op": "adoptions", "limit": 256})
    assert len(deep["records"]) == disk_restore.MAX_ADOPTION_RECORDS
    assert deep["ring_dropped"] == 10
    assert deep["ring_written"] == total
    assert deep["adopted"] == total


def test_the_new_op_is_in_every_list_that_has_to_agree():
    from kvtier_dev_plugin import config, endpoint
    assert "adoptions" in disk_store.DISK_OPS
    assert "adoptions" in disk_store._DISK_OPS
    # Read-only: a safe method must be able to reach it, and a mutating list entry
    # would turn a GET into a 405.
    assert "adoptions" not in disk_store.MUTATING_DISK_OPS
    assert "adoptions" in endpoint.ARG_SPECS[config.ROUTE_DISK]["op"]["values"]
    assert set(endpoint.ARG_SPECS[config.ROUTE_DISK]["op"]["values"]) == set(
        disk_store.DISK_OPS)


# --- the wire contract ------------------------------------------------------

def test_the_wire_contract_names_every_field_this_ring_publishes():
    """`snapshot.py`'s module docstring IS the contract the other repo reads.

    A field added to the payload without a sentence in that docstring is a field the
    reader in `llm-api/kv_tier_diagnostics.py` will have to guess the meaning of -
    which is how an absent row becomes a fabricated zero.
    """
    contract = snapshot.__doc__ or ""
    for name in ("recent_adoptions", "adoption_meaning", "adoption_perishable",
                 "adoption_record_errors", "adoption_ring_depth",
                 "adoptions_published", "adoptions_dropped", "adoptions_recorded",
                 "request_id", "tokens", "blocks", "spans", "epoch", "ts", "mono",
                 "rank", "role", "restore_source", "adoption_record_errors",
                 "MAX_ADOPTIONS_PUBLISHED", "MAX_ADOPTION_RECORDS"):
        assert name in contract, f"{name} is published but not in the contract"
    # The two readings a UI gets wrong if the contract is vague.
    assert "PERISHABLE" in contract
    assert "ABSENCE IS NOT ZERO" in contract
    assert "not_reported" in contract, "an older writer must read as not_reported"
    assert "0.2.61" in contract, "the field's arrival version must be named"
