r"""A PROOF GAP IS NOT A FAULT: per-request read evidence, and what fuses a span.

THE DEFECT THIS SUITE EXISTS FOR (0.2.60). `note_reported` asked a per-request
question - "was a disk read applied behind THIS finished load?" - with a boot-wide
predicate (`applied_reads_evidence()["reads_applied"] >= 1`, "has ANY read EVER been
applied in this process?"), and then handed the answer to `suppress_load(span=...)`,
which fuses the SPAN for the rest of the boot. Two consequences, both live:

1. THE FIRST RESTORE OF EVERY BOOT IS FUSED. The counter is still 0 (or the worker's
   rank file has not been read yet), so a load that was adopted and finished
   legitimately is declared fake. The span key is the run's FIRST block, so the NEXT
   turn of the same conversation - a different request id, the same first block - is
   refused before any index read, and on a multi-group model the engine resets
   `num_computed_tokens` to 0: a 241,740-token prompt recomputed in 66.3 s. The engine
   line was `no further disk offer for request ... the load was reported finished with
   no applied disk read behind it` - 34 occurrences in 5 days.
2. Once any read has landed the guard can NEVER fire again, so it is inert for the
   failures it was written to catch.

THE RULE THIS SUITE PINS:

* (a) A PROOF GAP COSTS THE REQUEST, NEVER THE SPAN. A finished load with no evidence
  behind it is still `no_read` and still black-holes ITS OWN request (the loop-killer
  that bounded 123,474 offers in 3.9 min needs the request key only, because the fork
  reschedules the same request), but a different request wanting the same span is
  offered normally.
* (b) A STRUCTURAL failure - a data-kind load failure, or `group_hit_truncated` - still
  fuses the span. That is evidence about the BYTES, not about our knowledge of them.
* (c) THE LOOP STAYS DEAD. A request that comes back with no progress gets no third
  offer whether its report was a fault or a proof gap.
* (d) A DELIVERED disk->GPU write is applied evidence. `disk_gpu_write`'s delivery mode
  is on in the live launcher, and its counters were consulted by nothing, so a boot
  that only ever delivered looked like a boot that never read.
* (e) `_suppressed_spans` is BOUNDED, like `_suppressed`.
* AND THE CHANNEL: the apply loop runs in the WORKER process, so per-request evidence
  reaches the scheduler through the rank file, and this suite drives that round trip
  for real (`test_the_rank_file_carries_the_applied_slots_...`).

Harness is the sibling suite's: a `tmp_path` store, ctypes-backed CPU slots, the real
reader thread settling on a bounded poll.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import test_disk_restore as base

from kvtier_dev_plugin import (
    config,
    disk_gpu_write,
    disk_restore,
    worker_diag,
)

armed = base.armed                          # noqa: F811 - the gate open, switch up
allow_tmp_root = base.allow_tmp_root        # noqa: F811 - autouse
fresh = base.fresh                          # noqa: F811 - autouse: a fresh boot
fill = base.fill
offer = base.offer
settle = base.settle
bind = base.bind
disk_meta = base.disk_meta
merged_cpu = base.merged_cpu
BLOCK = base.BLOCK
CACHEABLE = base.CACHEABLE
WIDTH = base.WIDTH
Manager = base.Manager
Req = base.Req
Meta = base.Meta
Worker = base.Worker


def _ship(manager, request_id="req-1"):
    """Offer, then EMIT the load's paths, so the load counts as ADOPTED.

    Adoption is measured from emission (`_mark_adopted` reads `record["emitted"]`,
    which only `_build_load_paths` sets), and a load that never reached a step is
    released by a different branch of `note_reported` than the one under test.
    """
    offer(manager, Req(request_id=request_id, blocks=1))
    slots = merged_cpu(manager, request_id)
    disk_restore.build_load_paths(manager, Meta(
        load_event=7, cpu=slots, gpu=list(range(len(slots))),
        reqs={7: [request_id]}))
    record = disk_restore._loads[request_id]
    assert record["adopted"] is True, "the harness must reach the adopted shape"
    return record


def _finished(*request_ids):
    return base.fakes_debug.Obj(name="KVConnectorOutput",
                                finished_recving=set(request_ids),
                                finished_sending=set())


def _diag_dir() -> Path:
    return Path(os.environ[config.ENV_DIAG_DIR])


def _write_rank_file(section: dict, rank="PP0_TP0") -> None:
    """One worker's rank file, fresh, carrying whatever section the test names."""
    directory = _diag_dir()
    directory.mkdir(parents=True, exist_ok=True)
    document = {"version": 1, "epoch": "aabbccdd",
                "ts": round(time.time(), 6), "role": "worker", "rank": rank,
                "arena": {}, "counters": {},
                "writer": {"epoch_scope": "process"},
                "disk_restore": section}
    (directory / f"rank-{rank}.json").write_text(json.dumps(document))


# --- (a) THE REGRESSION ------------------------------------------------------

def test_a_proof_gap_on_the_first_restore_of_a_boot_does_not_fuse_the_span(armed):
    """THE 66-SECOND REGRESSION: a fresh boot, one legitimate load, no read yet.

    Every counter that `applied_reads_evidence()` reads is 0 - the scheduler never
    reads a file itself, and no rank file has been read yet - so the old predicate
    answered "no read has EVER been applied" and `suppress_load(span=...)` fused the
    span for the whole boot. The span key is the run's first block, so the next turn of
    the same conversation (a DIFFERENT request id, the same first block) was refused
    before any index read and recomputed its whole prompt.

    A proof gap must cost the request that showed it and nothing else.
    """
    fill(armed, blocks=1)
    assert disk_restore.applied_reads_evidence()["reads_applied"] == 0, \
        "a fresh boot has no applied-read evidence at all"
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")

    report = disk_restore.note_reported(manager, _finished("req-1"))
    assert report["no_read"] == 1, "the gap is still counted honestly"
    assert report["restorations"] == 0

    # THE PROPERTY THAT COST A USER 66.3 s: the NEXT turn of the same conversation is a
    # different request id on the SAME first block, and it must still be offered.
    again = offer(Manager(reqs_to_load={"req-2": object()}),
                  Req(request_id="req-2", blocks=1))
    assert again == (BLOCK, True), \
        f"the next turn of the same span got no offer: {again}"
    assert disk_restore.counters()["refused"].get("suppressed_after_failure", 0) == 0
    assert disk_restore._suppressed_spans == {}, \
        "a proof gap is not evidence about the span: it must not be fused"
    assert disk_restore.counters()["suppressed_spans"] == 0


def test_the_request_itself_is_still_black_holed_by_its_own_proof_gap(armed):
    """The loop-killer survives: the REQUEST that reported the gap is fused.

    Dropping the span is not dropping the fuse. The fork reschedules the SAME request,
    so the request key alone is what bounds the offer loop that ran 123,474 times in
    3.9 minutes on 2026-10-05.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")
    disk_restore.note_reported(manager, _finished("req-1"))
    assert "req-1" in disk_restore._suppressed

    before = disk_restore.counters()["slots_allocated"]
    again = offer(Manager(reqs_to_load={"req-1": object()}),
                  Req(request_id="req-1", blocks=1))
    assert again == (0, False), "a re-offered failure is the loop itself"
    counters = disk_restore.counters()
    assert counters["suppressed_hits"] == 1
    assert counters["refused"]["suppressed_after_failure"] == 1
    assert counters["slots_allocated"] == before, \
        "a suppressed request must not allocate a single slot"


# --- (c) THE LOOP STAYS BOUNDED ---------------------------------------------

def test_a_request_that_returns_twice_with_no_progress_gets_no_third_offer(armed):
    """The 0.2.32 property, re-pinned on the NEW proof-gap path.

    Offer, report (with no evidence behind it), come back at the same computed tokens:
    two askings, one offer, and the third asking is refused too. A proof gap grants no
    race re-offer, because nothing here says the loss was a race.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")
    disk_restore.note_reported(manager, _finished("req-1"))

    for _ in range(3):
        assert offer(Manager(reqs_to_load={"req-1": object()}),
                     Req(request_id="req-1", blocks=1)) == (0, False)
    counters = disk_restore.counters()
    assert counters["offers"] == 1, \
        f"one load produced {counters['offers']} offers; the loop is back"
    assert counters["refused"]["suppressed_after_failure"] == 3
    assert disk_restore.counters()["races_not_fused"] == 0, \
        "a proof gap says nothing about a race, so it grants no re-offer"


# --- (b) A STRUCTURAL FAILURE STILL FUSES -----------------------------------

def test_a_data_kind_load_failure_still_fuses_the_span(armed):
    """Structural evidence keeps its fuse: bad bytes are a fact about the SPAN.

    A `size_mismatch` filed at the apply point (`where="apply"`) is a DATA kind, so
    `_fuse_or_spare` fuses - request AND span - and a different request wanting the
    same span is refused. This is the half of the 0.2.32 partition that must NOT move
    when the proof-gap fuse moves.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")
    disk_restore._note_load_failure("size_mismatch", where="apply",
                                    reason="the file is 4095 B, the slot is 4096 B",
                                    request_id="req-1", job_id=0)
    failed = base.fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                                  finished_sending=set(), failed_recving={"req-1"},
                                  invalid_block_ids=set())
    report = disk_restore.note_reported(manager, failed)
    assert report["failures"] == 1
    assert disk_restore._suppressed_spans, \
        "a data-kind failure is structural evidence: the span is implicated"

    other = offer(Manager(reqs_to_load={"req-9": object()}),
                  Req(request_id="req-9", blocks=1))
    assert other == (0, False)
    assert disk_restore.counters()["suppressed_spans"] == 1


def test_group_hit_truncated_still_fuses_the_span(armed):
    """The other structural site: the fork's take is shorter than the run we built.

    `observe_group_take_truncation` refuses with `group_hit_truncated` and fuses the
    span, because the arithmetic of the offered run against the fork's own take is a
    fact about the span, not about our knowledge of a worker. Pinned here so the
    proof-gap change cannot be read as "no span is ever fused again".
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")
    span = disk_restore._span_key_of(disk_restore._loads["req-1"])
    disk_restore.suppress_load("req-1", "group_hit_truncated", span=span)
    assert span in disk_restore._suppressed_spans
    assert offer(Manager(reqs_to_load={"req-9": object()}),
                 Req(request_id="req-9", blocks=1)) == (0, False)


# --- per-request evidence ---------------------------------------------------

def test_the_real_apply_point_records_the_slots_it_applied(armed):
    """The worker window is filled by the real apply loop, not by a test helper.

    `file -> bounce buffer -> memmove -> CPU slot` for three groups, and every slot it
    memmoved into is remembered as an applied slot. This is the site the per-request
    evidence is built from, so it is pinned at the site itself.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    assert bind(worker, metadata)["ok"] is True
    settle(disk_restore._batch_id(11))
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True
    assert report["applied"] == len(CACHEABLE)
    applied = disk_restore.applied_slots()
    assert set(applied) == set(metadata.load_cpu_blocks), \
        "every slot that got bytes is an applied slot, and only those"
    assert disk_restore.counters()["reads_applied"] == len(CACHEABLE)


def test_a_load_whose_slots_were_applied_is_a_restoration_not_a_no_read(armed):
    """The OTHER half of the first-restore story: evidence for THIS load wins.

    Same fresh boot, same single load - but the apply point really did memmove into
    the slots this load owns. The per-request predicate sees that and counts a
    restoration, which the boot-wide predicate could only do by accident.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    record = _ship(manager, "req-1")
    for slot in record["cpu_block_ids"]:
        disk_restore._note_applied_slot(slot)

    report = disk_restore.note_reported(manager, _finished("req-1"))
    assert report["restorations"] == 1, report
    assert report["no_read"] == 0
    assert disk_restore._suppressed == {}
    assert disk_restore._suppressed_spans == {}


def test_an_applied_slot_for_another_load_is_not_evidence_for_this_one(armed):
    """THE INERT GUARD, resurrected: a read for a DIFFERENT load proves nothing here.

    Under the boot-wide predicate this shape could never fire once any read had landed
    - which is consequence 2 of the defect. Per-request attribution makes the original
    lie detectable again: a finished report whose slots nobody filled is `no_read`.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")
    disk_restore._note_applied_slot(999_999)   # somebody else's slot

    report = disk_restore.note_reported(manager, _finished("req-1"))
    assert report["no_read"] == 1, \
        "a read applied for another load is not a read behind THIS one"
    assert report["restorations"] == 0
    assert disk_restore._suppressed_spans == {}, "and still no span fuse"


def test_the_evidence_for_a_load_names_the_source_that_answered(armed):
    """`applied_read_evidence_for` says WHICH channel proved the read.

    The two channels are the apply window in this process and the workers' rank files,
    and a scheduler-only process can only ever have the second one - so the source is
    the difference between "no read" and "this process cannot know".
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    record = _ship(manager, "req-1")

    empty = disk_restore.applied_read_evidence_for(record)
    assert empty["applied"] is False
    assert "no applied read" in empty["source"]

    disk_restore._note_applied_slot(record["cpu_block_ids"][0])
    hit = disk_restore.applied_read_evidence_for(record)
    assert hit["applied"] is True
    assert hit["source"] == "this process"
    assert hit["matched"], "the evidence names the slot it matched on"


def test_the_rank_file_carries_the_applied_slots_across_the_process_boundary(armed):
    """THE CHANNEL, end to end, because the apply loop is not in this process.

    `_apply_buffer` runs inside `start_load_kv` - the WORKER - so the scheduler's own
    `reads_applied` is 0 by construction and the only per-request evidence a scheduler
    can have about a worker's apply point is what the rank file carries. A fresh rank
    file naming one of this load's CPU slots must turn the finished report into a
    restoration, in a process that never applied anything itself.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    record = _ship(manager, "req-1")
    assert disk_restore.applied_slots() == [], \
        "this process is the scheduler: it applies nothing"
    _write_rank_file({"installed": True, "armed": True, "worker_fitted": True,
                      "reads_applied": 1,
                      "applied_slots": [int(record["cpu_block_ids"][0])]})

    report = disk_restore.note_reported(manager, _finished("req-1"))
    assert report["restorations"] == 1, report
    assert report["no_read"] == 0
    assert disk_restore._suppressed_spans == {}
    assert disk_restore.counters()["ranks_applied_slots"] >= 1


def test_a_rank_file_slot_for_another_load_does_not_vouch_for_this_one(armed):
    """The channel is per-request in BOTH directions.

    A rank file attesting a slot that belongs to no in-flight load must not rescue a
    finished report whose own slots are unfilled - otherwise the boot-wide flaw has
    just moved into the window.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")
    _write_rank_file({"installed": True, "armed": True, "worker_fitted": True,
                      "reads_applied": 1, "applied_slots": [4242]})

    report = disk_restore.note_reported(manager, _finished("req-1"))
    assert report["no_read"] == 1, report
    assert report["restorations"] == 0
    assert disk_restore._suppressed_spans == {}


def test_a_rank_that_publishes_no_window_falls_back_to_the_boot_wide_counter(armed):
    """MIXED VERSIONS: a worker too old to publish slots leaves no per-request evidence.

    The honest answer to a question nobody can answer is the answer the boot used to
    get - the boot-wide attestation, NAMED as such - and not a wall of `no_read` that
    black-holes every request in a boot whose workers are serving normally. The fallback
    is unreachable the moment a live rank publishes a window (pinned by
    `test_a_rank_file_slot_for_another_load_does_not_vouch_for_this_one`), which is what
    stops it from becoming the defect again.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    _ship(manager, "req-1")
    _write_rank_file({"installed": True, "armed": True, "worker_fitted": True,
                      "reads_applied": 4})      # a pre-0.2.60 worker: no slot window
    # The `armed` fixture stubs `worker_proof`, so the proof read that would have
    # summed that rank's counter into the boot-wide attestation did not run; this is
    # exactly what it would have left behind.
    disk_restore._state["ranks_reads_applied"] = 4
    evidence = disk_restore.applied_read_evidence_for(
        disk_restore._loads["req-1"])
    assert evidence["applied"] is True
    assert "boot-wide read counter" in evidence["source"], evidence["source"]
    report = disk_restore.note_reported(manager, _finished("req-1"))
    assert report["restorations"] == 1
    assert report["no_read"] == 0


def test_the_applied_slot_reprobe_is_throttled(armed, monkeypatch):
    """The completion path reads the rank files at most once a second.

    `note_reported` runs on the scheduler's step path; an unbounded file read there is
    a new way to make the step slow, so the fresh read that closes the rank-file lag is
    throttled and counted.
    """
    fill(armed, blocks=1)
    calls = []
    real = disk_restore._read_applied_slots_from_ranks

    def counting():
        calls.append(len(calls))
        return real()

    monkeypatch.setattr(disk_restore, "_read_applied_slots_from_ranks", counting)
    manager = Manager(reqs_to_load={"req-1": object()})
    record = _ship(manager, "req-1")
    for _ in range(4):
        disk_restore.applied_read_evidence_for(record)
    assert len(calls) == 1, f"{len(calls)} rank reads for 4 asks: not throttled"
    assert disk_restore.counters()["applied_slot_reprobes"] == 1


# --- (d) DELIVERY MODE IS APPLIED EVIDENCE ----------------------------------

def test_a_delivered_write_counts_as_applied_evidence(armed):
    """A disk->GPU delivery is a disk read that reached the engine.

    Delivery mode (`VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY`, on in the live
    launcher) enqueues the host->device copy for entries whose bytes `_apply_buffer`
    already memmoved - so it normally moves together with `reads_applied` and the
    evidence must not double count it. It is consulted anyway, and named, because a
    counter that proves bytes moved and is read by nothing is how a healthy delivery
    boot reads as "nothing has applied a disk read in this boot".
    """
    assert disk_restore.applied_reads_evidence()["reads_applied"] == 0
    disk_gpu_write._state["counters"]["delivery_writes"] = 3
    evidence = disk_restore.applied_reads_evidence()
    assert evidence["reads_applied"] >= 1
    assert "deliver" in evidence["source"], evidence["source"]
    assert "deliver" in disk_restore.applied_reads_evidence()["source"]


def test_a_delivered_write_is_not_counted_twice(armed):
    """The ordinary case: the apply point counted it, so the total is the apply total.

    Delivery is a second copy of bytes that were already memmoved into the CPU slot;
    `reads_applied` is the count of applied ENTRIES, and a delivered entry is one
    applied entry, not two.
    """
    disk_restore._state["reads_applied"] = 7
    disk_gpu_write._state["counters"]["delivery_writes"] = 5
    evidence = disk_restore.applied_reads_evidence()
    assert evidence["reads_applied"] == 7
    assert evidence["source"] == "this process"


# --- (e) BOUNDS -------------------------------------------------------------

def test_suppressed_spans_cannot_grow_without_bound(armed, monkeypatch):
    """`_suppressed_spans` is a fuse, not a database: it is bounded, oldest-first.

    `_suppressed` has always been bounded by `MAX_SUPPRESSED_REQUESTS`; the span table
    was written without any bound at all, so a long boot that lost many spans grew one
    short string per span forever. Forgetting the oldest costs one more offer; leaking
    without bound costs the boot its memory.
    """
    monkeypatch.setattr(disk_restore, "MAX_SUPPRESSED_SPANS", 32)
    for index in range(200):
        disk_restore.suppress_load(f"req-{index}", "a structural loss",
                                   span=f"span-{index}")
    assert len(disk_restore._suppressed_spans) <= 32, \
        f"{len(disk_restore._suppressed_spans)} spans remembered under a cap of 32"
    assert "span-199" in disk_restore._suppressed_spans, \
        "the eviction is oldest-first: the newest fuse must survive"
    assert "span-0" not in disk_restore._suppressed_spans


def test_suppressing_a_request_without_a_span_touches_no_span(armed):
    """The signature's default is the whole change: `span=None` means no span fuse."""
    disk_restore.suppress_load("req-1", "a proof gap")
    assert disk_restore._suppressed_spans == {}
    assert "req-1" in disk_restore._suppressed


def test_the_applied_slot_window_is_bounded(armed, monkeypatch):
    """The worker's applied-slot window is bounded, oldest-first, like every table here.

    It runs on the apply path in the engine's process; an unbounded dict there is a
    leak in the process that must never be the reason a step was slow.
    """
    monkeypatch.setattr(disk_restore, "MAX_APPLIED_SLOTS_TRACKED", 16)
    for slot in range(500):
        disk_restore._note_applied_slot(slot)
    assert len(disk_restore._applied_slots) <= 16
    applied = disk_restore.applied_slots()
    assert 499 in applied and 0 not in applied, "oldest-first eviction"


def test_the_worker_report_still_fits_the_rank_file_cap(armed):
    """The new rank-file key must never cost the WHOLE section.

    `worker_diag` REFUSES a contribution over `MAX_CONTRIBUTION_BYTES`, and a rank file
    with no `disk_restore` section is as useless to the offer gate as a stale one - so
    a key that pushes the section over its cap would close the gate it was added to
    inform. The window is capped, and the section pays for it before it is dropped.
    """
    for slot in range(disk_restore.MAX_APPLIED_SLOTS_TRACKED):
        disk_restore._note_applied_slot(100_000 + slot)
    for index in range(4):
        disk_restore._note_load_failure("read_failed", where="apply",
                                        reason="x" * 150,
                                        request_id="req-" + "y" * 40, job_id=index,
                                        bytes_needed=11_468_800)
    report = disk_restore.worker_report()
    assert "applied_slots" in report
    assert len(report["applied_slots"]) <= disk_restore.MAX_APPLIED_SLOTS_PUBLISHED
    assert len(json.dumps(report)) <= worker_diag.MAX_CONTRIBUTION_BYTES, \
        f"{len(json.dumps(report))} bytes over the rank section cap"


def test_an_old_rank_file_without_the_key_is_still_evidence(armed):
    """A mixed-version fleet: a rank file that predates the key says nothing, not zero.

    The scheduler must not read an absent `applied_slots` as "this rank applied
    nothing" - the same three-valued rule `worker_fitted` already follows, and the
    reason the reader clips rather than guesses.
    """
    _write_rank_file({"installed": True, "armed": True, "worker_fitted": True,
                      "reads_applied": 4})
    assert disk_restore._read_applied_slots_from_ranks() == []
    _write_rank_file({"installed": True, "armed": True, "worker_fitted": True,
                      "applied_slots": [12, "13", -5, 999, True, 1.75, None]})
    slots = disk_restore._read_applied_slots_from_ranks()
    assert slots == [12, 13, 999], \
        f"untrusted ids must be clipped to real ones: {slots}"
