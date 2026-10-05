r"""The ledger's own honesty: a load that aged out must not erase the truth about it.

FOUR REPORTING FAULTS, ALL FOUND ON THE SAME 00:08 REQUEST, ALL FIXED IN 0.2.25. The
request that answered garbage had, beside its wrong answer, a payload that could not
say what happened to it:

* `restorations` stayed 0 while 36 reads were applied, because the ledger entry was
  released (the fork's own stale-pin release, or the deadline) BEFORE the fork's
  `finished_recving` arrived, and the report loop then found `_loads.get(id) is None`
  and `continue`d. Silence, in the one branch where the truth was still available.
* `watchdog_fires`, `no_read` and the rest of the release-without-restoration evidence
  existed in `_state` and were never in the deployed payload, so the branch had to be
  INFERRED from `suppressions`/`loads_released` - the operator's word for it was
  "guessed", and a guess is not an instrument.
* Nothing anywhere coupled "we claimed these GPU blocks for a disk span" to "the fork
  finished these GPU blocks". `disk_gpu_blocks_dma_verified` is that coupling, and it
  is the only field in this module that answers "were the blocks actually written?".
* `witness_kind=offers_not_adopted` was published beside `adopted=1`. The kind was
  true when it was SPOKEN and false when it was READ, because emissions dedupe on
  `(verdict, kind)` and a counter can move without changing either.

WHAT THIS SUITE DOES NOT TOUCH: the deadline's protective purpose. The measured window
says the expiry was never a disk too slow - a read completes in 9-30 ms
(`read_seconds_max` 0.018, 9-12 ms waited of a 139 ms settle budget) against a
`RESTORE_DEADLINE_SECONDS` of 45, about fifteen hundred times the read it guards. So
the deadline stays exactly where it is and keeps fusing the request, and what changed
is only that the late truth is now counted instead of dropped. A longer deadline would
have slowed the fuse that stops the re-offer loop that starved the box on 2026-10-05,
to hide a bug that was never the deadline.
"""

from __future__ import annotations

import time

import fakes_debug
import test_disk_restore as base

from kvtier_dev_plugin import disk_restore

armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

BLOCK = base.BLOCK
Manager = base.Manager
Meta = base.Meta
Req = base.Req
fill = base.fill
offer = base.offer
merged_cpu = base.merged_cpu


def output(recving=(), failed=(), invalid=()):
    """A `KVConnectorOutput` with the three halves the scheduler reads."""
    return fakes_debug.Obj(name="KVConnectorOutput",
                           finished_recving=set(recving),
                           finished_sending=set(),
                           failed_recving=set(failed),
                           invalid_block_ids=set(invalid))


def adopt(manager, request_id="req-1", event=4, gpu_start=70_000):
    """Emit the offer's paths in a step, which is what makes it ADOPTED.

    Returns `(cpu_ids, gpu_ids)` as the fork's step carried them, index-aligned: the
    GPU half is what `disk_gpu_blocks_dma_verified` later counts, so a test that
    invented a different length here would be measuring a fiction.
    """
    cpu = merged_cpu(manager, request_id)
    gpu = [gpu_start + index for index in range(len(cpu))]
    disk_restore.build_load_paths(manager, Meta(load_event=event, cpu=cpu, gpu=gpu,
                                                reqs={event: [request_id]}))
    return cpu, gpu


def expire(manager, request_id="req-1"):
    """Drive the ledger's own sweep past this load's deadline. BOUNDED, no sleeping.

    The deadline is moved rather than waited out; `time.monotonic()` is not patched, so
    what the sweep decides is the real branch's real decision.
    """
    record = disk_restore._loads[request_id]
    record["deadline_mono"] = time.monotonic() - 1
    # `build_load_paths` sweeps at the top of every step; that is the door the live box
    # hits, so the test goes through it rather than calling the sweep privately.
    disk_restore.build_load_paths(manager, Meta())


# --- 1. the vanish, closed (item 3) ------------------------------------------

def test_a_load_whose_bytes_landed_and_then_aged_out_is_recorded_as_such(armed):
    """THE REPORTING BUG, END TO END: adopted, read, aged out, REPORTED LATE.

    Before 0.2.25 that sequence produced `restorations=0, loads_released=1` and a
    `continue`. Now the release leaves a tombstone and the late report is matched to
    it: the ledger still refuses to call it a `restoration` (it only counts a report it
    held), and the new counters carry the truth the drop used to destroy.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1     # the worker's attestation
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    _, gpu = adopt(manager)
    expire(manager)

    aged = disk_restore.counters()
    assert aged["aged_out_loads"] == 1
    assert aged["aged_out_with_applied_reads"] == 1
    assert aged["loads_released"] == 1
    assert aged["watchdog_fires"] == 1, ("the deadline branch is still the "
                                             "deadline branch")
    assert aged["suppressions"] == 1, "and it still fuses the request"
    assert "adopted=true" in aged["last_aged_out"]
    assert f"gpu_blocks_mapped={len(gpu)}" in aged["last_aged_out"]

    report = disk_restore.note_reported(manager, output(recving=["req-1"]))
    counters = disk_restore.counters()
    assert report["restorations"] == 0, ("`restorations` keeps meaning 'reported while "
                                        "the ledger held it'")
    assert counters["late_finished_reports"] == 1
    assert counters["restored_after_release"] == 1
    assert counters["disk_gpu_blocks_dma_verified"] == len(gpu)
    assert counters["restorations"] == 0


def test_a_load_the_fork_released_underneath_us_is_tombstoned_too(armed):
    """THE OTHER RELEASE DOOR, WHICH THE FORK OPENS ITSELF AND OFTEN.

    `update_state_after_alloc` returns early - releasing its own pin - whenever it is
    handed no external tokens, and the live engine logs that line in about 10% of its
    output. When it does, this request is never in `_reqs_to_load`, our ledger entry is
    stale on the next sweep, and the fork may still call the load finished later. Both
    doors now leave the same kind of tombstone, so the late report is correlated
    whichever one fired - which matters because the two doors were previously
    indistinguishable from the payload (`watchdog_fires` was not even published).
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    adopt(manager)
    manager._reqs_to_load.pop("req-1")        # the fork's own release
    disk_restore.build_load_paths(manager, Meta())

    counters = disk_restore.counters()
    assert counters["aged_out_loads"] == 1
    assert counters["watchdog_fires"] == 0, ("this is the fork's door, not our "
                                             "deadline: and now the payload can tell "
                                             "the two apart")
    assert counters["suppressions"] == 0, "the deadline's fuse does not fire here"
    assert "no longer has a load state" in counters["last_aged_out"]
    assert disk_restore.note_reported(manager, output(recving=["req-1"]))
    assert disk_restore.counters()["late_finished_reports"] == 1
    assert disk_restore.counters()["restored_after_release"] == 1


def test_a_late_report_without_read_evidence_is_named_but_not_claimed_restored(armed):
    """THE SAME BRANCH, THE HONEST OUTCOME: no read behind it means no restoration.

    `aged_out_with_applied_reads` and `restored_after_release` exist to separate "the
    bytes were on their way and the ledger let go" from "the ledger let go and we do not
    know". Both are counted; only the first is allowed to read as a restoration, and
    this is the test that stops the second one from ever being upgraded.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    adopt(manager)
    expire(manager)                                  # no read applied in this boot
    assert disk_restore.counters()["aged_out_with_applied_reads"] == 0
    disk_restore.note_reported(manager, output(recving=["req-1"]))
    counters = disk_restore.counters()
    assert counters["late_finished_reports"] == 1
    assert counters["restored_after_release"] == 0
    assert counters["disk_gpu_blocks_dma_verified"] == 0


def test_a_release_at_the_ledger_cap_is_remembered_as_well(armed):
    """THE THIRD DOOR. An eviction is the ledger giving up, and says so.

    The `_loads` cap is 64 requests; a sixty-fifth evicts the oldest, and until 0.2.25
    that evicted request's later report vanished exactly like the deadline's did.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    adopt(manager)
    # The cap door is `while len(_loads) >= cap: _evict_oldest_load()` in
    # `_register_load`, and the helper's whole job is "drop the oldest". Our record IS
    # the oldest, so calling it is the branch, no sixty-four offers required.
    disk_restore._evict_oldest_load()
    counters = disk_restore.counters()
    assert counters["aged_out_loads"] == 1
    assert "at its cap" in counters["last_aged_out"]
    assert counters["restorations"] == 0


# --- 2. the blocks, not just the requests (item 5) ---------------------------

def test_the_gpu_blocks_the_fork_reported_are_counted_as_written(armed):
    """THE COUPLING. Our claim names GPU blocks; the fork's report names the request.

    ADOPTING a disk offer is the only moment the two meet in this module's world: the
    step's `load_cpu_blocks` carries a slot WE allocated and its `load_gpu_blocks`
    carries the block the fork will DMA into. So the ledger remembers the GPU half from
    that step, and a `finished_recving` that matches the record adds those blocks. This
    is the field that answers "were the GPU blocks actually written", which
    `restorations` (a per-REQUEST count) never could.
    """
    fill(armed, blocks=2)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=2), 0, (0, False))
    _, gpu = adopt(manager)
    assert len(gpu) == 6, ("two blocks x the three eligible groups, exactly as "
                             "the step carried them")
    report = disk_restore.note_reported(manager, output(recving=["req-1"]))
    counters = disk_restore.counters()
    assert report["restorations"] == 1
    assert counters["disk_gpu_blocks_dma_verified"] == 6
    assert counters["disk_gpu_blocks_released_unreported"] == 0


def test_an_offer_that_was_never_adopted_verifies_no_gpu_blocks(armed):
    """A CLAIM IS NOT A WRITE. Offering reserves nothing on the GPU and counts nothing.

    The counter is not movable by an offer, a refusal, or an offer the scheduler threw
    away below its adoption floor: only an emitted step plus a completion report moves
    it, which is what makes it evidence rather than an echo of `offers`.
    """
    fill(armed, blocks=2)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=2), 0, (0, False))
    assert disk_restore.counters()["offers"] == 1
    assert disk_restore.counters()["disk_gpu_blocks_dma_verified"] == 0
    disk_restore.note_reported(manager, output(recving=["req-1"]))
    counters = disk_restore.counters()
    assert counters["restorations"] == 0, "never reached a worker"
    assert counters["disk_gpu_blocks_dma_verified"] == 0


def test_a_step_that_repeats_the_same_load_does_not_count_the_blocks_twice(armed):
    """IDEMPOTENT ACROSS STEPS. The fork may carry a load for several steps.

    `_reqs_to_load` survives until the load event completes, so the same CPU/GPU pair
    can be emitted more than once. The ledger maps a GPU block once per load, which is
    what makes `disk_gpu_blocks_dma_verified` a block count and not a step count.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    cpu, gpu = adopt(manager)
    again = Meta(load_event=5, cpu=cpu, gpu=gpu, reqs={5: ["req-1"]})
    disk_restore.build_load_paths(manager, again)
    disk_restore.build_load_paths(manager, again)
    assert len(disk_restore._loads["req-1"]["gpu_block_ids"]) == len(gpu)
    disk_restore.note_reported(manager, output(recving=["req-1"]))
    assert disk_restore.counters()["disk_gpu_blocks_dma_verified"] == len(gpu)


def test_a_step_whose_two_lists_disagree_maps_no_gpu_blocks(armed):
    """AN UNALIGNED STEP SAYS NOTHING ABOUT BLOCKS, and is not guessed at.

    The worker's own bind refuses a step where `load_cpu_blocks` and `load_gpu_blocks`
    differ in length; the scheduler must not do what the worker would refuse, so a
    disagreeing pair attributes NO gpu ids rather than a plausible-looking prefix. A
    wrong pairing here would attribute a finished DMA to a block nobody wrote.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    cpu = merged_cpu(manager)
    disk_restore.build_load_paths(manager, Meta(load_event=4, cpu=cpu,
                                                gpu=[70_000], reqs={4: ["req-1"]}))
    assert disk_restore._loads["req-1"]["gpu_block_ids"] == []
    assert disk_restore.counters()["adopted"] == 1, ("the ADOPTION still counts: this "
                                                     "is only the block coupling "
                                                     "declining to guess")


# --- 3. the counters that existed and were not published (item 4) ------------

def test_the_evidence_the_00_08_investigation_had_to_infer_is_published(armed):
    """ONE PAYLOAD READ, no inference: release-without-restoration, countable.

    These fields all lived in `_state` and never reached the deployed `/snapshot`,
    which is why the incident's own summary read `suppressions=1, suppressed_requests=1,
    loads_released=1, restorations=0` and had to GUESS that a report had arrived late.
    Every one of them is now a published integer with a units entry, and
    `last_aged_out` says which door released the last load in prose.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    adopt(manager)
    expire(manager)
    disk_restore.note_reported(manager, output(recving=["req-1"]))
    section = disk_restore.scheduler_section()
    published = ("watchdog_fires", "no_read", "invalid_reports", "suppressed_requests",
                 "suppressed_spans", "aged_out_loads",
                 "aged_out_with_applied_reads", "late_finished_reports",
                 "restored_after_release", "disk_gpu_blocks_dma_verified",
                 "disk_gpu_blocks_released_unreported", "last_aged_out")
    for key in published:
        assert key in section, f"{key} is published"
        assert key in section["units"], f"{key} has a units entry"
    assert section["watchdog_fires"] == 1
    assert section["late_finished_reports"] == 1
    assert section["restored_after_release"] == 1
    assert section["disk_gpu_blocks_released_unreported"] == 3
    assert isinstance(section["last_aged_out"], str)
    # The scheduler's own `invalid_reports` is 0 by construction, and the units entry
    # has to SAY that: the live number lives in the worker's rank file, and a bare 0
    # in a scheduler payload otherwise reads like a contradiction of it.
    assert section["invalid_reports"] == 0
    assert "rank file" in section["units"]["invalid_reports"]


# --- 4. the kind that lied (item 6) ------------------------------------------

def test_the_as_of_emission_kind_cannot_be_mistaken_for_a_live_claim(armed,
                                                                      monkeypatch):
    """`witness_kind=offers_not_adopted` beside `adopted=1`, and what makes it honest.

    The emission dedupes on `(verdict, kind)` and only speaks when one changes, so a
    kind spoken before `_mark_adopted` stays published after it - the sentence was true
    at emission and false at read time, and the REFUTED verdict beside it was right.
    The published payload now carries both readings: the kind the log line carried, and
    the kind the current counters would give, with a flag naming the disagreement.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    disk_restore.witness_verdict()                  # spoken BEFORE adoption
    frozen = disk_restore._state["witness_kind"]
    assert frozen == "offers_not_adopted"
    adopt(manager)                                  # adopted=1, verdict unchanged

    now = disk_restore.witness_now()
    assert now["kind"] == "adopted_never_reported"
    assert now["stale_as_of_emission"] is True
    monkeypatch.setattr(disk_restore, "fitted_methods_here",
                        lambda: list(disk_restore.SCHED_METHODS))
    disk_restore._state["sched_fitted"] = True
    section = disk_restore.scheduler_section()
    assert section["witness_kind"] == frozen, "the as-of-emission name is not rewritten"
    assert section["witness_kind_now"] == "adopted_never_reported"
    assert section["witness_kind_is_stale"] is True
    assert "AS OF THE LAST EMISSION" in section["units"]["witness_kind"]


def test_witness_now_reads_the_state_and_writes_nothing(armed):
    """A PAYLOAD READ MUST NOT REWRITE HISTORY. `witness_now()` is pure.

    If it wrote `witness_kind`/`witness_verdict`, polling the dashboard would both
    rewrite the as-of-emission record and change what the next `emit_witness()` decides
    to log - a diagnostic whose act of reading moves the thing it dedupes on. So this
    asserts the property the split of `_verdict_and_kind` exists to give.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    disk_restore.witness_verdict()
    before = (disk_restore._state["witness_verdict"],
              disk_restore._state["witness_kind"],
              disk_restore._state["witness_reason"],
              disk_restore._state["witness_emissions"],
              disk_restore._witness_state["key"])
    for _ in range(3):
        disk_restore.witness_now()
    after = (disk_restore._state["witness_verdict"],
             disk_restore._state["witness_kind"],
             disk_restore._state["witness_reason"],
             disk_restore._state["witness_emissions"],
             disk_restore._witness_state["key"])
    assert before == after


def test_the_refuted_reason_vocabulary_and_its_live_twin_agree(armed):
    """THE TWO READINGS USE ONE ARITHMETIC, so they cannot drift.

    `witness_verdict()` and `witness_now()` share `_verdict_and_kind`; a fork of the
    verdict logic into the read path would be the same bug wearing a different hat -
    two answers to one question, disagreeing, in one payload.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    offer(manager, Req(request_id="req-1", blocks=1), 0, (0, False))
    spoken = disk_restore.witness_verdict()
    live = disk_restore.witness_now()
    assert spoken["verdict"] == live["verdict"] == "REFUTED"
    assert spoken["kind"] == live["kind"]
    assert live["stale_as_of_emission"] is False
