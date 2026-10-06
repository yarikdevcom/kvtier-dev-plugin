r"""WHAT A FAILED LOAD COSTS: a race may cost the tokens, a fault costs the offer.

Before 0.2.32 every failure report called `suppress_load` - black-holing the request and
fusing the span for the rest of the boot - while the reason string it handed that call
already contained `sorted(_recent_failure_kinds())`. The code knew which kind it was and
decided without looking, and on 2026-10-06 that cost a live box the following: a request
with a 28,583-token prompt had 12 jobs / 137.6 MB offered, the settle formula granted
23.6 ms for reads whose bind-to-apply race gap measured 89.7 / 97.1 / 129.7 / 135.8 ms,
and the loss was `drop_running` (a kernel read mid-flight - the bytes were FINE). The
bill came as `cached_tokens: 0`, a 12.26 s recompute, `suppressed_requests: 3` and
`suppressed_after_failure: 5`, and the suppression had no expiry to wait for.

This suite pins the replacement, and every one of these tests is a property that was
FALSE in 0.2.31:

1. THE PARTITION. The 26 names in `WORKER_FAILURES` split into timing, capacity and
   data, each name in exactly one class, published in the payload so an operator can
   argue with it instead of reading this file.
2. A RACE DOES NOT FUSE. A report whose kinds are all timing/capacity costs the request
   its tokens and NOT its next offer, and does not implicate the span for other
   requests.
3. A FAULT STILL FUSES. Corruption, a short read, a failed copy: the fuse is exactly
   where the 2026-10-05 loop-killer put it, and an UNKNOWN report fuses too - "nobody
   saw it" is not proof of a race.
4. THE LOOP STAYS BOUNDED. A spared request gets exactly ONE re-offer, spent when
   used and never refilled, so no request is ever offered a third time whether its
   losses were races or faults.
5. THE ENGINE'S REPORT SHAPE IS NOT EVIDENCE. `failed_recving` and `invalid_block_ids`
   name the CHANNEL the bad news arrived on; `_note_load_failure` files them under
   `read_failed`, so a classifier reading the plain ring would call every report a
   corruption and the race would never be spared.

Also pinned, because it is the half of the change the plugin can actually do: an entry
that does not land within the allowance is PRUNED and its GPU block reported, while the
entries that DID land stay in the step. The prefix survives; only the hole recomputes -
as far as the fork's multi-group truncation rule allows (see
`partial_load_verdict`'s item 5 for where the engine itself refuses to shrink).
"""

from __future__ import annotations

import json

import test_disk_restore as base

from kvtier_dev_plugin import (
    config,
    disk_read,
    disk_restore,
)

armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

CACHEABLE = base.CACHEABLE
Manager = base.Manager
Req = base.Req
fill = base.fill
offer = base.offer
bind = base.bind
settle = base.settle
disk_meta = base.disk_meta
Worker = base.Worker
BLOCK = base.BLOCK


def report(**fields):
    """A `KVConnectorOutput` with the four fields the report reader touches."""
    return base.fakes_debug.Obj(
        name="KVConnectorOutput",
        finished_recving=set(fields.get("recving") or ()),
        finished_sending=set(fields.get("sending") or ()),
        failed_recving=set(fields.get("failed") or ()),
        invalid_block_ids=set(fields.get("invalid") or ()))


def worker_saw(kind, request_id="req-1", span=None):
    """The record a WORKER writes when a read does not happen, in this process's ring.

    In production this arrives as a rank file and lands in
    `_state["rank_failure_records"]`; the classifier reads both, and the scheduler's own
    `reads_failed` is zero by construction, so this is how a test says "the code that
    touched the bytes blamed THIS".
    """
    return disk_restore._note_load_failure(kind, where="apply",
                                           reason="test: the worker's own verdict",
                                           request_id=request_id, span=span)


def adopt(manager, request_id="req-1", event=3, gpu_start=700):
    """Emit the offer's paths in a step, which is what makes it ADOPTED."""
    cpu = base.merged_cpu(manager, request_id)
    gpu = [gpu_start + index for index in range(len(cpu))]
    disk_restore.build_load_paths(manager, base.Meta(load_event=event, cpu=cpu,
                                                     gpu=gpu,
                                                     reqs={event: [request_id]}))
    return cpu, gpu


# --- 1: the partition --------------------------------------------------------

def test_every_failure_kind_belongs_to_exactly_one_class():
    """The partition, closed and non-overlapping, so the fuse rule is total.

    An unclassified name is DATA by `failure_kind_class`'s fail-closed default - which
    is safe, but a NEW kind added to `WORKER_FAILURES` and left out of this partition
    would silently re-fuse races, so the union is pinned against the vocabulary.
    """
    timing, capacity, data = (disk_restore.LOAD_TIMING_FAILURES,
                              disk_restore.LOAD_CAPACITY_FAILURES,
                              disk_restore.LOAD_DATA_FAILURES)
    for name in disk_restore.WORKER_FAILURES:
        classes = [name in timing, name in capacity, name in data]
        assert sum(classes) == 1, f"{name} in {sum(classes)} classes"
    assert sorted(timing + capacity + data) == sorted(disk_restore.WORKER_FAILURES)
    # THE PARTITION, as the operator ruled it. Pinned literally because a silent move of
    # one name across the line changes what a production failure costs.
    assert set(timing) == {"not_settled", "deadline_expired"}
    assert set(capacity) == {"queue_full", "buffer_cap"}
    for name in ("short_read", "size_mismatch", "digest_disagreement",
                 "identity_disagreement", "copy_failed", "read_error",
                 "file_missing", "path_count_mismatch", "bad_path", "bad_size",
                 "bad_batch", "bad_job", "batch_cap", "incomplete", "no_channel",
                 "thread_dead", "fingerprint_incomplete", "not_ours",
                 "slot_plan_failed", "not_tracked", "abandoned", "read_failed"):
        assert name in data, name


def test_the_classifier_answers_three_ways_and_fuses_the_unknown():
    """`data` outranks `timing`; only an EMPTY set is `unknown`; unknown fuses."""
    classify = disk_restore.classify_failure_kinds
    assert classify(set()) == "unknown"
    assert classify({"not_settled"}) == "timing"
    assert classify({"deadline_expired", "queue_full"}) == "timing"
    assert classify({"buffer_cap"}) == "timing", "a full pool is not corruption"
    # ONE bad byte in the window outranks any number of races.
    assert classify({"not_settled", "short_read"}) == "data"
    # A name outside the vocabulary is a drift, and a drift is not a race.
    assert classify({"something_new"}) == "data"
    # And the classes travel in the payload, so this is checkable on a live box.
    classes = disk_restore.failure_kind_classes()
    assert classes["kinds_total"] == len(disk_restore.WORKER_FAILURES)
    assert classes["unattributable_are_data"] is True
    assert "rule" in classes and "read_from" in classes
    section = disk_restore.scheduler_section()
    assert section["failure_kind_classes"]["kinds_total"] == 26
    assert section["units"]["failure_kind_classes"]


def test_the_engines_report_shape_is_not_evidence_of_a_fault():
    """`failed_recving` / `invalid_block_ids` name a CHANNEL, not a cause.

    `_note_load_failure` files a name outside `WORKER_FAILURES` under `read_failed`, and
    `note_reported` writes those two records BEFORE it decides. A classifier reading the
    plain ring would therefore see `read_failed` (a DATA kind) on every report ever
    written and would never spare a race - the fix would be inert while every test for
    it still passed by accident. This pins the exclusion that keeps it alive.
    """
    for shape in ("failed_recving", "invalid_block_ids"):
        disk_restore._note_load_failure(shape, where="report",
                                        reason="the engine named a failure",
                                        request_id="req-1", provenance=False)
    assert disk_restore._recent_failure_kinds(), "the plain ring DOES see them"
    assert disk_restore._attributable_failure_kinds() == set()
    assert disk_restore.classify_failure_kinds(
        disk_restore._attributable_failure_kinds()) == "unknown"


# --- 2: a race does not fuse -------------------------------------------------

def test_a_race_report_costs_the_tokens_and_not_the_next_offer(armed):
    """THE FIX, END TO END: `not_settled` on a `failed_recving` must not fuse.
    Tonight's shape, in one step: the engine names the request's load as failed while
    the worker's own record says the read had simply not landed. The ledger closes (the
    slots go back, the load leaves the table) and the request is NOT black-holed.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)
    worker_saw("not_settled", request_id="req-1")
    before = disk_restore.counters()
    out = disk_restore.note_reported(manager, report(failed={"req-1"}))
    counters = disk_restore.counters()
    assert out["failures"] == 1 and out["races_not_fused"] == 1, out
    assert counters["suppressions"] == before["suppressions"], "no fuse was raised"
    assert counters["races_not_fused"] == 1
    assert counters["fuses_on_data_kinds"] == 0
    assert counters["fuses_on_unknown_kinds"] == 0
    assert counters["loads_released"] == 1, "the ledger still closed"
    assert counters["tracked_slots"] == 0, "and the slots went back with it"
    assert counters["suppressed_requests"] == 0
    assert counters["suppressed_spans"] == 0, "the span is not implicated by a race"
    assert "verdict=timing" in counters["last_fuse_decision"]
    # And the request is still offerable: this is the property 0.2.31 failed.
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)


def test_a_race_on_the_invalid_block_shape_does_not_fuse_either(armed):
    """The shape the live box actually produced: ids, and an EMPTY `failed_recving`.

    66 requests came back with `invalid_block_ids=1..3` and no request named, and that
    path burned every adopted load on the step - including the three that had only lost
    a race. Same rule, same answer.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    _, gpu = adopt(manager)
    worker_saw("not_settled", request_id="req-1")
    out = disk_restore.note_reported(manager, report(invalid=set(gpu)))
    counters = disk_restore.counters()
    assert out["failures"] == 1 and out["races_not_fused"] == 1, out
    assert counters["suppressions"] == 0
    assert counters["loads_released"] == 1


def test_a_capacity_loss_is_not_corruption_either(armed):
    """`buffer_cap` at the bind is the pool being busy, not a bad file.

    With the offer path pricing the pool up front, a bind-time `buffer_cap` means a
    CONCURRENT offer committed the bytes first (the arbiter is still TODO 14) - a moment
    the next offer re-prices. Fusing it for the boot was the 0.2.30 shape.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)
    worker_saw("buffer_cap", request_id="req-1")
    disk_restore.note_reported(manager, report(failed={"req-1"}))
    counters = disk_restore.counters()
    assert counters["races_not_fused"] == 1
    assert counters["suppressions"] == 0


def test_a_data_failure_still_fuses_the_request_and_the_span(armed):
    """The half that must NOT be weakened: bad bytes still cost the offer, forever.

    A short read means the store and the index disagree about something, and re-offering
    it is the 2026-10-05 loop (123,474 offers in 3.9 minutes). The fuse stays exactly
    where that incident put it - request AND span.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)
    worker_saw("short_read", request_id="req-1")
    out = disk_restore.note_reported(manager, report(failed={"req-1"}))
    counters = disk_restore.counters()
    assert out["failures"] == 1 and out["races_not_fused"] == 0, out
    assert counters["suppressions"] == 1
    assert counters["fuses_on_data_kinds"] == 1
    assert counters["races_not_fused"] == 0
    assert "verdict=data" in counters["last_fuse_decision"]
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=2)) == (0, False), \
        "a data failure must not be offered again"
    # And the SPAN half: a different request wanting the same bytes is refused too.
    other = offer(Manager(reqs_to_load={"req-9": object()}),
                  Req(request_id="req-9", blocks=2))
    assert other == (0, False)
    assert disk_restore.counters()["suppressed_spans"] == 1


def test_an_unattributable_report_fuses_and_says_so(armed):
    """UNKNOWN is not a race. The first loss of an episode usually fuses, and the
    payload has to be able to say why `races_not_fused` is 0.

    The scheduler learns a worker's kinds through the rank file, which lags
    (`PROOF_CACHE_SECONDS`), so guessing "race" here is how the loop comes back. A quiet
    `races_not_fused` therefore means "the kinds never arrived", and
    `fuses_on_unknown_kinds` is the field that distinguishes that from "no races".
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)
    disk_restore.note_reported(manager, report(failed={"req-1"}))
    counters = disk_restore.counters()
    assert counters["fuses_on_unknown_kinds"] == 1
    assert counters["suppressions"] == 1
    assert counters["races_not_fused"] == 0
    assert "verdict=unknown" in counters["last_fuse_decision"]


# --- 4: the loop stays bounded ----------------------------------------------

def test_a_race_earns_exactly_one_re_offer_and_never_a_third(armed):
    """THE SMALLEST LOOP RULE: one re-offer per request, spent, never refilled.

    Without it, deleting the permanent fuse for a race would delete the loop's only
    bound. With it, a request that loses the race twice gets no third offer - so the
    worst-case loop is TWO offers per request whether the losses are races or faults,
    against the 123,474 that starved the box on 2026-10-05.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)
    worker_saw("not_settled", request_id="req-1")
    disk_restore.note_reported(manager, report(failed={"req-1"}))
    # The engine rescheduled it: the same id, back at the SAME computed tokens.
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True), \
        "the one re-offer is unspent, so this offer must happen"
    assert disk_restore.counters()["race_reoffers_granted"] == 1
    # It loses the race again. The allowance is spent, so the third asking is refused.
    worker_saw("not_settled", request_id="req-1")
    disk_restore.note_reported(manager, report(failed={"req-1"}))
    counters = disk_restore.counters()
    assert counters["races_not_fused"] == 2, "both losses were races"
    assert counters["suppressions"] == 0
    third = offer(Manager(reqs_to_load={"req-1": object()}),
                  Req(request_id="req-1", blocks=2))
    assert third == (0, False), "a request is offered at most TWICE per boot"
    counters = disk_restore.counters()
    assert counters["suppressions"] == 1, "the exhausted allowance fused it"
    assert counters["suppressed_hits"] == 1
    assert counters["refused"]["suppressed_after_failure"] == 1
    assert counters["offers"] == 2, "exactly two offers for three askings"


def test_a_restoration_clears_the_allowance(armed):
    """A load that LANDED has no use for a pending re-offer, so the row goes away.

    Also the bound in a long boot: the table is keyed by request id and a request that
    succeeds must not sit in it for the rest of the run.
    """
    fill(armed, blocks=1)
    disk_restore._state["reads_applied"] = 1        # the worker's attestation
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    adopt(manager)
    worker_saw("not_settled", request_id="req-1")
    disk_restore.note_reported(manager, report(failed={"req-1"}))
    assert "req-1" in disk_restore._race_reoffer, "spared, so eligible"
    again = Manager(reqs_to_load={"req-1": object()})
    assert offer(again, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    adopt(again)
    out = disk_restore.note_reported(again, report(recving={"req-1"}))
    assert out["restorations"] == 1, out
    assert "req-1" not in disk_restore._race_reoffer, "a success spent the row away"


# --- 5: the plugin-side half of the shrink ----------------------------------

def one_hole(armed, monkeypatch, job_id=0, *, cap_ms="20", until=3600.0):
    """Let every disk entry land EXCEPT one, which never settles within the test.

    The bytes underneath the missing one are genuinely absent (`batch_buffer` answers
    None for that job alone, which is what "not DONE" means) while the batch reports one
    read pending and that read RUNNING - the `drop_running` shape of the live incident.
    """
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, cap_ms)
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    assert bind(worker, metadata)["ok"] is True
    real_status = disk_read.batch_status
    real_buffer = disk_read.batch_buffer
    batch_id = disk_restore._batch_id(11)
    stop_lying = __import__("time").monotonic() + until

    def fake_status(bid):
        if bid != batch_id or __import__("time").monotonic() >= stop_lying:
            return real_status(bid)
        return {"known": True, "settled": False, "pending": 1, "done": 0,
                "failed": 0, "running": 1, "queued": 0, "incomplete": False,
                "deadline_expired": False, "accepted": len(CACHEABLE),
                "expected": len(CACHEABLE), "bytes_read": 0,
                "seconds_left": 29.0, "elapsed_seconds": 0.004, "failures": []}

    def fake_buffer(bid, job):
        if bid == batch_id and int(job) == int(job_id) \
                and __import__("time").monotonic() < stop_lying:
            return None
        return real_buffer(bid, job)

    def fake_states(bid, limit=1024):
        if bid != batch_id or __import__("time").monotonic() >= stop_lying:
            return disk_read.job_states(bid, limit)
        return {"known": True, "truncated": False, "count": len(CACHEABLE),
                "states": {index: (disk_read.RUNNING if index == int(job_id)
                                   else disk_read.DONE)
                           for index in range(len(CACHEABLE))}}

    monkeypatch.setattr(disk_read, "batch_status", fake_status)
    monkeypatch.setattr(disk_read, "batch_buffer", fake_buffer)
    monkeypatch.setattr(disk_read, "job_states", fake_states)
    return worker, metadata


def test_a_hole_costs_one_entry_and_not_the_step(armed, monkeypatch):
    """THE SHRINK THE PLUGIN CAN DO: the landed prefix stays in the step, the hole goes.

    Not zero, and not the whole batch: `applied` is the entries that had bytes, the
    surviving pairs stay in `load_cpu_blocks`/`load_gpu_blocks` so the fork still DMAs
    them, and the ONLY GPU block reported to the engine is the one whose bytes are
    missing. That is the difference between "recompute a hole" and "recompute a prompt".
    """
    worker, metadata = one_hole(armed, monkeypatch, job_id=0)
    report = disk_restore.worker_before_load(worker)
    counters = disk_restore.counters()
    assert report["dropped"] == 1, report
    assert report["applied"] == len(CACHEABLE) - 1, report
    assert report["submitted"] is True, "the survivors were still DMA'd"
    assert len(metadata.load_cpu_blocks) == len(CACHEABLE) - 1
    assert len(metadata.load_gpu_blocks) == len(CACHEABLE) - 1
    assert counters["reads_applied"] == len(CACHEABLE) - 1
    assert counters["reads_failed"]["not_settled"] == 1
    assert counters["dropped_running"] == 1
    assert counters["dropped_queued"] == 0
    # The apply point waits the allowance and then decides: this is what an honest wait
    # metric is FOR, and `budget_over_recompute` reported 0 here.
    window = counters["last_apply_window"]
    assert window["dropped_after_wait"] is True
    assert window["allowance_spent"] is True
    assert counters["settle_allowance_spent"] == 1
    # And the apply point is NOT where a request gets fused.
    assert counters["suppressions"] == 0
    assert counters["races_not_fused"] == 0


def test_the_worker_reports_the_hole_and_the_scheduler_spares_the_offer(armed,
                                                                        monkeypatch):
    """THE WHOLE TONIGHT, in one process: race at the apply point, fuse decision next.

    The worker drops the entry it could not read, reports its GPU block, and writes a
    `not_settled` record with `where="apply"`. The report that reaches the scheduler
    then reads THAT record and spares the request - which is precisely the decision
    0.2.31 made backwards, and the reason `suppressed_requests` reached 3 while a 12 ms
    race was billed as a 12-second recompute.
    """
    worker, _metadata = one_hole(armed, monkeypatch, job_id=0)
    disk_restore.worker_before_load(worker)
    after_apply = disk_restore.counters()
    assert after_apply["dropped_running"] == 1
    assert after_apply["suppressions"] == 0
    # The scheduler side of the same boot adopts a load and then receives the bad news.
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    _, gpu = adopt(manager)
    out = disk_restore.note_reported(manager, report(invalid=set(gpu)))
    counters = disk_restore.counters()
    assert out["races_not_fused"] == 1, out
    assert counters["suppressions"] == 0
    assert counters["fuses_on_data_kinds"] == 0
    assert counters["fuses_on_unknown_kinds"] == 0
    assert "verdict=timing" in counters["last_fuse_decision"]
    # The decision is a payload field, not only a log line.
    section = disk_restore.scheduler_section()
    assert section["races_not_fused"] == 1
    assert section["suppressions"] == 0
    assert section["units"]["races_not_fused"]


# --- 6: nothing that was deleted may still be read ---------------------------

def test_the_deleted_settle_knobs_and_fields_are_gone_from_every_surface():
    """A knob nothing computes on is worse than no knob: a payload still reports it.
    `posture.scope` was this exact bug class - a field rendering a documentation
    sentence as a posture fact. So the deleted inputs are pinned by BEHAVIOUR/PAYLOAD
    rather than by grep: name every deleted env var with an absurd value, ask the module
    what it grants, and demand that the answer, the counters and every published section
    contain none of the deleted names.
    """
    for name, value in (("VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S", "1"),
                        ("VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S", "1")):
        monkey = __import__("os").environ
        monkey[name] = value
    try:
        # The grammar module must not even HAVE the names, and must parse exactly one.
        assert not hasattr(config, "ENV_DISK_SETTLE_READ_BPS")
        assert not hasattr(config, "ENV_DISK_PREFILL_TOKENS_PER_S")
        assert set(config.settle_env()) == {"cap_ms", "problems"}
        # ...and a deleted knob must not move the grant by a millisecond.
        granted = disk_restore.settle_budget(11_468_800 * 50)["budget_ms"]
        assert granted == disk_restore.SETTLE_MS_DEFAULT == 5000.0
        views = (disk_restore.settle_budget(1),
                 disk_restore.apply_window(),
                 disk_restore.apply_window()["wait"],
                 disk_restore.counters(),
                 disk_restore.scheduler_section(),
                 disk_restore.section(),
                 disk_restore.worker_report())
        blob = json.dumps(views, default=str)
        for gone in ("read_bytes_per_s", "safety_margin", "proportional_ms",
                     "prefill_tokens_per_s", "recompute_ms",
                     "budget_over_recompute", "over_recompute",
                     "settle_bound_min", "settle_bound_proportional",
                     "settle_bound_cap", "settle_budget_over_recompute",
                     "VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S",
                     "VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S"):
            assert f'"{gone}"' not in blob, f"{gone} is still published"
        # `min_ms` survives as a WINDOW statistic (the bind-to-apply spread); what must
        # be gone is the settle FLOOR, which used to sit in the posture beside `cap_ms`.
        assert "min_ms" not in disk_restore.apply_window()["wait"]["posture"]
        assert "min_ms" not in disk_restore.settle_budget(1)
        for gone in ("SETTLE_MIN_MS", "SETTLE_SAFETY_MARGIN",
                     "SETTLE_READ_BPS_DEFAULT", "PREFILL_TOKENS_PER_S_DEFAULT"):
            assert not hasattr(disk_restore, gone), gone
    finally:
        for name in ("VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S",
                     "VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S"):
            __import__("os").environ.pop(name, None)


def test_the_bound_vocabulary_is_the_honest_three(monkeypatch):
    """`flat | deadline | off`, and no fourth meaning invented for an old name.

    The published set must be exactly the set the code can decide, because the whole
    point of the field is to tell an operator WHICH clock stopped the wait.
    """
    assert disk_restore.settle_budget(0)["bound"] == "flat"
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "0")
    assert disk_restore.settle_budget(0)["bound"] == "off"
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "5000")
    assert disk_restore.settle_budget(0, seconds_left=0.001)["bound"] == "deadline"
    wait = disk_restore.apply_window()["wait"]
    assert set(wait["bound"]) == {"flat", "deadline", "off"}
    assert "budget_over_recompute" not in wait
    assert wait["allowance_spent"] == 0
