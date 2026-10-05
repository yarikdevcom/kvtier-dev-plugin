r"""THE RESTORE RACE AS A MEASUREMENT, and the bounded wait that may fix it (0.2.23).

`test_disk_restore.py` pins that an unsettled read is PRUNED and never awaited, which
was the right rule when the only alternative was an unbounded wait inside an engine
step. What it could never say is HOW OFTEN the race is lost or BY HOW MUCH - the whole
finding was one counter (`dma_unsettled_entries`) and a `read_seconds_max` of 0.0 that
was read as "no read ever ran" when it actually means "no read that reached the kernel
ever finished". This suite pins the three things that turn that into an engineering
decision:

1. THE WINDOW IS PUBLISHED. `bind_connector_metadata` and `start_load_kv` are called
   in the same engine step (`kv_connector_model_runner_mixin.py:80-93`), so the gap a
   read has to win is one CPU-side forward launch - and it is now a measured number
   with a per-batch record, boot-wide min/max, and a median that declares how many
   samples it covers. Nothing about the wait is discussable until this is on the box.
2. A DROP SAYS WHICH KIND IT IS. A read still QUEUED lost to thread scheduling (a wait
   fixes that); a read RUNNING lost to a device slower than the step (a wait only
   helps if the budget beats the read). One `not_settled` counter covered both and
   explained neither. A state table that cannot be read is counted as UNKNOWN, never
   folded into one of the two.
3. THE WAIT IS PROPORTIONAL, CAPPED, AND FOUR-WAYS BOUNDED. Bytes over an assumed read
   throughput times a safety margin, floored at one scheduler tick, capped by the
   operator's environment variable, and clamped again by the read's own deadline - and
   every batch says WHICH of those decided, next to the recompute cost it was racing.
   The default cap is 0, and these tests pin that a default boot waits not at all.

Also pinned here, because it was found by reading the counters this release publishes:
a load retired through a FAILURE report used to bypass `_drop_load`, so `loads_released`
never moved and the slot->path mappings outlived the load. See
`test_a_burned_load_releases_its_ledger_entry`.
"""

from __future__ import annotations

import time

import pytest
import test_disk_restore as base

from kvtier_dev_plugin import (
    config,
    disk_read,
    disk_restore,
    worker_diag,
)

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them (pytest collects fixtures from imported names).
armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

CACHEABLE = base.CACHEABLE
WIDTH = base.WIDTH
Manager = base.Manager
Req = base.Req
fill = base.fill
offer = base.offer
bind = base.bind
settle = base.settle
disk_meta = base.disk_meta
Worker = base.Worker

WINDOW_KEYS = ("settled_on_arrival", "bind_to_apply_ms", "elapsed_ms", "known",
               "accepted", "expected", "running", "queued", "settled",
               "bytes_asked", "bytes_at_stake", "budget_ms", "bound",
               "waited_ms", "settled_after_wait", "dropped_after_wait", "wait_granted",
               "recompute_ms", "offered_tokens_estimate", "read_bytes_per_s",
               "safety_margin", "proportional_ms", "prefill_tokens_per_s")


def unsettled_status(monkeypatch, *, queued=0, running=0, seconds_left=29.0,
                     until_settled=5.0):
    """Make the apply point see a batch that has not settled.

    `until_settled` is a SECONDS delay before the lie stops: the bytes underneath are
    genuinely there (the caller settled the batch first), so this exercises the real
    decision - wait, observe settled, apply - instead of a fixture that cannot apply
    anything. It defaults to LONGER THAN ANY TEST, because a default of 0 would stop
    lying before the apply point looked and the test would then measure whatever the
    reader thread happened to do - the flaky nothing these fields exist to replace.
    `queued`/`running` and the matching per-job state table are what the per-entry
    attribution reads.
    """
    real_status = disk_read.batch_status
    stop_lying = time.monotonic() + until_settled
    pending = queued + running

    def fake_status(batch_id):
        if time.monotonic() >= stop_lying:
            return real_status(batch_id)
        return {"known": True, "settled": False, "pending": pending, "done": 0,
                "failed": 0, "running": running, "queued": queued,
                "incomplete": False, "deadline_expired": False,
                "accepted": 3, "expected": 3, "bytes_read": 0,
                "seconds_left": seconds_left, "elapsed_seconds": 0.004,
                "failures": []}

    real_states = disk_read.job_states

    def fake_states(batch_id, limit=1024):
        if time.monotonic() >= stop_lying:
            return real_states(batch_id, limit)
        return {"known": True, "truncated": False, "count": 3,
                "states": {n: (disk_read.QUEUED if queued else disk_read.RUNNING)
                           for n in range(3)}}

    real_buffer = disk_read.batch_buffer

    def fake_buffer(batch_id, job_id):
        # A job that has not settled HAS no buffer - that is what "not DONE" means -
        # so lying about the status without lying about the buffer would describe a
        # state that cannot exist, and the apply point would (correctly) take the bytes.
        if time.monotonic() >= stop_lying:
            return real_buffer(batch_id, job_id)
        return None

    monkeypatch.setattr(disk_read, "batch_status", fake_status)
    monkeypatch.setattr(disk_read, "job_states", fake_states)
    monkeypatch.setattr(disk_read, "batch_buffer", fake_buffer)


# --- 1: the window, published -------------------------------------------------

def test_the_apply_point_publishes_the_window_it_observed(armed):
    """Every field the settle decision needs, at the one point that can see it.

    The assertion is about PRESENCE and SHAPE as much as value: this is the record an
    operator will read on the live box to decide whether waiting can help at all, and
    a missing key in a production run is a wasted day.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    assert bind(worker, metadata)["ok"] is True
    settle(disk_restore._batch_id(11))
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True and report["applied"] == len(CACHEABLE)

    counters = disk_restore.counters()
    window = counters["last_apply_window"]
    for key in WINDOW_KEYS:
        assert key in window, key
    assert window["expected"] == len(CACHEABLE) == window["accepted"]
    assert window["settled"] is True and window["settled_on_arrival"] is True
    assert window["running"] == 0 and window["queued"] == 0
    assert window["bytes_asked"] == WIDTH * len(CACHEABLE)
    # The gap is the fork's own bind -> start_load_kv distance. It cannot be negative,
    # and a value in the seconds range here would mean the two call sites are not in
    # the same step - which is the claim the fork's source makes and this measures.
    assert 0 <= window["bind_to_apply_ms"] < 5000, window["bind_to_apply_ms"]
    assert window["elapsed_ms"] >= 0
    assert counters["apply_observations"] == 1
    assert counters["apply_settled_on_arrival"] == 1
    assert counters["applied_settled_immediately"] == len(CACHEABLE)
    assert counters["dropped_queued"] == 0
    assert counters["dropped_running"] == 0

    view = disk_restore.apply_window()
    assert view["observations"] == 1 and view["settled_on_arrival"] == 1
    for label in ("bind_to_apply_ms", "apply_elapsed_ms"):
        span = view["windows"][label]
        for field in ("min_ms", "max_ms", "avg_ms", "median_ms", "median_samples"):
            assert field in span, (label, field)
        assert span["min_ms"] <= span["median_ms"] <= span["max_ms"], span
        assert span["median_samples"] == 1, "the median must say what it covers"
    assert view["recompute"]["read_seconds_max"] >= 0.0


def test_the_gap_is_measured_from_the_bind_and_not_from_the_apply(armed):
    """A 40 ms pause between the fork's two call sites must SHOW UP in the gap.

    Without the timestamp taken at bind, the only number available at the apply point
    is `elapsed_seconds` - how long the READ has been alive - which is a different
    quantity and would have made this whole instrument read like a tautology.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    assert bind(worker, metadata)["ok"] is True
    time.sleep(0.04)
    settle(disk_restore._batch_id(11))
    disk_restore.worker_before_load(worker)
    window = disk_restore.counters()["last_apply_window"]
    assert window["bind_to_apply_ms"] >= 35.0, window["bind_to_apply_ms"]


def test_min_and_max_are_boot_wide_and_the_median_declares_its_window(armed):
    """Three observations: min <= median <= max, and the sample count is published."""
    for seed in (1, 2, 3):
        fill(armed, blocks=seed)
        worker = Worker()
        metadata = disk_meta(armed, seeds=(seed,), ram=0, event=10 + seed)
        assert bind(worker, metadata)["ok"] is True
        settle(disk_restore._batch_id(10 + seed))
        disk_restore.worker_before_load(worker)
    counters = disk_restore.counters()
    assert counters["apply_observations"] == 3
    assert counters["bind_to_apply_min_ms"] <= counters["bind_to_apply_max_ms"]
    view = disk_restore.apply_window()
    span = view["windows"]["bind_to_apply_ms"]
    assert span["median_samples"] == 3
    assert span["min_ms"] <= span["median_ms"] <= span["max_ms"]
    assert view["samples_retained"] == disk_restore.MAX_SETTLE_SAMPLES


def test_an_untracked_batch_observations_no_window_and_no_crash(armed):
    """Disk paths with no batch behind them: prune, and do not invent a window."""
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    worker.bind_connector_metadata(metadata)      # no bind() call: no batch at all
    report = disk_restore.worker_before_load(worker)
    assert report["ok"] is True and report["dropped"] == len(CACHEABLE)
    counters = disk_restore.counters()
    assert counters["apply_observations"] == 0, "no apply window without a bind"
    assert counters["dropped_queued"] == 0 and counters["dropped_running"] == 0


# --- 2: a drop says WHICH kind it is -----------------------------------------

def test_a_queued_drop_is_counted_as_never_started(armed, monkeypatch):
    """A read still QUEUED when the engine arrived: a scheduling loss, and the kind a
    wait can fix."""
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    bind(worker, metadata)
    unsettled_status(monkeypatch, queued=len(CACHEABLE))
    report = disk_restore.worker_before_load(worker)
    assert report["dropped"] == len(CACHEABLE)
    counters = disk_restore.counters()
    assert counters["dma_unsettled_entries"] == len(CACHEABLE)
    assert counters["dropped_queued"] == len(CACHEABLE)
    assert counters["dropped_running"] == 0
    assert counters["dropped_unattributed"] == 0


def test_a_running_drop_is_counted_as_mid_flight(armed, monkeypatch):
    """A read RUNNING when the engine arrived: a device-speed loss.

    Same drop, different diagnosis - and the only way the operator running the
    experiment can tell "raise the cap" from "the disk is the floor" is that these two
    numbers are separate.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1, event=12)
    bind(worker, metadata)
    unsettled_status(monkeypatch, running=len(CACHEABLE))
    assert disk_restore.worker_before_load(worker)["dropped"] == len(CACHEABLE)
    counters = disk_restore.counters()
    assert counters["dropped_running"] == len(CACHEABLE)
    assert counters["dropped_queued"] == 0


def test_an_unreadable_state_table_is_unknown_and_never_guessed(armed, monkeypatch):
    """No state table means `dropped_unattributed`, not a coin flip.

    A counter that quietly absorbs the unknown case is how a measurement turns into a
    story, and the story this one would tell - "it was always scheduling" - is exactly
    the conclusion the run is trying to be allowed to reach.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    bind(worker, metadata)
    unsettled_status(monkeypatch, queued=len(CACHEABLE))
    monkeypatch.setattr(disk_read, "job_states",
                        lambda *a, **k: {"known": False, "states": {}, "count": 0,
                                         "truncated": False})
    assert disk_restore.worker_before_load(worker)["dropped"] == len(CACHEABLE)
    counters = disk_restore.counters()
    assert counters["dropped_unattributed"] == len(CACHEABLE)
    assert counters["dropped_queued"] == 0 and counters["dropped_running"] == 0


# --- 3: the default boot waits not at all ------------------------------------

def test_the_budget_is_sized_on_the_bytes_still_in_flight(armed, monkeypatch):
    """A batch three-quarters delivered is waiting for a quarter, and pays for that.

    Sizing the budget on the OFFER rather than on what is outstanding would make the
    wait a function of how much the request asked for - so a batch that had already
    read nine of ten slots would be granted a ten-slot budget and sit in front of
    bytes that are not coming. Both numbers are published (`bytes_asked`,
    `bytes_at_stake`) so the rule is checkable from a payload alone.
    """
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "20000")
    monkeypatch.setenv(config.ENV_DISK_SETTLE_READ_BPS, "1000")
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1, event=13)
    bind(worker, metadata)
    unsettled_status(monkeypatch, queued=1, running=0, until_settled=0.01)
    disk_restore.worker_before_load(worker)
    window = disk_restore.counters()["last_apply_window"]
    assert window["bytes_asked"] == WIDTH * len(CACHEABLE)
    assert window["bytes_at_stake"] == WIDTH, window
    expected = WIDTH / 1000.0 * 1000.0 * disk_restore.SETTLE_SAFETY_MARGIN
    assert window["bound"] == "proportional", window
    assert window["budget_ms"] == pytest.approx(expected, abs=0.01), window


def test_a_batch_that_came_in_free_is_not_counted_as_a_wait(armed):
    """A settled batch counts no wait, whatever the budget arithmetic says.

    The counter and the behaviour are decided by the same boolean for this reason:
    `settle_waits_granted` is the denominator of the wait's hit rate, and a boot that
    counted batches it never sat in front of would make every later ratio meaningless.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    bind(worker, metadata)
    settle(disk_restore._batch_id(11))
    disk_restore.worker_before_load(worker)
    counters = disk_restore.counters()
    assert counters["apply_observations"] == 1
    assert counters["settle_waits_granted"] == 0
    assert counters["settle_bytes_waited"] == 0
    assert counters["last_apply_window"]["wait_granted"] is False

def test_a_default_boot_observates_the_window_and_waits_zero(armed, monkeypatch):
    """THE byte-identical default, pinned on the fields and not on a hope.

    With `VLLM_E2_TIER_DISK_SETTLE_MS` unset the apply point must drop exactly the
    entries it used to drop, in the same call, having slept not once. `bound == "off"`
    and a wall-clock budget are both asserted because "it is off" has to be falsifiable
    from the outside: the release ships into a box serving traffic.
    """
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    bind(worker, metadata)
    # 60 ms of "not settled" that the default boot must NOT sit through.
    unsettled_status(monkeypatch, queued=len(CACHEABLE), until_settled=0.06)
    started = time.monotonic()
    report = disk_restore.worker_before_load(worker)
    took = time.monotonic() - started
    assert report["dropped"] == len(CACHEABLE), "the drop rule is unchanged"
    assert took < 0.05, f"a default boot waited {took * 1000:.1f} ms"
    counters = disk_restore.counters()
    window = counters["last_apply_window"]
    assert window["bound"] == "off"
    assert window["budget_ms"] == 0.0
    assert window["waited_ms"] == 0.0
    assert window["settled_after_wait"] is False
    assert window["dropped_after_wait"] is False
    assert counters["settle_waits_granted"] == 0
    assert counters["settle_ms_max"] is None
    assert counters["settle_bytes_waited"] == 0
    assert counters["dropped_queued"] == len(CACHEABLE)
    assert disk_restore.apply_window()["wait"]["waits_granted"] == 0


# --- 4: the budget, and which bound decided it -------------------------------

def test_the_budget_is_proportional_to_bytes_and_named_by_the_floor_or_the_cap(
        monkeypatch):
    """The formula, at both ends and in the middle.

    Numbers are the live box's: 11,468,800 B per (span, group) per rank, and the two
    measured offers - a 9,400-token prompt at ~172 MB and a 32,000-token prompt at
    ~573 MB. The point of publishing `proportional_ms` next to `budget_ms` is that a
    reader can verify this arithmetic on the box without reading the source.
    """
    span_group = 11_468_800
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "10000")
    small = disk_restore.settle_budget(span_group * 15, 9400)
    big = disk_restore.settle_budget(span_group * 50, 32000)
    assert small["enabled"] is True
    # 9,400 tokens recompute in 2.4 s measured, and 32,000 in ~8.2 s at that rate.
    assert abs(small["recompute_ms"] - 2400.0) < 60.0, small["recompute_ms"]
    assert abs(big["recompute_ms"] - 8170.0) < 200.0, big["recompute_ms"]
    for verdict in (small, big):
        assert verdict["safety_margin"] == disk_restore.SETTLE_SAFETY_MARGIN
        assert verdict["bound"] == "proportional", verdict
        expected = (verdict["bytes_at_stake"] / verdict["read_bytes_per_s"]
                    * 1000.0 * verdict["safety_margin"])
        assert abs(verdict["budget_ms"] - expected) < 0.5, verdict
        assert verdict["budget_over_recompute"] is False, verdict

    # The floor: one slot of one group is not worth measuring, so it gets the tick.
    tiny = disk_restore.settle_budget(WIDTH, 100)
    assert tiny["bound"] == "min"
    assert tiny["budget_ms"] == disk_restore.SETTLE_MIN_MS

    # The cap: 573 MB at a 40 ms ceiling is a cap decision, and a batch counted there
    # is an operator whose cap is too small - which is the only way to say that.
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "40")
    capped = disk_restore.settle_budget(span_group * 50, 32000)
    assert capped["bound"] == "cap" and capped["budget_ms"] == 40.0

    # The deadline: a read with 5 ms left cannot be promised 40 ms of waiting.
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "5000")
    expiring = disk_restore.settle_budget(span_group * 50, 32000,
                                          seconds_left=0.005)
    assert expiring["bound"] == "deadline"
    assert expiring["budget_ms"] == pytest.approx(5.0, abs=0.01)


def test_a_zero_cap_grants_no_budget_at_all(monkeypatch):
    """Cap 0 is OFF, not "a budget of zero that still polls"."""
    monkeypatch.delenv(config.ENV_DISK_SETTLE_MS, raising=False)
    verdict = disk_restore.settle_budget(11_468_800 * 50, 32000, seconds_left=29.0)
    assert verdict["enabled"] is False
    assert verdict["budget_ms"] == 0.0 and verdict["bound"] == "off"


def test_an_absurd_cap_is_clamped_and_says_which_ceiling_clamped_it(monkeypatch):
    """1e9 ms is a typo, not a budget: the clamp is visible, never silent."""
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "1000000000")
    verdict = disk_restore.settle_budget(1000)
    assert verdict["cap_ms"] == disk_restore.SETTLE_MS_MAX
    assert verdict["ceiling_clamped"] is True


def test_a_typo_in_the_cap_is_said_out_loud_and_leaves_the_boot_off(
        armed, monkeypatch, caplog):
    """`VLLM_E2_TIER_DISK_SETTLE_MS=10s` means OFF, and says so once.

    The dangerous failure of a wait knob is a typo that silently means something else.
    Here it means the behaviour that has been running all along, and the line names the
    variable that was wrong.
    """
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "10s")
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    bind(worker, metadata)
    unsettled_status(monkeypatch, queued=len(CACHEABLE), until_settled=0.05)
    started = time.monotonic()
    assert disk_restore.worker_before_load(worker)["dropped"] == len(CACHEABLE)
    assert time.monotonic() - started < 0.05
    lines = [record.getMessage() for record in caplog.records
             if "DISK[settle] IGNORED" in record.getMessage()]
    assert len(lines) == 1, lines
    assert config.ENV_DISK_SETTLE_MS in lines[0]
    assert disk_restore.counters()["settle_waits_granted"] == 0
    assert "ignored" in disk_restore.counters()["settle_notice"]


# --- 5: the wait, when it is on ----------------------------------------------

def test_a_batch_that_settles_during_the_wait_is_applied_and_counted(
        armed, monkeypatch):
    """The whole point of the feature, end to end: wait, see the bytes, memmove them.

    The reads underneath are genuinely complete before the apply point runs; the batch
    simply REPORTS itself unsettled for 30 ms. That keeps the test deterministic while
    still driving the real path - poll, observe settled, classify with the fresh
    status - rather than a fixture where nothing could be applied either way.
    """
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "5000")
    # A slow device, so the proportional term is what sets the budget: at the real
    # 11.5 GB/s a 12 KB test offer would be granted the 0.5 ms floor and no wait could
    # ever be observed at this scale. The throughput is an env var precisely so the
    # arithmetic can be exercised at a scale a unit test can reach.
    monkeypatch.setenv(config.ENV_DISK_SETTLE_READ_BPS, "1000")
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    assert bind(worker, metadata)["ok"] is True
    settle(disk_restore._batch_id(11))
    unsettled_status(monkeypatch, queued=len(CACHEABLE), until_settled=0.03)
    started = time.monotonic()
    report = disk_restore.worker_before_load(worker)
    took = time.monotonic() - started

    assert report["applied"] == len(CACHEABLE), report
    assert report["dropped"] == 0
    assert took >= 0.02, "the wait did not actually wait"
    counters = disk_restore.counters()
    window = counters["last_apply_window"]
    assert window["settled_after_wait"] is True
    assert window["waited_ms"] >= 20.0
    assert window["bound"] in ("min", "proportional", "cap", "deadline"), window
    assert counters["settled_after_wait"] == 1
    assert counters["dropped_after_wait"] == 0
    assert counters["settle_waits_granted"] == 1
    assert counters["settle_bytes_waited"] == WIDTH * len(CACHEABLE)
    # `waited` and `read_seconds` side by side, because either number alone is
    # uninterpretable: 30 ms waited against a 2 ms read is a different story than
    # 30 ms waited against a 28 ms read.
    wait = disk_restore.apply_window()["wait"]
    assert wait["waited_ms"]["max"] >= 20.0
    assert disk_restore.apply_window()["recompute"]["read_seconds_max"] > 0.0
    assert counters["applied_settled_immediately"] == 0, \
        "bytes bought by a wait are not bytes that arrived for free"


def test_the_wait_never_outruns_its_budget(armed, monkeypatch):
    """A batch that never settles must cost the step its budget AND NOTHING MORE.

    This is the hang-shaped hazard: an unbounded wait on the engine's step thread is a
    box that stops serving. The budget is the only thing standing between the feature
    and that, so the bound is asserted on the wall clock, not on a field.
    """
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "30")
    monkeypatch.setenv(config.ENV_DISK_SETTLE_READ_BPS, "1000")
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1)
    bind(worker, metadata)
    # Never settles within the test: `until_settled` is beyond any budget.
    unsettled_status(monkeypatch, running=len(CACHEABLE), until_settled=3600.0)
    started = time.monotonic()
    report = disk_restore.worker_before_load(worker)
    took = time.monotonic() - started
    assert report["dropped"] == len(CACHEABLE), "and then it drops, as it always did"
    assert took < 0.15, f"the wait outran its 30 ms budget: {took * 1000:.1f} ms"
    counters = disk_restore.counters()
    window = counters["last_apply_window"]
    assert window["waited_ms"] >= 25.0, "it did wait, it just stopped"
    assert window["dropped_after_wait"] is True
    assert counters["dropped_after_wait"] == 1
    assert counters["settled_after_wait"] == 0
    assert counters["dropped_running"] == len(CACHEABLE)


def test_the_wait_stops_at_the_reads_own_deadline(armed, monkeypatch):
    """The fourth bound: a read with 5 ms left is not promised 5 seconds.

    `disk_read` cancels the job at its deadline; a wait that kept going past that
    would be waiting for bytes that have already been given up, which is a second bug
    stacked on the first.
    """
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "5000")
    monkeypatch.setenv(config.ENV_DISK_SETTLE_READ_BPS, "1000")
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=1, event=12)
    bind(worker, metadata)
    unsettled_status(monkeypatch, running=len(CACHEABLE), seconds_left=0.005,
                     until_settled=3600.0)
    started = time.monotonic()
    report = disk_restore.worker_before_load(worker)
    took = time.monotonic() - started
    assert report["dropped"] == len(CACHEABLE)
    assert took < 0.10, f"ignored the batch's own deadline: {took * 1000:.1f} ms"
    window = disk_restore.counters()["last_apply_window"]
    assert window["bound"] == "deadline", window
    assert disk_restore.counters()["settle_bound_deadline"] == 1


def test_the_settle_line_is_logged_once_per_boot_not_per_step(armed, monkeypatch,
                                                               caplog):
    """Three disk steps, one line. A per-step line is how this module lost its log."""
    monkeypatch.setenv(config.ENV_DISK_SETTLE_MS, "5000")
    for seed in (1, 2, 3):
        fill(armed, blocks=seed)
        worker = Worker()
        metadata = disk_meta(armed, seeds=(seed,), ram=0, event=20 + seed)
        assert bind(worker, metadata)["ok"] is True
        settle(disk_restore._batch_id(20 + seed))
        disk_restore.worker_before_load(worker)
    lines = [record.getMessage() for record in caplog.records
             if "DISK[settle] wait=ON" in record.getMessage()]
    assert len(lines) == 1, lines
    assert "cap_ms=5000.0" in lines[0]
    assert disk_restore.counters()["apply_observations"] == 3


# --- 6: the accounting hole the counters exposed -----------------------------

def test_a_burned_load_releases_its_ledger_entry(armed):
    """THE ACCOUNTING HOLE, pinned. 0.2.22 and earlier: `loads_released` never moved.

    On the live box every failure arrived as `invalid_block_ids` with an EMPTY
    `failed_recving`, and that path retired the load with a hand-rolled `_loads.pop()`
    that skipped `_drop_load`'s two side effects. The reading it produced -
    "registered=1, released=0, no_read=0, therefore the load was never registered" -
    was wrong in a way that sent the investigation to the scheduler instead of to the
    worker, and the slot->path mappings outlived the load they named.

    The SUPPRESSION decision is unchanged and re-pinned here: the request is fused and
    a re-offer is still refused. Only the bookkeeping was lying.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (base.BLOCK, True)
    slots = base.merged_cpu(manager)
    disk_restore.build_load_paths(manager, base.Meta(load_event=3, cpu=slots,
                                                     gpu=list(range(len(slots))),
                                                     reqs={3: ["req-1"]}))
    before = disk_restore.counters()
    assert before["loads_released"] == 0
    output = base.fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                                  finished_sending=set(), failed_recving=set(),
                                  invalid_block_ids={700 + n
                                                     for n in range(len(slots))})
    report = disk_restore.note_reported(manager, output)
    assert report["failures"] == 1, report
    counters = disk_restore.counters()
    assert counters["loads_released"] == 1, "the ledger entry was retired, so say so"
    assert counters["tracked_slots"] == 0, \
        "and its slot->path mappings went with it"
    assert counters["suppressions"] == 1, "the fuse is the same decision it was"
    assert counters["no_read"] == 0, \
        ("no_read stays a claim about finished reports; this load was already "
         "counted as a failure, and moving both would double-book one release")
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=1)) == (0, False), \
        "the suppression decision this fix must NOT change"


def test_a_named_failure_releases_its_ledger_entry_too(armed):
    """The `failed_recving` shape had the identical hole, on the same line of code."""
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (base.BLOCK, True)
    slots = base.merged_cpu(manager)
    disk_restore.build_load_paths(manager, base.Meta(load_event=3, cpu=slots,
                                                     gpu=list(range(len(slots))),
                                                     reqs={3: ["req-1"]}))
    output = base.fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                                  finished_sending=set(),
                                  failed_recving={"req-1"},
                                  invalid_block_ids=set())
    assert disk_restore.note_reported(manager, output)["failures"] == 1
    counters = disk_restore.counters()
    assert counters["loads_released"] == 1
    assert counters["tracked_slots"] == 0


def test_a_burned_slot_stops_naming_the_request_that_lost_its_load(armed):
    """The hazard the popped-without-dropping left behind, not just the counter.

    `_slot_paths` maps a CPU slot to the path its bytes should come from. A mapping
    whose load is gone makes the NEXT `build_load_paths` emit a path for a slot the
    fork may already be using as a host-RAM source - bytes read into the wrong slot,
    applied over a RAM hit, DMA'd to a GPU block that was never offered. The sweep
    freed these mappings eventually; "eventually" is the bug.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (base.BLOCK, True)
    slots = base.merged_cpu(manager)
    disk_restore.build_load_paths(manager, base.Meta(load_event=3, cpu=slots,
                                                     gpu=list(range(len(slots))),
                                                     reqs={3: ["req-1"]}))
    output = base.fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                                  finished_sending=set(), failed_recving=set(),
                                  invalid_block_ids={700})
    disk_restore.note_reported(manager, output)
    assert disk_restore._slot_paths == {}, "no mapping may outlive its load"
    # And the next step over the same slots emits NO disk path for them.
    following = base.Meta(load_event=4, cpu=slots, gpu=list(range(len(slots))),
                          reqs={4: ["req-1"]})
    disk_restore.build_load_paths(manager, following)
    assert not [path for path in following.load_paths if path], \
        following.load_paths


def test_a_late_finished_report_on_a_burned_load_is_no_read(armed):
    """A report of success for a load we burned is the lie `no_read` exists for.

    Before 0.2.23 the ledger entry was gone by then, so the report matched nothing and
    vanished - leaving `no_read` under-counting the one thing it counts. It is NOT a
    second release: the load was released when it burned.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (base.BLOCK, True)
    slots = base.merged_cpu(manager)
    disk_restore.build_load_paths(manager, base.Meta(load_event=3, cpu=slots,
                                                     gpu=list(range(len(slots))),
                                                     reqs={3: ["req-1"]}))
    burned = base.fakes_debug.Obj(name="KVConnectorOutput", finished_recving=set(),
                                  finished_sending=set(), failed_recving={"req-1"},
                                  invalid_block_ids=set())
    assert disk_restore.note_reported(manager, burned)["failures"] == 1
    late = base.fakes_debug.Obj(name="KVConnectorOutput",
                                finished_recving={"req-1"}, finished_sending=set(),
                                failed_recving=set(), invalid_block_ids=set())
    report = disk_restore.note_reported(manager, late)
    assert report["no_read"] == 1, report
    assert report["released"] == 0, "it was already released when it burned"
    assert report["restorations"] == 0
    counters = disk_restore.counters()
    assert counters["no_read"] == 1
    assert counters["loads_released"] == 1, "one release, not two"


# --- 7: publication surfaces -------------------------------------------------

def test_the_rank_file_section_still_fits_its_byte_cap(armed):
    """The new fields are a guest in a byte-capped file, and stay under it."""
    fill(armed, blocks=1)
    worker = Worker()
    metadata = disk_meta(armed, seeds=(1,), ram=0)
    bind(worker, metadata)
    settle(disk_restore._batch_id(11))
    disk_restore.worker_before_load(worker)
    report = disk_restore.worker_report()
    assert report["race"]["obs"] == 1
    assert report["race"]["applied_now"] == len(CACHEABLE)
    assert report["settle"]["waits"] == 0
    assert len(__import__("json").dumps(report)) <= worker_diag.MAX_CONTRIBUTION_BYTES


def test_the_snapshot_section_carries_the_window_and_the_wait(armed):
    """One `GET /plugins/kvtier/snapshot` answers the whole race question."""
    view = disk_restore.section()
    assert "apply_window" in view
    window = view["apply_window"]
    for key in ("observations", "settled_on_arrival", "windows", "wait",
                "recompute"):
        assert key in window, key
    for key in ("settled_after_wait", "dropped_after_wait", "waited_ms",
                "budget_ms", "bound", "budget_over_recompute", "last", "notice"):
        assert key in window["wait"], key


def test_every_settle_counter_has_a_unit():
    """The units rule, applied to every number this release adds.

    `test_disk_restore.py` walks `counters()` and `scheduler_section()` for this;
    this pins the NEW names by hand, so a renamed field is a failed test here and not
    a bare number in the payload.
    """
    units = disk_restore.counters_view(disk_restore.counters())["units"]
    for name in ("apply_observations", "apply_settled_on_arrival",
                 "applied_settled_immediately", "dropped_queued",
                 "dropped_running", "dropped_unattributed",
                 "settle_waits_granted", "settled_after_wait",
                 "dropped_after_wait", "settle_bytes_waited",
                 "settle_bound_min", "settle_bound_proportional",
                 "settle_bound_cap", "settle_bound_deadline",
                 "settle_budget_over_recompute"):
        assert name in units, name
        assert units[name], name


def test_the_wait_reads_its_environment_through_config_only():
    """`disk_restore` still reads NO env var itself.

    `test_no_env_var_arms_the_restore_path` pins that the GATE reads nothing new;
    this pins the same discipline for the wait - the env grammar lives in `config`,
    which is also where a bad value is caught and named. A fourth switch that could
    arm the restore path would be a fourth way to have a half-configured box, and a
    cap on a wait inside an already-armed path is not that switch.
    """
    import inspect

    source = inspect.getsource(disk_restore)
    assert "os.environ" not in source and "getenv" not in source
    for name in (config.ENV_DISK_SETTLE_MS, config.ENV_DISK_SETTLE_READ_BPS,
                 config.ENV_DISK_PREFILL_TOKENS_PER_S):
        assert name.startswith("VLLM_E2_TIER_DISK_"), name
