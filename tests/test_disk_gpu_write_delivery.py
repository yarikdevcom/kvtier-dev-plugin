r"""DELIVERY MODE (0.2.52): the plugin's copy becomes the DELIVERY, not a witness.

STEP 3 OF RETIRING THE ARENA. Until now `disk_gpu_write` was a WITNESS: it sampled one
applied entry in 64, capped itself at one write per step, kept a 0.5 s floor between
writes, and the fork's copy ran after its on the same stream and kept the last word on
every byte. Those three numbers are the reason it could be a witness at all - it cost
nothing and it was never the answer. To make the prune safe, the copy has to become the
thing that DELIVERS the bytes, which means lifting all three, and a writer that is the
delivery cannot also be allowed to be wrong quietly. This file is ordered by those two
halves.

1. THE FLAG IS A LAYER ON TOP OF THE WRITER, NOT A SECOND WRITER. Delivery mode without
   `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE` writes nothing and SAYS that is why; the
   environment grammar is the same grammar as every other device-touching flag; the
   three
   throttles move TOGETHER or not at all (a sampler at 1 with a step cap still at 1 is a
   "delivery" that writes one entry per step and reads as a broken feature).
2. THE THROTTLES REALLY GO AWAY. N applied entries in one step produce N writes and N
   entries in `written_indices`, in ONE step, with the interval floor gone - the 1:1
   correspondence the prune's membership test depends on. The witness posture's cap is
   pinned by the OTHER file (`..._the_sample_rate_and_the_step_cap_decide_...`) and must
   keep passing unchanged: this file adds a mode, it does not move the default.
3. THE FOURTH LIMIT IS NOT A THROTTLE. The pinned pool holds
   `MAX_PINNED_ROWS_PER_REGION` (2) rows and a witnessed write holds one row per lane,
   so
   the pool admits EXACTLY ONE in-flight record - lifting the three named throttles and
   leaving the pool alone would still produce a one-write-per-step writer, refusing the
   rest with `no_pinned_buffer`, which reads as a memory shortage and not as a design
   limit. So in delivery mode the COPY needs no row and the readback is what degrades:
   `delivered_unwitnessed` counts the copies that went without a witness, and the
   delivery is never refused for want of a buffer.
4. THE KILL-SWITCH FIRES ON SAME-EXECUTION EVIDENCE ONLY. 0.2.48 learned in
   `disk_gpu_shadow` that not every difference is a fault: a shadow from one execution
   paired with a store file from another legitimately differs, and every such pair used
   to be logged as corruption. This lane's version of that trap is the CLASS of the
   mismatch. A `readback_mismatch_stable` is same-execution and ours by construction
   (both readbacks were enqueued on our stream around our own copy, and the file side is
   the file this restore read), so it latches. `..._gpu_changed_on_recheck` names
   another
   stream and `..._unrechecked` names nothing, and neither may retire the feature - a
   kill-switch that fires on a race is a feature that turns itself off for no reason.
5. THE LATCH IS PROCESS-LIFETIME, ONE LINE, AND IT SURVIVES BEING ASKED TWICE. It
   restores the throttles, it makes the prune refuse, it logs once ever, and a second
   mismatch of any class is counted in `kill_switch_declined` rather than re-announced.
6. EVERY VIEW IS STABLE ACROSS ARMED, DELIVERING, LATCHED AND DEGRADED. A key that
   appears only when the mode is on is a key the scheduler process does not have, and a
   reader diffing two rank files cannot tell "not applicable" from "this build does not
   report it".

WHAT IS FAKE, STATED WHERE IT CANNOT BE MISSED: exactly what
`test_disk_gpu_write.py` states at its top - there is no CUDA device, the three
device-touching expressions are this module's own seams backed by real ctypes memory,
and
the stream is a fake context manager that journals submissions. Nothing here is faked
more than that file already fakes it; a delivery-mode test that needed a GPU would be a
test that never ran.
"""

from __future__ import annotations

import json

import pytest

# THE HELPERS AND THE FAKES THIS FILE SHARES WITH THE SIBLING. The helper functions and
# the fake classes are plain imports; `copy_seams` is imported under an alias and
# re-bound through `pytest.fixture` below, because a module-level IMPORT named
# `copy_seams` is shadowed by every test's `copy_seams` PARAMETER and ruff reads that as
# a redefinition - seven warnings for one deliberate reuse. An assignment does not trip
# the check, and the thing being reused is unchanged either way. `journal` is defined as
# a fixture here for the same reason (it is two lines over the sibling's `Journal`
# class). The alternative - copying `Journal`, `WriteWorker` and `copy_seams` into this
# file - would put two implementations of the fake GPU region in the suite, which is how
# a test ends up passing against a fake the product does not describe.
from test_disk_gpu_write import (  # noqa: F401 - re-bound as a fixture below
    CPU_BLOCK,
    GPU_BLOCK,
    GROUP_WIDTHS,
    KEY16,
    WIDTH,
    FakeRegion,
    Journal,
    Meta,
    WriteWorker,
    fill_pool,
    one_pass,
    seed_window,
    spill_file,
)
from test_disk_gpu_write import (
    copy_seams as _sibling_copy_seams,
)
from test_disk_gpu_write import (
    restore_stage as _sibling_restore_stage,
)
from test_disk_restore_verify import batch_of, job, write_window

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    config,
    disk_gpu_write,
    disk_restore,
)

# THE SIBLING'S STAGE FIXTURE, UNDER THIS FILE'S NAME (same aliasing reason as
# `copy_seams`: it arrives decorated, and an import of the plain name would be shadowed
# by every test's parameter). It is what lets the last test drive the REAL `worker_bind`
# -> `worker_before_load` path instead of a fake batch.
restore_stage = _sibling_restore_stage

# THE SIBLING'S SEAM-WRAPPER, UNDER THIS FILE'S NAME. It arrives already decorated, so
# this is a plain ALIAS and not `pytest.fixture(...)` - applying the decorator twice
# raises. And an assignment, unlike an import of that name, does not read to ruff as a
# shadowed import every time a test takes a `copy_seams` parameter.
copy_seams = _sibling_copy_seams

NARROW = 4                    # the 8-token ring: a group width of its own


@pytest.fixture
def journal():
    """The submission journal, the sibling's `Journal` and nothing else.

    Defined here rather than imported as a fixture for the same reason as `copy_seams`:
    the parameter name every test uses would shadow an imported module-level name.
    """
    return Journal()


@pytest.fixture(autouse=True)
def fresh_consumer_state():
    """A FRESH `disk_restore` FOR EVERY TEST IN THIS FILE.

    `conftest.py::reset_dev_state` snapshots and restores the state of every module that
    owns dev-side state, and `disk_restore` is deliberately NOT in that list - its own
    suite resets it through an autouse `fresh` fixture instead. A file that reaches into
    the CONSUMER (`prune_adopted_restores`, `prune_state`) therefore has to reset it
    itself, or a refusal counter incremented by one test is read as a fact about the
    boot
    by the next one. That is exactly what happened here before this fixture existed: the
    kill-switch test's `kill_switch_fired: 1` leaked into a later test's "fresh boot"
    assertion, so this file's green-ness was a property of test order.
    """
    disk_restore.reset_state_for_tests()
    yield


@pytest.fixture
def worker(journal):
    """A worker whose GPU region is WIDE ENOUGH FOR A DELIVERED STEP.

    This OVERRIDES the sibling file's `worker` for this file only, and the reason is
    that the sibling's fake region has 8 blocks and starts its writes at block 5, so a
    step with five DISTINCT destinations runs off the end of it. Under the witness
    posture that never showed (one write per step), and the module REFUSING an
    out-of-range block is correct behaviour - pinned by name in the sibling file. But a
    delivery test that asked for five
    entries would then be asserting `gpu_block_out_of_range` instead of delivery, so the
    fake grows here rather than the test shrinking to three and quietly testing a
    three-entry step's "delivery".
    """
    made = WriteWorker(journal=journal,
                       gpu={"g0": FakeRegion("g0", 32, WIDTH, journal=journal)})
    return made


@pytest.fixture
def wide_worker(worker):
    """THE SAME WIDE WORKER UNDER THE NAME THE INTEGRATION TEST USES.

    An alias, so the one test that binds three entries can say in its signature that it
    needs a region wide enough for three destinations, without a second fixture building
    a
    second fake GPU region.
    """
    return worker


def counters():
    return disk_gpu_write.counters()


@pytest.fixture
def witness(monkeypatch):
    """THE WRITER ARMED AS A WITNESS: the shipped throttles, the shipped cost.

    Every test here that lifts the throttles names this fixture too, because delivery
    mode is a LAYER on the writer and not a replacement for it - `delivery` below sets
    one environment variable and nothing else, so a test that forgot to arm the writer
    would be testing that two flags unset behave like zero flags.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    disk_gpu_write._state["env_resolved"] = False
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)
    return True


@pytest.fixture
def delivery(witness, monkeypatch):
    """THE FLAG UNDER TEST, and it is ONE environment variable.

    Deliberately NOT a fixture that also relaxes the pool, the sample or the cap: those
    are the three things the mode is supposed to move by itself, and a fixture that set
    them would make every assertion below a tautology. What it does provide is a
    `_ensure_thread` that says yes without starting a daemon (the sibling file's
    shortcut, for the same reason: determinism), and nothing else.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    disk_gpu_write._state["delivery_resolved"] = False
    return True


def many(path, n, *, event=11, **kwargs) -> dict:
    """One bound step with `n` eligible entries, each to its OWN GPU block.

    Distinct destinations because `written_indices` is a SET of step indexes and a test
    that used the same block for all of them could not tell "N copies" from "one copy
    that ran last".
    """
    return batch_of([job(path, group=kwargs.pop("group", 0),
                         gpu_block=GPU_BLOCK + offset,
                         cpu_block=kwargs.pop("cpu_block", CPU_BLOCK),
                         key16=KEY16, width=kwargs.pop("width", WIDTH))
                     for offset in range(n)], event=event)


# --- 1. the flag is a layer, and its grammar is the package's grammar ----------

def test_delivery_mode_is_off_by_default_and_off_is_the_witness_posture(witness):
    """DEFAULT OFF, and OFF MEANS THE THREE NUMBERS DID NOT MOVE.

    Asserted on the numbers and not on a boolean: `delivery_mode.effective False` is
    also
    what a module with a broken reader says. With the delivery flag unset the sample is
    64, the cap is 1 and the floor is 0.5 s - the shipped cost model, unchanged - so the
    flag-off path a user gets today is this test, not an absence of evidence.
    """
    limits = disk_gpu_write.limits_in_force(8)
    assert limits["delivery"] is False
    assert limits["sample_every_n_applied"] == disk_gpu_write.SAMPLE_EVERY_N_APPLIED
    assert limits["max_writes_per_step"] == disk_gpu_write.MAX_WRITES_PER_STEP
    assert limits["min_interval_seconds"] == disk_gpu_write.MIN_INTERVAL_SECONDS
    view = disk_gpu_write.delivery_state()
    assert view["requested"] is False and view["effective"] is False
    assert view["env_var"] == config.ENV_DISK_GPU_WRITE_DELIVERY
    assert view["raw"] is None


def test_a_typo_in_the_delivery_flag_stays_off_and_names_itself(witness, monkeypatch):
    """`tru` is not `true`, and the direction a typo moves in is the witness posture.

    Same grammar as every other device-touching flag in this package, tested because the
    failure it protects against is a boot that un-throttles a writer into being the
    delivery because someone typed `tru` at 2 a.m.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "tru")
    disk_gpu_write._state["delivery_resolved"] = False
    assert disk_gpu_write.effective_delivery_mode() is False
    view = disk_gpu_write.delivery_state()
    assert view["recognised"] is False
    assert view["raw"] == "tru"
    assert view["effective"] is False
    assert disk_gpu_write.limits_in_force(8)["sample_every_n_applied"] == \
        disk_gpu_write.SAMPLE_EVERY_N_APPLIED


def test_delivery_without_the_writer_is_inert_and_says_which_flag_is_missing(
        monkeypatch):
    """THE LAYERING, PROVEN AS A REFUSAL AND NOT AS A COMMENT.

    Delivery mode asks the writer to write everything; with the writer off there is no
    write to un-throttle, and the mode is inert. A rank file that said `effective:
    false`
    and nothing else would be read as "the flag did not take", so the reason names the
    OTHER flag - the one thing an operator has to do next.
    """
    monkeypatch.delenv(config.ENV_DISK_GPU_DIRECT_WRITE, raising=False)
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    disk_gpu_write._state["env_resolved"] = False
    disk_gpu_write._state["delivery_resolved"] = False
    view = disk_gpu_write.delivery_state()
    assert view["requested"] is True
    assert view["writer_enabled"] is False
    assert view["effective"] is False
    assert config.ENV_DISK_GPU_DIRECT_WRITE in view["reason"]
    # INERT MEANS NOTHING MOVES: not a copy, not a throttle change.
    assert disk_gpu_write.limits_in_force(8)["sample_every_n_applied"] == \
        disk_gpu_write.SAMPLE_EVERY_N_APPLIED


# --- 2. the three throttles move together, and the 1:1 correspondence holds ----

def test_the_three_throttles_move_as_one_dict_and_never_independently(
        delivery, worker):
    """ONE DICT, THREE NUMBERS, READ ONCE PER STEP.

    The trap this closes is a half-lifted writer: a sampler at 1 with the step cap still
    at 1 writes one entry and spends the rest of the step counting `throttled_step`, so
    the flag says "delivering" and the counters say "nothing happened". Every caller
    reads `limits_in_force`, so the three cannot be seen to disagree - and this test
    pins
    all three against the mode's own constants in one assertion.
    """
    limits = disk_gpu_write.limits_in_force(7)
    assert limits["delivery"] is True
    assert limits["sample_every_n_applied"] == 1, "the sampler is gone, not raised"
    assert limits["max_writes_per_step"] == 7, "the cap is the step's own size"
    assert limits["min_interval_seconds"] == 0.0, "the floor is gone, not shortened"
    # THE FLAG AND THE NUMBERS AGREE, from the same read.
    assert disk_gpu_write.delivery_state()["effective"] is True
    # AND THE SECTION PUBLISHES THE SAME THREE, because a reader of the rank file must
    # be able to see the three moved TOGETHER without importing the module.
    published = disk_gpu_write.section()["limits_in_force"]
    assert published["sample_every_n_applied"] == 1
    assert published["min_interval_seconds"] == 0.0


def test_one_step_delivers_every_entry_and_reports_each_index(delivery, worker,
                                                              tmp_path):
    """THE 1:1 CORRESPONDENCE THE PRUNE'S MEMBERSHIP TEST IS BUILT ON.

    Five eligible entries in ONE step produce FIVE writes and a `written_indices` set
    holding exactly the five step indexes. This is the property `prune_adopted_restores`
    consumes: it prunes index `i` only if `i` is in that set, so the set must name every
    entry the copy went into and no entry it did not. A writer that sampled, or capped,
    or floored would leave indexes OUT of the set and the prune would leave those
    destinations to the fork - correct, but not a delivery.

    THE POOL IS FILLED AND THE TEST STILL PASSES FOR THE RIGHT REASON: five entries need
    readback rows for at most one of them (`MAX_PINNED_ROWS_PER_REGION` is 2 and a
    witnessed write holds one row per lane), so four of these five copies go WITHOUT a
    witness. That is the next test's subject; here it is what lets one step be delivered
    at all.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x71" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    disk_gpu_write._state["last_write_mono"] = None
    answer = disk_gpu_write.observe_apply(worker, many(path, 5), {})
    assert answer["delivery_mode"] is True
    assert answer["seen"] == 5 and answer["written"] == 5
    assert answer["written_indices"] == {0, 1, 2, 3, 4}
    assert answer["reason"] is None
    # THE COPIES REALLY HAPPENED, at five DISTINCT destinations, not one copy last.
    # `Journal.of("h2d")` is every host->device submission, and the destination is the
    # third field of the record (`(kind, region, block, source, stream)`).
    destinations = [entry[2] for entry in worker.journal.of("h2d")]
    assert sorted(destinations) == [GPU_BLOCK + i for i in range(5)]
    # AND NOTHING WAS THROTTLED. These three counters are the old posture's, and a
    # delivery step must leave all three at zero - the assertion that says the mode
    # lifted the throttles rather than raising one of them.
    for name in ("sampled_out", "throttled_step", "throttled_interval"):
        assert counters()[name] == 0, name
    assert counters()["delivery_writes"] == 5
    assert counters()["delivery_steps"] == 1


def test_the_interval_floor_does_not_hold_the_second_step_back(delivery, worker,
                                                               tmp_path):
    """THE THIRD THROTTLE, ON ITS OWN: two steps, one tick apart, both delivered.

    The witness posture keeps `MIN_INTERVAL_SECONDS` (0.5 s) between writes, which is
    fine for a sampled witness and fatal for a step that has to deliver its own restore.
    With the mode on and the clock UNTOUCHED (no monkeypatched time, no private clock
    reset), a second immediate step still writes. This is the only test that may say so:
    the sibling file's `one_write` resets the private clock, which is exactly the help
    this test is not allowed.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x72" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    first = disk_gpu_write.observe_apply(worker, many(path, 1, event=11), {})
    assert first["written"] == 1
    second = disk_gpu_write.observe_apply(worker, many(path, 1, event=12), {})
    assert second["written"] == 1, "the floor held a delivery step back"
    assert counters()["throttled_interval"] == 0


def test_a_witness_step_still_caps_at_one_and_a_delivery_step_does_not(
        witness, worker, tmp_path, monkeypatch):
    """THE COMPARISON IN ONE TEST, SO THE MODE CANNOT BE AN ENVIRONMENT THEIGNORERS.

    The same five-entry step, the same worker, the same pool: witness posture delivers
    ONE
    and counts `throttled_step`; delivery posture delivers FIVE and counts none. If the
    mode were read at import, or cached past the env reset, or consulted only in
    `_write_one`, one half of this test fails.

    The witness half is asserted HERE as well as in the sibling file because the thing
    under test is the DIFFERENCE, and a difference is not pinned by pinning one side.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x73" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 1)
    disk_gpu_write._state["last_write_mono"] = None
    witness_answer = disk_gpu_write.observe_apply(worker, many(path, 5, event=11), {})
    assert witness_answer["written"] == 1, "the shipped cap is one per step"
    assert witness_answer["written_indices"] == {0}
    assert counters()["throttled_step"] == 4, counters()

    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    disk_gpu_write._state["delivery_resolved"] = False
    worker.complete(11)
    one_pass(worker)                       # release the witness step's row
    fill_pool(worker)
    disk_gpu_write._state["last_write_mono"] = None
    delivered = disk_gpu_write.observe_apply(worker, many(path, 5, event=12), {})
    assert delivered["written"] == 5
    assert delivered["written_indices"] == {0, 1, 2, 3, 4}


# --- 3. the fourth limit: the pool may cost a witness, never a delivery -------

def test_an_empty_pool_costs_the_readback_and_never_the_delivery(delivery, worker,
                                                                 tmp_path):
    """THE FOURTH LIMIT, WHICH IS NOT A THROTTLE AND WAS NOT IN THE BRIEF.

    The pool holds `MAX_PINNED_ROWS_PER_REGION` rows per region and a witnessed write
    holds one row per lane, so it admits EXACTLY ONE in-flight record. Un-lifting the
    three named throttles and leaving this alone produces a writer that still writes one
    entry per step and refuses the rest by name (`no_pinned_buffer`) - a MISSING
    DELIVERY
    that reads as a memory shortage, in a boot where the prune has already removed the
    fork's copy. That is the configuration this test exists for.

    THE POOL IS DELIBERATELY EMPTY, and that is the production shape being tested rather
    than a shortcut: this module's thread allocates rows only when it runs a pass, so a
    step that arrives before the thread has ever run finds nothing - which is the 0.2.43
    cold-pool bug, now on the delivery path. So: every entry is delivered, and
    `delivered_unwitnessed` says how many went without a witness. Nothing is refused.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x74" * 64)
    seed_window(worker, path)
    # NO `fill_pool`: the pool has never been filled, which is what a real first step
    # after a boot looks like.
    disk_gpu_write._state["last_write_mono"] = None
    answer = disk_gpu_write.observe_apply(worker, many(path, 4), {})
    assert answer["written"] == 4, "a cold pool refused a DELIVERY: the fourth limit"
    assert answer["written_indices"] == {0, 1, 2, 3}
    assert answer["reason"] is None
    assert counters()["no_pinned_buffer"] == 0, "a delivery is not a refusal"
    assert counters()["delivered_unwitnessed"] == 4
    assert counters()["delivery_writes"] == 4
    # AND THE COPIES ARE AT FOUR DISTINCT WINDOWS, so "4" is four copies and not a
    # counter agreeing with itself.
    assert len(worker.journal.of("h2d")) == 4
    # NOTHING IS IN FLIGHT: an unwitnessed delivery holds no rows and leaves the
    # module's thread nothing to wait for, which is what lets one apply point deliver a
    # whole step.
    assert disk_gpu_write.pool_view()["in_flight"] == 0


def test_a_delivery_that_cannot_copy_is_still_a_named_refusal(delivery, worker,
                                                              tmp_path,
                                                              monkeypatch):
    """A REFUSAL IS STILL A REFUSAL: the mode does not turn a failed copy into a write.

    If `delivery` swallowed copy errors the module would report a delivery that never
    happened, and `written_indices` would name an index whose window nobody touched -
    which the prune would then remove from the fork's lists. That is the wrong-answer
    shape this whole feature has to avoid, so the copy's failure is asserted to be a
    named refusal that leaves the index OUT of the covered set.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x75" * 64)
    seed_window(worker, path)

    def boom(*args, **kwargs):
        raise RuntimeError("the copy raised")

    monkeypatch.setattr(disk_gpu_write, "_copy_host_to_device", boom)
    disk_gpu_write._state["last_write_mono"] = None
    answer = disk_gpu_write.observe_apply(worker, many(path, 2), {})
    assert answer["written"] == 0
    assert answer["written_indices"] == set(), "a failed copy must not be covered"
    assert answer["reason"] == "write_failed"
    assert counters()["delivery_writes"] == 0
    assert counters()["delivered_unwitnessed"] == 0


# --- 4. the kill-switch fires on same-execution evidence only ------------------

def armed_one(path, **kwargs) -> dict:
    """One bound step with ONE eligible entry."""
    return many(path, 1, **kwargs)


def _mismatch_verdict(worker, seams) -> str:
    """Complete the load event and run one check pass; return the verdict's category."""
    worker.complete(11)
    results = one_pass(worker)["results"]
    assert len(results) == 1
    return results[0].get("category")


def test_a_stable_mismatch_fires_the_kill_switch_and_names_the_copy_as_wrong(
        delivery, worker, tmp_path, copy_seams):
    """THE KILL-SWITCH'S REASON, AND IT IS THE SAME-EXECUTION ONE.

    Only the COPY's destination is shifted, by the test; the readbacks still read the
    window the entry named. Both readbacks were enqueued on our own stream around our
    own copy inside one `start_load_kv`, and the file side is the file this restore read
    into
    the slot this copy read from - so there is no second execution anywhere in the
    comparison and no recomputation in it. The window is exactly what it was before we
    touched it, which can only mean OUR copy landed in the wrong window. That is the
    failure the prune cannot survive, so it latches.

    THE SHIFT IS THE TEST'S OWN MUTATION and not a state production would produce, which
    is why this belongs to the negative-control tradition in the sibling file: a
    kill-switch that never fires is indistinguishable from a kill-switch that was never
    wired up, and only a real mismatch can tell them apart.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x76" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    copy_seams["shift"] = 1
    disk_gpu_write._state["last_write_mono"] = None
    assert disk_gpu_write.observe_apply(worker, armed_one(path), {})["written"] == 1
    assert _mismatch_verdict(worker, copy_seams) == "readback_mismatch_stable"
    view = disk_gpu_write.kill_switch_view()
    assert view["fired"] is True
    assert view["category"] == "readback_mismatch_stable"
    # THE EVIDENCE IS KEPT, because a latch with no evidence is a bug report that cannot
    # be investigated: which window, which group, which key, what shape.
    assert view["evidence"]["gpu_block"] == GPU_BLOCK
    assert view["evidence"]["shape"] == "window-unchanged"
    assert view["evidence"]["delivery"] is True, "the write judged was a delivery write"
    assert view["effect"], "the latch must say what it changed"
    assert counters()["kill_switch_fired"] == 1


def test_a_window_rewritten_by_another_stream_does_NOT_fire_the_kill_switch(
        delivery, worker, tmp_path, copy_seams):
    """THE TRAP, AND IT IS 0.2.48'S TRAP IN A NEW LANE.

    `disk_gpu_shadow` learned that a shadow from one execution paired with a store file
    from another legitimately differs (two fp8 recomputations of one key are not
    bit-identical) and used to call every such pair corruption. This lane cannot meet
    that exact pair - it compares a GPU window with the file whose bytes it just copied,
    and a
    spill file is content-addressed and never overwritten - but it has the SAME SHAPE of
    benign difference: a THIRD stream writes the window between our two readbacks. The
    after-state is then neither ours nor the before-state, which requires work on
    another
    stream and says NOTHING about where our copy went.

    A kill-switch that fired here would retire the prune because a compute kernel
    touched
    a block, which is a race this package expects, documents and does not act on. So the
    mismatch must be counted, must be classified, and must NOT latch - and
    `kill_switch_declined` must record that it was offered and refused, because
    "declined" and "never asked" are different readings of a quiet switch.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x77" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    # The SECOND `d2h` is the post-copy readback (the first is the pre-copy one), so a
    # churn there is another stream rewriting the window between our two reads.
    copy_seams["churn_on"] = 2
    disk_gpu_write._state["last_write_mono"] = None
    assert disk_gpu_write.observe_apply(worker, armed_one(path), {})["written"] == 1
    assert _mismatch_verdict(worker, copy_seams) == \
        "readback_mismatch_gpu_changed_on_recheck"
    assert disk_gpu_write.kill_switch_engaged() is False, \
        "the kill-switch fired on a race: this is the feature-killer"
    assert counters()["kill_switch_fired"] == 0
    assert counters()["kill_switch_declined"] == 1
    # THE MODE SURVIVES IT, which is the operator-visible half of the same fact.
    assert disk_gpu_write.delivery_state()["effective"] is True


def test_a_mismatch_with_no_before_state_is_declined_and_not_blamed(delivery, worker,
                                                                    tmp_path,
                                                                    copy_seams,
                                                                    monkeypatch):
    """UNATTRIBUTABLE IS NOT A FAULT: no pre-copy readback, no claim about our copy.

    The module's own comment on this branch says the disagreement is real but it "cannot
    say whether another stream moved the window or its own copy missed". Firing here
    would
    be GUESSING, and the asymmetry that decides it is the opposite of
    `disk_gpu_shadow._attribute_pair`'s: there, ambiguity must never be called benign
    because being wrong costs a missed bug report; here, being wrong costs the FEATURE
    for
    the rest of the boot. So an unattributable disagreement is counted in
    `kill_switch_declined` and the switch stays unfired.

    THE MISSING BEFORE-STATE IS THE PRODUCTION CONDITION, not a contrivance: the
    record's
    pre-copy lane can be absent (a row that came back unlanded is refused at the read),
    and `_lane_rows(record["rows"], "pre")` returning nothing is exactly that. The copy
    is ALSO shifted, so the after-state really disagrees - without that half the verdict
    would
    be `equal` and the test would prove nothing about the unattributable branch.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x78" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    copy_seams["shift"] = 1               # the after-state disagrees
    real_read = disk_gpu_write._read_rows
    reads = {"n": 0}

    def unreadable_pre(rows, width):
        # THE FIRST `_read_rows` INSIDE THE CHECK is the pre-copy lane, the second the
        # post-copy one (`_check_record_inner` reads them in that order). Patching the
        # READER and not `_lane_rows` matters: `_lane_rows` is also on the write path,
        # so emptying the pre lane there refuses the WRITE and the test asserts a
        # refusal instead of the unattributable verdict it is about.
        reads["n"] += 1
        if reads["n"] == 1:
            return {"ok": False, "kind": "readback_unreadable",
                    "reason": "the pre-copy readback row is unreadable"}
        return real_read(rows, width)

    monkeypatch.setattr(disk_gpu_write, "_read_rows", unreadable_pre)
    disk_gpu_write._state["last_write_mono"] = None
    assert disk_gpu_write.observe_apply(worker, armed_one(path), {})["written"] == 1
    assert _mismatch_verdict(worker, copy_seams) == "readback_mismatch_unrechecked"
    assert disk_gpu_write.kill_switch_engaged() is False, \
        "the switch fired on evidence that attributes nothing"
    assert counters()["kill_switch_fired"] == 0
    assert counters()["kill_switch_declined"] == 1
    # THE ARITHMETIC HOLDS ACROSS ALL THREE CATEGORIES: every mismatch that arrived was
    # either accepted as a fault or declined, and this branch returns EARLY from the
    # check, so it is the one place the decline could have been forgotten. `fired +
    # declined == mismatch` is a property of the module, not of this category.
    counts = counters()
    assert counts["kill_switch_fired"] + counts["kill_switch_declined"] == \
        counts["readback_mismatch"], counts


# --- 5. the latch: process-lifetime, one line, and it changes behaviour --------

def test_the_latch_restores_the_throttles_and_the_prune_refuses(
        delivery, worker, tmp_path, copy_seams, monkeypatch):
    """WHAT FIRING ACTUALLY DOES, which is the only reason a latch exists.

    A latch that only sets a boolean is a diagnostic, not a control. Firing must (a) put
    the three throttles back, so the plugin stops being the delivery, and (b) make the
    prune refuse every later step, so the fork's copy comes back. Those two halves are
    the whole safety story: with both, a proven-bad writer is out of the delivery path
    and the step is served the way it was before this feature existed.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x79" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    copy_seams["shift"] = 1
    disk_gpu_write._state["last_write_mono"] = None
    disk_gpu_write.observe_apply(worker, armed_one(path), {})
    _mismatch_verdict(worker, copy_seams)
    assert disk_gpu_write.kill_switch_engaged() is True

    # (a) THE THROTTLES ARE BACK, read from the same dict every caller uses.
    limits = disk_gpu_write.limits_in_force(8)
    assert limits["delivery"] is False
    assert limits["sample_every_n_applied"] == disk_gpu_write.SAMPLE_EVERY_N_APPLIED
    assert limits["max_writes_per_step"] == disk_gpu_write.MAX_WRITES_PER_STEP
    assert limits["min_interval_seconds"] == disk_gpu_write.MIN_INTERVAL_SECONDS
    # THE FLAG STILL SAYS IT WAS ASKED, and `reason` says why it is not in force. A
    # reader who set two environment variables must be able to see which one lost.
    view = disk_gpu_write.delivery_state()
    assert view["requested"] is True
    assert view["kill_switch_latched"] is True
    assert view["effective"] is False
    assert "KILL-SWITCH" in view["reason"]

    # (b) THE PRUNE REFUSES, under its own name, for the rest of this process. The prune
    # flag is raised first: with it OFF the answer is the flag-off `attempted: False`,
    # and a test that read that as the kill-switch would be reading the wrong refusal.
    monkeypatch.setenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, "1")
    disk_restore._state["prune_switch"] = None
    refusal = disk_restore.prune_adopted_restores(None, {"jobs": {}}, set())
    assert refusal["ok"] is False
    assert refusal["reason_kind"] == "kill_switch_fired"
    assert disk_restore.prune_state()["kill_switch_fired"] is True
    assert disk_restore.prune_state()["delivery_mode"] is False

    # (c) A LATER WRITE IS NOT DELIVERED ANY MORE: the mode is off, so the cap is one
    # and a five-entry step is throttled again. This is the latch acting on the writer.
    # THE
    # SAMPLE IS LIFTED FIRST because "off" also means the shipped rate of 64, and a step
    # whose entries were all `sampled_out` would report `written == 0` - a number the
    # cap did not produce. The rate is pinned by its own test in the sibling file.
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 1)
    copy_seams["shift"] = 0
    worker.complete(12)
    one_pass(worker)
    fill_pool(worker)
    disk_gpu_write._state["last_write_mono"] = None
    later = disk_gpu_write.observe_apply(worker, many(path, 5, event=20), {})
    assert later["delivery_mode"] is False
    assert later["written"] == 1, "a latched writer is still delivering"


def test_the_latch_says_one_line_ever_and_counts_every_later_mismatch(
        delivery, worker, tmp_path, copy_seams, monkeypatch, caplog):
    """ONE LINE, MANY MISMATCHES: the log is per LATCH and the counter is per event.

    The mismatch check runs on the module's daemon thread, so an unbounded line here is
    one line per checked record forever - the engine's log filled by a diagnostic that
    already told you what it found. The counter keeps counting, because the operator
    needs to know whether this is one bad entry or every entry.
    """
    # THE SAMPLE LIFTED FOR THE LATER STEPS. Once the switch fires the mode is off, and
    # off means the SHIPPED sample rate of 64 - so without this the second and third
    # steps would be `sampled_out`, no mismatch would be produced, and the assertion
    # below would be satisfied by the throttles rather than by the latch. The rate is
    # pinned elsewhere.
    monkeypatch.setattr(disk_gpu_write, "SAMPLE_EVERY_N_APPLIED", 1)
    path = write_window(str(tmp_path / "kvslot-window"), b"\x7a" * 64)
    seed_window(worker, path)
    copy_seams["shift"] = 1
    for event in (11, 12, 13):
        fill_pool(worker)
        disk_gpu_write._state["last_write_mono"] = None
        disk_gpu_write.observe_apply(worker, armed_one(path, event=event), {})
        worker.complete(event)
        one_pass(worker)
    lines = [line for line in caplog.records if "KILL-SWITCH" in str(line.message)]
    assert len(lines) == 1, ("the latch announced itself "
                             f"{len(lines)} times, not once")
    # ONE-SHOT, not one-per-hit: the latch keeps the FIRST fault's evidence.
    assert counters()["kill_switch_fired"] == 1
    # The later stable mismatches are REAL and still counted: one-shot logging must not
    # become one-shot counting, or a systemic addressing bug reads as a single event.
    assert counters()["readback_mismatch_stable"] == 3, counters()


def test_firing_is_idempotent_and_the_first_evidence_wins(delivery, worker, tmp_path,
                                                          copy_seams):
    """THE LATCH KEEPS THE EVIDENCE THAT FIRED IT.

    A latch that overwrote itself with the second mismatch would lose the record of the
    first - and the first is the one that caused the retirement. Re-firing must be a
    no-op on the evidence and must not double-count the event.
    """
    path = write_window(str(tmp_path / "kvslot-window"), b"\x7b" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    copy_seams["shift"] = 1
    disk_gpu_write._state["last_write_mono"] = None
    disk_gpu_write.observe_apply(worker, armed_one(path), {})
    _mismatch_verdict(worker, copy_seams)
    first = disk_gpu_write.kill_switch_view()["wall"]
    again = disk_gpu_write.fire_kill_switch("a second opinion", "should not replace",
                                            category="readback_mismatch_stable")
    assert again is False, "a latched switch is not re-armed"
    view = disk_gpu_write.kill_switch_view()
    assert view["wall"] == first
    assert view["reason"] != "a second opinion"
    assert counters()["kill_switch_fired"] == 1


# --- 6. every view is stable across armed, delivering, latched and degraded ----

def test_every_view_has_the_same_keys_armed_delivering_and_latched(
        delivery, worker, tmp_path, copy_seams):
    """THE KEY SET DOES NOT MOVE, so a reader can diff two rank files.

    Three states are compared: the writer armed as a witness (the flag off), the writer
    delivering, and the writer latched off by its own readback. The SCHEDULER PROCESS is
    the fourth state and the one that would break this: it never writes a window, never
    starts a thread and can never fire a switch, and if `kill_switch` or `delivery_mode`
    appeared only in the armed case the scheduler's rank file would look like a build
    that does not report them.

    THE LATCHED CASE IS BUILT BY FIRING IT FOR REAL, not by hand-editing the state, so a
    latch path that forgot to publish something is caught here rather than in the field.
    """
    def shape():
        section = disk_gpu_write.section()
        return (set(section), set(section["delivery_mode"]),
                set(section["kill_switch"]))

    # THE WITNESS POSTURE'S OWN VIEW, captured before the mode is on, as the baseline
    # the delivering and latched views are compared against.
    off = disk_gpu_write.delivery_state()
    armed_off = disk_gpu_write.kill_switch_view()
    delivering = disk_gpu_write.delivery_state()
    assert set(delivering) == set(off), "the delivery view's keys moved with the mode"
    assert set(disk_gpu_write.kill_switch_view()) == set(armed_off)

    path = write_window(str(tmp_path / "kvslot-window"), b"\x7c" * 64)
    seed_window(worker, path)
    fill_pool(worker)
    copy_seams["shift"] = 1
    disk_gpu_write._state["last_write_mono"] = None
    disk_gpu_write.observe_apply(worker, armed_one(path), {})
    _mismatch_verdict(worker, copy_seams)
    assert disk_gpu_write.kill_switch_engaged() is True

    keys, delivery_keys, kill_keys = shape()
    fresh_section = disk_gpu_write.section()
    assert set(fresh_section) == keys, "the section's key set moved after a latch"
    assert set(fresh_section["delivery_mode"]) == delivery_keys
    assert set(fresh_section["kill_switch"]) == kill_keys
    # THE FAT CASE TOO: a mismatch sample, full-length reasons and live geometry is the
    # payload the trim ladder has to survive, and the latched state is exactly the state
    # where a section that silently vanishes costs the most.
    for index in range(disk_gpu_write.MAX_REASON_SAMPLES + 4):
        disk_gpu_write._reason(f"seed{index}", "a condition described at length " * 6)
    fat = disk_gpu_write.section()
    assert set(fat) == keys, "the key set moved when there was a lot to report"
    assert set(fat["delivery_mode"]) == delivery_keys
    assert set(fat["kill_switch"]) == kill_keys
    size = len(json.dumps(fat, separators=(",", ":"), default=str).encode())
    assert size <= disk_gpu_write.MAX_SECTION_BYTES, f"{size} B over budget"


def test_the_prune_side_publishes_the_mode_and_the_latch_in_every_process(
        monkeypatch):
    """THE CONSUMER'S VIEW, WITHOUT A WORKER, A STREAM OR A SINGLE WRITE.

    `disk_restore.prune_state()` is read in the scheduler process, where nothing here
    can
    ever be true: no apply point, no copy, no mismatch, no latch. The keys must still be
    there and still be FALSE, because the reader that learns the most from them is the
    one
    trying to work out why the prune is doing nothing - and the honest answer in this
    process is "delivery is not in force here", which is a key and not an absence.
    """
    monkeypatch.delenv(config.ENV_DISK_GPU_WRITE_DELIVERY, raising=False)
    monkeypatch.delenv(config.ENV_DISK_GPU_DIRECT_WRITE, raising=False)
    disk_gpu_write._state["env_resolved"] = False
    disk_gpu_write._state["delivery_resolved"] = False
    state = disk_restore.prune_state()
    for key in ("delivery_mode", "delivery", "kill_switch_fired", "kill_switch",
                "candidates", "covered_entries", "uncovered_entries",
                "direct_writes"):
        assert key in state, key
    assert state["delivery_mode"] is False
    assert state["kill_switch_fired"] is False
    assert state["delivery"]["effective"] is False
    assert state["kill_switch"]["fired"] is False
    # THE NEW REFUSAL NAMES ARE PUBLISHED AT ZERO, in the closed vocabulary, so a reader
    # can tell "never refused this way" from "this build has no such refusal".
    assert state["refused"]["not_delivered"] == 0
    assert state["refused"]["kill_switch_fired"] == 0
    assert set(state["refused"]) == set(disk_restore.PRUNE_REFUSALS)
    # AND EVERY NEW KEY HAS A UNIT, because an un-united number is a number a reader has
    # to guess the meaning of.
    for key in ("prune_adopted_entries", "prune_uncovered_entries",
                "prune_direct_writes", "prune_unwitnessed_entries"):
        assert str(disk_restore.counters()["units"].get(key, "")).strip(), key


# --- 7. THE REAL APPLY POINT, end to end: delivery, prune, and the event -------

def test_the_real_apply_point_delivers_every_entry_and_prunes_only_those(
        restore_stage, monkeypatch, wide_worker, tmp_path):
    """THE WHOLE CHAIN, THROUGH THE PRODUCT, WITH NOTHING STANDING IN.

    Every test above reaches delivery through a fake batch or the `delivered` stand-in.
    This one runs the real `worker_bind` -> `worker_before_load`, with the prune flag
    and the delivery flag both raised and the real writer computing `written_indices`
    itself.
    It is the test that would fail if the two modules' index spaces disagreed, if the
    covered set never reached the prune, or if the prune pruned an entry the writer had
    not touched - the three ways this feature could be wrong while every unit test
    stayed
    green.

    THREE ENTRIES IN ONE STEP, deliberately: with the witness posture this step yields
    one write and two `throttled_step`. Here it must yield three copies, three indexes,
    and a prune of exactly two of them - the LAST one kept by the emptied-step guard,
    because a step the prune emptied records no load event and nothing would then prove
    these copies
    landed before the next forward read the windows.

    `wide_worker` rather than the sibling's `worker` for the reason the fixture states:
    three distinct destinations must all be real blocks of the fake region.
    """
    monkeypatch.setenv(config.ENV_DISK_GPU_DIRECT_WRITE, "1")
    monkeypatch.setenv(config.ENV_DISK_GPU_WRITE_DELIVERY, "1")
    monkeypatch.setenv(config.ENV_DISK_PRUNE_RESTORED_LOADS, "1")
    disk_gpu_write._state["env_resolved"] = False
    disk_gpu_write._state["delivery_resolved"] = False
    disk_restore._state["prune_switch"] = None
    monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)

    paths = [spill_file(restore_stage, seed, 0) for seed in (1, 2, 3)]
    meta = Meta(load_event=11, cpu=[0, 1, 2], gpu=[0, 1, 2], paths=list(paths),
                reqs={11: ["req-1"]})
    disk_restore.worker_bind(wide_worker, meta)
    wide_worker.bind_connector_metadata(meta)
    # ONE ROW PAIR IN THE POOL, filled by the module's own production function. With two
    # rows the FIRST write is witnessed and the other two go `delivered_unwitnessed` -
    # which is the fourth limit doing exactly what 0.2.52 says it does, and it is also
    # what gives the readback below something to check. Leaving the pool empty would
    # make the whole step unwitnessed and the last assertion vacuous.
    assert fill_pool(wide_worker) == 2
    report = disk_restore.worker_before_load(wide_worker)

    # THE WRITER DELIVERED ALL THREE, into three distinct windows.
    assert counters()["delivery_writes"] == 3, disk_gpu_write.section()["reasons"]
    assert counters()["throttled_step"] == 0
    destinations = sorted(entry[2] for entry in wide_worker.journal.of("h2d"))
    assert destinations == [0, 1, 2]
    # THE PRUNE REMOVED TWO OF THREE, and the survivor is the guard's doing, not a
    # refusal.
    assert report["pruned"] == 2
    assert report["submitted"] is True, "the kept entry means the fork still copies"
    assert meta.load_cpu_blocks == [2] and meta.load_gpu_blocks == [2]
    state = disk_restore.prune_state()
    assert state["entries"] == 2
    assert state["covered_entries"] == 2
    assert state["kept_last_entries"] == 1
    assert state["emptied_steps"] == 0
    # THE INVARIANT, on the real path: nothing was pruned that the writer had not
    # covered.
    assert state["unwitnessed_entries"] == 0

    # AND THE EVENT WITNESSES THE COPIES, which is the whole reason the entry was kept.
    # The fork runs AFTER us on the same stream, so its `event.record(stream)` can only
    # complete after our three copies; the readback verdicts then come back EQUAL, which
    # is the positive proof that the delivery landed where the entry said it would.
    wide_worker.start_load_kv()
    assert wide_worker.journal.kinds[-1] == "fork-copy"
    wide_worker.complete(11)
    results = one_pass(wide_worker)["results"]
    # EXACTLY ONE verdict, and that is the fourth limit, not a bug: the pool admitted
    # one row pair, so one write carried a witness and two went unwitnessed.
    assert len(results) == 1, counters()["delivered_unwitnessed"]
    assert results[0]["equal"] is True, results[0]
    with open(paths[0], "rb") as handle:
        raw = handle.read(WIDTH)
    assert wide_worker.gpu.block_bytes(0)[:GROUP_WIDTHS[0]] == raw[:GROUP_WIDTHS[0]]
