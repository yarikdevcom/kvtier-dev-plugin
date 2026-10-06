r"""THE DISK TIER'S BOOT DEFAULT (0.2.33): `KVTIER_DISK_BOOT_ENABLE`.

The feature is one flag with a default of ON, and these tests are ordered by what an
operator would notice first, because the failure this release exists to end is not a
crash - it is a boot that serves nothing while reporting that truth honestly.

1. **The default boot arms and serves with NO POST.** The headline: run the real boot
   (install, guard, tier arm, restore fit, arena verdict, then the disk default) against
   a configured disk tier and the box comes up `serving_as=disk_authority` with an armed
   spill path that is LATCHED (0.2.35: no expiry) at the maximum legal TTL. Nobody
   POSTed anything.
2. **`KVTIER_DISK_BOOT_ENABLE=0` is today's boot, exactly.** Disarmed spill, posture
   down, `serving_as=off`, no refusal recorded against the disk default, and the two dev
   verbs still the only lever. A test that inherited the flag from a developer's shell
   would be testing the wrong product, which is why `conftest.ENV_VARS` deletes it.
3. **A manual disarm always beats the boot default.** A human's `{"op":"disarm"}` (or
   `{"op":"authority","enabled":false}`) latches a VETO, and the boot decision then
   refuses to re-arm - the property the brief makes non-negotiable, tested through the
   VERB and not by poking the latch. Since 0.2.35 the veto guards the boot decision
   rather than a renewal tick, which makes it stronger: with the arm a latch, a
   re-arm would be forever instead of two hours.
4. **THE ARM IS A LATCH (0.2.35).** No expiry, no renewal, and an engine restart inside
   one kernel uptime cannot inherit an arm file from an earlier boot - the boot's own
   decision deletes it. This is where the deleted renewal suite used to live.
5. **Every failure leaves the boot disarmed and says so once.** A failed enable must
   never fail an engine boot and never leave a partially-armed fleet.
6. **THE PROHIBITION (`test_the_prohibition_...`).** A boot-time re-arm from a persisted
   `disk-spill-arm.json` is forbidden by the sibling llm-api repo's 2026-10-03 incident
   entry, and it is the one thing this feature could have been built to do by accident.
   The guard is behavioural (a file from a previous boot changes nothing about this
   boot's decision) AND structural (the module never names the arm file at all).

The real boot is driven through the same fake fork the auto-boot suite uses, so the
preconditions the disk default reads are the boot's own measurements, not fixtures built
for it. Where a test needs one gate to fail, it fails THAT gate rather than mocking the
decision.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import time

import conftest
import fixtures_pr56810 as PR  # noqa: F401 - imported for its module-level setup
import pytest
import test_backport as TB
import test_disk_restore as base

from kvtier_dev_plugin import (
    backport,
    config,
    disk_boot,
    disk_index,
    disk_restore,
    disk_spill,
    disk_store,
    guard_mode,
)

SWEEP = ["", "0", "1", "false", "true", "garbage", "no", "off"]

# The restore layer's own autouse fixtures, re-exported so they apply HERE: without
# `fresh` the serve posture and the offer counters leak from the previous test, and a
# suite about what a boot RAISES would be reading somebody else's posture.
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or state leaks
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp


# --- state hygiene -------------------------------------------------------------

@pytest.fixture(autouse=True)
def disk_boot_state():
    """A fresh boot's view of the flag, and the arms it may raise, in and out.

    The disk default is the first posture in this package an operator can reach WITHOUT
    asking for it, so a leak here is not a flaky counter - it is the next test finding a
    serving tier it never armed. Restored, not cleared: the boot also writes
    `backport`/`guard_mode` state, and that belongs to whoever ran before.
    """
    saved_boot = dict(disk_boot._state)
    saved_bp = dict(backport._state)
    saved_verdicts = dict(backport._state["method_verdicts"])
    saved_switch = dict(backport._state["switch_verdicts"])
    saved_guard = dict(guard_mode._state)
    backport.reset_state_for_tests()
    guard_mode.reset_state_for_tests()
    disk_boot.reset_state_for_tests()
    yield
    disk_boot._state.clear()
    disk_boot._state.update(saved_boot)
    backport._state.clear()
    backport._state.update(saved_bp)
    backport._state["method_verdicts"] = saved_verdicts
    backport._state["switch_verdicts"] = saved_switch
    guard_mode._state.clear()
    guard_mode._state.update(saved_guard)
    disk_restore.uninstall()


@pytest.fixture
def tier(monkeypatch, tmp_path):
    """A configured SSD tier: a store root, a diagnostics directory, a journal home.

    The three environment facts the boot default reads through their owners. The journal
    gets its own directory because the default's gate asks whether a journal line could
    EVER land where it lives, and an unwritable one is a real reason to stay disarmed.
    """
    root = tmp_path / "spill"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    journal = tmp_path / "journal"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(journal))
    disk_spill.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    return {"root": str(root), "diag": str(diag), "journal": str(journal),
            "path": tmp_path}


@pytest.fixture
def good_pre(tier):
    """The boot's own verdicts, in the shape `boot_enable` is handed, ALL PASSING.

    Every field here is a fact the real boot measures (the tier's arm, the restore
    layer's fit, the arena's verdict, the journal directory's writability). Tests that
    need one gate to fail override THAT gate, so a refusal is always attributed to the
    link that produced it and never to a fixture.
    """
    journal_state = disk_index.journal_directory_state()
    assert journal_state.get("writable") is True, journal_state
    return {"inert": False, "tier_armed": True, "restore_fitted": True,
            "arena_state": "ok", "arena_reason": None,
            "journal_writable": journal_state.get("writable"),
            "journal_error": journal_state.get("reason")}


@pytest.fixture
def arm_boot(good_pre):
    """`arm_boot(**overrides)` - one boot decision on all-passing preconditions.

    A fixture rather than a helper, because the preconditions ARE fixture state: a plain
    function would have to reach `good_pre` by name, which is the wrong shape and reads
    as a global. Overrides replace ONE gate, which is how the refusal tests attribute a
    refusal to the link that produced it.
    """
    def _boot(**overrides):
        merged = dict(good_pre)
        merged.update(overrides)
        return disk_boot.boot_enable(merged)
    return _boot


def run_engine_boot():
    """THE REAL BOOT, the way the engine runs it: install, then `boot_from_connector`.

    Same fake fork as `tests/test_auto_boot.py`, so the disk default reads the boot's
    own
    install verdict, its own guard verdict, its own restore fit and its own arena
    verdict
    - which is the only way to prove the wiring rather than the unit.
    """
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    report = backport.install(cls, TB.our_shape_config(), manager)
    return cls, manager, backport.boot_from_connector(cls, report, manager)


def expire_arm():
    """Age the arm out the way the CLOCK does - in this process AND in the file.

    No monkeypatching of the arm's own verdict: the expiry is real on both sides. This
    process's monotonic deadline is moved into the past, and the arm file's own
    `expires_ts` is aged in place, because `is_armed()` asks the file once the in-
    process
    arm lapses and a file still claiming two hours answers ARMED to a worker - which is
    the half the renewal feature exists for. Only the timestamps move; the file keeps
    its
    boot id, so this is an expiry and not the cross-boot fence.
    """
    disk_spill._state["arm_expires_mono"] = time.monotonic() - 1
    disk_spill._state["arm_file_checked_mono"] = None
    disk_spill._state["arm_file_verdict"] = False
    path = disk_spill._arm_path()
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        document["armed_ts"] = time.time() - 8000
        document["expires_ts"] = time.time() - 1
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
    assert disk_spill.is_armed() is False, "the arm did not expire on either side"


def arm_path(tier):
    return os.path.join(tier["diag"], disk_spill.ARM_FILE)


# --- 1. the flag itself --------------------------------------------------------

def test_unset_means_on_and_says_it_came_from_the_default():
    """The default is ON, and "where did this value come from" is an answer, not a
    guess.

    An operator debugging a boot needs the two cases told apart: a value he set, and the
    shipped default. `source` is what the boot line prints, so the payload and the log
    cannot disagree about which boot he is looking at.
    """
    resolved = config.disk_boot_enable()
    assert resolved["enabled"] is True
    assert resolved["source"] == "default"
    assert resolved["raw"] is None
    assert resolved["recognised"] is True
    assert resolved["env_var"] == "KVTIER_DISK_BOOT_ENABLE"
    assert config.ENV_DISK_BOOT_ENABLE == "KVTIER_DISK_BOOT_ENABLE"
    # An EMPTY value is "unset", the same reading as every other switch here.
    os.environ[config.ENV_DISK_BOOT_ENABLE] = "   "
    assert config.disk_boot_enable()["source"] == "default"


@pytest.mark.parametrize("raw", ["0", "false", "off", "no", "OFF", "False"])
def test_the_disable_words_disable(raw, monkeypatch):
    """The four words the house grammar already uses, and only those.

    Same falsy set as `VLLM_E2_TIER_DEV`: an operator who knows one variable knows the
    other, and a flag whose off-words are its own invention is a flag he mis-types.
    """
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, raw)
    resolved = config.disk_boot_enable()
    assert resolved["enabled"] is False and resolved["recognised"] is True
    assert resolved["source"] == "env" and resolved["raw"] == raw


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
def test_the_enable_words_enable_and_are_recognised(raw, monkeypatch):
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, raw)
    resolved = config.disk_boot_enable()
    assert resolved["enabled"] is True and resolved["recognised"] is True


def test_an_unrecognized_value_reads_as_on_but_is_reported(monkeypatch, tier,
                                                           arm_boot, caplog):
    """`KVTIER_DISK_BOOT_ENABLE=flase` must not silently mean "disabled".

    The default is ON, so an unknown word can only be read as ON - but the operator who
    typed it meant OFF, and the only honest answer is to obey the default AND say out
    loud that the word he wrote is not a disable word. A typo that reads as intent is
    how
    a box keeps serving from disk after somebody decided it should not.
    """
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "flase")
    resolved = config.disk_boot_enable()
    assert resolved["enabled"] is True
    assert resolved["recognised"] is False
    with caplog.at_level(logging.WARNING):
        out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_ARMED
    assert out["recognised"] is False
    assert out["raw_value"] == "flase"
    assert "is not a recognized value" in caplog.text


def test_the_new_env_var_is_listed_in_the_suite_env_cleansing():
    """A suite that inherited `KVTIER_DISK_BOOT_ENABLE=0` would test the wrong product.

    The DEFAULT of this flag is the claim under test, so inheriting a value from the
    shell that runs the suite is the one env leak that could make a green suite mean
    nothing. `conftest.ENV_VARS` is where that is prevented; this pins the entry exists.
    """
    assert config.ENV_DISK_BOOT_ENABLE in conftest.ENV_VARS


# --- 2. the default boot arms and serves, with no POST -------------------------

def test_the_default_boot_comes_up_arming_and_serving_with_no_post(tier, caplog):
    """THE HEADLINE: the real boot, a configured tier, and not one POST.

    Everything asserted here is reached without calling the dev verb at all: the spill
    path is armed - as a LATCH since 0.2.35, so there is no "maximum legal TTL" left to
    arm AT - the serve posture is up, the payload's `serving_as` says so, the boot
    line's `disk_boot=` token agrees with the payload, and the arm file the worker
    ranks read is on disk naming THIS boot and carrying no expiry. That last one is the
    fleet's door: an EngineCore-only arm arms the one process that has no KV memory.
    """
    with caplog.at_level(logging.WARNING):
        _cls, _manager, boot_report = run_engine_boot()

    assert boot_report["tier"] is True, "the RAM tier armed, or nothing else can"
    verdict = boot_report["disk_boot"]
    assert verdict["outcome"] == disk_boot.OUTCOME_ARMED, verdict["reason"]
    assert verdict["honoured"] is True
    assert verdict["requested"] is True and verdict["source"] == "default"
    # THE ARM: a latch, here and in the file the ranks obey, with no time on either.
    assert disk_spill.is_armed() is True
    assert verdict["arm_expires"] is False, "the boot view still promises an expiry"
    assert disk_spill.arm_state()["armed"] is True
    assert disk_spill.arm_state()["latched"] is True
    assert verdict["arm_file_written"] is True
    with open(arm_path(tier), encoding="utf-8") as handle:
        published = json.load(handle)
    assert published["armed"] is True
    assert "expires_ts" not in published and "seconds" not in published, \
        "an arm file written with a TTL by a boot default arms nobody at 02:00"
    assert published["boot_id"] == disk_spill._boot_id(), \
        "an arm file naming another boot is refused by the ranks - this one must not be"
    # THE POSTURE: disk authority, with no second switch and no POST.
    posture = disk_restore.disk_posture()
    assert posture["disk_authority"] is True
    assert posture["enabled"] is False, "authority is one decision, not two"
    assert posture["serving_as"] == "disk_authority"
    assert verdict["serving_as_after"] == "disk_authority"
    # The boot line and the payload agree, in the one token the operator greps.
    assert "disk_boot=ARMED" in boot_report["line"]
    assert backport.section()["boot"]["disk_boot"]["outcome"] \
        == disk_boot.OUTCOME_ARMED
    # ONE line, naming the flag, its source, and how to switch it off.
    line = [r for r in caplog.records
            if r.name == "kvtier_dev_plugin.disk_boot"
            and "KVTIER disk boot default=" in r.getMessage()]
    assert len(line) == 1, [r.getMessage() for r in caplog.records]
    text = line[0].getMessage()
    assert "KVTIER_DISK_BOOT_ENABLE unset -> default ON" in text
    assert "HONOURED" in text and "serving_as=disk_authority" in text
    assert "KVTIER_DISK_BOOT_ENABLE=0" in text, "the line must name how to turn it off"


def test_the_default_boot_still_needs_a_worker_and_says_so(tier, caplog):
    """Arming the boot default does NOT bypass the worker proof, and the gate says why.

    The boot default raises the two switches a POST raises - it does not open the offer
    path past its safety gates. Driven through the REAL boot (the RAM tier has to be
    armed for the gate to get as far as the worker proof), with no rank file attesting a
    fitted worker half, so it stays shut on `no_worker_proof`: the stale-KV defence that
    matters more than this feature does. A boot default that opened the offer path would
    be the 2026-10-05 incident re-written into the launcher.
    """
    with caplog.at_level(logging.WARNING):
        _cls, _manager, boot_report = run_engine_boot()
    assert boot_report["disk_boot"]["outcome"] == disk_boot.OUTCOME_ARMED
    assert disk_restore.disk_posture()["serving_as"] == "disk_authority"
    gate = disk_restore.gate()
    assert gate["ok"] is False
    assert gate["reason_kind"] == "no_worker_proof", gate
    assert disk_restore.offer(base.Manager(), base.Req(request_id="x", blocks=4), 0,
                              (0, False)) == (0, False)
    assert disk_restore.counters()["offers"] == 0


# --- 3. the off switch reproduces today's boot --------------------------------

def test_the_off_word_reproduces_todays_disarmed_boot_exactly(tier, monkeypatch,
                                                               caplog):
    """`KVTIER_DISK_BOOT_ENABLE=0` is the 0.2.32 boot, in every field that differs.

    Not "comes up disarmed" only: the spill tier's own arm view, the posture, the gate's
    refusal kind, the boot line's token, and the ABSENCE of any disk-boot refusal are
    all
    pinned, because "exactly as today" is the promise and a partial match is how a
    default-on release ships a boot nobody can get back.
    """
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "0")
    with caplog.at_level(logging.WARNING):
        _cls, _manager, boot_report = run_engine_boot()

    verdict = boot_report["disk_boot"]
    assert verdict["outcome"] == disk_boot.OUTCOME_OFF
    assert verdict["honoured"] is False and verdict["requested"] is False
    assert verdict["raw_value"] == "0" and verdict["source"] == "env"
    assert disk_spill.is_armed() is False
    assert disk_spill.arm_state()["armed"] is False
    assert not os.path.exists(arm_path(tier)), \
        "the boot default must not publish an arm the operator declined"
    assert disk_restore.disk_posture()["serving_as"] == "off"
    assert disk_restore.gate()["reason_kind"] == "disarmed"
    assert not [item for item in boot_report["refusals"]
                if item.startswith("disk_boot:")], \
        "a declined boot default is a posture, not a fault"
    assert "disk_boot=OFF" in boot_report["line"]
    text = "\n".join(r.getMessage() for r in caplog.records
                     if r.name == "kvtier_dev_plugin.disk_boot")
    assert "KVTIER disk boot default=OFF" in text
    assert "'0'" in text, "the line names the value the operator actually wrote"
    # And the two POSTs are still the whole lever, exactly as they were.
    disk_store.disk_for_engine(None, {"op": "arm", "seconds": 600})
    disk_store.disk_for_engine(None, {"op": "authority", "enabled": True})
    assert disk_restore.disk_posture()["serving_as"] == "disk_authority"


def test_the_inert_boot_beats_the_disk_default(tier, arm_boot):
    """`VLLM_E2_TIER_DEV_BOOT_INERT` is the global opt-out and it wins.

    It already decided this boot arms nothing; a disk-tier default that armed anyway
    would make the boot-wide switch a lie. Reported as its own outcome so an operator
    who
    set only the disk flag sees WHY nothing came up.
    """
    out = arm_boot(inert=True)
    assert out["outcome"] == disk_boot.OUTCOME_INERT
    assert out["honoured"] is False
    assert disk_spill.is_armed() is False
    assert disk_restore.disk_posture()["serving_as"] == "off"
    assert config.ENV_BOOT_INERT in out["reason"]


def test_an_engine_with_no_disk_tier_is_not_reported_as_a_failure(tier, monkeypatch,
                                                                  arm_boot):
    """No `VLLM_E2_TIER_DISK_ROOT`: nothing to honour, and NOT a refusal.

    The distinction keeps the boot line's refusal list honest: an engine with no SSD
    tier
    at all must not read as a broken one, while an engine that HAS one and could not be
    armed must. Both print the same one line, and the token differs.
    """
    monkeypatch.delenv(config.ENV_DISK_ROOT)
    disk_spill.reset_state_for_tests()
    out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_NO_STORE
    assert out["disk_configured"] is False
    assert out["honoured"] is False
    assert disk_spill.is_armed() is False


# --- 4. a manual disarm always beats the boot default -------------------------

# THE NON-NEGOTIABLE ORDERING LIVES IN SECTION 5 NOW
# (`test_the_operator_veto_outlives_a_second_boot_in_one_process`). What used to be here
# drove the property through `renew_if_due()`, which no longer exists; the new test
# drives it through the thing that can still re-arm a latched tier - `boot_enable()`
# itself - and keeps the two assertions that mattered: the tier is shut in the GATE (a
# disarm stops serving from disk, not only writing to it), and the refusal names the
# operator.

# SUPERSEDED BY SECTION 5
# (`test_lowering_the_posture_from_the_verb_vetoes_a_second_boot_too`), which proves the
# same latch against the path that can still re-arm a latched tier - a second
# `boot_enable()` - instead of against the deleted renewal tick.


def test_a_refused_authority_change_does_not_latch(tier, arm_boot):
    """A curl typo is not an instruction: `enabled: "false"` changes nothing and latches
    nothing.

    The setter refuses a non-boolean, so the latch has to stay down with the switch. A
    malformed command that latched renewal would leave the boot default silently dead
    because somebody typed a string.
    """
    arm_boot()
    out = disk_store.disk_for_engine(None, {"op": "authority", "enabled": "false"})
    assert out["ok"] is False
    assert disk_boot.view()["disarmed_by_operator"] is False
    assert disk_restore.disk_posture()["disk_authority"] is True


def test_a_hand_arm_lifts_the_veto_and_the_verb_still_reports_its_ttl_as_ignored(
        tier, arm_boot):
    """An operator who disarms and then arms by hand has asked for the tier back.

    Leaving the veto set would have a later boot in this process REFUSE to arm a box he
    just chose to serve from disk - and with the arm a latch, that refusal is forever
    rather than two hours. Only a human at the verb lifts it: `boot_enable`'s own
    fail-safe calls `disarm()` directly and must NOT latch (a rollback is not an
    instruction, and this is the test that keeps it that way).

    The `seconds` the call carries is asserted as IGNORED in the same breath, because
    this is the shape of the call the proxy makes today: it must keep arming, and it
    must not be allowed to look like it bought a lifetime.
    """
    arm_boot()
    disk_store.disk_for_engine(None, {"op": "disarm"})
    assert disk_boot.view()["disarmed_by_operator"] is True
    verdict = disk_store.disk_for_engine(None, {"op": "arm", "seconds": 600})
    assert verdict["armed"] is True and verdict["seconds_ignored"] is True
    assert disk_boot.view()["disarmed_by_operator"] is False
    assert disk_spill.is_armed() is True
    assert arm_boot()["outcome"] == disk_boot.OUTCOME_ARMED


def test_the_boot_default_rollback_does_not_latch(tier, arm_boot):
    """A failed enable rolls the arm back WITHOUT latching: not an operator's wish.

    If a rollback latched, the next manual arm would be the only way back and renewal
    would be silently dead for the boot - a diagnostic that disables the feature it is
    reporting on.
    """
    out = arm_boot(arena_state="tiny", arena_reason="the pool clamps to 1 slot")
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert disk_boot.view()["disarmed_by_operator"] is False


# --- 5. THE ARM IS A LATCH (0.2.35): no expiry, nothing re-arms it --------------
#
# UNTIL 0.2.34 THIS SECTION WAS "RENEWAL": seven tests for a tick that beat the arm's
# 7,200 s cap, plus an `expire_arm()` fixture that faked the lapse the tick existed to
# fix. All of it is DELETED, because the arm stopped being a timer - the operator's
# ruling is in `disk_spill`'s module docstring (an expired arm froze the LRU shedding
# that runs from the spill writer, and no setting of the timer bounded the store: one
# rank at ~38 MB/s fills a 192 GiB ceiling in ~90 minutes, so even the 1,800 s default
# permitted a third of a ceiling-fill). What replaces the section is the other side of
# the same worry: with no timer, WHAT ELSE COULD ARM A BOX BEHIND AN OPERATOR'S BACK,
# and what stops a file from an earlier boot doing it. Those are the questions below,
# and they are the questions `disk_boot.py`'s fail-safe and operator veto exist to
# answer.

def test_an_arm_lands_as_a_latch_with_no_expiry_anywhere(tier, arm_boot):
    """The arm, the view and the FILE all say the same thing: nothing lowers it.

    Three surfaces carried a time and all three are checked, because a reader reaches
    the arm through any of them: `is_armed()` (this process), `arm_state()` (the
    payload, and until this release the proxy's countdown column), and the arm FILE the
    worker ranks read. A leftover `expires_ts` in the file would be worse than no field
    at all - a reader that honoured it would disarm a serving tier at an arbitrary hour,
    which is the bug being removed.
    """
    verdict = disk_spill.arm()
    assert verdict["armed"] is True and verdict["latched"] is True
    assert disk_spill.is_armed() is True

    state = disk_spill.arm_state()
    for gone in ("seconds_left", "expires_at_wallclock", "default_seconds",
                 "max_seconds"):
        assert gone not in state, f"the view still reports {gone}: the TTL is back"
    assert state["latched"] is True
    assert "LATCH" in state["note"]

    with open(arm_path(tier), encoding="utf-8") as handle:
        published = json.load(handle)
    assert published["armed"] is True and published["latched"] is True
    for gone in ("expires_ts", "seconds"):
        assert gone not in published, f"the file the RANKS read still carries {gone}"


def test_an_arm_survives_ten_hours_of_clock(monkeypatch, tier, arm_boot):
    """THE INVERSION OF THE DELETED `test_an_arm_expires_on_its_own_clock`.

    The old test advanced the clock an hour and asserted the arm had lapsed. The same
    manoeuvre now has to prove the opposite, against BOTH clocks the expired version
    read (the monotonic one `is_armed()` consulted and the wall-clock one the FILE was
    checked against), because a reader that still had a deadline anywhere would fail
    here. Advancing a clock is not a simulation of a long boot - it is exactly what a
    two-hour idle box looks like to the code that used to expire under it, and the
    request that arrived after that idle is the one the renewal tick existed to save.
    """
    disk_spill.arm()
    real_wall, real_mono = time.time, time.monotonic
    monkeypatch.setattr(time, "time", lambda: real_wall() + 10 * 3600.0)
    monkeypatch.setattr(time, "monotonic", lambda: real_mono() + 10 * 3600.0)
    disk_spill._state["arm_file_checked_mono"] = None   # force a file re-read
    assert disk_spill.is_armed() is True, "the arm still expires on a clock"
    assert disk_spill.arm_state()["armed"] is True
    assert disk_restore.gate()["reason_kind"] != "disarmed"


def test_arm_still_accepts_a_seconds_argument_and_says_it_ignored_it(tier,
                                                                     arm_boot):
    """WIRE COMPATIBILITY, not a knob: `{"op":"arm","seconds":7200}` still arms.

    The proxy the operator drives sends a TTL today, and its countdown column is removed
    in a separate, sequenced change in the sibling repo. So the argument must keep being
    ACCEPTED - a 400 or a `seconds must be a number` refusal on the day this ships would
    break the one call an operator actually makes. What it does is nothing, and the
    verdict says `seconds_ignored: true` rather than letting the number look meaningful.
    A non-numeric value, which the TTL version refused, is ignored too: there is no
    bound left to violate, and refusing it would be a refusal of a value that cannot
    matter.
    """
    for body in ({"op": "arm", "seconds": 7200}, {"op": "arm", "seconds": 999999},
                 {"op": "arm"}, {"op": "arm", "seconds": "7200"}):
        disk_spill.disarm()
        verdict = disk_store.disk_for_engine(None, dict(body))
        assert verdict["armed"] is True, verdict
        assert verdict["seconds_ignored"] is (body.get("seconds") is not None), verdict
        assert "seconds" not in verdict, "the verdict still reports a TTL"
    with open(arm_path(tier), encoding="utf-8") as handle:
        assert "expires_ts" not in json.load(handle)


def test_the_operator_veto_outlives_a_second_boot_in_one_process(tier, arm_boot):
    """THE VETO'S NEW JOB, and the reason the latch survived renewal's deletion.

    `note_operator_disarm` used to gate `renew_if_due()` and nothing else, so with
    renewal gone it guarded an empty set and could have been deleted. It is kept because
    with a LATCH there is a path worse than a renewal: `boot_enable()` ITSELF. A second
    connector boot inside one process - which a re-instantiated engine does, and which
    `test_a_second_boot_in_one_process_prints_one_line_and_counts_two` does on purpose -
    would otherwise re-arm a box an operator had just disarmed, and with no timer that
    re-arm would be FOREVER rather than two hours. So the veto is now a gate on the boot
    decision, and this test is what stands between a future refactor and that bug.

    The gate the whole feature is about is asserted too: a disarm closes SERVING from
    disk, not only writing to it, which is what `gate()`'s `disarmed` kind means.
    """
    arm_boot()
    assert disk_spill.is_armed() is True
    verdict = disk_store.disk_for_engine(None, {"op": "disarm"})
    assert verdict["armed"] is False
    assert not os.path.exists(arm_path(tier)), \
        "disarm left the file the ranks read: a half-disarmed fleet"
    assert disk_restore.gate()["reason_kind"] == "disarmed", \
        "a disarm must stop SERVING from disk, not only writing to it"

    second = arm_boot()
    assert second["outcome"] == disk_boot.OUTCOME_REFUSED, second
    assert "operator has disarmed" in second["reason"], second["reason"]
    assert disk_spill.is_armed() is False, \
        "a second boot re-armed a box the operator disarmed"
    assert disk_boot.view()["disarmed_by_operator"] is True
    assert '"op":"disarm"' in disk_boot.view()["latch_reason"]

    # The veto is the operator's to lift, and lifting it restores the boot default.
    disk_store.disk_for_engine(None, {"op": "arm"})
    assert disk_boot.view()["disarmed_by_operator"] is False
    third = arm_boot()
    assert third["outcome"] == disk_boot.OUTCOME_ARMED, third["reason"]
    assert disk_spill.is_armed() is True


def test_lowering_the_posture_from_the_verb_vetoes_a_second_boot_too(tier,
                                                                     arm_boot):
    """The cheaper "stop serving from disk" POST vetoes exactly as `disarm` does.

    `{"op":"authority","enabled":false}` is the switch an operator reaches for first, so
    if it did not latch, a later boot in the same process would put the arm back under
    him and the tier would be armed, serving nothing, and looking like a fault in the
    read path - the shape this suite keeps being written to catch.
    """
    arm_boot()
    disk_store.disk_for_engine(None, {"op": "authority", "enabled": False})
    assert disk_boot.view()["disarmed_by_operator"] is True
    assert '"enabled":false' in disk_boot.view()["latch_reason"]
    assert arm_boot()["outcome"] == disk_boot.OUTCOME_REFUSED
    assert disk_spill.is_armed() is False


def test_the_boot_disarms_a_file_left_by_an_earlier_boot_on_one_kernel(
        tier, monkeypatch, caplog):
    """THE INVARIANT THE LATCH DEPENDS ON, planted as the exact file that breaks it.

    A `disk-spill-arm.json` from an EARLIER ENGINE BOOT, naming THIS kernel's boot id
    and claiming `armed: true`. The boot fence cannot refuse it: `BOOT_ID_PATH` is per
    KERNEL boot, so four EngineCore boots inside one 2w3d uptime all carry the same id,
    and this file is a perfect lookalike of a live arm. While the arm had a TTL the TTL
    is what made it harmless; a latch never expires, so the rule is now THE ONLY THING
    THAT ARMS A PROCESS IS THAT PROCESS'S OWN BOOT DECISION, and the mechanism is
    `boot_enable()`'s fail-safe calling `disarm()`, which deletes the file
    (`_remove_arm_file`).

    Both halves are asserted, because they are different bugs: (a) the boot does not
    become armed, and (b) the file is GONE afterwards, so no later reader in this
    process - `disk_restore.gate()`, a worker's `_may_restore_worker`, the rank proof -
    can be fooled by a decision that was never made. (b) is the half a "just don't read
    it in that one place" fix would fail, and it is why the fail-safe deletes instead of
    merely declining.
    """
    stale = {"version": disk_spill.PROTOCOL_VERSION, "writer": "an-earlier-engine-boot",
             "armed": True, "armed_ts": time.time() - 86400, "latched": True,
             "root": tier["root"], "epoch": "some-other-process-token",
             "epoch_scope": "process", "pid": 1,
             # THIS kernel's boot id, deliberately: the file must be one the fence
             # cannot see, so what is under test is the boot decision and not the fence.
             # (`epoch_scope: "process"` for the same reason - a per-process epoch is
             # NOT evidence of another boot, so it cannot refuse this file either.)
             "boot_id": disk_spill._boot_id()}

    def plant():
        with open(arm_path(tier), "w", encoding="utf-8") as handle:
            json.dump(stale, handle)
        disk_spill._state["arm_file_checked_mono"] = None
        assert disk_spill.is_armed() is True, \
            "the fence refused this file, so this test would not be testing the " \
            "fail-safe - it would be testing the fence"

    # A boot that REFUSES (the arena cannot service an offer) must erase it.
    plant()
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "1")
    with caplog.at_level(logging.WARNING):
        out = disk_boot.boot_enable({"inert": False, "tier_armed": True,
                                     "restore_fitted": True, "arena_state": "tiny",
                                     "arena_reason": "the pool clamps to 1 slot",
                                     "journal_writable": True, "journal_error": None})
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert disk_spill.is_armed() is False
    assert not os.path.exists(arm_path(tier)), \
        "a boot that did not arm left a previous engine boot's file for the ranks"

    # And so must a boot that was told OFF: the flag is the operator's word, and the
    # file is somebody else's.
    plant()
    disk_boot.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "0")
    out = disk_boot.boot_enable({"inert": False, "tier_armed": True,
                                 "restore_fitted": True, "arena_state": "ok",
                                 "arena_reason": None, "journal_writable": True,
                                 "journal_error": None})
    assert out["outcome"] == disk_boot.OUTCOME_OFF
    assert disk_spill.is_armed() is False, \
        "the boot honoured a previous boot's arm file: the forbidden re-arm"
    assert not os.path.exists(arm_path(tier))


def test_every_outcome_that_is_not_ARMED_goes_through_the_fail_safe():
    """STRUCTURAL: `boot_enable()` has exactly two ways to leave, one of them the
    fail-safe.

    The behavioural tests plant files and clock time; this one makes the fail-safe
    unbypassable by SHAPE. Every `return` with a value inside `boot_enable` is either
    `_disarmed_exit(...)` or the ARMED path's `view()`, so a new early return cannot be
    written that skips the disarm - the way the deleted TTL used to make an abandoned
    file harmless without anyone having to remember it. It also pins that the rollback
    lives in ONE function: `disk_spill.disarm()` appears exactly once in the module, so
    there is no second, weaker rollback to keep in sync, and `disk_spill.arm()` appears
    exactly once, so there is no arm path a disarm cannot reach.
    """
    tree = ast.parse(inspect.getsource(disk_boot))

    def _dotted(node):
        parts = []
        head = node.func
        while isinstance(head, ast.Attribute):
            parts.append(head.attr)
            head = head.value
        if isinstance(head, ast.Name):
            parts.append(head.id)
        return ".".join(reversed(parts))

    boot = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "boot_enable")
    exits = sorted({_dotted(n.value) for n in ast.walk(boot)
                    if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)})
    assert exits == ["_disarmed_exit", "view"], exits
    dotted = [_dotted(n) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert dotted.count("disk_spill.disarm") == 1, dotted
    assert dotted.count("disk_spill.arm") == 1, dotted


def test_the_veto_is_process_memory_only_and_never_touches_the_store(tier, arm_boot):
    """A veto is a HUMAN'S WORD ABOUT THIS PROCESS, not state for the next boot.

    Nothing about it is written to disk: it lives in `disk_boot._state`, and the next
    process reads the environment's word like every other boot default. A persisted veto
    would be the 2026-10-03 prohibition wearing the other hat - a previous boot deciding
    what this boot does - and it would also let an operator disarm a box he had already
    rebooted away from and never get the tier back.
    """
    arm_boot()
    disk_store.disk_for_engine(None, {"op": "disarm"})
    assert disk_boot.view()["disarmed_by_operator"] is True
    leftovers = [name for name in os.listdir(tier["diag"]) + os.listdir(tier["root"])
                 if "veto" in name or "latch" in name or "boot" in name]
    assert leftovers == [], f"the veto persisted state: {leftovers}"


# --- 6. the fail-safe: every failure leaves the boot disarmed, loudly ----------

@pytest.mark.parametrize("gate,reason_kind", [
    ({"tier_armed": False}, "KV tier itself is not armed"),
    ({"restore_fitted": False}, "restore layer is not fitted"),
    ({"arena_state": "tiny", "arena_reason": "1 free block"}, "host-RAM arena"),
    ({"arena_state": "no_pool", "arena_reason": "no pool"}, "host-RAM arena"),
    ({"journal_writable": False, "journal_error": "EACCES"},
     "JOURNAL cannot be written"),
])
def test_every_failed_gate_leaves_the_boot_disarmed_and_names_the_gate(
        tier, arm_boot, caplog, gate, reason_kind):
    """A boot that cannot come up serving comes up DISARMED, ONCE, LOUDLY.

    One assertion set for five different gates, because the promise is the same for all
    of them and the shape of the promise is what an operator needs: nothing armed, no
    posture, a reason that names the link, exactly one WARNING line, and a boot that
    continues (nothing here raises). The reason strings are matched on their own words,
    so a reworded sentence keeps failing until the test is updated rather than passing
    on
    a vacuous substring.
    """
    with caplog.at_level(logging.WARNING):
        out = arm_boot(**gate)
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert out["honoured"] is False
    assert disk_spill.is_armed() is False, "a failed gate left the tier armed"
    assert not os.path.exists(arm_path(tier))
    assert disk_restore.disk_posture()["serving_as"] == "off"
    assert reason_kind in out["reason"], out["reason"]
    lines = [r for r in caplog.records if "NOT HONOURED" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in caplog.records]
    assert "KVTIER_DISK_BOOT_ENABLE=0" in lines[0].getMessage()


def test_an_arm_that_cannot_reach_the_ranks_is_refused_and_rolled_back(
        tier, monkeypatch, arm_boot):
    """No arm file means no ARMED FLEET: EngineCore has no KV memory, so this is the
    partially-armed shape the brief forbids, and the arm is rolled back.

    `VLLM_E2_TIER_DIAG` unset is the ordinary way to reach it (the arm reaches a worker
    only through the diagnostics directory). The rollback is the assertion that matters:
    an EngineCore-only arm would report success while every rank stayed disarmed, and
    the
    scheduler's AND-over-ranks proof would then close the gate anyway - armed-looking,
    serving nothing.
    """
    monkeypatch.delenv(config.ENV_DIAG_DIR)
    monkeypatch.delenv(config.ENV_DISK_JOURNAL_DIR)
    disk_index.reset_state_for_tests()
    out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert out["arm_file_written"] is False
    assert disk_spill.is_armed() is False, "a refused boot left the tier armed here"
    assert disk_restore.disk_posture()["serving_as"] == "off"
    assert "worker ranks" in out["reason"] or "diagnostics" in out["reason"]


def test_a_posture_that_cannot_be_raised_rolls_the_arm_back(tier, arm_boot,
                                                            monkeypatch):
    """The pair is all-or-nothing in BOTH orders: if the posture fails, the arm goes.

    Reaching this needs the setter to refuse, which it does not do for a real boolean -
    so the failure is injected at the setter's own door, the only seam that can prove
    the
    rollback runs.
    """
    monkeypatch.setattr(disk_restore, "set_disk_authority",
                        lambda enabled: {"ok": False, "reason": "injected refusal"})
    out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert disk_spill.is_armed() is False, "the arm survived a failed posture"
    assert "rolled back" in out["reason"]


def test_the_boot_default_cannot_break_a_boot(tier, caplog):
    """`boot_enable` runs on the engine's connector-construction path.

    A flag whose own machinery raises must still leave a boot: disarmed, reported on the
    log line, and never an exception that reaches the connector. The pathological input
    is a precondition dict whose every read explodes.
    """
    class Exploding(dict):
        def get(self, key, default=None):
            raise RuntimeError(f"cannot read {key}")

    with caplog.at_level(logging.WARNING):
        out = disk_boot.boot_enable(Exploding())
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert "raised RuntimeError" in out["reason"]
    assert disk_restore.disk_posture()["serving_as"] == "off"
    assert "KVTIER_DISK_BOOT_ENABLE" in caplog.text, "the line names the flag"


def test_the_boot_default_arming_refusal_is_attributed_not_masked(tier, arm_boot,
                                                                  monkeypatch):
    """When `disk_spill.arm()` itself refuses, ITS reason is the reason.

    The boot default does not paraphrase the arm path - it reports the refusal it was
    given, so a reader sees the store's own objection rather than "the boot default
    failed".
    """
    monkeypatch.setattr(disk_spill, "arm",
                        lambda seconds=None: {"armed": False,
                                              "reason": "the root is on tmpfs"})
    out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert "the root is on tmpfs" in out["reason"]
    assert disk_restore.disk_posture()["serving_as"] == "off"


# --- 7. the prohibition: no boot-time re-arm from persisted arm state ----------

def test_the_prohibition_no_boot_time_re_arm_from_persisted_arm_state(tier, arm_boot,
                                                                      monkeypatch,
                                                                      caplog):
    """THE GUARD. `llm-api/AGENTS.md`, 2026-10-03: "Do not add a boot-time re-arm."

    The forbidden thing is a process RESURRECTING A POSTURE FROM A PERSISTED FILE
    written by an earlier boot. The test is behavioural, in the direction that matters:
    a `disk-spill-arm.json` is planted claiming armed, with a boot id that is NOT this
    boot's (which is what a reboot leaves behind), and the decision is taken with the
    flag OFF and then with the flag ON.

    With the flag OFF the boot comes up DISARMED even though a file claiming "armed"
    sits in the directory: the decision came from the environment, not from the file.
    SINCE 0.2.35 the same boot also DELETES that file (the fail-safe disarms on every
    non-ARMED outcome), so the assertion changed shape and got stronger - it used to be
    "the fence read the file and refused it", and it is now "the file is gone, so
    nothing downstream has to be trusted to refuse it". The fence is still proved to
    work, and it is proved to work on the flag-ON path, where an arm file IS written and
    the planted one must not be the one the ranks end up reading.

    Two claims, both needed: the fence (this file, a different KERNEL boot, is refused
    by name) and the fail-safe (a file the fence CANNOT see is deleted anyway - see
    `test_the_boot_disarms_a_file_left_by_an_earlier_boot_on_one_kernel`).
    """
    stale = {"version": disk_spill.PROTOCOL_VERSION, "writer": "planted",
             "armed": True, "armed_ts": time.time() - 3600, "latched": True,
             "root": tier["root"], "epoch": "this-process-token",
             # `epoch_scope: "process"` on purpose: with a per-process epoch the
             # epoch is NOT evidence (two processes of one boot legitimately disagree
             # on it), so
             # the fence that must fire is the KERNEL BOOT ID - the one field that moves
             # on a reboot, which is the event the prohibition is about.
             "epoch_scope": "process", "pid": 1,
             "boot_id": "00000000-0000-0000-0000-000000000000"}
    with open(arm_path(tier), "w", encoding="utf-8") as handle:
        json.dump(stale, handle)

    # --- the flag OFF: a file claiming "armed, forever" decides nothing.
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "0")
    with caplog.at_level(logging.WARNING):
        out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_OFF
    assert disk_spill.is_armed() is False, \
        "the boot honoured a previous boot's arm file: the forbidden re-arm"
    assert disk_restore.disk_posture()["serving_as"] == "off"
    # THE FAULT LINE. Before 0.2.35 the stale file survived this boot and the fence was
    # the only thing between it and a worker; now the boot's own fail-safe removes it,
    # so the fence's counter may legitimately be zero and the file's absence is the
    # claim worth testing. Both directions are pinned so a "helpful" change that stops
    # deleting the file has to fight this assertion.
    assert not os.path.exists(arm_path(tier)), \
        "a boot that declined to arm left a previous boot's arm file behind"

    # --- the flag ON: armed because the FLAG says so, and the fence still refuses.
    disk_boot.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    monkeypatch.delenv(config.ENV_DISK_BOOT_ENABLE)
    with open(arm_path(tier), "w", encoding="utf-8") as handle:
        json.dump(stale, handle)
    # Read the file's verdict FIRST, so the fence is provably consulted on the planted
    # file before this boot's own arm replaces it. A previous boot's file arms nobody.
    assert disk_spill.is_armed() is False
    assert disk_spill.arm_state()["arm_file_boot_refusals"] >= 1
    assert "boot id mismatch" in disk_spill.arm_state()["arm_file_reason"]
    out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_ARMED
    assert out["source"] == "default"
    assert out["reason"] and "POST" in out["reason"]
    # The arm file this boot WROTE replaced the planted one and names THIS boot.
    with open(arm_path(tier), encoding="utf-8") as handle:
        published = json.load(handle)
    assert published["writer"] == disk_spill.WRITER_TAG
    assert published["boot_id"] == disk_spill._boot_id()


def test_the_prohibition_structurally_the_module_never_reads_an_arm_file():
    """The same guard, against the SOURCE: `disk_boot.py` never names the arm file.

    Behavioural tests catch a mistake in the path they exercise. This one catches it
    anywhere: the module that decides the boot posture has no vocabulary for reading
    persisted arm state - not the file name, not the reader function, not an `open()`,
    not
    the boot id, not `json`. It reaches the arm ONLY through
    `disk_spill.arm()`/`.disarm()
    `, the same doors the dev verb uses - one call for the boot, one for renewal.
    """
    # The module's PROSE is full of the words this guard forbids in its CODE - it is
    # documentation about the prohibition, and a guard that could not tell a warning
    # from
    # a `json.load()` would have to be deleted. So the analysis runs on the syntax tree:
    # every name, attribute, keyword and non-docstring string constant, plus every call.
    tree = ast.parse(inspect.getsource(disk_boot))
    docstrings = {id(node) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                       ast.AsyncFunctionDef))}
    words = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            words.append(node.value)
        elif isinstance(node, ast.Name):
            words.append(node.id)
        elif isinstance(node, ast.Attribute):
            words.append(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            words.append(node.arg)
        elif isinstance(node, ast.Call):
            calls.append(node)
    haystack = "\n".join(words)
    # Docstring text is excluded by dropping the string constants that ARE a docstring.
    doc_text = []
    for holder in ast.walk(tree):
        if id(holder) in docstrings:
            doc = ast.get_docstring(holder)
            if doc:
                doc_text.append(doc)
    for doc in doc_text:
        haystack = haystack.replace(doc, "")
    for forbidden in ("disk-spill-arm", "ARM_FILE", "_read_arm_file", "_arm_path",
                      "json.load", "boot_id", "_boot_id", "makedirs", "os.remove",
                      "os.stat"):
        assert forbidden not in haystack, \
            "the boot default must not touch persisted arm state, but its CODE names " \
            f"{forbidden!r}"
    # And the calls it can make, enumerated: no `open`, no `json`, no `os` at all.
    called = set()
    for node in calls:
        head = node.func
        while isinstance(head, ast.Attribute):
            head = head.value
        if isinstance(head, ast.Name):
            called.add(head.id)
    for forbidden_call in ("open", "json", "os", "pathlib", "shutil", "subprocess"):
        assert forbidden_call not in called, \
            f"the boot default calls into {forbidden_call!r}, so it can read state"
    # The doors it DOES use: the same two functions the dev verb uses, `arm` ONCE (the
    # boot decision - there is no renewal call left to count) and `disarm` ONCE (the
    # fail-safe, which lives in one function so a rollback cannot be half-remembered),
    # and the posture setter ONCE. 0.2.34 counted two of each because the renewal tick
    # armed and the posture-failure path disarmed; both halves of that went with the
    # tick, and the single remaining `disarm` is the fail-safe that replaced them.
    def _dotted(node):
        parts = []
        head = node.func
        while isinstance(head, ast.Attribute):
            parts.append(head.attr)
            head = head.value
        if isinstance(head, ast.Name):
            parts.append(head.id)
        return ".".join(reversed(parts))

    dotted = [_dotted(node) for node in calls]
    assert dotted.count("disk_spill.arm") == 1, dotted
    assert dotted.count("disk_spill.disarm") == 1, dotted
    assert dotted.count("disk_restore.set_disk_authority") == 1, dotted
    assert dotted.count("disk_restore.set_offer_switch") == 0, dotted


# --- 8. discoverability -------------------------------------------------------

def test_the_disk_verb_and_the_snapshot_payload_expose_the_boot_default(tier,
                                                                        arm_boot):
    """Status answers "was the boot default on, was it honoured, does the arm expire,
    has an operator disarmed" in one read, on the door an operator already opens.

    The fields the brief asks for, on BOTH payloads (the `/disk` verb's `status` and the
    snapshot's `disk_restore.boot_default`), built by the same function - because two
    views of one decision is how two views start disagreeing in production. The field
    that used to answer "did renewal run" answers "does this arm expire", and the answer
    is a flat `false`: a reader that used to watch a number count down now gets a fact
    that cannot go stale.
    """
    arm_boot()
    verb = disk_store.disk_for_engine(None, {"op": "status"})
    view = verb["disk_boot_default"]
    assert view["requested"] is True
    assert view["honoured"] is True and view["outcome"] == disk_boot.OUTCOME_ARMED
    assert view["arm_expires"] is False
    for gone in ("renewals", "renewal_failures", "last_renewal", "seconds_left",
                 "renewal_would_run", "renewal_check_seconds", "renew_ahead_seconds",
                 "arm_max_seconds", "arm_seconds"):
        assert gone not in view, f"the verb's view still carries {gone}"
    assert view["disarmed_by_operator"] is False
    assert view["armed_now"] is True
    assert config.ENV_DISK_BOOT_ENABLE in view["env_var"]
    assert "KVTIER_DISK_BOOT_ENABLE=0" in view["disable_how"]
    # The same view in the snapshot's one-call payload, from the same builder. The live
    # arm read is excluded from the equality (the two reads are milliseconds apart); the
    # KEY SETS are the contract, and every other field must be identical.
    section = disk_restore.scheduler_section()["boot_default"]
    live = {"armed_now"}
    assert set(section) == set(view)
    assert {k: v for k, v in section.items() if k not in live} \
        == {k: v for k, v in view.items() if k not in live}


def test_the_view_carries_the_same_keys_on_every_path(tier, arm_boot, monkeypatch):
    """A degraded answer is never thinner than a healthy one.

    Four paths - never booted, honoured, refused, flag-off - plus the arm-unreadable
    fallback (`static_view`), all compared key-for-key. A payload that loses fields on
    the path where something went wrong is a payload whose absence a reader has to
    interpret, which is the wrong shape for the only read that matters.
    """
    never = disk_boot.view()
    static = disk_boot.static_view()
    assert set(never) == set(static), set(never) ^ set(static)
    assert sorted(never) == sorted(disk_boot._KEYS), \
        "the named key set and the built view have drifted"
    armed = arm_boot()
    disk_boot.reset_state_for_tests()
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "0")
    off = arm_boot()
    disk_boot.reset_state_for_tests()
    refused = arm_boot(arena_state="unreadable", arena_reason="exploded")
    for name, out in (("armed", armed), ("off", off), ("refused", refused)):
        assert set(out) == set(never), (name, set(out) ^ set(never))
    # And a live view whose arm reads explode still carries every key: the live field is
    # overlaid on the static base, so the base is the contract.
    def explode(*_args, **_kwargs):
        raise RuntimeError("no arm view from here")

    monkeypatch.setattr(disk_spill, "arm_state", explode)
    monkeypatch.setattr(disk_spill, "is_armed", explode)
    broken = disk_boot.view()
    assert set(broken) == set(never)
    assert broken["armed_now"] is None
    # `arm_expires` is NOT a live read: it is a fact about the build, so a view that
    # could not read the arm still refuses to promise an expiry.
    assert broken["arm_expires"] is False


def test_the_arm_view_says_what_replaced_renewal_instead_of_going_quiet(tier,
                                                                        arm_boot):
    """0.2.34 told an operator whether renewal would run; 0.2.35 has to say something.

    Deleting a field is easy and silently misleading. The question the old field
    answered - "will this tier go dark on its own?" - still has an answer, and it is
    now carried
    by two fields that cannot rot: `arm_expires: false` on the boot view, and
    `latched: true` plus a note naming the disarm POST on `disk_spill.arm_state()`. If a
    future change reintroduces a lifetime, it has to change these words, and this test
    is what makes that a deliberate edit rather than a drift.
    """
    arm_boot()
    state = disk_spill.arm_state()
    assert state["latched"] is True
    note = state["note"]
    assert "LATCH" in note and "no clock" in note
    assert "0.2.35" in note, "the note must date the change it is describing"
    # Present-tense promises of a lifetime are what would mislead; the note is allowed
    # to say WHY the renewal was deleted, in past tense, because that is the sentence an
    # operator reading a 0.2.34 runbook needs to be told is now history.
    for obsolete in ("expires on its own", "seconds left", "will renew", "renews"):
        assert obsolete not in note, f"the note promises a lifetime: {obsolete!r}"
    assert disk_boot.view()["arm_expires"] is False
    # And the mid-boot sentence an operator is handed agrees with it.
    assert "LATCH" in disk_boot.disable_how()


def test_the_boot_line_token_agrees_with_the_payload_token(tier, arm_boot):
    """The log token and the payload token come from ONE vocabulary.

    `disk_boot.field_token` is the only owner of the spelling, so a payload that says
    `REFUSED` and a log line that says `FAILED` is not possible. An operator greps one
    and curls the other; they have to be the same word.
    """
    for outcome in disk_boot.OUTCOMES:
        assert disk_boot.field_token(outcome) == outcome
    assert disk_boot.field_token("invented") == disk_boot.OUTCOME_REFUSED
    assert disk_boot.field_token(None) == disk_boot.OUTCOME_UNDECIDED
    out = arm_boot()
    assert disk_boot.field_token(out["outcome"]) == "ARMED"


def test_a_second_boot_in_one_process_prints_one_line_and_counts_two(tier, arm_boot):
    """Two boots in one process (a reload, or a test) is ONE operator line.

    The line is the record of the boot posture, and a second copy of it is noise in the
    log an operator greps at 3 a.m. The counter still counts, because "did this ever run
    twice?" is a real question the payload answers.
    """
    arm_boot()
    second = arm_boot()
    assert second["boots"] == 2
    assert disk_boot._state["line_logged"] is True


def test_the_note_says_the_arm_needs_no_heartbeat_in_the_note_and_the_line(tier,
                                                                           arm_boot,
                                                                           caplog):
    """0.2.34 had to explain that renewal rides the request path, because renewal needed
    a request. A latch needs nothing, and neither the note nor the boot line may imply
    otherwise.

    The old test asserted the note promised a request-path heartbeat and published the
    cadence constants. The claim that survives is the operationally important one:
    NOTHING has to happen for the arm to stay up - not a request, not a tick, and not a
    timer thread this plugin has never had and will not grow - and the only thing that
    brings it down is a disarm POST. Asserting the ABSENCE of the heartbeat language is
    the point: a note that said "keeps the arm alive" would describe a mechanism that is
    gone, and an operator reading it would hunt for a log line that no longer prints.
    Checked on both surfaces an operator reads without opening source: the payload's
    `note` and the one WARNING line the boot prints.
    """
    with caplog.at_level(logging.WARNING):
        out = arm_boot()
    note = out["note"]
    assert "LATCH" in note
    assert "no renewal, because there is no expiry" in note
    for heartbeat in ("rides the request path", "keeps it up", "renews every"):
        assert heartbeat not in note, \
            f"the note still describes a heartbeat: {heartbeat!r}"
    assert '{\"op\":\"disarm\"}' in note, \
        "the note must name the ONLY thing that lowers the latch"

    line = [r.getMessage() for r in caplog.records
            if r.name == "kvtier_dev_plugin.disk_boot"
            and "KVTIER disk boot default=" in r.getMessage()]
    assert len(line) == 1, line
    assert "The arm is a LATCH" in line[0]
    assert "nothing to renew" in line[0]
    for heartbeat in ("Renewal keeps it up", "first renewal"):
        assert heartbeat not in line[0]


def test_the_disable_sentence_names_both_ways_off(tier):
    """Two different "offs", both named: the next boot's, and THIS boot's right now.

    An operator under pressure needs the right one. Naming `VLLM_E2_TIER_DEV_BOOT_INERT`
    too is deliberate - it is the stronger switch, and a line that named only the disk
    flag would leave him with two half-truths.
    """
    text = disk_boot.disable_how()
    assert "KVTIER_DISK_BOOT_ENABLE=0" in text
    assert config.ENV_BOOT_INERT in text
    assert '"op":"authority","enabled":false' in text
    assert '"op":"disarm"' in text
    assert config.ROUTE_PREFIX + config.ROUTE_DISK in text


@pytest.mark.parametrize("raw", SWEEP)
def test_the_boot_line_names_the_value_and_its_origin_for_every_input(
        raw, monkeypatch, tier, arm_boot, caplog):
    """Whatever the operator wrote, the payload says what it was and where it came from.

    The parameterized sweep is the point: there is no value of this flag whose boot
    posture leaves him guessing, because the flag is the thing he is told to edit when a
    boot misbehaves. The two vocabularies (off-words, on-words) and the unknown-word
    path
    all resolve to the same three fields: value, origin, outcome.
    """
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, raw)
    disk_boot.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    resolved = config.disk_boot_enable()
    out = arm_boot()
    assert out["source"] == resolved["source"]
    assert out["requested"] == resolved["enabled"]
    assert disk_boot.view()["outcome"] == out["outcome"]
    # The log line repeats what the payload says, including the exact text he wrote:
    # the line is the artifact an operator greps when a boot misbehaves, and a line that
    # said "from the environment" without the value would leave him guessing whether his
    # export actually reached the engine.
    with caplog.at_level(logging.WARNING):
        disk_boot._state["line_logged"] = False
        disk_boot._log_boot_line()
    printed = "\n".join(r.getMessage() for r in caplog.records
                         if r.name == "kvtier_dev_plugin.disk_boot")
    assert printed
    if raw:
        assert out["raw_value"] == raw
        assert repr(raw) in printed
        assert "unset -> default ON" not in printed
    else:
        assert out["raw_value"] is None
        assert "unset -> default ON" in printed
