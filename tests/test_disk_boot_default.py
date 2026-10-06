r"""THE DISK TIER'S BOOT DEFAULT (0.2.33): `KVTIER_DISK_BOOT_ENABLE`.

The feature is one flag with a default of ON, and these tests are ordered by what an
operator would notice first, because the failure this release exists to end is not a
crash - it is a boot that serves nothing while reporting that truth honestly.

1. **The default boot arms and serves with NO POST.** The headline: run the real boot
   (install, guard, tier arm, restore fit, arena verdict, then the disk default) against
   a configured disk tier and the box comes up `serving_as=disk_authority` with an armed
   spill path at the maximum legal TTL. Nobody POSTed anything.
2. **`KVTIER_DISK_BOOT_ENABLE=0` is today's boot, exactly.** Disarmed spill, posture
   down, `serving_as=off`, no refusal recorded against the disk default, and the two dev
   verbs still the only lever. A test that inherited the flag from a developer's shell
   would be testing the wrong product, which is why `conftest.ENV_VARS` deletes it.
3. **A manual disarm always beats the boot default.** A human's `{"op":"disarm"}` (or
   `{"op":"authority","enabled":false}`) latches, and renewal then refuses to re-arm -
   the property the brief makes non-negotiable, tested through the VERB and not by
   poking the latch.
4. **Renewal runs, logs once, and never arms a box that is not serving.** The two-hour
   arm cap is why renewal exists at all.
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
    path is armed at the hard cap, the serve posture is up, the payload's `serving_as`
    says so, the boot line's `disk_boot=` token agrees with the payload, and the arm
    file
    the worker ranks read is on disk naming THIS boot. That last one is the fleet's
    door:
    an EngineCore-only arm arms the one process that has no KV memory.
    """
    with caplog.at_level(logging.WARNING):
        _cls, _manager, boot_report = run_engine_boot()

    assert boot_report["tier"] is True, "the RAM tier armed, or nothing else can"
    verdict = boot_report["disk_boot"]
    assert verdict["outcome"] == disk_boot.OUTCOME_ARMED, verdict["reason"]
    assert verdict["honoured"] is True
    assert verdict["requested"] is True and verdict["source"] == "default"
    # THE ARM: the maximum legal TTL, here and in the file the ranks obey.
    assert disk_spill.is_armed() is True
    assert verdict["arm_seconds"] == disk_spill.ARM_MAX_SECONDS == 7200.0
    assert disk_spill.arm_state()["armed"] is True
    assert verdict["arm_file_written"] is True
    with open(arm_path(tier), encoding="utf-8") as handle:
        published = json.load(handle)
    assert published["armed"] is True
    assert published["seconds"] == disk_spill.ARM_MAX_SECONDS
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

def test_a_manual_disarm_after_a_boot_default_arm_is_not_re_armed(tier, arm_boot):
    """THE NON-NEGOTIABLE ORDERING, driven through the VERB, not by poking a latch.

    Sequence: the boot default comes up serving; a human POSTs `{"op":"disarm"}`; the
    arm
    is then EXPIRED (the state renewal exists to fix); renewal runs. It must refuse, the
    tier must stay disarmed, and the refusal must name the operator as the reason. If a
    future change ever re-arms here, this fails.
    """
    arm_boot()
    assert disk_spill.is_armed() is True
    verdict = disk_store.disk_for_engine(None, {"op": "disarm"})
    assert verdict["armed"] is False
    expire_arm()

    out = disk_boot.renew_if_due()
    assert out["renewed"] is False
    assert "disarm" in out["reason"]
    assert disk_spill.is_armed() is False, \
        "renewal re-armed a box the operator disarmed"
    view = disk_boot.view()
    assert view["disarmed_by_operator"] is True
    assert view["renewal_would_run"] is False
    assert '"op":"disarm"' in view["latch_reason"]
    # The tier is genuinely shut, not merely unarmed-looking: the offer gate refuses
    # on the arm, which is the gate `serving_as` never bypasses - the posture switches
    # and the arm are separate facts, so the GATE is the assertion that a disarm
    # actually stopped serving from disk.
    assert disk_restore.gate()["reason_kind"] == "disarmed"

def test_lowering_the_posture_from_the_verb_latches_too(tier, arm_boot):
    """`{"op":"authority","enabled":false}` is a human taking the box out of service.

    The cheaper way to say "stop serving from disk" must latch exactly as `disarm` does,
    or the boot default's renewal would keep the arm alive under an operator who has
    just
    turned the posture off - and then the tier would be armed, serving nothing, and
    looking like a fault in the read path.
    """
    arm_boot()
    disk_store.disk_for_engine(None, {"op": "authority", "enabled": False})
    expire_arm()
    assert disk_boot.renew_if_due()["renewed"] is False
    assert disk_spill.is_armed() is False
    assert disk_boot.view()["disarmed_by_operator"] is True
    assert '"enabled":false' in disk_boot.view()["latch_reason"]


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


def test_a_hand_arm_lifts_the_latch(tier, arm_boot):
    """An operator who disarms and then arms by hand has asked for the tier back.

    Leaving the latch set would switch renewal off for the rest of the boot on a box he
    just chose to serve from disk - the two-hour dark tier again, wearing a latch. Only
    a
    human at the verb lifts it: `boot_enable`'s own rollback calls `disarm()` directly
    and
    must NOT latch (a rollback is not an instruction).
    """
    arm_boot()
    disk_store.disk_for_engine(None, {"op": "disarm"})
    assert disk_boot.view()["disarmed_by_operator"] is True
    disk_store.disk_for_engine(None, {"op": "arm", "seconds": 600})
    assert disk_boot.view()["disarmed_by_operator"] is False
    expire_arm()
    assert disk_boot.renew_if_due()["renewed"] is True


def test_the_boot_default_rollback_does_not_latch(tier, arm_boot):
    """A failed enable rolls the arm back WITHOUT latching: not an operator's wish.

    If a rollback latched, the next manual arm would be the only way back and renewal
    would be silently dead for the boot - a diagnostic that disables the feature it is
    reporting on.
    """
    out = arm_boot(arena_state="tiny", arena_reason="the pool clamps to 1 slot")
    assert out["outcome"] == disk_boot.OUTCOME_REFUSED
    assert disk_boot.view()["disarmed_by_operator"] is False


# --- 5. renewal ---------------------------------------------------------------

def test_renewal_runs_while_enabled_and_logs_once(tier, arm_boot, caplog):
    """Renewal exists because `ARM_MAX_SECONDS` is a hard cap: 7200 s, then dark.

    Two things are pinned: that it RUNS (the arm comes back at the cap after an expiry),
    and that it says so exactly ONCE per boot - the tick rides the request path, and a
    line per request is a line nobody reads. The first-renewal line is also the only
    evidence in the log that a 0.2.33 boot will not go dark at the two-hour mark.
    """
    arm_boot()
    expire_arm()
    with caplog.at_level(logging.WARNING):
        first = disk_boot.renew_if_due()
    assert first["renewed"] is True, first
    assert disk_spill.is_armed() is True
    assert disk_spill.arm_state()["seconds_left"] > disk_boot.RENEW_AHEAD_SECONDS
    renewals = [r for r in caplog.records if "first renewal" in r.getMessage()]
    assert len(renewals) == 1

    # A second expiry, renewed: the counter moves, the line does not repeat.
    expire_arm()
    disk_boot._state["last_renewal_check_mono"] = None
    with caplog.at_level(logging.WARNING):
        assert disk_boot.renew_if_due()["renewed"] is True
    assert disk_boot.view()["renewals"] == 2
    renewals = [r for r in caplog.records if "first renewal" in r.getMessage()]
    assert len(renewals) == 1, "one renewal line per boot, not one per tick"
    assert disk_boot.view()["last_renewal"].startswith("renewal #2")


def test_the_renewal_tick_is_bounded_and_rides_the_request_path(tier, arm_boot):
    """The tick is cheap per lookup and cannot run twice inside its own floor.

    The plugin has no timer thread and will not grow one, so the tick hangs off the
    scheduler's offer path. That makes two properties load-bearing: the per-lookup cost
    while not due is a monotonic comparison (asserted by the "not due" verdict and the
    unchanged renewal count), and the FIRST LOOKUP AFTER AN EXPIRY re-arms BEFORE the
    gate reads the arm - so the request that arrives after a two-hour idle is served,
    not refused.
    """
    arm_boot()
    expire_arm()
    # The tick has not run since the boot: it is due, so an offer re-arms before gating.
    disk_boot._state["last_renewal_check_mono"] = None
    disk_restore.offer(base.Manager(), base.Req(request_id="after-idle", blocks=4), 0,
                       (0, False))
    assert disk_spill.is_armed() is True, "the tick must precede the gate"
    assert disk_boot.view()["renewals"] == 1
    # Now inside the floor: further lookups report "not due" and renew nothing.
    expire_arm()
    out = disk_boot.renew_if_due()
    assert out["renewed"] is False and out["reason"] == "not due"
    assert disk_boot.view()["renewals"] == 1


def test_renewal_never_arms_a_boot_that_is_not_serving(tier, arm_boot):
    """THE BELT OVER THE LATCH: no posture, no renewal - latch or no latch.

    The latch is set by the verbs, and this test deliberately bypasses them (it lowers
    the switch with the setter directly) to prove renewal has its own second condition.
    Renewal must never put an arm on a box that is not serving: "the plugin armed the
    spill tier by itself and nothing is served" is the state that has cost whole
    evenings, and it must be unreachable from here even with the latch missing.
    """
    arm_boot()
    disk_restore.set_disk_authority(False)      # NOT through the verb: no latch
    assert disk_boot.view()["disarmed_by_operator"] is False
    expire_arm()
    out = disk_boot.renew_if_due()
    assert out["renewed"] is False
    assert "switch is up" in out["reason"]
    assert disk_spill.is_armed() is False


def test_renewal_does_not_touch_a_boot_the_default_never_raised(tier, monkeypatch):
    """Renewal is not a way to arm a boot that never had the posture.

    A boot that came up OFF, NO_STORE or INERT has `honoured: false`, and renewal must
    be
    a no-op there forever. Without this, renewal would be the second arm path in the
    package - and the second arm path is exactly what the 2026-10-03 prohibition is
    about.
    """
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "0")
    disk_spill.arm(60)
    disk_restore.set_disk_authority(True)
    expire_arm()
    out = disk_boot.renew_if_due()
    assert out["renewed"] is False
    assert "boot default" in out["reason"]
    assert disk_spill.is_armed() is False
    assert disk_boot.view()["renewals"] == 0


def test_a_renewal_failure_is_counted_and_loud(tier, arm_boot, monkeypatch, caplog):
    """A renewal that cannot arm is the silent failure this release exists to end.

    So it is counted (`renewal_failures`) and logged - once per distinct reason, never
    once per tick, because the tick runs per request.
    """
    arm_boot()
    expire_arm()
    monkeypatch.setattr(disk_spill, "arm",
                        lambda seconds=None: {"armed": False,
                                              "reason": "the store root vanished"})
    disk_boot._state["last_renewal_check_mono"] = None
    with caplog.at_level(logging.WARNING):
        out = disk_boot.renew_if_due()
    assert out["renewed"] is False
    assert disk_boot.view()["renewal_failures"] == 1
    assert "could NOT be renewed" in caplog.text
    # A repeat of the SAME reason does not print again.
    disk_boot._state["last_renewal_check_mono"] = None
    with caplog.at_level(logging.WARNING):
        disk_boot.renew_if_due()
    assert caplog.text.count("could NOT be renewed") == 1
    assert disk_boot.view()["renewal_failures"] == 2


def test_renewal_reports_an_arm_the_worker_ranks_cannot_read(tier, arm_boot,
                                                             monkeypatch, caplog):
    """A renewal HERE is not a renewal FLEET-WIDE: the arm file is the ranks' only door.

    A renewal that lands in the EngineCore without an arm file is an arm nobody can use,
    so it is counted as a failure and named - the same door the boot's own gate reads.
    """
    arm_boot()
    expire_arm()
    monkeypatch.setattr(disk_spill, "arm",
                        lambda seconds=None: {"armed": True, "seconds": 7200.0,
                                              "arm_file_written": False,
                                              "arm_file_reason": "no diagnostics dir"})
    disk_boot._state["last_renewal_check_mono"] = None
    with caplog.at_level(logging.WARNING):
        assert disk_boot.renew_if_due()["renewed"] is True
    assert disk_boot.view()["renewal_failures"] == 1
    assert "renewal that did not happen" in caplog.text


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

def test_the_prohibition_no_boot_time_re_arm_from_persisted_arm_state(tier,
                                                                       arm_boot,
                                                                       monkeypatch,
                                                                       caplog):
    """THE GUARD. `llm-api/AGENTS.md`, 2026-10-03: "Do not add a boot-time re-arm."

    The forbidden thing is a process RESURRECTING A POSTURE FROM A PERSISTED FILE
    written
    by an earlier boot. The test is behavioural, in the direction that matters: a
    `disk-spill-arm.json` is planted claiming armed for hours, with a boot id that is
    NOT
    this boot's (which is what a reboot leaves behind), and the decision is taken with
    the
    flag OFF and then with the flag ON.

    With the flag OFF the boot comes up DISARMED even though a file claiming "armed,
    expires in hours" sits in the directory: the decision came from the environment, not
    from the file. With the flag ON the boot comes up armed BECAUSE OF THE FLAG - and
    the
    planted file is separately refused by the boot fence, which the fence's own counter
    proves was read and rejected. That is the difference between a boot default and the
    deleted "restore posture on engine-ready" hook, and this test is the line.
    """
    stale = {"version": disk_spill.PROTOCOL_VERSION, "writer": "planted",
             "armed": True, "armed_ts": time.time() - 3600,
             "expires_ts": time.time() + 7200, "seconds": 7200.0,
             "root": tier["root"], "epoch": "this-process-token",
             # `epoch_scope: "process"` on purpose: with a per-process epoch the epoch
             # is
             # NOT evidence (two processes of one boot legitimately disagree on it), so
             # the fence that must fire is the KERNEL BOOT ID - the one field that moves
             # on a reboot, which is the event the prohibition is about.
             "epoch_scope": "process", "pid": 1,
             "boot_id": "00000000-0000-0000-0000-000000000000"}
    with open(arm_path(tier), "w", encoding="utf-8") as handle:
        json.dump(stale, handle)

    # --- the flag OFF: a file claiming "armed for another 2 hours" decides nothing.
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "0")
    with caplog.at_level(logging.WARNING):
        out = arm_boot()
    assert out["outcome"] == disk_boot.OUTCOME_OFF
    assert disk_spill.is_armed() is False, \
        "the boot honoured a previous boot's arm file: the forbidden re-arm"
    assert disk_restore.disk_posture()["serving_as"] == "off"
    assert disk_spill.arm_state()["arm_file_boot_refusals"] >= 1, \
        "the boot fence never read the planted file"
    assert "boot id mismatch" in disk_spill.arm_state()["arm_file_reason"]

    # --- the flag ON: armed because the FLAG says so, and the fence still refuses.
    disk_boot.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    monkeypatch.delenv(config.ENV_DISK_BOOT_ENABLE)
    # Read the file's verdict FIRST, so the fence is provably consulted on the planted
    # file before this boot's own arm replaces it. A previous boot's file arms nobody.
    assert disk_spill.is_armed() is False
    assert disk_spill.arm_state()["arm_file_boot_refusals"] >= 1
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
    # The doors it DOES use: the same two functions the dev verb uses, `arm` twice (the
    # boot and the renewal) and `disarm` for the rollbacks, and the posture setter ONCE.
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
    assert dotted.count("disk_spill.arm") == 2, dotted
    assert dotted.count("disk_spill.disarm") == 2, dotted
    assert dotted.count("disk_restore.set_disk_authority") == 1, dotted
    assert dotted.count("disk_restore.set_offer_switch") == 0, dotted


# --- 8. discoverability -------------------------------------------------------

def test_the_disk_verb_and_the_snapshot_payload_expose_the_boot_default(tier,
                                                                        arm_boot):
    """Status answers "was the boot default on, was it honoured, did renewal run, has an
    operator disarmed" in one read, on the door an operator already opens.

    The four fields the brief asks for, on BOTH payloads (the `/disk` verb's `status`
    and
    the snapshot's `disk_restore.boot_default`), built by the same function - because
    two
    views of one decision is how two views start disagreeing in production.
    """
    arm_boot()
    assert "renewal line" in disk_boot.renew_if_due()["reason"]
    expire_arm()
    # An hour of serving, then an expiry: the floor is dropped to simulate the elapsed
    # hour, because the tick is deliberately bounded to one look per 30 s.
    disk_boot._state["last_renewal_check_mono"] = None
    assert disk_boot.renew_if_due()["renewed"] is True
    verb = disk_store.disk_for_engine(None, {"op": "status"})
    view = verb["disk_boot_default"]
    assert view["requested"] is True
    assert view["honoured"] is True and view["outcome"] == disk_boot.OUTCOME_ARMED
    assert view["renewals"] >= 1
    assert view["disarmed_by_operator"] is False
    assert view["armed_now"] is True
    assert view["arm_seconds"] == disk_spill.ARM_MAX_SECONDS
    assert config.ENV_DISK_BOOT_ENABLE in view["env_var"]
    assert "KVTIER_DISK_BOOT_ENABLE=0" in view["disable_how"]
    # The same view in the snapshot's one-call payload, from the same builder. The live
    # countdown is excluded from the equality (the two reads are milliseconds apart);
    # the
    # KEY SETS are the contract, and every other field must be identical.
    section = disk_restore.scheduler_section()["boot_default"]
    live = {"armed_now", "seconds_left"}
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
    armed = arm_boot()
    disk_boot.reset_state_for_tests()
    monkeypatch.setenv(config.ENV_DISK_BOOT_ENABLE, "0")
    off = arm_boot()
    disk_boot.reset_state_for_tests()
    refused = arm_boot(arena_state="unreadable", arena_reason="exploded")
    for name, out in (("armed", armed), ("off", off), ("refused", refused)):
        assert set(out) == set(never), (name, set(out) ^ set(never))
    # And a live view whose arm reads explode still carries every key: the two live
    # fields are overlaid on the static base, so the base is the contract.
    def explode(*_args, **_kwargs):
        raise RuntimeError("no arm view from here")

    monkeypatch.setattr(disk_spill, "arm_state", explode)
    monkeypatch.setattr(disk_spill, "is_armed", explode)
    broken = disk_boot.view()
    assert set(broken) == set(never)
    assert broken["armed_now"] is None and broken["seconds_left"] is None


def test_the_renewal_verdicts_are_named_not_inferred(tier, arm_boot):
    """Every "no" from the tick says which condition said no.

    Five distinct verdicts (never honoured, plenty of life left, inside the floor,
    latched, posture down) and one acceptance. A bare `renewed: false` on a tier that
    went dark at the two-hour mark is the same silence this release replaces, in a new
    field.
    """
    assert "boot default" in disk_boot.renew_if_due()["reason"]
    arm_boot()
    # Plenty of life left: the tick looks, and declines with the number that decided it.
    assert "renewal line" in disk_boot.renew_if_due()["reason"]
    # And a second call inside the floor does not even look: one comparison, `not due`.
    assert disk_boot.renew_if_due()["reason"] == "not due"
    disk_boot._state["disarmed_by_operator"] = True
    disk_boot._state["latch_reason"] = "POST disarm"
    disk_boot._state["last_renewal_check_mono"] = None
    assert "manual disarm" in disk_boot.renew_if_due()["reason"]
    disk_boot._state["disarmed_by_operator"] = False
    disk_boot._state["last_renewal_check_mono"] = None
    disk_restore.set_disk_authority(False)
    assert "switch is up" in disk_boot.renew_if_due()["reason"]
    disk_restore.set_disk_authority(True)
    disk_boot._state["last_renewal_check_mono"] = None
    expire_arm()
    assert disk_boot.renew_if_due()["renewed"] is True


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


def test_the_payload_says_renewal_rides_the_request_path(tier, arm_boot):
    """The honest limit, in the payload: no timer thread, so a box that serves NOTHING
    lets the arm lapse and the next request re-arms it.

    Said in the payload because a reader who infers "it renews every hour" from a number
    will be surprised by an idle box. The cadence constants are published too, so the
    claim is checkable rather than aspirational.
    """
    out = arm_boot()
    assert "request path" in out["note"]
    assert out["renewal_check_seconds"] == disk_boot.RENEW_CHECK_SECONDS
    assert out["renew_ahead_seconds"] == disk_boot.RENEW_AHEAD_SECONDS
    assert out["arm_max_seconds"] == disk_spill.ARM_MAX_SECONDS
    assert out["renewal_would_run"] is True


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
