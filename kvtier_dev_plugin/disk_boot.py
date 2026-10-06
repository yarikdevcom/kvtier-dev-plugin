"""THE DISK TIER'S OWN BOOT DEFAULT (0.2.33): come up armed and serving, unless one
flag says no.

WHAT THIS IS
------------
Until 0.2.32 every engine boot came up with the SSD tier doing nothing, and stayed
that way until somebody POSTed twice: `{"op":"arm","seconds":7200}` and then
`{"op":"authority","enabled":true}`. Both are per boot, so every restart - every wake,
every crash, every "let me just restart it" - silently reverted to a tier that serves
nothing while `serving_as` truthfully reported `off`. That is the most common way this
feature looks broken to a human, and it cost six evenings of the same conversation.

This module is the fix, and it is exactly three things:

  1. ONE flag, `KVTIER_DISK_BOOT_ENABLE`, whose default is ON. Set `0`/`false`/`off`/
     `no` and the boot behaves exactly as it did in 0.2.32.
  2. ONE boot line (`KVTIER disk boot default=...`) naming the value read, whether it
     came from the environment or from the default, whether it was honoured, and the
     exact way to turn it off - so an operator whose boot misbehaves can find the flag
     from the log without reading source.
  3. ONE renewal path, so the posture does not go dark two hours in.

It invents no third way to arm and no third way to serve: the boot calls
`disk_spill.arm()` and `disk_restore.set_disk_authority()`, the same two functions the
dev verb calls. Everything they refuse, this refuses.

THE PROHIBITION THIS MODULE EXISTS TO RESPECT - READ BEFORE EDITING
-------------------------------------------------------------------
`llm-api/AGENTS.md` (the 2026-10-03 incident section) forbids a BOOT-TIME RE-ARM in
these words: "A stale `disk-spill-arm.json` did NOT re-arm workers nine hours later;
ranks report `armed=False` on their own. Do not add a boot-time re-arm." A proxy-side
"restore posture on engine-ready" hook was written for exactly that and was DELETED
rather than left default-off.

The distinction this module lives on:

  FORBIDDEN: a process resurrecting a posture from a PERSISTED FILE written by an
      earlier boot. "The arm file is still there, so we must still be armed" is a box
      serving live traffic from SSD with nobody having asked, on a boot that may be
      three boots and nine hours after the experiment that wrote the file.

  THIS: the plugin's OWN BOOT DEFAULT, computed fresh at connector creation from one
      environment variable, trusting nothing. It reads no arm file, no state file, no
      marker, nothing any earlier boot wrote. `disk-spill-arm.json` appears in this
      file only as something `disk_spill.arm()` WRITES; it is never read here and never
      consulted for a decision. The boot fence inside `disk_spill._read_arm_file` (which
      refuses a file naming a different kernel boot id) stays the authority on what a
      surviving file may do - which is nothing.

`tests/test_disk_boot_default.py` carries this as an executable guard
(`test_the_prohibition_no_boot_time_re_arm_from_persisted_arm_state`), and it fails if
this module ever grows a read of the arm file or of any previous boot's state.

WHY THE DEFAULT IS ON AND WHY THAT IS NOT "AN ENV VAR ARMS IT"
--------------------------------------------------------------
Every other facility in this package starts OFF and is armed by a token-gated POST,
because a launcher script must not be able to turn a facility that reads or writes real
KV bytes on. That rule is untouched: guard mode, the RAM tier's arm, the byte dump and
the one-slot spill request all still start off and still have no env var.

`KVTIER_DISK_BOOT_ENABLE` is not an exception to that rule, it is a DISABLE on a
default - the same shape as `VLLM_E2_TIER_DEV_BOOT_INERT`, which also only ever turns a
boot effect OFF and is documented as such. Unsetting it cannot arm anything the boot is
not already on; setting it can only take the boot back to the manual posture.
"""

from __future__ import annotations

import logging
import time

from . import config, disk_spill

logger = logging.getLogger("kvtier_dev_plugin.disk_boot")

# The renewal policy, in three numbers, because this is where the honest part of the
# feature lives. `ARM_MAX_SECONDS` (7200 s) is a HARD cap inside `disk_spill`, so a
# boot-default posture that nobody renews goes dark two hours into the boot - silently,
# which is the exact failure this release is chasing (`restorations: 0` because nobody
# re-armed). Renewal is what makes "armed by default" mean "armed for this boot".
#
#   RENEW_CHECK_SECONDS  how often the tick is allowed to look at the arm at all. The
#                        tick rides the request path (see `renew_if_due`), so this is
#                        what makes its per-lookup cost one monotonic comparison.
#   RENEW_AHEAD_SECONDS  how much life must be left before a renewal is DUE. At half
#                        the cap the tier renews hourly, never at the last minute.
RENEW_CHECK_SECONDS = 30.0
RENEW_AHEAD_SECONDS = 3600.0

# The outcomes, as ONE closed vocabulary: the boot line's `disk_boot=` token, the
# payload's `outcome` field and the log line all draw from this list, so a reader can
# never see a word in one place that means something else in another.
OUTCOME_ARMED = "ARMED"          # honoured: armed here, posture up
OUTCOME_OFF = "OFF"              # the flag said no: today's behaviour, exactly
OUTCOME_NO_STORE = "NO_STORE"    # no disk root at all: nothing to arm (normal)
OUTCOME_INERT = "INERT"          # VLLM_E2_TIER_DEV_BOOT_INERT won: no boot effects
OUTCOME_REFUSED = "REFUSED"      # configured, but a gate said no: FAIL-SAFE DISARMED
OUTCOME_UNDECIDED = "UNDECIDED"   # no boot has run in this process yet
OUTCOMES = (OUTCOME_ARMED, OUTCOME_OFF, OUTCOME_NO_STORE, OUTCOME_INERT,
            OUTCOME_REFUSED, OUTCOME_UNDECIDED)

_state: dict = {
    # WHAT THE FLAG SAID, resolved once at boot and never re-read afterwards. Re-reading
    # the environment mid-boot would let a process's behaviour change under a reader who
    # already read the boot line, and the boot line is the record.
    "outcome": OUTCOME_UNDECIDED,
    "requested": False,        # the boot default is ON (from env or from the default)
    "source": None,            # "default" | "env"
    "raw_value": None,         # what the operator actually wrote, verbatim
    "recognised": True,        # False: a value that is not a word in either vocabulary
    "env_var": config.ENV_DISK_BOOT_ENABLE,
    # WHAT THE BOOT DID ABOUT IT.
    "honoured": False,         # True only when the arm AND the posture both landed
    "reason": None,            # why not; or, when honoured, what it did
    "arm_seconds": None,
    "arm_file_written": None,  # the fleet's door: no file, no armed workers
    "serving_as_after": "off",  # the posture as this boot left it
    "disk_configured": False,
    # RENEWAL, AND THE OPERATOR'S VETO OVER IT.
    "renewals": 0,
    "renewal_failures": 0,
    "last_renewal": None,      # one sentence, newest
    "last_renewal_check_mono": None,
    # THE LATCH. Set by a human's `{"op":"disarm"}` or `{"op":"authority","enabled":
    # false}`, cleared only by a human's `{"op":"arm"}`. While it is set, renewal does
    # NOTHING: a manual disarm always beats the boot default, and there is no path -
    # renewal, a second boot in one process, anything - that re-arms a box an operator
    # just disarmed.
    "disarmed_by_operator": False,
    "latch_reason": None,
    # BOOKKEEPING.
    "boots": 0,
    "line_logged": False,
    "notes": {},               # log-once ledger, the `asleep_notice_once` idiom
}

# The payload's key set, named once so the degraded view cannot drift from the healthy
# one (the rule `test_the_view_carries_the_same_keys_on_every_path` enforces).
_KEYS = ("outcome", "requested", "source", "raw_value", "recognised", "env_var",
         "honoured", "reason", "arm_seconds", "arm_file_written", "serving_as_after",
         "disk_configured", "renewals", "renewal_failures", "last_renewal",
         "disarmed_by_operator", "latch_reason", "boots", "armed_now", "seconds_left",
         "renewal_would_run", "renewal_check_seconds", "renew_ahead_seconds",
         "arm_max_seconds", "disable_how", "note")


def reset_state_for_tests() -> None:
    """A fresh boot's view of the flag. Test-only, like every sibling reset."""
    _state.update({
        "outcome": OUTCOME_UNDECIDED, "requested": False, "source": None,
        "raw_value": None, "recognised": True,
        "env_var": config.ENV_DISK_BOOT_ENABLE, "honoured": False, "reason": None,
        "arm_seconds": None, "arm_file_written": None, "serving_as_after": "off",
        "disk_configured": False, "renewals": 0, "renewal_failures": 0,
        "last_renewal": None, "last_renewal_check_mono": None,
        "disarmed_by_operator": False, "latch_reason": None, "boots": 0,
        "line_logged": False, "notes": {}})


def _route() -> str:
    return config.ROUTE_PREFIX + config.ROUTE_DISK


def _note_once(key: str, message: str) -> bool:
    """One log line per episode, never one per tick. Returns True when it logged."""
    if _state["notes"].get(key):
        return False
    _state["notes"][key] = True
    logger.warning("kvtier-dev-plugin: %s", message)
    return True


# --- the view an operator reads ----------------------------------------------

def _arm_max_seconds() -> float:
    """The arm's hard cap, or a constant. A view is never worth a fault."""
    try:
        return float(disk_spill.ARM_MAX_SECONDS)
    except Exception:  # noqa: BLE001 - a missing constant reads as the shipped cap
        return 7200.0


def static_view() -> dict:
    """The view WITHOUT the live arm reads: the same key set, built from state alone.

    This is the fallback a caller uses when reading the arm is not possible, and it is
    also what makes "the degraded view has the same keys" a structural property rather
    than a copy: `view()` overlays the two live fields on top of this dict, so the two
    cannot drift apart by construction (and
    `test_the_view_carries_the_same_keys_on_every_path` pins it anyway).
    """
    honoured = bool(_state["honoured"])
    latched = bool(_state["disarmed_by_operator"])
    posture_up = _posture_up()
    return {
        "outcome": _state["outcome"],
        "requested": bool(_state["requested"]),
        "source": _state["source"],
        "raw_value": _state["raw_value"],
        "recognised": bool(_state["recognised"]),
        "env_var": config.ENV_DISK_BOOT_ENABLE,
        "honoured": honoured,
        "reason": _state["reason"],
        "arm_seconds": _state["arm_seconds"],
        "arm_file_written": _state["arm_file_written"],
        "serving_as_after": _state["serving_as_after"],
        "disk_configured": bool(_state["disk_configured"]),
        "renewals": int(_state["renewals"]),
        "renewal_failures": int(_state["renewal_failures"]),
        "last_renewal": _state["last_renewal"],
        "disarmed_by_operator": latched,
        "latch_reason": _state["latch_reason"],
        "boots": int(_state["boots"]),
        "armed_now": None,
        "seconds_left": None,
        # THE ANSWER TO "WOULD THE TIER COME BACK IF IT EXPIRED?", not a prediction:
        # renewal runs only while the boot default was on, this boot honoured it, no
        # operator has disarmed since, AND the posture is still up. The last clause is
        # the belt over the latch - renewal never arms a box that is not serving.
        "renewal_would_run": bool(honoured and not latched and posture_up),
        "renewal_check_seconds": RENEW_CHECK_SECONDS,
        "renew_ahead_seconds": RENEW_AHEAD_SECONDS,
        "arm_max_seconds": _arm_max_seconds(),
        "disable_how": disable_how(),
        "note": ("the disk tier's BOOT DEFAULT (0.2.33): the boot arms the spill path "
                 "and raises this boot's serve posture, computed fresh from "
                 f"{config.ENV_DISK_BOOT_ENABLE} and nothing else. It reads no arm "
                 "file and no state an earlier boot wrote - a restart is still a "
                 "rollback and a previous boot's arm file still arms nobody. Renewal "
                 "rides the request path, so a boot that serves nothing lets the arm "
                 "lapse and the next request re-arms it; a manual POST "
                 f"{_route()} {{\"op\":\"disarm\"}} stops renewal for the boot and "
                 "nothing re-arms it behind your back"),
    }


def view() -> dict:
    """The boot default, as one stable-key dict. Never raises.

    EVERY PATH RETURNS EVERY KEY: a boot that never ran the disk default answers with
    the same shape as one that did, because a degraded view that loses fields is a
    payload whose absence a reader has to interpret at 3 a.m. The two live fields
    (`armed_now`, `seconds_left`) come from `disk_spill`, the owner of the arm; nothing
    here recomputes an arm verdict, and `static_view()` above is the base this overlays
    them on - so the healthy and degraded views share their key set structurally.
    """
    out = static_view()
    try:
        out["armed_now"] = bool(disk_spill.is_armed())
        out["seconds_left"] = disk_spill.arm_state().get("seconds_left")
    except Exception as exc:  # noqa: BLE001 - a view never costs a payload
        logger.debug("kvtier-dev-plugin: disk_boot could not read the arm: %s", exc)
    return {key: out[key] for key in _KEYS}


def disable_how() -> str:
    """The sentence that tells an operator how to turn this off, without source.

    Both halves belong here, because there are two different "offs" and an operator
    under pressure needs the right one: the ENVIRONMENT one (the next boot is manual
    again) and the MID-BOOT one (this boot, right now, and it sticks - renewal obeys
    it). Naming both is the whole reason the boot line is allowed to be long.
    """
    return (f"to stop every future boot doing this: set "
            f"{config.ENV_DISK_BOOT_ENABLE}=0 in the engine's environment and restart "
            f"({config.ENV_BOOT_INERT}=1 turns off the whole auto-boot, not just the "
            f"disk tier); to stop THIS boot right now: POST {_route()} "
            "{\"op\":\"authority\",\"enabled\":false} and then POST "
            f"{_route()} {{\"op\":\"disarm\"}} - a manual disarm latches, and renewal "
            "never re-arms a box the operator just disarmed")


# --- the boot decision --------------------------------------------------------

def _posture_up() -> bool:
    """Is this boot serving from disk at all? Reads `disk_restore`, decides nothing."""
    try:
        from . import disk_restore  # noqa: PLC0415 - it imports this module lazily
        posture = disk_restore.disk_posture()
        return bool(posture["disk_authority"] or posture["enabled"])
    except Exception as exc:  # noqa: BLE001 - unreadable posture is no posture
        logger.debug("kvtier-dev-plugin: disk_boot could not read the posture: %s", exc)
        return False


def _serving_as() -> str:
    try:
        from . import disk_restore  # noqa: PLC0415 - it imports this module lazily
        return str(disk_restore.serving_as())
    except Exception as exc:  # noqa: BLE001 - a view falls back to the safe word
        logger.debug("kvtier-dev-plugin: disk_boot could not read serving_as: %s", exc)
        return "off"


def _gate_verdict(pre: dict) -> tuple:
    """(outcome, reason) when a PRECONDITION says the boot must not come up serving.

    Every gate here is a fact the boot already measured, read from the report handed in
    - not a new rule. Each one names a state where arming would produce the boot this
    suite keeps being written to catch: armed-looking, serving nothing.

    Order matters for the operator's sake, not for safety: the flag first (it is his
    decision), then the inert boot (also his, and global), then "there is no disk tier"
    (normal, and therefore NOT a refusal - a boot with no `VLLM_E2_TIER_DISK_ROOT` has
    nothing to honour and nothing to complain about), then the four preconditions whose
    failure means arming would lie.
    """
    if not _state["requested"]:
        return OUTCOME_OFF, (
            f"{config.ENV_DISK_BOOT_ENABLE}={_state['raw_value']!r}: the boot default "
            "is OFF, so this boot comes up disarmed with serving_as=off exactly as "
            "through 0.2.32, and the dev-verb POSTs are the only lever")
    if bool(pre.get("inert")):
        return OUTCOME_INERT, (
            f"{config.ENV_BOOT_INERT} is set: that is the boot-wide opt-out and it "
            "covers the disk tier too - the boot installs and stops there, so nothing "
            "here arms or serves. It is a stronger switch than "
            f"{config.ENV_DISK_BOOT_ENABLE} and wins")
    if not _state["disk_configured"]:
        return OUTCOME_NO_STORE, (
            f"{config.ENV_DISK_ROOT} is not set (or names a refused path), so there is "
            "no store to arm: arming would be a knob that lies. Not a fault - this "
            "engine has no SSD tier")
    if not bool(pre.get("tier_armed")):
        return OUTCOME_REFUSED, (
            "the KV tier itself is not armed by this boot, so the connector schedules "
            "no load for a disk offer to ride on: a disk-only arm here is a tier that "
            "writes bytes nobody can be served from. Arm the tier (or fix the install "
            "verdict) and the disk default follows it")
    if not bool(pre.get("restore_fitted")):
        return OUTCOME_REFUSED, (
            "the scheduler half of the restore layer is not fitted in this process, so "
            "this boot could write spilled KV and never read a byte of it back. Coming "
            "up armed-and-serving on that boot is the shape that has been mistaken for "
            "a working tier for hours at a time")
    arena = str(pre.get("arena_state") or "unseen")
    if arena in ("tiny", "no_pool", "unreadable"):
        return OUTCOME_REFUSED, (
            f"the host-RAM arena cannot service an offer ({arena}: "
            f"{str(pre.get('arena_reason') or 'no reason recorded')[:200]}): the disk "
            "tier owns no KV bytes and moves them through this arena in both "
            "directions, so an armed tier over it is inert while its arm file says "
            "ARMED")
    journal_writable = pre.get("journal_writable")
    if journal_writable is False:
        return OUTCOME_REFUSED, (
            f"the spill JOURNAL cannot be written where it lives "
            f"({str(pre.get('journal_error') or 'no reason recorded')[:200]}): every "
            "spill this boot made would commit a KV file that no index can ever name, "
            "so serving from disk would be a promise the store cannot keep")
    return OUTCOME_ARMED, None


def boot_enable(preconditions: dict | None = None) -> dict:
    """THE BOOT DECISION, run once per boot by `backport.boot_from_connector`.

    `preconditions` is the boot's own report, not a new measurement: `{inert,
    tier_armed, restore_fitted, arena_state, arena_reason, journal_writable,
    journal_error}`. A missing or unreadable entry reads as "the gate says no", because
    a boot that cannot tell whether its arena works is a boot that must not come up
    serving.

    THE FAIL-SAFE DIRECTION, and it is the whole design: every possible outcome except
    `ARMED` leaves this boot DISARMED, and `ARMED` requires BOTH halves to have landed.
    If the posture goes up and the arm does not - or the arm lands and the posture does
    not - the pair is rolled back with `disk_spill.disarm()`, because a partially-armed
    fleet is worse than a disarmed one: the scheduler's proof is an AND over ranks, and
    one disarmed rank keeps the scheduler closed while the armed ones do work nobody
    asked for. Nothing here can raise either half again once an operator has disarmed.

    Never raises. A boot report must not cost an engine its boot, so the outermost
    `try` is the contract; a swallowed failure is reported as `REFUSED` and counted as a
    plugin error by the caller's own bookkeeping.
    """
    pre = preconditions if isinstance(preconditions, dict) else {}
    try:
        resolved = config.disk_boot_enable()
        _state["boots"] += 1
        _state["requested"] = bool(resolved["enabled"])
        _state["source"] = resolved["source"]
        _state["raw_value"] = resolved["raw"]
        _state["recognised"] = bool(resolved.get("recognised", True))
        _state["env_var"] = config.ENV_DISK_BOOT_ENABLE
        try:
            _state["disk_configured"] = bool(
                disk_spill.configured()["configured"])
        except Exception as exc:  # noqa: BLE001 - unreadable config is no config
            logger.debug("kvtier-dev-plugin: disk_boot could not read the tier "
                         "configuration: %s", exc)
            _state["disk_configured"] = False
        outcome, reason = _gate_verdict(pre)
        if outcome is not OUTCOME_ARMED:
            _state["outcome"] = outcome
            _state["honoured"] = False
            _state["reason"] = reason
            _state["serving_as_after"] = _serving_as()
            _log_boot_line()
            return view()
        # THE TWO HALVES, in the order that cannot leave a posture with no arm: the
        # spill path first (it is the thing that can fail - no root, no channel to the
        # ranks), then the serve posture. `disk_spill.arm()` is the same call the dev
        # verb makes, with the maximum legal TTL: inventing a third arm path is how a
        # boot ends up armed in a way no disarm can reach.
        verdict = disk_spill.arm(disk_spill.ARM_MAX_SECONDS)
        _state["arm_seconds"] = verdict.get("seconds")
        _state["arm_file_written"] = bool(verdict.get("arm_file_written"))
        if not verdict.get("armed"):
            _state["outcome"] = OUTCOME_REFUSED
            _state["honoured"] = False
            _state["reason"] = (f"arming the spill path was refused: "
                                f"{verdict.get('reason') or 'no reason given'}")
            _state["serving_as_after"] = _serving_as()
            _log_boot_line()
            return view()
        if not _state["arm_file_written"]:
            # THE FLEET'S DOOR. A worker's arm is the FILE - the EngineCore has no KV
            # memory and no channel to a worker, so an arm that could not be published
            # arms one process that can do nothing and leaves the four ranks that could
            # do it disarmed. That is a partially-armed fleet by definition, so it is
            # refused here rather than reported as a success.
            disk_spill.disarm()
            _state["outcome"] = OUTCOME_REFUSED
            _state["honoured"] = False
            _state["reason"] = (
                f"the arm could not be published to the worker ranks "
                f"({verdict.get('arm_file_reason') or 'no arm file was written'}): a "
                "spill arm reaches a worker only through the diagnostics directory "
                f"({config.ENV_DIAG_DIR}), and an EngineCore-only arm would leave the "
                "fleet half-armed, which the scheduler's AND-over-ranks proof then "
                "closes anyway - while looking armed")
            _state["serving_as_after"] = _serving_as()
            _log_boot_line()
            return view()
        from . import disk_restore  # noqa: PLC0415 - it imports this module lazily
        raised = disk_restore.set_disk_authority(True)
        if not raised.get("ok"):
            # Roll the arm back: the posture is the half an operator asked for, and an
            # armed tier that serves nothing is the state he is trying NOT to boot into.
            disk_spill.disarm()
            _state["arm_seconds"] = None
            _state["arm_file_written"] = None
            _state["outcome"] = OUTCOME_REFUSED
            _state["honoured"] = False
            _state["reason"] = (f"the serve posture could not be raised: "
                                f"{raised.get('reason') or 'no reason given'}; the "
                                "spill arm was rolled back so this boot is uniformly "
                                "disarmed")
            _state["serving_as_after"] = _serving_as()
            _log_boot_line()
            return view()
        _state["outcome"] = OUTCOME_ARMED
        _state["honoured"] = True
        _state["serving_as_after"] = _serving_as()
        _state["reason"] = (
            f"armed the spill path for {verdict.get('seconds')} s and raised this "
            f"boot's serve posture (serving_as={_state['serving_as_after']}), with no "
            "POST; renewal rides the request path while the posture stays up")
        _log_boot_line()
        return view()
    except Exception as exc:  # noqa: BLE001 - a boot default must never cost a boot
        _state["outcome"] = OUTCOME_REFUSED
        _state["honoured"] = False
        _state["reason"] = (f"the boot default raised {type(exc).__name__} and was "
                            "abandoned; the boot continues disarmed")
        try:
            _log_boot_line()
        except Exception as exc:  # noqa: BLE001 - nothing here is worth a boot
            logger.debug("kvtier-dev-plugin: the disk boot line could not be printed: "
                         "%s", exc)
        return view()


def _log_boot_line() -> None:
    """THE ONE LINE. Printed once per process, at WARNING, like `BOOT[...]`.

    It has to answer, without the operator opening a source file or a runbook: what the
    flag said, where that came from, whether the boot honoured it, and HOW TO TURN IT
    OFF. The "how to turn it off" half is not decoration - it is the reason this flag
    ships with a default of ON at all, and the reason the sentence is allowed to be
    long. A boot that misbehaves should be one `systemctl edit` away from the old
    behaviour, not a code read away.
    """
    if _state["line_logged"]:
        return
    _state["line_logged"] = True
    value = "ON" if _state["requested"] else "OFF"
    # The parenthetical is the whole provenance in one place: either "this variable was
    # never set, so the shipped default decided", or the exact text the operator wrote.
    # Naming the variable twice ("KVTIER_DISK_BOOT_ENABLE KVTIER_DISK_BOOT_ENABLE=0") is
    # the kind of noise that makes an operator distrust the line.
    origin = (f"{config.ENV_DISK_BOOT_ENABLE} unset -> default ON"
              if _state["source"] == "default"
              else f"{config.ENV_DISK_BOOT_ENABLE}={_state['raw_value']!r}")
    head = (f"KVTIER disk boot default={value} ({origin}): ")
    if not _state["recognised"]:
        # Say it, because a value that is not a disable word READS like one to the
        # person who typed it. `KVTIER_DISK_BOOT_ENABLE=flase` must not quietly leave a
        # boot serving from disk that its operator meant to take out of service.
        head += (f"NOTE: {config.ENV_DISK_BOOT_ENABLE}={_state['raw_value']!r} is not "
                 "a recognized value; only 0/false/off/no disable the boot default, so "
                 "it was read as ON. ")
    outcome = _state["outcome"]
    if outcome == OUTCOME_ARMED:
        body = (f"HONOURED: armed the spill path for {_state['arm_seconds']} s and set "
                f"this boot's serve posture, so it comes up "
                f"serving_as={_state['serving_as_after']} with NO POST. Renewal keeps "
                f"it up while the posture stays up (>= {int(RENEW_AHEAD_SECONDS)} s of "
                f"the {int(disk_spill.ARM_MAX_SECONDS)} s cap left, checked at most "
                f"once every {int(RENEW_CHECK_SECONDS)} s on the request path). ")
    elif outcome == OUTCOME_OFF:
        body = ("NOT applied: this boot comes up DISARMED with serving_as=off, exactly "
                "as every release through 0.2.32, and the two dev-verb POSTs are the "
                f"only lever (POST {_route()} {{\"op\":\"arm\",\"seconds\":7200}} then "
                f"{{\"op\":\"authority\",\"enabled\":true}}). ")
    elif outcome == OUTCOME_INERT:
        body = ("NOT applied: the boot-wide inert opt-out already decided that this "
                "boot arms nothing. ")
    elif outcome == OUTCOME_NO_STORE:
        body = ("NOT applied: this engine has no SSD tier configured, so there is "
                "nothing to arm - not a fault. ")
    else:
        body = (f"NOT HONOURED: {_state['reason']} -> this boot comes up DISARMED with "
                "serving_as=off, and nothing else about the boot changed. ")
    logger.warning("kvtier-dev-plugin: %s%s", head + body, disable_how())


# --- renewal ------------------------------------------------------------------

def renew_if_due() -> dict:
    """Keep a boot-default posture alive. Cheap, contained, and obedient to the latch.

    WHY IT RIDES THE REQUEST PATH, stated plainly because it is a real limit. This
    plugin has no timer thread and will not grow one: every periodic thing it does is
    driven by the engine's own step or request path, and inventing a thread inside a
    plugin that patches a serving engine is a worse risk than a renewal that needs a
    request. So the tick hangs off the scheduler's offer path, which is (a) the only
    path that consults the arm at all and (b) therefore exactly the moment a lapse
    would cost somebody a disk restore. The consequence, in the operator's terms: a box
    that serves NOTHING for two hours lets the arm expire, and the NEXT request re-arms
    it before the gate reads it - the tier never serves a request from an expired arm,
    which is the failure that actually costs tokens. The payload's `note` and
    `renewal_would_run` say so in words rather than leaving the cadence to be inferred.

    THREE CONDITIONS, all required, and the third is the safety one:
      * the boot default was ON and this boot HONOURED it (renewal is not a way to
        arm a boot that never had the posture);
      * no operator has disarmed since (`disarmed_by_operator` - a manual disarm always
        beats the boot default, which is the property the brief makes non-negotiable);
      * the posture is STILL UP. If both serve switches are down, there is nothing to
        serve and nothing to renew: this is the belt over the latch, so even a latch
        this module somehow missed cannot re-arm a box that is not serving.
    """
    out = {"renewed": False, "reason": None}
    try:
        if not _state["honoured"] or not _state["requested"]:
            out["reason"] = ("the boot default did not raise this boot's disk posture, "
                             "so there is nothing for renewal to keep alive")
            return out
        if _state["disarmed_by_operator"]:
            out["reason"] = (f"a human disarmed this boot "
                             f"({_state['latch_reason'] or 'no reason recorded'}): a "
                             "manual disarm beats the boot default and renewal will "
                             "not re-arm it")
            return out
        if not _posture_up():
            out["reason"] = ("neither serve switch is up any more, so there is nothing "
                             "to serve and nothing to renew")
            return out
        now = time.monotonic()
        last = _state["last_renewal_check_mono"]
        if isinstance(last, (int, float)) and (now - last) < RENEW_CHECK_SECONDS:
            out["reason"] = "not due"
            return out
        _state["last_renewal_check_mono"] = now
        state = disk_spill.arm_state()
        left = state.get("seconds_left")
        if isinstance(left, (int, float)) and left > RENEW_AHEAD_SECONDS:
            out["reason"] = f"{round(float(left), 1)} s of arm left (above the " \
                            f"{int(RENEW_AHEAD_SECONDS)} s renewal line)"
            return out
        # Reaching here means the arm is due, or GONE (expired, or dropped under us).
        # Re-arming an expired boot-default arm is the point of renewal; the two
        # conditions above are what make it safe.
        reason = "the arm expired" if left is None else \
            f"{round(float(left), 1)} s of arm left, below the " \
            f"{int(RENEW_AHEAD_SECONDS)} s renewal line"
        verdict = disk_spill.arm(disk_spill.ARM_MAX_SECONDS)
        if not verdict.get("armed"):
            _state["renewal_failures"] += 1
            line = (f"DISK[boot] the boot-default disk posture could NOT be renewed "
                    f"({reason}): {verdict.get('reason') or 'no reason given'}. The "
                    "SSD tier stops serving when the arm lapses - this is the silent "
                    "failure this release exists to end, so read it, do not ignore it")
            # One line per distinct reason, never one per tick: the tick runs per
            # request, and a diagnostic that repeats every step stops being read.
            _note_once(f"renew-fail:{str(verdict.get('reason'))[:80]}", line)
            out["reason"] = str(verdict.get("reason"))
            return out
        _state["renewals"] += 1
        _state["arm_seconds"] = verdict.get("seconds")
        _state["arm_file_written"] = bool(verdict.get("arm_file_written"))
        _state["last_renewal"] = (f"renewal #{int(_state['renewals'])}: re-armed for "
                                  f"{verdict.get('seconds')} s ({reason})")
        if int(_state["renewals"]) == 1:
            # THE FIRST RENEWAL PER BOOT, LOUD ONCE: it is the moment a 0.2.33 boot
            # proves it will not go dark at the two-hour mark, and the only evidence in
            # the log that renewal is running at all.
            logger.warning("kvtier-dev-plugin: DISK[boot] first renewal of the "
                           "boot-default disk posture: re-armed the spill path for "
                           "%s s (%s). Renewal rides the request path, so it only runs "
                           "while the engine is taking requests; a manual POST %s "
                           "{\"op\":\"disarm\"} stops it for this boot and nothing "
                           "re-arms behind your back",
                           verdict.get("seconds"), reason, _route())
        if not verdict.get("arm_file_written"):
            _state["renewal_failures"] += 1
            _note_once("renew-nofile",
                       "DISK[boot] a renewal landed HERE but its arm file did not, so "
                       "the worker ranks cannot see it "
                       f"({verdict.get('arm_file_reason')}): check "
                       f"{config.ENV_DIAG_DIR}. A renewal the ranks cannot read is an "
                       "EngineCore-only arm, which is a renewal that did not happen")
        out["renewed"] = True
        out["reason"] = None
        return out
    except Exception as exc:  # noqa: BLE001 - a renewal never costs a request
        _state["renewal_failures"] += 1
        snapshot_error(exc, "disk_boot renewal")
        out["reason"] = f"the renewal tick raised {type(exc).__name__}"
        return out


def snapshot_error(exc: BaseException, where: str) -> None:
    """Route an internal failure to the plugin's error ledger, without depending on it.

    `snapshot._error` is the package's ledger, but `snapshot` is the read side and this
    module runs on the scheduler's request path; the import is done lazily and its own
    failure is ignored, because a renewal tick that raises on the way to reporting a
    raise would be a bug report that files itself as a crash.
    """
    try:
        from . import snapshot  # noqa: PLC0415 - the read side, never imported at top
        snapshot._error(exc, where)
    except Exception:  # noqa: BLE001 - the ledger is not worth a second fault
        logger.debug("kvtier-dev-plugin: %s raised %s (unreported)", where,
                     type(exc).__name__)


# --- the operator's veto ------------------------------------------------------

def note_operator_disarm(reason: str) -> dict:
    """LATCH the operator's veto: renewal stops for this boot, and nothing un-latches it
    except an explicit `note_operator_arm()`.

    Called from the dev-verb handlers in `disk_store.py` - the human's door - and from
    nowhere else. That placement is the point: this module's own rollback on a failed
    boot calls `disk_spill.disarm()` directly and does NOT latch, because a rollback is
    not an operator's instruction, while `POST {"op":"disarm"}` is one and outranks a
    boot default forever after.
    """
    _state["disarmed_by_operator"] = True
    _state["latch_reason"] = str(reason)[:200]
    logger.warning("kvtier-dev-plugin: DISK[boot] the operator's posture now beats the "
                   "boot default for this boot (%s): renewal will not re-arm it. %s",
                   _state["latch_reason"], disable_how())
    return {"disarmed_by_operator": True, "reason": _state["latch_reason"]}


def note_operator_arm(reason: str) -> dict:
    """An operator armed by hand: lift the veto, so renewal may resume.

    A manual `{"op":"arm"}` after a manual disarm is the operator asking for the tier
    back. Leaving the latch set would silently disable renewal for the rest of the boot
    on a box he just chose to serve from disk - which is the two-hour dark tier again,
    wearing a latch. Only a human reaches this.
    """
    if not _state["disarmed_by_operator"]:
        return {"disarmed_by_operator": False, "reason": None}
    logger.warning("kvtier-dev-plugin: DISK[boot] a manual arm lifts the operator's "
                   "disarm for this boot (%s): renewal resumes if the boot default "
                   "raised this boot in the first place", str(reason)[:200])
    _state["disarmed_by_operator"] = False
    _state["latch_reason"] = None
    return {"disarmed_by_operator": False, "reason": None}


# --- the boot line's one token -------------------------------------------------

def field_token(outcome: str | None) -> str:
    """The `disk_boot=` token: one of OUTCOMES, uppercase, never a space.

    Kept here rather than in `backport` so the vocabulary has ONE owner: the payload's
    `outcome`, the boot line's `disk_boot=` and the log line's headline cannot drift to
    three spellings of the same state.
    """
    token = str(outcome or OUTCOME_UNDECIDED).upper()
    return token if token in OUTCOMES else OUTCOME_REFUSED
