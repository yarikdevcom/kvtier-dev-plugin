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
  3. ONE fail-safe direction: every outcome that is not ARMED ends with the tier
     DISARMED - `disarm()` deletes the arm file, so a boot that never chose to arm
     leaves nothing behind for a later boot, or a later process, to read.

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

SINCE 0.2.35 THERE IS NO RENEWAL IN THIS MODULE, because there is nothing left to
renew: the arm is a LATCH (see `disk_spill`'s module docstring for the ruling - an
expired arm froze LRU shedding as well as growth, and no setting of the timer bounded
the store). That removal leaves one hole this module is now responsible for. The boot
fence in `disk_spill._arm_boot_fence` separates KERNEL boots by boot id, so it cannot
see an ENGINE RESTART inside one long uptime: an arm file written by boot #1 of a
two-week uptime is still "this boot" to the fence, and while a TTL existed, the TTL is
what killed such a file. The rule that replaces the clock is:

    THE ONLY THING THAT ARMS A PROCESS IS THAT PROCESS'S OWN BOOT DECISION.

Hence the fail-safe below - `boot_enable()` calls `disk_spill.disarm()` on EVERY
outcome except `ARMED`, and `disarm()` removes the file. A boot that did not choose to
arm therefore erases whatever an earlier engine boot left, instead of inheriting it.
`tests/test_disk_boot_default.py`
(`test_the_boot_disarms_an_arm_file_left_by_an_earlier_engine_boot_on_the_same_kernel`)
plants exactly that file and fails if this module ever grows a dependency on one.
"""

from __future__ import annotations

import logging

from . import config, disk_spill

logger = logging.getLogger("kvtier_dev_plugin.disk_boot")

# NO RENEWAL POLICY LIVES HERE ANY MORE (0.2.35). Until then this module held
# `RENEW_CHECK_SECONDS` / `RENEW_AHEAD_SECONDS` and a `renew_if_due()` tick that rode
# the request path to beat the arm's 7,200 s cap - a mechanism whose entire purpose was
# to keep a TIMER from going dark. The arm is a latch now, so the tick, its constants,
# its counters, its two live view fields and the `disk_restore.offer()` call that drove
# it are all deleted: there is no lapse to forestall, and a heartbeat that re-arms a
# box an operator has disarmed is exactly what the veto below exists to prevent.

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
    "arm_file_written": None,  # the fleet's door: no file, no armed workers
    "serving_as_after": "off",  # the posture as this boot left it
    "disk_configured": False,
    # THE OPERATOR'S VETO. Set by a human's `{"op":"disarm"}` or
    # `{"op":"authority","enabled":false}`, cleared only by a human's `{"op":"arm"}`.
    # IN 0.2.34 THIS GATED `renew_if_due()` AND NOTHING ELSE; with renewal gone it
    # could have been deleted, and it is kept instead because it has a stronger job to
    # do now that the arm never lapses on its own: `boot_enable()` REFUSES to arm while
    # it stands. That is what makes "a manual disarm always beats the boot default"
    # true for the whole life of a process rather than until the next timer expired -
    # a second connector boot inside one process (which the tests do, and a
    # re-instantiated engine does) can no longer re-arm a box an operator just
    # disarmed. Nothing is persisted here: the veto is this process's memory only, and
    # the next process starts with the environment's word, exactly as the boot default
    # doctrine requires.
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
         "honoured", "reason", "arm_file_written", "serving_as_after",
         "disk_configured", "disarmed_by_operator", "latch_reason", "boots",
         "armed_now", "arm_expires", "disable_how", "note")


def reset_state_for_tests() -> None:
    """A fresh boot's view of the flag. Test-only, like every sibling reset."""
    _state.update({
        "outcome": OUTCOME_UNDECIDED, "requested": False, "source": None,
        "raw_value": None, "recognised": True,
        "env_var": config.ENV_DISK_BOOT_ENABLE, "honoured": False, "reason": None,
        "arm_file_written": None, "serving_as_after": "off",
        "disk_configured": False,
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

def static_view() -> dict:
    """The view WITHOUT the live arm read: the same key set, built from state alone.

    This is the fallback a caller uses when reading the arm is not possible, and it is
    also what makes "the degraded view has the same keys" a structural property rather
    than a copy: `view()` overlays the live field on top of this dict, so the two
    cannot drift apart by construction (and
    `test_the_view_carries_the_same_keys_on_every_path` pins it anyway).
    """
    return {
        "outcome": _state["outcome"],
        "requested": bool(_state["requested"]),
        "source": _state["source"],
        "raw_value": _state["raw_value"],
        "recognised": bool(_state["recognised"]),
        "env_var": config.ENV_DISK_BOOT_ENABLE,
        "honoured": bool(_state["honoured"]),
        "reason": _state["reason"],
        "arm_file_written": _state["arm_file_written"],
        "serving_as_after": _state["serving_as_after"],
        "disk_configured": bool(_state["disk_configured"]),
        "disarmed_by_operator": bool(_state["disarmed_by_operator"]),
        "latch_reason": _state["latch_reason"],
        "boots": int(_state["boots"]),
        "armed_now": None,
        # THE ANSWER TO "WHEN DOES THIS GO DARK?", in the field that used to be
        # `seconds_left` and carried a number. Never. 0.2.35 made the arm a latch, so a
        # reader that used to count down a countdown is given the fact instead of an
        # absence - the absence would have to be interpreted at 3 a.m., the fact does
        # not. (The proxy's countdown column is the other half of this change and is
        # removed in the proxy repo, in sequence after this release.)
        "arm_expires": False,
        "disable_how": disable_how(),
        "note": ("the disk tier's BOOT DEFAULT (0.2.33): the boot arms the spill path "
                 "and raises this boot's serve posture, computed fresh from "
                 f"{config.ENV_DISK_BOOT_ENABLE} and nothing else. It reads no arm "
                 "file and no state an earlier boot wrote - a restart is still a "
                 "rollback and a previous boot's arm file still arms nobody. SINCE "
                 "0.2.35 THE ARM IS A LATCH: there is no renewal, because there is no "
                 "expiry, so what keeps the posture up is that nothing lowers it. The "
                 "consequence this module owns is the fail-safe - an outcome that is "
                 "not ARMED calls disarm(), which DELETES the arm file, so an engine "
                 f"restart inside one kernel boot cannot inherit one. POST {_route()} "
                 "{\"op\":\"disarm\"} lowers the latch for this boot and the "
                 "operator veto below keeps this module's own boot decision from "
                 "raising it again"),
    }


def view() -> dict:
    """The boot default, as one stable-key dict. Never raises.

    EVERY PATH RETURNS EVERY KEY: a boot that never ran the disk default answers with
    the same shape as one that did, because a degraded view that loses fields is a
    payload whose absence a reader has to interpret at 3 a.m. The one live field
    (`armed_now`) comes from `disk_spill`, the owner of the arm; nothing here
    recomputes an arm verdict, and `static_view()` above is the base this overlays it
    on - so the healthy and degraded views share their key set structurally.
    """
    out = static_view()
    try:
        out["armed_now"] = bool(disk_spill.is_armed())
    except Exception as exc:  # noqa: BLE001 - a view never costs a payload
        logger.debug("kvtier-dev-plugin: disk_boot could not read the arm: %s", exc)
    return {key: out[key] for key in _KEYS}


def disable_how() -> str:
    """The sentence that tells an operator how to turn this off, without source.

    Both halves belong here, because there are two different "offs" and an operator
    under pressure needs the right one: the ENVIRONMENT one (the next boot is manual
    again) and the MID-BOOT one (this boot, right now). Since 0.2.35 the mid-boot one is
    the STRONGER of the two: the arm is a latch, so a disarm holds until someone arms
    again, and the operator veto stops this module's own boot decision from re-arming
    the box either. Naming both is the whole reason the boot line is allowed to be long.
    """
    return (f"to stop every future boot doing this: set "
            f"{config.ENV_DISK_BOOT_ENABLE}=0 in the engine's environment and restart "
            f"({config.ENV_BOOT_INERT}=1 turns off the whole auto-boot, not just the "
            f"disk tier); to stop THIS boot right now: POST {_route()} "
            "{\"op\":\"authority\",\"enabled\":false} and then POST "
            f"{_route()} {{\"op\":\"disarm\"}} - the arm is a LATCH, so that holds "
            "until someone arms again, and a manual disarm vetoes this module's own "
            "boot decision for the rest of the process too")


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
    # THE OPERATOR'S VETO, FIRST, because it is the most recent human decision in the
    # process and it post-dates the environment. It never used to be a gate: while the
    # arm had a TTL, the veto only had to stop `renew_if_due()`, and a fresh boot in the
    # same process happily re-armed. With a latch there is no timer to bring the tier
    # back and no timer to take it down either, so the veto is the ONLY thing standing
    # between "an operator disarmed this" and "a second connector boot in this process
    # silently re-armed it".
    if _state["disarmed_by_operator"]:
        return OUTCOME_REFUSED, (
            f"an operator has disarmed this process "
            f"({_state['latch_reason'] or 'no reason recorded'}): a manual disarm "
            "beats the boot default, and with the arm a LATCH a re-arm here would be "
            "the last word forever rather than a lapse away. POST "
            f"{_route()} {{\"op\":\"arm\"}} to lift the veto")
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

    THE FAIL-SAFE, STATED AS AN INVARIANT (0.2.35, and the reason `_disarmed_exit()`
    exists): AN OUTCOME THAT IS NOT `ARMED` ENDS THIS PROCESS WITH THE TIER DISARMED
    AND NO ARM FILE ON DISK. That is not only tidiness. The boot fence in
    `disk_spill._arm_boot_fence` compares kernel BOOT IDs, so four engine boots inside
    one long uptime are all "this boot" to it - and while the arm carried a TTL, the TTL
    is what made a file left by engine boot #1 harmless to engine boot #2. The TTL is
    gone, so the file goes too: every non-ARMED path runs `disk_spill.disarm()`, which
    deletes it (`_remove_arm_file`). The invariant is `THE ONLY THING THAT ARMS A
    PROCESS IS THAT PROCESS'S OWN BOOT DECISION`, and it is what
    `test_the_boot_disarms_an_arm_file_left_by_an_earlier_engine_boot_on_the_same_kernel`
    pins. `_disarmed_exit()` is the ONLY place the fail-safe is written, so a new early
    `return` that forgets it fails that test rather than shipping a hole.

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
            return _disarmed_exit(outcome, reason)
        # THE TWO HALVES, in the order that cannot leave a posture with no arm: the
        # spill path first (it is the thing that can fail - no root, no channel to the
        # ranks), then the serve posture. `disk_spill.arm()` is the same call the dev
        # verb makes, and it takes no lifetime argument any more: inventing a third arm
        # path is how a boot ends up armed in a way no disarm can reach.
        verdict = disk_spill.arm()
        _state["arm_file_written"] = bool(verdict.get("arm_file_written"))
        if not verdict.get("armed"):
            return _disarmed_exit(
                OUTCOME_REFUSED,
                f"arming the spill path was refused: "
                f"{verdict.get('reason') or 'no reason given'}")
        if not _state["arm_file_written"]:
            # THE FLEET'S DOOR. A worker's arm is the FILE - the EngineCore has no KV
            # memory and no channel to a worker, so an arm that could not be published
            # arms one process that can do nothing and leaves the four ranks that could
            # do it disarmed. That is a partially-armed fleet by definition, so it is
            # refused here rather than reported as a success.
            return _disarmed_exit(
                OUTCOME_REFUSED,
                f"the arm could not be published to the worker ranks "
                f"({verdict.get('arm_file_reason') or 'no arm file was written'}): a "
                "spill arm reaches a worker only through the diagnostics directory "
                f"({config.ENV_DIAG_DIR}), and an EngineCore-only arm would leave the "
                "fleet half-armed, which the scheduler's AND-over-ranks proof then "
                "closes anyway - while looking armed",
                arm_file_written=False)
        from . import disk_restore  # noqa: PLC0415 - it imports this module lazily
        raised = disk_restore.set_disk_authority(True)
        if not raised.get("ok"):
            # Roll the arm back: the posture is the half an operator asked for, and an
            # armed tier that serves nothing is the state he is trying NOT to boot into.
            return _disarmed_exit(
                OUTCOME_REFUSED,
                f"the serve posture could not be raised: "
                f"{raised.get('reason') or 'no reason given'}; the spill arm was "
                "rolled back so this boot is uniformly disarmed")
        _state["outcome"] = OUTCOME_ARMED
        _state["honoured"] = True
        _state["arm_file_written"] = True
        _state["serving_as_after"] = _serving_as()
        _state["reason"] = (
            "armed the spill path as a LATCH and raised this boot's serve posture "
            f"(serving_as={_state['serving_as_after']}), with no POST. The arm has no "
            "expiry and nothing renews it because nothing expires: it comes down on "
            f"a POST {{\"op\":\"disarm\"}}, and only on that")
        _log_boot_line()
        return view()
    except Exception as exc:  # noqa: BLE001 - a boot default must never cost a boot
        return _disarmed_exit(
            OUTCOME_REFUSED,
            f"the boot default raised {type(exc).__name__} and was abandoned; the "
            "boot continues disarmed, and any arm file this boot could have written "
            "is deleted rather than left for the next boot to find")


def _disarmed_exit(outcome: str, reason: str | None,
                   arm_file_written: bool | None = None) -> dict:
    """AN OUTCOME THAT IS NOT `ARMED`, CLOSED OUT: disarm here, delete the file there.

    EVERY non-ARMED exit of `boot_enable()` comes through this one function, which is
    the whole point of it - the fail-safe is a single call site, so it cannot be
    forgotten by the next early return, and
    `test_the_boot_disarms_an_arm_file_left_by_an_earlier_engine_boot_on_the_same_
    kernel` plus `test_every_outcome_that_is_not_ARMED_leaves_the_tier_disarmed_and_
    the_file_gone` fail if it is ever bypassed.

    The disarm is not a formality on the paths where nothing was armed. It is how a
    boot that REFUSED - or a boot whose engine restarted inside one kernel uptime, which
    the boot-id fence cannot see - erases an arm file written by an EARLIER engine boot
    and thereby keeps its own decision from being made for it.
    """
    _state["outcome"] = outcome
    _state["honoured"] = False
    _state["reason"] = reason
    # `None` means "this boot never got as far as trying to publish an arm"; the two
    # paths that DID try hand in what they learned, so the payload keeps the ability to
    # tell "never attempted" from "attempted and could not reach the ranks" - the second
    # one is a misconfigured `VLLM_E2_TIER_DIAG`, the first is a normal boot.
    _state["arm_file_written"] = arm_file_written
    try:
        verdict = disk_spill.disarm()
        if not verdict.get("arm_file_removed"):
            logger.debug("kvtier-dev-plugin: the disk boot fail-safe found no arm "
                         "file to remove (%s)", verdict.get("arm_file_removed"))
    except Exception as exc:  # noqa: BLE001 - a fail-safe must not cost a boot
        logger.warning("kvtier-dev-plugin: the disk boot fail-safe could not disarm "
                       "the tier (%s): the outcome is %s and an arm file may have "
                       "survived this boot", type(exc).__name__, outcome)
    _state["serving_as_after"] = _serving_as()
    _log_boot_line()
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
        body = (f"HONOURED: armed the spill path and set this boot's serve posture, "
                f"so it comes up serving_as={_state['serving_as_after']} with NO POST. "
                "The arm is a LATCH (0.2.35): it has no expiry, there is nothing to "
                "renew, and it comes down on a POST that disarms it and not on a "
                "clock. It gates SERVING from disk as well as writing to it, so the "
                "same POST stops both. ")
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


# --- (0.2.35) THE RENEWAL PATH THAT USED TO BE HERE -----------------------------
#
# This is where `renew_if_due()` lived: a tick that rode `disk_restore.offer()`, read
# `arm_state()["seconds_left"]`, and re-armed the spill path when less than
# `RENEW_AHEAD_SECONDS` (3,600 s) of a `disk_spill.ARM_MAX_SECONDS` (7,200 s) arm was
# left, with counters (`renewals`, `renewal_failures`, `last_renewal`), a log-once
# failure line, and a `renewal_would_run` field that told an operator whether it would
# fire. All of it is DELETED, and the reason is the ruling in `disk_spill`'s module
# docstring: the arm stopped being a timer, so a mechanism whose only job was to beat a
# timer has no job. `disk_restore.offer()` no longer calls it, its four state keys and
# its three view fields are gone from `_KEYS`, and nothing in the package re-arms a
# tier behind an operator's back - which the veto in `_gate_verdict()` now enforces at
# the boot decision too, the place renewal could never reach.
#


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
    """LATCH the operator's veto over THIS PROCESS's boot decision.

    What the veto protects changed in 0.2.35, and the name did not need to: it used to
    stop `renew_if_due()` from re-arming a box an operator had disarmed, and with the
    renewal deleted it could have been deleted with it. It is kept, and it now guards
    the one path that can still raise the arm without a human - `_gate_verdict()`
    refuses to arm while it stands. That matters MORE, not less, now that the arm is a
    latch: in 0.2.34 a mistaken re-arm would have expired on its own inside two hours,
    and in 0.2.35 nothing will.

    Called from the dev-verb handlers in `disk_store.py` - the human's door - and from
    nowhere else. That placement is the point: this module's own fail-safe calls
    `disk_spill.disarm()` directly and does NOT latch, because a rollback is not an
    operator's instruction, while `POST {"op":"disarm"}` is one and outranks a boot
    default for the rest of this process. Nothing here is written to disk: the next
    process reads the environment's word, which is the boot-default doctrine.
    """
    _state["disarmed_by_operator"] = True
    _state["latch_reason"] = str(reason)[:200]
    logger.warning("kvtier-dev-plugin: DISK[boot] the operator's posture now beats the "
                   "boot default for this boot (%s): nothing in this process will arm "
                   "it again, including a second connector boot. %s",
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
