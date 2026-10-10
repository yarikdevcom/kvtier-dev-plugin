r"""THE LOOKUP LEDGER ARMS AT BOOT - so a refusal reason is one query, not two hunts.

THE INCIDENT THAT ASKED FOR THIS. The per-request lookup ledger - the ring that joins
each `get_num_new_matched_tokens` offer to the `update_state_after_alloc` adopt that
followed it - was DISARMED on the boot where it was needed. The question "why did the
16:34 request get 0 tokens off the arena after a 15-hour suspend" was unrecoverable:
the ring records nothing while disarmed, and `calls_before_arming` is a count with no
records behind it. Two investigations answered in one query what the ledger would have
answered for free, forever, had it been armed.

Observability, so the fail-open rule applies with full force:

  * a boot that cannot arm the ledger BOOTS. It counts the refusal and logs ONCE. The
    arm never joins `boot["refusals"]`: a missing diagnostic is not a tier that
    declined to serve (the `journal_drain` discipline, not the `disk_boot` one).
  * the verdict is on the boot LINE as a bracketed token - `LOOKUP_LEDGER[armed]`,
    `[env-refused]`, `[refused]`, `[inert]`, `[not_run]` - because a bare absence
    cannot be told from a value the reader did not grep. Bracketed and NOT
    `key=value`, for the reason `mamba_anchor.boot_token` gives: the restart verifier
    in llm-api pulls the PASS criterion with `grep -oE "tier=[A-Za-z_]+" | tail -1`,
    and a token with no `=` can never become a candidate match.
  * it arms the RING and nothing else. Structurally pinned below: the boot step's own
    source may not name a serve posture, an offer switch, a tier arm or the spill arm.
    Arming a recorder must not be able to change what the engine serves, and this file
    is where that is proved rather than asserted.

The gate is `VLLM_E2_TIER_DEV_LOOKUP_LEDGER_BOOT_ARM`, default ON, the same grammar as
`config.mamba_anchor_patch()`: unset is on, a disable word is `env-refused`, and a
value in neither vocabulary stays ON and says so.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import test_auto_boot as AB  # noqa: F401 - the boot fixtures and the boot() helper
import test_backport as TB  # noqa: F401 - the fake fork class + production manager

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    backport,
    config,
    lookup_ledger,
)

# The boot-state fixture, re-exported by name so it is autouse HERE too: the boot
# arms the tier and the guard, and a test that booted without restoring that hands a
# live arm to whatever file runs next.
boot_state = AB.boot_state  # noqa: F811 - autouse: the boot's backport+guard state is saved and restored


def decisions(boot: dict) -> dict:
    """EVERY DECISION a boot made, with the per-process counters stripped out.

    WHY NOT `boot == baseline` field for field: two records legitimately carry a
    counter that differs between boots of one process (`disk_boot.boots`,
    `journal_drain.attempted`), and a test that compares them would fail for a reason
    that has nothing to do with the claim. The claim here is that a missing diagnostic
    changed no DECISION - what the boot armed, what it refused, what it fitted - and
    that is what this function returns.
    """
    restore = boot.get("restore") or {}
    arena = boot.get("arena") or {}
    disk_boot = boot.get("disk_boot") or {}
    drain = boot.get("journal_drain") or {}
    return {"backport": boot["backport"], "backport_source": boot["backport_source"],
            "guard": boot["guard"], "guard_armed": boot["guard_armed"],
            "tier": boot["tier"], "faults_at_boot": boot["faults_at_boot"],
            "restore_fitted": restore.get("fitted"), "restore_gate":
            restore.get("gate"),
            "arena_state": arena.get("state"),
            "disk_boot_outcome": disk_boot.get("outcome"),
            "disk_boot_armed": disk_boot.get("armed"),
            "journal_drain_stopped_at": drain.get("stopped_at"),
            "refusals": list(boot["refusals"]),
            "not_covered": list(boot["not_covered"])}


# --- 1. the default: a boot arms the ledger ----------------------------------

def test_the_boot_arms_the_lookup_ledger_by_default_and_the_token_renders():
    """A clean boot records the offer/adopt pair without anyone POSTing `arm` first.

    This is the property the incident removed: the ring must be recording from the
    FIRST request of the boot, because the request you need it for is always the one
    that already happened.
    """
    assert lookup_ledger.is_armed() is False, "a fresh process must not start armed"
    _report, boot, _cls, _manager = AB.boot()
    assert boot["guard"] is True, "the fixture boots a guarded engine"
    assert lookup_ledger.is_armed() is True
    record = boot["lookup_ledger"]
    assert record["armed"] is True
    assert record["verdict"] == lookup_ledger.VERDICT_ARMED
    assert "LOOKUP_LEDGER[armed]" in boot["line"]
    # Arming the recorder armed the RECORDER only.
    assert record["cap"] == lookup_ledger.DEFAULT_CAP


def test_the_booted_ledger_records_a_real_lookup():
    """Armed at boot means a lookup on the guarded path is a RECORD, not a counted miss.

    Driven through the same reader the verb uses, so "armed" is not a flag someone
    could set without the ring existing: `record_from_call` on a tier-shaped call has
    to land a live record.
    """
    AB.boot()
    manager = TB._production_manager(TB._six_group_class())
    lookup_ledger.record_from_call(
        lookup_ledger.METHOD_LOOKUP,
        (TB._Request(1, "req-boot-armed"), 3200), (800, True), manager)
    counters = lookup_ledger.counters()
    assert counters["lookups"] == 1
    assert counters["records"] == 1
    assert counters["live"] == 1


# --- 2. the gate --------------------------------------------------------------

@pytest.mark.parametrize("value", ["0", "false", "off", "no", " 0 "])
def test_the_env_var_disarms_the_boot_default_and_the_token_says_env_refused(
        monkeypatch, value):
    """`0` is the rollback, and it is distinguishable from a refusal in the payload.

    THE CONTROL IS THE REST OF THE BOOT, not just this field: the same engine booted
    with the flag off must be identical in every other verdict, or "an observability
    flag" would be a serving switch with extra steps.
    """
    baseline = AB.boot()[1]
    lookup_ledger.reset_state_for_tests()
    monkeypatch.setenv(config.ENV_LOOKUP_LEDGER_BOOT_ARM, value)
    _report, boot, _cls, _manager = AB.boot()
    assert lookup_ledger.is_armed() is False
    record = boot["lookup_ledger"]
    assert record["armed"] is False
    assert record["verdict"] == lookup_ledger.VERDICT_ENV_REFUSED
    assert "LOOKUP_LEDGER[env-refused]" in boot["line"]
    # The boot itself is UNCHANGED, field for field, apart from the ledger's own.
    moved = {key: (value, decisions(baseline)[key])
             for key, value in decisions(boot).items()
             if value != decisions(baseline)[key]}
    assert not moved, f"the ledger's gate moved a boot decision: {moved}"
    assert boot["refusals"] == baseline["refusals"]
    # A rollback is not a fault: nothing was counted as a refusal.
    assert lookup_ledger.counters()["boot_arm_refusals"] == 0


def test_an_unrecognised_value_arms_anyway_and_reports_the_complaint(monkeypatch):
    """`flase` does not disarm: ON is the shipped default, and it says the word was bad.

    `mamba_anchor_patch()`'s rule, for the same reason - a reader must see that the
    value he wrote is not a disable word, rather than being handed a payload that
    reads as his intent.
    """
    monkeypatch.setenv(config.ENV_LOOKUP_LEDGER_BOOT_ARM, "flase")
    _report, boot, _cls, _manager = AB.boot()
    assert lookup_ledger.is_armed() is True
    record = boot["lookup_ledger"]
    assert record["verdict"] == lookup_ledger.VERDICT_ARMED
    assert record["recognised"] is False
    assert record["raw"] == "flase"
    assert record["problems"]


def test_an_inert_boot_arms_nothing_and_says_inert(monkeypatch):
    """`VLLM_E2_TIER_DEV_BOOT_INERT` means "install and stop there", so no recorder.

    The ring costs a dict lookup per lookup when disarmed and an allocation at arm;
    the inert posture is the operator's explicit opt-out from every engine-changing
    effect of the plugin, so arming a ring here would be doing work the opt-out
    refused - and the token distinguishes it from a refusal, which is a different fact.
    """
    monkeypatch.setenv(config.ENV_BOOT_INERT, "1")
    _report, boot, _cls, _manager = AB.boot()
    assert lookup_ledger.is_armed() is False
    assert boot["lookup_ledger"]["verdict"] == lookup_ledger.VERDICT_INERT
    assert "LOOKUP_LEDGER[inert]" in boot["line"]
    assert lookup_ledger.counters()["boot_arm_refusals"] == 0


# --- 3. fail-open: a refusal costs nothing but a counter and one line ---------

def test_a_boot_that_cannot_arm_the_ledger_boots_normally_counts_it_and_logs_once(
        monkeypatch, caplog):
    """The arm raising is a counted, logged, SINGLE line - and the boot completes.

    THE CONTROL IS THE FIELD-BY-FIELD COMPARISON: everything the boot decided about
    the tier, the guard, the restore layer and the journal drain must be identical to
    the same boot with the ledger armed. A diagnostic that changes the boot it
    observes is worse than no diagnostic.
    """
    baseline = AB.boot()[1]
    lookup_ledger.reset_state_for_tests()

    def explode(*_args, **_kwargs):
        raise RuntimeError("the ring cannot be allocated")

    monkeypatch.setattr(lookup_ledger, "arm", explode)
    with caplog.at_level("WARNING"):
        _report, boot, _cls, _manager = AB.boot()
    assert boot["line"], "the boot still printed its line"
    assert decisions(boot) == decisions(baseline), (
        "a boot that could not arm the ledger decided nothing differently: that is the "
        "whole fail-open claim, and this is where it is checked")
    # An observability refusal is NOT a tier refusal: it never joins `refusals`.
    assert boot["refusals"] == baseline["refusals"]
    assert boot["lookup_ledger"]["armed"] is False
    assert boot["lookup_ledger"]["verdict"] == lookup_ledger.VERDICT_REFUSED
    assert "RuntimeError" in boot["lookup_ledger"]["reason"]
    assert "LOOKUP_LEDGER[refused]" in boot["line"]
    counters = lookup_ledger.counters()
    assert counters["boot_arm_refusals"] == 1
    # ONE line per episode, not one per lookup or per retry.
    lines = [record for record in caplog.records
             if "LOOKUP LEDGER" in record.getMessage()]
    assert len(lines) == 1, [record.getMessage() for record in lines]


def test_a_boot_without_the_guard_wrappers_refuses_and_records_the_reason():
    """The ring is fed by the guard wrappers, so no wrappers means nothing to record.

    Arming a ring that nothing writes into would be the worst shape of a lie: the
    payload would say `armed: true` and a reader would ask why the ring is empty. So
    the step asks about the wrappers it depends on and says NO with its reason - still
    counted, still one line, still no effect on the rest of the boot.
    """
    def unguarded(cls, *_args, **_kwargs):
        return {"installed": False, "armed": False, "nested": [],
                "reason": "the guard layer refused this manager"}

    original = backport._boot_guard
    backport._boot_guard = unguarded
    try:
        _report, boot, _cls, _manager = AB.boot()
    finally:
        backport._boot_guard = original
    assert boot["guard"] is False
    assert lookup_ledger.is_armed() is False
    record = boot["lookup_ledger"]
    assert record["verdict"] == lookup_ledger.VERDICT_REFUSED
    assert "wrapper" in record["reason"].lower()
    assert lookup_ledger.counters()["boot_arm_refusals"] == 1


# --- 4. the verdict travels everywhere the ledger does ------------------------

def test_the_boot_verdict_is_published_in_the_counters_and_the_snapshot_section():
    """A reader of the section can tell "armed at boot" from "armed by a POST".

    The distinction is the whole lesson of the incident: `armed: true` with a boot
    that never armed it means someone armed it mid-boot, and the records before that
    moment do not exist. `boot_armed` + `armed_at` say which.
    """
    AB.boot()
    counters = lookup_ledger.counters()
    assert counters["boot_arms"] == 1
    assert counters["boot_arm_refusals"] == 0
    assert counters["boot_verdict"] == lookup_ledger.VERDICT_ARMED
    section = lookup_ledger.section()
    assert section["boot_arms"] == 1
    assert section["boot_verdict"] == lookup_ledger.VERDICT_ARMED
    assert "boot_arm_refusals" in section
    # And the disarm path stays honest afterwards: arming at boot is not a latch.
    lookup_ledger.disarm()
    assert lookup_ledger.counters()["boot_verdict"] == lookup_ledger.VERDICT_ARMED
    assert lookup_ledger.counters()["armed"] is False


def test_the_token_renders_not_run_before_any_boot_ran():
    """A boot dict that never reached the step renders `[not_run]`, not `[disabled]`.

    Read from the BOOT DICT (the `JOURNAL_DRAIN` discipline), because a token taken
    from module state would describe the last boot in the process rather than the one
    whose line this is - and `backport.boot_token()` gets a fresh `boot_state` fixture
    in the same file, so module state is exactly what a stale token comes from.
    """
    assert backport.lookup_ledger_token({}) == "LOOKUP_LEDGER[not_run]"
    assert backport.lookup_ledger_token({"lookup_ledger": None}) == \
        "LOOKUP_LEDGER[not_run]"
    assert backport.lookup_ledger_token(
        {"lookup_ledger": {"verdict": "nonsense"}}) == "LOOKUP_LEDGER[refused]"


# --- 5. spill-side discipline: the boot step cannot move a posture ------------

FORBIDDEN = ("set_disk_authority", "set_offer_switch", "set_armed", "disk_authority",
             "offers_enabled", "serving_as", "serve_as", "disk_boot", "boot_enable",
             "disk_spill.arm", "spill_slot")


def _code_only(func) -> str:
    """Source with the FUNCTION'S docstring removed: prose may NAME the switches it
    deliberately does not touch, so a structural test must not read prose.

    THE ONE LINE THAT MAKES THIS A TEST AND NOT A FORMALITY: `ast.get_docstring` answers
    about the node you hand it, and the node parsed out of `inspect.getsource(func)` is
    a
    Module whose first statement is a FunctionDef - so asking the Module for its
    docstring
    returns None, nothing is stripped, and the assertion silently becomes "this
    function's
    English does not contain the word", which a comment could satisfy and a docstring
    would fail. Ask the FunctionDef instead.
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    target = tree.body[0]
    doc = ast.get_docstring(target, clean=False)
    if not doc:
        return source
    stmt = target.body[0]
    lines = source.splitlines()
    kept = [line for number, line in enumerate(lines, start=1)
            if not stmt.lineno <= number <= stmt.end_lineno]
    assert len(kept) < len(lines), "the docstring strip stripped nothing"
    return "\n".join(kept)


@pytest.mark.parametrize("func", [backport._boot_lookup_ledger,
                                  backport.lookup_ledger_token,
                                  lookup_ledger.boot_arm])
def test_the_boot_arm_step_names_no_posture_or_arm_writer(func):
    """STRUCTURE: arming the ledger is an observability step, proved on its source.

    The same class of pin `tests/test_mamba_anchor_patch.py` puts on the anchor patch:
    behavioural equality over one run cannot rule out a door that fires on another
    path, so the claim is made about the code. Docstrings and comments are stripped
    first, because this module's prose names the posture switches precisely in order
    to say the step never touches them.
    """
    code = _code_only(func)
    for word in FORBIDDEN:
        assert word not in code, f"{func.__name__} names {word!r}"


def test_the_boot_line_keeps_every_field_name_unique_and_the_token_beside_its_peers():
    """The new token does not disturb the line's existing contract.

    `tests/test_auto_boot.py` pins that every `=`-shaped field name appears exactly
    once, and the restart verifier in llm-api pulls the last `tier=`-shaped match -
    so a token that introduced an `=` would be read as a tier verdict by a grep that
    was never meant to see it.
    """
    import re
    _report, boot, _cls, _manager = AB.boot()
    line = boot["line"]
    # THE FIELD CONTRACT, COUNTED AS A FIELD AND NOT AS A SUBSTRING. A naive
    # `line.count("tier=") == 1` matches the `tier=` inside `ram_tier=` and
    # `disk_tier=` too - three matches on a line that is correct by design (the
    # two-tier naming above), so the assertion would fail a good boot. The rule the
    # restart verifier actually depends on: the LAST `tier=`-shaped token is the bare
    # one, and no field NAME appears twice.
    for field in ("backport=", "guard=", "ram_tier=", "disk_tier=", "tier=",
                  "faults=", "arena=", "disk_boot="):
        hits = re.findall(rf"(?:^| ){re.escape(field)}", line)
        assert len(hits) == 1, f"{field} appears {len(hits)} times: {line}"
    bare = re.findall(r"(?:^| )tier=[A-Za-z_]+", line)
    assert len(bare) == 1, bare
    assert line.rstrip().rsplit("tier=", 1)[0].endswith("ram_") is False
    assert "LOOKUP_LEDGER[" in line
    # THE TOKEN CARRIES NO `=`: the verifier pulls its PASS criterion with
    # `grep -oE "tier=[A-Za-z_]+" | tail -1`, so a token with an `=` in it becomes a
    # candidate match for a grep that was never meant to see it.
    token = line[line.index("LOOKUP_LEDGER["):].split("]", 1)[0] + "]"
    assert "=" not in token, token
    assert "LOOKUP_LEDGER=" not in line
