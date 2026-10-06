r"""The global disk-authority switch: what the brief asks for, in one file.

The feature is one boolean with four properties, and each one gets its own test rather
than being inferred from the offer path's behaviour:

1. **Its import-time value is OFF.** A process that has run no boot has no disk
   authority, and the value is a module constant these tests refuse to let drift. Since
   0.2.33 a BOOT may raise it - by calling this module's own setter, at connector
   creation, from the plugin's own boot default (`disk_boot.py`) - which is a CALLER of
   the writer, not a second writer, and is pinned as such below.
2. **The writers are the two public setters, and every caller in the package is named.**
   Not a config file, not the spill arm, not the gap-fill switch, not a request field,
   not a failure or a recovery path. Proved by scanning every assignment to the field in
   the module's own syntax tree and naming each caller in the rest of the package.
3. **When it is on, it applies to every request.** Not "to requests that ask": two
   different requests with different fields get the same posture, and neither can opt
   out.
4. **State is unmistakable.** Boot line, witness line, `counters()`, `counters_view`
   units, `disk_posture()`, the snapshot payload and the log line all say the same
   thing, so an operator never has to assemble a posture from two places.

Everything here runs against the module's real state machine; nothing monkeypatches the
switch, because a test that patches the thing under test proves nothing about it.
"""

from __future__ import annotations

import ast
import inspect
import logging

import test_disk_restore as base

from kvtier_dev_plugin import (
    backport,
    config,
    disk_restore,
    disk_spill,
    disk_store,
    endpoint,
)

armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
armed_gap_fill = base.armed_gap_fill  # noqa: F811 - only the gap-fill switch up
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

BLOCK = base.BLOCK
Manager = base.Manager
Req = base.Req
fill = base.fill
offer = base.offer


# --- 1. the boot default ------------------------------------------------------

def test_the_boot_default_is_off_and_says_so():
    """A process that has run NO BOOT gives disk authority to nobody, and every surface
    agrees.

    Still the property the design rests on, and asserted with NOTHING set up: no
    fixture, no arm, no store. What 0.2.33 changed is that a BOOT may now raise the
    switch - by calling this module's own public setter at connector creation, because
    `KVTIER_DISK_BOOT_ENABLE` defaults to ON. The IMPORT-TIME default below is what a
    process is before any boot has decided anything, and it stays False: an import that
    turned serving behaviour on would make the plugin's import a side effect, and
    `disk_boot`'s fail-safe direction ("every outcome except ARMED leaves the boot
    disarmed") is only expressible against a default-off start. The boot-time half of
    the story is `tests/test_disk_boot_default.py`.
    """
    assert disk_restore.DISK_AUTHORITY_DEFAULT is False
    posture = disk_restore.disk_posture()
    assert posture["disk_authority"] is False
    assert posture["serving_as"] == "off"
    counters = disk_restore.counters()
    assert counters["disk_authority"] is False
    assert counters["serving_as"] == "off"
    assert counters["disk_authority_changes"] == 0
    assert counters["authority_lookups"] == 0
    line = disk_restore.boot_line()
    assert "disk_authority=OFF" in line
    assert "serving_as=off" in line


def test_the_state_field_is_a_boolean_and_not_a_string_of_one():
    """`False` and `"false"` are different things to a gate; only one is a default."""
    assert isinstance(disk_restore.counters()["disk_authority"], bool)
    assert isinstance(disk_restore.disk_posture()["disk_authority"], bool)
    disk_restore.set_disk_authority(True)
    assert isinstance(disk_restore.counters()["disk_authority"], bool)


# --- 2. only the route op turns it on ----------------------------------------

def test_the_only_writer_of_the_switch_is_the_setters_shared_body():
    """EVERY write to `_state[...]` that could name the switch, from the module's AST.

    The claim "nothing else turns it on" is a claim about the source, so it is checked
    against the source. The module writes `_state` two ways: by literal key, and through
    the one dynamic key in `_switch_verdict`. So the test collects every subscript
    assignment into `_state`, splits it into the ones that can only ever name one key
    and
    the ones whose key is a variable, and asserts:

      * the literal writes that name `disk_authority` are the boot-default state builder
        and the test reset - the two places that must set it to False;
      * the ONLY dynamic write in the module is in `_switch_verdict`, and
        `_switch_verdict`
        is called only by the two public switch setters.

    A new code path that raised the switch to "recover" from a failure - the shape this
    whole concern is about - shows up here with its function name.
    """
    tree = ast.parse(inspect.getsource(disk_restore))

    def owner_of(node):
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and node in ast.walk(fn):
                return fn.name
        return "<module>"

    literal_writers = set()
    variable_key_writers = set()
    other_dynamic_writers = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "_state"):
                continue
            if isinstance(target.slice, ast.Constant):
                if target.slice.value == "disk_authority":
                    literal_writers.add(owner_of(node))
            elif isinstance(target.slice, ast.Name):
                # A key that is a VARIABLE: this is the only shape that can write the
                # switch from a function that never names it, so it gets its own bucket.
                variable_key_writers.add((owner_of(node), target.slice.id))
            else:
                # A computed key (an f-string over a loop variable, a reset loop):
                # inspectable as not-the-switch, because the shapes here build key names
                # that no switch field has.
                other_dynamic_writers.add(owner_of(node))
    # Nobody writes the key by name: the fresh-boot value comes from the `_state` dict
    # literal and from `reset_state_for_tests`, both of which build the whole dict.
    assert literal_writers == set(), sorted(literal_writers)
    # Two variable-key writers, and both are accounted for: `_switch_verdict` (the
    # setter, keyed by its `field` parameter) and `reset_state_for_tests`, which copies
    # a literal dict whose `disk_authority` entry is the module default - the fresh-boot
    # builder, checked below. Neither can raise the switch on a live boot: the first is
    # the only public setter, the second runs only to build a fresh state.
    assert variable_key_writers == {("_switch_verdict", "field"),
                                    ("reset_state_for_tests", "key")}, \
        sorted(variable_key_writers)
    assert other_dynamic_writers <= {"reset_state_for_tests", "_note_apply_window"}, \
        sorted(other_dynamic_writers)
    # The boot defaults, asserted where they are written: both state builders carry the
    # key with the module constant, so a fresh boot cannot come up on.
    source = inspect.getsource(disk_restore)
    assert source.count('"disk_authority": DISK_AUTHORITY_DEFAULT') == 2, \
        "exactly two state builders, both defaulting the switch off"
    assert disk_restore.DISK_AUTHORITY_DEFAULT is False
    # And the reset builder is the SAME literal the module builds at import, so the two
    # cannot drift to different defaults.
    reset_source = inspect.getsource(disk_restore.reset_state_for_tests)
    assert '"disk_authority": DISK_AUTHORITY_DEFAULT' in reset_source
    # And the variable-key writer can only ever be pointed at the two switch fields:
    # every call site passes one of those two literals.
    passed = {call.args[0].value for fn in ast.walk(tree)
              if isinstance(fn, ast.FunctionDef)
              for call in ast.walk(fn)
              if isinstance(call, ast.Call)
              and getattr(call.func, "id", None) == "_switch_verdict"
              and call.args and isinstance(call.args[0], ast.Constant)}
    assert passed == {"disk_authority", "offers_enabled"}, passed
    callers = sorted({fn.name for fn in ast.walk(tree)
                      if isinstance(fn, ast.FunctionDef)
                      for call in ast.walk(fn)
                      if isinstance(call, ast.Call)
                      and getattr(call.func, "id", None) == "_switch_verdict"})
    assert callers == ["set_disk_authority", "set_offer_switch"], callers
    # And the only caller of the setter, inside this module, is the public one.
    callers = {fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
               for call in ast.walk(fn)
               if isinstance(call, ast.Call)
               and getattr(call.func, "id", None) == "_switch_verdict"}
    assert callers == {"set_disk_authority", "set_offer_switch"}, sorted(callers)


def test_nothing_outside_the_module_turns_it_on():
    """The same claim about the REST of the package: the setter has TWO callers, and
    each one is named.

    `disk_store._op_authority` is the operator's door. Since 0.2.33 `disk_boot
    .boot_enable` is the other: this package's own BOOT DEFAULT, which calls the same
    setter because the operator asked for a boot that comes up serving by default, and
    whose flag (`KVTIER_DISK_BOOT_ENABLE`) can only ever turn that back off. Both are
    accounted for here, and the assertion is what keeps the pair small: a boot-recovery
    path, a config loader or a test helper that grows a call means the posture stops
    being a decision anybody made, and this fails naming the file.

    The two callers are not equally powerful, and the extras keep it that way: the boot
    default reaches the switch through the public setter and nowhere else, never touches
    the cheaper switch, and the module that owns the switch still reads no environment -
    the flag is resolved in `config`, turned into a real boolean, and handed over.
    """
    import pathlib
    import re
    root = pathlib.Path(inspect.getsourcefile(disk_restore)).parent
    found = []
    for path in sorted(root.glob("*.py")):
        if path.name == "disk_restore.py":
            continue
        text = path.read_text()
        if re.search(r"\bset_disk_authority\b", text):
            found.append(path.name)
    assert found == ["disk_boot.py", "disk_store.py"], found
    # ...and there it sits in exactly the one handler that answers the op.
    source = (root / "disk_store.py").read_text()
    assert source.count("set_disk_authority") == 1, \
        "one call site: the op handler; two means a second door"
    assert "_op_authority" in source
    # THE BOOT DEFAULT'S CALL: exactly one CALL SITE (not one mention - the module's
    # docstring names the setter on purpose), and the function that makes it is the
    # boot's own decision, not a helper a later path can reach. The cheaper switch is
    # is untouched, because the boot default makes ONE decision ("come up serving") and
    # not one per posture - a boot that raised both switches would quietly retire the
    # distinction the 2026-10-05 incident made worth having.
    boot_source = (root / "disk_boot.py").read_text()
    assert "set_offer_switch" not in boot_source, \
        "the boot default raises one posture, not both"
    tree = ast.parse(boot_source)
    sites = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) \
                    and getattr(call.func, "attr", None) == "set_disk_authority":
                sites.append(fn.name)
    assert sites == ["boot_enable"], sites
    assert "disk_restore.set_disk_authority(True)" in boot_source, \
        "the one call raises the switch - reusing the setter, not a third way in"
    # And the module that owns the switch stays free of the environment: the flag is
    # resolved in `config`, and `disk_restore` is handed a real boolean or nothing.
    assert "os.environ" not in inspect.getsource(disk_restore), \
        "the switch's own module must stay free of the environment"


def test_no_environment_variable_or_config_key_names_the_switch():
    """No configuration vocabulary names the switch - including the new boot flag.

    `config.py` owns every environment name in this plugin, so the check is: no config
    constant mentions the switch, `config`'s source never mentions it either, and
    `config.apply`/reload cannot produce one. A key that is not in the accepted config
    vocabulary is refused by the reload route's own rule, so there is no silent path
    either.

    SINCE 0.2.33 THERE IS ONE MORE ENV VAR IN THIS AREA (`KVTIER_DISK_BOOT_ENABLE`), and
    this is where the distinction has to be pinned rather than argued: that variable
    names a BOOT POSTURE ("come up armed and serving, or don't"), not the switch. It
    reaches the switch the way a human does - by calling the public setter from
    `disk_boot.boot_enable` - while `config.py` still never names the field, still holds
    no key that could raise it, and still has no path that writes engine state.
    """
    names = [name for name in vars(config) if name.startswith(("ENV_", "DEFAULT_"))]
    assert [name for name in names if "AUTHORITY" in name.upper()] == []
    source = inspect.getsource(config)
    assert "authority" not in source.lower()
    # The boot flag exists, is named for the BOOT and not for the switch, and its
    # resolver reads the environment and nothing else: no file, no persisted posture, no
    # write of any state. That last clause is the 2026-10-03 prohibition, enforced
    # where the flag is read.
    assert config.ENV_DISK_BOOT_ENABLE == "KVTIER_DISK_BOOT_ENABLE"
    assert "AUTHORITY" not in config.ENV_DISK_BOOT_ENABLE
    resolver = inspect.getsource(config.disk_boot_enable)
    assert "os.environ" in resolver
    for forbidden in ("open(", "disk-spill-arm", "ARM_FILE", "_read_arm_file",
                      "set_disk_authority", "_state", "arm_state"):
        assert forbidden not in resolver, (
            f"the boot flag's resolver must read the environment and nothing else, "
            f"but it mentions {forbidden!r}")


def test_the_spill_arm_raises_nothing(armed):
    """The confusion this design exists to end, tested on the arm that caused it.

    `armed` puts up the spill tier, the RAM tier, a store root, a journal and a worker
    proof - the whole production-shaped state. Drop the switch and every one of those
    arms stays up while nothing serves: on 2026-10-05 an arm raised for a WRITE
    experiment was read as consent to SERVE, 123,474 times.
    """
    disk_restore.set_disk_authority(False)
    assert disk_spill.is_armed() is True and backport.is_armed() is True, \
        "the arms are up: this is not the inert-boot case"
    assert disk_restore.disk_posture()["disk_authority"] is False
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    fill(armed, seeds=[1, 2, 3, 4])
    assert offer(Manager(), Req(request_id="armed-only", blocks=4), 0,
                 (0, False)) == (0, False), \
        "an armed box with the switch down serves nothing from disk"
    assert disk_restore.counters()["offers"] == 0


def test_the_gap_fill_switch_does_not_grant_authority(armed_gap_fill):
    """Two switches, and raising the cheap one must not read as the expensive one."""
    assert disk_restore.disk_posture()["disk_authority"] is False
    assert disk_restore.disk_posture()["serving_as"] == "ram_first_gap_fill"
    assert disk_restore.counters()["authority_lookups"] == 0
    fill(armed_gap_fill, seeds=[1, 2, 3, 4])
    assert offer(Manager(), Req(request_id="gap", blocks=4), 0,
                 (0, False)) == (4 * BLOCK, True), \
        "the cheap posture does serve, which is what makes the distinction matter"
    counters = disk_restore.counters()
    assert counters["offers"] == 1
    assert counters["authority_lookups"] == 0, \
        "gap-fill lookups are not authority lookups"


# --- 3. one POST, and it applies to every request ----------------------------

def test_one_post_serves_every_request_without_another_call(armed):
    """The operator's whole procedure is one command: `armed` raised nothing else.

    The fixture calls `set_disk_authority(True)` and nothing else, and from that single
    change the gate opens, a first request is offered, and a SECOND unrelated request is
    offered too. There is no per-request step in between, because there is no
    per-request decision left to make.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    assert disk_restore.gate()["ok"] is True
    assert disk_restore.disk_posture()["enabled"] is False, \
        "the gap-fill switch was never touched - one POST, not two"
    first = offer(Manager(), Req(request_id="first", blocks=4), 0, (0, False))
    assert first == (4 * BLOCK, True)
    second = offer(Manager(), Req(request_id="second", blocks=4), 0, (0, False))
    assert second == (4 * BLOCK, True)
    counters = disk_restore.counters()
    assert counters["offers"] == 2
    assert counters["authority_lookups"] == 2, \
        "both lookups ran under the switch, with no per-request input"


def test_a_request_carrying_a_marker_gets_the_same_posture(armed):
    """What a request sends changes nothing: same posture, same counters, same offer.

    The pair is the point. Under the retired design, this field flipped a request into
    or out of disk authority; now two requests that differ ONLY in a client field take
    the same path, because there is no code that reads it.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    plain = Req(request_id="plain", blocks=4)
    marked = Req(request_id="marked", blocks=4)
    marked.kv_transfer_params = {"kvtier_disk_restore": True}
    unmarked = Req(request_id="unmarked", blocks=4)
    unmarked.kv_transfer_params = {"kvtier_disk_restore": False}
    results = [offer(Manager(), request, 0, (0, False))
               for request in (plain, marked, unmarked)]
    assert results == [(4 * BLOCK, True)] * 3
    counters = disk_restore.counters()
    assert counters["offers"] == 3
    assert counters["authority_lookups"] == 3


def test_turning_it_off_stops_serving_without_a_restart(armed):
    """The off switch is as good as the on switch: one POST, effective immediately."""
    fill(armed, seeds=[1, 2, 3, 4])
    assert offer(Manager(), Req(request_id="on", blocks=4), 0,
                 (0, False)) == (4 * BLOCK, True)
    disk_restore.set_disk_authority(False)
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    assert offer(Manager(), Req(request_id="off", blocks=4), 0,
                 (0, False)) == (0, False)
    assert disk_restore.counters()["offers"] == 1


# --- 4. state is unmistakable ------------------------------------------------

def test_every_surface_agrees_about_the_posture(armed):
    """Six surfaces, one truth. An operator must never assemble a posture by hand.

    boot line, witness line, `counters()`, `counters_view`'s units, `disk_posture()`,
    the scheduler payload, and the `/disk` verb's own response - all six, in both
    states of the switch, compared to each other rather than to expected strings, so a
    change in one place cannot quietly disagree with another.
    """
    for wanted in (True, False):
        disk_restore.set_disk_authority(wanted)
        verdict = disk_restore.disk_posture()
        counters = disk_restore.counters()
        section = disk_restore.scheduler_section()
        verb = disk_store.disk_for_engine(None, {"op": "authority"})
        boot = disk_restore.boot_line()
        witness = disk_restore.witness_line()

        assert verdict["disk_authority"] is wanted
        assert counters["disk_authority"] is wanted
        assert section["disk_authority"] is wanted
        assert verb["posture"]["disk_authority"] is wanted
        assert counters["serving_as"] == verdict["serving_as"] == section["serving_as"]
        assert f"disk_authority={'ON' if wanted else 'OFF'}" in boot, boot
        assert f"disk_authority={'on' if wanted else 'off'}" in witness, witness
        assert f"serving_as={counters['serving_as']}" in boot, boot
        assert f"serving_as={counters['serving_as']}" in witness, witness
    # Every one of those fields carries a unit, because a bare value in a payload is a
    # value whose meaning is a guess.
    units = disk_restore.counters()["units"]
    for key in ("disk_authority", "serving_as", "disk_authority_changes",
                "offer_switch_changes", "authority_lookups"):
        assert str(units.get(key, "")).strip(), key


def test_the_log_line_says_which_way_it_moved_and_what_is_left(armed, caplog):
    """The operator's permanent record of "when did this boot start serving".

    A line that reads the same going up as coming down is how a boot's posture stops
    being recoverable from its own log, so the ON sentence names the consequence (every
    request, host-RAM hit given back) and the OFF sentence names what is still serving -
    because turning authority off while the gap-fill switch is up does NOT stop
    serving, and an operator reading "OFF" must not be told a falsehood.
    """
    with caplog.at_level(logging.WARNING):
        disk_restore.set_disk_authority(True)
        disk_restore.set_offer_switch(True)
        disk_restore.set_disk_authority(False)
    text = caplog.text
    assert "disk_authority=ON" in text
    assert "EVERY request is served from disk" in text
    assert "disk_authority=OFF" in text
    assert "ram_first_gap_fill" in text, \
        "the OFF line must name the posture that is still serving"


def test_the_change_counter_counts_moves_not_lookups(armed):
    """`disk_authority_changes` is the audit trail of POSTURE changes, not of traffic.

    Two requests and a no-op re-set must not move it; one real change must. It answers
    "how many times did someone change the posture this boot", which is the question a
    confused experiment asks first.
    """
    fill(armed, seeds=[1, 2, 3, 4])
    before = disk_restore.counters()["disk_authority_changes"]
    offer(Manager(), Req(request_id="a", blocks=4), 0, (0, False))
    disk_restore.set_disk_authority(True)          # already true: no move
    assert disk_restore.counters()["disk_authority_changes"] == before
    disk_restore.set_disk_authority(False)
    assert disk_restore.counters()["disk_authority_changes"] == before + 1


def test_the_verdict_payload_reports_its_own_lifetime(armed):
    """"Is this sticky?" is answered by the response that moved it.

    `per_boot` and `survives_restart` are in the read response too, not only in the
    write response, because the operator asking the question is usually the one who did
    not make the change.
    """
    out = disk_store.disk_for_engine(None, {"op": "authority"})
    assert out["per_boot"] is True
    assert out["survives_restart"] is False
    out = disk_store.disk_for_engine(None, {"op": "authority", "enabled": True})
    assert out["per_boot"] is True
    assert out["survives_restart"] is False
    assert out["disk_authority"]["field"] == "disk_authority"
    assert out["disk_authority"]["enabled"] is True


def test_the_op_is_post_only_and_the_switch_is_unreachable_by_a_page_load():
    """`MUTATING_DISK_OPS` is what turns a mutating op into a 405 on GET.

    A switch this expensive must not be flippable by an <img src>, a proxy retry, or a
    browser prefetch - which is the whole reason the mutating list exists.
    """
    assert "authority" in disk_store.MUTATING_DISK_OPS
    assert "authority" in disk_store.DISK_OPS
    assert "authority" in disk_store._DISK_OPS
    spec = endpoint.ARG_SPECS[config.ROUTE_DISK]
    assert "authority" in spec["op"]["values"]
    assert spec["enabled"]["kind"] == "bool"
    assert "scope" not in spec, "the retired scope argument must not still be accepted"


def test_a_refused_switch_change_says_so_instead_of_answering_ok(armed):
    """`ok: true` for a command that did nothing is the bug this whole op avoids.

    Every hostile value gets `ok: false`, a reason naming the field, and a switch that
    did not move - so a curl typo can neither turn serving on nor turn it off.
    """
    disk_restore.set_disk_authority(True)
    for junk in ("true", 1, "yes", None, [], {}, 1.0):
        out = disk_store.disk_for_engine(None, {"op": "authority", "enabled": junk})
        assert out["ok"] is False, (junk, out)
        assert "disk_authority" in out["reason"], (junk, out)
        assert disk_restore.disk_posture()["disk_authority"] is True, junk
    # A read, by contrast, is always ok and always reports without changing.
    out = disk_store.disk_for_engine(None, {"op": "authority"})
    assert out["ok"] is True
    assert "disk_authority" not in out, "a read carries no write verdict"


def test_the_retired_vocabulary_is_unreferenced_across_the_package():
    """The last line of the removal: no module, route or payload still NAMES the marker.

    Checked as identifiers and non-docstring string constants across the package's
    source, so a re-import, a copy-pasted helper or a payload key fails here. Prose that
    explains the retirement is allowed and wanted - it is the WHY, and it is the only
    place the reason survives - so docstrings are excluded from the scan.
    """
    import pathlib
    root = pathlib.Path(inspect.getsourcefile(disk_restore)).parent
    retired = {"RESTORE_MARKER_KEY", "RESTORE_SCOPE_OPTIN", "RESTORE_SCOPE_ALL",
               "RESTORE_SCOPES", "RESTORE_SCOPE_DEFAULT", "request_flagged",
               "set_scope", "skipped_no_marker", "requests_seen",
               "kv_transfer_params", "kvtier_disk_restore", "offer_posture"}
    offenders = {}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        documented = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body \
                    and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant):
                documented.add(id(body[0].value))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute)}
        names |= {node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Constant)
                  and id(node) not in documented
                  and isinstance(node.value, str)}
        hits = sorted(names & retired)
        if hits:
            offenders[path.name] = hits
    assert offenders == {}, offenders
