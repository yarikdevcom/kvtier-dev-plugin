r"""The boot-scoped switches, the offer posture, and the two incident shapes.

`test_disk_restore.py` asks what the restore path does once it decides to act. This
file asks the earlier question - WHEN is it allowed to speak at all - plus the two
failures that took a serving engine down on 2026-10-05:

* FUEL. The scheduler was armed from a spill arm file, four worker ranks were not, and
  the proof read looked only at `installed`, so 160,000 tokens were offered 123,474
  times to workers that would not bind a single slot.
* THE LOOP. Each failed load made the engine reschedule the request, which asked
  `get_num_new_matched_tokens` again, which got the same answer, forever: 520 offers a
  second until the harness watchdog reset the box.

Both are gate logic, so both are tested through the real door (`offer`) against real
rank files and a real journal - not against a monkeypatched proof.

THE NAME, AND WHAT IT NO LONGER MEANS. This file was written when the answer to "when
may it speak?" had a second clause - `scope`, which asked whether a request had to
carry an opt-in marker. That marker and that scope are gone (0.2.31): a request body
must not decide whether its KV comes off NVMe. The question now has one answer, which
is the boot's posture, so what is pinned here is the posture - two per-boot switches,
both off at boot, both raised only by the dev verb, and nothing a client sends able to
move either.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time
import types

import pytest
import test_disk_restore as base

from kvtier_dev_plugin import (
    backport,
    config,
    disk_layout,
    disk_restore,
    disk_spill,
    disk_store,
    endpoint,
)

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them (pytest collects fixtures from imported names).
armed = base.armed  # noqa: F811 - re-exported as a fixture, not shadowed
armed_gap_fill = base.armed_gap_fill  # noqa: F811 - only the gap-fill switch up
switches_down = base.switches_down  # noqa: F811 - `armed` with both switches down
allow_tmp_root = base.allow_tmp_root  # noqa: F811 - autouse: tmp_path is under /tmp
fresh = base.fresh  # noqa: F811 - autouse: a fresh boot per test, or counters leak

BLOCK = base.BLOCK
DIGEST = base.DIGEST
WIDTH = base.WIDTH
Manager = base.Manager
Req = base.Req
fill = base.fill
offer = base.offer


@pytest.fixture
def offering(armed):
    """Alias of `armed`, kept for the tests that only need "the gate is open".

    It used to raise the switch AND set `scope=all`; `armed` now raises disk authority,
    which opens the same gate, so the two are the same fixture and this name exists only
    so the incident-shape tests below keep reading as they always did.
    """
    return armed


# --- the posture: a fresh boot offers nothing ---------------------------------

def test_a_fresh_boot_offers_nothing_even_with_every_tier_armed(tmp_path,
                                                                monkeypatch):
    """`gate()` closes on the boot switches ALONE, with every other switch up.

    This is the module default, and it is the whole answer to "how did an experiment's
    arm put disk offers in front of live traffic": the restore path used to key off the
    spill arm like everything else. Arming the spill tier is not consent to restore.
    """
    root = tmp_path / "spill"
    root.mkdir()
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": True, "count": 1,
                                             "ranks": ["PP0_TP0"], "reason": None,
                                             "ranks_armed": 1, "ranks_seen": 1,
                                             "reads_applied": 0,
                                             "reason_kind": None})
    assert disk_restore._state["offers_enabled"] is False, \
        "the import-time default is the safety property; do not move it"
    assert disk_restore._state["disk_authority"] is False, \
        "a fresh boot gives disk authority to NOBODY, and that default is the whole " \
        "safety argument for making the switch POST-only"
    verdict = disk_restore.gate()
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "offers_disabled"
    assert '"op":"offers"' in verdict["reason"] \
        and '"op":"authority"' in verdict["reason"], \
        "the refusal has to name BOTH verbs that open the gate"
    # ...and it is the FIRST thing the offer path does, so a fresh boot costs the
    # request nothing beyond the fork's own lookup.
    assert offer(Manager(), Req(request_id="fresh", blocks=4)) == (0, False)
    assert disk_restore.counters()["refused"]["offers_disabled"] == 1


def test_both_switches_accept_only_an_explicit_boolean():
    """A typo in a payload must not change engine behaviour in either direction.

    The `scope` argument used to be validated here too, against two names. It is gone,
    and what it left behind is stricter rather than looser: a JSON boolean for either
    switch, and no third way to say "which requests?" - because there is no longer a
    question that needs asking.
    """
    for setter in (disk_restore.set_offer_switch, disk_restore.set_disk_authority):
        for junk in ("true", "false", 1, 0, None, [], {}, "", "all", "optin"):
            assert setter(junk)["ok"] is False, (setter.__name__, repr(junk))
        assert setter(True)["ok"] is True
        assert setter(False)["ok"] is True


def test_no_environment_variable_can_arm_or_authorize_anything():
    """The posture is in-process state or nothing.

    An env var that arms the restore path would be a distance-spelling of "restart and
    it is live", which is exactly the property the per-boot switch exists to remove.
    Asserted structurally: no constant in this module names an env var, and the
    posture setters read no environment.
    """
    import ast
    import inspect
    assert [name for name in vars(disk_restore)
            if name.startswith("ENV_")] == [], "this module owns no env var"
    tree = ast.parse(inspect.getsource(disk_restore))
    wanted = {"set_offer_switch", "set_disk_authority", "gate", "disk_posture",
              "serving_as", "suppress_load"}
    reading_env = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr in ("environ",
                                                                       "getenv"):
                    reading_env.append(node.name)
    assert reading_env == [], f"these read the environment: {sorted(set(reading_env))}"


def test_arming_the_tiers_raises_neither_switch(armed):
    """The 2026-10-05 confusion, as an assertion: an arm is not consent to serve.

    `armed` arms the spill tier and the RAM tier for every test in this file, and both
    boot switches still start down - so each has to be raised by name, and each opens
    the gate BY ITSELF. That second half is the operator-facing property: the README
    tells them to POST one op, and a gate that needed both would make that command a
    lie.
    """
    disk_restore.set_offer_switch(False)
    disk_restore.set_disk_authority(False)
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    disk_restore.set_offer_switch(True)
    assert disk_restore.gate()["ok"] is True
    disk_restore.set_offer_switch(False)
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    disk_restore.set_disk_authority(True)
    assert disk_restore.gate()["ok"] is True, \
        "disk authority opens the gate by itself: one POST is the whole change"


# --- nothing a client sends is a door ---------------------------------------

def test_the_module_reads_no_request_field_that_could_authorize_a_disk_read():
    """THE REMOVAL, PINED STRUCTURALLY: a behavioural test cannot prove an absence.

    The marker used to be read at one place on the offer path, and the module kept a
    constant naming the key so the route help, the payload and the log could all quote
    it. All of that is gone, and the way to keep it gone is to name the CODE shapes that
    must not come back:

      * no attribute read of `kv_transfer_params` anywhere in the module (the read that
        carried the marker), proved on the syntax tree, not on a string search;
      * no `RESTORE_MARKER_KEY` / `RESTORE_SCOPE*` constant, so nothing can quote a
        marker key or a scope name even by accident;
      * no `request_flagged` / `set_scope` function, so there is no door to re-open;
      * no counter named for the marker's bookkeeping, because a permanently-zero
        counter in a payload is a payload that lies.

    Prose about the retired marker is allowed and wanted - the WHY is the part worth
    keeping - so this checks identifiers and attribute reads, not sentences.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(disk_restore))
    read_attrs = {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute)}
    assert "kv_transfer_params" not in read_attrs, \
        "something reads the engine field the marker travelled in"
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for retired in ("request_flagged", "set_scope", "_skip_unless_flagged"):
        assert retired not in defined, retired
    assigned = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    for retired in ("RESTORE_MARKER_KEY", "RESTORE_SCOPE_OPTIN", "RESTORE_SCOPE_ALL",
                    "RESTORE_SCOPES", "RESTORE_SCOPE_DEFAULT"):
        assert retired not in assigned, retired
    counters = disk_restore.counters()
    for retired in ("flagged", "skipped_no_marker", "requests_seen", "scope",
                    "scope_changes"):
        assert retired not in counters, retired
    assert disk_restore.serving_as() in ("off", "disk_authority",
                                         "ram_first_gap_fill")


def test_a_boot_with_the_switches_down_touches_no_index_and_no_journal(switches_down,
                                                                      monkeypatch):
    """THE COST GUARANTEE: the fork's answer, and no lookup, no refresh, no alloc.

    The gate is the first thing in the path, so a boot that has not been told to serve
    cannot pay for a journal refresh or an allocation. A check placed AFTER the
    expensive work would be a gate in name only, so this asserts the ABSENCE of the work
    rather
    than the value that came back - and then raises a switch on the same boot to prove
    the silence above was the posture and not a broken fixture.

    This is what the retired "unflagged request" version of the test pinned. The
    population changed (it is now "a boot that was not turned on" rather than "a request
    that did not ask") and the guarantee did not.
    """
    fill(switches_down, blocks=4)
    lookups, refreshes = [], []
    real_contains = disk_restore.disk_index.contains
    watching = {"on": True}

    def spy_contains(*a, **k):
        lookups.append(a)
        # Spy during the closed-gate call, then let the real index answer so the
        # open-gate half of this test fails only if the restore path really fails.
        return False if watching["on"] else real_contains(*a, **k)

    monkeypatch.setattr(disk_restore.disk_index, "contains", spy_contains)
    real_refresh = disk_restore._refresh_index

    def counted_refresh(*a, **k):
        refreshes.append(1)
        return real_refresh(*a, **k)

    monkeypatch.setattr(disk_restore, "_refresh_index", counted_refresh)
    assert offer(Manager(), Req(request_id="quiet", blocks=4)) == (0, False)
    assert lookups == [], "a closed boot must not query the disk index"
    assert refreshes == [], "a closed boot must not refresh the journal"
    counters = disk_restore.counters()
    assert counters["refused"]["offers_disabled"] == 1
    assert counters["offers"] == 0
    assert counters["slots_allocated"] == 0
    # The same boot, switched on, does the work - so the silence above was the posture.
    watching["on"] = False
    asked = len(lookups)
    disk_restore.set_disk_authority(True)
    assert offer(Manager(), Req(request_id="loud", blocks=4)) == (4 * BLOCK, True)
    assert refreshes, "an open boot must reach the journal"
    assert len(lookups) > asked, "an open boot must query the index"


def test_the_gap_fill_switch_offers_without_anything_from_the_client(armed_gap_fill):
    """`scope=all`'s job, now done by a switch: every request is a candidate.

    Same store, same plain request, and the offer happens - because the boot says so.
    Nothing on the request was consulted, which is the property the whole change bought:
    what used to need a per-request marker now needs one POST.
    """
    fill(armed_gap_fill, blocks=4)
    assert offer(Manager(), Req(request_id="plain", blocks=4)) == (4 * BLOCK, True)
    counters = disk_restore.counters()
    assert counters["offers"] == 1
    assert counters["authority_lookups"] == 0, \
        "gap fill is not authority: the counters must keep the two postures apart"


def test_the_gate_verdict_outranks_everything_else_in_why_no_restore(armed,
                                                                     monkeypatch):
    """A closed gate outranks every other missing link in `why_no_restore`.

    Several things can be true of the same request, and the operator's question is
    "could this boot have served it at all". Answering with a downstream link while the
    gate is shut sends them chasing the wrong thing entirely - on 2026-10-05 it sent me
    into client payloads while the arm file was the answer.
    """
    fill(armed, blocks=2)
    monkeypatch.setattr(disk_restore, "_state", dict(disk_restore._state))
    disk_restore._state["sched_fitted"] = True
    monkeypatch.setattr(disk_restore, "fitted_methods_here",
                        lambda: list(disk_restore.SCHED_METHODS))
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    why = disk_restore.scheduler_section()["why_no_restore"]
    assert "disarmed" in why, why
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    disk_restore.set_disk_authority(False)
    why = disk_restore.scheduler_section()["why_no_restore"]
    assert "switch" in why or "authority" in why, why
    disk_restore.set_disk_authority(True)
    why = disk_restore.scheduler_section()["why_no_restore"]
    assert "switch" not in why, \
        f"an open boot must not keep blaming the switch: {why}"


def test_the_posture_is_reported_wherever_it_can_be_read(armed):
    """Both switches, the one-word verdict, the offer cap and the rank split.

    Boot line, witness line, counters and payload. A posture that lives only in a config
    file - or only in a sentence an operator has to remember the spelling of - is a
    posture nobody finds at 3 a.m., and `disk_authority` is the one field that answers
    "is this boot serving everyone from SSD?".
    """
    line = disk_restore.boot_line()
    for field in ("disk_authority=ON", "serving_as=disk_authority",
                  "offers_switch=OFF", "offer_cap=", "ranks_armed=", "suppressed=",
                  "authority_lookups="):
        assert field in line, field
    for retired in ("scope=", "marker="):
        assert retired not in line, retired
    posture = disk_restore.disk_posture()
    assert posture["disk_authority"] is True, "the `armed` fixture raises authority"
    assert posture["enabled"] is False
    assert posture["serving_as"] == "disk_authority"
    assert posture["per_boot"] is True and posture["survives_restart"] is False
    counters = disk_restore.counters()
    for key in ("disk_authority", "offers_enabled", "serving_as",
                "authority_lookups", "disk_authority_changes",
                "offer_switch_changes",
                "suppressed_hits", "suppressed_spans", "suppressed_requests",
                "ranks_seen", "no_read", "offer_clamped", "witness_kind"):
        assert key in counters, key
        assert key in counters["units"], key
    # The witness line, which is what a rank prints when it disagrees.
    witness = disk_restore.witness_line()
    assert "disk_authority=on" in witness and "serving_as=disk_authority" in witness
    assert "scope=" not in witness


# --- the posture in the one-call payload, as FIELDS ---------------------------

def test_the_snapshot_payload_carries_the_posture_as_fields(armed):
    """Every posture fact is a JSON field of `disk_restore`, in BOTH switch states.

    The question the snapshot exists to answer at 3 a.m. is "what will the restore path
    do?", and for a while it could not be answered from the payload: the section carried
    36 keys and not one of them was the switch, the scope, the cap, the armed-rank split
    or the fuse - all of it was reachable only as text inside `boot_line`, and a
    dashboard cannot grep a sentence. Every value below is a pass-through from the
    function that owns it, asserted AS that function's value, which also pins "the
    payload and the verb cannot disagree".
    """
    section = disk_restore.scheduler_section()
    posture = disk_restore.disk_posture()
    counters = disk_restore.counters()
    new_keys = ("disk_authority", "offers_enabled", "serving_as", "per_boot",
                "survives_restart", "disk_authority_changes", "offer_switch_changes",
                "authority_lookups",
                "offer_cap", "ranks_armed",
                "ranks_seen", "suppressed_hits", "suppressed_spans",
                "suppressed_requests", "suppressions",
                # THE APPLIED-READ EVIDENCE. `restorations` is the counter that read
                # as success through the 2026-10-05 livelock; these are the direct
                # ones, and they existed in `_state` while the payload had none.
                "reads_applied", "disk_reads_applied", "ranks_reads_applied",
                "loads_released", "witness_verdict",
                "witness_reason", "witness_kind")
    for key in new_keys:
        assert key in section, key
        assert str(section["units"].get(key, "")).strip(), \
            f"{key} has no unit: a bare number under a units map"
    # THE VALUES, FROM THEIR OWNERS.
    assert section["disk_authority"] is True is posture["disk_authority"]
    assert section["offers_enabled"] is False is posture["enabled"]
    assert section["serving_as"] == posture["serving_as"] == "disk_authority"
    # The lifetime is part of the posture, not a footnote: a restart gives disk
    # authority to nobody, and the payload has to say so for the reader who was not
    # here when the POST happened.
    assert section["per_boot"] is True
    assert section["survives_restart"] is False
    assert section["offer_cap"] == disk_restore.MAX_OFFER_TOKENS_PER_OFFER
    assert section["ranks_armed"] == 1 and section["ranks_seen"] == 1
    for key in ("suppressed_hits", "suppressed_spans", "suppressed_requests",
                "suppressions"):
        assert section[key] == counters[key] == 0, key
    # THE APPLIED-READ PASS-THROUGH, asserted AS `counters()`' value: this section
    # recomputes nothing, so a payload and a verb can never disagree about whether a
    # read was applied. `proof_reads` is deliberately NOT in this family - it counts
    # rank files opened by `worker_proof()`, not KV reads - and its unit says so.
    applied = ("reads_applied", "disk_reads_applied", "ranks_reads_applied",
               "loads_released", "authority_lookups")
    for key in applied:
        assert section[key] == counters[key] == 0, key
    for key in ("witness_verdict", "witness_reason", "witness_kind"):
        assert section[key] == counters[key], key
    assert "NOT KV reads" in section["units"]["proof_reads"], \
        "proof_reads counts rank files, and must not be misread as a KV read"

    # THE OTHER SWITCH STATE. Same keys, different verdict: a payload that drops the
    # posture when the posture is off is a payload that cannot tell "off" from "gone".
    disk_restore.set_offer_switch(False)
    off = disk_restore.scheduler_section()
    assert off["offers_enabled"] is False
    assert set(off) == set(section), "the shape must not depend on the posture"
    # The applied-read evidence is present in the OFF state too: a predicate that
    # watches `reads_applied` must be able to read it before the switch is raised,
    # and a key that appears only once traffic is allowed is a key that cannot
    # distinguish "off" from "gone".
    for key in ("reads_applied", "disk_reads_applied", "ranks_reads_applied",
                "loads_released", "authority_lookups", "witness_verdict",
                "witness_reason", "witness_kind"):
        assert key in off, key
    # THE OTHER SWITCH, MOVING INDEPENDENTLY of authority: a boot can run the cheap
    # posture with authority down, and the payload must show exactly that.
    disk_restore.set_disk_authority(False)
    disk_restore.set_offer_switch(True)
    scoped = disk_restore.scheduler_section()
    assert scoped["offers_enabled"] is True and scoped["disk_authority"] is False
    assert scoped["serving_as"] == "ram_first_gap_fill"
    disk_restore.set_disk_authority(True)
    assert disk_restore.scheduler_section()["serving_as"] == "disk_authority"


def test_the_payload_names_the_armed_ranks_when_only_the_scheduler_is_armed(four_ranks):
    """`ranks_armed` as data: the incident's shape, readable without reading prose.

    123,474 offers went out on a boot where the scheduler was armed and FOUR ranks
    refused to service a load. The proof has always computed the split, but the payload
    published only `ranks_proven`, which reads as "nothing proven" rather than "4 live,
    0 servicing" - two different diagnoses, and only one of them is the incident.
    """
    four_ranks({"worker_fitted": False,
                "service_reason_kind": "worker_half_not_fitted"})
    disk_restore.set_offer_switch(True)
    section = disk_restore.scheduler_section()
    assert section["ranks_armed"] == 0
    assert section["ranks_seen"] == 4
    assert section["ranks_proven"] == 0
    assert section["offers_enabled"] is True, "the switch was up; the ranks were not"
    assert disk_restore.gate()["ranks_armed"] == "0/4"
    # Lift the veto on the same fleet: the split moves, and nothing else about the
    # payload does.
    four_ranks({"armed": True, "reads": 1})
    armed = disk_restore.scheduler_section()
    assert armed["ranks_armed"] == 4 and armed["ranks_seen"] == 4
    assert armed["ranks_proven"] == 4


def test_the_fuse_is_reported_as_data_in_the_payload(armed):
    """The suppression counters reach the payload, so the loop is visible while it runs.

    The livelock was invisible in the payload and visible only as a repeating log line.
    These three counters are the difference between "nothing is happening" and "a
    request is being re-offered and refused on a fuse" - the same offer count, with no
    way to tell the two apart from numbers alone.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    failed = fakes_output(failed={"req-1"})
    assert disk_restore.note_reported(manager, failed)["failures"] == 1
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=1)) == (0, False)
    counters = disk_restore.counters()
    assert counters["suppressed_hits"] == 1
    assert counters["suppressed_requests"] == 1
    assert counters["suppressions"] == 1
    section = disk_restore.scheduler_section()
    for key in ("suppressed_hits", "suppressed_spans", "suppressed_requests",
                "suppressions"):
        assert section[key] == counters[key], key
    assert section["suppressed_hits"] == 1
    assert section["suppressed_requests"] == 1
    assert section["suppressions"] == 1, "the lifetime total, not only the live table"


def test_the_offer_is_capped_below_the_incident_size(armed):
    """160,000 tokens per offer is unreachable at any prompt length.

    Every one of the incident's 123,474 offers was EXACTLY 160,000 tokens (50 spans):
    a cap expressed only in blocks and slots is a cap the request's own length can
    reach. The ceiling is in tokens now, and the clamp keeps whole spans so the fork's
    "one whole block per entry" rule still holds afterwards.
    """
    blocks = (disk_restore.MAX_OFFER_TOKENS_PER_OFFER // BLOCK) + 8
    fill(armed, blocks=blocks)
    tokens, flag = offer(Manager(), Req(request_id="big", blocks=blocks))
    assert flag is True
    assert 0 < tokens <= disk_restore.MAX_OFFER_TOKENS_PER_OFFER
    assert tokens % BLOCK == 0, "a clamp must not cut a span in half"
    assert disk_restore.counters()["offer_clamped"] == 1
    assert disk_restore.MAX_OFFER_TOKENS_PER_OFFER < 160_000


# --- the incident's fuel: armed scheduler, disarmed workers -------------------

def _rank(directory, rank, *, installed=True, armed=True, reads=0, reason=None,
          worker_fitted=True, service_reason_kind=None):
    """One rank file, in the shape `worker_diag` writes.

    0.2.28: the per-rank SERVICE facts are `worker_fitted` and
    `service_reason_kind`; the TIER ARM is no longer read from these files at all
    (the proof reads `disk-spill-arm.json` through `disk_spill._read_arm_file`,
    boot fence intact), so a file's `armed: false` is a lagged mirror - counted as
    a stale claim, never a veto. The incident shape (a rank that cannot bind) is
    spelled `worker_fitted=False` now.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"rank-{rank}.json").write_text(json.dumps({
        "version": 1, "epoch": "aabbccdd", "ts": round(time.time(), 6),
        "role": "worker", "rank": rank, "arena": {}, "counters": {},
        "writer": {"epoch_scope": "process"},
        "disk_restore": {"installed": installed, "armed": armed,
                         "worker_fitted": worker_fitted,
                         "service_reason_kind": service_reason_kind,
                         "reads_applied": reads, "reason": reason}}))


@pytest.fixture
def four_ranks(tmp_path, monkeypatch):
    """A scheduler-side boot with its own diagnostics directory and a filled journal.

    The rank files are REAL files read by the real `worker_proof`: the gate is the
    thing under test, so nothing here monkeypatches the proof.
    """
    root = tmp_path / "spill"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setattr(config, "diagnostics_dir", lambda: diag)
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(disk_spill, "geometry_digest", lambda: DIGEST)
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    monkeypatch.setattr(disk_store, "counters", lambda: {
        "geometry_digest": DIGEST,
        "fingerprint": {"total_bytes_per_block": WIDTH}, "rank": "PP0_TP0"})
    utils = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    utils.make_block_hash_with_group_id = base.pack
    utils.resolve_block_hashes = base.resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, utils)
    disk_restore._proof_cache.update({"mono": None, "result": None})

    RANKS = ("PP0_TP0", "PP1_TP0", "PP2_TP0", "PP3_TP0")

    # THE ARM FILE, as the proof reads it now (0.2.28): the tier arm is a
    # rank-agnostic boot fact owned by EngineCore, so a scheduler-side boot that
    # is armed HAS the arm file - these tests patch `is_armed` for the GATE's
    # cached read, and the proof's direct read needs the file itself.
    (diag / disk_spill.ARM_FILE).write_text(json.dumps(
        {"armed": True, "expires_ts": time.time() + 3600.0}))

    def write(shared=None, **by_rank):
        shared = shared or {}
        for rank in RANKS:
            fields = dict(shared)
            fields.update(by_rank.get(rank) or {})
            _rank(diag, rank, **fields)
        disk_restore._proof_cache.update({"mono": None, "result": None})

    fill(str(root), blocks=4)
    return write


def test_armed_scheduler_with_unrefused_workers_offers_nothing(four_ranks):
    """THE FUEL TEST: scheduler armed, every rank unable to bind, zero external
    tokens.

    This is the one line of gate logic that would have turned 123,474 load failures, a
    wedged harness and a watchdog reset into a boot that did nothing at all.

    0.2.28 spells the incident as `worker_fitted=False` - the per-rank fact the
    rank file uniquely knows. The rank files' `armed: false` no longer spells it:
    that verdict is a lagged mirror of the arm file, and refusing on it is the
    2026-10-20 lockout (pinned by `test_disk_index_journal_drain.py`).
    """
    four_ranks({"worker_fitted": False,
                "service_reason_kind": "worker_half_not_fitted"})
    disk_restore.set_offer_switch(True)
    gate = disk_restore.gate()
    assert gate["ok"] is False
    assert gate["reason_kind"] == "scheduler_armed_workers_disarmed"
    assert gate["ranks_armed"] == "0/4", gate

    tokens, flag = offer(Manager(), Req(request_id="fuel", blocks=4))
    assert (tokens, flag) == (0, False), "the fuel test IS the incident"
    counters = disk_restore.counters()
    assert counters["refused"]["workers_not_servicing"] == 1
    assert counters["slots_allocated"] == 0
    assert counters["offered_tokens"] == 0
    assert "workers_not_servicing" in str(counters["last_refusal"]) and \
        "refuse disk service" in counters["last_reason"], \
        "the refusal is carried by name AND by the rank-file split"

    # Pull the veto on all four ranks: the veto was the entire story, nothing
    # else moved.
    four_ranks({"armed": True, "reads": 2})
    assert disk_restore.gate()["ok"] is True
    assert offer(Manager(), Req(request_id="fuel-2", blocks=4)) == (4 * BLOCK, True)


def test_a_partial_fleet_refuses_like_a_whole_one(four_ranks):
    """3 armed + 1 refusing is a refusal. The load event spans every rank.

    A "serve it on the ranks that can" offer is the incident in a nicer uniform: the
    refusing rank leaves blocks the fork already committed to, and one such block
    truncates the whole request - including the legitimate host-RAM hit it was riding
    with.
    """
    four_ranks(**{rank: {"worker_fitted": False,
                         "service_reason_kind": "worker_half_not_fitted"}
                  for rank in ("PP0_TP0", "PP1_TP0", "PP2_TP0")} |
                {"PP3_TP0": {"armed": True, "reads": 4}})
    disk_restore.set_offer_switch(True)
    verdict = disk_restore.worker_proof(force=True)
    assert verdict["ok"] is False and verdict["ranks_armed"] == 1
    assert verdict["ranks_seen"] == 4
    assert "ranks_armed=1/4" in verdict["reason"]
    assert offer(Manager(), Req(request_id="partial", blocks=4)) == (0, False)


# --- the incident's loop ------------------------------------------------------

def test_a_failed_load_is_never_offered_again_to_the_same_request(armed):
    """OFFER, FAIL, ASK AGAIN ON THE SAME ID => no second offer, counter moved.

    This fails against the code before the fuse exists, which is the point: with no
    memory of the failure the second answer is the first answer, and the loop is what
    the engine's own recovery drives.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    request = Req(request_id="req-1", blocks=2)
    assert offer(manager, request) == (2 * BLOCK, True)
    allocated = disk_restore.counters()["slots_allocated"]
    assert allocated > 0, "the first offer has to really have allocated"

    failed = fakes_output(failed={"req-1"})
    report = disk_restore.note_reported(manager, failed)
    assert report["failures"] == 1
    assert disk_restore.counters()["suppressions"] == 1

    again = offer(Manager(reqs_to_load={"req-1": object()}), request)
    assert again == (0, False), "a re-offered failure is the loop itself"
    counters = disk_restore.counters()
    assert counters["refused"]["suppressed_after_failure"] == 1
    assert counters["suppressed_hits"] == 1
    assert counters["slots_allocated"] == allocated, \
        "a suppressed request must not allocate one more slot"
    assert counters["offers"] == 1, "one offer happened, for two askings"


def test_a_failed_span_is_refused_for_every_other_request_too(armed):
    """The span half of the fuse: a fresh id wanting the same span is refused.

    The harness that caused the incident replayed the same prompts, so the failing
    span was re-requested by new request ids over and over. Fusing the id alone would
    have let all of them through, and each one would have truncated its own request.
    """
    fill(armed, blocks=2)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=2)) == (2 * BLOCK, True)
    disk_restore.note_reported(manager, fakes_output(failed={"req-1"}))
    assert offer(Manager(reqs_to_load={"req-9": object()}),
                 Req(request_id="req-9", blocks=2)) == (0, False)
    assert disk_restore.counters()["suppressed_spans"] == 1


def test_invalid_block_ids_without_a_request_name_still_burn_the_load(armed):
    """The block-id-only failure shape is not allowed to keep offering.

    `get_block_ids_with_load_errors` is OURS (the base connector returns an empty
    set), so invalid ids on a step that carried our slots are ours even when
    `failed_recving` stays empty. Deliberately over-broad: a wrongly-fused request
    loses one restore, an unfused failure feeds the loop.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    slots = base.merged_cpu(manager)
    disk_restore.build_load_paths(manager, base.Meta(load_event=3, cpu=slots,
                                                     gpu=list(range(len(slots))),
                                                     reqs={3: ["req-1"]}))
    out = fakes_output(invalid={700 + n for n in range(len(slots))})
    assert disk_restore.note_reported(manager, out)["failures"] == 1
    assert disk_restore.counters()["suppressions"] == 1
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=1)) == (0, False)


def test_a_load_that_outlived_its_deadline_is_a_failure_too(armed, monkeypatch):
    """The watchdog expiry is a failed load, and a failed load is a fuse.

    The fork has no timeout for WAITING_FOR_REMOTE_KVS, so without this the request
    would be re-offered indefinitely by a step that never reported.
    """
    fill(armed, blocks=1)
    manager = Manager(reqs_to_load={"req-1": object()})
    assert offer(manager, Req(request_id="req-1", blocks=1)) == (BLOCK, True)
    disk_restore._loads["req-1"]["deadline_mono"] = time.monotonic() - 1.0
    disk_restore.build_load_paths(manager, base.Meta(load_event=1, cpu=[], gpu=[],
                                                     reqs={}))
    assert disk_restore.counters()["suppressions"] == 1
    assert offer(Manager(reqs_to_load={"req-1": object()}),
                 Req(request_id="req-1", blocks=1)) == (0, False)


def fakes_output(*, finished=(), failed=(), invalid=()):
    """A `KVConnectorOutput` with every field the scheduler reads on it."""
    return types.SimpleNamespace(finished_recving=set(finished),
                                 finished_sending=set(),
                                 failed_recving=set(failed),
                                 invalid_block_ids=set(invalid))


def test_the_suppression_table_is_bounded():
    """A fuse that grows without bound is its own leak, so it is capped and oldest-
    first: forgetting the oldest suppression costs at most one more offer."""
    for ordinal in range(disk_restore.MAX_SUPPRESSED_REQUESTS + 500):
        disk_restore.suppress_load(f"req-{ordinal}", "test pressure")
    assert len(disk_restore._suppressed) <= disk_restore.MAX_SUPPRESSED_REQUESTS
    newest = f"req-{disk_restore.MAX_SUPPRESSED_REQUESTS + 499}"
    assert newest in disk_restore._suppressed


def test_suppress_load_survives_a_hostile_request_id():
    """A fuse must never be the thing that raises into the scheduler."""
    for junk in ("", None, 17, {"a": 1}, "x" * 5000, ["list"], object()):
        disk_restore.suppress_load(junk, "hostile")
    assert isinstance(disk_restore.counters()["suppressions"], int)


# --- the wording fix ----------------------------------------------------------

def test_the_plugin_load_deferral_reads_as_a_deferral_not_a_refusal(caplog):
    """`DISK[restore-install] deferred=YES`, at INFO, not a WARNING-shaped refusal.

    At plugin load the tier classes are legitimately not imported yet - that is the
    ORDER of the engine, not a fault. Counting it as a refusal and logging it at
    WARNING made every healthy boot's first line look like a failure, and made the
    real refusals indistinguishable from it.
    """
    sys.modules.pop(disk_restore.SCHED_MODULE, None)
    sys.modules.pop(disk_restore.WORKER_MODULE, None)
    with caplog.at_level(logging.INFO, logger="kvtier_dev_plugin.disk_restore"):
        result = disk_restore.install()
    assert result.get("installed") is not True
    lines = [rec for rec in caplog.records if "restore-install" in rec.getMessage()]
    assert lines, "a deferral has to say so"
    for record in lines:
        assert "deferred=YES" in record.getMessage(), record.getMessage()
        assert record.levelno <= logging.INFO, record.getMessage()
    counters = disk_restore.counters()
    assert counters["install_refusals"] == 0
    assert counters["install_deferrals"] >= 1


def test_a_real_install_refusal_is_still_a_warning(caplog, monkeypatch):
    """The other half of the wording fix: a real refusal keeps its volume up."""
    module = types.ModuleType(disk_restore.SCHED_MODULE)

    class Present:
        """There, but with none of the methods the layer needs."""

    module.SimpleCPUOffloadScheduler = Present
    monkeypatch.setitem(sys.modules, disk_restore.SCHED_MODULE, module)
    sys.modules.pop(disk_restore.WORKER_MODULE, None)
    with caplog.at_level(logging.WARNING, logger="kvtier_dev_plugin.disk_restore"):
        disk_restore.install()
    warnings = [rec.getMessage() for rec in caplog.records
                if rec.levelno >= logging.WARNING]
    assert any("restore-install" in text and "refused=YES" in text
               for text in warnings), warnings
    assert disk_restore.counters()["install_refusals"] >= 1

# --- the dev verb that raises the switches ------------------------------------

def test_the_offers_op_reads_the_posture_without_changing_it(armed):
    """`{"op":"offers"}` alone is a read: an operator checking the state must not
    change it, and every mutating op here is POST-only by the route's own rule."""
    disk_restore.set_offer_switch(False)
    before = disk_restore.disk_posture()
    out = disk_store.disk_for_engine(None, {"op": "offers"})
    assert out["ok"] is True
    assert out["posture"]["enabled"] == before["enabled"] is False
    assert out["posture"]["disk_authority"] == before["disk_authority"]
    assert out["per_boot"] is True and out["survives_restart"] is False


def test_the_authority_op_reads_the_posture_without_changing_it(armed):
    """Same rule for the new op. A read must be a read."""
    out = disk_store.disk_for_engine(None, {"op": "authority"})
    assert out["ok"] is True
    assert out["posture"]["disk_authority"] is True, "`armed` raised it"
    assert out["posture"]["serving_as"] == "disk_authority"


def test_the_authority_op_turns_the_switch_on_and_off(armed):
    """ONE POST is the whole change: authority alone opens the gate and serves.

    The gap-fill switch stays down throughout, because the operator's command is the
    authority op and it must not need a second call. Then back off, without a restart.
    """
    disk_restore.set_disk_authority(False)
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    out = disk_store.disk_for_engine(None, {"op": "authority", "enabled": True})
    assert out["ok"] is True, out
    assert out["disk_authority"]["enabled"] is True
    assert out["per_boot"] is True and out["survives_restart"] is False
    posture = disk_store.disk_for_engine(None, {"op": "authority"})["posture"]
    assert posture["disk_authority"] is True
    assert posture["enabled"] is False, "the other switch is not this op's business"
    assert posture["serving_as"] == "disk_authority"
    assert disk_restore.gate()["ok"] is True
    # And back down: a boot can be made inert again without a restart.
    assert disk_store.disk_for_engine(None, {"op": "authority",
                                             "enabled": False})["ok"] is True
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"


def test_the_authority_op_refuses_a_value_that_is_not_a_boolean(armed):
    """A typo must answer `ok: false`, not silently do nothing: the operator has to know
    the switch did not move."""
    for junk in ("true", 1, "yes", "on", None):
        out = disk_store.disk_for_engine(None, {"op": "authority", "enabled": junk})
        assert out["ok"] is False, (junk, out)
        assert disk_restore.disk_posture()["disk_authority"] is True, \
            f"{junk!r} was refused but still moved the switch"


def test_the_offers_op_refuses_a_value_that_is_not_a_boolean(armed):
    for junk in ("true", 1, "yes"):
        out = disk_store.disk_for_engine(None, {"op": "offers", "enabled": junk})
        assert out["ok"] is False, (junk, out)
    # `scope` was an accepted key until 0.2.31. It is not one now, and a caller still
    # sending it must be told rather than served - the route's unknown-key rule does
    # that at the door; here the op-level handler must not invent a meaning for it.
    out = disk_store.disk_for_engine(None, {"op": "offers", "scope": "all"})
    assert out["ok"] is True, "a body this handler ignores is ignored, not obeyed"
    assert disk_restore.disk_posture().get("scope") is None


def test_both_switch_ops_are_listed_and_post_only():
    """The route's GET/POST split is the guard on every mutating op, so a new op has
    to be in both lists or it is either unreachable or reachable by a page load."""
    for op in ("offers", "authority"):
        assert op in disk_store.DISK_OPS, op
        assert op in disk_store.MUTATING_DISK_OPS, op
    schema = endpoint.ARG_SPECS[config.ROUTE_DISK]
    assert "authority" in schema["op"]["values"]
    assert "offers" in schema["op"]["values"]
    assert schema["enabled"]["kind"] == "bool"
    # The scope argument is gone from the route's accepted vocabulary, so a client still
    # sending `?scope=all` is refused by the unknown-key rule instead of being obeyed.
    assert "scope" not in schema


def test_the_verb_documentation_no_longer_offers_a_request_marker():
    """The help text is the operator's only source, and a stale one is trusted.

    Until 0.2.31 it told them to put `{"kv_transfer_params": {"kvtier_disk_restore":
    true}}` in the request body. That sentence has to be GONE, not merely added to:
    an operator who follows it today gets a silently un-marked request and concludes the
    feature is broken. What replaces it is the op, the boot-only lifetime, and the fact
    that no client field decides anything - in the route help and in the README.
    """
    doc = endpoint.route_disk_post.__doc__ or ""
    assert "kvtier_disk_restore\": true" not in doc, \
        "the help still tells a client to send the retired marker"
    assert '"op":"authority"' in doc, doc
    assert "op: status|arm|disarm|spill-one|ls|contains|offers|authority|failures" \
        in doc, doc
    assert "EVERY" in doc and "no client can ask for disk authority or decline it" \
        in doc, "the posture must be described in operator words"
    readme = (pathlib.Path(__file__).resolve().parent.parent
              / "README.md").read_text()
    assert '"op":"authority"' in readme, "the README must name the op"
    assert '"kv_transfer_params": {"kvtier_disk_restore": true}' not in readme


# --- the applied-read evidence reaches the payload ---------------------------

def test_the_applied_read_counters_reach_the_payload_as_a_pass_through(armed):
    """A number that only exists in `_state` is a number no poller can use.

    The 2026-10-05 livelock is the reason for this test's shape: `restorations` rose
    and the witness said CONFIRMED while the applied-read counter sat at zero, and
    the payload carried only the first of those two. So the split is asserted here in
    BOTH directions - a read that really was applied moves the payload, and a
    restoration with no read behind it is visible as exactly that.
    """
    before = disk_restore.scheduler_section()
    assert before["reads_applied"] == 0

    # A worker-side read applied: the DIRECT counter, not the reported-back one.
    disk_restore._state["reads_applied"] = 7
    disk_restore._state["disk_reads_applied"] = 3
    disk_restore._state["ranks_reads_applied"] = 11
    disk_restore._state["loads_released"] = 5
    disk_restore._state["authority_lookups"] = 9
    after = disk_restore.scheduler_section()
    counters = disk_restore.counters()
    for key, value in (("reads_applied", 7), ("disk_reads_applied", 3),
                       ("ranks_reads_applied", 11), ("loads_released", 5),
                       ("authority_lookups", 9)):
        assert after[key] == value, key
        # A PASS-THROUGH, not a recomputation: the section and the counters agree.
        assert after[key] == counters[key], key

    # THE COSTUME COUNTER, caught. A finished report with no read behind it is
    # `no_read`, never a restoration; the payload shows both numbers so a predicate
    # can tell "bytes came back" from "bytes were read".
    disk_restore._state["restorations"] = 4
    disk_restore._state["no_read"] = 4
    split = disk_restore.scheduler_section()
    assert split["restorations"] == 4
    assert disk_restore.counters()["no_read"] == 4, (
        "the refusal of the costume counter stays in counters()")
    assert split["reads_applied"] == 7, "the direct counter is unaffected by either"
    assert "reads memmoved" in split["units"]["reads_applied"]


def test_the_witness_verdict_travels_with_the_counters_it_quotes(armed):
    """`witness_verdict` is prose's number and must not be read instead of one.

    Its unit says so, and the three `witness_*` fields ship together so a reader who
    takes the verdict also has the stable kind and the sentence that explains it -
    while `reads_applied` sits in the same payload, unlifted.
    """
    disk_restore._state["witness_verdict"] = "NO READ"
    disk_restore._state["witness_kind"] = "no_read"
    disk_restore._state["witness_reason"] = "no applied read behind any report"
    section = disk_restore.scheduler_section()
    assert section["witness_verdict"] == "NO READ"
    assert section["witness_kind"] == "no_read"
    assert section["witness_reason"] == "no applied read behind any report"
    assert section["reads_applied"] == 0, "the verdict does not move the counter"
    for key in ("witness_verdict", "witness_reason", "witness_kind"):
        assert str(section["units"][key]).strip(), key
