r"""The request-scoped gate, the offer posture, and the two incident shapes.

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
    """The gate open, the switch raised, and scope `all`.

    Only for tests about behaviour PAST the posture. The `armed` fixture already raises
    the switch and leaves the scope at its default `optin`.
    """
    assert disk_restore.set_scope(disk_restore.RESTORE_SCOPE_ALL)["ok"] is True
    return armed


# --- the posture: a fresh boot offers nothing ---------------------------------

def test_a_fresh_boot_offers_nothing_even_with_every_tier_armed(tmp_path,
                                                                monkeypatch):
    """`gate()` closes on the offer switch ALONE, with every other switch up.

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
    verdict = disk_restore.gate()
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "offers_disabled"
    assert "offers" in verdict["reason"] and "op" in verdict["reason"], \
        "the refusal has to name the verb that raises the switch"
    # ...and it is the FIRST thing the offer path does, so a fresh boot costs the
    # request nothing beyond the fork's own lookup.
    assert offer(Manager(), Req(request_id="fresh", blocks=4)) == (0, False)
    assert disk_restore.counters()["refused"]["offers_disabled"] == 1


def test_the_switch_and_the_scope_accept_only_an_explicit_value():
    """A typo in a payload must not change engine behaviour in either direction."""
    for junk in ("true", "false", 1, 0, None, [], {}):
        assert disk_restore.set_offer_switch(junk)["ok"] is False, repr(junk)
    assert disk_restore.set_offer_switch(True)["ok"] is True
    for junk in ("Optin", "ALL", "", None, 1, ["all"]):
        assert disk_restore.set_scope(junk)["ok"] is False, repr(junk)
    assert disk_restore.set_scope("all")["ok"] is True
    assert disk_restore.set_scope("optin")["ok"] is True
    assert disk_restore.offer_posture()["scope"] == "optin"


def test_no_environment_variable_can_arm_or_scope_anything():
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
    wanted = {"set_offer_switch", "set_scope", "gate", "offer_posture",
              "request_flagged", "suppress_load"}
    reading_env = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr in ("environ",
                                                                       "getenv"):
                    reading_env.append(node.name)
    assert reading_env == [], f"these read the environment: {sorted(set(reading_env))}"


def test_arming_the_tiers_does_not_raise_the_offer_switch(armed):
    disk_restore.set_offer_switch(False)
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    disk_restore.set_offer_switch(True)
    assert disk_restore.gate()["ok"] is True


# --- the opt-in marker --------------------------------------------------------

def test_only_the_boolean_true_flags_a_request():
    """Strictness, spelled out: a real dict, the key, and the boolean True.

    A client controls this field. The cost of being loose about it is disk work on a
    request that never asked for it, so `"true"`, `1`, `1.0` and a nested object all
    mean "not asked", and a request object without the attribute is simply not flagged.
    """
    request = Req(restore=False)
    for value in (True,):
        request.kv_transfer_params = {disk_restore.RESTORE_MARKER_KEY: value}
        assert disk_restore.request_flagged(request) is True
    for value in ("true", "True", 1, 1.0, [True], {"y": True}, False, None, 0):
        request.kv_transfer_params = {disk_restore.RESTORE_MARKER_KEY: value}
        assert disk_restore.request_flagged(request) is False, repr(value)
    for params in (None, "yes", 7, [disk_restore.RESTORE_MARKER_KEY], object()):
        request.kv_transfer_params = params
        assert disk_restore.request_flagged(request) is False, repr(params)
    assert disk_restore.request_flagged(object()) is False
    request.kv_transfer_params = {disk_restore.RESTORE_MARKER_KEY: True,
                                  "other": 1}
    assert disk_restore.request_flagged(request) is True


def test_an_unflagged_request_touches_no_index_and_no_journal(armed, monkeypatch):
    """THE COST GUARANTEE: the fork's answer, and no lookup, no refresh, no alloc.

    The marker check is the first thing in the path so an unflagged request cannot pay
    for a journal refresh or an allocation. A scope switch placed AFTER the expensive
    work would be a scope switch in name only, so this asserts the ABSENCE of the work
    rather than the value that came back.
    """
    fill(armed, blocks=4)
    lookups, refreshes = [], []
    real_contains = disk_restore.disk_index.contains
    watching = {"on": True}

    def spy_contains(*a, **k):
        lookups.append(a)
        # Spy during the unflagged call, then let the real index answer so the
        # flagged half of this test fails only if the restore path really fails.
        return False if watching["on"] else real_contains(*a, **k)

    monkeypatch.setattr(disk_restore.disk_index, "contains", spy_contains)
    real_refresh = disk_restore._refresh_index

    def counted_refresh(*a, **k):
        refreshes.append(1)
        return real_refresh(*a, **k)

    monkeypatch.setattr(disk_restore, "_refresh_index", counted_refresh)
    assert offer(Manager(), Req(request_id="quiet", blocks=4,
                                restore=False)) == (0, False)
    assert lookups == [], "an unflagged request must not query the disk index"
    assert refreshes == [], "an unflagged request must not refresh the journal"
    counters = disk_restore.counters()
    assert counters["skipped_no_marker"] == 1
    assert counters["offers"] == 0
    assert counters["slots_allocated"] == 0
    # The flagged request on the same boot does the work, so the skip above was the
    # scope and not a broken fixture.
    watching["on"] = False
    asked = len(lookups)
    assert offer(Manager(), Req(request_id="loud", blocks=4)) == (4 * BLOCK, True)
    assert refreshes, "the flagged request must reach the journal"
    assert len(lookups) > asked, "the flagged request must query the index"


def test_scope_all_offers_without_a_marker(offering):
    fill(offering, blocks=4)
    assert offer(Manager(), Req(request_id="plain", blocks=4,
                                restore=False)) == (4 * BLOCK, True)
    assert disk_restore.counters()["skipped_no_marker"] == 0


def test_the_gate_verdict_outranks_the_scope_verdict(armed, monkeypatch):
    """(the scheduler half is fitted here, or `why_no_restore` answers a different
    question than the gate and this test would be about the wrong link)"""
    """A closed gate outranks an unflagged request in `why_no_restore`.

    Both are true of the same request, and the operator's question is "could this boot
    have served it at all". Answering "the request did not ask" while the gate is shut
    sends them chasing client payloads instead of the arm file.
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
    disk_restore.set_offer_switch(False)
    why = disk_restore.scheduler_section()["why_no_restore"]
    assert "not been told to offer" in why or "offers" in why, why


def test_the_posture_is_reported_wherever_it_can_be_read(armed):
    """Switch, scope, marker key, offer cap and the rank split - boot line + payload.

    A posture that lives only in a config file is a posture nobody finds at 3 a.m.
    """
    line = disk_restore.boot_line()
    for field in ("offers_switch=", "scope=optin",
                  f"marker={disk_restore.RESTORE_MARKER_KEY}",
                  "offer_cap=", "ranks_armed=", "suppressed=", "skipped_no_marker="):
        assert field in line, field
    posture = disk_restore.offer_posture()
    assert posture["enabled"] is True, "the `armed` fixture raises the switch"
    assert posture["scope"] == "optin"
    assert posture["marker_key"] == disk_restore.RESTORE_MARKER_KEY
    counters = disk_restore.counters()
    for key in ("offers_enabled", "scope", "flagged", "skipped_no_marker",
                "suppressed_hits", "suppressed_spans", "suppressed_requests",
                "ranks_seen", "no_read", "offer_clamped", "witness_kind"):
        assert key in counters, key
        assert key in counters["units"], key


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

def _rank(directory, rank, *, installed=True, armed=True, reads=0, reason=None):
    """One rank file, in the shape `worker_diag` writes."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"rank-{rank}.json").write_text(json.dumps({
        "version": 1, "epoch": "aabbccdd", "ts": round(time.time(), 6),
        "role": "worker", "rank": rank, "arena": {}, "counters": {},
        "writer": {"epoch_scope": "process"},
        "disk_restore": {"installed": installed, "armed": armed,
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

    def write(shared=None, **by_rank):
        shared = shared or {}
        for rank in RANKS:
            fields = dict(shared)
            fields.update(by_rank.get(rank) or {})
            _rank(diag, rank, **fields)
        disk_restore._proof_cache.update({"mono": None, "result": None})

    fill(str(root), blocks=4)
    return write


def test_armed_scheduler_with_disarmed_workers_offers_nothing(four_ranks):
    """THE FUEL TEST: scheduler armed, every rank refusing, zero external tokens.

    This is the one line of gate logic that would have turned 123,474 load failures, a
    wedged harness and a watchdog reset into a boot that did nothing at all.
    """
    four_ranks({"armed": False, "reason": "the disk tier is disarmed"})
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
    assert "scheduler_armed_workers_disarmed" in str(counters["last_refusal"]) or \
        "disarmed" in counters["last_reason"]

    # Arm the same four ranks: the veto was the entire story, nothing else moved.
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
    four_ranks(**{rank: {"armed": False, "reason": "disarmed"}
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

# --- the dev verb that raises the switch --------------------------------------

def test_the_offers_op_reads_the_posture_without_changing_it(armed):
    """`{"op":"offers"}` alone is a read: an operator checking the state must not
    change it, and every mutating op here is POST-only by the route's own rule."""
    disk_restore.set_offer_switch(False)
    before = disk_restore.offer_posture()
    out = disk_store.disk_for_engine(None, {"op": "offers"})
    assert out["ok"] is True
    assert out["posture"]["enabled"] == before["enabled"] is False
    assert out["posture"]["scope"] == before["scope"]


def test_the_offers_op_raises_the_switch_and_sets_the_scope(armed):
    out = disk_store.disk_for_engine(None, {"op": "offers", "enabled": True,
                                             "scope": "all"})
    assert out["ok"] is True, out
    posture = disk_store.disk_for_engine(None, {"op": "offers"})["posture"]
    assert posture["enabled"] is True and posture["scope"] == "all"
    assert disk_restore.gate()["ok"] is True
    # And back down: a boot can be made inert again without a restart.
    assert disk_store.disk_for_engine(None, {"op": "offers",
                                             "enabled": False})["ok"] is True
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"


def test_the_offers_op_refuses_a_value_that_is_not_a_boolean_or_a_scope(armed):
    """A typo must 400-ish, not silently do nothing: the operator has to know the
    switch did not move."""
    for junk in ("true", 1, "yes"):
        out = disk_store.disk_for_engine(None, {"op": "offers", "enabled": junk})
        assert out["ok"] is False, (junk, out)
    out = disk_store.disk_for_engine(None, {"op": "offers", "scope": "evryone"})
    assert out["ok"] is False and "optin" in out["reason"]


def test_the_offers_op_is_listed_and_post_only():
    """The route's GET/POST split is the guard on every mutating op, so a new op has
    to be in both lists or it is either unreachable or reachable by a page load."""
    assert "offers" in disk_store.DISK_OPS
    assert "offers" in disk_store.MUTATING_DISK_OPS
    schema = endpoint.ARG_SPECS[config.ROUTE_DISK]
    assert "offers" in schema["op"]["values"]
    assert schema["scope"]["values"] == ("optin", "all")
    assert schema["enabled"]["kind"] == "bool"


def test_the_verb_documentation_names_the_opt_in_key():
    """The key is only usable if the verb's own help says what to send.

    The opt-in lives in a client request body, not in this repo, so the one place an
    operator can read it is the route help - where a stale description is worse than
    none, because it will be trusted.
    """
    doc = endpoint.route_disk_post.__doc__ or ""
    assert disk_restore.RESTORE_MARKER_KEY in doc
    assert '"kv_transfer_params": {' in doc, doc
    # The deployed chat-completions request declares no `extra_args` field and ignores
    # unknown fields, so telling an operator to nest the marker there would produce a
    # silently un-flagged request. The help must send the top-level shape.
    assert '{"extra_args": {"kv_transfer_params"' not in doc
    readme = (pathlib.Path(__file__).resolve().parent.parent
              / "README.md").read_text()
    assert '{"kv_transfer_params": {"kvtier_disk_restore": true}}' in readme
