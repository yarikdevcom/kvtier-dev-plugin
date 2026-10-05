r"""The counters that make a marked request, and a closed gate, visible from data.

Two observability gaps from the 2026-10-05 restore experiment, both in
`disk_restore`'s scheduler-side view, both fixed without changing what the restore path
DOES (its return value is the fork's object, byte-identical, in every case here - which
is what the last group of tests pins).

CHANGE 2 - THE MARKER WAS INVISIBLE. `flagged`, `requests_seen` and `lookups` existed in
`counters()` with units and were never published by `scheduler_section()`, so proving a
marker had survived the engine's own field-folding required a CONTROL EXPERIMENT: send a
paired unflagged request and watch `skipped_no_marker` move for the control and not for
the marked one. Genuinely clever, and it costs a second request through a live engine to
learn one bit. Publishing the three makes a marked request self-evident.

CHANGE 3 - A CLOSED GATE LEFT NO TRACE. A marked request declined at the gate called
`_note()` (which writes `last_reason` and nothing else) and returned, while only
`offers_disabled` / `workers_not_servicing` / `no_worker_proof` were counted. So
`refused == {}` was ambiguous between "nothing refused us" and "the gate refused every
request of the boot silently", and that ambiguity cost the afternoon: an empty `refused`
was read as evidence that the ALIGNMENT guard had declined the request when the gate had
declined it. `not_configured`, `disarmed` and `tier_disarmed` are counted by name now,
in the same region that already counts the other three.

`last_reason` behaviour is unchanged on purpose: something may be parsing it, and the
brief's rule is that the restore path's behaviour must not move - only its bookkeeping.
"""

from __future__ import annotations

import pytest

from kvtier_dev_plugin import (
    backport,
    disk_index,
    disk_layout,
    disk_read,
    disk_restore,
    disk_spill,
)


@pytest.fixture(autouse=True)
def fresh():
    """A fresh boot per test: no ledgers, no counters, no installed layer.

    The same fixture `test_disk_restore.py` uses, for the same reason: `disk_restore`
    keeps process-wide state that conftest deliberately does not touch (installed
    wrappers on classes), so a suite that reads its counters must reset them itself
    or inherit the previous file's.
    """
    disk_restore.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    disk_read.reset_state_for_tests()
    yield
    disk_restore.uninstall()
    disk_restore.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    disk_read.reset_state_for_tests()


@pytest.fixture(autouse=True)
def allow_tmp_root(monkeypatch):
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())


class Req:
    """A request, flagged or not, the way the engine's own field carries it.

    `kv_transfer_params.kvtier_disk_restore` must be the JSON boolean `True`; anything
    else (including `"true"` and `1`) is "not asked", which is `test_the_marker_is_the_
    engines_own_field_and_strictly_a_boolean` in `test_disk_restore.py`'s business.
    """

    def __init__(self, flagged=True, request_id="req-1") -> None:
        self.kv_transfer_params = ({disk_restore.RESTORE_MARKER_KEY: True}
                                   if flagged else {"other": 1})
        self.request_id = request_id


FORK_ANSWER = (4096, True)


def offer(request, manager=None, computed=0, ram=FORK_ANSWER):
    return disk_restore.offer(manager if manager is not None else object(),
                               request, computed, ram)


def refused():
    return disk_restore.counters()["refused"]


def section():
    return disk_restore.scheduler_section()


# --- change 2: the marker's own evidence, published -------------------------

@pytest.mark.parametrize("key", ["flagged", "requests_seen", "lookups"])
def test_the_three_marker_counters_reach_the_payload_with_units(key):
    """`scheduler_section()` is the one-call payload an experiment reads, and a counter
    that lives only in `counters()` is a counter nobody sees. Units are not optional:
    the suite that walks the section fails on a bare number."""
    view = section()
    assert key in view, f"{key} is in counters() but not in the payload"
    assert isinstance(view[key], int)
    assert key in view["units"], f"{key} reaches the payload with no unit"
    assert view["units"][key], f"{key} has an empty unit"
    # The unit must say which counters() it came from, the rule the neighbouring
    # pass-throughs follow: two views of one counter that cannot be traced to their
    # owner are two numbers waiting to disagree.
    assert "counters()" in view["units"][key]


def test_a_marked_request_is_now_self_evident_without_a_control_request():
    """THE WORKAROUND, retired. One marked and one unflagged lookup, in one payload:
    `requests_seen` moves for both, `flagged` moves for one, `skipped_no_marker` for the
    other. Before, the only way to see that the marker arrived was to send the
    unflagged one as a control and subtract.
    """
    disk_restore.set_offer_switch(True)
    disk_restore.set_scope(disk_restore.RESTORE_SCOPE_OPTIN)
    before = section()
    offer(Req(flagged=False))
    middle = section()
    assert middle["requests_seen"] == before["requests_seen"] + 1
    assert middle["skipped_no_marker"] == before["skipped_no_marker"] + 1
    assert middle["flagged"] == before["flagged"], "an unflagged request is not flagged"

    offer(Req(flagged=True))
    after = section()
    assert after["requests_seen"] == middle["requests_seen"] + 1
    assert after["flagged"] == middle["flagged"] + 1
    assert after["skipped_no_marker"] == middle["skipped_no_marker"]
    assert after["lookups"] == middle["lookups"] + 1


def test_lookups_counts_the_path_entry_before_any_gate_or_scope():
    """`lookups` is the offer path's own entry count, so a `lookups` that rises while
    `offers` sits at zero is the refusal vocabulary's job to explain - which needs
    the refusals counted too (change 3). The pair is the diagnostic.
    """
    for _ in range(3):
        offer(Req(flagged=True))
    view = section()
    assert view["lookups"] == 3
    assert view["requests_seen"] == 3
    assert view["offers"] == 0
    assert view["refused"]["not_configured"] == 3


# --- change 3: a closed gate, counted by name -------------------------------

def test_a_gate_closed_marked_request_is_counted_as_what_the_gate_said():
    """The whole point: `refused` tells the truth on its own now.

    This is the state the experiment ran in - no store root configured for the restore
    path's own process - and the state whose empty `refused` was mistaken for "nothing
    refused us".
    """
    verdict = disk_restore.gate()
    assert verdict["ok"] is False and verdict["reason_kind"] == "not_configured"
    result = offer(Req(flagged=True))
    assert refused()["not_configured"] == 1
    assert result == FORK_ANSWER
    assert disk_restore.counters()["refusals_total"] == 1


@pytest.mark.parametrize("switch,expected", [
    # root configured, spill tier disarmed: the common resting state of a boot
    ("disarmed", "disarmed"),
    # spill armed, the RAM tier not: an offer has nothing to ride on
    ("tier_disarmed", "tier_disarmed"),
])
def test_each_closed_switch_has_its_own_bucket(monkeypatch, tmp_path, switch,
                                               expected):
    """Every switch the gate reads gets its OWN counter name, matching the `reason_kind`
    the gate itself publishes. A single "gate_closed" bucket would put me back to
    reading a sentence to find out which switch shut - the reason there are three."""
    root = tmp_path / "spill"
    root.mkdir()
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: switch != "disarmed")
    monkeypatch.setattr(backport, "is_armed", lambda: switch != "tier_disarmed")
    verdict = disk_restore.gate()
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == expected
    offer(Req(flagged=True))
    assert refused()[expected] == 1
    other = "disarmed" if expected != "disarmed" else "tier_disarmed"
    assert refused()[other] == 0, "the buckets must not borrow each other's name"
    # `bad_call` is `_refuse`'s fallback for an unknown name: a non-zero here would mean
    # the gate's reason_kind was silently filed under someone else's counter.
    assert refused()["bad_call"] == 0


def test_the_refused_dict_alone_now_names_the_gate_as_the_cause():
    """A boot whose every decline was the gate no longer reads as `refused: {}`.

    The exact misreading from 2026-10-05: an empty `refused` beside an alignment-guard
    `last_reason` was taken as "the alignment guard refused me". Now the payload's own
    `refused` says which guard refused, and the guard vocabulary is closed.
    """
    for _ in range(2):
        offer(Req(flagged=True))
    view = section()
    # `refused` publishes the WHOLE closed vocabulary (every bucket, zeros included), so
    # the readable form is the non-zero one - and it is non-zero now.
    assert {name: count for name, count in view["refused"].items() if count} == {
        "not_configured": 2}
    # ...and the vocabulary that explains it is published beside it, so the number can
    # be read without opening this file.
    assert "not_configured" in view["refusal_reasons"]
    assert "gate" in view and view["gate"]["ok"] is False
    assert view["gate"]["reason_kind"] == "not_configured"


def test_last_reason_is_still_written_exactly_as_before():
    """`_refuse` sets `last_reason` from the same sentence `_note` did, so a consumer
    parsing the old field sees the same text. The new counters are additive.
    """
    reason = disk_restore.gate()["reason"]
    offer(Req(flagged=True))
    counters = disk_restore.counters()
    assert counters["last_reason"] == reason[:160]
    assert counters["last_refusal"].startswith("not_configured: ")


# --- the behaviour that must NOT move ---------------------------------------

def test_the_forks_answer_comes_back_as_the_same_object(monkeypatch, tmp_path):
    """THE INVARIANT. Every refusal in this region returns the fork's tuple - the same
    OBJECT, not an equal copy - because "no plugin at all" and "a plugin that is
    refusing" must be indistinguishable to the scheduler.

    Counting on the way out must not consume, rebuild or replace it: an offer path that
    returns `(0, False)` instead of the fork's `(4096, True)` would silently delete a
    host-RAM hit, and the counter would still look perfect.
    """
    root = tmp_path / "spill"
    root.mkdir()
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: False)
    ram = (4096, True)
    result = offer(Req(flagged=True), ram=ram)
    assert result is ram
    assert refused()["disarmed"] == 1

    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    result = offer(Req(flagged=True), ram=ram)
    assert result is ram
    assert refused()["tier_disarmed"] == 1

    monkeypatch.setattr(disk_spill, "disk_root", lambda: None)
    result = offer(Req(flagged=True), ram=ram)
    assert result is ram
    assert refused()["not_configured"] == 1


def test_an_unflagged_request_is_still_a_skip_and_never_a_refusal(monkeypatch):
    """The rule from `test_disk_restore.py`, re-pinned here because change 3 touched the
    very region that decides it: in `optin` scope an unflagged request never reaches the
    gate, so it must not move any refusal bucket - a skip says "it did not ask", a
    refusal says "we could not".
    """
    disk_restore.set_offer_switch(True)
    disk_restore.set_scope(disk_restore.RESTORE_SCOPE_OPTIN)
    result = offer(Req(flagged=False))
    counters = disk_restore.counters()
    assert result is FORK_ANSWER
    assert counters["refusals_total"] == 0, \
        "an unflagged request is a skip, not a refusal"
    assert not [name for name, count in counters["refused"].items() if count], \
        "every refusal bucket stays at zero for a request that never asked"
    assert counters["skipped_no_marker"] == 1
    assert counters["flagged"] == 0


def test_a_request_in_all_scope_is_counted_at_the_gate_like_any_other(monkeypatch):
    """`scope=all` means every request is asked, so every one of them is now visible at
    the gate. The counters say so without a control request in either scope.
    """
    disk_restore.set_offer_switch(True)
    disk_restore.set_scope(disk_restore.RESTORE_SCOPE_ALL)
    offer(Req(flagged=False))
    view = section()
    assert view["requests_seen"] == 1
    assert view["skipped_no_marker"] == 0, "scope=all asks on everyone's behalf"
    assert view["refused"]["not_configured"] == 1


def test_the_gate_function_itself_is_unchanged(monkeypatch):
    """Change 3 counts what `gate()` already said. `gate()`'s own verdicts, order and
    reasons must be untouched - the four-switch rule is pinned in its own suite, and a
    counter that changed which switch answers first would be a behaviour change wearing
    an observability costume.
    """
    assert disk_restore.gate()["reason_kind"] == "not_configured"
    monkeypatch.setattr(disk_spill, "disk_root", lambda: "/tmp/whatever")
    monkeypatch.setattr(disk_spill, "is_armed", lambda: False)
    assert disk_restore.gate()["reason_kind"] == "disarmed"
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    assert disk_restore.gate()["reason_kind"] == "tier_disarmed"
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    # THE GATE'S OWN ORDER: the offer switch is read BEFORE the worker proof, so a
    # boot that has never been told to offer does not look for proof at all. Unchanged
    # by change 3 - and reached here by raising the switch, as an operator would.
    disk_restore.set_offer_switch(True)
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": False, "count": 0, "ranks": [],
                                             "reason": "no worker proved itself"})
    assert disk_restore.gate()["reason_kind"] == "no_worker_proof"
    # Counting the gate's answer never called the gate on the caller's behalf.
    assert disk_restore.counters()["refusals_total"] == 0
