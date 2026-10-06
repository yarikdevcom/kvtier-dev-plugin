r"""The counters that make a boot's POSTURE and a closed gate visible from data.

Observability gaps from the 2026-10-05 restore experiment, all in `disk_restore`'s
scheduler-side view, all fixed without changing what the restore path DOES (its return
value is the fork's object, byte-identical, in every case here - which is what the last
group of tests pins).

CHANGE 2 - THE POSTURE WAS INVISIBLE. `scheduler_section()` published neither the boot's
switches nor the one-word verdict they collapse to, so answering "is this boot serving
every request from SSD, or only extending RAM shortfalls?" meant reading `boot_line` as
prose. `disk_authority`, `serving_as`, `authority_lookups` and `lookups` are published
here with units now, from the same one call an experiment already makes.

What change 2 used to be, and why it is gone: these were `flagged`, `requests_seen` and
`lookups`, published so that a CLIENT-SUPPLIED MARKER could be proved to have arrived
without a control request. The marker was removed in 0.2.31 - a request body must not
decide whether its KV comes off NVMe - so the three counters it fed went with it, and
the question the payload answers now is about the boot, which is the only thing that
decides posture.

CHANGE 3 - A CLOSED GATE LEFT NO TRACE. A request declined at the gate called `_note()`
(which writes `last_reason` and nothing else) and returned, while only
`offers_disabled` / `workers_not_servicing` / `no_worker_proof` were counted. So
`refused == {}` was ambiguous between "nothing refused us" and "the gate refused every
request of the boot silently", and that ambiguity cost the afternoon: an empty `refused`
was read as evidence that the ALIGNMENT guard had declined the request when the gate had
declined it. `not_configured`, `disarmed` and `tier_disarmed` are counted by name now,
in the same region that already counts the other three.

One consequence of removing the marker deserves its own sentence, because it reverses a
rule this file used to pin: with no per-request door, a boot with its switches down is a
counted REFUSAL naming the ops that open the gate, not a "skip". There is no longer a
category of request that "never asked" - every request is asked, and if the gate is shut
that is a fact about the boot worth a counter.

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
    """A request: an id, and nothing that decides anything.

    Deliberately no `kv_transfer_params`. The offer path has no reason to read a client
    field since 0.2.31, and a fake that cannot supply one turns a leftover read into an
    AttributeError instead of a passing test.
    """

    def __init__(self, request_id="req-1") -> None:
        self.request_id = request_id


FORK_ANSWER = (4096, True)


def offer(request, manager=None, computed=0, ram=FORK_ANSWER):
    return disk_restore.offer(manager if manager is not None else object(),
                               request, computed, ram)


def refused():
    return disk_restore.counters()["refused"]


def section():
    return disk_restore.scheduler_section()


# --- change 2: the boot's posture, published --------------------------------

@pytest.mark.parametrize("key", ["disk_authority", "offers_enabled", "serving_as",
                                 "authority_lookups", "lookups"])
def test_the_posture_fields_reach_the_payload_with_units(key):
    """`scheduler_section()` is the one-call payload an experiment reads, and a counter
    that lives only in `counters()` is a counter nobody sees. Units are not optional:
    the suite that walks the section fails on a bare number."""
    view = section()
    assert key in view, f"{key} is in counters() but not in the payload"
    assert key in view["units"], f"{key} reaches the payload with no unit"
    assert view["units"][key], f"{key} has an empty unit"
    # The unit must say which counters() it came from, the rule the neighbouring
    # pass-throughs follow: two views of one counter that cannot be traced to their
    # owner are two numbers waiting to disagree.
    assert "counters()" in view["units"][key] or "disk_posture()" in view["units"][key]


@pytest.mark.parametrize("retired", ["flagged", "requests_seen", "skipped_no_marker",
                                     "scope", "marker_key"])
def test_the_marker_fields_are_gone_from_the_payload(retired):
    """A removed feature that leaves its counters behind is a feature that was not
    removed, and a payload full of permanently-zero counters is a payload that lies: an
    operator reading `flagged: 0` on a boot that IS serving everyone from SSD would
    conclude the opposite of the truth.
    """
    assert retired not in section(), retired
    assert retired not in disk_restore.counters(), retired


def test_the_posture_needs_no_control_request_to_be_read():
    """ONE payload read answers "which switch is up", for both switches.

    The old proof was a pair of requests and a subtraction. The new proof is one field
    per switch plus `serving_as`, and the two switches are independently readable, so a
    boot running the cheaper posture cannot be mistaken for one running authority.
    """
    view = section()
    assert (view["disk_authority"], view["offers_enabled"], view["serving_as"]) == (
        False, False, "off")
    disk_restore.set_disk_authority(True)
    view = section()
    assert (view["disk_authority"], view["offers_enabled"], view["serving_as"]) == (
        True, False, "disk_authority")
    disk_restore.set_offer_switch(True)
    view = section()
    assert view["serving_as"] == "disk_authority", \
        "both up is authority, and the payload must not make the reader guess"
    disk_restore.set_disk_authority(False)
    view = section()
    assert view["serving_as"] == "ram_first_gap_fill"
    # The change counters are the audit trail: two moves in, one more each.
    assert view["disk_authority_changes"] == 2
    assert view["offer_switch_changes"] == 1


def test_authority_lookups_and_lookups_measure_different_things():
    """`lookups` is the path's entry count; `authority_lookups` is a POSTURE's count.

    With the store unconfigured every lookup is turned away at the gate, so `lookups`
    rises under both switches while `authority_lookups` stays at zero: the authority
    line is drawn only where an offer is actually decided, and a counter that moved on a
    refused lookup would count intentions instead of work. (The pair moving TOGETHER on
    a boot that can serve is pinned in `test_disk_restore_authority.py`, where the store
    exists.)
    """
    disk_restore.set_offer_switch(True)
    for _ in range(2):
        offer(Req())
    view = section()
    assert view["lookups"] == 2
    assert view["authority_lookups"] == 0
    disk_restore.set_offer_switch(False)
    disk_restore.set_disk_authority(True)
    for _ in range(3):
        offer(Req())
    view = section()
    assert view["lookups"] == 5
    assert view["authority_lookups"] == 0, (
        "the gate refused every one of these, so no lookup ever ran under authority")
    assert view["refused"]["not_configured"] == 5


def test_lookups_counts_the_path_entry_before_any_gate():
    """`lookups` is the offer path's own entry count, so a `lookups` that rises while
    `offers` sits at zero is the refusal vocabulary's job to explain - which needs
    the refusals counted too (change 3). The pair is the diagnostic.
    """
    for _ in range(3):
        offer(Req())
    view = section()
    assert view["lookups"] == 3
    assert view["offers"] == 0
    assert view["refused"]["not_configured"] == 3


# --- change 3: a closed gate, counted by name -------------------------------

def test_a_gate_closed_request_is_counted_as_what_the_gate_said():
    """The whole point: `refused` tells the truth on its own now.

    This is the state the experiment ran in - no store root configured for the restore
    path's own process - and the state whose empty `refused` was mistaken for "nothing
    refused us".
    """
    verdict = disk_restore.gate()
    assert verdict["ok"] is False and verdict["reason_kind"] == "not_configured"
    result = offer(Req())
    assert refused()["not_configured"] == 1
    assert result == FORK_ANSWER
    assert disk_restore.counters()["refusals_total"] == 1


def test_a_boot_with_no_switch_up_refuses_by_name_and_names_the_way_out(monkeypatch,
                                                                        tmp_path):
    """Both switches down is `offers_disabled`, counted, with the two ops in the text.

    Since the marker is gone there is no request that "did not ask", so the closed gate
    is the only no-op left on the path and it must never be silent: the counter is the
    number of lookups the boot turned away, and the reason string names both ops that
    would open it. An operator who deploys without POSTing anything gets a log and a
    counter that both say what to do next.

    The store root and both arms are set here because the gate reads them BEFORE the
    switches: without them the honest answer is `not_configured`, a different refusal
    with its own bucket and its own test above.
    """
    root = tmp_path / "spill"
    root.mkdir()
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    assert disk_restore.gate()["reason_kind"] == "offers_disabled"
    result = offer(Req())
    counters = disk_restore.counters()
    assert result is FORK_ANSWER
    assert counters["refused"]["offers_disabled"] == 1
    assert counters["refusals_total"] == 1
    assert counters["serving_as"] == "off"
    reason = disk_restore.OFFER_REFUSAL_WHY["offers_disabled"]
    assert '"op":"offers"' in reason and '"op":"authority"' in reason


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
    offer(Req())
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
        offer(Req())
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
    offer(Req())
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
    result = offer(Req(), ram=ram)
    assert result is ram
    assert refused()["disarmed"] == 1

    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(backport, "is_armed", lambda: False)
    result = offer(Req(), ram=ram)
    assert result is ram
    assert refused()["tier_disarmed"] == 1

    monkeypatch.setattr(disk_spill, "disk_root", lambda: None)
    result = offer(Req(), ram=ram)
    assert result is ram
    assert refused()["not_configured"] == 1


def test_the_gate_function_itself_is_unchanged(monkeypatch):
    """Change 3 counts what `gate()` already said. `gate()`'s own verdicts, order and
    reasons must be untouched - the switch rule is pinned in its own suite, and a
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
    # THE GATE'S OWN ORDER: the boot switches are read BEFORE the worker proof, so a
    # boot that has never been told to serve does not look for proof at all. Unchanged
    # by change 3 - and reached here by raising a switch, as an operator would.
    disk_restore.set_disk_authority(True)
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": False, "count": 0, "ranks": [],
                                             "reason": "no worker proved itself"})
    assert disk_restore.gate()["reason_kind"] == "no_worker_proof"
    # Counting the gate's answer never called the gate on the caller's behalf.
    assert disk_restore.counters()["refusals_total"] == 0
