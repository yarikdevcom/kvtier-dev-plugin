r"""The LOOKUP LEDGER: the offer/adopt pair, joined per request id, bounded.

The live engine logs "update_state_after_alloc called with no external tokens but
get_num_new_matched_tokens() unexpectedly recorded a pending CPU hit". The fork
pins a CPU hit ONLY when `hit_length > 0` (manager.py:363-374), so the connector
really did return a positive number - and the scheduler really did pass 0. The
ledger is the instrument that turns that into data: `local` as passed in, the hit
returned, and the `num_external_tokens` the scheduler later passed.

What this file is obliged to prove, in order of how much it would cost to learn
it in production instead:

1. BOUNDS. A hard cap, a monotonic `dropped`, and no growth while armed. This
   runs on the serving loop.
2. THE JOIN. Requests interleave, get skipped, and adopt without ever having
   looked up (scheduler.py:1046-1060 does exactly that to a WAITING request with
   num_computed_tokens > 0). Matching by call ORDER would be wrong on the first
   concurrent batch.
3. COMPOSITION. The ledger owns NO method of its own: it reads through the guard
   wrapper, so it must work over a fork method, over a backport-replaced method,
   and over a re-install - without nesting a second wrapper and without dropping
   another owner's layer. The assertions walk the MARKER CHAIN, because "it did
   not crash" is not the property.
4. CONTAINMENT. A fault inside the fork must propagate exactly as it would
   without this plugin (the ledger is not guard mode), and a fault inside the
   ledger's own bookkeeping must never reach the scheduler or corrupt the ring.
"""

from __future__ import annotations

import sys
import types

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import (
    backport,
    config,
    guard_mode,
    lookup_ledger,
    snapshot,
    trace_ring,
)

# --- the fake tier -----------------------------------------------------------
#
# Exec'd into a module NAMED like the fork's (fakes_debug.fork_class), because
# `guard_mode.install` refuses a function that is neither ours nor the fork's -
# the very check that stops this plugin overwriting a third party.

LEDGER_TIER_SOURCE = '''
FAILING = set()


class SimpleCPUOffloadScheduler:
    """The fork's geometry constants for the live box (3200 / 800)."""

    block_size = 3200
    hash_block_size = 800

    def __init__(self):
        self.calls = []
        self.offered = {}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.calls.append(("offer", request.request_id, num_computed_tokens))
        if "get_num_new_matched_tokens" in FAILING:
            raise RuntimeError("the fork faulted on the lookup")
        return self.offered.get(request.request_id, (0, False))

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append(("adopt", request.request_id, num_external_tokens))
        if "update_state_after_alloc" in FAILING:
            raise RuntimeError("the fork faulted on the adopt")
        return None
'''


@pytest.fixture(autouse=True)
def clean_ledger():
    """The ledger's ring and counters are process-wide. A ledger-only test does not
    ask for `reset_dev_state`, so reset here as well: without it `dropped` from one
    test is the next test's starting point and every bound assertion is meaningless.
    """
    lookup_ledger.reset_state_for_tests()
    yield
    lookup_ledger.reset_state_for_tests()


@pytest.fixture
def tier(reset_dev_state):
    """The fork-shaped class, its module (with the failure switch), an instance."""
    cls, module = fake.fork_class(guard_mode.TARGET_MODULE,
                                  guard_mode.TARGET_CLASS,
                                  LEDGER_TIER_SOURCE, register=True)
    manager = cls()
    module.FAILING.clear()
    yield cls, module, manager
    sys.modules.pop(guard_mode.TARGET_MODULE, None)


@pytest.fixture
def engine(tier):
    _cls, _module, manager = tier
    scheduler = types.SimpleNamespace(connector=types.SimpleNamespace(
        scheduler_manager=manager))
    return fake.engine(scheduler)


def arm(cap=None):
    """Arm the ledger the way the verb does (install included, engine optional)."""
    return lookup_ledger.lookups_for_engine(None, _with_cap("arm", cap))


def _with_cap(op, cap):
    args = {"op": op}
    if cap is not None:
        args["cap"] = cap
    return args


def request(request_id):
    return types.SimpleNamespace(request_id=request_id)


def records():
    return lookup_ledger.read(limit=256)["entries"]


def find(req_id):
    for record in records():
        if record["req"] == req_id:
            return record
    return None


def counts():
    return lookup_ledger.verdict()["counts"]


# --- 1. bounds ---------------------------------------------------------------

def test_the_ring_is_preallocated_and_the_cap_is_hard():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    ring_before = lookup_ledger._state["ring"]
    assert len(ring_before) == 16
    for index in range(60):
        lookup_ledger.record_lookup(f"req-{index}", 3200, 800, False, 3200, 800)
    state = lookup_ledger._state
    assert state["ring"] is ring_before, "the ring was reallocated after arming"
    assert len(state["ring"]) == 16, "the ring grew past the cap"
    assert lookup_ledger.counters()["live"] == 16
    # 60 written, 16 readable, 44 lost: nothing else could have eaten them.
    assert lookup_ledger.counters()["records"] == 60
    assert lookup_ledger.counters()["dropped"] == 44
    assert lookup_ledger.counters()["dropped_unjoined"] == 44


def test_dropped_is_monotonic_and_survives_a_clear():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    for index in range(20):
        lookup_ledger.record_lookup(f"r{index}", 0, 0, False, 3200, 800)
    after_wrap = lookup_ledger.counters()["dropped"]
    assert after_wrap == 4
    lookup_ledger.lookups_for_engine(None, {"op": "clear"})
    assert lookup_ledger.counters()["dropped"] == after_wrap
    for index in range(20):
        lookup_ledger.record_lookup(f"q{index}", 0, 0, False, 3200, 800)
    assert lookup_ledger.counters()["dropped"] > after_wrap


def test_the_pending_index_cannot_outgrow_the_cap():
    """The join index is bounded BY the ring: one entry per live record, and an
    evicted record's entry goes with it. Checked directly because an unbounded
    dict on the serving loop is the bug this whole module exists to not have."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    for index in range(200):
        lookup_ledger.record_lookup(f"req-{index}", 3200, 800, False, 3200, 800)
    assert len(lookup_ledger._pending) <= 16
    assert lookup_ledger.counters()["pending_now"] == len(lookup_ledger._pending)
    # And the entries that survived are the NEWEST 16, each pointing at its slot.
    assert max(int(key.split("-")[1]) for key in lookup_ledger._pending) == 199


def test_a_disarmed_ledger_records_nothing_and_touches_nothing():
    """Not "cheap": inert. It must not even evaluate the arguments it is handed,
    so a disarmed ledger cannot be the reason a serving loop allocation changed."""
    lookup_ledger.record_from_call(lookup_ledger.METHOD_LOOKUP,
                                   (request("a"), 3200), (800, True), None)
    lookup_ledger.record_from_call(lookup_ledger.METHOD_ADOPT,
                                   (request("a"), None, 800), None, None)
    counters = lookup_ledger.counters()
    assert counters["armed"] is False
    assert counters["records"] == 0
    assert counters["lookups"] == 0 and counters["adopts"] == 0
    assert lookup_ledger._state["ring"] is None, "disarmed armed the ring"
    assert counters["record_errors"] == 0


def test_disarming_stops_recording_but_keeps_what_was_recorded():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_lookup("a", 3200, 800, False, 3200, 800)
    lookup_ledger.lookups_for_engine(None, {"op": "disarm"})
    lookup_ledger.record_lookup("b", 3200, 800, False, 3200, 800)
    assert find("a") is not None
    assert find("b") is None
    assert lookup_ledger.counters()["lookups"] == 1


def test_clear_forgets_contents_but_neither_the_sequence_nor_the_arming():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_lookup("a", 3200, 800, False, 3200, 800)
    seq = lookup_ledger.counters()["seq"]
    payload = lookup_ledger.lookups_for_engine(None, {"op": "clear"})
    assert payload["cleared"] is True and payload["armed"] is True
    assert lookup_ledger.counters()["live"] == 0
    assert lookup_ledger._pending == {}, "a cleared ring kept joinable lookups"
    assert lookup_ledger.counters()["seq"] == seq, "the reader's cursor moved"
    lookup_ledger.record_lookup("b", 0, 0, False, 3200, 800)
    assert records()[0]["seq"] >= seq, "seq restarted after a clear"


# --- 2. the join -------------------------------------------------------------

def test_interleaved_lookups_and_adopts_join_by_id_not_by_call_order():
    """Three requests offered, then adopted in a DIFFERENT order, with a lookup
    in between: a stack or a queue would attribute the wrong number to someone."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_lookup("a", 32000, 800, True, 3200, 800)
    lookup_ledger.record_lookup("b", 32000, 20000, True, 3200, 800)
    lookup_ledger.record_lookup("c", 32000, 0, False, 3200, 800)
    lookup_ledger.record_adopt("b", 20000, 3200)
    lookup_ledger.record_lookup("d", 3200, 800, True, 3200, 800)
    lookup_ledger.record_adopt("a", 0, 3200)
    assert find("a")["adopted"] == 0
    assert find("b")["adopted"] == 20000
    assert find("c")["adopted"] is None
    assert find("d")["adopted"] is None
    assert counts()["dropped_to_zero"] == 1
    assert counts()["adopted_full"] == 1
    assert counts()["lookup_without_adopt"] == 2


def test_an_adopt_with_no_prior_lookup_is_recorded_and_counted():
    """Not an anomaly: scheduler.py:1046-1060 calls update_state_after_alloc with
    0 for a WAITING request that never did a lookup this pass."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_adopt("resumed", 0, 3200)
    record = find("resumed")
    assert record["kind"] == "adopt_only"
    assert record["adopted"] == 0
    assert record["hit_returned"] is None
    assert counts()["adopt_without_lookup"] == 1


def test_a_lookup_whose_request_is_skipped_stays_unjoined():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_lookup("skipped", 32000, 800, True, 3200, 800)
    assert find("skipped")["adopted"] is None
    assert counts()["lookup_without_adopt"] == 1
    assert lookup_ledger.counters()["pending_now"] == 1


def test_a_zero_hit_lookup_is_recorded_as_no_host_copy():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_lookup("cold", 32000, 0, False, 3200, 800)
    lookup_ledger.record_adopt("cold", 0, 3200)
    assert counts()["no_host_copy"] == 1
    assert counts()["dropped_to_zero"] == 0


def test_a_connector_that_refused_to_answer_is_its_own_bucket():
    """(None, False) makes the scheduler SKIP the request (scheduler.py:982-988):
    no adopt follows, and it is NOT the drop we are hunting."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_lookup("noidea", 32000, None, False, 3200, 800)
    assert counts()["no_verdict_from_connector"] == 1
    assert counts()["dropped_to_zero"] == 0
    assert counts()["no_host_copy"] == 0


def test_a_second_lookup_for_the_same_request_pairs_with_the_latest_one():
    """The fork releases a stale pin and re-offers on a retry after a failed
    allocate_slots (manager.py:326-329). The adopt belongs to the newest offer."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 16})
    lookup_ledger.record_lookup("retry", 32000, 20000, True, 3200, 800)
    lookup_ledger.record_lookup("retry", 32000, 800, True, 3200, 800)
    lookup_ledger.record_adopt("retry", 800, 3200)
    assert lookup_ledger.counters()["relookups"] == 1
    assert lookup_ledger.counters()["joins"] == 1
    the_new_one = [item for item in records()
                   if item["req"] == "retry" and item["adopted"] is not None]
    assert len(the_new_one) == 1
    assert the_new_one[0]["hit_returned"] == 800
    # The first offer never got an adopt: it is a lookup_without_adopt, not a drop
    assert counts()["lookup_without_adopt"] == 1


def test_a_recycled_slot_takes_its_pending_entry_with_it():
    """Write past the cap: the evicted lookup can never be joined, so its entry in
    the join index must go with it - otherwise a later adopt for that id would
    stamp a record that is no longer its own."""
    cap = lookup_ledger.MIN_CAP
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": cap})
    for index in range(cap):
        lookup_ledger.record_lookup(f"r{index}", 32000, 800, True, 3200, 800)
    assert len(lookup_ledger._pending) == cap
    lookup_ledger.record_lookup("extra", 32000, 800, True, 3200, 800)
    assert len(lookup_ledger._pending) == cap, "the join index grew past the cap"
    assert "r0" not in lookup_ledger._pending, "a dead lookup stayed joinable"
    assert lookup_ledger.counters()["dropped_unjoined"] == 1
    lookup_ledger.record_adopt("r0", 800, 3200)
    assert counts()["adopt_without_lookup"] == 1
    assert counts()["adopted_full"] == 0, "a stale adopt stamped someone else"
    lookup_ledger.record_adopt("extra", 800, 3200)
    assert counts()["adopted_full"] == 1


def test_a_join_is_refused_rather_than_fabricated_when_the_slot_moved():
    """The narrower case: the pending entry survives while the slot it points at
    is overwritten by a DIFFERENT record with the SAME id. Damaged on purpose."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": lookup_ledger.MIN_CAP})
    lookup_ledger.record_lookup("x", 32000, 800, True, 3200, 800)
    slot, seq = lookup_ledger._pending["x"]
    lookup_ledger._state["ring"][slot] = {"seq": seq + 9, "req": "x",
                                          "kind": "lookup", "local": 0,
                                          "hit_returned": 0,
                                          "load_kv_async": False, "adopted": None,
                                          "local_mod_block": 0, "block_size": 3200,
                                          "hash_block_size": 800,
                                          "mono_ns": 0, "_slot": slot}
    lookup_ledger.record_adopt("x", 800, 3200)
    assert lookup_ledger.counters()["join_refusals"] == 1
    assert counts()["adopt_without_lookup"] == 1


# --- 3. the verdict ----------------------------------------------------------

def test_the_verdict_names_the_drop_and_carries_the_partial_tail_evidence():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 32})
    # The case that is actually being hunted: a hit that fits in one block, so the
    # partial-tail rule CAN explain it (a tail at least as long as the hit).
    lookup_ledger.record_lookup("tail-suspect", 32000, 800, True, 3200, 800)
    lookup_ledger.record_adopt("tail-suspect", 0, 3200)
    # A hit bigger than any possible tail: the tail rule CANNOT produce this.
    lookup_ledger.record_lookup("impossible", 32000, 20000, True, 3200, 800)
    lookup_ledger.record_adopt("impossible", 0, 3200)
    # Control cases.
    lookup_ledger.record_lookup("adopted", 32000, 20000, True, 3200, 800)
    lookup_ledger.record_adopt("adopted", 20000, 3200)
    lookup_ledger.record_lookup("partial", 32000, 20000, True, 3200, 800)
    lookup_ledger.record_adopt("partial", 800, 3200)
    lookup_ledger.record_lookup("empty", 32000, 0, False, 3200, 800)
    lookup_ledger.record_adopt("empty", 0, 3200)

    verdict = lookup_ledger.verdict()
    assert verdict["counts"]["dropped_to_zero"] == 2
    assert verdict["counts"]["adopted_full"] == 1
    assert verdict["counts"]["adopted_partial"] == 1
    assert verdict["counts"]["no_host_copy"] == 1
    by_req = {item["req"]: item for item in verdict["dropped_examples"]}
    suspect = by_req["tail-suspect"]
    assert suspect["local_as_passed"] == 32000
    # The value the connector is handed is ALREADY block-aligned, so this is 0 by
    # construction - which is exactly why the tail has to be inferred.
    assert suspect["local_mod_block"] == 0
    assert suspect["partial_tail_observed"] is None
    assert suspect["min_partial_tail_the_rule_needs"] == 800
    assert suspect["implied_true_local_range"] == [32800, 35199]
    assert suspect["tail_rule_can_explain"] is True
    assert suspect["hit_mod_hash_block"] == 0
    impossible = by_req["impossible"]
    assert impossible["tail_rule_can_explain"] is False
    assert impossible["implied_true_local_range"] is None, (
        "an empty implied range must be null, never an inverted interval")
    assert "partial_tail" in verdict["rule_under_test"]


def test_a_dropped_record_without_a_block_size_says_unknown_rather_than_guessing():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 8})
    lookup_ledger.record_lookup("noblock", 32000, 800, True, None, None)
    lookup_ledger.record_adopt("noblock", 0)
    example = lookup_ledger.verdict()["dropped_examples"][0]
    assert example["block_size"] is None
    assert example["tail_rule_can_explain"] is None
    assert example["implied_true_local_range"] is None


def test_the_verdict_lists_what_it_cannot_see():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 8})
    verdict = lookup_ledger.verdict()
    joined = " ".join(verdict["not_covered"])
    assert "partial_tail" in joined
    assert "disarmed" in joined
    assert "guard-DECLINED" in joined or "guard" in joined
    assert verdict["records_live"] == 0
    assert all(sum(verdict["counts"].values()) == verdict["records_scanned"]
               for _ in [0]), "every record must land in exactly one bucket"


def test_a_record_with_adopted_greater_than_offered_gets_its_own_bucket():
    """Impossible under the fork's rules; its own bucket so it is never folded
    into "adopted_partial" and quietly read as a token-budget trim."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 8})
    lookup_ledger.record_lookup("weird", 32000, 800, True, 3200, 800)
    lookup_ledger.record_adopt("weird", 1600, 3200)
    assert counts()["adopted_more_than_offered"] == 1


# --- 4. composition with the guard and the backport --------------------------

def test_the_wrapper_records_both_halves_of_a_real_call(tier):
    """The end-to-end proof of where the hook sits: call the WRAPPED tier methods
    with guard mode OFF and nothing else armed, and both halves of the pair appear
    - joined, with the scheduler's own inputs. No test-side call into the ledger.
    """
    cls, _module, manager = tier
    armed = lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    assert armed["install"]["installed"] is True
    assert guard_mode.is_armed() is False

    manager.offered["r1"] = (20000, True)
    assert manager.get_num_new_matched_tokens(request("r1"), 32000) == (20000, True)
    assert manager.update_state_after_alloc(request("r1"), None, 20000) is None

    joined = find("r1")
    assert joined["local"] == 32000, "the scheduler's input was not recorded"
    assert joined["hit_returned"] == 20000
    assert joined["load_kv_async"] is True
    assert joined["adopted"] == 20000
    assert joined["local_mod_block"] == 0
    assert joined["block_size"] == 3200
    assert joined["hash_block_size"] == 800
    assert counts()["adopted_full"] == 1
    # The fork's own body still ran: the recorder is a passenger, not a gate.
    assert manager.calls == [("offer", "r1", 32000), ("adopt", "r1", 20000)]


def test_a_pre_existing_instance_is_recorded_after_the_wrappers_landed(tier, engine):
    """The construct-first trap: the instance was built long before the install.
    A class attribute resolves per call, so the wrapper must see this call."""
    cls, _module, manager = tier
    built_first = cls()          # constructed before anything was fitted
    built_first.offered["old"] = (800, True)
    lookup_ledger.lookups_for_engine(engine, {"op": "arm"})
    # The arm installed the wrappers through guard_mode.ensure_installed(engine).
    assert guard_mode.section()["installed"] is True
    built_first.update_state_after_alloc(request("old"), None, 800)
    assert find("old")["adopted"] == 800


def test_arming_the_ledger_installs_the_wrappers_and_disarming_removes_them(tier,
                                                                            engine):
    cls, _module, _manager = tier
    watched = lookup_ledger.LEDGER_METHODS
    before = {name: getattr(cls, name) for name in watched}
    armed = lookup_ledger.lookups_for_engine(engine, {"op": "arm"})
    assert armed["install"]["installed"] is True
    assert any(getattr(cls, name) is not before[name] for name in before)
    disarmed = lookup_ledger.lookups_for_engine(engine, {"op": "disarm"})
    assert disarmed["uninstall"]["uninstalled"] is True
    for name in before:
        assert getattr(cls, name) is before[name], (
            f"{name} was not restored to the fork's own function")


def test_the_ledger_composes_over_a_backport_replaced_method(tier):
    """`update_state_after_alloc` is REPLACED by the PR #56810 backport, and the
    guard goes over that. The marker chain must stay intact and the recording must
    see the arguments the scheduler passed, not the backport's re-derivation."""
    cls, _module, manager = tier
    fork_function = cls.update_state_after_alloc

    def backport_body(self, request_, blocks, num_external_tokens):
        # The upstream replacement: same call, different insides.
        return fork_function(self, request_, blocks, num_external_tokens)

    backport_body.__name__ = "update_state_after_alloc"
    backport_body.__module__ = "kvtier_dev_plugin.backport"
    setattr(backport_body, backport.OWNER_ATTR, backport.OWNER_TAG)
    setattr(backport_body, backport.ORIGINAL_ATTR, fork_function)
    cls.update_state_after_alloc = backport_body

    armed = lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    assert armed["install"]["installed"] is True
    assert "update_state_after_alloc" in armed["install"]["nested"], (
        "the guard layer did not report composing OVER the backport's replacement")

    layers, bottom = backport._layers_of(cls.update_state_after_alloc)
    assert [kind for kind, _fn in layers] == ["guard", "backport"]
    assert bottom is fork_function, "the fork's own body was lost under the layers"

    manager.update_state_after_alloc(request("stacked"), None, 800)
    assert find("stacked")["adopted"] == 800
    assert manager.calls == [("adopt", "stacked", 800)], (
        "the call never reached the fork's body")


def test_the_ledger_composes_over_the_kill_switch_layer_too(tier, monkeypatch):
    """The production stack on this attribute is switch-over-fork (the kill switch
    wraps `get_num_new_matched_tokens`), not guard-over-fork. The ledger must sit on
    top of whichever stack exists and record what the SCHEDULER actually received -
    including (0, False) from a disarmed switch, which is the truth, not a hit."""
    cls, _module, manager = tier
    fitted = backport._install_switch(cls, "get_num_new_matched_tokens")
    assert fitted["wrapped"] is True
    monkeypatch.setitem(backport._state, "armed", False)
    lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    layers, bottom = backport._layers_of(cls.get_num_new_matched_tokens)
    assert [kind for kind, _fn in layers] == ["guard", "switch"]
    assert bottom is not None

    manager.offered["off"] = (800, True)
    assert manager.get_num_new_matched_tokens(request("off"), 32000) == (0, False)
    manager.update_state_after_alloc(request("off"), None, 0)
    assert find("off")["hit_returned"] == 0, (
        "the scheduler was told 0: recording 800 would describe a call that "
        "never happened")
    assert counts()["no_host_copy"] == 1

    monkeypatch.setitem(backport._state, "armed", True)
    manager.offered["on"] = (800, True)
    assert manager.get_num_new_matched_tokens(request("on"), 32000) == (800, True)
    manager.update_state_after_alloc(request("on"), None, 800)
    assert find("on")["hit_returned"] == 800
    assert find("on")["adopted"] == 800
    assert counts()["adopted_full"] == 1
    # The switch underneath still does its job: the disarmed offer never reached
    # the fork's body, the armed one did, and the ledger recorded what each caller
    # was actually told.
    assert manager.calls == [("adopt", "off", 0),
                             ("offer", "on", 32000), ("adopt", "on", 800)]


def test_a_reinstall_does_not_nest_a_second_layer_and_records_once(tier):
    """Re-install is the production path (a boot hook, a verb, a poll). A re-install
    RE-BUILDS the layer from the stored original (guard_mode's own idiom: that is how
    a reload loop cannot nest), so object identity is the wrong assertion. The marker
    CHAIN is the property: exactly one guard layer, the fork's body still underneath,
    and one record per call."""
    cls, _module, manager = tier
    fork_function = cls.update_state_after_alloc
    lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    guard_mode.ensure_installed(cls)
    guard_mode.install(cls)                      # a plain re-install
    guard_mode.ensure_installed(cls)             # and through the engine path
    lookup_ledger.lookups_for_engine(None, {"op": "arm"})   # re-arm, same cap

    depth = 0
    current = cls.update_state_after_alloc
    while getattr(current, guard_mode.OWNER_ATTR, None) == guard_mode.OWNER_TAG:
        depth += 1
        assert depth < 4, f"nested {depth} guard layers: double-wrapped"
        current = getattr(current, guard_mode.ORIGINAL_ATTR)
    assert depth == 1, f"{depth} guard layers on one attribute"
    assert current is fork_function, (
        "the layer beneath the guard is no longer the fork's own function")

    manager.update_state_after_alloc(request("once"), None, 800)
    assert len([item for item in records() if item["req"] == "once"]) == 1, (
        "a double-wrapped method records twice per call")
    assert lookup_ledger.counters()["adopts"] == 1


def test_a_foreign_owner_is_refused_and_the_ledger_says_it_recorded_nothing(tier):
    """The rule that stops this plugin overwriting a third party's patch: a
    foreign marker is refused, and the arm report must not imply recording."""
    cls, _module, _manager = tier

    def foreign_lookup(self, request_, num_computed_tokens):
        return (0, False)

    def foreign_adopt(self, request_, blocks, num_external_tokens):
        return None

    foreign_lookup.__name__ = "get_num_new_matched_tokens"
    foreign_adopt.__name__ = "update_state_after_alloc"
    for foreign in (foreign_lookup, foreign_adopt):
        setattr(foreign, guard_mode.OWNER_ATTR, "some-other-plugin/1")
        setattr(cls, foreign.__name__, foreign)

    armed = lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    assert armed["install"]["installed"] is False
    reason = armed["reason"].lower()
    assert "nothing is wrapped yet" in reason
    assert "another plugin's owner marker" in reason, (
        "the decline must name the REAL reason, not a generic 'no tier class'")
    assert "update_state_after_alloc" in reason
    assert set(armed["install"]["skipped"]) >= set(lookup_ledger.LEDGER_METHODS)
    assert cls.update_state_after_alloc is foreign_adopt
    assert cls.get_num_new_matched_tokens is foreign_lookup
    assert guard_mode.section()["installed"] is False
    # Recording is armed but inert: no wrapper, no records, and no error either.
    assert lookup_ledger.counters()["records"] == 0
    assert lookup_ledger.counters()["record_errors"] == 0


def test_a_trace_disarm_does_not_blind_the_ledger(tier, engine):
    """ONE set of wrappers, THREE consumers. The uninstallers ask
    `guard_mode.wrapper_customers()` now, so the trace ring cannot remove the
    layer the ledger is recording through."""
    cls, _module, manager = tier
    lookup_ledger.lookups_for_engine(engine, {"op": "arm"})
    trace_ring.trace_for_engine(fake.engine(None), {"op": "arm"})
    assert "the lookup ledger" in guard_mode.wrapper_customers()
    payload = trace_ring.trace_for_engine(fake.engine(None), {"op": "disarm"})
    assert payload["uninstall"]["tier_wrappers"]["uninstalled"] is False
    assert "lookup ledger" in payload["uninstall"]["tier_wrappers"]["reason"]
    manager.update_state_after_alloc(request("still-live"), None, 800)
    assert find("still-live")["adopted"] == 800


def test_a_guard_disarm_does_not_blind_the_ledger(tier, engine):
    cls, _module, manager = tier
    lookup_ledger.lookups_for_engine(engine, {"op": "arm"})
    guard_mode.set_guard_mode(True, engine)
    report = guard_mode.set_guard_mode(False, engine)
    assert report["uninstall"] is None
    assert "lookup ledger" in report["reason"]
    assert guard_mode.section()["installed"] is True
    manager.update_state_after_alloc(request("after-guard-off"), None, 0)
    assert find("after-guard-off")["adopted"] == 0


def test_guard_mode_armed_still_declines_and_records_the_pairs_it_saw(tier):
    """Composition with the DECLINE path: a faulted lookup is captured as data and
    declined, and the ledger deliberately does NOT record a value it never got."""
    cls, module, manager = tier
    lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    guard_mode.set_guard_mode(True, None)
    module.FAILING.add("get_num_new_matched_tokens")
    declined = manager.get_num_new_matched_tokens(request("boom"), 32000)
    assert declined == (0, False), "guard mode must still decline"
    assert guard_mode.section()["declined"] == 1
    assert find("boom") is None, (
        "a declined call has no offer to record: recording (0, False) would "
        "invent a no-host-copy we never observed")
    # ... while a call that DID answer is recorded normally.
    manager.offered["fine"] = (800, True)
    module.FAILING.clear()
    assert manager.get_num_new_matched_tokens(request("fine"), 32000) == (800, True)
    assert find("fine")["hit_returned"] == 800


# --- 5. containment ----------------------------------------------------------

def test_a_raising_fork_method_propagates_and_leaves_the_ring_usable(tier):
    """The ledger is not guard mode: with guard OFF the fork's exception must
    reach the engine loop exactly as it would without this plugin."""
    cls, module, manager = tier
    lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    guard_mode.ensure_installed(cls)
    lookup_ledger.record_lookup("good", 32000, 800, True, 3200, 800)
    module.FAILING.add("update_state_after_alloc")
    with pytest.raises(RuntimeError, match="faulted on the adopt"):
        manager.update_state_after_alloc(request("bad"), None, 800)
    module.FAILING.clear()
    assert find("bad") is None, "a call that raised has no inputs to record"
    manager.update_state_after_alloc(request("good"), None, 800)
    assert find("good")["adopted"] == 800
    # One record: the lookup, with its adopt JOINED into it. The call that raised
    # produced nothing, because nothing was ever returned to record.
    assert lookup_ledger.counters()["live"] == 1
    assert lookup_ledger.counters()["adopts"] == 1
    assert lookup_ledger.counters()["joins"] == 1
    assert counts()["adopted_full"] == 1


def test_a_broken_call_shape_is_counted_and_never_reaches_the_scheduler():
    """`args` is a positional tuple by contract. Give the hook a shape that
    violates it and the ledger must count it, not raise into the step loop."""
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 8})
    lookup_ledger.record_from_call(lookup_ledger.METHOD_LOOKUP, 42, (800, True))
    lookup_ledger.record_from_call(lookup_ledger.METHOD_ADOPT, None, None)
    lookup_ledger.record_from_call("not_a_method", (request("x"),), None)
    assert lookup_ledger.counters()["record_errors"] >= 1
    assert snapshot.error_counters()["errors"] >= 1
    # The ring is intact and usable afterwards.
    lookup_ledger.record_lookup("after", 32000, 800, True, 3200, 800)
    assert find("after")["hit_returned"] == 800


def test_a_request_without_a_usable_id_is_still_recorded():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 8})
    lookup_ledger.record_from_call(lookup_ledger.METHOD_LOOKUP,
                                   (object(), 32000), (800, True), None)
    entry = records()[-1]
    assert entry["req"] == "<unknown>"
    assert entry["hit_returned"] == 800
    # An id that is not a string is still an id: keep its value, cheaply.
    lookup_ledger.record_lookup(42, 32000, 800, True, 3200, 800)
    assert records()[-1]["req"] == "42"
    # An id that is an arbitrary object is named by TYPE, never str()'d: a fork
    # type with a heavy __repr__ would be paid for on every request.
    lookup_ledger.record_lookup(types.SimpleNamespace(), 32000, 800, True, 3200, 800)
    assert records()[-1]["req"] == "SimpleNamespace"


def test_a_long_req_id_is_clipped_rather_than_stored_whole():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 8})
    lookup_ledger.record_lookup("x" * 5000, 32000, 800, True, 3200, 800)
    stored = records()[-1]["req"]
    assert len(stored) <= lookup_ledger.MAX_REQ_ID_CHARS + 3
    assert stored.endswith("...")


# --- 6. the verb and the snapshot -------------------------------------------

def test_the_verb_reports_the_shape_it_serves_and_declines_unknown_ops():
    payload = lookup_ledger.lookups_for_engine(None, {"op": "frobnicate"})
    assert "unknown op" in payload["error"]
    assert "verdict" in payload["error"]
    armed = lookup_ledger.lookups_for_engine(None, {"op": "arm"})
    # No tier class in this process: the honest decline, not a silent success.
    assert armed["armed"] is True
    assert "nothing is wrapped yet" in armed["reason"].lower()
    assert "is imported in this process" in armed["reason"]
    assert lookup_ledger.lookups_for_engine(None, {"op": "status"})["counters"]
    verdict = lookup_ledger.lookups_for_engine(None, {"op": "verdict"})
    assert verdict["op"] == "verdict" and verdict["armed"] is True
    assert verdict["lookup_ledger_armed"] is True
    assert verdict["not_covered"]
    assert lookup_ledger.lookups_for_engine(None, {"op": "clear"})["cleared"]
    assert lookup_ledger.lookups_for_engine(None, {"op": "read"})["entries"] == []


def test_the_verb_never_raises_even_when_its_arguments_are_nonsense():
    for args in ({"op": 17}, {"op": None}, "not a dict", {"op": "read",
                                                            "limit": "x"}, {}):
        payload = lookup_ledger.lookups_for_engine(None, args)
        assert isinstance(payload, dict)


def test_the_snapshot_carries_a_bounded_preview_and_the_armed_flag():
    lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": 64})
    for index in range(40):
        lookup_ledger.record_lookup(f"req-{index}", 32000, 800, True, 3200, 800)
    section = snapshot.dev_state_sections()
    ledger = section["lookup_ledger"]
    assert section["lookup_ledger_armed"] is True
    assert ledger["armed"] is True
    assert ledger["cap"] == 64
    assert ledger["records"] == 40
    assert ledger["dropped"] == 0
    assert len(ledger["preview"]) == lookup_ledger.DEFAULT_PREVIEW
    # Newest records, oldest first inside the window: a reader appends in order.
    assert ledger["preview"][0]["req"] == "req-34"
    assert ledger["preview"][-1]["req"] == "req-39"
    assert ledger["preview_truncated"] is True
    assert ledger["pending_requests"], "the pending set is how a reader joins offline"
    assert ledger["not_covered"]
    assert ledger["development_only"] is True
    # A monotonic ns stamp must never be shipped as a raw number.
    assert all("age_ms" in item and "mono_ns" not in item
               for item in ledger["preview"])


def test_the_snapshot_survives_a_broken_ledger_section(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("the ledger section blew up")

    monkeypatch.setattr(lookup_ledger, "section", explode)
    section = snapshot.dev_state_sections()
    assert section["lookup_ledger"]["error"] == "RuntimeError"
    assert section["lookup_ledger_armed"] is None


def test_the_arm_reason_names_a_missing_tier_class_and_a_rearm_keeps_records():
    cap = lookup_ledger.MIN_CAP
    first = lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": cap})
    assert first["reallocated"] is True
    lookup_ledger.record_lookup("kept", 32000, 800, True, 3200, 800)
    again = lookup_ledger.lookups_for_engine(None, {"op": "arm", "cap": cap})
    assert again["armed"] is True and again["reallocated"] is False
    assert "already armed" in again["reason"]
    assert find("kept") is not None, "a re-arm threw away the records"
    bigger = lookup_ledger.lookups_for_engine(None, {"op": "arm",
                                                    "cap": cap * 2})
    assert bigger["reallocated"] is True
    assert find("kept") is None, "a new cap must start a new ring, and say so"


def test_the_route_and_the_argument_spec_exist():
    """`status` derives its surface from these tables; a route in config with no
    spec accepts nothing, which is safe but wrong, so pin the pairing."""
    assert config.ROUTE_LOOKUPS in config.ROUTES
    assert config.ROUTE_LOOKUPS in config.DEV_STATE_ROUTES
    assert config.METHOD_LOOKUPS in config.METHODS
    spec = __import__("kvtier_dev_plugin.endpoint",
                      fromlist=["ARG_SPECS"]).ARG_SPECS[config.ROUTE_LOOKUPS]
    assert set(spec) == {"op", "cap", "limit"}
    assert "verdict" in spec["op"]["values"]
