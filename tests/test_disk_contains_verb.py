r"""`op=contains`: "is the span I parked actually on the disk, and for which ranks?"

THE QUESTION THAT COST AN AFTERNOON. On 2026-10-05 the first end-to-end restore
experiment came back `FAILURE_NO_OFFER`, and the only evidence available about the
parked span was "the store grew engine-wide" - which attributes nothing to the span you
asked about. `disk_index.contains()` had existed for three releases with no HTTP caller,
so the answer was in the process's memory and nowhere an operator could read it.

This suite pins the verb that exposes it, in the order the failure would mislead:

1. THE THREE "I CANNOT ANSWER" VERDICTS COME OUT AS THEMSELVES, with `present: null`.
   A key that cannot name an identity, a process with no journal directory, and an index
   that has replayed nothing since boot are each a DIFFERENT statement from "the entry
   is not here" - and `entries: 0` was read as "my span is not on disk" for hours when
   the fact was that nobody had replayed a journal into this process yet.
2. A PRESENCE IS A SET OF RANKS, not a bool. One parked span is one file per rank, so
   an answer that reports one rank lets a two-ranks-of-four span look complete.
3. THE ANSWER NEVER COMES FROM THE STORE. `op=ls` walks; a membership test that scanned
   every file under a polled endpoint is how you build an I/O storm. The test asserts
   the walk helper is not called, by making it explode.
4. A BAD KEY IS NOT A MISS. "You asked badly" and "it is not there" never share a
   response, at the verb (a `present: null` with a reason) and at the route (a 400 that
   never reaches the engine).
5. THE PER-RANK VIEW AND THE ENTRY TABLE CANNOT DRIFT. A removal line, a truncated
   journal and the entry cap each drop both views together, because the offer path reads
   one and this verb reads the other - and a disagreement between them is exactly the
   kind of confident lie this plugin exists to stop.

Stdlib plus the plugin; `tmp_path` for every file; no engine, no threads, no store walk.
"""

from __future__ import annotations

import os

import pytest
from test_routes import DISK, FakeAsyncLLM, build_app, client_for, token_headers

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_restore,
    disk_spill,
    disk_store,
)

pytest.importorskip("fastapi")

KEY16 = "a3f0b7c1d4e5f607"
GROUP = 3
KEY = KEY16 + f"{GROUP:08x}"                  # 24 hex: the minimum usable key
OTHER_KEY16 = "0f1e2d3c4b5a6978"
OTHER_KEY = OTHER_KEY16 + f"{GROUP:08x}"
WIDTH = 11_468_800
RANKS = ("PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1")


class RealVerbEngine:
    """A fake utility channel that RUNS the real `disk` verb (see the docstring of
    `test_post_query_arguments`, same reason: the answer must be the engine's)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if method == config.METHOD_DISK:
            return disk_store.disk_for_engine(None, args[0] if args else {})
        return {"ok": True, "method": method}


@pytest.fixture(autouse=True)
def fresh_index(tmp_path, monkeypatch):
    """A journal directory, a rank, an empty index - and the index is per-process
    memory, so both the setup and the teardown reset it (a leak here would be the next
    file's "impossible" hit)."""
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANKS[0])
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(tmp_path / "kv-store"))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(64 * WIDTH))
    # The route-level tests below need the plugin live and a token to spend; the
    # index-level ones ignore both.
    monkeypatch.setenv(config.ENV_ENABLE, "1")
    monkeypatch.setenv(config.ENV_TOKEN, "tk")
    disk_restore.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    yield directory
    disk_index.reset_state_for_tests()
    disk_restore.reset_state_for_tests()


def write_journal(directory, rank, lines, start=1):
    """One rank's journal, in the journal's own line format.

    Written by hand rather than with `journal_append`, because `journal_append` writes
    THIS process's journal: the multi-rank case this verb exists for needs four files,
    and four ranks is the whole point.
    """
    path = os.path.join(str(directory),
                        f"{disk_index.JOURNAL_PREFIX}{rank}"
                        f"{disk_index.JOURNAL_SUFFIX}")
    with open(path, "w", encoding="utf-8") as handle:
        for offset, (op, key16, size, group) in enumerate(lines):
            handle.write(f"{start + offset} {op} {key16} {size} {group}\n")
    return path


def spill(ranks=RANKS, key16=KEY16, group=GROUP, size=WIDTH):
    """Four ranks claim one identity, then the index replays. Returns the journals."""
    directory = config.diagnostics_dir()
    paths = [write_journal(directory, rank, [("+", key16, size, group)])
             for rank in ranks]
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    return paths


def membership(key=KEY):
    return disk_index.membership(key)


# --- 1. a hit: the identity, and every rank that claims it ------------------

def test_a_hit_answers_present_with_every_rank_that_claimed_it(fresh_index):
    spill()
    answer = membership()
    assert answer["ok"] is True, answer["reason"]
    assert answer["present"] is True
    assert answer["verdict"] == disk_index.VERDICT_PRESENT
    assert answer["key16"] == KEY16
    assert answer["group_id"] == GROUP
    assert answer["identity"] == disk_index.identity_of(KEY16, GROUP)
    assert answer["shard"] == KEY16[:disk_layout.SHARD_CHARS]
    assert answer["bytes"] == WIDTH, "the entry's own byte claim"
    assert answer["journal"], "the journal the record came from is part of the answer"
    assert answer["rank"], "the entry names the rank whose journal wrote it"
    # THE PART A `contains: true` COULD NEVER SAY. One parked span is one file per
    # rank; four ranks' lines means the span is complete, two means it is not.
    assert answer["ranks_present"] == 4
    assert [row["rank"] for row in answer["ranks"]] == sorted(RANKS)
    for row in answer["ranks"]:
        assert row["bytes"] == WIDTH
        assert row["journal"].endswith(f"{disk_index.JOURNAL_PREFIX}{row['rank']}"
                                       f"{disk_index.JOURNAL_SUFFIX}")
        assert row["seq"] == 1
    assert answer["bytes_all_ranks"] == 4 * WIDTH


def test_a_two_rank_span_reports_two_ranks_and_not_a_hit_that_looks_complete(
        fresh_index):
    """THE PARTIAL SPAN, which is the reason the per-rank detail is not optional.

    `contains()` would answer True for this: the identity IS in the index. But half the
    files behind the span do not exist, so a restore of it must fail - and an answer
    that reported one rank ("present") would have let the operator conclude the span was
    complete. The count is the fact; `next_step` says what to compare it against.
    """
    spill(ranks=RANKS[:2])
    answer = membership()
    assert answer["present"] is True
    assert answer["ranks_present"] == 2
    assert [row["rank"] for row in answer["ranks"]] == sorted(RANKS[:2])
    assert answer["bytes_all_ranks"] == 2 * WIDTH


def test_the_verb_says_what_to_compare_the_rank_count_against(fresh_index):
    """`ranks_present: 2` is only actionable next to the number of ranks the boot has
    armed, and next to the one thing this verb cannot name (a rank's geometry digest).
    The answer spells out both, so the next request is not guesswork."""
    spill(ranks=RANKS[:2])
    answer = disk_store.disk_for_engine(None, {"op": "contains", "key": KEY})
    assert answer["ranks_present"] == 2
    assert "restore.digests_expected" in answer["next_step"]
    assert "op" in answer["next_step"] and "shard" in answer["next_step"]


def test_membership_never_walks_the_store(fresh_index, monkeypatch):
    """`ls` walks; this does not. The endpoint is polled, and a membership test that
    scanned 2,720 files per question is a self-inflicted outage.

    The walk helper is replaced with something that RAISES, so the proof is not an
    assertion about a counter someone forgot to increment.
    """
    def explode(*args, **kwargs):
        raise AssertionError("op=contains must not walk the store directory")

    monkeypatch.setattr(disk_spill, "_walk_store", explode)
    spill()
    answer = disk_store.disk_for_engine(None, {"op": "contains", "key": KEY})
    assert answer["present"] is True
    assert answer["ranks_present"] == 4


def test_the_verb_reports_the_index_it_answered_from(fresh_index):
    spill()
    answer = disk_store.disk_for_engine(None, {"op": "contains", "key": KEY})
    assert answer["op"] == "contains"
    assert answer["read_only"] is True
    index = answer["index"]
    assert index["refreshes"] >= 1
    assert index["entries"] == 1
    assert index["journals_tracked"] == 4
    assert index["bytes_indexed"] == WIDTH
    # THE SAME ANSWER ALSO CARRIES THE RESTORE SIDE (the verb wrapper attaches it), so
    # `ranks_present` and `restore.ranks_armed` can be read without a second request.
    assert "restore" in answer
    assert "units" in disk_index.section()
    assert "rank_records" in disk_index.section()["units"]


# --- 2. a miss, and the three verdicts that are NOT a miss ------------------

def test_a_miss_says_not_present_and_names_what_it_compared_against(fresh_index):
    spill()                            # the index holds one identity...
    answer = membership(OTHER_KEY)     # ...and this is not it
    assert answer["ok"] is True
    assert answer["present"] is False
    assert answer["verdict"] == disk_index.VERDICT_NOT_PRESENT
    assert answer["ranks"] == []
    assert answer["index"]["entries"] == 1
    assert OTHER_KEY16 in answer["reason"] or "does not name" in answer["reason"]


def test_an_index_that_has_never_refreshed_is_not_answered_as_a_miss(fresh_index):
    """`present: null`, because "nothing has been replayed" is not evidence.

    This single field is the most valuable thing on the verb: on 2026-10-05 `entries:
    0` was indistinguishable from "my span is not on disk" and sent the investigation to
    a nonexistent bug for an afternoon.
    """
    write_journal(config.diagnostics_dir(), RANKS[0], [("+", KEY16, WIDTH, GROUP)])
    answer = membership()              # no refresh() has run in this process
    assert answer["present"] is None
    assert answer["verdict"] == disk_index.VERDICT_NEVER_REFRESHED
    assert answer["index"]["refreshes"] == 0
    assert "not present" not in str(answer["verdict"])


def test_a_journal_replay_that_named_no_file_is_its_own_verdict(fresh_index):
    """The index HAS read the journals; there was simply nothing in them. Different
    from never having read them, and from "your identity is not among the entries"."""
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    answer = membership()
    assert answer["present"] is None
    assert answer["verdict"] == disk_index.VERDICT_INDEX_EMPTY
    assert answer["index"]["refreshes"] >= 1
    assert answer["index"]["entries"] == 0


def test_a_process_with_no_journal_directory_says_that(fresh_index, monkeypatch):
    """Neither env var: the index can never be populated in this process. Saying
    `not_present` here would send an operator to the store instead of the env."""
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    monkeypatch.delenv(config.ENV_DISK_JOURNAL_DIR, raising=False)
    answer = membership()
    assert answer["present"] is None
    assert answer["verdict"] == disk_index.VERDICT_NO_JOURNAL_DIR
    assert config.ENV_DISK_JOURNAL_DIR in answer["reason"]


def test_every_verdict_the_verb_can_give_is_in_the_closed_vocabulary(fresh_index):
    spill()
    given = [membership()["verdict"], membership(OTHER_KEY)["verdict"]]
    for verdict in given:
        assert verdict in disk_index.MEMBERSHIP_VERDICTS, verdict


# --- 3. a bad key is never a miss -------------------------------------------

def test_a_malformed_key_is_refused_with_a_reason_not_present_false(fresh_index):
    spill()
    answer = disk_store.disk_for_engine(None, {"op": "contains",
                                               "key": "zzf0b7c1d4e5f60700000003"})
    assert answer["ok"] is False
    assert answer["present"] is None, "'you asked badly' is not 'it is not there'"
    assert answer["verdict"] == disk_index.VERDICT_KEY_REJECTED
    assert "hexadecimal" in answer["reason"]


def test_a_too_short_key_is_refused_and_says_the_minimum(fresh_index):
    """A bare 16-hex `key16` cannot say which GROUP's bytes are meant - the ambiguity
    the identity carries the group to avoid. Refused, with the minimum spelled out."""
    spill()
    answer = disk_store.disk_for_engine(None, {"op": "contains", "key": KEY16})
    assert answer["ok"] is False
    assert answer["present"] is None
    assert answer["verdict"] == disk_index.VERDICT_KEY_REJECTED
    assert str(disk_layout.MIN_KEY_HEX_CHARS) in answer["reason"]


def test_a_missing_key_is_refused(fresh_index):
    answer = disk_store.disk_for_engine(None, {"op": "contains"})
    assert answer["ok"] is False
    assert answer["present"] is None
    assert "key" in answer["reason"]


def test_one_key_rule_for_the_route_and_the_verb(fresh_index):
    """`contains_key_problem` is the ONE definition: the route answers a 400 from it and
    the verb answers data from it, so the two cannot drift into two notions of a key."""
    assert disk_store.contains_key_problem(KEY) is None
    assert disk_store.contains_key_problem(KEY16) is not None
    assert disk_store.contains_key_problem(None) is not None
    assert disk_store.contains_key_problem("") is not None


# --- 4. the two views cannot drift ------------------------------------------

def test_a_removal_line_drops_the_entry_and_the_ranks_together(fresh_index):
    """What a peer's removal line may and may not do (0.2.46).

    WHAT CHANGED, AND WHAT DID NOT. This test used to pin a removal line dropping EVERY
    rank's claim beside the entry, reasoning that a per-rank table still naming three
    ranks next to an entry the offer path could not see "would be a truer-LOOKING lie".
    That put the cart before the horse: those three ranks held READABLE FILES, and the
    thing that was lying was the OFFER answer, not the table. One rank unlinking its own
    slice cannot withdraw three other ranks' files - refusing them was the false-absent
    condition measured at ~442 identities per rank on the box.

    THE INVARIANT THIS TEST EXISTS FOR IS UNTOUCHED AND STILL ASSERTED: the verb and the
    offer path cannot disagree. `present` IS the offer answer, so a span one rank cannot
    read still answers `present: false` and `contains()` still answers False. What
    arrives
    with it now is the rank the span is waiting for, and the three ranks that do hold
    files, instead of a table claiming nobody held anything.
    """
    spill()
    directory = config.diagnostics_dir()
    write_journal(directory, RANKS[0], [("+", KEY16, WIDTH, GROUP),
                                        ("-", KEY16, 0, GROUP)])
    # The first line is already applied, so replaying it would stop the journal on a
    # non-increasing sequence: rewrite the file as the removal alone, which is what the
    # rank would append next after the entry it recorded in the previous refresh.
    path = os.path.join(str(directory),
                        f"{disk_index.JOURNAL_PREFIX}{RANKS[0]}"
                        f"{disk_index.JOURNAL_SUFFIX}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"2 {disk_index.REMOVE_LINE} {KEY16} 0 {GROUP}\n")
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    assert refreshed["removals"] == 1
    answer = membership()
    assert answer["present"] is False, "the offer path will not see this span"
    for field in ("offerable", "ranks_missing", "ranks_expected"):
        assert field in answer, (
            f"`{field}` is not published, so an answer of `present: false` cannot say "
            "WHICH rank the span is waiting for - the difference between a rank's "
            "ceiling pass and a prefix nobody ever spilled")
    assert answer["offerable"] is False, answer
    assert answer["ranks_present"] == 3, (
        "three ranks hold readable files and the per-rank view must say so: a removal "
        f"line retires the claim of the rank that wrote it ({answer['ranks']})")
    assert answer["ranks_missing"] == [RANKS[0]], answer
    assert answer["verdict"] == disk_index.VERDICT_PARTIAL_RANKS, answer
    assert disk_index.contains(KEY) is False, "both views must agree about the offer"
    try:
        three_only = disk_index.contains(KEY, RANKS[1:])
    except TypeError as exc:
        raise AssertionError(
            "the offer predicate takes no rank set, so it cannot answer for the three "
            f"ranks that still hold their files ({exc})") from exc
    assert three_only is True, (
        "asked about the three ranks that hold the files, the predicate must say yes")


def test_a_truncated_journal_forgets_that_rank_and_only_that_rank(fresh_index):
    """The entry survives when another rank still backs it; the truncating rank's claim
    goes, so the rank count drops rather than lying.

    AND SINCE 0.2.46 THE SPAN IS NO LONGER OFFERABLE, which is the safe half of the same
    rule rather than a weakening of it. A journal this drain can no longer read is a
    rank
    whose files it cannot PROVE, and a restore reads one file per rank - promising the
    span on three peers' claims is exactly the false present that reaches the fourth
    rank
    as `file_missing`. The entry stays (the other ranks' files are real bytes the
    capacity
    view must keep pricing), the rank count drops to three (the truth), and
    `ranks_missing`
    names the rank whose journal went.
    """
    paths = spill()
    victim = paths[0]
    with open(victim, "w", encoding="utf-8") as handle:
        handle.write("")
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    answer = membership()
    assert disk_index.contains(KEY) is False, (
        "a rank whose journal cannot be read cannot be proved to hold its file, and a "
        "restore needs every rank's file")
    assert answer["present"] is False and answer["offerable"] is False, answer
    assert answer["bytes"] > 0, "the entry survives: the peers' files are real bytes"
    assert answer["ranks_present"] == 3
    assert RANKS[0] not in [row["rank"] for row in answer["ranks"]]
    assert answer["ranks_missing"] == [RANKS[0]], answer


def test_the_entry_cap_drops_the_rank_rows_with_the_entry(fresh_index, monkeypatch):
    """An identity evicted from the entry table must not leave rank rows behind: the
    verb would report ranks for an identity the offer path has already forgotten."""
    monkeypatch.setattr(disk_index, "MAX_INDEX_ENTRIES", 2)
    directory = config.diagnostics_dir()
    # ONE journal, four lines: four identities into a two-entry table, so the cap
    # evicts for real. (Four separate files for one rank would each SHRINK the previous
    # one and stop that journal instead - a different rule, pinned elsewhere.)
    write_journal(directory, RANKS[0],
                  [("+", f"{index:016x}", WIDTH, GROUP) for index in range(4)])
    disk_index.refresh(force=True)
    assert disk_index.counters()["entries_dropped"] >= 1
    assert disk_index.counters()["rank_records"] <= 2
    # The dropped identity: no entry, no rank rows, and no `entry_dropped...` verdict.
    answer = membership(f"{0:016x}" + f"{GROUP:08x}")
    assert answer["present"] is False
    assert answer["ranks"] == []
    assert answer["verdict"] == disk_index.VERDICT_NOT_PRESENT


def test_a_dropped_entry_with_rank_lines_remaining_names_itself(fresh_index,
                                                                monkeypatch):
    """The one state where the two views legitimately disagree: rank lines exist, the
    entry does not. The verb says exactly that, and `present: false`, because the offer
    path will not see it either."""
    spill()
    disk_index._state["entries"] = {}      # the entry table loses it, ranks stay
    answer = membership()
    assert answer["present"] is False
    assert answer["verdict"] == disk_index.VERDICT_ENTRY_DROPPED
    assert answer["ranks_present"] == 4
    assert "dropped" in answer["reason"]


def test_the_rank_cap_is_honoured_and_counted(fresh_index, monkeypatch):
    """More ranks than the cap: the identity keeps the ranks it recorded and the rest
    are COUNTED as dropped. Never a silent truncation, because a rank count read as
    "2 of 4" when the truth is "4 of 4" would end an investigation early."""
    monkeypatch.setattr(disk_index, "MAX_RANKS_PER_IDENTITY", 2)
    spill()
    answer = membership()
    assert answer["ranks_present"] == 2
    assert disk_index.counters()["rank_records_dropped"] >= 2


# --- 5. the witness's ledger is not this verb's -----------------------------

def test_asking_the_verb_does_not_move_the_witness_counters(fresh_index):
    """`lookup()` counts hits and misses for the restore witness; an experiment reads
    them as evidence about the OFFER path. An operator asking a question twice must not
    change that evidence - so `membership()` counts its own checks instead."""
    spill()
    before = disk_index.counters()
    membership()
    membership(OTHER_KEY)
    after = disk_index.counters()
    assert after["hits"] == before["hits"]
    assert after["misses"] == before["misses"]
    assert after["membership_checks"] == before["membership_checks"] + 2
    assert disk_index.section()["membership_checks"] == 2


def test_membership_never_raises(fresh_index, monkeypatch):
    """The verb is contained by contract: whatever the index does internally, the answer
    is a dict with a verdict, never an exception the caller has to survive."""
    def explode(*args, **kwargs):
        raise RuntimeError("the index is having a day")

    monkeypatch.setattr(disk_index, "journal_directory", explode)
    answer = membership()
    assert answer["ok"] is False
    assert answer["present"] is None
    assert answer["verdict"] == disk_index.VERDICT_CHECK_FAILED
    assert "RuntimeError" in answer["reason"]


# --- 6. the published surface -----------------------------------------------

def test_the_verb_is_in_every_table_that_describes_the_surface(fresh_index):
    """`DISK_OPS`, the dispatch table, the route's argument spec and the op index
    `/status` publishes are four descriptions of one surface. A `contains` reachable
    through one and missing from another is an undocumented verb (or a documented one
    that 400s), which is the reason these are pinned together."""
    assert "contains" in disk_store.DISK_OPS
    assert "contains" in disk_store._DISK_OPS
    assert "contains" not in disk_store.MUTATING_DISK_OPS, \
        "it reads: it must not be refused as a mutating op, nor arm anything"
    spec = endpoint_spec()
    assert "contains" in spec["op"]["values"]


def endpoint_spec():
    from kvtier_dev_plugin import endpoint
    return endpoint.ARG_SPECS[config.ROUTE_DISK]


def test_the_op_index_status_publishes_lists_contains(fresh_index):
    client = client_for(build_app(FakeAsyncLLM(RealVerbEngine())))
    status = client.get(config.ROUTE_PREFIX + "/status").json()
    assert status["disk"]["ops"] == list(disk_store.DISK_OPS)
    assert "contains" in status["disk"]["ops"]
    # THE ARGUMENT TABLE AND THE OP LIST AGREE, in both directions: an op the verb
    # dispatches that the route will not carry is invisible, and one the route carries
    # that the verb refuses is a 400 an operator has to discover by hand.
    assert set(endpoint_spec()["op"]["values"]) == set(disk_store.DISK_OPS)


def test_the_route_answers_the_query_form_and_the_body_form_alike(fresh_index):
    """Both forms must work: the operator's tooling uses both, and this verb is the one
    that answers the central question, so a form that silently 400s is not an option."""
    spill()
    engine = RealVerbEngine()
    client = client_for(build_app(FakeAsyncLLM(engine)))
    query = client.post(f"{DISK}?op=contains&key={KEY}", headers=token_headers())
    body = client.post(DISK, json={"op": "contains", "key": KEY},
                       headers=token_headers())
    assert query.status_code == 200, query.text
    assert body.status_code == 200, body.text
    assert query.json()["present"] is True
    assert body.json()["present"] is True
    assert query.json()["ranks_present"] == 4
    assert body.json()["ranks_present"] == 4


def test_a_bad_key_is_a_400_that_never_reaches_the_engine(fresh_index):
    spill()
    engine = RealVerbEngine()
    client = client_for(build_app(FakeAsyncLLM(engine)))
    response = client.post(f"{DISK}?op=contains&key={KEY16}",
                           headers=token_headers())
    assert response.status_code == 400
    assert str(disk_layout.MIN_KEY_HEX_CHARS) in response.json()["detail"]
    assert engine.calls == [], "the refusal happened above the hop, as it must"
    missing = client.post(f"{DISK}?op=contains", headers=token_headers())
    assert missing.status_code == 400
    assert "key" in missing.json()["detail"]


def test_the_safe_method_answers_contains_and_400s_a_bad_key_too(fresh_index):
    """`contains` is a read, so GET answers it - and a GET gets the SAME 400 for a bad
    key as a POST. A GET is the form an operator reaches for when they are already
    unsure; answering a malformed key with `present: false` there would be the single
    most expensive ambiguity on the verb, on its cheapest path.
    """
    spill()
    client = client_for(build_app(FakeAsyncLLM(RealVerbEngine())))
    good = client.get(f"{DISK}?op=contains&key={KEY}", headers=token_headers())
    assert good.status_code == 200, good.text
    assert good.json()["present"] is True
    assert good.json()["ranks_present"] == 4
    bad = client.get(f"{DISK}?op=contains&key={KEY16}", headers=token_headers())
    assert bad.status_code == 400
    assert str(disk_layout.MIN_KEY_HEX_CHARS) in bad.json()["detail"]


def test_a_hit_over_http_carries_the_rank_detail_and_the_restore_side(fresh_index):
    """The payload an operator actually reads, as one response: the per-rank detail,
    the index it came from, and the restore section the verb wrapper attaches - so
    "is it there, for whom, and would the scheduler offer it?" is ONE request."""
    spill()
    client = client_for(build_app(FakeAsyncLLM(RealVerbEngine())))
    answer = client.post(DISK, json={"op": "contains", "key": KEY},
                         headers=token_headers()).json()
    assert answer["present"] is True
    assert answer["ranks_present"] == 4
    assert answer["index"]["refreshes"] >= 1
    assert answer["restore"]["gate"]["ok"] is False, "nothing is armed in this test"
    assert answer["restore"]["refused"] == {} or isinstance(
        answer["restore"]["refused"], dict)


def test_the_shard_filter_the_verb_points_at_is_reachable(fresh_index):
    """`contains` cannot name a rank's FILE (the journal line carries no digest), so it
    sends the operator to `op=ls` with a `shard`. That filter has been implemented in
    the verb for ages and was unreachable because the route's key whitelist did not
    list `shard` - every attempt answered `unsupported request keys: ['shard']`. A
    pointer to a 400 would be its own little 2026-10-05, so the whitelist is pinned."""
    client = client_for(build_app(FakeAsyncLLM(RealVerbEngine())))
    response = client.post(f"{DISK}?op=ls&shard={KEY16[:2]}&limit=5",
                           headers=token_headers())
    assert response.status_code == 200, response.text
    assert response.json()["shard"] == KEY16[:2]
    bad = client.post(f"{DISK}?op=ls&shard=nope", headers=token_headers())
    assert bad.status_code == 200
    assert "shard is not usable" in bad.json()["reason"], "the engine's rule still owns"
