r"""THE BOUNCE-BUFFER POOL AS A PRICE THE OFFER PAYS BEFORE IT EXISTS (0.2.31).

Every pre-existing clamp on an offer asks how many DESTINATIONS it can have: CPU
blocks, tokens, the run's own anchors. None of them asked how many READS the process
can carry, and a read is not free - it takes a host-RAM bounce buffer at issue and
returns it at the apply point, one forward pass later, so an offer of N jobs needs the
pool to hold all N buffers AT ONCE. `disk_read` refuses rather than queueing when it
cannot (`request_read`'s `buffer_cap`), and the 0.2.30 consequence of not pricing that
was the sequence this suite exists to make impossible:

    13 jobs offered to a pool that fits 11 -> the surplus refused at bind -> pruned
    into `invalid_block_ids` -> the engine reset the request's computed tokens and
    rescheduled it -> `note_reported` read that same set back and fused the request.

So the tests here are in the order the money moves:

1. THE PRICE. One read costs one bounce buffer of exactly the file's size, so the
   offer is priced at the LARGEST shape in play - from the index when it knows, from a
   rank's measured slot width when it does not, from a named constant as a last resort.
   An under-estimate is exactly the cliff, so every test asserts the conservative side.
2. THE CAP. A run the pool cannot carry is SHORTENED to the job count it can, and a
   run whose minimum (one span plus the boundary states it cannot be claimed without)
   does not fit makes NO OFFER AT ALL - the same shape as `no_disk_run`, because a
   shorter offer is what got refused at bind last time.
3. WHICH CEILING BOUND IT. `offer_cap` (tokens) keeps its semantics; the pool budget
   is an additional, tighter constraint, and `bounded_by` says which one (or both)
   actually decided the offer's length.
4. THE RECORD AND THE LINE. A refusal that cannot be read is a refusal that cannot be
   fixed, so a `buffer_cap` record carries the five byte/job figures that decided it
   and the provenance of the block it was reading, and the log line for an episode is
   printed ONCE, not once per step.
5. THE CROSS-PROCESS VIEW. A worker sees refusals the scheduler cannot, and the only
   worker-to-scheduler channel is a GPU block id - so the records ride the rank-file
   transport and `/plugins/kvtier/disk` `{"op": "failures"}` merges them.

Offline fakes only: no engine, no torch, no CUDA, no network, no live box.
"""

from __future__ import annotations

import test_disk_gdn_boundary_restore as gdn
import test_disk_restore as base

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_restore,
    disk_spill,
    disk_store,
)

# The fixtures live in the sibling module; importing the names is how this suite reuses
# them, autouse ones included (a fresh boot per test, and a /tmp root that is allowed).
armed = base.armed                      # noqa: F811 - the gate open, the switch up
armed_gap_fill = base.armed_gap_fill              # noqa: F811 - scope=all
allow_tmp_root = base.allow_tmp_root    # noqa: F811 - autouse
fresh = base.fresh                      # noqa: F811 - autouse
fill = base.fill
offer = base.offer
hybrid = gdn.hybrid                     # the live box's align-mode geometry

BLOCK = base.BLOCK
WIDTH = base.WIDTH
CACHEABLE = base.CACHEABLE
Manager = base.Manager
Req = base.Req
HYBRID_CACHEABLE = gdn.HYBRID_CACHEABLE
GDN_GROUPS = gdn.GDN_GROUPS
ATTENTION_GROUP = gdn.ATTENTION_GROUP

# THE TWO SHAPES THE LIVE BOX MEASURES AT THE SAME MOMENT: 11,468,800 B on a PP0 rank
# and 12,902,400 B on a PP1 rank (one rank slot, one spill file). The fallback constant
# has to be the LARGER one, and these numbers are the reason.
PP0_SHAPE = 11_468_800
PP1_SHAPE = 12_902_400


def key16_of(seed: int, group: int) -> str:
    """The 16-hex `key16` a spilled file's identity is stored under.

    Taken from the ONE rule that splits an identity (`disk_layout.split_block_key`)
    rather than by slicing, because `key16` is not "the first 16 characters of
    something" - it is the digest part of the engine's group-qualified hash, and a
    test that invented its own slice would be asserting about a string no writer
    produces.
    """
    split = disk_layout.split_block_key(base.key_hex(seed, group))
    assert split["ok"], split
    return split["key16"]


def fill_sized(root, blocks, groups=CACHEABLE, size=WIDTH, seeds=None):
    """Journal one spill line per (block, group) at a CHOOSEN byte size.

    `base.fill` journals its own `WIDTH`; the pool's price comes from the journalled
    size, so a test that wants to price an offer at the real 11,468,800-byte shape
    needs to say so here. Journal-only (no file is written): the offer path reads the
    INDEX, and `disk_read` is what would open the file - so writing 12.9 MB per job to
    prove an offer-time number would be 300 MB of fixture for an assertion about a
    dictionary.
    """
    seeds = list(seeds if seeds is not None else range(1, blocks + 1))
    for seed in seeds:
        for group in groups:
            appended = disk_index.journal_append(disk_index.SPILL_LINE,
                                                 base.key_hex(seed, group), size)
            assert appended["ok"], appended["reason"]
    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"], refreshed
    return seeds


def pool_at_most_jobs(monkeypatch, jobs, size):
    """Make the bounce-buffer pool hold exactly `jobs` reads of `size` bytes.

    Written through the REAL environment variable and a re-resolution of the reader's
    limits - the same path a systemd drop-in takes - because the number under test is
    the one `disk_read.pool_state()` publishes, not a value a test could hand to the
    offer path directly.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, str(jobs * size))
    disk_read.reset_state_for_tests()
    return jobs * size


def last_pool() -> dict:
    return disk_restore._state["last_offer_pool"] or {}


# --- 1. the price ---------------------------------------------------------------

def test_the_price_is_the_largest_file_size_this_run_would_read(armed,
                                                                monkeypatch):
    """A store holding TWO shapes prices the offer at the bigger one, from the index.

    The live box measures 11,468,800 B on PP0 and 12,902,400 B on PP1 at the same
    instant, and one offer spans every rank - so pricing at the smaller shape is an
    over-promise about the pool on exactly the ranks that cost more. The index carries
    each identity's own journalled size, so this is not a guess: the offer walks the
    identities it is about to read and takes the maximum.
    """
    fill_sized(armed, blocks=4, groups=GDN_GROUPS, size=PP0_SHAPE)
    fill_sized(armed, blocks=4, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)
    # A pool wide enough that the PRICE is the only thing under test here; the cap
    # that this price then produces is the subject of the tests below.
    pool_at_most_jobs(monkeypatch, 20, PP1_SHAPE)
    answer = offer(hybrid(), Req(blocks=4), 0, (0, False))
    assert answer == (4 * BLOCK, True), disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["bytes_per_job"] == PP1_SHAPE, (
        "the bigger of the two shapes in the run wins the price")
    assert pricing["price_source"].startswith("exact:"), pricing["price_source"]
    assert "8 identit" in pricing["price_source"], pricing["price_source"]


def test_the_rank_width_prices_the_offer_when_no_identity_names_a_size(armed,
                                                                       monkeypatch):
    """EngineCore has no KV tensors, so the next-best size is a rank's own measurement.

    With the index asked and silent, the offer falls to the widest slot width a live
    rank file published - and says so, in the same field, with how many ranks
    published it. `bytes_per_file_max` is the widest, never the first or the average,
    for the same reason as above: the cap has to be one that cannot be optimistic for
    somebody.
    """
    fill_sized(armed, blocks=4, groups=HYBRID_CACHEABLE)
    monkeypatch.setattr(disk_index, "entry_bytes", lambda key: None)
    monkeypatch.setattr(disk_spill, "slot_width_bytes", lambda: None)
    monkeypatch.setattr(
        disk_restore, "worker_proof",
        lambda force=False: {"ok": True, "count": 1, "ranks": ["PP0_TP0"],
                             "reason": None, "ranks_armed": 1, "ranks_seen": 1,
                             "reads_applied": 0, "reason_kind": None,
                             "bytes_per_file_max": PP1_SHAPE,
                             "bytes_per_file_ranks": 2})
    pool_at_most_jobs(monkeypatch, 20, PP1_SHAPE)
    answer = offer(hybrid(), Req(blocks=4), 0, (0, False))
    assert answer == (4 * BLOCK, True), disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["bytes_per_job"] == PP1_SHAPE
    assert pricing["price_source"].startswith("rank width:"), pricing["price_source"]
    assert "2 rank(s) published" in pricing["price_source"]


def test_the_fallback_is_the_larger_shape_and_names_the_smaller_one(monkeypatch):
    """With nothing to ask, the price is a CONSTANT an operator can see the origin of.

    A pre-index offer must still be priced, and the safe direction is a small offer,
    not a refused one: 12,902,400 (the larger of the two shapes this build writes)
    caps a little harder than the truth on a PP0 rank. The source string names the
    smaller figure too, so a reader looking at a cap that seems too tight can see that
    the constant was chosen deliberately rather than guessed.
    """
    monkeypatch.setattr(disk_index, "entry_bytes", lambda key: None)
    monkeypatch.setattr(disk_spill, "slot_width_bytes", lambda: None)
    monkeypatch.setattr(disk_restore, "worker_proof", lambda force=False: {})
    monkeypatch.setitem(disk_restore._state, "ranks_bytes_per_file_max", None)
    price, source = disk_restore._pool_price_bytes([], 0)
    assert price == disk_restore.POOL_PRICE_BYTES_PER_JOB_FALLBACK == PP1_SHAPE
    assert source.startswith("fallback:")
    # The smaller shape is named too, so a cap that looks too tight reads as a
    # deliberate choice rather than as a guess.
    assert "11,468,800" in source, source


# --- 2. the cap -----------------------------------------------------------------

def test_thirteen_jobs_are_capped_to_the_eleven_the_pool_can_carry(armed, monkeypatch):
    """THE INCIDENT'S OWN ARITHMETIC, priced before the offer exists.

    Nine blocks on the live geometry cost 9 positional reads PLUS one boundary-state
    read per GDN group: 13 jobs - exactly the offer that was refused at bind. With the
    pool set to carry 11, the honest offer is the largest run that fits (7 span blocks
    + the 4 boundary states it cannot be claimed without = 11 jobs), and every figure
    of that decision is published: what it needed, what the pool carries, what was
    dropped, and which ceiling bound it.
    """
    fill_sized(armed, blocks=9, groups=GDN_GROUPS, size=PP0_SHAPE)
    fill_sized(armed, blocks=9, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)
    pool_at_most_jobs(monkeypatch, 11, PP1_SHAPE)
    answer = offer(hybrid(), Req(blocks=9), 0, (0, False))
    assert answer == (7 * BLOCK, True), disk_restore._state.get("last_reason")
    pricing = last_pool()
    assert pricing["jobs_the_offer_needed"] == 13
    assert pricing["jobs_the_pool_can_carry"] == 11
    assert pricing["jobs_offered"] == 11
    assert pricing["jobs_dropped"] == 2
    assert pricing["blocks_offered"] == 7
    assert pricing["bytes_offered"] == 11 * PP1_SHAPE
    assert pricing["bounded_by"] == ["pool_budget"], (
        "the pool, not the token ceiling, is what shortened this offer")
    counters = disk_restore.counters()
    assert counters["offers"] == 1
    assert counters["offer_pool_capped"] == 1
    assert counters["offer_pool_refused"] == 0
    # The CPU slots follow the shortened offer: 7 span blocks + 4 boundary states.
    assert disk_restore._state["slots_allocated"] == 11
    # And the pool's own view says the same thing to a reader who never looks at a
    # refusal: an offer the pool could carry is one offer's business, and the number
    # of jobs one offer can EVER have is the whole pool divided by one job.
    pool = disk_restore.section()["offer_pool"]
    assert pool["jobs_the_pool_can_carry"] == 11
    assert pool["bytes_per_job"] == PP1_SHAPE
    assert pool["capped_offers"] == 1


def test_a_pool_holding_the_boundary_states_but_no_span_makes_no_offer(armed,
                                                                       monkeypatch):
    """FOUR of the thirteen jobs are boundary states, and four is NOT an offer.

    The pool here holds the 4 GDN state reads and nothing else. Making the offer would
    mean a 4-block tuple with no span, which is a claim about a run of zero length -
    so the answer is the `no_disk_run` shape: no offer, the fork's answer returned
    unchanged, and a refusal that says the pool can hold the boundary states but not
    one span on top of them.
    """
    fill_sized(armed, blocks=9, groups=GDN_GROUPS, size=PP1_SHAPE)
    fill_sized(armed, blocks=9, groups=(ATTENTION_GROUP,), size=PP1_SHAPE)
    pool_at_most_jobs(monkeypatch, 4, PP1_SHAPE)
    ram = (0, False)
    answer = offer(hybrid(), Req(blocks=9), 0, ram)
    assert answer is ram, "no offer at all, and the fork's answer byte-identical"
    counters = disk_restore.counters()
    assert counters["offers"] == 0
    assert counters["offer_pool_refused"] == 1
    assert counters["refused"]["no_pool_budget"] == 1
    reason = counters["last_refusal"]
    assert reason.startswith("no_pool_budget: "), reason
    # THE ARITHMETIC COMES FIRST BECAUSE `last_refusal` IS CLIPPED (MAX_STRING_CHARS):
    # a message that put the price source in the middle used to have its decisive tail
    # cut off, which is a refusal an operator cannot act on.
    assert "the pool holds 4 read(s) of 12902400 bytes" in reason, reason
    assert "budget 51609600 B" in reason, reason
    assert "1 positional + 4 boundary-state read(s)" in reason, reason
    assert "offering the 13-job" in reason, reason
    # The record keeps the same arithmetic in its own words.
    record = disk_restore.load_failures()[0]
    assert "cannot hold one span" in record["reason"], record["reason"]
    assert "1 positional + 4 boundary-state reads" in record["reason"], record["reason"]
    assert record["jobs_needed"] == 13 and record["jobs_fit"] == 4


def test_a_pool_carrying_too_little_for_one_span_makes_no_offer_at_all(armed,
                                                                       monkeypatch):
    """A CAP OF ZERO AFFORDABLE JOBS IS NO OFFER, and the reason carries the bytes.

    A pool of 8,192 bytes against a 4,096-byte read holds two jobs; one span on this
    geometry costs three positional reads, so the run cannot be serviced at any
    length. The refusal names the budget, the unit price, the source of that price and
    what one span costs - the four numbers an operator needs to tell "make the pool
    bigger" from "this model's spans cost more than you thought".
    """
    fill(armed, blocks=8)                      # journalled at WIDTH = 4,096 B
    budget = pool_at_most_jobs(monkeypatch, 2, WIDTH)
    ram = (0, False)
    answer = offer(Manager(), Req(blocks=8), 0, ram)
    assert answer is ram
    counters = disk_restore.counters()
    assert counters["offers"] == 0
    assert counters["refused"]["no_pool_budget"] == 1
    reason = counters["last_refusal"]
    assert reason.startswith("no_pool_budget: "), reason
    assert f"the pool holds 2 read(s) of {WIDTH} bytes" in reason, reason
    assert f"(budget {budget} B)" in reason, reason
    assert "3 positional + 0 boundary-state read(s)" in reason, reason
    assert "offering the 24-job run" in reason, reason
    # Where the price came from is published in the offer's own pricing (the reason is
    # clipped, and the arithmetic outranks the provenance of a number).
    assert last_pool()["price_source"].startswith("exact:"), last_pool()


# --- 3. which ceiling bound it --------------------------------------------------

def test_bounded_by_names_the_token_ceiling_when_only_tokens_bound_it(armed,
                                                                      monkeypatch):
    """A run that is merely too LONG says `offer_cap`, and the pool stays silent.

    The pre-existing token ceiling is untouched by this change and keeps its own
    counter; the pool view must not claim credit for a shortening it did not do, or
    an operator reading "pool" turns a knob that changes nothing.
    """
    fill(armed, blocks=12)                       # 12 wanted, `offer_cap` allows 10
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, str(200 * WIDTH))
    disk_read.reset_state_for_tests()
    assert offer(Manager(), Req(blocks=12), 0, (0, False)) == (10 * BLOCK, True)
    assert last_pool()["bounded_by"] == ["offer_cap"], last_pool()
    assert disk_restore.counters()["offer_clamped"] == 1
    assert disk_restore.counters()["offer_pool_capped"] == 0


def test_bounded_by_names_the_pool_when_only_the_pool_bounds_it(armed, monkeypatch):
    """A run the pool cannot carry says `pool_budget`, and nothing else."""
    fill(armed, blocks=8)                        # 8 blocks = 24 jobs at 4,096 B
    pool_at_most_jobs(monkeypatch, 11, WIDTH)
    assert offer(Manager(), Req(blocks=8), 0, (0, False)) == (3 * BLOCK, True)
    assert last_pool()["bounded_by"] == ["pool_budget"], last_pool()
    counters = disk_restore.counters()
    assert counters["offer_clamped"] == 0
    assert counters["offer_pool_capped"] == 1


def test_bounded_by_lists_both_ceilings_in_the_order_they_were_applied(armed,
                                                                      monkeypatch):
    """Guilty of both: clamped to 10 blocks by tokens, then to 3 by the pool.

    Without this an operator reading "offer shortened" has to guess which knob to
    turn, and turning the wrong one is how a pool fix becomes a token-ceiling fix that
    changes nothing.
    """
    fill(armed, blocks=12)
    pool_at_most_jobs(monkeypatch, 11, WIDTH)
    assert offer(Manager(), Req(blocks=12), 0, (0, False)) == (3 * BLOCK, True)
    assert last_pool()["bounded_by"] == ["offer_cap", "pool_budget"], last_pool()
    counters = disk_restore.counters()
    assert counters["offer_clamped"] == 1 and counters["offer_pool_capped"] == 1


# --- 4. the record and the line -------------------------------------------------

def test_a_buffer_cap_record_carries_the_five_numbers_and_the_provenance(armed,
                                                                        monkeypatch):
    """The refusal that used to be unpublishable, published with its arithmetic.

    `pool_budget_bytes`, `pool_committed_bytes`, `bytes_needed`, `jobs_needed` and
    `jobs_fit` answer the two questions a `buffer_cap` can mean: "is the pool too
    small for an offer this big" (budget versus needed) and "is something holding
    buffers it is not returning" (committed). Plus the block's provenance, and the
    limit of what provenance can claim: a journal line is
    `<seq> <+|-> <key16> <bytes> <group>`, so WHICH REQUEST wrote a block is not
    knowable and the record says that rather than inventing a field for it.
    """
    fill(armed, blocks=8)
    budget = pool_at_most_jobs(monkeypatch, 2, WIDTH)
    offer(Manager(), Req(request_id="req-7", blocks=8), 0, (0, False))
    records = disk_restore.load_failures()
    assert records, "the refusal is a record, not only a counter"
    record = records[0]
    assert record["kind"] == "buffer_cap"
    assert record["where"] == "offer"
    assert record["request_id"] == "req-7"
    assert record["pool_budget_bytes"] == budget
    assert record["pool_committed_bytes"] == 0
    assert record["bytes_needed"] == WIDTH
    assert record["jobs_needed"] == 24
    assert record["jobs_fit"] == 2
    assert record["key16"] and record["group"] is not None
    prov = record["block_provenance"]
    assert prov is not None, "the scheduler holds the index, so it can name the block"
    assert prov["known_to_this_index"] is True
    assert prov["bytes"] == WIDTH
    assert prov["writer_rank"] == "PP0_TP0"
    assert isinstance(prov["writer_journal_seq"], int)
    assert prov["learned_at_is_learn_time_not_write_time"] is True
    assert prov["learned_age_seconds"] is not None and prov["learned_age_seconds"] >= 0
    # WHAT IT DOES NOT CLAIM: no invented writer-request field, in the record or in
    # the provenance beside it.
    assert "writer_request_id" not in record and "writer_request_id" not in prov
    assert "request_id" not in prov
    report = disk_restore.load_failure_report()
    assert "NOT knowable" in report["provenance_meaning"]


def test_the_failure_line_is_once_per_episode_not_once_per_step(armed, monkeypatch,
                                                                caplog):
    """One line per episode, because the flood is a known failure of this module.

    The refusal here repeats EVERY step a matching request is scheduled - which is
    exactly the shape that once produced 1,048 lines a second and cost the module its
    log. So the line is rate-limited per kind, the suppression is counted (a silent
    rate limiter is a lie by omission), and the window is a named constant an operator
    can reason about.
    """
    fill(armed, blocks=8)
    pool_at_most_jobs(monkeypatch, 2, WIDTH)
    manager, request = Manager(), Req(blocks=8)
    with caplog.at_level("WARNING", logger="kvtier_dev_plugin.disk_restore"):
        for _ in range(5):
            offer(manager, request, 0, (0, False))
    lines = [rec.getMessage() for rec in caplog.records
             if "DISK[load-fail]" in rec.getMessage()]
    assert len(lines) == 1, lines
    assert "kind=buffer_cap" in lines[0]
    assert "pool_budget=" in lines[0] and "pool_committed=" in lines[0]
    counters = disk_restore.counters()
    assert counters["refused"]["no_pool_budget"] == 5
    assert counters["failure_log_lines"] == 1
    assert counters["failure_log_suppressed"] == 4
    # The window is a WINDOW, not a mute: with it gone, the next episode is logged.
    monkeypatch.setattr(disk_restore, "REFUSAL_LOG_MIN_SECONDS", 0.0)
    with caplog.at_level("WARNING", logger="kvtier_dev_plugin.disk_restore"):
        offer(manager, request, 0, (0, False))
    assert len([rec for rec in caplog.records
                if "DISK[load-fail]" in rec.getMessage()]) == 2


# --- 5. the cross-process view --------------------------------------------------

def test_the_report_merges_this_process_with_the_rank_files(armed):
    """A worker sees refusals the scheduler structurally cannot, and says so.

    `disk_restore.reads_failed` on EngineCore is zero by construction, because the
    read happens in the worker and the only worker-to-scheduler channel is a set of
    GPU block ids. The records therefore ride the rank-file transport (the same one
    that carries `ranks_reads_applied`) and this merges them newest-first - attaching
    the provenance in the SCHEDULER, the one process holding an index, for a record
    whose worker-side lookup could only ever have been empty.
    """
    fill(armed, blocks=2)
    rank_record = {"seq": 99, "ts": 1.0, "kind": "buffer_cap", "where": "bind",
                   "request_id": "req-9", "group": 0, "key16": key16_of(1, 0),
                   "jobs_needed": 13, "jobs_fit": 11,
                   "pool_budget_bytes": 11 * PP0_SHAPE,
                   "pool_committed_bytes": 11 * PP0_SHAPE,
                   "reason": "refused at bind in the worker", "rank": "PP1_TP0",
                   "block_provenance": None}
    disk_restore._state["rank_failure_records"] = [rank_record]
    disk_restore._state["ranks_load_failures"] = 1
    _note_a_local_failure()
    report = disk_restore.load_failure_report()
    assert report["ok"] is True
    assert report["records_here"] == 1
    assert report["records_from_rank_files"] == 1
    assert len(report["records"]) == 2
    assert report["records"][0]["where"] == "offer", "newest first"
    from_rank = report["records"][1]
    assert from_rank["rank"] == "PP1_TP0"
    assert from_rank["jobs_needed"] == 13 and from_rank["jobs_fit"] == 11
    assert from_rank["block_provenance"] is not None, (
        "the worker's index is empty by construction; the scheduler fills it in")
    assert from_rank["block_provenance"]["known_to_this_index"] is True
    assert "buffer_cap" in report["kinds"]


def _note_a_local_failure() -> None:
    """One scheduler-side record, so the merge has two sources to order."""
    disk_restore._note_load_failure("buffer_cap", where="offer",
                                    reason="a local refusal", request_id="req-1",
                                    key16=key16_of(2, 0), group=0,
                                    jobs_needed=1, jobs_fit=0)


def test_rank_records_are_validated_before_they_reach_a_payload(armed):
    """A rank file is another process's bytes, so every field is re-read defensively.

    `worker_report()` is written by a worker with its own version of this module, and
    the diagnostics directory is world-writable tmpfs. The validator clips every
    string, re-reads every number, coerces a kind it does not know to `read_failed`
    rather than letting a fourth spelling of "short read" into a counter-aligned
    payload, drops anything not shaped like a record, and caps the list at the
    per-file budget - so a half-written file cannot put a fake reason in front of the
    operator and cannot make this reader raise.
    """
    records = [
        {"kind": "buffer_cap", "where": "bind", "jobs_needed": 13, "jobs_fit": 11,
         "pool_budget_bytes": 128, "key16": "a" * 16, "group": 0},
        {"kind": "nonsense", "jobs_needed": "thirteen"},
        {"kind": "short_read", "key16": "x" * 5000, "group": [1]},
        "not a dict at all",
    ]
    kept = disk_restore._rank_failure_records(records)
    assert len(kept) == disk_restore.MAX_FAILURES_PER_RANK_FILE == 2, kept
    assert kept[0]["kind"] == "buffer_cap" and kept[0]["jobs_needed"] == 13
    assert kept[0]["source"] == "rank file" and kept[0]["block_provenance"] is None
    assert kept[1]["kind"] == "read_failed", "an unknown kind is coerced, not trusted"
    assert kept[1]["kind_as_reported"] == "nonsense"
    assert disk_restore._rank_failure_records("nope") == []
    assert disk_restore._rank_failure_records(None) == []
    assert disk_restore._rank_failure_records([]) == []


def test_the_failures_op_answers_and_names_its_own_limits(armed):
    """`{"op": "failures"}` is how a hunt that cost hours becomes one request.

    Read-only (so a safe method reaches it), bounded by the ring's depth, and honest
    about what the records can and cannot say. The unknown-op suggestion list must
    carry the new name, or a typo'd op reads as "this verb does not know failures"
    rather than as a typo.
    """
    _note_a_local_failure()
    answer = disk_store.disk_for_engine(None, {"op": "failures"})
    assert answer["ok"] is True
    assert answer["records"][0]["kind"] == "buffer_cap"
    assert answer["ring_depth"] == disk_restore.MAX_LOAD_FAILURE_RECORDS
    assert "NOT knowable" in answer["provenance_meaning"]
    assert disk_store.disk_for_engine(None, {"op": "failures", "limit": 1})[
        "records"].__len__() == 1
    assert "failures" in disk_store.DISK_OPS
    refused = disk_store.disk_for_engine(None, {"op": "nope"})
    assert refused["ok"] is False
    assert "failures" in refused["reason"], "a typo names the real op"


def test_the_pool_view_is_a_mapping_and_the_offer_view_keeps_its_shape(armed):
    """A payload field is a CONTRACT: `{...}`, not `[{...}]`.

    The pool view went out once wrapped by a stray comma, and a reader doing
    `payload["offer_pool"]["jobs_the_pool_can_carry"]` - the obvious thing to write,
    and what the units table describes - got a TypeError instead of a number. Cheap
    to pin, and impossible to notice in a terminal that prints the whole dict.
    """
    fill(armed, blocks=2)
    offer(Manager(), Req(blocks=2), 0, (0, False))
    section = disk_restore.section()
    assert isinstance(section["offer_pool"], dict), (
        "a one-element tuple from a trailing comma is not a mapping")
    scheduler = disk_restore.scheduler_section()
    assert isinstance(scheduler["offer_pool"], dict)
    for key in ("bytes_per_job", "price_source", "pool_budget_bytes",
                "jobs_the_pool_can_carry", "jobs_that_fit_now", "capped_offers",
                "refused_offers", "last_offer", "note"):
        assert key in section["offer_pool"], key
        assert key in scheduler["offer_pool"], key
    assert scheduler["last_offer_pool"]["bounded_by"] == []
