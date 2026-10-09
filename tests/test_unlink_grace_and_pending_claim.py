r"""A capacity unlink waits for the clock and for the reader (0.2.62).

WHY THIS FILE EXISTS. Since 0.2.29 the eviction clock is a TOUCH clock: a landed
restore read bumps `st_mtime` (`disk_read.promote_recency`), so an LRU that ranks by
`st_mtime` ranks by last use. That fixed one bug and left two holes, and both are the
subject of this file.

1. THE PASS COULD RACE ITS OWN STORE. A ceiling pass runs at the end of every spill
   (`disk_spill._handle_job` -> `_enforce_ceiling`), so the file that triggered the
   pass was committed seconds earlier - and a pass walks up to
   `disk_layout.MAX_EVICTION_PLAN` (4096) entries and unlinks up to 4096 files. An
   entry written while the pass is walking, or a span whose restore is mid-flight,
   sits in the same list the pass ranked before it knew they existed. Nothing in the
   old code noticed, because "oldest mtime first" has no notion of "too young to
   judge". Now a capacity unlink refuses a file whose touch clock is inside
   `disk_layout.UNLINK_GRACE_SECONDS`, in the planner AND again in the executor - the
   executor's copy matters because the touch can land between the two.

2. NOTHING KNEW A READ WAS IN FLIGHT. `disk_read` promises, on the reader thread, to
   apply bytes it has already queued; `disk_spill` deletes on the writer thread. An
   age window cannot express that promise - a file being read right now may have been
   touched a minute ago (the throttle) or an hour ago (its first service ever). So
   the reader now publishes the claim itself (`disk_read.inflight_paths`), the spill
   pass reads it once per pass, and the executor refuses a claimed path whatever its
   clock says.

WHAT EACH TEST OWNS, in order of what a regression costs:

* the rule at both levels: the planner withholds a young entry and COUNTS it
  (`excluded_grace`), the executor refuses a young path a plan named anyway.
* the pending claim: a path an open batch has promised is refused however old it is,
  and is deleted by the same call with the claim gone - the claim is the cause.
* the exemption: an operator's sweep (`for_capacity=False`) still reclaims a young
  file's bytes. A grace that blocked the sweep is a grace that never lets anything go.
* the end-to-end failure mode: a store of nothing but young files comes up SHORT and
  says so in `ceiling.grace_entries_last_pass`, rather than evicting or spinning.
* `inflight_paths` claims exactly the live jobs (queued/running/done), releases the
  settled ones, and reports the cap it honoured.
* the two cross-module invariants the numbers rest on: the grace fits inside the peer
  liveness window (or a fair-share pass would have nothing it may shed), and it
  exceeds the touch-throttle interval (or it is a second copy of a hole that exists).

No engine, no torch, no network, no sleeping: a `tmp_path` store of real committed
files written through `disk_layout`'s own commit protocol, the real reader threads
where a claim has to be real, and a monkeypatched clock where a wall-clock race would
make a test flaky - the posture of `test_touch_promotion_and_peer_eviction.py`.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_spill,
)

# --- the shapes this test controls -------------------------------------------

SLOT_BYTES = 4096
OLD = 1_600_000_000.0            # an mtime far in the past, as a spill's would be
SHARD = "0a"


def fingerprint(**overrides) -> dict:
    base = {
        "model_id": "flash-next-v2",
        "model_revision": "8f3c1d2e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d",
        "cache_dtype": "auto",
        "dtype": "torch.bfloat16",
        "total_bytes_per_block": SLOT_BYTES,
        "group_block_sizes": [1, 4, 4, 8, 16],
        "scheduler_block_size": 16,
        "hash_block_size": 16,
        "prefix_cacheable_group_ids": [0, 1, 2],
        "prefix_caching_hash_algo": "sha256_cbor",
        "none_hash_seed": "0xd4e5f6a7b8c9d0e1",
        "rank": "PP0_TP0",
        "world_size": 4,
        "dcp_world_size": 1,
        "cp_world_size": 1,
    }
    base.update(overrides)
    return base


def digest_of(**overrides) -> str:
    made = disk_layout.fingerprint_digest(fingerprint(**overrides))
    assert made["ok"], made["reason"]
    return made["geometry_digest"]


OWN_GEO = digest_of()
# AN OLD BOOT's digest. A sweep (`for_capacity=False`) exists to reclaim STALE bytes and
# deliberately refuses this boot's own current file, so every sweep test here needs a
# file the sweep is allowed to want - which is also the honest shape: the only young
# files a sweep is ever pointed at are a previous boot's.


STALE_GEO = digest_of(model_revision="0" * 40)
assert STALE_GEO != OWN_GEO


def slot_key(index: int, group: int = 3) -> str:
    return f"{SHARD}{index:014x}" + "0" * 48 + f"{group:08x}"


def make_file(root, geo, index: int, *, size=SLOT_BYTES, when=OLD,
              group=3) -> str:
    """A committed spill file of `size` bytes with the mtime this test chose."""
    made = disk_layout.build_slot_path(str(root), slot_key(index, group), geo)
    assert made["ok"], made["reason"]
    written = disk_layout.write_committed_file(made["path"], b"x" * size,
                                               sync=False)
    assert written["ok"], written["reason"]
    os.utime(made["path"], (when, when))
    return made["path"]


def entry(path) -> dict:
    """One walk entry, in the shape `lru_eviction_plan` is handed."""
    info = os.stat(path)
    return {"path": str(path), "filename": os.path.basename(str(path)),
            "size_bytes": int(info.st_size), "mtime_ns": int(info.st_mtime_ns)}


def plan_for(entries, *, need_every_byte: bool = True, **kwargs) -> dict:
    """A plan that NEEDS everything, so a withheld entry is visible as a short plan."""
    total = sum(int(item["size_bytes"]) for item in entries)
    return disk_layout.lru_eviction_plan(
        entries, geometry_digest=OWN_GEO,
        ceiling_bytes=max(1, total // 10) if need_every_byte else total * 10,
        used_bytes=total, **kwargs)


@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    """No inherited environment, no leftover reader, no leftover batch."""
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    disk_read.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    yield
    disk_read.stop_reader(timeout_seconds=2.0)
    disk_spill.shutdown_evict_pool()
    disk_read.reset_state_for_tests()
    disk_spill.reset_state_for_tests()


@pytest.fixture
def tier(tmp_path, monkeypatch):
    """A configured spill tier rooted in `tmp_path`, with a journalling channel."""
    root = tmp_path / "kv-store"
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    disk_spill.configure(fingerprint())
    disk_spill.configured()
    disk_spill._state["ceiling_bytes"] = 1
    return {"root": root, "diag": directory}


# ===========================================================================
# 1. THE GRACE, AT BOTH LEVELS
# ===========================================================================

@pytest.mark.parametrize("age_seconds", [0.0, 30.0,
                                          65.0])   # TOUCH + 5: past the throttle
def test_a_file_touched_this_recently_is_not_in_the_destructive_set(tmp_path,
                                                                    age_seconds):
    """The planner's rule, and the three ages that all mean "just now".

    0 s is the spill that triggered this very pass; 30 s is a span written during the
    window a ceiling pass can take; 65 s is the case the TOUCH THROTTLE creates -
    `disk_read.TOUCH_MIN_INTERVAL_SECONDS` (60 s) lets a serving read skip re-stamping
    a file it is using right now, so a file can be in service and still read as
    "touched a minute ago". All three are inside
    `disk_layout.UNLINK_GRACE_SECONDS`, and all three must be withheld rather than
    ranked: the whole point of the touch clock is that a young `st_mtime` means "this
    byte is being used", not "this byte is a fresh mistake".
    """
    now = time.time()
    old = make_file(tmp_path, OWN_GEO, 1, when=OLD)
    young = make_file(tmp_path, OWN_GEO, 2, when=now - age_seconds)
    assert age_seconds < disk_layout.UNLINK_GRACE_SECONDS

    plan = plan_for([entry(old), entry(young)])

    assert plan["ok"] is True, plan["reason"]
    assert plan["paths"] == [str(old)], \
        "the young file is not capacity this pass may take"
    # THE WITHHOLDING IS COUNTED, not swallowed: a reader looking at a plan that came
    # in short has to be able to see that it was the clock, not the classifier.
    assert plan["excluded_grace"] == 1, plan
    assert plan["excluded_grace_bytes"] == SLOT_BYTES, plan
    assert plan["grace_seconds"] == pytest.approx(
        disk_layout.UNLINK_GRACE_SECONDS)


def test_the_grace_is_a_window_and_not_a_ban(tmp_path):
    """The same file, one second outside the window, is capacity again.

    Without this the rule could be "never delete anything younger than a century" and
    every other test here would still pass.
    """
    now = time.time()
    just_outside = make_file(tmp_path, OWN_GEO, 3,
                             when=now - (disk_layout.UNLINK_GRACE_SECONDS + 1.0))
    plan = plan_for([entry(just_outside)])
    assert plan["paths"] == [str(just_outside)], plan
    assert plan["excluded_grace"] == 0, plan

    # `grace_seconds=0` is the documented way to ask for the pre-0.2.62 ranking, and
    # it must return the YOUNGEST file too, not merely a different answer.
    fresh = make_file(tmp_path, OWN_GEO, 4, when=now)
    assert plan_for([entry(fresh)], grace_seconds=0.0)["paths"] == [str(fresh)]
    assert plan_for([entry(fresh)], now_ns=int(now * 1e9))["excluded_grace"] == 1


def shard(path) -> str:
    """The directory the file is DIRECTLY inside - what `execute_unlink_plan` takes.

    The store is sharded (`root/0a/kvslot-...bin`) and the executor's contract is that
    it never walks: it refuses a path that is not directly inside the directory it was
    handed. Every real caller (`disk_spill._unlink_batch`) groups by directory first,
    and a test that passed the store root would be testing the refusal, not the rule.
    """
    return os.path.dirname(str(path))


def claimed(path) -> frozenset:
    """The spelling of a path the executor will compare a claim against."""
    return frozenset({os.path.realpath(str(path))})


def test_the_executor_refuses_a_young_file_a_plan_named_anyway(tmp_path):
    """The executor's OWN copy of the rule, which is the half that closes the race.

    The plan is a snapshot of a moment; the touch clock is not. A file committed after
    the walk - or promoted by a restore read while the unlink batch was being built -
    reaches the executor with a plan line and no protection unless the executor tests
    the clock itself at the instant it is about to call `unlink()`. This is that
    instant, and this test hands it a plan the planner would never have produced.
    """
    young = make_file(tmp_path, OWN_GEO, 5, when=time.time())
    verdict = disk_layout.execute_unlink_plan(
        shard(young), [young], dry_run=False, geometry_digest=OWN_GEO,
        for_capacity=True)

    assert verdict["ok"] is True, verdict["reason"]
    assert verdict["counts"]["unlinked"] == 0, verdict["actions"]
    assert verdict["counts"]["refused"] == 1, verdict["actions"]
    action = verdict["actions"][0]
    assert action["action"] == "refused"
    # A MACHINE-READABLE tag, so the spill writer counts the rule instead of
    # string-matching a sentence that a future rewording would silently break.
    assert action["rule"] == disk_layout.UNLINK_RULE_GRACE, action
    assert os.path.exists(young), "a refused file must still be there"


def test_a_sweep_still_reclaims_a_just_written_files_bytes(tmp_path):
    """THE EXEMPTION. `for_capacity=False` is an operator naming bytes to reclaim.

    The sweep's whole doctrine is "I know these are stale; take them", and its allowed
    classification set is deliberately narrower than the eviction one. A grace that
    bound the sweep would mean a store that can never be emptied on purpose, so the
    rule is scoped to the capacity path and this test is the scoping.
    """
    young = make_file(tmp_path, STALE_GEO, 6, when=time.time())
    verdict = disk_layout.execute_unlink_plan(
        shard(young), [young], dry_run=False, geometry_digest=OWN_GEO,
        for_capacity=False)
    assert verdict["counts"]["unlinked"] == 1, verdict["actions"]
    assert not os.path.exists(young)


# ===========================================================================
# 2. THE PENDING CLAIM
# ===========================================================================

def test_a_path_a_reader_has_promised_is_not_unlinked_however_old(tmp_path):
    """The claim, not the clock: the age rule cannot express this and must not try.

    The file here is ancient on purpose. Every age-based rule in the world would let
    this one go, and it is precisely the file that must survive, because a reader
    thread has already promised the scheduler it will hand back its bytes.
    """
    ancient = make_file(tmp_path, OWN_GEO, 7, when=OLD)
    protected = claimed(ancient)

    refused = disk_layout.execute_unlink_plan(
        shard(ancient), [ancient], dry_run=False, geometry_digest=OWN_GEO,
        for_capacity=True, protected=protected)
    assert refused["counts"]["unlinked"] == 0, refused["actions"]
    assert refused["actions"][0]["rule"] == disk_layout.UNLINK_RULE_CLAIM, \
        refused["actions"]
    assert os.path.exists(ancient)

    # THE CONTROL: the same call, the same plan, no claim - the file goes. Without
    # this line the test above could be passing because of the digest, the classifier
    # or the directory check, and would not be testing the claim at all.
    deleted = disk_layout.execute_unlink_plan(
        shard(ancient), [ancient], dry_run=False, geometry_digest=OWN_GEO,
        for_capacity=True)
    assert deleted["counts"]["unlinked"] == 1, deleted["actions"]
    assert not os.path.exists(ancient)


def _wait_for(predicate, limit=600, pause=0.005):
    """Wait for a reader-thread fact. Never sleeps a fixed amount."""
    for _ in range(limit):
        if predicate():
            return True
        time.sleep(pause)
    return False


def test_a_claim_stops_a_sweep_too(tmp_path):
    """The exemption is about AGE, not about promises.

    An operator who sweeps a store does not intend to yank the floor out from under a
    read that is running. The grace does not bind the sweep; the claim does.
    """
    old = make_file(tmp_path, STALE_GEO, 8, when=OLD)
    verdict = disk_layout.execute_unlink_plan(
        shard(old), [old], dry_run=False, geometry_digest=OWN_GEO,
        for_capacity=False, protected=claimed(old))
    assert verdict["counts"]["unlinked"] == 0, verdict["actions"]
    assert verdict["actions"][0]["rule"] == disk_layout.UNLINK_RULE_CLAIM
    assert os.path.exists(old)


def test_inflight_paths_claims_exactly_the_reads_that_are_live(tier, monkeypatch):
    """`inflight_paths` is the reader's own witness, so its edges are load-bearing.

    A job claims a path from the moment it is queued until its bytes have been applied
    or thrown away. The two shapes a ceiling pass can meet are made real here by
    stalling one read inside the kernel-facing helper on the REAL reader thread: the
    job that is inside the file (`running`) and the job still in the queue behind it
    (`queued`) both claim, and neither of them is the file the pass would most like to
    delete - an LRU ranks the cold first, and a restore is reading what the LRU thought
    was cold.
    """
    root = tier["root"]
    a = make_file(root, OWN_GEO, 9, when=OLD)
    b = make_file(root, OWN_GEO, 10, when=OLD)

    release = threading.Event()
    real_read = disk_read._read_file

    def stalling(path, buffer, size):
        release.wait(10.0)
        return real_read(path, buffer, size)

    monkeypatch.setattr(disk_read, "_read_file", stalling)
    made = disk_read.begin_batch("live", 2)
    assert made["ok"] is True, made["reason"]
    assert disk_read.request_read("live", 0, a, SLOT_BYTES)["ok"] is True
    assert disk_read.request_read("live", 1, b, SLOT_BYTES)["ok"] is True
    assert _wait_for(lambda: disk_read.batch_status("live")["running"] >= 1), \
        disk_read.batch_status("live")

    claim = disk_read.inflight_paths()
    assert claim["count"] == 2, claim
    assert {os.path.realpath(str(a)), os.path.realpath(str(b))} \
        <= set(claim["paths"]), claim
    assert claim["truncated"] is False, claim

    # THE CAP IS THE CALLER'S BOUND, and it must be reported: a ceiling pass may name
    # `MAX_EVICTION_PLAN` paths, so a claim set capped below that is a SUBSET and the
    # reader of the witness has to be able to tell.
    capped = disk_read.inflight_paths(1)
    assert capped["count"] == 1 and capped["truncated"] is True, capped

    # A batch whose caller has taken delivery has no claim left. A claim that outlived
    # its batch would make a file permanently unsheddable - the same bug class as a
    # latched guard, and the reason `queued`, `running` and `done` claim but
    # `failed`/`cancelled` do not.
    release.set()
    assert _wait_for(lambda: disk_read.batch_status("live")["done"] == 2), \
        disk_read.batch_status("live")
    assert disk_read.finish_batch("live")["ok"] is True
    assert disk_read.inflight_paths()["count"] == 0


def test_the_spill_pass_asks_the_reader_once_and_hands_it_to_the_executor(
        tier, monkeypatch):
    """The wiring, in both directions, without trusting either side's comment.

    ONE call per pass, because `inflight_paths` walks every open batch and a pass that
    called it per directory would pay for a witness it already has. And the paths must
    actually REACH the executor - a claim that is read and dropped is a claim that
    protects nothing, which is the failure this whole feature could ship with.
    """
    root = tier["root"]
    for index in (11, 12):
        make_file(root, OWN_GEO, index, when=OLD)
    disk_spill._state["estimated_bytes"] = 4 * SLOT_BYTES

    asked = []
    real_inflight = disk_read.inflight_paths

    def spy_inflight(limit=None):
        made = (real_inflight(limit) if limit is not None else real_inflight())
        asked.append(made)
        return made

    handed = []
    real_execute = disk_layout.execute_unlink_plan

    def spy_execute(directory, paths, **kwargs):
        handed.append(kwargs.get("protected"))
        return real_execute(directory, paths, **kwargs)

    monkeypatch.setattr(disk_read, "inflight_paths", spy_inflight)
    monkeypatch.setattr(disk_layout, "execute_unlink_plan", spy_execute)
    disk_spill._enforce_ceiling()

    assert len(asked) == 1, (
        f"the pass asked the reader for its claims {len(asked)} times; "
        "one pass, one ask")
    assert handed and all(claims is not None for claims in handed), \
        "a batch reached the executor without the claim set attached"
    # THE OBSERVABILITY THE OPERATOR READS: how many paths were claimed when this
    # pass ran, and whether that set was capped.
    section = disk_spill.section()["ceiling_measurement"]
    assert section["claim_paths_last_pass"] == 0, section
    assert section["claim_set_truncated"] is False, section


# ===========================================================================
# 3. THE END-TO-END FAILURE MODE
# ===========================================================================

def test_a_store_of_nothing_but_young_files_comes_up_short_and_says_so(tier):
    """The pass must refuse to evict AND say why, in one line and in the counters.

    This is the shape the grace creates on a box whose whole store was written in the
    last ninety seconds - a rank that has just booted and spilled hard, which is also
    the moment a ceiling pass is most likely to fire. The correct behaviour is not to
    delete the young files and not to spin: it is to come up over the ceiling, mark the
    pass short, and publish the withholding so the reader can tell "the grace saved
    these" from "the classifier found nothing".
    """
    root = tier["root"]
    now = time.time()
    young = [make_file(root, OWN_GEO, index, when=now)
             for index in range(1, 4)]
    disk_spill._state["estimated_bytes"] = 3 * SLOT_BYTES

    disk_spill._enforce_ceiling()

    assert all(os.path.exists(path) for path in young), \
        "a capacity pass deleted bytes the grace says are in use"
    section = disk_spill.section()["ceiling_measurement"]
    assert section["grace_entries_last_pass"] == 3, section
    assert section["grace_bytes_last_pass"] == 3 * SLOT_BYTES, section
    assert section["grace_entries_excluded"] >= 3, section
    assert section["grace_seconds"] == pytest.approx(
        disk_layout.UNLINK_GRACE_SECONDS), section
    # NOTHING WAS FREED, and the tier says so rather than reporting success: the
    # eviction counter and the removal journal both stayed at zero.
    assert disk_spill.section()["evictions"] == 0, disk_spill.section()["evictions"]
    assert disk_spill.section()["journal_removals"] == 0


# ===========================================================================
# 4. THE ARITHMETIC, AS AN ASSERTION
# ===========================================================================

def test_the_grace_fits_inside_the_peer_liveness_window():
    """`UNLINK_GRACE_SECONDS` < `PEER_DIGEST_FRESH_SECONDS`, and why that is a rule.

    Four ranks share one root, and the fair-share machinery sheds a live sibling's
    stale bytes last but must be able to shed them AT ALL. A digest counts as a live
    sibling while its files are inside `PEER_DIGEST_FRESH_SECONDS`; if the grace
    reached that far or further, every file that still PROVES a peer is alive would be
    unsheddable, and a pass on a shared root would have nothing it is allowed to
    delete. The 90/120 pairing leaves a 30-second band that is both live and
    sheddable - which is the band `test_ceiling_fair_share_multi_rank` writes its
    fixtures in, and this is the assertion that keeps that fixture honest.
    """
    assert disk_layout.UNLINK_GRACE_SECONDS > 0.0
    assert disk_layout.UNLINK_GRACE_SECONDS < disk_spill.PEER_DIGEST_FRESH_SECONDS, (
        f"grace {disk_layout.UNLINK_GRACE_SECONDS}s must stay under the "
        f"{disk_spill.PEER_DIGEST_FRESH_SECONDS}s peer-liveness window")


def test_the_grace_outlives_the_throttle_it_is_there_to_cover():
    """The window must exceed the throttle's blind spot, not duplicate it.

    `TOUCH_MIN_INTERVAL_SECONDS` is the hole: a serving read inside the interval does
    NOT re-stamp the file, so the clock under-reports how recently a byte was used.
    A grace at or under that interval would protect exactly the files the touch clock
    already protects and nothing else - the change would be a no-op wearing a name.
    """
    assert disk_layout.UNLINK_GRACE_SECONDS > disk_read.TOUCH_MIN_INTERVAL_SECONDS, (
        f"grace {disk_layout.UNLINK_GRACE_SECONDS}s does not exceed the "
        f"{disk_read.TOUCH_MIN_INTERVAL_SECONDS}s touch throttle")
    # And it must cover a restore's own deadline: a file named by an offer can be
    # handed to a worker up to `RESTORE_DEADLINE_SECONDS` later, twice over.
    assert disk_layout.UNLINK_GRACE_SECONDS >= 2 * 45.0, (
        "the grace must span an offer plus one full load lifetime")


def test_a_capacity_pass_and_a_sweep_disagree_about_the_clock_by_design(tmp_path):
    """The two rules, side by side on one file, so the split cannot drift.

    Same young file, same executor, same second: the capacity pass refuses it, the
    sweep deletes it. If someone later "simplifies" this by gating both, or neither,
    this test names which one moved.
    """
    young = make_file(tmp_path, OWN_GEO, 13, when=time.time())
    capacity = disk_layout.execute_unlink_plan(
        shard(young), [young], dry_run=False, geometry_digest=OWN_GEO,
        for_capacity=True)
    assert capacity["actions"][0]["rule"] == disk_layout.UNLINK_RULE_GRACE
    assert os.path.exists(young)

    # The same clock, on bytes a sweep is ALLOWED to want: a previous boot's file,
    # written this second. The sweep takes it; the capacity pass would not.
    restamped = make_file(tmp_path, STALE_GEO, 14, when=time.time())
    sweep = disk_layout.execute_unlink_plan(
        shard(restamped), [restamped], dry_run=False, geometry_digest=OWN_GEO,
        for_capacity=False)
    assert sweep["counts"]["unlinked"] == 1, sweep["actions"]
