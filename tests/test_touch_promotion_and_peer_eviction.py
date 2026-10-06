r"""The eviction clock is a TOUCH clock, a peer rank's live file is not capacity,
and the ceiling pass has a deadline and a pool of hands (0.2.29).

WHY THIS FILE EXISTS. The operator was about to run a multi-hour fill test against
the spill store's 192 GiB ceiling and asked two questions: is the disk cache an LRU,
and does eviction block. The source said NO and YES, in three separate ways.

1. THE CLOCK ONLY SAW WRITES. Eviction ranked by `st_mtime`, and nothing moved
   `st_mtime` but a spill. A span restored a hundred times was evicted by a span
   written an hour later and never read - so the tier shed exactly the bytes its own
   restore path had proven valuable. The policy the operator asked for is "most read
   + most written, i.e. most touched, has priority", so a read that produced usable
   bytes now bumps the mtime (`disk_read.promote_recency`), throttled, and the single
   existing clock keeps working with no new durable state to replay after a restart.

2. EVERY RANK SHED ITS SIBLINGS' LIVE DATA FIRST. Four ranks share one spill root,
   and `lru_eviction_plan` was called WITHOUT `peer_digests` even though it accepts
   them. A foreign digest classifies as `stale_geometry` - "pure capacity, shed
   first" - so each rank's cheapest file to delete was one of its three peers' live,
   loadable, hottest spans. Under a fill test that is four evictors destroying each
   other in a loop. A rank now LEARNS its siblings' digests from the walk it already
   performs (a foreign digest with committed files being written now is a rank that
   is alive) and such files shed LAST.

3. THE PASS HAD NO DEADLINE AND ONE HAND. Up to 100,000 entry reads twice, 20,000
   `stat` calls and 4,096 unlinks, serially, on the spill writer thread - the only
   thing draining a 256-job / 256 MiB queue, at the one moment the disk is fullest.
   The pass now has a time budget it stops at (partial by the existing
   `plan_truncated` rule, so the next spill continues) and it can hand its unlinks to
   a small worker pool. The pool defaults to ONE worker because the measurement in
   the release notes says four parallel `unlink()`s are ~38% SLOWER on this
   filesystem: the mechanism ships, the default does not pretend.

WHAT EACH TEST OWNS, in order of what a regression costs:

* promotion moves the clock, is throttled, is bounded, and is FAIL-CLOSED on the
  verdict: a `file_missing` / `size_mismatch` / `short_read` / `not_ours` /
  `read_error` read must never keep a bad file alive.
* the operator's own request as a behaviour: an older file that was READ survives a
  newer file that was only WRITTEN. That one fails without Item 1.
* a known peer's live file is not the first thing shed; and when the peer digest
  cannot be confirmed, today's behaviour is preserved and COUNTED (the fail-safe
  direction is "sheds like before", never "cannot evict").
* the deadline truncates, releases bytes, marks itself, and never half-applies a
  plan or strands a deleted file without its journal record.
* the pool changes which thread unlinks and nothing else: same files, same bytes,
  same journalled removals at 1 worker and at 4, and no path attempted twice.

No engine, no torch, no network, no sleeping: a `tmp_path` store of real committed
files built with `disk_layout`'s own naming and commit protocol, the real reader
thread, and monkeypatched clocks where a wall-clock race would make a test flaky -
the same posture `test_disk_ceiling_walk.py` and `test_disk_read_threads.py` use.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import pytest

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_restore,
    disk_spill,
)

# --- the shapes this test controls -------------------------------------------

SLOT_BYTES = 4096
PP1_BYTES = 12_902_400
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
# A SIBLING RANK's digest: same model, same dtype, a different rank and slot width.
# This is exactly what a peer writes into the same shared root, and exactly what this
# boot used to call `stale_geometry` and delete first.
PEER_GEO = digest_of(rank="PP1_TP0", total_bytes_per_block=PP1_BYTES)
assert PEER_GEO != OWN_GEO


def slot_key(index: int, group: int = 3) -> str:
    """A group-qualified block hash whose `key16` lands in `SHARD`."""
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


def mtime_ns(path) -> int:
    return int(os.stat(path).st_mtime_ns)


def survivors(paths) -> list:
    return [str(path) for path in paths if os.path.exists(path)]


def wait_for(predicate, limit=600, pause=0.005):
    for _ in range(limit):
        if predicate():
            return True
        time.sleep(pause)
    return False


def journal_lines(directory) -> list:
    where = disk_index.journal_path()
    assert where["ok"] is True, where["reason"]
    if not os.path.exists(where["path"]):
        return []
    with open(where["path"], "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def key16_of(path) -> str:
    """The identity a journal line carries for this file (the name's `key16`)."""
    parsed = disk_layout.parse_path(str(path))
    assert parsed["ok"], parsed["reason"]
    return parsed["key16"]


def journalled_removals(directory) -> list:
    """The `key16`s the removal journal names, newest entry last.

    A journal line is `<seq> <op> <key16> <size> <group>`: the index maps a KV
    identity to a file, so it names the key and not the path, and a test that compared
    paths would be testing the wrong contract.
    """
    out = []
    for line in journal_lines(directory):
        parts = line.split(" ")
        if len(parts) > 2 and parts[1] == "-":
            out.append(parts[2])
    return out


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_tier(tmp_path, monkeypatch):
    """No inherited environment, no leftover reader, no leftover evict pool.

    Both new env vars are listed in `conftest.ENV_VARS`, but this fixture is still
    explicit about the pools: a reader or an evict worker that survives a test mutates
    the NEXT test's counters, which is the failure `test_disk_read_threads.py` already
    documents in its own teardown.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    disk_read.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    yield
    _stop_readers()
    disk_spill.shutdown_evict_pool()
    disk_read.reset_state_for_tests()
    disk_spill.reset_state_for_tests()


def _stop_readers() -> None:
    """Drain and stop every reader this test started. Best-effort, never hangs."""
    pending = disk_read._queue
    readers = [thread for thread in threading.enumerate()
               if str(thread.name).startswith("kvtier-disk-read")]
    if pending is not None:
        while True:
            try:
                pending.get_nowait()
            except queue.Empty:
                break
        for _ in readers:
            try:
                pending.put_nowait(disk_read._STOP)
            except queue.Full:
                break
    for thread in readers:
        thread.join(3.0)


@pytest.fixture
def channel(tmp_path, monkeypatch):
    """A diagnostics directory + rank, so a removal CAN be journalled."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    return directory


@pytest.fixture
def tier(tmp_path, monkeypatch):
    """A configured spill tier rooted in `tmp_path`, ceiling set by the test."""
    root = tmp_path / "kv-store"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    disk_spill.configure(fingerprint())
    disk_spill.configured()
    return {"root": root}


def set_ceiling(bytes_value: int) -> None:
    """Point `_state` at a ceiling, the way `configured()` does from the environment."""
    disk_spill._state["ceiling_bytes"] = int(bytes_value)


# ===========================================================================
# 1. A READ THAT LANDED MOVES THE CLOCK (Item 1)
# ===========================================================================

def test_a_read_that_landed_moves_the_eviction_clock(tmp_path):
    """The hook, end to end: real reader thread, real file, real mtime.

    The call chain under test is `_reader_loop` -> `_handle_read` ->
    `promote_recency`, which is a reader-thread chain: `get_finished`, the scheduler
    step and every engine thread are simply not on it. That is the property this test
    exists for, so it drives the real thread rather than calling the helper.
    """
    path = make_file(tmp_path, OWN_GEO, 1, when=OLD)
    before = mtime_ns(path)
    assert disk_read.begin_batch("promote", 1)["ok"] is True
    assert disk_read.request_read("promote", 0, path, SLOT_BYTES)["ok"] is True
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 1), \
        disk_read.batch_status("promote")
    disk_read.finish_batch("promote")

    after = mtime_ns(path)
    assert after > before, "a landed read must move the eviction clock"
    # THE CLOCK THAT OWNS st_mtime IS WALL CLOCK. Writing the monotonic throttle
    # clock into the file would date every promoted file to 1970 and evict the
    # hottest spans FIRST - an inversion no counter would reveal.
    assert after > int((time.time() - 120.0) * 1e9), \
        f"the promoted mtime is not wall-clock now: {after}"
    state = disk_read.counters()
    assert state["recency_promotions"] == 1
    assert state["recency_promotions_throttled"] == 0
    assert state["recency_promotion_failures"] == 0
    assert state["recency_promotion_map_entries"] == 1


def test_a_second_read_inside_the_window_does_not_touch_the_file_again(tmp_path,
                                                                       monkeypatch):
    """The throttle: one metadata write per file per interval, not one per read.

    The clock is monkeypatched rather than waited out - a suite that sleeps 60 s to
    prove a 60 s window is a suite that never runs.
    """
    monkeypatch.setattr(disk_read, "_touch_clock", lambda: 1_000.0)
    path = make_file(tmp_path, OWN_GEO, 2, when=OLD)

    assert disk_read.begin_batch("first", 1)["ok"] is True
    assert disk_read.request_read("first", 0, path, SLOT_BYTES)["ok"] is True
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 1)
    disk_read.finish_batch("first")
    first_touch = mtime_ns(path)
    assert first_touch > int(OLD * 1e9)

    time.sleep(0.002)      # above the filesystem's timestamp granularity, not the
                           # throttle window: this only makes a stray touch visible.
    assert disk_read.begin_batch("second", 1)["ok"] is True
    assert disk_read.request_read("second", 0, path, SLOT_BYTES)["ok"] is True
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 2)
    disk_read.finish_batch("second")

    state = disk_read.counters()
    assert state["reads_completed"] == 2, "both reads must have landed"
    assert state["recency_promotions"] == 1
    assert state["recency_promotions_throttled"] == 1
    assert mtime_ns(path) == first_touch, \
        "the second read inside the window must not write the file's metadata again"


def test_the_window_elapses_and_the_file_is_touched_again(tmp_path, monkeypatch):
    """The throttle DELAYS a promotion; it never vetoes one. A box read an hour later
    is a box that must not be evicted by one read an hour before it."""
    clock = {"now": 1_000.0}
    monkeypatch.setattr(disk_read, "_touch_clock", lambda: clock["now"])
    path = make_file(tmp_path, OWN_GEO, 3, when=OLD)

    verdict = disk_read.promote_recency(path)
    assert verdict == disk_read.TOUCH_PROMOTED
    first = mtime_ns(path)
    clock["now"] += disk_read.TOUCH_MIN_INTERVAL_SECONDS + 1.0
    time.sleep(0.002)
    assert disk_read.promote_recency(path) == disk_read.TOUCH_PROMOTED
    assert mtime_ns(path) > first
    assert disk_read.counters()["recency_promotions"] == 2


@pytest.mark.parametrize("kind", ["file_missing", "size_mismatch", "short_read",
                                  "not_ours", "read_error"])
def test_a_read_that_did_not_land_never_promotes(tmp_path, monkeypatch, kind):
    """FAIL-CLOSED ON THE VERDICT: keeping a wrong-sized or unreadable file alive is
    strictly worse than evicting it, so only an `ok` outcome reaches the promotion."""
    path = make_file(tmp_path, OWN_GEO, 4, when=OLD)
    before = mtime_ns(path)
    monkeypatch.setattr(disk_read, "_read_file",
                        lambda *_a, **_k: disk_read._failed(kind, "test verdict"))

    assert disk_read.begin_batch(f"bad-{kind}", 1)["ok"] is True
    assert disk_read.request_read(f"bad-{kind}", 0, path, SLOT_BYTES)["ok"] is True
    assert wait_for(lambda: disk_read.batch_status(
        f"bad-{kind}")["failed"] == 1 and not disk_read.batch_status(
            f"bad-{kind}")["pending"]), disk_read.batch_status(f"bad-{kind}")
    disk_read.finish_batch(f"bad-{kind}")

    state = disk_read.counters()
    assert state["reads_completed"] == 0
    assert state["failed"][kind] == 1
    assert state["recency_promotions"] == 0, f"{kind} must not promote"
    assert state["recency_promotions_throttled"] == 0
    assert mtime_ns(path) == before, f"a {kind} read moved the eviction clock"


def test_the_promotion_map_is_bounded_and_never_grows_past_its_cap(tmp_path):
    """The throttle map is a throttle, not a second unbounded structure.

    Paths need not exist: the map records the ATTEMPT before the syscall, which is
    what makes a failing path cost one write per interval instead of one per read.
    """
    cap = disk_read.MAX_TOUCH_TRACKED
    assert cap > 16
    for index in range(cap + 500):
        missing = os.path.join(str(tmp_path),
                               f"kvslot-{index:016x}-g3-abcdef123456.bin")
        assert disk_read.promote_recency(missing) in (disk_read.TOUCH_FAILED,
                                                      disk_read.TOUCH_PROMOTED)
    assert len(disk_read._touch_map) <= cap, \
        f"the throttle map grew to {len(disk_read._touch_map)} past its cap {cap}"
    state = disk_read.counters()
    assert state["recency_promotion_map_entries"] <= cap
    assert state["recency_promotion_failures"] > 0, \
        "a path with no file behind it is counted, not silently dropped"
    # And the map still throttles AFTER being saturated: the newest entry is in it.
    newest = os.path.join(str(tmp_path),
                          f"kvslot-{cap + 499:016x}-g3-abcdef123456.bin")
    assert disk_read.promote_recency(newest) == disk_read.TOUCH_THROTTLED


def test_promotion_refuses_a_part_file_and_a_name_that_is_not_ours(tmp_path):
    """The second lock on the door: never move the clock of a file whose bytes are not
    a committed block this plugin wrote."""
    path = make_file(tmp_path, OWN_GEO, 5, when=OLD)
    assert disk_read.promote_recency(path + disk_layout.SUFFIX_PART) \
        == disk_read.TOUCH_SKIPPED
    foreign = tmp_path / "notes.txt"
    foreign.write_text("someone else's file")
    before = mtime_ns(foreign)
    assert disk_read.promote_recency(foreign) == disk_read.TOUCH_SKIPPED
    assert mtime_ns(foreign) == before
    assert disk_read.promote_recency(None) == disk_read.TOUCH_SKIPPED
    assert disk_read.counters()["recency_promotions"] == 0


def test_promotion_surfaces_where_the_read_counters_already_live():
    """A feature that cannot be witnessed during the experiment did not happen."""
    state = disk_read.counters()
    for field in ("recency_promotions", "recency_promotions_throttled",
                  "recency_promotion_failures", "recency_promotion_map_entries",
                  "recency_promotion_interval_seconds",
                  "recency_promotion_map_cap"):
        assert field in state, field
    section = disk_read.section()
    assert section["recency_promotion_interval_seconds"] \
        == disk_read.TOUCH_MIN_INTERVAL_SECONDS
    assert section["recency_promotion_map_cap"] == disk_read.MAX_TOUCH_TRACKED
    assert disk_read.TOUCH_MIN_INTERVAL_SECONDS >= 10.0, \
        "an interval below a few tens of seconds is a promotion per read again"


# ===========================================================================
# 2. THE OPERATOR'S REQUEST, AS A BEHAVIOUR (Item 1's point)
# ===========================================================================

def test_the_span_that_was_read_survives_the_span_that_was_only_written(tier,
                                                                         channel):
    """THE regression the whole change exists for.

    Two files under THIS boot's digest. `read` was written first, then restored (its
    mtime is now); `written_later` was written second and never read. The old clock
    said `read` was the coldest file in the store and shed it; the touch clock says
    `written_later` is.
    """
    root = tier["root"]
    read_first = make_file(root, OWN_GEO, 11, when=OLD)
    written_later = make_file(root, OWN_GEO, 12, when=OLD + 100_000.0)
    assert mtime_ns(read_first) < mtime_ns(written_later)
    colder_than = mtime_ns(written_later)      # the pass deletes it; remember the clock

    # The read that changes the answer, on the real reader thread.
    assert disk_read.begin_batch("touch", 1)["ok"] is True
    assert disk_read.request_read("touch", 0, read_first, SLOT_BYTES)["ok"] is True
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 1), \
        disk_read.batch_status("touch")
    disk_read.finish_batch("touch")

    set_ceiling(9_000)                     # target 8,100 of 8,192 used: one file goes
    disk_spill._state["estimated_bytes"] = 9_000
    disk_spill._enforce_ceiling()

    # THE BEHAVIOUR FIRST, with no reference to any new field: pre-0.2.29 this fails
    # here, because the file the tier sheds is the one a restore had just read.
    assert not os.path.exists(written_later), \
        "the file that was only ever WRITTEN is the one the LRU must shed"
    assert os.path.exists(read_first), \
        "the file a restore just read must survive: recency means last TOUCH"

    # ...and the witnesses that say it happened the way the change says it did.
    assert disk_read.counters()["recency_promotions"] == 1
    assert mtime_ns(read_first) > colder_than, \
        "the touch must have moved the file's clock PAST the file that wins the old " \
        "clock - read after the pass, because the pass deletes the other file"
    assert disk_spill._state["journal_failures"] == 0
    assert journalled_removals(channel) == [key16_of(written_later)]


# ===========================================================================
# 3. A PEER RANK'S LIVE FILE IS NOT CAPACITY (Item 2)
# ===========================================================================

def test_the_planner_sheds_our_own_files_before_a_named_peer_and_says_so():
    """The comparator, as a pure function: naming a peer digest moves that digest's
    files from "shed first" to "shed last", and the plan reports the difference."""
    rows = [
        {"path": f"/store/{SHARD}/ours.bin",
         "filename": disk_layout.build_slot_name(slot_key(21), OWN_GEO)["filename"],
         "size_bytes": SLOT_BYTES, "mtime_ns": int(OLD * 1e9)},
        {"path": f"/store/{SHARD}/peer.bin",
         "filename": disk_layout.build_slot_name(slot_key(22), PEER_GEO)["filename"],
         "size_bytes": SLOT_BYTES, "mtime_ns": int((OLD + 100_000.0) * 1e9)},
    ]
    blind = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                          ceiling_bytes=9_000, used_bytes=8_192)
    assert blind["planned"] == 1
    assert blind["paths"] == ["/store/0a/peer.bin"], \
        "with no peer digests the foreign one still goes first (fail-safe)"
    assert blind["stale_first"] == 1
    assert blind["peer_rank"] == 0

    informed = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                             ceiling_bytes=9_000, used_bytes=8_192,
                                             peer_digests=(PEER_GEO,))
    assert informed["planned"] == 1
    assert informed["paths"] == ["/store/0a/ours.bin"], \
        "a live sibling's file must not be preferred over this boot's own"
    assert informed["stale_first"] == 0
    assert informed["peer_rank"] == 0


def test_a_known_peers_live_file_is_not_the_first_thing_shed(tier, channel):
    """The four-rank mutual-destruction loop, stopped.

    Two committed files under a SIBLING's digest, written seconds ago (so the walk can
    see that rank is alive), and one of OURS, written long ago. Old behaviour: the
    peer files are `stale_geometry` - "pure capacity" - and go first. New behaviour:
    this boot sheds its own cold file and leaves a sibling's hot data alone.
    """
    root = tier["root"]
    now = time.time()
    ours = make_file(root, OWN_GEO, 31, when=now - 10_000.0)
    peer_a = make_file(root, PEER_GEO, 32, when=now - 1.0, group=4)
    peer_b = make_file(root, PEER_GEO, 33, when=now - 2.0, group=5)

    set_ceiling(12_000)                    # 12,288 used, target 10,800: one file goes
    disk_spill._state["estimated_bytes"] = 12_000
    disk_spill._enforce_ceiling()

    # THE BEHAVIOUR FIRST: pre-0.2.29 the two sibling files are `stale_geometry` -
    # "pure capacity, shed first" - and this boot's own cold file survives. That is
    # the mutual-destruction loop, and it fails HERE, on a filesystem, not on a field.
    assert not os.path.exists(ours), "this boot's own cold file is the one to shed"
    assert os.path.exists(peer_a) and os.path.exists(peer_b), \
        "a live sibling rank's data must not be the cheapest thing to delete"

    peers = disk_spill.section()["peer_ranks"]
    assert peers["live_digests"] == 1, peers
    assert peers["learned_digests"] == 1, peers
    assert peers["files_planned_last_pass"] == 0, \
        "the plan had a live peer's files available and must not have chosen one"
    assert journalled_removals(channel) == [key16_of(ours)]


def test_a_digest_that_cannot_be_confirmed_keeps_todays_behaviour_and_is_counted(
        tier, channel):
    """THE FAIL-SAFE DIRECTION, pinned.

    A foreign digest nobody can confirm is treated exactly as 0.2.28 treated it: shed
    first. Being unable to identify a sibling must NEVER mean being unable to release
    capacity - the store still has to come down under its ceiling - and the fact that
    this rank was blind is COUNTED rather than left as a silence.
    """
    root = tier["root"]
    now = time.time()
    ours = make_file(root, OWN_GEO, 41, when=now - 10_000.0)
    # Old enough that "somebody is writing this digest now" is not a claim the walk
    # can honestly make: an hour, against a 120 s freshness window.
    dead = make_file(root, PEER_GEO, 42, when=now - 3_600.0, group=4)
    dead_b = make_file(root, PEER_GEO, 43, when=now - 3_601.0, group=5)

    set_ceiling(12_000)
    disk_spill._state["estimated_bytes"] = 12_000
    disk_spill._enforce_ceiling()

    peers = disk_spill.section()["peer_ranks"]
    assert peers["live_digests"] == 0, peers
    assert peers["decisions_without_peers"] >= 1, \
        "a ceiling decision taken blind to its peers must be counted"
    assert peers["foreign_files_seen"] >= 2, peers

    assert not os.path.exists(dead) or not os.path.exists(dead_b), \
        "an unconfirmable foreign digest still sheds first: the old behaviour"
    assert os.path.exists(ours), \
        "and it is still NOT this boot's own file that pays for being blind"


def test_a_single_sighting_is_not_enough_to_call_a_digest_live(tier, channel):
    """One stray recent file is an anomaly, not a fleet member.

    Without the two-sighting floor a single fresh file left behind by some other boot
    would make its whole digest un-sheddable-as-stale, which is the direction that can
    stop a ceiling from being a ceiling.
    """
    root = tier["root"]
    now = time.time()
    ours = make_file(root, OWN_GEO, 51, when=now - 10_000.0)
    stray = make_file(root, PEER_GEO, 52, when=now - 1.0, group=4)

    set_ceiling(8_500)                     # 8,192 used... under the 7,650 target? no:
    disk_spill._state["estimated_bytes"] = 8_500   # 8,192 > 7,650, so exactly one goes
    disk_spill._enforce_ceiling()

    assert disk_spill.section()["peer_ranks"]["live_digests"] == 0
    assert os.path.exists(ours), "the stray foreign file is still the cheapest shed"
    assert not os.path.exists(stray)


# ===========================================================================
# 4. THE PASS HAS A DEADLINE (Item 3)
# ===========================================================================

def _arm_the_deadline_after_the_first_batch(monkeypatch):
    """Make the pass's budget run out exactly once ONE batch has landed.

    Timing a real unlink loop against a wall clock would be a test that fails on a
    busy CI box and passes on an idle one (this repo already refuses loop-jitter
    dependencies in its timing tests), so the budget's ANSWER is what gets driven.
    The budget mechanism itself - "past the deadline means expired" - is pinned next
    to it by `test_the_deadline_predicate`, and the env plumbing by
    `test_the_budget_comes_from_the_environment_in_the_existing_idiom`.
    """
    state = {"armed": False, "batches": []}
    monkeypatch.setattr(disk_spill, "_deadline_expired",
                        lambda deadline: state["armed"])

    def spy(directory, paths, peers):
        state["batches"].append(list(paths))
        state["armed"] = True
        return disk_layout.execute_unlink_plan(
            directory, paths, dry_run=False,
            geometry_digest=disk_spill._state["geometry_digest"],
            peer_digests=peers, for_capacity=True)

    monkeypatch.setattr(disk_spill, "_unlink_batch", spy)
    return state


def test_the_deadline_predicate_bounds_a_pass_and_none_never_expires():
    assert disk_spill._deadline_expired(None) is False
    assert disk_spill._deadline_expired(time.monotonic() + 3600.0) is False
    assert disk_spill._deadline_expired(time.monotonic() - 1.0) is True


def test_the_budget_comes_from_the_environment_in_the_existing_idiom(monkeypatch):
    """Unset = the shipped default; a typo = the default plus a complaint; 0 = refused.
    """
    monkeypatch.delenv(config.ENV_DISK_CEILING_BUDGET_SECONDS, raising=False)
    assert config.ceiling_env()["budget_seconds"] is None
    assert disk_spill._ceiling_budget_seconds()["seconds"] \
        == disk_spill.CEILING_PASS_BUDGET_SECONDS
    assert disk_spill.CEILING_PASS_BUDGET_SECONDS <= 2.0, \
        "the writer thread's default budget must be short enough to keep the queue"

    monkeypatch.setenv(config.ENV_DISK_CEILING_BUDGET_SECONDS, "0.25")
    assert config.ceiling_env()["budget_seconds"] == 0.25
    assert disk_spill._ceiling_budget_seconds()["seconds"] == 0.25

    # A budget of 0 would make every pass do nothing while the ceiling reads as
    # configured, so it is refused at the grammar and the default stands.
    monkeypatch.setenv(config.ENV_DISK_CEILING_BUDGET_SECONDS, "0")
    refused = config.ceiling_env()
    assert refused["budget_seconds"] is None and refused["problems"]
    assert disk_spill._ceiling_budget_seconds()["seconds"] \
        == disk_spill.CEILING_PASS_BUDGET_SECONDS
    monkeypatch.setenv(config.ENV_DISK_CEILING_BUDGET_SECONDS, "not-a-number")
    assert config.ceiling_env()["problems"]
    # An absurd value is a VISIBLE clamp, never a silently-ignored setting.
    monkeypatch.setenv(config.ENV_DISK_CEILING_BUDGET_SECONDS, "1e9")
    big = disk_spill._ceiling_budget_seconds()
    assert big["clamped"] is True
    assert big["seconds"] == disk_spill.CEILING_BUDGET_SECONDS_MAX


def test_a_pass_out_of_time_stops_early_and_still_frees_bytes(tier, channel,
                                                              monkeypatch):
    """Truncated, marked, and NOT zero-progress.

    A budget that stopped the pass before it deleted anything would be worse than no
    budget: the only thing that re-runs the pass is the next committed spill, arriving
    into a queue that is still full. So the first batch always runs.
    """
    root = tier["root"]
    dirs = {}
    for shard in range(6):
        paths = []
        for index in range(4):
            key = f"{shard:02x}{index:014x}" + "0" * 48 + "00000003"
            # THE LAYOUT IS `build_slot_path`'s, including the shard sub-directory:
            # a store this test hand-nested would not be the store the walk reads.
            made = disk_layout.build_slot_path(str(root), key, OWN_GEO)
            assert made["ok"], made["reason"]
            written = disk_layout.write_committed_file(made["path"],
                                                       b"x" * SLOT_BYTES, sync=False)
            assert written["ok"], written["reason"]
            os.utime(made["path"], (OLD + index, OLD + index))
            paths.append(made["path"])
        dirs[f"{shard:02x}"] = paths

    total = sum(len(paths) for paths in dirs.values()) * SLOT_BYTES
    set_ceiling(1_000)                     # everything is over the target
    disk_spill._state["estimated_bytes"] = total
    watched = _arm_the_deadline_after_the_first_batch(monkeypatch)

    disk_spill._enforce_ceiling()

    assert len(watched["batches"]) == 1, \
        f"the pass kept submitting after its budget: {len(watched['batches'])}"
    measurement = disk_spill.section()["ceiling_measurement"]
    assert measurement["plan_truncated"] is True
    assert measurement["deadline_hit"] is True
    assert measurement["deadline_hits"] == 1
    assert measurement["pass_seconds"] >= 0.0
    assert disk_spill._state["evicted_bytes"] == SLOT_BYTES * 4, \
        "the first batch's bytes must be released and accounted exactly once"
    assert disk_spill._state["evict_files_unlinked"] == 4

    # The other five directories are untouched, and untouched WHOLE: truncation
    # happens between whole per-directory batches, never inside one.
    for shard, paths in dirs.items():
        gone = [path for path in paths if not os.path.exists(path)]
        if shard == watched["batches"][0][0].split(os.sep)[-2]:
            assert len(gone) == 4, f"{shard} was half-applied: {len(gone)} of 4 gone"
        else:
            assert not gone, f"{shard} was deleted by a pass that had run out of time"


def test_the_next_spill_continues_a_truncated_pass_without_double_deleting(
        tier, channel, monkeypatch):
    """Idempotence across passes: no path twice, no half file, no stranded deletion.

    Every file that vanished is named exactly once in the removal journal, across BOTH
    passes - which is the property that makes a truncated pass safe rather than a way
    to lose a file the index still offers.
    """
    root = tier["root"]
    all_paths = []
    for shard in range(4):
        for index in range(3):
            key = f"{shard:02x}{index:014x}" + "0" * 48 + "00000003"
            made = disk_layout.build_slot_path(str(root), key, OWN_GEO)
            written = disk_layout.write_committed_file(made["path"],
                                                       b"x" * SLOT_BYTES, sync=False)
            assert written["ok"], written["reason"]
            os.utime(made["path"], (OLD + index, OLD + index))
            all_paths.append(made["path"])

    total = len(all_paths) * SLOT_BYTES
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = total
    watched = _arm_the_deadline_after_the_first_batch(monkeypatch)

    # The budget is re-armed before every pass: each spill starts a fresh budget, and
    # the point of the test is that THREE successive truncated passes make progress
    # without ever deleting the same file twice.
    for _ in range(3):
        watched["armed"] = False
        disk_spill._enforce_ceiling()

    removals = journalled_removals(channel)
    assert len(removals) == len(set(removals)), \
        f"a file was journalled as removed twice: {removals}"
    assert len(watched["batches"]) == 3
    attempted = [path for batch in watched["batches"] for path in batch]
    assert len(attempted) == len(set(attempted)), "a path was attempted by two batches"
    vanished = [path for path in all_paths if not os.path.exists(path)]
    assert sorted(key16_of(path) for path in vanished) == sorted(removals), \
        "every deletion must have exactly one journal record, and vice versa"
    assert disk_spill._state["evicted_bytes"] == len(vanished) * SLOT_BYTES
    assert disk_spill._state["journal_failures"] == 0


def test_a_tiny_real_budget_truncates_the_walk_and_deletes_nothing(tier, monkeypatch):
    """The deadline is real, not just a flag: an unreachable budget stops the WALK.

    This is the shape that must NOT do damage: a walk that measured nothing must not
    write a 0 into the estimate (that is how a ceiling goes quiet forever), so the
    pass declines to plan and the next spill asks again.
    """
    root = tier["root"]
    for index in range(8):
        make_file(root, OWN_GEO, 60 + index, when=OLD + index)
    monkeypatch.setenv(config.ENV_DISK_CEILING_BUDGET_SECONDS, "1e-9")
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = 8 * SLOT_BYTES
    before = int(disk_spill._state["estimated_bytes"])

    disk_spill._enforce_ceiling()

    measurement = disk_spill.section()["ceiling_measurement"]
    assert measurement["deadline_hit"] is True
    assert measurement["walk_deadline_hit"] is True
    assert disk_spill._state["evictions"] == 0
    assert disk_spill._state["evicted_bytes"] == 0
    assert len(list(root.rglob("*.bin"))) == 8
    assert int(disk_spill._state["estimated_bytes"]) == before, \
        "an unmeasured walk must not write a floor into the estimate"


# ===========================================================================
# 5. THE UNLINK POOL (Item 4)
# ===========================================================================

def _pooled_store(root, shards=5, per_shard=4):
    """A store spread over `shards` shard directories, oldest-index-first mtimes.

    `build_slot_path` owns the shard sub-directory (it is the first byte of the key),
    so the test hands it the ROOT and gets the layout the capacity walk actually reads.
    """
    all_paths = []
    for shard in range(shards):
        for index in range(per_shard):
            key = f"{shard:02x}{index:014x}" + "0" * 48 + "00000003"
            made = disk_layout.build_slot_path(str(root), key, OWN_GEO)
            assert made["ok"], made["reason"]
            written = disk_layout.write_committed_file(made["path"],
                                                       b"x" * SLOT_BYTES, sync=False)
            assert written["ok"], written["reason"]
            os.utime(made["path"], (OLD + index, OLD + index))
            all_paths.append(made["path"])
    return all_paths


@pytest.mark.parametrize("workers", [1, 4])
def test_the_pool_changes_which_thread_unlinks_and_nothing_else(tier, channel,
                                                                monkeypatch,
                                                                workers):
    """Correctness must not depend on the thread count: same files, same bytes, same
    journal. That is the whole argument for shipping the mechanism with a default of
    one worker."""
    monkeypatch.setenv(config.ENV_DISK_EVICT_THREADS, str(workers))
    paths = _pooled_store(tier["root"])
    set_ceiling(6_000)                     # target 5,400: shed until <= 1 file remains
    disk_spill._state["estimated_bytes"] = 20 * SLOT_BYTES

    disk_spill._enforce_ceiling()

    left = survivors(paths)
    assert len(left) == 1, "one file must survive a pass that sheds to the target"
    assert disk_spill._state["evicted_bytes"] == 19 * SLOT_BYTES
    assert disk_spill._state["evictions"] == 19
    assert disk_spill._state["evict_files_unlinked"] == 19
    assert disk_spill._state["evict_files_failed"] == 0
    assert disk_spill._state["journal_failures"] == 0
    removals = journalled_removals(channel)
    assert len(removals) == 19
    assert sorted(removals) == sorted(key16_of(path) for path in paths
                                      if str(path) not in left)
    pool = disk_spill.section()["evict_pool"]
    assert pool["evict_threads_configured"] == workers
    assert pool["evict_threads_requested"] == workers
    assert pool["pool_fallbacks"] == 0


@pytest.mark.parametrize("workers", [1, 4])
def test_no_path_is_ever_handed_to_two_workers(tier, monkeypatch, workers):
    """The plan is PARTITIONED, not shared. Groups are disjoint by construction
    (one per shard directory), which is what makes `evicted_bytes` unable to
    double-count regardless of how many threads run."""
    monkeypatch.setenv(config.ENV_DISK_EVICT_THREADS, str(workers))
    _pooled_store(tier["root"], shards=6, per_shard=5)
    seen: list = []
    real = disk_layout.execute_unlink_plan

    def spy(directory, paths, **kwargs):
        seen.extend(list(paths))
        return real(directory, paths, **kwargs)

    monkeypatch.setattr(disk_layout, "execute_unlink_plan", spy)
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = 30 * SLOT_BYTES

    disk_spill._enforce_ceiling()

    assert seen, "the pass must have unlinked something"
    assert len(seen) == len(set(seen)), \
        f"a path was handed to two workers: {len(seen)} attempts over " \
        f"{len(set(seen))} paths"
    # Every attempt came from exactly one directory batch, and the batches are whole.
    by_dir: dict = {}
    for path in seen:
        by_dir.setdefault(os.path.dirname(path), []).append(path)
    assert len(by_dir) >= 2, "the store must span enough directories to fan out"


def test_the_serial_default_reports_one_worker_and_no_threads(tier):
    """1 is the shipped default, and it is distinguishable from "the pool failed"."""
    pool = disk_spill.section()["evict_pool"]
    assert pool["evict_threads_configured"] == 0, "no pool before the first pass"
    assert disk_spill.EVICT_THREADS_DEFAULT == 1, \
        "parallel unlink measured SLOWER on this filesystem; the default stays serial"
    assert disk_spill.MAX_EVICT_THREADS >= 4

    _pooled_store(tier["root"], shards=2, per_shard=2)
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = 4 * SLOT_BYTES
    disk_spill._enforce_ceiling()
    serial = disk_spill.section()["evict_pool"]
    assert serial["evict_threads_configured"] == 1
    assert serial["evict_threads_alive"] == 0, "the serial path builds no threads"
    assert serial["evict_threads_requested"] == 1
    assert serial["batches_serial"] >= 1
    assert serial["batches_submitted"] == 0
    assert serial["pool_fallbacks"] == 0
    assert not any(str(thread.name).startswith(disk_spill.EVICT_THREAD_PREFIX)
                   for thread in threading.enumerate()), \
        "a default boot must not spawn eviction threads it never uses"


def test_a_configured_pool_builds_named_workers_once(tier, monkeypatch):
    """The pool is built on first use, reports its count, and names its workers the

    way the readers do - so "did parallel eviction run?" is one field and one `ps`.
    """
    monkeypatch.setenv(config.ENV_DISK_EVICT_THREADS, "4")
    _pooled_store(tier["root"], shards=3, per_shard=2)
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = 6 * SLOT_BYTES
    disk_spill._enforce_ceiling()

    pooled = disk_spill.section()["evict_pool"]
    assert pooled["evict_threads_requested"] == 4
    assert pooled["evict_threads_configured"] == 4
    assert pooled["evict_threads_alive"] >= 1, "a built pool must report its workers"
    assert pooled["batches_submitted"] >= 1
    assert pooled["batches_serial"] == 0
    assert any(str(thread.name).startswith(disk_spill.EVICT_THREAD_PREFIX)
               for thread in threading.enumerate()), \
        "the workers must be identifiable by name, as the readers are"
    # ONE pool per process: a second pass reuses it rather than rebuilding it.
    before = pooled["evict_threads_alive"]
    _pooled_store(tier["root"], shards=3, per_shard=1)
    disk_spill._state["estimated_bytes"] = 6 * SLOT_BYTES
    disk_spill._enforce_ceiling()
    assert disk_spill.section()["evict_pool"]["evict_threads_alive"] == before


def test_the_deadline_still_governs_a_pooled_pass_and_await_what_it_submitted(
        tier, channel, monkeypatch):
    """The ordering rule, as a behaviour: stop SUBMITTING, never abandon.

    A submitted-but-uncollected unlink is a deleted file the journal never heard
    about, which is the one index error this tier rates worse than a recompute. So the
    whole in-flight wave is awaited and collected even when the budget expired
    mid-wave, and the truncation only refuses to hand out the NEXT wave.
    """
    monkeypatch.setenv(config.ENV_DISK_EVICT_THREADS, "4")
    root = tier["root"]
    all_paths = _pooled_store(root, shards=7, per_shard=3)
    watched = {"batches": 0, "armed": False}
    real = disk_layout.execute_unlink_plan

    def spy(directory, paths, **kwargs):
        watched["batches"] += 1
        watched["armed"] = True          # the budget runs out during the first wave
        return real(directory, paths, **kwargs)

    monkeypatch.setattr(disk_layout, "execute_unlink_plan", spy)
    monkeypatch.setattr(disk_spill, "_deadline_expired",
                        lambda deadline: watched["armed"])
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = len(all_paths) * SLOT_BYTES

    disk_spill._enforce_ceiling()

    assert watched["batches"] == 4, \
        "the whole submitted wave must be awaited, and no more than one wave run"
    measurement = disk_spill.section()["ceiling_measurement"]
    assert measurement["plan_truncated"] is True
    assert measurement["deadline_hit"] is True
    pool = disk_spill.section()["evict_pool"]
    assert pool["batches_submitted"] == 4
    assert pool["files_unlinked"] == 12          # 4 whole directories, none half-done

    removals = journalled_removals(channel)
    vanished = [str(path) for path in all_paths if not os.path.exists(path)]
    assert sorted(key16_of(path) for path in vanished) == sorted(removals), \
        "a truncated pooled pass must journal every deletion it made, exactly once"
    assert disk_spill._state["evicted_bytes"] == 12 * SLOT_BYTES


def test_a_batch_that_could_not_run_is_counted_and_never_journalled(tier, channel,
                                                                    monkeypatch):
    """A worker that loses its batch reports a batch that did nothing: no freed bytes,
    no removal journal entry, and a counter that says so."""
    monkeypatch.setenv(config.ENV_DISK_EVICT_THREADS, "4")
    paths = _pooled_store(tier["root"], shards=2, per_shard=2)

    def broken(_directory, _paths, _peers):
        return None

    monkeypatch.setattr(disk_spill, "_unlink_batch", broken)
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = 4 * SLOT_BYTES

    disk_spill._enforce_ceiling()

    assert survivors(paths) == [str(path) for path in paths]
    assert disk_spill._state["evicted_bytes"] == 0
    assert disk_spill._state["evictions"] == 4, \
        "the files were attempted, which is what `evictions` has always counted"
    assert journalled_removals(channel) == []
    assert disk_spill._state["evict_files_failed"] >= 1
    assert disk_spill._state["last_reason"]


# ===========================================================================
# 6. THE WORKER'S OWN REPORT (what actually reaches the proxy host)
# ===========================================================================

def test_the_rank_report_carries_the_promotion_and_the_pool_and_never_a_placeholder(
        tier, channel, monkeypatch):
    """The rank file is the ONLY worker-side witness on the proxy host, so a field
    that is a constant there is worse than no field: it reads as a measurement.

    `evict.deadline_hits` was a literal `0` while this was written, and it is the
    exact shape of bug this pins - every new concurrency/recency field is asserted to
    MOVE, not merely to exist.
    """
    import json

    from kvtier_dev_plugin import worker_diag

    quiet = disk_restore.worker_report()
    assert quiet["readers"]["recency_promotions"] == 0
    # `configured` is 0 until a pass has BUILT a pool, and that is the honest
    # reading: the operator's ask lives beside it in `spill.section()` as
    # `evict_threads_requested`, so the pair distinguishes "asked for 4, nothing has
    # run yet" from "running 4".
    assert quiet["evict"] == {"configured": 0, "alive": 0, "batches_submitted": 0,
                              "files_unlinked": 0, "files_failed": 0,
                              "pool_fallbacks": 0, "deadline_hits": 0}, \
        "a boot that has never run a pass reports an untouched pool"

    # A promotion, then a deadline-truncated pass, and the report must show both.
    path = make_file(tier["root"], OWN_GEO, 21, when=OLD)
    assert disk_read.begin_batch("rep", 1)["ok"] is True
    assert disk_read.request_read("rep", 0, path, SLOT_BYTES)["ok"] is True
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 1)
    disk_read.finish_batch("rep")
    assert disk_restore.worker_report()["readers"]["recency_promotions"] == 1

    _pooled_store(tier["root"], shards=3, per_shard=2)
    armed = _arm_the_deadline_after_the_first_batch(monkeypatch)
    set_ceiling(1_000)
    disk_spill._state["estimated_bytes"] = 6 * SLOT_BYTES
    disk_spill._enforce_ceiling()
    assert armed["armed"] is True, "the pass must have run out of time"

    moved = disk_restore.worker_report()
    assert moved["evict"]["deadline_hits"] >= 1, \
        "the report must carry the boot's real deadline count, not a placeholder"
    assert moved["evict"]["files_unlinked"] == disk_spill.counters()[
        "evict_files_unlinked"]
    assert moved["evict"]["configured"] == 1, \
        "the serial path counts as one configured worker once a pass has run"
    assert moved.get("truncated") is None, moved
    raw = json.dumps(moved, default=str)
    assert len(raw) < worker_diag.MAX_CONTRIBUTION_BYTES, \
        f"the contribution grew to {len(raw)} of " \
        f"{worker_diag.MAX_CONTRIBUTION_BYTES} bytes: a section that overflows is " \
        f"REPLACED by the minimal shape, which would silently lose every field " \
        f"added here"
