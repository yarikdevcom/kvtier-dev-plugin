r"""FOUR RANKS, ONE SHARED CEILING: the pass must converge store-wide (0.2.30).

WHY THIS FILE EXISTS, IN THE OPERATOR'S NUMBERS. On the live box on the night of
2026-10-22 a four-rank boot sharing ONE spill root wrote 92 files each and evicted
2,106 / 2,106 / 1,872 / 1,872 files - 7,956 files, 156 GiB down to 68 GiB - under a
148 GiB ceiling whose `EVICT_TARGET_FRACTION` of 0.9 made the one-pass target 133 GiB.
One pass should have shed ~23 GiB. FOUR passes shed ~23 GiB EACH, because every rank
compared the WHOLE STORE's bytes against the WHOLE CEILING and each then acted on the
same directory. Nothing miscounted; the effective capacity of that store was
`ceiling / rank_count`, and the more ranks a boot has, the more of its own cache it
destroys.

THE SHAPE OF THIS TEST IS THE INTERLEAVING, NOT THE AVERAGE. Run four ranks
SEQUENTIALLY against a live walk and the bug hides: the first rank sheds, the second
walks a store that is already smaller, and everyone converges. The incident needed the
thing that actually happens under concurrency - every rank measures the store BEFORE
any of them has shed, and then all four act on that stale view. So the test is two
phases:

  PHASE A  every rank walks the pre-shed store and learns which sibling digests are
           live. Read-only, exactly as production's read-only walk is.
  PHASE B  every rank runs its enforcement pass against the snapshot it took in phase
           A. The unlinks are real files, and the file each rank deletes is a file the
           OTHER ranks still believe exists - which is the race.

WHAT IS ASSERTED, and which one the defect fails:

  1. the store ends at or under `0.9 x ceiling` (a capacity assertion, which the defect
     PASSES - a store can be too small and still be "under the ceiling"),
  2. the store does NOT end absurdly far under the target, and the bytes freed are
     within ~1x of the overshoot rather than N times it. THE DEFECT FAILS HERE: it
     freed ~4x what the overshoot was, and left a tenth of the intended cache behind.
     This is the assertion that would have caught 2026-10-22.
  3. a second round of passes declines and deletes nothing: convergence, and the
     `ceiling_passes_declined` witness ("one rank did the work, three declined") that
     makes the fix observable live instead of reconstructed from four rank files.
  4. no rank deletes a live sibling's file - the property 0.2.29 started and 0.2.30
     makes structural (a live peer's entries leave the destructive set entirely).

THE SINGLE-RANK MIRROR at the end is the other half of the brief: with one rank and no
peers, the share IS the whole ceiling, and the pass must shed EXACTLY what every
release before this one shed. The fix must not make a lone rank too timid.

Real files built by `disk_layout`'s own naming and commit protocol; the real walk, the
real planner and the real unlinker. No engine, no torch, no network, no sleeping, and
the store is a `tmp_path` - nothing here touches a serving box.
"""

from __future__ import annotations

import copy
import os
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

SLOT_BYTES = 102_400           # one spill file: a whole KiB-multiple, easy to sum
RANKS = 4
FILES_PER_RANK = 40
STORE_BYTES = RANKS * FILES_PER_RANK * SLOT_BYTES          # 16,384,000
CEILING = 14_000_000                                       # under the store on purpose
TARGET = int(CEILING * disk_spill.EVICT_TARGET_FRACTION)   # 12,600,000
# 3,784,000: the bytes ONE pass owes. Four ranks sharing the work owe this ONCE.
OVERSHOOT = STORE_BYTES - TARGET
SHARE = CEILING // RANKS                                   # 3,500,000
SHARE_TARGET = int(SHARE * disk_spill.EVICT_TARGET_FRACTION)


def fingerprint(rank: str, **overrides) -> dict:
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
        "rank": rank,
        "world_size": 4,
        "dcp_world_size": 1,
        "cp_world_size": 1,
    }
    base.update(overrides)
    return base


def digest_of(rank: str) -> str:
    made = disk_layout.fingerprint_digest(fingerprint(rank))
    assert made["ok"], made["reason"]
    return made["geometry_digest"]


# The four ranks of one PP2xTP2 boot. Different `rank` means a different geometry
# digest, which is how a shared root can tell whose file is whose - and it is the ONLY
# thing that can, because the directory is shared and the digest is in the filename.
RANK_IDS = ("PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1")
DIGESTS = {rank: digest_of(rank) for rank in RANK_IDS}
assert len(set(DIGESTS.values())) == RANKS, "ranks must have distinct digests"


def slot_key(index: int, group: int = 3) -> str:
    return f"{index:016x}" + "0" * 48 + f"{group:08x}"


def make_file(root, geo, index: int, *, size=SLOT_BYTES, when=None) -> str:
    """A committed spill file, with an mtime this test chooses."""
    made = disk_layout.build_slot_path(str(root), slot_key(index), geo)
    assert made["ok"], made["reason"]
    written = disk_layout.write_committed_file(made["path"], b"x" * size, sync=False)
    assert written["ok"], written["reason"]
    stamp = time.time() if when is None else when
    os.utime(made["path"], (stamp, stamp))
    return made["path"]


def share_view() -> dict:
    """This process's `ceiling_share` block, or an empty one.

    Deliberately tolerant, and the reason is the regression demonstration: on the code
    before 0.2.30 the block does not exist, and this file must then fail on the BYTES
    ("four passes freed four times the overshoot"), not on a missing key. A test that
    only proves a field is absent is a test that cannot have caught an incident.
    """
    return dict(disk_spill.section().get("ceiling_share") or {})


def store_bytes(root) -> int:
    """WHAT IS ACTUALLY ON DISK. The assertion number, never a module's estimate."""
    total = 0
    files = 0
    for directory, _subdirs, names in os.walk(str(root)):
        for name in names:
            if not name.startswith("kvslot-") or not name.endswith(".bin"):
                continue
            try:
                total += int(os.stat(os.path.join(directory, name)).st_size)
            except OSError:
                continue     # deleted between the walk and the stat: it counted as gone
            files += 1
    store_bytes.files = files
    return total


def digest_of_file(path) -> str:
    parsed = disk_layout.parse_path(str(path))
    assert parsed["ok"], parsed["reason"]
    return str(parsed["geometry_digest"])


def survivors_by_digest(root) -> dict:
    out = {digest: 0 for digest in DIGESTS.values()}
    for directory, _subdirs, names in os.walk(str(root)):
        for name in names:
            if not name.startswith("kvslot-") or not name.endswith(".bin"):
                continue
            try:
                digest = digest_of_file(os.path.join(directory, name))
            except AssertionError:
                continue
            if digest in out:
                out[digest] += 1
    return out


def boot_rank(rank: str, root, ceiling: int) -> None:
    """Make `disk_spill` THIS rank: its digest, its root, its ceiling.

    The same three fields a worker process gets from its own fingerprint and its own
    environment; four of them share one `root`, which is the entire point.
    """
    disk_spill.reset_state_for_tests()
    disk_spill.configure(fingerprint(rank))
    disk_spill._state["root"] = str(root)
    disk_spill._state["ceiling_bytes"] = int(ceiling)


def measure(rank: str, root, ceiling: int) -> dict:
    """PHASE A: this rank walks the store as it is NOW and learns its live siblings.

    Read-only. The returned dict is the rank's VIEW, which is what it will still be
    holding when it comes to decide - because under concurrency no rank sees another's
    deletions before it makes its own.
    """
    boot_rank(rank, root, ceiling)
    walk = disk_spill._walk_store(str(root))
    assert walk["ok"], walk["reason"]
    assert walk["truncated"] is False, walk["measurement_basis"]
    peers = disk_spill._live_peer_digests()
    return {"walk": walk, "peers": peers,
            "sightings": copy.deepcopy(disk_spill._peer_sightings)}


def enforce_from_view(rank: str, root, ceiling: int, view: dict) -> dict:
    """PHASE B: this rank enforces the ceiling using the view it took in phase A.

    `_walk_store` is replaced by the captured snapshot - that substitution IS the
    interleaving, and everything downstream of it (`_attribute_walk_bytes`, the share,
    `lru_eviction_plan`, `execute_unlink_plan`) is the production code, on real files.
    The substitution is undone before returning, so a later round of this same test
    walks the store that the passes actually left behind instead of a stale snapshot
    that a leaked patch would have handed it.
    """
    boot_rank(rank, root, ceiling)
    disk_spill._peer_sightings.clear()
    disk_spill._peer_sightings.update(copy.deepcopy(view["sightings"]))
    # A rank that has been writing all boot carries a running estimate; seeding it is
    # what lets the trigger fire, exactly as a real boot's does.
    disk_spill._state["estimated_bytes"] = int(view["walk"]["used_bytes"])
    original = disk_spill._walk_store
    disk_spill._walk_store = lambda *a, **k: view["walk"]
    try:
        disk_spill._enforce_ceiling()
    finally:
        disk_spill._walk_store = original
    return share_view()


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_store(tmp_path, monkeypatch):
    """One shared root, no inherited environment, no leftover state between tests."""
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    for name in (config.ENV_DISK_ROOT, config.ENV_DISK_BYTES,
                 config.ENV_DIAG_DIR, config.ENV_DIAG_RANK):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "kv-store"
    root.mkdir()
    disk_spill.reset_state_for_tests()
    disk_read.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    disk_spill.shutdown_evict_pool()
    yield root
    disk_spill.shutdown_evict_pool()
    disk_spill.reset_state_for_tests()
    disk_read.reset_state_for_tests()
    disk_index.reset_state_for_tests()


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """A diagnostics directory + rank, so a removal CAN be journalled."""
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    return directory


def build_shared_store(root) -> dict:
    """Four ranks' files in ONE directory: 160 committed files, 16,384,000 bytes.

    Written fresh, because peer-liveness is `PEER_DIGEST_FRESH_SECONDS`-bounded and a
    rank whose sibling stopped writing 20 minutes ago is correctly NOT a live sibling.
    """
    paths = {}
    index = 0
    for rank in RANK_IDS:
        paths[rank] = []
        for _ in range(FILES_PER_RANK):
            index += 1
            paths[rank].append(make_file(root, DIGESTS[rank], index))
    assert store_bytes(root) == STORE_BYTES, store_bytes(root)
    return paths


# ===========================================================================
# THE DEFECT: four passes over the overshoot, paid four times
# ===========================================================================

def test_four_ranks_measuring_one_store_converge_to_one_target(clean_store, journal):
    """The 2026-10-22 incident, replayed on 160 files instead of 29,532.

    Read the assertions in order of what a regression costs, and note that the FIRST
    one does not catch the defect: a store can be far too small and still be "under the
    ceiling". The one that catches it is the second: the bytes freed must be about the
    overshoot, not N times it.
    """
    root = clean_store
    paths = build_shared_store(root)

    # PHASE A: nobody has shed anything, and all four measure the whole store.
    views = {rank: measure(rank, root, CEILING) for rank in RANK_IDS}
    for rank, view in views.items():
        assert int(view["walk"]["used_bytes"]) == STORE_BYTES, rank
        # Every rank sees its three siblings alive: `n_effective` is 4, and that is the
        # rank count the share divides by. It comes from the walk the pass already does.
        assert len(view["peers"]) == RANKS - 1, (rank, view["peers"])

    before = store_bytes(root)
    # PHASE B: all four act on that stale view, one after another, as a real boot's
    # four writer threads do at almost the same instant.
    shares = {rank: enforce_from_view(rank, root, CEILING, views[rank])
              for rank in RANK_IDS}
    after = store_bytes(root)

    # (1) CAPACITY: the store is under the target. The old code passes this too.
    assert after <= TARGET, f"{after} > target {TARGET}"

    # (2) THE ONE THAT WOULD HAVE CAUGHT THE NIGHT. One pass owes OVERSHOOT bytes
    # (3,784,000). Four ranks sharing the work owe the SAME overshoot - not four of
    # them. The pre-0.2.30 code frees ~4x here and leaves a tenth of the cache.
    freed = before - after
    assert freed <= 2 * OVERSHOOT, (
        f"the four passes together freed {freed} bytes for an overshoot of "
        f"{OVERSHOOT}: that is {freed / OVERSHOOT:.1f}x the work one pass owed, which "
        f"is the 2026-10-22 defect (it freed 4.0x and took the store to 68 GiB under a "
        f"133 GiB target)")
    assert after >= TARGET - 2 * OVERSHOOT, (
        f"the store ended at {after} against a target of {TARGET}: a store this far "
        "under the target is over-eviction whatever the ceiling says")

    # (3) EVERY RANK LANDED ON ITS OWN SHARE, so the sum lands on the store's target.
    # This is the arithmetic the fix rests on: N x (0.9 x C/N) = 0.9 x C.
    for rank, share in shares.items():
        assert share.get("share_divisor") == RANKS, (rank, share)
        assert share.get("share_bytes") == SHARE, (rank, share)
        assert share.get("own_bytes") <= SHARE_TARGET, (rank, share)
    counts = survivors_by_digest(root)
    for rank in RANK_IDS:
        assert counts[DIGESTS[rank]] <= SHARE_TARGET // SLOT_BYTES, (
            rank, counts[DIGESTS[rank]])

    # (4) NOBODY SHED A LIVE SIBLING'S FILE. 0.2.29 ranked them last; 0.2.30 removes
    # them from the destructive set, so this is structural rather than an ordering.
    for rank in RANK_IDS:
        survivors = [path for path in paths[rank] if os.path.exists(path)]
        assert len(survivors) >= counts[DIGESTS[rank]], rank
        assert counts[DIGESTS[rank]] == len(survivors), (
            f"{rank}: {counts[DIGESTS[rank]]} file(s) survive under its digest and "
            f"{len(survivors)} of the files IT wrote are on disk - a rank deleted a "
            "sibling's bytes and counted them under its own digest")
    for rank in RANK_IDS:
        assert shares[rank].get("peer_entries_excluded", 0) > 0, (
            rank, shares[rank].get("peer_entries_excluded"))


def test_the_second_round_declines_and_deletes_nothing(clean_store, journal):
    """Convergence, and the witness that says ONE RANK DID THE WORK AND THREE DECLINED.

    Each rank still carries the estimate it had before its siblings shed, so every one
    of them still believes it is over its share and runs a pass. The pass re-measures,
    finds its own bytes already inside its share, and does nothing - counted, not
    logged, because it runs on the spill writer thread.
    """
    root = clean_store
    build_shared_store(root)
    views = {rank: measure(rank, root, CEILING) for rank in RANK_IDS}
    for rank in RANK_IDS:
        enforce_from_view(rank, root, CEILING, views[rank])
    settled = store_bytes(root)
    assert settled <= TARGET, settled

    # Round two: same stale belief in the estimate, but the pass now WALKS (no
    # substitution), so it sees the store its siblings have already shed.
    declines = {}
    for rank in RANK_IDS:
        boot_rank(rank, root, CEILING)
        disk_spill._peer_sightings.clear()
        disk_spill._peer_sightings.update(copy.deepcopy(views[rank]["sightings"]))
        # THE POINT OF THE ROUND: a stale, too-big estimate still triggers a pass (the
        # fail-safe direction: a pass that runs and declines is a pass that cost a
        # walk), and the walk is what stops it deleting anything.
        disk_spill._state["estimated_bytes"] = STORE_BYTES
        disk_spill._enforce_ceiling()
        share = share_view()
        declines[rank] = share
        assert share.get("passes_declined", 0) >= 1, (rank, share)
        assert (share.get("declined_accountable_bytes") or 0) <= (
            share.get("declined_target_bytes") or 0), (rank, share)
    assert store_bytes(root) == settled, (
        "a round-two pass deleted bytes after the store had already converged: the "
        "decline is supposed to be the whole action of that pass")

    # Four declines is the signature the operator asked to be able to see live.
    assert all(row["passes_declined"] >= 1 for row in declines.values()), declines
    for row in declines.values():
        assert row.get("peer_live_bytes", 0) > 0, row
        assert (row.get("store_estimate_bytes") or 0) <= TARGET + OVERSHOOT, row


def test_a_lone_rank_still_sheds_to_nine_tenths_of_the_whole_ceiling(clean_store,
                                                                     journal):
    """THE SINGLE-RANK MIRROR. One rank, no peers: the share IS the whole ceiling.

    This is the no-regression half of the brief, stated as a byte count rather than as
    a flag. A rank with no live sibling must shed EXACTLY what every release before
    0.2.30 shed - `0.9 x ceiling` of the store it measured - because the fail-safe
    direction has to be "sheds like before", never "too timid to protect the disk".
    """
    root = clean_store
    # The whole store, one writer: no sibling to share the ceiling with.
    lone = [make_file(root, DIGESTS["PP0_TP0"], index)
            for index in range(1, FILES_PER_RANK * RANKS + 1)]
    assert store_bytes(root) == STORE_BYTES

    boot_rank("PP0_TP0", root, CEILING)
    view = measure("PP0_TP0", root, CEILING)
    assert view["peers"] == (), view["peers"]      # nobody else is writing
    disk_spill._state["estimated_bytes"] = STORE_BYTES
    original = disk_spill._walk_store
    disk_spill._walk_store = lambda *a, **k: view["walk"]
    try:
        disk_spill._enforce_ceiling()
    finally:
        disk_spill._walk_store = original

    share = share_view()
    assert share.get("share_divisor") == 1, share
    assert share.get("share_bytes") == CEILING, share     # the WHOLE ceiling, as before
    assert share.get("target_bytes") == TARGET, share
    assert share.get("share_source") == disk_spill.SHARE_SOURCE_NO_PEERS, share
    after = store_bytes(root)
    assert after <= TARGET, after
    # And not absurdly far under: the old code's target was `TARGET`, and it still is.
    assert after > TARGET - 2 * SLOT_BYTES, after
    assert len([path for path in lone if os.path.exists(path)]) == store_bytes(root) \
        // SLOT_BYTES


def test_the_share_arithmetic_never_hands_out_more_than_the_ceiling():
    """`_share_of_ceiling` is pure, so its floor and its edge cases are pinnable.

    A share rounded UP would make N ranks' combined budget larger than the configured
    ceiling, which is a quieter version of the bug this release removes.
    """
    for divisor in (1, 2, 3, 4, 8):
        share = disk_spill._share_of_ceiling(CEILING, divisor, "test")
        assert share["divisor"] == divisor
        assert share["share_bytes"] * divisor <= CEILING, (divisor, share)
        assert share["share_bytes"] == CEILING // divisor, (divisor, share)
    one = disk_spill._share_of_ceiling(CEILING, 1, disk_spill.SHARE_SOURCE_NO_PEERS)
    assert one["share_bytes"] == CEILING and one["target_bytes"] == TARGET, one
    # A ceiling smaller than the rank count is a misconfiguration, and 1 byte is a
    # budget: 0 would mean "delete the store", which is not a fail-safe direction.
    tiny = disk_spill._share_of_ceiling(3, 4, "test")
    assert tiny["share_bytes"] == 1, tiny
    assert tiny["target_bytes"] >= 0, tiny


def test_the_trigger_fires_no_later_than_the_store_wide_rule_ever_did(clean_store,
                                                                      journal):
    """Capacity safety cannot regress: the old clause is still INSIDE the trigger.

    `_ceiling_trigger` ORs the new fair share with the pre-0.2.30 store-wide clause, so
    a pass can only ever START sooner than every earlier release did, never later. A
    rank under its share but over the whole ceiling must still be told to run.
    """
    root = clean_store
    build_shared_store(root)
    boot_rank("PP0_TP0", root, CEILING)
    # Attributed, and comfortably inside a share of 3.5 MB ...
    disk_spill._state["bytes_attributed"] = True
    disk_spill._state["estimated_own_bytes"] = 1_000_000
    disk_spill._state["estimated_foreign_dead_bytes"] = 0
    disk_spill._state["estimated_unattributed_bytes"] = 0
    disk_spill._state["share_bytes"] = SHARE
    disk_spill._state["share_target_bytes"] = SHARE_TARGET
    disk_spill._state["estimated_peer_live_bytes"] = STORE_BYTES - 1_000_000
    # ... while the STORE is over the whole ceiling, which is 4 ranks having overrun.
    disk_spill._state["estimated_bytes"] = CEILING + 1
    trigger = disk_spill._ceiling_trigger()
    assert trigger["over_share"] is False, trigger
    assert trigger["over_store_ceiling"] is True, trigger
    assert trigger["run"] is True, trigger
    # Under both, the pass does not run at all - the decline is a pass that RAN.
    disk_spill._state["estimated_bytes"] = 1_000_000
    idle = disk_spill._ceiling_trigger()
    assert idle["run"] is False and "under" in idle["reason"], idle
