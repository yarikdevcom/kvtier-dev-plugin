r"""SILENCE IS NOT DEATH - but a DEAD WORKER IS, and an orphaned digest is.

THE OPERATOR'S RULE THIS FILE EXISTS FOR.

    "remove expiration time on disk cache etc etc, so it stays infinite till lru CACHED"

A block written to the disk store persists INDEFINITELY and may be discarded only by
LRU-by-touch eviction under the byte quota. No age rule may delete cached bytes, and no
age rule may refuse to serve bytes that are on disk. The store's own data files already
obeyed that (`st_mtime` is a touch clock; the ceiling is the only deleter) and the
0.2.34
gate already refused to shed an UNATTRIBUTABLE name first. What both of them left open
was
the map that decides WHOSE bytes those are: it was aged by write traffic, so a rank that
merely had no request for two minutes lost its claim and its live, restorable KV became
the cheapest capacity on the box.

THE DEFECT, AS THE CODE READS BEFORE THE FIX. Four ranks share one spill root and each
rank's files carry its own geometry digest (`rank` and `total_bytes_per_block` are
fingerprint fields, so PP0_TP0, PP0_TP1, PP1_TP0 and PP1_TP1 compute four different
digests for one model). Rank X's evictor learned its siblings from the walk it already
does: `_observe_peer_digest` records a foreign digest from the `st_mtime` of its
committed
files, `_live_peer_digests()` believed a digest only while it had
`PEER_DIGEST_MIN_SIGHTINGS` files written inside `PEER_DIGEST_FRESH_SECONDS` (120 s) and
then FORGOT the ones that aged out. So:

  * rank X has been idle for two minutes (no request touched its prefix, so no spill),
  * rank Y is busy, so the peer map is still CONFIRMED - just without X,
  * `disk_layout.attribute_name` then called X's committed files `ATTRIBUTION_DEAD`
    ("positively nobody's"), which `SHED_ORDER_BY_CLASSIFICATION` puts at bucket 0: the
    FIRST thing unlinked, charged against every rank's fair share.

Live production, one boot: `peer_files_removed = 2955` / `38.1 GB` on each PP1 rank with
`peer_files_removed_known_peer = 0`.

THE FIX, AND WHY IT IS NOT "never delete a foreign digest". `attribute_name` keeps the
`stale_geometry` / `stale_key_shape` reclaim path, because that path is load-bearing: on
2026-10-07 the operator moved `KV_OFFLOAD_SIZE` 64 GiB -> 8 GiB, which moved the slot
width, which moved the geometry digest - so EVERY pre-change file on the store now names
a
digest no current rank owns and is unservable forever. Those bytes are correct to
reclaim,
and a rank that could not reclaim them would watch the store fill with unservable,
unreclaimable data until the ceiling pass refused to evict anything at all. What changed
is the EVIDENCE. A foreign digest is `dead` only when a caller can say it sees every
live
rank's identity (`peers_known`), and the second liveness witness - the digest a live
worker publishes in its own rank file, which is already this tier's proof that a fitted
worker exists - keeps a rank that is merely QUIET inside the live set indefinitely.

THE TWO CASES, AND THE COUNTER THAT TELLS THEM APART (the operator's "single thing that
would have saved me an investigation"): `stale_geometry_removed` counts removals
licensed
by an orphaned digest (cleanup, expected, carries its own byte total) and
`peer_files_removed_live` counts removals of a rank that is ALIVE (since 0.2.45 a
capacity
pass cannot plan one, so non-zero is a bug report). `peer_files_removed` remains the
superclass of both.

Real files built by `disk_layout`'s own naming and commit protocol; the real walk, the
real planner, the real unlinker, and real rank files in the real diagnostics directory
naming convention. No engine, no torch, no network, no sleeping, and no serving box: the
only thing this file invents is mtimes and file contents, because those two ARE the two
liveness witnesses.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_read,
    disk_restore,
    disk_spill,
    worker_diag,
)

# --- the shapes this test controls -------------------------------------------

SLOT_BYTES = 4096
PP1_BYTES = 12_902_400
OLD = 1_600_000_000.0          # an mtime far in the past, as a cold spill's would be
QUIET_AGE_SECONDS = 300.0      # > PEER_DIGEST_FRESH_SECONDS: idle, NOT dead
GONE_AGE_SECONDS = 3600.0      # > PEER_RANK_FILE_FRESH_SECONDS: the worker IS gone
SHARD = "0a"

OWN_RANK = "PP0_TP0"
BUSY_RANK = "PP1_TP0"          # a sibling writing right now
QUIET_RANK = "PP1_TP1"         # a sibling idle for longer than the peer window
ORPHAN_RANK = "PP1_TP2"        # a rank nobody on this box runs any more (old config)

CEILING = 30_000               # this rank is over its share on purpose; see below


def fingerprint(rank: str = OWN_RANK, **overrides) -> dict:
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


def digest_of(rank: str = OWN_RANK, **overrides) -> str:
    made = disk_layout.fingerprint_digest(fingerprint(rank, **overrides))
    assert made["ok"], made["reason"]
    return made["geometry_digest"]


OWN_GEO = digest_of(OWN_RANK)
BUSY_GEO = digest_of(BUSY_RANK, total_bytes_per_block=PP1_BYTES)
QUIET_GEO = digest_of(QUIET_RANK, total_bytes_per_block=PP1_BYTES)
ORPHAN_GEO = digest_of(ORPHAN_RANK, total_bytes_per_block=PP1_BYTES)
assert len({OWN_GEO, BUSY_GEO, QUIET_GEO, ORPHAN_GEO}) == 4


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


def exists(paths) -> list:
    return [str(path) for path in paths if os.path.exists(path)]


# --- the second witness: a live worker's published identity -------------------

def write_rank_file(directory, rank: str, digest, *, age: float = 0.0,
                    publish: bool = True) -> str:
    """One `rank-<rank>.json`, written the way `worker_diag` writes it.

    `publish=False` is the mixed-version case: a live worker whose file carries no
    readable digest. It is ALIVE (fresh `ts`) and it is UNNAMEABLE, which is exactly the
    state in which this rank must not declare any foreign digest dead.
    """
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "version": worker_diag.SCHEMA_VERSION,
        "epoch": "aabbccdd",
        "ts": round(time.time() - age, 6),
        "role": "worker",
        "rank": rank,
        "arena": {"bytes_total": 0, "bytes_used": 0, "num_blocks": 0},
        "counters": {"stored": 0, "restored_hits": 0},
        "writer": {"plugin": worker_diag.WRITER_TAG,
                   "epoch_scope": "process"},
        "error": None,
    }
    if publish:
        # The field `disk_store.worker_report()` publishes today, unchanged in name.
        document["disk"] = {"section": "disk", "digest": digest,
                            "installed": True, "armed": True}
    path = os.path.join(str(directory), f"rank-{rank}.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document))
    return path


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    for name in (config.ENV_DISK_ROOT, config.ENV_DISK_BYTES,
                 config.ENV_DIAG_DIR, config.ENV_DIAG_RANK):
        monkeypatch.delenv(name, raising=False)
    disk_spill.reset_state_for_tests()
    disk_read.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    disk_spill.shutdown_evict_pool()
    yield
    disk_spill.shutdown_evict_pool()
    disk_spill.reset_state_for_tests()
    disk_read.reset_state_for_tests()
    disk_index.reset_state_for_tests()


@pytest.fixture
def tier(tmp_path, monkeypatch):
    """This rank's store AND a diagnostics directory for the rank files."""
    root = tmp_path / "kv-store"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, OWN_RANK)
    disk_spill.configure(fingerprint(OWN_RANK))
    disk_spill.configured()
    disk_spill._state["ceiling_bytes"] = CEILING
    # A rank that has been writing all boot carries a running estimate; seeding it is
    # what lets the trigger fire, exactly as a real boot's does.
    disk_spill._state["estimated_bytes"] = 8 * SLOT_BYTES
    return {"root": root, "diag": diag}


def mine(root, count=8) -> list:
    """This rank's own files, cold and in LRU order (index 0 is the oldest touch)."""
    return [make_file(root, OWN_GEO, index, when=OLD + float(index))
            for index in range(1, count + 1)]


def peer_ranks_view() -> dict:
    return dict(disk_spill.section()["peer_ranks"])


# ===========================================================================
# 1. A QUIET SIBLING IS NOT FREE CAPACITY
# ===========================================================================

def test_a_quiet_rank_s_bytes_survive_a_busy_rank_s_ceiling_pass(tier):
    """Rank X idle for five minutes is not rank X's bytes being free for the taking.

    Rank X is idle in BOTH the only senses that used to matter - no writes for five
    minutes, so it has aged out of the write-activity map - and it is still alive, which
    its rank file says and its heartbeat keeps saying. The ceiling still gets met, so
    "protected" is not another word for "the ceiling stopped working": this rank
    pays with
    its own bytes, oldest touch first.
    """
    root, diag = tier["root"], tier["diag"]
    now = time.time()
    own = mine(root)
    quiet = [make_file(root, QUIET_GEO, 200 + index,
                       when=now - QUIET_AGE_SECONDS - float(index), group=4 + index)
             for index in range(3)]
    busy = [make_file(root, BUSY_GEO, 100 + index, when=now - float(index),
                      group=4 + index) for index in range(3)]
    # The identity witness: all three ranks' workers are alive and publishing.
    write_rank_file(diag, OWN_RANK, OWN_GEO)
    write_rank_file(diag, QUIET_RANK, QUIET_GEO, age=QUIET_AGE_SECONDS)
    write_rank_file(diag, BUSY_RANK, BUSY_GEO)

    disk_spill._enforce_ceiling()

    # (1) SILENCE IS NOT DEATH. THE DEFECT FAILS HERE: all three are unlinked.
    assert exists(quiet) == quiet, (
        f"a rank that has not written for {QUIET_AGE_SECONDS:.0f} s had "
        f"{len(quiet) - len(exists(quiet))} of its {len(quiet)} files UNLINKED as "
        "'dead' bytes while its worker was alive and publishing that very digest. A "
        "quiet rank is not a gone rank.")

    # (2) THIS RANK PAID, IN LRU ORDER.
    shed = [path for path in own if not os.path.exists(path)]
    assert shed, ("the pass ran and shed nothing: with the quiet rank's bytes "
                  "protected, this rank must still be able to meet its share from its "
                  "own files")
    assert shed == own[:len(shed)], \
        "this rank's own bytes are shed oldest-touch first (the clock is st_mtime)"

    # (3) A BUSY SIBLING IS UNTOUCHED, AS IT ALWAYS WAS.
    assert exists(busy) == busy

    # (4) THE WITNESS, AS AN OPERATOR READS IT: an idle rank is PROTECTED.
    peers = peer_ranks_view()
    assert peers["source"] == "rank_files", peers
    assert peers["identity_known"] is True, peers["identity_reason"]
    assert peers["identity_published_digests"] == 2, peers
    # The quiet rank counts as live even though no sibling wrote under its digest inside
    # `PEER_DIGEST_FRESH_SECONDS` - that is the whole fix.
    assert peers["live_digests"] == 2, peers
    assert int(disk_spill.section()["ceiling_share"]["share_divisor"]) == 3, \
        "three live ranks share the ceiling, so the fair share is a third"


def test_an_idle_rank_being_protected_and_an_idle_rank_being_shed_read_differently(
        tier):
    """The operator's question, answered by two different counters.

    "Is an idle rank being protected, or is an idle rank being shed?" must be readable
    from ONE snapshot, without diffing two files or grepping the journal. Protected: the
    idle rank's digest is published by a live worker and it appears in `live_digests`,
    while `stale_geometry_removed` and `peer_files_removed_live` stay 0. Shed: those two
    counters move, and each names a different reason. This test pins the protected half;
    the next two pin the shed halves.
    """
    root, diag = tier["root"], tier["diag"]
    now = time.time()
    mine(root)
    quiet = [make_file(root, QUIET_GEO, 200 + index,
                       when=now - QUIET_AGE_SECONDS - float(index), group=4 + index)
             for index in range(3)]
    write_rank_file(diag, OWN_RANK, OWN_GEO)
    write_rank_file(diag, QUIET_RANK, QUIET_GEO, age=QUIET_AGE_SECONDS)

    disk_spill._enforce_ceiling()

    peers = peer_ranks_view()
    assert exists(quiet) == quiet
    assert int(peers["live_digests"]) == 1
    assert int(peers["peer_files_removed"]) == 0, peers
    assert int(peers["peer_files_removed_live"]) == 0, peers
    assert int(peers["stale_geometry_removed"]) == 0, peers
    assert int(peers["stale_geometry_removed_bytes"]) == 0, peers


# ===========================================================================
# 2. THE RECLAIM PATH STAYS OPEN, AND SAYS SO
# ===========================================================================

def test_a_digest_no_live_rank_publishes_is_still_shed_ahead_of_my_own_bytes(tier):
    """THE CONSTRAINT. A configuration change's orphans stay shed-first, on identity.

    On 2026-10-07 the operator moved `KV_OFFLOAD_SIZE` 64 GiB -> 8 GiB. Slot width
    moved,
    so the geometry digest moved, so every pre-change file in the store now names a
    digest
    NO live rank owns - unservable by anybody, forever, and correct to reclaim.
    Disabling
    that would fill the store with bytes that can never be served and never be
    reclaimed,
    and the ceiling pass would eventually refuse to evict anything.

    The rank files here name every live rank (this one, publishing its own digest),
    so the
    orphan digest is not a live rank's current one and `attribute_name` may call it
    `dead`.
    """
    root, diag = tier["root"], tier["diag"]
    own = mine(root)
    orphan = [make_file(root, ORPHAN_GEO, 300 + index, when=OLD + 500.0 + float(index))
              for index in range(3)]
    write_rank_file(diag, OWN_RANK, OWN_GEO)

    disk_spill._enforce_ceiling()

    # THE ORPHAN PAYS FIRST, BEFORE THIS RANK'S OWN COLD BYTES.
    assert exists(orphan) == [], (
        "bytes under a digest no live rank publishes or writes must stay reclaimable: "
        "this is what an arena-size change leaves behind, and the store cannot be "
        "allowed to fill with it")
    shed = [path for path in own if not os.path.exists(path)]
    assert shed, ("the orphan files alone did not reach the target, so this rank had "
                  "to shed its own too - which is the point of shed-ORDER, not a "
                  "failure")

    # AND THE JUSTIFICATION IS NAMED, WHICH IS WHAT 5,910 removals could not say before.
    peers = peer_ranks_view()
    assert int(peers["peer_files_removed"]) == len(orphan), peers
    assert int(peers["stale_geometry_removed"]) == len(orphan), peers
    assert int(peers["stale_geometry_removed_bytes"]) \
        == len(orphan) * SLOT_BYTES, peers
    # The two readings of `peer_files_removed` are now SEPARATE numbers, and the
    # alarming one is zero.
    assert int(peers["peer_files_removed_live"]) == 0, peers
    assert int(peers["peer_files_removed_known_peer"]) == 0, peers


def test_a_stale_rank_file_stops_protecting_its_digest(tier):
    """A rank whose WORKER has gone does eventually free its bytes - no timer needed.

    The other half of "how do you tell a gone rank from a quiet one": the quiet rank's
    file is old and its worker's heartbeat would have rewritten it; this one's file has
    passed `PEER_RANK_FILE_FRESH_SECONDS`, which is the SAME window `disk_restore`
    uses to
    refuse serving from a rank file - a bound on a PROCESS's liveness, never a lifetime
    over cached bytes. Once no live rank publishes the digest, the bytes under it are
    `stale_geometry` and the quota reclaims them.
    """
    root, diag = tier["root"], tier["diag"]
    now = time.time()
    own = mine(root)
    gone = [make_file(root, QUIET_GEO, 200 + index,
                      when=now - QUIET_AGE_SECONDS - float(index), group=4 + index)
            for index in range(3)]
    write_rank_file(diag, OWN_RANK, OWN_GEO)
    # The worker behind QUIET_GEO stopped heartbeating an hour ago.
    write_rank_file(diag, QUIET_RANK, QUIET_GEO, age=GONE_AGE_SECONDS)

    disk_spill._enforce_ceiling()

    assert exists(gone) == [], (
        "a rank whose worker has stopped rewriting its rank file is GONE, not quiet, "
        "and the bytes under its digest have to be reclaimable or they are lost "
        "forever")
    assert [path for path in own if not os.path.exists(path)], \
        "and this rank still pays for the rest of its share"
    peers = peer_ranks_view()
    assert int(peers["identity_files_stale"]) == 1, peers
    assert int(peers["stale_geometry_removed"]) == len(gone), peers
    assert int(peers["peer_files_removed_live"]) == 0, peers


def test_an_unnameable_live_rank_freezes_foreign_bytes_instead_of_freeing_them(tier):
    """FAIL CLOSED: an identity I cannot read means nobody's digest is dead.

    The mixed-version case - a sibling running a plugin that publishes no digest, or a
    contributor that raised - produces a FRESH rank file with nothing readable in
    it. "Not
    in my list" is not evidence of absence, so this rank withholds every foreign
    committed
    file, pays with its own, and lets the ceiling come up SHORT. A short ceiling is a
    spill refusal and a filesystem an operator can see; a deleted sibling prefix is
    a hole
    in a serving path.
    """
    root, diag = tier["root"], tier["diag"]
    own = mine(root, count=4)
    foreign = [make_file(root, ORPHAN_GEO, 300 + index, when=OLD + 500.0 + float(index))
               for index in range(8)]
    write_rank_file(diag, OWN_RANK, OWN_GEO)
    write_rank_file(diag, BUSY_RANK, None, publish=False)   # alive, unnameable

    disk_spill._enforce_ceiling()

    peers = peer_ranks_view()
    assert peers["identity_available"] is True
    assert peers["identity_known"] is False, peers["identity_reason"]
    assert int(peers["identity_files_unnamed"]) == 1, peers
    assert exists(foreign) == foreign, \
        "an identity this rank cannot read must never be treated as nobody's"
    assert int(peers["unattributable_entries_excluded"]) >= len(foreign), peers
    assert int(peers["stale_geometry_removed"]) == 0, peers
    # This rank paid everything it was allowed to pay, and the ceiling STILL came up
    # SHORT: the store is over its share because the withheld bytes stay counted as
    # capacity. That is the intended failure mode, and it is loud.
    assert [path for path in own if not os.path.exists(path)] == own, \
        "a rank that cannot attribute foreign bytes sheds its OWN to the last file"
    assert peers["plan_short"] is True, peers
    assert int(peers["plan_short_bytes"]) > 0, peers
    assert int(peers["passes_short"]) >= 1, peers


# ===========================================================================
# 3. OWNERSHIP STILL WORKS BOTH WAYS
# ===========================================================================

def test_the_quiet_rank_still_owns_and_can_shed_its_own_bytes(tier, monkeypatch):
    """Protection is not immunity: the owning rank still LRU-sheds the same files.

    Becoming unsheddable-for-others must not make a byte
    unreclaimable-for-its-owner. Boot
    as the QUIET rank and the very same files are this rank's own `current` bytes, shed
    oldest-touch first under its own ceiling.
    """
    root, diag = tier["root"], tier["diag"]
    now = time.time()
    mine(root)
    quiet = [make_file(root, QUIET_GEO, 200 + index,
                       when=now - QUIET_AGE_SECONDS - float(index), group=4 + index)
             for index in range(3)]
    write_rank_file(diag, QUIET_RANK, QUIET_GEO, age=QUIET_AGE_SECONDS)
    write_rank_file(diag, OWN_RANK, OWN_GEO)

    disk_spill.reset_state_for_tests()
    disk_spill.configure(fingerprint(QUIET_RANK,
                                     total_bytes_per_block=PP1_BYTES))
    disk_spill._state["root"] = str(root)
    disk_spill._state["ceiling_bytes"] = 8_000
    disk_spill._state["estimated_bytes"] = 8 * SLOT_BYTES

    disk_spill._enforce_ceiling()

    gone = [path for path in quiet if not os.path.exists(path)]
    assert gone, ("the rank that OWNS these bytes must still be able to shed them: "
                  "protecting a quiet rank from its siblings must not make its own LRU "
                  "stop working")
    assert gone == quiet[:len(gone)], "its own bytes go oldest-touch first"


# ===========================================================================
# 4. THE PURE RULE, AT THE LAYER THAT OWNS IT
# ===========================================================================

def test_a_foreign_digest_is_dead_only_when_the_liveness_view_is_known():
    """`attribute_name`'s three verdicts, as a pure function.

    THE DEFAULT MATTERS: `peers_known` defaults to True, so every existing caller means
    what it always meant and `stale_geometry` stays reclaimable for them.
    `disk_spill` is
    the caller that can actually answer the question, and it answers it from the rank
    files.
    """
    quiet_name = disk_layout.build_slot_name(slot_key(21), QUIET_GEO)["filename"]

    # A digest a live rank owns - in the map - is that rank's, whoever else is alive.
    assert disk_layout.attribute_name(
        quiet_name, OWN_GEO, (QUIET_GEO,))["attribution"] \
        == disk_layout.ATTRIBUTION_PEER

    # Not in the map, and the caller sees every live rank: an orphan. Reclaimable.
    dead = disk_layout.attribute_name(quiet_name, OWN_GEO, (BUSY_GEO,))
    assert dead["attribution"] == disk_layout.ATTRIBUTION_DEAD
    assert dead["peers_known"] is True
    assert "configuration change" in dead["reason"] or "no live rank" in dead["reason"]

    # Not in the map, and the caller CANNOT name everybody: not free capacity.
    blind = disk_layout.attribute_name(quiet_name, OWN_GEO, (BUSY_GEO,),
                                       peers_known=False)
    assert blind["attribution"] == disk_layout.ATTRIBUTION_UNATTRIBUTABLE
    assert blind["peers_known"] is False
    assert blind["peers_unconfirmed"] is False, \
        "`peers_known` is a separate question from whether the map is empty"


def test_the_planner_withholds_only_when_the_caller_cannot_name_every_live_rank():
    """The planner side of the same line: withheld when blind, shed-first when known.

    Both halves in one test so the boundary cannot drift: the SAME entry list, the SAME
    ceiling, and the only difference is `peers_known`.
    """
    rows = [
        {"path": f"/store/{SHARD}/own-old", "size_bytes": 100,
         "mtime_ns": int(OLD * 1e9),
         "filename": disk_layout.build_slot_name(slot_key(31), OWN_GEO)["filename"]},
        {"path": f"/store/{SHARD}/foreign-old", "size_bytes": 100,
         "mtime_ns": int((OLD + 100_000.0) * 1e9),
         "filename": disk_layout.build_slot_name(slot_key(32), QUIET_GEO)["filename"]},
    ]
    known = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                          ceiling_bytes=150, used_bytes=200,
                                          peer_digests=(BUSY_GEO,),
                                          peers_known=True)
    assert known["peers_known"] is True
    assert known["excluded_unattributable"] == 0
    assert known["stale_first"] == 1, \
        "an orphan digest is still the FIRST thing shed - the config-change path works"
    # The orphan goes first even though the OWN file is the OLDER touch: bucket
    # beats clock.
    assert known["paths"] == [f"/store/{SHARD}/foreign-old"], known["paths"]
    assert known["short"] is False

    unknown = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                            ceiling_bytes=150, used_bytes=200,
                                            peer_digests=(BUSY_GEO,),
                                            peers_known=False)
    assert unknown["peers_known"] is False
    assert unknown["excluded_unattributable"] == 1
    assert unknown["stale_first"] == 0
    assert unknown["paths"] == [f"/store/{SHARD}/own-old"], unknown["paths"]
    # Own bytes happened to cover the target here, so this plan is NOT short - and
    # shortness is the other half of the rule, so it is pinned too: the same rows with a
    # `used_bytes` the own file cannot cover must report the shortfall out loud.
    assert unknown["short"] is False
    starving = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                             ceiling_bytes=150, used_bytes=1000,
                                             peer_digests=(BUSY_GEO,),
                                             peers_known=False)
    assert starving["paths"] == [f"/store/{SHARD}/own-old"], starving["paths"]
    # 900 withheld bytes over a 135-byte target, after shedding the only 100 own bytes:
    # 765 short. That shortfall is the ONLY signal an operator gets that the ceiling is
    # being honoured by refusal rather than by deletion.
    assert starving["short"] is True and starving["short_by_bytes"] == 765, starving
    # The gate's key set is stable across every state, or a reader cannot read it.
    assert (set(known) - {"entries"}) == (set(unknown) - {"entries"})


def test_an_inflight_part_and_the_pre_group_shape_stay_dead_by_identity():
    """The gate is about OWNERSHIP, so what nobody can read is still sheddable.

    Neither verdict reads a clock and neither reads a rank file:
      * an in-flight `.part` is not a committed file, so it holds nobody's restore;
      * the pre-group key shape is this plugin's OWN lineage and no group-aware
      reader can
        load it whoever wrote it (`stale_key_shape`).
    Refusing to shed these would be the failure direction 0.2.34 warned about: a
    store of
    bytes no boot can ever read, unsheddable, with the ceiling as a comment.
    """
    part = disk_layout.build_slot_path("/store", slot_key(41), QUIET_GEO)["path"] \
        + ".part"
    assert disk_layout.attribute_name(
        os.path.basename(part), OWN_GEO, (BUSY_GEO,),
        peers_known=False)["attribution"] == disk_layout.ATTRIBUTION_DEAD, \
        "an uncommitted .part holds nobody's restore, whoever's rank file says what"

    legacy = (f"{disk_layout.SLOT_FILE_PREFIX}{slot_key(42)[:16]}-{QUIET_GEO}"
              f"{disk_layout.SUFFIX_DATA}")
    verdict = disk_layout.attribute_name(legacy, OWN_GEO, (BUSY_GEO,),
                                         peers_known=False)
    assert verdict["classification"] == disk_layout.NAME_STALE_KEY_SHAPE
    assert verdict["attribution"] == disk_layout.ATTRIBUTION_DEAD, \
        "the pre-group key shape is ungroupable by identity, not aged out by silence"
    plan = disk_layout.lru_eviction_plan(
        [{"path": f"/store/{SHARD}/k", "size_bytes": 100,
          "mtime_ns": int(OLD * 1e9), "filename": legacy}],
        geometry_digest=OWN_GEO, ceiling_bytes=50, used_bytes=100,
        peer_digests=(BUSY_GEO,), peers_known=False)
    assert plan["planned"] == 1 and plan["excluded_unattributable"] == 0, \
        "positively-unloadable bytes still buy capacity, or the ceiling stops being one"


def test_an_operators_sweep_still_reclaims_a_gone_ranks_bytes(tier):
    """The escape hatch for a rank that is REALLY gone is an operator's IDENTITY claim.

    A sweep is not a ceiling pass: `execute_unlink_plan` without `for_capacity` applies
    `UNLINKABLE_CLASSIFICATIONS` and does NOT consult the attribution gate, because a
    sweep is an explicit human assertion that these bytes belong to nobody. That is the
    legitimate explicit-identity path, and it must survive this change: even if some
    other
    witness went quiet, an operator's decision is what reclaims a foreign digest.
    """
    root = tier["root"]
    now = time.time()
    gone = make_file(root, QUIET_GEO, 51, when=now - QUIET_AGE_SECONDS)
    own = make_file(root, OWN_GEO, 52, when=OLD)

    shard = os.path.join(str(root), SHARD)
    sweep = disk_layout.boot_sweep(shard, OWN_GEO, peer_digests=(BUSY_GEO,))
    assert sweep["ok"], sweep["reason"]
    assert gone in sweep["planned_paths"], \
        "a sweep must still name a foreign digest's committed file as reclaimable"

    executed = disk_layout.execute_unlink_plan(shard, [gone],
                                               geometry_digest=OWN_GEO,
                                               peer_digests=(BUSY_GEO,),
                                               dry_run=False, for_capacity=False)
    assert executed["ok"], executed["reason"]
    assert executed["counts"]["unlinked"] == 1
    assert not os.path.exists(gone)
    assert os.path.exists(own), "a sweep never touches this rank's current files"


# ===========================================================================
# 5. THE ONE WINDOW, SHARED BY TWO READERS
# ===========================================================================

def test_the_rank_file_liveness_window_is_one_number_not_two():
    """`disk_spill` and `disk_restore` read the SAME file and must agree who is alive.

    `disk_restore` refuses to serve on a rank file older than `RANK_FILE_FRESH_SECONDS`;
    `disk_spill` stops protecting a digest when its publisher's file passes that same
    window. Two readers, one file, one window - and a drift between them would mean the
    tier serving from a worker it simultaneously considers gone. Pinned rather than
    commented, because a comment has never stopped a constant.
    """
    assert disk_spill.PEER_RANK_FILE_FRESH_SECONDS \
        == disk_restore.RANK_FILE_FRESH_SECONDS
    assert disk_restore.RANK_FILE_FRESH_SECONDS == 900.0
    assert disk_spill.MAX_PEER_RANK_FILES == disk_restore.MAX_RANK_FILES_PROBED
    assert disk_spill.MAX_PEER_RANK_FILE_BYTES == disk_restore.MAX_RANK_FILE_BYTES


def test_the_peer_map_still_decays_because_it_only_decides_a_FAIR_SHARE(tier):
    """The write-activity window stays - and it is a bound over a TRAFFIC witness.

    The 120 s window is not gone, and must not be: it is what stops a rank that has
    finished booting from dividing the ceiling by ranks that stopped spilling hours ago,
    and it now decides ONLY the fair share and the reporting. What it can no longer
    do is
    move a byte into the destructive set - which is why the quiet rank's bytes
    survive the
    pass above while its digest does age out of the write-activity map here.
    """
    root, diag = tier["root"], tier["diag"]
    now = time.time()
    mine(root)
    make_file(root, QUIET_GEO, 200, when=now - QUIET_AGE_SECONDS, group=4)
    write_rank_file(diag, OWN_RANK, OWN_GEO)
    write_rank_file(diag, QUIET_RANK, QUIET_GEO, age=QUIET_AGE_SECONDS)

    assert disk_spill._live_peer_digests() == (), \
        "the write-activity witness still ages a quiet digest out (it measures traffic)"
    live, known = disk_spill._peer_liveness()
    assert live == (QUIET_GEO,), \
        "the identity witness still counts it live, which protects its bytes"
    assert known is True
