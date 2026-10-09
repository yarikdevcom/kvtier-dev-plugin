r"""An UNATTRIBUTABLE name never buys you capacity (0.2.34, fail-closed eviction).

WHY THIS FILE EXISTS - THE INCIDENT, IN ONE PARAGRAPH. Four ranks share ONE spill
root (`/home/user/kv-tier/disk`, 256 shard directories); files are told apart only by
the geometry digest in their name. Every rank runs its own ceiling pass, and the shed
order said: bytes this boot cannot read FIRST, its own files second, a *known* peer's
files last. The middle of that rule was a lie, because `classify_name` cannot tell "a
digest nobody can ever read" from "a digest I have not learned belongs to a live
sibling" - with an empty `peer_digests` both answers are `stale_geometry`, bucket 0.
So a rank that could not name its siblings classified their LIVE KV data as the
cheapest thing on the box and deleted it first, and four evictors did that to each
other. The evidence is the persistent journal at `/home/user/kv-tier/journal/`: 15,848
removal lines of size 11,468,800 - the OTHER rank's block size (PP0 is 11,468,800 B,
PP1 is 12,902,400 B) - recorded by ranks that were not the writer. The production cost
was a `file_missing` read in the middle of a restored prefix: everything behind the
hole, 19,200 tokens, recomputed.

THE FIX THIS FILE PINS, in three parts:

1. THE GATE. Within the ceiling path, a committed file under a digest that is not this
   boot's, seen while NO peer digest is confirmed, is UNATTRIBUTABLE and is never a
   candidate for unlinking. Own files and bytes POSITIVELY claimed by no live rank
   (an in-flight `.part`, the pre-group key shape, or a digest outside a CONFIRMED
   peer map) stay sheddable, so a single-rank box and a store of old boots' geometry
   behave exactly as they did. This makes the unknown case consistent with the
   malformed-name doctrine that has always been in `disk_layout`: reported, never
   deleted.
2. THE CONSEQUENCE IS HONEST. A plan that cannot meet its target without shedding
   unattributable files comes up SHORT and says so - `short` / `short_by_bytes` /
   `excluded_unattributable` on the plan, `plan_short` / `plan_short_bytes` /
   `passes_short` / a one-line `plan_short_reason` on the spill section. Exceeding the
   ceiling is the correct failure mode: spilling back-pressures, `no_free_space` still
   guards the filesystem, and an operator can act. Deleting a sibling's live restore
   corrupts a serving path in silence.
3. THE WITNESS IS PUBLISHED. `peer_digests_live` / `peer_digests_unconfirmed` /
   `peer_digests_dropped` / `peer_files_planned` used to exist only inside
   `disk_spill._state` - so when the journal proved cross-rank deletion was happening,
   the one question that separated "shed a KNOWN peer because nothing else could meet
   the target" from "could not name the peers at all" could not be answered from any
   payload. They now travel in the per-rank `rank-*.json` under `disk.peer_eviction`,
   with the same key set on the healthy, never-ran-a-pass, counters-raised and
   byte-cap-trimmed paths, and the dev door sums them across ranks.
   `peer_files_removed` / `peer_files_removed_known_peer` are the attributable removal:
   the cross-rank case, visible in one read.

Posture: no engine, no torch, no network, no sleeping. Real committed files under
`tmp_path`, built with `disk_layout`'s own naming and commit protocol, and the real
`_enforce_ceiling()` - the same fixtures `test_touch_promotion_and_peer_eviction.py`
and `test_disk_ceiling_walk.py` use.
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
    disk_spill,
    disk_store,
    worker_diag,
)

# --- the shapes this test controls -------------------------------------------

SLOT_BYTES = 4096
PP1_BYTES = 12_902_400            # the live box's other PP stage
SHARD = "0a"
OLD = 1_600_000_000.0             # an mtime far in the past


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
# A SIBLING RANK's digest: same model, different rank AND slot width. A file named
# under it is the incident's victim: live, loadable, and 11,468,800 bytes wide on the
# box. `OTHER_GEO` is a third digest - an OLD BOOT's geometry, which a CONFIRMED peer
# map makes positively dead (and so sheddable) while an empty one makes unattributable.
PEER_GEO = digest_of(rank="PP1_TP0", total_bytes_per_block=PP1_BYTES)
OTHER_GEO = digest_of(none_hash_seed="0x0000000000000001")
assert PEER_GEO != OWN_GEO and OTHER_GEO != OWN_GEO and PEER_GEO != OTHER_GEO


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


def journal_lines(directory) -> list:
    where = disk_index.journal_path()
    assert where["ok"] is True, where["reason"]
    if not os.path.exists(where["path"]):
        return []
    with open(where["path"], "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def key16_of(path) -> str:
    parsed = disk_layout.parse_path(str(path))
    assert parsed["ok"], parsed["reason"]
    return parsed["key16"]


def journalled_removals(directory) -> list:
    """The `key16`s the removal journal names, newest last (`seq op key16 size grp`)."""
    out = []
    for line in journal_lines(directory):
        parts = line.split(" ")
        if len(parts) > 2 and parts[1] == "-":
            out.append(parts[2])
    return out


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_tier(tmp_path, monkeypatch):
    """No inherited environment, no leftover reader thread."""
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    disk_read.reset_state_for_tests()
    disk_spill.reset_state_for_tests()
    disk_index.reset_state_for_tests()
    yield
    disk_spill.shutdown_evict_pool()
    disk_spill.reset_state_for_tests()


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
    """A configured spill tier rooted in `tmp_path`; the test sets the ceiling."""
    root = tmp_path / "kv-store"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    disk_spill.configure(fingerprint())
    disk_spill.configured()
    return {"root": root}


def set_ceiling(bytes_value: int) -> None:
    """Point `_state` at a ceiling, the way `configured()` does from the environment."""
    disk_spill._state["ceiling_bytes"] = int(bytes_value)


def plan_rows(*specs) -> list:
    """`specs` is (name, size_bytes, mtime_ns, digest) with the digest optional."""
    out = []
    for index, (name, size, when, *rest) in enumerate(specs):
        geo = rest[0] if rest else OWN_GEO
        made = disk_layout.build_slot_name(slot_key(200 + index), geo)
        assert made["ok"], made["reason"]
        out.append({"path": f"/store/{SHARD}/{name}", "filename": made["filename"],
                    "size_bytes": size, "mtime_ns": int(when * 1e9)})
    return out


# ===========================================================================
# 1. THE INCIDENT, AS A BEHAVIOUR ON A FILESYSTEM
# ===========================================================================

def test_the_incident_a_peer_blind_ceiling_pass_never_unlinks_a_peers_restore(
        tier, channel):
    """THE REGRESSION, NAMED FOR THE INCIDENT: `peer` + `unattributable`.

    Three files: one of OURS, written long ago, and two under a sibling's digest,
    also old - so the walk honestly cannot claim a live rank is writing that digest
    (the freshness window is 120 s) and `peer_digests` comes back EMPTY. That is the
    state the four ranks were in on the box, and the state every previous release
    answered by deleting the sibling's live files first. Now: the sibling's files
    stay, this rank's own file pays what it can, and the plan reports the shortfall it
    could not close.
    """
    root = tier["root"]
    now = time.time()
    ours = make_file(root, OWN_GEO, 61, when=now - 10_000.0)
    sibling_a = make_file(root, PEER_GEO, 62, when=now - 3_600.0, group=4)
    sibling_b = make_file(root, PEER_GEO, 63, when=now - 3_601.0, group=5)

    # 12,288 accountable bytes against an 8,000-byte ceiling: a 7,200-byte target this
    # rank can only reach by deleting 5,088 bytes, and it owns 4,096 of them.
    set_ceiling(8_000)
    disk_spill._state["estimated_bytes"] = 12_000
    disk_spill._enforce_ceiling()
    # `section()` calls `is_armed()`, which itself writes `last_reason`: read the
    # pass's own sentence before any payload is built.
    reason_line = str(disk_spill._state["last_reason"])

    assert os.path.exists(sibling_a) and os.path.exists(sibling_b), \
        "a peer-looking file is NEVER the thing that buys capacity"
    assert not os.path.exists(ours), "this rank's own bytes are the capacity it owns"
    assert journalled_removals(channel) == [key16_of(ours)]

    peers = disk_spill.section()["peer_ranks"]
    assert peers["live_digests"] == 0, peers
    assert peers["decisions_without_peers"] >= 1, peers
    assert peers["unattributable_entries_last_pass"] == 2, peers
    assert peers["unattributable_entries_excluded"] == 2, peers
    assert peers["unattributable_bytes_last_pass"] == 2 * SLOT_BYTES, peers
    assert peers["peer_files_removed"] == 0, peers
    assert peers["peer_files_removed_known_peer"] == 0, peers

    # THE SHORTFALL, said out loud rather than bought with a sibling's data.
    assert peers["plan_short"] is True, peers
    assert peers["passes_short"] == 1, peers
    assert peers["plan_short_bytes"] == 8_192 - 7_200, peers
    assert "UNATTRIBUTABLE" in (peers["plan_short_reason"] or ""), peers
    assert "2 file(s) withheld" in peers["plan_short_reason"], \
        "the reason must quote the count the payload publishes, not the planner's"
    assert "UNATTRIBUTABLE" in reason_line, \
        "the one-line reason the caller logs must name the gate, not a stack trace"
    # The tier is honestly OVER its ceiling after the pass - which is the point.
    assert disk_spill._state["estimated_bytes"] == 12_288 - SLOT_BYTES
    assert disk_spill._state["ceiling_passes_nothing_sheddable"] == 0, \
        "this is not the 'nothing sheddable' case: it shed what it owned"


def test_a_known_peers_file_is_never_in_this_ranks_destructive_set_at_all(tier,
                                                                         channel):
    """The fair-share half, pinned next to the gate so nobody confuses them.

    Same shape as the incident, but the sibling is WRITING NOW (two files inside the
    120 s window), so its digest is CONFIRMED. Two consequences, and this test owns
    the second: the digest counts as a live rank for the share (`n_effective` 2), and
    its files are excluded from this rank's destructive set by `_attribute_walk_bytes`
    before the planner is even called. So `peer_files_planned` is structurally 0 in
    production, and `peer_files_removed` with it - the witness that says "no rank on
    this boot deleted a live sibling's file", readable in one field.
    """
    root = tier["root"]
    now = time.time()
    ours = make_file(root, OWN_GEO, 71, when=now - 10_000.0)
    sibling_a = make_file(root, PEER_GEO, 72, when=now - 1.0, group=4)
    sibling_b = make_file(root, PEER_GEO, 73, when=now - 2.0, group=5)

    set_ceiling(8_000)
    disk_spill._state["estimated_bytes"] = 12_000
    disk_spill._enforce_ceiling()

    peers = disk_spill.section()["peer_ranks"]
    shares = disk_spill.section()["ceiling_share"]
    assert peers["live_digests"] == 1, peers
    assert shares["n_effective"] == 2, shares
    assert peers["unattributable_entries_last_pass"] == 0, \
        "a CONFIRMED peer is attributed, so the gate has nothing to withhold"
    assert shares["peer_entries_excluded"] == 2, shares
    assert peers["files_planned_last_pass"] == 0, peers
    assert peers["peer_files_removed"] == 0, peers
    assert peers["plan_short"] is False, peers

    assert os.path.exists(sibling_a) and os.path.exists(sibling_b)
    assert not os.path.exists(ours)
    assert journalled_removals(channel) == [key16_of(ours)]


def test_a_digest_outside_a_confirmed_peer_map_is_dead_capacity_again(tier,
                                                                     channel):
    """THE OTHER DIRECTION, and the reason this gate is not "never delete foreign".

    One live peer (two fresh files) makes the peer map CONFIRMED. A third digest that
    is not in it is then positively nobody's - an old boot's geometry - and keeps
    shedding FIRST, exactly as it did before. Without this, a store of old boots'
    files would become unsheddable and the ceiling would become a comment: the gate
    is about BLINDNESS, not about foreign digests.
    """
    root = tier["root"]
    now = time.time()
    ours = make_file(root, OWN_GEO, 81, when=now - 10_000.0)
    live_a = make_file(root, PEER_GEO, 82, when=now - 1.0, group=4)
    live_b = make_file(root, PEER_GEO, 83, when=now - 2.0, group=5)
    dead = make_file(root, OTHER_GEO, 84, when=now - 3_600.0, group=6)

    # accountable = own 4,096 + dead 4,096 = 8,192 against half of a 10,000-byte
    # ceiling (one live peer => share 5,000, target 4,500): the dead digest alone
    # meets it, so this rank's own file must survive.
    set_ceiling(10_000)
    disk_spill._state["estimated_bytes"] = 16_000
    disk_spill._enforce_ceiling()
    reason_line = str(disk_spill._state["last_reason"])

    peers = disk_spill.section()["peer_ranks"]
    assert peers["live_digests"] == 1, peers
    assert peers["unattributable_entries_last_pass"] == 0, peers
    # The dead digest's file is bucket 0, so it goes before this rank's own - and it
    # is the ONLY removal that carries a digest that is not this rank's, which is
    # what `peer_files_removed` exists to make visible.
    assert not os.path.exists(dead), "a digest no live rank claims is capacity"
    assert os.path.exists(ours)
    assert peers["peer_files_removed"] == 1, peers
    assert peers["peer_files_removed_bytes"] == SLOT_BYTES, peers
    assert peers["peer_files_removed_known_peer"] == 0, \
        "it was a dead geometry, not a live sibling - and the counters say which"
    assert os.path.exists(live_a) and os.path.exists(live_b)
    # Fix 3's one sentence, read BEFORE `section()` (which overwrites `last_reason`).
    assert "peer_files_removed=1" in reason_line, reason_line


# ===========================================================================
# 2. THE PLANNER, AS A PURE FUNCTION
# ===========================================================================

def test_unknown_names_are_never_shed_while_peers_are_unconfirmed():
    """The gate's own bookkeeping: withheld, counted, and not mixed up with junk.

    A name that does not parse at all is `skipped` (it is not this tier's capacity),
    while a name that parses under a foreign digest is `excluded_unattributable` (it
    IS capacity, and this rank declined to buy it). Collapsing the two would make a
    store full of other programs' files look like a store full of withheld capacity.
    """
    rows = plan_rows(("ours", 100, OLD),
                     ("foreign", 100, OLD + 100.0, PEER_GEO))
    rows.append({"path": f"/store/{SHARD}/notes.txt", "filename": "notes.txt",
                 "size_bytes": 10, "mtime_ns": int(OLD * 1e9)})
    rows.append("not even a mapping")

    blind = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                          ceiling_bytes=100, used_bytes=200)
    assert blind["excluded_unattributable"] == 1
    assert blind["excluded_unattributable_bytes"] == 100
    assert blind["skipped"] == 2, "the non-slot name and the non-mapping row"
    assert blind["paths"] == [f"/store/{SHARD}/ours"]
    assert blind["short"] is True and blind["short_by_bytes"] == 10

    informed = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                             ceiling_bytes=100, used_bytes=200,
                                             peer_digests=(PEER_GEO,))
    assert informed["excluded_unattributable"] == 0
    assert informed["skipped"] == 2
    assert informed["peer_rank"] == 1, "a KNOWN peer is sheddable, last"
    assert informed["short"] is False


def test_a_known_peers_bucket_is_reached_only_when_own_bytes_cannot_meet_the_target():
    """The shed order the README promises: own files first, a known peer's only when
    the target cannot be met without one - and then it IS met, not "short".
    """
    rows = plan_rows(("ours", 100, OLD), ("peer", 100, OLD + 100.0, PEER_GEO))
    enough_from_own = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                                    ceiling_bytes=150,
                                                    used_bytes=200,
                                                    peer_digests=(PEER_GEO,))
    assert enough_from_own["paths"] == [f"/store/{SHARD}/ours"]
    assert enough_from_own["peer_rank"] == 0

    forced = disk_layout.lru_eviction_plan(rows, geometry_digest=OWN_GEO,
                                           ceiling_bytes=100, used_bytes=200,
                                           peer_digests=(PEER_GEO,))
    assert forced["planned"] == 2
    assert [item["classification"] for item in forced["entries"]] == \
        [disk_layout.NAME_CURRENT, disk_layout.NAME_PEER_RANK], \
        "own bytes first, the sibling's only after they were not enough"
    assert forced["peer_rank"] == 1
    assert forced["short"] is False


def test_the_gate_lets_an_in_flight_part_of_a_foreign_digest_still_be_capacity():
    """The one documented exception, pinned so it cannot drift silently.

    A `.part` is not a committed file, so it holds no restore for anybody: deleting a
    stranger's in-flight write costs that stranger one wasted spill, never a readable
    span. The capacity walk does not even count `.part` files, so in production this
    is a `boot_sweep`-style caller's concern - but "never delete what you cannot
    attribute" must have an EXPLICIT exception list, not an implicit one.
    """
    made = disk_layout.build_slot_path("/store", slot_key(99), PEER_GEO)
    assert made["ok"], made["reason"]
    part = made["path"] + disk_layout.SUFFIX_PART
    attributed = disk_layout.attribute_name(os.path.basename(part), OWN_GEO, ())
    assert attributed["attribution"] == disk_layout.ATTRIBUTION_DEAD
    assert attributed["peers_unconfirmed"] is True

    committed = disk_layout.attribute_name(os.path.basename(made["path"]),
                                           OWN_GEO, ())
    assert committed["attribution"] == disk_layout.ATTRIBUTION_UNATTRIBUTABLE
    assert committed["classification"] == disk_layout.NAME_STALE_GEOMETRY, \
        "`classify_name` still says stale_geometry: the gate is the second question"
    named = disk_layout.attribute_name(os.path.basename(made["path"]), OWN_GEO,
                                       (PEER_GEO,))
    assert named["attribution"] == disk_layout.ATTRIBUTION_PEER


# ===========================================================================
# 3. THE EXECUTOR: THE GATE AT THE MOMENT OF DELETION
# ===========================================================================

def test_the_executor_refuses_an_unattributable_path_even_when_handed_it(tier):
    """Defence in depth: nothing planned upstream is trusted at `os.unlink`.

    A caller that hands `execute_unlink_plan` a foreign-digest path while peer
    identity is unconfirmed is refused, and the file survives - even though
    `for_capacity=True` says "this is a ceiling eviction and it may delete live
    data". With the peer CONFIRMED the same call succeeds: bucket 2 is legitimate,
    bucket unknown is not.
    """
    root = tier["root"]
    foreign = make_file(root, PEER_GEO, 91)
    shard = os.path.dirname(foreign)

    blind = disk_layout.execute_unlink_plan(shard, [foreign], dry_run=False,
                                            geometry_digest=OWN_GEO,
                                            peer_digests=(), for_capacity=True)
    assert blind["counts"]["unlinked"] == 0
    assert blind["counts"]["refused"] == 1
    assert "never buys capacity" in blind["actions"][0]["reason"]
    assert os.path.exists(foreign)

    known = disk_layout.execute_unlink_plan(shard, [foreign], dry_run=False,
                                            geometry_digest=OWN_GEO,
                                            peer_digests=(PEER_GEO,),
                                            for_capacity=True)
    assert known["counts"]["unlinked"] == 1, known["actions"]
    assert not os.path.exists(foreign)


def test_the_sweep_path_keeps_its_own_doctrine_unchanged(tier):
    """THE SCOPE BOUNDARY, pinned rather than accidental.

    `for_capacity=False` is the boot sweep's "the operator asked me to reclaim the
    bytes nobody can read" path, whose allowed set is `UNLINKABLE_CLASSIFICATIONS`.
    This release does NOT change it: it is not wired into production (the ceiling pass
    is the only production deleter), and silently rewriting a sweep operator's intent
    while fixing an eviction bug is exactly the kind of scope creep this repo refuses.
    If the sweep is ever wired up, this test is the place the decision gets made.
    """
    root = tier["root"]
    foreign = make_file(root, PEER_GEO, 92)
    swept = disk_layout.execute_unlink_plan(os.path.dirname(foreign), [foreign],
                                            dry_run=False,
                                            geometry_digest=OWN_GEO,
                                            peer_digests=())
    assert swept["counts"]["unlinked"] == 1, swept["actions"]


# ===========================================================================
# 4. THE WITNESS: THE SAME KEYS ON EVERY PATH
# ===========================================================================

def test_the_peer_counters_reach_the_rank_file_with_one_key_set_on_every_path(
        tier, channel, monkeypatch):
    """The blind spot that cost an hour, closed.

    Four paths, ONE key set: a rank that has run a real ceiling pass, a rank that has
    never run one, a report whose internals raised, and a report trimmed by the
    contribution seam's byte cap. A reader that walks `disk.peer_eviction` in four
    rank files must never have to ask whether a field is allowed to be missing - that
    is the same rule the degraded-view tests hold everywhere else in this repo.
    """
    shapes = []

    # (a) NEVER RAN A CEILING PASS.
    shapes.append(disk_store.worker_report())

    # (b) A REAL PASS, with the gate withholding something.
    root = tier["root"]
    now = time.time()
    make_file(root, OWN_GEO, 101, when=now - 10_000.0)
    make_file(root, PEER_GEO, 102, when=now - 3_600.0, group=4)
    make_file(root, PEER_GEO, 103, when=now - 3_601.0, group=5)
    set_ceiling(8_000)
    disk_spill._state["estimated_bytes"] = 12_000
    disk_spill._enforce_ceiling()
    healthy = disk_store.worker_report()
    shapes.append(healthy)
    assert healthy["peer_eviction"]["unattributable_entries_last_pass"] == 2
    assert healthy["peer_eviction"]["plan_short"] is True
    assert healthy["peer_eviction"]["peer_digests_unconfirmed"] >= 1

    # (c) THE COUNTERS RAISED.
    def explode(*args, **kwargs):
        raise RuntimeError("the counters exploded")

    monkeypatch.setattr(disk_spill, "counters", explode)
    broken = disk_store.worker_report()
    assert broken["report_error"] == "RuntimeError"
    shapes.append(broken)
    monkeypatch.undo()

    # (d) THE BYTE-CAP TRIAGE, forced by shrinking the seam's cap so the report
    # reaches its minimal shape. That step drops everything the writer does not
    # require - and keeps `peer_eviction`, which is why this leg exists.
    monkeypatch.setattr(worker_diag, "MAX_CONTRIBUTION_BYTES", 200)
    trimmed = disk_store.worker_report()
    monkeypatch.undo()
    shapes.append(trimmed)
    assert trimmed.get("truncated") is True, trimmed
    assert len(json.dumps(trimmed, default=str)) <= 4096

    keys = set(shapes[0]["peer_eviction"])
    for name, shape in zip(("never-ran", "healthy", "raised", "trimmed"), shapes,
                           strict=True):
        assert set(shape["peer_eviction"]) == keys, (name, shape["peer_eviction"])
        assert shape["peer_eviction"]["peer_digests_live"] is not None
    # The four counters that existed and were unreadable, plus the gate's outcome and
    # the attributable removal, are all in the block.
    for field in ("peer_digests_live", "peer_digests_unconfirmed",
                  "peer_digests_dropped", "peer_files_planned",
                  "unattributable_entries_excluded",
                  "unattributable_entries_last_pass",
                  "unattributable_bytes_last_pass", "plan_short",
                  "plan_short_bytes", "passes_short", "peer_files_removed",
                  "peer_files_removed_known_peer"):
        assert field in keys, field


def test_the_spill_sections_peer_block_ships_the_same_keys_before_and_after(tier):
    """Same rule one level up: `peer_ranks` before any pass and after one."""
    before = set(disk_spill.section()["peer_ranks"])
    root = tier["root"]
    now = time.time()
    make_file(root, OWN_GEO, 111, when=now - 10_000.0)
    set_ceiling(8_000)
    disk_spill._state["estimated_bytes"] = 12_000
    disk_spill._enforce_ceiling()
    after = disk_spill.section()["peer_ranks"]
    assert set(before) == set(after), set(before) ^ set(after)
    for name, unit in after["units"].items():
        assert unit != "unknown", name
    assert after["live_digests"] == 0
    assert after["peer_files_removed"] == 0


def test_the_dev_door_sums_the_cross_rank_removals_across_ranks(tmp_path,
                                                                monkeypatch):
    """`peer_files_removed` is the number the operator could not get.

    The door reads the rank files the workers write, so a fleet-wide "did any rank
    delete a file whose digest was not its own, and how often did a rank come up
    short" is one HTTP call and not a journal parse.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    for rank, removed, short in (("PP0_TP0", 3, 7), ("PP1_TP0", 0, 2)):
        document = {"version": worker_diag.SCHEMA_VERSION,
                    "epoch": "abcd0000", "ts": round(time.time(), 6),
                    "role": "worker", "rank": rank,
                    "arena": {"bytes_total": 1, "bytes_used": 0, "num_blocks": 1},
                    "counters": {"stored": 0, "restored_hits": 0,
                                 "store_events_pending": 0,
                                 "load_events_pending": 0},
                    "disk": {"evictions": 10,
                             "peer_eviction": {"peer_files_removed": removed,
                                               "peer_files_removed_known_peer":
                                                   removed,
                                               "peer_digests_unconfirmed": 5,
                                               "unattributable_entries_excluded": 9,
                                               "passes_short": short}}}
        (directory / f"rank-{rank}.json").write_text(json.dumps(document),
                                                     encoding="utf-8")
    rollup = disk_store.rank_rollup(force=True)
    assert rollup["totals"]["peer_files_removed"] == 3
    assert rollup["totals"]["peer_files_removed_known_peer"] == 3
    assert rollup["totals"]["ceiling_passes_short"] == 9
    assert rollup["totals"]["unattributable_entries_excluded"] == 18
    assert "peer_files_removed_known_peer" in rollup["totals_reported_by"]
    assert "peer_files_removed" in rollup["meanings"]


# ===========================================================================
# 5. FIX 3 END TO END: A REMOVAL UNDER ANOTHER DIGEST IS COUNTED, NOT BURIED
# ===========================================================================

def test_a_removal_under_a_known_peers_digest_is_counted_and_said(tier, channel,
                                                                 monkeypatch):
    """The counter that replaces "parse 15,848 journal lines and compare block sizes".

    Production excludes a CONFIRMED peer's entries from the destructive set before the
    planner runs (see the fair-share test above), so the only way a live sibling's
    file is unlinked is a caller that hands it to the planner - which
    `lru_eviction_plan` accepts by contract (`peer_digests` is a documented argument,
    and `NAME_PEER_RANK` is in `EVICTABLE_CLASSIFICATIONS`). This test simulates that
    caller by restoring the peer entries to the destructive set, and pins what the
    tier then owes the reader: both counters moved, `last_reason` says so in one line,
    the removal is journalled, and `peer_files_removed_known_peer` - the field that
    cannot be confused with an old boot of this same rank's geometry - is the one that
    names it.
    """
    root = tier["root"]
    now = time.time()
    # THE AGES, and they are load-bearing since 0.2.62: OUTSIDE the unlink grace (a
    # file touched inside `disk_layout.UNLINK_GRACE_SECONDS` is not sheddable at all)
    # and INSIDE `PEER_DIGEST_FRESH_SECONDS` (the walk must still believe this peer is
    # LIVE, which is the whole point of the counters under test). A live sibling whose
    # newest file was written 100 s ago is exactly the shape both rules describe.
    shed_age = disk_layout.UNLINK_GRACE_SECONDS + 10.0
    assert shed_age + 2.0 < disk_spill.PEER_DIGEST_FRESH_SECONDS, (
        "the grace and the peer-liveness window no longer overlap: this test needs "
        "an age that is sheddable and still live")
    make_file(root, OWN_GEO, 121, when=now - 10_000.0)
    sibling_a = make_file(root, PEER_GEO, 122, when=now - shed_age - 1.0, group=4)
    sibling_b = make_file(root, PEER_GEO, 123, when=now - shed_age - 2.0, group=5)
    real = disk_spill._attribute_walk_bytes

    def handing_peers_back(walk, peers, peers_known=True):
        """What a caller that COUNTS and OFFERS a known peer's bytes looks like.

        Both halves move together, because a peer file this rank neither counts nor
        offers is a peer file the planner can never see (that is the 0.2.30 fair
        share), and the point of this test is what the tier owes the reader when a
        sibling's file DOES get deleted.
        """
        attribution = real(walk, peers, peers_known)
        own_digest = disk_spill._state["geometry_digest"]
        peer_rows = [row for row in (walk.get("entries") or [])
                     if isinstance(row, dict)
                     and disk_layout.classify_name(
                         row.get("filename") or "", own_digest,
                         peers)["classification"] == disk_layout.NAME_PEER_RANK]
        return dict(attribution,
                    accountable_bytes=int(attribution["accountable_bytes"])
                    + sum(int(row.get("size_bytes") or 0) for row in peer_rows),
                    sheddable_entries=list(attribution["sheddable_entries"])
                    + peer_rows)

    monkeypatch.setattr(disk_spill, "_attribute_walk_bytes", handing_peers_back)
    # accountable 12,288 against a 5,000-byte share (one live peer) => a 4,500 target.
    # Own bytes (4,096) are not enough, so the plan reaches into the peer bucket: the
    # oldest peer file goes too, and that is the event Fix 3 must make visible.
    set_ceiling(10_000)
    disk_spill._state["estimated_bytes"] = 12_000
    disk_spill._enforce_ceiling()
    reason_line = str(disk_spill._state["last_reason"])

    peers = disk_spill.section()["peer_ranks"]
    assert peers["live_digests"] == 1, peers
    assert peers["files_planned_last_pass"] == 1, peers
    assert peers["peer_files_removed"] == 1, peers
    assert peers["peer_files_removed_bytes"] == SLOT_BYTES, peers
    assert peers["peer_files_removed_last_pass"] == 1, peers
    assert peers["peer_files_removed_known_peer"] == 1, \
        "a removal under a digest this walk believed LIVE must be distinguishable"
    assert "peer_files_removed=1" in reason_line, reason_line
    assert os.path.exists(sibling_a), "the NEWER peer file was not needed"
    assert not os.path.exists(sibling_b), "the oldest peer file is the one shed"
    removals = journalled_removals(channel)
    assert len(removals) == 2, removals
    assert key16_of(sibling_b) in removals
    assert disk_spill.section()["ceiling_share"]["peer_live_bytes"] >= 0
