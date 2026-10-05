r"""The capacity walk's SAMPLE, and the ceiling decision made from it.

Why this file exists: the live box ran a 192 GiB ceiling (`VLLM_E2_TIER_DISK_BYTES`
= 206,158,430,208) over a 334.9 GiB / 29,532-file store while every payload reported
an estimate of ~170 GiB. Neither number was a lie. The store is bigger than
`MAX_WALK_FILES`, so the capacity walk measured a SAMPLE - and, worse, a depth-first
sample of the FIRST shard directories, because the old walk's file counter was never
reset per shard and its `truncated` broke the outer loop too. Two failures followed
from one walk: the byte total came from ~2/3 of the store, and a file the walk never
opened could not appear in an eviction plan, so the tail of the key space became a
sink that spilled into forever and could never be shed. (The oldest surviving file
was 2.8 hours old while the store grew 145 GiB an hour - eviction was running and
structurally blind at the same time.)

What this file is obliged to prove, in order of what each failure costs:

1. A TRUNCATED WALK ESTIMATES THE WHOLE STORE. `used_bytes` stays the bytes actually
   stat-ed; `estimated_total_bytes` is the sample scaled by `total_files / files`,
   and the ceiling is decided on the second. The fixture puts the bytes in the LAST
   shard directories in `os.scandir` order, which is the exact shape the old walk
   was blind to.
2. THE SAMPLE COVERS THE KEY SPACE, so the oldest-first plan can act on files in any
   shard. The budget is spread as a uniform RATE across shards, never depth-first.
3. A STORE INSIDE THE CAPS IS MEASURED, NOT ESTIMATED - and byte-identically to the
   old code. This tier is live on a box serving traffic; an eviction order or a
   threshold that moved for a store under 20,000 files is not acceptable, so the
   full-measurement path is pinned: same bytes, same entries, same order, same plan.
4. THE WITNESS IS A WITNESS. `section()` publishes the sampled-vs-total counts, the
   measured bytes, the basis, and the truncation flags, so a reader can tell "the
   store is 170 GiB" from "I sampled 20,000 files and extrapolated 170 GiB".
5. EVERY PASS IS BOUNDED. Both the count (directory entries, no `stat`) and the
   measure (`stat` calls) stop at a cap, and a count that stops early extrapolates
   and SAYS SO rather than reporting a floor.

No engine, no torch, no network: a `tmp_path` store of real committed files, built
with `disk_layout`'s own naming and commit protocol, and caps tightened by
monkeypatch - the idiom `test_disk_spill.py` already uses for `MAX_QUEUE_DEPTH` and
`MAX_FILES_PER_WINDOW`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kvtier_dev_plugin import (
    config,
    disk_index,
    disk_layout,
    disk_spill,
)

# --- the fixture store -------------------------------------------------------

# A tiny `MAX_WALK_FILES`, so a 60-file store is "too big to measure" the way a
# 29,532-file store is on the box. The shape of the failure does not depend on the
# size; the size only decides how long the walk takes.
CAP_FILES = 24
SHARDS = 6
PER_SHARD = 10                    # 60 files: 2.5x the cap
SMALL_BYTES = 1024
BIG_BYTES = 512 * 1024
# The first half of the shard list (in `os.scandir` order) holds the small files and
# the SECOND half holds the big ones. That is the shape that defeated the old walk:
# it stat-ed 24 files, all small, and reported 24 KiB for a 15 MiB store.
FIRST_HALF = SHARDS // 2
GROUP = 3

# The ceiling for the enforcement tests: ABOVE the old walk's floor (24 KiB, so the
# old code evicted nothing) and BELOW the scaled measurement (15 MiB, so the fixed
# code evicts). 4 MiB is comfortably inside that window.
CEILING_BYTES = 4 * 1024 * 1024

SLOT_BYTES = 4096


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


def digest() -> str:
    return disk_layout.fingerprint_digest(fingerprint())["geometry_digest"]


def slot_key(shard: str, index: int) -> str:
    """A group-qualified block hash whose `key16` starts with `shard`.

    `key16` is the first 16 hex chars of the digest half and the shard is its first
    two, so leading with the shard is what puts this file in that directory - and
    `index` in the remaining 14 chars keeps the names inside one shard distinct.
    """
    return f"{shard}{index:014x}" + "0" * 48 + f"{GROUP:08x}"


def slot_path(root, geo, shard: str, index: int) -> dict:
    """The path `disk_layout` itself would name this slot's file. No hand-built names.
    """
    made = disk_layout.build_slot_path(root, slot_key(shard, index), geo)
    assert made["ok"], made["reason"]
    return made


def _write(path: Path, size: int, when: float) -> Path:
    """A committed file of exactly `size` bytes with an mtime this test chose."""
    written = disk_layout.write_committed_file(str(path), b"x" * size, sync=False)
    assert written["ok"], written["reason"]
    os.utime(path, (when, when))
    return path


def _shard_order(root: Path, count: int) -> list:
    """Create `count` shard directories, in THE ORDER THE WALK WILL READ THEM.

    Every assertion about "early" and "late" shards is a statement about
    `os.scandir` order, so the order is measured rather than assumed - hard-coding
    `00, 01, 02...` would be a test of `scandir`'s sort order instead of of the walk.
    """
    for index in range(count):
        (root / f"{index:02x}").mkdir(parents=True, exist_ok=True)
    return [entry.name for entry in os.scandir(root)]


def build_skewed_store(root: Path, geo: str, *, small_bytes=SMALL_BYTES,
                       big_bytes=BIG_BYTES, per_shard=PER_SHARD) -> dict:
    """Half the shards small, half the shards big, big files OLDER than small ones.

    Returns the ground truth: the file list per shard, the byte total, and which
    shards the walk has to reach to see any of the big bytes.
    """
    order = _shard_order(root, SHARDS)
    old = 1_600_000_000.0
    recent = old + 100_000.0
    per_shard_paths: dict = {}
    small: list[Path] = []
    big: list[Path] = []
    for position, shard in enumerate(order):
        size = small_bytes if position < FIRST_HALF else big_bytes
        stamp = recent if position < FIRST_HALF else old
        made = []
        for index in range(per_shard):
            path = _write(Path(slot_path(root, geo, shard, index)["path"]), size,
                          stamp)
            made.append(path)
            (big if position >= FIRST_HALF else small).append(path)
        per_shard_paths[shard] = made
    return {"order": order, "paths": per_shard_paths, "small": small, "big": big,
            "total_bytes": len(small) * small_bytes + len(big) * big_bytes,
            "small_bytes": small_bytes, "big_bytes": big_bytes,
            "late_shards": order[FIRST_HALF:]}


@pytest.fixture
def capped(monkeypatch):
    """The walk's `stat` budget at `CAP_FILES`, so 60 files already overflow it."""
    monkeypatch.setattr(disk_spill, "MAX_WALK_FILES", CAP_FILES)
    return CAP_FILES


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A configured tier rooted in `tmp_path` and a skewed 60-file store inside it.

    The tmpfs refusal is cleared for the prefix list only (pytest's `tmp_path` lives
    under `/tmp`), which is what `test_disk_spill.armed_tier` does for the same
    reason. Nothing here is armed: these tests call `_enforce_ceiling()` directly,
    the way `test_disk_spill.py` does.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    root = tmp_path / "kv-store"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(CEILING_BYTES))
    disk_spill.configure(fingerprint())
    # `configured()` is the module's own door from the environment into `_state`
    # (root and ceiling): `_enforce_ceiling` reads `_state`, not the environment.
    disk_spill.configured()
    geo = disk_spill.geometry_digest()
    truth = build_skewed_store(root, geo)
    return {"root": root, "geo": geo, "truth": truth}


@pytest.fixture
def channel(tmp_path, monkeypatch):
    """A diagnostics directory and a rank: the whole requirement for a spill journal.

    Without it a removal cannot be journalled, and "every file the ceiling deleted
    is journalled" is one of the properties this change must not weaken.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    return directory


def journal_lines(directory: Path) -> list:
    where = disk_index.journal_path()
    assert where["ok"] is True, where["reason"]
    if not os.path.exists(where["path"]):
        return []
    with open(where["path"], "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def sizes_of(paths) -> int:
    return sum(path.stat().st_size for path in paths)


def surviving(paths) -> list[Path]:
    return [path for path in paths if path.exists()]


def shards_in(entries) -> set:
    return {os.path.basename(os.path.dirname(str(row["path"])))
            for row in entries}


# --- 1. a truncated walk estimates the store instead of reporting a floor ----

def test_a_truncated_walk_reports_both_what_it_saw_and_what_the_store_holds(store,
                                                                            capped):
    """The defect, measured. The sample is a FLOOR of the truth, and the old code
    handed that floor to the ceiling as if it were the measurement.

    In this fixture the old depth-first walk stopped inside the small shards and
    reported `CAP_FILES * SMALL_BYTES` = 24 KiB for a 15 MiB store: 0.15% of the
    truth, and under any ceiling worth the name.
    """
    truth = store["truth"]
    walk = disk_spill._walk_store(store["root"])
    # What the OLD code would have decided on: 24 files, all taken from the early
    # (small) shards, because the file counter was never reset per shard.
    old_floor = CAP_FILES * truth["small_bytes"]

    assert walk["ok"] is True
    assert walk["truncated"] is True, "60 files against a 24-file cap is a sample"
    assert walk["files"] == CAP_FILES
    # The sample now spans the shard space, so `used_bytes` holds big files too - but
    # it is still a floor of the truth, which is why it cannot be the decision number.
    assert walk["used_bytes"] == (CAP_FILES // 2) * truth["small_bytes"] \
        + (CAP_FILES // 2) * truth["big_bytes"]
    assert walk["used_bytes"] < walk["estimated_total_bytes"]
    assert walk["used_bytes"] < truth["total_bytes"]
    # The scaled estimate is exact here: the sample takes the same fraction of every
    # shard, and the sizes within a shard are identical.
    assert walk["estimated_total_bytes"] == truth["total_bytes"]
    assert walk["estimated_total_bytes"] > 600 * old_floor, \
        "the floor the old code decided on was three orders of magnitude under"
    assert walk["total_files"] == len(truth["small"]) + len(truth["big"])
    assert walk["measurement_basis"] == "sample_scaled"
    assert walk["count_truncated"] is False, "60 entries is nowhere near the entry cap"


def test_the_scaled_measurement_is_close_even_when_the_shards_disagree(store, capped,
                                                                       monkeypatch):
    """The scaling is an estimate, so say how good it is rather than claim exactness.

    A second store whose shards disagree in SIZE as well as in count: one shard holds
    a tenth of the files and 90% of the bytes. A uniform-rate sample still lands
    within a factor of two, where the depth-first floor was off by an order of
    magnitude.
    """
    root = store["root"] / "skewed"
    geo = store["geo"]
    order = _shard_order(root, 5)
    monkeypatch.setattr(disk_spill, "MAX_WALK_FILES", 10)
    sizes = []
    # The BIG files are in the LAST shard in scandir order: the depth-first walk read
    # ten small ones and reported 10 KiB for a 2 MB store.
    for position, shard in enumerate(order):
        size = 200_000 if position == len(order) - 1 else 1024
        for index in range(10):
            _write(Path(slot_path(root, geo, shard, index)["path"]), size,
                   1_600_000_000.0)
            sizes.append(size)
    truth = sum(sizes)

    walk = disk_spill._walk_store(root)
    # What the depth-first walk would have handed the ceiling: the ten small files of
    # the first shard it read, and nothing else.
    old_floor = 10 * 1024
    assert walk["truncated"] is True
    assert walk["total_files"] == len(sizes)
    assert truth / 2 <= walk["estimated_total_bytes"] <= truth * 2
    # ...and the floor the old code decided on was two orders of magnitude under the
    # truth, which is the whole bug in one assertion.
    assert walk["estimated_total_bytes"] > 100 * old_floor
    assert walk["used_bytes"] < truth / 4


def test_the_sample_covers_every_shard_so_the_plan_can_reach_the_late_files(store,
                                                                            capped):
    """Option (b): the budget is spread across the shard space, not depth-first.

    The old walk's entries came from the first 2.4 shards and nothing else. These
    came from all six, at the same rate in each - which is what makes an
    oldest-first plan able to act on a file in the last shard it reads.
    """
    truth = store["truth"]
    walk = disk_spill._walk_store(store["root"])

    assert shards_in(walk["entries"]) == set(truth["order"]), \
        "a sample that skips a shard is the same blindness in a smaller hat"
    per_shard: dict = {}
    for row in walk["entries"]:
        shard = os.path.basename(os.path.dirname(str(row["path"])))
        per_shard[shard] = per_shard.get(shard, 0) + 1
    # 24 files over 6 shards at a uniform rate: 4 each, i.e. 40% of every shard.
    assert set(per_shard.values()) == {CAP_FILES // SHARDS}
    big_sampled = [row for row in walk["entries"]
                   if int(row["size_bytes"]) == truth["big_bytes"]]
    assert big_sampled, "the big bytes are the whole point of the sample"
    assert {os.path.basename(os.path.dirname(str(row["path"])))
            for row in big_sampled} == set(truth["late_shards"])


def test_the_eviction_plan_selects_files_from_the_shards_the_old_walk_never_reached(
        store, capped):
    truth = store["truth"]
    walk = disk_spill._walk_store(store["root"])
    plan = disk_layout.lru_eviction_plan(
        walk["entries"], geometry_digest=store["geo"],
        ceiling_bytes=CEILING_BYTES, used_bytes=walk["estimated_total_bytes"],
        target_fraction=disk_spill.EVICT_TARGET_FRACTION)

    assert plan["ok"] is True
    assert plan["needed"] is True
    assert plan["paths"], "a 15 MiB store under a 4 MiB ceiling must evict something"
    planned_shards = {os.path.basename(os.path.dirname(path))
                      for path in plan["paths"]}
    assert planned_shards & set(truth["late_shards"]), \
        "the plan must be able to name files in the late shards"
    # Oldest first: every big (old) file is shed before any small (recent) one.
    big_planned = [path for path in plan["paths"]
                   if path in [str(item) for item in truth["big"]]]
    small_planned = [path for path in plan["paths"]
                     if path in [str(item) for item in truth["small"]]]
    assert big_planned and len(big_planned) >= len(small_planned)


# --- 2. the ceiling decision itself -----------------------------------------

def test_the_ceiling_sheds_the_big_files_the_old_walk_could_never_see(store, capped,
                                                                      channel):
    """THE regression this whole change exists for, as a behaviour, not a field.

    Before the fix this test fails on `evictions == 0`: the walk handed the plan a
    24 KiB floor, `lru_eviction_plan` compared it to a 4 MiB target and reported
    "the store is under its target", and 15 MiB of KV bytes sat under a 4 MiB
    ceiling forever - while the shards the walk did read kept shedding.
    """
    truth = store["truth"]
    disk_spill._state["estimated_bytes"] = truth["total_bytes"]
    disk_spill._enforce_ceiling()

    assert disk_spill._state["evictions"] > 0, "the ceiling must do something"
    assert surviving(truth["big"]), "some big files must survive a single pass"
    assert len(surviving(truth["big"])) < len(truth["big"]), \
        "a pass that could only see the small shards deleted none of the big ones"
    # Every deletion is journalled: an un-journalled removal leaves the index
    # offering a dead file, which costs a reported load failure, not a recompute.
    assert disk_spill._state["journal_failures"] == 0
    assert disk_spill._state["journal_removals"] >= disk_spill._state["evictions"]
    removals = [line for line in journal_lines(channel)
                if len(line.split(" ")) > 1 and line.split(" ")[1] == "-"]
    assert removals, journal_lines(channel)


def test_the_passes_converge_to_the_target_because_enforcement_runs_after_every_write(
        store, capped):
    """One truncated pass is PARTIAL by design. The next spill runs another.

    `lru_eviction_plan` can only shed what the sample showed it, so a store 4x over
    its ceiling takes several passes; `_handle_job` calls `_enforce_ceiling` after
    every committed write, and each delete frees a directory slot so the next
    sample sees different files. Bounded here, and convergent.
    """
    truth = store["truth"]
    disk_spill._state["estimated_bytes"] = truth["total_bytes"]
    target = int(CEILING_BYTES * disk_spill.EVICT_TARGET_FRACTION)
    before = sizes_of(list(store["root"].rglob("*.bin")))

    passes = 0
    for _ in range(10):
        disk_spill._enforce_ceiling()
        passes += 1
        if int(disk_spill._state["estimated_bytes"]) < CEILING_BYTES:
            break
    after = sizes_of(list(store["root"].rglob("*.bin")))

    assert after < before, "the store must actually shrink"
    assert after <= target, (f"the store must land under the target: {after} > "
                             f"{target}")
    assert passes < 10, "convergence must not need an unbounded number of passes"
    assert disk_spill._state["ceiling_checks"] == passes
    # The estimate agrees with the bytes on disk, within the sample's own error.
    assert int(disk_spill._state["estimated_bytes"]) <= target


def test_a_ceiling_that_the_estimate_is_under_still_does_no_work(store, capped):
    """The fast path is untouched: no walk, no plan, no unlink, no new state."""
    truth = store["truth"]
    disk_spill._state["estimated_bytes"] = truth["total_bytes"] - 1
    disk_spill._state["ceiling_bytes"] = truth["total_bytes"]
    disk_spill._enforce_ceiling()
    assert disk_spill._state["ceiling_checks"] == 0
    assert disk_spill._state["evictions"] == 0
    assert len(list(store["root"].rglob("*.bin"))) == 60


# --- 3. a store inside the caps is measured, and byte-identically to before --

def test_a_store_inside_the_caps_is_measured_and_not_scaled(store):
    """THE BEHAVIOUR PIN. Below the caps the two numbers are ONE number.

    This tier is live on a box serving real traffic: for any store the walk can
    measure, nothing may have moved. Same bytes, every file present, no truncation,
    basis `full_measurement`, and `estimated_total_bytes` equal to `used_bytes`
    rather than a scaled version of it.
    """
    truth = store["truth"]
    monkeyfree = disk_spill._walk_store(store["root"],
                                        max_files=len(truth["small"])
                                        + len(truth["big"]))
    assert monkeyfree["truncated"] is False
    assert monkeyfree["count_truncated"] is False
    assert monkeyfree["measurement_basis"] == "full_measurement"
    assert monkeyfree["files"] == 60
    assert monkeyfree["used_bytes"] == truth["total_bytes"]
    assert monkeyfree["estimated_total_bytes"] == truth["total_bytes"]
    assert {row["path"] for row in monkeyfree["entries"]} \
        == {str(path) for path in truth["small"] + truth["big"]}


def test_the_cap_boundary_is_exactly_where_a_measurement_becomes_a_sample(store):
    """`total_files == cap` is a measurement; `cap - 1` is a sample. No drift."""
    truth = store["truth"]
    whole = len(truth["small"]) + len(truth["big"])

    at_the_line = disk_spill._walk_store(store["root"], max_files=whole)
    assert at_the_line["truncated"] is False
    assert at_the_line["measurement_basis"] == "full_measurement"
    assert at_the_line["estimated_total_bytes"] == at_the_line["used_bytes"]

    one_short = disk_spill._walk_store(store["root"], max_files=whole - 1)
    assert one_short["truncated"] is True
    assert one_short["measurement_basis"] == "sample_scaled"
    assert 0 < one_short["files"] < whole


def test_the_eviction_order_of_a_measured_store_is_the_oldest_first_order_it_always_was(
        store):
    """Not just the numbers: the PLAN. A changed eviction order is a changed store.

    The plan built from a full walk is compared against an order computed straight
    from the filesystem - oldest `st_mtime` first, path as the tie-break, which is
    `disk_layout.lru_eviction_plan`'s own rank - with the walk not involved at all.
    """
    truth = store["truth"]
    walk = disk_spill._walk_store(store["root"], max_files=60)
    plan = disk_layout.lru_eviction_plan(
        walk["entries"], geometry_digest=store["geo"],
        ceiling_bytes=CEILING_BYTES, used_bytes=walk["estimated_total_bytes"],
        target_fraction=disk_spill.EVICT_TARGET_FRACTION)
    on_disk = sorted(truth["big"] + truth["small"],
                     key=lambda path: (path.stat().st_mtime_ns, str(path)))
    expected = [str(path) for path in on_disk]

    assert plan["needed"] is True
    assert plan["planned"] > 0
    assert plan["paths"] == expected[:plan["planned"]], \
        "a full walk's plan must be the oldest-first order the filesystem itself says"


# --- 4. the witness: a reader must be able to tell a sample from a measurement

def test_the_published_witness_says_the_decision_came_from_a_sample(store, capped):
    truth = store["truth"]
    disk_spill._state["estimated_bytes"] = truth["total_bytes"]
    disk_spill._enforce_ceiling()
    section = disk_spill.section()

    assert section["walk_truncated"] is True
    witness = section["ceiling_measurement"]
    assert witness["basis"] == "sample_scaled"
    assert witness["sampled_files"] == CAP_FILES
    assert witness["total_files"] == 60
    assert witness["sampled_files"] < witness["total_files"]
    # The measured bytes and the number the ceiling was decided on are BOTH
    # published, so nobody has to reverse-engineer the scaling from two sections.
    assert witness["measured_bytes"] == disk_spill._state["walk_measured_bytes"]
    assert witness["measured_bytes"] < section["estimated_bytes"]
    assert section["estimated_bytes"] == disk_spill._state["estimated_bytes"]
    assert witness["walk_truncated"] is True
    assert witness["count_truncated"] is False
    assert witness["ceiling_checks"] >= 1
    # The witness is on the polled snapshot, not only on a verb nobody curls.
    from kvtier_dev_plugin import snapshot
    view = snapshot.dev_state_sections()
    assert view["disk_spill"]["ceiling_measurement"]["basis"] == "sample_scaled"
    assert json.dumps(section, default=str)


def test_a_full_measurement_is_published_as_a_measurement(store):
    """The other side of the witness: a store the walk measured says so.

    60 files against the shipped 20,000-file cap is a MEASUREMENT, and the payload
    must not hedge it into looking like a sample.
    """
    disk_spill._state["estimated_bytes"] = 10 ** 15
    disk_spill._state["ceiling_bytes"] = CEILING_BYTES
    disk_spill._enforce_ceiling()
    witness = disk_spill.section()["ceiling_measurement"]

    assert witness["basis"] == "full_measurement"
    assert witness["walk_truncated"] is False
    assert witness["count_truncated"] is False
    assert witness["sampled_files"] == 60
    assert witness["sampled_files"] == witness["total_files"]
    assert witness["measured_bytes"] == disk_spill._state["walk_measured_bytes"]


def test_the_state_counters_copy_carries_the_witness(store, capped):
    disk_spill._state["estimated_bytes"] = store["truth"]["total_bytes"]
    disk_spill._enforce_ceiling()
    counters = disk_spill.counters()
    for key in ("walk_truncated", "walk_basis", "walk_sampled_files",
                "walk_total_files", "walk_measured_bytes", "ceiling_checks",
                "walk_entries_examined", "walk_skipped_files"):
        assert key in counters, key


# --- 5. every pass is bounded ----------------------------------------------

def test_the_counting_pass_is_bounded_by_entries_and_extrapolates_when_it_stops(
        store, capped, monkeypatch):
    """A count that stops early is an estimate too, and must be labelled as one.

    `MAX_WALK_ENTRIES` bounds directory ITERATIONS (not files), so the bound holds
    even in a store full of entries this tier will not measure - which is the case
    the old walk had no bound on at all: its cap counted statted FILES, so a shard
    of a million non-files was iterated to the end.
    """
    truth = store["truth"]
    monkeypatch.setattr(disk_spill, "MAX_WALK_ENTRIES", 30)
    walk = disk_spill._walk_store(store["root"])

    # Each of the two passes gets the bound; the walk as a whole reads at most twice
    # it, and `entries_examined` is the honest sum of both.
    assert walk["entries_examined"] <= 2 * 30, "the counting bound is the point of it"
    assert walk["count_truncated"] is True
    assert walk["truncated"] is True
    # Six uniform shards, ten files each: the mean of the shards it finished
    # extrapolates to the store's real size, and the basis says it extrapolated.
    assert walk["total_files"] == 60
    assert walk["measurement_basis"] == "sample_scaled_count_extrapolated"
    assert walk["estimated_total_bytes"] > walk["used_bytes"]
    assert truth["total_bytes"] // 3 <= walk["estimated_total_bytes"] \
        <= truth["total_bytes"] * 3


def test_the_measuring_pass_still_stops_at_its_stat_cap(store):
    """Two bounds, both honoured: entries and stats. Neither is unbounded."""
    walk = disk_spill._walk_store(store["root"], max_files=5, max_entries=1000)
    assert walk["files"] <= 5
    assert walk["truncated"] is True
    assert walk["entries_examined"] <= 2 * 1000


def test_files_this_tier_cannot_measure_are_counted_rather_than_invisible(store):
    """The promise the old walk made in a comment and never kept.

    A `.part` left by a killed spill, or someone else's file, holds real bytes that
    this tier will neither measure nor shed. They stay out of `estimated_bytes`
    (correct: they are not this tier's to delete) but they must not be invisible to
    the reader of the witness, or the ceiling logic under-counts a store it cannot
    read and says nothing about it.
    """
    truth = store["truth"]
    shard = store["truth"]["order"][0]
    directory = store["root"] / shard
    leftover = directory / (str(slot_path(store["root"], store["geo"], shard,
                                            900)["filename"]) + ".part")
    _write(leftover, 5_000_000, 1_600_000_000.0)
    _write(directory / "not-ours-at-all.bin", 5_000_000, 1_600_000_000.0)

    walk = disk_spill._walk_store(store["root"], max_files=200)
    assert walk["skipped_files"] == 2
    assert walk["total_files"] == 60, "invisible files must not inflate the count"
    assert walk["used_bytes"] == truth["total_bytes"]
    assert all(not str(row["path"]).endswith(".part") for row in walk["entries"])
    # And they are never deleted by the capacity path, which is why the count - and
    # not a byte total - is the honest thing to publish for them.
    disk_spill._state["estimated_bytes"] = 10 ** 15
    disk_spill._state["ceiling_bytes"] = CEILING_BYTES
    disk_spill._enforce_ceiling()
    assert leftover.exists(), "a .part is not this plan's to delete"


def test_the_walk_still_refuses_to_follow_a_link(store):
    """Unchanged from the old walk: a symlink is skipped, never followed."""
    truth = store["truth"]
    shard = truth["order"][0]
    link = store["root"] / shard / "kvslot-a3f0b7c1d4e5f607-g3-0123456789ab.bin"
    os.symlink(str(truth["small"][0]), link)
    try:
        walk = disk_spill._walk_store(store["root"], max_files=200)
        assert str(link) not in {row["path"] for row in walk["entries"]}
        assert walk["total_files"] == 60
    finally:
        os.unlink(link)


def test_a_missing_store_is_an_absence_and_not_a_zero(tmp_path):
    """`used_bytes: 0` would read as "the store is empty"; a failed walk says nothing.

    The old failure payload carried a real-looking 0. Nothing in this repository
    reads a number out of a failed walk, which is exactly what this pins.
    """
    walk = disk_spill._walk_store(tmp_path / "definitely-not-here")
    assert walk["ok"] is False
    assert walk["used_bytes"] is None
    assert walk["estimated_total_bytes"] is None
    assert walk["measurement_basis"] == "unavailable"
    assert walk["truncated"] is True
    json.dumps(walk, default=str)


def test_the_counting_pass_is_reachable_only_through_the_walk():
    """`op=contains` replaces `_walk_store` with something that RAISES, and must stay
    walking-free (`test_disk_contains_verb.py`). The new counting pass therefore has
    to live BEHIND `_walk_store`, not beside it - otherwise a polled verb gains a
    directory scan, which is the self-inflicted outage that test was written for.
    """
    import ast

    tree = ast.parse(Path(disk_spill.__file__).read_text(encoding="utf-8"))
    doors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = {getattr(child, "id", None)
                     for child in ast.walk(node)
                     if isinstance(child, ast.Name)}
            if "_count_store" in names:
                doors.append(node.name)
    assert doors == ["_walk_store"], doors


# --- 6. the seed: a fresh process must see the store it did not write -------
#
# `estimated_bytes` is a per-process counter of THIS process's own committed writes
# and starts at 0, and the walk that could correct it only ran once that counter ALONE
# reached the ceiling. On the box tonight that meant a fresh boot had to spill 192 GiB
# of NEW files (17,384 of them, ~48 min at the measured ~4 GiB/min) before it was
# allowed to LOOK at a 335 GiB store with 60 GiB of `/home` left. The filesystem fills
# first, and honest scaling inside the walk buys nothing when the walk never runs.

ONE_SLOT = SMALL_BYTES          # what a fresh rank's first committed write adds


def test_a_process_that_wrote_nothing_otherwise_never_sees_the_store_it_booted_into(
        store, capped):
    """THE hole the ceiling could not fall into, shown as a behaviour.

    The store holds 15 MiB; the ceiling is 4 MiB; this process has written one slot.
    Before the seed the ceiling does exactly nothing - not because the measurement is
    wrong, but because it is never taken. After the seed it converges.
    """
    truth = store["truth"]
    disk_spill._state["estimated_bytes"] = ONE_SLOT

    # The hole, asserted explicitly so this test cannot pass by accident: the same
    # store, the same ceiling, one enforcement pass, and no walk is even attempted.
    disk_spill._enforce_ceiling()
    assert disk_spill._state["ceiling_checks"] == 0
    assert disk_spill._state["evictions"] == 0
    assert len(list(store["root"].rglob("*.bin"))) == 60

    # The seed: one bounded walk, and the estimate becomes the store's size.
    disk_spill._maybe_seed_estimate()
    assert disk_spill._state["estimate_seed_walks"] == 1
    assert disk_spill._state["estimated_bytes"] == truth["total_bytes"]
    assert disk_spill._state["walk_total_files"] == 60

    disk_spill._enforce_ceiling()
    assert disk_spill._state["ceiling_checks"] >= 1
    assert disk_spill._state["evictions"] > 0
    assert len(list(store["root"].rglob("*.bin"))) < 60


def test_the_live_recovery_converges_from_a_fresh_boot_over_a_full_store(store,
                                                                          capped):
    """The operator's scenario, end to end: boot, arm, and a store already 4x over.

    `_handle_job` runs the seed and then the enforcement, per committed write, so the
    loop below is the same two calls in the same order for the same number of writes.
    It must land under the ceiling within a few passes - not after 192 GiB of new
    spill, which is what the box does not have.
    """
    truth = store["truth"]
    # The precondition this test is about: the store on disk is already 4x the ceiling
    # while this process has written one slot.
    assert truth["total_bytes"] > 3 * CEILING_BYTES
    disk_spill._state["estimated_bytes"] = ONE_SLOT

    spills = 0
    while spills < 10:
        spills += 1
        disk_spill._maybe_seed_estimate()
        disk_spill._enforce_ceiling()
        if int(disk_spill._state["estimated_bytes"]) < CEILING_BYTES:
            break

    assert spills < 10, f"did not converge in 10 writes ({spills})"
    assert int(disk_spill._state["estimated_bytes"]) < CEILING_BYTES
    assert disk_spill._state["estimate_seed_walks"] == 1, \
        "exactly one seed per configuration, whatever the number of passes"
    assert disk_spill._state["ceiling_checks"] == spills
    after = sizes_of(list(store["root"].rglob("*.bin")))
    assert after <= int(CEILING_BYTES * disk_spill.EVICT_TARGET_FRACTION)


def test_the_seed_happens_once_per_configuration_and_not_once_per_spill(store, capped):
    """One walk per (root, ceiling) pair. A spill path must not pay it again.

    The bound on the seed's cost is the point of the key: without it the enforcement
    path would walk before every write, which is exactly the I/O storm the caps exist
    to prevent.
    """
    disk_spill._state["estimated_bytes"] = ONE_SLOT
    for _ in range(5):
        disk_spill._maybe_seed_estimate()
    assert disk_spill._state["estimate_seed_walks"] == 1

    # A different ceiling is a different decision to make, so it re-seeds once.
    disk_spill._state["ceiling_bytes"] = CEILING_BYTES * 2
    disk_spill._maybe_seed_estimate()
    assert disk_spill._state["estimate_seed_walks"] == 2
    disk_spill._maybe_seed_estimate()
    assert disk_spill._state["estimate_seed_walks"] == 2


def test_a_store_that_cannot_be_read_leaves_the_running_estimate_alone(tmp_path,
                                                                       monkeypatch):
    """A failed seed is an absence of knowledge, never a 0.

    And it is attempted ONCE per configuration: a store that cannot be read must not
    buy a walk on every spill of the boot - that is the storm the bounds are for.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(tmp_path / "not-yet-a-store"))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(CEILING_BYTES))
    disk_spill.configure(fingerprint())
    disk_spill.configured()
    disk_spill._state["estimated_bytes"] = 123_456

    disk_spill._maybe_seed_estimate()
    disk_spill._maybe_seed_estimate()

    assert disk_spill._state["estimated_bytes"] == 123_456, "no 0 written into it"
    assert disk_spill._state["estimate_seed_walks"] == 1, \
        "one attempt per configuration, not one per spill"
    assert "could not be seeded" in disk_spill._state["last_reason"]


def test_the_seed_and_the_store_it_reported_are_both_published(store, capped):
    """`seed_walks` in the witness, because "the estimate is 15 MiB" and "the estimate
    is 15 MiB because a fresh process finally looked" are different claims - and the
    number that shows a ceiling that never woke up is a seed of 0 over a full store.
    """
    truth = store["truth"]
    disk_spill._state["estimated_bytes"] = ONE_SLOT
    disk_spill._maybe_seed_estimate()
    witness = disk_spill.section()["ceiling_measurement"]
    assert witness["seed_walks"] == 1
    assert witness["total_files"] == 60
    assert witness["entries_examined"] <= 2 * disk_spill.MAX_WALK_ENTRIES
    assert witness["measured_bytes"] < truth["total_bytes"]
    # The estimate itself: one slot's worth before the seed, the store's size after.
    assert disk_spill.section()["estimated_bytes"] == truth["total_bytes"]
    assert json.dumps(disk_spill.section(), default=str)


def test_no_ceiling_or_no_root_means_no_seed_walk_at_all(store, capped):
    """The seed is for a decision that is going to be made. No ceiling, no walk."""
    disk_spill._state["ceiling_bytes"] = None
    disk_spill._maybe_seed_estimate()
    assert disk_spill._state["estimate_seed_walks"] == 0
    disk_spill._state["ceiling_bytes"] = CEILING_BYTES
    disk_spill._state["root"] = None
    disk_spill._maybe_seed_estimate()
    assert disk_spill._state["estimate_seed_walks"] == 0


def test_the_status_verb_publishes_how_the_ceiling_was_measured(store, capped):
    """`op=status` is the door an operator reads when the disk is 99% full.

    `disk_store._op_status` builds its `spill` block from an explicit whitelist, so a
    witness that lives only in `disk_spill.section()` is invisible exactly where it is
    needed: "the store is 396 GiB" and "the store is 396 GiB, extrapolated from 20,000
    of 34,896 files" have to be two different payloads over HTTP too. And the number
    that tells the operator the ceiling never woke up is `estimate_seed_walks: 0` over
    a full store, which is worth nothing if the verb does not carry it.
    """
    from kvtier_dev_plugin import disk_store

    truth = store["truth"]
    disk_spill._state["estimated_bytes"] = ONE_SLOT
    disk_spill._maybe_seed_estimate()
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    assert answer["ok"] is True, answer.get("reason")
    spill = answer["spill"]
    assert spill["estimated_bytes"] == truth["total_bytes"]
    assert spill["estimate_seed_walks"] == 1
    assert spill["walk_total_files"] == 60
    assert spill["walk_sampled_files"] <= disk_spill.MAX_WALK_FILES
    assert spill["walk_basis"] in ("full_measurement", "sample_scaled",
                                   "sample_scaled_count_extrapolated",
                                   "sample_unmeasured", "unavailable")
    assert spill["ceiling_checks"] == 0, "the seed walked; it did not decide"
    assert json.dumps(answer, default=str)


# --- the bounds themselves are the contract --------------------------------

def test_the_seed_runs_from_the_writer_thread_and_nowhere_else():
    """The seed belongs on the spill path and nowhere else.

    Not at import, not in `configure()`, not in a polled status handler: it walks, and
    the doors that must never walk (`op=contains`, the snapshot section) are the whole
    reason this module has a walking budget at all. `_handle_job` is the one place a
    byte was just committed, so a walk there is already paid for by the design.
    """
    import ast

    tree = ast.parse(Path(disk_spill.__file__).read_text(encoding="utf-8"))
    doors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = {getattr(child, "id", None)
                     for child in ast.walk(node)
                     if isinstance(child, ast.Name)}
            if "_maybe_seed_estimate" in names:
                doors.append(node.name)
    assert doors == ["_handle_job"], doors


def test_the_walk_bounds_are_stated_in_the_currency_each_one_pays():
    """Three caps, three currencies - and none of them is "unbounded".

    Directories, `stat` calls, directory entries. The entry bound is what makes the
    COUNT cheap enough to run at all, and it is deliberately several times the stat
    cap: a full count is exact for any store this box can hold (100,000 files at the
    widest measured slot is ~1.2 TiB), and past that it extrapolates and says so.
    """
    assert disk_spill.MAX_WALK_ENTRIES > disk_spill.MAX_WALK_FILES
    assert disk_spill.MAX_WALK_ENTRIES // disk_spill.MAX_WALK_FILES == 5
    assert disk_spill.MAX_SHARD_DIRS >= 256, "the shard space is 256 wide"


def test_a_store_of_junk_cannot_make_the_walk_run_to_the_end_of_a_directory(store,
                                                                             monkeypatch):
    """The bound that did not exist before, and what a bounded walk refuses to do.

    The old cap counted statted FILES, so a directory full of entries this tier will
    not measure (a `.part` from a killed spill, another program's files) was
    iterated to the end - an unbounded readdir inside the spill writer's enforcement
    path. `MAX_WALK_ENTRIES` bounds the ITERATION, and a walk that then measures
    NOTHING reports `sample_unmeasured` instead of writing a 0 into the estimate.
    """
    root = store["root"] / "junk"
    order = _shard_order(root, 1)
    for index in range(2000):
        _write(root / order[0] / f"not-ours-{index}.bin", 8, 1_600_000_000.0)
    monkeypatch.setattr(disk_spill, "MAX_WALK_ENTRIES", 50)
    # The enforcement test below asks about THIS store, so this is the root the
    # ceiling believes it owns.
    monkeypatch.setitem(disk_spill._state, "root", str(root))

    walk = disk_spill._walk_store(root)
    assert walk["entries_examined"] <= 2 * 50, "the iteration is what the bound bounds"
    assert walk["count_truncated"] is True
    assert walk["truncated"] is True
    assert walk["files"] == 0
    assert walk["measurement_basis"] == "sample_unmeasured"
    assert walk["skipped_files"] > 0

    # And the enforcement path declines to act on it, keeping the estimate it had.
    disk_spill._state["ceiling_bytes"] = CEILING_BYTES
    disk_spill._state["estimated_bytes"] = CEILING_BYTES
    disk_spill._enforce_ceiling()
    assert disk_spill._state["estimated_bytes"] == CEILING_BYTES, \
        "a walk that measured nothing must not lower the estimate to its floor"
    assert disk_spill._state["evictions"] == 0
    assert disk_spill._state["walk_basis"] == "sample_unmeasured"
    assert "could not be measured" in disk_spill._state["last_reason"]
