r"""The SSD tier's layout layer: names, offsets, the index codec, the boot sweep.

Stage 1 is inert - nothing imports `disk_layout` into the engine path yet - which
is exactly why it gets a real test file before stage 2 touches it. Everything
here is proven against a `tmp_path` with no engine, no network, no CUDA and no
`/tmp/kv-tier*` dependency.

What this file is obliged to prove, in order of what each failure would cost on
the live box:

1. OFFSET MATH. A wrong offset is a read of the wrong KV, which is a wrong token
   and not a miss. Slot 0, the last slot, one past the end, a non-boundary offset
   and a partial trailing write are each their own test.
2. THE VISIBILITY RULE. A record is visible only if fully written up to a
   committed watermark. That is tested as a PROPERTY over every truncation length,
   not as one example: a torn-tail heuristic that is right for the length someone
   happened to test is not a rule.
3. THE SWEEP MUST NOT DELETE ANYTHING IT DOES NOT OWN. A near-miss name, a peer
   rank's file, this boot's own file, a symlink, a directory, a traversal path, a
   geometry that is merely different - each must survive.
4. BOUNDS. A corrupt length must not make the reader allocate; a big index must
   not be materialised; a sweep must stop at its cap and say so.
5. UNITS. Every number in every payload declares its unit, because unit confusion
   in `block_bytes` is failure mode 1.
6. Purity. The engine-side read path must never pull torch/vllm in, so the
   import rule is asserted by AST over this file's source, as elsewhere in the
   suite.
"""

from __future__ import annotations

import ast
import os
import struct
import sys
from pathlib import Path

import pytest

from kvtier_dev_plugin import disk_layout as dl

PKG = Path(dl.__file__).resolve().parent

# The live geometry, so a test that reads a number is reading the real one and
# not a toy: ~11.2 MB per slot on PP0, ~12.6 MB on PP1, 4 ranks.
PP0_BYTES = 11_200_000
PP1_BYTES = 12_600_000
EPOCH = "ab12cd34"


def geometry(pp=0, tp=0, block_bytes=PP0_BYTES, epoch=EPOCH, groups=6,
             world=4, **extra):
    built = dl.build_geometry(epoch=epoch, pp_rank=pp, tp_rank=tp,
                              world_size=world, block_bytes=block_bytes,
                              num_groups=groups, **extra)
    assert built["ok"], built["reason"]
    return built["geometry"]


def name_of(geom, suffix=dl.SUFFIX_DATA):
    made = dl.build_filename(geom, suffix)
    assert made["ok"], made["reason"]
    return made["filename"]


def record(seq, geom=None, slot=None, key=None, block_bytes=None):
    geom = geom or geometry()
    made = dl.encode_index_record(
        seq=seq, key_digest=key if key is not None else bytes([seq % 251]) * 32,
        slot=seq if slot is None else slot,
        block_bytes=geom["block_bytes"] if block_bytes is None else block_bytes,
        geometry_digest=geom["geometry_digest"])
    assert made["ok"], made["reason"]
    return made["frame"]


def commit(seq, entries):
    made = dl.encode_index_commit(committed_seq=seq, entries_in_commit=entries)
    assert made["ok"], made["reason"]
    return made["frame"]


def write_index(directory, frames, name="index.idx"):
    path = Path(directory) / name
    path.write_bytes(b"".join(frames))
    return path


# --- purity: the read side stays light --------------------------------------

HEAVY = {"torch", "vllm", "cuda", "numpy", "triton", "numba", "cupy", "psutil"}


def test_disk_layout_imports_nothing_heavy():
    """By AST, so an import hidden inside a branch does not pass."""
    tree = ast.parse(Path(dl.__file__).read_text(encoding="utf-8"))
    pulled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pulled.add(node.module.split(".")[0])
    assert not (pulled & HEAVY), sorted(pulled & HEAVY)
    assert pulled <= {*STDLIB_ALLOWED, "disk_layout"}, sorted(
        pulled - STDLIB_ALLOWED)


STDLIB_ALLOWED = {"hashlib", "os", "struct", "zlib", "__future__"}


def test_importing_it_pulls_in_nothing_heavy():
    for name in ("torch", "vllm"):
        assert name not in sys.modules
    geometry()
    for name in ("torch", "vllm"):
        assert name not in sys.modules


# --- geometry and names ------------------------------------------------------

def test_a_geometry_carries_its_units_and_a_digest_of_the_LAYOUT_only():
    geom = geometry()
    assert geom["rank"] == "PP0_TP0"
    assert geom["block_bytes"] == PP0_BYTES
    assert geom["units"]["block_bytes"] == "bytes"
    assert geom["units"]["world_size"] == "ranks"
    # The digest is of the slot's LAYOUT, and the rank is its own NAME field, so
    # every rank of one boot shares a digest and still gets its own file.
    assert geometry(tp=1)["geometry_digest"] == geom["geometry_digest"]
    assert geometry(pp=1)["geometry_digest"] == geom["geometry_digest"]
    assert name_of(geometry(pp=1)) != name_of(geom)
    assert name_of(geometry(tp=1)) != name_of(geom)
    # Layout facts DO move it: a different slot size, group count or world size
    # is a different meaning for every offset in the file.
    assert geometry(block_bytes=PP1_BYTES)["geometry_digest"] != \
        geom["geometry_digest"]
    assert geometry(groups=7)["geometry_digest"] != geom["geometry_digest"]
    assert geometry(world=8)["geometry_digest"] != geom["geometry_digest"]
    assert geometry(regions_digest=dl.compute_regions_digest(
        ["k", "v"]))["geometry_digest"] != geom["geometry_digest"]
    # The epoch is NOT in it, or every new boot would look like a new geometry.
    assert geometry(epoch="ffff0000")["geometry_digest"] == \
        geom["geometry_digest"]


@pytest.mark.parametrize("bad", [
    pytest.param({"block_bytes": 512}, id="a-token-count-is-not-bytes"),
    pytest.param({"block_bytes": 40 * 1024 ** 3}, id="a-pool-total-not-a-slot"),
    pytest.param({"block_bytes": 11.2e6}, id="a-float-is-not-an-int"),
    pytest.param({"block_bytes": "11200000"}, id="a-string-is-not-an-int"),
    pytest.param({"world_size": 0}, id="no-ranks"),
    pytest.param({"pp_rank": -1}, id="a-negative-rank"),
    pytest.param({"epoch": "2026-10-05-boot1"}, id="an-epoch-with-dashes"),
    pytest.param({"epoch": "a" * 40}, id="an-epoch-over-the-cap"),
    pytest.param({"epoch": "x/../y"}, id="an-epoch-with-a-path"),
])
def test_a_geometry_that_could_not_come_from_a_real_boot_is_refused(bad):
    fields = {"epoch": EPOCH, "pp_rank": 0, "tp_rank": 0, "world_size": 4,
              "block_bytes": PP0_BYTES}
    fields.update(bad)
    out = dl.build_geometry(**fields)
    assert not out["ok"] and out["reason"]
    assert "geometry" not in out       # a refusal carries no half-built layout


def test_an_epoch_with_dashes_is_refused_not_silently_rewritten():
    """Silently mapping '-' to '_' could make a stale boot look like this one."""
    raw = dl.normalize_epoch("2026-10-05-boot1")
    assert raw["ok"] and raw["changed"] and raw["epoch"] == "2026_10_05_boot1"
    built = dl.build_geometry(epoch="2026-10-05-boot1", pp_rank=0, tp_rank=0,
                              world_size=4, block_bytes=PP0_BYTES)
    assert not built["ok"] and "normalize" in built["reason"]


def test_the_name_round_trips_and_nothing_else_is_accepted():
    geom = geometry()
    filename = name_of(geom)
    assert filename == f"kvtier-{EPOCH}-PP0_TP0-g{geom['geometry_digest']}.bin"
    parsed = dl.parse_path(filename)
    assert parsed["ok"] and parsed["ours"]
    assert (parsed["epoch"], parsed["rank"], parsed["geometry_digest"]) == \
        (EPOCH, "PP0_TP0", geom["geometry_digest"])
    # The index sibling parses identically apart from its suffix.
    assert dl.parse_path(name_of(geom, dl.SUFFIX_INDEX))["suffix"] == ".idx"


@pytest.mark.parametrize("name, classification", [
    pytest.param("notes.bin", dl.NAME_NOTOURS, id="someone-elses-file"),
    pytest.param("kvtier.conf", dl.NAME_NOTOURS, id="our-prefix-wrong-suffix"),
    pytest.param("kvtier.conf.bin", dl.NAME_NOTOURS,
                 id="a-similar-name-wrong-prefix"),
    pytest.param("kvtier-backup-notes.bin", dl.NAME_MALFORMED,
                 id="our-shape-unparseable-fields"),
    pytest.param(f"kvtier-{EPOCH}-PP0_TP0-gabc.bin", dl.NAME_MALFORMED,
                 id="a-short-geometry-field"),
    pytest.param(f"kvtier-{EPOCH}-PP0TP0-g{'a' * 12}.bin", dl.NAME_MALFORMED,
                 id="a-rank-without-a-underscore"),
    pytest.param(f"kvtier-{EPOCH}-PP0_TP0-x{'a' * 12}.bin", dl.NAME_MALFORMED,
                 id="a-geometry-field-without-g"),
    pytest.param(f"kvtier-{EPOCH}-PP0_TP0-PP1_TP1-g{'a' * 12}.bin",
                 dl.NAME_MALFORMED, id="too-many-fields"),
    pytest.param("", dl.NAME_NOTOURS, id="the-empty-name"),
    pytest.param(".", dl.NAME_NOTOURS, id="this-directory"),
    pytest.param("..", dl.NAME_NOTOURS, id="the-parent-directory"),
])
def test_parse_path_is_total_and_classifies_rather_than_raising(name,
                                                                classification):
    parsed = dl.parse_path(name)
    assert parsed["ok"] is False
    assert parsed["classification"] == classification
    assert parsed["reason"]
    assert parsed["filename"] is not None
    # A name that does not parse is never, in any shape, deletable.
    assert parsed.get("unlinkable", False) is False


def test_parse_path_survives_types_that_are_not_names_at_all():
    for junk in (None, 7, b"kvtier-abc-PP0_TP0-gaaaaaaaaaaaa.bin", ["x"],
                 object(), 3.5, True):
        out = dl.parse_path(junk)
        assert out["ok"] is False and out["reason"]


def test_build_paths_joins_and_creates_nothing(tmp_path):
    built = dl.build_paths(tmp_path, geometry())
    assert built["ok"] and built["created"] is False
    assert os.path.dirname(built["data_path"]) == str(tmp_path)
    assert not os.path.exists(built["data_path"])
    assert not os.path.exists(built["index_path"])
    refused = dl.build_paths(str(tmp_path) + "\x00/x", geometry())
    assert not refused["ok"]


# --- the sweep ---------------------------------------------------------------

def touch(path, size=0):
    Path(path).write_bytes(b"\x00" * size)
    return str(path)


def test_the_sweep_plans_stale_boots_and_touches_nothing_else(tmp_path):
    current = geometry()
    ours = touch(tmp_path / name_of(current))
    stale = touch(tmp_path / name_of(geometry(epoch="ffff0000")), 4096)
    other_boot = touch(tmp_path / name_of(geometry(epoch="eeee1111"),
                                           dl.SUFFIX_INDEX))
    report = dl.boot_sweep(tmp_path, current)
    assert report["ok"] and report["directory_state"] == "ok"
    assert sorted(report["planned_paths"]) == sorted([stale, other_boot])
    assert ours not in report["planned_paths"]
    assert os.path.exists(ours)
    assert report["counts"]["current"] == 1
    assert report["counts"]["stale_epoch"] == 2
    assert report["counts"]["planned_bytes"] == 4096


def test_two_geometries_coexist_and_neither_can_evict_the_other(tmp_path):
    pp0 = geometry(pp=0, block_bytes=PP0_BYTES)
    pp1 = geometry(pp=1, block_bytes=PP1_BYTES)
    file_pp0 = touch(tmp_path / name_of(pp0))
    file_pp1 = touch(tmp_path / name_of(pp1))
    for mine, _theirs in ((pp0, file_pp1), (pp1, file_pp0)):
        report = dl.boot_sweep(tmp_path, mine)
        assert report["planned_paths"] == []
        assert report["counts"]["current"] == 1
        # The peer's file is not_ours BEFORE it is a geometry mismatch: PP0 and
        # PP1 slots are different sizes, so a digest-first order would have made
        # every rank plan to delete its peers.
        assert report["counts"][dl.NAME_NOTOURS] == 1
        assert report["counts"]["stale_geometry"] == 0
        peer = next(item for item in report["files"]
                    if item["filename"] != name_of(mine))
        assert peer["unlinkable"] is False
        assert "peer rank" in peer["reason"]
    # Both still there, and a PP0 rank never plans a PP1 file even when it IS
    # stale: the plan is rank-scoped, not epoch-scoped.
    old_pp1 = touch(tmp_path / name_of(geometry(pp=1, epoch="deadbeef",
                                                block_bytes=PP1_BYTES)))
    plan_pp0 = dl.boot_sweep(tmp_path, pp0)["planned_paths"]
    assert old_pp1 not in plan_pp0
    assert old_pp1 in dl.boot_sweep(tmp_path, pp1)["planned_paths"]


def test_a_changed_world_size_makes_another_ranks_data_stale(tmp_path):
    geom = geometry(world=4)
    touch(tmp_path / name_of(geometry(world=8, geometry_digest=
                                      geom["geometry_digest"])))
    assert dl.boot_sweep(tmp_path, geom)["planned_paths"] == []
    assert len(dl.boot_sweep(tmp_path, geom,
                             world_size_matches=False)["planned_paths"]) == 1


def test_a_similar_but_not_ours_name_is_left_alone(tmp_path):
    geom = geometry()
    strangers = [
        touch(tmp_path / "kvtier-backup-notes.bin", 11),
        touch(tmp_path / "kvtier.conf", 11),
        touch(tmp_path / f"{name_of(geom)}.bak", 11),
        touch(tmp_path / f"{name_of(geom)}.bin", 11),
        touch(tmp_path / "kvstore-ab12cd34-PP0_TP0-g000000000000.bin", 11),
    ]
    strangers.append(str(tmp_path / name_of(geometry(epoch="99999999"))) + "x")
    Path(strangers[-1]).write_bytes(b"x")
    report = dl.boot_sweep(tmp_path, geom)
    assert report["planned_paths"] == []
    # Four are simply not ours - our prefix or our suffix, but never both. Two
    # wear our EXACT shape with fields that do not parse: `...bin.bin` (a
    # geometry field with a dot in it) and `kvtier-backup-notes.bin` (two fields
    # instead of three). Those are reported as malformed and still left alone,
    # because our prefix plus our suffix is not proof the file is ours.
    assert report["counts"][dl.NAME_NOTOURS] == 4
    assert report["counts"][dl.NAME_MALFORMED] == 2
    assert report["counts"]["planned"] == 0
    for path in strangers:
        assert os.path.exists(path)


def test_a_malformed_name_is_classified_excluded_and_never_unlinked(tmp_path):
    geom = geometry()
    bad = touch(tmp_path / "kvtier-not-a-real-field-here.bin", 32)
    report = dl.boot_sweep(tmp_path, geom)
    entry = [item for item in report["files"] if item["filename"].startswith(
        "kvtier-not")]
    assert len(entry) == 1
    assert entry[0]["classification"] == dl.NAME_MALFORMED
    assert "reason" in entry[0] and entry[0]["reason"]
    assert entry[0]["unlinkable"] is False
    assert report["counts"][dl.NAME_STALE_EPOCH] == 0
    assert report["planned_paths"] == []
    assert os.path.exists(bad)


def test_a_symlink_never_escapes_the_directory(tmp_path):
    geom = geometry()
    outside = tmp_path.parent / "outside-the-sweep.bin"
    stale_name = name_of(geometry(epoch="cafe0000"))
    victim = tmp_path / stale_name
    victim.symlink_to(outside)
    Path(outside).write_bytes(b"not ours")
    inside_stale = touch(tmp_path / name_of(geometry(epoch="beef0000")), 16)
    report = dl.boot_sweep(tmp_path, geom)
    entry = next(item for item in report["files"]
                 if item["filename"] == stale_name)
    assert entry["classification"] == dl.NAME_REFUSED
    assert entry["symlink"] is True and entry["unlinkable"] is False
    assert report["counts"][dl.NAME_STALE_EPOCH] == 1
    assert report["planned_paths"] == [inside_stale]
    assert outside.exists() and victim.exists()
    # Executing the plan must not follow it either.
    executed = dl.execute_unlink_plan(tmp_path, [str(victim)], dry_run=False,
                                      geometry=geom)
    assert executed["counts"]["refused"] == 1 and executed["counts"]["unlinked"] == 0
    assert outside.exists() and victim.exists()


def test_a_directory_that_is_a_symlink_is_refused_outright(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    report = dl.boot_sweep(link, geometry())
    assert not report["ok"] and "symlink" in report["reason"]
    assert report["directory_state"] == dl.NAME_REFUSED


def test_a_subdirectory_is_refused_and_not_walked(tmp_path):
    nested = tmp_path / name_of(geometry(epoch="aaaa1111"))
    nested.mkdir()
    report = dl.boot_sweep(tmp_path, geometry())
    entry = report["files"][0]
    assert entry["classification"] == dl.NAME_REFUSED
    assert "regular file" in entry["reason"]
    assert report["planned_paths"] == []


@pytest.mark.parametrize("state", ["missing", "not-a-directory"])
def test_an_absent_or_wrong_directory_is_a_classified_state(tmp_path, state):
    geom = geometry()
    if state == "missing":
        target = tmp_path / "not-created"
    else:
        target = tmp_path / "a-file"
        target.write_bytes(b"x")
    report = dl.boot_sweep(target, geom)
    assert report["ok"]
    assert report["directory_state"] in (dl._MISSING, dl._NOT_A_DIRECTORY)
    assert report["reason"] and report["planned_paths"] == []
    # Absent is reported as absent, never as "0 files, nothing stale".
    assert report["counts"]["files"] == 0


def test_the_plan_is_dry_by_default_and_revalidates_before_deleting(tmp_path):
    geom = geometry()
    stale = touch(tmp_path / name_of(geometry(epoch="old12345")), 64)
    current = touch(tmp_path / name_of(geom), 64)
    plan = dl.boot_sweep(tmp_path, geom)["planned_paths"]
    dry = dl.execute_unlink_plan(tmp_path, plan, geometry=geom)
    assert dry["dry_run"] is True and dry["counts"]["unlinked"] == 0
    assert os.path.exists(stale)
    # A current file cannot be deleted by a caller that names it directly.
    refused = dl.execute_unlink_plan(tmp_path, [current], dry_run=False,
                                     geometry=geom)
    assert refused["counts"]["refused"] == 1 and os.path.exists(current)
    # Traversal is refused even though the file itself is a stale one.
    elsewhere = Path(tmp_path).parent / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    smuggled = touch(elsewhere / name_of(geometry(epoch="old12345")))
    assert dl.execute_unlink_plan(tmp_path, [str(smuggled)], dry_run=False,
                                  geometry=geom)["counts"]["refused"] == 1
    assert os.path.exists(smuggled)
    # The real thing.
    executed = dl.execute_unlink_plan(tmp_path, plan, dry_run=False,
                                      geometry=geom)
    assert executed["counts"]["unlinked"] == 1
    assert not os.path.exists(stale) and os.path.exists(current)
    assert executed["counts"]["planned_bytes"] == 64


def test_a_geometry_that_changed_between_planning_and_executing_saves_the_file(
        tmp_path):
    old = geometry(epoch="aaaabbbb")
    stale = touch(tmp_path / name_of(old), 32)
    plan = dl.boot_sweep(tmp_path, geometry(epoch="ccccdddd"))["planned_paths"]
    assert stale in plan
    # By execution time the operator has changed the geometry, so that file is
    # now the CURRENT one. Executing against the new geometry must refuse.
    refused = dl.execute_unlink_plan(tmp_path, [stale], dry_run=False,
                                     geometry=old)
    assert refused["counts"]["refused"] == 1 and os.path.exists(stale)


def test_the_sweep_stops_at_its_file_cap_and_says_so(tmp_path):
    geom = geometry()
    for index in range(6):
        touch(tmp_path / name_of(geometry(epoch=f"e{index:07d}")), 8)
    report = dl.boot_sweep(tmp_path, geom, max_files=3)
    assert report["walk_truncated"] is True
    assert report["counts"]["files"] == 3
    assert report["ok"]                       # capped, not failed
    small = dl.boot_sweep(tmp_path, geom, max_plan=2)
    assert small["plan_truncated"] is True
    assert len(small["planned_paths"]) == 2


@pytest.mark.parametrize("junk", ["3", None, 2.5, [2], True])
def test_the_sweep_refuses_unusable_caps_rather_than_defaulting(tmp_path, junk):
    """A float would floor and a numeric string would coerce: both lose a bound."""
    out = dl.boot_sweep(tmp_path, geometry(), max_files=junk)
    assert not out["ok"] and "max_files" in out["reason"]


# --- offset arithmetic: an off-by-one here corrupts live KV ------------------

def test_slot_zero_starts_at_the_header_and_the_stride_is_the_slot_size():
    geom = geometry()
    first = dl.offset_for_slot(geom, 0)
    assert first["byte_offset"] == 0 and first["end_offset_bytes"] == PP0_BYTES
    assert first["units"]["byte_offset"] == "bytes"
    second = dl.offset_for_slot(geom, 1)
    assert second["byte_offset"] == PP0_BYTES
    # The stride IS the per-rank slot size: 12.6 MB on PP1, not 11.2 MB.
    assert dl.offset_for_slot(geometry(pp=1, block_bytes=PP1_BYTES),
                             2)["byte_offset"] == 2 * PP1_BYTES


def test_a_header_offset_shifts_every_slot_rather_than_breaking_the_stride():
    geom = geometry()
    head = dl.encode_header(geom)["header_bytes"]
    shifted = dl.offset_for_slot(geom, 3, head)
    assert shifted["byte_offset"] == head + 3 * PP0_BYTES
    assert dl.slot_for_offset(geom, shifted["byte_offset"],
                              head)["slot"] == 3
    assert dl.slot_for_offset(geom, head - 1, head)["ok"] is False


def test_the_last_slot_is_readable_and_one_past_the_end_is_refused():
    geom = geometry()
    size = 4 * PP0_BYTES
    bounds = dl.bounds_for_file(geom, size)
    assert bounds["capacity_slots"] == 4
    assert bounds["last_complete_slot"] == 3
    assert bounds["one_past_last_slot"] == 4
    assert bounds["partial_tail_bytes"] == 0
    assert dl.slot_is_readable(geom, 0, size)["readable"] is True
    assert dl.slot_is_readable(geom, 3, size)["readable"] is True
    past = dl.slot_is_readable(geom, 4, size)
    assert past["readable"] is False and "one past" in past["reason"]
    assert dl.slot_is_readable(geom, 9, size)["readable"] is False
    assert "outside" in dl.slot_is_readable(geom, 9, size)["reason"]
    # A negative slot is not a slot that is unreadable, it is no slot at all:
    # refusal, with no number emitted for it.
    negative = dl.slot_is_readable(geom, -1, size)
    assert negative["ok"] is False and "negative" in negative["reason"]
    assert "readable" not in negative


def test_a_partial_trailing_write_is_not_a_slot():
    geom = geometry()
    size = 3 * PP0_BYTES + 7              # a crash in the middle of slot 3
    bounds = dl.bounds_for_file(geom, size)
    assert bounds["capacity_slots"] == 3
    assert bounds["has_partial_tail"] is True
    assert bounds["partial_tail_bytes"] == 7
    assert dl.slot_is_readable(geom, 2, size)["readable"] is True
    # The offset still exists, but the file cannot supply it: reading it would
    # run past EOF and call whatever follows EOF a KV block.
    assert dl.offset_for_slot(geom, 3)["byte_offset"] == 3 * PP0_BYTES
    assert dl.slot_is_readable(geom, 3, size)["readable"] is False


def test_an_offset_that_is_not_a_slot_boundary_is_refused():
    geom = geometry()
    for offset in (1, 512, PP0_BYTES + 1, 2 * PP0_BYTES - 1):
        out = dl.slot_for_offset(geom, offset)
        assert out["ok"] is False and "past the start" in out["reason"]
        assert "slot" not in out or out["slot"] is None


def test_the_arena_size_caps_reads_even_when_the_file_is_bigger():
    geom = geometry()
    size = 10 * PP0_BYTES
    assert dl.slot_is_readable(geom, 9, size,
                               capacity_slots=100)["readable"] is True
    tight = dl.slot_is_readable(geom, 5, size, capacity_slots=5)
    assert tight["readable"] is False
    assert tight["capacity_source"] == "slot_space"
    assert "slot_space" in tight["reason"]
    assert dl.slot_is_readable(geom, 4, size, capacity_slots=5)["readable"]


@pytest.mark.parametrize("bad", [
    pytest.param({"slot": -1}, id="a-negative-slot"),
    pytest.param({"slot": True}, id="a-bool-is-not-a-slot"),
    pytest.param({"slot": "3"}, id="a-string-is-not-a-slot"),
    pytest.param({"slot": dl.MAX_SLOT + 1}, id="past-the-slot-cap"),
    pytest.param({"header_bytes": -1}, id="a-negative-header"),
    pytest.param({"header_bytes": dl.MAX_HEADER_BYTES + 1},
                 id="past-the-header-cap"),
])
def test_offset_math_refuses_rather_than_clamping(bad):
    fields = {"slot": 3, "header_bytes": None}
    fields.update(bad)
    out = dl.offset_for_slot(geometry(), **fields)
    assert out["ok"] is False and "byte_offset" not in out
    # The file-size variants too: no number is ever emitted for a bad input.
    if "slot" in bad:
        probe = dl.slot_is_readable(geometry(), fields["slot"], PP0_BYTES * 4)
        assert probe["ok"] is False


def test_bounds_and_offsets_refuse_a_geometry_that_is_not_a_geometry():
    for junk in (None, "11200000", {}, {"block_bytes": None}, 11.2e6):
        assert dl.offset_for_slot(junk, 1)["ok"] is False
        assert dl.bounds_for_file(junk, 1000)["ok"] is False
        assert dl.slot_for_offset(junk, 1000)["ok"] is False


def test_the_slot_space_cap_cannot_be_negative_or_a_float():
    geom = geometry()
    for junk in (-1, 4.0, "4", True):
        assert dl.slot_is_readable(geom, 0, 4 * PP0_BYTES,
                                   capacity_slots=junk)["ok"] is False


# --- the index codec ---------------------------------------------------------

@pytest.mark.parametrize("key_len", [1, 8, 16, 31, 32, 63, 64])
@pytest.mark.parametrize("slot", [0, 1, 4095, 65_536, 1_000_000])
def test_a_record_round_trips_byte_exact(key_len, slot):
    geom = geometry()
    key = bytes((index * 7 + key_len) & 0xFF for index in range(key_len))
    encoded = dl.encode_index_record(seq=77, key_digest=key, slot=slot,
                                     block_bytes=geom["block_bytes"],
                                     geometry_digest=geom["geometry_digest"])
    assert encoded["ok"]
    assert len(encoded["frame"]) == encoded["entry_bytes"]
    back = dl.decode_index_frame(encoded["frame"])
    assert back["ok"] and back["kind"] == "data"
    assert back["key_digest"] == key          # byte-exact, same length
    assert (back["seq"], back["slot"], back["block_bytes"]) == (
        77, slot, geom["block_bytes"])
    assert back["geometry_digest"] == geom["geometry_digest"]


def test_a_hex_key_and_a_bytes_key_are_the_same_record():
    args = {"seq": 3, "slot": 9, "block_bytes": PP0_BYTES,
            "geometry_digest": geometry()["geometry_digest"]}
    raw = dl.encode_index_record(key_digest=b"\xde\xad\xbe\xef" * 8, **args)
    hexed = dl.encode_index_record(key_digest="deadbeef" * 8, **args)
    assert raw["frame"] == hexed["frame"]


@pytest.mark.parametrize("bad", [
    pytest.param({"seq": 0}, id="seq-zero-would-be-uncommitted-forever"),
    pytest.param({"seq": -1}, id="a-negative-seq"),
    pytest.param({"seq": "1"}, id="a-string-seq"),
    pytest.param({"slot": -1}, id="a-negative-slot"),
    pytest.param({"block_bytes": 512}, id="a-block-size-that-is-not-a-slot"),
    pytest.param({"key_digest": b""}, id="an-empty-key"),
    pytest.param({"key_digest": b"\x01" * 65}, id="a-key-over-the-cap"),
    pytest.param({"key_digest": 1234}, id="a-key-that-is-not-bytes"),
    pytest.param({"key_digest": "not-hex"}, id="a-key-that-is-not-hex"),
    pytest.param({"geometry_digest": "abc"}, id="a-short-geometry-digest"),
    pytest.param({"geometry_digest": "a" * 13}, id="a-long-geometry-digest"),
    pytest.param({"geometry_digest": "g" * 11 + "-"},
                 id="a-non-alphanumeric-digest"),
])
def test_an_unencodable_record_is_refused_with_a_reason(bad):
    fields = {"seq": 1, "key_digest": b"\x01" * 32, "slot": 1,
              "block_bytes": PP0_BYTES, "geometry_digest": "a" * 12}
    fields.update(bad)
    out = dl.encode_index_record(**fields)
    assert out["ok"] is False and out["reason"]


def test_a_commit_frame_round_trips_and_names_its_watermark():
    made = dl.encode_index_commit(committed_seq=12, entries_in_commit=12)
    back = dl.decode_index_frame(made["frame"])
    assert back["ok"] and back["kind"] == "commit"
    assert back["committed_seq"] == 12 and back["entries_in_commit"] == 12
    for junk in ({"committed_seq": -1}, {"entries_in_commit": -3},
                 {"entries_in_commit": dl.MAX_RECORDS + 1}):
        fields = {"committed_seq": 1, "entries_in_commit": 1}
        fields.update(junk)
        assert dl.encode_index_commit(**fields)["ok"] is False


@pytest.mark.parametrize("tamper", [
    pytest.param("crc", id="a-corrupt-frame-is-excluded-not-half-believed"),
    pytest.param("magic", id="not-our-frame"),
    pytest.param("kind", id="an-unknown-frame-kind"),
])
def test_a_damaged_frame_classifies_itself(tmp_path, tamper):
    frames = [record(1), commit(1, 1)]
    blob = bytearray(b"".join(frames))
    if tamper == "crc":
        blob[-1] ^= 0xFF
    elif tamper == "magic":
        blob[0] = 0x5A
    else:
        blob[1] = 0x58
    path = write_index(tmp_path, [bytes(blob)])
    read = dl.read_index(path)
    assert read["state"] == dl._TRUNCATED
    assert read["records_visible"] == 0 and read["records"] == []
    assert read["failures"][-1]["failure"] in dl.FRAME_FAILURES


@pytest.mark.parametrize("slot", [0, 1, 4096, 2 ** 32 - 1, 2 ** 32,
                                 2 ** 40 - 1])
def test_every_legal_slot_fits_the_codec(slot):
    """The slot field must be as wide as the slot CAP the offset math allows.

    This is where the codec was wrong once: slot was validated against MAX_SLOT
    (2**40) but packed as uint32, so a legal-but-large slot raised `struct.error`
    out of a function whose whole contract is that it never raises.
    """
    geom = geometry()
    made = dl.encode_index_record(seq=1, key_digest=b"\x01" * 32, slot=slot,
                                  block_bytes=geom["block_bytes"],
                                  geometry_digest=geom["geometry_digest"])
    assert made["ok"], made["reason"]
    back = dl.decode_index_frame(made["frame"])
    assert back["ok"] and back["slot"] == slot


def test_the_codec_head_width_matches_the_struct_it_describes():
    assert struct.calcsize(dl.RECORD_PAYLOAD_STRUCT) == dl.RECORD_PAYLOAD_HEAD
    assert struct.calcsize(dl.FRAME_HEADER_STRUCT) == dl.FRAME_HEADER_BYTES
    assert struct.calcsize(dl.HEADER_STRUCT) == dl.HEADER_BYTES
    budget = dl.index_batch_bytes()
    assert budget["payload_bytes"] == (dl.RECORD_PAYLOAD_HEAD
                                       + dl.MAX_KEY_DIGEST_BYTES
                                       + dl.GEOMETRY_DIGEST_CHARS)
    assert budget["payload_bytes"] <= dl.MAX_FRAME_PAYLOAD_BYTES


def test_a_slot_past_the_cap_is_refused_not_packed():
    out = dl.encode_index_record(seq=1, key_digest=b"\x01" * 32,
                                 slot=dl.MAX_SLOT + 1,
                                 block_bytes=PP0_BYTES,
                                 geometry_digest="a" * 12)
    assert out["ok"] is False and "slot" in out["reason"]


def test_an_uncommitted_record_is_invisible_even_though_it_is_on_disk(tmp_path):
    geom = geometry()
    path = write_index(tmp_path, [record(1, geom), record(2, geom),
                                  record(3, geom)])
    read = dl.read_index(path)
    assert read["ok"] and read["state"] == "ok"
    assert read["records_parsed"] == 3
    assert read["records_visible"] == 0
    assert read["uncommitted_records"] == 3
    assert read["records"] == []
    assert read["committed_seq"] == 0 and read["highest_seq_seen"] == 3


def test_the_watermark_commits_by_sequence_not_by_position(tmp_path):
    """Committing seq 5 makes 1..5 visible even though the frames are ordered."""
    path = write_index(tmp_path, [record(n) for n in (1, 2, 3, 4, 5)]
                       + [commit(5, 5), record(6), record(7)])
    read = dl.read_index(path)
    assert [item["seq"] for item in read["records"]] == [1, 2, 3, 4, 5]
    assert read["uncommitted_records"] == 2
    assert read["highest_seq_seen"] == 7


def test_a_torn_tail_is_invisible_at_every_truncation_length(tmp_path):
    """The property, over EVERY length - not the one someone happened to test.

    The only commit sits at the very end, so no truncation before it may show a
    record, and the full blob must show all of them. A rule that passes one
    length and fails the next is not a rule.
    """
    frames = [record(n) for n in range(1, 21)] + [commit(20, 20)]
    blob = b"".join(frames)
    commit_start = len(b"".join(frames[:-1]))
    for length in range(1, len(blob)):
        path = write_index(tmp_path, [blob[:length]], name=f"torn-{length}.idx")
        read = dl.read_index(path)
        visible = [item["seq"] for item in read["records"]]
        assert len(set(visible)) == len(visible)   # never shown twice
        assert len(visible) <= read["records_parsed"]
        if length < len(blob):
            assert visible == [], (length, visible)
            assert read["records_visible"] == 0
        # A torn frame must be REPORTED as torn, never as a clean read of
        # something else, once we are inside or past a frame boundary.
        if commit_start < length < len(blob):
            assert read["state"] == dl._TRUNCATED
    full = dl.read_index(write_index(tmp_path, [blob], name="full.idx"))
    assert full["state"] == "ok"
    assert len(full["records"]) == 20
    assert full["committed_seq"] == 20


def test_a_torn_tail_after_a_commit_keeps_the_committed_records(tmp_path):
    frames = [record(n) for n in range(1, 6)] + [commit(5, 5)] \
        + [record(n) for n in range(6, 11)]
    blob = b"".join(frames)
    for length in range(1, len(blob)):
        path = write_index(tmp_path, [blob[:length]], name=f"mix-{length}.idx")
        read = dl.read_index(path)
        visible = [item["seq"] for item in read["records"]]
        # Past the commit, everything reached is visible; nothing un-reached.
        assert all(seq <= read["committed_seq"] for seq in visible)
        assert len(visible) <= 5
        assert read["uncommitted_records"] == read["records_parsed"] - len(visible)
    tail = len(blob) - 5            # kills the CRC of record 10
    read = dl.read_index(write_index(tmp_path, [blob[:tail]], name="tail.idx"))
    assert [item["seq"] for item in read["records"]] == [1, 2, 3, 4, 5]
    assert read["state"] == dl._TRUNCATED


def test_an_uncommitted_record_still_never_appears_after_a_reopen(tmp_path):
    frames = [record(1), commit(1, 1), record(2)]
    path = write_index(tmp_path, frames)
    first = dl.read_index(path)
    second = dl.read_index(path)
    assert [item["seq"] for item in first["records"]] == [1]
    assert first["records"] == second["records"]


def test_append_index_writes_then_reads_back(tmp_path):
    path = tmp_path / "live.idx"
    geom = geometry()
    batch = [record(1, geom), record(2, geom), commit(2, 2)]
    written = dl.append_index(path, batch)
    assert written["ok"] and written["bytes_written"] == sum(
        len(frame) for frame in batch)
    assert written["records"] == 2
    assert dl.read_index(path)["records_visible"] == 2
    # Appending more without a commit changes nothing a reader can see.
    dl.append_index(path, [record(3, geom)])
    assert dl.read_index(path)["records_visible"] == 2
    assert dl.append_index(path, []).get("bytes_written") == 0
    assert dl.append_index(tmp_path / "no" / "such" / "x.idx",
                           batch)["ok"] is False


def test_append_index_refuses_an_oversize_batch_rather_than_splitting_it(
        tmp_path):
    path = tmp_path / "big.idx"
    frames = [record(1)] * (dl.MAX_RECORDS + 2)
    out = dl.append_index(path, frames)
    assert out["ok"] is False and "cap" in out["reason"]
    assert not path.exists()


def test_a_record_written_twice_is_counted_not_double_shown(tmp_path):
    path = write_index(tmp_path, [record(4, slot=1), record(4, slot=2),
                                  record(5, key=b"\xff" * 32, slot=1),
                                  record(6, key=b"\xff" * 32, slot=1),
                                  commit(6, 4)])
    read = dl.read_index(path)
    assert read["records_visible"] == 4
    assert read["duplicate_seqs"] == 1        # seq 4 twice
    assert read["duplicate_keys"] == 2        # a key re-parked, once per reuse
    assert read["conflicting_slots"] == 1     # one slot, two keys
    assert read["ok"]                          # reported, never fatal


def test_an_index_with_no_records_is_a_clean_empty(tmp_path):
    read = dl.read_index(write_index(tmp_path, [], name="empty.idx"))
    assert read["ok"] and read["state"] == "ok"
    assert read["records"] == [] and read["records_parsed"] == 0
    assert read["index_bytes"] == 0
    missing = dl.read_index(tmp_path / "absent.idx")
    assert missing["ok"] and missing["state"] == dl._MISSING
    assert missing["index_bytes"] is None      # absent, never 0
    assert missing["exists"] is False


def test_a_missing_index_and_a_healthy_one_have_the_same_keys(tmp_path):
    """The degraded view is built through the same code as the healthy one."""
    healthy = dl.read_index(write_index(tmp_path, [record(1), commit(1, 1)]))
    missing = dl.read_index(tmp_path / "absent.idx")
    torn = dl.read_index(write_index(tmp_path, [b"\x00" * 40]))
    caps = dl.read_index(write_index(tmp_path, [record(1)]), max_records=None)
    for other in (missing, torn, caps):
        assert set(other) == set(healthy), sorted(set(healthy) ^ set(other))


def test_a_frame_claiming_a_huge_payload_is_refused_before_being_read(tmp_path):
    """The reader checks the length BEFORE reading it, so it cannot be made to
    materialise a 10^6-entry index - or a 4 GB frame - out of a small file."""
    bogus = struct.pack("!2sI", b"KD", 0xFFFFFFF0) + b"tail"
    path = write_index(tmp_path, [bogus])
    read = dl.read_index(path)
    assert read["state"] == dl._TRUNCATED
    assert read["failures"][-1]["failure"] == "oversize_payload"
    assert read["index_bytes"] == os.path.getsize(path)
    assert read["index_bytes"] < 32


def test_a_big_index_is_read_to_its_cap_and_says_truncated(tmp_path):
    count = 20_000
    frames = [record(seq) for seq in range(1, count + 1)] + [commit(count, count)]
    path = write_index(tmp_path, frames)
    read = dl.read_index(path, max_records=256)
    assert read["truncated"] is True
    # The WORK is bounded, not just the output: it parsed 256 records out of
    # 20,000 and stopped, and it never claimed to have read the file.
    assert read["records_parsed"] == 256
    assert read["frames"] == 256
    # The only commit sits at the end, so a capped read has seen no watermark:
    # everything it parsed is counted as uncommitted rather than shown as truth.
    assert read["records_visible"] == 0
    assert read["uncommitted_records"] == 256
    assert read["state"] == "ok"
    by_frames = dl.read_index(path, max_records=count + 1, max_frames=300)
    assert by_frames["truncated"] is False     # the frame bound stopped it
    assert by_frames["frames"] == 300
    assert by_frames["records_parsed"] == 300
    whole = dl.read_index(path, max_records=count + 1)
    assert whole["records_visible"] == count
    assert whole["records_parsed"] == count
    assert whole["truncated"] is False
    # A cap of 0 is a legitimate "read nothing", not a crash.
    assert dl.read_index(path, max_records=0)["records"] == []


@pytest.mark.parametrize("junk", ["64", None, 2.5, [4], -1, True])
def test_the_read_caps_are_refused_not_defaulted(tmp_path, junk):
    path = write_index(tmp_path, [record(1), commit(1, 1)])
    read = dl.read_index(path, max_records=junk)
    assert read["ok"] is False and read["state"] == dl._READ_FAILED
    assert read["records"] == []


def test_the_reader_survives_an_unreadable_index(tmp_path):
    path = write_index(tmp_path, [record(1)])
    os.chmod(path, 0o000)
    try:
        read = dl.read_index(path)
    except OSError:                            # pragma: no cover - as root only
        pytest.skip("this user cannot be denied a file they own")
    finally:
        os.chmod(path, 0o600)
    assert read["ok"] is False or read["state"] == dl._READ_FAILED
    assert read["reason"] and read["records"] == []


def test_the_writer_budget_is_derived_from_the_codec_not_from_a_guess():
    budget = dl.index_batch_bytes()
    assert budget["ok"]
    biggest = dl.encode_index_record(seq=1, key_digest=b"\x01" * 64, slot=1,
                                     block_bytes=PP0_BYTES,
                                     geometry_digest="a" * 12)
    assert biggest["entry_bytes"] <= budget["entry_bytes"]
    assert budget["units"]["entry_bytes"] == "bytes"
    assert dl.encode_index_record(seq=1, key_digest=b"\x01" * 65, slot=1,
                                  block_bytes=PP0_BYTES,
                                  geometry_digest="a" * 12)["ok"] is False


# --- units: every number declares what it measures ---------------------------

def _walk(payload, trail="$"):
    """Every dict carrying numbers must carry a `units` map describing them all.

    A number whose unit is implied is how a KiB becomes a MiB becomes an outage,
    so this walks the whole payload rather than spot-checking keys: a field added
    next month is covered by it the moment it is returned.
    """
    if isinstance(payload, dict):
        numbers = [key for key, value in payload.items()
                   if isinstance(value, (int, float))
                   and not isinstance(value, bool)]
        if numbers:
            assert "units" in payload, f"{trail} has {numbers} but no units"
            units = payload["units"]
            for key in numbers:
                assert key in units, f"{trail}.{key} has no declared unit"
                assert units[key] != "unknown", f"{trail}.{key} has no unit entry"
                assert units[key] in {"bytes", "slots", "records", "ranks",
                                      "groups", "frames", "files", "entries",
                                      "percent", "version", "chars",
                                      "operations"}
        for key, value in payload.items():
            if key == "units":
                continue
            _walk(value, f"{trail}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            _walk(item, f"{trail}[{index}]")


def test_every_payload_and_every_nested_payload_declares_its_units(tmp_path):
    geom = geometry()
    path = write_index(tmp_path, [record(1), record(2), commit(2, 2)])
    payloads = [
        dl.build_geometry(epoch=EPOCH, pp_rank=0, tp_rank=0, world_size=4,
                          block_bytes=PP0_BYTES, num_groups=6),
        dl.build_rank(pp_rank=1, tp_rank=2),
        dl.normalize_epoch("plain_epoch"),
        dl.build_filename(geom),
        dl.build_paths(tmp_path, geom),
        dl.parse_path(name_of(geom)),
        dl.classify_name(name_of(geometry(epoch="zzz11111")), geom),
        dl.classify_name("not-ours.bin", geom),
        dl.geometry_section(geom),
        dl.geometry_section(None),
        dl.offset_for_slot(geom, 3),
        dl.slot_for_offset(geom, 3 * PP0_BYTES),
        dl.bounds_for_file(geom, 7 * PP0_BYTES + 5),
        dl.slot_is_readable(geom, 7, 7 * PP0_BYTES + 5),
        dl.encode_index_record(seq=1, key_digest=b"\x01" * 32, slot=1,
                               block_bytes=PP0_BYTES,
                               geometry_digest=geom["geometry_digest"]),
        dl.encode_index_commit(committed_seq=1, entries_in_commit=1),
        dl.decode_index_frame(record(1)),
        dl.read_index(path),
        dl.read_index(tmp_path / "absent.idx"),
        dl.index_batch_bytes(),
        dl.append_index(tmp_path / "append.idx", [record(1), commit(1, 1)]),
        dl.boot_sweep(tmp_path, geom),
        dl.execute_unlink_plan(tmp_path, [], geometry=geom),
        dl.free_space(tmp_path),
        dl.space_check(tmp_path, watermark_bytes=1024),
        dl.space_check(tmp_path, watermark_percent=1.0),
        dl.space_check(tmp_path / "absent"),
        dl.encode_header(geom),
        dl.decode_header(dl.encode_header(geom)["header"]),
    ]
    for payload in payloads:
        _walk(payload)


# --- free space: absent, unreadable and genuinely full are three facts -------

def test_free_space_reports_real_bytes_on_a_real_filesystem(tmp_path):
    space = dl.free_space(tmp_path)
    assert space["ok"] and space["available"] is True
    assert space["free_bytes"] > 0 and space["total_bytes"] > 0
    assert space["units"]["free_bytes"] == "bytes"
    assert space["free_bytes"] <= space["total_bytes"]


@pytest.mark.parametrize("path", ["missing-dir", "missing-dir/deeper"])
def test_an_unreadable_filesystem_is_absent_never_zero(tmp_path, path):
    space = dl.free_space(tmp_path / path)
    assert space["available"] is False
    assert space["free_bytes"] is None          # NEVER 0: a lie that fills a disk
    assert space["total_bytes"] is None
    assert space["reason"]
    check = dl.space_check(tmp_path / path, watermark_bytes=1024)
    assert check["space_available"] is False
    assert check["below_watermark"] is None     # unknown is not "there is room"
    assert check["free_bytes"] is None


def test_the_watermark_accepts_bytes_or_a_percent_and_says_which(tmp_path):
    # Force the two verdicts with a real read plus a watermark either side of it.
    free = dl.free_space(tmp_path)["free_bytes"]
    low = dl.space_check(tmp_path, watermark_bytes=free // 2)
    assert low["watermark_source"] == "bytes"
    assert low["below_watermark"] is False
    high = dl.space_check(tmp_path, watermark_bytes=free * 2)
    assert high["below_watermark"] is True
    assert high["free_bytes"] == free
    percent = dl.space_check(tmp_path, watermark_percent=50.0)
    assert percent["watermark_source"] == "percent_of_total"
    assert percent["watermark_bytes"] == int(
        percent["total_bytes"] * 50.0 / 100.0)
    none = dl.space_check(tmp_path)
    assert none["below_watermark"] is None and "no watermark" in none["reason"]


@pytest.mark.parametrize("kwargs", [
    pytest.param({"watermark_bytes": -1}, id="a-negative-watermark"),
    pytest.param({"watermark_bytes": 1024.5}, id="a-float-watermark"),
    pytest.param({"watermark_percent": 150}, id="a-percent-over-total"),
    pytest.param({"watermark_percent": "10"}, id="a-string-percent"),
    pytest.param({"watermark_percent": True}, id="a-bool-percent"),
])
def test_a_bad_watermark_is_refused_rather_than_ignored(tmp_path, kwargs):
    out = dl.space_check(tmp_path, **kwargs)
    assert out["ok"] is False and out["below_watermark"] is None
    assert out["free_bytes"] is None and out["reason"]


def test_free_space_refuses_a_path_that_is_not_a_path():
    for junk in (None, 7, b"bytes", ["x"], "", "a\x00b"):
        out = dl.free_space(junk)
        assert out["available"] is False and out["free_bytes"] is None


# --- the header codec: forward compatibility, defined now, used later --------

def test_a_header_round_trips_and_carries_the_rank_it_claims():
    geom = geometry(pp=1, tp=3)
    made = dl.encode_header(geom)
    assert made["ok"] and made["header_bytes"] == dl.HEADER_BYTES + len(EPOCH)
    back = dl.decode_header(made["header"])
    assert back["ok"] and back["state"] == "ok"
    assert (back["pp_rank"], back["tp_rank"]) == (1, 3)
    assert back["epoch"] == EPOCH
    assert back["geometry_digest"] == geom["geometry_digest"]
    assert back["header_bytes"] == made["header_bytes"]


def test_a_header_from_another_format_version_is_refused_not_misread():
    """A reader that guessed at a future layout would read garbage as KV."""
    geom = geometry()
    made = bytearray(dl.encode_header(geom)["header"])
    made[8] = 9                        # the version byte
    out = dl.decode_header(bytes(made))
    assert out["ok"] is False and out["state"] == "version_mismatch"
    assert out["found_version"] == 9 and out["want_version"] == dl.FORMAT_VERSION


def test_the_header_codec_classifies_garbage_rather_than_raising():
    blobs = [b"", b"short", b"NOTOURS1" + b"\x00" * 40,
             dl.HEADER_MAGIC + b"\x01\x00\x00\x00\x00\x00\x00\x08" + b"ab",
             b"\x00" * 200, 17, None]
    for blob in blobs:
        out = dl.decode_header(blob)
        assert out["ok"] is False and out["reason"]
        assert out.get("found_version") is None or "version" in out["state"]


def test_a_header_for_an_unknown_geometry_is_refused():
    assert dl.encode_header(None)["ok"] is False
    assert dl.encode_header({"epoch": EPOCH})["ok"] is False


# --- the module is inert: nothing here touches the engine or the network -----

def test_no_function_here_opens_a_socket_or_touches_the_engine(tmp_path, monkeypatch):
    """Stage 1 is inert by construction: no fd outside a directory we were given.

    Proved by forbidding `socket.socket` for the duration and exercising the
    whole surface: if a later stage adds a socket, a vllm import or a CUDA call
    to this module, this test is where it fails.
    """
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("disk_layout reached for a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    geom = geometry()
    path = write_index(tmp_path, [record(1), commit(1, 1)])
    for call in (lambda: dl.build_paths(tmp_path, geom),
                 lambda: dl.boot_sweep(tmp_path, geom),
                 lambda: dl.read_index(path),
                 lambda: dl.free_space(tmp_path),
                 lambda: dl.space_check(tmp_path, watermark_bytes=1),
                 lambda: dl.offset_for_slot(geom, 1),
                 lambda: dl.encode_header(geom)):
        assert call() is not None


def test_the_module_never_creates_a_file_it_was_not_told_to(tmp_path):
    before = set(os.listdir(tmp_path))
    geom = geometry()
    dl.build_paths(tmp_path, geom)
    dl.build_filename(geom, dl.SUFFIX_INDEX)
    dl.boot_sweep(tmp_path, geom)
    dl.free_space(tmp_path)
    dl.index_batch_bytes()
    assert set(os.listdir(tmp_path)) == before
