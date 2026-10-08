r"""WHO WROTE EACH SIDE OF A DIFFERING PAIR - and what the bytes actually do.

0.2.47 and earlier logged EVERY differing pair at ERROR with the sentence "that is a
fault in the GPU-direct write path or a live file that changed after the spill". That
sentence claimed a reliability the comparator does not have: it pairs files by NAME,
and a name is an identity that outlives the completion that created it. Neither tree
ever overwrites an existing name (`disk_layout.write_committed_file` declines a
duplicate with `duplicate=True, write_bytes=0`, and the spill path writes the live file
through that same function at disk_spill.py:1934), so the two sides of a pair hold the
bytes of whichever EXECUTION OF THE WRITE PATH created that name on that side - and for
fp8 KV those bytes are not bit-reproducible.

WHAT THE LIVE BOX SHOWED (2026-10-09, read-only), and why the boot boundary is the
wrong reference. Eight same-named pairs differed. Every one of them had BOTH sides
written after the box's last boot (`uptime -s` 2026-10-08 23:20:12, both mtimes later),
so a reference that answers "did this process's engine start before or after that file
was written?" would have called all eight FAULTS:

    ef/kvslot-ef88716efb746ada-g5-...  shadow 00:44:20.875  live 00:05:27.231 (+2333 s)
    50/kvslot-50ed223d55f5384a-g1-...  shadow 00:43:59.074  live 00:43:58.942 (+ 132 ms)

The second line is the one that kills the boot model: a 132 ms gap is the SAME timing
signature as an identical pair (the identical pairs on that box sat at 0.25-1.02 s), and
both files are this-boot - yet the bytes disagree, because the live name had already
been created by an earlier spill of the same key. The reference that decides correctly
is therefore NOT "when did the engine start" but "when did THIS shadow's own store
complete": the live file's mtime is either older than that instant (so it cannot be the
file our spill wrote - the store never overwrites a name, so it belongs to an earlier
execution of the write path, in this boot or in a previous one) or it is not.

WHAT IS FAKE HERE. The stage is the writer's own: real files, real writer thread, real
comparison. Only two things are manufactured - the mtime of a live file (moved with
`os.utime`, exactly the state an earlier spill leaves behind) and one corrupted byte of
a live file (exactly the state "the live file changed after the spill" leaves behind).
"""

from __future__ import annotations

import logging
import os
import time

import pytest
import test_disk_gpu_shadow as shadow_stage
from test_disk_shadow import bind_and_complete, match_the_arena
from test_disk_store import LONG_KEY, metadata_shape

from kvtier_dev_plugin import (  # noqa: E402
    disk_gpu_shadow,
    disk_shadow,
    disk_spill,
)

# THE STAGE IS THE WRITER'S OWN, imported rather than duplicated - the house idiom
# (tests/test_disk_cpu_slot_reserve.py:73), because pytest finds a fixture by NAME and
# the name has to be in this module's namespace.
counters = shadow_stage.counters                    # noqa: F811 - a helper, not a fixture
drain_everything = shadow_stage.drain_everything    # noqa: F811 - both writer threads
shadow_files = shadow_stage.shadow_files            # noqa: F811 - {relative: bytes}
wait_for_pending = shadow_stage.wait_for_pending    # noqa: F811 - bounded, names what it found
tier = shadow_stage.tier                            # noqa: F811 - an armed store + env
verifier = shadow_stage.verifier                    # noqa: F811 - the verifier on
writer = shadow_stage.writer                        # noqa: F811 - the shadow WRITER on
stage = shadow_stage.stage                          # noqa: F811 - worker, arena, GPU, seam
worker_class = shadow_stage.worker_class            # noqa: F811 - the fork's worker class
patched_engine = shadow_stage.patched_engine        # noqa: F811 - autouse: store_keys exist
none_hash_seed = shadow_stage.none_hash_seed        # noqa: F811 - autouse: the seed module

NEW_COUNTERS = ("pairs_cross_execution", "pairs_fault", "pairs_unattributed")


# --- the stage ---------------------------------------------------------------
@pytest.fixture(autouse=True)
def shape():
    with metadata_shape(True):
        yield


def hold_the_pass(monkeypatch) -> dict:
    """Stop the writer thread's compare/retention pass, so the test decides the order.

    The `disk_gpu_shadow` idiom, used there for the same reason: with a live thread the
    pass and the test race, and a test that asserts a counter has to be able to MAKE the
    order that decides it.
    """
    real = disk_gpu_shadow._run_pass
    hold = {"yes": True}

    def gate():
        if hold["yes"]:
            return {"reason": "held by the test"}
        return real()

    monkeypatch.setattr(disk_gpu_shadow, "_run_pass", gate)
    return hold


def only_pair(root, shadow_root) -> tuple:
    """The single `(live_path, shadow_path, name)` this test just created."""
    files = shadow_files(shadow_root)
    assert len(files) == 1, files
    relative = next(iter(files))
    return (os.path.join(str(root), relative),
            os.path.join(str(shadow_root), relative),
            os.path.basename(relative))


def corrupt_one_byte(path, offset: int = 37, value: int = 0xA3) -> int:
    """Change ONE byte of one side and return the delta that resulted.

    The delta is measured, not assumed: the tests assert on `max_byte_delta`, so they
    must not care which way round the two values happened to fall.
    """
    with open(path, "r+b") as handle:
        handle.seek(offset)
        was = handle.read(1)[0]
        replacement = value if value != was else value - 1
        handle.seek(offset)
        handle.write(bytes([replacement]))
    return abs(int(replacement) - int(was))


def written_pair(directory, shadow_bytes: bytes, live_bytes: bytes,
                 store_wall_ns=None, live_mtime_ns=None,
                 shadow_mtime_ns=None, name="pair.bin") -> dict:
    """A pending-pair entry over two real files, with the attribution inputs pinned.

    `store_wall_ns=None` is not a convenience: it is the state a pair is in when the
    instant was never captured, and the module must call THAT unattributed instead of
    guessing a direction.
    """
    shadow = directory / "shadow" / name
    live = directory / "live" / name
    shadow.parent.mkdir(parents=True, exist_ok=True)
    live.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(shadow_bytes)
    live.write_bytes(live_bytes)
    if live_mtime_ns is not None:
        os.utime(live, ns=(live_mtime_ns, live_mtime_ns))
    if shadow_mtime_ns is not None:
        os.utime(shadow, ns=(shadow_mtime_ns, shadow_mtime_ns))
    return {"name": name, "shadow_path": str(shadow), "live_path": str(live),
            "width": max(len(shadow_bytes), len(live_bytes)),
            "since": time.monotonic(), "store_wall_ns": store_wall_ns}


# --- 1. THE BENIGN CASE, WHICH IS THE COMMON ONE ------------------------------
def test_a_pair_whose_live_half_predates_our_store_is_cross_execution_info_once(
        writer, stage, monkeypatch, caplog):
    """An older live half is EXPECTED, and must never reach the operator as ERROR.

    The live name already existed when this store spilled, so the store declined the
    duplicate and that file holds an earlier execution's fp8 bytes. That is the common
    case on a boot that re-spills hot prefixes, and 0.2.47 called it a corruption fault.
    The assertions are the three halves of the fix: the count lands in the
    cross-execution bucket, the two other buckets stay at zero, and the only log line is
    one INFO naming both mtimes and the reference it decided on.
    """
    hold = hold_the_pass(monkeypatch)
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert disk_spill.drain()["drained"] is True
    wait_for_pending(1)
    live, _shadow, name = only_pair(stage["root"], stage["shadow_root"])

    # THE STATE AN EARLIER SPILL LEAVES BEHIND: the name already had a file, so the
    # store kept those bytes and stamped them earlier. One byte differs, which is what
    # a recomputation of the same key looks like at the coarsest.
    delta = corrupt_one_byte(live)
    older = time.time() - 1000.0
    os.utime(live, (older, older))

    with caplog.at_level(logging.INFO, logger="disk_gpu_shadow"):
        hold["yes"] = False
        assert drain_everything()["shadow"]["drained"] is True

    seen = counters()
    assert seen["pairs_differing"] == 1, seen
    assert seen["pairs_cross_execution"] == 1, seen
    assert seen["pairs_fault"] == 0, seen
    assert seen["pairs_unattributed"] == 0, seen
    assert seen["pairs_identical"] == 0, seen

    differ_lines = [(record.levelno, record.getMessage())
                    for record in caplog.records if "differ" in record.getMessage()]
    assert differ_lines, [record.getMessage() for record in caplog.records]
    assert all(level != logging.ERROR for level, _ in differ_lines), differ_lines
    infos = [text for level, text in differ_lines if level == logging.INFO]
    assert len(infos) == 1, differ_lines
    line = infos[0]
    assert name in line
    assert "cross_execution" in line
    for needed in ("mtime", "reference"):
        assert needed in line.lower(), line

    last = disk_gpu_shadow.view()["last_difference"]
    assert last["classification"] == "cross_execution"
    assert last["live_predates_this_process_start"] is True
    assert last["first_difference"] == 37
    assert last["live_mtime_seconds"] < last["reference_seconds"]
    # The bytes were sampled while both files were still here, because retention
    # deletes this evidence within minutes and the question cannot be asked later.
    assert last["sample_available"] is True
    assert last["sample_offset"] == 37
    assert last["differing_bytes_in_sample"] == 1
    assert last["max_byte_delta"] == delta


def test_a_cross_execution_differ_never_occupies_the_name_an_operator_must_look_at(
        writer, stage, monkeypatch):
    """`differing_names` is the "go look at this file" list, so benign stays out.

    Six entries is the whole cap (`MAX_NAME_SAMPLES`) and a boot that re-spills hot
    prefixes produces benign differs far faster than six of them, so a list that
    recorded every differ would push the ONE name an operator needs off the section.
    """
    hold = hold_the_pass(monkeypatch)
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert disk_spill.drain()["drained"] is True
    wait_for_pending(1)
    live, _shadow, _name = only_pair(stage["root"], stage["shadow_root"])
    # ORDER MATTERS AND THE TEST SAYS SO: writing a file stamps it, so the corruption
    # goes first and the backdating second, or the mtime is "now" again and the pair
    # would be a same-execution differ - which is the OTHER test's subject.
    corrupt_one_byte(live)
    os.utime(live, (time.time() - 1000.0, time.time() - 1000.0))
    hold["yes"] = False
    assert drain_everything()["shadow"]["drained"] is True
    assert counters()["pairs_cross_execution"] == 1
    assert disk_gpu_shadow.section()["differing_names"] == []


# --- 2. THE FAULT, UNWEAKENED -------------------------------------------------
def test_a_same_execution_one_byte_difference_is_still_an_error_once(
        writer, stage, monkeypatch, caplog):
    """THE PROPERTY THE MODULE EXISTS FOR, unweakened.

    Both sides came from this store: the live file's mtime is not older than the store,
    so it IS the file our spill committed, and the shadow holds the bytes the verifier
    called `equal` for that same store. One byte out of 11.4 MB disagreeing under those
    conditions is a fault, and it must still be one ERROR and the fault counter - no
    tolerance, no "close enough", no shorter scan.
    """
    hold = hold_the_pass(monkeypatch)
    match_the_arena(stage, 5, 5)
    bind_and_complete(stage, slots=[5], gpu_blocks=[5], keys=[LONG_KEY], event=3)
    assert disk_spill.drain()["drained"] is True
    wait_for_pending(1)
    live, _shadow, name = only_pair(stage["root"], stage["shadow_root"])
    delta = corrupt_one_byte(live, offset=37, value=0xA3)

    with caplog.at_level(logging.INFO, logger="disk_gpu_shadow"):
        hold["yes"] = False
        assert drain_everything()["shadow"]["drained"] is True

    seen = counters()
    assert seen["pairs_differing"] == 1, seen
    assert seen["pairs_fault"] == 1, seen
    assert seen["pairs_cross_execution"] == 0, seen
    assert seen["pairs_unattributed"] == 0, seen

    errors = [record.getMessage() for record in caplog.records
              if record.levelno >= logging.ERROR]
    assert len(errors) == 1, errors
    assert name in errors[0]
    last = disk_gpu_shadow.view()["last_difference"]
    assert last["classification"] == "same_execution"
    assert last["first_difference"] == 37
    assert last["scanned_bytes"] >= 37
    assert last["live_predates_this_process_start"] is False
    assert last["differing_bytes_in_sample"] == 1
    assert last["max_byte_delta"] == delta
    # The name an operator must look at IS in the list, unlike the benign case.
    assert disk_gpu_shadow.section()["differing_names"] == [name]


def test_the_fault_branch_still_compares_every_byte_of_the_slot(tmp_path, caplog):
    """No tolerance, proved by a difference that lives at the very end.

    The attribution is computed AFTER the comparison, so structurally the fault branch
    cannot scan fewer bytes than any other - but "structurally" is not a test. A pair
    that agrees for 199,999 bytes and disagrees on the last one must still be decided,
    at the right offset, as the fault.
    """
    left = bytes([9]) * 200_000
    right = bytearray(left)
    right[-1] = 10
    entry = written_pair(tmp_path, left, bytes(right), store_wall_ns=time.time_ns())
    with caplog.at_level(logging.ERROR, logger="disk_gpu_shadow"):
        assert disk_gpu_shadow._compare_pair(entry) == "decided"
    last = disk_gpu_shadow.view()["last_difference"]
    assert last["first_difference"] == 199_999
    assert last["scanned_bytes"] >= 200_000
    assert last["classification"] == "same_execution"
    assert counters()["pairs_fault"] == 1
    # And the line does NOT hedge: 199,999 is inside the verifier's 256 KB window, so
    # the witness really did compare the byte this differ is about.
    errors = [record.getMessage() for record in caplog.records
              if record.levelno >= logging.ERROR]
    assert len(errors) == 1, errors
    assert "covers the first" not in errors[0], errors[0]


def test_a_fault_says_when_the_live_half_is_newer_than_its_own_shadow(tmp_path, caplog):
    """The one way a `same_execution` reference can be stale, named in the line.

    `store_wall_ns` is THIS store's, so a live file at or after it is this store's file
    - unless this store's spill never reached the disk. Then the pair sat pending, a
    LATER store wrote the live name (neither tree overwrites, so nothing else would
    ever notice) and the ERROR is our old shadow against a foreign completion. That is
    still worth the ERROR - a spill that silently failed is a fault - but the two cases
    look identical in the counters, and only the mtimes tell them apart.
    """
    now = time.time_ns()
    entry = written_pair(tmp_path, bytes([1]) * 64, bytes([2]) * 64,
                         store_wall_ns=now - 1_000_000_000,
                         shadow_mtime_ns=now - 500_000_000)   # live is the newest file
    with caplog.at_level(logging.ERROR, logger="disk_gpu_shadow"):
        assert disk_gpu_shadow._compare_pair(entry) == "decided"
    errors = [record.getMessage() for record in caplog.records
              if record.levelno >= logging.ERROR]
    assert len(errors) == 1, errors
    assert "NEWER than this shadow file" in errors[0], errors[0]
    assert "pairs_failed" in errors[0], errors[0]


def test_a_fault_whose_live_half_predates_the_shadow_carries_no_such_note(
        tmp_path, caplog):
    """The note above is a claim about mtime order, so it must be absent when that
    order does not hold - otherwise it is noise that trains a reader to ignore it.
    Here the live file is newer than the store instant (so still this store's) but
    older than our shadow file, which is the ordinary ordering.
    """
    now = time.time_ns()
    entry = written_pair(tmp_path, bytes([1]) * 64, bytes([2]) * 64,
                         store_wall_ns=now - 2_000_000_000,
                         live_mtime_ns=now - 1_000_000_000)    # shadow is the newest
    with caplog.at_level(logging.ERROR, logger="disk_gpu_shadow"):
        assert disk_gpu_shadow._compare_pair(entry) == "decided"
    assert counters()["pairs_fault"] == 1
    errors = [record.getMessage() for record in caplog.records
              if record.levelno >= logging.ERROR]
    assert len(errors) == 1, errors
    assert "NEWER than this shadow file" not in errors[0], errors[0]


def test_a_same_execution_differ_past_the_verifier_window_says_so(
        tmp_path, caplog):
    """The witness reads 256 KB of an 11.4 MB slot, and the line must not hide that.

    `disk_shadow.MAX_DIFF_SCAN_BYTES` bounds the verifier's scan, so an `equal` verdict
    covers the first 256 KB and nothing else. A same-execution differ beyond that is
    still a real disagreement between the GPU's bytes and the arena's - and still worth
    the ERROR - but implying that a witness had compared this byte would be a lie, so
    the line names the window instead.
    """
    window = disk_shadow.MAX_DIFF_SCAN_BYTES
    left = bytes([5]) * (window + 4096)
    right = bytearray(left)
    right[window + 100] = 6
    entry = written_pair(tmp_path, left, bytes(right), store_wall_ns=time.time_ns())
    with caplog.at_level(logging.ERROR, logger="disk_gpu_shadow"):
        assert disk_gpu_shadow._compare_pair(entry) == "decided"
    assert counters()["pairs_fault"] == 1
    errors = [record.getMessage() for record in caplog.records
              if record.levelno >= logging.ERROR]
    assert len(errors) == 1, errors
    assert f"covers the first {window} B" in errors[0], errors[0]
    assert disk_gpu_shadow.view()["last_difference"]["first_difference"] \
        == window + 100


# --- 3. UNKNOWN IS ITS OWN ANSWER ---------------------------------------------
def test_a_pair_with_no_store_instant_is_unattributed_at_warning_not_guessed(
        tmp_path, caplog):
    """No reference is not a licence to pick a side: it is `pairs_unattributed`.

    Folding "unknown" into the benign bucket would suppress faults; folding it into the
    fault bucket would re-create the false alarm this release exists to fix. So it is a
    bucket of its own, at WARNING, and the line must be there rather than quiet.
    """
    shadow_bytes = bytes([7]) * 512
    live_bytes = bytes([7]) * 37 + bytes([9]) + bytes([7]) * 474
    entry = written_pair(tmp_path, shadow_bytes, live_bytes, store_wall_ns=None)
    with caplog.at_level(logging.WARNING, logger="disk_gpu_shadow"):
        assert disk_gpu_shadow._compare_pair(entry) == "decided"
    seen = counters()
    assert seen["pairs_differing"] == 1, seen
    assert seen["pairs_unattributed"] == 1, seen
    assert seen["pairs_cross_execution"] == 0, seen
    assert seen["pairs_fault"] == 0, seen
    warnings = [record.getMessage() for record in caplog.records
                if record.levelno == logging.WARNING]
    assert any("unattribut" in text.lower() for text in warnings), warnings
    last = disk_gpu_shadow.view()["last_difference"]
    assert last["classification"] == "unattributed"
    assert last["classification_reason"]


def test_a_clock_that_moved_makes_the_pair_unattributable(tmp_path, caplog):
    """A wall clock that ran backwards between the store and the comparison.

    Every attribution here is an `mtime <= store instant` comparison, so a backwards
    step would silently relabel a same-execution fault as benign. The module notices
    that the clock it trusted now reads EARLIER than the instant it captured, and says
    it cannot tell rather than deciding on a broken ruler.
    """
    shadow_bytes = bytes([1]) * 256
    live_bytes = bytes([1]) * 8 + bytes([200]) + bytes([1]) * 247
    entry = written_pair(tmp_path, shadow_bytes, live_bytes,
                         store_wall_ns=time.time_ns() + 60 * 1_000_000_000)
    with caplog.at_level(logging.WARNING, logger="disk_gpu_shadow"):
        assert disk_gpu_shadow._compare_pair(entry) == "decided"
    seen = counters()
    assert seen["pairs_unattributed"] == 1, seen
    assert seen["pairs_cross_execution"] == 0 and seen["pairs_fault"] == 0, seen
    assert disk_gpu_shadow.view()["last_difference"]["classification"] \
        == "unattributed"


def test_an_unreadable_live_mtime_is_unattributed(tmp_path, monkeypatch, caplog):
    """A side that vanished between the comparison and the attribution is UNKNOWN.

    The comparison itself succeeded (both files were there a moment ago), so dropping
    the pair would lose a differ and guessing a direction would lie. The stat fails, so
    the attribution fails, and the pair is counted where a reader can see it.
    """
    shadow_bytes = bytes([3]) * 128
    live_bytes = bytes([3]) * 4 + bytes([4]) + bytes([3]) * 123
    entry = written_pair(tmp_path, shadow_bytes, live_bytes,
                         store_wall_ns=time.time_ns())
    monkeypatch.setattr(disk_gpu_shadow, "_file_mtime_ns", lambda path: None)
    with caplog.at_level(logging.WARNING, logger="disk_gpu_shadow"):
        assert disk_gpu_shadow._compare_pair(entry) == "decided"
    assert counters()["pairs_unattributed"] == 1
    assert counters()["pairs_cross_execution"] == 0
    assert counters()["pairs_fault"] == 0


# --- 4. THE SAMPLE ------------------------------------------------------------
def test_the_sample_counts_differing_bytes_and_the_largest_delta(tmp_path):
    """`differing_bytes_in_sample` and `max_byte_delta`, on a difference we made.

    The crafted window is 64 bytes with a KNOWN difference set: seven differing
    positions, the largest absolute delta 200, and that largest delta sits NINTH, not
    first - so a `max_byte_delta` that reported the first difference's delta, or the
    last one's, or a count, cannot satisfy it. The deltas run BOTH ways (offset 5 has
    the shadow side smaller, offset 9 larger), so a signed slip cannot pass either.
    """
    shadow = bytearray(b"-" * 64)
    live = bytearray(b"-" * 64)
    plan = {0: (200, 199), 1: (10, 12), 2: (5, 5), 5: (0, 1), 9: (200, 0),
            20: (7, 7), 33: (100, 40), 41: (3, 4), 63: (1, 2)}
    for offset, (left, right) in plan.items():
        shadow[offset] = left
        live[offset] = right
    entry = written_pair(tmp_path, bytes(shadow), bytes(live),
                         store_wall_ns=time.time_ns())
    sample = disk_gpu_shadow._sample_difference(entry["shadow_path"],
                                                entry["live_path"], 0,
                                                disk_gpu_shadow.DIFF_SAMPLE_BYTES)
    assert sample["sample_available"] is True
    assert sample["sample_bytes"] == 64
    assert sample["differing_bytes_in_sample"] == 7, sample
    assert sample["max_byte_delta"] == 200, sample

    # AND THE OTHER WAY ROUND: the same difference, sides swapped, same answer.
    swapped = disk_gpu_shadow._sample_difference(entry["live_path"],
                                                 entry["shadow_path"], 0,
                                                 disk_gpu_shadow.DIFF_SAMPLE_BYTES)
    assert swapped["differing_bytes_in_sample"] == 7, swapped
    assert swapped["max_byte_delta"] == 200, swapped


def test_the_sample_window_starts_at_the_first_difference(tmp_path):
    """A prefix from byte 0 would have been USELESS, and this proves it.

    The differing offsets seen on the live box were 0, 2 and 30720: a fixed prefix
    sample would have read 1 KiB of identical bytes for the third and reported "nothing
    differs in the sample" about a real differ. So the window is `DIFF_SAMPLE_BYTES`
    long and starts at the first difference.
    """
    shadow = bytearray(b"a" * 4096)
    live = bytearray(b"a" * 4096)
    live[3072] = 1
    entry = written_pair(tmp_path, bytes(shadow), bytes(live),
                         store_wall_ns=time.time_ns())
    sample = disk_gpu_shadow._sample_difference(entry["shadow_path"],
                                                entry["live_path"], 3072, 1024)
    assert sample["sample_available"] is True
    assert sample["sample_offset"] == 3072
    assert sample["differing_bytes_in_sample"] == 1
    assert sample["max_byte_delta"] == 96


def test_the_sample_read_never_reads_more_than_its_cap(tmp_path, monkeypatch):
    """Bounded work is a claim about BYTES READ, so count the bytes read.

    The seam (`_OPEN`) is the one the shipped code reads through and the wrapper is a
    real file object, so this measures the shipped reads rather than a re-implementation
    of them: two handles, each reading at most `DIFF_SAMPLE_BYTES`, whatever the size of
    the files.
    """
    shadow = tmp_path / "s.bin"
    live = tmp_path / "l.bin"
    shadow.write_bytes(bytes([1]) * 4_000_000)
    live.write_bytes(bytes([2]) * 4_000_000)
    reads: list = []

    class Counting:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size=-1):
            reads.append(size)
            return self._handle.read(size)

        def seek(self, *args):
            return self._handle.seek(*args)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    real_open = open

    def counting_open(path, mode="rb", **kwargs):
        return Counting(real_open(path, mode, **kwargs))

    monkeypatch.setattr(disk_gpu_shadow, "_OPEN", counting_open)
    sample = disk_gpu_shadow._sample_difference(str(shadow), str(live), 0,
                                                disk_gpu_shadow.DIFF_SAMPLE_BYTES)
    assert sample["sample_available"] is True
    assert reads, "the sample read no bytes at all"
    assert all(0 < size <= disk_gpu_shadow.DIFF_SAMPLE_BYTES for size in reads), reads
    assert len(reads) == 2, reads


def test_a_failed_sample_read_degrades_to_unavailable_and_never_to_identical(
        tmp_path, monkeypatch):
    """A sample that could not be read says so, and its numbers stay UNKNOWN.

    `differing_bytes_in_sample: 0` would be the most expensive kind of lie in this
    module: it reads as "the sample agrees", which is the sentence that closes an
    investigation. So a failure is `sample_available: false` with null numbers and one
    log line, and NO change at all to the verdict or the counters.
    """
    shadow_bytes = bytes([1]) * 64
    live_bytes = bytes([1]) * 10 + bytes([250]) + bytes([1]) * 53
    entry = written_pair(tmp_path, shadow_bytes, live_bytes,
                         store_wall_ns=time.time_ns())

    def refusing(*args, **kwargs):
        raise OSError("the device refused the read")

    monkeypatch.setattr(disk_gpu_shadow, "_OPEN", refusing)
    assert disk_gpu_shadow._compare_pair(entry) == "decided"
    last = disk_gpu_shadow.view()["last_difference"]
    assert last["sample_available"] is False
    assert last["differing_bytes_in_sample"] is None
    assert last["max_byte_delta"] is None
    assert last["sample_reason"]
    # The VERDICT is untouched by a failed sample: still a fault, still counted.
    assert last["classification"] == "same_execution"
    assert counters()["pairs_differing"] == 1
    assert counters()["pairs_fault"] == 1


# --- 5. THE PUBLISHED SHAPE ---------------------------------------------------
def test_the_difference_record_has_the_same_keys_in_every_state(tmp_path, monkeypatch):
    """benign, fault, unattributed and sample-unavailable all emit ONE key set.

    The section's stable-shape rule reaches nested records too, because the reader is
    an agent diffing two rank files: a key that appears only in the fault branch would
    read as "the format changed" in exactly the branch that matters.
    """
    refusing = {"now": False}

    def maybe_refuse(path, mode="rb", **kwargs):
        if refusing["now"]:
            raise OSError("the device refused the read")
        return open(path, mode, **kwargs)

    monkeypatch.setattr(disk_gpu_shadow, "_OPEN", maybe_refuse)

    def differ(label, store_wall_ns, live_older_by_ns=0):
        entry = written_pair(tmp_path / label, b"bytes one", b"bytes two",
                             store_wall_ns=store_wall_ns,
                             live_mtime_ns=(time.time_ns() - live_older_by_ns
                                            if live_older_by_ns else None))
        disk_gpu_shadow._compare_pair(entry)
        return dict(disk_gpu_shadow.view()["last_difference"])

    benign = differ("benign", time.time_ns(), live_older_by_ns=5 * 10 ** 9)
    fault = differ("fault", time.time_ns())
    unknown = differ("unknown", None)
    refusing["now"] = True
    nosample = differ("nosample", time.time_ns())

    keys = sorted(benign)
    assert keys
    for label, record in (("fault", fault), ("unattributed", unknown),
                          ("sample_unavailable", nosample)):
        assert sorted(record) == keys, (label, sorted(record), keys)
    assert benign["classification"] == "cross_execution"
    assert fault["classification"] == "same_execution"
    assert unknown["classification"] == "unattributed"
    assert nosample["classification"] == "same_execution"
    assert nosample["sample_available"] is False
    section = disk_gpu_shadow.section()
    for key in NEW_COUNTERS:
        assert isinstance(section[key], int), key
    assert "differ_reference" in section


def test_the_three_differ_buckets_partition_the_differ_total(tmp_path):
    """`pairs_differing == cross_execution + fault + unattributed`, always.

    A reader who sees `pairs_differing: 9` must be able to account for all nine. The
    umbrella keeps its old meaning (every byte-differing pair) so an existing dashboard
    keeps working; the three new counters say WHICH of the nine deserve attention.
    """
    for index, (store_wall_ns, older_by_ns) in enumerate(
            ((time.time_ns(), 5 * 10 ** 9), (time.time_ns(), 0), (None, 0))):
        right = bytearray(b"x" * 64)
        right[10] = 9
        entry = written_pair(tmp_path / f"p{index}", b"x" * 64, bytes(right),
                             store_wall_ns=store_wall_ns,
                             live_mtime_ns=(time.time_ns() - older_by_ns
                                            if older_by_ns else None))
        disk_gpu_shadow._compare_pair(entry)
    seen = counters()
    assert seen["pairs_differing"] == 3, seen
    assert seen["pairs_cross_execution"] == 1, seen
    assert seen["pairs_fault"] == 1, seen
    assert seen["pairs_unattributed"] == 1, seen
    assert (seen["pairs_differing"] == seen["pairs_cross_execution"]
            + seen["pairs_fault"] + seen["pairs_unattributed"])


def test_the_new_counters_are_declared_published_and_start_at_zero():
    """Declared in `_COUNTER_KEYS`, published by `section()`, zeroed by the reset.

    A counter missing from the tuple is invisible to the section and to the
    contribution seam's shape, so the declaration IS the contract and this pins it.
    """
    for name in NEW_COUNTERS:
        assert name in disk_gpu_shadow._COUNTER_KEYS, name
    seen = disk_gpu_shadow.counters()
    for name in NEW_COUNTERS:
        assert seen[name] == 0, name
    section = disk_gpu_shadow.section()
    for name in NEW_COUNTERS:
        assert section[name] == 0, name


def test_the_reference_is_this_process_and_not_an_environment_value(monkeypatch):
    """The boot reference is THIS process's start, never a value from the launcher.

    `VLLM_E2_TIER_DIAG_EPOCH` is deliberately shared by every process of a boot - and
    when it is unset it is a lazily generated random token with no time in it at all
    (config.py:923-942, snapshot.py:81-88). This module reports a differ in a WORKER, so
    the only start time worth having is the one belonging to the process that observed
    the pair: the same lesson 0.2.47 learned about the snapshot's writer block.
    """
    monkeypatch.setenv("VLLM_E2_TIER_DIAG_EPOCH", "1700000000")
    reference = disk_gpu_shadow._resolve_process_start()
    assert reference["pid"] == os.getpid()
    assert "epoch" not in reference["source"].lower()
    assert reference["process_start_seconds"] is not None
    assert reference["process_start_seconds"] <= time.time()
    assert reference["process_start_seconds"] > time.time() - 86_400
    section = disk_gpu_shadow.section()
    assert set(section["differ_reference"]) == set(reference)
    assert section["differ_reference"]["pid"] == os.getpid()


def test_the_section_shape_survives_a_recorded_differ(writer, stage, tmp_path):
    """A differ recorded in the section changes no KEY, only values.

    The trim path documents that `last_difference` is never a trim target; this proves
    the other half - a section carrying a differ record is the SAME SHAPE as one
    carrying nothing, which is what lets a reader diff two rank files at all.
    """
    quiet = disk_gpu_shadow.section()
    entry = written_pair(tmp_path, bytes([1]) * 64, bytes([2]) * 64,
                         store_wall_ns=time.time_ns())
    disk_gpu_shadow._compare_pair(entry)
    busy = disk_gpu_shadow.section()
    assert set(busy) == set(quiet)
    assert set(busy["differ_reference"]) == set(quiet["differ_reference"])
    assert busy["pairs_fault"] == 1
