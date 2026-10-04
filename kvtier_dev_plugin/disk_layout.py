"""Stage 1 of an SSD tier under the host-RAM KV tier: layout, index, codec.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
The live tier parks evicted GPU KV into pinned host-RAM slots. This is the
*layout* layer for the SSD tier beneath it: file naming, slot -> offset
arithmetic, the append-only index codec, the boot sweep that makes a crashed
boot cheap, and a free-space gate. It is PURE. No fd is opened for serving, no
thread, no vllm, no torch; importing it and calling everything in it writes no
byte to disk except inside a directory the caller hands in. Nothing imports it
yet - stages 2-3 wire it into the worker - so this stage is inert by
construction, which is the only reason it is allowed to define a format.

WHY WE WRITE OUR OWN DISK TIER INSTEAD OF USING THE FORK'S
----------------------------------------------------------
The fork's disk backend (`vllm/v1/simple_kv_offload/disk/worker.py`) is unusable
for a tier we must be able to switch OFF:

  * it picks disk-vs-none AT BOOT (worker.py:176-205) with no runtime switch, so
    "turn the SSD tier off" costs an engine restart;
  * its boot path is unguarded, so a bad ``disk_path`` means the engine comes up
    and serves NOTHING;
  * its IO thread can die and the backend then stalls silently forever - there is
    no timeout anywhere on the wait;
  * it never populates ``failed_recving``, so the engine's recompute fallback is
    unreachable and a failed load becomes a wrong answer instead of a miss;
  * it has zero counters, and its ``shutdown()`` is never called, so a ~100 GB
    file leaks on every boot;
  * its ``disk_capacity_bytes`` is not divided by ``world_size`` (4 ranks here),
    so the cap it enforces is 4x the intended one.

Every one of those is a policy failure, not a physics one. So the primitives here
are shaped to make them impossible in the layers above: content-addressed keys
(switching the tier ON rehashes nothing, because a key is a function of content
and not of state), one file per rank at a fixed slot stride (large contiguous
``pwrite``s, and a crash sweep of O(1) unlinks instead of O(10^5)), an index whose
visibility is a COMMITTED WATERMARK rather than "the last line looks fine", and
names that identify their own boot and geometry so a stale file is identifiable
and removable instead of being a landmine for the next boot.

THE UNIT OF SPILL IS ONE RAM SLOT
---------------------------------
A slot's bytes are addressed exactly as ``blockdump.py`` addresses them -
``data_ptr() + slot * stride(0) * element_size`` per region, ctypes, no CUDA, no
refcount and no LRU perturbation (that contract is proven in ``blockdump``'s
docstring; this module does the *disk* half of the same arithmetic). One record is
therefore one slot's bytes across all of that rank's region names at that slot
index, contiguously, and ``block_bytes`` means:

    block_bytes = bytes for ONE slot ON ONE RANK
                = sum over that rank's region tensors of stride(0)*element_size()

NOT per GPU block, NOT per token, NOT the whole 4-rank world. On the live box a
slot is ~11.2 MB on the PP0 ranks and ~12.6 MB on the PP1 ranks, which is why it
is per-rank and per-geometry and NOT a constant: PP0 and PP1 coexist, and if one
number is shared the cheaper rank evicts into the dearer one's space.
``build_geometry`` refuses a ``block_bytes`` outside [1 KiB, 1 GiB] precisely
because unit confusion in this field corrupts live KV rather than merely being
wrong.

LAYOUT, AND WHY IT IS ONE FILE
------------------------------
``kvtier-<epoch>-PP<n>_TP<m>-g<geomdigest>.bin`` holds slot ``s`` at
``header_bytes + s * block_bytes``; the append-only index is the sibling ``.idx``.
One file per rank avoids 10^5-file inode churn and turns the crash sweep into two
unlinks. Buffered IO + ``posix_fadvise(DONTNEED)``, NOT ``O_DIRECT``: a
multi-region slot span carries no 4096-alignment guarantee, and unlike the
engine's backend we are not trying to escape pinned-RAM pressure (that belongs to
stage 2/3, which owns the fd; this module owns the math).

CROSS-BOOT REUSE IS DECLINED, AND THE FORMAT KEEPS THE DOOR OPEN
----------------------------------------------------------------
A boot sweeps and unlinks files matching OUR exact prefix/suffix whose epoch or
geometry differ from its own; it reuses nothing. Reusing another boot's bytes
would need header magic plus a per-entry content hash, and the benefit is
unmeasurable without generating load on a box that serves traffic, while the
poisoning risk is asymmetric (a wrong byte served as KV is a wrong token, not a
miss). The forward-compat requirement is therefore only "no format break": the
data file's offset math already carries ``header_bytes``, and
``encode_header``/``decode_header`` below are a complete, versioned,
version-refusing header codec that stage 4 can turn on behind the same toggle.

INDEX VISIBILITY IS A COMMITTED WATERMARK, NOT A TAIL HEURISTIC
---------------------------------------------------------------
The index is a stream of CRC'd frames: DATA frames (one per record) and COMMIT
frames carrying a committed sequence watermark. A record is visible iff
``seq <= the highest committed watermark seen before the scan stopped``, and the
scan stops at the first torn or bad-CRC frame because a single append-only writer
can only have torn the TAIL. Consequences, all deliberate:

  * a crash mid-DATA leaves the record invisible (a recompute, never a wrong
    read) even though its bytes may be fully on disk;
  * a crash mid-COMMIT advances nothing;
  * a record whose commit landed but whose slot bytes did NOT cannot happen
    within one boot. Cross-boot reuse - the case that WOULD need an fsync of the
    data file before the COMMIT frame is appended, since page-cache writeback
    order is not guaranteed - is the one thing deliberately not enabled. Do not
    enable it without re-reading this paragraph.

BOUNDED, ALWAYS
---------------
Every list has a cap, every read has a length cap checked BEFORE the read (the
index is streamed frame-by-frame, never slurped), the sweep is one non-recursive
``scandir`` with a file cap, and the unlink plan is capped. A 10^6-entry index is
read up to ``MAX_RECORDS`` and reported as truncated - materialising it is
refused, not attempted.

NEVER A ZERO FOR AN ABSENT MEASUREMENT
--------------------------------------
``statvfs`` failing, a missing directory, an unreadable directory and a genuinely
full disk are four different facts and are reported as four different things:
``available`` plus a ``reason`` and ``None`` counts. A ``0`` here would be read as
"the drive is full" or "there are no records", both of which are lies that change
what the tier does. Bad input is classified with a ``reason`` and EXCLUDED from
totals, never raised, and the degraded view is built through the same code path as
the healthy one so the key sets cannot drift.

UNIT DISCIPLINE
---------------
We have been burned by unit confusion in this project, so every numeric field
name ends in its unit (``_bytes``, ``_slots``, ``_records``, ``_ranks``,
``_groups``) and every dict carrying numbers carries a ``units`` map generated
from ONE table (``UNITS_BY_NAME``). An unrecognised name reports ``"unknown"``
rather than being guessed at, and a test walks every payload to prove no numeric
field is left undescribed.

Stdlib only: ``hashlib``, ``os``, ``struct``, ``zlib``. Every entry point is total
- it returns a classified failure instead of raising.
"""

from __future__ import annotations

import hashlib
import os
import struct
import zlib

# --- format identity ---------------------------------------------------------

FORMAT_VERSION = 1                 # the on-disk data/index format written
LAYOUT_SCHEMA_VERSION = 1          # this module's own payload contract

FILE_PREFIX = "kvtier-"
SUFFIX_DATA = ".bin"
SUFFIX_INDEX = ".idx"
KNOWN_SUFFIXES = (SUFFIX_DATA, SUFFIX_INDEX)

FILENAME_TEMPLATE = "kvtier-<epoch>-PP<n>_TP<m>-g<geomdigest><suffix>"

MAX_EPOCH_CHARS = 32
# '-' is a FIELD SEPARATOR in the file name, so it can never appear inside a
# field; '.' is the suffix boundary. Everything else filesystem-safe is allowed
# and anything else is refused rather than guessed at.
EPOCH_ALPHABET = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

MIN_PP_RANK = 0
MAX_PP_RANK = 4096
MIN_TP_RANK = 0
MAX_TP_RANK = 4096
MIN_WORLD_SIZE = 1
MAX_WORLD_SIZE = 4096
MIN_NUM_GROUPS = 1
MAX_NUM_GROUPS = 1024

# A slot is ~11.2 MB (PP0) / ~12.6 MB (PP1) on the live box. These bounds are not
# arbitrary: 1 KiB means "someone handed us a token count or a KiB-denominated
# number" and 1 GiB means "someone multiplied by the pool size".
MIN_BLOCK_BYTES = 1024
MAX_BLOCK_BYTES = 1024 * 1024 * 1024
MAX_SLOT = 1 << 40                 # absurd on purpose: a slot INDEX, not a size
MAX_OFFSET = 1 << 62               # the practical uint64 span of one file
MAX_HEADER_BYTES = 64 * 1024 * 1024

GEOMETRY_DIGEST_BYTES = 6          # -> 12 hex characters in the file name
GEOMETRY_DIGEST_CHARS = GEOMETRY_DIGEST_BYTES * 2
MAX_REGIONS_DIGEST_CHARS = 64
REGIONS_DIGEST_INPUT_LIMIT = 4096  # region names hashed, never stored

# --- the index codec ---------------------------------------------------------

INDEX_FRAME_MAGIC = 0x4B                       # 'K'
INDEX_FRAME_DATA = 0x44                        # 'D'
INDEX_FRAME_COMMIT = 0x43                      # 'C'
INDEX_FRAME_KINDS = (INDEX_FRAME_DATA, INDEX_FRAME_COMMIT)

FRAME_HEADER_STRUCT = "!2sI"                   # (magic, kind) then payload length
FRAME_HEADER_BYTES = struct.calcsize(FRAME_HEADER_STRUCT)     # 6
FRAME_TRAILER_BYTES = 4                        # crc32 of header + payload
MAX_FRAME_PAYLOAD_BYTES = 256

# A slot index is 64-bit on purpose: MAX_SLOT is 2**40 and the offset cap is
# 2**62 bytes, so a uint32 slot field would overflow the pack at a slot the
# offset math is legal for, and a struct.error escaping a codec is a contract
# violation in a module whose every entry point is total.
RECORD_PAYLOAD_STRUCT = "!QQIH"                 # seq, slot, block_bytes, key_len
RECORD_PAYLOAD_HEAD = struct.calcsize(RECORD_PAYLOAD_STRUCT)  # 22
COMMIT_PAYLOAD_STRUCT = "!QI"                   # seq, entries_in_commit
COMMIT_PAYLOAD_BYTES = struct.calcsize(COMMIT_PAYLOAD_STRUCT)  # 12

MAX_KEY_DIGEST_BYTES = 64
MAX_SEQ = (1 << 63) - 1

# The read caps. A frame is ~110 B, so MAX_RECORDS bounds the PARSED output well
# under 30 MiB no matter how large the file on disk is.
DEFAULT_MAX_RECORDS = 65_536
MAX_RECORDS = 262_144
DEFAULT_MAX_FRAMES = 1_000_000
MAX_FRAMES = 10_000_000

# --- the sweep and the space gate -------------------------------------------

MAX_SWEEP_FILES = 4096
MAX_UNLINK_PLAN = 256
MAX_PATH_CHARS = 4096
MAX_STAT_ERROR_CHARS = 160
MIN_WATERMARK_PERCENT = 0.0
MAX_WATERMARK_PERCENT = 99.0

# Header codec. Defined now, UNUSED by stage 1: see the docstring's
# forward-compat paragraph. Its whole purpose is that turning persistence on later
# is a parameter (`header_bytes`) and not a format break.
HEADER_MAGIC = b"KVTDISK1"
HEADER_STRUCT = "!8sBIIH12s"      # magic, version, pp, tp, epoch chars, geom
HEADER_BYTES = struct.calcsize(HEADER_STRUCT)  # 31
HEADER_EPOCH_CHARS = 24

# Classifications of a name, and of a file in the sweep. Closed vocabularies: a
# reader must be able to enumerate them from the payload.
NAME_CURRENT = "current"
NAME_STALE_EPOCH = "stale_epoch"
NAME_STALE_GEOMETRY = "stale_geometry"
NAME_MALFORMED = "malformed"
NAME_NOTOURS = "not_ours"
NAME_REFUSED = "refused"
NAME_CLASSIFICATIONS = (NAME_CURRENT, NAME_STALE_EPOCH, NAME_STALE_GEOMETRY,
                        NAME_MALFORMED, NAME_NOTOURS, NAME_REFUSED)

# A malformed name is REPORTED and never unlinked: our prefix plus our suffix is
# not proof the file is OURS, and the cost of deleting someone's
# `kvtier-backup-notes.bin` is infinite next to the cost of leaving it.
UNLINKABLE_CLASSIFICATIONS = (NAME_STALE_EPOCH, NAME_STALE_GEOMETRY)

# Directory and stream states. A missing directory and an unreadable one are
# different facts and are never collapsed into "no files".
_OK_DIR = "ok"
_MISSING = "missing"
_NOT_A_DIRECTORY = "not_a_directory"
_UNREADABLE = "unreadable"
_READ_FAILED = "read_failed"
_TRUNCATED = "torn_tail"
_DIR_STATES = (_OK_DIR, _MISSING, _NOT_A_DIRECTORY, _UNREADABLE, _READ_FAILED,
               NAME_REFUSED)

# The frame reader's failure vocabulary, so a reader can enumerate what can go
# wrong without reading the codec.
FRAME_FAILURES = ("truncated", "bad_magic", "bad_kind", "bad_crc", "bad_payload",
                  "oversize_payload")

# One table, so no two payloads can describe the same field two different ways.
UNITS_BY_NAME = {
    "block_bytes": "bytes",
    "bytes": "bytes",
    "byte_offset": "bytes",
    "capacity_bytes": "bytes",
    "commit_bytes": "bytes",
    "consumed_bytes": "bytes",
    "end_offset_bytes": "bytes",
    "entry_bytes": "bytes",
    "file_bytes": "bytes",
    "file_size_bytes": "bytes",
    "fragment_bytes": "bytes",
    "free_bytes": "bytes",
    "header_bytes": "bytes",
    "index_bytes": "bytes",
    "max_key_digest_bytes": "bytes",
    "offset_bytes": "bytes",
    "payload_bytes": "bytes",
    "bytes_written": "bytes",
    "planned_bytes": "bytes",
    "read_bytes": "bytes",
    "reserved_for_root_bytes": "bytes",
    "size_bytes": "bytes",
    "span_bytes": "bytes",
    "stopped_at_bytes": "bytes",
    "total_bytes": "bytes",
    "unit_size_bytes": "bytes",
    "used_bytes": "bytes",
    "watermark_bytes": "bytes",
    "capacity_slots": "slots",
    "conflicting_slots": "slots",
    "first_slot": "slots",
    "last_complete_slot": "slots",
    "last_slot": "slots",
    "num_slots": "slots",
    "one_past_last_slot": "slots",
    "partial_tail_bytes": "bytes",
    "partial_tail_slots": "slots",
    "slot": "slots",
    "slot_index": "slots",
    "written_slots": "slots",
    "committed_seq": "records",
    "commit_seq": "records",
    "duplicate_keys": "records",
    "duplicate_seqs": "records",
    "entries_in_commit": "records",
    "highest_seq_seen": "records",
    "next_seq": "records",
    "records": "records",
    "records_parsed": "records",
    "records_visible": "records",
    "seq": "records",
    "uncommitted_records": "records",
    "watermark_seq": "records",
    "commit_count_mismatches": "frames",
    "frames": "frames",
    "out_of_order_commits": "frames",
    "pp_rank": "ranks",
    "tp_rank": "ranks",
    "world_size": "ranks",
    "num_groups": "groups",
    "free_percent_of_total": "percent",
    "watermark_percent": "percent",
    "format_version": "version",
    "version": "version",
    "want_version": "version",
    "found_version": "version",
    "name_chars": "chars",
    "epoch_chars": "chars",
    "current": "files",
    "stale_epoch": "files",
    "stale_geometry": "files",
    "malformed": "files",
    "not_ours": "files",
    "refused": "files",
    "planned": "files",
    "unlinked": "files",
    "files": "files",
    "scanned": "entries",
    "errors": "operations",
    "files_free": "files",
    "watermark_bytes_configured": "bytes",
}


def _fail(reason: str, **extra) -> dict:
    """The one failure shape: ``ok: False``, a reason, and NO zeros for numbers."""
    out = {"ok": False, "reason": reason}
    out.update(extra)
    return out


def _ok(units=None, **fields) -> dict:
    """The one success shape. `units` comes from `_units()`, never from a literal."""
    out = {"ok": True, "reason": None}
    out.update(fields)
    out["units"] = dict(units or {})
    return out


def _cap(value, name: str, *, low: int, high: int) -> dict:
    """Clamp a caller-supplied bound, refusing anything that is not an int.

    A float would silently floor and a numeric string would silently coerce: both
    lose a bound without saying so, which is the failure this module is not
    allowed to have.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return _fail(f"{name} must be an int, got {type(value).__name__}")
    if value < low:
        return _fail(f"{name} {value} is below the floor {low}")
    return {"ok": True, "reason": None, "value": min(value, high)}


def _units(*names) -> dict:
    """Build a `units` map from field NAMES, from the one table.

    ``_units("slot", "block_bytes")`` -> ``{"slot": "slots", "block_bytes":
    "bytes"}``. An unrecognised name becomes ``"unknown"``: a bug in this module
    that a test can see, rather than an absent key or a guessed unit.
    """
    return {name: UNITS_BY_NAME.get(name, "unknown") for name in names}


# --- rank geometry -----------------------------------------------------------

def compute_geometry_digest(*, block_bytes: int, world_size: int,
                            num_groups: int | None = None,
                            regions_digest: str | None = None) -> str:
    """A short digest of everything that changes a slot's byte LAYOUT.

    Deliberately EXCLUDES the epoch: ``stale_geometry`` and ``stale_epoch`` are
    different facts in the sweep, and folding the epoch in here would make every
    new boot look like a new geometry. Deliberately INCLUDES ``world_size``,
    because a changing world size is exactly where the fork's
    not-divided-by-world_size capacity bug shows up.
    """
    material = "|".join((
        "k1",
        f"block_bytes={int(block_bytes)}",
        f"world_size={int(world_size)}",
        f"num_groups={'' if num_groups is None else int(num_groups)}",
        f"regions={regions_digest or ''}",
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[
        :GEOMETRY_DIGEST_CHARS]


def compute_regions_digest(regions) -> str:
    """A digest of the region NAMES that make up one slot span.

    Two boots whose region sets differ must never accept each other's files: a
    slot span is those regions concatenated in a fixed order, so a different set
    means a different ``block_bytes`` AND a different meaning for every offset
    inside the slot. Sorted, so the digest is order-independent; the ORDER is
    stage 2's contract and lives in the geometry only as far as its content does.
    """
    names = [str(name) for name in (regions or ())][:REGIONS_DIGEST_INPUT_LIMIT]
    if not names:
        return ""
    material = "\n".join(sorted(names))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[
        :GEOMETRY_DIGEST_CHARS]


def normalize_epoch(raw) -> dict:
    """Say whether an operator-supplied epoch is usable in a file name.

    The engine's own per-process epoch is ``secrets.token_hex(4)`` and needs
    nothing done to it. An operator-set one may not: ``-`` is our field separator
    and ``.`` our suffix boundary. Characters outside the alphabet CAN be mapped
    to ``_``, which is lossy - two epochs can collide - so the caller gets
    ``changed`` and ``note`` and decides. ``build_geometry`` refuses instead of
    calling this for you: a silently renamed epoch could make a stale boot's file
    look like the current one, and that is the one error this sweep must never
    make.
    """
    if not isinstance(raw, str):
        return _fail(f"epoch must be a string, got {type(raw).__name__}")
    text = raw.strip()
    if not text:
        return _fail("epoch is empty")
    if len(text) > MAX_EPOCH_CHARS:
        return _fail(f"epoch is {len(text)} chars; the cap is {MAX_EPOCH_CHARS}")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return _fail("epoch contains a control character")
    if "/" in text or "\\" in text:
        return _fail("epoch contains a path separator")
    safe = "".join(char if char in EPOCH_ALPHABET else "_" for char in text)
    return _ok(units={}, epoch=safe, changed=safe != text,
               note=("characters outside [A-Za-z0-9_] became '_', which can "
                     "collide with another epoch" if safe != text else None))


def build_rank(*, pp_rank, tp_rank) -> dict:
    """Validate and render the rank label ``PP<n>_TP<m>`` (one name field)."""
    for name, value in (("pp_rank", pp_rank), ("tp_rank", tp_rank)):
        if isinstance(value, bool) or not isinstance(value, int):
            return _fail(f"{name} must be an int, got {type(value).__name__}")
    if not MIN_PP_RANK <= pp_rank <= MAX_PP_RANK:
        return _fail(f"pp_rank {pp_rank} is outside [{MIN_PP_RANK}, "
                     f"{MAX_PP_RANK}]")
    if not MIN_TP_RANK <= tp_rank <= MAX_TP_RANK:
        return _fail(f"tp_rank {tp_rank} is outside [{MIN_TP_RANK}, "
                     f"{MAX_TP_RANK}]")
    return _ok(units=_units("pp_rank", "tp_rank"),
               label=f"PP{pp_rank}_TP{tp_rank}", pp_rank=pp_rank,
               tp_rank=tp_rank)


def build_geometry(*, epoch, pp_rank, tp_rank, world_size, block_bytes,
                   num_groups=None, regions_digest=None,
                   geometry_digest=None) -> dict:
    """A rank's geometry: the ONLY thing a file name may depend on.

    Refuses, with a reason naming the field, anything that could not have come
    from a real boot. ``block_bytes`` is per-rank bytes for ONE slot (module
    docstring) and is range-checked, because BOTH directions of the usual mistake
    (a token count, or a pool-wide total) are silently catastrophic offset math
    later. ``geometry_digest`` may be supplied by a caller that already computed
    one; otherwise it is derived from the layout fields.
    """
    rank = build_rank(pp_rank=pp_rank, tp_rank=tp_rank)
    if not rank["ok"]:
        return _fail(str(rank["reason"]))
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        return _fail(f"world_size must be an int, got "
                     f"{type(world_size).__name__}")
    if not MIN_WORLD_SIZE <= world_size <= MAX_WORLD_SIZE:
        return _fail(f"world_size {world_size} is outside [{MIN_WORLD_SIZE}, "
                     f"{MAX_WORLD_SIZE}]")
    if isinstance(block_bytes, bool) or not isinstance(block_bytes, int):
        return _fail(f"block_bytes must be an int, got "
                     f"{type(block_bytes).__name__} (unit: bytes for one slot on "
                     "one rank)")
    if not MIN_BLOCK_BYTES <= block_bytes <= MAX_BLOCK_BYTES:
        return _fail(
            f"block_bytes {block_bytes} bytes is outside [{MIN_BLOCK_BYTES}, "
            f"{MAX_BLOCK_BYTES}]: this field is bytes for ONE slot ON ONE RANK "
            "(a live slot is ~11.2 MB on PP0, ~12.6 MB on PP1), so a value this "
            "small or large is a unit mix-up, not a geometry")
    if num_groups is not None:
        if isinstance(num_groups, bool) or not isinstance(num_groups, int):
            return _fail(f"num_groups must be an int or None, got "
                         f"{type(num_groups).__name__}")
        if not MIN_NUM_GROUPS <= num_groups <= MAX_NUM_GROUPS:
            return _fail(f"num_groups {num_groups} is outside [{MIN_NUM_GROUPS}, "
                         f"{MAX_NUM_GROUPS}]")
    if regions_digest is not None:
        if not isinstance(regions_digest, str) or not regions_digest.isalnum():
            return _fail("regions_digest must be an alphanumeric short digest, "
                         "not a region list")
        if len(regions_digest) > MAX_REGIONS_DIGEST_CHARS:
            return _fail(f"regions_digest is {len(regions_digest)} chars; the cap "
                         f"is {MAX_REGIONS_DIGEST_CHARS}")
    normalized = normalize_epoch(epoch)
    if not normalized["ok"]:
        return _fail(str(normalized["reason"]))
    if normalized["changed"]:
        return _fail(f"epoch is not usable in a file name as given "
                     f"({normalized['note']}); normalize it deliberately, not "
                     "here")
    if geometry_digest is None:
        digest = compute_geometry_digest(block_bytes=block_bytes,
                                         world_size=world_size,
                                         num_groups=num_groups,
                                         regions_digest=regions_digest)
    else:
        if (not isinstance(geometry_digest, str)
                or len(geometry_digest) != GEOMETRY_DIGEST_CHARS
                or not geometry_digest.isalnum()):
            return _fail(f"geometry_digest must be exactly "
                         f"{GEOMETRY_DIGEST_CHARS} alphanumeric characters")
        digest = geometry_digest
    geometry = {"epoch": normalized["epoch"],
                "pp_rank": pp_rank,
                "tp_rank": tp_rank,
                "rank": rank["label"],
                "world_size": world_size,
                "block_bytes": block_bytes,
                "num_groups": num_groups,
                "regions_digest": regions_digest,
                "geometry_digest": digest,
                "format_version": FORMAT_VERSION,
                "units": _units("pp_rank", "tp_rank", "world_size",
                                "block_bytes", "num_groups", "format_version")}
    return _ok(units={}, geometry=geometry)


def geometry_from_fields(fields) -> dict:
    """Rebuild a geometry from a parsed-name dict plus the caller's own sizes.

    A file name carries epoch, rank and geometry digest only: ``world_size`` and
    ``block_bytes`` come from ``fields`` when the caller has them (a stage-2
    caller reading its own boot's geometry does; a sweep does not need them).
    Refuses a partial or wrongly-shaped set rather than defaulting into a layout
    that would then be used for offset math.
    """
    if not isinstance(fields, dict):
        return _fail(f"geometry fields must be a dict, got "
                     f"{type(fields).__name__}")
    return build_geometry(epoch=fields.get("epoch"),
                          pp_rank=fields.get("pp_rank"),
                          tp_rank=fields.get("tp_rank"),
                          world_size=fields.get("world_size"),
                          block_bytes=fields.get("block_bytes"),
                          num_groups=fields.get("num_groups"),
                          regions_digest=fields.get("regions_digest"),
                          geometry_digest=fields.get("geometry_digest"))


def geometry_section(geometry) -> dict:
    """The payload view of a geometry. An absent geometry says so, with nulls."""
    fields = ("epoch", "rank", "pp_rank", "tp_rank", "world_size", "block_bytes",
              "num_groups", "regions_digest", "geometry_digest",
              "format_version")
    units = _units("pp_rank", "tp_rank", "world_size", "block_bytes",
                   "num_groups", "format_version")
    if not isinstance(geometry, dict):
        out = {name: None for name in fields}
        out.update({"present": False,
                    "reason": "no geometry was handed in",
                    "units": units})
        return out
    out = {name: geometry.get(name) for name in fields}
    out.update({"present": True, "reason": None, "units": units})
    return out


# --- file names --------------------------------------------------------------

def build_filename(geometry, suffix: str = SUFFIX_DATA) -> dict:
    """``kvtier-<epoch>-PP<n>_TP<m>-g<geomdigest>.bin`` - and nothing else.

    Refuses a geometry that did not come from ``build_geometry``: a name emitted
    here is a name ``parse_path`` must accept and the sweep must be able to
    classify, so a half-built geometry is refused now rather than becoming an
    unclassifiable file on a live host later.
    """
    if suffix not in KNOWN_SUFFIXES:
        return _fail(f"suffix {suffix!r} is not one of {list(KNOWN_SUFFIXES)}")
    if not isinstance(geometry, dict):
        return _fail("geometry must be a dict from build_geometry")
    missing = [name for name in ("epoch", "rank", "geometry_digest")
               if not geometry.get(name)]
    if missing:
        return _fail(f"geometry is missing {missing}")
    normalized = normalize_epoch(geometry["epoch"])
    if not normalized["ok"]:
        return _fail(str(normalized["reason"]))
    if normalized["changed"]:
        return _fail("the geometry's epoch is not file-name safe as given")
    digest = geometry["geometry_digest"]
    if len(digest) != GEOMETRY_DIGEST_CHARS or not digest.isalnum():
        return _fail(f"geometry_digest {digest!r} is not "
                     f"{GEOMETRY_DIGEST_CHARS} alphanumeric characters")
    rank = geometry["rank"]
    parsed_rank = _parse_rank(rank)
    if not parsed_rank["ok"]:
        return _fail(f"rank label {rank!r} is not in the name shape "
                     f"(PP<n>_TP<m>): {parsed_rank['reason']}")
    name = f"{FILE_PREFIX}{normalized['epoch']}-{rank}-g{digest}{suffix}"
    if len(name) > MAX_PATH_CHARS:
        return _fail(f"the file name is {len(name)} chars; the cap is "
                     f"{MAX_PATH_CHARS}")
    return _ok(units=_units("name_chars"), filename=name, suffix=suffix,
               name_chars=len(name))


def build_paths(directory, geometry) -> dict:
    """The two paths this rank owns. Creates nothing, resolves nothing.

    ``directory`` is only joined, never ``realpath``-ed: stage 2 points it at
    wherever the operator said, and this module deliberately never walks.
    """
    if not isinstance(directory, (str, os.PathLike)):
        return _fail(f"directory must be a path, got {type(directory).__name__}")
    try:
        text = os.fspath(directory)
    except TypeError as exc:
        return _fail(f"directory is not a filesystem path: {type(exc).__name__}")
    if not isinstance(text, str):
        return _fail("directory must be a str path (bytes paths are refused)")
    if not text or "\x00" in text:
        return _fail("directory is empty or contains a NUL byte")
    if len(text) > MAX_PATH_CHARS:
        return _fail(f"directory is {len(text)} chars; the cap is "
                     f"{MAX_PATH_CHARS}")
    data = build_filename(geometry, SUFFIX_DATA)
    index = build_filename(geometry, SUFFIX_INDEX)
    if not data["ok"]:
        return _fail(str(data["reason"]))
    if not index["ok"]:
        return _fail(str(index["reason"]))
    return _ok(units={}, directory=text,
               data_path=os.path.join(text, data["filename"]),
               index_path=os.path.join(text, index["filename"]),
               data_filename=data["filename"],
               index_filename=index["filename"],
               created=False)


def _parse_rank(text: str) -> dict:
    """``PP<n>_TP<m>`` exactly, or a reason. No fuzziness: ``PP0TP1`` is not it."""
    if not isinstance(text, str):
        return _fail(f"rank field {text!r} is not a string")
    parts = text.split("_")
    if len(parts) != 2:
        return _fail(f"rank field {text!r} must be PP<n>_TP<m>")
    head, tail = parts
    if not head.startswith("PP") or not tail.startswith("TP"):
        return _fail(f"rank field {text!r} must be PP<n>_TP<m>")
    pp, tp = head[2:], tail[2:]
    if not pp.isdigit() or not tp.isdigit():
        return _fail(f"rank field {text!r} has a non-numeric rank")
    if len(pp) > 8 or len(tp) > 8:
        return _fail(f"rank field {text!r} has an implausibly long rank number")
    return {"ok": True, "reason": None, "pp_rank": int(pp), "tp_rank": int(tp)}


def parse_path(name) -> dict:
    """Parse one of our file names. TOTAL: never raises, always classifies.

    ``ok`` only for a name that is exactly ours and fully parses. A name that does
    not start with our prefix or does not end in a known suffix is ``not_ours`` -
    the sweep leaves those alone and does not even call them malformed, because
    they belong to someone else. A name that looks like ours but whose fields do
    not parse is ``malformed`` with a reason naming the field.
    """
    text = _name_text(name)
    if not text["ok"]:
        return _fail(str(text["reason"]), classification=NAME_REFUSED,
                     ours=False, filename=None)
    base = os.path.basename(text["text"].rstrip("/")) if text["text"] else ""
    if not base or "\x00" in base:
        return _fail("the name is empty or contains a NUL byte",
                     classification=NAME_NOTOURS, ours=False, filename=base)
    if base in (".", ".."):
        return _fail("the name is a directory reference",
                     classification=NAME_NOTOURS, ours=False, filename=base)
    if not base.startswith(FILE_PREFIX):
        return _fail(f"the name does not start with {FILE_PREFIX!r}",
                     classification=NAME_NOTOURS, ours=False, filename=base)
    suffix = next((tail for tail in KNOWN_SUFFIXES if base.endswith(tail)), None)
    if suffix is None:
        return _fail(f"the name does not end with one of {list(KNOWN_SUFFIXES)}",
                     classification=NAME_NOTOURS, ours=False, filename=base)
    body = base[len(FILE_PREFIX):-len(suffix)]
    if "/" in body or "\\" in body:
        return _fail("the name carries a path separator inside its fields",
                     classification=NAME_MALFORMED, ours=False, filename=base)
    fields = body.split("-")
    if len(fields) != 3:
        return _fail(f"expected exactly 3 '-'-separated fields after "
                     f"{FILE_PREFIX!r}, found {len(fields)}",
                     classification=NAME_MALFORMED, ours=False, filename=base)
    epoch, rank, geom_field = fields
    if not epoch or any(char not in EPOCH_ALPHABET for char in epoch):
        return _fail(f"the epoch field {epoch!r} is empty or outside "
                     "[A-Za-z0-9_]", classification=NAME_MALFORMED, ours=False,
                     filename=base)
    rank_info = _parse_rank(rank)
    if not rank_info["ok"]:
        return _fail(str(rank_info["reason"]), classification=NAME_MALFORMED,
                     ours=False, filename=base)
    digest = geom_field[1:] if geom_field.startswith("g") else ""
    if (not digest or len(digest) != GEOMETRY_DIGEST_CHARS
            or not digest.isalnum()):
        return _fail(f"the geometry field {geom_field!r} must be 'g' plus exactly "
                     f"{GEOMETRY_DIGEST_CHARS} alphanumeric characters",
                     classification=NAME_MALFORMED, ours=False, filename=base)
    return {"ok": True,
            "reason": None,
            "classification": None,     # a bare name cannot be current or stale
            "ours": True,
            "filename": base,
            "suffix": suffix,
            "epoch": epoch,
            "rank": rank,
            "pp_rank": rank_info["pp_rank"],
            "tp_rank": rank_info["tp_rank"],
            "geometry_digest": digest,
            "units": _units("pp_rank", "tp_rank")}


def _name_text(name) -> dict:
    """Get a str name out of whatever was handed in, or a refusal. Never raises."""
    if name is None:
        return _fail("the name is None")
    if isinstance(name, bytes):
        return _fail("the name must be a str path, not bytes")
    if not isinstance(name, (str, os.PathLike)):
        return _fail(f"the name must be a path or string, got "
                     f"{type(name).__name__}")
    try:
        text = os.fspath(name)
    except TypeError as exc:
        return _fail(f"the name is not a filesystem path: {type(exc).__name__}")
    if not isinstance(text, str):
        return _fail(f"the name must resolve to str, got {type(text).__name__}")
    return {"ok": True, "reason": None, "text": text}


def classify_name(name, geometry, world_size_matches: bool = True) -> dict:
    """Is this file OURS, and if so current, stale-epoch or stale-geometry?

    THE ORDER IS A CONTRACT: rank, then epoch, then geometry.
    ``classify_name(PP1 file, PP0 geometry)`` must answer BEFORE it looks at the
    epoch or the digest: the PP0 and PP1 slots are different sizes (11.2 MB and
    12.6 MB), so PP1's file is always a geometry mismatch from PP0's point of
    view, and an ordering that checked the digest first would hand every rank a
    plan to delete its peers' data. A peer rank's file is ``not_ours``, with a
    reason that says so. Then epoch, then geometry: a file from last boot with a
    different geometry is ``stale_epoch``, because the epoch is the fact an
    operator recognises and the reason it is unlinkable is the same either way.
    ``world_size_matches`` is handed in and never guessed from the name (world
    size is not in it): a boot whose world size changed cannot reuse another
    rank's file, so stage 2 passes False and this says ``stale_geometry``.
    """
    parsed = parse_path(name)
    if not parsed["ok"]:
        return {"ok": False,
                "reason": str(parsed["reason"]),
                "classification": parsed["classification"],
                "ours": bool(parsed["ours"]),
                "filename": parsed["filename"],
                "unlinkable": False,
                "units": {}}
    if not isinstance(geometry, dict) or not geometry.get("epoch"):
        return _fail("no current geometry to compare against", units={},
                     classification=NAME_REFUSED, ours=True,
                     filename=parsed["filename"], unlinkable=False)
    if parsed["rank"] != geometry.get("rank"):
        return _ok(units={},
                   reason=(f"rank {parsed['rank']!r} is not this rank's "
                           f"{geometry.get('rank')!r}: a peer rank's file is not "
                           "this rank's to unlink"),
                   classification=NAME_NOTOURS, ours=False,
                   filename=parsed["filename"], unlinkable=False)
    if parsed["epoch"] != geometry["epoch"]:
        return _ok(units={},
                   reason=(f"epoch {parsed['epoch']!r} is not this boot's "
                           f"{geometry['epoch']!r}"),
                   classification=NAME_STALE_EPOCH, ours=True,
                   filename=parsed["filename"], unlinkable=True)
    if (parsed["geometry_digest"] != geometry.get("geometry_digest")
            or not world_size_matches):
        if not world_size_matches:
            detail = ("this boot's world_size differs from the one that wrote "
                      f"{parsed['filename']} (the fork's capacity is not divided "
                      "by world_size, so a changed size means a wrong cap)")
        else:
            detail = (f"geometry {parsed['geometry_digest']!r} is not this boot's "
                      f"{geometry.get('geometry_digest')!r}")
        return _ok(units={}, reason=detail,
                   classification=NAME_STALE_GEOMETRY, ours=True,
                   filename=parsed["filename"], unlinkable=True)
    return _ok(units={}, reason=None, classification=NAME_CURRENT, ours=True,
               filename=parsed["filename"], unlinkable=False)


# --- offset arithmetic -------------------------------------------------------

def _block_bytes_of(geometry) -> dict:
    """Accept a geometry dict or a bare ``block_bytes`` int, and validate it."""
    value = geometry.get("block_bytes") if isinstance(geometry, dict) else geometry
    if isinstance(value, bool) or not isinstance(value, int):
        return _fail("block_bytes is missing or not an int (unit: bytes for one "
                     "slot on one rank)")
    if not MIN_BLOCK_BYTES <= value <= MAX_BLOCK_BYTES:
        return _fail(f"block_bytes {value} bytes is outside "
                     f"[{MIN_BLOCK_BYTES}, {MAX_BLOCK_BYTES}]")
    return {"ok": True, "reason": None, "block_bytes": value}


def _header_of(header_bytes) -> dict:
    if header_bytes is None:
        return {"ok": True, "reason": None, "header_bytes": 0}
    if isinstance(header_bytes, bool) or not isinstance(header_bytes, int):
        return _fail(f"header_bytes must be an int or None, got "
                     f"{type(header_bytes).__name__} (unit: bytes)")
    if not 0 <= header_bytes <= MAX_HEADER_BYTES:
        return _fail(f"header_bytes {header_bytes} bytes is outside "
                     f"[0, {MAX_HEADER_BYTES}]")
    return {"ok": True, "reason": None, "header_bytes": header_bytes}


def offset_for_slot(geometry, slot, header_bytes=None) -> dict:
    """The file offset of one slot: ``header_bytes + slot * block_bytes``.

    An off-by-one here silently corrupts live KV, so every boundary is REFUSED
    rather than clamped: a negative slot, a slot past ``MAX_SLOT``, and a span
    ending past ``MAX_OFFSET`` all come back as a classified failure carrying no
    number. Bounds against the FILE are a separate question (`bounds_for_file`,
    `slot_is_readable`) because "where does slot N live" and "has slot N been
    written" are different facts, and conflating them is how a partial tail gets
    read as a slot.
    """
    block = _block_bytes_of(geometry)
    if not block["ok"]:
        return _fail(str(block["reason"]))
    head = _header_of(header_bytes)
    if not head["ok"]:
        return _fail(str(head["reason"]))
    if isinstance(slot, bool) or not isinstance(slot, int):
        return _fail(f"slot must be an int, got {type(slot).__name__} "
                     "(unit: slots)")
    if slot < 0:
        return _fail(f"slot {slot} is negative")
    if slot > MAX_SLOT:
        return _fail(f"slot {slot} exceeds the cap {MAX_SLOT} slots")
    offset = head["header_bytes"] + slot * block["block_bytes"]
    end = offset + block["block_bytes"]
    if offset > MAX_OFFSET or end > MAX_OFFSET:
        return _fail(f"the slot's span {offset}..{end} bytes is past the cap "
                     f"{MAX_OFFSET} bytes")
    return _ok(units=_units("byte_offset", "end_offset_bytes", "slot",
                            "block_bytes", "header_bytes"),
               byte_offset=offset, end_offset_bytes=end, slot=slot,
               block_bytes=block["block_bytes"],
               header_bytes=head["header_bytes"])


def slot_for_offset(geometry, offset, header_bytes=None) -> dict:
    """The inverse of `offset_for_slot`, refusing anything off a slot boundary.

    A non-zero remainder means the offset does not start a slot. That is not a
    rounding question: a read from a non-boundary returns the middle of one slot
    plus the head of the next, which is KV-shaped garbage rather than a slightly
    wrong answer. So: refuse.
    """
    block = _block_bytes_of(geometry)
    if not block["ok"]:
        return _fail(str(block["reason"]))
    head = _header_of(header_bytes)
    if not head["ok"]:
        return _fail(str(head["reason"]))
    if isinstance(offset, bool) or not isinstance(offset, int):
        return _fail(f"offset must be an int, got {type(offset).__name__} "
                     "(unit: bytes)")
    if offset > MAX_OFFSET:
        return _fail(f"offset {offset} bytes exceeds the cap {MAX_OFFSET}")
    if offset < head["header_bytes"]:
        return _fail(f"offset {offset} bytes is inside the "
                     f"{head['header_bytes']}-byte header")
    remainder = (offset - head["header_bytes"]) % block["block_bytes"]
    if remainder:
        return _fail(f"offset {offset} bytes is {remainder} bytes past the start "
                     f"of a {block['block_bytes']}-byte slot")
    slot = (offset - head["header_bytes"]) // block["block_bytes"]
    if slot > MAX_SLOT:
        return _fail(f"slot {slot} exceeds the cap {MAX_SLOT} slots")
    return _ok(units=_units("slot", "block_bytes", "offset_bytes"),
               slot=slot, offset_bytes=offset - head["header_bytes"],
               block_bytes=block["block_bytes"])


def bounds_for_file(geometry, file_size_bytes, header_bytes=None) -> dict:
    """What a file of THIS apparent length can be read for, and its partial tail.

    A file is written by ``pwrite`` at ``slot * block_bytes``, so its length is a
    multiple of ``block_bytes`` only if every write landed whole. A crash
    mid-write leaves a partial tail, and a partial tail is NOT a slot:
    ``capacity_slots`` floors, ``last_complete_slot`` is the highest readable one
    and ``one_past_last_slot`` is the first refused one. Compare against THOSE,
    never against ``st_size``.
    """
    block = _block_bytes_of(geometry)
    if not block["ok"]:
        return _fail(str(block["reason"]))
    head = _header_of(header_bytes)
    if not head["ok"]:
        return _fail(str(head["reason"]))
    if isinstance(file_size_bytes, bool) or not isinstance(file_size_bytes, int):
        return _fail(f"file_size_bytes must be an int, got "
                     f"{type(file_size_bytes).__name__} (unit: bytes)")
    if file_size_bytes < 0:
        return _fail("file_size_bytes is negative")
    body = file_size_bytes - head["header_bytes"]
    if body < 0:
        return _fail(f"the file is {file_size_bytes} bytes, shorter than its "
                     f"{head['header_bytes']}-byte header")
    capacity = body // block["block_bytes"]
    tail = body - capacity * block["block_bytes"]
    return _ok(units=_units("file_size_bytes", "header_bytes", "block_bytes",
                            "capacity_slots", "last_complete_slot",
                            "one_past_last_slot", "partial_tail_bytes",
                            "partial_tail_slots"),
               file_size_bytes=file_size_bytes,
               header_bytes=head["header_bytes"],
               block_bytes=block["block_bytes"],
               capacity_slots=capacity,
               last_complete_slot=capacity - 1,
               one_past_last_slot=capacity,
               partial_tail_bytes=tail,
               has_partial_tail=tail > 0,
               partial_tail_slots=0 if tail == 0 else 1)


def slot_is_readable(geometry, slot, file_size_bytes, header_bytes=None,
                     capacity_slots=None) -> dict:
    """May slot N be READ from this file? With the reason it may not.

    Two caps are honoured when both are known and the TIGHTER wins: the file's
    apparent length, and the caller's slot-space size (a RAM arena has
    ``num_cpu_blocks`` slots; a file that outgrew it is a bug we want NAMED, not a
    read that quietly succeeds). Never says yes to the slot one past the end - the
    exact off-by-one that would read whatever the filesystem happens to hold past
    EOF and call it KV - and a partial trailing write is excluded by the floor in
    `bounds_for_file`, which puts it at slot ``capacity``, not at
    ``last_complete_slot``.

    THIS IS AN UPPER BOUND, NOT A LIVENESS CLAIM. ``pwrite`` at slot 5 makes slots
    0..4 readable BY LENGTH while they hold nothing but sparse zeros, so a caller
    that reads "readable" as "there is a record here" would serve zeroes as KV.
    The index is the only thing that says a slot holds a record; this function
    only says the file is long enough to hold it.
    """
    bounds = bounds_for_file(geometry, file_size_bytes, header_bytes)
    if not bounds["ok"]:
        return _fail(str(bounds["reason"]))
    if isinstance(capacity_slots, bool) or (capacity_slots is not None
                                             and not isinstance(capacity_slots,
                                                                int)):
        return _fail("capacity_slots must be an int or None (unit: slots)")
    if capacity_slots is not None and capacity_slots < 0:
        return _fail("capacity_slots is negative")
    limit = bounds["capacity_slots"]
    limit_source = "file_length"
    if capacity_slots is not None and capacity_slots < limit:
        limit = capacity_slots
        limit_source = "slot_space"
    probe = offset_for_slot(geometry, slot, header_bytes)
    if not probe["ok"]:
        return _fail(str(probe["reason"]))
    allowed = 0 <= slot < limit
    reason = None
    if not allowed:
        if slot == limit:
            reason = (f"slot {slot} is one past the last slot this file can "
                      f"supply ({limit} slots by {limit_source})")
        else:
            reason = f"slot {slot} is outside [0, {limit}) by {limit_source}"
    return _ok(units=_units("slot", "capacity_slots", "last_complete_slot"),
               readable=allowed, reason=reason, slot=slot,
               capacity_slots=limit, capacity_source=limit_source,
               last_complete_slot=bounds["last_complete_slot"])


# --- the index codec ---------------------------------------------------------

def encode_index_record(*, seq, key_digest, slot, block_bytes,
                        geometry_digest) -> dict:
    """One DATA frame's bytes. Pure: no fd, no clock, no vllm.

    ``key_digest`` is the engine's own block-hash digest bytes - content-addressed
    and immutable, which is what makes the runtime toggle cheap: switching ON
    rehashes nothing, because a key is a function of content and not of state.
    Fixed-layout ``struct`` rather than JSON because a frame's length is checked
    BEFORE it is read (so a corrupt length cannot make a reader allocate) and a
    text encoding would put a decode exception on the serving path.
    """
    for name, value in (("seq", seq), ("slot", slot), ("block_bytes", block_bytes)):
        if isinstance(value, bool) or not isinstance(value, int):
            return _fail(f"{name} must be an int, got {type(value).__name__}")
    if not 0 < seq <= MAX_SEQ:
        return _fail(f"seq {seq} is not in (0, {MAX_SEQ}] (unit: records)")
    if slot < 0 or slot > MAX_SLOT:
        return _fail(f"slot {slot} is not in [0, {MAX_SLOT}] (unit: slots)")
    if not MIN_BLOCK_BYTES <= block_bytes <= MAX_BLOCK_BYTES:
        return _fail(f"block_bytes {block_bytes} bytes is outside "
                     f"[{MIN_BLOCK_BYTES}, {MAX_BLOCK_BYTES}]")
    if isinstance(key_digest, str):
        try:
            key_digest = bytes.fromhex(key_digest)
        except ValueError:
            return _fail("key_digest is a str but not hexadecimal")
    if not isinstance(key_digest, (bytes, bytearray)):
        return _fail(f"key_digest must be bytes or a hex string, got "
                     f"{type(key_digest).__name__}")
    if not 1 <= len(key_digest) <= MAX_KEY_DIGEST_BYTES:
        return _fail(f"key_digest is {len(key_digest)} bytes; the range is "
                     f"[1, {MAX_KEY_DIGEST_BYTES}]")
    if (not isinstance(geometry_digest, str) or not geometry_digest.isalnum()
            or len(geometry_digest) != GEOMETRY_DIGEST_CHARS):
        return _fail(f"geometry_digest must be exactly {GEOMETRY_DIGEST_CHARS} "
                     "alphanumeric characters")
    geom = geometry_digest.encode("ascii")
    try:
        payload = (struct.pack(RECORD_PAYLOAD_STRUCT, seq, slot, block_bytes,
                                len(key_digest))
                   + bytes(key_digest) + geom)
    except struct.error as exc:
        # Unreachable by the checks above; here because "never raises" is the
        # contract and a validated field must not be the thing that breaks it.
        return _fail(f"the record does not fit the codec: {exc}")
    if len(payload) > MAX_FRAME_PAYLOAD_BYTES:
        return _fail(f"the payload is {len(payload)} bytes, past the cap "
                     f"{MAX_FRAME_PAYLOAD_BYTES}")
    return _ok(units=_units("payload_bytes", "entry_bytes"),
               frame=_frame(INDEX_FRAME_DATA, payload),
               payload_bytes=len(payload),
               entry_bytes=FRAME_HEADER_BYTES + len(payload) + FRAME_TRAILER_BYTES)


def encode_index_commit(*, committed_seq, entries_in_commit) -> dict:
    """One COMMIT frame: the seq watermark up to which records are visible.

    ``entries_in_commit`` is the writer's count of data frames in this commit. The
    reader REPORTS a mismatch instead of acting on it: the entries themselves are
    CRC-valid, so dropping them on a count disagreement would silently lose data,
    while a counter keeps the disagreement visible.
    """
    for name, value in (("committed_seq", committed_seq),
                        ("entries_in_commit", entries_in_commit)):
        if isinstance(value, bool) or not isinstance(value, int):
            return _fail(f"{name} must be an int, got {type(value).__name__}")
    if not 0 <= committed_seq <= MAX_SEQ:
        return _fail(f"committed_seq {committed_seq} is not in [0, {MAX_SEQ}]")
    if not 0 <= entries_in_commit <= MAX_RECORDS:
        return _fail(f"entries_in_commit {entries_in_commit} is not in "
                     f"[0, {MAX_RECORDS}]")
    payload = struct.pack(COMMIT_PAYLOAD_STRUCT, committed_seq, entries_in_commit)
    return _ok(units=_units("payload_bytes"),
               frame=_frame(INDEX_FRAME_COMMIT, payload),
               payload_bytes=len(payload))


def _frame(kind: int, payload: bytes) -> bytes:
    header = bytes((INDEX_FRAME_MAGIC, kind)) + struct.pack("!I", len(payload))
    crc = zlib.crc32(header + payload) & 0xFFFFFFFF
    return header + payload + struct.pack("!I", crc)


def decode_index_frame(data) -> dict:
    """Decode ONE frame from its exact bytes. Never raises; classifies.

    Exposed for the tests and used by the streaming reader below, so a frame that
    round-trips here is a frame the reader accepts: the two can never disagree
    about which records are visible.
    """
    if not isinstance(data, (bytes, bytearray)):
        return _fail(f"a frame must be bytes, got {type(data).__name__}",
                     failure="bad_payload")
    raw = bytes(data)
    if len(raw) < FRAME_HEADER_BYTES + FRAME_TRAILER_BYTES:
        return _fail(f"the frame is {len(raw)} bytes, shorter than a header plus "
                     f"a trailer", failure="truncated")
    header = raw[:FRAME_HEADER_BYTES]
    if header[0] != INDEX_FRAME_MAGIC:
        return _fail(f"the frame magic is {header[0]!r}, not ours",
                     failure="bad_magic")
    kind = header[1]
    if kind not in INDEX_FRAME_KINDS:
        return _fail(f"the frame kind {kind!r} is unknown", failure="bad_kind")
    length = struct.unpack("!I", header[2:FRAME_HEADER_BYTES])[0]
    if length > MAX_FRAME_PAYLOAD_BYTES:
        return _fail(f"the frame claims {length} payload bytes; the cap is "
                     f"{MAX_FRAME_PAYLOAD_BYTES}", failure="oversize_payload")
    if len(raw) != FRAME_HEADER_BYTES + length + FRAME_TRAILER_BYTES:
        return _fail(f"the frame is {len(raw)} bytes but its header claims "
                     f"{FRAME_HEADER_BYTES + length + FRAME_TRAILER_BYTES}",
                     failure="truncated")
    payload = raw[FRAME_HEADER_BYTES:FRAME_HEADER_BYTES + length]
    want = struct.unpack("!I", raw[-FRAME_TRAILER_BYTES:])[0]
    if (zlib.crc32(header + payload) & 0xFFFFFFFF) != want:
        return _fail("the frame CRC does not match its payload",
                     failure="bad_crc")
    if kind == INDEX_FRAME_DATA:
        if length < RECORD_PAYLOAD_HEAD:
            return _fail(f"a data payload of {length} bytes is shorter than the "
                         f"{RECORD_PAYLOAD_HEAD}-byte record head",
                         failure="bad_payload")
        seq, slot, block_bytes, key_len = struct.unpack(
            RECORD_PAYLOAD_STRUCT, payload[:RECORD_PAYLOAD_HEAD])
        if length - RECORD_PAYLOAD_HEAD - key_len != GEOMETRY_DIGEST_CHARS:
            return _fail("the data payload's tail is not a geometry digest",
                         failure="bad_payload")
        return _ok(units=_units("seq", "slot", "block_bytes"),
                   kind="data", seq=seq, slot=slot, block_bytes=block_bytes,
                   key_digest=payload[RECORD_PAYLOAD_HEAD:
                                       RECORD_PAYLOAD_HEAD + key_len],
                   geometry_digest=payload[-GEOMETRY_DIGEST_CHARS:].decode(
                       "ascii", "replace"))
    if length != COMMIT_PAYLOAD_BYTES:
        return _fail(f"a commit payload of {length} bytes is not "
                     f"{COMMIT_PAYLOAD_BYTES}", failure="bad_payload")
    seq, entries = struct.unpack(COMMIT_PAYLOAD_STRUCT, payload)
    return _ok(units=_units("committed_seq", "entries_in_commit"),
               kind="commit", committed_seq=seq, entries_in_commit=entries)


def read_index(path, *, max_records: int = DEFAULT_MAX_RECORDS,
               max_frames: int = DEFAULT_MAX_FRAMES) -> dict:
    """Read the index. Streams it, applies the watermark, never raises.

    Frames are read one at a time and a frame's length is checked against the cap
    BEFORE its payload is read, so a corrupt or hostile length cannot make this
    allocate. The scan STOPS at the first torn or bad-CRC frame: with one
    append-only writer a break can only be in the tail, and anything past a break
    is unprovable, so it is not read and ``stopped_at_bytes`` says where. Records
    are visible iff ``seq <= watermark``, the watermark being the highest
    committed seq a reached COMMIT frame claimed - never "everything in the file".
    """
    try:
        size = int(os.path.getsize(path))
    except FileNotFoundError:
        return _index_view(_MISSING,
                           "the index file is not there: no records, which is a "
                           "fact and not a read failure",
                           path=str(path), exists=False, index_bytes=None)
    except OSError as exc:
        return _index_view(_READ_FAILED,
                           f"{type(exc).__name__}: "
                           f"{str(exc)[:MAX_STAT_ERROR_CHARS]}",
                           path=str(path), exists=True, index_bytes=None)
    payload = _read_index_stream(path, max_records=max_records,
                                 max_frames=max_frames)
    return _index_view(payload["state"], payload["reason"], path=str(path),
                       exists=True, index_bytes=size, values=payload)


def _index_view(state: str, reason: str | None, *, path: str, exists: bool,
                index_bytes: int | None, values: dict | None = None) -> dict:
    """One key set for EVERY index answer: present, missing, unreadable, torn.

    A degraded payload that is MISSING keys is how a reader downstream starts
    treating an absent measurement as a zero, so every shape is assembled from
    `_empty_index_payload` and then overwritten with what was actually measured.
    """
    out = _empty_index_payload(state, reason)
    for key, value in (values or {}).items():
        if key in out:
            out[key] = value
    out["path"] = path
    out["exists"] = exists
    out["index_bytes"] = index_bytes
    return out


def _index_units() -> dict:
    return _units("index_bytes", "records", "records_parsed", "records_visible",
                  "committed_seq", "highest_seq_seen", "frames",
                  "uncommitted_records", "duplicate_keys", "duplicate_seqs",
                  "conflicting_slots", "out_of_order_commits",
                  "commit_count_mismatches", "stopped_at_bytes")


def _empty_index_payload(state: str, reason: str) -> dict:
    """The absent view, built through the SAME keys as the healthy one.

    A degraded payload that is missing keys is how a reader starts treating an
    absent measurement as a zero, so the degraded and healthy shapes are the same
    dict with different values.
    """
    return _ok(units=_index_units(), state=state, reason=reason,
               path=None, exists=False, index_bytes=None, records=[],
               committed_seq=None, highest_seq_seen=0, records_parsed=0,
               records_visible=0, frames=0, truncated=False,
               stopped_at_bytes=None, failures=[],
               uncommitted_records=0, duplicate_keys=0, duplicate_seqs=0,
               conflicting_slots=0, out_of_order_commits=0,
               commit_count_mismatches=0)


def _read_index_stream(path, *, max_records: int, max_frames: int) -> dict:
    """The streaming reader: ONE open, ONE frame at a time, bounded by both caps.

    This is the ONLY place the visibility rule lives, so `read_index` and any
    future caller cannot disagree about which records exist.
    """
    records_cap = _cap(max_records, "max_records", low=0, high=MAX_RECORDS)
    frames_cap = _cap(max_frames, "max_frames", low=1, high=MAX_FRAMES)
    if not records_cap["ok"]:
        return _index_refusal(records_cap["reason"])
    if not frames_cap["ok"]:
        return _index_refusal(frames_cap["reason"])
    limit_records = records_cap["value"]
    limit_frames = frames_cap["value"]
    counts = dict.fromkeys(("frames", "records_parsed", "duplicate_keys",
                            "duplicate_seqs", "conflicting_slots",
                            "out_of_order_commits", "commit_count_mismatches"), 0)
    records: list[dict] = []
    seen_keys: set[bytes] = set()
    seen_seqs: set[int] = set()
    slot_owner: dict[int, bytes] = {}
    watermark = 0
    highest = 0
    state = _OK_DIR
    stopped_at = None
    failures: list[dict] = []
    truncated = False
    try:
        with open(path, "rb") as handle:
            while counts["frames"] < limit_frames:
                start = handle.tell()
                header = handle.read(FRAME_HEADER_BYTES)
                if not header:
                    break                                   # clean EOF
                if len(header) < FRAME_HEADER_BYTES:
                    state, stopped_at = _TRUNCATED, start
                    failures.append(_failure_at(start, "truncated"))
                    break
                if header[0] != INDEX_FRAME_MAGIC:
                    state, stopped_at = _TRUNCATED, start
                    failures.append(_failure_at(start, "bad_magic"))
                    break
                kind = header[1]
                if kind not in INDEX_FRAME_KINDS:
                    state, stopped_at = _TRUNCATED, start
                    failures.append(_failure_at(start, "bad_kind"))
                    break
                length = struct.unpack("!I", header[2:FRAME_HEADER_BYTES])[0]
                if length > MAX_FRAME_PAYLOAD_BYTES:
                    # Checked BEFORE the read: this is the whole reason a corrupt
                    # file cannot make this reader allocate.
                    state, stopped_at = _TRUNCATED, start
                    failures.append(_failure_at(start, "oversize_payload"))
                    break
                body = handle.read(length + FRAME_TRAILER_BYTES)
                if len(body) < length + FRAME_TRAILER_BYTES:
                    state, stopped_at = _TRUNCATED, start
                    failures.append(_failure_at(start, "truncated"))
                    break
                counts["frames"] += 1
                frame = decode_index_frame(header + body)
                if not frame["ok"]:
                    state, stopped_at = _TRUNCATED, start
                    failures.append(_failure_at(start, str(frame["failure"])))
                    break
                if frame["kind"] == "commit":
                    if frame["committed_seq"] < watermark:
                        counts["out_of_order_commits"] += 1
                    else:
                        watermark = int(frame["committed_seq"])
                    if counts["records_parsed"] < frame["entries_in_commit"]:
                        counts["commit_count_mismatches"] += 1
                    continue
                seq = int(frame["seq"])
                highest = max(highest, seq)
                counts["records_parsed"] += 1
                counts["duplicate_seqs"] += int(seq in seen_seqs)
                seen_seqs.add(seq)
                key = bytes(frame["key_digest"])
                counts["duplicate_keys"] += int(key in seen_keys)
                seen_keys.add(key)
                slot = int(frame["slot"])
                previous = slot_owner.get(slot)
                counts["conflicting_slots"] += int(previous is not None
                                                   and previous != key)
                slot_owner[slot] = key
                records.append({"seq": seq,
                                "slot": slot,
                                "key_digest": key,
                                "key_digest_hex": key.hex(),
                                "block_bytes": int(frame["block_bytes"]),
                                "geometry_digest": frame["geometry_digest"],
                                "units": _units("seq", "slot", "block_bytes")})
                if len(records) >= limit_records:
                    truncated = True
                    break
    except OSError as exc:
        return _index_refusal(
            f"the index could not be read: {type(exc).__name__}")
    # The watermark, applied once at the end: a record is visible only if a COMMIT
    # the scan actually reached claims it. Records past it are counted, not shown.
    visible = [item for item in records if item["seq"] <= watermark]
    reason = (None if state == _OK_DIR else
              f"the scan stopped at byte {stopped_at}: "
              f"{failures[-1]['failure'] if failures else _TRUNCATED}")
    payload = _empty_index_payload(state, reason)
    payload.update(records=visible, committed_seq=watermark,
                   highest_seq_seen=highest,
                   records_parsed=counts["records_parsed"],
                   records_visible=len(visible), frames=counts["frames"],
                   truncated=truncated, stopped_at_bytes=stopped_at,
                   failures=failures,
                   uncommitted_records=len(records) - len(visible),
                   duplicate_keys=counts["duplicate_keys"],
                   duplicate_seqs=counts["duplicate_seqs"],
                   conflicting_slots=counts["conflicting_slots"],
                   out_of_order_commits=counts["out_of_order_commits"],
                   commit_count_mismatches=counts["commit_count_mismatches"])
    return payload


def _index_refusal(reason: str) -> dict:
    """A read that could not be attempted: the healthy keys, all of them null."""
    out = _empty_index_payload(_READ_FAILED, reason)
    out["ok"] = False
    return out


def _failure_at(offset: int, failure: str) -> dict:
    return {"stopped_at_bytes": int(offset), "failure": failure,
            "units": _units("stopped_at_bytes")}


def index_batch_bytes() -> dict:
    """The writer's budget: bytes per frame, worst case, plus the caps.

    So a batch buffer is sized by the codec rather than by a guess:
    ``entry_bytes`` is the WORST case (a maximum-length key digest), which lets
    stage 2 allocate one buffer once and never grow it per spill.
    """
    worst = RECORD_PAYLOAD_HEAD + MAX_KEY_DIGEST_BYTES + GEOMETRY_DIGEST_CHARS
    if worst > MAX_FRAME_PAYLOAD_BYTES:
        return _fail(f"the worst-case record payload {worst} bytes exceeds the "
                     f"frame cap {MAX_FRAME_PAYLOAD_BYTES}")
    return _ok(units=_units("entry_bytes", "payload_bytes", "commit_bytes",
                            "max_key_digest_bytes", "records"),
               entry_bytes=FRAME_HEADER_BYTES + worst + FRAME_TRAILER_BYTES,
               payload_bytes=worst,
               commit_bytes=FRAME_HEADER_BYTES + COMMIT_PAYLOAD_BYTES
               + FRAME_TRAILER_BYTES,
               max_key_digest_bytes=MAX_KEY_DIGEST_BYTES,
               records=MAX_RECORDS)


def append_index(path, frames, *, sync: bool = False) -> dict:
    """Append frames: ONE ``os.write`` for the whole batch, opened O_APPEND.

    A single write of a bounded buffer is the smallest reasonable atomic unit
    short of O_DIRECT, and correctness never depends on it succeeding whole: a
    partial or torn tail is invisible to `read_index` because of the watermark
    rule. This does NOT fsync the data file - read the docstring's cross-boot
    paragraph before changing that. An oversize batch is a refusal, not a silent
    split: making another commit is the same thing a crash would have forced.
    """
    if isinstance(frames, (bytes, bytearray)):
        frames = [frames]
    try:
        items = [bytes(item) for item in (frames or [])]
    except TypeError as exc:
        return _fail(f"every frame must be bytes: {type(exc).__name__}",
                     units=_units("bytes_written", "records"))
    if not items:
        return _ok(units=_units("bytes_written", "records"), bytes_written=0,
                   records=0, reason="no frames were handed in")
    if len(items) > MAX_RECORDS + 1:
        return _fail(f"{len(items)} frames in one batch; the cap is "
                     f"{MAX_RECORDS} records plus a commit",
                     units=_units("bytes_written", "records"))
    blob = b"".join(items)
    limit = (MAX_FRAME_PAYLOAD_BYTES + FRAME_HEADER_BYTES + FRAME_TRAILER_BYTES) \
        * (MAX_RECORDS + 1)
    if len(blob) > limit:
        return _fail(f"the batch is {len(blob)} bytes, past the cap {limit}",
                     units=_units("bytes_written", "records"))
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError as exc:
        return _fail(f"the index could not be opened for append: "
                     f"{type(exc).__name__}",
                     units=_units("bytes_written", "records"))
    try:
        written = int(os.write(fd, blob))
        if sync:
            os.fsync(fd)
    except OSError as exc:
        os.close(fd)
        return _fail(f"the index append failed: {type(exc).__name__}",
                     units=_units("bytes_written", "records"))
    os.close(fd)
    data_frames = sum(1 for item in items
                      if item[:2] == bytes((INDEX_FRAME_MAGIC,
                                            INDEX_FRAME_DATA)))
    return _ok(units=_units("bytes_written", "records"),
               bytes_written=written, records=data_frames)


# --- the boot sweep ----------------------------------------------------------

def boot_sweep(directory, geometry, *, world_size_matches: bool = True,
               max_files: int = MAX_SWEEP_FILES,
               max_plan: int = MAX_UNLINK_PLAN) -> dict:
    """Classify the files in ``directory`` and return an UNLINK PLAN. Read-only.

    Nothing here deletes: the plan is returned for the caller (and, today, for a
    human) to carry through `execute_unlink_plan`, which re-validates every entry.
    That two-phase shape is deliberate - this is the one function in the plugin
    whose success destroys data.

    Rules, in order of how much getting them wrong costs:
      * a name that is not exactly ours is ``not_ours``: counted, reported, never
        in the plan. A near-miss like ``kvtier-backup-notes.bin`` is
        ``malformed`` (our shape, unparseable fields) and is ALSO never in the
        plan, because our prefix and suffix are not proof of ownership;
      * a symlink, a directory, a socket or anything else that is not a regular
        file is ``refused`` and never touched: this sweep does not follow links
        out of a directory it was merely pointed at;
      * ``stale_epoch`` and ``stale_geometry`` files of OUR OWN rank are the plan,
        capped by ``max_plan`` and reported by ``plan_truncated`` when the bites;
      * the walk is one non-recursive ``scandir`` bounded by ``max_files``, and
        ``walk_truncated`` says when that bound was hit, because an incomplete
        sweep must never be read as "there is nothing stale left".
    """
    state = _sweep_state(directory)
    if state["state"] != _OK_DIR:
        return _ok(units=_sweep_units(), directory=str(directory),
                   directory_state=state["state"], reason=state["reason"],
                   files=[], planned_paths=[], counts=_empty_sweep_counts(),
                   scanned=0, plan_truncated=False, walk_truncated=False)
    if os.path.islink(directory):
        # A symlinked DIRECTORY is the one way `directory` itself could walk out
        # of the tree, and an operator-owned path is exactly where that happens.
        # Refuse rather than resolve: deleting through an unvetted link is the
        # failure this function exists to avoid.
        return _fail("the directory is itself a symlink: refusing to sweep "
                     "through a link we did not vet",
                     units=_sweep_units(), directory=str(directory),
                     directory_state=NAME_REFUSED, files=[], planned_paths=[],
                     counts=_empty_sweep_counts(), scanned=0,
                     plan_truncated=False, walk_truncated=False)
    files_cap = _cap(max_files, "max_files", low=1, high=MAX_SWEEP_FILES)
    plan_cap = _cap(max_plan, "max_plan", low=0, high=MAX_UNLINK_PLAN)
    refusal = files_cap["reason"] or plan_cap["reason"]
    if refusal:
        return _fail(refusal, units=_sweep_units(), directory=str(directory),
                     directory_state=_READ_FAILED, files=[], planned_paths=[],
                     counts=_empty_sweep_counts(), scanned=0,
                     plan_truncated=False, walk_truncated=False)
    limit_files = files_cap["value"]
    limit_plan = plan_cap["value"]

    files: list[dict] = []
    planned: list[str] = []
    counts = _empty_sweep_counts()
    scanned = 0
    walk_truncated = False
    plan_truncated = False
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if scanned >= limit_files:
                    walk_truncated = True
                    break
                scanned += 1
                item = _classify_entry(entry, directory, geometry,
                                       world_size_matches)
                counts[item["classification"]] += 1
                counts["files"] += 1
                if item["unlinkable"] and item["regular"]:
                    if len(planned) < limit_plan:
                        planned.append(item["path"])
                        counts["planned"] += 1
                        counts["planned_bytes"] += int(item["size_bytes"] or 0)
                    else:
                        plan_truncated = True
                if len(files) < limit_files:
                    files.append(item)
    except OSError as exc:
        # Half a sweep is not a sweep: report what was seen, and plan nothing.
        return _fail(f"the directory could not be listed: {type(exc).__name__}",
                     units=_sweep_units(), directory=str(directory),
                     directory_state=_UNREADABLE, files=files,
                     planned_paths=[], counts=counts, scanned=scanned,
                     plan_truncated=plan_truncated, walk_truncated=True)
    return _ok(units=_sweep_units(), directory=str(directory),
               directory_state=_OK_DIR, reason=None, files=files,
               planned_paths=planned, counts=counts, scanned=scanned,
               plan_truncated=plan_truncated, walk_truncated=walk_truncated)


def _sweep_state(directory) -> dict:
    try:
        os.stat(directory)
    except FileNotFoundError:
        return {"state": _MISSING,
                "reason": "the directory is not there: nothing to sweep, which "
                          "is not a failure"}
    except NotADirectoryError:
        return {"state": _NOT_A_DIRECTORY,
                "reason": "the path is not a directory"}
    except OSError as exc:
        return {"state": _READ_FAILED,
                "reason": f"the directory could not be stated: "
                          f"{type(exc).__name__}"}
    if not os.path.isdir(directory):
        return {"state": _NOT_A_DIRECTORY,
                "reason": "the path exists but is not a directory"}
    if not os.access(directory, os.R_OK):
        return {"state": _UNREADABLE, "reason": "the directory is not readable"}
    return {"state": _OK_DIR, "reason": None}


def _classify_entry(entry, directory, geometry, world_size_matches) -> dict:
    """One directory entry: classified, with its reason and its safety flags."""
    path = os.path.join(str(directory), entry.name)
    name = classify_name(entry.name, geometry,
                         world_size_matches=world_size_matches)
    regular = False
    symlink = False
    size_bytes = None
    try:
        symlink = bool(entry.is_symlink())
        entry_stat = entry.stat(follow_symlinks=False)
        regular = bool(os.path.stat.S_ISREG(entry_stat.st_mode))
        size_bytes = int(entry_stat.st_size)
    except OSError as exc:
        # A file that vanishes mid-sweep is a fact about the sweep, not an error.
        # It is refused: we cannot prove what it was, so we do not delete it.
        return {"filename": entry.name, "path": path,
                "classification": NAME_REFUSED, "ours": bool(name.get("ours")),
                "reason": f"its metadata could not be read: "
                          f"{type(exc).__name__}",
                "unlinkable": False, "regular": False, "symlink": symlink,
                "size_bytes": None, "units": _units("size_bytes")}
    classification = name["classification"] or NAME_REFUSED
    reason = name["reason"]
    unlinkable = bool(name.get("unlinkable"))
    if symlink:
        classification = NAME_REFUSED
        reason = ("it is a symlink: this sweep never follows a link out of the "
                  "directory it was handed")
        unlinkable = False
    elif not regular:
        classification = NAME_REFUSED
        reason = "it is not a regular file (directory, socket, fifo or device)"
        unlinkable = False
    return {"filename": entry.name, "path": path,
            "classification": classification, "ours": bool(name.get("ours")),
            "reason": reason, "unlinkable": unlinkable, "regular": regular,
            "symlink": symlink, "size_bytes": size_bytes,
            "units": _units("size_bytes")}


def _sweep_units() -> dict:
    return _units("scanned")


def _sweep_counts_units() -> dict:
    keys = (*NAME_CLASSIFICATIONS, "planned", "unlinked", "errors", "files")
    out = {key: UNITS_BY_NAME[key] for key in keys}
    out["planned_bytes"] = "bytes"
    return out


def _empty_sweep_counts() -> dict:
    out = {name: 0 for name in NAME_CLASSIFICATIONS}
    out["files"] = 0
    out["planned"] = 0
    out["planned_bytes"] = 0
    out["units"] = _sweep_counts_units()
    return out


def execute_unlink_plan(directory, paths, *, dry_run: bool = True, geometry=None,
                        world_size_matches: bool = True,
                        max_unlinks: int = MAX_UNLINK_PLAN) -> dict:
    """Unlink the plan, re-validating EVERY entry. Destructive; dry-run by default.

    Nothing from `boot_sweep` is trusted here. For every path: it must resolve
    directly inside ``directory`` (so a plan that grew a traversal cannot walk),
    its name must parse as ours, its classification must STILL be unlinkable
    against the geometry handed in now - a caller whose geometry changed between
    planning and executing cannot delete what is now current - and it must be a
    regular file by ``lstat``, never a link. Every refusal is counted and carries
    a reason; a directory that is itself a symlink is refused outright.
    """
    if not isinstance(paths, (list, tuple)):
        return _fail(f"paths must be a list, got {type(paths).__name__}")
    plan_cap = _cap(max_unlinks, "max_unlinks", low=0, high=MAX_UNLINK_PLAN)
    if not plan_cap["ok"]:
        return _fail(plan_cap["reason"])
    limit = plan_cap["value"]
    root = os.path.realpath(str(directory))
    if os.path.isdir(root) and os.path.islink(str(directory)):
        return _fail("the directory is itself a symlink: refusing to delete "
                     "through a link we did not vet")
    counts = dict.fromkeys((*NAME_CLASSIFICATIONS, "unlinked", "refused",
                            "errors", "files"), 0)
    pending = list(paths)
    if len(pending) > limit:
        counts["refused"] += 1
        pending = pending[:limit]
        overflow = {"path": None, "action": "refused",
                    "reason": f"the plan names {len(paths)} paths; the cap is "
                              f"{limit} (call again for the rest)",
                    "size_bytes": None, "units": _units("size_bytes")}
    else:
        overflow = None
    actions: list[dict] = []
    if overflow:
        actions.append(overflow)
    bytes_freed = 0
    for path in pending:
        verdict = _validate_unlink(path, root, geometry, world_size_matches)
        if not verdict["ok"]:
            counts["refused"] += 1
            actions.append({"path": str(path), "action": "refused",
                            "reason": verdict["reason"], "size_bytes": None,
                            "units": _units("size_bytes")})
            continue
        if dry_run:
            actions.append({"path": verdict["path"], "action": "would_unlink",
                            "reason": None, "size_bytes": verdict["size_bytes"],
                            "units": _units("size_bytes")})
            continue
        try:
            os.unlink(verdict["path"])
        except FileNotFoundError:
            counts["refused"] += 1
            actions.append({"path": verdict["path"], "action": "gone",
                            "reason": "it was gone by the time we got to it",
                            "size_bytes": None, "units": _units("size_bytes")})
        except OSError as exc:
            counts["errors"] += 1
            actions.append({"path": verdict["path"], "action": "error",
                            "reason": type(exc).__name__, "size_bytes": None,
                            "units": _units("size_bytes")})
        else:
            counts["unlinked"] += 1
            counts["files"] += 1
            bytes_freed += int(verdict["size_bytes"] or 0)
            actions.append({"path": verdict["path"], "action": "unlinked",
                            "reason": None, "size_bytes": verdict["size_bytes"],
                            "units": _units("size_bytes")})
    counts["planned_bytes"] = bytes_freed
    return _ok(units={}, directory=str(directory), dry_run=bool(dry_run),
               counts=counts | {"units": _sweep_counts_units()},
               actions=actions,
               reason=("dry run: nothing was deleted" if dry_run else None))


def _validate_unlink(path, root, geometry, world_size_matches) -> dict:
    """Prove one path is a regular, non-link, stale, ours, in-directory file."""
    if not isinstance(path, (str, os.PathLike)):
        return _fail(f"the path is not a filesystem path: {type(path).__name__}")
    text = os.fspath(path)
    if not isinstance(text, str) or not text or "\x00" in text:
        return _fail("the path must be a non-empty str without a NUL byte")
    base = os.path.basename(text.rstrip("/"))
    if not base or base in (".", ".."):
        return _fail("the path has no usable final component")
    # The file must live DIRECTLY inside the directory we were given, by real
    # path: that is what makes "this sweep never walks" true even for a plan
    # someone hand-edited into `../../etc/passwd`.
    if os.path.realpath(os.path.dirname(os.path.abspath(text))) != root:
        return _fail(f"{base} does not resolve directly inside {root}: this "
                     "sweep never walks out of the directory it was handed")
    parsed = parse_path(base)
    if not parsed["ok"]:
        return _fail(f"{base} is not ours to delete: {parsed['reason']}")
    if geometry is not None:
        fresh = classify_name(base, geometry,
                              world_size_matches=world_size_matches)
        if fresh["classification"] not in UNLINKABLE_CLASSIFICATIONS:
            detail = fresh["reason"] or "it is the current boot's own file"
            return _fail(f"{base} is {fresh['classification']} against the "
                         f"current geometry: {detail}")
    target = os.path.join(root, base)
    try:
        info = os.lstat(target)
    except OSError as exc:
        return _fail(f"{base} could not be inspected: {type(exc).__name__}")
    if not os.path.stat.S_ISREG(info.st_mode):
        return _fail(f"{base} is not a regular file")
    if os.path.dirname(os.path.realpath(target)) != root:
        return _fail(f"{base} resolves outside {root}")
    return {"ok": True, "reason": None, "path": target,
            "size_bytes": int(info.st_size)}


# --- free space --------------------------------------------------------------

def free_space(path) -> dict:
    """Free bytes for the filesystem under ``path``, or an honest absence.

    ``f_bavail`` (not ``f_bfree``) is what a NON-privileged writer can actually
    allocate, so that is what "free" means here; the root reserve is published
    separately rather than silently inflating the number. A failure - a missing
    path, a permission problem, an unsupported filesystem - is
    ``available: False`` with a reason and ``None`` counts. NEVER a zero: a zero
    here would be read as "the drive is full", which is a lie that changes what
    the tier does.
    """
    if not isinstance(path, (str, os.PathLike)):
        return _space_fail(f"path must be a str or PathLike, got "
                           f"{type(path).__name__}")
    text = os.fspath(path)
    if not isinstance(text, str) or not text or "\x00" in text:
        return _space_fail("path must be a non-empty str without a NUL byte")
    try:
        stat = os.statvfs(text)
    except FileNotFoundError:
        return _space_fail("the path is not there, so the filesystem under it "
                           "is unknown")
    except NotADirectoryError:
        return _space_fail("the path is not a directory")
    except PermissionError:
        return _space_fail("the path is not searchable, so the filesystem under "
                           "it cannot be read")
    except OSError as exc:
        return _space_fail(f"statvfs failed: {type(exc).__name__}")
    unit = int(stat.f_frsize or 0)
    if unit <= 0:
        return _space_fail("statvfs reported a zero fragment size, so no byte "
                           "count can be derived from it")
    free = int(stat.f_bavail) * unit
    total = int(stat.f_blocks) * unit
    available = int(stat.f_bfree) * unit
    return _ok(units=_units("free_bytes", "total_bytes", "used_bytes",
                            "reserved_for_root_bytes", "unit_size_bytes",
                            "fragment_bytes", "files_free"),
               available=True, reason=None, statvfs_state="ok",
               path=text, free_bytes=free, total_bytes=total,
               used_bytes=max(0, total - available),
               reserved_for_root_bytes=max(0, available - free),
               unit_size_bytes=unit, fragment_bytes=int(stat.f_frsize),
               read_only=bool(stat.f_flag & 1),             # ST_RDONLY
               files_free=int(stat.f_favail))


def _space_fail(reason: str) -> dict:
    return _ok(units=_units("free_bytes", "total_bytes", "used_bytes",
                            "reserved_for_root_bytes", "unit_size_bytes",
                            "fragment_bytes", "files_free"),
               available=False, statvfs_state=_READ_FAILED, path=None,
               free_bytes=None, total_bytes=None, used_bytes=None,
               reserved_for_root_bytes=None, unit_size_bytes=None,
               fragment_bytes=None, read_only=None, files_free=None,
               reason=reason)


def space_check(path, *, watermark_bytes=None, watermark_percent=None) -> dict:
    """Is there room to write? The free-space read plus the watermark verdict.

    The watermark accepts BYTES or a PERCENT of total, and the EFFECTIVE byte
    value is published so a reader never has to re-derive it. Percent-of-total is
    a proxy, not a promise: on a RAID0 with a large root reserve the two can
    disagree, and the byte figure is the one that decides.

    Three separate facts, never collapsed: `space_available` (did we measure?),
    `below_watermark` (True/False, or None when we did not measure OR no watermark
    was configured - an unknown is never a False), and a genuinely full drive
    (available, watermark set, `below_watermark` True, with a real byte count).
    """
    space = free_space(path)
    has_bytes = watermark_bytes is not None
    has_percent = watermark_percent is not None
    if has_bytes and (isinstance(watermark_bytes, bool)
                      or not isinstance(watermark_bytes, int)):
        return _space_check_fail("watermark_bytes must be an int (unit: bytes)")
    if has_percent:
        if isinstance(watermark_percent, bool) or not isinstance(
                watermark_percent, (int, float)):
            return _space_check_fail("watermark_percent must be a number (unit: "
                                     "percent)")
        if not MIN_WATERMARK_PERCENT <= float(watermark_percent) \
                <= MAX_WATERMARK_PERCENT:
            return _space_check_fail(f"watermark_percent {watermark_percent} is "
                                     f"outside [{MIN_WATERMARK_PERCENT}, "
                                     f"{MAX_WATERMARK_PERCENT}] (it is a "
                                     "percentage of total bytes)")
    if has_bytes and watermark_bytes < 0:
        return _space_check_fail("watermark_bytes is negative")
    if not space["available"]:
        return _space_check_fail(str(space["reason"]),
                                 space_available=False, space_reason=str(
                                     space["reason"]))
    effective = 0
    source = "none"
    if has_bytes:
        effective = int(watermark_bytes)
        source = "bytes"
    elif has_percent:
        effective = int(space["total_bytes"] * float(watermark_percent) / 100.0)
        source = "percent_of_total"
    below = None
    reason = None
    if source == "none":
        reason = "no watermark was configured, so there is no verdict to give"
    else:
        below = space["free_bytes"] < effective
    return _ok(units=_units("free_bytes", "total_bytes", "watermark_bytes",
                            "used_bytes", "reserved_for_root_bytes",
                            "unit_size_bytes", "free_percent_of_total",
                            "watermark_bytes_configured", "watermark_percent"),
               ok=True, reason=reason, space_available=True,
               statvfs_state="ok", path=space["path"],
               free_bytes=space["free_bytes"], total_bytes=space["total_bytes"],
               used_bytes=space["used_bytes"],
               reserved_for_root_bytes=space["reserved_for_root_bytes"],
               unit_size_bytes=space["unit_size_bytes"],
               watermark_bytes=(effective if source != "none" else None),
               watermark_bytes_configured=(int(watermark_bytes) if has_bytes
                                           else None),
               watermark_percent=(float(watermark_percent) if has_percent
                                  else None),
               watermark_source=source, below_watermark=below,
               read_only=space["read_only"],
               free_percent_of_total=(
                   round(100.0 * space["free_bytes"] / space["total_bytes"], 3)
                   if space["total_bytes"] > 0 else None))


def _space_check_fail(reason: str, space_available: bool = True,
                      space_reason: str | None = None) -> dict:
    return _ok(units=_units("free_bytes", "total_bytes", "watermark_bytes",
                            "used_bytes", "reserved_for_root_bytes",
                            "unit_size_bytes", "free_percent_of_total",
                            "watermark_bytes_configured", "watermark_percent"),
               ok=False, reason=reason, space_available=space_available,
               space_reason=space_reason, statvfs_state=_READ_FAILED, path=None,
               free_bytes=None, total_bytes=None, used_bytes=None,
               reserved_for_root_bytes=None, unit_size_bytes=None,
               watermark_bytes=None, watermark_bytes_configured=None,
               watermark_percent=None, watermark_source="none",
               below_watermark=None, read_only=None,
               free_percent_of_total=None)


# --- the data-file header (defined now, unused by stage 1) -------------------

def encode_header(geometry, *, epoch_text=None) -> dict:
    """The data file's optional preamble: magic, version, rank, epoch, digest.

    This exists so that turning cross-boot persistence ON later is a parameter
    change (``header_bytes``) and NOT a format break; stage 1 never writes it, and
    the data file's layout is already correct for either choice. ``version`` is in
    the codec so a reader can REFUSE a future writer instead of mis-parsing it -
    the failure mode that turns an upgrade into corrupted KV.
    """
    if not isinstance(geometry, dict) or not geometry.get("epoch"):
        return _fail("a geometry from build_geometry is required")
    epoch = str(geometry["epoch"] if epoch_text is None else epoch_text)
    if len(epoch) > HEADER_EPOCH_CHARS:
        return _fail(f"the epoch is {len(epoch)} chars; the header carries "
                     f"{HEADER_EPOCH_CHARS}")
    digest = str(geometry.get("geometry_digest") or "")
    if len(digest) != GEOMETRY_DIGEST_CHARS or not digest.isalnum():
        return _fail("the geometry digest does not fit the header field")
    body = struct.pack(HEADER_STRUCT, HEADER_MAGIC, FORMAT_VERSION,
                       int(geometry["pp_rank"]), int(geometry["tp_rank"]),
                       len(epoch), digest.encode("ascii"))
    return _ok(units=_units("header_bytes", "epoch_chars", "format_version"),
               header=body + epoch.encode("ascii"),
               header_bytes=len(body) + len(epoch),
               epoch_chars=len(epoch), format_version=FORMAT_VERSION)


def decode_header(blob, *, expect_version: int = FORMAT_VERSION) -> dict:
    """Read a preamble back, REFUSING a version or magic that is not ours.

    A refusal is a clean, classified result and never an exception: a stale or
    foreign first page must be reason enough to treat the file as not ours. The
    rank fields are reported under their real names, because a header claiming
    ``PP0_TP1`` on a file named ``PP1_TP0`` is exactly the mismatch a reader has
    to be able to see.
    """
    if not isinstance(blob, (bytes, bytearray)):
        return _fail("a header must be bytes")
    raw = bytes(blob)
    if len(raw) < HEADER_BYTES:
        return _fail(f"the blob is {len(raw)} bytes, shorter than the "
                     f"{HEADER_BYTES}-byte header", state="too_short")
    magic, version, pp_rank, tp_rank, epoch_chars, digest = struct.unpack(
        HEADER_STRUCT, raw[:HEADER_BYTES])
    if magic != HEADER_MAGIC:
        return _fail("the magic is not ours", state="not_ours", magic=magic)
    if len(raw) < HEADER_BYTES + int(epoch_chars):
        return _fail("the blob stops before the epoch field ends",
                     state="too_short")
    if version != int(expect_version):
        return _fail(f"the header says format version {version}; this build "
                     f"understands {int(expect_version)}",
                     state="version_mismatch", want_version=int(expect_version),
                     found_version=int(version))
    epoch = raw[HEADER_BYTES:HEADER_BYTES + int(epoch_chars)].decode(
        "ascii", "replace")
    return _ok(units=_units("header_bytes", "epoch_chars", "format_version",
                            "pp_rank", "tp_rank"),
               state="ok", format_version=int(version),
               pp_rank=int(pp_rank), tp_rank=int(tp_rank),
               epoch=epoch, epoch_chars=int(epoch_chars),
               geometry_digest=digest.decode("ascii", "replace"),
               header_bytes=HEADER_BYTES + int(epoch_chars))
