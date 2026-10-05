"""The SSD tier's layout layer: fingerprint, names, the commit protocol, capacity.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
The live tier parks evicted GPU KV into pinned host-RAM slots. This is the
*layout* layer for the one-file-per-slot SSD tier beneath it: the integrity
fingerprint that decides what a file's bytes MEAN, the content-addressed file
name and its shard directory, the commit protocol that makes "a file at the
final name" a synonym for "complete", the boot sweep and the LRU plan that keep
the store inside its ceiling, and the free-space gate. It is PURE. No fd is kept
open, no thread, no torch, no vllm; importing it writes no byte to disk except
inside a directory the caller hands in. The worker-side caller is
``disk_spill.py`` (gating, bounds, the queue and its own thread); this module owns
the on-disk contract.

STAGE 1'S LAYOUT WAS A DIFFERENT DESIGN, AND IS GONE
----------------------------------------------------
Stage 1 (commit 22be200) laid the tier out as ONE big file per rank at a fixed
slot stride, with an append-only CRC'd index whose visibility was a committed
watermark, a data-file header codec, ``offset_for_slot``/``slot_for_offset``/
``bounds_for_file``/``slot_is_readable`` offset arithmetic, and an epoch field in
the file name so a boot could tell its own files from a dead boot's. The design
that was measured and chosen instead is one file per (block key, rank): at
11,468,800 B per slot on PP0, a 64 GiB store is ~5,900 files, so per-file costs
almost nothing and buys the two things the big-file layout could not give
cheaply - a file whose NAME is the identity of its bytes, and a crash story with
no tail, no watermark, no sequence and nothing to replay. The consequence,
stated plainly: the index codec, the offset arithmetic, the header codec, the
epoch field and the ``build_paths`` pair-of-paths API are DELETED, along with
their tests. Nothing here is left in "for now" - a codec with no caller is a
liability, not an option.

THE UNIT OF SPILL IS ONE SLOT, AND THE FILE IS THAT SLOT
--------------------------------------------------------
A slot's bytes are addressed exactly as ``blockdump.py`` addresses them -
``data_ptr() + slot * stride(0) * element_size`` per region, ctypes, no CUDA, no
refcount and no LRU perturbation (that contract is proven in ``blockdump``'s
docstring; ``disk_spill.py`` does the reading). So ``total_bytes_per_block``
means:

    bytes for ONE slot ON ONE RANK
      = sum over that rank's region tensors of stride(0) * element_size()

NOT per GPU block, NOT per token, NOT the whole 4-rank world. On the live box a
worker's region set is EXACTLY ONE tensor (``worker.py`` dedupes by ``data_ptr``),
and the slot is 11,468,800 B on the PP0 ranks and 12,902,400 B on the PP1 ranks -
the two pipeline stages genuinely differ, which is why the width is a fingerprint
FIELD and never a constant. ``validate_fingerprint`` refuses a width outside
[1 KiB, 1 GiB] because both directions of the usual unit mistake (a token count,
a pool-wide total) would corrupt live KV rather than merely be wrong.

THE NAME IS THE IDENTITY OF THE BYTES
-------------------------------------
    ``<root>/<key16[0:2]>/kvslot-<key16>-g<group>-<geodigest12>.bin``
``<key16>`` is the first 16 hex characters of the engine's block hash DIGEST and
``g<group>`` is the KV-cache group id the engine appended to that hash - the
4-byte big-endian suffix ``make_block_hash_with_group_id``
(``kv_cache_utils.py:83``) adds and ``get_group_id`` (``key[-4:]``) reads back.
Both are taken from the FULL group-qualified key by ``split_block_key``, the ONE
place that suffix is unpacked; nothing here re-implements the packing.

WHY THE GROUP IS A SEPARATE NAME COMPONENT, AND NOT PART OF THE DIGEST. The
first version of this name truncated the group-qualified hash to its leading 16
hex characters and called the result the identity, documenting the key as "the
block hash INCLUDING the group id". That sentence was true of the INPUT and
false of the IDENTITY: the group sits at the END of the value, so the first 16
hex characters are entirely digest, and two prefix-cacheable groups hashing the
same token chain named the SAME FILE with different bytes. The write side
noticed nothing (the second writer found the file and was counted as a
``duplicate``); only the read side would have known, and it had no way to tell
whose bytes it had been handed. A discriminator that is truncated away is not an
identity, so the group is now a name component a reader can SEE - and a name
without one is ``stale_key_shape``, unlinkable and never matchable by a lookup,
because a file from before this fix cannot be attributed to a group at all.

Hex is lower-cased so one key cannot become two files. The two-character shard
directory keeps a future 100k-file store in cheap directories.
``<geodigest12>`` is a digest of the whole integrity fingerprint - model id and
revision, cache dtype, tensor dtype, the slot width, every group's block size,
the scheduler and hash block sizes, which groups are
prefix-cacheable, the hash algorithm, the ``NONE_HASH`` seed, the rank, the world
size, and the decode/CONTEXT parallel widths. Everything that changes the MEANING
of the bytes is in it, so a file written under a different configuration is not
misread, it is simply never found: un-findability is the safety property, not a
side effect. A file whose name carries a digest this boot did not compute is
``stale_geometry`` to this boot and is unlinkable, never loadable. THE PROPERTY
THAT HAD TO SURVIVE THE GROUP CHANGE IS THIS ONE, UNWEAKENED: a configuration
change produces a different FILENAME, so stale bytes are un-findable rather than
mis-read. ``<geodigest12>`` carries the rank/stride/dtype half of that and
``g<group>`` carries the group half.

A COMMIT IS A RENAME
--------------------
``write_committed_file`` writes ``<name>.part``, ``os.replace``s it into place,
and does NOT fsync anything on the production path (``sync=`` is the opt-in and
the spill tier deliberately does not take it - see the durability posture in
``write_committed_file``'s own docstring).
A file at the final name is ALWAYS complete AS SEEN BY THIS BOOT: no torn tails,
no index, no watermarks, no replay. A duplicate is benign -
identical content under an identical name - and is reported as one, not as an
error. Every step is bounded, and a failed attempt leaves the final name absent,
which is the whole point.

CAPACITY IS A CEILING AND A CLOCK
---------------------------------
``st_mtime`` is the LRU clock. ``st_atime`` is NOT: ``/home`` is mounted
``relatime`` here, so a read does not reliably move it and an LRU built on it
would evict the most-recently-read file. Deleting a file is always safe (the cost
is a recompute), so the plan is: shed files this boot can never read first, then
this boot's own oldest-first, until the used estimate is back under the ceiling.

NEVER A ZERO FOR AN ABSENT MEASUREMENT
--------------------------------------
``statvfs`` failing, a missing directory, an unreadable directory and a genuinely
full disk are four different facts and are reported as four different things:
``available`` plus a ``reason`` and ``None`` counts. A ``0`` would be read as "the
drive is full", which is a lie that changes what the tier does. Bad input is
classified with a ``reason`` and EXCLUDED from totals, never raised, and the
degraded view is built through the same code path as the healthy one so the key
sets cannot drift.

Stdlib only: ``hashlib``, ``os``, ``stat``, ``time``. Every entry point is total -
it returns a classified failure instead of raising.
"""

from __future__ import annotations

import hashlib
import os
import stat as stat_module
import time

# --- format identity ---------------------------------------------------------

SLOT_FILE_PREFIX = "kvslot-"
SUFFIX_DATA = ".bin"
SUFFIX_PART = ".part"              # the in-flight name; never a committed file
FILENAME_TEMPLATE = "kvslot-<key16>-g<group>-<geodigest12>.bin"

KEY_DIGEST_CHARS = 16              # <key16>: hex chars of the block hash DIGEST
GEOMETRY_DIGEST_BYTES = 6          # -> 12 hex characters in the file name
GEOMETRY_DIGEST_CHARS = GEOMETRY_DIGEST_BYTES * 2
SHARD_CHARS = 2                    # <key16[0:2]>, one directory per 256 keys
HEX_DIGITS = frozenset("0123456789abcdef")

# The group id the engine appends: FOUR BYTES big-endian at the END of the block
# hash (`kv_cache_utils.py:83`), so EIGHT hex characters on the wire the engine's
# `BlockHashWithGroupId` renders to. `MIN_KEY_HEX_CHARS` is the shortest key that
# can carry a digest AND that suffix: a shorter value has no group in it, and a
# name built from one would be the pre-fix identity again - the ambiguous one -
# rather than a refusal.
GROUP_SUFFIX_BYTES = 4
GROUP_SUFFIX_HEX_CHARS = GROUP_SUFFIX_BYTES * 2
MIN_KEY_HEX_CHARS = KEY_DIGEST_CHARS + GROUP_SUFFIX_HEX_CHARS
MIN_KEY_BYTES = KEY_DIGEST_CHARS // 2 + GROUP_SUFFIX_BYTES

# The `g<group>` name component. DECIMAL and unpadded on purpose: the group id is
# a small ordinal, decimal is what a human greps for, and leading zeros are
# refused so one identity cannot have two spellings (the same rule that folds a
# key's case). The bound is deliberately far above any real group count - the
# fingerprint caps groups at `MAX_NUM_GROUPS` - because its only job is to refuse
# a suffix that is not a group id at all (a truncated key, a mis-packed digest,
# a hand-edited name) rather than to police an engine.
GROUP_FIELD_PREFIX = "g"
MAX_GROUP_ID = 4096

# The whole key's length is bounded too, so a value that is not a block hash at all
# (a concatenation, a doubled hex, a hand-pasted blob) cannot become a 300-character
# file name. The cap is far above the longest digest the engine's hash algorithms
# render (`sha256` -> 64 hex, `sha512` -> 128 hex, both plus the 8-char group
# suffix), so it refuses nonsense and never a real key.
MAX_KEY_BYTES = 128
MAX_KEY_HEX_CHARS = MAX_KEY_BYTES * 2

MIN_WORLD_SIZE = 1
MAX_WORLD_SIZE = 4096
MAX_NUM_GROUPS = 1024

# A slot is 11,468,800 B (PP0) / 12,902,400 B (PP1) on the live box. These bounds
# are not arbitrary: 1 KiB means "someone handed us a token count or a KiB
# number" and 1 GiB means "someone multiplied by the pool size".
MIN_BLOCK_BYTES = 1024
MAX_BLOCK_BYTES = 1024 * 1024 * 1024
MAX_PATH_CHARS = 4096
MAX_NAME_CHARS = 255               # ext4's per-component limit
MAX_FP_STRING_CHARS = 256
MIN_SIZE_VALUE = 1
MAX_SIZE_VALUE = 1 << 40           # a block SIZE in tokens, not bytes
MIN_WATERMARK_PERCENT = 0.0
MAX_WATERMARK_PERCENT = 99.0

MAX_SWEEP_FILES = 4096
MAX_UNLINK_PLAN = 256
MAX_EVICTION_PLAN = 4096

# Roots that are NOT a disk tier. /tmp on the box is tmpfs: writing the tier there
# spends the RAM the tier exists to relieve, and a reboot deletes the store on a
# filesystem that never hit platter. /dev/shm is the same trap. These literals are
# the REFUSAL list, which is why S108 ("insecure temporary usage") reads them.
FORBIDDEN_ROOT_PREFIXES = ("/tmp", "/dev/shm", "/run", "/var/run")  # noqa: S108

# The journal's tmpfs check reads the kernel's own mount table rather than guessing
# from a name, because the journal is not refused for living on tmpfs (a spill tier
# whose journal is in /tmp still works - it just forgets its map at reboot) and a
# warning that cried wolf on a bind-mounted `/tmp` would be ignored the way every
# other noisy warning is. Bounded read, never a walk, never a raise.
PROC_MOUNTS_PATH = "/proc/mounts"
TMPFS_FSTYPES = ("tmpfs", "ramfs")
MAX_MOUNTS_BYTES = 64 * 1024

# Classifications of a name, and of a file in a sweep. Closed vocabularies: a
# reader must be able to enumerate them from the payload.
NAME_CURRENT = "current"
NAME_STALE_GEOMETRY = "stale_geometry"
NAME_STALE_KEY_SHAPE = "stale_key_shape"
NAME_PEER_RANK = "peer_rank"
NAME_MALFORMED = "malformed"
NAME_NOTOURS = "not_ours"
NAME_REFUSED = "refused"
NAME_CLASSIFICATIONS = (NAME_CURRENT, NAME_STALE_GEOMETRY, NAME_STALE_KEY_SHAPE,
                        NAME_PEER_RANK, NAME_MALFORMED, NAME_NOTOURS,
                        NAME_REFUSED)

# A malformed name is REPORTED and never unlinked: our prefix plus our suffix is
# not proof the file is OURS, and the cost of deleting someone's
# `kvslot-backup-notes.bin` is infinite next to the cost of leaving it. A
# peer-rank file is not stale either: it is another rank's LIVE data, sitting in
# the same store, and only capacity eviction may touch it (see the plan order).
#
# `stale_key_shape` IS ours and IS unlinkable, and that is deliberately the
# opposite of the malformed rule: a 2-field `kvslot-<key16>-<geodigest12>.bin` is
# the identity this plugin emitted before the group fix, so we know who wrote it
# and we know what it cannot prove - that its bytes belong to the group its
# `key16` is now associated with. Its digest can match this boot, so nothing but
# this classification keeps it out of a load path; and leaving it in place is what
# makes a store 9 GiB of unattributable capacity the ceiling logic cannot even
# see, let alone shed.
UNLINKABLE_CLASSIFICATIONS = (NAME_STALE_GEOMETRY, NAME_STALE_KEY_SHAPE)
EVICTABLE_CLASSIFICATIONS = (NAME_STALE_GEOMETRY, NAME_STALE_KEY_SHAPE,
                             NAME_CURRENT, NAME_PEER_RANK)

# Directory states. A missing directory and an unreadable one are different facts
# and are never collapsed into "no files".
_OK_DIR = "ok"
_MISSING = "missing"
_NOT_A_DIRECTORY = "not_a_directory"
_UNREADABLE = "unreadable"
_READ_FAILED = "read_failed"

# One table, so no two payloads can describe the same field two different ways.
UNITS_BY_NAME = {
    "accepted": "requests",
    "arm_default_seconds": "seconds",
    "arm_seconds": "seconds",
    "arm_file_boot_refusals": "refusals",
    "bad_call": "refusals",
    "bad_digest": "refusals",
    "bad_key": "refusals",
    "bad_key_text": "refusals",
    "block_id": "blocks",
    "bytes_on_disk": "bytes",
    "bytes_per_file": "bytes",
    "bytes_written": "bytes",
    "ceiling_bytes": "bytes",
    # The capacity walk's own honesty witness (`disk_spill._walk_store`, and the
    # `ceiling_measurement` block of the spill section): what was counted, what was
    # stat-ed, and what the sample was scaled to.
    "ceiling_checks": "passes",
    "cp_world_size": "ranks",
    "current": "files",
    "dcp_world_size": "ranks",
    "default_seconds": "seconds",
    "digest_group_overflow": "events",
    "digest_groups_cap": "key16s",
    "digest_groups_dropped": "key16s",
    "digest_groups_tracked": "key16s",
    "disarmed": "refusals",
    "dropped": "jobs",
    "drops": "jobs",
    "duplicates": "slots",
    "elapsed_seconds": "seconds",
    "entries": "files",
    "entries_examined": "entries",
    "errors": "operations",
    "estimated_bytes": "bytes",
    "estimated_total_bytes": "bytes",
    "events": "events",
    "evict_bytes": "bytes",
    "evict_target_fraction": "fraction",
    "evicted_bytes": "bytes",
    "evictions": "files",
    "expired": "refusals",
    "expires_at_wallclock": "epoch_seconds",
    "files": "files",
    "files_free": "files",
    "fingerprint": "fingerprints",
    "fingerprint_incomplete": "refusals",
    "fingerprint_invalid": "refusals",
    "fingerprint_mismatch": "refusals",
    "fingerprint_missing": "refusals",
    "fragment_bytes": "bytes",
    "free_bytes": "bytes",
    "free_percent_of_total": "percent",
    "freed_bytes": "bytes",
    # The tmpfs verdict for a directory (`tmpfs_verdict`): a bool/str triple, not
    # numbers, but declared so no payload can carry an `"unknown"` unit.
    "fstype": "filesystem type",
    "mount_point": "paths",
    "on_tmpfs": "verdict",
    "journal_on_tmpfs": "verdict",
    "journal_tmpfs_warnings": "warnings",
    "geometry_chars": "chars",
    "group_collisions": "events",
    "group_id": "groups",
    "group_id_a": "groups",
    "group_id_b": "groups",
    "hash_block_size": "tokens",
    "headroom_bytes": "bytes",
    "journal_cap_bytes": "bytes",
    "journal_directory": "paths",
    "journal_directory_attempts": "attempts",
    "journal_directory_errors": "failures",
    "journal_fields": "fields",
    "journal_failures": "writes",
    "journal_lines": "lines",
    "journal_records": "files",
    "journal_removals": "files",
    "journal_seq_seed": "lines",
    "journal_writes": "lines",
    "journal_write_errors": "writes",
    "key_chars": "chars",
    "key_count_mismatch": "refusals",
    "length_bytes": "bytes",
    "lru_files": "files",
    "malformed": "files",
    "mean_files_per_shard": "files",
    "max_bytes_per_file": "bytes",
    "max_files_per_window": "files",
    "max_queue_depth": "jobs",
    "max_seconds": "seconds",
    "max_shard_dirs": "directories",
    "max_walk_entries": "entries",
    "max_walk_files": "files",
    "measured_bytes": "bytes",
    "min_free_bytes": "bytes",
    "mtime_ns": "nanoseconds",
    "name_chars": "chars",
    "no_channel": "refusals",
    "no_free_space": "refusals",
    "no_key_for_slot": "refusals",
    "no_region_tensors": "refusals",
    "no_store_event": "refusals",
    "no_store_slots": "refusals",
    "not_armed": "refusals",
    "not_configured": "refusals",
    "not_ours": "files",
    "num_groups": "groups",
    "over_rate_window": "refusals",
    "over_slot_cap": "refusals",
    "over_tick_cap": "refusals",
    "path_chars": "chars",
    "peer_rank": "files",
    "planned": "files",
    "planned_bytes": "bytes",
    "queue_depth": "jobs",
    "queue_depth_peak": "jobs",
    "queue_full": "refusals",
    "queue_max": "jobs",
    "queued": "jobs",
    "read_bytes": "bytes",
    "read_failed": "refusals",
    "read_seconds": "seconds",
    "refused": "files",
    "region_not_contiguous": "refusals",
    "region_width_mismatch": "refusals",
    "requests": "requests",
    "reserve_bytes": "bytes",
    "reserved_for_root_bytes": "bytes",
    "retained_events": "events",
    "sibling_probe_failures": "probes",
    "scanned": "entries",
    "sampled_files": "files",
    "seed_walks": "walks",
    "scheduler_block_size": "tokens",
    "seconds_left": "seconds",
    "shard_chars": "chars",
    "shard_dirs_counted": "directories",
    "shard_dirs_total": "directories",
    "shard_dirs_walked": "directories",
    "size_bytes": "bytes",
    "skipped": "files",
    "skipped_files": "files",
    "slot": "blocks",
    "slot_out_of_range": "refusals",
    "slots": "slots",
    "spill_window_seconds": "seconds",
    "spilled": "slots",
    "stale_first": "files",
    "stale_geometry": "files",
    "stale_key_shape": "files",
    "target_bytes": "bytes",
    "thread_alive": "boolean",
    "thread_started": "threads",
    "total_bytes": "bytes",
    "total_bytes_per_block": "bytes",
    "total_files": "files",
    "unit_size_bytes": "bytes",
    "unlinked": "files",
    "unretained_event": "refusals",
    "used_bytes": "bytes",
    "watermark_bytes": "bytes",
    "watermark_bytes_configured": "bytes",
    "watermark_percent": "percent",
    "window_files": "files",
    "write_errors": "writes",
    "world_size": "ranks",
    "write_bytes": "bytes",
    "write_failed": "refusals",
    "write_seconds": "seconds",
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

    An unrecognised name becomes ``"unknown"``: a bug in this module that a test
    can see, rather than an absent key or a guessed unit.
    """
    return {name: UNITS_BY_NAME.get(name, "unknown") for name in names}


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- the integrity fingerprint -----------------------------------------------
# EVERY field here changes the MEANING of a slot's bytes. The digest is computed
# over all of them, so a file written by another model, another dtype, another
# group layout, another rank, another world size or another hash seed is not
# misread by this boot - it is never named by this boot, and therefore never
# found. A missing field is a REFUSAL to write: an under-specified fingerprint is
# a future corruption traded for a small convenience today.

# The engine's `NONE_HASH` seed: the value a prefix hash chain starts from. Two
# boots that disagree about it compute different hashes for identical content, so
# a file named under one seed must never be considered under the other.
FP_SEED_FIELD = "none_hash_seed"
FINGERPRINT_FIELDS = (
    ("model_id", "string"),
    ("model_revision", "string"),
    ("cache_dtype", "string"),
    ("dtype", "string"),
    ("total_bytes_per_block", "bytes"),
    ("group_block_sizes", "token_list"),
    ("scheduler_block_size", "tokens"),
    ("hash_block_size", "tokens"),
    ("prefix_cacheable_group_ids", "group_set"),
    ("prefix_caching_hash_algo", "string"),
    (FP_SEED_FIELD, "seed"),
    ("rank", "rank"),
    ("world_size", "ranks"),
    ("dcp_world_size", "ranks"),
    ("cp_world_size", "ranks"),
)


def _validate_string(name: str, value, *,
                     allow_path_separator: bool = False) -> dict:
    """A fingerprint string: printable, bounded, never a NUL.

    `allow_path_separator` exists because of a measured outage, and the reasoning
    is the whole point of the parameter. A fingerprint field is NEVER a path
    component: the file name is built from `key16`, its 2-char shard, the
    `g<group>` component and the digest (`build_slot_path`), every one of them
    hexadecimal or decimal by validation, and the fingerprint itself only ever
    enters a SHA digest as digested text. The values that DO reach a path are
    validated elsewhere and stay strict (`split_block_key`,
    `validate_geometry_digest`, `validate_root`).

    So refusing a separator here could not protect a path - it could only refuse
    the identity itself, and that is exactly what it did: the box boots from a
    local model directory (`vllm serve /home/user/<model>`), so
    `model_config.model` - the first candidate for `model_id` - is an absolute
    path, and every step on every rank was refused as `fingerprint_incomplete`
    with zero files written. A rule that silently disables the feature it guards
    is not a safety rule. Rendering the identity honestly also matters: taking a
    basename to dodge the ban would fold two different model directories into one
    digest, which is the one failure this layer exists to prevent.

    The rule that stays is the one with teeth: printable text only (a control
    character in a digested field is a writer bug), a length cap, and never a NUL
    - and a rank keeps the strict form regardless (see the `rank` kind).
    """
    if not isinstance(value, str):
        return _fail(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        return _fail(f"{name} is empty")
    if len(text) > MAX_FP_STRING_CHARS:
        return _fail(f"{name} is {len(text)} chars; the cap is "
                     f"{MAX_FP_STRING_CHARS}")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        return _fail(f"{name} contains a control character")
    if "\x00" in text:
        return _fail(f"{name} contains a NUL byte")
    if not allow_path_separator and ("/" in text or "\\" in text):
        return _fail(f"{name} contains a path separator")
    return {"ok": True, "reason": None, "value": text}


def _parse_rank(text: str) -> dict:
    """``PP<n>_TP<m>`` exactly, or a reason. No fuzziness: ``PP0TP1`` is not it."""
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


def _validate_int(name: str, value, *, low: int, high: int,
                  unit: str) -> dict:
    if not _is_plain_int(value):
        return _fail(f"{name} must be an int, got {type(value).__name__} "
                     f"(unit: {unit})")
    if not low <= value <= high:
        return _fail(f"{name} {value} is outside [{low}, {high}] (unit: {unit})")
    return {"ok": True, "reason": None, "value": value}


def _validate_token_list(name: str, value, *, allow_empty: bool) -> dict:
    if not isinstance(value, (list, tuple)):
        return _fail(f"{name} must be a list of ints, got "
                     f"{type(value).__name__}")
    items = list(value)
    if not items and not allow_empty:
        return _fail(f"{name} is empty: a boot with no groups has no slot to "
                     "write, so there is nothing to fingerprint")
    if len(items) > MAX_NUM_GROUPS:
        return _fail(f"{name} has {len(items)} entries; the cap is "
                     f"{MAX_NUM_GROUPS} groups")
    for item in items:
        if not _is_plain_int(item) or item < 0:
            return _fail(f"{name} contains {item!r}, which is not a non-negative "
                         "int")
    return {"ok": True, "reason": None, "value": items}


def _validate_seed(name: str, value) -> dict:
    """The NONE_HASH seed, as the engine reports it: an int, hex, or opaque text.

    Three shapes, because the FORK has all three. `_validate_seed` used to demand
    hex, and the live fork's own `DEFAULT_NONE_HASH_SEED` is `"vllm-none-hash"`
    (`v1/core/kv_cache_utils.py:121`) - the value a cryptographic hash function
    gets when `PYTHONHASHSEED` is unset, which is the normal production boot. A
    hex-only rule therefore refused a REAL engine value, and a worker that cannot
    render the seed cannot build a fingerprint, cannot name a file, and spills
    nothing forever. Refusing a value the engine actually uses is not safety, it is
    a broken feature - so an opaque string is accepted, and what stays refused is
    anything that cannot be rendered into the digest at all: empty, absurdly long,
    a NUL, a control character, a float, a bool, None.

    The canonical form is always prefixed - `hex:<digits>` for a hex-shaped value
    (whichever hex spelling arrived, and for an int), `int:` accepted as an
    alternative spelling of the same digits, and `str:<text>` for anything else
    kept VERBATIM, case included. Two consequences that matter: validation is
    IDEMPOTENT (`disk_spill` re-validates on every spill, so a canonical value must
    not be refused by the check that produced it), and an opaque seed's case is
    part of the identity, because a secret seed's cases are different secrets.

    WHY THE SEED IS IN THE FINGERPRINT AT ALL: with a non-cryptographic hash
    function and no `PYTHONHASHSEED`, `get_none_hash_seed()` returns fresh random
    bytes PER PROCESS (`kv_cache_utils.py:139-143`) and the fork logs that block
    hashes are then not reproducible across processes. A boot whose seed changed
    mid-life must not find yesterday's files, so the seed moves the digest.
    """
    if _is_plain_int(value):
        if value < 0:
            return _fail(f"{name} is negative")
        return {"ok": True, "reason": None, "value": f"hex:{value:x}"}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return _fail(f"{name} is empty")
        if len(text) > MAX_FP_STRING_CHARS:
            return _fail(f"{name} is {len(text)} chars; the cap is "
                         f"{MAX_FP_STRING_CHARS}")
        if any(ord(char) < 32 or ord(char) == 127 for char in text):
            return _fail(f"{name} contains a control character")
        if "\x00" in text:
            return _fail(f"{name} contains a NUL byte")
        lowered = text.lower()
        if lowered.startswith("str:"):
            return {"ok": True, "reason": None, "value": text}
        for kind in ("int:", "hex:"):
            if lowered.startswith(kind):
                digits = text[len(kind):]
                if not digits or len(digits) > 64:
                    return _fail(f"{name} {value!r} is not a canonical seed")
                if kind == "int:":
                    if not digits.isdigit():
                        return _fail(f"{name} {value!r} is not a canonical seed")
                    return {"ok": True, "reason": None,
                            "value": f"hex:{int(digits):x}"}
                if any(char not in HEX_DIGITS for char in digits.lower()):
                    return _fail(f"{name} {value!r} is not a canonical seed")
                return {"ok": True, "reason": None, "value": f"hex:{digits.lower()}"}
        hexish = lowered.removeprefix("0x")
        if len(hexish) <= 64 and hexish and all(
                char in HEX_DIGITS for char in hexish):
            return {"ok": True, "reason": None, "value": f"hex:{hexish}"}
        # Not hex-shaped, so it is an opaque seed (the fork's `vllm-none-hash`, or
        # an operator's own). It is still a fact about this boot, and a fact we
        # refuse to render is a fingerprint we cannot build.
        return {"ok": True, "reason": None, "value": f"str:{text}"}
    return _fail(f"{name} must be an int or a string, got "
                 f"{type(value).__name__}")


def validate_fingerprint(fingerprint) -> dict:
    """Validate a whole fingerprint. MISSING OR UNKNOWN fields are refusals.

    Returns a NORMALIZED copy (strings stripped, the seed rendered canonically,
    the group-id set sorted) so two boots that read the same facts through two
    different call sites compute ONE digest. ``group_block_sizes`` is index-
    ORDERED and therefore kept in order; ``prefix_cacheable_group_ids`` is a set
    and is sorted. Unknown keys are refused rather than ignored: an ignored field
    is a field that silently stops being part of the identity, which is how a
    future change to this list becomes a silent mis-load instead of a rename.
    """
    if not isinstance(fingerprint, dict):
        return _fail(f"a fingerprint must be a dict, got "
                     f"{type(fingerprint).__name__}")
    missing = [name for name, _kind in FINGERPRINT_FIELDS
               if name not in fingerprint or fingerprint[name] is None]
    if missing:
        return _fail("the fingerprint is missing required field(s) "
                     f"{missing}: writing a file whose identity is "
                     "under-specified is a future corruption, not a convenience",
                     missing_fields=missing)
    unknown = sorted(set(fingerprint) - {name for name, _kind
                                         in FINGERPRINT_FIELDS})
    if unknown:
        return _fail(f"the fingerprint carries unknown field(s) {unknown}: "
                     "refusing to name a file after fields this build does not "
                     "understand", unknown_fields=unknown)

    normalized: dict = {}
    for name, kind in FINGERPRINT_FIELDS:
        value = fingerprint[name]
        if kind == "string":
            # A digested identity field, never a path component: see
            # `_validate_string` for why a local model PATH is legal here.
            check = _validate_string(name, value, allow_path_separator=True)
        elif kind == "rank":
            text = _validate_string(name, value)
            if not text["ok"]:
                check = text
            else:
                parsed = _parse_rank(text["value"])
                check = ({"ok": True, "reason": None, "value": text["value"]}
                         if parsed["ok"] else parsed)
        elif kind == "bytes":
            check = _validate_int(name, value, low=MIN_BLOCK_BYTES,
                                  high=MAX_BLOCK_BYTES,
                                  unit="bytes for ONE slot ON ONE RANK")
        elif kind == "tokens":
            check = _validate_int(name, value, low=MIN_SIZE_VALUE,
                                  high=MAX_SIZE_VALUE, unit="tokens")
        elif kind == "ranks":
            check = _validate_int(name, value, low=MIN_WORLD_SIZE,
                                  high=MAX_WORLD_SIZE, unit="ranks")
        elif kind == "token_list":
            check = _validate_token_list(name, value, allow_empty=False)
        elif kind == "group_set":
            check = _validate_token_list(name, value, allow_empty=True)
        else:                                    # kind == "seed"
            check = _validate_seed(name, value)
        if not check["ok"]:
            return _fail(str(check["reason"]))
        if kind == "group_set":
            normalized[name] = sorted(set(check["value"]))
        elif kind == "token_list":
            normalized[name] = list(check["value"])
        else:
            normalized[name] = check["value"]
    # Cross-field: the group id is packed INTO the block hash, so a fingerprint
    # that claims a group is prefix-cacheable but has no such group is a
    # fingerprint that does not describe this engine.
    groups = len(normalized["group_block_sizes"])
    out_of_range = [g for g in normalized["prefix_cacheable_group_ids"]
                    if g >= groups]
    if out_of_range:
        return _fail(f"prefix_cacheable_group_ids {out_of_range} name a group "
                     f"outside the {groups} group(s) group_block_sizes describes")
    return _ok(units=_units("total_bytes_per_block", "scheduler_block_size",
                            "hash_block_size", "num_groups", "world_size",
                            "dcp_world_size", "cp_world_size"),
               fingerprint=normalized,
               num_groups=groups,
               total_bytes_per_block=normalized["total_bytes_per_block"])


def canonical_fingerprint(fingerprint) -> str:
    """The digest's input, rendered deterministically and injectively.

    Every field is rendered as ``name=kind:len:value``, so no two different field
    sets can concatenate into the same string (the length prefix is what makes
    that true for strings containing ``:`` or ``;``). Field ORDER is the
    declaration order above, not insertion order: two boots that read the same
    facts through different call sites must hash the same bytes.
    """
    parts = ["kvtier-disk-fp-v2"]
    for name, kind in FINGERPRINT_FIELDS:
        value = fingerprint[name]
        if isinstance(value, (list, tuple)):
            rendered = ",".join(str(item) for item in value)
            parts.append(f"{name}=list:{len(rendered)}:{rendered}")
        else:
            text = str(value)
            parts.append(f"{name}={kind}:{len(text)}:{text}")
    return ";".join(parts)


def compute_geometry_digest(fingerprint) -> str:
    """The 12-hex-char identity of a VALIDATED fingerprint's MEANING.

    Takes the normalized dict `validate_fingerprint` returns; it does not
    re-validate, because a caller that skips validation is a bug in a caller, and
    `fingerprint_digest` is the door that validates. Digest length is
    ``GEOMETRY_DIGEST_CHARS`` (12 hex, 48 bits): far beyond what the number of
    live files needs, and short enough to stay readable in a file name.
    """
    material = canonical_fingerprint(fingerprint)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[
        :GEOMETRY_DIGEST_CHARS]


def fingerprint_digest(fingerprint) -> dict:
    """Validate THEN digest. The one way a caller should get a digest."""
    validated = validate_fingerprint(fingerprint)
    if not validated["ok"]:
        return _fail(str(validated["reason"]))
    normalized = validated["fingerprint"]
    return _ok(units=_units("geometry_chars", "num_groups",
                            "total_bytes_per_block", "scheduler_block_size",
                            "hash_block_size", "world_size", "dcp_world_size",
                            "cp_world_size"),
               geometry_digest=compute_geometry_digest(normalized),
               fingerprint=normalized,
               num_groups=validated["num_groups"],
               total_bytes_per_block=normalized["total_bytes_per_block"],
               scheduler_block_size=normalized["scheduler_block_size"],
               hash_block_size=normalized["hash_block_size"],
               world_size=normalized["world_size"],
               dcp_world_size=normalized["dcp_world_size"],
               cp_world_size=normalized["cp_world_size"],
               geometry_chars=GEOMETRY_DIGEST_CHARS)


def fingerprint_section(fingerprint) -> dict:
    """The payload view of a fingerprint. An absent one says so, with nulls."""
    names = [name for name, _kind in FINGERPRINT_FIELDS]
    units = _units("total_bytes_per_block", "scheduler_block_size",
                   "hash_block_size", "world_size", "dcp_world_size",
                   "cp_world_size")
    if not isinstance(fingerprint, dict):
        out = {name: None for name in names}
        out.update({"present": False, "reason": "no fingerprint was handed in",
                    "missing_fields": list(names), "units": units})
        return out
    out = {name: fingerprint.get(name) for name in names}
    out.update({"present": True,
                "reason": None,
                "missing_fields": [name for name in names
                                   if fingerprint.get(name) is None],
                "units": units})
    return out


# --- names -------------------------------------------------------------------

def validate_key16(key) -> dict:
    """The DIGEST half of an identity: exactly 16 hex characters, lower-cased.

    THIS IS NOT THE ENGINE'S KEY. `key16` is the leading 16 hex characters of the
    block hash DIGEST, i.e. `split_block_key(...)["key16"]`; the group id the
    engine appends is NOT in it (it lives at the END of the engine's value, which
    is exactly why the first version of this layer, which truncated to these 16
    characters and stopped there, gave two groups the same file name). Anything
    that names a file, journals a file or looks one up must go through
    `split_block_key` and carry the group as well.

    Upper case is NORMALISED, not refused: ext4 is case-sensitive, so accepting
    both would let one key own two files, and a key that is findable under two
    names is a cache that halves itself. Wrong length or non-hex is refused - a
    truncated key is a DIFFERENT identity, not a shorter spelling of the same one.
    """
    if not isinstance(key, str):
        return _fail(f"a key must be a string of {KEY_DIGEST_CHARS} hex chars, "
                     f"got {type(key).__name__}")
    text = key.strip().lower()
    if len(text) != KEY_DIGEST_CHARS:
        return _fail(f"the key is {len(text)} chars; it must be exactly "
                     f"{KEY_DIGEST_CHARS} hex characters of the block hash")
    if any(char not in HEX_DIGITS for char in text):
        return _fail("the key is not hexadecimal")
    return {"ok": True, "reason": None, "value": text}


def split_block_key(key) -> dict:
    """Unpack the engine's group-qualified block hash into (digest, group id).

    THE ONE PLACE THE 4-BYTE GROUP SUFFIX IS READ. The engine builds
    `BlockHashWithGroupId` by appending `group_id.to_bytes(4, "big")` to the block
    hash (`kv_cache_utils.py:83`), and reads it back as `key[:-4]` /
    `int.from_bytes(key[-4:], "big")` (`get_block_hash` / `get_group_id`). Those
    are the two accessors used here, on the raw BYTES when a caller hands bytes and
    on the same eight hex characters when a caller hands the hex the engine puts on
    the wire. Nothing here re-packs, re-hashes or re-derives a key: a plugin that
    invented its own packing would disagree with the engine about which bytes are
    the group, which is the same bug wearing a different hat.

    REFUSALS, never exceptions, for: a value that is not bytes or str; a value too
    SHORT to carry the suffix (`MIN_KEY_BYTES` / `MIN_KEY_HEX_CHARS` - this is the
    pre-fix 16-hex key, and accepting it is what put two groups in one file);
    non-hex text; and a group id outside `[0, MAX_GROUP_ID]`, which cannot be a
    group ordinal at all. A refusal carries no numbers, so a caller cannot spill a
    half-resolved identity by accident.
    """
    if isinstance(key, bytes):
        if len(key) < MIN_KEY_BYTES:
            return _fail(f"the key is {len(key)} bytes; a group-qualified block "
                         f"hash needs at least {MIN_KEY_BYTES} "
                         f"({KEY_DIGEST_CHARS // 2} digest + "
                         f"{GROUP_SUFFIX_BYTES} group bytes), so this one carries "
                         "no group id and cannot name a file")
        if len(key) > MAX_KEY_BYTES:
            return _fail(f"the key is {len(key)} bytes; the cap is "
                         f"{MAX_KEY_BYTES}, so this is not a block hash and cannot "
                         "name a file")
        digest_hex = key[:-GROUP_SUFFIX_BYTES].hex()
        group_id = int.from_bytes(key[-GROUP_SUFFIX_BYTES:], "big",
                                  signed=False)
    elif isinstance(key, str):
        text = key.strip().lower()
        if len(text) < MIN_KEY_HEX_CHARS:
            return _fail(f"the key is {len(text)} hex chars; a group-qualified "
                         f"block hash needs at least {MIN_KEY_HEX_CHARS} "
                         f"({KEY_DIGEST_CHARS} digest + "
                         f"{GROUP_SUFFIX_HEX_CHARS} group chars), so this one "
                         "carries no group id and cannot name a file")
        if len(text) > MAX_KEY_HEX_CHARS:
            return _fail(f"the key is {len(text)} hex chars; the cap is "
                         f"{MAX_KEY_HEX_CHARS}, so this is not a block hash and "
                         "cannot name a file")
        digest_hex = text[:-GROUP_SUFFIX_HEX_CHARS]
        group_hex = text[-GROUP_SUFFIX_HEX_CHARS:]
        if any(char not in HEX_DIGITS for char in text):
            return _fail("the key is not hexadecimal")
        group_id = int(group_hex, 16)
    else:
        return _fail("a key must be the engine's block hash bytes or their hex, "
                     f"got {type(key).__name__}")
    if group_id > MAX_GROUP_ID:
        return _fail(f"the group id {group_id} is outside [0, {MAX_GROUP_ID}]: "
                     "that is not a KV-cache group ordinal, so the suffix this key "
                     "ends with is not a group id and the name it would produce "
                     "means nothing")
    key16 = digest_hex[:KEY_DIGEST_CHARS]
    return _ok(units=_units("key_chars", "group_id", "shard_chars"),
               key16=key16, group_id=group_id,
               group_field=f"{GROUP_FIELD_PREFIX}{group_id}",
               digest_hex=digest_hex,
               key_hex=digest_hex + f"{group_id:08x}",
               shard=key16[:SHARD_CHARS], key_chars=len(key16))


def validate_root(path) -> dict:
    """Is this a filesystem root the tier may write to? Creates nothing.

    ``/tmp``, ``/dev/shm`` and ``/run`` are refused by name even if they exist and
    are writable, because on the box they are tmpfs: a "disk" tier there spends
    the RAM the tier exists to relieve and evaporates on reboot. The rest is the
    ordinary shape check - a str, non-empty, no NUL, not absurdly long.
    """
    if not isinstance(path, (str, os.PathLike)):
        return _fail(f"a root must be a path, got {type(path).__name__}")
    try:
        text = os.fspath(path)
    except TypeError as exc:
        return _fail(f"the root is not a filesystem path: {type(exc).__name__}")
    if not isinstance(text, str):
        return _fail("the root must be a str path (bytes paths are refused)")
    if not text or "\x00" in text:
        return _fail("the root is empty or contains a NUL byte")
    if len(text) > MAX_PATH_CHARS:
        return _fail(f"the root is {len(text)} chars; the cap is {MAX_PATH_CHARS}")
    absolute = os.path.abspath(text)
    for forbidden in FORBIDDEN_ROOT_PREFIXES:
        if absolute == forbidden or absolute.startswith(forbidden + os.sep):
            return _fail(f"{absolute} is a tmpfs-backed runtime directory: the "
                         "tier would spend the RAM it exists to relieve, so it "
                         "is refused even when writable")
    if absolute == os.sep:
        return _fail("the root is the filesystem root: refusing")
    return {"ok": True, "reason": None, "value": text, "absolute": absolute}


def _mount_table() -> list:
    """`[(mount_point, fstype)]` from the kernel's own table. Bounded; [] on any
    failure. A missing /proc is not an error, it is an absence of knowledge."""
    try:
        fd = os.open(PROC_MOUNTS_PATH, os.O_RDONLY)
    except OSError:
        return []
    try:
        raw = b""
        while len(raw) < MAX_MOUNTS_BYTES:
            chunk = os.read(fd, MAX_MOUNTS_BYTES - len(raw))
            if not chunk:
                break
            raw += chunk
    except OSError:
        return []
    finally:
        try:  # noqa: SIM105 - one descriptor, no contextlib import for it
            os.close(fd)
        except OSError:
            pass
    rows = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        fields = line.split(" ")
        if len(fields) < 3:
            continue
        # /proc/mounts escapes a space in a mount point as octal `\040`.
        rows.append((fields[1].replace("\\040", " "), fields[2]))
    return rows


def tmpfs_verdict(path) -> dict:
    """Is ``path`` on a filesystem that dies at reboot? NEVER raises, never refuses.

    THE SAME QUESTION ``validate_root`` ASKS THE STORE ROOT, for a path that is not
    refused for the answer. The store root is refused by NAME (``/tmp``,
    ``/dev/shm``, ``/run``, ``/var/run``) because a "disk" tier there spends the RAM
    the tier exists to relieve. The spill JOURNAL is a different case: a journal in
    ``/tmp`` works perfectly and simply forgets its map at reboot, so the answer
    drives a WARNING and a reported field, never a refusal - which is why it is
    worth reading the kernel's mount table instead of trusting a name (a bind
    mounted ``/tmp`` on a real filesystem must not be warned about, or the warning
    gets ignored the way every noisy one is).

    ``on_tmpfs`` is ``True`` / ``False``, or ``None`` when nothing could be learned
    (no readable mount table AND no name match): never a confident guess and never a
    0-for-absence. ``method`` says how the answer was reached - ``mounts`` (the
    longest mount-point prefix under ``path``, and its filesystem type), ``name``
    (one of the forbidden runtime prefixes, believed because the table was
    unreadable), or ``unknown``. A path that does not exist is answered by its
    Mount table alone, which is correct: the question is about the FILESYSTEM, not
    about the directory.
    """
    if not isinstance(path, (str, os.PathLike)):
        return _fail(f"a directory must be a path, got {type(path).__name__}",
                     on_tmpfs=None, method="unknown", fstype=None,
                     mount_point=None)
    try:
        text = os.fspath(path)
    except TypeError as exc:
        return _fail(f"the directory is not a filesystem path: "
                     f"{type(exc).__name__}", on_tmpfs=None, method="unknown",
                     fstype=None, mount_point=None)
    if not isinstance(text, str) or not text or "\x00" in text:
        return _fail("the directory must be a non-empty str without a NUL byte",
                     on_tmpfs=None, method="unknown", fstype=None,
                     mount_point=None)
    absolute = os.path.normpath(os.path.abspath(text))
    name_match = None
    for forbidden in FORBIDDEN_ROOT_PREFIXES:
        if absolute == forbidden or absolute.startswith(forbidden + os.sep):
            name_match = forbidden
            break
    # Longest matching mount point wins: `/home/user/kv` under a `/home` mount and a
    # `/home/user` mount of a different type is answered by the more specific one.
    best_point, best_type = None, None
    for point, fstype in _mount_table():
        if point == absolute or absolute.startswith(point.rstrip(os.sep) + os.sep):
            if best_point is None or len(point) > len(best_point):
                best_point, best_type = point, fstype
        elif absolute == "/" and point == "/":
            best_point, best_type = point, fstype
    if best_type is not None:
        return _ok(on_tmpfs=best_type in TMPFS_FSTYPES, method="mounts",
                   fstype=best_type, mount_point=best_point,
                   absolute=absolute, name_match=name_match,
                   units=_units("on_tmpfs", "fstype", "mount_point"))
    if name_match is not None:
        return _ok(on_tmpfs=True, method="name", fstype=None,
                   mount_point=name_match, absolute=absolute,
                   name_match=name_match,
                   units=_units("on_tmpfs", "fstype", "mount_point"))
    return _ok(on_tmpfs=None, method="unknown", fstype=None, mount_point=None,
               absolute=absolute, name_match=None,
               units=_units("on_tmpfs", "fstype", "mount_point"))


def validate_geometry_digest(digest) -> dict:
    """Is this a 12-hex-char geometry digest? Validates and lower-cases; derives
    nothing, ever.

    THE ONE RULE THIS FUNCTION EXISTS TO KEEP: a digest that arrives from another
    process (the scheduler's `store_fingerprint`, on the wire to a worker) is
    VALIDATED AND USED VERBATIM. Re-deriving it worker-side would name files this
    boot can never find, because the digest covers scheduler-only fields the
    worker does not have - the hash seed above all. So the only legal operations
    here are the shape check and the case normalisation `validate_key16` already
    applies to a key (ext4 is case-sensitive: one digest owning two file names
    halves the store).
    """
    if not isinstance(digest, str):
        return _fail(f"a geometry digest must be a string of "
                     f"{GEOMETRY_DIGEST_CHARS} hex chars, got "
                     f"{type(digest).__name__}")
    text = digest.strip().lower()
    if len(text) != GEOMETRY_DIGEST_CHARS:
        return _fail(f"the geometry digest is {len(text)} chars; it must be "
                     f"exactly {GEOMETRY_DIGEST_CHARS} hex characters of the "
                     "config fingerprint")
    if any(char not in HEX_DIGITS for char in text):
        return _fail("the geometry digest is not hexadecimal")
    return {"ok": True, "reason": None, "value": text}


def build_slot_name(key, geometry_digest) -> dict:
    """``kvslot-<key16>-g<group>-<geodigest12>.bin``, and nothing else.

    `key` is the FULL group-qualified block hash (the engine's bytes, or their
    hex); `split_block_key` is what turns it into the digest and the `g<group>`
    component, so a caller cannot pass a half-key and get the pre-fix ambiguous
    name. Every part is validated here, so a name built by this function is a name
    `parse_path` accepts and a sweep can classify. The digest is NOT re-derived
    from a fingerprint: the caller already has it (or is testing what a different
    digest produces, which is the whole point of the scheme).
    """
    split = split_block_key(key)
    if not split["ok"]:
        return _fail(str(split["reason"]))
    key16 = split["key16"]
    group_field = split["group_field"]
    checked_digest = validate_geometry_digest(geometry_digest)
    if not checked_digest["ok"]:
        return _fail(f"geometry_digest {geometry_digest!r}: "
                     f"{checked_digest['reason']}")
    digest = checked_digest["value"]
    name = (f"{SLOT_FILE_PREFIX}{key16}-{group_field}-{digest}{SUFFIX_DATA}")
    if len(name) > MAX_NAME_CHARS:
        return _fail(f"the file name is {len(name)} chars; the cap is "
                     f"{MAX_NAME_CHARS}")
    return _ok(units=_units("name_chars", "key_chars", "group_id",
                            "geometry_chars", "shard_chars"),
               filename=name, key16=key16, group_id=split["group_id"],
               group_field=group_field, geometry_digest=digest,
               shard=key16[:SHARD_CHARS], part_name=name + SUFFIX_PART,
               name_chars=len(name), key_chars=len(key16),
               geometry_chars=len(digest), shard_chars=SHARD_CHARS)


def build_slot_path(root, key, geometry_digest) -> dict:
    """``<root>/<shard>/kvslot-<key16>-g<group>-<geodigest12>.bin``. Creates nothing.

    ``key`` is the FULL group-qualified block hash, exactly as in
    `build_slot_name`: the group is part of the identity, so it is part of the
    path, and a caller holding only a 16-hex digest gets a refusal rather than a
    path it cannot own.

    ``root`` is only joined and vetted (never ``realpath``-ed, never walked):
    stage 2 points it at wherever the operator said, and this module deliberately
    never resolves a link it was not told about.
    """
    checked_root = validate_root(root)
    if not checked_root["ok"]:
        return _fail(str(checked_root["reason"]))
    named = build_slot_name(key, geometry_digest)
    if not named["ok"]:
        return _fail(str(named["reason"]))
    directory = os.path.join(checked_root["value"], named["shard"])
    path = os.path.join(directory, named["filename"])
    if len(path) > MAX_PATH_CHARS:
        return _fail(f"the path is {len(path)} chars; the cap is "
                     f"{MAX_PATH_CHARS}")
    return _ok(units=_units("path_chars", "shard_chars", "group_id"),
               path=path, directory=directory, filename=named["filename"],
               part_path=os.path.join(directory, named["part_name"]),
               key16=named["key16"], group_id=named["group_id"],
               group_field=named["group_field"],
               geometry_digest=named["geometry_digest"],
               shard=named["shard"], path_chars=len(path))


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


def parse_path(name) -> dict:
    """Parse one of our file names. TOTAL: never raises, always classifies.

    ``ok`` only for a name that is exactly ours AND fully parses AS A GROUP-
    QUALIFIED IDENTITY. A name that does not start with our prefix or does not end
    in our suffix is ``not_ours`` - the sweep leaves those alone and does not even
    call them malformed, because they belong to someone else. A name that looks
    like ours but whose fields do not parse is ``malformed`` with a reason naming
    the field. OUR OLD TWO-FIELD NAME IS NOT malformed and is not ok either: it is
    ``stale_key_shape``, counted and unlinkable, because a `key16` with no group
    component cannot say whose bytes it holds (see the module docstring). A
    committed ``.part`` name is OURS but never a committed file: it is classified
    here so the sweep can clean it up, and `classify_name` keeps it out of any load
    path by construction (a partial write has no final name to be found under).
    """
    text = _name_text(name)
    if not text["ok"]:
        return _fail(str(text["reason"]), classification=NAME_REFUSED,
                     ours=False, filename=None, partial=False)
    base = os.path.basename(text["text"].rstrip("/")) if text["text"] else ""
    if not base or "\x00" in base:
        return _fail("the name is empty or contains a NUL byte",
                     classification=NAME_NOTOURS, ours=False, filename=base,
                     partial=False)
    if base in (".", ".."):
        return _fail("the name is a directory reference",
                     classification=NAME_NOTOURS, ours=False, filename=base,
                     partial=False)
    partial = base.endswith(SUFFIX_PART)
    body = base[:-len(SUFFIX_PART)] if partial else base
    if not body.startswith(SLOT_FILE_PREFIX):
        return _fail(f"the name does not start with {SLOT_FILE_PREFIX!r}",
                     classification=NAME_NOTOURS, ours=False, filename=base,
                     partial=partial)
    if not body.endswith(SUFFIX_DATA):
        return _fail(f"the name does not end with {SUFFIX_DATA!r}",
                     classification=NAME_NOTOURS, ours=False, filename=base,
                     partial=partial)
    fields = body[len(SLOT_FILE_PREFIX):-len(SUFFIX_DATA)]
    if "\\" in fields:
        # `os.path.basename` already removed any '/', so a backslash here is a
        # backslash INSIDE a field - a Windows-shaped or hand-edited name, which
        # is refused rather than split.
        return _fail("the name carries a path separator inside its fields",
                     classification=NAME_MALFORMED, ours=False, filename=base,
                     partial=partial)
    parts = fields.split("-")
    if len(parts) == 2:
        # THE PRE-FIX SHAPE: `kvslot-<key16>-<geodigest12>.bin`. Recognised, named
        # and NEVER matched by a lookup. A key16 alone does not say which group's
        # bytes are inside, so the identity is ambiguous by construction and no
        # group-qualified lookup may resolve to it - which is why this is its own
        # classification rather than `malformed` (we know who wrote it) or
        # `current` (we cannot say what it holds).
        legacy_key, legacy_digest = parts
        legacy_key_ok = validate_key16(legacy_key)["ok"]
        legacy_digest_ok = (len(legacy_digest) == GEOMETRY_DIGEST_CHARS
                            and all(char in HEX_DIGITS
                                    for char in legacy_digest.lower()))
        if legacy_key_ok and legacy_digest_ok:
            return _fail("the name carries no g<group> component: it is the "
                         "identity this plugin emitted before group-qualified "
                         "naming, whose bytes cannot be attributed to a KV-cache "
                         "group, so it is stale by construction and never "
                         "loadable",
                         classification=NAME_STALE_KEY_SHAPE, ours=True,
                         filename=base, partial=partial,
                         key16=legacy_key.lower(),
                         geometry_digest=legacy_digest.lower())
        return _fail(f"expected 3 '-'-separated fields after {SLOT_FILE_PREFIX!r} "
                     f"(<key16>, g<group>, <geodigest12>), found {len(parts)}, "
                     "and the 2-field form is not a valid pre-fix name either",
                     classification=NAME_MALFORMED, ours=False, filename=base,
                     partial=partial)
    if len(parts) != 3:
        return _fail(f"expected 3 '-'-separated fields after "
                     f"{SLOT_FILE_PREFIX!r}, found {len(parts)}",
                     classification=NAME_MALFORMED, ours=False, filename=base,
                     partial=partial)
    key_text, group_text, digest = parts
    checked_key = validate_key16(key_text)
    if not checked_key["ok"]:
        return _fail(f"the key field {key_text!r} is not {KEY_DIGEST_CHARS} hex "
                     f"characters: {checked_key['reason']}",
                     classification=NAME_MALFORMED, ours=False, filename=base,
                     partial=partial)
    checked_group = _parse_group_field(group_text)
    if not checked_group["ok"]:
        return _fail(f"the group field {group_text!r} is not usable: "
                     f"{checked_group['reason']}",
                     classification=NAME_MALFORMED, ours=False, filename=base,
                     partial=partial)
    if (len(digest) != GEOMETRY_DIGEST_CHARS
            or any(char not in HEX_DIGITS for char in digest.lower())):
        return _fail(f"the geometry field {digest!r} must be exactly "
                     f"{GEOMETRY_DIGEST_CHARS} hexadecimal characters",
                     classification=NAME_MALFORMED, ours=False, filename=base,
                     partial=partial)
    return {"ok": True,
            "reason": None,
            "classification": None,     # a bare name cannot be current or stale
            "ours": True,
            "filename": base,
            "partial": partial,
            "key16": checked_key["value"],
            "group_id": checked_group["value"],
            "group_field": f"{GROUP_FIELD_PREFIX}{checked_group['value']}",
            "shard": checked_key["value"][:SHARD_CHARS],
            "geometry_digest": digest.lower(),
            "units": _units("key_chars", "group_id", "geometry_chars",
                            "shard_chars")}


def _parse_group_field(text) -> dict:
    """`g<group>` exactly: the prefix, plain decimal, canonical, in bounds.

    Leading zeros are refused because they would give ONE identity TWO names -
    `g01` and `g1` cannot both be findable for group 1, which is the same "one key,
    one file" rule that folds a key's case. A group id above `MAX_GROUP_ID` is
    refused for the same reason `split_block_key` refuses it: it is not a group
    ordinal, so a name carrying it means nothing.
    """
    if not isinstance(text, str):
        return _fail(f"a group field must be a string, got {type(text).__name__}")
    if not text.startswith(GROUP_FIELD_PREFIX):
        return _fail(f"it must start with {GROUP_FIELD_PREFIX!r}, so a reader can "
                     "see that the component is a group id")
    digits = text[len(GROUP_FIELD_PREFIX):]
    if not digits or not digits.isdigit():
        return _fail("its digits are missing or are not decimal")
    if len(digits) > 1 and digits[0] == "0":
        return _fail("it has a leading zero: g01 and g1 would be two names for one "
                     "identity")
    if len(digits) > 10:
        return _fail(f"its digits are {len(digits)} chars long, which is not a "
                     "plausible group id")
    value = int(digits)
    if value > MAX_GROUP_ID:
        return _fail(f"the group id {value} is above {MAX_GROUP_ID}, so it is not a "
                     "group ordinal this engine could have produced")
    return {"ok": True, "reason": None, "value": value}


def classify_name(name, geometry_digest, peer_digests=()) -> dict:
    """Is this file OURS for THIS boot? Never a load decision, always a name one.

    ``current`` means the digest in the name is the digest the caller computed from
    its own fingerprint - i.e. this boot would name this file exactly this way,
    which is the only claim a name can honestly make. A different digest is
    ``stale_geometry``: a file written under a configuration whose MEANING differs
    (another model, dtype, slot width, group layout, rank, world size or hash
    seed), which is un-findable by construction and safe to delete.
    ``peer_digests`` names the digests of sibling ranks that share this root; a
    file carrying one of those is ``peer_rank`` - not stale, not ours to sweep, and
    reachable only by capacity eviction. A ``.part`` file is never current: an
    in-flight write is, by definition, not committed.

    A name in the pre-group SHAPE is ``stale_key_shape`` and is unlinkable BEFORE
    any digest comparison: even a byte-identical digest cannot make a name that
    does not say which group its bytes belong to into a loadable identity.
    """
    parsed = parse_path(name)
    if not parsed["ok"]:
        stale_shape = parsed["classification"] == NAME_STALE_KEY_SHAPE
        return {"ok": False,
                "reason": str(parsed["reason"]),
                "classification": parsed["classification"],
                "ours": bool(parsed["ours"]),
                "filename": parsed["filename"],
                "unlinkable": bool(stale_shape),
                "units": {}}
    if parsed["partial"]:
        return _ok(units={}, reason=("it is an in-flight .part file: not a "
                                     "committed file, never loadable, safe to "
                                     "remove"),
                   classification=NAME_STALE_GEOMETRY, ours=True,
                   filename=parsed["filename"], unlinkable=True)
    peers = {str(digest).lower() for digest in (peer_digests or ())}
    if geometry_digest is None or not isinstance(geometry_digest, str):
        return _fail("no current geometry digest to compare against", units={},
                     classification=NAME_REFUSED, ours=True,
                     filename=parsed["filename"], unlinkable=False)
    if parsed["geometry_digest"] != geometry_digest.lower():
        if parsed["geometry_digest"] in peers:
            return _ok(units={}, reason=(f"its digest {parsed['geometry_digest']!r} "
                                         "belongs to a peer rank sharing this "
                                         "root: that rank's live data, not this "
                                         "boot's to sweep"),
                       classification=NAME_PEER_RANK, ours=True,
                       filename=parsed["filename"], unlinkable=False)
        return _ok(units={}, reason=(f"its digest {parsed['geometry_digest']!r} is "
                                     f"not this boot's {geometry_digest.lower()!r}: "
                                     "a different fingerprint means these bytes "
                                     "mean something else, so this boot can never "
                                     "read it"),
                   classification=NAME_STALE_GEOMETRY, ours=True,
                   filename=parsed["filename"], unlinkable=True)
    return _ok(units={}, reason=None, classification=NAME_CURRENT, ours=True,
               filename=parsed["filename"], unlinkable=False)


# --- the sweep ---------------------------------------------------------------

def boot_sweep(directory, geometry_digest, *, peer_digests=(),
               max_files: int = MAX_SWEEP_FILES,
               max_plan: int = MAX_UNLINK_PLAN) -> dict:
    """Classify the files in ONE directory and return an UNLINK PLAN. Read-only.

    Nothing here deletes: the plan is returned for the caller (and, today, for a
    human) to carry through `execute_unlink_plan`, which re-validates every entry.
    That two-phase shape is deliberate - this is the one function in the plugin
    whose success destroys data. ``directory`` is ONE shard directory (or the
    root, whose shard sub-directories are then reported as not-a-regular-file and
    left alone); the walk is one non-recursive ``scandir`` bounded by
    ``max_files``, and ``walk_truncated`` says when the bound was hit, because an
    incomplete sweep must never be read as "there is nothing stale left".
    """
    state = _sweep_state(directory)
    if state["state"] != _OK_DIR:
        return _ok(units=_sweep_units(), directory=str(directory),
                   directory_state=state["state"], reason=state["reason"],
                   files=[], planned_paths=[], counts=_empty_sweep_counts(),
                   scanned=0, plan_truncated=False, walk_truncated=False)
    if os.path.islink(directory):
        # A symlinked DIRECTORY is the one way `directory` itself could walk out of
        # the tree, and an operator-owned path is exactly where that happens.
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
                item = _classify_entry(entry, directory, geometry_digest,
                                       peer_digests)
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


def _classify_entry(entry, directory, geometry_digest, peer_digests) -> dict:
    """One directory entry: classified, with its reason and its safety flags."""
    path = os.path.join(str(directory), entry.name)
    name = classify_name(entry.name, geometry_digest, peer_digests)
    regular = False
    symlink = False
    size_bytes = None
    mtime_ns = None
    try:
        symlink = bool(entry.is_symlink())
        entry_stat = entry.stat(follow_symlinks=False)
        regular = bool(stat_module.S_ISREG(entry_stat.st_mode))
        size_bytes = int(entry_stat.st_size)
        mtime_ns = int(entry_stat.st_mtime_ns)
    except OSError as exc:
        # A file that vanishes mid-sweep is a fact about the sweep, not an error.
        # It is refused: we cannot prove what it was, so we do not delete it.
        return {"filename": entry.name, "path": path,
                "classification": NAME_REFUSED, "ours": bool(name.get("ours")),
                "reason": f"its metadata could not be read: "
                          f"{type(exc).__name__}",
                "unlinkable": False, "regular": False, "symlink": symlink,
                "size_bytes": None, "mtime_ns": None,
                "units": _units("size_bytes", "mtime_ns")}
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
            "symlink": symlink, "size_bytes": size_bytes, "mtime_ns": mtime_ns,
            "units": _units("size_bytes", "mtime_ns")}


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


# --- capacity: an LRU clock that is not st_atime -----------------------------

def lru_eviction_plan(entries, *, geometry_digest, ceiling_bytes,
                      used_bytes=None, peer_digests=(),
                      target_fraction: float = 0.9,
                      max_plan: int = MAX_EVICTION_PLAN) -> dict:
    """Which files to delete, oldest-first, to get back under a byte ceiling.

    THE CLOCK IS ``st_mtime``. ``/home`` on the box is mounted ``relatime``, so a
    read does not reliably move ``st_atime``: an LRU built on it would rank a
    file read a million times ago as "cold" and evict the newest written file -
    the opposite of an LRU. mtime moves on write, which is the only signal we
    generate for a KV slot, so it is both honest and sufficient.

    ORDER: files whose digest is not this boot's are shed FIRST (this boot can
    never read them, so they are pure capacity), then this boot's own files oldest
    first. ``target_fraction`` (default 0.9) is the hysteresis: the plan aims at
    ``ceiling * target_fraction`` rather than at the ceiling, because a plan that
    targets the line it is standing on deletes nothing and the next spill trips
    the same check again.

    Input is a LIST of ``{"path", "size_bytes", "mtime_ns", "filename"}`` mappings
    (what a sweep or an `os.scandir` walk gives you), so this stays a pure
    function a test can drive with no filesystem at all. Deleting is
    `execute_unlink_plan`'s job, and it re-validates everything.
    """
    if not isinstance(entries, (list, tuple)):
        return _fail(f"entries must be a list, got {type(entries).__name__}")
    if not _is_plain_int(ceiling_bytes) or ceiling_bytes <= 0:
        return _fail("ceiling_bytes must be a positive int (unit: bytes)")
    if used_bytes is not None and (not _is_plain_int(used_bytes)
                                   or used_bytes < 0):
        return _fail("used_bytes must be a non-negative int or None (unit: bytes)")
    if isinstance(target_fraction, bool) or not isinstance(
            target_fraction, (int, float)):
        return _fail("target_fraction must be a number (unit: fraction)")
    if not 0.1 <= float(target_fraction) <= 1.0:
        return _fail(f"target_fraction {target_fraction} is outside [0.1, 1.0] "
                     "(it is a fraction of the ceiling, not a percentage)")
    plan_cap = _cap(max_plan, "max_plan", low=0, high=MAX_EVICTION_PLAN)
    if not plan_cap["ok"]:
        return _fail(str(plan_cap["reason"]))
    limit = plan_cap["value"]

    usable: list[dict] = []
    skipped = 0
    for item in entries:
        if not isinstance(item, dict):
            skipped += 1
            continue
        size = item.get("size_bytes")
        when = item.get("mtime_ns")
        path = item.get("path")
        if not _is_plain_int(size) or size < 0 or not _is_plain_int(when) \
                or not isinstance(path, str) or not path:
            skipped += 1
            continue
        name = classify_name(item.get("filename") or os.path.basename(path),
                             geometry_digest, peer_digests)
        classification = name["classification"] or NAME_REFUSED
        if classification not in EVICTABLE_CLASSIFICATIONS:
            skipped += 1
            continue
        usable.append({"path": path, "size_bytes": int(size),
                       "mtime_ns": int(when), "classification": classification,
                       "units": _units("size_bytes", "mtime_ns")})

    measured = 0 if used_bytes is None else int(used_bytes)
    if used_bytes is None:
        measured = sum(int(item["size_bytes"]) for item in usable)
    target = int(ceiling_bytes * float(target_fraction))
    verdict = _ok(units=_units("ceiling_bytes", "used_bytes", "target_bytes",
                               "planned_bytes", "freed_bytes", "planned",
                               "stale_first", "lru_files", "skipped", "files"),
                  needed=False, ceiling_bytes=int(ceiling_bytes),
                  used_bytes=measured, target_bytes=target,
                  paths=[], planned=0, planned_bytes=0, freed_bytes=0,
                  stale_first=0, lru_files=0, skipped=skipped,
                  files=len(usable), truncated=False,
                  reason=("the store is under its target, so nothing is "
                          "evictable yet"))
    if measured <= target:
        return verdict

    def rank(item):
        # Foreign digests first, then oldest mtime, then the path, so two boots
        # with the same directory produce the SAME plan byte for byte.
        return (0 if item["classification"] != NAME_CURRENT else 1,
                int(item["mtime_ns"]), item["path"])

    ordered = sorted(usable, key=rank)
    remaining = measured
    planned: list[dict] = []
    freed = 0
    truncated = False
    for item in ordered:
        if remaining <= target:
            break
        if len(planned) >= limit:
            truncated = True
            break
        planned.append(item)
        freed += int(item["size_bytes"])
        remaining -= int(item["size_bytes"])
    stale_first = sum(1 for item in planned
                      if item["classification"] != NAME_CURRENT)
    reason = None
    if planned:
        reason = (f"{len(planned)} file(s) shed the measured "
                  f"{measured} bytes toward the {target}-byte target "
                  f"({stale_first} of them under a foreign digest)")
    elif truncated:
        reason = f"the plan cap {limit} was reached before the target was met"
    return _ok(units=_units("ceiling_bytes", "used_bytes", "target_bytes",
                            "planned_bytes", "freed_bytes", "planned",
                            "stale_first", "lru_files", "skipped", "files"),
               needed=True, ceiling_bytes=int(ceiling_bytes),
               used_bytes=measured, target_bytes=target,
               paths=[item["path"] for item in planned],
               entries=planned, planned=len(planned), planned_bytes=freed,
               freed_bytes=freed, stale_first=stale_first,
               lru_files=len(planned) - stale_first, skipped=skipped,
               files=len(usable), truncated=truncated, reason=reason)


# --- the commit protocol -----------------------------------------------------

def write_committed_file(path, blob, *, sync: bool = True) -> dict:
    """Write ``blob`` at ``path`` so that "it exists" means "it is complete".

    The order is the whole design:
      1. make the containing directory (never through a symlink of ours);
      2. write ``<path>.part`` with ``O_CREAT|O_EXCL`` (buffered: with
         ``sync=False`` the page cache owns device ordering and nothing is
         fsynced);
      3. ``os.replace`` the part onto the final name (atomic within a filesystem);
      4. with ``sync=True`` only, ``fsync`` the DIRECTORY, so the rename itself
         survives a crash.

    A file at the final name is therefore always a complete slot AGAINST A TORN
    WRITE - the property ``O_CREAT|O_EXCL`` plus the atomic rename give, and which
    no durability primitive is needed for. That property stays whether or not
    ``sync`` is set, and MUST NOT be removed together with the fsyncs.

    WHY THE SPILL TIER DOES NOT FSYNC (the posture the production caller uses).
    These bytes are a CACHE, not a system of record: the cost of losing a span is
    one recompute, and a fsync of an 11.4 MB file buys nothing against that. The
    safety property that makes dropping durability FREE is that a journal entry
    whose file never reached the device is caught at READ time and reported to the
    engine, which truncates the span and recomputes it: ``disk_read._read_file``
    classifies it ``file_missing`` (no file at the name the index claims) or
    ``size_mismatch``/``short_read`` (a file that came up short), and
    ``disk_restore.worker_before_load``/``abandon_disk_entries`` prune every failed
    entry out of the DMA before a stale byte is copied. FALSE PRESENCE COSTS A
    RECOMPUTE, NEVER WRONG BYTES. A crash can therefore only ever make this tier
    slow for one request, never wrong.

    Every failure path removes the part and returns a classified refusal with no
    path, and ``final_exists`` says whether the final name is still absent
    afterwards - the property a test must be able to assert. ``FileExistsError``
    is benign in both places it can appear (a stale part from a dead attempt, and
    a duplicate whose final name is already there): a duplicate is the SAME bytes
    under the SAME name, which is a successful no-op, not an error.

    ``sync=False`` is the PRODUCTION setting for the spill path (buffered writes,
    no fsync, ``directory_synced: False`` reported honestly); the flag is kept
    because a test - or any caller that genuinely wants a durable file - can still
    ask for one.
    """
    if not isinstance(path, (str, os.PathLike)):
        return _fail(f"path must be a path, got {type(path).__name__}",
                     final_exists=False, duplicate=False)
    try:
        text = os.fspath(path)
    except TypeError as exc:
        return _fail(f"path is not a filesystem path: {type(exc).__name__}",
                     final_exists=False, duplicate=False)
    if not isinstance(text, str) or not text or "\x00" in text:
        return _fail("path must be a non-empty str without a NUL byte",
                     final_exists=False, duplicate=False)
    if len(text) > MAX_PATH_CHARS:
        return _fail(f"the path is {len(text)} chars; the cap is "
                     f"{MAX_PATH_CHARS}", final_exists=False, duplicate=False)
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return _fail(f"a blob must be bytes, got {type(blob).__name__}",
                     final_exists=False, duplicate=False)
    data = bytes(blob)
    directory = os.path.dirname(text) or "."
    part = text + SUFFIX_PART
    if len(os.path.basename(part)) > MAX_NAME_CHARS:
        return _fail(f"the part name is {len(os.path.basename(part))} chars; the "
                     f"cap is {MAX_NAME_CHARS}", final_exists=False,
                     duplicate=False)
    started = time.monotonic()
    try:
        os.makedirs(directory, mode=0o750, exist_ok=True)
    except FileExistsError:
        pass                          # benign: someone else made the directory
    except OSError as exc:
        return _fail(f"the directory could not be created: "
                     f"{type(exc).__name__}", final_exists=False, duplicate=False)
    if os.path.islink(part) or os.path.islink(text):
        _remove_quietly(part)
        return _fail("the target or its part is a symlink: refusing to write "
                     "through a link we did not create",
                     final_exists=os.path.exists(text), duplicate=False)
    if os.path.exists(text):
        # A duplicate, and a benign one: the name is the identity of the bytes, so
        # a file already sitting at this exact name carries this exact identity.
        # Re-writing it would be a 11.4 MB no-op on the serving box's disk, so the
        # spill path declines it and counts it. (The `FileExistsError` handlers
        # below cover the races this check cannot: a name that appears between the
        # check and the rename, on a filesystem that refuses an overwrite replace.)
        _remove_quietly(part)
        return _ok(units=_units("write_bytes", "elapsed_seconds"),
                   duplicate=True, write_bytes=0,
                   path=text, final_exists=True,
                   reason="a committed file with this identity is already there",
                   elapsed_seconds=round(time.monotonic() - started, 6))
    fd = None
    try:
        try:
            fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # A leftover from an attempt that died. It is OUR part name, so
            # removing it is ours to do; then the write proceeds normally.
            _remove_quietly(part)
            fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        written = os.write(fd, data)
        if written != len(data):
            raise OSError(f"the part write accepted {written} of {len(data)} "
                          "bytes")
        if sync:
            os.fsync(fd)
    except FileExistsError as exc:
        _remove_quietly(part)
        return _ok(units=_units("write_bytes", "elapsed_seconds"), duplicate=True,
                   write_bytes=0, path=text,
                   reason=f"a committed file is already there ({exc})",
                   final_exists=True,
                   elapsed_seconds=round(time.monotonic() - started, 6))
    except OSError as exc:
        # THE property: on every failure the final name is still absent.
        _remove_quietly(part)
        return _fail(f"the part write failed: {type(exc).__name__}",
                     final_exists=os.path.exists(text), duplicate=False,
                     write_bytes=0)
    finally:
        if fd is not None:
            try:  # noqa: SIM105 - no contextlib import to close one descriptor
                os.close(fd)
            except OSError:
                pass
    try:
        os.replace(part, text)
    except FileExistsError as exc:
        # Benign on the rare platform/filesystem combination that refuses an
        # atomic replace onto an existing name: the committed file that is there
        # carries this exact name, so it is the same identity.
        _remove_quietly(part)
        return _ok(units=_units("write_bytes", "elapsed_seconds"), duplicate=True,
                   write_bytes=0, path=text,
                   reason=f"a committed file is already there ({exc})",
                   final_exists=True,
                   elapsed_seconds=round(time.monotonic() - started, 6))
    except OSError as exc:
        _remove_quietly(part)
        return _fail(f"the rename failed: {type(exc).__name__}",
                     final_exists=os.path.exists(text), duplicate=False,
                     write_bytes=0)
    if sync:
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError as exc:
            return _ok(units=_units("write_bytes", "elapsed_seconds"), duplicate=False,
                       write_bytes=len(data), path=text, final_exists=True,
                       directory_synced=False,
                       reason=f"the file is committed but the directory could "
                              f"not be opened to fsync it: {type(exc).__name__}",
                       elapsed_seconds=round(time.monotonic() - started, 6))
        try:
            os.fsync(dir_fd)
        except OSError as exc:
            return _ok(units=_units("write_bytes", "elapsed_seconds"), duplicate=False,
                       write_bytes=len(data), path=text, final_exists=True,
                       directory_synced=False,
                       reason=f"the file is committed but its directory was not "
                              f"fsynced: {type(exc).__name__}",
                       elapsed_seconds=round(time.monotonic() - started, 6))
        finally:
            os.close(dir_fd)
    return _ok(units=_units("write_bytes", "elapsed_seconds"), duplicate=False,
               write_bytes=len(data), path=text, final_exists=True,
               # `directory_synced: False` is the NORMAL state of the spill path,
               # not an anomaly: the production caller passes `sync=False` on
               # purpose (buffered writes, the page cache owns device ordering),
               # and a lost file or short file is caught at read time as
               # `file_missing`/`size_mismatch` and recomputed. Read it as "this
               # tier does not pay for durability", never as a failure.
               directory_synced=bool(sync), reason=None,
               elapsed_seconds=round(time.monotonic() - started, 6))


def _remove_quietly(path) -> None:
    """Remove a part file, ignoring its absence. Failure-path bookkeeping."""
    try:  # noqa: SIM105 - no contextlib import for best-effort cleanup
        os.remove(path)
    except OSError:
        pass


def execute_unlink_plan(directory, paths, *, dry_run: bool = True,
                        geometry_digest=None, peer_digests=(),
                        for_capacity: bool = False,
                        max_unlinks: int = MAX_UNLINK_PLAN) -> dict:
    """Unlink the plan, re-validating EVERY entry. Destructive; dry-run by default.

    Nothing from `boot_sweep` or `lru_eviction_plan` is trusted here. For every
    path: it must resolve directly inside ``directory`` (so a plan that grew a
    traversal cannot walk), its name must parse as ours, its classification must
    STILL be deletable against the digest handed in now - a caller whose digest
    changed between planning and executing cannot delete what became current under
    a sweep, which is what ``for_capacity`` is for - and it must be a regular file
    by ``lstat``, never a link.

    ``for_capacity`` is the flag the LRU eviction passes and NOTHING else: it
    widens the allowed set from "stale" to `EVICTABLE_CLASSIFICATIONS`, i.e. it is
    the only way this boot's own current files - or a peer rank's live ones - can
    be deleted, and the only reason that is legitimate is the byte ceiling.
    Deleting a file is always safe for correctness (there is no read path here);
    it is never safe for capacity planning unless the caller is shedding to meet a
    ceiling it was configured with. Every refusal is counted and carries a reason;
    a directory that is itself a symlink is refused outright.
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
    overflow = None
    if len(pending) > limit:
        counts["refused"] += 1
        pending = pending[:limit]
        overflow = {"path": None, "action": "refused",
                    "reason": f"the plan names {len(paths)} paths; the cap is "
                              f"{limit} (call again for the rest)",
                    "size_bytes": None, "units": _units("size_bytes")}
    actions: list[dict] = []
    if overflow:
        actions.append(overflow)
    bytes_freed = 0
    for path in pending:
        verdict = _validate_unlink(path, root, geometry_digest, peer_digests,
                                   for_capacity)
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


def _validate_unlink(path, root, geometry_digest, peer_digests,
                     for_capacity) -> dict:
    """Prove one path is a regular, non-link, ours, in-directory, deletable file."""
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
                     "deletion never walks out of the directory it was handed")
    parsed = parse_path(base)
    if not parsed["ok"] and parsed["classification"] != NAME_STALE_KEY_SHAPE:
        return _fail(f"{base} is not ours to delete: {parsed['reason']}")
    # A name that does not parse AS CURRENT is not automatically foreign: the
    # pre-group shape is OURS and is named stale on purpose, so it has to reach the
    # classification check below instead of stopping at "this name does not parse".
    # Without that, a sweep could PLAN the ambiguous files and this executor could
    # then refuse them - invisible capacity nobody reclaims, which is the exact
    # dead weight the classification exists to make removable.
    if geometry_digest is not None:
        fresh = classify_name(base, geometry_digest, peer_digests)
        classification = fresh["classification"]
        allowed = set(UNLINKABLE_CLASSIFICATIONS)
        if for_capacity:
            allowed |= set(EVICTABLE_CLASSIFICATIONS)
        if classification not in allowed:
            detail = fresh["reason"] or "it is this boot's own current file"
            return _fail(f"{base} is {classification} against the current "
                         f"fingerprint: {detail}")
    target = os.path.join(root, base)
    try:
        info = os.lstat(target)
    except OSError as exc:
        return _fail(f"{base} could not be inspected: {type(exc).__name__}")
    if not stat_module.S_ISREG(info.st_mode):
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
        st = os.statvfs(text)
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
    unit = int(st.f_frsize or 0)
    if unit <= 0:
        return _space_fail("statvfs reported a zero fragment size, so no byte "
                           "count can be derived from it")
    free = int(st.f_bavail) * unit
    total = int(st.f_blocks) * unit
    available = int(st.f_bfree) * unit
    return _ok(units=_units("free_bytes", "total_bytes", "used_bytes",
                            "reserved_for_root_bytes", "unit_size_bytes",
                            "fragment_bytes", "files_free"),
               available=True, reason=None, statvfs_state="ok",
               path=text, free_bytes=free, total_bytes=total,
               used_bytes=max(0, total - available),
               reserved_for_root_bytes=max(0, available - free),
               unit_size_bytes=unit, fragment_bytes=int(st.f_frsize),
               read_only=bool(st.f_flag & 1),             # ST_RDONLY
               files_free=int(st.f_favail))


def _space_fail(reason: str) -> dict:
    return _ok(units=_units("free_bytes", "total_bytes", "used_bytes",
                            "reserved_for_root_bytes", "unit_size_bytes",
                            "fragment_bytes", "files_free"),
               available=False, statvfs_state=_READ_FAILED, path=None,
               free_bytes=None, total_bytes=None, used_bytes=None,
               reserved_for_root_bytes=None, unit_size_bytes=None,
               fragment_bytes=None, read_only=None, files_free=None,
               reason=reason)


def space_check(path, *, watermark_bytes=None, watermark_percent=None,
                extra_bytes=None) -> dict:
    """Is there room to write? The free-space read plus the watermark verdict.

    The watermark accepts BYTES or a PERCENT of total, and the EFFECTIVE byte
    value is published so a reader never has to re-derive it. ``extra_bytes`` is
    the reservation for the write ABOUT to happen (one slot's bytes), so a writer
    asks "is there room for THIS slot" rather than "is there room for something".
    Percent-of-total is a proxy, not a promise: on a RAID0 with a large root
    reserve the two can disagree, and the byte figure is the one that decides.

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
    if extra_bytes is not None and (isinstance(extra_bytes, bool)
                                    or not isinstance(extra_bytes, int)):
        return _space_check_fail("extra_bytes must be a non-negative int (unit: "
                                 "bytes)")
    if extra_bytes is not None and extra_bytes < 0:
        return _space_check_fail("extra_bytes is negative")
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
    reserve = int(extra_bytes or 0)
    headroom = space["free_bytes"] - reserve
    below = None
    reason = None
    if source == "none":
        reason = "no watermark was configured, so there is no verdict to give"
    else:
        below = headroom < effective
    return _ok(units=_units("free_bytes", "total_bytes", "used_bytes",
                            "watermark_bytes", "watermark_bytes_configured",
                            "watermark_percent", "reserved_for_root_bytes",
                            "unit_size_bytes", "free_percent_of_total",
                            "reserve_bytes", "headroom_bytes"),
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
               reserve_bytes=reserve, headroom_bytes=headroom,
               read_only=space["read_only"],
               free_percent_of_total=(
                   round(100.0 * space["free_bytes"] / space["total_bytes"], 3)
                   if space["total_bytes"] > 0 else None))


def _space_check_fail(reason: str, space_available: bool = True,
                      space_reason: str | None = None) -> dict:
    return _ok(units=_units("free_bytes", "total_bytes", "used_bytes",
                            "watermark_bytes", "watermark_bytes_configured",
                            "watermark_percent", "reserved_for_root_bytes",
                            "unit_size_bytes", "free_percent_of_total",
                            "reserve_bytes", "headroom_bytes"),
               ok=False, reason=reason, space_available=space_available,
               space_reason=space_reason, statvfs_state=_READ_FAILED, path=None,
               free_bytes=None, total_bytes=None, used_bytes=None,
               reserved_for_root_bytes=None, unit_size_bytes=None,
               watermark_bytes=None, watermark_bytes_configured=None,
               watermark_percent=None, watermark_source="none",
               below_watermark=None, reserve_bytes=None, headroom_bytes=None,
               read_only=None, free_percent_of_total=None)
