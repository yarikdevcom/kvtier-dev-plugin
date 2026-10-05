"""What the SSD tier holds, as the SCHEDULER process can legitimately know it.

WHY THIS EXISTS AT ALL, AND WHY IT IS A JOURNAL
-----------------------------------------------
A restore begins in the scheduler: `get_num_new_matched_tokens` has to answer
"do you have these tokens?" before anything is loaded. But the files are written
in a WORKER, by `disk_spill`'s writer thread, and they are deleted there too, by
its LRU eviction - and this fork's scheduler and workers are different processes
with no shared memory (the whole reason `worker_diag` writes rank files,
`blockdump` and `spill-one` use a request/receipt pair, and `disk_spill`'s arm
crosses as a file). So the scheduler cannot read a worker's memory; it can read
what the worker WROTE DOWN.

That is this module: `disk_spill` appends one line per committed spill and one
per eviction to a per-rank journal (it does: its writer thread calls
`record_spill` after the commit's `os.replace` and `record_removal` for every
file the ceiling plan unlinks, both from the spill thread and never from an
engine thread), and this module replays those lines into an in-process
`identity -> {bytes, rank}` index. It is the same cursor-over-a-file idiom the
spill tier's arm file and the plugin's byte-dump receipts already use, for the
same reason: an append-only file written by one process and read by another with a
byte cursor is the smallest thing that cannot lie about ordering.

A JOURNAL FAILURE IS NEVER A SPILL FAILURE. The spill is the product; this index
is an index of it. Every entry point a writer calls - `record_spill`,
`record_removal`, `journal_append` - catches `Exception`, counts, and returns a
classified refusal, and none of them may raise into, block or fail the write that
called them. A worker whose journal cannot be written keeps spilling.

THE INDEX MAY ONLY EVER BE STALE IN THE SAFE DIRECTION
------------------------------------------------------
A false ABSENCE costs a recompute. A false PRESENCE costs a load that fails -
which is also a recompute, but one that has to be reported (see
`disk_restore`'s failure path and the `WAITING_FOR_REMOTE_KVS` rule). Both
directions degrade, so this index is allowed to be wrong about *capacity* and is
never allowed to raise:

  * an eviction removes the entry (`-` journal line), so a shed file stops being
    offered within one refresh;
  * the entry cap drops the OLDEST entries and counts it, so memory is bounded
    by `MAX_INDEX_ENTRIES` whatever the store holds;
  * a journal that is truncated from outside, malformed, or past its size cap
    STOPS being applied and says so. It never guesses, and it never claims a
    file it did not see written.

WHAT A KEY IS, AND WHY THE PATH IS ONLY A GUESS
-----------------------------------------------
The index is keyed by an IDENTITY, `<key16>:g<group>`: the leading 16 hex
characters of the block hash DIGEST plus the KV-cache group id the engine appends
to it (`kv_cache_utils.py:83`), unpacked by `disk_layout.split_block_key` - the
same split that names the file. Keying the index on `key16` alone was the same
identity bug one layer up: two prefix-cacheable groups over one token chain share
a `key16`, so one entry would have stood for two files and a restore could be
offered for a group whose bytes are not in it. A journal line therefore carries
the group explicitly, and a caller that hands `lookup()` a bare 16-hex digest gets
a miss rather than a guess.

The PATH is not stored at all. Four ranks name one identity four different ways,
because the digest covers rank-local fields (`rank`, and `total_bytes_per_block`,
which is 11,468,800 B on PP0 and 12,902,400 B on PP1), and no single path could be
right for all of them. `disk_restore` sends the identity to the workers inside a
path-shaped string and every worker resolves the name IT would have written; a
worker that cannot name it reports a miss. What the journal records per identity
is the byte count (for the capacity view) and which rank's journal contributed it
(for diagnosis), and nothing else.

Stdlib only. No thread, no vllm, no torch. Every entry point returns a dict and
never raises into the caller, because `refresh()` runs on the scheduler's lookup
path.
"""

from __future__ import annotations

import os
import time

from . import config, disk_layout, snapshot

# --- the caps, and the unit of every one of them ----------------------------

MAX_INDEX_ENTRIES = 65_536          # KEYS held in memory (each drops the oldest)
MAX_JOURNAL_BYTES = 32 * 1024 * 1024  # BYTES one rank's journal may reach
MAX_JOURNAL_FILES = 64              # journal FILES one refresh will open
MAX_LINES_PER_REFRESH = 8_192       # LINES applied per refresh (bounded work)
MAX_READ_PER_REFRESH = 512 * 1024   # BYTES read across all journals per refresh
MAX_JOURNAL_RANK_CHARS = 32         # CHARS of a rank id in a journal file name
MAX_STRING_CHARS = 160              # CHARS of a path/reason kept in state
REFRESH_MIN_INTERVAL_SECONDS = 1.0  # SECONDS between journal refreshes

JOURNAL_PREFIX = "disk-spill-journal-"
JOURNAL_SUFFIX = ".jsonl"
JOURNAL_FIELDS = 5                  # "<seq> <+|-> <key16> <bytes> <group>"
SPILL_LINE = "+"
REMOVE_LINE = "-"
LINE_SEP = " "
IDENTITY_SEP = ":"
GROUP_PREFIX = "g"                  # `<key16>:g<group>` is an index key
MAX_JOURNAL_SIZE_BYTES = 1 << 40    # BYTES one line may claim (a slot is 11.5 MB)
SEQ_TAIL_READ_BYTES = 4096          # BYTES of the journal tail read to seed a seq

# Why a line was not applied. A closed vocabulary, one counter each.
DROP_REASONS = ("malformed_line", "bad_op", "bad_key", "bad_size", "bad_seq",
                "bad_group", "journal_shrank", "journal_capped", "line_cap",
                "read_cap", "stopped")

_state: dict = {
    "entries": {},                    # identity -> entry dict (insertion ordered)
    "cursors": {},                    # journal path -> byte offset
    "seq": {},                        # journal path -> last applied seq
    "stopped": {},                    # journal path -> reason it stopped
    "applied": 0,
    "spills": 0,
    "removals": 0,
    "misses": 0,
    "hits": 0,
    "entries_dropped": 0,
    "dropped": {reason: 0 for reason in DROP_REASONS},
    "journal_writes": 0,
    "journal_write_errors": 0,
    "journal_full": False,
    "journal_lines": 0,
    "journal_bytes": 0,
    "journal_seeded": False,
    "journal_seq_seed": 0,
    "refreshes": 0,
    "refresh_skipped": 0,
    "last_reason": None,
    "last_error": None,
    "last_journal": None,
    "last_refresh_mono": None,
    "bytes_indexed": 0,
}


def reset_state_for_tests() -> None:
    """Back to a fresh boot: empty index, no cursors, no counters.

    The whole module is per-process memory, so this is the only teardown it
    needs; there is no thread and no open descriptor to stop.
    """
    _state.update({
        "entries": {}, "cursors": {}, "seq": {}, "stopped": {},
        "applied": 0, "spills": 0, "removals": 0, "misses": 0, "hits": 0,
        "entries_dropped": 0,
        "dropped": {reason: 0 for reason in DROP_REASONS},
        "journal_writes": 0, "journal_write_errors": 0, "journal_full": False,
        "journal_lines": 0, "journal_bytes": 0,
        "journal_seeded": False, "journal_seq_seed": 0,
        "refreshes": 0, "refresh_skipped": 0,
        "last_reason": None, "last_error": None, "last_journal": None,
        "last_refresh_mono": None, "bytes_indexed": 0,
    })


def counters() -> dict:
    """A copy for a payload. The ENTRIES are never included (only their count)."""
    out = dict(_state)
    out["entries"] = int(len(_state["entries"]))
    return out


# --- small helpers ------------------------------------------------------------

def _note(reason) -> None:
    _state["last_reason"] = str(reason)[:MAX_STRING_CHARS]


def _fail(reason, **extra) -> dict:
    out = {"ok": False, "reason": str(reason)[:MAX_STRING_CHARS]}
    out.update(extra)
    return out


def _ok(**fields) -> dict:
    out = {"ok": True, "reason": None}
    out.update(fields)
    return out


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def identity_of(key16, group_id) -> str:
    """`<key16>:g<group>`, the one key this index is stored under.

    Built here and nowhere else, so the writer of a journal line and the reader of
    a lookup cannot drift into two different spellings of one identity (which
    would look exactly like a missing file).
    """
    return f"{key16}{IDENTITY_SEP}{GROUP_PREFIX}{int(group_id)}"


# --- the journal: one rank's write/eviction stream ---------------------------

def journal_directory():
    """Where journals live: the diagnostics directory, or None.

    The same directory the rank files, the arm file and the dump receipts use,
    so the operator names ONE place and every cross-process channel of the
    plugin goes there. Unset means the restore path has no way to learn what
    the workers spilled, which is reported, not assumed.
    """
    directory = config.diagnostics_dir()
    return None if directory is None else str(directory)


def journal_path(rank=None) -> dict:
    """`(path | None, why)` for one rank's journal.

    `rank` defaults to this process's own rank (the ONE rule lives in
    `worker_diag.rank_id()`, so a worker's journal is named by the same id its
    rank file uses). A scheduler process has no rank and must pass one to read
    a specific file; `refresh()` globs instead.
    """
    directory = journal_directory()
    if directory is None:
        return _fail(f"{config.ENV_DIAG_DIR} is not set: there is no directory "
                     "for a spill journal to live in", path=None)
    if rank is None:
        from . import worker_diag  # noqa: PLC0415 - the one rank rule, not two
        rank, why = worker_diag.rank_id()
        if rank is None:
            return _fail(f"no rank to name this process's journal with: {why}",
                         path=None)
    text = str(rank)
    if not text or len(text) > MAX_JOURNAL_RANK_CHARS:
        return _fail(f"the rank id {text[:40]!r} is empty or longer than "
                     f"{MAX_JOURNAL_RANK_CHARS} characters", path=None)
    if any(char in text for char in ("/", "\\", "\n", "\x00")):
        # A rank id becomes a FILE NAME component here, so this is the one place
        # the shape has to be enforced: `disk_layout._validate_string` keeps
        # separators out of the fingerprint's rank FIELD, and this keeps them out
        # of the journal's name. A traversal through a rank id is not a
        # theoretical worry - `VLLM_E2_TIER_DIAG_RANK` is operator input.
        return _fail("a rank id may not contain a path separator or a NUL",
                     path=None)
    return _ok(path=os.path.join(directory,
                                 f"{JOURNAL_PREFIX}{text}{JOURNAL_SUFFIX}"),
               rank=text)


def _read_tail_seq(path) -> dict:
    """The last complete line's sequence number, from the file's tail.

    WHY A NEW BOOT CARES ABOUT AN OLD BOOT'S SEQUENCE: the journal lives in the
    diagnostics directory and OUTLIVES the process that wrote it, while `_state`
    starts at zero. Seeding the first line of boot 2 at 1 would hand the
    scheduler's replay a DECREASING sequence, and the replay's one rule about a
    journal it can no longer trust is to STOP applying it - which would silently
    un-wire this whole index after the first restart. So the sequence is seeded
    from the file, monotone across boots, and nothing is deleted or renamed.

    Bounded: the last `SEQ_TAIL_READ_BYTES` of the file, one read, once per
    process. No complete line parses to a positive int means `ok: False`, and the
    writer refuses to guess a number it cannot prove is increasing.
    """
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return _ok(seq=0, empty=True)
    except OSError as exc:
        return _fail(f"the journal's size could not be read: "
                     f"{type(exc).__name__}")
    if size <= 0:
        return _ok(seq=0, empty=True)
    want = min(size, SEQ_TAIL_READ_BYTES)
    try:
        with open(path, "rb") as handle:
            handle.seek(size - want)
            raw = handle.read(want)
    except OSError as exc:
        return _fail(f"the journal's tail could not be read: "
                     f"{type(exc).__name__}")
    if size > want:
        # The window may start mid-line: drop everything before the first newline
        # so a partial line can never be parsed as a sequence.
        cut = raw.find(b"\n")
        raw = raw[cut + 1:] if cut >= 0 else b""
    lines = [row for row in raw.split(b"\n") if row.strip()]
    if size > want and not lines:
        return _fail("the journal is larger than the tail window and that window "
                     "holds no complete line")
    for row in reversed(lines):
        fields = row.decode("utf-8", errors="replace").split(LINE_SEP)
        if len(fields) < JOURNAL_FIELDS:
            continue
        try:
            seq = int(fields[0])
        except ValueError:
            continue
        if seq >= 1:
            return _ok(seq=seq, empty=False)
    return _fail("no complete journal line in the tail carries a usable sequence "
                 "number")


def journal_append(op, key, size_bytes=None, group_id=None) -> dict:
    """Append one line to THIS process's journal. Bounded; never raises.

    Called from `disk_spill`'s writer thread (the only thread that knows a file
    was committed, and the only one that deletes one), never from an engine
    thread. One `write()` of one line, so a reader that stops at the last
    newline can never apply half a record.

    `key` is the engine's FULL group-qualified block hash, or a `key16` with the
    `group_id` handed in beside it (what `record_spill` does, because it parses a
    PATH rather than carrying a key through the writer thread). Both forms resolve
    through `disk_layout.split_block_key`/`validate_key16`, so the writer cannot
    invent a third spelling of an identity - and a line that cannot name a group is
    refused, because an identity without one is the bug this journal exists above.

    Once the journal reaches `MAX_JOURNAL_BYTES` it STOPS: the writer refuses
    and counts, and the scheduler's index then simply stops learning new files.
    Rotating instead would be the dangerous choice - a rotated file loses the
    record of what the index still believes exists.
    """
    try:
        if op not in (SPILL_LINE, REMOVE_LINE):
            return _fail(f"the journal op must be {SPILL_LINE!r} or "
                         f"{REMOVE_LINE!r}, got {op!r}")
        if group_id is not None:
            checked = disk_layout.validate_key16(key)
            if not checked["ok"]:
                return _fail(f"the key is not a usable journal key: "
                             f"{checked['reason']}")
            key16 = checked["value"]
            if not _is_plain_int(group_id) or not 0 <= int(group_id) <= (
                    disk_layout.MAX_GROUP_ID):
                return _fail(f"the group id {group_id!r} is not an int in "
                             f"[0, {disk_layout.MAX_GROUP_ID}]")
            group = int(group_id)
        else:
            split = disk_layout.split_block_key(key)
            if not split["ok"]:
                return _fail(f"the key is not a usable journal key: "
                             f"{split['reason']}")
            key16, group = split["key16"], split["group_id"]
        size = int(size_bytes or 0)
        if not _is_plain_int(size) or size < 0 or size > MAX_JOURNAL_SIZE_BYTES:
            return _fail(f"the size must be an int in [0, "
                         f"{MAX_JOURNAL_SIZE_BYTES}], got {size_bytes!r} "
                         "(unit: bytes)")
        named = journal_path()
        if not named["ok"]:
            _state["journal_write_errors"] += 1
            _note(named["reason"])
            return _fail(named["reason"], reason_kind="no_channel")
        path = named["path"]
        try:
            current = os.path.getsize(path)
        except FileNotFoundError:
            current = 0
        except OSError as exc:
            _state["journal_write_errors"] += 1
            return _fail(f"the journal could not be stat'ed: "
                         f"{type(exc).__name__}", reason_kind="journal_error")
        if current >= MAX_JOURNAL_BYTES:
            _state["journal_full"] = True
            _state["dropped"]["journal_capped"] += 1
            _note(f"the journal reached its {MAX_JOURNAL_BYTES}-byte cap, so "
                  "nothing further is recorded: the scheduler's index stops "
                  "learning about this rank's files (never a wrong claim)")
            return _fail("the journal is at its size cap",
                         reason_kind="journal_capped", path=path)
        if not _state["journal_seeded"]:
            seeded = _read_tail_seq(path)
            if not seeded["ok"]:
                # A number we cannot prove is increasing is not a sequence, and a
                # decreasing one would stop the reader for this rank forever.
                _state["journal_write_errors"] += 1
                _note(seeded["reason"])
                return _fail(f"the journal's sequence could not be seeded: "
                             f"{seeded['reason']}",
                             reason_kind="journal_error", path=path)
            _state["journal_lines"] = int(seeded["seq"])
            _state["journal_seq_seed"] = int(seeded["seq"])
            _state["journal_seeded"] = True
            _state["journal_bytes"] = int(current)
        seq = int(_state["journal_lines"]) + 1
        line = (f"{seq}{LINE_SEP}{op}{LINE_SEP}{key16}{LINE_SEP}{size}"
                f"{LINE_SEP}{group}\n")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "ab") as handle:
                handle.write(line.encode("utf-8"))
        except OSError as exc:
            _state["journal_write_errors"] += 1
            _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
            return _fail(f"the journal line could not be written: "
                         f"{type(exc).__name__}", reason_kind="journal_error",
                         path=path)
        _state["journal_writes"] += 1
        _state["journal_lines"] = seq
        _state["journal_bytes"] = current + len(line)
        _state["last_journal"] = path
        return _ok(path=path, seq=seq, bytes=len(line), rank=named["rank"])
    except Exception as exc:  # noqa: BLE001 - a journal never breaks a spill
        _state["journal_write_errors"] += 1
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "disk_index journal_append")
        return _fail(f"journaling raised {type(exc).__name__}",
                     reason_kind="journal_error")


# --- the reader: replay every rank's journal into the index -----------------

def refresh(*, force: bool = False) -> dict:
    """Apply every rank's new journal lines. Cheap, bounded, never raises.

    Called from the lookup path, so the whole refresh is bounded by
    `MAX_JOURNAL_FILES` files, `MAX_READ_PER_REFRESH` bytes and
    `MAX_LINES_PER_REFRESH` lines, and each file's read is bounded by the same
    budget. A refresh opens at most one `stat` per journal and reads only the
    bytes after its cursor, so an idle store costs one stat per rank.

    `force` ignores the age floor (a verb that wants the truth right now); the
    lookup path uses the floor so a burst of requests does not turn into a burst
    of stats.
    """
    started = time.monotonic()
    out = {"ok": True, "reason": None, "journals": 0, "lines": 0, "spills": 0,
           "removals": 0, "dropped": 0, "entries": int(len(_state["entries"])),
           "applied": False}
    try:
        directory = journal_directory()
        if directory is None:
            out["reason"] = (f"{config.ENV_DIAG_DIR} is not set: no spill journal "
                             "can be read, so the disk index stays empty")
            _note(out["reason"])
            return out
        now = time.monotonic()
        last = _state["last_refresh_mono"]
        if (not force and isinstance(last, (int, float))
                and (now - last) < REFRESH_MIN_INTERVAL_SECONDS):
            # The lookup path is per REQUEST; the journal is per second. Skipping
            # is not a lie - the index simply has at most this much latency, and
            # a restore that races its own journal entry is a recompute.
            _state["refresh_skipped"] += 1
            out["reason"] = (f"refreshed {round(now - float(last), 3)}s ago; the "
                             f"floor is {REFRESH_MIN_INTERVAL_SECONDS}s")
            return out
        _state["refreshes"] += 1
        _state["last_refresh_mono"] = now
        names = _journal_names(directory)
        if not names:
            out["reason"] = ("no spill journal file exists in the diagnostics "
                             "directory: no rank has spilled anything yet")
            return out
        budget = MAX_READ_PER_REFRESH
        lines = MAX_LINES_PER_REFRESH
        for name in names:
            path = os.path.join(directory, name)
            out["journals"] += 1
            if budget <= 0 or lines <= 0:
                _state["dropped"]["read_cap"] += 1
                out["dropped"] += 1
                continue
            read, applied, spills, removes = _apply_journal(path, budget, lines)
            budget -= read
            lines -= applied
            out["lines"] += applied
            out["spills"] += spills
            out["removals"] += removes
        out["entries"] = int(len(_state["entries"]))
        out["applied"] = out["lines"] > 0
        out["elapsed_seconds"] = round(time.monotonic() - started, 6)
        if out["journals"]:
            out["reason"] = None
        return out
    except Exception as exc:  # noqa: BLE001 - the lookup path must not raise
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "disk_index refresh")
        out["reason"] = f"the journal refresh raised {type(exc).__name__}"
        out["ok"] = False
        return out


def _journal_names(directory) -> list:
    """The journal file names, bounded and sorted. An unreadable dir is none."""
    try:
        names = [name for name in os.listdir(directory)
                 if name.startswith(JOURNAL_PREFIX)
                 and name.endswith(JOURNAL_SUFFIX)]
    except OSError as exc:
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        _note(f"the diagnostics directory could not be listed: "
              f"{type(exc).__name__}")
        return []
    names.sort()
    if len(names) > MAX_JOURNAL_FILES:
        _note(f"{len(names)} journal files exist; only the first "
              f"{MAX_JOURNAL_FILES} are read")
        names = names[:MAX_JOURNAL_FILES]
    return names


def _apply_journal(path, budget, line_budget) -> tuple:
    """Replay one journal from its cursor.

    Returns ``(bytes_read, lines_applied, spills_applied, removals_applied)``.

    Only COMPLETE lines (ending in a newline) are applied, and the cursor only
    ever advances to the last newline: a line the writer is halfway through is
    nobody's fact yet. A file that got SMALLER was truncated by someone who is
    not this plugin - the entries contributed by that rank are dropped, because
    we can no longer tell what they were, and the replay starts from the top.
    """
    read_spills_local = 0
    read_removes_local = 0
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return 0, 0, 0, 0
    except OSError as exc:
        # A stat failure is REPORTED, never counted as a data loss: the cursor
        # stays where it is, so the next refresh tries the same bytes again.
        _note(f"{os.path.basename(path)} could not be stat'ed: "
              f"{type(exc).__name__}")
        return 0, 0, 0, 0
    cursor = int(_state["cursors"].get(path, 0))
    if size < cursor:
        _state["dropped"]["journal_shrank"] += 1
        _forget_rank(path)
        _state["cursors"][path] = 0
        _state["seq"].pop(path, None)
        _note(f"{os.path.basename(path)} shrank from {cursor} to {size} bytes: "
              "someone truncated it, so the entries this rank contributed are "
              "dropped and the journal is replayed from the start (a lost line "
              "costs a missed restore, never a wrong one)")
        cursor = 0
    if _state["stopped"].get(path):
        _state["dropped"]["stopped"] += 1
        return 0, 0, 0, 0
    if size <= cursor:
        return 0, 0, 0, 0
    want = min(size - cursor, max(0, budget))
    if want <= 0:
        _state["dropped"]["read_cap"] += 1
        return 0, 0, 0, 0
    try:
        with open(path, "rb") as handle:
            handle.seek(cursor)
            raw = handle.read(want)
    except OSError as exc:
        _note(f"{os.path.basename(path)} could not be read: "
              f"{type(exc).__name__}")
        return 0, 0, 0, 0
    last_newline = raw.rfind(b"\n")
    if last_newline < 0:
        # Nothing complete yet. The cursor stays, so the next refresh retries.
        return 0, 0, 0, 0
    body = raw[:last_newline + 1]
    applied = 0
    for text in body.decode("utf-8", errors="replace").splitlines():
        if applied >= line_budget:
            _state["dropped"]["line_cap"] += 1
            break
        applied += 1
        kind = _apply_line(text, path)
        if kind is None:
            break
        if kind == SPILL_LINE:
            read_spills_local += 1
        else:
            read_removes_local += 1
    _state["cursors"][path] = cursor + last_newline + 1
    _state["last_journal"] = path
    return last_newline + 1, applied, read_spills_local, read_removes_local


def _forget_rank(path) -> None:
    """Drop every index entry that came from this journal."""
    entries = _state["entries"]
    doomed = [key for key, entry in entries.items()
              if entry.get("journal") == path]
    for key in doomed:
        entries.pop(key, None)
    _state["bytes_indexed"] = max(0, sum(int(entry["bytes"])
                                         for entry in entries.values()))


def _apply_line(text, path):
    """Apply one journal line: `<seq> <+|-> <key16> <bytes> <group>`.

    Returns the line's op on success, or None to mean STOP: the stream is no
    longer trustworthy and nothing further from this journal may be applied.

    A line without the group field is not "an old line with an unknown group", it
    is an identity that cannot be named: applying it would put a `key16`-only entry
    in an index whose whole contract is that an entry says which GROUP's bytes it
    stands for. So the 4-field shape stops the journal, exactly like any other
    line this build cannot interpret - and it can only appear if someone hand-edits
    a journal, because nothing in this repo wrote one before the group fix.
    """
    fields = text.split(LINE_SEP)
    if len(fields) != JOURNAL_FIELDS:
        return _stop(path, f"the line {text[:80]!r} has {len(fields)} fields; "
                           f"{JOURNAL_FIELDS} are required (a line without a group "
                           "field names an identity this index cannot hold)",
                     "malformed_line")
    seq_text, op, key_text, size_text, group_text = fields
    if op not in (SPILL_LINE, REMOVE_LINE):
        return _stop(path, f"the op {op!r} is neither {SPILL_LINE!r} nor "
                           f"{REMOVE_LINE!r}", "bad_op")
    checked = disk_layout.validate_key16(key_text)
    if not checked["ok"]:
        return _stop(path, f"the key {key_text!r} is not "
                           f"{disk_layout.KEY_DIGEST_CHARS} hex characters",
                     "bad_key")
    key16 = checked["value"]
    try:
        seq = int(seq_text)
        size = int(size_text)
        group = int(group_text)
    except ValueError:
        return _stop(path, f"the line {text[:80]!r} carries a non-integer "
                           "sequence, size or group", "bad_size")
    if size < 0 or size > MAX_JOURNAL_SIZE_BYTES:
        return _stop(path, f"the size {size} is outside [0, "
                           f"{MAX_JOURNAL_SIZE_BYTES}]", "bad_size")
    if not 0 <= group <= disk_layout.MAX_GROUP_ID:
        return _stop(path, f"the group id {group} is outside [0, "
                           f"{disk_layout.MAX_GROUP_ID}]", "bad_group")
    if seq < 1 or seq <= int(_state["seq"].get(path, 0)):
        return _stop(path, f"the sequence {seq} does not increase (the last "
                           f"applied was "
                           f"{int(_state['seq'].get(path, 0))})", "bad_seq")
    _state["seq"][path] = seq
    _state["applied"] += 1
    identity = identity_of(key16, group)
    if op == REMOVE_LINE:
        _state["removals"] += 1
        gone = _state["entries"].pop(identity, None)
        if gone is not None:
            _state["bytes_indexed"] = max(0, int(_state["bytes_indexed"])
                                          - int(gone["bytes"]))
        return REMOVE_LINE
    _state["spills"] += 1
    _insert(identity, size, path, group)
    return SPILL_LINE


def _stop(path, reason, kind):
    """Stop applying a journal, once, with the reason remembered. Returns None."""
    _state["dropped"][kind] += 1
    already = _state["stopped"].get(path)
    if already is None:
        _state["stopped"][path] = reason
        _note(f"{os.path.basename(path)}: {reason}; this rank's journal is no "
              "longer applied (an index that stops growing is a tier that offers "
              "less, never one that offers wrong)")
    return


def _insert(identity, size, path, group_id=None) -> None:
    """Add or refresh one entry, honouring the hard cap. Keyed by IDENTITY."""
    entries = _state["entries"]
    existing = entries.get(identity)
    if existing is not None:
        existing["bytes"] = int(size)
        existing["journal"] = path
        existing["seen_ts"] = round(time.time(), 6)
        return
    while len(entries) >= MAX_INDEX_ENTRIES:
        oldest = next(iter(entries))
        dropped = entries.pop(oldest)
        _state["entries_dropped"] += 1
        _state["bytes_indexed"] = max(0, int(_state["bytes_indexed"])
                                      - int(dropped["bytes"]))
        _state["last_reason"] = (f"the index is at its {MAX_INDEX_ENTRIES}-entry "
                                 f"cap: {oldest} was dropped to admit a newer "
                                 "identity (a dropped entry is a missed restore)")
    entries[identity] = {"bytes": int(size), "journal": path,
                         "group": (int(group_id) if _is_plain_int(group_id)
                                   else None),
                         "seen_ts": round(time.time(), 6)}
    _state["bytes_indexed"] = int(_state["bytes_indexed"]) + int(size)


# --- the index itself --------------------------------------------------------

def lookup(key) -> dict:
    """`{present, key16, group_id, bytes, rank, reason}` for one identity.

    `key` is the engine's FULL group-qualified block hash (bytes or hex). A bare
    16-hex `key16` is a MISS with a reason, never a guess: it does not say which
    group's bytes are wanted, and offering the wrong group's file is the exact
    silent-corruption class the identity carries the group to avoid. Never raises.
    """
    try:
        split = disk_layout.split_block_key(key)
        if not split["ok"]:
            return _fail(f"the index cannot look this up: {split['reason']}",
                         present=False, key16=None, group_id=None)
        key16, group = split["key16"], split["group_id"]
        entry = _state["entries"].get(identity_of(key16, group))
        if entry is None:
            _state["misses"] += 1
            return _ok(present=False, key16=key16, group_id=group, bytes=0,
                       rank=None,
                       reason="this boot's journal does not name a file for "
                              "this key and group")
        _state["hits"] += 1
        return _ok(present=True, key16=key16, group_id=group,
                   bytes=int(entry["bytes"]),
                   rank=_rank_of(entry.get("journal")), reason=None)
    except Exception as exc:  # noqa: BLE001 - a lookup never breaks a schedule
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "disk_index lookup")
        return _fail(f"the lookup raised {type(exc).__name__}", present=False,
                     key16=None, group_id=None)


def _rank_of(journal_path_value):
    if not journal_path_value:
        return None
    name = os.path.basename(str(journal_path_value))
    if not (name.startswith(JOURNAL_PREFIX) and name.endswith(JOURNAL_SUFFIX)):
        return None
    return name[len(JOURNAL_PREFIX):-len(JOURNAL_SUFFIX)]


def contains(key) -> bool:
    """The bare membership test the offer loop uses, without a counter move.

    `lookup()` counts hits and misses for the witness; the offer loop probes the
    SAME chain for several groups and must not inflate those counters, so it asks
    this instead. It still resolves the identity through `split_block_key` - an
    ungrouped key is not present, and answering "present" to one would hand the
    offer loop exactly the ambiguity the identity exists to remove. Cheaper than
    `lookup()` by the counters and the dict build, not by the parsing.
    """
    try:
        split = disk_layout.split_block_key(key)
        if not split["ok"]:
            return False
        return identity_of(split["key16"], split["group_id"]) in _state["entries"]
    except Exception:  # noqa: BLE001 - a membership test cannot be fatal
        return False


def remove(key) -> dict:
    """Drop one entry, because its file is gone. Bounded and counted.

    Takes the same full group-qualified key as `lookup()`. Used by eviction notices
    that arrive in-process (the tests, and a future same-process spill) as well as
    by reasoning that a file this rank deleted is no longer restorable.
    """
    try:
        split = disk_layout.split_block_key(key)
        if not split["ok"]:
            return _fail(str(split["reason"]))
        identity = identity_of(split["key16"], split["group_id"])
        gone = _state["entries"].pop(identity, None)
        if gone is None:
            return _ok(removed=False, reason="no entry for this identity")
        _state["bytes_indexed"] = max(0, int(_state["bytes_indexed"])
                                      - int(gone["bytes"]))
        _state["removals"] += 1
        return _ok(removed=True, reason=None, bytes=int(gone["bytes"]),
                   key16=split["key16"], group_id=split["group_id"])
    except Exception as exc:  # noqa: BLE001
        snapshot._error(exc, "disk_index remove")
        return _fail(f"the removal raised {type(exc).__name__}")


def forget_all(reason: str | None = None) -> dict:
    """Empty the index. The honest answer to 'I can no longer trust this'."""
    count = len(_state["entries"])
    _state["entries"] = {}
    _state["bytes_indexed"] = 0
    _note(reason or "the index was cleared")
    return {"cleared": count, "reason": reason}


# --- the view ----------------------------------------------------------------

def section() -> dict:
    """The index's state for a snapshot or a verb. Reads nothing from disk."""
    return {
        "entries": int(len(_state["entries"])),
        "entry_cap": MAX_INDEX_ENTRIES,
        "entries_dropped": int(_state["entries_dropped"]),
        "bytes_indexed": int(_state["bytes_indexed"]),
        "journals_tracked": int(len(_state["cursors"])),
        "journals_stopped": sorted(_state["stopped"]),
        "journal_writes": int(_state["journal_writes"]),
        "journal_lines": int(_state["journal_lines"]),
        "journal_seq_seed": int(_state["journal_seq_seed"]),
        "journal_full": bool(_state["journal_full"]),
        "journal_write_errors": int(_state["journal_write_errors"]),
        "journal_fields": JOURNAL_FIELDS,
        "journal_cap_bytes": MAX_JOURNAL_BYTES,
        "refreshes": int(_state["refreshes"]),
        "applied_lines": int(_state["applied"]),
        "spills_seen": int(_state["spills"]),
        "removals_seen": int(_state["removals"]),
        "lookup_hits": int(_state["hits"]),
        "lookup_misses": int(_state["misses"]),
        "dropped": dict(_state["dropped"]),
        "journal_directory": journal_directory(),
        "last_reason": _state["last_reason"],
        "last_error": _state["last_error"],
        "units": {"entries": "identities", "entry_cap": "identities",
                  "entries_dropped": "identities", "bytes_indexed": "bytes",
                  "journals_tracked": "files", "journal_writes": "lines",
                  "journal_lines": "lines", "journal_write_errors": "writes",
                  "journal_seq_seed": "lines", "journal_fields": "fields",
                  "journal_cap_bytes": "bytes",
                  "refreshes": "refreshes", "applied_lines": "lines",
                  "spills_seen": "files", "removals_seen": "files",
                  "lookup_hits": "lookups", "lookup_misses": "lookups",
                  # `dropped` is a closed vocabulary (DROP_REASONS), one counter per
                  # reason, and the walk descends into it - so each reason needs its
                  # own unit rather than inheriting a parent's guess.
                  **{reason: "lines" for reason in DROP_REASONS},
                  # ...except the one that is not per-line: a shrunken journal is a
                  # whole file stopped, counted once per replay that noticed it.
                  "journal_shrank": "journals"},
        "note": ("the scheduler-side view of what THIS BOOT wrote to the SSD "
                 "tier, replayed from each rank's spill journal. Entries are "
                 "keyed by IDENTITY, `<key16>:g<group>` - a `key16` alone does "
                 "not say whose bytes it holds. It can be wrong only in the safe "
                 "direction: an entry dropped by the cap or lost to a truncated "
                 "journal costs a restore, and a file evicted but not yet "
                 "un-indexed costs a reported load failure (a recompute). No path "
                 "is published: four ranks name one identity four ways, so every "
                 "worker resolves the name IT would have written."),
    }


def record_spill(path, size_bytes) -> dict:
    """The door `disk_spill` calls after committing a file. Never raises.

    Takes a PATH rather than a key so the caller does not have to carry the key
    through the writer thread: the name IS the identity, and
    `disk_layout.parse_path` is the one parser of it - which is why the group half
    of the identity arrives with the path rather than being re-derived (or, in the
    first version of this seam, dropped).

    A refusal here is INVISIBLE to the spill that called it, by design: the write
    already happened, and the worst a journal can do is fail to mention it.
    """
    return _record_line(path, size_bytes, SPILL_LINE)


def record_removal(path, size_bytes=0) -> dict:
    """The door `disk_spill` calls after unlinking a file. Never raises."""
    return _record_line(path, size_bytes, REMOVE_LINE)


def _record_line(path, size_bytes, op) -> dict:
    try:
        parsed = disk_layout.parse_path(path)
        if not parsed["ok"]:
            return _fail(f"the file name is not ours to journal: "
                         f"{parsed['reason']}", reason_kind="not_ours")
        if parsed["partial"]:
            return _fail("an in-flight .part name is not a committed file",
                         reason_kind="partial")
        size = int(size_bytes or 0)
        return journal_append(op, parsed["key16"], size,
                              parsed["group_id"])
    except Exception as exc:  # noqa: BLE001 - a journal never breaks a write
        snapshot._error(exc, "disk_index record")
        return _fail(f"journaling raised {type(exc).__name__}",
                     reason_kind="journal_error")
