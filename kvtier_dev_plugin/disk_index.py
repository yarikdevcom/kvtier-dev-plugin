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

WHERE THE JOURNAL LIVES, AND WHY THAT IS A FILESYSTEM QUESTION
--------------------------------------------------------------
`journal_directory()` is the ONE read point for the path: the append, the tail-read
that seeds a second boot's sequence, and `refresh()`'s glob all compose it there.
It reads `VLLM_E2_TIER_DISK_JOURNAL_DIR` and falls back to `VLLM_E2_TIER_DIAG`
(the pre-0.2.19 behaviour, so nothing changes for an existing deployment). The
variable exists because the journal is the only map from a KV identity to a spilled
file and it cannot be rebuilt: on 2026-10-05 a reboot destroyed the journal in
`/tmp` while 107 GB of KV files survived on the RAID array, and the index came up at
`entries: 0`. Only the JOURNAL follows the new variable - the spill tier's ARM FILE
stays in the diagnostics directory and must keep dying at reboot, because a cold box
booting with the restore path armed is the precondition of a livelock, not a
convenience (see `disk_spill._read_arm_file`, which fails closed on its contents for
the same reason).

MAKING THE DIRECTORY IS PART OF READING THE PATH, AND IT IS NOT A SPILL FAILURE
-------------------------------------------------------------------------------
An operator who sets `VLLM_E2_TIER_DISK_JOURNAL_DIR=/home/user/kv-tier/journal`
before that path exists has pointed the tier at the exact failure the variable
exists to prevent, only self-inflicted: every journal append fails, the KV file
still commits, and the index never learns the file exists. So the ONE place a
journal path is composed also CREATES the directory
(`_ensure_journal_directory()`, called from `journal_path()` and nowhere else -
never inside `refresh()` or `_read_tail_seq()`, which read). Creation is
attempted ONCE per resolved directory and cached with its outcome, because the
append path is meant to be one `write()` of one line and not a filesystem walk;
a failed attempt is retried no more than once per `JOURNAL_DIR_RETRY_SECONDS`, so
a directory the operator fixes by hand starts working without a restart. It never
raises: a directory that cannot be made is a counted, named reason
(`journal_directory_writable: false`, `journal_directory_error`, and
`journal_dir=unwritable` on the boot line) and the spill still lands.

`journal_directory_state()` answers the same question WITHOUT creating anything,
which is what the boot line and every status view read.

`journal_fs_state()` asks `disk_layout.tmpfs_verdict` - the same tmpfs logic the
store root is refused by - whether that directory survives a reboot. On the DEFAULT
configuration the journal still lands in `/tmp`, so this can only be a warning:
logged once per episode (`notice_once`, the `miss_attribution` idiom), counted as
`journal_tmpfs_warnings`, and published as `journal_on_tmpfs` / `journal_dir=tmpfs`
on the boot line. It must never raise and must never refuse to spill.

THE READER SPENDS ITS BUDGET ON THE NEWEST LINES, NOT THE OLDEST
----------------------------------------------------------------
A journal is append-only, so a cursor that walks front-to-back hands the index the
spills the tier has ALREADY SERVED and leaves the recent ones - the ones a restore is
waiting for - to be reached last. On 2026-10-05 that meant 0 of the 36 newest
identities were known to the index while four journals sat ~60k lines deep and one
refresh carried a tenth of one of them. `refresh()` now takes each journal's window
from the END of its unread run, and gives every journal a SHARE of the byte and line
budgets instead of letting the first rank spend it all. Whatever a pass cannot carry is
forfeited - and forfeiting is safe ONLY because of the ordering it preserves: every
forfeited line is older than every applied line, so the applied stream is still exactly
the file's order (`+` then `-` still means gone), and the entries the skipped region
taught are forgotten outright. The caps are unchanged; what was missing was an
admission. `lag_bytes`, `lag_journals`, `catch_up_jumps`, `bytes_skipped` and
`lines_skipped` say how far behind the reader is and what it gave up to get there.

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

import logging
import os
import time

from . import config, disk_layout, snapshot

# --- the caps, and the unit of every one of them ----------------------------

MAX_INDEX_ENTRIES = 65_536          # KEYS held in memory (each drops the oldest)
MAX_JOURNAL_BYTES = 32 * 1024 * 1024  # BYTES one rank's journal may reach
MAX_JOURNAL_FILES = 64              # journal FILES one refresh will open
MAX_LINES_PER_REFRESH = 8_192       # LINES applied per refresh (bounded work)
MAX_READ_PER_REFRESH = 512 * 1024   # BYTES read across all journals per refresh
# THE FLOOR ON ONE JOURNAL'S SHARE of a refresh's budget. Below this a window is
# smaller than one rank's newest spill burst and the catch-up rule below would buy
# nothing; the TOTAL cap (`MAX_READ_PER_REFRESH`) is still what one refresh reads.
MIN_JOURNAL_WINDOW_BYTES = 4 * 1024  # BYTES: the smallest share one journal gets
# BYTES scanned forward from a mid-line offset to find the next line start. Bounded
# because it is a read on the lookup path; a journal whose lines are longer than
# this simply yields no window this pass (a missed restore, never a half-line).
CATCHUP_ALIGN_PROBE_BYTES = 64 * 1024
MAX_JOURNAL_RANK_CHARS = 32         # CHARS of a rank id in a journal file name
# HOW MANY RANKS one identity may be recorded as written by. A parked span is one
# file PER RANK (each names it with its own geometry digest), so the honest answer to
# "is this span on disk?" is a per-rank set, not a bool. This cap bounds that set for
# one identity against a hostile or misconfigured journal directory; a real boot has
# as many entries as it has ranks.
MAX_RANKS_PER_IDENTITY = 32         # RANKS remembered per identity
MAX_RANK_RECORDS = 262_144          # (identity, rank) PAIRS held in memory in total
# The label for a journal whose file name carries no rank (`disk-spill-journal-.jsonl`).
# A contribution we cannot attribute is still a contribution; naming it `None` would
# make it indistinguishable from "no rank" in a payload.
UNNAMED_RANK = "unnamed"
MAX_STRING_CHARS = 160              # CHARS of a path/reason kept in state
REFRESH_MIN_INTERVAL_SECONDS = 1.0  # SECONDS between journal refreshes

# How many distinct tmpfs notices one process may log. The idiom is
# `miss_attribution.notice_once`: a dashboard polls the disk verb every few
# seconds, so a per-poll warning would drown the log it is trying to write.
MAX_FS_NOTICES = 32

# The boot-line / snapshot token for the journal directory's filesystem. Fixed
# vocabulary: a grep has to be able to match it.
FS_TMPFS = "tmpfs"
FS_DISK = "disk"
FS_UNSET = "unset"
FS_UNKNOWN = "unknown"
# "We tried, and we cannot put the journal there." Deliberately NOT `unknown`:
# `unknown` means the mount table could not tell us, which asks for a look; this
# means the first journal line cannot be written at all, which asks for a chmod.
# Conflating them loses exactly the distinction that matters at 3 a.m.
FS_UNWRITABLE = "unwritable"

# The mode of a journal directory this plugin creates. It is NOT invented here:
# `disk_layout.write_committed_file` gives the store's own shard directories
# 0o750 (disk_layout.py:1696), and the journal sits on the same spill path, so
# one box has one convention for a directory the spill tier made for itself.
JOURNAL_DIR_MODE = 0o750

# SECONDS a FAILED directory attempt is believed before it is retried. A success
# is cached for the life of the process (the append path owes one `write()`); a
# failure is retried slowly so an operator who chmods or mkdirs the path
# mid-boot heals the tier without a restart, while a permanently-wrong path costs
# the writer thread one `makedirs` a minute and nothing else.
JOURNAL_DIR_RETRY_SECONDS = 60.0

# HOW FAR UP to look for an existing ancestor when the journal directory does not
# exist yet (a writability probe, not a walk that creates anything).
MAX_ANCESTOR_WALK = 32

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

# THE ANSWER `membership()` gives, a closed vocabulary. The distinction that matters
# is between "I looked and it is not there" and "I have never looked" - on
# 2026-10-05 `entries: 0` was read as "my span is not on disk" for an afternoon when
# the real fact was that no journal had been replayed into this process yet.
VERDICT_PRESENT = "present"
VERDICT_NOT_PRESENT = "not_present"
VERDICT_ENTRY_DROPPED = "entry_dropped_but_rank_lines_remain"
VERDICT_NEVER_REFRESHED = "index_never_refreshed"
VERDICT_NO_JOURNAL_DIR = "index_has_no_journal_directory"
VERDICT_INDEX_EMPTY = "index_empty_no_spill_lines"
VERDICT_KEY_REJECTED = "key_rejected"
VERDICT_CHECK_FAILED = "membership_check_failed"
MEMBERSHIP_VERDICTS = (VERDICT_PRESENT, VERDICT_NOT_PRESENT, VERDICT_ENTRY_DROPPED,
                       VERDICT_NEVER_REFRESHED, VERDICT_NO_JOURNAL_DIR,
                       VERDICT_INDEX_EMPTY, VERDICT_KEY_REJECTED,
                       VERDICT_CHECK_FAILED)
"""Every answer `membership()` may give. FIVE of them mean "I cannot answer", not
"it is not there", and answer `present: null` for exactly that reason: a key that
cannot name an identity, a journal directory this process was never pointed at, an
index that has replayed nothing yet, an index whose replay named no file at all (an
empty map over a store that may be full - the 2026-10-05 trap read directly), and a
check that raised. On 2026-10-05 an `entries: 0` was read as "my span is not on disk"
and cost an afternoon, when the fact was that nothing had been replayed into the
process yet."""

# Why a line was not applied. A closed vocabulary, one counter each.
DROP_REASONS = ("malformed_line", "bad_op", "bad_key", "bad_size", "bad_seq",
                "bad_group", "journal_shrank", "journal_capped", "line_cap",
                "read_cap", "stopped")

_state: dict = {
    "entries": {},                    # identity -> entry dict (insertion ordered)
    # identity -> {rank -> {bytes, journal, seq, seen_ts}}. `entries` is what the OFFER
    # path asks (one claim per identity, the last journal applied wins); this is the
    # SAME replay seen per rank, the only way to answer "is the whole span there, or
    # two ranks of four?" without walking the store.
    "rank_lines": {},
    "rank_records": 0,                # (identity, rank) PAIRS held
    "rank_records_dropped": 0,        # pairs refused by a cap
    "membership_checks": 0,           # `op=contains` answers given here
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
    # ONE-LINE reason of the last journal failure, beside the counter that counts
    # them: the counter says the index is falling behind, this says why, in the
    # same payload, without a log search.
    "journal_last_error": None,
    # THE DIRECTORY ITSELF: the one cached creation attempt for the directory this
    # process resolved, and how many attempts/failures it has taken.
    "journal_dir": None,
    "journal_directory_attempts": 0,
    "journal_directory_errors": 0,
    "journal_full": False,
    "journal_lines": 0,
    "journal_bytes": 0,
    "journal_seeded": False,
    "journal_seq_seed": 0,
    "refreshes": 0,
    "refresh_skipped": 0,
    # HOW FAR BEHIND THIS INDEX IS, in the unit the reader can actually measure.
    # `lag_bytes` is the unread tail summed over every journal AFTER this refresh
    # applied what it could; it reaches 0 when every cursor is at end-of-file. It is
    # bytes and not lines because counting lines would mean reading the very bytes
    # the lag is made of, and this module does not guess numbers it has not read.
    "lag_bytes": 0,
    "lag_journals": 0,
    # WHAT THE NEWEST-FIRST RULE FORFEITED: lines and bytes passed over without being
    # applied, plus how many times a journal had to jump forward to reach its tail.
    # A jump is not free - it forgets what that rank taught us - so it is counted.
    "bytes_skipped": 0,
    "lines_skipped": 0,
    "catch_up_jumps": 0,
    "last_reason": None,
    "last_error": None,
    "last_journal": None,
    "last_refresh_mono": None,
    "bytes_indexed": 0,
    # THE JOURNAL'S FILESYSTEM, cached per directory (a mount table does not move
    # within a boot, and `journal_directory()` is read on the append path).
    "journal_fs": None,
    "journal_on_tmpfs": None,
    "journal_tmpfs_warnings": 0,
    "journal_fs_directory": None,
}

# Logged-once-per-episode keys, the `miss_attribution.notice_once` idiom.
_notices: set = set()
logger = logging.getLogger("kvtier_dev_plugin.disk_index")


def reset_state_for_tests() -> None:
    """Back to a fresh boot: empty index, no cursors, no counters.

    The whole module is per-process memory, so this is the only teardown it
    needs; there is no thread and no open descriptor to stop.
    """
    _state.update({
        "entries": {}, "rank_lines": {}, "rank_records": 0,
        "rank_records_dropped": 0, "membership_checks": 0,
        "cursors": {}, "seq": {}, "stopped": {},
        "applied": 0, "spills": 0, "removals": 0, "misses": 0, "hits": 0,
        "entries_dropped": 0,
        "dropped": {reason: 0 for reason in DROP_REASONS},
        "journal_writes": 0, "journal_write_errors": 0,
        "journal_last_error": None, "journal_dir": None,
        "journal_directory_attempts": 0, "journal_directory_errors": 0,
        "journal_full": False,
        "journal_lines": 0, "journal_bytes": 0,
        "journal_seeded": False, "journal_seq_seed": 0,
        "refreshes": 0, "refresh_skipped": 0,
        "lag_bytes": 0, "lag_journals": 0,
        "bytes_skipped": 0, "lines_skipped": 0, "catch_up_jumps": 0,
        "last_reason": None, "last_error": None, "last_journal": None,
        "last_refresh_mono": None, "bytes_indexed": 0,
        "journal_fs": None, "journal_on_tmpfs": None,
        "journal_tmpfs_warnings": 0, "journal_fs_directory": None,
    })
    _notices.clear()


def counters() -> dict:
    """A copy for a payload. The ENTRIES are never included (only their count).

    Neither is the PER-RANK table, for the same reason and one more: it is a live nested
    dict of the index's own memory, and a payload that hands one out can be mutated by
    whoever receives it. Counts, always counts.
    """
    out = dict(_state)
    out["entries"] = int(len(_state["entries"]))
    out["rank_lines"] = int(len(_state["rank_lines"]))
    return out


# --- small helpers ------------------------------------------------------------

def _note(reason) -> None:
    _state["last_reason"] = str(reason)[:MAX_STRING_CHARS]


def notice_once(key: str, message: str) -> bool:
    """Log `message` at most once per key per process. Bounded, never raises.

    Returns True when it actually logged, so a caller can count the episode and
    not the call. The repo's idiom (`miss_attribution.notice_once`): these views
    are polled every few seconds by a dashboard, and a warning that repeats becomes
    noise an operator stops reading - which is how tonight's silence stayed silent.
    """
    try:
        if key in _notices:
            return False
        if len(_notices) >= MAX_FS_NOTICES:
            _notices.clear()
        _notices.add(key)
        logger.warning("[KVTIER_DISK] %s", str(message)[:MAX_STRING_CHARS * 4])
        return True
    except Exception as exc:  # noqa: BLE001 - a log line is never worth a fault
        snapshot._error(exc, "disk_index notice")
        return False


def journal_fs_state() -> dict:
    """The journal directory's FILESYSTEM, and the loud part of a tmpfs answer.

    WHY THIS IS A WARNING AND NOT A REFUSAL: a journal in `/tmp` works perfectly -
    until the box reboots, when the map dies and the KV files it describes do not.
    That is the 2026-10-05 failure: 107 GB of spilled KV on the RAID array and an
    index that came up at `entries: 0`, with not one line in the log saying the map
    was on a filesystem that dies at reboot. So: say it once, count it, publish it
    (`journal_on_tmpfs` here, `journal_dir=tmpfs` on the boot line), and NEVER
    refuse to spill. A degraded-but-loud tier beats a silent one in either
    direction, and a tier that refuses to write because of where its INDEX lives
    would be a diagnostic that causes the outage it is measuring.

    The verdict itself is `disk_layout.tmpfs_verdict` - the same tmpfs logic the
    store root is refused by - cached per directory, because a mount table does not
    move within a boot and this is read by the snapshot, the verb and the boot
    line. It cannot raise: an unreadable mount table is `on_tmpfs: None`, which is
    reported as `journal_dir=unknown` and warned about never.

    WRITABILITY TRUMPS FILESYSTEM, and the two are published apart. `writable:
    false` means the first journal line cannot be written there at all, so the
    boot line answers `journal_dir=unwritable` rather than going on to claim
    `disk`: "we tried and failed" and "we could not tell" (`unknown`) are
    different facts and get different words. The tmpfs half of the answer stays
    published next to it, so nothing is lost by the louder one winning the field.
    """
    directory = journal_directory()
    cached = _state.get("journal_fs")
    if directory is None:
        if _state["journal_fs_directory"] != "__unset__":
            _state["journal_fs"] = None
            _state["journal_on_tmpfs"] = None
            _state["journal_fs_directory"] = "__unset__"
        return {"directory": None, "on_tmpfs": None, "method": "unset",
                "fstype": None, "mount_point": None, "warned": False,
                "label": FS_UNSET,
                "writable": None, "writable_basis": "unset",
                "directory_error": None,
                "reason": f"{config.ENV_DISK_JOURNAL_DIR} is not set, and neither "
                          f"is {config.ENV_DIAG_DIR}: there is no directory to "
                          "write a spill journal into"}
    if isinstance(cached, dict) and cached.get("directory") == directory:
        verdict = cached
    else:
        verdict = disk_layout.tmpfs_verdict(directory)
        verdict["directory"] = directory
        _state["journal_fs"] = verdict
        _state["journal_on_tmpfs"] = verdict.get("on_tmpfs")
        _state["journal_fs_directory"] = directory
    on_tmpfs = verdict.get("on_tmpfs")
    # READ-ONLY (it creates nothing) and deliberately NOT cached: an operator who
    # chmods the path mid-boot must see the view change on the next poll.
    directory_state = journal_directory_state()
    writable = directory_state.get("writable")
    label = (FS_UNWRITABLE if writable is False else
             FS_TMPFS if on_tmpfs is True else
             FS_UNKNOWN if on_tmpfs is None else FS_DISK)
    warned = False
    if on_tmpfs is True:
        _note(f"the spill journal is on tmpfs ({verdict.get('mount_point')}), "
              "so the restore index dies at reboot while the KV files it "
              "describes survive")
        if notice_once(
                f"journal-tmpfs:{directory}",
                f"the spill JOURNAL directory is on a filesystem that dies at "
                f"reboot: {directory} sits on "
                f"{verdict.get('fstype') or verdict.get('mount_point')} "
                f"(method {verdict.get('method')}). The spilled KV files on the "
                "store root survive a reboot; this journal does not, and it is "
                "the only map from a KV identity to those files, so the restore "
                "index will come up at entries: 0 after one. Set "
                f"{config.ENV_DISK_JOURNAL_DIR} to a path on persistent storage "
                "to keep the map (the arm file deliberately stays in the "
                "diagnostics directory and dies at boot, as it must)."):
            _state["journal_tmpfs_warnings"] += 1
            warned = True
    return {"directory": directory, "on_tmpfs": on_tmpfs,
            "method": verdict.get("method"), "fstype": verdict.get("fstype"),
            "mount_point": verdict.get("mount_point"),
            "absolute": verdict.get("absolute"),
            "warned": warned, "label": label,
            # CAN A JOURNAL LINE BE WRITTEN HERE AT ALL, and how do we know.
            # `writable: false` outranks `label`'s filesystem answer: a map that
            # cannot be written is a worse fact than a map that will not survive a
            # reboot, and `directory_error` says what to do about it.
            "writable": writable,
            "writable_basis": directory_state.get("basis"),
            "directory_error": directory_state.get("reason"),
            "warnings": int(_state["journal_tmpfs_warnings"]),
            "units": {"warnings": "warnings"},
            "reason": verdict.get("reason")}


def journal_fs_label() -> str:
    """One token for the boot line: `tmpfs` / `disk` / `unset` / `unknown` /
    `unwritable`.

    `unwritable` is the newest of the five and the one an operator can act on: it
    means the directory is not there and cannot be made (or is there and cannot be
    written), so NO journal line will ever land there. It is deliberately not
    folded into `unknown`, which means "we could not tell" and invites a look
    rather than a chmod.

    Never raises: a boot line that cannot print a field prints the field's failure,
    not no line at all.
    """
    try:
        return str(journal_fs_state().get("label") or FS_UNKNOWN)
    except Exception as exc:  # noqa: BLE001 - a boot line never fails on a field
        snapshot._error(exc, "disk_index journal fs label")
        return FS_UNKNOWN


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
    """Where journals live: the journal directory, else the diagnostics one, else None.

    THE ONE READ POINT FOR THAT PATH. `journal_path()` (the append), `_read_tail_seq`
    (the sequence seed a second boot needs) and `refresh()` (the glob over every
    rank's journal) all compose their path here, so the journal filename has exactly
    one composition site - there is no second place to keep in sync, and a grep for
    `JOURNAL_PREFIX` finds only name-handling, never a path build.

    `VLLM_E2_TIER_DISK_JOURNAL_DIR` lets the journal be put on a filesystem that
    survives a reboot while the rest of the diagnostics directory stays where it is.
    UNSET means the diagnostics directory, exactly as before 0.2.19: the same
    directory the rank files, the arm file and the dump receipts use, so the
    operator can still name ONE place and every cross-process channel of the plugin
    goes there. The arm file deliberately does NOT follow this variable - an arm has
    to die at reboot, and `disk_spill._read_arm_file` fails closed on its own
    contents as well. Unset both and the restore path has no way to learn what the
    workers spilled, which is reported, not assumed.
    """
    directory = config.disk_journal_dir()
    return None if directory is None else str(directory)


def _probe_journal_directory(directory: str) -> dict:
    """Could a journal be written at `directory`, judged WITHOUT creating it?

    Two shapes, both cheap (one `isdir`, one `access`, plus one `access` per
    missing level bounded by `MAX_ANCESTOR_WALK`):

      * the directory EXISTS: the question is whether this process can write into
        it, which an `makedirs(exist_ok=True)` would happily answer "yes" to and
        the `open()` would then answer with an errno - so ask directly;
      * it does NOT exist yet: the question belongs to the nearest directory that
        does, because that is where the first `makedirs` has to land.

    `writable: None` means "nothing in the chain could be examined" - reported as
    `unknown`, never as `unwritable`, because a failure we did not observe is not
    a failure we may claim. Never raises: a probe that cannot answer says so.
    """
    try:
        text = str(directory)
        if os.path.isdir(text):
            if os.access(text, os.W_OK | os.X_OK):
                return {"exists": True, "writable": True, "reason": None}
            return {"exists": True, "writable": False,
                    "reason": f"{text} exists but this process cannot write into "
                              "it (EACCES): no spill journal line can be created "
                              "there"}
        parent = os.path.dirname(os.path.normpath(text))
        steps = 0
        while parent and steps < MAX_ANCESTOR_WALK:
            if os.path.isdir(parent):
                if os.access(parent, os.W_OK | os.X_OK):
                    return {"exists": False, "writable": True, "reason": None}
                return {"exists": False, "writable": False,
                        "reason": f"{text} does not exist and {parent} cannot be "
                                  "written to by this process (EACCES), so the "
                                  "journal directory cannot be created"}
            if os.path.exists(parent):
                # Something that is NOT a directory sits on the way down, and no
                # amount of writability makes a path through it creatable
                # (`makedirs` answers NotADirectoryError). Without this the walk
                # would hop over the obstruction, find a writable ancestor and
                # report a directory that can never exist as `writable: true`.
                return {"exists": False, "writable": False,
                        "reason": f"{parent} exists and is not a directory, so "
                                  f"{text} cannot be created"}
            if os.path.dirname(parent) == parent:
                break
            parent = os.path.dirname(parent)
            steps += 1
        return {"exists": False, "writable": None,
                "reason": "no existing directory was found above the journal "
                          "directory, so its writability could not be told"}
    except Exception as exc:  # noqa: BLE001 - a probe that cannot tell says so
        snapshot._error(exc, "disk_index journal dir probe")
        return {"exists": None, "writable": None,
                "reason": f"the journal directory could not be examined: "
                          f"{type(exc).__name__}"}


def _ensure_journal_directory(directory: str) -> dict:
    """Create the journal directory, ONCE per resolved directory. Never raises.

    WHY IT EXISTS: `VLLM_E2_TIER_DISK_JOURNAL_DIR` names a path the operator may
    not have made yet, and an append into a directory that is not there fails -
    quietly, because `journal_append` never raises - which leaves a store full of
    committed KV files that no index can name. Making the directory is therefore
    part of composing the path, not a boot-time ceremony.

    WHY IT IS CACHED: the append runs on `disk_spill`'s writer thread and is meant
    to cost one `write()` of one line. A successful attempt is remembered for the
    life of the process (the cache hit does no filesystem call at all); a FAILED
    one is retried when a cheap `access` shows the path has become writable, and
    otherwise at most once per `JOURNAL_DIR_RETRY_SECONDS`, so a path the operator
    fixes mid-boot resumes journaling on the next line and a hopeless path costs
    one `makedirs` a minute.

    The outcome is the record `journal_directory_state()` reports on an `attempt`
    basis: this is the only place in the plugin that creates a journal directory.
    """
    cached = _state.get("journal_dir")
    now = time.monotonic()
    if isinstance(cached, dict) and cached.get("directory") == directory:
        if cached.get("ok"):
            return cached
        tried = cached.get("attempted_mono")
        inside = bool(isinstance(tried, (int, float))
                      and (now - float(tried)) < JOURNAL_DIR_RETRY_SECONDS)
        # The clock bounds the expensive part (`makedirs`), not the recovery: one
        # `access` says whether the operator has fixed the path under us, and if
        # they have, the attempt happens NOW rather than up to
        # `JOURNAL_DIR_RETRY_SECONDS` from now. A tier that stayed blind for a
        # minute after a chmod would be its own small incident.
        if inside and not _probe_journal_directory(directory).get("writable"):
            return cached
    outcome = {"directory": directory, "ok": False, "created": False,
               "reason": None, "attempted_mono": now}
    _state["journal_directory_attempts"] += 1
    try:
        if os.path.isdir(directory):
            # Not `makedirs(exist_ok=True)`: that answers "already there" for a
            # directory this process may not be allowed to write into, and the
            # `open()` would be the one to report the errno - three layers away.
            if os.access(directory, os.W_OK | os.X_OK):
                outcome["ok"] = True
            else:
                outcome["reason"] = (f"{directory} exists but this process cannot "
                                     "write into it (EACCES)")
        else:
            os.makedirs(directory, mode=JOURNAL_DIR_MODE, exist_ok=True)
            outcome["created"] = True
            outcome["ok"] = True
    except Exception as exc:  # noqa: BLE001 - a journal directory is never fatal
        outcome["reason"] = (f"{type(exc).__name__}: "
                             f"{str(exc)[:MAX_STRING_CHARS - 40]}")
    _state["journal_dir"] = outcome
    if not outcome["ok"]:
        _state["journal_directory_errors"] += 1
        notice_once(f"journal-unwritable:{directory}",
                    f"the spill JOURNAL directory cannot be created or written: "
                    f"{directory} ({outcome['reason']}). Every spill this rank "
                    "makes still lands on the store, but nothing about it is "
                    "recorded, so the scheduler's restore index will never see "
                    "those files (a recompute, and a store that grows with "
                    "unreachable bytes). Create the directory, or point "
                    f"{config.ENV_DISK_JOURNAL_DIR} somewhere writable.")
    return outcome


def journal_directory_state() -> dict:
    """Can a spill journal be written where `journal_directory()` points?

    The READ-ONLY answer, for the boot line and every status view: it creates
    nothing. Two kinds of evidence, kept apart on purpose, because "we tried and
    failed", "we can see it cannot work" and "we cannot tell" are three different
    facts at 3 a.m.:

      * `basis: "attempt"` - this process already tried to prepare the directory
        on the way to a journal line, and a real errno beats any prediction;
      * `basis: "probe"` - nobody has written a line yet, so an `os.access` on the
        directory (or on its nearest existing parent) answers instead;
      * `writable: None` - neither could tell: that is `unknown`, and it must
        never be reported as `unwritable`.

    A failed attempt that the probe now contradicts (the operator made the
    directory by hand) reports the probe, so the view heals while the process
    keeps running. Never raises.
    """
    try:
        directory = journal_directory()
        if directory is None:
            return {"directory": None, "exists": False, "writable": None,
                    "basis": "unset", "reason": None}
        probed = _probe_journal_directory(directory)
        attempted = _state.get("journal_dir")
        if (isinstance(attempted, dict)
                and attempted.get("directory") == directory):
            if attempted.get("ok"):
                return {"directory": directory, "exists": True,
                        "writable": True, "basis": "attempt", "reason": None}
            if probed.get("writable") is True:
                return {"directory": directory,
                        "exists": bool(probed.get("exists")),
                        "writable": True, "basis": "probe", "reason": None}
            return {"directory": directory,
                    "exists": bool(probed.get("exists")), "writable": False,
                    "basis": "attempt",
                    "reason": attempted.get("reason") or probed.get("reason")}
        return {"directory": directory,
                "exists": bool(probed.get("exists")),
                "writable": probed.get("writable"),
                "basis": "probe", "reason": probed.get("reason")}
    except Exception as exc:  # noqa: BLE001 - a view is never worth a fault
        snapshot._error(exc, "disk_index journal directory state")
        return {"directory": None, "exists": None, "writable": None,
                "basis": "unknown",
                "reason": f"the journal directory state raised "
                          f"{type(exc).__name__}"}


def _journal_failure(reason) -> None:
    """Record one counted, NAMED journal failure. Never raises.

    THE SILENCE IS THE BUG THIS EXISTS TO KILL. A spill whose journal line fails
    costs disk space; refusing that spill would cost the whole tier for a logging
    problem - so the tier keeps writing and the failure becomes impossible to
    miss: `journal_write_errors` for a dashboard to alert on, `journal_last_error`
    for the human reading the same payload, both published by `section()`, the
    `/disk` verb and the snapshot.
    """
    text = str(reason)[:MAX_STRING_CHARS]
    _state["journal_write_errors"] += 1
    _state["journal_last_error"] = text
    _state["last_error"] = text
    _note(text)


def journal_path(rank=None) -> dict:
    """`(path | None, why)` for one rank's journal.

    `rank` defaults to this process's own rank (the ONE rule lives in
    `worker_diag.rank_id()`, so a worker's journal is named by the same id its
    rank file uses). A scheduler process has no rank and must pass one to read
    a specific file; `refresh()` globs instead.

    THIS IS ALSO THE ONLY PLACE THE JOURNAL DIRECTORY IS CREATED. Every journal
    path - append, tail-seed, the refresh glob - is composed here or from
    `journal_directory()`, so making the directory once, at the one composition
    point, is what keeps "written here" and "read from here" the same place;
    creating it inside `journal_path()` AND `refresh()` AND `_read_tail_seq()`
    would be three sites to keep in sync for no extra guarantee. See
    `_ensure_journal_directory()` for the once-per-directory cache.
    """
    directory = journal_directory()
    if directory is None:
        return _fail(f"{config.ENV_DISK_JOURNAL_DIR} is not set, and neither is "
                     f"{config.ENV_DIAG_DIR}: there is no directory "
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
    # The directory comes with the path: a journal line written into a directory
    # that does not exist is a line that was never written, and the spill that
    # asked for it does not fail - it just becomes invisible to the index.
    prepared = _ensure_journal_directory(directory)
    if not prepared["ok"]:
        return _fail(f"the spill journal directory {directory} could not be "
                     f"prepared: {prepared['reason']}", path=None,
                     reason_kind="journal_unwritable")
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


def _write_journal_line(path, line: str) -> tuple:
    """Append one line. `(ok, why)`, and the ONLY place a journal line is written.

    Returns the errno NAME on failure so the published reason names the syscall's
    verdict instead of guessing at it. Never raises: the caller is a spill, and the
    whole contract of this module is that its own failures are data.
    """
    try:
        with open(path, "ab") as handle:
            handle.write(line.encode("utf-8"))
        return True, None
    except OSError as exc:  # noqa: BLE001 - reported, never re-raised
        return False, type(exc).__name__


def journal_append(op, key, size_bytes=None, group_id=None) -> dict:
    """Append one line to THIS process's journal. Bounded; never raises.

    Called from `disk_spill`'s writer thread (the only thread that knows a file
    was committed, and the only one that deletes one), never from an engine
    thread. One `write()` of one line, so a reader that stops at the last
    newline can never apply half a record - which is also why the DIRECTORY is
    not created here but at the one path-composition point (`journal_path`),
    once per resolved directory and cached: an append owes no filesystem walk.

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
            _journal_failure(named["reason"])
            # `no_channel` is the historical reason (nothing is configured at all);
            # a directory that exists but cannot be made or written names itself
            # `journal_unwritable` instead, because the operator's fix is a chmod,
            # not an export.
            return _fail(named["reason"],
                         reason_kind=named.get("reason_kind") or "no_channel")
        path = named["path"]
        try:
            current = os.path.getsize(path)
        except FileNotFoundError:
            current = 0
        except OSError as exc:
            _journal_failure(f"the journal could not be stat'ed: "
                             f"{type(exc).__name__}")
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
                _journal_failure(f"the journal's sequence could not be seeded: "
                                 f"{seeded['reason']}")
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
        written, why = _write_journal_line(path, line)
        if not written:
            # A cached "the directory is there" can be wrong - an operator clearing
            # a tmpfs, a mount that moved - and a dropped line drops the FACT for
            # good, so this is the one place that pays a second filesystem trip:
            # forget the cache, which makes `journal_path()` prepare the directory
            # again, and write the line once more. Only a second failure is one.
            _state["journal_dir"] = None
            again = journal_path()
            if again["ok"]:
                path = again["path"]
            written, why = _write_journal_line(path, line)
        if not written:
            _journal_failure(f"the journal line could not be written: {why}")
            return _fail(f"the journal line could not be written: {why}",
                         reason_kind="journal_error", path=path)
        _state["journal_writes"] += 1
        _state["journal_lines"] = seq
        _state["journal_bytes"] = current + len(line)
        _state["last_journal"] = path
        return _ok(path=path, seq=seq, bytes=len(line), rank=named["rank"])
    except Exception as exc:  # noqa: BLE001 - a journal never breaks a spill
        _journal_failure(f"journaling raised {type(exc).__name__}")
        snapshot._error(exc, "disk_index journal_append")
        return _fail(f"journaling raised {type(exc).__name__}",
                     reason_kind="journal_error")


# --- the reader: replay every rank's journal into the index -----------------

def refresh(*, force: bool = False) -> dict:
    """Apply every rank's new journal lines. Cheap, bounded, never raises.

    Called from the lookup path, so the whole refresh is bounded by
    `MAX_JOURNAL_FILES` files, `MAX_READ_PER_REFRESH` bytes and
    `MAX_LINES_PER_REFRESH` lines, and each file's read is bounded by its share of
    the same budget. A refresh opens at most one `stat` per journal and reads only
    the bytes after its cursor, so an idle store costs one stat per rank.

    THE BUDGET IS SPENT ON THE NEWEST LINES, NOT THE OLDEST, and that is the whole
    point of this function. A journal is append-only, so a front-to-back cursor hands
    the index the SPILLS THE TIER HAS ALREADY SERVED and leaves the lines the restore
    path is waiting for - the ones written minutes ago - to be reached last. On
    2026-10-05 that cost 0 of 36 newest identities: four journals ~60k lines deep,
    one refresh covering a tenth of one of them, and the offer path asking about the
    span it had parked seconds earlier. The rule now is: read what the budget
    carries, taking the END of the unread run; forfeit whatever that leaves behind,
    count it, and say so. Forfeiting is the SAFE direction because every line left
    unread is OLDER than every line applied (see `_apply_journal`), and the rank's
    previously-learned entries are dropped when a jump forfeits them, so the index
    never keeps a claim it did not just read.

    `force` ignores the age floor (a verb that wants the truth right now); the
    lookup path uses the floor so a burst of requests does not turn into a burst
    of stats.
    """
    started = time.monotonic()
    out = {"ok": True, "reason": None, "journals": 0, "lines": 0, "spills": 0,
           "removals": 0, "dropped": 0, "entries": int(len(_state["entries"])),
           "applied": False, "lag_bytes": 0, "lag_journals": 0,
           "bytes_skipped": 0, "lines_skipped": 0, "jumps": 0}
    try:
        directory = journal_directory()
        if directory is None:
            out["reason"] = (f"{config.ENV_DISK_JOURNAL_DIR} is not set, and "
                             f"neither is {config.ENV_DIAG_DIR}: no spill journal "
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
            # The last MEASURED lag, not zero: a refresh that did not look must not
            # report "caught up". `lag_measured: false` tells the reader this number
            # is the previous pass's, which is the honest form of "we did not look".
            out["lag_bytes"] = int(_state["lag_bytes"])
            out["lag_journals"] = int(_state["lag_journals"])
            out["lag_measured"] = False
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
        # EVERY JOURNAL GETS A SHARE, not whatever the previous one left over. One
        # rank reading the whole budget is one rank reading NOTHING, and a span needs
        # every rank to have journalled it: a partially-caught-up index is the one
        # shape that turns "the file is gone" into a offered load that fails. The
        # TOTAL caps are still what one refresh may read and apply; the share only
        # decides how that budget is divided.
        share_bytes = max(MIN_JOURNAL_WINDOW_BYTES,
                          MAX_READ_PER_REFRESH // max(1, len(names)))
        share_lines = max(1, MAX_LINES_PER_REFRESH // max(1, len(names)))
        for name in names:
            path = os.path.join(directory, name)
            out["journals"] += 1
            want_bytes = min(share_bytes, budget)
            want_lines = min(share_lines, lines)
            if want_bytes <= 0 or want_lines <= 0:
                _state["dropped"]["read_cap"] += 1
                out["dropped"] += 1
                continue
            result = _apply_journal(path, want_bytes, want_lines)
            budget -= int(result["read"])
            lines -= int(result["applied"])
            out["lines"] += int(result["applied"])
            out["spills"] += int(result["spills"])
            out["removals"] += int(result["removals"])
            out["lag_bytes"] += int(result["lag_bytes"])
            out["bytes_skipped"] += int(result["bytes_skipped"])
            out["lines_skipped"] += int(result["lines_skipped"])
            out["jumps"] += int(result["jumps"])
            if int(result["lag_bytes"]) > 0:
                out["lag_journals"] += 1
        _state["lag_bytes"] = int(out["lag_bytes"])
        _state["lag_journals"] = int(out["lag_journals"])
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
        _note(f"the journal directory could not be listed: "
              f"{type(exc).__name__}")
        return []
    names.sort()
    if len(names) > MAX_JOURNAL_FILES:
        _note(f"{len(names)} journal files exist; only the first "
              f"{MAX_JOURNAL_FILES} are read")
        names = names[:MAX_JOURNAL_FILES]
    return names


def _align_forward(path, start: int, size: int) -> int:
    """The first line boundary at or after `start`, or `size` if there is none.

    WHY THIS EXISTS: a window taken from mid-file starts mid-line, and a half line
    is a MALFORMED line - which `_apply_line` answers by stopping that journal for
    the rest of the boot. So a tail window is worthless unless it begins on a line
    boundary, and this is the one read that pays for the privilege: at most
    `CATCHUP_ALIGN_PROBE_BYTES`, looking for a newline.

    `start > 0` is checked against the byte BEFORE it, because "the byte before the
    window is a newline" is exactly the statement "the window starts a line".
    """
    if start <= 0:
        return 0
    if start >= size:
        return size
    probe = min(CATCHUP_ALIGN_PROBE_BYTES, max(1, size - start + 1))
    try:
        with open(path, "rb") as handle:
            handle.seek(start - 1)
            raw = handle.read(probe)
    except OSError as exc:
        _note(f"{os.path.basename(path)} could not be read to align a catch-up "
              f"window: {type(exc).__name__}")
        return size
    if not raw:
        return size
    if raw[0] == 0x0A:
        return start
    found = raw.find(b"\n", 1)
    if found < 0:
        # No line start inside the probe: the window would open mid-line, and an
        # unread pass is better than a journal stopped by its own first half-line.
        return size
    return start + found


def _apply_journal(path, budget, line_budget) -> dict:
    """Replay the NEWEST part of one journal that `budget` can carry.

    Returns `{read, applied, spills, removals, lag_bytes, bytes_skipped,
    lines_skipped, jumps}`.

    Only COMPLETE lines (ending in a newline) are applied, and the cursor only ever
    advances to the last newline: a line the writer is halfway through is nobody's
    fact yet. A file that got SMALLER was truncated by someone who is not this
    plugin - the entries contributed by that rank are dropped, because we can no
    longer tell what they were, and the replay starts from the top.

    NEWEST-FIRST, AND WHY IT IS THE SAFE DIRECTION. When the unread run is bigger
    than this pass can carry, the window is taken from the END of the file and the
    older bytes are forfeited - and forfeiting is only safe because of the ordering
    it preserves: every forfeited line is OLDER than every applied line, so the
    applied stream is still exactly the file's order. A `+` we read is a file we saw
    written; a `-` we read for an identity we never learned is a no-op; a `-` we
    forfeit can only leave an entry we have JUST forgotten, which is why
    `_forget_rank(path)` runs before the window is applied. Nothing this does can
    manufacture a presence the journal did not describe in the window it read.

    The same rule applies inside a window when `line_budget` binds: the OLDEST lines
    of the window are the ones forfeited.
    """
    out = {"read": 0, "applied": 0, "spills": 0, "removals": 0, "lag_bytes": 0,
           "bytes_skipped": 0, "lines_skipped": 0, "jumps": 0}
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return out
    except OSError as exc:
        # A stat failure is REPORTED, never counted as a data loss: the cursor
        # stays where it is, so the next refresh tries the same bytes again.
        _note(f"{os.path.basename(path)} could not be stat'ed: "
              f"{type(exc).__name__}")
        return out
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
        return out
    if size <= cursor:
        return out
    window = min(max(int(budget), 0), size)
    if window <= 0:
        _state["dropped"]["read_cap"] += 1
        return out
    if size - cursor > window:
        # BURIED. Reach the tail and forget the rest of this rank's history, which
        # is the only way the index can be honest about a claim it did not read.
        # The sequence is deliberately NOT reset: it still increases, and a reset
        # would let an old line be applied twice.
        start = _align_forward(path, size - window, size)
        if start >= size:
            # No line start to be found inside the probe. Reading would open on a
            # half-line and STOP this journal for the boot, so this pass reads
            # nothing and says how far behind that leaves us.
            out["lag_bytes"] = size - cursor
            return out
        if start > cursor:
            skipped = start - cursor
            _forget_rank(path)
            out["bytes_skipped"] = skipped
            out["jumps"] = 1
            _state["bytes_skipped"] = int(_state["bytes_skipped"]) + skipped
            _state["catch_up_jumps"] += 1
            _note(f"{os.path.basename(path)} is {size - cursor} bytes ahead of its "
                  f"cursor and one refresh carries {window}: reading the newest "
                  f"{size - start} bytes and forgetting what this rank taught us "
                  "from the skipped bytes (a forgotten entry is a missed restore, "
                  "never a wrong one)")
            cursor = start
    if size <= cursor:
        return out
    want = min(size - cursor, window)
    try:
        with open(path, "rb") as handle:
            handle.seek(cursor)
            raw = handle.read(want)
    except OSError as exc:
        _note(f"{os.path.basename(path)} could not be read: "
              f"{type(exc).__name__}")
        out["lag_bytes"] = size - cursor
        return out
    last_newline = raw.rfind(b"\n")
    if last_newline < 0:
        # Nothing complete yet. The cursor stays, so the next refresh retries.
        out["lag_bytes"] = size - cursor
        return out
    body = raw[:last_newline + 1]
    lines = body.decode("utf-8", errors="replace").splitlines()
    if len(lines) > int(line_budget):
        # Forfeit the OLDEST lines of the window, never the newest: same ordering
        # rule as the byte window above, applied one level down.
        forfeit = len(lines) - int(line_budget)
        lines = lines[forfeit:]
        out["lines_skipped"] += forfeit
        _state["lines_skipped"] = int(_state["lines_skipped"]) + forfeit
        _state["dropped"]["line_cap"] += 1
    applied = 0
    for index, text in enumerate(lines):
        kind = _apply_line(text, path)
        if kind is None:
            # The stream stopped being trustworthy: everything after it in this
            # window goes unread too, and that is counted rather than silent.
            out["lines_skipped"] += len(lines) - index
            _state["lines_skipped"] = (int(_state["lines_skipped"])
                                       + len(lines) - index)
            break
        applied += 1
        if kind == SPILL_LINE:
            out["spills"] += 1
        else:
            out["removals"] += 1
    _state["cursors"][path] = cursor + last_newline + 1
    _state["last_journal"] = path
    out["read"] = last_newline + 1
    out["applied"] = applied
    out["lag_bytes"] = max(0, size - int(_state["cursors"][path]))
    return out


def _forget_rank(path) -> None:
    """Drop every index entry that came from this journal."""
    entries = _state["entries"]
    doomed = [key for key, entry in entries.items()
              if entry.get("journal") == path]
    for key in doomed:
        entries.pop(key, None)
    _state["bytes_indexed"] = max(0, sum(int(entry["bytes"])
                                         for entry in entries.values()))
    # The PER-RANK view loses exactly that rank's claims - and an identity whose last
    # remaining claim came from the truncated journal goes with it, so the two views
    # cannot disagree about whether a rank backs an entry.
    rank_lines = _state["rank_lines"]
    for identity in list(rank_lines):
        rows = rank_lines[identity]
        for name in [name for name, row in rows.items()
                     if row.get("journal") == path]:
            rows.pop(name, None)
            _state["rank_records"] = max(0, int(_state["rank_records"]) - 1)
        if not rows:
            rank_lines.pop(identity, None)


def _rank_label(path) -> str:
    """The rank a journal line came from, from the ONE rule that names a journal."""
    return _rank_of(path) or UNNAMED_RANK


def _record_rank(identity, path, size, seq) -> None:
    """Remember that THIS rank's journal claims this identity, bounded.

    Called from the same place that inserts the entry, so the per-rank view and the
    entry can never be built by different rules. Three caps: ranks per identity, pairs
    in total, and the identity cap that `entries` already enforces (an identity evicted
    from the entry table loses its rank rows here, so the two cannot drift apart).
    """
    rank_lines = _state["rank_lines"]
    rows = rank_lines.get(identity)
    if rows is None:
        if int(_state["rank_records"]) >= MAX_RANK_RECORDS:
            _state["rank_records_dropped"] += 1
            return
        rows = {}
        rank_lines[identity] = rows
    name = _rank_label(path)
    if name not in rows and len(rows) >= MAX_RANKS_PER_IDENTITY:
        _state["rank_records_dropped"] += 1
        return
    if name not in rows:
        _state["rank_records"] = int(_state["rank_records"]) + 1
    rows[name] = {"bytes": int(size), "journal": path, "seq": int(seq),
                  "seen_ts": round(time.time(), 6)}


def _drop_identity_ranks(identity) -> int:
    """Drop every rank claim for one identity. Returns how many went."""
    rows = _state["rank_lines"].pop(identity, None)
    if not rows:
        return 0
    _state["rank_records"] = max(0, int(_state["rank_records"]) - len(rows))
    return len(rows)


def rank_claims(identity) -> dict:
    """The per-rank claims for one identity, as a fresh dict. Never the live one."""
    return {name: dict(row) for name, row in
            _state["rank_lines"].get(identity, {}).items()}


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
        # The per-rank view goes with the entry, deliberately WITHOUT the rank's
        # identity being the deciding fact: one rank reporting the file unlinked drops
        # the identity for the offer path today, and a per-rank table that still
        # claimed all four would tell a truer-looking lie than the answer the offer
        # path is actually going to give.
        _drop_identity_ranks(identity)
        if gone is not None:
            _state["bytes_indexed"] = max(0, int(_state["bytes_indexed"])
                                          - int(gone["bytes"]))
        return REMOVE_LINE
    _state["spills"] += 1
    _insert(identity, size, path, group)
    _record_rank(identity, path, size, seq)
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
        _drop_identity_ranks(oldest)
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


def membership(key) -> dict:
    """The OPERATOR's answer to "is this span in the disk index?", per RANK.

    WHY THIS EXISTS WHEN `lookup()` AND `contains()` EXISTED. Both of those answer the
    OFFER path ("may I offer this identity?" - one claim, last journal wins). An
    operator asking "did the span I parked actually land?" needs three things neither
    gives: WHICH rank's claim it is reading (one parked span is one file per rank, so
    a single answer can call a two-ranks-of-four span complete), whether the index is
    even in a state to answer, and no side effect on the witness counters. So this
    reads the SAME replayed journals, per rank, and:

      * NEVER walks the store directory. `op=ls` does that; a `contains` that scanned
        thousands of files under a polled endpoint is how you build an I/O storm.
        Everything here is in-process memory plus one `journal_directory()` resolve.
      * Does NOT move `hits`/`misses`. Those are the witness's ledger and an
        experiment reads them as evidence of offer-path behaviour; an operator asking
        a question twice must not change them. `membership_checks` counts this verb.
      * Answers "I cannot answer" as its own verdict with `present: null`, never as
        `present: false`. A key that cannot name an identity, a journal directory
        that was never configured, an index that has replayed nothing, and an index
        that replayed and holds nothing are four DIFFERENT statements from "the entry
        is not here".

    Never raises; the key parsing is `disk_layout.split_block_key`'s, so there is no
    second spelling of an identity here.
    """
    try:
        _state["membership_checks"] = int(_state["membership_checks"]) + 1
        directory = journal_directory()
        view = {"refreshes": int(_state["refreshes"]),
                "entries": int(len(_state["entries"])),
                "journals_tracked": int(len(_state["cursors"])),
                # DID THE REPLAY EVER SEE A SPILL LINE? This is what separates "the
                # journals name no file" (never any spill, or a journal stopped at its
                # first line) from "the map holds nothing NOW" (all it learned has been
                # removed or evicted). Both read as `entries: 0`; only one of
                # them means the store was never described to this process.
                "spill_lines_applied": int(_state["spills"]),
                "removal_lines_applied": int(_state["removals"]),
                "bytes_indexed": int(_state["bytes_indexed"]),
                "journal_directory": directory,
                "journal_write_errors": int(_state["journal_write_errors"]),
                "rank_records": int(_state["rank_records"]),
                "rank_records_dropped": int(_state["rank_records_dropped"]),
                # HOW FAR BEHIND THE ANSWER IS. A `not_present` from a lagging index
                # is "this boot has not read that line yet", which is a different
                # claim from "the file was never spilled" - and it is the claim that
                # cost the 0-of-36 reading on 2026-10-05. Say it beside the answer.
                "lag_bytes": int(_state["lag_bytes"]),
                "lag_journals": int(_state["lag_journals"]),
                "catch_up_jumps": int(_state["catch_up_jumps"]),
                "bytes_skipped": int(_state["bytes_skipped"]),
                "lines_skipped": int(_state["lines_skipped"])}
        blank = {"key16": None, "group_id": None, "identity": None, "shard": None,
                 "bytes": 0, "bytes_all_ranks": 0, "journal": None, "rank": None,
                 "ranks": [], "ranks_present": 0, "index": view}
        split = disk_layout.split_block_key(key)
        if not split["ok"]:
            return {**blank, "ok": False, "present": None,
                    "verdict": VERDICT_KEY_REJECTED,
                    "reason": (f"the key cannot name an identity: "
                               f"{split['reason']}"),
                    "answer": "the request cannot be answered: this key does not "
                              "name a group-qualified identity"}
        key16, group = split["key16"], split["group_id"]
        identity = identity_of(key16, group)
        located = {"key16": key16, "group_id": group, "identity": identity,
                   "shard": key16[:disk_layout.SHARD_CHARS]}
        if directory is None:
            return {**blank, **located, "ok": True, "present": None,
                    "verdict": VERDICT_NO_JOURNAL_DIR,
                    "reason": (f"neither {config.ENV_DISK_JOURNAL_DIR} nor "
                               f"{config.ENV_DIAG_DIR} is set, so no journal could "
                               "ever be replayed into this index"),
                    "answer": "cannot answer: this process has no journal directory"}
        if view["refreshes"] == 0:
            return {**blank, **located, "ok": True, "present": None,
                    "verdict": VERDICT_NEVER_REFRESHED,
                    "reason": (f"the index has replayed no journal since this "
                               f"process started ({directory} is the journal "
                               "directory); an answer of 'not present' here would "
                               "be a claim about a map nobody read"),
                    "answer": "cannot answer yet: nothing has refreshed the index"}
        entry = _state["entries"].get(identity)
        claims = rank_claims(identity)
        rows = [{"rank": name, "bytes": int(row["bytes"]),
                 "journal": str(row["journal"]), "seq": int(row["seq"]),
                 "seen_ts": row["seen_ts"]}
                for name, row in sorted(claims.items())]
        total = sum(int(row["bytes"]) for row in rows)
        found = {**blank, **located, "ok": True, "ranks": rows,
                 "ranks_present": len(rows), "bytes_all_ranks": total}
        if entry is not None:
            return {**found, "present": True, "verdict": VERDICT_PRESENT,
                    "bytes": int(entry["bytes"]),
                    "journal": str(entry.get("journal") or ""),
                    "rank": _rank_of(entry.get("journal")),
                    "reason": None,
                    "answer": (f"present for {len(rows)} rank(s) of this identity; "
                               "one parked span is one file per rank, so compare "
                               "this count with the ranks the boot has armed "
                               "(`restore.ranks_armed` in the same answer)")}
        if rows:
            return {**found, "present": False, "verdict": VERDICT_ENTRY_DROPPED,
                    "reason": (f"{len(rows)} rank journal line(s) name this "
                               "identity, but no entry backs it: the identity was "
                               "dropped (the entry cap, a journal that stopped, or a "
                               "removal line), so the offer path will not see it "
                               "either"),
                    "answer": "not offered: rank lines remain but the entry is gone"}
        if view["entries"] == 0 and view["spill_lines_applied"] == 0:
            return {**found, "present": None, "verdict": VERDICT_INDEX_EMPTY,
                    "reason": (f"the index holds no identities at all after "
                               f"{view['refreshes']} refresh(es) of "
                               f"{view['journals_tracked']} journal file(s): "
                               f"nothing in this boot's journals names a spilled "
                               f"file, so {identity} cannot be found in it"),
                    "answer": "not in the index: the journal replay named no file"}
        return {**found, "present": False, "verdict": VERDICT_NOT_PRESENT,
                "reason": (f"this boot's replayed journals do not name {identity} "
                           f"({view['entries']} other identities are indexed over "
                           f"{view['journals_tracked']} journal file(s), "
                           f"{view['removal_lines_applied']} removal line(s) "
                           f"applied)"),
                "answer": f"not in the index: no entry and no rank line for "
                          f"{identity}"}
    except Exception as exc:  # noqa: BLE001 - a verb never breaks a schedule
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "disk_index membership")
        return _fail(f"the membership check raised {type(exc).__name__}",
                     present=None, verdict=VERDICT_CHECK_FAILED, key16=None,
                     group_id=None, ranks=[], ranks_present=0)


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
        _drop_identity_ranks(identity)
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
    _state["rank_lines"] = {}
    _state["rank_records"] = 0
    _state["bytes_indexed"] = 0
    _note(reason or "the index was cleared")
    return {"cleared": count, "reason": reason}


# --- the view ----------------------------------------------------------------

def section() -> dict:
    """The index's state for a snapshot or a verb. Reads nothing from disk."""
    # ONE read of the journal directory's filesystem + writability, reused by the
    # fields below and by `journal_fs`, so a payload cannot contradict itself.
    fs = journal_fs_state()
    return {
        "entries": int(len(_state["entries"])),
        "entry_cap": MAX_INDEX_ENTRIES,
        # THE PER-RANK VIEW `op=contains` answers from: how many (identity, rank) pairs
        # are held, and how many a cap refused to remember. A dropped pair is a span
        # that reads as fewer ranks than it has - never a false presence, always a
        # claim this index can no longer back.
        "rank_records": int(_state["rank_records"]),
        "rank_records_dropped": int(_state["rank_records_dropped"]),
        "rank_cap_per_identity": MAX_RANKS_PER_IDENTITY,
        "membership_checks": int(_state["membership_checks"]),
        "entries_dropped": int(_state["entries_dropped"]),
        "bytes_indexed": int(_state["bytes_indexed"]),
        "journals_tracked": int(len(_state["cursors"])),
        "journals_stopped": sorted(_state["stopped"]),
        "journal_writes": int(_state["journal_writes"]),
        "journal_lines": int(_state["journal_lines"]),
        "journal_seq_seed": int(_state["journal_seq_seed"]),
        "journal_full": bool(_state["journal_full"]),
        # THE FAILURE THAT USED TO BE SILENT. A spill whose journal line is not
        # written costs disk space and nothing else (refusing it would cost the
        # whole tier for a logging problem), so the tier keeps spilling and THESE
        # carry the cost out loud: a count to alert on, one line to read, and the
        # directory's own verdict with the reason that names the fix.
        "journal_write_errors": int(_state["journal_write_errors"]),
        "journal_last_error": _state["journal_last_error"],
        "journal_directory_attempts": int(_state["journal_directory_attempts"]),
        "journal_directory_errors": int(_state["journal_directory_errors"]),
        "journal_directory_writable": fs.get("writable"),
        "journal_directory_error": fs.get("directory_error"),
        "journal_fields": JOURNAL_FIELDS,
        "journal_cap_bytes": MAX_JOURNAL_BYTES,
        "refreshes": int(_state["refreshes"]),
        # THE READER'S OWN HONESTY: how much of the journal this process has NOT read
        # (`lag_bytes`, measured after the last refresh), how many journals that left
        # (`lag_journals`), and what the newest-first rule had to forfeit to reach the
        # tail at all. A non-zero lag makes every `not_present` in this payload a
        # statement about the READER, not about the store.
        "lag_bytes": int(_state["lag_bytes"]),
        "lag_journals": int(_state["lag_journals"]),
        "catch_up_jumps": int(_state["catch_up_jumps"]),
        "bytes_skipped": int(_state["bytes_skipped"]),
        "lines_skipped": int(_state["lines_skipped"]),
        "applied_lines": int(_state["applied"]),
        "spills_seen": int(_state["spills"]),
        "removals_seen": int(_state["removals"]),
        "lookup_hits": int(_state["hits"]),
        "lookup_misses": int(_state["misses"]),
        "dropped": dict(_state["dropped"]),
        "journal_directory": journal_directory(),
        # WHERE THAT DIRECTORY LIVES, filesystem-wise. `journal_on_tmpfs: true` is
        # the 2026-10-05 failure made visible: the map survives nothing and the KV
        # files survive everything. A warning, never a refusal.
        "journal_fs": fs,
        "journal_on_tmpfs": _state["journal_on_tmpfs"],
        "journal_tmpfs_warnings": int(_state["journal_tmpfs_warnings"]),
        "last_reason": _state["last_reason"],
        "last_error": _state["last_error"],
        "units": {"entries": "identities", "entry_cap": "identities",
                  "rank_records": "(identity, rank) pairs: one parked span is one "
                                  "file PER RANK, so an identity is backed by as many "
                                  "records as it has ranks",
                  "rank_records_dropped": "rank records a cap refused: the identity "
                                          "may read as fewer ranks than it "
                                          "has (never a false presence)",
                  "rank_cap_per_identity": "ranks remembered per identity",
                  "membership_checks": "answers `op=contains` has given here",
                  "entries_dropped": "identities", "bytes_indexed": "bytes",
                  "journals_tracked": "files", "journal_writes": "lines",
                  "journal_lines": "lines", "journal_write_errors": "writes",
                  "journal_directory_attempts": "attempts",
                  "journal_directory_errors": "failures",
                  "journal_seq_seed": "lines", "journal_fields": "fields",
                  "journal_cap_bytes": "bytes",
                  "journal_tmpfs_warnings": "warnings",
                  "lag_bytes": "bytes of journal this process has NOT read yet; 0 "
                                "means every cursor is at end-of-file",
                  "lag_journals": "journals still carrying unread bytes",
                  "catch_up_jumps": "times a journal was deep enough that one "
                                    "refresh could not reach its tail, so the older "
                                    "bytes were forfeited and that rank's entries "
                                    "were forgotten",
                  "bytes_skipped": "journal bytes forfeited to reach the tail (unit: "
                                   "bytes, NOT lines: this module does not guess a "
                                   "line count it has not read)",
                  "lines_skipped": "journal lines passed over without being applied "
                                   "(the line cap, or a stopped stream)",
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
                 "worker resolves the name IT would have written. Read `lag_bytes` "
                 "first: while it is non-zero this index has not read the newest "
                 "spill lines, and a `not_present` is then a statement about the "
                 "reader."),
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
