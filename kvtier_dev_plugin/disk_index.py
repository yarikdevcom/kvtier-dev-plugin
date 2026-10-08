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

THE READER DRAINS EVERY JOURNAL FRONT TO BACK, IN BOUNDED PASSES
----------------------------------------------------------------
A journal is append-only, and the only way the index can be honest about what it has
read is to read it IN ORDER. Two failures framed this rule from opposite ends. On
2026-10-05 a cursor that walked from byte 0 with one shared budget left the NEWEST
spills unread (0 of 36 just-spilled identities known while four journals sat ~60k
lines deep), and the 0.2.22 answer was to read each journal's TAIL and FORFEIT
everything older. That fixed the afternoon and caused the next one: a forfeited region
is UNRECOVERABLE (the cursor jumps to end-of-file and cursors only ever advance), the
jump lands `lag_bytes` at zero so the index reads as caught up while it is blind, and
a box with 4.7 MB / 132,099 journalled records and ~6,290 net-live identities
reported `entries: 146` for hours while `no_disk_run` refused every offer. A restart
did not help: 1.4 MB is still deeper than one refresh's share, so the first pass
buried the history again, deterministically.

The rule now is a DRAIN. A refresh reads from each journal's cursor FORWARD, applies
whole lines in the file's own order, and RETURNS WITH THE CURSOR SHORT OF EOF when
its budget binds; the next refresh continues where this one stopped. The first
refresh of a process runs several such passes under its own larger budget
(`BOOT_DRAIN_MAX_BYTES`, `BOOT_DRAIN_DEADLINE_SECONDS`) so a cold boot converges
before the first offer instead of dozens of one-second refreshes later; every later
refresh is ONE bounded pass, exactly the old caps. A partial-but-advancing index is
strictly better than partial-and-frozen: the drain cannot lose history, converges in
a number of passes a reader can compute (`unread_bytes / share`), and every
intermediate state is publishable. While any cursor is short of its journal's end the
index says so (`converged: false`, `lag_bytes`, `unread_journals` naming each
journal and its unread bytes), and a `not_present` out of a partial index is a
statement about the READER, never about the store. The one legitimate use of
`_forget_rank` that remains is an externally TRUNCATED journal, where the bytes the
entries were learned from no longer exist.

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

import heapq
import logging
import os
import time
from collections import OrderedDict

from . import config, disk_layout, snapshot

# --- the caps, and the unit of every one of them ----------------------------

# THE ENTRY CAP, and the memory it buys. MEASURED IN THIS REPO, not estimated: a 20,000-
# entry index holding a real 72-hex identity string and a real entry dict costs 6.8
# MB, so
# ~342 bytes an entry. At the default that is UNDER 90 MB AT the cap - and the dict only
# grows to the identities the journals actually name, so a small store costs what a
# small
# store holds and the number below is a ceiling on memory, never an allocation.
#
# WHY THE DEFAULT MOVED UP (0.2.45). It was 65,536, which the live store walks into:
# 12,738
# files across four ranks under a 148 GiB ceiling today, ~30,000 possible, and a
# converged
# drain over a big store lives AT the cap. Walking into it is not an error - it is the
# index answering `not_present` for files that are sitting on the RAID, whole and
# readable,
# which spends real capacity on spans the tier refuses to offer. An operator can set it
# with `VLLM_E2_TIER_DISK_INDEX_ENTRIES` (`config.disk_index_env` holds the grammar).
MAX_INDEX_ENTRIES = 262_144         # IDENTITIES held in memory (LRU-by-touch)
# THE POLICY BOUNDS ON AN OPERATOR'S NUMBER. Below the floor the index remembers so
# little
# that every disk offer is a miss, which reads as "configured" while being the most
# broken
# state the module can be in; above the ceiling the scheduler process is being asked to
# hold hundreds of megabytes of map. Either is CLAMPED WITH A REASON, never trusted.
INDEX_ENTRIES_FLOOR = 1_024         # IDENTITIES: the smallest cap honoured
INDEX_ENTRIES_CEILING = 4_194_304   # IDENTITIES: ~1.4 GB at the measured 342 B/entry
# THE PRESENCE SCAN, bounded. When the cap drops an entry the one question worth
# asking is
# whether the FILE is still on the store - that is what makes a drop a missed restore
# instead of bookkeeping. The answer needs a directory listing, and a cap-drop burst is
# exactly the moment thousands of identities would like one (the 0.2-28 drain dropped
# 800
# in a single refresh), so one refresh lists at most this many shard directories and
# holds
# at most `PRESENCE_SCAN_MAX_ENTRIES` names per listing. Everything the budget did not
# reach is counted UNVERIFIED, because "did not look" is a fact and "absent" would be a
# guess in the comfortable direction.
PRESENCE_SCAN_MAX_SHARDS_PER_REFRESH = 4
PRESENCE_SCAN_MAX_ENTRIES = 200_000   # NAMES one shard listing may hold
# BYTES one index entry costs, MEASURED in this repo (a 20,000-entry index holding a
# real
# 72-hex identity and a real entry dict: 6,835,398 bytes, so 342 an entry). Published so
# the cap can be chosen from arithmetic instead of hope, and so `bytes_at_cap` in
# `entry_cap()` means what it says.
ENTRY_BYTES_MEASURED = 342
MAX_JOURNAL_BYTES = 32 * 1024 * 1024  # BYTES one rank's journal may reach
MAX_JOURNAL_FILES = 64              # journal FILES one refresh will open
MAX_LINES_PER_REFRESH = 8_192       # LINES applied per refresh (bounded work)
MAX_READ_PER_REFRESH = 512 * 1024   # BYTES read across all journals per refresh
# THE FLOOR ON ONE JOURNAL'S SHARE of a refresh's budget. Below this a window is
# smaller than one rank's newest spill burst; the TOTAL cap (`MAX_READ_PER_REFRESH`)
# is still what one refresh reads.
MIN_JOURNAL_WINDOW_BYTES = 4 * 1024  # BYTES: the smallest share one journal gets
# THE BOOT DRAIN: the FIRST refresh of a process may make several bounded passes so a
# cold boot with deep journals converges BEFORE the first offer instead of one share
# per second afterwards. The bounds are total bytes READ and wall time for that first
# refresh only - every later refresh is one pass under `MAX_READ_PER_REFRESH` /
# `MAX_LINES_PER_REFRESH` exactly as before. 32 MiB covers four 32 MiB-cap'd journals
# at 512 KiB passes only partially; what matters is that neither bound can stall a
# boot: on the observed shape (4 journals, 4.7 MB, 132k records) the drain reads
# every byte in ONE refresh in well under a second, and a box that exceeds either
# bound simply keeps `converged: false` and finishes through ordinary refreshes.
BOOT_DRAIN_MAX_BYTES = 32 * 1024 * 1024   # BYTES the first refresh may read in total
BOOT_DRAIN_DEADLINE_SECONDS = 5.0         # SECONDS the first refresh may take
# HOW MANY journals the per-journal unread map names in a payload before the rest are
# folded into `unread_journals_omitted`. The bytes are always summed; only names cap.
MAX_LAG_NAMED_JOURNALS = 8
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

# PER-RANK PRESENCE (0.2.46), and the defect that earned it. A parked span is not one
# file, it is ONE FILE PER RANK (`kvslot-<key16>-g<group>-<geodigest12>.bin`, every rank
# naming it with its OWN geometry digest), and a restore reads them all. So "is this
# span on disk?" is a per-rank SET, never a bool, and a removal line means "THE RANK
# THAT WROTE THIS LINE unlinked its own file" - nothing more. HEAD's removal handler
# retired the whole identity (`_drop_identity_ranks`) and the offer path read a single
# global table, which lied in both directions at once: one rank's ceiling pass denied
# every peer its still-readable files (measured on the box 2026-10-26 as ~442 identities
# per rank, about 19.5 GB of KV no rank was allowed to read), and a peer's next `+`
# line resurrected the identity that MY rank had just deleted - the false present that
# reaches a worker as `file_missing` and a hole in the middle of a claimed run. Making
# presence per-rank needed no journal format change: the rank is in the FILE NAME, and
# the drain already knew which rank wrote a line. The removal handler was not listening.
MAX_RANKS_EXPECTED = 64             # RANKS one presence answer is intersected over
MAX_RANK_NAME_CHARS = 64            # CHARACTERS one rank label may carry (a verb's)

# THE SAMPLED AUDIT (0.2.46). The condition this change fixes is rank-shaped and a
# box has 262,144 identities in the table, so the live measurement is a bounded SAMPLE
# taken every few refreshes, never a census: an exact classification per refresh would
# share the offer path's process and its request budget. Every key it publishes is
# named `sampled_*` for exactly that reason - it is a rate over a sample, not a count
# of the store.
AUDIT_SAMPLE_IDENTITIES = 64        # IDENTITIES one audit pass classifies
AUDIT_EVERY_N_REFRESHES = 4         # REFRESHES between two audit passes
MAX_IDENTITIES_PER_CALL = 512       # IDENTITIES one `op=identities` page may carry

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
# WHAT A 3-OF-4 SPAN ANSWERS (0.2.46). `present: false` is still the OFFER answer - a
# span one rank cannot read is not a span this boot can restore - but it arrives WITH
# the rank it is waiting for. Before this token existed, "three ranks hold readable
# files and one deleted its slice" and "nobody ever spilled this" printed the same
# three words, and the first is a rank's ceiling pass while the second is a
# configuration problem. They are different investigations and the name says so.
VERDICT_PARTIAL_RANKS = "partial_ranks"
VERDICT_CHECK_FAILED = "membership_check_failed"
MEMBERSHIP_VERDICTS = (VERDICT_PRESENT, VERDICT_NOT_PRESENT, VERDICT_ENTRY_DROPPED,
                       VERDICT_NEVER_REFRESHED, VERDICT_NO_JOURNAL_DIR,
                       VERDICT_INDEX_EMPTY, VERDICT_KEY_REJECTED,
                       VERDICT_PARTIAL_RANKS, VERDICT_CHECK_FAILED)
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
    "entries": OrderedDict(),       # identity -> entry dict, ORDERED BY LAST TOUCH
    # identity -> {rank -> {bytes, journal, seq, seen_ts}}. `entries` is what the OFFER
    # path asks (one claim per identity, the last journal applied wins); this is the
    # SAME replay seen per rank, the only way to answer "is the whole span there, or
    # two ranks of four?" without walking the store.
    "rank_lines": {},
    "rank_records": 0,                # (identity, rank) PAIRS held
    "rank_records_dropped": 0,        # pairs refused by a cap
    # PER-RANK PRESENCE (0.2.46): the counters of the two lies this table used to tell,
    # counted at apply time so reading them costs no scan and no `stat`.
    #   claims_removed - a `-` line that retired the CLAIMING rank's own claim: the
    #     whole of what a removal line has always meant, and the only one of these four
    #     that HEAD also counted (as a whole identity).
    #   claims_kept_past_peer_removal - (identity, rank) PAIRS that survived another
    #     rank's removal line. THE UNIT IS THE PAIR because that is the unit the box
    #     measured: one removal line used to withdraw several ranks' files, and this is
    #     the ~442-per-rank false-absent, now counted instead of served.
    #   rank_claims_unmatched - a `-` for a claim that rank did not hold (a replayed
    #     line, an eviction never journalled as a spill). It retires nothing and must
    #     move no byte, which is the double-debit a global entry table made easy.
    #   inserts_incomplete_span - a `+` line whose identity is NOT held by every rank
    #     that has to read. Ordinary during a boot drain and across divergent ceiling
    #     passes; what matters is that this used to be the OFFER answer too, and a run
    #     was promised across it.
    "claims_removed": 0,
    "claims_kept_past_peer_removal": 0,
    "rank_claims_unmatched": 0,
    "inserts_incomplete_span": 0,
    # WHOSE PRESENCE AN ANSWER IS INTERSECTED OVER, and who said so. Learned from the
    # journal FILE NAMES the drain already lists - they ARE the rank set, and they exist
    # before any request does - and monotonic within a boot: losing a rank from the set
    # would RELAX the intersection, which is the one direction that is not safe. The
    # offer path overrides it with the ranks the worker proof names, because that is who
    # will actually read; this is the fallback and the published comparison.
    "ranks_expected": [],
    "ranks_expected_source": None,
    # THE SAMPLED AUDIT, so the condition has a live number that is not an operator
    # running a verb over 262,144 identities. Bounded by AUDIT_SAMPLE_IDENTITIES per
    # pass and AUDIT_EVERY_N_REFRESHES between passes, and it does NO I/O at all: it
    # classifies the tables this process already holds.
    "last_audit_at_refresh": None,
    "sampled_runs": 0,
    "sampled_identities": 0,
    "sampled_partial_ranks": 0,
    "sampled_offerable": 0,
    "sampled_claims_without_entry": 0,
    "sampled_refreshes_since": 0,
    "sampled_last": None,
    "membership_checks": 0,           # `op=contains` answers given here
    "cursors": {},                    # journal path -> byte offset
    "seq": {},                        # journal path -> last applied seq
    "stopped": {},                    # journal path -> reason it stopped
    "applied": 0,
    "spills": 0,
    "removals": 0,
    "misses": 0,
    "hits": 0,
    # THE DRAIN: whether the first refresh has run, whether every cursor sits at its
    # journal's end, the total bytes this process has read out of the journals, and
    # the per-journal unread bytes measured at the last real refresh.
    "boot_drained": False,
    "converged": False,
    "bytes_read": 0,
    "unread_journals": {},
    "entries_dropped": 0,
    # WHICH DROPS COST A FILE (0.2.45). `entries_dropped` says the cap evicted an entry;
    # these three say whether the file it described was still on the store, and they
    # BALANCE the total so a reader can tell a verified answer from an unverified one.
    # `_file_present` is the number that matters: a whole span on disk that the
    # index has
    # stopped offering. `_file_absent` is not a loss at all - the file was reclaimed and
    # the entry was the last memory of it. `_unverified` is the scan budget being
    # spent, or
    # no store root this process can see, and is deliberately never folded into either.
    "entries_dropped_file_present": 0,
    "entries_dropped_file_absent": 0,
    "entries_dropped_unverified": 0,
    # THE LRU CLOCK'S OWN LEDGER (0.2.45): how many touches moved an entry to the
    # back of
    # the queue, and how many shard listings the presence scan performed. Without
    # `entries_touched` a reader cannot tell a working LRU from a no-op.
    "entries_touched": 0,
    "restores_noted": 0,
    "presence_scans": 0,
    "presence_scan_skipped": 0,
    # THE ENTRY CAP'S OWN TRUTH: set the first time the cap drops an entry, and
    # published so no payload can present a capped index as a caught-up one.
    "entry_cap_hit": False,
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
    # WHAT A READER FORFEITED. The 0.2.28 drain makes these STRUCTURALLY ZERO for a
    # fresh boot - a journal's body is now drained, never skipped - and they stay
    # published for two reasons: a process whose journal was STOPPED by a malformed
    # line still counts the lines after it (permanently unread), and a payload
    # produced by an older plugin still has to mean the same thing to a reader.
    # Any non-zero `bytes_skipped`/`catch_up_jumps` is forfeited history and every
    # view must say so: spans in a region that was skipped are NOT KNOWABLE.
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
        "entries": OrderedDict(), "rank_lines": {}, "rank_records": 0,
        "rank_records_dropped": 0, "membership_checks": 0,
        "cursors": {}, "seq": {}, "stopped": {},
        "applied": 0, "spills": 0, "removals": 0, "misses": 0, "hits": 0,
        "boot_drained": False, "converged": False, "bytes_read": 0,
        "unread_journals": {},
        "entries_dropped": 0, "entry_cap_hit": False,
        "entries_dropped_file_present": 0, "entries_dropped_file_absent": 0,
        "entries_dropped_unverified": 0, "entries_touched": 0,
        "restores_noted": 0, "presence_scans": 0, "presence_scan_skipped": 0,
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
        "claims_removed": 0, "claims_kept_past_peer_removal": 0,
        "rank_claims_unmatched": 0, "inserts_incomplete_span": 0,
        "ranks_expected": [], "ranks_expected_source": None,
        "last_audit_at_refresh": None, "sampled_runs": 0,
        "sampled_identities": 0, "sampled_partial_ranks": 0,
        "sampled_offerable": 0, "sampled_claims_without_entry": 0,
        "sampled_refreshes_since": 0, "sampled_last": None,
    })
    _presence["refresh"] = 0
    _presence["listings"] = {}
    _presence["budget"] = PRESENCE_SCAN_MAX_SHARDS_PER_REFRESH
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
    out["unread_journals"] = _unread_view()
    return out


def index_lag() -> dict:
    """THE DRAIN'S STATE, in memory only: what this index has NOT read, and what it
    once forfeited. Never a stat, never a scan - this is what the offer path reads to
    keep a `no_disk_run` refusal honest.

    Two different blindnesses, kept apart because they read differently:

      * `converged: false` with `lag_bytes` > 0 means the drain is IN PROGRESS: the
        unread region is named in `unread_journals` and will be read. A `not_present`
        for a span that might live in it is "spilled but not yet indexed", never
        "never spilled".
      * `catch_up_jumps`/`bytes_skipped` non-zero means history was FORFEITED (the
        0.2.22 reader, or a build from before this one): those regions are NOT
        KNOWABLE for this process for as long as it runs - a cursor only ever
        advances. Both readings mean `no_disk_run` cannot conclude "never spilled".
    """
    unread = _unread_view()
    return {"converged": bool(_state["converged"]),
            "boot_drained": bool(_state["boot_drained"]),
            "lag_bytes": int(_state["lag_bytes"]),
            "lag_journals": int(_state["lag_journals"]),
            "unread_journals": dict(sorted(unread.items(),
                                           key=lambda row: -row[1])
                                   [:MAX_LAG_NAMED_JOURNALS]),
            "unread_journals_omitted": max(0, len(unread)
                                           - MAX_LAG_NAMED_JOURNALS),
            "catch_up_jumps": int(_state["catch_up_jumps"]),
            "bytes_skipped": int(_state["bytes_skipped"]),
            "lines_skipped": int(_state["lines_skipped"]),
            "bytes_read": int(_state["bytes_read"]),
            "entry_cap_hit": bool(_state["entry_cap_hit"]),
            "entries_dropped": int(_state["entries_dropped"]),
            "journals_stopped": sorted(str(os.path.basename(str(p)))
                                       for p in _state["stopped"])}


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

    THE DRAIN RULE. Each journal is replayed FROM ITS CURSOR FORWARD: a pass reads
    at most the journal's share of `MAX_READ_PER_REFRESH` bytes and
    `MAX_LINES_PER_REFRESH` lines, applies whole lines in the file's own order, and
    if the unread run is deeper than that it leaves the cursor short of end-of-file
    and the NEXT refresh continues from there. Nothing is forfeited: the index is
    temporarily partial-but-advancing, which is observable (`lag_bytes`,
    `unread_journals`, `converged: false`) and strictly better than the 0.2.22
    behavior of jumping to the tail, forgetting the rank's history and reporting
    `lag_bytes: 0` forever after (the `entries: 146` over 132,099 records incident).

    THE BOOT DRAIN. The FIRST refresh of this process repeats the bounded pass until
    every journal's cursor reaches end-of-file, bounded TOTAL by
    `BOOT_DRAIN_MAX_BYTES` bytes read and `BOOT_DRAIN_DEADLINE_SECONDS` of wall time;
    either bound ending the loop is normal (logged, `converged: false`) and ordinary
    refreshes finish the job. It runs wherever the first refresh runs - the
    scheduler's offer path (`disk_restore._refresh_index`) - so a cold boot pays it
    inside the FIRST offer attempt rather than spreading it over a minute of
    one-second refreshes.

    WORK BOUND (the invariant the offer path depends on): any refresh after the
    first reads at most `MAX_READ_PER_REFRESH` bytes and applies at most
    `MAX_LINES_PER_REFRESH` lines, opens at most `MAX_JOURNAL_FILES` files and one
    `stat` per journal; an idle store costs one stat per rank. `force` ignores the
    age floor (a verb that wants the truth right now); the lookup path uses the
    floor so a burst of requests does not turn into a burst of stats.
    """
    started = time.monotonic()
    out = {"ok": True, "reason": None, "journals": 0, "lines": 0, "spills": 0,
           "removals": 0, "dropped": 0, "entries": int(len(_state["entries"])),
           "applied": False, "lag_bytes": 0, "lag_journals": 0,
           "bytes_skipped": 0, "lines_skipped": 0, "jumps": 0,
           "bytes_read": 0, "converged": bool(_state["converged"]),
           "unread_journals": _unread_view()}
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
        # A NEW SCAN BUDGET for this refresh, and no listing carried forward:
        # the presence
        # question is "is the file there NOW", and a cap-drop burst in one refresh must
        # not pay for listings another refresh would have needed.
        _begin_presence_refresh()
        names = _journal_names(directory)
        # WHOSE PRESENCE AN ANSWER IS INTERSECTED OVER (0.2.46): the journal files
        # themselves name the ranks, and they exist before any request does. The set
        # only grows within a boot - see `_observe_journal_ranks`.
        _observe_journal_ranks(names)
        if not names:
            out["reason"] = ("no spill journal file exists in the diagnostics "
                             "directory: no rank has spilled anything yet")
            # Nothing to read is caught up, at least as far as any cursor is
            # concerned - and it stays so even before the first boot drain, which
            # had no journal to drain.
            _state["converged"] = True
            out["converged"] = True
            return out
        boot = not _state["boot_drained"]
        _state["boot_drained"] = True
        deadline = started + BOOT_DRAIN_DEADLINE_SECONDS if boot else None
        unread: dict = {}
        last_lag = None
        while True:
            pass_budget = MAX_READ_PER_REFRESH
            if boot:
                left = BOOT_DRAIN_MAX_BYTES - out["bytes_read"]
                if left <= 0:
                    _note(f"the boot drain stopped at its {BOOT_DRAIN_MAX_BYTES}-"
                          "byte budget with unread journal bytes left; ordinary "
                          "refreshes will finish the drain (converged: false "
                          "until they do)")
                    break
                pass_budget = min(pass_budget, left)
            res = _refresh_pass(directory, names, pass_budget, unread)
            out["journals"] += res["journals"]
            out["lines"] += res["applied"]
            out["spills"] += res["spills"]
            out["removals"] += res["removals"]
            out["dropped"] += res["dropped"]
            out["lag_bytes"] = res["lag_bytes"]
            out["lag_journals"] = res["lag_journals"]
            out["bytes_skipped"] += res["bytes_skipped"]
            out["lines_skipped"] += res["lines_skipped"]
            out["jumps"] += res["jumps"]
            out["bytes_read"] += res["read"]
            if res["lag_bytes"] <= 0:
                break
            if not boot:
                break
            if last_lag is not None and res["lag_bytes"] == last_lag:
                # A pass that changed nothing (a torn tail line the writer has not
                # finished, or a read that failed): re-reading the same bytes in a
                # tight loop would spin the boot deadline away for the same view.
                # The next refresh - one floor interval away - tries again.
                break
            last_lag = int(res["lag_bytes"])
            if time.monotonic() >= deadline:
                _note(f"the boot drain stopped at its "
                      f"{BOOT_DRAIN_DEADLINE_SECONDS}s deadline with "
                      f"{res['lag_bytes']} unread journal bytes over "
                      f"{res['lag_journals']} journal(s); ordinary refreshes will "
                      "finish the drain (converged: false until they do)")
                break
        _state["lag_bytes"] = int(out["lag_bytes"])
        _state["lag_journals"] = int(out["lag_journals"])
        _state["bytes_read"] = int(_state["bytes_read"]) + out["bytes_read"]
        _state["unread_journals"] = dict(
            sorted(unread.items(), key=lambda row: -row[1])
            [:MAX_LAG_NAMED_JOURNALS])
        _state["converged"] = out["lag_bytes"] <= 0
        out["converged"] = bool(_state["converged"])
        out["unread_journals"] = _unread_view()
        out["entries"] = int(len(_state["entries"]))
        out["applied"] = out["lines"] > 0
        out["elapsed_seconds"] = round(time.monotonic() - started, 6)
        # THE SAMPLED AUDIT, on its cadence rather than every pass: a bounded
        # classification of the tables this refresh just built, no I/O, so the
        # condition has a live number that never competes with the offer path for a
        # request's time. `sampled_*` throughout, because it is a rate over a sample.
        _state["sampled_refreshes_since"] = int(_state["sampled_refreshes_since"]) + 1
        if _state["sampled_refreshes_since"] >= AUDIT_EVERY_N_REFRESHES:
            _state["sampled_refreshes_since"] = 0
            _state["last_audit_at_refresh"] = int(_state["refreshes"])
            audit_now()
        if out["journals"]:
            out["reason"] = None
        return out
    except Exception as exc:  # noqa: BLE001 - the lookup path must not raise
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "disk_index refresh")
        out["reason"] = f"the journal refresh raised {type(exc).__name__}"
        out["ok"] = False
        return out


def _unread_view() -> dict:
    """A copy of the per-journal unread map (basename -> bytes). Never the live one."""
    return {str(name): int(value)
            for name, value in _state["unread_journals"].items()}


def _refresh_pass(directory, names, byte_budget, unread_out) -> dict:
    """ONE bounded pass over every journal. `unread_out` learns per-journal bytes.

    The TOTAL caps (`byte_budget`, `MAX_LINES_PER_REFRESH`) bind the whole pass; the
    SHARE of each journal only decides how the budget is divided, so one deep rank
    cannot starve the others. A span needs EVERY rank to have journalled it, so a
    uniformly-behind drain is honest where a partially-caught-up index would turn
    "the file is gone for rank 3" into an offered load that fails.
    """
    res = {"journals": 0, "applied": 0, "spills": 0, "removals": 0, "dropped": 0,
           "lag_bytes": 0, "lag_journals": 0, "bytes_skipped": 0,
           "lines_skipped": 0, "jumps": 0, "read": 0}
    budget = int(byte_budget)
    lines = MAX_LINES_PER_REFRESH
    share_bytes = max(MIN_JOURNAL_WINDOW_BYTES,
                      MAX_READ_PER_REFRESH // max(1, len(names)))
    share_lines = max(1, MAX_LINES_PER_REFRESH // max(1, len(names)))
    for name in names:
        path = os.path.join(directory, name)
        res["journals"] += 1
        want_bytes = min(share_bytes, budget)
        want_lines = min(share_lines, lines)
        if want_bytes <= 0 or want_lines <= 0:
            _state["dropped"]["read_cap"] += 1
            res["dropped"] += 1
            continue
        result = _apply_journal(path, want_bytes, want_lines)
        budget -= int(result["read"])
        lines -= int(result["applied"])
        res["applied"] += int(result["applied"])
        res["spills"] += int(result["spills"])
        res["removals"] += int(result["removals"])
        res["read"] += int(result["read"])
        res["lag_bytes"] += int(result["lag_bytes"])
        res["bytes_skipped"] += int(result["bytes_skipped"])
        res["lines_skipped"] += int(result["lines_skipped"])
        res["jumps"] += int(result["jumps"])
        if int(result["lag_bytes"]) > 0:
            res["lag_journals"] += 1
            unread_out[os.path.basename(path)] = int(result["lag_bytes"])
        else:
            unread_out.pop(os.path.basename(path), None)
    return res


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


def _apply_journal(path, budget, line_budget) -> dict:
    """Replay the NEXT bounded run of one journal, FROM ITS CURSOR FORWARD.

    Returns `{read, applied, spills, removals, lag_bytes, bytes_skipped,
    lines_skipped, jumps}`.

    THE DRAIN, NOT THE JUMP. The window is the FIRST `budget` bytes after the cursor
    - the file's own next lines - never a tail window. When the unread run is deeper
    than `budget` (or holds more complete lines than `line_budget`), the pass applies
    what fits, advances the cursor only past the lines it APPLIED, and leaves the
    rest as `lag_bytes` for the next refresh. Nothing is forfeited and nothing is
    forgotten: an entry the index claims was read, an entry it does not claim was
    either never journalled, removed, or is sitting in a region the payload names
    (`unread_journals`). The 0.2.22 rule - read the tail, `_forget_rank` the
    skipped history, land the cursor at EOF - is GONE, and with it the "don't
    re-read history" optimization: it bought freshness with unrecoverable amnesia
    (the cursor jump put `lag_bytes` at 0 on a blind index, and a restart re-buried
    the same 1.4 MB on its first pass). Freshness without amnesia is what the boot
    drain plus the per-second share provide: a writer would have to out-run
    `MAX_READ_PER_REFRESH` bytes per second for the unread run to grow at all, and
    the journals of the incident boot grew ~80 bytes a minute.

    Only COMPLETE lines (ending in a newline) are applied, and the cursor only ever
    advances to the end of the last APPLIED line: a line the writer is halfway
    through is nobody's fact yet, and lines past the line cap are not read YET, not
    read-and-discarded. A file that got SMALLER was truncated by someone who is not
    this plugin - the entries contributed by that rank are dropped, because we can
    no longer tell what they were (the bytes they were learned from are gone; this
    is the one legitimate `_forget_rank`), and the replay starts from the top.

    A line the reader cannot interpret STOPS this journal (see `_stop`): the lines
    after the stop are counted in `lines_skipped` because for THIS journal they are
    permanently unread - the only shape of skip the drain cannot promise to repair.
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
    window = min(max(int(budget), 0), size - cursor)
    if window <= 0:
        _state["dropped"]["read_cap"] += 1
        out["lag_bytes"] = size - cursor
        return out
    try:
        with open(path, "rb") as handle:
            handle.seek(cursor)
            raw = handle.read(window)
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
    rows = body.split(b"\n")
    rows.pop()                       # the body ends on its newline: tail is empty
    complete = len(rows)
    if complete > int(line_budget):
        # PAUSE at the line cap: apply the FIRST `line_budget` lines and leave the
        # rest unread. The old rule forfeited the OLDEST lines of a tail window;
        # under the drain there is no window to trim, only a queue to work through.
        rows = rows[:int(line_budget)]
        _state["dropped"]["line_cap"] += 1
    applied = 0
    consumed = 0
    for index, row in enumerate(rows):
        kind = _apply_line(row.decode("utf-8", errors="replace"), path)
        if kind is None:
            # The stream stopped being trustworthy: every complete line of this
            # window from here on is permanently unread (the journal is stopped),
            # and that is counted rather than silent.
            out["lines_skipped"] += complete - index
            _state["lines_skipped"] = (int(_state["lines_skipped"])
                                       + complete - index)
            # Move the cursor past the whole window: the journal will never be
            # applied again, so leaving its bytes "unread" would fake a drain that
            # can never finish.
            _state["cursors"][path] = cursor + last_newline + 1
            out["read"] = window
            out["lag_bytes"] = max(0, size - int(_state["cursors"][path]))
            return out
        consumed += len(row) + 1
        applied += 1
        if kind == SPILL_LINE:
            out["spills"] += 1
        else:
            out["removals"] += 1
    _state["cursors"][path] = cursor + consumed
    _state["last_journal"] = path
    out["read"] = window
    out["applied"] = applied
    out["lag_bytes"] = max(0, size - int(_state["cursors"][path]))
    return out


def _forget_rank(path) -> None:
    """Drop the claims THIS journal backs, and only the entries nothing else backs.

    PER RANK, AND ONLY SINCE 0.2.46. This used to pop every entry whose last writer was
    this journal, and when presence was a single global table that looked like the safe
    direction. It was not: a truncated journal means THIS RANK can no longer prove its
    files, while the peers' claims - real bytes on the store, priced in
    `bytes_indexed` - are untouched. So the rank's rows go, and an entry goes only when
    the LAST rank claim goes with them, which is the same rule the removal line now
    follows in `_drop_rank_claim`.
    """
    rank_lines = _state["rank_lines"]
    entries = _state["entries"]
    for identity in list(rank_lines):
        rows = rank_lines[identity]
        for name in [name for name, row in rows.items()
                     if row.get("journal") == path]:
            rows.pop(name, None)
            _state["rank_records"] = max(0, int(_state["rank_records"]) - 1)
        if rows:
            continue
        rank_lines.pop(identity, None)
        # NOTHING backs this identity any more, so the entry that priced it goes too.
        entries.pop(identity, None)
    _state["bytes_indexed"] = max(0, sum(int(entry["bytes"])
                                         for entry in entries.values()))


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
            if notice_once(
                    "rank-records-cap",
                    f"the per-rank table reached its {MAX_RANK_RECORDS}-pair cap: "
                    f"{int(_state['rank_records_dropped'])} rank record(s) refused "
                    "so far (rank_records_dropped). An identity whose records a cap "
                    "refused reads as FEWER ranks than it has - never a false "
                    "presence, always a claim this index can no longer back. The "
                    "converged drain makes this reachable on a big store: it means "
                    "the live identity count times ranks has outgrown the table."):
                _state["last_reason"] = (
                    f"the per-rank table is at its {MAX_RANK_RECORDS}-pair cap "
                    "(rank_records_dropped is counting what it refuses)")
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


# --- the rank set an answer is intersected over (0.2.46) ---------------------

def _expected_from_names(names) -> list:
    """The rank labels a set of journal file names describes. Bounded, sorted.

    THE RANK IS IN THE FILE NAME (`disk-spill-journal-PP0_TP0.jsonl`), which is why
    making presence per-rank needed no change to the journal format: the drain already
    knows which rank wrote every line, and the removal handler was not listening.
    """
    found = {_rank_label(name) for name in names}
    found.discard(UNNAMED_RANK)      # a journal with no rank in its name names nobody
    out = sorted(found)
    if len(out) > MAX_RANKS_EXPECTED:
        if notice_once(
                "ranks-expected-cap",
                f"{len(out)} journal rank labels exist and the expected-rank set "
                f"names only {MAX_RANKS_EXPECTED}: a journal is never deleted, so a "
                "box whose topology was renamed keeps names from the old one, and "
                "intersecting over ranks that no longer exist withholds every span. "
                "The set is published as `ranks_expected`, and the offer path prefers "
                "the ranks the worker proof names - move the dead journals aside."):
            _note(f"the expected-rank set is at its {MAX_RANKS_EXPECTED}-rank cap")
        out = out[:MAX_RANKS_EXPECTED]
    return out


def _observe_journal_ranks(names) -> None:
    """Record which ranks the journals seen this refresh belong to.

    MONOTONIC ON PURPOSE: the set only ever grows within a boot. An unreadable
    directory, or a journal that momentarily vanishes, must not RELAX the intersection,
    because a relaxed intersection is the false present this change exists to remove -
    losing a rank from the set is the one way to make the index promise a span a rank
    cannot read. Only a restart, or the offer path naming the armed ranks directly,
    decides who has to read.
    """
    found = _expected_from_names(names)
    if not found:
        return
    merged = sorted(set(_state["ranks_expected"]) | set(found))[:MAX_RANKS_EXPECTED]
    changed = merged != list(_state["ranks_expected"])
    _state["ranks_expected"] = merged
    _state["ranks_expected_source"] = "journal files"
    if changed:
        _note(f"the index now intersects presence over {len(merged)} rank(s) "
              f"named by the journals: {','.join(merged)[:96]}")


def expected_ranks() -> list:
    """The rank labels this index intersects a presence answer over. A copy."""
    return list(_state["ranks_expected"])


def expected_rank_source():
    """WHO said the expected-rank set: "journal files", or None before a first drain."""
    return _state["ranks_expected_source"]


def _expected_ranks(ranks=None) -> tuple:
    """The set to intersect over for ONE answer: the caller's, else the index's.

    A caller that names ranks gets EXACTLY those ranks - no union, no silent widening.
    That is what lets the offer path ask about the ranks the boot has armed (who will
    actually read) so a dead rank's leftover journal cannot veto the living ones, while
    an armed rank that has spilled nothing vetoes everything.
    """
    if ranks is None:
        return tuple(_state["ranks_expected"])
    if isinstance(ranks, (str, bytes)):
        return (str(ranks),)[:MAX_RANKS_EXPECTED]
    try:
        names = sorted({str(name) for name in ranks if str(name)})
    except TypeError:
        return ()
    return tuple(names[:MAX_RANKS_EXPECTED])


def _missing_ranks(identity, expected) -> list:
    """Which of `expected` ranks hold no claim to this identity, in given order."""
    if not expected:
        return []
    rows = _state["rank_lines"].get(identity)
    if not rows:
        return list(expected)
    return [name for name in expected if name not in rows]


def presence(identity, ranks=None) -> dict:
    """The whole presence verdict for one identity, per rank. No counters, no I/O.

    WHY `offerable` IS NOT `present`. A parked span is one file per rank and a restore
    reads every one of them, so the offer predicate is an INTERSECTION over the ranks
    that have to read: three ranks holding a block and one rank that deleted its slice
    is not a span this boot can restore, and promising it is the `file_missing` /
    `hole_in_middle` the box served. `ranks_missing` is the answer's other half: the
    same `false` that reads "never spilled" and the one that reads "PP1_TP1 evicted
    its slice" are different incidents, and only one of them is a capacity problem.
    """
    expected = _expected_ranks(ranks)
    rows = _state["rank_lines"].get(identity) or {}
    missing = _missing_ranks(identity, expected)
    entry = _state["entries"].get(identity)
    # With no rank set named and none learned from the journals there is nothing to
    # intersect, so the answer degrades to the entry table's - and the published
    # `ranks_expected: []` beside it is what says that happened.
    offerable = (bool(rows) and not missing) if expected else (entry is not None)
    return {"identity": identity,
            "present": bool(offerable),
            "ranks_expected": list(expected),
            "ranks_expected_source": _state["ranks_expected_source"],
            "ranks_present": sorted(rows),
            "ranks_missing": list(missing),
            "offerable": bool(offerable),
            "entry": entry is not None,
            "bytes": int(entry["bytes"]) if entry else 0}


# WHAT A SAMPLED AUDIT CALLS ONE IDENTITY. A closed vocabulary, and `malformed` is
# deliberately absent: this reads the index's OWN tables, not a file from elsewhere.
AUDIT_CLASSES = ("offerable", "partial_ranks", "absent", "claims_without_entry")


def audit_now() -> dict:
    """A bounded, cheap classification of a SAMPLE of the tables. Runs no I/O.

    WHY SAMPLED, AND WHY THE KEYS ARE NAMED FOR IT. The condition this measures (some
    rank holds an identity and some rank that has to read does not) is rank-shaped, and
    a box lives at hundreds of thousands of identities: classifying all of them on every
    refresh is a second job sharing the offer path's process. So one pass looks at at
    most AUDIT_SAMPLE_IDENTITIES identities every AUDIT_EVERY_N_REFRESHES refreshes, and
    every key is named `sampled_*` so no reader can take a rate for a census.

    It reads tables this process already holds - no store walk, no journal read, no
    stat - so it costs a dict walk on the refresh path that already did the work.
    """
    expected = tuple(_state["ranks_expected"])
    entries = _state["entries"]
    rank_lines = _state["rank_lines"]
    # THE LEAST-RECENTLY-TOUCHED FIRST, which is where a cap-evicted or divergent
    # region shows up first: the entries nobody has written or read for the longest
    # time are where two ranks' journals disagree, not the hot prefix.
    names = list(entries.keys())[:AUDIT_SAMPLE_IDENTITIES]
    if len(rank_lines) > len(names):
        names += [identity for identity in list(rank_lines)[:AUDIT_SAMPLE_IDENTITIES]
                  if identity not in entries][:AUDIT_SAMPLE_IDENTITIES]
    names = names[:AUDIT_SAMPLE_IDENTITIES]
    offered = partial = orphan = 0
    for identity in names:
        rows = rank_lines.get(identity) or {}
        if not rows:
            continue                       # nothing any rank claims: not a case
        if identity not in entries:
            orphan += 1                    # a claim with no entry: the cap, not a peer
            continue
        if expected and _missing_ranks(identity, expected):
            partial += 1
        else:
            offered += 1
    _state["sampled_runs"] = int(_state["sampled_runs"]) + 1
    _state["sampled_identities"] = int(_state["sampled_identities"]) + len(names)
    _state["sampled_partial_ranks"] = int(_state["sampled_partial_ranks"]) + partial
    _state["sampled_offerable"] = int(_state["sampled_offerable"]) + offered
    _state["sampled_claims_without_entry"] = (
        int(_state["sampled_claims_without_entry"]) + orphan)
    out = {"sampled": True,
           "sampled_identities": len(names),
           "sampled_offerable": offered,
           "sampled_partial_ranks": partial,
           "sampled_claims_without_entry": orphan,
           "classes": AUDIT_CLASSES,
           "of_identities": int(len(entries)),
           "every_n_refreshes": AUDIT_EVERY_N_REFRESHES,
           "ranks_expected": list(expected),
           "units": {
               "sampled_identities": "identities THIS pass looked at: a rate over a "
                                     "bounded sample, never a census of the index",
               "sampled_partial_ranks": "of those: some rank claims the identity and "
                                        "some rank that has to read does not - "
                                        "withheld, correctly, and the number to watch "
                                        "when the ranks are expected to agree",
               "sampled_offerable": "of those: every rank that has to read claims it, "
                                    "so the offer path may promise it",
               "sampled_claims_without_entry": "of those: a rank's journal claims it "
                                               "and no entry backs it (the entry cap, "
                                               "never a peer)"}}
    _state["sampled_last"] = out
    return out


def scan_identities(after=None, limit=None, rank=None, state=None) -> dict:
    """A bounded, paged, DIGEST-ONLY enumeration of the per-rank claims.

    WHY A VERB AND NOT A LOG LINE. The claim this change rests on - a rank holds a
    readable file the index will not attribute to it, or the index believes a rank whose
    file is gone - is checkable against the store, and an operator checks it by diffing
    this enumeration against `op=ls`. So it names identities and the ranks that hold
    them and NOTHING else: no paths, no sizes as a walk, no directory listing. The
    digest each rank named its file with is `disk_layout`'s geometry digest for that
    rank, which is what pairs a row with an `op=ls` line.

    BOUNDED THREE WAYS: at most `limit` identities, one heap walk over the identity
    table (no sort of 262,144 keys), and a cursor so the caller pages instead of
    asking for everything.
    """
    wanted = int(limit) if limit is not None else AUDIT_SAMPLE_IDENTITIES
    wanted = max(1, min(wanted, MAX_IDENTITIES_PER_CALL))
    only_rank = str(rank) if rank not in (None, "") else None
    want = str(state).strip().lower() if state not in (None, "") else "all"
    if want not in ("all", "offerable", "partial"):
        return _fail("`which` must be one of all, offerable, partial", identities=[])
    expected = tuple(_state["ranks_expected"])
    after_text = str(after) if after is not None else None
    rows = []
    nxt = None
    table = _state["rank_lines"]
    try:
        # A heap of the smallest `wanted + 1` identities past the cursor: O(N log
        # limit) over the table, where sorting 262,144 keys would be a stall on the
        # endpoint this is polled from.
        candidates = heapq.nsmallest(
            wanted + 1, (identity for identity in table
                         if after_text is None or str(identity) > after_text))
    except TypeError:
        return _fail("the identity table cannot be ordered", identities=[])
    for identity in candidates:
        if len(rows) >= wanted:
            nxt = str(identity)
            break
        rows.append({"identity": str(identity),
                     "ranks": sorted(table.get(identity) or {}),
                     "ranks_missing": _missing_ranks(identity, expected),
                     "offerable": bool(presence(identity)["offerable"])})
    if only_rank is not None:
        rows = [row for row in rows if only_rank in row["ranks"]]
    if want == "offerable":
        rows = [row for row in rows if row["offerable"]]
    elif want == "partial":
        rows = [row for row in rows if not row["offerable"] and row["ranks"]]
    truncated = nxt is not None or len(candidates) > wanted
    return _ok(identities=rows, count=len(rows), limit=wanted,
               truncated=bool(truncated), next=nxt,
               ranks_expected=list(expected),
               ranks_expected_source=_state["ranks_expected_source"],
               units={"identities": "identity -> the RANKS whose journal claims it; "
                                    "digests only, never a path: the digest each rank "
                                    "named its file with is what pairs a row with "
                                    "`op=ls`",
                      "next": "the identity to pass as `after` for the next page; "
                              "null when the walk reached the end",
                      "truncated": "more identities exist past this page: a census is "
                                   "this verb paged to the end, never one big call"})


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
        # PER RANK, AND ONLY PER RANK (0.2.46). The rank this line came from is the rank
        # that unlinked a file - its OWN file, named with its OWN geometry digest. Until
        # now the line retired the whole per-rank table and the global entry with it, so
        # one rank's ceiling pass withdrew a span from the three ranks still holding
        # readable files: the ~442 identities per rank measured on the box, some
        # 19.5 GB of KV no rank was allowed to read. A removal now retires the
        # removing rank's
        # claim and NOTHING else; the entry, which is the bytes/cap/LRU view, survives
        # while any rank backs it, and the OFFER answer became an intersection that
        # names the rank which actually removed something.
        _drop_rank_claim(identity, _rank_label(path))
        return REMOVE_LINE
    _state["spills"] += 1
    _insert(identity, size, path, group)
    _record_rank(identity, path, size, seq)
    # THE MIRROR OF THE SAME BUG, counted here: a spill makes the ENTRY exist, and under
    # a global presence that was the whole answer - so a peer re-spilling an identity my
    # rank had deleted made the span offerable, and the worker was handed a read for a
    # file it had unlinked (`file_missing`, `hole_in_middle`). This counts every spill
    # that did NOT complete the span across every rank that has to read: ordinary during
    # a boot drain or a divergent ceiling pass, and it used to be the offer answer too.
    # The number is here so the rate is visible, and `sampled_partial_ranks` says
    # whether any of it is still being withheld.
    if _missing_ranks(identity, tuple(_state["ranks_expected"])):
        _state["inserts_incomplete_span"] = (int(_state["inserts_incomplete_span"]) + 1)
    return SPILL_LINE


def _drop_rank_claim(identity, rank) -> bool:
    """Retire ONE RANK's claim to one identity. Returns whether a claim went.

    The entry follows only when the LAST claim goes: `entries` is the table that prices
    an identity and bounds the map, and while any rank still holds a file the identity
    is real bytes on the store. Debiting `bytes_indexed` any earlier would double-count
    against the entry's single credit in `_insert`, and a `bytes_indexed` that drifts is
    a capacity view no operator can price against the ceiling.
    """
    rank_lines = _state["rank_lines"]
    rows = rank_lines.get(identity)
    row = rows.pop(rank, None) if rows is not None else None
    if row is None:
        # A `-` for a claim this rank does not hold: a replayed line, or an eviction the
        # rank never journalled as a spill. It retires nothing, so NOTHING here moves -
        # above all not the byte accounting, which a global entry popped twice.
        _state["rank_claims_unmatched"] = int(_state["rank_claims_unmatched"]) + 1
        return False
    _state["rank_records"] = max(0, int(_state["rank_records"]) - 1)
    _state["claims_removed"] = int(_state["claims_removed"]) + 1
    kept = len(rows)
    if not rows:
        rank_lines.pop(identity, None)
    entries = _state["entries"]
    entry = entries.get(identity)
    if entry is not None and not kept:
        entries.pop(identity, None)
        _state["bytes_indexed"] = max(0, int(_state["bytes_indexed"])
                                      - int(entry["bytes"]))
    if kept:
        # THE NUMBER THIS FIX IS MEASURED BY: a peer's removal that used to take this
        # rank's file away from the index, and no longer does. The unit is the PAIR,
        # because that is the unit the box's ~442 was counted in.
        _state["claims_kept_past_peer_removal"] = (
            int(_state["claims_kept_past_peer_removal"]) + kept)
    return True


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


# --- the cap, the LRU clock, and what a drop actually costed -----------------

def entry_cap() -> dict:
    """THE CAP THIS INDEX IS HONOURING, and why it is that number.

    `config.disk_index_env()` parses; this decides. The three sources a reader can get,
    and the one thing each is telling them:

      * `env`      - the operator set `VLLM_E2_TIER_DISK_INDEX_ENTRIES` and it was
      usable.
      * `default`  - the operator said nothing, or said something unusable (the
      problem is
                     reported by `config.disk_index_env()`, and the
                     shipped cap stands - a
                     typo must never leave an index that remembers three entries).
      * clamped    - a usable number outside [floor, ceiling], with the `reason`
      naming the
                     bound that bit. Clamped rather than honoured,
                     because the two ways to
                     get this wrong are "the index forgets nearly everything" and "the
                     scheduler process ate the box", and neither is what was asked for.

    THE MEMORY, so the number can be chosen rather than guessed: ~342 bytes an entry as
    measured above, so a cap is `cap x 342 B` of map at the MOST, reached only if the
    journals actually name that many identities.

    `MAX_INDEX_ENTRIES` is read at CALL time, not bound at import, which is what lets a
    test lower the cap by setting the module attribute - and what means an
    operator's env
    value needs no restart of the module to take effect on the next admission.
    """
    parsed = config.disk_index_env()
    wanted = parsed["entries"]
    shipped = MAX_INDEX_ENTRIES
    if wanted is None:
        return {"value": int(shipped), "source": "default", "clamped": False,
                "reason": None, "floor": INDEX_ENTRIES_FLOOR,
                "ceiling": INDEX_ENTRIES_CEILING,
                "problems": list(parsed["problems"]),
                "bytes_at_cap": int(shipped) * ENTRY_BYTES_MEASURED}
    clamped = wanted
    reason = None
    if wanted < INDEX_ENTRIES_FLOOR:
        clamped = INDEX_ENTRIES_FLOOR
        reason = (f"{config.ENV_DISK_INDEX_ENTRIES}={wanted} is below the "
                  f"{INDEX_ENTRIES_FLOOR}-entry floor and was raised to it: an index "
                  "that small answers `not_present` for nearly the whole store, which "
                  "reads as configured while refusing every disk offer")
    elif wanted > INDEX_ENTRIES_CEILING:
        clamped = INDEX_ENTRIES_CEILING
        reason = (f"{config.ENV_DISK_INDEX_ENTRIES}={wanted} is above the "
                  f"{INDEX_ENTRIES_CEILING}-entry ceiling and was lowered to it: at "
                  f"~{ENTRY_BYTES_MEASURED} bytes an entry: that is the bound on the "
                  "map this process is asked to hold")
    return {"value": int(clamped),
            "source": "env" if reason is None else "env_clamped",
            "clamped": reason is not None, "reason": reason,
            "floor": INDEX_ENTRIES_FLOOR, "ceiling": INDEX_ENTRIES_CEILING,
            "requested": int(wanted), "problems": [],
            "bytes_at_cap": int(clamped) * ENTRY_BYTES_MEASURED}


def _touch(identity) -> bool:
    """Move one entry to the BACK of the eviction queue. A touch, not an admission.

    WHY `OrderedDict.move_to_end` AND NOT pop-then-set. This is called from a reader
    thread (through `note_restored_path`, from `disk_read.promote_recency`) while the
    scheduler thread may be applying a journal's `-` line for the SAME identity. A
    pop and
    re-insert can therefore RESURRECT an entry the journal just removed - a ghost the
    offer path would offer for a file that is gone, which is the corruption
    direction, not
    the waste direction. `move_to_end` relinks without ever dropping the key from the
    mapping, so a concurrent removal wins and a touch on a vanished entry raises
    `KeyError`, which is caught here and costs nothing.

    Returns whether a move happened. A touch for an identity this index does not hold is
    NOT an admission: the cap bounds memory, and a clock that grew the dict would be a
    leak with a nice name.
    """
    entries = _state["entries"]
    try:
        entries.move_to_end(identity)
    except (KeyError, AttributeError):
        return False
    _state["entries_touched"] = int(_state["entries_touched"]) + 1
    return True


def note_restored_path(path) -> dict:
    """A file at `path` produced usable bytes: touch the index entry that names it.

    WHY THE FILE PATH IS ENOUGH. A committed slot name carries the whole identity -
    `kvslot-<key16>-g<group>-<geodigest12>.bin` - so the index's own key is recoverable
    from the name the reader already holds, with one parse and no I/O. That is what lets
    the read-side touch hang off the SINGLE place the tier admits a file was used
    (`disk_read.promote_recency`, called for a read that produced bytes and for no
    failed
    read), instead of inventing a second, drift-prone notion of "recently restored".

    THE POINT IS SYMMETRY: the file's clock is `st_mtime` bumped at that same call,
    so if
    the index were left on insertion order the two clocks would mean different
    things and
    the tier would evict a span's MAP while keeping the span - the exact inversion this
    module exists to remove.
    """
    out = {"promoted": False, "key16": None, "group_id": None, "reason": None}
    try:
        parsed = disk_layout.parse_path(path)
        if not parsed["ok"]:
            out["reason"] = str(parsed["reason"])
            return out
        key16, group = parsed.get("key16"), parsed.get("group_id")
        if not isinstance(key16, str) or not _is_plain_int(group):
            out["reason"] = ("the name carries no complete identity, so no index entry "
                             "can be moved by it")
            return out
        out["key16"], out["group_id"] = key16, int(group)
        identity = identity_of(key16, int(group))
        if identity not in _state["entries"]:
            # The common case for a span this rank restored from a file the index never
            # held (a boot that started after the spill). Not an admission - see _touch.
            out["reason"] = ("this index holds no entry for this identity: a touch "
                             "never admits, because the cap is what bounds memory")
            return out
        out["promoted"] = _touch(identity)
        _state["restores_noted"] = int(_state["restores_noted"]) + 1
        return out
    except Exception as exc:  # noqa: BLE001 - a touch is never worth a step fault
        _state["last_error"] = f"{type(exc).__name__}"[:MAX_STRING_CHARS]
        snapshot._error(exc, "disk_index note_restored_path")
        out["reason"] = f"the touch raised {type(exc).__name__}"
        return out


# --- the presence scan: did a dropped entry still have a file? ---------------

# ONE per process, reset at the top of every refresh: the listings this refresh paid
# for, and how much of its shard budget is left. Not module state that ages - a scan
# budget that carries over between refreshes would starve a later burst for no reason.
_presence: dict = {"refresh": 0, "listings": {}, "budget": 0}


def _store_root_for_presence():
    """The store root to ask about files, or None. Vetted, never created."""
    raw = os.environ.get(config.ENV_DISK_ROOT, "").strip()
    if not raw:
        return None
    checked = disk_layout.validate_root(raw)
    return checked["value"] if checked["ok"] else None


def _slot_names_in(directory, budget: int) -> set | None:
    """The identities ONE shard directory names, or None if it could not be listed.

    Bounded by `budget` entries and never recursive. The names are reduced to the
    index's own identity spelling with string slicing on the fixed shape
    `kvslot-<key16>-g<group>-<digest>.bin`, because calling `parse_path` on up to
    `PRESENCE_SCAN_MAX_ENTRIES` names is a per-name parse on the refresh path.
    """
    out: set = set()
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if len(out) >= budget:
                    break
                name = entry.name
                if not name.startswith(disk_layout.SLOT_FILE_PREFIX) \
                        or not name.endswith(disk_layout.SUFFIX_DATA):
                    continue
                body = name[len(disk_layout.SLOT_FILE_PREFIX):
                            -len(disk_layout.SUFFIX_DATA)]
                fields = body.split("-")
                if len(fields) != 3:
                    continue
                key16, group_text = fields[0], fields[1]
                if not group_text.startswith(disk_layout.GROUP_FIELD_PREFIX):
                    continue
                try:
                    group = int(group_text[len(disk_layout.GROUP_FIELD_PREFIX):])
                except ValueError:
                    continue
                if len(key16) != disk_layout.KEY_DIGEST_CHARS:
                    continue
                out.add(f"{key16}{IDENTITY_SEP}{GROUP_PREFIX}{group}")
    except OSError:
        return None
    return out


def _classify_drop(identity) -> str:
    """Was the file one dropped entry described still on the store? Bounded, once.

    `present` / `absent` only from an ACTUAL listing; anything else is `unverified`,
    which is its own counter and never a stand-in for either. Called from the cap-drop
    loop, so the whole cost of a 800-drop burst is at most
    `PRESENCE_SCAN_MAX_SHARDS_PER_REFRESH` listings of shards that burst touched.
    """
    fields = str(identity or "").split(IDENTITY_SEP)
    key16 = fields[0]
    shard = key16[:disk_layout.SHARD_CHARS]
    if len(key16) != disk_layout.KEY_DIGEST_CHARS:
        return "unverified"
    root = _store_root_for_presence()
    if root is None:
        _state["presence_scan_skipped"] = int(_state["presence_scan_skipped"]) + 1
        return "unverified"
    listings = _presence["listings"]
    if shard in listings:
        found = listings[shard]
        if found is None:
            return "unverified"
        return "present" if identity in found else "absent"
    if _presence["budget"] <= 0:
        _state["presence_scan_skipped"] = int(_state["presence_scan_skipped"]) + 1
        return "unverified"
    _presence["budget"] -= 1
    _state["presence_scans"] = int(_state["presence_scans"]) + 1
    found = _slot_names_in(os.path.join(root, shard), PRESENCE_SCAN_MAX_ENTRIES)
    listings[shard] = found
    if found is None:
        return "unverified"
    return "present" if identity in found else "absent"


def _begin_presence_refresh() -> None:
    """A fresh scan budget for a fresh refresh. Never carries a listing forward.

    A listing is deliberately NOT reused across refreshes: the whole question is whether
    a file is on the store NOW, and a cache would answer that with last refresh's truth.
    """
    _presence["refresh"] = int(_state["refreshes"])
    _presence["listings"] = {}
    _presence["budget"] = PRESENCE_SCAN_MAX_SHARDS_PER_REFRESH


def _insert(identity, size, path, group_id=None) -> None:
    """Add or refresh one entry, honouring the cap. Keyed by IDENTITY, ordered by TOUCH.

    THE ORDER IS THE POINT (0.2.45). The queue is ordered by LAST TOUCH, so the
    entry the
    cap evicts is the one nobody wrote or read for the longest time. It used to be
    `next(iter(entries))` of an INSERTION-ordered dict, which evicts the entry this rank
    has restored most often - the exact inverse of the file-side LRU, which ranks by the
    `st_mtime` that `disk_read.promote_recency` bumps on a read. A refresh of an
    existing
    identity is therefore a touch too: the journal just said it wrote that key again.
    """
    entries = _state["entries"]
    existing = entries.get(identity)
    if existing is not None:
        existing["bytes"] = int(size)
        existing["journal"] = path
        existing["seen_ts"] = round(time.time(), 6)
        # A RE-SPILL IS A WRITE, so it is a touch: without this the LRU would only ever
        # move for readers, and a rank that only writes would have an insertion-ordered
        # index wearing an LRU's name.
        _touch(identity)
        return
    cap = entry_cap()["value"]
    while len(entries) >= cap:
        oldest = next(iter(entries))
        dropped = entries.pop(oldest)
        _drop_identity_ranks(oldest)
        _state["entries_dropped"] = int(_state["entries_dropped"]) + 1
        # WHAT THE DROP COST. `entries_dropped` alone could not tell "the file was
        # reclaimed long ago" from "this span is on the RAID right now and the index has
        # stopped offering it", and those are different incidents - one is bookkeeping,
        # the other is capacity burned on a span the tier refuses to read.
        verdict = _classify_drop(oldest)
        if verdict == "present":
            _state["entries_dropped_file_present"] = int(
                _state["entries_dropped_file_present"]) + 1
        elif verdict == "absent":
            _state["entries_dropped_file_absent"] = int(
                _state["entries_dropped_file_absent"]) + 1
        else:
            _state["entries_dropped_unverified"] = int(
                _state["entries_dropped_unverified"]) + 1
        _state["bytes_indexed"] = max(0, int(_state["bytes_indexed"])
                                      - int(dropped["bytes"]))
        _state["last_reason"] = (f"the index is at its {cap}-entry "
                                 f"cap: {oldest} was dropped to admit a newer "
                                 "identity (a dropped entry is a missed restore)")
        if notice_once(
                "index-entry-cap",
                f"the disk index reached its {cap}-identity cap and is DROPPING THE "
                f"LEAST-RECENTLY-TOUCHED entries to admit newer ones "
                f"(entries_dropped: {int(_state['entries_dropped'])}, of which "
                f"{int(_state['entries_dropped_file_present'])} name a file that is "
                "STILL ON THE STORE - a missed restore each). Raise "
                f"{config.ENV_DISK_INDEX_ENTRIES} (now: "
                f"{entry_cap()['source']}); a converged drain over a big store can "
                "live at this cap. The section publishes `entry_cap_hit: true`, "
                "`entries_dropped_file_present` and "
                "`entries_dropped_unverified` beside the count, the last of those "
                "being the drops the bounded presence scan could not check."):
            _state["entry_cap_hit"] = True
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
        identity = identity_of(key16, group)
        entry = _state["entries"].get(identity)
        if entry is None:
            _state["misses"] += 1
            return _ok(present=False, key16=key16, group_id=group, bytes=0,
                       rank=None,
                       reason="this boot's journal does not name a file for "
                              "this key and group")
        _state["hits"] += 1
        # THE READ-SIDE TOUCH, for the caller that counts a hit. `lookup()` is
        # the accessor
        # that answers "may I restore this?" and moves the witness's ledger,
        # so it is the
        # one index entry point whose probe means "this identity is in use"; the
        # non-ledger probes (`contains`, `entry_bytes`, `entry_for`, `membership`) stay
        # side-effect-free on purpose, because their whole contract is that they do not
        # move a counter - and an offer loop probes several groups per span,
        # so promoting
        # there would move the clock on identities that were never offered.
        _touch(identity)
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


def entry_bytes(key) -> int | None:
    """The size one identity's file was journalled at, in bytes. None if unknown.

    WHY THIS EXISTS BESIDE `lookup()`: the offer path must PRICE a read before it
    offers one (a read costs one bounce buffer of exactly this many bytes), and it
    must not move `hits`/`misses` - those are the witness's ledger and an experiment
    reads them as evidence of offer-path behaviour. `contains()` is the same rule for
    a yes/no question; this is the same rule for the size the writer journalled, read
    straight out of the entry the journal already built. No I/O, no counters, and
    None rather than 0: a zero price is a number a caller multiplies.
    """
    try:
        split = disk_layout.split_block_key(key)
        if not split["ok"]:
            return None
        entry = _state["entries"].get(identity_of(split["key16"],
                                                  split["group_id"]))
        if entry is None:
            return None
        size = entry.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            return None
        return int(size)
    except Exception as exc:  # noqa: BLE001 - a price is never worth a fault
        snapshot._error(exc, "disk_index entry_bytes")
        return None


def entry_for(key16, group_id) -> dict:
    """`{present, bytes, seen_ts, journal, reason}` for a `key16` + group. No counters.

    WHY THIS EXISTS WHEN `lookup()` AND `entry_bytes()` EXISTED. Both of those take
    the ENGINE's full group-qualified block hash, because that is what the offer path
    holds. A load-failure record holds the other half of that split: a 16-hex `key16`
    and a group id, which is what a spilled PATH names and therefore what a worker
    can report across a process boundary. Synthesizing a full key out of those two is
    not possible - the digest between the `key16` and the group suffix is exactly the
    part that was dropped - so the honest accessor takes the two fields it actually
    has. Like `entry_bytes`, it moves no `hits`/`misses` (those are the witness's
    ledger) and does no I/O: this is the entry the journal already built.
    """
    try:
        key = str(key16 or "").strip().lower()
        group = group_id
        if isinstance(group, bool) or group is None:
            return {"present": False, "bytes": None, "seen_ts": None,
                    "journal": None,
                    "reason": "a key16 without a group id names no identity"}
        try:
            group = int(group)
        except (TypeError, ValueError):
            return {"present": False, "bytes": None, "seen_ts": None,
                    "journal": None, "reason": f"group {group_id!r} is not an int"}
        if not key:
            return {"present": False, "bytes": None, "seen_ts": None,
                    "journal": None, "reason": "no key16 was given"}
        entry = _state["entries"].get(identity_of(key, group))
        if entry is None:
            return {"present": False, "bytes": None, "seen_ts": None,
                    "journal": None,
                    "reason": "this boot's index holds no entry for this identity"}
        return {"present": True, "bytes": entry.get("bytes"),
                "seen_ts": entry.get("seen_ts"),
                "journal": entry.get("journal"), "reason": None}
    except Exception as exc:  # noqa: BLE001 - a provenance read is never fatal
        snapshot._error(exc, "disk_index entry_for")
        return {"present": False, "bytes": None, "seen_ts": None, "journal": None,
                "reason": f"reading the entry raised {type(exc).__name__}"}


def contains(key, ranks=None) -> bool:
    """The offer predicate: does EVERY RANK THAT HAS TO READ hold this identity?

    `lookup()` counts hits and misses for the witness; the offer loop probes the SAME
    chain for several groups and must not inflate those counters, so it asks this
    instead. It still resolves the identity through `split_block_key` - an ungrouped key
    is not present, and answering "present" to one would hand the offer loop exactly the
    ambiguity the identity exists to remove. Cheaper than `lookup()` by the counters and
    the dict build, not by the parsing.

    SINCE 0.2.46 THIS IS AN INTERSECTION, AND THAT IS THE WHOLE FIX. A parked span is
    one file per rank and a restore reads all of them, so a single rank's absence is
    enough to answer False: asked about `ranks` (or, when the caller names none, about
    the ranks its own journals name), the index answers for THOSE ranks and nobody else.
    A peer's `+` line can no longer make a span offerable for a rank that deleted its
    slice - the false present that reached a worker as `file_missing` - and a peer's `-`
    line can no longer deny the ranks that kept their files.
    """
    try:
        split = disk_layout.split_block_key(key)
        if not split["ok"]:
            return False
        identity = identity_of(split["key16"], split["group_id"])
        expected = _expected_ranks(ranks)
        rows = _state["rank_lines"].get(identity)
        if not expected:
            # Nobody has named a rank set: the entry table is all there is to answer
            # with. Say so where the answer is read - the published
            # `ranks_expected: []` is this branch.
            return identity in _state["entries"]
        if not rows:
            return False
        return not _missing_ranks(identity, expected)
    except Exception:  # noqa: BLE001 - a membership test cannot be fatal
        return False


def membership(key, ranks=None) -> dict:
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
                "lines_skipped": int(_state["lines_skipped"]),
                # THE DRAIN: `converged: false` means a cursor sits short of its
                # journal and the named `unread_journals` WILL be read; a
                # `not_present` while it is false is the reader's silence, not the
                # store's. Non-zero `catch_up_jumps`/`bytes_skipped` means an older
                # build forfeited history - that region is not knowable at all.
                "converged": bool(_state["converged"]),
                "boot_drained": bool(_state["boot_drained"]),
                "unread_journals": _unread_view(),
                "bytes_read": int(_state["bytes_read"]),
                "entry_cap_hit": bool(_state["entry_cap_hit"])}
        blank = {"key16": None, "group_id": None, "identity": None, "shard": None,
                 "bytes": 0, "bytes_all_ranks": 0, "journal": None, "rank": None,
                 "ranks": [], "ranks_present": 0, "ranks_missing": [],
                 # WHO THIS ANSWER IS INTERSECTED OVER (0.2.46), beside the answer in
                 # every branch: an intersection is only readable if the reader can see
                 # the sets it joined. `journal files` means the index learned the rank
                 # set from the journals it drains; the offer path asks with the ranks
                 # the worker proof names, and passes them in.
                 "ranks_expected": list(_expected_ranks(ranks)),
                 "ranks_expected_source": _state["ranks_expected_source"],
                 "offerable": False, "index": view}
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
        # THE INTERSECTION, spelled out beside the answer (0.2.46). `present` IS the
        # offer answer, so a span some rank cannot read answers `false` here and the
        # two views can never disagree; `offerable` is the same verdict by name, and
        # `ranks_missing` is the part that ends an investigation instead of starting
        # one - "three ranks hold it and PP1_TP1 deleted its slice" and "nobody ever
        # spilled this" are different incidents that used to print the same three words.
        expected = _expected_ranks(ranks)
        missing = _missing_ranks(identity, expected)
        offerable = (bool(rows) and not missing) if expected else (entry is not None)
        found = {**blank, **located, "ok": True, "ranks": rows,
                 "ranks_present": len(rows), "bytes_all_ranks": total,
                 "ranks_missing": list(missing), "offerable": bool(offerable)}
        if rows and expected and missing:
            return {**found, "present": False, "verdict": VERDICT_PARTIAL_RANKS,
                    # THE ENTRY STILL PRICES THESE FILES. A partial span is not an
                    # absent one: the peers' slices are real bytes on the store, the
                    # capacity view still counts them, and a reader comparing this
                    # answer with `op=ls` needs the size to do it.
                    "bytes": int(entry["bytes"]) if entry else 0,
                    "journal": str(entry.get("journal") or "") if entry else "",
                    "rank": _rank_of(entry.get("journal")) if entry else None,
                    "reason": (f"{len(rows)} of {len(expected)} rank(s) that have to "
                               f"read claim this identity; {', '.join(missing)} holds "
                               "no file for it, so a restore would read a slice that "
                               "is not there: the span is withheld until that rank "
                               "spills it again"),
                    "answer": (f"not offered: missing for "
                               f"{', '.join(missing)} of "
                               f"{', '.join(expected)}")}
        if entry is not None and offerable:
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
                     group_id=None, ranks=[], ranks_present=0, ranks_missing=[],
                     offerable=False)


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
    _state["entries"] = OrderedDict()
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
    cap = entry_cap()
    return {
        "entries": int(len(_state["entries"])),
        # THE CAP IN FORCE, plus where it came from: `default`, `env`, or `env_clamped`
        # (with `entry_cap_reason`). An operator raising it needs to see that the raise
        # landed, and a reader needs to know the number was not silently clamped.
        "entry_cap": cap["value"],
        "entry_cap_source": cap["source"],
        "entry_cap_at_bytes": cap["bytes_at_cap"],
        "entry_cap_reason": cap["reason"],
        # THE LRU CLOCK (0.2.45): touches move an entry to the back of the
        # eviction queue.
        # A cap drop evicts the least-recently-TOUCHED entry, where a touch is
        # a spill line
        # that re-names an identity or a restore that produced bytes (the SAME
        # event that
        # bumps the file's `st_mtime`). `entries_touched: 0` with a non-zero
        # `entries_dropped` would mean the clock never moved.
        "entries_touched": int(_state["entries_touched"]),
        "restores_noted": int(_state["restores_noted"]),
        # WHAT THE DROPS COST, balancing `entries_dropped` exactly. See the
        # counters block.
        "entries_dropped_file_present": int(
            _state["entries_dropped_file_present"]),
        "entries_dropped_file_absent": int(
            _state["entries_dropped_file_absent"]),
        "entries_dropped_unverified": int(_state["entries_dropped_unverified"]),
        "presence_scans": int(_state["presence_scans"]),
        "presence_scan_skipped": int(_state["presence_scan_skipped"]),
        # THE PER-RANK VIEW `op=contains` answers from: how many (identity, rank) pairs
        # are held, and how many a cap refused to remember. A dropped pair is a span
        # that reads as fewer ranks than it has - never a false presence, always a
        # claim this index can no longer back.
        "rank_records": int(_state["rank_records"]),
        "rank_records_dropped": int(_state["rank_records_dropped"]),
        "rank_cap_per_identity": MAX_RANKS_PER_IDENTITY,
        # PER-RANK PRESENCE (0.2.46): WHO the answer is intersected over, and the two
        # lies this table used to tell, counted at apply time. `ranks_expected_source`
        # is part of the number - "journal files" is the index's own knowledge, and the
        # offer path prefers the ranks the worker proof names.
        "ranks_expected": list(_state["ranks_expected"]),
        "ranks_expected_source": _state["ranks_expected_source"],
        "claims_removed": int(_state["claims_removed"]),
        "claims_kept_past_peer_removal": int(_state["claims_kept_past_peer_removal"]),
        "rank_claims_unmatched": int(_state["rank_claims_unmatched"]),
        "inserts_incomplete_span": int(_state["inserts_incomplete_span"]),
        # THE SAMPLED AUDIT: a rate over at most AUDIT_SAMPLE_IDENTITIES identities per
        # pass, never a census of the store. Named `sampled_*` so a reader cannot take
        # one for a total.
        "sampled_runs": int(_state["sampled_runs"]),
        "sampled_identities": int(_state["sampled_identities"]),
        "sampled_partial_ranks": int(_state["sampled_partial_ranks"]),
        "sampled_offerable": int(_state["sampled_offerable"]),
        "sampled_claims_without_entry": int(_state["sampled_claims_without_entry"]),
        "sampled_last": _state["sampled_last"],
        "membership_checks": int(_state["membership_checks"]),
        "entries_dropped": int(_state["entries_dropped"]),
        # THE ENTRY CAP, AS ITS OWN TRUTH: once true, this index answers
        # `not_present` for identities whose files are on disk and were simply
        # dropped to make room. A capped index is not a caught-up index.
        "entry_cap_hit": bool(_state["entry_cap_hit"]),
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
        # THE DRAIN, AS DATA. `converged` is the one answer to "is this index done
        # reading its journals?" - `lag_bytes: 0` alone could also mean a forfeit
        # landed the cursor at EOF, which is exactly the blindness that hid the
        # `entries: 146` for hours. `unread_journals` names each still-draining
        # journal and its unread bytes; `bytes_read` totals what this process has
        # actually pulled out of them.
        "converged": bool(_state["converged"]),
        "boot_drained": bool(_state["boot_drained"]),
        "unread_journals": _unread_view(),
        "bytes_read": int(_state["bytes_read"]),
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
                  "entry_cap_at_bytes": "bytes the map costs AT the cap (~342 B an "
                                        "entry, measured; the dict only grows to what "
                                        "the journals name)",
                  "entry_cap_source": "default | env | env_clamped: who decided "
                                      "the cap",
                  "entries_touched": "touches that moved an entry to the back of the "
                                     "eviction queue (the LRU clock: a write or a "
                                     "restore)",
                  "restores_noted": "restores this index was told produced bytes",
                  "entries_dropped_file_present": "of the drops: the ones whose file "
                                                  "is STILL on the store: capacity "
                                                  "spent on spans the index will not "
                                                  "offer",
                  "entries_dropped_file_absent": "of the drops: the ones whose file "
                                                 "was already reclaimed: bookkeeping, "
                                                 "not a loss",
                  "entries_dropped_unverified": "of the drops: the ones the bounded "
                                                "presence scan did not check (no store "
                                                "root, or the per-refresh shard budget "
                                                "was spent)",
                  "presence_scans": "shard directory listings this index performed to "
                                    "answer the two above (bounded per refresh)",
                  "presence_scan_skipped": "drops answered without a listing",
                  "rank_records": "(identity, rank) pairs: one parked span is one "
                                  "file PER RANK, so an identity is backed by as many "
                                  "records as it has ranks",
                  "rank_records_dropped": "rank records a cap refused: the identity "
                                          "may read as fewer ranks than it "
                                          "has (never a false presence)",
                  "rank_cap_per_identity": "ranks remembered per identity",
                  "ranks_expected": "rank labels a presence answer is intersected "
                                    "over: the ranks that would have to READ a span, "
                                    "learned from the journal file names this drain "
                                    "reads (the offer path asks with the ranks the "
                                    "worker proof names instead)",
                  "ranks_expected_source": "journal files | None: who said so",
                  "claims_removed": "removal lines that retired the claiming rank's "
                                    "own claim: what a `-` line has always meant",
                  "claims_kept_past_peer_removal": "(identity, rank) pairs that "
                                                   "survived another rank's removal "
                                                   "line: the withheld-identity "
                                                   "condition this fix exists for, "
                                                   "counted instead of served",
                  "rank_claims_unmatched": "removal lines for a claim that rank did "
                                           "not hold: they retire nothing and move no "
                                           "byte",
                  "inserts_incomplete_span": "spill lines whose identity is not held "
                                             "by every rank that has to read: ordinary "
                                             "in a drain, and the answer that used to "
                                             "be the offer answer",
                  "sampled_runs": "audit passes since boot (at most a bounded sample "
                                  "of identities each; no I/O, no store walk)",
                  "sampled_identities": "identities the audit has classified IN TOTAL: "
                                        "a rate over a sample, NOT a count of the "
                                        "index",
                  "sampled_partial_ranks": "of those: some rank claims the identity "
                                           "and some rank that has to read does not - "
                                           "withheld, correctly, and the number to "
                                           "watch when the ranks are expected to agree",
                  "sampled_offerable": "of those: every rank that has to read "
                                       "claims it",
                  "sampled_claims_without_entry": "of those: a rank's journal "
                                                  "claims it and no entry backs it "
                                                  "(the entry cap, never a peer)",
                  "sampled_last": "the last audit pass's own numbers, with its sample "
                                  "size and the rank set it intersected over",
                  "membership_checks": "answers `op=contains` has given here",
                  "entries_dropped": "identities: the ones the entry cap evicted (a "
                                     "missed restore); balanced by the three "
                                     "entries_dropped_file_* / _unverified keys",
                  "bytes_indexed": "bytes",
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
                  "lines_skipped": "journal lines permanently unread: the lines "
                                   "after a stream that stopped being trustworthy "
                                   "(a journal the drain can never finish)",
                  "converged": "true when every journal cursor sits at "
                               "end-of-file; while false a not_present is the "
                               "READER being behind, not the store being empty",
                  "boot_drained": "whether the first-refresh drain has run in this "
                                  "process",
                  "unread_journals": "per-journal bytes the cursor has not reached "
                                    "yet (unit: bytes; named newest-lagging first, "
                                    "capped at MAX_LAG_NAMED_JOURNALS names)",
                  "bytes_read": "bytes this process has read out of the journals "
                                "since boot (unit: bytes)",
                  "entry_cap_hit": "true once the MAX_INDEX_ENTRIES cap has dropped "
                                   "even one identity: a capped index answers "
                                   "not_present for files that ARE on disk - a "
                                   "capped index is not a caught-up index",
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
                 "worker resolves the name IT would have written. Read `converged` "
                 "first: while it is false this index has not finished reading its "
                 "journals (`lag_bytes` and `unread_journals` name the bytes and the "
                 "files), and a `not_present` is then a statement about the reader, "
                 "not the store - spilled-but-not-yet-indexed, not never-spilled.")
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
