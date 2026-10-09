"""Environment gating and the one path this plugin is ever allowed to load.

Import of this module has no side effects: it reads no environment value at
import time. Every gate is a function called per request / per plugin load, so
the operator can turn the plugin on and off with an env var and a restart of
the *reader* of that env (the engine process), never a re-install.

Env vars (this fork's `VLLM_E2_*` diagnostic family; see vllm/envs.py:392-456
for `VLLM_E2_MEMSTATS`, `VLLM_E2_SCHED_LOG`, `VLLM_E2_GUARD_MODE`, ...):

  VLLM_E2_TIER_DEV          -- master switch. Anything not in
                               {"", "0", "false", "off", "no"} enables it.
                               Unset = the plugin is inert: the general plugin
                               attaches nothing, the endpoint answers 503.
  VLLM_E2_TIER_DEV_TOKEN    -- required bearer token for the ONE mutating verb
                               (policy reload). Unset/empty = reload is refused
                               (fail closed), reads still work.
  VLLM_E2_TIER_DEV_TIMEOUT  -- seconds the HTTP layer waits for the engine
                               utility call before answering 504. Default 5.0,
                               clamped to [0.05, 60.0]; garbage falls back.
  VLLM_E2_TIER_DEV_BOOT_INERT -- THE BOOT OPT-OUT. UNSET (the default) = the
                               boot brings the tier up WORKING: at connector
                               creation the plugin installs PR #56810, arms GUARD
                               MODE, and arms the tier -- and only arms it, if the
                               install VERIFIES (all methods patched, live object
                               safe). Set it to anything not in
                               {"", "0", "false", "off", "no"} to get the old
                               inert boot back: patched at connector creation, no
                               guard, tier left disarmed until a human POSTs
                               {"op":"arm"}. It never ARMS anything by itself:
                               truthy = inert, falsy = auto. `status` and the
                               engine's BOOT log line both say which mode ran.
  VLLM_E2_TIER_DEV_ALLOW    -- comma-separated list of IP addresses or CIDR
                               networks allowed in ADDITION to loopback, e.g.
                               `192.168.2.64,192.168.2.120` or `192.168.2.0/24`.
                               UNSET OR EMPTY = LOOPBACK ONLY, which is the
                               pre-allowlist behaviour and the safe default.
                               A malformed entry is logged and skipped, never
                               fatal and never an opening. A peer admitted here
                               must still present VLLM_E2_TIER_DEV_TOKEN on
                               every request, reads included (loopback peers do
                               not need it for reads). Traffic is plaintext
                               HTTP, so keep this list as narrow as possible.

There is deliberately NO env var that selects the policy file. The reloadable
policy is `kvtier_dev_plugin/policy.py` next to this file, and nothing else.

THE WORKER DIAGNOSTICS FAMILY (all optional; see `worker_diag.py`'s docstring for
the file contract the proxy host's reader expects):

  VLLM_E2_TIER_DIAG              where one JSON file per rank is written, on the
                                 ENGINE's host. UNSET = nothing is wrapped and no
                                 file is ever created, which is the default.
  VLLM_E2_TIER_DIAG_EPOCH        the boot id shared by every writer. Set it once
                                 per boot if you want the reader's cross-rank
                                 verdicts; unset, each process stamps its own id
                                 and says `writer.epoch_scope: "process"`.
  VLLM_E2_TIER_DIAG_INTERVAL     write cadence in steps (default 50, clamped).
  VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS
                                 the rate floor under it (default 1.0 s), so a
                                 changing signature cannot make this a per-step
                                 writer.
  VLLM_E2_TIER_DIAG_RANK         pin the rank id (`PP<n>_TP<m>`) instead of
                                 deriving it from the process's distributed state.
  VLLM_E2_TIER_MISS_LOG          a path to append one JSON line per tier lookup
                                 to. Unset = no file, no open(), no I/O of any
                                 kind (see miss_attribution.py).
  VLLM_E2_TIER_DISK_ROOT         the SSD tier's store root, one file per KV
                                 slot. UNSET = the tier cannot exist: no
                                 directory is created, no thread starts, no byte
                                 is written. tmpfs roots are refused.
  VLLM_E2_TIER_DISK_BYTES        the tier's byte ceiling in bytes; above it files
                                 are shed LRU by st_mtime - a TOUCH clock since
                                 0.2.29 (a write, or a restore read that produced
                                 usable bytes). Unset = no ceiling, NEVER zero.
                                 Neither variable ARMS the tier: the arm is
                                 `disk_spill.arm()`. Since 0.2.35 that arm is a
                                 LATCH - no expiry and nothing that renews it, so
                                 it comes down on a disarm POST and not on a
                                 clock, which is unlike the raw-byte dump's
                                 (still a TTL). Disarming stops serving from disk
                                 as well as writing to it.
  VLLM_E2_TIER_DISK_JOURNAL_DIR  where the spill JOURNAL lives, apart from the rest
                                 of the diagnostics directory. UNSET = fall back to
                                 VLLM_E2_TIER_DIAG (the pre-0.2.19 behaviour).
                                 Set it to a path on a filesystem that SURVIVES A
                                 REBOOT: the journal is the only map from a KV
                                 identity to a spilled file, and a reboot that eats
                                 it while the files survive leaves the restore index
                                 at `entries: 0`. Only the JOURNAL moves here - the
                                 arm file stays in the diagnostics directory and
                                 dies at boot, on purpose. A journal directory on
                                 tmpfs is warned about once and reported as
                                 `journal_dir=tmpfs` on the boot line.
  THE READER THREADS AND THE BOUNCE-BUFFER POOL (`disk_read.py`) -- both inert by
  default, both bytes-identical to the single-thread release when unset:

  VLLM_E2_TIER_DISK_READ_THREADS      how many threads issue spill reads at once.
                                      UNSET or `1` = ONE thread, exactly the
                                      behaviour of every release before 0.2.24.
                                      Clamped to [1, 16]; a value that is not a
                                      positive integer means "the operator said
                                      nothing", i.e. one thread, and is reported in
                                      `read.threads.env_problems`.
  VLLM_E2_TIER_DISK_READ_BUFFER_BYTES the bounce-buffer pool's byte ceiling. UNSET
                                      = `disk_read.MAX_BUFFERS_BYTES` (96 MiB),
                                      today's value. INCREASING IT COSTS HOST RAM
                                      PER WORKER PROCESS: the pool is per-process
                                      and every engine rank is its own process, so
                                      +64 MiB here is +64 MiB x rank count of
                                      pinned-adjacent anonymous memory. See
                                      `disk_read_env` and README's
                                      *Reading the spill in parallel*.

  THE CEILING PASS'S TIME BUDGET (`disk_spill.py`) -- one line, because it protects
  the queue rather than the disk:

  VLLM_E2_TIER_DISK_CEILING_BUDGET_SECONDS
                                      SECONDS one enforcement pass may spend walking
                                      and unlinking on the spill writer thread.
                                      UNSET = `disk_spill.CEILING_PASS_BUDGET_SECONDS`
                                      (1.5 s). When it expires the pass stops early
                                      and reports `plan_truncated`; the next spill
                                      continues the work. A value that is not a
                                      positive number means "the operator said
                                      nothing" and is reported in
                                      `spill.ceiling_measurement.env_problems`.
                                      0 is REFUSED (it would make every pass do
                                      nothing, which is how a ceiling stops being a
                                      ceiling).

  VLLM_E2_TIER_DISK_EVICT_THREADS  how many threads unlink at once inside one
                                      ceiling pass. UNSET or `1` = ONE, the serial
                                      behaviour of every release before 0.2.29.
                                      Clamped to [1, `disk_spill.MAX_EVICT_THREADS`];
                                      the pool is built ONCE PER PROCESS, so changing
                                      this needs a restart (the payload reports
                                      `evict_threads_requested` beside
                                      `evict_threads_configured` when they differ).

  THE ONE BOUND ON A DELIVERY STEP (`disk_gpu_write.py`, 0.2.57) ----------------

  VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS
                                      SECONDS one delivery step may hold the engine's
                                      step thread, checked before each entry. UNSET =
                                      `disk_gpu_write.DELIVERY_APPLY_TIMEOUT_SECONDS`
                                      (10 s). ONE generous timeout, deliberately, and
                                      NOT a per-entry budget: 0.2.56's 0.25 s step
                                      budget, its 64 MiB byte budget and the refusal
                                      that went with it are GONE. A step that reaches
                                      the bound DEFERS the rest of its entries, and a
                                      deferred entry is delivered by the FORK - a
                                      missed optimization, never a hole. Clamped at
                                      `DELIVERY_APPLY_TIMEOUT_MAX` (60 s) and reported
                                      (`delivery_mode.timeout_clamped`); a value that
                                      is not a finite number above 0 means the shipped
                                      default and is logged once. Read PER APPLY
                                      POINT, so a drop-in needs no restart.

  THE CPU-SLOT RESERVE AN OFFER LEAVES ALONE (`disk_restore.py`, 0.2.38) ------

  VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE
                                      CPU slots of the host-RAM arena's block pool that
                                      an offer must NOT take. UNSET = `disk_restore.
                                      CPU_SLOT_RESERVE` (64), the value every release
                                      before 0.2.38 used and the value an unset boot
                                      still uses. LOWER IT TO MAKE OFFERS BIGGER on a
                                      small arena: at 166 free slots per rank a reserve
                                      of 64 leaves 98 spans of an offer, a reserve of
                                      16 leaves 146 (before the 128-block walk cap).
                                      0 IS HONOURED = "an offer may take the arena's
                                      last slot", which leaves the RAM tier nothing to
                                      park with. A NON-INTEGER or a NEGATIVE value is
                                      refused, falls back to 64, and is announced once
                                      as `DISK[reserve] IGNORED`; there is NO ceiling,
                                      because a reserve larger than the arena correctly
                                      declines every offer as `no_cpu_slots`. See
                                      `cpu_slot_reserve_env` and README's
                                      *Sizing the CPU-slot reserve*.

  THE DISK TIER'S BOOT DEFAULT (`disk_boot.py`, 0.2.33) -----------------------

  KVTIER_DISK_BOOT_ENABLE    the SSD tier's BOOT posture, and the ONE flag that
                             turns it off. UNSET (the default) = the boot comes up
                             ARMED AND SERVING: at connector creation the plugin
                             arms the spill path for the maximum legal TTL and
                             raises this boot's serve posture, so the disk - not
                             whatever host RAM happened to hold - decides every
                             request's prefix, with no POST at all.
                             Set it to `0`/`false`/`off`/`no` and the boot behaves
                             exactly as every release up to 0.2.32: disarmed,
                             `serving_as=off`, and the two dev-verb POSTs are the
                             only lever.

                             WHY THE DEFAULT IS ON AND WHY THAT IS NOT AN ARM BY
                             ENV VAR. Every other facility in this package starts
                             off and is armed by a POST, because a launcher script
                             must not be able to turn a KV-reading diagnostic on.
                             The disk tier's POSTure is different in kind: it is
                             the feature the operator deployed the tier FOR, and
                             for six evenings it reverted to doing nothing on
                             every restart while `serving_as` reported the truth
                             nobody had asked for. So this is a DISABLE switch on
                             a working default - the same shape as
                             `VLLM_E2_TIER_DEV_BOOT_INERT` above, which also only
                             ever turns a boot effect OFF. Nothing reads this
                             variable to warm anything that a boot would not
                             otherwise do, and it never touches guard mode, the
                             RAM tier's arm or the byte dump.

                             This flag is not a re-arm from persisted state and
                             must never become one: it decides what THIS boot
                             does from the environment, fresh, and reads no arm
                             file from a previous boot (see `disk_boot.py` and the
                             prohibition quoted there). The boot log prints one
                             `KVTIER disk boot default=` line naming the value it
                             read, whether it came from the environment or the
                             default, and how to turn it off.

  THE GPU-DIRECT BYTE WITNESSES (`disk_shadow.py`, `disk_gpu_shadow.py`) ------

  VLLM_E2_TIER_DISK_SHADOW   the READ half of the GPU-direct proof (0.2.36). OFF by
                             default: unset, or a value in neither the on nor the off
                             vocabulary, reads no device memory and reports
                             `enabled: false`. `1`/`true`/`yes`/`on` compares one
                             completed store in 32 - the GPU's own bytes against the
                             blob the arena path captured for the same block - and
                             publishes the verdict (with staleness classified, never
                             counted as corruption) in the worker's rank file.
  VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES
                             the WRITE half (0.2.37), and the same grammar with the
                             same default. ON, a block the read half just called
                             `equal` is written AGAIN from the GPU's bytes into a
                             SEPARATE directory under the live store's own file name,
                             and the two files are compared - that diff is the gate on
                             the GPU -> NVMe route. It never writes the live store and
                             refuses a shadow root that is or contains it.
  VLLM_E2_TIER_DISK_GPU_SHADOW_ROOT
                             where those shadow files go. UNSET = a sibling of the live
                             store root, `<store>/../gpu-shadow`. Read once per process.
                             Both are read ONCE at first use (a live boot's environment
                             does not change) and both have a POST latch that arms only
                             the process answering - the worker is armed by the
                             environment. See README's
                             *Does the GPU still hold what the arena spilled?*.

  THE SPILL HOOK (`disk_store.py`) HAS NO ENV VAR OF ITS OWN, and never will. It
  is installed in every worker whenever the plugin is on, and its per-step cost
  while unarmed is two `os.environ.get` calls; the two switches that make it do
  anything are the root above and the arm, which only
  `POST /plugins/kvtier/disk {"op":"arm"}` grants. `VLLM_E2_TIER_DIAG` is the
  channel for the verb's `spill-one` op (a worker is unreachable over the utility
  channel), the same one the rank files and the byte dump use.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path

ENV_ENABLE = "VLLM_E2_TIER_DEV"
ENV_TOKEN = "VLLM_E2_TIER_DEV_TOKEN"  # noqa: S105 - a variable NAME, not a secret
ENV_TIMEOUT = "VLLM_E2_TIER_DEV_TIMEOUT"
ENV_ALLOW = "VLLM_E2_TIER_DEV_ALLOW"  # empty/unset = loopback only
# THE BOOT OPT-OUT. Unset = the boot arms (guard + tier) once the install
# verifies; truthy = the old inert boot. The name deliberately says what it
# DISABLES rather than what it enables: no env var of this plugin ARMS guard mode,
# which is the property `test_no_env_var_can_arm_guard_mode_and_at_import_it_is_off`
# pins. The boot arms it; this variable only takes the boot back to inert.
ENV_BOOT_INERT = "VLLM_E2_TIER_DEV_BOOT_INERT"

# --- the worker-side diagnostics writer (`worker_diag.py`) ------------------
# The directory one JSON file per rank is written into, on the ENGINE's host. The
# proxy host has a reader for the same layout (`llm-api/kv_tier_diagnostics.py`);
# the operator ships the files over, or points both sides at a shared path.
# UNSET = the writer installs nothing and creates no files anywhere, which is the
# default even when every other part of the plugin is live.
ENV_DIAG_DIR = "VLLM_E2_TIER_DIAG"
# The boot id shared by every writer. The reader compares epochs across writers
# and declares an equality conclusion `meaningless` when they disagree, and a
# worker process CANNOT learn the EngineCore's boot id (the handshake flows one
# way: worker -> EngineCore). Set this once per boot in the launcher's
# environment so all ranks agree; unset, each process stamps its own id and the
# file says `writer.epoch_scope = "process"`.
ENV_DIAG_EPOCH = "VLLM_E2_TIER_DIAG_EPOCH"
# Write cadence: at most once every N steps or on a change of facts, and never
# more than once per MIN seconds. See `worker_diag`'s BUDGET section.
ENV_DIAG_INTERVAL = "VLLM_E2_TIER_DIAG_INTERVAL"
ENV_DIAG_MIN_INTERVAL = "VLLM_E2_TIER_DIAG_MIN_INTERVAL_SECONDS"
# Pin the rank id (`PP<n>_TP<m>`) instead of deriving it from the process's
# distributed state. Optional; a value in the wrong shape is refused loudly
# rather than written, because the reader classifies a file whose `rank` does not
# match its file name as mislabeled.
ENV_DIAG_RANK = "VLLM_E2_TIER_DIAG_RANK"
# THE RAW-BYTE DUMP (`blockdump.py`) HAS NO ENV VAR OF ITS OWN, and never will.
# Its channel is this directory; its arm is `POST /plugins/kvtier/blockdump
# {"op":"arm"}` and it expires by itself. That is deliberate, the same posture as
# guard mode: a launcher script must not be able to turn a facility that reads
# real KV bytes on. With this directory unset the dump verb reports the honest
# three-state marker as `not_deployed` and no worker hook is installed at all.
# THE SSD TIER (`disk_spill.py`) -----------------------------------------------
# Where the one-file-per-slot store lives, and how big it may get. BOTH are
# required for the tier to exist at all, and neither ARMS it: the arm is the
# in-process/arm-file gate in `disk_spill.arm()` (no env var arms a facility that
# writes real KV bytes - the `guard_mode`/`blockdump` rule, pinned by a test).
# Unset root = no directory is created, no thread starts, no byte is written.
# `/tmp`, `/dev/shm` and `/run` are refused by `disk_layout.validate_root`: on the
# box they are tmpfs, so a "disk" tier there spends the RAM the tier relieves.
ENV_DISK_ROOT = "VLLM_E2_TIER_DISK_ROOT"
# The ceiling in BYTES of stored slot files (a 64 GiB store is ~5,900 slots at
# 11.4 MB). Unset or unparseable = no ceiling (never 0: a 0 ceiling would mean
# "evict everything", which is a destructive reading of a typo).
ENV_DISK_BYTES = "VLLM_E2_TIER_DISK_BYTES"
# HOW MANY IDENTITIES THE RESTORE INDEX MAY HOLD IN MEMORY. Unset = the shipped default
# (`disk_index.MAX_INDEX_ENTRIES`). This is the ONE knob that decides how much of the
# store the index can REMEMBER: an identity that falls out of the index is a file that
# is
# still on the disk, whole and readable, which `contains()` will answer `not_present`
# for
# - a missed restore, invisible except as a wasted span and a recompute.
#
# WHY AN OPERATOR NEEDS IT. The cap used to be a module constant, so the only way to
# raise
# it was to patch the plugin. The live store holds 12,738 files across four ranks
# under a
# 148 GiB ceiling and can reach ~30,000, and a converged drain over a big store lives AT
# the cap: the number has to be settable from the environment, next to the ceiling that
# decides how many files there will be. Grammar only here; the floor, the clamp and the
# reason live with the memory they buy (`disk_index.entry_cap`).
ENV_DISK_INDEX_ENTRIES = "VLLM_E2_TIER_DISK_INDEX_ENTRIES"
# WHERE THE SPILL JOURNAL LIVES, separately from everything else in the
# diagnostics directory. UNSET = fall back to VLLM_E2_TIER_DIAG, which is the
# pre-0.2.19 behaviour and changes nothing for an existing deployment.
#
# WHY THE JOURNAL GETS ITS OWN VARIABLE AND ITS OWN FILESYSTEM. The journal IS the
# map from a KV identity to a file on the RAID array, and it is the one piece of
# cross-process state in this tier that cannot be rebuilt: a reboot that destroys
# it while the 107 GB of KV files it describes survives leaves the restore index at
# `entries: 0` - a huge pile of real data with no map to it (observed 2026-10-05).
# So an operator points this at persistent storage (e.g. `/home/user/kv-tier/diag`)
# and keeps `VLLM_E2_TIER_DIAG` wherever the rank files belong.
#
# WHY THE SPLIT IS PER-CATEGORY AND NOT "MOVE THE DIRECTORY". The same diagnostics
# directory holds the spill tier's ARM FILE, and an arm is supposed to die at
# reboot: a cold box booting with the tier armed exposes the restore path to live
# traffic with no operator having asked. Moving the whole directory to disk would
# silently break that invariant, so only the journal moves here - and the arm is
# separately made to fail closed across a boot by its own contents (see
# `disk_spill._read_arm_file`). `GET /plugins/kvtier/disk` reports the journal
# directory's filesystem under `journal.fs`, and the boot line carries
# `journal_dir=tmpfs` when it lands on a filesystem that dies at reboot.
ENV_DISK_JOURNAL_DIR = "VLLM_E2_TIER_DISK_JOURNAL_DIR"

# THE RESTORE SETTLE WAIT (`disk_restore.py`) ---------------------------------
# THE ALLOWANCE on how long one apply point (`start_load_kv`) may wait for its disk
# reads, in MILLISECONDS. **Default 5,000 = give the read up to five seconds**
# (0.2.32). `0` means never wait, which is every boot up to and including 0.2.31 and
# the reason the read path loses its race: reads are queued at `bind_connector_metadata`
# and applied one step later, so an unsettled read was dropped and recomputed.
#
# It is a FLAT allowance and not a computed budget, deliberately: until 0.2.31 the
# grant was proportional to the bytes in flight over an assumed read throughput times a
# safety margin, and the box measured what that guessing was worth - a 23.6 ms grant
# against a 135.8 ms race gap, and a 28,583-token recompute to save the 23.6 ms. See
# `disk_restore.settle_budget` for the deleted arithmetic and the operator's ruling that
# replaced it. The only other bound is the read's own deadline, so this number can never
# outrun the clock that cancels the read.
#
# This is NOT an arm and it arms nothing: the restore path still needs the disk root,
# the spill tier's arm file, the tier's arm, this boot's offer switch and a proven
# worker rank before it reads one byte. A value that cannot be parsed means OFF, never a
# default wait, because the safe failure of a typo in a systemd drop-in is the behaviour
# that does not block a step at all.
ENV_DISK_SETTLE_MS = "VLLM_E2_TIER_DISK_SETTLE_MS"

# THE PARALLEL READER (`disk_read.py`, 0.2.24) --------------------------------
# How many threads issue spill reads at once. ONE is today: a single
# `kvtier-disk-read` thread draining one bounded queue, which the box measures at
# 1.32 GB/s per reader against 5.14 GB/s for four concurrent readers on the same
# spill array (dd iflag=direct, 50 files, 610.7 MiB). The engine already has four
# ranks reading at once; WITHIN a rank the units were serialized, and a 10-span
# offer on the live boot (one eligible group, g5) is 10 of those serialised reads.
#
# It is a THREAD COUNT and nothing else: which job goes to which thread is the
# queue's business, and the queue stays FIFO. FIFO dispatch means a job never STARTS
# while an earlier-submitted job is still unstarted, so it can only ever COMPLETE
# ahead of the jobs already inside the kernel when it started - at most
# (threads - 1) of them. Default 1, and 1 is pinned by a test as byte-identical to
# the single-thread release.
ENV_DISK_READ_THREADS = "VLLM_E2_TIER_DISK_READ_THREADS"
# The BOUNCE-BUFFER POOL, in bytes, per process. This is the constraint the thread
# count runs INTO: in-flight bytes are bounded by the pool no matter how many
# threads there are, and a buffer is taken at ISSUE time and only returned at the
# apply point (`finish_batch`), so one offer needs the pool to hold the WHOLE offer
# at once. Two numbers decide whether it does:
#   * units per offer per rank = spans x eligible groups, and the live boot's
#     `per_block` (disk_restore.py:2260) is 1 - only g5 is eligible, so a 10-span
#     offer is 10 units, not 50 (and ~11 spans is the DEEPEST contiguous run the live
#     store held on the boot the operator measured - spill follows RAM-tier parking and
#     the CPU arena measured 187 blocks per rank, so run depth, not the offer ceiling,
#     was what bound it);
#   * bytes per unit = one rank's CPU slot: 11,468,800 B on the live narrow width,
#     12,902,400 B on the wide one.
# At the default 96 MiB the pool holds 8 of the narrow unit and 7 of the wide one, so
# the 10-unit big-prompt offer misses by TWO units and loses them to `buffer_cap` at
# issue - which is why raising threads WITHOUT raising this buys nothing on a big
# prompt. 128 MiB (11 narrow / 10 wide) is the smallest setting that holds a whole
# 10-unit offer on both widths.
#
# HOST-RAM COST, stated because it is the operator's to size deliberately: the pool
# is per-process and every engine rank is a process, so the boot pays
# `increase x rank_count` in resident bytes (4 ranks: +512 MiB of pool = +2 GiB).
# Unset = `disk_read.MAX_BUFFERS_BYTES` exactly, so the default boot allocates the
# same bytes it allocated in 0.2.23.
ENV_DISK_READ_BUFFER_BYTES = "VLLM_E2_TIER_DISK_READ_BUFFER_BYTES"

# THE CEILING PASS'S TIME BUDGET (`disk_spill.py`, 0.2.29) --------------------
# SECONDS one capacity-enforcement pass may spend. The pass runs on the SPILL WRITER
# THREAD after every committed write, and until now it had no deadline at all: up to
# `MAX_WALK_ENTRIES` (100,000) directory entries read TWICE (count and measure),
# `MAX_WALK_FILES` (20,000) `stat` calls, and up to `MAX_EVICTION_PLAN` (4,096)
# unlinks. That is potentially seconds, and the writer thread is the only thing that
# drains the spill queue - so while it walks, the queue (256 jobs / 256 MiB) backs up
# and new spills are DROPPED at exactly the moment the disk is fullest.
# A budget does not make the work disappear: the pass stops early, reports itself
# partial (`plan_truncated`), and the next committed spill runs the next pass, which
# is the same partial-plan convergence rule the walk has always used on a store
# bigger than its caps. Set it from the experiment's own
# `spill.ceiling_measurement.pass_seconds`.
ENV_DISK_CEILING_BUDGET_SECONDS = "VLLM_E2_TIER_DISK_CEILING_BUDGET_SECONDS"
# HOW MANY THREADS UNLINK AT ONCE inside one ceiling pass (`disk_spill.py`, 0.2.29).
# The SAME knob shape as `VLLM_E2_TIER_DISK_READ_THREADS`: a per-process thread count,
# resolved once per process (the pool is built once and is never rebuilt mid-boot),
# clamped, and reported when clamped. `1` is the serial behaviour of every release
# before 0.2.29 and is the shipped default: parallel unlink is METADATA work, and the
# measurement in this release's notes decides whether more than one is worth anything
# on this array. It changes WHICH thread does an unlink, never WHICH files are chosen
# or HOW MANY: the plan is partitioned across the workers, so no path is ever attempted
# twice and the pass's accounting does not depend on this number.
ENV_DISK_EVICT_THREADS = "VLLM_E2_TIER_DISK_EVICT_THREADS"

# THE SHADOW VERIFIER (`disk_shadow.py`, 0.2.36). The read-only witness that compares
# a completed store's GPU bytes against the bytes the spill path captured for the same
# block, and publishes the disagreement - with staleness classified, never counted as
# corruption. DEFAULT OFF, and it stays off unless the operator names it: this reads
# REAL KV BYTES off the device on the engine's step thread (bounded to
# `MAX_BLOCKS_PER_STEP` blocks per bind window), and "a launcher script must not be
# able to turn a facility that reads real KV bytes on" is this package's rule for
# `guard_mode` and `blockdump` - so the default here is off in the same way theirs is.
# The live lever is `POST /plugins/kvtier/disk {"op":"shadow"}`, which arms the process
# that answers (it writes no control file: this module opens no file handle for
# writing). It never changes a spill or a restore decision.
ENV_DISK_SHADOW = "VLLM_E2_TIER_DISK_SHADOW"

# THE GPU-DIRECT SHADOW WRITER (`disk_gpu_shadow.py`, 0.2.37). Step 2 of the
# GPU -> NVMe route: for a block `disk_shadow` has already called `equal`, write the
# GPU's bytes AGAIN, to a directory that is not the live store, named exactly the way
# the store names that block, so the two trees can be diffed on real traffic. It is
# the first code in this package that writes KV bytes from a GPU-direct read, so the
# default is OFF for the same reason the verifier's is: a launcher script must not be
# able to turn on a facility that writes files from a device read. The one difference
# from `ENV_DISK_SHADOW` is what it can touch: this module never opens the live store
# for writing and REFUSES a shadow root that is or contains it
# (`refused_root_conflict`), so nothing here can ever become authoritative.
ENV_DISK_GPU_SHADOW_WRITES = "VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES"
# WHERE the shadow files go. UNSET = `<the live store root>/../gpu-shadow`, a SIBLING of
# the store (the two trees stay diffable and the store's own sweep never sees them).
# Set it to a path on the same device you want to measure, and note that the same
# refusal applies to it: same-as or nested-in the live store is refused, not merged.
# Read ONCE per process, like the enablement. Not a `VLLM_E2_TIER_*` behaviour switch:
# it names a location, and it is validated by `disk_layout.validate_root`.
ENV_DISK_GPU_SHADOW_ROOT = "VLLM_E2_TIER_DISK_GPU_SHADOW_ROOT"

# THE RESTORE-SIDE VERIFIER (`disk_restore_verify.py`, 0.2.39). The read-only witness
# for the OTHER direction: for a sampled restored entry, do the bytes now in the GPU
# window match the bytes in the spill file, over the GROUP'S OWN leading width?
# `ENV_DISK_SHADOW` proves that what we SHIPPED to NVMe is what the GPU held; this
# proves that what came BACK from NVMe is what the GPU now holds - which is where a
# one-slot index shift would actually be visible, because a shift moves the
# index-aligned lists together and so every file name stays well-formed while the
# engine serves the wrong token boundary. DEFAULT OFF for the same reason both other
# device-reading flags are: it reads REAL KV BYTES off the device and opens spill files
# for reading on a live box, so a launcher script must not be able to turn it on. The
# work runs on the module's own bounded thread - the step thread pays a counter and a
# small dict of ints at the apply point - and it never changes a restore decision. The
# live lever is `POST /plugins/kvtier/disk {"op":"restore-verify"}`, which latches the
# ANSWERING process and writes no control file; the process that applies a restore is a
# worker, and a worker is armed here.
ENV_DISK_RESTORE_VERIFY = "VLLM_E2_TIER_DISK_RESTORE_VERIFY"

# THE GPU-DIRECT WRITE (`disk_gpu_write.py`, 0.2.43) ----------------------------
# The one flag in this package that WRITES INTO LIVE KV MEMORY, so the sentence about
# what it does has to be exact: for one applied restore entry in
# `SAMPLE_EVERY_N_APPLIED`, the plugin issues its OWN host->device copy of that entry's
# pinned CPU slot into the GPU
# window the fork is about to load, on the tier's own `load_stream`, and then reads the
# window back TWICE and compares the leading `W_g` bytes with the spill file. It is a
# SECOND WRITER of the destination, never a replacement: the fork's own arena->GPU copy
# is submitted after ours on the same stream and therefore has the last word on every
# byte, which is what makes a wrong addressing guess here a wrong WITNESS rather than a
# wrong answer. The ordering is structural, not hoped for: the hook runs inside
# `start_load_kv`'s wrapper before `original(self, ...)`, and `original` is what submits
# the fork's copy.
#
# WHY OFF IS THE ONLY HONEST DEFAULT: this is the only dev flag here whose failure mode
# is a device write. Everything it does is contained and fail-open - a refusal is a name
# and the fork's copy runs alone, exactly as it does when this is unset - but "the
# diagnostic is contained" is not the same as "it writes nothing by default". The live
# lever is `POST /plugins/kvtier/disk {"op":"gpu-write"}`, which latches the ANSWERING
# process and writes no control file; only a WORKER applies a restore, so only a worker
# can actually write, and a worker is armed here.
ENV_DISK_GPU_DIRECT_WRITE = "VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE"

# STEP 3 OF RETIRING THE ARENA: THE SAME COPY AS THE DELIVERY, NOT AS A WITNESS
# (`disk_gpu_write.py`, 0.2.52) ------------------------------------------------
# `ENV_DISK_GPU_DIRECT_WRITE` makes the plugin's own host->device copy a SAMPLED
# WITNESS: one applied entry in `SAMPLE_EVERY_N_APPLIED` (64), at most
# `MAX_WRITES_PER_STEP` (1) per step, with a `MIN_INTERVAL_SECONDS` (0.5 s) floor
# between writes. Three limits, and all three must be lifted together: a writer that
# samples but still writes one entry per step is a MORE confusing failure than one that
# writes nothing, because the counters then say "delivery" while the traffic says
# "witness".
#
# WITH THIS ON (and the writer itself on), the writer is DELIVERING: every applied entry
# of the step gets its own host->device copy, so the set of entries the writer ENQUEUED
# is a superset of the set `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS` is allowed to prune,
# and `prune_unwitnessed_entries` is 0 BY CONSTRUCTION rather than by hope. The gate for
# pruning is "the copy was ENQUEUED" - a synchronous, deterministic fact known at the
# apply point - and the asynchronous readback stays what it always was: the KILL-SWITCH.
# A same-execution readback mismatch disables the prune from the next step onward and
# puts the fork's copy back, which is what converts "one silently wrong request" into
# "the old path comes back".
#
# WHY OFF IS THE ONLY HONEST DEFAULT, in one line: with this ON and the prune ON, the
# plugin's copy is the only thing standing between a restored entry and an uninitialised
# GPU window, so a wrong `gpu_block_id` is a WRONG ANSWER and not a wrong witness. The
# ordering guard (`disk_gpu_write.py`'s `unsubmitted` ledger check) is load-bearing for
# exactly that reason, and it stays a hard refuse: no ledger verdict, no write, and no
# prune either.
ENV_DISK_GPU_WRITE_DELIVERY = "VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY"

# HOW LONG ONE DELIVERY STEP MAY HOLD THE STEP THREAD (`disk_gpu_write.py`,
# 0.2.57) -----------------------------------------------------------------
# THE ONE BOUND ON THE DELIVERY LOOP, replacing 0.2.56's two per-entry budgets (a
# 0.25 s step budget and a 64 MiB byte budget) after the operator ruled on it: this
# path has been strangled by fine-grained caps before - the disk read/write path - and
# the standing answer here is ONE generous timeout, not a budget. A 0.25 s cap also
# deferred ~92% of a normal large restore, which turned the prune it gates into a
# no-op: a guard that disables the feature is not a safety feature.
#
# It is an escape hatch against a copy that never returns, not a throughput lever, and
# the measured shape of the work says so: ~735 MB of host->device pageable PCIe at
# ~5 GB/s is ~0.15 s, so a 10 s bound is ~60x above normal and ~30x inside the only
# deadline the engine actually enforces (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, 300 s).
# See `disk_gpu_write.DELIVERY_APPLY_TIMEOUT_SECONDS` for the arithmetic and the
# numbers this was measured against.
#
# Grammar only, like `ceiling_env`: a value that is not a finite number ABOVE 0 is
# reported in `problems` and left as None, which the consumer reads as "the operator
# said nothing" - so a typo in a systemd drop-in gives the shipped 10 s and never a
# step with no bound at all. The clamp is policy and lives with the consumer.
ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT = (
    "VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS")

# HOW MANY CPU SLOTS AN OFFER MUST LEAVE FREE IN THE ARENA (`disk_restore.py`,
# 0.2.38) ---------------------------------------------------------------
# `disk_restore.CPU_SLOT_RESERVE` (64) was a constant, and it is a number that was
# sized for an arena 8x the size of the one the operator is now running: shrinking the
# engine's KV arena from 64 GiB to 8 GiB left 166 free slots per rank on the PP1 pair,
# so the reservation ate 41% of the arena BEFORE the offer path saw a single slot. It
# is now a floor the operator can lower, which is the cheapest remaining host-RAM win
# because it converts reserved slots into restorable tokens with no engine change.
#
# WHAT THE RESERVE IS FOR, and so what it is safe to set it to: an offer allocates its
# destination slots OUT OF THIS POOL, and the pool is the arena the RAM tier needs to
# park its own blocks. The reserve is the headroom an offer deliberately leaves for
# that traffic - a reserve of 0 means "a disk offer may take the arena to its last
# slot", which is legal and honoured (see below) but leaves the RAM tier nothing.
#
# 0 IS HONOURED HERE, which is the opposite of `ceiling_env`'s and `disk_read_env`'s
# floors, and the reason is that 0 is a real posture rather than a dead knob: it means
# "spend the whole arena on restore". Only a NON-INTEGER or a NEGATIVE value is
# refused, and both fall back to the shipped default of 64 and are announced once in
# the engine log, because a value that quietly stopped being what the operator typed is
# the failure shape this package refuses everywhere else.
#
# THERE IS NO CEILING, deliberately. An absurd value (say 10,000) is honoured, and the
# arithmetic then declines every offer with `no_cpu_slots` - the safe direction, since
# the reserve's whole job is to make an offer SMALLER. Clamping it here would turn "the
# operator asked for a reserve larger than the arena" into "the operator asked for
# 64", which is a silently different machine.
ENV_DISK_CPU_SLOT_RESERVE = "VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE"

# STEP 2 OF RETIRING THE ARENA: STOP THE FORK COPYING WHAT THE RESTORE DELIVERS
# (`disk_restore.py`, 0.2.49) -------------------------------------------------
# TODAY A DISK RESTORE IS CARRIED BY THE ARENA TWICE OVER. The read lands in a CPU
# slot and the fork's own `start_load_kv` then DMAs that slot into the GPU block named
# beside it. The pair is positional: `copy_blocks` pairs `load_cpu_blocks[i]` with
# `load_gpu_blocks[i]` by INDEX and has no way to notice one list is shorter than the
# other (`cuda_mem_ops.py:244`), which is why the entry is dropped from BOTH lists at
# the SAME index or not at all - a one-sided prune is a silent one-block shift of every
# later destination, a wrong-output bug and not a crash.
#
# WITH THIS FLAG ON, the entries of an ADOPTED and SUCCESSFULLY READ disk restore are
# removed from both load lists at the apply point, so the engine is no longer asked to
# copy blocks whose destination it does not need. That is the first step that takes
# work off the arena path; step 3 stops filling the arena slot at all and step 4 shrinks
# `KV_OFFLOAD_SIZE` to a vestige.
#
# READ THIS BEFORE TURNING IT ON. The fork's copy is currently the ONLY thing that puts
# a restored block's bytes into its GPU window - `disk_gpu_write`'s own host->device
# copy
# is a SAMPLED WITNESS (one entry in 64, at most one per step) and is not the delivery.
# Pruning an entry therefore removes the copy that carries its bytes, so a boot with
# this flag on and no authoritative plugin-side write serves that request from
# uninitialised KV: a WRONG ANSWER, not an outage. The counters say exactly this -
# `prune_unwitnessed_entries` counts pruned entries the plugin did NOT write into the
# GPU itself - and the README says it plainly. The default is OFF, the grammar is the
# same DEFAULT-OFF shape as `disk_shadow_enable()` (a typo moves the direction that
# prunes nothing), and with the flag off the apply point produces byte-identical lists
# to every release before this one - pinned by a test, because "we did not change the
# default path" is the claim this release must not get wrong.
ENV_DISK_PRUNE_RESTORED_LOADS = "VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS"

# The miss-attribution ledger's OPTIONAL file mirror: a path to append one JSON
# line per tier lookup to. Unset = no file, no open(), no I/O of any kind. It is
# env-only on purpose: an HTTP argument naming a path the engine appends to would
# be a write surface this plugin does not have anywhere else. See
# miss_attribution.py (bounded by MAX_MIRROR_BYTES, which it stops at, counted).
# THE DISK TIER'S BOOT DEFAULT (`disk_boot.py`, 0.2.33). The name deliberately does
# NOT follow the `VLLM_E2_TIER_*` family: the operator asked for one flag he can find
# by name when a boot misbehaves, and this is it. UNSET = the boot brings the SSD tier
# up ARMED AND SERVING; `0`/`false`/`off`/`no` = the pre-0.2.33 posture (disarmed, the
# dev-verb POSTs are the only lever). Like `ENV_BOOT_INERT`, this variable only ever
# DISABLES a boot effect - it arms nothing that the boot default is not already on -
# and it has no influence on guard mode, the RAM tier's arm or the raw-byte dump.
ENV_DISK_BOOT_ENABLE = "KVTIER_DISK_BOOT_ENABLE"

# The words that turn the disk tier's boot default OFF. Everything else - including a
# value that is not a word at all - reads as ENABLED, because the default is on and a
# typo must not silently produce the off posture (a boot the operator believes is
# serving, serving nothing, is the failure this whole flag exists to end). A value
# outside both vocabularies is still reported verbatim in `raw_value` and named on the
# boot line, so `KVTIER_DISK_BOOT_ENABLE=flase` cannot pass unnoticed.
_DISK_BOOT_OFF_WORDS = frozenset({"0", "false", "off", "no"})
_DISK_BOOT_ON_WORDS = frozenset({"1", "true", "yes", "on"})

ENV_MISS_LOG = "VLLM_E2_TIER_MISS_LOG"

logger = logging.getLogger("kvtier_dev_plugin.config")

# Engine-side method names. `EngineCore` dispatches a UTILITY message with
# `getattr(self, method_name)` (vllm/v1/engine/core.py:1669-1671), so these
# must be unique on the EngineCore class. The `kvtier_dev_` prefix is the only
# name collision protection we have; `grep -rn kvtier vllm/` finds nothing.
METHOD_SNAPSHOT = "kvtier_dev_snapshot"
METHOD_RELOAD_POLICY = "kvtier_dev_reload_policy"
METHOD_KVCONFIG = "kvtier_dev_kvconfig"
METHOD_INSPECT = "kvtier_dev_inspect"
METHOD_TRACE = "kvtier_dev_trace"
METHOD_GUARD = "kvtier_dev_guard"
METHOD_TIER = "kvtier_dev_tier"
METHOD_LOOKUPS = "kvtier_dev_lookups"
# The raw-byte dump verb. Read-only, bounded, and DISARMED until a human arms it
# (there is no env var that arms it): see blockdump.py.
METHOD_BLOCKDUMP = "kvtier_dev_blockdump"
# The SSD spill tier's verb: status / arm / disarm / spill-one / ls. It can ARM a
# facility that writes real KV bytes to disk, so it is a dev-state route; the arm
# expires and no env var produces one. See disk_store.py.
METHOD_DISK = "kvtier_dev_disk"

# Every engine method this plugin owns, in attach order. `attach_methods` walks
# this, so a new verb cannot be attached in one place and forgotten elsewhere.
METHODS = (METHOD_SNAPSHOT, METHOD_RELOAD_POLICY, METHOD_KVCONFIG, METHOD_INSPECT,
           METHOD_TRACE, METHOD_GUARD, METHOD_TIER, METHOD_LOOKUPS,
           METHOD_BLOCKDUMP, METHOD_DISK)

# All HTTP surface lives here, never under /v1 (see docs/usage/security.md:349:
# `--api-key` only protects the /v1, /v2, /inference, /cohere prefixes).
ROUTE_PREFIX = "/plugins/kvtier"

# Route names, for status and the tests. The verbs that CAN change dev state
# (`/guard`, `/trace`, `/tier`, `/lookups`) require the token in every shape,
# loopback included.
ROUTE_STATUS = "/status"
ROUTE_SNAPSHOT = "/snapshot"
ROUTE_RELOAD = "/policy/reload"
ROUTE_KVCONFIG = "/kvconfig"
ROUTE_BLOCK = "/block"
ROUTE_REQUEST = "/request"
ROUTE_TRACE = "/trace"
ROUTE_GUARD = "/guard"
ROUTE_TIER = "/tier"
ROUTE_LOOKUPS = "/lookups"
# The bounded raw-KV-byte dump. It can ARM a diagnostic that reads real KV bytes,
# so it is a dev-state route: token required in every shape, loopback included.
ROUTE_BLOCKDUMP = "/blockdump"
# The SSD spill tier's control verb. It can ARM a facility that writes real KV
# bytes to disk, and `spill-one` makes a worker write one, so it is a dev-state
# route on the same rule as `/blockdump`: token in every shape, loopback included.
ROUTE_DISK = "/disk"

# The same routes as one tuple, in the order a reader wants them, so `status`
# and the tests enumerate the surface from ONE place. Suffixes: the router
# carries ROUTE_PREFIX itself.
ROUTES = (ROUTE_STATUS, ROUTE_SNAPSHOT, ROUTE_KVCONFIG, ROUTE_BLOCK, ROUTE_REQUEST,
          ROUTE_TRACE, ROUTE_GUARD, ROUTE_TIER, ROUTE_LOOKUPS, ROUTE_BLOCKDUMP,
          ROUTE_DISK, ROUTE_RELOAD)

# The routes that can change engine dev state. Token in every shape, loopback
# included - see endpoint.require_dev_state_token.
DEV_STATE_ROUTES = (ROUTE_TRACE, ROUTE_GUARD, ROUTE_TIER, ROUTE_LOOKUPS,
                    ROUTE_BLOCKDUMP, ROUTE_DISK)

# Name used for logs, for `VLLM_PLUGINS` allowlisting and as the entry-point
# name in both groups. One name allowlists both halves at once.
PLUGIN_NAME = "kvtier_dev"

DEFAULT_TIMEOUT_SECONDS = 5.0
MIN_TIMEOUT_SECONDS = 0.05
MAX_TIMEOUT_SECONDS = 60.0

DEFAULT_DIAG_INTERVAL_STEPS = 50
MIN_DIAG_INTERVAL_STEPS = 1
MAX_DIAG_INTERVAL_STEPS = 100_000
DEFAULT_DIAG_MIN_INTERVAL_SECONDS = 1.0
MIN_DIAG_MIN_INTERVAL_SECONDS = 0.0
MAX_DIAG_MIN_INTERVAL_SECONDS = 3600.0
MAX_DIAG_EPOCH_CHARS = 32
MAX_DIAG_RANK_CHARS = 32

_PKG_DIR = Path(__file__).resolve().parent
_FALSY = frozenset({"", "0", "false", "off", "no"})

# The single-slot allowlist cache. `_ALLOW_CACHE_RAW` starts as None, which no
# env value can equal, so the first call always parses.
_ALLOW_CACHE_RAW: str | None = None
_ALLOW_CACHE_VALUE: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()


class PolicyPathError(ValueError):
    """A requested policy path was refused. Carries the human-readable why."""


def enabled() -> bool:
    """Master switch. False unless VLLM_E2_TIER_DEV says otherwise."""
    return os.environ.get(ENV_ENABLE, "").strip().lower() not in _FALSY


def boot_inert() -> bool:
    """True = the INERT boot (patch only: no guard, no tier arming).

    The same falsy set as `enabled()`, so `0`/`false`/`off`/`no`/empty all mean
    "not inert" = the auto boot. Read per call, never at import: the mode an
    operator wants changes with a restart of the engine, not a re-install, and
    `status` reports the mode it read at the last boot.
    """
    return os.environ.get(ENV_BOOT_INERT, "").strip().lower() not in _FALSY


def disk_boot_enable() -> dict:
    """The SSD tier's boot posture, resolved ONCE per boot by `disk_boot.py`.

    Returns `{enabled, source, raw, recognised, env_var}` and never raises:

      * UNSET or empty -> `{"enabled": True, "source": "default", "raw": None}`.
        The default is ON: the boot arms the spill path and raises this boot's serve
        posture, so a restart no longer silently reverts to a tier that serves nothing.
      * `0`/`false`/`off`/`no` -> `{"enabled": False, "source": "env"}` - exactly the
        behaviour of every release through 0.2.32, with the two dev-verb POSTs as the
        only lever.
      * anything else -> enabled, and `recognised: False` when the word is not one the
        grammar defines. It is reported rather than guessed at, because an operator who
        meant to disable a boot and typed `flase` needs the boot line to tell him the
        value he wrote is not a disable word - not a payload that reads as intent.

    THE ONE THING THIS FUNCTION MUST NEVER DO: read a persisted arm or posture from an
    earlier boot and honour it. It reads `os.environ` and nothing else, and its result
    is a decision about THIS boot computed at connector creation. `tests/
    test_disk_boot_default.py` pins that, and `llm-api/AGENTS.md`'s 2026-10-03 entry
    ("do not add a boot-time re-arm") is the reason it is a rule rather than a taste.

    Note what is NOT here: no name, key or wording of the disk-serve switch itself.
    That switch's own name belongs to `disk_restore`, and keeping this module free of it
    is what pins that no configuration vocabulary can raise it directly - the suite
    named for that switch asserts exactly this about this file.
    """
    raw = os.environ.get(ENV_DISK_BOOT_ENABLE, "").strip()
    if not raw:
        return {"enabled": True, "source": "default", "raw": None,
                "recognised": True, "env_var": ENV_DISK_BOOT_ENABLE}
    lowered = raw.lower()
    if lowered in _DISK_BOOT_OFF_WORDS:
        return {"enabled": False, "source": "env", "raw": raw,
                "recognised": True, "env_var": ENV_DISK_BOOT_ENABLE}
    return {"enabled": True, "source": "env", "raw": raw,
            "recognised": lowered in _DISK_BOOT_ON_WORDS,
            "env_var": ENV_DISK_BOOT_ENABLE}


def disk_shadow_enable() -> dict:
    """The shadow verifier's boot posture, read by `disk_shadow.py`. The MIRROR of
    `disk_boot_enable()` in every respect except the default, and the default is the
    whole point:

      * UNSET or empty -> `{"enabled": False, "source": "default"}`. OFF. The spill
        tier defaults ON because serving is its job; the shadow defaults OFF because it
        reads REAL KV BYTES off the device on the engine's step thread, and a facility
        that does that is opted into, never defaulted into (`guard_mode`, `blockdump`).
      * `1`/`true`/`yes`/`on` -> enabled.
      * a disable word -> disabled, the same answer as the default, said with
        `source: "env"` so a rank file can tell an operator's no from nobody's answer.
      * anything else -> disabled AND `recognised: False`. The safe direction is the one
        that does not start reading device memory on a typo, and the unrecognised value
        is reported verbatim in `raw` rather than guessed at.

    It reads `os.environ` and nothing else: no persisted latch, no arm file, no
    decision from an earlier boot. The live lever is the POST op, and the two are
    reported separately by `disk_shadow.enable_state()`.
    """
    raw = os.environ.get(ENV_DISK_SHADOW, "").strip()
    if not raw:
        return {"enabled": False, "source": "default", "raw": None,
                "recognised": True, "env_var": ENV_DISK_SHADOW}
    lowered = raw.lower()
    if lowered in _DISK_BOOT_ON_WORDS:
        return {"enabled": True, "source": "env", "raw": raw,
                "recognised": True, "env_var": ENV_DISK_SHADOW}
    return {"enabled": False, "source": "env", "raw": raw,
            "recognised": lowered in _DISK_BOOT_OFF_WORDS,
            "env_var": ENV_DISK_SHADOW}


def disk_restore_verify_enable() -> dict:
    """The restore-side verifier's boot posture, read by `disk_restore_verify.py`. The
    SAME grammar as `disk_shadow_enable()`, because it deserves the same sentence: unset
    is OFF, an on word is ON, a disable word is OFF said with `source: "env"`, and
    anything else is OFF *and* reported as unrecognised. There is no fourth case.

    WHY OFF IS THE ONLY SAFE DEFAULT, in one line: this flag turns on "read a KV block
    off the device and open a spill file for reading while the box serves traffic". The
    one difference from `disk_shadow` is what it reads - a block that a RESTORE just
    wrote, at the apply point, instead of a block a STORE just read - and the answer it
    is looking for: the group's own leading width, never the window's tail.
    """
    raw = os.environ.get(ENV_DISK_RESTORE_VERIFY, "").strip()
    if not raw:
        return {"enabled": False, "source": "default", "raw": None,
                "recognised": True, "env_var": ENV_DISK_RESTORE_VERIFY}
    lowered = raw.lower()
    if lowered in _DISK_BOOT_ON_WORDS:
        return {"enabled": True, "source": "env", "raw": raw,
                "recognised": True, "env_var": ENV_DISK_RESTORE_VERIFY}
    return {"enabled": False, "source": "env", "raw": raw,
            "recognised": lowered in _DISK_BOOT_OFF_WORDS,
            "env_var": ENV_DISK_RESTORE_VERIFY}


def disk_gpu_direct_write_enable() -> dict:
    """The GPU-direct write's boot posture, read by `disk_gpu_write.py`. The SAME
    grammar as `disk_shadow_enable()`: unset is OFF, an on word is ON, a disable word is
    OFF said with `source: "env"`, and anything else is OFF *and* reported as
    unrecognised. There is no fourth case, and for this flag that is not a style rule:
    it is the switch behind "issue a host->device copy into a live KV window", so the
    direction a typo can move in must be the direction that writes nothing.

    THE ONE THING THAT MAKES THIS FLAG DIFFERENT from the two device-READING flags: what
    it turns on writes as well as reads. What makes it SAFE to offer at all is that the
    fork's own copy of the same bytes is submitted after ours on the same stream, so it
    keeps the last word on every byte and a wrong guess by this plugin is a wrong
    witness rather than a wrong answer (see `disk_gpu_write.py`'s module docstring for
    the ordering argument and for the two counters that keep it honest).
    """
    raw = os.environ.get(ENV_DISK_GPU_DIRECT_WRITE, "").strip()
    if not raw:
        return {"enabled": False, "source": "default", "raw": None,
                "recognised": True, "env_var": ENV_DISK_GPU_DIRECT_WRITE}
    lowered = raw.lower()
    if lowered in _DISK_BOOT_ON_WORDS:
        return {"enabled": True, "source": "env", "raw": raw,
                "recognised": True, "env_var": ENV_DISK_GPU_DIRECT_WRITE}
    return {"enabled": False, "source": "env", "raw": raw,
            "recognised": lowered in _DISK_BOOT_OFF_WORDS,
            "env_var": ENV_DISK_GPU_DIRECT_WRITE}


def disk_prune_restored_loads() -> dict:
    """Step 2 of retiring the arena, read by `disk_restore.py`. The SAME grammar as
    `disk_shadow_enable()`: unset is OFF, an on word is ON, a disable word is OFF said
    with `source: "env"`, and anything else is OFF *and* reported as unrecognised.

    WHY OFF IS THE ONLY HONEST DEFAULT, in one line: with the flag ON the fork's own
    arena->GPU copy is removed for the entries this plugin prunes, and until a
    plugin-side write is the delivery (step 3), those bytes reach no GPU window. Every
    refusal this flag can make is counted and logged once per episode, and a refused
    prune is only a missed optimization - which is the correct trade, because the
    alternative failure is a silently shifted destination list.
    """
    raw = os.environ.get(ENV_DISK_PRUNE_RESTORED_LOADS, "").strip()
    if not raw:
        return {"enabled": False, "source": "default", "raw": None,
                "recognised": True, "env_var": ENV_DISK_PRUNE_RESTORED_LOADS}
    lowered = raw.lower()
    if lowered in _DISK_BOOT_ON_WORDS:
        return {"enabled": True, "source": "env", "raw": raw,
                "recognised": True, "env_var": ENV_DISK_PRUNE_RESTORED_LOADS}
    return {"enabled": False, "source": "env", "raw": raw,
            "recognised": lowered in _DISK_BOOT_OFF_WORDS,
            "env_var": ENV_DISK_PRUNE_RESTORED_LOADS}


def disk_gpu_write_delivery() -> dict:
    """DELIVERY MODE for the GPU-direct write, read by `disk_gpu_write.py`. The SAME
    grammar as `disk_gpu_direct_write_enable()`: unset is OFF, an on word is ON, a
    disable word is OFF said with `source: "env"`, and anything else is OFF *and*
    reported as unrecognised.

    WHAT IT CHANGES, precisely, and only when the writer itself is also on: the three
    throttles that make that copy a witness stop applying to the entries a prune could
    cover - the sampler becomes 1, the per-step cap becomes the step's own survivor
    count, and the 0.5 s interval floor is dropped. Everything else about the write is
    the same code, the same stream and the same ordering guard, and the guard stays a
    hard refuse: a load event the worker already recorded means this hook is no longer
    ahead of the fork's copy, so nothing is written and, because the prune's gate is
    membership of the ENQUEUED set, nothing is pruned either.

    WHY OFF IS THE ONLY HONEST DEFAULT: this flag is what makes
    `VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS` safe to turn on, and on its own it makes
    the plugin the writer of every restored window in the step. With the prune still off
    that is a redundant copy the fork overwrites; with the prune on it is the delivery.
    The direction a typo moves in is therefore the direction that copies LESS, which is
    the correct direction for a flag whose failure mode is a wrong answer.
    """
    raw = os.environ.get(ENV_DISK_GPU_WRITE_DELIVERY, "").strip()
    if not raw:
        return {"enabled": False, "source": "default", "raw": None,
                "recognised": True, "env_var": ENV_DISK_GPU_WRITE_DELIVERY}
    lowered = raw.lower()
    if lowered in _DISK_BOOT_ON_WORDS:
        return {"enabled": True, "source": "env", "raw": raw,
                "recognised": True, "env_var": ENV_DISK_GPU_WRITE_DELIVERY}
    return {"enabled": False, "source": "env", "raw": raw,
            "recognised": lowered in _DISK_BOOT_OFF_WORDS,
            "env_var": ENV_DISK_GPU_WRITE_DELIVERY}


def disk_gpu_shadow_writes() -> dict:
    """The GPU-direct shadow writer's boot posture, read by `disk_gpu_shadow.py`. The
    SAME grammar as `disk_shadow_enable()` - unset is OFF, an on word is ON, a disable
    word is OFF said with `source: "env"`, and anything else is OFF *and* reported as
    unrecognised. There is no fourth case and there must never be one: this is the flag
    that turns on "write a KV file from bytes read off the device", and the direction a
    typo can move in is the one that writes nothing.
    """
    raw = os.environ.get(ENV_DISK_GPU_SHADOW_WRITES, "").strip()
    if not raw:
        return {"enabled": False, "source": "default", "raw": None,
                "recognised": True, "env_var": ENV_DISK_GPU_SHADOW_WRITES}
    lowered = raw.lower()
    if lowered in _DISK_BOOT_ON_WORDS:
        return {"enabled": True, "source": "env", "raw": raw,
                "recognised": True, "env_var": ENV_DISK_GPU_SHADOW_WRITES}
    return {"enabled": False, "source": "env", "raw": raw,
            "recognised": lowered in _DISK_BOOT_OFF_WORDS,
            "env_var": ENV_DISK_GPU_SHADOW_WRITES}


def disk_gpu_shadow_root() -> dict:
    """Where the shadow files go, as the operator named it. `raw` is None when he did
    not name it, which is the interesting case: `disk_gpu_shadow` then derives a sibling
    of the live store root. No validation here on purpose - the shape rules and the
    refusal of the live store itself belong to `disk_gpu_shadow._resolve_root`, next to
    the code that writes, and a second validator here is how two sites disagree about
    one path. Empty-after-strip is `None`, not "the empty path": `X=` means "I said
    nothing", the same reading every other variable here gives."""
    raw = os.environ.get(ENV_DISK_GPU_SHADOW_ROOT, "").strip()
    return {"raw": raw or None, "env_var": ENV_DISK_GPU_SHADOW_ROOT}


def token_configured() -> bool:
    return bool(os.environ.get(ENV_TOKEN, "").strip())


def mutating_token() -> str:
    """The token a mutating request must present. "" means "reload refused"."""
    return os.environ.get(ENV_TOKEN, "").strip()


def allow_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Extra peer networks admitted by VLLM_E2_TIER_DEV_ALLOW.

    An EMPTY tuple means "loopback only" - today's behaviour, and the default
    when the variable is unset, empty, or entirely garbage. Nothing here ever
    widens access on a parse failure: an entry that does not parse contributes
    nothing to the result, so a typo in the env opens a door for nobody.

    Both a bare address (`192.168.2.64`, stored as a /32) and a network
    (`192.168.2.0/24`) are accepted, hence `strict=False`. Loopback is NOT
    added here; the guard always accepts loopback and uses this only to widen.

    Cost: this runs per request in the HTTP process, so the parse is memoised
    in a single slot keyed on the raw env string - no lock (a torn read of two
    module globals is impossible in CPython, and the assignment order below
    means a reader that matches on the key always sees the matching value), and
    an env change invalidates it by definition because the key IS the env.
    """
    global _ALLOW_CACHE_RAW, _ALLOW_CACHE_VALUE
    raw = os.environ.get(ENV_ALLOW, "")
    if raw == _ALLOW_CACHE_RAW:
        return _ALLOW_CACHE_VALUE
    parsed: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            parsed.append(ipaddress.ip_network(entry, strict=False))
        except ValueError as exc:
            # Never fatal, never an opening: the entry is simply absent.
            logger.warning("[KVTIER_DEV] ignoring malformed %s entry %r (%s)",
                           ENV_ALLOW, entry, exc)
    # Value before key: a concurrent reader can only ever pair a raw string
    # with the value parsed from that same raw string.
    _ALLOW_CACHE_VALUE = tuple(parsed)
    _ALLOW_CACHE_RAW = raw
    return _ALLOW_CACHE_VALUE


def timeout_seconds() -> float:
    raw = os.environ.get(ENV_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    if not (MIN_TIMEOUT_SECONDS <= value <= MAX_TIMEOUT_SECONDS):
        return DEFAULT_TIMEOUT_SECONDS
    return value


def diagnostics_dir() -> Path | None:
    """Where the per-rank worker files go. None = the writer must not run.

    Read per write, not at import, so turning it on is an env change in the
    engine's environment (which means a restart) and turning it off takes effect
    on the next write without one.
    """
    raw = os.environ.get(ENV_DIAG_DIR, "").strip()
    if not raw or "\x00" in raw:
        return None
    return Path(raw)


def disk_journal_dir() -> Path | None:
    """Where the spill JOURNAL lives, or None when nothing is configured at all.

    ONE READ POINT, and the only one: `disk_index.journal_directory()` is its sole
    caller, and every journal path (append, tail-seed, refresh glob) is composed
    there, so there is no second place a journal filename could be built.

    `VLLM_E2_TIER_DISK_JOURNAL_DIR` UNSET falls back to `diagnostics_dir()`, which
    is the pre-0.2.19 behaviour: an existing deployment changes nothing until the
    operator sets the new variable. The fallback is deliberate - the variable exists
    so the journal can be moved to a filesystem that survives a reboot, not to make
    the tier demand another knob.

    A path named here is not required to exist: `disk_index.journal_path()` creates
    it (once per resolved directory, mode `disk_index.JOURNAL_DIR_MODE`) on the way
    to the first journal line, because a journal that cannot be written leaves a
    store full of KV files no index can name. A path that cannot be CREATED is a
    counted, published failure (`journal_write_errors`,
    `journal_directory_writable: false`, `journal_dir=unwritable` on the boot line)
    and never a refused spill.
    """
    raw = os.environ.get(ENV_DISK_JOURNAL_DIR, "").strip()
    if raw and "\x00" not in raw:
        return Path(raw)
    return diagnostics_dir()


def diagnostics_epoch() -> tuple[str, str]:
    """(epoch, scope). `scope` is "env" or "process", and it ships in the file.

    A per-process epoch is honest but incomparable: the reader would (correctly)
    refuse to compare facts across ranks. An operator who wants the cross-rank
    verdicts sets VLLM_E2_TIER_DIAG_EPOCH once per boot, and every writer uses it
    verbatim (bounded, whitespace-stripped, control characters refused).
    """
    raw = os.environ.get(ENV_DIAG_EPOCH, "").strip()
    if raw and len(raw) <= MAX_DIAG_EPOCH_CHARS and not any(
            ord(char) < 32 for char in raw):
        return raw, "env"
    return snapshot_epoch(), "process"


def snapshot_epoch() -> str:
    """The plugin's per-process epoch. Imported lazily: `snapshot` is the read
    side and this module must stay importable on its own."""
    from . import snapshot  # noqa: PLC0415 - config is imported by snapshot
    return snapshot.epoch()


def worker_write_interval_steps() -> int:
    """The step cadence, clamped. Garbage falls back to the default."""
    raw = os.environ.get(ENV_DIAG_INTERVAL, "").strip()
    if not raw:
        return DEFAULT_DIAG_INTERVAL_STEPS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_DIAG_INTERVAL_STEPS
    return max(MIN_DIAG_INTERVAL_STEPS, min(MAX_DIAG_INTERVAL_STEPS, value))


def worker_min_interval_seconds() -> float:
    """The rate floor, clamped - the same policy as the interval above.

    Out of range is CLAMPED (a negative floor means "no floor", which is a legal
    choice; an absurd one becomes the cap), while unparseable falls back to the
    default. A typo must not silently remove a budget.
    """
    raw = os.environ.get(ENV_DIAG_MIN_INTERVAL, "").strip()
    if not raw:
        return DEFAULT_DIAG_MIN_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_DIAG_MIN_INTERVAL_SECONDS
    if value != value:                      # NaN: not comparable, so not usable
        return DEFAULT_DIAG_MIN_INTERVAL_SECONDS
    return max(MIN_DIAG_MIN_INTERVAL_SECONDS,
               min(MAX_DIAG_MIN_INTERVAL_SECONDS, value))


def diagnostics_rank() -> str | None:
    """An operator-pinned rank id, or None. Shape is validated by the writer."""
    raw = os.environ.get(ENV_DIAG_RANK, "").strip()
    if not raw or len(raw) > MAX_DIAG_RANK_CHARS:
        return None
    return raw


def disk_index_env() -> dict:
    """The restore index's entry cap, parsed and NOT interpreted.

    The same split as `settle_env` and `ceiling_env`: the grammar lives here, the policy
    that spends the number (the floor, the ceiling and the clamp reason in
    `disk_index.entry_cap`) lives with the index. Unreadable means None, which the index
    reads as "the operator said nothing" and answers with its shipped default - so a
    typo
    in a systemd drop-in gives an index that remembers the usual amount, never one that
    remembers three entries and quietly stops offering the store.

    Zero and negatives are REFUSED by the grammar, like `ceiling_env` refuses a 0
    budget:
    a cap of 0 would mean "hold nothing", which reads as "the index is configured" while
    making every disk offer a miss - the silent-failure shape this tier keeps being
    bitten
    by. How large is too large is policy, and lives with the memory figure.
    """
    problems: list = []
    raw = os.environ.get(ENV_DISK_INDEX_ENTRIES, "").strip()
    entries: int | None = None
    if raw:
        try:
            wanted = int(raw)
        except ValueError:
            problems.append(f"{ENV_DISK_INDEX_ENTRIES}={raw!r} is not an integer and "
                            "was ignored (the index keeps its shipped cap)")
            wanted = None
        if wanted is not None and wanted <= 0:
            problems.append(f"{ENV_DISK_INDEX_ENTRIES}={raw!r} is not above 0 and was "
                            "ignored (a cap of 0 would remember nothing and refuse "
                            "every disk offer)")
            wanted = None
        entries = wanted
    return {"entries": entries, "problems": problems}


def settle_env() -> dict:
    """The settle-wait environment variable, parsed and NOT interpreted.

    ONE variable, because 0.2.32 deleted the arithmetic that spent the other two: the
    allowance is `min(VLLM_E2_TIER_DISK_SETTLE_MS, the load's own deadline)` and there
    is no throughput figure and no prefill rate left to feed (`VLLM_E2_TIER_DISK_SETTLE_
    READ_BYTES_PER_S` and `VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S` are gone, and reading
    them would report a number nothing computes on).

    The split is deliberate: the env grammar ("what did the operator write") lives
    here with every other env var, and the policy that spends the number
    (`disk_restore.settle_budget`) lives with the policy it belongs to. A number read
    here has already been rejected if it cannot be a number, so a typo cannot reach
    the arithmetic at all.

    A value that will not parse is reported in `problems` and left as None, which the
    consumer reads as "the operator said nothing". That is the CONSERVATIVE reading for
    every other knob and the DELIBERATELY EXPLICIT one here: since 0.2.32 the unset
    value is a 5,000 ms allowance, so the only way a typo leaves the boot waiting
    nothing at all is the line `problems` produces - which `settle_budget` publishes and
    the apply point logs once as `DISK[settle] IGNORED`.
    """
    problems: list = []

    def number(name: str, floor: float) -> float | None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            problems.append(f"{name}={raw!r} is not a number and was ignored")
            return None
        if value != value or value < floor:
            problems.append(f"{name}={raw!r} is not a finite number at or above "
                            f"{floor} and was ignored")
            return None
        return value

    # SYNTAX, not policy: the value is checked for "is this the number the operator
    # meant" and nothing else. The ceiling on it is policy and lives with the policy
    # (`disk_restore.SETTLE_MS_MAX`), which reports when it clamps one - so an absurd
    # value is visibly clamped in a published field instead of being silently discarded
    # here and reading as "the operator never set it".
    return {"cap_ms": number(ENV_DISK_SETTLE_MS, 0.0),
            "problems": problems}


def disk_read_env() -> dict:
    """The two parallel-reader env vars, parsed and NOT interpreted.

    The same split as `settle_env`: the grammar lives here, the policy that spends
    the numbers (`disk_read`'s clamp and its defaults) lives with the reader. Two
    consequences worth stating because they are the whole safety argument:

    * A value that is not a positive integer / not a positive byte count is left as
      None and reported in `problems`, which the consumer reads as "the operator
      said nothing" - so a typo in a systemd drop-in gives today's single-thread
      behaviour, never a surprise. There is no default thread count here and no
      default pool size here, precisely so this module cannot become the place a
      boot's behaviour is decided.
    * The ceilings are NOT applied here (they are policy, and `disk_read` publishes
      when it clamps), so an absurd value is a VISIBLE clamp in a payload rather
      than a silently-ignored setting that reads as "never set".

    The buffer ceiling is deliberately NOT multiplied by anything: it is the pool of
    ONE process, and the engine runs one such process per rank.
    """
    problems: list = []

    def integer(name: str) -> int | None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            problems.append(f"{name}={raw!r} is not an integer and was ignored")
            return None
        if value < 1:
            problems.append(f"{name}={raw!r} is below 1 and was ignored")
            return None
        return value

    return {"threads": integer(ENV_DISK_READ_THREADS),
            "buffer_bytes": integer(ENV_DISK_READ_BUFFER_BYTES),
            "problems": problems}


def ceiling_env() -> dict:
    """The ceiling pass's time budget, parsed and NOT interpreted.

    The same split as `settle_env` and `disk_read_env`: the grammar lives here, the
    policy that spends the number (`disk_spill._enforce_ceiling`'s deadline and its
    clamp) lives with the pass. Unreadable means None, which the consumer reads as
    "the operator said nothing" and answers with its own default constant - so a typo
    in a systemd drop-in gives the shipped budget, never a pass with no deadline and
    never a pass with none at all.

    Zero is REFUSED rather than honoured. A budget of 0 would make every pass do
    nothing, which reads as "the ceiling is configured" while shedding nothing - the
    exact silent-failure shape this tier has been bitten by before. A floor of 0 is
    therefore part of the grammar, not of the policy.

    `evict_threads` rides along in the same report because it belongs to the same
    pass: it is the unlink worker count, and the budget is how long that pass may run.
    Grammar only - the ceiling on the count (`disk_spill.MAX_EVICT_THREADS`) is policy
    and lives with the pool that clamps it and reports the clamp.
    """
    problems: list = []
    raw = os.environ.get(ENV_DISK_CEILING_BUDGET_SECONDS, "").strip()
    budget: float | None = None
    if raw:
        try:
            value = float(raw)
        except ValueError:
            problems.append(f"{ENV_DISK_CEILING_BUDGET_SECONDS}={raw!r} is not a "
                            "number and was ignored")
            value = None
        if value is not None:
            if value != value or value <= 0.0:
                problems.append(
                    f"{ENV_DISK_CEILING_BUDGET_SECONDS}={raw!r} is not a finite "
                    "number above 0 and was ignored (0 would make every ceiling "
                    "pass do nothing)")
            else:
                budget = value
    # THE UNLINK WORKER COUNT, same grammar, same "a typo means the shipped default"
    # rule. It rides along with the budget because both belong to the ONE pass: this is
    # how many threads do the unlinks, that is how long the pass may run. The ceiling
    # on the value is policy and lives with the pool that clamps it.
    evict_raw = os.environ.get(ENV_DISK_EVICT_THREADS, "").strip()
    evict_threads: int | None = None
    if evict_raw:
        try:
            wanted = int(evict_raw)
        except ValueError:
            problems.append(f"{ENV_DISK_EVICT_THREADS}={evict_raw!r} is not an "
                            "integer and was ignored")
            wanted = None
        if wanted is not None:
            if wanted < 1:
                problems.append(f"{ENV_DISK_EVICT_THREADS}={evict_raw!r} is below 1 "
                                "and was ignored (0 workers could unlink nothing)")
            else:
                evict_threads = wanted
    return {"budget_seconds": budget, "problems": problems,
            "evict_threads": evict_threads}


def delivery_timeout_env() -> dict:
    """The delivery step's wall-clock bound, parsed and NOT interpreted.

    The same split as `ceiling_env`, `settle_env` and `disk_read_env`: the grammar
    ("what did the operator write") lives here, the policy that spends the number
    (`disk_gpu_write._delivery_apply_timeout_seconds`, which every apply point reads and
    publishes the effective value of) lives with the loop it bounds. There is therefore
    NO default of 10 s in this function: `seconds=None` means "the operator said
    nothing", which is the only way the constant in `disk_gpu_write` stays a fact a
    reader can check instead of a number this module invented.

    Zero is REFUSED rather than honoured, exactly like `ceiling_env` refuses a 0 budget.
    A bound of 0 would defer every entry of every delivery step, which reads as "the
    timeout is configured" while delivering nothing - the same silent-no-op shape, in
    the one place where the entries the bound refuses are the entries the prune may not
    remove. Negative and non-finite are the same story. Anything the grammar rejects is
    named in `problems` and means the shipped default.
    """
    problems: list = []
    raw = os.environ.get(ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT, "").strip()
    seconds: float | None = None
    if raw:
        try:
            value = float(raw)
        except ValueError:
            problems.append(f"{ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT}={raw!r} is not a "
                            "number and was ignored")
            value = None
        if value is not None:
            # `!= value` is NaN; the explicit +inf test is what `ceiling_env`'s idiom
            # misses, and an infinite bound is exactly the "no bound at all" a typo must
            # never produce. Negative infinity is already caught by `<= 0.0`.
            if value != value or value == float("inf") or value <= 0.0:
                problems.append(
                    f"{ENV_DISK_GPU_WRITE_DELIVERY_TIMEOUT}={raw!r} is not a finite "
                    "number above 0 and was ignored (0 would defer every entry of "
                    "every delivery step)")
            else:
                seconds = value
    return {"seconds": seconds, "problems": problems}


def cpu_slot_reserve_env() -> dict:
    """The CPU-slot reserve the operator asked for, parsed and NOT interpreted.

    The same split as `settle_env`, `disk_read_env` and `ceiling_env`: the grammar
    ("what did the operator write") lives here, the policy that spends the number
    (`disk_restore.cpu_slot_reserve`, which the offer arithmetic, the boot verdict and
    the payload all read) lives with the arithmetic it belongs to. So there is no
    default of 64 in this function: `slots=None` means "the operator said nothing" and
    the consumer answers with its own constant, which is the ONLY way the constant at
    `disk_restore.CPU_SLOT_RESERVE` stays a fact a reader can check instead of a number
    that has quietly stopped deciding anything.

    WHERE THIS KNOB IS DIFFERENT FROM THE OTHERS, and why: `ceiling_env` refuses 0
    because a ceiling pass with a 0-second budget does nothing while reading as
    configured, and `disk_read_env` refuses 0 threads for the same reason. This one
    HONOURES 0. A reserve of 0 is not a dead knob - it is the posture "a disk offer may
    take the arena's last free slot", which is exactly what an operator running a small
    arena on purpose may want to buy, and the arithmetic below it stays correct at 0
    (`free - 0 - boundary_slots`). A NEGATIVE value is a different animal: it would
    hand the offer MORE slots than the pool has, so it is refused as a grammar error
    and reported, and there is no arithmetic that can reach it.

    No ceiling either, for the reason in the constant's comment: an over-large reserve
    declines offers, which is the direction the reserve exists to bias. A non-integer
    (`16.0`, `16 slots`, an empty-but-present drop-in value) is left as None and named
    in `problems`, which `disk_restore.cpu_slot_reserve` logs ONCE per distinct
    complaint as `DISK[reserve] IGNORED` - one line per boot, not one per offer.
    """
    problems: list = []
    raw = os.environ.get(ENV_DISK_CPU_SLOT_RESERVE, "").strip()
    slots: int | None = None
    if raw:
        try:
            wanted = int(raw)
        except ValueError:
            problems.append(f"{ENV_DISK_CPU_SLOT_RESERVE}={raw!r} is not an integer "
                            "and was ignored")
            wanted = None
        if wanted is not None:
            if wanted < 0:
                problems.append(
                    f"{ENV_DISK_CPU_SLOT_RESERVE}={raw!r} is below 0 and was ignored "
                    "(a negative reserve would hand the offer more CPU slots than the "
                    "arena has)")
            else:
                slots = wanted
    return {"slots": slots, "raw_value": raw or None,
            "env_var": ENV_DISK_CPU_SLOT_RESERVE, "problems": problems}


def canonical_policy_path() -> Path:
    """The only file the reload verb can ever load."""
    return _PKG_DIR / "policy.py"


def validate_policy_target(candidate: object) -> Path:
    """Return the resolved Path for `candidate`, or raise PolicyPathError.

    The rule is deliberately narrower than "inside the plugin directory":
    the target must resolve to exactly `<package dir>/policy.py`. That kills
    `../`, symlinks out of the tree, `policy2.py`, module names like
    `os.path`, and absolute paths to anything else - from the wire or from
    anywhere else. Nothing here trusts the caller.
    """
    if not isinstance(candidate, (str, Path)):
        raise PolicyPathError("policy path must be a string, got "
                              f"{type(candidate).__name__}")
    text = str(candidate).strip()
    if not text:
        raise PolicyPathError("policy path is empty")
    if "\x00" in text:
        raise PolicyPathError("policy path contains a NUL byte")

    path = Path(text)
    if not path.is_absolute():
        path = _PKG_DIR / path
    try:
        # strict=False: a missing file must be reported by name below, not as
        # an OSError from resolve(). resolve() follows symlinks, which is what
        # makes the parent/name comparison below a real containment check.
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:  # ELOOP, deep nesting, ...
        raise PolicyPathError(f"policy path does not resolve: {exc}") from exc

    if resolved.parent != _PKG_DIR or resolved.name != "policy.py":
        raise PolicyPathError(
            "refused: the only loadable policy is "
            f"{str(canonical_policy_path())!r}, not {str(resolved)!r}"
        )
    if not resolved.is_file():
        raise PolicyPathError(f"refused: {str(resolved)!r} is not a file")
    return resolved
