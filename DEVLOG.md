# kvtier-dev-plugin - development record

This is the dated development record of this plugin, cut verbatim out of
`README.md` when that file was reduced to an install-and-operate README.
Nothing here was rewritten: every entry below is byte-identical to the text it
came from, including its measurements and its wrong turns.

**Order: chronological, oldest first**, by the `0.2.NN` in each entry's heading.
An entry whose original heading carried no version shows one anyway, with how it
was obtained: `(text)` = a release named in the entry's own body, `(version from
git: ...)` = the commit that introduced the text and the package version in force
there, `(inherited from the preceding entry)` = ordered where it sat in
`README.md`, which does **not** assert it was written in that release.

**Wrong turns are deliberate.** Several entries state a belief the work held and
then records the measurement that falsified it. Do not "clean up" those
paragraphs: the falsified belief is the reason the rule exists, and an entry
stripped of its wrong turn reads like a design that was always right.

Cross-references inside these entries of the form *see X above / below* were
written while all of this lived in one `README.md`. Where the target moved, `X`
names an entry title in this file (or a section that stayed in `README.md`) -
search for the title rather than trusting the word "above".

## Making the tier safe to enable: the CPU arena capacity clamp (`cpu_capacity.py`) - 0.1.0 (version from git: text landed in 425763f, package was 0.1.0)

The defect this exists to close (measured on the live TP2xPP2 layout,
`VLLM_PP_LAYER_PARTITION=25,23` + MTP on the last stage):

* the scheduler sizes its CPU block-id space from ONE aggregate view
  (`manager.py:270-272`, `manager.py:167`), while
* every worker sizes its OWN pinned arena from its OWN stage's bytes/block
  (`worker.py:162`). Here that is 11,468,800 B on PP0 vs 12,902,400 B on PP1, so
  at `--kv-offloading-size 8` the scheduler emits ids up to **187** while the
  last rank's arena holds **166** (always ~13%), and
* `cuda_mem_ops.py:243-249` computes `src_bases[:, None] + ids[None, :] *
  bpb[:, None]` with **no bounds check**: an out-of-range id is a write past the
  end of that rank's pinned host arena.

Dropping an out-of-range copy in the worker is not a fix: a dropped *store* is a
lost park, but a dropped *load* is the model attending to garbage KV, because the
scheduler already counted those tokens as computed. So the requirement is that
the scheduler **never emits** an out-of-range id.

The mechanism is the fork's own PP-aware handshake, which runs inside
`EngineCore.__init__` (`core.py:288-302`) after KV-cache registration and before
any traffic: `collective_rpc("get_kv_connector_handshake_metadata")`
(`abstract.py:214-217`, `gpu_worker.py:868-895`) collects one dict per worker,
they are merged into `content: dict[(pp_rank, tp_rank), metadata]`, and handed to
`kv_connector.set_xfer_handshake_metadata_pp_aware(content)`
(`base.py:716-731`). Three wrappers, installed by the existing
`vllm.general_plugins` entry point:

| side | wrapped | what it does |
|------|---------|--------------|
| worker | `KVConnectorBase_V1.get_handshake_metadata` (`base.py:447-456`) | calls the original, then attaches this rank's `worker_handler.num_cpu_blocks`. A `None` payload becomes a bare carrier; a `dict` payload becomes a **copy** plus our key; anything else is wrapped under a second key and unwrapped again on the scheduler side, so a connector with a typed payload still receives the original object. |
| scheduler | `KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware` (`base.py:716`) | calls the original **first**, with exactly the payload the fork would have received without the plugin (our in-band keys stripped, carriers contributing no entry at all - `base.py:725` raises on a `pp_rank > 0` entry this connector does not support, so an naive carrier would be a boot crash, not a clamp), then reads every reported capacity and clamps. |
| clamp | the manager's CPU `BlockPool` | one-time surgery at handshake time. |

**The clamp is surgery, not a hot-path check.** Ids `>= cap` are withdrawn from
the CPU pool's free structure once (`FreeKVCacheBlockQueue.remove` is O(1) per
node, and the LRU order of the ids that stay is untouched), and they are evicted
from the prefix-cache map first so no hash entry can name a block that can no
longer be allocated. `get_num_free_blocks()` (`block_pool.py:861`) and
`get_new_blocks()` (`block_pool.py:664`) then agree with the cap *by
construction*, every existing tier counter stays honest, and the per-step code
path costs nothing. The mirrors follow: `manager.num_cpu_blocks`
(`manager.py:167`), the pool's own `num_gpu_blocks` (`block_pool.py:173`, which
`get_usage()` and `reset_prefix_cache()` do arithmetic on) and the manager's CPU
`kv_cache_config.num_blocks` (`manager.py:164`, a `replace()`-derived copy -
never the engine's own GPU config). The pool's `blocks` list keeps its full
length on purpose, so `pool.blocks[bid]` for an id the fork already emitted
cannot become an `IndexError`. A tiny **instance-level** wrapper (never
class-level - the same `BlockPool` class backs the GPU pool) counts the
allocations the fork's own guard refuses; it decides nothing.

**Unknown means park nothing.** If no rank reports, a rank reports 0, a rank
reports without a capacity, or the payload is not keyed by `(pp_rank, tp_rank)`,
the cap is 0: the CPU pool keeps only its null block. Every tier allocation site
tests free space *before* allocating (`manager.py:698-701` eager,
`manager.py:586/600` lazy, `manager.py:829/834` the positional scan), so zero
free CPU blocks is a silent no-op step, never an exception, and the engine boots
and serves. Uncapped is never the fallback. As a backstop, the snapshot and
reload verbs call `adopt_if_missed()`: the handshake runs in `__init__` while a
utility handler can only run from the busy loop, so if the handshake never
delivered a ceiling by the first snapshot, the ceiling is invented at 0 and
reported as `adopted_by_snapshot`.

The whole answer is one object in the snapshot, `cpu_capacity`: `installed` and
`reason`, `per_rank` / `per_rank_list` (the arena sizes each rank reported,
keyed `"pp_rank,tp_rank"` because a tuple is not a msgpack key), `chosen_cap`,
`cap_source`, `clamp_applied`, `zero_mode` + `zero_reason`,
`adopted_by_snapshot`, `clamp_failed_open` (we could NOT prove a ceiling - the
field to page on), `cpu_blocks_withheld`, `pool_free_after`,
`manager_num_cpu_blocks_before/after`, `allocations_refused_by_cap`,
`entries_without_capacity`, `original_call_error`, and `live_cpu_pool` read off
the manager that is actually running.

## Making the tier bootable at all: the PR #56810 backport (`backport.py`) - 0.1.0 (version from git: text landed in 64e657f, package was 0.1.0)

The tier cannot be enabled on our model without this fix, so it is installed on
the way through every verb - while **the tier itself is behind a switch that is
OFF at boot** (see *Booting the tier* below).

Upstream issue #56396 / PR #56810 (merged 2026-09-20, NOT in the `v0.30.0` tag
this fork is on). `SimpleCPUOffloadScheduler` assumed every KV cache group's
block size is a multiple of the hash block size. Core only validates that over
the PREFIX-CACHEABLE groups (`kv_cache_utils.py:768-781`), so a hybrid model
dies in the store path:

```
manager.py:842 _select_eager_blocks_to_store -> resolve_block_hashes(hashes, 3200, 8)
kv_cache_utils.py:2888:  assert block_size % hash_block_size == 0
```

Our model is five `block_size=3200` groups plus the QSA raw-key ring, a
`CircularBufferSpec` of block size **8**. The ring's size is computed
(`qsa_cache.py:900-901`: `compress_ratio * cdiv(compress_ratio +
num_speculative_tokens, compress_ratio)` = `4 * cdiv(8, 4) = 8`), so it is not a
knob we can turn. It reaches the assert because the loop's only filter was
`has_positionally_stable_blocks`, which `CircularBufferManager` inherits as True
(unlike `MambaManager`).

**`--prefix-match-unit 4` or `8` is a dead end, and must not be re-proposed.**
It would make the sizes divisible by lowering `hash_block_size`, but (1) a
ring block is positionally UNSTABLE - a circular buffer reuses slots - so
hashing it in 8-token units creates prefix-cache entries that can hit against
stale ring content; the group is `prefix_cacheable=False` because its content
must never be matched, and the assert is telling us something true; and (2) it
multiplies prefix hashing by ~400x (every 8 tokens instead of every 3200, per
group, per step) for entries that can never be hit.

Four methods change, all inside `SimpleCPUOffloadScheduler`: record
`prefix_cacheable_group_ids`; iterate `prefix_cacheable_groups` when estimating
the lazy target; give a non-cacheable group `cpu_hit_blocks.append([])`; and
`continue` past a non-cacheable group in the store selector, placed BEFORE the
`len(gpu_block_ids) >= num_free` break so a scratch group can never consume CPU
blocks or emit `BlockStored`.

### Two guards, because a wrong patch here corrupts output silently

**Source fingerprints.** For each method the plugin stores the SHA-256 of
`inspect.getsource()` at fork HEAD `10b8e07`. At install time a mismatch means
the function moved: that method is REFUSED (reported per-method as
`patched: false` with the reason), never patched "close enough". Three methods
are replaced by a byte-identical copy of upstream's body exec'd in the
ORIGINAL function's `__globals__` (a wrapper cannot inject a `continue` into a
loop it cannot see); `__init__` is WRAPPED because its change is purely
additive and provably identical to setting the attribute after the original
returns.

**The geometry guard.** `manager.py:456-458` silently skips any group with no
parked blocks, so a *blanket* "skip whatever trips the assert" policy is an
output-corruption machine: drop a genuinely cacheable group and its KV is
simply missing on resume, with no error. `geometry_guard()` recomputes the set
the patch would skip from the live `KVCacheConfig` and REFUSES to install if any
of those groups reports `prefix_cacheable=True`, naming the group. If the patch
is already live on a class and a LATER config fails the guard, the tier is
auto-disarmed rather than left running (`auto_disarmed: true`).

### Booting the tier: the boot brings it up working (`VLLM_E2_TIER_DEV_BOOT_INERT`)

> **A tier-env boot ends with the tier ARMED and GUARDED, without a human in the
> loop - and it arms nothing it cannot verify.**

The install cannot land when the plugin *loads*: `load_general_plugins()` runs at
`core.py:217`, but the Scheduler - and with it the connector and the offload
manager - is not constructed until `core.py:260`, so `register()`'s install
attempt is *always* refused at boot (`cls is None`). For a while the answer was
"then the boot stays inert and a human arms it", which is safe but not a system:
every boot needed a curl, and a forgotten curl was a tier that silently parked
nothing. Both halves are now handled at the one moment the class exists:

* **`BOOT_ARMED = False` still holds at plugin load.** Nothing is armed before
  there is anything to arm, so no code path can ever arm an unpatched engine.
* **The install no longer waits for a verb.** `register()` fits a hook on
  `KVConnectorFactory.create_connector` (`factory.py:43`). When that returns a
  `SimpleCPUOffloadConnector` carrying a `scheduler_manager`, the hook runs the
  whole install against the class the connector module just imported
  (`simple_cpu_offload_connector.py:21`) - the first instant that class exists,
  still inside `Scheduler.__init__` (`scheduler.py:164`, reached from
  `core.py:260`), long before the first request - and it passes the manager's
  LIVE `kv_cache_config`, so the geometry guard gets a real check at boot rather
  than `checked: false`. The order inside `install()` is **wrappers FIRST, then
  the backport**, so a refused or exploded patch still leaves "wrappers fitted +
  disarmed", which gates *both* routes to the selector (see *The kill switch*).
  `pr56810.connector_hook` says what happened (`fitted`, `tier_connectors`,
  `worker_role_no_manager`, `installs_run`, `errors`, `last`), and
  `pr56810.patched_before_first_request` is the one-word answer.
* **Then the BOOT runs: guard, then arm.** `backport.boot_from_connector()` is the
  whole sequence, and it is strict in one direction only - it will not arm what it
  cannot verify:

  1. **Guard mode is installed and armed.** Its wrappers are *composed* over the
     backport's and the switch's (`nested`), never put under them, so there is no
     window in which a store or a lookup can run unguarded.
  2. **The tier is armed** - only if the install reported `installed: true` (the
     all-or-nothing verdict: every required method patched, every kill-switch
     wrapper fitted), **and** the live-object check passed on the connector's own
     `scheduler_manager` (the method *objects* it resolves are the ones carrying
     our markers, so an instance built before the patch was fitted cannot pass),
     **and** the guard is a layer of every entry point the class actually has.
  3. Anything else leaves the tier disarmed and puts the reason in
     `pr56810.boot.refusals` - never a silent no.
  4. **Then the disk tier's own boot default runs** (`KVTIER_DISK_BOOT_ENABLE`, ON by
     default since 0.2.33): it arms the SSD spill tier - as a LATCH since 0.2.35, with no
     expiry - and raises this boot's serve posture, so the box comes up `serving_as=disk_authority` with no
     POST. It calls the same two functions the `/disk` verb calls - `disk_spill.arm()` and
     `disk_restore.set_disk_authority()` - invents no third way to arm, runs after the
     RAM tier's arm (a disk arm on a boot that armed no tier is refused), and rolls the
     pair back to uniformly disarmed if either half fails. Its verdict is
     `pr56810.boot.disk_boot`, its token is `disk_boot=` on the BOOT line, its own log
     line is `KVTIER disk boot default=...`, and the whole design is in *The SSD tier's
     boot default (0.2.33)* below; `disk_restore.boot_default` in the snapshot (and
     `restore.boot_default` on every `/disk` answer) is where a poller reads it.

  Guard before arm is not a style preference: arming the tier without the guard
  means the first real tier bug is an EngineCore death instead of a recorded
  decline, and the arm gate therefore treats "the guard refused this method" as a
  reason not to arm.
* **One line says all of it.** `pr56810.boot.line`, also written to the engine log
  at WARNING so it survives without curling anything:

  `BOOT[auto] backport=YES guard=YES ram_tier=ARMED disk_tier=ARMED tier=ARMED faults=0 journal_dir=tmpfs arena=OK disk_boot=ARMED MAMBA_ANCHOR[patched] JOURNAL_DRAIN[converged] guard_over=... not_covered=... restore=YES restore_gate=closed restore_ranks=0 captures=...`

  `tier=` is the **RAM/host tier** and stays exactly as it was - the restart verifier
  greps it as a PASS criterion - while `ram_tier=` repeats that verdict under an
  unambiguous name and `disk_tier=` reports the **disk spill tier's arm**, whose file
  lives on tmpfs and is gone after every reboot. **What changed in 0.2.33 is what that
  field says on a default boot: `ARMED`, not `DISARMED`**, because the boot default armed
  it - and `disk_boot=` names the boot default's own verdict (`ARMED` / `OFF` /
  `NO_STORE` / `INERT` / `REFUSED` / `UNDECIDED`). A boot can still legitimately print
  `disk_tier=ARMED` next to a restore refusal (the worker proof is a separate gate -
  `restore_gate=closed` above says so): arming says bytes may be written, the serve
  posture says requests may be served, and the worker proof says a worker will fill the
  slots. The two named fields are printed *before* `tier=` so that a
  `grep -oE 'tier=[A-Za-z_]+' | tail -1` still lands on the bare field.

  `mode` is `auto` or `inert` and says which env posture ran; `refused=` names each
  gate that failed with its reason; `not_covered=` lists the methods the guard does
  **not** wrap (methods the fork's class does not define, or ones it inherits from
  outside the tier module) so the line never implies more coverage than it has;
  `faults=` is the guard's fault count at that instant, repeated as `faults_now`
  when you read the payload later.

```bash
# 1. boot the engine (tier env set, VLLM_E2_TIER_DEV_BOOT_INERT unset). Expect THREE
#    lines in /tmp/vllm-serve.log:
#    "kvtier-dev-plugin: connector-creation hook fitted on KVConnectorFactory..."
#    "kvtier-dev-plugin: KVTIER disk boot default=ON (KVTIER_DISK_BOOT_ENABLE unset -> default ON): HONOURED: ... serving_as=disk_authority with NO POST. ..."
#    "kvtier-dev-plugin: BOOT[auto] backport=YES guard=YES ram_tier=ARMED disk_tier=ARMED tier=ARMED ... disk_boot=ARMED ..."
#    and NOTHING else to do. `grep 'BOOT\[' /tmp/vllm-serve.log | tail -1` is the
#    whole boot verdict; `grep 'disk boot default' /tmp/vllm-serve.log | tail -1` is the
#    disk tier's half, and it names the way to turn the default off.

# 2. read it back. This no longer PATCHES or ARMS anything (the boot did), it
#    CONFIRMS it: expect installed: true, armed: true, boot.mode: "auto",
#    boot.ran: true, patched_before_first_request: true
ssh llm 'curl -s "http://127.0.0.1:8000/plugins/kvtier/tier?op=status"' \
  | python3 -c 'import sys,json;p=json.load(sys.stdin);b=p["pr56810"]["boot"];print(p["armed"],b["mode"],b["backport"],b["guard"],b["tier"],b["faults_at_boot"])'

# 3. prove the boot was CLEAN, not merely UP. A guard-arm means a tier exception
#    is now a DECLINE, so a broken tier looks healthy from `armed` alone.
ssh llm "curl -s -X POST http://127.0.0.1:8000/plugins/kvtier/guard \
  -d '{\"op\":\"captures\",\"limit\":5}'" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["count"],"faults");[print(c["method"],c["exception_type"],c["message"][:80]) for c in d["captures"]]'

# 4. send traffic.
```

Steps 2 and 3 are *confirmations*, not the mechanism. The verbs remain the
backstop - `ensure_installed` re-fits the hook and re-runs the install on `/tier`,
`/snapshot` and `/policy/reload`, which is what covers a process whose connector
factory was imported only after the plugin loaded, or a manager class that
appeared later. A boot where `armed: false` survives step 2 with the mode `auto` is
NOT "nobody has armed it yet": it is a refusal, and `boot.refusals` names the gate
(the install, the live-object check, or the guard's coverage of the class).

The deliberate shortcut in that design is worth naming: the guard covers the nine
entry points it knows and refuses to wrap what it cannot own, and the boot treats a
method the class does not define as *not covered* rather than as a failure -
otherwise a fork that renames or inherits a method would take the whole tier down.
That is why the BOOT line prints `not_covered=` instead of a bare `guard=YES`.

An `arm` is refused, never forced, by the boot or by a verb, and the refusal kind
tells you which gate you hit (`arm_block.kind`, `state.kind`, `disarm_reason`):

| kind | meaning | what to do |
|---|---|---|
| `incomplete_install` | a required method was refused (a fingerprint drift: the fork moved that function); names the method(s) | fix/re-record the source, then arm again |
| `geometry` | the guard vetoed this config: the patch would skip a group that is still `prefix_cacheable=True` | do **not** arm; boot without the tier or fix the geometry |
| `no_manager` | no `SimpleCPUOffloadScheduler` exists in this process - the tier connector is not configured, or the engine is not built yet | there is nothing to arm here; check the tier env of the engine you reached |

And because `armed: false` on a tier-env boot is now an *exception to explain*,
every payload says so where an operator will read it: `pr56810.boot` (the verdict
and its reasons), `pr56810.boot_default_armed`, `pr56810.boot_is_inert`,
`pr56810.arm_requires`, and `pr56810.operator_note` - a sentence that answers "the
engine booted with the tier on, why is it parking nothing?". The `/tier` verb
repeats `armed`, `boot_default_armed`, `operator_note` and a `next_step` at the top
level, next to `applied`.

### The SSD tier's boot default (0.2.33) (`KVTIER_DISK_BOOT_ENABLE`)

> **A default boot arms the SSD tier and serves from it, with no POST - and the flag can
> only ever take a boot back to manual, never forward into serving.**

Until 0.2.32 every boot came up with the SSD tier doing nothing and stayed that way until
somebody POSTed twice - `{"op":"arm","seconds":7200}` and then
`{"op":"authority","enabled":true}`. Both are per boot, so every restart, every wake and
every "let me just restart it" silently reverted to a tier that serves nothing while
`serving_as` truthfully reported `off`. That is the most common way this feature looks
broken to a human, and it is what this release removes.

**One flag, ON by default.** `KVTIER_DISK_BOOT_ENABLE` unset (or `1`/`true`/`yes`/`on`)
means the boot that armed the RAM tier also arms the spill tier for the full cap and
raises the serve posture, so the engine comes up `serving_as=disk_authority`. Set
`0`/`false`/`off`/`no` and the boot is byte-for-byte the 0.2.32 boot: disarmed spill tier,
`serving_as=off`, `restore.gate.reason_kind: "disarmed"`, and the two POSTs are the only
lever. A value that is not a word in either vocabulary (`flase`) reads as the default -
**ON** - and says so on the boot line, because a typo that silently means "disabled" is a
box still serving from disk after somebody decided it should not.

**It is a disable switch, not an arm switch.** Every other facility here starts off and
is armed by a POST (peer-gated, like everything else since 0.2.68), and that rule is
untouched: guard mode, the RAM tier's
arm, the byte dump and `spill-one` have no env var. `KVTIER_DISK_BOOT_ENABLE` has the
same shape as `VLLM_E2_TIER_DEV_BOOT_INERT` - it can only turn a boot effect **off** - and
`VLLM_E2_TIER_DEV_BOOT_INERT` outranks it, because an inert boot arms nothing at all and
the disk default obeys that (reported as `INERT`, not as a failure).

**THE PROHIBITION IT EXISTS TO RESPECT.** `llm-api/AGENTS.md` (the 2026-10-03 incident)
forbids a boot-time re-arm: *"A stale `disk-spill-arm.json` did NOT re-arm workers nine
hours later; ranks report `armed=False` on their own. Do not add a boot-time re-arm."* A
proxy-side "restore posture on engine-ready" hook was written for exactly that and was
deleted rather than left default-off. The line this feature walks, and the only line
there is: a **previous boot's persisted state decides nothing**. The boot's decision comes
from this one environment variable and from the boot's own measurements, and
`disk_boot.py` contains no vocabulary for reading an arm file - no `open`, no `json`, no
`_arm_path`, no `boot_id` - which `test_the_prohibition_structurally_the_module_never_reads_an_arm_file`
pins against the module's syntax tree, while
`test_the_prohibition_no_boot_time_re_arm_from_persisted_arm_state` pins it behaviourally
by planting an arm file that claims `"armed": true` and naming a **different kernel boot
id** (what a reboot actually leaves behind): the boot comes up disarmed with the flag `0`
and armed with the flag unset, and in both cases the planted file is refused by
`disk_spill`'s own boot fence - and with the flag `0` the boot's fail-safe **deletes** it,
because since 0.2.35 a fence that cannot see an engine restart is no longer enough. The
sibling test `test_the_boot_disarms_a_file_left_by_an_earlier_boot_on_one_kernel` plants
the file the fence CANNOT refuse (this kernel's boot id) and pins that it is gone. A surviving arm file still arms nobody; a restart is
still a rollback.

**The gates, and the fail-safe direction.** Every outcome except `ARMED` leaves this boot
uniformly disarmed, and `ARMED` requires **both** halves to have landed:

| gate | outcome | why it is a refusal |
|---|---|---|
| the flag says no | `OFF` | today's boot, exactly |
| `VLLM_E2_TIER_DEV_BOOT_INERT` set | `INERT` | that boot arms nothing, disk included |
| no `VLLM_E2_TIER_DISK_ROOT` | `NO_STORE` | nothing to arm - **not** a fault |
| the RAM tier did not arm | `REFUSED` | a disk offer needs scheduled load to ride on |
| the restore layer is not fitted in this process | `REFUSED` | it would write spilled KV and never read a byte |
| the host-RAM arena is tiny / has no pool / unreadable | `REFUSED` | the tier moves bytes through that arena both ways |
| the spill journal cannot be written where it lives | `REFUSED` | every spill would be a file no index can name |
| `disk_spill.arm()` refused | `REFUSED` | the store's own objection, reported verbatim |
| the arm could not be published to the ranks | `REFUSED` + rollback | an EngineCore-only arm is a half-armed fleet |
| the posture could not be raised | `REFUSED` + rollback | an armed tier that serves nothing is the state to avoid |

Nothing here can raise an engine's boot into an exception: the whole path is wrapped, and
an internal raise is reported as `REFUSED` with the exception type named.

**THE ARM IS A LATCH (0.2.35). There is no expiry, so there is nothing to renew.** Until
0.2.34 the arm was a timer (`ARM_DEFAULT_SECONDS` 1,800 s, `ARM_MAX_SECONDS` 7,200 s, both
now deleted) and this module carried a `renew_if_due()` tick that rode the scheduler's
offer path to beat the cap - a whole heartbeat mechanism whose only job was to keep a
clock from going dark. Deleting the clock deleted the tick, its constants, its counters
(`renewals`, `renewal_failures`, `last_renewal`) and its payload fields; `arm_state()`
reports `latched: true` and no time, and the boot view answers `arm_expires: false` where
it used to answer `seconds_left`.

**Why the timer went.** The operator's ruling, after the live box made the case: KV
persists until the LRU sheds it under the ceiling, and the ceiling eviction that sheds it
**runs from the spill writer path** - so an arm that expired froze *shedding* along with
growth, which is the opposite of the product: a tier that cannot shed is a tier that
refuses spills forever. The timer was not buying a bound either. At the live 11,468,800 B
slot and 200 files / 60 s one rank writes ~38.2 MB/s, which fills a 192 GiB ceiling in
about 90 minutes: a 7,200 s arm permits ~1.34 ceiling-fills and the 1,800 s default ~0.33.
Capacity is bounded independently of the arm - by the byte ceiling, the per-rank fair
share, LRU-by-touch, the `MIN_FREE_BYTES` free-space floor and the write-rate window - and
two of those mechanisms run **only while the arm is up**. The 2026-10-05 livelock the
expiry was partly credited with preventing is closed by six mechanisms that read no arm
expiry (the AND-over-ranks proof with per-rank service facts, the arm read live from one
authoritative file, the request and span fuses, the forward-progress fuse, the
contiguous-run offer bound, and a CONFIRMED witness that cannot be reached without proof).

**What the latch does not survive is a decision nobody made.** The boot fence in
`disk_spill._arm_boot_fence` still refuses an arm file naming a different **kernel** boot,
but `/proc/sys/kernel/random/boot_id` cannot see an ENGINE RESTART inside one long uptime
- four boots in a 2w3d uptime all look like "this boot" - and while the arm had a TTL, the
TTL is what made a file from boot #1 harmless to boot #2. The rule that replaces it is
**THE ONLY THING THAT ARMS A PROCESS IS THAT PROCESS'S OWN BOOT DECISION**: `boot_enable()`
calls `disk_spill.disarm()` on every outcome except `ARMED`, and `disarm()` deletes the
file. A boot that did not choose to arm leaves nothing behind to be misread.

**A human always wins, and the ordering is not negotiable - and since 0.2.35 it is the
veto that enforces it.** `POST /plugins/kvtier/disk {"op":"disarm"}` - or
`{"op":"authority","enabled":false}` - latches `disarmed_by_operator: true` with the
reason, and the boot decision **refuses to arm while it stands**. That is the veto's new
job: it used to gate `renew_if_due()` and nothing else, so when renewal was deleted it
guarded an empty set and could have gone with it. It is kept, for a stronger reason - with
a latch the path that can still re-arm a box behind its operator is `boot_enable()`
itself, and a second connector boot inside one process (which a re-instantiated engine
does) would otherwise raise an arm that, with no timer, would never come down. A manual
`{"op":"arm"}` lifts the veto: only a human at the verb does, and only because that human
asked for the tier back. A *refused* POST latches nothing, so a malformed curl cannot
silently kill the feature. And because the latch gates **both** directions, disarming
closes serving from disk as well as writing to it - `gate()` answers
`reason_kind: "disarmed"`, and `boot_enable`'s fail-safe and `_may_restore_worker` read
the same verdict.

**One line, at boot, naming how to turn it off.** It is WARNING-level, printed once per
process, and it is the answer to "what did this boot decide and how do I change it?"
without opening a source file. Verbatim, on a default boot:

```
kvtier-dev-plugin: KVTIER disk boot default=ON (KVTIER_DISK_BOOT_ENABLE unset -> default ON): HONOURED: armed the spill path and set this boot's serve posture, so it comes up serving_as=disk_authority with NO POST. The arm is a LATCH (0.2.35): it has no expiry, there is nothing to renew, and it comes down on a POST that disarms it and not on a clock. It gates SERVING from disk as well as writing to it, so the same POST stops both. to stop every future boot doing this: set KVTIER_DISK_BOOT_ENABLE=0 in the engine's environment and restart (VLLM_E2_TIER_DEV_BOOT_INERT=1 turns off the whole auto-boot, not just the disk tier); to stop THIS boot right now: POST /plugins/kvtier/disk {"op":"authority","enabled":false} and then POST /plugins/kvtier/disk {"op":"disarm"} - the arm is a LATCH, so that holds until someone arms again, and a manual disarm vetoes this module's own boot decision for the rest of the process too
```

```bash
# What the boot decided, three ways. All three read the same state through the same
# builder, so they cannot disagree: the boot line's `disk_boot=` token, the snapshot's
# `disk_restore.boot_default`, and `/disk`'s own `disk_boot_default`.
grep 'disk boot default' /tmp/vllm-serve.log | tail -1
curl -s $B/snapshot | python3 -c \
  'import sys,json;b=json.load(sys.stdin)["disk_restore"]["boot_default"];print(b["outcome"],b["honoured"],b["armed_now"],b["arm_expires"],b["disarmed_by_operator"])'
curl -s -X POST -d '{"op":"status"}' $B/disk | python3 -c \
  'import sys,json;print(json.load(sys.stdin)["disk_boot_default"]["reason"])'

# Take it off THIS boot, and permanently, in that order.
curl -s -X POST -d '{"op":"authority","enabled":false}' $B/disk
curl -s -X POST -d '{"op":"disarm"}' $B/disk
# ...and for the next boot: Environment=KVTIER_DISK_BOOT_ENABLE=0, then restart.
```

The payload is the same dict on every door (`disk_restore.boot_default` in the snapshot,
`restore.boot_default` on every `/disk` answer, and `disk_boot_default` again at the top
of a `/disk` `{"op":"status"}`), built by one function, and it carries **the same key set
on every path** - a boot that never ran the default, one that was refused, one whose flag
was `0` and one whose arm could not be read all answer with the same fields, so an
absence never has to be interpreted on the path where something went wrong.

## Measured retention: three answers 785x apart, resolved by measuring - 0.2.6 (version from git: text landed in 9ed1250, package was 0.2.6)

The authoritative path in the running tree is
`num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // kv_cache_tensors[0].size)`
(`manager.py:273-275`); there is no `cpu_block_size = lcm(...) * sum(page_sizes)`
formula in it. The disputed part was never the pool size, it was what one slot
means:

* a CPU slot is **one GROUP's span**: the store loop allocates inside the per-group
  loop (`manager.py:711`, `:757`), so a 3,200-token prefix costs ONE SLOT PER
  prefix-cacheable group (five on this box - group 4, the QSA scratch ring, is
  skipped by `manager.py:838` and parks nothing).
* therefore `index_entries x hash_block_size` counts INDEX KEYS at 40-token
  granularity rather than spans and under-counts by `block_size / hash_block_size`
  (80x); `total_blocks x block_size` ignores the per-group multiplication
  (over-counts by the group count); and `GiB / summed page sizes` assumes one slot
  holds every group at once, which the per-group allocation refutes.
* `joint_parked_tokens_max` is the `min` over the prefix-cacheable groups of
  `spans x group_block_sizes[g]` - ONE conversation's retrievable depth;
  `aggregate_parked_group_tokens` is the total volume across conversations. On the
  measured box (1,331 slots, 1,330 parked, spans `[3200 x5, 8]`, per-group parked
  `[259,259,260,260,-,292]`): ~828,800 tokens of joint depth right now and a
  one-conversation ceiling (`per_conversation_ceiling_tokens`) of ~850,000 tokens.
  A 200k-token prefix costs ~313 of 1,331 slots, so the RAM tier is NOT structurally
  incapable of rescuing one.
* **`free` is not `empty`.** `BlockPool.free_blocks` appends HASHED blocks back into
  the free queue as reuse candidates (`block_pool.py:796-798`), so
  `cpu_pool.free_blocks == parked spans` is the FULL-tier signature. The retrievable
  number is `cached_idle` (+ `referenced_hashed`, pinned mid-restore but still
  parked), and `free_uncached` is what is genuinely empty.

## The free-order witness: a hypothesis, made falsifiable - 0.2.6 (inherited from the preceding entry)

`kv_cache_coordinator.py:413` frees the GPU path's blocks with `reversed(...)`, and
`FreeKVCacheBlockQueue`'s own docstring (`kv_cache_utils.py:257-262`) says the tail
of a chain must sit at the FRONT of the queue and that the reversing is done
OUTSIDE that class. `v1/simple_kv_offload/` contains **zero** `reversed()` calls,
and its four `cpu_block_pool.free_blocks(...)` sites (`manager.py:1113`, `:1121`,
`:1272`, `:1297`) pass chain order. `free_blocks` appends hashed blocks in the order
given and `get_new_blocks` pops the front - so chain order in would mean the parked
ROOT is queued before its own tail, and since a missing root makes every later block
unhit-able, the whole tail would be worthless. `op=order` MEASURES it: a
park-sequence number per CPU block id (one int array the size of the pool), then per
`free_blocks()` call the up/down tally of consecutive pairs, published as
`CONFIRMED`, `REFUTED`, or `INCONCLUSIVE` (blocks parked before arming carry no
sequence, so arm first). A confirming trace is `root_first > root_last` with
`asc_pairs` ~ the chain length per group and `desc_pairs` ~ the number of group
joins. What it does NOT claim is in the payload too: it measures the ORDER blocks
enter the free queue, not whether allocation pressure then pops that root before a
restore needs it.

## The posture is self-evident, and a closed gate is counted - 0.2.21 (text)

**A marked request is self-evident, and a closed gate is counted (both 0.2.21).**
**The posture is self-evident, and a closed gate is counted.**
`restore.disk_authority`, `restore.offers_enabled` and `restore.serving_as` are
published beside `restore.authority_lookups` and `restore.lookups`, so "is this boot
serving everyone from SSD, and did the rule actually fire?" is one payload read with no
control experiment: `lookups` moves for every lookup, `authority_lookups` only for one
that ran while disk authority was up. And a request turned away by the
**gate** now moves its own bucket - `refused.not_configured`, `refused.disarmed`,
`refused.tier_disarmed`, `refused.offers_disabled` - in the same region that counts
`workers_not_servicing` and `no_worker_proof`. Before, a gate decline wrote `last_reason`
and no counter, so an empty `refused` could not be told apart from "nothing refused us":
that ambiguity is what made an alignment-guard refusal look like the answer on
2026-10-05 when the gate had refused the request. `last_reason` is written exactly as
before (something may parse it), and the value handed back to the scheduler is the fork's
own tuple, unchanged - the change is bookkeeping, not behaviour.

## Disk authority is what makes the read path testable at all - 0.2.22 (text)

**Disk authority is what makes the read path testable at all (0.2.22 as a marker rule,
0.2.31 as a boot switch).** With RAM consulted, an ordinary small
prompt either hit RAM completely (nothing left to offer, `disk_run_below_one_block`) or
hit it with a remainder (`ram_hit_not_block_aligned`) - so exercising one disk read
meant making GPU and RAM forget the prefix first, ~1.3M tokens of filler per attempt.
Under disk authority a request needs only that the span is in the index, which is why
the operator's one-command way to exercise the read path is now the authority op instead
of a marker on every probe. The posture is also what a whole-workload SSD measurement
needs, and it is now one decision an operator makes rather than one every client has to
cooperate with.

Three limits, stated because they are the ones an operator will hit:

* **Authority does not zero `num_computed_tokens`.** That number is the third field of
  the fork's pending tuple and `update_state_after_alloc` (`manager.py:441-463`) uses it
  as the *offset* into the request's whole block list - it is where the CPU-to-GPU copy
  writes. Zeroing it while the scheduler still counts a cached local prefix would DMA
  disk bytes into the request's already-cached, ref-counted GPU prefix blocks (shared
  with every other request reading them) and leave the newly allocated external region
  holding whatever was in those slots. So a request whose prefix the GPU already
  computed has nothing to restore and recomputes; the disk span starts where the local
  prefix ends, always.
* **Ignoring the RAM hit means releasing the pin it came with.**
  `get_num_new_matched_tokens` (`manager.py:360-372`) `touch()`es every CPU block it
  found and leaves the tuple in `_pending_cpu_hits`; a disk-authority offer overwrites
  that entry, so `disk_restore._release_fork_pin` calls the fork's own
  `_free_pending_cpu_hit` at the commit point - after the write, so every earlier
  failure path still leaves the fork's tuple intact. A manager without that method is
  refused by name (`refused.pin_release_unavailable`) rather than allowed to leak a
  pinned span per request. `pin_release_failures` must stay 0.
* **No token is claimed twice.** The offer's token count is the disk's alone (never
  `ram + disk`), and the merged tuple the fork consumes contains only our freshly
  allocated slots - no RAM block ids appear in it. The scheduler recomputes whatever
  neither tier covered.

Read it in the payload: `restore.last_offer` carries
`ram_tokens_reported` (what the fork found) beside `ram_tokens_used` (zero, while disk
authority is up), `start_block`, `blocks`, `disk_tokens`; `restore.authority_lookups`
counts the
lookups the rule fired on; `restore.local_hit_ignored` and
`restore.local_hit_tokens_ignored` count only OFFERS - a refusal leaves the fork's hit
exactly where the fork put it, and must not be counted as a sacrifice the request never
made. A forced read also logs one `DISK[restore] serving ... from DISK against a
N-token host-RAM hit` warning; a request the disk cannot serve stays silent and
leaves no allocation behind.

## The index now reads the newest spill lines first (kept here as history) - 0.2.22 (text)

**The index now reads the newest spill lines first, and publishes how far behind it is
(0.2.22; the rule below was replaced by the 0.2.28 drain - kept here as history).** `refresh()` used to
advance a per-journal cursor from byte 0, sharing one
512 KiB / 8,192-line budget across every rank's journal. On four ~60k-line journals that
covered about a tenth of one rank per pass, so the lines the restore path was waiting
for - spills from the last few minutes, which is what "the GPU forgot this and RAM never
had it" means - were the last thing it would reach, and it was reached only by a request
that was itself waiting on the answer. Measured consequence: **0 of the 36 newest
identities in the store were in the index**, while `index.entries` sat at 6,892 claiming
files the ceiling had long since reclaimed.
The rule now: a refresh spends at most the same caps, but takes its window from the END
of each journal's unread run, and every journal gets a *share* of the budget rather than
whatever the previous rank left. Whatever that forfeits is older than everything it
applies - which is what makes forfeiting safe: a `+` we read is a file we saw written, a
`-` we read for an identity we never learned is a no-op, and a jump forgets the entries
its skipped region taught it (`_forget_rank`), so the index claims only what it read.
Nothing was forfeited silently: `index.lag_bytes` (unread journal bytes; 0 means every
cursor is at end-of-file), `index.lag_journals`, `index.catch_up_jumps`,
`index.bytes_skipped` and `index.lines_skipped` are published, a refresh skipped by the
1 s floor re-publishes the last *measured* lag with `lag_measured: false` rather than
claiming zero, and `restore.why_no_restore` names the lag when `no_disk_run` is the
answer. While `lag_bytes` is non-zero, a `not_present` from `op=contains` is a statement
about the reader and not about the store - and `op=contains` now carries those fields in
its own `index` view for exactly that reason.

## The restore race, measured, and the wait that stopped guessing - 0.2.23 (text)

**The restore race, measured, and the wait that stopped guessing (0.2.23, flat since
0.2.32).** Reads are queued at `bind_connector_metadata` and applied at
`start_load_kv`, and the fork calls **both inside one engine step**
(`kv_connector_model_runner_mixin.py:80-93`). Because every disk hit is offered with
`load_kv_async=True`, `has_sync_kv_loads` stays false and the fork defers
`start_load_kv` to after the forward **launch** - so the window a read has to win is one
CPU-side forward launch, not a forward pass. Until now the only evidence about that
window was one counter (`dma_unsettled_entries`) plus a `read_seconds_max` of `0.0` that
reads as "no read ever ran" and actually means "no read that reached the kernel ever
finished" - a queued job's buffer is handed back before the reader thread dequeues it.

The apply point now publishes the window itself. `restore.apply_window.last` is the most
recent batch as the engine saw it: `elapsed_ms`, `accepted`/`expected`, `running`,
`queued`, `settled`, the **measured bind→apply gap** (`bind_to_apply_ms`, stamped at the
first line of `_worker_bind`), `bytes_asked` beside `bytes_at_stake` (the bytes still in
flight, which is what a wait is waiting for), `budget_ms`, `waited_ms`, `bound`,
`settled_after_wait`, and `allowance_spent` (did this batch use the whole grant and still
have bytes in flight - the field that replaced the arithmetic-based `over_recompute`,
which could not see a dropped load). `windows` carries
`min`/`max` **boot-wide**, an `avg`, and a `median` over the last `MAX_SETTLE_SAMPLES`
(512) observations that publishes `median_samples` so a truncated median can never be
misread as the boot's. `read_seconds_avg`/`read_seconds_max` sit beside
`waited_ms` on purpose: either number alone has no unit of comparison.

A drop now says **which kind** it is. `dropped_queued` counts entries whose read had not
started (the job sat in the queue and the engine got there first - thread scheduling on
the proxy host, and the kind a wait fixes); `dropped_running` counts entries whose read
was inside the kernel and still lost (a device slower than the step, which a wait only
fixes if the budget beats the read); an unreadable per-job state table is
`dropped_unattributed` and never folded into either. `applied_settled_immediately`
counts the entries the box already wins for free, so `settled_after_wait` over it is the
wait's hit rate.

The wait itself is **flat, bounded, and an allowance rather than an arm** - an unbounded
wait inside an engine step is a hang wearing a performance patch. The rule, as it reads
in `disk_restore.settle_budget`:

```
granted_ms = min(VLLM_E2_TIER_DISK_SETTLE_MS, the batch's own seconds_left)
             # the read's expiry, from LOAD_DEADLINE_SECONDS: the same clock that
             # cancels the read, so a wait can never outlive the bytes it waits for
```

That is the whole formula - one number the operator sets (`SETTLE_MS_DEFAULT`, **5,000
ms**, overridable by `VLLM_E2_TIER_DISK_SETTLE_MS`) and the load's own deadline, the only
other clock with a claim on the wait. `SETTLE_POLL_SECONDS = 0.0005` is a poll and never
a condition wait (the reader thread owns those bytes and the engine thread must not queue
behind its lock), `SETTLE_MS_MAX = 60000` clamps a typo with `ceiling_clamped` saying it
happened, and `0` is still reachable and still means never wait.

**Every term of the old formula was a guess, and the live box said so.** Through 0.2.31
the allowance was `clamp(bytes_in_flight / read_bytes_per_s * 1000 * safety_margin,
SETTLE_MIN_MS, cap)` - an assumed read throughput (it was the *write* rate, because the
read side had never been measured), an assumed safety margin, a floor of one scheduler
tick, and an assumed prefill rate used only to publish the cost a wait was racing. On
2026-10-06 a live request with a 28,583-token prompt had 12 jobs / 137.6 MB offered and
the formula granted **23.6 ms**; the measured bind→apply race gap for the batches it
lost was **89.7 / 97.1 / 129.7 / 135.8 ms**; the request recomputed **28,583 tokens
(12.26 s, `cached_tokens: 0`)** to save 23.6 ms, and the loss was `drop_running` - a
kernel read mid-flight, bytes perfectly good underneath it. A budget that gambles
against a 12-second recompute has no winning side, so the arithmetic is deleted:
`SETTLE_MIN_MS`, `SETTLE_SAFETY_MARGIN`, `SETTLE_READ_BPS_DEFAULT`,
`PREFILL_TOKENS_PER_S_DEFAULT`, `VLLM_E2_TIER_DISK_SETTLE_READ_BYTES_PER_S` and
`VLLM_E2_TIER_DISK_PREFILL_TOKENS_PER_S` are gone, and `bytes_at_stake` is PUBLISHED
beside the grant while deciding nothing - it is the quantity a reader of
`last_apply_window` wants, not an input. Which bound applied is published per batch as
`wait.bound` in exactly three values - `flat` (the allowance decided it), `deadline` (the
read's own expiry did, so a drop here is a deadline problem and not an allowance
problem), `off` (this boot does not wait) - and the three are counted boot-wide.

**The counter that could not see the loss is replaced by one that can.**
`budget_over_recompute` compared a grant with the arithmetic that produced it, so the
23.6 ms grant above reported `over_recompute: 0` the whole time it was costing twelve
seconds. `wait.allowance_spent` asks the only question a flat allowance can answer - did
this batch consume the WHOLE granted allowance and still have bytes in flight? - and
`settle_allowance_spent` is its boot-wide count. It is the field that says "5 s is not
enough on this box" without anyone measuring a throughput.

**The default is ON, which is a behaviour change on an unchanged command line.** Before
0.2.32 an unset variable meant never wait; now it means a 5 s ceiling, so a boot started
the old way will wait where it used to drop. The bound is the load's deadline, not the
cap, so a wait can never outlive the read - but a step that used to cost 0 ms of waiting
can now cost up to the race gap, which is exactly the trade the operator asked for ("give
it up to 5 s to read; only when something bad happens do we ignore"). `0` puts the old
boot back. An unset `VLLM_E2_TIER_DISK_SETTLE_MS` therefore announces what it settled on
once per boot (`DISK[settle] wait=ON allowance_ms=5000 (VLLM_E2_TIER_DISK_SETTLE_MS is
unset; the default since 0.2.32 is 5000 ms - set it to 0 for the pre-0.2.32
never-wait boot)`) - the line NAMING the ignored variable is mandatory, because "the
operator wrote 40 and the box waited 5 s" must never be a silent reading. A value that
will not parse falls back to the default and says so, naming the variable and the value,
once per boot, in `env_problems` as well as the log.

**What the counters exposed on the way.** Every failure on the live box arrived as
`invalid_block_ids` with an EMPTY `failed_recving`, and that path retired its ledger
entry with a hand-rolled `_loads.pop()` instead of `_drop_load` - so `loads_released`
never moved (the honest-looking reading was "the load was never registered" rather than
"it was registered and burned") and the load's slot→path mappings outlived it, which is
a slot the next `build_load_paths` will emit a path for while the fork may already be
using it as a host-RAM source. Both failure shapes now close the ledger through
`_drop_load` (or `_burn_load`, when the loss is a data failure and the request is fused
- see *What a failed load costs* below), and `no_read` deliberately does **not** move at
burn time (the load is already counted as a failure - moving both would double-book one
release). A `finished_recving` that arrives *after* a burn is counted as `no_read`
rather than vanishing, because it is the fork calling a load finished whose bytes this
plugin knows it never delivered.

## The ceiling walk, the sample, and the store that grew past its ceiling (0.2.25)

The box measured this on a live boot: `du -sxb /home/user/kv-tier/disk` = **334.9 GiB**
against `VLLM_E2_TIER_DISK_BYTES` = 206,158,430,208 (**192 GiB**), **29,532 files**, and
the oldest surviving file **2.8 hours** old. Eviction was running, and the store was
still growing ~145 GiB an hour into 122 GiB of free space. Both halves were true at
once, because one bounded walk was being read as a measurement.

**What the walk was, and what it silently was not.** `_walk_store` stats at most
`MAX_WALK_FILES` (20,000) files, so on a bigger store it measures a *sample*. The old
walk took that sample depth-first - its file counter was never reset per shard, and its
own `truncated` flag broke the shard loop - so it measured the FIRST shard directories
and stopped. Two failures came out of that one shape, and they point the same way: the
byte total came from ~2/3 of the key space, so the trigger compared a floor against the
ceiling; and a file the walk never opened could not enter `lru_eviction_plan`, so the
`~1/3` of the key space it never reached became a sink that spilled into forever and
could never be shed. Measured on that shape: the old decision number was **218.7 GiB**
for a **326.2 GiB** store, which makes the *effective* ceiling **286 GiB** - and it
rises with the store, because the sampled fraction keeps shrinking.

**The fix is two bounded passes, and a sample that is honest about being a sample.**

1. `_count_store` counts the capacity-visible files with readdir only - no `stat` - so
   the walk knows the size of the population it is about to sample. Bounded by
   `MAX_WALK_ENTRIES` directory entries (100,000, five times the `stat` cap, per pass),
   and bounded in ENTRIES rather than files on purpose: the `stat` cap counted measured
   files, so before this the *iteration itself* was unbounded, and a directory of
   entries this tier will not measure walked to the end inside the spill writer's
   enforcement path. Cost, measured on the shape above: **29,532 entries in 0.099 s**,
   zero `stat` calls.
2. The measuring pass then stats a **proportional** sample: shard `i` is asked for
   `cap_files * files_i / total_files` of its files, so every shard is sampled at the
   same fraction of itself and the whole budget is spent however the files are spread.
   Proportional rather than even-per-shard because an even split wastes the budget on
   the small shards. The plan built from that sample can name a file in any shard, and
   enforcement runs after every committed write, so a partial pass is a pass and not a
   verdict: on the shape above the estimate lands **326.2 GiB against a 326.2 GiB
   truth (0.00% error)**, and the measured catch-down of the real store is several
   passes, each capped by `disk_layout.MAX_EVICTION_PLAN` = 4,096 files - see the table
   below.
3. `_enforce_ceiling` decides on `estimated_total_bytes` - the sample scaled by
   `total_files / files` - and never on the floor. A truncated walk that stat-ed
   *nothing* (`sample_unmeasured`) is declined outright: the estimate is kept, nothing
   is planned, and the reason says so, because a 0 read as "the store is empty" is the
   most dangerous number this module can emit.

**A store inside the caps is measured, not estimated, and byte-identically to before.**
When the walk sees every file, `estimated_total_bytes` IS `used_bytes` - the same
number, the same entries, the same eviction order. `test_a_store_inside_the_caps_is_
measured_and_not_scaled` and `test_the_eviction_order_of_a_measured_store_is_the_oldest_
first_order_it_always_was` pin that, because this tier is live and an eviction order
that moved for a 19,000-file store would not be an acceptable fix.

**The witness.** `spill.ceiling_measurement` (and the same fields in
`disk_spill.counters()` as `walk_*`) publishes `basis`
(`full_measurement` | `sample_scaled` | `sample_scaled_count_extrapolated` |
`sample_unmeasured`), `measured_bytes`, `sampled_files`, `total_files`,
`entries_examined`, `skipped_files`, `walk_truncated`, `count_truncated`,
`plan_truncated` and `ceiling_checks`. "The store is 170 GiB" and "I stat-ed 20,000 of
29,532 files and extrapolated 170 GiB" are now two different payloads. `skipped_files`
also finally keeps a promise the old walk made in a comment and never kept: files this
tier can neither measure nor shed - a `.part` left by a killed spill, a malformed name,
someone else's file - are counted instead of being invisible, which matters because
they are real bytes that the ceiling will never reclaim.

**The ceiling had to be woken up first.** `estimated_bytes` is a **per-process** counter
of this process's own committed writes and starts at 0, and the walk that could
correct it only ran once that counter *alone* reached the ceiling. A rank booting into
a store somebody else filled was therefore blind to it in proportion to how full it
already was: on this box a fresh boot had to spill **192 GiB of new files (17,384 of
them, ~48 minutes at the measured ~4 GiB/min)** before it was allowed to *look* at a
335 GiB store with 60 GiB of `/home` left. The filesystem fills long before the
ceiling wakes up, and honest scaling inside the walk buys nothing when the walk never
runs. So the first committed spill of a `(root, ceiling)` pair takes **one** bounded
walk and adopts its answer (`_maybe_seed_estimate`, writer thread only, 0.22 s once
per configuration, published as `ceiling_measurement.seed_walks`); a store that cannot
be read leaves the running estimate alone instead of writing a 0 into it, and is
attempted once, not per spill. With that, the recovery the ceiling exists to perform
actually happens: seed, then shed toward the target.

**Convergence, measured by replaying the live store.** By the time the file list was
taken the store had grown to **34,896 files / 395.8 GiB** (256 shard dirs, mean slot
12,178,780 B). Replaying that exact list - names and per-file sizes, rebuilt sparse in
a scratch directory - through the real `_maybe_seed_estimate` / `_walk_store` /
`_enforce_ceiling` / `lru_eviction_plan` / `execute_unlink_plan(dry_run=False)` at the
live 192 GiB ceiling:

| | old code | 0.2.25 |
|---|---|---|
| decision number | 225.4 GiB (20,000 files, DFS floor) | **395.8 GiB** (+0.01% vs truth) |
| ceiling actually enforced | 337 GiB | 192 GiB |
| passes to get under it | never (each pass was aimed at 202 GiB from a floor) | **5** |
| freed | - | **19,292 files / 223.0 GiB** |
| final store | - | 172.8 GiB = the `0.9 x` target, on the nose |
| `/home` free | 61 GiB | **284 GiB** |

Each pass sheds at most `MAX_EVICTION_PLAN` = 4,096 files (~47 GiB here) and cost
**0.456 s of writer-thread time** measured end to end (a 0.22 s walk plus the unlinks);
the whole 5-pass catch-down was **2.28 s**. Passes are triggered by committed writes, so
the catch-down needs 5 spills, and the steady state is one pass per ~1,700 files (~19 GiB)
- one walk every ~5 minutes at the measured ~4 GiB/min, about 0.1% of the writer thread.
One pass sheds ~11 minutes of growth: out-running the enforcement would take more than
4,096 files/minute - about 47 GiB/min. Measured on that filesystem with the store's own
layout: 4,096 `unlink()` = 0.050 s (12 us each); `disk_index` appends the removal line
without `fsync` (~4 us), so the unlinks are not the cost. Had the journal fsync'd per
line it would have been 1.4 ms x 4,096 = 5.6 s per pass - it does not, and that is
load-bearing for convergence, so it is stated rather than assumed. During the
catch-down the writer thread pays that cost per spill, which throttles the spill path
itself while the store is over the line; that self-throttling is the point, and it holds
at any spill rate below ~47 GiB/min.

**And the ceiling only runs while the tier is armed.** Enforcement is called from the
writer thread after a committed write, so a disarmed tier with a full store reclaims
nothing - correct by design (the ceiling is a property of the spill path, not a janitor),
but it means "`/home` is 99% full" and "spill is disarmed" together leave the 396 GiB
exactly where it is until the tier is armed again. At the measured pass cost, re-arming
reclaims 223 GiB in about 2.3 s of writer-thread time.

Tests: `tests/test_disk_ceiling_walk.py`, built on a fixture store whose bytes sit in
the LAST shard directories in `os.scandir` order - the exact shape the old walk was
blind to - plus a fresh-boot fixture whose own writes are one slot against a full
store. Every one of its 29 tests fails against the pre-fix code.

## The claim must match the bytes: a hybrid model gets no disk offer at all (0.2.25)

**A disk offer claims every prefix-cacheable group, so it must be able to restore every
one of them.** This is a correctness fix found on live traffic, not a hardening
exercise: at 00:08 on 2026-10-06 a disk-authority request on a 31,826-token prompt got a
mechanically perfect disk read - `reads_started=9 reads_applied=9 failed={}` per rank,
18 ms, settled on the first settle check - and answered `duct Register Register
Register...` at `temperature=0`, where the same prompt recomputed from scratch returned
the planted code word. Nothing failed on the way. The answer was wrong anyway.

The model is `Qwen3.8-Flash-Next-W4A16-Merlin`: groups 0-3 are `MambaSpec`/GDN under
`MambaManager` in `align` mode (7/6/6/1 layers), group 4 is a QSA ring at `block_size`
8, group 5 is the full-attention group (12 MLA layers), and the RAM tier's own
`prefix_cacheable_group_ids` is `[0,1,2,3,5]`. The restore path had narrowed its
eligible set to *prefix-cacheable AND whole-scheduler-block AND
`has_positionally_stable_blocks`*, which on this model is `{5}` - because
`MambaManager.has_positionally_stable_blocks` is False in align mode
(`vllm/v1/core/single_type_kv_cache_manager.py:1494`), and the fork's own spill path
skips those groups (`vllm/v1/simple_kv_offload/manager.py:874-876`). Then:

* the number that set produces is what `get_num_new_matched_tokens` returns
  (`manager.py:372`), and that return advances `num_computed_tokens` for the **whole
  model** - the engine skipped prefill of the GDN and QSA layers across 28,800 tokens;
* `update_state_after_alloc` builds load entries only for groups that have non-null CPU
  blocks, and its `n_ext_g == 0 -> continue` (`manager.py:445-470`) is a **silent
  no-op** - so groups 0-3 were claimed and never loaded, no group reported a failure,
  nothing came back through `get_block_ids_with_load_errors()`, and nothing recomputed
  the claimed range;
* what a claimed span actually needs from an align-mode GDN group is **one** block, not
  a span of KV: `MambaManager.find_longest_cache_hit`
  (`single_type_kv_cache_manager.py:1533-1610`) answers a hit with
  `[null_block] * (n - 1)` plus one real boundary-state block.

  **The sentence 0.2.25 concluded from that was wrong, and 0.2.27 is the correction.**
  What I wrote next was "that one block has no file name to read back", reasoning from
  the recurrent state being one running value per sequence. The fork disagrees in three
  places: the align hand-off names that block and hands it to the connector
  (`single_type_kv_cache_manager.py:2051-2080`, contract at `kv_cache_manager.py:907-921`),
  the tier stores it under exactly the hash a lookup will ask for (`manager.py:1050-1078`),
  and the store on the box proves it - one shard directory holds
  `kvslot-a36ec46fe316bd0d-g0-…`, `-g1-…`, `-g2-…`, `-g3-…` and `-g5-…` files for the
  SAME key16, five groups at one token boundary. `has_positionally_stable_blocks` is
  about whether offload *scans* may follow a block table (`:62-69`, `:1494-1497`: align
  mode "can null interior states and relocate speculative blocks in place"), not about
  whether a block is *named*. Nulling a block does not un-hash it.

So positional stability is a **necessary** condition (a name to read by) that was being
read as a **sufficient** one (a complete claim). The module already had the right rule
in the right place - `_disk_run`'s docstring: *"A block is restorable only when EVERY
eligible group has a file for it: the fork pairs per group, so one missing group's slot
would be a load entry with no bytes behind it - a whole-request recompute for a
partially-restored span."* The bug was never the rule. It was the **set** the rule was
applied to: narrow the set to `{5}` first and the safeguard is vacuous. The gate is that
same invariant applied to the set the **claim** covers.

**The gate: an incomplete group set claims nothing.** If any prefix-cacheable group
cannot be restored, the disk refuses the offer under its own countable name,
`refused.incomplete_group_set`, with the reason naming the blocking groups and the
sentence beginning with the verdict (because `last_refusal` is clipped at 160
characters). On this model the disk tier offers nothing, forever - that is the correct
behaviour, and it is stated as one rather than hedged. The predicate is derived entirely
from the managers (`prefix_cacheable_group_ids`, `group_block_sizes` against the
scheduler block size, `has_positionally_stable_blocks` per group), so a pure-attention
model has nothing blocked and is offered at **exactly as before**
(`test_a_pure_attention_configuration_still_offers_normally`).

**Superseded in 0.2.27, in its diagnosis only.** The sentences here promising "on this
model the disk tier offers nothing, forever", and the one saying the GDN groups have no
name to read by, were wrong about the fork; the gate now opens on this geometry because
the boundary-state contract is priced per run. What did NOT change is the invariant the
gate exists for - an offer must have bytes for every prefix-cacheable group - and a
group with no implementable contract still refuses under this same name. The next
section is what replaced them.

**Why this is a refusal and not a clamp - do not "optimise" it.** The tempting change is
to offer the intersection, the way the joint CPU lookup clamps. It is the same nothing,
twice paid. The live lookup ledger already shows the RAM path clamping on exactly these
groups: over the last 256 external lookups, 190 returned a hit and **4 were adopted**,
the typical record `local=134400 hit=1000 adopted=0` attributed
`partial_hit_clamp_suspected` - "a state group clamped at a length this lookup cannot
see". A clamped disk claim would be 1-3k tokens (about 0.3 s of prefill saved against
~20 ms of read and the whole claim risk) and would still be a claim whose boundary state
nobody produced. Refusing is the disk path agreeing with a rule the RAM path already
enforces, and it costs production almost literally nothing.

**Read it in the payload**, in the same section that carries `last_offer`:

* `group_serviceability` - the whole arithmetic of the last lookup: the cacheable set,
  the positionally stable set, the restorable set, `blocked_group_ids` with a
  `blocked_why` per group, and `complete`. `complete: false` means this model can never
  be served from disk, and an operator learns that from the payload without knowing what
  a `MambaSpec` is.
* `last_groups_required` - what a claim of `claimed_blocks` costs **each** cacheable
  group: `blocks_needed`, `null_blocks`, the `contract` it came from, and `restored`.
  The 00:08 claim priced this way reads "9 blocks needed in g5, 1 boundary-state block
  needed in each of g0-g3, 0 of those 4 filled", which is the whole incident in one
  object. It is published on the **refusal** door too (`offered: false`), because a
  requirement is a fact about the claim, not about whether it happened. The same rows
  ride along inside `last_offer.groups_required` on an offer that does happen.
* `why_no_restore` now says `THIS MODEL CANNOT BE SERVED FROM DISK AT ALL` with the
  groups that decided it, instead of leaving a reader at "no offer has been made, read
  `refused`".

**A boot that was not turned on touches nothing.** Both boot switches are read before
the gate's expensive work, so a boot with neither up never reaches the predicate at all
- no arithmetic computed, no index query, no journal refresh, no counter moved beyond
the one refusal bucket that names both ops, and the fork's own answer returned
identically (`test_a_boot_with_the_switches_down_touches_no_index_and_no_journal`).
Since 0.2.31 the population this covers is "a boot nobody switched on" rather than "a
request that did not carry the marker", and the guarantee is the same one the
2026-10-05 incident asked for.

Tests: `tests/test_disk_restore_group_gate.py` (the hybrid refusal, the pure-attention
control, the predicate-not-a-group-list proof, the requirement table on both doors) and
`tests/test_disk_restore_ledger_reporting.py` (the four reporting faults below).

## The ledger's own honesty: aged-out loads, DMA-verified blocks, and a witness kind that could not lie about the present (0.2.25)


The same 00:08 request had a second problem: its payload could not say what happened to
it. Four reporting faults, all fixed, none of them a change to a protective rule:

* **A load that aged out erased its own evidence.** `restorations` stayed 0 while 36
  reads were applied, because the ledger entry was released - by the fork's own *"no
  external tokens but `get_num_new_matched_tokens()` unexpectedly recorded a pending CPU
  hit; releasing the stale pin"* path, which the live engine logs in ~10% of its lines,
  or by the deadline - **before** `finished_recving` arrived. The report loop then found
  `_loads.get(request_id) is None` and `continue`d. Every report-less release now leaves
  a tombstone (`last_aged_out` names the door), and a late report is matched to it:
  `late_finished_reports`, `aged_out_with_applied_reads`, `restored_after_release`.
  `restorations` keeps its exact meaning - *reported while the ledger held it* - so
  nothing is inflated; the truth just stops being dropped. **The deadline did not move:**
  a read completes in 9-30 ms (`read_seconds_max` 0.018) against 45 s, about fifteen
  hundred times the read it guards, so the expiry was never a slow disk and a longer
  deadline would only slow the fuse that stops the re-offer loop of 2026-10-05.
* **Counters that existed and were never published.** `watchdog_fires`, `no_read`,
  `invalid_reports` and the aged-out family are now in the scheduler section with units,
  which is what turns "the release branch must have fired" from an inference into a
  reading - and distinguishes the fork's release door (`watchdog_fires: 0`) from our
  deadline door (`watchdog_fires: 1`). `invalid_reports` is drained on the **worker**
  side, so a scheduler's 0 is correct and its units entry says so.
* **The field that answers "were the GPU blocks actually written".**
  `disk_gpu_blocks_dma_verified` counts GPU blocks this module mapped for a disk span
  (their CPU slots carried our emitted paths - the one moment the scheduler sees our slot
  paired with the fork's destination block) that the fork then reported DMA-complete, in
  **blocks**, not requests. An offer cannot move it; only an adopted load reported
  finished can. A step whose `load_cpu_blocks` and `load_gpu_blocks` disagree maps
  nothing, because the worker's own bind would refuse that step.
* **`witness_kind` was a past sentence read as a present claim.** It was published as
  `offers_not_adopted` while `adopted=1`: emissions dedupe on `(verdict, kind)`, so a
  kind spoken before `_mark_adopted` stayed on the page after it. The verdict (`REFUTED`)
  was right; the kind was true at emission and false at read time. Both readings are
  published now - `witness_kind` (as of the last emission, unchanged) and
  `witness_kind_now` / `witness_verdict_now` (the same arithmetic on the current
  counters), plus `witness_kind_is_stale` when they disagree. `witness_now()` writes
  nothing, so polling cannot rewrite the record the log dedupes on.

Tests: `tests/test_disk_restore_ledger_reporting.py`.

**One release, two authors.** 0.2.25 also carries the ceiling-walk fix above; both
landed in the same uncommitted working tree (HEAD is 0.2.21 and 0.2.22-0.2.24 were never
committed), and neither change touches the other's files.

## The door that showed zeros: a cross-process rollup of the ranks' spill counters (0.2.26)

The reclaim was real and the payload said nothing happened. On the night of 2026-10-06
the store went **395.8 GiB / 34,896 files -> 161.5 GiB / 14,233 files** (234 GiB shed,
`/home` free 61 -> 293 GiB), and `POST /plugins/kvtier/disk {"op":"status"}` answered:

```json
{"spill": {"evictions": 0, "evicted_bytes": 0, "estimated_bytes": 0,
           "ceiling_checks": 0, "estimate_seed_walks": 0, "walk_basis": null,
           "walk_sampled_files": 0, "thread_alive": false}}
```

The truth was already on disk, in `/tmp/kv-tier-diagnostics/rank-PP0_TP0.json`:
`disk.evictions: 15237`, `disk.spill_queued: 79`, `disk.spill_errors: 0`,
`disk.ceiling_bytes: 206158430208`. Nothing was miscounted and nothing was lost.
`disk_spill._state` is **per process**; the spill writer thread and
`disk_spill._enforce_ceiling()` run in the processes that commit writes - the WORKERS,
four of them here (`PP0_TP0`, `PP0_TP1`, `PP1_TP0`, `PP1_TP1`) plus the scheduler - and
the `/disk` door is answered by whichever process the utility call landed on. That
process ran no walk, so every counter it published was honestly zero, and 0.2.25's
whitelist of `ceiling_checks` / `walk_*` - correct as far as it went - was unreadable in
practice because the publishing process was the wrong one. **A witness nobody can read
is not a witness.**

**What 0.2.26 adds: `spill_rollup`, a sibling of `spill`** (`disk_store.rank_rollup()`,
published by `op:status`). It reads the per-rank files `worker_diag` already writes in
`VLLM_E2_TIER_DIAG` and sums what each rank's own process counted. Read-only, no shared
memory, no new IPC channel, no new file name, no new thread, no new env var, no new
write. `spill` keeps its keys and its meanings exactly as they were (pinned by a
byte-identical test), because this adds a view and changes no policy.

```json
{"spill": {"evictions": 0, "thread_alive": false},
 "spill_rollup": {
   "basis": "rank_files_fleet_unknown",
   "totals": {"evictions": 30760, "spilled": 24474, "spill_queued": 143,
              "spill_errors": 2, "bytes_written": 337686041600,
              "queue_depth": 0},
   "totals_reported_by": {"evictions": 4, "spilled": 4, "queue_depth": 4},
   "totals_are_lower_bound": true,
   "ranks_counted": ["PP0_TP0", "PP0_TP1", "PP1_TP0", "PP1_TP1"],
   "ranks_expected": null,
   "ranks_expected_source": "no capacity handshake has been recorded in this process",
   "ranks_missing": null,
   "complete": null,
   "per_rank": [{"rank": "PP0_TP0", "age_seconds": 1.4, "evictions": 15237,
                 "spilled": 11841, "queue_depth": 0, "ceiling_bytes": 206158430208,
                 "writer_thread_alive": true, "epoch": "9f2c1a07"}],
   "stale_ranks": [], "excluded_files": [], "excluded_total": 0,
   "ceiling_bytes": {"values": [206158430208], "agree": true, "min": 206158430208,
                     "max": 206158430208, "reported_by": 4},
   "writer_threads": {"alive": 4, "not_alive": 0, "unknown": 0},
   "this_process": {"rank": null, "rank_source": "...not imported here",
                    "counted_in_rollup": null}}}
```

**The totals are a sum of four processes' counters, not one fleet-wide counter**, and
they are labelled as one (`scope`, `kinds`: `evictions` is a `counter-since-boot`,
`queue_depth` is a `gauge`). There is deliberately no attempt at shared memory or a new
IPC channel: reading files the ranks already write is the same channel `disk_restore`'
`worker_proof()` already depends on, and the alternative is far riskier than the gap.

**`basis` says which files the numbers came from**, in the same spirit as
`ceiling_measurement.basis` above: `rank_files` (every rank the boot has, fresh),
`rank_files_partial` (something missing, stale, unreadable or over the read cap),
`rank_files_fleet_unknown` (every file that exists was fresh and counted, but this
process does not know the rank set), `rank_files_unusable`, `no_rank_files`,
`no_diagnostics_dir`, `directory_unreadable`, `rollup_failed`.

**A rank file that is missing or stale makes the rollup say so, never quietly agree.**
`totals_are_lower_bound` with `lower_bound_reason` naming the ranks; `ranks_missing`
lists an expected rank the directory holds **no file at all** for; `stale_ranks`
publishes a stale rank's last-known counters *outside* the totals, with its
`age_seconds`; `excluded_files` lists a file that exists and is unusable. A rank is
named in **at most one** of the three, so a reader adding them up can neither
double-count a gap nor miss one. A silent low total is the dangerous case: it is
indistinguishable from "the ceiling is holding", which is the exact misreading that cost
the 0.2.25 investigation, re-opened one level up. The freshness bar is
`RANK_ROLLUP_FRESH_SECONDS` (900 s, the same heartbeat bar `disk_restore` uses for its
proof) and a `ts` in the future is treated as *not* fresh, because these files cross a
process boundary.

**`ranks_expected` comes from the capacity handshake** (`cpu_capacity`, the one place
rank ids arrive in this process - the same rule `blockdump._expected_ranks()` uses, and
this calls it rather than writing a second one). When this process has recorded no
handshake, `ranks_expected` and `ranks_missing` are `null` and `complete` is `null`:
"I cannot say" is published as itself, never as an empty set, which would read as
"nothing was missing". And `this_process` says whether THIS process's own `spill`
numbers are *inside* the totals - the difference between "my process did nothing and
four others did" and "you are looking at one of the four", and the reason nobody can add
`spill` to `totals` and get a number wrong by exactly one rank.

**A rank file is untrusted input.** Size-capped, type-checked, per-field bounded, and
every rejection is *named*: `malformed`, `too_large`, `bad_version`, `mislabeled` (its
`rank` disagrees with its file name), `no_ts`, `clock_ahead`, `no_disk_section`, plus
per-field `fields_rejected` for a counter that is negative, non-integer or over
`RANK_ROLLUP_COUNTER_MAX`. A bad file costs that field for that rank only, and the door
still answers `ok: true` - never a 500 because a diagnostic file was truncated.

Costs: `op:status` now opens at most `MAX_RANK_ROLLUP_FILES` (32) files of at most 256
KiB each and writes nothing (pinned). There is no cache: the rollup is the truth about
the last file write, and freshness is the point of it. **This needs an engine restart to
take effect** - it is engine-side code reached through the utility call, so the running
boot keeps answering with the 0.2.25 payload until it is restarted into 0.2.26.

Tests: `tests/test_disk_spill_rank_rollup.py`.

## The SSD tier serves the GDN groups: a second block contract (0.2.27)

**The disk tier now restores a hybrid GDN + full-attention model instead of refusing
it.** The gate of 0.2.25 was correct about the invariant and wrong about the engine, and
this is the change the second half of that sentence asked for. Nothing about the
invariant moved: an offer still requires bytes for **every** prefix-cacheable group, and
a group whose block shape this tier cannot price still refuses under
`refused.incomplete_group_set`. What changed is that an align-mode `MambaManager` group
has a shape, so it is restorable - and the shape is what makes the offer cheap.

**Two contracts, derived per group from the manager itself.** `_group_contract()` reads
`has_positionally_stable_blocks` and `mamba_cache_mode` (from the manager, falling back
to its `kv_cache_spec`) and answers one of `positional`, `boundary_state`, `unstable`,
`geometry`, `outside`, `unknown`. No model name, no group id, no new switch: the same
two attributes the fork's own `has_positionally_stable_blocks` override is computed
from (`single_type_kv_cache_manager.py:1494-1497`), which is why the gate opens **by
construction** rather than by an opt-in.

| contract | a claimed run of `n` spans costs | where that comes from |
|---|---|---|
| `positional` | `n` real blocks, one file each | the fork pairs one CPU slot per group block |
| `boundary_state` | `[null]*(n-1)` plus **one** state block at the run's END | `find_longest_cache_hit` (`:1591-1607`), the align hand-off (`:2051-2080`), the store's own `-g0-` files |
| `unstable` | unknowable (`blocks_needed: null`) | not stable, not an align hand-off: no name to read by |
| `unknown` | unknowable, and refused | the predicate is unreadable - a rename, not a shape |

**A run is anchored, never truncated.** `_disk_run` walks the positional groups to the
first block any of them cannot name, then SHORTENS to the longest length whose end every
`boundary_state` group can name, and refuses with the new `no_boundary_anchor` if no
length qualifies. On the live geometry this is the difference between a right answer and
the 00:08 one: if group 5 holds blocks 1-4 and the GDN state files exist only at 1 and 2,
the offer is **two**. The witness says why in one field - `last_offer.positional_run_blocks`
is the length the attention group could have claimed alone, and `run_reanchored: true`
says the offer is shorter than that on purpose. Every later clamp goes through
`_cut_run`, which may only land on a length `_disk_run` proved anchored: a capacity clamp
that took `keys[:3]` would move the state block to a span no group has a state for, which
is the incident in miniature.

**The list the fork consumes is `[null, null, null, real]`.** `_merge_into_pending` builds
an align-mode group's entries from the CPU pool's own `null_block` - the same object the
fork's Mamba hit uses - and puts the one real state block last, because
`update_state_after_alloc` skips entries whose `is_null` is set **while keeping the
index** (`manager.py:460-478`). That index-preservation is the whole mechanism: build the
list the obvious way (real block first) and the recurrent state is paired with the run's
FIRST GPU block, which is a wrong KV state the model then uses to answer the prompt.
`test_the_state_block_pairs_with_the_run_s_last_gpu_block` asserts the pairing, not the
list, and `no_null_block` is the new refusal for a pool that cannot supply the
placeholder - the alternative is a plausible restore that is off by a whole run.

**Copy-at-completion: the queue carries bytes, not slot numbers.** The spill used to read
the CPU slot on `disk_spill`'s writer thread, and the module documented why that was
safe: no file could be read back, so a slot the RAM tier had recycled between completion
and the write cost a wasted write. Now a file CAN be served, and one wrong align-mode
boundary state answers confidently and wrongly at `temperature=0`. So `request_spill`
copies the slot synchronously, on the thread the completion arrived on, and the job
carries the snapshot; `_handle_job` REFUSES a job that carries a slot number without one,
so the old door cannot be reopened by accident. The cost is one memcpy of a rank slot per
spilled slot, bounded by the new `MAX_QUEUED_BYTES` (256 MiB): the queue is admitted on
two currencies now and the byte cap has its own drop name, `queue_bytes_full`, because
"the queue is 256 deep" and "we would have had to hold another slot" are different
diagnoses. `snapshot_failed` is the other new name - a slot that cannot be read is
reported where the slot still exists, not by the thread half a second later. Both the
race and its fix are tested (`test_the_bytes_that_land_are_the_bytes_the_store_completed_with`,
and `test_deferring_the_read_writes_the_next_block_s_bytes` keeps the old behaviour
provable rather than asserted).

**Task 0: what `block_bytes_by_group` was, settled by the files on disk.** The rank
diagnostics published `[80281600, 58982400, 58982400, 1638400, 82944, 132710400]` while
the same rank's spill files are **11,468,800 B** (12,902,400 on PP1) - so the published
per-block figure for group 0 was 7x the entire slot it described. The publisher computed
`len(group.layer_names) * spec.page_size_bytes`, but on the worker side of a merged
config `page_size_bytes` is already the group total: `MambaSpec.merge()` sums it across
the merged layers and `MambaSpec.page_size_bytes` multiplies by its own `num_speces`,
which the merge just folded in (`kv_cache_interface.py:1151-1160`, `:1135-1143`). The
error therefore differed per group (group 5 is an attention spec, merged by MAX) and per
PP rank - which is how two ranks published different "per-group" figures for one model.
Fixes, all at the publisher: the field is `page_size_bytes` **alone** with
`layer_count_by_group` beside it, `arena.bytes_per_block` publishes the rank's MEASURED
slot width as the cross-check, and the `meaning` strings name the mistake they prevent.
Two consequences for anyone reading the old payloads: the figures describe the merged
group's whole block, so no I/O path is affected (nothing reads that field), and
`MAX_BYTES_PER_READ` (64 MiB) was never the limit - the real unit is under 13 MB. The
arithmetic is pinned by tests, not prose: `test_a_spill_file_is_the_measured_slot_width_not_layers_times_page_size`
measures a written file against the figure, and `test_the_publisher_stops_multiplying_the_layer_count_in`
fails if the multiplication comes back.

**Capacity bookkeeping says what a span costs, per contract.** `disk_spill.section().span_cost_bytes`
publishes `positional_bytes_per_span` and `boundary_state_bytes_per_span` in the bytes the
ceiling is configured in, with the sentence that the two numbers being equal does not mean
the two contracts cost the same per span. A reader who counts spilled files concludes a GDN
group is cheap to keep: true per hit, false as a plan, because the `n-1` spans it did not
buy are spans it did not keep.

**Read it in the payload.** `last_offer` gained `positional_groups`, `boundary_groups`,
`run_blocks`, `positional_run_blocks`, `run_reanchored` and `loads` (per group:
`real_blocks`, `null_blocks`, `contract`); `group_serviceability` gained
`group_contracts`, `boundary_state_group_ids` and per-group `contract` rows;
`last_groups_required.blocks_needed` is `null` for a group with no implementable
contract instead of a guess of 1; `why_no_restore` names the contract vocabulary and says
outright that an align-mode GDN group is NOT in the blocked list.

**What is still refused, and why that is correct.** A `boundary_state` group whose run
end no engine ever spilled (a first-block-only chain, a request that starts at the run's
beginning with no state at any length) gets `no_boundary_anchor` and recomputes. A fork
that renames `has_positionally_stable_blocks` gets `unknown`, which is a refusal: a
restore we cannot justify is a restore we do not do. A GDN group that is unstable for a
reason that is not the align hand-off stays blocked, unchanged.

Tests: `tests/test_disk_gdn_boundary_restore.py` (21 tests: the derived contracts, the
offer end-to-end on the live geometry, the null-then-real list, the last-GPU-block
pairing, the anchor cut, both new refusals, the named-block/slot invariant that caught a
double-appended anchor during this change, copy-at-completion and its negative control,
the byte cap, and the byte figure settled against a written file).
`tests/test_disk_restore_group_gate.py` keeps the refusal, re-aimed at the
`unstable`/`unknown` shapes, and adds `test_the_align_hybrid_groups_are_restorable_by_contract`.

## The buried journal: the index DRAINS now, and a partial index cannot masquerade as caught up (0.2.28)

**One defect, one deploy.** The 0.2.22 rule - take each journal's newest window and
forfeit everything older - was unrecoverable by construction and blind in exactly the
place that mattered. Measured on the box: four journals totalling **4,711,140 bytes /
132,099 records** with ~**6,290 net-live identities** (proven by a full front-to-back
replay) and 14,729 live files - and the index reported **`entries: 146`**, so
`no_disk_run` refused every offer for hours. Each first pass was bigger than its share
(131,072 of a 512 KiB budget over four journals), so it read the newest ~131 KiB,
`_forget_rank`ed everything older and landed the cursor at EOF - and a cursor only ever
advances, so a restart re-buried the same 1.4 MB, deterministically. The tail that night
was 93% removal lines (a ceiling reclamation), so the surviving window was 146
identities. `lag_bytes` sat at 0 because the cursor WAS at EOF, and `why_no_restore`
only spoke when `lag_bytes != 0`: the one blindness it was silent about was the one that
was happening. The plugin's own `op:contains` contradicted the store - identities whose
`+` sits early in a journal and was never retracted answered `not_present` while the
file sat committed on disk (`disk_spill` journals a `+` only after `os.replace` commits).

**The rule now: DRAIN, in bounded passes.** `_apply_journal` reads each journal FROM
ITS CURSOR FORWARD, applies whole lines in the file's own order, and when the budget
binds it RETURNS WITH THE CURSOR SHORT OF EOF - the next refresh continues where this
one stopped. Nothing is forfeited, nothing is forgotten, and the buried branch (and its
mid-line alignment probe) is gone: the "don't re-read history" optimization it traded
for - amnesia - is provided honestly by the cursor, which never moves backwards and
never re-reads a converged journal beyond its new tail. **A single refresh after the
first still reads at most `MAX_READ_PER_REFRESH` (512 KiB) and applies at most
`MAX_LINES_PER_REFRESH` (8,192) lines** - the offer-path bound is unchanged and now
auditable through the published `bytes_read`. The FIRST refresh of a process runs the
passes back to back under its own budget - `BOOT_DRAIN_MAX_BYTES` (32 MiB read) and
`BOOT_DRAIN_DEADLINE_SECONDS` (5 s wall) - so a cold boot converges inside the first
offer attempt instead of one share per second afterwards. Tonight's shape costs the
boot: ~4.71 MB read (each byte once, plus at most one line-capped window re-read),
~17 passes, well under a second with warm page cache, on the scheduler thread, inside
the first `get_num_new_matched_tokens` after engine start. Either bound ending the
drain early is normal: the index stays `converged: false`, says so everywhere, and
finishes through ordinary refreshes.

*(0.2.65: both of those numbers moved - 32 MiB to 64 MiB, 5 s to 45 s - because on the
night they were shipped this the boot still stopped with 2.3 MB unread, and the drain is
now issued at connector creation instead of waiting for the first offer. See* Who pays
for the drain (`disk_index.py`, 0.2.65)*.)*

**A partial index can no longer read as a caught-up one.** `disk_index.section()`,
`counters()`, the `restore.index` payload and `op:contains`'s own view now publish
`converged` (every cursor at end-of-file), `boot_drained`, `unread_journals` (per-journal
unread BYTES, named), `bytes_read` and `entry_cap_hit`. `why_no_restore` fires its
blindness sentence on BOTH blindnesses - a drain in flight *and* forfeited history
(`catch_up_jumps`/`bytes_skipped` non-zero with a zero `lag_bytes`, the exact silent
shape) - and the `no_disk_run` refusal carries it too, so the answer reads "spilled but
not yet indexed", never "never spilled", unless the data says it cannot. At the
`MAX_INDEX_ENTRIES` cap the drain drops the OLDEST identities (the restore path wants
the recent spills), counts every drop in `entries_dropped`, flips `entry_cap_hit: true`
and logs ONE warning per boot - a capped index is loud, never a caught-up costume.

**The arm proof reads its own authority.** `disk_restore.armed` in a rank file is
written by `worker_diag._maybe_write` on cadence (50 steps) or a signature change - and
the signature never hashed the armed verdict. So an arm with no spill activity rewrote
nothing, and an idle box wrote nothing at all: the scheduler's `_read_worker_proof`
read a stale `armed: false` and refused with `workers_not_servicing` although every
worker would serve (reproduced live: 0/4 ranks "armed" for 120 s after an arm, then 4/4
seconds after a spill burst). The arm is a **rank-agnostic boot fact owned by the
EngineCore process that runs the scheduler**, so the proof now reads
`disk-spill-arm.json` directly through `disk_spill._read_arm_file` - boot fence and
expiry untouched, a reboot still cannot arm a cold box, and no rank file can arm a
disarmed tier by remembering `armed: true`. Rank files keep the facts they uniquely
know: `installed`, the new `worker_fitted`, `service_reason_kind: "not_configured"`
(a worker without its own disk root still vetoes the whole fleet, exactly as the
2026-10-05 incident demands) and `reads_applied`. A rank file's lagged `armed: false`
under a live arm file no longer vetoes - it is counted as `stale_arm_claims`. As
defence in depth the armed verdict is now folded into the writer's change signature
through a new `signal=` hook on the contribution seam, so an arm transition rewrites
the rank file on the next step; it is not the fix, because an idle box writes no rank
file at any cadence.

Costs: one bounded read of the arm file per `worker_proof()` cache miss (1 s floor) and
one signal evaluation per step inside the writer's existing signature test. Tests:
`tests/test_disk_index_journal_drain.py` (the incident shape, the monotone convergence,
the exact byte bounds, the cap's shout, both blindness sentences, the arm-proof and its
boot-fence refusals), `tests/test_disk_index_catchup.py` (rewritten to pin the drain),
`tests/test_worker_diag.py` (the signal seam), and the retargeted worker-proof tests in
`tests/test_disk_restore.py` / `tests/test_disk_restore_scope.py` (the incident is now
spelled `worker_fitted: false`; the old `armed: false` spelling is pinned as the
lockout it was).

## The eviction clock is a touch clock, and nobody's neighbour is capacity (0.2.29)

**What was asked**: "is the disk cache LRU, and does eviction happen non-blockingly?"
**What the source said**: no, and no - in three separate ways.

**1. The clock only ever saw writes.** Eviction ranks by `st_mtime` (`/home` is
mounted `relatime`, so `st_atime` is not a clock anyone can trust here), and until now
nothing moved `st_mtime` except a spill. A span restored a hundred times was therefore
evicled by a span written an hour later and never read - the tier preferentially threw
away the bytes its own restore path had just proven valuable. A restore now *promotes*:
`disk_read.promote_recency` bumps the file's mtime, so **recency means last touch, and a
touch is a write OR a successful restore read**. Three rules make that safe, and all
three are tested: it is **throttled** to one metadata write per file per
`TOUCH_MIN_INTERVAL_SECONDS` (60 s, on the monotonic clock; the timestamp *written* is
wall clock - writing the monotonic value would date every promoted file to 1970 and
evict the hottest spans first), the throttle map is **bounded** at 4,096 paths and
evolves its own oldest entry, and it is **fail-closed on the verdict** - `file_missing`,
`size_mismatch`, `short_read`, `not_ours` and `read_error` never promote, because
keeping a wrong-sized or unreadable file alive is strictly worse than evicting it. It
runs on the **reader thread** (`_reader_loop` → `_handle_read` → here, after the job has
settled), never on `get_finished` or the scheduler step. Promotion is an mtime and not a
side table on purpose: there is no new durable state and nothing to replay after a
restart. Watch it work in `read.recency_promotions` /
`read.recency_promotions_throttled` / `read.recency_promotion_failures`.

**2. Every rank shed its siblings' live data first.** Four ranks share ONE spill root,
and `lru_eviction_plan` was called without the `peer_digests` parameter it already
accepts. A digest that is not this boot's classifies as `stale_geometry` - "this boot can
never read it, it is pure capacity, **shed it first**" - so each rank's cheapest file to
delete was one of its three peers' live, loadable, hottest spans. Four evictors running
that rule against one shared root is a mutual-destruction loop, and a fill test is exactly
when it bites. The shed order is now
`stale_geometry`/`stale_key_shape` → **this boot's own files, oldest touch first** → a
**live peer rank's** files, and a rank *learns* its siblings' digests from the store walk
it already performs: a foreign digest with at least two committed files written inside
`PEER_DIGEST_FRESH_SECONDS` (120 s) is a rank that is writing right now.

**That protection was conditional, and 0.2.34 made it honest.** "Peers shed last" was
only ever true for a peer the evictor could NAME. The learning is a write-freshness
witness, not an identity: a sibling that is alive but has not written a spill in the last
120 s is not confirmed, and with nothing confirmed every foreign digest classified
`stale_geometry` and went FIRST. What was written at the time as "the fail-safe direction
is deliberate: if no digest qualifies, the pass behaves exactly as 0.2.28 did (foreign
first)" was the bug, not the safety property - on a shared root "foreign first" is four
ranks deleting each other's live KV data. See *An unattributable name never buys
capacity* below for the gate, the short plans it permits, and the witnesses it publishes.

**3. The pass had no deadline and one hand.** Up to 100,000 directory entries read
twice, 20,000 `stat` calls and 4,096 unlinks, serially, on the writer thread that is the
only thing draining a spill queue of 256 jobs / 256 MiB - so a full store dropped spills
while its evictor walked. The pass now has a **time budget** (`1.5 s`,
`VLLM_E2_TIER_DISK_CEILING_BUDGET_SECONDS`) honoured by the walk and the unlink loop
alike. Truncation happens **between whole per-directory batches**, never inside one, so
no plan is ever half-applied; the **first batch always runs**, because a pass that
released nothing only defers the work to a spill that will arrive into the same full
queue; and a submitted unlink is **always awaited and always collected**, so a truncated
pass can never strand a deleted file without its removal-journal record. The pass reports
`ceiling_measurement.pass_seconds`, `.deadline_hit`, `.deadline_hits` and
`.budget_seconds` so "is the ceiling keeping up with this fill rate?" is a reading and
not a hope.

**4. The pool, and why its default is 1.** The unlink batches (already grouped by shard
directory, which is both the filesystem-friendly shape and the partition that guarantees
no path is attempted twice) can fan out to a small named worker pool,
`kvtier-disk-evict`, built once per process. **Measured before shipping, on 20,000 files
across 64 subdirectories, three paired runs**: 1 thread deleted them in **0.402 s** and 4
threads in **0.557 s** - parallel `unlink()` is metadata work, and here it is ~38%
**SLOWER**, not faster. So the mechanism ships and the default stays serial
(`VLLM_E2_TIER_DISK_EVICT_THREADS`, unset = 1). *Caveat worth repeating*: that measurement
ran on the development host's `btrfs`/NVMe, because the serving box was not touched; if a
measurement on the serving array's `/dev/md1` (XFS) ever shows a real win, the knob is
there and the tests already pin that 1 worker and 4 workers shed identical files.

**No direct I/O anywhere, confirmed rather than assumed.** The read path is plain
buffered `open(path, "rb")` + `readinto`, and the write path is buffered
`O_WRONLY|O_CREAT|O_EXCL` + `os.replace`; there is no `O_DIRECT`/`O_SYNC`/`O_DSYNC` in
the spill or read path, and this release adds none. That is a deliberate choice kept
checked: this store reads at **25.4 GB/s from the page cache against 6.65 GB/s cold**, so
the page cache is the tier's largest single asset and `O_DIRECT` would discard it.

Tests: `tests/test_touch_promotion_and_peer_eviction.py` (the operator's request as a
behaviour - the read file survives the merely-written one; the throttle and its fake
clock; every failure verdict refusing to promote; the bounded map; a known peer's live
file not being the first thing shed, and an unconfirmable digest keeping the old
behaviour *and* being counted; a single sighting not being enough; the deadline stopping
the submission, releasing bytes, marking itself, and re-running to convergence without
deleting anything twice; the pool at 1 and 4 shedding the same files and never handing a
path to two workers).

## Four ranks, one ceiling: the fair share, and `disk_fill` (0.2.30, block-equal since 0.2.58)

**What the operator measured**: a four-rank boot sharing one spill root wrote 92 files
per rank and evicted **2,106 / 2,106 / 1,872 / 1,872** files - **7,956 files, 156 GiB
down to 68 GiB** - under a **148 GiB** ceiling whose `EVICT_TARGET_FRACTION` of 0.9 made
the one-pass target 133 GiB. One pass owed ~23 GiB. Four passes shed ~23 GiB each.

**What was wrong**: nothing miscounted. Every rank measured the WHOLE store (156 GiB),
every rank correctly wanted it at 133 GiB, and all four then acted on the SAME
directory. A shared store enforced per-rank against the whole ceiling has an effective
capacity of `ceiling / rank_count`, so the more ranks a boot has, the more of its own
cache it destroys.

**The rule now.** A rank sizes only the bytes it is ACCOUNTABLE for - its own geometry
digest's files, plus the files that belong to no live rank (`stale_geometry` /
`stale_key_shape`: a reader enforces an exact digest and byte-size match, so a mismatched
file can never be applied by anybody and every rank may shed it) - against
`ceiling / n_effective`, where `n_effective = 1 + (live peer digests its own walks
learned)`. A live sibling's bytes are then **neither counted nor sheddable**: its
entries leave the planner's input entirely (`peer_entries_excluded` counts what was
withheld), because the sibling accounts for them under its own share. The convergence is
arithmetic, not tuning: if every rank holds `own + dead <= 0.9 x C/N`, the store holds
`<= 0.9 x C`.

**The fail-safe direction, stated because it is the first question asked.** With no peer
learned, `n_effective` is 1, the share IS the whole ceiling, and the pass sheds byte for
byte what every release before this one shed - including a single-rank boot. The old
store-wide clause is kept INSIDE `_ceiling_trigger` (`estimated_bytes >= ceiling` still
starts a pass), so the new rule can only ever make a pass start SOONER, never later:
capacity safety cannot regress by construction, and `too timid` is not a direction this
exposes. The worst case it permits is a boot whose ranks are mutually invisible - each
holding up to `0.9 x ceiling` of unattributed bytes - which is bounded in practice by the
walk the deciding pass itself performs, and hard-bounded by the unchanged `statvfs`
floor (`MIN_FREE_BYTES`, refusal `no_free_space`).

**The witnesses.** `disk_spill.section().ceiling_share` publishes `accountable_bytes`,
`own_bytes`, `dead_bytes`, `peer_live_bytes`, `unattributed_bytes`, `share_bytes`,
`target_bytes`, `share_divisor`/`n_effective`, `share_source`,
**`share_blocks`/`share_basis`/`own_width_bytes`/`peer_widths_known`** (0.2.58, below) and
**`passes_declined`** - the count of passes that walked, found their accountable bytes
already inside their share, and shed nothing. That is the "one rank did the work and
three declined" signature, counted rather than logged, because the pass runs on the
spill writer thread. `passes_nothing_sheddable` is the different fact: over its share and
nothing it is allowed to shed (where the filesystem floor is the control).

**Equal in BLOCKS, not in bytes (0.2.58).** `ceiling / n_effective` split the store
equally in a currency a restore cannot spend. The ranks' block widths differ - on the
live box the PP0 ranks spill `11,468,800`-byte blocks and the PP1 ranks `12,902,400`-byte
ones - so one byte budget was a different cache depth per rank: under a
`206,158,430,208`-byte ceiling every rank was handed `51,539,607,552` bytes, which holds
**4,493** blocks on PP0 and **3,994** on PP1. A disk restore is servable only when the
block is present on EVERY rank, so the store's useful depth is the MINIMUM across ranks
and the 499 surplus PP0 blocks per rank were disk space holding something that could
never be served. The share is therefore taken in blocks first:
`blocks = ceiling // sum(widths)`, `share_i = blocks * width_i` - the floor taken once, on
the quantity every rank must agree on, so `sum(shares) <= ceiling` still holds and the
same ceiling buys **4,229** blocks of servable depth per rank instead of 3,994. The
widths come from this rank's own fingerprint and each live sibling's `arena.bytes_per_block`
in the rank file its identity read already opened; **a missing width is never guessed**,
and the split falls back to the equal-byte arithmetic with `share_basis: "equal_bytes"`
(the pre-0.2.58 shape) rather than sum a set of widths that is missing a rank - an
under-counted sum makes every share too big, which is the over-eviction direction.
Consequence to expect on a mixed-geometry boot: the ranks' `ceiling_share_bytes` now
legitimately DIFFER (same depth, wider block costs more), so `cross_process_verdict`
judges agreement in blocks - `blocks_equal`, `share_blocks`, `share_blocks_values`,
`share_basis` - and `share_bytes` there is `null` when the ranks hold equal depth at
different widths.

**`disk_fill`: one block that answers "how full is the SSD tier, and is that a
problem"**, at the top level of `GET|POST /plugins/kvtier/snapshot` and `GET
/plugins/kvtier/status`. It is built from the rank files the WORKERS write (behind
`rank_rollup`'s one-second cache, keyed on a content signature, so a rank that wrote is
visible on the next read), it makes no engine call, and it emits rank entries only -
capped at `DISK_FILL_RANK_CAP` with `per_rank_truncated` said out loud. The keys a proxy
should write against:

| key | meaning |
|-----|---------|
| `used_bytes`, `ceiling_bytes`, `used_fraction`, `target_fraction`, `target_bytes`, `headroom_bytes`, `over_target`, `over_ceiling` | the fill, as arithmetic - no caller-side maths |
| `basis` | **how the size was measured**: `full_measurement`, `sample_scaled: n of m`, `no_rank_files`, `no_rank_reported_a_measurement` |
| `read_basis` / `ceiling_basis` | how complete the READ was / whose ceiling this is (`ranks_agree`, `ranks_disagree`, `this_process_env_only`) |
| `used_bytes_source`, `used_bytes_spread_bytes` | which rank the number came from, and how far the ranks' views were apart |
| `ranks_expected`, `ranks_reported`, `missing_ranks` | fleet coverage. `ranks_expected`/`missing_ranks` are **null, not `[]`**, when this process cannot know the rank set |
| `per_rank[]` | `rank`, `file_age_seconds`, `files_written`/`spilled`, `bytes_written`, `evictions`, `journal_removals`, `drops`, `queue_depth`, `recency_promotions`, `recency_promotions_throttled`, `ceiling_share_bytes`, **`ceiling_share_blocks`**, **`share_basis`**, `share_divisor`, `declined_because_already_under`, `estimated_bytes`, `walk_basis`, `writer_thread_alive` |
| `totals`, `totals_are_lower_bound`, `lower_bound_reason` | the fleet sum over FRESH ranks only, with the ranks it could not see named |
| `cross_process_verdict` | `shared_share` / `single_rank_share` / `ranks_disagree_on_the_share` / `no_ranks_reported`, with `ranks_acting`, `ranks_declining_names`, `declines_total`, `n_effective`, and (0.2.58) `blocks_equal`, `share_blocks`, `share_blocks_values`, `share_basis`/`share_basis_values` - agreement is judged on BLOCKS, because equal bytes across ranks of unequal width is the withheld-capacity state, not the healthy one |
| `per_process_note` | the sentence that says the flat `disk_spill` / `disk_store` counters beside it are THIS process's own and read zero on the API process |

`used_bytes` is the FRESHEST rank's estimate and NEVER a sum: four ranks measuring one
directory produced four views of one number, and adding them would be the same mistake
the ceiling just stopped making. A store nobody has measured is `used_bytes: null`, never
`0` - "empty" and "unmeasured" are different facts, and on a polled dashboard only one of
them is a lie.

**The flat sections now label themselves.** The zeros an operator reads at 03:00 are
usually not wrong, just mis-scoped, so `disk_spill.scope` and `disk_store.scope` state in
the payload that every counter in those blocks is THIS process's own - the writer thread,
the ceiling pass and the spill hook all live in the process that steps tokens, so on the
API process they are zero *by construction* - and point at `disk_fill` / `spill_rollup`
for the fleet's numbers.

Tests: `tests/test_ceiling_fair_share_multi_rank.py` (four ranks that all measured the
same pre-shed store and then all acted on it, the incident's interleaving: the store
lands at or under `0.9 x ceiling` AND the bytes freed are within ~1x of the overshoot -
the pre-0.2.30 code frees **4.0x** it, which is what the incident was; the second round
declines and deletes nothing; a lone rank still sheds to 9/10 of the whole ceiling, byte
for byte as before; the share's floor arithmetic; and the trigger firing no later than
the store-wide rule ever did), `tests/test_disk_fill_block.py` (the fill as arithmetic,
the `basis`/`read_basis`/`ceiling_basis` distinction, nulls instead of invented numbers,
missing ranks named, a stale rank aged, the size cap at 24 ranks, and the signature
cache reparsing when a rank file changes), and eleven tests in `tests/test_routes.py`
(the proxy peer on `192.168.2.64` reading the block with no header at all, a peer off
the allowlist refused on peer grounds while holding the old token, loopback needing
nothing, a spoofed `X-Forwarded-For` buying nothing, the block travelling on
`/snapshot` too, and an inert plugin 503-ing before any fill read).

## What a failed load costs: a race may lose its tokens, a fault loses the offer (0.2.32)

Through 0.2.31 every failure report called `suppress_load`, which black-holes the
**request** and fuses the **span** for the rest of the boot - whatever the failure was.
The reason string handed to that call already contained
`sorted(_recent_failure_kinds())`: the code had the answer in its hand and decided
without looking. The night the settle formula was measured, that meant a span that lost a
135.8 ms race in one group of one span cost a 28,583-token recompute **and**
`suppressed_requests: 3` - a boot-length fuse for a millisecond-scale timing loss, with
no expiry to wait out.

**The 26 failure kinds are partitioned, and the partition is the policy.**
`LOAD_TIMING_FAILURES` (`not_settled`, `deadline_expired`) and `LOAD_CAPACITY_FAILURES`
(`queue_full`, `buffer_cap`) say *the bytes are fine and the moment went wrong*; every
other name is `LOAD_DATA_FAILURES` - a wrong digest, a short read, a size mismatch, a
failed copy, a step whose lists disagree, a broken read path, or a loss nobody
attributed. `failure_kind_classes()` publishes the three lists with the decision rule
beside them so an operator can argue with the partition instead of reading source, and
`test_disk_restore_races.py` pins every name into exactly one class.

**The rule.** Any data kind visible means the fuse stays exactly where the 2026-10-05
incident put it (request and span, forever). Only timing/capacity kinds mean the request
keeps its next offer and the span is not implicated. NOTHING visible means `unknown`, and
`unknown` fuses: the scheduler learns a worker's kinds through the rank file, which lags
(`PROOF_CACHE_SECONDS`), so guessing "race" when the evidence has not arrived is how the
loop that starved the box comes back. Every decision is counted - `races_not_fused`,
`fuses_on_data_kinds`, `fuses_on_unknown_kinds`, `race_reoffers_granted` - and the last
one is written as a sentence in `last_fuse_decision`, because "why is `races_not_fused`
zero?" has to be answerable without a control experiment.

**The engine's report shape is not evidence.** `failed_recving` and `invalid_block_ids`
named the CHANNEL the bad news arrived on; `_note_load_failure` files any name outside
the `WORKER_FAILURES` vocabulary under `read_failed`, which is a data kind. A classifier
reading the plain ring would therefore see a corruption on every report ever written and
would never spare a race - the fix would be inert while looking alive. The classifier
reads only what the code that touched the bytes wrote (`where` = `apply`, `bind`,
`offer`, locally or carried by a live rank file) and excludes `where="report"`.

**The loop stays bounded without the permanent fuse.** A spared request gets exactly one
re-offer (`MAX_RACE_REOFFERS_PER_REQUEST`, spent when used, never refilled), so no
request is ever offered more than TWICE per boot whether its losses are races or faults -
against the 123,474 offers in 3.9 minutes that starved the box on 2026-10-05.

**The engine blocker, named rather than worked around: on this box a hole costs the whole
prompt, and a RACE costs it too.** The plugin-side half of the shrink is in place - an
entry without bytes is pruned, its GPU block reported, and the entries that landed stay in
the step, so the contiguous prefix survives the hole. What the engine does with the
reported blocks is not ours to choose: under
`--kv-transfer-config '{"kv_load_failure_policy":"recompute"}'` the scheduler
(`vllm/v1/core/sched/scheduler.py:3193-3215`) keeps `num_computed_tokens = 0` (the
assignment is at `:3205`) for a
request whose KV cache config has MORE THAN ONE group - this box runs 5 serviceable
groups, so all 12 reported blocks collapse to one contiguous prefix of ZERO tokens and
the whole prompt recomputes. The shrink-to-a-prefix mechanism exists only on the
single-group path (`scheduler.py:3217-3252`, with the truncation in
`_get_blocked_tp_blocks`). So on this deployment the flat allowance is not a way to
recompute a little less - it is the reason to wait at all: a race lost costs the full
recompute either way, and a 5 s wait costs the tokens only when it fails. `partial_load_verdict`
publishes that reading beside the counts.

## An unattributable name never buys capacity: fail-closed eviction (0.2.34)

**What happened.** Four ranks share ONE spill root and tell files apart only by the
geometry digest in the name. The persistent journal at `/home/user/kv-tier/journal/`
carries **15,848 removal lines of size 11,468,800** - the block size of the OTHER rank
(PP0 is 11,468,800 B, PP1 is 12,902,400 B) - recorded by ranks that were not the writer.
A `file_missing` read then opened a hole in the middle of a restored prefix and
everything behind it, **19,200 tokens**, was recomputed.

**Why the peers were never known.** The peer map is not an identity channel, it is a
write-freshness witness learned from the walk the ceiling pass already performs
(`_observe_peer_digest`, one entry per foreign-digest file the sample stats). A digest is
only believed with `PEER_DIGEST_MIN_SIGHTINGS` (2) committed files written inside
`PEER_DIGEST_FRESH_SECONDS` (120 s), and a digest that ages out is **deleted** from the
map (`disk_spill._live_peer_digests`). So a rank that is alive but idle - which is most
of a serving box's life, and the normal state of three ranks while one fills the store -
is UNCONFIRMABLE, and the next pass runs with `peer_digests=()`. That is not a boot-order
bug: both call sites compute the tuple *after* the walk that would have fed it, so no
reordering helps - the input cannot support the claim. And with an empty peer map
`classify_name` has no peer to name, so a sibling's live restore answers
`stale_geometry`, which was bucket 0: shed before this rank's own files.

**The gate.** Within the ceiling/eviction path, a committed file whose name parses under
a digest that is **not this boot's**, seen while **no peer digest is confirmed**, is
`unattributable` and is never a candidate for unlinking (`disk_layout.attribute_name`, enforced
in `lru_eviction_plan`, again in `_attribute_walk_bytes`, and a third time in
`_validate_unlink` for `for_capacity=True` - nothing planned upstream is trusted at the
point of deletion). It is the unknown case of the rule this module has always had for
malformed names: **reported, counted, never deleted**. Still sheddable, deliberately:
this boot's own files; the pre-group `stale_key_shape` (our lineage, unloadable by any
group-aware reader); an in-flight `.part` (not a committed file, so nobody's restore);
and, when the peer map IS confirmed, any digest outside it - which is exactly what a
confirmed map means, and what keeps a store of old boots' geometry sheddable instead of
turning the ceiling into a comment.

**The consequence is a short plan, not a silent no-op.** If the ceiling cannot be met
without shedding unattributable files, the plan sheds what it is allowed to and comes up
**SHORT**: `short` / `short_by_bytes` / `excluded_unattributable` /
`excluded_unattributable_bytes` on the plan, and `plan_short`, `plan_short_bytes`,
`plan_short_reason` (one line, gate-named, not a stack trace) and the boot total
`passes_short` on `disk_spill.section().peer_ranks`. **Exceeding the ceiling is the
correct failure mode**: spilling back-pressures, the unchanged `statvfs` floor
(`MIN_FREE_BYTES`, refusal `no_free_space`) still guards the filesystem, and an operator
can act. Deleting a sibling's live restore corrupts a serving path in silence.

**The witnesses, published this time.** `peer_digests_live` / `peer_digests_unconfirmed`
/
`peer_digests_dropped` / `peer_files_planned` used to exist only inside
`disk_spill._state`, so the journal evidence could not be resolved into a mechanism. They
now travel in the per-rank file under **`disk.peer_eviction`** (`disk_store
.worker_report`), with the SAME key set on the healthy path, on a rank that never ran a
ceiling pass, on the path where the counters raised, and through the contribution seam's
byte-cap triage. `disk_spill.section().peer_ranks` is the full view, and
`disk_store.rank_rollup()` sums `peer_files_removed`, `peer_files_removed_known_peer`,
`peer_digests_unconfirmed`, `unattributable_entries_excluded` and `ceiling_passes_short`
across ranks, so the dev route answers it in one call.

**The removal is attributable.** A `-` journal line says an identity left the store; it
cannot say whose FILE that was, and a `-` from rank A followed by a `+` for the same
identity leaves the index offering an identity that physically exists only under another
rank's digest. So every unlink whose name carries a digest that is not this rank's
bumps **`peer_files_removed`** (and `peer_files_removed_bytes`), and the one that cannot
lie - the digest belonged to a rank this walk believed LIVE - bumps
**`peer_files_removed_known_peer`**. `last_reason` gets one sentence naming both. The
journal's five-field wire format (`seq op key16 size group`) is deliberately untouched:
`disk_index` parses it, it is durable shared state, and rank-aware identity is its own
bigger change with its own review.

**Two scope boundaries, pinned rather than accidental.** `execute_unlink_plan` with
`for_capacity=False` is the boot sweep's "the operator asked me to reclaim the bytes
nobody can read" path and keeps its old allowed set (it is not wired into production -
the ceiling pass is the only production deleter). And the gate cannot tell a sibling's
digest from THIS rank's own obsolete geometry: `geometry_digest` is a hash of the whole
fingerprint, so a name carries no rank to compare. A rank that restarts under a new
fingerprint therefore cannot shed its own pre-restart files while it cannot confirm
peers; the fix is a small durable "digests this rank has used" history, and it is a
follow-up, not a silent extra write on an engine path.

Tests: `tests/test_unattributable_eviction_fail_closed.py` (the incident as a behaviour on
a filesystem - a peer-blind pass leaves the sibling's files and reports the shortfall;
a confirmed peer's files are never in the destructive set at all; a digest outside a
confirmed map is dead capacity again; the planner's own arithmetic, and its stable key
set; the `.part` exception, pinned explicitly; the executor refusing an unattributable
path a caller handed it, and the sweep path keeping its own doctrine; the four paths of
`disk.peer_eviction` shipping one key set; the dev door summing the cross-rank removals;
and a removal under a known peer's digest being counted, not buried), plus the rewritten
doctrine tests in `tests/test_disk_layout.py` and
`tests/test_touch_promotion_and_peer_eviction.py`.

## Does the GPU still hold what the arena spilled? (`disk_shadow.py`, 0.2.36)

The spill path captures a slot by reading the **arena** - the worker's
`cpu_kv_caches`, the copy vLLM itself makes every step - and that is the only copy of
a block's bytes the tier ever writes. If that read captured the wrong bytes, or
captured them in an order the restore side does not reassemble, the tier would serve
silently wrong KV, and **no counter that exists today would move**: a byte-level
mismatch is invisible to a hit rate. The shadow verifier is the witness for exactly
that.

It is a measurement, not a gate. Default **OFF**, read-only, fail-open: it never
changes a spill or a restore decision, never refuses a write, never opens a file
handle for writing, and touches the GPU only to READ one block of it. At store **bind**
it retains the GPU block ids the engine named next to the CPU slots (ints only - never
a tensor, never the metadata object). At store **completion**, for a sampled subset, it
narrows that one block on the device, copies it to the host on the current stream, and
compares the bytes against the blob the spill path captured for the same block.

```bash
# the worker's switch is the environment, because the worker is another process:
#   VLLM_E2_TIER_DISK_SHADOW=1        (read once at first use; restart to change it)
# this op reads the counters, and arms the process that ANSWERS (it writes no file):
curl -s -X POST -d '{"op":"shadow"}' $B/disk
```

**A mismatch is a QUESTION, and the categories say which one.** A completed store's GPU
block is the engine's to reuse the moment the store finished, so the shadow re-reads the
block once and asks whether the GPU's own bytes moved between the two reads:

| counter | what it means | what you do |
|---|---|---|
| `equal` | the GPU and the arena agree | nothing; this is the number growing |
| `mismatch_gpu_changed_on_recheck` | the GPU rewrote the block between the reads | nothing: expected block reuse |
| `mismatch_stable` | the GPU held still and still differs | **this is the one** - a layout/ordering question |
| `mismatch_unrechecked` | the re-read could not run | look at `reasons` |
| `mismatch_rebound_later` | a block that mismatched was named again by a later bind | evidence of reuse, not proof of timing |
| `width_mismatch` | the GPU's region widths disagree with the arena's | a layout answer, never counted as staleness |
| `no_gpu_block` / `no_gpu_caches` / `no_blob` | no GPU id, no GPU tensors, or the spill refused | context for the sample rate, not a fault |

The last three are why there is no single "corruption" counter: "the two paths
disagreed", "there was nothing to compare" and "the engine reused the block" are three
different facts, and merging them into one integer would make the number useless on a
busy boot.

**What it costs, and where the bound is.** The copy runs on the engine's step thread, in
the same tick `MAX_SLOTS_PER_TICK` bounds. `MAX_BLOCKS_PER_STEP` (2) is the hard cap per
bind window - exceeding it increments `throttled_step` and does no work - with
`MIN_INTERVAL_SECONDS` (0.5, counted as `throttled_interval`) and 1-store-in-32 sampling
on top, so the cost is a constant rather than a function of the store rate. One block
per region is `narrow`ed BEFORE the copy, so it never moves a whole KV cache across the
link.

**Two invariants it exists to test.** The bytes of a slot are the sum over the region
tensors and the order of that sum is `sorted(name)`, never dict-insertion order (0.2.8
shipped that bug) - so the shadow does not re-implement the ordering: it hands the
copied one-block regions to `disk_spill.read_slot_bytes`, THE SPILL PATH'S OWN READER,
and a mismatch therefore cannot be "two readers summed differently". And because
everything downstream is `ctypes.string_at(data_ptr())`, a device pointer read as a host
address is a SIGSEGV rather than an exception, so the copy's result is checked against
its OWN `device` field before a byte is read (`_host_landed`).

The verdict goes into the worker's rank file through `worker_diag` as the `disk_shadow`
section - a guest section, never a second writer, never a second file - and its key set
is identical whether or not this process can read a GPU at all, so a reader diffing two
rank files can tell "moved" from "gone". `tests/test_disk_shadow.py` is the executable
spec; the device->host copy is the single expression it replaces, because that
environment has no CUDA device.

**Reading a mismatch in the file.** `last_mismatch` is a SHAPE, never bytes:
`category`, `slot`, `gpu_block`, `bytes`, `first_difference` - the byte offset where the
two blobs first differ, or `null` when they agree for the whole `scanned_bytes` window
(`MAX_DIFF_SCAN_BYTES`, 256 KB, is the scan cap, so "they differ past 256 KB" is an
honest answer and a 11.4 MB Python scan on the step thread is not) - and the two recheck
verdicts. It is never a trim target and never gains a key. `trimmed` names what the
section shed to fit its own `MAX_SECTION_BYTES` (3072, tighter than the seam's 4096, so
the writer never has to refuse it) and `section_trimmed` counts how many times that
changed shape; the counters are never shed, so a thin section is still a complete set of
numbers. If a section is over budget after every shed, the writer refuses it and the
worker log says `section STILL over its byte budget` - a missing `disk_shadow` key with
that line beside it is a size problem, not a missing facility.

Its `equal` verdict is also the *admission rule* for the next step of the route - writing
those GPU bytes to a second tree and diffing the two files - see *Writing the block twice:
the GPU-direct shadow* below.

## Writing the block twice: the GPU-direct shadow (`disk_gpu_shadow.py`, 0.2.37; whose bytes a differ is, 0.2.48)

Entry 27 of the plan established that the host-RAM arena cannot simply be deleted: it is
the spill trigger, the byte source, the restore landing pad and the only GPU<->host
copier. Entry 28 decided the route - build a GPU-direct write and a GPU-direct read, and
prove each one as a shadow before letting either be authoritative. The verifier above
(`disk_shadow.py`) is step 1 and it said `equal` in production. **This module is step 2:**
the GPU-direct WRITE path, still not authoritative.

For a block the verifier has just called `equal`, this module writes those same GPU bytes
a second time, to a **separate directory**, under **exactly the name the live store uses**
for that block - and then compares the two files. The live store's file remains the only
thing any read path loads. Nothing in a spill, a restore, an offer or an eviction decision
reads a byte written here. Default OFF, fail-open, and its output is disposable evidence
with a retention cap of its own.

```bash
# the worker's switch is the environment, because the worker is the process that writes:
#   VLLM_E2_TIER_DISK_GPU_SHADOW_WRITES=1     (read once at first use; restart to change)
#   VLLM_E2_TIER_DISK_GPU_SHADOW_ROOT=/path   (optional; default = <store>/../gpu-shadow)
# this op reads the counters, and arms the process that ANSWERS (which is usually NOT
# the worker - the answer says so in `scope`):
curl -s -X POST -d '{"op":"gpu-shadow"}' $B/disk
```

**Files land beside the store, never inside it.** With the box's store at
`/home/user/kv-tier/disk`, the shadow tree is `/home/user/kv-tier/gpu-shadow/<shard>/
kvslot-<key16>-g<group>-<geodigest12>.bin` - the same relative name, a different root. A
shadow root that IS the live store, or is nested in it in either direction, is REFUSED
(`refused_root_conflict`, one log line, nothing written): merging would put bytes in the
store's shard tree that a restore could load, which is the one outcome this module exists
to make impossible. `/a/store` and `/a/store2` are compared on path components and are
not confusable.

**The one command that reads the result.** Run it on the box, after the feature has been
on through some real traffic:

```bash
diff -rq /home/user/kv-tier/disk /home/user/kv-tier/gpu-shadow | grep -v '^Only in '
```

**When everything agrees it prints nothing.** `Only in ...` lines are expected and mean
nothing is wrong: the shadow tree holds only the sampled subset (one store in 32, capped
per bind window), so the live store always has files it does not.

**A `differ` line is not one fact, it is three (0.2.48).** Until 0.2.48 this page said
that a line of the form `Files .../disk/3f/kvslot-....bin and .../gpu-shadow/3f/kvslot-....bin
differ` "is a fault in the GPU-direct write path". That was wrong, and it has cost two
investigations on a live box. A name outlives the completion that created it: **neither
tree ever overwrites an existing name** (`disk_layout.write_committed_file` declines a
duplicate with `duplicate=True, write_bytes=0`, and the spill path writes the live file
through that same function), so each side of a pair keeps the bytes of whichever
execution of the write path first created that name on that side - and fp8 KV
recomputation is not bit-reproducible, so two executions of the same key legitimately
give different bytes. The comparator does not guess. It asks one question
(`disk_gpu_shadow._attribute_pair`): **is the live file older than the instant before
this shadow's own spill was requested?** If it is, the store could not have written it,
and the pair was never going to agree.

| outcome | counter | log | what it means | what you do |
|---|---|---|---|---|
| `cross_execution` | `pairs_cross_execution` | **one INFO per episode** | the live half predates this store, so it is an earlier spill's (or an earlier boot's) bytes under a name that outlived them | nothing. This is expected on every boot, and it is not an alarm |
| `same_execution` | `pairs_fault` | **one ERROR per episode** | both sides came from THIS store, so the GPU-direct write path disagrees with the spill it claims to mirror - or the live file changed after the spill. The line adds two honesty notes when they apply: the offset is past the verifier's 256 KB window (bytes no witness read), or the live half is newer than our shadow file (which is also what a LATER store's file looks like if this store's spill never reached the disk - check `pairs_failed`) | **this is the one.** Read `last_difference`, then the two files |
| `unattributed` | `pairs_unattributed` | **one WARNING per episode** | the module could not order the two sides (no store instant on the plan, a clock that moved, a file that vanished mid-decision) | read `classification_reason`; it is deliberately neither of the above |

`pairs_differing` stays the umbrella, and the three always sum to it: nine differs are
accounted for by their three buckets, and the buckets are what tells you whether nine
deserve attention. The split is deliberately asymmetric - ambiguity goes to
`unattributed`, never to the benign bucket, so the tolerance can make the module timid
but can never make it call a fault benign.

**The grep that tells the three apart.** They are one word in one line each, and each
kind logs once per episode, so this is the whole reading:

```bash
grep -a 'disk_gpu_shadow] a shadow file' /tmp/vllm-serve.log \
  | grep -oE 'same_execution|cross_execution|unattributed' | sort | uniq -c
# and the only line that should ever be non-empty:
grep -a 'differ (.*): same_execution' /tmp/vllm-serve.log
```

**The read-only sequence that shows the new fields after a deploy.** Every command here
is a read; none of them touches the engine. The box's login shell is zsh, so a bare glob
aborts the whole command line ("no matches found") - `find` with explicit paths, and the
JSON is read through a heredoc (`<<'PY'` quotes it against both shells) rather than a
nested `python3 -c` with escaped quotes:

```bash
# which rank files exist, newest last:
ssh llm 'find /tmp/kv-tier-diagnostics -maxdepth 1 -name "rank-*.json" -printf "%T@ %p\n" | sort -n'

# one rank's shadow section: the three buckets, the reference, the last differ:
ssh llm 'python3 -' <<'PY'
import json, glob, os
f = max(glob.glob("/tmp/kv-tier-diagnostics/rank-*.json"), key=os.path.getmtime)
s = json.load(open(f)).get("disk_gpu_shadow")
print(f)
keys = ("pairs_checked", "pairs_identical", "pairs_differing",
        "pairs_cross_execution", "pairs_fault", "pairs_unattributed",
        "pairs_size_disagree")
print({k: s.get(k, "ABSENT") for k in keys})     # ABSENT = this boot predates 0.2.48
print("reference:", s.get("differ_reference", "ABSENT"))
print("last differ:", json.dumps(s.get("last_difference"), indent=1))
PY

# the same verdict in the engine's own log (each worker writes its own lines, prefixed
# by `(Worker_PPn_TPm_EPk pid=...)`, and each kind is one line per episode):
ssh llm 'grep -a "disk_gpu_shadow] a shadow file" /tmp/vllm-serve.log | tail -6'

# and the evidence itself, while retention still allows it (the cap is 64 files /
# 512 MiB, and a differing pair is usually pruned within minutes - which is exactly why
# the byte sample is published in `last_difference` instead of left to be re-read):
ssh llm 'find /home/user/kv-tier/gpu-shadow -type f -printf "%T@ %s %p\n" | sort -n | tail -8'
```

`last_difference` is the record of the most recent differ, and it has the same keys in
every state (benign, fault, unattributed, and the state where the byte sample could not
be read): `classification`, `classification_reason`, `first_difference`,
`scanned_bytes`, `shadow_bytes`, `live_bytes`, `size_disagree`, both mtimes, the
`reference_seconds`/`reference_source` the decision used, `live_predates_this_process_start`,
and the sample: `sample_offset`, `sample_bytes`, `sample_available`,
`differing_bytes_in_sample`, `max_byte_delta`, `sample_reason`. The sample is
`DIFF_SAMPLE_BYTES` (1 KiB) read from BOTH sides starting at the first differing byte -
bounded by construction, and nulls rather than zeros when it could not be read, because
`differing_bytes_in_sample: 0` would read as "the sample agrees".

**`differ_reference`, and why it is not `VLLM_E2_TIER_DIAG_EPOCH`.** The section also
publishes when THIS process started (`pid`, `process_start_seconds`, `source`). The
epoch variable is the obvious-looking existing thing and it is the wrong tool: it is an
opaque identifier, not a time, so it cannot order an mtime; when set it is deliberately
the same in every process of the boot, the API server included, so it says nothing about
when this worker started; and when unset it degrades to `snapshot.epoch()`, four random
hex bytes with no time content. This module reports a differ in a worker, so the start
time worth having is the one read from `/proc/self` in that same worker (the fallback, if
`/proc` is unreadable, is the wall clock at first use HERE, and the `source` string says
so because it is only an upper bound). It is published to DESCRIBE the evidence -
"a previous boot wrote the live half" is worth saying - and it is not what decides:
see the box evidence below.

| counter | what it means | what you do |
|---|---|---|
| `written`, `bytes_written` | shadow files committed | nothing; this is the number growing |
| `pairs_checked` / `pairs_identical` | files compared against the live store's file of the same name, and byte-identical | nothing; `identical == checked` is the gate |
| `pairs_differing` | a same-named pair disagrees, **of any kind** | read the three buckets below; the umbrella alone no longer says anything about danger |
| `pairs_cross_execution` | its live half predates this store (0.2.48) | nothing; expected wherever a key is re-spilled, and expected again after every reboot |
| `pairs_fault` | both sides came from this store (0.2.48) | **this is the one** - `last_difference.first_difference`, `differing_bytes_in_sample`, `max_byte_delta`, `differing_names` |
| `pairs_unattributed` | the two sides could not be ordered (0.2.48) | read `classification_reason`; it is neither benign nor a fault, and it is not silent |
| `pairs_size_disagree` | the same-named pair has different lengths | a truncated or half-written shadow file |
| `pairs_unpaired_expired` | the live partner never appeared | the spill refused, was evicted, or the name is not the same file |
| `pairs_shadow_missing` | the SHADOW file went away before its pair was compared (retention shed it, or the directory was cleared) | raise `MAX_SHADOW_FILES`/`MAX_SHADOW_BYTES` if you want every pair decided; it is never counted against the store |
| `refused_not_equal` | the verifier did not vouch for the block, so nothing was written | nothing; this is the admission rule working |
| `refused_not_checked` | there was nothing to compare (a device fault, a refused spill) | context, not a fault |
| `refused_root_conflict` | the shadow root is the live store or nested in it | fix the environment; nothing is written meanwhile |
| `refused_write_failed` / `refused_*` | the directory refused the write, the queue is full, the name disagreed | look at `reasons` |
| `retention_removed`, `walk_truncated` | the cap (`MAX_SHADOW_FILES`/`MAX_SHADOW_BYTES`) shed evidence, or the walk hit its bound | normal; the files are evidence, not data |

**What the box showed on 2026-10-09, and why the boot boundary is the wrong reference.**
Eight same-named pairs differed. All eight had BOTH sides written after the box's last
boot (`uptime -s` 2026-10-08 23:20:12), so ordering files against "when did this process
start?" would have reported eight faults:

```
ef/kvslot-ef88716efb746ada-g5-...   shadow 00:44:20.875   live 00:05:27.231  (+2333 s)
50/kvslot-50ed223d55f5384a-g1-...   shadow 00:43:59.074   live 00:43:58.942  (+ 132 ms)
```

The second pair is the one that settles it: 132 ms is the timing signature of an
identical pair on that box (they sat at 0.25-1.02 s), both files are this-boot, and the
bytes still disagree - because the live name had already been created by an earlier
spill of the same key. Its bytes, measured before retention took the shadow file, were
4,673,131 of 11,468,800 differing with deltas dominated by +/-1 code and a long tail
(the shape of independently recomputed fp8, not of a shifted or truncated block). The
store instant is what separates that pair from a real fault; the boot reference only
describes it.

**Runbook: after every reboot, expect `cross_execution` differs.** A reboot starts a new
execution while both trees keep their names, so the first boot that re-spills a hot
prefix will find live files older than its own stores and log INFO once. That is the
trees working, not KV corruption: the line you must never see is
`differ (...): same_execution`, and `pairs_fault` is the number to watch.


**What a SAME-EXECUTION diff proves, said exactly.** Only blocks the verifier called
`equal` are written, and the bytes written are the bytes that vouching read - so when the
live half is the file this store wrote, the two files agree by construction *at capture
time* - over the window the witness read, which is the first
`disk_shadow.MAX_DIFF_SCAN_BYTES` (256 KB) of the slot, not its whole 11.4 MB. So the
diff is not re-answering "do the GPU and the arena hold the same bytes?" (the verifier
answers that per block, live, in its own counters, for as far as it scans), and a
same-execution difference beyond that window is reported as what it is: bytes no witness
compared. What a same-execution diff does prove is everything this module ADDS: that `disk_layout.build_slot_path`
under a different root produces the same relative name the store wrote (the identity a
restore looks up - the module checks the two naming sites agree and writes nothing if
they ever differ), that what landed is a whole slot and not half of one, and that the
`.part`/`os.replace` discipline held under real traffic. What it does NOT prove, and
what 0.2.48 stopped claiming, is that two trees stay byte-for-byte the same file across
a store turnover: a name that survives a turnover keeps the bytes of the execution that
created it on each side, and those two executions may legitimately disagree.

**It inherits the ordering rule instead of re-implementing it.** The bytes come from
`disk_shadow._read_gpu_slot` - the ONE device->host seam in the package - which sums the
copied one-block regions through `disk_spill.read_slot_bytes`, the spill path's own
reader, in `sorted(name)` order. This module never walks `gpu_kv_caches`, never orders
anything and never forms an address, so it cannot disagree with the store about layout,
and there is no second device-copy mechanism to get wrong. The same reader's width check
is also what refuses the second trap in entry 28: the KV groups OVERLAY one backing
allocation (the slot is the MAX over groups, not the sum), so a design that walked group
specs would try to write a 43 MB file into an 11.5 MB name.

**It never blocks the engine.** The only function the step thread calls is `consider()`
- two dict lookups, a width check, a cap check, one `put_nowait`, and no `stat` or
`makedirs` anywhere in it. Naming, writing, comparing and pruning happen on this module's
own daemon writer thread, bounded at `MAX_QUEUE_JOBS`/`MAX_QUEUED_BYTES` for the queue,
`MAX_PAIRS_PER_PASS` pairs and one slot width of scanning per comparison pass, and
`MAX_SHADOW_FILES`/`MAX_SHADOW_BYTES` for the directory. Retention deletes through
`disk_layout.execute_unlink_plan`, which re-validates every path at the instant of
deletion - so "this never deletes anything outside its own root" is that proven
function's property, tested by planting a file in the live store with the same name as a
shadow file and watching which one disappears.

`tests/test_disk_gpu_shadow.py` (25 tests) is the executable spec: the default creates no
directory, an `equal` block produces a file whose bytes ARE the arena's, a block the
verifier did not vouch for produces none, the LIVE STORE is byte-for-byte what it would
have been without the feature, a device fault changes nothing on disk, a shadow root that
is the store is refused, a queue that is full refuses on the step thread instead of
growing or waiting, retention stays inside its own directory, a pair whose shadow file
retention removed is counted as ours going away and never as the store's absence, the
shadow file is the `sorted(name)` concatenation and not the insertion-order one, and the
published key set does not change shape.

`tests/test_disk_gpu_shadow_attribution.py` (19 tests, 0.2.48) is the spec of the
*differ*: a pair whose live half predates this store is `pairs_cross_execution` logged
once at INFO and never at ERROR, a pair whose two sides came from this store and differ
by ONE BYTE is still one ERROR and `pairs_fault` (and still compares the whole slot, so
the fault claim is worth making), a pair that cannot be ordered is `pairs_unattributed`
at WARNING and is neither benign nor a fault, the sample counts the differing bytes and
the largest delta in a window we craft (with the largest delta deliberately not first, so
it cannot be an artifact of ordering) and never reads more than `DIFF_SAMPLE_BYTES` per
side - counted, with a wrapper around the real file objects - a sample read that raises
says "sample unavailable" with null numbers instead of a `0` that reads as agreement,
the three buckets always sum to `pairs_differing`, the difference record has the same key
set in the benign, fault, unattributed and sample-unavailable states, and the boot
reference is this process's own start even when `VLLM_E2_TIER_DIAG_EPOCH` is set.

## Sizing the CPU-slot reserve an offer leaves alone (`disk_restore.py`, 0.2.38)

An offer does not own a KV byte, but it does own **destination slots**: the blocks a disk
restore reads INTO come out of the host-RAM arena's CPU block pool, the same pool the RAM
tier parks its own blocks in and the same pool the spill path reads OUT of. So an offer
that drained the pool would trade a recompute for a spill that can never happen - which
is why the offer path holds a **reserve** back and clamps itself to

```text
spans = (free_slots - reserve - boundary_slots) // slots_per_span
```

`free_slots` is the engine's own `cpu_block_pool.get_num_free_blocks()`, read exactly the
way the offer reads it; `slots_per_span` is the number of prefix-cacheable positional
groups (1 on the live Qwen3.8-Flash-Next box, whose four GDN groups are *boundary states*
and cost one slot each at the run's end); and `reserve` **was the constant 64 and is now
`VLLM_E2_TIER_DISK_CPU_SLOT_RESERVE`**, defaulting to 64.

**Why it became a knob: the arena it was sized for is gone.** 64 slots was chosen against
a 64 GiB arena. The operator has since run the engine at 8 GiB, which leaves **187
allocatable slots per rank on the PP0 (narrow) pair and 166 on the PP1 (wide) one** - so
the reservation is 41% of what is left on the wide rank before the offer path sees a
slot, and the whole difference is restorable tokens that are already sitting on the SSD:

| free slots/rank | reserve 64 (shipped default) | reserve 32 | reserve 16 | reserve 0 |
|---|---|---|---|---|
| **166** (PP1, 8 GiB arena) | **98 spans = 313,600 tok** | 130 spans of arithmetic -> **128** (walk cap) | **146** -> **128** (walk cap) | 162 -> **128** (walk cap) |
| **187** (PP0, 8 GiB arena) | **119 spans = 380,800 tok** | 151 -> **128** (walk cap) | **167** -> **128** (walk cap) | 183 -> **128** (walk cap) |

Tokens are spans x the engine's 3,200-token scheduler block. **Read the cap column, not
the arithmetic column**: `MAX_RESTORE_BLOCKS_PER_OFFER` is 128 spans, so a *single* offer
cannot exceed 128 spans (409,600 tokens) whatever the reserve says. That makes **~34 slots
the useful floor on a 166-slot rank and ~55 on a 187-slot one**: below that, a smaller
reserve buys no longer offers and spends the RAM tier's headroom for nothing.

**And what the reserve is worth below that floor, stated as a failure mode** - because the
first reader of a "lower it" instruction will reach for 0:

- **0 is honoured**, and it means *"an offer may take the arena's last free slot."* The
  arithmetic stays correct, so this is a posture and not a dead knob - but a burst of disk
  offers then takes the arena to zero free, the RAM tier above has nowhere to park, the
  next spill has no destination, and an offer can even allocate the last slot out from
  under a request that was about to need it. Honourable on a box that restores everything
  from SSD and parks nothing in RAM; the reason the default is 64 and not 0.
- **A reserve at or above the free count declines offers** (`no_cpu_slots`), allocates
  nothing, and raises nothing. This is why there is **no ceiling** on the variable: the
  reserve can only ever make an offer *smaller*, so honouring an absurd value is the safe
  direction, while clamping it would silently replace the machine the operator asked for
  with the one they didn't.
- **A non-integer or a negative value falls back to 64 and logs once**, `DISK[reserve]
  IGNORED: ...; the shipped default 64 stands`. A negative is refused as grammar, not as
  policy: `free - (-1)` would hand the offer more slots than the pool has, and the fork's
  own `get_new_blocks` raises when a count exceeds the free list.

**What it looks like at boot** - the same line that prints the other per-offer bounds,
with its origin, because the number alone cannot answer "did my drop-in arrive?":

```text
DISK[restore] installed=YES ... offer_block_cap=128 offer_block_cap_tokens=409600 \
  cpu_slot_reserve=16(env) ...
```

`(default)` when the variable is unset, `(env)` when the environment's number won,
`(env-ignored)` when the grammar refused what was typed.

**Where to read it without grepping the log.** The value in force is published in the
restore section's `limits.cpu_slot_reserve` (with `cpu_slot_reserve_default` /
`_source` / `_env_var`), in `offer_pool` beside `jobs_the_pool_can_carry`, in the
per-offer `last_offer_pool` next to the bound it decided
(`cpu_slots_free`, `spans_the_cpu_slots_can_carry`, and `bounded_by: ["cpu_slots"]` when
it was the one that cut), in the boot verdict's `arena` dict, and inside the
`no_cpu_slots` refusal reason. So the question *"why was my offer only 98 jobs?"* is
answered by the payload: `166 free - 64 reserved - 4 boundary states = 98 spans of 1
slot`. The rank file deliberately gained nothing: it lives inside
`worker_diag.MAX_CONTRIBUTION_BYTES`.

`tests/test_disk_cpu_slot_reserve.py` (24 tests) is the executable spec: the unset boot
reserves 64 and the pool-with-no-room-above-64 refusal still fires, the affordability
table above is asserted as arithmetic at both ranks' real free counts, the live offer path
is measured at the real 12,902,400-byte width (98 spans at reserve 64, 146 at 16, a
difference of exactly the 48 slots freed), an over-large reserve declines without
allocating or raising, 0 is honoured, every invalid value falls back with one log line,
the value reaches every surface that names the bound, no key set moves when the knob
moves, and an AST scan pins that the bare constant is read by exactly one function - the
resolver - so the default can never become a second source of truth.

## Do the RESTORED bytes match the file? (`disk_restore_verify.py`, 0.2.39-0.2.42)

The two modules above answer a WRITE question: after the arena spilled a block, does the
GPU still hold those bytes, and can those bytes be written straight to NVMe? This module
answers the READ question, which is the one nobody had answered: **after a disk restore,
does the GPU block hold what the spill file holds?** Default OFF, worker-side, read-only,
sampled, and it changes no restore decision.

Why the question is not already answered by the ledger. A disk restore is a chain of
**index-aligned lists**: the scheduler's `token_load_job_list` carries a
`(cpu_block_id, gpu_block_id)` per entry, `_worker_bind` walks the lists beside each other,
`disk_read` fills a CPU slot, the plugin memmoves that slot into the bounce buffer the
entry names, and the fork's `start_load_kv` DMAs the slot into `gpu_block_id`. Every step
is index arithmetic, and a **one-slot shift moves the aligned lists together** - the name
stays well-formed, the ledger row is complete, the counters agree, and the engine serves a
window that belongs to a different token boundary. Nothing in the stack compares the bytes
at the destination, which is exactly the check `disk_shadow` provides for the write path.

**The five assertions, in the order that makes each failure nameable on its own.**

1. **The file is window-wide.** `os.path.getsize == bytes_per_block` for this rank. A file
   that is 12 B (or double-wide) is a geometry or a rank-crossing answer, so it is refused
   as `file_not_window_wide` with BOTH widths in the reason and never compared - comparing
   under a width that cannot be right manufactures a finding.
2. **The group is narrower than the file.** A group whose own `page_size_bytes` exceeds the
   slot width would read past EOF and call the zeros a mismatch: `group_wider_than_file`,
   refused before the file is opened.
3. **The leading `W_g` bytes agree - and only those.** `W_g` is the GROUP's width, and the
   rest of the window is the other groups' bytes, stale by construction. So the file is
   read with `read(W_g)` (the tail is never read from disk at all) and the device bytes are
   sliced `[:W_g]` before they reach the compare. The refusal is structural, not a
   convention: a verifier that compared the whole window would report a mismatch on every
   narrow group forever, and the one real finding would drown in it. `tail_bytes_excluded`
   is published beside every verdict so the reader can see how much nobody claimed.
4. **The name comes from the blocks the restore was asked for** - the binding oracle. It
   re-derives the file's `key16` from `block_hashes` through the engine's OWN helpers
   (`make_block_hash_with_group_id` + `resolve_block_hashes`) and compares that against the
   name in the path. Name-vs-name cannot catch a one-slot shift, because a shift moves the
   index-aligned lists together and both strings stay well-formed; re-derivation from the
   blocks at the index actually used can. **On this fork a worker has no requests** - the
   fork's own metadata module states that "the worker never knows about request identities"
   - so in production every entry counts `oracle_unavailable` and the reason names the file
   that says so. That is the honest answer, published rather than a silent skip, and
   `set_oracle(provider)` is the seam that changes it the moment a source exists.
5. **The truncation shape.** A batch that restored the positional groups and named NOTHING
   for an align-mode group is the Gate B shape: `n_take_g` (`manager.py:445-448`) truncates
   each group's hit list, and an align group's list is `[null]*(n-1) + [one real block]`,
   so a truncation before the run's end removes that group's ONLY real entry while every
   positional group keeps its bytes. The verifier cannot see a missing entry, so it counts
   its ABSENCE (`boundary_entry_missing`, per group). The align groups come from the worker's
   own specs (`spec.mamba_cache_mode == "align"`, a field only `MambaSpec` carries); an entry
   that was present and whose read FAILED is excluded - that is `disk_restore`'s finding,
   already counted where it happened.

**Where the GPU bytes come from: `disk_shadow`'s seam and no other.**
`disk_shadow._read_gpu_slot` -> `_copy_slot_to_host` -> the host-landing guard ->
`disk_spill.read_slot_bytes`. There is no second device-copy mechanism in this package and
no `ctypes` read of a device address in this module (the module has no `import ctypes` at
all), because the hazard there is a segmentation fault rather than an exception, and because
one seam means the two witnesses cannot disagree about what a failed device read means. The
comparison also waits for the fork's own completion watermark (`worker._load_hwm >= event`):
the apply point runs BEFORE the DMA is issued, so comparing at the apply point would read
bytes that have not arrived; a worker with no watermark gets `no_load_hwm` and is never
compared.

**What it costs, and on which thread.** The engine's step thread pays `observe_apply()`,
which is bookkeeping only - a counter, a sample decision, and a dict of ints plus one path
string. No file I/O and no device read happen on a step thread; both happen on this module's
single lazy daemon thread, bounded to `MAX_CHECKS_PER_PASS` comparisons and
`MAX_RECHECKS_PER_PASS` re-reads per pass with a `MIN_INTERVAL_SECONDS` floor that COUNTS
(`throttled_interval`) instead of sleeping, because a floor that slept would put a wait on
whoever called it. Retention is `MAX_RETAINED_PER_STEP` per apply point inside
`MAX_RETAINED_JOBS` overall, dropping the OLDEST first: an entry that has waited longest has
had longest for its GPU block to be handed out again, so it is the least likely to be
evidence. Entry `i` is sampled every `SAMPLE_EVERY_N_RESTORES` (32nd), counted both ways
(`applies_seen` / `sampled_out`) so the sample rate is a rate over restores that actually
moved bytes.

**A disagreement is classified, never proclaimed.** A GPU block is handed out again at any
time, so a mismatch is re-read before it is reported: a re-read that differs is
`mismatch_gpu_changed_on_recheck` (the engine moved - an expected race), a re-read that held
still is `mismatch_stable` (a layout candidate, the ONLY one an operator acts on), and a
re-read that the device refused is `mismatch_unrechecked`. Collapsing those three would make
the number an operator acts on the number he cannot act on.

**How to turn it on.** It is a WORKER feature, so the environment is the switch that matters:

```bash
#   VLLM_E2_TIER_DISK_RESTORE_VERIFY=1        (read once at first use; restart to change)
curl -s -X POST -d '{"op":"restore-verify"}' $B/disk
```

The POST latches only the process that answers - which for this module is the EngineCore,
NOT the worker that applies a restore - so the answer carries `scope` and says so. It is
POST-only because it changes what the process does (reads device memory), and `op` is a
query parameter on that route, so a `<img>` preload must not be able to arm it. A non-boolean
`enabled` is refused (`JSON boolean` expected) and changes nothing.

**Reading it.** `rank-<rank>.json` gains a fifth contributed section, `disk_restore_verify`
(the seam's allow-list held five sections at the time; 0.2.43 has since taken the sixth slot
for `disk_gpu_write`, and the rank file still carries every field the
proxy host's reader requires inside its own byte ceiling). It publishes the whole counter set
with a stable key set, the geometry it read off its own worker (`window_bytes`,
`group_bytes`, the align group ids, and the note when a config has none), the pending queue,
the last mismatch with its offset and its excluded tail, and the oracle's state - and, since
0.2.40, the SCHEDULER half beside it: the `scheduler_*` counters (since 0.2.41 including the
alloc-hook and expectation-lifecycle counters and `scheduler_expectations_now`),
`scheduler_jobs_now`, `scheduler_mismatch_samples`, `scheduler_oracle`,
`scheduler_oracle_hook`, `roles` and `oracle_rule`. The rank FILE is
written by workers, so the scheduler's numbers reach a reader through `/snapshot` and
`/disk {"op":"restore-verify"}` in the EngineCore; the section is the same object either way.
The verb, `status()` and the section are built by ONE builder, so a number cannot differ by
door.

**Two limits, stated where they cannot be missed.** The truncation shape is counted at BATCH
granularity, so a batch containing two requests that both lost the same align entry counts
once - under-counting, never inventing. And the worker's binding oracle is unreachable in a
worker process on this fork, so `oracle_checked` stays 0 in production while
`oracle_unavailable` counts every verified entry: the recomputation is real code (tested
against the same key helpers the offer path is tested against) waiting for a provider, not a
stub.

## THE OTHER HALF OF THE SAME MODULE: did we offer the RIGHT file? (0.2.40; alloc hook since 0.2.41; the hash view accepted since 0.2.42)

Assertion 4 above - the binding oracle - is unreachable **in a worker**, and for 0.2.39
that was the honest end of the story. It is not the end of the check. The pairing that a
worker cannot see was made one process earlier, by the **scheduler** - and 0.2.40's first
attempt at witnessing it failed in production in a way worth stating exactly, because the
failure is the design.

**What 0.2.40 believed, and what the box said.** 0.2.40 asked at the emission point
(`_build_load_paths`): "this file goes to that GPU block - re-derive the name for that
block's index". Its input was `scheduler_output.kv_connector_block_state`, the fork's
connector-only view of block tables, attached at `sched/scheduler.py:1475` and cleared at
`:1497`. The first armed boot answered every sampled entry with
`scheduler_oracle_unavailable: block_state_does_not_cover_request` and `scheduler_checked`
stayed **0**: zero comparisons, exactly as if the oracle had not been fitted. The reason is
a step-boundary that only a live engine shows. `kv_connector_block_state` carries the block
tables of the requests the CURRENT step scheduled
(`scheduler.py:1412-1425`); `build_connector_meta` emits load states for EVERY request
still sitting in `manager._reqs_to_load` - and a load entry is emitted, and re-emitted, in
steps **later** than the one that allocated its GPU blocks (the request waits for its
`load_event`; the entry rides each step's metadata until the load completes). So at the one
moment the emitter could name the file, the table that would say which GPU block the
pairing used had already forgotten the request. The offer point can name but cannot see;
the alloc point can see but has named nothing. **No single connector call holds both**, and
a witness that pretends otherwise gets the verdict 0.2.40 got: nothing.

**The hook where the pairing is decided: `update_state_after_alloc`.** The fork's
`SimpleCPUOffloadScheduler.update_state_after_alloc(request, blocks, num_external_tokens)`
(`simple_kv_offload/manager.py:381`) IS the moment the pairing is made: it reads the GPU
block tables from its OWN argument (`blocks.get_block_ids()`), pops the CPU-hit record
`get_num_new_matched_tokens` left in `_pending_cpu_hits` (`:401`), gates on
`prefix_cacheable_group_ids` with a `None`-means-all rule, truncates each group to
`cdiv(num_external_tokens, group_block_sizes[g])`, skips `is_null_block` entries WITHOUT
advancing the index, and pairs `group_gpu_ids[num_computed_tokens // g_block_size + i]`
with the i-th CPU block (`:446-470`). Those pairs are handed to the worker as
`load_cpu_blocks` and never re-examined - a wrong pairing here is a file read into the
wrong GPU block, and no later check can see it. So 0.2.41's witness runs HERE, as a
read-only observer fitted by `disk_restore` (`METHOD_ALLOC`), replaying the fork's pairing
loop EXACTLY - the same gate, the same truncation, the same null-skip, the same index
arithmetic - and for every pair the fork will form, it captures the expectation that
pair's file must satisfy: the `(gpu_block, cpu_block)` key, the group, the block index, and
a clipped copy of the engine's own `block_hashes`. It never raises, never mutates, and
pops NOTHING - it peeks `_pending_cpu_hits`, because the fork pops it two lines later and
the next reader is the fork itself.

**Why the wrapper must observe BEFORE the original, which is not how the other wrappers
work:** the fork pops `_pending_cpu_hits[req_id]` at `manager.py:401` and registers the
request in `_reqs_to_load` at `:483-484` inside the same call, so after it returns the
table is gone and only the request survives; before it runs, the table is there and only
the request survives - except that at entry the request is an ARGUMENT. The observer reads
the arguments and the pre-pop pending state, which is the only complete view the hook
offers. (The offer path still passes `scheduler_output` through to the witness unchanged,
but since 0.2.41 nothing reads it there.)

**Then the comparison happens where the name is made** - `_build_load_paths`, unchanged
position, new input. For each entry the fork is about to emit `(cpu_block -> gpu_block)`,
the witness asks the captured map for that exact pair - identity, not position, because
"which GPU block did the fork pair THIS cpu block with" must survive re-emit steps. A hit
carries the group, the index and the hashes the fork itself used, and the verdict is the
same re-derivation 0.2.40 specified: resolve the engine's own
`resolve_block_hashes(captured block_hashes, manager.hash_block_size,
group_block_sizes[group])`, take `view[index]`, pack it with the group through the engine's
own `make_block_hash_with_group_id`, and compare 16-hex digests with the name in the
offered file. **Nothing in this module re-hashes or re-packs**: both calls go through
`disk_restore._key_helpers()`, the same resolver the OFFER path uses, so the code that
wrote the name and the code that re-derives it cannot disagree about which bytes are the
group.

**THE SHAPE OF THAT VIEW, AND WHY 0.2.41 RAN WITH `scheduler_checked = 0` (0.2.42).**
`resolve_block_hashes` does not answer a list. Its declared answer is the union
`BlockHashList = list[BlockHash] | BlockHashListWithBlockSize`
(`vllm/v1/core/kv_cache_utils.py:2850`), and the second member is a **lazy view that is
not a `Sequence`** - it neither inherits from nor registers
`collections.abc.Sequence` (`kv_cache_utils.py:2783-2848`). 0.2.40/0.2.41 checked
`isinstance(view, (list, tuple))`, refused the view, and the live boot reported exactly
what a decorative witness reports: `scheduler_expectations_consumed = 2`,
`scheduler_oracle_refused = 2` with reason
`resolve_shape: ... BlockHashListWithBlockSize, not a sequence`, and `scheduler_checked =
0`. Refusing a shape this module cannot index was the right behaviour; the list of
understood shapes was too short. **0.2.42 accepts the union by capability** - `len(view)`
must answer and integer `view[i]` must exist, both looked up on the `type`, the only place
Python looks them up - and still REFUSES, counted and reasoned, anything without that
protocol, because comparing a shape you cannot index is how a file gets read under the
wrong name.

**AND THE VIEW IS INDEXED, NEVER THE LIST UNDER IT - on this box they differ by a factor
of 80.** The live numbers are `hash_block_size = 40` against
`group_block_sizes = [3200, 3200, 3200, 3200, 8, 3200]`, so a serving group's view has
`scale_factor = 3200 / 40 = 80`: `view[i]` is `raw[(i + 1) * 80 - 1]` and `len(view)` is
`len(raw) // 80`. Indexing the raw list at `i` does not give a near miss - it gives the
digest of tokens `[0, 40)` compared against a file named for tokens `[0, 3200)`, a
DIFFERENT TOKEN RANGE, which would read as a storm of false mismatches or as an accidental
agreement that looks like a passed safety check. So `view[index]` is the only read in this
module and `view.block_hashes` is never touched, and the tests pin it from both sides: the
correct pairing must verify (a raw-indexing oracle calls it a mismatch), and a name built
from `raw[index]` must be REFUSED as a mismatch whose `expected_key16` is the view's; a
second test pins the collapsed length, because block 9 exists in a 640-entry list and does
not exist in the 8-entry view, and comparing past the view is comparing a digest for a
token range nobody named.

**A GROUP THE ENGINE CANNOT VIEW IS COUNTED, NEVER COMPARED.** Group 4's blocks are 8
tokens against a 40-token hash, and the fork's view-builder requires the target to be a
multiple of the hash granularity - it asserts it twice (`kv_cache_utils.py:2826`, `:2887`) -
so asking for group 4 raises `AssertionError`. Group 4 is the parked QSA ring: not
prefix-cacheable, never spilled, so the oracle has no business being asked about it; but
the classification walk enumerates EVERY other group of a mismatching record, so it walks
into that assert on this box on every finding. `_scheduler_view` contains it, names it
(`scheduler_hash_view_unbuildable`, with a reason that says which group and which
arithmetic), and never lets it reach the caller as an exception. A skipped group with no
counter is precisely how a witness goes decorative again, so the counter lives at the one
place that BUILDS a view - every call site, including the walk, is covered by
construction.

**Both negative controls fail loudly, driven through the real hook signature.** A
**one-slot shift** - the failure every list-vs-list check in this plugin is blind to,
because a shift moves the index-aligned lists together and every count still agrees - is
`scheduler_mismatch_neighbour_index`, and the sample carries the SIGNED offset (which way
it moved is the diagnosis). Its mirror is classified with the opposite sign. A **group
swap** is `scheduler_mismatch_other_group`, in both of its shapes: a name whose own 4-byte
group suffix contradicts the group the entry is paired with (a contradiction visible with
NO engine input, so it outranks the digest comparison), and a name with the RIGHT suffix
built from the WRONG GROUP'S HASH VIEW - what a per-group view resolved with another
group's `block_size` produces, which no suffix check can see. The controls are not
simulated: every negative test drives `observe_alloc` at its real signature
`(manager, request, blocks, num_external_tokens)` and a `build_load_paths` entry shaped the
way the fork shapes it, because a control that skips the arguments that decide the
pairing proves nothing. A name that matches none of these is `scheduler_mismatch_unknown` -
reported under that name rather than guessed at.

**The lifecycle of an expectation, because the memory bound lives there.** Captured at the
alloc hook into one insertion-ordered dict, capped at `MAX_SCHED_EXPECTATIONS` entries and
`MAX_EXPECTATIONS_PER_ALLOC` pairs per alloc (an alloc that overflows the per-alloc cap
counts `scheduler_expectations_capped`; evicting an expectation that was never consumed
counts `scheduler_expectations_dropped`). A consumed expectation is MARKED, not removed:
the entry that rode three emit steps must be checkable on all three, and only an
UNCONSUMED expectation can ever be evicted - so the bounded quantity is memory, never
correctness, and the bound holds at the alloc hook, which is the only place a burst of
traffic can pile expectations up before anyone compares them. `scheduler_expectations_now`
is the live gauge. The regression this buys is the production failure itself, and there is
a test for exactly it: alloc in step 1, emit in a LATER step - the check still compares,
because what it needs was captured at the alloc, and the emission step's
`SchedulerOutput` is irrelevant to it.

**What a shift costs, and what an absence means.** A captured expectation whose pair never
appears at the emitter is simply evicted (dropped) - allocs happen for requests whose loads
were sampled out, or cancelled, or never emitted, and that is normal. What must NOT happen
is an entry the emitter sampled but the map cannot answer: that is
`scheduler_oracle_unavailable` with a named reason (`this step's alloc hook captured no
pairing for this entry`, `no CPU hit was pending at update_state_after_alloc`,
`prefix_cacheable_group_ids excluded this manager's groups`,
`num_external_tokens truncated the pair before this entry`, ...). On the SCHEDULER side
that counter means the check is NOT running - an outage, not a fact of the fork. The
worker's `oracle_unavailable` stays exactly what 0.2.39 documented (a worker has no
requests; it grows and means nothing). The two counters are separate keys, the section
publishes `roles` and:

```jsonc
"oracle_rule": "worker oracle_unavailable is EXPECTED (a worker has no requests); scheduler_oracle_unavailable means the check is NOT running"
```

**The section names the hook, because "the oracle is fitted" was the claim that was
false.** `scheduler_oracle_hook` is `"update_state_after_alloc: fitted"` when
`disk_restore`'s observer is on the method and
`"update_state_after_alloc: NOT WIRED here"` when it is not - a renames-the-method fork, a
refused install, an uninstall all say NOT WIRED, and the install path reports its verdict
through `disk_restore_verify.note_alloc_hook()` rather than the section assuming one. The
observer is deliberately NOT part of `disk_restore.SCHED_METHODS`: a diagnostic witness must
never be able to cost the RESTORE layer its install, and it composes ON TOP of whatever
chain the method already carries (a bottom-insert would rebuild the chain through
`backport._rewrap`, which replaces the backport layer it finds there - fitting the observer
that way would silently delete a real patch while `backport` still believed it was fitted).

**What it costs the engine threads.** The alloc hook is the one place this module adds work
to a scheduler path, and it is bookkeeping only: the gate set, a dict `peek`, and at most
`MAX_EXPECTATIONS_PER_ALLOC` iterations storing 5 small ints per pair - no hashing, no
packing, no file or device, no copy of a block table beyond what `blocks.get_block_ids()`
already returns, no engine object retained (the hashes are a clipped copy). The step path
pays what it paid before: a sampling decision before any foreign state is read, and a dict
lookup for sampled entries. All hashing, packing and the neighbour walk happen on the same
single daemon thread the worker half already had; `run_pass()` drains the scheduler's ring
FIRST so a shutdown that takes the worker cannot strand the pairing verdicts.

**What a boot with one caught shift looks like** (the whole section is one object; this is
the scheduler half of it, plus the keys that interpret it):

```jsonc
{
  "scheduler_allocs_observed": 38, "scheduler_expectations_captured": 121,
  "scheduler_expectations_consumed": 119, "scheduler_expectations_dropped": 2,
  "scheduler_expectations_capped": 0, "scheduler_expectations_now": 4,
  "scheduler_steps_seen": 41, "scheduler_entries_seen": 126, "scheduler_sampled_out": 122,
  "scheduler_retained": 4, "scheduler_retained_dropped": 0, "scheduler_jobs_now": 0,
  "scheduler_checked": 4, "scheduler_equal": 3, "scheduler_mismatch": 1,
  "scheduler_mismatch_stable": 1, "scheduler_mismatch_unstable": 0,
  "scheduler_mismatch_neighbour_index": 1, "scheduler_mismatch_other_group": 0,
  "scheduler_mismatch_unknown": 0,
  "scheduler_oracle_unavailable": 0, "scheduler_oracle_refused": 0,
  "scheduler_hash_view_unbuildable": 0,
  "scheduler_index_out_of_range": 0, "scheduler_errors": 0,
  "scheduler_oracle_hook": "update_state_after_alloc: fitted",
  "scheduler_mismatch_samples": [
    {"request_id": "cmpl-a91f3c07b4-0", "group": 5, "index": 12,
     "key16": "0d6a1f47c2b58a90", "expected_key16": "77b2e0c4a19d3f65",
     "name_group": 5, "gpu_block": 1484, "classification": "neighbour_index",
     "offset": 1, "stable": true}],
  "scheduler_oracle": {"unreachable_reason": "nothing unavailable yet", "checked": 4,
                       "equal": 3, "mismatch": 1, "unavailable": 0, "refused": 0,
                       "index_out_of_range": 0},
  "roles": {"worker_apply_point": false, "scheduler_offer_point": true,
            "worker": "worker: the apply point",
            "scheduler": "scheduler: the offer point",
            "meaning": "unprefixed counters = a WORKER's apply point; scheduler_* = the EngineCore's offer point; never add the two"},
  "oracle_rule": "worker oracle_unavailable is EXPECTED (a worker has no requests); scheduler_oracle_unavailable means the check is NOT running",
  "neighbour_window": 2, "max_scheduler_mismatch_samples": 2,
  "reasons": ["scheduler_mismatch:neighbour_index: a one-slot shift (+1): the name carries the digest of block index 13 while the GPU block the fork paired it with is the engine's block index 12. A shift moves both index-aligned lists together, so it is invisible to every list-vs-list check this plugin has"]
}
```

The sample is the newest verdict and there is no separate `scheduler_last_mismatch` key.
`neighbour_window` says how far the classifier looked for the digest the name ACTUALLY
carries - a shift larger than the window is still a mismatch, it simply cannot say by how
much. Stability means something different for this role than for the worker's: the captured
hashes are a COPY, so the re-derivation cannot drift the way 0.2.40's live-object
re-derivation could; `scheduler_mismatch_unstable` now means the pair's captured bytes
disagreed with themselves on the re-walk - a thread bug, which is exactly as alarming as it
sounds.

**Its own named counters, all of them in the section's mandatory key set:**
`scheduler_steps_seen`, `scheduler_entries_seen`, `scheduler_sampled_out`,
`scheduler_retained`, `scheduler_retained_dropped`, `scheduler_thread_unavailable`,
`scheduler_checked`, `scheduler_equal`, `scheduler_mismatch`,
`scheduler_mismatch_stable`, `scheduler_mismatch_unstable`,
`scheduler_mismatch_neighbour_index`, `scheduler_mismatch_other_group`,
`scheduler_mismatch_unknown`, `scheduler_oracle_unavailable`, `scheduler_oracle_refused`,
`scheduler_hash_view_unbuildable` (a group the engine cannot view at its own block size -
group 4's 8 tokens against this box's 40-token hash - which the classification walk would
otherwise skip in silence), `scheduler_index_out_of_range`, `scheduler_errors`,
`scheduler_allocs_observed`,
`scheduler_expectations_captured`, `scheduler_expectations_consumed`,
`scheduler_expectations_dropped`, `scheduler_expectations_capped`.
`scheduler_mismatch_gpu_block_not_in_table` is GONE with the observation point it came
from: it existed because the emitter had to ASK whether a GPU block belonged to the
request; the alloc hook receives the pairing as arguments, so a GPU block outside the
group's table cannot be paired at all, and an entry whose expectation is structurally
unsound is a named `scheduler_oracle_refused`, not a re-derived finding.

**A section that bounds itself.** The budget is `MAX_SECTION_BYTES`, checked before the
section is handed to the seam (which refuses over-cap sections). The mandatory scheduler
content grew again in 0.2.41 - five expectation counters, the hook line and the gauge over
the retired classification - to ~3.5 KiB fresh, so `MAX_SECTION_BYTES` moved from 3800 to
3920 (0.2.42 added one more mandatory counter, `scheduler_hash_view_unbuildable`, and the
fresh section measures 3531 B against that cap, so no further move was needed), and that is
the whole seam story: the cap is 4096, the section ships its mandatory
keys FIRST, and `_fit()` sheds in a stated order (reason text, then the boundary
illustration lists, then the scheduler's mismatch samples oldest first, then the
explanatory prose, then the reasons). `roles`/`oracle_rule` are shed LAST because a reader
who cannot tell which process wrote a number cannot act on it, while a second mismatch
sample is only a nicer-to-read version of a fact the counters and the `reasons` ring carry
in full. A test walks every key of the fresh section against the post-trim view at full
load, so the ladder can never again shed a finding to buy room for prose.

**No new contribution, and that was forced.** As of 0.2.42 `worker_diag.MAX_CONTRIBUTIONS`
was 5 and its allow-list held exactly 5 names; `worker_diag` REFUSES an unregistered
contribution rather
than truncating it, so a sixth section would not have been "not truncated" - it would have
made `disk_restore_verify` the section missing from `diagnostics-rank.json`. Both roles are
therefore the SAME section, told apart by prefix, and the rank file gains no key it could
fail to write. The EngineCore's numbers are readable through `POST /disk
{"op":"restore-verify"}` and `/snapshot` (`dev_state_sections()` publishes the section);
no new op was added, and the scheduler role needs none. (The sixth slot exists now, and it is
the rule this paragraph states that decided how: `disk_gpu_write` took it in 0.2.43 by raising
`MAX_CONTRIBUTIONS` in the SAME change that registered the section, because the alternative was
a diagnostic silently absent from the file. The scheduler role still asks for nothing of its
own.)

`tests/test_disk_restore_verify_scheduler.py` (49 tests) is that spec: the correctly-paired
alloc is CONFIRMED at the emitter (without the positive case every negative case below is
hollow); a shift and its mirror are classified with the right SIGN; a digest from another
group's view and a name with a contradicting suffix are both `other_group`; EVERY negative
control is driven through `observe_alloc(manager, request, blocks, num_external_tokens)` -
the real signature, gate included - and through an entry shaped the way
`build_load_paths` shapes its own; the production regression has its own test (alloc in one
step, load emitted in a LATER step - the check still compares, and the same entry re-emitted
is checked again because a consumed expectation is not removed); the memory bound is proved
at the alloc hook (an expectation-per-alloc overflow and an LRU eviction are both counted,
and the map never exceeds `MAX_SCHED_EXPECTATIONS`); the gate refuses with a NAMED reason
and never a mismatch; `num_external_tokens` truncation, the `None`-gate-all rule and the
null-block skip match the fork's own loop line for line; the OFF flag means the hook counts
one line and stores nothing; the section's key set is identical fresh / after an alloc /
after an emit / after a mismatch / fat, and it fits the seam's cap under the five-section
arithmetic; and two tests drive the REAL `disk_restore.install` wrapper over
`update_state_after_alloc` feeding the fork's own pairing loop and then the real
`build_load_paths` - the hook site pinned by the product, not assumed, including the
install-then-uninstall round trip that must report NOT WIRED again. Since 0.2.42 the same
file holds the LAZY-VIEW section: a replicated `BlockHashListWithBlockSize` (never the
fork's class, never a `Sequence`) driven at this box's real 80-to-1 ratio, which pins that
the digest COMPARED is `view[index]` and never `raw[index]` (from both directions, plus the
collapsed `len(view)` as the bound), that the shift/signed-shift/group-swap controls still
fail through the view, that group 4's unviewable 8-token ring is contained and counted
rather than raised, that a shape with no index protocol is still refused, and that the
plain scale-one list answer is unregressed.

`tests/test_disk_restore_verify.py` (44 tests) is the executable spec: that the default reads
no file and no device and moves no counter (proved with the readers instrumented to RAISE,
not with the module reporting its own silence); that one flipped byte inside `W_g` is
`mismatch_stable` while a difference confined to the tail is `equal`; that a destination one
slot off FAILS rather than passes; that a reused block is classified on the re-read; that a
file which is not window-wide and a group wider than the window are refused without being
compared; that an entry waits for its DMA and a watermark-less worker is never compared; that
the oracle catches a shifted index name-vs-name cannot see, and counts itself unreachable
when no provider is installed; that the apply point touches no file and no device and barely
counts; and that two tests drive the REAL `worker_bind` -> `worker_before_load` path so the
hook site and the field names it reads are pinned by the product rather than by a fake.

## Writing the window ourselves, and watching it land (`disk_gpu_write.py`, 0.2.43-0.2.44)

Entry 28's route has a write side and a read side, and the two verifiers above each read the
GPU. This module is the write side of that route taken for real - and it is the **one dev
feature in this package that writes into live KV memory**, which is why it is default OFF,
why it is sampled, and why the interesting part of it is not the copy but the witness.

```bash
# the worker's switch is the environment, because the worker applies the restore and is the
# only process that can write a window (read once at first use; restart to change):
#   VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE=1
# this op READS: the counters, the pinned pool's live shape, the geometry the comparison
# used and the last verdicts. Its `scope` field says why an EngineCore will report
# `writes: 0` for ever - it has no apply point to write from:
curl -s -X POST -d '{"op":"gpu-write"}' $B/disk
# ... and this one lATCHES the process that answers. For this feature that is the wrong
# process: the write happens at the restore apply point, which only a worker runs, so arm
# the worker with the environment variable above and read the result here.
curl -s -X POST -d '{"op":"gpu-write","enabled":true}' $B/disk
```

**Why a wrong write here cannot corrupt anything.** At the restore apply point the plugin
knows the bytes (`key16`, the spill file, the CPU slot they were just memmoved into) and the
destination (`gpu_block_id`). So this module copies the arena slot into the GPU window itself,
on the tier's own `load_stream`, from inside the `start_load_kv` wrapper's call that runs
BEFORE the fork's body. The fork's own arena-to-GPU copy then still runs, on that same
stream, afterwards - so it has the last word on every byte. A module that guesses the
addressing wrongly puts the right bytes in the wrong place and is immediately overwritten;
what it cannot do is leave the KV cache wrong. The whole point of the feature is that the
mistake becomes *visible* instead of hypothetical.

**The witness is a pair of readbacks, and the file is the oracle.** Both readbacks are
enqueued on the same stream, one before our copy and one after, so on that stream they read
the window as it was before us and as our copy left it. The bytes they come back with are
compared against the **spill file**, never against the CPU slot the copy read from - a
test that compares the readback with its own source says `equal` for every write this module
can perform, which is a mirror and not a witness. Only the leading `W_g` bytes are compared,
for the same reason the verifier gives at length: the rest of the window belongs to other
groups and is stale by construction.

The before-state is what makes the disagreement decidable, and it is why there are two
mismatch counters and not one:

| what the readback says | counter | what it means |
|---|---|---|
| the window holds the file's `W_g` bytes | `readback_equal` | our copy landed where the entry said it would |
| the window is **exactly what it was before us** | `readback_mismatch_stable` | **the one to act on**: our copy went somewhere else. `last_verdict.shape` says whether it is `window-unchanged` (wrong block or wrong region) or `partial-span` (right place, wrong width) |
| the window changed, and not to what we wrote | `readback_mismatch_gpu_changed_on_recheck` | another stream owns the block now - the expected race, and acting on it would mean chasing the engine's normal traffic forever |
| a mismatch with no readable before-state | `readback_mismatch_unrechecked` | real disagreement, undiagnosable; it does not join the row above |

**The ordering is an observable, not a comment, and it has its own pair.** The premise of the
design is that this hook runs before the fork submits its copy, and the fork's own
`worker._load_events` ledger says whether the copy for this load event has been submitted yet.
So `readback_enqueued_before_original` moves only while that premise holds; if the hook ever
moves below the original, `event_already_submitted` and `readback_enqueued_after_original`
start moving, nothing is written, and the regression is loud rather than silent. A worker with
no ledger to read is `no_event_ledger`.

**Pinned rows are the cost, and they are released by the fork's event or not at all.** A
readback destination has to be page-locked or the read is synchronous on the step thread, and
`pin_memory=True` means a `cudaHostAlloc` - so rows are allocated ONLY on this module's own
thread, never at the apply point. The price is that the first apply after a boot finds an
empty pool and refuses; that is counted, published and correct, but since **0.2.44** it is
named as the COLD case (`pool_never_filled`, counted alongside `no_pinned_buffer`) instead of
sharing one name with the opposite diagnosis. A row handed to a `copy_(non_blocking=True)` is
still being written until the load event completes, so rows are held until the tier's own load
watermark passes the entry's event and are released by nothing else. An entry whose event never
passes has an unknowable DMA state: its rows go to `buffers_leaked` and are never handed out
again. That is the direction this module fails in - a witness that cannot afford a buffer
refuses, and a row of unknowable provenance is never a witness twice.

**THE COLD START, AND HOW TO RECOGNISE IT (0.2.44).** Rows exist only because this module's
thread allocates them, and in 0.2.43 that thread was started on the line AFTER a successful
`_take_rows` - so a cold pool refused the write, the refusal returned, and nothing ever started
the thread that could have filled the pool. The first write was unreachable, therefore every
write was: the live rank file said `applies_seen=64 writes=0 checked=0 passes=0 pool_refills=0
buffers_allocated=0 no_pinned_buffer=1 errors=0` and wrote nothing, ever, with `errors=0` the
reason nobody read it as a fault. The fix is both ends of that loop: an ARMED process starts
its writer thread when the module registers its contribution (so before the first restore, and
a thread born that way allocates nothing because no worker is registered yet), and the write
attempt asks for the thread at its TOP, before it touches the pool, so even an unregistered
module starts it on its first refusal. **An operator recognises the dead state in one field:
`liveness.state` is `armed but never wrote`, and `liveness.verdict` spells it out with the
numbers - `passes`, `pool_refills` and `buffers_allocated` all zero with `applies_seen`
climbing is a writer that never ran, and `thread_started=0` is the proof.** `liveness.state`
is one of `off` / `armed, no apply point seen in this process` (an EngineCore: no apply point,
so it can never write) / `armed but never wrote` / `writing`. The one refusal name that used to
carry two facts is now counted apart: `no_pinned_buffer` with `pool_never_filled` beside it
means "the witness thread has not warmed up, the next apply will be written", and `no_pinned_buffer`
alone means the pool filled and drained - rows held by records whose load event has not passed,
or leaked, which never recovers by waiting. The third member of that family, the in-flight cap,
was already separable through `in_flight_dropped`.

**What it costs when it is on**, all of it published in the section: one entry per
`sample_every_n_applied` (64) applies, at most `max_writes_per_step` (1) write per step and
one per `min_interval_seconds` (0.5), at most `max_checks_per_pass` (2) comparisons per pass,
at most `max_in_flight` (4) records waiting for their watermark, at most
`MAX_PINNED_ROWS_PER_REGION` (2) rows per region name and width. The step thread reads
dictionaries and enqueues copies; it allocates nothing, waits for nothing and cannot raise.
Every refusal is a name (`width_mismatch`, `region_pair_missing`, `no_gpu_block`,
`no_group_width`, `row_too_wide`, `stream_unavailable`, `refused_thread_unavailable`,
`write_failed`, ...) and leaves the entry ENTIRELY to the fork's copy, which was always going
to run.

**It copies by name, in the order the file is built in.** `sorted(name)`, GPU region `name`
paired with CPU region `name`, and the per-region widths must total the entry's own byte count
exactly - the fork's `build_params` assertion and `_plan_regions`' total, not a new invention.
A GPU region with no CPU region of the same name is `region_pair_missing`, because the only
other way to pair them is positionally, and one region's bytes in another region's window is
the silent corruption that rule exists to prevent.

`disk_gpu_write` is the **sixth** contributed section in `diagnostics-rank.json`, and the cap
had to move from 5 to 6 to let it exist: the seam refuses an unregistered section, so a sixth
section without the raise would have been silently absent - the worst place for a diagnostic
about live KV writes to be missing, because absence reads as "nothing to report". Six sections
can ask for 24 KiB of the 32 KiB ceiling, which is why contributions are what the byte triage
sheds first.

`tests/test_disk_gpu_write.py` (37 tests) is the executable spec. The default posture is
proved against seams that RAISE if touched, not against a counter agreeing with itself. One
flipped environment variable separates that from a write that puts the file's bytes in a GPU
block and a verdict that says so. The journal of submissions reads `pre-readback, our copy,
post-readback, the fork's copy` as a list, and running the fork first stops the ordering
counter while `event_already_submitted` starts. The negative controls are the point: shifting
only the copy's destination must report `readback_mismatch_stable` (the single counter that
means our addressing is wrong), halving only its extent must report the same counter with
`shape: partial-span`, letting another stream rewrite the window must report
`..._gpu_changed_on_recheck` instead, and moving the FILE must move the verdict with the copy
untouched - because a comparison against the copy's own source could not. The lifetime is
pinned as a fact about a pool: a slow watermark delays the release, provably refuses the next
write for want of a buffer, and ages out by leaking rather than recycling; and the step thread
is proved never to have called the allocator at all. The tail is refused in both directions, a
plan that does not total writes nothing, the two fake cache dicts are deliberately built in
different insertion orders so a positional pairing fails, and two tests drive the REAL
`worker_bind` -> `worker_before_load` path so the hook site, the field names and the order
against the fork's copy are pinned by the product rather than by a fake batch.

### Delivering the window ourselves (0.2.52): delivery mode, and the kill-switch

Everything above this heading describes the writer as a WITNESS: sampled, capped, floored,
and always answered by the fork's copy afterwards. That posture is what made it safe to
write into live KV at all - a wrong address was a wrong witness. Step 3 asks this module to
be the DELIVERY, which removes the safety net, so the mode and a control come together.

`VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY=1` un-throttles it: every applied entry of the step
gets its own copy, and `observe_apply` answers with `written_indices` - the SET of step
indexes it enqueued - beside the count it always returned. `limits_in_force()` is the only
place the three throttles are read, so the sample, the per-step cap and the interval floor
move together or not at all. A half-lifted writer is the trap that function exists to close:
a sampler at 1 with the step cap still at 1 writes one entry, counts `throttled_step` for
the rest, and reads as a broken feature in the counters while the flag says it is on.

**The fourth limit was not in the plan, and it is not a throttle.** The pinned pool holds
`MAX_PINNED_ROWS_PER_REGION` rows per region and a witnessed write holds one row per lane,
so it admits EXACTLY ONE in-flight record. Lifting the three named throttles and leaving the
pool alone yields a writer that still delivers one entry per step and refuses the rest by
name (`no_pinned_buffer`) - a missing delivery that reads as a memory shortage, in the one
configuration where the prune has already removed the fork's copy. The fix is that a
delivery COPY needs no pinned row: its source is the CPU arena slot, not a pinned buffer.
So a row that is unavailable costs the WITNESS and never the write. `delivered_unwitnessed`
counts the copies that went unread back, and it is expected to be most of them - that is
what one apply point delivering a whole step costs.

**The kill-switch is the control that makes the mode runnable, and it fires on one class
only.** A `readback_mismatch_stable` means the window is exactly what it was before our copy
(or ours up to an offset and the old bytes from there), and both readbacks were enqueued on
our own stream around our own copy inside one `start_load_kv`, with the file side being the
file this restore read into the slot this copy read from. There is no second execution in
that comparison, so it is same-execution evidence about OUR addressing, and it latches for
the life of the process: the throttles come back and `disk_restore` refuses every later
prune as `kill_switch_fired`. `readback_mismatch_gpu_changed_on_recheck` names another
stream, and `readback_mismatch_unrechecked` names nothing; neither latches. That is the
0.2.48 lesson of `disk_gpu_shadow` - where a shadow from one execution paired with a store
file from another legitimately differs and every such pair used to be logged as corruption -
applied in a lane where the benign differ is a racing stream rather than a recomputation.
The asymmetry is deliberately OPPOSITE to that module's: there, ambiguity must never be
called benign because being wrong costs a missed bug report; here being wrong about "fault"
costs the feature for the rest of the boot, so anything unattributable is DECLINED
(`kill_switch_declined`) and the switch stays unfired. The arithmetic
`kill_switch_fired + kill_switch_declined == readback_mismatch` holds across all three
categories, including the unrechecked one that returns early from the check.

**The gate is the ENQUEUE and never the readback.** A readback verdict is produced on this
module's daemon thread up to `JOB_MAX_AGE_SECONDS` (30 s) after the copy, while the prune
runs synchronously inside `start_load_kv` and its answer is applied in the same step. There
is nothing to wait for without delaying the forward pass, and nothing to gate on before it
exists. So the enqueue gates and the readback retires the gate later - which is the only
honest reading of "gate the prune on the write being right".

`tests/test_disk_gpu_write_delivery.py` is the suite for all of this. The two tests to read
first are the 1:1 correspondence (five applied entries, five writes, five indexes - the
property the prune's membership test stands on) and `test_an_empty_pool_costs_the_readback_
and_never_the_delivery`, which runs a step with the pool deliberately never filled because
that is what a real first step after a boot looks like.

**A cold-start gap was closed beside it (0.2.53).** Every test in that file inherits the
`witness` fixture's `monkeypatch.setattr(disk_gpu_write, "_ensure_thread", lambda: True)`, and
its end-to-end test additionally calls `fill_pool(worker)` before its first apply. Both are
pieces a real boot does not hand over: the module's thread is started BY the first armed apply,
rows are allocated only inside a refill, and a thread born before a worker exists refills
nothing because `_resolve_worker()` answers None until an apply registers one. So the first
apply of every boot finds an empty pool - the one moment delivery mode's claim ("the cold pool
costs the witness, never the copy") could be false, and the one moment nothing in that file
could assert `thread_started`, because the thread it believed absent was a stub.
`tests/test_disk_gpu_write_cold_delivery.py` (5 tests, 0.2.53) arms the mode with two
environment variables and nothing else: the first delivery of a boot lands and starts the
module's own real thread; the cold boot goes on to WITNESS a delivery (asserted on the thread
names every pinned allocation ran on, so the kill-switch lane is shown reachable and not merely
claimed); an injected wrong copy LATCHES the switch from that cold boot and the consumer's prune
refuses under `kill_switch_fired`; and the last test names both pre-arms and machine-checks that
this file provides neither. Put the sibling's `lambda: True` back and four of the five fail -
that is the measurement of the gap, not a claim about it.

**One generous timeout, not a budget (0.2.57).** Delivery mode lifted the sample, the
per-step cap and the interval floor, and the copy it issues cannot be interrupted: its
source is the fork's pageable arena, so `non_blocking=True` is a lie on this path and the
step thread pays the whole transfer. 0.2.56 answered that with TWO per-entry budgets (a
0.25 s step budget and a 64 MiB byte budget, plus a refusal of a first entry too wide to
fit the byte budget). 0.2.57 keeps the composition and throws the arithmetic away: ONE
wall-clock bound per apply point (`DELIVERY_APPLY_TIMEOUT_SECONDS` = 10 s, overridable by
`VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY_TIMEOUT_SECONDS`, read per apply point), checked
BEFORE each entry, and a stop is still a `continue` - so the deferred entries never enter
`written_indices`, the prune cannot remove them, and the fork's copy serves them. The
byte budget and `delivery_refused_entry_too_wide` are deleted outright: a 11,468,800 B
window is ~2 ms of PCIe against a 10 s bound, so the branch could only fire on a box
whose PCIe had already stopped, and a counter that can never move is worse than no
counter. The operator's rule is the reason, and the numbers agree with it - fine-grained
caps on the disk read/write path used to strangle throughput until they were replaced by
one timeout - and a 0.25 s cap would have deferred most of every large restore, which is
to say the guard would have quietly switched the feature off. The two keys that say
whether the bound has ever bitten survive from 0.2.56:
`delivery_mode.step_seconds_last` / `step_seconds_peak`, now read beside the published
`delivery_mode.timeout_seconds`. `tests/test_disk_gpu_write_step_budget.py` is the
reproduction: 48 entries at a fake copy that COSTS what a pageable copy costs, on a
cold-boot fixture, on a stand-in step thread with a bounded join - so the unfixed loop is
caught as "the apply point never returned" in 1.5 s instead of as a CI hang.

## Silence is not death: three clocks that were reading the wrong thing (0.2.45)

The operator's rule for this store is one sentence:

> **LRU/quota is the only deleter; silence is not death.**

Every file in the store obeys it - there is no TTL on a cached block, and the ceiling
pass is the only thing that unlinks. Three OTHER clocks in the tier did not obey it, and
all three cost real cache or real service on a quiet box. What they had in common: each
one treated "I have not heard from X recently" as "X is gone". A rank that has not
written recently is not a dead rank. An index entry nobody touched recently is not a
file that left the disk. A worker that has taken no step recently is not a dead worker.

**1. A quiet rank's bytes were the first casualty of a sibling's ceiling pass.** A rank
that has not spilled for a while has no fresh digest in its peers' maps: the peer map
aged its digest out at `PEER_DIGEST_FRESH_SECONDS` (120 s), the walk then found a file
named by a digest nobody could confirm, classified it `ATTRIBUTION_DEAD` ("a config-change
orphan"), and shed it in bucket 0 - ahead of the passing rank's OWN cold bytes. Measured
on the unpatched tree: a quiet rank with a live worker had 3 of 3 of its files unlinked
as dead bytes. A quiet rank is not a gone rank.

The fix separates *liveness* from *geometry*, because the two questions were being
answered by one lookup. Liveness now comes from the rank files (`rank-*.json`, which every
live worker rewrites every 60 s) and not from the peer digest map, so a rank that is alive
but quiet is LIVE and its current files are never `ATTRIBUTION_DEAD`. A foreign-GEOMETRY
file stays sheddable - the `stale_geometry` path is untouched and still runs ahead of this
rank's own cold bytes, because a file no engine on this box can read is genuinely dead
capacity. `peers_known` is derived, fail-closed: an empty peer map claims nothing, so
`attribute_name` refuses to call an identity dead on no evidence, while a rank that can
see its own rank files CAN reclaim real orphans.

*New, and deliberately separate so the two doctrines cannot be confused:*
`stale_geometry_removed` counts files removed as unreadable-geometry orphans (a config
change), `peer_files_removed_live` counts the unforgivable one - a live sibling's CURRENT
bytes - and on the fixed tree the first moves and the second stays exactly zero.
`identity_known` / `identity_available` / `liveness_source` in `disk.peer_eviction` name
which evidence the pass actually had, so "why did it delete that?" is a reading.
Tests: `tests/test_quiet_rank_bytes_are_not_dead_bytes.py`.

**2. The disk index forgot bytes that were still on disk.** `disk_index` capped its
identity map at a hard-coded count and evicted in INSERTION order, so a burst of restores
could push out the identity of the file that was just restored (the expensive one to
relearn) while a never-touched identity from the boot sweep stayed. The cap was also not
configurable: the store grows past it in normal operation.

Eviction is now LRU-by-**touch** (`OrderedDict.move_to_end`), a touch is recorded on the
same event that bumps a file's `st_mtime` (`disk_read.promote_recency`), so the file clock
and the index clock cannot drift apart, and the cap is
`VLLM_E2_TIER_DISK_INDEX_ENTRIES` (default 262,144 - about 90 MB at the measured 342
bytes per entry). A just-restored identity is the LAST to leave.

*The counter that makes the loss honest:* `entries_dropped` is now balanced exactly by
`entries_dropped_lru` (evicted by policy), `entries_dropped_unverified` (never checked
against the directory) and `entries_dropped_unparsable`. Distinguishing them matters
because only one of them means "we forgot a file that is still there":
`entries_dropped_file_present` counts the index forgetting a live file, and
`note_restored_path()` re-admits an identity a restore just proved exists. A drop is never
guessed at - an entry the bounded presence scan did not check is `unverified`, not
"absent". Tests: `tests/test_index_forgets_no_bytes_on_disk.py`.

**3. An idle engine stopped serving from disk.** `disk_restore` may only offer a disk
restore when a WORKER has attested that its restore layer is fitted (separate processes;
offering without the proof means `start_load_kv` DMAs CPU slots nobody filled and the
request is served from STALE KV). The attestation is the rank file and the reader refuses
one older than `RANK_FILE_FRESH_SECONDS` (900 s). Both halves are correct. But the file
was rewritten only by `get_finished` - an ENGINE STEP - so an idle worker's proof aged
out and the tier closed its disk path on a healthy box, reopening only once a burst of
steps arrived. Which is the worst possible moment: the first request after an idle spell
is the one that wanted the disk.

`worker_diag` now keeps its own proof alive from a bounded daemon thread
(`kvtier-worker-diag-heartbeat`, `HEARTBEAT_INTERVAL_SECONDS` 60 s = fifteen beats inside
the 900 s window, write floor 5 s; since 0.2.51 that interval and that floor are both kept
on `CLOCK_BOOTTIME`, which a suspend cannot freeze - see *Silence is not death, part
three*), started at `install()` and NOT at the first step - the
0.2.44 cold-start lesson, because an idle worker is precisely a worker whose step-driven
starter never runs. The beat goes through the same `write_snapshot` a step uses, so a
reader can never tell the two files apart. The handler is held as a WEAK reference (a
diagnostic writer must not pin the CPU KV arena), and a beat with no handler still proves
liveness, because the FACT the file carries is "this process is alive".

What it does NOT do is weaken the gate: the window stays 900 s, so a worker that has
genuinely gone stops being believed on time. A PROOF IS NOT DATA - nothing here keeps a
cached byte alive, it keeps a live worker legible.

*New counters:* `heartbeats`, `heartbeat_threads`, `heartbeat_respawns` (a thread that
died and was re-armed by the next step), `heartbeat_floor_skips`, `heartbeat_errors`,
plus `heartbeat_alive` / `last_heartbeat_age_seconds` / `heartbeat_beats_in_window` in the
section, which answer "is this worker's proof being kept alive, and how tight is it?"
without arithmetic. Tests: `tests/test_idle_engine_keeps_its_disk_proof_alive.py`.

**And what that fix could not see (0.2.47).** The 0.2.45 suite passed on a box where the
tier still went dark, for two reasons that are each other. The suite drove `install()`
directly, with the worker class handed to it and the interval shortened; in production
the worker's install lands on the handshake, and the ONLY thing that re-arms a dead
supervisor is `heartbeat_status(force=True)` inside the wrapped `get_finished` - an engine
step, which is exactly what an idle worker does not take. And every field that could
answer "is the pulse running?" lived in the process that served the snapshot, which for
this plugin is the API server: a worker answers no HTTP route, so `heartbeat_alive`,
`last_heartbeat_age_seconds` and `beats_in_window` read `false` / `None` / `0` about a
process that has no rank file to keep, and `heartbeats: 0` in a snapshot is a fact about
the API server, never about a worker. The one artifact a worker definitely writes is the
rank file, and it carried no heartbeat state at all. See *Silence is not death, part two*
below for what 0.2.47 changed.

## The disk tier went off by itself on a quiet box (0.2.45; corrected 0.2.47)

> **A rank file is a proof of life, and a proof of life needs a pulse - and the pulse has
> to be READ where the worker writes it.**

If `gate().reason_kind` is `no_worker_proof` (or the disk path simply stopped serving
after a quiet period) while the engine is healthy and idle, the rank files are stale, not
absent. Since 0.2.45 each worker writer keeps its own file alive on a 60 s beat - so the
question is whether that beat is running, which is a read:

```bash
# The worker's own answer, per rank. SINCE 0.2.47 the state is IN THE FILE: a worker
# answers no HTTP route, so `writer.heartbeat` is the only place its own pulse can be
# read. A stamp older than 900 s with a live engine is the symptom, not a fault.
ssh llm 'for f in /tmp/kv-tier-diagnostics/rank-*.json; do
  python3 -c "import json,sys,time
  d=json.load(open(sys.argv[1])); h=(d.get(\"writer\",{}).get(\"heartbeat\") or {})
  print(sys.argv[1].rsplit(\"/\",1)[-1], int(time.time()-d[\"ts\"]), h.get(\"state\"),
        \"beat_writes=\"+str(h.get(\"beat_writes\")),
        \"stalled=\"+str(h.get(\"stalled_beats\")), \"jammed=\"+str(h.get(\"jammed_beats\")),
        \"respawns=\"+str(h.get(\"respawns\")),
        \"written_by_beat=\"+str((d.get(\"writer\") or {}).get(\"steps_since_write\", 0) > 0))" "$f"
done'

# And WHEN the gate shut, with what evidence: one line per transition in the ENGINE log
# (in production `disk_restore` runs in the engine process, not in the API server).
ssh llm "grep -a 'DISK\[proof\] GATE' /tmp/vllm-serve.log | tail -20"
```

* `state` is the one-word answer: `beating`, `alive_not_beating` (a live thread that is
  not writing - the state that was invisible before 0.2.47), `started_not_alive`,
  `never_started` (with `state_reason`, the reason `install()` used to swallow), or
  `not_applicable` (no worker writer in this process). `beat_writes` frozen while `state`
  is not `beating` IS the outage.
* `written_by_beat: True` on a fresh file is independent corroboration: a step-driven write
  RESETS `writer.steps_since_write`, so a fresh file whose `steps_since_write` is above the
  write interval was written by the heartbeat and not by a step.
* **Do not read `heartbeat_alive`, `heartbeats`, `heartbeat_threads`,
  `heartbeat_respawns`, `heartbeat_errors`, `last_heartbeat_age_seconds` or
  `heartbeat_beats_in_window` from `snapshot` as the worker's pulse.** That block is read
  in the process that served the HTTP request, and `installs: 0` inside it says outright
  that this process fitted no worker writer: for that process the fields are `false` / `0`
  / `null` whether or not every worker on the box is beating. Measured on a live box with
  four fresh rank files written by a running beat: `heartbeat_alive: false`, `heartbeats:
  0`, `heartbeat_threads: 0`, `last_heartbeat_age_seconds: null`. `heartbeat_beats_in_window
  : 15` is `900 // 60`, arithmetic about the shipped cadence, and it reads the same in a
  process where nothing beats at all.
* `heartbeat_beats_in_window` remains the MARGIN, in the rank file now as `beats_in_window`:
  the shipped 60 s inside 900 s is fifteen beats, so an operator who raises
  `VLLM_E2_TIER_WORKER_HEARTBEAT_SECONDS` is one missed tick from closing the disk path
  again, and this field makes that visible before it happens.
* **Before 0.2.45 this symptom had no cause to find.** An idle worker simply wrote nothing
  and the 900 s window elapsed. Nothing was broken; a heartbeat was missing. Before 0.2.47
  the CAUSE was still unfalsifiable, because the fix's own counters lived in a process that
  answers no request - see *Silence is not death, part two* above.

## Presence is per rank: a peer's journal cannot decide what my rank may read (0.2.46)

**The store keeps one file per rank; the index answered as if it kept one.** A parked span is
`kvslot-<key16>-g<group>-<geodigest12>.bin`, written by EVERY armed rank under its own geometry
digest, and a restore reads all of them. The index kept `rank_lines` per identity, but the
answer the offer path asked was `disk_index.contains()`, which read ONE global entry table - and
a journal REMOVE line called `_drop_identity_ranks(identity)`, retiring every rank's claim
instead of the claim of the rank whose journal the line arrived in. Two failure directions, both
live on the box:

| shape | mechanism | what the operator saw |
|---|---|---|
| **false present** | rank A's `- <key16>` unlinks its slice; rank B's next `+` for the same identity resurrects the global entry; `contains()` says present | a read offered for a file rank A deleted: `failed={'file_missing': 5}`, `hole_in_middle` |
| **false absent** | rank A's `- <key16>` dropped the entry AND every peer's claim; the peers' later `+` lines then read as "a removal of something nobody holds", so their claims went unmatched | spans whose files were all on disk, refused as `span is not fully indexed` |

**Measured read-only on the box first, by replaying the journals' own per-rank lines** (the rank
was never missing information - it is in the FILE NAME of the journal that carries the line):
**1,768 removal-retired (identity, rank) claims across 446 unique identities**, `disk_used =
146,643,927,040` of a 298 GiB store. So "~442 identities per rank" and "~1,768 claims" are ONE
population counted at two units: the identity count is 446, and 1,768 is 446 x ~3.96 ranks. No
identity was found resurrected by a peer while its own file was absent from the store, so the
loss that box was taking was one-directional - withheld cache, not a failed read.

**The rule now.** A REMOVE line retires the claim of the rank whose journal it appeared in
(`_drop_rank_claim`) and nothing else; the entry follows only when the LAST claim goes, and
`bytes_indexed` is debited once against the entry's own size. The offer predicate is an
INTERSECTION: `disk_index.contains(key, ranks)` is true only when every rank in `ranks` holds a
claim AND the entry exists. Truncation at a hole needed no new walk logic - `_disk_run`'s
positional scan already stops at the first miss, and the miss is now the HOLE instead of the end
of the span, so a span with one rank's slice missing keeps offering its long prefix instead of
losing the prefix and the run.

**Who the intersection runs over, and where that set came from.** The offer path asks with the
ranks the worker proof ARMS (`worker_proof()["ranks"]` -> `_disk_run(ranks=...)`), because those
are the ranks that will execute the load: a dead rank's leftover journal cannot veto the living
ones, and an armed rank that has spilled nothing vetoes everything. A proof whose rank list came
back TRUNCATED is not proof of a rank set, so that case falls back. With no caller-named set the
index uses the ranks its own journal FILE NAMES declare (`kvslot-index-<RANK>.journal`), a set
that grows monotonically within a boot - forgetting a rank would RELAX the intersection, and
relaxing it is the direction that hands out missing files. The set and its provenance are both
published (`ranks_expected`, `ranks_expected_source` = `"worker proof"` | `"journal files"`) in
`disk_index.section()`, `counters()`, `restore.index` and the `no_disk_run` refusal: "we expected
4 and 3 showed up" is a different diagnosis from "we expected 3".

**`op=contains` gained the verdict for the in-between case.** An identity some ranks hold and
others do not answers `verdict: "partial_ranks"` with `present: false`, `offerable: false`,
`ranks_missing: ["PP1_TP1"]`, `ranks_present: 3` and the surviving entry's `bytes` / `journal` /
`rank` rows beside it - "three ranks hold this and PP1_TP1 does not" is a different fact from
"nothing holds this", and only the first has a fix. The refusal says the same thing in the run's
own data: `no_disk_run` carries `stop_index` and `stop_detail`, e.g. `block 3 of group 5 is held
by 3 of 4 rank(s) that have to read and is missing for PP1_TP1`, appended to the sentence the
operator reads. Computing it costs ONE `presence()` call, at the stop index, on a path that has
already decided not to serve.

**Counters - every one counted at apply time, none from a scan.**

| counter | unit | a non-zero value means |
|---|---|---|
| `index.claims_removed` | (identity, rank) claims | a REMOVE line retired a claim that existed: the honest count of the events the old code applied globally |
| `index.claims_kept_past_peer_removal` | (identity, rank) claims | a PEER's REMOVE line arrived for a claim this rank still holds, and the claim SURVIVED. This is the false-absent rate, measured; on a 4-rank box it runs ~3.96 per withheld identity |
| `index.rank_claims_unmatched` | journal lines | a REMOVE for an (identity, rank) nobody holds: it retires nothing, moves no byte, debits nothing |
| `index.inserts_incomplete_span` | journal lines | a `+` that landed an identity not every expected rank holds - an offerable=false span in the making |

**The audit is a sample and says so.** `sampled_runs`, `sampled_identities`,
`sampled_partial_ranks`, `sampled_offerable`, `sampled_claims_without_entry` and `sampled_last`
come from `disk_index.audit_now()`: at most `AUDIT_SAMPLE_IDENTITIES` (64) identities per pass,
once every `AUDIT_EVERY_N_REFRESHES` (4) refreshes, doing NO I/O - no `stat`, no listing, no
journal read, a dict walk only. Its payload carries `sampled: true` with the sample size; a
`scan_identities` payload carries `sampled: false`. Nothing in this counts every identity: a
census of 262,144 of them on a serving path is the thing this module exists to avoid.

**`op=identities` is the enumeration half of the diff.** Bounded (default 50, hard cap
`MAX_IDENTITIES_LIMIT` = 200 identities per page), paged (`after` is the cursor the last page
returned), filtered (`rank`, and `which` = `all` | `offerable` | `partial`) and DIGEST-ONLY: a row
is `{identity, ranks, ranks_missing, offerable}` - an identity (`<key16>:g<group>`), the rank
labels whose journal claims it, which of the expected ranks do not, and whether the offer path
may promise it. Never a path, never a size walk: the digest each rank named its file with is the
layout's geometry digest for that rank, and the journal line (`<seq> <+|-> <key16> <bytes>
<group>`) cannot know it, so pairing a row with an `op=ls` line is the operator's join. The verb
walks no store, moves no counter and reads no file, and it is not in `MUTATING_DISK_OPS`.

```bash
B=http://127.0.0.1:8000/plugins/kvtier/disk
# 1. what the index intersects over, and what the last sampled pass saw
curl -s "$B?op=status" | jq '.restore.index | {ranks_expected, ranks_expected_source,
    claims_removed, claims_kept_past_peer_removal, rank_claims_unmatched, inserts_incomplete_span,
    sampled_last}'
# 2. the index's per-rank claims: 200 identities, digests only, no paths
curl -s -X POST -d '{"op":"identities","limit":200}' $B | jq '.identities[0]'
# 3. keep paging until `truncated` is false - that is the census, not this page
curl -s -X POST -d '{"op":"identities","limit":200,"after":"<next from page 2>"}' $B
# 4. only the withheld spans; only one rank's claims
curl -s -X POST -d '{"op":"identities","limit":200,"which":"partial"}' $B
curl -s -X POST -d '{"op":"identities","limit":200,"rank":"PP1_TP1"}' $B
# 5. the STORE's side of the same question, by shard: real FILE NAMES, one per rank
curl -s -X POST -d '{"op":"ls","shard":"a3f0","limit":50}' $B
```

Both directions of the `identities`-vs-`ls` diff mean something, and the verb's `next_step`
says which is which: an identity reported WITHOUT a rank for which `ls` shows that rank's digest
is a WITHHELD file (the false absent, and the ~19.5 GiB of parked KV behind it); an identity
reported WITH a rank whose file `ls` no longer shows is a FALSE PRESENT, the one that reaches a
worker as `file_missing`. On a box that is behaving, the second list is empty and the first
matches `claims_kept_past_peer_removal` growing at the rate the ceiling pass reclaims.

**Three existing tests changed, each narrowed to what is actually true.**
`test_a_removal_line_drops_the_entry_and_the_ranks_together` asserted the entry died with the
first removal while its own docstring said "the per-rank view goes on carrying the rank that
wrote it" - it now asserts the entry survives with three ranks, `contains()` still refuses the
GLOBAL question, and `ranks_missing` names the one rank. `test_a_truncated_journal_forgets_that_rank_and_only_that_rank`
asserted the same all-ranks-died shape and that `membership` reports no bytes; the entry's bytes
are real (the other ranks' files are on disk), so it now asserts the surviving three, the named
missing rank and `bytes > 0`. `test_one_deep_journal_cannot_starve_another_rank` asserted that an
identity a lagging rank has not reached is NOT read at all; it is read - and the new semantics
say what the old test meant, that it is not OFFERABLE until the lagging rank's own journal proves
its file, which is the same protection with the correct subject.
`test_disk_restore_scope.py`'s four-rank fixture changed too: it armed four ranks and spilled
from one journal, which is a rank set that does not exist outside a test; the fixture now mirrors
the spill into all four journals so the span is genuinely complete and the geometry/eligibility
rules that test exists to check are what it checks.

**What it does NOT do.** No journal format change (the rank was already in the filename), no
`stat` per entry on the serving path (the live numbers are sampled, and named `sampled_*`), no
new `worker_diag` contribution (the cap of 6 is untouched), and `contains()`/`membership()` only
gained fields. Tests: `tests/test_disk_rank_presence.py`.

## Silence is not death, part two: the beat was running and the proof still died (0.2.47)

> **"The thread is running" is not a proof of life. A file being written is.**

The box that already had the 0.2.45 heartbeat went dark again: a **16.8-minute idle gap
(1008 s, past the 900 s window) and then `gate_reason_kind: "no_worker_proof"` on the one
request that followed it.** The plugin could not say whether the heartbeat was running at
the time, and - this is the part that made it a code change and not only a logging change
- it could not have said, in any artifact, by any means. What was read off the live box
(read-only, with the engine serving traffic) explains both halves.

* **Four workers, four fresh rank files, and a beat that DOES run.** All four of
  `PP0_TP0`, `PP0_TP1`, `PP1_TP0`, `PP1_TP1` were 0.1-0.2 s old, each carrying
  `writer.writes` 2717 / 2716 / 2710 / 2710 with `writer.steps_since_write: 102`. Sampled
  again six seconds later, `writes` and `steps_since_write` were unchanged while the age
  had grown to 6.2 s - and because a step-driven write RESETS `steps_since_write` to 0, a
  file written 0.2 s ago reading `steps_since_write: 102` can only have been written by
  the heartbeat. **The thread starts in production and it beats.** The operator's
  hypothesis ("it never starts in a worker") is wrong, and the fix is not a starter.
* **The snapshot's `worker_diag` block is about the wrong process, always.** The same
  moment, from `GET /plugins/kvtier/snapshot`:

  ```
  "worker_diag": { "configured": true, "dir": "/tmp/kv-tier-diagnostics",
      "installs": 0, "heartbeat_started": false, "heartbeat_alive": false,
      "heartbeats": 0, "heartbeat_threads": 0, "heartbeat_respawns": 0,
      "heartbeat_errors": 0, "last_heartbeat_age_seconds": null,
      "heartbeat_window_seconds": 900.0, "heartbeat_beats_in_window": 15 }
  ```

  `installs: 0` in that same block is the tell: the process that served the snapshot
  fitted no worker writer, so it has no rank file to keep and every liveness field in it
  is `false` / `0` / `null` **by construction, whether or not all four workers are
  beating**. This is precisely the read the README told the operator to take, and it
  answers a question no worker was ever asked.
* **The rank file - the only artifact a worker definitely writes - said nothing.** The
  live files' `counters` are
  `['load_events_pending', 'restored_hits', 'store_events_pending', 'stored']`: there is
  **no `heartbeats`, no `heartbeat_errors`, no `heartbeat_respawns` in a worker's rank
  file at all** (the 0.2.45 runbook command reads `counters.heartbeats` and gets `None` on
  every rank on this box). The nearest thing to a pulse was `writer.writes` - a monotonic
  write count that says "no write happened" when it freezes and never says WHY.
* **The refusal itself has no timestamp anywhere.** The engine log's witness line for that
  boot (pid 2386, engine started 10-08 23:21:34) reads
  `refused=no_boundary_anchor:51,no_disk_run:208,no_worker_proof:1,suppressed_after_failure:4`.
  One `no_worker_proof` refusal, a monotonic counter, and no record of when it happened or
  how old the freshest rank file was at that instant - which is the only question an
  idle-spell outage asks. (This box also runs `VLLM_E2_TIER_WORKER_WRITE_INTERVAL_STEPS=50`
  and a 1.0 s write floor, so its step path writes far more often than the defaults; the
  beat only ever matters when the steps stop.)

**Why the mechanism dies anyway.** Three defects, all found by reading the code against
the failure:

* **The only re-arming was driven by the thing whose absence is the fault.** A beat is
  issued by `heartbeat_status(force=True)`, and its sole caller with `force=True` is the
  wrapped `get_finished` - an ENGINE STEP. A supervisor that died during an idle spell
  stayed dead for the whole spell. This is the 0.2.44 cold-start lesson ("started at
  `install()` and NOT at the first step") wearing a third outfit: the STARTER moved to
  `install()`, the RE-ARMER never left the step path. It is also why the 0.2.45 suite
  passed: `test_a_thread_lost_at_cold_start...` deliberately *takes a step* to bring the
  thread back, so it asserts the very design that cannot recover unattended.
* **A beat ran inline in the supervisor's loop.** A contributor that blocks - six other
  modules are asked for their section, and their lock may be held across a long NVMe pass
  - stops the proof exactly as dead does, while `alive` keeps answering True because the
  thread is alive and parked. No budget, no in-flight bound, no counter for it.
  `heartbeat_alive: true` beside a stale rank file was a reachable, guaranteed state.
* **The exception classes were wrong at all three levels.** The beat, the contributor loop
  and the supervisor each caught `Exception`. A `SystemExit` (a C extension, a library
  that decides to exit) escaped the beat, escaped the containment and ended the thread -
  and then needed an engine step to come back.

**What changed.** A beat is now *issued*, not *run*: the supervisor's loop only launches
short-lived beat threads (`kvtier-worker-diag-beat`), each bounded by
`HEARTBEAT_BEAT_BUDGET_SECONDS` (20 s) with at most `HEARTBEAT_MAX_INFLIGHT_BEATS` (3)
alive, so a wedged beat costs one tick and never the tier; a beat that outlives its budget
is counted (`heartbeat_stalled`) and, past the bound, named (`heartbeat_jammed`); and the
loop catches `BaseException`, because with beats off-loop the loop is the last line. The
contributor loop catches `BaseException` per contributor: **a contributor is a guest, it
may cost its own section and nothing else** (asserted with a `SystemExit` contributor whose
rank file keeps moving). `start_heartbeat()` re-opens the beat gate that `stop_heartbeat()`
closes, so a supervisor cannot be re-armed into silence.

**And what the worker now says about itself.** `writer.heartbeat` is a new block in the
rank file - the one artifact a worker definitely writes - with `state` as the one-word
answer and a FIXED key set in all five states:

| `state` | meaning |
|---|---|
| `beating` | alive, and its last beat wrote |
| `alive_not_beating` | **alive and not writing**: budget, gate, or the floor refused it; `state_reason` says which |
| `started_not_alive` | it started and its thread is gone; `respawns` says whether anything has re-armed it |
| `never_started` | the start was refused, with the reason that refused it |
| `not_applicable` | no worker writer in this process, so there is no proof to keep alive |

Beside it, under the names the FILE uses (the ones below are the contract; a runbook key
that does not exist prints `None` and reads as a fault): `beats` / `beat_writes` /
`inflight_beats` / `last_beat_age_seconds` / `last_beat_duration_seconds` / `errors` /
`fatal_beats` / `stalled_beats` / `jammed_beats` / `floor_skips` / `respawns`, the window
the reader will apply (`window_seconds`) and how many beats fit inside it
(`beats_in_window`), and the writer's own `writer_pid`, so a reader can ask whether that
process still exists.

From 0.2.51 the same block also answers *where the beat's time comes from* and *who woke
it*: `cadence_clock` (the short name of the clock the deadline is kept on - `boottime`,
or `monotonic` on a host without it), `ticks` (wake-ups that led to a beat), and
`beats_from_clock` / `beats_from_step` / `beats_after_resume` - a partition of the ticks,
so `beats_from_clock + beats_from_step + beats_after_resume == ticks` holds while this
build is the writer. Since 0.2.69 there is ONE BEAT NO TICK ISSUED - the cold-boot proof
the worker handshake writes (`engine_side._prime_worker_proof`) - and it is counted under
its own name, `beats_from_prime`, rather than folded into a cause that would make the
suspend statistics describe an event that did not happen: the partition above is
UNCHANGED, and `beat_writes` is bounded by `ticks + beats_from_prime`. `beat_writes` is
NOT part of the partition identity: an attributed beat can still be refused by the write
floor, so `beat_writes <= ticks + beats_from_prime`, and the gap between them is how many
beats were decided and then not written. `refusals` counts the times the beat had something to write and
could not (no directory, no rank), and `resumes` the resumes the cadence clock has
noticed. Every one of these is `0` rather than absent when nothing has happened, and the
key SET is identical in every state - including the shell of a file written before this
build - so a runbook question cannot fail because of a state. The age beside them
(`last_beat_age_seconds`) is measured on `cadence_clock`, while `worker_proof()` ages the
FILE by the wall clock, because a file's `mtime` is stamped in wall-clock seconds whether
or not the writer's clock believed the suspend.

**The refusal now has a timestamp.** `worker_proof()` logs **one line per transition**
(never per poll - the `notice_once` rule this repo already holds for the sleep-notice and
the settle-refusal), with the reason kind, the per-file counts, and both the age and the
wall-clock time of the freshest rank file:

```
DISK[proof] GATE AT BOOT open=False reason_kind=no_worker_proof ranks=0/0 rank_files=0 ...
DISK[proof] GATE OPEN again after 0.4s closed; ranks=2/2 rank_files=2 fresh=2 stale=0 ...
DISK[proof] GATE CLOSED reason_kind=no_worker_proof ranks=0/0 rank_files=2 fresh=0 stale=2 \
    youngest=965.2s last_proof_written=2026-10-08T23:34:11 (965.2s ago); reason: ...
```

**The boot posture changed in 0.2.69.** The first line above is what a boot prints when
nothing has proved itself yet, and before 0.2.69 EVERY boot printed it, because the first
rank proof was due a full interval after install. A worker now primes its proof at the
handshake, so an ordinary boot prints `GATE AT BOOT open=True reason_kind=open rank_files=1
fresh=1 ...`; the closed posture is still the true reading of a boot whose prime failed, or
of a scheduler asked ahead of every worker, and it is still one line and never a `GATE
CLOSED`.

A rank file that is too OLD to count and a rank file that is FRESH BUT SAYS NOTHING (no
`disk_restore` section: a contributor raised, overran its byte cap, or the restore layer
never fitted) are both `no_worker_proof`, and from the scheduler's side they were one
number - because the reason for the second lives in the worker's `_contrib_state`, again in
a process nobody polls. They are now `stale=N` and `sectionless=N` on that line, and
`stale_rank_files` / `sectionless_rank_files` / `youngest_rank_file_age_seconds` /
`youngest_rank_file_written_at` in every `worker_proof()` and `gate()` verdict. The gate's
edges are counters too: `disk_restore.counters()` publishes `proof_gate_open`,
`proof_closures`, `proof_openings` and `proof_last_transition`, so a monotonic closure
count across a window names the first refusal instead of hiding it.

**The commands, and what each can honestly see.** The rank file is on the GPU box and needs
no engine call. The gate's log lives in the ENGINE log, because in production
`disk_restore` runs inside the engine process, not in this plugin's API server:

```bash
# 1. THE PULSE, per rank, in the one artifact a worker definitely writes. `state` is the
#    answer; `beat_writes` frozen while `state` is not `beating` IS the outage.
ssh llm 'for f in /tmp/kv-tier-diagnostics/rank-*.json; do
  python3 -c "import json,sys,time
  d=json.load(open(sys.argv[1])); h=(d.get(\"writer\",{}).get(\"heartbeat\") or {})
  print(sys.argv[1].rsplit(\"/\",1)[-1], int(time.time()-d[\"ts\"]), h.get(\"state\"),
        \"writes=\"+str(h.get(\"beat_writes\")), \"stalled=\"+str(h.get(\"stalled_beats\")),
        \"jammed=\"+str(h.get(\"jammed_beats\")), \"pid=\"+str(h.get(\"writer_pid\")),
        \"reason=\"+str(h.get(\"state_reason\")))" "$f"
done'

# 2. THE GATE'S OWN TIMELINE: when it shut, on what evidence, and when it came back.
ssh llm "grep -a 'DISK\[proof\] GATE' /tmp/vllm-serve.log | tail -20"

# 3. THE POSTURE AND THE TALLY, from the process that decides.
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/tier?op=status' \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['disk_restore']; c=d['counters']
  print(d['gate']['open'], d['gate']['reason_kind'],
        {k: c[k] for k in ('proof_gate_open','proof_closures','proof_openings') if k in c},
        c.get('proof_last_transition'))"
```

* `state: "beating"` with a `ts` younger than 900 s means the proof is being kept. If the
  gate is shut anyway, the answer is in the `disk_restore` section of the same file
  (`installed`, `worker_fitted`, `not_configured_reason`) or on the GATE line - not in the
  heartbeat.
* `state: "alive_not_beating"` is the state that used to be invisible: a live thread that
  is not writing. Read what refuses it, all of it in the same block: `stalled_beats`
  climbing with `last_beat_duration_seconds` over 20 s is a beat parked on someone else's
  lock; `inflight_beats` at 3 with `jammed_beats` climbing is the supervisor having
  stopped launching; `floor_skips` climbing is the write floor refusing beats; and since
  0.2.51, `refusals` climbing with `state_reason` naming the cause is a beat that reached
  its write and declined it (no diagnostics directory, or a rank this process cannot
  derive) - the two refusals that used to vanish without a counter.
  **Since 0.2.51 this state is a TRANSIENT AT STARTUP, not an outage state**: the first
  beat now counts itself before it writes, so a healthy worker's file says `beating` from
  its very first beat; if a file says `alive_not_beating` more than one interval after
  the boot, one of the four counters above is moving.
* `state: "never_started"` plus `state_reason` is the boot-time answer this repo could not
  get before: the reason `install()` used to swallow, now in the file and, when the
  diagnostics directory is the cause, in the engine log once at WARNING.
* **`heartbeat_alive` in `snapshot` is NOT the worker's pulse.** Read `installs` in that
  same block first: for a deployment whose HTTP surface is the API server it is `0`, and
  then `heartbeat_alive: false`, `heartbeats: 0` and `last_heartbeat_age_seconds: null` are
  facts about a process that has no rank file to keep. `heartbeat_beats_in_window: 15` is
  arithmetic, not evidence - it reads the same in a process where nothing beats.
* `writer.writes` and `writer.steps_since_write` remain the cross-check: a file that is
  fresh with `steps_since_write` above the write interval was written by a BEAT (a step
  write resets that field), which is how the four ranks above were shown to be beating
  with no steps in flight.

**What this does NOT do.** The 900 s window is unchanged and the gate keeps its teeth: a
worker that is genuinely gone stops being believed on time (a stopped beat, a
future-stamped file, and a stripped `disk_restore` section are each asserted CLOSED by
`tests/test_idle_worker_proof_stays_alive_and_legible.py`). The beat's cadence is
unchanged at 60 s, so the fifteen-beats-inside-900 margin is exactly as it was - the
supervisor makes a beat *reliable*, never more frequent. Three beats in flight is the
ceiling, and past it the supervisor stops launching rather than parking a thread per tick
for the life of the process. And nothing here makes an idle worker produce steps: it
removes the step path from the recovery of a proof that was never supposed to depend on
one.

Tests: `tests/test_idle_worker_proof_stays_alive_and_legible.py` (24 tests, driven from
`engine_side.register()` plus the handshake hook exactly as a worker does, with zero engine
steps, and the READER's clock moved so a test can wait fifteen minutes).

## Pruning the pairs an adopted restore no longer needs (`disk_restore.py`, 0.2.49)

Four steps retire the host-RAM arena. Step 1 (0.2.43-0.2.44) made the plugin able to write a
GPU window itself and *witness* that write. **Step 2, this release, is the flag that stops
asking the fork to copy a restored block at all**: when a disk restore is adopted and its
bytes have landed, the entry is removed from both load lists at the same index, in one place,
behind a default-OFF variable. **Step 3 (0.2.52) makes a plugin-side write the *delivery* of
those bytes** (`VLLM_E2_TIER_DISK_GPU_WRITE_DELIVERY`, see *Delivering the window ourselves*
above) and gates this flag's population on it; Step 4 deletes the arena. What Step 2 alone
does depends entirely on Step 3 now: with the delivery flag unset the prune refuses every
step as `not_delivered` and changes nothing at all, which is the safe direction and the
reason the two flags are a pair rather than a sequence. Reading the flag on and the delivery
flag off is a normal staged rollout - the fork keeps every destination and
`prune_uncovered_entries` tells you how much the plugin has not taken over yet.

```bash
# The worker's switch is the environment: the WORKER applies the restore, so only a worker can
# prune (read once at first use; restart to change). Unset = off = today's behaviour, byte for
# byte, and no `prune_*` counter moves.
#   VLLM_E2_TIER_DISK_PRUNE_RESTORED_LOADS=1
# READ-ONLY look at the scheduler-side view. An EngineCore has NO apply point, so it answers the
# switch and zeros, and says so; the counters that MOVE are in the worker's rank file, below.
# Every `/disk` answer carries the restore verdict under `restore`:
ssh llm 'curl -s http://127.0.0.1:8000/plugins/kvtier/disk'   | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['restore']['load_prune'], indent=2))"
# THE COUNTERS THAT MOVE, from the rank files (the WORKER is the process with an apply point):
ssh llm 'for f in /tmp/kv-tier-diagnostics/rank-*.json; do python3 -c "
import json,sys
d = json.load(open(sys.argv[1])).get("disk_restore") or {}
print(sys.argv[1], {k: v for k, v in d.items() if k.startswith("prune")})" "$f"; done'
```


**Why this flag is different from every other one here.** Every previous dev feature fails slow
or fails silent-but-harmless. This one can fail *wrong*, because the two lists the restore ships
are a positional pair: `copy_blocks` pairs `src[i]` with `dst[i]` and has no way to notice that
one list got shorter (`copy_blocks` at `cuda_mem_ops.py:227`; its descriptor arrays are built
from the two id lists independently at :239-248, and `total = len(src_block_ids) * num_layers`
at :248). Remove an entry from `load_cpu_blocks` only, and
block K's bytes are written into block K+1's window - no crash, no error, a confident wrong
answer. Everything below exists to make that impossible rather than unlikely.

**One door, one rewrite, and a refusal that hands back nothing.**
`prune_load_pairs(cpu, gpu, cpu_drop, gpu_drop=None, groups=…, expected=…)` is the only place a
load pair is removed, and the apply point makes exactly one call to `_replace_load_lists` per
step, so a failed read's removal and a prune's removal cannot be decided against two different
snapshots of the same pair. The two index parameters exist *so that* a one-sided call is
constructible and detectable: `cpu_drop != gpu_drop` is refused (`one_sided_prune`), as are a
step whose sides already disagree (`length_mismatch`), an index outside the step
(`out_of_range`), a step over `MAX_PRUNE_ENTRIES_PER_STEP` (256) (`over_cap`), a per-group count
that no longer matches on the two sides (`group_delta_mismatch`) and an entry whose ids are not
the ones this module recorded at bind (`destination_drift`). Every refusal is in-band - a
returned `reason_kind`, never an `assert`, because `python -O` strips asserts and the cost here
is a wrong answer - and a refusal returns **no lists**, so a caller cannot ship a refused prune
by accident. The step then falls back to the pre-0.2.49 rewrite, byte for byte.

**Gate B: the index set is READ, never computed.** Group `g`'s entry at index `i` binds to
`block_ids_by_group[g][num_computed // group_block_sizes[g] + i]` (`manager.py:447-468`), and on
the live model four `boundary_state` groups contribute `[null] * (B - 1) + [real]` - one entry
at index `B - 1`, because a run's boundary state lives at its END - while the full-attention
group contributes all `B`. So an index derived from tokens, block sizes or an id range mis-names
a boundary group's entry, silently. The prune's population is therefore the bound jobs
themselves: a job already carries `cpu_block_id`, `gpu_block_id` and `group`, keyed by its index
in the lists. The pairing check re-reads those ids after choosing the set, and
`pruned_by_group` / `group_deltas` are published over **every** group the step attributed -
touched or not - so a reader can tell "group 1 lost nothing" from "group 1 is not reported".
There is a second, stronger reason the differing-size case cannot reach the prune: the offer
gate refuses a model where any prefix-cacheable group's `group_block_sizes[g]` differs from the
scheduler's (`_group_serviceability`, `incomplete_group_set`, "a slot for it would land at the
wrong GPU offset"), so every group whose bytes can appear in these lists is at the scheduler's
block size. That is why the Gate-B test uses the box's real Qwen3-Next sizes: this code cannot
build a 1,600-token boundary group, and the test that pins the refusal is part of the proof.

**What is never pruned.** An entry whose read failed (the apply point removes it and reports
its GPU block through `take_invalid_block_ids()` exactly as it always has, so the engine
recomputes those tokens; the flag changes neither the drop count nor that id set - a test runs
the identical scenario both ways and compares them). An entry the scheduler shipped without a
path (its bytes belong to the RAM tier). An entry the scheduler did not adopt: the fork takes
`cdiv(num_external_tokens, group_block_sizes[g])` leading entries per group, so "offered N,
shipped M" is the normal shape on this box and the prune's population is M - the published
chain is `pruned <= entries_in_step <= jobs_bound <= blocks_offered`.

**The hazard this release cannot fix, and the number that prices it.** The fork's copy is today
the only thing that carries a restored block's bytes into its GPU window: `disk_gpu_write` is a
*witness*, sampling one entry in 64 at most one per step. Remove the pair and those bytes reach
no device at all - the request is answered from uninitialised KV. So the flag publishes its own
cost instead of hiding it: `prune_unwitnessed_entries` (pruned entries with no plugin-side write
behind them), `prune_emptied_steps` (prunes that left the fork nothing to submit, where the load
event is *released* rather than recorded so a later completion cannot make this one reportable -
the phantom rule, guarded by `phantom_guard_trips`), and one `notice_once` line per episode,
never one per step. **Since 0.2.52 `prune_unwitnessed_entries` is expected to be ZERO and
cannot move: the prune's population is filtered by membership in the writer's enqueued set
before anything is removed, so a NON-ZERO reading is a bug report about the membership test,
not a condition to be watched.** (In 0.2.49/0.2.50 the sentence above was the opposite - it
was arithmetic on a count, `witnesses < candidates`, and a boot with three sampled writes and
thirty adopted entries looked clean while the gap was twenty-seven.)

Two spellings, because the two payloads have different budgets. A **worker rank file**
(`disk_restore.prune`) carries the compact block a byte-capped file can afford - `on`,
`steps`, `entries`, `refused`, `skipped_failed`, `unwitnessed`, `emptied` - and `on` is that
rank's own reading of the variable, which is the only question a rank can answer honestly:
the process that prunes is the process that applies a restore. The **verb / `section()`
side** carries the long names below, the closed refusal vocabulary with every key always
present, the per-group maps and the last decision (`load_prune`).

| what to watch | it means | act when |
|---|---|---|
| `prune_entries`, `prune_steps` | pairs removed, apply points where the switch was consulted | never on its own; these are the volume |
| `prune_unwitnessed_entries` | **a broken coverage gate**, since 0.2.52: it cannot move by construction | ALWAYS, at once - a non-zero reading names entries served from uninitialised KV |
| `prune_uncovered_entries` | adopted, read entries the plugin did NOT write, left to the fork: the missed optimization | if it climbs while `gpu_write.writes` is flat, the writer is refusing or off, not the step lacking disk work |
| `delivery_mode` / `kill_switch_fired` on the prune side | whether this process's own copy is the delivery, and whether a proven-bad copy retired the flag | `kill_switch_fired: true` means the plugin stopped delivering for the rest of the boot and the fork's copy is carrying everything |
| `prune_emptied_steps` | a step left with nothing to copy, so the load event was released | with any `disk_restore_verify` disagreement, or while `delivery_mode` is false |
| `prune_refused` (closed vocabulary, all keys always present) | the prune was asked and declined: the step kept every destination it had | never a fault by itself; `lists_not_in_step` / `destination_drift` mean the step arrived unlike anything the module understands |
| `prune_skipped_failed_entries` | bound entries the prune did not touch because their read failed | this is the recompute property, counted: it should track your read failures |
| `disk_restore_verify.mismatch_stable`, `disk_gpu_shadow` counters | the byte-level oracles for the restore path | any movement with the flag on is the experiment telling you the pair mattered |

**Shape discipline.** A refused prune answers with the same keys as an accepted one, and every
time the switch is consulted the step's answer replaces `last_prune` - so a refusal can never
leave the *previous* step's numbers looking like the current ones. `prune_switch()` publishes
which way this process decided, from where, and whether the value was recognised at all; a typo
(`tru`) is OFF, reported once, and is the direction that changes nothing.

**Tests.** `tests/test_disk_prune_restored_loads.py` (22 tests, 0.2.49) is the spec, ordered by
what hurts: off-is-the-default and off-is-unchanged (the expected lists are recomputed by the
pre-0.2.49 algorithm, not snapshotted); the shift constructed and refused (and the shift it
prevented computed, so the refusal is not a formality); both-sided correctness asserted as a
PAIRING and never as a length; Gate B, both halves - the offer gate that keeps a differing-size
group unreachable, and the live hybrid geometry where the pruned GPU ids are recomputed from the
fork's per-group formula and the "B leading entries per group" rule is shown to name ids that
are not in the step at all; adopted-not-offered; failed-read-keeps-its-range (asserted off vs on
as an equality of the load-error id set); and the shape/log discipline. Two tests drive the real
`worker_bind` -> `worker_before_load` path, so the hook site and the order against the fork's
copy are pinned by the product and not by a fake batch.

## Silence is not death, part three: the clock the beat waits on (0.2.51)

**The report this answers.** After a boot, all four rank files read `beats: 0`,
`beat_writes: 0`, `floor_skips: 0`, `last_beat_age_seconds: null`,
`state: "alive_not_beating"` about seven minutes into a quiet engine, and one request
later they read `beats: 1`. The theory offered was *"beats are issued only while the
engine is taking steps"*. **That theory is refuted, and the mechanism is the box's
suspend.** The supervisor has had its own clock since 0.2.45 (`_heartbeat_loop` waits,
then issues one beat); no engine step has ever issued a beat. The live rank files said so
while the engine stepped roughly once a second:

```
beats       17 -> 18 in 60.0 s      one beat per interval, not per step
floor_skips  0                      no step ever tried to beat through the 5 s floor
```

**Where the silence came from.** Three clocks are involved and they do not agree across a
suspend. `threading.Event.wait(timeout)` - what the beat loop waited on - is
`CLOCK_MONOTONIC`, which on Linux **does not advance while the system is suspended**.
The gate that consumes this file ages it on a **wall clock** (`disk_restore.py:3410`,
`age = time.time() - ts`), which charges the entire sleep to the stamp. Measured on the
GPU box on 2026-10-09: `CLOCK_BOOTTIME - CLOCK_MONOTONIC = 19,612 s` with
`suspend_stats/success = 2` - the box had been asleep for five and a half hours of its
seven-hour-old engine's life. So the deadline a beat was waiting on was still unpaid when
the box woke, the beat that had to follow the wake was refused by a write floor measured
on the clock the sleep had just skipped, and the file reported
`last_beat_age_seconds: 3.0` next to a gate reading it as many minutes stale. The first
request after a wake - the one that wants a disk restore more than any other - is the one
the gate closes on, with `no_worker_proof`.

**What the beat cadence is now.** Still one bounded wait on a `threading.Event` in a
thread, and still never inline on a caller's thread. Three changes:

| | 0.2.50 | 0.2.51 |
|---|---|---|
| the deadline | `monotonic + interval` (frozen by a suspend) | `BOOTTIME + interval` (cannot be frozen) |
| the wait | one `Event.wait(interval)` | the same wait, polled in slices of `HEARTBEAT_RESUME_POLL_SECONDS` (5 s) |
| a resume | invisible | detected from BOOTTIME gaining what MONOTONIC did not, and **beaten for at once** |
| the write floor | `monotonic` (did not expire across a sleep) | the cadence clock, so a directory untouched for a whole sleep is not "too recently written" |
| a step's role | nothing | may `wake_heartbeat()` when the supervisor is behind its deadline; never runs a beat |

**What is NOT claimed.** A box that has slept longer than `RANK_FILE_FRESH_SECONDS`
wakes with a stamp that is stale on arrival, and nothing can un-age it: the gate is closed
at the wake by design, because during the sleep the worker was frozen and proved nothing.
What 0.2.51 changes is how long it stays closed with nobody at fault - up to one interval
plus the floor (about 65 s shipped) down to about one poll (5 s), and to less still when
the first engine step after the wake finds the supervisor behind its deadline. The first
request that lands inside that window is still refused a restore and recomputes. A
follow-up, deliberately not attempted here because it is not contained: teach
`worker_proof()` that a stamp older than the window but younger than the box's own
suspend interval is "asleep, not dead", which needs a source of suspend truth the reader
can trust.

**The keys 0.2.51 added** (to `writer.heartbeat`, in EVERY state, and to
`heartbeat_status()`): `cadence_clock` (`"boottime"` on the box, `"monotonic"` where the
kernel has no BOOTTIME - it names the clock every age in the block is kept on), `ticks`
(the supervisor's own wakeups), `beats_from_clock`, `beats_from_step`,
`beats_after_resume`, `refusals` (a beat that reached its write and declined it, now
named in `state_reason`), `resumes`. 0.2.69 adds `beats_from_prime` beside them, in the
same block and in every state: the count of rank proofs written by the cold-boot prime,
which is the one write no supervisor tick issued (see the identity above).

> **Is the idle path alive? Read `beats_from_clock`.** It counts only the beats the
> supervisor issued because its OWN CLOCK EXPIRED. `beats` totals every cause and
> `beats_from_step` counts beats a busy engine would have written anyway, so neither
> answers the question; `beats_from_clock` rising while `counters.steps` is frozen is the
> idle path, measured.

**After you deploy and restart, confirm it with these three reads** (all read-only, none
touches the engine):

```bash
# 1. THE CLOCK, and the idle path, per rank. Run this after the engine has been idle for
#    more than one interval (60 s) with no traffic through the proxy.
#    `find`, not a glob: the box's login shell is zsh and a glob with no match aborts the
#    whole command line, which is the worst possible failure for a diagnostic.
ssh llm 'find /tmp/kv-tier-diagnostics -maxdepth 1 -name "rank-*.json" -type f \
    | sort | while read -r f; do
  python3 -c "import json,sys,time
d=json.load(open(sys.argv[1])); w=d.get(\"writer\",{}); h=(w.get(\"heartbeat\") or {})
print(sys.argv[1].rsplit(\"/\",1)[-1], h.get(\"state\"),
      \"from_clock=\"+str(h.get(\"beats_from_clock\")),
      \"ticks=\"+str(h.get(\"ticks\")),
      \"cadence=\"+str(h.get(\"cadence_clock\")), \"resumes=\"+str(h.get(\"resumes\")),
      \"age=\"+str(h.get(\"last_beat_age_seconds\")),
      \"refusals=\"+str(h.get(\"refusals\")),
      \"reason=\"+str(h.get(\"state_reason\")))" "$f"
done'
#    WANT: state=beating, cadence=boottime, and from_clock RISING on every rank while no
#    request is in flight. `from_clock` frozen at 0 with ticks moving means the beats are
#    being refused - `refusals` and `reason` now say by what.

# 2. THE CLOCKS THEMSELVES, to see what the beat is being kept on:
ssh llm 'python3 -c "import time; m=time.clock_gettime(time.CLOCK_MONOTONIC); b=time.clock_gettime(time.CLOCK_BOOTTIME); print(\"mono %.0f boot %.0f suspended %.0f s\"%(m,b,b-m))"'

# 3. THE RESUME, named once in the engine log, and the gate around it:
ssh llm "grep -a 'WORKERDIAG] this process woke from a' /tmp/vllm-serve.log | tail -4"
ssh llm "grep -a 'DISK\\[proof\\] GATE' /tmp/vllm-serve.log | tail -8"
#    WANT: one line per resume naming the sleep length, followed by GATE OPEN within a
#    few seconds and with no request in between.
```

**One reading that is a trap, and is now closed.** A rank file cannot report a counter it
has not yet incremented, so until 0.2.51 the FIRST beat's own file read
`state: "alive_not_beating"` with `beats: 0` - a window as long as the interval (60 s
shipped) during which a perfectly healthy worker's rank file said the idle path was dead.
That is the reading the report above was made from. The beat now counts itself before it
writes and gives the count back if the write fails, so `beating` is true from the first
beat.

**The rule "nothing a beat can do may end this loop" applies to log lines.** Writing the
resume note found the hole: the note sat outside the loop's `try`, so a note that raised
ended the supervisor on its way past the counter it had just incremented. What that
leaves is the worst artifact in this file - `resumes` moved, so the state says the resume
was seen; the log is empty, so nothing says it; `fatal_beats` is 0, so the counter for a
killed loop reports nothing; and the thread is gone. A note is formatting plus whatever
its arguments read, which is exactly as fallible as a write, so it is guarded like one
and the failure is reported through `snapshot._error` instead of swallowed.

Tests: `tests/test_heartbeat_beats_on_a_clock_the_suspend_cannot_freeze.py` (15 tests:
the cold start with a DERIVED rank - waiting for the supervisor's OWN clock beat, since
the first file on disk is the prime's - the gate reopening after a suspend with nothing
arriving (in a quiet window: the supervisor restarted behind a long deadline and the air
drained, so the resume detector is the only thing that can reopen the gate), a resume beat
not locked out by a floor a racing write spent, the quiet period longer than the freshness
window, the step
that wakes a beat off the stepping thread (counted as deltas around the restart's own
immediate first beat), the refusals that used to vanish, the partition of the ticks plus
the prime's term, and a raising log line that must cost a line and nothing else).

## The truncation gate: is a claimed span left with nothing to pair? (`disk_restore.py`, 0.2.53)

`SimpleCPUOffloadScheduler.update_state_after_alloc` trims the per-group CPU hit lists it was
handed before it pairs anything (`manager.py:437-454`):

```python
n_take_g = cdiv(num_external_tokens, g_block_size)
cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])
```

It claims `num_external_tokens` external tokens and pairs the **leading** entries it kept with
`group_gpu_ids[gpu_ext_start + i]` (`manager.py:460-478`). So a group whose list is LONGER than
its take has trailing entries the slice discards - and on this box's geometry that is not a
hypothetical. Groups 0-3 are align-mode GDN groups at a 3,200-token block, and the list
`_merge_into_pending` builds for one is `[null]*(B-1) + [ONE state block]`: the entry at the END
is the only real block, because the fork's pairing loop preserves the index of a skipped null
and that is what lands the recurrent state on the run's LAST GPU block rather than its first.
Trim that list below `B` and the nulls survive while the state is discarded, and the tokens are
still claimed as loaded - a GDN/SSM recurrence resumed from a GPU block nobody wrote. That is
the **2026-10-06 00:08** answer: wrong output, no exception, nothing in a log.

The equality this depends on - `n_take_g == len(list[g])` - is maintained by two files this
repository does not own (`MambaManager.find_longest_cache_hit` sizes an align group's list by
the ACCEPTED length, and the scheduler adopts what `get_num_new_matched_tokens` reported) and
by nothing else. No test in the fork pins it. **0.2.53 is the observer that says when the two
stop agreeing, on the plugin's side of the boundary.**

**No flag, because a gate is not a feature.** It runs inside the `update_state_after_alloc`
wrapper, BEFORE the fork's body - the only moment the numbers exist, because the fork POPS
`_pending_cpu_hits[req_id]` inside that body (`manager.py:415-429`). Every other hook in this
module is fitted AFTER the fork (a restore must not run before the fork decided); this one is
fitted first and still returns the fork's value untouched. `disk_restore_verify.observe_alloc`
sees the same inputs and asks "does the pairing line up?" behind an env switch; this asks "is a
claimed span about to have no bytes behind it?" on every step, armed or not, verifier on or
off.

**Not every gap is the bug, and the gate does not pretend otherwise.** A partial adoption that
trims our NULL PADDING discards no bytes: the group keeps a real block inside the claim, the
answer is right, and the restore is simply shorter than offered. Refusing that would refuse the
fork spending its own token budget, and a gate like that gets switched off inside a day. So the
tail is inspected, not assumed, and only one reading is fatal - `dropped_real > 0` (a real entry
went away) **and** `kept_real == 0` (nothing real is left in the claim). That combination is
`unbacked_claim`, and it is the only thing the gate refuses on.

| counter | it means | act when |
|---|---|---|
| `group_take_observations` | alloc observations the gate measured. **THE DENOMINATOR**: it is what makes `dropped_groups: 0` read as "audited and clean" instead of "never looked" | never on its own; this is the proof the others mean something |
| `group_take_dropped_groups` / `_entries` | groups (and entries) where the fork's slice drops trailing entries, whatever they are made of | rising beside a flat `group_take_unbacked_groups` is the fork trimming padding: a cost, not a bug |
| `group_take_unbacked_groups` | **THE WRONG-ANSWER COUNT**: a real entry dropped AND no real entry left in the claim | ALWAYS, at once. A non-zero reading means a claimed span has no bytes behind it |
| `group_take_refusals` | disk offers the gate withdrew for an unbacked claim, also counted as `refused[group_hit_truncated]` | with the reason in `OFFER_REFUSAL_WHY`; this is the refusal that costs a recompute instead of an answer |
| `group_take_unavailable` | observations the gate could not measure (no `prefix_cacheable_group_ids`, no `group_block_sizes`, no pending hit) | a clean zero and a blind pass are different readings, and this is the key that keeps them apart |
| `last_group_take_gap` | the arithmetic of the last sighting in the fork's own field names: group, its `group_block_sizes` entry, `hit_len`, `n_take`, `dropped`, `dropped_real`, `kept_real`, `unbacked_claim`, and whether an offer of ours was in the tuple | read it before believing any of the counters above |

**What the gate does about it, and what it honestly cannot.** At the alloc door the scheduler has
already adopted `num_external_tokens` and advanced the request's computed count, so the plugin
cannot un-claim a span from outside the fork, and rewriting the pending tuple there would leave
GPU blocks the engine counts as computed with nothing in them - the same hazard, self-inflicted
and unreported. Two levers remain, and both are pulled: the request's disk fuse is pulled
(`suppress_load`, so the next step cannot offer it into the same shape a second time - the same
loop-killer that stopped 123,474 offers in 3.9 minutes on 2026-10-05) and the claimed-but-
unwritten GPU blocks of the UNBACKED groups go onto the load-error channel, the same one a failed
disk read uses and the only one that makes the engine recompute instead of reading a block as KV.
The range reported is the claimed span `[gpu_ext_start, gpu_ext_start + n_take)`, never the
dropped tail's own blocks: those sit outside the claim and reporting them would truncate an
unrelated request that later holds one. It logs **one line per episode**, never one per alloc
(this is the alloc path, once per scheduled request per step); a clean observation closes the
episode, so "happened, stopped, came back" is three lines and not one. The counters keep moving
the whole time - the line is rate-limited, the evidence is not.

The refusal reason is in the existing taxonomy: `group_hit_truncated` is a member of
`OFFER_REFUSALS` with an `OFFER_REFUSAL_WHY` entry, so `refused[group_hit_truncated]` and
`last_refusal` carry it without a new payload key.

**Tests.** `tests/test_disk_take_truncation_gate.py` (28 tests, 0.2.53 + 0.2.54 + 0.2.55) opens
with the pure
predicate - exact take, shorter list, one entry longer, a tail of nulls, an unmeasurable
geometry - and then the door itself. The test to read first is
`test_the_mutated_padding_line_moves_the_truncation_counter`: it does not hand the gate a
hand-built bad tuple, it re-executes the shipped `_merge_into_pending` with ONE line changed -
the align group's null padding, `range(max(0, spans - 1))` becoming `range(max(0, spans))`, the
line whose whole job is making an align group's list exactly as long as the fork's take - and
runs the SAME offer on the live hybrid geometry as its own control. Control: the gate audits five
cacheable groups and finds nothing. Mutant: the four GDN lists are five long, the take is four,
the discarded entry is each group's only state block, `kept_real` is 0, and every counter above
moves. The mutation is pinned to the shipped source (it fails loudly if that line moves), and it
is reverted by restoring the one attribute it touched rather than by unwinding `monkeypatch`,
which would also tear down the spill root the test is standing on. The rest of the suite covers:
the fuse plus the exact GPU-block range reported (and that the attention group's span is NOT
reported, and that the drain drains); one-line-per-episode with the episode closing on a clean
alloc; a foreign claim (a tuple with no `_loads` entry) reported but NOT refused; nonsense shapes
(string entries, `None` sizes, an argument-less request) coming back as a status and never an
exception; and the wiring itself asserted through `make_wrapper(METHOD_ALLOC)` - the fork's body
pops the tuple, and the gate still saw it.

### Reading the gate in production (`load_prune.group_take_*`, 0.2.54)

0.2.53 shipped the gate with a hole in it: **its counters were not readable on the wire.** They
lived in `disk_restore._state`, and `counters()` - which copies that dict wholesale - is rendered
by no HTTP route. The route publishes `disk_restore/scheduler_section()`, whose `load_prune` block
is `prune_state()`, and `prune_state()` assembles its keys by hand from the `prune_*` names and had
never heard of the gate. Verified live on the running 0.2.53 box: `GET /plugins/kvtier/snapshot`'s
`disk_restore.load_prune` carried 21 keys, none of them the gate's, while its own `units` prose
described numbers the document did not contain. The suite was green throughout, because every test
in it asserted `disk_restore._state` - a counter in the state dict and a counter in the published
document are two different facts, and only one of them can be read during an incident.

0.2.54 publishes them, as pass-throughs, under the same names. **`group_take_refusals` is
deliberately not among them**: that event is already in the document once, as
`refused[group_hit_truncated]`, incremented from the same two lines, and a second name for one
event is how a reader ends up counting it twice.

```bash
# READ-ONLY, and needs nothing: the snapshot route is gated on the peer address alone.
# The box's shell is zsh: every path is quoted and no glob is allowed to fail.
ssh llm "curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot" \
  | python3 -c 'import json,sys
 d = json.load(sys.stdin).get("disk_restore") or {}
 p = d.get("load_prune") or {}
 for k in ("group_take_observations", "group_take_dropped_groups",
           "group_take_dropped_entries", "group_take_unbacked_groups",
           "group_take_unavailable", "last_group_take_gap"):
     print(f"{k:29}", p.get(k))   # .get(): a 0.2.53 or older build prints None, and says so
 print("refused[group_hit_truncated]  ", (d.get("refused") or {}).get("group_hit_truncated"))
 print("role", d.get("role"), "installed_here", d.get("layer_installed_here"))'
```

**What was actually observed on the live box** (read-only, just now, on the deployed 0.2.53): the
five counters and `last_group_take_gap` printed `None` - the hole, on the wire, in six lines - and
the one gate fact the build *did* route, `refused[group_hit_truncated]`, read **0**. `role
scheduler+worker`, `layer_installed_here True`. On 0.2.54 the same command reads six zeros plus
that same `0` on a box that has served traffic all day, and that is the expected healthy reading.

**A zero there is a measurement of "the gate never fired", not of "the gate is working."** The
proof it is working is the denominator beside it: `group_take_observations` must be climbing (it
moves once per alloc that reaches `update_state_after_alloc`), and the blind looks must be
attributable - see the next section, because 0.2.54's instruction to keep `unavailable` at 0 was
wrong. And a zero in an answer from a process that never allocs means "this process is not the
one that takes": the gate runs on the scheduler-side alloc path, which is why the numbers are
published where they move (`load_prune` in the route's and the verb's answer) and NOT in a worker
rank file, where they could only ever be permanent zeros in a byte-capped file.

### What `unavailable` means, now that it names its cause (`load_prune.group_take_unavailable_reasons`, 0.2.55)

0.2.54 shipped the total and nothing else, and it went blind on a healthy boot five minutes after
the deploy: `group_take_observations: 2`, `group_take_unavailable: 1`, `last_group_take_gap:
null`, and no line in the engine log. The cause was not recoverable from any artifact the boot
produced, because the code counted the blindness and threw away the reason - `_take_gate()`
returned a bare `None`, and the pending-hit branch beside it incremented **the same counter**.

There are **seven** distinct ways an alloc observation goes blind, and they are not the same
finding:

| reason | what it means |
|---|---|
| `no_group_block_sizes` | the object has no `group_block_sizes` - **unaudited** |
| `no_cacheable_group_set` | sizes, but no `prefix_cacheable_group_ids` and no `block_ids` - **unaudited** |
| `gate_read_failed` | reading the fork's gate raised (the exception is in the log) - **unaudited** |
| `no_pending_hit_table` | the manager has no `_pending_cpu_hits` at all - **unaudited** |
| `empty_request_id` | the request carried no id, so no lookup could run - **unaudited** |
| `pending_hit_wrong_shape` | the table's entry is not the fork's tuple - **unaudited** |
| `no_pending_hit_for_request` | **the ordinary case**: this request had no host-RAM hit, so the fork had nothing to pair |

`sum(group_take_unavailable_reasons.values()) == group_take_unavailable` always: the breakdown is
a partition of the total, so **never add the two**. And the reading that the total alone could not
support is now the first thing the payload says - `unavailable: 3` with all three named
`no_pending_hit_for_request` is a healthy boot serving cache misses, while `unavailable: 3` with
`no_group_block_sizes: 3` is a take invariant nobody is watching. That distinction is the whole
change.

One INFO line per reason per episode says it in the log (`TRUNCATION GATE: an alloc observation
could not be measured (reason ...)`), naming the reason, the **type** of the object handed to
`update_state_after_alloc` and the request. A look that measures closes the episode, so "blind at
boot, then fine" and "fine, then blind again" are two lines - and the counter keeps climbing
regardless, because dedup is a log rule and never a counting rule.

```bash
# READ-ONLY, on the same peer-gated route. Shows the cause of every blind look.
ssh llm "curl -s http://127.0.0.1:8000/plugins/kvtier/snapshot" \
  | python3 -c 'import json,sys
 d = json.load(sys.stdin).get("disk_restore") or {}
 p = d.get("load_prune") or {}
 print("observations", p.get("group_take_observations"),
       "| blind", p.get("group_take_unavailable"))
 r = p.get("group_take_unavailable_reasons")
 if r is None:
     print("this build predates 0.2.55: the total has no cause on the wire")
 else:
     for k, v in r.items():
         if v: print(f"  {k:28} {v}")
     print("last blind look:", p.get("last_group_take_unavailable_reason"))'
```

**What was actually observed on the live 0.2.54 box.** `observations: 15`, `unavailable: 3`, every
other gate counter 0, `lookups: 13`, `offers: 2`, and no `_take_gate` line in the engine log. The
cause is settled by that arithmetic rather than guessed: `_take_gate` returning `None` is a
property of the manager object, so it would have blinded **all 15** observations, not 3 - and 15
observations against 13 lookups means at least two allocs belonged to requests this module never
looked up, which therefore cannot have a pending tuple. So the live `unavailable` is
`no_pending_hit_for_request`: **benign**, and the counter's published meaning was the actual
defect. What the numbers alone cannot settle is which of the two table-side names each of the
three was, and `group_take_unavailable_reasons` is what settles it on the next boot - which is the
same conclusion this change was built to make possible.

**How to prove a counter increments without touching production.** You do not need a load, a
request or a restart: `uv run pytest tests/test_disk_take_truncation_gate.py -k
move_in_the_published` re-executes the shipped `_merge_into_pending` in memory with the ONE line
changed that makes an align group's hit list longer than the fork's take, runs the same offer as
its own control, and asserts the counters moved **in `scheduler_section()["load_prune"]`** - the
builder the route serves, not the state dict. It runs on ctypes fakes: no engine, no torch, no
network, no box.

## A span is fused on evidence about the BYTES, never on a gap in our own knowledge - 0.2.60 (text)

**A span is fused on evidence about the BYTES, never on a gap in our own knowledge
(0.2.60).** A span key is the run's FIRST block, so fusing one refuses every later
request that starts there - which is the next turn of the same conversation, under a
different request id. That price is payable for a data-kind load failure or
`group_hit_truncated`, and it is NOT payable for `no_read` (a load reported finished
that this process could not show an applied read for), which fuses the request only.
Before this release the first restore of every boot looked like `no_read` - the
boot-wide counter was still 0, and the worker's rank file had not been read yet - and
cost the span for the whole boot: a 241,740-token prompt recomputed in 66.3 s, 34 times
in 5 days, with `no further disk offer for request ... the load was reported finished
with no applied disk read behind it` in the engine log.

## `restorations` cannot rise without a read - and the witness is asked about reads at the instant it answers - 0.2.60 (text)

**`restorations` cannot rise without a read - and the read is asked about per load
(0.2.60).** The fork reports a load finished for its
own host-RAM blocks too, so a finished report is not evidence of a restore. A
completion counts as a restoration only when an applied disk read can be shown for
**that load's own CPU slots**: `applied_read_evidence_for` matches the slots named in
the load's ledger record against the window the apply point records (a worker's own
process) and against `applied_slots`, the newest-first list of filled CPU slots each
worker now publishes in its rank file. The apply loop (`_apply_buffer`, inside
`start_load_kv`) runs in the **worker** process, so a scheduler-only process has no
window of its own and its per-request evidence can only ever come from that file; the
rank file is rewritten on cadence, so the completion path is allowed one fresh, bounded
read per `APPLIED_SLOT_REPROBE_MIN_SECONDS` to close the gap - `applied_slot_reprobes`
counts those reads and `ranks_applied_slots` is how many slot ids the live ranks named.
While no live rank publishes a window at all (a worker on an older plugin) the evidence
falls back to the boot-wide counter and says so in its `source`; the moment a window
exists the fallback is unreachable. A delivered disk->GPU write
(`disk_gpu_write.delivery_writes`, `VLLM_E2_TIER_DISK_GPU_DIRECT_WRITE_DELIVERY`) is
applied evidence too - delivery writes only the entries that already passed
`_apply_buffer`, so it counts once and never twice. A completion with no read behind it
counts as `no_read` and
fuses **its request, never its span** (see the paragraph above). `verdict=CONFIRMED` needs both, which is why the witness line
carries `reads_applied=` and `no_read=`. The witness is emitted once per
**(verdict, reason-kind)**, never once per step: keying it on the sentence re-issued
it 1,048 times a second on the incident boot, and `witness_suppressed` counts the
withheld lines so a quiet log is not mistaken for a dead module.

**The witness is asked about reads at the instant it answers (the 13:24 cold boot).**
It fires on events - the end of an offer and the end of a completion batch - and it used
to compute `reads_applied` from `ranks_reads_applied`, whatever the last `worker_proof()`
read left behind. A rank file is rewritten on the heartbeat and on steps, so a boot that
had just served 169,600 tokens was pronounced
`verdict=REFUTED reads_applied=0 reason=adopted_without_a_read` while the same rank files
reported 228 applied reads minutes later: the verdict had outran its evidence, which is
the second time (`verdict=CONFIRMED reads_applied=0`, 2026-10-05, was the first). So the
worker-attested transport is now consulted live at verdict time - the walk the completion
path already makes reports the counter it was holding anyway
(`_note_rank_reads_seen`), and a verdict whose every cached source says zero takes one
bounded read per `WITNESS_RANK_READ_MIN_SECONDS`, counted as `witness_rank_reads`, with no
read at all once evidence exists. `CONFIRMED` still needs a restoration AND a read, and a
REFUTED about reads now stamps what it asked and when
(`rank transport read for this line: 0 applied read(s) over 4 live rank file(s)`) - the
stamp sits before the reason's tail because the line clips it to 160 characters, and it
adds no colon of its own, so the dedupe key cannot move with it.

## Which request got how many tokens off the disk (`disk_restore.py`, 0.2.61)

`_mark_adopted()` has always known the number and thrown it away. "Adopted" means the
scheduler put our CPU slot ids into a step's `load_cpu_blocks` - the only observable
proof the fork took the external tokens - and at that instant the load ledger row holds
the request id AND the token count it adopted. Everything published afterwards is a
BOOT total (`adopted`, `adopted_tokens`), which cannot fill a table of requests: with
`adopted_tokens: 297,600` on the page, no dashboard can say how much of ANY single
prompt came from NVMe. The per-request number now goes into a bounded ring that the
snapshot payload publishes as `disk_restore.recent_adoptions` and the `disk` verb's
`adoptions` op reads deeper.

**One row per adopted request**, newest first: `seq` (monotone for the boot, never
renumbered by a wrap, so a gap is visible), `request_id`, **`tokens`** (this request's
tokens adopted from disk - `0` is a real "adopted nothing", `null` only a malformed
ledger row), `blocks` (entries, one per group per span), `spans` (distinct whole
scheduler blocks), `epoch` (the boot that knew it), `ts` + `mono` (both clocks), and
`rank` + `role` (which process recorded it: in production `role: "scheduler"`, `rank:
null`, because a worker never adopts).

**THE CAP, AND THE ARITHMETIC.** 256 records deep, 48 of them in the polled payload.
An adoption is at most one per request, so arrivals bound the fill rate: the busiest
minute in 5.5 months of proxy rows is 142 requests (`llm_requests.db`, 2026-07-27
15:55) and the mean busy minute is 5.66, while the live box today adopts 4 times in 28
minutes (its own `restore-witness` lines) because the whole-scheduler-block floor
rejects most offers. 256/142 = 108 s of history - 36 intervals of a 3 s poll - in the
worst minute this box has ever had, and ~45 minutes at the typical busy rate; 48 rows
cost 11,371 B measured on a 51,839 B snapshot document (+21.9%) and still cover 20 s at
the worst rate. A record is ~237 B of ten scalars, so the full ring is ~60 KiB next to
the 7.99 GiB arena. The shape mirrors `failures` (ring 32, published 6, the verb reads
the rest).

**A FULL RING DROPS THE OLDEST AND SAYS SO.** `adoptions_dropped` counts every
recycled row; the writer never blocks, never waits and never refuses the newest row
(it is the row a reader polling now wants), and it runs inside the step thread, so it
is contained whole: a row that cannot be built moves `adoption_record_errors` and costs
nothing else, because `adopted`/`adopted_tokens` were already updated and stay exact.

**READ THIS BEFORE DRAWING A ZERO: the ring is perishable.** A request missing from
`recent_adoptions` may have adopted nothing OR may have adopted 89,600 tokens in a row
the ring already evicted. Only the boot totals (`adopted`, `adopted_tokens`) may carry
a "nothing came from disk" claim, with `adoptions_dropped` as the tripwire that rows
were lost. A `disk_restore` section with NO `recent_adoptions` key is a writer older
than 0.2.61 and reads as `not_reported` - never as an empty list, and never as 0. The
wire contract the reader in `llm-api/kv_tier_diagnostics.py` is written against is
`snapshot.py`'s module docstring.

**WHY THE SNAPSHOT AND NOT A RANK FILE:** an adoption is a scheduler fact and
`rank-*.json` files are written by workers, which never see it; carrying it there would
be new cross-process state and a second writer for one integer. `tier.gpu_pool_census`
is the existing precedent for a route-only, scheduler-side field.

## A capacity unlink waits for the clock and for the reader (`disk_layout.py`, 0.2.62)

**What was asked**: the eviction clock became a touch clock in 0.2.29, so a restored span
outranks a never-read one. Does that make a capacity pass safe?

**What the source said**: not yet, in two ways, both of them races between the one thread
that deletes and the threads that write and read.

**1. A pass can race its own store.** A ceiling pass runs at the end of *every spill*
(`disk_spill._handle_job` -> `_enforce_ceiling`), so the file that triggered the pass was
committed seconds earlier, and the pass walks up to `MAX_EVICTION_PLAN` (4096) entries and
unlinks up to 4096 of them - a window in which the store moved under the walk. Ranking by
`st_mtime` does not notice: "oldest" has no notion of "too young to judge yet". So a
capacity unlink now refuses a file whose touch clock sits inside
`disk_layout.UNLINK_GRACE_SECONDS` (90 s), **in the planner and again in the executor** -
the executor's copy is the one that closes the race, because a touch can land after the
plan was built. A withheld entry is counted (`excluded_grace`, and in the spill section
`ceiling_measurement.grace_entries_last_pass` / `grace_bytes_last_pass` /
`grace_entries_excluded`), and a refusal by the executor is counted by rule name
(`grace_unlink_refused`), never by string-matching a sentence.

**2. Nothing knew a read was in flight.** `disk_read` promises, on a reader thread, to
apply bytes it has queued; `disk_spill` deletes on the writer thread. No age window can
express that promise - a file being read right now may have been touched a minute ago (the
throttle) or an hour ago (its first service). The reader now publishes the claim itself:
`disk_read.inflight_paths()` walks the open batches and returns the paths of every job
that is `queued`, `running` or `done` (bytes not yet applied), capped at
`MAX_EVICTION_PLAN` so a claim can never be dropped for being outside a set smaller than
the plan that wants to delete it, with `truncated` saying whether the cap bit. The spill
pass asks **once per pass**, hands the set to `execute_unlink_plan(protected=...)`, and
publishes `claim_paths_last_pass`, `claim_set_truncated` and `claim_unlink_refused`. A
claimed path is refused **whatever its clock says and whatever path the caller took** -
including an operator's sweep, which keeps its authority over *age* but not over a
promise.

**The arithmetic, since a window with no derivation is a knob with no owner.** From
below: the writer's own cap (`MAX_FILES_PER_WINDOW` = 200 files / `SPILL_WINDOW_SECONDS` =
60 s, so ~130 files per minute per rank at 11.4 MB slots) re-triggers the pass constantly,
a restore's `RESTORE_DEADLINE_SECONDS` = 45 s bounds one offer-to-apply lifetime, and
`disk_read.TOUCH_MIN_INTERVAL_SECONDS` = 60 s is the blind spot the rule exists to cover.
From above: `disk_spill.PEER_DIGEST_FRESH_SECONDS` = 120 s is how long a foreign digest
counts as a live sibling, and a grace reaching that far would make every file that *proves*
a peer is alive unsheddable - a four-rank fair-share pass would have nothing it is allowed
to delete. 90 s = 2 x 45 s sits between the two, leaving a 30 s band that is both live and
sheddable; both bounds are asserted in
`tests/test_unlink_grace_and_pending_claim.py`, so a future retune fails a test rather than
a box. **The condition that would force this number up**: one full ceiling pass must fit
inside it, and no pass duration is observable on the live store
(`evictions: 0`, `journal_removals: 0`, `ceiling_passes_declined: 0`) because no pass has
run since the store was written. Read `spill.ceiling_measurement.pass_seconds` when one
does; if it approaches 90 s, the honest fix is to bound the pass, not to keep growing the
grace.

**What this deliberately does NOT do: promote a whole span on one hit.** The plan was to
extend a restore's `os.utime` to every file of every span it served, plus the boundary
anchors it might have used. Both were dropped on measurement, and the reason is worth
keeping. A successful restore already reads the anchor it adopted: `_disk_run` puts the
run-end boundary pairs into the same `keys` list as the positional ones
(`disk_restore.py:5878`), `_merge_into_pending` builds a `load_paths` entry for each, and
the worker reads every non-empty path - so the anchor of an adopted run **is** touched by
the serving that needed it, and a span-wide promotion protects nothing that serving does
not already protect. Worse, the live boot reports `boundary_groups_known: 0` ("no group's
spec reports 'align' mamba_cache_mode"), so on the geometry that filled this store there
are no anchors at all and no read-set that touches groups 0-3: every one of the 356 files
whose clock has moved since its write is a group-5 copy. Promoting siblings for that read
would pin bytes the current restore rule cannot load, which is the opposite of what the
tier is for. If a future boot runs an align-mode group and its anchors are seen to age out
anyway, this section is the place to say so.

## The run-end anchor patch (`mamba_anchor.py`, 0.2.64)

**THE SECOND RUNTIME-PATCH TARGET, and the shape is not a new framework.**
`backport.py` established the discipline for patching a live vLLM from a plugin -
target constants, per-method `inspect.getsource()` fingerprint pins, a stale
fingerprint as a REFUSAL and never a guess, installs that cannot raise into a boot,
foreign ownership markers refused rather than overwritten, `BaseException`
propagating, refusals logged once per episode and surfaced in both the payload and
the boot line. `mamba_anchor.py` copies that shape for one more target and invents
nothing else: no registry, no decorator, no second patching API. One module per
target, each carrying the same discipline, exactly as `backport.py`,
`guard_mode.py` and `disk_restore.py` each carry their own layers.

**THE BUG, and the three lines.** An align-mode (`mamba_cache_mode == "align"`)
group is the group a restore must find its boundary-state ANCHOR in, and the fork
decides which of its state blocks survive `cache_full_blocks` in one pure
classmethod: `MambaManager.reachable_block_mask`
(`single_type_kv_cache_manager.py:1610-1669`). It marks segment-boundary states and
reachable-boundary states and nothing else, so a cached range whose LAST block is
neither returns `mask[-1] is False`. That block is never published, so it is never
hashed, so the align group has no file at the run's end - which is precisely the
`no_boundary_anchor` refusal (`disk_restore._disk_run`: "the positional groups have
the whole run and NO run end has a boundary-state file in every align-mode group")
and the 20.0% orphan rate the 0.2.63 spill-side counters were added to see. The
whole effective change is

    if mask and not mask[-1]:
        mask[-1] = True

A `None` mask is the fork's dense verdict and is returned untouched; an EMPTY mask is
untouched too (writing `mask[-1]` into one would address a block the caller never
asked a mask for); a mask that already ends True is untouched and, deliberately, NOT
counted - counting the fork's own boundaries would make the counter measure the
retention interval instead of the patch.

**WHY THIS ONE IS SAFE TO ATTEMPT, AND WHAT IT MAY NEVER DO.** The patch sits on the
scheduler's prefix-cache RETENTION decision: a pure function of
`(start_block, end_block, alignment, spec, retention, boundaries)` returning
`list[bool] | None`. It is spill-side only - it can add data to the store, and it has
no door to an arm, a serving posture or a request path. That claim is pinned twice,
and both pins are in the suite: behaviourally (`test_nothing_about_serving_posture_changed`:
every posture and arm identical before and after an install plus a mark) and
structurally (`test_the_modules_source_names_no_posture_or_arm_writer`: no identifier,
attribute or code string in `mamba_anchor.py` names the serve switch, the offer
switch or an arm, and the module reads no `os.environ` of its own). The AST-pinned
serve-switch tests (`test_disk_authority_switch.py:95/:191/:251`) keep passing
unchanged, and their owner lists did not grow. **If one of those ever fails, the
patch has grown a door: stop, do not relax the test.**

**FINGERPRINTS, AND WHERE THEY CAME FROM.** `ORIG_SHA256` is pinned from the live-tree
reconstruction `/home/user/Source/experiment/vllm-live-repro`, branch **`live`**
(= `built` = clean upstream v0.30.0 + the 58-patch series, plus our four kvtier hunks,
byte-identical to what the engine imports). That branch's
`vllm/v1/core/single_type_kv_cache_manager.py` has md5
`52a3c5ad0fc4e6911c69bb554b2d914a`, and so do branch `built` and
`/home/user/Source/experiment/vllm-dev` - the three trees are identical FOR THIS FILE,
which is what makes the pin a fact about the engine rather than about one checkout.
This is deliberately NOT true of `simple_kv_offload/manager.py`, whose live copy
(`91a3aa65`) differs from the dev copy (`26097a02`) because our own runtime hunks live
there: no fingerprint for this target came from that file, and none may.
`tests/fixtures_mamba.py` carries the same method text, generated by
`tools/gen_mamba_fixture.py`, and the suite ties pin, fixture and file md5 together.
The extraction difference between "read the file" and "`inspect.getsource()`" is
measured, not assumed: the same AST line-span extraction reproduces all four shipped
#56810 fingerprints, which were captured with real `inspect.getsource()` under the
venv (`test_the_extraction_method_reproduces_the_inspect_fingerprints`).

**WHERE IT LANDS.** At connector creation, in `_after_connector_created`, before
`boot_from_connector` - the first instant the EngineCore process provably has a tier
manager, with `vllm/v1/engine/core.py:60` having already imported
`single_type_kv_cache_manager`, so the class is resolvable through `sys.modules`
without this plugin importing it. Not in `register()`: `general_plugins` loads in
workers and arg-parsing processes too, and the mask runs in the scheduler and nowhere
else, so a boot with no tier connector gets no anchor patch and says so.
`backport.ensure_installed()` retries it for the one case the hook cannot reach.

**ENABLE, ROLLBACK, AND THE BOOT TOKEN.** `VLLM_E2_TIER_DEV_MAMBA_ANCHOR_PATCH`,
default ON, grammar identical to `KVTIER_DISK_BOOT_ENABLE` (`0`/`false`/`off`/`no`
disables; an unrecognised value stays ON and reports `recognised: false`). `0` is a
clean rollback: `install()` returns before it resolves the class at all, so there is
no state in which a wrapper is half-fitted. The boot line carries the verdict as
`MAMBA_ANCHOR[patched|refused|disabled|env-refused]` - bracketed and with no `=`, so
the llm-api restart verifier's `grep -oE "tier=[A-Za-z_]+" | tail -1` cannot land on
it, and so that "installed" and "silently absent" are one glance apart.

**THE FOUR COUNTERS, and the honest limits of the attribution.**

| counter | unit | door |
|---|---|---|
| `mamba_patch_installed` | bool, this process | `backport.section()`'s `mamba_anchor` block, and the `MAMBA_ANCHOR[...]` token |
| `anchor_offers_total` | run-end blocks the patch additionally marked | `snapshot.dev_state_sections()`'s `disk_store` section (via `disk_store.section()`, `hook_section()` and the `/disk` verb) |
| `anchor_spills_completed` | align-mode anchor files a rank wrote | `disk_store.rank_rollup()` totals - so it had to join BOTH the writer (`worker_report()`) and the reader's whitelist (`RANK_ROLLUP_FIELDS`, which `_rank_file_rollup_row()` walks) |
| `restore_anchor_hits_total` | run ends anchored on a patched boot | `disk_restore.counters()` (+`units`), `scheduler_section()`, and the rank file's `disk_restore` section |

The three spill/restore-side counters are the two ENDS of one pipe, and they are
deliberately not a per-block causal chain. A store hand-off is a list of
`(slot, block-hash)` pairs (`disk_store._remember_event`) with no token position and
no request id, and `reachable_block_mask` is a pure classmethod with no request
either - so "did THIS file exist because the patch marked THIS block" has no answer
in-process, and no counter here pretends otherwise. The restore-side counter is
therefore gated on the patch being live (zero by construction when the flag is off),
and the spill-side one is not gated at all (the workers spill, the scheduler patches -
a local gate would read zero forever). The patch's real effect is read as the RATE of
the second against `spill_incomplete_state` MOVING when the flag flips.

## Who pays for the drain (`disk_index.py`, 0.2.65)

**THE MEASUREMENT.** A boot at 23:00:43; a request at 23:02 that recomputed **129,285
tokens** because the restore tier answered `blind: true` and could not say whether the
span was on the NVMe tier; a journal replay that finished around **23:12**. The journals
were four files of ~155k lines each, ~25 MB in total.

**IT WAS NOT A DEADLOCK, AND SAYING SO IS THE POINT.** `disk_restore._refresh_index()`
runs on the offer path at `disk_restore.py` and it runs *before* blindness is consulted
(blindness is refusal text appended afterwards, in the `no_disk_run` sentence), so the
drain advanced regardless of the `blind` flag. A self-sustaining blind window would have
been a far worse bug and there was none. What there was is a **budget** and a drain that
is *traffic-driven*: the boot spent its 5-second deadline with 2.3 MB of journal unread,
and the rest was read at `MAX_READ_PER_REFRESH` (512 KiB) per arriving request behind a
1 s floor. About six seconds of reading was charged to eleven minutes of traffic, and
the first 90 seconds of that boot paid the largest share of it.

**THE FIX IS A MOMENT, NOT A SECOND DRAIN.** `boot_from_connector` runs from the
connector-creation hook, and the connector is built by `KVConnectorFactory.create_connector`
at `vllm/v1/core/sched/scheduler.py:164` - inside `Scheduler.__init__`, which
`EngineCore.__init__` (`core.py:260`) constructs before `run_busy_loop()` is ever entered
(`core.py:1452`). Nothing can be served from that moment, so the drain's seconds are paid
by STARTUP instead of by a user's request. `_boot_journal_drain` runs as the boot's last
step, reaches `disk_index` and nothing else, cannot arm, cannot offer, cannot pick a
posture, and never joins `boot.refusals` - a slow index costs a recompute, and a boot
that failed to read a journal is still a boot. An inert boot
(`VLLM_E2_TIER_DEV_BOOT_INERT`) drains nothing: inert means inert.

**THE TWO NUMBERS, AND WHY THOSE.** See the env table for
`VLLM_E2_TIER_DISK_BOOT_DRAIN_SECONDS` (45 s, the hard ceiling on the startup cost) and
`VLLM_E2_TIER_DISK_BOOT_DRAIN_BYTES` (64 MiB). The bytes number is the interesting one:
a bounded pass reads a whole window and applies only the lines its **line cap** fits,
and the cursor advances past APPLIED lines only - so the tail of every window is read
again next pass. Measured in this repo on the incident's shape (4 ranks x 110k lines,
16.2 MB): **28.0 MB read = 1.73x amplification**. 25 MB of journal therefore costs ~43 MB
of reading, which is why the 32 MiB budget could not have finished it even with an
infinite deadline. At the box's measured ~2.2 MB/s the incident journal converges in
~11-12 modelled seconds of boot; the deadline's job is to make the *worst* case 45 s and
no worse, whatever the journal's size.

**WHAT IS NEW IN THE PAYLOAD, AND HOW TO PROVE IT WORKED.**
`GET /plugins/kvtier/disk` → `restore.index.boot_drain`:
`{ran, attempted, trigger, stopped_at, bytes_read, lines, entries_at_end,
elapsed_seconds, converged, lag_bytes, lag_journals, passes, deadline_seconds,
max_bytes, error}` - beside the `converged` / `lag_bytes` / `blind` fields it explains,
with its own `units` - and `restore.index.drain_limits` for the three bounds in force
with each one's origin (`default` / `env` / `env_ignored`) and any refused value's
complaint. After a good boot read `{"ran": true, "stopped_at": "converged",
"lag_bytes": 0, "converged": true, "trigger": "connector creation"}` and
`restore.index.blind: false`. The BOOT line carries the one-glance verdict as
`JOURNAL_DRAIN[converged|deadline|budget|no_progress|empty|no_dir|inert|error|not_run]`
- bracketed and with no `=`, for the two reasons `MAMBA_ANCHOR[...]` gives: a closed
vocabulary, and invisible to the llm-api restart verifier's
`grep -oE "tier=[A-Za-z_]+" | tail -1`.

**WHAT DID NOT CHANGE.** The steady-state cap (`MAX_READ_PER_REFRESH`, 512 KiB), the
line cap (8,192), the 1 s refresh floor, the offer path's work bound, and the rule that
whatever a boot leaves keeps draining one bounded pass at a time. `drain_journal_at_boot()`
carries `mamba_anchor.install()`'s fail-open contract - any `Exception` is counted
(`boot_drain.error`), logged once, and leaves today's behaviour; `BaseException` still
propagates.

**Tests:** `tests/test_boot_journal_drain.py` (12). The deep fixture is 4 journals of
110k well-formed lines (~16 MB, 440k lines) driven by a clock that advances by
`bytes read / 2.2 MB/s`, because a wall-clock deadline measured in *this host's* time
(this host reads the same journal ~4.5x faster than the box's array) would be measuring
the machine it ran on. Its headline test asserts its own fixture is deeper than the
retired 5.0 s budget and shallower than the shipped 45 s, so the test cannot drift into
passing either way.

## The arena is a buffer, not a store: mirror mode (`disk_store.py`, 0.2.66)

**THE MEASUREMENT.** One conversation - **207,861 tokens** at 01:22, asked again at 16:34
after a single **15-hour suspend** - came back with **0.0% reuse** and `reads_applied: 0`
for the whole fresh process. Nothing had been evicted to make room for it and nothing had
been discarded: the prefix was still resident in the host-RAM arena when the proxy stopped
the backend to suspend the box, and it died with the process. The spill path had evicted
**nothing** that night, which is the fact that names the bug - a block reached NVMe only
when the arena needed a slot *back*, so a stop that happened while nobody needed the slots
destroyed a prefix that was never written anywhere.

**THE FIX IS A DOOR, NOT A PIPELINE.** `disk_store.on_store_completed` is the one seam that
holds both halves of the fact at once: the `(cpu_slot, block_hash)` pairs were copied out at
`bind_connector_metadata` (the only moment the metadata is in this process's hands, per the
fork's contract at `worker.py:111`) and the completion diff says those slots now hold the
bytes those names describe. A slot is reusable by the NEXT store the instant its event
completes, so a spill that waits for anything later is either a stale read or a byte that
never lands. **MIRROR MODE** queues the slot at that moment, through `disk_spill
.request_spill` - the existing queue, the existing writer thread, the existing deduplication
(one boot absorbed **1,552** duplicate writes for free), the existing journal. There is no
second writer, no second queue, no second journal path, and none is planned.

**WHAT THE COST WAS, BEFORE THE DECISION AND NOT AFTER.** This boot's rank files showed
`slots_offered 3,076` against `spill_queued 2,556`: mirroring every store is **~1.20x**
today's write volume, not 4x. The ratio is published per rank as
`spill_on_store_queued / slots_offered`, because a cost assumed in advance is a claim and a
cost measured is a number.

**THE ONE RISK IT CARRIES, AND THE GUARD THAT CARRIES IT.** The live store sits at **961 GB
of a 1 TB** soft ceiling and the filesystem behind it has **~400 GB** free, so the way this
change becomes an outage is a mirror that outruns the evictor and fills `/home`. The evictor
cannot be the guard (it runs *after* a committed write), and `request_spill` cannot be it
either (by then the bytes are already snapshotted into the job), so `disk_spill
.mirror_headroom()` is asked **before the copy**: `_accountable_bytes()` - the same currency
the ceiling pass itself decides in - plus the bytes already queued, plus this slot's width,
against the ceiling. A refusal is counted in the door's own table as `ceiling`, and **no
byte is copied**. The table is `MIRROR_REFUSAL_REASONS` = `ceiling` / `queue_full` /
`disabled` / `other`, and it is deliberately NOT `REFUSAL_REASONS`: those are the spill
mechanism's reasons, and putting "mirror mode filled the ceiling" in the same table as
"this rank was never armed" merges two statements with opposite meanings.

**WHAT THE DOOR MAY NEVER DO**, each half pinned by a test in
`tests/test_spill_mirror_mode.py`: block the serving path (its cost is one ceiling
comparison plus the queue's own `put_nowait`; a full queue is a counted `queue_full`, never
a wait); exceed the ceiling (the guard above, and `mirror_headroom` returns the arithmetic
it used in `basis`, because a refusal a reader cannot reconstruct is a refusal he has to
re-derive by hand); or touch a serving posture. The last one is proved against the door's
**source**, not its behaviour - `spill_on_store_verdict`, `mirror_admission` and
`note_mirror_refusal` may not name `set_disk_authority`, `set_offer_switch`, `serving_as` or
`arm(` once their docstrings are stripped. Behavioural equality over one run cannot rule out
a door that fires on another path.

**THE ROLLBACK IS STRUCTURAL.** `VLLM_E2_TIER_DISK_SPILL_ON_STORE=0` closes the door at boot
and `spill_on_store_verdict()` reports it as `source: "env", recognised: true`; a value that
is neither an enable word nor a disable word (`flase`) stays **ON** and reports
`recognised: false` with its complaint in `problems`, because a reader who meant to close the
door and typed it wrong must not be handed a payload that reads as his intent.

**RELEASE DISCIPLINE - SAID IN THE PAYLOAD, NOT ONLY IN THE SOURCE.** Once the arena's bytes
reach NVMe promptly, its slots could become releasable as soon as a spill commits instead of
when the arena needs them back: a much smaller `KV_OFFLOAD_SIZE` for the same prefix depth.
**THIS CHANGE DOES NOT DO THAT.** No arena release, eviction or capacity policy is touched
here, and the sentence saying so (`SPILL_ON_STORE_RELEASE_NOTE`, published inside every
`spill_on_store` block) is in the payload because the idea is the kind that gets implemented
by whoever finds it undocumented.

### `op=flush`: the primitive that says the backlog arrived (0.2.66)

Mirror mode makes the spill asynchronous by design, so the instant a stop takes the process
down there can still be jobs in the queue - the exact data the change exists to protect. The
verb is the bounded wait that closes that gap:

```
POST /plugins/kvtier/disk  {"op":"flush","timeout_seconds":20}
```

Flat answer, no nesting: `ok`, `drained`, `queue_depth_before`, `queue_depth_after`,
`bytes_written`, `elapsed_seconds`, `timed_out`, `reason`, plus `budget_seconds`/`clamped`.
`ok` describes the **call**, `drained` describes the **queue** - a budget that expires is
`ok: true, drained: false, reason: "timeout"`, because a proxy that read `ok` as "the bytes
are safe" would suspend on a store that still holds the conversation. The named reasons are
`timeout` (a live writer that could not finish), `writer_down` (nobody will drain this:
answered in milliseconds rather than after the budget, because the caller's decision is the
same either way), `disabled` (`VLLM_E2_TIER_DISK_SHUTDOWN_FLUSH_SECONDS=0`) and a refusal
before the queue is touched for a malformed `timeout_seconds`.

Two properties are load-bearing. **An empty queue returns in milliseconds** - the suspend
path calls this on every sleep, and a flush that slept first and looked afterwards would cost
every suspend 100 ms and teach the operator to stop calling it (a test asserts the bound).
**The budget is clamped at 30 s** (`FLUSH_BUDGET_SECONDS_MAX`) whoever asked, and the clamp is
reported rather than applied in silence: this waits on a thread that can be holding a suspend.

**THE SCOPE IS IN THE PAYLOAD, AND THIS IS THE PART TO READ.** `disk_spill`'s queue is
**per process**, and the writer thread runs in the **worker** ranks (the processes that own
the CPU slots), while `/plugins/kvtier/disk` is answered by whichever process the utility call
landed on - the same trap that made `op:status` report `evictions: 0` for a store that had
shed 234 GiB. So the answer names its `scope` and adds `ranks`, read from the rank files each
worker writes. A caller that wants "no rank in this boot is still writing" must read `ranks`;
a `drained: true` that meant "the process that answered happened to be idle" would be a
silent no-op at the worst possible moment, which is the failure mode this op could otherwise
have shipped.

### The automatic drain, and why it is not an `atexit` handler (0.2.66)

The same wait runs on the engine's own shutdown, so an operator does not have to remember to
call it. Where it hooks was decided by reading the live fork rather than by assumption:

* **`atexit` does not run in a worker.** A worker ends its *orderly* shutdown in
  `v1/executor/multiproc_executor.py:_exit_worker_process` with `os._exit(exit_code)`, which
  skips `atexit`, every pending `finally` and interpreter finalization. A drain registered
  only there would be silent in the one process that OWNS the spill queue - silently, on
  exactly the path it was written for.
* **The connector's own `shutdown` does run.** `Worker.shutdown()`
  (`v1/worker/gpu_worker.py:1699`) calls `ensure_kv_transfer_shutdown()`
  (`distributed/kv_transfer/kv_transfer_state.py:97`), which calls the connector agent's
  `shutdown()` - and that happens **before** the `os._exit`. So the connector instance the
  plugin already sees at creation is the seam in the right process at the right moment.
  `disk_spill.install_shutdown_flush()` wraps that ONE instance's method, marks it, refuses
  to wrap twice, runs the drain **first** (what follows it in the fork is
  `destroy_model_parallel()` and then the exit), calls the original afterwards, and lets the
  ORIGINAL's exception propagate: that failure is the fork's story. Our own failure is owned
  here - counted in `flush_errors`, logged once, and the fork's teardown runs regardless.
  `atexit` is kept as the secondary seam for processes that really do exit normally.

**THE BUDGET, AND THE WALL THAT ACTUALLY BINDS.** The unit is `KillMode=control-group` with
`KillSignal=15` and `TimeoutStopSec=90`, and the engine is launched with `--shutdown-timeout
60`, so SIGTERM reaches the EngineCore and every worker directly and 15 s fits inside both.
But on an idle box the wall that binds first is neither of those:
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` (unset on the box, so **5 s**) plus the executor's
4 s SIGTERM wait is ~9 s from the workers. The shipped drain default is 15 s because the
operator asked for it; on a box that has not raised that variable, a drain that needs more
than ~9 s will be interrupted by the executor's ladder, and the honest answer to "did the
backlog make it" is then the `SHUTDOWN FLUSH` line in the engine log, not the budget.

Fail-open is the whole contract: a budget that expires costs ONE WARNING line
(`SHUTDOWN FLUSH did NOT drain the spill queue (reason=..., N job(s) left)`) and the stop
proceeds. A drain that could hold a suspend hostage would be a diagnostic causing the outage
it measures. And the rollback is structural: `VLLM_E2_TIER_DISK_SHUTDOWN_FLUSH_SECONDS=0`
installs **nothing**, because a wrapper that runs and does nothing is still a wrapper that
can fail on the stop path.

### The lookup ledger arms at boot (`lookup_ledger.py`, 0.2.66)

The per-request offer/adopt ring was DISARMED on the boot where it was needed. The question
"why did the 16:34 request get 0 tokens off the arena after a 15-hour suspend" was
unrecoverable - the ring records nothing while disarmed, and `calls_before_arming` is a count
with no records behind it. Two investigations answered in one query what the ledger would have
answered for free, forever, had it been armed. So the boot arms it, and a boot that cannot
arm it **boots normally**, counts the refusal in `boot_arm_refusals`, and logs ONE line.

Verdicts are on the boot LINE as a bracketed token - `LOOKUP_LEDGER[armed]`,
`[env-refused]`, `[refused]`, `[inert]`, `[not_run]` - because a bare absence cannot be told
from a value nobody grepped. Bracketed and never `key=value` for the standing reason: the
restart verifier in `llm-api` pulls its PASS criterion with
`grep -oE "tier=[A-Za-z_]+" | tail -1`, and a token containing an `=` would become a candidate
match for a grep that was never meant to see it.

**AN OBSERVABILITY STEP, ALLOWED TO FAIL.** Its verdict never joins `boot["refusals"]` - a
missing diagnostic is not a tier that declined to serve - and `tests
/test_lookup_ledger_boot_arm.py` compares every *decision* of a boot that could not arm the
ring against the same boot with it armed. Arming a ring that changes the boot it observes is
worse than no diagnostic. The gate is `VLLM_E2_TIER_DEV_LOOKUP_LEDGER_BOOT_ARM` (default ON),
and the step is pinned by AST to name no serve posture, no offer switch and no arm writer.

## The proof that arrived too late: a cold boot's first request found the gate shut (0.2.69)

THE MEASUREMENT. Three boots after a suspend longer than the freshness window came up
with the scheduler-side restore gate CLOSED, and the first request behind each got
`refused=no_worker_proof` and paid a **61 s cold prefill on 231,392 tokens**. The
measured close-to-open lag, from the box being up to the gate proving restorable: **14.2
s, 17.6 s, 19.9 s**. Roughly 35 of ~38 boots look fine only because the previous boot's
rank file was still under 900 s old - the defect is invisible on a box that restarts
inside the window and expensive on one that does not.

THE CAUSE, IN ONE LINE. The only writers of a rank proof are `worker_diag`'s step path
and its heartbeat, and on a cold boot neither had written anything by the time the first
request arrived: `install()` starts the heartbeat, and that loop's first beat was
scheduled a **full interval later** (`deadline = BOOTTIME + interval`, 60 s), while
`install()` itself writes nothing on purpose. So for the first minute of every boot the
worker is alive, fitted, armed, and believed to be nothing.

WHAT A NAIVE EARLY WRITE WOULD HAVE LIED ABOUT, and what this fix therefore does not do.
Two false statements were available, and the fix is built to avoid both. (1) `install()`
runs FIRST inside the worker handshake loop, before `disk_restore.install()` raises
`worker_fitted` and registers its contribution - so a proof written there carries no
`disk_restore` section, which the reader verdicts `no_restore_section` and the gate
still reports as `no_worker_proof`: an early write in the wrong place is a proof that
proves nothing. (2) The handler is normally captured from `get_finished`, so a beat fired
before any step would publish `arena.bytes_total: 0` - not an omission but a **false
number**, in the one field a capacity reader divides by. Both are asserted against, not
merely avoided.

THE FIX, TWO LINES. **(a)** `engine_side._prime_worker_proof` runs after the worker
handshake loop has installed all four facilities, notes the connector's
`worker_handler` - which is REAL by handshake time, because `register_kv_caches` fills
`num_cpu_blocks` before the handshake is collected - and fires one contained
`heartbeat_once()`. It is strictly best-effort: a prime that raises costs one log line
and cannot break a worker startup, and with no diagnostics directory it refuses and
creates nothing, so an inert boot stays inert. It runs in the **worker** and nowhere
else, because the EngineCore's gate can only read a proof another process wrote; a
proof written from the scheduler would be exactly the attestation the design refuses to
trust. **(b)** the heartbeat's first tick is due immediately (`deadline = _cadence_now()`),
which is the belt behind that braces: `stop.wait()` still comes FIRST and the beat is
still delegated to its own thread, so `install()` writes no file inline and no beat runs
on the handshake thread - the first write simply stops waiting for a clock nobody is
using yet. The interval between beats is unchanged at 60 s, so the fifteen-beats-inside
900 margin is exactly as it was.

**A boot with zero requests now ends up with a gate-open proof on disk.** The prime is
not driven by a request, a step, or the heartbeat: a worker that boots, fits, and is
armed writes a fresh, `disk_restore`-bearing, arena-measuring rank file at handshake and
the gate proves restorable before the first token of the first request is scheduled.

**The prime is counted, and counted separately.** A rank file at boot therefore reads
`beat_writes: 1, ticks: 0, beats_from_clock: 0` unless something names that write, which
is the unaccountable artifact 0.2.51's cause scheme exists to prevent - so it is named,
as `beats_from_prime`, and NOT folded into `beats_from_clock` or `beats_after_resume`:
those two, with `resumes`, are how suspend behaviour is debugged in production, and a
prime is neither a clock beat nor a resume. The documented partition
(`clock + step + after_resume == ticks`) is untouched; `beat_writes` is bounded by
`ticks + beats_from_prime`.

**One consequence of an immediate first tick: a beat can be in the air when the box
sleeps.** Such a beat reaches its write AFTER the wake, and its document is built and
stamped post-wake, so it is an honest proof that earns the gate - but it was DECIDED before
the sleep, so it carries no resume attribution, and it stamps `heartbeat_last_beat_time`,
so the shipped 5 s write floor then refuses the beat whose whole job is to say the box came
back. Two halves fix it. **(a) production:** `_issue_beat` exempts the `"resume"` cause from
the floor and nothing else (`heartbeat_once(force=...)`); with that exemption removed, the
dedicated test fails 12 runs in 12. **(b) the suite:** the wake test opens a quiet window
before it suspends (restart behind a 30 s deadline, drain `heartbeat_inflight`) and waits
for the attribution in the rank file instead of reading it once at the instant the gate
flips. Measured: 0/10 failures at 0.2.68, 2/10 on the 0.2.69 tree with the original setup,
20/20 after.

**Two writers at boot is the design now, so the temp name had to stop being shared.** An
atomic replace is `write a temp file, then rename it`, and the temp file used to be named
`.rank-<rank>.json.<pid>.<writes>.tmp` - where `writes` counts COMPLETED writes. Two beats
in flight read the same value, write the same path, and the loser's `os.replace` raises
`FileNotFoundError` because the winner renamed the file out from under it. One process,
one writer at a time, made that theoretical; the prime and the immediately-due first tick
make it a boot-time race that this release loses on purpose - and the loser is a cold-boot
proof that silently did not land, with a rank file whose `state_reason` is a
`FileNotFoundError`. The name now carries the thread ident and a nanosecond stamp, so it is
unique per write (measured: 6 of 8 runs lost a write before, 0 of 8 after).

Tests: `tests/test_cold_boot_primes_the_rank_proof.py` - a fitted worker plus the
worker-report hook with ZERO steps must satisfy `disk_restore.worker_proof(force=True)`
on the spot (the old code answers `no_worker_proof`); the primed file must carry a
NONZERO arena and a `disk_restore` section; and a prime that raises must not reach worker
startup while the loop's later beats still write.

## The gate that could not fail: the peer-address check is removed, and the safe-method wall becomes a matrix (0.2.72)

The peer rule read `scope["client"]` and admitted loopback. A local proxy **is** the
loopback peer. So `llama-swap`'s `/upstream/` path, or nginx on the box, or any sidecar
forwarding `/plugins/*`, republished the whole dev surface to its own clients while the
check saw a permitted address and logged a pass. A control that cannot fail for exactly
the deployment that is exposed is worse than no control: it manufactures assurance, and
the operator's log says "permitted". The check is **deleted**, not widened, not made
stricter, not defaulted off - `require_peer`, `peer_denial`, `loopback_denial`,
`require_loopback`, `_peer_address_unwrapped` and the `GuardDenied` they raised are gone,
`config.allow_networks()`/`ENV_ALLOW` with them, `ipaddress` gone from the import list.
`config.enabled()` is the whole gate. Tests assert the deletion twice over: an AST pass
that `endpoint.py` touches no `request.headers`, no `forwarded` attribute and never
imports `ipaddress`, and a peer matrix (loopback, the production proxy, a TEST-NET-3
address, `testclient`, `127.0.0.1:1234`) served identically on every route.

Honesty replaced the protection, in three places that a reader actually reaches: the 503
detail (the answer an operator reads while the surface is *off* is the right place to say
what turning it on costs), the boot `WARNING` when the plugin is enabled, and
`GET /status`, which now publishes `peer_gate.removed`, an `exposure` sentence naming
"EVERY client that can reach this engine's port" and `--api-key`'s non-coverage, and
`mutating_ops_refused_on_get` read live from the policy so it cannot drift.
`allow_networks`/`loopback_only` are **deleted** from the payload rather than answered
falsely: a truthful `[]`/`false` there would read as "an empty allowlist", which is a lie
that looks like a fact. `allow_env` keeps its literal meaning ("which variable") and
gains `allow_env_gates: false`, because the live unit file still exports the name and a
set-but-dead knob must stay visible - that is how a dead control gets found instead of
assumed.

**The second finding was worse and is what made this a security fix rather than a
documentation change.** With the peer check gone, the safe-method rule is load-bearing -
and it existed on exactly one route. `/disk` had refused a mutating `op` on a `GET` with
405 since the rule was introduced; `/trace`, `/guard`, `/tier`, `/lookups` and
`/blockdump` had no answer to "does this op change state?" at all. Every one of them takes
its `op` from a **query parameter**, so these URLs worked from a browser address bar and
from an `<img src>` with no preflight and no custom header: `GET /guard?op=arm` armed
guard mode (which masks real bugs), `GET /tier?op=arm` flipped the tier's kill switch,
`GET /lookups?op=arm` installed wrappers on the live tier, `GET /trace?op=clear` emptied
the ring, `GET /blockdump?op=arm` armed the facility that reads parked KV bytes, and
`GET /guard?armed=true` armed it through the ergonomic spelling an op list cannot see.

Fix: `kvtier_dev_plugin/route_policy.py` is the one list of truth (`MUTATING_OPS` per
route, `MUTATING_ARGUMENTS` for `armed`, `/disk`'s list read from `disk_store` rather than
copied - a copy is how two answers to one question start to differ). The HTTP half answers
405 naming the op and the POST route before the engine is touched; the engine half refuses
the same call on `safe_method` with both refusal vocabularies (`reason`, `error`) so no
caller has to know which dialect a refusal is written in. `/disk`'s existing inline check
now calls the shared function, so there is one implementation and not a sixth opinion.
`dump` is classified as mutating on purpose: it writes the pending-request file that makes
four processes read real KV bytes, and the rate window it charges is state.

Wrong turn kept: the first draft of the engine-side matrix test asserted the refusal for
every mutating op *without* the `safe_method` stamp, which would have made the backstop a
gate - a POST carries no stamp, so refusing an unstamped mutation takes away the only
route that exists to run it. The stamp describes the METHOD, never the op, and reads carry
it too; three existing tests that asserted exact wire payloads for `GET /trace`,
`GET /guard` and `GET /lookups` had to gain `"safe_method": true` for that reason.

Tests: 5 files, 612 passed (baseline 417). New: the 405 matrix over 14 (route, op) pairs
plus `/guard`'s `armed` spelling, the engine-side refusal for all five verbs asked
directly, a test that the two layers read the same list (`/disk`'s included), the AST
no-header test, the retired-allowlist-is-inert sweep over every route x four values x two
peers, and the real-socket tests inverted to assert the LAN peer is **served**.

## The default that had to be earned: `kv_load_failure_policy` (`load_failure_policy.py`) - 0.2.72

**The engine's default kills the request, and we had been asking operators to undo it by hand.**
`vllm/config/kv_transfer.py` declares `kv_load_failure_policy: Literal["recompute","fail"] = "fail"`,
and it is read exactly once — `Scheduler.__init__` sets `self.recompute_kv_load_failures` from it,
which decides `finish_requests(…, FINISHED_ERROR)`, `_handle_invalid_blocks`'s `should_fail`, and the
`sync_blocks_to_evict` branch. The plugin never read or wrote it; `README.md` simply told the operator
to put `"kv_load_failure_policy":"recompute"` in the connector JSON. The upstream owner named this in
review, and the correct fix was to stop depending on a flag nobody remembers to pass.

**"Did the operator choose this?" is a question the config object cannot answer.** `KVTransferConfig`
is a pydantic dataclass and exposes no `model_fields_set` / `__pydantic_fields_set__` — verified on
pydantic 2.13 for both the constructor and `validate_json` — so an absent field and an explicit
`"fail"` are indistinguishable on the object, and EngineCore receives a pickle rather than a fresh
parse. So the witness is the operator's own text: the `--kv-transfer-config` JSON in `sys.argv`, key
presence via `json.loads`. Confidence is high for a CLI boot because argparse feeds that string to
`TypeAdapter(KVTransferConfig).validate_json` and spawn restores the parent's argv — and
`test_the_witness_survives_the_process_boundary…` fails if that ever stops being true. An unparseable
value is **not** evidence of absence (`reliable: false`). With no witness at all the code falls back to
value-vs-engine-default, reading the default live through `dataclasses.fields` instead of hardcoding
it, and that comparison is deliberately **one-directional: it can only ever say "chosen", so it never
authorises a write** — the verdict is `unresolved`, the field is left alone, and the remedy is printed.
Nine verdicts in total (`plugin-set`, `operator`, `operator-value`, `engine-already`, `unresolved`,
`opted-out`, `inert`, `no-config`, `write-refused`), and the boot line's `LOAD_POLICY[what:who]` token
reports the value read BACK after the write, never the value intended.

**The recompute default was audited against the loop that starved this box, not assumed safe.** A
2026-10-05 incident produced 123,474 offers in 3.9 minutes, and "recompute" is exactly the policy that
makes the engine reset and reschedule. The bound is the tier's own and it holds: `_offered_at[request_id]`
refuses any re-ask at the same-or-lower computed value, and `MAX_RACE_REOFFERS_PER_REQUEST = 1` is spent
via a `setdefault` that never refills — a hard cap of **two offers per request that made zero progress**.
DATA-kind and `unknown` kinds still get the permanent request-and-span fuse from `_fuse_or_spare()`, so
the classification was left untouched. The honest leak, recorded rather than smoothed: the
`_suppressed` / `_suppressed_spans` / `_offered_at` tables evict oldest-first (4096 / 1024), so a boot
with more failing requests than that can pay one extra offer per evicted key.

**Two environment findings that cost time and should not be rediscovered.** First, three suites —
`test_disk_index.py`, `test_disk_layout.py`, `test_disk_spill.py` — **pass on a tmpfs basetemp and fail
on an ext4 one**, because they assert the real filesystem type of their own scratch directory; they fail
identically at HEAD, so three red tests there are not a regression. The consequence is that this suite is
pinned to `/tmp`, which is a 7.7 GB tmpfs, and a full run with insufficient headroom produced **39
failures that were all `OSError: [Errno 28]`**. Second, an implementing agent ran `rm -rf
/tmp/pytest-of-user` to free that space. That path is the **shared per-UID `tmp_path` root**, so removing
it destroys scratch belonging to concurrent sessions — it was disclosed, steered away from, and is
recorded here because the temptation is structural: a suite that demands a nearly-full tmpfs will keep
producing this mistake. Green gate for the change: 224 targeted tests, then **3397 passed** full-suite
with ≥2.7 GB of tmpfs free. Commits `732d65b` (the access-model change) and `53670e0` (this one).

## The guard that had to be placed, not written: SSM recovery refuses at boot (`ssm_guard.py`) - 0.2.72

**The reviewer's finding:** *"nothing orders Mamba boundary stores after the RecoverSSM commit, so lifting the
replayssm refusal would persist the wrong state."* The first design mapped a runtime guard — a named refusal in
`disk_store.REFUSAL_REASONS` enforced in the `on_store_completed` pair loop *and* in `disk_spill.request_spill`, plus
a new contract kind in `_group_serviceability` so `_eligible_groups` names it in `OFFER_REFUSALS`. That is the correct
design if we ever need to *run* under SSM recovery. We do not: the operator's decision was to refuse the boot, which
costs ~110 logic lines instead of a change through three files that hold the correctness of every claim the tier makes.

**Placement was the entire task; the check itself is trivia.** Three candidate homes would have produced a gate that
cannot fail, and two of them are in this plugin on purpose: `_after_connector_created` runs inside its own
`try/except Exception` ("nothing here may raise into `create_connector`"), and `engine_side.register()` is wholly
wrapped *and* is not even passed a `vllm_config`, so it cannot answer the question there. The guard is therefore called
from the hook *wrapper* in `backport._make_create_connector_hook.create_connector`, after the fork builds the connector
and **before** `_after_connector_created` installs or arms anything — the moment the tier exists at all. From there the
`ValueError` crosses `Scheduler.__init__` (no `try` around the `create_connector` call) and `EngineCore.__init__`, and
the background `run_engine_core`'s `except Exception` logs `EngineCore failed to start.` and **re-raises**. Proven by a
test with a real fitted hook (raises, and `hook_errors == 0`, `hook_installs == 0`, `not is_armed()`), and pinned
structurally by an AST test so that moving the call back inside the contained region fails there rather than on a GPU
box. This is the same class of defect the access-model change removed: a control that reads as protection and cannot
fail.

**The load-bearing branch is the one that does not refuse.** An engine where the attribute does not *exist* is a fork
without the feature, and must boot — otherwise this guard bricks every older engine to protect one configuration nobody
runs. `_read_flag` distinguishes that from a flag legitimately set to `None` with a `_MISSING` **sentinel**, because
`None` is a value and absence is not, and collapsing the two is exactly how a probe becomes a liar. Five verdicts
(`clear`, `refused`, `refused-inert`, `indeterminate`, `no-config`); truthy outranks an unreadable sibling; the flag is
checked under all three names this ecosystem uses (`use_replayssm`, `use_recoverssm`, `use_kda_recoverssm`) even though
`use_recoverssm` turns up in **neither** fork on this disk — checked defensively rather than by guessing which fork
serves next. Two deliberate non-refusals, both tested: a **foreign** connector is not refused, because the tier never
arms behind one (`_is_tier_connector` → `hook_other_connector`) so it spills nothing and there is no wrong claim to
prevent — refusing somebody else's boot would be this plugin reaching past its own tier; and a
`VLLM_E2_TIER_DEV_BOOT_INERT` boot warns instead of refusing, since an install-only boot arms nothing. The
**fail-open** on an unreadable flag or a missing `cache_config` is deliberate and logged as loudly as a refusal
(`SSM BOOT GUARD: NOT ESTABLISHED`), because the engine's own `validate_mamba_cached_kernel` is the primary defence and
a plugin that dies because its own probe was confused is a new failure mode nobody needed. There is no env opt-out
— a test sweeps the source for one — and the message says so: the remedy is to drop the flag.

**What this still cannot prove.** No boot with `--use-replayssm` and this tier has ever run, so we have never
*witnessed* the wrong state being spilled; the guard closes a configuration, it does not measure the ordering gap, and
nobody should describe it as proven-correct on a live SSM engine. Two existing DRAFT-0007 suites guard the load/commit
collision engine-side and were left untouched — they are complementary, not redundant. Numbers: 27 new tests, and the
suite total moved 3397 → **3426**, which is 27 + 2 and those 2 are the package sweeps (`test_disk_authority_switch.py`,
`test_guards.py` iterate every `*.py` in the package) enrolling the new module automatically — a count that does not
add up is worth chasing to an explanation rather than accepting. Full suite green with ≥2.7 GB of tmpfs free; an
intermediate run showed 39 failures at 1.6 GB free, all `no_free_space`, correctly reported as environment and not
worked around. Commit `75983f1`.
