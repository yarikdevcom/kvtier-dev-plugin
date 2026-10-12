# Test suite: what each file guards

Moved from `README.md`. The per-file ledger of `tests/`. How to run the suite is
in `README.md` ("Tests"); the suites of the newer modules are described here.

2867 tests, no engine and no vllm required (`.venv/bin/python -m pytest -q`; the
count in this README is a snapshot - trust `pytest --collect-only`). The shadow
verifier is `tests/test_disk_shadow.py`: that the default reads no device memory, that
the sampling and the per-bind-window cap throttle instead of queueing, that a moving GPU
block is classified as staleness and a stable one as the layout candidate, that every
fault (no GPU caches, a device error, a width disagreement, a refused spill) is a
counter and never an exception on the step thread, and that the spill lands the same
bytes with the witness on as with it off. The GPU-direct shadow writer is
`tests/test_disk_gpu_shadow.py`: that the default creates no directory, that an `equal`
block produces a file whose bytes ARE the arena's bytes and a block the verifier did not
vouch for produces none, that the LIVE STORE is byte-for-byte what it would have been
without the feature, that a writer which raises or a root that cannot be written is a
counter and not a crash, that retention removes its own file and not a same-named file in
the store, that the shadow file is the `sorted(name)` concatenation and not the
insertion-order one, and that a fat published section sheds its illustrations rather than
its numbers. The attribution of a differ is `tests/test_disk_gpu_shadow_attribution.py`:
that a live half predating this store is `cross_execution` at INFO and never an ERROR,
that a one-byte same-execution difference is still one ERROR and `pairs_fault` with the
whole slot compared, that an unordered pair is `pairs_unattributed` at WARNING, that the
byte sample is correct and bounded, that a failed sample is "unavailable" and never
"identical", and that the published shape does not change between those states. The
restore-side verifier is `tests/test_disk_restore_verify.py`: that
the default opens no file and reads no device (proved with both readers instrumented
to raise), that a flipped byte inside `W_g` is a stable mismatch while a difference
confined to the tail is `equal`, that a destination one slot off FAILS rather than
passes, that a reused GPU block is classified on the re-read, that a file which is
not window-wide is refused rather than compared, that an entry waits for the fork's
DMA watermark, that the binding oracle catches a shifted index that name-vs-name
cannot see and counts itself unreachable when no provider is installed, and that two
tests drive the real `worker_bind` -> `worker_before_load` path so the hook site is
pinned by the product. The GPU-direct writer is `tests/test_disk_gpu_write.py`: that the
default issues no copy and no page-locked allocation (proved with the three device seams
instrumented to raise), that the submission journal reads pre-readback, our copy,
post-readback, then the fork's copy, that shifting only the copy's destination reports
`readback_mismatch_stable` while another stream rewriting the window reports
`..._gpu_changed_on_recheck`, that moving the spill FILE moves the verdict with the copy
untouched (so the comparison cannot be a mirror of its own source), that a pinned row is
released by the fork's load event and by nothing else and ages out by leaking rather than
being reused, that the step thread never calls the allocator at all, that a COLD armed boot
reaches a real write through the production entry point alone - no patched `_ensure_thread`,
no pre-filled pool, the product's own sample rate - because the cold refusal must start the
thread and the allocations must all land on the module's own thread and never on the step
thread's (the 0.2.44 regression), that `liveness` names which of the four states the writer is
in with the same key set in all four, and that a cold pool and a drained pool are two different
readings rather than one name, and that the region
pairing is by name in `sorted(name)` order - pinned by two fake caches deliberately built
in different insertion orders. The per-offer
bounds after the fixed 32,000-token cap was deleted are in
`tests/test_disk_offer_bounds.py`, with the affordability ladder that replaces the cap
as the guard against un-placeable offers. The 0.2.30
fair-share ceiling is in `tests/test_ceiling_fair_share_multi_rank.py`, the fill block in
`tests/test_disk_fill_block.py` and its access rules in `tests/test_routes.py`; the 0.2.29
touch-clock / peer-rank / deadline / evict-pool behaviour is in
`tests/test_touch_promotion_and_peer_eviction.py`. They cover:
loopback accept/reject (including a spoofed `X-Forwarded-For` from a real remote
peer), the allowlist (bare address, CIDR, IPv6, the `::ffff:` unwrap, unset =
loopback-only, malformed entries that neither raise nor admit, cache invalidation
when the env shrinks), **that no route asks for a token in any shape** (the whole
surface swept route-by-route from an allowlisted peer with no header, including the
six dev-state routes from loopback, and every refusal asserted to name the PEER rule
and never a token), that the dead `VLLM_E2_TIER_DEV_TOKEN` moves exactly one field of
`status` (`token_configured`) and nothing else, that the boot warns once when it is
set and says nothing when it is not, and that a mutating `/disk` `op` on a `GET` is
still a 405, the guard order (peer -> enabled), the `status` payload's
`allow_networks` / `loopback_only` / `non_loopback_peers_require_token` fields and
its `guard_mode` /
`dev_state` / `worker_diagnostics` fields (including `guard_mode` being `null` -
not `false` - until an engine has actually said something), path rejection for the
reload verb, holder-based reload actually changing behaviour (same-size/same-mtime
edit), rollback with the traceback as data, containment (a raising policy and a
raising engine both come back as data, `BaseException` propagates), the 504/502/503
answers, that every installed engine method takes exactly one optional argument and
answers under both dispatch shapes, and that the mount point is exactly the twelve
`/plugins/kvtier/*` paths (a list derived from `config.ROUTES`, so a new route
cannot go unlisted, with the openapi surface checked against it too). Three of them
(`test_real_socket_*`) drive a real uvicorn listener over a real TCP connection -
one from loopback, two from a source-bound non-loopback client socket
on this host's own LAN address (skipped on a box with no such address).

The raw-byte dump (`tests/test_blockdump.py`) is the one that tests bytes rather
than claims, because the question it serves is a byte-level one: a real `ctypes`
buffer stands in for the pinned arena, so a known pattern written by the test must
come back through the response file identical, at an offset, with a matching
`sha256`. Around that it pins the discipline: every cap REFUSES with the numbers
named (a range that crosses the slot, a length over the cap, eight 200 KiB ranges
that would total 1.5 MiB, a whole-block read of a 2 MiB slot) and nothing is ever
clipped; a fresh boot does no dump work even with `VLLM_E2_TIER_DIAG` set for the
rank-file writer (a pending request file with no arm file is never opened, never
consumed, and answered by no file); non-perturbation is proven by a fake CPU block
pool that records every attribute read on itself and records NONE during a dump,
with `ref_cnt`, the free-queue order and the hash index identical after; a region
whose `data_ptr` raises and a region whose `__getattr__` raises are both contained
and BOTH consume the request, so one bad region is answered once instead of retried
every poll; the wrapper hands back the fork's object IDENTICALLY and does not
consume a generator result (the class of bug that would make the engine free
nothing); the arm expires on the EngineCore's monotonic clock and in the file a
worker reads, and a garbage/oversized arm file fails closed to disarmed; the
capability marker's three states are asserted in `section()`, in `snapshot` and in
`/status`, including that `/status` says `..._unobserved` rather than inventing
`armed: false`; and the wire bound on `length` is pinned equal to the engine's own
`MAX_BYTES_PER_REQUEST` rather than to a second copy of the literal.

The disk tier's **boot default** (`tests/test_disk_boot_default.py`) is written as the
two properties it must not lose. One half drives the REAL boot (install, guard, tier arm,
restore fit, arena verdict, then the disk default) and asserts the box comes up
`disk_authority` with an arm at the cap, an arm file naming this boot, and not one POST -
then asserts `KVTIER_DISK_BOOT_ENABLE=0` reproduces the 0.2.32 boot field for field, down
to the gate's `reason_kind`. The other half is the **prohibition**: an arm file from a
previous boot (a different kernel boot id, which is what a reboot leaves) is planted
claiming hours of life, and the boot's decision is shown to come from the flag and not
from the file - behaviourally, and structurally, by walking `disk_boot.py`'s syntax tree
for any name, attribute, string constant or call that could read persisted arm state.
Between them: a manual disarm beats the boot default - driven through the VERB and proved
against the only path that can still re-arm a latched tier, a SECOND `boot_enable()` in the
same process - the veto survives until a human lifts it by arming, an arm survives ten hours
of advanced clock (the inversion of the deleted expiry test), an old-format arm file with a
past `expires_ts` still arms (the upgrade path), `{"op":"arm","seconds":7200}` still arms and
reports the number `seconds_ignored`, a file planted with THIS kernel's boot id by an earlier
engine boot is deleted by a boot that did not arm, every exit of `boot_enable` that is not
`ARMED` provably routes through the fail-safe (an AST check on the function's `return`s),
every failed gate leaves the boot uniformly disarmed and prints exactly one line naming the
gate, and the status payload carries the same key set on every path including the one where
the arm cannot be read. The deleted renewal suite's seven tests went with the tick they
tested.

The spill wiring (`tests/test_disk_store.py`) is the one that tests naming, because a
wrong name is the failure that cannot be repaired after the fact. The worker's
fingerprint is assembled from a fake handler that carries the fork's own names
(`cache_config.prefix_match_unit`, `kv_cache_groups[i].kv_cache_spec
.prefix_cacheable`, `parallel_config.decode_context_parallel_size`) and the fork's own
`get_none_hash_seed()` - including its non-hex `DEFAULT_NONE_HASH_SEED`, the value a
real unseeded boot returns; a field that cannot be read is refused BY NAME with the
paths tried and never defaulted, and the rebuild is rate-limited so an unreadable
config does not re-walk fifteen attribute paths every step. Two ranks of one
configuration get two digests and cannot form each other's file name, a changed model
id / hash algorithm / group layout moves the digest, and the seed moves it too. The
pairing rules are pinned as a refusal, not a heuristic: a length mismatch refuses the
whole step naming both numbers, while one `""` key costs its own slot and the rest of
the batch still spills. A completed store writes the region's own bytes - read out of
a real `ctypes` buffer by the same copy the worker performs - at the path the
fingerprint dictates, through the fork's `bind_connector_metadata` and
`get_finished`, with the fork's return value handed back by identity and the sibling
`worker_diag`/`blockdump` layers still installed underneath. `store_fingerprint` is
proved to be a cross-check and nothing else: `""` (today's engine) keeps the step, a
disagreement refuses it naming both digests, a non-digest is a separate counted
reason. An unpatched metadata is the AUTOUSE default of the file, so "correct on the
engine that exists" cannot be inherited from another test by accident, and the
no-engine-shape case reports `unknown` rather than `absent`. Every bound is asserted
as a counted event: the per-step hand-off cap, the retention cap (oldest step evicted
first, and completing it is an `unretained_event`, never a spill), the once-per-event
de-duplication, the scan cap that skips a huge completed map instead of copying it.
`spill-one` is exercised end to end across the relay - request file, rank poll, the
receipt's `queued` verdict and its `expected_path`, and the file really present with
the slot's bytes after a drain - with a second poll refusing to answer the same ask
twice, a malformed or wrong-version or expired request counted and ignored, and a
rank refusing an ask whose fingerprint is not its own. Nothing raises: every door is
opened with a tensor whose attributes explode, a `None` config, an `op` that is an
object; `KeyboardInterrupt` from the fork still escapes; and an AST walk asserts the
module imports only the standard library and its own package (function-level imports
included), so the engine is reached by NAME and never at import time.

The lookup ledger (`tests/test_lookup_ledger.py`) is the one that proves the
composition rules instead of asserting them: the cap and the monotonic `dropped`,
lookups and adopts interleaved in a different order and joined by request id, the
adopt that never had a lookup, the lookup whose request was skipped, a re-offer after
a failed allocation, a re-install that must leave exactly ONE guard layer (it walks
the marker chain and asserts the fork's function is still underneath), a
backport-replaced method with `backport._layers_of` saying `["guard", "backport"]`,
a foreign owner marker refused with the refusal QUOTED in the arm reason (not a
generic "no tier class"), a disarmed ledger that never even evaluates its arguments,
a trace disarm that cannot blind it, and a fork method that raises - which must
propagate exactly as it would unwrapped, because the ledger is not guard mode. The
parked-KV index reading is pinned in `tests/test_snapshot.py`: the live 664-free /
4372-entries pair reported side by side, per-group counts decoded from the packed key,
the scan and group caps honoured, a `len()` that raises named as that rather than as
a missing handle, and three separate ways to get a `reason` instead of a number.

Per-request miss attribution (`tests/test_miss_attribution.py`, 45 of them) is
pinned on the numbers and on the honesty of the gaps: the query length, the cap the
fork puts on it and the residue, each carrying the unit it claims; the derived
first-miss position and the walk property it is derived from; every verdict in the
enum reachable or the file fails; a clamp group named ONLY when the probe's minimum
actually binds the served length (and never the parked-nothing scratch ring);
`unknown` emitted when the CPU pool cannot be measured, instead of a confident
`root_missing`; that nothing recorded keeps an engine block alive (weak references,
because holding one is the PR #56810 bug class); the ledger bounded under 5,000
lookups; a `requests` read REFUSING over its byte budget instead of silently
returning a shorter list of "the requests that missed"; the measured retention split
into joint and aggregate from UNEQUAL per-group spans (a fixture where the groups tie
cannot tell a minimum from a maximum); `free != empty`, including the pinned-but-
parked spans; the free-order witness both CONFIRMING and REFUTING, and reporting
`INCONCLUSIVE` when it has seen nothing; the coordinator and pool wraps being
instance-level and restoring the stored originals; a ratio of scheduler steps against
blocks being refused outright; the file mirror env-gated and stopping at its cap; and
that a GENERATOR free list (`manager.py:1297`) reaches the engine's own
`free_blocks` INTACT - observing it first would leave the engine freeing nothing,
which is how a read-only witness would have become a leak; and ten source mutations
run against `_decide` and the retention arithmetic
(`offered > 0` -> `>=`, a residue sign flip, an alignment-gate sign flip,
`first_miss_span` off by one, joint `min` -> `max`, dropping the referenced spans
from the parked total, the deep-probe budget ignored, and the generator guard
        removed) each one shown to break a test.

The backport and the kill switch (`tests/test_backport.py`, 67 of them) drive the
fork's own
method text (fingerprints match, so the real install path runs) against the
production geometry, with a poison-pill `resolve_block_hashes` that reproduces
the assert - so "the patch works" and "the unpatched selector crashes" are
behavioural claims. The boot posture has its own section: a **fresh subprocess
that only imports the plugin is disarmed** (the fixtures reset state, so they
cannot prove the first instant of a real process); the boot-time install refusal
leaves the tier inert and reports `arm_block.kind: "no_manager"`; the cold-boot
`status` -> `arm` sequence installs then arms; arming seeks its own proof when a
caller skips `ensure_installed`; each refusal kind is refused, named, and
explained in the verb payload; a class-level install cannot forget a geometry
veto a live config already gave; a disarmed tier produces no store/load work as
observed **through the wrappers** (the fork's own no-op shapes, with the engine's
methods never called).

The invariant itself is proven rather than asserted, and paired with a control so
it cannot be vacuous: with the backport REFUSED (the store selector left
unpatched, the worst install there is) and the tier disarmed, a spy on
`_select_eager_blocks_to_store` records **no call at all** from
`request_finished_all_groups` or `build_connector_meta`; the mirror with the fix
installed and the tier armed sees both routes reach it (and skip the ring); and
the control - the same class with NO wrappers fitted - raises the production
assertion on both routes. The install ORDER is pinned by an install that explodes
halfway through the patch (all four wrappers are already on the class, the tier is
disarmed, both routes stay gated) and by a geometry veto, which now patches
nothing and therefore MUST get the gate. And the boot itself is tested without a
single HTTP verb: `register()` fits the factory hook, a fork-shaped
`create_connector(config=..., role=..., kv_cache_config=...)` returns the very
connector it built, and by then `installed` and every wrapper marker are true -
including a LIVE geometry check, which a class-only install can never do. The
worker role is a `no_manager` answer, not an error; a hook that raises is counted
and the engine still gets its connector; the fork's own `create_connector`
exceptions propagate untouched; a foreign hook marker is refused; and guard mode
composes with the new wrapper in BOTH fit orders, handing it back by identity.

Bringing the tier up working, unattended, is its own file
(`tests/test_auto_boot.py`, 30 of them), because the claim is a sequence and a
sequence needs ordering proven: a clean install arms guard mode and the tier with
**no verb anywhere in the test**; the guard is armed *before* the tier is armed
(spy-recorded, not asserted about, so a reorder fails here); a partial install, an
instance the live-object check cannot make safe, and an entry point the guard
refused to wrap each refuse to arm and name themselves in the payload and in the
one BOOT line; `not_covered` reports what the guard does not wrap without vetoing
the arm - including the production shape where `take_events`/`reset`/
`update_connector_output` are inherited from the connector base module and can
never carry a guard layer (an earlier gate read "attribute exists, therefore guard
it" and would have refused to arm on every real engine); an inherited limit does
not relax an *owned* unguarded method, which still blocks; a second boot composes
into exactly one guard layer with the backport's and switch's layers intact
beneath it; the opt-out resolves with the same falsy vocabulary as every other
flag and says so in the payload; and a raising tier method becomes a recorded
decline that survives a disarm, a re-arm and the next boot's `faults=N`.

The debugger verbs add six files:

| File | Tests | What it pins down |
|------|-------|-------------------|
| `tests/test_kvconfig.py` | 40 | the divisibility table against the fork's assert (`kv_cache_utils.py:2888`), a Mamba group's different `hash_block_size`, the QSA ring recognised by class name, `prefix_cacheable` never inferred from `host_rotation`, the geometry-vs-manager disagreement, every cap honouring `dropped` counts, a missing operand reported as `unknown` and never as safe |
| `tests/test_kvinspect.py` | 41 | block/request inspection against a production-shaped fake: the packed block-hash decode, `in_cached_map` via the tier's own dict, `coverable_blocks` and `binding_term` recomputed by hand, the `ids_cap`/drop reporting, every "does not exist" answer naming what does |
| `tests/test_trace_ring.py` | 36 | the ring's fixed capacity and wrap accounting, `read` never walking O(depth), the `gap_records` a slow poller gets, `clear` not resetting `seq`, arming installing and disarming removing the observers, a disarmed ring costing no allocation and no wrap, the step observer returning the fork's value untouched |
| `tests/test_guard_mode.py` | 49 | every declined method returning the fork's documented no-op shape (and the one method that must re-raise), `BaseException` never swallowed, the metadata rebuild falling back to a re-raise if the fork renames its class, captures bounded with the full text only via the `captures` op, `guard_mode: true` in every payload, disarm restoring the originals by identity, tracing keeping the wrappers alive after a disarm, **no env var that can arm it** (the boot does, and the opt-out can only turn that off) |
| `tests/test_worker_diag.py` | 43 | the file schema against the reader's contract (`version`/`epoch`/`ts`/`role`/`rank`/`arena`/`counters`), the rank name matching the file name, an unknown rank meaning no file at all, the cadence and the time floor, "a change writes now", the empty pool still writing zeros, atomic replace and the no-truncate rule, the size cap trimming prose before data, a full disk failing a step never, `bytes_used` as a proven floor |
| `tests/test_loaded_version.py` | 46 | WHICH CODE IS RUNNING, published: every surface (`/snapshot`, `/status`, the rank file) publishes `kvtier_dev_plugin.__version__` in EVERY state it can be in (tier present/absent, guard armed/disarmed, the containment path; never-observed/mirrored/blockdump-armed/fill-degraded/inert status; plain, in-flight, all-sections, refused, raised and degraded rank documents), a runtime change to `__version__` moves the payload so no field can be a copied literal while the schema `version` stays 1 in all three, `importlib.metadata` is never consulted and its answer never appears in a payload, installed-vs-loaded disagreement resolves to the loaded value, the three surfaces name three different processes and `/status` cannot forward the engine's value nor `/snapshot` the API server's, the keys survive the rank file's byte ceiling and its section budget, a contributor cannot take a writer-owned key, the file is still `counted` by `disk_store`'s reader, and the version is one fact in `__init__.py`, `pyproject.toml` and `uv.lock` |
| `tests/test_quiet_rank_bytes_are_not_dead_bytes.py` | 12 | a rank that has not written for 300 s keeps its bytes while its worker is alive and publishing that digest (the defect, on a real filesystem); the peer map dropping a quiet rank's digest is not evidence of death; liveness from the RANK FILES, not the digest map; `stale_geometry` reclaim still runs and still runs ahead of this rank's own cold bytes; `stale_geometry_removed` and `peer_files_removed_live` separate the two doctrines; an empty peer map claims nothing, so no identity may be called dead on it; `identity_known`/`liveness_source` name the evidence a pass had |
| `tests/test_index_forgets_no_bytes_on_disk.py` | 13 | eviction is LRU-by-TOUCH and not insertion order (a just-restored identity is the last to leave, and on the unpatched tree it was the first); a read that promotes a file touches the index on the throttled path too; `VLLM_E2_TIER_DISK_INDEX_ENTRIES` configures the cap and a bad value falls back and reports; `entries_dropped` balances its three reason counters exactly; `entries_dropped_file_present` counts the index forgetting a live file and never guesses at an unchecked one (`unverified`); `note_restored_path` re-admits an identity a restore proved |
| `tests/test_idle_engine_keeps_its_disk_proof_alive.py` | 10 | install alone with ZERO engine steps produces and refreshes the rank file (the defect: only a step used to write it); `disk_restore.worker_proof` stays open across an idle spell and re-opens after a forced-stale file with no step in sight; the heartbeat starts at `install()` and never stacks a thread; it is a daemon with a findable name; the handler is held WEAKLY and a beat with no handler still writes; the write floor bounds forced beats; no `VLLM_E2_TIER_DIAG` means no thread and no file; the shipped interval fits the 900 s window with four beats to spare (pinned against the UNPATCHED default); the payload reports age and margin; a dead thread is respawned and counted |
| `tests/test_idle_worker_proof_stays_alive_and_legible.py` | 24 | THE 0.2.47 SUITE, driven from `engine_side.register()` plus the handshake hook exactly as a worker walks them (the plugin-load install must DECLINE there, the handshake must land it), with zero engine steps and the READER's clock moved so a test can wait fifteen minutes: the gate stays open across two idle spells nearly the length of the window and still closes on a beat that stopped, on a future-stamped file, and on a file whose `disk_restore` section is gone; a contributor wedged for three budgets cannot stop the rank file moving; a beat raising `SystemExit` cannot kill the supervisor; a `SystemExit` contributor costs its own section only; the in-flight bound says `jammed` instead of leaking threads; `writer.heartbeat` in all five states with one key set inside its byte budget; `GATE AT BOOT` / `GATE CLOSED` / `GATE OPEN again` once per flip, naming the reason, the per-file counts and the freshest file's age AND wall clock, and distinguishing `stale=` from `sectionless=`; the BOOT posture itself - OPEN on an untouched 0.2.69 boot, and still `open=False rank_files=0` (a posture, never a closure) for a boot whose proofs are gone - and the `GATE OPEN` line earned by a beat that writes; the refusal that never started is said once at WARNING; the shipped cadence is pinned against the UNPATCHED default |
| `tests/test_cold_boot_primes_the_rank_proof.py` | 5 | THE 0.2.69 FIX, on the SHIPPED 60 s cadence so the supervisor's own beat cannot be what satisfies an assertion: a fitted worker plus the handshake hook with ZERO engine steps must satisfy `disk_restore.worker_proof(force=True)` on the spot (the old code answers `no_worker_proof`, which cost a 61 s cold prefill on 231,392 tokens); the primed file must carry the fork's EXACT arena (`bytes_total: 4096`, because a beat with no handler yet publishes a zero, and a zero is a false number rather than an empty one) AND a `disk_restore` section with `worker_fitted` raised (a section-less fresh file is verdicted `no_restore_section` and still reads as no proof, which is why the prime runs after the install loop); a prime raising inside `heartbeat_once` must cost one counted containment and one log line, never the handshake, and the loop's later beats must still write the file; no `VLLM_E2_TIER_DIAG` means the prime declines before the beat - no rank file, no thread, no prime line; and a boot that receives NO REQUEST AT ALL ends with the gate `open`, this boot's file counted fresh, and zero sectionless rank files; and eight CONCURRENT writes of one rank file lose NO proof - since 0.2.69 the handshake prime and the immediately-due first tick write at the same instant by design, and the temp half of the atomic replace used to be named with a completed-write counter, so two beats in flight shared a path and the loser's `os.replace` raised `FileNotFoundError` (6 of 8 runs lost a write; the name now carries the thread ident and a nanosecond stamp) |
| `tests/test_routes_dev_verbs.py` | 57 | every argument rule twice (GET query and POST body) against one table, an unknown key `400`-ing without reaching the engine, a `413` before any parse, bool-is-not-int and int-is-not-bool, control characters refused, the route->method map being un-hijackable by a body, the empty body sending no positional argument, `guard`/`trace` never mutating the policy module, the `/status` mirror staying `null` until an engine speaks |
| `tests/test_dev_surface_e2e.py` | 29 | the routes wired to the REAL engine-side verbs on a fake engine, with the fork's own `len(args) <= len(parameters)` assert reproduced on the wire: every verb driven with and without an argument, every payload surviving JSON, a block/request read against the production-shaped geometry, a captured fault agreeing with the ring, the snapshot and the `/status` mirror, disarm restoring the fork's functions, and the guard composing with this package's own tier wrapper instead of leaving the fault-prone methods unguarded |
| `tests/test_boot_journal_drain.py` | 12 | THE 0.2.65 FIX: a journal of the incident's shape (4 ranks x 110k real `<seq> <+\|-> <key16> <bytes> <group>` lines, ~16 MB, built once per module, digest-pinned as READ-ONLY) drained on a clock that advances by `bytes read / 2.2 MB/s`, the box's measured throughput - a wall-clock deadline timed in this host's time (it reads the same journal ~4.5x faster) measures the host, not the box. The headline test asserts its OWN fixture is deeper than the retired 5.0 s boot budget and shallower than the shipped 45 s, then requires `converged: true` with `lag_bytes: 0` and an `elapsed_seconds` ABOVE the retired bound (the same drain the old deadline stopped blind); the deadline and the byte budget each stop a deeper journal, are REPORTED as `stopped_at: deadline` / `byte_budget` with the bound in force, do not raise, and leave the rest draining one bounded pass per refresh on the request path; an unparseable env value leaves each default standing with `*_source: env_ignored` and quotes the variable and its raw value back in `problems`, while a well-formed one lands and is the number the next refresh honours; UNSET is today's numbers, the steady-state cap still 512 KiB and the line cap still 8,192; the record and its limits reach `restore.index` beside `converged`/`blind` and carry their own `units`; no arm, no posture, no switch moves - behaviourally AND as an AST fact about `disk_index.py` and about the boot's own `_boot_journal_drain`; a drain that raises is counted, logged once, recorded as `error` and leaves `boot_drained: false`, while `KeyboardInterrupt` propagates; and the fork's own `create_connector` (the `scheduler.py:164` shape, fake factory in `sys.modules`) returns a connector whose index is ALREADY drained - with the drain's state captured DURING construction proving the fork's constructor ran first - the BOOT line carrying `JOURNAL_DRAIN[converged]` without disturbing the restart verifier's `tier=` last-match; an inert boot reads no journal byte at all |
| `tests/test_mamba_anchor_patch.py` | 28 | THE 0.2.64 TARGET SPEC, on the fork's own unpatched `reachable_block_mask` text in `fixtures_mamba.py`: the mask marks a run end only under the patch, a dense `None` and an EMPTY mask pass through UNTOUCHED, a mask already ending True is not recounted (the anti-vacuity control that keeps the counter measuring the patch and not the retention interval), a call's other arguments and return identity survive; a mutated body REFUSES with both hashes named and the unpatched answer intact (the shipped fingerprint is re-derived from the fixture in the test, so editing the pin cannot fix a red test); a `staticmethod`, a subclass hiding the classmethod, an unknown class and an already-foreign-patched class all refuse and leave `mamba_patch_installed` False while the test's own class stays exactly the object it was; the wrapper's OWN body is `except BaseException: raise` (AST-pinned, like the connector's kill-switch wrappers), and a raising original costs the counter; the env flag reads `KVTIER_DISK_BOOT_ENABLE`'s grammar from `config.py` and `env-refused` is counted as enabled + a reason in the payload; the boot token is bracketed so the restart verifier's `tier=` grep cannot match it; the counter doors each fail independently (the spill counter reaches the cross-rank rollup via `RANK_ROLLUP_FIELDS` alone; the restore counter moves for an anchored run and NOT for a run with no anchor, same walk, same boot); and nothing about serving posture changed - behaviourally AND as an AST fact about this module's own source, with the restore layer's only reach into the patch pinned to the one read-only liveness call |

The store-policy hook (`tests/test_store_hook.py`, 55 of them) runs against a
stub built by `exec`ing a source string into a real module object, so the
wrapper resolves `_StoreAdmission` out of that module's globals exactly as the
fork's own call sites do. It covers: the stub classifying like `manager.py:932`
before we touch it; a veto stopping a store the engine would have made; a veto
landing in `skipped_already_cached` and *never* in `dropped_cpu_full`; the policy
never being asked about a block the engine already declined; a raising policy and
a raising holder both keeping the engine's verdict while counting a fallback; a
`BaseException` still propagating; every unrecognised verdict (a typo, `True`,
`1`, `{}`) falling through and being counted; a renamed enum making the hook
veto nothing; double install being a no-op that cannot nest wrappers; a foreign
owner (foreign module, or another plugin's marker) being refused; the absence of
the module being reported without importing `vllm`; the `ctx` shape; the
snapshot's `store_hook` section (including `governs_current_mode` going false in
lazy mode); the per-call cost; and the kill-switch procedure end to end through
the two verbs the plugin actually exposes.

The capacity clamp (`tests/test_cpu_capacity.py`, 63 of them) runs against two
stub layers built to the fork's shapes: the connector base class `exec`'d into a
module NAMED `cpu_capacity.BASE_MODULE` (so the ownership check sees the module
the real methods are defined in, and a foreign `__module__` is refused), carrying
`base.py:716-731` including its `pp_rank > 0` `ValueError`; and a
`BlockPool`/`FreeKVCacheBlockQueue`/`KVCacheBlock` transcription with the fake
head/tail sentinels that make "is this block in the queue?" a decidable
question. It covers: min-across-ranks being the cap and a disagreement never
picking the max; the cap never being raised above the scheduler's own view; a
rank reporting nothing, reporting 0, or sending a payload without a capacity all
selecting zero mode instead of uncapped; a bare carrier contributing a capacity
and NO entry (the crash case); the original payload reaching the original method
intact, including a non-dict payload handed back as the identical object; the
original's own exception still propagating with the clamp applied first; double
install, foreign markers, renamed and non-callable targets; the drained pool
never yielding an id at or above the cap while `cap-1` is handed out; the GPU
pool acquiring nothing; a failed surgery reporting `clamp_failed_open` and never
a false `zero_mode`; and adoption.

Against the real fork classes (CPU-only, outside the fork's tree) run
`scratch/test_cpu_capacity_clamp_live.py` from the fork root:

```bash
cd /home/user/Source/experiment/vllm-dev
PYTHONPATH=/home/user/Source/experiment/vllm-testenv:\
/home/user/Source/experiment/kvtier-dev-plugin \
/home/user/Source/experiment/vllm-testenv/.venv/bin/python -m pytest \
    --noconftest -q /home/user/Source/experiment/scratch/test_cpu_capacity_clamp_live.py
# -> 10 passed
```

It builds a real `SimpleCPUOffloadConnector` (both roles) and a real
`SimpleCPUOffloadScheduler` with the fork's own test builders, installs the
wrappers on the real `KVConnectorBase_V1`, and asserts the pool ends up capped,
that no id at or above the cap can leave the real free queue, that the fork's own
consumer did not raise, and that `SimpleCPUOffloadConnector` never grew its own
copy of either method. The fork's own tier suite is unaffected: `84 passed`
(`tests/v1/simple_kv_offload/{test_scheduler,test_kv_events,test_hip_mem_ops}.py`).

