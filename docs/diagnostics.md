# HTTP surface: argument rules and payload fields

Moved from `README.md`. Reference for the dev route surface: the shared argument
validation and the fields `/snapshot` publishes. The route table and the
read-only commands an operator runs stayed in `README.md`.

**Arguments.** GET takes them as query parameters, POST as a JSON object body, and
both are validated against the SAME per-route table (`endpoint.ARG_SPECS`), so a
bound cannot be evaded by changing the method. An unrecognised key is a `400` that
never reaches the engine - including a stray `?cachebuster=1` on `/snapshot`. A
**POST reads BOTH forms** (since 0.2.21): `POST /disk?op=arm&seconds=3600` with an empty
body arms the tier, and a key given in the query string AND the body must carry the same
value or the answer is a `400` naming both - there is no silent winner, because until
0.2.21 the POST handlers read the body only and a query-string `op` was **ignored while
answering `ok: true`** (`?op=arm` ran `status`, the arm file kept its mtime, and the
response claimed success). A validated body is forwarded to the engine as ONE dict argument
(`call_utility_async(method, args)` -> `getattr(self, name)(*args)`,
`vllm/v1/engine/core.py:1669-1671`); a call with no arguments still sends none, so
nothing that worked before arguments changed. Which engine method a request reaches
is fixed by the route and can never come from the body.


`snapshot` returns the CPU pool totals/free, GPU pool totals/free,
`pending_stores`, the ten in-flight/queue lengths, the load/store event counters,
the eight `BoundaryStoreStats` decline reasons, scheduler lengths, the plugin's
per-boot `epoch` (proof that two snapshots came from the same boot), `plugin_version`
(the version of the code the ENGINE process imported; the top-level `version` beside
it is the document's SCHEMA and has been `1` all along), the policy's
own verdict plus `decide_us` and `build_us`, the `store_hook` patch report (below)
that says whether the runtime veto is actually live, the `cpu_capacity` clamp
report, and the development sections - `guard_mode` (a top-level copy of the answer,
because a guarded engine must never be mistaken for a production one), `guard` (its
bounded previews), `trace` (the ring's counters and the step observer's), and
`worker_diag` (what this process knows about the per-rank file writer, with
`worker_scope` saying that the truth lives in the files).

