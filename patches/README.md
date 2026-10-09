# `patches/` — the vLLM patch series (canonical home)

Four hunks on two files of the running vLLM. **This directory is the source of
truth for those bytes.** Nothing else may carry a copy of them: the llm-api repo
holds only the SSH transport that carries them to the GPU box
(`config/box/kvtier-patch.sh`, which reads these directories — see "Who reads
this directory" below). A duplicate that can drift is the outcome this layout
exists to prevent.

```
patches/
  README.md                  this file: the provenance, end to end
  base.manifest              the same facts in machine-readable form (sourced by the applier)
  series                     the patch directories, in the order they must apply
  apply-kvtier-patches.sh    the applier for a LOCAL vLLM checkout (status|apply|revert)
  0001-kvtier-pr56810/         manager.py.patch + patch.manifest
  0002-kvtier-cpu-release-order/  manager.py.patch + patch.manifest
  0003-kvtier-disk-restore-meta/  metadata.py.patch + patch.manifest
  0004-kvtier-disk-restore/       manager.py.patch + patch.manifest
```

Each patch directory holds a plain `git apply -p1` diff and a shell-sourced
`patch.manifest`: the patch's own `patch_sha256`, the target file's sha256 before
and after (`file_sha_pre`, `file_sha_post`), the per-method fingerprints
(`methods`, `methods_pre_sha256`, `methods_post_sha256` — the same quantity
`kvtier_dev_plugin/backport.py` hashes at runtime), and the box-side coordinates
(`target_abs`, `git_root`, `git_rel`, `engine_python`) the transport needs.

## The base these four hunks sit on

Three layers, each with a hash that identifies it. A reader with nothing but this
file can rebuild the box's source tree:

| Layer | Identity |
|---|---|
| 1. Upstream vLLM | `vllm-project/vllm` at `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, which is the `v0.30.0` tag |
| 2. The packaging delta | `halt95/qwen38-flash-next-3090s` — `https://github.com/halt95/qwen38-flash-next-3090s.git` — 58 commits in `release/v2.2/patches/` (`0001-…patch` … `0058-…patch`), applied on top of the base commit by `scripts/build-v2.2.sh` |
| 3. **This series** | the four hunks in `series`, applied in that order |

Layer 1 + layer 2 produce a git **tree** with one permanent name, independent of
how the commits were re-packed:

```
base tree  eebcd6480dba463294e9ca34f24eb770a45e11fe
```

That tree hash is what `git rev-parse 'HEAD^{tree}'` answers in the GPU box's
vLLM checkout (an editable install: the engine imports that checkout), and it is
what the box reports as its HEAD tree — so the box is *exactly* these three
layers, with nothing hand-edited on top. Layer 1 + layer 2 + this directory
produce:

```
live tree  ed5e079fe945de96b09c01c4559f0952fcf809f1
```

Rebuild it from scratch (the packaging repo's own script does the fetching and
asserts both hashes itself):

```
git clone https://github.com/halt95/qwen38-flash-next-3090s.git
cd qwen38-flash-next-3090s
scripts/build-v2.2.sh ~/vllm-workspace ~/venv        # asserts commit ced6857… and tree eebcd648…
<plugin>/patches/apply-kvtier-patches.sh ~/vllm-workspace apply \
    --expect-tree ed5e079fe945de96b09c01c4559f0952fcf809f1
```

`--expect-tree` is the whole acceptance check: the applier computes the tree the
working copy would have if the patched files were committed (in a scratch index —
your own index is never touched) and fails unless it equals that hash. On
2026-10-11 this was run against `/home/user/Source/experiment/vllm-live-repro`,
whose branch `built` is layer 1+2 (tree `eebcd648…`) and whose branch `live` is
layer 1+2+3 (tree `ed5e079f…`): applying `series` to `built` reproduced `live`
byte-for-byte on both files and changed nothing else, and the computed tree was
`ed5e079fe945de96b09c01c4559f0952fcf809f1`. `tests/test_patch_series.py` runs
exactly that check on every suite run (and skips with a printed reason, never a
fake pass, if that reconstruction clone is absent).

## Order is a dependency

`series` is the order, and the applier enforces it (numbered `0001…0004` with no
gaps, and it refuses a patch directory that `series` does not list). Two distinct
reasons the order is not a preference:

1. **A hash chain on `vllm/v1/simple_kv_offload/manager.py`.** Each patch's
   `file_sha_pre` is the previous patch's `file_sha_post`:
   `0001 → 0002 → 0004`. Out of order, the later patch's pre-image simply does
   not exist and the applier refuses, naming both hashes.
2. **A semantic dependency across files.** `0003` (on `metadata.py`) declares
   `store_keys`, `store_fingerprint` and `load_paths` on the metadata object;
   `0004` passes those fields as constructor arguments. They are different files,
   so no hash chain can express it — the numbering does.

The chain is also why `apply` is idempotent the way it is: for each target file
the applier finds the highest post-image the file already carries, reports every
patch up to and including it as `ALREADY`, and applies only the rest. A
fully-patched checkout therefore writes nothing and says so per patch, rather
than double-applying.

## What the four hunks are

| Patch | File | Change |
|---|---|---|
| `0001-kvtier-pr56810` (+9/−1, 4 hunks) | `manager.py` | Upstream PR #56810 (issue #56396): honour `prefix_cacheable_group_ids`. Without it a hybrid model whose scratch/QSA group block size (8) does not divide the hash block size dies on `assert block_size % hash_block_size == 0` on the FIRST store request. |
| `0002-kvtier-cpu-release-order` (+23/−6, 4 hunks) | `manager.py` | Release CPU blocks tail-first, so a partly-reclaimed chain keeps its root (a parked prefix survives instead of being orphaned). |
| `0003-kvtier-disk-restore-meta` (+22/−0, 2 hunks) | `metadata.py` | Declare `store_keys`, `store_fingerprint`, `load_paths` — the data shape the worker reads. |
| `0004-kvtier-disk-restore` (+32/−0, 2 hunks) | `manager.py` | Fill those fields on the scheduler side: put the block hash on the metadata object the worker reads (SSD tier, part 1/2). |

## Who reads this directory

* `patches/apply-kvtier-patches.sh` (here) — a **local** checkout: `status`,
  `apply`, `revert`, `--dry-run`, `--expect-tree`. No SSH, no engine.
* `llm-api`'s `config/box/kvtier-patch.sh` — the **GPU box**: the same
  directories over SSH, adding the marker ("patched by us"), the timestamped
  backups outside the checkout, and verification with the engine's own
  interpreter (`py_compile`, a real import, the four method fingerprints). It
  resolves this directory from `KVTIER_PLUGIN_REPO` (default
  `/home/user/Source/experiment/kvtier-dev-plugin`) and refuses loudly, with the
  clone command, when that checkout is absent — it has no fallback copy to fall
  back to, because these hunks no longer live in that repo.

Both appliers obey the same rule, stated in `apply-kvtier-patches.sh`'s header:
**a stale fingerprint is a refusal naming both hashes, never a best-effort
merge.** There is no `--force` in either one. This repo ships a patch plus
fingerprints, not whole files, so the only post-image either tool can prove is
the one `git apply` reproduces from the pre-image; forcing could only install
bytes nobody can verify. Neither tool starts, stops, restarts or signals the
engine, and neither sends it a request: a patched file is inert until the
operator boots it.

## Changing a hunk

1. Edit the engine file in a checkout that sits on `base_tree`, regenerate with
   `git diff` so the result still applies at `-p1` with zero offset.
2. Re-pin **every** hash in that directory's `patch.manifest` (patch sha, both
   file hashes, both method lists) and, if the change moves the tip, `live_tree`
   in `base.manifest` and in this file.
3. Run `tests/test_patch_series.py` — the acceptance test is what makes "re-pinned"
   mean something.
4. Then, and only then, apply: locally with the applier here, on the box with
   `config/box/kvtier-patch.sh <name> apply` (which refuses a non-pre-image and
   verifies on the box before it writes the marker).
