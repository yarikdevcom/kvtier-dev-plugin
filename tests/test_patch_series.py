"""Tests for `patches/` - the vLLM patch series and the applier that owns it.

WHAT THIS SUITE IS FOR
----------------------
`patches/` is the canonical home of the four hunks that modify the running vLLM
on the GPU box. Two of its artifacts are code and therefore testable: the
`series`/`*.manifest` metadata, and `apply-kvtier-patches.sh`. The applier's
contract is a refusal contract - "a stale fingerprint is a refusal naming both
hashes, never a best-effort merge" - and that is what most of what is below
pins, because a patch tool that degrades into "tried its best" is how a
half-patched engine gets booted.

Two layers of testing, deliberately:

* A SYNTHETIC series (three small patches over two small files, generated into a
  scratch git repo) drives the applier's behaviour: ordering, idempotence,
  refusal text, dry-run, revert, the untouched index. Synthetic because the real
  patches carry vLLM's own context lines and cannot be applied to a hand-shaped
  fake - and faking one would test a patch that does not exist.
* The REAL series is checked two ways: internally (numbering, the per-file hash
  chain, every `patch_sha256` describing the bytes it ships beside, and the
  machine-readable provenance agreeing with `README.md`, which is what a human
  reads) and against the live-tree reconstruction described in `base.manifest`:
  branch `built` is upstream v0.30.0 + the 58-patch packaging series, branch
  `live` is that plus these four hunks and is byte-identical to what the box
  imports. `test_applying_the_series_to_built_reproduces_the_live_branch...`
  applies the real series to the real base and compares trees and bytes. It
  SKIPS with the reason printed when that clone or its branches are not what
  `base.manifest` claims - a skipped acceptance check is honest, a simulated one
  is a lie, and this suite is the only thing standing between a re-pinned
  manifest and a box that no longer matches it.

Nothing here touches the GPU box, an engine, or a network. Every git operation
runs in a `tmp_path`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches"
APPLIER = PATCHES / "apply-kvtier-patches.sh"
BASE_MANIFEST = PATCHES / "base.manifest"
SERIES_FILE = PATCHES / "series"
README = PATCHES / "README.md"

GIT = shutil.which("git") or "git"
GIT_CONFIG = [
    "-c", "user.name=patch-series-test",
    "-c", "user.email=patch-series-test@example.invalid",
    "-c", "commit.gpgsign=false",
    "-c", "init.defaultBranch=main",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(cwd: Path, *args: str) -> str:
    r = subprocess.run(  # noqa: S603 - a fixed git binary, fixed argument shape
        [GIT, *GIT_CONFIG, *args], cwd=str(cwd),
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r.stdout


def read_manifest(path: Path) -> dict:
    """Read a shell-sourced manifest the way the shell would (quotes stripped)."""
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key.strip()] = val
    return values


def series_entries(path: Path) -> list:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def run_applier(series_dir: Path, checkout: Path, verb: str = "status",
                *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # KVTIER_VLLM_CHECKOUT / KVTIER_PATCH_SERIES_DIR are the applier's own
    # knobs; a value leaking in from the caller's shell would silently retarget
    # a test, so they are pinned here rather than merely unset.
    env["KVTIER_PATCH_SERIES_DIR"] = str(series_dir)
    env["KVTIER_VLLM_CHECKOUT"] = str(checkout)
    return subprocess.run(  # noqa: S603 - a fixed interpreter running a fixed script
        ["/bin/bash", str(APPLIER), verb, *extra],
        capture_output=True, text=True, env=env, timeout=300, check=False,
    )


def combined(res: subprocess.CompletedProcess) -> str:
    return res.stdout + res.stderr


def verdict_lines(out: str, kind: str) -> list:
    """The per-patch verdict lines only - not the summary sentence.

    "APPLIED  3 patch state(s) written" is a sentence the tool says; "APPLIED
    0001-first ..." is a verdict. Counting the substring would call the first
    three verdicts, which is exactly the difference a refusal test needs.
    """
    return [line.strip() for line in out.splitlines()
            if line.startswith("     " + kind)]


# ---------------------------------------------------------------------------
# A synthetic series, so the applier's behaviour can be driven without vLLM.
# `pkg/mod.py` carries a TWO-PATCH CHAIN (patch 1's post-image is patch 2's
# pre-image, exactly like manager.py's 0001 -> 0002 -> 0004) and `pkg/other.py`
# a single patch, because "one file, several hunks" and "two files, one
# independent" are the two shapes the real series has.
# ---------------------------------------------------------------------------
MOD_BASE = """class Scheduler:
    def __init__(self, cfg):
        self.cfg = cfg

    def target(self, groups):
        total = 0
        for g in groups:
            total += g.size
        return total

    def pick(self, blocks):
        return [b for b in blocks if b.free]
"""

MOD_ONE = MOD_BASE.replace(
    "        self.cfg = cfg\n",
    "        self.cfg = cfg\n        self.cacheable = cfg.cacheable\n",
).replace(
    "        for g in groups:\n",
    "        for g in groups.cacheable:\n",
)

MOD_TWO = MOD_ONE.replace(
    "    def pick(self, blocks):\n        return [b for b in blocks if b.free]\n",
    "    def pick(self, blocks):\n"
    "        # tail-first, so a partly-reclaimed chain keeps its root\n"
    "        return [b for b in reversed(blocks) if b.free]\n",
)

OTHER_BASE = "FIELDS = (\"tokens\", \"blocks\")\n"
OTHER_ONE = "FIELDS = (\"tokens\", \"blocks\", \"store_keys\")\n"


def make_patch(tmp: Path, pre_text: str, post_text: str, rel: str) -> str:
    """A real `git diff` patch, paths rewritten so `git apply -p1` applies it."""
    work = tmp / "_patchgen"
    (work / "a" / Path(rel).parent).mkdir(parents=True, exist_ok=True)
    (work / "b" / Path(rel).parent).mkdir(parents=True, exist_ok=True)
    (work / "a" / rel).write_text(pre_text)
    (work / "b" / rel).write_text(post_text)
    r = subprocess.run(  # noqa: S603 - a fixture building a patch from its own bytes
        [GIT, "diff", "--no-index", "--src-prefix=a/", "--dst-prefix=b/", "a", "b"],
        cwd=str(work), capture_output=True, text=True, check=False)
    out = []
    for line in r.stdout.splitlines(keepends=True):
        if line.startswith("diff --git a/a/"):
            line = f"diff --git a/{rel} b/{rel}\n"
        elif line.startswith("--- a/a/"):
            line = f"--- a/{rel}\n"
        elif line.startswith("+++ b/b/"):
            line = f"+++ b/{rel}\n"
        out.append(line)
    text = "".join(out)
    shutil.rmtree(work)
    assert "@@" in text, "the fixture patch has no hunks"
    return text


def write_patch_dir(series_dir: Path, name: str, pid: str, rel: str,
                    pre_text: str, post_text: str, tmp: Path) -> dict:
    d = series_dir / name
    d.mkdir(parents=True)
    text = make_patch(tmp, pre_text, post_text, rel)
    pf = d / f"{Path(rel).name}.patch"
    pf.write_text(text)
    added = sum(1 for line in text.splitlines()
                if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in text.splitlines()
                  if line.startswith("-") and not line.startswith("---"))
    hunks = text.count("\n@@") + (1 if text.startswith("@@") else 0)
    # The real manifests are SOURCED by the applier, so the fixture's must be
    # valid shell too: comments need '#', and a value with a space needs quotes.
    manifest = "\n".join([
        "# shell-sourced fixture manifest",
        f"patch_id={pid}",
        f'title="{pid} (synthetic fixture)"',
        f"git_rel={rel}",
        'target_abs="/nowhere/does/not/matter (the local applier uses git_rel)"',
        "target_mode=644",
        f"file_sha_pre={sha_text(pre_text)}",
        f"file_sha_post={sha_text(post_text)}",
        f"patch_file={pf.name}",
        f"patch_sha256={sha_file(pf)}",
        f"patch_hunks={hunks}",
        f"patch_added={added}",
        f"patch_removed={removed}",
        "",
    ])
    (d / "patch.manifest").write_text(manifest)
    return read_manifest(d / "patch.manifest")


@pytest.fixture()
def synth(tmp_path: Path):
    """A synthetic base checkout + a synthetic 3-patch series on top of it."""
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    checkout = tmp_path / "checkout"
    (checkout / "pkg").mkdir(parents=True)
    (checkout / "pkg" / "mod.py").write_text(MOD_BASE)
    (checkout / "pkg" / "other.py").write_text(OTHER_BASE)
    (checkout / "README").write_text("an untracked-by-the-series file\n")
    git(checkout, "init", "-q")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-q", "-m", "the synthetic base")
    base_tree = git(checkout, "rev-parse", "HEAD^{tree}").strip()

    write_patch_dir(series_dir, "0001-first", "first", "pkg/mod.py",
                    MOD_BASE, MOD_ONE, tmp_path)
    write_patch_dir(series_dir, "0002-second", "second", "pkg/mod.py",
                    MOD_ONE, MOD_TWO, tmp_path)
    write_patch_dir(series_dir, "0003-other", "other", "pkg/other.py",
                    OTHER_BASE, OTHER_ONE, tmp_path)
    (series_dir / "series").write_text(
        "0001-first\n0002-second\n0003-other\n")
    (series_dir / "base.manifest").write_text(
        "base_repo=synthetic/base\nbase_version=0.0.0\n"
        "base_commit=" + "0" * 40 + "\n"
        f"base_tree={base_tree}\n")
    # the applier refuses a missing README.md only by not being runnable at all,
    # but keep the fixture shape honest: the real directory has one.
    (series_dir / "README.md").write_text("synthetic\n")
    return {"series": series_dir, "checkout": checkout, "base_tree": base_tree,
            "tmp": tmp_path}


def patched_paths(checkout: Path) -> list:
    return [checkout / "pkg" / "mod.py", checkout / "pkg" / "other.py"]


# ===========================================================================
# 1. The real series' metadata, pinned from the bytes that ship.
# ===========================================================================
def test_the_series_directory_and_its_files_are_self_consistent():
    """`series`, the numbering, and every manifest-describes-its-patch claim.

    Before this directory existed there was nothing to assert: the four patch
    directories lived in another repo with an implicit order held in prose.
    """
    assert SERIES_FILE.is_file() and APPLIER.is_file()
    entries = series_entries(SERIES_FILE)
    on_disk = sorted(p.name for p in PATCHES.iterdir()
                     if p.is_dir() and p.name[:4].isdigit())
    assert entries == on_disk, \
        "the series file and the directories on disk disagree about what exists"
    assert entries == [f"{i + 1:04d}-{n}" for i, n in
                       enumerate(e.split("-", 1)[1] for e in entries)]
    assert len(entries) == 4
    for entry in entries:
        d = PATCHES / entry
        m = read_manifest(d / "patch.manifest")
        for key in ("patch_id", "title", "git_rel", "file_sha_pre",
                    "file_sha_post", "patch_file", "patch_sha256",
                    "patch_hunks", "patch_added", "patch_removed"):
            assert m.get(key), f"{entry}/patch.manifest does not set {key}"
        patch = (d / m["patch_file"]).read_bytes()
        assert hashlib.sha256(patch).hexdigest() == m["patch_sha256"], \
            f"{entry}: patch.manifest no longer describes {m['patch_file']}"
        assert m["file_sha_pre"] != m["file_sha_post"]
        text = patch.decode()
        assert f"--- a/{m['git_rel']}" in text
        assert f"+++ b/{m['git_rel']}" in text
        assert text.count("\n@@") == int(m["patch_hunks"]), \
            f"{entry}: the manifest counts {m['patch_hunks']} hunks"
        added = sum(1 for line in text.splitlines()
                    if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in text.splitlines()
                      if line.startswith("-") and not line.startswith("---"))
        assert (added, removed) == (int(m["patch_added"]), int(m["patch_removed"])), \
            f"{entry}: the manifest's +/- counts do not describe the patch"


def test_the_series_is_a_hash_chain_per_file_and_the_semantic_order_holds():
    """Each patch's pre-image is the previous patch's post-image on that file.

    This is why order is a dependency rather than a preference: a re-ordered
    `series` is not "probably still fine", it is a pre-image that does not exist.
    """
    entries = series_entries(SERIES_FILE)
    by_file: dict = {}
    for entry in entries:
        m = read_manifest(PATCHES / entry / "patch.manifest")
        by_file.setdefault(m["git_rel"], []).append((entry, m))
    for rel, chain in by_file.items():
        for (name, m), (_, nxt) in zip(chain, chain[1:], strict=False):
            assert m["file_sha_post"] == nxt["file_sha_pre"], (
                f"{rel}: {name}'s post-image is not the next patch's pre-image, "
                "so the chain is broken and the order in `series` is a lie")
    # The cross-file semantic rule the chain cannot express: 0004 passes as
    # constructor arguments the fields 0003 declares on the metadata object.
    order = {e.split("-", 1)[1]: i for i, e in enumerate(entries)}
    assert order["kvtier-disk-restore-meta"] < order["kvtier-disk-restore"]


def test_base_manifest_provenance_and_readme_agree():
    """The machine-readable base and the human-readable one cannot drift.

    `README.md` is what a stranger rebuilds the box's tree from; `base.manifest`
    is what the applier enforces. If one names a hash the other does not, the
    rebuild instructions are fiction.
    """
    base = read_manifest(BASE_MANIFEST)
    doc = README.read_text()
    for key in ("base_commit", "base_version", "base_tree", "live_tree",
                "series_repo", "series_patches"):
        assert base.get(key), f"base.manifest does not set {key}"
    for value in (base["base_commit"], base["base_version"], base["base_tree"],
                  base["live_tree"], base["series_repo"], base["series_patches"]):
        assert value in doc, f"README.md never mentions {value!r} from base.manifest"
    assert "release/v2.2/patches" in doc and "scripts/build-v2.2.sh" in doc


def test_the_applier_carries_no_way_to_touch_the_engine():
    """The safety text is a claim about the file, so check the file.

    Same prohibition `config/box/kvtier-patch.sh` pins in the llm-api suite: a
    patch tool that can boot the engine is a patch tool that will eventually
    boot the engine while a request is being served.
    """
    text = APPLIER.read_text()
    lowered = text.lower()
    for word in ("systemctl", "pkill", "killall", "improved-serve", "serve-v2",
                 "nohup", "tmux", "reboot", "shutdown", "vllm serve",
                 "chat/completions", "curl", "wget", "systemd-run", "service "):
        assert word not in lowered, f"the applier mentions {word!r}"
    # It is a LOCAL tool: no transport, and no way to reach the box at all.
    for word in ("ssh", "scp", "asyncssh", "BOX_SSH"):
        assert word not in text.replace("SSH", "").replace("the box transport", ""), \
            f"the applier reaches for {word!r}; the transport lives in llm-api"
    # An option the doctrine forbids must not be a token the parser accepts.
    assert '"--force"' not in text and "--fuzz" not in text.split("EOF")[0]


# ===========================================================================
# 2. The applier's behaviour, driven on a synthetic series.
# ===========================================================================
def test_applying_the_series_to_a_synthetic_base_reproduces_the_expected_content(synth):
    res = run_applier(synth["series"], synth["checkout"], "apply")
    out = combined(res)
    assert res.returncode == 0, out
    assert len(verdict_lines(out, "APPLIED")) == 3, out
    assert "file_sha_post" in out
    assert (synth["checkout"] / "pkg" / "mod.py").read_text() == MOD_TWO
    assert (synth["checkout"] / "pkg" / "other.py").read_text() == OTHER_ONE
    # only the series' own two files changed, and nothing else was written
    status = git(synth["checkout"], "status", "--porcelain").splitlines()
    assert sorted(status) == [" M pkg/mod.py", " M pkg/other.py"], status
    # and the applier printed a tree hash for the result, plus HEAD
    assert "HEAD " in out and "tree now " in out


def test_a_base_that_has_moved_is_refused_naming_both_hashes(synth):
    target = synth["checkout"] / "pkg" / "mod.py"
    target.write_text(MOD_BASE + "\n# an operator edit no tracked patch made\n")
    moved = sha_file(target)
    before = {p: sha_file(p) for p in patched_paths(synth["checkout"])}

    res = run_applier(synth["series"], synth["checkout"], "apply")
    out = combined(res)
    expected_pre = read_manifest(
        synth["series"] / "0001-first" / "patch.manifest")["file_sha_pre"]
    assert res.returncode == 2, out
    assert "REFUSED" in out
    assert moved in out, "the refusal does not print the hash it found"
    assert expected_pre in out, "the refusal does not print the hash it expected"
    assert "no fuzz" in out and "no --force" in out
    assert {p: sha_file(p) for p in patched_paths(synth["checkout"])} == before, \
        "a refusal still wrote something: the gate is the refusal, not the rollback"


def test_applying_twice_is_detected_and_writes_nothing(synth):
    first = run_applier(synth["series"], synth["checkout"], "apply")
    assert first.returncode == 0, combined(first)
    hashes = {p: sha_file(p) for p in patched_paths(synth["checkout"])}
    stamps = {p: p.stat().st_mtime_ns for p in patched_paths(synth["checkout"])}

    second = run_applier(synth["series"], synth["checkout"], "apply")
    out = combined(second)
    assert second.returncode == 0, out
    assert len(verdict_lines(out, "ALREADY")) == 3, out
    assert "nothing was written" in out
    assert verdict_lines(out, "APPLIED") == [], \
        "the second run applied instead of detecting"
    assert {p: sha_file(p) for p in patched_paths(synth["checkout"])} == hashes
    assert {p: p.stat().st_mtime_ns
            for p in patched_paths(synth["checkout"])} == stamps, \
        "a re-apply rewrote files it had just reported as ALREADY"

    # And the state a bisect actually leaves: patch 0001's post-image on mod.py,
    # other.py back at its base. The applier must call 0001 ALREADY and apply
    # only the rest - "already applied" is a per-file, per-chain fact, not a
    # property of the whole run.
    run_applier(synth["series"], synth["checkout"], "revert")
    (synth["checkout"] / "pkg" / "mod.py").write_text(MOD_ONE)
    half = run_applier(synth["series"], synth["checkout"], "apply")
    out = combined(half)
    assert half.returncode == 0, out
    assert "ALREADY  0001-first" in out, out
    assert "APPLIED  0002-second" in out and "APPLIED  0003-other" in out, out
    assert len(verdict_lines(out, "APPLIED")) == 2, out


def test_dry_run_reports_without_writing(synth):
    res = run_applier(synth["series"], synth["checkout"], "apply", "--dry-run")
    out = combined(res)
    assert res.returncode == 1, out      # 1 = would-change, not an error
    assert len(verdict_lines(out, "PENDING")) == 3, out
    assert "DRY RUN" in out
    assert git(synth["checkout"], "status", "--porcelain") == "", \
        "a dry run wrote to the checkout"
    assert (synth["checkout"] / "pkg" / "mod.py").read_text() == MOD_BASE


def test_revert_takes_the_whole_series_off_and_leaves_a_clean_checkout(synth):
    assert run_applier(synth["series"], synth["checkout"], "apply").returncode == 0
    res = run_applier(synth["series"], synth["checkout"], "revert",
                      "--expect-tree", synth["base_tree"])
    out = combined(res)
    assert res.returncode == 0, out
    assert len(verdict_lines(out, "REVERTED")) == 3, out
    assert "EXPECTED   PASS" in out
    assert (synth["checkout"] / "pkg" / "mod.py").read_text() == MOD_BASE
    assert (synth["checkout"] / "pkg" / "other.py").read_text() == OTHER_BASE
    assert git(synth["checkout"], "status", "--porcelain") == "", \
        "revert left the checkout dirty: it did not restore the bytes it found"
    # revert is idempotent too
    again = run_applier(synth["series"], synth["checkout"], "revert")
    assert again.returncode == 0, combined(again)
    assert "nothing to take off" in combined(again)


def test_revert_dry_run_plans_the_whole_chain_and_writes_nothing(synth):
    """--dry-run must describe the SAME sequence a real run performs.

    A chain makes this a real bug and not a formality: nothing is written in a
    dry run, so a walk that compared each link against the file's actual bytes
    would take 0002 off and then refuse at 0001, whose pre-image only exists
    after 0002 is gone. The plan has to be virtual, or the preview is a refusal
    and the command it previews is unreachable.
    """
    assert run_applier(synth["series"], synth["checkout"], "apply").returncode == 0
    hashes = {p: sha_file(p) for p in patched_paths(synth["checkout"])}
    dirty = git(synth["checkout"], "status", "--porcelain")
    assert dirty, "the apply did not make the checkout dirty: nothing to revert"
    res = run_applier(synth["series"], synth["checkout"], "revert", "--dry-run")
    out = combined(res)
    assert res.returncode == 1, out          # would-change, not an error
    assert len(verdict_lines(out, "TAKE-OFF")) == 3, out
    assert "REFUSE" not in out, out
    assert {p: sha_file(p) for p in patched_paths(synth["checkout"])} == hashes
    assert git(synth["checkout"], "status", "--porcelain") == dirty
    # and the plan it printed is exactly what the real run then does
    real = combined(run_applier(synth["series"], synth["checkout"], "revert"))
    assert len(verdict_lines(real, "REVERTED")) == 3, real
    assert git(synth["checkout"], "status", "--porcelain") == ""


def test_revert_undoes_the_chain_from_the_top_not_from_the_bottom(synth):
    """0001's post-image IS 0002's pre-image: reverting bottom-up is impossible.

    The verdicts must therefore come out descending for one file, which is the
    visible sign the tool is walking the chain the only direction it closes.
    """
    assert run_applier(synth["series"], synth["checkout"], "apply").returncode == 0
    out = combined(run_applier(synth["series"], synth["checkout"], "revert"))
    names = [line.split()[1] for line in verdict_lines(out, "REVERTED")]
    assert names == ["0002-second", "0001-first", "0003-other"], names


def test_the_operators_index_is_never_touched(synth):
    """The resulting-tree hash must be computed in a SCRATCH index.

    `git write-tree` on the caller's own index would silently turn staged,
    half-edited work into the tree the tool reports - so the report has to come
    from GIT_INDEX_FILE, and the staged set has to look identical afterwards.
    """
    # A staged change of ours, in a file the series does not touch.
    (synth["checkout"] / "README").write_text("staged work in progress\n")
    git(synth["checkout"], "add", "README")
    staged_before = git(synth["checkout"], "diff", "--cached", "--name-only")
    staged_blob = git(synth["checkout"], "show", ":README")
    assert staged_before == "README\n"
    assert run_applier(synth["series"], synth["checkout"], "apply").returncode == 0
    assert git(synth["checkout"], "diff", "--cached", "--name-only") == staged_before
    assert git(synth["checkout"], "show", ":README") == staged_blob
    assert run_applier(synth["series"], synth["checkout"], "revert").returncode == 0
    assert git(synth["checkout"], "diff", "--cached", "--name-only") == staged_before
    assert git(synth["checkout"], "show", ":README") == staged_blob
    assert "git add" not in APPLIER.read_text(), \
        "the applier stages things: that is how it would move your index"


def test_a_tampered_patch_file_is_refused_naming_both_hashes(synth):
    pf = synth["series"] / "0001-first" / "mod.py.patch"
    pf.write_text(pf.read_text() + "\n")           # one byte of drift
    before = (synth["checkout"] / "pkg" / "mod.py").read_text()

    res = run_applier(synth["series"], synth["checkout"], "apply")
    out = combined(res)
    assert res.returncode == 2, out
    assert "REFUSED" in out and "not the patch this manifest describes" in out
    assert "on disk   sha256=" in out and "manifest  sha256=" in out
    assert "A stale fingerprint is a refusal" in out
    assert (synth["checkout"] / "pkg" / "mod.py").read_text() == before


def test_an_unnumbered_or_unlisted_patch_directory_is_refused(synth):
    # a directory the series does not name is a patch nobody applies
    (synth["series"] / "9999-sneaky").mkdir()
    shutil.copytree(synth["series"] / "0001-first",
                    synth["series"] / "0004-sneaky")
    res = run_applier(synth["series"], synth["checkout"], "status")
    out = combined(res)
    assert res.returncode == 2, out
    assert "does not list" in out and "0004-sneaky" in out

    # and the numbering must be 0001..NN in order, with no gaps
    shutil.rmtree(synth["series"] / "9999-sneaky")
    shutil.rmtree(synth["series"] / "0004-sneaky")
    (synth["series"] / "series").write_text(
        "0002-second\n0001-first\n0003-other\n")
    res = run_applier(synth["series"], synth["checkout"], "status")
    out = combined(res)
    assert res.returncode == 2, out
    assert "series order is broken" in out and "does not start with 0001-" in out


def test_a_missing_target_is_a_refusal_not_a_creation(synth):
    (synth["checkout"] / "pkg" / "mod.py").unlink()
    res = run_applier(synth["series"], synth["checkout"], "apply")
    out = combined(res)
    assert res.returncode == 2, out
    assert "ABSENT" in out and "refusing to guess, and refusing to create it" in out
    assert not (synth["checkout"] / "pkg" / "mod.py").exists()


def test_a_manifest_that_disagrees_with_its_patch_installs_nothing(synth):
    """The built image must hash to file_sha_post BEFORE anything is written.

    Simulated honestly: a manifest re-pinned to a hash the patch cannot produce.
    The first patch then still lands (it is fine), and the second must abort the
    run and undo what the first wrote - the series is all-or-nothing per run.
    """
    m2 = synth["series"] / "0002-second" / "patch.manifest"
    manifest = read_manifest(m2)
    text = m2.read_text()
    text = text.replace("file_sha_post=" + manifest["file_sha_post"],
                        "file_sha_post=" + "f" * 64)
    m2.write_text(text)

    res = run_applier(synth["series"], synth["checkout"], "apply")
    out = combined(res)
    assert res.returncode == 2, out
    assert "REFUSED 0002-second" in out
    assert "built       sha256=" in out and "manifest    sha256=" in out
    assert "REFUSED and UNDONE" in out
    assert (synth["checkout"] / "pkg" / "mod.py").read_text() == MOD_BASE, \
        "the run aborted but left patch 0001 installed: a refusal is not a rollback"
    assert (synth["checkout"] / "pkg" / "other.py").read_text() == OTHER_BASE, \
        "patch 0003 landed after the refusal: the walk did not stop where it said"


# ===========================================================================
# 3. The acceptance check that matters: the real series on the real base.
# ===========================================================================
def reference_repo() -> Path:
    base = read_manifest(BASE_MANIFEST)
    path = Path(base.get("reference_clone", "") or ".")
    if not (path / ".git").exists():
        pytest.skip(f"{path} is not a git clone - the acceptance check needs the "
                    "live-tree reconstruction described in patches/base.manifest "
                    "(base_branch + live_branch); refusing to simulate it")
    for ref in (base.get("reference_base_branch", ""),
                base.get("reference_live_branch", "")):
        r = subprocess.run(  # noqa: S603 - a fixed git binary on a fixed argument list
            [GIT, "rev-parse", "--verify", "--quiet", f"refs/heads/{ref}"],
            cwd=str(path), capture_output=True, text=True, check=False)
        if r.returncode != 0:
            pytest.skip(f"{path} has no branch '{ref}' - the acceptance check "
                        "cannot run against a reconstruction that is not the one "
                        "base.manifest describes")
    return path


@pytest.fixture(scope="module")
def live_tree_from_reference() -> str:
    """`live^{tree}`, computed by git, not copied from base.manifest."""
    path = reference_repo()
    base = read_manifest(BASE_MANIFEST)
    ref = base.get("reference_live_branch")
    assert ref, "base.manifest does not name reference_live_branch"
    return git(path, "rev-parse", f"{ref}^{{tree}}").strip()


def test_the_reference_reconstruction_is_what_base_manifest_claims():
    """`built` is the recorded base tree, and `live` is `built` + these hunks.

    If this is false, the acceptance test below would be comparing against a
    tree that is not the box's, so it is checked separately rather than assumed.
    """
    path = reference_repo()
    base = read_manifest(BASE_MANIFEST)
    assert git(path, "rev-parse",
               f"{base['reference_base_branch']}^{{tree}}").strip() \
        == base["base_tree"]
    assert git(path, "rev-parse",
               f"{base['reference_live_branch']}^{{tree}}").strip() \
        == base["live_tree"]
    diff = git(path, "diff", "--stat", base["reference_base_branch"],
               base["reference_live_branch"])
    touched = [line.split(" |")[0].strip()
               for line in diff.splitlines() if " |" in line]
    real_files = {read_manifest(PATCHES / e / "patch.manifest")["git_rel"]
                  for e in series_entries(SERIES_FILE)}
    assert set(touched) == real_files, \
        f"the box's two trees differ in more files than the series touches: {touched}"


def test_applying_the_series_to_built_reproduces_the_live_branch_byte_for_byte(
        tmp_path, live_tree_from_reference):
    """THE check: `built` + this series == `live`, and nothing else changed."""
    path = reference_repo()
    base = read_manifest(BASE_MANIFEST)
    clone = tmp_path / "rebuild"
    r = subprocess.run(  # noqa: S603 - a local clone of a local path, no network
        [GIT, "clone", "--quiet", "--no-checkout", str(path), str(clone)],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    # The remote ref, not the bare name: `git checkout --detach built`
    # DWIMs a local branch creation and refuses "--detach cannot be used
    # with -b", which is a git parsing artefact, not a finding.
    git(clone, "checkout", "--quiet", "--detach",
        "refs/remotes/origin/" + base["reference_base_branch"])
    assert git(clone, "rev-parse", "HEAD^{tree}").strip() == base["base_tree"]

    res = run_applier(PATCHES, clone, "apply", "--expect-tree",
                      live_tree_from_reference)
    out = combined(res)
    assert res.returncode == 0, out
    assert len(verdict_lines(out, "APPLIED")) == 4, out
    assert "EXPECTED   PASS" in out, out
    assert f"tree now   {live_tree_from_reference}" in out

    # Byte-for-byte, per file, against the blobs git holds for `live` - not
    # against a hash copied into a manifest.
    for entry in series_entries(SERIES_FILE):
        rel = read_manifest(PATCHES / entry / "patch.manifest")["git_rel"]
        want = hashlib.sha256(git(path, "show",
                                 f"{base['reference_live_branch']}:{rel}"
                                 ).encode()).hexdigest()
        got = sha_file(clone / rel)
        assert got == want, f"{rel}: applied result {got} != live blob {want}"

    # "nothing else changed": the working copy vs the live branch is empty.
    remote_live = subprocess.run(  # noqa: S603 - fixed binary, fixed arguments
        [GIT, "rev-parse", "--verify", "--quiet", "refs/remotes/origin/"
         + base["reference_live_branch"]],
        cwd=str(clone), capture_output=True, text=True, check=False)
    assert remote_live.returncode == 0, "the clone did not bring the live branch"
    live_ref = f"origin/{base['reference_live_branch']}"
    assert git(clone, "diff", live_ref, "--", ".") == ""
    # ...and only the series' two files are modified relative to `built`.
    assert sorted(git(clone, "status", "--porcelain").splitlines()) == [
        " M vllm/v1/simple_kv_offload/manager.py",
        " M vllm/v1/simple_kv_offload/metadata.py",
    ]

    # Reverting the real series returns the real base tree.
    back = run_applier(PATCHES, clone, "revert", "--expect-tree", base["base_tree"])
    out = combined(back)
    assert back.returncode == 0, out
    assert "EXPECTED   PASS" in out
    assert git(clone, "status", "--porcelain") == "", \
        "revert of the real series left the checkout dirty"


def test_the_real_series_refuses_a_base_that_is_not_its_base(tmp_path):
    """The same base, moved: one added line, and the real series refuses."""
    path = reference_repo()
    base = read_manifest(BASE_MANIFEST)
    clone = tmp_path / "moved"
    r = subprocess.run(  # noqa: S603 - a local clone of a local path, no network
        [GIT, "clone", "--quiet", "--no-checkout", str(path), str(clone)],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    # The remote ref, not the bare name: `git checkout --detach built`
    # DWIMs a local branch creation and refuses "--detach cannot be used
    # with -b", which is a git parsing artefact, not a finding.
    git(clone, "checkout", "--quiet", "--detach",
        "refs/remotes/origin/" + base["reference_base_branch"])
    target = clone / read_manifest(
        PATCHES / series_entries(SERIES_FILE)[0] / "patch.manifest")["git_rel"]
    target.write_text(target.read_text() + "\n# the fork moved\n")

    res = run_applier(PATCHES, clone, "apply")
    out = combined(res)
    assert res.returncode == 2, out
    assert "REFUSED 0001-kvtier-pr56810" in out
    first = read_manifest(PATCHES / series_entries(SERIES_FILE)[0] / "patch.manifest")
    assert first["file_sha_pre"] in out, "the refusal omits the hash it wanted"
    assert sha_file(target) in out, "the refusal omits the hash it found"
    # and it tells you where to look instead of offering a flag
    assert "git -C" in out and "diff -- " in out
    assert "no --force" in out
    # Nothing at all was written: the moved file is exactly as left by the test.
    assert target.read_text().endswith("# the fork moved\n")
    assert git(clone, "status", "--porcelain").splitlines() == [
        " M vllm/v1/simple_kv_offload/manager.py",
    ]


def test_a_head_that_is_not_the_recorded_base_is_announced(synth):
    """The base-tree gate is about HEAD; the fingerprint gate is about bytes.

    A checkout on a DIFFERENT tree still applies cleanly if the hunks fit, and
    that is worth printing loudly rather than silently allowing - "these hunks
    landed" and "this is the tree they were made against" are two claims.
    """
    (synth["checkout"] / "pkg" / "mod.py").write_text(MOD_BASE)
    (synth["checkout"] / "extra.py").write_text("an 59th patch of our own\n")
    git(synth["checkout"], "add", "-A")
    git(synth["checkout"], "commit", "-q", "-m", "the base moved on")
    out = combined(run_applier(synth["series"], synth["checkout"], "status"))
    assert "WARNING   the checkout's tree is NOT the recorded base tree" in out
    assert synth["base_tree"] in out
    # the series still applies: the per-file fingerprints are the real gate
    assert run_applier(synth["series"], synth["checkout"], "apply").returncode == 0


def test_reverting_the_real_series_from_live_reproduces_the_base_tree(tmp_path):
    """`live` - these four hunks == `built`, byte for byte and tree for tree.

    The apply direction alone would not catch a patch that "applies" while its
    reverse-apply produces something else: `git apply -R` is checked against
    file_sha_pre here, and the result against the blobs git holds for `built`.
    """
    path = reference_repo()
    base = read_manifest(BASE_MANIFEST)
    clone = tmp_path / "unlive"
    r = subprocess.run(  # noqa: S603 - a local clone of a local path, no network
        [GIT, "clone", "--quiet", "--no-checkout", str(path), str(clone)],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    git(clone, "checkout", "--quiet", "--detach",
        "refs/remotes/origin/" + base["reference_live_branch"])
    assert git(clone, "rev-parse", "HEAD^{tree}").strip() == base["live_tree"]

    res = run_applier(PATCHES, clone, "revert", "--expect-tree", base["base_tree"])
    out = combined(res)
    assert res.returncode == 0, out
    assert len(verdict_lines(out, "REVERTED")) == 4, out
    assert "EXPECTED   PASS" in out, out
    for entry in series_entries(SERIES_FILE):
        rel = read_manifest(PATCHES / entry / "patch.manifest")["git_rel"]
        want = hashlib.sha256(git(path, "show",
                                 f"{base['reference_base_branch']}:{rel}").encode()
                              ).hexdigest()
        assert sha_file(clone / rel) == want, f"{rel} did not revert to its pre-image"
    base_ref = f"origin/{base['reference_base_branch']}"
    assert git(clone, "diff", base_ref, "--", ".") == ""
