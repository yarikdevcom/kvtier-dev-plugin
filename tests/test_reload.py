"""Hot reload: the holder's late binding, the `.pyc` trap, and path refusal.

The constraint this suite exists for (verified on the engine's 3.13.5):
`importlib.reload(M)` reuses the module object and updates `M.__dict__` in
place, so anything that resolves a module-global name at call time sees the new
function, while `from policy import fn` keeps the old one forever. Hence the
holder: `_POLICY.decide(...)` looks the function up on every call, and the
reload verb can only ever load one specific file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from kvtier_dev_plugin import config, snapshot
from kvtier_dev_plugin.policy_holder import PolicyHolder

V1 = """\"\"\"policy v1\"\"\"
VERSION = 1


def decide(view):
    return {"v": 1, "note": "first"}
"""

# Derived from V1 so the byte length is IDENTICAL by construction (only the two
# value bytes differ), which is exactly the edit a (mtime, size) bytecode cache
# validation cannot distinguish.
V2_SAME_SIZE = (V1.replace("VERSION = 1", "VERSION = 2")
                .replace('"v": 1', '"v": 2')
                .replace('"first"', '"next!"'))

V2_LONGER = """\"\"\"policy v2\"\"\"
VERSION = 2


def decide(view):
    return {"v": 2, "note": "a clearly longer second version of this file"}
"""

BROKEN = "VERSION = 3\ndef broken(:\n    pass\n"
NO_DECIDE = "VERSION = 3\nhelper = 1\n"


def write(path: Path, text: str, *, keep_mtime_ns: tuple[int, int] | None = None
          ) -> None:
    """Write `text`, optionally restoring a (atime_ns, mtime_ns) pair captured
    BEFORE the write, so the file's identity to a bytecode cache is unchanged."""
    path.write_text(text, encoding="utf-8")
    if keep_mtime_ns is not None:
        os.utime(path, ns=keep_mtime_ns)


@pytest.fixture
def policy_dir(tmp_path):
    directory = tmp_path / "pkg"
    directory.mkdir()
    return directory


# --- late binding -----------------------------------------------------------

def test_holder_exposes_the_live_decide(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    assert holder.decide is None
    assert holder.ensure_loaded()["ok"] is True
    assert holder.decide({}) == {"v": 1, "note": "first"}


def test_ensure_loaded_loads_once(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    assert holder.ensure_loaded()["loaded_from_cold"] is True
    assert holder.ensure_loaded()["loaded_from_cold"] is False
    assert holder.describe()["loads"] == 1


def test_reload_changes_what_the_holder_runs(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    holder.ensure_loaded()
    before = holder.describe()["live"]["source_sha256_8"]

    write(policy, V2_LONGER)
    report = holder.reload()
    assert report["ok"] is True and report["rolled_back"] is False
    assert report["kept"]["source_sha256_8"] == before
    assert report["applied"]["source_sha256_8"] != before
    assert holder.decide({}) == {"v": 2, "note": "a clearly longer second "
                                                 "version of this file"}


def test_a_frozen_binding_is_stale_and_the_holder_is_not(policy_dir):
    """Exactly the trap the design is built around.

    `direct = holder.decide` is the same shape as `from policy import decide`:
    a reference to the function object, immune to every later reload. Nothing
    in this package may hold one, and this test is why.
    """
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    holder.ensure_loaded()
    direct = holder.decide

    write(policy, V2_LONGER)
    assert holder.reload()["ok"] is True
    assert direct({}) == {"v": 1, "note": "first"}          # frozen: still v1
    assert holder.decide({})["v"] == 2                      # holder: live


def test_snapshot_reports_the_new_policy_after_a_reload(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    engine = FakeLightEngine()

    first = snapshot.snapshot_for_engine(engine, holder)
    assert first["policy"]["result"]["v"] == 1
    write(policy, V2_LONGER)
    assert snapshot.reload_policy_on_engine(holder)["ok"] is True
    second = snapshot.snapshot_for_engine(engine, holder)
    assert second["policy"]["result"]["v"] == 2
    assert second["epoch"] == first["epoch"]                # same boot, new policy


class FakeLightEngine:
    """The minimum engine shape: no scheduler at all."""


# --- the bytecode-cache trap ------------------------------------------------

def test_reload_picks_up_an_edit_with_the_same_size_and_mtime(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    stamp = os.stat(policy)
    identity = (stamp.st_atime_ns, stamp.st_mtime_ns)
    assert V2_SAME_SIZE != V1
    assert len(V2_SAME_SIZE.encode()) == len(V1.encode())   # the trap is armed

    holder = PolicyHolder(str(policy))
    holder.ensure_loaded()
    write(policy, V2_SAME_SIZE, keep_mtime_ns=identity)
    assert os.stat(policy).st_mtime_ns == identity[1]
    assert os.stat(policy).st_size == len(V1.encode())

    assert holder.reload()["ok"] is True
    assert holder.decide({}) == {"v": 2, "note": "next!"}, \
        "the reload must not be fooled by an identical (size, mtime)"


def test_importlib_would_have_been_fooled_by_the_same_edit(policy_dir):
    """Documents the trap rather than asserting our own behaviour.

    Verified on CPython 3.13: a same-size, same-mtime edit is served from
    `__pycache__`, so `importlib.reload()` keeps the OLD function and a reload
    looks like a no-op. Our holder compiles the bytes it just read, so it
    cannot be fooled this way.
    """
    # importlib.reload() re-resolves the spec by NAME, so the module has to be
    # importable: give it a name and a sys.path entry, and take them back.
    name = "kvtier_trap_policy"
    policy = policy_dir / f"{name}.py"
    sys.path.insert(0, str(policy_dir))
    write(policy, V1)
    spec = importlib.util.spec_from_file_location(name, str(policy))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        assert module.decide({})["v"] == 1
        assert list(policy_dir.rglob("__pycache__")), "the first load must have " \
                                                      "written a bytecode cache"

        stamp = os.stat(policy)
        identity = (stamp.st_atime_ns, stamp.st_mtime_ns)
        write(policy, V2_SAME_SIZE, keep_mtime_ns=identity)

        reloaded = importlib.reload(module)
        assert reloaded.decide({})["v"] == 1, (
            "importlib served the cached bytecode for this edit, which is the "
            "trap the holder does not have")
    finally:
        del sys.modules[name]
        sys.path.remove(str(policy_dir))

    # The same edit, through the holder: picked up.
    holder = PolicyHolder(str(policy))
    holder.ensure_loaded()
    assert holder.reload()["ok"] is True
    assert holder.decide({}) == {"v": 2, "note": "next!"}


def test_holder_writes_no_bytecode_anywhere(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    PolicyHolder(str(policy)).ensure_loaded()
    assert list(policy_dir.rglob("__pycache__")) == []
    assert list(policy_dir.rglob("*.pyc")) == []


# --- rollback ---------------------------------------------------------------

@pytest.mark.parametrize("bad_source,expect", [
    (BROKEN, "SyntaxError"),
    ("raise RuntimeError('module-level boom')\n", "RuntimeError"),
    (NO_DECIDE, "no callable decide"),
    ("VERSION = [1, 2]\n\n\ndef decide(view):\n    return {}\n", "VERSION must be"),
])
def test_a_bad_policy_keeps_the_old_one_and_comes_back_as_data(policy_dir,
                                                               bad_source,
                                                               expect):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    holder.ensure_loaded()

    write(policy, bad_source)
    report = holder.reload()
    assert report["ok"] is False
    assert report["rolled_back"] is True
    assert report["applied"] is None
    assert expect in (report["traceback"] or "") + (report["error"] or "")
    assert holder.decide({}) == {"v": 1, "note": "first"}   # still serving v1
    assert holder.describe()["failures"] == 1

    # ...and a later good edit still lands.
    write(policy, V2_LONGER)
    assert holder.reload()["ok"] is True
    assert holder.decide({})["v"] == 2


def test_reload_report_is_json_serialisable(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    holder.ensure_loaded()
    write(policy, BROKEN)
    json.dumps(holder.reload())          # it crosses a msgpack boundary


def test_traceback_is_capped(policy_dir):
    policy = policy_dir / "policy.py"
    write(policy, V1)
    holder = PolicyHolder(str(policy))
    holder.ensure_loaded()
    noise = "\n".join(f"# line {i} of filler padding padding" for i in range(400))
    write(policy, "raise ValueError('x')\n" + noise)
    report = holder.reload()
    assert len(report["traceback"]) <= 4200


# --- path validation --------------------------------------------------------

def test_the_canonical_policy_path_is_the_package_file():
    target = config.validate_policy_target(config.canonical_policy_path())
    assert target == config.canonical_policy_path()
    assert target.name == "policy.py"


@pytest.mark.parametrize("rejected", [
    "/etc/passwd",
    "/",
    "",
    "   ",
    "policy2.py",
    "policy.py.evil",
    "../outside/policy.py",
    "subdir/policy.py",
    "os.path",                                   # a module name is not a path
    "kvtier_dev_plugin/policy.py",               # relative to cwd, not the pkg
    __file__,                                    # a test file
    "/tmp/policy.py",
    "a\x00b/policy.py",
    None,
    17,
    ["policy.py"],
])
def test_the_validator_refuses_everything_else(rejected):
    with pytest.raises(config.PolicyPathError):
        config.validate_policy_target(rejected)


def test_a_path_that_spells_the_canonical_file_is_accepted():
    canonical = config.canonical_policy_path()
    spelled = str(canonical.parent / "." / "policy.py")
    assert config.validate_policy_target(spelled) == canonical
    round_about = (canonical.parent.parent / canonical.parent.name
                   / canonical.name)
    assert config.validate_policy_target(round_about) == canonical


def test_a_missing_canonical_file_is_reported_not_crashed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_PKG_DIR", tmp_path)
    with pytest.raises(config.PolicyPathError, match="not a file"):
        config.validate_policy_target(tmp_path / "policy.py")


def test_a_symlink_out_of_the_package_is_refused(monkeypatch, tmp_path,
                                                 policy_dir):
    real_dir = tmp_path / "install" / "kvtier_dev_plugin"
    real_dir.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    evil = elsewhere / "payload.py"
    evil.write_text("VERSION = 9\n\n\ndef decide(view):\n    return {'v': 9}\n")
    link = real_dir / "policy.py"
    link.symlink_to(evil)

    monkeypatch.setattr(config, "_PKG_DIR", real_dir.resolve())
    with pytest.raises(config.PolicyPathError, match="refused"):
        config.validate_policy_target(link)


def test_the_shipped_policy_is_loadable_and_cheap():
    """The file that ships with the package must actually work."""
    holder = PolicyHolder(str(config.canonical_policy_path()))
    assert holder.ensure_loaded()["ok"] is True
    result = holder.decide({"tier": {"present": False}})
    assert result["verdict"] == "no-tier"
    result = holder.decide({"tier": {"present": True, "num_cpu_blocks": 100,
                                     "cpu_free_blocks": 5, "counts": {},
                                     "boundary_store_stats": {}}})
    assert result["verdict"] == "cpu-pool-nearly-full"
    assert isinstance(snapshot.json_safe({"policy": result}), dict)
