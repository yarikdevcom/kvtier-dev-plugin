r"""The journal directory is MADE, and a journal that cannot be written is LOUD.

WHY THIS FILE EXISTS. `VLLM_E2_TIER_DISK_JOURNAL_DIR` (0.2.19) lets the spill
journal live on storage that survives a reboot. Nobody made the directory. So the
variable that exists to stop the 2026-10-05 failure - a reboot that ate the journal
while 107 GB of KV files survived, leaving an index at `entries: 0` - could produce
the same failure self-inflicted: point it at `/home/user/kv-tier/journal` before
that path exists and every journal append fails, while `journal_append` (never
raising, by design) turns it into one anonymous counter. The spill commits the KV
file, the journal records nothing, and the tier fills the store with bytes no index
can name - strictly worse than the tmpfs story, because a reboot at least explains
itself.

The two decisions these tests hold still:

1. MAKING THE DIRECTORY IS PART OF READING THE PATH, in exactly one place
   (`disk_index.journal_path()` -> `_ensure_journal_directory()`), cached per
   resolved directory so an append stays one `write()`, and it never raises. The
   READ side (the boot line, the verbs, the snapshot) creates nothing: it asks via
   `journal_directory_state()`, whose answer distinguishes "we tried and failed"
   (`unwritable`) from "we cannot tell" (`unknown`).
2. KEEP SPILLING, MAKE THE SILENCE IMPOSSIBLE. A journal failure costs disk space;
   refusing the spill would cost the whole tier for a logging problem. So the spill
   lands and `journal_write_errors` / `journal_last_error` /
   `journal_directory_writable` / `journal_directory_error` / the boot line's
   `journal_dir=unwritable` all say so, in the payloads an operator actually reads.

Stdlib plus the plugin; `tmp_path` for every file; no engine, no threads. The two
tests that need a directory this process cannot write are skipped for root, for
whom no mode is unwritable.
"""

from __future__ import annotations

import json
import os
import shutil
import stat

import pytest

from kvtier_dev_plugin import (
    backport,
    config,
    disk_index,
    disk_layout,
    disk_restore,
    disk_store,
)

KEY16 = "a3f0b7c1d4e5f607"
GROUP = 3
KEY = KEY16 + "1" * 48 + f"{GROUP:08x}"
RANK = "PP0_TP0"
GEO = "abcdef123456"
JOURNAL_NAME = f"disk-spill-journal-{RANK}.jsonl"

needs_unwritable = pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root is not refused by a 0o555 directory: nothing here is unwritable")


def current_umask() -> int:
    """The process umask, read by setting it back to itself.

    `os.makedirs(mode=...)` is filtered by the umask, so a test that hard-coded
    0o750 would fail on a host that runs pytest with a private-group umask - and
    fail by accusing the plugin of a bug it does not have.
    """
    previous = os.umask(0)
    os.umask(previous)
    return previous


def expected_mode() -> int:
    return disk_index.JOURNAL_DIR_MODE & ~current_umask()


def mode_of(path) -> int:
    return stat.S_IMODE(os.stat(str(path)).st_mode)


def write_slot(root, key=KEY, geo=GEO, size=4096):
    """A committed slot file, by the same door the spill thread commits with."""
    made = disk_layout.build_slot_path(root, key, geo)
    assert made["ok"], made["reason"]
    written = disk_layout.write_committed_file(made["path"], b"\0" * size,
                                               sync=False)
    assert written["ok"], written["reason"]
    return made


@pytest.fixture(autouse=True)
def channel(tmp_path, monkeypatch):
    """A diagnostics directory and a rank, and a fresh index for every test.

    The journal directory itself is deliberately NOT created by this fixture: the
    point of the first tests is that the plugin creates it.
    """
    diag = tmp_path / "diag"
    diag.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, RANK)
    monkeypatch.delenv(config.ENV_DISK_JOURNAL_DIR, raising=False)
    disk_index.reset_state_for_tests()
    yield
    disk_index.reset_state_for_tests()


def use_journal(tmp_path, monkeypatch, name="a/b/journal"):
    """Point the journal at a NESTED path under `tmp_path` that does not exist."""
    journal = tmp_path
    for part in name.split("/"):
        journal = journal / part
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(journal))
    assert not os.path.exists(str(journal))
    return journal


# --- 1. the directory comes into existence, once, at the one read point ------

def test_a_nested_journal_path_makes_itself_and_journals_from_the_first_line(
        tmp_path, monkeypatch):
    """The self-inflicted 2026-10-05 failure, fixed at the point it is created.

    Three missing levels, one spill, one line: the directory has to appear, the
    line has to be IN it, and the reader that the scheduler uses has to find the
    same file - which is only true because the writer and the reader resolve the
    path at the same one place.
    """
    journal = use_journal(tmp_path, monkeypatch)
    made = write_slot(tmp_path / "store")
    report = disk_index.record_spill(made["path"], 4096)
    assert report["ok"] is True, report["reason"]
    assert os.path.isdir(str(journal)), "the journal directory was never created"
    written = os.path.join(str(journal), JOURNAL_NAME)
    assert os.path.exists(written)
    with open(written, "r", encoding="utf-8") as handle:
        assert len(handle.read().splitlines()) == 1
    assert mode_of(journal) == expected_mode()
    # The reader resolves the SAME path: an index that came up empty over a
    # directory it cannot find is the bug this whole module is above.
    refreshed = disk_index.refresh(force=True)
    assert refreshed["applied"] is True, refreshed
    assert disk_index.lookup(KEY)["present"] is True


def test_the_created_directory_uses_the_stores_own_mode(tmp_path, monkeypatch):
    """0o750, because that is what `disk_layout` gives the store's shard dirs.

    The task said 0o775; the codebase says 0o750 (`write_committed_file`,
    disk_layout.py). Matching the existing convention is the requirement, so this
    test compares the two directories the plugin creates instead of trusting a
    number - a future change to one of them fails here rather than silently
    giving a box two conventions.
    """
    assert disk_index.JOURNAL_DIR_MODE == 0o750
    store = tmp_path / "store"
    made = write_slot(store)
    shard_mode = mode_of(os.path.dirname(made["path"]))
    journal = use_journal(tmp_path, monkeypatch)
    assert disk_index.journal_append("+", KEY16, 10, GROUP)["ok"] is True
    assert mode_of(journal) == shard_mode


def test_the_directory_is_made_once_and_not_once_per_line(tmp_path, monkeypatch):
    """Five lines, one `makedirs`. The append path owes one `write()`.

    A `makedirs(exist_ok=True)` per append would be harmless in the kernel and
    fatal to the intent: this runs on `disk_spill`'s writer thread, whose whole
    design is that a journal line is one cheap write next to an 11 MB spill.
    """
    journal = use_journal(tmp_path, monkeypatch)
    calls: list[str] = []
    real_makedirs = os.makedirs

    def spy(path, *args, **kwargs):
        calls.append(str(path))
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(os, "makedirs", spy)
    for index in range(6):
        verdict = disk_index.journal_append("+", f"{index:016x}", 10, GROUP)
        assert verdict["ok"] is True, verdict["reason"]
    assert calls.count(str(journal)) == 1, calls
    assert disk_index.counters()["journal_directory_attempts"] == 1
    assert disk_index.counters()["journal_directory_errors"] == 0
    assert disk_index.counters()["journal_writes"] == 6
    assert disk_index.counters()["journal_write_errors"] == 0


def test_reading_the_journal_view_creates_no_directory_at_all(tmp_path,
                                                              monkeypatch):
    """The read side asks; it does not build. Creation belongs to the writer.

    A status verb or a boot line that mkdir'd the configured path would leave a
    directory behind on a box where nothing ever spills, in a location an operator
    never asked a daemon to write to, and would make "the tier is inert" untestable.
    """
    journal = use_journal(tmp_path, monkeypatch)
    for _ in range(3):
        assert disk_index.journal_directory() == str(journal)
        assert disk_index.journal_directory_state()["writable"] is True
        assert json.dumps(disk_index.journal_fs_state(), default=str)
        assert json.dumps(disk_index.section(), default=str)
    assert not os.path.exists(str(journal)), (
        "a read created a directory: creation must stay at the writer's single "
        "path-composition point")
    assert disk_index.counters()["journal_directory_attempts"] == 0


# --- 2. a directory that cannot be made is counted, named, and never fatal ---

def test_a_journal_path_that_cannot_be_created_is_counted_and_named(tmp_path,
                                                                     monkeypatch):
    """A file where a directory must go: no raise, one count, one named reason.

    Chosen over a read-only parent so it fails the same way for every user, root
    included: `makedirs` answers `NotADirectoryError` whichever uid asks.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(blocker / "journal"))
    # The spill's product is committed FIRST, and stays committed: this is the
    # same `write_committed_file` the spill thread commits with.
    made = write_slot(tmp_path / "store")
    report = disk_index.record_spill(made["path"], 4096)
    assert report["ok"] is False
    assert report["reason_kind"] == "journal_unwritable"
    assert os.path.exists(made["path"]), "the KV file must survive its index"
    counters = disk_index.counters()
    assert counters["journal_write_errors"] == 1
    assert counters["journal_directory_attempts"] == 1
    assert counters["journal_directory_errors"] == 1
    assert counters["journal_writes"] == 0
    assert "NotADirectoryError" in str(counters["journal_last_error"])
    section = disk_index.section()
    assert section["journal_write_errors"] == 1
    assert section["journal_directory_writable"] is False
    assert "NotADirectoryError" in str(section["journal_directory_error"])
    assert json.dumps(section, default=str)


def test_the_boot_line_answers_unwritable_and_never_borrows_unknown(tmp_path,
                                                                    monkeypatch):
    """`unwritable` and `unknown` are different 3 a.m. sentences.

    `unknown` means the mount table could not tell us: go and look. `unwritable`
    means the first journal line cannot be written: chmod it, or move the path.
    Folding one into the other is how a real failure becomes a shrug, so both
    directions are pinned here, with the boot line itself as the surface.
    """
    def boot_line() -> str:
        return backport.boot_line({"mode": "test", "backport": True,
                                   "guard": True, "tier": True})

    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(blocker / "journal"))
    assert disk_index.journal_append("+", KEY16, 10, GROUP)["ok"] is False
    assert disk_index.journal_fs_label() == "unwritable"
    assert disk_index.journal_fs_state()["writable"] is False
    assert disk_index.journal_fs_state()["writable_basis"] == "attempt"
    line = boot_line()
    assert "journal_dir=unwritable" in line, line
    assert "journal_dir=unknown" not in line, line

    # An unreadable mount table over a perfectly writable directory is `unknown`
    # and NOT `unwritable`: we did not try and fail, we could not tell.
    disk_index.reset_state_for_tests()
    monkeypatch.setattr(disk_layout, "PROC_MOUNTS_PATH",
                        str(tmp_path / "no-such-mounts"))
    elsewhere = tmp_path / "somewhere"
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(elsewhere))
    assert disk_index.journal_fs_label() == "unknown"
    assert disk_index.journal_directory_state()["writable"] is True
    assert "journal_dir=unknown" in boot_line()
    assert "journal_dir=unwritable" not in boot_line()

    # Nothing configured at all is a third thing, and also not `unwritable`.
    disk_index.reset_state_for_tests()
    monkeypatch.delenv(config.ENV_DISK_JOURNAL_DIR, raising=False)
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    assert disk_index.journal_fs_label() == "unset"
    assert disk_index.journal_directory_state()["writable"] is None


@needs_unwritable
def test_an_unwritable_journal_directory_is_refused_loudly_and_then_heals(
        tmp_path, monkeypatch):
    """A read-only parent: named before the first line, counted when it is tried,
    and believed again the moment the operator fixes it - no restart.

    The probe answers at boot (the boot line can therefore say `unwritable` before
    any spill has been attempted), the attempt answers once a line is tried, and a
    failed attempt that the probe contradicts yields to the probe, because a tier
    that stayed blind for a minute after a `chmod` would be its own incident.
    """
    blocked = tmp_path / "locked"
    blocked.mkdir()
    blocked.chmod(0o555)
    journal = blocked / "journal"
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(journal))
    try:
        state = disk_index.journal_directory_state()
        assert state["writable"] is False and state["basis"] == "probe"
        assert "cannot be written" in str(state["reason"])
        assert disk_index.journal_fs_label() == "unwritable"

        verdict = disk_index.journal_append("+", KEY16, 10, GROUP)
        assert verdict["ok"] is False
        assert verdict["reason_kind"] == "journal_unwritable"
        assert "PermissionError" in str(verdict["reason"])
        assert disk_index.counters()["journal_write_errors"] == 1
        assert disk_index.section()["journal_directory_writable"] is False

        # The operator fixes it. The next line works; nothing restarted, and the
        # view stops claiming `unwritable` on the same read it became true on.
        blocked.chmod(0o755)
        assert disk_index.journal_append("+", KEY16, 10, GROUP)["ok"] is True
        assert disk_index.counters()["journal_writes"] == 1
        assert disk_index.journal_fs_label() != "unwritable"
        assert disk_index.journal_directory_state()["writable"] is True
        assert disk_index.counters()["journal_write_errors"] == 1  # not re-counted
    finally:
        blocked.chmod(0o755)  # let pytest remove the tree


@needs_unwritable
def test_a_journal_file_that_cannot_be_written_is_a_counted_journal_error(
        tmp_path, monkeypatch):
    """The directory is fine; the FILE is not. Same rule: counted, named, quiet.

    The failure the directory cache cannot see is a write errno, and the retry that
    re-prepares the directory must not double-count it: one failed line is one
    error, whichever half of the pair stopped us.
    """
    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(journal))
    (journal / JOURNAL_NAME).write_bytes(b"1 + " + KEY16.encode() + b" 10 3\n")
    os.chmod(str(journal / JOURNAL_NAME), 0o444)
    verdict = disk_index.journal_append("+", KEY16, 10, GROUP)
    assert verdict["ok"] is False
    assert verdict["reason_kind"] == "journal_error"
    counters = disk_index.counters()
    assert counters["journal_write_errors"] == 1
    assert "PermissionError" in str(counters["journal_last_error"])
    os.chmod(str(journal / JOURNAL_NAME), 0o644)


def test_a_journal_directory_removed_mid_boot_costs_no_lines(tmp_path,
                                                             monkeypatch):
    """The cache can be wrong (an operator cleans a tmpfs); the line must not pay.

    A dropped journal line drops the FACT for good, so a failed write is retried
    once after the directory is re-prepared. Without that retry the first spill
    after the wipe would be invisible forever, silently.
    """
    journal = use_journal(tmp_path, monkeypatch, name="journal")
    assert disk_index.journal_append("+", KEY16, 10, GROUP)["ok"] is True
    shutil.rmtree(str(journal))
    assert not os.path.exists(str(journal))
    again = disk_index.journal_append("+", "b" + KEY16[1:], 10, GROUP)
    assert again["ok"] is True, again["reason"]
    assert os.path.isdir(str(journal)), "the directory was not re-created"
    assert disk_index.counters()["journal_write_errors"] == 0
    assert disk_index.counters()["journal_directory_attempts"] == 2


# --- 3. the invariants that make the above safe to rely on ------------------

def test_the_journal_directory_has_exactly_one_creation_site(tmp_path):
    """One place creates the directory, and one place still reads the path.

    The single-read-point invariant (test_disk_index.py) says a journal FILE NAME
    is composed in one place. This is its twin: a second CREATION site is a second
    set of failure semantics, and "created here but not there" is exactly the
    shape of "written where the reader will never look".
    """
    plugin = os.path.dirname(os.path.abspath(disk_index.__file__))
    sources = {}
    for name in sorted(os.listdir(plugin)):
        if name.endswith(".py"):
            with open(os.path.join(plugin, name), "r",
                      encoding="utf-8") as handle:
                sources[name] = handle.read()
    index_source = sources["disk_index.py"]
    # `makedirs` appears once in this module, and it is the ensured one; the helper
    # is defined and called in this module alone.
    assert index_source.count("os.makedirs(") == 1, "a second creation site"
    for name in sources:
        if name != "disk_index.py":
            assert "_ensure_journal_directory" not in sources[name], name
    # The helper is defined once and called once, both in disk_index.py.
    assert index_source.count("def _ensure_journal_directory(") == 1
    assert index_source.count("_ensure_journal_directory(directory)") == 1
    # The single read point, restated from this file so the pair cannot drift:
    # `config.disk_journal_dir()` still has exactly one caller in the plugin.
    callers = [name for name, text in sorted(sources.items())
               if "disk_journal_dir" in text]
    assert callers == ["config.py", "disk_index.py"], callers
    # And the writer's path is the only thing that creates: `refresh()` and
    # `_read_tail_seq()` do not, which is what the two counts above prove.
    assert "os.mkdir(" not in index_source


def test_no_journal_door_raises_when_the_filesystem_refuses_everything(
        tmp_path, monkeypatch):
    """`isdir`, `makedirs` and `getsize` all explode: every door still returns.

    "Never raises" is not a comment, it is a property of the writer thread (which
    shares its failure with an engine's step) and of the scheduler's lookup path
    (which shares its failure with a request). A hostile filesystem is the cheapest
    way to hold the three of them to it.
    """
    journal = use_journal(tmp_path, monkeypatch)
    boom = OSError("the filesystem said no")
    def explode(*args, **kwargs):
        raise boom

    monkeypatch.setattr(os.path, "isdir", explode)
    monkeypatch.setattr(os, "makedirs", explode)
    monkeypatch.setattr(os.path, "getsize", explode)
    verdict = disk_index.journal_append("+", KEY16, 10, GROUP)
    assert verdict["ok"] is False
    doors = [lambda: disk_index.record_spill("kvslot-x.bin", 10),
             lambda: disk_index.record_removal("kvslot-x.bin", 10),
             lambda: disk_index.journal_path(),
             lambda: disk_index.journal_path(RANK),
             disk_index.refresh,
             lambda: disk_index.refresh(force=True),
             disk_index.section,
             disk_index.counters,
             disk_index.journal_directory_state,
             disk_index.journal_fs_state,
             disk_index.journal_fs_label]
    for door in doors:
        assert door() is not None
    assert disk_index.journal_fs_label() in ("unwritable", "unknown", "tmpfs",
                                             "disk")
    counters = disk_index.counters()
    assert counters["journal_write_errors"] >= 1
    # The probe could not tell, or told the truth; either way it did not invent a
    # failure, and no door raised.
    assert disk_index.journal_directory_state()["writable"] in (None, False)
    # Nothing was created on the way to the failure: an exploding filesystem that
    # this module cannot see must not leave a half-built path behind either.
    assert not os.path.exists(str(journal))


def test_the_surfaces_an_operator_reads_all_say_it(tmp_path, monkeypatch):
    """`/disk?op=status`, the snapshot's `disk_restore` section, and the index's own.

    The decision under test is "keep spilling, make the silence impossible", and it
    is only true if the loudness is on the surfaces a human actually queries - not
    in a counter no verb prints. These are those surfaces, all three.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR, str(blocker / "journal"))
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(tmp_path / "store"))
    assert disk_index.journal_append("+", KEY16, 10, GROUP)["ok"] is False

    status = disk_store.disk_for_engine(None, {"op": "status"})
    assert status["ok"] is True, status
    assert status["journal_write_errors"] == 1
    assert status["journal_directory_writable"] is False
    assert status["journal"]["write_errors"] == 1
    assert status["journal"]["writable"] is False
    assert "NotADirectoryError" in str(status["journal"]["directory_error"])
    assert "journal directory" in str(status["journal"]["last_error"])
    assert json.dumps(status, default=str)

    restore = disk_restore.scheduler_section()
    assert restore["index"]["journal_write_errors"] == 1
    assert restore["index"]["journal_directory_writable"] is False
    assert "journal_write_errors" in json.dumps(restore, default=str)
    assert "journal_last_error" in restore["index"]
