r"""The SSD spill path: one KV slot's bytes into one content-addressed file.

Everything runs against a `tmp_path` store and a ctypes-backed fake of the fork's
region tensor - no engine, no torch, no CUDA, no network. What this file is
obliged to prove, in order of what each failure would cost on the live box:

1. A WRITTEN FILE IS THE BYTES WE HANDED IT, at the name the fingerprint dictates.
   A real byte pattern round-trips; a fingerprint change produces a DIFFERENT
   NAME (not a refusal at load: un-findability is the mechanism), and a missing
   fingerprint is a REFUSAL to write at all.
2. NOTHING HAPPENS WHEN IT IS OFF. A fresh boot starts no thread, creates no
   directory, writes no byte, and every entry point says so with a reason and a
   counter. No env var arms it.
3. NOTHING BLOCKS AND NOTHING RAISES. A full queue is a counted DROP, an entry
   point with hostile arguments returns a dict, and `BaseException` still escapes
   (a diagnostic that swallows Ctrl-C is a different kind of outage).
4. THE BOUNDS ARE REAL. A non-contiguous region is refused with the same reasoning
   `blockdump` uses - an address that could leave the allocation is a SIGSEGV, not
   an exception - and a slot index or a region width outside its bounds is refused.
5. THE COMMIT ORDER IS OBSERVABLE. Fail the write at the rename and the final name
   is still absent; spill the same slot twice and the second is a benign duplicate.
6. CAPACITY USES mtime, and the free-space gate refuses rather than filling the
   filesystem.
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from kvtier_dev_plugin import (
    blockdump,
    config,
    disk_index,
    disk_layout,
    disk_spill,
    snapshot,
)

# A small but real slot width: 4 KiB keeps the byte assertions honest while
# staying above disk_layout.MIN_BLOCK_BYTES (1 KiB), which is the floor that
# catches "someone handed us a token count".
SLOT_BYTES = 4096
REGIONS = {"language_model.model.layers.0.linear_attn": SLOT_BYTES}


class FakeRegion:
    """A region tensor, shaped the way the fork's is, backed by real host memory.

    `data_ptr()` is the address of a real ctypes buffer, so `read_slot_bytes`
    performs the same `ctypes.string_at` copy the worker will - the tests read
    bytes rather than a mock's return value. `nbytes` can be set to disagree with
    `shape[0] * stride(0) * element_size()` to stand in for a non-contiguous view.
    """

    def __init__(self, blocks, bytes_per_block, *, itemsize=1, nbytes=None,
                 seed=0) -> None:
        self.shape = (blocks, bytes_per_block // itemsize)
        self._blocks = blocks
        self._bytes_per_block = bytes_per_block
        self._itemsize = itemsize
        self._buffer = ctypes.create_string_buffer(blocks * bytes_per_block)
        for index in range(blocks):
            start = index * bytes_per_block
            pattern = bytes(((index * 7 + offset) % 251)
                            for offset in range(bytes_per_block))
            ctypes.memmove(ctypes.addressof(self._buffer) + start, pattern,
                           bytes_per_block)
        self.nbytes = blocks * bytes_per_block if nbytes is None else nbytes

    def data_ptr(self):
        return ctypes.addressof(self._buffer)

    def stride(self, axis):
        return self._bytes_per_block if axis == 0 else self._itemsize

    def element_size(self):
        return self._itemsize

    def slot_bytes(self, slot):
        start = slot * self._bytes_per_block
        return ctypes.string_at(self.data_ptr() + start, self._bytes_per_block)


class HostileRegion:
    """A tensor-shaped object whose every attribute access raises.

    Standalone rather than a subclass: `FakeRegion.__init__` assigns `shape`, and a
    property that refuses to be assigned would fail in the fixture rather than in
    the code under test.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
    @property
    def shape(self):
        raise RuntimeError("the shape exploded")

    def data_ptr(self):
        raise RuntimeError("the tensor exploded")

    def stride(self, axis):
        raise RuntimeError("the stride exploded")

    def element_size(self):
        raise RuntimeError("the element size exploded")


def caches(blocks=8, **overrides) -> dict:
    regions = dict(REGIONS)
    tensors = {name: FakeRegion(blocks, width, **overrides.get(name, {}))
               for name, width in regions.items()}
    return tensors


def worker(blocks=8, tensors=None):
    return type("FakeWorker", (), {"cpu_kv_caches": tensors or caches(blocks)})()


def fingerprint(**overrides) -> dict:
    base = {
        "model_id": "flash-next-v2",
        "model_revision": "8f3c1d2e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d",
        "cache_dtype": "auto",
        "dtype": "torch.bfloat16",
        "total_bytes_per_block": SLOT_BYTES,
        "group_block_sizes": [1, 4, 4, 8, 16],
        "scheduler_block_size": 16,
        "hash_block_size": 16,
        "prefix_cacheable_group_ids": [0, 1, 2],
        "prefix_caching_hash_algo": "sha256_cbor",
        "none_hash_seed": "0xd4e5f6a7b8c9d0e1",
        "rank": "PP0_TP0",
        "world_size": 4,
        "dcp_world_size": 1,
        "cp_world_size": 1,
    }
    base.update(overrides)
    return base


# THE IDENTITY IS GROUP-QUALIFIED: `KEY` is the engine's full block-hash hex (the
# digest half plus the 4-byte big-endian group id at the END), which is the only
# thing `spill_slot` will name a file with. `KEY16` is the digest half - the
# `key16` a name carries - and is what a refusal for "no group in it" is about.
def full_key(index: int, group: int = 3) -> str:
    """A group-qualified block hash for test slot `index` (group 3 by default)."""
    return f"{index:016x}" + "0" * 48 + f"{group:08x}"


KEY16 = "a3f0b7c1d4e5f607"
GROUP = 3
KEY = KEY16 + "1" * 48 + f"{GROUP:08x}"
# Same digest, different group: the second identity the old name could not express.
KEY_OTHER_GROUP = KEY16 + "1" * 48 + f"{7:08x}"
REAL_FORBIDDEN_ROOTS = disk_layout.FORBIDDEN_ROOT_PREFIXES


@pytest.fixture(autouse=True)
def armed_tier(tmp_path, monkeypatch, reset_dev_state):
    """A configured, ARMED tier rooted in this test's tmp directory.

    The autouse `reset_dev_state` fixture from conftest stops the spill thread and
    zeroes the counters before and after, so no test inherits another's queue. The
    tmpfs refusal is cleared ONLY for the prefix list, because pytest's tmp_path
    lives under /tmp: the rule itself is proved by
    `test_a_tmpfs_root_never_becomes_a_store`.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    root = tmp_path / "kv-store"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(64 * SLOT_BYTES))
    disk_spill.configure(fingerprint())
    disk_spill.arm()
    return root


@pytest.fixture
def disarmed(monkeypatch):
    disk_spill.disarm()
    return disk_spill.disarm()


# --- a fresh boot does zero work --------------------------------------------

def test_a_fresh_boot_does_zero_disk_work(monkeypatch, reset_dev_state):
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    disk_spill.reset_state_for_tests()
    assert disk_spill.capability() == "not_configured"
    report = disk_spill.request_spill(caches(), 3, KEY)
    assert report["ok"] is False
    assert report["reason_kind"] == "not_configured"
    assert disk_spill._state["thread_started"] == 0
    assert disk_spill._state["thread_alive"] is False
    assert disk_spill._state["refused"]["not_configured"] == 1
    # No thread, no queue object, and no directory anywhere.
    assert disk_spill._queue is None


def test_configuring_the_root_without_arming_still_writes_nothing(tmp_path):
    assert disk_spill.capability() == "armed_and_ready"
    disk_spill.disarm()
    assert disk_spill.capability() == "configured_disarmed"
    report = disk_spill.request_spill(caches(), 3, KEY)
    assert report["ok"] is False and report["reason_kind"] == "disarmed"
    assert not tmp_path.joinpath("kv-store").exists()
    assert disk_spill._state["thread_started"] == 0


def test_no_env_var_arms_the_spill_path():
    """The `guard_mode`/`blockdump` rule, applied to the tier that writes KV to disk.

    `config.py` owns every env var the plugin reads. The two disk ones name a
    LOCATION and a SIZE: neither can arm anything, so a launcher script or a
    systemd unit that outlives the operator cannot turn this on.
    """
    names = [name for name in dir(config) if name.startswith("ENV_")]
    assert names
    assert not [name for name in names
                if any(word in name.upper() for word in
                       ("DUMP", "BLOCK", "ARM", "SPILL"))]


def test_arm_requires_a_root_because_a_knob_that_cannot_act_is_a_lie(
        monkeypatch, reset_dev_state):
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    disk_spill.reset_state_for_tests()
    report = disk_spill.arm()
    assert report["armed"] is False
    assert report["capability"] == "not_configured"
    assert "nowhere to write" in report["reason"]


def test_an_arm_expires_on_its_own_clock():
    disk_spill.arm(seconds=5.0)
    assert disk_spill.is_armed() is True
    # Backdate the deadline rather than patching `time.monotonic`: the module reads
    # the clock through the global `time`, so patching it would move the clock out
    # from under pytest itself.
    disk_spill._state["arm_expires_mono"] = time.monotonic() - 1.0
    assert disk_spill.is_armed() is False
    assert disk_spill.capability() == "configured_disarmed"


def test_the_arm_crosses_to_a_worker_as_a_bounded_file(tmp_path, monkeypatch):
    """A worker cannot be told anything in-process, so the arm is a small file.

    The same one-way-per-process problem `blockdump` solves, with the same answer:
    a bounded, stat-first read that fails CLOSED. A file that is too big, does not
    parse, or belongs to a dead boot means DISARMED.
    """
    directory = tmp_path / "diag"
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    disk_spill.disarm()
    assert disk_spill.is_armed() is False
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / disk_spill.ARM_FILE
    path.write_bytes(json.dumps({"armed": True, "expires_ts": time.time() + 60}
                                ).encode())
    disk_spill._state["arm_file_checked_mono"] = None
    assert disk_spill.is_armed() is True
    path.write_bytes(b"{" * 10_000)
    disk_spill._state["arm_file_checked_mono"] = None
    assert disk_spill.is_armed() is False                  # too big: disarmed
    path.write_bytes(b"{not json")
    disk_spill._state["arm_file_checked_mono"] = None
    assert disk_spill.is_armed() is False                  # broken: disarmed
    path.write_bytes(json.dumps({"armed": True,
                                 "expires_ts": time.time() - 1}).encode())
    disk_spill._state["arm_file_checked_mono"] = None
    assert disk_spill.is_armed() is False                  # expired: disarmed


# --- the arm dies at a reboot, by its own contents -------------------------
# The arm file used to be checked for `armed` + `expires_ts` alone, so a file that
# survived a reboot armed the NEW boot - a cold box with the restore path open to
# live traffic and no operator having asked. The protection used to be "`/tmp` is
# tmpfs", which is exactly what the journal move would have destroyed, so the fence
# is in the FILE: a boot identity the reader compares against its own.

def _arm_document(**overrides) -> dict:
    """A well-formed arm document for THIS boot, with the test's edits applied."""
    epoch, scope = config.diagnostics_epoch()
    document = {"version": disk_spill.PROTOCOL_VERSION,
                "writer": disk_spill.WRITER_TAG, "armed": True,
                "armed_ts": time.time(), "expires_ts": time.time() + 600,
                "seconds": 600.0, "root": str(disk_spill._state["root"]),
                "epoch": epoch, "epoch_scope": scope, "pid": os.getpid(),
                "boot_id": disk_spill._boot_id()}
    document.update(overrides)
    return document


def _read_as_a_worker(document, tmp_path, monkeypatch):
    """Write `document` as the arm file and ask the WORKER's question: armed?

    The in-process arm is cleared and the file-verdict cache is dropped, which is
    the state a worker process is in: it never saw the verb, it only has the file.
    """
    directory = tmp_path / "diag-as-a-worker"
    directory.mkdir(exist_ok=True)
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    path = directory / disk_spill.ARM_FILE
    path.write_bytes(json.dumps(document).encode())
    disk_spill._state["armed"] = False
    disk_spill._state["arm_expires_mono"] = None
    disk_spill._state["arm_file_verdict"] = None
    disk_spill._state["arm_file_checked_mono"] = None
    return disk_spill.is_armed(), disk_spill._state["last_reason"], path


def test_an_arm_file_with_a_foreign_epoch_does_not_arm_the_tier(tmp_path,
                                                                monkeypatch):
    """THE REQUESTED PROOF, and it does not depend on the directory being tmpfs.

    The epoch here is the operator's per-boot one (`VLLM_E2_TIER_DIAG_EPOCH`), which
    is the only kind of epoch that can name a boot - with `epoch_scope: "process"` two
    workers of one boot legitimately disagree, and refusing that would disarm a
    healthy tier. The reason NAMES the mismatch: "disarmed" with no reason is how an
    armed-looking box quietly does nothing.
    """
    monkeypatch.setenv(config.ENV_DIAG_EPOCH, "boot-2026-10-05-a")
    armed, reason, _path = _read_as_a_worker(
        _arm_document(epoch="boot-of-the-previous-boot", epoch_scope="env"),
        tmp_path, monkeypatch)
    assert armed is False
    assert "epoch mismatch" in reason, reason
    assert "boot-of-the-previous-boot" in reason, reason
    assert disk_spill._state["arm_file_boot_refusals"] == 1
    assert disk_spill.arm_state()["arm_file_boot_refusals"] == 1
    assert "epoch mismatch" in disk_spill.arm_state()["arm_file_reason"]
    # And the tier really is closed: a spill is refused, not written.
    refused = disk_spill.request_spill(caches(), 1, KEY)
    assert refused["ok"] is False and refused["reason_kind"] == "disarmed"


def test_an_arm_file_naming_another_boot_id_does_not_arm_the_tier(tmp_path,
                                                                  monkeypatch):
    """The fence that needs NO operator configuration: the kernel's own boot id.

    This is the half that covers the default deployment, where the epoch is a
    per-process token and therefore proves nothing about a boot.
    """
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("11111111-2222-3333-4444-555555555555\n", encoding="utf-8")
    monkeypatch.setattr(disk_spill, "BOOT_ID_PATH", str(boot_id))
    armed, reason, _path = _read_as_a_worker(
        _arm_document(boot_id="99999999-8888-7777-6666-555555555555"),
        tmp_path, monkeypatch)
    assert armed is False
    assert "boot id mismatch" in reason, reason
    assert "99999999-8888-7777-6666-555555555555" in reason, reason


def test_an_arm_of_this_boot_still_arms_a_process_that_only_has_the_file(tmp_path,
                                                                         monkeypatch):
    """THE OTHER DIRECTION, which matters just as much: the fence must not become
    "never armed". A worker sees only the file, so a file from THIS boot arms it.
    """
    monkeypatch.setenv(config.ENV_DIAG_EPOCH, "boot-2026-10-06-a")
    armed, reason, path = _read_as_a_worker(_arm_document(), tmp_path, monkeypatch)
    assert armed is True, reason
    assert path.exists()
    # A real arm, written by this process, is readable by a worker of the same boot:
    # the document carries the identity it will be checked against.
    disk_spill.arm()
    written = json.loads(Path(disk_spill.arm()["arm_file"]).read_text(
        encoding="utf-8"))
    assert written["epoch"] == config.diagnostics_epoch()[0]
    assert written["epoch_scope"] == config.diagnostics_epoch()[1]
    assert written["boot_id"] == disk_spill._boot_id()
    disk_spill._state["armed"] = False
    disk_spill._state["arm_file_checked_mono"] = None
    assert disk_spill.is_armed() is True, disk_spill._state["last_reason"]
    assert disk_spill._state["arm_file_boot_refusals"] == 0


def test_a_fellow_worker_of_this_boot_is_never_refused_by_its_own_epoch(tmp_path,
                                                                        monkeypatch):
    """`epoch_scope: "process"` means the epoch is NOT a boot identity.

    With `VLLM_E2_TIER_DIAG_EPOCH` unset every process stamps its own token, so the
    EngineCore's arm file and a worker's token differ WITHIN one boot. Refusing on
    that would silently disarm every worker on the box, which is the failure mode to
    avoid while closing the reboot hole; the boot id is what closes the reboot hole.
    """
    monkeypatch.delenv(config.ENV_DIAG_EPOCH, raising=False)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("11111111-2222-3333-4444-555555555555\n", encoding="utf-8")
    monkeypatch.setattr(disk_spill, "BOOT_ID_PATH", str(boot_id))
    armed, reason, _path = _read_as_a_worker(
        _arm_document(epoch="deadbeef", epoch_scope="process",
                      boot_id="11111111-2222-3333-4444-555555555555"),
        tmp_path, monkeypatch)
    assert armed is True, reason
    assert disk_spill._state["arm_file_boot_refusals"] == 0


def test_an_arm_file_from_before_the_fence_is_believed_and_said_so(tmp_path,
                                                                   monkeypatch):
    """A file with no boot identity in it cannot be PROVEN stale, so it is believed.

    Refusing it would break the documented "arm the tier, then restart the engine"-
    workflow, which is a feature - a REBOOT is what the arm must not survive, and a
    reboot moves the boot id. Files this version writes always carry the identity, so
    the gap is one release wide at most (and the diagnostics directory is tmpfs
    today, so no such file survives the upgrade reboot either).
    """
    armed, _reason, _path = _read_as_a_worker(
        {"armed": True, "expires_ts": time.time() + 600, "epoch": "aabbccdd"},
        tmp_path, monkeypatch)
    assert armed is True
    assert disk_spill._state["arm_file_boot_refusals"] == 0


def test_the_boot_id_is_an_absence_and_never_a_guess(tmp_path, monkeypatch):
    """No readable `/proc/sys/kernel/random/boot_id` = no boot-id evidence.

    The fence must not become a way for a container without that file to disarm the
    tier: an unreadable identity means the rule does not fire, and the epoch rule is
    still there when the operator set a per-boot epoch.
    """
    monkeypatch.setattr(disk_spill, "BOOT_ID_PATH",
                        str(tmp_path / "does-not-exist"))
    assert disk_spill._boot_id() is None
    armed, reason, _path = _read_as_a_worker(_arm_document(boot_id=None),
                                             tmp_path, monkeypatch)
    assert armed is True, reason
    monkeypatch.setattr(disk_spill, "BOOT_ID_PATH", str(tmp_path / "huge"))
    huge = tmp_path / "huge"
    huge.write_text("x" * 4096, encoding="utf-8")
    assert disk_spill._boot_id() is None


# --- the round trip: real bytes, the name the fingerprint dictates ----------

def test_a_slot_round_trips_to_the_file_the_fingerprint_names():
    tensors = caches(blocks=4)
    region = next(iter(tensors.values()))
    expected = region.slot_bytes(2)
    result = disk_spill.spill_slot(tensors, 2, KEY)
    assert result["ok"] is True, result
    assert result["bytes_written"] == SLOT_BYTES
    assert result["sha256"] == hashlib.sha256(expected).hexdigest()
    digest = disk_layout.fingerprint_digest(fingerprint())["geometry_digest"]
    assert result["group_id"] == GROUP
    assert result["group_field"] == f"g{GROUP}"
    assert Path(result["path"]).name == f"kvslot-{KEY16}-g{GROUP}-{digest}.bin"
    assert Path(result["path"]).read_bytes() == expected
    assert result["path"].endswith(f"/a3/kvslot-{KEY16}-g{GROUP}-{digest}.bin")


def test_a_key_without_the_group_suffix_is_refused_before_any_byte_is_written():
    """The identity bug, at the door that writes.

    A bare `key16` is what the engine's hash LOOKS like after truncation, and it is
    exactly what two groups share, so naming a file from it is refused - with no
    directory created and no byte written - rather than silently producing a name
    another group will overwrite.
    """
    root = disk_spill._state["root"]
    refused = disk_spill.spill_slot(caches(), 1, KEY16)
    assert refused["ok"] is False
    assert refused["reason_kind"] == "bad_key"
    assert "group" in refused["reason"]
    planned = disk_spill.plan_slot_write(caches(), 1, KEY16)
    assert planned["ok"] is False
    assert not os.path.exists(os.path.join(str(root), KEY16[:2]))


def test_two_groups_over_one_chain_get_two_files_and_a_counter():
    """The collision witness, proven end to end on real bytes.

    Same token chain, two KV-cache groups, so two identities sharing a `key16`:
    BOTH spill, to two files, with the group id visible in each name - and the
    second one counts a `group_collision`, which is the number the tier is armed to
    measure. Nothing is refused: the names are distinct, so the event is data.
    """
    first = disk_spill.spill_slot(caches(), 1, KEY)
    assert first["ok"] is True
    assert first["group_collision"] is None
    again_same_group = disk_spill.spill_slot(caches(), 1, KEY)
    assert again_same_group["duplicate"] is True
    assert disk_spill.counters()["group_collisions"] == 0
    other = disk_spill.spill_slot(caches(), 1, KEY_OTHER_GROUP)
    assert other["ok"] is True
    assert other["duplicate"] is False
    assert other["path"] != first["path"]
    assert "-g7-" in other["path"] and "-g3-" in first["path"]
    state = disk_spill.counters()
    assert state["group_collisions"] == 1
    assert state["refused"]["write_failed"] == 0
    sample = state["group_collision_samples"][-1]
    assert sample["key16"] == KEY16
    assert sorted([sample["group_id_a"], sample["group_id_b"]]) == [3, 7]
    assert sample["sibling_on_disk"] is True
    assert json.dumps(state["group_collision_samples"], default=str)
    # ...and the third group of the same chain is a third event, not a re-count of
    # the first pair.
    disk_spill.spill_slot(caches(), 1,
                          KEY16 + "1" * 48 + f"{11:08x}")
    assert disk_spill.counters()["group_collisions"] == 2


def test_a_multi_region_slot_concatenates_in_name_order():
    tensors = {"b_region": FakeRegion(3, 2048, seed=1),
               "a_region": FakeRegion(3, 2048, seed=2)}
    disk_spill.configure(fingerprint(total_bytes_per_block=4096))
    read = disk_spill.read_slot_bytes(tensors, 1, 4096)
    assert read["ok"] is True
    assert read["regions"] == ["a_region", "b_region"]
    assert read["bytes"] == (tensors["a_region"].slot_bytes(1)
                             + tensors["b_region"].slot_bytes(1))


def test_a_fingerprint_change_produces_a_different_name_not_a_load_error():
    """The requirement, at the spill boundary: two names, and neither is findable
    by the other boot. There is no reader to refuse - that is the design."""
    spill = disk_spill.spill_slot(caches(), 1, KEY)
    assert spill["ok"] is True
    other = disk_layout.fingerprint_digest(
        fingerprint(total_bytes_per_block=SLOT_BYTES,
                    dtype="torch.float16"))["geometry_digest"]
    assert disk_layout.build_slot_path(str(disk_spill._state["root"]), KEY,
                                       other)["path"] != spill["path"]
    disk_spill.configure(fingerprint(dtype="torch.float16"))
    looked = disk_spill.verify_slot(KEY)
    assert looked["ok"] is False
    assert looked["exists"] is False


def test_verify_reads_back_exactly_what_was_written():
    tensors = caches(blocks=3)
    disk_spill.spill_slot(tensors, 2, KEY)
    checked = disk_spill.verify_slot(KEY)
    assert checked["ok"] is True
    assert checked["read_bytes"] == SLOT_BYTES
    assert checked["truncated"] is False
    assert checked["sha256"] == hashlib.sha256(
        next(iter(tensors.values())).slot_bytes(2)).hexdigest()


def test_verifying_a_slot_that_was_never_written_is_a_clean_absence():
    checked = disk_spill.verify_slot("0123456789abcdef" + "0" * 8)
    assert checked["ok"] is False
    assert checked["exists"] is False
    assert "not there" in checked["reason"]
    # A key with no group suffix cannot even be resolved to a name.
    ungrouped = disk_spill.verify_slot("0123456789abcdef")
    assert ungrouped["ok"] is False
    assert "group" in ungrouped["reason"]


# --- the fingerprint: missing is a refusal, never a guess ------------------

def test_a_missing_fingerprint_refuses_before_any_byte_is_written():
    disk_spill.configure(None)
    refused = disk_spill.spill_slot(caches(), 0, KEY)
    assert refused["ok"] is False
    assert refused["reason_kind"] == "fingerprint_missing"
    queued = disk_spill.request_spill(caches(), 0, KEY)
    assert queued["ok"] is False
    assert queued["reason_kind"] == "fingerprint_missing"
    # ONE counter: `spill_slot` is the synchronous core and the counters live at
    # the acceptance boundary (the engine thread pays a counter, not two).
    assert disk_spill._state["refused"]["fingerprint_missing"] == 1


def test_an_unknown_fingerprint_field_is_refused_and_counted():
    report = disk_spill.configure(fingerprint(sliding_window=2048))
    assert report["configured"] is False
    assert "unknown field" in report["reason"]
    assert disk_spill._state["refused"]["fingerprint_invalid"] == 1
    assert disk_spill.geometry_digest() is None


@pytest.mark.parametrize("field", ["model_revision", "cache_dtype", "rank",
                                   "none_hash_seed", "hash_block_size"])
def test_a_fingerprint_with_a_hole_is_refused_at_spill_time(field):
    broken = fingerprint()
    broken.pop(field)
    refused = disk_spill.spill_slot(caches(), 0, KEY, fingerprint=broken)
    assert refused["ok"] is False
    assert refused["reason_kind"] == "fingerprint_invalid"


# --- the bounds: what may be read out of a region tensor -------------------

def test_a_non_contiguous_region_is_refused_not_read():
    """`shape[0] * stride(0) * itemsize != nbytes` is a SIGSEGV waiting to happen."""
    broken = {"region": FakeRegion(4, SLOT_BYTES, nbytes=SLOT_BYTES * 2)}
    refused = disk_spill.read_slot_bytes(broken, 0, SLOT_BYTES)
    assert refused["ok"] is False
    assert "not a flat run" in refused["reason"]
    spilled = disk_spill.spill_slot(broken, 0, KEY)
    assert spilled["ok"] is False
    assert spilled["reason_kind"] == "region_not_contiguous"


@pytest.mark.parametrize("slot", [-1, 8, 1000, "3", None, True, 1.5])
def test_a_slot_index_outside_the_region_is_refused(slot):
    refused = disk_spill.read_slot_bytes(caches(blocks=8), slot, SLOT_BYTES)
    assert refused["ok"] is False
    assert refused["reason"]


def test_a_slot_wider_than_the_fingerprint_claims_is_refused():
    """A region whose width alone exceeds the declared slot width is a lie."""
    refused = disk_spill.read_slot_bytes({"r": FakeRegion(4, SLOT_BYTES * 2)},
                                         0, SLOT_BYTES)
    assert refused["ok"] is False
    assert "past the slot" in refused["reason"]


def test_regions_that_do_not_add_up_to_the_fingerprint_are_refused():
    """The PP0/PP1 trap: 11,468,800 B and 12,902,400 B are different engines."""
    pp1_bytes = 12_902_400
    refused = disk_spill.read_slot_bytes(caches(blocks=2), 0, pp1_bytes)
    assert refused["ok"] is False
    assert refused["reason_kind"] == "region_width_mismatch"
    assert "not what this fingerprint describes" in refused["reason"]


def test_a_region_set_that_is_not_a_dict_is_refused():
    for junk in (None, [], "tensors", {}, 7):
        assert disk_spill.read_slot_bytes(junk, 0, SLOT_BYTES)["ok"] is False


def test_a_tensor_that_raises_on_every_attribute_is_a_refusal_not_a_fault():
    refused = disk_spill.read_slot_bytes({"bad": HostileRegion(2, 1024)}, 0, 1024)
    assert refused["ok"] is False
    assert refused["reason"]


def test_a_sample_of_a_slot_and_the_spilled_bytes_agree_byte_for_byte():
    """`blockdump`'s own reader over the same tensor: the two paths cannot disagree."""
    tensors = caches(blocks=4)
    tensor = tensors["language_model.model.layers.0.linear_attn"]
    region_name = "language_model.model.layers.0.linear_attn"
    named = disk_spill.sample_slot_bytes(worker(tensors=tensors), region_name,
                                         2, 512)
    assert named["ok"] is True
    assert named["bytes"] == tensor.slot_bytes(2)[:512]
    assert named["sha256"] == hashlib.sha256(tensor.slot_bytes(2)[:512]).hexdigest()
    assert named["bytes_per_file"] == SLOT_BYTES
    assert named["region"] == region_name
    assert blockdump.MAX_BYTES_PER_REQUEST >= 512


def test_a_sample_naming_an_unknown_region_names_the_regions_it_does_have():
    refused = disk_spill.sample_slot_bytes(worker(tensors=caches()), "7", 0, 16)
    assert refused["ok"] is False
    assert "linear_attn" in refused["reason"]


# --- the commit protocol, at the spill boundary ----------------------------

def test_the_spill_path_calls_no_fsync_at_all(monkeypatch):
    """NO FSYNC ON THE SPILL PATH, PROVEN BY A COUNT, on both write routes.

    These are cache bytes: the price of a lost span is one recompute, and the
    safety net is that a journaled-but-absent file is classified `file_missing` at
    read time and pruned before any DMA (see `disk_layout.write_committed_file`).
    A fsync of an 11.4 MB slot on the writer thread buys nothing against that and
    costs latency, so the production caller passes `sync=False`.

    The assertion is a COUNT of `os.fsync` calls, not a reading of the flag, so a
    future edit that puts a sync back on this path fails HERE. The same test then
    shows the `sync=` mechanism itself is intact - a caller that asks for
    durability still gets it - because keeping the capability is part of the deal.
    """
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))

    direct = disk_spill.spill_slot(caches(), 1, KEY)
    assert direct["ok"] is True, direct
    assert direct["directory_synced"] is False, "the spill path does not sync"
    assert calls == [], f"the synchronous spill fsynced {len(calls)}x"

    # The writer thread goes through the same call site; prove it too.
    queued = disk_spill.request_spill(caches(), 2,
                                      "b" + KEY16[1:] + "1" * 48
                                      + f"{GROUP:08x}")
    assert queued["ok"] is True, queued
    assert disk_spill.drain(timeout_seconds=5.0)["drained"] is True
    assert calls == [], f"the spill thread fsynced {len(calls)}x"

    # The bytes still landed: dropping durability must not change one byte.
    assert Path(direct["path"]).read_bytes() == next(
        iter(caches().values())).slot_bytes(1)

    # AND THE MECHANISM IS STILL THERE: a caller that wants a durable file asks
    # for it and gets real fsyncs (a test, or any future system-of-record user).
    durable = disk_layout.write_committed_file(
        str(Path(disk_spill._state["root"]) / "durable.bin"), b"bytes",
        sync=True)
    assert durable["ok"] is True and durable["directory_synced"] is True
    assert len(calls) == 2, "sync=True must fsync the file AND the directory"


def test_a_failed_rename_leaves_no_file_at_the_final_name(monkeypatch):
    def fail_replace(a, b, **kwargs):
        raise OSError("the rename blew up")

    monkeypatch.setattr(os, "replace", fail_replace)
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is False
    assert result["reason_kind"] == "write_failed"
    assert result["final_exists"] is False
    left = sorted(name for shard in Path(disk_spill._state["root"]).glob("*")
                  for name in os.listdir(shard)) if os.path.isdir(
        disk_spill._state["root"]) else []
    assert left == []


def test_spilling_the_same_slot_twice_is_a_benign_duplicate():
    """Content-addressed means a repeat is the same identity: a no-op, not an error.

    The first write commits, the second finds the name already occupied and
    declines WITHOUT rewriting 4 KiB (or 11.4 MB on the box) of identical bytes.
    Nothing is counted as a failure, and the file's mtime does not move.
    """
    first = disk_spill.spill_slot(caches(), 1, KEY)
    assert first["ok"] is True and first["duplicate"] is False
    stamp = os.stat(first["path"]).st_mtime_ns
    second = disk_spill.spill_slot(caches(), 1, KEY)
    assert second["ok"] is True and second["duplicate"] is True
    assert second["bytes_written"] == 0
    assert os.stat(first["path"]).st_mtime_ns == stamp
    # Through the queue, the counters say the same thing: a repeat is a duplicate
    # and a new identity is a spill, and neither is a failure.
    disk_spill.request_spill(caches(), 1, KEY)             # written above
    disk_spill.request_spill(caches(), 2,
                             "b" + KEY16[1:] + "1" * 48 + f"{GROUP:08x}")
    assert disk_spill.drain(timeout_seconds=5.0)["drained"] is True
    assert disk_spill._state["spilled"] == 1
    assert disk_spill._state["duplicates"] == 1
    assert disk_spill._state["refused"]["write_failed"] == 0


def test_a_write_that_the_filesystem_refuses_is_counted_not_raised(monkeypatch):
    original = os.write

    def refuse(fd, data):
        raise OSError("ENOSPC")

    monkeypatch.setattr(os, "write", refuse)
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is False
    assert result["reason_kind"] == "write_failed"
    assert original is not None


def test_a_full_filesystem_refuses_the_write_rather_than_filling_it(monkeypatch):
    """`statvfs` first, always: a tier that fills `/home` is an outage."""
    monkeypatch.setattr(disk_layout, "free_space",
                        lambda path: disk_layout._space_fail("only 10 bytes free"))
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is False
    assert result["reason_kind"] == "no_free_space"
    assert result["final_exists"] is False


def test_a_key_that_is_not_the_engine_key_shape_is_refused():
    for junk in ("nope", "a3f0", None, 7, ""):
        refused = disk_spill.spill_slot(caches(), 0, junk)
        assert refused["ok"] is False
        assert refused["reason_kind"] == "bad_key"


def test_a_journal_on_tmpfs_still_spills_and_says_so_loudly(tmp_path, monkeypatch,
                                                            caplog):
    """A degraded-but-loud tier beats a silent one, in EITHER direction.

    The journal's filesystem is a WARNING and a reported field, never a gate: a
    tier that refused to write because its INDEX lives in `/tmp` would be a
    diagnostic causing the outage it measures. So the spill lands, the warning
    fires once, and the verb says `journal.fs.on_tmpfs: true`.
    """
    mounts = tmp_path / "mounts"
    mounts.write_text("tmpfs /tmp tmpfs rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(disk_layout, "PROC_MOUNTS_PATH", str(mounts))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")   # a rank names a journal
    monkeypatch.setenv(config.ENV_DISK_JOURNAL_DIR,
                       str(tmp_path / "journal"))   # under /tmp -> tmpfs
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is True, result
    assert result["journalled"] is True, result     # the write AND the index worked
    section = disk_spill.section()
    assert section["journal"]["fs"]["on_tmpfs"] is True
    assert section["journal"]["fs"]["label"] == "tmpfs"
    assert section["journal_failures"] == 0
    assert sum(disk_spill._state["refused"].values()) == 0
    warnings = [rec.getMessage() for rec in caplog.records
                if "spill JOURNAL directory" in rec.getMessage()]
    assert len(warnings) == 1, warnings
    # Again, and nothing new is logged: the idiom is once per episode.
    assert disk_spill.spill_slot(caches(), 2,
                                 "c" + KEY16[1:] + "1" * 48
                                 + f"{GROUP:08x}")["ok"] is True
    assert len([rec for rec in caplog.records
                if "spill JOURNAL directory" in rec.getMessage()]) == 1


# --- the queue: bounded, dropping, and never blocking ----------------------

def test_a_full_queue_drops_and_counts_rather_than_growing(monkeypatch):
    """The engine thread never waits on a disk that is busy.

    `put_nowait` into a queue at its cap is the whole policy: the job is dropped,
    a counter moves, and the call returns. An unbounded queue on a box that is
    already slow is a memory leak with a nicer name.
    """
    monkeypatch.setattr(disk_spill, "MAX_QUEUE_DEPTH", 2)
    monkeypatch.setattr(disk_spill, "_ensure_thread", lambda: True)
    monkeypatch.setattr(disk_spill, "_window_open", lambda now: True)
    accepted = [disk_spill.request_spill(caches(), slot, KEY)["ok"]
                for slot in range(5)]
    assert accepted == [True, True, False, False, False]
    assert disk_spill._state["drops"] == 3
    assert disk_spill._state["refused"]["queue_full"] == 3
    assert disk_spill._state["queue_depth"] == 2
    assert disk_spill._state["queue_depth_peak"] == 2


def test_requesting_a_spill_never_blocks_the_caller():
    """20 distinct slots: the call is a `put_nowait`, and the thread does the I/O."""
    started = time.perf_counter()
    for slot in range(20):
        disk_spill.request_spill(caches(blocks=64), slot, full_key(slot))
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0                       # 20 accepted jobs, all queued
    drained = disk_spill.drain(timeout_seconds=10.0)
    assert drained["drained"] is True
    assert disk_spill._state["spilled"] == 20
    assert disk_spill._state["bytes_written"] == 20 * SLOT_BYTES


def test_a_queued_job_lands_at_the_name_the_fingerprint_names():
    disk_spill.request_spill(caches(), 3, KEY)
    assert disk_spill.drain(timeout_seconds=5.0)["drained"] is True
    checked = disk_spill.verify_slot(KEY)
    assert checked["ok"] is True
    assert checked["read_bytes"] == SLOT_BYTES


def test_the_write_rate_window_refuses_past_its_cap(monkeypatch):
    monkeypatch.setattr(disk_spill, "MAX_FILES_PER_WINDOW", 3)
    monkeypatch.setattr(disk_spill, "MAX_QUEUE_DEPTH", 100)
    accepted = [disk_spill.request_spill(caches(), slot, KEY)["ok"]
                for slot in range(6)]
    assert accepted == [True, True, True, False, False, False]
    assert disk_spill._state["refused"]["over_rate_window"] == 3


def test_the_writer_thread_is_started_once_and_is_a_single_thread():
    for slot in range(4):
        disk_spill.request_spill(caches(), slot, KEY)
    disk_spill.drain(timeout_seconds=5.0)
    assert disk_spill._state["thread_started"] == 1
    alive = [thread.name for thread in __import__("threading").enumerate()
             if thread.name == "kvtier-disk-spill"]
    assert len(alive) == 1


# --- the injected resolver: the shape of the wiring gap --------------------

def test_a_resolver_named_slot_spills_and_an_unnamed_slot_is_counted():
    keys = {3: KEY}
    ok = disk_spill.request_spill(caches(), 3, resolver=keys.get)
    assert ok["ok"] is True
    missing = disk_spill.request_spill(caches(), 9, resolver=keys.get)
    assert missing["ok"] is False
    assert missing["reason_kind"] == "no_key_for_slot"
    assert disk_spill._state["refused"]["no_key_for_slot"] == 1


def test_a_resolver_that_raises_is_a_refusal_not_a_fault():
    def angry(slot):
        raise RuntimeError("the resolver exploded")

    refused = disk_spill.request_spill(caches(), 1, resolver=angry)
    assert refused["ok"] is False
    assert refused["reason_kind"] == "no_key_for_slot"


def test_a_resolver_that_ignores_base_exception_still_lets_it_escape():
    """`except Exception`, never `except BaseException`: Ctrl-C reaches the engine."""
    def ctrl_c(slot):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        disk_spill.request_spill(caches(), 1, resolver=ctrl_c)


def test_a_spill_with_no_key_and_no_resolver_reports_the_engine_gap():
    refused = disk_spill.request_spill(caches(), 4)
    assert refused["ok"] is False
    assert "does not carry a key to the worker" in refused["reason"]


def test_the_worker_object_is_read_by_name_like_worker_diag_does():
    disk_spill.request_spill(worker=worker(tensors=caches()), slot=2, key=KEY)
    assert disk_spill.drain(timeout_seconds=5.0)["drained"] is True
    assert disk_spill._state["spilled"] == 1


# --- capacity: mtime is the clock, and deleting is safe -------------------

def test_eviction_sheds_the_oldest_file_by_mtime_first():
    written = []
    for index in range(3):
        result = disk_spill.spill_slot(caches(), index, full_key(index))
        assert result["ok"] is True
        written.append(Path(result["path"]))
    # Backdate them so the ORDER is chosen by the test and not by the clock.
    for offset, path in enumerate(reversed(written)):
        stamp = 1_600_000_000 + offset * 100
        os.utime(path, (stamp, stamp))
    oldest = written[-1]
    newest = written[0]
    disk_spill._state["ceiling_bytes"] = SLOT_BYTES + SLOT_BYTES // 2
    disk_spill._state["estimated_bytes"] = 3 * SLOT_BYTES
    disk_spill._enforce_ceiling()
    assert not oldest.exists(), "the oldest file by mtime had to go"
    assert newest.exists(), "the newest file must survive an LRU pass"
    assert disk_spill._state["evictions"] >= 1


def test_no_eviction_happens_under_the_ceiling():
    disk_spill.spill_slot(caches(), 0, KEY)
    disk_spill._state["ceiling_bytes"] = 100 * SLOT_BYTES
    disk_spill._enforce_ceiling()
    assert disk_spill._state["evictions"] == 0


def test_a_ceiling_that_is_not_configured_never_deletes_anything():
    disk_spill.spill_slot(caches(), 0, KEY)
    disk_spill._state["ceiling_bytes"] = None
    disk_spill._state["estimated_bytes"] = 10 ** 12
    disk_spill._enforce_ceiling()
    assert disk_spill._state["evictions"] == 0


def test_a_tmpfs_root_never_becomes_a_store(monkeypatch, tmp_path):
    """`/tmp` is tmpfs on the box: the tier there would spend the RAM it relieves."""
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES",
                        REAL_FORBIDDEN_ROOTS)
    monkeypatch.setenv(config.ENV_DISK_ROOT, "/tmp/kv-tier/disk")
    assert disk_spill.disk_root() is None
    assert disk_spill.capability() == "not_configured"
    refused = disk_spill.request_spill(caches(), 0, KEY)
    assert refused["ok"] is False
    assert refused["reason_kind"] == "not_configured"


def test_a_garbage_ceiling_falls_back_to_no_ceiling_never_to_zero(
        monkeypatch):
    monkeypatch.setenv(config.ENV_DISK_BYTES, "sixty-four-gigabytes")
    assert disk_spill.capacity_ceiling_bytes() is None
    monkeypatch.setenv(config.ENV_DISK_BYTES, "0")
    assert disk_spill.capacity_ceiling_bytes() is None
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(64 * 1024 ** 3))
    assert disk_spill.capacity_ceiling_bytes() == 64 * 1024 ** 3


# --- containment: every door catches Exception and reports ----------------

def test_no_entry_point_can_raise_an_exception(reset_dev_state):
    """The `cpu_capacity`/`blockdump` posture, at every public door."""
    for call in (lambda: disk_spill.request_spill(),
                 lambda: disk_spill.request_spill(None, None, None),
                 lambda: disk_spill.request_spill(object(), "x", 7),
                 lambda: disk_spill.request_spill(caches(), 1, KEY,
                                                  fingerprint=7),
                 lambda: disk_spill.request_spill(worker=None, slot=None),
                 lambda: disk_spill.spill_slot(None, None, None),
                 lambda: disk_spill.spill_slot({"a": object()}, 0, KEY),
                 lambda: disk_spill.plan_slot_write(None, None, None),
                 lambda: disk_spill.read_slot_bytes(None, None, None),
                 lambda: disk_spill.sample_slot_bytes(None, None, 0, 8),
                 lambda: disk_spill.verify_slot(None),
                 lambda: disk_spill.verify_slot("zz"),
                 lambda: disk_spill.configure(7),
                 lambda: disk_spill.configure(None),
                 lambda: disk_spill.arm("soon"),
                 lambda: disk_spill.arm(60),
                 lambda: disk_spill.disarm(),
                 lambda: disk_spill.drain(0),
                 lambda: disk_spill.stop_writer(0),
                 lambda: disk_spill.section(),
                 lambda: disk_spill.counters(),
                 lambda: disk_spill.capability(),
                 lambda: disk_spill.arm_state(),
                 lambda: (disk_spill._enforce_ceiling(), "counted")[1],
                 lambda: (disk_spill._handle_job({}), "counted")[1],
                 lambda: disk_spill._walk_store("/definitely/not/here")):
        result = call()
        assert result is not None
        json.dumps(result, default=str)


def test_the_status_section_is_stable_and_reports_every_counter():
    disk_spill.request_spill(caches(), 0, KEY)
    disk_spill.request_spill(caches(), 1, "nope")
    disk_spill.drain(timeout_seconds=5.0)
    section = disk_spill.section()
    for key in ("capability", "configured", "root", "ceiling_bytes", "armed",
                "spilled", "duplicates", "bytes_written", "write_seconds",
                "read_seconds", "drops", "refused", "evictions", "evicted_bytes",
                "errors", "thread_started", "queue_depth", "caps", "naming",
                "wiring", "read_path", "inert_when_off", "units"):
        assert key in section, key
    # Every refusal reason is a counter AND a described number: a new reason that
    # skips the units table fails here rather than arriving dimensionless.
    assert set(section["refused"]) - {"units"} == set(disk_spill.REFUSAL_REASONS)
    assert set(section["refused"]["units"]) == set(disk_spill.REFUSAL_REASONS)
    assert section["spilled"] == 1
    assert section["refused"]["bad_key"] == 1
    assert section["capability"] == "armed_and_ready"
    assert section["caps"]["max_bytes_per_file"] == SLOT_BYTES
    assert section["caps"]["units"]["max_queue_depth"] == "jobs"
    # The wiring says, in the status payload, WHO owns the engine hook and what
    # happens on an engine that does not carry the metadata fields yet. The first
    # version of this text read "NO engine hook is installed", which stopped being
    # true the day `disk_store` wrapped bind_connector_metadata/get_finished; the
    # honest statement is that the hook exists and REFUSES on an unpatched boot.
    assert "disk_store" in section["wiring"]
    assert "UNPATCHED engine" in section["wiring"]
    assert "cached_block_hash_to_block" in section["read_path"]


def test_a_broken_snapshot_section_is_data_not_a_500(monkeypatch):
    def explode():
        raise RuntimeError("the section exploded")

    monkeypatch.setattr(disk_spill, "section", explode)
    view = snapshot.dev_state_sections()
    assert view["disk_spill"]["error"].startswith("RuntimeError")
    assert view["disk_spill_armed"] is None


def test_the_snapshot_publishes_the_disk_tier_counters():
    view = snapshot.dev_state_sections()
    assert view["disk_spill"]["capability"] == "armed_and_ready"
    assert view["disk_spill_armed"] is True
    assert view["disk_spill"]["refused"]["not_configured"] == 0


def test_the_module_imports_only_the_standard_library():
    """AST over the file, so a conditional import cannot hide (the repo rule)."""
    tree = ast.parse(Path(disk_spill.__file__).read_text(encoding="utf-8"))
    pulled = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pulled.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pulled.add(node.module.split(".")[0])
    assert pulled <= {"__future__", "ctypes", "hashlib", "json", "os", "queue",
                      "threading", "time", "blockdump", "config", "disk_layout",
                      "snapshot"}, sorted(pulled)
    for module in ("torch", "vllm", "numpy"):
        assert module not in sys.modules


# --- the units contract, applied to the disk payloads ------------------------
# The plugin has a units contract (`test_observability::test_every_number_...`):
# every number in a published payload declares its unit. The disk payloads are new
# numbers an operator will read off the snapshot, so they are held to it too - and
# `disk_spill` publishes the write-side timings (elapsed_seconds, bytes_written)
# that nothing else in the plugin reports.

def _walk_payload_numbers(node, units=None, found=None):
    """Yield (payload dict, units map) for every dict carrying a `units` map."""
    if found is None:
        found = []
    if isinstance(node, dict):
        own = node.get("units")
        current = own if isinstance(own, dict) else units
        if current is not None:
            found.append((node, current))
        for value in node.values():
            _walk_payload_numbers(value, current, found)
    elif isinstance(node, list):
        for value in node:
            _walk_payload_numbers(value, units, found)
    return found


def digest(**overrides):
    """A concrete digest for the layout payloads built in these tests."""
    return disk_layout.fingerprint_digest(fingerprint(**overrides))["geometry_digest"]


def _verify_and_plan_payloads():
    """The payloads built INSIDE a spill, captured for the units walk.

    These use the ambient `armed_tier` fixture's root and ceiling rather than
    arguments: `configure(fingerprint)` takes only the fingerprint, because the
    location and the ceiling are the operator's env, not a caller's choice.
    """
    planned = disk_spill.plan_slot_write(caches(), 0, KEY)
    disk_spill.spill_slot(caches(), 0, KEY)
    verified = disk_spill.verify_slot(KEY)
    sampled = disk_spill.sample_slot_bytes(worker(tensors=caches()),
                                           next(iter(REGIONS)), 0, SLOT_BYTES)
    return [planned, verified, sampled]


def _entries(root, *specs):
    """Sweep-shaped entries: (index, size_bytes, mtime_ns) with real names."""
    geo = digest()
    named = [disk_layout.build_slot_name(full_key(index), geo)["filename"]
             for index, _size, _when in specs]
    return [{"path": f"{root}/a3/{name}",
             "filename": name,
             "size_bytes": size, "mtime_ns": when}
            for name, (_index, size, when) in zip(named, specs, strict=True)]


def _all_disk_payloads():
    """Every disk payload the two modules can build, healthy and refused."""
    with tempfile.TemporaryDirectory() as work:
        root = Path(work) / "store"
        geo = digest()
        made = disk_layout.build_slot_path(root, KEY, geo)
        yield disk_layout.fingerprint_section(fingerprint())
        yield disk_layout.fingerprint_section(None)
        yield disk_layout.validate_fingerprint(fingerprint())
        yield disk_layout.validate_fingerprint({})
        yield disk_layout.fingerprint_digest(fingerprint())
        yield disk_layout.fingerprint_digest({})
        yield disk_layout.validate_key16(KEY16)
        yield disk_layout.validate_key16(KEY16[:12])
        yield disk_layout.split_block_key(KEY)
        yield disk_layout.split_block_key(KEY16)
        yield disk_layout.split_block_key(None)
        yield disk_layout.validate_root(root)
        yield disk_layout.validate_root("/tmp/store")
        yield disk_layout.build_slot_name(KEY, geo)
        yield disk_layout.build_slot_name("nope", geo)
        yield made
        yield disk_layout.build_slot_path("", KEY, geo)
        yield disk_layout.parse_path(made["filename"])
        yield disk_layout.parse_path("kvslot-x.bin")
        yield disk_layout.classify_name(made["filename"], geo)
        yield disk_layout.classify_name(made["filename"], "a" * 12,
                                        peer_digests=(geo,))
        yield disk_layout.boot_sweep(root, geo)
        yield disk_layout.boot_sweep(root, geo, peer_digests=("a" * 12,))
        yield disk_layout.boot_sweep(root / "missing", geo)
        yield disk_layout.boot_sweep(root, "not a digest")
        yield disk_layout.write_committed_file(made["path"], b"x" * 8)
        yield disk_layout.write_committed_file(made["path"], b"")
        yield disk_layout.lru_eviction_plan(_entries(root, (0, 4096, 1)),
                                            geometry_digest=geo,
                                            ceiling_bytes=8,
                                            used_bytes=4096)
        yield disk_layout.lru_eviction_plan([], geometry_digest=geo,
                                            ceiling_bytes=8)
        yield disk_layout.execute_unlink_plan(root, [], dry_run=True)
        yield disk_layout.execute_unlink_plan(root, [made["path"]],
                                              dry_run=False,
                                              geometry_digest=geo)
        yield disk_layout.free_space(root)
        yield disk_layout.free_space(root / "missing")
        yield disk_layout.space_check(root, watermark_bytes=1024)
        yield disk_layout.space_check(root / "missing", watermark_bytes=1024)
        yield disk_layout.space_check(root, watermark_percent=50,
                                      extra_bytes=SLOT_BYTES)
        for payload in _verify_and_plan_payloads():
            yield payload
        yield disk_spill.section()
        yield disk_spill.drain(timeout_seconds=0.01)
        yield disk_spill.stop_writer()
        # The journal's own payloads are held to the same units contract: a counter
        # that reaches a dashboard without a unit is the same bug in a new module.
        yield disk_index.section()
        yield disk_index.record_spill("kvslot-nope.bin", 10)
        yield disk_index.journal_append("+", KEY16, 10, 3)
        yield disk_index.journal_path("not/a/rank")
        yield disk_index.lookup(KEY)
        yield disk_index.lookup(KEY16)


def test_every_number_in_a_disk_payload_declares_its_unit():
    """Two directions: nothing says `unknown`, and nothing is left undeclared.

    The `unknown` direction is the plugin's existing rule (a unit table that does
    not know a name says so loudly). The undeclared direction is the one that
    rots silently: someone adds a field to a payload and never touches the table,
    and the number arrives on a dashboard dimensionless.
    """
    seen = 0
    gaps = []
    for payload in _all_disk_payloads():
        for node, units in _walk_payload_numbers(payload):
            gaps.extend(sorted(
                key for key, value in node.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and key not in units
            ))
            for name, unit in units.items():
                assert unit != "unknown", (name, payload)
                seen += 1
    assert not sorted(set(gaps)), gaps
    assert seen > 100


# --- the journal: the spill telling the scheduler the file exists -----------

@pytest.fixture
def channel(tmp_path, monkeypatch):
    """A diagnostics directory and a rank: the whole requirement for a journal.

    The tier stays armed by the autouse `armed_tier`; this only opens the channel
    the journal lives in, so a test that does not ask for it proves the tier works
    WITHOUT one.
    """
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    return directory


def journal_lines():
    where = disk_index.journal_path()
    assert where["ok"] is True, where["reason"]
    if not os.path.exists(where["path"]):
        return []
    with open(where["path"], "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def test_a_committed_spill_is_journalled_and_the_replay_offers_that_identity(
        channel):
    """THE WIRING, from the engine-facing door to the index a restore reads.

    Before this, `record_spill` had no caller: the journal was never written, so the
    restore index was starved no matter how many files landed. The assertions run in
    the order the failure would appear: the spill's own receipt, the line on disk,
    then what the replay makes of it.
    """
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is True and result["journalled"] is True
    assert result["group_id"] == GROUP
    assert disk_spill.counters()["journal_records"] == 1
    assert disk_spill.counters()["journal_failures"] == 0

    lines = journal_lines()
    assert len(lines) == 1
    seq, op, key_text, size_text, group_text = lines[0].split(" ")
    assert (seq, op) == ("1", "+")
    assert key_text == KEY16 and size_text == str(SLOT_BYTES)
    assert group_text == str(GROUP)

    refreshed = disk_index.refresh(force=True)
    assert refreshed["ok"] is True and refreshed["entries"] == 1
    found = disk_index.lookup(KEY)
    assert found["present"] is True
    assert found["bytes"] == SLOT_BYTES and found["group_id"] == GROUP
    assert disk_index.contains(KEY) is True


def test_a_second_spill_of_the_same_identity_adds_no_second_record(channel):
    """A duplicate did not land, so there is nothing new to say.

    The file was already recorded - possibly by an earlier boot whose journal this
    one continues - and appending a second `+` for it would make the index count one
    file's bytes twice.
    """
    assert disk_spill.spill_slot(caches(), 1, KEY)["journalled"] is True
    second = disk_spill.spill_slot(caches(), 1, KEY)
    assert second["duplicate"] is True and second["journalled"] is False
    assert disk_spill.counters()["journal_records"] == 1
    assert len(journal_lines()) == 1


def test_a_write_that_never_landed_is_never_journalled(monkeypatch, channel):
    """The record follows the `os.replace`, never precedes it.

    A journal that says a file exists when it does not is the ONE corruption this
    index cannot survive: the scheduler would offer a restore that fails on every
    attempt. So the append happens after the commit, and a write that died at the
    rename leaves the journal as it found it - empty.
    """
    def fail_replace(a, b, **kwargs):
        raise OSError("the rename blew up")

    monkeypatch.setattr(os, "replace", fail_replace)
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is False
    assert journal_lines() == []
    assert disk_spill.counters()["journal_records"] == 0
    assert disk_index.refresh(force=True)["entries"] == 0


def test_a_journal_that_raises_cannot_fail_the_spill(channel, monkeypatch):
    """The bytes are the product; the index is a report about them.

    `disk_index` catches its own exceptions, so a raise here means a future edit
    broke that rule - and the spill must STILL succeed, because the alternative is a
    diagnostic costing the tier its whole purpose. The failure is counted and named
    rather than swallowed.
    """
    def boom(*args, **kwargs):
        raise ZeroDivisionError("a future edit divided by zero in the journal")

    monkeypatch.setattr(disk_index, "record_spill", boom)
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is True
    assert result["journalled"] is False
    assert os.path.exists(result["path"])
    counters = disk_spill.counters()
    assert counters["journal_records"] == 0
    assert counters["journal_failures"] == 1
    assert counters["last_error"] == "ZeroDivisionError"


def test_a_journal_that_refuses_cannot_fail_the_spill(channel, monkeypatch):
    """The same rule for the ordinary case: a refusal (no space, capped file) is
    reported, not propagated. No exception is involved and nothing changes."""
    monkeypatch.setattr(
        disk_index, "record_spill",
        lambda *a, **k: {"ok": False, "reason": "the journal is capped"})
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is True and result["journalled"] is False
    assert disk_spill.counters()["journal_failures"] == 1
    assert "capped" in disk_spill.counters()["last_reason"]
    assert os.path.exists(result["path"])


def test_without_a_diagnostics_channel_the_spill_still_lands(monkeypatch):
    """The journal lives in the diagnostics directory; the tier does not.

    A box can arm the tier and leave diagnostics off, and then EVERY spill fails to
    journal. That is a counted, reported condition - never a reason to stop spilling
    to disk, which is the only thing the tier was asked to do.
    """
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    result = disk_spill.spill_slot(caches(), 1, KEY)
    assert result["ok"] is True
    assert result["journalled"] is False
    assert os.path.exists(result["path"])
    assert disk_spill.counters()["journal_failures"] == 1
    assert disk_spill.counters()["journal_records"] == 0


def test_an_evicted_file_is_journalled_as_a_removal_and_stops_being_offered(
        channel):
    """Capacity is the other half of the journal, and it is the dangerous half.

    A file deleted by the ceiling plan with its record still in the index is a
    restore that cannot be served. So the unlink is journalled too - and the entry
    the replay builds from the journal is gone, not merely stale.
    """
    written = []
    for index in range(3):
        result = disk_spill.spill_slot(caches(), index, full_key(index))
        assert result["ok"] is True
        written.append(Path(result["path"]))
    for offset, path in enumerate(reversed(written)):
        stamp = 1_600_000_000 + offset * 100
        os.utime(path, (stamp, stamp))
    disk_index.refresh(force=True)
    assert disk_index.counters()["entries"] == 3
    oldest = written[-1]
    disk_spill._state["ceiling_bytes"] = SLOT_BYTES + SLOT_BYTES // 2
    disk_spill._state["estimated_bytes"] = 3 * SLOT_BYTES
    disk_spill._enforce_ceiling()
    assert not oldest.exists()

    lines = journal_lines()
    removals = [line for line in lines if line.split(" ")[1] == "-"]
    assert removals, lines
    key16 = disk_layout.parse_path(oldest.name)["key16"]
    group = disk_layout.parse_path(oldest.name)["group_id"]
    assert removals[0].split(" ")[2] == key16
    assert removals[0].split(" ")[4] == str(group)
    assert disk_spill.counters()["journal_removals"] >= 1

    refreshed = disk_index.refresh(force=True)
    assert refreshed["removals"] >= 1
    assert disk_index.lookup(full_key(2))["present"] is False
    # The survivors are still offered: a removal journalled badly would be the one
    # bug that takes the whole index down with it.
    assert disk_index.lookup(full_key(0))["present"] is True


def test_the_journal_counters_reach_the_payloads_the_dashboard_reads(channel):
    """A counter nobody publishes is a counter that does not exist at 3 a.m.

    The spill's own section, the worker's contribution seam and the snapshot payload
    all have to carry the journal's state, because "the tier is writing but the
    index is empty" is a question that can only be answered from a payload.
    """
    disk_spill.spill_slot(caches(), 1, KEY)
    section = disk_spill.section()
    assert json.dumps(section, default=str)
    assert section["journal_records"] == 1
    assert section["journal_failures"] == 0
    counters = disk_spill.counters()
    assert counters["journal_records"] == 1
    assert disk_index.section()["journal_writes"] == 1
