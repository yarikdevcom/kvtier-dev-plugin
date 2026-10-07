r"""The RESTORE-side verifier: do the bytes in the GPU window match the file?

`disk_restore_verify` answers one question about a sampled restored entry - is
`file[0:W_g]` the same as `gpu_window[gpu_block_id][0:W_g]`? - and the ordering of this
file is the ordering of the ways that answer could be worthless:

1. OFF IS OFF. The default posture opens no file, copies no device memory, starts no
   thread and moves no counter. "Off" is proved by seams that COUNT attempts, not by a
   counter agreeing with itself.
2. THE COMPARISON IS OVER THE RIGHT BYTES. `W_g` is the GROUP's width and the rest of
the
   window is other groups' stale bytes by construction, so a difference at or past `W_g`
   must NOT be a mismatch, a difference inside it must be, and a destination one slot
   off must fail rather than pass. Those three together are the reason the tail is
   refused: a verifier that compared the whole window would report a mismatch on every
   narrow group forever, and the one real finding would drown in it.
3. A DISAGREEMENT IS CLASSIFIED, NOT PROCLAIMED. A GPU block is handed out again at any
   time, so a re-read that differs is `mismatch_gpu_changed_on_recheck` (the engine
   moved, an expected race) and only a re-read that held still is `mismatch_stable` (a
   layout candidate, the only one an operator acts on).
4. THE REFUSALS REFUSE. A file that is not window-wide, a group wider than the window, a
   GPU whose widths disagree, a worker with no completion watermark: each is a named
   counter and NOT a comparison, because comparing bytes under a geometry that cannot be
   right manufactures findings.
5. THE BINDING ORACLE IS EITHER RUN OR REPORTED AS UNREACHABLE. It re-derives the file
   name from `block_hashes` through the engine's own helpers, so it can catch a one-slot
   shift that name-vs-name cannot. A worker has no requests, so in production it counts
   `oracle_unavailable` - counted and published, never a silent skip.
6. THE TRUNCATION SHAPE IS COUNTED. A batch that restored the positional groups and
   nothing for an align-mode group lost that group's only real entry to the fork's
   `n_take_g`.
7. IT COSTS NOTHING IT DID NOT SAY. The step thread pays a counter and a dict of ints:
   proved with the file reader and the GPU reader both instrumented. The per-step bound,
   the queue bound, the sampling rate and the interval floor each do their job, and the
   floor counts rather than sleeps. The thread is one, lazy, and stopped by a reset.
8. THE VIEW NEVER CHANGES SHAPE, and the section fits the contribution seam's cap - with
   FIVE sections now registered, the rank file's required fields survive it.

WHAT IS FAKE, STATED WHERE IT CANNOT BE MISSED. There is no CUDA device here, so the
device->host copy is the one expression replaced (`disk_shadow._copy_slot_to_host`, the
seam this module deliberately REUSES rather than re-implementing), and the GPU region is
real ctypes memory shaped like the fork's tensor. Everything else - the sorted-name
order rule, the width check, the ctypes read, the file reads, the sampler, the throttle,
the re-read classification, the key derivation - is the shipped code running. Two tests
drive the REAL restore path (`disk_restore.worker_bind` then `worker_before_load`) so
the hook site itself is pinned, not assumed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

import pytest
from test_disk_restore import (
    DIGEST,
    WIDTH,
    Meta,
    Worker,
    key_hex,
    pack,
    resolve,
)
from test_disk_shadow import CopySeam, FakeGpuRegion

from kvtier_dev_plugin import (  # noqa: E402 - the modules under test
    backport,
    config,
    disk_index,
    disk_layout,
    disk_restore,
    disk_restore_verify,
    disk_shadow,
    disk_spill,
    disk_store,
    endpoint,
    worker_diag,
)

GPU_BLOCKS = 8
# Per-group whole-block bytes for a five-group config, widest last-but-one: group 3 is a
# narrow scratch group, so the window (4096 B, the rank's slot) is wider than its 1024 B
# and the tail-refusal is a live property of every entry this file tests.
GROUP_WIDTHS = (4096, 4096, 2048, 1024, 4096)
ALIGN_GROUP = 2                    # the group whose spec says `mamba_cache_mode=align`
KEY16 = "a36ec46fe316bd0d"         # the name a real shard directory carried


# --- the worker's side of the round trip --------------------------------------
class Spec:
    """One `kv_cache_spec`, as far as the geometry reads reach.

    `page_size_bytes` is the number `W_g` comes from and `mamba_cache_mode` the one the
    align-mode classification reads - the two fields this module can evaluate on a
    worker, where there is no manager to ask.
    """

    def __init__(self, page_size_bytes, mamba_cache_mode="none") -> None:
        self.page_size_bytes = page_size_bytes
        self.mamba_cache_mode = mamba_cache_mode


class Group:
    def __init__(self, spec, layers=("layer-a",)) -> None:
        self.kv_cache_spec = spec
        self.layer_names = layers


class CacheConfig:
    """`worker.kv_cache_config`: the groups, in group order."""

    def __init__(self, widths, align=()) -> None:
        self.kv_cache_groups = [
            Group(Spec(width, "align" if index in align else "none"))
            for index, width in enumerate(widths)]


class VerifierWorker(Worker):
    """`SimpleCPUOffloadWorker` plus the three fields only a verifier touches.

    It IS `test_disk_restore.Worker` (so the CPU side, the bind and the apply point are
    the same fake that suite pins) with `gpu_kv_caches`, `kv_cache_config` and the
    fork's `_load_hwm` watermark added.

    `load_hwm` DEFAULTS TO A HIGH VALUE, and that is a statement about the test, not
    about the product: the watermark is the gate that says "the fork's DMA has landed",
    it is pinned by its own two tests, and every other test here would otherwise be
    asserting the gate instead of the comparison. A test that wants the wait says
    `load_hwm=-1`.
    """

    def __init__(self, *, widths=GROUP_WIDTHS, align=(ALIGN_GROUP,), gpu_blocks=None,
                 gpu_width=WIDTH, load_hwm=1000) -> None:
        super().__init__()
        self.gpu_kv_caches = {"g0": FakeGpuRegion(gpu_blocks or GPU_BLOCKS, gpu_width)}
        self.kv_cache_config = CacheConfig(widths, align)
        self._load_hwm = load_hwm

    @property
    def gpu(self):
        return self.gpu_kv_caches["g0"]

    def complete(self, event) -> None:
        """The copy stream reports this load event's CUDA event as done."""
        self._load_hwm = max(self._load_hwm, int(event))


def job(path, *, group=0, gpu_block=5, cpu_block=3, key16=KEY16, width=WIDTH) -> dict:
    """One retained job, EXACTLY the shape `_worker_bind` stores.

    Field names are not a convenience here: the verifier reads `path`, `key16`, `group`,
    `cpu_block_id`, `gpu_block_id` and `bytes` out of the apply point's own dict, so a
    fake that renamed one would test a shape the product never produces.
    """
    return {"path": path, "key16": key16, "group": group, "cpu_block_id": cpu_block,
            "gpu_block_id": gpu_block, "bytes": width}


def batch_of(jobs, event=11) -> dict:
    """A bound batch holding exactly `jobs`, in bind order."""
    return {"batch_id": event, "jobs": {index: item for index, item in enumerate(jobs)},
            "failures": {}, "event": event, "req_ids": ["req-1"],
            "deadline_mono": time.monotonic()}


def write_window(path: str, head: bytes, *, size=WIDTH) -> str:
    """A spill file: `head` at the front, a distinct filler behind it, `size` bytes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = bytes((offset * 7 + 1) % 251 for offset in range(size))
    payload = bytearray(body)
    payload[:len(head)] = head
    with open(path, "wb") as handle:
        handle.write(bytes(payload))
    return path


def window_file(tmp_path, name="kvslot-window", *, head=None, size=WIDTH) -> str:
    """A window-wide file under this test's store directory."""
    head = bytes((offset * 3 + 5) % 251 for offset in range(64)) if head is None \
        else head
    return write_window(str(tmp_path / "store" / name), head, size=size)


def gpu_matches(gpu, block: int, path: str, *, tail=None) -> None:
    """Put the file's window into a GPU block, optionally with a different tail.

    The tail is the part this verifier must never look at, so a test that wants "equal"
    while the tail disagrees is a test that the refusal is real.
    """
    with open(path, "rb") as handle:
        raw = bytearray(handle.read())
    if tail is not None:
        raw[len(raw) - len(tail):] = tail
    gpu.set_block(block, bytes(raw))


def one_pass(worker):
    """One bounded pass, with the interval floor lifted.

    The floor is pinned by its OWN test; every other test that reaches the second pass
    in a row would otherwise be asserting the throttle instead of the comparison.
    Touching the private clock is the honest way to say "the floor is not what this test
    is about".
    """
    disk_restore_verify._state["last_pass_mono"] = None
    return disk_restore_verify.run_pass(worker)


def waiting_worker(**kwargs) -> VerifierWorker:
    """A worker whose copy has NOT landed (`_load_hwm` at a fresh worker's -1).

    Named rather than inlined because two tests need it and a bare `load_hwm=-1` in a
    call site says nothing about why the number matters.
    """
    return VerifierWorker(load_hwm=-1, **kwargs)


def counters():
    return disk_restore_verify.counters()


# --- fixtures -----------------------------------------------------------------
@pytest.fixture
def seam(monkeypatch):
    """The device->host seam, instrumented, for the GPU the test worker owns.

    Patched on `disk_shadow`, NOT on this module: the verifier calls through
    `disk_shadow._read_gpu_slot`, so redirecting the seam is what proves the two
    witnesses share ONE device-copy mechanism. A test that patched this module's own
    read would be testing that a fake calls a fake.

    Calling it installs (once per test) and hands back the seam, so the test can read
    `.calls` afterwards - the ORDER and BLOCK of every device read is the observable
    most of these tests assert on.
    """
    holder: dict = {}

    def install(caches):
        if "seam" in holder:
            return holder["seam"]
        local = CopySeam(caches)
        monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", local)
        holder["seam"] = local
        return local

    return install


@pytest.fixture
def worker(request):
    """A worker with a GPU region, five groups, one of them align-mode."""
    kwargs = getattr(request, "param", {}) or {}
    return VerifierWorker(**kwargs)


@pytest.fixture
def armed_env(monkeypatch):
    """The environment's own yes. DEFAULT OFF is the posture; this is the opt-in.

    Set as an environment variable rather than through `set_enabled`, because the env
    path is what arms a WORKER (the process that applies a restore) and the POST latch
    is the process that answers - so every comparison test here runs the way a worker
    would.
    """
    monkeypatch.setenv(config.ENV_DISK_RESTORE_VERIFY, "1")
    return True


@pytest.fixture
def no_thread(monkeypatch):
    """`_ensure_thread` says yes without starting a thread, so a pass is deterministic.

    The `disk_gpu_shadow` idiom, for the same reason: with a live thread the test and
    the thread race over one retained entry, and the assertion that fails is whichever
    lost. The thread's own lifecycle has its own test.
    """
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread", lambda: True)
    return


@pytest.fixture
def armed(armed_env, no_thread, monkeypatch, tmp_path):
    """Armed verifier, patched seam, and a store directory. The comparison stage."""
    root = tmp_path / "spill"
    root.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    return root


@pytest.fixture(autouse=True)
def sample_every_entry(monkeypatch):
    """Every test retains every applied entry, unless it is the test ABOUT the rate.

    Not a convenience for its own sake: the rate is 32, so within one test only the very
    first applied entry would be sampled and a second batch would be silently
    `sampled_out` - which turns "the verifier refused to compare these bytes" into "the
    verifier never looked". That confusion is exactly what this file must not produce,
    so the rate is pinned by ONE test
    (`test_the_sample_rate_is_a_rate_over_applied_entries`) and everything else runs
    with the sampler open.
    """
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 1)
    return


@pytest.fixture
def key_helpers(monkeypatch):
    """`vllm.v1.core.kv_cache_utils` with the two helpers the oracle needs.

    The fork's real ones are `make_block_hash_with_group_id` (a digest plus a 4-byte
    group id) and `resolve_block_hashes` (one hash per whole group block). These are the
    same two reimplementations `test_disk_restore.py` uses, imported rather than copied,
    so the oracle is checked against the SAME arithmetic the offer path is checked
    against - a second reimplementation here could agree with a bug.
    """
    module = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    module.make_block_hash_with_group_id = pack
    module.resolve_block_hashes = resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, module)
    return module


# --- 1: off is off -------------------------------------------------------------
def test_default_off_opens_no_file_reads_no_gpu_and_moves_no_counter(tmp_path,
                                                                     monkeypatch):
    """Unset environment: the apply point is a counter-free no-op and no thread starts.

    Every proof here is a SEAM THAT COUNTS, not a counter agreeing with itself: the file
    reader and the GPU reader both raise if reached, so "nothing was read" is a fact
    about the code, not about this module reporting its own silence.
    """
    def refused(*_args, **_kwargs):
        raise AssertionError("the disabled verifier touched a file or the device")

    monkeypatch.setattr(disk_restore_verify, "_read_file_head", refused)
    monkeypatch.setattr(disk_shadow, "_read_gpu_slot", refused)
    target = VerifierWorker()
    report = disk_restore_verify.observe_apply(
        target, batch_of([job(window_file(tmp_path))]), {})
    assert report["reason"] == "disabled" and report["retained"] == 0
    assert counters() == {key: 0 for key in disk_restore_verify._COUNTER_KEYS}
    assert one_pass(target)["reason"] == "disabled"
    assert disk_restore_verify.view()["pending"] == {}
    status = disk_restore_verify.status()
    assert status["enabled"] is False
    assert status["env_var"] == config.ENV_DISK_RESTORE_VERIFY
    # The DISABLED STATE IS NAMED, not merely empty: a reader must be able to tell
    # "nobody turned it on" from "it is on and found nothing".
    assert status["enable_source"].startswith("default")
    assert status["thread_alive"] is False
    assert status["sample_every_n_restores"] == disk_restore_verify \
        .SAMPLE_EVERY_N_RESTORES


def test_an_unrecognised_env_value_leaves_it_off_and_says_so(monkeypatch):
    """A typo moves the flag in the direction that reads nothing, and is reported."""
    monkeypatch.setenv(config.ENV_DISK_RESTORE_VERIFY, "maybe")
    state = disk_restore_verify.enable_state()
    assert state["enabled"] is False and state["env_recognised"] is False
    assert state["env_raw"] == "maybe"
    disk_restore_verify._resolve_env()
    assert any("env_value_unrecognised" in line
               for line in disk_restore_verify.view()["reasons"])


def test_the_post_latch_arms_only_this_process_and_says_its_scope():
    """`set_enabled` is in-process and writes nothing; `scope` names the env.

    This matters more here than for `shadow`: the apply point is worker code, so a POST
    at the EngineCore latches a process that can never verify anything.
    """
    answer = disk_restore_verify.set_enabled(True)
    assert answer["enabled"] is True and answer["latch"] is True
    assert answer["source"] == "post op"
    assert config.ENV_DISK_RESTORE_VERIFY in answer["scope"]
    assert disk_restore_verify.effective_enabled() is True
    assert disk_restore_verify.set_enabled(False)["enabled"] is False


# --- 2: the comparison is over the right bytes --------------------------------
def test_a_matching_head_is_equal_and_the_tail_is_never_compared(armed, worker, seam,
                                                                 tmp_path):
    """The widest group's own width IS the window: `equal`, and nothing is excluded.

    The baseline the narrow case is measured against. `tail_bytes_excluded` is 0 here on
    purpose - group 0's `W_g` is the whole 4096 B slot, so there is no span this test
    could be quietly skipping, and the comparison below covers every byte of the file.
    """
    local = seam(worker.gpu_kv_caches)
    path = window_file(tmp_path, head=b"HEAD" * 8)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path, group=0)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["equal"] is True and verdict["bytes"] == WIDTH
    assert verdict["tail_bytes_excluded"] == 0
    assert counters()["checked"] == 1 and counters()["equal"] == 1
    assert counters()["mismatch"] == 0
    assert local.calls == [("g0", 5)]              # one block, the right one


def test_a_narrow_groups_equal_case_excludes_the_stale_tail(armed, worker, seam,
                                                            tmp_path):
    """Group 3's own width is 1024 B of a 4096 B window: only those 1024 B are compared.

    The GPU's bytes 1024..4096 are deliberately the file's bytes shifted by one - the
    shape of "the other groups left their state here" - and the verdict must still be
    `equal`, with `tail_bytes_excluded` naming the 3072 B nobody claimed.
    """
    seam(worker.gpu_kv_caches)
    head = bytes((offset + 1) % 251 for offset in range(1024))
    path = window_file(tmp_path, head=head)
    with open(path, "rb") as handle:
        window = handle.read()
    # Same first 1024 bytes, EVERY byte past them different (reversed): a compare that
    # reached past W_g would call this a mismatch at byte 1024.
    worker.gpu.set_block(4, bytes(window[:1024]) + bytes(reversed(window[1024:])))
    disk_restore_verify.observe_apply(worker, batch_of([job(path, group=3,
                                                             gpu_block=4)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["equal"] is True
    assert verdict["bytes"] == 1024
    assert verdict["tail_bytes_excluded"] == WIDTH - 1024
    assert counters()["equal"] == 1 and counters()["mismatch"] == 0


def test_a_flip_inside_the_group_width_is_a_stable_mismatch(armed, worker, seam,
                                                            tmp_path):
    """One byte inside `[0, W_g)` differs, the GPU holds still: `mismatch_stable`.

    This is the only category an operator acts on, so it is asserted by name and by the
    offset: `first_difference` must point at the flipped byte, and the re-read must have
    happened (one extra seam call), because a mismatch nobody re-read is unclassifiable.
    """
    local = seam(worker.gpu_kv_caches)
    head = b"\x10" * 64
    path = window_file(tmp_path, head=head)
    gpu_matches(worker.gpu, 5, path)
    flipped = head[:40] + b"\x11" + head[41:]
    worker.gpu.set_block(5, bytes(flipped).ljust(WIDTH, b"\x00"))
    disk_restore_verify.observe_apply(worker, batch_of([job(path, group=0)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["equal"] is False
    assert verdict["category"] == "mismatch_stable"
    assert verdict["first_difference"] == 40
    assert counters()["mismatch_stable"] == 1
    assert counters()["mismatch_gpu_changed_on_recheck"] == 0
    assert counters()["rechecks"] == 1
    assert len(local.calls) == 2                    # the read AND the re-read
    last = disk_restore_verify.view()["last_mismatch"]
    assert last["compared_bytes"] == WIDTH and last["gpu_block"] == 5


def test_a_destination_one_slot_off_fails_rather_than_passing(armed, worker, seam,
                                                              tmp_path):
    """THE finding this module exists for: the file's bytes are in the NEXT block over.

    A one-slot shift moves the index-aligned lists together, so every name stays
    well-formed and no other check in the stack notices. Here the verifier reads the
    block the restore path NAMED, finds the file's head one block away, and reports a
    stable mismatch with an offset - it cannot be talked into `equal`.
    """
    seam(worker.gpu_kv_caches)
    head = bytes((offset * 5) % 251 for offset in range(256))
    path = window_file(tmp_path, head=head)
    gpu_matches(worker.gpu, 6, path)                # the right bytes, the wrong block
    disk_restore_verify.observe_apply(worker,
                                      batch_of([job(path, gpu_block=5)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["equal"] is False
    assert verdict["category"] == "mismatch_stable"
    assert verdict["first_difference"] is not None
    assert counters()["equal"] == 0 and counters()["mismatch_stable"] == 1


def test_a_reused_destination_block_is_classified_on_recheck(armed, worker, seam,
                                                             tmp_path):
    """The engine rewrites the block between the two reads: `_on_recheck`, NOT stable.

    A GPU block is handed out again at any time, so this is an expected race, not
    corruption. The seam's `churn_on_call` is what makes the GPU move on cue: the second
    call (the re-read) sees different bytes, so the ONLY honest verdict is that the GPU
    changed under us. Collapsing this into `mismatch_stable` would make the number an
    operator acts on the number he cannot act on.
    """
    local = seam(worker.gpu_kv_caches)
    local.churn_on_call = 2
    path = window_file(tmp_path, head=b"FILEHEAD" * 4)
    gpu_matches(worker.gpu, 5, path)
    worker.gpu.set_block(5, b"\x00" * WIDTH)        # so the first read mismatches
    disk_restore_verify.observe_apply(worker, batch_of([job(path, group=0)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["category"] == "mismatch_gpu_changed_on_recheck"
    assert counters()["mismatch_gpu_changed_on_recheck"] == 1
    assert counters()["mismatch_stable"] == 0
    assert len(local.calls) == 2


def test_a_mismatch_the_recheck_cannot_read_is_unrechecked(armed, worker, seam,
                                                           monkeypatch, tmp_path):
    """A device fault on the re-read is its own category, never a guessed one.

    `mismatch_unrechecked` says "we saw a disagreement and could not classify it". It
    must not become `mismatch_stable` (which would claim the GPU held still) and must
    not vanish. The seam here is hand-written to answer the FIRST call and fault the
    second, because the thing being pinned is exactly "which call failed".
    """
    path = window_file(tmp_path, head=b"NOPE" * 8)
    gpu_matches(worker.gpu, 5, path)
    worker.gpu.set_block(5, b"\x00" * WIDTH)
    inner = seam(worker.gpu_kv_caches)          # the seam is the "real" device read
    calls: list = []

    def flaky(tensor, index):
        calls.append(int(index))
        if len(calls) > 1:
            raise RuntimeError("the device faulted on the re-read")
        return inner(tensor, index)

    monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", flaky)
    disk_restore_verify.observe_apply(worker, batch_of([job(path, group=0)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["category"] == "mismatch_unrechecked"
    assert counters()["mismatch_unrechecked"] == 1
    assert counters()["mismatch_stable"] == 0
    assert counters()["mismatch_gpu_changed_on_recheck"] == 0
    assert calls == [5, 5]      # the SECOND call is the re-read, and it is the one that
                                # faulted: the first read is what produced the mismatch
    assert any("recheck:gpu_read_failed" in line
               for line in disk_restore_verify.view()["reasons"])


# --- 3: the refusals refuse ---------------------------------------------------
def test_a_file_that_is_not_window_wide_is_refused_never_compared(armed, worker, seam,
                                                                  tmp_path):
    """Assertion 1. A 12-byte file (or a double-wide one) is a geometry answer.

    Both widths go in the reason, because "the file is 4096 B and this rank's window is
    12,902,400 B" is the sentence that names a PP-crossing bug; and NOTHING is compared,
    so `checked` stays 0 and the seam is never called.
    """
    local = seam(worker.gpu_kv_caches)
    short = window_file(tmp_path, head=b"T", size=12)
    disk_restore_verify.observe_apply(worker, batch_of([job(short)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["reason"] == "file_not_window_wide"
    assert "12 B" in verdict["detail"] and "4096 B" in verdict["detail"]
    assert counters()["file_not_window_wide"] == 1
    assert counters()["checked"] == 0 and local.calls == []

    wide = window_file(tmp_path, name="wide", head=b"T", size=WIDTH * 2)
    disk_restore_verify.observe_apply(worker, batch_of([job(wide)]), {})
    assert one_pass(worker)["results"][0]["reason"] == "file_not_window_wide"
    assert counters()["file_not_window_wide"] == 2 and counters()["checked"] == 0


def test_a_group_wider_than_the_window_refuses_without_comparing(armed, seam, tmp_path):
    """Assertion 2, the geometry contradiction: W_g > file_size.

    Reading past EOF and calling the zeros a mismatch is the failure this refuses. The
    group's width comes from its own spec, so the fake worker declares group 3 at twice
    the slot width.
    """
    wide_group = VerifierWorker(widths=(4096, 4096, 2048, 8192, 4096))
    local = seam(wide_group.gpu_kv_caches)
    path = window_file(tmp_path)
    disk_restore_verify._state["last_pass_mono"] = None
    disk_restore_verify.observe_apply(wide_group,
                                      batch_of([job(path, group=3, width=WIDTH)]), {})
    verdict = one_pass(wide_group)["results"][0]
    assert verdict["reason"] == "group_wider_than_file"
    assert counters()["group_wider_than_file"] == 1
    assert counters()["checked"] == 0 and local.calls == []


def test_a_group_with_no_readable_width_is_named_not_guessed(armed, seam, tmp_path):
    """No `page_size_bytes`, no W_g - and without W_g there is no correct compare.

    Guessing the window width here would compare the tail, which is the exact thing this
    module's central assertion forbids, so it is a counted refusal with a reason.
    """
    blind = VerifierWorker()
    blind.kv_cache_config = CacheConfig(())         # no groups at all
    local = seam(blind.gpu_kv_caches)
    path = window_file(tmp_path)
    disk_restore_verify.observe_apply(blind, batch_of([job(path)]), {})
    assert one_pass(blind)["reason"] == "nothing retained"
    # Nothing was retained at all, so the per-entry width counter never moved: the
    # refusal is the one `geometry_unavailable`, counted once for the whole config.
    assert counters()["no_group_width"] == 0
    assert counters()["geometry_unavailable"] == 1
    assert local.calls == []
    assert any("group_widths_unavailable" in line
               for line in disk_restore_verify.view()["reasons"])


def test_a_missing_file_is_its_own_answer(armed, worker, seam, tmp_path):
    """The LRU can evict between the read and the verification. Not a mismatch."""
    seam(worker.gpu_kv_caches)
    gone = str(tmp_path / "store" / "kvslot-gone")
    disk_restore_verify.observe_apply(worker, batch_of([job(gone)]), {})
    assert one_pass(worker)["results"][0]["reason"] == "file_missing"
    assert counters()["file_missing"] == 1 and counters()["mismatch"] == 0


def test_a_device_fault_is_a_counter_and_never_an_exception(armed, worker, monkeypatch,
                                                            tmp_path):
    """No driver, a refused copy, a bad block: one counted category, no raise.

    The pass runs on this module's thread; an escaping exception there would stop the
    thread silently and every later entry would pile up until the queue dropped them -
    which reads as "the verifier found nothing".
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("no CUDA driver in this process")

    monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", explode)
    path = window_file(tmp_path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["reason"] == "gpu_read_failed"
    assert counters()["gpu_read_failed"] == 1 and counters()["errors"] == 0
    assert disk_restore_verify.status()["thread_alive"] is False


def test_a_gpu_width_disagreement_is_width_mismatch_not_staleness(armed, worker, seam,
                                                                   tmp_path):
    """The GPU's regions do not add up to the arena's window: a LAYOUT answer.

    Counting it as a device failure (or as a mismatch) would hide a real fault behind a
    number an operator has learned to ignore, so `disk_shadow`'s own
    `region_width_mismatch` is carried through as its own counter.
    """
    narrow = VerifierWorker(gpu_width=WIDTH // 2)
    seam(narrow.gpu_kv_caches)
    path = window_file(tmp_path)
    disk_restore_verify.observe_apply(narrow, batch_of([job(path)]), {})
    verdict = one_pass(narrow)["results"][0]
    assert verdict["reason"] == "width_mismatch"
    assert counters()["width_mismatch"] == 1
    assert counters()["mismatch"] == 0 and counters()["gpu_read_failed"] == 0


def test_a_gpu_block_id_the_regions_do_not_have_is_named(armed, worker, seam, tmp_path):
    """`no_gpu_block`: the id is outside the region, so there are no bytes to read.

    The id came from the scheduler's list, so an id past the end of the arena is a
    list-length finding, not a verifier limitation - and it is refused before any read.
    """
    seam(worker.gpu_kv_caches)
    path = window_file(tmp_path)
    disk_restore_verify.observe_apply(worker,
                                      batch_of([job(path,
                                               gpu_block=GPU_BLOCKS + 9)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["reason"] == "no_gpu_block"
    assert counters()["no_gpu_block"] == 1
    assert counters()["checked"] == 0


def test_a_worker_with_no_completion_watermark_never_compares(armed, seam, tmp_path):
    """No `_load_hwm` means nobody can say the DMA landed, so nothing is ever compared.

    The apply point runs BEFORE the fork's `start_load_kv` issues the copy, so comparing
    at the apply point would read bytes that have not arrived. The watermark is the only
    honest witness, and a worker that does not have one gets a counted reason rather
    than a race.
    """
    bare = waiting_worker()
    del bare._load_hwm
    local = seam(bare.gpu_kv_caches)
    path = window_file(tmp_path)
    gpu_matches(bare.gpu, 5, path)
    disk_restore_verify.observe_apply(bare, batch_of([job(path)]), {})
    assert one_pass(bare)["results"][0]["reason"] == "no_load_hwm"
    assert counters()["no_load_hwm"] == 1 and counters()["checked"] == 0
    assert local.calls == []


def test_an_entry_waits_for_its_dma_and_then_verifies(armed, seam, tmp_path):
    """The gate is `worker._load_hwm >= event`: not yet, then equal.

    The same retained entry, the same file, the same GPU bytes - the ONLY change is that
    the fork's watermark moved past the event. That is the proof the wait is the gate.
    """
    worker = waiting_worker()
    local = seam(worker.gpu_kv_caches)
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)], event=11), {})
    first = one_pass(worker)
    assert first["waiting"] == 1 and first["checked"] == 0
    assert local.calls == []
    assert disk_restore_verify.view()["pending"]      # still retained
    worker.complete(11)
    assert one_pass(worker)["results"][0]["equal"] is True


def test_an_entry_whose_dma_never_completes_is_dropped_as_stale(armed, seam,
                                                                monkeypatch,
                                                                tmp_path):
    """A load event that never lands is `dropped_waiting_dma`, not an endless queue.

    The age bound is monkeypatched to nothing rather than waited on: the point is the
    outcome, and a test that slept 30 s to reach it would be a slow test, not a better
    one.
    """
    worker = waiting_worker()
    seam(worker.gpu_kv_caches)
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)], event=11), {})
    assert one_pass(worker)["waiting"] == 1
    monkeypatch.setattr(disk_restore_verify, "JOB_MAX_AGE_SECONDS", -1.0)
    assert one_pass(worker)["results"][0]["reason"] == "dropped_waiting_dma"
    assert counters()["dropped_waiting_dma"] == 1
    assert counters()["checked"] == 0
    assert disk_restore_verify.view()["pending"] == {}


# --- 4: the binding oracle ----------------------------------------------------
def _oracle_entry(group=0, index=0, hash_block=128, group_size=128):
    """A provider answering with the engine's OWN inputs for one entry.

    `group_size == hash_block_size` keeps `resolve_block_hashes` at one hash per entry
    (step 1), so an index in the resolved view means exactly what the provider says it
    means and the shift test below shifts by one block, not by a step.
    """
    hashes = [bytes([(seed + i) % 251 for i in range(32)])
              for seed in range(1, 30)]
    return {"block_hashes": hashes, "hash_block_size": hash_block,
            "group_size": group_size, "index": index}


def test_the_oracle_confirms_a_name_it_can_re_derive(armed, worker, seam, key_helpers,
                                                     monkeypatch, tmp_path):
    """With a provider installed, the name is re-derived from `block_hashes` and agrees.

    The expected `key16` is built here the way the oracle builds it - the engine's
    `resolve_block_hashes` view, then `make_block_hash_with_group_id`, then the ONE
    place this package reads the group suffix - so the assertion is about agreement, not
    about the oracle agreeing with itself.
    """
    seam(worker.gpu_kv_caches)
    from kvtier_dev_plugin import disk_layout
    source = _oracle_entry(group=0, index=2)
    view = resolve(source["block_hashes"], source["hash_block_size"],
                   source["group_size"])
    named = disk_layout.split_block_key(bytes(pack(view[2], 0)).hex())
    assert named["ok"]
    disk_restore_verify.set_oracle(lambda entry: dict(source), note="test provider")
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(
        worker, batch_of([job(path, key16=named["key16"])]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["oracle"]["state"] == "oracle_equal"
    state = disk_restore_verify.oracle_state()
    assert state["provider"] is True and state["checked"] == 1 and state["equal"] == 1
    assert state["mismatch"] == 0
    disk_restore_verify.set_oracle(None)


def test_the_oracle_catches_a_shifted_index_that_name_vs_name_cannot(armed, worker,
                                                                     seam, key_helpers,
                                                                     tmp_path):
    """THE reason the oracle is not name-vs-name: a shift moves the DERIVED key.

    The file is named for block N; the entry's GPU index is the one the fork used for
    block N+1. Both strings are well-formed hex and a name-vs-name check passes, because
    a one-slot shift moves the index-aligned lists together. Re-derived from
    `block_hashes` at the index actually used, the key differs - and that is the whole
    argument for the oracle existing.
    """
    seam(worker.gpu_kv_caches)
    from kvtier_dev_plugin import disk_layout
    source = _oracle_entry(group=0, index=3)
    view = resolve(source["block_hashes"], source["hash_block_size"],
                   source["group_size"])
    # The name in the path is for index 3; the fork's GPU index resolves to index 4.
    named = disk_layout.split_block_key(bytes(pack(view[3], 0)).hex())
    shifted = dict(source, index=4)
    disk_restore_verify.set_oracle(lambda entry: dict(shifted))
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(
        worker, batch_of([job(path, key16=named["key16"])]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["oracle"]["state"] == "oracle_mismatch"
    assert verdict["oracle"]["expected_key16"] != named["key16"]
    state = disk_restore_verify.oracle_state()
    assert state["mismatch"] == 1 and state["equal"] == 0
    assert state["last"]["name_key16"] == named["key16"]
    assert state["last"]["expected_key16"] == verdict["oracle"]["expected_key16"]
    disk_restore_verify.set_oracle(None)


def test_an_unreachable_block_hashes_counts_oracle_unavailable(armed, worker, seam,
                                                               tmp_path):
    """Production on this fork: a worker has no requests, so the oracle is UNREACHABLE.

    It is not skipped and not silent. Every verified entry counts `oracle_unavailable`,
    the published reason names the fork's own statement of why, and `set_oracle` is
    named as the thing that would change that. A verifier that quietly omitted the
    binding check would leave an operator believing it had run.
    """
    seam(worker.gpu_kv_caches)
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    verdict = one_pass(worker)["results"][0]
    assert verdict["oracle"]["state"] == "oracle_unavailable"
    state = disk_restore_verify.oracle_state()
    assert state["provider"] is False and state["reachable"] is False
    assert state["unavailable"] == 1 and state["checked"] == 0
    assert "metadata.py" in state["unreachable_reason"]
    assert "request identities" in state["unreachable_reason"]


def test_an_oracle_that_answers_badly_is_refused_not_skipped(armed, worker, seam,
                                                             key_helpers, tmp_path):
    """A provider is not a promise. A raise or a malformed answer is `oracle_refused`.

    Three shapes are exercised: a provider that raises, one that answers with nothing
    (which IS reachable-but-absent, so `oracle_unavailable`), and one that omits a
    field.
    """
    seam(worker.gpu_kv_caches)
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)

    def boom(entry):
        raise ValueError("the provider cannot read this entry")

    disk_restore_verify.set_oracle(boom)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    assert one_pass(worker)["results"][0]["oracle"]["state"] == "oracle_refused"
    assert counters()["oracle_refused"] == 1

    disk_restore_verify.set_oracle(lambda entry: None)
    disk_restore_verify._state["last_pass_mono"] = None
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    assert one_pass(worker)["results"][0]["oracle"]["state"] == "oracle_unavailable"

    disk_restore_verify.set_oracle(lambda entry: {"block_hashes": []})
    disk_restore_verify._state["last_pass_mono"] = None
    diskrestore = disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    assert diskrestore["retained"] == 1
    assert one_pass(worker)["results"][0]["oracle"]["state"] == "oracle_refused"
    assert counters()["oracle_refused"] == 2
    disk_restore_verify.set_oracle(None)


def test_the_oracle_refuses_to_reimplement_the_engines_keying(armed, key_helpers,
                                                              monkeypatch):
    """No `kv_cache_utils` in `sys.modules` is a refusal, never a guess.

    A plugin that invented its own packing would eventually disagree with the engine
    about which bytes are the group, and that disagreement is a file read under the
    wrong name - the exact failure this oracle exists to catch, so it cannot be allowed
    to cause it.
    """
    monkeypatch.delitem(sys.modules, disk_restore.KV_UTILS_MODULE)
    verdict = disk_restore_verify.oracle_expect_key16(_oracle_entry(), 0)
    assert verdict["ok"] is False
    assert verdict["kind"] == "no_key_helpers"
    missing = disk_restore_verify.oracle_expect_key16({"block_hashes": [b"x"]}, 0)
    assert missing["ok"] is False and missing["kind"] == "oracle_source_shape"


# --- 5: the truncation shape (Gate B) ----------------------------------------
def test_a_batch_that_skipped_the_align_group_is_counted(armed, worker, seam):
    """The Gate B shape: the positional groups restored, the state block did not.

    The fork truncates each group's hit list with `n_take_g = cdiv(num_external_tokens,
    g_block_size)`, and an align-mode group's hit list is `[null]*n-1 + [one real
    block]` - so a truncation that lands before the run's end removes that group's ONLY
    real entry while every positional group keeps its entries. The verifier cannot see
    the missing entry, so it counts its ABSENCE in a batch that restored something.
    """
    disk_restore_verify.observe_apply(
        worker, batch_of([job("/tmp/a", group=0), job("/tmp/b", group=1),
                          job("/tmp/c", group=4)]), {})
    assert counters()["boundary_entry_missing"] == 1
    assert counters()["boundary_groups_known"] == 1
    last = disk_restore_verify.view()["last_boundary"]
    assert last["missing"] == [ALIGN_GROUP]
    assert last["groups_present"] == [0, 1, 4]
    assert disk_restore_verify.section()["boundary"]["missing_by_group"] == {
        ALIGN_GROUP: 1}


def test_a_batch_that_included_the_align_group_is_not_counted(armed, worker):
    """The negative control: the same batch WITH the state block is a clean shape."""
    disk_restore_verify.observe_apply(
        worker, batch_of([job("/tmp/a", group=0), job("/tmp/b", group=ALIGN_GROUP),
                          job("/tmp/c", group=4)]), {})
    assert counters()["boundary_entry_missing"] == 0
    assert counters()["batches_seen"] == 1
    assert disk_restore_verify.view()["last_boundary"] is None


def test_a_boot_with_no_align_group_counts_nothing(armed, seam, tmp_path):
    """A config with no align-mode group has no boundary shape to be missing.

    The counter must stay at zero rather than invent a missing group from a config that
    never had one - which is also the reason the worker's own spec, not a model name or
    a group id, is what this classification reads.
    """
    plain = VerifierWorker(align=())
    seam(plain.gpu_kv_caches)
    disk_restore_verify.observe_apply(plain, batch_of([job("/tmp/a", group=0)]), {})
    assert counters()["boundary_entry_missing"] == 0
    assert counters()["boundary_groups_known"] == 0
    # The ABSENCE of a boundary group is published as a reason in the geometry view, not
    # as a counter: "this boot has no state group" is a fact about the config, and a
    # reader who does not know it would read a zero as "the truncation never bit".
    assert "no group's spec reports" in (
        disk_restore_verify.view()["geometry"]["boundary_note"] or "")


def test_an_entry_failed_at_the_apply_point_does_not_count_as_missing(armed, worker):
    """`failed` is subtracted before the shape is read: a dropped entry is not a shift.

    If the align entry was there and its READ failed, that is `disk_restore`'s finding
    (`file_missing`, `not_settled`), already counted there. Counting it again as a
    truncation shape would make one failure two, in two different files.
    """
    bound = batch_of([job("/tmp/a", group=0), job("/tmp/b", group=ALIGN_GROUP)])
    disk_restore_verify.observe_apply(worker, bound, {1: "not_settled"})
    assert counters()["boundary_entry_missing"] == 0


# --- 6: what the step thread pays --------------------------------------------
def test_the_apply_point_touches_no_file_and_no_device_and_barely_counts(
        armed, worker, monkeypatch, tmp_path):
    """`observe_apply` is bookkeeping: no file I/O, no GPU read, no CUDA, bounded.

    Both readers are instrumented to raise, so a step-thread read is a test failure
    rather than a claim; the batch is deliberately wider than the per-step retention
    bound to prove the bound is enforced on the step thread, not on the verifier's.
    """
    def refused(*_args, **_kwargs):
        raise AssertionError("the apply point reached the file system or the device")

    monkeypatch.setattr(disk_restore_verify, "_read_file_head", refused)
    monkeypatch.setattr(disk_shadow, "_read_gpu_slot", refused)
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 1)
    path = window_file(tmp_path)
    jobs = [job(path, group=0) for _ in
            range(disk_restore_verify.MAX_ENTRIES_SCANNED + 40)]
    report = disk_restore_verify.observe_apply(worker, batch_of(jobs), {})
    assert report["retained"] == disk_restore_verify.MAX_RETAINED_PER_STEP
    assert report["seen"] <= disk_restore_verify.MAX_ENTRIES_SCANNED
    assert counters()["retained"] == disk_restore_verify.MAX_RETAINED_PER_STEP


def test_the_sample_rate_is_a_rate_over_applied_entries(armed, worker, monkeypatch,
                                                        tmp_path):
    """Every 32nd applied entry, counted both ways: `sampled_out` + `retained`.

    The population is the entries the apply point ACTUALLY applied, so the rate is a
    rate over restores that moved bytes - the figure an operator divides by.
    """
    path = window_file(tmp_path)
    jobs = [job(path, group=0) for _ in range(64)]
    disk_restore_verify.observe_apply(worker, batch_of(jobs), {})
    assert counters()["applies_seen"] == 64
    # 64 entries, every 32nd one -> 2 sampled, and 62 counted as passed over.
    assert counters()["sampled_out"] == 62
    assert counters()["retained"] == 2


def test_the_retention_ring_drops_the_oldest_and_says_so(armed, worker, monkeypatch,
                                                         tmp_path):
    """A bounded queue that drops the OLDEST: the oldest entry's GPU block is the one
    most likely to have been handed out again, so it is the least likely to be evidence.
    """
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 1)
    monkeypatch.setattr(disk_restore_verify, "MAX_RETAINED_PER_STEP", 8)
    monkeypatch.setattr(disk_restore_verify, "MAX_RETAINED_JOBS", 4)
    path = window_file(tmp_path)
    for _ in range(3):
        disk_restore_verify.observe_apply(
            worker, batch_of([job(path, group=0, gpu_block=5) for _ in range(4)]), {})
    assert len(disk_restore_verify.view()["pending"]) == 4
    assert counters()["retained_dropped"] >= 1


def test_no_thread_means_no_verification_and_a_named_refusal(armed, worker, monkeypatch,
                                                             tmp_path):
    """With no thread to hand a job to, the entry is refused NOW, not queued forever.

    Retained-but-never-verified bytes are a lie waiting to be counted, so the refusal is
    taken on the spot and counted - the `disk_gpu_shadow` posture, for the same reason.
    """
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread", lambda: False)
    path = window_file(tmp_path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    assert counters()["refused_thread_unavailable"] == 1
    assert disk_restore_verify.view()["pending"] == {}


def test_the_interval_floor_counts_rather_than_sleeps(armed, worker, seam, monkeypatch,
                                                      tmp_path):
    """Two passes in a row: the second is refused work, and it returns immediately.

    A floor that SLEPT would put a wait on whoever called it; this one answers and
    counts. The elapsed time is asserted to be small, which is the only way to tell
    those two designs apart from the outside.
    """
    seam(worker.gpu_kv_caches)
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    started = time.monotonic()
    first = disk_restore_verify.run_pass(worker)
    second = disk_restore_verify.run_pass(worker)
    assert first["ran"] is True and second["ran"] is False
    assert second["reason"] == "throttled_interval"
    assert counters()["throttled_interval"] == 1
    assert time.monotonic() - started < 0.2          # a sleep of 0.5 s would fail this


def test_the_thread_starts_lazily_and_drains_without_being_asked(armed_env,
                                                                monkeypatch, tmp_path,
                                                                worker, seam):
    """One thread, started by the first retained entry, and it does the work itself.

    This test deliberately does NOT use the `armed` fixture: that one patches
    `_ensure_thread` to keep the thread off, and this is the test about the thread. The
    store root therefore comes from `VLLM_E2_TIER_DISK_ROOT` (the real path the file
    reader resolves) instead of from the fake seam's directory.

    The wait is bounded: a verifier that never drains is a failure, not a slow test.
    """
    root = tmp_path / "spill"
    root.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    local = seam(worker.gpu_kv_caches)
    assert disk_restore_verify.counters()["thread_started"] == 0
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    assert counters()["thread_started"] == 1
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and counters()["checked"] == 0:
        time.sleep(0.02)
    assert counters()["checked"] == 1, "the thread never verified the retained entry"
    assert counters()["file_missing"] == 0
    assert local.calls == [("g0", 5)]
    assert disk_restore_verify.status()["thread_alive"] is True


def test_a_dead_worker_drops_its_entries_rather_than_copying_freedom_memory(armed_env,
                                                                            no_thread,
                                                                            seam,
                                                                            tmp_path):
    """The worker is held weakly: when it is gone, entries are dropped, not read.

    The thread that copies from KV tensors is not the thread that frees them. A strong
    reference would keep freed-but-alive tensors alive, and a device copy of a freed
    region is not a diagnostic, so the answer is a counter and a stop.

    The worker is built HERE rather than taken from the `worker` fixture for a reason: a
    fixture value is still referenced by pytest's own machinery after `del`, so the
    weakref would stay alive and this test would silently assert the happy path. Owning
    the only reference is what makes `del` mean "the worker is gone".
    """
    target = VerifierWorker()
    local = seam(target.gpu_kv_caches)
    path = window_file(tmp_path)
    disk_restore_verify.observe_apply(target, batch_of([job(path)]), {})
    assert len(disk_restore_verify.view()["pending"]) == 1
    del target
    verdict = one_pass(None)
    assert verdict["reason"] == "worker_gone"
    assert counters()["worker_gone"] == 1
    assert counters()["checked"] == 0
    assert disk_restore_verify.view()["pending"] == {}
    assert local.calls == []          # no device read after the worker died


def test_an_observe_apply_that_raises_is_a_counter_not_an_exception(armed, monkeypatch,
                                                                    tmp_path):
    """A witness that can break a restore is not a witness, it is an outage.

    The hook sits inside `disk_restore`'s apply point, so the only acceptable failure
    mode here is a counter: the step must proceed exactly as it would have.
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("the geometry walk met a config nobody has seen")

    monkeypatch.setenv(config.ENV_DISK_RESTORE_VERIFY, "1")
    monkeypatch.setattr(disk_restore_verify, "_refresh_geometry", explode)
    report = disk_restore_verify.observe_apply(VerifierWorker(),
                                               batch_of([job("/tmp/a")]), {})
    assert report["reason"].startswith("error_")
    assert counters()["errors"] == 1


# --- 7: the hook site, driven through the real restore path -------------------
@pytest.fixture
def restore_stage(monkeypatch, tmp_path):
    """The gate open, a rank identity pinned, the key machinery present.

    The same six switches `test_disk_restore.armed` raises - it is a fixture in that
    file and pytest does not share fixtures across modules, so it is rebuilt here rather
    than forked into a helper that would make two gate definitions. The point of having
    it at all is these two tests: they drive the REAL `worker_bind` and
    `worker_before_load`, so the verifier's hook site and the field names it reads are
    pinned by the product, not by a fake.
    """
    root = tmp_path / "spill"
    root.mkdir()
    diag = tmp_path / "diag"
    diag.mkdir()
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(diag))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    monkeypatch.setattr(disk_spill, "disk_root", lambda: str(root))
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    monkeypatch.setattr(disk_spill, "geometry_digest", lambda: DIGEST)
    monkeypatch.setattr(backport, "is_armed", lambda: True)
    monkeypatch.setattr(disk_restore, "worker_proof",
                        lambda force=False: {"ok": True, "count": 1,
                                             "ranks": ["PP0_TP0"], "reason": None,
                                             "ranks_armed": 1, "ranks_seen": 1,
                                             "reads_applied": 0, "reason_kind": None})
    monkeypatch.setattr(disk_store, "counters", lambda: {
        "geometry_digest": DIGEST, "fingerprint": {"total_bytes_per_block": WIDTH},
        "rank": "PP0_TP0"})
    module = types.ModuleType(disk_restore.KV_UTILS_MODULE)
    module.make_block_hash_with_group_id = pack
    module.resolve_block_hashes = resolve
    monkeypatch.setitem(sys.modules, disk_restore.KV_UTILS_MODULE, module)
    disk_restore.set_disk_authority(True)
    return root


def spill_file(root, seed, group, *, body=None) -> str:
    """One window-wide file at the path the WORKER will re-derive for this key."""
    made = disk_layout.build_slot_path(str(root), key_hex(seed, group), DIGEST)
    assert made["ok"], made["reason"]
    os.makedirs(os.path.dirname(made["path"]), exist_ok=True)
    payload = bytes((seed * 11 + group + offset) % 251 for offset in range(WIDTH)) \
        if body is None else body
    with open(made["path"], "wb") as handle:
        handle.write(payload[:WIDTH].ljust(WIDTH, b"\x00"))
    appended = disk_index.journal_append(disk_index.SPILL_LINE, key_hex(seed, group),
                                         WIDTH)
    assert appended["ok"], appended["reason"]
    disk_index.refresh(force=True)
    return made["path"]


def restore_meta(root, entries, event=11):
    """A metadata whose disk entries carry the scheduler's identity-in-path shape.

    GPU ids are small on purpose (0, 1, 2...): they have to be indexable in the fake GPU
    region, and a misalignment between the CPU and GPU lists is still visible because
    the bytes in each block differ.
    """
    cpu, gpu, paths = [], [], []
    for path in entries:
        cpu.append(len(cpu))
        gpu.append(len(gpu))
        paths.append(path)
    return Meta(load_event=event, cpu=cpu, gpu=gpu, paths=paths,
                reqs={event: ["req-1"]})


def test_the_real_apply_point_hands_the_verifier_the_applied_entry(restore_stage,
                                                                   monkeypatch, worker,
                                                                   seam):
    """bind -> settle -> start_load_kv, with the verifier watching: it retained EXACTLY
    the entry the apply point applied, with the fields the real code writes, and the
    comparison then says the GPU holds the file's bytes.

    This is the test that pins the hook SITE. If the call moved above the apply loop it
    would see entries that never applied; if it moved below the prune it could not tell
    applied from dropped; if it read the wrong field name it would retain nothing.
    """
    monkeypatch.setenv(config.ENV_DISK_RESTORE_VERIFY, "1")
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread", lambda: True)
    monkeypatch.setattr(disk_restore_verify, "SAMPLE_EVERY_N_RESTORES", 1)
    local = seam(worker.gpu_kv_caches)
    path = spill_file(restore_stage, 1, 0)
    gpu_matches(worker.gpu, 0, path)
    meta = restore_meta(restore_stage, [path])
    applied_before = int(disk_restore._state["reads_applied"])
    disk_restore.worker_bind(worker, meta)
    worker.bind_connector_metadata(meta)
    disk_restore.worker_before_load(worker)
    # THE DELTA, not the absolute: `disk_restore`'s own applied counter is process state
    # that a previous test in this file has already moved, and a test that asserted an
    # absolute number would be asserting the order the file happens to run in.
    assert disk_restore._state["reads_applied"] == applied_before + 1
    assert counters()["applies_seen"] == 1
    pending = disk_restore_verify.view()["pending"]
    assert len(pending) == 1
    retained = list(pending.values())[0]
    assert retained["path"] == path and retained["group"] == 0
    assert retained["gpu_block"] == 0 and retained["bytes"] == WIDTH
    assert retained["event"] == 11
    worker.complete(11)
    assert one_pass(worker)["results"][0]["equal"] is True
    assert local.calls == [("g0", 0)]


def test_the_real_apply_point_with_the_verifier_off_changes_nothing(restore_stage,
                                                                    monkeypatch, worker,
                                                                    seam):
    """The same restore with the flag unset: one applied read, ZERO verifier state.

    The mirror of the test above, on the same path: the difference between them is one
    environment variable, so "default off" is proved against the real apply point rather
    than against a fake batch.
    """
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread",
                        lambda: (_ for _ in ()).throw(AssertionError(
                            "the disabled verifier started a thread")))
    local = seam(worker.gpu_kv_caches)
    path = spill_file(restore_stage, 1, 0)
    meta = restore_meta(restore_stage, [path])
    applied_before = int(disk_restore._state["reads_applied"])
    disk_restore.worker_bind(worker, meta)
    worker.bind_connector_metadata(meta)
    assert disk_restore.worker_before_load(worker)["ok"] is True
    # The restore REALLY happened (one more applied read), so the zero verifier state
    # below is "the witness was not there", not "nothing happened to witness".
    assert disk_restore._state["reads_applied"] == applied_before + 1
    assert counters() == {key: 0 for key in disk_restore_verify._COUNTER_KEYS}
    assert disk_restore_verify.view()["pending"] == {}
    assert local.calls == []


# --- 8: the view, the seam's cap, and the verb -------------------------------
def test_the_section_key_set_never_changes_and_fits_the_contribution_cap():
    """The degraded view and the healthy view have the SAME keys, and the payload fits.

    `worker_diag` REFUSES an over-cap section rather than truncating it, so a section
    that gets fat on a bad boot disappears on the boot someone is reading the file
    about. Hence a self-imposed budget - and the budget is asserted here with the reason
    list FULL, which is the worst case, not the quiet one.
    """
    quiet = disk_restore_verify.section()
    disk_restore_verify._state["reasons"] = [
        f"tag{index}: " + "a persistent condition described at length " * 4
        for index in range(disk_restore_verify.MAX_REASON_SAMPLES)]
    disk_restore_verify._state["boundary_missing_groups"].update(
        {index: 1 for index in range(40)})
    loud = disk_restore_verify.section()
    assert set(quiet) == set(loud)
    size = len(json.dumps(loud, separators=(",", ":"),
                          default=str).encode("utf-8"))
    assert size <= worker_diag.MAX_CONTRIBUTION_BYTES
    assert size <= disk_restore_verify.MAX_SECTION_BYTES
    assert loud["section_trimmed"] >= 1
    assert loud["trimmed"]                       # it says WHAT it shed
    disk_restore_verify.reset_state_for_tests()


def test_the_fifth_contribution_is_registered_and_the_rank_file_survives_six():
    """The allow-list now holds SIX; the required fields and the cap still hold.

    The risk of `MAX_CONTRIBUTIONS` 4 -> 5 was a FILE risk, not a reader risk, and 5 ->
    6 is the same risk one step further: six sections can ask for 24 KiB of a 32 KiB
    ceiling. THE RAISE WAS NOT THIS MODULE'S ASK - `disk_gpu_write` (0.2.43) needed the
    sixth slot, and because the seam REFUSES an unregistered section, without the raise
    the new diagnostic would have been silently ABSENT from the rank file. So this
    writes a rank file carrying all SIX sections and asserts (a) all six were written,
    (b) every `KEEP_KEYS` field the proxy host's reader requires is present, and (c) the
    file stays under the ceiling - and it asserts the cap and the allow-list moved
    together, which is the drift that makes a new section silently absent.
    """
    assert len(worker_diag.CONTRIBUTION_SECTIONS) == worker_diag.MAX_CONTRIBUTIONS == 6
    assert "disk_restore_verify" in worker_diag.CONTRIBUTION_SECTIONS
    registered = disk_restore_verify.register_contribution()
    assert registered["registered"] is True, registered
    assert worker_diag.contributions()["registered"] == ["disk_restore_verify"]
    for name in ("disk", "disk_restore", "disk_shadow", "disk_gpu_shadow",
                 "disk_gpu_write"):
        made = worker_diag.register_contribution(
            name, lambda: {"note": "a section standing in for its owner"},
            owner=f"test/{name}")
        assert made["registered"] is True, made
    document = worker_diag.build_document(None, "PP0_TP0", "env")
    written = document["writer"]["contributions"]["written"]
    for name in worker_diag.CONTRIBUTION_SECTIONS:
        assert name in written, name
    for key in ("version", "epoch", "ts", "role", "rank", "arena", "counters"):
        assert key in document, key
    assert document["rank"] == "PP0_TP0"
    assert "disk_restore_verify" in document
    assert len(json.dumps(document)) <= worker_diag.MAX_FILE_BYTES


def test_the_verb_reads_and_arms_and_refuses_a_non_boolean():
    """`restore-verify` reads; `enabled` latches; a non-boolean changes nothing.

    Same contract as `shadow` and `gpu-shadow`: a key that is PRESENT is an instruction,
    a key that is absent is a question. The answer carries `scope`, because the process
    being armed here is not the process that applies a restore.
    """
    read = disk_store.disk_for_engine(args={"op": "restore-verify"})
    assert read["ok"] is True and read["restore_verify"]["enabled"] is False
    assert read["restore_verify"]["env_var"] == config.ENV_DISK_RESTORE_VERIFY
    armed_answer = disk_store.disk_for_engine(args={"op": "restore-verify",
                                                    "enabled": True})
    assert armed_answer["restore_verify"]["enabled"] is True
    assert config.ENV_DISK_RESTORE_VERIFY in armed_answer["note"]
    bad = disk_store.disk_for_engine(args={"op": "restore-verify", "enabled": "yes"})
    assert bad["ok"] is False and "JSON boolean" in bad["reason"]
    assert bad["restore_verify"]["enabled"] is True     # refused, not guessed
    disk_store.disk_for_engine(args={"op": "restore-verify", "enabled": False})


def test_the_op_is_mutating_listed_post_only_and_in_the_endpoint_enum():
    """It writes nothing, but it changes what the process DOES: POST-only.

    `op` is a query parameter on this route, so a browser's `<img>` preload could arm a
    device-reading verifier from a GET if this listing slipped. The endpoint's own enum
    is
    asserted against `disk_store.DISK_OPS` because the two lists drifting is how a verb
    becomes unreachable.
    """
    assert "restore-verify" in disk_store.DISK_OPS
    assert "restore-verify" in disk_store.MUTATING_DISK_OPS
    refused = disk_store.disk_for_engine(args={"op": "restore-verify",
                                               "safe_method": True})
    assert refused["ok"] is False and "POST" in refused["reason"]
    values = endpoint.ARG_SPECS[config.ROUTE_DISK]["op"]["values"]
    assert set(values) == set(disk_store.DISK_OPS)


def test_the_status_and_the_verb_and_the_rank_file_report_the_same_numbers():
    """One builder, three doors: a number that differs by door is unusable.

    `status()` builds its payload from `section()`, the contribution calls
    `worker_report()` which is `section()`, and the verb reads `status()` - so this
    asserts the identity rather than trusting the comment that claims it.
    """
    with disk_restore_verify._lock:
        disk_restore_verify._state["counters"]["batches_seen"] = 7
    verb = disk_store.disk_for_engine(args={"op": "restore-verify"})["restore_verify"]
    contribution = disk_restore_verify.worker_report()
    assert verb["batches_seen"] == 7
    assert {key: contribution[key] for key in disk_restore_verify._COUNTER_KEYS} == \
        {key: verb[key] for key in disk_restore_verify._COUNTER_KEYS}
    assert contribution["geometry"]["window_bytes"] == WIDTH or \
        contribution["geometry"]["window_bytes"] is None


def test_the_gpu_read_is_disks_shadow_seam_and_not_a_second_mechanism(monkeypatch,
                                                                      worker, tmp_path):
    """The module has no device-copy code of its own, and no ctypes of a device address.

    Two proofs, because the hazard is a SIGSEGV rather than an exception:
      * monkeypatching `disk_shadow._copy_slot_to_host` redirects THIS module's reads
        (they are the same call path, so a test can reach the device half at all);
      * the source contains no device-pointer read of its own. The invariant
        `tests/test_snapshot.py::test_no_heavy_imports_in_the_read_side` enforces for
        imports covers the same ground for symbols, and this pins the rest.
    """
    seen: list = []

    def spy(tensor, index):
        seen.append((id(tensor), int(index)))
        return CopySeam({"g0": tensor})(tensor, index)

    monkeypatch.setenv(config.ENV_DISK_RESTORE_VERIFY, "1")
    monkeypatch.setattr(disk_restore_verify, "_ensure_thread", lambda: True)
    monkeypatch.setattr(disk_shadow, "_copy_slot_to_host", spy)
    path = window_file(tmp_path)
    gpu_matches(worker.gpu, 5, path)
    disk_restore_verify.observe_apply(worker, batch_of([job(path)]), {})
    assert one_pass(worker)["checked"] == 1
    assert seen == [(id(worker.gpu), 5)]
    with open(disk_restore_verify.__file__, encoding="utf-8") as handle:
        source = handle.read()
    for forbidden in ("string_at(", "memmove(", "data_ptr(", "import ctypes"):
        assert forbidden not in source, forbidden

