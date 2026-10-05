r"""The parallel spill reader (0.2.24): N threads, the pool that bounds them, and
the ordering that concurrency exposes.

Why each group below exists, in the order a failure would cost the box:

1. **THE DEFAULTS ARE THE OLD BOOT.** Everything here ships into a process serving
   real traffic, so the default thread count and the default pool must reproduce
   0.2.23 exactly - same single thread name, same bytes of pool, one unit in flight
   at a time, and no log line. Pinned directly, not argued.
2. **THE ENV GRAMMAR FAILS CLOSED.** A typo in `VLLM_E2_TIER_DISK_READ_THREADS`
   means "the operator said nothing" (one reader), reported in `env_problems`; an
   absurd value is CLAMPED and says so. `config.py` remains the only reader of the
   environment: `disk_read.py` and `disk_restore.py` contain no `os.environ`, and a
   test reads their source to keep it that way.
3. **THE POOL, NOT THE THREADS, BOUNDS IN-FLIGHT WORK.** A buffer is taken at issue
   and returned at the apply point, so a too-small pool shows up as `buffer_cap` at
   ISSUE time and no number of threads can fix it. These tests pin the refusal, the
   witness (`concurrency.starved_on`) and the arithmetic (pool bytes / unit bytes).
4. **N THREADS REALLY READ AT ONCE - AND ONE DOES NOT.** A `threading.Barrier` is
   the proof: it only trips if the reads are simultaneously inside the read. The
   control is the single-thread boot, where `units_in_flight_peak` can only be 1.
5. **THE OLD INVARIANTS SURVIVE.** `thread_dead` still refuses a batch when nothing
   can read; a dead slot is respawned; cancellation with N threads never returns one
   buffer twice (the silent-corruption shape); `MAX_QUEUE_DEPTH` still bounds the
   queue; and no lock is held across a file read, which is what keeps the engine's
   step thread from queueing behind a kernel read.
6. **COMPLETION ORDER, MEASURED AND NOT DECIDED.** Whether a PARTIAL restore may
   have its tokens counted as computed belongs to another change. These tests
   publish the shape a partial restore takes with N readers (a hole in the MIDDLE
   instead of a hole at the end) and the bound on the disorder (a FIFO queue: a job
   cannot start while more than threads-1 earlier jobs are unstarted).

Stdlib and the plugin only. `tmp_path` for every file; no engine, no vllm, no box.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from kvtier_dev_plugin import config, disk_layout, disk_read, disk_restore, worker_diag

KEY16 = "a3f0b7c1d4e5f607"
GEO = "abcdef123456"
# Small on purpose: the point is the concurrency and the accounting, not 11 MB of
# bytes. The REAL unit width is what the arithmetic in `pool` publishes, and that is
# pinned against the constant rather than against a number typed into a test.
UNIT = 4096


@pytest.fixture(autouse=True)
def fresh_reader(tmp_path, monkeypatch):
    """A fresh boot per test, with NO reader thread carried across the boundary.

    Why the teardown is more than `reset_state_for_tests()`: a reader thread is bound
    to the queue OBJECT it was handed, and `_state` is a module global. A thread that
    leaves this test parked inside a stand-in read (or with this test's jobs still in
    its queue) will finish them under the NEXT test's boot - and every counter that
    test then waits on can be satisfied by the leftover's read. That is not a
    hypothesis: a four-thread barrier test failed with three reads attributed, four
    threads alive and `reads_completed == 4`, because a previous test's parked reader
    woke up, called the module-level `_read_file` (which the NEXT test had by then
    replaced with its own stand-in), and incremented the new boot's total.

    So teardown releases every parked stand-in, DROPS the jobs still in the queue (a
    leftover job must never be read under a boot that did not queue it), posts one
    sentinel per reader thread the process still has - including a thread the module
    no longer publishes, which is why the count comes from `threading.enumerate()` and
    not from `_threads` - and JOINS them. After teardown nothing can write to `_state`
    except the next test. Setup runs the same sweep without the assertion, so a leak
    from another file cannot poison this one either.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    _stop_all_readers(strict=False)
    disk_read.reset_state_for_tests()
    disk_restore.reset_state_for_tests()
    yield
    _stop_all_readers(strict=True)
    disk_read.reset_state_for_tests()
    disk_restore.reset_state_for_tests()


_GATES: list = []


def _release_parked_reads() -> None:
    """Let every stand-in read this test parked finish, so its thread can be joined."""
    for gate in _GATES:
        gate.hold.set()
    _GATES.clear()


def _stop_all_readers(strict: bool) -> None:
    _release_parked_reads()
    readers = [thread for thread in threading.enumerate()
               if str(thread.name).startswith("kvtier-disk-read")]
    pending = disk_read._queue
    if pending is not None:
        while True:                   # leftovers first: sentinels must be next out
            try:
                pending.get_nowait()
            except queue.Empty:
                break
        for _ in readers:
            try:
                pending.put_nowait(disk_read._STOP)
            except queue.Full:
                break
    for thread in readers:
        thread.join(3.0)
    left = sorted(str(thread.name) for thread in threading.enumerate()
                  if str(thread.name).startswith("kvtier-disk-read"))
    if strict and left:
        pytest.fail(f"a reader thread survived teardown and could write into the "
                    f"next test's ledger: {left}")


def slot_path(root, index=0, group=0):
    """A committed-file NAME this plugin would accept, one per (span, group)."""
    key = KEY16 + f"{index:048x}" + f"{group:08x}"
    made = disk_layout.build_slot_path(str(root), key, GEO)
    assert made["ok"], made["reason"]
    return made["path"], made["filename"]


def write_slot(root, index=0, group=0, size=UNIT):
    path, _name = slot_path(root, index, group)
    written = disk_layout.write_committed_file(path, b"\0" * size, sync=False)
    assert written["ok"], written["reason"]
    return path


def paths_for(root, count, groups=5):
    """`count` slot paths in SUBMISSION order: span-major, group-minor, exactly the
    order `disk_restore._merge_pending_hit` builds them in."""
    out = []
    for index in range(count):
        out.append(write_slot(root, index=index // groups, group=index % groups))
    return out


def issue(batch_id, paths, size=UNIT, start=0):
    """Queue one read per path. Returns (accepted, refusals)."""
    accepted, refused = [], []
    for offset, path in enumerate(paths):
        answer = disk_read.request_read(batch_id, start + offset, path, size)
        (accepted if answer["ok"] else refused).append(answer)
    return accepted, refused


def wait_for(predicate, limit=400, pause=0.005):
    for _ in range(limit):
        if predicate():
            return True
        time.sleep(pause)
    return False


def pooled_identities():
    return [id(buffer) for buffer, _size in disk_read._buffers]


def _job_id_of(path):
    """The job id a stand-in read stands in for: the `g<group>` field, the only
    per-unit number in a spill file name these tests control."""
    return int(str(path).rsplit("-g", 1)[1].split("-")[0])


# --- 1: the defaults are 0.2.23 exactly ---------------------------------------

def test_defaults_reproduce_the_single_thread_boot_exactly(tmp_path):
    """No env set = one thread, the old name, the old pool, one unit in flight."""
    limits = disk_read._limits_now()
    assert limits["threads"] == disk_read.DEFAULT_READ_THREADS == 1
    assert limits["buffer_bytes"] == disk_read.MAX_BUFFERS_BYTES
    assert limits["threads_requested"] is None
    assert limits["buffer_bytes_requested"] is None
    assert disk_read.MAX_POOLED_BUFFERS > disk_read.MAX_BUFFERS_BYTES \
        // 11_468_800, "the object bound must not bind before the byte bound"

    paths = paths_for(tmp_path, 6)
    assert disk_read.begin_batch("defaults", len(paths))["ok"] is True
    accepted, refused = issue("defaults", paths)
    assert refused == [] and len(accepted) == 6
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 6), \
        disk_read.batch_status("defaults")
    disk_read.finish_batch("defaults")

    state = disk_read.counters()
    assert state["thread_alive"] is True
    assert state["threads_alive"] == 1
    assert state["threads_configured"] == 1
    assert state["pool_limit_bytes"] == disk_read.MAX_BUFFERS_BYTES
    # ONE author for every byte this boot read, under the pre-0.2.24 thread name.
    detail = state["threads_detail"]
    assert len(detail) == 1, f"a default boot spawned {len(detail)} threads"
    assert detail[0]["name"] == "kvtier-disk-read"
    assert detail[0]["alive"] is True
    assert detail[0]["jobs_read"] == 6
    assert detail[0]["bytes_read"] == 6 * UNIT
    # A single reader can never have two units inside the kernel at once.
    assert state["units_in_flight"] == 0
    assert state["units_in_flight_peak"] == 1
    # Summed reader seconds over wall seconds: with one author that ratio cannot be
    # anything but <= 1.0, and it is the number to read next to `threads_configured`.
    assert state["throughput"]["parallel_efficiency"] <= 1.0


def test_the_default_resolution_logs_each_episode_once(monkeypatch):
    """Once per boot, never once per step: the `notice_once` idiom this repo already
    uses everywhere on the engine path. And the DEFAULT resolution announces nothing,
    because a boot that changes nothing must not add a line to the engine log.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4")
    disk_read._notices.clear()
    disk_read._limits_resolved = False
    disk_read._limits = {}
    assert disk_read._limits_now()["threads"] == 4
    assert "limits" in disk_read._notices, "a changed posture went unannounced"
    disk_read._limits_resolved = False
    disk_read._limits = {}
    disk_read._limits_now()
    assert len([key for key in disk_read._notices if key == "limits"]) == 1
    monkeypatch.delenv(config.ENV_DISK_READ_THREADS)
    disk_read._notices.clear()
    disk_read._limits_resolved = False
    disk_read._limits = {}
    disk_read._limits_now()
    assert "limits" not in disk_read._notices, "a default boot logged a change line"


def test_reset_leaves_the_default_shape_even_after_a_parallel_boot(monkeypatch):
    """The isolation the per-thread witnesses depend on."""
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4")
    assert disk_read.begin_batch("parallel", 0)["ok"] is True
    assert disk_read._limits_now()["threads"] == 4
    disk_read.stop_reader(timeout_seconds=1.0)
    disk_read.reset_state_for_tests()
    monkeypatch.delenv(config.ENV_DISK_READ_THREADS)
    assert disk_read._ensure_thread() is True
    assert disk_read._limits_now()["threads"] == 1
    assert disk_read.counters()["threads_configured"] == 1


# --- 2: the env grammar -------------------------------------------------------

def test_env_var_sets_threads_and_pool(monkeypatch):
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4")
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, "1048576")
    limits = disk_read._limits_now()
    assert limits["threads"] == 4 and limits["threads_requested"] == 4
    assert limits["buffer_bytes"] == 1048576
    assert limits["buffer_bytes_requested"] == 1048576
    assert limits["problems"] == []


def test_a_typo_means_the_old_behaviour_and_is_reported(monkeypatch):
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "four")
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, "-8")
    limits = disk_read._limits_now()
    assert limits["threads"] == 1 and limits["buffer_bytes"] == \
        disk_read.MAX_BUFFERS_BYTES
    assert len(limits["problems"]) == 2, limits["problems"]
    assert disk_read.counters()["env_problems"] == limits["problems"]


def test_an_absurd_request_is_clamped_and_says_so(monkeypatch):
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4096")
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, "999999999999999")
    limits = disk_read._limits_now()
    assert limits["threads"] == disk_read.MAX_READ_THREADS
    assert limits["threads_clamped"] is True
    assert limits["buffer_bytes"] == disk_read.MAX_BUFFERS_BYTES_CEILING
    assert limits["buffer_bytes_clamped"] is True
    caps = disk_read.section()["caps"]
    assert caps["read_threads"] == disk_read.MAX_READ_THREADS
    assert caps["max_buffers_bytes"] == disk_read.MAX_BUFFERS_BYTES_CEILING
    assert caps["max_buffers_bytes_default"] == disk_read.MAX_BUFFERS_BYTES


def test_the_reader_still_owns_no_environment_reading():
    """`config.py` is the only env reader. Read the CODE, not the prose: a rule a
    test can check is a rule that survives the next well-meaning edit.

    Checked on the syntax tree rather than on the text, because a comment that
    MENTIONS `os.environ` to explain why the module does not read it must not fail
    the test that exists to keep it that way.
    """
    import ast
    import pathlib

    root = pathlib.Path(disk_read.__file__).parent
    for name in ("disk_read.py", "disk_restore.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("environ", "getenv"), \
                    f"{name}:{node.lineno} touches the environment directly"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "getenv", \
                    f"{name}:{node.lineno} calls getenv"
        # And the rule is not vacuous: config.py, which owns the env, does use it.
    config_tree = ast.parse((root / "config.py").read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.Attribute) and node.attr == "environ"
               for node in ast.walk(config_tree))


# --- 3: the pool is the binding constraint ------------------------------------

def test_a_pool_that_cannot_hold_the_offer_refuses_at_issue(monkeypatch, tmp_path):
    """`buffer_cap` is what a big prompt looks like when the pool is the limit.

    Three buffers of pool, five units offered: two refusals, and the witness names
    buffers as the thing the boot starved on. This is the live box's 10-span offer
    in miniature, and it is why the pool env var is the one that matters first.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, str(3 * UNIT))
    paths = paths_for(tmp_path, 5)
    assert disk_read.begin_batch("small-pool", len(paths))["ok"] is True
    accepted, refused = issue("small-pool", paths)
    assert len(accepted) == 3 and len(refused) == 2
    assert {item["reason_kind"] for item in refused} == {"buffer_cap"}
    counters = disk_read.counters()
    assert counters["refused"]["buffer_cap"] == 2
    assert counters["pool_limit_bytes"] == 3 * UNIT
    assert counters["pool_headroom_bytes"] == 0
    section = disk_read.section()
    assert section["concurrency"]["starved_on"] == "buffers"
    assert section["pool"]["refused_for_no_buffer"] == 2
    # THE ARITHMETIC THE OPERATOR SIZES ON, at the store's real unit widths.
    pool = section["pool"]
    assert pool["units_of_11468800"] == (3 * UNIT) // 11_468_800
    assert disk_read.section()["pool"]["limit_bytes"] == 3 * UNIT
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 3)
    disk_read.finish_batch("small-pool")


def test_the_default_pool_holds_eight_units_of_the_measured_width():
    """The published arithmetic, pinned against the constant, not a typed number."""
    pool = disk_read.section()["pool"]
    assert pool["limit_bytes"] == 96 * 1024 * 1024
    assert pool["units_of_11468800"] == 8
    assert pool["units_of_12902400"] == 7


def test_the_pool_sizing_table_in_the_readme_is_the_code(monkeypatch):
    """README tells the operator to set 128 MiB and says what each setting buys. That
    table is arithmetic over `MAX_BUFFERS_BYTES` and the two unit widths, so it is
    pinned here rather than left to be re-derived - and re-derived WRONG, which is how
    a 10-unit offer got described as a 50-unit one once already.

    The live answer to "how many threads are useful" falls out of this table: a thread
    above `floor(pool / unit)` has no buffer to read into.
    """
    widths = (11_468_800, 12_902_400)
    table = {96 * 1024 * 1024: (8, 7),
             128 * 1024 * 1024: (11, 10),
             192 * 1024 * 1024: (17, 15)}
    for size, expected in table.items():
        got = tuple(size // width for width in widths)
        assert got == expected, f"{size} B buys {got} units, README says {expected}"
    # 128 MiB is the recommendation because it is the smallest setting that holds a
    # whole 10-span offer (10 units) on BOTH widths - the wide rank is the binding one.
    wide_offer = 10 * widths[1]
    narrow_offer = 10 * widths[0]
    default_pool = 96 * 1024 * 1024
    recommended = 128 * 1024 * 1024
    assert wide_offer <= recommended and narrow_offer <= recommended
    assert narrow_offer > default_pool, "the default must miss the big offer"

    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, str(128 * 1024 * 1024))
    pool = disk_read.section()["pool"]
    assert pool["limit_bytes"] == 128 * 1024 * 1024
    assert pool["units_of_11468800"] == 11
    assert pool["units_of_12902400"] == 10


def test_the_pool_publishes_the_thread_count_it_could_actually_feed(tmp_path,
                                                                    monkeypatch):
    """`threads_useful_max` answers the operator's question in the payload instead of
    in a comment: with the unit size this boot really reads, how many readers can be
    inside a buffer at once? Anything above that is `reader_starved_polls`.

    The pool is pinned small here (3 units of the test width) so the number is small
    and readable; on the box the same field reads 8 against an 11,468,800 B unit.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_BUFFER_BYTES, str(3 * UNIT))
    paths = paths_for(tmp_path, 2)
    assert disk_read.begin_batch("useful", 2)["ok"] is True
    issue("useful", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 2)
    section = disk_read.section()
    assert section["pool"]["measured_unit_bytes"] == UNIT
    assert section["pool"]["units_of_measured_unit"] == 3
    assert section["concurrency"]["threads_useful_max"] == 3
    disk_read.finish_batch("useful")
    # No reads yet means no measured unit size, and an honest None rather than a 0 or
    # a figure lifted from a comment.
    disk_read.reset_state_for_tests()
    blank = disk_read.section()
    assert blank["pool"]["measured_unit_bytes"] is None
    assert blank["concurrency"]["threads_useful_max"] is None


def test_queue_depth_cap_is_unchanged_and_still_bounds_the_queue(monkeypatch,
                                                                 tmp_path):
    """N threads drain the queue N-at-a-time, so 64 is if anything roomier. Pin it.

    Four threads do not get to make the cap do something different: the cap is about
    how much UNBOUNDED work the process may be holding, and that argument does not
    mention the thread count at all.
    """
    assert disk_read.MAX_QUEUE_DEPTH == 64
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4")
    gate = _Gated(block=True)
    monkeypatch.setattr(disk_read, "_read_file", gate)
    paths = paths_for(tmp_path, disk_read.MAX_QUEUE_DEPTH + 4)
    assert disk_read.begin_batch("deep", len(paths))["ok"] is True
    accepted, refused = issue("deep", paths)
    assert refused, "the queue let an unbounded pile in"
    assert {item["reason_kind"] for item in refused} == {"queue_full"}
    # What may sit between "issued" and "read" is the queue PLUS one job per reader,
    # and nothing else: the bound holds with four readers exactly as it did with one.
    assert len(accepted) <= disk_read.MAX_QUEUE_DEPTH + 4
    assert disk_read.counters()["queue_depth_peak"] <= disk_read.MAX_QUEUE_DEPTH
    gate.hold.set()
    disk_read.cancel_batch("deep", "test teardown")


# --- 4: N readers really read at once -----------------------------------------

class _Gated:
    """A `_read_file` stand-in that records the order and can hold the read open.

    `block=True` parks every read until `hold.set()`, which is how the tests put jobs
    INTO the queue faster than any thread can drain it. `barrier` is the concurrency
    proof: a wait only succeeds if that many reads are inside the read at the same
    moment, and `trips` counts the successes (a `Barrier.n` is a constant and proves
    nothing, so it is not what is asserted).
    """

    def __init__(self, barrier=None, block=False, wait_timeout=5.0):
        self.barrier = barrier
        self.wait_timeout = wait_timeout
        self.hold = threading.Event() if block else _opened()
        self.started: list = []
        self.finished: list = []
        self.trips = 0
        self.lock = threading.Lock()
        # Registered so teardown can release a read this test parked but never
        # released: a thread parked on `hold.wait(5.0)` is unjoinable, and when it
        # finally wakes up it calls whatever `_read_file` is installed THEN - the next
        # test's - and charges its read to that boot's ledger.
        _GATES.append(self)

    def __call__(self, path, buffer, size):
        job_id = _job_id_of(path)
        with self.lock:
            self.started.append(job_id)
        if self.barrier is not None:
            try:
                self.barrier.wait(timeout=self.wait_timeout)
                self.trips += 1
            except threading.BrokenBarrierError:
                pass
        self.hold.wait(timeout=5.0)
        with self.lock:
            self.finished.append(job_id)
        return disk_read._ok(bytes=size, buffer=buffer)


def _opened():
    event = threading.Event()
    event.set()
    return event


def test_four_threads_read_four_units_at_once(monkeypatch, tmp_path):
    """The proof is a barrier: it only trips if the reads are SIMULTANEOUS.

    `units_in_flight_peak == 4` is the published witness of the same fact, so the
    next live run can be read instead of inferred - and a boot that reported four
    threads with a peak of 1 is a boot that never parallelised.

    The `sum(jobs_read) == 4` line is not decoration: it pins that every thread is
    RECOGNISED as a witness by the time it reads its first job. It failed reliably
    enough to be found (3 of 4 attributed, 4 threads alive), because `_spawn_reader`
    published `_threads[slot]` AFTER `Thread.start()`, so a thread that reached a job
    immediately was reading while `_witness(slot)` still said "nobody here" - and the
    ledger under-reported the exact parallelism it exists to show.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4")
    barrier = threading.Barrier(4, timeout=5.0)
    gate = _Gated(barrier=barrier)
    monkeypatch.setattr(disk_read, "_read_file", gate)
    paths = paths_for(tmp_path, 4, groups=4)
    assert disk_read.begin_batch("four", len(paths))["ok"] is True
    accepted, refused = issue("four", paths)
    assert refused == [] and len(accepted) == 4
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 4), \
        disk_read.batch_status("four")
    assert gate.trips == 4, "the four reads were never simultaneously inside a read"
    status = disk_read.batch_status("four")
    assert status["settled"] is True and status["done"] == 4
    counters = disk_read.counters()
    assert counters["units_in_flight_peak"] == 4
    assert counters["threads_alive"] == 4
    alive = [item for item in counters["threads_detail"] if item["alive"]]
    assert len(alive) == 4
    assert sum(item["jobs_read"] for item in alive) == 4, \
        f"the queue did not spread the work: {alive}"
    assert {item["name"] for item in alive} == {
        "kvtier-disk-read", "kvtier-disk-read-1", "kvtier-disk-read-2",
        "kvtier-disk-read-3"}
    disk_read.finish_batch("four")


def test_a_single_thread_boot_cannot_read_two_units_at_once(monkeypatch, tmp_path):
    """THE CONTROL for the test above. With one reader the barrier is unmeetable, so
    the reads only complete because the stand-in gives up waiting - and the peak
    stays 1 forever, which is the invariant a default boot rests on."""
    barrier = threading.Barrier(2, timeout=0.2)
    gate = _Gated(barrier=barrier, wait_timeout=0.2)
    monkeypatch.setattr(disk_read, "_read_file", gate)
    paths = paths_for(tmp_path, 4, groups=4)
    assert disk_read.begin_batch("one", len(paths))["ok"] is True
    issue("one", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 4), \
        disk_read.batch_status("one")
    assert disk_read.counters()["units_in_flight_peak"] == 1
    assert gate.trips == 0, "a single reader somehow met a two-reader barrier"
    disk_read.finish_batch("one")


def test_no_lock_is_held_across_a_file_read(monkeypatch, tmp_path):
    """The discipline that keeps the engine's step out of the reader's way, pinned
    with N threads instead of one.

    `_state_lock` is taken for dict work only. If a reader ever held it across a
    read, the engine's `batch_status` at the apply point would queue behind a kernel
    read - the mistake `SETTLE_POLL_SECONDS` in `disk_restore` exists to avoid, and
    N threads make it N times easier to commit by accident.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "3")
    gate = _Gated(block=True)
    monkeypatch.setattr(disk_read, "_read_file", gate)
    paths = paths_for(tmp_path, 3, groups=3)
    assert disk_read.begin_batch("locks", len(paths))["ok"] is True
    issue("locks", paths)
    assert wait_for(lambda: disk_read.batch_status("locks")["running"] == 3), \
        disk_read.batch_status("locks")
    started = time.monotonic()
    assert disk_read._state_lock.acquire(timeout=2.0), \
        "a reader thread held _state_lock across a file read"
    disk_read._state_lock.release()
    assert disk_read.batch_status("locks")["running"] == 3
    assert disk_read.counters()["units_in_flight"] == 3
    assert disk_read.job_states("locks")["count"] == 3
    assert disk_read.batch_buffer("locks", 0) is None   # RUNNING is not DONE
    assert time.monotonic() - started < 1.0, "the engine path waited on the reader"
    gate.hold.set()
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 3)
    disk_read.finish_batch("locks")
    assert disk_read.counters()["units_in_flight"] == 0


# --- 5: the old invariants, with N of them ------------------------------------

def test_no_reader_at_all_refuses_the_batch(monkeypatch, tmp_path):
    """`thread_dead` still means "nothing can read this", and `reads_started` can
    therefore never be nonzero with no reader in the process."""
    monkeypatch.setattr(disk_read, "_spawn_reader", lambda slot: False)
    answer = disk_read.begin_batch("dead", 3)
    assert answer["ok"] is False and answer["reason_kind"] == "thread_dead"
    assert disk_read.counters()["refused"]["thread_dead"] == 1
    assert disk_read.batch_status("dead")["known"] is False
    assert disk_read.thread_alive() is False


def test_a_partial_spawn_failure_degrades_and_publishes(monkeypatch, tmp_path):
    """Three asked for, one startable: that is a degraded boot, not an unreadable one.

    It must NOT refuse the load (a read is still possible) and it must NOT pretend:
    `threads_alive` 1 against `threads_configured` 3 is the whole claim.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "3")
    real_spawn = disk_read._spawn_reader
    monkeypatch.setattr(disk_read, "_spawn_reader",
                        lambda slot: real_spawn(slot) if slot == 0 else False)
    paths = paths_for(tmp_path, 2)
    assert disk_read.begin_batch("half-dead", len(paths))["ok"] is True
    counters = disk_read.counters()
    assert counters["threads_configured"] == 3
    assert counters["threads_alive"] == 1
    assert counters["thread_alive"] is True, "one live reader can read"
    accepted, refused = issue("half-dead", paths)
    assert refused == []
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 2)
    detail = disk_read.counters()["threads_detail"]
    assert len(detail) == 3 and sum(1 for row in detail if row["alive"]) == 1
    disk_read.finish_batch("half-dead")


def test_a_thread_start_refusal_is_counted_and_refuses_the_batch(monkeypatch):
    """The counter's own path: `Thread.start()` said no, so the batch is refused.

    A boot that cannot spawn must say so with the name the restore layer already
    buckets (`thread_dead`), not open a batch that nothing will ever read.
    """
    class _Refusing(threading.Thread):
        def start(self):
            raise RuntimeError("reached the thread limit")

    monkeypatch.setattr(disk_read.threading, "Thread", _Refusing)
    answer = disk_read.begin_batch("refused", 1)
    assert answer["ok"] is False and answer["reason_kind"] == "thread_dead"
    counters = disk_read.counters()
    assert counters["thread_spawn_failures"] == 1
    assert counters["refused"]["thread_dead"] == 1
    assert counters["batches_cancelled"] == 1
    assert disk_read.batch_status("refused")["known"] is False


def test_a_dead_slot_is_respawned_and_an_orphan_is_not_a_witness(monkeypatch):
    """A slot whose thread is gone is respawned at the next bind.

    The thread that was in that slot keeps draining the queue it was handed until its
    next EMPTY poll, at which point the self-exit rule retires it (`_witness` no longer
    recognises it, so it belongs to a boot that does not exist). Either way its work
    cannot land in the new reader's ledger, and the ledger cannot double-count the
    respawned one, because `_witness` resolves a slot to the thread the module CURRENTLY
    publishes for it.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "3")
    assert disk_read.begin_batch("respawn", 0)["ok"] is True
    assert disk_read.counters()["threads_alive"] == 3
    orphan = disk_read._threads[1]
    disk_read._threads[1] = None            # the slot lost its thread
    assert disk_read.counters()["threads_alive"] == 2
    assert disk_read.begin_batch("respawn-again", 0)["ok"] is True
    counters = disk_read.counters()
    assert counters["threads_alive"] == 3
    assert counters["thread_started"] >= 4
    replacement = disk_read._threads[1]
    assert replacement is not orphan and replacement.is_alive()
    # `_witness` resolves a slot to the thread the module CURRENTLY publishes for it,
    # so it answers only when the caller IS that thread. From the engine thread (or a
    # test) it is None for every slot - which is exactly why a leftover orphan cannot
    # write into a ledger it is not part of.
    assert disk_read._witness(1) is None
    assert disk_read._slots[1]["spawned"] == 2
    assert disk_read._slot_alive(1) is True
    disk_read.cancel_batch("respawn-again", "test teardown")


def test_an_unpublished_reader_retires_instead_of_polluting_the_next_boot(monkeypatch):
    """The self-exit rule, pinned as product behaviour rather than test hygiene.

    `reset_state_for_tests()` drops the thread table, and a reader thread is bound to
    the queue OBJECT it was handed - so without this rule the threads of a boot that
    no longer exist keep polling, and any job that ever reaches them is charged to
    whoever owns `_state` at the time. A reader therefore checks whether the module
    still publishes IT in this slot, and retires on its next empty poll if not. A
    published reader always has its record, so a healthy thread never takes this exit,
    and a thread with work in front of of it never reaches the branch at all.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "2")
    assert disk_read.begin_batch("orphan", 0)["ok"] is True
    assert disk_read.counters()["threads_alive"] == 2
    first_boot = [thread for thread in disk_read._threads if thread is not None]
    disk_read.reset_state_for_tests()            # that boot is gone
    deadline = time.monotonic() + 3.0 * disk_read.READ_POLL_SECONDS
    while time.monotonic() < deadline:
        if not any(thread.is_alive() for thread in first_boot):
            break
        time.sleep(0.02)
    surviving = [thread.name for thread in first_boot if thread.is_alive()]
    assert surviving == [], f"an unpublished reader would still be polling: {surviving}"
    counters = disk_read.counters()
    assert counters["threads_alive"] == 0 and counters["thread_alive"] is False
    # A starved poll is the only thing an unpublished thread could have counted, and
    # it counted none: the ledger it wrote is empty because it wrote nothing.
    assert counters["reader_starved_polls"] == 0


def test_stop_reader_stops_every_thread_and_a_later_bind_resumes(monkeypatch,
                                                                 tmp_path):
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "3")
    assert disk_read.begin_batch("stop", 0)["ok"] is True
    assert disk_read.counters()["threads_alive"] == 3
    stopped = disk_read.stop_reader(timeout_seconds=2.0)
    assert stopped["stopped"] is True
    assert stopped["threads_stopped"] == 3
    assert disk_read.counters()["threads_alive"] == 0
    assert disk_read.thread_alive() is False
    assert disk_read.begin_batch("restart", 0)["ok"] is True
    assert disk_read.counters()["threads_alive"] == 3


def test_cancelling_with_n_threads_never_returns_a_buffer_twice(monkeypatch,
                                                                tmp_path):
    """The silent-corruption shape, with four readers racing the cancel.

    A QUEUED job's buffer belongs to whoever cancels it and a RUNNING job's to the
    thread inside the kernel - and the handover happens exactly once. Get it wrong
    and the pool holds one buffer twice, hands it to two later reads, and one
    request's file bytes land in another request's CPU KV slot. 0.2.23 had a real
    version of this bug in `finish_batch`; `_handle_read` had the mirror of it, and
    four threads widen any window that exists.
    """
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4")
    gate = _Gated(block=True)
    monkeypatch.setattr(disk_read, "_read_file", gate)
    paths = paths_for(tmp_path, 12, groups=4)
    assert disk_read.begin_batch("cancel", len(paths))["ok"] is True
    accepted, _refused = issue("cancel", paths)
    assert len(accepted) == 12
    assert wait_for(lambda: disk_read.batch_status("cancel")["running"] >= 1)
    cancelled = disk_read.cancel_batch("cancel", "the load went away")
    assert cancelled["cancelled"] is True
    gate.hold.set()
    assert wait_for(lambda: disk_read.counters()["units_in_flight"] == 0)
    time.sleep(0.05)
    identities = pooled_identities()
    assert len(identities) == len(set(identities)), \
        f"the pool holds the SAME buffer twice: {identities}"
    # A cancelled read's bytes are dropped, never applied, and never counted DONE.
    assert disk_read.batch_status("cancel")["known"] is False
    assert disk_read.counters()["batches_cancelled"] == 1


def test_finish_batch_with_reads_in_flight_still_holds_the_same_rule(
        monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, "4")
    gate = _Gated(block=True)
    monkeypatch.setattr(disk_read, "_read_file", gate)
    paths = paths_for(tmp_path, 8, groups=4)
    assert disk_read.begin_batch("finish", len(paths))["ok"] is True
    issue("finish", paths)
    assert wait_for(lambda: disk_read.batch_status("finish")["running"] >= 1)
    assert disk_read.finish_batch("finish")["closed"] is True
    gate.hold.set()
    assert wait_for(lambda: disk_read.counters()["units_in_flight"] == 0)
    time.sleep(0.05)
    identities = pooled_identities()
    assert len(identities) == len(set(identities))


def test_queue_depth_returns_to_zero_after_a_concurrent_batch(tmp_path):
    """The depth counter has N authors now, so its decrement is under the lock."""
    paths = paths_for(tmp_path, 20, groups=5)
    assert disk_read.begin_batch("depth", len(paths))["ok"] is True
    issue("depth", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 20)
    counters = disk_read.counters()
    assert counters["queue_depth"] == 0
    assert counters["queue_depth_peak"] >= 1
    disk_read.finish_batch("depth")


def test_a_poison_queue_item_never_kills_a_reader(tmp_path):
    """A reader that dies on one bad item stops serving every later one, and the only
    clue would be `thread_dead` on the next bind. Drop it, stay alive, count it.

    The 0.2.23 loop did die here: `_handle_read` raised on the non-dict, its own error
    handler called `_settle`, and `_settle` raised again indexing a string - an
    unhandled exception that ended the thread. With N readers the same poison costs N
    readers, so the guard belongs in the loop and in `_settle`.
    """
    pending = disk_read._get_queue()
    pending.put_nowait("not a job")
    paths = paths_for(tmp_path, 2)
    assert disk_read.begin_batch("poison", len(paths))["ok"] is True
    issue("poison", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 2), \
        "the reader thread died on the poison item instead of dropping it"
    assert disk_read.thread_alive() is True
    assert disk_read.counters()["errors"] >= 1
    assert disk_read.batch_status("poison")["done"] == 2
    disk_read.finish_batch("poison")


# --- 6: the ordering question, measured ---------------------------------------

class _Ordered:
    """A `_read_file` stand-in that makes a LOW job id the SLOW one - which is what
    concurrency does to a batch on its own, without help - and records the module's
    own dispatch state at the instant each read begins.

    That recording is how the FIFO bound is checked WITHOUT inferring it from timing:
    `job_states` is read after this job is already RUNNING, so any earlier-submitted
    job still marked QUEUED at that moment would be a job this one jumped over in the
    queue. With a FIFO `queue.Queue` that set is empty by construction, and the test
    asserts it is empty rather than asserting a sleep-derived guess.
    """

    def __init__(self, batch_id, slow_first=8):
        self.batch_id = batch_id
        self.slow_first = slow_first
        self.lock = threading.Lock()
        self.started: list = []
        self.finished: list = []
        self.queued_earlier: dict = {}

    def __call__(self, path, buffer, size):
        job_id = _job_id_of(path)
        states = disk_read.job_states(self.batch_id)["states"]
        jumped = sorted(int(other) for other, state in states.items()
                        if state == disk_read.QUEUED and int(other) < job_id)
        with self.lock:
            self.started.append(job_id)
            self.queued_earlier[job_id] = jumped
        time.sleep(0.004 * max(1, self.slow_first - job_id))
        with self.lock:
            self.finished.append(job_id)
        return disk_read._ok(bytes=size, buffer=buffer)


def test_completion_order_under_four_threads_and_the_fifo_bound(monkeypatch,
                                                                tmp_path):
    """THE EVIDENCE the partial-restore analysis needs, produced by real threads.

    Three claims, all pinned here:
      * completions DO interleave: `completion_order` is not the submission order and
        `out_of_order_completions` counts the inversions;
      * a job never jumps the QUEUE: no job starts while an earlier-submitted job is
        still QUEUED, so the disorder comes from completion, never from dispatch;
      * `settled` still means the same thing, so concurrency cannot make a batch look
        finished earlier than it is.
    """
    threads = 4
    monkeypatch.setenv(config.ENV_DISK_READ_THREADS, str(threads))
    ordered = _Ordered("order", slow_first=8)
    monkeypatch.setattr(disk_read, "_read_file", ordered)
    paths = paths_for(tmp_path, 8, groups=8)   # group field == job id, for the probe
    assert disk_read.begin_batch("order", len(paths))["ok"] is True
    issue("order", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 8), \
        disk_read.batch_status("order")
    status = disk_read.batch_status("order")
    assert status["settled"] is True, "concurrency must not change what settled means"
    assert status["completion_order"] != list(range(8)), \
        "N threads completed nothing out of order: the test lost its teeth"
    assert status["out_of_order_completions"] > 0
    assert disk_read.counters()["completions_out_of_order"] >= \
        status["out_of_order_completions"]
    jumped = {job_id: earlier for job_id, earlier in ordered.queued_earlier.items()
              if earlier}
    assert jumped == {}, f"a reader overtook an earlier job IN the queue: {jumped}"
    # The same batch, seen by the restore layer: with 8 units over 2 groups a span is
    # a pair, and the unit that finished LAST can belong to the FIRST span - which is
    # the shape `spans_hole_in_the_middle` exists to count. `span_shape` measures it
    # and decides nothing: the token-counting question belongs to another change.
    shape = disk_restore.span_shape({"jobs": {
        index: {"group": index % 2} for index in range(8)}}, failed={1, 2})
    assert shape["spans"] == 4
    # Two of the four pairs come away half-done: the units of spans 0 and 1 landed on
    # opposite sides of the completion order. THIS is what a mid-flight batch looks
    # like with N readers, and it is why `spans_hole_in_the_middle` is published.
    assert shape["spans_partial"] == 2
    assert shape["partial_spans"] == [0, 1]
    assert shape["applied_is_prefix"] is False
    disk_read.finish_batch("order")


def test_a_single_thread_batch_completes_in_submission_order(monkeypatch, tmp_path):
    """The contrast that makes the test above a finding rather than a curiosity: with
    ONE reader and the same slow-first schedule, answers arrive in submission order,
    so an interrupted batch is missing a SUFFIX and never a middle."""
    ordered = _Ordered("serial", slow_first=6)
    monkeypatch.setattr(disk_read, "_read_file", ordered)
    paths = paths_for(tmp_path, 6, groups=6)
    assert disk_read.begin_batch("serial", len(paths))["ok"] is True
    issue("serial", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 6)
    status = disk_read.batch_status("serial")
    assert status["completion_order"] == list(range(6))
    assert status["out_of_order_completions"] == 0
    assert disk_read.counters()["completions_out_of_order"] == 0
    disk_read.finish_batch("serial")


# --- the span shape, as a pure function ---------------------------------------

def _jobs(count, groups):
    return {index: {"group": index % groups, "bytes": UNIT}
            for index in range(count)}


def test_span_shape_reads_a_suffix_as_the_old_boot_left_it():
    """Two whole spans, then nothing: `applied_is_prefix` True, no hole."""
    shape = disk_restore.span_shape({"jobs": _jobs(6, 3)}, failed={3, 4, 5})
    assert shape["spans"] == 2 and shape["entries"] == 6
    assert shape["spans_whole"] == 1
    assert shape["spans_empty"] == 1 and shape["spans_partial"] == 0
    assert shape["applied_is_prefix"] is True


def test_span_shape_reads_a_hole_in_the_middle():
    """Span 1 partial, span 2 whole: only concurrent readers can produce this, and
    the counter that counts it is the concurrency finding."""
    shape = disk_restore.span_shape({"jobs": _jobs(6, 3)}, failed={1})
    assert shape["spans"] == 2
    assert shape["spans_partial"] == 1 and shape["spans_whole"] == 1
    assert shape["applied_is_prefix"] is False
    assert shape["partial_spans"] == [0] and shape["truncated"] is False
    disk_restore._note_span_shape(shape)
    counters = disk_restore.counters()
    assert counters["spans_partial"] == 1
    assert counters["spans_hole_in_the_middle"] == 1
    assert counters["spans_observed"] == 2
    assert counters["last_span_shape"]["spans"] == 2


def test_a_partial_span_after_a_gap_is_a_hole_too():
    """The case `applied_is_prefix` used to get wrong: whole, empty, PARTIAL. The
    applied bytes are not a prefix of the span sequence either way - a caller that
    counted them as one would be counting tokens for a span in the middle.
    """
    shape = disk_restore.span_shape({"jobs": _jobs(6, 2)}, failed={2, 3, 5})
    assert shape["spans"] == 3
    assert shape["spans_whole"] == 1 and shape["spans_empty"] == 1
    assert shape["spans_partial"] == 1 and shape["partial_spans"] == [2]
    assert shape["applied_is_prefix"] is False
    disk_restore._note_span_shape(shape)
    assert disk_restore.counters()["spans_hole_in_the_middle"] == 1


def test_a_partial_span_first_is_still_a_prefix():
    """The shape one reader produced for free: the FIRST span came away short and
    everything after it is empty, so what was applied is a prefix and the counter for
    a hole in the middle must stay at zero - the witness distinguishes the two.
    """
    shape = disk_restore.span_shape({"jobs": _jobs(6, 2)}, failed={1, 2, 3, 4, 5})
    assert shape["spans_partial"] == 1 and shape["spans_empty"] == 2
    assert shape["spans_whole"] == 0 and shape["entries_applied"] == 1
    assert shape["applied_is_prefix"] is True
    disk_restore._note_span_shape(shape)
    assert disk_restore.counters()["spans_hole_in_the_middle"] == 0


def test_span_shape_survives_a_second_request_on_the_same_step():
    """Job ids are STEP-wide indices, not span numbers: a RAM entry (no path, no job)
    or a second request in the same step leaves holes in the id sequence, and a
    `job_id // groups` rule would split those spans in the wrong place. The run rule
    (a span ends where the group id stops advancing) does not care."""
    jobs = {}
    for index in (0, 1, 2, 5, 6, 7):      # 3 and 4 named no file
        jobs[index] = {"group": index % 3 if index < 3 else (index - 5) % 3}
    shape = disk_restore.span_shape({"jobs": jobs}, failed=set())
    assert shape["spans"] == 2, shape
    assert shape["spans_whole"] == 2
    assert shape["entries"] == 6


def test_span_shape_of_nothing_is_a_view_and_not_a_fault():
    shape = disk_restore.span_shape({"jobs": {}}, failed=set())
    assert shape["spans"] == 0 and shape["reason"]
    assert shape["applied_is_prefix"] is None
    disk_restore._note_span_shape(shape)      # counts nothing, raises nothing
    assert disk_restore.counters()["spans_observed"] == 0


def test_span_shape_names_an_entry_without_a_group_rather_than_dropping_it():
    jobs = {0: {"group": 0}, 1: {}, 2: {"group": 2}}
    shape = disk_restore.span_shape({"jobs": jobs}, failed=set())
    assert shape["entries"] == 3 and shape["spans"] >= 1
    assert "named no group" in (shape["reason"] or "")


def test_span_shape_reports_only_a_bounded_number_of_partial_spans():
    jobs = {}
    for span in range(40):
        for group in range(3):
            jobs[span * 3 + group] = {"group": group}
    failed = {span * 3 + 1 for span in range(40)}
    shape = disk_restore.span_shape({"jobs": jobs}, failed=failed)
    assert shape["spans_partial"] == 40
    assert len(shape["partial_spans"]) == disk_restore.MAX_SPANS_REPORT
    assert shape["truncated"] is True


def test_the_single_eligible_group_shape_is_the_live_one():
    """The shape the RUNNING boot actually produces, pinned so the witness is not
    read through the wrong lens.

    `per_block` on the live box is 1 (only g5 is eligible), so every job in a batch
    names the SAME group and the span walk makes one span per entry. That is not a
    degenerate failure of the walk - with one group per span, a span IS one file - but
    it does mean `spans_partial` can never fire there, and what the counters describe
    is "which UNITS of the offer landed". The prefix question survives intact: with a
    unit missing in the MIDDLE the applied bytes are not a prefix of the offer, which
    is exactly the case the token-counting analysis needs named.
    """
    one_group = {index: {"group": 5} for index in range(10)}
    shape = disk_restore.span_shape({"jobs": one_group}, failed=set())
    assert shape["spans"] == 10 and shape["entries"] == 10
    assert shape["spans_whole"] == 10 and shape["spans_partial"] == 0
    assert shape["applied_is_prefix"] is True

    # A suffix missing (what one reader produced, and what a `buffer_cap` refusal at
    # the END of an offer still produces): a prefix, and no hole counted.
    suffix = disk_restore.span_shape({"jobs": one_group}, failed={8, 9})
    assert suffix["spans_whole"] == 8 and suffix["spans_empty"] == 2
    assert suffix["applied_is_prefix"] is True

    # A unit missing in the middle: the shape that makes "count the tokens anyway"
    # wrong, and with 10 units offered and 8 landing it is the LIKELY shape on the
    # live boot, because the refusal happens at issue in id order but the forfeit
    # rule takes the whole batch's entries with it.
    middle = disk_restore.span_shape({"jobs": one_group}, failed={3})
    assert middle["spans_whole"] == 9 and middle["spans_empty"] == 1
    assert middle["spans_partial"] == 0, "one group per span: a span cannot be partial"
    assert middle["applied_is_prefix"] is False
    disk_restore._note_span_shape(middle)
    assert disk_restore.counters()["spans_hole_in_the_middle"] == 1

    # And the multi-group shape the same counters produce the day g0-g3 return: then a
    # span really is several files, and `spans_partial` has work to do.
    five_groups = {index: {"group": index % 5} for index in range(10)}
    both = disk_restore.span_shape({"jobs": five_groups}, failed={1, 2, 7, 8})
    assert both["spans"] == 2 and both["spans_partial"] == 2
    assert both["partial_spans"] == [0, 1]


def test_the_shape_is_published_on_the_restore_section():
    section = disk_restore.section()
    assert "spans" in section
    note = section["spans"]["note"]
    assert "prefix" in note and "one reader" in note
    assert section["spans"]["observed"] == 0


# --- the published witnesses --------------------------------------------------

def test_every_concurrency_witness_is_published_and_named(tmp_path):
    """Each field exists to answer one question; a field that is missing answers none.

    Enumerated here so a rename has to be a decision, not a slip: `disk_restore`
    embeds this section into the rank file and the proxy reader parses the names.
    """
    paths = paths_for(tmp_path, 3)
    assert disk_read.begin_batch("views", len(paths))["ok"] is True
    issue("views", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 3)
    section = disk_read.section()
    for key in ("threads", "threads_configured", "threads_alive",
                "threads_requested", "threads_clamped", "throughput", "pool",
                "concurrency", "units_in_flight", "units_in_flight_peak",
                "reader_starved_polls", "completions_recorded",
                "completions_out_of_order", "env_problems"):
        assert key in section, f"disk_read.section() lost {key!r}"
    for field in ("index", "name", "alive", "jobs_read", "bytes_read",
                  "read_seconds", "in_flight", "peak_in_flight",
                  "starved_polls", "errors", "spawned"):
        assert field in section["threads"][0], f"per-thread view lost {field!r}"
    for field in ("bytes_per_reader_second", "bytes_per_second_wall",
                  "parallel_efficiency", "wall_seconds", "reader_seconds"):
        assert field in section["throughput"], f"throughput view lost {field!r}"
    for field in ("limit_bytes", "held_bytes", "peak_bytes", "headroom_bytes",
                  "units_of_11468800", "refused_for_no_buffer", "clamped"):
        assert field in section["pool"], f"pool view lost {field!r}"
    for field in ("threads_configured", "threads_alive", "units_in_flight_peak",
                  "reader_starved_polls", "starved_on", "queue_depth_peak"):
        assert field in section["concurrency"]
    assert section["throughput"]["bytes_read"] == 3 * UNIT
    assert section["throughput"]["bytes_per_reader_second"] > 0
    assert section["concurrency"]["starved_on"] in (
        "buffers", "queue", "disk-or-threads-are-idle")
    for key in ("threads_alive", "units_in_flight_peak", "reader_starved_polls",
                "completions_out_of_order"):
        assert key in section["units"], f"`units` has no entry for {key!r}"
    disk_read.finish_batch("views")


def test_the_restore_payload_carries_the_reader_concurrency(tmp_path):
    """The rank file is the only way the proxy host sees a worker, so the witnesses
    have to be inside `disk_restore.section()` and not only in the module."""
    paths = paths_for(tmp_path, 2)
    assert disk_read.begin_batch("payload", len(paths))["ok"] is True
    issue("payload", paths)
    assert wait_for(lambda: disk_read.counters()["reads_completed"] == 2)
    view = disk_restore.section()
    assert view["read"]["threads_configured"] == 1
    assert view["read"]["caps"]["read_threads_default"] == 1
    assert "spans" in view


def test_the_rank_file_reports_concurrency_and_shape_under_its_byte_cap():
    """The worker's own report is the only worker-side witness that reaches the proxy
    host, and it lives in a byte-capped file: a section that overflows is REPLACED by
    the minimal shape, which would silently lose everything added here. So the keys
    and the headroom are both pinned.
    """
    report = disk_restore.worker_report()
    assert report["readers"]["configured"] == 1
    assert report["readers"]["peak_units"] == 0
    assert report["readers"]["buffer_refusals"] == 0
    assert report["spans"] == {"seen": 0, "whole": 0, "partial": 0, "empty": 0,
                                "hole_in_middle": 0}
    assert report.get("truncated") is None, report
    raw = json.dumps(report, default=str)
    assert len(raw) < worker_diag.MAX_CONTRIBUTION_BYTES, \
        f"the disk_restore contribution is {len(raw)} bytes of " \
        f"{worker_diag.MAX_CONTRIBUTION_BYTES}"
