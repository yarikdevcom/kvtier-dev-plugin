r"""THE ARENA THE DISK TIER HAS TO MOVE BYTES THROUGH, asked at boot.

The disk tier owns no KV bytes. Spill reads them OUT of the host-RAM arena's pinned
CPU tensors (`disk_spill.read_slot_bytes` over the worker's `cpu_kv_caches`) and is
TRIGGERED by the RAM tier's own store-completion events; restore allocates its
destination slots FROM the arena's CPU block pool - the same free count that clamps an
offer at `no_cpu_slots` - and memmoves into them before the fork DMAs them to GPU. So
an arena that is absent, or clamps to a slot or two per rank (a mis-set
`KV_OFFLOAD_SIZE`, a missing `VLLM_USE_SIMPLE_KV_OFFLOAD`, the fork's own `max(1, ...)`
clamp), leaves the tier STRUCTURALLY INERT while its arm file still says ARMED.

That combination produced a healthy-looking boot with a dead tier, and nothing detected
it: `disk_tier=ARMED` was, literally, true and useless. So the boot line gained an
`arena=` field beside it and a refusal line naming the arena, and these tests pin both
directions - including the one that matters most to the operator, that a `TINY` arena
is reported as the reason rather than being silently armed.

Two rules the tests hold the code to:

* THE FIELD NEVER LIES ABOUT A THING IT CANNOT SEE. EngineCore has no KV tensors, so
  the worker's region tensors are not this function's to judge; their absence is
  reported where it is known (`fingerprint_incomplete`, `no_worker_proof`) and here
  reads `UNSEEN`.
* THE EXISTING BOOT-LINE CONTRACT IS UNTOUCHED. `tier=` keeps its position and value
  (the restart verifier in the llm-api repo takes the LAST `tier=`-shaped token), and
  `disk_tier=` keeps meaning one thing: the spill tier's arm.

No engine, no torch, no CUDA, no network, no live box.
"""

from __future__ import annotations

import re

import fixtures_pr56810 as PR  # noqa: F401 - imported for its module-level setup
import test_auto_boot as AB
import test_backport as TB

from kvtier_dev_plugin import backport, disk_restore, disk_spill

# The boot harness lives in the sibling module: its autouse fixture saves and restores
# the process-wide arm state the boot writes, which is exactly what a test that boots
# must not leak.
boot_state = AB.boot_state  # noqa: F811 - re-exported as a fixture, not shadowed

RESERVE = disk_restore.CPU_SLOT_RESERVE


class Pool:
    """`cpu_block_pool`: a free count and a block list, with two failure shapes."""

    def __init__(self, free=1024, blocks=1024, raises=False, unreadable=None):
        self._free = free
        self.blocks = [object() for _ in range(blocks)]
        self._raises = raises
        self._unreadable = unreadable

    def get_num_free_blocks(self):
        if self._raises:
            raise RuntimeError("the pool is not readable")
        return self._unreadable if self._unreadable is not None else self._free


def manager_with(pool):
    """The smallest object the arena verdict reads: a `cpu_block_pool` by name."""
    return type("Mgr", (), {"cpu_block_pool": pool})()


# --- 1. the verdict itself ------------------------------------------------------

def test_an_arena_whose_reserve_eats_everything_is_tiny_and_names_the_knobs():
    """THE OPERATOR'S CASE: an arena that clamps to a slot or two per rank.

    A free count equal to the reserve leaves ZERO blocks for an offer, which is why
    every offer dies at `no_cpu_slots` on such a boot. The reason must name the two
    environment knobs that produce it, because "the arena is small" is not an action.
    """
    verdict = disk_restore.arena_verdict(manager_with(Pool(free=RESERVE,
                                                           blocks=RESERVE)))
    assert verdict["state"] == "tiny"
    assert verdict["free_blocks"] == RESERVE
    assert verdict["blocks_usable_for_an_offer"] == 0
    assert verdict["cpu_slot_reserve"] == RESERVE
    assert "KV_OFFLOAD_SIZE" in verdict["reason"]
    assert "VLLM_USE_SIMPLE_KV_OFFLOAD" in verdict["reason"]
    assert "no_cpu_slots" in verdict["reason"]


def test_a_manager_that_exposes_no_cpu_block_pool_blames_the_arena_by_name():
    """No pool, no arena, no byte moved - and the reason says so in those words.

    This is the shape a missing `VLLM_USE_SIMPLE_KV_OFFLOAD` leaves behind: a connector
    with no host-RAM tier at all. The spill tier's arm is a FILE and says ARMED
    regardless, which is precisely the contradiction this verdict exists to break.
    """
    verdict = disk_restore.arena_verdict(manager_with(None))
    assert verdict["state"] == "no_pool"
    assert "cpu_block_pool" in verdict["reason"]
    assert "cannot move one byte" in verdict["reason"]
    assert verdict["free_blocks"] is None


def test_an_unreadable_free_count_is_unreadable_and_never_reads_as_ok():
    """A verdict that cannot be taken is a THIRD answer, never a silent `ok`.

    Both shapes: a pool that raises, and a pool that answers something that is not a
    number. Either way an offer cannot be priced against it, and a boot line that said
    `arena=OK` here would be the field lying.
    """
    raised = disk_restore.arena_verdict(manager_with(Pool(raises=True)))
    assert raised["state"] == "unreadable"
    assert "RuntimeError" in raised["reason"]
    weird = disk_restore.arena_verdict(manager_with(Pool(unreadable="many")))
    assert weird["state"] == "unreadable"
    assert "get_num_free_blocks" in weird["reason"]


def test_a_normal_arena_says_ok_with_the_numbers_that_make_the_verdict_checkable():
    """The control: a healthy arena, and the arithmetic beside the word.

    `blocks_usable_for_an_offer` is the free count minus the reserve - the same
    subtraction the offer path makes - so a reader can verify `OK` instead of taking it
    on faith, and a boot that is one block from `TINY` is visible as a number.
    """
    verdict = disk_restore.arena_verdict(manager_with(Pool(free=1024, blocks=1024)))
    assert verdict["state"] == "ok"
    assert verdict["free_blocks"] == 1024
    assert verdict["pool_total_blocks"] == 1024
    assert verdict["blocks_usable_for_an_offer"] == 1024 - RESERVE


def test_no_manager_at_all_is_unseen_rather_than_a_verdict_about_nothing():
    """A process that was handed no connector cannot report on its arena.

    `UNSEEN` (mapped from `no_manager`) is the honest answer there: claiming `TINY`
    from an absent manager would cry wolf on every process that legitimately has no
    scheduler, which is most of them.
    """
    verdict = disk_restore.arena_verdict(None)
    assert verdict["state"] == "no_manager"
    assert backport._arena_field({"arena": verdict}) == "UNSEEN"


# --- 2. the boot line and the refusal it produces -------------------------------

def test_disk_armed_without_an_arena_reports_the_arena_as_the_reason(monkeypatch):
    """THE PIN: an armed disk tier over an arena that cannot service an offer is not a
    silent ARMED.

    The spill tier is armed (its arm file is up), the install and the guard passed, and
    the arena holds exactly the reserve and nothing more. Before this field the line
    read `disk_tier=ARMED` and the boot looked healthy while every offer died at
    `no_cpu_slots`. Now the same line carries `arena=TINY`, the refusal tail names the
    arena and the knobs that set its size, and the payload says the same thing.
    """
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    cls = TB._six_group_class()
    manager = TB._production_manager(cls)
    manager.cpu_block_pool = Pool(free=RESERVE, blocks=RESERVE)
    report = backport.install(cls, TB.our_shape_config(), manager)
    _report, boot_report, _cls, _manager = AB.boot(cls=cls, manager=manager,
                                                   report=report)
    line = boot_report["line"]
    assert boot_report["tier"] is True, "the RAM tier is armed, as the install passed"
    assert "disk_tier=ARMED" in line, line
    assert "arena=TINY" in line, line
    arena_refusals = [item for item in boot_report["refusals"]
                      if item.startswith("disk:")]
    assert len(arena_refusals) == 1, boot_report["refusals"]
    assert "KV_OFFLOAD_SIZE" in arena_refusals[0], arena_refusals[0]
    assert "no_cpu_slots" in arena_refusals[0], arena_refusals[0]
    # And the same fact is in the payload an operator reads without a grep.
    section = backport.boot_section()
    assert section["arena"]["state"] == "tiny"
    assert section["arena"]["blocks_usable_for_an_offer"] == 0


def test_a_healthy_boot_reports_arena_ok_and_adds_no_disk_refusal(monkeypatch):
    """THE OTHER DIRECTION: the new field must not cry wolf on a good boot.

    Same harness, production-sized pool: `arena=OK`, and no `disk:` line in the
    refusals. A field that fires on every boot is a field that gets ignored, and this
    is the half of the pair that keeps it worth reading.
    """
    monkeypatch.setattr(disk_spill, "is_armed", lambda: True)
    _report, boot_report, _cls, _manager = AB.boot()
    line = boot_report["line"]
    assert "arena=OK" in line, line
    assert not [item for item in boot_report["refusals"]
                if item.startswith("disk:")], boot_report["refusals"]


def test_the_arena_field_leaves_the_restart_verifier_reading_the_tier_it_always_read():
    """A new field must not steal the PASS criterion from the restart verifier.

    The verifier in the llm-api repo pulls the tier verdict with
    `grep -oE "tier=[A-Za-z_]+" | tail -1` - the LAST `tier=`-shaped token on the line.
    `arena=` does not contain `tier=`, and it is printed after the bare `tier=` anyway,
    so the extraction is unchanged; this reproduces the grep so a future field cannot
    quietly break a healthy boot's restart check.
    """
    _report, boot_report, _cls, _manager = AB.boot()
    line = boot_report["line"]
    assert re.findall(r"tier=[A-Za-z_]+", line)[-1] == "tier=ARMED", line
    assert "arena=" in line


def test_an_unreadable_arena_reports_itself_and_still_prints_a_line(monkeypatch):
    """A boot verdict never breaks a boot, and never reports `OK` by accident.

    `arena_verdict` cannot raise, so the only way to reach the wrapper's failure path is
    the import or a future accessor - and when that happens the field says
    `UNREADABLE`, the refusal tail names it, and the RAM verdict the verifier reads is
    unaffected.
    """
    def explode(_manager):
        raise RuntimeError("no arena here")

    monkeypatch.setattr(disk_restore, "arena_verdict", explode)
    verdict = backport._boot_arena(manager_with(Pool()))
    assert verdict["state"] == "unreadable"
    assert "RuntimeError" in verdict["reason"]
    line = backport.boot_line({"mode": "auto", "backport": True, "guard": True,
                               "tier": True, "faults_at_boot": 0, "nested": [],
                               "refusals": [], "not_covered": [],
                               "guard_armed": False, "arena": verdict})
    assert "arena=UNREADABLE" in line, line
    assert "tier=ARMED" in line, line


def test_a_boot_written_without_an_arena_verdict_says_unset():
    """The no-tier note (a worker, a MultiConnector) shares the format and has no pool.

    `UNSET` is a fourth answer beside OK/TINY/UNREADABLE: this line was written by a
    process that never had a connector to ask about. It must not print `arena=OK` for a
    thing it never looked at, and it must not print a refusal an operator would go and
    fix in a process that has no arena to fix.
    """
    line = backport.boot_line({"mode": "worker", "backport": False, "guard": False,
                               "tier": False, "faults_at_boot": 0, "nested": [],
                               "refusals": [], "not_covered": [],
                               "guard_armed": False})
    assert "arena=UNSET" in line, line
