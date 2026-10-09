r"""THE SHARED CEILING IS SPLIT IN BLOCKS, NOT IN BYTES (0.2.58).

WHY THIS FILE EXISTS, IN THE OPERATOR'S MEASUREMENT. On the live box, four ranks under
one 206,158,430,208-byte spill ceiling each reported `ceiling_share_bytes` of
51,539,607,552 - an equal BYTE share. The ranks' block widths are not equal: the PP0
ranks spill 11,468,800-byte blocks and the PP1 ranks 12,902,400-byte ones. One byte
budget over two different widths is two different budgets in the only currency a restore
spends:

    PP0:  51,539,607,552 // 11,468,800  =  4,493 blocks
    PP1:  51,539,607,552 // 12,902,400  =  3,994 blocks

A disk restore is only servable when the block is present on EVERY rank, so the useful
capacity of the store is the MINIMUM across ranks and everything above it is disk space
holding something that can never be served: 499 blocks per PP0 rank, 12.5% of what those
two ranks were keeping. The measured retention-vs-claims ratio (PP0 0.886 / PP1 0.776)
points the same way, which is why the mechanism is believed.

THE FIX IS ONE FORMULA, AND THE TEST THAT CATCHES IT NEEDS TWO DIFFERENT WIDTHS.
`blocks = ceiling // sum(widths)` then `share_i = blocks * width_i`: the floor is taken
ONCE, on the quantity every rank must agree on, so the four shares hold the same number
of blocks by construction and their sum still fits under the ceiling. THE REASON THIS
SURVIVED EVERY EARLIER TEST IS THE FIXTURE: with one equal width shared by every rank,
equal bytes and equal blocks are the same arithmetic and no assertion can tell them
apart. Every number here is the live box's own.

WHAT IS NOT TESTED HERE: the delivery path (`disk_gpu_write`) or the prune logic in
`disk_restore` - this change touches neither, and the diffs are meant to stay separable.
"""

from __future__ import annotations

import json
import time

import pytest

from kvtier_dev_plugin import (
    config,
    disk_spill,
    disk_store,
    worker_diag,
)

# --- the live box's geometry, verbatim ---------------------------------------

WIDTH_PP0 = 11_468_800           # bytes in ONE PP0 spill file
WIDTH_PP1 = 12_902_400           # bytes in ONE PP1 spill file
SHARE_BYTES_OLD = 51_539_607_552  # what every rank was handed on 2026-10-27
CEILING = 4 * SHARE_BYTES_OLD     # 206,158,430,208: the configured ceiling

WIDTHS = {"PP0_TP0": WIDTH_PP0, "PP0_TP1": WIDTH_PP0,
          "PP1_TP0": WIDTH_PP1, "PP1_TP1": WIDTH_PP1}
# The two counts the equal-byte split produced, computed here so the test that names
# them cannot drift from the arithmetic it claims to describe.
BLOCKS_PP0_OLD = SHARE_BYTES_OLD // WIDTH_PP0     # 4,493
BLOCKS_PP1_OLD = SHARE_BYTES_OLD // WIDTH_PP1     # 3,994

# The whole-block budget the shared ceiling affords every rank, once the widths are
# summed: 206,158,430,208 // (2x11,468,800 + 2x12,902,400).
BLOCKS_EQUAL = CEILING // (2 * WIDTH_PP0 + 2 * WIDTH_PP1)   # 4,229


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """A fresh `disk_spill`, and no arm-file or environment carry-over."""
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    monkeypatch.delenv(config.ENV_DISK_BYTES, raising=False)
    disk_spill.reset_state_for_tests()
    yield
    disk_spill.reset_state_for_tests()


def share_for(rank: str, widths=WIDTHS, ceiling: int = CEILING) -> dict:
    """THE SHARE ONE RANK GETS, with every live sibling's width in hand.

    The same call `_run_ceiling_pass` makes: its own width (fingerprint) plus one width
    per live peer (their rank files), and `divisor = 1 + peers` exactly as
    `_attribute_walk_bytes` computes it.
    """
    mine = widths[rank]
    peers = tuple(widths[other] for other in widths if other != rank)
    return disk_spill._share_of_ceiling(ceiling, 1 + len(peers),
                                        disk_spill.SHARE_SOURCE_PEERS,
                                        own_width=mine, peer_widths=peers)


# --- the defect, with the fixture that can actually see it -------------------

def test_equal_bytes_are_unequal_blocks_at_the_live_widths():
    """THE FIXTURE CAN SEE THE BUG - the property every earlier fixture lacked.

    With one width shared by four ranks, `ceiling / 4` and `blocks / 4` are the same
    number and no assertion distinguishes them; that is how the equal-byte split passed
    2,725 tests. With the box's two real widths, the old arithmetic is provably a
    different budget per rank.
    """
    assert WIDTH_PP0 != WIDTH_PP1
    equal_bytes = disk_spill._share_of_ceiling(CEILING, 4,
                                               disk_spill.SHARE_SOURCE_PEERS)
    assert equal_bytes["share_bytes"] == SHARE_BYTES_OLD, equal_bytes
    old_blocks = {
        "PP0": SHARE_BYTES_OLD // WIDTH_PP0,
        "PP1": SHARE_BYTES_OLD // WIDTH_PP1,
    }
    assert old_blocks == {"PP0": 4493, "PP1": 3994}, old_blocks
    # THE DEFECT IN ONE LINE: the same byte budget is a different cache depth per rank,
    # and the deeper one is the unservable half.
    assert old_blocks["PP0"] != old_blocks["PP1"]
    surplus = old_blocks["PP0"] - old_blocks["PP1"]
    assert surplus == 499, surplus
    # The operator's retention measurement (PP0 0.886 / PP1 0.776) corroborates it: the
    # block-count ratio is the same direction and within ~1.5% of the measured one.
    assert abs((old_blocks["PP1"] / old_blocks["PP0"]) - 0.889) < 0.001


def test_every_rank_holds_the_same_number_of_blocks_under_the_real_widths():
    """THE ASSERTION THE DEFECT FAILS: four ranks, ONE block count.

    Before the fix this fails on the first assertion of the PP1 rank - its share came
    back as 3,994 blocks while PP0 held 4,493, from the same byte budget.
    """
    shares = {rank: share_for(rank) for rank in WIDTHS}
    blocks = {rank: share["share_blocks"] for rank, share in shares.items()}
    assert blocks == {rank: BLOCKS_EQUAL for rank in WIDTHS}, blocks
    # One block count means one cache depth, which is the only depth a restore can use.
    assert len(set(blocks.values())) == 1, blocks
    for rank, share in shares.items():
        assert share["share_basis"] == disk_spill.SHARE_BASIS_BLOCK_EQUAL, share
        # The budget is a WHOLE number of this rank's own blocks: no ragged tail that a
        # pass can neither keep nor half-write.
        assert share["share_bytes"] % WIDTHS[rank] == 0, share
        assert share["share_bytes"] == BLOCKS_EQUAL * WIDTHS[rank], share
        assert share["peer_widths_known"] == 3, share


def test_the_block_equal_shares_still_fit_under_one_ceiling():
    """Weighting must not print capacity: `sum(shares) <= ceiling`, always.

    The floor is taken on the shared block count, so the four shares together are one
    block-count times the sum of the widths - at most the ceiling, never more.
    """
    shares = [share_for(rank) for rank in WIDTHS]
    total = sum(int(share["share_bytes"]) for share in shares)
    assert total <= CEILING, (total, CEILING)
    # The residue is smaller than one round of blocks, i.e. the split is tight and not
    # merely safe: the unspent bytes are what the floor could not place.
    assert CEILING - total < 2 * WIDTH_PP0 + 2 * WIDTH_PP1, CEILING - total
    # Every rank's own target moves with its share, and stays inside the share.
    for share in shares:
        assert 0 < share["target_bytes"] <= share["share_bytes"], share


def test_the_fix_trades_the_unservable_surplus_for_real_depth():
    """WHAT THE FIX BUYS, stated as blocks a restore can actually serve.

    PP0 gives up 264 blocks it could never serve; PP1 gains 235 it could not afford.
    The store's servable depth - the MINIMUM across ranks - rises from 3,994 to 4,229
    blocks at the SAME ceiling. This is the entire point of the change, so it is pinned
    as a number rather than left as a comment.
    """
    shares = {rank: share_for(rank) for rank in WIDTHS}
    depth_before = min(BLOCKS_PP0_OLD, BLOCKS_PP1_OLD)
    depth_after = min(int(share["share_blocks"]) for share in shares.values())
    assert depth_before == 3994 and depth_after == 4229, (depth_before, depth_after)
    assert shares["PP0_TP0"]["share_bytes"] < SHARE_BYTES_OLD   # PP0 sheds surplus
    assert shares["PP1_TP0"]["share_bytes"] > SHARE_BYTES_OLD   # PP1 affords real depth
    assert shares["PP0_TP0"]["share_blocks"] - BLOCKS_PP0_OLD == -264
    assert shares["PP1_TP0"]["share_blocks"] - BLOCKS_PP1_OLD == 235


# --- the fail-safe: no width, no guess ---------------------------------------

def test_a_missing_peer_width_falls_back_to_equal_bytes_and_says_so():
    """An unknown width is NOT licensed to guess, and the reader is told which ran.

    A sum that is missing a rank is SMALLER than the truth, which makes every share too
    BIG - the over-eviction direction, the one this module has repeatedly been burned
    by. So the split reverts to the equal-byte arithmetic and `share_basis` says so.
    """
    share = disk_spill._share_of_ceiling(
        CEILING, 4, disk_spill.SHARE_SOURCE_PEERS,
        own_width=WIDTH_PP0, peer_widths=(WIDTH_PP0, WIDTH_PP1, None))
    assert share["share_basis"] == disk_spill.SHARE_BASIS_EQUAL_BYTES, share
    assert share["share_bytes"] == SHARE_BYTES_OLD, share     # the pre-0.2.58 number
    assert share["peer_widths_known"] == 0, share
    # The block count is still DERIVED where a width is known, because that is exactly
    # how the asymmetry was spotted from outside: the field that shows the imbalance is
    # more useful than one that hides it during the fallback.
    assert share["share_blocks"] == BLOCKS_PP0_OLD, share

    no_own_width = disk_spill._share_of_ceiling(
        CEILING, 4, disk_spill.SHARE_SOURCE_PEERS,
        own_width=None, peer_widths=(WIDTH_PP0, WIDTH_PP1, WIDTH_PP1))
    assert no_own_width["share_basis"] == disk_spill.SHARE_BASIS_EQUAL_BYTES
    # No width at all means no block claim: None, never 0 (a 0 a reader will multiply).
    assert no_own_width["share_blocks"] is None, no_own_width


def test_one_rank_or_a_tiny_ceiling_keep_their_promises():
    """The two documented fail-safe promises, re-pinned against the new arithmetic.

    `divisor == 1` returns the WHOLE ceiling byte for byte - a lone rank must shed like
    every release before it - and a ceiling narrower than the rank count still yields a
    1-byte budget rather than a 0 that means "delete the store".
    """
    solo = disk_spill._share_of_ceiling(CEILING, 1,
                                        disk_spill.SHARE_SOURCE_NO_PEERS,
                                        own_width=WIDTH_PP0, peer_widths=())
    assert solo["share_bytes"] == CEILING, solo
    assert solo["share_basis"] == disk_spill.SHARE_BASIS_EQUAL_BYTES, solo
    # A ceiling that cannot hold one block per rank never enters the block path.
    tiny = disk_spill._share_of_ceiling(48_000_000, 4,
                                        disk_spill.SHARE_SOURCE_PEERS,
                                        own_width=WIDTH_PP0,
                                        peer_widths=(WIDTH_PP0, WIDTH_PP1,
                                                     WIDTH_PP1))
    assert tiny["share_basis"] == disk_spill.SHARE_BASIS_EQUAL_BYTES, tiny
    assert tiny["share_bytes"] >= 1, tiny
    # A width that is not a sane number is refused rather than multiplied.
    junk = disk_spill._share_of_ceiling(CEILING, 4,
                                        disk_spill.SHARE_SOURCE_PEERS,
                                        own_width=0, peer_widths=(1, 2, 3))
    assert junk["share_basis"] == disk_spill.SHARE_BASIS_EQUAL_BYTES, junk
    assert junk["own_width_bytes"] is None, junk


# --- the widths have to ARRIVE: the rank file is the only source -------------

@pytest.fixture
def diag_dir(tmp_path, monkeypatch):
    directory = tmp_path / "diag"
    directory.mkdir()
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(directory))
    disk_store.reset_rank_rollup_cache_for_tests()
    yield directory
    disk_store.reset_rank_rollup_cache_for_tests()


def rank_file(directory, rank, *, digest, width, disk=None, ts=None) -> None:
    """A rank file as `worker_diag` writes it: the width lives in `arena`, NOT `disk`.

    The key path is the point. `disk` carries no width at all, so a reader that asks for
    `disk.bytes_per_block` finds nothing and concludes the box publishes no widths;
    `arena.bytes_per_block` is what `worker_diag.build_document` measures off the rank's
    own region tensors.
    """
    document = {"version": worker_diag.SCHEMA_VERSION, "epoch": "aabbccdd",
                "ts": round(time.time(), 6) if ts is None else ts,
                "role": "worker", "rank": rank, "error": None,
                "arena": {"bytes_total": 0, "bytes_used": 0, "num_blocks": 0},
                "disk": dict(disk or {}, section="disk", digest=digest)}
    if width is not None:
        document["arena"]["bytes_per_block"] = width
    (directory / f"rank-{rank}.json").write_text(json.dumps(document),
                                                 encoding="utf-8")


def test_the_peer_width_is_read_from_the_rank_file_it_already_opened(diag_dir):
    """`_peer_identity_view` learns widths from the SAME files it reads digests from.

    No extra open, no extra read: one parse of one document answers who is alive, what
    digest they write under, and how wide their blocks are.
    """
    disk_spill._state["geometry_digest"] = "a" * 12
    rank_file(diag_dir, "PP0_TP0", digest="b" * 12, width=WIDTH_PP0)
    rank_file(diag_dir, "PP0_TP1", digest="c" * 12, width=WIDTH_PP0)
    rank_file(diag_dir, "PP1_TP0", digest="d" * 12, width=WIDTH_PP1)
    rank_file(diag_dir, "PP1_TP1", digest="e" * 12, width=WIDTH_PP1)
    view = disk_spill._peer_identity_view(force=True)
    assert view["available"] is True and view["known"] is True, view["reason"]
    assert view["peer_widths"] == {"b" * 12: WIDTH_PP0, "c" * 12: WIDTH_PP0,
                                   "d" * 12: WIDTH_PP1, "e" * 12: WIDTH_PP1}, \
        view["peer_widths"]
    # In PEER ORDER, so the tuple lines up with the digest list that set `divisor`.
    widths = disk_spill._peer_widths_for_share(("d" * 12, "b" * 12))
    assert widths == (WIDTH_PP1, WIDTH_PP0), widths
    # A live peer whose file published NO width is a reason not to weight at all.
    assert disk_spill._peer_widths_for_share(("d" * 12, "f" * 12)) is None


def test_a_stale_rank_files_width_is_not_a_width(diag_dir):
    """A width from a file too old to vouch for its rank is not a width.

    The freshness rule is the digest's rule, applied to the width by the same read:
    summing a dead boot's geometry into a live share would size today's budget with
    yesterday's model.
    """
    disk_spill._state["geometry_digest"] = "a" * 12
    rank_file(diag_dir, "PP1_TP1", digest="e" * 12, width=WIDTH_PP1,
              ts=time.time() - disk_spill.PEER_RANK_FILE_FRESH_SECONDS - 10)
    view = disk_spill._peer_identity_view(force=True)
    assert view["peer_widths"] == {}, view["peer_widths"]
    assert view["files_stale"] == 1, view


# --- publish it: the equality must be checkable from OUTSIDE -----------------

def test_the_rank_file_and_the_fleet_view_both_carry_the_block_count(diag_dir):
    """`ceiling_share_blocks` reaches `/snapshot` without the reader knowing geometry.

    Four rank files with equal BYTES and unequal BLOCKS - the state the box was in -
    must be visible as inequality from the polled block, not reconstructed by hand.
    """
    sections = {}
    for rank, width in WIDTHS.items():
        blocks = SHARE_BYTES_OLD // width
        sections[rank] = blocks
        document = {"version": worker_diag.SCHEMA_VERSION, "epoch": "aabbccdd",
                    "ts": round(time.time(), 6), "role": "worker", "rank": rank,
                    "error": None,
                    "disk": {"section": "disk", "writer": "kvtier-dev-plugin/x",
                             "installed": True, "armed": True,
                             "ceiling_bytes": CEILING,
                             "ceiling_share_bytes": SHARE_BYTES_OLD,
                             "ceiling_share_blocks": blocks,
                             "share_basis": disk_spill.SHARE_BASIS_EQUAL_BYTES,
                             "ceiling_divisor": 4,
                             "ceiling_passes_declined": 0,
                             "files_written": 10, "bytes_written": 10 * width,
                             "evictions": 0, "journal_removals": 0, "drops": 0,
                             "spill_queued": 0, "spill_errors": 0,
                             "queue_depth": 0, "writer_thread_alive": True,
                             "estimated_bytes": 1000,
                             "ceiling_measurement_basis": "full_measurement"}}
        (diag_dir / f"rank-{rank}.json").write_text(json.dumps(document),
                                                    encoding="utf-8")
    fill = disk_store.disk_fill()
    per_rank = {row["rank"]: row for row in fill["per_rank"]}
    assert len(per_rank) == 4, per_rank
    for rank, row in per_rank.items():
        assert row["ceiling_share_blocks"] == sections[rank], (rank, row)
        assert row["ceiling_share_bytes"] == SHARE_BYTES_OLD, (rank, row)
        assert row["share_basis"] == disk_spill.SHARE_BASIS_EQUAL_BYTES, row
    verdict = fill["cross_process_verdict"]
    # THE READER'S QUESTION, ANSWERED: the bytes agree, the blocks do not - and since
    # 0.2.58 the verdict keys on the blocks, so the unequal-depth state the box was in
    # is REPORTED rather than mistaken for agreement.
    assert verdict["verdict"] == "ranks_disagree_on_the_share", verdict
    assert verdict["share_blocks_values"] == [BLOCKS_PP1_OLD, BLOCKS_PP0_OLD], verdict
    assert verdict["blocks_equal"] is False, verdict
    assert verdict["share_bytes_values"] == [SHARE_BYTES_OLD], verdict
    # And once the ranks agree on blocks, the same fields say so - the fix's own
    # witness, readable from the same place - even though their byte shares now differ,
    # which is what a depth-equal split on unequal geometries MUST look like.
    for rank in WIDTHS:
        path = diag_dir / f"rank-{rank}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["ts"] = round(time.time(), 6)
        document["disk"]["ceiling_share_blocks"] = BLOCKS_EQUAL
        document["disk"]["ceiling_share_bytes"] = BLOCKS_EQUAL * WIDTHS[rank]
        document["disk"]["share_basis"] = disk_spill.SHARE_BASIS_BLOCK_EQUAL
        path.write_text(json.dumps(document), encoding="utf-8")
    disk_store.reset_rank_rollup_cache_for_tests()
    after = disk_store.disk_fill()
    verdict = after["cross_process_verdict"]
    assert verdict["verdict"] == "shared_share", verdict
    assert verdict["blocks_equal"] is True, verdict
    assert verdict["share_blocks"] == BLOCKS_EQUAL, verdict
    assert verdict["share_basis"] == disk_spill.SHARE_BASIS_BLOCK_EQUAL, verdict
    # The bytes now DISAGREE, and that is correct and no longer an alarm: equal depth on
    # two widths is two byte budgets. `share_bytes` is null for exactly that reason.
    assert len(verdict["share_bytes_values"]) == 2, verdict
    assert verdict["share_bytes"] is None, verdict


def test_the_section_publishes_the_share_basis_for_this_process():
    """`disk_spill.section()` carries the same verdict for the process answering.

    A per-process view and a fleet-wide view that disagree about which arithmetic ran is
    how two dashboards start telling two stories about one box.
    """
    disk_spill._apply_attribution(
        {"attributed": True, "own_bytes": 1, "dead_bytes": 0, "peer_bytes": 0,
         "unattributed_bytes": 0, "accountable_bytes": 1,
         "peers": ("b" * 12, "c" * 12, "d" * 12), "divisor": 4,
         "source": disk_spill.SHARE_SOURCE_PEERS, "sheddable_entries": [],
         "peer_entries": 0, "unattributable_entries": 0,
         "unattributable_bytes": 0},
        1,
        share_for("PP1_TP0"))
    block = disk_spill.section()["ceiling_share"]
    assert block["share_blocks"] == BLOCKS_EQUAL, block
    assert block["own_width_bytes"] == WIDTH_PP1, block
    assert block["peer_widths_known"] == 3, block
    assert block["share_basis"] == disk_spill.SHARE_BASIS_BLOCK_EQUAL, block
    # Every new field is declared with a unit, or the payload ships it dimensionless.
    units = block["units"]
    for name in ("share_blocks", "own_width_bytes", "peer_widths_known",
                 "share_basis"):
        assert units[name] != "unknown", (name, units[name])
