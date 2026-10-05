"""`inspect`: "show me block 412" and "what does this request hold?".

Two verbs, one engine method, both bounded. They exist because the two questions
an operator actually asks about a KV cache cannot be answered from a snapshot:

* a snapshot gives counters; the questions are about ONE block ("is it free? does
  it still carry a hash? which group's hash?") and ONE request ("which blocks
  does it hold in each group, and how many of them could the tier park?").
* the answer must be bounded. A request in a 32k-token sequence with a 128-token
  block size holds 256 blocks in a group; six groups is 1536 ids. So every list is
  capped and every cap says what it dropped.

The per-group "coverable hashes" figure is the fork's own expression, not an
invention: `manager.py:820-831` computes

    confirmed = num_computed_tokens - num_output_placeholders
    aligned   = confirmed // alignment * alignment      # alignment is
               # hash_block_size when enable_partial_hash_hits else block_size
    ready     = min(len(group_gpu_block_ids), aligned // group_block_size)

and this module reproduces it, additionally capped by the number of block hashes
the tier can view at that group's size, and says WHICH term bound the answer.
"""

from __future__ import annotations

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import kvinspect

# --- helpers ----------------------------------------------------------------

def engine_with_request(blocks_per_group=None, *, groups=None, managers=None,
                        tier=True, request_id="cmpl-a", **request_fields):
    """The production geometry with `request_id` holding real blocks per group.

    `tier` is True (a default tier), False (no connector at all) or a tier object
    the test has already arranged.
    """
    tier_manager = (fake.tier_manager() if tier is True
                    else (tier if tier else None))
    if groups is None or managers is None:
        _engine, info = fake.production_geometry()
        groups, managers = info["groups"], info["managers"]
    blocks_per_group = blocks_per_group or {}
    for index, manager in enumerate(managers):
        ids = blocks_per_group.get(index, 0)
        manager.req_to_blocks = {
            request_id: [fake.Block(i, block_hash=fake.hash_key(i, index),
                                    ref_cnt=1) for i in range(ids)]}
    sched = fake.scheduler(groups=groups, managers=managers,
                           pool=fake.Pool([fake.Block(i, block_hash=fake.hash_key(i))
                                           for i in range(64)],
                                          num_gpu_blocks=8192, free=32),
                           tier=tier_manager,
                           running=[request_id],
                           requests={request_id: fake.request(request_id,
                                                              **request_fields)})
    return fake.engine(sched)


def block(engine, **args):
    return kvinspect.inspect_for_engine(engine, args)


# --- the block inspector ----------------------------------------------------

def test_a_block_by_id_and_group_answers_the_forks_own_fields():
    engine, _ = fake.production_geometry()
    payload = block(engine, block_id=5, group=0)
    assert payload["verb"] == "block"
    assert payload["found"] is True
    assert payload["requested"] == {"block_id": 5, "group": 0}
    view = payload["block"]
    assert view["block_id"] == 5
    assert view["ref_cnt"] == 1
    assert view["is_null"] is False
    # A referenced block is not on the free list, and the hash decodes: the fork
    # packs the group id into the LAST four bytes (kv_cache_utils.py:86-92).
    assert view["in_free_queue"] is False
    assert view["hash_group_id"] == 5 % 6
    assert view["hash_group_matches_request"] is (5 % 6 == 0)
    assert view["hash_bytes"] == 20
    assert view["hash_key_bytes"] == 24
    assert view["has_hash"] is True
    assert int(view["hash_hex"], 16)            # real hex, not a repr
    # The O(1) keyed lookup the tier itself uses (manager.py:940).
    assert view["in_cached_map"] is True


def test_a_free_block_reports_its_place_in_the_free_list():
    engine, _ = fake.production_geometry()
    view = block(engine, block_id=45)["block"]
    assert view["in_free_queue"] is True
    assert view["prev_free_block_id"] == 44
    assert view["next_free_block_id"] == 46
    assert view["in_cached_map"] is None        # no hash: the map cannot answer


def test_a_block_with_no_hash_is_reported_as_having_none():
    engine = fake.engine(fake.scheduler(
        pool=fake.Pool([fake.Block(0, block_hash=None)])))
    view = block(engine, block_id=0)["block"]
    assert view["has_hash"] is False
    assert "hash_hex" not in view
    assert view["in_cached_map"] is None


def test_a_hash_that_is_not_bytes_is_named_rather_than_decoded():
    engine = fake.engine(fake.scheduler(
        pool=fake.Pool([fake.Block(0, block_hash=("not", "bytes"))])))
    view = block(engine, block_id=0)["block"]
    assert view["has_hash"] is True
    assert view["hash_unrenderable"] == "tuple"
    assert "hash_hex" not in view


@pytest.mark.parametrize("block_id", [64, 4096, -1])
def test_a_block_id_that_is_not_nameable_says_why_and_names_the_range(block_id):
    engine, _ = fake.production_geometry()
    payload = block(engine, block_id=block_id)
    assert payload["found"] is False
    assert "64 block objects" in payload["reason"]
    assert "block" not in payload


def test_a_pool_that_is_not_a_list_is_a_reason_not_a_crash():
    engine = fake.engine(fake.scheduler(pool=fake.Pool()))
    engine.scheduler.kv_cache_manager.block_pool.blocks = "not a list"
    payload = block(engine, block_id=3)
    assert payload["found"] is False
    assert "block_pool.blocks" in payload["reason"]


def test_the_group_geometry_travels_with_the_block_answer():
    """A block id means nothing without the group it was asked about: ONE pool is
    shared by every group (kv_cache_coordinator), so the group's stride is what
    makes the id interpretable."""
    engine, _ = fake.production_geometry()
    payload = block(engine, block_id=2, group=4)
    geometry = payload["group_geometry"]
    assert geometry["present"] is True
    assert geometry["group"] == 4
    assert geometry["spec_class"] == "MambaSpec"
    assert geometry["block_size"] == 128
    assert geometry["hash_block_size"] == 128
    assert geometry["divisibility"]["divisible"] is True
    assert geometry["layer_count"] == 8


def test_a_group_index_that_does_not_exist_is_reported_as_absent():
    engine, _ = fake.production_geometry()
    assert block(engine, block_id=1, group=99)["group_geometry"]["present"] is False


def test_no_group_means_no_group_geometry():
    engine, _ = fake.production_geometry()
    assert block(engine, block_id=1)["group_geometry"] is None


def test_the_tier_cpu_slot_with_the_same_id_is_reported_as_a_different_pool():
    engine, info = fake.production_geometry()
    payload = block(engine, block_id=3)
    slot = payload["tier_cpu_slot"]
    assert slot["present"] is True and slot["found"] is True
    assert slot["pool"]["bound"] is True
    assert slot["block"]["block_id"] == 3
    # A CPU block id and a GPU block id are different pools and mean nothing to
    # each other; the request says so by naming the pool it came from.
    assert slot["pool"]["total_blocks"] == 64
    info["tier"].cpu_block_pool = fake.Pool([fake.Block(i) for i in range(2)])
    payload = block(engine, block_id=3)
    assert payload["tier_cpu_slot"]["found"] is False
    assert "not nameable in the CPU pool" in payload["tier_cpu_slot"]["reason"]


def test_no_tier_means_no_cpu_slot_key_at_all():
    engine = engine_with_request({0: 4}, tier=False)
    assert "tier_cpu_slot" not in block(engine, block_id=1)


def test_a_pool_that_raises_on_a_getter_is_data(reset_snapshot_state):
    pool = fake.Pool([fake.Block(0)], raise_on_free=True)
    engine = fake.engine(fake.scheduler(pool=pool))
    payload = block(engine, block_id=0)
    assert payload["pool"]["free_blocks"] is None
    assert payload["found"] is True
    assert payload["self"]["errors"] >= 1


# --- the request inspector --------------------------------------------------

def test_a_request_view_lists_its_blocks_per_group_and_what_the_tier_could_park():
    engine = engine_with_request({0: 8, 1: 2, 4: 1})
    payload = block(engine, request_id="cmpl-a")
    assert payload["verb"] == "request"
    assert payload["found"] is True
    assert payload["request"]["request_id"] == "cmpl-a"
    assert payload["request"]["status_name"] == "RUNNING"
    assert payload["request"]["status_value"] == 1
    assert payload["request"]["num_tokens"] == 4096
    assert payload["request"]["block_hashes"] == 32
    groups = payload["groups"]
    assert [entry["group"] for entry in groups] == [0, 1, 2, 3, 4, 5]
    assert groups[0]["block_ids"] == list(range(8))
    assert groups[0]["block_ids_dropped"] == 0
    assert groups[1]["block_ids"] == [0, 1]
    assert groups[2]["block_ids"] == []


def test_the_coverable_hashes_figure_is_the_fork_s_own_arithmetic():
    """manager.py:826, reproduced and shown with its terms.

    The default request: computed 3968 - placeholders 64 = confirmed 3904;
    partial hash hits ON, so alignment = hash_block_size = 128 -> aligned 3840;
    group 0 block size 1024 -> 3 blocks; hashes 32 at scale 8 -> 4; blocks held 8.
    min(8, 3, 4) = 3, bound by the aligned tokens.
    """
    engine = engine_with_request({0: 8})
    entry = block(engine, request_id="cmpl-a")["groups"][0]
    assert entry["hash_scale_factor"] == 8          # 1024 // 128
    assert entry["hashes_at_group_size"] == 4       # 32 // 8
    assert entry["coverable_blocks"] == 4
    view = entry["tier_view"]
    assert view["confirmed_tokens"] == 3904
    assert view["aligned_tokens"] == 3840
    assert view["alignment"] == 128
    assert view["ready_blocks"] == 3
    assert view["binding_term"] == "aligned_blocks"
    assert "manager.py:826" in view["meaning"]


def test_the_binding_term_names_the_limit_that_binds():
    # Only two blocks held: the block list is now the binding term.
    engine = engine_with_request({0: 2})
    view = block(engine, request_id="cmpl-a")["groups"][0]["tier_view"]
    assert view["ready_blocks"] == 2
    assert view["binding_term"] == "block_count"


def test_the_hash_view_can_be_the_binding_term():
    # Plenty of blocks and plenty of aligned tokens, but few hashes: at scale 8,
    # 8 hashes give ONE hash-visible block.
    engine = engine_with_request({0: 16}, block_hashes=8,
                                 num_computed_tokens=16384,
                                 num_output_placeholders=0)
    entry = block(engine, request_id="cmpl-a")["groups"][0]
    assert entry["hashes_at_group_size"] == 1
    assert entry["tier_view"]["ready_blocks"] == 1
    assert entry["tier_view"]["binding_term"] == "hashes_at_group_size"


def test_a_group_that_fails_divisibility_could_park_nothing():
    """The fork would assert before it ever stored, so "coverable" is zero - the
    verb must not report a number the tier cannot act on."""
    specs = [fake.attention_spec(block_size=1000)]
    groups = [fake.group(specs[0], ["l0"])]
    managers = [fake.manager_item(0, specs[0])]
    engine = engine_with_request({0: 8}, groups=groups, managers=managers)
    entry = block(engine, request_id="cmpl-a")["groups"][0]
    assert entry["divisibility"]["would_fail"] is True
    assert entry["coverable_blocks"] == 0
    assert entry["hash_scale_factor"] is None
    # ready is still computed (the fork would still try), and says which term bound
    assert entry["tier_view"]["ready_blocks"] == 0


def test_a_mamba_group_reports_its_own_alignment_and_scale():
    engine = engine_with_request({4: 4})
    entry = block(engine, request_id="cmpl-a")["groups"][4]
    assert entry["spec_class"] == "MambaSpec"
    assert entry["block_size"] == 128
    assert entry["hash_scale_factor"] == 1          # 128 // 128
    assert entry["hashes_at_group_size"] == 32
    assert entry["prefix_cacheable"] is True
    assert entry["has_positionally_stable_blocks"] is True


def test_block_ids_are_capped_and_say_what_was_dropped():
    engine = engine_with_request({0: 100})
    entry = block(engine, request_id="cmpl-a")["groups"][0]
    assert entry["block_count"] == 100
    assert len(entry["block_ids"]) == kvinspect.DEFAULT_IDS
    assert entry["block_ids_dropped"] == 100 - kvinspect.DEFAULT_IDS


def test_the_id_cap_is_a_bound_not_a_suggestion():
    engine = engine_with_request({0: 100})
    payload = block(engine, request_id="cmpl-a", ids_cap=10)
    assert payload["ids_cap"] == 10
    entry = payload["groups"][0]
    assert len(entry["block_ids"]) == 10
    assert entry["block_ids_dropped"] == 90
    # Absurd caps come back clamped, and 0/negative means "use the default".
    assert block(engine, request_id="cmpl-a",
                 ids_cap=10 ** 6)["ids_cap"] == kvinspect.MAX_IDS_LIMIT
    assert block(engine, request_id="cmpl-a", ids_cap=0)["ids_cap"] == \
        kvinspect.DEFAULT_IDS
    assert block(engine, request_id="cmpl-a", ids_cap=-5)["ids_cap"] == \
        kvinspect.DEFAULT_IDS


def test_the_tier_alignment_choice_is_reported_with_its_consequence():
    engine = engine_with_request({0: 8})
    alignment = block(engine, request_id="cmpl-a")["tier_alignment"]
    assert alignment["present"] is True
    assert alignment["enable_partial_hash_hits"] is True
    assert alignment["store_alignment"] == 128      # the hash unit, not 1024
    engine, info = fake.production_geometry()
    info["tier"].cpu_coordinator.enable_partial_hash_hits = False
    engine.scheduler.connector.scheduler_manager = info["tier"]
    sched_engine = engine_with_request({0: 8}, tier=info["tier"])
    alignment = block(sched_engine, request_id="cmpl-a")["tier_alignment"]
    assert alignment["store_alignment"] == 1024     # whole blocks only


def test_no_tier_explains_why_there_is_no_alignment():
    engine = engine_with_request({0: 4}, tier=False)
    alignment = block(engine, request_id="cmpl-a")["tier_alignment"]
    assert alignment["present"] is False
    assert "SimpleCPUOffloadConnector" in alignment["reason"]


def test_the_request_counters_and_the_scheduler_context_travel_together():
    engine = engine_with_request({0: 1})
    payload = block(engine, request_id="cmpl-a")
    assert payload["scheduler"]["requests"] == 1
    assert payload["scheduler"]["running"] == 1
    assert payload["scheduler"]["current_step"] == 1234


def test_an_unknown_request_id_lists_the_ids_the_engine_has():
    engine = engine_with_request({0: 1})
    payload = block(engine, request_id="cmpl-gone")
    assert payload["found"] is False
    assert payload["known_request_ids"] == ["cmpl-a"]
    assert payload["known_request_ids_dropped"] == 0
    assert "is finished, or the id is wrong" in payload["reason"]


def test_no_request_id_asks_for_the_id_list():
    engine = engine_with_request({0: 1})
    payload = block(engine)
    assert payload["found"] is False
    assert payload["known_request_ids"] == ["cmpl-a"]
    assert "request_id" in payload["reason"]


def test_the_id_list_is_capped():
    requests = {f"r{i}": fake.request(f"r{i}") for i in range(200)}
    engine = fake.engine(fake.scheduler(requests=requests))
    payload = block(engine)
    assert len(payload["known_request_ids"]) == kvinspect.MAX_REQUEST_IDS
    assert payload["known_request_ids_dropped"] == 200 - kvinspect.MAX_REQUEST_IDS


def test_a_request_id_that_is_not_a_string_is_refused_as_data():
    engine = engine_with_request({0: 1})
    payload = block(engine, request_id=7)
    assert payload["found"] is False
    assert payload["reason"] == "request_id must be a string"


def test_naming_both_ids_inspects_the_block_and_says_so():
    engine = engine_with_request({0: 1})
    payload = block(engine, block_id=2, request_id="cmpl-a")
    assert payload["verb"] == "block"
    assert "the request was not" in payload["note"]


def test_a_misaligned_req_to_blocks_mapping_is_data(reset_snapshot_state):
    """A manager whose table raises is reported as "no blocks", not a crash."""
    class RaisingTable(dict):
        """A mapping that refuses: the plugin catches KeyError/TypeError here, so a
        table of the wrong shape costs the block list and nothing else."""

        def get(self, key):
            raise TypeError("the table refused")

    specs = [fake.attention_spec()]
    manager = fake.manager_item(0, specs[0])
    manager.req_to_blocks = RaisingTable()
    engine = fake.engine(fake.scheduler(
        groups=[fake.group(specs[0], ["l"])], managers=[manager],
        requests={"cmpl-a": fake.request("cmpl-a")}))
    payload = block(engine, request_id="cmpl-a")
    assert payload["found"] is True
    assert payload["groups"][0]["block_ids"] == []
    assert payload["self"]["errors"] >= 1


# --- the handler's contract -------------------------------------------------

def test_an_empty_call_answers_with_the_usage_note():
    payload = kvinspect.inspect_for_engine(fake.engine(None), {})
    assert "usage" in payload
    assert payload["usage"]["ids_cap_default"] == kvinspect.DEFAULT_IDS
    assert payload["usage"]["ids_cap_max"] == kvinspect.MAX_IDS_LIMIT
    assert "block_id" in payload["usage"]["verbs"]["block"]


def test_unrecognised_argument_keys_are_reported_not_swallowed():
    payload = kvinspect.inspect_for_engine(fake.engine(None),
                                           {"block_id": 1, "colour": "blue"})
    assert payload["unknown_args"] == ["colour"]


@pytest.mark.parametrize("args", [None, "not a dict", 5, ["block_id"]])
def test_an_argument_that_is_not_a_dict_degrades_to_the_id_list(args):
    engine = engine_with_request({0: 1})
    payload = kvinspect.inspect_for_engine(engine, args)
    assert payload["found"] is False
    assert payload["known_request_ids"] == ["cmpl-a"]


class ExplodingRequest:
    def __getattr__(self, name):
        raise RuntimeError(f"no {name} on this request")


def test_a_hostile_request_becomes_an_error_payload_and_never_raises(
        reset_snapshot_state):
    engine = fake.engine(fake.scheduler(requests={"cmpl-a": ExplodingRequest()}))
    payload = kvinspect.inspect_for_engine(engine, {"request_id": "cmpl-a"})
    assert payload["error"].startswith("RuntimeError")
    assert payload["verb"] == "inspect"
    assert "usage" in payload                    # a wrong call still teaches
    assert reset_snapshot_state["errors"] >= 1


def test_an_engine_with_no_scheduler_answers_about_the_request_view():
    payload = kvinspect.inspect_for_engine(fake.engine(None),
                                           {"request_id": "cmpl-a"})
    assert payload["found"] is False
    assert payload["reason"] == "scheduler.requests is not a dict on this engine"


def test_the_payload_carries_the_wire_and_dev_state_fields():
    engine = engine_with_request({0: 1})
    payload = block(engine, request_id="cmpl-a")
    assert payload["version"] == kvinspect.PROTOCOL_VERSION
    assert payload["guard_mode"] is False
    assert payload["trace_armed"] is False
    assert isinstance(payload["epoch"], str)
    assert payload["build_us"] >= 0
    assert "self" in payload


def test_the_request_view_reports_groups_dropped_at_the_cap():
    specs = [fake.attention_spec() for _ in range(kvinspect.MAX_GROUP_ENTRIES + 3)]
    managers = [fake.manager_item(i, spec) for i, spec in enumerate(specs)]
    groups = [fake.group(spec, ["l"]) for spec in specs]
    engine = engine_with_request({}, groups=groups, managers=managers)
    payload = block(engine, request_id="cmpl-a")
    assert len(payload["groups"]) == kvinspect.MAX_GROUP_ENTRIES
    assert payload["groups_dropped"] == 3


# --- THE TWO POOLS, AND THE BOUND THAT BELONGS TO EACH OF THEM --------------
# Measured on the live box on 2026-10-05: the GPU pool holds 317 blocks and the
# tier's CPU pool holds 1331 slots. `block_id` used to be validated against the GPU
# pool and the answer returned early, so CPU slots 317..1330 - TWO THIRDS OF THE
# HOST-RAM ARENA - could not be named through the documented verb, and it failed
# SILENTLY: a 200 with `cpu_slot: no`, not an error. `spill-one` needs a CPU slot and
# a key, and `/block` is how an operator finds one, so this is the shape of the live
# numbers, pinned.

GPU_BLOCKS = 317
CPU_BLOCKS = 1331


def live_pools():
    """The live geometry: a 317-block GPU pool and a 1331-slot CPU pool."""
    engine, info = fake.production_geometry()
    gpu_blocks = ([fake.Block(i, block_hash=fake.hash_key(i, i % 6), ref_cnt=1,
                              num_tokens=1024) for i in range(200)]
                  + [fake.Block(i) for i in range(200, GPU_BLOCKS)])
    info["pool"].blocks = gpu_blocks
    info["pool"].num_gpu_blocks = GPU_BLOCKS
    cpu_blocks = [fake.Block(i, block_hash=fake.hash_key(i, 2), ref_cnt=0)
                  for i in range(CPU_BLOCKS)]
    info["tier"].cpu_block_pool = fake.Pool(cpu_blocks,
                                            num_gpu_blocks=CPU_BLOCKS,
                                            free=CPU_BLOCKS - 100)
    return engine, info


def test_a_cpu_slot_above_the_gpu_pool_size_is_nameable():
    """THE BUG, at the live numbers: slot 317 and slot 600 used to answer "no slot".

    Each section is bounded by the pool it describes, so the CPU half resolves while
    the GPU half honestly says the id is out of ITS range.
    """
    engine, _info = live_pools()
    for slot in (317, 600, CPU_BLOCKS - 1):
        payload = block(engine, block_id=slot)
        assert payload["found"] is False, "the GPU pool has 317 blocks"
        assert payload["reason_kind"] == "gpu_block_id_out_of_range", payload
        assert "317 block objects" in payload["reason"], payload["reason"]
        assert "block" not in payload
        slot_view = payload["tier_cpu_slot"]
        assert slot_view["present"] is True and slot_view["found"] is True, slot_view
        assert slot_view["block"]["block_id"] == slot
        # The parked block's `BlockHashWithGroupId` decodes here, which is the whole
        # point: this is how an operator gets the KEY for `spill-one`.
        assert slot_view["block"]["has_hash"] is True
        assert slot_view["block"]["hash_group_id"] == 2
        assert len(slot_view["block"]["hash_hex"]) == 32  # 16 bytes
        assert payload["sections"] == {"gpu": False, "cpu": True,
                                       "answered": True,
                                       "gpu_pool_blocks": GPU_BLOCKS,
                                       "cpu_pool_blocks": CPU_BLOCKS}


def test_an_id_out_of_range_for_both_pools_names_both_pools():
    """A named reason for each section - never a bare "no cpu slot"."""
    engine, _info = live_pools()
    payload = block(engine, block_id=CPU_BLOCKS + 69)
    assert payload["found"] is False
    assert payload["sections"]["answered"] is False
    assert payload["sections"] == {"gpu": False, "cpu": False, "answered": False,
                                   "gpu_pool_blocks": GPU_BLOCKS,
                                   "cpu_pool_blocks": CPU_BLOCKS}
    slot_view = payload["tier_cpu_slot"]
    assert slot_view["found"] is False
    assert slot_view["reason_kind"] == "cpu_block_id_out_of_range"
    assert f"{CPU_BLOCKS} objects" in slot_view["reason"], slot_view["reason"]
    # The top-level reason carries BOTH bounds, so a reader cannot take the GPU
    # pool's size for the CPU pool's.
    assert "GPU pool" in payload["reason"] and "CPU pool" in payload["reason"]
    assert str(GPU_BLOCKS) in payload["reason"]
    assert str(CPU_BLOCKS) in payload["reason"]


@pytest.mark.parametrize("block_id", [-1, 0, 316, 317, 5000])
def test_the_gpu_sections_own_bound_still_behaves_as_before(block_id):
    """No tier at all: the GPU bound answers exactly as it always did.

    `tier_cpu_slot` stays ABSENT (not "no cpu slot") when there is no connector -
    that distinction is the only thing telling an operator the tier is missing
    rather than the id being wrong.
    """
    engine = fake.engine(fake.scheduler(
        pool=fake.Pool([fake.Block(i) for i in range(GPU_BLOCKS)])))
    payload = block(engine, block_id=block_id)
    assert ("block" in payload) is (0 <= block_id < GPU_BLOCKS), payload
    assert payload["found"] is (0 <= block_id < GPU_BLOCKS)
    assert payload["sections"] == {
        "gpu": 0 <= block_id < GPU_BLOCKS, "cpu": False,
        "answered": 0 <= block_id < GPU_BLOCKS,
        "gpu_pool_blocks": GPU_BLOCKS, "cpu_pool_blocks": None}
    assert "tier_cpu_slot" not in payload
    if not payload["found"]:
        assert "not nameable" in payload["reason"]
        assert "317 block objects" in payload["reason"]


def test_an_id_inside_the_gpu_pool_answers_both_halves_as_it_did():
    """The healthy path is unchanged: GPU block AND the CPU slot of the same id."""
    engine, _info = live_pools()
    payload = block(engine, block_id=5, group=2)
    assert payload["found"] is True and payload["block"]["block_id"] == 5
    assert payload["tier_cpu_slot"]["found"] is True
    assert payload["tier_cpu_slot"]["block"]["block_id"] == 5
    assert payload["sections"]["answered"] is True
    assert payload["sections"]["gpu"] is True and payload["sections"]["cpu"] is True
    assert "reason" not in payload or payload["reason"] is None


def test_a_cpu_pool_that_cannot_be_read_says_so_rather_than_out_of_range():
    """`present: false` (no list to bound against) is NOT "the id is out of range"."""
    engine, info = live_pools()
    info["tier"].cpu_block_pool = fake.Pool([])
    info["tier"].cpu_block_pool.blocks = "not a list"
    payload = block(engine, block_id=600)
    slot_view = payload["tier_cpu_slot"]
    assert slot_view["present"] is False
    assert slot_view["reason_kind"] == "no_cpu_block_list"
    assert payload["sections"]["cpu"] is False
