"""`kvconfig`: the KV-cache geometry, read as data instead of crashed on.

The reason this verb exists is one line of the fork:

    vllm/v1/core/kv_cache_utils.py:2888
        assert block_size % hash_block_size == 0

repeated inside `BlockHashListWithBlockSize.__init__` (:2823) and consumed by the
tier on its FIRST store for a group
(`resolve_block_hashes(request.block_hashes, hash_block_size, group_block_size)`,
`vllm/v1/simple_kv_offload/manager.py`). An engine that boots happily can still
hold a group whose block size is not a multiple of the hash unit - and the crash
comes later, at the first store, far from the cause.

So this module answers, before anything is stored: what is every group's block
size, what is the hash unit, and does each division hold. Everything here is a
read; nothing is armed; nothing is wrapped.
"""

from __future__ import annotations

import json

import fakes_debug as fake
import pytest

from kvtier_dev_plugin import kvconfig

# --- helpers ----------------------------------------------------------------

def one_bad_group_engine(bad_index=2, bad_size=1000, hash_block_size=128):
    specs = [fake.attention_spec() for _ in range(4)]
    specs[bad_index] = fake.attention_spec(block_size=bad_size)
    groups = [fake.group(spec, [f"layer_{i}"]) for i, spec in enumerate(specs)]
    managers = [fake.manager_item(i, spec) for i, spec in enumerate(specs)]
    sched = fake.scheduler(groups=groups, managers=managers,
                           hash_block_size=hash_block_size)
    return fake.engine(sched)


def payload_for(engine):
    return kvconfig.kvconfig_for_engine(engine)


# --- the geometry -----------------------------------------------------------

def test_the_geometry_reports_every_group_with_its_own_block_size():
    engine, info = fake.production_geometry()
    payload = payload_for(engine)
    geometry = payload["geometry"]
    assert geometry["num_blocks"] == 8192
    assert geometry["num_groups"] == 6
    groups = geometry["groups"]
    assert [entry["index"] for entry in groups] == [0, 1, 2, 3, 4, 5]
    assert [entry["block_size"] for entry in groups] == [1024, 1024, 1024, 1024,
                                                         128, 1024]
    assert [entry["spec_class"] for entry in groups] == [
        "FullAttentionSpec", "FullAttentionSpec", "FullAttentionSpec",
        "FullAttentionSpec", "MambaSpec", "QsaRingSpec"]
    # The per-group page size is what a park costs; a group that reports it
    # wrong makes every byte number downstream wrong.
    assert [entry["page_size_bytes"] for entry in groups] == [16384] * 4 + [8192,
                                                                            4096]
    assert groups[info["ring_index"]]["is_circular_buffer"] is True
    assert groups[info["mamba_index"]]["is_circular_buffer"] is False


def test_the_spec_block_reports_the_attributes_that_make_a_group_what_it_is():
    engine, _ = fake.production_geometry()
    groups = payload_for(engine)["geometry"]["groups"]
    mamba = groups[4]["spec"]
    assert mamba["class"] == "MambaSpec"
    assert mamba["mamba_type"] == "gdn"
    assert mamba["num_prefill_checkpoint_blocks"] == 1
    # A Mamba group has no head geometry to report, so it must not grow a wall of
    # nulls: an absent field is ABSENT, not None.
    assert "num_heads" not in mamba
    fa = groups[0]["spec"]
    assert fa["num_kv_heads"] == 2
    assert fa["dtype"] == "torch.bfloat16"
    assert "mamba_type" not in fa


def test_a_fraction_and_bytes_are_rendered_not_dropped():
    """`tokens_per_state` is `int | Fraction` and a block hash is bytes; neither
    is encodable, and the engine already owns the object, so render it here."""
    spec = fake.attention_spec(tokens_per_state=fake.Fraction(1, 2))
    engine = fake.engine(fake.scheduler(
        groups=[fake.group(spec, ["l0"])], managers=[fake.manager_item(0, spec)]))
    spec_view = payload_for(engine)["geometry"]["groups"][0]["spec"]
    assert spec_view["tokens_per_state"] == "1/2"


def test_layer_names_are_capped_and_say_what_was_dropped():
    spec = fake.attention_spec()
    names = [f"layer.{i}" for i in range(100)]
    engine = fake.engine(fake.scheduler(groups=[fake.group(spec, names)],
                                        managers=[fake.manager_item(0, spec)]))
    entry = payload_for(engine)["geometry"]["groups"][0]
    assert entry["layer_count"] == 100
    assert len(entry["layer_names"]) == kvconfig.MAX_LAYER_NAMES
    assert entry["layer_names_dropped"] == 100 - kvconfig.MAX_LAYER_NAMES


def test_more_groups_than_the_cap_are_reported_as_dropped_not_silently_cut():
    specs = [fake.attention_spec() for _ in range(kvconfig.MAX_GROUPS + 5)]
    engine = fake.engine(fake.scheduler(
        groups=[fake.group(spec, ["l"]) for spec in specs],
        managers=[fake.manager_item(i, spec) for i, spec in enumerate(specs)]))
    payload = payload_for(engine)
    assert len(payload["geometry"]["groups"]) == kvconfig.MAX_GROUPS
    summary = payload["geometry"]["divisibility_summary"]
    assert summary["groups_reported"] == kvconfig.MAX_GROUPS
    assert summary["groups_dropped"] == 5
    assert payload["scheduler"]["managers"]["truncated"] is True


# --- divisibility: the point of the verb ------------------------------------

def test_divisibility_is_the_forks_assert_computed_as_data():
    answer = kvconfig.divisibility(1024, 128)
    assert answer["divisible"] is True
    assert answer["would_fail"] is False
    assert answer["remainder"] == 0
    assert answer["scale_factor"] == 8          # 8 hash units per group block
    assert "kv_cache_utils.py:2888" in answer["rule"]


@pytest.mark.parametrize("size,unit,expected", [
    (1000, 128, 104),      # a group that WILL assert on its first store
    (128, 1024, 128),      # smaller than the unit: the same failure, reversed
])
def test_a_failing_division_is_named_with_its_remainder(size, unit, expected):
    answer = kvconfig.divisibility(size, unit)
    assert answer["divisible"] is False
    assert answer["would_fail"] is True
    assert answer["remainder"] == expected
    assert answer["scale_factor"] is None       # no scale to report


@pytest.mark.parametrize("size,unit", [
    (None, 128), (1024, None), ("1024", 128), (True, 128), (1024, 0), (1024, -1),
])
def test_an_unprovable_division_says_unknown_rather_than_safe(size, unit):
    """"unknown" and "safe" must never look the same: `would_fail` is None here,
    and every such group is listed in `groups_unknown` of the summary."""
    answer = kvconfig.divisibility(size, unit)
    assert answer["divisible"] is None
    assert answer["would_fail"] is None
    assert "reason" in answer


def test_a_group_that_fails_divisibility_is_named_and_the_summary_says_so():
    engine = one_bad_group_engine(bad_index=2)
    payload = payload_for(engine)
    summary = payload["geometry"]["divisibility_summary"]
    assert summary["hash_block_size"] == 128
    assert summary["groups_failing"] == [2]
    assert summary["groups_unknown"] == []
    assert summary["any_fail"] is True
    assert summary["all_divisible"] is False
    assert summary["groups_checked"] == 4
    assert "resolve_block_hashes" in summary["meaning"]
    failing = payload["geometry"]["groups"][2]["divisibility"]
    assert failing["group_block_size"] == 1000
    assert failing["would_fail"] is True


def test_a_healthy_geometry_says_all_divisible():
    engine, _ = fake.production_geometry()
    summary = payload_for(engine)["geometry"]["divisibility_summary"]
    assert summary["all_divisible"] is True
    assert summary["any_fail"] is False
    assert summary["groups_failing"] == []


def test_an_unknown_group_keeps_all_divisible_false():
    """A group whose operands we cannot read must not be counted as safe."""
    spec = fake.attention_spec()
    del spec.__dict__["block_size"]
    engine = fake.engine(fake.scheduler(groups=[fake.group(spec, ["l"])],
                                        managers=[fake.manager_item(0, spec)]))
    summary = payload_for(engine)["geometry"]["divisibility_summary"]
    assert summary["groups_unknown"] == [0]
    assert summary["all_divisible"] is False
    assert summary["any_fail"] is False


# --- the tier's own view, and where the two disagree ------------------------

def test_the_tier_crosscheck_agrees_by_default():
    engine, _ = fake.production_geometry()
    cross = payload_for(engine)["tier_crosscheck"]
    assert cross["present"] is True
    assert cross["class"] == "SimpleCPUOffloadScheduler"
    assert cross["hash_block_size"] == 128
    assert cross["group_block_sizes"] == [1024, 1024, 1024, 1024, 128, 1024]
    assert cross["disagreements"]["hash_block_size"] is False
    assert cross["disagreements"]["block_size"] is False
    assert cross["disagreements"]["group_block_sizes"] == []
    assert all(entry["agree"] for entry in cross["per_group"])


def test_a_tier_hash_unit_that_differs_from_the_schedulers_is_named():
    _engine, info = fake.production_geometry()
    info["tier"].hash_block_size = 256
    info["scheduler"].connector.scheduler_manager = info["tier"]
    cross = payload_for(_engine)["tier_crosscheck"]
    assert cross["disagreements"]["hash_block_size"] is True
    # And the tier-side division is recomputed against the TIER's operands, which
    # is the only way to see a failure the engine itself cannot see.
    assert cross["per_group"][0]["tier_divisibility"]["would_fail"] is False
    info["tier"].group_block_sizes[3] = 1000
    cross = payload_for(_engine)["tier_crosscheck"]
    assert cross["disagreements"]["group_block_sizes"] == [3]
    assert cross["per_group"][3]["tier_divisibility"]["would_fail"] is True


def test_a_group_size_the_tier_does_not_know_about_is_not_an_agreement():
    engine, info = fake.production_geometry()
    info["tier"].group_block_sizes = [1024, 1024]     # fewer than the groups
    cross = payload_for(engine)["tier_crosscheck"]
    assert [entry["agree"] for entry in cross["per_group"]][:2] == [True, True]
    assert all(entry["agree"] is False for entry in cross["per_group"][2:])


def test_no_tier_says_so_instead_of_reporting_a_vacuous_agreement():
    engine = fake.engine(fake.scheduler(groups=[], managers=[], tier=None))
    cross = payload_for(engine)["tier_crosscheck"]
    assert cross["present"] is False
    assert "SimpleCPUOffloadConnector" in cross["reason"]


def test_a_spec_and_a_manager_that_disagree_on_block_size_are_named():
    """The manager ALLOCATES with its own block_size; a disagreement means a park
    under the wrong stride, which is a corruption, not a crash."""
    specs = [fake.attention_spec(block_size=1024),
             fake.attention_spec(block_size=512)]
    managers = [fake.manager_item(0, specs[0]),
                fake.manager_item(1, specs[1], block_size=256)]
    engine = fake.engine(fake.scheduler(groups=[fake.group(s, ["l"]) for s in specs],
                                        managers=managers))
    groups = payload_for(engine)["geometry"]["groups"]
    assert "manager_block_size_mismatch" not in groups[0]
    assert groups[1]["manager_block_size_mismatch"] == {
        "spec_block_size": 512, "manager_block_size": 256}


# --- the surrounding state --------------------------------------------------

def test_managers_are_aligned_by_their_own_group_id_not_by_position():
    specs = [fake.attention_spec(), fake.mamba_spec()]
    managers = [fake.manager_item(7, specs[1]), fake.manager_item(0, specs[0])]
    engine = fake.engine(fake.scheduler(
        groups=[fake.group(spec, ["l"]) for spec in specs], managers=managers))
    report = payload_for(engine)["scheduler"]["managers"]
    assert report["present"] is True
    assert report["count"] == 2
    assert set(report["by_group_id"]) == {"0", "7"}
    assert report["by_group_id"]["7"]["position"] == 0
    assert report["scheduler_block_size"] == 1024
    assert report["cache_hit_alignment_tokens"] == 128
    assert report["eagle_group_ids"] == []


def test_the_group_view_uses_the_positional_manager_and_says_which_it_was():
    specs = [fake.attention_spec(), fake.mamba_spec()]
    managers = [fake.manager_item(1, specs[0]), fake.manager_item(0, specs[1])]
    engine = fake.engine(fake.scheduler(
        groups=[fake.group(spec, ["l"]) for spec in specs], managers=managers))
    groups = payload_for(engine)["geometry"]["groups"]
    assert groups[0]["manager"]["kv_cache_group_id"] == 1
    assert groups[0]["manager"]["present"] is True


def test_the_block_pool_is_reported_once_and_shared():
    engine, info = fake.production_geometry()
    pool = payload_for(engine)["geometry"]["block_pool"]
    assert pool["bound"] is True
    assert pool["total_blocks"] == 8192
    assert pool["free_blocks"] == 8192 - 40
    assert pool["block_objects"] == 64
    assert pool["cached_entries"] == 20
    assert pool["free_queue_nodes"] == 8192 - 40


def test_a_lying_pool_is_data_and_not_a_crash(reset_snapshot_state):
    specs = [fake.attention_spec()]
    pool = fake.Pool([fake.Block(0)], raise_on_free=True)
    engine = fake.engine(fake.scheduler(groups=[fake.group(specs[0], ["l"])],
                                        managers=[fake.manager_item(0, specs[0])],
                                        pool=pool))
    payload = payload_for(engine)
    assert payload["geometry"]["block_pool"]["free_blocks"] is None
    assert payload["self"]["errors"] >= 1


def test_no_pool_bound_is_bound_false():
    engine = fake.engine(fake.scheduler(groups=[], managers=[], pool=None))
    # `scheduler.kv_cache_manager.block_pool` is absent -> {"bound": False}.
    engine.scheduler.kv_cache_manager.block_pool = None
    assert payload_for(engine)["geometry"]["block_pool"] == {"bound": False}


def test_the_cache_config_knobs_and_the_prefix_match_unit_rule():
    engine, _ = fake.production_geometry()
    config_view = payload_for(engine)["config"]
    assert config_view["present"] is True
    assert config_view["block_size"] == 1024
    assert config_view["prefix_match_unit"] == 128
    # config/cache.py:91-99 says prefix_match_unit must divide block_size: the
    # same rule as hash_block_size, checked once here rather than at boot.
    assert config_view["prefix_match_unit_divisibility"]["divisible"] is True
    assert config_view["enable_prefix_caching"] is True


def test_a_missing_cache_config_is_a_reason_not_a_crash():
    engine = fake.engine(fake.scheduler(groups=[], managers=[]))
    engine.scheduler.cache_config = None
    view = payload_for(engine)["config"]
    assert view == {"present": False, "reason": "scheduler exposes no cache_config"}


def test_the_group_and_transfer_classification_the_tier_actually_needs():
    engine, info = fake.production_geometry()
    geometry = payload_for(engine)["geometry"]
    # The scratch ring is prefix_cacheable=False -> scratch, never parkable.
    assert geometry["scratch_group_ids"] == [info["ring_index"]]
    # It also has enable_kv_transfer=False -> not in the transfer set either.
    assert geometry["transfer_group_ids"] == [0, 1, 2, 3, 4]
    assert geometry["prefix_cacheable_group_ids"] == [0, 1, 2, 3, 4]
    assert info["ring_index"] not in geometry["prefix_cacheable_group_ids"]


# --- the ring spec: by name, through the MRO --------------------------------

def test_is_circular_buffer_matches_the_name_not_the_shape():
    assert kvconfig.is_circular_buffer(fake.circular_buffer_spec()) is True
    assert kvconfig.is_circular_buffer(
        fake.circular_buffer_spec(subclass=False)) is True
    # A ring-shaped spec that is NOT named for the fork's class is not assumed to
    # be one: guessing here would park blocks in a group that holds one per
    # request (kv_cache_interface.py:885).
    lookalike = fake.Obj(name="RingLike", block_size=1024, prefix_cacheable=False,
                         uses_slot_mapping=False)
    assert kvconfig.is_circular_buffer(lookalike) is False
    assert kvconfig.is_circular_buffer(None) is None


def test_a_circular_buffer_group_is_never_in_the_prefix_cacheable_set():
    ring = fake.circular_buffer_spec()
    engine = fake.engine(fake.scheduler(groups=[fake.group(ring, ["l"])],
                                        managers=[fake.manager_item(0, ring)]))
    geometry = payload_for(engine)["geometry"]
    assert geometry["scratch_group_ids"] == [0]
    assert geometry["prefix_cacheable_group_ids"] == []


# --- containment and the wire contract --------------------------------------

class Exploding:
    """Every attribute read raises: the hostile-object case for a pure reader."""

    def __getattr__(self, name):
        raise RuntimeError(f"no {name} here")


def test_a_hostile_engine_degrades_field_by_field_and_never_raises(
        reset_snapshot_state):
    """Every read goes through `_attribute`, so one exploding attribute costs ONE
    field, not the payload. The verb still answers with its normal shape."""
    payload = kvconfig.kvconfig_for_engine(Exploding())
    assert payload["verb"] == "kvconfig"
    assert "error" not in payload
    assert payload["geometry"]["groups"] == []
    assert payload["guard_mode"] is False
    assert payload["trace_armed"] is False
    assert reset_snapshot_state["errors"] >= 1


def test_a_failure_inside_the_builder_is_reported_as_an_error_payload(
        monkeypatch, reset_snapshot_state):
    """The containment path itself: if the payload cannot be built at all, the
    answer is a hand-built dict of formatted strings that cannot raise again."""
    def explode(_engine):
        raise RuntimeError("the builder refused")

    monkeypatch.setattr(kvconfig, "kvconfig_section", explode)
    engine, _ = fake.production_geometry()
    payload = kvconfig.kvconfig_for_engine(engine)
    assert payload["error"].startswith("RuntimeError")
    assert payload["verb"] == "kvconfig"
    assert payload["guard_mode"] is False
    assert payload["counters"]["errors"] >= 1


def test_a_lying_property_on_a_group_is_data_not_a_crash(reset_snapshot_state):
    """A `@property` that raises is a missing field, reported as absent."""
    class Lying:
        @property
        def kv_cache_spec(self):
            raise RuntimeError("the spec refused")

        @property
        def layer_names(self):
            raise RuntimeError("the layers refused")

    specs = [fake.attention_spec()]
    sched = fake.scheduler(groups=[Lying()], managers=[fake.manager_item(0,
                                                                         specs[0])])
    payload = kvconfig.kvconfig_for_engine(fake.engine(sched))
    assert "error" not in payload
    group = payload["geometry"]["groups"][0]
    assert group["spec_class"] is None
    assert group["layer_count"] is None
    assert payload["self"]["errors"] >= 1


def test_an_engine_with_nothing_still_answers_with_the_same_shape():
    empty = kvconfig.kvconfig_for_engine(fake.engine(None))
    full = kvconfig.kvconfig_for_engine(fake.engine(fake.scheduler()))
    assert "error" not in empty and "error" not in full
    # The KEYS are identical whether or not a scheduler exists: a reader can index
    # the degraded payload exactly like the healthy one.
    # The TOP-LEVEL shape is identical whatever the engine holds; a reader can
    # index a degraded payload exactly like a healthy one. Inside `config` the
    # shape may differ, because `present: false` carries a `reason` instead.
    assert set(empty) == set(full)
    assert empty["geometry"]["groups"] == []
    assert empty["scheduler"]["present"] is False
    assert empty["config"]["present"] is False


def test_the_payload_is_json_encodable_end_to_end():
    engine, _ = fake.production_geometry()
    text = json.dumps(payload_for(engine))
    assert "group_block_sizes" in text


def test_args_are_accepted_and_reported_as_ignored():
    """`kvconfig` takes `args` for arity symmetry and reads none of it; saying so
    in the payload is what keeps that honest to a caller who sent something."""
    engine, _ = fake.production_geometry()
    payload = kvconfig.kvconfig_for_engine(engine, {"group": 3})
    assert payload["args_ignored"] == ["group"]
    assert kvconfig.kvconfig_for_engine(engine)["args_ignored"] == []


def test_the_verb_publishes_the_development_state_it_is_running_under():
    engine, _ = fake.production_geometry()
    payload = payload_for(engine)
    assert payload["guard_mode"] is False
    assert payload["trace_armed"] is False
    assert payload["version"] == kvconfig.PROTOCOL_VERSION
    assert isinstance(payload["epoch"], str) and payload["epoch"]
    assert payload["build_us"] >= 0


def test_the_verb_costs_no_more_than_a_bounded_pass(reset_snapshot_state):
    """64 groups at the cap is still a fast read; a geometry dump that takes
    seconds is a reason not to run it, which is worse than not having it."""
    specs = [fake.attention_spec() for _ in range(kvconfig.MAX_GROUPS)]
    engine = fake.engine(fake.scheduler(
        groups=[fake.group(spec, [f"l{i}" for i in range(40)]) for spec in specs],
        managers=[fake.manager_item(i, spec) for i, spec in enumerate(specs)]))
    payload = payload_for(engine)
    assert payload["build_us"] < 200_000          # 200 ms of headroom
