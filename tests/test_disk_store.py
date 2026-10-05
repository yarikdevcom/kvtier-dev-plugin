r"""The worker-side wiring: a completed store becomes a named file, or a refusal.

`disk_store` is the module that decides whether a KV slot's bytes reach the SSD tier
and what file they land in, so the failures here are ordered by what each one costs
on the live box:

1. A WRONG NAME IS WORSE THAN NO FILE. Bytes written under a name no reader can
   reconstruct are unrecoverable and invisible. So: the digest comes from the
   fingerprint THIS worker builds for itself (its own measured slot width, its own
   rank), every field of which is read or measured and never invented; a field that
   cannot be read is a counted refusal; a config change moves the digest; two ranks
   cannot forge each other's name.
2. AN UNPATCHED ENGINE MUST BE CORRECT, NOT LUCKY. Today's
   `SimpleCPUOffloadMetadata` has no `store_keys`. The hook refuses and counts
   (`no_key_for_slot`), writes zero bytes, and `engine_fields` says `absent` - which
   is how the operator knows the engine half is not live without grepping a log.
3. THE PAIRING IS NEVER GUESSED. `store_keys[i]` names `store_cpu_blocks[i]`; a
   length mismatch is a peer bug that refuses the whole step, while one empty key is
   one slot's missing identity and costs that slot only. Truncating to the shorter
   list would write someone's bytes under someone else's name.
4. NOTHING BREAKS THE ENGINE. The fork's methods run first and their return value is
   handed back untouched; every entry point returns a dict; `BaseException` escapes;
   the sibling layers (`worker_diag`, `blockdump`) stay installed under this one.
5. OFF MEANS OFF. Unarmed, the per-step path builds no fingerprint, starts no thread,
   touches no directory and queues nothing.

Everything runs against a `tmp_path` store, a ctypes-backed region tensor (so the
bytes are real memory, copied by the same `ctypes.string_at` the worker will use), a
fork-shaped fake `SimpleCPUOffloadWorker` exec'd under the fork's module name, and a
fake `parallel_state` that either answers with a rank or refuses the way a process
without `init_process_group` does.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import pathlib
import sys
import types

import fakes_debug as fake
import pytest
from test_blockdump import _parallel_state_module, a_worker_rank
from test_disk_spill import SLOT_BYTES, FakeRegion, HostileRegion
from test_routes import (
    STATUS,
    FakeAsyncLLM,
    FakeEngineCoreClient,
    build_app,
    client_for,
)

from kvtier_dev_plugin import (  # noqa: E402 - the module under test
    blockdump,
    config,
    disk_layout,
    disk_spill,
    disk_store,
    engine_side,
    snapshot,
    worker_diag,
)

KEY = "a3f0b7c1d4e5f607"              # a key16, as the layout layer defines it
LONG_KEY = "a3f0b7c1d4e5f607" + "1" * 56   # the hex a BlockHashWithGroupId renders to
OTHER_KEY = "0f1e2d3c4b5a6978"
GROUP_SIZES = [1, 4, 4, 8, 16]


# --- the worker as the fork shapes it ----------------------------------------

# The worker's two methods, in the shape the live `worker.py` has them: `bind` keeps
# the metadata and queues its store event, `get_finished` polls the store stream and
# RECORDS completion in `_completed_store_events`. The hook's whole design comes from
# that split, so the fake must make completion happen the way the fork makes it
# happen - a fake that let the test set `_completed_store_events` directly would
# prove nothing about the diff the wrapper takes.
FORK_WORKER_SOURCE = """
class SimpleCPUOffloadWorker:
    def __init__(self):
        self.cpu_kv_caches = {}
        self.vllm_config = None
        self.kv_cache_config = None
        self._connector_metadata = None
        self._pending_store_event_indices = []
        self._completed_store_events = {}
        self._store_hwm = 0
        self.bind_calls = 0
        self.get_finished_calls = 0
        self.delivered = []                # which events the stream reports done
        self.finished_return = (set(), None)

    def bind_connector_metadata(self, metadata):
        self.bind_calls += 1
        self._connector_metadata = metadata
        if metadata.store_event != -1:     # INVALID_JOB_ID, metadata.py:12
            self._pending_store_event_indices.append(metadata.store_event)

    def get_finished(self, finished_req_ids):
        self.get_finished_calls += 1
        for index in list(self._pending_store_event_indices):
            if index in self.delivered:
                self._pending_store_event_indices.remove(index)
                self._completed_store_events[index] = 1
        return self.finished_return
"""

# Both shapes of the fork's metadata, as dataclasses: the real one (no new fields)
# and the one the engine patch produces (`kvtier-disk-restore-meta` declares them
# with `field(default_factory=list)`, so they have NO class attribute - the reason
# `engine_fields` reads the dataclass field list and not just `hasattr`).
METADATA_UNPATCHED_SOURCE = '''
from dataclasses import dataclass, field


@dataclass
class SimpleCPUOffloadMetadata:
    """`vllm/v1/simple_kv_offload/metadata.py:16-38` as it stands today."""

    store_cpu_blocks: list = field(default_factory=list)
    store_gpu_blocks: list = field(default_factory=list)
    store_event: int = -1
    load_event: int = -1


'''

METADATA_PATCHED_SOURCE = '''
from dataclasses import dataclass, field


@dataclass
class SimpleCPUOffloadMetadata:
    """The same, with the two fields the engine patch adds (plus restore's)."""

    store_cpu_blocks: list = field(default_factory=list)
    store_gpu_blocks: list = field(default_factory=list)
    store_event: int = -1
    load_event: int = -1
    store_keys: list = field(default_factory=list)
    store_fingerprint: str = ""
    load_paths: list = field(default_factory=list)


'''

SEED_MODULE_SOURCE = '''
HOLDER = {"seed": "vllm-none-hash", "raises": None}


def get_none_hash_seed():
    if HOLDER["raises"]:
        raise RuntimeError(HOLDER["raises"])
    return HOLDER["seed"]
'''


def register(name: str, source: str):
    """Exec `source` into a module NAMED `name` and put it in `sys.modules`.

    The `fork_class` posture: the plugin resolves the fork's names only against a
    module that is ALREADY imported, so a test has to make that claim honestly -
    a module object bearing the fork's own `__name__`, not a mock of it.

    Registered BEFORE the exec, because `@dataclass` resolves `cls.__module__`
    through `sys.modules` while it builds its field list: exec first and the
    metadata source dies with `AttributeError: 'NoneType' object has no attribute
    '__dict__'`, which says nothing about the thing under test.
    """
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(source, module.__dict__)  # noqa: S102 - our own source, never input
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def handler(caches=None, *, rank_width=None, with_parallel=True, **overrides):
    """A worker handler with everything the fingerprint builder reads by name.

    Every field is the FORK'S name and the FORK'S nesting (`model_config.model`,
    `cache_config.prefix_match_unit`, `parallel_config.decode_context_parallel_size`,
    `kv_cache_groups[i].kv_cache_spec.prefix_cacheable`). `with_parallel=False`
    stands in for a process whose config does not carry the parallel facts, which is
    the case the `parallel_state` group getters exist for.
    """
    model = fake.Obj(model="flash-next-v2", revision=None,
                     dtype="torch.bfloat16")
    cache = fake.Obj(cache_dtype="auto", prefix_caching_hash_algo="sha256_cbor",
                     block_size=16, prefix_match_unit=16)
    parallel = fake.Obj(world_size=4, decode_context_parallel_size=1,
                        prefill_context_parallel_size=1)
    vllm_config = fake.Obj(model_config=model, cache_config=cache,
                           parallel_config=parallel)
    groups = [fake.Obj(kv_cache_spec=fake.Obj(block_size=size,
                                              prefix_cacheable=(index < 3)))
              for index, size in enumerate(GROUP_SIZES)]
    out = fake.Obj(cpu_kv_caches=caches if caches is not None else
                   {"g0": FakeRegion(8, rank_width or SLOT_BYTES)},
                   vllm_config=vllm_config,
                   kv_cache_config=fake.Obj(kv_cache_groups=groups),
                   _connector_metadata=None,
                   _pending_store_event_indices=[],
                   _completed_store_events={})
    for name, value in overrides.items():
        setattr(out, name, value)
    if not with_parallel:
        out.vllm_config.parallel_config = None
    return out


def metadata(**fields):
    """A metadata object of whichever shape is currently registered.

    Built from the CLASS the plugin is also reading (so `engine_fields`' instance
    probe and the resolver see the same object the fork would have handed over), and
    `store_keys`/`store_fingerprint` are dropped entirely on the unpatched shape:
    the contract's placeholder is `""` INSIDE a present list, not a missing field.
    """
    cls = disk_store.metadata_class()
    assert cls is not None, "a metadata module must be registered"
    fields.setdefault("store_event", 3)
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    return cls(**{name: value for name, value in fields.items()
                  if name in allowed})


def spill_one_request(directory, **overrides):
    """One spill-one request file, written the way `_op_spill_one` writes it."""
    import time as _time
    record = {"version": disk_store.PROTOCOL_VERSION, "role": disk_store.ROLE,
              "op": "spill-one", "seq": 1, "epoch": "test-epoch",
              "requested_ts": round(_time.time(), 6),
              "expires_ts": round(_time.time() + 120.0, 6),
              "slot": 2, "key": KEY, "fingerprint": "",
              "writer": "the test"}
    record.update(overrides)
    directory = str(directory)
    path = os.path.join(directory, disk_store.PENDING_FILE)
    with open(path, "wb") as handle:
        handle.write(json.dumps(record, sort_keys=True).encode("utf-8"))
    return record


def store_files(root):
    """Every file under the spill root, as `{relative path: bytes}`. The store is
    small in these tests, so reading it all is cheaper than reasoning about it."""
    found = {}
    for base, _dirs, names in os.walk(str(root)):
        for name in names:
            full = os.path.join(base, name)
            with open(full, "rb") as handle:
                found[os.path.relpath(full, str(root))] = handle.read()
    return found


# --- fixtures: the tier, a rank, and the fork's metadata shape ---------------

@pytest.fixture
def tier(tmp_path, monkeypatch, reset_dev_state):
    """A configured, ARMED store under this test's tmp directory, with a rank.

    `reset_dev_state` is requested EXPLICITLY, the way `test_disk_spill.py`'s
    `armed_tier` does: it is not autouse, and without it this module's state -
    including an adopted fingerprint - leaks into the next test, where the very
    first `adopt_fingerprint` answers "already the naming authority" and a
    fingerprint test silently tests a cache hit.

    Deliberately NOT pre-`configure()`d: on the box nobody hands the worker a
    fingerprint, so a test that configured one would hide the very path this module
    exists to provide (`adopt_fingerprint`). The fixture only sets the two env vars
    that name a LOCATION and a SIZE, arms the gate, and pins the rank through the
    operator's own escape hatch.

    The tmpfs refusal is relaxed for the prefix list ONLY, because pytest's
    `tmp_path` lives under /tmp; the rule itself is proved in
    `test_disk_spill.py::test_a_tmpfs_root_never_becomes_a_store`.
    """
    monkeypatch.setattr(disk_layout, "FORBIDDEN_ROOT_PREFIXES", ())
    root = tmp_path / "kv-store"
    monkeypatch.setenv(config.ENV_DISK_ROOT, str(root))
    monkeypatch.setenv(config.ENV_DISK_BYTES, str(64 * SLOT_BYTES))
    monkeypatch.setenv(config.ENV_DIAG_DIR, str(tmp_path / "diag"))
    monkeypatch.setenv(config.ENV_DIAG_RANK, "PP0_TP0")
    (tmp_path / "diag").mkdir()
    disk_spill.arm()
    return root


@pytest.fixture
def worker_class():
    """The fork's worker class, installed under the fork's module name, and undone.

    `fork_class` builds a FRESH class per call and registers the module, so this
    class is not in `fakes_debug._NAMED_CLASSES` and `unwrap_everything` cannot see
    it: the fixture uninstalls the hook itself, the way `test_blockdump.py` pops the
    module it registered.
    """
    cls, _module = fake.fork_class(disk_store.TARGET_MODULE,
                                   disk_store.TARGET_CLASS,
                                   FORK_WORKER_SOURCE, register=True)
    try:
        yield cls
    finally:
        disk_store.uninstall()
        sys.modules.pop(disk_store.TARGET_MODULE, None)


@contextlib.contextmanager
def metadata_shape(patched: bool):
    """Make this process's `SimpleCPUOffloadMetadata` the patched or the real one."""
    saved = sys.modules.get(disk_store.METADATA_MODULE)
    source = METADATA_PATCHED_SOURCE if patched else METADATA_UNPATCHED_SOURCE
    register(disk_store.METADATA_MODULE, source)
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(disk_store.METADATA_MODULE, None)
        else:
            sys.modules[disk_store.METADATA_MODULE] = saved


@pytest.fixture(autouse=True)
def unpatched_engine():
    """The DEFAULT world: today's engine, whose metadata carries no `store_keys`.

    Autouse because "correct on an unpatched engine" is the property the whole
    module claims, and a test that forgot to install a patched shape must fail that
    way rather than by inheriting the previous test's.
    """
    with metadata_shape(False):
        yield


@pytest.fixture(autouse=True)
def none_hash_seed():
    """The fork's `get_none_hash_seed()`, importable, with a knob.

    The default is the fork's own `DEFAULT_NONE_HASH_SEED` (`kv_cache_utils.py:121`),
    i.e. the value a real unseeded boot returns - so the fingerprint path is proved
    against the text the fork actually produces, not against a tidy hex stand-in.
    """
    saved = sys.modules.get(disk_store.SEED_MODULE)
    module = register(disk_store.SEED_MODULE, SEED_MODULE_SOURCE)
    try:
        yield module.HOLDER
    finally:
        if saved is None:
            sys.modules.pop(disk_store.SEED_MODULE, None)
        else:
            sys.modules[disk_store.SEED_MODULE] = saved


@pytest.fixture(autouse=True)
def no_fake_distributed_state():
    """Never let a fake `parallel_state` leak into another test's rank answer."""
    yield
    sys.modules.pop(config_diag_parallel_state, None)


config_diag_parallel_state = "vllm.distributed.parallel_state"


# --- 1. the fingerprint: read or measured, never invented -------------------

def test_the_worker_builds_its_own_fingerprint_from_its_own_facts(tier):
    """The correction, in one test: the worker - not a broadcast - assembles the
    identity its files are named under, and every field says WHERE it came from."""
    built = disk_store.build_worker_fingerprint(handler())
    assert built["ok"] is True, built["reason"]
    fingerprint = built["fingerprint"]
    assert set(fingerprint) == {name for name, _kind
                                in disk_layout.FINGERPRINT_FIELDS}
    # The two rank-local fields, measured/resolved HERE rather than received.
    assert fingerprint["total_bytes_per_block"] == SLOT_BYTES
    assert built["sources"]["total_bytes_per_block"].startswith("measured:")
    assert fingerprint["rank"] == "PP0_TP0"
    assert "worker_diag.rank_id()" in built["sources"]["rank"]
    # The rank-invariant ones, read by the fork's names.
    assert fingerprint["model_id"] == "flash-next-v2"
    assert fingerprint["hash_block_size"] == 16     # cache_config.prefix_match_unit
    assert fingerprint["scheduler_block_size"] == 16  # cache_config.block_size
    assert fingerprint["group_block_sizes"] == GROUP_SIZES
    assert fingerprint["prefix_cacheable_group_ids"] == [0, 1, 2]
    assert fingerprint["world_size"] == 4
    # The fork's own default seed text, from the fork's own accessor.
    assert fingerprint["none_hash_seed"] == "str:vllm-none-hash"
    assert built["geometry_digest"] == disk_layout.compute_geometry_digest(
        fingerprint)


def test_a_revision_of_none_is_normalised_and_said_so(tier):
    """`model_config.revision` is None for any model not pinned by hash - the common
    boot. Refusing it would make the tier unusable everywhere; the stable rendering
    keeps the field's actual job (a weight change under one name must not revalidate
    old files) intact."""
    built = disk_store.build_worker_fingerprint(handler())
    assert built["normalized_none"] == ["model_revision"]
    assert built["fingerprint"]["model_revision"] == "none"
    assert "stable text" in built["sources"]["model_revision"]


def test_the_bytes_own_dtype_beats_the_configs_claim(tier):
    """Two fields claim to know the dtype. The tensors are the bytes, so they win -
    and the payload says which answered, because a silent preference is a bug that
    shows up as a name nobody can reproduce."""
    caches = {"g0": FakeRegion(8, SLOT_BYTES)}
    caches["g0"].dtype = "torch.float8_e4m3fn"
    built = disk_store.build_worker_fingerprint(handler(caches))
    assert built["fingerprint"]["dtype"] == "torch.float8_e4m3fn"
    assert "the bytes' own dtype" in built["sources"]["dtype"]
    # ...and the fallback is honest about being a claim, not a measurement.
    fallback = disk_store.build_worker_fingerprint(handler())
    assert fallback["fingerprint"]["dtype"] == "torch.bfloat16"
    assert "the config's claim" in fallback["sources"]["dtype"]


def test_a_field_that_cannot_be_read_is_refused_by_name_never_defaulted(tier):
    """The whole anti-guessing rule, tested at the door.

    A worker whose config does not carry a field could have been "helped" with a
    default; the digest would then be a fact about a configuration this process does
    not have, and the files it names would be readable by nobody. The refusal names
    each missing field and the paths tried.
    """
    broken = handler(with_parallel=False)
    built = disk_store.build_worker_fingerprint(broken)
    assert built["ok"] is False
    missing = {row["field"] for row in built["missing_fields"]}
    assert {"world_size", "dcp_world_size", "cp_world_size"} <= missing
    assert all(row["tried"] for row in built["missing_fields"])
    assert "fingerprint_partial" in built

    # No groups at all: the group layout is two fields, and both are reported.
    no_groups = handler()
    no_groups.kv_cache_config = None
    missing = {row["field"] for row
               in disk_store.build_worker_fingerprint(no_groups)["missing_fields"]}
    assert {"group_block_sizes", "prefix_cacheable_group_ids"} <= missing


def test_the_seed_moves_the_digest_and_is_published_like_the_rest_of_the_identity(
        tier, none_hash_seed):
    """Two facts about the seed, both load-bearing.

    It MOVES THE DIGEST: with a non-cryptographic hash function and no
    `PYTHONHASHSEED`, `get_none_hash_seed()` returns fresh random bytes per process
    (`kv_cache_utils.py:139-143`), so a boot whose seed changed must not be able to
    find files written under the old one.

    And it IS published, in exactly one place - the fingerprint block, in its
    canonical rendering, the same view `disk_spill.section()` publishes. A name
    cannot be reconstructed without it, so redacting it from one view and not the
    other would leave two authorities for one identity; the mitigation is that this
    is a dev-state surface behind the dev token, on loopback by default.
    """
    first = disk_store.build_worker_fingerprint(handler())
    none_hash_seed["seed"] = "a-different-secret"
    second = disk_store.build_worker_fingerprint(handler())
    assert first["geometry_digest"] != second["geometry_digest"]

    disk_store.adopt_fingerprint(handler())
    report = disk_store.fingerprint_report()
    assert report["fingerprint"]["none_hash_seed"] == "str:a-different-secret"
    for payload in (disk_store.disk_for_engine(None, {"op": "status"}),
                    disk_store.section()):
        text = json.dumps(payload, default=str)
        assert "a-different-secret" in text        # reproducible names, not magic
        assert text.count("a-different-secret") == 1   # ONE place: the identity block
    # And the same canonical text the spill mechanism publishes for one fingerprint.
    assert (disk_spill.section()["fingerprint"]["none_hash_seed"]
            == report["fingerprint"]["none_hash_seed"])


def test_an_accessor_that_raises_is_a_refusal_not_an_exception(tier, none_hash_seed):
    none_hash_seed["raises"] = "the seed accessor exploded"
    built = disk_store.build_worker_fingerprint(handler())
    assert built["ok"] is False
    assert any(row["field"] == "none_hash_seed"
               for row in built["missing_fields"])


def test_the_group_getters_are_the_fallback_for_a_config_without_them(
        tier, monkeypatch):
    """`parallel_config` may not carry the facts; `parallel_state` is then the only
    witness, and a process without distributed state must not be able to invent them.

    The operator's rank pin is dropped first: `worker_diag.rank_id()` prefers
    `VLLM_E2_TIER_DIAG_RANK` over `parallel_state`, so leaving the fixture's pin in
    place would test the env branch and call it the distributed one.
    """
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)
    _parallel_state_module()                # the fake's holder dict
    state = sys.modules[config_diag_parallel_state]
    state.get_world_group = lambda: fake.Obj(world_size=8)
    state.get_dcp_group = lambda: fake.Obj(world_size=1)
    state.get_pcp_group = lambda: fake.Obj(world_size=1)
    try:
        with a_worker_rank(1, 0):
            built = disk_store.build_worker_fingerprint(handler(with_parallel=False))
    finally:
        for name in ("get_world_group", "get_dcp_group", "get_pcp_group"):
            delattr(state, name)
    assert built["ok"] is True, built["reason"]
    assert built["fingerprint"]["world_size"] == 8
    assert built["fingerprint"]["rank"] == "PP1_TP0"
    assert "parallel_state.get_world_group()" in \
        built["sources"]["world_size"]


def test_a_process_without_distributed_state_cannot_invent_a_rank(tier, monkeypatch):
    """The EngineCore/API role: `get_pp_group()` raises because that process never
    calls `init_process_group`. The refusal must name `rank` and say the getter
    refused - not fall back to a default rank, which would make four ranks' bytes
    share one name."""
    monkeypatch.delenv(config.ENV_DIAG_RANK, raising=False)
    module = types.ModuleType(config_diag_parallel_state)

    def refusing():
        raise AssertionError("DEFAULT_PG is not initialized")
    module.get_pp_group = refusing
    module.get_tp_group = refusing
    sys.modules[config_diag_parallel_state] = module
    built = disk_store.build_worker_fingerprint(handler(with_parallel=False))
    assert built["ok"] is False
    missing = {row["field"]: row["tried"] for row in built["missing_fields"]}
    assert "rank" in missing
    assert "refused" in missing["rank"][0]
    assert "world_size" in missing


def test_a_fingerprint_build_that_fails_is_rate_limited(tier):
    """A worker whose config is unreadable binds a step every step. Re-walking
    fifteen attribute paths per step to produce the same refusal is a busy loop with
    a diagnostic attached, so the rebuild is rate-limited - while the refusal itself
    stays counted and the reason stays readable."""
    broken = handler(with_parallel=False)
    first = disk_store.adopt_fingerprint(broken)
    assert first["reason_kind"] == "fingerprint_incomplete"
    attempts_after_first = disk_store.counters()["fingerprint_attempts"]
    second = disk_store.adopt_fingerprint(broken)
    assert second["reason_kind"] == "fingerprint_incomplete"
    assert "rate-limited" in second["reason"]
    assert disk_store.counters()["fingerprint_attempts"] == attempts_after_first
    assert disk_store.counters()["refused"]["fingerprint_incomplete"] >= 1
    # `force` is the operator's way past the window (`op:spill-one` uses it).
    assert disk_store.adopt_fingerprint(broken, force=True)["reason_kind"] == \
        "fingerprint_incomplete"
    assert disk_store.counters()["fingerprint_attempts"] > attempts_after_first


def test_the_adopted_fingerprint_is_the_naming_authority(tier):
    """`disk_spill.configure` is the ONLY thing that makes a name possible, so
    adopting must go through it and a second adopt must be a no-op, not a re-write."""
    first = disk_store.adopt_fingerprint(handler())
    assert first["ok"] is True and first["cached"] is False
    assert disk_spill.geometry_digest() == first["geometry_digest"]
    again = disk_store.adopt_fingerprint(handler())
    assert again["cached"] is True
    assert again["geometry_digest"] == first["geometry_digest"]
    report = disk_store.fingerprint_report()
    assert report["configured"] is True and report["rank"] == "PP0_TP0"
    assert report["geometry_digest"] == first["geometry_digest"]


# --- 2. an unpatched engine is the normal case, and it is VISIBLE -----------

def test_an_unpatched_engine_refuses_counts_and_writes_nothing(tier):
    """Today's box, end to end. The metadata has no `store_keys`, so nothing can be
    named; the hook says which fact is missing, counts it, and writes zero bytes -
    the same build being correct on the engine that exists and the one that is
    coming."""
    assert disk_store.engine_fields()["verdict"] == "absent"
    worker = handler()
    answer = disk_store.observe_bind(worker, metadata(store_cpu_blocks=[1, 2]))
    assert answer["ok"] is False
    assert answer["reason_kind"] == "no_key_for_slot"
    assert answer["count"] == 2
    assert "engine patch" in answer["reason"]
    assert disk_store.counters()["refused"]["no_key_for_slot"] == 1
    assert disk_store.retained_view()["events"] == 0
    assert disk_spill.counters()["requests"] == 0
    assert disk_spill.counters()["thread_started"] == 0
    assert store_files(tier) == {}
    assert not os.path.exists(str(tier))


def test_the_patched_shape_is_answerable_before_a_single_store(tier):
    """"Is the engine patch live?" must be answerable without generating a store,
    because the way to make one is to put load on a serving box. The TYPE answers:
    the already-imported dataclass declares `store_keys`."""
    with metadata_shape(True):
        view = disk_store.engine_fields()
    assert view["verdict"] == "present"
    assert view["store_keys"] is True
    assert view["class_probe"] == {"store_keys": True, "store_fingerprint": True,
                                   "load_paths": True}
    # Mutable dataclass defaults have no class attribute, so a `hasattr`-only probe
    # would have called this engine unpatched. The probe reads the field list.
    assert not hasattr(disk_store.metadata_class(), "store_keys")


def test_a_process_that_cannot_see_the_type_says_unknown_not_absent(tier):
    """"Absent" is a claim about this boot's metadata; "unknown" is a claim about
    this process. Collapsing them would make an API server report that a patched
    engine is unpatched."""
    saved = sys.modules.pop(disk_store.METADATA_MODULE, None)
    try:
        view = disk_store.engine_fields()
    finally:
        sys.modules[disk_store.METADATA_MODULE] = saved
    assert view["verdict"] == "unknown"
    assert view["class_probe"] == "not imported"
    assert "not imported" in view["evidence"]


def test_a_bound_metadata_object_is_runtime_proof_of_the_fields(tier):
    """The instance probe and the type probe are different evidence, and the payload
    says which one answered: an EngineCore can only ever produce the type's."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1], store_keys=[LONG_KEY])
        disk_store.observe_bind(handler(), step)
        view = disk_store.engine_fields()
    assert view["observed_on_a_step"] is True
    assert view["fingerprint_observed_on_a_step"] is True
    assert "bound carried store_keys" in view["evidence"]


# --- 3. the pairing is never guessed ----------------------------------------

def test_a_length_mismatch_refuses_the_step_rather_than_truncating(tier):
    """`store_keys[i]` names `store_cpu_blocks[i]`, and once the two lists are in
    different hands nothing else can re-establish that. Pairing 2 keys with 3 slots
    by truncation would write one block's bytes under another block's name - the
    one failure the naming scheme exists to make impossible, so the whole step is
    refused and both numbers are named."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1, 2, 3], store_keys=[LONG_KEY, OTHER_KEY])
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is False
    assert answer["reason_kind"] == "key_count_mismatch"
    assert answer["count"] == 3 and answer["keys"] == 2
    assert "refused whole rather than truncated" in answer["reason"]
    assert disk_store.retained_view()["events"] == 0
    assert store_files(tier) == {}


def test_a_placeholder_key_costs_its_own_slot_and_not_the_step(tier):
    """The contract's placeholder is `""` INSIDE a present, correctly-sized list:
    that ONE slot has no identity, and the rest of the batch still spills. Treating a
    placeholder as a peer bug would throw away real work on every partially-hashed
    step; treating a length mismatch as a placeholder would corrupt the store."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[4, 5, 6],
                        store_keys=[LONG_KEY, disk_store.KEY_PLACEHOLDER, OTHER_KEY])
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is True, answer
    assert answer["count"] == 3
    assert disk_store.retained_view()["slots"] == 3       # the step's size, reported
    kept = disk_store._retained[step.store_event]["pairs"]
    assert [slot for slot, _key in kept] == [4, 6]        # the placeholder is gone
    assert [key for _slot, key in kept] == [KEY, OTHER_KEY]


def test_a_key_that_is_not_a_usable_hash_refuses_that_slot_only(tier):
    """A junk entry is one slot's bad luck, not the batch's - and it is COUNTED as
    `bad_key_text`, because a step where every key is junk is a wire bug worth
    seeing, while one junk key among good ones is the engine's placeholder rule
    arriving slightly differently."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1, 2, 3],
                        store_keys=[LONG_KEY, "not-a-hash", OTHER_KEY])
        answer = disk_store.observe_bind(handler(), step)
        counters = disk_store.counters()
    assert answer["ok"] is True
    assert counters["refused"]["bad_key_text"] == 1
    kept = disk_store._retained[step.store_event]["pairs"]
    assert [slot for slot, _key in kept] == [1, 3]


def test_every_slot_unnamed_is_a_refusal_and_not_an_empty_retention(tier):
    """A step whose every key is the placeholder cannot name anything: refusing it
    is the same statement as `store_keys` being absent, one level down."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1, 2], store_keys=["", ""])
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is False
    assert answer["reason_kind"] == "no_key_for_slot"
    assert "2 of 2" in answer["reason"]


@pytest.mark.parametrize(("raw", "expected", "why"), [
    (LONG_KEY, KEY, "the layout's own truncation: first 16 hex chars"),
    (LONG_KEY.upper(), KEY, "hex is case-folded, so one hash is one key"),
    ("  " + LONG_KEY + "  ", KEY, "surrounding space is not part of a key"),
    (KEY, KEY, "already a key16"),
    ("0" * 15, None, "one char short of the identity the name carries"),
    ("a3f0b7c1d4e5f60g", None, "not hexadecimal"),
    ("", None, "the placeholder: no hash for this slot"),
    (None, None, "not text at all"),
    (7, None, "an int is not a block hash"),
])
def test_the_key_is_the_layouts_truncation_and_nothing_inventive(raw, expected, why):
    """`BlockHashWithGroupId` is bytes (a digest plus a 4-byte group id) and the
    engine sends its hex - far more than 16 chars. Taking the first 16 is what
    `disk_layout.validate_key16` DOCUMENTS the key to be, so this is a contract, not
    a choice: same engine, same hash, same 16 chars, in every process."""
    key, reason = disk_store.key16(raw)
    assert key == expected, why
    assert reason is None if expected else bool(reason)


# --- 4. a completed store becomes a real file at the right name -------------

def fork_worker_instance(cls, caches):
    """An instance of the fake fork worker, holding what a real worker holds."""
    instance = cls()
    role = handler(caches)
    instance.cpu_kv_caches = role.cpu_kv_caches
    instance.vllm_config = role.vllm_config
    instance.kv_cache_config = role.kv_cache_config
    return instance


def test_a_completed_store_writes_the_slots_bytes_at_the_named_path(tier,
                                                                    worker_class):
    """THE round trip, through the fork's own two methods.

    `bind_connector_metadata` keeps the pairing, `get_finished` learns the store
    finished (the fake does it the way `worker.py:330-334` does, by moving pending
    events into `_completed_store_events`), and the bytes land on disk on the spill
    thread - never on the step thread. The assertion that matters is the second one:
    the file's CONTENT is the region's own bytes for that slot, read out of real
    host memory by the same copy the worker performs.
    """
    assert disk_store.install()["installed"] is True
    caches = {"g0": FakeRegion(8, SLOT_BYTES)}
    worker = fork_worker_instance(worker_class, caches)
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[5], store_keys=[LONG_KEY])
        worker.bind_connector_metadata(step)
        assert disk_store.retained_view()["events"] == 1
        worker.delivered = [step.store_event]
        worker.get_finished(set())
    assert disk_store.counters()["spill_queued"] == 1
    drained = disk_spill.drain()
    assert drained["drained"] is True, drained
    files = store_files(tier)
    assert len(files) == 1, files
    ((relative, payload),) = files.items()
    assert payload == caches["g0"].slot_bytes(5)
    named = disk_layout.build_slot_path(str(tier), KEY,
                                        disk_spill.geometry_digest())
    assert os.path.relpath(named["path"], str(tier)) == relative
    # The engine paid nothing for this: the fork's own counters moved, and the
    # spill thread did the read and the write.
    assert disk_spill.counters()["thread_alive"] is True
    assert disk_spill.counters()["bytes_written"] == SLOT_BYTES


def test_two_ranks_cannot_write_under_each_others_name(tier, worker_class):
    """THE reason the digest is computed per rank, and the reason the impossible
    contract had to be replaced: PP1's slot is 12,902,400 B where PP0's is
    11,468,800 B, so PP1 cannot form PP0's name even by copying the key. The
    isolation is now carried by the identity itself rather than by an operator
    keeping four configurations apart."""
    pp0 = disk_store.build_worker_fingerprint(handler(rank_width=SLOT_BYTES))
    pp1 = disk_store.build_worker_fingerprint(handler(rank_width=SLOT_BYTES + 4096))
    assert pp0["ok"] and pp1["ok"]
    assert pp0["fingerprint"]["total_bytes_per_block"] != \
        pp1["fingerprint"]["total_bytes_per_block"]
    first = disk_layout.build_slot_path(str(tier), KEY, pp0["geometry_digest"])
    second = disk_layout.build_slot_path(str(tier), KEY, pp1["geometry_digest"])
    assert first["path"] != second["path"]
    assert first["filename"] != second["filename"]
    # A rank's id is in its digest too, so two ranks of ONE stage never collide.
    monkey_rank = disk_store.build_worker_fingerprint(handler())
    assert monkey_rank["fingerprint"]["rank"] == "PP0_TP0"


def test_a_configuration_change_produces_a_different_filename(tier, monkeypatch):
    """The property the whole scheme exists for: change the configuration and the
    old files stop being findable. Not "the loader rejects them" - un-findability is
    the mechanism, because a loader that mis-reads stale KV bytes is silent."""
    base = disk_store.build_worker_fingerprint(handler())
    variants = {}
    role = handler()
    role.vllm_config.model_config.model = "flash-next-v3"
    variants["model"] = disk_store.build_worker_fingerprint(role)
    role2 = handler()
    role2.vllm_config.cache_config.prefix_caching_hash_algo = "sha256"
    variants["algo"] = disk_store.build_worker_fingerprint(role2)
    role3 = handler()
    role3.kv_cache_config.kv_cache_groups[1].kv_cache_spec.block_size = 8
    variants["groups"] = disk_store.build_worker_fingerprint(role3)
    digests = {base["geometry_digest"]}
    for name, built in variants.items():
        assert built["ok"] is True, (name, built["reason"])
        assert built["geometry_digest"] != base["geometry_digest"], name
        digests.add(built["geometry_digest"])
    assert len(digests) == 4


# --- 5. store_fingerprint is a cross-check, and empty means nothing ---------

def test_an_empty_cross_check_is_nothing_and_the_step_is_kept(tier):
    """The patched engine sends `""` today. Reading that as "no fingerprint, refuse"
    is the mistake the field invites, and it is the difference between a tier that
    works on the engine that exists and one that never spills a byte."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1], store_keys=[LONG_KEY],
                        store_fingerprint="")
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is True, answer
    assert answer["digest"] == disk_store.counters()["geometry_digest"]


def test_a_cross_check_that_disagrees_refuses_the_step_and_names_both(tier):
    """A non-empty `store_fingerprint` that is not this rank's digest means two
    sites computed the same field and got different answers. Spilling under either
    is wrong: the writer's own digest is what a later reader of THIS rank will
    reconstruct, and the sender's is what something else thinks the layout is. So
    nothing is written, and the refusal names both digests."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1], store_keys=[LONG_KEY],
                        store_fingerprint="abcdef012345")
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is False
    assert answer["reason_kind"] == "fingerprint_mismatch"
    assert answer["scheduler_digest"] == "abcdef012345"
    assert answer["worker_digest"] == disk_store.counters()["geometry_digest"]
    assert disk_store.retained_view()["events"] == 0
    assert disk_store.counters()["refused"]["fingerprint_mismatch"] == 1


def test_a_cross_check_that_agrees_is_accepted(tier):
    own = disk_store.adopt_fingerprint(handler())["geometry_digest"]
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1], store_keys=[LONG_KEY],
                        store_fingerprint=own)
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is True and answer["digest"] == own


def test_a_malformed_cross_check_is_refused_as_bad_digest(tier):
    """Not a digest at all is a different fault from a disagreeing one: the first
    says the sender is not producing what the contract says, the second says two
    producers disagree. Both refuse the step; only the second names two digests."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1], store_keys=[LONG_KEY],
                        store_fingerprint="not-a-digest")
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is False
    assert answer["reason_kind"] == "bad_digest"


# --- 6. off means off, and the cost of off is a couple of dict lookups ------

def test_disarmed_the_per_step_path_reads_no_config_and_touches_nothing(
        tier, monkeypatch):
    """The gate closes BEFORE the fingerprint build, so an unarmed tier does not
    walk fifteen attribute paths per step, does not start a thread, and does not
    create the store directory. This is the state every boot starts in, and it has to
    be cheap enough to leave installed - which is why the hook has no env-var gate of
    its own: the gate that matters is the one the operator can move at runtime."""
    disk_spill.disarm()
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1, 2], store_keys=[LONG_KEY, OTHER_KEY])
        answer = disk_store.observe_bind(handler(), step)
    counters = disk_store.counters()
    assert answer["ok"] is True and answer["retained"] is False
    assert "not configured and armed" in answer["reason"]
    assert counters["fingerprint_attempts"] == 0
    assert counters["events_retained"] == 0
    assert sum(counters["refused"].values()) == 0
    assert disk_spill.counters()["thread_started"] == 0
    assert not os.path.exists(str(tier))
    # `get_finished` asks the same gate, so the completion path is inert too.
    assert disk_store.completed_snapshot(handler()) is None
    assert disk_store.poll_spill_one(handler())["polled"] is False


def test_no_root_closes_the_gate_before_anything_is_read(monkeypatch, tier):
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1], store_keys=[LONG_KEY])
        answer = disk_store.observe_bind(handler(), step)
    assert answer["retained"] is False
    assert disk_store.counters()["fingerprint_attempts"] == 0


def test_an_idle_step_is_not_a_refusal(tier):
    """The fork allocates a store event only when it has blocks to park, so a step
    without one is the normal case: millions of them per boot. Counting it as a
    refusal would make the refusal counter meaningless, and the whole point of the
    counter is that a non-zero `refused` MEANS something."""
    step = metadata(store_cpu_blocks=[], store_event=disk_store.INVALID_JOB_ID)
    answer = disk_store.observe_bind(handler(), step)
    counters = disk_store.counters()
    assert answer["ok"] is True and answer["retained"] is False
    assert counters["steps_idle"] == 1
    assert sum(counters["refused"].values()) == 0


# --- 7. the bounds are real, and every one of them is counted ---------------

def test_a_step_beyond_the_slot_cap_is_refused_whole(tier, monkeypatch):
    """One step that stored more slots than the retention record may hold is refused
    rather than partially remembered: a half-remembered step spills half a batch
    under real names, and the operator has no way to learn which half."""
    monkeypatch.setattr(disk_store, "MAX_SLOTS_PER_EVENT", 4)
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=list(range(5)),
                        store_keys=[LONG_KEY] * 5)
        answer = disk_store.observe_bind(handler(), step)
    assert answer["ok"] is False and answer["reason_kind"] == "over_slot_cap"
    assert disk_store.counters()["refused"]["over_slot_cap"] == 1
    assert disk_store.retained_view()["events"] == 0


def test_the_tick_cap_bounds_one_step_and_counts_what_it_left(tier, monkeypatch):
    """The queue and the write-rate window are `disk_spill`'s bounds; this is the
    bound on how much work one engine step may hand over, so a step that completed a
    thousand slots cannot make the step thread build a thousand queue jobs."""
    monkeypatch.setattr(disk_store, "MAX_SLOTS_PER_TICK", 2)
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[1, 2, 3, 4], store_keys=[LONG_KEY] * 4)
        disk_store.observe_bind(handler(), step)
        report = disk_store.on_store_completed(step_handler_with_completion(step),
                                               [])
    assert report["queued"] == 2
    assert report["events"][0]["spilled"] == 2
    counters = disk_store.counters()
    assert counters["slots_capped"] == 2
    assert counters["refused"]["over_tick_cap"] == 1
    assert counters["spill_queued"] == 2


def step_handler_with_completion(step):
    """A handler whose `_completed_store_events` says `step`'s event just finished -
    the state the fork's `get_finished` leaves behind, without needing the class."""
    role = handler()
    role._completed_store_events = {step.store_event: 1}
    return role


def test_retention_is_bounded_and_the_oldest_step_dies_first(tier, monkeypatch):
    """The retention map is between-call memory on a path that runs every step, so an
    unbounded one is an outage with a timer. Eviction is oldest-first and LOUD: the
    cost is a spill that never happens, which is a missed file, never a wrong one."""
    monkeypatch.setattr(disk_store, "MAX_RETAINED_EVENTS", 2)
    with metadata_shape(True):
        for event in (1, 2, 3):
            disk_store.observe_bind(handler(), metadata(store_cpu_blocks=[event],
                                                        store_keys=[LONG_KEY],
                                                        store_event=event))
    view = disk_store.retained_view()
    assert view["events"] == 2 and view["capacity"] == 2
    assert view["dropped"] == 1
    assert disk_store.counters()["retained_dropped"] == 1
    assert 1 not in disk_store._retained and 3 in disk_store._retained
    assert "evicted from retention" in disk_store.counters()["last_reason"]
    # Completing the evicted event is a counted `unretained_event`, not a spill.
    report = disk_store.on_store_completed(step_handler_with_completion(
        metadata(store_event=1)), [])
    assert report["spilled"] == 0
    assert report["events"][0]["reason_kind"] == "unretained_event"


def test_an_event_is_spilled_once_even_while_the_fork_has_not_drained_its_dict(tier):
    """`_completed_store_events` is drained into the worker metadata by
    `build_connector_worker_meta`, so an index can sit there across several steps.
    Without the de-duplication set, every one of those steps would queue the same
    slot again - the same bytes, the same name, over and over, for free bandwidth."""
    with metadata_shape(True):
        step = metadata(store_cpu_blocks=[7, 8], store_keys=[LONG_KEY, OTHER_KEY])
        role = handler()
        disk_store.observe_bind(role, step)
        role._completed_store_events = {step.store_event: 1}
        first = disk_store.on_store_completed(role, [])
        second = disk_store.on_store_completed(role, [])
    assert first["queued"] == 2
    assert second["spilled"] == 0
    assert disk_store.counters()["spill_queued"] == 2


def test_a_completed_map_too_big_to_scan_is_skipped_and_counted(tier, monkeypatch):
    """Copying a dictionary the fork might have grown large is not a thing a step
    should do. The cap is honoured by NOT diffing, and the skip is visible."""
    monkeypatch.setattr(disk_store, "MAX_COMPLETED_SCAN", 4)
    with metadata_shape(True):
        disk_store.observe_bind(handler(), metadata(store_cpu_blocks=[1],
                                                   store_keys=[LONG_KEY]))
    role = handler()
    role._completed_store_events = {index: 1 for index in range(50)}
    assert disk_store.completed_snapshot(role) is None
    assert disk_store.counters()["scan_skipped"] == 1


def test_the_retention_view_publishes_shape_and_never_the_keys(tier):
    """The retained keys are block hashes - the identity of the tenant's KV content.
    A snapshot is polled from a dashboard; it gets counts, never hashes."""
    with metadata_shape(True):
        disk_store.observe_bind(handler(), metadata(store_cpu_blocks=[1],
                                                   store_keys=[LONG_KEY]))
    text = json.dumps(disk_store.retained_view(), default=str)
    assert KEY not in text and LONG_KEY not in text
    assert disk_store.retained_view()["events"] == 1


# --- 8. composition: the fork first, the value back, the siblings intact ----

def test_install_wraps_both_methods_and_returns_the_forks_value_untouched(
        tier, worker_class):
    """`get_finished`'s return value is the fork's own, by IDENTITY. The scheduler
    unpacks it into finished-request ids and a connector output; a wrapper that
    "helpfully" rebuilt the tuple would be a silent behaviour change on a path that
    runs every step."""
    assert disk_store.install()["installed"] is True
    sentinel = ({"req-1"}, object())
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    worker.finished_return = sentinel
    assert worker.get_finished(set()) is sentinel
    assert worker.get_finished_calls == 1
    assert disk_store.counters()["completions"] == 0   # nothing was retained
    # Both methods are ours now, and each remembers the fork's own underneath.
    for name in disk_store.WRAPPED_METHODS:
        method = worker_class.__dict__[name]
        assert getattr(method, disk_store.OWNER_ATTR) == disk_store.OWNER_TAG
        assert callable(getattr(method, disk_store.ORIGINAL_ATTR))


def test_install_composes_on_top_of_the_sibling_layers_in_the_engines_order(
        tier, worker_class):
    """Three layers, one method. `worker_diag` writes the rank file from
    `get_finished` and `blockdump` answers byte dumps from it; this module spills
    from it too. Refusing a sibling would make the INSTALL ORDER decide whether the
    rank files exist - the lesson `blockdump` learned the hard way - so a sibling is
    composed with and recorded, and only an unknown owner is refused.

    `tier` already points `VLLM_E2_TIER_DIAG` at this test's directory, which is what
    both siblings need to install at all - so a failure here is the composition, not
    the fixtures.
    """
    assert worker_diag.ensure_installed()["installed"] is True
    assert blockdump.ensure_installed()["installed"] is True
    report = disk_store.ensure_installed()
    assert report["installed"] is True, report
    # Each wrapped method records the sibling it composed onto, so the payload shows
    # the layering instead of hiding it behind "installed: true".
    assert len(disk_store.hook_section()["wraps"]) == len(disk_store.WRAPPED_METHODS)
    # Undoing the outer layer must leave the layer under it installed, not the bare
    # fork method: the rank writer is still on the class for the next test's sake.
    disk_store.uninstall()
    for name in disk_store.WRAPPED_METHODS:
        current = worker_class.__dict__[name]
        assert getattr(current, worker_diag.OWNER_ATTR, None) == \
            worker_diag.OWNER_TAG or \
            getattr(current, blockdump.OWNER_ATTR, None) == blockdump.OWNER_TAG


def test_a_wrapper_owned_by_someone_else_is_refused_not_stacked(tier, worker_class):
    """Composing with a SIBLING is safe because its code is reviewable; stacking on a
    third-party wrapper is not, because nobody knows what it does with the return
    value or the engine thread. The refusal is a `skip_reason`, never an exception."""
    def foreign_bind(self, metadata):  # noqa: ARG001 - a stranger's wrapper
        pass
    foreign_bind.__name__ = disk_store.METHOD_BIND_METADATA
    setattr(foreign_bind, disk_store.OWNER_ATTR, "some-other-plugin/1")
    worker_class.bind_connector_metadata = foreign_bind
    report = disk_store.install()
    assert report["installed"] is False
    assert "another owner" in report["reason"]
    assert disk_store.counters()["skip_reason"] == report["reason"]
    assert worker_class.__dict__["bind_connector_metadata"] is foreign_bind


def test_a_second_install_is_idempotent_and_a_restart_reinstalls_once(
        tier, worker_class):
    """Two different double-installs, and they must behave differently.

    Calling `install()` twice in one boot is a no-op at the top of the function: the
    module already believes it is hooked, and wrapping its own wrapper would spill
    twice per completion.

    A RELOADED plugin is the other case: the process state came back clean while the
    class kept the layer from before the reload (that is exactly what
    `engine_side._on_worker_report` meets when it installs again). Only the layer
    check can see that, and it counts a REINSTALL rather than stacking - which is
    what keeps one bind equal to one spill.
    """
    assert disk_store.install()["installed"] is True
    second = disk_store.install()
    assert second["installed"] is True and "already installed" in second["reason"]
    assert disk_store.counters()["reinstalls"] == 0

    disk_store._state["installed"] = False          # the reload, as the class sees it
    third = disk_store.install()
    assert third["installed"] is True
    assert disk_store.counters()["reinstalls"] == 1
    assert disk_store.counters()["installs"] == 1   # no second wrap, no second layer

    with metadata_shape(True):
        worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
        worker.bind_connector_metadata(metadata(store_cpu_blocks=[1],
                                               store_keys=[LONG_KEY]))
    assert disk_store.counters()["binds"] == 1


def test_install_without_the_connector_imported_reports_the_gap(tier):
    """The API server and any process that never imported the connector: nothing to
    hook, and the reason is a sentence an operator can read in a snapshot."""
    saved = sys.modules.pop(disk_store.TARGET_MODULE, None)
    try:
        report = disk_store.install()
    finally:
        if saved is not None:
            sys.modules[disk_store.TARGET_MODULE] = saved
    assert report["installed"] is False
    assert "not imported in this process" in report["reason"]
    assert disk_store.counters()["skip_reason"] == report["reason"]


def test_uninstall_puts_the_forks_own_methods_back(tier, worker_class):
    """A test fixture and a clean undo, so a wrapper cannot outlive the test that
    installed it on a class the next test shares."""
    original_bind = worker_class.__dict__[disk_store.METHOD_BIND_METADATA]
    original_finished = worker_class.__dict__[disk_store.METHOD_GET_FINISHED]
    disk_store.install()
    assert worker_class.__dict__[disk_store.METHOD_BIND_METADATA] is not original_bind
    report = disk_store.uninstall()
    assert report["uninstalled"] is True
    assert report["reason"].startswith("2 of 2")
    assert worker_class.__dict__[disk_store.METHOD_BIND_METADATA] is original_bind
    assert worker_class.__dict__[disk_store.METHOD_GET_FINISHED] is original_finished


def test_the_forks_exception_still_escapes_the_wrapper(tier, worker_class):
    """A diagnostic that swallows `KeyboardInterrupt` is a different kind of outage.
    The wrappers catch `Exception` only, so a `BaseException` from the fork travels
    straight out - and the hook never ran to swallow anything."""
    class Stop(KeyboardInterrupt):
        pass

    def exploding_bind(self, metadata):  # noqa: ARG001 - the fork itself explodes
        raise Stop("the engine is being shut down")
    exploding_bind.__name__ = disk_store.METHOD_BIND_METADATA
    worker_class.bind_connector_metadata = exploding_bind
    disk_store.install()
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(2, SLOT_BYTES)})
    binds_before = disk_store.counters()["binds"]
    with pytest.raises(Stop):
        worker.bind_connector_metadata(metadata(store_cpu_blocks=[1]))
    assert disk_store.counters()["binds"] == binds_before


# --- 9. the control verb ----------------------------------------------------

def spill_slot_now(caches, slot, key):
    """Write one slot synchronously, so `ls` has something honest to list.

    `disk_spill.spill_slot` is the writer thread's core, exposed: a test that faked
    the files would be listing bytes the tier never produced. The fingerprint is
    adopted first, because that is what the worker does before it can name anything -
    a test that configured a fingerprint behind the module's back would be testing a
    state no boot reaches.
    """
    adopted = disk_store.adopt_fingerprint(handler(caches))
    assert adopted["ok"] is True, adopted
    result = disk_spill.spill_slot(caches, slot, key)
    assert result["ok"] is True, result
    return result["path"]


def test_the_verb_refuses_an_unknown_op_and_names_the_real_ones(tier):
    answer = disk_store.disk_for_engine(None, {"op": "spill-two"})
    assert answer["ok"] is False
    assert "spill-two" in answer["reason"]
    assert answer["ops"] == list(disk_store.DISK_OPS)


def test_a_safe_method_cannot_reach_a_mutating_op(tier):
    """The `op` is a query parameter, so the URL that arms a facility writing KV to
    disk is as short as the URL that reads status - and a browser sends a GET with no
    preflight and no custom header, which means any page the operator's client can
    load can issue it. The engine-side refusal is the backstop behind the route's
    405: the two must fail the same way, because a route that forgets its check must
    still not arm anything."""
    for op in disk_store.MUTATING_DISK_OPS:
        answer = disk_store.disk_for_engine(None, {"op": op, "safe_method": True})
        assert answer["ok"] is False, op
        assert "will not run from a method" in answer["reason"]
    assert disk_spill.is_armed() is True          # armed by the fixture, untouched
    disk_spill.disarm()
    for op in disk_store.MUTATING_DISK_OPS:
        assert disk_store.disk_for_engine(
            None, {"op": op, "safe_method": True})["ok"] is False
    # A read is still a read.
    assert disk_store.disk_for_engine(None, {"op": "ls",
                                             "safe_method": True})["ok"] is True


def test_arm_and_disarm_through_the_verb_move_the_real_gate(tier):
    """The verb's `arm` IS `disk_spill.arm` - one gate, one expiry, one file a worker
    reads. A second implementation of "armed" is how a tier ends up writing in a
    process nobody armed."""
    disk_spill.disarm()
    assert disk_store.disk_for_engine(None, {"op": "arm", "seconds": 5})["armed"] \
        is True
    assert disk_spill.is_armed() is True
    answer = disk_store.disk_for_engine(None, {"op": "disarm"})
    assert answer["armed"] is False
    assert disk_spill.is_armed() is False


def test_ls_lists_real_files_and_reads_nothing(tier):
    """`ls` is the operator's proof that the naming works, so it must list what the
    WRITER wrote - and prove by the mtimes and contents that a listing never writes,
    touches, or rewrites a single byte. The only destructive action in this design is
    `disk_spill`'s own LRU eviction, and `ls` is not it."""
    caches = {"g0": FakeRegion(4, SLOT_BYTES)}
    first = spill_slot_now(caches, 1, KEY)
    second = spill_slot_now(caches, 2, OTHER_KEY)
    before = store_files(tier)
    assert len(before) == 2
    answer = disk_store.disk_for_engine(None, {"op": "ls"})
    assert answer["ok"] is True and answer["read_only"] is True
    assert answer["count"] == 2
    names = {row["name"] for row in answer["files"]}
    assert names == {os.path.basename(first), os.path.basename(second)}
    assert all(row["bytes"] == SLOT_BYTES for row in answer["files"])
    assert answer["walk_used_bytes"] == 2 * SLOT_BYTES
    assert store_files(tier) == before            # the listing changed nothing


def test_ls_is_bounded_by_its_limit_and_says_so(tier):
    caches = {"g0": FakeRegion(4, SLOT_BYTES)}
    for index, key in enumerate((KEY, OTHER_KEY, "1111222233334444")):
        spill_slot_now(caches, index, key)
    answer = disk_store.disk_for_engine(None, {"op": "ls", "limit": 2})
    assert answer["count"] == 2 and answer["truncated"] is True
    # Newest first: the newest write is the one an operator wants to see.
    newest = max(os.path.getmtime(os.path.join(str(tier), *relative.split(os.sep)))
                 for relative in store_files(tier))
    assert max(row["mtime_ns"] for row in answer["files"]) / 1e9 == pytest.approx(
        newest, abs=2.0)


def test_ls_refuses_a_shard_that_is_not_two_hex_chars(tier):
    answer = disk_store.disk_for_engine(None, {"op": "ls", "shard": "../../etc"})
    assert answer["ok"] is False and "2 hex chars" in answer["reason"]


def test_ls_reports_an_unconfigured_store_without_inventing_a_root(tier, monkeypatch):
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    answer = disk_store.disk_for_engine(None, {"op": "ls"})
    assert answer["ok"] is True and answer["root"] is None
    assert answer["count"] == 0 and answer["read_only"] is True


def test_a_payload_is_trimmed_rather_than_allowed_to_grow(tier, monkeypatch):
    """A verb answer goes over the utility channel into a JSON response a dashboard
    polls every few seconds. The listing is the only part that can grow with the
    store, so it is what gets eaten - and `truncated` is what makes the silence
    legible.

    The budget is derived from the untrimmed answer rather than written as a literal:
    the payload carries a fixed explanation block, and a hard-coded number would be a
    test that passes only while that text happens to be a certain size.
    """
    caches = {"g0": FakeRegion(4, SLOT_BYTES)}
    for key in (KEY, OTHER_KEY, "1111222233334444"):
        spill_slot_now(caches, 1, key)
    full = json.dumps(disk_store.disk_for_engine(None, {"op": "ls"}),
                      default=str).encode()
    budget = len(full) - 80            # enough to force one row out, not all of them
    monkeypatch.setattr(disk_store, "MAX_RESPONSE_BYTES", budget)
    answer = disk_store.disk_for_engine(None, {"op": "ls"})
    assert len(json.dumps(answer, default=str).encode()) <= budget
    assert answer["truncated"] is True
    # Rows are eaten from the tail until the payload fits, so how many survive
    # depends on the budget's arithmetic, not on a promise: what is pinned is that
    # SOME rows went and the payload said so.
    assert 0 < answer["count"] < 3 and len(answer["files"]) == answer["count"]
    assert "trimmed" in answer["note"]
    assert answer["reason"].endswith("(TRIMMED)")
    # And when even the trimmed payload cannot fit, the answer is an honest refusal
    # rather than a silently oversized object.
    monkeypatch.setattr(disk_store, "MAX_RESPONSE_BYTES", 40)
    refused = disk_store.disk_for_engine(None, {"op": "ls"})
    assert refused["ok"] is False and "would exceed" in refused["reason"]


def test_the_status_reports_one_copy_of_the_identity_and_the_arms_state(tier):
    """Two copies of one fingerprint in one payload is how a naming argument starts,
    so `op:status` publishes it once, at the top level, and the hook block points
    there instead of repeating it."""
    disk_store.adopt_fingerprint(handler())
    answer = disk_store.disk_for_engine(None, {"op": "status"})
    assert answer["rank"] == "PP0_TP0" and answer["rank_source"] == "env"
    assert answer["armed"] is True and answer["capability"] == "armed_and_ready"
    digest = disk_spill.geometry_digest()
    assert answer["geometry_digest"] == digest
    assert answer["worker_fingerprint"]["geometry_digest"] == digest
    assert answer["hook"]["worker_fingerprint"] == "see the top level"
    text = json.dumps(answer, default=str)
    assert text.count(digest) == 2                # the two views of ONE value
    assert answer["engine_fields"]["verdict"] in ("present", "absent", "unknown")
    assert answer["spill"]["bytes_written"] == 0  # nothing spilled in this test


def test_an_op_that_raises_becomes_data_and_the_verb_still_answers(tier, monkeypatch):
    """Every op runs inside one `except Exception`, so a bug in a future op cannot
    become a 500 on the utility channel or, worse, an exception in the EngineCore's
    utility-call handler."""
    def explode(_body):
        raise ZeroDivisionError("a future op divided by zero")
    monkeypatch.setitem(disk_store._DISK_OPS, "ls", explode)
    answer = disk_store.disk_for_engine(None, {"op": "ls"})
    assert answer["ok"] is False
    assert "ZeroDivisionError" in answer["reason"]
    assert answer["engine_fields"]["verdict"] in ("present", "absent", "unknown")
    assert disk_store.counters()["errors"] == 1


# --- 10. spill-one: one EngineCore ask, answered by every rank --------------

def test_spill_one_requires_the_arm(tier):
    """A request every rank must refuse is not a diagnostic, it is litter: the file
    would sit in the channel until it expires, and each rank would read it and answer
    `not_armed`. The EngineCore checks the SAME gate a worker does - the arm FILE -
    so this is honest across the process boundary, not a read of memory it shares
    with nobody."""
    disk_spill.disarm()
    answer = disk_store.disk_for_engine(None, {"op": "spill-one", "slot": 2,
                                               "key": KEY})
    assert answer["ok"] is False and "not armed" in answer["reason"]
    assert disk_store._pending_path() is None or not os.path.exists(
        disk_store._pending_path())


def test_spill_one_needs_a_key_the_layout_would_accept(tier):
    for bad in (None, "", "zz", KEY * 4):
        answer = disk_store.disk_for_engine(None, {"op": "spill-one", "slot": 1,
                                                   "key": bad})
        assert answer["ok"] is False, bad
    assert disk_store.disk_for_engine(None, {"op": "spill-one", "slot": -1,
                                             "key": KEY})["ok"] is False


def test_a_rank_answers_the_ask_and_the_file_lands_where_the_receipt_said(
        tier, worker_class):
    """The relay's whole claim, end to end: a request file appears, a worker polls it
    on its next step, the spill is QUEUED (not performed on the step thread), the
    receipt names the file this rank will write, and the file is really there with the
    slot's bytes.

    The receipt says `queued`, never `written`: promising a path before the writer
    thread has run would be a claim the queue can still invalidate by dropping.
    """
    assert disk_store.install()["installed"] is True
    caches = {"g0": FakeRegion(6, SLOT_BYTES)}
    worker = fork_worker_instance(worker_class, caches)
    asked = disk_store.disk_for_engine(None, {"op": "spill-one", "slot": 3,
                                              "key": KEY, "wait_s": 0})
    assert asked["ok"] is True
    assert asked["completeness"] == "unknown"      # no handshake recorded here
    assert asked["receipts"] == {}
    polled = disk_store.poll_spill_one(worker)
    assert polled["polled"] is True and polled["adopted"] is True
    answer = polled["answer"]
    assert answer["queued"] is True and answer["refused"] is False
    assert answer["rank"] == "PP0_TP0"
    assert disk_spill.drain()["drained"] is True
    files = store_files(tier)
    assert len(files) == 1
    ((relative, payload),) = files.items()
    assert payload == caches["g0"].slot_bytes(3)
    assert answer["expected_path"].endswith(relative)
    # The receipt is on disk, small, and says which rank is speaking.
    receipt_path = os.path.join(str(pathlib.Path(config.diagnostics_dir())),
                                disk_store.RECEIPT_FILE.format(rank="PP0_TP0"))
    assert os.path.exists(receipt_path)
    assert os.path.getsize(receipt_path) <= disk_store.MAX_RECEIPT_FILE_BYTES
    with open(receipt_path, "rb") as handle:
        receipt = json.loads(handle.read())
    assert receipt["rank"] == "PP0_TP0" and receipt["queued"] is True
    assert receipt["request_seq"] == asked["seq"]
    assert "THIS rank's answer only" in receipt["scope"]


def test_a_second_poll_does_not_answer_the_same_ask_twice(tier, worker_class):
    """De-duplicated by `epoch:seq:requested_ts`, because a worker can never learn
    the EngineCore's boot id (the `worker_diag` epoch problem, in the other
    direction) and a slot spilled twice per step is free work on a busy box."""
    assert disk_store.install()["installed"] is True
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    disk_store.disk_for_engine(None, {"op": "spill-one", "slot": 1, "key": KEY,
                                      "wait_s": 0})
    assert disk_store.poll_spill_one(worker)["adopted"] is True
    disk_store._state["last_poll_mono"] = None    # past the poll rate limit
    second = disk_store.poll_spill_one(worker)
    assert second["adopted"] is False
    assert second["reason"] == "already served by this rank"
    assert disk_store.counters()["fulfilled"] == 1


def test_the_poll_is_rate_limited_and_suppressed_when_the_gate_is_closed(
        tier, worker_class):
    """The poll is the only file read a step may do, so it is bounded twice: the
    gate closes it entirely when the tier is inert, and a clock bounds it to one read
    per interval however often the engine steps."""
    assert disk_store.install()["installed"] is True
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    first = disk_store.poll_spill_one(worker)
    assert first["polled"] is True
    second = disk_store.poll_spill_one(worker)
    assert second["polled"] is False and "polled recently" in second["reason"]
    disk_spill.disarm()
    disk_store._state["last_poll_mono"] = None
    third = disk_store.poll_spill_one(worker)
    assert third["polled"] is False
    assert disk_store.counters()["poll_suppressed"] >= 1


@pytest.mark.parametrize("tamper,reason_fragment", [
    ({"version": 1}, "protocol version"),
    ({"seq": 0}, "positive integer seq"),
    ({"requested_ts": "yesterday"}, "numeric requested_ts"),
    ({"slot": "three"}, "names no slot"),
    ({"key": "not-a-key"}, "not 16 hex characters"),
    ({"fingerprint": "zzz"}, "neither empty nor 12 hex"),
])
def test_a_request_that_does_not_match_the_contract_is_counted_and_ignored(
        tier, worker_class, tamper, reason_fragment):
    """The pending file is untrusted input from another process - possibly an older
    build of this plugin, possibly a hand-edited file. Every field is checked, the
    malformed count moves, and the worker does the one safe thing: it says nothing."""
    assert disk_store.install()["installed"] is True
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    directory = config.diagnostics_dir()
    spill_one_request(directory, **tamper)
    polled = disk_store.poll_spill_one(worker)
    assert polled["adopted"] is False
    assert reason_fragment in polled["reason"]
    assert disk_store.counters()["pending_malformed"] == 1
    assert disk_store.counters()["fulfilled"] == 0
    assert store_files(tier) == {}


def test_an_expired_ask_is_refused_and_said_so(tier, worker_class):
    """The TTL exists because a request the operator wrote twenty minutes ago is a
    question about a slot that may have been recycled since; answering it would spill
    whatever the slot holds NOW under whatever name the old key implies."""
    assert disk_store.install()["installed"] is True
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    spill_one_request(config.diagnostics_dir(), expires_ts=1.0)
    polled = disk_store.poll_spill_one(worker)
    assert polled["adopted"] is False
    assert "expired" in polled["reason"]
    assert disk_store.counters()["refused"]["expired"] == 1


def test_a_rank_refuses_an_ask_whose_fingerprint_is_not_its_own(tier, worker_class):
    """The cross-check, on the relay path too. A request that disagrees with this
    rank's own digest means two sites computed the same field differently, and the
    answer is a refusal with both digests in it - never "mine is newer"."""
    assert disk_store.install()["installed"] is True
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    spill_one_request(config.diagnostics_dir(), fingerprint="abcdef012345")
    polled = disk_store.poll_spill_one(worker)
    answer = polled["answer"]
    assert answer["refused"] is True and answer["queued"] is False
    assert "abcdef012345" in answer["reason"]
    assert disk_spill.geometry_digest() in answer["reason"]
    assert store_files(tier) == {}


def test_a_huge_or_unparseable_request_file_is_refused_without_being_read(
        tier, worker_class, monkeypatch):
    """Two ways a file can be unwelcome: too big to trust, and too broken to parse.
    Both are counted as malformed, and the size check happens BEFORE the read."""
    assert disk_store.install()["installed"] is True
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    path = os.path.join(str(config.diagnostics_dir()), disk_store.PENDING_FILE)
    with open(path, "wb") as handle:
        handle.write(b"[" * (disk_store.MAX_PENDING_FILE_BYTES + 1))
    assert "refusing to trust it" in disk_store.poll_spill_one(worker)["reason"]
    disk_store._state["last_poll_mono"] = None    # past the poll rate limit
    with open(path, "wb") as handle:
        handle.write(b"{not json")
    assert "did not parse" in disk_store.poll_spill_one(worker)["reason"]
    assert disk_store.counters()["pending_malformed"] == 2


def test_a_rank_without_a_diagnostics_directory_cannot_be_asked_anything(
        tier, monkeypatch, worker_class):
    monkeypatch.delenv(config.ENV_DIAG_DIR, raising=False)
    assert disk_store.install()["installed"] is True
    worker = fork_worker_instance(worker_class, {"g0": FakeRegion(4, SLOT_BYTES)})
    assert disk_store.poll_spill_one(worker)["polled"] is False
    assert disk_store.fulfil_spill_one(worker, {"seq": 1, "slot": 1, "key": KEY,
                                                "epoch": "e",
                                                "requested_ts": 1.0})["written"] \
        is False


# --- 11. nothing raises, nothing blocks, nothing leaks ----------------------

def hostile_handler():
    role = handler(caches={"g0": HostileRegion()})
    role.vllm_config = None
    role.kv_cache_config = 17
    return role


DOORS = [
    ("observe_bind", lambda x: disk_store.observe_bind(hostile_handler(), x)),
    ("observe_bind_none", lambda x: disk_store.observe_bind(None, None)),
    ("completed_snapshot", lambda x: disk_store.completed_snapshot(x)),
    ("on_store_completed", lambda x: disk_store.on_store_completed(x, [1, 2])),
    ("poll_spill_one", lambda x: disk_store.poll_spill_one(x)),
    ("fulfil_spill_one", lambda x: disk_store.fulfil_spill_one(
        x, {"seq": 3, "slot": 1, "key": KEY, "epoch": "e",
            "requested_ts": 1.0, "expires_ts": 9e12, "fingerprint": ""})),
    ("resolve_store_keys", lambda x: disk_store.resolve_store_keys(x)),
    ("build_worker_fingerprint", lambda x: disk_store.build_worker_fingerprint(x)),
    ("adopt_fingerprint", lambda x: disk_store.adopt_fingerprint(x)),
    ("verb", lambda x: disk_store.disk_for_engine(x, {"op": x})),
    ("verb_weird_body", lambda x: disk_store.disk_for_engine(x, "not a dict")),
]


@pytest.mark.parametrize("name,door", DOORS)
def test_no_entry_point_can_raise_an_exception(name, door, tier):
    """Every door is opened with the worst thing this file can build - a tensor whose
    attributes explode, a config that is None, a metadata object that is a bare int,
    an `op` that is an object. A diagnostic that raises into `bind_connector_metadata`
    or `get_finished` stops a serving engine, so the contract is a dict and a counter,
    never an exception. (`BaseException` is the exception, and
    `test_the_forks_exception_still_escapes_the_wrapper` pins it.)"""
    victims = [hostile_handler(), None, 17, "metadata", [1, 2],
               metadata(store_cpu_blocks=[1]), object()]
    for victim in victims:
        try:
            outcome = door(victim)
        except BaseException as exc:  # noqa: BLE001 - the assertion is the point
            pytest.fail(f"{name} raised {type(exc).__name__} on {victim!r}: {exc}")
        assert isinstance(outcome, (dict, frozenset, type(None))), name


def test_a_hostile_tensor_is_a_refusal_that_names_the_field(tier):
    """`read_slot_bytes` refuses a non-contiguous region because an address that
    could leave the allocation is a SIGSEGV, not an exception. The fingerprint path
    inherits that posture: an inspecting call that raises becomes a named missing
    field, because "I cannot measure the slot width" is a fact and 0 is a lie."""
    built = disk_store.build_worker_fingerprint(handler(caches={"g0":
                                                                HostileRegion()}))
    assert built["ok"] is False
    missing = {row["field"] for row in built["missing_fields"]}
    assert "total_bytes_per_block" in missing
    assert "raised RuntimeError" in built["missing_fields"][0]["tried"][0]


def test_a_fingerprint_that_is_unreadable_never_reaches_the_writer(tier):
    """The order is the safety: the fingerprint is adopted BEFORE anything is
    retained, so a worker that cannot name a file never queues a job for one."""
    broken = handler(with_parallel=False)
    monkey = broken
    monkey.vllm_config.parallel_config = None
    with metadata_shape(True):
        answer = disk_store.observe_bind(broken, metadata(store_cpu_blocks=[1],
                                                          store_keys=[LONG_KEY]))
    assert answer["ok"] is False
    assert answer["reason_kind"] == "fingerprint_incomplete"
    assert disk_spill.counters()["requests"] == 0
    assert disk_store.retained_view()["events"] == 0


# --- 12. what the snapshot and the source promise ---------------------------

def test_the_snapshot_reports_the_wiring_and_survives_a_broken_section(monkeypatch):
    """`snapshot` is polled every few seconds by a dashboard. A section that raises
    becomes `{"error": "ZeroDivisionError"}` and the rest of the snapshot still
    ships - the same posture as every other dev-state section here."""
    view = snapshot.dev_state_sections()
    assert view["disk_store"]["wiring"].startswith("THE ENGINE CALLS THIS")
    assert view["disk_store_installed"] in (True, False)

    def explode():
        raise ZeroDivisionError("the section divided by zero")
    monkeypatch.setattr(disk_store, "section", explode)
    view = snapshot.dev_state_sections()
    assert view["disk_store"] == {"error": "ZeroDivisionError"}
    assert view["disk_store_installed"] is None
    assert "disk_spill" in view                    # its sibling is untouched


def test_the_section_shape_does_not_depend_on_the_arm(tier, monkeypatch):
    """A polled payload whose KEYS move when the gate moves is a payload a dashboard
    cannot render and a test cannot compare. The values change; the keys do not."""
    armed = set(disk_store.section())
    disk_spill.disarm()
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    disk_store.reset_state_for_tests()
    assert set(disk_store.section()) == armed


def test_the_wiring_says_what_an_unpatched_boot_does(tier):
    """The claim is in the payload, not only in this file's docstring: a reader of
    `/snapshot` can see that the hook is installed AND that the engine is not
    carrying the field it needs, in the same object."""
    section = disk_store.section()
    assert "bind_connector_metadata" in section["wiring"]
    assert "one scheduler digest cannot be four ranks" in section["wiring"]
    assert section["engine_fields"]["verdict"] == "absent"
    assert "engine_fields=absent" in section["wiring"]


def test_disk_store_imports_only_the_standard_library_and_its_own_package():
    """The plugin is loaded into a serving engine by a `sitecustomize`-style hook, so
    its import graph is a stability surface: no engine symbol at import time, no
    third-party dependency, and nothing that can fail to resolve on the box.

    The AST walk covers function-level imports too - `from . import worker_diag`
    inside a function is exactly as much a dependency as a top-level one, and it is
    where a cycle would hide.
    """
    tree = ast.parse(pathlib.Path(disk_store.__file__).read_text())
    external: set = set()
    local: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            external |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                local.add(node.module or "")
            elif node.module:
                external.add(node.module.split(".")[0])
    assert external <= {"__future__", "contextlib", "dataclasses", "json", "os",
                        "sys", "time"}, sorted(external)
    package = disk_store.__name__.rpartition(".")[0]
    assert local == {"", package} or local <= {"", package}
    # And the engine is reached by NAME only: no vllm, no torch, at any nesting.
    assert not {name for name in external if name in ("vllm", "torch", "numpy")}
    source = pathlib.Path(disk_store.__file__).read_text()
    assert "import vllm" not in source and "import torch" not in source


# --- 13. the boot path: engine_side is what puts this module on the class ----

def test_the_boot_path_installs_the_spill_hook_last_and_the_verb_is_attached(
        tier, worker_class):
    """Nothing in this module runs by itself at boot: `engine_side` attaches the
    utility verb and drives the three worker-class listeners off the capacity
    handshake. Two things are pinned here, both of which broke silently once in this
    package's history:

    * the verb is in the factory table, so `/plugins/kvtier/disk` reaches a handler
      instead of failing as an unknown utility call;
    * the handshake hook calls every listener as `hook(connector)`, so a listener
      whose signature does not accept that argument is skipped SILENTLY (each call is
      contained, by design) and the tier is never installed. The nesting count is the
      proof: this hook landed on top of the two siblings, which only happens if it
      actually ran.
    """
    assert config.METHOD_DISK in engine_side._FACTORIES
    engine_side._on_worker_report(connector=object())
    counters = disk_store.counters()
    assert counters["installed"] is True, counters["skip_reason"]
    assert len(counters["nested"]) == len(disk_store.WRAPPED_METHODS)
    note = disk_store.boot_note()
    assert "installed on SimpleCPUOffloadWorker" in note
    assert "engine_fields=absent" in note            # the autouse unpatched shape
    assert "armed" in note and "store_keys" in note


def test_the_boot_note_is_a_sentence_about_the_real_posture(tier, worker_class,
                                                            monkeypatch):
    """A boot log is the only place the two halves of "why is nothing spilling" meet:
    the hook's own state and the engine's. Both belong in the one line."""
    saved = sys.modules.pop(disk_store.TARGET_MODULE, None)
    try:
        assert disk_store.boot_note().startswith("NOT installed (never attempted")
    finally:
        if saved is not None:
            sys.modules[disk_store.TARGET_MODULE] = saved
    disk_store.install()
    disk_spill.disarm()
    monkeypatch.delenv(config.ENV_DISK_ROOT, raising=False)
    note = disk_store.boot_note()
    assert "disarmed, NO root configured" in note


def test_the_fingerprint_says_which_fields_are_this_ranks_own(tier):
    """A reader who sees four digests for one model needs the explanation IN the
    payload: these two fields are measured and resolved per rank, so four ranks MUST
    have four digests and that is the isolation, not a fault. Without this line the
    natural conclusion is "the fingerprint is broken", which is the misreading this
    whole module exists to retire."""
    disk_store.adopt_fingerprint(handler())
    report = disk_store.fingerprint_report()
    assert report["rank_local_fields"] == list(disk_store.LOCAL_FIELDS)
    assert "total_bytes_per_block" in report["rank_local_fields"]
    assert "rank" in report["rank_local_fields"]
    assert "model_id" not in report["rank_local_fields"]


def test_the_status_route_advertises_the_ops_the_verb_actually_accepts(enabled):
    """The route's self-description and the verb's dispatch table are one list.

    A documented op the verb refuses is a support call; an op the verb accepts that
    the route does not mention is an undocumented way to write KV bytes to disk. Both
    are prevented by making the two tables agree here, rather than in prose.
    """
    status = client_for(build_app(FakeAsyncLLM(FakeEngineCoreClient()))
                        ).get(STATUS).json()
    block = status["disk"]
    assert block["ops"] == list(disk_store.DISK_OPS)
    # Nothing dispatched in secret: the table the verb consults is the table the
    # route describes.
    assert block["ops"] == list(disk_store._DISK_OPS)
    assert block["route"] == config.ROUTE_PREFIX + config.ROUTE_DISK
    assert block["root_env"] == config.ENV_DISK_ROOT
