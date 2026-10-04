"""Fake engine structures for the DEBUG verbs (`kvconfig`, `inspect`, `trace`,
`guard`, the worker writer), built with the FORK's attribute names and nothing
else.

The rule this file exists to enforce: a debug verb reads by NAME and never
imports vllm, so a fake with the right names is a complete test double. Every
name here is copied from the fork at HEAD 10b8e07:

    vllm/v1/core/kv_cache_interface.py   KVCacheSpec / AttentionSpec /
                                         CircularBufferSpec / MambaSpec /
                                         KVCacheGroupSpec / KVCacheConfig
    vllm/v1/core/block_pool.py           BlockPool, KVCacheBlock
    vllm/v1/core/kv_cache_utils.py       BlockHashWithGroupId (bytes + 4-byte
                                         group id, :86-92), the divisibility
                                         assert (:2888)
    vllm/v1/core/sched/single_type_kv_cache_manager.py   the per-group manager
    vllm/v1/simple_kv_offload/manager.py the tier (group_block_sizes,
                                         hash_block_size, cpu_block_pool)

`fork_class()` is the second half of the idiom: `guard_mode`, `worker_diag` and
`cpu_capacity` only wrap a function whose `__module__` is the fork's own module,
so a stub class has to be exec'd into a module bearing that name (the same trick
`tests/test_cpu_capacity.py` uses).
"""

from __future__ import annotations

import sys
import types

_NAMED_CLASSES: dict = {}


def _named_class(name: str):
    """One real class per name, so `type(x).__name__` is the FORK's name.

    The verbs report `type(spec).__name__`, `type(manager).__name__` and the MRO
    class names - they read the CLASS, not a `__name__` attribute - so a fake that
    only carried a label would report "Obj" everywhere and the tests would be
    asserting nothing. A class per name is cheap and makes the double honest.
    """
    cls = _NAMED_CLASSES.get(name)
    if cls is None:
        cls = type(name, (Obj,), {})
        _NAMED_CLASSES[name] = cls
    return cls


class Obj:
    """A namespace whose TYPE carries the fork's class name.

    `Obj(name="BlockPool", ...)` is an instance of a class literally named
    `BlockPool`; a bare `Obj()` keeps the name `Obj`.
    """

    def __new__(cls, name=None, **fields):     # noqa: ARG002 - signature contract
        if cls is Obj and isinstance(name, str) and name:
            return _named_class(name)()
        return super().__new__(cls)

    def __init__(self, name=None, **fields):
        self.__name__ = name or type(self).__name__
        self.__dict__.update(fields)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {sorted(self.__dict__)}>"


# --- specs ------------------------------------------------------------------

def attention_spec(block_size=1024, page_size_bytes=16384, prefix_cacheable=True,
                   name="FullAttentionSpec", **extra):
    """The FA group's spec: the fields the tier's stride depends on."""
    spec = _named_class(name)()
    spec.__dict__.update(block_size=block_size,
               page_size_bytes=page_size_bytes, prefix_cacheable=prefix_cacheable,
               tokens_per_state=1, num_states=1, state_content_size_bytes=16384,
               num_heads=8, head_size=128, head_size_v=128, num_kv_heads=2,
               dtype="torch.bfloat16", sliding_window=None,
               attention_chunk_size=None, page_size_padded=None,
               uses_slot_mapping=True, block_table_token_alignment=1,
               kv_quant_mode="none", non_causal=False)
    spec.__dict__.update(extra)
    return spec


class CircularBufferSpec:
    """Named EXACTLY like the fork's class (kv_cache_interface.py:876-897).

    `is_circular_buffer()` decides by class name through the MRO, because
    `isinstance` would need the import this plugin is forbidden to do. A fake that
    is merely shaped like a ring must therefore NOT be recognised - which is what
    `test_is_circular_buffer_matches_the_name_not_the_shape` pins.
    """


class QsaRingSpec(CircularBufferSpec):
    """A fork SUBCLASS of the ring spec: still a ring, by the MRO walk."""


def circular_buffer_spec(block_size=1024, page_size_bytes=4096, subclass=True):
    """The scratch ring group's spec: one block per request, never parked."""
    spec = (QsaRingSpec() if subclass else CircularBufferSpec())
    spec.__dict__.update(block_size=block_size,
                         page_size_bytes=page_size_bytes,
                         prefix_cacheable=False, tokens_per_state=1, num_states=1,
                         state_content_size_bytes=page_size_bytes,
                         uses_slot_mapping=False, mamba_type="qsa_ring")
    return spec


def mamba_spec(block_size=128, page_size_bytes=8192):
    spec = _named_class("MambaSpec")()
    spec.__dict__.update(block_size=block_size,
               page_size_bytes=page_size_bytes, prefix_cacheable=True,
               tokens_per_state=128, num_states=1,
               state_content_size_bytes=8192, shapes=[(128, 256)],
               dtypes=["torch.bfloat16"], mamba_type="gdn",
               mamba_cache_mode="aligned", num_speculative_blocks=0,
               num_prefill_checkpoint_blocks=1,
               prefill_checkpoint_alignment=128, tp_replicated=False,
               uses_slot_mapping=False)
    return spec


class Fraction:
    """`tokens_per_state` is `int | Fraction` (kv_cache_interface.py:492)."""

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.numerator}/{self.denominator}"


# --- blocks and pools -------------------------------------------------------

class CachedMap:
    """`cached_block_hash_to_block`, duck-typed the way the tier uses it."""

    def __init__(self, entries=None):
        self._entries = dict(entries or {})

    def __len__(self) -> int:
        return len(self._entries)

    def get_one_block(self, key):
        return self._entries.get(bytes(key))


class Block:
    """`KVCacheBlock` (block_pool.py): id, ref count, hash, free-list links."""

    def __init__(self, block_id, block_hash=None, ref_cnt=0, is_null=False,
                 prev_free_block=None, next_free_block=None, num_tokens=None):
        self.block_id = block_id
        self.block_hash = block_hash
        self.block_hash_num_tokens = num_tokens
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.prev_free_block = prev_free_block
        self.next_free_block = next_free_block


class Pool:
    """`BlockPool`: ONE instance shared by every group (kv_cache_coordinator)."""

    def __init__(self, blocks=None, num_gpu_blocks=None, free=0, cached=None,
                 raise_on_free=False):
        self.blocks = list(blocks or [])
        self.num_gpu_blocks = num_gpu_blocks if num_gpu_blocks is not None \
            else len(self.blocks)
        self._free = free
        self.cached_block_hash_to_block = CachedMap(cached)
        self.free_block_queue = Obj(name="FreeKVCacheBlockQueue",
                                    num_free_blocks=free)
        self.raise_on_free = raise_on_free

    def get_num_free_blocks(self):
        if self.raise_on_free:
            raise RuntimeError("the pool refused to answer")
        return self._free


def hash_key(num_tokens_seed: int, group_id: int = 0, size: int = 20) -> bytes:
    """A `BlockHashWithGroupId`: the hash bytes then a 4-byte big-endian group id
    (kv_cache_utils.py:86-92)."""
    return bytes([num_tokens_seed % 251] * size) + group_id.to_bytes(4, "big")


# --- the scheduler side -----------------------------------------------------

def manager_item(group_id, spec, req_to_blocks=None, block_size=None,
                 stable=True):
    """One entry of `coordinator.single_type_managers`."""
    return Obj(name="KVCacheManager",
               kv_cache_group_id=group_id,
               # getattr, not attribute access: a fake with a field deleted is how
               # the "the fork renamed it" case is tested.
               block_size=getattr(spec, "block_size", None) if block_size is None
               else block_size,
               kv_cache_spec=spec,
               req_to_blocks=dict(req_to_blocks or {}),
               has_positionally_stable_blocks=stable)


def coordinator(managers, **extra):
    return Obj(name="KVCacheCoordinator", single_type_managers=list(managers),
               scheduler_block_size=extra.pop("scheduler_block_size", 1024),
               num_reprefillable_tokens=extra.pop("num_reprefillable_tokens", 0),
               cache_hit_alignment_tokens=extra.pop("cache_hit_alignment_tokens",
                                                    128),
               eagle_group_ids=extra.pop("eagle_group_ids", set()),
               **extra)


def kv_cache_manager(pool, coord):
    return Obj(name="KVCacheManager", block_pool=pool, coordinator=coord,
               retained_hit_group_ids=set())


def group(spec, layer_names, **extra):
    return Obj(name="KVCacheGroupSpec", kv_cache_spec=spec,
               layer_names=list(layer_names),
               is_eagle_group=extra.pop("is_eagle_group", False),
               host_resident=extra.pop("host_resident", False),
               enable_kv_transfer=extra.pop("enable_kv_transfer", True),
               role=extra.pop("role", "attention"),
               **extra)


def kv_cache_config(groups, num_blocks=8192, layout="attention", **extra):
    return Obj(name="KVCacheConfig", num_blocks=num_blocks,
               kv_cache_groups=list(groups), kv_cache_layout=layout,
               prefix_cache_retention_interval=extra.pop(
                   "prefix_cache_retention_interval", 0),
               **extra)


def cache_config(**extra):
    return Obj(name="CacheConfig",
               mamba_cache_mode=extra.pop("mamba_cache_mode", "none"),
               prefix_match_unit=extra.pop("prefix_match_unit", 128),
               block_size=extra.pop("block_size", 1024),
               num_gpu_blocks=extra.pop("num_gpu_blocks", 8192),
               num_cpu_blocks=extra.pop("num_cpu_blocks", 4096),
               enable_prefix_caching=extra.pop("enable_prefix_caching", True),
               enable_mamba_fine_grained_prefix_cache=extra.pop(
                   "enable_mamba_fine_grained_prefix_cache", False),
               kv_cache_size_tokens=extra.pop("kv_cache_size_tokens", 8388608),
               **extra)


def tier_manager(group_sizes=(1024, 1024, 1024, 1024, 128, 1024),
                 hash_block_size=128, block_size=1024, num_cpu_blocks=4096,
                 cpu_blocks=None, **extra):
    """`SimpleCPUOffloadScheduler`'s geometry fields (manager.py)."""
    return Obj(name="SimpleCPUOffloadScheduler",
               group_block_sizes=list(group_sizes),
               hash_block_size=hash_block_size, block_size=block_size,
               fa_block_size=extra.pop("fa_block_size", 1024),
               fa_gidx=extra.pop("fa_gidx", 0),
               num_cpu_blocks=num_cpu_blocks,
               cpu_block_pool=Pool(cpu_blocks if cpu_blocks is not None
                                   else [Block(i) for i in range(64)]),
               # The store-alignment choice (manager.py:820-831).
               cpu_coordinator=Obj(name="CPUCoordinator",
                                   enable_partial_hash_hits=extra.pop(
                                       "enable_partial_hash_hits", True)),
               **extra)


class RequestStatus(int):
    """Stands in for the fork's `RequestStatus` IntEnum: an int that has a `.name`.

    The reader asks for BOTH (`status.name` for a human, the int value for the
    wire), so an ordinary int would answer one of the two and a string would fail
    the other.
    """

    def __new__(cls, value, name="RUNNING"):
        obj = super().__new__(cls, value)
        obj._name = name
        return obj

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name


def request(request_id, *, num_tokens=4096, num_prompt_tokens=4000,
            num_output_tokens=96, num_computed_tokens=3968,
            num_output_placeholders=64, block_hashes=32, status_name="RUNNING",
            status_value=1, all_token_ids=4096, preemptions=0,
            skip_reading_prefix_cache=False):
    """`Request` (vllm/v1/core/sched/request.py), counters only."""
    return Obj(name="Request", request_id=request_id,
               status=RequestStatus(status_value, status_name),
               num_tokens=num_tokens, num_prompt_tokens=num_prompt_tokens,
               num_output_tokens=num_output_tokens,
               num_computed_tokens=num_computed_tokens,
               num_output_placeholders=num_output_placeholders,
               num_preemptions=preemptions, all_token_ids=list(range(all_token_ids)),
               block_hashes=[hash_key(i) for i in range(block_hashes)],
               skip_reading_prefix_cache=skip_reading_prefix_cache)


class Scheduler:
    """The fake scheduler: the fork's fields plus the TWO methods an observer wraps.

    `step_trace` wraps `schedule` and `update_from_output`, so the double has to
    own them - and it must return the fork's answer untouched, because the single
    most important property of an observer is that the caller cannot tell it is
    there. Registered in `_NAMED_CLASSES` so `unwrap_everything()` restores it.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def schedule(self, *args, **kwargs):
        return self.__dict__.get("step_output")

    def update_from_output(self, *args, **kwargs):
        return None


_NAMED_CLASSES["Scheduler"] = Scheduler


def scheduler(*, groups=None, managers=None, pool=None, tier=None, requests=None,
              hash_block_size=128, block_size=1024, config=None,
              kv_cache_config_=None, current_step=1234, **extra):
    groups = groups if groups is not None else []
    return Scheduler(
               kv_cache_config=kv_cache_config_ or kv_cache_config(groups),
               cache_config=config if config is not None else cache_config(),
               kv_cache_manager=kv_cache_manager(pool or Pool(),
                                                 coordinator(managers or [])),
               connector=None if tier is None else Obj(
                   name="SimpleCPUOffloadConnector", scheduler_manager=tier),
               requests=dict(requests or {}),
               running=extra.pop("running", []), waiting=extra.pop("waiting", []),
               block_size=block_size, hash_block_size=hash_block_size,
               current_step=current_step, has_mamba_layers=False,
               needs_kv_cache_zeroing=False, **extra)


def engine(scheduler_=None):
    return Obj(name="EngineCore", scheduler=scheduler_)


# --- the production-shaped engine -------------------------------------------

def production_geometry():
    """Six groups, the shape the live box serves: four FA, one Mamba, one ring.

    Returns (engine, info) where `info` names the interesting indices, so a test
    can say "group 4 is the scratch ring" instead of a bare 4.
    """
    fa = [attention_spec() for _ in range(4)]
    mamba = mamba_spec()
    ring = circular_buffer_spec()          # a SUBCLASS: the name lives in the MRO
    specs = fa + [mamba, ring]
    groups = [group(spec, [f"layer_{index}.{layer}" for layer in range(8)])
              for index, spec in enumerate(specs)]
    groups[4].host_resident = True          # a Mamba group may live in host RAM
    groups[5].enable_kv_transfer = False    # the ring is never transferred
    managers = [manager_item(index, spec) for index, spec in enumerate(specs)]
    # 0..39 are referenced (on no free list, still hashed, and the first 20 are
    # in the prefix-cache map); 40..63 are free: unhashed and linked into the LRU
    # free queue, head and tail each missing one link exactly like a real list.
    blocks = [Block(i, block_hash=hash_key(i, i % len(specs)), ref_cnt=1,
                    num_tokens=1024) for i in range(40)]
    blocks += [Block(i) for i in range(40, 64)]
    for index in range(40, 64):
        block = blocks[index]
        block.prev_free_block = blocks[index - 1] if index > 40 else None
        block.next_free_block = blocks[index + 1] if index < 63 else None
    pool = Pool(blocks, num_gpu_blocks=8192, free=8192 - 40,
                cached={hash_key(i, i % len(specs)): blocks[i] for i in range(20)})
    tier = tier_manager()
    sched = scheduler(groups=groups, managers=managers, pool=pool, tier=tier,
                      requests={"cmpl-a": request("cmpl-a")},
                      running=["cmpl-a"], config=cache_config())
    return engine(sched), {"specs": specs, "groups": groups,
                           "managers": managers, "pool": pool, "tier": tier,
                           "scheduler": sched, "ring_index": 5,
                           "mamba_index": 4}


# --- a fork-module class stub ----------------------------------------------

def fork_class(module_name: str, class_name: str, source: str,
               register: bool = False):
    """Exec `source` into a module NAMED `module_name` and return the class.

    `guard_mode`, `worker_diag` and `cpu_capacity` refuse to wrap a function whose
    `__module__` is not the fork's own module, which is exactly the check that
    stops a plugin from overwriting something a third party owns. A class defined
    in this test file would carry `fakes_debug` and be (correctly) refused, so the
    stub is built from a source string exec'd inside a module bearing the fork's
    name - the `tests/test_cpu_capacity.py` idiom.
    """
    module = types.ModuleType(module_name)
    exec(source, module.__dict__)  # noqa: S102 - our own source string, never input
    if register:
        sys.modules[module_name] = module
    return getattr(module, class_name), module


TIER_MANAGER_SOURCE = """
class SimpleCPUOffloadScheduler:
    def __init__(self):
        self.calls = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.calls.append("get_num_new_matched_tokens")
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.calls.append("update_state_after_alloc")
        return None

    def build_connector_meta(self, scheduler_output):
        self.calls.append("build_connector_meta")
        return SimpleCPUOffloadMetadata()

    def prepare_store_specs(self, specs):
        self.calls.append("prepare_store_specs")
        return [1], [2], [3], None

    def request_finished(self, request, block_ids):
        self.calls.append("request_finished")
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        self.calls.append("request_finished_all_groups")
        return False, None

    def take_events(self):
        self.calls.append("take_events")
        return ()

    def reset(self):
        self.calls.append("reset")
        return True

    def update_connector_output(self, connector_output):
        self.calls.append("update_connector_output")


class SimpleCPUOffloadMetadata:
    def __init__(self):
        self.load_event = -1
        self.store_event = -1
        self.load_gpu_blocks = []
        self.store_gpu_blocks = []
"""

def unwrap_everything() -> int:
    """Put back every method this package ever wrapped on a fake class.

    `_named_class` hands out ONE class per name, so a wrapper installed by one
    test is still installed when the next one runs - the exact leak that would
    make a later test assert on a nested wrapper. A fixture calls this on teardown.
    """
    from kvtier_dev_plugin import (  # noqa: PLC0415 - unwrap, do not import at top
        blockdump,
        guard_mode,
        step_trace,
        worker_diag,
    )
    restored = 0
    for cls in _NAMED_CLASSES.values():
        # `blockdump` FIRST: it composes on TOP of `worker_diag`'s layer on the
        # same method (`get_finished`), so removing the outer layer first is what
        # puts the writer's layer back on the class for the writer's own unwrap to
        # find. The reverse order would leave a wrapped method behind.
        for module in (blockdump, guard_mode, step_trace, worker_diag):
            for name in getattr(module, "WRAPPED_METHODS", ()):
                current = cls.__dict__.get(name)
                if getattr(current, module.OWNER_ATTR, None) == module.OWNER_TAG:
                    setattr(cls, name, getattr(current, module.ORIGINAL_ATTR))
                    restored += 1
    return restored


WORKER_SOURCE = """
class SimpleCPUOffloadWorker:
    def __init__(self):
        self.cpu_kv_caches = {"g0": _Cache(4096)}
        self.num_cpu_blocks = 4
        self._connector_metadata = None
        self._pending_store_event_indices = []
        self._pending_load_event_indices = []
        self._completed_store_events = {}
        self._store_hwm = 0
        self._load_hwm = 0
        self.cpu_capacity_bytes = 16384
        self.kv_cache_config = None
        self.get_finished_calls = 0
        self.bind_calls = 0

    def bind_connector_metadata(self, metadata):
        self.bind_calls += 1
        self._connector_metadata = metadata

    def get_finished(self, finished_req_ids):
        self.get_finished_calls += 1
        return set(), None


class _Cache:
    def __init__(self, nbytes):
        self.nbytes = nbytes
"""


# The same tier entry points, with a switch: `module.FAILING.add("reset")` makes
# the FORK's own function raise. That is the only honest way to exercise a guard -
# the wrapper closes over the original function, so replacing the attribute
# afterwards would test a path that cannot happen.
_GUARD_METHODS = ("get_num_new_matched_tokens", "update_state_after_alloc",
                  "build_connector_meta", "prepare_store_specs",
                  "request_finished", "request_finished_all_groups", "take_events",
                  "reset", "update_connector_output")

_PREAMBLE = """FAILING = set()


def _boom(name):
    if name in FAILING:
        raise ZeroDivisionError("the tier divided by zero in " + name)


"""


def _with_failure_injection(source: str, methods) -> str:
    """Put `_boom("<method>")` first in each listed method body."""
    out = []
    for line in source.splitlines():
        out.append(line)
        stripped = line.strip()
        if stripped.startswith("def "):
            name = stripped[4:].split("(", 1)[0].strip()
            if name in methods:
                out.append(f"        _boom({name!r})")
    return _PREAMBLE + "\n".join(out) + "\n"


GUARD_TIER_SOURCE = _with_failure_injection(TIER_MANAGER_SOURCE, _GUARD_METHODS)
