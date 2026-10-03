"""PR #56810 patch source for `SimpleCPUOffloadScheduler`.
Generated from a `git apply` of the manager.py hunk of /tmp/pr56810.diff
onto a byte-exact scratch copy of the fork's
`vllm/v1/simple_kv_offload/manager.py` (byte-identical to upstream v0.30.0).
Do not hand-edit: regenerate, and change the fingerprints together with the
bodies. `test_backport.py` re-checks every body against the fork at HEAD.
"""

from __future__ import annotations

# original `inspect.getsource()` sha256 at fork HEAD 10b8e07:
# 12c3a9c52f1b0d0ef03e07a0d19a67433829ff65bb3bfd2969d3e2dcbe366925
_SRC___INIT__ = '''    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
        disk_capacity_bytes: int = 0,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # When disk mode is active, the offload pool size is disk-based.
        offload_capacity = (
            disk_capacity_bytes if disk_capacity_bytes > 0 else cpu_capacity_bytes
        )
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.cp_world_size = dcp_world_size
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, offload_capacity
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        self.prefix_cacheable_group_ids = (
            self.cpu_kv_cache_config.prefix_cacheable_group_ids
        )
        self.kv_event_medium = MEDIUM_STORAGE if disk_capacity_bytes > 0 else MEDIUM_CPU
        # Find the full attention kv group for prefix cache matching.
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)
        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d offload blocks "
            "(%.2f GB, mode=%s, backend=%s)",
            self.num_cpu_blocks,
            offload_capacity / (1024**3),
            "lazy" if lazy_offload else "eager",
            "disk" if disk_capacity_bytes > 0 else "cpu",
        )

        spec_config = vllm_config.speculative_config
        use_eagle_block_drop = (
            spec_config is not None and spec_config.use_eagle_block_drop()
        )
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle_block_drop,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            allow_partial_hash_hits=not lazy_offload,
        )
        self.group_block_sizes = self.cpu_coordinator.group_block_sizes
        # FA group's own resolved block_size; divides scheduler_block_size (the
        # LCM) but is NOT assumed to equal it.
        self.fa_block_size: int = self.group_block_sizes[self.fa_gidx]
        assert self.block_size % self.fa_block_size == 0
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length, num_computed_tokens) tuples from
        # find_longest_cache_hit, kept pinned via touch() while awaiting
        # update_state_after_alloc(). ``num_computed_tokens`` is the local
        # prefix the hit was resolved against; the load path needs it to place
        # external blocks and must not re-derive it from block metadata.
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int, int]
        ] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
        else:
            self._target_free = 0
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        self._abandoned_store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        self._in_flight_store_gpu_blocks: set[int] = set()
        self._pending_finished_stores: list[TransferMeta] = []
        self._abandoned_reqs_to_load: dict[str, LoadRequestState] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0
        self.boundary_store_stats = BoundaryStoreStats()

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}
'''

# original `inspect.getsource()` sha256 at fork HEAD 10b8e07:
# dac52a01e42095cbb6c852859c3439307dcf539ac32c8d690e19ae5f2400f735
_SRC__ESTIMATE_LAZY_TARGET_BLOCKS = '''    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        cp_world_size: int = 1,
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.prefix_cacheable_groups:
            spec = g.kv_cache_spec
            # Only full attention is sharded across DCP ranks; replicated specs
            # (mamba, sliding window, chunked-local) keep their own block size.
            block_size = spec.block_size * dcp_world_size_for_kv_cache_spec(
                spec, cp_world_size
            )
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return int(target * (1 + WATERMARK_RATIO))
'''

# original `inspect.getsource()` sha256 at fork HEAD 10b8e07:
# ed45065111c1e287e921162f7f38da6ae7c368100c5dfa0b999b6c68b2a391ec
_SRC_UPDATE_STATE_AFTER_ALLOC = '''    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _, num_computed_tokens = pending

        assert num_computed_tokens % self.block_size == 0, (
            "SimpleCPU external loads must start at a scheduler-block boundary"
        )
        # Fine-grained hits are hash-block aligned, not group-block aligned, so
        # the per-group block count below rounds up instead of dividing exactly.
        # Hash alignment is still required: it is what makes the accepted
        # external suffix start on a boundary every group can address.
        assert num_external_tokens % self.hash_block_size == 0, (
            f"num_external_tokens={num_external_tokens} is not aligned to "
            f"hash_block_size={self.hash_block_size}"
        )

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            if g not in self.prefix_cacheable_group_ids:
                cpu_hit_blocks.append([])
                continue
            g_block_size = self.group_block_sizes[g]
            n_take_g = cdiv(num_external_tokens, g_block_size)
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            g_block_size = self.group_block_sizes[g]
            gpu_ext_start = num_computed_tokens // g_block_size
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request, transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids)
        )
'''

# original `inspect.getsource()` sha256 at fork HEAD 10b8e07:
# 12b3854a8dcfb55330db95141fb4ce910222f67ea0fac1a5899832360c236187
_SRC__SELECT_EAGER_BLOCKS_TO_STORE = '''    def _select_eager_blocks_to_store(
        self,
        state: StoreRequestState,
        block_ids_by_group: tuple[list[int], ...],
    ) -> tuple[
        list[int],
        list[int],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        """Return confirmed eager blocks, cursor advances, and event metadata."""
        assert self._gpu_block_pool is not None
        gpu_block_ids: list[int] = []
        advanced_per_group = [0] * len(self.cpu_kv_cache_config.kv_cache_groups)
        block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = (
            [] if self.enable_kv_cache_events else None
        )
        request = state.request
        confirmed_tokens = request.num_computed_tokens - request.num_output_placeholders
        # Truncate to the granularity a lookup can actually land on. With
        # fine-grained hits that is hash_block_size; otherwise hits only land on
        # the scheduler block (the group LCM). Using the LCM unconditionally
        # would drop whole blocks that sit between the last LCM boundary and a
        # reachable fine-grained boundary, so the other groups would hold that
        # boundary and this one would not, and the joint hybrid lookup would
        # reconcile to zero.
        store_alignment = (
            self.hash_block_size
            if self.cpu_coordinator.enable_partial_hash_hits
            else self.block_size
        )
        aligned_tokens = confirmed_tokens // store_alignment * store_alignment
        num_free = self.cpu_block_pool.get_num_free_blocks()

        for g, group_gpu_ids in enumerate(block_ids_by_group):
            if g not in self.prefix_cacheable_group_ids:
                continue
            if len(gpu_block_ids) >= num_free:
                break
            group_manager = self.cpu_coordinator.single_type_managers[g]
            if not group_manager.has_positionally_stable_blocks:
                continue
            # FIXME (yifan): handle CPU cache eviction, where
            # num_stored_blocks can be stale and omit evicted blocks in
            # the middle of the request.
            group_size = self.group_block_sizes[g]
            ready = min(len(group_gpu_ids), aligned_tokens // group_size)
            resolved_hashes = resolve_block_hashes(
                request.block_hashes, self.hash_block_size, group_size
            )
            curr_mm_idx = 0
            secondary_mm_idx = 0
            start = state.num_stored_blocks[g]
            for i, gpu_block_id in enumerate(group_gpu_ids[start:ready], start=start):
                gpu_block = self._gpu_block_pool.blocks[gpu_block_id]
                if gpu_block.is_null:
                    # Sliding-window groups null pages that left the window
                    # before the connector sees the block table, but the
                    # retained prefix-cache tail stays hashed in the GPU free
                    # queue; store it from there.
                    recovered = self._cached_gpu_block(resolved_hashes, i, g)
                    if recovered is None:
                        advanced_per_group[g] += 1
                        continue
                    gpu_block = recovered
                    gpu_block_id = gpu_block.block_id
                if (
                    self._classify_store_candidate(gpu_block)
                    is not _StoreAdmission.READY
                ):
                    advanced_per_group[g] += 1
                    continue
                if len(gpu_block_ids) >= num_free:
                    break
                primary_block_hash = gpu_block.block_hash
                assert primary_block_hash is not None
                gpu_block_ids.append(gpu_block_id)
                advanced_per_group[g] += 1
                if block_meta is not None:
                    token_start = i * group_size
                    token_end = token_start + group_size
                    parent_hash = (
                        None
                        if i == 0
                        else maybe_convert_block_hash(resolved_hashes[i - 1])
                    )
                    lora_req = request.lora_request
                    extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                        request, token_start, token_end, curr_mm_idx
                    )
                    meta_by_hash = {
                        primary_block_hash: BlockStoreMeta(
                            token_ids=list(
                                request.all_token_ids[token_start:token_end]
                            ),
                            parent_block_hash=parent_hash,
                            lora_id=lora_req.adapter_id if lora_req else None,
                            lora_name=lora_req.name if lora_req else None,
                            extra_keys=extra_keys,
                        )
                    }
                    first_hash_idx = token_start // self.hash_block_size
                    last_hash_idx = token_end // self.hash_block_size
                    for hash_idx in range(first_hash_idx, last_hash_idx):
                        block_hash = make_block_hash_with_group_id(
                            request.block_hashes[hash_idx], g
                        )
                        if block_hash is None or block_hash == primary_block_hash:
                            continue
                        hash_start = hash_idx * self.hash_block_size
                        hash_end = hash_start + self.hash_block_size
                        secondary_extra_keys, secondary_mm_idx = (
                            generate_block_hash_extra_keys(
                                request,
                                hash_start,
                                hash_end,
                                secondary_mm_idx,
                            )
                        )
                        meta_by_hash[block_hash] = BlockStoreMeta(
                            token_ids=list(request.all_token_ids[hash_start:hash_end]),
                            parent_block_hash=(
                                None
                                if hash_idx == 0
                                else maybe_convert_block_hash(
                                    request.block_hashes[hash_idx - 1]
                                )
                            ),
                            lora_id=lora_req.adapter_id if lora_req else None,
                            lora_name=lora_req.name if lora_req else None,
                            extra_keys=secondary_extra_keys,
                            block_size=self.hash_block_size,
                        )
                    block_meta.append(meta_by_hash)

        return gpu_block_ids, advanced_per_group, block_meta
'''

# --- the POST-patch fingerprints: what "the fix is already in the tree" means --
#
# `orig_sha256` above answers "is this the v0.30.0 body we know how to replace?".
# These answer the other half: "is this ALREADY the upstream #56810 body?". That
# question is what lets the runtime backport stand down on an engine whose own
# source tree carries the fix (the tracked `config/box/kvtier-pr56810/` patch in
# the llm-api repo), instead of wrapping a fixed body and reporting a fork change
# that never happened.
#
# Pinned as literals on purpose: each value is `sha256(SOURCES[name]["source"])`,
# so `test_pinned_post_patch_fingerprints_match_the_stored_bodies` fails if a
# body is ever edited without re-pinning, exactly like `orig_sha256` does.
PATCHED_SHA256 = {
    '__init__': '70cea026db307075bf736554c5b6eace54017fc977898093579568d89fa8c21a',
    '_estimate_lazy_target_blocks':
        '21e73419defaae87ec69d79e820ba3c61843d6dcdccec070c95042d00d16d9b2',
    'update_state_after_alloc':
        'f7543190de8e17e8ebe159a2fcc94ef04dcb4ade9dd64f7e146f7c1321eb4a51',
    '_select_eager_blocks_to_store':
        'd39d9df7f72e0dbff2508eed458843b99c6028b3a1c4b2cc09eebb1ab20c68f3',
}

# Whole-file fingerprints of the tracked patch, for the operator-facing line
# (the tool in the llm-api repo is what applies them; these are for reporting).
MANAGER_FILE_SHA256_PRE = (
    '3cc841ff97e4ec237800d4f81dcb5dfb5191c639f275a32892915e8b1af43eb7')
MANAGER_FILE_SHA256_POST = (
    'e4b00291991cfd6fdcabb8a22379b119073fca63341d2f488954b280b4a996eb')

# Where the ORIGINAL bodies above came from, so a fingerprint can be traced to a
# commit rather than to a claim.
FORK_PIN_HEAD = '10b8e073321f22113da977856b5d5593ab6fa5d5'
UPSTREAM_ISSUE = 'vllm-project/vllm#56396'   # the crash on the first store request
UPSTREAM_PR = 'vllm-project/vllm#56810'      # the fix, +9/-1 in manager.py

SOURCES = {
    '__init__': {
        "source": _SRC___INIT__,
        "orig_sha256": '12c3a9c52f1b0d0ef03e07a0d19a67433829ff65bb3bfd2969d3e2dcbe366925',
        "is_staticmethod": False,
    },
    '_estimate_lazy_target_blocks': {
        "source": _SRC__ESTIMATE_LAZY_TARGET_BLOCKS,
        "orig_sha256": 'dac52a01e42095cbb6c852859c3439307dcf539ac32c8d690e19ae5f2400f735',
        "is_staticmethod": True,
    },
    'update_state_after_alloc': {
        "source": _SRC_UPDATE_STATE_AFTER_ALLOC,
        "orig_sha256": 'ed45065111c1e287e921162f7f38da6ae7c368100c5dfa0b999b6c68b2a391ec',
        "is_staticmethod": False,
    },
    '_select_eager_blocks_to_store': {
        "source": _SRC__SELECT_EAGER_BLOCKS_TO_STORE,
        "orig_sha256": '12b3854a8dcfb55330db95141fb4ce910222f67ea0fac1a5899832360c236187',
        "is_staticmethod": False,
    },
}
