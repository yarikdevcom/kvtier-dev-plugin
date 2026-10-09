"""Fork-authoritative `MambaManager.reachable_block_mask` text (generated, do not edit).

Captured by tools/gen_mamba_fixture.py from the live-tree reconstruction
`/home/user/Source/experiment/vllm-live-repro`, branch `live` (= upstream v0.30.0 + the 58-patch
series + our four kvtier hunks, byte-identical to what the engine imports).
That file's md5 is
    52a3c5ad0fc4e6911c69bb554b2d914a
which is ALSO its md5 on branch `built` and in `/home/user/Source/experiment/vllm-dev`, so the
fingerprint below is not a property of one checkout. The method is
    sha256 37869f1c89bdd4eae3136ce7fa3e57998ca03696ee4be790dda2a8fe4cf8627d
decorators included, exactly as `inspect.getsource()` yields it (validated
against the #56810 fingerprints by `test_the_extraction_method_reproduces_the_
inspect_fingerprints`).

This is the UNPATCHED text: `build_fork_mamba_class()` rebuilds a stand-in
`MambaManager` so the plugin suite can drive the REAL install path - the
fingerprints match, so `mamba_anchor.install()` accepts it - without importing
vLLM.
"""

from __future__ import annotations

import linecache

FORK_REF = 'live'
FORK_FILE = 'vllm/v1/core/single_type_kv_cache_manager.py'
FORK_FILE_MD5 = '52a3c5ad0fc4e6911c69bb554b2d914a'
FORK_FILE_SHA256 = '3ce57d34a3ca4412173e896fec89252ca0f0695e1c94537e69873e660d28eb79'
CLASS_NAME = 'MambaManager'
METHOD_NAME = 'reachable_block_mask'
METHOD_DECORATORS = ('classmethod',)
METHOD_SHA256 = '37869f1c89bdd4eae3136ce7fa3e57998ca03696ee4be790dda2a8fe4cf8627d'

_ORIG_REACHABLE_BLOCK_MASK = '    @classmethod\n    def reachable_block_mask(\n        cls,\n        start_block: int,\n        end_block: int,\n        alignment_tokens: int | None,\n        kv_cache_spec: KVCacheSpec,\n        use_eagle: bool,\n        retention_interval: int | None = None,\n        reachable_boundaries: Sequence[int] = (),\n        dcp_world_size: int = 1,\n        final_segment_end_block: int | None = None,\n    ) -> list[bool] | None:\n        """Sparse Mamba state-snapshot retention.\n\n        ``retention_interval``:\n\n          ``None`` -> dense (cache every block; default, unchanged behavior)\n          ``0``    -> keep only the ``reachable_boundaries`` states\n          ``> 0``  -> keep one state per ``retention_interval``-sized segment\n\n        ``reachable_boundaries`` are proven reuse points (the replay boundary and\n        any cross-request shared-prefix junction, Marconi-style APC); their\n        boundary state is always kept so sparse retention does not defeat reuse.\n        """\n        if retention_interval is None or alignment_tokens is None:\n            # Dense caching (default) or no alignment constraint imposed.\n            return None\n        assert isinstance(kv_cache_spec, MambaSpec)\n        block_size = kv_cache_spec.block_size\n        mask = [False] * (end_block - start_block)\n\n        # (1) Segment-boundary states. A Mamba hit needs exactly the single\n        # state block ending on the boundary (no window, and draft models have\n        # no mamba layers, so no eagle shift). Block ``i`` ends at token\n        # ``(i + 1) * block_size``.\n        segment_tokens = None if retention_interval == 0 else retention_interval\n        if segment_tokens is not None:\n            per_segment = segment_tokens // block_size\n            if per_segment <= 1:\n                # Interval at/below the block size: every block is a boundary.\n                return None\n            first_boundary = (\n                start_block + per_segment\n            ) // per_segment * per_segment - 1\n            for i in range(first_boundary - start_block, len(mask), per_segment):\n                mask[i] = True\n\n        # (2) Reachable-boundary states: the replay boundary (``num_prompt - 1``,\n        # capped by ``get_computed_blocks``) and any shared-prefix junction, both\n        # of which segments would otherwise skip under sparse retention. A Mamba\n        # hit needs exactly the single state block ending on the boundary.\n        for boundary_tokens in reachable_boundaries:\n            aligned = boundary_tokens // alignment_tokens * alignment_tokens\n            boundary_block = aligned // block_size - 1\n            if start_block <= boundary_block < end_block:\n                mask[boundary_block - start_block] = True\n\n        return mask\n'

ORIGINAL_SOURCES = {
    "reachable_block_mask": _ORIG_REACHABLE_BLOCK_MASK,
}

REBUILT_SOURCE_FILE = '<fixtures_mamba_fork>'


def class_text() -> str:
    """The stand-in class source, assembled from the fork's own text."""
    return ("class MambaManager(SingleTypeKVCacheManagerStub):\n"
            + "".join(ORIGINAL_SOURCES[name] if name.endswith("\n")
                    else ORIGINAL_SOURCES[name] + "\n"
                    for name in (METHOD_NAME,)))


def build_fork_mamba_class(ns=None):
    """Rebuild a stand-in `MambaManager` from the fork's own text.

    `ns` supplies the module-level names the method closes over (`MambaSpec`,
    `Sequence`, `KVCacheSpec`). Like `fixtures_pr56810`, the source is registered
    in `linecache` under `REBUILT_SOURCE_FILE` so `inspect.getsource()` -
    which is what the plugin's fingerprint check calls - works on the rebuilt
    method. Returns (cls, module_namespace).
    """
    src = class_text()
    linecache.cache[REBUILT_SOURCE_FILE] = (
        len(src), None, src.splitlines(True), REBUILT_SOURCE_FILE)
    module = {"SingleTypeKVCacheManagerStub": object}
    module.update(ns or {})
    module["__name__"] = "fixtures_mamba_fork"
    module["__file__"] = REBUILT_SOURCE_FILE
    exec(compile(src, REBUILT_SOURCE_FILE, "exec"), module, module)
    cls = module["MambaManager"]
    cls.__module__ = "fixtures_mamba_fork"
    return cls, module
