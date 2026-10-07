"""kvtier-dev-plugin: a development window onto a live vLLM KV-cache tier.

Two entry points, one name (`kvtier_dev`), so a single `VLLM_PLUGINS=kvtier_dev`
allowlists both halves:

  vllm.general_plugins  -> kvtier_dev_plugin.engine_side:register
      Attaches the eight `kvtier_dev_*` UTILITY methods to `EngineCore` in the
      engine process (snapshot, reload_policy, kvconfig, inspect, trace, guard,
      tier, lookups), fits the connector-creation hook that lands the PR #56810
      backport before the first request, and wires the logger.

  vllm.endpoint_plugins -> kvtier_dev_plugin.endpoint:KVTierDevEndpointPlugin
      Adds the /plugins/kvtier/* routes (status, snapshot, kvconfig, block,
      request, trace, guard, tier, lookups, policy/reload) to the API server and
      reaches the methods above with one `call_utility_async` hop.

Importing this package is side-effect-free and imports nothing heavy: no vllm,
no torch, and (in the read side) not even fastapi. Everything is inert until
VLLM_E2_TIER_DEV is set - see config.py for the env vars, README.md for the
security model and the install/rollback procedure.
"""

from __future__ import annotations

__version__ = "0.2.44"

__all__ = ["__version__"]
