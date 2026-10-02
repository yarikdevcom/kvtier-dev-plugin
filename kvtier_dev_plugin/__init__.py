"""kvtier-dev-plugin: a development window onto a live vLLM KV-cache tier.

Two entry points, one name (`kvtier_dev`), so a single `VLLM_PLUGINS=kvtier_dev`
allowlists both halves:

  vllm.general_plugins  -> kvtier_dev_plugin.engine_side:register
      Attaches two UTILITY methods to `EngineCore` in the engine process:
      `kvtier_dev_snapshot` (read-only scheduler-side tier view) and
      `kvtier_dev_reload_policy` (re-read <package dir>/policy.py).

  vllm.endpoint_plugins -> kvtier_dev_plugin.endpoint:KVTierDevEndpointPlugin
      Adds /plugins/kvtier/{status,snapshot,policy/reload} to the API server and
      reaches the methods above with one `call_utility_async` hop.

Importing this package is side-effect-free and imports nothing heavy: no vllm,
no torch, and (in the read side) not even fastapi. Everything is inert until
VLLM_E2_TIER_DEV is set - see config.py for the env vars, README.md for the
security model and the install/rollback procedure.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
