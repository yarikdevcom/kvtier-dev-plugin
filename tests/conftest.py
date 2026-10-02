"""Shared fixtures. The plugin is importable without vllm installed, so nothing
here stubs an engine: where a test needs an engine it builds a plain object with
the real attribute names (see test_snapshot.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvtier_dev_plugin import config, snapshot  # noqa: E402

ENV_VARS = (config.ENV_ENABLE, config.ENV_TOKEN, config.ENV_TIMEOUT,
            config.ENV_ALLOW)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with the plugin inert and no token."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(config.ENV_ENABLE, "1")


@pytest.fixture
def reset_snapshot_state():
    """Containment counters are process-wide; keep tests from stacking on them."""
    saved = dict(snapshot._state)
    snapshot._state.update({"calls": 0, "errors": 0, "logged": 0, "epoch": None})
    yield snapshot._state
    snapshot._state.update(saved)
