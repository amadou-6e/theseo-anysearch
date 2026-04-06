"""
Unit test configuration.

Patches the global registry file to a per-test temp path so tests are isolated
from each other and from the user's real ~/.anysearch/registry.yaml.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Redirect the global registry to a per-test temp path via env var."""
    tmp_registry = tmp_path / ".anysearch" / "registry.yaml"
    tmp_registry.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ANYSEARCH_REGISTRY", str(tmp_registry))
    return tmp_registry
