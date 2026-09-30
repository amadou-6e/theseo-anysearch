"""The provider SDK stays importable without the AnySearch runtime stack."""

import subprocess
import sys


def test_top_level_sdk_import_does_not_load_host_or_ray():
    script = """
import sys
import theseo_world_providers
from theseo_world_providers import bundle, fixtures, service
assert 'theseo_anysearch' not in sys.modules
assert 'ray' not in sys.modules
assert fixtures.FixtureBoxesProvider.info.name == 'fixture-boxes'
assert isinstance(service.remote_catalog(), list)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_legacy_imports_share_sdk_types_and_entry_point_group():
    from theseo_anysearch.environments.routing_manifests import RoutingWorldRecord as LegacyWorld
    from theseo_anysearch.world_providers.api import ENTRY_POINT_GROUP as legacy_group
    from theseo_anysearch.world_providers.api import ProviderInfo as LegacyInfo
    from theseo_world_providers.api import ENTRY_POINT_GROUP, ProviderInfo
    from theseo_world_providers.routing_manifests import RoutingWorldRecord

    assert LegacyInfo is ProviderInfo
    assert LegacyWorld is RoutingWorldRecord
    assert legacy_group == ENTRY_POINT_GROUP == "theseo_anysearch.world_providers"


def test_catalog_paths_remain_a_host_only_contract():
    import pytest

    from theseo_world_providers.world_manifest import world_contract

    with pytest.raises(ValueError, match="AnySearch host adapter"):
        world_contract({"compiled_world_catalog_path": "catalog.json"})
