"""AnySearch world manifest adapter over portable extent and identity contracts."""

from typing import Any

from theseo_world_providers.world_manifest import *  # noqa: F401,F403


def world_contract(env_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve host-specific seeded catalogs before building the shared contract."""
    raw_extent = env_config.get("extent")
    scalar_size = env_config.get("grid_size")
    if raw_extent is not None and scalar_size is not None:
        explicit = WorldExtent.from_value(raw_extent)
        shorthand = WorldExtent.from_value(int(scalar_size))
        if explicit != shorthand:
            raise ValueError("grid_size and extent describe different world bounds")
    if raw_extent is None:
        raw_extent = int(scalar_size if scalar_size is not None else 32)
    extent = WorldExtent.from_value(raw_extent)
    raw_origin = env_config.get("source_origin", (0, 0, 0))
    if not isinstance(raw_origin, (tuple, list)) or len(raw_origin) != 3:
        raise ValueError("source_origin must contain exactly three axes")
    contract = {
        "schema_version": WORLD_SCHEMA_VERSION,
        "coordinate_type": COORDINATE_TYPE,
        "storage_coordinate_convention": STORAGE_COORDINATE_CONVENTION,
        "environment_coordinate_convention": ENVIRONMENT_COORDINATE_CONVENTION,
        "environment_min": [1, 1, 1],
        "source_origin": [int(value) for value in raw_origin],
        "extent": list(extent.as_tuple()),
        "identity_sha256": env_config.get("world_identity_sha256"),
    }
    if env_config.get("compiled_world_catalog_path"):
        from theseo_anysearch.worlds.seeded_catalog import load_catalog

        contract["catalog_identity_sha256"] = load_catalog(
            env_config["compiled_world_catalog_path"]
        ).identity_sha256
    return contract
