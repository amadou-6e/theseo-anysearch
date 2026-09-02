"""Content-addressed accepted geometry-task and evaluation-suite manifests."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.environments.validation import (
    GeometryValidationResult,
    TaskFeasibilityResult,
)

IDENTITY_SCHEMA_VERSION = 1
VALIDATOR_VERSION = "geometry-task-v1"


def _identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_identity(payload: dict[str, Any]) -> str:
    """Return the canonical SHA-256 identity of one JSON-compatible contract."""

    return _identity(payload)


def geometry_content_identity(coordinates: Iterable[tuple[int, int, int]]) -> str:
    """Hash sorted resolved occupancy, independent of source path and iteration order."""

    return _identity({"coordinates": sorted(tuple(item) for item in coordinates)})


def _file_content_identity(path: Any) -> str | None:
    candidate = Path(str(path)) if path else None
    if candidate is None or not candidate.is_file():
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def configured_geometry_identity(env_config: dict[str, Any]) -> str:
    """Hash configured geometry content without machine-local path names."""

    pool = env_config.get("geometry_pool") or {}
    pool_dir_value = pool.get("pool_dir") if isinstance(pool, dict) else None
    pool_hashes: list[str] = []
    if pool_dir_value and Path(str(pool_dir_value)).is_dir():
        pool_hashes = sorted(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(str(pool_dir_value)).rglob("*.npy")
        )
    return _identity(
        {
            "compiled_world": env_config.get("world_identity_sha256"),
            "stl": _file_content_identity(env_config.get("stl_path")),
            "stls": sorted(
                identity
                for identity in (
                    _file_content_identity(path)
                    for path in (env_config.get("stl_paths") or [])
                )
                if identity is not None
            ),
            "pool": pool_hashes,
            "boxes": env_config.get("geometry_boxes") or [],
        }
    )


def configured_task_contract(env_config: dict[str, Any]) -> dict[str, Any]:
    """Return path-independent semantics used by dataset/checkpoint caches."""

    waypoints = env_config.get("waypoints")
    waypoint_path = env_config.get("waypoints_file")
    if waypoint_path and Path(str(waypoint_path)).is_file():
        waypoints = json.loads(Path(str(waypoint_path)).read_text(encoding="utf-8"))
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "base_geometry_identity": configured_geometry_identity(env_config),
        "geometry_validation": env_config.get("geometry_validation"),
        "pool_transformations": (
            (env_config.get("geometry_pool") or {}).get("augmentation")
            if isinstance(env_config.get("geometry_pool"), dict)
            else None
        ),
        "waypoints": waypoints,
        "waypoint_route": env_config.get("waypoint_route"),
        "waypoint_curriculum": env_config.get("waypoint_curriculum"),
        "task": env_config.get("task"),
        "action_mode": env_config.get("action_mode", "discrete_26"),
    }


class AcceptedTaskManifest(BaseModel):
    """Portable identity and validation evidence for one accepted task."""

    model_config = ConfigDict(frozen=True)
    schema_version: int = IDENTITY_SCHEMA_VERSION
    identity_sha256: str
    geometry_identity_sha256: str
    seed: int
    start: tuple[int, int, int]
    route: tuple[tuple[int, int, int], ...]
    action_mode: str
    transformations: dict[str, Any]
    validator_version: str = VALIDATOR_VERSION
    planner_settings: dict[str, Any]
    geometry_validation: GeometryValidationResult
    task_feasibility: TaskFeasibilityResult


def accepted_task_manifest(
    *,
    coordinates: Iterable[tuple[int, int, int]],
    geometry_identity_sha256: str | None = None,
    seed: int,
    start: tuple[int, int, int],
    route: Iterable[tuple[int, int, int]],
    action_mode: str,
    transformations: dict[str, Any],
    planner_settings: dict[str, Any],
    geometry_validation: GeometryValidationResult,
    task_feasibility: TaskFeasibilityResult,
) -> AcceptedTaskManifest:
    geometry_identity = geometry_identity_sha256 or geometry_content_identity(coordinates)
    contract = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "geometry_identity_sha256": geometry_identity,
        "seed": int(seed),
        "start": tuple(start),
        "route": tuple(tuple(item) for item in route),
        "action_mode": action_mode,
        "transformations": transformations,
        "validator_version": VALIDATOR_VERSION,
        "planner_settings": planner_settings,
    }
    return AcceptedTaskManifest(
        **contract,
        identity_sha256=_identity(contract),
        geometry_validation=geometry_validation,
        task_feasibility=task_feasibility,
    )


class EvaluationSuiteManifest(BaseModel):
    """Fixed ordered suite membership and its difficulty distribution."""

    model_config = ConfigDict(frozen=True)
    schema_version: int = IDENTITY_SCHEMA_VERSION
    identity_sha256: str
    members: tuple[AcceptedTaskManifest, ...]
    difficulty_distribution: dict[str, int]


def build_evaluation_suite(
    members: Iterable[AcceptedTaskManifest],
) -> EvaluationSuiteManifest:
    ordered = tuple(members)
    distribution = dict(
        sorted(Counter(item.task_feasibility.difficulty_band or "unbanded" for item in ordered).items())
    )
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "members": [item.identity_sha256 for item in ordered],
        "difficulty_distribution": distribution,
    }
    return EvaluationSuiteManifest(
        identity_sha256=_identity(payload),
        members=ordered,
        difficulty_distribution=distribution,
    )


def publish_or_load_evaluation_suite(
    path: Path, expected: EvaluationSuiteManifest
) -> EvaluationSuiteManifest:
    """Persist once, or fail if an existing suite has changed semantics."""

    if path.exists():
        stored = EvaluationSuiteManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if stored.identity_sha256 != expected.identity_sha256:
            raise ValueError(
                "evaluation suite identity mismatch: expected "
                f"{expected.identity_sha256}, found {stored.identity_sha256}"
            )
        return stored
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(expected.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return expected


__all__ = [
    "AcceptedTaskManifest",
    "EvaluationSuiteManifest",
    "accepted_task_manifest",
    "build_evaluation_suite",
    "configured_task_contract",
    "configured_geometry_identity",
    "contract_identity",
    "geometry_content_identity",
    "publish_or_load_evaluation_suite",
]
