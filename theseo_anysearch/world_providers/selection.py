"""Attach one verified world to a runnable, checked experiment configuration."""

from __future__ import annotations

import json
import os
import tempfile
from difflib import unified_diff
from io import StringIO
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML

from theseo_anysearch.environments.routing_manifests import storage_to_task
from theseo_anysearch.world_providers.models import WorldSelectionConfig
from theseo_anysearch.world_providers.service import load_verified_world
from theseo_anysearch.worlds.compiler import (
    NpySource,
    WorldCompilerConfig,
    compile_world,
    compiler_identity,
    validate_compiled_world,
)


def _relative(target: Path, base: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def _yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML must be a mapping")
    return raw


def _check_study_assignments(selection: WorldSelectionConfig, config_path: Path) -> None:
    """Read actual YAML files in the declared study, never a registry cache."""

    study = (config_path.resolve().parent / selection.study_root).resolve(strict=True)
    if not config_path.resolve().is_relative_to(study):
        raise ValueError("worlds.study_root must contain the experiment YAML")
    seen = 0
    for candidate in sorted((*study.rglob("*.yaml"), *study.rglob("*.yml"))):
        if ".anysearch" in candidate.parts or candidate.resolve() == config_path.resolve():
            continue
        seen += 1
        if seen > 2000:
            raise ValueError("study contains too many YAML files to check split assignments")
        data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        other = data.get("worlds")
        if not isinstance(other, dict) or "root_geometry_id" not in other:
            continue
        other_root = (candidate.parent / Path(other.get("study_root", "."))).resolve()
        if other_root != study or other.get("study_id") != selection.study_id:
            continue
        if other.get("split_identity_sha256") != selection.split_identity_sha256:
            raise ValueError(
                f"same-study YAMLs must reference one split record: {candidate}; "
                "multi-world study updates are not supported yet"
            )
        if (
            other["root_geometry_id"] == selection.root_geometry_id
            and other.get("role") != selection.role
        ):
            raise ValueError(
                f"root geometry crosses split roles within the study: {candidate}"
            )


def validate_selection(selection: WorldSelectionConfig, config_path: Path, geometry: Any, waypoints_file: str | None) -> None:
    """Recheck actual world, split, pack, and task on every experiment load."""

    base = config_path.resolve().parent
    _check_study_assignments(selection, config_path)
    root = (base / selection.root).resolve(strict=True)
    bundle = load_verified_world(root, use="training" if selection.role == "train" else "evaluation")
    if (
        bundle.world.identity_sha256 != selection.world_identity_sha256
        or bundle.world.root_geometry_id != selection.root_geometry_id
        or bundle.dataset_identity_sha256 != selection.dataset_identity_sha256
        or bundle.split.identity_sha256 != selection.split_identity_sha256
    ):
        raise ValueError("selected world or split manifest changed")
    if len(bundle.split.members) != 1 or bundle.split.members[0].partition != selection.role:
        raise ValueError("selected world is assigned to another split role")
    task = next((item for item in bundle.tasks if item.identity_sha256 == selection.task_identity_sha256), None)
    if task is None:
        raise ValueError("selected task is absent from verified world")
    compiled_path = geometry.compiled_world_path
    if compiled_path is None:
        raise ValueError("selected world has no compiled runtime pack")
    compiled_path = (base / compiled_path).resolve(strict=True)
    compiled = validate_compiled_world(compiled_path)
    expected_pack, _ = compiler_identity(
        (NpySource(root / "occupancy.npy"),), bundle.world.extent, WorldCompilerConfig()
    )
    if compiled.manifest.identity_sha256 != expected_pack or compiled.manifest.extent != bundle.world.extent:
        raise ValueError("compiled runtime pack is not the selected voxel world")
    if geometry.extent != bundle.world.extent.as_tuple():
        raise ValueError("configured extent differs from selected voxel world")
    if waypoints_file is None:
        raise ValueError("selected routing task has no waypoint file")
    waypoint_path = (base / waypoints_file).resolve(strict=True)
    waypoints = json.loads(waypoint_path.read_text(encoding="utf-8"))
    if waypoints != {
        "start": list(storage_to_task(task.start_storage, bundle.world.extent)),
        "goal": list(storage_to_task(task.goal_storage, bundle.world.extent)),
    }:
        raise ValueError("runtime waypoints disagree with selected verified task")


def add_to_experiment(world: Path, config_path: Path) -> dict[str, str]:
    """Select a verified world; do not start training or silently replace geometry."""

    config_path = config_path.resolve(strict=True)
    original = config_path.read_text(encoding="utf-8")
    _yaml(config_path)
    editor = YAML(typ="rt")
    editor.preserve_quotes = True
    editor.allow_duplicate_keys = False
    raw = editor.load(original)
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML must be a mapping")
    if "sweep" in raw:
        raise ValueError("sweep YAML selection is not supported; use a training/tune_config YAML")
    env = raw.get("env")
    training = raw.get("training")
    if not isinstance(env, dict) or not isinstance(training, dict):
        raise ValueError("worlds add requires a training or tuning experiment YAML")
    existing = raw.get("worlds")
    if existing is not None and not isinstance(existing, dict):
        raise ValueError("worlds config must be a mapping")
    if existing and existing.get("role") not in {"train", "validation", "test", "calibration"}:
        raise ValueError("worlds.role must identify the target data split")
    role = existing.get("role", "train") if existing else "train"
    if role != "train":
        raise ValueError("runtime selection currently supports training worlds only")
    bundle = load_verified_world(world, use="training")
    if len(bundle.split.members) != 1 or bundle.split.members[0].partition != role:
        raise ValueError("world split record does not assign it to the requested role")
    task = sorted(bundle.tasks, key=lambda item: item.identity_sha256)[0]
    selection = WorldSelectionConfig(
        role=role,
        study_id=str(existing.get("study_id", config_path.stem) if existing else config_path.stem),
        study_root=Path(existing.get("study_root", ".") if existing else "."),
        root=_relative(bundle.root, config_path.parent),
        world_identity_sha256=bundle.world.identity_sha256,
        root_geometry_id=bundle.world.root_geometry_id,
        dataset_identity_sha256=bundle.dataset_identity_sha256,
        split_identity_sha256=bundle.split.identity_sha256,
        task_identity_sha256=task.identity_sha256,
    )
    if existing and "root" in existing:
        if WorldSelectionConfig.model_validate(existing) == selection:
            from theseo_anysearch.experiments.loader import load_experiment

            load_experiment(config_path)
            return {"status": "unchanged", "world_identity_sha256": bundle.world.identity_sha256}
        raise ValueError("config already selects a different world; use a new experiment YAML")
    _check_study_assignments(selection, config_path)
    geometry = env.setdefault("geometry", {})
    if not isinstance(geometry, dict):
        raise ValueError("env.geometry must be a mapping")
    conflicting = ("stl_path", "stl_paths", "boxes", "pool", "compiled_world_path")
    if any(geometry.get(name) is not None for name in conflicting) or env.get("waypoints_file"):
        raise ValueError("config already specifies another geometry or waypoint source")
    cache = config_path.parent / ".anysearch" / "worldpacks"
    compiled = compile_world((NpySource(bundle.root / "occupancy.npy"),), bundle.world.extent, cache)
    waypoint_dir = config_path.parent / ".anysearch" / "world-tasks"
    waypoint_dir.mkdir(parents=True, exist_ok=True)
    waypoint_path = waypoint_dir / f"{task.identity_sha256}.json"
    waypoint_bytes = (json.dumps({
        "start": storage_to_task(task.start_storage, bundle.world.extent),
        "goal": storage_to_task(task.goal_storage, bundle.world.extent),
    }, sort_keys=True) + "\n").encode("utf-8")
    if waypoint_path.exists():
        if waypoint_path.read_bytes() != waypoint_bytes:
            raise ValueError("existing waypoint artifact differs from verified task")
    else:
        waypoint_path.write_bytes(waypoint_bytes)
    raw["worlds"] = selection.model_dump(mode="json")
    grid_comment = getattr(geometry, "ca", None)
    grid_comment = grid_comment.items.get("grid_size") if grid_comment is not None else None
    geometry["extent"] = list(bundle.world.extent.as_tuple())
    if grid_comment is not None:
        geometry.ca.items["extent"] = grid_comment
    geometry.pop("grid_size", None)
    geometry["compiled_world_path"] = _relative(compiled.root, config_path.parent)
    env["waypoints_file"] = _relative(waypoint_path, config_path.parent)
    stream = StringIO()
    editor.dump(raw, stream)
    encoded = stream.getvalue()
    diff = "".join(unified_diff(
        original.splitlines(keepends=True), encoded.splitlines(keepends=True),
        fromfile=str(config_path), tofile=str(config_path),
    ))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix=".worlds-", dir=config_path.parent, delete=False, encoding="utf-8") as stream:
            stream.write(encoded)
            temporary_path = Path(stream.name)
        from theseo_anysearch.experiments.loader import load_experiment

        load_experiment(temporary_path)
        os.replace(temporary_path, config_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "status": "added",
        "world_identity_sha256": bundle.world.identity_sha256,
        "selected_task_identity_sha256": task.identity_sha256,
        "config": str(config_path),
        "diff": diff,
    }
