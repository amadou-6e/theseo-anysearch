"""Versioned, inspectable recipes for re-executing checkpointed policies."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.experiments.loader import _resolve_typed_configs
from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.experiments.native_extensions import (
    ABI_VERSION,
    NativeExtensionManifest,
)
from theseo_anysearch.worlds.manifest import WorldManifest

CONFIG_SCHEMA_VERSION = 1

_LEGACY_ENV_FIELDS = {
    "stl_path": ("geometry", "stl_path"), "stl_paths": ("geometry", "stl_paths"),
    "scale": ("geometry", "scale"), "scale_range": ("geometry", "scale_range"),
    "grid_size": ("geometry", "grid_size"), "extent": ("geometry", "extent"),
    "geometry_boxes": ("geometry", "boxes"), "obs_mode": ("observation", "mode"),
    "geometry_pool_size": ("geometry", "pool_size"),
    "scale_variants_per_map": ("geometry", "scale_variants_per_map"),
    "geometry_padding": ("geometry", "padding"), "geometry_pool": ("geometry", "pool"),
    "box_radius": ("observation", "box_radius"), "box_radii": ("observation", "box_radii"),
    "ray_max_len": ("observation", "ray_max_len"), "action_mode": ("action", "mode"),
    "step_cost": ("rewards", "step_cost"), "collision_cost": ("rewards", "collision_cost"),
    "goal_reward": ("rewards", "goal_reward"), "distance_shaping": ("rewards", "distance_shaping"),
    "distance_reward_mode": ("rewards", "distance_reward_mode"),
    "zone_reward_min": ("rewards", "zone_reward_min"),
    "zone_reward_max": ("rewards", "zone_reward_max"),
    "zone_reward_curve": ("rewards", "zone_reward_curve"),
    "distance_metric": ("rewards", "distance_metric"),
    "invalid_action_cost": ("rewards", "invalid_action_cost"),
    "construction_residual_weight": ("rewards", "construction_residual_weight"),
    "construction_overshoot_weight": ("rewards", "construction_overshoot_weight"),
}

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        name = item.name if path.is_file() else item.relative_to(path).as_posix()
        data = item.read_bytes()
        digest.update(len(name).to_bytes(8, "little")); digest.update(name.encode())
        digest.update(len(data).to_bytes(8, "little")); digest.update(data)
    return digest.hexdigest()


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    path: str
    sha256: str


class GeometryDecision(BaseModel):
    """Explicit treatment of geometry-dependent configuration after a world swap."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["preserve", "clear", "replace"]
    value: dict[str, Any] | None = None

    @classmethod
    def preserve(cls) -> "GeometryDecision":
        return cls(mode="preserve")


class Overrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world_manifest: str | None = None
    task: GeometryDecision | None = None
    routes: GeometryDecision | None = None
    disabled_capabilities: list[str] = Field(default_factory=list)
    replacements: dict[str, str] = Field(default_factory=dict)


class ConfigMigration(BaseModel):
    """Auditable conversion of an archived YAML into today's typed schema."""

    model_config = ConfigDict(extra="forbid")
    source_schema: Literal["unversioned"] = "unversioned"
    target_schema: Literal[1] = CONFIG_SCHEMA_VERSION
    transforms: list[str] = Field(default_factory=list)
    materialized_defaults: list[str] = Field(default_factory=list)


class ExecutionRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[3] = 3
    scope: Literal["evaluation", "continuation", "fine_tuning"]
    source_run: str
    checkpoint: Artifact
    experiment: Artifact
    checkpoint_state: dict[str, Any]
    policy_contract: dict[str, Any]
    config_migration: ConfigMigration = Field(default_factory=ConfigMigration)
    extension: list[Artifact] = Field(default_factory=list)
    assets: list[Artifact] = Field(default_factory=list)
    extension_bindings: list[str] = Field(default_factory=list)
    overrides: Overrides = Field(default_factory=Overrides)
    provenance_gaps: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExecutionRecipe":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") == 1:
            experiment = Path(raw["experiment"]["path"])
            if not experiment.is_absolute():
                experiment = path.parent / experiment
            config_raw = yaml.safe_load(experiment.read_text(encoding="utf-8"))
            config, migration = _migrate_config(config_raw)
            raw["schema_version"] = 2
            raw["policy_contract"] = _policy_contract(config)
            raw["config_migration"] = migration.model_dump(mode="json")
            raw.setdefault("provenance", {}).setdefault("recipe_migrations", []).append("1 -> 2")
        if raw.get("schema_version") == 2:
            raw["schema_version"] = 3
            raw.setdefault("provenance", {}).setdefault("recipe_migrations", []).append("2 -> 3")
        return cls.model_validate(raw)


def _missing_paths(raw: Any, canonical: Any, prefix: str = "") -> list[str]:
    if not isinstance(canonical, dict):
        return []
    source = raw if isinstance(raw, dict) else {}
    result: list[str] = []
    for key, value in canonical.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in source:
            result.append(path)
        elif isinstance(value, dict):
            result.extend(_missing_paths(source[key], value, path))
    return result


def _migrate_config(raw: dict[str, Any]) -> tuple[ExperimentConfig, ConfigMigration]:
    """Apply named legacy transforms before strict current-schema validation."""
    migrated = deepcopy(raw)
    transforms: list[str] = []
    env = migrated.get("env")
    if not isinstance(env, dict):
        raise ValueError("archived configuration requires an env mapping")
    original_blocks = set(env)
    for old, (block, field) in _LEGACY_ENV_FIELDS.items():
        if old not in env:
            continue
        if block in original_blocks:
            raise ValueError(f"cannot migrate env.{old}: env.{block} already exists")
        env.setdefault(block, {})[field] = env.pop(old)
        transforms.append(f"env.{old} -> env.{block}.{field}")
    rewards = env.get("rewards")
    if isinstance(rewards, dict) and "custom" in rewards:
        if "provider" in rewards:
            raise ValueError("cannot migrate env.rewards.custom: provider already exists")
        rewards["provider"] = rewards.pop("custom")
        transforms.append("env.rewards.custom -> env.rewards.provider")
    config = ExperimentConfig(**_resolve_typed_configs(migrated))
    canonical = config.model_dump(by_alias=True, mode="json")
    return config, ConfigMigration(
        transforms=transforms,
        materialized_defaults=sorted(_missing_paths(migrated, canonical)),
    )


def _policy_contract(config: ExperimentConfig) -> dict[str, Any]:
    predicates, outcomes = config.env.action.resolved_pipeline(trail_mode=config.env.trail_mode)
    algorithm = config.training.algorithm.lower()
    return {
        "algorithm": algorithm,
        "model": config.training.model,
        "algorithm_config": config.algorithm_config.model_dump(mode="json"),
        "model_config": config.model_cfg.model_dump(mode="json"),
        "observation": config.env.observation.model_dump(mode="json"),
        "action": config.env.action.model_dump(mode="json"),
        "resolved_action_pipeline": {
            "predicates": [item.model_dump(mode="json") for item in predicates],
            "outcomes": [item.model_dump(mode="json") for item in outcomes],
        },
        "connectors": {
            "api_stack": "legacy" if algorithm == "appo" else "connector_v2",
            "action_masking": config.env.action.masking.enabled,
        },
        "agents": {
            "count": config.env.agent_count,
            "definitions": ([item.model_dump(mode="json") for item in config.env.agents]
                            if config.env.agents is not None else None),
        },
        "task": config.env.task.model_dump(mode="json"),
        "rewards": config.env.rewards.model_dump(mode="json"),
        "lifecycle": config.env.lifecycle.model_dump(mode="json"),
        "scenarios": config.env.scenarios.model_dump(mode="json"),
        "geometry_provider": (config.env.geometry.provider.model_dump(mode="json")
                              if config.env.geometry.provider else None),
    }


def _selected_extension_bindings(config: ExperimentConfig) -> set[str]:
    selected: set[str] = set()
    if config.env.rewards.provider:
        selected.add(f"reward:{config.env.rewards.provider.name}")
    if config.env.geometry.provider:
        selected.add(f"geometry:{config.env.geometry.provider.name}")
    if config.env.scenarios.provider:
        selected.add(f"scenario:{config.env.scenarios.provider.name}")
    predicates, outcomes = config.env.action.resolved_pipeline(trail_mode=config.env.trail_mode)
    selected.update(f"predicate:{item.name}" for item in predicates)
    selected.update(f"outcome:{item.name}" for item in outcomes)
    return selected


def clone(checkpoint: Path, scope: str) -> ExecutionRecipe:
    checkpoint = checkpoint.resolve()
    run = checkpoint.parent.parent
    config_path = run / "experiment.yaml"
    state_path = checkpoint / "state.json"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("checkpoint requires parent run experiment.yaml and checkpoint state.json")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("env"), dict) or not isinstance(raw.get("training"), dict):
        raise ValueError("checkpoint parent configuration is not one experiment")
    config, migration = _migrate_config(raw)
    contract = _policy_contract(config)
    extension = []
    assets = []
    gaps = []
    ext_dir = run / "native_extension"
    bindings = []
    if (ext_dir / "extension.json").is_file():
        manifest = json.loads((ext_dir / "extension.json").read_text(encoding="utf-8"))
        for plural, singular in (("rewards", "reward"), ("predicates", "predicate"),
                                 ("outcomes", "outcome"), ("scenarios", "scenario"),
                                 ("geometries", "geometry")):
            bindings.extend(f"{singular}:{name}" for name in manifest.get(plural, []))
        for role, path in (("extension_manifest", ext_dir / "extension.json"),
                           ("extension_binary", ext_dir / manifest["library"])):
            extension.append(Artifact(role=role, path=str(path.resolve()), sha256=_sha(path)))
    geometry = raw["env"].get("geometry") or {}
    catalog_value = geometry.get("compiled_world_catalog_path")
    if catalog_value:
        relative = Path(str(catalog_value))
        candidates = [config_path.parent / relative]
        candidates.extend(parent / relative for parent in config_path.parents if (parent / ".git").exists())
        catalog = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if catalog is not None:
            assets.append(Artifact(role="geometry_catalog", path=str(catalog.parent), sha256=_sha(catalog.parent)))
        else:
            gaps.append(f"geometry catalog could not be resolved: {catalog_value}")
    archived_worlds = run / "worlds"
    if archived_worlds.is_dir():
        assets.append(Artifact(role="archived_worlds", path=str(archived_worlds.resolve()), sha256=_sha(archived_worlds)))
    provenance = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("provenance.json", "source.json", "run.json"):
        path = run / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in ("source_commit", "git_sha", "commit_sha"):
                if payload.get(key): provenance["source_revision"] = str(payload[key])
    if not provenance.get("source_revision"):
        gaps.append("source revision is not recorded in this run")
    return ExecutionRecipe(scope=scope, source_run=str(run.resolve()),
        checkpoint=Artifact(role="rllib_checkpoint", path=str(checkpoint), sha256=_sha(checkpoint)),
        experiment=Artifact(role="resolved_experiment", path=str(config_path.resolve()), sha256=_sha(config_path)),
        checkpoint_state=json.loads(state_path.read_text(encoding="utf-8")), policy_contract=contract,
        config_migration=migration,
        extension=extension, assets=assets, extension_bindings=bindings,
        provenance_gaps=gaps, provenance=provenance)


def make_portable(recipe: ExecutionRecipe, directory: Path) -> ExecutionRecipe:
    """Copy immutable inputs into a new, self-verifying relocatable bundle."""
    directory = directory.resolve()
    if directory.exists():
        raise FileExistsError(f"bundle destination already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    copied = []
    try:
        artifacts = staging / "artifacts"
        artifacts.mkdir()
        inputs = [recipe.checkpoint, recipe.experiment, *recipe.extension, *recipe.assets]
        for index, artifact in enumerate(inputs):
            source = Path(artifact.path)
            if artifact.role == "rllib_checkpoint": target = artifacts / "checkpoint"
            elif artifact.role == "resolved_experiment": target = artifacts / "experiment.yaml"
            elif artifact.role.startswith("extension_"): target = artifacts / "native_extension" / source.name
            else: target = artifacts / f"{index:02d}-{artifact.role}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir(): shutil.copytree(source, target)
            else: shutil.copy2(source, target)
            copied.append(Artifact(role=artifact.role,
                path=target.relative_to(staging).as_posix(), sha256=_sha(target)))
        staging.replace(directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    extension_end = 2 + len(recipe.extension)
    return recipe.model_copy(update={"checkpoint": copied[0], "experiment": copied[1],
                                     "extension": copied[2:extension_end], "assets": copied[extension_end:]})


_BUILTIN_CAPABILITIES = {
    "predicate": {"valid_action", "bounds", "unoccupied"},
    "outcome": {"cursor_movement", "trail_placement"},
}


def _qualified(value: str) -> tuple[str, str]:
    kind, separator, name = value.partition(":")
    if not separator or kind not in {"reward", "predicate", "outcome", "scenario", "geometry"} or not name:
        raise ValueError(f"capability must be qualified as kind:name: {value}")
    return kind, name


def _replace_selector(items: list[dict[str, Any]] | None, source: str, target: str) -> list[dict[str, Any]] | None:
    if items is None:
        return None
    replaced = []
    for item in items:
        current = item if isinstance(item, dict) else {"name": item}
        replaced.append({**current, "name": target} if current.get("name") == source else current)
    return replaced


def _apply_capability_overrides(raw: dict[str, Any], recipe: ExecutionRecipe) -> list[str]:
    disabled = recipe.overrides.disabled_capabilities
    replacements = recipe.overrides.replacements
    unknown_replacements = sorted(set(replacements).difference(disabled))
    if unknown_replacements:
        raise ValueError(f"replacement sources are not disabled: {unknown_replacements}")
    changes: list[str] = []
    env = raw["env"]
    for source in disabled:
        if source not in recipe.extension_bindings:
            raise ValueError(f"disabled extension binding does not exist: {source}")
        if source not in replacements:
            raise ValueError(f"disabled capability requires explicit replacement: {source}")
        source_kind, source_name = _qualified(source)
        target = replacements[source]
        target_kind, target_name = _qualified(target)
        if target_kind != source_kind:
            raise ValueError(f"capability replacement changes kind: {source} -> {target}")
        if target in recipe.extension_bindings and target in disabled:
            raise ValueError(f"capability replacement is also disabled: {target}")
        if source_kind == "reward":
            if target_name not in {"none", "builtin"} and target not in recipe.extension_bindings:
                raise ValueError(f"unknown reward replacement: {target}")
            rewards = env["rewards"]
            if target_name == "none":
                for key in ("step_cost", "collision_cost", "goal_reward", "distance_shaping",
                            "invalid_action_cost", "construction_residual_weight",
                            "construction_overshoot_weight"):
                    rewards[key] = 0.0
                rewards["distance_reward_mode"] = "progress"
                rewards["provider"] = None
            elif target_name == "builtin":
                rewards["provider"] = None
            else:
                rewards["provider"] = {"name": target_name, "parameters": {}}
        elif source_kind in {"predicate", "outcome"}:
            if target not in recipe.extension_bindings and target_name not in _BUILTIN_CAPABILITIES[source_kind]:
                raise ValueError(f"unknown {source_kind} replacement: {target}")
            key = f"{source_kind}s"
            env["action"][key] = _replace_selector(env["action"].get(key), source_name, target_name)
        elif source_kind == "scenario":
            if target_name == "none":
                env["scenarios"]["provider"] = None
            elif target not in recipe.extension_bindings:
                raise ValueError(f"unknown scenario replacement: {target}")
            else:
                env["scenarios"]["provider"] = {"name": target_name, "parameters": {}}
        elif source_kind == "geometry":
            if target_name == "none":
                env["geometry"]["provider"] = None
            elif target not in recipe.extension_bindings:
                raise ValueError(f"unknown geometry replacement: {target}")
            else:
                env["geometry"]["provider"] = {"name": target_name, "parameters": {}}
        changes.append(f"{source} -> {target}")
    return changes


def resolve(recipe: ExecutionRecipe, world: Path | None = None,
            base: Path | None = None) -> tuple[ExperimentConfig, list[dict[str, Any]], list[str]]:
    """Resolve explicit overrides into one strictly validated experiment configuration."""
    experiment_path = Path(recipe.experiment.path)
    if not experiment_path.is_absolute():
        experiment_path = (base or Path.cwd()) / experiment_path
    archived = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    config, _ = _migrate_config(archived)
    raw = config.model_dump(by_alias=True, mode="json")
    effective_world = world or (Path(recipe.overrides.world_manifest)
                                if recipe.overrides.world_manifest else None)
    changes: list[dict[str, Any]] = []
    if effective_world is not None:
        if not effective_world.is_absolute():
            effective_world = (base or Path.cwd()) / effective_world
        if recipe.overrides.task is None or recipe.overrides.routes is None:
            raise ValueError("world replacement requires explicit task and routes decisions")
        manifest = WorldManifest.model_validate_json(effective_world.read_text(encoding="utf-8"))
        extent = list(manifest.extent.as_tuple())
        geometry = raw["env"]["geometry"]
        geometry.update({"extent": extent, "grid_size": None,
                         "compiled_world_path": str(effective_world.parent.resolve()),
                         "compiled_world_catalog_path": None,
                         "world_identity_sha256": manifest.identity_sha256})
        for name in ("task", "routes"):
            decision = getattr(recipe.overrides, name)
            assert decision is not None
            if decision.mode == "replace" and decision.value is None:
                raise ValueError(f"{name} replacement requires a value")
            if decision.mode != "replace" and decision.value is not None:
                raise ValueError(f"{name} {decision.mode} decision cannot include a value")
        task = recipe.overrides.task
        routes = recipe.overrides.routes
        if task.mode == "clear":
            raw["env"]["task"] = {}
        elif task.mode == "replace":
            raw["env"]["task"] = task.value
        if routes.mode == "clear":
            raw["env"]["waypoint_curriculum"] = {"enabled": False}
        elif routes.mode == "replace":
            raw["env"]["waypoint_curriculum"] = routes.value
        old = recipe.checkpoint_state.get("world_contract") or {}
        if old.get("extent") and list(old["extent"]) != extent:
            changes.append({"component": "world.extent", "from": old["extent"], "to": extent})
        changes.append({"component": "world", "from": old.get("identity_sha256")
                        or old.get("catalog_identity_sha256"), "to": manifest.identity_sha256,
                        "manifest": str(effective_world.resolve())})
    capability_changes = _apply_capability_overrides(raw, recipe)
    resolved = ExperimentConfig(**_resolve_typed_configs(raw))
    if effective_world is not None:
        extent = resolved.env.geometry.extent
        assert extent is not None
        points = []
        curriculum = resolved.env.waypoint_curriculum
        if curriculum.initial_start: points.append(curriculum.initial_start)
        if curriculum.initial_goal: points.append(curriculum.initial_goal)
        for route in curriculum.routes:
            points.append(route.start); points.extend(route.waypoints)
        points.extend(resolved.env.task.construction_target_voxels)
        goal = resolved.env.task.goal
        if hasattr(goal, "position") and goal.position is not None: points.append(goal.position)
        if hasattr(goal, "voxels"): points.extend(goal.voxels)
        if any(any(value < 1 or value > extent[index] for index, value in enumerate(point))
               for point in points):
            raise ValueError("replacement world does not contain configured task/curriculum coordinates")
    return resolved, changes, capability_changes


def validate(recipe: ExecutionRecipe, world: Path | None = None, base: Path | None = None) -> dict[str, Any]:
    artifacts = [recipe.checkpoint, recipe.experiment, *recipe.extension, *recipe.assets]
    verified = []
    for artifact in artifacts:
        path = Path(artifact.path)
        if not path.is_absolute(): path = (base or Path.cwd()) / path
        if not path.exists():
            raise FileNotFoundError(f"missing {artifact.role}: {path}")
        if _sha(path) != artifact.sha256:
            raise ValueError(f"tampered {artifact.role}: {path}")
        verified.append(artifact.role)
    experiment_path = Path(recipe.experiment.path)
    if not experiment_path.is_absolute(): experiment_path = (base or Path.cwd()) / experiment_path
    raw = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    config, migration = _migrate_config(raw)
    if migration != recipe.config_migration:
        raise ValueError("archived configuration migration record does not match recipe")
    current_contract = _policy_contract(config)
    if current_contract != recipe.policy_contract:
        raise ValueError("policy contract does not match the archived experiment")
    verified.append("policy_contract")
    manifest_artifact = next((item for item in recipe.extension if item.role == "extension_manifest"), None)
    if manifest_artifact is not None:
        manifest_path = Path(manifest_artifact.path)
        if not manifest_path.is_absolute(): manifest_path = (base or Path.cwd()) / manifest_path
        extension_manifest = NativeExtensionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if extension_manifest.abi_version != ABI_VERSION:
            raise ValueError(f"extension ABI {extension_manifest.abi_version} is not runtime ABI {ABI_VERSION}")
        binary_artifact = next((item for item in recipe.extension if item.role == "extension_binary"), None)
        if binary_artifact is None:
            raise ValueError("extension manifest has no bundled extension binary")
        binary_path = Path(binary_artifact.path)
        if not binary_path.is_absolute():
            binary_path = (base or Path.cwd()) / binary_path
        if binary_path.name != extension_manifest.library:
            raise ValueError("extension binary name does not match the archived manifest")
        binary_sha256 = hashlib.sha256(binary_path.read_bytes()).hexdigest()
        if binary_sha256 != extension_manifest.binary_sha256:
            raise ValueError("extension binary content does not match the archived manifest")
        if extension_manifest.platform != sys.platform or extension_manifest.machine != platform.machine():
            raise ValueError(
                "extension target does not match this runtime: "
                f"{extension_manifest.platform}/{extension_manifest.machine} != "
                f"{sys.platform}/{platform.machine()}"
            )
        bindings = []
        for plural, singular in (("rewards", "reward"), ("predicates", "predicate"),
                                 ("outcomes", "outcome"), ("scenarios", "scenario"),
                                 ("geometries", "geometry")):
            bindings.extend(f"{singular}:{name}" for name in getattr(extension_manifest, plural))
        if bindings != recipe.extension_bindings:
            raise ValueError("extension semantic bindings do not match the archived manifest")
        extension_names = set(bindings)
        manifest_kinds = {item.split(":", 1)[0] for item in bindings}
        missing = sorted(item for item in _selected_extension_bindings(config)
                         if item.split(":", 1)[0] in manifest_kinds and item not in extension_names)
        if missing:
            raise ValueError(f"selected extension bindings are absent from manifest: {missing}")
        verified.append("extension_bindings")
    resolved, changes, capability_changes = resolve(recipe, world, base)
    return {"valid": True, "scope": recipe.scope, "verified": verified, "changes": changes,
            "capability_changes": capability_changes,
            "active_extension_bindings": [item for item in recipe.extension_bindings
                                          if item not in recipe.overrides.disabled_capabilities],
            "effective_config": resolved.model_dump(by_alias=True, mode="json"),
            "inactive": (["learner", "optimizer", "exploration", "curriculum_adaptation"]
                         if recipe.scope == "evaluation" else []),
            "provenance_gaps": recipe.provenance_gaps,
            "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
            "config_migration": recipe.config_migration.model_dump(mode="json"),
            "execution_supported": False,
            "execution_blocker": "this implementation slice validates recipes; policy execution is not enabled yet"}
