"""Inference-only execution of a verified checkpoint recipe."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

import yaml

from theseo_anysearch.experiments.execution_recipe import ExecutionRecipe, _migrate_config, _sha, validate
from theseo_anysearch.experiments.output import OutputStore
from theseo_anysearch.experiments.trajectory import TrajectoryWriter, collect_eval_episodes
from theseo_anysearch.rllib.trainer.evaluation.evaluator import EvaluationMetrics


def _artifact_path(path: str, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def _safe_output_root(recipe: ExecutionRecipe, base: Path, output_dir: Path) -> Path:
    """Keep new runs out of the immutable recipe and its verified inputs."""
    root = output_dir.resolve()
    if any(not Path(item.path).is_absolute() for item in
           (recipe.checkpoint, recipe.experiment, *recipe.extension, *recipe.assets)):
        if root.is_relative_to(base.resolve()):
            raise ValueError("output directory cannot be inside a portable recipe bundle")
    for item in (recipe.checkpoint, recipe.experiment, *recipe.extension, *recipe.assets):
        path = _artifact_path(item.path, base).resolve()
        protected = path if path.is_dir() else path.parent
        if root.is_relative_to(protected):
            raise ValueError(f"output directory overlaps verified {item.role}")
    return root


def _verify_replacement_world(recipe: ExecutionRecipe, world: Path | None, base: Path) -> None:
    selected = world or (Path(recipe.overrides.world_manifest)
                         if recipe.overrides.world_manifest else None)
    if selected is None:
        return
    if not selected.is_absolute():
        selected = base / selected
    from theseo_anysearch.worlds.compiler import validate_compiled_world

    compiled = validate_compiled_world(selected.resolve().parent)
    if selected.resolve() != compiled.root / "manifest.json":
        raise ValueError("replacement world must name its compiled manifest.json")


def _verify_policy_spaces(recipe: ExecutionRecipe, effective_env: dict,
                          base: Path, *, changed: bool) -> None:
    """Reject changed observation/action shapes before restoring policy weights."""
    if not changed:
        return
    from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

    archived_path = _artifact_path(recipe.experiment.path, base)
    archived, _ = _migrate_config(yaml.safe_load(archived_path.read_text(encoding="utf-8")))
    source_env = archived.env.to_runtime_dict()
    _rebind_artifacts(recipe, source_env, base, False)
    source = VoxelEnv(source_env)
    try:
        target = VoxelEnv(effective_env)
        try:
            if source.observation_space != target.observation_space:
                raise ValueError("replacement changes the policy observation space")
            if source.action_space != target.action_space:
                raise ValueError("replacement changes the policy action space")
        finally:
            target.close()
    finally:
        source.close()


def preflight_runtime(recipe: ExecutionRecipe, report: dict, *, base: Path,
                      world: Path | None = None, output_dir: Path | None = None) -> dict:
    """Resolve runtime assets and spaces without starting Ray or writing a run."""
    from theseo_anysearch.experiments.models import ExperimentConfig

    _verify_replacement_world(recipe, world, base)
    if output_dir is not None:
        _safe_output_root(recipe, base, output_dir)
    config = ExperimentConfig.model_validate(report["effective_config"])
    env = config.env.to_runtime_dict()
    _rebind_artifacts(recipe, env, base, world is not None or recipe.overrides.world_manifest is not None)
    _verify_policy_spaces(recipe, env, base, changed=bool(report["changes"] or report["capability_changes"]))
    return env


def _policy_digest(algorithm: Any) -> str:
    """Hash inference module parameters, not mutable optimizer/runner state."""
    runner = getattr(algorithm, "env_runner", None)
    module = getattr(runner, "module", None)
    if module is None and runner is not None and hasattr(runner, "get_module"):
        module = runner.get_module()
    if module is None:
        raise RuntimeError("restored algorithm has no inference module to verify")
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _rebind_artifacts(recipe: ExecutionRecipe, env: dict[str, Any], base: Path,
                      replacing_world: bool) -> None:
    """Use only verified bundle artifacts; never fall back to archived absolute paths."""
    manifest = next((item for item in recipe.extension if item.role == "extension_manifest"), None)
    if manifest is not None:
        env["native_extension_manifest"] = str(_artifact_path(manifest.path, base).resolve())
    if replacing_world:
        return
    catalog = next((item for item in recipe.assets if item.role == "geometry_catalog"), None)
    if catalog is not None:
        root = _artifact_path(catalog.path, base)
        name = Path(str(env.get("compiled_world_catalog_path"))).name
        matches = list(root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"bundled geometry catalog is ambiguous or missing: {name}")
        env["compiled_world_catalog_path"] = str(matches[0].resolve())
        env["compiled_world_path"] = None  # The catalog selects its own first world.
    elif env.get("compiled_world_catalog_path"):
        raise ValueError("archived geometry catalog was not bundled")
    if env.get("compiled_world_path") and catalog is None:
        worlds = next((item for item in recipe.assets if item.role == "archived_worlds"), None)
        if worlds is None:
            raise ValueError("archived compiled world was not bundled")
        identity = env.get("world_identity_sha256")
        path = _artifact_path(worlds.path, base) / str(identity)
        if not path.is_dir():
            raise ValueError(f"bundled compiled world is missing: {identity}")
        env["compiled_world_path"] = str(path.resolve())


def evaluate(recipe: ExecutionRecipe, *, base: Path, output_dir: Path,
             episodes: int, seed: int, world: Path | None = None) -> Path:
    """Evaluate a checkpoint without training, exploration, or optimizer updates."""
    if recipe.scope != "evaluation":
        raise ValueError("policy execution currently supports evaluation scope only")
    if episodes < 1:
        raise ValueError("evaluation requires at least one episode")
    report = validate(recipe, world, base)
    env = preflight_runtime(recipe, report, base=base, world=world, output_dir=output_dir)
    output_root = output_dir.resolve()
    if recipe.provenance_gaps:
        # Gaps are recorded, not disguised as complete reproducibility.
        report["reproducibility"] = "provenance gaps recorded"
    from theseo_anysearch.experiments.models import ExperimentConfig
    from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer, _evaluation_env_config

    config = ExperimentConfig.model_validate(report["effective_config"])
    if config.training.algorithm.lower() != "ppo" or config.env.agent_count != 1:
        raise ValueError("inference executor currently supports single-agent PPO only")
    run_id = uuid.uuid4().hex
    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    store = OutputStore(destination)
    settings = config.to_settings()
    settings.training.output_dir = destination
    settings.training.num_env_runners = 0
    settings.training.num_gpus = 0
    settings.training.require_gpu = False
    settings.evaluation.enabled = False
    settings.evaluation.num_env_runners = 0
    settings.evaluation.parallel_to_training = False
    store.write_json("resolved_recipe.json", recipe.model_dump(mode="json"))
    store.write_json("difference_manifest.json", {
        "source_run": recipe.source_run, "checkpoint_sha256": recipe.checkpoint.sha256,
        "changes": report["changes"], "capability_changes": report["capability_changes"],
        "provenance_gaps": recipe.provenance_gaps, "seed": seed, "episodes": episodes,
    })
    store.write_json("effective_config.json", report["effective_config"])
    algorithm = None
    try:
        algorithm = PPOTrainer.build_algorithm_from_settings(settings, env_config=env)
        evaluation_env = _evaluation_env_config(settings, env)
        store.write_json("runtime_env_config.json", evaluation_env)
        algorithm.restore(str(_artifact_path(recipe.checkpoint.path, base).resolve()))
        before = _policy_digest(algorithm)
        batch = collect_eval_episodes(algorithm, evaluation_env, episodes, seed=seed)
        after = _policy_digest(algorithm)
        if before != after:
            raise RuntimeError("policy weights changed during evaluation")
        checkpoint_after = _sha(_artifact_path(recipe.checkpoint.path, base))
        if checkpoint_after != recipe.checkpoint.sha256:
            raise RuntimeError("checkpoint files changed during evaluation")
        metrics = EvaluationMetrics.from_voxel_episodes(
            batch, evaluation_env, min_success_rate=config.evaluation.min_success_rate)
        writer = TrajectoryWriter(store, trajectory_every=1)
        paths = []
        for index, episode in enumerate(batch):
            path = f"trajectories/episode_{index:06d}.json.zst"
            writer.write_episode(path, episode, experiment_name=config.experiment.name, run_id=run_id)
            paths.append(path)
        store.write_json("metrics.json", metrics.model_dump(mode="json"))
        store.write_json("execution.json", {
            "run_id": run_id, "scope": "evaluation", "seed": seed,
            "episode_seeds": list(range(seed, seed + episodes)),
            "trajectories": paths, "policy_sha256_before": before,
            "policy_sha256_after": after, "policy_unchanged": True,
            "checkpoint_sha256": checkpoint_after, "checkpoint_unchanged": True,
        })
    except BaseException as exc:
        store.write_json("execution_failure.json", {
            "run_id": run_id, "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    finally:
        if algorithm is not None:
            algorithm.stop()
    return destination
