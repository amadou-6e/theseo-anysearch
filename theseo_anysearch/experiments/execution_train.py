"""Explicit continuation and weight-only fine-tuning from execution recipes."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import yaml

from theseo_anysearch.experiments.execution_evaluate import (
    _artifact_path, _policy_digest, preflight_runtime,
)
from theseo_anysearch.experiments.execution_recipe import ExecutionRecipe, _sha, validate
from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.experiments.output import OutputStore
from theseo_anysearch.rllib.trainer.checkpointing import CheckpointState


def _snapshot_requirements(config: ExperimentConfig, checkpoint: Path) -> list[tuple[str, Path]]:
    required = []
    if config.env.waypoint_curriculum.enabled:
        required.append("curriculum/state.json")
    if config.training.early_stop.enabled:
        required.append("early_stop_state.json")
    missing = [name for name in required
               if not (checkpoint / "anysearch_state" / name).is_file()]
    if missing:
        raise ValueError("exact continuation needs checkpoint-local state: " + ", ".join(missing))
    return [(name, checkpoint / "anysearch_state" / name) for name in required]


def _bind_geometry(config: ExperimentConfig, env: dict) -> None:
    geometry = config.env.geometry
    if env.get("compiled_world_catalog_path"):
        geometry.compiled_world_catalog_path = Path(env["compiled_world_catalog_path"])
        geometry.compiled_world_path = None
    elif env.get("compiled_world_path"):
        geometry.compiled_world_catalog_path = None
        geometry.compiled_world_path = Path(env["compiled_world_path"])
    if env.get("world_identity_sha256"):
        geometry.world_identity_sha256 = env["world_identity_sha256"]


def train_recipe(recipe: ExecutionRecipe, *, base: Path, output_dir: Path,
                 iterations: int, world: Path | None = None) -> Path:
    """Run a bounded, identity-separated continuation or fine-tune job."""
    if recipe.scope not in {"continuation", "fine_tuning"}:
        raise ValueError("training execution requires continuation or fine_tuning scope")
    if iterations < 1:
        raise ValueError("training requires an explicit positive iteration limit")
    report = validate(recipe, world, base)
    if not report["execution_supported"]:
        raise ValueError(str(report["execution_blocker"]))
    env = preflight_runtime(recipe, report, base=base, world=world, output_dir=output_dir)
    output_root = output_dir.resolve()
    config = ExperimentConfig.model_validate(report["effective_config"])
    if config.training.algorithm.lower() != "ppo" or config.env.agent_count != 1:
        raise ValueError("training executor currently supports single-agent PPO only")
    if config.env.geometry.stl_path or config.env.geometry.stl_paths or config.env.geometry.pool:
        raise ValueError("training executor requires bundled compiled geometry or generated grid")
    checkpoint = _artifact_path(recipe.checkpoint.path, base).resolve()
    state = CheckpointState.model_validate_json((checkpoint / "state.json").read_text(encoding="utf-8"))
    if recipe.scope == "continuation":
        if report["changes"] or report["capability_changes"]:
            raise ValueError("exact continuation cannot change world, task, routes, or capabilities")
        snapshots = _snapshot_requirements(config, checkpoint)
        total_iterations = state.iteration + iterations
    else:
        snapshots = []
        total_iterations = iterations
    if recipe.scope == "continuation":
        from theseo_anysearch.worlds.manifest import world_contract

        if state.world_contract is None or state.world_contract != world_contract(env):
            raise ValueError("exact continuation requires a matching checkpoint world contract")
    _bind_geometry(config, env)
    run_id = uuid.uuid4().hex
    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    store = OutputStore(destination)
    config.training.output_dir = destination
    config.training.iterations = total_iterations
    config.experiment.name = f"{config.experiment.name}-{recipe.scope}"
    # Pretraining is a source-run event, never repeated by clone/apply.
    config.imitation.enabled = False
    config_path = destination / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(by_alias=True, mode="json")), encoding="utf-8")
    for artifact in recipe.extension:
        source = _artifact_path(artifact.path, base)
        target = destination / "native_extension" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name, source in snapshots:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    resets = (["run_identity", "imitation_repetition"] if recipe.scope == "continuation"
              else ["run_identity", "learner", "optimizer", "iteration", "episodes_total",
                    "curriculum", "early_stop", "imitation_repetition"])
    store.write_json("resolved_recipe.json", recipe.model_dump(mode="json"))
    store.write_json("difference_manifest.json", {
        "scope": recipe.scope, "source_run": recipe.source_run,
        "source_checkpoint_sha256": recipe.checkpoint.sha256,
        "changes": report["changes"], "capability_changes": report["capability_changes"],
        "resets": resets, "provenance_gaps": recipe.provenance_gaps,
        "requested_iterations": iterations, "target_total_iterations": total_iterations,
        "source_iteration": state.iteration,
    })
    store.write_json("effective_config.json", config.model_dump(by_alias=True, mode="json"))
    trainer = None
    source_algorithm = None
    source_policy_digest = None
    try:
        from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer

        settings = config.to_settings()
        trainer = PPOTrainer.from_settings(settings)
        if recipe.scope == "continuation":
            trainer._algo = trainer._build_algorithm()
            trainer._algo.restore(str(checkpoint))
            source_policy_digest = _policy_digest(trainer._algo)
            trainer._iteration = state.iteration
            trainer._episodes_total = state.episodes_total
        else:
            source_algorithm = PPOTrainer.build_algorithm_from_settings(settings, env_config=env)
            source_algorithm.restore(str(checkpoint))
            source_policy_digest = _policy_digest(source_algorithm)
            weights = source_algorithm.get_weights()
            source_algorithm.stop()
            source_algorithm = None
            trainer._algo = trainer._build_algorithm()
            trainer._algo.learner_group.set_weights(weights)
            trainer._algo.env_runner_group.sync_weights(
                from_worker_or_learner_group=trainer._algo.learner_group,
                timeout_seconds=30.0,
                inference_only=True,
            )
            if trainer._algo.eval_env_runner_group is not None:
                trainer._algo.eval_env_runner_group.sync_weights(
                    from_worker_or_learner_group=trainer._algo.learner_group,
                    timeout_seconds=30.0,
                    inference_only=True,
                )
        initial_policy_digest = _policy_digest(trainer._algo)
        if source_policy_digest != initial_policy_digest:
            raise RuntimeError("restored policy weights differ from the source checkpoint")
        results = trainer.train()
        final_checkpoint = destination / "checkpoints" / f"iter_{trainer._iteration:06d}"
        if not (final_checkpoint / "state.json").is_file():
            final_checkpoint = trainer.checkpoint()
        if _sha(checkpoint) != recipe.checkpoint.sha256:
            raise RuntimeError("source checkpoint changed during training execution")
        store.write_json("execution.json", {
            "run_id": run_id, "scope": recipe.scope,
            "source_iteration": state.iteration,
            "completed_iterations": trainer._iteration - (state.iteration if recipe.scope == "continuation" else 0),
            "final_iteration": trainer._iteration,
            "episodes_total": trainer._episodes_total,
            "final_checkpoint": str(final_checkpoint.relative_to(destination).as_posix()),
            "source_checkpoint_unchanged": True,
            "source_policy_sha256": source_policy_digest,
            "initial_policy_sha256": initial_policy_digest,
            "result_count": len(results), "resets": resets,
        })
    except BaseException as exc:
        store.write_json("execution_failure.json", {
            "run_id": run_id, "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    finally:
        if source_algorithm is not None:
            source_algorithm.stop()
        if trainer is not None and trainer._algo is not None:
            trainer._algo.stop()
    return destination
