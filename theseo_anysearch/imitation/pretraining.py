"""Behavior-clone a PPO policy from heuristic demonstrations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from ray.rllib.models import ModelCatalog

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.imitation.dataset import (
    DemonstrationDataset,
    collect_demonstrations,
    dataset_fingerprint,
    load_compatible_dataset,
    save_dataset,
)
from theseo_anysearch.imitation.models import ImitationConfig, ImitationResult


def _parameter_part(name: str) -> str:
    """Classify common RLlib/custom-model parameter names for handoff."""

    lowered = name.lower()
    if "value" in lowered or "vf" in lowered:
        return "value"
    if "policy_head" in lowered or "logits" in lowered:
        return "policy"
    return "encoder"


def _part_enabled(part: str, imitation: ImitationConfig) -> bool:
    handoff = imitation.handoff
    return {
        "encoder": handoff.initialize_encoder,
        "policy": handoff.initialize_policy,
        "value": handoff.initialize_value_head,
    }[part]


def _policy_logits(model: Any, observations: torch.Tensor) -> torch.Tensor:
    """Run either a legacy TorchModelV2 or a modern RLModule."""

    if hasattr(model, "forward_train"):
        from ray.rllib.core.columns import Columns

        outputs = model.forward_train({Columns.OBS: observations})
        return outputs[Columns.ACTION_DIST_INPUTS]

    logits, _ = model(
        {"obs": observations, "obs_flat": observations},
        [],
        None,
    )
    return logits


def _evaluate(
    model: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    label_smoothing: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        labels = torch.as_tensor(actions, dtype=torch.long, device=device)
        logits = _policy_logits(model, obs)
        loss = functional.cross_entropy(
            logits,
            labels,
            label_smoothing=label_smoothing,
        )
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return float(loss.item()), float(accuracy.item())


def behavior_clone_policy(
    policy: Any,
    dataset: DemonstrationDataset,
    imitation: ImitationConfig,
    output_dir: Path,
) -> ImitationResult:
    """Optimize policy logits and retain only configured handoff parameters."""

    model = getattr(policy, "model", policy)
    device = next(model.parameters()).device
    initial_state = copy.deepcopy(model.state_dict())
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if _parameter_part(name) != "value"
    ]
    if not trainable:
        raise RuntimeError("PPO policy exposes no trainable imitation parameters")
    optimizer = torch.optim.Adam(
        trainable,
        lr=imitation.pretraining.learning_rate,
    )
    rng = np.random.default_rng(imitation.collection.seed_start)
    best_loss = float("inf")
    best_accuracy = 0.0
    best_state = copy.deepcopy(initial_state)
    epochs_without_improvement = 0
    metrics_path = output_dir.joinpath("metrics.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_completed = 0

    for epoch in range(1, imitation.pretraining.epochs + 1):
        model.train()
        indices = rng.permutation(len(dataset.train_actions))
        losses: list[float] = []
        for offset in range(0, len(indices), imitation.pretraining.batch_size):
            batch_indices = indices[offset:offset + imitation.pretraining.batch_size]
            observations = torch.as_tensor(
                dataset.train_observations[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                dataset.train_actions[batch_indices],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _policy_logits(model, observations)
            loss = functional.cross_entropy(
                logits,
                actions,
                label_smoothing=imitation.pretraining.label_smoothing,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        validation_loss, validation_accuracy = _evaluate(
            model,
            dataset.validation_observations,
            dataset.validation_actions,
            imitation.pretraining.label_smoothing,
            device,
        )
        record = {
            "epoch": epoch,
            "training_loss": sum(losses) / len(losses),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")
        epochs_completed = epoch

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                epochs_without_improvement
                >= imitation.pretraining.early_stopping_patience
            ):
                break

    handed_off_state = {}
    for name, trained_value in best_state.items():
        part = _parameter_part(name)
        handed_off_state[name] = (
            trained_value if _part_enabled(part, imitation) else initial_state[name]
        )
    model.load_state_dict(handed_off_state)
    checkpoint_path = output_dir.joinpath("policy_state.pt")
    torch.save(
        {
            "model_state": handed_off_state,
            "manifest": dataset.manifest.model_dump(mode="json"),
            "handoff": imitation.handoff.model_dump(mode="json"),
        },
        checkpoint_path,
    )
    return ImitationResult(
        epochs_completed=epochs_completed,
        best_validation_loss=best_loss,
        validation_accuracy=best_accuracy,
        training_samples=len(dataset.train_actions),
        validation_samples=len(dataset.validation_actions),
        checkpoint_path=str(checkpoint_path),
    )


def _dataset_for_run(
    env_config: dict[str, Any],
    imitation: ImitationConfig,
    dataset_dir: Path,
) -> DemonstrationDataset:
    """Reuse a compatible run dataset or collect a new one."""

    manifest_path = dataset_dir.joinpath("manifest.json")
    arrays_path = dataset_dir.joinpath("demonstrations.npz")
    if imitation.collection.reuse_dataset and manifest_path.exists() and arrays_path.exists():
        env = VoxelEnv(env_config)
        preprocessor = ModelCatalog.get_preprocessor_for_space(env.observation_space)
        expected = dataset_fingerprint(
            env_config,
            imitation,
            int(np.prod(preprocessor.shape)),
            int(env.action_space.n),
        )
        env.close()
        return load_compatible_dataset(dataset_dir, expected)

    dataset = collect_demonstrations(env_config, imitation)
    save_dataset(dataset, dataset_dir)
    return dataset


def _trainable_policy_or_module(algorithm: Any) -> tuple[Any, bool]:
    """Return the learner-owned RLModule or a legacy Policy."""

    learner_group = getattr(algorithm, "learner_group", None)
    if learner_group is not None:
        if not learner_group.is_local:
            raise RuntimeError(
                "imitation pretraining currently requires num_learners: 0"
            )
        multi_module = learner_group._learner.module
        module = (
            multi_module.get("default_policy")
            if hasattr(multi_module, "get")
            else multi_module["default_policy"]
        )
        return module, True
    return algorithm.get_policy(), False


def run_imitation_pretraining(
    algorithm: Any,
    env_config: dict[str, Any],
    imitation: ImitationConfig,
    run_dir: Path,
) -> ImitationResult | None:
    """Run the configured pre-PPO imitation stage in the current algorithm."""

    if not imitation.enabled:
        return None
    imitation_dir = run_dir.joinpath("imitation")
    dataset_dir = (
        Path(imitation.collection.dataset_dir).resolve()
        if imitation.collection.dataset_dir
        else imitation_dir.joinpath("dataset")
    )
    dataset = _dataset_for_run(
        env_config,
        imitation,
        dataset_dir,
    )
    policy_or_module, uses_modern_module = _trainable_policy_or_module(algorithm)
    result = behavior_clone_policy(
        policy_or_module,
        dataset,
        imitation,
        imitation_dir,
    )
    if uses_modern_module:
        algorithm.env_runner_group.sync_weights(
            from_worker_or_learner_group=algorithm.learner_group,
            inference_only=False,
        )
    from theseo_anysearch.experiments.trajectory import collect_eval_episodes
    from theseo_anysearch.rllib.trainer.evaluation import EvaluationMetrics

    episodes = collect_eval_episodes(
        algorithm,
        env_config,
        1,
        seed=int(env_config.get("seed", 42)),
    )
    pre_rl_metrics = EvaluationMetrics.from_voxel_episodes(
        episodes,
        env_config,
        min_success_rate=1.0,
    )
    result = result.model_copy(
        update={"pre_rl_success_rate": pre_rl_metrics.success_rate}
    )
    imitation_dir.joinpath("pre_rl_evaluation.json").write_text(
        pre_rl_metrics.model_dump_json(indent=2),
        encoding="utf-8",
    )
    imitation_dir.joinpath("result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return result
