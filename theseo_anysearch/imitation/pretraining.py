"""Behavior-clone a PPO policy from heuristic demonstrations."""

from __future__ import annotations

import copy
import json
import shutil
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
from theseo_anysearch.imitation.models import (
    DemonstrationManifest,
    ImitationConfig,
    ImitationResult,
)
from theseo_anysearch.imitation.cache import (
    cache_key_lock,
    load_cached_pretraining,
    pretraining_cache_key,
    publish_cached_pretraining,
)


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


def _policy_outputs(
    model: Any,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return policy logits and value predictions from either RLlib stack."""

    if hasattr(model, "forward_train"):
        from ray.rllib.core.columns import Columns

        outputs = model.forward_train({Columns.OBS: observations})
        values = outputs.get(Columns.VF_PREDS)
        if values is None:
            raise RuntimeError("PPO RLModule did not expose value predictions")
        return outputs[Columns.ACTION_DIST_INPUTS], values.reshape(-1)

    logits, _ = model(
        {"obs": observations, "obs_flat": observations},
        [],
        None,
    )
    values = model.value_function()
    return logits, values.reshape(-1)


def _policy_logits(model: Any, observations: torch.Tensor) -> torch.Tensor:
    """Return policy logits while preserving the existing helper contract."""

    return _policy_outputs(model, observations)[0]


def _evaluate(
    model: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    label_smoothing: float,
    train_value_head: bool,
    value_loss_coefficient: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    with torch.no_grad():
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        labels = torch.as_tensor(actions, dtype=torch.long, device=device)
        targets = torch.as_tensor(returns, dtype=torch.float32, device=device)
        logits, values = _policy_outputs(model, obs)
        policy_loss, accuracy = _supervised_metrics(
            logits, labels, label_smoothing
        )
        value_loss = (
            functional.smooth_l1_loss(values, targets)
            if train_value_head
            else torch.zeros((), device=device)
        )
        loss = policy_loss + value_loss_coefficient * value_loss
    return (
        float(loss.item()),
        float(accuracy.item()),
        float(policy_loss.item()),
        float(value_loss.item()),
    )


def _supervised_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute categorical or component-wise MultiDiscrete BC metrics."""

    if labels.ndim == 1:
        return (
            functional.cross_entropy(logits, labels, label_smoothing=label_smoothing),
            (logits.argmax(dim=1) == labels).float().mean(),
        )
    branch_count = labels.shape[1]
    if logits.shape[1] % branch_count:
        raise ValueError("policy logits cannot be divided across action components")
    branches = logits.chunk(branch_count, dim=1)
    losses = [
        functional.cross_entropy(branch, labels[:, index], label_smoothing=label_smoothing)
        for index, branch in enumerate(branches)
    ]
    predictions = torch.stack([branch.argmax(dim=1) for branch in branches], dim=1)
    return torch.stack(losses).mean(), (predictions == labels).all(dim=1).float().mean()


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
    train_value_head = imitation.handoff.initialize_value_head
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if _parameter_part(name) != "value" or train_value_head
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
    best_policy_loss = float("inf")
    best_value_loss: float | None = None
    best_state = copy.deepcopy(initial_state)
    epochs_without_improvement = 0
    metrics_path = output_dir.joinpath("metrics.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_completed = 0

    for epoch in range(1, imitation.pretraining.epochs + 1):
        model.train()
        indices = rng.permutation(len(dataset.train_actions))
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
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
            returns = torch.as_tensor(
                dataset.train_returns[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, values = _policy_outputs(model, observations)
            policy_loss, _ = _supervised_metrics(
                logits, actions, imitation.pretraining.label_smoothing
            )
            value_loss = (
                functional.smooth_l1_loss(values, returns)
                if train_value_head
                else torch.zeros((), device=device)
            )
            loss = (
                policy_loss
                + imitation.pretraining.value_loss_coefficient * value_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))

        (
            validation_loss,
            validation_accuracy,
            validation_policy_loss,
            validation_value_loss,
        ) = _evaluate(
            model,
            dataset.validation_observations,
            dataset.validation_actions,
            dataset.validation_returns,
            imitation.pretraining.label_smoothing,
            train_value_head,
            imitation.pretraining.value_loss_coefficient,
            device,
        )
        record = {
            "epoch": epoch,
            "training_loss": sum(losses) / len(losses),
            "training_policy_loss": sum(policy_losses) / len(policy_losses),
            "training_value_loss": sum(value_losses) / len(value_losses),
            "validation_loss": validation_loss,
            "validation_policy_loss": validation_policy_loss,
            "validation_value_loss": validation_value_loss,
            "validation_accuracy": validation_accuracy,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")
        epochs_completed = epoch

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            best_policy_loss = validation_policy_loss
            best_value_loss = (
                validation_value_loss if train_value_head else None
            )
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
        best_validation_policy_loss=best_policy_loss,
        best_validation_value_loss=best_value_loss,
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

    env = VoxelEnv(env_config)
    preprocessor = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    expected = dataset_fingerprint(
        env_config,
        imitation,
        int(np.prod(preprocessor.shape)),
        (
            [int(value) for value in env.action_space.nvec]
            if hasattr(env.action_space, "nvec")
            else int(env.action_space.n)
        ),
    )
    env.close()
    lock_dir = dataset_dir.parent.joinpath(".imitation_dataset_locks")
    with cache_key_lock(
        lock_dir,
        f"dataset-{expected}",
        imitation.cache.lock_timeout_seconds,
    ):
        manifest_path = dataset_dir.joinpath("manifest.json")
        arrays_path = dataset_dir.joinpath("demonstrations.npz")
        if (
            imitation.collection.reuse_dataset
            and manifest_path.exists()
            and arrays_path.exists()
        ):
            stored_manifest = DemonstrationManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if stored_manifest.fingerprint == expected:
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
    model = getattr(policy_or_module, "model", policy_or_module)
    result: ImitationResult | None = None
    cache_key: str | None = None
    cache = imitation.cache
    if cache.enabled:
        cache_dir = (
            Path(cache.directory).resolve()
            if cache.directory
            else run_dir.parent.joinpath("imitation_cache")
        )
        cache_key, contract = pretraining_cache_key(
            model,
            dataset.manifest,
            imitation,
            policy_id="default_policy",
        )
        with cache_key_lock(cache_dir, cache_key, cache.lock_timeout_seconds):
            entry = cache_dir.joinpath(cache_key)
            if cache.refresh and entry.is_dir():
                shutil.rmtree(entry)
            if not cache.refresh:
                result = load_cached_pretraining(
                    model,
                    cache_dir,
                    cache_key,
                    contract,
                    imitation_dir,
                )
            if result is None:
                result = behavior_clone_policy(
                    policy_or_module,
                    dataset,
                    imitation,
                    imitation_dir,
                ).model_copy(update={"cache_key": cache_key})
                publish_cached_pretraining(
                    cache_dir,
                    cache_key,
                    contract,
                    result,
                )
    else:
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
