"""Tests for heuristic demonstration collection and PPO behavior cloning."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from ray.rllib.core.columns import Columns

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
)
from theseo_anysearch.imitation.pretraining import behavior_clone_policy


class TinyPolicyModel(torch.nn.Module):
    """Small TorchModelV2-shaped module used to verify weight handoff."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 8)
        self.policy_head = torch.nn.Linear(8, 2)
        self.value_head = torch.nn.Linear(8, 1)

    def forward(self, input_dict, state, sequence_lengths):
        del sequence_lengths
        features = torch.relu(self.encoder(input_dict["obs_flat"]))
        self.value_head(features)
        return self.policy_head(features), state


class TinyPolicy:
    """Policy wrapper exposing the model attribute used by pretraining."""

    def __init__(self) -> None:
        self.model = TinyPolicyModel()


class TinyRLModule(TinyPolicyModel):
    """Modern RLModule-shaped model used to exercise direct cloning."""

    def forward_train(self, batch):
        features = torch.relu(self.encoder(batch[Columns.OBS]))
        self.value_head(features)
        return {Columns.ACTION_DIST_INPUTS: self.policy_head(features)}


def _manifest() -> DemonstrationManifest:
    return DemonstrationManifest(
        fingerprint="abc",
        teacher_type="astar",
        teacher_weight=None,
        requested_episodes=4,
        successful_episodes=4,
        accepted_episodes=4,
        attempted_episodes=4,
        training_episodes=3,
        validation_episodes=1,
        training_samples=4,
        validation_samples=2,
        observation_size=2,
        action_count=2,
        seeds=[1, 2, 3, 4],
    )


def _linearly_separable_dataset() -> DemonstrationDataset:
    return DemonstrationDataset(
        train_observations=np.asarray(
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        ),
        train_actions=np.asarray([0, 0, 1, 1], dtype=np.int64),
        validation_observations=np.asarray(
            [[-1.5, 0.0], [1.5, 0.0]], dtype=np.float32
        ),
        validation_actions=np.asarray([0, 1], dtype=np.int64),
        manifest=_manifest(),
    )


def test_teacher_weight_validation_is_explicit():
    with pytest.raises(ValidationError, match="only valid for weighted_astar"):
        ImitationConfig(teacher={"type": "astar", "weight": 2.0})


def test_collection_attempt_budget_must_cover_requested_episodes():
    with pytest.raises(ValidationError, match="max_attempts must be at least episodes"):
        ImitationConfig(collection={"episodes": 5, "max_attempts": 4})


def test_collection_accepts_shared_dataset_directory():
    config = ImitationConfig(
        collection={
            "episodes": 2,
            "max_attempts": 2,
            "dataset_dir": "runtime/shared-demonstrations",
        }
    )

    assert config.collection.dataset_dir == "runtime/shared-demonstrations"


def test_behavior_cloning_learns_actions_and_preserves_value_head(tmp_path):
    torch.manual_seed(4)
    policy = TinyPolicy()
    value_before = {
        name: value.detach().clone()
        for name, value in policy.model.state_dict().items()
        if "value" in name
    }
    config = ImitationConfig(
        enabled=True,
        collection={"episodes": 4, "max_attempts": 4},
        pretraining={
            "epochs": 100,
            "batch_size": 4,
            "learning_rate": 0.03,
            "early_stopping_patience": 20,
        },
    )

    result = behavior_clone_policy(
        policy,
        _linearly_separable_dataset(),
        config,
        tmp_path,
    )

    assert result.validation_accuracy == pytest.approx(1.0)
    assert tmp_path.joinpath("policy_state.pt").is_file()
    for name, value in policy.model.state_dict().items():
        if "value" in name:
            assert torch.equal(value, value_before[name])


def test_behavior_cloning_supports_modern_rl_module(tmp_path):
    torch.manual_seed(4)
    module = TinyRLModule()
    config = ImitationConfig(
        enabled=True,
        collection={"episodes": 4, "max_attempts": 4},
        pretraining={
            "epochs": 100,
            "batch_size": 4,
            "learning_rate": 0.03,
            "early_stopping_patience": 20,
        },
    )

    result = behavior_clone_policy(
        module,
        _linearly_separable_dataset(),
        config,
        tmp_path,
    )

    assert result.validation_accuracy == pytest.approx(1.0)


def test_dataset_reuse_rejects_schema_mismatch(tmp_path):
    save_dataset(_linearly_separable_dataset(), tmp_path)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_compatible_dataset(tmp_path, "different")


def test_dataset_fingerprint_canonicalizes_geometry_paths(tmp_path, monkeypatch):
    geometry = tmp_path.joinpath("geometry.stl")
    geometry.write_bytes(b"solid test")
    monkeypatch.chdir(tmp_path)
    config = ImitationConfig(collection={"episodes": 2, "max_attempts": 2})

    relative = dataset_fingerprint(
        {"stl_path": "geometry.stl"}, config, observation_size=36, action_count=26
    )
    absolute = dataset_fingerprint(
        {"stl_path": str(geometry.resolve())},
        config,
        observation_size=36,
        action_count=26,
    )

    assert relative == absolute


def test_astar_collection_records_pre_action_observations(tmp_path):
    waypoints = tmp_path.joinpath("waypoints.json")
    waypoints.write_text(
        json.dumps({"start": [4, 4, 4], "goal": [4, 4, 6]}),
        encoding="utf-8",
    )
    env_config = {
        "waypoints_file": str(waypoints),
        "grid_size": 8,
        "max_steps": 10,
        "agent_count": 1,
        "obs_mode": "radial",
        "ray_max_len": 10,
        "trail_mode": False,
    }
    config = ImitationConfig(
        enabled=True,
        collection={
            "episodes": 2,
            "seed_start": 10,
            "max_attempts": 2,
            "validation_fraction": 0.5,
        },
    )

    dataset = collect_demonstrations(env_config, config)

    assert dataset.manifest.successful_episodes == 2
    assert dataset.manifest.training_samples == 2
    assert dataset.manifest.validation_samples == 2
    assert dataset.train_observations.shape[1] == dataset.manifest.observation_size
    assert np.all((dataset.train_actions >= 0) & (dataset.train_actions < 26))
