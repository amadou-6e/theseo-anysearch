"""Tests for update-counted comparative pilot execution."""
from __future__ import annotations

import torch

from theseo_anysearch.garden.pilots.comparative import (
    ComparativeTrialConfig,
    ordered_trial_batch,
    train_comparative_trial,
)
from theseo_anysearch.garden.splits import GeometryDescriptor


def _descriptors() -> list[GeometryDescriptor]:
    return [
        GeometryDescriptor(
            geometry_id=f"trial-{index}",
            family=("open", "thin_obstacle", "topology", "imported")[index % 4],
            occupancy_band=("low", "medium", "high")[index % 3],
            source="unit-test",
        )
        for index in range(12)
    ]


def test_trial_config_locks_five_checkpoint_updates() -> None:
    config = ComparativeTrialConfig("T0", 1e-4, 0, 2_000)
    assert config.checkpoint_updates == (200, 500, 1_000, 1_500, 2_000)
    assert len(config.identity_sha256) == 64


def test_ordered_batches_are_paired_across_objectives() -> None:
    left = ComparativeTrialConfig("T0", 1e-4, 0, 2)
    right = ComparativeTrialConfig("T3", 3e-4, 1, 2)
    first = ordered_trial_batch(_descriptors(), left, 1, device=torch.device("cpu"))
    second = ordered_trial_batch(_descriptors(), right, 1, device=torch.device("cpu"))
    assert [item.observation_id for item in first.observations] == [
        item.observation_id for item in second.observations
    ]
    assert first.level.features.shape == (2, 4, 33, 33, 33)


def test_short_cpu_trial_counts_updates_and_preserves_frozen_callback() -> None:
    config = ComparativeTrialConfig(
        "T0", 1e-3, 0, 2, batch_size=1, checkpoint_fractions=(0.5, 1.0)
    )
    calls: list[int] = []

    def evaluate(model: torch.nn.Module, update: int) -> dict[str, object]:
        calls.append(update)
        return {"finite": all(torch.isfinite(value).all() for value in model.state_dict().values())}

    _, result = train_comparative_trial(
        _descriptors(), config, device=torch.device("cpu"), evaluate=evaluate
    )
    assert result.status == "completed"
    assert result.resources["updates"] == 2
    assert result.resources["observations"] == 2
    assert result.initial_encoder_sha256 != result.final_encoder_sha256
    assert calls == [1, 2]
