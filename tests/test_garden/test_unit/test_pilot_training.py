"""Tests for mask isolation, objectives, EMA, and update-counted training."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from theseo_anysearch.garden.masking import (
    DenseMaskAwareEncoder,
    hidden_jacobian_max_abs,
    mask_isolation_max_abs,
    mask_shortcut_advantage,
    sample_patch_mask,
)
from theseo_anysearch.garden.models.objectives import (
    EMATeacher,
    ESDFObjective,
    LatentTargetObjective,
    OccupancyObjective,
    build_pilot_objective,
)
from theseo_anysearch.garden.models.outputs import (
    EncoderMetadata,
    EncoderOutput,
    VoxelLevel,
)
from theseo_anysearch.garden.trainer import UpdateTrainer, UpdateTrainingConfig


def _encoded(batch: int = 2, channels: int = 4, side: int = 5) -> EncoderOutput:
    local = torch.randn(batch, channels, side, side, side)
    global_embedding = local.mean(dim=(2, 3, 4))
    return EncoderOutput(
        global_embedding=global_embedding,
        scale_embeddings={1: global_embedding},
        local_feature_volume=local,
        local_validity_mask=torch.ones(
            batch, 1, side, side, side, dtype=torch.bool
        ),
        metadata=EncoderMetadata((1,), torch.ones(batch, 1)),
    ).validate()


def test_patch_mask_is_reproducible_and_hides_occupied_values() -> None:
    occupancy = torch.zeros(2, 1, 9, 9, 9)
    occupancy[:, :, 4, 4, 4] = 1
    first = sample_patch_mask(occupancy, ratio=0.40, patch_side=2, seed=4)
    second = sample_patch_mask(occupancy, ratio=0.40, patch_side=2, seed=4)
    assert torch.equal(first.hidden_mask, second.hidden_mask)
    assert first.actual_ratio >= 0.40
    assert torch.all((first.hidden_mask & occupancy.bool()).flatten(1).any(dim=1))
    assert sum(first.center_strata.values()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_patch_mask_matches_cpu_selection_on_cuda() -> None:
    occupancy = torch.zeros(2, 1, 9, 9, 9)
    occupancy[:, :, 4, 4, 4] = 1
    expected = sample_patch_mask(occupancy, ratio=0.40, patch_side=2, seed=4)
    actual = sample_patch_mask(
        occupancy.cuda(), ratio=0.40, patch_side=2, seed=4
    )
    assert torch.equal(actual.hidden_mask.cpu(), expected.hidden_mask)
    assert actual.actual_ratio == expected.actual_ratio
    assert actual.center_strata == expected.center_strata


def test_dense_mask_aware_encoder_passes_intervention_and_jacobian_gates() -> None:
    occupancy = torch.zeros(1, 9, 9, 9)
    occupancy[:, 4, 4, 4] = 1
    level = VoxelLevel.from_occupancy(occupancy)
    hidden = torch.zeros_like(level.validity_mask)
    hidden[:, :, 3:6, 3:6, 3:6] = True
    encoder = DenseMaskAwareEncoder(
        stem_width=2, embedding_dim=8, local_channels=2
    )
    assert mask_isolation_max_abs(encoder, level, hidden) <= 1e-6
    assert hidden_jacobian_max_abs(encoder, level, hidden) <= 1e-8
    assert not encoder.executes_sparse_operations


def test_objective_bundles_return_finite_losses_and_counts() -> None:
    encoded = _encoded()
    classes = torch.randint(0, 3, (2, 5, 5, 5))
    mask = torch.zeros(2, 1, 5, 5, 5, dtype=torch.bool)
    mask[:, :, 2:, 2:, 2:] = True
    occupancy = OccupancyObjective(4, focal_gamma=2)(
        encoded, classes, supervision_mask=mask
    )
    assert torch.isfinite(occupancy.loss) and occupancy.supervised_values == 54
    esdf = ESDFObjective(4, truncation=4)(
        encoded, torch.randn(2, 5, 5, 5)
    )
    assert torch.isfinite(esdf.loss) and esdf.supervised_values == 250
    latent = LatentTargetObjective(4)(
        encoded, _encoded(), supervision_mask=mask
    )
    assert torch.isfinite(latent.loss) and latent.supervised_values == 54
    assert isinstance(
        build_pilot_objective("T0", local_channels=4), OccupancyObjective
    )
    assert isinstance(
        build_pilot_objective("T1", local_channels=4), OccupancyObjective
    )
    assert isinstance(
        build_pilot_objective("T2", local_channels=4, truncation=2), ESDFObjective
    )
    assert isinstance(
        build_pilot_objective("T3", local_channels=4), LatentTargetObjective
    )


def test_ema_teacher_is_frozen_and_moves_toward_online_encoder() -> None:
    online = nn.Sequential(nn.Linear(2, 2))
    teacher = EMATeacher(online, decay=0.5)
    before = next(teacher.encoder.parameters()).clone()
    with torch.no_grad():
        next(online.parameters()).add_(2)
    teacher.update(online)
    after = next(teacher.encoder.parameters())
    assert torch.allclose(after, before + 1)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_mask_shortcut_is_relative_to_stratum_frequency_baseline() -> None:
    target = torch.tensor([0, 0, 1, 1, 1, 0])
    strata = torch.tensor([0, 0, 0, 1, 1, 1])
    frequency_prediction = torch.tensor([0, 0, 0, 1, 1, 1])
    assert mask_shortcut_advantage(frequency_prediction, target, strata) == 0
    assert mask_shortcut_advantage(target, target, strata) > 0


def test_update_trainer_counts_optimizer_updates_and_dense_work() -> None:
    module = nn.Linear(2, 1)
    trainer = UpdateTrainer(
        module,
        UpdateTrainingConfig(
            total_updates=2,
            peak_learning_rate=1e-3,
            accumulation_steps=2,
        ),
    )
    for micro_step in range(4):
        prediction = module(torch.ones(3, 2))
        updated = trainer.step(
            prediction.square().mean(),
            observations=3,
            encoded_views=3,
            valid_voxels=81,
        )
        assert updated == (micro_step % 2 == 1)
    report = trainer.report()
    assert report.updates == 2
    assert report.observations == report.encoded_views == 12
    assert report.valid_voxels == 324
    assert report.dense_masked_voxels_skipped == 0
    assert report.trainable_parameters == 3
    assert report.checkpoint_bytes == 12
