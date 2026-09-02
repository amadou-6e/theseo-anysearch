"""Tests for frozen probes, cross-fitting, and paired controls."""
from __future__ import annotations

import torch
import pytest
from torch import nn

from theseo_anysearch.garden.evaluation.controls import (
    control_target_assignment,
    evaluate_controls,
    shuffled_embedding_output,
    zero_embedding_output,
)
from theseo_anysearch.garden.evaluation.probes import (
    CoordinateProbe,
    GlobalLinearProbe,
    PairTopologyProbe,
    TopologyDecoder,
    encoder_state_sha256,
    extract_frozen,
    make_cross_fit_folds,
    train_probe_step,
)
from theseo_anysearch.garden.models.backbones import DenseResidualBackbone
from theseo_anysearch.garden.models.outputs import EncoderMetadata, EncoderOutput, VoxelLevel


def _encoded(batch: int = 4) -> EncoderOutput:
    global_embedding = torch.arange(batch * 6, dtype=torch.float32).reshape(batch, 6)
    local = torch.arange(batch * 3 * 5**3, dtype=torch.float32).reshape(batch, 3, 5, 5, 5)
    return EncoderOutput(
        global_embedding=global_embedding,
        scale_embeddings={1: global_embedding.clone()},
        local_feature_volume=local,
        local_validity_mask=torch.ones(batch, 1, 5, 5, 5, dtype=torch.bool),
        metadata=EncoderMetadata(
            active_strides=(1,), validity_fractions=torch.ones(batch, 1)
        ),
    ).validate()


def test_cross_fit_uses_three_disjoint_eight_geometry_report_blocks() -> None:
    geometry_ids = [f"geometry-{index:02d}" for index in range(24)]
    folds = make_cross_fit_folds(geometry_ids, seed=17)
    assert folds == make_cross_fit_folds(list(reversed(geometry_ids)), seed=17)
    assert [len(fold.selection_geometry_ids) for fold in folds] == [16, 16, 16]
    assert [len(fold.report_geometry_ids) for fold in folds] == [8, 8, 8]
    assert set().union(*(set(fold.report_geometry_ids) for fold in folds)) == set(geometry_ids)
    assert all(
        not set(fold.selection_geometry_ids) & set(fold.report_geometry_ids) for fold in folds
    )


def test_control_targets_are_deterministic_and_preserve_each_stratum_marginal() -> None:
    labels = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
    strata = ["a"] * 4 + ["b"] * 4
    geometry_ids = [f"g-{index}" for index in range(8)]
    query_ids = [f"q-{index}" for index in range(8)]
    kwargs = {
        "strata": strata,
        "split": "pilot-dev",
        "geometry_ids": geometry_ids,
        "query_ids": query_ids,
        "control_seed": 9,
    }
    first = control_target_assignment(labels, **kwargs)
    second = control_target_assignment(labels, **kwargs)
    assert torch.equal(first, second)
    for indices in (slice(0, 4), slice(4, 8)):
        assert torch.equal(torch.sort(first[indices]).values, torch.sort(labels[indices]).values)


def test_zero_and_shuffle_controls_cover_every_learned_feature() -> None:
    encoded = _encoded()
    zeroed = zero_embedding_output(encoded)
    assert torch.count_nonzero(zeroed.global_embedding) == 0
    assert torch.count_nonzero(zeroed.local_feature_volume) == 0
    assert torch.equal(zeroed.local_validity_mask, encoded.local_validity_mask)

    geometry_ids = ["a", "a", "b", "b"]
    shuffled = shuffled_embedding_output(encoded, geometry_ids, seed=3)
    for destination, row in enumerate(shuffled.global_embedding):
        source = int(torch.where((encoded.global_embedding == row).all(dim=1))[0])
        assert geometry_ids[source] != geometry_ids[destination]
        assert torch.equal(
            shuffled.local_feature_volume[destination], encoded.local_feature_volume[source]
        )


def test_control_evaluation_uses_better_embedding_ablation() -> None:
    result = evaluate_controls(
        real_score=0.80,
        control_target_score=0.65,
        zero_embedding_score=0.60,
        shuffled_embedding_score=0.72,
        selectivity_min=0.10,
        embedding_necessity_min=0.10,
    )
    assert result.selectivity == pytest.approx(0.15)
    assert result.embedding_necessity == pytest.approx(0.08)
    assert result.passes_selectivity
    assert not result.passes_embedding_necessity


def test_probe_output_contracts() -> None:
    encoded = _encoded(batch=2)
    coordinates = torch.tensor(
        [[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]] * 2, dtype=torch.float32
    )
    metadata = torch.ones(2)

    assert GlobalLinearProbe(6, 4)(encoded).shape == (2, 4)
    coordinate_values, coordinate_valid = CoordinateProbe(3, 6, 4)(
        encoded, coordinates, voxel_size=metadata, stride=metadata
    )
    assert coordinate_values.shape == (2, 2, 4)
    assert coordinate_valid.all()
    pair_values, pair_valid = PairTopologyProbe(3, 6)(
        encoded,
        coordinates[:, :1],
        coordinates[:, 1:],
        voxel_size=metadata,
        stride=metadata,
    )
    assert pair_values.shape == (2, 1, 2)
    assert pair_valid.all()
    decoded, valid = TopologyDecoder(3, 2)(encoded)
    assert decoded.shape == (2, 2, 5, 5, 5)
    assert valid.shape == (2, 1, 5, 5, 5)


def test_frozen_extraction_and_probe_update_never_mutate_encoder() -> None:
    torch.manual_seed(4)
    encoder = DenseResidualBackbone(
        stem_width=2, blocks_per_stage=(1, 1, 1, 1), embedding_dim=8, local_channels=2
    )
    level = VoxelLevel.from_occupancy(torch.zeros(2, 9, 9, 9))
    before = encoder_state_sha256(encoder)
    encoded = extract_frozen(encoder, level)
    assert encoder_state_sha256(encoder) == before
    assert not encoded.global_embedding.requires_grad

    probe = GlobalLinearProbe(8, 1)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.1)
    probe_before = probe.linear.weight.detach().clone()
    loss = train_probe_step(
        encoder,
        probe,
        (level,),
        lambda module, output: module(output).squeeze(1),
        torch.ones(2),
        nn.MSELoss(),
        optimizer,
    )
    assert loss >= 0
    assert encoder_state_sha256(encoder) == before
    assert not torch.equal(probe.linear.weight, probe_before)


def test_encoder_hash_detects_parameter_changes() -> None:
    module = nn.Linear(2, 2)
    before = encoder_state_sha256(module)
    with torch.no_grad():
        module.weight[0, 0] += 1
    assert encoder_state_sha256(module) != before


def test_encoder_hash_supports_scalar_integer_buffers() -> None:
    module = nn.BatchNorm3d(2)
    before = encoder_state_sha256(module)
    module.num_batches_tracked += 1
    assert encoder_state_sha256(module) != before
