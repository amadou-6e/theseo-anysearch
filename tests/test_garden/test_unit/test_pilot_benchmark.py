"""Tests for the fixed comparative frozen-probe benchmark."""
from __future__ import annotations

import torch

from theseo_anysearch.garden.models.backbones import DenseResidualBackbone
from theseo_anysearch.garden.pilots.benchmark import (
    ProbeProtocol,
    evaluate_frozen_representation,
)
from theseo_anysearch.garden.pilots.contracts import ScoreAnchor
from theseo_anysearch.garden.splits import GeometryDescriptor


def _descriptors(prefix: str) -> list[GeometryDescriptor]:
    return [
        GeometryDescriptor(
            geometry_id=f"{prefix}-{index}",
            family=("open", "thin_obstacle", "topology", "imported")[index % 4],
            occupancy_band=("low", "medium", "high")[index % 3],
            source="unit-test",
        )
        for index in range(4)
    ]


def _anchors() -> dict[str, ScoreAnchor]:
    return {
        name: ScoreAnchor(
            higher_is_better=name not in {"clearance_nmae", "geodesic_nmae"},
            floor=0.0 if name not in {"clearance_nmae", "geodesic_nmae"} else 1.0,
            ceiling=1.0 if name not in {"clearance_nmae", "geodesic_nmae"} else 0.0,
            floor_source="unit",
            ceiling_source="unit",
        )
        for name in (
            "occupied_iou",
            "boundary_f1",
            "clearance_nmae",
            "reachability_auprc",
            "geodesic_nmae",
        )
    }


def test_tiny_probe_benchmark_returns_per_geometry_rows_without_mutation() -> None:
    encoder = DenseResidualBackbone(
        stem_width=2,
        blocks_per_stage=(1, 1, 1, 1),
        embedding_dim=192,
        local_channels=16,
    )
    protocol = ProbeProtocol(
        coordinate_train_queries=32,
        coordinate_dev_queries=16,
        pair_train_queries=32,
        pair_dev_queries=16,
        intermediate_updates=1,
        final_updates=1,
        batch_size=16,
        hidden_dim=8,
    )
    result = evaluate_frozen_representation(
        encoder,
        _descriptors("train"),
        _descriptors("dev"),
        _anchors(),
        protocol=protocol,
        seed=0,
        device=torch.device("cpu"),
        final=False,
    )
    assert result["query_counts"] == {
        "coordinate_train": 32,
        "coordinate_dev": 16,
        "pair_train": 32,
        "pair_dev": 16,
    }
    assert len(result["real"]["per_geometry"]) == 4
    assert result["controls"] is None
    assert set(result["real"]["components"]) == {
        "occupied_iou",
        "boundary_f1",
        "clearance_nmae",
        "reachability_auprc",
        "geodesic_nmae",
    }
