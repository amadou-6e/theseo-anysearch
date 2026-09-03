"""Tests for real-data materialization used by amended P0C E1."""
from __future__ import annotations

from theseo_anysearch.garden.pilots.v2r1 import v2r1_geometry_records
from theseo_anysearch.garden.pilots.v2r1_data import (
    materialize_v2r1_calibration_datasets,
)


def test_small_real_data_bank_is_aligned_and_train_calibration_disjoint() -> None:
    records = v2r1_geometry_records()
    train = records[:2]
    calibration = records[12:14]
    datasets, reachability = materialize_v2r1_calibration_datasets(
        train,
        calibration,
        coordinate_train_queries=32,
        coordinate_evaluation_queries=32,
        pair_train_queries=20,
        pair_evaluation_queries=20,
        seed=3330,
    )
    assert set(datasets) == {
        "occupied_iou",
        "boundary_f1",
        "clearance_nmae",
        "reachability_auprc",
    }
    train_ids = {record.geometry_id for record in train}
    calibration_ids = {record.geometry_id for record in calibration}
    assert train_ids.isdisjoint(calibration_ids)
    for name, dataset in datasets.items():
        dataset.validate()
        assert set(dataset.evaluation_geometry_ids) == calibration_ids
        assert set(dataset.train_controls) == {
            "frequency",
            "coordinates_only",
            "pca",
            "fixed_random_projection",
        }
        if name == "reachability_auprc":
            assert set(dataset.evaluation_targets) == {0.0, 1.0}
        elif name != "clearance_nmae":
            assert set(dataset.evaluation_targets) == {0.0, 1.0}
    assert len(reachability.geometry_ids) == 20
    assert len(reachability.distance_bins) == 20
