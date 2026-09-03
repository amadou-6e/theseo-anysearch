"""Integrated, model-free calibration path for the v2r1 pilot amendment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theseo_anysearch.garden.evaluation.ceilings import (
    classification_metric_ceiling,
    regression_metric_ceiling,
)
from theseo_anysearch.garden.evaluation.metrics import (
    binary_iou,
    binary_ranking_metrics,
    normalized_mae,
)
from theseo_anysearch.garden.evaluation.triviality import assess_triviality
from theseo_anysearch.garden.pilots.contracts import RevisedScoreAnchor, TrivialityCheck


ACTIVE_COMPONENTS = (
    "occupied_iou",
    "boundary_f1",
    "clearance_nmae",
    "reachability_auprc",
)
GEODESIC_DEFERRAL = (
    "frequency NMAE 0.022515 is below the 0.15 pilot noise floor and the "
    "supervised reference did not improve it; deferred to Stage 2 wide context"
)


@dataclass(frozen=True)
class CalibrationDataset:
    """Train/evaluation arrays for one fixed revised probe component."""

    train_context: np.ndarray
    train_null: np.ndarray
    train_targets: np.ndarray
    evaluation_context: np.ndarray
    evaluation_null: np.ndarray
    evaluation_targets: np.ndarray
    evaluation_geometry_ids: tuple[str, ...]
    normalizer: float = 1.0

    def validate(self) -> "CalibrationDataset":
        train_rows = len(self.train_targets)
        evaluation_rows = len(self.evaluation_targets)
        if train_rows < 16 or evaluation_rows < 16:
            raise ValueError("calibration components require at least 16 train/evaluation rows")
        if len(self.train_context) != train_rows or len(self.train_null) != train_rows:
            raise ValueError("training features and targets must align")
        if (
            len(self.evaluation_context) != evaluation_rows
            or len(self.evaluation_null) != evaluation_rows
            or len(self.evaluation_geometry_ids) != evaluation_rows
        ):
            raise ValueError("evaluation features, targets, and geometry IDs must align")
        if any(not value for value in self.evaluation_geometry_ids) or self.normalizer <= 0:
            raise ValueError("geometry IDs must be non-empty and normalizer positive")
        return self


def _ridge_predict(dataset: CalibrationDataset, *, classification: bool) -> np.ndarray:
    train = np.asarray(dataset.train_null, dtype=np.float64)
    evaluation = np.asarray(dataset.evaluation_null, dtype=np.float64)
    targets = np.asarray(dataset.train_targets, dtype=np.float64)
    train = np.column_stack((np.ones(len(train)), train))
    evaluation = np.column_stack((np.ones(len(evaluation)), evaluation))
    ridge = 1e-3 * np.eye(train.shape[1])
    ridge[0, 0] = 0
    weights = np.linalg.solve(train.T @ train + ridge, train.T @ targets)
    prediction = evaluation @ weights
    if classification:
        prediction = 1.0 / (1.0 + np.exp(-np.clip(prediction, -30, 30)))
    return prediction


def _binary_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    true_positive = np.count_nonzero(prediction & target)
    denominator = np.count_nonzero(prediction) + np.count_nonzero(target)
    return 1.0 if denominator == 0 else float(2.0 * true_positive / denominator)


def _floor_value(component: str, dataset: CalibrationDataset) -> float:
    classification = component != "clearance_nmae"
    prediction = _ridge_predict(dataset, classification=classification)
    target = np.asarray(dataset.evaluation_targets)
    if component == "occupied_iou":
        return binary_iou(prediction >= 0.5, target)
    if component == "boundary_f1":
        return _binary_f1(prediction >= 0.5, target)
    if component == "reachability_auprc":
        return binary_ranking_metrics(prediction, target).auprc
    if component == "clearance_nmae":
        return normalized_mae(prediction, target, normalizer=dataset.normalizer)
    raise ValueError(f"unsupported active component {component!r}")


def _triviality(component: str, dataset: CalibrationDataset, *, seed: int) -> TrivialityCheck:
    evidence = assess_triviality(
        dataset.evaluation_context,
        dataset.evaluation_null,
        dataset.evaluation_targets,
        task_type="regression" if component == "clearance_nmae" else "binary",
        null_input="coordinates_only",
        min_pvi_gain=0.05,
        seed=seed,
    )
    return TrivialityCheck(
        null_input=evidence.null_input,
        pvi_embedding=evidence.pvi_embedding,
        pvi_null=evidence.pvi_null,
        pvi_gain=evidence.pvi_gain,
        mdl_embedding_bits=evidence.mdl_embedding_bits,
        mdl_null_bits=evidence.mdl_null_bits,
        min_pvi_gain=evidence.min_pvi_gain,
        passes=evidence.passes,
    )


def calibrate_revised_anchors(
    datasets: dict[str, CalibrationDataset], *, seed: int = 3290
) -> tuple[dict[str, RevisedScoreAnchor], dict[str, object]]:
    """Measure all active denominators once and record geodesic deferral."""

    if set(datasets) != set(ACTIVE_COMPONENTS):
        raise ValueError("the revised path requires exactly four active component datasets")
    anchors: dict[str, RevisedScoreAnchor] = {}
    diagnostics: dict[str, object] = {}
    for index, component in enumerate(ACTIVE_COMPONENTS):
        dataset = datasets[component].validate()
        floor = _floor_value(component, dataset)
        if component == "clearance_nmae":
            estimate = regression_metric_ceiling(
                dataset.evaluation_context,
                dataset.evaluation_targets,
                normalizer=dataset.normalizer,
            )
            higher_is_better = False
        else:
            estimate = classification_metric_ceiling(
                dataset.evaluation_context,
                dataset.evaluation_targets,
                metric=component,
            )
            higher_is_better = True
        triviality = _triviality(component, dataset, seed=seed + index)
        anchors[component] = RevisedScoreAnchor(
            higher_is_better=higher_is_better,
            floor=float(floor),
            ceiling=estimate.value,
            floor_source="measured:coordinates_only_ridge:pilot_calibration",
            ceiling_source=f"measured:{estimate.method}:pilot_calibration",
            ceiling_method=estimate.method,
            ceiling_non_collapse_verified=True,
            triviality=triviality,
            status="active",
        )
        diagnostics[component] = {
            "floor": float(floor),
            "ceiling": estimate.value,
            "ceiling_samples": estimate.sample_count,
            "triviality": triviality.model_dump(mode="json"),
        }

    deferred_triviality = TrivialityCheck(
        null_input="coordinates_only",
        pvi_embedding=0.0,
        pvi_null=0.0,
        pvi_gain=0.0,
        mdl_embedding_bits=1.0,
        mdl_null_bits=1.0,
        min_pvi_gain=0.05,
        passes=False,
    )
    anchors["geodesic_nmae"] = RevisedScoreAnchor(
        higher_is_better=False,
        floor=0.022515243950901476,
        ceiling=0.04816410017751241,
        floor_source="measured:frequency:voxel-encoder-pilot-v2-p0-calibration-1",
        ceiling_source="measured:supervised_reference:voxel-encoder-pilot-v2-p0-calibration-1",
        ceiling_method="regularized_reference",
        ceiling_non_collapse_verified=False,
        ceiling_effective_rank_fraction=0.010898,
        triviality=deferred_triviality,
        status="deferred",
        deferral_reason=GEODESIC_DEFERRAL,
    )
    diagnostics["geodesic_nmae"] = {"status": "deferred", "reason": GEODESIC_DEFERRAL}
    return anchors, diagnostics


def deterministic_smoke_datasets(*, seed: int = 3290) -> dict[str, CalibrationDataset]:
    """Construct an easy but nontrivial fixture for the complete CPU path."""

    rng = np.random.default_rng(seed)

    def dataset(component: str) -> CalibrationDataset:
        train_context = rng.normal(size=(192, 6))
        evaluation_context = rng.normal(size=(192, 6))
        train_null = rng.normal(size=(192, 3))
        evaluation_null = rng.normal(size=(192, 3))
        if component == "clearance_nmae":
            train_targets = 1.5 * train_context[:, 0] - 0.5 * train_context[:, 1]
            evaluation_targets = 1.5 * evaluation_context[:, 0] - 0.5 * evaluation_context[:, 1]
            normalizer = 4.0
        else:
            train_targets = (train_context[:, 0] + 0.4 * train_context[:, 1] > 0).astype(float)
            evaluation_targets = (
                evaluation_context[:, 0] + 0.4 * evaluation_context[:, 1] > 0
            ).astype(float)
            normalizer = 1.0
        geometry_ids = tuple(f"fixture-{index % 24:02d}" for index in range(192))
        return CalibrationDataset(
            train_context,
            train_null,
            train_targets,
            evaluation_context,
            evaluation_null,
            evaluation_targets,
            geometry_ids,
            normalizer,
        )

    return {component: dataset(component) for component in ACTIVE_COMPONENTS}


__all__ = [
    "ACTIVE_COMPONENTS",
    "CalibrationDataset",
    "calibrate_revised_anchors",
    "deterministic_smoke_datasets",
]
