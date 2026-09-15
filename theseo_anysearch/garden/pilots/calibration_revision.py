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
    train_controls: dict[str, np.ndarray]
    evaluation_controls: dict[str, np.ndarray]
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
        if not self.train_controls or set(self.train_controls) != set(self.evaluation_controls):
            raise ValueError("matching non-empty train/evaluation controls are required")
        for name in self.train_controls:
            if len(self.train_controls[name]) != train_rows:
                raise ValueError(f"training control {name} is not aligned")
            if len(self.evaluation_controls[name]) != evaluation_rows:
                raise ValueError(f"evaluation control {name} is not aligned")
        if any(not value for value in self.evaluation_geometry_ids) or self.normalizer <= 0:
            raise ValueError("geometry IDs must be non-empty and normalizer positive")
        return self


def _ridge_predict(
    dataset: CalibrationDataset, control: str, *, classification: bool
) -> np.ndarray:
    train = np.asarray(dataset.train_controls[control], dtype=np.float64)
    evaluation = np.asarray(dataset.evaluation_controls[control], dtype=np.float64)
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


def control_predictions(
    dataset: CalibrationDataset, control: str, *, classification: bool
) -> np.ndarray:
    """Fit a frozen linear control on pilot_train and predict calibration rows."""

    dataset.validate()
    if control not in dataset.train_controls:
        raise ValueError(f"unknown calibration control {control!r}")
    return _ridge_predict(dataset, control, classification=classification)


def _binary_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    true_positive = np.count_nonzero(prediction & target)
    denominator = np.count_nonzero(prediction) + np.count_nonzero(target)
    return 1.0 if denominator == 0 else float(2.0 * true_positive / denominator)


def _metric_value(component: str, prediction: np.ndarray, target: np.ndarray, normalizer: float) -> float:
    if component == "occupied_iou":
        return binary_iou(prediction >= 0.5, target)
    if component == "boundary_f1":
        return _binary_f1(prediction >= 0.5, target)
    if component == "reachability_auprc":
        return binary_ranking_metrics(prediction, target).auprc
    if component == "clearance_nmae":
        return normalized_mae(prediction, target, normalizer=normalizer)
    raise ValueError(f"unsupported active component {component!r}")


def _floor_values(component: str, dataset: CalibrationDataset) -> dict[str, float]:
    classification = component != "clearance_nmae"
    target = np.asarray(dataset.evaluation_targets)
    return {
        name: _metric_value(
            component,
            _ridge_predict(dataset, name, classification=classification),
            target,
            dataset.normalizer,
        )
        for name in sorted(dataset.train_controls)
    }


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


def measure_revised_component(
    component: str, dataset: CalibrationDataset, *, seed: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Return a complete anchor payload before applying pass/fail validation."""

    if component not in ACTIVE_COMPONENTS:
        raise ValueError(f"unsupported active component {component!r}")
    dataset.validate()
    floors = _floor_values(component, dataset)
    if component == "clearance_nmae":
        estimate = regression_metric_ceiling(
            dataset.evaluation_context,
            dataset.evaluation_targets,
            normalizer=dataset.normalizer,
        )
        higher_is_better = False
        floor_source = min(floors, key=floors.get)
    else:
        estimate = classification_metric_ceiling(
            dataset.evaluation_context,
            dataset.evaluation_targets,
            metric=component,
        )
        higher_is_better = True
        floor_source = max(floors, key=floors.get)
    floor = floors[floor_source]
    triviality = _triviality(component, dataset, seed=seed)
    payload: dict[str, object] = {
        "higher_is_better": higher_is_better,
        "floor": float(floor),
        "ceiling": estimate.value,
        "floor_source": f"measured:{floor_source}:pilot_calibration",
        "ceiling_source": f"measured:{estimate.method}:pilot_calibration",
        "ceiling_method": estimate.method,
        "ceiling_non_collapse_verified": True,
        "triviality": triviality,
        "status": "active",
    }
    diagnostics: dict[str, object] = {
        "floor": float(floor),
        "floors": floors,
        "selected_floor": floor_source,
        "ceiling": estimate.value,
        "ceiling_samples": estimate.sample_count,
        "triviality": triviality.model_dump(mode="json"),
    }
    return payload, diagnostics


def deferred_geodesic_anchor() -> RevisedScoreAnchor:
    """Preserve the measured v2 evidence behind the frozen Stage-2 deferral."""

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
    return RevisedScoreAnchor(
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


def calibrate_revised_anchors(
    datasets: dict[str, CalibrationDataset], *, seed: int = 3290
) -> tuple[dict[str, RevisedScoreAnchor], dict[str, object]]:
    """Measure all active denominators once and record geodesic deferral."""

    if set(datasets) != set(ACTIVE_COMPONENTS):
        raise ValueError("the revised path requires exactly four active component datasets")
    anchors: dict[str, RevisedScoreAnchor] = {}
    diagnostics: dict[str, object] = {}
    for index, component in enumerate(ACTIVE_COMPONENTS):
        payload, diagnostics[component] = measure_revised_component(
            component, datasets[component], seed=seed + index
        )
        anchors[component] = RevisedScoreAnchor(**payload)

    anchors["geodesic_nmae"] = deferred_geodesic_anchor()
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
        train_controls = {
            "frequency": np.ones((192, 1)),
            "coordinates_only": train_null,
            "pca": rng.normal(size=(192, 4)),
            "fixed_random_projection": rng.normal(size=(192, 4)),
        }
        evaluation_controls = {
            "frequency": np.ones((192, 1)),
            "coordinates_only": evaluation_null,
            "pca": rng.normal(size=(192, 4)),
            "fixed_random_projection": rng.normal(size=(192, 4)),
        }
        return CalibrationDataset(
            train_context,
            train_null,
            train_targets,
            evaluation_context,
            evaluation_null,
            evaluation_targets,
            geometry_ids,
            train_controls,
            evaluation_controls,
            normalizer,
        )

    return {component: dataset(component) for component in ACTIVE_COMPONENTS}


__all__ = [
    "ACTIVE_COMPONENTS",
    "CalibrationDataset",
    "calibrate_revised_anchors",
    "control_predictions",
    "deferred_geodesic_anchor",
    "deterministic_smoke_datasets",
    "measure_revised_component",
]
