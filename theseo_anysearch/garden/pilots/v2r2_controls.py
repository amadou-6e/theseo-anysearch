"""Observed-input R0 controls with geometry-disjoint, fixed-budget fitting.

This producer does not generate or preregister an evidential audit corpus. It can
also be exercised on explicitly non-evidential development fixtures.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from pydantic import Field
from scipy import ndimage

from theseo_anysearch.garden.pilots.contracts import FrozenModel
from theseo_anysearch.garden.pilots.io import payload_sha256
from theseo_anysearch.garden.pilots.v2r2_audit import (
    CONTROLS, STRATA, CounterfactualContext, PredictionFold,
    validate_counterfactual_arrays,
)


class R0ControlRecipe(FrozenModel):
    updates: int = Field(strict=True, ge=1)
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    weight_decay: float = Field(ge=0, allow_inf_nan=False)
    visible_hidden_width: int = Field(strict=True, ge=1)
    forbidden_hidden_width: int = Field(strict=True, ge=1)
    seed: int = Field(strict=True, ge=0)
    maximum_wall_seconds: float = Field(gt=0, allow_inf_nan=False)
    optimizer: Literal["adamw_full_batch"]
    checkpoint: Literal["last_update_no_evaluation_selection"]
    weighting: Literal["equal_geometry_then_equal_context"]
    geometric_prior: Literal["linear_observed_density_unknown_density_l1_separation"]


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(payload_sha256({"shape": list(array.shape), "dtype": array.dtype.str}).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class R0ControlExample:
    context_id: str
    geometry_id: str
    generator_configuration: str
    bootstrap_stratum: str
    stratum: str
    observed_occupancy: np.ndarray
    unknown: np.ndarray
    start: tuple[int, int, int]
    goal: tuple[int, int, int]
    labels: tuple[int, ...]
    forbidden_features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (
            self.context_id, self.geometry_id, self.generator_configuration, self.bootstrap_stratum,
        )) or self.stratum not in STRATA:
            raise ValueError("invalid example identity or stratum")
        observed, unknown = np.asarray(self.observed_occupancy), np.asarray(self.unknown)
        if observed.ndim != 3 or any(side < 2 for side in observed.shape):
            raise ValueError("observations must be genuinely three-dimensional")
        if observed.dtype != np.bool_ or unknown.dtype != np.bool_ or unknown.shape != observed.shape:
            raise ValueError("observed occupancy and unknown mask must be aligned boolean volumes")
        if not unknown.any() or unknown.all() or observed[unknown].any():
            raise ValueError("invalid mask or hidden occupancy exposed in observed input")
        for point in (self.start, self.goal):
            if len(point) != 3 or any(
                not isinstance(c, Integral) or isinstance(c, (bool, np.bool_))
                or c < 0 or c >= observed.shape[i] for i, c in enumerate(point)
            ):
                raise ValueError("query endpoints must be integral in-bounds coordinates")
            if observed[tuple(point)] or unknown[tuple(point)]:
                raise ValueError("query endpoints must be observed free cells")
        if self.start == self.goal:
            raise ValueError("reachability queries require distinct endpoints")
        if len(self.labels) < 2 or any(label not in (0, 1) for label in self.labels):
            raise ValueError("binary counterfactual labels are required")
        if not self.forbidden_features or not np.isfinite(self.forbidden_features).all():
            raise ValueError("finite forbidden features are required")
        object.__setattr__(self, "start", tuple(int(x) for x in self.start))
        object.__setattr__(self, "goal", tuple(int(x) for x in self.goal))
        object.__setattr__(self, "labels", tuple(int(x) for x in self.labels))
        object.__setattr__(self, "forbidden_features", tuple(float(x) for x in self.forbidden_features))
        for name, array in (("observed_occupancy", observed), ("unknown", unknown)):
            snapshot = array.copy()
            snapshot.setflags(write=False)
            object.__setattr__(self, name, snapshot)

    @property
    def observation_sha256(self) -> str:
        return payload_sha256({
            "occupancy": _array_sha(self.observed_occupancy), "unknown": _array_sha(self.unknown),
        })

    @property
    def visible_input_sha256(self) -> str:
        return payload_sha256({
            "observation": self.observation_sha256,
            "start": [int(x) for x in self.start], "goal": [int(x) for x in self.goal],
        })


def example_from_completions(
    *, context_id: str, geometry_id: str, generator_configuration: str,
    bootstrap_stratum: str, stratum: str, observed_occupancy: np.ndarray,
    unknown: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int],
    completions: np.ndarray, forbidden_features: tuple[float, ...],
) -> R0ControlExample:
    """Validate actual counterfactual voxels and derive six-connected labels.

    No completed occupancy, component ID or completion label becomes a feature.
    The caller still owns the preregistered, label-independent query/bin policy.
    """

    validate_counterfactual_arrays(observed_occupancy, unknown, completions)
    # Validate endpoints before indexing the completed fields.
    example = R0ControlExample(
        context_id, geometry_id, generator_configuration, bootstrap_stratum, stratum,
        observed_occupancy, unknown, start, goal, (0, 0), forbidden_features,
    )
    labels = []
    structure = ndimage.generate_binary_structure(3, 1)
    for completion in completions:
        components, _ = ndimage.label(~completion, structure=structure)
        labels.append(int(components[start] != 0 and components[start] == components[goal]))
    return R0ControlExample(
        context_id, geometry_id, generator_configuration, bootstrap_stratum, stratum,
        example.observed_occupancy, example.unknown, start, goal, tuple(labels), forbidden_features,
    )


def observed_control_features(example: R0ControlExample) -> dict[str, np.ndarray]:
    """An allowlist boundary: metadata and labels enter only the named ablation."""

    shape = np.asarray(example.observed_occupancy.shape) - 1
    coordinates = np.concatenate((np.asarray(example.start) / shape, np.asarray(example.goal) / shape))
    separation = np.abs(coordinates[:3] - coordinates[3:]).sum()
    return {
        "geometric_prior": np.asarray([example.observed_occupancy.mean(), example.unknown.mean(), separation], dtype=np.float32),
        "coordinates_only": coordinates.astype(np.float32),
        "visible_context": np.concatenate((
            example.observed_occupancy.ravel(), example.unknown.ravel(), coordinates,
        )).astype(np.float32),
        "forbidden": np.asarray(example.forbidden_features, dtype=np.float32),
    }


@dataclass(frozen=True)
class R0ControlFit:
    contexts: tuple[CounterfactualContext, ...]
    fold: PredictionFold
    artifact: dict[str, object]
    resources: dict[str, object]


def fit_r0_controls(
    training: tuple[R0ControlExample, ...], evaluation: tuple[R0ControlExample, ...],
    *, fold_id: str, domain: Literal["in_domain", "heldout_generator"],
    recipe: R0ControlRecipe, device: Literal["cpu", "cuda"],
) -> R0ControlFit:
    """Fit four controls, returning held-out probabilities and content hashes.

    Targets are per-context completion means. Full-batch weighted BCE is exactly
    the mean completion BCE without duplicating large voxel inputs on the GPU.
    Evaluation data never sets preprocessing, weights, budgets or checkpoints.
    """

    import torch
    from torch import nn

    started = time.perf_counter()
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be explicitly cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    if not training or not evaluation:
        raise ValueError("training and evaluation examples are required")
    training = tuple(sorted(training, key=lambda row: row.context_id))
    evaluation = tuple(sorted(evaluation, key=lambda row: row.context_id))
    all_rows = training + evaluation
    if len({row.context_id for row in all_rows}) != len(all_rows):
        raise ValueError("context identities must be unique across fitting and evaluation")
    if len({row.observed_occupancy.shape for row in all_rows}) != 1:
        raise ValueError("a control fit requires one fixed volume shape")
    if len({len(row.forbidden_features) for row in all_rows}) != 1:
        raise ValueError("forbidden feature widths must agree")
    train_ids = tuple(sorted({row.geometry_id for row in training}))
    evaluation_ids = tuple(sorted({row.geometry_id for row in evaluation}))
    configurations = tuple(sorted({row.generator_configuration for row in training}))
    fold = PredictionFold(
        fold_id=fold_id, training_geometry_ids=train_ids, evaluation_geometry_ids=evaluation_ids,
        training_generator_configurations=configurations, domain=domain, fit_artifact_sha256="0" * 64,
    )
    for row in evaluation:
        if (row.generator_configuration in configurations) != (domain == "in_domain"):
            raise ValueError("evaluation generator configuration disagrees with requested domain")
    owners: dict[str, str] = {}
    geometry_metadata: dict[str, tuple[str, str]] = {}
    for row in all_rows:
        if owners.setdefault(row.observation_sha256, row.geometry_id) != row.geometry_id:
            raise ValueError("identical visible contexts cross geometry groups")
        metadata = (row.generator_configuration, row.bootstrap_stratum)
        if geometry_metadata.setdefault(row.geometry_id, metadata) != metadata:
            raise ValueError("inconsistent sibling geometry metadata")

    def deadline() -> None:
        if time.perf_counter() - started > recipe.maximum_wall_seconds:
            raise TimeoutError("R0 control wall-clock cap exceeded; no completed fit artifact")

    features = [observed_control_features(row) for row in training]
    heldout = [observed_control_features(row) for row in evaluation]
    group_counts = {group: sum(row.geometry_id == group for row in training) for group in train_ids}
    weights_np = np.asarray([1 / (len(train_ids) * group_counts[row.geometry_id]) for row in training], dtype=np.float32)
    weights = torch.from_numpy(weights_np).to(device)
    targets = torch.tensor([np.mean(row.labels) for row in training], dtype=torch.float32, device=device)
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, object] = {}
    deadline()
    # Initialization occurs on CPU inside a restored RNG scope; fitting has no
    # dropout or randomized batches. No evaluation-driven selection is possible.
    with torch.random.fork_rng(devices=[]):
        for index, name in enumerate(CONTROLS):
            torch.random.default_generator.manual_seed(recipe.seed + index)
            x = np.stack([row[name] for row in features])
            mean = np.sum(x.astype(np.float64) * weights_np[:, None], axis=0)
            variance = np.sum((x - mean) ** 2 * weights_np[:, None], axis=0)
            scale = np.sqrt(variance)
            scale[scale < 1e-6] = 1
            normalized = ((x - mean) / scale).astype(np.float32)
            test = ((np.stack([row[name] for row in heldout]) - mean) / scale).astype(np.float32)
            if not np.isfinite(normalized).all() or not np.isfinite(test).all():
                raise ValueError("nonfinite standardized control features")
            x_tensor = torch.from_numpy(normalized).to(device)
            width = recipe.visible_hidden_width if name == "visible_context" else recipe.forbidden_hidden_width
            model = (
                nn.Linear(x.shape[1], 1) if name in ("geometric_prior", "coordinates_only")
                else nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(), nn.Linear(width, 1))
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
            model.train()
            for update in range(recipe.updates):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_tensor).squeeze(-1)
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).sum()
                loss.backward()
                optimizer.step()
                if update % 16 == 0 or update == recipe.updates - 1:
                    if not torch.isfinite(loss).item():
                        raise ValueError("nonfinite control training loss")
                    deadline()
            model.eval()
            with torch.no_grad():
                p = model(torch.from_numpy(test).to(device)).squeeze(-1).sigmoid().cpu().numpy()
            if not np.isfinite(p).all():
                raise ValueError("nonfinite held-out predictions")
            predictions[name] = p
            models[name] = {
                "mean_sha256": _array_sha(mean), "scale_sha256": _array_sha(scale),
                "state_sha256": payload_sha256({key: _array_sha(value.detach().cpu().numpy()) for key, value in model.state_dict().items()}),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "input_width": x.shape[1], "updates": recipe.updates,
            }
            deadline()
    artifact = {
        "kind": "r0_control_fit_v1", "recipe": recipe.model_dump(mode="json"),
        "training_geometry_ids": list(train_ids), "training_generator_configurations": list(configurations),
        "training_sha256": payload_sha256([
            {"context": row.context_id, "geometry": row.geometry_id,
             "visible": row.visible_input_sha256, "labels": list(row.labels),
             "forbidden_features": list(row.forbidden_features)} for row in training
        ]),
        "models": models, "device": device, "torch_version": str(torch.__version__),
        "forbidden_features_reach_visible_model": False,
        "authorizes_comparative_run": False,
    }
    fold = PredictionFold.model_validate({**fold.model_dump(), "fit_artifact_sha256": payload_sha256(artifact)})
    contexts = tuple(CounterfactualContext(
        context_id=row.context_id, geometry_id=row.geometry_id, fold_id=fold_id,
        generator_configuration=row.generator_configuration, bootstrap_stratum=row.bootstrap_stratum,
        stratum=row.stratum, visible_input_sha256=row.visible_input_sha256, labels=row.labels,
        observation_sha256=row.observation_sha256,
        unknown_fraction=float(row.unknown.mean()),
        **{name: float(predictions[name][index]) for name in CONTROLS},
    ) for index, row in enumerate(evaluation))
    elapsed = time.perf_counter() - started
    deadline()
    return R0ControlFit(contexts, fold, artifact, {
        "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "wall_seconds": elapsed,
        "accelerator_hours_upper_bound": elapsed / 3600 if device == "cuda" else 0.0,
        "candidate_training_updates": 0, "control_training_updates": 4 * recipe.updates,
    })
