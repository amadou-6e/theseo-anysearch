"""Objective-independent geometry, topology, and representation metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


SIX_CONNECTED = ndimage.generate_binary_structure(3, 1)


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.count_nonzero(prediction & target)
    union = np.count_nonzero(prediction | target)
    return 1.0 if union == 0 else float(intersection / union)


def macro_f1(prediction: np.ndarray, target: np.ndarray, *, classes: tuple[int, ...]) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    scores: list[float] = []
    for class_id in classes:
        predicted = prediction == class_id
        actual = target == class_id
        true_positive = np.count_nonzero(predicted & actual)
        denominator = 2 * true_positive + np.count_nonzero(predicted & ~actual) + np.count_nonzero(~predicted & actual)
        scores.append(1.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def boundary_f1(
    prediction: np.ndarray, target: np.ndarray, *, tolerance: int = 1
) -> float:
    """Symmetric boundary F1 with a Chebyshev-voxel tolerance."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target boundaries must share a shape")
    if tolerance < 0:
        raise ValueError("boundary tolerance cannot be negative")
    if not prediction.any() and not target.any():
        return 1.0
    structure = np.ones((3,) * prediction.ndim, dtype=bool)
    predicted_region = ndimage.binary_dilation(prediction, structure=structure, iterations=tolerance) if tolerance else prediction
    target_region = ndimage.binary_dilation(target, structure=structure, iterations=tolerance) if tolerance else target
    precision = np.count_nonzero(prediction & target_region) / max(1, np.count_nonzero(prediction))
    recall = np.count_nonzero(target & predicted_region) / max(1, np.count_nonzero(target))
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


@dataclass(frozen=True)
class BinaryRankingMetrics:
    auroc: float
    auprc: float
    balanced_accuracy: float
    false_open_rate: float
    false_closed_rate: float


def binary_ranking_metrics(
    scores: np.ndarray, target: np.ndarray, *, threshold: float = 0.5
) -> BinaryRankingMetrics:
    """Reachability ranking and asymmetric topology error metrics."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    if len(scores) != len(target) or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("binary scores and targets must be finite, non-empty, and aligned")
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC/AUPRC require both target classes")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_target = target[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    cumulative_true = np.cumsum(sorted_target)[group_ends]
    cumulative_false = (group_ends + 1) - cumulative_true
    true_positive_rate = np.r_[0.0, cumulative_true / positives, 1.0]
    false_positive_rate = np.r_[0.0, cumulative_false / negatives, 1.0]
    auroc = float(np.trapezoid(true_positive_rate, false_positive_rate))
    recall = cumulative_true / positives
    precision = cumulative_true / (group_ends + 1)
    auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    predicted = scores >= threshold
    true_positive_rate_at_threshold = np.count_nonzero(predicted & target) / positives
    true_negative_rate = np.count_nonzero(~predicted & ~target) / negatives
    return BinaryRankingMetrics(
        auroc=auroc,
        auprc=auprc,
        balanced_accuracy=float((true_positive_rate_at_threshold + true_negative_rate) / 2),
        false_open_rate=float(np.count_nonzero(predicted & ~target) / negatives),
        false_closed_rate=float(np.count_nonzero(~predicted & target) / positives),
    )


def normalized_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    normalizer: float,
    mask: np.ndarray | None = None,
) -> float:
    if normalizer <= 0:
        raise ValueError("MAE normalizer must be positive")
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("MAE prediction and target must share a shape")
    selected = np.ones(target.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != target.shape or not selected.any():
        raise ValueError("MAE mask must match the target and select at least one value")
    return float(np.mean(np.abs(prediction[selected] - target[selected])) / normalizer)


def variation_of_information(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    foreground_mask: np.ndarray | None = None,
) -> float:
    """Foreground-restricted VI normalized by log(number of evaluated cells)."""

    prediction = np.asarray(prediction_labels)
    target = np.asarray(target_labels)
    if prediction.shape != target.shape:
        raise ValueError("component label volumes must share a shape")
    mask = np.ones(prediction.shape, dtype=bool) if foreground_mask is None else np.asarray(foreground_mask, dtype=bool)
    if mask.shape != prediction.shape:
        raise ValueError("foreground mask must match component labels")
    predicted = prediction[mask]
    actual = target[mask]
    count = len(actual)
    if count <= 1:
        return 0.0
    _, predicted_inverse = np.unique(predicted, return_inverse=True)
    _, actual_inverse = np.unique(actual, return_inverse=True)
    contingency = np.zeros((predicted_inverse.max() + 1, actual_inverse.max() + 1), dtype=np.int64)
    np.add.at(contingency, (predicted_inverse, actual_inverse), 1)
    joint = contingency / count
    predicted_probability = joint.sum(axis=1, keepdims=True)
    actual_probability = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mutual_information = np.sum(
        joint[nonzero]
        * np.log(
            joint[nonzero]
            / (predicted_probability @ actual_probability)[nonzero]
        )
    )
    predicted_entropy = -np.sum(predicted_probability[predicted_probability > 0] * np.log(predicted_probability[predicted_probability > 0]))
    actual_entropy = -np.sum(actual_probability[actual_probability > 0] * np.log(actual_probability[actual_probability > 0]))
    return float((predicted_entropy + actual_entropy - 2 * mutual_information) / np.log(count))


def adapted_rand_error(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    foreground_mask: np.ndarray | None = None,
) -> float:
    """One minus the adapted Rand F-score over the evaluated foreground."""

    prediction = np.asarray(prediction_labels)
    target = np.asarray(target_labels)
    if prediction.shape != target.shape:
        raise ValueError("component label volumes must share a shape")
    mask = np.ones(prediction.shape, dtype=bool) if foreground_mask is None else np.asarray(foreground_mask, dtype=bool)
    if mask.shape != prediction.shape:
        raise ValueError("foreground mask must match component labels")
    predicted = prediction[mask]
    actual = target[mask]
    if len(actual) < 2:
        return 0.0
    _, predicted_inverse = np.unique(predicted, return_inverse=True)
    _, actual_inverse = np.unique(actual, return_inverse=True)
    contingency = np.zeros((predicted_inverse.max() + 1, actual_inverse.max() + 1), dtype=np.int64)
    np.add.at(contingency, (predicted_inverse, actual_inverse), 1)
    pairs = lambda values: np.sum(values * (values - 1))
    true_positive = float(pairs(contingency))
    predicted_pairs = float(pairs(contingency.sum(axis=1)))
    actual_pairs = float(pairs(contingency.sum(axis=0)))
    precision = true_positive / predicted_pairs if predicted_pairs else float(actual_pairs == 0)
    recall = true_positive / actual_pairs if actual_pairs else float(predicted_pairs == 0)
    f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(1 - f_score)


def _cubical_euler_characteristic(foreground: np.ndarray) -> int:
    cubes = np.argwhere(foreground)
    vertices: set[tuple[int, int, int]] = set()
    edges: set[tuple[int, int, int, int]] = set()
    faces: set[tuple[int, int, int, int]] = set()
    for x, y, z in cubes:
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    vertices.add((int(x + dx), int(y + dy), int(z + dz)))
        for axis in range(3):
            other = [candidate for candidate in range(3) if candidate != axis]
            for first in (0, 1):
                for second in (0, 1):
                    coordinate = [int(x), int(y), int(z)]
                    coordinate[other[0]] += first
                    coordinate[other[1]] += second
                    edges.add((axis, *coordinate))
            for offset in (0, 1):
                coordinate = [int(x), int(y), int(z)]
                coordinate[axis] += offset
                faces.add((axis, *coordinate))
    return len(vertices) - len(edges) + len(faces) - len(cubes)


def cubical_betti_numbers(foreground: np.ndarray) -> tuple[int, int, int]:
    """Exact beta-0/1/2 for a union of axis-aligned occupied unit cubes."""

    foreground = np.asarray(foreground, dtype=bool)
    if foreground.ndim != 3:
        raise ValueError("cubical Betti numbers require a 3D volume")
    beta_0 = int(ndimage.label(foreground, structure=SIX_CONNECTED)[1])
    padded_background = np.pad(~foreground, 1, constant_values=True)
    background_components = int(ndimage.label(padded_background, structure=SIX_CONNECTED)[1])
    beta_2 = max(0, background_components - 1)
    euler = _cubical_euler_characteristic(foreground)
    beta_1 = beta_0 + beta_2 - euler
    if beta_1 < 0:
        raise RuntimeError("invalid cubical-complex Betti calculation")
    return beta_0, int(beta_1), beta_2


def connectivity_change_fraction(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    pairs: np.ndarray,
) -> float:
    """Fraction of sampled cell pairs whose foreground connectivity changed."""

    prediction = np.asarray(prediction_labels).reshape(-1)
    target = np.asarray(target_labels).reshape(-1)
    pair_indices = np.asarray(pairs, dtype=np.int64)
    if prediction.shape != target.shape:
        raise ValueError("connectivity labels must share a shape")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2 or len(pair_indices) == 0:
        raise ValueError("connectivity pairs must have shape (P, 2)")
    if pair_indices.min() < 0 or pair_indices.max() >= len(target):
        raise ValueError("connectivity pair index is outside the label volume")
    left, right = pair_indices.T
    predicted_connected = (prediction[left] > 0) & (prediction[left] == prediction[right])
    target_connected = (target[left] > 0) & (target[left] == target[right])
    return float(np.mean(predicted_connected != target_connected))


@dataclass(frozen=True)
class TopologyReconstructionMetrics:
    component_count_error: int
    connectivity_change_fraction: float
    normalized_variation_of_information: float
    adapted_rand_error: float
    beta_0_error: int
    beta_1_error: int
    occupied_boundary_f1: float


def topology_reconstruction_metrics(
    predicted_occupancy: np.ndarray,
    target_occupancy: np.ndarray,
    *,
    valid_mask: np.ndarray,
    pairs: np.ndarray,
    boundary_tolerance: int = 1,
) -> TopologyReconstructionMetrics:
    """Compute the mandatory component/topology reconstruction bundle."""

    predicted_occupancy = np.asarray(predicted_occupancy, dtype=bool)
    target_occupancy = np.asarray(target_occupancy, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted_occupancy.shape != target_occupancy.shape or valid.shape != target_occupancy.shape:
        raise ValueError("occupancy and validity volumes must share a shape")
    predicted_free = valid & ~predicted_occupancy
    target_free = valid & ~target_occupancy
    predicted_labels, predicted_components = ndimage.label(predicted_free, structure=SIX_CONNECTED)
    target_labels, target_components = ndimage.label(target_free, structure=SIX_CONNECTED)
    union_foreground = predicted_free | target_free
    predicted_betti = cubical_betti_numbers(predicted_free)
    target_betti = cubical_betti_numbers(target_free)
    predicted_boundary = predicted_occupancy & ndimage.binary_dilation(
        ~predicted_occupancy, structure=SIX_CONNECTED
    )
    target_boundary = target_occupancy & ndimage.binary_dilation(
        ~target_occupancy, structure=SIX_CONNECTED
    )
    return TopologyReconstructionMetrics(
        component_count_error=abs(int(predicted_components) - int(target_components)),
        connectivity_change_fraction=connectivity_change_fraction(
            predicted_labels, target_labels, pairs
        ),
        normalized_variation_of_information=variation_of_information(
            predicted_labels, target_labels, foreground_mask=union_foreground
        ),
        adapted_rand_error=adapted_rand_error(
            predicted_labels, target_labels, foreground_mask=union_foreground
        ),
        beta_0_error=abs(predicted_betti[0] - target_betti[0]),
        beta_1_error=abs(predicted_betti[1] - target_betti[1]),
        occupied_boundary_f1=boundary_f1(
            predicted_boundary, target_boundary, tolerance=boundary_tolerance
        ),
    )


@dataclass(frozen=True)
class CollapseDiagnostics:
    effective_rank: float
    effective_rank_fraction: float
    near_dead_dimensions: int
    near_dead_fraction: float
    dominant_component_share: float
    alpha_req_exponent: float
    alpha_req_r_squared: float
    mean_off_diagonal_abs_correlation: float
    norm_mean: float
    norm_std: float
    singular_values: tuple[float, ...]


def collapse_diagnostics(embeddings: np.ndarray) -> CollapseDiagnostics:
    """Compute complete-spectrum RankMe and alpha-ReQ diagnostics."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("collapse diagnostics require an (N, D) matrix with N,D >= 2")
    centered = values - values.mean(axis=0, keepdims=True)
    standard_deviation = centered.std(axis=0, ddof=1)
    near_dead = int(np.count_nonzero(standard_deviation < 1e-3))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    spectral_mass = singular_values / max(np.finfo(float).eps, singular_values.sum())
    positive = spectral_mass > 0
    effective_rank = float(np.exp(-np.sum(spectral_mass[positive] * np.log(spectral_mass[positive]))))
    variance = singular_values**2
    dominant = float(variance[0] / variance.sum()) if variance.sum() else 1.0

    positive_values = singular_values[singular_values > np.finfo(float).eps]
    ranks = np.arange(1, len(positive_values) + 1, dtype=np.float64)
    if len(positive_values) >= 2:
        slope, intercept = np.polyfit(np.log(ranks), np.log(positive_values), 1)
        fitted = slope * np.log(ranks) + intercept
        residual = np.sum((np.log(positive_values) - fitted) ** 2)
        total = np.sum((np.log(positive_values) - np.log(positive_values).mean()) ** 2)
        alpha, r_squared = float(-slope), float(1 - residual / total) if total else 1.0
    else:
        alpha, r_squared = float("nan"), float("nan")

    active = centered[:, standard_deviation > 0]
    if active.shape[1] < 2:
        correlation = 0.0
    else:
        matrix = np.corrcoef(active, rowvar=False)
        correlation = float((np.abs(matrix).sum() - active.shape[1]) / (active.shape[1] * (active.shape[1] - 1)))
    norms = np.linalg.norm(values, axis=1)
    return CollapseDiagnostics(
        effective_rank=effective_rank,
        effective_rank_fraction=effective_rank / values.shape[1],
        near_dead_dimensions=near_dead,
        near_dead_fraction=near_dead / values.shape[1],
        dominant_component_share=dominant,
        alpha_req_exponent=alpha,
        alpha_req_r_squared=r_squared,
        mean_off_diagonal_abs_correlation=correlation,
        norm_mean=float(norms.mean()),
        norm_std=float(norms.std(ddof=1)),
        singular_values=tuple(float(value) for value in singular_values),
    )


def raw_and_l2_collapse_diagnostics(
    embeddings: np.ndarray,
) -> dict[str, CollapseDiagnostics]:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, np.finfo(float).eps)
    return {"raw": collapse_diagnostics(values), "l2_normalized": collapse_diagnostics(normalized)}


def linear_discriminant_rank(
    embeddings: np.ndarray, view_labels: np.ndarray, *, tolerance: float = 1e-6
) -> int:
    """LiDAR-style rank of whitened between-view scatter when views are meaningful."""

    values = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(view_labels).reshape(-1)
    if values.ndim != 2 or len(values) != len(labels) or len(values) < 2:
        raise ValueError("linear-discriminant rank requires aligned embeddings and labels")
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("linear-discriminant rank requires at least two view identities")
    global_mean = values.mean(axis=0)
    within = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    between = np.zeros_like(within)
    for label in unique:
        group = values[labels == label]
        group_centered = group - group.mean(axis=0)
        within += group_centered.T @ group_centered
        mean_delta = group.mean(axis=0) - global_mean
        between += len(group) * np.outer(mean_delta, mean_delta)
    within /= max(1, len(values) - len(unique))
    between /= max(1, len(unique) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(
        within + np.eye(values.shape[1]) * 1e-6
    )
    inverse_root = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12)))) @ eigenvectors.T
    discriminant = inverse_root @ between @ inverse_root
    spectrum = np.linalg.eigvalsh(discriminant)
    maximum = max(float(spectrum.max()), 0.0)
    return int(np.count_nonzero(spectrum > max(tolerance, maximum * tolerance)))
