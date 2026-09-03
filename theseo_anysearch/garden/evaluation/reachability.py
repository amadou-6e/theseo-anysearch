"""Reachability / false-open probe redesign (F4).

P0C showed the v1/v2 reachability probe was a label/threshold artifact: a fixed
random projection reached AUPRC 0.93 and false-open 0.148, and the
``false_open_rate > 0.05`` veto was unpassable by construction because random
pair sampling makes almost every negative trivially far apart.

This module provides the testable primitives for the redesign:

- ``sample_reachability_pairs`` stratifies positive pairs by geodesic distance,
  enforces a separation margin for component negatives, and adds
  obstacle-perturbation hard negatives (a true positive severed by a thin plug)
  and hard positives (a component negative joined by removing one voxel);
- ``per_bin_auprc`` reports AUPRC per geodesic-distance bin, not one saturated
  aggregate;
- ``calibrate_decision_threshold`` picks the operating point on a held-out fold
  (Youden's J) instead of a hard-coded 0.5;
- ``derive_false_open_veto`` sets the veto relative to the calibrated baseline
  distribution rather than an absolute 0.05;
- ``two_way_agreement`` keeps a "connected" call only when both directions agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from theseo_anysearch.garden.evaluation.metrics import binary_ranking_metrics

_SIX = ndimage.generate_binary_structure(3, 1)
DEFAULT_BINS = 5
DEFAULT_COMPONENT_MARGIN = 4.0
DEFAULT_BOUNDARY_FRACTION = 0.30


@dataclass(frozen=True)
class PairPlan:
    """Stratified reachability pairs with per-pair provenance."""

    starts: np.ndarray  # (N, 3) int
    goals: np.ndarray  # (N, 3) int
    reachable: np.ndarray  # (N,) bool
    distance_bin: np.ndarray  # (N,) int, -1 for unreachable
    kind: np.ndarray  # (N,) str
    perturbation_coordinates: np.ndarray  # (N, 3), -1 when unmodified
    perturbation_occupied: np.ndarray  # (N,), -1 unmodified, 0 open, 1 plug

    def __len__(self) -> int:
        return int(self.starts.shape[0])


@dataclass(frozen=True)
class FalseOpenVeto:
    """False-open veto derived from the calibrated baseline distribution."""

    threshold: float
    best_baseline: str
    best_baseline_false_open: float
    margin: float
    method: str = "best_baseline_minus_margin"

    def rejects(self, candidate_false_open: float) -> bool:
        return candidate_false_open >= self.threshold


def _free_mask(occupancy: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.asarray(valid_mask, dtype=bool) & ~np.asarray(occupancy, dtype=bool)


def geodesic_distance_bins(max_distance: float, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """Return ``n_bins + 1`` edges spanning ``(0, max_distance]`` in equal steps."""

    if max_distance <= 0 or n_bins < 1:
        raise ValueError("max_distance must be positive and n_bins >= 1")
    return np.linspace(0.0, float(max_distance), n_bins + 1)


def _bin_of(distance: float, edges: np.ndarray) -> int:
    if not np.isfinite(distance) or distance <= 0:
        return -1
    index = int(np.searchsorted(edges, distance, side="left")) - 1
    return int(np.clip(index, 0, len(edges) - 2))


def _severs_pair(
    free: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int]
) -> np.ndarray | None:
    """Return an obstacle plug (mask) that disconnects start from goal, or None.

    Tries every free neighbour of ``start`` as a one-voxel plug; accepts the
    first that leaves start and goal in different components.
    """

    for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        plug = tuple(np.array(start) + offset)
        if any(c < 0 or c >= free.shape[i] for i, c in enumerate(plug)):
            continue
        if not free[plug] or plug == goal:
            continue
        trial = free.copy()
        trial[plug] = False
        labels, _ = ndimage.label(trial, structure=_SIX)
        if labels[start] != labels[goal] or labels[goal] == 0:
            mask = np.zeros_like(free)
            mask[plug] = True
            return mask
    return None


def _joins_pair(
    free: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int]
) -> np.ndarray | None:
    """Return a one-voxel opening that connects start to goal, or None."""

    for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        opening = tuple(np.array(start) + offset)
        if any(c < 0 or c >= free.shape[i] for i, c in enumerate(opening)):
            continue
        if free[opening]:
            continue
        trial = free.copy()
        trial[opening] = True
        labels, _ = ndimage.label(trial, structure=_SIX)
        if labels[start] != 0 and labels[start] == labels[goal]:
            mask = np.zeros_like(free)
            mask[opening] = True
            return mask
    return None


def sample_reachability_pairs(
    occupancy: np.ndarray,
    valid_mask: np.ndarray,
    free_component_labels: np.ndarray,
    *,
    count: int,
    seed: int,
    n_bins: int = DEFAULT_BINS,
    component_margin: float = DEFAULT_COMPONENT_MARGIN,
    boundary_fraction: float = DEFAULT_BOUNDARY_FRACTION,
) -> PairPlan:
    """Sample a stratified, margin-controlled reachability pair set."""

    rng = np.random.default_rng(seed)
    free = _free_mask(occupancy, valid_mask)
    labels = np.asarray(free_component_labels, dtype=np.int64)
    components = [value for value in np.unique(labels) if value > 0]
    if not components:
        raise ValueError("observation has no free component")
    diameter = float(np.linalg.norm(free.shape))
    edges = geodesic_distance_bins(diameter, n_bins)

    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    reachable: list[bool] = []
    distance_bin: list[int] = []
    kind: list[str] = []
    perturbation_coordinates: list[np.ndarray] = []
    perturbation_occupied: list[int] = []

    def append(
        start: np.ndarray | tuple[int, int, int],
        goal: np.ndarray | tuple[int, int, int],
        is_reachable: bool,
        bin_index: int,
        pair_kind: str,
        perturbation: np.ndarray | None = None,
        occupied_value: int = -1,
    ) -> None:
        starts.append(np.asarray(start, dtype=np.int64))
        goals.append(np.asarray(goal, dtype=np.int64))
        reachable.append(is_reachable)
        distance_bin.append(bin_index)
        kind.append(pair_kind)
        coordinate = (
            np.full(3, -1, dtype=np.int64)
            if perturbation is None
            else np.argwhere(perturbation)[0].astype(np.int64)
        )
        perturbation_coordinates.append(coordinate)
        perturbation_occupied.append(occupied_value)

    n_boundary = int(round(count * boundary_fraction))
    n_component_neg = (count - n_boundary) // 3
    n_positive = count - n_boundary - n_component_neg

    # --- geodesic-distance-stratified positives --------------------------------
    per_bin = max(1, n_positive // n_bins)
    from theseo_anysearch.garden.targets import geodesic_distances as _gd  # local dep

    class _T:  # geodesic_distances only needs these three attributes
        def __init__(self) -> None:
            self.occupancy = np.asarray(occupancy, dtype=np.uint8)
            self.valid_mask = np.asarray(valid_mask, dtype=bool)

    targets = _T()
    bin_targets = {b: per_bin for b in range(n_bins)}
    attempts = 0
    while sum(bin_targets.values()) > 0 and attempts < 40 * n_positive:
        attempts += 1
        component = components[int(rng.integers(0, len(components)))]
        cells = np.argwhere(labels == component)
        if len(cells) < 2:
            continue
        start = tuple(int(v) for v in cells[int(rng.integers(0, len(cells)))])
        field = _gd(targets, start)
        candidates = np.argwhere(np.isfinite(field) & (field > 0))
        if not len(candidates):
            continue
        goal = tuple(int(v) for v in candidates[int(rng.integers(0, len(candidates)))])
        b = _bin_of(float(field[goal]), edges)
        if b < 0 or bin_targets.get(b, 0) <= 0:
            continue
        bin_targets[b] -= 1
        append(start, goal, True, b, "stratified_positive")

    # --- component negatives with a separation margin ------------------------
    if len(components) >= 2:
        made = 0
        guard = 0
        while made < n_component_neg and guard < 60 * max(1, n_component_neg):
            guard += 1
            first, second = rng.choice(len(components), size=2, replace=False)
            a = np.argwhere(labels == components[first])
            b_cells = np.argwhere(labels == components[second])
            start = a[int(rng.integers(0, len(a)))]
            goal = b_cells[int(rng.integers(0, len(b_cells)))]
            if np.linalg.norm(start - goal) < component_margin:
                continue
            append(start, goal, False, -1, "component_negative")
            made += 1

    # --- obstacle-perturbation hard negatives and positives ------------------
    made = 0
    guard = 0
    positives = [i for i, k in enumerate(kind) if k == "stratified_positive"]
    while made < n_boundary and guard < 80 * max(1, n_boundary) and positives:
        guard += 1
        idx = positives[int(rng.integers(0, len(positives)))]
        start = tuple(int(v) for v in starts[idx])
        goal = tuple(int(v) for v in goals[idx])
        plug = _severs_pair(free, start, goal)
        if plug is None:
            continue
        append(start, goal, False, -1, "boundary_negative", plug, 1)
        made += 1

    if len(components) >= 2:
        guard = 0
        want_positive = n_boundary - made
        while made < n_boundary and guard < 80 * max(1, want_positive):
            guard += 1
            first, second = rng.choice(len(components), size=2, replace=False)
            a = np.argwhere(labels == components[first])
            b_cells = np.argwhere(labels == components[second])
            start = tuple(int(v) for v in a[int(rng.integers(0, len(a)))])
            goal = tuple(int(v) for v in b_cells[int(rng.integers(0, len(b_cells)))])
            if np.linalg.norm(np.array(start) - np.array(goal)) > component_margin:
                continue
            opening_mask = _joins_pair(free, start, goal)
            if opening_mask is None:
                continue
            append(start, goal, True, 0, "boundary_positive", opening_mask, 0)
            made += 1

    if not starts:
        raise ValueError("could not sample any reachability pairs for this observation")
    return PairPlan(
        starts=np.stack(starts).astype(np.int64),
        goals=np.stack(goals).astype(np.int64),
        reachable=np.asarray(reachable, dtype=bool),
        distance_bin=np.asarray(distance_bin, dtype=np.int64),
        kind=np.asarray(kind, dtype=object),
        perturbation_coordinates=np.stack(perturbation_coordinates).astype(np.int64),
        perturbation_occupied=np.asarray(perturbation_occupied, dtype=np.int8),
    )


def per_bin_auprc(
    scores: np.ndarray, reachable: np.ndarray, distance_bin: np.ndarray
) -> dict[str, float]:
    """AUPRC overall and within each positive geodesic-distance bin."""

    scores = np.asarray(scores, dtype=np.float64)
    reachable = np.asarray(reachable, dtype=bool)
    distance_bin = np.asarray(distance_bin, dtype=np.int64)
    out: dict[str, float] = {}
    if reachable.any() and (~reachable).any():
        out["overall"] = binary_ranking_metrics(scores, reachable).auprc
    negatives = ~reachable
    for b in sorted({int(v) for v in distance_bin if v >= 0}):
        selection = negatives | (distance_bin == b)
        subset_pos = reachable[selection]
        if subset_pos.any() and (~subset_pos).any():
            out[f"bin_{b}"] = binary_ranking_metrics(
                scores[selection], reachable[selection]
            ).auprc
    return out


def calibrate_decision_threshold(
    scores: np.ndarray, reachable: np.ndarray
) -> float:
    """Youden's J optimal threshold on a held-out fold."""

    scores = np.asarray(scores, dtype=np.float64)
    reachable = np.asarray(reachable, dtype=bool)
    if not reachable.any() or not (~reachable).any():
        raise ValueError("threshold calibration needs both classes")
    candidates = np.unique(scores)
    best_threshold, best_j = 0.5, -np.inf
    for threshold in candidates:
        predicted = scores >= threshold
        tpr = np.count_nonzero(predicted & reachable) / reachable.sum()
        fpr = np.count_nonzero(predicted & ~reachable) / (~reachable).sum()
        if tpr - fpr > best_j:
            best_j, best_threshold = tpr - fpr, float(threshold)
    return best_threshold


def false_open_false_closed(
    scores: np.ndarray, reachable: np.ndarray, threshold: float
) -> tuple[float, float]:
    """False-open (predict connected when blocked) and false-closed rates."""

    scores = np.asarray(scores, dtype=np.float64)
    reachable = np.asarray(reachable, dtype=bool)
    predicted = scores >= threshold
    negatives = ~reachable
    false_open = (
        np.count_nonzero(predicted & negatives) / negatives.sum()
        if negatives.any()
        else 0.0
    )
    false_closed = (
        np.count_nonzero(~predicted & reachable) / reachable.sum()
        if reachable.any()
        else 0.0
    )
    return float(false_open), float(false_closed)


def two_way_agreement(
    scores_forward: np.ndarray, scores_backward: np.ndarray, threshold: float
) -> np.ndarray:
    """A pair is called connected only when both directions cross the threshold."""

    forward = np.asarray(scores_forward, dtype=np.float64) >= threshold
    backward = np.asarray(scores_backward, dtype=np.float64) >= threshold
    return forward & backward


def derive_false_open_veto(
    baseline_false_open: dict[str, float],
    *,
    margin: float = 0.02,
) -> FalseOpenVeto:
    """Set the false-open veto below the best non-neural baseline.

    A learned encoder must drive false-open *below* what a fixed random
    projection / PCA achieves; an absolute 0.05 was unpassable in P0C.
    """

    if not baseline_false_open:
        raise ValueError("at least one baseline false-open rate is required")
    best_baseline = min(baseline_false_open, key=baseline_false_open.get)
    best_value = float(baseline_false_open[best_baseline])
    threshold = max(0.0, best_value - margin)
    return FalseOpenVeto(
        threshold=threshold,
        best_baseline=best_baseline,
        best_baseline_false_open=best_value,
        margin=float(margin),
    )
