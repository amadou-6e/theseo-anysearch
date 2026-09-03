"""Fixed frozen-probe benchmark used by comparative perception pilots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from theseo_anysearch.garden.evaluation.metrics import (
    binary_iou,
    binary_ranking_metrics,
    boundary_f1,
    collapse_diagnostics,
    normalized_mae,
)
from theseo_anysearch.garden.evaluation.probes import encoder_state_sha256
from theseo_anysearch.garden.evaluation.statistics import interquartile_mean, performance_profile
from theseo_anysearch.garden.masking import DenseMaskAwareEncoder
from theseo_anysearch.garden.models.outputs import EncoderOutput, VoxelLevel
from theseo_anysearch.garden.pilots.contracts import RevisedScoreAnchor, ScoreAnchor
from theseo_anysearch.garden.pilots.corpus import V1_PROGRAM, make_pilot_observation
from theseo_anysearch.garden.splits import GeometryDescriptor
from theseo_anysearch.garden.targets import (
    compute_geometry_targets,
    geodesic_distances,
)


COMPONENTS = (
    "occupied_iou",
    "boundary_f1",
    "clearance_nmae",
    "reachability_auprc",
    "geodesic_nmae",
)
REVISED_PROFILE_THRESHOLDS = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0)
CALIBRATION_TEMPLATE_BARS = {
    "boundary_f1": {"minimum_absolute_headroom": 0.10},
    "clearance_nmae": {"minimum_relative_error_reduction": 0.20},
}


@dataclass(frozen=True)
class ProbeProtocol:
    coordinate_train_queries: int = 100_000
    coordinate_dev_queries: int = 20_000
    pair_train_queries: int = 50_000
    pair_dev_queries: int = 10_000
    intermediate_updates: int = 200
    final_updates: int = 500
    batch_size: int = 1_024
    learning_rate: float = 1e-3
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        values = (
            self.coordinate_train_queries,
            self.coordinate_dev_queries,
            self.pair_train_queries,
            self.pair_dev_queries,
            self.intermediate_updates,
            self.final_updates,
            self.batch_size,
            self.hidden_dim,
        )
        if any(value < 1 for value in values) or self.learning_rate <= 0:
            raise ValueError("probe protocol counts and learning rate must be positive")
        if self.final_updates > 5_000:
            raise ValueError("pilot probes cannot exceed 5,000 updates")


@dataclass(frozen=True)
class _ProbeBank:
    features: torch.Tensor
    targets: torch.Tensor
    learned_dimensions: int
    geometry_ids: tuple[str, ...]
    families: tuple[str, ...]
    occupancy_bands: tuple[str, ...]
    embeddings: torch.Tensor


def _encode(
    encoder: nn.Module,
    occupancy: np.ndarray,
    unknown: np.ndarray,
    device: torch.device,
) -> EncoderOutput:
    level = VoxelLevel.from_occupancy(
        torch.from_numpy(occupancy[None]).to(device=device, dtype=torch.float32),
        unknown_mask=torch.from_numpy(unknown[None]).to(device=device),
    )
    hidden = torch.zeros_like(level.validity_mask)
    output = (
        encoder(level, hidden)
        if isinstance(encoder, DenseMaskAwareEncoder)
        else encoder(level)
    )
    if not isinstance(output, EncoderOutput):
        raise TypeError("frozen benchmark requires EncoderOutput-compatible encoders")
    return output.validate(embedding_dim=192)


def _allocated_counts(total: int, groups: int) -> list[int]:
    quotient, remainder = divmod(total, groups)
    return [quotient + (index < remainder) for index in range(groups)]


def _coordinate_rows(
    output: EncoderOutput,
    targets: object,
    unknown: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    occupancy = np.asarray(targets.occupancy, dtype=bool)
    boundary = np.asarray(targets.boundary, dtype=bool)
    valid = np.asarray(targets.valid_mask, dtype=bool)
    categories = (
        np.argwhere(boundary),
        np.argwhere(occupancy & ~boundary),
        np.argwhere(valid & ~occupancy),
        np.argwhere(unknown),
    )
    fallback = np.argwhere(np.ones_like(occupancy, dtype=bool))
    coordinates: list[np.ndarray] = []
    for index in range(count):
        available = categories[index % len(categories)]
        if len(available) == 0:
            available = fallback
        coordinates.append(available[int(rng.integers(0, len(available)))])
    xyz = np.stack(coordinates).astype(np.int64)
    side = occupancy.shape[0]
    normalized = xyz.astype(np.float32) * (2.0 / (side - 1)) - 1.0
    local = output.local_feature_volume[
        0,
        :,
        torch.from_numpy(xyz[:, 0]).to(output.local_feature_volume.device),
        torch.from_numpy(xyz[:, 1]).to(output.local_feature_volume.device),
        torch.from_numpy(xyz[:, 2]).to(output.local_feature_volume.device),
    ].T
    global_features = output.global_embedding.expand(count, -1)
    metadata = torch.ones(count, 2, device=local.device)
    features = torch.cat(
        (
            local,
            global_features,
            torch.from_numpy(normalized).to(local.device),
            metadata,
        ),
        dim=1,
    )
    truncation = max(1.0, (side // 2) * 0.25)
    target = np.column_stack(
        (
            occupancy[tuple(xyz.T)].astype(np.float32),
            boundary[tuple(xyz.T)].astype(np.float32),
            np.asarray(targets.signed_distance)[tuple(xyz.T)] / truncation,
        )
    ).astype(np.float32)
    return features.detach().cpu(), torch.from_numpy(target)


def _sample_pair_coordinates(
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = [value for value in np.unique(labels) if value > 0]
    cells = {value: np.argwhere(labels == value) for value in components}
    if not components:
        raise ValueError("pair probe observation has no valid free component")
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    reachable: list[bool] = []
    for index in range(count):
        use_reachable = index % 2 == 0 or len(components) == 1
        first = components[index % len(components)]
        second = first if use_reachable else components[(index + 1) % len(components)]
        first_cells = cells[first]
        second_cells = cells[second]
        starts.append(first_cells[int(rng.integers(0, min(len(first_cells), 8)))])
        goals.append(second_cells[int(rng.integers(0, len(second_cells)))])
        reachable.append(use_reachable)
    return np.stack(starts), np.stack(goals), np.asarray(reachable, dtype=bool)


def _pair_rows(
    output: EncoderOutput,
    targets: object,
    *,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    starts, goals, reachable = _sample_pair_coordinates(
        np.asarray(targets.free_component_labels), count, rng
    )
    device = output.local_feature_volume.device

    def local_at(coordinates: np.ndarray) -> torch.Tensor:
        return output.local_feature_volume[
            0,
            :,
            torch.from_numpy(coordinates[:, 0]).to(device),
            torch.from_numpy(coordinates[:, 1]).to(device),
            torch.from_numpy(coordinates[:, 2]).to(device),
        ].T

    start_features = local_at(starts)
    goal_features = local_at(goals)
    side = np.asarray(targets.occupancy).shape[0]
    start_normalized = starts.astype(np.float32) * (2.0 / (side - 1)) - 1.0
    goal_normalized = goals.astype(np.float32) * (2.0 / (side - 1)) - 1.0
    features = torch.cat(
        (
            start_features + goal_features,
            torch.abs(start_features - goal_features),
            start_features * goal_features,
            output.global_embedding.expand(count, -1),
            torch.from_numpy(start_normalized + goal_normalized).to(device),
            torch.from_numpy(np.abs(start_normalized - goal_normalized)).to(device),
            torch.ones(count, 2, device=device),
        ),
        dim=1,
    )
    distances = np.zeros(count, dtype=np.float32)
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for index, (start, goal) in enumerate(zip(starts, goals)):
        if not reachable[index]:
            continue
        key = tuple(int(value) for value in start)
        if key not in cache:
            cache[key] = geodesic_distances(targets, key)
        distances[index] = cache[key][tuple(int(value) for value in goal)]
    distances /= max(1, 3 * (side - 1))
    target = torch.from_numpy(np.column_stack((reachable, distances)).astype(np.float32))
    return features.detach().cpu(), target


def _make_banks(
    encoder: nn.Module,
    descriptors: Sequence[GeometryDescriptor],
    *,
    coordinate_queries: int,
    pair_queries: int,
    seed: int,
    device: torch.device,
    corpus_program: str,
) -> tuple[_ProbeBank, _ProbeBank]:
    encoder_was_training = encoder.training
    encoder.eval()
    coordinate_counts = _allocated_counts(coordinate_queries, len(descriptors))
    pair_counts = _allocated_counts(pair_queries, len(descriptors))
    coordinate_features: list[torch.Tensor] = []
    coordinate_targets: list[torch.Tensor] = []
    pair_features: list[torch.Tensor] = []
    pair_targets: list[torch.Tensor] = []
    coordinate_metadata: list[tuple[str, str, str]] = []
    pair_metadata: list[tuple[str, str, str]] = []
    embeddings: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for index, descriptor in enumerate(descriptors):
                radius = 8 if index % 2 == 0 else 16
                coordinate_observation = make_pilot_observation(
                    descriptor, 1, radius=radius, program=corpus_program
                )
                coordinate_output = _encode(
                    encoder,
                    coordinate_observation.occupancy,
                    coordinate_observation.unknown_mask,
                    device,
                )
                coordinate_geometry = compute_geometry_targets(
                    coordinate_observation.occupancy,
                    unknown_mask=coordinate_observation.unknown_mask,
                    truncation=radius * 0.25,
                )
                features, targets = _coordinate_rows(
                    coordinate_output,
                    coordinate_geometry,
                    coordinate_observation.unknown_mask,
                    count=coordinate_counts[index],
                    seed=seed * 10_000 + index,
                )
                coordinate_features.append(features)
                coordinate_targets.append(targets)
                coordinate_metadata.extend(
                    [(descriptor.geometry_id, descriptor.family, descriptor.occupancy_band)]
                    * len(features)
                )
                embeddings.append(coordinate_output.global_embedding.cpu())

                pair_observation_index = 5 + 10 * (index % 3)
                pair_observation = make_pilot_observation(
                    descriptor,
                    pair_observation_index,
                    radius=radius,
                    program=corpus_program,
                )
                pair_output = _encode(
                    encoder,
                    pair_observation.occupancy,
                    pair_observation.unknown_mask,
                    device,
                )
                pair_geometry = compute_geometry_targets(
                    pair_observation.occupancy,
                    unknown_mask=pair_observation.unknown_mask,
                    truncation=radius * 0.25,
                )
                features, targets = _pair_rows(
                    pair_output,
                    pair_geometry,
                    count=pair_counts[index],
                    seed=seed * 20_000 + index,
                )
                pair_features.append(features)
                pair_targets.append(targets)
                pair_metadata.extend(
                    [(descriptor.geometry_id, descriptor.family, descriptor.occupancy_band)]
                    * len(features)
                )
                embeddings.append(pair_output.global_embedding.cpu())

                rank_observation = make_pilot_observation(
                    descriptor, 2, radius=radius, program=corpus_program
                )
                rank_output = _encode(
                    encoder,
                    rank_observation.occupancy,
                    rank_observation.unknown_mask,
                    device,
                )
                embeddings.append(rank_output.global_embedding.cpu())
    finally:
        encoder.train(encoder_was_training)

    def bank(
        features: list[torch.Tensor],
        targets: list[torch.Tensor],
        metadata: list[tuple[str, str, str]],
    ) -> _ProbeBank:
        geometry_ids, families, bands = zip(*metadata)
        combined_features = torch.cat(features)
        return _ProbeBank(
            features=combined_features,
            targets=torch.cat(targets),
            learned_dimensions=combined_features.shape[1] - (
                5 if targets[0].shape[1] == 3 else 8
            ),
            geometry_ids=tuple(geometry_ids),
            families=tuple(families),
            occupancy_bands=tuple(bands),
            embeddings=torch.cat(embeddings),
        )

    return (
        bank(coordinate_features, coordinate_targets, coordinate_metadata),
        bank(pair_features, pair_targets, pair_metadata),
    )


class _ProbeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _fit_probe(
    train: _ProbeBank,
    dev: _ProbeBank,
    *,
    task: str,
    protocol: ProbeProtocol,
    updates: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(seed)
    head = _ProbeHead(
        train.features.shape[1], protocol.hidden_dim, train.targets.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=protocol.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    positive_weights = []
    classification_columns = (0, 1) if task == "coordinate" else (0,)
    for column in classification_columns:
        positives = train.targets[:, column].sum().clamp_min(1)
        negatives = len(train.targets) - positives
        positive_weights.append((negatives / positives).clamp(max=100))
    for _ in range(updates):
        indices = torch.randint(
            len(train.features),
            (min(protocol.batch_size, len(train.features)),),
            generator=generator,
        )
        features = train.features[indices].to(device)
        target = train.targets[indices].to(device)
        prediction = head(features)
        if task == "coordinate":
            loss = F.binary_cross_entropy_with_logits(
                prediction[:, 0], target[:, 0], pos_weight=positive_weights[0].to(device)
            )
            loss = loss + F.binary_cross_entropy_with_logits(
                prediction[:, 1], target[:, 1], pos_weight=positive_weights[1].to(device)
            )
            loss = loss + F.smooth_l1_loss(prediction[:, 2], target[:, 2])
        else:
            loss = F.binary_cross_entropy_with_logits(
                prediction[:, 0], target[:, 0], pos_weight=positive_weights[0].to(device)
            )
            reachable = target[:, 0].bool()
            loss = loss + F.smooth_l1_loss(
                prediction[reachable, 1], target[reachable, 1]
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    predictions: list[torch.Tensor] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(dev.features), 8_192):
            predictions.append(head(dev.features[start : start + 8_192].to(device)).cpu())
    return torch.cat(predictions).numpy()


def _transform_bank(bank: _ProbeBank, mode: str, *, seed: int) -> _ProbeBank:
    features = bank.features.clone()
    learned_dimensions = bank.learned_dimensions
    targets = bank.targets.clone()
    if mode == "zero":
        features[:, :learned_dimensions] = 0
    elif mode == "shuffle":
        counts: dict[str, list[int]] = {}
        for index, geometry_id in enumerate(bank.geometry_ids):
            counts.setdefault(geometry_id, []).append(index)
        geometry_groups = list(counts.values())
        permutation = list(range(len(features)))
        for index, destination in enumerate(geometry_groups):
            source = geometry_groups[(index + 1) % len(geometry_groups)]
            for offset, destination_row in enumerate(destination):
                permutation[destination_row] = source[offset % len(source)]
        features[:, :learned_dimensions] = features[permutation, :learned_dimensions]
    elif mode == "control_target":
        groups: dict[tuple[str, str], list[int]] = {}
        for index, key in enumerate(zip(bank.families, bank.occupancy_bands)):
            groups.setdefault(key, []).append(index)
        generator = torch.Generator().manual_seed(seed)
        for indices in groups.values():
            order = torch.randperm(len(indices), generator=generator)
            targets[indices] = targets[torch.tensor(indices)[order]]
    elif mode != "real":
        raise ValueError(mode)
    return _ProbeBank(
        features,
        targets,
        learned_dimensions,
        bank.geometry_ids,
        bank.families,
        bank.occupancy_bands,
        bank.embeddings,
    )


def _metric_rows(
    coordinate_bank: _ProbeBank,
    coordinate_prediction: np.ndarray,
    pair_bank: _ProbeBank,
    pair_prediction: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    geometry_ids = sorted(set(coordinate_bank.geometry_ids))
    for geometry_id in geometry_ids:
        coordinate_indices = np.asarray(
            [value == geometry_id for value in coordinate_bank.geometry_ids]
        )
        pair_indices = np.asarray([value == geometry_id for value in pair_bank.geometry_ids])
        coordinate_target = coordinate_bank.targets.numpy()[coordinate_indices]
        pair_target = pair_bank.targets.numpy()[pair_indices]
        occupancy_prediction = torch.sigmoid(
            torch.from_numpy(coordinate_prediction[coordinate_indices, 0])
        ).numpy()
        boundary_prediction = torch.sigmoid(
            torch.from_numpy(coordinate_prediction[coordinate_indices, 1])
        ).numpy()
        reachability = torch.sigmoid(
            torch.from_numpy(pair_prediction[pair_indices, 0])
        ).numpy()
        ranking = binary_ranking_metrics(reachability, pair_target[:, 0])
        components = {
            "occupied_iou": binary_iou(occupancy_prediction >= 0.5, coordinate_target[:, 0]),
            "boundary_f1": boundary_f1(
                boundary_prediction >= 0.5, coordinate_target[:, 1], tolerance=0
            ),
            "clearance_nmae": normalized_mae(
                coordinate_prediction[coordinate_indices, 2],
                coordinate_target[:, 2],
                normalizer=1,
            ),
            "reachability_auprc": ranking.auprc,
            "geodesic_nmae": normalized_mae(
                pair_prediction[pair_indices, 1],
                pair_target[:, 1],
                normalizer=1,
                mask=pair_target[:, 0].astype(bool),
            ),
        }
        first = int(np.flatnonzero(coordinate_indices)[0])
        rows.append(
            {
                "geometry_id": geometry_id,
                "family": coordinate_bank.families[first],
                "occupancy_band": coordinate_bank.occupancy_bands[first],
                "components": components,
                "false_open_rate": ranking.false_open_rate,
                "false_closed_rate": ranking.false_closed_rate,
            }
        )
    aggregate = {
        name: float(np.mean([row["components"][name] for row in rows]))
        for name in COMPONENTS
    }
    false_open = float(np.mean([row["false_open_rate"] for row in rows]))
    return aggregate, rows, false_open


def _pilot_score(metrics: dict[str, float], anchors: dict[str, ScoreAnchor]) -> float:
    normalized: list[float] = []
    for name in COMPONENTS:
        anchor = anchors[name]
        value = metrics[name]
        if anchor.higher_is_better:
            normalized.append((value - anchor.floor) / (anchor.ceiling - anchor.floor))
        else:
            normalized.append((anchor.floor - value) / (anchor.floor - anchor.ceiling))
    return float(np.mean(normalized))


def revised_metric_report(
    per_geometry_rows: Sequence[dict[str, object]],
    anchors: dict[str, RevisedScoreAnchor],
) -> dict[str, dict[str, object]]:
    """Report each revised metric independently without a composite score."""

    if not per_geometry_rows:
        raise ValueError("revised metric reporting requires per-geometry rows")
    report: dict[str, dict[str, object]] = {}
    for name in COMPONENTS:
        anchor = anchors[name]
        raw = np.asarray(
            [float(row["components"][name]) for row in per_geometry_rows],
            dtype=np.float64,
        )
        entry: dict[str, object] = {
            "status": anchor.status,
            "raw_mean": float(raw.mean()),
            "raw_iqm": interquartile_mean(raw),
            "per_geometry": raw.tolist(),
        }
        if anchor.status == "deferred":
            entry["deferral_reason"] = anchor.deferral_reason
            report[name] = entry
            continue
        denominator = (
            anchor.ceiling - anchor.floor
            if anchor.higher_is_better
            else anchor.floor - anchor.ceiling
        )
        normalized = (
            (raw - anchor.floor) / denominator
            if anchor.higher_is_better
            else (anchor.floor - raw) / denominator
        )
        entry.update(
            {
                "normalized_iqm": interquartile_mean(normalized),
                "profile_thresholds": list(REVISED_PROFILE_THRESHOLDS),
                "performance_profile": performance_profile(
                    normalized, REVISED_PROFILE_THRESHOLDS
                ).tolist(),
            }
        )
        report[name] = entry
    return report


def pareto_retained_candidates(
    candidate_metric_iqms: dict[str, dict[str, float]],
    *,
    maximum: int = 2,
) -> tuple[str, ...]:
    """Retain nondominated candidates without averaging away weak metrics."""

    if not candidate_metric_iqms or maximum < 1:
        raise ValueError("Pareto retention requires candidates and a positive maximum")
    metric_sets = {frozenset(values) for values in candidate_metric_iqms.values()}
    if len(metric_sets) != 1 or not next(iter(metric_sets)):
        raise ValueError("candidate metric IQMs must share one nonempty metric set")
    candidates = sorted(candidate_metric_iqms)
    nondominated: list[str] = []
    for candidate in candidates:
        values = candidate_metric_iqms[candidate]
        dominated = any(
            other != candidate
            and all(
                candidate_metric_iqms[other][metric] >= value
                for metric, value in values.items()
            )
            and any(
                candidate_metric_iqms[other][metric] > value
                for metric, value in values.items()
            )
            for other in candidates
        )
        if not dominated:
            nondominated.append(candidate)
    if len(nondominated) <= maximum:
        return tuple(nondominated)
    minimums = {
        candidate: min(candidate_metric_iqms[candidate].values())
        for candidate in nondominated
    }
    return tuple(sorted(nondominated, key=lambda item: (-minimums[item], item))[:maximum])


def evaluate_frozen_representation(
    encoder: nn.Module,
    train_descriptors: Sequence[GeometryDescriptor],
    dev_descriptors: Sequence[GeometryDescriptor],
    anchors: dict[str, ScoreAnchor],
    *,
    protocol: ProbeProtocol,
    seed: int,
    device: torch.device,
    final: bool,
    include_controls: bool | None = None,
    corpus_program: str = V1_PROGRAM,
) -> dict[str, object]:
    """Fit fixed probes and return aggregate plus bootstrap-ready geometry rows."""

    before = encoder_state_sha256(encoder)
    train_coordinate, train_pair = _make_banks(
        encoder,
        train_descriptors,
        coordinate_queries=protocol.coordinate_train_queries,
        pair_queries=protocol.pair_train_queries,
        seed=seed,
        device=device,
        corpus_program=corpus_program,
    )
    dev_coordinate, dev_pair = _make_banks(
        encoder,
        dev_descriptors,
        coordinate_queries=protocol.coordinate_dev_queries,
        pair_queries=protocol.pair_dev_queries,
        seed=seed + 1,
        device=device,
        corpus_program=corpus_program,
    )
    updates = protocol.final_updates if final else protocol.intermediate_updates
    include_controls = final if include_controls is None else include_controls
    modes = (
        ("real", "control_target", "zero", "shuffle")
        if include_controls
        else ("real",)
    )
    evaluations: dict[str, dict[str, object]] = {}
    for mode_index, mode in enumerate(modes):
        coordinate_train = _transform_bank(train_coordinate, mode, seed=seed + mode_index)
        pair_train = _transform_bank(train_pair, mode, seed=seed + mode_index)
        coordinate_dev = _transform_bank(dev_coordinate, mode, seed=seed + mode_index)
        pair_dev = _transform_bank(dev_pair, mode, seed=seed + mode_index)
        coordinate_prediction = _fit_probe(
            coordinate_train,
            coordinate_dev,
            task="coordinate",
            protocol=protocol,
            updates=updates,
            seed=seed + mode_index * 2,
            device=device,
        )
        pair_prediction = _fit_probe(
            pair_train,
            pair_dev,
            task="pair",
            protocol=protocol,
            updates=updates,
            seed=seed + mode_index * 2 + 1,
            device=device,
        )
        metrics, rows, false_open = _metric_rows(
            dev_coordinate, coordinate_prediction, dev_pair, pair_prediction
        )
        evaluations[mode] = {
            "components": metrics,
            "pilot_score": _pilot_score(metrics, anchors),
            "false_open_rate": false_open,
            "per_geometry": rows,
        }
    collapse = collapse_diagnostics(dev_coordinate.embeddings.numpy())
    real_score = float(evaluations["real"]["pilot_score"])
    controls = None
    if include_controls:
        control_score = float(evaluations["control_target"]["pilot_score"])
        zero_score = float(evaluations["zero"]["pilot_score"])
        shuffle_score = float(evaluations["shuffle"]["pilot_score"])
        controls = {
            "control_target_selectivity": real_score - control_score,
            "embedding_necessity": real_score - max(zero_score, shuffle_score),
            "passes_selectivity": real_score - control_score >= 0.05,
            "passes_embedding_necessity": real_score - max(zero_score, shuffle_score) >= 0.05,
        }
    if encoder_state_sha256(encoder) != before:
        raise RuntimeError("frozen benchmark mutated encoder state")
    return {
        "real": evaluations["real"],
        "controls": controls,
        "control_evaluations": {
            name: value for name, value in evaluations.items() if name != "real"
        },
        "collapse": {
            "effective_rank": collapse.effective_rank,
            "effective_rank_fraction": collapse.effective_rank_fraction,
            "near_dead_dimensions_fraction": collapse.near_dead_fraction,
            "largest_component_fraction": collapse.dominant_component_share,
        },
        "probe_updates": updates,
        "query_counts": {
            "coordinate_train": len(train_coordinate.features),
            "coordinate_dev": len(dev_coordinate.features),
            "pair_train": len(train_pair.features),
            "pair_dev": len(dev_pair.features),
        },
    }
