"""Calibration-repair data and measured reference models for voxel pilot v2."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

from theseo_anysearch.garden.models.backbones import DenseResidualBackbone
from theseo_anysearch.garden.models.outputs import EncoderMetadata, EncoderOutput, VoxelLevel
from theseo_anysearch.garden.pilots.benchmark import COMPONENTS, ProbeProtocol, evaluate_frozen_representation
from theseo_anysearch.garden.pilots.comparative import ComparativeTrialConfig, ordered_trial_batch
from theseo_anysearch.garden.pilots.contracts import FreshDrawIdentity, PoolIdentity, ScoreAnchor
from theseo_anysearch.garden.pilots.corpus import GENERATOR_VERSION, V2_PROGRAM, make_pilot_observation
from theseo_anysearch.garden.splits import GeometryDescriptor, query_sha256


V2_DATASET_ID = "voxel-encoder-pilot-v2-dataset-1"
V2_PREREGISTRATION_ID = "voxel-encoder-pilot-v2-preregistration-1"
V2_CALIBRATION_RUN_ID = "voxel-encoder-pilot-v2-p0-calibration-1"
V2_DIAGNOSTIC_RUN_ID = "voxel-encoder-pilot-v2-t3-diagnostic-1"
V2_P1_RUN_ID = "voxel-encoder-pilot-v2-p1-1"
V2_POOL_SIZES = {
    "pilot_train": 96,
    "pilot_dev_early": 24,
    "pilot_dev_arch": 24,
    "pilot_dev_interaction": 24,
    "pilot_calibration": 24,
    "pilot_diagnostic": 24,
    "pilot_confirm": 32,
}
V2_POOL_OBSERVATIONS = {
    "pilot_train": 24_000,
    "pilot_dev_early": 6_000,
    "pilot_dev_arch": 6_000,
    "pilot_dev_interaction": 6_000,
    "pilot_calibration": 6_000,
    "pilot_diagnostic": 6_000,
    "pilot_confirm": 12_000,
}
_FAMILIES = ("open", "thin_obstacle", "topology", "imported")
_BANDS = ("low", "medium", "high")


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _rank(seed: int, scope: str, geometry_id: str) -> str:
    return _canonical_sha({"seed": seed, "scope": scope, "geometry_id": geometry_id})


def v2_geometry_records() -> list[GeometryDescriptor]:
    """Return 248 fresh geometry identities with the frozen v2 strata."""

    records: list[GeometryDescriptor] = []
    for index in range(216):
        family = _FAMILIES[index % 4]
        records.append(
            GeometryDescriptor(
                geometry_id=f"pilot-v2-{index:03d}",
                family=family,
                occupancy_band=_BANDS[(index // 4) % 3],
                source=(
                    "synthetic_mesh_import_fixture_v2"
                    if family == "imported"
                    else "procedural_voxel_fixture_v2"
                ),
            )
        )
    groups = (("ordinary", 16), ("heldout_topology", 8), ("heldout_imported", 8))
    offset = 0
    for group, count in groups:
        for local_index in range(count):
            family = (
                "topology"
                if group == "heldout_topology"
                else "imported"
                if group == "heldout_imported"
                else _FAMILIES[local_index % 4]
            )
            records.append(
                GeometryDescriptor(
                    geometry_id=f"pilot-v2-confirm-{offset + local_index:02d}",
                    family=family,
                    occupancy_band=_BANDS[local_index % 3],
                    source=(
                        "heldout_synthetic_mesh_import_fixture_v2"
                        if family == "imported"
                        else "heldout_procedural_voxel_fixture_v2"
                    ),
                    confirmation_group=group,
                )
            )
        offset += count
    return records


def assign_v2_pools(
    records: Sequence[GeometryDescriptor], *, seed: int
) -> dict[str, tuple[str, ...]]:
    """Assign all v2 geometries while keeping every pool stratum-complete."""

    if len(records) != 248 or len({record.geometry_id for record in records}) != 248:
        raise ValueError("v2 requires exactly 248 globally unique geometries")
    if any(record.parent_split != "train" for record in records):
        raise ValueError("v2 geometries must come from the parent training partition")
    regular = [record for record in records if record.confirmation_group is None]
    groups: dict[tuple[str, str], list[GeometryDescriptor]] = {}
    for record in regular:
        groups.setdefault((record.family, record.occupancy_band), []).append(record)
    if set(groups) != {(family, band) for family in _FAMILIES for band in _BANDS}:
        raise ValueError("v2 regular geometries must cover all twelve strata")
    for key, values in groups.items():
        values.sort(key=lambda item: _rank(seed, f"v2:{key}", item.geometry_id))

    pools: dict[str, tuple[str, ...]] = {}
    for pool in (
        "pilot_train",
        "pilot_dev_early",
        "pilot_dev_arch",
        "pilot_dev_interaction",
        "pilot_calibration",
        "pilot_diagnostic",
    ):
        selected: list[GeometryDescriptor] = []
        while len(selected) < V2_POOL_SIZES[pool]:
            progressed = False
            for key in sorted(groups):
                if groups[key] and len(selected) < V2_POOL_SIZES[pool]:
                    selected.append(groups[key].pop(0))
                    progressed = True
            if not progressed:
                raise ValueError(f"not enough v2 geometries for {pool}")
        pools[pool] = tuple(record.geometry_id for record in selected)

    confirmation: list[GeometryDescriptor] = []
    for group, count in (("ordinary", 16), ("heldout_topology", 8), ("heldout_imported", 8)):
        eligible = [record for record in records if record.confirmation_group == group]
        eligible.sort(key=lambda item: _rank(seed, f"v2:confirm:{group}", item.geometry_id))
        if len(eligible) != count:
            raise ValueError(f"v2 confirmation group {group} requires {count} geometries")
        confirmation.extend(eligible)
    pools["pilot_confirm"] = tuple(record.geometry_id for record in confirmation)

    assigned = [geometry_id for values in pools.values() for geometry_id in values]
    if len(assigned) != 248 or len(set(assigned)) != 248:
        raise ValueError("v2 pool assignment must use every geometry exactly once")
    by_id = {record.geometry_id: record for record in records}
    for pool, geometry_ids in pools.items():
        selected = [by_id[geometry_id] for geometry_id in geometry_ids]
        if {record.family for record in selected} != set(_FAMILIES):
            raise ValueError(f"{pool} lacks a geometry family")
        if {record.occupancy_band for record in selected} != set(_BANDS):
            raise ValueError(f"{pool} lacks an occupancy band")
    return pools


def v2_query_plan(pool: str, geometry_ids: tuple[str, ...]) -> list[dict[str, object]]:
    """Return frozen query identities for one v2 pool."""

    if pool == "pilot_train":
        counts = (12_000, 100_000, 50_000, 8_000)
    elif pool == "pilot_confirm":
        counts = (6_000, 40_000, 20_000, 4_000)
    else:
        counts = (3_000, 20_000, 10_000, 2_000)
    return [
        {
            "pool": pool,
            "probe": probe,
            "count": count,
            "seed": 1700 + index,
            "geometry_ids_sha256": _canonical_sha(list(geometry_ids)),
            "assignment": "sha256_rank_round_robin_within_geometry_strata_v2",
        }
        for index, (probe, count) in enumerate(
            zip(("global", "coordinate", "pair", "topology"), counts)
        )
    ]


def build_v2_pool_identities(
    *, seed: int
) -> tuple[list[GeometryDescriptor], dict[str, PoolIdentity], dict[str, FreshDrawIdentity]]:
    records = v2_geometry_records()
    assigned = assign_v2_pools(records, seed=seed)
    pools: dict[str, PoolIdentity] = {}
    for pool, geometry_ids in assigned.items():
        pools[pool] = PoolIdentity(
            geometry_ids=geometry_ids,
            observations=V2_POOL_OBSERVATIONS[pool],
            assignment_sha256=_canonical_sha(
                {"dataset": V2_DATASET_ID, "seed": seed, "pool": pool, "geometry_ids": geometry_ids}
            ),
            query_sha256=query_sha256(v2_query_plan(pool, geometry_ids)),
        )
    fresh_draws = {
        pilot: FreshDrawIdentity(
            seed=draw_seed,
            pool=pool,
            assignment_sha256=pools[pool].assignment_sha256,
            query_sha256=pools[pool].query_sha256,
        )
        for pilot, draw_seed, pool in (
            ("P4", 204, "pilot_dev_arch"),
            ("P6", 206, "pilot_dev_interaction"),
            ("P7", 207, "pilot_confirm"),
        )
    }
    return records, pools, fresh_draws


class FixedFeatureEncoder(nn.Module):
    """Frozen raw-grid projection used for frequency, random, and PCA controls."""

    def __init__(
        self,
        local_weight: torch.Tensor,
        global_weight: torch.Tensor,
        global_mean: torch.Tensor,
        *,
        name: str,
    ) -> None:
        super().__init__()
        if local_weight.shape != (16, 4) or global_weight.shape != (192, 256):
            raise ValueError("fixed control projections have incorrect shapes")
        self.name = name
        self.embedding_dim = 192
        self.register_buffer("local_weight", local_weight.float())
        self.register_buffer("global_weight", global_weight.float())
        self.register_buffer("global_mean", global_mean.float())

    def forward(self, level: VoxelLevel) -> EncoderOutput:
        local = torch.einsum("oc,bcdhw->bodhw", self.local_weight, level.masked_features)
        pooled = F.adaptive_avg_pool3d(level.masked_features, 4).flatten(1)
        embedding = (pooled - self.global_mean) @ self.global_weight.T
        validity = level.validity_mask.float().flatten(1).mean(1, keepdim=True)
        return EncoderOutput(
            global_embedding=embedding,
            scale_embeddings={level.stride: embedding},
            local_feature_volume=local,
            local_validity_mask=level.validity_mask,
            metadata=EncoderMetadata((level.stride,), validity),
        ).validate(embedding_dim=192)


def frequency_control() -> FixedFeatureEncoder:
    return FixedFeatureEncoder(
        torch.zeros(16, 4), torch.zeros(192, 256), torch.zeros(256), name="frequency"
    )


def random_projection_control(seed: int) -> FixedFeatureEncoder:
    generator = torch.Generator().manual_seed(seed)
    local = torch.randn(16, 4, generator=generator) / math.sqrt(4)
    global_weight = torch.randn(192, 256, generator=generator) / math.sqrt(256)
    return FixedFeatureEncoder(local, global_weight, torch.zeros(256), name="fixed_random_projection")


def fit_pca_control(
    descriptors: Sequence[GeometryDescriptor], *, seed: int
) -> FixedFeatureEncoder:
    """Fit both fixed PCA projections using only pilot_train observations."""

    local_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    for index, descriptor in enumerate(descriptors):
        observation = make_pilot_observation(
            descriptor, index % 7, radius=8 if index % 2 == 0 else 16, program=V2_PROGRAM
        )
        level = VoxelLevel.from_occupancy(
            torch.from_numpy(observation.occupancy[None]).float(),
            unknown_mask=torch.from_numpy(observation.unknown_mask[None]),
        )
        features = level.masked_features[0].permute(1, 2, 3, 0).reshape(-1, 4).numpy()
        rng = np.random.default_rng(seed + index)
        chosen = rng.choice(len(features), size=min(256, len(features)), replace=False)
        local_rows.append(features[chosen])
        global_rows.append(F.adaptive_avg_pool3d(level.masked_features, 4).flatten().numpy())
    local_values = np.concatenate(local_rows).astype(np.float64)
    global_values = np.stack(global_rows).astype(np.float64)
    local_mean = local_values.mean(0, keepdims=True)
    global_mean = global_values.mean(0)
    _, _, local_vt = np.linalg.svd(local_values - local_mean, full_matrices=False)
    _, _, global_vt = np.linalg.svd(global_values - global_mean, full_matrices=False)
    local_weight = np.zeros((16, 4), dtype=np.float32)
    local_weight[: len(local_vt)] = local_vt.astype(np.float32)
    global_weight = np.zeros((192, 256), dtype=np.float32)
    global_weight[: len(global_vt)] = global_vt.astype(np.float32)
    return FixedFeatureEncoder(
        torch.from_numpy(local_weight),
        torch.from_numpy(global_weight),
        torch.from_numpy(global_mean.astype(np.float32)),
        name="pca",
    )


class SupervisedReference(nn.Module):
    """Wide supervised upper reference; task heads are discarded after selection."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = DenseResidualBackbone(
            stem_width=32,
            blocks_per_stage=(2, 2, 2, 2),
            embedding_dim=192,
            local_channels=16,
        )
        self.coordinate_head = nn.Conv3d(16, 4, kernel_size=1)
        self.global_head = nn.Sequential(nn.Linear(192, 256), nn.SiLU(), nn.Linear(256, 3))

    def forward(self, level: VoxelLevel) -> EncoderOutput:
        return self.encoder(level)


@dataclass(frozen=True)
class SupervisedReferenceResult:
    model: SupervisedReference
    curve: tuple[dict[str, float | int], ...]
    selected_update: int
    selection_error: float
    residual_training_error: float
    trainable_parameters: int
    tiny_parameters: int


def _reference_loss(model: SupervisedReference, batch: object) -> torch.Tensor:
    output = model(batch.level)
    prediction = model.coordinate_head(output.local_feature_volume)
    class_loss = F.cross_entropy(prediction[:, :3], batch.classes)
    valid = batch.level.validity_mask[:, 0]
    distance_loss = F.smooth_l1_loss(prediction[:, 3][valid], batch.esdf[valid])
    occupancy_fraction = batch.level.features[:, 1].flatten(1).mean(1)
    unknown_fraction = batch.level.features[:, 2].flatten(1).mean(1)
    component_counts = []
    for observation in batch.observations:
        known_free = ~(observation.occupancy | observation.unknown_mask)
        _, count = ndimage.label(known_free)
        component_counts.append(min(count, 16) / 16)
    global_target = torch.stack(
        (
            occupancy_fraction,
            unknown_fraction,
            torch.tensor(component_counts, device=occupancy_fraction.device),
        ),
        dim=1,
    )
    global_loss = F.smooth_l1_loss(model.global_head(output.global_embedding), global_target)
    return class_loss + distance_loss + global_loss


def train_supervised_reference(
    descriptors: Sequence[GeometryDescriptor],
    *,
    device: torch.device,
    seed: int,
    updates: int,
    learning_rate: float,
    selection_interval: int,
) -> SupervisedReferenceResult:
    """Train and select the ceiling without accessing pilot_calibration."""

    if len(descriptors) < 17:
        raise ValueError("supervised reference requires at least 17 pilot_train geometries")
    ordered = sorted(descriptors, key=lambda item: _rank(seed, "reference-fold", item.geometry_id))
    train, selection = ordered[:-16], ordered[-16:]
    torch.manual_seed(seed)
    model = SupervisedReference().to(device)
    tiny = DenseResidualBackbone(
        stem_width=16, blocks_per_stage=(1, 1, 1, 1), embedding_dim=192, local_channels=16
    )
    tiny_parameters = sum(parameter.numel() for parameter in tiny.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters < 4 * tiny_parameters:
        raise RuntimeError("supervised reference is below the four-times-Tiny capacity floor")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    warmup = max(1, round(updates * 0.05))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    config = ComparativeTrialConfig(
        "T0", learning_rate, seed, updates, batch_size=1, corpus_program=V2_PROGRAM
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_error = float("inf")
    best_update = 0
    curve: list[dict[str, float | int]] = []
    final_train_loss = float("inf")
    for update in range(updates):
        model.train()
        batch = ordered_trial_batch(train, config, update, device=device)
        loss = _reference_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("supervised reference loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        final_train_loss = float(loss.detach())
        completed = update + 1
        if completed % selection_interval == 0 or completed == updates:
            model.eval()
            losses: list[float] = []
            with torch.no_grad():
                for offset in range(min(8, len(selection))):
                    selection_batch = ordered_trial_batch(
                        selection, config, completed + offset, device=device
                    )
                    losses.append(float(_reference_loss(model, selection_batch)))
            selection_error = float(np.mean(losses))
            curve.append(
                {
                    "update": completed,
                    "training_error": final_train_loss,
                    "selection_error": selection_error,
                }
            )
            if selection_error < best_error:
                best_error = selection_error
                best_update = completed
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("supervised reference produced no selectable checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return SupervisedReferenceResult(
        model=model,
        curve=tuple(curve),
        selected_update=best_update,
        selection_error=best_error,
        residual_training_error=final_train_loss,
        trainable_parameters=trainable_parameters,
        tiny_parameters=tiny_parameters,
    )


def measure_score_anchors(
    controls: dict[str, nn.Module],
    supervised_reference: nn.Module,
    train_descriptors: Sequence[GeometryDescriptor],
    calibration_descriptors: Sequence[GeometryDescriptor],
    *,
    protocol: ProbeProtocol,
    seed: int,
    device: torch.device,
) -> tuple[
    dict[str, ScoreAnchor],
    dict[str, dict[str, object]],
    dict[str, dict[str, float | str]],
]:
    """Measure fixed floors and ceiling once on pilot_calibration."""

    evaluations: dict[str, dict[str, object]] = {}
    neutral_anchors = {
        component: ScoreAnchor(
            higher_is_better=component not in {"clearance_nmae", "geodesic_nmae"},
            floor=(1.0 if component in {"clearance_nmae", "geodesic_nmae"} else 0.0),
            ceiling=(0.0 if component in {"clearance_nmae", "geodesic_nmae"} else 1.0),
            floor_source="calibration-placeholder",
            ceiling_source="calibration-placeholder",
        )
        for component in COMPONENTS
    }
    for name, encoder in controls.items():
        encoder.to(device).eval()
        evaluations[name] = evaluate_frozen_representation(
            encoder,
            train_descriptors,
            calibration_descriptors,
            neutral_anchors,
            protocol=protocol,
            seed=seed,
            device=device,
            final=True,
            include_controls=False,
            corpus_program=V2_PROGRAM,
        )
    supervised_reference.to(device).eval()
    evaluations["supervised_reference"] = evaluate_frozen_representation(
        supervised_reference,
        train_descriptors,
        calibration_descriptors,
        neutral_anchors,
        protocol=protocol,
        seed=seed,
        device=device,
        final=True,
        include_controls=False,
        corpus_program=V2_PROGRAM,
    )
    anchors, failures = construct_score_anchors(
        {
            name: {
                component: float(result["real"]["components"][component])
                for component in COMPONENTS
            }
            for name, result in evaluations.items()
        }
    )
    return anchors, evaluations, failures


def construct_score_anchors(
    measured_components: dict[str, dict[str, float]],
) -> tuple[dict[str, ScoreAnchor], dict[str, dict[str, float | str]]]:
    """Construct valid anchors while retaining complete evidence for failed gates."""

    required = {"frequency", "pca", "fixed_random_projection", "supervised_reference"}
    if set(measured_components) != required:
        raise ValueError("measured anchors require exactly three controls and one reference")
    anchors: dict[str, ScoreAnchor] = {}
    failures: dict[str, dict[str, float | str]] = {}
    for component in COMPONENTS:
        higher = component not in {"clearance_nmae", "geodesic_nmae"}
        control_values = {
            name: float(result[component])
            for name, result in measured_components.items()
            if name != "supervised_reference"
        }
        floor_source = max(control_values, key=control_values.get) if higher else min(
            control_values, key=control_values.get
        )
        floor = control_values[floor_source]
        ceiling = float(measured_components["supervised_reference"][component])
        try:
            anchors[component] = ScoreAnchor(
                higher_is_better=higher,
                floor=floor,
                ceiling=ceiling,
                floor_source=f"measured:{floor_source}:pilot_calibration",
                ceiling_source="measured:supervised_reference:pilot_calibration",
            )
        except ValueError as error:
            failures[component] = {
                "floor": floor,
                "ceiling": ceiling,
                "floor_source": floor_source,
                "reason": str(error),
            }
    return anchors, failures


__all__ = [
    "GENERATOR_VERSION",
    "V2_CALIBRATION_RUN_ID",
    "V2_DATASET_ID",
    "V2_DIAGNOSTIC_RUN_ID",
    "V2_P1_RUN_ID",
    "V2_PREREGISTRATION_ID",
    "FixedFeatureEncoder",
    "SupervisedReference",
    "assign_v2_pools",
    "build_v2_pool_identities",
    "construct_score_anchors",
    "fit_pca_control",
    "frequency_control",
    "measure_score_anchors",
    "random_projection_control",
    "train_supervised_reference",
    "v2_geometry_records",
]
