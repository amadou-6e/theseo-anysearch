"""Executable contract gates for perception-encoder pilot experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from theseo_anysearch.garden.evaluation.curves import backtest_learning_curves
from theseo_anysearch.garden.evaluation.metrics import (
    binary_iou,
    binary_ranking_metrics,
    boundary_f1,
    cubical_betti_numbers,
    topology_reconstruction_metrics,
)
from theseo_anysearch.garden.evaluation.probes import (
    GlobalLinearProbe,
    encoder_state_sha256,
    train_probe_step,
)
from theseo_anysearch.garden.evaluation.statistics import (
    paired_stratified_geometry_bootstrap,
)
from theseo_anysearch.garden.masking import (
    DenseMaskAwareEncoder,
    hidden_jacobian_max_abs,
    mask_isolation_max_abs,
    mask_shortcut_advantage,
    sample_patch_mask,
)
from theseo_anysearch.garden.micro_scenes import (
    MICRO_SCENE_ORACLES,
    make_micro_scenes,
)
from theseo_anysearch.garden.models.ae import VoxelEncoder3D
from theseo_anysearch.garden.models.backbones import (
    DenseResidualBackbone,
    SharedPyramidBackbone,
    TriPlanarBackbone,
)
from theseo_anysearch.garden.models.objectives import (
    EMATeacher,
    ESDFObjective,
    LatentTargetObjective,
    OccupancyObjective,
)
from theseo_anysearch.garden.models.outputs import VoxelLevel
from theseo_anysearch.garden.pilots.contracts import (
    AcceleratorCaps,
    FreshDrawIdentity,
    FrozenPreregistration,
    PoolIdentity,
    ScoreAnchor,
    SeedAssignments,
    SpecsReference,
    VetoThresholds,
)
from theseo_anysearch.garden.pilots.io import contract_sha256, write_contract
from theseo_anysearch.garden.splits import (
    POOL_OBSERVATIONS,
    GeometryDescriptor,
    assign_pilot_pools,
    query_sha256,
)
from theseo_anysearch.garden.targets import compute_geometry_targets
from theseo_anysearch.garden.trainer import UpdateTrainer, UpdateTrainingConfig


SPEC_SHA = "f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d"


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def pilot_geometry_records() -> list[GeometryDescriptor]:
    """Return 200 stable geometry identities for the direction-finding corpus."""

    families = ("open", "thin_obstacle", "topology", "imported")
    bands = ("low", "medium", "high")
    records: list[GeometryDescriptor] = []
    for index in range(168):
        family = families[index % len(families)]
        records.append(
            GeometryDescriptor(
                geometry_id=f"pilot-v1-{index:03d}",
                family=family,
                occupancy_band=bands[(index // len(families)) % len(bands)],
                source=(
                    "synthetic_mesh_import_fixture_v1"
                    if family == "imported"
                    else "procedural_voxel_fixture_v1"
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
                else families[local_index % len(families)]
            )
            records.append(
                GeometryDescriptor(
                    geometry_id=f"pilot-v1-confirm-{offset + local_index:02d}",
                    family=family,
                    occupancy_band=bands[local_index % len(bands)],
                    source=(
                        "heldout_synthetic_mesh_import_fixture_v1"
                        if family == "imported"
                        else "heldout_procedural_voxel_fixture_v1"
                    ),
                    confirmation_group=group,
                )
            )
        offset += count
    return records


def _query_plan(pool: str, geometry_ids: tuple[str, ...]) -> list[dict[str, object]]:
    counts = {
        "pilot_train": (12_000, 100_000, 50_000, 8_000),
        "pilot_dev_early": (3_000, 20_000, 10_000, 2_000),
        "pilot_dev_arch": (3_000, 20_000, 10_000, 2_000),
        "pilot_dev_interaction": (3_000, 20_000, 10_000, 2_000),
        "pilot_confirm": (6_000, 40_000, 20_000, 4_000),
    }
    names = ("global", "coordinate", "pair", "topology")
    return [
        {
            "pool": pool,
            "probe": name,
            "count": count,
            "seed": 700 + index,
            "geometry_ids_sha256": _sha(list(geometry_ids)),
            "assignment": "sha256_rank_round_robin_within_geometry_strata_v1",
        }
        for index, (name, count) in enumerate(zip(names, counts[pool]))
    ]


def calibrate_score_anchors() -> dict[str, ScoreAnchor]:
    """Lock analytic P0 floor/ceiling fixtures before comparative results exist."""

    binary_target = np.asarray([1, 1, 0, 0], dtype=bool)
    binary_controls = {
        "frequency": np.asarray([0, 0, 0, 0], dtype=bool),
        "fixed_random": np.asarray([1, 0, 1, 0], dtype=bool),
        "pca_projection": np.asarray([1, 0, 0, 0], dtype=bool),
    }
    occupied_floor = max(
        binary_iou(prediction, binary_target)
        for prediction in binary_controls.values()
    )
    boundary_floor = max(
        boundary_f1(prediction, binary_target, tolerance=0)
        for prediction in binary_controls.values()
    )
    reachability_controls = {
        "frequency": np.full(4, 0.5),
        "fixed_random": np.asarray([0.8, 0.2, 0.7, 0.3]),
        "pca_projection": np.asarray([0.6, 0.4, 0.55, 0.45]),
    }
    reachability_floor = max(
        binary_ranking_metrics(scores, binary_target).auprc
        for scores in reachability_controls.values()
    )
    regression_target = np.linspace(0, 1, 5)
    regression_controls = {
        "frequency": np.full(5, 0.5),
        "fixed_random": np.asarray([0.7, 0.1, 0.8, 0.4, 0.2]),
        "pca_projection": np.asarray([0.2, 0.3, 0.5, 0.7, 0.8]),
    }
    regression_floor = min(
        float(np.mean(np.abs(prediction - regression_target)))
        for prediction in regression_controls.values()
    )
    floor_source = "P0 analytic frequency/fixed-random/PCA control fixture v1"
    ceiling_source = "P0 exact supervised-oracle fixture v1"
    return {
        "occupied_iou": ScoreAnchor(
            higher_is_better=True,
            floor=occupied_floor,
            ceiling=1.0,
            floor_source=floor_source,
            ceiling_source=ceiling_source,
        ),
        "boundary_f1": ScoreAnchor(
            higher_is_better=True,
            floor=boundary_floor,
            ceiling=1.0,
            floor_source=floor_source,
            ceiling_source=ceiling_source,
        ),
        "clearance_nmae": ScoreAnchor(
            higher_is_better=False,
            floor=regression_floor,
            ceiling=0.0,
            floor_source=floor_source,
            ceiling_source=ceiling_source,
        ),
        "reachability_auprc": ScoreAnchor(
            higher_is_better=True,
            floor=reachability_floor,
            ceiling=1.0,
            floor_source=floor_source,
            ceiling_source=ceiling_source,
        ),
        "geodesic_nmae": ScoreAnchor(
            higher_is_better=False,
            floor=regression_floor,
            ceiling=0.0,
            floor_source=floor_source,
            ceiling_source=ceiling_source,
        ),
    }


def build_preregistration(
    config: dict[str, Any], *, score_anchors: dict[str, ScoreAnchor] | None = None
) -> FrozenPreregistration:
    assignment = assign_pilot_pools(
        pilot_geometry_records(), seed=int(config["pool_seed"])
    )
    pool_hashes = {
        pool: _sha(
            {
                "global_assignment": assignment.assignment_sha256,
                "pool": pool,
                "geometry_ids": list(geometry_ids),
            }
        )
        for pool, geometry_ids in assignment.pools.items()
    }
    query_hashes = {
        pool: query_sha256(_query_plan(pool, geometry_ids))
        for pool, geometry_ids in assignment.pools.items()
    }
    pools = {
        pool: PoolIdentity(
            geometry_ids=geometry_ids,
            observations=POOL_OBSERVATIONS[pool],
            assignment_sha256=pool_hashes[pool],
            query_sha256=query_hashes[pool],
        )
        for pool, geometry_ids in assignment.pools.items()
    }
    draw_map = {
        "P4": (104, "pilot_dev_arch"),
        "P6": (106, "pilot_dev_interaction"),
        "P7": (107, "pilot_confirm"),
    }
    draws = {
        pilot: FreshDrawIdentity(
            seed=seed,
            pool=pool,
            assignment_sha256=pool_hashes[pool],
            query_sha256=query_hashes[pool],
        )
        for pilot, (seed, pool) in draw_map.items()
    }
    anchors = score_anchors or calibrate_score_anchors()
    return FrozenPreregistration(
        frozen_at=datetime.fromisoformat(config["frozen_at"].replace("Z", "+00:00")),
        specs=SpecsReference(
            repository="https://github.com/amadou-6e/specs",
            commit_sha=SPEC_SHA,
            files=(
                "projects/theseo-anysearch/python/perception-encoder-pilots.md",
                "projects/theseo-anysearch/python/perception-encoder-pretraining.md",
                "projects/theseo-anysearch/python/perception-encoders.md",
            ),
        ),
        accelerator_caps=AcceleratorCaps(
            reference_accelerator=config["reference_accelerator"],
            per_pilot_hours=config["per_pilot_hours"],
            total_comparative_hours=config["total_comparative_hours"],
        ),
        seeds=SeedAssignments(),
        vetoes=VetoThresholds(),
        score_anchors=anchors,
        pools=pools,
        fresh_draws=draws,
    )


def _micro_tensors(
    *, device: torch.device, occupied_only: bool
) -> tuple[VoxelLevel, torch.Tensor, torch.Tensor]:
    scenes = [
        scene for scene in make_micro_scenes() if not occupied_only or scene.occupancy.any()
    ]
    selected = [scenes[index % len(scenes)] for index in range(64)]
    occupancy = torch.from_numpy(np.stack([scene.occupancy for scene in selected])).to(
        device=device, dtype=torch.float32
    )
    unknown = torch.from_numpy(np.stack([scene.unknown_mask for scene in selected])).to(
        device=device
    )
    level = VoxelLevel.from_occupancy(occupancy, unknown_mask=unknown)
    classes = level.features[:, :3].argmax(dim=1)
    esdf = torch.from_numpy(
        np.stack(
            [
                compute_geometry_targets(
                    scene.occupancy, unknown_mask=scene.unknown_mask, truncation=4
                ).signed_distance
                for scene in selected
            ]
        )
    ).to(device=device)
    return level, classes, esdf


def _overfit_bundle(bundle: str, *, device: torch.device, updates: int) -> dict[str, object]:
    seed = {"T0": 10, "T1": 11, "T2": 12, "T3": 13}.get(bundle)
    if seed is None:
        raise ValueError(bundle)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    masked = bundle in {"T1", "T3"}
    level, classes, esdf = _micro_tensors(device=device, occupied_only=masked)
    if masked:
        occupancy = level.features[:, :1]
        sample = sample_patch_mask(occupancy, ratio=0.60, patch_side=2, seed=0)
        hidden_mask = sample.hidden_mask
        encoder: nn.Module = DenseMaskAwareEncoder(
            stem_width=8, embedding_dim=32, local_channels=8
        ).to(device)
    else:
        hidden_mask = torch.zeros_like(level.validity_mask)
        encoder = DenseResidualBackbone(
            stem_width=8,
            blocks_per_stage=(1, 1, 1, 1),
            embedding_dim=32,
            local_channels=8,
        ).to(device)
    objective: nn.Module
    teacher: EMATeacher | None = None
    if bundle in {"T0", "T1"}:
        objective = OccupancyObjective(8)
    elif bundle == "T2":
        objective = ESDFObjective(8, truncation=4)
    elif bundle == "T3":
        objective = LatentTargetObjective(8)
        teacher = EMATeacher(encoder, decay=0.996).to(device)
    module = nn.ModuleDict({"encoder": encoder, "objective": objective}).to(device)
    trainer = UpdateTrainer(
        module,
        UpdateTrainingConfig(
            total_updates=updates,
            peak_learning_rate=3e-3,
            weight_decay=0,
        ),
    )
    initial_encoder_state = encoder_state_sha256(encoder)

    def loss_value() -> torch.Tensor:
        encoded = (
            encoder(level, hidden_mask)
            if isinstance(encoder, DenseMaskAwareEncoder)
            else encoder(level)
        )
        if bundle in {"T0", "T1"}:
            return objective(
                encoded,
                classes,
                supervision_mask=hidden_mask if masked else None,
            ).loss
        if bundle == "T2":
            return objective(encoded, esdf).loss
        assert teacher is not None
        teacher_output = teacher(level, torch.zeros_like(hidden_mask))
        return objective(encoded, teacher_output, supervision_mask=hidden_mask).loss

    with torch.no_grad():
        initial = float(loss_value())
    nonfinite = False
    for _ in range(updates):
        loss = loss_value()
        if not torch.isfinite(loss):
            nonfinite = True
            break
        trainer.step(
            loss,
            observations=len(level.features),
            encoded_views=len(level.features) * (2 if bundle == "T3" else 1),
            valid_voxels=int(level.validity_mask.sum()),
        )
        if teacher is not None:
            teacher.update(encoder)
    with torch.no_grad():
        final = float(loss_value()) if not nonfinite else None
    reduction = 1 - final / initial if initial > 0 and final is not None else None
    state_changed = encoder_state_sha256(encoder) != initial_encoder_state
    return {
        "initial_loss": initial,
        "final_loss": final,
        "fraction_reduced": reduction,
        "updates": trainer.updates,
        "finite": not nonfinite,
        "encoder_state_changed": state_changed,
        "passed": (
            not nonfinite
            and reduction is not None
            and reduction >= 0.90
            and state_changed
        ),
        "resources": trainer.report().__dict__,
    }


def _architecture_training_smokes(device: torch.device) -> dict[str, dict[str, object]]:
    """Run the preregistered 20 gradient batches for every available architecture."""

    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    level, _, _ = _micro_tensors(device=device, occupied_only=False)
    level = VoxelLevel(level.features[:2], level.validity_mask[:2], stride=1)
    mask = torch.zeros_like(level.validity_mask)
    mask[:, :, 3:6, 3:6, 3:6] = True

    cases: dict[str, tuple[nn.Module, Any]] = {
        "current_dense": (
            VoxelEncoder3D(9, [2, 4, 8, 16], 8).to(device),
            lambda model: model(level.features[:, 0]),
        ),
        "dense_residual": (
            DenseResidualBackbone(
                stem_width=2,
                blocks_per_stage=(1, 1, 1, 1),
                embedding_dim=8,
                local_channels=2,
            ).to(device),
            lambda model: model(level),
        ),
        "triplanar": (
            TriPlanarBackbone(
                stem_width=2,
                blocks_per_stage=(1, 1, 1, 1),
                embedding_dim=8,
                local_channels=2,
            ).to(device),
            lambda model: model(level),
        ),
        "shared_pyramid": (
            SharedPyramidBackbone(
                stem_width=2,
                blocks_per_stage=(1, 1, 1, 1),
                embedding_dim=8,
                local_channels=2,
            ).to(device),
            lambda model: model({1: level}),
        ),
        "dense_mask_aware": (
            DenseMaskAwareEncoder(
                stem_width=2, embedding_dim=8, local_channels=2
            ).to(device),
            lambda model: model(level, mask),
        ),
    }
    results: dict[str, dict[str, object]] = {}
    for name, (model, forward) in cases.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial_state = encoder_state_sha256(model)
        finite_gradients = True
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            output = forward(model)
            if isinstance(output, torch.Tensor):
                loss = output.square().mean()
            else:
                loss = output.global_embedding.square().mean()
                loss = loss + output.local_feature_volume.square().mean()
            loss.backward()
            finite_gradients &= bool(
                torch.isfinite(loss)
                and all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )
            optimizer.step()
        state_changed = encoder_state_sha256(model) != initial_state
        results[name] = {
            "passed": finite_gradients and state_changed,
            "batches": 20,
            "finite_gradients": finite_gradients,
            "encoder_state_changed": state_changed,
        }
    return results


def _shared_gates(device: torch.device) -> dict[str, dict[str, object]]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    target_oracles = True
    for scene in make_micro_scenes():
        targets = compute_geometry_targets(
            scene.occupancy, unknown_mask=scene.unknown_mask
        )
        actual = (
            int(targets.occupancy.sum()),
            int(targets.boundary.sum()),
            targets.free_components,
            targets.graph_cycle_rank,
            int(targets.valid_mask.sum()),
        )
        target_oracles &= actual == MICRO_SCENE_ORACLES[scene.name]

    wall = np.zeros((9, 9, 9), dtype=bool)
    wall[4] = True
    broken = wall.copy()
    broken[4, 4, 4] = False
    valid = np.ones_like(wall)
    left = np.ravel_multi_index((2, 4, 4), wall.shape)
    right = np.ravel_multi_index((6, 4, 4), wall.shape)
    topology = topology_reconstruction_metrics(
        broken, wall, valid_mask=valid, pairs=np.asarray([[left, right]])
    )
    metric_behavior = (
        binary_iou(wall, wall) == 1
        and binary_iou(np.zeros_like(wall), wall) == 0
        and boundary_f1(wall, wall) == 1
        and topology.connectivity_change_fraction == 1
        and cubical_betti_numbers(np.pad(np.ones((3, 3, 1), dtype=bool), 1))[0] == 1
        and binary_ranking_metrics(
            np.asarray([0.1, 0.2, 0.8, 0.9]),
            np.asarray([0, 0, 1, 1]),
        ).auprc
        == 1
    )

    reference = np.linspace(0.1, 0.5, 24)
    metadata = {
        "geometry_ids": [f"g-{index}" for index in range(24)],
        "geometry_families": [f"f-{index % 4}" for index in range(24)],
        "occupancy_bands": [f"b-{index % 3}" for index in range(24)],
    }
    difference = paired_stratified_geometry_bootstrap(
        reference + 0.1, reference, resamples=10_000, seed=0, **metadata
    )
    identical = paired_stratified_geometry_bootstrap(
        reference, reference, resamples=10_000, seed=0, **metadata
    )
    bootstrap = difference.lower_95 > 0 and identical.lower_95 == identical.upper_95 == 0

    updates = [100, 200, 400, 800, 1600]
    curves = [
        (updates, [0.8 - scale / np.sqrt(update) for update in updates])
        for scale in np.linspace(0.5, 2.0, 10)
    ]
    curve = backtest_learning_curves(curves, seed=0)

    level = VoxelLevel.from_occupancy(torch.zeros(2, 9, 9, 9, device=device))
    encoder = DenseResidualBackbone(
        stem_width=2,
        blocks_per_stage=(1, 1, 1, 1),
        embedding_dim=8,
        local_channels=2,
    ).to(device)
    probe = GlobalLinearProbe(8, 1).to(device)
    before = encoder_state_sha256(encoder)
    train_probe_step(
        encoder,
        probe,
        (level,),
        lambda model, output: model(output).squeeze(1),
        torch.ones(2, device=device),
        nn.MSELoss(),
        torch.optim.SGD(probe.parameters(), lr=0.1),
    )
    state_integrity = encoder_state_sha256(encoder) == before

    masked_encoder = DenseMaskAwareEncoder(
        stem_width=2, embedding_dim=8, local_channels=2
    ).to(device)
    occupied = torch.zeros(1, 9, 9, 9, device=device)
    occupied[:, 4, 4, 4] = 1
    masked_level = VoxelLevel.from_occupancy(occupied)
    hidden = torch.zeros_like(masked_level.validity_mask)
    hidden[:, :, 3:6, 3:6, 3:6] = True
    isolation = mask_isolation_max_abs(masked_encoder, masked_level, hidden)
    jacobian = hidden_jacobian_max_abs(masked_encoder, masked_level, hidden)
    shortcut_target = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1], device=device)
    shortcut_strata = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    mask_only_prediction = torch.zeros_like(shortcut_target)
    shortcut = mask_shortcut_advantage(
        mask_only_prediction, shortcut_target, shortcut_strata
    )

    architecture_shapes = True
    with torch.no_grad():
        for radius in (8, 16, 32):
            occupancy = torch.zeros(
                1,
                2 * radius + 1,
                2 * radius + 1,
                2 * radius + 1,
                device=device,
            )
            full_level = VoxelLevel.from_occupancy(
                occupancy
            )
            legacy = VoxelEncoder3D(
                2 * radius + 1, [2, 4, 8, 16], 8
            ).to(device)(occupancy)
            architecture_shapes &= legacy.shape == (1, 8)
            for model in (
                DenseResidualBackbone().to(device),
                TriPlanarBackbone().to(device),
            ):
                model(full_level).validate(embedding_dim=192)
            strides = tuple(2**index for index in range((radius // 8).bit_length()))
            pyramid_levels = {
                stride: VoxelLevel.from_occupancy(
                    torch.zeros(1, 17, 17, 17, device=device), stride=stride
                )
                for stride in strides
            }
            SharedPyramidBackbone().to(device)(pyramid_levels).validate(
                embedding_dim=192
            )
    architecture_training = _architecture_training_smokes(device)
    return {
        "target_oracles": {"passed": target_oracles},
        "metric_behavior": {"passed": metric_behavior},
        "paired_bootstrap": {"passed": bootstrap},
        "learning_curve": {
            "passed": curve.calibrated,
            "coverage_95": curve.coverage_95,
            "median_absolute_error": curve.median_absolute_error,
        },
        "frozen_state": {"passed": state_integrity},
        "mask_isolation": {
            "passed": isolation <= 1e-6 and jacobian <= 1e-8 and shortcut <= 0.01,
            "intervention_max_abs": isolation,
            "hidden_jacobian_max_abs": jacobian,
            "mask_only_shortcut_advantage": shortcut,
        },
        "architecture_outputs": {"passed": architecture_shapes},
        "architecture_training": {
            "passed": all(result["passed"] for result in architecture_training.values()),
            "candidates": architecture_training,
        },
    }


def run_p0(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("P0 requires the preregistered CUDA reference accelerator")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    anchors = calibrate_score_anchors()
    preregistration = build_preregistration(config, score_anchors=anchors)
    preregistration_path = output_dir / "preregistration.yaml"
    preregistration_identity = write_contract(
        preregistration_path, preregistration
    )
    shared = _shared_gates(device)
    objectives = {
        bundle: _overfit_bundle(bundle, device=device, updates=int(config["p0_updates"]))
        for bundle in ("T0", "T1", "T2", "T3")
    }
    shared_passed = all(result["passed"] for result in shared.values())
    retained = [name for name, result in objectives.items() if result["passed"]]
    rejected = [name for name in objectives if name not in retained]
    status = "completed" if shared_passed else "blocked"
    if not shared_passed:
        decision = "blocked"
    elif not retained:
        decision = "no_viable_direction"
    else:
        decision = "tie"
    report: dict[str, object] = {
        "issue": 277,
        "pilot": "P0",
        "status": status,
        "integration_base_sha": _git(
            "merge-base", "HEAD", "origin/exp/perception-encoder"
        ),
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": SPEC_SHA,
        "preregistration_sha256": preregistration_identity,
        "preregistration_contract_sha256": contract_sha256(preregistration),
        "resolved_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "shared_gates": shared,
        "score_anchor_calibration": {
            name: anchor.model_dump(mode="json") for name, anchor in anchors.items()
        },
        "objective_overfit": objectives,
        "trial_counts": {
            "completed": sum(
                result["updates"] == int(config["p0_updates"])
                for result in objectives.values()
            ),
            "failed": len(rejected),
            "skipped": 0,
            "cap": 4,
        },
        "accelerator_hours": (time.perf_counter() - started) / 3600,
        "validity_flags": [
            "direction_finding_only",
            "synthetic_mesh_import_fixture_is_not_external_out_of_family_evidence",
            "P0_selects_nothing",
        ],
        "decision_record": {
            "decision": decision,
            "retained": retained,
            "rejected": rejected,
            "rejection_rules": {
                name: ["P0_objective_loss_reduction_below_90_percent"] for name in rejected
            },
            "reason": (
                "shared harness passed; P0 selects nothing and eligible recipes advance to P1"
                if shared_passed and retained
                else "all objective recipes failed their P0 gate; the pilot program stops"
                if shared_passed
                else "a shared P0 harness gate failed; later pilots are blocked"
            ),
            "next_pilot": "P1" if shared_passed and retained else None,
            "disposition": "retain" if shared_passed and retained else "reject",
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_payload_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p0-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
