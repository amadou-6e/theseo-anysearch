"""Frozen implementation-health replay for the v2 T3 latent-target recipe."""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from theseo_anysearch.garden.evaluation.metrics import collapse_diagnostics
from theseo_anysearch.garden.evaluation.probes import encoder_state_sha256
from theseo_anysearch.garden.masking import DenseMaskAwareEncoder, sample_patch_mask
from theseo_anysearch.garden.models.objectives import EMATeacher, LatentTargetObjective
from theseo_anysearch.garden.models.outputs import EncoderOutput, VoxelLevel
from theseo_anysearch.garden.pilots.benchmark import ProbeProtocol, evaluate_frozen_representation
from theseo_anysearch.garden.pilots.comparative import (
    ComparativeTrialConfig,
    build_bundle_modules,
    ordered_trial_batch,
)
from theseo_anysearch.garden.pilots.contracts import ScoreAnchor
from theseo_anysearch.garden.pilots.corpus import V2_PROGRAM, make_pilot_observation
from theseo_anysearch.garden.splits import GeometryDescriptor


_TELEMETRY_UPDATES = tuple(range(0, 2_001, 100))
_PROBE_UPDATES = {200, 500, 1_000, 1_500, 2_000}


def _matrix_statistics(values: torch.Tensor) -> dict[str, object]:
    matrix = values.detach().float().reshape(-1, values.shape[-1]).cpu().numpy()
    if len(matrix) > 8_192:
        matrix = matrix[np.linspace(0, len(matrix) - 1, 8_192, dtype=np.int64)]
    diagnostics = collapse_diagnostics(matrix)
    return {
        "per_channel_std": np.std(matrix, axis=0).tolist(),
        "effective_rank_fraction": diagnostics.effective_rank_fraction,
        "largest_component_fraction": diagnostics.dominant_component_share,
        "near_dead_dimensions_fraction": diagnostics.near_dead_fraction,
    }


def _feature_matrices(
    encoder: nn.Module,
    teacher: EMATeacher,
    objective: LatentTargetObjective,
    descriptors: Sequence[GeometryDescriptor],
    *,
    device: torch.device,
) -> tuple[dict[str, dict[str, object]], float, float, float]:
    online_local: list[torch.Tensor] = []
    online_global: list[torch.Tensor] = []
    teacher_local: list[torch.Tensor] = []
    teacher_global: list[torch.Tensor] = []
    predictor_local: list[torch.Tensor] = []
    predictor_global: list[torch.Tensor] = []
    cosine: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    encoder.eval()
    teacher.eval()
    objective.eval()
    with torch.no_grad():
        for observation_offset in (0, 1, 2):
            for descriptor in descriptors:
                observation = make_pilot_observation(
                    descriptor,
                    observation_offset,
                    radius=8,
                    program=V2_PROGRAM,
                )
                level = VoxelLevel.from_occupancy(
                    torch.from_numpy(observation.occupancy[None]).to(device=device, dtype=torch.float32),
                    unknown_mask=torch.from_numpy(observation.unknown_mask[None]).to(device=device),
                )
                hidden = torch.zeros_like(level.validity_mask)
                online = encoder(level, hidden)
                target = teacher(level, hidden)
                if not isinstance(online, EncoderOutput):
                    raise TypeError("T3 diagnostics require EncoderOutput")
                prediction = objective.predictor(online.local_feature_volume)
                online_local.append(online.local_feature_volume.permute(0, 2, 3, 4, 1).reshape(-1, 16)[::8].cpu())
                teacher_local.append(target.local_feature_volume.permute(0, 2, 3, 4, 1).reshape(-1, 16)[::8].cpu())
                predictor_local.append(prediction.permute(0, 2, 3, 4, 1).reshape(-1, 16)[::8].cpu())
                online_global.append(online.global_embedding.cpu())
                teacher_global.append(target.global_embedding.cpu())
                predictor_global.append(F.adaptive_avg_pool3d(prediction, 1).flatten(1).cpu())
                cosine.append(
                    F.cosine_similarity(
                        online.local_feature_volume.flatten(1),
                        target.local_feature_volume.flatten(1),
                    ).cpu()
                )
                targets.append(target.local_feature_volume.flatten().cpu())
    matrices = {
        "online_encoder": {
            "local": _matrix_statistics(torch.cat(online_local)),
            "global": _matrix_statistics(torch.cat(online_global)),
        },
        "predictor": {
            "local": _matrix_statistics(torch.cat(predictor_local)),
            "global": _matrix_statistics(torch.cat(predictor_global)),
        },
        "ema_teacher": {
            "local": _matrix_statistics(torch.cat(teacher_local)),
            "global": _matrix_statistics(torch.cat(teacher_global)),
        },
    }
    target_values = torch.cat(targets)
    return (
        matrices,
        float(torch.cat(cosine).mean()),
        float(torch.linalg.vector_norm(target_values) / math.sqrt(target_values.numel())),
        float(target_values.var()),
    )


def _parameter_drift(left: nn.Module, right: nn.Module) -> float:
    right_parameters = dict(right.named_parameters())
    squared = torch.zeros((), device=next(left.parameters()).device)
    for name, value in left.named_parameters():
        squared += (value.detach() - right_parameters[name].detach()).square().sum()
    return float(torch.sqrt(squared))


def _parameter_update_norm(before: list[torch.Tensor], module: nn.Module) -> float:
    squared = torch.zeros((), device=before[0].device)
    for previous, current in zip(before, module.parameters()):
        squared += (current.detach() - previous).square().sum()
    return float(torch.sqrt(squared))


def _health_labels(telemetry: Sequence[dict[str, object]]) -> tuple[str, ...]:
    labels: list[str] = []
    consecutive_target = 0
    consecutive_online = 0
    for row in telemetry:
        if int(row["update"]) == 0:
            continue
        features = row["features"]
        target = features["ema_teacher"]["local"]
        online = features["online_encoder"]["global"]
        target_bad = (
            target["effective_rank_fraction"] < 0.25
            or target["near_dead_dimensions_fraction"] > 0.05
        )
        online_bad = (
            online["effective_rank_fraction"] < 0.25
            or online["near_dead_dimensions_fraction"] > 0.05
        )
        consecutive_target = consecutive_target + 1 if target_bad else 0
        consecutive_online = consecutive_online + 1 if online_bad else 0
        if consecutive_target >= 2 and "target_collapse" not in labels:
            labels.append("target_collapse")
        if consecutive_online >= 2 and "representation_collapse" not in labels:
            labels.append("representation_collapse")
    return tuple(labels)


def run_t3_cell(
    train_descriptors: Sequence[GeometryDescriptor],
    diagnostic_descriptors: Sequence[GeometryDescriptor],
    anchors: dict[str, ScoreAnchor],
    *,
    learning_rate: float,
    protocol: ProbeProtocol,
    device: torch.device,
) -> dict[str, object]:
    """Run one frozen 2,000-update T3 diagnostic cell with full telemetry."""

    config = ComparativeTrialConfig(
        "T3",
        learning_rate,
        5,
        2_000,
        batch_size=2,
        mask_ratio=0.60,
        ema_decay=0.996,
        corpus_program=V2_PROGRAM,
    )
    encoder, objective_module, teacher = build_bundle_modules(config, device=device)
    if not isinstance(encoder, DenseMaskAwareEncoder):
        raise TypeError("T3 diagnostic requires the mask-aware encoder")
    if not isinstance(objective_module, LatentTargetObjective) or teacher is None:
        raise TypeError("T3 diagnostic requires latent objective and EMA teacher")
    module = nn.ModuleDict({"encoder": encoder, "objective": objective_module})
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=0.05)
    warmup = 100

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (2_000 - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    initial_encoder_hash = encoder_state_sha256(encoder)
    initial_teacher_hash = encoder_state_sha256(teacher.encoder)
    telemetry: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    implementation_errors: list[str] = []
    started = time.perf_counter()
    peak_allocated = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def record(
        update: int,
        loss: float,
        lr: float,
        gradient_norm: float,
        clipped: bool,
        parameter_update_norm: float,
    ) -> None:
        features, similarity, target_norm, target_variance = _feature_matrices(
            encoder,
            teacher,
            objective_module,
            diagnostic_descriptors,
            device=device,
        )
        values = (loss, lr, gradient_norm, parameter_update_norm, similarity, target_norm, target_variance)
        if not all(np.isfinite(value) for value in values):
            implementation_errors.append(f"non_finite_telemetry_at_{update}")
        telemetry.append(
            {
                "update": update,
                "pretext_loss": loss,
                "learning_rate": lr,
                "unclipped_gradient_norm": gradient_norm,
                "clipping_occurred": clipped,
                "encoder_parameter_update_norm": parameter_update_norm,
                "features": features,
                "ema_parameter_drift": _parameter_drift(encoder, teacher.encoder),
                "online_teacher_feature_cosine_similarity": similarity,
                "target_norm": target_norm,
                "target_variance": target_variance,
            }
        )

    initial_batch = ordered_trial_batch(train_descriptors, config, 0, device=device)
    initial_hidden = sample_patch_mask(
        initial_batch.level.features[:, :1],
        unknown_mask=initial_batch.level.features[:, 2:3].bool(),
        ratio=0.60,
        patch_side=4,
        seed=10_000,
    ).hidden_mask
    initial_online = encoder(initial_batch.level, initial_hidden)
    initial_target = teacher(initial_batch.level, torch.zeros_like(initial_hidden))
    initial_loss = objective_module(
        initial_online, initial_target, supervision_mask=initial_hidden
    ).loss
    optimizer.zero_grad(set_to_none=True)
    initial_loss.backward()
    initial_gradient = float(torch.nn.utils.clip_grad_norm_(module.parameters(), float("inf")))
    optimizer.zero_grad(set_to_none=True)
    record(0, float(initial_loss.detach()), 0.0, initial_gradient, False, 0.0)

    for update in range(2_000):
        encoder.train()
        objective_module.train()
        batch = ordered_trial_batch(train_descriptors, config, update, device=device)
        hidden = sample_patch_mask(
            batch.level.features[:, :1],
            unknown_mask=batch.level.features[:, 2:3].bool(),
            ratio=0.60,
            patch_side=4,
            seed=5 * 2_000 + update,
        ).hidden_mask
        online = encoder(batch.level, hidden)
        target = teacher(batch.level, torch.zeros_like(hidden))
        loss = objective_module(online, target, supervision_mask=hidden).loss
        if not torch.isfinite(loss):
            implementation_errors.append(f"non_finite_loss_at_{update + 1}")
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0))
        before = [parameter.detach().clone() for parameter in encoder.parameters()]
        current_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        teacher.update(encoder)
        parameter_update = _parameter_update_norm(before, encoder)
        completed = update + 1
        if completed in _TELEMETRY_UPDATES:
            record(
                completed,
                float(loss.detach()),
                current_lr,
                gradient_norm,
                gradient_norm > 1.0,
                parameter_update,
            )
        if completed in _PROBE_UPDATES:
            probes.append(
                {
                    "update": completed,
                    **evaluate_frozen_representation(
                        encoder,
                        train_descriptors,
                        diagnostic_descriptors,
                        anchors,
                        protocol=protocol,
                        seed=5,
                        device=device,
                        final=completed == 2_000,
                        include_controls=True,
                        corpus_program=V2_PROGRAM,
                    ),
                }
            )
    final_encoder_hash = encoder_state_sha256(encoder)
    final_teacher_hash = encoder_state_sha256(teacher.encoder)
    if final_encoder_hash == initial_encoder_hash:
        implementation_errors.append("encoder_state_unchanged")
    if final_teacher_hash == initial_teacher_hash:
        implementation_errors.append("ema_teacher_state_unchanged")
    if len(telemetry) != len(_TELEMETRY_UPDATES):
        implementation_errors.append("incomplete_telemetry")
    if len(probes) != len(_PROBE_UPDATES):
        implementation_errors.append("incomplete_frozen_probes")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
    return {
        "learning_rate": learning_rate,
        "seed": 5,
        "updates": 2_000,
        "telemetry": telemetry,
        "frozen_probes": probes,
        "health_labels": list(_health_labels(telemetry)),
        "implementation_errors": sorted(set(implementation_errors)),
        "initial_encoder_sha256": initial_encoder_hash,
        "final_encoder_sha256": final_encoder_hash,
        "initial_teacher_sha256": initial_teacher_hash,
        "final_teacher_sha256": final_teacher_hash,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": peak_allocated,
    }


def classify_t3_replay(cells: Sequence[dict[str, object]]) -> dict[str, object]:
    """Apply preregistered cross-learning-rate health classifications."""

    if {float(cell["learning_rate"]) for cell in cells} != {0.0001, 0.0003}:
        raise ValueError("T3 replay requires exactly the two frozen learning rates")
    labels_by_rate = {
        f"{float(cell['learning_rate']):g}": tuple(str(value) for value in cell["health_labels"])
        for cell in cells
    }
    shared = set(labels_by_rate["0.0001"]) & set(labels_by_rate["0.0003"])
    losses = {
        f"{float(cell['learning_rate']):g}": {
            int(row["update"]): float(row["pretext_loss"]) for row in cell["telemetry"]
        }
        for cell in cells
    }
    late_instability = all(
        values[2_000] > 1.25 * min(loss for update, loss in values.items() if update >= 1_000)
        for values in losses.values()
    )
    if late_instability:
        shared.add("late_optimization_instability")
    implementation_failure = any(cell["implementation_errors"] for cell in cells)
    mechanism_failure = bool(
        shared & {"target_collapse", "representation_collapse", "late_optimization_instability"}
    )
    return {
        "labels_by_learning_rate": {key: list(value) for key, value in labels_by_rate.items()},
        "shared_labels": sorted(shared),
        "late_optimization_instability": late_instability,
        "implementation_failure": implementation_failure,
        "mechanism_health_failure": mechanism_failure,
        "passed": not implementation_failure and not mechanism_failure,
    }


__all__ = ["classify_t3_replay", "run_t3_cell"]
