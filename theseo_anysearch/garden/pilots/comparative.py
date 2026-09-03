"""Reusable update-counted training runtime for comparative encoder pilots."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from theseo_anysearch.garden.evaluation.probes import encoder_state_sha256
from theseo_anysearch.garden.masking import DenseMaskAwareEncoder, sample_patch_mask
from theseo_anysearch.garden.models.backbones import DenseResidualBackbone
from theseo_anysearch.garden.models.objectives import (
    EMATeacher,
    ESDFObjective,
    LatentTargetObjective,
    OccupancyObjective,
)
from theseo_anysearch.garden.models.outputs import EncoderOutput, VoxelLevel
from theseo_anysearch.garden.pilots.corpus import (
    PilotObservation,
    V1_PROGRAM,
    make_pilot_observation,
    proper_cube_rotation,
)
from theseo_anysearch.garden.splits import GeometryDescriptor
from theseo_anysearch.garden.targets import compute_geometry_targets
from theseo_anysearch.garden.trainer import UpdateTrainer, UpdateTrainingConfig


CheckpointCallback = Callable[[nn.Module, int], dict[str, object]]


@dataclass(frozen=True)
class ComparativeTrialConfig:
    bundle: str
    peak_learning_rate: float
    seed: int
    updates: int
    batch_size: int = 2
    mask_ratio: float = 0.60
    ema_decay: float = 0.996
    focal_gamma: float = 0.0
    surface_only: bool = False
    esdf_radius_fraction: float = 0.25
    cube_rotations: bool = False
    density_multiplier: int = 1
    corpus_program: str = V1_PROGRAM
    checkpoint_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)

    def __post_init__(self) -> None:
        if self.bundle not in {"T0", "T1", "T2", "T3"}:
            raise ValueError(f"unsupported pilot bundle: {self.bundle}")
        if self.peak_learning_rate <= 0 or self.updates < 1 or self.batch_size < 1:
            raise ValueError("learning rate, updates, and batch size must be positive")
        if not 0 < self.mask_ratio < 1 or not 0 < self.ema_decay < 1:
            raise ValueError("mask ratio and EMA decay must be in (0, 1)")
        if self.esdf_radius_fraction <= 0 or self.density_multiplier not in {1, 4}:
            raise ValueError("invalid ESDF scale or observation-density multiplier")
        if self.corpus_program not in {"voxel-encoder-pilot-v1", "voxel-encoder-pilot-v2"}:
            raise ValueError("unsupported corpus program")
        if self.checkpoint_fractions != tuple(sorted(set(self.checkpoint_fractions))):
            raise ValueError("checkpoint fractions must be unique and increasing")
        if not self.checkpoint_fractions or self.checkpoint_fractions[-1] != 1.0:
            raise ValueError("checkpoint fractions must include the final update")
        if any(not 0 < fraction <= 1 for fraction in self.checkpoint_fractions):
            raise ValueError("checkpoint fractions must be in (0, 1]")

    @property
    def checkpoint_updates(self) -> tuple[int, ...]:
        return tuple(max(1, round(self.updates * value)) for value in self.checkpoint_fractions)

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class TrialBatch:
    level: VoxelLevel
    classes: torch.Tensor
    esdf: torch.Tensor
    surface_mask: torch.Tensor
    observations: tuple[PilotObservation, ...]


def ordered_trial_batch(
    descriptors: Sequence[GeometryDescriptor],
    config: ComparativeTrialConfig,
    update: int,
    *,
    device: torch.device,
) -> TrialBatch:
    """Materialize one paired batch from immutable IDs and the update index."""

    if not descriptors or update < 0:
        raise ValueError("a nonempty descriptor pool and nonnegative update are required")
    radius = 8 if update % 2 == 0 else 16
    observations: list[PilotObservation] = []
    occupancies: list[np.ndarray] = []
    unknown_masks: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    esdf: list[np.ndarray] = []
    surface_masks: list[np.ndarray] = []
    for batch_index in range(config.batch_size):
        sequence_index = update * config.batch_size + batch_index
        descriptor = descriptors[sequence_index % len(descriptors)]
        observation_index = sequence_index // len(descriptors)
        observation = make_pilot_observation(
            descriptor,
            observation_index,
            radius=radius,
            density_multiplier=config.density_multiplier,
            program=config.corpus_program,
        )
        occupancy = observation.occupancy
        unknown = observation.unknown_mask
        if config.cube_rotations:
            rotation = config.seed * config.updates + sequence_index
            occupancy = proper_cube_rotation(occupancy, rotation)
            unknown = proper_cube_rotation(unknown, rotation)
        targets = compute_geometry_targets(
            occupancy,
            unknown_mask=unknown,
            truncation=radius * config.esdf_radius_fraction,
        )
        class_target = np.where(unknown, 2, occupancy.astype(np.int64))
        frontier = ndimage.binary_dilation(unknown) & ~unknown
        surface = ndimage.binary_dilation(targets.boundary) | frontier
        observations.append(observation)
        occupancies.append(occupancy)
        unknown_masks.append(unknown)
        classes.append(class_target)
        esdf.append(targets.signed_distance)
        surface_masks.append(surface)
    occupancy_tensor = torch.from_numpy(np.stack(occupancies)).to(
        device=device, dtype=torch.float32
    )
    unknown_tensor = torch.from_numpy(np.stack(unknown_masks)).to(device=device)
    level = VoxelLevel.from_occupancy(occupancy_tensor, unknown_mask=unknown_tensor)
    return TrialBatch(
        level=level,
        classes=torch.from_numpy(np.stack(classes)).to(device=device),
        esdf=torch.from_numpy(np.stack(esdf)).to(device=device),
        surface_mask=torch.from_numpy(np.stack(surface_masks))[:, None].to(device=device),
        observations=tuple(observations),
    )


def build_bundle_modules(
    config: ComparativeTrialConfig,
    *,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, EMATeacher | None]:
    """Build parameter-compatible Tiny encoder and disposable objective modules."""

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    masked = config.bundle in {"T1", "T3"}
    encoder: nn.Module = (
        DenseMaskAwareEncoder(stem_width=16, embedding_dim=192, local_channels=16)
        if masked
        else DenseResidualBackbone(
            stem_width=16,
            blocks_per_stage=(1, 1, 1, 1),
            embedding_dim=192,
            local_channels=16,
        )
    ).to(device)
    if config.bundle in {"T0", "T1"}:
        objective: nn.Module = OccupancyObjective(16, focal_gamma=config.focal_gamma)
    elif config.bundle == "T2":
        objective = ESDFObjective(16, truncation=4)
    else:
        objective = LatentTargetObjective(16)
    objective = objective.to(device)
    teacher = (
        EMATeacher(encoder, decay=config.ema_decay).to(device)
        if config.bundle == "T3"
        else None
    )
    return encoder, objective, teacher


def _trial_loss(
    encoder: nn.Module,
    objective: nn.Module,
    teacher: EMATeacher | None,
    batch: TrialBatch,
    config: ComparativeTrialConfig,
    *,
    mask_seed: int,
) -> tuple[torch.Tensor, int]:
    masked = config.bundle in {"T1", "T3"}
    hidden_mask = (
        sample_patch_mask(
            batch.level.features[:, :1],
            unknown_mask=batch.level.features[:, 2:3].bool(),
            ratio=config.mask_ratio,
            patch_side=4,
            seed=mask_seed,
        ).hidden_mask
        if masked
        else torch.zeros_like(batch.level.validity_mask)
    )
    encoded = (
        encoder(batch.level, hidden_mask)
        if isinstance(encoder, DenseMaskAwareEncoder)
        else encoder(batch.level)
    )
    if not isinstance(encoded, EncoderOutput):
        raise TypeError("comparative encoders must return EncoderOutput")
    if config.bundle in {"T0", "T1"}:
        supervision = hidden_mask if masked else None
        if config.surface_only:
            supervision = batch.surface_mask & (
                hidden_mask if masked else batch.level.validity_mask
            )
        result = objective(encoded, batch.classes, supervision_mask=supervision)
    elif config.bundle == "T2":
        assert isinstance(objective, ESDFObjective)
        objective.truncation = batch.level.features.shape[-1] // 2 * config.esdf_radius_fraction
        result = objective(encoded, batch.esdf, boundary_mask=batch.surface_mask)
    else:
        assert teacher is not None
        teacher_output = teacher(batch.level, torch.zeros_like(hidden_mask))
        result = objective(encoded, teacher_output, supervision_mask=hidden_mask)
    return result.loss, result.supervised_values


@dataclass
class ComparativeTrialResult:
    config_sha256: str
    initial_encoder_sha256: str
    final_encoder_sha256: str
    checkpoint_results: list[dict[str, object]] = field(default_factory=list)
    loss_checkpoints: list[dict[str, float | int]] = field(default_factory=list)
    resources: dict[str, object] = field(default_factory=dict)
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    wall_seconds: float = 0.0
    status: str = "completed"
    error: str | None = None


def train_comparative_trial(
    descriptors: Sequence[GeometryDescriptor],
    config: ComparativeTrialConfig,
    *,
    device: torch.device,
    evaluate: CheckpointCallback | None = None,
) -> tuple[nn.Module, ComparativeTrialResult]:
    """Train one complete paired trial and evaluate its five frozen checkpoints."""

    started = time.perf_counter()
    encoder, objective, teacher = build_bundle_modules(config, device=device)
    module = nn.ModuleDict({"encoder": encoder, "objective": objective})
    trainer = UpdateTrainer(
        module,
        UpdateTrainingConfig(
            total_updates=config.updates,
            peak_learning_rate=config.peak_learning_rate,
            weight_decay=0.05,
            warmup_fraction=0.05,
            gradient_clip_norm=1.0,
        ),
    )
    initial_hash = encoder_state_sha256(encoder)
    result = ComparativeTrialResult(
        config_sha256=config.identity_sha256,
        initial_encoder_sha256=initial_hash,
        final_encoder_sha256=initial_hash,
    )
    checkpoints = set(config.checkpoint_updates)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for update in range(config.updates):
            batch = ordered_trial_batch(descriptors, config, update, device=device)
            loss, supervised = _trial_loss(
                encoder,
                objective,
                teacher,
                batch,
                config,
                mask_seed=config.seed * config.updates + update,
            )
            trainer.step(
                loss,
                observations=len(batch.observations),
                encoded_views=len(batch.observations) * (2 if teacher is not None else 1),
                valid_voxels=int(batch.level.validity_mask.sum()),
            )
            if teacher is not None:
                teacher.update(encoder)
            completed = update + 1
            if completed in checkpoints:
                result.loss_checkpoints.append(
                    {"update": completed, "loss": float(loss.detach()), "supervised": supervised}
                )
                if evaluate is not None:
                    before = encoder_state_sha256(encoder)
                    checkpoint = evaluate(encoder, completed)
                    if encoder_state_sha256(encoder) != before:
                        raise RuntimeError("frozen checkpoint evaluation mutated the encoder")
                    result.checkpoint_results.append({"update": completed, **checkpoint})
        result.final_encoder_sha256 = encoder_state_sha256(encoder)
        if result.final_encoder_sha256 == initial_hash:
            raise RuntimeError("encoder state did not change during comparative training")
    except (RuntimeError, FloatingPointError) as error:
        result.status = "failed"
        result.error = f"{type(error).__name__}: {error}"
    resource_report = trainer.report()
    result.resources = asdict(resource_report)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result.peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        result.peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    result.wall_seconds = time.perf_counter() - started
    return encoder, result
