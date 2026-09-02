"""P3 random-weight architecture feasibility profile."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from theseo_anysearch.garden.models.ae import VoxelEncoder3D
from theseo_anysearch.garden.models.backbones import (
    DenseResidualBackbone,
    SharedPyramidBackbone,
    TriPlanarBackbone,
    pilot_backbone_capabilities,
)
from theseo_anysearch.garden.models.outputs import EncoderOutput, VoxelLevel


CANDIDATES = ("current_dense", "dense_residual", "triplanar", "shared_pyramid")
RADII = (8, 16, 32)


def _nvidia_driver_version() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _warm_cuda_training_runtime(device: torch.device) -> None:
    module = nn.Linear(8, 8).to(device)
    optimizer = torch.optim.AdamW(module.parameters())
    optimizer.zero_grad(set_to_none=True)
    module(torch.ones(1, 8, device=device)).sum().backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    del module, optimizer
    torch.cuda.empty_cache()


@dataclass(frozen=True)
class ProfileCell:
    candidate: str
    radius: int
    status: str
    parameters: int
    checkpoint_bytes: int
    flops: int
    latency_p50_ms: float
    latency_p95_ms: float
    peak_inference_allocated_bytes: int
    peak_inference_reserved_bytes: int
    peak_training_allocated_bytes: int
    peak_training_reserved_bytes: int
    training_examples_per_second: float
    output_shapes: dict[str, object]
    output_contract: str
    error: str | None = None


def _pyramid_strides(radius: int) -> tuple[int, ...]:
    return tuple(2**index for index in range(int((radius // 8)).bit_length()))


def _build(
    candidate: str, radius: int, device: torch.device
) -> tuple[nn.Module, Callable[[], torch.Tensor | EncoderOutput]]:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    if candidate == "current_dense":
        model = VoxelEncoder3D(2 * radius + 1, [18, 36, 72, 144], 192).to(device)
        occupancy = torch.randint(
            0, 2, (1, 2 * radius + 1, 2 * radius + 1, 2 * radius + 1), device=device
        ).float()
        return model, lambda: model(occupancy)
    if candidate == "shared_pyramid":
        model = SharedPyramidBackbone().to(device)
        levels = {
            stride: VoxelLevel.from_occupancy(
                torch.randint(0, 2, (1, 17, 17, 17), device=device).float(),
                stride=stride,
            )
            for stride in _pyramid_strides(radius)
        }
        return model, lambda: model(levels)
    model_type = {
        "dense_residual": DenseResidualBackbone,
        "triplanar": TriPlanarBackbone,
    }[candidate]
    model = model_type().to(device)
    level = VoxelLevel.from_occupancy(
        torch.randint(
            0, 2, (1, 2 * radius + 1, 2 * radius + 1, 2 * radius + 1), device=device
        ).float()
    )
    return model, lambda: model(level)


def _output_shapes(output: torch.Tensor | EncoderOutput) -> tuple[str, dict[str, object]]:
    if isinstance(output, EncoderOutput):
        output.validate(embedding_dim=192)
        return "global_scale_local_v1", {
            "global": list(output.global_embedding.shape),
            "scales": {str(key): list(value.shape) for key, value in output.scale_embeddings.items()},
            "local": list(output.local_feature_volume.shape),
            "validity": list(output.local_validity_mask.shape),
        }
    if output.shape != (1, 192) or not torch.isfinite(output).all():
        raise ValueError("current encoder failed its legacy global embedding contract")
    return "legacy_global_v1", {"global": list(output.shape)}


def _estimate_flops(model: nn.Module, forward: Callable[[], object]) -> int:
    total = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            kernel = 1
            for side in module.kernel_size:
                kernel *= side
            total += 2 * output.numel() * kernel * module.in_channels // module.groups
        elif isinstance(module, nn.Linear):
            total += 2 * output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        forward()
    for handle in handles:
        handle.remove()
    return total


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def profile_cell(
    candidate: str,
    radius: int,
    *,
    device: torch.device,
    warmup: int,
    measured: int,
    training_steps: int,
) -> ProfileCell:
    model, forward = _build(candidate, radius, device)
    model.eval()
    with torch.no_grad():
        output = forward()
    contract, shapes = _output_shapes(output)
    flops = _estimate_flops(model, forward)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    with torch.no_grad():
        for _ in range(warmup):
            forward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(measured)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(measured)]
        with torch.no_grad():
            for start, end in zip(starts, ends):
                start.record()
                forward()
                end.record()
        torch.cuda.synchronize()
        latencies = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        peak_inference_allocated = torch.cuda.max_memory_allocated(device)
        peak_inference_reserved = torch.cuda.max_memory_reserved(device)
    else:
        latencies = []
        with torch.no_grad():
            for _ in range(measured):
                started = time.perf_counter()
                forward()
                latencies.append((time.perf_counter() - started) * 1000)
        peak_inference_allocated = 0
        peak_inference_reserved = 0

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        training_output = forward()
        loss = (
            training_output.square().mean()
            if isinstance(training_output, torch.Tensor)
            else training_output.global_embedding.square().mean()
            + training_output.local_feature_volume.square().mean()
        )
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_training_allocated = torch.cuda.max_memory_allocated(device)
        peak_training_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_training_allocated = 0
        peak_training_reserved = 0
    elapsed = time.perf_counter() - started
    return ProfileCell(
        candidate=candidate,
        radius=radius,
        status="completed",
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        checkpoint_bytes=len(buffer.getvalue()),
        flops=flops,
        latency_p50_ms=statistics.median(latencies),
        latency_p95_ms=_percentile(latencies, 0.95),
        peak_inference_allocated_bytes=peak_inference_allocated,
        peak_inference_reserved_bytes=peak_inference_reserved,
        peak_training_allocated_bytes=peak_training_allocated,
        peak_training_reserved_bytes=peak_training_reserved,
        training_examples_per_second=training_steps / elapsed,
        output_shapes=shapes,
        output_contract=contract,
    )


def decide(cells: list[ProfileCell]) -> dict[str, object]:
    by_candidate = {
        candidate: [cell for cell in cells if cell.candidate == candidate]
        for candidate in CANDIDATES
    }
    rejected: dict[str, list[str]] = {}
    retained: list[str] = []
    for candidate, candidate_cells in by_candidate.items():
        if len(candidate_cells) != len(RADII) or any(cell.status != "completed" for cell in candidate_cells):
            rejected[candidate] = ["incomplete_or_failed_profile"]
        else:
            retained.append(candidate)
    sparse = pilot_backbone_capabilities()["sparse_residual"]
    rejected["sparse_residual"] = [sparse.reason or "optional_backend_unavailable"]
    return {
        "decision": "tie",
        "retained": retained[:5],
        "rejected": rejected,
        "quality_claim": False,
        "next_pilot": "P4",
    }


def _failed_cell(candidate: str, radius: int, error: Exception) -> ProfileCell:
    status = "oom" if "out of memory" in str(error).lower() else "failed"
    return ProfileCell(
        candidate=candidate,
        radius=radius,
        status=status,
        parameters=0,
        checkpoint_bytes=0,
        flops=0,
        latency_p50_ms=0,
        latency_p95_ms=0,
        peak_inference_allocated_bytes=0,
        peak_inference_reserved_bytes=0,
        peak_training_allocated_bytes=0,
        peak_training_reserved_bytes=0,
        training_examples_per_second=0,
        output_shapes={},
        output_contract="unavailable",
        error=f"{type(error).__name__}: {error}",
    )


def run(output: Path, *, warmup: int, measured: int, training_steps: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("P3 requires the declared CUDA reference accelerator")
    device = torch.device("cuda:0")
    _warm_cuda_training_runtime(device)
    started = time.perf_counter()
    cells: list[ProfileCell] = []
    for candidate in CANDIDATES:
        for radius in RADII:
            try:
                cell = profile_cell(
                    candidate,
                    radius,
                    device=device,
                    warmup=warmup,
                    measured=measured,
                    training_steps=training_steps,
                )
            except (RuntimeError, ValueError) as error:
                cell = _failed_cell(candidate, radius, error)
                torch.cuda.empty_cache()
            cells.append(cell)
    sparse_reason = pilot_backbone_capabilities()["sparse_residual"].reason
    counts = {
        "completed": sum(cell.status == "completed" for cell in cells),
        "failed": sum(cell.status in {"failed", "oom"} for cell in cells),
        "skipped": len(RADII),
        "cap": (len(CANDIDATES) + 1) * len(RADII),
    }
    report: dict[str, object] = {
        "issue": 279,
        "pilot": "P3",
        "status": "completed" if counts["failed"] == 0 else "failed",
        "integration_base_sha": _git(
            "merge-base", "HEAD", "origin/exp/perception-encoder"
        ),
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": "f64ea0b1b30ce07c28dfe2dc688a56d48a931c0d",
        "dataset_identity": "not_applicable_random_weight_profile",
        "query_identity": "not_applicable_random_weight_profile",
        "configuration": {
            "batch_size": 1,
            "precision": "fp32",
            "compilation": False,
            "warmup_inferences": warmup,
            "measured_inferences": measured,
            "training_steps": training_steps,
            "radii": list(RADII),
            "seed": 0,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "driver": _nvidia_driver_version(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "cells": [asdict(cell) for cell in cells],
        "optional_sparse": {
            "status": "skipped",
            "radii": list(RADII),
            "reason": sparse_reason,
        },
        "trial_counts": counts,
        "accelerator_hours": (time.perf_counter() - started) / 3600,
        "decision_record": decide(cells),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_payload_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measured", type=int, default=500)
    parser.add_argument("--training-steps", type=int, default=20)
    args = parser.parse_args()
    run(
        args.output,
        warmup=args.warmup,
        measured=args.measured,
        training_steps=args.training_steps,
    )


if __name__ == "__main__":
    main()
