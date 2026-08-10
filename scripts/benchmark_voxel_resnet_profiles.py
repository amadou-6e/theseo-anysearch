"""Measure Garden voxel encoder size and warm CPU forward latency."""
from __future__ import annotations

import argparse
import time

import torch

from theseo_anysearch.garden.models.ae import VoxelEncoder3D, VoxelResNet3D


def _latency_ms(model: torch.nn.Module, batch: int, n: int, iterations: int) -> float:
    inputs = torch.zeros(batch, n, n, n)
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            model(inputs)
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
    return (time.perf_counter() - start) * 1_000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    models = {
        "r2:v1 architecture": VoxelEncoder3D(5, [32, 64, 64], 64),
        "voxel_resnet_b0": VoxelResNet3D(5, "b0"),
        "voxel_resnet_b1": VoxelResNet3D(5, "b1"),
    }
    print("model,parameters,batch1_ms,batch128_ms")
    for name, model in models.items():
        parameters = sum(parameter.numel() for parameter in model.parameters())
        batch1 = _latency_ms(model, 1, 5, args.iterations)
        batch128 = _latency_ms(model, 128, 5, args.iterations)
        print(f"{name},{parameters},{batch1:.3f},{batch128:.3f}")


if __name__ == "__main__":
    main()
