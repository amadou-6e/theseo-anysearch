"""Validate a frozen Garden voxel encoder before policy integration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from theseo_anysearch.garden.models.ae import build_encoder


def make_box_grids(count: int, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    grids = np.zeros((count, n, n, n), dtype=np.float32)
    for sample in grids:
        for _ in range(int(rng.integers(1, 5))):
            starts = rng.integers(0, n, size=3)
            stops = np.array([rng.integers(start + 1, n + 1) for start in starts])
            sample[
                starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]
            ] = 1.0
        if rng.random() < 0.5:
            sample[n // 2, n // 2, n // 2] = 1.0
    return torch.from_numpy(grids)


def make_ood_grids(count: int, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    coords = np.indices((n, n, n)).transpose(1, 2, 3, 0)
    grids = np.zeros((count, n, n, n), dtype=np.float32)
    for sample in grids:
        center = rng.uniform(1, n - 2, size=3)
        radius = rng.uniform(0.8, max(1.0, n / 2.0))
        sphere = np.linalg.norm(coords - center, axis=-1) <= radius
        sample[sphere] = 1.0
        axis = int(rng.integers(0, 3))
        index = int(rng.integers(0, n))
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = index
        sample[tuple(slices)] = 1.0 - sample[tuple(slices)]
    return torch.from_numpy(grids)


class ProbeDecoder(nn.Module):
    def __init__(self, latent_dim: int, voxel_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, voxel_count),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


def encode(encoder: nn.Module, grids: torch.Tensor, batch_size: int) -> torch.Tensor:
    encoder.eval()
    outputs = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(grids), batch_size=batch_size):
            outputs.append(encoder(batch))
    return torch.cat(outputs)


def train_decoder(
    latent: torch.Tensor,
    grids: torch.Tensor,
    latent_dim: int,
    epochs: int,
    seed: int,
) -> ProbeDecoder:
    torch.manual_seed(seed)
    decoder = ProbeDecoder(latent_dim, grids[0].numel())
    optimizer = torch.optim.Adam(decoder.parameters(), lr=2e-3)
    targets = grids.flatten(1)
    positives = targets.sum()
    negatives = targets.numel() - positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives).clamp(max=20.0))
    loader = DataLoader(TensorDataset(latent, targets), batch_size=128, shuffle=True)
    decoder.train()
    for _ in range(epochs):
        for features, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(decoder(features), target)
            loss.backward()
            optimizer.step()
    return decoder.eval()


def reconstruction_metrics(
    decoder: nn.Module, latent: torch.Tensor, grids: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        probabilities = torch.sigmoid(decoder(latent))
    target = grids.flatten(1).bool()
    predicted = probabilities >= 0.5
    intersection = (predicted & target).sum().item()
    union = (predicted | target).sum().item()
    true_positive = intersection
    false_positive = (predicted & ~target).sum().item()
    false_negative = (~predicted & target).sum().item()
    return {
        "bce": float(nn.functional.binary_cross_entropy(probabilities, target.float())),
        "iou": true_positive / max(union, 1),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "accuracy": (predicted == target).float().mean().item(),
    }


def fit_linear_probe(
    train_latent: torch.Tensor,
    train_grids: torch.Tensor,
    test_latent: torch.Tensor,
    test_grids: torch.Tensor,
) -> dict[str, float]:
    # Ridge-regularized closed-form probe for occupied fraction.
    x = torch.cat([train_latent, torch.ones(len(train_latent), 1)], dim=1)
    y = train_grids.flatten(1).mean(1, keepdim=True)
    ridge = 1e-3 * torch.eye(x.shape[1])
    weights = torch.linalg.solve(x.T @ x + ridge, x.T @ y)
    test_x = torch.cat([test_latent, torch.ones(len(test_latent), 1)], dim=1)
    truth = test_grids.flatten(1).mean(1, keepdim=True)
    prediction = test_x @ weights
    residual = ((truth - prediction) ** 2).sum()
    total = ((truth - truth.mean()) ** 2).sum()

    # Six center-face occupancy values are directly relevant to movement feasibility.
    n = train_grids.shape[-1]
    c = n // 2
    face_indices = [(c - 1, c, c), (c + 1, c, c), (c, c - 1, c), (c, c + 1, c), (c, c, c - 1), (c, c, c + 1)]
    train_faces = torch.stack([train_grids[:, i, j, k] for i, j, k in face_indices], dim=1)
    face_weights = torch.linalg.solve(x.T @ x + ridge, x.T @ train_faces)
    test_faces = torch.stack([test_grids[:, i, j, k] for i, j, k in face_indices], dim=1)
    face_prediction = (test_x @ face_weights) >= 0.5
    return {
        "fill_fraction_r2": float(1.0 - residual / total.clamp_min(1e-12)),
        "face_occupancy_accuracy": float((face_prediction == test_faces.bool()).float().mean()),
    }


def changed_voxel_sensitivity(encoder: nn.Module, grids: torch.Tensor) -> dict[str, float]:
    changed = grids.clone()
    n = grids.shape[-1]
    changed[:, n // 2, n // 2, min(n - 1, n // 2 + 1)] = 1.0 - changed[:, n // 2, n // 2, min(n - 1, n // 2 + 1)]
    original_z = encode(encoder, grids[:256], 256)
    changed_z = encode(encoder, changed[:256], 256)
    delta = torch.linalg.vector_norm(changed_z - original_z, dim=1)
    scale = torch.linalg.vector_norm(original_z, dim=1).clamp_min(1e-8)
    return {
        "single_voxel_delta_l2_mean": float(delta.mean()),
        "single_voxel_relative_delta_mean": float((delta / scale).mean()),
    }


def latency_ms(
    encoder: nn.Module, n: int, device: str = "cpu", repeats: int = 100
) -> dict[str, float]:
    encoder = encoder.to(device)
    sample = torch.zeros(1, n, n, n, device=device)
    synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
    encoder.eval()
    with torch.no_grad():
        for _ in range(10):
            encoder(sample)
        synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            encoder(sample)
        synchronize()
        single = (time.perf_counter() - started) * 1000.0 / repeats
        batch = sample.repeat(128, 1, 1, 1)
        for _ in range(5):
            encoder(batch)
        synchronize()
        started = time.perf_counter()
        for _ in range(10):
            encoder(batch)
        synchronize()
        batch_latency = (time.perf_counter() - started) * 100.0
    encoder.cpu()
    return {"single_ms": single, "batch_128_ms": batch_latency}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("r2", "r3"), default="r2")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Garden model directory containing meta.json and encoder.pt.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("runtime", "encoder_validation.json"))
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(1)
    configs = {
        "r2": {"radius": 2, "channels": [32, 64, 64], "latent_dim": 64},
        "r3": {"radius": 3, "channels": [32, 64, 128], "latent_dim": 128},
    }
    if args.model_dir is not None:
        meta = json.loads((args.model_dir / "meta.json").read_text())
        cfg = {
            "radius": meta["box_radius"],
            "channels": meta["conv_channels"],
            "latent_dim": meta["latent_dim"],
            "architecture": meta["architecture"],
            "profile": meta.get("profile"),
            "pooling": meta.get("pooling", "global_avg"),
            "residual_blocks": meta.get("residual_blocks"),
        }
        checkpoint = args.model_dir / "encoder.pt"
        model_name = f"{meta['name']}:{meta.get('tag') or 'untagged'}"
    else:
        cfg = {
            **configs[args.preset],
            "architecture": "voxel_box_3dcnn",
            "profile": None,
            "pooling": "global_avg",
            "residual_blocks": None,
        }
        checkpoint = Path(
            "theseo_anysearch", "garden", "pretrained", args.preset, "v1", "encoder.pt"
        )
        model_name = f"{args.preset}:v1"
    n = 2 * cfg["radius"] + 1
    train_grids = make_box_grids(2400, n, 42)
    test_grids = make_box_grids(600, n, 43)
    ood_grids = make_ood_grids(600, n, 44)

    pretrained = build_encoder(
        cfg["architecture"], n, cfg["channels"], cfg["latent_dim"], cfg["profile"],
        cfg["pooling"], cfg["residual_blocks"]
    )
    pretrained.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    random_encoder = build_encoder(
        cfg["architecture"], n, cfg["channels"], cfg["latent_dim"], cfg["profile"],
        cfg["pooling"], cfg["residual_blocks"]
    )

    results: dict[str, object] = {
        "preset": model_name,
        "grid_shape": [n, n, n],
        "latent_dim": cfg["latent_dim"],
        "encoder_parameters": sum(parameter.numel() for parameter in pretrained.parameters()),
        "estimated_parameter_bytes_fp32": sum(parameter.numel() for parameter in pretrained.parameters()) * 4,
    }
    for name, encoder in (("pretrained", pretrained), ("random", random_encoder)):
        train_latent = encode(encoder, train_grids, 256)
        test_latent = encode(encoder, test_grids, 256)
        ood_latent = encode(encoder, ood_grids, 256)
        decoder = train_decoder(train_latent, train_grids, cfg["latent_dim"], args.epochs, 42)
        results[name] = {
            "in_distribution_reconstruction": reconstruction_metrics(decoder, test_latent, test_grids),
            "ood_reconstruction": reconstruction_metrics(decoder, ood_latent, ood_grids),
            "linear_probes": fit_linear_probe(train_latent, train_grids, test_latent, test_grids),
            "sensitivity": changed_voxel_sensitivity(encoder, test_grids),
            "cpu_latency": latency_ms(encoder, n),
            "gpu_latency": (
                latency_ms(encoder, n, device="cuda")
                if torch.cuda.is_available()
                else None
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
