"""Probe-triviality instrumentation for the amended P0C calibration (F2).

P0C failed because PCA / fixed-random-projection of the raw voxel input already
solve ``occupied_iou`` (IoU 1.0) and ``reachability_auprc`` (0.93): the probe
was measuring the query/input, not the representation.

This module quantifies that with two well-established diagnostics, comparing the
real embedding against a *null input* (zeros, or coordinates only):

- pointwise-V-information (Ethayarajh, Choi, Swayamdipta 2022): usable
  information a small probe family can extract about the label. A task the null
  input already solves has ``V-info(embedding) approx V-info(null)``;
- online / block minimum description length (Voita & Titov 2020): codelength of
  the labels given the input. Trivial tasks compress to a tiny codelength for
  any input.

``assess_triviality`` returns numbers shaped for
``contracts.TrivialityCheck``; an active revised anchor requires
``pvi_gain >= min_pvi_gain``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

_LN2 = float(np.log(2.0))
DEFAULT_MIN_PVI_GAIN = 0.05
_PROBE_STEPS = 300
_PROBE_LR = 0.05
_MDL_BLOCKS = 8


@dataclass(frozen=True)
class TrivialityResult:
    """Usable-information and codelength evidence for one probe task."""

    null_input: str
    task_type: str
    pvi_embedding: float
    pvi_null: float
    pvi_gain: float
    mdl_embedding_bits: float
    mdl_null_bits: float
    min_pvi_gain: float
    passes: bool


def _as_matrix(values: np.ndarray) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("probe inputs must be (N,) or (N, D)")
    tensor = torch.from_numpy(array).double()
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (tensor - mean) / std


def _fit_linear_probe(
    x: torch.Tensor, y: torch.Tensor, *, task_type: str
) -> nn.Module:
    torch.manual_seed(0)
    out = 1 if task_type in ("binary", "regression") else int(y.max().item()) + 1
    probe = nn.Linear(x.shape[1], out).double()
    optim = torch.optim.Adam(probe.parameters(), lr=_PROBE_LR)
    for _ in range(_PROBE_STEPS):
        optim.zero_grad()
        logits = probe(x)
        loss = _probe_loss(logits, y, task_type)
        loss.backward()
        optim.step()
    return probe


def _probe_loss(logits: torch.Tensor, y: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "binary":
        return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), y.double())
    if task_type == "regression":
        # Gaussian NLL with a fixed unit variance -> MSE up to a constant.
        return nn.functional.mse_loss(logits.squeeze(-1), y.double())
    return nn.functional.cross_entropy(logits, y.long())


def _codelength_bits(
    probe: nn.Module, x: torch.Tensor, y: torch.Tensor, *, task_type: str
) -> float:
    """Held-out negative log likelihood of the labels in bits."""

    with torch.no_grad():
        logits = probe(x)
        if task_type == "binary":
            nll = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), y.double(), reduction="sum"
            )
            return float(nll / _LN2)
        if task_type == "regression":
            resid = logits.squeeze(-1) - y.double()
            variance = resid.var(unbiased=False).clamp_min(1e-6)
            nll = 0.5 * (
                torch.log(2 * torch.pi * variance) + resid.pow(2) / variance
            ).sum()
            return float(nll / _LN2)
        nll = nn.functional.cross_entropy(logits, y.long(), reduction="sum")
        return float(nll / _LN2)


def _label_entropy_bits(y: torch.Tensor, *, task_type: str) -> float:
    if task_type == "regression":
        variance = y.double().var(unbiased=False).clamp_min(1e-6)
        per_sample = 0.5 * float(torch.log(2 * torch.pi * variance) + 1.0) / _LN2
        return per_sample * y.shape[0]
    counts = torch.bincount(y.long(), minlength=int(y.max().item()) + 1).double()
    probs = (counts / counts.sum()).clamp_min(1e-12)
    per_sample = float(-(probs * torch.log(probs)).sum() / _LN2)
    return per_sample * y.shape[0]


def _v_information_bits(
    features: np.ndarray, labels: np.ndarray, *, task_type: str, seed: int
) -> float:
    """Held-out V-information: label entropy minus probe codelength, in bits/sample."""

    x = _as_matrix(features)
    y = torch.from_numpy(np.asarray(labels))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(x.shape[0], generator=generator)
    split = int(0.7 * x.shape[0])
    train, test = permutation[:split], permutation[split:]
    probe = _fit_linear_probe(x[train], y[train], task_type=task_type)
    entropy = _label_entropy_bits(y[test], task_type=task_type)
    codelength = _codelength_bits(probe, x[test], y[test], task_type=task_type)
    return (entropy - codelength) / max(1, test.shape[0])


def _online_mdl_bits(
    features: np.ndarray, labels: np.ndarray, *, task_type: str, seed: int
) -> float:
    """Block/online codelength: uniform-code the first block, then predict forward."""

    x = _as_matrix(features)
    y = torch.from_numpy(np.asarray(labels))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(x.shape[0], generator=generator)
    x, y = x[permutation], y[permutation]
    edges = np.linspace(0, x.shape[0], _MDL_BLOCKS + 1, dtype=int)
    total = _label_entropy_bits(y[edges[0] : edges[1]], task_type=task_type)
    for index in range(1, _MDL_BLOCKS):
        train_end = edges[index]
        block = slice(edges[index], edges[index + 1])
        probe = _fit_linear_probe(x[:train_end], y[:train_end], task_type=task_type)
        total += _codelength_bits(probe, x[block], y[block], task_type=task_type)
    return float(total)


def assess_triviality(
    embedding_features: np.ndarray,
    null_features: np.ndarray,
    labels: np.ndarray,
    *,
    task_type: str,
    null_input: str,
    min_pvi_gain: float = DEFAULT_MIN_PVI_GAIN,
    seed: int = 0,
) -> TrivialityResult:
    """Compare usable information in the embedding against a null input."""

    if task_type not in ("binary", "regression", "multiclass"):
        raise ValueError(f"unsupported task_type {task_type!r}")
    if null_input not in ("zeros", "coordinates_only"):
        raise ValueError(f"unsupported null_input {null_input!r}")
    labels = np.asarray(labels)
    if len(embedding_features) != len(labels) or len(null_features) != len(labels):
        raise ValueError("embedding, null, and labels must be aligned")

    pvi_embedding = _v_information_bits(
        embedding_features, labels, task_type=task_type, seed=seed
    )
    pvi_null = _v_information_bits(null_features, labels, task_type=task_type, seed=seed)
    mdl_embedding = _online_mdl_bits(
        embedding_features, labels, task_type=task_type, seed=seed
    )
    mdl_null = _online_mdl_bits(null_features, labels, task_type=task_type, seed=seed)
    gain = pvi_embedding - pvi_null
    return TrivialityResult(
        null_input=null_input,
        task_type=task_type,
        pvi_embedding=float(pvi_embedding),
        pvi_null=float(pvi_null),
        pvi_gain=float(gain),
        mdl_embedding_bits=float(max(mdl_embedding, 1e-6)),
        mdl_null_bits=float(max(mdl_null, 1e-6)),
        min_pvi_gain=float(min_pvi_gain),
        passes=bool(gain >= min_pvi_gain),
    )
