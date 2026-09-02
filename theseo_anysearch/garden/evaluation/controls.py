"""Control targets and embedding-necessity ablations for frozen probes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import torch

from theseo_anysearch.garden.models.outputs import EncoderMetadata, EncoderOutput


def _rank(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def control_target_assignment(
    labels: torch.Tensor,
    *,
    strata: Sequence[object],
    split: str,
    geometry_ids: Sequence[str],
    query_ids: Sequence[str],
    control_seed: int,
) -> torch.Tensor:
    """Deterministically permute labels within strata without crossing a split."""

    rows = labels.shape[0]
    if not split:
        raise ValueError("control-target split cannot be empty")
    if any(len(values) != rows for values in (strata, geometry_ids, query_ids)):
        raise ValueError("control-target metadata must align with label rows")
    if len(set(query_ids)) != rows:
        raise ValueError("control-target query IDs must be unique within a split")

    groups: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        groups.setdefault(_rank(stratum), []).append(index)
    assigned = torch.empty_like(labels)
    for stratum_hash, indices in groups.items():
        source = sorted(
            indices,
            key=lambda index: _rank(
                {
                    "role": "source",
                    "split": split,
                    "stratum": stratum_hash,
                    "geometry_id": geometry_ids[index],
                    "query_id": query_ids[index],
                    "seed": control_seed,
                }
            ),
        )
        destination = sorted(
            indices,
            key=lambda index: _rank(
                {
                    "role": "destination",
                    "split": split,
                    "stratum": stratum_hash,
                    "geometry_id": geometry_ids[index],
                    "query_id": query_ids[index],
                    "seed": control_seed,
                }
            ),
        )
        for source_index, destination_index in zip(source, destination):
            assigned[destination_index] = labels[source_index]
    return assigned


def _permutation_across_geometries(geometry_ids: Sequence[str], *, seed: int) -> torch.Tensor:
    if len(set(geometry_ids)) < 2:
        raise ValueError("shuffled embeddings require at least two geometry IDs")
    ordered = sorted(
        range(len(geometry_ids)),
        key=lambda index: _rank(
            {"seed": seed, "geometry_id": geometry_ids[index], "row": index}
        ),
    )
    for offset in range(1, len(ordered)):
        source_for_destination = {
            ordered[index]: ordered[(index + offset) % len(ordered)]
            for index in range(len(ordered))
        }
        if all(
            geometry_ids[destination] != geometry_ids[source]
            for destination, source in source_for_destination.items()
        ):
            return torch.tensor(
                [source_for_destination[index] for index in range(len(ordered))],
                dtype=torch.long,
            )
    raise ValueError("batch geometry multiplicities do not permit cross-geometry shuffling")


def _index(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(0, permutation.to(device=tensor.device))


def zero_embedding_output(encoded: EncoderOutput) -> EncoderOutput:
    """Zero every learned feature while retaining query-independent validity metadata."""

    return EncoderOutput(
        global_embedding=torch.zeros_like(encoded.global_embedding),
        scale_embeddings={
            stride: torch.zeros_like(embedding)
            for stride, embedding in encoded.scale_embeddings.items()
        },
        local_feature_volume=torch.zeros_like(encoded.local_feature_volume),
        local_validity_mask=encoded.local_validity_mask,
        metadata=encoded.metadata,
    ).validate(embedding_dim=encoded.global_embedding.shape[1])


def shuffled_embedding_output(
    encoded: EncoderOutput, geometry_ids: Sequence[str], *, seed: int
) -> EncoderOutput:
    """Move complete representations only between rows from different geometries."""

    if len(geometry_ids) != encoded.global_embedding.shape[0]:
        raise ValueError("geometry IDs must align with the encoded batch")
    permutation = _permutation_across_geometries(geometry_ids, seed=seed)
    return EncoderOutput(
        global_embedding=_index(encoded.global_embedding, permutation),
        scale_embeddings={
            stride: _index(embedding, permutation)
            for stride, embedding in encoded.scale_embeddings.items()
        },
        local_feature_volume=_index(encoded.local_feature_volume, permutation),
        local_validity_mask=_index(encoded.local_validity_mask, permutation),
        metadata=EncoderMetadata(
            active_strides=encoded.metadata.active_strides,
            validity_fractions=_index(encoded.metadata.validity_fractions, permutation),
        ),
    ).validate(embedding_dim=encoded.global_embedding.shape[1])


@dataclass(frozen=True)
class ControlEvaluation:
    """Normalized selectivity and embedding-necessity result for one metric."""

    real_score: float
    control_target_score: float
    zero_embedding_score: float
    shuffled_embedding_score: float
    selectivity: float
    embedding_necessity: float
    passes_selectivity: bool
    passes_embedding_necessity: bool


def evaluate_controls(
    *,
    real_score: float,
    control_target_score: float,
    zero_embedding_score: float,
    shuffled_embedding_score: float,
    selectivity_min: float,
    embedding_necessity_min: float,
) -> ControlEvaluation:
    """Evaluate paired controls after all metrics are normalized higher-is-better."""

    scores = (real_score, control_target_score, zero_embedding_score, shuffled_embedding_score)
    if any(score < 0 or score > 1 for score in scores):
        raise ValueError("control scores must be normalized to [0, 1]")
    if selectivity_min < 0 or embedding_necessity_min < 0:
        raise ValueError("control thresholds cannot be negative")
    selectivity = real_score - control_target_score
    necessity = real_score - max(zero_embedding_score, shuffled_embedding_score)
    return ControlEvaluation(
        real_score=real_score,
        control_target_score=control_target_score,
        zero_embedding_score=zero_embedding_score,
        shuffled_embedding_score=shuffled_embedding_score,
        selectivity=selectivity,
        embedding_necessity=necessity,
        passes_selectivity=selectivity >= selectivity_min,
        passes_embedding_necessity=necessity >= embedding_necessity_min,
    )
