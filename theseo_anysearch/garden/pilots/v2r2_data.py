"""Native-volume, visible-conditioned empirical patch completions for R0.

This is a declared empirical completion prior, not an exact posterior of the
native generator. Conditional draws are IID given the fixed donor bank. Neither
completed reachability nor forbidden metadata is used to select donors/queries.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib

import numpy as np

from theseo_anysearch.garden.pilots.corpus import GENERATOR_VERSION, V2R2_PROGRAM, _seed, make_pilot_observation
from theseo_anysearch.garden.pilots.io import payload_sha256
from theseo_anysearch.garden.pilots.v2r2_controls import example_from_completions
from theseo_anysearch.garden.splits import GeometryDescriptor


FAMILIES = ("open", "thin_obstacle", "topology", "imported")
BANDS = ("low", "medium", "high")
STRATUM_WIDTHS = {"1-2": 1, "3-5": 3, "6+": 6}
GENERATOR = "native-v2-visible-conditioned-patches-v1"


def seed_for(seed: int, identity: str, purpose: str) -> int:
    return int(payload_sha256([seed, identity, purpose])[:16], 16)


def identity_plan(seed: int, train_per_stratum: int, audit_per_stratum: int, donors: int) -> dict:
    pools = {}
    for pool, count in (("r0_train", train_per_stratum), ("r0_in_domain", audit_per_stratum), ("r0_transfer", audit_per_stratum)):
        records = []
        for stratum in STRATUM_WIDTHS:
            for i in range(2 * count):
                record = GeometryDescriptor(
                    f"pilot-v2r2-s{seed}-{pool}-{stratum}-{i:03d}",
                    FAMILIES[i % 4], BANDS[(i // 4) % 3], "native_procedural_r0",
                )
                records.append({**asdict(record), "stratum": stratum, "configuration": "B" if pool == "r0_transfer" else "A"})
        pools[pool] = records
    for pool, count in (("donors_A", donors), ("donors_B", donors),
                        ("pilot_train", 96), ("pilot_calibration", 24), ("pilot_diagnostic", 24),
                        ("pilot_dev_early", 24), ("pilot_dev_arch", 24), ("pilot_dev_interaction", 24), ("pilot_confirm", 32)):
        pools[pool] = [asdict(GeometryDescriptor(
            f"pilot-v2r2-s{seed}-{pool}-{i:03d}", FAMILIES[i % 4], BANDS[(i // 4) % 3],
            "reserved_v2r2_native_procedural",
        )) for i in range(count)]
    for i, record in enumerate(pools["pilot_confirm"]):
        record["confirmation_group"] = "ordinary" if i < 16 else "heldout_topology" if i < 24 else "heldout_imported"
        if i >= 16:
            record["family"] = "topology" if i < 24 else "imported"
    return pools


def native_volume(record: dict, configuration: str) -> np.ndarray:
    if configuration not in ("A", "B"):
        raise ValueError("unknown native generator configuration")
    descriptor = GeometryDescriptor(**{key: record[key] for key in (
        "geometry_id", "family", "occupancy_band", "source", "parent_split", "confirmation_group",
    )})
    observation = make_pilot_observation(descriptor, 1, radius=8 if configuration == "A" else 16, program=V2R2_PROGRAM)
    if observation.unknown_mask.any():
        raise ValueError("native donor/source must be fully observed before applying audit mask")
    return observation.occupancy.copy() if configuration == "A" else observation.occupancy[::2, ::2, ::2].copy()


def nearest_visible_donors(observed: np.ndarray, unknown: np.ndarray, donors: np.ndarray, k: int) -> np.ndarray:
    if donors.dtype != np.bool_ or donors.ndim != 4 or donors.shape[1:] != observed.shape:
        raise ValueError("donors must be aligned boolean volumes")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(donors):
        raise ValueError("invalid donor neighbour count")
    # Full visible Hamming distance, stable donor-index ties; hidden values and
    # family/density metadata are absent from the conditioning calculation.
    distance = np.count_nonzero(donors[:, ~unknown] != observed[~unknown], axis=1)
    return np.argsort(distance, kind="stable")[:k]


def completed_draws(observed: np.ndarray, unknown: np.ndarray, donors: np.ndarray, *, k: int, count: int, seed: int):
    selected = nearest_visible_donors(observed, unknown, donors, k)
    choices = np.random.default_rng(seed).choice(selected, size=count, replace=True)
    completed = np.repeat(observed[None], count, axis=0)
    completed[:, unknown] = donors[choices][:, unknown]
    return completed, selected, choices


def observed_query(observed: np.ndarray, unknown: np.ndarray, *, seed: int, stratum: str, attempts: int):
    if attempts != 1:
        raise ValueError("this execution uses one label-independent aligned query")
    hidden_planes = np.flatnonzero(unknown.any(axis=(1, 2)))
    left, right = int(hidden_planes[0] - 1), int(hidden_planes[-1] + 1)
    candidates = np.argwhere(~observed[left] & ~observed[right])
    if not len(candidates):
        return None
    rng = np.random.default_rng(seed)
    h, w = (int(x) for x in candidates[rng.integers(len(candidates))])
    # The unique Manhattan-shortest optimistic path is straight across the slab.
    # All its interior cells are unknown, so no completion label sets the bin.
    return (left, h, w), (right, h, w), len(hidden_planes)


def build_context(record: dict, donors: np.ndarray, *, seed: int, k: int, completions: int, query_attempts: int):
    native = native_volume(record, record["configuration"])
    width = STRATUM_WIDTHS[record["stratum"]]
    unknown = np.zeros_like(native)
    lower = 8 - width // 2
    unknown[lower : lower + width] = True
    observed = native & ~unknown
    identity = record["geometry_id"]
    query = observed_query(observed, unknown, seed=seed_for(seed, identity, "query"),
                           stratum=record["stratum"], attempts=query_attempts)
    audit = {"geometry_id": identity, "stratum": record["stratum"], "configuration": record["configuration"],
             "source_sha256": hashlib.sha256(native.tobytes()).hexdigest()}
    if not observed.any():
        return None, {**audit, "status": "no_visible_occupancy", "replacement_drawn": False}
    if query is None:
        return None, {**audit, "status": "no_observed_query", "replacement_drawn": False}
    start, goal, span = query
    completed, selected, choices = completed_draws(observed, unknown, donors, k=k, count=completions,
        seed=seed_for(seed, identity, "completions"))
    forbidden = (
        _seed(GENERATOR_VERSION, V2R2_PROGRAM, identity) / (2**64 - 1),
        *[float(record["family"] == family) for family in FAMILIES],
        *[float(record["occupancy_band"] == band) for band in BANDS],
        float(record["configuration"] == "B"), width / 17,
    )
    example = example_from_completions(
        context_id=identity + ":q0", geometry_id=identity,
        generator_configuration=record["configuration"], bootstrap_stratum=record["family"] + ":" + record["occupancy_band"],
        stratum=record["stratum"], observed_occupancy=observed, unknown=unknown, start=start, goal=goal,
        completions=completed, forbidden_features=forbidden,
    )
    return example, {**audit, "status": "generated", "observed_span": span,
        "observation_sha256": example.observation_sha256, "query_sha256": example.visible_input_sha256,
        "completion_payload_sha256": hashlib.sha256(completed.tobytes()).hexdigest(),
        "donor_indices": [int(x) for x in selected], "sampled_donor_indices": [int(x) for x in choices]}
