"""Attribution backends for policy explanations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np

from theseo_anysearch.rllib.explain.features import FeatureSchema
from theseo_anysearch.rllib.explain.scoring import PolicyScorer


class Explainer(ABC):
    """Base class for feature attribution backends."""

    method = "unknown"

    @abstractmethod
    def explain_margin(
        self,
        observation: Mapping[str, np.ndarray],
        chosen_action: int,
        best_safe_action: int,
    ) -> dict[str, float]:
        """Explain the margin between chosen and best safe action."""


class OcclusionExplainer(Explainer):
    """Dependency-free grouped occlusion explainer.

    Parameters
    ----------
    schema : FeatureSchema
        Observation flattening schema.
    scorer : PolicyScorer
        Policy action scorer.
    background : Sequence[Mapping[str, np.ndarray]]
        Background observations used as mask values.
    """

    method = "occlusion"

    def __init__(
        self,
        schema: FeatureSchema,
        scorer: PolicyScorer,
        background: Sequence[Mapping[str, np.ndarray]],
    ) -> None:
        self._schema = schema
        self._scorer = scorer
        self._baseline = self._build_baseline(background)

    def _build_baseline(self, background: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Build mean background values for every observation group."""

        if not background:
            raise ValueError("occlusion explainer requires at least one background observation")
        flat = self._schema.flatten_batch(background)
        mean_row = flat.mean(axis=0)
        return self._schema.unflatten(mean_row)

    def explain_margin(
        self,
        observation: Mapping[str, np.ndarray],
        chosen_action: int,
        best_safe_action: int,
    ) -> dict[str, float]:
        """Return grouped occlusion attributions for an action-score margin."""

        original_margin = self._margin(observation, chosen_action, best_safe_action)
        attributions: dict[str, float] = {}
        for group in self._schema.groups:
            masked = self._masked_observation(observation)
            masked[group.name] = self._baseline[group.name].copy()
            masked_margin = self._margin(masked, chosen_action, best_safe_action)
            attributions[group.name] = float(original_margin - masked_margin)
        if "local_grid" in observation:
            direction = self._schema.action_directions[chosen_action]
            grid_index = self._schema.local_grid_index(direction)
            attributions["chosen_destination_cell"] = self._single_feature_attribution(
                observation,
                "local_grid",
                grid_index,
                original_margin,
                chosen_action,
                best_safe_action,
            )
        elif "ray_hits" in observation and "ray_hit_types" in observation:
            attributions["chosen_ray_hit"] = self._single_feature_attribution(
                observation, "ray_hits", chosen_action, original_margin,
                chosen_action, best_safe_action,
            )
            attributions["chosen_ray_type"] = self._single_feature_attribution(
                observation, "ray_hit_types", chosen_action, original_margin,
                chosen_action, best_safe_action,
            )
        return attributions

    def _masked_observation(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return a mutable float32 copy of an observation."""

        return {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in observation.items()
        }

    def _margin(
        self,
        observation: Mapping[str, np.ndarray],
        chosen_action: int,
        best_safe_action: int,
    ) -> float:
        """Return chosen-action score minus best-safe-action score."""

        scores = self._scorer.score_all([observation]).values[0]
        return float(scores[chosen_action] - scores[best_safe_action])

    def _single_feature_attribution(
        self,
        observation: Mapping[str, np.ndarray],
        group_name: str,
        feature_index: int,
        original_margin: float,
        chosen_action: int,
        best_safe_action: int,
    ) -> float:
        """Return occlusion attribution for one action-aligned feature."""

        masked = self._masked_observation(observation)
        masked[group_name][feature_index] = self._baseline[group_name][feature_index]
        masked_margin = self._margin(masked, chosen_action, best_safe_action)
        return float(original_margin - masked_margin)
