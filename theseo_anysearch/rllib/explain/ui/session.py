"""Long-lived policy restoration and immediate explanation session."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from theseo_anysearch.rllib.explain.backend import PolicyExplanationBackend
from theseo_anysearch.rllib.explain.explainers import OcclusionExplainer
from theseo_anysearch.rllib.explain.models import ExplanationReport, ExplanationRequest
from theseo_anysearch.rllib.explain.scenarios import validate_observation
from theseo_anysearch.rllib.explain.service import PolicyExplanationService
from theseo_anysearch.rllib.explain.traces import ObservationTrace, ObservationTraceStep


class InteractiveExplanationSession:
    """Restore a policy once and repeatedly explain edited observations."""

    def __init__(self, run_dir: Path, checkpoint: str = "latest") -> None:
        self.service = PolicyExplanationService(run_dir, checkpoint=checkpoint)
        self.checkpoint = checkpoint
        self.observation_space = self.service.observation_space()
        self._background = self.service.initial_observation()

    def initial_observation(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Return a real environment observation suitable as an editing baseline."""

        return self.service.initial_observation(seed)

    def explain(
        self,
        observation: Mapping[str, object],
        chosen_action: str | int = "policy",
    ) -> ExplanationReport:
        """Validate and explain one edited observation without writing artifacts."""

        validated = validate_observation(
            {name: np.asarray(value).tolist() for name, value in observation.items()},
            self.observation_space,
        )
        scores = self.service.scorer.score_all([validated])
        action = int(np.argmax(scores.values[0])) if chosen_action == "policy" else int(chosen_action)
        if action < 0 or action >= scores.action_count():
            raise ValueError(f"chosen action {action} is outside the supported action space")
        trace = ObservationTrace(
            [
                ObservationTraceStep(
                    step=0,
                    observation=validated,
                    action=action,
                    reward=0.0,
                    cursor_before=(0.0, 0.0, 0.0),
                    cursor_after=(0.0, 0.0, 0.0),
                    done=False,
                    collision=None,
                )
            ],
            algorithm="dqn",
        )
        schema = self.service.feature_schema(validated)
        request = ExplanationRequest(
            run_ref=str(self.service.run_dir),
            checkpoint=self.checkpoint,
            trajectory="interactive-observation",
            focus="all",
            max_steps=1,
            scenario_validity="not_environment_validated",
        )
        return PolicyExplanationBackend(
            schema,
            trace,
            self.service.scorer,
            explainer=OcclusionExplainer(
                schema, self.service.scorer, [self._background]
            ),
        ).explain(request)
