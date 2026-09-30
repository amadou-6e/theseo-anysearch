"""Backend orchestration for policy explanation runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from theseo_anysearch.rllib.explain.explainers import Explainer, OcclusionExplainer
from theseo_anysearch.rllib.explain.features import FeatureSchema
from theseo_anysearch.rllib.explain.models import ExplanationReport, ExplanationRequest
from theseo_anysearch.rllib.explain.reports import ExplanationReportBuilder, ExplanationReportWriter
from theseo_anysearch.rllib.explain.scoring import PolicyScorer
from theseo_anysearch.rllib.explain.selectors import CollisionStepSelector, ExplicitStepSelector, StepSelector
from theseo_anysearch.rllib.explain.traces import ObservationTrace


class PolicyExplanationBackend:
    """Coordinate trace selection, scoring, attribution, and report writing.

    Parameters
    ----------
    schema : FeatureSchema
        Observation feature schema.
    trace : ObservationTrace
        Source trace containing pre-action observations.
    scorer : PolicyScorer
        Policy action scorer.
    selector : StepSelector | None, optional
        Step selector.  Defaults to collision selection.
    explainer : Explainer | None, optional
        Attribution backend.  Defaults to grouped occlusion.
    report_writer : ExplanationReportWriter | None, optional
        Optional writer for report artifacts.
    """

    def __init__(
        self,
        schema: FeatureSchema,
        trace: ObservationTrace,
        scorer: PolicyScorer,
        selector: StepSelector | None = None,
        explainer: Explainer | None = None,
        report_writer: ExplanationReportWriter | None = None,
    ) -> None:
        self._schema = schema
        self._trace = trace
        self._scorer = scorer
        self._selector = selector or CollisionStepSelector()
        self._explainer = explainer or OcclusionExplainer(schema, scorer, trace.observations())
        self._report_writer = report_writer

    def explain(self, request: ExplanationRequest) -> ExplanationReport:
        """Run the backend explanation workflow."""

        selector = self._selector_for_request(request)
        selected_steps = selector.select(self._trace, request.max_steps)
        if not selected_steps:
            raise ValueError(f"no trace steps matched explanation focus {request.focus!r}")

        selected_observations = [self._trace.step(index).observation for index in selected_steps]
        score_table = self._scorer.score_all(selected_observations)
        state_value_rows = self._scorer.state_values(selected_observations)
        score_rows = {
            step_index: score_table.row(row_index)
            for row_index, step_index in enumerate(selected_steps)
        }

        builder = ExplanationReportBuilder(self._schema, self._explainer.method)
        attributions = self._explain_selected_steps(builder, selected_steps, score_rows)
        report = builder.build(
            request,
            self._trace,
            selected_steps,
            score_rows,
            attributions,
            score_table.score_type,
            state_values=(
                None
                if state_value_rows is None
                else {
                    step_index: float(state_value_rows[row_index])
                    for row_index, step_index in enumerate(selected_steps)
                }
            ),
            output_dir=(
                self._report_writer.output_dir if self._report_writer is not None else None
            ),
        )
        if self._report_writer is not None:
            self._report_writer.write(report)
            self._report_writer.write_observations(self._trace, selected_steps)
        return report

    def _selector_for_request(self, request: ExplanationRequest) -> StepSelector:
        """Return the selector implied by the request."""

        if request.focus == "explicit":
            return ExplicitStepSelector(request.explicit_steps)
        if request.focus == "collisions":
            return self._selector
        if request.focus == "all":
            return ExplicitStepSelector(tuple(range(len(self._trace))))
        raise ValueError(f"unsupported explanation focus: {request.focus}")

    def _explain_selected_steps(
        self,
        builder: ExplanationReportBuilder,
        selected_steps: list[int],
        score_rows: dict[int, np.ndarray],
    ) -> dict[int, dict[str, float]]:
        """Run the explainer for every selected step."""

        attributions: dict[int, dict[str, float]] = {}
        for step_index in selected_steps:
            trace_step = self._trace.step(step_index)
            scores = score_rows[step_index]
            best_safe_action = builder.best_safe_action_for_observation(
                trace_step.observation,
                scores,
                excluded_action=trace_step.action,
            )
            attributions[step_index] = self._explainer.explain_margin(
                trace_step.observation,
                trace_step.action,
                best_safe_action,
            )
        return attributions


def explain_policy(
    request: ExplanationRequest,
    *,
    schema: FeatureSchema,
    trace: ObservationTrace,
    scorer: PolicyScorer,
    output_dir: Path | None = None,
) -> ExplanationReport:
    """Convenience wrapper for backend-only explanation runs."""

    writer = ExplanationReportWriter(output_dir) if output_dir is not None else None
    backend = PolicyExplanationBackend(
        schema=schema,
        trace=trace,
        scorer=scorer,
        report_writer=writer,
    )
    return backend.explain(request)
