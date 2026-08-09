"""Backend objects for explaining trained RLlib policies.

The package starts with dependency-free backend primitives that can be tested
without Ray, a checkpoint, or a CLI command.  The public convenience entry point
is ``explain_policy``; larger integrations should instantiate
``PolicyExplanationBackend`` with explicit collaborators.

Examples
--------
Run a backend explanation with explicitly supplied collaborators::

    report = backend.explain(request)
"""

from theseo_anysearch.rllib.explain.backend import PolicyExplanationBackend, explain_policy
from theseo_anysearch.rllib.explain.explainers import OcclusionExplainer
from theseo_anysearch.rllib.explain.features import FeatureSchema
from theseo_anysearch.rllib.explain.models import (
    ActionScoreTable,
    ExplainedStep,
    ExplanationReport,
    ExplanationRequest,
)
from theseo_anysearch.rllib.explain.reports import ExplanationReportBuilder, ExplanationReportWriter
from theseo_anysearch.rllib.explain.requests import ExplanationRequestFile
from theseo_anysearch.rllib.explain.scenarios import EnvironmentScenario, ObservationScenario
from theseo_anysearch.rllib.explain.scoring import DQNPolicyScorer, LinearMockPolicyScorer, MockPolicyScorer
from theseo_anysearch.rllib.explain.selectors import CollisionStepSelector, ExplicitStepSelector
from theseo_anysearch.rllib.explain.traces import (
    ObservationTrace,
    ObservationTraceStep,
    PolicyEvaluationTraceCollector,
)

__all__ = [
    "ActionScoreTable",
    "CollisionStepSelector",
    "ExplicitStepSelector",
    "ExplainedStep",
    "ExplanationReport",
    "ExplanationReportBuilder",
    "ExplanationReportWriter",
    "ExplanationRequest",
    "ExplanationRequestFile",
    "FeatureSchema",
    "DQNPolicyScorer",
    "LinearMockPolicyScorer",
    "MockPolicyScorer",
    "ObservationTrace",
    "ObservationTraceStep",
    "ObservationScenario",
    "OcclusionExplainer",
    "PolicyExplanationBackend",
    "PolicyEvaluationTraceCollector",
    "EnvironmentScenario",
    "explain_policy",
]
