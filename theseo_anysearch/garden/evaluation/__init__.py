"""Encoder-only evaluation tools for perception pilot experiments."""

from .ceilings import (
    CeilingEstimate,
    bayes_error_direct,
    bayes_error_knn,
    bayes_error_mst,
    ceiling_effective_rank_fraction,
    classification_metric_ceiling,
    metric_ceiling_method,
    regression_metric_ceiling,
)
from .controls import (
    ControlEvaluation,
    control_target_assignment,
    evaluate_controls,
    shuffled_embedding_output,
    zero_embedding_output,
)
from .probes import (
    CoordinateProbe,
    CrossFitFold,
    GlobalLinearProbe,
    PairTopologyProbe,
    TopologyDecoder,
    encoder_state_sha256,
    extract_frozen,
    make_cross_fit_folds,
    train_probe_step,
)

__all__ = [
    "CeilingEstimate",
    "ControlEvaluation",
    "CoordinateProbe",
    "CrossFitFold",
    "GlobalLinearProbe",
    "PairTopologyProbe",
    "TopologyDecoder",
    "bayes_error_direct",
    "bayes_error_knn",
    "bayes_error_mst",
    "ceiling_effective_rank_fraction",
    "classification_metric_ceiling",
    "control_target_assignment",
    "encoder_state_sha256",
    "metric_ceiling_method",
    "regression_metric_ceiling",
    "evaluate_controls",
    "extract_frozen",
    "make_cross_fit_folds",
    "shuffled_embedding_output",
    "train_probe_step",
    "zero_embedding_output",
]
