from pathlib import Path

import pytest

from theseo_anysearch.experiments.custom_metrics import (
    CustomMetricError,
    EvaluationContext,
    compute_custom_metrics,
    copy_metric_sources,
    discover_metric_sources,
    load_metric_providers,
    write_metric_manifest,
)


def _write_metric(path: Path, body: str = "return {'score': 2.5}") -> None:
    path.write_text(f"def compute_metrics(context):\n    {body}\n", encoding="utf-8")


def test_specific_module_overrides_shared_fallback(tmp_path: Path) -> None:
    config = tmp_path.joinpath("trial.yaml")
    config.write_text("experiment: {}\n", encoding="utf-8")
    shared = tmp_path.joinpath("evaluation_metrics.py")
    specific = tmp_path.joinpath("evaluation_metrics.trial.py")
    _write_metric(shared, "return {'shared': 1.0}")
    _write_metric(specific)

    assert discover_metric_sources(config)["evaluation"] == specific
    provider = load_metric_providers(config).evaluation
    context = EvaluationContext(
        iteration=1,
        episodes=(),
        standard_metrics={},
        env_config={},
        final_infos=(),
    )
    assert compute_custom_metrics(provider, context, reserved_names=set()) == {
        "evaluation_score": 2.5
    }


def test_copy_and_manifest_preserve_metric_source(tmp_path: Path) -> None:
    source = tmp_path.joinpath("source")
    destination = tmp_path.joinpath("run")
    source.mkdir()
    config = source.joinpath("trial.yaml")
    config.write_text("experiment: {}\n", encoding="utf-8")
    _write_metric(source.joinpath("training_metrics.py"))

    copied = copy_metric_sources(config, destination)
    providers = load_metric_providers(destination.joinpath("trial.yaml"))
    manifest = write_metric_manifest(providers, destination)

    copied_source = copied["training"].read_text(encoding="utf-8")
    assert copied_source.startswith("def compute_metrics")
    assert manifest is not None
    assert "sha256" in manifest.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "body, message",
    [
        ("return {'bad-name': 1.0}", "identifiers"),
        ("return {'score': float('nan')}", "finite"),
        ("return {'score': True}", "numeric"),
    ],
)
def test_invalid_metric_results_fail_fast(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    config = tmp_path.joinpath("trial.yaml")
    config.write_text("experiment: {}\n", encoding="utf-8")
    _write_metric(tmp_path.joinpath("evaluation_metrics.py"), body)
    provider = load_metric_providers(config).evaluation
    context = EvaluationContext(
        iteration=1,
        episodes=(),
        standard_metrics={},
        env_config={},
        final_infos=(),
    )

    with pytest.raises(CustomMetricError, match=message):
        compute_custom_metrics(provider, context, reserved_names=set())
