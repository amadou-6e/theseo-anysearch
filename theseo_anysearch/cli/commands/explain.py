"""Command-line entry point for reproducible policy explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from theseo_anysearch.rllib.explain.requests import (
    AttributionSettings,
    ExplanationRequestFile,
    OutputSettings,
    RequestSource,
    load_request_file,
)
from theseo_anysearch.rllib.explain.service import (
    PolicyExplanationService,
    resolve_run_dir,
)


def run_explain(
    run: str,
    checkpoint: str,
    trace: Optional[str],
    scenario: Optional[Path],
    request: Optional[Path],
    method: Optional[str],
    focus: Optional[str],
    steps: Optional[str],
    max_steps: Optional[int],
    background: Optional[str],
    output: Optional[Path],
    seed: Optional[int],
) -> Path:
    """Resolve CLI/request values, execute attribution, and return its directory."""

    values = load_request_file(request) if request is not None else ExplanationRequestFile(
        checkpoint=checkpoint,
        source=RequestSource(trace=trace, scenario=scenario),
    )
    if request is not None and (trace is not None or scenario is not None):
        raise ValueError("--request cannot be combined with --trace or --scenario")
    resolved_method = method or values.explanation.method
    if resolved_method != "occlusion":
        raise ValueError("only the 'occlusion' explanation method is supported")
    resolved_focus = focus or values.explanation.focus
    explicit_steps = values.explanation.steps
    if steps is not None:
        explicit_steps = tuple(int(value.strip()) for value in steps.split(",") if value.strip())
    attribution = AttributionSettings(
        method=resolved_method,
        focus=resolved_focus,
        max_steps=max_steps or values.explanation.max_steps,
        background=background or values.explanation.background,
        steps=explicit_steps,
    )
    output_settings = OutputSettings(directory=output or values.output.directory)
    run_dir = resolve_run_dir(run)
    service = PolicyExplanationService(
        run_dir,
        checkpoint=checkpoint if request is None else values.checkpoint,
    )
    source = values.source
    common = {
        "focus": attribution.focus,
        "max_steps": attribution.max_steps,
        "explicit_steps": attribution.steps,
        "background": attribution.background,
        "output_dir": output_settings.directory,
    }
    if source.trace is not None:
        report = service.explain_trace(
            source.trace,
            seed=seed if seed is not None else values.seed,
            **common,
        )
    else:
        report = service.explain_scenario(source.scenario, **common)
    if report.output_dir is None:
        raise ValueError("explanation report is missing its output directory")
    destination = report.output_dir
    typer.echo(f"Explained {len(report.steps)} step(s): {destination}")
    return destination
