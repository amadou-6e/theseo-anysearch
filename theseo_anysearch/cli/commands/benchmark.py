"""Adaptive resource benchmark CLI."""

from __future__ import annotations

import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    help="Measure rollout throughput and recommend resource settings.")


@app.command()
def resources(
    ref: str = typer.Argument(
        ...,
        help="Experiment YAML, directory, or registered experiment name.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help=
        "Artifact directory (default: <experiment output>/benchmarks/<timestamp>).",
    ),
    decline_patience: int = typer.Option(
        3,
        "--decline-patience",
        min=1,
        help="Stop after this many consecutive throughput declines.",
    ),
    decline_tolerance: float = typer.Option(
        0.02,
        "--decline-tolerance",
        min=0.0,
        max=0.99,
        help="Relative regression treated as measurement noise.",
    ),
    warmup_iterations: int = typer.Option(1, "--warmup-iterations", min=1),
    measure_iterations: int = typer.Option(3, "--measure-iterations", min=1),
    repeats: int = typer.Option(3, "--repeats", min=1),
    max_envs_per_worker: int = typer.Option(
        16,
        "--max-envs-per-worker",
        min=1,
    ),
    max_workers: int = typer.Option(20, "--max-workers", min=1),
    max_gpu_utilization: float = typer.Option(
        95.0,
        "--max-gpu-utilization",
        min=0.01,
        max=100.0,
        help=
        "Stop worker scaling when median repeat-average GPU utilization reaches this percentage.",
    ),
    max_duration_minutes: float = typer.Option(
        30.0,
        "--max-duration-minutes",
        min=0.01,
        help="Soft wall-clock budget; active candidates finish cleanly.",
    ),
    open_report: bool = typer.Option(
        False,
        "--open",
        help="Open the standalone HTML report when complete.",
    ),
) -> None:
    """Find efficient vector-environment and rollout-worker counts."""
    from theseo_anysearch.benchmarking.runner import ResourceBenchmarkRunner
    from theseo_anysearch.cli.registry import resolve_config_and_dir
    from theseo_anysearch.experiments.loader import load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig

    try:
        config_path, _ = resolve_config_and_dir(ref)
        experiment = load_experiment(config_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    if not isinstance(experiment, ExperimentConfig):
        typer.echo("Error: resource benchmarks require one experiment config.",
                   err=True)
        raise typer.Exit(1)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir = output_dir or (experiment.run_output_dir / "benchmarks" /
                                  timestamp)
    typer.echo(f"Benchmark artifacts: {artifact_dir}")
    typer.echo("Phase 1: environments per rollout worker")

    runner = ResourceBenchmarkRunner(
        experiment,
        config_path,
        output_dir=artifact_dir,
        decline_patience=decline_patience,
        decline_tolerance=decline_tolerance,
        warmup_iterations=warmup_iterations,
        measure_iterations=measure_iterations,
        repeats=repeats,
        max_envs_per_worker=max_envs_per_worker,
        max_workers=max_workers,
        max_gpu_utilization=max_gpu_utilization,
        max_duration_minutes=max_duration_minutes,
    )
    try:
        result, artifacts = runner.run()
    except KeyboardInterrupt:
        typer.echo("Benchmark interrupted; active Ray resources were stopped.",
                   err=True)
        raise typer.Exit(130)

    recommendation = result.recommendation
    typer.echo("Recommended: "
               f"{recommendation.num_env_runners} workers x "
               f"{recommendation.num_envs_per_env_runner} environments "
               f"({recommendation.steps_per_second:.1f} steps/s, "
               f"{recommendation.speedup:.2f}x baseline)")
    typer.echo(
        json.dumps({
            key: str(path)
            for key, path in artifacts.items()
        },
                   indent=2))
    if open_report:
        webbrowser.open(artifacts["html"].resolve().as_uri())
