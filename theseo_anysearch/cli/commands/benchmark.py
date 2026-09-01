"""Adaptive resource benchmark CLI."""

from __future__ import annotations

import sys
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
    debug: bool = typer.Option(
        False,
        "--debug",
        help=
        "Show PPO and Ray startup diagnostics instead of quiet progress bars.",
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
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    console = Console(file=sys.stdout)
    starting = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    starting.add_column(style="dim", min_width=20)
    starting.add_column()
    starting.add_row("experiment", experiment.experiment.name)
    starting.add_row("algorithm", experiment.training.algorithm)
    starting.add_row("environment limit",
                     f"{max_envs_per_worker} per rollout worker")
    starting.add_row("worker limit", f"{max_workers} rollout workers")
    starting.add_row(
        "stopping rules",
        f"{decline_patience} declines beyond {decline_tolerance:.0%} · "
        f"GPU target {max_gpu_utilization:g}% · "
        f"{max_duration_minutes:g} minute budget",
    )
    starting.add_row(
        "measurement",
        f"{warmup_iterations} warmup · {measure_iterations} measured iterations · "
        f"{repeats} repeats",
    )
    starting.add_row("output", str(artifact_dir))
    console.print(
        Panel(
            starting,
            title="[bold]Starting resource benchmark[/bold]",
            title_align="left",
        ))

    progress = Progress(
        SpinnerColumn("line"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    )
    environment_task = progress.add_task(
        "Environments per worker",
        total=max_envs_per_worker,
        status="initializing Ray",
    )
    worker_task = progress.add_task(
        "Rollout workers",
        total=max_workers,
        status="waiting",
        visible=False,
    )
    report_opened = False

    def update_progress(event, phase, candidate, summary) -> None:
        nonlocal report_opened
        if debug:
            if event == "completed" and summary is not None:
                typer.echo(
                    f"{phase} {candidate}: {summary.steps_per_second:.1f} steps/s"
                )
            return
        task = environment_task if phase == "environments" else worker_task
        if event == "started":
            progress.update(
                task,
                visible=True,
                status=f"measuring tick {candidate}",
            )
            return
        if summary is not None:
            gpu = summary.gpu_utilization_percent
            gpu_text = f", GPU {gpu:.1f}%" if gpu is not None else ""
            progress.update(
                task,
                completed=candidate,
                visible=True,
                status=f"{summary.steps_per_second:.1f} steps/s{gpu_text}",
            )
        if open_report and not report_opened:
            webbrowser.open((artifact_dir / "report.html").resolve().as_uri())
            report_opened = True

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
        debug=debug,
        progress_callback=update_progress,
    )
    try:
        if debug:
            result, artifacts = runner.run()
        else:
            with progress:
                result, artifacts = runner.run()
    except KeyboardInterrupt:
        typer.echo("Benchmark interrupted; active Ray resources were stopped.",
                   err=True)
        raise typer.Exit(130)

    recommendation = result.recommendation
    complete = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    complete.add_column(style="dim", min_width=20)
    complete.add_column()
    complete.add_row(
        "recommended",
        f"[bold]{recommendation.num_env_runners} workers × "
        f"{recommendation.num_envs_per_env_runner} environments[/bold]",
    )
    complete.add_row(
        "throughput",
        f"{recommendation.steps_per_second:.1f} steps/s · "
        f"[green]{recommendation.speedup:.2f}× baseline[/green]",
    )
    complete.add_row("environment sweep", result.environment_sweep.stop_reason)
    complete.add_row("worker sweep", result.worker_sweep.stop_reason)
    complete.add_row("elapsed", f"{result.elapsed_seconds / 60.0:.1f} minutes")
    complete.add_row("report", str(artifacts["html"]))
    complete.add_row(
        "diagnostics",
        f"{artifacts['stdout_log'].name} · {artifacts['stderr_log'].name}",
    )
    console.print(
        Panel(
            complete,
            title="[bold]Resource benchmark complete[/bold]",
            title_align="left",
        ))
    if not debug:
        console.print(
            "[dim]Use --debug to stream PPO and Ray diagnostics while retaining "
            "the same log files.[/dim]")
    if open_report and not report_opened:
        webbrowser.open(artifacts["html"].resolve().as_uri())
