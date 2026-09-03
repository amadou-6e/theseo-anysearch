"""CLI helpers for listing, inspecting, repeating, and resuming experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="[deprecated] Use top-level anysearch run / inspect / resume / repeat / list.")

_DEPRECATION_PREFIX = (
    "[deprecated] anysearch experiment {sub} is deprecated. "
    "Use: anysearch {modern}\n"
)

_MODERN = {
    "run":     "run <dir>",
    "resume":  "resume <name:run_id>",
    "repeat":  "repeat <name:run_id>",
    "inspect": "inspect <name:run_id>",
    "list":    "list [<name>]",
}


def _load(config_path: Path) -> "ExperimentConfig | SweepConfig":  # noqa: F821
    from theseo_anysearch.experiments.loader import load_experiment
    return load_experiment(config_path)


def _output_base(config: "ExperimentConfig", override: Optional[Path]) -> Path:  # noqa: F821
    if override is not None:
        return override / config.experiment.name
    return config.run_output_dir


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to the experiment YAML."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override the output_dir in the YAML."
    ),
    tag: Optional[str] = typer.Option(
        None, "--tag", help="Override the run tag (tune sweep subdirectory name)."
    ),
) -> None:
    """Run a training experiment or sweep from a YAML config file."""
    typer.echo(_DEPRECATION_PREFIX.format(sub="run", modern=_MODERN["run"]), err=True)
    from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig, SweepConfig
    from theseo_anysearch.experiments.runner import ExperimentRunner

    experiment = load_experiment(config)

    if isinstance(experiment, SweepConfig):
        entries = expand_sweep(experiment)
        typer.echo(f"Sweep: {len(entries)} experiments")
        for exp in entries:
            if output_dir is not None:
                from pydantic import BaseModel
                exp = exp.model_copy(update={
                    "experiment": exp.experiment.model_copy(update={"output_dir": output_dir})
                })
            runner = ExperimentRunner(exp, config)
            typer.echo(f"  Running '{exp.experiment.name}' ...")
            info = runner.run()
            typer.echo(f"  Done: run_id={info.run_id} status={info.status}")
    else:
        if output_dir is not None:
            experiment = experiment.model_copy(update={
                "experiment": experiment.experiment.model_copy(
                    update={"output_dir": output_dir}
                )
            })
        if experiment.tune_config is not None:
            from theseo_anysearch.experiments.tune_runner import TuneRunner
            run_tag = tag or experiment.experiment.name
            typer.echo(
                f"Running tune sweep '{run_tag}' "
                f"({experiment.tune_config.num_samples} trials, "
                f"scheduler={experiment.tune_config.scheduler}) ..."
            )
            runner_tune = TuneRunner(experiment, config, tag=run_tag)
            result = runner_tune.run()
            print(json.dumps(result, indent=2, default=str))
        else:
            runner = ExperimentRunner(experiment, config)
            typer.echo(f"Running experiment '{experiment.experiment.name}' ...")
            info = runner.run()
            print(json.dumps(info.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

@app.command()
def resume(
    run_id: str = typer.Option(..., "--run-id", help="8-char hex run ID to resume."),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Experiment YAML (must match the original). Required if --output-dir is set.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Root output directory to search for the run.",
    ),
) -> None:
    """Continue a training run from its latest checkpoint."""
    typer.echo(_DEPRECATION_PREFIX.format(sub="resume", modern=_MODERN["resume"]), err=True)
    from theseo_anysearch.experiments.loader import load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig
    from theseo_anysearch.experiments.runner import ExperimentRunner, _find_run_dir

    if config is None and output_dir is None:
        typer.echo(
            "Error: provide at least --config or --output-dir to locate the run.",
            err=True,
        )
        raise typer.Exit(code=1)

    if config is not None:
        experiment = load_experiment(config)
        if not isinstance(experiment, ExperimentConfig):
            typer.echo("Error: --config must point to a single experiment, not a sweep.", err=True)
            raise typer.Exit(code=1)
        if output_dir is not None:
            experiment = experiment.model_copy(update={
                "experiment": experiment.experiment.model_copy(update={"output_dir": output_dir})
            })
    else:
        # Try to read experiment.yaml from the run itself
        run_dir = _find_run_dir(output_dir, run_id)
        src_yaml = run_dir / "experiment.yaml"
        if not src_yaml.exists():
            typer.echo(
                f"experiment.yaml not found in {run_dir}. Provide --config explicitly.",
                err=True,
            )
            raise typer.Exit(code=1)
        experiment = load_experiment(src_yaml)

    runner = ExperimentRunner(experiment)
    typer.echo(f"Resuming run '{run_id}' ...")
    info = runner.resume(run_id)
    print(json.dumps(info.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

@app.command()
def repeat(
    run_id: str = typer.Option(..., "--run-id", help="8-char hex run ID to repeat."),
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        "--output-dir",
        help="Root output directory to search for the original run.",
    ),
) -> None:
    """Re-run a completed experiment from scratch with the same config."""
    typer.echo(_DEPRECATION_PREFIX.format(sub="repeat", modern=_MODERN["repeat"]), err=True)
    from theseo_anysearch.experiments.runner import ExperimentRunner, _find_run_dir
    from theseo_anysearch.experiments.loader import load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig

    run_dir = _find_run_dir(output_dir, run_id)
    src_yaml = run_dir / "experiment.yaml"
    if not src_yaml.exists():
        typer.echo(f"experiment.yaml not found in {run_dir}.", err=True)
        raise typer.Exit(code=1)

    experiment = load_experiment(src_yaml)
    if not isinstance(experiment, ExperimentConfig):
        typer.echo("Sweep repeats are not yet supported.", err=True)
        raise typer.Exit(code=1)

    runner = ExperimentRunner(experiment, src_yaml)
    typer.echo(f"Repeating run '{run_id}' as a new run ...")
    info = runner.run()
    print(json.dumps(info.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@app.command()
def inspect(
    run_id: str = typer.Option(..., "--run-id", help="8-char hex run ID to inspect."),
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        "--output-dir",
        help="Root output directory to search for the run.",
    ),
) -> None:
    """Print resolved config, metrics summary, and artifact paths for a run."""
    typer.echo(_DEPRECATION_PREFIX.format(sub="inspect", modern=_MODERN["inspect"]), err=True)
    from theseo_anysearch.experiments.runner import ExperimentRunner
    result = ExperimentRunner.inspect(run_id, output_dir)
    print(json.dumps(result.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_runs(
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        "--output-dir",
        help="Root directory to search for runs.",
    ),
) -> None:
    """List all runs found under the output directory."""
    typer.echo(_DEPRECATION_PREFIX.format(sub="list", modern=_MODERN["list"]), err=True)
    from theseo_anysearch.experiments.runner import ExperimentRunner
    runs = ExperimentRunner.list_runs(output_dir)
    print(json.dumps(runs, indent=2, default=str))
