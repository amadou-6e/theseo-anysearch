from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from theseo_anysearch.cli.commands import experiment as experiment_cmd
from theseo_anysearch.cli.commands import garden as garden_cmd
from theseo_anysearch.cli.commands import mlflow_ui as mlflow_cmd
from theseo_anysearch.cli.commands import ray_cmd
from theseo_anysearch.cli.commands import replay as replay_cmd
from theseo_anysearch.cli.commands import train as train_cmd
from theseo_anysearch.cli.commands import tune as tune_cmd

app = typer.Typer(
    name="anysearch",
    help="Theseo AnySearch — train and tune Rust-backed RL environments.",
    no_args_is_help=True,
)


def _print_run_summary(name: str, experiment, config_path: Path | None) -> None:
    """Print a Rich panel summarising the run before training starts."""
    import sys
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console(file=sys.stdout, highlight=False)
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style="dim", min_width=18, no_wrap=True)
    table.add_column(overflow="fold")

    # Core
    table.add_row("name", name)
    table.add_row("algorithm", experiment.training.algorithm)
    table.add_row("iterations", str(experiment.training.iterations))
    chk = experiment.training.checkpoint_interval
    if chk:
        table.add_row("checkpoint every", str(chk))
    gpu = experiment.training.require_gpu
    table.add_row("gpu", "required" if gpu else "not required")

    # Env
    env = experiment.env
    table.add_row("", "")
    if getattr(env, "stl_path", None):
        table.add_row("stl", str(env.stl_path))
        scale = getattr(env, "scale", None)
        scale_range = getattr(env, "scale_range", None)
        if scale_range:
            table.add_row("scale range", f"{scale_range[0]} – {scale_range[1]}")
        elif scale is not None:
            table.add_row("scale", str(scale))
    table.add_row("grid size", str(getattr(env, "grid_size", 32)))
    table.add_row("obs mode", str(getattr(env, "obs_mode", "scalar")))
    box_radius = getattr(env, "box_radius", None)
    if box_radius is not None:
        table.add_row("box radius", str(box_radius))
    agents = getattr(env, "agent_count", None)
    if agents:
        table.add_row("agents", str(agents))
    table.add_row("max steps", str(getattr(env, "max_steps", "?")))

    # Model
    mc = experiment.model_cfg
    custom_model = (mc.get("custom_model") if isinstance(mc, dict) else getattr(mc, "custom_model", None)) if mc else None
    if custom_model:
        table.add_row("", "")
        table.add_row("model", str(custom_model))
        cmc = (mc.get("custom_model_config") if isinstance(mc, dict) else getattr(mc, "custom_model_config", None)) or {}
        if cmc.get("pretrained_encoder"):
            table.add_row("encoder", str(cmc["pretrained_encoder"]))
            table.add_row("freeze encoder", str(cmc.get("freeze_encoder", False)))

    # Algo highlights
    ac = experiment.algorithm_config or {}
    lr = ac.get("lr") if isinstance(ac, dict) else getattr(ac, "lr", None)
    bs = ac.get("train_batch_size") if isinstance(ac, dict) else getattr(ac, "train_batch_size", None)
    if lr is not None or bs is not None:
        table.add_row("", "")
    if lr is not None:
        table.add_row("lr", str(lr))
    if bs is not None:
        table.add_row("batch size", str(bs))

    # Paths
    table.add_row("", "")
    table.add_row("output", str(experiment.run_output_dir))
    if config_path:
        table.add_row("config", str(config_path))

    try:
        console.print(Panel(table, title="[bold]Starting run[/bold]", title_align="left"), new_line_start=True)
    except UnicodeEncodeError:
        typer.echo(f"\nRunning '{name}' ({experiment.training.algorithm}, {experiment.training.iterations} iters) ...")


def _print_tune_summary(name: str, experiment, config_path: Path | None, tag: str | None) -> None:
    """Print a Rich panel summarising a tune sweep before it starts."""
    import sys
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console(file=sys.stdout, highlight=False)
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style="dim", min_width=18)
    table.add_column()

    tc = experiment.tune_config
    table.add_row("name", name)
    table.add_row("algorithm", experiment.training.algorithm)
    table.add_row("scheduler", str(getattr(tc, "scheduler", "?")))
    table.add_row("trials", str(getattr(tc, "num_samples", "?")))
    table.add_row("concurrency", str(getattr(tc, "max_concurrent", "?")))
    table.add_row("iterations", str(experiment.training.iterations))
    table.add_row("tag", tag or "latest")
    gpu = experiment.training.require_gpu
    table.add_row("gpu", "required" if gpu else "not required")

    # Search space keys
    ss = getattr(tc, "search_space", None) or {}
    if ss:
        table.add_row("", "")
        for section, params in ss.items():
            if isinstance(params, dict):
                for k in params:
                    table.add_row(f"search  {section}.{k}", "")

    # Paths
    table.add_row("", "")
    table.add_row("output", str(experiment.run_output_dir))
    if config_path:
        table.add_row("config", str(config_path))
    table.add_row("tensorboard", f"anysearch tensorboard {name}")

    try:
        console.print(Panel(table, title="[bold]Starting tune sweep[/bold]", title_align="left"), new_line_start=True)
    except UnicodeEncodeError:
        typer.echo(f"\nTune sweep '{name}' ({experiment.training.algorithm}, {getattr(tc, 'num_samples', '?')} trials) ...")


# ---------------------------------------------------------------------------
# Hot path: run
# ---------------------------------------------------------------------------

@app.command()
def run(
    ref: str = typer.Argument(
        ...,
        help="Experiment directory or registered name (e.g. ppo-baseline or usage/experiments/ppo).",
    ),
    tag: Optional[str] = typer.Option(
        None, "--tag", "-t", help="Sweep tag / output subdirectory name (tune sweeps only)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override output location (default: the experiment directory)."
    ),
) -> None:
    """Start a training run or sweep from an experiment directory or YAML file."""
    from theseo_anysearch.cli.registry import (
        add_experiment, resolve_config_and_dir,
    )
    from theseo_anysearch.experiments.loader import expand_sweep, load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig, SweepConfig
    from theseo_anysearch.experiments.runner import ExperimentRunner

    try:
        config_path, experiment_dir = resolve_config_and_dir(ref)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    # Auto-register under the config yaml path (name = yaml stem).
    # For canonical names like config.yaml inside a dedicated directory, register
    # the directory instead so the name comes from the directory basename.
    _reg_path = experiment_dir if config_path.name in ("config.yaml", "experiment.yaml") else config_path
    name = add_experiment(_reg_path)

    experiment = load_experiment(config_path)

    # --output-dir overrides the output_dir already resolved by load_experiment
    def _apply_output(exp: ExperimentConfig, out: Path) -> ExperimentConfig:
        return exp.model_copy(
            update={"experiment": exp.experiment.model_copy(update={"output_dir": out})}
        )

    effective_output = output_dir if output_dir is not None else experiment.experiment.output_dir

    if isinstance(experiment, SweepConfig):
        entries = expand_sweep(experiment)
        typer.echo(f"Sweep: {len(entries)} experiments")
        for exp in entries:
            exp = _apply_output(exp, effective_output)
            runner = ExperimentRunner(exp, config_path)
            typer.echo(f"  Running '{exp.experiment.name}' ...")
            info = runner.run()
            typer.echo(f"  Done: run_id={info.run_id} status={info.status}")
    elif experiment.tune_config is not None:
        from theseo_anysearch.experiments.tune_runner import TuneRunner
        experiment = _apply_output(experiment, effective_output)
        _print_tune_summary(name, experiment, config_path, tag)
        runner_tune = TuneRunner(experiment, config_path, tag=tag or None)
        result = runner_tune.run()
        print(json.dumps(result, indent=2, default=str))
    else:
        experiment = _apply_output(experiment, effective_output)
        runner = ExperimentRunner(experiment, config_path)
        _print_run_summary(name, experiment, config_path)
        info = runner.run()
        print(json.dumps(info.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# Hot path: list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_experiments(
    ref: Optional[str] = typer.Argument(
        None,
        help="Registered name or directory to scope to (default: all registered experiments).",
    ),
    short: bool = typer.Option(
        False, "--short", "-s", help="Omit the experiment YAML contents."
    ),
) -> None:
    """List runs and sweeps. No argument shows all registered experiments."""
    from theseo_anysearch.cli.registry import load_registry, _resolve_dir, resolve_config_and_dir
    from theseo_anysearch.experiments.runner import ExperimentRunner

    if ref is not None:
        reg = load_registry()
        if ref in reg:
            stored = Path(reg[ref])
            display_name = ref
        else:
            stored = _resolve_dir(ref)
            display_name = ref
        dirs = {display_name: stored}
    else:
        reg = load_registry()
        if not reg:
            typer.echo("No experiments registered. Run: anysearch add <dir>")
            raise typer.Exit()
        dirs = {name: Path(path) for name, path in reg.items()}

    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    console = Console(file=sys.stdout, highlight=False)
    scoped = ref is not None

    for display_name, stored_path in dirs.items():
        # Resolve config yaml and experiment directory from stored path
        try:
            config_path, experiment_dir = resolve_config_and_dir(str(stored_path))
        except Exception:
            config_path = None
            experiment_dir = stored_path if stored_path.is_dir() else stored_path.parent

        # Resolve output_dir and experiment name from raw YAML (avoids importing torch/ray)
        resolved_out: Path | None = None
        experiment_name: str | None = None
        if config_path and config_path.exists():
            try:
                import yaml as _yaml
                raw = _yaml.safe_load(config_path.read_text()) or {}
                exp_section = raw.get("experiment", {})
                experiment_name = exp_section.get("name")
                out_raw = exp_section.get("output_dir")
                if out_raw is not None:
                    yaml_dir = config_path.resolve().parent
                    out_path = Path(out_raw)
                    resolved_out = (yaml_dir / out_path).resolve() if not out_path.is_absolute() else out_path
                else:
                    resolved_out = config_path.resolve().parent
            except Exception:
                pass

        # Runs live under output_dir/experiment_name (mirrors run_output_dir property)
        if resolved_out is not None and experiment_name:
            search_dir = resolved_out / experiment_name
        elif resolved_out is not None:
            search_dir = resolved_out
        else:
            search_dir = experiment_dir
        runs = ExperimentRunner.list_runs(search_dir)

        # Build panel content
        lines: list = []

        if scoped:
            # Show yaml and output paths
            meta = Table(box=None, show_header=False, padding=(0, 1, 0, 0))
            meta.add_column(style="dim", min_width=12, no_wrap=True)
            meta.add_column(overflow="fold")
            if config_path:
                meta.add_row("config", str(config_path))
            if resolved_out:
                meta.add_row("output_dir", str(resolved_out))
            lines.append(meta)

        # Runs table
        run_table = Table(box=None, show_header=bool(runs), padding=(0, 1, 0, 0))
        run_table.add_column("TAG / RUN ID", style="cyan", min_width=18)
        run_table.add_column("STATUS", min_width=14)
        run_table.add_column("STARTED")

        if not runs:
            lines.append(Text("(no runs found)", style="dim"))
        else:
            for r in reversed(runs):
                started = r["start_time"][:10] if r.get("start_time") else "?"
                status = r["status"]
                status_style = "green" if status == "COMPLETED" else "yellow" if status == "RUNNING" else "red"
                label = r["run_id"]
                if "sweep_trials" in r:
                    label += f"  [dim]({r['sweep_trials']} trials)[/dim]"
                run_table.add_row(label, Text(status, style=status_style), started)
            lines.append(run_table)

        # Show yaml content unless --short
        if scoped and not short and config_path and config_path.exists():
            lines.append(Text(""))
            lines.append(Syntax(config_path.read_text(), "yaml", theme="ansi_dark", line_numbers=False))

        from rich.console import Group
        content = Group(*lines)
        title = f"[bold]{display_name}[/bold]"

        try:
            console.print(Panel(content, title=title, title_align="left"))
        except UnicodeEncodeError:
            typer.echo(f"\n{display_name}")
            if scoped and config_path:
                typer.echo(f"  config     {config_path}")
            if not runs:
                typer.echo("  (no runs found)")
            else:
                typer.echo(f"  {'TAG / RUN ID':<18} {'STATUS':<14} {'STARTED'}")
                for r in reversed(runs):
                    started = r["start_time"][:10] if r.get("start_time") else "?"
                    typer.echo(f"  {r['run_id']:<18} {r['status']:<14} {started}")
            if scoped and not short and config_path and config_path.exists():
                typer.echo(config_path.read_text())

    typer.echo(
        "\n  To replay:   anysearch replay <name:tag>\n"
        "  To inspect:  anysearch inspect <name:run_id>"
    )


# ---------------------------------------------------------------------------
# add (register an experiment directory)
# ---------------------------------------------------------------------------

@app.command()
def add(
    directory: Path = typer.Argument(..., help="Experiment directory to register."),
    name: Optional[str] = typer.Argument(
        None, help="Short name (default: experiment.name from config, then dir basename)."
    ),
) -> None:
    """Register an experiment YAML or directory under a short name."""
    from theseo_anysearch.cli.registry import add_experiment

    if not directory.exists():
        typer.echo(f"Error: not found: {directory}", err=True)
        raise typer.Exit(1)

    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    # Pass the path as-is; add_experiment handles YAML vs directory naming
    registered_name = add_experiment(directory, name)

    console = Console(file=sys.stdout, highlight=False)
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style="dim", min_width=10, no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("name", registered_name)
    table.add_row("path", str(directory.resolve()))
    try:
        console.print(Panel(table, title="[bold]Registered[/bold]", title_align="left"), new_line_start=True)
    except UnicodeEncodeError:
        typer.echo(f"Registered: {registered_name} -> {directory.resolve()}")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def _summarise_run_dir(path: Path) -> str:
    parts = []
    if (path / "checkpoints").exists():
        n = sum(1 for _ in (path / "checkpoints").iterdir())
        parts.append(f"{n} checkpoint(s)")
    if (path / "trajectories").exists():
        n = sum(1 for _ in (path / "trajectories").glob("*.json"))
        parts.append(f"{n} trajectory file(s)")
    if (path / "renders").exists():
        n = sum(1 for _ in (path / "renders").iterdir())
        parts.append(f"{n} render(s)")
    mb = _dir_size_mb(path)
    parts.append(f"{mb:.1f} MB")
    return ", ".join(parts) if parts else "(empty)"


@app.command()
def delete(
    ref: str = typer.Argument(
        ...,
        help="Experiment name (delete all runs + deregister) or name:run_id (delete one run).",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
    cached: bool = typer.Option(
        False, "--cached", help="Only remove from registry — do not delete any files."
    ),
) -> None:
    """Delete run output directories and/or deregister an experiment.

    \b
    anysearch delete ppo_baseline            — delete all runs + deregister
    anysearch delete ppo_baseline:a1b2c3d4   — delete one run directory
    anysearch delete ppo_baseline --cached   — deregister only, keep files
    """
    import shutil as _shutil
    import sys
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from theseo_anysearch.cli.registry import resolve_ref, resolve_config_and_dir, load_registry

    console = Console(file=sys.stdout, highlight=False)
    err = Console(file=sys.stderr, highlight=False)

    experiment_dir, run_id = resolve_ref(ref)

    # Resolve output_dir and experiment.name from raw YAML
    run_output_dir: Path | None = None
    experiment_name: str | None = None
    try:
        config_path, _ = resolve_config_and_dir(ref)
        if config_path and config_path.exists():
            import yaml as _yaml
            raw = _yaml.safe_load(config_path.read_text()) or {}
            exp = raw.get("experiment", {})
            experiment_name = exp.get("name")
            out_raw = exp.get("output_dir")
            if out_raw is not None:
                p = Path(out_raw)
                base = (config_path.parent / p).resolve() if not p.is_absolute() else p
            else:
                base = config_path.parent.resolve()
            if experiment_name:
                run_output_dir = base / experiment_name
    except Exception:
        pass

    # Determine what to delete
    dirs_to_delete: list[Path] = []

    if run_id is not None:
        # Single run
        from theseo_anysearch.experiments.runner import _find_run_dir
        search = run_output_dir or experiment_dir
        try:
            run_dir = _find_run_dir(search, run_id)
            dirs_to_delete = [run_dir]
        except FileNotFoundError as exc:
            err.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
    else:
        # All runs for this experiment
        search = run_output_dir or experiment_dir
        if search.exists():
            dirs_to_delete = [
                d for d in search.iterdir()
                if d.is_dir() and (d / "run.json").exists()
            ]

    # Determine registry entry to remove
    reg = load_registry()
    name_in_registry: str | None = None
    for n in reg:
        if ref == n or ref.startswith(f"{n}:"):
            name_in_registry = n
            break

    if not dirs_to_delete and name_in_registry is None:
        err.print(f"[red]Error:[/red] nothing found for '{ref}'.")
        raise typer.Exit(1)

    # Show what will happen
    if cached:
        console.print(f"\n[bold]Registry entry to remove:[/bold] {name_in_registry or '(not registered)'}")
        console.print("[dim]Files will not be touched (--cached).[/dim]\n")
    else:
        table = Table(box=None, show_header=True, padding=(0, 2, 0, 0))
        table.add_column("Directory", style="cyan")
        table.add_column("Contents", style="dim")
        for d in dirs_to_delete:
            table.add_row(str(d), _summarise_run_dir(d))
        if run_id is None and name_in_registry:
            console.print(f"\n[bold]Will deregister:[/bold] {name_in_registry}")
        console.print(table)

    if not force:
        answer = typer.prompt("\nProceed? [y/N]", default="N")
        if answer.strip().lower() != "y":
            typer.echo("Aborted.")
            raise typer.Exit(0)

    # Execute
    if not cached:
        for d in dirs_to_delete:
            _shutil.rmtree(d)
            console.print(f"  [red]Deleted[/red]  {d}")

    if run_id is None and name_in_registry:
        from theseo_anysearch.cli.registry import save_registry
        save_registry({k: v for k, v in reg.items() if k != name_in_registry})
        console.print(f"  [dim]Removed[/dim]  {name_in_registry} from registry")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@app.command()
def inspect(
    ref: str = typer.Argument(
        ..., help="<name:run_id> or <dir:run_id> — e.g. ppo-baseline:a1b2c3d4"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override experiment directory search root."
    ),
) -> None:
    """Print resolved config, metrics, and artifact paths for a run."""
    from theseo_anysearch.cli.registry import resolve_ref
    from theseo_anysearch.experiments.runner import ExperimentRunner, _find_run_dir

    experiment_dir, run_id = resolve_ref(ref)
    if run_id is None:
        typer.echo("Error: ref must include a run_id, e.g. ppo-baseline:a1b2c3d4", err=True)
        raise typer.Exit(1)

    search_root = output_dir if output_dir is not None else experiment_dir
    run_dir = _find_run_dir(search_root, run_id)
    result = ExperimentRunner.inspect(run_id, run_dir.parent)
    print(json.dumps(result.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

@app.command()
def resume(
    ref: str = typer.Argument(
        ..., help="<name:run_id> or <dir:run_id> — e.g. ppo-baseline:a1b2c3d4"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override experiment directory search root."
    ),
) -> None:
    """Continue training from the latest checkpoint."""
    from theseo_anysearch.cli.registry import resolve_ref
    from theseo_anysearch.experiments.runner import ExperimentRunner, _find_run_dir
    from theseo_anysearch.experiments.loader import load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig

    experiment_dir, run_id = resolve_ref(ref)
    if run_id is None:
        typer.echo("Error: ref must include a run_id, e.g. ppo-baseline:a1b2c3d4", err=True)
        raise typer.Exit(1)

    search_root = output_dir if output_dir is not None else experiment_dir
    run_dir = _find_run_dir(search_root, run_id)
    src_yaml = run_dir / "experiment.yaml"
    if not src_yaml.exists():
        typer.echo(f"experiment.yaml not found in {run_dir}.", err=True)
        raise typer.Exit(1)

    experiment = load_experiment(src_yaml)
    if not isinstance(experiment, ExperimentConfig):
        typer.echo("Sweep resume is not yet supported.", err=True)
        raise typer.Exit(1)

    runner = ExperimentRunner(experiment)
    typer.echo(f"Resuming {ref} ...")
    info = runner.resume(run_id)
    print(json.dumps(info.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

@app.command()
def repeat(
    ref: str = typer.Argument(
        ..., help="<name:run_id> or <dir:run_id> — e.g. ppo-baseline:a1b2c3d4"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override experiment directory search root."
    ),
) -> None:
    """Re-run from scratch with the same config (new run_id)."""
    from theseo_anysearch.cli.registry import resolve_ref
    from theseo_anysearch.experiments.runner import ExperimentRunner, _find_run_dir
    from theseo_anysearch.experiments.loader import load_experiment
    from theseo_anysearch.experiments.models import ExperimentConfig

    experiment_dir, run_id = resolve_ref(ref)
    if run_id is None:
        typer.echo("Error: ref must include a run_id, e.g. ppo-baseline:a1b2c3d4", err=True)
        raise typer.Exit(1)

    search_root = output_dir if output_dir is not None else experiment_dir
    run_dir = _find_run_dir(search_root, run_id)
    src_yaml = run_dir / "experiment.yaml"
    if not src_yaml.exists():
        typer.echo(f"experiment.yaml not found in {run_dir}.", err=True)
        raise typer.Exit(1)

    experiment = load_experiment(src_yaml)
    if not isinstance(experiment, ExperimentConfig):
        typer.echo("Sweep repeats are not yet supported.", err=True)
        raise typer.Exit(1)

    runner = ExperimentRunner(experiment, src_yaml)
    typer.echo(f"Repeating {ref} as a new run ...")
    info = runner.run()
    print(json.dumps(info.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# tensorboard
# ---------------------------------------------------------------------------

@app.command()
def tensorboard(
    ref: Optional[str] = typer.Argument(
        None,
        help="Experiment reference: registered name, name:run_id, name:tag, or directory. "
             "Omit to use the current directory.",
    ),
    port: int = typer.Option(6006, "--port", help="Port to serve TensorBoard on."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind TensorBoard to."),
) -> None:
    """Launch TensorBoard for an experiment, run, or sweep."""
    import shutil
    import subprocess
    from theseo_anysearch.cli.registry import resolve_ref

    if ref is None:
        logdir = (Path(".") / "runtime" / "experiments").resolve()
        if not logdir.exists():
            logdir = Path(".").resolve()
    else:
        config_path, experiment_dir = None, None
        try:
            from theseo_anysearch.cli.registry import resolve_config_and_dir
            config_path, experiment_dir = resolve_config_and_dir(ref)
        except Exception:
            pass

        identifier: str | None = None
        try:
            _, identifier = resolve_ref(ref)
        except Exception:
            pass

        # Resolve output_dir from the YAML (avoids importing torch/ray)
        output_dir: Path | None = None
        if config_path and config_path.exists():
            try:
                import yaml as _yaml
                raw = _yaml.safe_load(config_path.read_text()) or {}
                out_raw = raw.get("experiment", {}).get("output_dir")
                if out_raw is not None:
                    p = Path(out_raw)
                    output_dir = (config_path.parent / p).resolve() if not p.is_absolute() else p
                else:
                    output_dir = config_path.parent.resolve()
            except Exception:
                pass

        base = output_dir or (experiment_dir.resolve() if experiment_dir else Path(".").resolve())
        if identifier is not None:
            candidate = base / identifier
            logdir = candidate if candidate.exists() else base
        else:
            logdir = base

    typer.echo(f"TensorBoard logdir: {logdir}")
    typer.echo(f"Open: http://{host}:{port}")

    tb_exe = shutil.which("tensorboard")
    if tb_exe is None:
        typer.echo("Error: tensorboard not found on PATH. Install it with: pip install tensorboard", err=True)
        raise typer.Exit(1)

    try:
        subprocess.run(
            [tb_exe, "--logdir", str(logdir), "--host", host, "--port", str(port)],
            check=True,
        )
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError:
        typer.echo(
            f"Error: TensorBoard exited with an error. "
            f"If port {port} is already in use, try: anysearch tensorboard {ref or ''} --port {port + 1}".strip(),
            err=True,
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Subcommand groups
# ---------------------------------------------------------------------------

app.add_typer(replay_cmd.app, name="replay")
app.add_typer(mlflow_cmd.app, name="mlflow")
app.add_typer(ray_cmd.app, name="ray")
app.add_typer(garden_cmd.app, name="garden")

# Deprecated groups — kept for backward compatibility
app.add_typer(
    experiment_cmd.app,
    name="experiment",
    deprecated=True,
    help="[deprecated] Use top-level anysearch run / inspect / resume / repeat / list.",
)
app.add_typer(
    train_cmd.app,
    name="train",
    deprecated=True,
    help="[deprecated] Use: anysearch run <dir>",
)
app.add_typer(
    tune_cmd.app,
    name="tune",
    deprecated=True,
    help="[deprecated] Use: anysearch run <dir> --tag <tag>",
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
