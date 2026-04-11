"""Typer entrypoint for the AnySearch command-line interface.

Examples
--------
Run a training config::

    anysearch run usage/experiments/train/ppo_maps.yaml

Replay a saved run::

    anysearch replay ppo-maps:7367dc57
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

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
# Geometry pool validation (used by run before starting training)
# ---------------------------------------------------------------------------

def _resolve_geometry_pool_path(experiment: Any, config_path: Path) -> Any:
    """Return experiment with geometry_pool.pool_dir resolved to an absolute path.

    The YAML stores a path relative to the config file's directory.  Ray workers
    run with a different CWD, so we must resolve to absolute here in the main
    process before the config is serialised and shipped to workers.
    """
    env = getattr(experiment, "env", None)
    if env is None:
        return experiment
    pool_cfg = getattr(env, "geometry_pool", None)
    if not isinstance(pool_cfg, dict) or not pool_cfg.get("pool_dir"):
        return experiment
    pool_dir = Path(str(pool_cfg["pool_dir"]))
    if pool_dir.is_absolute():
        return experiment
    resolved = str((config_path.parent / pool_dir).resolve())
    new_pool_cfg = {**pool_cfg, "pool_dir": resolved}
    new_env = env.model_copy(update={"geometry_pool": new_pool_cfg})
    return experiment.model_copy(update={"env": new_env})


def _check_geometry_pool(experiment: Any, config_path: Path) -> None:
    """If the experiment uses a geometry pool, validate it and exit with a hint if not ready."""
    env = getattr(experiment, "env", None)
    pool_cfg = None
    if env is not None:
        pool_cfg = getattr(env, "geometry_pool", None)
        if pool_cfg is None and isinstance(env, dict):
            pool_cfg = env.get("geometry_pool")
    if not pool_cfg:
        return

    pool_dir_raw = (
        pool_cfg.get("pool_dir") if isinstance(pool_cfg, dict)
        else getattr(pool_cfg, "pool_dir", None)
    )
    if not pool_dir_raw:
        return

    pool_dir = Path(str(pool_dir_raw))
    if not pool_dir.is_absolute():
        pool_dir = config_path.parent / pool_dir

    try:
        from theseo_anysearch.environments.geometry_pool import GeometryPool
        gp = GeometryPool(pool_dir)
        ok, msg = gp.validate(min_per_source=10)
        if not ok:
            typer.echo(
                f"\nError: geometry pool '{pool_dir}' is not ready:\n  {msg}\n\n"
                f"Run first:\n\n  anysearch extract <config_or_stl> --pool-dir {pool_dir}\n",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Pool: {msg}")
    except FileNotFoundError as exc:
        typer.echo(
            f"\nError: {exc}\n\nRun first:\n\n  anysearch extract <config_or_stl> --pool-dir {pool_dir}\n",
            err=True,
        )
        raise typer.Exit(1)


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
    resume_tune: bool = typer.Option(
        False,
        "--resume-tune",
        help="Resume the most recent interrupted tune sweep segment for the selected tag.",
    ),
    extra_trials: int = typer.Option(
        0,
        "--extra-trials",
        help="Append more trials to an existing completed tune sweep tag.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override output location (default: the experiment directory)."
    ),
) -> None:
    """Start a training run or sweep from an experiment directory or YAML file."""
    from theseo_anysearch.cli.registry import (
        RegistryAccessError, add_experiment, resolve_config_and_dir,
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
    try:
        name = add_experiment(_reg_path)
    except RegistryAccessError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    experiment = load_experiment(config_path)
    experiment = _resolve_geometry_pool_path(experiment, config_path)

    # --- Pool validation ---
    _check_geometry_pool(experiment, config_path)

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
        runner_tune = TuneRunner(
            experiment,
            config_path,
            tag=tag or None,
            resume=resume_tune,
            extra_trials=extra_trials,
        )
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

    from theseo_anysearch.cli.registry import RegistryAccessError

    # Pass the path as-is; add_experiment handles YAML vs directory naming
    try:
        registered_name = add_experiment(directory, name)
    except RegistryAccessError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

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
        from theseo_anysearch.cli.registry import RegistryAccessError, save_registry
        try:
            save_registry({k: v for k, v in reg.items() if k != name_in_registry})
        except RegistryAccessError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1)
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

# ---------------------------------------------------------------------------
# show-data
# ---------------------------------------------------------------------------

@app.command(name="show-data")
def show_data(
    source: Path = typer.Argument(
        ..., help="YAML experiment config, .stl file, .npy pool entry, or pool directory."
    ),
    scale: Optional[float] = typer.Option(
        None, "--scale", help="Override: voxels on the longest edge."
    ),
    grid_size: Optional[int] = typer.Option(
        None, "--grid-size", "-g", help="Override: voxel grid side length."
    ),
    padding: int = typer.Option(
        2, "--padding", help="Free voxels on each side of the geometry (circumnavigation margin)."
    ),
    no_viewer: bool = typer.Option(
        False, "--no-viewer", help="Skip opening the eframe viewer."
    ),
) -> None:
    """Show geometry stats and open the voxelized geometry in the eframe viewer."""
    import collections
    import json
    import sys
    import tempfile

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(file=sys.stdout, highlight=False)

    if not source.exists():
        typer.echo(f"Error: {source} not found", err=True)
        raise typer.Exit(1)

    # --- .npy pool entry or pool directory → pool-explorer binary ---
    if source.suffix.lower() == ".npy" or (source.is_dir() and (source / "pool_meta.json").exists()):
        import subprocess
        import sys

        suffix = ".exe" if sys.platform == "win32" else ""
        candidates = [
            Path("theseo_anysearch/core/target/release") / f"pool-explorer{suffix}",
            Path("theseo_anysearch/core/target/debug") / f"pool-explorer{suffix}",
        ]
        binary = next((p for p in candidates if p.exists()), None)
        if binary is None:
            typer.echo(
                "pool-explorer binary not found. Build it with:\n"
                "  cd theseo_anysearch/core && cargo build --bin pool-explorer",
                err=True,
            )
            raise typer.Exit(1)

        if not no_viewer:
            typer.echo(f"Opening Pool Explorer: {source}")
            subprocess.run([str(binary), str(source)], check=True)
        return

    # Resolve STL path, scale, and grid_size from YAML or direct STL argument.
    stl_path: Path | None = None
    eff_scale: float = scale if scale is not None else 32.0
    eff_grid: int = grid_size if grid_size is not None else 32

    if source.suffix.lower() == ".stl":
        stl_path = source
    else:
        from theseo_anysearch.experiments.loader import load_experiment
        experiment = load_experiment(source)
        env = experiment.env
        raw_stl = getattr(env, "stl_path", None)
        if raw_stl is None:
            typer.echo("Error: no stl_path found in config", err=True)
            raise typer.Exit(1)
        stl_path = Path(raw_stl)
        if not stl_path.is_absolute() and not stl_path.exists():
            stl_path = source.parent / stl_path
        if scale is None:
            eff_scale = float(getattr(env, "scale", 32.0))
        if grid_size is None:
            eff_grid = int(getattr(env, "grid_size", 32))
        padding = int(getattr(env, "geometry_padding", padding))

    if not stl_path.exists():
        typer.echo(f"Error: STL not found: {stl_path}", err=True)
        raise typer.Exit(1)

    # Voxelize with the same logic as the env.
    from theseo_anysearch.environments.pettingzoo.multi_voxel_env import _load_stl_geometry
    console.print(
        f"[dim]Voxelizing {stl_path.name}  scale={eff_scale}  grid={eff_grid}  padding={padding}...[/dim]"
    )
    geometry = _load_stl_geometry(str(stl_path), eff_scale, eff_grid, padding=padding)

    total = eff_grid ** 3
    filled = len(geometry)
    free = total - filled
    fill_pct = 100.0 * filled / total

    # BFS connectivity on navigable (free) voxels.
    filled_set = set(geometry)
    free_cells = [
        (x, y, z)
        for x in range(1, eff_grid + 1)
        for y in range(1, eff_grid + 1)
        for z in range(1, eff_grid + 1)
        if (x, y, z) not in filled_set
    ]
    connected = 0
    if free_cells:
        visited: set[tuple[int, int, int]] = {free_cells[0]}
        q: collections.deque[tuple[int, int, int]] = collections.deque([free_cells[0]])
        while q:
            cx, cy, cz = q.popleft()
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                nb = (cx + dx, cy + dy, cz + dz)
                if (
                    nb not in visited
                    and nb not in filled_set
                    and 1 <= nb[0] <= eff_grid
                    and 1 <= nb[1] <= eff_grid
                    and 1 <= nb[2] <= eff_grid
                ):
                    visited.add(nb)
                    q.append(nb)
        connected = len(visited)
    conn_pct = 100.0 * connected / free if free > 0 else 0.0
    conn_ok = connected >= 0.8 * free

    # Geometry voxel-space bounds.
    if geometry:
        gxs = [c[0] for c in geometry]
        gys = [c[1] for c in geometry]
        gzs = [c[2] for c in geometry]
        bounds = f"x[{min(gxs)},{max(gxs)}]  y[{min(gys)},{max(gys)}]  z[{min(gzs)},{max(gzs)}]"
    else:
        bounds = "empty"

    # Print Rich summary panel.
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style="dim", min_width=20)
    table.add_column(overflow="fold")
    table.add_row("stl", stl_path.name)
    table.add_row("scale", f"{eff_scale:.1f}  (longest edge -> {eff_scale:.0f} voxels)")
    table.add_row("grid", f"{eff_grid}^3 = {total:,} voxels total")
    table.add_row("padding", str(padding))
    table.add_row("", "")
    table.add_row("filled (geometry)", f"{filled:,}  ({fill_pct:.1f}%)")
    table.add_row("free (navigable)", f"{free:,}  ({100-fill_pct:.1f}%)")
    conn_label = "[green]OK[/green]" if conn_ok else "[red]LOW[/red]"
    table.add_row("connected free", f"{connected:,} / {free:,}  ({conn_pct:.1f}%)  {conn_label}")
    table.add_row("geometry bounds", bounds)
    try:
        console.print(Panel(table, title="[bold]Geometry preview[/bold]", title_align="left"), new_line_start=True)
    except UnicodeEncodeError:
        typer.echo(
            f"\nGeometry preview: {stl_path.name}  scale={eff_scale}  grid={eff_grid}^3"
            f"\n  filled={filled:,} ({fill_pct:.1f}%)  free={free:,}  connected={connected:,} ({conn_pct:.1f}%)"
            f"\n  bounds: {bounds}"
        )

    if not no_viewer:
        import subprocess
        from theseo_anysearch.cli.commands.replay import _find_binary

        # Write a geometry-only trajectory JSON (no agent steps) and open in eframe.
        traj = {
            "experiment_name": stl_path.stem,
            "run_id": "preview",
            "iteration": 0,
            "episode_reward_mean": 0.0,
            "agent_count": 0,
            "max_steps": 0,
            "obs_mode": "box",
            "episode": {
                "total_reward": 0.0,
                "steps_taken": 0,
                "success": False,
                "init_filled": [[x, y, z] for x, y, z in geometry],
                "start_positions": [],
                "goal_positions": [],
                "steps": [],
            },
        }
        tmp = Path(tempfile.mktemp(suffix="_geometry_preview.json"))
        tmp.write_text(json.dumps(traj))
        typer.echo(f"Opening eframe viewer...")
        try:
            binary = _find_binary()
            subprocess.run([str(binary), str(tmp)], check=True)
        except FileNotFoundError as exc:
            typer.echo(f"voxel-replay binary not found: {exc}", err=True)
        finally:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

@app.command(name="extract")
def extract(
    sources: list[Path] = typer.Argument(
        ...,
        help=(
            "STL files, folders of STL files, or a single YAML extract config. "
            "If a YAML file is provided all other flags are read from it."
        ),
    ),
    target: int = typer.Option(40, "--target", "-n", help="Base geometries per source STL."),
    scale_min: float = typer.Option(100.0, "--scale-min", help="Minimum scale (voxels on longest edge)."),
    scale_max: float = typer.Option(500.0, "--scale-max", help="Maximum scale."),
    rotate: bool = typer.Option(True, "--rotate/--no-rotate", help="Random SO(3) rotation per sample."),
    pool_dir: Optional[Path] = typer.Option(None, "--pool-dir", help="Output directory for the geometry pool."),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel worker processes."),
    resume: bool = typer.Option(False, "--resume", help="Skip already-written .npy files."),
    min_fill_pct: float = typer.Option(5.0, "--min-fill-pct", help="Reject if filled < N%%."),
    min_free_pct: float = typer.Option(10.0, "--min-free-pct", help="Reject if free < N%%."),
    no_connectivity_check: bool = typer.Option(False, "--no-connectivity-check"),
    padding: int = typer.Option(2, "--padding", help="Free voxels on each side of the geometry."),
    seed: int = typer.Option(0, "--seed", help="Base random seed."),
) -> None:
    """Build a geometry pool from STL files for training-time diversity.

    Examples::

      anysearch extract usage/geometries/ --target 40 \\
          --scale-min 100 --scale-max 500 --rotate \\
          --pool-dir runtime/geometry_pools/highres

      anysearch extract usage/extract/highres.yaml

      anysearch extract usage/geometries/ --target 40 --resume \\
          --pool-dir runtime/geometry_pools/highres
    """
    from theseo_anysearch.cli.commands.extract import run_extract

    # --- YAML config path? ---
    stl_files: list[Path] = []
    map_files: list[Path] = []
    eff_pool_dir = pool_dir
    eff_target = target
    eff_scale_min = scale_min
    eff_scale_max = scale_max
    eff_rotate = rotate
    eff_workers = workers
    eff_resume = resume
    eff_min_fill_pct = min_fill_pct
    eff_min_free_pct = min_free_pct
    eff_connectivity_check = not no_connectivity_check
    eff_padding = padding
    eff_seed = seed
    # grid_size is derived after all params are resolved (see below)

    if len(sources) == 1 and sources[0].suffix.lower() in (".yaml", ".yml"):
        # Load config from YAML
        import yaml  # type: ignore[import-untyped]
        cfg = yaml.safe_load(sources[0].read_text())
        ec = cfg.get("extract", {})
        eff_pool_dir = Path(ec.get("pool_dir", str(pool_dir or "runtime/geometry_pools/default")))
        eff_target = ec.get("target_per_source", target)
        sr = ec.get("scale_range", [scale_min, scale_max])
        eff_scale_min, eff_scale_max = float(sr[0]), float(sr[1])
        eff_rotate = ec.get("rotate", rotate)
        eff_workers = ec.get("workers", workers)
        eff_min_fill_pct = ec.get("min_fill_pct", min_fill_pct)
        eff_min_free_pct = ec.get("min_free_pct", min_free_pct)
        eff_connectivity_check = ec.get("connectivity_check", not no_connectivity_check)
        raw_sources = ec.get("sources", [])
        for s in raw_sources:
            p = Path(s)
            if not p.is_absolute():
                p = sources[0].parent / p
            if p.is_dir():
                stl_files.extend(sorted(p.glob("*.stl")))
                map_files.extend(sorted(p.glob("*.3dmap.zip")))
            elif p.suffix.lower() == ".stl" and p.exists():
                stl_files.append(p)
            elif p.name.endswith(".3dmap.zip") and p.exists():
                map_files.append(p)
            else:
                typer.echo(f"Warning: source not found: {p}", err=True)
    else:
        # Expand folders and collect STL + map files from CLI args
        for src in sources:
            if src.is_dir():
                stl_files.extend(sorted(src.glob("*.stl")))
                map_files.extend(sorted(src.glob("*.3dmap.zip")))
            elif src.suffix.lower() == ".stl" and src.exists():
                stl_files.append(src)
            elif src.name.endswith(".3dmap.zip") and src.exists():
                map_files.append(src)
            else:
                typer.echo(f"Warning: {src} is not a .stl/.3dmap.zip file or folder — skipping", err=True)

    if not stl_files and not map_files:
        typer.echo("Error: no .stl or .3dmap.zip files found in the given sources.", err=True)
        raise typer.Exit(1)

    if eff_pool_dir is None:
        typer.echo("Error: --pool-dir is required (or set pool_dir in YAML config).", err=True)
        raise typer.Exit(1)

    # STL grid_size derived from scale_max; map grid_size = int(scale_max) directly (crop window)
    eff_grid_size = int(eff_scale_max) + 2 * eff_padding + 1
    eff_map_grid_size = int(eff_scale_max)

    parts = []
    if stl_files:
        parts.append(f"{len(stl_files)} STL (grid={eff_grid_size} derived)")
    if map_files:
        parts.append(f"{len(map_files)} map (grid={eff_map_grid_size})")
    typer.echo(
        f"Sources: {', '.join(parts)}  |  "
        f"target={eff_target}/source  |  "
        f"scale=[{eff_scale_min:.0f},{eff_scale_max:.0f}]  |  rotate={eff_rotate}"
    )
    typer.echo(f"Pool dir: {eff_pool_dir}")

    run_extract(
        stl_files=stl_files,
        pool_dir=eff_pool_dir,
        target_per_source=eff_target,
        grid_size=eff_grid_size,
        scale_range=(eff_scale_min, eff_scale_max),
        padding=eff_padding,
        rotate=eff_rotate,
        workers=eff_workers,
        resume=eff_resume,
        min_fill_pct=eff_min_fill_pct,
        min_free_pct=eff_min_free_pct,
        connectivity_check=eff_connectivity_check,
        seed=eff_seed,
        map_files=map_files,
        map_grid_size=eff_map_grid_size,
    )


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
    """Run the AnySearch Typer application.

    Returns
    -------
    None
        This function delegates to the root Typer app.
    """
    app()


if __name__ == "__main__":
    main()
