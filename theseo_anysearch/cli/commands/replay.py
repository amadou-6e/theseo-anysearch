"""CLI helpers for replaying saved trajectories and tune sweeps.

Examples
--------
Replay the best episode from a run::

    anysearch replay ppo-maps:7367dc57 --best

Browse all trajectories in a sweep::

    anysearch replay multi_agent_ppo_asha:latest
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import typer
from typer.core import TyperGroup


class _SoftGroup(TyperGroup):
    """TyperGroup that treats unknown subcommand names as positional args.

    In Click 8.3+, Group.invoke checks ``ctx._protected_args`` first: if it is
    non-empty it calls ``resolve_command`` and immediately ``assert cmd is not
    None``, so there is no way to handle unknown names after the fact.  The fix
    is to intercept in ``parse_args``: if the first remaining token is not a
    registered subcommand name we move it out of ``_protected_args`` into
    ``ctx.args``, leaving ``_protected_args`` empty.  An empty
    ``_protected_args`` causes Click to take the ``invoke_without_command``
    branch instead, which calls only the group callback — which then reads the
    ref from ``ctx.args``.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        rest = super().parse_args(ctx, args)
        # _protected_args[0] is the candidate subcommand name.  If it is not a
        # known command, demote it to ctx.args so the callback can handle it.
        protected = getattr(ctx, "_protected_args", [])
        if protected and self.get_command(ctx, protected[0]) is None:
            ctx.args = [*protected, *ctx.args]
            ctx._protected_args = []
        return rest


app = typer.Typer(
    cls=_SoftGroup,
    invoke_without_command=True,
    help=(
        "Replay recorded VoxelEnv trajectories in the eframe viewer.\n\n"
        "Usage:\n\n"
        "  anysearch replay <name>             — auto-detect training or tune sweep\n\n"
        "  anysearch replay <name:tag>         — browse all trials (T/Y keys)\n\n"
        "  anysearch replay <name:run_id>      — all periodic snapshots\n\n"
        "  anysearch replay <name:run_id> --best\n\n"
        "  anysearch replay file <path.json>   — open any trajectory file directly\n\n"
        "Training vs tuning and the STL geometry are inferred automatically from\n"
        "the saved run configuration — no extra flags needed."
    ),
    # allow_interspersed_args lets options like --best appear after the ref argument.
    context_settings={"allow_interspersed_args": True},
)

_BINARY_NAME = "voxel-replay"
_BINARY_REL = Path("theseo_anysearch/core/target/debug") / _BINARY_NAME
_BINARY_REL_RELEASE = Path("theseo_anysearch/core/target/release") / _BINARY_NAME


def _find_binary() -> Path:
    """Locate the voxel-replay binary, preferring release over debug."""
    suffix = ".exe" if sys.platform == "win32" else ""
    for candidate in (_BINARY_REL_RELEASE, _BINARY_REL):
        p = candidate.with_suffix(suffix)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"voxel-replay binary not found. Build it with:\n"
        f"  cd theseo_anysearch/core && cargo build --bin voxel-replay"
    )


def _find_trajectory(run_dir: Path, iteration: int) -> Path:
    """Return the trajectory JSON path for a specific iteration."""
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        raise FileNotFoundError(
            f"No trajectories directory in {run_dir}. "
            "Re-run the experiment with trajectory_every > 0."
        )
    p = traj_dir / f"iter_{iteration:06d}.json"
    if not p.exists():
        available = sorted(traj_dir.glob("iter_*.json"))
        hint = (
            ", ".join(f.stem.split("iter_")[-1] for f in available)
            if available
            else "none"
        )
        raise FileNotFoundError(
            f"iter_{iteration:06d}.json not found. Available: {hint}"
        )
    return p


def _all_iter_trajectories(run_dir: Path) -> list[Path]:
    """Return all iter_*.json files sorted by iteration."""
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        raise FileNotFoundError(
            f"No trajectories directory in {run_dir}. "
            "Re-run the experiment with trajectory_every > 0."
        )
    files = sorted(traj_dir.glob("iter_*.json"))
    if not files:
        # Fall back to best.json if no periodic snapshots exist
        best = traj_dir / "best.json"
        if best.exists():
            return [best]
        raise FileNotFoundError(f"No trajectory files found in {traj_dir}.")
    return files


def _replay_sweep_dir(sweep_dir: Path) -> None:
    """Open a sweep directory (containing Ray Tune trial subdirs) in the viewer."""
    trial_dirs = [
        p for p in sweep_dir.iterdir()
        if p.is_dir() and (p / "ray_runtime.json").exists()
    ]
    if not trial_dirs:
        typer.echo(
            f"No trial directories found under {sweep_dir}.\n"
            "Make sure this is a valid Ray Tune sweep output directory.",
            err=True,
        )
        raise typer.Exit(1)
    binary = _find_binary()
    typer.echo(f"Opening sweep ({len(trial_dirs)} trials): {sweep_dir}")
    subprocess.run([str(binary), "--tune-dir", str(sweep_dir)], check=True)


# ---------------------------------------------------------------------------
# Default command: anysearch replay <ref> [--best] [--iter N]
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def replay(
    ctx: typer.Context,
    best: bool = typer.Option(False, "--best", help="Open best.json only."),
    iteration: Optional[int] = typer.Option(
        None, "--iter", "-i", help="Open a specific iteration snapshot."
    ),
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        "--output-dir",
        help="Fallback base experiments directory (used when ref has no directory component).",
    ),
) -> None:
    """
    Replay a run's trajectories in the eframe viewer.

    \b
    anysearch replay <name:run_id>            — all periodic snapshots
    anysearch replay <name:run_id> --best     — best.json only
    anysearch replay <name:run_id> --iter 5   — a single iteration
    anysearch replay <name:tag>               — sweep: browse all trials (T/Y keys)
    anysearch replay <name>                   — auto-open single tag or list available tags

    REF is the first non-option argument (options may appear before or after it).
    """
    # ref is the first positional arg.  When a known subcommand name (file/
    # tune/list) is given, Click dispatches it normally and _protected_args
    # holds the name.  When an unknown ref is given, _SoftGroup.parse_args has
    # moved it to ctx.args so Click takes the invoke_without_command path and
    # calls only this callback.
    protected = getattr(ctx, "_protected_args", ctx.protected_args)
    if protected:
        # Known subcommand — let Click dispatch it.
        return
    ref: Optional[str] = ctx.args[0] if ctx.args else None

    if ref is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    from theseo_anysearch.experiments.runner import _find_run_dir
    from theseo_anysearch.cli.registry import resolve_ref, resolve_config_and_dir

    experiment_dir, identifier = resolve_ref(ref)

    # Resolve the YAML for this ref without including the :identifier suffix.
    def _config_path_for(exp_dir: Path) -> Path | None:
        try:
            # Try the registered name (first colon-segment of ref, handling Windows drive letters)
            from theseo_anysearch.cli.registry import load_registry
            parts = ref.split(":")
            name = parts[2] if (len(parts) >= 3 and len(parts[0]) == 1 and parts[0].isalpha()) else parts[0]
            reg = load_registry()
            if name in reg:
                p = Path(reg[name])
                if p.suffix in (".yaml", ".yml") and p.exists():
                    return p
            # Fall back to looking for a unique YAML in exp_dir
            yamls = list(exp_dir.glob("*.yaml")) + list(exp_dir.glob("*.yml"))
            if len(yamls) == 1:
                return yamls[0]
        except Exception:
            pass
        return None

    def _read_output_dir(config_path: Path) -> Path | None:
        try:
            import yaml as _yaml
            raw = _yaml.safe_load(config_path.read_text()) or {}
            out_raw = raw.get("experiment", {}).get("output_dir")
            if out_raw is not None:
                p = Path(out_raw)
                return (config_path.parent / p).resolve() if not p.is_absolute() else p
        except Exception:
            pass
        return None

    def _read_exp_name(config_path: Path) -> str | None:
        try:
            import yaml as _yaml
            return (_yaml.safe_load(config_path.read_text()) or {}).get("experiment", {}).get("name")
        except Exception:
            return None

    # If identifier is None, try to resolve from the experiment's output directory
    if identifier is None:
        candidate = Path(ref)
        if candidate.exists() and candidate.is_dir():
            _replay_sweep_dir(candidate)
            raise typer.Exit()

        # Resolve output_dir from YAML and auto-detect a single sweep or run
        try:
            config_path = _config_path_for(experiment_dir)
            resolved_out = _read_output_dir(config_path) if config_path else None
            exp_name = _read_exp_name(config_path) if config_path else None
            if resolved_out:
                search = (resolved_out / exp_name) if exp_name else resolved_out
                sweep_dirs = []
                if search.exists():
                    for child in search.iterdir():
                        if child.is_dir() and any((d / "ray_runtime.json").exists() for d in child.iterdir() if d.is_dir()):
                            sweep_dirs.append(child)
                if len(sweep_dirs) == 1:
                    _replay_sweep_dir(sweep_dirs[0])
                    raise typer.Exit()
                if len(sweep_dirs) > 1:
                    typer.echo("Available tags — specify one:", err=True)
                    for s in sorted(sweep_dirs, key=lambda p: p.name):
                        typer.echo(f"  anysearch replay {ref}:{s.name}", err=True)
                    raise typer.Exit(1)
                # No sweep tags — check for single-run dirs
                if search.exists():
                    run_dirs = [d for d in search.iterdir() if d.is_dir() and (d / "run.json").exists()]
                    if run_dirs:
                        typer.echo("Available runs — specify one:", err=True)
                        for d in sorted(run_dirs, key=lambda p: p.name):
                            typer.echo(f"  anysearch replay {ref}:{d.name}", err=True)
                        raise typer.Exit(1)
        except (typer.Exit, subprocess.CalledProcessError, SystemExit):
            raise
        except Exception:
            pass

        typer.echo(
            f"Error: no runs found for '{ref}'. "
            "Use: anysearch replay <name:tag>  or  anysearch replay <name:run_id>",
            err=True,
        )
        raise typer.Exit(1)

    # Resolve output_dir and experiment name from the YAML via the already-known experiment_dir.
    _cfg = _config_path_for(experiment_dir)
    resolved_output = _read_output_dir(_cfg) if _cfg else None
    _exp_name = _read_exp_name(_cfg) if _cfg else None

    base_root = resolved_output or (experiment_dir if experiment_dir.exists() else output_dir)
    if _exp_name and (base_root / _exp_name).exists():
        base_root = base_root / _exp_name
    search_root = base_root

    # Check if identifier resolves to a sweep tag (dir whose children have ray_runtime.json)
    sweep_candidate = search_root / identifier
    if sweep_candidate.is_dir() and any(
        p.is_dir() and (p / "ray_runtime.json").exists()
        for p in sweep_candidate.iterdir()
    ):
        _replay_sweep_dir(sweep_candidate)
        raise typer.Exit()

    # Otherwise treat identifier as a run_id
    run_dir = _find_run_dir(search_root, identifier)
    traj_dir = run_dir / "trajectories"
    binary = _find_binary()

    if best:
        p = traj_dir / "best.json"
        if not p.exists():
            typer.echo(f"best.json not found in {traj_dir}.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Replaying best trajectory: {p}")
        files = [p]
    elif iteration is not None:
        files = [_find_trajectory(run_dir, iteration)]
        typer.echo(f"Replaying iteration {iteration}: {files[0]}")
    else:
        files = _all_iter_trajectories(run_dir)
        typer.echo(f"Replaying {len(files)} iteration(s) from {traj_dir}")

    subprocess.run([str(binary)] + [str(f) for f in files], check=True)
    raise typer.Exit()


# ---------------------------------------------------------------------------
# replay file  (open any trajectory JSON by path)
# ---------------------------------------------------------------------------

@app.command()
def file(
    path: Path = typer.Argument(..., help="Path to a trajectory JSON file."),
) -> None:
    """Open an arbitrary trajectory JSON file in the eframe viewer."""
    if not path.exists():
        typer.echo(f"File not found: {path}", err=True)
        raise typer.Exit(1)
    binary = _find_binary()
    typer.echo(f"Replaying: {path}")
    subprocess.run([str(binary), str(path)], check=True)


# ---------------------------------------------------------------------------
# replay tune  (open a full tune sweep with trial navigation)
# ---------------------------------------------------------------------------

@app.command()
def tune(
    tune_dir: Path = typer.Argument(
        ...,
        help="Path to a tune sweep output directory (e.g. runtime/experiments/v11).",
    ),
) -> None:
    """
    Open a tune sweep directory in the eframe viewer with trial navigation.

    \b
    Keyboard shortcuts inside the viewer:
      T / Y       next / prev trial  (sorted best-first by reward)
      [ / ]       next / prev iteration within the current trial
      ← / →       step forward / backward
      Space       play / pause
    """
    if not tune_dir.exists():
        typer.echo(f"Directory not found: {tune_dir}", err=True)
        raise typer.Exit(1)

    # Check that at least one trial subdir exists (identified by ray_runtime.json).
    trial_dirs = [
        p for p in tune_dir.iterdir()
        if p.is_dir() and (p / "ray_runtime.json").exists()
    ]
    if not trial_dirs:
        typer.echo(
            f"No trial directories found under {tune_dir}.\n"
            "Make sure this is a valid Ray Tune sweep output directory.",
            err=True,
        )
        raise typer.Exit(1)

    binary = _find_binary()
    typer.echo(f"Opening tune sweep ({len(trial_dirs)} trials): {tune_dir}")
    subprocess.run([str(binary), "--tune-dir", str(tune_dir)], check=True)


# ---------------------------------------------------------------------------
# replay list  (show available trajectory iterations for a run)
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_trajectories(
    ref: str = typer.Argument(..., help="Run reference: name:run_id, dir:run_id, or bare run_id."),
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        "--output-dir",
        help="Fallback base experiments directory.",
    ),
) -> None:
    """List available trajectory iterations for a run."""
    from theseo_anysearch.experiments.runner import _find_run_dir
    from theseo_anysearch.cli.registry import resolve_ref
    experiment_dir, identifier = resolve_ref(ref)
    if identifier is None:
        typer.echo(f"Error: ref must include a run_id, e.g. name:{ref}", err=True)
        raise typer.Exit(1)
    search_root = experiment_dir if experiment_dir.exists() else output_dir
    run_dir = _find_run_dir(search_root, identifier)
    run_id = identifier
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        typer.echo("No trajectories directory found.")
        raise typer.Exit(1)

    periodic = sorted(traj_dir.glob("iter_*.json"))
    best_path = traj_dir / "best.json"
    meta_path = traj_dir / "best_meta.json"

    typer.echo(f"Run: {run_id}  ({run_dir})")
    typer.echo(f"Periodic snapshots ({len(periodic)}):")
    for p in periodic:
        typer.echo(f"  {p.name}")

    if best_path.exists():
        meta = ""
        if meta_path.exists():
            import json as _json
            m = _json.loads(meta_path.read_text())
            meta = f"  (iter {m.get('iteration','?')}, reward {m.get('episode_reward_mean', '?'):.3f})"
        typer.echo(f"Best: best.json{meta}")
