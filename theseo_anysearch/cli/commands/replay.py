from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    invoke_without_command=True,
    help="Replay recorded VoxelEnv trajectories in the eframe viewer.",
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


# ---------------------------------------------------------------------------
# Default command: anysearch replay <run_id> [--best] [--iter N]
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def replay(
    ctx: typer.Context,
    run_id: Optional[str] = typer.Argument(None, help="Run ID (8-char hex) to replay."),
    best: bool = typer.Option(False, "--best", help="Open best.json only."),
    iteration: Optional[int] = typer.Option(
        None, "--iter", "-i", help="Open a specific iteration snapshot."
    ),
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        "--output-dir",
        help="Base experiments directory.",
    ),
) -> None:
    """
    Replay a run's trajectories in the eframe viewer.

    \b
    anysearch replay <run_id>          — all periodic snapshots (default)
    anysearch replay <run_id> --best   — best.json only
    anysearch replay <run_id> --iter 5 — a single specific iteration
    """
    if ctx.invoked_subcommand is not None:
        return  # let subcommands handle it

    if run_id is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    from theseo_anysearch.experiments.runner import _find_run_dir
    run_dir = _find_run_dir(output_dir, run_id)
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
# replay list  (show available trajectory iterations for a run)
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_trajectories(
    run_id: str = typer.Argument(..., help="Run ID (8-char hex)."),
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        "--output-dir",
        help="Base experiments directory.",
    ),
) -> None:
    """List available trajectory iterations for a run."""
    from theseo_anysearch.experiments.runner import _find_run_dir
    run_dir = _find_run_dir(output_dir, run_id)
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
