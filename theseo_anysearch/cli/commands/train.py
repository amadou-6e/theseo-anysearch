"""CLI helpers for starting and managing training runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(help="[deprecated] Use: anysearch run <dir>")


@app.callback(invoke_without_command=True)
def train(
    stl: Path = typer.Option(..., help="Path to the STL geometry file."),
    scale: float = typer.Option(40.0, help="Voxelization scale factor."),
    agents: int = typer.Option(5, help="Number of concurrent agents."),
    max_steps: int = typer.Option(400, help="Max steps per episode."),
    seed: int = typer.Option(1337, help="Global RNG seed."),
    iterations: int = typer.Option(20, help="Number of training iterations."),
    checkpoint_interval: int = typer.Option(10, help="Iterations between checkpoints."),
    output_dir: Path = typer.Option(Path("runtime/pathfinding_videos"), help="Output directory for GIFs and summaries."),
    video_every: int = typer.Option(10, help="Episodes between GIF frames."),
    origin_x: int = typer.Option(260, help="World origin X."),
    origin_y: int = typer.Option(260, help="World origin Y."),
    origin_z: int = typer.Option(260, help="World origin Z."),
    camera_yaw: float = typer.Option(245.0, help="Camera yaw angle (degrees)."),
    camera_pitch: float = typer.Option(20.0, help="Camera pitch angle (degrees)."),
    disable_culling: bool = typer.Option(True, help="Disable back-face culling."),
) -> None:
    """Train a surface-pathfinding agent using the Rust simulation backend."""
    typer.echo(
        "[deprecated] anysearch train is deprecated. Use: anysearch run <dir>\n",
        err=True,
    )
    try:
        import theseo_core
    except ImportError:
        err = {"error": "theseo_core not found", "hint": "Build the Rust wheel with maturin develop"}
        print(json.dumps(err), file=sys.stderr)
        raise typer.Exit(code=1)

    if not stl.exists():
        err = {"error": "stl file not found", "path": str(stl)}
        print(json.dumps(err), file=sys.stderr)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    stl_ascii = stl.read_text(encoding="utf-8")

    result = theseo_core.py_train_videos(
        stl_ascii,
        origin_x,
        origin_y,
        origin_z,
        scale,
        agents,
        max_steps,
        seed,
        iterations,
        video_every,
        str(output_dir),
        camera_yaw,
        camera_pitch,
        disable_culling,
    )
    print(json.dumps(result if isinstance(result, dict) else {"result": str(result)}))
