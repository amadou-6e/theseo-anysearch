"""Launcher for the native replayer explainability interface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from theseo_anysearch.cli.commands.replay import _all_iter_trajectories, _find_binary
from theseo_anysearch.rllib.explain.service import resolve_run_dir


def launch_explain_ui(run: str, checkpoint: str) -> None:
    """Launch the native replayer with its observation editor open."""
    run_dir = resolve_run_dir(run)
    files = _all_iter_trajectories(run_dir)
    if not files:
        raise FileNotFoundError(f"no trajectories found under {run_dir}")
    environment = dict(os.environ)
    environment["ANYSEARCH_PYTHON"] = sys.executable
    subprocess.run(
        [
            str(_find_binary()),
            "--explain-run",
            str(run_dir),
            "--checkpoint",
            checkpoint,
            "--open-observation-editor",
            *[str(path) for path in files],
        ],
        env=environment,
        check=True,
    )
