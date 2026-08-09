"""Launcher for the optional Streamlit explainability interface."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def launch_explain_ui(run: str, checkpoint: str, port: int) -> None:
    """Launch the UI in a Streamlit child process and propagate failures."""

    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(
            "the explainability UI is not installed; install "
            "'theseo-anysearch[explain-ui]'"
        )
    app_path = Path(
        __file__
    ).parents[2].joinpath("rllib", "explain", "ui", "app.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--",
            "--run-ref",
            run,
            "--checkpoint",
            checkpoint,
        ],
        check=True,
    )
