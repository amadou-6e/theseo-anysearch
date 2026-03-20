from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Launch the MLflow tracking UI for an experiment.")


@app.callback(invoke_without_command=True)
def mlflow_ui(
    name: Optional[str] = typer.Argument(
        None,
        help="Registered experiment name (e.g. ppo-baseline). "
             "Looks up the experiment directory from the registry.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Explicit experiment output directory (overrides registry lookup).",
    ),
    port: int = typer.Option(5000, help="Port to serve the MLflow UI on."),
    host: str = typer.Option("127.0.0.1", help="Host to bind the MLflow UI to."),
    db: str = typer.Option(
        "mlflow.db",
        help="SQLite filename inside the experiment directory. Ignored when --tracking-uri is set.",
    ),
    tracking_uri: str = typer.Option(
        "",
        help="Override tracking URI (e.g. sqlite:///path/to/mlflow.db or http://...).",
    ),
) -> None:
    """Launch the MLflow UI for a registered experiment or output directory.

    \b
    anysearch mlflow ppo-baseline          -- registry lookup
    anysearch mlflow --output-dir <dir>    -- explicit directory
    anysearch mlflow --tracking-uri <uri>  -- fully custom URI
    """
    if tracking_uri:
        uri = tracking_uri
    else:
        # Resolve experiment directory
        if output_dir is not None:
            search_dir = output_dir
        elif name is not None:
            from theseo_anysearch.cli.registry import _resolve_dir
            search_dir = _resolve_dir(name)
        else:
            search_dir = Path("runtime/experiments")

        db_path = (search_dir / db).resolve()
        if not db_path.exists():
            typer.echo(
                f"[error] MLflow database not found: {db_path}\n"
                f"Run an experiment first, or pass a name / --output-dir / --tracking-uri.",
                err=True,
            )
            raise typer.Exit(code=1)
        uri = f"sqlite:///{db_path}"

    typer.echo(f"Starting MLflow UI -> {uri}")
    typer.echo(f"Open: http://{host}:{port}")

    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", uri,
        "--host", host,
        "--port", str(port),
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
