from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Launch the MLflow tracking UI for an experiment output directory.")


@app.callback(invoke_without_command=True)
def mlflow_ui(
    output_dir: Path = typer.Option(
        Path("runtime/experiments"),
        help="Experiment output directory (must contain mlflow.db or a custom tracking URI).",
    ),
    port: int = typer.Option(5000, help="Port to serve the MLflow UI on."),
    host: str = typer.Option("127.0.0.1", help="Host to bind the MLflow UI to."),
    db: str = typer.Option(
        "mlflow.db",
        help="SQLite filename inside --output-dir. Ignored when --tracking-uri is set.",
    ),
    tracking_uri: str = typer.Option(
        "",
        help="Override tracking URI (e.g. sqlite:///path/to/mlflow.db or http://...).",
    ),
) -> None:
    """Launch the MLflow UI pointed at the experiment output directory."""
    if tracking_uri:
        uri = tracking_uri
    else:
        db_path = (output_dir / db).resolve()
        if not db_path.exists():
            typer.echo(
                f"[error] MLflow database not found: {db_path}\n"
                f"Run an experiment first, or pass --output-dir / --tracking-uri.",
                err=True,
            )
            raise typer.Exit(code=1)
        uri = f"sqlite:///{db_path}"

    typer.echo(f"Starting MLflow UI → {uri}")
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
