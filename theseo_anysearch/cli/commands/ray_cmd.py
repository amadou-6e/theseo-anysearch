"""CLI helpers for Ray-related diagnostics and utilities."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Inspect and manage Ray cluster sessions.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anysearch_tmp_dirs() -> list[Path]:
    return sorted(Path(tempfile.gettempdir()).glob("anysearch_*"))


def _session_name_from_dir(anysearch_dir: Path) -> tuple[str, Optional[Path]]:
    """Return (session_name, session_path) for the most recent session inside an anysearch tmp dir."""
    session_dirs = sorted(anysearch_dir.glob("session_*"))
    if not session_dirs:
        return "unknown", None
    sd = session_dirs[-1]
    return sd.name, sd


def _gcs_addr(session_dir: Optional[Path]) -> str:
    if session_dir is None:
        return "?"
    gcs_file = session_dir / "gcs_address"
    if gcs_file.exists():
        return gcs_file.read_text(encoding="utf-8").strip()
    return "?"


def _extract_session_name(cmdline: str) -> Optional[str]:
    m = re.search(r"(session_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\w+)", cmdline)
    return m.group(1) if m else None


_RAY_KEYWORDS = (
    "default_worker.py",
    "--node-ip-address",
    "ray::IDLE",
    "raylet",
    "gcs_server",
    "log_monitor",
    "ray_client_server",
    "monitor.py",
)


def _is_gpu_worker(cmdline: str) -> bool:
    m = re.search(r"--num-gpus[= ](\S+)", cmdline)
    if m:
        try:
            return float(m.group(1)) > 0
        except ValueError:
            pass
    return False


def _ray_processes() -> list[dict]:
    try:
        import psutil
    except ImportError:
        typer.echo("[warn] psutil not available — cannot enumerate workers", err=True)
        return []

    procs = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline_parts = proc.info.get("cmdline") or []
            cmdline = " ".join(cmdline_parts)
            if any(kw in cmdline for kw in _RAY_KEYWORDS):
                procs.append({
                    "pid": proc.info["pid"],
                    "cmdline": cmdline,
                    "is_gpu": _is_gpu_worker(cmdline),
                    "session": _extract_session_name(cmdline),
                })
        except Exception:
            continue
    return procs


# ---------------------------------------------------------------------------
# ps
# ---------------------------------------------------------------------------

@app.command()
def ps() -> None:
    """List active Ray sessions and their worker processes."""
    W = 60

    # Discover sessions from anysearch temp dirs
    sessions: dict[str, dict] = {}  # session_name -> info dict
    for td in _anysearch_tmp_dirs():
        name, sd = _session_name_from_dir(td)
        sessions[name] = {
            "session_name": name,
            "tmp_dir": str(td),
            "session_dir": sd,
            "gcs_addr": _gcs_addr(sd),
            "pids": [],
            "gpu_pids": [],
        }

    # Attach worker processes to sessions
    orphan_pids: list[int] = []
    orphan_gpu_pids: list[int] = []
    for p in _ray_processes():
        sname = p["session"]
        if sname and sname in sessions:
            sessions[sname]["pids"].append(p["pid"])
            if p["is_gpu"]:
                sessions[sname]["gpu_pids"].append(p["pid"])
        elif sname:
            # Process belongs to a session whose tmp dir we didn't find
            if sname not in sessions:
                sessions[sname] = {
                    "session_name": sname,
                    "tmp_dir": "?",
                    "session_dir": None,
                    "gcs_addr": "?",
                    "pids": [],
                    "gpu_pids": [],
                }
            sessions[sname]["pids"].append(p["pid"])
            if p["is_gpu"]:
                sessions[sname]["gpu_pids"].append(p["pid"])
        else:
            orphan_pids.append(p["pid"])
            if p["is_gpu"]:
                orphan_gpu_pids.append(p["pid"])

    # Orphan processes → "unknown" session
    if orphan_pids:
        if "unknown" not in sessions:
            sessions["unknown"] = {
                "session_name": "unknown",
                "tmp_dir": "?",
                "session_dir": None,
                "gcs_addr": "?",
                "pids": [],
                "gpu_pids": [],
            }
        sessions["unknown"]["pids"].extend(orphan_pids)
        sessions["unknown"]["gpu_pids"].extend(orphan_gpu_pids)

    # Remove empty sessions (tmp dir found but no processes)
    sessions = {k: v for k, v in sessions.items() if v["pids"] or v["tmp_dir"] != "?"}

    total_workers = sum(len(v["pids"]) for v in sessions.values())

    typer.echo("\n" + "-" * W)
    typer.echo(f"  Active Ray sessions: {len(sessions)}")
    typer.echo("-" * W)

    for info in sessions.values():
        pids = info["pids"]
        gpu_count = len(info["gpu_pids"])
        pid_str = ", ".join(str(p) for p in pids)
        gpu_str = f"  [{gpu_count} on GPU]" if gpu_count else ""
        typer.echo(f"\n  Session : {info['session_name']}")
        typer.echo(f"  Workers : {len(pids)} PIDs — [{pid_str}]{gpu_str}")
        typer.echo(f"  GCS     : {info['gcs_addr']}")
        typer.echo(f"  Tmp dir : {info['tmp_dir']}")

    typer.echo("\n" + "-" * W)
    typer.echo(f"  Total workers: {total_workers}")
    typer.echo("-" * W)
    typer.echo("\n  To stop: anysearch ray stop\n")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

def _ray_exe() -> str:
    """Return the path to the ray executable in the same venv as the current interpreter."""
    import shutil

    # Prefer the ray binary sitting next to the current Python executable
    scripts_dir = Path(sys.executable).parent
    for candidate in (scripts_dir / "ray", scripts_dir / "ray.exe"):
        if candidate.exists():
            return str(candidate)

    # Fall back to whatever is on PATH
    found = shutil.which("ray")
    if found:
        return found

    typer.echo("[error] ray executable not found. Is Ray installed in this environment?", err=True)
    raise typer.Exit(1)


@app.command()
def stop(
    force: bool = typer.Option(False, "--force", "-f", help="Kill worker processes directly if `ray stop` fails."),
) -> None:
    """Stop all active Ray sessions."""
    import subprocess

    typer.echo("Stopping Ray...")
    result = subprocess.run(
        [_ray_exe(), "stop", "--force"],
        capture_output=False,
    )

    if result.returncode != 0 and force:
        typer.echo("[warn] ray stop failed — killing Ray processes directly...", err=True)
        try:
            import psutil
        except ImportError:
            typer.echo("[error] psutil not available for force kill.", err=True)
            raise typer.Exit(1)
        killed = 0
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(p.info.get("cmdline") or [])
                if any(kw in cmdline for kw in _RAY_KEYWORDS):
                    p.kill()
                    killed += 1
            except Exception:
                continue
        typer.echo(f"Killed {killed} Ray worker process(es).")
    elif result.returncode == 0:
        typer.echo("Ray stopped.")
