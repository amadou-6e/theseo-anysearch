"""Authoritative workspace discovery for the native AnySearch UI.

The UI intentionally receives structured data from this module instead of
reimplementing YAML or Pydantic validation in Rust.  CLI-created artifacts and
UI-created artifacts are therefore discovered through the same files.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Diagnostic(BaseModel):
    """One unmodified configuration diagnostic."""

    path: str
    message: str


class WorkspaceFile(BaseModel):
    """One file visible in the workspace tree."""

    path: str
    kind: Literal["file", "yaml", "anysearch", "invalid_anysearch"]
    diagnostics: tuple[Diagnostic, ...] = ()


class WorkspaceRun(BaseModel):
    """One self-describing run discovered from its manifest."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    path: str
    status: str
    source_yaml: str | None = None
    algorithm: str | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)


class WorkspaceIndex(BaseModel):
    """Disposable index reconstructed from a single workspace."""

    schema_version: int = 1
    workspace: str
    files: tuple[WorkspaceFile, ...]
    runs: tuple[WorkspaceRun, ...]
    file_count: int
    yaml_count: int
    configuration_count: int
    invalid_configuration_count: int


_ANYSEARCH_ROOT_KEYS = frozenset(
    {"experiment", "env", "training", "evaluation", "algorithm_config", "tune_config"}
)
_GENERATED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".worktrees",
        "__pycache__",
        "build",
        "dist",
        "mlruns",
        "pytest_tmp_root",
        "target",
        "theseo_anysearch.egg-info",
    }
)


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _diagnostics(exc: Exception) -> tuple[Diagnostic, ...]:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        return tuple(
            Diagnostic(
                path=".".join(str(part) for part in error.get("loc", ())),
                message=str(error.get("msg", error)),
            )
            for error in errors()
        )
    return (Diagnostic(path="", message=str(exc)),)


def validate_configuration(path: Path) -> tuple[Diagnostic, ...]:
    """Validate one YAML with the same loader used by training and Tune."""

    from theseo_anysearch.environment_rules import preflight_environment_rules
    from theseo_anysearch.experiments.loader import load_experiment

    try:
        configuration = load_experiment(path)
        preflight_environment_rules(configuration, path)
    except Exception as exc:
        return _diagnostics(exc)
    return ()


def _classify_yaml(path: Path) -> tuple[str, tuple[Diagnostic, ...]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "yaml", _diagnostics(exc)
    if not isinstance(document, dict) or not (_ANYSEARCH_ROOT_KEYS & document.keys()):
        return "yaml", ()
    diagnostics = validate_configuration(path)
    return ("invalid_anysearch", diagnostics) if diagnostics else ("anysearch", ())


def _read_run(path: Path, workspace: Path) -> WorkspaceRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or path.parent.name)
    source = payload.get("source_yaml") or payload.get("config_path")
    return WorkspaceRun(
        run_id=run_id,
        path=_relative(path.parent, workspace),
        status=str(payload.get("status", "unknown")),
        source_yaml=str(source) if source is not None else None,
        algorithm=payload.get("algorithm"),
        manifest=payload,
    )


def _workspace_files(root: Path):
    """Yield user-visible files without descending into generated dependency trees."""

    for current, directories, names in os.walk(root):
        directories[:] = sorted(
            directory for directory in directories if directory not in _GENERATED_DIRECTORIES
        )
        current_path = Path(current)
        if {"run.json", "ray_runtime.json"} & set(names):
            directories.clear()
            for name in sorted(
                {"run.json", "ray_runtime.json", "experiment.yaml", "terminal.log"} & set(names)
            ):
                yield current_path.joinpath(name)
            continue
        for name in sorted(names):
            yield current_path.joinpath(name)


def scan_workspace(workspace: Path) -> WorkspaceIndex:
    """Rebuild a workspace index solely from ordinary files and run manifests."""

    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files: list[WorkspaceFile] = []
    runs: list[WorkspaceRun] = []
    yaml_count = 0
    configuration_count = 0
    invalid_count = 0
    paths = list(_workspace_files(root))
    artifact_roots = tuple(
        path.parent for path in paths if path.name in {"run.json", "ray_runtime.json"}
    )
    yaml_paths = [
        path
        for path in paths
        if path.suffix.lower() in {".yaml", ".yml"}
        and not any(path.is_relative_to(artifact_root) for artifact_root in artifact_roots)
    ]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(yaml_paths)))) as executor:
        yaml_results = dict(zip(yaml_paths, executor.map(_classify_yaml, yaml_paths), strict=True))
    for path in paths:
        relative = _relative(path, root)
        if path.name == "run.json":
            runs.append(_read_run(path, root))
        suffix = path.suffix.lower()
        if suffix not in {".yaml", ".yml"}:
            files.append(WorkspaceFile(path=relative, kind="file"))
            continue
        yaml_count += 1
        kind, diagnostics = yaml_results.get(path, ("yaml", ()))
        if kind == "anysearch":
            configuration_count += 1
        elif kind == "invalid_anysearch":
            invalid_count += 1
        files.append(WorkspaceFile(path=relative, kind=kind, diagnostics=diagnostics))
    return WorkspaceIndex(
        workspace=str(root),
        files=tuple(files),
        runs=tuple(runs),
        file_count=len(files),
        yaml_count=yaml_count,
        configuration_count=configuration_count,
        invalid_configuration_count=invalid_count,
    )
