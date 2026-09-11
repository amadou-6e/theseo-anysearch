"""
Global registry mapping short experiment names to YAML config files or directories.

The registry prefers a repo-local `.anysearch/registry.yaml` when the current
working directory is inside a repository root, and otherwise falls back to the
user-home registry at `~/.anysearch/registry.yaml`.

Values stored are absolute paths, either to a YAML config file or to a
directory containing one. The distinction is preserved so that experiments
that share a directory, such as `usage/experiments/tune/*.yaml`, each get
their own registry entry pointing directly to their YAML.

Reference forms accepted by `resolve_ref()`:
  name:tag      global registry lookup -> experiment_dir, identifier = tag
  name:run_id   global registry lookup -> experiment_dir, identifier = run_id
  dir:tag       direct path            -> experiment_dir, identifier = tag
  name          global registry lookup -> experiment_dir, identifier = None
  dir           direct path            -> experiment_dir, identifier = None
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml


class RegistryAccessError(RuntimeError):
    """Raised when the experiment registry cannot be read or written cleanly."""


def _repo_root_from(start: Path) -> Path | None:
    """Return the nearest repository root containing common project markers."""
    current = start.resolve()
    if current.is_file():
        current = current.parent
    markers = (".anysearch", ".git", ".claude", ".github")
    for candidate in (current, *current.parents):
        if any(candidate.joinpath(marker).exists() for marker in markers):
            return candidate
    return None


def _repo_registry_file() -> Path | None:
    """Return the preferred repo-local registry file when inside a repository."""
    override_root = os.environ.get("ANYSEARCH_REPO_ROOT")
    if override_root:
        return Path(override_root).joinpath(".anysearch", "registry.yaml")
    repo_root = _repo_root_from(Path.cwd())
    if repo_root is None:
        return None
    return repo_root.joinpath(".anysearch", "registry.yaml")


def _home_registry_file() -> Path:
    """Return the user-home registry file path."""
    return Path.home().joinpath(".anysearch", "registry.yaml")


def _registry_file() -> Path:
    """
    Return the registry path.

    Override with `ANYSEARCH_REGISTRY` for tests and CI.
    By default, use repo-local `.anysearch/registry.yaml` when a repository
    root can be found from the current working directory, otherwise fall back
    to `~/.anysearch/registry.yaml`.
    """
    override = os.environ.get("ANYSEARCH_REGISTRY")
    if override:
        return Path(override)
    repo_registry = _repo_registry_file()
    if repo_registry is not None:
        return repo_registry
    return _home_registry_file()


def _load_registry_from(path: Path) -> dict[str, str]:
    """Load a registry mapping from a specific registry file path."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return raw.get("experiments", {})


# Module-level attribute for monkeypatching in unit tests
_REGISTRY_FILE: Path = _registry_file()


def load_registry() -> dict[str, str]:
    """Return the registry mapping from the active registry search path."""
    override = os.environ.get("ANYSEARCH_REGISTRY")
    if override:
        return _load_registry_from(Path(override))

    repo_registry = _repo_registry_file()
    if repo_registry is not None and repo_registry.exists():
        return _load_registry_from(repo_registry)
    return _load_registry_from(_home_registry_file())


def save_registry(experiments: dict[str, str]) -> None:
    """
    Persist experiment registry entries to disk.

    Parameters
    ----------
    experiments : dict[str, str]
        Mapping from registered experiment names to absolute filesystem paths.

    Returns
    -------
    None
        This function writes the registry file in place.
    """
    registry_file = _registry_file()
    try:
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text(
            yaml.dump({"experiments": experiments}, default_flow_style=False, sort_keys=True)
        )
    except PermissionError as exc:
        home_registry = _home_registry_file().resolve()
        if registry_file.resolve() == home_registry:
            raise RegistryAccessError(
                "Could not write the AnySearch registry in your home directory.\n"
                f"Tried: {registry_file}\n\n"
                "Run the command from inside a repository to use a repo-local "
                "`.anysearch/registry.yaml`, or set `ANYSEARCH_REGISTRY` to a "
                "writable path."
            ) from exc
        raise RegistryAccessError(
            f"Could not write the AnySearch registry at {registry_file}."
        ) from exc


def add_experiment(path: Path, name: Optional[str] = None) -> str:
    """
    Register a YAML file or directory under name.

    Returns the name used. The absolute path is stored so the registry works
    from any current working directory.
    """
    abs_path = path.resolve()
    if name is None:
        name = _name_from_path(abs_path)
    registry = load_registry()
    registry[name] = str(abs_path)
    save_registry(registry)
    return name


def _name_from_path(path: Path) -> str:
    """
    Derive a registry name from a path.

    YAML files use the filename stem and directories use the directory basename.
    """
    if path.suffix in (".yaml", ".yml"):
        return path.stem
    return path.name


def resolve_name(name: str) -> Path:
    """
    Look up a registered name and return its absolute path.

    Raises
    ------
    KeyError
        Raised when the requested experiment name is not registered.
    """
    registry = load_registry()
    if name not in registry:
        raise KeyError(
            f"Experiment '{name}' not in registry. "
            f"Run: anysearch add <dir-or-yaml> {name}"
        )
    return Path(registry[name])


def resolve_experiment_dir(name: str) -> Path:
    """Return the experiment directory for a registered name."""
    resolved = resolve_name(name)
    if resolved.suffix in (".yaml", ".yml"):
        return resolved.parent
    return resolved


def resolve_config_and_dir(ref: str) -> tuple[Path, Path]:
    """
    Return `(config_yaml_path, experiment_dir)` for a ref string.

    Handles direct YAML paths, registered names pointing to YAML files,
    registered names pointing to directories, and bare directory paths.
    """
    candidate = Path(ref)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        resolved = candidate.resolve()
        return resolved, resolved.parent

    registry = load_registry()
    if ref in registry:
        stored = Path(registry[ref])
        if stored.suffix in (".yaml", ".yml"):
            return stored, stored.parent
        config = find_config_in_dir(stored)
        return config, stored

    resolved_dir = _resolve_dir(ref)
    config = find_config_in_dir(resolved_dir)
    return config, resolved_dir


def resolve_ref(ref: str) -> tuple[Path, Optional[str]]:
    """
    Parse a reference string into `(experiment_dir, identifier_or_none)`.

    The identifier is a sweep tag or run ID. The caller is responsible for
    interpreting it in context, such as replay or inspect mode.
    """
    parts = ref.split(":")
    if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
        if len(parts) == 2:
            return Path(ref), None
        path_str = ":".join(parts[:-1])
        identifier = parts[-1] or None
        return Path(path_str), identifier

    if ":" in ref:
        head, _, tail = ref.partition(":")
        identifier = tail or None
        experiment_dir = _resolve_dir(head)
        return experiment_dir, identifier

    return _resolve_dir(ref), None


def _resolve_dir(value: str) -> Path:
    """Resolve a string to an experiment directory path."""
    candidate = Path(value)
    if candidate.exists() or os.sep in value or "/" in value or value.startswith("."):
        if candidate.suffix in (".yaml", ".yml") and candidate.exists():
            return candidate.parent
        return candidate
    try:
        return resolve_experiment_dir(value)
    except KeyError:
        return candidate


def find_config_in_dir(directory: Path) -> Path:
    """
    Find the experiment config YAML inside an experiment directory.

    Lookup order:
    1. config.yaml
    2. experiment.yaml
    3. the single `*.yaml` file present, if there is exactly one
    """
    for name in ("config.yaml", "experiment.yaml"):
        candidate = directory.joinpath(name)
        if candidate.exists():
            return candidate
    yamls = sorted(directory.glob("*.yaml"))
    if len(yamls) == 1:
        return yamls[0]
    if len(yamls) > 1:
        names = ", ".join(path.name for path in yamls)
        raise ValueError(
            f"Multiple YAML files in {directory}: {names}. "
            "Rename one to config.yaml or use anysearch run <config-file> directly."
        )
    raise FileNotFoundError(
        f"No config YAML found in {directory}. "
        "Expected config.yaml, experiment.yaml, or a single *.yaml file."
    )
