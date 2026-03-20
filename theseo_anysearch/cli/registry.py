"""
Global registry mapping short experiment names to YAML config files or directories.

The registry lives at ~/.anysearch/registry.yaml (user home) so that
`anysearch list` returns the same result from any working directory.

Values stored are absolute paths — either to a .yaml config file or to a
directory containing one. The distinction is preserved so that experiments
that share a directory (e.g. usage/experiments/tune/*.yaml) each get their
own registry entry pointing directly to their YAML.

Reference forms accepted by resolve_ref():
  name:tag      global registry lookup → experiment_dir, identifier = tag
  name:run_id   global registry lookup → experiment_dir, identifier = run_id
  dir:tag       direct path            → experiment_dir, identifier = tag
  name          global registry lookup → experiment_dir, identifier = None
  dir           direct path            → experiment_dir, identifier = None
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

def _registry_file() -> Path:
    """
    Return the registry path.
    Override with ANYSEARCH_REGISTRY env var (used in tests / CI).
    Default: ~/.anysearch/registry.yaml (global, CWD-independent).
    """
    import os
    override = os.environ.get("ANYSEARCH_REGISTRY")
    if override:
        return Path(override)
    return Path.home() / ".anysearch" / "registry.yaml"


# Module-level attribute for monkeypatching in unit tests
_REGISTRY_FILE: Path = _registry_file()


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_registry() -> dict[str, str]:
    """Return {name: abs_path_str} from the registry file."""
    f = _registry_file()
    if not f.exists():
        return {}
    raw = yaml.safe_load(f.read_text()) or {}
    return raw.get("experiments", {})


def save_registry(experiments: dict[str, str]) -> None:
    f = _registry_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        yaml.dump({"experiments": experiments}, default_flow_style=False, sort_keys=True)
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def add_experiment(path: Path, name: Optional[str] = None) -> str:
    """
    Register a YAML file or directory under name. Returns the name used.
    The absolute path is stored so the registry works from any CWD.
    """
    abs_path = path.resolve()
    if name is None:
        name = _name_from_path(abs_path)
    reg = load_registry()
    reg[name] = str(abs_path)
    save_registry(reg)
    return name


def _name_from_path(path: Path) -> str:
    """
    Derive a registry name from a path.
    - YAML file  → filename stem  (multi_agent_ppo_asha.yaml → multi_agent_ppo_asha)
    - Directory  → directory basename
    """
    if path.suffix in (".yaml", ".yml"):
        return path.stem
    return path.name


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def resolve_name(name: str) -> Path:
    """
    Look up a registered name → absolute path (YAML file or directory).
    Raises KeyError if absent.
    """
    reg = load_registry()
    if name not in reg:
        raise KeyError(
            f"Experiment '{name}' not in registry. "
            f"Run:  anysearch add <dir-or-yaml> {name}"
        )
    return Path(reg[name])


def resolve_experiment_dir(name: str) -> Path:
    """Return the experiment directory for a registered name."""
    p = resolve_name(name)
    if p.suffix in (".yaml", ".yml"):
        return p.parent
    return p


def resolve_config_and_dir(ref: str) -> tuple[Path, Path]:
    """
    Return (config_yaml_path, experiment_dir) for a ref string (no colon splitting).

    Handles:
    - Direct .yaml path on disk
    - Registered name pointing to a .yaml file
    - Registered name pointing to a directory
    - Bare directory path on disk
    """
    # Direct YAML file on disk
    p = Path(ref)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve(), p.resolve().parent

    # Registry lookup
    reg = load_registry()
    if ref in reg:
        stored = Path(reg[ref])
        if stored.suffix in (".yaml", ".yml"):
            return stored, stored.parent
        config = find_config_in_dir(stored)
        return config, stored

    # Directory on disk (or relative path)
    resolved = _resolve_dir(ref)
    config = find_config_in_dir(resolved)
    return config, resolved


def resolve_ref(ref: str) -> tuple[Path, Optional[str]]:
    """
    Parse a <ref> string into (experiment_dir, identifier_or_None).

    The identifier is a sweep tag or run_id. The caller is responsible for
    interpreting it in context (e.g. replay vs inspect).

    Handles Windows drive letters (e.g. C:\\path:id) by ignoring the
    single-letter drive colon and splitting on the next colon only.
    """
    parts = ref.split(":")
    # Windows drive letter: first segment is a single alpha char (e.g. "C")
    if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
        if len(parts) == 2:
            # C:\path — no identifier
            return Path(ref), None
        # C:\path:identifier — identifier is the last segment
        path_str = ":".join(parts[:-1])
        identifier: Optional[str] = parts[-1] or None
        return Path(path_str), identifier

    if ":" in ref:
        head, _, tail = ref.partition(":")
        id_val: Optional[str] = tail or None
        experiment_dir = _resolve_dir(head)
        return experiment_dir, id_val

    return _resolve_dir(ref), None


def _resolve_dir(s: str) -> Path:
    """Resolve a string to an experiment directory path."""
    p = Path(s)
    # Treat as a filesystem path if it looks like one
    if p.exists() or os.sep in s or "/" in s or s.startswith("."):
        # If it's a yaml file, return its parent directory
        if p.suffix in (".yaml", ".yml") and p.exists():
            return p.parent
        return p
    # Otherwise look up in registry
    try:
        return resolve_experiment_dir(s)
    except KeyError:
        # Not registered — return as Path anyway; caller will error if missing
        return p


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def find_config_in_dir(directory: Path) -> Path:
    """
    Find the experiment config YAML inside an experiment directory.

    Lookup order (per spec):
    1. config.yaml
    2. experiment.yaml
    3. The single *.yaml file present — error if zero or more than one
    """
    for name in ("config.yaml", "experiment.yaml"):
        p = directory / name
        if p.exists():
            return p
    yamls = sorted(directory.glob("*.yaml"))
    if len(yamls) == 1:
        return yamls[0]
    if len(yamls) > 1:
        names = ", ".join(y.name for y in yamls)
        raise ValueError(
            f"Multiple YAML files in {directory}: {names}. "
            "Rename one to config.yaml or use anysearch run <config-file> directly."
        )
    raise FileNotFoundError(
        f"No config YAML found in {directory}. "
        "Expected config.yaml, experiment.yaml, or a single *.yaml file."
    )
