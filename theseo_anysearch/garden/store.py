"""GardenStore: save and load encoder checkpoints."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from theseo_anysearch.garden.data_config import GardenConfig
from theseo_anysearch.garden.encoder import extract_encoder


# ---------------------------------------------------------------------------
# Garden config registry  (~/.anysearch/garden_configs.yaml)
# Maps short name → absolute path to garden YAML.
# ---------------------------------------------------------------------------

def _config_registry_file() -> Path:
    override = os.environ.get("ANYSEARCH_GARDEN_CONFIGS")
    if override:
        return Path(override)
    return Path.home() / ".anysearch" / "garden_configs.yaml"


def register_config(name: str, path: Path) -> None:
    f = _config_registry_file()
    reg: dict[str, str] = yaml.safe_load(f.read_text()) or {} if f.exists() else {}
    reg[name] = str(path.resolve())
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.dump(reg, default_flow_style=False, sort_keys=True))


def resolve_config(name: str) -> Path | None:
    """Return the registered YAML path for *name*, or None if not found."""
    f = _config_registry_file()
    if not f.exists():
        return None
    reg: dict[str, str] = yaml.safe_load(f.read_text()) or {}
    if name in reg:
        p = Path(reg[name])
        return p if p.exists() else None
    return None


def list_configs() -> dict[str, str]:
    f = _config_registry_file()
    return yaml.safe_load(f.read_text()) or {} if f.exists() else {}


# ---------------------------------------------------------------------------
# Global garden registry  (~/.anysearch/garden_registry.yaml)
# ---------------------------------------------------------------------------

def _garden_registry_file() -> Path:
    override = os.environ.get("ANYSEARCH_GARDEN_REGISTRY")
    if override:
        return Path(override)
    return Path.home() / ".anysearch" / "garden_registry.yaml"


def _load_garden_registry() -> dict[str, str]:
    f = _garden_registry_file()
    if not f.exists():
        return {}
    return yaml.safe_load(f.read_text()) or {}


def _save_garden_registry(reg: dict[str, str]) -> None:
    f = _garden_registry_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.dump(reg, default_flow_style=False, sort_keys=True))


# ---------------------------------------------------------------------------
# Built-in pretrained encoders  (bundled with the package)
# ---------------------------------------------------------------------------

def _builtin_presets_dir() -> Path:
    return Path(__file__).parent / "pretrained"


def register_builtin_presets() -> None:
    """Register package-bundled encoder checkpoints into the global registry.

    Safe to call multiple times — skips names already registered to a path
    that still exists.
    """
    presets_dir = _builtin_presets_dir()
    if not presets_dir.exists():
        return

    reg = _load_garden_registry()
    changed = False

    for model_dir in sorted(presets_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for tag_dir in sorted(model_dir.iterdir()):
            if not tag_dir.is_dir():
                continue
            if not (tag_dir / "encoder.pt").exists():
                continue
            name = model_dir.name
            tag = tag_dir.name
            ref = f"{name}:{tag}"
            # Only register if not already pointing somewhere valid
            if ref not in reg or not Path(reg[ref]).exists():
                reg[ref] = str(tag_dir.resolve())
                changed = True
            # Also register untagged family name to the latest tag dir
            if name not in reg or not Path(reg[name]).exists():
                reg[name] = str(tag_dir.resolve())
                changed = True

    if changed:
        _save_garden_registry(reg)


class GardenStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _model_dir(self, name: str, tag: str | None = None) -> Path:
        if tag:
            return self.root / name / tag
        return self.root / name

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        name: str,
        model: nn.Module,
        cfg: GardenConfig,
        epochs_trained: int,
        loss_curve: list[dict],
        final_val_loss: float,
        tag: str | None = None,
    ) -> Path:
        d = self._model_dir(name, tag)
        d.mkdir(parents=True, exist_ok=True)

        # Full model (for reconstruction / visualisation)
        torch.save(model.state_dict(), d / "model.pt")

        # Encoder weights only (for RL use)
        encoder = extract_encoder(model)
        torch.save(encoder.state_dict(), d / "encoder.pt")

        # Metadata
        meta = {
            "name": name,
            "tag": tag,
            "approach": cfg.approach,
            "architecture": cfg.architecture,
            "latent_dim": cfg.encoder.latent_dim,
            "box_radius": cfg.encoder.box_radius,
            "box_radii": cfg.encoder.box_radii,
            "conv_channels": cfg.encoder.conv_channels,
            "epochs_trained": epochs_trained,
            "final_val_loss": final_val_loss,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "augmentation": cfg.augmentation.model_dump(exclude_none=True),
            "data_sources": cfg.data.model_dump(mode="json") if not isinstance(cfg.data, Path) else {},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        (d / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))

        # Register in global registry
        reg = _load_garden_registry()
        reg[name] = str(self.root / name)           # name → family root (latest untagged)
        if tag:
            reg[f"{name}:{tag}"] = str(d)           # name:tag → specific versioned dir
        _save_garden_registry(reg)

        return d / "encoder.pt"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_encoder_state(self, name: str, device: str = "cpu") -> dict:
        ckpt = self._model_dir(name) / "encoder.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No encoder for {name!r} at {ckpt}")
        return torch.load(ckpt, map_location=device, weights_only=True)

    def meta(self, name: str) -> dict:
        path = self._model_dir(name) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"No meta.json for {name!r}")
        return json.loads(path.read_text())

    def loss_curve(self, name: str) -> list[dict]:
        path = self._model_dir(name) / "loss_curve.json"
        if not path.exists():
            return []
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # List / inspect
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        rows = []
        if not self.root.exists():
            return rows
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            meta_file = d / "meta.json"
            if meta_file.exists():
                rows.append(json.loads(meta_file.read_text()))
            else:
                # Tagged versions live as subdirs (name/tag/meta.json)
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and (sub / "meta.json").exists():
                        rows.append(json.loads((sub / "meta.json").read_text()))
        return rows

    def exists(self, name: str, tag: str | None = None) -> bool:
        return (self._model_dir(name, tag) / "encoder.pt").exists()

    # ------------------------------------------------------------------
    # Global registry resolution
    # ------------------------------------------------------------------

    @classmethod
    def resolve_ref(cls, ref: str) -> tuple[Path, Optional[str]]:
        """Parse name[:tag] → (model_dir, tag_or_None). Consults the global registry."""
        # Windows drive-letter colon: single alpha char before first colon
        parts = ref.split(":")
        if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
            return Path(ref), None

        if ":" in ref:
            head, _, tail = ref.partition(":")
            tag: Optional[str] = tail or None
        else:
            head, tag = ref, None

        reg = _load_garden_registry()

        # Exact key match (covers both "name" and "name:tag" registry entries)
        if ref in reg:
            return Path(reg[ref]), tag

        # Family name match + optional tag lookup
        if head in reg:
            family_dir = Path(reg[head])
            if tag:
                tag_dir = family_dir / tag
                if (tag_dir / "meta.json").exists():
                    return tag_dir, tag
                raise KeyError(f"Tag '{tag}' not found for model '{head}'")
            return family_dir, None

        # Fallback: treat as a filesystem path
        p = Path(ref)
        if p.exists():
            return p, tag

        raise KeyError(
            f"Garden model '{head}' not in registry.\n"
            f"Run:  anysearch garden train <config.yaml>"
        )

    # ------------------------------------------------------------------
    # Training-run registry
    # ------------------------------------------------------------------

    def _runs_file(self) -> Path:
        return self.root / "cache" / "runs.json"

    def register_run(self, run_ref: str, run_dir: Path, obs_count: int, cache_hash: str) -> None:
        f = self._runs_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        registry = json.loads(f.read_text()) if f.exists() else {}
        registry[run_ref] = {
            "run_dir": str(run_dir),
            "obs_count": obs_count,
            "cache_hash": cache_hash,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        f.write_text(json.dumps(registry, indent=2))

    def list_runs(self) -> dict:
        f = self._runs_file()
        return json.loads(f.read_text()) if f.exists() else {}
