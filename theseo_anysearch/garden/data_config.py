"""Pydantic models for garden YAML config."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Geometry sources
# ---------------------------------------------------------------------------

class EnvGeometryConfig(BaseModel):
    """Fixed single geometry — mirrors EnvConfig geometry fields."""
    model_config = ConfigDict(extra="forbid")

    stl_path: Path | None = None
    geometry_boxes: list[list[int]] | None = None
    scale: float = 40.0
    grid_size: int = 32


class StlListSamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["stl_list"]
    dir: Path | None = None
    paths: list[Path] | None = None
    scale: float = 40.0
    total_samples: int = 60000
    seed: int | None = 42

    def resolve_paths(self) -> list[Path]:
        """Return all STL files to use, in order."""
        if self.paths:
            return [Path(p) for p in self.paths]
        if self.dir:
            d = Path(self.dir)
            index = d / "index.yaml"
            if index.exists():
                raw = yaml.safe_load(index.read_text()) or {}
                entries = raw.get("geometries", [])
                return [
                    d / e["path"]
                    for e in entries
                    if not e.get("exclude", False)
                ]
            return sorted(p for p in d.rglob("*.stl") if not p.stem.startswith("_"))
        return []


class RandomBoxesSamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["random_boxes"]
    seed: int = 42
    num_geometries: int = 50
    num_boxes: list[int] = Field(default=[1, 5])
    box_min_size: list[int] = Field(default=[3, 3, 3])
    box_max_size: list[int] = Field(default=[12, 12, 12])
    grid_size: int = 32
    total_samples: int = 50000


SamplerConfig = Annotated[
    Union[StlListSamplerConfig, RandomBoxesSamplerConfig],
    Field(discriminator="type"),
]


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: EnvGeometryConfig | None = None
    sampler: SamplerConfig | None = None


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

class PositionSampleSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["position_sample"]
    geometry: GeometryConfig
    num_samples: int | None = None  # only used when geometry.env is set


class TrainingRunSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["training_run"]
    run: str           # registered name e.g. "multi_agent_ppo_asha:v3"
    obs_key: str = "local_grid"


SourceConfig = Annotated[
    Union[PositionSampleSourceConfig, TrainingRunSourceConfig],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Cache / split
# ---------------------------------------------------------------------------

class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path = Path("runtime/garden/cache")
    force_refresh: bool = False


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = 0.1
    seed: int = 42


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceConfig]
    cache: CacheConfig = Field(default_factory=CacheConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    min_fill_pct: float = 0.0        # a sample must have at least this fill% to count as non-empty
    empty_sample_ratio: float = 0.05  # fraction of the final dataset that should be empty samples


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class CutoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = 2
    min_size: int = 1
    max_size: int = 2


class GeometryPasteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability: float = 0.3
    paste_size: int = 2


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rotate90: bool = False
    flip: bool = False
    cutout: CutoutConfig | None = None
    geometry_paste: GeometryPasteConfig | None = None
    noise_prob: float = 0.0
    translation_jitter: int = 0
    morph_prob: float = 0.0


# ---------------------------------------------------------------------------
# Encoder architecture
# ---------------------------------------------------------------------------

class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box_radius: int = 2
    box_radii: list[int] | None = None  # if set, trains on all resolutions simultaneously
    conv_channels: list[int] = Field(default=[32, 64, 64])
    latent_dim: int = 128


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: Literal["ae", "vae"] = "ae"
    beta: float = 1.0


class MVMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mask_ratio: float = 0.75
    decoder_layers: int = 2
    decoder_dim: int = 64
    decoder_heads: int = 4


class CustomEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(alias="class")
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def instantiate(self):
        module_path, class_name = self.class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        return cls(**self.kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class GardenTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    lr_scheduler: Literal["cosine", "constant"] = "cosine"
    early_stop_patience: int = 10
    pos_weight_factor: float = 0.0  # 0 = auto from batch fill rate; >0 = fixed weight for filled voxels
    focal_gamma: float = 0.0        # 0 = plain (weighted) BCE; >0 = focal loss gamma


# ---------------------------------------------------------------------------
# Tune config  (optional; enables anysearch garden tune)
# ---------------------------------------------------------------------------

class GardenTuneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_samples: int = 20
    max_concurrent: int = 4
    grace_period: int = 5         # ASHA: min epochs before pruning
    reduction_factor: int = 3     # ASHA reduction factor
    # search_space keys mirror GardenConfig sections: encoder.*, training.*
    search_space: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level garden config
# ---------------------------------------------------------------------------

class GardenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    architecture: Literal[
        "voxel_box_3dcnn", "voxel_box_2dcnn", "voxel_triplanar_2dcnn", "custom"
    ] = "voxel_box_3dcnn"
    approach: Literal["autoencoder", "vae", "mvmencoder"] = "autoencoder"

    # data can be an inline DataConfig dict or a path to a standalone data YAML
    data: DataConfig | Path

    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    mvmencoder: MVMConfig = Field(default_factory=MVMConfig)
    custom_encoder: CustomEncoderConfig | None = None
    training: GardenTrainingConfig = Field(default_factory=GardenTrainingConfig)
    tune: GardenTuneConfig | None = None


def load_garden_config(path: Path) -> GardenConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    top = raw.get("model_garden", raw)  # tolerate both with and without wrapper key
    # Resolve data: path reference relative to the config file
    if isinstance(top.get("data"), str):
        data_path = (path.parent / top["data"]).resolve()
        data_raw = yaml.safe_load(data_path.read_text()) or {}
        top["data"] = data_raw.get("data", data_raw)
    return GardenConfig.model_validate(top)


def load_data_config(path: Path) -> DataConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    return DataConfig.model_validate(raw.get("data", raw))
