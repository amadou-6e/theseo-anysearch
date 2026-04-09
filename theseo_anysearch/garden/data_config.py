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
    """Configuration for sampling training geometry from a list of STL files.

    Parameters
    ----------
    type : Literal["stl_list"]
        Discriminator used by the sampler union.
    dir : Path | None
        Directory containing STL files or an ``index.yaml`` manifest.
    paths : list[Path] | None
        Explicit STL paths to use instead of scanning a directory.
    scale : float
        Fixed voxelization scale.
    total_samples : int
        Total number of observations to collect across all STL files.
    seed : int | None
        Optional random seed used when sampling.
    """
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
    """Configuration for procedural random-box geometry sampling.

    Parameters
    ----------
    type : Literal["random_boxes"]
        Discriminator used by the sampler union.
    seed : int
        Random seed for procedural generation.
    num_geometries : int
        Number of distinct procedural scenes to generate.
    num_boxes : list[int]
        Inclusive min and max number of boxes per scene.
    box_min_size : list[int]
        Inclusive minimum box side lengths.
    box_max_size : list[int]
        Inclusive maximum box side lengths.
    grid_size : int
        Side length of the generated voxel grid.
    total_samples : int
        Total number of observations to collect.
    """
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
    """Geometry source definition for garden data collection.

    Parameters
    ----------
    env : EnvGeometryConfig | None
        Fixed environment geometry definition.
    sampler : SamplerConfig | None
        Procedural or STL-based geometry sampler.
    """
    model_config = ConfigDict(extra="forbid")

    env: EnvGeometryConfig | None = None
    sampler: SamplerConfig | None = None


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

class PositionSampleSourceConfig(BaseModel):
    """Collect observations by sampling positions from geometry.

    Parameters
    ----------
    type : Literal["position_sample"]
        Discriminator used by the source union.
    geometry : GeometryConfig
        Geometry source or sampler used for collection.
    num_samples : int | None
        Number of observations to collect for fixed environment geometry.
    """
    model_config = ConfigDict(extra="forbid")

    type: Literal["position_sample"]
    geometry: GeometryConfig
    num_samples: int | None = None  # only used when geometry.env is set


class TrainingRunSourceConfig(BaseModel):
    """Collect observations from a previously recorded training run.

    Parameters
    ----------
    type : Literal["training_run"]
        Discriminator used by the source union.
    run : str
        Registered training run reference.
    obs_key : str
        Observation key extracted from saved trajectories.
    """
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
    """Cache configuration for garden dataset collection.

    Parameters
    ----------
    path : Path
        Root directory for cached garden artifacts.
    force_refresh : bool
        Whether existing cached observations should be ignored and rebuilt.
    """
    model_config = ConfigDict(extra="forbid")

    path: Path = Path("runtime/garden/cache")
    force_refresh: bool = False


class SplitConfig(BaseModel):
    """Train-validation split configuration for collected datasets.

    Parameters
    ----------
    val_fraction : float
        Fraction of observations assigned to validation.
    seed : int
        Seed used when shuffling the split.
    """
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = 0.1
    seed: int = 42


class DataConfig(BaseModel):
    """Top-level garden data configuration.

    Parameters
    ----------
    sources : list[SourceConfig]
        Data sources used to build the garden dataset.
    cache : CacheConfig
        Cache behavior for collected observations.
    split : SplitConfig
        Train-validation split definition.
    min_fill_pct : float
        Minimum occupied percentage for a sample to count as non-empty.
    empty_sample_ratio : float
        Fraction of the final dataset reserved for empty samples.
    """
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
    """Cutout augmentation configuration.

    Parameters
    ----------
    count : int
        Number of cutout regions to apply.
    min_size : int
        Minimum side length of each cutout region.
    max_size : int
        Maximum side length of each cutout region.
    """
    model_config = ConfigDict(extra="forbid")

    count: int = 2
    min_size: int = 1
    max_size: int = 2


class GeometryPasteConfig(BaseModel):
    """Geometry paste augmentation configuration.

    Parameters
    ----------
    probability : float
        Probability of applying geometry pasting.
    paste_size : int
        Size of pasted geometry fragments.
    """
    model_config = ConfigDict(extra="forbid")

    probability: float = 0.3
    paste_size: int = 2


class AugmentationConfig(BaseModel):
    """Online garden augmentation configuration.

    Parameters
    ----------
    rotate90 : bool
        Whether to apply 90-degree rotations.
    flip : bool
        Whether to apply axis flips.
    cutout : CutoutConfig | None
        Optional cutout augmentation settings.
    geometry_paste : GeometryPasteConfig | None
        Optional geometry paste settings.
    noise_prob : float
        Probability of random voxel noise.
    translation_jitter : int
        Maximum voxel translation jitter.
    morph_prob : float
        Probability of morphological augmentation.
    """
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
    """Encoder architecture configuration for garden models.

    Parameters
    ----------
    box_radius : int
        Radius of the local voxel observation cube.
    box_radii : list[int] | None
        Optional set of radii for multi-resolution training.
    conv_channels : list[int]
        Convolution channel widths.
    latent_dim : int
        Latent embedding size.
    """
    model_config = ConfigDict(extra="forbid")

    box_radius: int = 2
    box_radii: list[int] | None = None  # if set, trains on all resolutions simultaneously
    conv_channels: list[int] = Field(default=[32, 64, 64])
    latent_dim: int = 128


class AutoencoderConfig(BaseModel):
    """Autoencoder-specific configuration.

    Parameters
    ----------
    variant : {"ae", "vae"}
        Autoencoder family to train.
    beta : float
        KL weighting factor for VAE training.
    """
    model_config = ConfigDict(extra="forbid")

    variant: Literal["ae", "vae"] = "ae"
    beta: float = 1.0


class MVMConfig(BaseModel):
    """Masked voxel modeling configuration.

    Parameters
    ----------
    mask_ratio : float
        Fraction of voxels masked during training.
    decoder_layers : int
        Number of decoder layers.
    decoder_dim : int
        Decoder hidden size.
    decoder_heads : int
        Number of decoder attention heads.
    """
    model_config = ConfigDict(extra="forbid")

    mask_ratio: float = 0.75
    decoder_layers: int = 2
    decoder_dim: int = 64
    decoder_heads: int = 4


class CustomEncoderConfig(BaseModel):
    """Configuration for instantiating a custom encoder class.

    Parameters
    ----------
    class_path : str
        Fully qualified import path for the encoder class.
    kwargs : dict[str, Any]
        Keyword arguments passed to the custom encoder constructor.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(alias="class")
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def instantiate(self):
        """Import and instantiate the configured custom encoder class.

        Returns
        -------
        object
            Instantiated custom encoder object.
        """
        module_path, class_name = self.class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        return cls(**self.kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class GardenTrainingConfig(BaseModel):
    """Training loop configuration for garden pretraining.

    Parameters
    ----------
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    lr_scheduler : {"cosine", "constant"}
        Learning-rate scheduler type.
    early_stop_patience : int
        Validation patience before early stopping.
    pos_weight_factor : float
        Optional fixed positive-class weighting factor.
    focal_gamma : float
        Optional focal-loss gamma value.
    """
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
    """Hyperparameter tuning configuration for garden training.

    Parameters
    ----------
    num_samples : int
        Number of sampled trials.
    max_concurrent : int
        Maximum number of concurrent trials.
    grace_period : int
        Minimum epochs before ASHA pruning.
    reduction_factor : int
        ASHA reduction factor.
    search_space : dict[str, Any]
        Search space definition keyed by config section and field.
    """
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
    """Top-level garden configuration.

    Parameters
    ----------
    name : str
        Garden model family name.
    architecture : {"voxel_box_3dcnn", "voxel_box_2dcnn", "voxel_triplanar_2dcnn", "custom"}
        Encoder architecture family.
    approach : {"autoencoder", "vae", "mvmencoder"}
        Training objective family.
    data : DataConfig | Path
        Inline data config or path to a standalone data YAML.
    augmentation : AugmentationConfig
        Online augmentation configuration.
    encoder : EncoderConfig
        Encoder architecture parameters.
    autoencoder : AutoencoderConfig
        Autoencoder-specific parameters.
    mvmencoder : MVMConfig
        Masked voxel modeling parameters.
    custom_encoder : CustomEncoderConfig | None
        Optional custom encoder definition.
    training : GardenTrainingConfig
        Training loop parameters.
    tune : GardenTuneConfig | None
        Optional tuning configuration.
    """
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
    """Load and validate a garden configuration from YAML.

    Parameters
    ----------
    path : Path
        Path to the garden YAML file.

    Returns
    -------
    GardenConfig
        Validated garden configuration.
    """
    raw = yaml.safe_load(path.read_text()) or {}
    top = raw.get("model_garden", raw)  # tolerate both with and without wrapper key
    # Resolve data: path reference relative to the config file
    if isinstance(top.get("data"), str):
        data_path = (path.parent / top["data"]).resolve()
        data_raw = yaml.safe_load(data_path.read_text()) or {}
        top["data"] = data_raw.get("data", data_raw)
    return GardenConfig.model_validate(top)


def load_data_config(path: Path) -> DataConfig:
    """Load and validate a standalone garden data configuration from YAML.

    Parameters
    ----------
    path : Path
        Path to the data YAML file.

    Returns
    -------
    DataConfig
        Validated data configuration.
    """
    raw = yaml.safe_load(path.read_text()) or {}
    return DataConfig.model_validate(raw.get("data", raw))
