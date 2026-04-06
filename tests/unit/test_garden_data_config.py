"""Unit tests for garden data_config Pydantic models and loaders."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from theseo_anysearch.garden.data_config import (
    AugmentationConfig,
    AutoencoderConfig,
    CacheConfig,
    CutoutConfig,
    DataConfig,
    EncoderConfig,
    EnvGeometryConfig,
    GardenConfig,
    GardenTrainingConfig,
    GardenTuneConfig,
    GeometryConfig,
    GeometryPasteConfig,
    MVMConfig,
    PositionSampleSourceConfig,
    RandomBoxesSamplerConfig,
    SplitConfig,
    StlListSamplerConfig,
    TrainingRunSourceConfig,
    load_garden_config,
)


# ---------------------------------------------------------------------------
# Geometry source models
# ---------------------------------------------------------------------------

class TestEnvGeometryConfig:
    def test_defaults(self):
        g = EnvGeometryConfig()
        assert g.stl_path is None
        assert g.geometry_boxes is None
        assert g.scale == 40.0
        assert g.grid_size == 32

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            EnvGeometryConfig(unknown_field=1)

    def test_geometry_boxes(self):
        g = EnvGeometryConfig(geometry_boxes=[[0, 0, 0, 10, 10, 10]])
        assert g.geometry_boxes == [[0, 0, 0, 10, 10, 10]]


class TestStlListSamplerConfig:
    def test_type_literal(self):
        s = StlListSamplerConfig(type="stl_list")
        assert s.type == "stl_list"

    def test_defaults(self):
        s = StlListSamplerConfig(type="stl_list")
        assert s.scale == 40.0
        assert s.total_samples == 60000
        assert s.seed == 42

    def test_resolve_paths_from_paths(self, tmp_path):
        f1 = tmp_path / "a.stl"
        f2 = tmp_path / "b.stl"
        f1.touch()
        f2.touch()
        s = StlListSamplerConfig(type="stl_list", paths=[f1, f2])
        resolved = s.resolve_paths()
        assert Path(f1) in [Path(p) for p in resolved]
        assert Path(f2) in [Path(p) for p in resolved]

    def test_resolve_paths_from_dir_glob(self, tmp_path):
        (tmp_path / "a.stl").touch()
        (tmp_path / "b.stl").touch()
        (tmp_path / "_hidden.stl").touch()
        s = StlListSamplerConfig(type="stl_list", dir=tmp_path)
        paths = s.resolve_paths()
        names = {p.name for p in paths}
        assert "a.stl" in names
        assert "b.stl" in names
        assert "_hidden.stl" not in names  # excluded by leading _

    def test_resolve_paths_from_dir_index(self, tmp_path):
        (tmp_path / "a.stl").touch()
        (tmp_path / "b.stl").touch()
        index = {"geometries": [{"path": "a.stl"}, {"path": "b.stl", "exclude": True}]}
        (tmp_path / "index.yaml").write_text(yaml.dump(index))
        s = StlListSamplerConfig(type="stl_list", dir=tmp_path)
        paths = s.resolve_paths()
        names = {p.name for p in paths}
        assert "a.stl" in names
        assert "b.stl" not in names  # excluded via index

    def test_resolve_paths_empty(self):
        s = StlListSamplerConfig(type="stl_list")
        assert s.resolve_paths() == []


class TestRandomBoxesSamplerConfig:
    def test_type_literal(self):
        r = RandomBoxesSamplerConfig(type="random_boxes")
        assert r.type == "random_boxes"

    def test_defaults(self):
        r = RandomBoxesSamplerConfig(type="random_boxes")
        assert r.num_geometries == 50
        assert r.num_boxes == [1, 5]
        assert r.grid_size == 32
        assert r.total_samples == 50000

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            RandomBoxesSamplerConfig(type="random_boxes", bad=1)


# ---------------------------------------------------------------------------
# Data source models
# ---------------------------------------------------------------------------

class TestPositionSampleSourceConfig:
    def test_type_literal(self):
        src = PositionSampleSourceConfig(
            type="position_sample",
            geometry=GeometryConfig(),
        )
        assert src.type == "position_sample"

    def test_num_samples_optional(self):
        src = PositionSampleSourceConfig(
            type="position_sample",
            geometry=GeometryConfig(),
        )
        assert src.num_samples is None


class TestTrainingRunSourceConfig:
    def test_basics(self):
        src = TrainingRunSourceConfig(type="training_run", run="multi_agent_ppo:v3")
        assert src.run == "multi_agent_ppo:v3"
        assert src.obs_key == "local_grid"


# ---------------------------------------------------------------------------
# Cache / split
# ---------------------------------------------------------------------------

class TestCacheConfig:
    def test_default_path(self):
        c = CacheConfig()
        assert c.path == Path("runtime/garden/cache")
        assert c.force_refresh is False


class TestSplitConfig:
    def test_defaults(self):
        s = SplitConfig()
        assert s.val_fraction == 0.1
        assert s.seed == 42

    def test_custom(self):
        s = SplitConfig(val_fraction=0.2, seed=7)
        assert s.val_fraction == 0.2


class TestDataConfig:
    def _make(self, **kw):
        return DataConfig(
            sources=[PositionSampleSourceConfig(
                type="position_sample",
                geometry=GeometryConfig(),
            )],
            **kw,
        )

    def test_defaults(self):
        d = self._make()
        assert d.min_fill_pct == 0.0
        assert d.empty_sample_ratio == 0.05

    def test_extra_forbidden(self):
        with pytest.raises(Exception):
            self._make(bad_key=1)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class TestCutoutConfig:
    def test_defaults(self):
        c = CutoutConfig()
        assert c.count == 2
        assert c.min_size == 1
        assert c.max_size == 2


class TestGeometryPasteConfig:
    def test_defaults(self):
        g = GeometryPasteConfig()
        assert g.probability == 0.3
        assert g.paste_size == 2


class TestAugmentationConfig:
    def test_all_off_by_default(self):
        a = AugmentationConfig()
        assert a.rotate90 is False
        assert a.flip is False
        assert a.cutout is None
        assert a.geometry_paste is None
        assert a.noise_prob == 0.0
        assert a.translation_jitter == 0
        assert a.morph_prob == 0.0

    def test_enable_all(self):
        a = AugmentationConfig(
            rotate90=True,
            flip=True,
            cutout=CutoutConfig(count=3, min_size=1, max_size=3),
            noise_prob=0.1,
            translation_jitter=2,
            morph_prob=0.5,
        )
        assert a.rotate90 is True
        assert a.cutout.count == 3

    def test_extra_forbidden(self):
        with pytest.raises(Exception):
            AugmentationConfig(unknown=True)


# ---------------------------------------------------------------------------
# Encoder / training
# ---------------------------------------------------------------------------

class TestEncoderConfig:
    def test_defaults(self):
        e = EncoderConfig()
        assert e.box_radius == 2
        assert e.box_radii is None
        assert e.conv_channels == [32, 64, 64]
        assert e.latent_dim == 128

    def test_multi_radius(self):
        e = EncoderConfig(box_radii=[1, 2, 3])
        assert e.box_radii == [1, 2, 3]


class TestGardenTrainingConfig:
    def test_defaults(self):
        t = GardenTrainingConfig()
        assert t.epochs == 100
        assert t.batch_size == 256
        assert t.lr == pytest.approx(3e-4)
        assert t.lr_scheduler == "cosine"
        assert t.early_stop_patience == 10
        assert t.pos_weight_factor == 0.0
        assert t.focal_gamma == 0.0

    def test_focal_loss_config(self):
        t = GardenTrainingConfig(focal_gamma=2.0, pos_weight_factor=5.0)
        assert t.focal_gamma == 2.0
        assert t.pos_weight_factor == 5.0


class TestAutoencoderConfig:
    def test_defaults(self):
        a = AutoencoderConfig()
        assert a.variant == "ae"
        assert a.beta == 1.0


class TestMVMConfig:
    def test_defaults(self):
        m = MVMConfig()
        assert m.mask_ratio == 0.75
        assert m.decoder_layers == 2


class TestGardenTuneConfig:
    def test_defaults(self):
        t = GardenTuneConfig()
        assert t.num_samples == 20
        assert t.max_concurrent == 4
        assert t.grace_period == 5
        assert t.search_space == {}


# ---------------------------------------------------------------------------
# GardenConfig + loaders
# ---------------------------------------------------------------------------

class TestGardenConfig:
    def _minimal_data(self):
        return {
            "sources": [
                {"type": "position_sample", "geometry": {}}
            ]
        }

    def test_minimal_valid(self):
        cfg = GardenConfig(
            name="test_model",
            data=self._minimal_data(),
        )
        assert cfg.name == "test_model"
        assert cfg.architecture == "voxel_box_3dcnn"
        assert cfg.approach == "autoencoder"

    def test_invalid_architecture(self):
        with pytest.raises(Exception):
            GardenConfig(name="x", data=self._minimal_data(), architecture="bad_arch")

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            GardenConfig(name="x", data=self._minimal_data(), not_a_field=1)

    def test_tune_optional(self):
        cfg = GardenConfig(name="x", data=self._minimal_data())
        assert cfg.tune is None

        cfg2 = GardenConfig(
            name="x",
            data=self._minimal_data(),
            tune={"num_samples": 5},
        )
        assert cfg2.tune.num_samples == 5


class TestLoadGardenConfig:
    def test_roundtrip_minimal(self, tmp_path):
        raw = {
            "name": "my_model",
            "approach": "autoencoder",
            "architecture": "voxel_box_3dcnn",
            "data": {
                "sources": [
                    {"type": "position_sample", "geometry": {}}
                ]
            },
            "encoder": {"box_radius": 2, "latent_dim": 64, "conv_channels": [32, 64]},
            "training": {"epochs": 10},
        }
        f = tmp_path / "garden.yaml"
        f.write_text(yaml.dump(raw))
        cfg = load_garden_config(f)
        assert cfg.name == "my_model"
        assert cfg.encoder.latent_dim == 64
        assert cfg.training.epochs == 10

    def test_model_garden_wrapper_key(self, tmp_path):
        raw = {
            "model_garden": {
                "name": "wrapped",
                "data": {
                    "sources": [
                        {"type": "position_sample", "geometry": {}}
                    ]
                },
            }
        }
        f = tmp_path / "garden.yaml"
        f.write_text(yaml.dump(raw))
        cfg = load_garden_config(f)
        assert cfg.name == "wrapped"

    def test_data_as_path_reference(self, tmp_path):
        data_raw = {
            "data": {
                "sources": [{"type": "position_sample", "geometry": {}}]
            }
        }
        data_file = tmp_path / "data.yaml"
        data_file.write_text(yaml.dump(data_raw))

        garden_raw = {
            "name": "ref_model",
            "data": "data.yaml",
        }
        garden_file = tmp_path / "garden.yaml"
        garden_file.write_text(yaml.dump(garden_raw))

        cfg = load_garden_config(garden_file)
        assert cfg.name == "ref_model"
        assert len(cfg.data.sources) == 1
