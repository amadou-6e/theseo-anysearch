"""Unit tests for GardenStore and config registry helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from theseo_anysearch.garden.data_config import (
    AugmentationConfig,
    DataConfig,
    EncoderConfig,
    GardenConfig,
    GardenTrainingConfig,
    GeometryConfig,
    PositionSampleSourceConfig,
)
from theseo_anysearch.garden.store import (
    GardenStore,
    list_configs,
    register_config,
    resolve_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_garden_registry(tmp_path, monkeypatch):
    """Redirect both garden registry files to tmp_path so tests are isolated."""
    configs_file = tmp_path / "garden_configs.yaml"
    registry_file = tmp_path / "garden_registry.yaml"
    monkeypatch.setenv("ANYSEARCH_GARDEN_CONFIGS", str(configs_file))
    monkeypatch.setenv("ANYSEARCH_GARDEN_REGISTRY", str(registry_file))


def _minimal_garden_config(name: str = "test_model") -> GardenConfig:
    return GardenConfig(
        name=name,
        data=DataConfig(
            sources=[PositionSampleSourceConfig(type="position_sample", geometry=GeometryConfig())]
        ),
        encoder=EncoderConfig(box_radius=2, latent_dim=16, conv_channels=[8, 16]),
        training=GardenTrainingConfig(epochs=1),
    )


def _tiny_model() -> nn.Module:
    """Minimal nn.Module with an encoder sub-module (matching GardenStore.save expectations)."""
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

    return FullModel()


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

class TestConfigRegistry:
    """Tests ConfigRegistry."""
    def test_register_and_resolve(self, tmp_path):
        yaml_file = tmp_path / "my_model.yaml"
        yaml_file.touch()
        register_config("my_model", yaml_file)
        resolved = resolve_config("my_model")
        assert resolved == yaml_file

    def test_resolve_missing_name_returns_none(self):
        assert resolve_config("does_not_exist") is None

    def test_resolve_missing_file_returns_none(self, tmp_path):
        yaml_file = tmp_path / "gone.yaml"
        register_config("gone", yaml_file)
        # File never created — resolve should return None
        assert resolve_config("gone") is None

    def test_list_configs_empty(self):
        assert list_configs() == {}

    def test_list_configs_after_register(self, tmp_path):
        f = tmp_path / "a.yaml"
        f.touch()
        register_config("model_a", f)
        configs = list_configs()
        assert "model_a" in configs

    def test_overwrite_existing_name(self, tmp_path):
        f1 = tmp_path / "v1.yaml"
        f2 = tmp_path / "v2.yaml"
        f1.touch()
        f2.touch()
        register_config("m", f1)
        register_config("m", f2)
        resolved = resolve_config("m")
        assert resolved == f2


# ---------------------------------------------------------------------------
# GardenStore.save
# ---------------------------------------------------------------------------

class TestGardenStoreSave:
    """Tests GardenStoreSave."""
    def test_save_creates_files(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config()
        store.save("test_model", model, cfg, epochs_trained=5, loss_curve=[], final_val_loss=0.05)
        model_dir = tmp_path / "garden" / "test_model"
        assert (model_dir / "model.pt").exists()
        assert (model_dir / "encoder.pt").exists()
        assert (model_dir / "meta.json").exists()
        assert (model_dir / "loss_curve.json").exists()

    def test_save_with_tag(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config()
        store.save("test_model", model, cfg, epochs_trained=3, loss_curve=[], final_val_loss=0.1, tag="v1")
        assert (tmp_path / "garden" / "test_model" / "v1" / "encoder.pt").exists()

    def test_meta_contents(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config("my_enc")
        store.save("my_enc", model, cfg, epochs_trained=10, loss_curve=[], final_val_loss=0.042)
        meta = json.loads((tmp_path / "garden" / "my_enc" / "meta.json").read_text())
        assert meta["name"] == "my_enc"
        assert meta["epochs_trained"] == 10
        assert meta["final_val_loss"] == pytest.approx(0.042)
        assert meta["latent_dim"] == 16

    def test_loss_curve_saved(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config()
        curve = [{"epoch": 1, "val_loss": 0.5}, {"epoch": 2, "val_loss": 0.3}]
        store.save("m", model, cfg, epochs_trained=2, loss_curve=curve, final_val_loss=0.3)
        saved = json.loads((tmp_path / "garden" / "m" / "loss_curve.json").read_text())
        assert len(saved) == 2
        assert saved[0]["val_loss"] == pytest.approx(0.5)

    def test_returns_encoder_pt_path(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config()
        result = store.save("m", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0)
        assert result.name == "encoder.pt"
        assert result.exists()


# ---------------------------------------------------------------------------
# GardenStore.exists
# ---------------------------------------------------------------------------

class TestGardenStoreExists:
    """Tests GardenStoreExists."""
    def test_exists_false_before_save(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        assert not store.exists("missing")

    def test_exists_true_after_save(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config()
        store.save("m", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0)
        assert store.exists("m")

    def test_exists_with_tag(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config()
        store.save("m", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0, tag="v2")
        assert store.exists("m", tag="v2")
        assert not store.exists("m", tag="v99")


# ---------------------------------------------------------------------------
# GardenStore.meta / loss_curve
# ---------------------------------------------------------------------------

class TestGardenStoreMeta:
    """Tests GardenStoreMeta."""
    def test_meta_roundtrip(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config("enc")
        store.save("enc", model, cfg, epochs_trained=7, loss_curve=[], final_val_loss=0.01)
        meta = store.meta("enc")
        assert meta["name"] == "enc"

    def test_meta_missing_raises(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        with pytest.raises(FileNotFoundError):
            store.meta("nonexistent")

    def test_loss_curve_empty_default(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        # No save → loss_curve.json doesn't exist
        assert store.loss_curve("nonexistent") == []


# ---------------------------------------------------------------------------
# GardenStore.list_models
# ---------------------------------------------------------------------------

class TestGardenStoreListModels:
    """Tests GardenStoreListModels."""
    def test_empty_root(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        assert store.list_models() == []

    def test_lists_saved_model(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config("listed")
        store.save("listed", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0)
        rows = store.list_models()
        assert any(r["name"] == "listed" for r in rows)

    def test_lists_tagged_models(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config("enc")
        store.save("enc", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0, tag="v1")
        store.save("enc", model, cfg, epochs_trained=2, loss_curve=[], final_val_loss=0.0, tag="v2")
        rows = store.list_models()
        tags = {r.get("tag") for r in rows}
        assert "v1" in tags
        assert "v2" in tags


# ---------------------------------------------------------------------------
# GardenStore.resolve_ref
# ---------------------------------------------------------------------------

class TestGardenStoreResolveRef:
    """Tests GardenStoreResolveRef."""
    def test_resolve_registered_name(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config("enc")
        store.save("enc", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0)
        path, tag = GardenStore.resolve_ref("enc")
        assert tag is None
        assert path.exists()

    def test_resolve_name_with_tag(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config("enc")
        store.save("enc", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0, tag="v1")
        path, tag = GardenStore.resolve_ref("enc:v1")
        assert tag == "v1"
        assert (path / "encoder.pt").exists()

    def test_resolve_unknown_raises(self):
        with pytest.raises(KeyError, match="not in registry"):
            GardenStore.resolve_ref("does_not_exist_xyz")

    def test_resolve_known_name_unknown_tag_raises(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        model = _tiny_model()
        cfg = _minimal_garden_config("enc")
        store.save("enc", model, cfg, epochs_trained=1, loss_curve=[], final_val_loss=0.0, tag="v1")
        with pytest.raises(KeyError, match="Tag"):
            GardenStore.resolve_ref("enc:v99")


# ---------------------------------------------------------------------------
# Training-run registry
# ---------------------------------------------------------------------------

class TestGardenStoreRunRegistry:
    """Tests GardenStoreRunRegistry."""
    def test_register_and_list(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        store.register_run("ppo:v1", tmp_path / "run1", obs_count=1000, cache_hash="abc123")
        runs = store.list_runs()
        assert "ppo:v1" in runs
        assert runs["ppo:v1"]["obs_count"] == 1000

    def test_list_runs_empty(self, tmp_path):
        store = GardenStore(tmp_path / "garden")
        assert store.list_runs() == {}
