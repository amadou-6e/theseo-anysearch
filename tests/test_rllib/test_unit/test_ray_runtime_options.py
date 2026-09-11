from theseo_anysearch.experiments.models import TuneConfig
from theseo_anysearch.experiments.tune_runner import (
    _ray_runtime_options,
    _should_shutdown_ray,
)


def test_external_runtime_uses_address_without_local_temp_dir():
    options = _ray_runtime_options("auto", "ignored")

    assert options["address"] == "auto"
    assert "_temp_dir" not in options


def test_local_runtime_uses_isolated_temp_dir():
    options = _ray_runtime_options(None, "ray-temp")

    assert options["_temp_dir"] == "ray-temp"
    assert "address" not in options


def test_shutdown_defaults_to_owned_local_runtime_only():
    assert _should_shutdown_ray(None, was_initialized=False, address=None)
    assert not _should_shutdown_ray(None, was_initialized=True, address=None)
    assert not _should_shutdown_ray(None, was_initialized=False, address="auto")


def test_explicit_shutdown_setting_overrides_ownership():
    assert _should_shutdown_ray(True, was_initialized=True, address="auto")
    assert not _should_shutdown_ray(False, was_initialized=False, address=None)


def test_tune_config_accepts_persistent_ray_runtime_settings():
    config = TuneConfig(ray_address="auto", shutdown_ray_on_complete=False)

    assert config.ray_address == "auto"
    assert config.shutdown_ray_on_complete is False