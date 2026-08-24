"""Tests for pre-Ray imitation provider validation."""

import pytest

from theseo_anysearch.imitation.models import ImitationConfig
from theseo_anysearch.imitation.preflight import (
    ImitationPreflightError,
    preflight_imitation_providers,
)


def test_disabled_imitation_skips_preflight():
    preflight_imitation_providers(ImitationConfig(enabled=False))


def test_built_in_generation_and_sampling_names_pass():
    preflight_imitation_providers(
        ImitationConfig(
            enabled=True,
            generation={"provider": "astar"},
            sampling={"provider": "uniform_episode"},
        )
    )


def test_unknown_generation_provider_fails_without_python_source():
    with pytest.raises(ImitationPreflightError, match="unknown generation provider"):
        preflight_imitation_providers(
            ImitationConfig(enabled=True, generation={"provider": "not_a_real_provider"})
        )


def test_unknown_sampling_provider_fails():
    with pytest.raises(ImitationPreflightError, match="unknown sampling provider"):
        preflight_imitation_providers(
            ImitationConfig(enabled=True, sampling={"provider": "not_a_real_sampler"})
        )


def test_python_provider_finds_matching_sibling_file(tmp_path):
    config_path = tmp_path.joinpath("experiment.yaml")
    config_path.write_text(
        "imitation:\n  generation:\n    provider: custom_generator\n", encoding="utf-8"
    )
    tmp_path.joinpath("imitation.py").write_text(
        "def custom_generator(context):\n    return {}\n", encoding="utf-8"
    )

    preflight_imitation_providers(
        ImitationConfig(enabled=True, generation={"provider": "custom_generator"}),
        config_path,
    )


def test_python_provider_using_reserved_built_in_name_fails(tmp_path):
    config_path = tmp_path.joinpath("experiment.yaml")
    config_path.write_text(
        "imitation:\n  generation:\n    provider: astar\n", encoding="utf-8"
    )
    tmp_path.joinpath("imitation.py").write_text(
        "def astar(context):\n    return {}\n", encoding="utf-8"
    )

    with pytest.raises(ImitationPreflightError, match="reserved built-in name"):
        preflight_imitation_providers(
            ImitationConfig(enabled=True, generation={"provider": "astar"}),
            config_path,
        )
