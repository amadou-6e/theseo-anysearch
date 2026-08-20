"""Tests for sibling-`imitation.py` generation-provider discovery."""

import json

import pytest

from theseo_anysearch.experiments.custom_imitation import (
    CustomGenerationError,
    available_python_generation_names,
    discover_generation_source,
    load_generation_provider,
)
from theseo_anysearch.imitation.generation_providers import EpisodeGenerationContext


IMITATION_SOURCE = '''
def straight_line_generator(context):
    return {
        "observations": [context.observation],
        "actions": [0],
        "success": True,
        "seed": context.seed,
    }
'''


def test_discover_generation_source_finds_sibling_file(tmp_path):
    config_path = tmp_path.joinpath("experiment.yaml")
    config_path.write_text("imitation:\n  generation:\n    provider: straight_line_generator\n")
    tmp_path.joinpath("imitation.py").write_text(IMITATION_SOURCE, encoding="utf-8")

    source = discover_generation_source(config_path, "straight_line_generator")

    assert source == tmp_path.joinpath("imitation.py")


def test_discover_generation_source_returns_none_without_sibling_file(tmp_path):
    config_path = tmp_path.joinpath("experiment.yaml")
    config_path.write_text("imitation:\n  generation:\n    provider: straight_line_generator\n")

    assert discover_generation_source(config_path, "straight_line_generator") is None


def test_available_python_generation_names_probes_exports(tmp_path):
    source = tmp_path.joinpath("imitation.py")
    source.write_text(IMITATION_SOURCE, encoding="utf-8")

    assert available_python_generation_names(
        source, ("straight_line_generator", "not_defined")
    ) == ("straight_line_generator",)


def test_load_generation_provider_runs_the_python_function(tmp_path):
    source = tmp_path.joinpath("imitation.py")
    source.write_text(IMITATION_SOURCE, encoding="utf-8")
    record = load_generation_provider(source, "straight_line_generator")

    episode = record.generate(
        EpisodeGenerationContext(env=object(), observation="obs", seed=7, attempt=0)
    )

    assert episode.success is True
    assert episode.seed == 7


def test_load_generation_provider_rejects_wrong_arity(tmp_path):
    source = tmp_path.joinpath("imitation.py")
    source.write_text("def bad(context, extra):\n    return {}\n", encoding="utf-8")

    with pytest.raises(CustomGenerationError, match="exactly one argument"):
        load_generation_provider(source, "bad")
