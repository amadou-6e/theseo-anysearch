"""Focused tests for Tune preflight diagnostics."""

from __future__ import annotations

from typing import Any

import pytest

from theseo_anysearch.cli.commands.tune import _validate_tune_config


@pytest.mark.parametrize(
    ("base_config", "search_space"),
    [
        ({"use_position_encoding": True}, {}),
        ({}, {"use_position_encoding": {"choice": [True, False]}}),
    ],
)
def test_inactive_position_encoding_has_precise_warning(
    base_config: dict[str, Any], search_space: dict[str, Any]
) -> None:
    issues = _validate_tune_config(base_config, search_space, num_gpus=0)

    assert len(issues) == 1
    assert "Inactive param 'use_position_encoding'" in issues[0]
    assert "will not change trial behavior" in issues[0]
    assert "Unrecognised" not in issues[0]


def test_supported_tune_parameters_remain_warning_free() -> None:
    issues = _validate_tune_config(
        {"lr": 0.001, "encoder_depth": 3},
        {"clip_param": {"uniform": [0.1, 0.3]}, "layer_size": 128},
        num_gpus=0,
    )

    assert issues == []
