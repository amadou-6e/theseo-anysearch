from pathlib import Path

import pytest
from pydantic import ValidationError

from theseo_anysearch.experiments.custom_rewards import (
    CustomRewardError,
    RewardContext,
    RewardResult,
    apply_custom_reward,
    copy_reward_source,
    discover_reward_source,
    load_reward_provider,
    write_reward_manifest,
)


def _context(**updates):
    values = {
        "step": 1, "action": 0, "action_index": 0,
        "previous_observation": {}, "observation": {},
        "previous_cursor": (1, 1, 1), "cursor": (2, 1, 1), "goal": (3, 1, 1),
        "previous_goal_distance": 2.0, "goal_distance": 1.0,
        "invalid_action": False, "collision": False,
        "terminated": False, "truncated": False,
        "standard_reward": -0.01, "standard_breakdown": {"step_cost": -0.01},
        "env_config": {},
    }
    values.update(updates)
    return RewardContext(**values)


def _write_rewards(path: Path, name: str, result: str) -> None:
    path.write_text(
        "from theseo_anysearch.experiments.custom_rewards import RewardResult\n\n"
        f"def {name}(context):\n"
        f"    return {result}\n",
        encoding="utf-8",
    )


def test_yaml_name_selects_function_from_rewards_module(tmp_path: Path) -> None:
    config = tmp_path.joinpath("trial.yaml")
    config.write_text("experiment: {}\n", encoding="utf-8")
    shared = tmp_path.joinpath("rewards.py")
    _write_rewards(shared, "movement_cost", "RewardResult(reward=-0.2)")

    assert discover_reward_source(config, "movement_cost") == shared
    archived = copy_reward_source(config, tmp_path.joinpath("run"), "movement_cost")
    provider = load_reward_provider(archived, "movement_cost")
    reward, breakdown = apply_custom_reward(provider, _context())

    assert reward == pytest.approx(-0.21)
    assert breakdown == {"step_cost": -0.01, "movement_cost": -0.2}
    manifest = write_reward_manifest(provider, tmp_path.joinpath("run"))
    assert manifest is not None
    assert '"name": "movement_cost"' in manifest.read_text(encoding="utf-8")


def test_missing_selected_function_is_rejected(tmp_path: Path) -> None:
    source = tmp_path.joinpath("rewards.py")
    _write_rewards(source, "another_reward", "RewardResult(reward=-0.2)")
    with pytest.raises(CustomRewardError, match="movement_cost"):
        load_reward_provider(source, "movement_cost")


def test_replace_mode_discards_standard_breakdown(tmp_path: Path) -> None:
    source = tmp_path.joinpath("rewards.py")
    _write_rewards(
        source, "replacement_reward",
        "RewardResult(reward=-0.5, components={'replacement': -0.5}, mode='replace')",
    )
    reward, breakdown = apply_custom_reward(
        load_reward_provider(source, "replacement_reward"), _context()
    )
    assert reward == pytest.approx(-0.5)
    assert breakdown == {"replacement": -0.5}


def test_component_collision_is_rejected(tmp_path: Path) -> None:
    source = tmp_path.joinpath("rewards.py")
    _write_rewards(
        source, "bad_reward",
        "RewardResult(reward=-0.1, components={'step_cost': -0.1})",
    )
    with pytest.raises(CustomRewardError, match="collide"):
        apply_custom_reward(load_reward_provider(source, "bad_reward"), _context())


def test_components_must_sum_to_reward() -> None:
    with pytest.raises(ValidationError, match="must sum"):
        RewardResult(reward=-0.2, components={"penalty": -0.1})
