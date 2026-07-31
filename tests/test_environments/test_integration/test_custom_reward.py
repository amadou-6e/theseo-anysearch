from __future__ import annotations

import pytest

from tests.test_environments.test_integration._voxel_validity_support import (
    ACTION_PLUS_Z,
    make_radial_test_env,
)


@pytest.mark.integration
def test_custom_reward_is_applied_and_reported(tmp_path):
    reward_module = tmp_path.joinpath("reward.py")
    reward_module.write_text(
        "from theseo_anysearch.experiments.custom_rewards import RewardResult\n\n"
        "def compute_reward(context):\n"
        "    penalty = -0.2 if context.cursor != context.previous_cursor else 0.0\n"
        "    return RewardResult(\n"
        "        reward=penalty,\n"
        "        components={'movement_penalty': penalty},\n"
        "    )\n",
        encoding="utf-8",
    )
    env = make_radial_test_env(
        tmp_path,
        reward_overrides={"reward_module_path": str(reward_module)},
    )
    env.reset(seed=0)

    _, reward, _, _, info = env.step(ACTION_PLUS_Z)

    assert info["reward_breakdown"]["movement_penalty"] == pytest.approx(-0.2)
    assert reward == pytest.approx(sum(info["reward_breakdown"].values()))
