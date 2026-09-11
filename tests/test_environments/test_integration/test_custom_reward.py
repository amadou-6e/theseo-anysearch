from __future__ import annotations

import pytest

from tests.test_environments.test_integration._voxel_validity_support import (
    ACTION_PLUS_Z,
    make_radial_test_env,
)


@pytest.mark.integration
def test_custom_reward_is_applied_and_reported(tmp_path):
    reward_module = tmp_path.joinpath("rewards.py")
    reward_module.write_text(
        "from theseo_anysearch.experiments.custom_rewards import RewardResult\n\n"
        "def movement_penalty(context):\n"
        "    configured = float(context.parameters['movement_penalty'])\n"
        "    penalty = configured if context.cursor != context.previous_cursor else 0.0\n"
        "    return RewardResult(\n"
        "        reward=penalty,\n"
        "        components={'movement_penalty': penalty},\n"
        "    )\n",
        encoding="utf-8",
    )
    env = make_radial_test_env(
        tmp_path,
        reward_overrides={
            "reward_module_path": str(reward_module),
            "custom_reward": "movement_penalty",
            "custom_reward_parameters": {"movement_penalty": -0.2},
        },
    )
    env.reset(seed=0)

    _, reward, _, _, info = env.step(ACTION_PLUS_Z)

    assert info["reward_breakdown"]["movement_penalty"] == pytest.approx(-0.2)
    assert reward == pytest.approx(sum(info["reward_breakdown"].values()))
