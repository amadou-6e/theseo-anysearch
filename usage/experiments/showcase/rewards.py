"""Penalize diagonal movement in ``quick_demo.yaml``."""

from theseo_anysearch.experiments.custom_rewards import RewardResult


def quick_demo(context):
    changed_axes = sum(
        previous != current
        for previous, current in zip(context.previous_cursor, context.cursor)
    )
    diagonal_penalty = -0.005 * max(changed_axes - 1, 0)
    return RewardResult(
        reward=diagonal_penalty,
        components={"diagonal_move_penalty": diagonal_penalty},
        mode="add",
    )
