"""Scenario providers for the dedicated extension showcase."""

import random

from theseo_anysearch.experiments.custom_scenarios import ScenarioContext, ScenarioResult


def adjacent_goal_python(context: ScenarioContext) -> ScenarioResult:
    """Place a goal in one selectable voxel adjacent to the grid center."""
    center_value = (context.grid_size + 1) // 2
    start = (center_value, center_value, center_value)
    if context.scope == "evaluation":
        seed_base = int(context.parameters["seed_base"])
        index = (context.seed - seed_base) % len(context.action_offsets)
    else:
        index = random.Random(context.seed).randrange(len(context.action_offsets))
    offset = context.action_offsets[index]
    goal = tuple(start[axis] + offset[axis] for axis in range(3))
    return ScenarioResult(
        start=start,
        goal=goal,
        scenario_id=f"adjacent-{index:02d}",
        metadata={"direction_index": index, "offset": offset},
    )
