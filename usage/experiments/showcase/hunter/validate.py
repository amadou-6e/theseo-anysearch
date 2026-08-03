"""Deterministically validate the compiled hunter predicate and outcome."""

import json
from pathlib import Path

import theseo_core


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT.joinpath(".anysearch", "extension.json")
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
library = MANIFEST.parent.joinpath(manifest["library"]).resolve()

predicates = json.dumps(
    [
        {"name": "valid_action"},
        {"name": "double_step_in_bounds", "parameters": {"multiplier": 2}},
    ]
)
outcomes = json.dumps(
    [{"name": "double_step", "parameters": {"multiplier": 2}}]
)
env = theseo_core.PyVoxelEnv(
    max_steps=10,
    trail_mode=False,
    grid_size=32,
    step_cost=0.0,
    goal_reward=0.0,
    distance_shaping=0.0,
    collision_cost=0.0,
    terminate_on_success=False,
    native_action_path=str(library),
    action_predicates_json=predicates,
    action_outcomes_json=outcomes,
)

env.set_waypoints((4, 4, 4), (20, 20, 20))
env.reset(42)
env.step(21)  # (+1, 0, 0)
assert env.cursor_pos() == (6, 4, 4)

env.set_waypoints((31, 4, 4), (20, 20, 20))
env.reset(42)
assert env.action_mask()[21] == 0
env.step(21)
assert env.cursor_pos() == (31, 4, 4)

print("Hunter extension validated: doubled movement and doubled-boundary predicate work.")