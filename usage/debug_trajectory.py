"""Generate a debug trajectory JSON for use with `anysearch replay file`.

Uses random actions — no trained policy needed.

Usage:
    python usage/debug_trajectory.py
    anysearch replay file debug_trajectory.json
"""

import json
import random
import sys
from pathlib import Path

# Run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from theseo_anysearch.experiments.trajectory import (
    collect_multi_eval_episode,
    _build_multi_payload,
)

ENV_CONFIG = {
    "agent_count": 3,
    "max_steps": 200,
    "trail_mode": True,
    "geometry_boxes": [[18, 18, 18, 28, 28, 28]],
    "grid_size": 32,
    "step_cost": -0.01,
    "goal_reward": 1.0,
    "distance_shaping": 0.1,
    "collision_cost": 0.0,
    "obs_mode": "scalar",
    "seed": 42,
}

OUT = Path("debug_trajectory.json")


class _RandomPolicy:
    def compute_single_action(self, obs, **kwargs):
        return random.randint(0, 25)


if __name__ == "__main__":
    print("Building environment and running one random episode …")
    episode = collect_multi_eval_episode(_RandomPolicy(), ENV_CONFIG, seed=7)
    payload = _build_multi_payload(episode, iteration=0, episode_reward_mean=0.0,
                                   experiment_name="debug", run_id="debug")
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Saved {OUT}")
    print(f"  agents:     {episode.agent_count}")
    print(f"  steps:      {len(episode.steps)}")
    print(f"  geometry:   {len(episode.init_filled)} voxels")
    print(f"  starts:     {episode.start_positions}")
    print(f"  goals:      {episode.goal_positions}")
    print()
    print(f"Open with:  anysearch replay file {OUT}")
