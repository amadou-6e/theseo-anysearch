"""Replay one accepted sample from `anysearch geometry sample`'s output.

Re-resolves the procedural geometry for one specific seed (matching what
`geometry sample` already reported as accepted), then walks it with the A*
heuristic oracle so the replayer shows the actual solved route rather than
random actions:

    python usage/experiments/showcase/geometry_capabilities/02_python_provider/replay_accepted_sample.py --seed 42
    anysearch replay file runtime/geometry_capabilities/02_python_provider/replay/trajectories/heuristic_astar.json
"""

import argparse
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from theseo_anysearch.experiments.output import OutputStore
    from theseo_anysearch.experiments.trajectory import (
        collect_heuristic_episode,
        write_heuristic_trajectory,
    )

    env_config = {
        "grid_size": 32,
        "max_steps": 256,
        "trail_mode": False,
        "action_mode": "discrete_6",
        "waypoints_file": str(SHOWCASE_DIR.joinpath("waypoints.json")),
        "geometry_provider": "procedural_walls",
        "geometry_module_path": str(SHOWCASE_DIR.joinpath("geometry.py")),
        "geometry_provider_parameters": {"wall_count": 2},
        "geometry_validation": {
            "enabled": True,
            "maximum_attempts": 1,
            "maximum_search_nodes": 100_000,
            "recovery_margin_steps": 8,
        },
        "seed": args.seed,
    }

    episode = collect_heuristic_episode(env_config, "astar", seed=args.seed)
    print(f"seed={args.seed} success={episode.success} steps={len(episode.steps)}")
    print(f"difficulty_band={episode.difficulty_band} routing_difficulty={episode.routing_difficulty}")

    store = OutputStore(
        SHOWCASE_DIR.parents[4].joinpath(
            "runtime", "geometry_capabilities", "02_python_provider", "replay"
        )
    )
    json_path = write_heuristic_trajectory(
        store, episode, heuristic_type="astar", weight=None,
        iteration=0, experiment_name="geometry-capabilities-python-provider",
        run_id=f"seed-{args.seed}",
    )
    full_path = store.root.joinpath(json_path)
    print(f"wrote {full_path}")
    print(f"Open with:  anysearch replay file {full_path}")


if __name__ == "__main__":
    main()
