"""Walk the *valid* compiled world's solved route and write a replay file.

Run prepare_world.py first, then:

    python usage/experiments/showcase/geometry_capabilities/03_compiled_world/replay_valid_route.py
    anysearch replay file runtime/geometry_capabilities/03_compiled_world/replay/trajectories/heuristic_astar.json

Uses the *valid* pack specifically -- the *blocked* pack has no path for the
heuristic oracle to walk.
"""

import re
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    from theseo_anysearch.experiments.output import OutputStore
    from theseo_anysearch.experiments.trajectory import (
        collect_heuristic_episode,
        write_heuristic_trajectory,
    )

    yaml_text = SHOWCASE_DIR.joinpath("experiment_valid.yaml").read_text(encoding="utf-8")
    match = re.search(r"compiled_world_path: (\S+)", yaml_text)
    if match is None or match.group(1) == "PREPARE_WORLD_PY_FILLS_THIS_IN":
        raise SystemExit("Run prepare_world.py first to compile the packs.")
    compiled_world_path = str(SHOWCASE_DIR.joinpath(match.group(1)).resolve())

    env_config = {
        "extent": (128, 96, 64),
        "max_steps": 400,
        "trail_mode": False,
        "action_mode": "discrete_6",
        "waypoints_file": str(SHOWCASE_DIR.joinpath("waypoints.json")),
        "compiled_world_path": compiled_world_path,
        "seed": 42,
    }

    episode = collect_heuristic_episode(env_config, "astar", seed=42)
    print(f"success={episode.success} steps={len(episode.steps)}")

    store = OutputStore(
        SHOWCASE_DIR.parents[4].joinpath(
            "runtime", "geometry_capabilities", "03_compiled_world", "replay"
        )
    )
    json_path = write_heuristic_trajectory(
        store, episode, heuristic_type="astar", weight=None,
        iteration=0, experiment_name="geometry-capabilities-compiled-world",
        run_id="valid",
    )
    full_path = store.root.joinpath(json_path)
    print(f"wrote {full_path}")
    print(f"Open with:  anysearch replay file {full_path}")


if __name__ == "__main__":
    main()
