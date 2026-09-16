"""Validate heterogeneous movement, capture, and escape rewards."""

from copy import deepcopy
from pathlib import Path

from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv
from theseo_anysearch.settings import load_experiment


ROOT = Path("usage", "experiments", "showcase", "hunter_and_hunted")


def runtime() -> dict:
    """Load the showcase and attach its compiled native extension."""
    config = load_experiment(Path(ROOT, "experiment.yaml"))
    result = config.env.to_runtime_dict()
    result["native_extension_manifest"] = str(Path(ROOT, ".anysearch", "extension.json").resolve())
    return result


def main() -> None:
    """Exercise the three heterogeneous-agent contracts."""
    config = runtime()
    env = MultiVoxelEnv(config)
    observations, _ = env.reset(seed=42)
    assert list(observations) == ["hunted", "hunter"]
    assert env._rust_env.cursor_positions() == [(16, 16, 16), (4, 4, 4)]
    env.step({"hunted": 16, "hunter": 16})
    assert env._rust_env.cursor_positions() == [(16, 17, 17), (4, 6, 6)]

    capture = deepcopy(config)
    capture["agents"][0]["start"] = [4, 1, 1]
    capture["agents"][1]["start"] = [1, 1, 1]
    capture_env = MultiVoxelEnv(capture)
    capture_env.reset(seed=42)
    capture_env.step({"hunted": 21, "hunter": 21})
    _, rewards, terminations, _, _ = capture_env.step(
        {"hunted": 21, "hunter": 21}
    )
    assert all(terminations.values())
    assert rewards["hunter"] == 1.0

    escape = deepcopy(config)
    escape["max_steps"] = 1
    escape["agents"][0]["start"] = [20, 20, 20]
    escape["agents"][1]["start"] = [1, 1, 1]
    escape_env = MultiVoxelEnv(escape)
    escape_env.reset(seed=42)
    _, rewards, terminations, _, _ = escape_env.step(
        {"hunted": 0, "hunter": 0}
    )
    assert all(terminations.values())
    assert rewards["hunted"] == 1.0
    print("Heterogeneous agents validated: ordered movement, capture, and timeout rewards work.")


if __name__ == "__main__":
    main()
