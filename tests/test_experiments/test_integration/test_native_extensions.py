from __future__ import annotations

import json
from pathlib import Path

from theseo_anysearch.experiments.custom_rewards import RewardContext
from theseo_anysearch.experiments.native_extensions import (
    NativeExtension,
    compile_native_extension,
    copy_native_extension,
)


def test_compile_load_execute_and_archive_native_extension(tmp_path: Path) -> None:
    source = Path("usage", "experiments", "showcase", "native_extension", "extension")
    experiment = tmp_path.joinpath("experiment")
    extension = experiment.joinpath("extension")
    import shutil
    shutil.copytree(source, extension)
    experiment.joinpath("experiment.yaml").write_text("experiment: {}\n", encoding="utf-8")

    manifest_path = compile_native_extension(experiment)
    loaded = NativeExtension.load(manifest_path)
    assert loaded is not None
    result = loaded.compute_reward(RewardContext(
        step=1, action=0, action_index=0, previous_observation={}, observation={},
        previous_cursor=(1, 1, 1), cursor=(1, 1, 1), goal=(2, 2, 2),
        previous_goal_distance=3.0, goal_distance=3.0, invalid_action=False,
        collision=True, terminated=False, truncated=False, standard_reward=-0.01,
        standard_breakdown={"step_cost": -0.01}, env_config={}, info={},
    ))
    assert result.reward == -0.02
    assert result.components == {"native_collision": -0.02}
    assert loaded.compute_metrics("training", {}) == {"native_hook_active": 1.0}

    archived = copy_native_extension(experiment.joinpath("experiment.yaml"), tmp_path.joinpath("run"))
    assert archived is not None
    assert NativeExtension.load(archived) is not None
    assert json.loads(archived.read_text())["source_sha256"]
