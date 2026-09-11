"""Tests for schema-driven interactive observation editing."""

from pathlib import Path

import numpy as np
from gymnasium import spaces

from theseo_anysearch.rllib.explain.scenarios import load_scenario
from theseo_anysearch.rllib.explain.ui.editor import ObservationEditor


def observation_space() -> spaces.Dict:
    """Return a compact box-observation space."""

    return spaces.Dict(
        {
            "local_grid": spaces.Box(0.0, 1.0, shape=(27,), dtype=np.float32),
            "goal_distance": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
    )


def test_editor_slices_follow_x_y_z_flattening_order() -> None:
    """Slice edits map exactly onto the Rust dx/dy/dz loop order."""

    editor = ObservationEditor(
        observation_space(),
        {
            "local_grid": np.zeros(27, dtype=np.float32),
            "goal_distance": np.asarray([0.5], dtype=np.float32),
        },
    )
    edited = np.zeros((3, 3), dtype=np.float32)
    edited[1, 2] = 1.0
    editor.set_slice("x", 0, edited)

    assert editor.values["local_grid"].reshape(3, 3, 3)[0, 1, 2] == 1.0
    assert editor.slice("y", 1)[0, 2] == 1.0
    assert editor.slice("z", 2)[0, 1] == 1.0


def test_editor_scenario_round_trip(tmp_path: Path) -> None:
    """An edited observation is preserved by strict scenario YAML."""

    editor = ObservationEditor(
        observation_space(),
        {
            "local_grid": np.zeros(27, dtype=np.float32),
            "goal_distance": np.asarray([0.5], dtype=np.float32),
        },
    )
    editor.set_field("goal_distance", [0.25])
    path = tmp_path.joinpath("scenario.yaml")
    editor.save_scenario(path)

    scenario = load_scenario(path)
    assert scenario.observation["goal_distance"] == [0.25]
