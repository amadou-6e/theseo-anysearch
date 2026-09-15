from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.test_environments.test_integration._voxel_validity_support import (
    ACTION_MINUS_X,
    make_radial_test_env,
)
from theseo_anysearch.experiments.native_extensions import (
    NativeExtension,
    compile_native_extension,
    copy_native_extension,
)


def test_compile_load_execute_in_core_and_archive_native_extension(tmp_path: Path) -> None:
    source = Path("usage", "experiments", "showcase", "native_extension", "extension")
    experiment = tmp_path.joinpath("experiment")
    extension = experiment.joinpath("extension")
    shutil.copytree(source, extension)
    experiment.joinpath("experiment.yaml").write_text(
        "experiment: {}\nenv:\n  rewards:\n    custom:\n"
        "      name: native_collision\n      parameters:\n"
        "        collision_penalty: -0.125\n",
        encoding="utf-8",
    )

    manifest_path = compile_native_extension(experiment)
    loaded = NativeExtension.load(manifest_path)
    assert loaded is not None
    assert loaded.compute_metrics("training", {}) == {"native_hook_active": 1.0}
    assert loaded.compute_metrics(
        "evaluation", {"standard_metrics": {"reward_mean": float("nan")}}
    ) == {"native_hook_active": 1.0}

    env = make_radial_test_env(
        tmp_path.joinpath("env"),
        start=(1, 1, 1),
        goal=(3, 1, 1),
        reward_overrides={
            "native_extension_manifest": str(manifest_path),
            "custom_reward": "native_collision",
            "custom_reward_parameters": {"collision_penalty": -0.125},
        },
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(ACTION_MINUS_X)
    assert info["reward_breakdown"]["native_collision"] == pytest.approx(-0.125)
    assert reward == pytest.approx(sum(info["reward_breakdown"].values()))

    second_env = make_radial_test_env(
        tmp_path.joinpath("second_env"),
        start=(1, 1, 1),
        goal=(3, 1, 1),
        reward_overrides={
            "native_extension_manifest": str(manifest_path),
            "custom_reward": "native_collision",
            "custom_reward_parameters": {"collision_penalty": -0.5},
        },
    )
    second_env.reset(seed=0)
    _, second_reward, _, _, second_info = second_env.step(ACTION_MINUS_X)
    assert second_info["reward_breakdown"]["native_collision"] == pytest.approx(-0.5)
    assert second_reward == pytest.approx(
        sum(second_info["reward_breakdown"].values())
    )
    assert info["reward_breakdown"]["native_collision"] == pytest.approx(-0.125)

    archived = copy_native_extension(
        experiment.joinpath("experiment.yaml"), tmp_path.joinpath("run")
    )
    assert archived is not None
    assert NativeExtension.load(archived) is not None
    assert json.loads(archived.read_text())["source_sha256"]


def test_compile_records_action_rule_metadata(tmp_path: Path) -> None:
    source = Path("usage", "experiments", "showcase", "native_extension", "extension")
    experiment = tmp_path.joinpath("action_metadata")
    shutil.copytree(source, experiment.joinpath("extension"))
    experiment.joinpath("experiment.yaml").write_text(
        "experiment: {}\nenv:\n  action:\n"
        "    predicates: [valid_action, bounds, unoccupied, avoid_repeated_collision]\n"
        "    outcomes: [cursor_movement, mark_destination]\n"
        "  rewards:\n    custom: native_collision\n",
        encoding="utf-8",
    )

    manifest = json.loads(
        compile_native_extension(experiment).read_text(encoding="utf-8")
    )
    metadata = {
        (item["kind"], item["name"]): item for item in manifest["rule_metadata"]
    }

    assert manifest["rule_metadata_schema_version"] == 1
    assert {
        ("reward", "native_collision"),
        ("predicate", "avoid_repeated_collision"),
        ("outcome", "mark_destination"),
        ("training_metrics", "training_metrics"),
        ("evaluation_metrics", "evaluation_metrics"),
    } <= metadata.keys()
    assert metadata[("predicate", "avoid_repeated_collision")][
        "environment_families"
    ] == ["voxel"]


def test_consecutive_collision_termination_is_owned_by_rust(tmp_path: Path) -> None:
    env = make_radial_test_env(
        tmp_path,
        start=(1, 1, 1),
        goal=(3, 1, 1),
        reward_overrides={"task": {"max_consecutive_collisions": 2}},
    )
    env.reset(seed=0)

    _, _, terminated, truncated, first_info = env.step(ACTION_MINUS_X)
    assert not terminated
    assert not truncated
    assert first_info["consecutive_collisions"] == 1

    _, reward, terminated, truncated, second_info = env.step(ACTION_MINUS_X)
    assert terminated
    assert not truncated
    assert second_info["termination_reason"] == "consecutive_collisions"
    assert second_info["consecutive_collisions"] == 2
    assert reward == pytest.approx(sum(second_info["reward_breakdown"].values()))
