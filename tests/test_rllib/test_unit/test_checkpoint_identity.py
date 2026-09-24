"""Checkpoint geometry/task compatibility tests without an RLlib dependency."""

import json

import pytest

from theseo_anysearch.rllib.trainer.checkpointing import (
    CheckpointManager,
    CheckpointState,
)


class _Algorithm:
    def save(self, path: str) -> str:
        return path

    def restore(self, path: str) -> None:
        self.restored = path


def test_checkpoint_round_trip_preserves_geometry_task_contract(tmp_path) -> None:
    contract = {"action_mode": "discrete_26", "waypoints": [[1, 1, 1], [2, 2, 2]]}
    manager = CheckpointManager(tmp_path, expected_geometry_task_contract=contract)
    checkpoint = manager.save(
        _Algorithm(), CheckpointState(iteration=3, rllib_path="unused")
    )

    restored = manager.restore(_Algorithm(), checkpoint)

    assert restored.geometry_task_contract == contract
    assert len(restored.geometry_task_fingerprint or "") == 64


def test_checkpoint_rejects_changed_geometry_task_contract(tmp_path) -> None:
    manager = CheckpointManager(
        tmp_path, expected_geometry_task_contract={"action_mode": "discrete_26"}
    )
    checkpoint = manager.save(
        _Algorithm(), CheckpointState(iteration=1, rllib_path="unused")
    )
    state_path = checkpoint.joinpath("state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["geometry_task_contract"]["action_mode"] = "discrete_6"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="geometry/task contract mismatch"):
        manager.restore(_Algorithm(), checkpoint)


def test_checkpoint_freezes_auxiliary_training_state(tmp_path) -> None:
    curriculum = tmp_path / "curriculum" / "state.json"
    curriculum.parent.mkdir()
    curriculum.write_text('{"stage": 2}')
    (tmp_path / "early_stop_state.json").write_text('{"consecutive": 1}')
    checkpoint = CheckpointManager(tmp_path).save(
        _Algorithm(), CheckpointState(iteration=3, rllib_path="unused"))
    curriculum.write_text('{"stage": 9}')
    assert (checkpoint / "anysearch_state/curriculum/state.json").read_text() == '{"stage": 2}'
    assert (checkpoint / "anysearch_state/early_stop_state.json").read_text() == '{"consecutive": 1}'
