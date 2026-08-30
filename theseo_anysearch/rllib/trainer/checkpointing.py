"""Checkpoint persistence for RLlib training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.worlds import world_contract_fingerprint


class CheckpointState(BaseModel):
    """State stored alongside an RLlib checkpoint.

    Parameters
    ----------
    iteration : int
        Last completed training iteration.
    episodes_total : int
        Lifetime number of completed episodes.
    rllib_path : str
        Exact path returned by RLlib when the checkpoint was saved.
    """

    model_config = ConfigDict(extra="forbid")

    iteration: int
    episodes_total: int = 0
    rllib_path: str
    world_contract: dict[str, Any] | None = None
    world_fingerprint: str | None = None


class CheckpointManager:
    """Persist and restore project and RLlib checkpoint state.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory of the training run.
    """

    def __init__(
        self,
        output_dir: Path,
        expected_world_contract: dict[str, Any] | None = None,
    ) -> None:
        self._root = Path(output_dir, "checkpoints")
        self._expected_world_contract = expected_world_contract
        self._expected_world_fingerprint = (
            world_contract_fingerprint(expected_world_contract)
            if expected_world_contract is not None
            else None
        )

    def save(self, algorithm: Any, state: CheckpointState) -> Path:
        """Save one RLlib and project checkpoint.

        Parameters
        ----------
        algorithm : Any
            RLlib algorithm or compatible object exposing save.
        state : CheckpointState
            Project state associated with the checkpoint.

        Returns
        -------
        pathlib.Path
            Created project checkpoint directory.
        """
        checkpoint_dir = Path(self._root, f"iter_{state.iteration:06d}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rllib_path = str(checkpoint_dir)
        if algorithm is not None:
            returned = algorithm.save(str(checkpoint_dir))
            if isinstance(returned, str) and returned:
                rllib_path = returned

        stored_state = state.model_copy(
            update={
                "rllib_path": rllib_path,
                "world_contract": self._expected_world_contract,
                "world_fingerprint": self._expected_world_fingerprint,
            }
        )
        self._write_json(
            Path(checkpoint_dir, "state.json"),
            stored_state.model_dump(),
        )
        self._write_json(
            Path(self._root, "latest.json"),
            {"path": str(checkpoint_dir), "iteration": state.iteration},
        )
        return checkpoint_dir

    def restore(self, algorithm: Any, checkpoint_dir: Path) -> CheckpointState:
        """Restore an RLlib algorithm and return project checkpoint state.

        Parameters
        ----------
        algorithm : Any
            RLlib algorithm or compatible object exposing restore.
        checkpoint_dir : pathlib.Path
            Project checkpoint directory.

        Returns
        -------
        CheckpointState
            Restored project state.
        """
        state_path = Path(checkpoint_dir, "state.json")
        if state_path.exists():
            state = CheckpointState.model_validate_json(
                state_path.read_text(encoding="utf-8")
            )
        else:
            state = CheckpointState(
                iteration=0,
                episodes_total=0,
                rllib_path=str(checkpoint_dir),
            )
        if (
            self._expected_world_contract is not None
            and (
                state.world_contract != self._expected_world_contract
                or state.world_fingerprint != self._expected_world_fingerprint
            )
        ):
            raise ValueError(
                "Checkpoint world contract mismatch: "
                f"expected {self._expected_world_contract}, "
                f"found {state.world_contract}"
            )
        algorithm.restore(state.rllib_path)
        return state

    def latest(self) -> Path | None:
        """Return the latest checkpoint directory when available.

        Returns
        -------
        pathlib.Path | None
            Latest checkpoint directory, or None when none is recorded.
        """
        latest_path = Path(self._root, "latest.json")
        if not latest_path.exists():
            return None
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        return Path(payload["path"])

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        """Write a JSON object with stable formatting.

        Parameters
        ----------
        path : pathlib.Path
            Destination file.
        payload : dict[str, Any]
            JSON-compatible object.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
