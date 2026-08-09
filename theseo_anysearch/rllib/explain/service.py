"""End-to-end orchestration behind the ``anysearch explain`` command."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from theseo_anysearch.cli.registry import resolve_ref
from theseo_anysearch.environments.action_spaces import (
    ACTION_OFFSETS_26,
    offsets_for_mode,
)
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.experiments.loader import load_experiment
from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.rllib.explain.backend import PolicyExplanationBackend
from theseo_anysearch.rllib.explain.explainers import OcclusionExplainer
from theseo_anysearch.rllib.explain.features import FeatureSchema
from theseo_anysearch.rllib.explain.models import (
    ExplanationReport,
    ExplanationRequest,
)
from theseo_anysearch.rllib.explain.reports import ExplanationReportWriter
from theseo_anysearch.rllib.explain.scenarios import (
    EnvironmentScenario,
    ObservationScenario,
    load_scenario,
    validate_observation,
)
from theseo_anysearch.rllib.explain.scoring import DQNPolicyScorer, PolicyScorer
from theseo_anysearch.rllib.explain.traces import (
    ObservationTrace,
    ObservationTraceStep,
    PolicyEvaluationTraceCollector,
)


def resolve_run_dir(run_ref: str) -> Path:
    """Resolve a direct run directory or registered ``name:run-id`` reference."""

    direct = Path(run_ref)
    if direct.joinpath("experiment.yaml").is_file():
        return direct.resolve()
    experiment_dir, identifier = resolve_ref(run_ref)
    if identifier is None:
        raise ValueError(
            f"run reference {run_ref!r} must identify a run, for example name:run-id"
        )
    matches = [
        candidate
        for candidate in experiment_dir.rglob(identifier)
        if candidate.is_dir() and candidate.joinpath("experiment.yaml").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"run reference {run_ref!r} resolved to {len(matches)} run directories"
        )
    return matches[0].resolve()


def resolve_trajectory(run_dir: Path, selector: str) -> Path:
    """Resolve a trajectory selector under a run directory."""

    candidate = Path(selector)
    if candidate.is_file():
        return candidate.resolve()
    directory = run_dir.joinpath("trajectories")
    if selector == "best":
        candidate = directory.joinpath("best.json")
    elif selector == "latest":
        choices = sorted(directory.glob("iter_*.json"))
        if not choices:
            raise ValueError(f"no iteration trajectories found under {directory}")
        candidate = choices[-1]
    else:
        candidate = directory.joinpath(
            selector if selector.endswith(".json") else f"{selector}.json"
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"trajectory not found: {candidate}")
    return candidate.resolve()


def resolve_occlusion_background(
    background: str, trace: ObservationTrace
) -> list[dict[str, np.ndarray]]:
    """Return the background observation set for one occlusion request.

    Raises when ``background`` requests a trace-derived baseline but the trace
    is too short to avoid a degenerate baseline that equals the observation
    being explained (every group attribution would silently be 0.0).
    """

    if background == "zeros":
        return [
            {
                name: np.zeros_like(value, dtype=np.float32)
                for name, value in trace.step(0).observation.items()
            }
        ]
    if background not in {"trace", "mean"}:
        raise ValueError(f"unsupported occlusion background: {background!r}")
    background_observations = trace.observations()
    if len(background_observations) <= 1:
        raise ValueError(
            "occlusion requires a background of at least two observations to "
            "avoid a degenerate baseline that equals the observation being "
            f"explained (got {len(background_observations)} from a "
            f"{'single_step' if len(trace) == 1 else 'short'} trace); "
            "use a multi-step rollout trajectory/scenario, or pass "
            "background='zeros'"
        )
    return background_observations


class PolicyExplanationService:
    """Restore a run and explain one trace or scenario."""

    def __init__(
        self,
        run_dir: Path,
        checkpoint: str = "latest",
        *,
        scorer: PolicyScorer | None = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.experiment_path = self.run_dir.joinpath("experiment.yaml")
        experiment = load_experiment(self.experiment_path)
        if not isinstance(experiment, ExperimentConfig):
            raise ValueError("explanations require a single resolved experiment")
        if experiment.training.algorithm != "dqn":
            raise ValueError(
                f"algorithm {experiment.training.algorithm!r} is not supported; expected 'dqn'"
            )
        self.experiment = experiment
        self.checkpoint = checkpoint
        self.scorer = scorer or DQNPolicyScorer.from_run_dir(
            self.run_dir, checkpoint=checkpoint
        )

    def explain_trace(
        self,
        selector: str,
        *,
        focus: str = "collisions",
        max_steps: int = 50,
        explicit_steps: tuple[int, ...] = (),
        background: str = "trace",
        output_dir: Path | None = None,
        seed: int | None = None,
    ) -> ExplanationReport:
        """Replay and explain a saved AnySearch trajectory."""

        path = resolve_trajectory(self.run_dir, selector)
        trace = self._replay_trajectory(path, seed)
        return self._explain(
            trace,
            trajectory=str(path),
            focus=focus,
            max_steps=max_steps,
            explicit_steps=explicit_steps,
            background=background,
            output_dir=output_dir,
            validity="environment_validated",
        )

    def explain_scenario(
        self,
        path: Path,
        *,
        focus: str = "all",
        max_steps: int = 50,
        explicit_steps: tuple[int, ...] = (),
        background: str = "trace",
        output_dir: Path | None = None,
    ) -> ExplanationReport:
        """Explain a strict environment or fictional-observation scenario."""

        scenario = load_scenario(path)
        if isinstance(scenario, ObservationScenario):
            env = self._build_env()
            try:
                observation = validate_observation(
                    scenario.observation, env.observation_space
                )
                action = (
                    self.scorer.select_action(observation)
                    if scenario.chosen_action == "policy"
                    else int(scenario.chosen_action)
                )
                if not env.action_space.contains(action):
                    raise ValueError(
                        f"chosen_action {action} is outside the policy action space"
                    )
            finally:
                env.close()
            cursor = self._cursor(observation)
            trace = ObservationTrace(
                [
                    ObservationTraceStep(
                        step=0,
                        observation=observation,
                        action=action,
                        reward=0.0,
                        cursor_before=cursor,
                        cursor_after=cursor,
                        done=False,
                        collision=None,
                    )
                ],
                algorithm="dqn",
            )
            validity = "not_environment_validated"
        else:
            trace = self._environment_scenario_trace(scenario, output_dir)
            validity = "environment_validated"
        return self._explain(
            trace,
            trajectory=str(path.resolve()),
            focus=focus,
            max_steps=max_steps,
            explicit_steps=explicit_steps,
            background=background,
            output_dir=output_dir,
            validity=validity,
        )

    def _explain(
        self,
        trace: ObservationTrace,
        *,
        trajectory: str,
        focus: str,
        max_steps: int,
        explicit_steps: tuple[int, ...],
        background: str,
        output_dir: Path | None,
        validity: str,
    ) -> ExplanationReport:
        """Run grouped occlusion and write reproducibility artifacts."""

        destination = output_dir or self.run_dir.joinpath(
            "explanations", uuid.uuid4().hex[:8]
        )
        request = ExplanationRequest(
            run_ref=str(self.run_dir),
            checkpoint=self.checkpoint,
            trajectory=trajectory,
            method="occlusion",
            focus=focus,
            max_steps=max_steps,
            explicit_steps=explicit_steps,
            background=background,
            scenario_validity=validity,
            output_dir=destination,
        )
        schema = FeatureSchema.from_observation(trace.step(0).observation)
        background_observations = resolve_occlusion_background(background, trace)
        backend = PolicyExplanationBackend(
            schema,
            trace,
            self.scorer,
            explainer=OcclusionExplainer(schema, self.scorer, background_observations),
            report_writer=ExplanationReportWriter(destination),
        )
        report = backend.explain(request)
        destination.joinpath("request.yaml").write_text(
            yaml.safe_dump(
                {
                    "run": str(self.run_dir),
                    "checkpoint": self.checkpoint,
                    "trajectory": trajectory,
                    "method": "occlusion",
                    "focus": focus,
                    "max_steps": max_steps,
                    "explicit_steps": list(explicit_steps),
                    "background": background,
                    "scenario_validity": validity,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return report

    def _build_env(self, overrides: Mapping[str, Any] | None = None) -> VoxelEnv:
        """Build an environment from authoritative run settings and state overrides."""

        config = self.experiment.env.to_runtime_dict()
        config.update(dict(overrides or {}))
        return VoxelEnv(config)

    def _environment_scenario_trace(
        self,
        scenario: EnvironmentScenario,
        output_dir: Path | None,
    ) -> ObservationTrace:
        """Construct and execute one controlled environment scenario."""

        if scenario.state.trail:
            raise ValueError(
                "pre-populated trail state is not yet representable by the Rust environment API"
            )
        boxes = [list(box) for box in scenario.state.geometry_boxes]
        directory = output_dir or self.run_dir.joinpath("explanations", "scenario_state")
        directory.mkdir(parents=True, exist_ok=True)
        waypoints = directory.joinpath("scenario_waypoints.json")
        waypoints.write_text(
            json.dumps(
                {
                    "start": list(scenario.state.cursor),
                    "goal": list(scenario.state.route[0]),
                }
            ),
            encoding="utf-8",
        )
        env = self._build_env(
            {"geometry_boxes": boxes, "waypoints_file": str(waypoints)}
        )
        try:
            if scenario.execution.mode == "rollout":
                trace = PolicyEvaluationTraceCollector(env, self.scorer, "dqn").collect(
                    seed=scenario.seed, max_steps=scenario.execution.max_steps
                )
            elif scenario.execution.mode == "single_step":
                trace = PolicyEvaluationTraceCollector(env, self.scorer, "dqn").collect(
                    seed=scenario.seed, max_steps=1
                )
            else:
                trace = self._collect_actions(env, scenario.seed, scenario.execution.actions)
        finally:
            env.close()
        return trace

    def _replay_trajectory(self, path: Path, seed: int | None) -> ObservationTrace:
        """Rebuild pre-action observations and fail at the first replay mismatch."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        start = tuple(payload.get("start_pos") or ())
        goal = tuple(payload.get("goal_pos") or ())
        if len(start) != 3 or len(goal) != 3:
            raise ValueError(f"trajectory {path} lacks start_pos or goal_pos")
        boxes = [list(coord) + list(coord) for coord in payload.get("init_filled", [])]
        state_dir = self.run_dir.joinpath("explanations", ".replay")
        state_dir.mkdir(parents=True, exist_ok=True)
        waypoint_path = state_dir.joinpath("waypoints.json")
        waypoint_path.write_text(
            json.dumps({"start": list(start), "goal": list(goal)}), encoding="utf-8"
        )
        env = self._build_env(
            {"geometry_boxes": boxes, "waypoints_file": str(waypoint_path)}
        )
        try:
            observation, _ = env.reset(
                seed=seed if seed is not None else self.experiment.env.seed
            )
            steps: list[ObservationTraceStep] = []
            expected_cursor = start
            actual_start = tuple(int(value) for value in env._rust_env.cursor_pos())
            if actual_start != expected_cursor:
                raise ValueError(
                    f"trajectory replay diverged at reset: expected {expected_cursor}, got {actual_start}"
                )
            for index, recorded in enumerate(payload.get("steps", [])):
                canonical_action = int(recorded["action"])
                policy_action = self._policy_action(canonical_action)
                before = self._cursor(observation)
                next_observation, reward, terminated, truncated, info = env.step(policy_action)
                actual = tuple(int(value) for value in env._rust_env.cursor_pos())
                expected = (
                    int(recorded["cursor_x"]),
                    int(recorded["cursor_y"]),
                    int(recorded["cursor_z"]),
                )
                if actual != expected:
                    raise ValueError(
                        f"trajectory replay diverged at step {index}: expected cursor "
                        f"{expected}, got {actual}"
                    )
                steps.append(
                    ObservationTraceStep(
                        step=index,
                        observation=self._copy_observation(observation),
                        action=policy_action,
                        reward=float(reward),
                        cursor_before=before,
                        cursor_after=self._cursor(next_observation),
                        done=bool(terminated or truncated),
                        collision=bool(info.get("collision", False)),
                        info=info,
                    )
                )
                observation = next_observation
        finally:
            env.close()
        if not steps:
            raise ValueError(f"trajectory {path} contains no steps")
        return ObservationTrace(steps, algorithm="dqn")

    def _collect_actions(
        self, env: VoxelEnv, seed: int, actions: tuple[int, ...]
    ) -> ObservationTrace:
        """Collect a trace from an explicit policy-action sequence."""

        observation, _ = env.reset(seed=seed)
        steps: list[ObservationTraceStep] = []
        for index, action in enumerate(actions):
            if not env.action_space.contains(action):
                raise ValueError(f"execution.actions[{index}]={action} is outside action space")
            before = self._cursor(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            steps.append(
                ObservationTraceStep(
                    step=index,
                    observation=self._copy_observation(observation),
                    action=action,
                    reward=float(reward),
                    cursor_before=before,
                    cursor_after=self._cursor(next_observation),
                    done=bool(terminated or truncated),
                    collision=bool(info.get("collision", False)),
                    info=info,
                )
            )
            observation = next_observation
            if terminated or truncated:
                break
        return ObservationTrace(steps, algorithm="dqn")

    def _policy_action(self, canonical_action: int) -> int:
        """Map a stored canonical action back into the run's policy action space."""

        direction = ACTION_OFFSETS_26[canonical_action]
        mode = self.experiment.env.action.mode
        if mode == "vector_3":
            raise ValueError("saved vector_3 trajectory replay is not yet supported")
        offsets = offsets_for_mode(mode)
        if direction not in offsets:
            raise ValueError(
                f"canonical action {canonical_action} is unavailable in action mode {mode}"
            )
        return offsets.index(direction)

    @staticmethod
    def _copy_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Detach a dictionary observation."""

        return {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in observation.items()
        }

    @staticmethod
    def _cursor(observation: Mapping[str, Any]) -> tuple[float, float, float]:
        """Read normalized cursor coordinates from an observation."""

        cursor = np.asarray(observation.get("cursor_pos", np.zeros(3)), dtype=np.float32)
        return tuple(float(value) for value in cursor)
